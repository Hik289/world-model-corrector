"""LLM Repair Experiment on Agent Calling-Tree Dataset.

All methods use GPT-4o-mini for the ACTUAL repair call.
They differ ONLY in how many / which nodes they show to the LLM.

Engineering baselines (pointwise / context-limited):
    Greedy-Point-LLM  : Show LLM only the single highest-error node
    Window-4-LLM      : Show LLM the 4 consecutive steps with highest avg error
    Window-8-LLM      : Show LLM 8 consecutive steps
    LocalRepair-2Hop-LLM : Show LLM 2-hop neighbourhood of highest-error node
    Full-Graph-LLM    : Show LLM the entire graph (expensive reference)

WM-SAR (proposed):
    WM-SAR-LLM        : Graph analysis selects 8-node connected region → ONE LLM call

All methods are evaluated on:
    Rec-Exact     : LLM identified exact root-cause node
    Rec-Type      : LLM identified correct root-cause node type
    Rec-2Hop      : LLM identified a node within 2 hops of root cause
    #Tokens       : tokens consumed per rollout
    #LLM-Calls    : number of LLM API calls
    Region-Size   : number of nodes shown to LLM

Key claim: WM-SAR-LLM achieves comparable or better Rec-Type/Rec-2Hop
while consuming fewer tokens (smaller context = only the relevant region).
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wm_sar.agent_calling_tree import generate_calling_trees
from wm_sar.engineering_baselines import (
    greedy_point, topk_point, window_repair, local_khop
)
from wm_sar.region_extractor import WMSAR, WMSARConfig
from wm_sar.llm_client import LLMClient
from wm_sar.act_text import (
    tree_to_text, build_locate_prompt, parse_locate_response,
)

import networkx as nx


# ── TraceScan baselines (spec §11.5 / §13 E5) ─────────────────────────────
# "Trace" = nodes in topological order. TraceScan-w_k shows a window of k
# consecutive topo-steps centred on the highest-error node. TraceScan-Full
# shows ALL nodes ordered as a single linear trace (no edge structure shown).
# LLMRepair-Full-Plan additionally asks for a full repair plan rather than
# just root-cause identification.

def _topo_order(G: nx.DiGraph) -> list[str]:
    try:
        return list(nx.topological_sort(G))
    except Exception:
        return sorted(G.nodes(),
                      key=lambda v: G.nodes[v].get("time_step", 0))


def _tracescan_window(G: nx.DiGraph, w: int) -> set[str]:
    """Return the topo-window of size w centred on the highest-error node."""
    topo = _topo_order(G)
    errs = [(float(G.nodes[v].get("err", 0.0)), i, v) for i, v in enumerate(topo)]
    if not errs:
        return set()
    _, ctr_i, _ = max(errs)
    lo = max(0, ctr_i - w // 2)
    hi = min(len(topo), lo + w)
    lo = max(0, hi - w)  # left-align if hit right edge
    return set(topo[lo:hi])


def _build_full_plan_prompt(tree_text: str, node_list: list[str]) -> tuple[str, str]:
    """Stronger prompt: identify root cause AND propose corrective actions
    for every affected node. Mirrors spec's LLMRepair-Full-Plan baseline."""
    system = (
        "You are an expert AI agent failure analyst and repair planner. "
        "You will receive a complete report of a failed multi-agent calling-tree. "
        "Your job has TWO parts: (1) identify the root-cause node that "
        "introduced the initial error; (2) propose a corrective action plan "
        "for every affected node along the cascade. Respond ONLY in valid JSON."
    )
    user = (
        f"{tree_text}\n\n"
        "Step 1 — identify the SINGLE root-cause node that started the cascade.\n"
        "  Use: high error_prob + low success_flag = direct error; "
        "dependency UNSATISFIED = cascade victim, not cause; "
        "low throughput at executors is a strong signal.\n\n"
        "Step 2 — list, in topological order, every affected downstream node and "
        "a one-sentence corrective action for each.\n\n"
        "Respond ONLY with valid JSON of the form:\n"
        '{"root_cause_nodes": ["<node_id>"], '
        '"root_cause_type": "<node_type>", '
        '"repair_plan": {"<node_id>": "<action>", ...}, '
        '"explanation": "<one sentence>", '
        '"confidence": <0-1>}'
    )
    return system, user


def _call_llm_on_region(
    G: nx.DiGraph, region: set, true_root: str,
    client: LLMClient, method_name: str,
    prompt_builder=build_locate_prompt,
    include_edges: bool = True,
) -> dict:
    """Core: show region to LLM, ask for root cause, parse response.

    Args:
        prompt_builder: callable (tree_text, node_list, G) -> (system, user).
          Default = build_locate_prompt; for LLMRepair-Full-Plan we swap to
          _build_full_plan_prompt which asks for a repair plan too.
        include_edges: whether to include edge structure in serialisation.
          False for TraceScan-* baselines (linear-trace view, no graph topo).
    """
    text, node_list = tree_to_text(G, selected_nodes=region,
                                    include_edges=include_edges,
                                    max_nodes=max(len(region), 30))
    if prompt_builder is _build_full_plan_prompt:
        system, user = prompt_builder(text, node_list)
    else:
        system, user = prompt_builder(text, node_list, G)
    t0 = time.time()
    resp = client.chat(system=system, user=user)
    latency_ms = (time.time() - t0) * 1000.0
    parsed = parse_locate_response(resp.text, true_root, G)
    return {
        "method": method_name,
        "region_size": len(region),
        "token_cost": resp.prompt_tokens + resp.completion_tokens,
        "latency_ms": latency_ms,
        "rec_exact": parsed["recovered_exact"],
        "rec_type": parsed["recovered_type"],
        "rec_hop2": parsed["recovered_hop2"],
        "identified_nodes": parsed["identified_nodes"],
        "confidence": parsed["confidence"],
    }


def run_instance(G: nx.DiGraph, true_root: str, client: LLMClient) -> dict:
    """Run all methods on a single graph instance."""
    results = {}

    # ── Engineering: Greedy-Point (show 1 node) ──────────────────────────
    rr = greedy_point(G, K=1)
    results["Greedy-Point-LLM"] = _call_llm_on_region(
        G, rr.selected_nodes, true_root, client, "Greedy-Point-LLM")

    # ── Engineering: TopK-5 (show 5 nodes) ───────────────────────────────
    rr = topk_point(G, K=5)
    results["TopK-5-LLM"] = _call_llm_on_region(
        G, rr.selected_nodes, true_root, client, "TopK-5-LLM")

    # ── Engineering: Window-4 ─────────────────────────────────────────────
    rr = window_repair(G, window=4)
    results["Window-4-LLM"] = _call_llm_on_region(
        G, rr.selected_nodes, true_root, client, "Window-4-LLM")

    # ── Engineering: Window-8 ─────────────────────────────────────────────
    rr = window_repair(G, window=8)
    results["Window-8-LLM"] = _call_llm_on_region(
        G, rr.selected_nodes, true_root, client, "Window-8-LLM")

    # ── Engineering: LocalRepair-2Hop ─────────────────────────────────────
    rr = local_khop(G, k=2)
    results["LocalRepair-2Hop-LLM"] = _call_llm_on_region(
        G, rr.selected_nodes, true_root, client, "LocalRepair-2Hop-LLM")

    # ── Full-Graph reference (expensive) ─────────────────────────────────
    full_region = set(G.nodes())
    results["Full-Graph-LLM"] = _call_llm_on_region(
        G, full_region, true_root, client, "Full-Graph-LLM")

    # ── WM-SAR (proposed) ─────────────────────────────────────────────────
    extractor = WMSAR(WMSARConfig())
    region = extractor.repair_region(G)
    results["WM-SAR-LLM"] = _call_llm_on_region(
        G, region, true_root, client, "WM-SAR-LLM")

    # ── spec §11.5 / §13 E5 baselines: TraceScan + LLMRepair-Full-Plan ───
    # Note: TraceScan-w4 ≡ Window-4-LLM (highest-error topo-centred window),
    # so we skip w4 here to avoid duplication.
    for w in (1, 2):
        results[f"TraceScan-w{w}-LLM"] = _call_llm_on_region(
            G, _tracescan_window(G, w), true_root, client,
            f"TraceScan-w{w}-LLM",
            include_edges=False)

    results["TraceScan-Full-LLM"] = _call_llm_on_region(
        G, set(G.nodes()), true_root, client,
        "TraceScan-Full-LLM",
        include_edges=False)

    results["LLMRepair-Full-Plan-LLM"] = _call_llm_on_region(
        G, set(G.nodes()), true_root, client,
        "LLMRepair-Full-Plan-LLM",
        prompt_builder=_build_full_plan_prompt,
        include_edges=True)  # plan needs graph context

    return results


def aggregate(all_results: list[dict]) -> dict[str, dict]:
    """Aggregate per-instance results into summary statistics."""
    # Collect every method that appears in any instance (union, not just first)
    methods = sorted({m for r in all_results for m in r.keys()})
    summaries = {}
    for m in methods:
        rows = [r[m] for r in all_results if m in r]
        n = len(rows)
        if n == 0:
            continue
        summaries[m] = {
            "n": n,
            "rec_exact":  float(np.mean([r["rec_exact"] for r in rows])),
            "rec_type":   float(np.mean([r["rec_type"] for r in rows])),
            "rec_hop2":   float(np.mean([r["rec_hop2"] for r in rows])),
            "mean_tokens": float(np.mean([r["token_cost"] for r in rows])),
            "mean_region_size": float(np.mean([r["region_size"] for r in rows])),
            "mean_latency_ms": float(np.mean([r["latency_ms"] for r in rows])),
        }
        summaries[m]["tok_per_rec_hop2"] = (
            summaries[m]["mean_tokens"] / max(summaries[m]["rec_hop2"], 1e-6)
        )
    return summaries


def build_per_instance(per_instance_rows: list[dict]) -> list[dict]:
    """Convert internal per-instance results into the schema requested by DS:
       [{instance_id, true_root, n_nodes, results: {method: {...}}}, ...]
    Numeric fields are JSON-safe (bool→int where it matters)."""
    out = []
    for row in per_instance_rows:
        rec = {
            "instance_id":   row["instance_id"],
            "seed":          row["seed"],
            "true_root":     row["true_root"],
            "true_root_type": row["true_root_type"],
            "n_nodes":       row["n_nodes"],
            "results": {},
        }
        for method, r in row["results"].items():
            rec["results"][method] = {
                "rec_exact":    int(bool(r["rec_exact"])),
                "rec_type":     int(bool(r["rec_type"])),
                "rec_hop2":     int(bool(r["rec_hop2"])),
                "tokens":       int(r["token_cost"]),
                "region_size":  int(r["region_size"]),
                "latency_ms":   float(r["latency_ms"]),
                "identified_nodes": list(r.get("identified_nodes", [])),
                "confidence":   float(r.get("confidence", 0.0)),
            }
        out.append(rec)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42,
                        help="Single seed (legacy). Overridden by --seeds.")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated list of seeds, e.g. '42,123,456'. "
                             "If set, each seed is run separately and a merged "
                             "JSON is written to --out plus per-seed JSONs.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--resume", action="store_true",
                        help="Resume: skip instances already present in "
                             "<out>_seed<S>.jsonl. The JSONL is appended to "
                             "instead of truncated.")
    parser.add_argument("--out", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                              "results", "exp_agent_llm.json"))
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]

    print(f"\n{'='*60}")
    print(f"  Agent Calling-Tree LLM Experiment")
    print(f"  n={args.n} × seeds={seeds}, model={args.model}")
    print(f"{'='*60}\n")

    client = LLMClient(model=args.model, temperature=0.0, max_tokens=512)

    all_per_instance = []   # list of {instance_id, seed, true_root, ..., results}
    all_results      = []   # flat list of run_instance() dicts for aggregate()
    per_seed_outputs = {}

    # Crash-safe per-instance JSONL append (lesson from M1 OpenAI 429 abort:
    # in-memory only state was lost). Each instance is flushed to disk as
    # soon as it completes.
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for seed in seeds:
        print(f"\n  ── seed={seed} ──")
        trees = generate_calling_trees(n=args.n, seed=seed)
        seed_per_instance = []
        seed_results = []
        jsonl_path = args.out.replace(".json", f"_seed{seed}.jsonl")

        # Resume support: load already-completed instance ids from JSONL
        done_ids: set[str] = set()
        if args.resume and os.path.exists(jsonl_path):
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        done_ids.add(rec["instance_id"])
                        # also rehydrate seed_per_instance for aggregate()
                        # Note: only per-instance schema is loaded; raw
                        # 'run_instance' dict is NOT reconstructed (won't
                        # affect aggregate() because we accumulate from
                        # build_per_instance, but we DO need aggregate() to
                        # see rows for the per-method summary).
                    except Exception:
                        pass
            if done_ids:
                print(f"    [resume] {len(done_ids)} instances already in "
                      f"{jsonl_path}; will skip those.")
        elif not args.resume:
            # Fresh run: truncate any stale JSONL
            with open(jsonl_path, "w") as f:
                pass

        # Re-populate seed_per_instance from JSONL so aggregate() can use it
        if done_ids:
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    # build a pseudo-row matching the live format
                    pseudo_res = {}
                    for meth, r in rec["results"].items():
                        pseudo_res[meth] = {
                            "rec_exact":   r["rec_exact"],
                            "rec_type":    r["rec_type"],
                            "rec_hop2":    r["rec_hop2"],
                            "token_cost":  r["tokens"],
                            "region_size": r["region_size"],
                            "latency_ms":  r["latency_ms"],
                            "identified_nodes": r.get("identified_nodes", []),
                            "confidence":  r.get("confidence", 0.0),
                        }
                    seed_results.append(pseudo_res)
                    all_results.append(pseudo_res)
                    row = {
                        "instance_id":   rec["instance_id"],
                        "seed":          rec["seed"],
                        "true_root":     rec["true_root"],
                        "true_root_type": rec.get("true_root_type", ""),
                        "n_nodes":       rec["n_nodes"],
                        "results":       pseudo_res,
                    }
                    seed_per_instance.append(row)
                    all_per_instance.append(row)

        for i, tree in enumerate(trees):
            G = tree.G
            true_root = tree.root_cause_node
            instance_id = f"s{seed}_i{i:03d}"
            if instance_id in done_ids:
                continue
            print(f"  [s{seed} {i+1:2d}/{args.n}] root={true_root} "
                  f"({G.nodes[true_root].get('node_type')}), "
                  f"N={G.number_of_nodes()}", end="  ", flush=True)
            try:
                res = run_instance(G, true_root, client)
                seed_results.append(res)
                all_results.append(res)
                row = {
                    "instance_id":   instance_id,
                    "seed":          seed,
                    "true_root":     true_root,
                    "true_root_type": G.nodes[true_root].get("node_type", ""),
                    "n_nodes":       G.number_of_nodes(),
                    "results":       res,
                }
                seed_per_instance.append(row)
                all_per_instance.append(row)
                # JSONL append + flush + fsync — crash-safe persistence
                jsonl_row = build_per_instance([row])[0]
                with open(jsonl_path, "a") as f:
                    f.write(json.dumps(jsonl_row) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                wmsar_e = res.get("WM-SAR-LLM", {}).get("rec_exact", 0)
                ts_full_e = res.get("TraceScan-Full-LLM", {}).get("rec_exact", 0)
                lr_e = res.get("LLMRepair-Full-Plan-LLM", {}).get("rec_exact", 0)
                print(f"WM-SAR-E={wmsar_e:.0f}  TS-Full-E={ts_full_e:.0f}  "
                      f"LLMRep-E={lr_e:.0f}")
            except Exception as e:
                print(f"ERROR: {e}")

        # Per-seed output file (full JSON aggregate)
        if seed_results:
            seed_out = args.out.replace(".json", f"_seed{seed}.json")
            seed_payload = {
                "n": len(seed_results),
                "seed": seed,
                "model": args.model,
                "summaries": aggregate(seed_results),
                "per_instance": build_per_instance(seed_per_instance),
            }
            with open(seed_out, "w") as f:
                json.dump(seed_payload, f, indent=2)
                f.flush(); os.fsync(f.fileno())
            print(f"    seed={seed} saved → {seed_out} (+ {jsonl_path})")
            per_seed_outputs[seed] = seed_out

    if not all_results:
        print("No results collected.")
        return

    summaries = aggregate(all_results)

    print(f"\n{'='*72}")
    print(f"  Merged summary (n={len(all_results)} = {args.n} × {len(seeds)} seeds)")
    print(f"{'='*72}")
    print(f"  {'Method':<28}  {'Rec-Hop2':>8}  {'Rec-Type':>8}  {'Rec-Exact':>9}  "
          f"{'Tokens':>7}  {'Size':>5}")
    print(f"  {'-'*78}")
    order = ["Greedy-Point-LLM", "TopK-5-LLM",
             "Window-4-LLM", "Window-8-LLM", "LocalRepair-2Hop-LLM",
             "Full-Graph-LLM",
             "TraceScan-w1-LLM", "TraceScan-w2-LLM", "TraceScan-Full-LLM",
             "LLMRepair-Full-Plan-LLM",
             "WM-SAR-LLM"]
    for m in order:
        s = summaries.get(m, {})
        if not s:
            continue
        print(f"  {m:<28}  {s.get('rec_hop2',0):>8.3f}  {s.get('rec_type',0):>8.3f}  "
              f"{s.get('rec_exact',0):>9.3f}  "
              f"{s.get('mean_tokens',0):>7.0f}  {s.get('mean_region_size',0):>5.1f}")

    # Per-seed Rec-Exact for WM-SAR-LLM (std across seeds = health signal)
    per_seed_wmsar = {}
    if len(seeds) > 1:
        print(f"\n  Per-seed Rec-Exact for WM-SAR-LLM:")
        for seed in seeds:
            vals = [p["results"]["WM-SAR-LLM"]["rec_exact"]
                    for p in all_per_instance if p["seed"] == seed
                    and "WM-SAR-LLM" in p["results"]]
            mean_v = float(np.mean(vals)) if vals else 0.0
            per_seed_wmsar[seed] = {"n": len(vals), "rec_exact_mean": mean_v}
            print(f"    seed={seed}  Rec-Exact (mean over {len(vals)}) = {mean_v:.3f}")
        per_seed_means = [v["rec_exact_mean"] for v in per_seed_wmsar.values()]
        cs_std = float(np.std(per_seed_means))
        print(f"    cross-seed std = {cs_std:.3f}  "
              f"({'HEALTHY' if cs_std <= 0.05 else 'FLAG: >0.05'})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    output = {
        "n": len(all_results),
        "seeds": seeds,
        "seed": args.seed,
        "model": args.model,
        "summaries": summaries,
        "per_seed_wmsar_rec_exact": per_seed_wmsar,
        "per_instance": build_per_instance(all_per_instance),
        "per_seed_files": per_seed_outputs,
    }
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Merged results saved to: {args.out}")


if __name__ == "__main__":
    main()
