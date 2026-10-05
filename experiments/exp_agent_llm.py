import argparse
import json
import math
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
    CONTEXT_PROTOCOL_VERSION, tree_to_text, build_locate_prompt, parse_locate_response,
)

import networkx as nx

LLM_METHODS = (
    "Greedy-Point-LLM", "TopK-5-LLM", "Window-4-LLM", "Window-8-LLM",
    "LocalRepair-2Hop-LLM", "Full-Graph-LLM", "WM-SAR-LLM",
    "TraceScan-w1-LLM", "TraceScan-w2-LLM", "TraceScan-Full-LLM",
    "LLMRepair-Full-Plan-LLM",
)


def load_resume_records(path, run_metadata, seed, n):
    records = {}
    expected_ids = {f"s{seed}_i{i:03d}" for i in range(n)}
    required = {
        "rec_exact", "rec_type", "rec_hop2", "tokens", "region_size",
        "latency_ms", "valid_json", "valid_response",
    }
    with open(path) as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid resume JSON at line {line_number}; use a new output path") from exc
            if not isinstance(record, dict) or record.get("run_metadata") != run_metadata:
                raise ValueError(f"Incompatible resume protocol or model at line {line_number}; use a new output path")
            instance_id = record.get("instance_id")
            results = record.get("results")
            if (not isinstance(instance_id, str) or instance_id not in expected_ids
                    or record.get("seed") != seed
                    or not isinstance(record.get("true_root"), str)
                    or not isinstance(record.get("n_nodes"), int)
                    or not isinstance(results, dict)
                    or set(results) != set(LLM_METHODS)):
                raise ValueError(f"Invalid resume instance at line {line_number}")
            for result in results.values():
                if (not isinstance(result, dict) or not required.issubset(result)
                        or not isinstance(result["valid_json"], bool)
                        or not isinstance(result["valid_response"], bool)
                        or any(result[k] not in (0, 1) for k in ("rec_exact", "rec_type", "rec_hop2"))):
                    raise ValueError(f"Invalid resume result at line {line_number}")
                if result["valid_response"] and not result["valid_json"]:
                    raise ValueError(f"Invalid resume validity flags at line {line_number}")
                region_size, latency = result["region_size"], result["latency_ms"]
                if (isinstance(region_size, bool) or not isinstance(region_size, int)
                        or region_size < 0 or isinstance(latency, bool)
                        or not isinstance(latency, (int, float))
                        or not math.isfinite(latency) or latency < 0):
                    raise ValueError(f"Invalid resume measurement at line {line_number}")
                if (not result["valid_response"]
                        and any(result[k] for k in ("rec_exact", "rec_type", "rec_hop2"))):
                    raise ValueError(f"Invalid resume recovery at line {line_number}")
                tokens = result["tokens"]
                if tokens is not None and (isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0):
                    raise ValueError(f"Invalid resume token count at line {line_number}")
            if instance_id in records and records[instance_id] != record:
                raise ValueError(f"Conflicting duplicate resume instance {instance_id}")
            records[instance_id] = record
    return list(records.values())


def _topo_order(G: nx.DiGraph) -> list[str]:
    try:
        return list(nx.topological_sort(G))
    except Exception:
        return sorted(G.nodes(),
                      key=lambda v: G.nodes[v].get("time_step", 0))


def _tracescan_window(G: nx.DiGraph, w: int) -> set[str]:

    topo = _topo_order(G)
    errs = [(float(G.nodes[v].get("err", 0.0)), i, v) for i, v in enumerate(topo)]
    if not errs:
        return set()
    _, ctr_i, _ = max(errs)
    lo = max(0, ctr_i - w // 2)
    hi = min(len(topo), lo + w)
    lo = max(0, hi - w)
    return set(topo[lo:hi])


def _build_full_plan_prompt(tree_text: str, node_list: list[str]) -> tuple[str, str]:


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
        f"Eligible node IDs: {json.dumps(node_list)}.\n\n"
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


    text, node_list = tree_to_text(G, selected_nodes=region,
                                    include_edges=include_edges,
                                    max_nodes=max(len(region), 30))
    if prompt_builder is _build_full_plan_prompt:
        system, user = prompt_builder(text, node_list)
    else:
        system, user = prompt_builder(text, node_list, G)
    t0 = time.time()
    api_error = ""
    token_cost = None
    try:
        resp = client.chat(system=system, user=user)
    except Exception as exc:
        parsed = parse_locate_response("", true_root, G,
                                       allowed_nodes=set(node_list))
        parsed["invalid_reason"] = "api_error"
        api_error = type(exc).__name__
    else:
        pt, ct = resp.prompt_tokens, resp.completion_tokens
        if all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in (pt, ct)):
            token_cost = pt + ct
        parsed = parse_locate_response(resp.text, true_root, G,
                                       allowed_nodes=set(node_list))
    latency_ms = (time.time() - t0) * 1000.0
    return {
        "method": method_name,
        "region_size": len(node_list),
        "token_cost": token_cost,
        "latency_ms": latency_ms,
        "rec_exact": parsed["recovered_exact"],
        "rec_type": parsed["recovered_type"],
        "rec_hop2": parsed["recovered_hop2"],
        "identified_nodes": parsed["identified_nodes"],
        "confidence": parsed["confidence"],
        "valid_json": parsed["valid_json"],
        "valid_response": parsed["valid_response"],
        "invalid_reason": parsed["invalid_reason"],
        "explanation": parsed["explanation"],
        "api_error": api_error,
    }


def run_instance(G: nx.DiGraph, true_root: str, client: LLMClient) -> dict:

    results = {}


    rr = greedy_point(G, K=1)
    results["Greedy-Point-LLM"] = _call_llm_on_region(
        G, rr.selected_nodes, true_root, client, "Greedy-Point-LLM")


    rr = topk_point(G, K=5)
    results["TopK-5-LLM"] = _call_llm_on_region(
        G, rr.selected_nodes, true_root, client, "TopK-5-LLM")


    rr = window_repair(G, window=4)
    results["Window-4-LLM"] = _call_llm_on_region(
        G, rr.selected_nodes, true_root, client, "Window-4-LLM")


    rr = window_repair(G, window=8)
    results["Window-8-LLM"] = _call_llm_on_region(
        G, rr.selected_nodes, true_root, client, "Window-8-LLM")


    rr = local_khop(G, k=2)
    results["LocalRepair-2Hop-LLM"] = _call_llm_on_region(
        G, rr.selected_nodes, true_root, client, "LocalRepair-2Hop-LLM")


    full_region = set(G.nodes())
    results["Full-Graph-LLM"] = _call_llm_on_region(
        G, full_region, true_root, client, "Full-Graph-LLM")


    extractor = WMSAR(WMSARConfig())
    region = extractor.repair_region(G)
    results["WM-SAR-LLM"] = _call_llm_on_region(
        G, region, true_root, client, "WM-SAR-LLM")


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
        include_edges=True)

    return results


def aggregate(all_results: list[dict]) -> dict[str, dict]:


    methods = sorted({m for r in all_results for m in r.keys()})
    summaries = {}
    for m in methods:
        rows = [r[m] for r in all_results if m in r]
        n = len(rows)
        if n == 0:
            continue
        valid_rows = [r for r in rows if r.get("valid_response") is True]
        token_values = [r["token_cost"] for r in rows if r.get("token_cost") is not None]
        summaries[m] = {
            "n": n,
            "n_attempted": n,
            "n_valid": len(valid_rows),
            "n_invalid": sum(r.get("valid_response") is False for r in rows),
            "n_unverified": sum(r.get("valid_response") is None for r in rows),
            "n_api_errors": sum(bool(r.get("api_error")) for r in rows),
            "n_token_measured": len(token_values),
            "recall_denominator": "attempted",
            "rec_exact":  float(np.mean([r["rec_exact"] for r in rows])),
            "rec_type":   float(np.mean([r["rec_type"] for r in rows])),
            "rec_hop2":   float(np.mean([r["rec_hop2"] for r in rows])),
            "mean_tokens": float(np.mean(token_values)) if token_values else None,
            "mean_region_size": float(np.mean([r["region_size"] for r in rows])),
            "mean_latency_ms": float(np.mean([r["latency_ms"] for r in rows])),
        }
        for metric in ("rec_exact", "rec_type", "rec_hop2"):
            summaries[m][f"{metric}_valid"] = (
                float(np.mean([r[metric] for r in valid_rows])) if valid_rows else None
            )
        summaries[m]["tok_per_rec_hop2"] = (
            sum(token_values) / sum(r["rec_hop2"] for r in rows)
            if len(token_values) == n and any(r["rec_hop2"] for r in rows) else None
        )
    return summaries


def build_per_instance(per_instance_rows: list[dict]) -> list[dict]:


    out = []
    for row in per_instance_rows:
        rec = {
            "instance_id":   row["instance_id"],
            "seed":          row["seed"],
            "true_root":     row["true_root"],
            "true_root_type": row["true_root_type"],
            "n_nodes":       row["n_nodes"],
            "results": {},
            "run_metadata": row.get("run_metadata"),
        }
        for method, r in row["results"].items():
            rec["results"][method] = {
                "rec_exact":    int(bool(r["rec_exact"])),
                "rec_type":     int(bool(r["rec_type"])),
                "rec_hop2":     int(bool(r["rec_hop2"])),
                "tokens":       int(r["token_cost"]) if r.get("token_cost") is not None else None,
                "region_size":  int(r["region_size"]),
                "latency_ms":   float(r["latency_ms"]),
                "identified_nodes": list(r.get("identified_nodes", [])),
                "confidence":   float(r.get("confidence", 0.0)),
                "valid_json": r.get("valid_json"),
                "valid_response": r.get("valid_response"),
                "invalid_reason": r.get("invalid_reason", ""),
                "explanation": r.get("explanation", ""),
                "api_error": r.get("api_error", ""),
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
    parser.add_argument("--model", type=str, default=os.environ.get("LLM_MODEL"))
    parser.add_argument("--resume", action="store_true",
                        help="Resume: skip instances already present in "
                             "<out>_seed<S>.jsonl. The JSONL is appended to "
                             "instead of truncated.")
    parser.add_argument("--out", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                              "results", "exp_agent_llm.json"))
    args = parser.parse_args()
    if args.n < 1:
        parser.error("n must be at least 1")
    if not args.out.endswith(".json"):
        parser.error("out must end with .json")

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    if len(set(seeds)) != len(seeds):
        parser.error("seeds must be unique")

    print(f"\n{'='*60}")
    print("  Agent Calling-Tree LLM Experiment")
    model_label = args.model or os.environ.get("MODEL_NAME") or "configured default"
    print(f"  n={args.n} × seeds={seeds}, model={model_label}")
    print(f"{'='*60}\n")

    client = LLMClient(model=args.model, temperature=0.0, max_tokens=512)
    run_metadata = {
        "context_protocol": CONTEXT_PROTOCOL_VERSION,
        "model": client.model,
        "backend": client.backend,
        "temperature": client.temperature,
        "max_tokens": client.max_tokens,
    }

    all_per_instance = []
    all_results      = []
    per_seed_outputs = {}


    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    for seed in seeds:
        print(f"\n  ── seed={seed} ──")
        trees = generate_calling_trees(n=args.n, seed=seed)
        seed_per_instance = []
        seed_results = []
        jsonl_path = args.out.replace(".json", f"_seed{seed}.jsonl")


        done_ids: set[str] = set()
        resume_records = []
        if args.resume and os.path.exists(jsonl_path):
            resume_records = load_resume_records(jsonl_path, run_metadata, seed, args.n)
            done_ids = {rec["instance_id"] for rec in resume_records}
            if done_ids:
                print(f"    [resume] {len(done_ids)} instances already in "
                      f"{jsonl_path}; will skip those.")
        elif not args.resume:

            with open(jsonl_path, "w") as f:
                pass


        for rec in resume_records:
            pseudo_res = {}
            for meth, r in rec["results"].items():
                pseudo_res[meth] = dict(r, token_cost=r["tokens"])
            seed_results.append(pseudo_res)
            all_results.append(pseudo_res)
            row = {
                "instance_id": rec["instance_id"],
                "seed": rec["seed"],
                "true_root": rec["true_root"],
                "true_root_type": rec.get("true_root_type", ""),
                "n_nodes": rec["n_nodes"],
                "results": pseudo_res,
                "run_metadata": run_metadata,
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
                    "run_metadata": run_metadata,
                }
                seed_per_instance.append(row)
                all_per_instance.append(row)

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
                raise RuntimeError(f"Failed to complete or persist {instance_id}") from e


        if seed_results:
            seed_out = args.out.replace(".json", f"_seed{seed}.json")
            seed_payload = {
                "n": len(seed_results),
                "seed": seed,
                "model": client.model,
                "run_metadata": run_metadata,
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
        mean_tokens = s.get("mean_tokens")
        tokens_text = f"{mean_tokens:>7.0f}" if mean_tokens is not None else f"{'n/a':>7}"
        print(f"  {m:<28}  {s.get('rec_hop2',0):>8.3f}  {s.get('rec_type',0):>8.3f}  "
              f"{s.get('rec_exact',0):>9.3f}  "
              f"{tokens_text}  {s.get('mean_region_size',0):>5.1f}")


    per_seed_wmsar = {}
    if len(seeds) > 1:
        print("\n  Per-seed Rec-Exact for WM-SAR-LLM:")
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

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    output = {
        "n": len(all_results),
        "seeds": seeds,
        "seed": args.seed,
        "model": client.model,
        "run_metadata": run_metadata,
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
