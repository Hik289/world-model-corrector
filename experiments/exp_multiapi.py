"""Multi-API Comparison Experiment.

Compare WM-SAR vs engineering baselines across different LLM APIs:
    gpt-4o-mini      — OpenAI established cheap baseline
    gpt-4o           — OpenAI flagship-tier
    gemini-2.5-flash — Google fast / cheap

(Wave-B: human researcher restricted us to {gpt-4o, gpt-4o-mini,
 gemini-2.5-flash}. The earlier gpt-4.1-nano / gpt-4.1-mini cells
 from REVISION are kept in exp_multiapi_pre_wave_b.json only.)

For each model × method pair, measure:
    Rec-Exact   : exact root-cause node identified
    Rec-Type    : correct node type identified
    Rec-2Hop    : within 2 hops of root cause
    Tokens      : prompt + completion tokens
    Latency     : ms per call

Key claim: WM-SAR's region selection advantage is LLM-AGNOSTIC.
All LLMs benefit from being shown the GEAF-guided causal region vs. a
random/greedy selection. Stronger models improve all methods, but
WM-SAR's relative advantage (Rec-Exact gap) is maintained or grows.

Methods compared (using same prompts):
    Greedy-Point-LLM  (1 node)
    Window-4-LLM      (4 nodes, context-limited)
    TopK-5-LLM        (5 nodes, disconnected)
    LocalRepair-2Hop  (18+ nodes, large context)
    WM-SAR-LLM        (7-8 nodes, amplification-guided)
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wm_sar.agent_calling_tree import generate_calling_trees
from wm_sar.engineering_baselines import greedy_point, topk_point, window_repair, local_khop
from wm_sar.region_extractor import WMSAR, WMSARConfig
from wm_sar.llm_client import LLMClient
from wm_sar.act_text import tree_to_text, build_locate_prompt, parse_locate_response

import networkx as nx

# ── Models to compare ────────────────────────────────────────────────────────
MODELS = {
    "gpt-4o-mini":      {"backend": "openai", "model": "gpt-4o-mini"},
    "gpt-4o":           {"backend": "openai", "model": "gpt-4o"},
    "gemini-2.5-flash": {"backend": "gemini", "model": "gemini-2.5-flash"},
}

# ── Methods to compare ────────────────────────────────────────────────────────
def get_regions(G: nx.DiGraph) -> dict[str, set]:
    """Pre-compute all repair regions for a graph (model-agnostic)."""
    extractor = WMSAR(WMSARConfig())
    return {
        "Greedy-Point":     greedy_point(G, K=1).selected_nodes,
        "TopK-5":           topk_point(G, K=5).selected_nodes,
        "Window-4":         window_repair(G, window=4).selected_nodes,
        "LocalRepair-2Hop": local_khop(G, k=2).selected_nodes,
        "WM-SAR":           extractor.repair_region(G),
    }


def call_llm_region(G, region, true_root, client, method_name) -> dict:
    text, node_list = tree_to_text(G, selected_nodes=region, include_edges=True)
    system, user = build_locate_prompt(text, node_list, G)
    t0 = time.time()
    resp = client.chat(system=system, user=user)
    lat = (time.time() - t0) * 1000
    parsed = parse_locate_response(resp.text, true_root, G)
    return {
        "method": method_name,
        "region_size": len(region),
        "tokens": resp.total_tokens,
        "latency_ms": lat,
        "rec_exact":  int(parsed["recovered_exact"]),
        "rec_type":   int(parsed["recovered_type"]),
        "rec_hop2":   int(parsed["recovered_hop2"]),
        "confidence": float(parsed["confidence"]),
    }


def run_one_model(model_key: str, cfg: dict, trees, n: int, verbose=True,
                  seed: int = 42):
    """Run all methods on n trees with one model.

    Returns:
        (per_method_results, per_instance_rows)
        per_method_results: {method: [row, ...]}  (compat with aggregate_model)
        per_instance_rows:  [{instance_id, true_root, n_nodes, results}, ...]
    """
    try:
        client = LLMClient(
            model=cfg["model"],
            backend=cfg.get("backend", "openai"),
            temperature=0.0,
            max_tokens=400,
        )
    except Exception as e:
        print(f"  [{model_key}] client init failed: {e}")
        return {}, []

    all_results = {m: [] for m in ["Greedy-Point", "TopK-5", "Window-4",
                                    "LocalRepair-2Hop", "WM-SAR"]}
    per_instance = []

    for i, tree in enumerate(trees[:n]):
        G = tree.G
        true_root = tree.root_cause_node
        if verbose and i % 5 == 0:
            print(f"    [{model_key}] {i+1}/{n} ...", end="\r", flush=True)

        try:
            regions = get_regions(G)
        except Exception:
            continue

        inst_rec = {
            "instance_id":    f"s{seed}_i{i:03d}",
            "seed":           seed,
            "model":          model_key,
            "true_root":      true_root,
            "true_root_type": G.nodes[true_root].get("node_type", ""),
            "n_nodes":        G.number_of_nodes(),
            "results":        {},
        }
        for method, region in regions.items():
            try:
                r = call_llm_region(G, region, true_root, client, method)
                all_results[method].append(r)
                inst_rec["results"][method] = {
                    "rec_exact":   int(bool(r["rec_exact"])),
                    "rec_type":    int(bool(r["rec_type"])),
                    "rec_hop2":    int(bool(r["rec_hop2"])),
                    "tokens":      int(r["tokens"]),
                    "region_size": int(r["region_size"]),
                    "latency_ms":  float(r["latency_ms"]),
                    "confidence":  float(r.get("confidence", 0.0)),
                }
            except Exception:
                pass  # skip failed calls
        per_instance.append(inst_rec)

    if verbose:
        print(f"    [{model_key}] done ({n} instances)        ")
    return all_results, per_instance


def aggregate_model(results: dict[str, list]) -> dict[str, dict]:
    out = {}
    for method, rows in results.items():
        if not rows:
            continue
        n = len(rows)
        out[method] = {
            "n": n,
            "rec_exact":  float(np.mean([r["rec_exact"] for r in rows])),
            "rec_type":   float(np.mean([r["rec_type"] for r in rows])),
            "rec_hop2":   float(np.mean([r["rec_hop2"] for r in rows])),
            "mean_tokens": float(np.mean([r["tokens"] for r in rows])),
            "mean_region_size": float(np.mean([r["region_size"] for r in rows])),
            "mean_latency_ms": float(np.mean([r["latency_ms"] for r in rows])),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=15,
                        help="instances per model (default 15)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", type=str, default="all",
                        help="comma-separated model keys or 'all'")
    parser.add_argument("--out", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                              "results", "exp_multiapi.json"))
    args = parser.parse_args()

    selected = MODELS if args.models == "all" else {
        k: MODELS[k] for k in args.models.split(",") if k in MODELS}

    print(f"\n{'='*64}")
    print(f"  Multi-API Experiment: {list(selected.keys())}")
    print(f"  n={args.n} per model, seed={args.seed}")
    print(f"{'='*64}\n")

    trees = generate_calling_trees(n=args.n, seed=args.seed)

    all_model_results = {}
    per_instance_rows = []
    for model_key, cfg in selected.items():
        print(f"  ── {model_key} ──")
        raw, per_inst = run_one_model(model_key, cfg, trees, args.n,
                                        seed=args.seed)
        all_model_results[model_key] = aggregate_model(raw)
        per_instance_rows.extend(per_inst)

    # ── Print comparison table ───────────────────────────────────────────────
    methods = ["Greedy-Point", "TopK-5", "Window-4", "LocalRepair-2Hop", "WM-SAR"]
    model_keys = list(all_model_results.keys())

    print(f"\n{'='*64}")
    print("  Rec-Exact comparison (rows=methods, cols=models)")
    print(f"{'='*64}")
    hdr = f"  {'Method':<22}" + "".join(f"  {k[:14]:>14}" for k in model_keys)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for m in methods:
        row = f"  {m:<22}"
        for k in model_keys:
            v = all_model_results[k].get(m, {}).get("rec_exact", float("nan"))
            row += f"  {v:>14.3f}"
        print(row)

    print(f"\n  Tokens comparison")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for m in methods:
        row = f"  {m:<22}"
        for k in model_keys:
            v = all_model_results[k].get(m, {}).get("mean_tokens", float("nan"))
            row += f"  {v:>14.0f}"
        print(row)

    # ── WM-SAR advantage (Rec-Exact gap vs best baseline) ───────────────────
    print(f"\n  WM-SAR Rec-Exact advantage over best engineering baseline:")
    for k in model_keys:
        wmsar = all_model_results[k].get("WM-SAR", {}).get("rec_exact", 0)
        others = [all_model_results[k].get(m, {}).get("rec_exact", 0)
                  for m in ["Greedy-Point", "TopK-5", "Window-4", "LocalRepair-2Hop"]]
        best_eng = max(others) if others else 0
        print(f"    {k:<20}  WM-SAR={wmsar:.3f}  best_eng={best_eng:.3f}  "
              f"gap={wmsar-best_eng:+.3f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    output = {
        "n": args.n,
        "seed": args.seed,
        "models": list(selected.keys()),
        "results": all_model_results,
        "per_instance": per_instance_rows,
    }
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {args.out}")


if __name__ == "__main__":
    main()
