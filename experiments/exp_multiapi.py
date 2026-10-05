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
from wm_sar.act_text import CONTEXT_PROTOCOL_VERSION, tree_to_text, build_locate_prompt, parse_locate_response

import networkx as nx


MODELS = {
    "gpt-4o-mini":      {"backend": "openai", "model": "gpt-4o-mini"},
    "gpt-4o":           {"backend": "openai", "model": "gpt-4o"},
    "gemini-2.5-flash": {"backend": "gemini", "model": "gemini-2.5-flash"},
}


def get_regions(G: nx.DiGraph) -> dict[str, set]:

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
    lat = (time.time() - t0) * 1000
    return {
        "method": method_name,
        "region_size": len(node_list),
        "tokens": token_cost,
        "latency_ms": lat,
        "rec_exact":  int(parsed["recovered_exact"]),
        "rec_type":   int(parsed["recovered_type"]),
        "rec_hop2":   int(parsed["recovered_hop2"]),
        "confidence": float(parsed["confidence"]),
        "identified_nodes": parsed["identified_nodes"],
        "valid_json": parsed["valid_json"],
        "valid_response": parsed["valid_response"],
        "invalid_reason": parsed["invalid_reason"],
        "explanation": parsed["explanation"],
        "api_error": api_error,
    }


def run_one_model(model_key: str, cfg: dict, trees, n: int, verbose=True,
                  seed: int = 42):


    client = LLMClient(
        model=cfg["model"],
        backend=cfg.get("backend", "openai"),
        temperature=0.0,
        max_tokens=400,
    )

    all_results = {m: [] for m in ["Greedy-Point", "TopK-5", "Window-4",
                                    "LocalRepair-2Hop", "WM-SAR"]}
    per_instance = []

    for i, tree in enumerate(trees[:n]):
        G = tree.G
        true_root = tree.root_cause_node
        if verbose and i % 5 == 0:
            print(f"    [{model_key}] {i+1}/{n} ...", end="\r", flush=True)

        regions = get_regions(G)

        inst_rec = {
            "instance_id":    f"s{seed}_i{i:03d}",
            "seed":           seed,
            "model":          model_key,
            "run_metadata": {
                "context_protocol": CONTEXT_PROTOCOL_VERSION,
                "model": client.model,
                "backend": client.backend,
                "temperature": client.temperature,
                "max_tokens": client.max_tokens,
            },
            "true_root":      true_root,
            "true_root_type": G.nodes[true_root].get("node_type", ""),
            "n_nodes":        G.number_of_nodes(),
            "results":        {},
        }
        for method, region in regions.items():
            r = call_llm_region(G, region, true_root, client, method)
            all_results[method].append(r)
            inst_rec["results"][method] = dict(r)
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
        valid_rows = [r for r in rows if r.get("valid_response") is True]
        token_values = [r["tokens"] for r in rows if r.get("tokens") is not None]
        out[method] = {
            "n": len(valid_rows),
            "n_attempted": n,
            "n_valid": len(valid_rows),
            "n_invalid": sum(r.get("valid_response") is False for r in rows),
            "n_unverified": sum(r.get("valid_response") is None for r in rows),
            "n_api_errors": sum(bool(r.get("api_error")) for r in rows),
            "n_token_measured": len(token_values),
            "recall_denominator": "valid_response",
            "mean_tokens": float(np.mean(token_values)) if token_values else None,
            "mean_region_size": float(np.mean([r["region_size"] for r in rows])),
            "mean_latency_ms": float(np.mean([r["latency_ms"] for r in rows])),
        }
        for metric in ("rec_exact", "rec_type", "rec_hop2"):
            out[method][metric] = (
                float(np.mean([r[metric] for r in valid_rows])) if valid_rows else None
            )
            out[method][f"{metric}_attempted"] = float(np.mean([r[metric] for r in rows]))
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
    if args.n < 1:
        parser.error("n must be at least 1")
    requested_models = [key.strip() for key in args.models.split(",")]
    if args.models != "all" and any(key not in MODELS for key in requested_models):
        parser.error("models contains an unknown model key")

    selected = MODELS if args.models == "all" else {
        k: MODELS[k] for k in requested_models}

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
            row += f"  {v:>14.3f}" if v is not None else f"  {'n/a':>14}"
        print(row)

    print("\n  Tokens comparison")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for m in methods:
        row = f"  {m:<22}"
        for k in model_keys:
            v = all_model_results[k].get(m, {}).get("mean_tokens", float("nan"))
            row += f"  {v:>14.0f}" if v is not None else f"  {'n/a':>14}"
        print(row)


    print("\n  WM-SAR Rec-Exact advantage over best engineering baseline:")
    for k in model_keys:
        wmsar = all_model_results[k].get("WM-SAR", {}).get("rec_exact")
        others = [all_model_results[k].get(m, {}).get("rec_exact")
                  for m in ["Greedy-Point", "TopK-5", "Window-4", "LocalRepair-2Hop"]]
        others = [v for v in others if v is not None]
        if wmsar is None or not others:
            print(f"    {k:<20}  insufficient valid responses")
            continue
        best_eng = max(others)
        print(f"    {k:<20}  WM-SAR={wmsar:.3f}  best_eng={best_eng:.3f}  "
              f"gap={wmsar-best_eng:+.3f}")


    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    output = {
        "n": args.n,
        "seed": args.seed,
        "models": list(selected.keys()),
        "context_protocol": CONTEXT_PROTOCOL_VERSION,
        "results": all_model_results,
        "per_instance": per_instance_rows,
    }
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {args.out}")


if __name__ == "__main__":
    main()
