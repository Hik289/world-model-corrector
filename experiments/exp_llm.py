"""
exp_llm.py — Main LLM experiment with real GPT-4o-mini API calls.

Compares:
  TraceScan-w1/w2/w4-LLM  : GPT-4o-mini with limited context window
  TraceScan-Full-LLM       : GPT-4o-mini with full trace context
  LLMRepair-Full-Plan-LLM  : GPT-4o-mini full replan
  LastError-Heuristic      : No LLM, highest numeric error
  WM-SAR-LLM               : Graph analysis (GEAF) → ONE GPT-4o-mini call

Usage:
  python experiments/exp_llm.py [--n 30] [--seed 42] [--outfile results/exp_llm.json]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from wm_sar import data_generator as dg, failure_graph as fg
from wm_sar.llm_client import LLMClient
from wm_sar.llm_baselines import run_all_llm_baselines
from wm_sar.text_scenarios import rollout_to_steps


def run_experiment(n: int = 30, seed: int = 42, verbose: bool = True) -> dict:
    print(f"=== LLM Experiment: n={n}, seed={seed} ===")
    print(f"  Using gpt-4o-mini for all baselines + WM-SAR repair")
    print()

    rng = np.random.default_rng(seed)

    # --- LLM clients ---
    client = LLMClient(
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=512,
    )

    # --- Generate rollouts ---
    rollouts = dg.generate_agent_wm_rollouts(n=n, seed=seed)
    print(f"  Generated {len(rollouts)} rollouts")

    # --- Accumulate results ---
    method_names = [
        "TraceScan-w1-LLM",
        "TraceScan-w2-LLM",
        "TraceScan-w4-LLM",
        "TraceScan-Full-LLM",
        "LLMRepair-Full-Plan-LLM",
        "LastError-Heuristic",
        "WM-SAR-LLM",
    ]
    agg = {
        m: {"recovered": [], "token_cost": [], "latency": [], "iou": [], "n_calls": []}
        for m in method_names
    }

    for idx, rollout in enumerate(rollouts):
        # Build failure graph
        G = fg.agent_rollout_to_graph(rollout)

        # Generate text for this rollout
        rollout_steps, task_desc, failure_desc = rollout_to_steps(
            rollout, rng=np.random.default_rng(seed + idx)
        )

        if verbose and (idx % 5 == 0):
            print(f"  [{idx+1}/{n}] T={len(rollout.steps)} root_t={rollout.root_cause_t} "
                  f"scenario='{failure_desc[:50]}...'")

        # Run all LLM baselines
        results = run_all_llm_baselines(
            G=G,
            rollout_steps=rollout_steps,
            rollout=rollout,
            failure_desc=failure_desc,
            client_fast=client,
            client_repair=client,
        )

        for method, res in results.items():
            if method not in agg:
                continue
            agg[method]["recovered"].append(int(res.recovered))
            agg[method]["token_cost"].append(res.token_cost)
            agg[method]["latency"].append(res.latency_ms)
            agg[method]["iou"].append(res.region_iou)
            agg[method]["n_calls"].append(res.n_llm_calls)

    # --- Compute summary ---
    summary = {}
    for method in method_names:
        d = agg[method]
        nn = len(d["recovered"])
        if nn == 0:
            continue
        rec = sum(d["recovered"]) / nn
        avg_tok = sum(d["token_cost"]) / nn
        avg_lat = sum(d["latency"]) / nn
        avg_iou = sum(d["iou"]) / nn
        avg_calls = sum(d["n_calls"]) / nn
        n_rec = max(1, sum(d["recovered"]))
        tok_per_rec = sum(d["token_cost"]) / n_rec
        summary[method] = {
            "recovery": round(rec, 4),
            "avg_tokens": round(avg_tok, 1),
            "tok_per_rec": round(tok_per_rec, 1),
            "avg_latency_ms": round(avg_lat, 1),
            "avg_iou": round(avg_iou, 4),
            "avg_llm_calls": round(avg_calls, 2),
            "n": nn,
        }

    # --- Print table ---
    print()
    print(f"{'Method':<28} {'Rec':>6} {'Tokens':>8} {'Tok/Rec':>9} "
          f"{'Lat(ms)':>8} {'IoU':>6} {'#Calls':>7}")
    print("-" * 75)
    for method in method_names:
        if method not in summary:
            continue
        s = summary[method]
        print(f"{method:<28} {s['recovery']:>6.3f} {s['avg_tokens']:>8.0f} "
              f"{s['tok_per_rec']:>9.0f} {s['avg_latency_ms']:>8.0f} "
              f"{s['avg_iou']:>6.3f} {s['avg_llm_calls']:>7.2f}")

    return {"summary": summary, "raw": agg, "n": n, "seed": seed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outfile", type=str,
                        default=str(Path(__file__).parent / "results" / "exp_llm.json"))
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    results = run_experiment(n=args.n, seed=args.seed)

    with open(args.outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.outfile}")


if __name__ == "__main__":
    main()
