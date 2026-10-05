from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wm_sar.agent_calling_tree import generate_calling_trees
from wm_sar.engineering_baselines import _aggregate, wmsar_repair
from wm_sar.region_extractor import ReCoreConfig


def variants():
    return {
        "ReCore": ReCoreConfig(),
        "Without GEAF": ReCoreConfig(use_geaf=False),
        "Without coupling": ReCoreConfig(use_coupling=False),
        "Without spectral-relief growth": ReCoreConfig(use_rho_relief=False),
        "Without pruning": ReCoreConfig(use_pruning=False),
        "Without growing": ReCoreConfig(use_growing=False),
    }


def run_ablation(graphs, H_max=32):
    if H_max < 1:
        raise ValueError("H_max must be at least 1")
    graphs = list(graphs)
    if not graphs:
        raise ValueError("at least one graph is required")
    summaries = {}
    for name, config in variants().items():
        results = []
        for graph in graphs:
            result = wmsar_repair(graph, config, H_max=H_max)
            result.method = name
            results.append(result)
        summaries[name] = _aggregate(results)
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--H_max", type=int, default=32)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "results" / "exp_agent_ablation.json")
    args = parser.parse_args()
    if args.n < 1 or args.H_max < 1:
        parser.error("n and H_max must be at least 1")
    trees = generate_calling_trees(n=args.n, seed=args.seed)
    output = {
        "n": args.n,
        "seed": args.seed,
        "H_max": args.H_max,
        "dispersion": "population_standard_deviation",
        "rollout_model": "legacy_single_channel_proxy",
        "regret_evaluation": "conditional_graph_estimated_bound",
        "spectral_relief_ablation_scope": "growth_score_and_acceptance_only",
        "configs": {name: asdict(config) for name, config in variants().items()},
        "summaries": run_ablation([tree.G for tree in trees], args.H_max),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, allow_nan=False)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
