"""Experiment: WM-SAR vs Engineering Baselines on Agent Calling-Tree Dataset.

Dataset: heterogeneous agent calling-tree testbed (matching analysis/main.tex §4.3)
    - 22-30 nodes, 9 types, 6 edge types, 8-dim state vectors
    - Failure: root-cause injection → cascade propagation (gain=1.1)

Baselines (engineering methods — all could use LLM for repair, differ in selection):
    Greedy-Point(K=1)  : Repair single highest-error node
    TopK-Point(K=3/5)  : Repair top-K nodes by error (disconnected)
    Window-2/4/8-Point : Sliding window of k steps
    LocalRepair-2/3Hop : k-hop neighbourhood of highest-error node
    CascadeRepair      : Topological scan until error drops
    Oracle             : Ground-truth corrupted region
    WM-SAR             : GEAF + ρ(B)-minimisation guided subgraph

Metrics:
    NodeMSE@H          : Post-repair error at horizon H (T1 simulation)
    GrowthSlope        : d(log e_k)/dk (T1 corollary: → log L_X)
    ρ(B) reduction     : How much coupling amplification is reduced (T2)
    ReturnError@H      : Planning regret bound from T4
    IoU vs GT          : Region localisation quality
    Connected          : Whether repair region is connected

Key claims:
    C1: Engineering pointwise methods leave ρ(B) nearly unchanged
        → multi-step error continues to grow (GrowthSlope ≈ before)
    C2: WM-SAR significantly reduces ρ(B) → flattens GrowthSlope
    C3: Window-k with k ≤ 2 produces disconnected regions far from root cause
    C4: ρ(B_G) > max(L_X, M_A) in most instances (T2 cross-coupling active)
    C5: WM-SAR NodeMSE@32 ≤ Oracle NodeMSE@32 × 1.05 (near-oracle performance)
"""

import argparse
import json
import os
import sys

import numpy as np

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wm_sar.agent_calling_tree import generate_calling_trees
from wm_sar.engineering_baselines import run_all_baselines, ALL_BASELINES
from wm_sar import amplification as amp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--H_max", type=int, default=32)
    parser.add_argument("--out", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                              "results", "exp_agent.json"))
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Agent Calling-Tree Experiment: n={args.n}, seed={args.seed}")
    print(f"{'='*60}\n")

    # --- Generate instances ---
    trees = generate_calling_trees(n=args.n, seed=args.seed)
    G_list = [t.G for t in trees]

    print(f"  Generated {len(G_list)} agent calling-tree failure graphs")
    sizes = [G.number_of_nodes() for G in G_list]
    print(f"  Node count: {np.mean(sizes):.1f} ± {np.std(sizes):.1f} "
          f"(range {min(sizes)}-{max(sizes)})")

    # --- Pre-experiment statistics ---
    rhos = [amp.rho_B(G, set(G.nodes()), weight_norm=1.0) for G in G_list]
    slopes = [amp.error_growth_slope(
                  amp.simulate_error_propagation(G, set(), H=args.H_max),
                  h_start=4, h_end=args.H_max)
              for G in G_list]
    # T2 claim: ρ(B) > max(L_X, M_A)?
    n_superadditive = 0
    for G in G_list:
        L_X, L_A, M_X, M_A = amp._estimate_propagation_gains(G)
        rho = amp.rho_B(G, set(G.nodes()))
        if rho > max(L_X, M_A) + 1e-4:
            n_superadditive += 1
    frac_super = n_superadditive / len(G_list)

    print(f"\n  Pre-repair statistics:")
    print(f"    mean ρ(B)         = {np.mean(rhos):.4f} ± {np.std(rhos):.4f}")
    print(f"    mean GrowthSlope  = {np.mean(slopes):.4f} ± {np.std(slopes):.4f}")
    print(f"    T2 super-add (ρ(B)>max(L_X,M_A)): {n_superadditive}/{len(G_list)} = {frac_super:.1%}")

    # --- Run all baselines ---
    print(f"\n  Running baselines...\n")
    print(f"  {'Method':<28}  {'ρ_red':>6}  {'MSE@32':>8}  {'slope':>7}  {'conn':>5}  {'IoU':>6}")
    print(f"  {'-'*68}")
    summaries = run_all_baselines(G_list, verbose=True)

    # --- Print comparison table ---
    print(f"\n{'='*60}")
    print(f"  Multi-step error table (NodeMSE@H)")
    print(f"{'='*60}")
    horizons = [1, 4, 8, 16, 32]
    header = f"  {'Method':<28}" + "".join(f"  H={H:2d}" for H in horizons) + \
             "  slope_after  rho_red"
    print(header)
    print("  " + "-" * (len(header) - 2))

    # Sort by MSE@32 after repair (ascending = better)
    methods_sorted = sorted(summaries.keys(),
                             key=lambda m: summaries[m].get("NodeMSE_after", {}).get(32, 99))
    for name in methods_sorted:
        s = summaries[name]
        mse_a = s.get("NodeMSE_after", {})
        row = f"  {name:<28}" + "".join(f"  {mse_a.get(H, float('nan')):.4f}" for H in horizons)
        row += f"  {s.get('mean_growth_slope_after', float('nan')):+.4f}"
        row += f"  {s.get('mean_rho_reduction', 0.0):.4f}"
        print(row)

    # --- T4 planning regret comparison ---
    print(f"\n  T4 Planning Regret Reduction:")
    for name in methods_sorted:
        s = summaries[name]
        rr = s.get("mean_regret_reduction", 0.0)
        rb_b = s.get("mean_return_bound_before", 0.0)
        rb_a = s.get("mean_return_bound_after", 0.0)
        print(f"    {name:<28}  regret_reduction={rr:.4f}  "
              f"bound_before={rb_b:.4f}  bound_after={rb_a:.4f}")

    # --- Save results ---
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    output = {
        "n": args.n,
        "seed": args.seed,
        "H_max": args.H_max,
        "dataset_stats": {
            "mean_n_nodes": float(np.mean(sizes)),
            "std_n_nodes": float(np.std(sizes)),
            "mean_rho_B": float(np.mean(rhos)),
            "std_rho_B": float(np.std(rhos)),
            "mean_growth_slope": float(np.mean(slopes)),
            "frac_t2_superadditive": float(frac_super),
        },
        "summaries": {
            name: {
                k: (v if not isinstance(v, dict) else {str(kk): vv for kk, vv in v.items()})
                for k, v in s.items()
                if not isinstance(v, set)
            }
            for name, s in summaries.items()
        },
    }
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {args.out}")


if __name__ == "__main__":
    main()
