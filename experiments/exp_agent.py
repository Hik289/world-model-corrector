import argparse
import json
import os
import sys

import numpy as np


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wm_sar.agent_calling_tree import generate_calling_trees
from wm_sar.engineering_baselines import run_all_baselines
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


    trees = generate_calling_trees(n=args.n, seed=args.seed)
    G_list = [t.G for t in trees]

    print(f"  Generated {len(G_list)} agent calling-tree failure graphs")
    sizes = [G.number_of_nodes() for G in G_list]
    print(f"  Node count: {np.mean(sizes):.1f} ± {np.std(sizes):.1f} "
          f"(range {min(sizes)}-{max(sizes)})")


    rhos = [amp.rho_B(G, set(G.nodes()), weight_norm=1.0) for G in G_list]
    slopes = [amp.error_growth_slope(
                  amp.simulate_error_propagation(G, set(), H=args.H_max),
                  h_start=4, h_end=args.H_max)
              for G in G_list]

    n_superadditive = 0
    for G in G_list:
        L_X, L_A, M_X, M_A = amp._estimate_propagation_gains(G)
        rho = amp.rho_B(G, set(G.nodes()))
        if rho > max(L_X, M_A) + 1e-4:
            n_superadditive += 1
    frac_super = n_superadditive / len(G_list)

    print("\n  Pre-repair statistics:")
    print(f"    mean ρ(B)         = {np.mean(rhos):.4f} ± {np.std(rhos):.4f}")
    print(f"    mean GrowthSlope  = {np.mean(slopes):.4f} ± {np.std(slopes):.4f}")
    print(f"    T2 super-add (ρ(B)>max(L_X,M_A)): {n_superadditive}/{len(G_list)} = {frac_super:.1%}")


    print("\n  Running baselines...\n")
    print(f"  {'Method':<28}  {'ρ_red':>6}  {'MSE@32':>8}  {'slope':>7}  {'conn':>5}  {'IoU':>6}")
    print(f"  {'-'*68}")
    summaries = run_all_baselines(G_list, verbose=True)


    print(f"\n{'='*60}")
    print("  Multi-step error table (NodeMSE@H)")
    print(f"{'='*60}")
    horizons = [1, 4, 8, 16, 32]
    header = f"  {'Method':<28}" + "".join(f"  H={H:2d}" for H in horizons) + \
             "  slope_after  rho_red"
    print(header)
    print("  " + "-" * (len(header) - 2))


    methods_sorted = sorted(summaries.keys(),
                             key=lambda m: summaries[m].get("NodeMSE_after", {}).get(32, 99))
    for name in methods_sorted:
        s = summaries[name]
        mse_a = s.get("NodeMSE_after", {})
        row = f"  {name:<28}" + "".join(f"  {mse_a.get(H, float('nan')):.4f}" for H in horizons)
        row += f"  {s.get('mean_growth_slope_after', float('nan')):+.4f}"
        row += f"  {s.get('mean_rho_reduction', 0.0):.4f}"
        print(row)


    print("\n  T4 Planning Regret Reduction:")
    for name in methods_sorted:
        s = summaries[name]
        rr = s.get("mean_regret_reduction", 0.0)
        rb_b = s.get("mean_return_bound_before", 0.0)
        rb_a = s.get("mean_return_bound_after", 0.0)
        print(f"    {name:<28}  regret_reduction={rr:.4f}  "
              f"bound_before={rb_b:.4f}  bound_after={rb_a:.4f}")


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
