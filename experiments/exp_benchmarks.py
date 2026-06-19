"""Experiment: WM-SAR on SWE-bench / WebArena / AgentBench-OS topologies.

Follows the same pattern as exp_agent.py — passes G_list to run_all_baselines.

Usage:
    python3 experiments/exp_benchmarks.py --n 50 --seed 42
"""

import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wm_sar.benchmark_graphs import BENCHMARK_GENERATORS
from wm_sar.engineering_baselines import run_all_baselines, ALL_BASELINES
from wm_sar import amplification as amp
from wm_sar.baselines import wm_sar as wm_sar_select


def benchmark_stats(trees) -> dict:
    Ns, rhos, superadd = [], [], []
    for t in trees:
        G = t.G
        Ns.append(G.number_of_nodes())
        all_nodes = set(G.nodes())
        rho = amp.rho_B(G, all_nodes)
        rhos.append(rho)
        blocks = amp._estimate_propagation_gains(G)
        L_X, L_A, M_X, M_A = blocks
        superadd.append(1.0 if rho > max(L_X, M_A) + 1e-6 else 0.0)
    return dict(
        mean_N=float(np.mean(Ns)),
        std_N=float(np.std(Ns)),
        mean_rhoB=float(np.mean(rhos)),
        std_rhoB=float(np.std(rhos)),
        superadditivity_pct=float(np.mean(superadd)) * 100,
    )


def wmsar_summary(trees) -> dict:
    """Run WM-SAR on each tree and compute summary."""
    from wm_sar.engineering_baselines import _evaluate_repair
    results = []
    for t in trees:
        G = t.G
        try:
            region = wm_sar_select(G)
            r = _evaluate_repair(G, region, "WM-SAR")
            results.append(r)
        except Exception as e:
            pass
    if not results:
        return {}
    return {
        "mean_rho_reduction": float(np.mean([r.rho_reduction for r in results])),
        "std_rho_reduction":  float(np.std( [r.rho_reduction for r in results])),
        "mean_mse_32":        float(np.mean([r.mse_profile_after.get(32, 0) for r in results])),
        "mean_slope":         float(np.mean([r.growth_slope_after for r in results])),
        "mean_region_size":   float(np.mean([r.region_size for r in results])),
        "mean_iou":           float(np.mean([r.iou_vs_gt for r in results])),
        "n": len(results),
    }


def print_table(bench_name: str, summaries: dict, wmsar_s: dict, stats: dict):
    METHODS = [
        "Greedy-Point(K=1)", "Window-4-Point", "TopK-Point(K=5)",
        "LocalRepair-2Hop", "LocalRepair-3Hop", "CascadeRepair",
    ]
    SEP = "─" * 75
    print(f"\n{'═'*75}")
    print(f"  {bench_name}   N={stats['mean_N']:.1f}  "
          f"ρ(B)={stats['mean_rhoB']:.3f}  T2={stats['superadditivity_pct']:.0f}%")
    print(f"{'═'*75}")
    print(f"{'Method':<26} {'ρ-red':>7} {'MSE@32':>8} {'Slope':>11} {'Size':>6} {'IoU':>6}")
    print(SEP)
    for m in METHODS:
        if m not in summaries:
            continue
        s = summaries[m]
        print(f"{m:<26} {s.get('mean_rho_reduction',0):>7.3f} "
              f"{s.get('mean_mse_32',0):>8.2f} "
              f"{s.get('mean_slope',0):>+11.5f} "
              f"{s.get('mean_region_size',0):>6.1f} "
              f"{s.get('mean_iou',0):>6.3f}")
    print(SEP)
    if wmsar_s:
        s = wmsar_s
        print(f"{'WM-SAR ★':<26} {s.get('mean_rho_reduction',0):>7.3f} "
              f"{s.get('mean_mse_32',0):>8.2f} "
              f"{s.get('mean_slope',0):>+11.5f} "
              f"{s.get('mean_region_size',0):>6.1f} "
              f"{s.get('mean_iou',0):>6.3f}")
    print(SEP)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",    type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out",  type=str,
                    default=os.path.join(os.path.dirname(__file__),
                                         "results", "exp_benchmarks.json"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    output = {"n": args.n, "seed": args.seed,
              "benchmarks": list(BENCHMARK_GENERATORS.keys()),
              "results": {}}

    for bench_name, generator in BENCHMARK_GENERATORS.items():
        print(f"\n{'='*60}\n  {bench_name}  (n={args.n})\n{'='*60}")
        t0 = time.time()
        trees  = generator(n=args.n, seed=args.seed)
        G_list = [t.G for t in trees]
        print(f"  Generated {len(trees)} graphs in {time.time()-t0:.1f}s")

        # Engineering baselines
        summaries = run_all_baselines(G_list, verbose=True)
        # WM-SAR
        wmsar_s   = wmsar_summary(trees)
        if wmsar_s:
            summaries["WM-SAR"] = wmsar_s

        stats = benchmark_stats(trees)
        print_table(bench_name, summaries, wmsar_s, stats)

        output["results"][bench_name] = {
            "dataset_stats": stats,
            "summaries": summaries,
        }

    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ Saved: {args.out}")


if __name__ == "__main__":
    main()
