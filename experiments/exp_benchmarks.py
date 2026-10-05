import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wm_sar.benchmark_graphs import BENCHMARK_GENERATORS
from wm_sar.engineering_baselines import run_all_baselines
from wm_sar import amplification as amp
from wm_sar.engineering_baselines import _aggregate, wmsar_repair


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
    return _aggregate([wmsar_repair(t.G) for t in trees])


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
              f"{s.get('NodeMSE_after', {}).get(32, float('nan')):>8.2f} "
              f"{s.get('mean_growth_slope_after', float('nan')):>+11.5f} "
              f"{s.get('mean_region_size',0):>6.1f} "
              f"{s.get('mean_iou',0):>6.3f}")
    print(SEP)
    if wmsar_s:
        s = wmsar_s
        print(f"{'WM-SAR ★':<26} {s.get('mean_rho_reduction',0):>7.3f} "
              f"{s.get('NodeMSE_after', {}).get(32, float('nan')):>8.2f} "
              f"{s.get('mean_growth_slope_after', float('nan')):>+11.5f} "
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
    if args.n < 1:
        ap.error("n must be at least 1")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    output = {"n": args.n, "seed": args.seed,
              "weight_norm": 0.9, "evaluation_kind": "benchmark_inspired_topology",
              "rollout_model": "legacy_single_channel_proxy",
              "benchmarks": list(BENCHMARK_GENERATORS.keys()),
              "results": {}}

    for bench_name, generator in BENCHMARK_GENERATORS.items():
        print(f"\n{'='*60}\n  {bench_name}  (n={args.n})\n{'='*60}")
        t0 = time.time()
        trees  = generator(n=args.n, seed=args.seed)
        G_list = [t.G for t in trees]
        print(f"  Generated {len(trees)} graphs in {time.time()-t0:.1f}s")


        summaries = run_all_baselines(G_list, verbose=True)

        wmsar_s = summaries.get("WM-SAR", {})

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
