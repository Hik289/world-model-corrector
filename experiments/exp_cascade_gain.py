"""Experiment: Cascade Gain Sensitivity.

Vary cascade gain α ∈ {0.7, 0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4}
(n=50, seed=42) and measure ρ(B)-reduction for key methods.

Hypothesis (from T4):
  When γ·ρ(B) > 1 (super-linear regret regime), WM-SAR's advantage
  over engineering methods GROWS because:
  (a) ρ(B) increases with α → T2 coupling becomes stronger
  (b) Engineering methods that ignore coupling are increasingly inadequate

Key claim: WM-SAR advantage = (ρ_WM-SAR - ρ_best_eng) grows monotonically with α.
"""

import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wm_sar.agent_calling_tree import generate_calling_trees
from wm_sar.engineering_baselines import run_all_baselines
from wm_sar import amplification as amp
from wm_sar.baselines import wm_sar as wm_sar_select
from wm_sar.engineering_baselines import _evaluate_repair


ALPHA_VALUES = [0.7, 0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4]
KEY_METHODS = ["Greedy-Point(K=1)", "Window-4-Point", "TopK-Point(K=5)",
               "LocalRepair-2Hop", "LocalRepair-3Hop", "WM-SAR"]


def inject_with_gain(G_orig, alpha: float):
    """Re-cascade errors with a new gain α (keeps graph topology + root cause)."""
    import copy, networkx as nx
    G = copy.deepcopy(G_orig)
    root = G.graph.get("gt_region", set())
    root_nodes = sorted(root)  # nodes with original error

    # Reset all errors
    for n in G.nodes():
        G.nodes[n]["err"] = 0.0

    # Re-inject with new gain
    if not root_nodes:
        return G
    # Set root cause error
    rc = root_nodes[0]
    G.nodes[rc]["err"] = 0.6  # canonical magnitude

    # BFS cascade with new alpha
    visited = {rc}
    queue = [rc]
    rng = np.random.default_rng(abs(hash(rc)) % 10000)
    while queue:
        nxt = []
        for n in queue:
            for child in G.successors(n):
                child_err = alpha * G.nodes[n]["err"] + abs(float(rng.normal(0, 0.04)))
                G.nodes[child]["err"] = max(G.nodes[child]["err"], child_err)
                if child not in visited:
                    visited.add(child)
                    nxt.append(child)
        queue = nxt
    return G


def wmsar_result(G):
    try:
        region = wm_sar_select(G)
        return _evaluate_repair(G, region, "WM-SAR")
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",    type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out",  type=str,
                    default=os.path.join(os.path.dirname(__file__),
                                         "results", "exp_cascade_gain.json"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"\n{'='*60}\n  Cascade Gain Sensitivity (n={args.n})\n{'='*60}")

    # Generate base trees once
    trees = generate_calling_trees(n=args.n, seed=args.seed)

    output = {"alphas": ALPHA_VALUES, "n": args.n, "seed": args.seed, "results": {}}

    for alpha in ALPHA_VALUES:
        print(f"\n  α={alpha:.2f} ...", end=" ", flush=True)
        t0 = time.time()

        # Re-cascade with this alpha
        G_list = [inject_with_gain(t.G, alpha) for t in trees]

        # Engineering baselines
        summaries = run_all_baselines(G_list, verbose=False)

        # WM-SAR
        wmsar_results = [wmsar_result(G) for G in G_list]
        wmsar_valid = [r for r in wmsar_results if r is not None]
        if wmsar_valid:
            summaries["WM-SAR"] = {
                "mean_rho_reduction": float(np.mean([r.rho_reduction for r in wmsar_valid])),
                "mean_region_size":   float(np.mean([r.region_size   for r in wmsar_valid])),
                "mean_mse_32":        float(np.mean([r.mse_profile_after.get(32, 0) for r in wmsar_valid])),
                "mean_slope":         float(np.mean([r.growth_slope_after for r in wmsar_valid])),
            }

        # Average ρ(B) before any repair
        rho_before = float(np.mean([
            amp.rho_B(G, set(G.nodes())) for G in G_list
        ]))

        output["results"][str(alpha)] = {
            "rho_before": rho_before,
            "summaries": {m: summaries[m] for m in KEY_METHODS if m in summaries},
        }

        # Quick print
        wmsar_rr = summaries.get("WM-SAR", {}).get("mean_rho_reduction", 0)
        best_eng = max(summaries.get(m, {}).get("mean_rho_reduction", 0)
                       for m in KEY_METHODS if m != "WM-SAR")
        print(f"ρ(B)={rho_before:.3f}  WM-SAR={wmsar_rr:.3f}  gap={wmsar_rr-best_eng:+.3f}  [{time.time()-t0:.1f}s]")

    # Print summary table
    print(f"\n{'='*65}")
    print(f"  {'α':>5}  {'ρ(B)':>6}  {'Greedy':>7}  {'LR-2H':>6}  {'LR-3H':>6}  {'WM-SAR':>7}  {'Gap':>6}")
    print(f"  {'-'*60}")
    for alpha in ALPHA_VALUES:
        r = output["results"][str(alpha)]
        s = r["summaries"]
        wmsar  = s.get("WM-SAR",              {}).get("mean_rho_reduction", 0)
        greedy = s.get("Greedy-Point(K=1)",   {}).get("mean_rho_reduction", 0)
        lr2    = s.get("LocalRepair-2Hop",    {}).get("mean_rho_reduction", 0)
        lr3    = s.get("LocalRepair-3Hop",    {}).get("mean_rho_reduction", 0)
        gap    = wmsar - max(greedy, lr2, lr3)
        print(f"  {alpha:>5.2f}  {r['rho_before']:>6.3f}  {greedy:>7.3f}  {lr2:>6.3f}  {lr3:>6.3f}  {wmsar:>7.3f}  {gap:>+6.3f}")

    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ Saved: {args.out}")


if __name__ == "__main__":
    main()
