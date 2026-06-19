"""Experiment: Budget Sensitivity.

Fix the repair budget K_max ∈ {2, 3, 5, 7, 8, 10, 12, 15, 20}
(n=50, seed=42) and measure ρ(B)-reduction for each method.

Hypothesis:
  WM-SAR reaches near-Oracle ρ-reduction at K≈8 (matching our observation
  of mean_region_size=8.2 in the main experiment).
  Engineering methods need K≈25 (full graph) to match WM-SAR.

Key insight: WM-SAR is *budget-efficient* — it extracts maximum spectral
reduction per node added.
"""

import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wm_sar.agent_calling_tree import generate_calling_trees
from wm_sar import amplification as amp
from wm_sar.engineering_baselines import _evaluate_repair
from wm_sar.region_extractor import WMSAR, WMSARConfig
from wm_sar.baselines import wm_sar as _wm_sar_fn
def wm_sar_default(G):
    r = _wm_sar_fn(G)
    return r.nodes if hasattr(r, "nodes") else set(r)


K_VALUES = [2, 3, 5, 7, 8, 10, 12, 15, 20]


def greedy_topk(G, k):
    """Greedy-Point: top-k nodes by error."""
    nodes = sorted(G.nodes(), key=lambda v: amp.node_error(G, v), reverse=True)
    return set(nodes[:k])


def window_k(G, k):
    """Window-k: k consecutive steps sorted by time_step."""
    order = sorted(G.nodes(), key=lambda v: G.nodes[v].get("time_step", 0))
    errors = {v: amp.node_error(G, v) for v in order}
    # Sliding window of size k, pick the window with highest total error
    best_region, best_err = set(), -1
    for i in range(len(order)):
        window = set(order[i:i+k])
        total  = sum(errors.get(v, 0) for v in window)
        if total > best_err:
            best_err = total
            best_region = window
    return best_region


def local_khop(G, k_budget):
    """Local-kHop: expand from highest-error node until k_budget nodes."""
    root = max(G.nodes(), key=lambda v: amp.node_error(G, v))
    region = {root}
    import networkx as nx
    frontier = set(nx.neighbors(G.to_undirected(), root))
    while len(region) < k_budget and frontier:
        best = max(frontier, key=lambda v: amp.node_error(G, v))
        region.add(best)
        frontier.discard(best)
        for nb in nx.neighbors(G.to_undirected(), best):
            if nb not in region:
                frontier.add(nb)
    return region


def wmsar_with_budget(G, k_budget):
    """WM-SAR with explicit K_max budget."""
    cfg = WMSARConfig(max_region_size=k_budget, n_seeds=min(3, k_budget))
    extractor = WMSAR(cfg)
    return extractor.repair_region(G)


def oracle_k(G, k_budget):
    """Oracle: ground-truth region, capped at k_budget nodes."""
    gt = G.graph.get("gt_region", set())
    if len(gt) <= k_budget:
        return gt
    # Take top-k from GT by error
    return set(sorted(gt, key=lambda v: amp.node_error(G, v), reverse=True)[:k_budget])


def evaluate(G, region, method):
    if not region:
        return {"rho_reduction": 0.0, "region_size": 0}
    try:
        r = _evaluate_repair(G, region, method)
        return {
            "rho_reduction": r.rho_reduction,
            "region_size":   r.region_size,
            "mse_32":        r.mse_profile_after.get(32, 0.0),
        }
    except Exception:
        return {"rho_reduction": 0.0, "region_size": len(region)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",    type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out",  type=str,
                    default=os.path.join(os.path.dirname(__file__),
                                         "results", "exp_budget.json"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"\n{'='*60}\n  Budget Sensitivity (n={args.n})\n{'='*60}")

    trees  = generate_calling_trees(n=args.n, seed=args.seed)
    G_list = [t.G for t in trees]

    # Oracle at full budget (no cap)
    oracle_full = float(np.mean([
        _evaluate_repair(G, G.graph.get("gt_region", set()), "Oracle").rho_reduction
        for G in G_list
    ]))
    # WM-SAR at default budget
    wmsar_full = float(np.mean([
        _evaluate_repair(G, wm_sar_default(G), "WM-SAR").rho_reduction
        for G in G_list
    ]))
    print(f"  Reference: Oracle(full)={oracle_full:.3f}  WM-SAR(default)={wmsar_full:.3f}")

    output = {
        "K_values": K_VALUES,
        "n": args.n,
        "seed": args.seed,
        "oracle_full": oracle_full,
        "wmsar_default": wmsar_full,
        "results": {}
    }

    SELECTORS = {
        "Greedy-TopK":   greedy_topk,
        "Window-K":      window_k,
        "Local-KHop":    local_khop,
        "WM-SAR":        wmsar_with_budget,
        "Oracle-K":      oracle_k,
    }

    for K in K_VALUES:
        print(f"\n  K={K} ...", end=" ", flush=True)
        t0 = time.time()
        K_results = {}
        for mname, selector_fn in SELECTORS.items():
            rr_list = []
            for G in G_list:
                try:
                    region = selector_fn(G, K)
                    res    = evaluate(G, region, mname)
                    rr_list.append(res["rho_reduction"])
                except Exception:
                    pass
            K_results[mname] = {
                "mean_rho_reduction": float(np.mean(rr_list)) if rr_list else 0.0,
                "std_rho_reduction":  float(np.std(rr_list))  if rr_list else 0.0,
            }

        output["results"][str(K)] = K_results
        wmsar_r = K_results["WM-SAR"]["mean_rho_reduction"]
        greedy_r = K_results["Greedy-TopK"]["mean_rho_reduction"]
        local_r  = K_results["Local-KHop"]["mean_rho_reduction"]
        pct_oracle = 100 * wmsar_r / oracle_full if oracle_full > 0 else 0
        print(f"Greedy={greedy_r:.3f}  Local={local_r:.3f}  WM-SAR={wmsar_r:.3f} ({pct_oracle:.0f}% of Oracle)  [{time.time()-t0:.1f}s]")

    # Print summary
    print(f"\n{'='*70}")
    print(f"{'K':>4} {'Greedy':>8} {'Window':>8} {'Local':>8} {'WM-SAR':>8} {'Oracle':>8} {'WM-SAR%':>8}")
    print(f"  {'-'*62}")
    for K in K_VALUES:
        r = output["results"][str(K)]
        g = r["Greedy-TopK"]["mean_rho_reduction"]
        w = r["Window-K"]["mean_rho_reduction"]
        l = r["Local-KHop"]["mean_rho_reduction"]
        s = r["WM-SAR"]["mean_rho_reduction"]
        o = r["Oracle-K"]["mean_rho_reduction"]
        pct = 100 * s / oracle_full if oracle_full > 0 else 0
        print(f"{K:>4} {g:>8.3f} {w:>8.3f} {l:>8.3f} {s:>8.3f} {o:>8.3f} {pct:>7.0f}%")

    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ Saved: {args.out}")


if __name__ == "__main__":
    main()
