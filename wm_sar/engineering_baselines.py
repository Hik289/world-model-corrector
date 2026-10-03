from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import networkx as nx
import numpy as np

from . import amplification as amp
from .failure_graph import node_error
from .region_extractor import WMSAR, WMSARConfig


@dataclass
class RepairResult:
    method: str
    selected_nodes: set
    is_connected: bool
    err_cover: float
    region_size: int
    rho_before: float
    rho_after_region: float
    rho_reduction: float
    mse_profile_before: dict
    mse_profile_after: dict
    growth_slope_before: float
    growth_slope_after: float
    iou_vs_gt: float

    return_bound_before: float
    return_bound_after: float
    regret_reduction: float


def _evaluate_repair(G: nx.DiGraph, selected: set, method: str, H_max: int = 32) -> RepairResult:


    all_nodes = set(G.nodes())
    rho_before = amp.rho_B(G, all_nodes, weight_norm=1.0)
    rho_after = amp.rho_B_complement(G, selected, weight_norm=1.0)


    mse_before = amp.simulate_error_propagation(G, repaired=set(), H=H_max)
    mse_after = amp.simulate_error_propagation(G, repaired=selected, H=H_max)

    slope_before = amp.error_growth_slope(mse_before, h_start=4, h_end=H_max)
    slope_after = amp.error_growth_slope(mse_after, h_start=4, h_end=H_max)


    rb = amp.return_error_bound(G, selected, H=H_max, gamma=0.95)


    gt = G.graph.get("gt_region", set())
    inter = len(selected & gt)
    union = len(selected | gt)
    iou = inter / union if union > 0 else 0.0


    if len(selected) <= 1:
        connected = True
    else:
        sub = G.subgraph(selected).to_undirected()
        connected = nx.is_connected(sub)

    return RepairResult(
        method=method,
        selected_nodes=selected,
        is_connected=connected,
        err_cover=sum(node_error(G, v) for v in selected),
        region_size=len(selected),
        rho_before=rho_before,
        rho_after_region=rho_after,
        rho_reduction=rho_before - rho_after,
        mse_profile_before=mse_before,
        mse_profile_after=mse_after,
        growth_slope_before=slope_before,
        growth_slope_after=slope_after,
        iou_vs_gt=iou,
        return_bound_before=rb["bound_pre"],
        return_bound_after=rb["bound_post"],
        regret_reduction=rb["regret_reduction"],
    )


def greedy_point(G: nx.DiGraph, K: int = 1) -> RepairResult:


    ranked = sorted(G.nodes(), key=lambda v: node_error(G, v), reverse=True)
    selected = set(ranked[:K])
    return _evaluate_repair(G, selected, f"Greedy-Point(K={K})")


def topk_point(G: nx.DiGraph, K: int = 3) -> RepairResult:


    ranked = sorted(G.nodes(), key=lambda v: node_error(G, v), reverse=True)
    selected = set(ranked[:K])
    return _evaluate_repair(G, selected, f"TopK-Point(K={K})")


def window_repair(G: nx.DiGraph, window: int = 4) -> RepairResult:


    nodes_by_time = sorted(G.nodes(),
                            key=lambda v: G.nodes[v].get("time_step", 0))
    if len(nodes_by_time) <= window:
        selected = set(nodes_by_time)
        return _evaluate_repair(G, selected, f"Window-{window}-Point")


    best_win, best_score = [], 0.0
    for i in range(len(nodes_by_time) - window + 1):
        w = nodes_by_time[i: i + window]
        score = np.mean([node_error(G, v) for v in w])
        if score > best_score:
            best_score, best_win = score, w

    selected = set(best_win)
    return _evaluate_repair(G, selected, f"Window-{window}-Point")


def local_khop(G: nx.DiGraph, k: int = 2) -> RepairResult:


    source = max(G.nodes(), key=lambda v: node_error(G, v))

    und = G.to_undirected(as_view=True)
    region = {source}
    frontier = {source}
    for _ in range(k):
        nxt = set()
        for u in frontier:
            nxt.update(und.neighbors(u))
        frontier = nxt - region
        region |= frontier
    return _evaluate_repair(G, region, f"LocalRepair-{k}Hop")


def cascade_repair(G: nx.DiGraph, err_threshold: float = 0.3,
                   max_nodes: int = 15) -> RepairResult:


    try:
        topo = list(nx.topological_sort(G))
    except Exception:
        topo = sorted(G.nodes(), key=lambda v: G.nodes[v].get("time_step", 0))

    selected: set = set()
    total_err = sum(node_error(G, v) for v in G.nodes())

    for v in topo:
        if len(selected) >= max_nodes:
            break
        err_v = node_error(G, v)
        if err_v > err_threshold:
            selected.add(v)
            total_err -= err_v
        if total_err <= err_threshold * len(G.nodes()) * 0.2:
            break

    if not selected:

        ranked = sorted(G.nodes(), key=lambda v: node_error(G, v), reverse=True)
        selected = set(ranked[:3])

    return _evaluate_repair(G, selected, "CascadeRepair")


def oracle_region(G: nx.DiGraph) -> RepairResult:

    gt = G.graph.get("gt_region", set())
    if not gt:

        best = max(G.nodes(), key=lambda v: node_error(G, v))
        gt = {best}
    return _evaluate_repair(G, gt, "Oracle")


def wmsar_repair(G: nx.DiGraph, cfg: WMSARConfig | None = None) -> RepairResult:

    cfg = cfg or WMSARConfig()
    extractor = WMSAR(cfg)
    region = extractor.repair_region(G)
    return _evaluate_repair(G, region, "WM-SAR")


ALL_BASELINES: dict[str, Callable] = {
    "Greedy-Point(K=1)":  lambda G: greedy_point(G, K=1),
    "TopK-Point(K=3)":    lambda G: topk_point(G, K=3),
    "TopK-Point(K=5)":    lambda G: topk_point(G, K=5),
    "Window-2-Point":     lambda G: window_repair(G, window=2),
    "Window-4-Point":     lambda G: window_repair(G, window=4),
    "Window-8-Point":     lambda G: window_repair(G, window=8),
    "LocalRepair-2Hop":   lambda G: local_khop(G, k=2),
    "LocalRepair-3Hop":   lambda G: local_khop(G, k=3),
    "CascadeRepair":      lambda G: cascade_repair(G),
    "Oracle":             lambda G: oracle_region(G),
    "WM-SAR":             lambda G: wmsar_repair(G),
}


def _aggregate(results: list[RepairResult]) -> dict:

    if not results:
        return {}

    def m(vals): return float(np.mean(vals))
    def s(vals): return float(np.std(vals))


    horizons = [1, 2, 4, 8, 16, 32]
    mse_before = {H: m([r.mse_profile_before.get(H, 0.0) for r in results]) for H in horizons}
    mse_after  = {H: m([r.mse_profile_after.get(H, 0.0)  for r in results]) for H in horizons}
    mse_reduction = {H: mse_before[H] - mse_after[H] for H in horizons}

    return {
        "n": len(results),
        "method": results[0].method if results else "",

        "mean_region_size": m([r.region_size for r in results]),
        "frac_connected": m([float(r.is_connected) for r in results]),
        "mean_err_cover": m([r.err_cover for r in results]),
        "mean_iou": m([r.iou_vs_gt for r in results]),

        "mean_rho_before": m([r.rho_before for r in results]),
        "mean_rho_after": m([r.rho_after_region for r in results]),
        "mean_rho_reduction": m([r.rho_reduction for r in results]),

        "NodeMSE_before": mse_before,
        "NodeMSE_after": mse_after,
        "NodeMSE_reduction": mse_reduction,

        "mean_growth_slope_before": m([r.growth_slope_before for r in results]),
        "mean_growth_slope_after": m([r.growth_slope_after for r in results]),
        "growth_slope_reduction": m([r.growth_slope_before - r.growth_slope_after
                                      for r in results]),

        "mean_return_bound_before": m([r.return_bound_before for r in results]),
        "mean_return_bound_after": m([r.return_bound_after for r in results]),
        "mean_regret_reduction": m([r.regret_reduction for r in results]),
    }


def run_all_baselines(G_list: list[nx.DiGraph],
                      methods: dict | None = None,
                      verbose: bool = True) -> dict[str, dict]:

    methods = methods or ALL_BASELINES
    summaries = {}
    for name, fn in methods.items():
        results = []
        for G in G_list:
            try:
                results.append(fn(G))
            except Exception as e:
                if verbose:
                    print(f"  [{name}] error on graph: {e}")
        summaries[name] = _aggregate(results)
        if verbose:
            s = summaries[name]
            rho_r = s.get("mean_rho_reduction", 0.0)
            mse32 = s.get("NodeMSE_after", {}).get(32, float("nan"))
            slope_a = s.get("mean_growth_slope_after", float("nan"))
            conn = s.get("frac_connected", 0.0)
            iou = s.get("mean_iou", 0.0)
            print(f"  {name:<28}  ρ_red={rho_r:.4f}  "
                  f"MSE@32={mse32:.4f}  slope={slope_a:.4f}  "
                  f"conn={conn:.2f}  IoU={iou:.3f}")
    return summaries
