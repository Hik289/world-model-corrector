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


def _evaluate_repair(G: nx.DiGraph, selected: set, method: str, H_max: int = 32,
                     weight_norm: float = 0.9, gamma: float = 0.95) -> RepairResult:

    if H_max < 1:
        raise ValueError("H_max must be at least 1")
    selected = set(selected)
    if not selected.issubset(G.nodes()):
        raise ValueError("repair region contains nodes outside the graph")

    all_nodes = set(G.nodes())
    rho_before = amp.rho_B(G, all_nodes, weight_norm=weight_norm)
    rho_after = amp.rho_B_complement(G, selected, weight_norm=weight_norm)


    mse_before = amp.simulate_error_propagation(G, repaired=set(), H=H_max,
                                              weight_norm=weight_norm)
    mse_after = amp.simulate_error_propagation(G, repaired=selected, H=H_max,
                                             weight_norm=weight_norm)

    slope_before = amp.error_growth_slope(mse_before, h_start=4, h_end=H_max)
    slope_after = amp.error_growth_slope(mse_after, h_start=4, h_end=H_max)


    rb = amp.return_error_bound(G, selected, H=H_max, gamma=gamma,
                                weight_norm=weight_norm)


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


def greedy_point(G: nx.DiGraph, K: int = 1, **evaluation) -> RepairResult:


    ranked = sorted(G.nodes(), key=lambda v: node_error(G, v), reverse=True)
    selected = set(ranked[:K])
    return _evaluate_repair(G, selected, f"Greedy-Point(K={K})", **evaluation)


def topk_point(G: nx.DiGraph, K: int = 3, **evaluation) -> RepairResult:


    ranked = sorted(G.nodes(), key=lambda v: node_error(G, v), reverse=True)
    selected = set(ranked[:K])
    return _evaluate_repair(G, selected, f"TopK-Point(K={K})", **evaluation)


def window_repair(G: nx.DiGraph, window: int = 4, **evaluation) -> RepairResult:


    nodes_by_time = sorted(G.nodes(),
                            key=lambda v: G.nodes[v].get("time_step", 0))
    if len(nodes_by_time) <= window:
        selected = set(nodes_by_time)
        return _evaluate_repair(G, selected, f"Window-{window}-Point", **evaluation)


    best_win, best_score = [], float("-inf")
    for i in range(len(nodes_by_time) - window + 1):
        w = nodes_by_time[i: i + window]
        score = np.mean([node_error(G, v) for v in w])
        if score > best_score:
            best_score, best_win = score, w

    selected = set(best_win)
    return _evaluate_repair(G, selected, f"Window-{window}-Point", **evaluation)


def local_khop(G: nx.DiGraph, k: int = 2, **evaluation) -> RepairResult:


    if not G:
        return _evaluate_repair(G, set(), f"LocalRepair-{k}Hop", **evaluation)
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
    return _evaluate_repair(G, region, f"LocalRepair-{k}Hop", **evaluation)


def cascade_repair(G: nx.DiGraph, err_threshold: float = 0.3,
                   max_nodes: int = 15, **evaluation) -> RepairResult:


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

    return _evaluate_repair(G, selected, "CascadeRepair", **evaluation)


def oracle_region(G: nx.DiGraph, **evaluation) -> RepairResult:

    gt = G.graph.get("gt_region", set())
    return _evaluate_repair(G, set(gt), "Oracle", **evaluation)


def wmsar_repair(G: nx.DiGraph, cfg: WMSARConfig | None = None,
                 **evaluation) -> RepairResult:

    if cfg is None:
        cfg = WMSARConfig(weight_norm=evaluation.get("weight_norm", 0.9),
                          gamma=evaluation.get("gamma", 0.95))
    for key in ("weight_norm", "gamma"):
        if key in evaluation and evaluation[key] != getattr(cfg, key):
            raise ValueError(f"{key} differs between selection and evaluation")
    extractor = WMSAR(cfg)
    region = extractor.repair_region(G)
    evaluation.setdefault("weight_norm", cfg.weight_norm)
    evaluation.setdefault("gamma", cfg.gamma)
    return _evaluate_repair(G, region, "WM-SAR", **evaluation)


ALL_BASELINES: dict[str, Callable] = {
    "Greedy-Point(K=1)":  lambda G, **ev: greedy_point(G, K=1, **ev),
    "TopK-Point(K=3)":    lambda G, **ev: topk_point(G, K=3, **ev),
    "TopK-Point(K=5)":    lambda G, **ev: topk_point(G, K=5, **ev),
    "Window-2-Point":     lambda G, **ev: window_repair(G, window=2, **ev),
    "Window-4-Point":     lambda G, **ev: window_repair(G, window=4, **ev),
    "Window-8-Point":     lambda G, **ev: window_repair(G, window=8, **ev),
    "LocalRepair-2Hop":   lambda G, **ev: local_khop(G, k=2, **ev),
    "LocalRepair-3Hop":   lambda G, **ev: local_khop(G, k=3, **ev),
    "CascadeRepair":      lambda G, **ev: cascade_repair(G, **ev),
    "Oracle":             lambda G, **ev: oracle_region(G, **ev),
    "WM-SAR":             lambda G, **ev: wmsar_repair(G, **ev),
}


def _aggregate(results: list[RepairResult]) -> dict:

    if not results:
        return {}

    def m(vals): return float(np.mean(vals))
    def s(vals): return float(np.std(vals))


    available = set.intersection(*[
        set(r.mse_profile_before) & set(r.mse_profile_after) for r in results
    ])
    requested = {1, 2, 4, 8, 16, 32}
    if available:
        requested.add(max(available))
    horizons = sorted(available & requested)
    mse_before = {H: m([r.mse_profile_before[H] for r in results]) for H in horizons}
    mse_after = {H: m([r.mse_profile_after[H] for r in results]) for H in horizons}
    mse_reduction = {H: mse_before[H] - mse_after[H] for H in horizons}

    return {
        "n": len(results),
        "method": results[0].method if results else "",

        "mean_region_size": m([r.region_size for r in results]),
        "std_region_size": s([r.region_size for r in results]),
        "frac_connected": m([float(r.is_connected) for r in results]),
        "mean_err_cover": m([r.err_cover for r in results]),
        "mean_iou": m([r.iou_vs_gt for r in results]),
        "std_iou": s([r.iou_vs_gt for r in results]),

        "mean_rho_before": m([r.rho_before for r in results]),
        "mean_rho_after": m([r.rho_after_region for r in results]),
        "mean_rho_reduction": m([r.rho_reduction for r in results]),
        "std_rho_reduction": s([r.rho_reduction for r in results]),

        "NodeMSE_before": mse_before,
        "NodeMSE_after": mse_after,
        "NodeMSE_before_std": {H: s([r.mse_profile_before[H] for r in results])
                               for H in horizons},
        "NodeMSE_after_std": {H: s([r.mse_profile_after[H] for r in results])
                              for H in horizons},
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
                      verbose: bool = True, H_max: int = 32,
                      weight_norm: float = 0.9,
                      gamma: float = 0.95) -> dict[str, dict]:

    if H_max < 1:
        raise ValueError("H_max must be at least 1")
    custom_methods = methods is not None
    methods = ALL_BASELINES if methods is None else methods
    evaluation = dict(H_max=H_max, weight_norm=weight_norm, gamma=gamma)
    summaries = {}
    for name, fn in methods.items():
        results = []
        for G in G_list:
            try:
                if custom_methods:
                    selected = fn(G).selected_nodes
                    result = _evaluate_repair(G, selected, name, **evaluation)
                else:
                    result = fn(G, **evaluation)
                results.append(result)
            except Exception as e:
                raise RuntimeError(f"{name} failed on graph {len(results)}") from e
        summaries[name] = _aggregate(results)
        if verbose:
            s = summaries[name]
            rho_r = s.get("mean_rho_reduction", 0.0)
            mse_last = s.get("NodeMSE_after", {}).get(H_max, float("nan"))
            slope_a = s.get("mean_growth_slope_after", float("nan"))
            conn = s.get("frac_connected", 0.0)
            iou = s.get("mean_iou", 0.0)
            print(f"  {name:<28}  ρ_red={rho_r:.4f}  "
                  f"MSE@{H_max}={mse_last:.4f}  slope={slope_a:.4f}  "
                  f"conn={conn:.2f}  IoU={iou:.3f}")
    return summaries
