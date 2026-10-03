from __future__ import annotations

import networkx as nx

from .failure_graph import node_error

REPAIR_STRENGTH = 0.9
PROP_GAIN = 0.85
ERR_EPS = 0.15
RECOVERY_FRAC = 0.4
RECOVERY_ABS = 0.5


def _topo(G: nx.DiGraph) -> list[str]:
    try:
        return list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:

        H = G.copy()
        while not nx.is_directed_acyclic_graph(H):
            cyc = nx.find_cycle(H)
            H.remove_edge(*cyc[0][:2])
        return list(nx.topological_sort(H))


def propagate_effective_error(
    G: nx.DiGraph, repaired: set[str], strength: float = REPAIR_STRENGTH
) -> dict[str, float]:

    if not 0 <= strength <= 1:
        raise ValueError("strength must be in [0, 1]")
    order = _topo(G)
    eff: dict[str, float] = {}
    for v in order:
        base = node_error(G, v)
        residual = base * (1.0 - strength) if v in repaired else base
        e = residual
        for u in G.predecessors(v):
            e = max(e, eff.get(u, 0.0) * PROP_GAIN)
        eff[v] = e
    return eff


def propagation_depth(G: nx.DiGraph, eff: dict[str, float]) -> int:

    active = {v for v, e in eff.items() if e > ERR_EPS}
    if not active:
        return 0
    sub = G.subgraph(active)
    if sub.number_of_nodes() == 0:
        return 0
    try:
        return int(nx.dag_longest_path_length(sub)) + 1
    except Exception:
        return len(active)


def local_inconsistency(G: nx.DiGraph, repaired: set[str], eff: dict[str, float]) -> int:


    count = 0
    for u, v in G.edges():
        ru, rv = u in repaired, v in repaired
        if ru != rv:
            other = v if ru else u
            if eff.get(other, 0.0) > ERR_EPS:
                count += 1
    return count


def region_iou(G: nx.DiGraph, repaired: set[str]) -> float:
    gt = set(G.graph.get("gt_region", set()))
    if not gt:
        return float("nan")
    inter = len(repaired & gt)
    union = len(repaired | gt)
    return inter / union if union else 0.0


def measure_recovery(G: nx.DiGraph, repaired: set[str]) -> dict:

    t_star = G.graph["t_star"]
    eff_before = propagate_effective_error(G, set())
    eff_after = propagate_effective_error(G, set(repaired))

    fb = eff_before[t_star]
    fa = eff_after[t_star]
    recovered = (fa <= RECOVERY_FRAC * fb) and (fa <= RECOVERY_ABS)

    down_before = sum(e for v, e in eff_before.items() if v != t_star)
    down_after = sum(e for v, e in eff_after.items() if v != t_star)

    return {
        "recovered": bool(recovered),
        "final_err_before": float(fb),
        "final_err_after": float(fa),
        "final_err_reduction": float(fb - fa),
        "downstream_err_before": float(down_before),
        "downstream_err_after": float(down_after),
        "downstream_err_reduction": float(down_before - down_after),
        "pd_before": propagation_depth(G, eff_before),
        "pd_after": propagation_depth(G, eff_after),
        "pd_reduction": propagation_depth(G, eff_before) - propagation_depth(G, eff_after),
        "local_inconsistency": local_inconsistency(G, set(repaired), eff_after),
        "region_iou": region_iou(G, set(repaired)),
        "region_size": len(repaired),
    }


def apply_repair(G: nx.DiGraph, repaired: set[str],
                 strength: float = REPAIR_STRENGTH) -> nx.DiGraph:


    eff = propagate_effective_error(G, set(repaired), strength)
    H = G.copy()
    for v in H.nodes():
        H.nodes[v]["err"] = float(eff[v])
    return H
