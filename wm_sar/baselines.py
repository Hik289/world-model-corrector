"""Repair baselines.

All "LLM" repairers are **simulated**: given a context window, the repairer
picks the highest-error item(s) inside that window (no real LLM call). The whole
point of the experiment is that short context windows are centered on the
*visible* failure, so they miss the root cause that lies earlier in the
target-reachable cone.

Every baseline returns a :class:`RepairPlan` carrying the repaired node set, the
simulated token cost, the number of pointwise edits, and a latency estimate.

Token-cost model (Section 11.3, simulated; base_tokens = 50/node):
    TraceScan-w1 :  1*1  * base  per edit
    TraceScan-w2 :  2*2  * base  per edit
    TraceScan-w4 :  4*4  * base  per edit
    TraceScan-Full        : |V| * base       per edit
    LLMRepair-Full-Plan   : |V| * base * 3    (single suffix-replan call)
    WM-SAR  : |region| * base + 0.2*|V|*base  (scored once, one subgraph)
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from . import amplification as amp
from .failure_graph import node_error, node_unc
from .region_extractor import WMSAR, WMSARConfig

BASE_TOKENS = 50
LATENCY_PER_KTOK = 0.8     # seconds per 1000 tokens
LATENCY_PER_EDIT = 0.05    # per pointwise diagnosis round-trip


@dataclass
class RepairPlan:
    method: str
    nodes: set[str]
    token_cost: float
    n_edits: int
    is_subgraph: bool
    latency: float
    window: int | None = None


def _latency(token_cost: float, n_edits: int) -> float:
    return token_cost / 1000.0 * LATENCY_PER_KTOK + n_edits * LATENCY_PER_EDIT


def _failure_time(G: nx.DiGraph) -> int:
    return int(G.nodes[G.graph["t_star"]].get("time_step", 0))


def _window_nodes(G: nx.DiGraph, w: int) -> list[str]:
    """Trace items within w steps of the *visible* failure (Section 11.1)."""
    ft = _failure_time(G)
    return [v for v, d in G.nodes(data=True)
            if abs(int(d.get("time_step", 0)) - ft) <= w]


def _topk_by(G: nx.DiGraph, key, k: int, pool=None) -> list[str]:
    pool = pool if pool is not None else list(G.nodes())
    pool = [v for v in pool if v != G.graph.get("t_star")]
    return [v for v in sorted(pool, key=key, reverse=True)[:k]]


# ---------------------------------------------------------------------------
# Context-limited pointwise scanners
# ---------------------------------------------------------------------------


def trace_scan_window(G: nx.DiGraph, w: int, budget: int = 4) -> RepairPlan:
    pool = _window_nodes(G, w)
    picks = _topk_by(G, lambda v: node_error(G, v), budget, pool)
    per_edit = (w * w) * BASE_TOKENS if w > 0 else BASE_TOKENS
    n_edits = max(1, len(picks))
    cost = per_edit * n_edits
    return RepairPlan(f"TraceScan-w{w}-Point", set(picks), cost, n_edits,
                      is_subgraph=False, latency=_latency(cost, n_edits), window=w)


def trace_scan_full(G: nx.DiGraph, budget: int = 4) -> RepairPlan:
    picks = _topk_by(G, lambda v: node_error(G, v), budget)
    per_edit = G.number_of_nodes() * BASE_TOKENS
    n_edits = max(1, len(picks))
    cost = per_edit * n_edits
    return RepairPlan("TraceScan-Full-Point", set(picks), cost, n_edits,
                      is_subgraph=False, latency=_latency(cost, n_edits), window=None)


def llm_repair_full_plan(G: nx.DiGraph) -> RepairPlan:
    """Sees the full trace and rewrites the whole target-reachable suffix.
    High-cost upper baseline (single big call)."""
    reach = amp.target_reachable(G)
    cost = G.number_of_nodes() * BASE_TOKENS * 3
    return RepairPlan("LLMRepair-Full-Plan", set(reach), cost, 1,
                      is_subgraph=True, latency=_latency(cost, 1), window=None)


# ---------------------------------------------------------------------------
# Simple pointwise heuristics
# ---------------------------------------------------------------------------


def last_error_point(G: nx.DiGraph) -> RepairPlan:
    ft = _failure_time(G)
    cand = [v for v, d in G.nodes(data=True)
            if int(d.get("time_step", 0)) == ft - 1] or list(G.nodes())
    pick = max(cand, key=lambda v: node_error(G, v))
    cost = BASE_TOKENS
    return RepairPlan("LastError-Point", {pick}, cost, 1, False, _latency(cost, 1), 1)


def first_failed_call_point(G: nx.DiGraph) -> RepairPlan:
    failed = [(d.get("time_step", 0), v) for v, d in G.nodes(data=True)
              if d.get("status") in ("failed", "degraded") or node_error(G, v) > 0.3]
    if failed:
        pick = min(failed)[1]
    else:
        pick = max(G.nodes(), key=lambda v: node_error(G, v))
    cost = BASE_TOKENS
    return RepairPlan("FirstFailedCall-Point", {pick}, cost, 1, False, _latency(cost, 1), 1)


def rule_scanner_point(G: nx.DiGraph) -> RepairPlan:
    cand = [v for v, d in G.nodes(data=True)
            if d.get("contradicts_observation") or d.get("status") in ("failed", "degraded")]
    cand = cand or list(G.nodes())
    pick = max(cand, key=lambda v: node_error(G, v))
    cost = BASE_TOKENS
    return RepairPlan("RuleScanner-Point", {pick}, cost, 1, False, _latency(cost, 1), 1)


# ---------------------------------------------------------------------------
# Pointwise graph repair (Top-B node/edge style)
# ---------------------------------------------------------------------------


def top_b_nodes(G: nx.DiGraph, budget: int = 4) -> RepairPlan:
    picks = _topk_by(G, lambda v: node_error(G, v), budget)
    cost = budget * 2 * BASE_TOKENS
    return RepairPlan("Top-B-Nodes", set(picks), cost, len(picks), False,
                      _latency(cost, len(picks)))


def top_b_edges(G: nx.DiGraph, budget: int = 4) -> RepairPlan:
    edges = sorted(G.edges(), key=lambda e: node_error(G, e[0]) + node_error(G, e[1]),
                   reverse=True)[:budget]
    nodes = set()
    for u, v in edges:
        nodes.update([u, v])
    nodes.discard(G.graph.get("t_star"))
    cost = budget * 2 * BASE_TOKENS
    return RepairPlan("Top-B-Edges", nodes, cost, len(edges), False,
                      _latency(cost, len(edges)))


def top_b_mixed(G: nx.DiGraph, budget: int = 4) -> RepairPlan:
    n = top_b_nodes(G, budget // 2 + 1).nodes
    e = top_b_edges(G, budget // 2 + 1).nodes
    nodes = (n | e)
    cost = budget * 2 * BASE_TOKENS
    return RepairPlan("Top-B-Mixed", nodes, cost, budget, False, _latency(cost, budget))


def degree_repair(G: nx.DiGraph, budget: int = 4) -> RepairPlan:
    deg = dict(G.degree())
    picks = _topk_by(G, lambda v: deg.get(v, 0), budget)
    cost = budget * 2 * BASE_TOKENS
    return RepairPlan("DegreeRepair", set(picks), cost, len(picks), False,
                      _latency(cost, len(picks)))


def pagerank_repair(G: nx.DiGraph, budget: int = 4) -> RepairPlan:
    pr = nx.pagerank(G) if G.number_of_edges() else {v: 0 for v in G}
    picks = _topk_by(G, lambda v: pr.get(v, 0), budget)
    cost = budget * 2 * BASE_TOKENS
    return RepairPlan("PageRankRepair", set(picks), cost, len(picks), False,
                      _latency(cost, len(picks)))


def uncertainty_repair(G: nx.DiGraph, budget: int = 4) -> RepairPlan:
    picks = _topk_by(G, lambda v: node_unc(G, v), budget)
    cost = budget * 2 * BASE_TOKENS
    return RepairPlan("UncertaintyRepair", set(picks), cost, len(picks), False,
                      _latency(cost, len(picks)))


def target_cone_repair(G: nx.DiGraph, budget: int = 4) -> RepairPlan:
    reach = amp.target_reachable(G)
    picks = _topk_by(G, lambda v: node_error(G, v), budget, pool=list(reach))
    cost = budget * 2 * BASE_TOKENS
    return RepairPlan("TargetConeRepair", set(picks), cost, len(picks), False,
                      _latency(cost, len(picks)))


# ---------------------------------------------------------------------------
# Subgraph heuristics
# ---------------------------------------------------------------------------


def _khop_ball(G: nx.DiGraph, center: str, k: int) -> set[str]:
    und = G.to_undirected(as_view=True)
    ball = nx.single_source_shortest_path_length(und, center, cutoff=k)
    return set(ball.keys())


def khop_last_error(G: nx.DiGraph, k: int = 2) -> RepairPlan:
    ft = _failure_time(G)
    cand = [v for v, d in G.nodes(data=True) if int(d.get("time_step", 0)) == ft - 1]
    center = max(cand or list(G.nodes()), key=lambda v: node_error(G, v))
    nodes = _khop_ball(G, center, k)
    nodes.discard(G.graph.get("t_star"))
    cost = len(nodes) * BASE_TOKENS
    return RepairPlan(f"kHop-LastError(k={k})", nodes, cost, 1, True, _latency(cost, 1))


def khop_first_failed(G: nx.DiGraph, k: int = 2) -> RepairPlan:
    center = first_failed_call_point(G).nodes.pop()
    nodes = _khop_ball(G, center, k)
    nodes.discard(G.graph.get("t_star"))
    cost = len(nodes) * BASE_TOKENS
    return RepairPlan(f"kHop-FirstFailed(k={k})", nodes, cost, 1, True, _latency(cost, 1))


def pagerank_subgraph(G: nx.DiGraph, budget: int = 6) -> RepairPlan:
    pr = nx.pagerank(G) if G.number_of_edges() else {v: 0 for v in G}
    seeds = _topk_by(G, lambda v: pr.get(v, 0), 2)
    nodes = set()
    for s in seeds:
        nodes |= _khop_ball(G, s, 1)
    nodes.discard(G.graph.get("t_star"))
    cost = len(nodes) * BASE_TOKENS
    return RepairPlan("PageRank-Subgraph", nodes, cost, 1, True, _latency(cost, 1))


def uncertainty_subgraph(G: nx.DiGraph, budget: int = 6) -> RepairPlan:
    seeds = _topk_by(G, lambda v: node_unc(G, v), 2)
    nodes = set()
    for s in seeds:
        nodes |= _khop_ball(G, s, 1)
    nodes.discard(G.graph.get("t_star"))
    cost = len(nodes) * BASE_TOKENS
    return RepairPlan("Uncertainty-Subgraph", nodes, cost, 1, True, _latency(cost, 1))


def target_cone_subgraph(G: nx.DiGraph) -> RepairPlan:
    nodes = set(amp.target_reachable(G))
    nodes.discard(G.graph.get("t_star"))
    cost = len(nodes) * BASE_TOKENS
    return RepairPlan("TargetCone-Subgraph", nodes, cost, 1, True, _latency(cost, 1))


# ---------------------------------------------------------------------------
# Upper bounds
# ---------------------------------------------------------------------------


def oracle_region(G: nx.DiGraph) -> RepairPlan:
    nodes = set(G.graph.get("gt_region", set()))
    nodes.discard(G.graph.get("t_star"))
    cost = len(nodes) * BASE_TOKENS
    return RepairPlan("OracleRegion", nodes, cost, 1, True, _latency(cost, 1))


def full_replan(G: nx.DiGraph) -> RepairPlan:
    nodes = set(G.nodes())
    nodes.discard(G.graph.get("t_star"))
    cost = G.number_of_nodes() * BASE_TOKENS * 3
    return RepairPlan("FullReplan", nodes, cost, 1, True, _latency(cost, 1))


# ---------------------------------------------------------------------------
# WM-SAR (the proposed method)
# ---------------------------------------------------------------------------


def wm_sar(G: nx.DiGraph, config: WMSARConfig | None = None,
           budget: float = 14.0, short_prompt: bool = False) -> RepairPlan:
    extractor = WMSAR(config)
    region = extractor.repair_region(G, budget)
    region.discard(G.graph.get("t_star"))
    scoring_overhead = 0.2 * G.number_of_nodes() * BASE_TOKENS
    region_cost = len(region) * BASE_TOKENS
    cost = region_cost + scoring_overhead
    if short_prompt:
        cost += 2 * BASE_TOKENS    # compact region-summary repair prompt
    name = "WM-SAR+ShortPrompt" if short_prompt else "WM-SAR"
    return RepairPlan(name, region, cost, 1, True, _latency(cost, 1))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def all_baselines(G: nx.DiGraph, budget: int = 4) -> dict[str, RepairPlan]:
    """Run the full baseline suite (excluding WM-SAR) on a graph."""
    plans = [
        last_error_point(G),
        first_failed_call_point(G),
        rule_scanner_point(G),
        trace_scan_window(G, 1, budget),
        trace_scan_window(G, 2, budget),
        trace_scan_window(G, 4, budget),
        trace_scan_full(G, budget),
        llm_repair_full_plan(G),
        top_b_nodes(G, budget),
        top_b_edges(G, budget),
        top_b_mixed(G, budget),
        degree_repair(G, budget),
        pagerank_repair(G, budget),
        uncertainty_repair(G, budget),
        target_cone_repair(G, budget),
        khop_last_error(G, 2),
        khop_first_failed(G, 2),
        pagerank_subgraph(G),
        uncertainty_subgraph(G),
        target_cone_subgraph(G),
        oracle_region(G),
        full_replan(G),
    ]
    return {p.method: p for p in plans}
