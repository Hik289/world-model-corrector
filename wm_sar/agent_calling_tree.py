"""Agent Calling-Tree Testbed.

Matches the heterogeneous agent-graph testbed described in analysis/main.tex §4.3:
    N ≈ 22–30 nodes
    9 node types:   planner, validator, executor, checker, aggregator,
                    reporter, logger, error_handler, final_answer
    6 edge types:   calls, validates, reports, errors, triggers, logs
    8-dim state vector per node (execution status features)
    Primary metric: sink-node success rate sr_sink

The testbed is used to evaluate multi-step planning error:
    NodeMSE@H   — prediction error at horizon H
    GrowthSlope — d(log e_k)/dk (contractive < 0, divergent > 0)
    ρ(B)        — coupled amplification
    ReturnError@H — planning regret from T4

Failure injection:
    A root-cause error is injected at a specific node (planner or executor).
    The error propagates through calls/validates/reports/triggers edges.
    Cascade gain: child error = parent error × α + ε  (α=1.1, ε~N(0,0.05))
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Node and edge type taxonomy
# ──────────────────────────────────────────────────────────────────────────────

NODE_TYPES = [
    "planner", "validator", "executor", "checker",
    "aggregator", "reporter", "logger", "error_handler", "final_answer",
]

EDGE_TYPES = ["calls", "validates", "reports", "errors", "triggers", "logs"]

# Structural roles and their typical connections
ROLE_CONNECTS = {
    "planner":       {"executor": "calls", "validator": "calls"},
    "executor":      {"checker": "triggers", "aggregator": "reports"},
    "validator":     {"executor": "validates", "checker": "validates"},
    "checker":       {"aggregator": "reports", "error_handler": "errors"},
    "aggregator":    {"reporter": "reports", "final_answer": "triggers"},
    "reporter":      {"logger": "logs", "final_answer": "reports"},
    "logger":        {"error_handler": "logs"},
    "error_handler": {"planner": "triggers"},
    "final_answer":  {},
}

# State feature indices (8-dim)
FEAT_DIM = 8
FEAT_NAMES = [
    "activation",   # 0: is this node currently active
    "load",         # 1: computational load (0-1)
    "latency",      # 2: response latency
    "error_prob",   # 3: probability of producing wrong output
    "throughput",   # 4: processed items / capacity
    "confidence",   # 5: self-reported confidence
    "dependency_ok",# 6: all upstream deps satisfied
    "success_flag", # 7: did this node succeed
]


# ──────────────────────────────────────────────────────────────────────────────
# Graph instance
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentCallTree:
    G: nx.DiGraph
    node_states: np.ndarray    # shape (N_nodes, 8)
    node_list: list            # ordered node list matching node_states
    root_cause_node: str
    failure_desc: str
    true_error: np.ndarray     # ground-truth error per node (post-cascade)
    horizon_mse: dict          # {H: NodeMSE@H} simulated from theory
    rollout_trajectory: list   # list of (step, node, state_before, state_after)


# ──────────────────────────────────────────────────────────────────────────────
# Generator
# ──────────────────────────────────────────────────────────────────────────────

def _random_state(rng: np.random.Generator, node_type: str) -> np.ndarray:
    """8-dim state vector for a healthy node."""
    s = np.zeros(FEAT_DIM)
    s[0] = 1.0                              # activation: on
    s[1] = rng.uniform(0.1, 0.5)           # load
    s[2] = rng.uniform(0.05, 0.3)          # latency
    s[3] = rng.uniform(0.0, 0.05)          # error_prob (healthy)
    s[4] = rng.uniform(0.7, 1.0)           # throughput
    s[5] = rng.uniform(0.8, 1.0)           # confidence
    s[6] = 1.0                              # dependency_ok
    s[7] = 1.0                              # success_flag
    if node_type == "error_handler":
        s[3] = rng.uniform(0.1, 0.3)       # error_handler has higher error_prob
    if node_type == "final_answer":
        s[5] = rng.uniform(0.5, 0.9)       # final_answer less confident
    return s


def _inject_failure(state: np.ndarray, failure_type: str, rng: np.random.Generator,
                    magnitude: float = 0.5) -> np.ndarray:
    """Corrupt a node state to simulate a failure."""
    s = state.copy()
    if failure_type == "prediction_drift":
        s[4] -= magnitude * rng.uniform(0.5, 1.0)   # throughput drops
        s[3] += magnitude * rng.uniform(0.3, 0.7)   # error_prob rises
    elif failure_type == "tool_misfire":
        s[5] = rng.uniform(0.0, 0.2)                # confidence collapses
        s[7] = 0.0                                   # success_flag = False
    elif failure_type == "cascading_subgoal":
        s[6] = 0.0                                   # dependency_ok = False
        s[3] += magnitude                            # error_prob soars
    elif failure_type == "validator_fail":
        s[7] = 0.0
        s[5] *= 0.3
    else:
        s[3] += magnitude * 0.5
        s[7] = 0.0
    s = np.clip(s, 0.0, 2.0)
    return s


def generate_calling_tree(
    seed: int = 42,
    n_nodes_range: tuple = (22, 30),
    cascade_gain: float = 1.1,
    cascade_noise: float = 0.05,
) -> AgentCallTree:
    """Generate one agent calling-tree instance with injected failure."""
    rng = np.random.default_rng(seed)
    N = int(rng.integers(*n_nodes_range))

    # Assign node types with realistic proportions
    type_counts = {
        "planner": max(1, N // 10),
        "validator": max(1, N // 8),
        "executor": max(2, N // 5),
        "checker": max(1, N // 8),
        "aggregator": max(1, N // 10),
        "reporter": max(1, N // 10),
        "logger": max(1, N // 10),
        "error_handler": max(1, N // 12),
        "final_answer": 1,
    }
    # Fill remaining with executors
    assigned = sum(type_counts.values())
    type_counts["executor"] += max(0, N - assigned)
    N = sum(type_counts.values())

    node_types: list[str] = []
    for t, cnt in type_counts.items():
        node_types.extend([t] * cnt)
    rng.shuffle(node_types)

    node_ids = [f"{t[:3]}_{i}" for i, t in enumerate(node_types)]
    G = nx.DiGraph()
    for nid, ntype in zip(node_ids, node_types):
        G.add_node(nid, node_type=ntype, time_step=node_types.index(ntype))

    # --- Build DAG edges following role connectivity ---
    # Group nodes by type
    type_to_nodes: dict[str, list] = {t: [] for t in NODE_TYPES}
    for nid, ntype in zip(node_ids, node_types):
        type_to_nodes[ntype].append(nid)

    # Connect following ROLE_CONNECTS
    for src_type, targets in ROLE_CONNECTS.items():
        src_nodes = type_to_nodes[src_type]
        for tgt_type, etype in targets.items():
            tgt_nodes = type_to_nodes[tgt_type]
            if not src_nodes or not tgt_nodes:
                continue
            # Each src connects to 1-2 targets of this type
            for s in src_nodes:
                n_conn = min(len(tgt_nodes), int(rng.integers(1, 3)))
                chosen = rng.choice(tgt_nodes, size=n_conn, replace=False)
                for t in chosen:
                    if s != t and not G.has_edge(s, t):
                        G.add_edge(s, t, edge_type=etype)

    # Ensure acyclicity (remove back edges)
    try:
        cycles = list(nx.simple_cycles(G))
        for cycle in cycles:
            if len(cycle) >= 2:
                G.remove_edge(cycle[-1], cycle[0])
    except Exception:
        pass

    # Set t_star = final_answer node (sink)
    sinks = type_to_nodes["final_answer"]
    t_star = sinks[0] if sinks else node_ids[-1]
    G.graph["t_star"] = t_star

    # --- Initial states ---
    states = {nid: _random_state(rng, ntype)
              for nid, ntype in zip(node_ids, node_types)}

    # --- Inject root-cause failure ---
    failure_types = ["prediction_drift", "tool_misfire", "cascading_subgoal", "validator_fail"]
    failure_type = rng.choice(failure_types)
    # Prefer planner or executor as root cause
    root_candidates = type_to_nodes["planner"] + type_to_nodes["executor"]
    root_cause_node = rng.choice(root_candidates)
    magnitude = float(rng.uniform(0.4, 0.7))
    states[root_cause_node] = _inject_failure(
        states[root_cause_node], failure_type, rng, magnitude)

    # --- Cascade error through the graph (T1 recursion) ---
    # Topological order propagation: child error = α * parent_error + ε
    true_error = {nid: 0.0 for nid in node_ids}
    root_err = float(np.linalg.norm(states[root_cause_node] -
                                     _random_state(rng, type_to_nodes[
                                         G.nodes[root_cause_node]["node_type"]][0]
                                         if type_to_nodes[G.nodes[root_cause_node]["node_type"]]
                                         else "executor")))
    true_error[root_cause_node] = root_err

    try:
        topo_order = list(nx.topological_sort(G))
    except Exception:
        topo_order = node_ids

    for nid in topo_order:
        for pred in G.predecessors(nid):
            if true_error[pred] > 0:
                propagated = cascade_gain * true_error[pred] + float(
                    abs(rng.normal(0, cascade_noise)))
                true_error[nid] = max(true_error[nid], propagated)
                # Also corrupt state slightly
                if nid != root_cause_node:
                    states[nid][3] += propagated * 0.3  # error_prob rises
                    states[nid][7] *= max(0.0, 1.0 - propagated)  # success_flag drops
                    states[nid] = np.clip(states[nid], 0.0, 2.0)

    # --- Simulate NodeMSE@H (forward error propagation using T1) ---
    # Use amplification.simulate_error_propagation
    from . import amplification as amp
    from .failure_graph import build_from_agent_calling_tree

    G_f = build_from_agent_calling_tree(G, states, true_error, t_star)
    horizon_mse = amp.simulate_error_propagation(G_f, repaired=set(), H=32)

    # Build rollout trajectory (for paper evaluation)
    trajectory = []
    for step, nid in enumerate(topo_order):
        s_before = states[nid].copy()
        s_before[3] = 0.0  # pretend error_prob was 0 (counterfactual clean state)
        trajectory.append({
            "step": step,
            "node": nid,
            "node_type": G.nodes[nid]["node_type"],
            "state": states[nid].tolist(),
            "error": true_error[nid],
            "is_root_cause": nid == root_cause_node,
        })

    node_list = node_ids
    node_states_arr = np.array([states[nid] for nid in node_list])
    true_error_arr = np.array([true_error[nid] for nid in node_list])

    desc = (f"Agent calling-tree failure: {failure_type} injected at "
            f"{G.nodes[root_cause_node]['node_type']} node '{root_cause_node}' "
            f"(magnitude={magnitude:.2f}), cascading to "
            f"{sum(1 for e in true_error.values() if e > 0.1)} nodes")

    return AgentCallTree(
        G=G_f,
        node_states=node_states_arr,
        node_list=node_list,
        root_cause_node=root_cause_node,
        failure_desc=desc,
        true_error=true_error_arr,
        horizon_mse=horizon_mse,
        rollout_trajectory=trajectory,
    )


def generate_calling_trees(n: int = 30, seed: int = 42) -> list[AgentCallTree]:
    """Generate a batch of agent calling-tree instances."""
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 10_000, size=n).tolist()
    return [generate_calling_tree(int(s)) for s in seeds]


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def sink_success_rate(tree: AgentCallTree) -> float:
    """Whether the final_answer node achieved success (success_flag > 0.5)."""
    G = tree.G
    t_star = G.graph.get("t_star")
    if t_star is None:
        return 0.0
    feat = G.nodes[t_star].get("state", None)
    if feat is None:
        # Fall back to error
        err = G.nodes[t_star].get("err", 1.0)
        return float(err < 0.3)
    if hasattr(feat, "__len__"):
        return float(feat[-1] > 0.5)  # success_flag
    return float(feat < 0.3)


def node_mse_at_horizon(tree: AgentCallTree, H: int) -> float:
    """NodeMSE@H: simulated multi-step error at horizon H."""
    return tree.horizon_mse.get(H, float("nan"))


def multi_horizon_profile(tree: AgentCallTree,
                           horizons: list = None) -> dict:
    """NodeMSE at multiple horizons."""
    if horizons is None:
        horizons = [1, 2, 4, 8, 16, 32]
    return {H: node_mse_at_horizon(tree, H) for H in horizons}
