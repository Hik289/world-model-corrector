from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


NODE_TYPES = [
    "planner", "validator", "executor", "checker",
    "aggregator", "reporter", "logger", "error_handler", "final_answer",
]

EDGE_TYPES = ["calls", "validates", "reports", "errors", "triggers", "logs"]


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


FEAT_DIM = 8
FEAT_NAMES = [
    "activation",
    "load",
    "latency",
    "error_prob",
    "throughput",
    "confidence",
    "dependency_ok",
    "success_flag",
]


@dataclass
class AgentCallTree:
    G: nx.DiGraph
    node_states: np.ndarray
    node_list: list
    root_cause_node: str
    failure_desc: str
    true_error: np.ndarray
    horizon_mse: dict
    rollout_trajectory: list


def _random_state(rng: np.random.Generator, node_type: str) -> np.ndarray:

    s = np.zeros(FEAT_DIM)
    s[0] = 1.0
    s[1] = rng.uniform(0.1, 0.5)
    s[2] = rng.uniform(0.05, 0.3)
    s[3] = rng.uniform(0.0, 0.05)
    s[4] = rng.uniform(0.7, 1.0)
    s[5] = rng.uniform(0.8, 1.0)
    s[6] = 1.0
    s[7] = 1.0
    if node_type == "error_handler":
        s[3] = rng.uniform(0.1, 0.3)
    if node_type == "final_answer":
        s[5] = rng.uniform(0.5, 0.9)
    return s


def _inject_failure(state: np.ndarray, failure_type: str, rng: np.random.Generator,
                    magnitude: float = 0.5) -> np.ndarray:

    s = state.copy()
    if failure_type == "prediction_drift":
        s[4] -= magnitude * rng.uniform(0.5, 1.0)
        s[3] += magnitude * rng.uniform(0.3, 0.7)
    elif failure_type == "tool_misfire":
        s[5] = rng.uniform(0.0, 0.2)
        s[7] = 0.0
    elif failure_type == "cascading_subgoal":
        s[6] = 0.0
        s[3] += magnitude
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

    rng = np.random.default_rng(seed)
    N = int(rng.integers(*n_nodes_range))


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


    type_to_nodes: dict[str, list] = {t: [] for t in NODE_TYPES}
    for nid, ntype in zip(node_ids, node_types):
        type_to_nodes[ntype].append(nid)


    for src_type, targets in ROLE_CONNECTS.items():
        src_nodes = type_to_nodes[src_type]
        for tgt_type, etype in targets.items():
            tgt_nodes = type_to_nodes[tgt_type]
            if not src_nodes or not tgt_nodes:
                continue

            for s in src_nodes:
                n_conn = min(len(tgt_nodes), int(rng.integers(1, 3)))
                chosen = rng.choice(tgt_nodes, size=n_conn, replace=False)
                for t in chosen:
                    if s != t and not G.has_edge(s, t):
                        G.add_edge(s, t, edge_type=etype)


    try:
        cycles = list(nx.simple_cycles(G))
        for cycle in cycles:
            if len(cycle) >= 2 and G.has_edge(cycle[-1], cycle[0]):
                G.remove_edge(cycle[-1], cycle[0])
    except Exception:
        pass


    sinks = type_to_nodes["final_answer"]
    t_star = sinks[0] if sinks else node_ids[-1]
    G.graph["t_star"] = t_star


    states = {nid: _random_state(rng, ntype)
              for nid, ntype in zip(node_ids, node_types)}


    failure_types = ["prediction_drift", "tool_misfire", "cascading_subgoal", "validator_fail"]
    failure_type = rng.choice(failure_types)

    root_candidates = type_to_nodes["planner"] + type_to_nodes["executor"]
    root_cause_node = rng.choice(root_candidates)
    magnitude = float(rng.uniform(0.4, 0.7))
    states[root_cause_node] = _inject_failure(
        states[root_cause_node], failure_type, rng, magnitude)


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

                if nid != root_cause_node:
                    states[nid][3] += propagated * 0.3
                    states[nid][7] *= max(0.0, 1.0 - propagated)
                    states[nid] = np.clip(states[nid], 0.0, 2.0)


    from . import amplification as amp
    from .failure_graph import build_from_agent_calling_tree

    G_f = build_from_agent_calling_tree(G, states, true_error, t_star)
    horizon_mse = amp.simulate_error_propagation(G_f, repaired=set(), H=32)


    trajectory = []
    for step, nid in enumerate(topo_order):
        s_before = states[nid].copy()
        s_before[3] = 0.0
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

    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 10_000, size=n).tolist()
    return [generate_calling_tree(int(s)) for s in seeds]


def sink_success_rate(tree: AgentCallTree) -> float:

    G = tree.G
    t_star = G.graph.get("t_star")
    if t_star is None:
        return 0.0
    feat = G.nodes[t_star].get("state", None)
    if feat is None:

        err = G.nodes[t_star].get("err", 1.0)
        return float(err < 0.3)
    if hasattr(feat, "__len__"):
        return float(feat[-1] > 0.5)
    return float(feat < 0.3)


def node_mse_at_horizon(tree: AgentCallTree, H: int) -> float:

    return tree.horizon_mse.get(H, float("nan"))


def multi_horizon_profile(tree: AgentCallTree,
                           horizons: list = None) -> dict:

    if horizons is None:
        horizons = [1, 2, 4, 8, 16, 32]
    return {H: node_mse_at_horizon(tree, H) for H in horizons}
