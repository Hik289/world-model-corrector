"""Convert a failed world-model rollout into a NetworkX failure graph G_f.

The failure graph is a directed graph whose sink is the final-failure target
node ``t_star``. Node and edge types follow the spec (Section 4). Every node
carries the feature attributes used downstream by the amplification field, the
region extractor, and the repair operators.

Unified per-node attributes added for the algorithmics:
    err  : float  -- magnitude of (prediction/observation/structure) error
    unc  : float  -- uncertainty
    cost : float  -- repair/token cost weight of touching this node

Graph-level attributes:
    G.graph["t_star"]      -- id of the final_failure target node
    G.graph["domain"]      -- "agent_wm" | "parametric_gwm"
    G.graph["gt_region"]   -- ground-truth set of corrupted node ids
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

NODE_TYPES = [
    "world_model_state", "predicted_state", "observed_state", "state_delta",
    "agent_action", "tool_call", "tool_response", "artifact", "belief_state",
    "subgoal", "dependency", "validator", "error_node", "final_failure",
]

EDGE_TYPES = [
    "transition_to", "predicted_by", "observed_as", "action_causes", "calls",
    "returns", "produces", "consumes", "updates_state", "depends_on",
    "contradicts", "blocks", "validates", "propagates_error_to", "causes_failure",
]


def _add_node(G: nx.DiGraph, nid: str, node_type: str, **attrs: Any) -> str:
    base = dict(
        node_type=node_type,
        time_step=attrs.get("time_step", 0),
        prediction_confidence=attrs.get("prediction_confidence", 1.0),
        prediction_error=attrs.get("prediction_error", 0.0),
        observed_error=attrs.get("observed_error", 0.0),
        agent_identity=attrs.get("agent_identity", ""),
        status=attrs.get("status", "ok"),
        risk=attrs.get("risk", 0.0),
        uncertainty=attrs.get("uncertainty", 0.0),
        on_target_path=attrs.get("on_target_path", True),
        contradicts_observation=attrs.get("contradicts_observation", False),
        downstream_of_error=attrs.get("downstream_of_error", False),
        err=attrs.get("err", 0.0),
        unc=attrs.get("unc", 0.0),
        cost=attrs.get("cost", 1.0),
        gt=attrs.get("gt", False),
    )
    G.add_node(nid, **base)
    return nid


# ---------------------------------------------------------------------------
# Agent world-model rollout -> failure graph
# ---------------------------------------------------------------------------


def agent_rollout_to_graph(roll: Any) -> nx.DiGraph:
    """Build a failure graph from an :class:`AgentRollout`-shaped object/dict."""
    if isinstance(roll, dict):
        steps = roll["steps"]
        rid = roll["rollout_id"]
        failure_type = roll["failure_type"]
        gt_steps = set(roll.get("gt_region_steps", []))
        rc_type = roll.get("root_cause_node_type", "")
    else:
        steps = [s.__dict__ for s in roll.steps]
        rid = roll.rollout_id
        failure_type = roll.failure_type
        gt_steps = set(roll.gt_region_steps)
        rc_type = roll.root_cause_node_type

    G = nx.DiGraph()
    G.graph["domain"] = "agent_wm"
    G.graph["rollout_id"] = rid
    G.graph["failure_type"] = failure_type

    prev_pred = None
    gt_region: set[str] = set()

    for s in steps:
        t = s["t"]
        downstream = s["downstream_of_error"]
        err = float(s["prediction_error"])
        unc = float(s["uncertainty"])
        in_gt = t in gt_steps

        common = dict(
            time_step=t,
            prediction_confidence=s["prediction_confidence"],
            prediction_error=err,
            observed_error=s["observed_error"],
            agent_identity=s["agent_identity"],
            status=s["status"],
            risk=s["risk"],
            uncertainty=unc,
            on_target_path=s["on_target_path"],
            contradicts_observation=s["contradicts_observation"],
            downstream_of_error=downstream,
            unc=unc,
        )

        # The predicted_state chain is the dominant error carrier: once the
        # root cause injects at root_t, error amplifies along transition_to.
        # Observations / actions / tool calls carry only sensing noise so the
        # amplification region is a clean connected chain (pred + root-cause node).
        noise = 0.04 + 0.02 * abs(float(np.tanh(t)))
        root_t = min(gt_steps) if gt_steps else -1
        at_root = (t == root_t)

        pred = _add_node(G, f"pred_{t}", "predicted_state",
                         err=err if downstream else noise,
                         gt=in_gt and rc_type == "predicted_state", **common)
        obs = _add_node(G, f"obs_{t}", "observed_state", err=noise, **common)
        act = _add_node(G, f"act_{t}", "agent_action", err=noise, **common)
        tcall = _add_node(G, f"tcall_{t}", "tool_call", err=noise, **common)
        tresp = _add_node(
            G, f"tresp_{t}", "tool_response",
            err=(err if (rc_type == "tool_response" and at_root) else noise),
            gt=in_gt and rc_type == "tool_response" and at_root, **common,
        )

        # observed_as: prediction explained by observation
        G.add_edge(pred, obs, edge_type="observed_as")
        # action_causes: action drives next predicted state
        G.add_edge(act, pred, edge_type="action_causes")
        # calls / returns
        G.add_edge(act, tcall, edge_type="calls")
        G.add_edge(tcall, tresp, edge_type="returns")
        G.add_edge(tresp, pred, edge_type="updates_state")

        # transition backbone
        if prev_pred is not None:
            G.add_edge(prev_pred, pred, edge_type="transition_to")
            if downstream:
                G.add_edge(prev_pred, pred, edge_type="propagates_error_to")
        prev_pred = pred

        # type-specific structure (only the root-step instance is a carrier)
        root_carrier = None
        if not s["subgoal_done"] or rc_type == "subgoal":
            carrier = rc_type == "subgoal" and at_root
            sg = _add_node(G, f"subgoal_{t}", "subgoal",
                           err=err if carrier else noise,
                           gt=carrier, **common)
            G.add_edge(sg, act, edge_type="depends_on")
            G.add_edge(sg, pred, edge_type="updates_state")
            if carrier:
                root_carrier = sg
        if not s["artifact_ok"] or rc_type == "artifact":
            carrier = rc_type == "artifact" and at_root
            art = _add_node(G, f"art_{t}", "artifact",
                            err=err if carrier else noise,
                            gt=carrier, **common)
            G.add_edge(tresp, art, edge_type="produces")
            G.add_edge(art, pred, edge_type="consumes")
            if carrier:
                root_carrier = art
        if s["contradicts_observation"]:
            carrier = rc_type == "belief_state" and at_root
            bel = _add_node(G, f"belief_{t}", "belief_state",
                            err=err if carrier else noise,
                            gt=carrier, **common)
            G.add_edge(bel, obs, edge_type="contradicts")
            G.add_edge(bel, pred, edge_type="updates_state")
            if carrier:
                root_carrier = bel
        if failure_type == "false_dependency" and t in gt_steps:
            carrier = rc_type == "dependency" and at_root
            dep = _add_node(G, f"dep_{t}", "dependency",
                            err=err if carrier else noise,
                            gt=carrier, **common)
            G.add_edge(dep, act, edge_type="depends_on")
            G.add_edge(dep, pred, edge_type="blocks")
            if carrier:
                root_carrier = dep
        if rc_type == "tool_response" and at_root:
            root_carrier = tresp

        # the amplification region = predicted_state chain (downstream) + the
        # root-cause carrier node at root_t
        if downstream:
            gt_region.add(pred)
        if root_carrier is not None:
            gt_region.add(root_carrier)

    # final failure target node
    T = steps[-1]["t"]
    # The target node's error is *derived* from inflow during propagation (it has
    # no intrinsic error of its own), so repairing the upstream amplification
    # region can drive it down.
    tstar = _add_node(
        G, "final_failure", "final_failure",
        time_step=T + 1, status="failed",
        prediction_error=0.0, err=0.0,
        unc=float(steps[-1]["uncertainty"]),
        downstream_of_error=True,
    )
    G.add_edge(f"pred_{T}", tstar, edge_type="causes_failure")
    G.add_edge(f"pred_{T}", tstar, edge_type="propagates_error_to")
    G.graph["t_star"] = tstar
    G.graph["gt_region"] = gt_region
    return G


# ---------------------------------------------------------------------------
# Parametric GWM rollout -> failure graph
# ---------------------------------------------------------------------------


def gwm_rollout_to_graph(roll: Any) -> nx.DiGraph:
    """Build a failure graph from a :class:`GWMRollout`-shaped object/dict."""
    if isinstance(roll, dict):
        d = roll
    else:
        d = roll.to_dict()

    pred_node = np.array(d["pred_node"])     # (T+1, n)
    true_node = np.array(d["true_node"])
    pred_adj = np.array(d["pred_adj"])       # (T+1, n, n)
    true_adj = np.array(d["true_adj"])
    T = d["horizon"]
    n = d["n_nodes"]
    root_t = d["root_cause_t"]
    root_node = d["root_cause_node"]
    gt_steps = set(d["gt_region_steps"])

    G = nx.DiGraph()
    G.graph["domain"] = "parametric_gwm"
    G.graph["rollout_id"] = d["rollout_id"]
    G.graph["failure_type"] = d["failure_type"]

    node_err = np.abs(pred_node - true_node)            # (T+1, n)
    edge_err = np.abs(pred_adj - true_adj).sum(axis=2)  # (T+1, n) outgoing edge err

    gt_region: set[str] = set()
    for t in range(T + 1):
        for i in range(n):
            err = float(node_err[t, i] + 0.3 * edge_err[t, i])
            unc = float(err * 0.8 + 0.02)
            is_gt = (t in gt_steps) and (i == root_node)
            nid = _add_node(
                G, f"n{i}_t{t}", "predicted_state",
                time_step=t, prediction_error=float(node_err[t, i]),
                observed_error=float(edge_err[t, i]),
                prediction_confidence=float(np.clip(1 - err, 0.02, 0.99)),
                risk=err, uncertainty=unc, err=err, unc=unc,
                downstream_of_error=(t >= root_t and i == root_node),
                gt=is_gt,
            )
            if is_gt:
                gt_region.add(nid)

    # temporal + structural edges propagating error toward later states
    for t in range(T):
        for i in range(n):
            G.add_edge(f"n{i}_t{t}", f"n{i}_t{t+1}", edge_type="transition_to")
            # structural propagation via predicted adjacency
            for j in range(n):
                if pred_adj[t + 1, i, j] > 0.5:
                    G.add_edge(f"n{i}_t{t}", f"n{j}_t{t+1}",
                               edge_type="propagates_error_to")

    # final failure: aggregate the worst final-step nodes
    # target error is derived from inflow during propagation (no intrinsic error)
    tstar = _add_node(
        G, "final_failure", "final_failure", time_step=T + 1, status="failed",
        prediction_error=0.0, err=0.0,
        unc=float(node_err[T].mean()), downstream_of_error=True,
    )
    worst = int(np.argmax(node_err[T]))
    for i in range(n):
        if node_err[T, i] > node_err[T].mean() or i == worst:
            G.add_edge(f"n{i}_t{T}", tstar, edge_type="causes_failure")
            G.add_edge(f"n{i}_t{T}", tstar, edge_type="propagates_error_to")
    G.graph["t_star"] = tstar
    G.graph["gt_region"] = gt_region
    return G


def world_model_failure_to_graph(failed_case: Any) -> nx.DiGraph:
    """Dispatch to the right builder based on the rollout domain."""
    domain = failed_case["domain"] if isinstance(failed_case, dict) else failed_case.domain
    if domain == "agent_wm":
        return agent_rollout_to_graph(failed_case)
    if domain == "parametric_gwm":
        return gwm_rollout_to_graph(failed_case)
    raise ValueError(f"unknown domain: {domain}")


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------


def build_from_agent_calling_tree(
    G_tree: nx.DiGraph,
    states: dict,           # {node_id: np.ndarray of shape (8,)}
    true_error: dict,       # {node_id: float}
    t_star: str,
) -> nx.DiGraph:
    """Convert an agent calling-tree instance into a failure graph G_f.

    Each node in G_tree becomes a node in G_f with:
        err  = true_error[node]
        unc  = derived from state[3] (error_prob feature)
        cost = 1.0 (uniform; could weight by node_type)
        node_type = from G_tree.nodes[node]["node_type"]
        time_step = topological index

    Edges are preserved with their original edge_type.
    G_f.graph["t_star"] is set to t_star.
    G_f.graph["gt_region"] is the set of nodes with err > mean_err.
    """
    G_f = nx.DiGraph()

    # Topological order → time_step
    try:
        topo = list(nx.topological_sort(G_tree))
    except Exception:
        topo = list(G_tree.nodes())
    topo_idx = {n: i for i, n in enumerate(topo)}

    errs = np.array(list(true_error.values()))
    mean_err = float(np.mean(errs)) if len(errs) > 0 else 0.0

    for n in G_tree.nodes():
        ntype = G_tree.nodes[n].get("node_type", "executor")
        err = float(true_error.get(n, 0.0))
        state = states.get(n)
        if isinstance(state, (np.ndarray, list)):
            unc = float(state[3]) if len(state) > 3 else 0.1  # error_prob as uncertainty
        else:
            unc = 0.1
        G_f.add_node(
            n,
            node_type=ntype,
            time_step=topo_idx.get(n, 0),
            err=err,
            unc=unc,
            cost=1.0,
            status="error" if err > mean_err else "ok",
            state=state if state is not None else [],
        )

    for u, v, data in G_tree.edges(data=True):
        G_f.add_edge(u, v, edge_type=data.get("edge_type", "calls"))

    gt_region = {n for n in G_tree.nodes() if true_error.get(n, 0.0) > mean_err}
    G_f.graph["t_star"] = t_star
    G_f.graph["gt_region"] = gt_region
    G_f.graph["domain"] = "agent_calling_tree"
    return G_f


def node_error(G: nx.DiGraph, v: str) -> float:
    return float(G.nodes[v].get("err", 0.0))


def node_unc(G: nx.DiGraph, v: str) -> float:
    return float(G.nodes[v].get("unc", 0.0))


def node_cost(G: nx.DiGraph, v: str) -> float:
    return float(G.nodes[v].get("cost", 1.0))
