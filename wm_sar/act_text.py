"""Natural-language description generator for agent calling-tree instances.

Converts numeric node states into human-readable step descriptions so that
LLMs can reason about agent calling-tree failures.

Each node is described as:
    "[NodeType] node '{id}': <feature summary>"
where the feature summary uses the 8-dim state vector:
    [activation, load, latency, error_prob, throughput, confidence,
     dependency_ok, success_flag]

The LLM is then asked: given these node descriptions,
identify which node(s) most likely contain the root cause of the failure.
"""

from __future__ import annotations

import numpy as np
import networkx as nx

# Feature names for the 8-dim state vector
FEAT_NAMES = [
    "activation", "load", "latency", "error_prob",
    "throughput", "confidence", "dependency_ok", "success_flag",
]

# Thresholds for semantic labels
HIGH = 0.7
LOW  = 0.3


def _state_to_sentence(node_id: str, node_type: str, state: list | np.ndarray,
                        error: float, is_root_cause: bool = False) -> str:
    """Convert an 8-dim state vector to a human-readable sentence."""
    if state is None or len(state) < 8:
        state = [1.0, 0.3, 0.1, 0.0, 0.9, 0.9, 1.0, 1.0]
    s = list(state)

    parts = []
    activation  = s[0]
    load        = s[1]
    latency     = s[2]
    error_prob  = s[3]
    throughput  = s[4]
    confidence  = s[5]
    dep_ok      = s[6]
    success     = s[7]

    if activation < LOW:
        parts.append("INACTIVE")
    if error_prob > HIGH:
        parts.append(f"error_prob={error_prob:.2f} (HIGH)")
    elif error_prob > 0.3:
        parts.append(f"error_prob={error_prob:.2f} (elevated)")
    else:
        parts.append(f"error_prob={error_prob:.2f} (normal)")

    if throughput < LOW:
        parts.append(f"throughput={throughput:.2f} (LOW)")
    elif throughput > HIGH:
        parts.append(f"throughput={throughput:.2f} (OK)")

    if load > HIGH:
        parts.append(f"load={load:.2f} (HIGH)")

    if latency > HIGH:
        parts.append(f"latency={latency:.2f} (SLOW)")

    if confidence < LOW:
        parts.append(f"confidence={confidence:.2f} (LOW)")

    if dep_ok < 0.5:
        parts.append("dependencies UNSATISFIED")

    if success < 0.5:
        parts.append("FAILED (success_flag=0)")
    else:
        parts.append("succeeded")

    if error > 0.5:
        parts.append(f"[cascade_error={error:.2f}]")

    detail = ", ".join(parts) if parts else "all metrics normal"
    return (f"[{node_type.upper()}] '{node_id}': {detail}")


def tree_to_text(
    G: nx.DiGraph,
    selected_nodes: set | None = None,
    max_nodes: int = 30,
    include_edges: bool = True,
) -> tuple[str, list[str]]:
    """Convert failure graph to structured text for LLM.

    Args:
        G: failure graph with node attrs (node_type, err, state, etc.)
        selected_nodes: if given, only describe these nodes (the repair region)
        max_nodes: cap number of nodes shown
        include_edges: whether to include edge descriptions

    Returns:
        (text_description, ordered_node_ids)
    """
    try:
        topo = list(nx.topological_sort(G))
    except Exception:
        topo = sorted(G.nodes(), key=lambda v: G.nodes[v].get("time_step", 0))

    if selected_nodes is not None:
        nodes_to_show = [v for v in topo if v in selected_nodes]
    else:
        nodes_to_show = topo

    nodes_to_show = nodes_to_show[:max_nodes]
    t_star = G.graph.get("t_star", "")

    lines = ["=== Agent Calling-Tree Failure Report ===",
             f"Sink node (failed): {t_star}",
             f"Total nodes: {G.number_of_nodes()}, "
             f"Showing {len(nodes_to_show)} nodes",
             ""]

    lines.append("--- Node States ---")
    for v in nodes_to_show:
        d = G.nodes[v]
        ntype = d.get("node_type", "unknown")
        err   = float(d.get("err", 0.0))
        state = d.get("state", [])
        desc  = _state_to_sentence(v, ntype, state, err)
        lines.append(f"  Step {topo.index(v):2d}: {desc}")

    if include_edges and selected_nodes is not None:
        lines.append("")
        lines.append("--- Edges in repair region ---")
        for u, v, data in G.edges(data=True):
            if u in selected_nodes and v in selected_nodes:
                etype = data.get("edge_type", "calls")
                lines.append(f"  {u} --[{etype}]--> {v}")

    lines.append("")
    lines.append(f"The final node '{t_star}' has failed (success_flag=0).")
    return "\n".join(lines), nodes_to_show


def build_locate_prompt(
    tree_text: str, node_list: list[str], G: nx.DiGraph
) -> tuple[str, str]:
    """Build the LLM prompt for root-cause identification.

    Returns: (system_prompt, user_prompt)
    """
    system = (
        "You are an expert AI agent failure analyst. "
        "You will receive a report of a failed multi-agent calling-tree. "
        "Each node is an AI sub-agent (planner, executor, validator, etc.) "
        "with a state vector. Identify the root cause of the failure: "
        "which node introduced the initial error that cascaded to the final failure. "
        "Respond ONLY in valid JSON."
    )

    node_names = ", ".join(f"'{n}'" for n in node_list[:20])
    user = (
        f"{tree_text}\n\n"
        "Based on the node states above, which node most likely INTRODUCED the "
        "initial error (root cause)? Consider:\n"
        "- High error_prob + low success_flag = strong evidence of direct error\n"
        "- dependency UNSATISFIED = cascade victim, not root cause\n"
        "- LOW throughput at an executor is a strong signal\n\n"
        "Respond ONLY with valid JSON:\n"
        '{"root_cause_nodes": ["<node_id>", ...], '
        '"root_cause_type": "<node_type>", '
        '"explanation": "<one sentence>", '
        '"confidence": <0-1>}'
    )
    return system, user


def build_repair_prompt(
    tree_text: str, node_list: list[str], G: nx.DiGraph,
    located_root: list[str] | None = None,
) -> tuple[str, str]:
    """Build the LLM prompt for repair planning.

    Returns: (system_prompt, user_prompt)
    """
    system = (
        "You are an expert AI agent repair system. "
        "You receive a connected subgraph region identified by graph error amplification analysis. "
        "Propose concrete repairs for each failing node to restore the pipeline. "
        "Respond ONLY in valid JSON."
    )

    root_hint = ""
    if located_root:
        root_hint = f"\nPrevious analysis identified root cause near: {located_root}"

    user = (
        f"{tree_text}{root_hint}\n\n"
        "For each node in the region that shows anomalous state:\n"
        "1. Identify what went wrong\n"
        "2. Propose a specific corrective action\n\n"
        "Respond ONLY with valid JSON:\n"
        '{"repaired_nodes": ["<node_id>", ...], '
        '"repairs": {"<node_id>": "<corrective_action>", ...}, '
        '"explanation": "<overall repair strategy>", '
        '"confidence": <0-1>}'
    )
    return system, user


def parse_locate_response(response_text: str, true_root: str,
                           G: nx.DiGraph) -> dict:
    """Parse the LLM's root-cause identification response.

    Recovery: any identified node of same node_type as root cause, or within
    2 hops in the graph, counts as a match (±tolerance).
    """
    import json, re

    result = {
        "identified_nodes": [],
        "identified_type": None,
        "confidence": 0.0,
        "recovered_exact": False,
        "recovered_type": False,
        "recovered_hop2": False,
        "raw": response_text[:500],
    }

    # Try to parse JSON
    try:
        # strip markdown code fences
        clean = re.sub(r"```[a-z]*\n?", "", response_text).strip()
        data = json.loads(clean)
        result["identified_nodes"] = data.get("root_cause_nodes", [])
        result["identified_type"] = data.get("root_cause_type", None)
        result["confidence"] = float(data.get("confidence", 0.5))
    except Exception:
        # Try to extract node names
        matches = re.findall(r"'([a-z]+_\d+)'|\"([a-z]+_\d+)\"", response_text)
        result["identified_nodes"] = list(set(m[0] or m[1] for m in matches))

    true_type = G.nodes[true_root].get("node_type", "") if true_root in G else ""

    # Exact recovery
    if true_root in result["identified_nodes"]:
        result["recovered_exact"] = True

    # Type recovery (correct node type, even if wrong specific node)
    # Case-insensitive comparison
    if (result["identified_type"] or "").lower() == true_type.lower():
        result["recovered_type"] = True
    for nid in result["identified_nodes"]:
        if nid in G and G.nodes[nid].get("node_type") == true_type:
            result["recovered_type"] = True
            break

    # 2-hop recovery (within 2 hops of root cause)
    if true_root in G:
        hop2 = {true_root}
        und = G.to_undirected(as_view=True)
        for _ in range(2):
            nxt = set()
            for v in hop2:
                nxt.update(und.neighbors(v))
            hop2 |= nxt
        for nid in result["identified_nodes"]:
            if nid in hop2:
                result["recovered_hop2"] = True
                break

    return result
