from __future__ import annotations

import json
import math
import re

import numpy as np
import networkx as nx

CONTEXT_PROTOCOL_VERSION = "recore-context-v2"


FEAT_NAMES = [
    "activation", "load", "latency", "error_prob",
    "throughput", "confidence", "dependency_ok", "success_flag",
]


HIGH = 0.7
LOW  = 0.3


def _state_to_sentence(node_id: str, node_type: str, state: list | np.ndarray,
                        error: float, is_root_cause: bool = False) -> str:

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
    max_nodes: int | None = None,
    include_edges: bool = True,
) -> tuple[str, list[str]]:


    if max_nodes is not None and max_nodes < 0:
        raise ValueError("max_nodes must be non-negative or None")
    if selected_nodes is not None and not set(selected_nodes).issubset(G.nodes()):
        raise ValueError("selected_nodes contains nodes outside the graph")

    try:
        topo = list(nx.topological_sort(G))
    except Exception:
        topo = sorted(G.nodes(), key=lambda v: G.nodes[v].get("time_step", 0))

    if selected_nodes is not None:
        nodes_to_show = [v for v in topo if v in selected_nodes]
    else:
        nodes_to_show = topo

    if max_nodes is not None:
        nodes_to_show = nodes_to_show[:max_nodes]
    visible_nodes = set(nodes_to_show)
    topo_index = {v: i for i, v in enumerate(topo)}
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
        state_values = np.asarray(state).tolist() if state is not None else []
        confidence = d.get("confidence", d.get("prediction_confidence"))
        if confidence is None and isinstance(state_values, list) and len(state_values) > 5:
            confidence = state_values[5]
        confidence_text = "unavailable" if confidence is None else f"{float(confidence):.6g}"
        lines.append(f"  Step {topo_index[v]:2d}: {desc}")
        lines.append(
            f"    state={json.dumps(state_values)}; observed_error={err:.6g}; "
            f"confidence={confidence_text}"
        )

    if include_edges:
        lines.append("")
        lines.append("--- Edges in repair region ---")
        for u, v, data in G.edges(data=True):
            if u in visible_nodes and v in visible_nodes:
                etype = data.get("edge_type", "calls")
                lines.append(f"  {u} --[{etype}]--> {v}")
        boundary_edges = [
            (u, v, data) for u, v, data in G.edges(data=True)
            if (u in visible_nodes) != (v in visible_nodes)
        ]
        if boundary_edges:
            lines.append("")
            lines.append("--- Boundary dependencies (external context only) ---")
            for u, v, data in boundary_edges:
                etype = data.get("edge_type", "calls")
                source = str(u) if u in visible_nodes else f"{u} [external]"
                target = str(v) if v in visible_nodes else f"{v} [external]"
                lines.append(f"  {source} --[{etype}]--> {target}")

    lines.append("")
    lines.append(f"The final node '{t_star}' has failed (success_flag=0).")
    return "\n".join(lines), nodes_to_show


def build_locate_prompt(
    tree_text: str, node_list: list[str], G: nx.DiGraph
) -> tuple[str, str]:


    system = (
        "You are an expert AI agent failure analyst. "
        "You will receive a report of a failed multi-agent calling-tree. "
        "Each node is an AI sub-agent (planner, executor, validator, etc.) "
        "with a state vector. Identify the root cause of the failure: "
        "which node introduced the initial error that cascaded to the final failure. "
        "Respond ONLY in valid JSON."
    )

    user = (
        f"{tree_text}\n\n"
        "Based on the node states above, which node most likely INTRODUCED the "
        "initial error (root cause)? Consider:\n"
        "- High error_prob + low success_flag = strong evidence of direct error\n"
        "- dependency UNSATISFIED = cascade victim, not root cause\n"
        "- LOW throughput at an executor is a strong signal\n\n"
        f"Eligible root-cause node IDs: {json.dumps(node_list)}.\n"
        "Return only eligible IDs; external boundary endpoints are context, "
        "not repair candidates. If none can be identified, return an empty list.\n\n"
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
        f"Eligible repair node IDs: {json.dumps(node_list)}.\n"
        "Do not repair external boundary endpoints.\n\n"
        "Respond ONLY with valid JSON:\n"
        '{"repaired_nodes": ["<node_id>", ...], '
        '"repairs": {"<node_id>": "<corrective_action>", ...}, '
        '"explanation": "<overall repair strategy>", '
        '"confidence": <0-1>}'
    )
    return system, user


def parse_locate_response(response_text: str, true_root: str,
                           G: nx.DiGraph,
                           allowed_nodes: set[str] | None = None) -> dict:

    def reject_constant(value):
        raise ValueError(f"non-finite JSON number: {value}")

    def unique_object(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError(f"duplicate JSON key: {key}")
            obj[key] = value
        return obj

    result = {
        "identified_nodes": [],
        "identified_type": None,
        "confidence": 0.0,
        "recovered_exact": False,
        "recovered_type": False,
        "recovered_hop2": False,
        "raw": response_text[:500] if isinstance(response_text, str) else "",
        "valid_json": False,
        "valid_response": False,
        "invalid_reason": "",
        "explanation": "",
    }

    if not isinstance(response_text, str):
        result["invalid_reason"] = "invalid_json"
        return result

    try:

        clean = response_text.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", clean,
                              flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            clean = fenced.group(1)
        data = json.loads(clean, parse_constant=reject_constant,
                          object_pairs_hook=unique_object)
    except (TypeError, ValueError):
        result["invalid_reason"] = "invalid_json"
        return result
    result["valid_json"] = True
    if not isinstance(data, dict):
        result["invalid_reason"] = "expected_object"
        return result
    nodes = data.get("root_cause_nodes")
    node_type = data.get("root_cause_type")
    explanation = data.get("explanation")
    confidence = data.get("confidence")
    if (not isinstance(nodes, list)
            or not all(isinstance(n, str) for n in nodes)
            or not isinstance(node_type, str)
            or not isinstance(explanation, str)
            or isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0.0 <= confidence <= 1.0
            or not math.isfinite(confidence)):
        result["invalid_reason"] = "invalid_schema"
        return result
    allowed = set(G.nodes()) if allowed_nodes is None else set(allowed_nodes) & set(G.nodes())
    if any(n not in allowed for n in nodes):
        result["invalid_reason"] = "node_outside_selected_region"
        return result
    result["identified_nodes"] = list(dict.fromkeys(nodes))
    result["identified_type"] = node_type
    result["confidence"] = float(confidence)
    result["explanation"] = explanation
    result["valid_response"] = True

    true_type = G.nodes[true_root].get("node_type", "") if true_root in G else ""


    if true_root in result["identified_nodes"]:
        result["recovered_exact"] = True


    if true_type and (result["identified_type"] or "").lower() == true_type.lower():
        result["recovered_type"] = True
    for nid in result["identified_nodes"]:
        if nid in G and G.nodes[nid].get("node_type") == true_type:
            result["recovered_type"] = True
            break


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
