from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

from .agent_calling_tree import (
    AgentCallTree,
)
from .failure_graph import build_from_agent_calling_tree
from . import amplification as amp


def _make_state(rng: np.random.Generator,
                node_type: str,
                err_level: float = 0.0) -> np.ndarray:

    s = np.array([
        1.0,
        float(rng.uniform(0.1, 0.5)),
        float(rng.uniform(0.05, 0.3)),
        float(rng.uniform(0.0, 0.05)),
        float(rng.uniform(0.7, 1.0)),
        float(rng.uniform(0.8, 1.0)),
        1.0,
        1.0,
    ], dtype=float)

    if err_level > 0.0:
        s[3] += err_level * float(rng.uniform(0.5, 0.9))
        s[5] *= max(0.0, 1.0 - err_level * 0.8)
        s[7]  = max(0.0, 1.0 - err_level)
        s[4] *= max(0.0, 1.0 - err_level * 0.5)
        s = np.clip(s, 0.0, 2.0)

    if node_type == "final_answer":
        s[5] = float(rng.uniform(0.5, 0.9))
    elif node_type == "validator":
        s[3] *= 0.5
    return np.clip(s, 0.0, 2.0)


def _cascade(G: nx.DiGraph,
             states: dict[str, np.ndarray],
             root: str,
             root_err: float,
             gain: float,
             noise: float,
             rng: np.random.Generator) -> dict[str, float]:

    errs: dict[str, float] = {n: 0.0 for n in G.nodes()}
    errs[root] = root_err


    visited: set[str] = set()
    queue = [root]
    visited.add(root)
    while queue:
        nxt = []
        for nid in queue:
            for child in G.successors(nid):
                child_gain = gain * errs[nid] + abs(float(rng.normal(0, noise)))
                errs[child] = max(errs[child], child_gain)
                if errs[child] > 0 and child != root:
                    s = states[child]
                    s[3] = min(2.0, s[3] + errs[child] * 0.3)
                    s[7] = max(0.0, s[7] - errs[child] * 0.5)
                    s[5] = max(0.0, s[5] - errs[child] * 0.4)
                    states[child] = s
                if child not in visited:
                    visited.add(child)
                    nxt.append(child)
        queue = nxt
    return errs


def _assemble(G: nx.DiGraph,
              states: dict[str, np.ndarray],
              errs: dict[str, float],
              sink: str,
              root: str,
              desc: str) -> AgentCallTree:

    G.graph["t_star"] = sink
    G_f = build_from_agent_calling_tree(G, states, errs, sink)
    horizon_mse = amp.simulate_error_propagation(G_f, repaired=set(), H=32)

    try:
        topo = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        topo = list(G.nodes())

    trajectory = []
    for step, nid in enumerate(topo):
        trajectory.append({
            "step": step,
            "node": nid,
            "node_type": G.nodes[nid].get("node_type", "executor"),
            "state": states[nid].tolist(),
            "error": errs[nid],
            "is_root_cause": (nid == root),
        })

    node_list  = list(G.nodes())
    node_arr   = np.array([states[n] for n in node_list])
    error_arr  = np.array([errs[n]   for n in node_list])

    return AgentCallTree(
        G=G_f,
        node_states=node_arr,
        node_list=node_list,
        root_cause_node=root,
        failure_desc=desc,
        true_error=error_arr,
        horizon_mse=horizon_mse,
        rollout_trajectory=trajectory,
    )


def _swe_single(seed: int) -> AgentCallTree:


    rng  = np.random.default_rng(seed)
    GAIN = 1.15; NOISE = 0.04

    K = int(rng.integers(3, 6))
    J = int(rng.integers(2, 4))
    M = int(rng.integers(2, 4))

    G = nx.DiGraph()


    ia  = "IssueAnalyzer";  G.add_node(ia,  node_type="planner",      time_step=0)
    re  = "RepoExplorer";   G.add_node(re,  node_type="executor",     time_step=1)
    G.add_edge(ia, re, edge_type="calls")

    fls = []
    for i in range(K):
        fl = f"FileLocator_{i}"; G.add_node(fl, node_type="executor", time_step=2)
        G.add_edge(ia, fl, edge_type="calls")
        G.add_edge(re, fl, edge_type="calls")
        fls.append(fl)

    cas = []
    ts  = 3
    for i, fl in enumerate(fls):
        for j in range(J):
            ca = f"CodeAnalyzer_{i}_{j}"
            G.add_node(ca, node_type="checker", time_step=ts + i)
            G.add_edge(fl, ca, edge_type="triggers")
            cas.append(ca)

    pw = "PatchWriter"; G.add_node(pw, node_type="executor", time_step=ts + K)
    for ca in cas:
        G.add_edge(ca, pw, edge_type="reports")

    trs = []
    for i in range(M):
        tr = f"TestRunner_{i}"
        G.add_node(tr, node_type="validator", time_step=ts + K + 1)
        G.add_edge(pw, tr, edge_type="triggers")
        trs.append(tr)

    ci = "CIChecker"; G.add_node(ci, node_type="validator", time_step=ts + K + 2)
    for tr in trs:
        G.add_edge(tr, ci, edge_type="validates")

    fa = "FinalAnswer"; G.add_node(fa, node_type="final_answer", time_step=ts + K + 3)
    G.add_edge(ci, fa, edge_type="triggers")


    G.add_edge(ci, ia, edge_type="errors")


    modes = ["wrong_file", "wrong_patch", "import_error"]
    mode  = str(rng.choice(modes))
    if mode == "wrong_file":
        root = fls[0]; mag = float(rng.uniform(0.50, 0.80))
        desc = f"SWE-bench:wrong_file — FileLocator_0 finds wrong file (err={mag:.2f})"
    elif mode == "wrong_patch":
        root = pw;     mag = float(rng.uniform(0.40, 0.70))
        desc = f"SWE-bench:wrong_patch — PatchWriter bad patch (err={mag:.2f})"
    else:
        root = cas[0]; mag = float(rng.uniform(0.45, 0.75))
        desc = f"SWE-bench:import_error — CodeAnalyzer_0_0 misread (err={mag:.2f})"

    states = {n: _make_state(rng, G.nodes[n]["node_type"],
                              mag if n == root else 0.0)
              for n in G.nodes()}
    errs = _cascade(G, states, root, mag * 0.82 + float(rng.uniform(0.05, 0.15)),
                    GAIN, NOISE, rng)
    return _assemble(G, states, errs, fa, root, desc)


def generate_swe_bench_graphs(n: int = 50, seed: int = 42) -> list[AgentCallTree]:

    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 100_000, size=n).tolist()
    out = []
    for s in seeds:
        try:
            out.append(_swe_single(int(s)))
        except Exception as exc:
            print(f"  [SWE-bench] seed={s} skip: {exc}")
    return out


def _webarena_single(seed: int) -> AgentCallTree:


    rng  = np.random.default_rng(seed)
    GAIN = 1.08; NOISE = 0.03

    hops = int(rng.integers(2, 5))

    G = nx.DiGraph()
    tp = "TaskPlanner"; G.add_node(tp, node_type="planner", time_step=0)

    navs, prs, ces = [], [], []
    prev = tp
    for i in range(hops):
        nav = f"Navigator_{i+1}"
        pr  = f"PageReader_{i+1}"
        ce  = f"ContentExtractor_{i+1}"
        ts  = 1 + i * 3
        G.add_node(nav, node_type="executor",   time_step=ts)
        G.add_node(pr,  node_type="checker",    time_step=ts+1)
        G.add_node(ce,  node_type="aggregator", time_step=ts+2)
        G.add_edge(prev, nav, edge_type="calls")
        G.add_edge(nav,  pr,  edge_type="triggers")
        G.add_edge(pr,   ce,  edge_type="reports")
        navs.append(nav); prs.append(pr); ces.append(ce)
        prev = nav

    ts = 1 + hops * 3
    ff = "FormFiller";    G.add_node(ff, node_type="executor", time_step=ts); ts += 1
    for ce in ces:        G.add_edge(ce, ff, edge_type="reports")

    fv = "FormValidator"; G.add_node(fv, node_type="validator", time_step=ts); ts += 1
    G.add_edge(ff, fv, edge_type="triggers")

    sub = "Submitter";    G.add_node(sub, node_type="executor",  time_step=ts); ts += 1
    G.add_edge(fv, sub, edge_type="validates")

    sc = "SuccessChecker"; G.add_node(sc, node_type="validator",   time_step=ts); ts += 1
    G.add_edge(sub, sc, edge_type="validates")

    fa = "FinalAnswer";   G.add_node(fa, node_type="final_answer", time_step=ts)
    G.add_edge(sc, fa, edge_type="triggers")


    G.add_edge(fv, navs[0], edge_type="errors")

    modes = ["wrong_url", "wrong_content", "form_error"]
    mode  = str(rng.choice(modes))
    if mode == "wrong_url":
        root = navs[0]; mag = float(rng.uniform(0.45, 0.75))
        desc = f"WebArena:wrong_url — Navigator_1 goes to wrong page (err={mag:.2f})"
    elif mode == "wrong_content":
        root = prs[0];  mag = float(rng.uniform(0.40, 0.70))
        desc = f"WebArena:wrong_content — PageReader_1 extracts wrong text (err={mag:.2f})"
    else:
        root = ff;      mag = float(rng.uniform(0.50, 0.80))
        desc = f"WebArena:form_error — FormFiller uses wrong field values (err={mag:.2f})"

    states = {n: _make_state(rng, G.nodes[n]["node_type"],
                              mag if n == root else 0.0)
              for n in G.nodes()}
    errs = _cascade(G, states, root, mag * 0.77 + float(rng.uniform(0.05, 0.15)),
                    GAIN, NOISE, rng)
    return _assemble(G, states, errs, fa, root, desc)


def generate_webarena_graphs(n: int = 50, seed: int = 42) -> list[AgentCallTree]:

    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 100_000, size=n).tolist()
    out = []
    for s in seeds:
        try:
            out.append(_webarena_single(int(s)))
        except Exception as exc:
            print(f"  [WebArena] seed={s} skip: {exc}")
    return out


def _agentbench_single(seed: int) -> AgentCallTree:


    rng  = np.random.default_rng(seed)
    GAIN = 1.12; NOISE = 0.035

    n_bash = int(rng.integers(3, 6))
    has_env = bool(rng.random() > 0.4)

    G = nx.DiGraph()
    cmd = "Commander"; G.add_node(cmd, node_type="planner", time_step=0)

    ts = 1
    env = None
    if has_env:
        env = "EnvSetup"; G.add_node(env, node_type="executor", time_step=ts)
        G.add_edge(cmd, env, edge_type="calls"); ts += 1

    bashes, parsers = [], []
    prev_b = None
    for i in range(n_bash):
        b = f"BashNode_{i+1}"; p = f"OutputParser_{i+1}"
        G.add_node(b, node_type="executor", time_step=ts)
        G.add_node(p, node_type="checker",  time_step=ts+1)
        if i == 0:
            G.add_edge(cmd, b, edge_type="calls")
        else:
            G.add_edge(prev_b, b, edge_type="triggers")
        if has_env and i < 2:
            G.add_edge(env, b, edge_type="calls")
        G.add_edge(b, p, edge_type="triggers")
        bashes.append(b); parsers.append(p); prev_b = b; ts += 2

    pn = "PipelineNode"; G.add_node(pn, node_type="aggregator", time_step=ts); ts += 1
    for p in parsers:    G.add_edge(p, pn, edge_type="reports")

    vf = "Verifier";     G.add_node(vf, node_type="validator",   time_step=ts); ts += 1
    G.add_edge(pn, vf, edge_type="validates")

    fa = "FinalAnswer";  G.add_node(fa, node_type="final_answer", time_step=ts)
    G.add_edge(vf, fa, edge_type="triggers")


    G.add_edge(vf, cmd, edge_type="errors")

    modes = ["cmd_error", "pipe_error"]
    if has_env:
        modes.append("env_error")
    mode = str(rng.choice(modes))

    if mode == "cmd_error":
        root = bashes[min(1, len(bashes)-1)]
        mag  = float(rng.uniform(0.45, 0.75))
        desc = f"AgentBench-OS:cmd_error — BashNode_2 wrong flags (err={mag:.2f})"
    elif mode == "pipe_error":
        root = pn; mag = float(rng.uniform(0.40, 0.65))
        desc = f"AgentBench-OS:pipe_error — PipelineNode merge error (err={mag:.2f})"
    else:
        root = env; mag = float(rng.uniform(0.50, 0.80))
        desc = f"AgentBench-OS:env_error — EnvSetup wrong env vars (err={mag:.2f})"

    states = {n: _make_state(rng, G.nodes[n]["node_type"],
                              mag if n == root else 0.0)
              for n in G.nodes()}
    errs = _cascade(G, states, root, mag * 0.80 + float(rng.uniform(0.05, 0.15)),
                    GAIN, NOISE, rng)
    return _assemble(G, states, errs, fa, root, desc)


def generate_agentbench_graphs(n: int = 50, seed: int = 42) -> list[AgentCallTree]:

    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 100_000, size=n).tolist()
    out = []
    for s in seeds:
        try:
            out.append(_agentbench_single(int(s)))
        except Exception as exc:
            print(f"  [AgentBench-OS] seed={s} skip: {exc}")
    return out


BENCHMARK_GENERATORS: dict[str, Any] = {
    "SWE-bench":     generate_swe_bench_graphs,
    "WebArena":      generate_webarena_graphs,
    "AgentBench-OS": generate_agentbench_graphs,
}
