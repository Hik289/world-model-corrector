"""Benchmark-specific failure graphs for WM-SAR.

Three benchmarks modelled after published agent trace structures:

  SWE-bench   (Jimenez et al., 2024) -- GitHub issue resolution
  WebArena    (Zhou et al., 2024)    -- browser-based task agents
  AgentBench-OS (Liu et al., 2024)  -- OS bash-command pipelines

Key design constraint
---------------------
Pure DAGs give ρ(B) ≈ 0 because their adjacency-matrix spectral radii
are zero (no cycles → no eigenvalues > 0).  Real agents have retry /
feedback edges (TestRunner failure → re-invoke CodeAnalyzer;
FormValidator failure → re-navigate; Verifier fail → re-run BashNode).
We add one realistic retry edge per graph to push ρ(B) into the same
[1.2, 2.5] range observed in the agent-calling-tree testbed.

Interface: each generator returns a list[AgentCallTree], identical to
generate_calling_trees() in agent_calling_tree.py.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np

from .agent_calling_tree import (
    AgentCallTree,
    FEAT_DIM,
    NODE_TYPES,
    EDGE_TYPES,
)
from .failure_graph import build_from_agent_calling_tree
from . import amplification as amp


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_state(rng: np.random.Generator,
                node_type: str,
                err_level: float = 0.0) -> np.ndarray:
    """8-dim state vector matching agent_calling_tree.py FEAT_NAMES."""
    s = np.array([
        1.0,                           # 0 activation
        float(rng.uniform(0.1, 0.5)),  # 1 load
        float(rng.uniform(0.05, 0.3)), # 2 latency
        float(rng.uniform(0.0, 0.05)), # 3 error_prob
        float(rng.uniform(0.7, 1.0)),  # 4 throughput
        float(rng.uniform(0.8, 1.0)),  # 5 confidence
        1.0,                           # 6 dependency_ok
        1.0,                           # 7 success_flag
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
    """Topological cascade — same logic as agent_calling_tree.py."""
    errs: dict[str, float] = {n: 0.0 for n in G.nodes()}
    errs[root] = root_err

    # Use BFS order (handles cycles via visited tracking)
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
    """Wrap a raw graph into AgentCallTree (mirrors generate_calling_trees)."""
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


# ──────────────────────────────────────────────────────────────────────────────
# SWE-bench  (GitHub issue resolution)
# ──────────────────────────────────────────────────────────────────────────────

def _swe_single(seed: int) -> AgentCallTree:
    """
    Topology  (N ≈ 15–22):
      IssueAnalyzer (planner)
        ├─► RepoExplorer (executor)
        └─► FileLocator_0..K-1 (executor)  ← K = 3-5
              └─► CodeAnalyzer_i_j (checker)   ← 2-3 per file
      All CodeAnalyzers ─► PatchWriter (executor)
      PatchWriter ─► TestRunner_0..M-1 (validator)  M = 2-3
      TestRunners ─► CIChecker (validator)
      CIChecker   ─► FinalAnswer

    Retry edge (cycle):
      CIChecker ──► IssueAnalyzer    (CI failure triggers re-analysis)

    Cascade gain α = 1.15 (deep fan-out amplifies errors).
    """
    rng  = np.random.default_rng(seed)
    GAIN = 1.15; NOISE = 0.04

    K = int(rng.integers(3, 6))   # 3-5 file locators
    J = int(rng.integers(2, 4))   # 2-3 code analyzers per file
    M = int(rng.integers(2, 4))   # 2-3 test runners

    G = nx.DiGraph()

    # nodes
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

    # ── Retry edge (creates cycle → non-zero ρ(B)) ──
    G.add_edge(ci, ia, edge_type="errors")   # CI fail → re-analyse issue

    # failure mode
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
    """N SWE-bench failure graphs (15-22 nodes, retry-loop, α=1.15)."""
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 100_000, size=n).tolist()
    out = []
    for s in seeds:
        try:
            out.append(_swe_single(int(s)))
        except Exception as exc:
            print(f"  [SWE-bench] seed={s} skip: {exc}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# WebArena  (browser-based task agents)
# ──────────────────────────────────────────────────────────────────────────────

def _webarena_single(seed: int) -> AgentCallTree:
    """
    Topology (N ≈ 9–15, sequential):
      TaskPlanner (planner)
        └─► Navigator_1 (executor) ─► PageReader_1 (checker) ─► ContentExtractor_1 (aggregator)
        └─► Navigator_2 (executor) ─► PageReader_2 (checker) ─► ContentExtractor_2 (aggregator)
        ...
      All ContentExtractors ─► FormFiller (executor)
      FormFiller ─► FormValidator (validator)
      FormValidator ─► Submitter (executor)
      Submitter ─► SuccessChecker (validator)
      SuccessChecker ─► FinalAnswer

    Navigators are chained sequentially (N1→N2→N3).

    Retry edge (cycle):
      FormValidator ──► Navigator_1   (validation fail → re-navigate)

    Cascade gain α = 1.08 (linear; errors propagate but don't fan out).
    """
    rng  = np.random.default_rng(seed)
    GAIN = 1.08; NOISE = 0.03

    hops = int(rng.integers(2, 5))   # 2-4 nav hops

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

    # ── Retry edge ──
    G.add_edge(fv, navs[0], edge_type="errors")   # validation fail → re-navigate

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
    """N WebArena failure graphs (9-15 nodes, retry-loop, α=1.08)."""
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 100_000, size=n).tolist()
    out = []
    for s in seeds:
        try:
            out.append(_webarena_single(int(s)))
        except Exception as exc:
            print(f"  [WebArena] seed={s} skip: {exc}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# AgentBench-OS  (bash pipeline)
# ──────────────────────────────────────────────────────────────────────────────

def _agentbench_single(seed: int) -> AgentCallTree:
    """
    Topology (N ≈ 8–12):
      Commander (planner)
        [EnvSetup (executor)]   ← 60 % of instances
        └─► BashNode_1 (executor) ─► OutputParser_1 (checker)
        └─► BashNode_2 (executor) ─► OutputParser_2 (checker)   [B1 → B2 pipe]
        ...
      All OutputParsers ─► PipelineNode (aggregator)
      PipelineNode ─► Verifier (validator)
      Verifier ─► FinalAnswer

    Retry edge (cycle):
      Verifier ──► Commander   (verification fail → re-plan)

    Cascade gain α = 1.12.
    """
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

    # ── Retry edge ──
    G.add_edge(vf, cmd, edge_type="errors")   # verifier fail → re-plan

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
    """N AgentBench-OS failure graphs (8-12 nodes, retry-loop, α=1.12)."""
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 100_000, size=n).tolist()
    out = []
    for s in seeds:
        try:
            out.append(_agentbench_single(int(s)))
        except Exception as exc:
            print(f"  [AgentBench-OS] seed={s} skip: {exc}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Combined interface
# ──────────────────────────────────────────────────────────────────────────────

BENCHMARK_GENERATORS: dict[str, Any] = {
    "SWE-bench":     generate_swe_bench_graphs,
    "WebArena":      generate_webarena_graphs,
    "AgentBench-OS": generate_agentbench_graphs,
}
