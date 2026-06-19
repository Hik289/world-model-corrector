"""Graph Error Amplification Field (GEAF) and joint node-edge coupling operator B.

Theory from analysis/main.tex (Theorems T1-T4):

    T1  Fixed-edge rollout error:
            e^X_k ≤ L_X^k · e_0  +  ε_X (L_X^k - 1)/(L_X - 1)
            L_X = L_σ · ρ(A) · ∏_ℓ ‖W_ℓ‖₂  ≡  GEAF_v (per-node proxy)

    T2  Dynamic-edge coupled operator:
            u_{k+1} ≼ B u_k + ε,
            B = [[L_X, L_A], [M_X, M_A]]
            ρ(B) = ½[(L_X + M_A) + √((L_X - M_A)² + 4 L_A M_X)]
            When L_A M_X > 0: ρ(B) > max(L_X, M_A)  ← node-edge coupling

    T3  GEAF as proxy: ρ(B) ≤ GEAF · (1 + R_X/‖A‖₂) ≤ 2·GEAF

    T4  Planning regret: super-linear growth when γ·ρ(B) > 1
            Φ_H(γ,ρ) = (1/(ρ-1)) · [(1-(γρ)^H)/(1-γρ) - (1-γ^H)/(1-γ)]

In a *failure graph*, we do not have GWM weights W_ℓ explicitly.  We estimate:
    ρ(A)    from the failure-graph local adjacency
    ∏‖W_ℓ‖  from the observed error-amplification ratio between steps
    L_A, M_X from edge-to-node / node-to-edge error covariation

The repair objective: choose region R ⊆ G_f to minimise ρ(B_{G_f ∖ R}),
i.e. reduce post-repair amplification, not just pointwise error.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from .failure_graph import node_error


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _node_index(G: nx.DiGraph) -> tuple[list, dict]:
    nodes = list(G.nodes())
    return nodes, {n: i for i, n in enumerate(nodes)}


def adjacency_matrix(G: nx.DiGraph) -> tuple[np.ndarray, list, dict]:
    nodes, idx = _node_index(G)
    n = len(nodes)
    A = np.zeros((n, n))
    for u, v in G.edges():
        A[idx[u], idx[v]] = 1.0
    return A, nodes, idx


def _spectral_radius(M: np.ndarray) -> float:
    if M.size == 0:
        return 0.0
    try:
        ev = np.linalg.eigvals(M)
        return float(np.max(np.abs(ev)))
    except Exception:
        return float(np.abs(M).max())


def target_reachable(G: nx.DiGraph, t_star: str | None = None) -> set:
    """Ancestors of t_star (inclusive) — nodes whose errors can propagate to failure."""
    t_star = t_star or G.graph.get("t_star")
    if t_star is None or t_star not in G:
        return set(G.nodes())
    anc = nx.ancestors(G, t_star)
    anc.add(t_star)
    return anc


# ──────────────────────────────────────────────────────────────────────────────
# Per-node GEAF  (T1 / T3 proxy)
# ──────────────────────────────────────────────────────────────────────────────

def geaf_node(G: nx.DiGraph, v: str, H: int = 4, weight_norm: float = 1.0) -> float:
    """Per-node GEAF proxy: e(v) · ρ(A_local) · weight_norm^H

    ρ(A_local) is the spectral radius of the H-hop subgraph rooted at v,
    capturing how broadly an error at v can spread.
    weight_norm ≈ ∏_ℓ ‖W_ℓ‖₂ (estimated from rollout error ratio or set = 1).
    """
    err = node_error(G, v)
    # collect H-hop successors
    local_nodes = {v}
    frontier = {v}
    for _ in range(H):
        nxt = set()
        for u in frontier:
            nxt.update(G.successors(u))
        frontier = nxt - local_nodes
        local_nodes |= nxt
        if not frontier:
            break
    sub_nodes = [n for n in local_nodes if G.has_node(n)]
    if len(sub_nodes) < 2:
        # isolated node: amplification = 1 (no topology)
        rho_local = 1.0
    else:
        idx = {n: i for i, n in enumerate(sub_nodes)}
        m = len(sub_nodes)
        A_loc = np.zeros((m, m))
        sub = G.subgraph(sub_nodes)
        for u, w in sub.edges():
            A_loc[idx[u], idx[w]] = 1.0
        rho_local = _spectral_radius(A_loc)
    return float(err * max(rho_local, 1.0) * (weight_norm ** H))


def geaf_all(G: nx.DiGraph, H: int = 4, weight_norm: float = 1.0) -> dict:
    """GEAF for every node in G."""
    return {v: geaf_node(G, v, H, weight_norm) for v in G.nodes()}


def geaf_global(G: nx.DiGraph, H: int = 4, weight_norm: float = 1.0) -> float:
    """Global GEAF = sum_v GEAF_v over target-reachable nodes (T3)."""
    reach = target_reachable(G)
    g = geaf_all(G, H, weight_norm)
    return float(sum(g[v] for v in reach))


# ──────────────────────────────────────────────────────────────────────────────
# Estimate propagation gains from failure graph structure
# ──────────────────────────────────────────────────────────────────────────────

def _estimate_propagation_gains(G: nx.DiGraph, weight_norm: float = 1.0
                                ) -> tuple[float, float, float, float]:
    """Estimate L_X, L_A, M_X, M_A from failure graph edge patterns.

    L_X: node-to-node error gain along directed edges (spectral, T1)
    L_A: edge-to-node coupling — how edge-type diversity affects next-node error
    M_X: node-to-edge coupling — how node error drives edge-type divergence
    M_A: edge-to-edge persistence — how edge error persists across hops

    These are *global estimates* over the whole failure graph.
    For a region R we re-estimate locally (see coupling_blocks_region).
    """
    nodes, idx = _node_index(G)
    n = len(nodes)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0

    A, _, _ = adjacency_matrix(G)
    # L_X: spectral radius of adjacency * weight_norm (T1 formula)
    L_X = weight_norm * _spectral_radius(A)

    # Count distinct edge types per node (proxy for edge diversity)
    edge_types_out: dict[str, set] = {v: set() for v in G.nodes()}
    edge_types_in: dict[str, set] = {v: set() for v in G.nodes()}
    for u, v, data in G.edges(data=True):
        etype = data.get("edge_type", "default")
        edge_types_out[u].add(etype)
        edge_types_in[v].add(etype)

    avg_out_diversity = float(np.mean([len(s) for s in edge_types_out.values()])) if n > 0 else 0.0
    avg_in_diversity = float(np.mean([len(s) for s in edge_types_in.values()])) if n > 0 else 0.0

    errs = np.array([node_error(G, v) for v in G.nodes()])
    mean_err = float(np.mean(errs)) + 1e-9

    # L_A: edge diversity * mean error → how edge structure steers node error
    #   High L_A = edges can re-route error toward high-error nodes
    L_A = weight_norm * avg_in_diversity * mean_err * 0.3

    # M_X: mean error * out-degree diversity → node error creates new causal edges
    M_X = weight_norm * avg_out_diversity * mean_err * 0.2

    # M_A: fraction of edges whose endpoint is a high-error node (edge error persists)
    high_err = set(v for v in G.nodes() if node_error(G, v) > mean_err)
    n_edges = G.number_of_edges()
    if n_edges > 0:
        high_err_edges = sum(1 for u, v in G.edges() if u in high_err or v in high_err)
        M_A = weight_norm * (high_err_edges / n_edges) * 0.5
    else:
        M_A = 0.0

    return float(L_X), float(L_A), float(M_X), float(M_A)


def coupling_blocks_region(G: nx.DiGraph, region: set, weight_norm: float = 1.0
                            ) -> tuple[float, float, float, float]:
    """Estimate B-matrix blocks (L_X, L_A, M_X, M_A) for a specific region R.

    This is the LOCAL version of T2: the 2x2 operator governing error
    amplification *within* the region and its direct couplings to the rest.
    """
    region = {r for r in region if G.has_node(r)}
    if not region:
        return 0.0, 0.0, 0.0, 0.0

    sub = G.subgraph(region)
    ridx = {r: i for i, r in enumerate(region)}
    m = len(region)

    # L_X: spectral radius of region adjacency * weight_norm
    A_R = np.zeros((m, m))
    for u, v in sub.edges():
        A_R[ridx[u], ridx[v]] = 1.0
    L_X = weight_norm * _spectral_radius(A_R)

    # Edge type diversity within region
    edge_types_in: dict[str, set] = {r: set() for r in region}
    edge_types_out: dict[str, set] = {r: set() for r in region}
    for u, v, d in sub.edges(data=True):
        etype = d.get("edge_type", "default")
        edge_types_out[u].add(etype)
        edge_types_in[v].add(etype)

    # Also count BOUNDARY edges (from outside into region and vice versa)
    for u, v, d in G.edges(data=True):
        etype = d.get("edge_type", "default")
        if u not in region and v in region:
            edge_types_in[v].add(etype)
        if u in region and v not in region:
            edge_types_out[u].add(etype)

    avg_in_div = float(np.mean([len(s) for s in edge_types_in.values()])) if m > 0 else 0.0
    avg_out_div = float(np.mean([len(s) for s in edge_types_out.values()])) if m > 0 else 0.0
    errs = np.array([node_error(G, r) for r in region])
    mean_err = float(np.mean(errs)) + 1e-9

    L_A = weight_norm * avg_in_div * mean_err * 0.3
    M_X = weight_norm * avg_out_div * mean_err * 0.2

    # M_A: fraction of edges whose both endpoints are high-error nodes
    n_sub_edges = sub.number_of_edges()
    if n_sub_edges > 0:
        high_err_set = set(r for r in region if node_error(G, r) > mean_err)
        both_high = sum(1 for u, v in sub.edges() if u in high_err_set and v in high_err_set)
        M_A = weight_norm * (both_high / n_sub_edges) * 0.5
    else:
        M_A = 0.0

    return float(L_X), float(L_A), float(M_X), float(M_A)


# ──────────────────────────────────────────────────────────────────────────────
# ρ(B) — Theorem T2 closed form
# ──────────────────────────────────────────────────────────────────────────────

def rho_B_from_blocks(L_X: float, L_A: float, M_X: float, M_A: float) -> float:
    """T2 closed form: ρ(B) = ½[(L_X+M_A) + √((L_X-M_A)²+4·L_A·M_X)]"""
    disc = (L_X - M_A) ** 2 + 4.0 * L_A * M_X
    return 0.5 * (L_X + M_A + np.sqrt(max(disc, 0.0)))


def coupling_factor(G: nx.DiGraph, v: str, weight_norm: float = 1.0) -> float:
    """Per-node coupling factor κ_v = L_A(v) · M_X(v).

    When κ_v > 0, ρ(B) > max(L_X, M_A) (T2 super-additivity).
    High-κ nodes are priority repair targets since they activate cross-coupling.
    """
    # Estimate coupling blocks in the 1-hop local region of v
    local = set(G.predecessors(v)) | set(G.successors(v)) | {v}
    _, L_A, M_X, _ = coupling_blocks_region(G, local, weight_norm)
    return float(L_A * M_X)


def rho_B(G: nx.DiGraph, region: set, weight_norm: float = 1.0) -> float:
    """ρ(B_R): coupled amplification over region R (T2)."""
    L_X, L_A, M_X, M_A = coupling_blocks_region(G, region, weight_norm)
    return rho_B_from_blocks(L_X, L_A, M_X, M_A)


def rho_B_complement(G: nx.DiGraph, region: set, weight_norm: float = 1.0) -> float:
    """ρ(B_{G\\R}): amplification of the REMAINING graph after removing region R.

    This is what WM-SAR minimises: post-repair residual amplification.
    """
    complement = set(G.nodes()) - region
    if not complement:
        return 0.0
    return rho_B(G, complement, weight_norm)


# ──────────────────────────────────────────────────────────────────────────────
# Φ_H — T4 planning regret scale function
# ──────────────────────────────────────────────────────────────────────────────

def phi_H_regret(H: int, gamma: float, rho: float) -> float:
    """Φ_H(γ,ρ) from Theorem T4.

        Φ_H = (1/(ρ-1)) · [(1-(γρ)^H)/(1-γρ) - (1-γ^H)/(1-γ)]

    Quantifies how rollout error at spectral radius ρ translates to planning
    regret over horizon H with discount γ.
    When γ·ρ > 1, Φ_H grows exponentially in H as (γρ)^H.
    """
    eps = 1e-9
    if abs(rho - 1.0) < eps:
        # L'Hopital at ρ=1
        if abs(gamma - 1.0) < eps:
            # Double limit at ρ=1, γ=1: Phi_H → H(H-1)/2
            return float(H * (H - 1) / 2.0)
        # Single limit at ρ=1, γ≠1 (closed form via L'Hopital):
        #   Phi_H = H * γ^H / (γ-1) + γ * (1 - γ^H) / (γ-1)^2
        gm1 = gamma - 1.0
        gH = gamma ** H
        return float(H * gH / gm1 + gamma * (1.0 - gH) / (gm1 * gm1))
    t1 = (1 - (gamma * rho) ** H) / (1 - gamma * rho + eps) if abs(1 - gamma * rho) > eps else H
    t2 = (1 - gamma ** H) / (1 - gamma + eps) if abs(1 - gamma) > eps else H
    return float((t1 - t2) / (rho - 1.0))


def return_error_bound(G: nx.DiGraph, region: set, H: int = 8,
                       gamma: float = 0.95, L_R: float = 1.0,
                       kappa: float = 1.0, epsilon: float = None,
                       weight_norm: float = 1.0) -> dict:
    """Return error bound from T4 before and after repairing region R.

    Returns dict with pre/post rho, Phi_H, and the regret bound.
    """
    # Pre-repair: full graph amplification
    L_X, L_A, M_X, M_A = _estimate_propagation_gains(G, weight_norm)
    rho_pre = rho_B_from_blocks(L_X, L_A, M_X, M_A)
    if epsilon is None:
        # mean one-step error as proxy for ε
        errs = [node_error(G, v) for v in G.nodes()]
        epsilon = float(np.mean(errs)) + 1e-9

    phi_pre = phi_H_regret(H, gamma, rho_pre)

    # Post-repair: complement graph amplification
    rho_post = rho_B_complement(G, region, weight_norm)
    phi_post = phi_H_regret(H, gamma, rho_post)

    epsilon_R = 2 * L_R * epsilon  # upper bound on reward model error

    bound_pre = 2 * L_R * kappa * epsilon * phi_pre + epsilon_R * H
    bound_post = 2 * L_R * kappa * epsilon * phi_post + epsilon_R * H

    return {
        "rho_pre": rho_pre,
        "rho_post": rho_post,
        "phi_pre": phi_pre,
        "phi_post": phi_post,
        "bound_pre": bound_pre,
        "bound_post": bound_post,
        "regret_reduction": bound_pre - bound_post,
        "rho_reduction": rho_pre - rho_post,
        "super_linear": gamma * rho_pre > 1.0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Multi-step error simulation (for NodeMSE@H evaluation)
# ──────────────────────────────────────────────────────────────────────────────

def simulate_error_propagation(G: nx.DiGraph, repaired: set,
                                H: int = 32, weight_norm: float = 1.0
                                ) -> dict[int, float]:
    """Simulate forward error propagation for H steps after repairing `repaired`.

    After repair: errors in `repaired` are set to 0. Remaining errors propagate
    via the failure graph structure using the T1 recursion:
        e_k(v) = L_X · e_{k-1}(v) + L_A · Σ_{u→v} e_{k-1}(u)

    Returns dict {horizon: NodeMSE@horizon}.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return {h: 0.0 for h in range(1, H + 1)}

    idx = {v: i for i, v in enumerate(nodes)}
    A, _, _ = adjacency_matrix(G)

    # Initial error vector (post-repair)
    e0 = np.array([node_error(G, v) if v not in repaired else 0.0 for v in nodes])

    # T1 recursion with weight_norm as amplification factor
    L_X, L_A, M_X, M_A = _estimate_propagation_gains(G, weight_norm)
    # Per-step propagation: e_{k+1}(v) ≤ L_X · e_k(v) + L_A · (A.T @ e_k)(v)
    result = {}
    e_k = e0.copy()
    for h in range(1, H + 1):
        e_next = L_X * e_k + L_A * (A.T @ e_k)
        # clip to prevent divergence (physical bound)
        e_next = np.clip(e_next, 0.0, 10.0 * (float(np.max(e0)) + 1e-9))
        e_k = e_next
        result[h] = float(np.mean(e_k ** 2))  # NodeMSE@h
    return result


def error_growth_slope(mse_dict: dict[int, float],
                        h_start: int = 4, h_end: int = 32) -> float:
    """GrowthSlope: slope of log(NodeMSE) from h_start to h_end.

    Corresponds to d(log e_k)/dk → log(L_X) in T1 (Corollary in analysis).
    Positive slope = error growing; negative = contracting.
    """
    hs = sorted(h for h in mse_dict if h_start <= h <= h_end)
    if len(hs) < 2:
        return 0.0
    vals = np.array([np.log(max(mse_dict[h], 1e-15)) for h in hs])
    hs_arr = np.array(hs, dtype=float)
    if hs_arr.std() < 1e-9:
        return 0.0
    slope = float(np.polyfit(hs_arr, vals, 1)[0])
    return slope


# ──────────────────────────────────────────────────────────────────────────────
# Legacy wrappers (kept for backward compatibility)
# ──────────────────────────────────────────────────────────────────────────────

def phi_H(G: nx.DiGraph, H: int = 4, weight_norm: float = 1.0) -> dict:
    """H-step node amplification field (legacy: phi_H(v) as walk count × gain)."""
    A, nodes, idx = adjacency_matrix(G)
    ones = np.ones(len(nodes))
    field = np.zeros(len(nodes))
    Ak = np.eye(len(nodes))
    for k in range(1, H + 1):
        Ak = Ak @ A
        field += (weight_norm ** k) * (Ak @ ones)
    return {nodes[i]: float(field[i]) for i in range(len(nodes))}


def phi_H_target(G: nx.DiGraph, H: int = 4, weight_norm: float = 1.0,
                  t_star: str | None = None) -> dict:
    """Target-aware amplification field (legacy)."""
    base = phi_H(G, H, weight_norm)
    reach = target_reachable(G, t_star)
    return {v: (base[v] if v in reach else 0.0) for v in base}


def phi_H_edge(G: nx.DiGraph, node_field: dict, alpha: float = 0.5) -> dict:
    """Edge amplification (legacy)."""
    try:
        bridge = nx.edge_betweenness_centrality(G)
    except Exception:
        bridge = {e: 0.0 for e in G.edges()}
    out = {}
    for u, v in G.edges():
        out[(u, v)] = 0.5 * (node_field.get(u, 0.0) + node_field.get(v, 0.0)) + \
                      alpha * bridge.get((u, v), 0.0)
    return out


def GEAF(G: nx.DiGraph, H: int = 4, weight_norm: float = 1.0) -> float:
    """Global GEAF (legacy wrapper → geaf_global)."""
    return geaf_global(G, H, weight_norm)


def error_slope(G: nx.DiGraph) -> float:
    """Linear slope of node error vs time step (legacy)."""
    ts, es = [], []
    for v, d in G.nodes(data=True):
        ts.append(float(d.get("time_step", 0)))
        es.append(node_error(G, v))
    ts, es = np.asarray(ts), np.asarray(es)
    if len(ts) < 2 or ts.std() < 1e-9:
        return 0.0
    return float(np.polyfit(ts, es, 1)[0])


def target_amplify(G: nx.DiGraph, region: set, H: int = 4,
                   weight_norm: float = 1.0, edge_field=None) -> float:
    """TargetAmplify(R) (legacy)."""
    tfield = phi_H_target(G, H, weight_norm)
    val = sum(tfield.get(v, 0.0) for v in region)
    if edge_field is None:
        edge_field = phi_H_edge(G, tfield)
    sub = G.subgraph([r for r in region if G.has_node(r)])
    val += sum(edge_field.get((u, v), 0.0) for u, v in sub.edges())
    return float(val)


def global_rho_B(G: nx.DiGraph, weight_norm: float = 1.0) -> float:
    return rho_B(G, target_reachable(G), weight_norm)


def spectral_summary(G: nx.DiGraph, H: int = 4, weight_norm: float = 1.0) -> dict:
    return {
        "GEAF": GEAF(G, H, weight_norm),
        "rho_B": global_rho_B(G, weight_norm),
        "error_slope": error_slope(G),
        "target_amplify": target_amplify(G, target_reachable(G), H, weight_norm),
    }


def coupling_blocks(G: nx.DiGraph, region: set, weight_norm: float = 1.0):
    """Legacy wrapper."""
    return coupling_blocks_region(G, region, weight_norm)
