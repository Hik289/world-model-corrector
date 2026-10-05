from __future__ import annotations

import networkx as nx
import numpy as np
from .failure_graph import node_error


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
    ev = np.linalg.eigvals(M)
    return float(np.max(np.abs(ev)))


def target_reachable(G: nx.DiGraph, t_star: str | None = None) -> set:

    t_star = t_star or G.graph.get("t_star")
    if t_star is None or t_star not in G:
        return set(G.nodes())
    anc = nx.ancestors(G, t_star)
    anc.add(t_star)
    return anc


def geaf_node(G: nx.DiGraph, v: str, H: int = 4, weight_norm: float = 0.9) -> float:


    err = node_error(G, v)

    local_nodes = nx.single_source_shortest_path_length(
        G.to_undirected(as_view=True), v, cutoff=H
    )
    A_loc, _, _ = adjacency_matrix(G.subgraph(local_nodes))
    rho_local = _spectral_radius(A_loc)
    return float(err * rho_local * (weight_norm ** H))


def geaf_all(G: nx.DiGraph, H: int = 4, weight_norm: float = 0.9) -> dict:

    return {v: geaf_node(G, v, H, weight_norm) for v in G.nodes()}


def geaf_global(G: nx.DiGraph, H: int = 4, weight_norm: float = 0.9) -> float:

    reach = target_reachable(G)
    g = geaf_all(G, H, weight_norm)
    return float(sum(g[v] for v in reach))


def _estimate_propagation_gains(G: nx.DiGraph, weight_norm: float = 0.9
                                ) -> tuple[float, float, float, float]:
    return coupling_blocks_region(G, set(G.nodes()), weight_norm)


def coupling_blocks_region(G: nx.DiGraph, region: set, weight_norm: float = 0.9
                            ) -> tuple[float, float, float, float]:


    region = {r for r in region if G.has_node(r)}
    if not region:
        return 0.0, 0.0, 0.0, 0.0

    sub = G.subgraph(region)
    ridx = {r: i for i, r in enumerate(region)}
    m = len(region)


    A_R = np.zeros((m, m))
    for u, v in sub.edges():
        A_R[ridx[u], ridx[v]] = 1.0
    L_X = weight_norm * _spectral_radius(A_R)


    edge_types_in: dict[str, set] = {r: set() for r in region}
    edge_types_out: dict[str, set] = {r: set() for r in region}
    for u, v, d in sub.edges(data=True):
        etype = d.get("edge_type", "default")
        edge_types_out[u].add(etype)
        edge_types_in[v].add(etype)


    avg_in_div = float(np.mean([len(s) for s in edge_types_in.values()])) if m > 0 else 0.0
    avg_out_div = float(np.mean([len(s) for s in edge_types_out.values()])) if m > 0 else 0.0
    errs = np.array([node_error(G, r) for r in region])
    mean_err = float(np.mean(errs))

    L_A = weight_norm * avg_in_div * mean_err * 0.3
    M_X = weight_norm * avg_out_div * mean_err * 0.2


    n_sub_edges = sub.number_of_edges()
    if n_sub_edges > 0:
        high_err_set = set(r for r in region if node_error(G, r) > mean_err)
        both_high = sum(1 for u, v in sub.edges() if u in high_err_set and v in high_err_set)
        M_A = weight_norm * (both_high / n_sub_edges) * 0.5
    else:
        M_A = 0.0

    return float(L_X), float(L_A), float(M_X), float(M_A)


def rho_B_from_blocks(L_X: float, L_A: float, M_X: float, M_A: float) -> float:

    disc = (L_X - M_A) ** 2 + 4.0 * L_A * M_X
    return 0.5 * (L_X + M_A + np.sqrt(max(disc, 0.0)))


def coupling_factor(G: nx.DiGraph, v: str, weight_norm: float = 0.9) -> float:


    local = set(G.predecessors(v)) | set(G.successors(v)) | {v}
    _, L_A, M_X, _ = coupling_blocks_region(G, local, weight_norm)
    return float(L_A * M_X)


def rho_B(G: nx.DiGraph, region: set, weight_norm: float = 0.9) -> float:

    L_X, L_A, M_X, M_A = coupling_blocks_region(G, region, weight_norm)
    return rho_B_from_blocks(L_X, L_A, M_X, M_A)


def rho_B_complement(G: nx.DiGraph, region: set, weight_norm: float = 0.9) -> float:


    complement = set(G.nodes()) - region
    if not complement:
        return 0.0
    return rho_B(G, complement, weight_norm)


def coupled_error_envelope(z0, B, epsilon, H: int) -> np.ndarray:
    if not isinstance(H, (int, np.integer)) or H < 0:
        raise ValueError("H must be a non-negative integer")
    state = np.asarray(z0, dtype=float)
    operator = np.asarray(B, dtype=float)
    forcing = np.asarray(epsilon, dtype=float)
    if state.shape != (2,) or forcing.shape != (2,) or operator.shape != (2, 2):
        raise ValueError("z0 and epsilon must have shape (2,), and B must have shape (2, 2)")
    if any(not np.all(np.isfinite(value)) or np.any(value < 0)
           for value in (state, operator, forcing)):
        raise ValueError("z0, B, and epsilon must be finite and non-negative")
    envelope = np.empty((H + 1, 2), dtype=float)
    envelope[0] = state
    for step in range(H):
        envelope[step + 1] = operator @ envelope[step] + forcing
    return envelope


def phi_H_regret(H: int, gamma: float, rho: float) -> float:
    if H < 0:
        raise ValueError("H must be non-negative")
    if gamma < 0 or rho < 0:
        raise ValueError("gamma and rho must be non-negative")
    factor = gamma * rho
    total = 0.0
    term = 1.0
    for _ in range(H):
        total += term
        term *= factor
    return float(total)


def return_error_bound(G: nx.DiGraph, region: set, H: int = 8,
                       gamma: float = 0.95, L_R: float = 1.0,
                       kappa: float = 1.0, epsilon: float = None,
                       weight_norm: float = 0.9,
                       epsilon_R: float | None = None) -> dict:


    L_X, L_A, M_X, M_A = _estimate_propagation_gains(G, weight_norm)
    rho_pre = rho_B_from_blocks(L_X, L_A, M_X, M_A)
    if epsilon is None:

        errs = [node_error(G, v) for v in G.nodes()]
        epsilon = float(np.mean(errs)) if errs else 0.0

    phi_pre = phi_H_regret(H, gamma, rho_pre)


    rho_post = rho_B_complement(G, region, weight_norm)
    phi_post = phi_H_regret(H, gamma, rho_post)

    if epsilon_R is None:
        epsilon_R = L_R * epsilon

    bound_pre = 2 * L_R * kappa * epsilon * phi_pre + 2 * epsilon_R * H
    bound_post = 2 * L_R * kappa * epsilon * phi_post + 2 * epsilon_R * H

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


def simulate_error_propagation(G: nx.DiGraph, repaired: set,
                                H: int = 32, weight_norm: float = 0.9
                                ) -> dict[int, float]:


    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return {h: 0.0 for h in range(1, H + 1)}

    A, _, _ = adjacency_matrix(G)


    e0 = np.array([node_error(G, v) if v not in repaired else 0.0 for v in nodes])


    L_X, L_A, M_X, M_A = _estimate_propagation_gains(G, weight_norm)

    result = {}
    e_k = e0.copy()
    for h in range(1, H + 1):
        e_next = L_X * e_k + L_A * (A.T @ e_k)

        e_next = np.clip(e_next, 0.0, 10.0 * (float(np.max(e0)) + 1e-9))
        e_k = e_next
        result[h] = float(np.mean(e_k ** 2))
    return result


def error_growth_slope(mse_dict: dict[int, float],
                        h_start: int = 4, h_end: int = 32) -> float:


    hs = sorted(h for h in mse_dict if h_start <= h <= h_end)
    if len(hs) < 2:
        return 0.0
    vals = np.array([np.log(max(mse_dict[h], 1e-15)) for h in hs])
    hs_arr = np.array(hs, dtype=float)
    if hs_arr.std() < 1e-9:
        return 0.0
    slope = float(np.polyfit(hs_arr, vals, 1)[0])
    return slope


def phi_H(G: nx.DiGraph, H: int = 4, weight_norm: float = 0.9) -> dict:

    A, nodes, idx = adjacency_matrix(G)
    ones = np.ones(len(nodes))
    field = np.zeros(len(nodes))
    Ak = np.eye(len(nodes))
    for k in range(1, H + 1):
        Ak = Ak @ A
        field += (weight_norm ** k) * (Ak @ ones)
    return {nodes[i]: float(field[i]) for i in range(len(nodes))}


def phi_H_target(G: nx.DiGraph, H: int = 4, weight_norm: float = 0.9,
                  t_star: str | None = None) -> dict:

    base = phi_H(G, H, weight_norm)
    reach = target_reachable(G, t_star)
    return {v: (base[v] if v in reach else 0.0) for v in base}


def phi_H_edge(G: nx.DiGraph, node_field: dict, alpha: float = 0.5) -> dict:

    try:
        bridge = nx.edge_betweenness_centrality(G)
    except Exception:
        bridge = {e: 0.0 for e in G.edges()}
    out = {}
    for u, v in G.edges():
        out[(u, v)] = 0.5 * (node_field.get(u, 0.0) + node_field.get(v, 0.0)) + \
                      alpha * bridge.get((u, v), 0.0)
    return out


def GEAF(G: nx.DiGraph, H: int = 4, weight_norm: float = 0.9) -> float:

    return geaf_global(G, H, weight_norm)


def error_slope(G: nx.DiGraph) -> float:

    ts, es = [], []
    for v, d in G.nodes(data=True):
        ts.append(float(d.get("time_step", 0)))
        es.append(node_error(G, v))
    ts, es = np.asarray(ts), np.asarray(es)
    if len(ts) < 2 or ts.std() < 1e-9:
        return 0.0
    return float(np.polyfit(ts, es, 1)[0])


def target_amplify(G: nx.DiGraph, region: set, H: int = 4,
                   weight_norm: float = 0.9, edge_field=None) -> float:

    tfield = phi_H_target(G, H, weight_norm)
    val = sum(tfield.get(v, 0.0) for v in region)
    if edge_field is None:
        edge_field = phi_H_edge(G, tfield)
    sub = G.subgraph([r for r in region if G.has_node(r)])
    val += sum(edge_field.get((u, v), 0.0) for u, v in sub.edges())
    return float(val)


def global_rho_B(G: nx.DiGraph, weight_norm: float = 0.9) -> float:
    return rho_B(G, target_reachable(G), weight_norm)


def spectral_summary(G: nx.DiGraph, H: int = 4, weight_norm: float = 0.9) -> dict:
    return {
        "GEAF": GEAF(G, H, weight_norm),
        "rho_B": global_rho_B(G, weight_norm),
        "error_slope": error_slope(G),
        "target_amplify": target_amplify(G, target_reachable(G), H, weight_norm),
    }


def coupling_blocks(G: nx.DiGraph, region: set, weight_norm: float = 0.9):

    return coupling_blocks_region(G, region, weight_norm)
