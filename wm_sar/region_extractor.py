from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
from . import amplification as amp
from .failure_graph import node_cost, node_error


@dataclass
class WMSARConfig:
    H: int = 4
    weight_norm: float = 0.9
    max_region_size: int = 20
    n_seeds: int = 6
    lambda1: float = 1.2
    lambda2: float = 1.5
    lambda3: float = 0.1
    merge_tau: float = 0.5
    gamma: float = 0.95

    use_geaf: bool = True
    use_coupling: bool = True
    use_growing: bool = True
    use_pruning: bool = True
    use_rho_relief: bool = True
    merge_candidates: bool = False

    def __post_init__(self):
        if self.H < 1:
            raise ValueError("H must be at least 1")
        if self.max_region_size < 1:
            raise ValueError("max_region_size must be at least 1")
        if self.n_seeds < 1:
            raise ValueError("n_seeds must be at least 1")
        if self.weight_norm < 0:
            raise ValueError("weight_norm must be non-negative")
        if self.lambda3 <= 0:
            raise ValueError("lambda3 must be positive")
        if not 0 <= self.merge_tau <= 1:
            raise ValueError("merge_tau must be in [0, 1]")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")


@dataclass
class Region:
    nodes: set = field(default_factory=set)
    seed: str = ""
    score: float = 0.0
    rho_relief: float = 0.0
    err_cover: float = 0.0

    def cost(self, G: nx.DiGraph) -> float:
        return float(sum(node_cost(G, v) for v in self.nodes))


class WMSAR:


    def __init__(self, config: WMSARConfig | None = None):
        self.cfg = config or WMSARConfig()
        self._geaf_cache: dict = {}
        self._kappa_cache: dict = {}
        self._rho_full: float = 0.0

    def _precompute(self, G: nx.DiGraph) -> None:

        c = self.cfg
        self._geaf_cache = amp.geaf_all(G, H=c.H, weight_norm=c.weight_norm)
        self._kappa_cache = {v: amp.coupling_factor(G, v, c.weight_norm)
                              for v in G.nodes()}

        self._rho_full = amp.rho_B(G, set(G.nodes()), c.weight_norm)

    def _rho_relief(self, G: nx.DiGraph, region: set) -> float:

        return self._rho_full - amp.rho_B_complement(G, region, self.cfg.weight_norm)


    def seeds(self, G: nx.DiGraph) -> list:
        c = self.cfg
        t_star = G.graph.get("t_star")
        scored = []
        for v in G.nodes():
            if v == t_star:
                continue
            err = node_error(G, v)
            geaf_v = self._geaf_cache.get(v, 0.0)
            kappa_v = self._kappa_cache.get(v, 0.0) if c.use_coupling else 0.0

            s = err * geaf_v * (1.0 + kappa_v) if c.use_geaf else err
            scored.append((s, v))
        scored.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
        return [v for _, v in scored[: c.n_seeds]]


    def grow(self, G: nx.DiGraph, seed: str) -> set:
        c = self.cfg
        region = {seed}
        if not c.use_growing:
            return region

        und = G.to_undirected(as_view=True)
        t_star = G.graph.get("t_star")


        rho_current = amp.rho_B_complement(G, region, c.weight_norm)

        for _ in range(c.max_region_size - 1):

            frontier: set = set()
            for r in region:
                frontier.update(und.neighbors(r))
            frontier -= region
            frontier.discard(t_star)
            if not frontier:
                break

            best_u, best_gain, best_relief = None, float("-inf"), 0.0
            for u in sorted(frontier, key=str):
                cand = region | {u}

                rho_cand = amp.rho_B_complement(G, cand, c.weight_norm)
                d_rho_relief = rho_current - rho_cand

                kappa_u = self._kappa_cache.get(u, 0.0) if c.use_coupling else 0.0
                d_err = node_error(G, u) * (1.0 + kappa_u)
                cost = 1.0 + len(set(und.neighbors(u)) & region) / len(region)

                if c.use_rho_relief:
                    gain = c.lambda1 * d_err + c.lambda2 * d_rho_relief - c.lambda3 * cost
                else:
                    gain = c.lambda1 * d_err - c.lambda3 * cost

                if gain > best_gain:
                    best_gain, best_u, best_relief = gain, u, d_rho_relief


            should_expand = best_relief > 0.0 if c.use_rho_relief else best_gain > 0.0
            if best_u is not None and should_expand:
                region.add(best_u)
                rho_current = amp.rho_B_complement(G, region, c.weight_norm)
            else:
                break

        return region


    def prune(self, G: nx.DiGraph, region: set) -> set:
        if not self.cfg.use_pruning or len(region) <= 1:
            return region
        t_star = G.graph.get("t_star")

        pruned = set(region)
        rho_pruned = amp.rho_B_complement(G, pruned, self.cfg.weight_norm)
        changed = True
        while changed and len(pruned) > 1:
            changed = False
            for v in sorted(pruned, key=str):
                if v == t_star:
                    continue
                smaller = pruned - {v}
                if not smaller:
                    continue
                if len(smaller) > 1 and not nx.is_connected(
                    G.subgraph(smaller).to_undirected()
                ):
                    continue
                rho_smaller = amp.rho_B_complement(G, smaller, self.cfg.weight_norm)

                if rho_smaller <= rho_pruned:
                    pruned.discard(v)
                    rho_pruned = rho_smaller
                    changed = True
        return pruned if pruned else region


    def score(self, G: nx.DiGraph, region: set) -> float:
        if not region:
            return 0.0
        err_cover = sum(
            node_error(G, r) * (
                1.0 + (self._kappa_cache.get(r, 0.0) if self.cfg.use_coupling else 0.0)
            )
            for r in region
        )
        rho_relief = self._rho_relief(G, region)
        cost = len(region) + 1.0

        num = err_cover * rho_relief
        return float(num / cost)


    def candidate_regions(self, G: nx.DiGraph) -> list[Region]:
        self._precompute(G)
        seed_nodes = self.seeds(G)
        raw: list[set] = []
        for s in seed_nodes:
            r = self.grow(G, s)
            r = self.prune(G, r)
            if r:
                raw.append(r)

        merged = self._merge(raw) if self.cfg.merge_candidates else raw
        seen = set()
        out = []
        for r in merged:
            if self.cfg.merge_candidates:
                r = self.prune(G, r)
            key = frozenset(r)
            if key in seen:
                continue
            seen.add(key)
            rr = Region(
                nodes=r,
                score=self.score(G, r),
                rho_relief=self._rho_relief(G, r),
                err_cover=sum(node_error(G, v) for v in r),
            )
            out.append(rr)
        out.sort(key=lambda x: x.score, reverse=True)
        return out

    def repair_region(self, G: nx.DiGraph, budget: float | None = None) -> set:

        regions = self.candidate_regions(G)
        if not regions:
            return set()
        if budget is None:
            return set(regions[0].nodes)
        if budget < 0:
            raise ValueError("budget must be non-negative")


        for region in regions:
            if region.cost(G) <= budget:
                return set(region.nodes)


        affordable = {
            node
            for region in regions
            for node in region.nodes
            if node_cost(G, node) <= budget
        }
        if not affordable:
            return set()
        best = max(
            affordable,
            key=lambda node: (
                node_error(G, node) / max(node_cost(G, node), 1e-12),
                str(node),
            ),
        )
        return {best}

    def _merge(self, regions: list[set]) -> list[set]:
        tau = self.cfg.merge_tau
        merged: list[set] = []
        for r in sorted(regions, key=len, reverse=True):
            placed = False
            for i, m in enumerate(merged):
                inter = len(r & m)
                union = len(r | m)
                if (
                    union
                    and union <= self.cfg.max_region_size
                    and inter / union > tau
                ):
                    merged[i] = m | r
                    placed = True
                    break
            if not placed:
                merged.append(set(r))
        return merged


ReCoreConfig = WMSARConfig
ReCore = WMSAR
