"""WM-SAR Region Extraction: GEAF-guided seed → grow → prune.

Core objective (from T2/T4):
    Select region R ⊆ G_f to MINIMISE post-repair amplification ρ(B_{G_f ∖ R})
    subject to a repair budget.

Intuition: Engineering methods (Greedy/Local/Window) reduce error at specific
nodes but leave the COUPLING STRUCTURE intact — ρ(B) barely changes.
WM-SAR selects a connected subgraph that "cuts" the high-amplification path,
actually reducing ρ(B_{G∖R}) and suppressing multi-step error growth.

Algorithm:
    1. SEED  — top-k nodes by e(v) · GEAF_v · (1 + κ_v)
               κ_v = L_A(v) · M_X(v) is the coupling factor (T2: cross-term)
               High κ_v means repairing v also reduces ρ(B) super-additively

    2. GROW  — greedily expand: at each step add neighbor u that
               maximises ΔErrCover(u) + λ₁·Δρ_relief(u) / Cost(u)
               where ρ_relief(u) = ρ(B_{G∖R}) - ρ(B_{G∖(R∪{u})})
               i.e., how much does adding u to the repair further reduce
               post-repair amplification?

    3. PRUNE — remove v from R if it does not lie on any path reaching t_star
               AND its removal does not increase ρ_relief

    4. SCORE — Score(R) = ErrCover(R) · κ̄(R) · ρ_relief(R) / (1 + Cost(R))
               where κ̄(R) = mean coupling factor over R
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from . import amplification as amp
from .failure_graph import node_cost, node_error


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class WMSARConfig:
    H: int = 4                    # spectral walk depth (T3 proxy)
    weight_norm: float = 1.0      # model weight product ≈ ∏_ℓ ‖W_ℓ‖₂
    max_region_size: int = 20     # budget on region size
    n_seeds: int = 6              # number of seed nodes
    lambda1: float = 1.2          # error coverage gain weight
    lambda2: float = 1.5          # ρ-relief gain weight (coupling reduction)
    lambda3: float = 0.1          # cost penalty
    merge_tau: float = 0.5        # Jaccard threshold for region merging
    gamma: float = 0.95           # planning discount factor (T4)
    # ablation flags
    use_geaf: bool = True         # use GEAF in seed scoring (else: error only)
    use_coupling: bool = True     # use coupling factor κ in seeds & growing
    use_growing: bool = True      # grow regions (else: seed only = pointwise GEAF)
    use_pruning: bool = True      # prune non-contributing nodes
    use_rho_relief: bool = True   # growing objective includes ρ-relief (else: error only)

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


# ──────────────────────────────────────────────────────────────────────────────
# Region dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Region:
    nodes: set = field(default_factory=set)
    seed: str = ""
    score: float = 0.0
    rho_relief: float = 0.0
    err_cover: float = 0.0

    def cost(self, G: nx.DiGraph) -> float:
        return float(sum(node_cost(G, v) for v in self.nodes))


# ──────────────────────────────────────────────────────────────────────────────
# WMSAR extractor
# ──────────────────────────────────────────────────────────────────────────────

class WMSAR:
    """WM-SAR region extractor and scorer."""

    def __init__(self, config: WMSARConfig | None = None):
        self.cfg = config or WMSARConfig()
        self._geaf_cache: dict = {}   # node -> GEAF value
        self._kappa_cache: dict = {}  # node -> coupling factor
        self._rho_full: float = 0.0

    def _precompute(self, G: nx.DiGraph) -> None:
        """Pre-compute GEAF and coupling factors for all nodes."""
        c = self.cfg
        self._geaf_cache = amp.geaf_all(G, H=c.H, weight_norm=c.weight_norm)
        self._kappa_cache = {v: amp.coupling_factor(G, v, c.weight_norm)
                              for v in G.nodes()}
        # Baseline amplification before any repair.
        self._rho_full = amp.rho_B(G, set(G.nodes()), c.weight_norm)

    def _rho_relief(self, G: nx.DiGraph, region: set) -> float:
        """ρ_relief(R) = ρ(B_G) - ρ(B_{G∖R}): how much repair reduces amplification."""
        return self._rho_full - amp.rho_B_complement(G, region, self.cfg.weight_norm)

    # ── 1. SEED ──────────────────────────────────────────────────────────────

    def seeds(self, G: nx.DiGraph) -> list:
        c = self.cfg
        t_star = G.graph.get("t_star")
        scored = []
        for v in G.nodes():
            if v == t_star:
                continue
            err = node_error(G, v)
            geaf_v = self._geaf_cache.get(v, 1.0) if c.use_geaf else 1.0
            kappa_v = self._kappa_cache.get(v, 0.0) if c.use_coupling else 0.0
            # Score: error × GEAF (topology amplification) × (1 + coupling boost)
            s = err * max(geaf_v, 1e-9) * (1.0 + kappa_v)
            scored.append((s, v))
        scored.sort(reverse=True)
        return [v for _, v in scored[: c.n_seeds]]

    # ── 2. GROW ──────────────────────────────────────────────────────────────

    def grow(self, G: nx.DiGraph, seed: str) -> set:
        c = self.cfg
        region = {seed}
        if not c.use_growing:
            return region

        und = G.to_undirected(as_view=True)
        t_star = G.graph.get("t_star")

        # Current post-repair amplification
        rho_current = amp.rho_B_complement(G, region, c.weight_norm)

        for _ in range(c.max_region_size - 1):
            # Collect frontier (neighbors not yet in region, not t_star)
            frontier: set = set()
            for r in region:
                frontier.update(und.neighbors(r))
            frontier -= region
            frontier.discard(t_star)
            if not frontier:
                break

            best_u, best_gain = None, -1e9
            for u in sorted(frontier, key=str):
                cand = region | {u}
                # ρ-relief: how much does adding u reduce post-repair amplification?
                rho_cand = amp.rho_B_complement(G, cand, c.weight_norm)
                d_rho_relief = rho_current - rho_cand  # positive = better

                d_err = node_error(G, u) * (1.0 + self._kappa_cache.get(u, 0.0))
                cost = node_cost(G, u)

                if c.use_rho_relief:
                    gain = (c.lambda1 * d_err + c.lambda2 * d_rho_relief) / (c.lambda3 + cost)
                else:
                    gain = (c.lambda1 * d_err) / (c.lambda3 + cost)

                if gain > best_gain:
                    best_gain, best_u = gain, u

            # Only expand if there's positive marginal gain
            if best_u is not None and best_gain > 0.0:
                region.add(best_u)
                rho_current = amp.rho_B_complement(G, region, c.weight_norm)
            else:
                break

        return region

    # ── 3. PRUNE ─────────────────────────────────────────────────────────────

    def prune(self, G: nx.DiGraph, region: set) -> set:
        if not self.cfg.use_pruning or len(region) <= 1:
            return region
        t_star = G.graph.get("t_star")
        # Only keep nodes whose removal increases ρ_complement (i.e., they matter)
        pruned = set(region)
        rho_pruned = amp.rho_B_complement(G, pruned, self.cfg.weight_norm)
        for v in sorted(region, key=str):
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
            # If removing v doesn't increase ρ (no benefit), drop it
            if rho_smaller <= rho_pruned + 1e-6:
                pruned.discard(v)
                rho_pruned = rho_smaller
        return pruned if pruned else region

    # ── 4. SCORE ─────────────────────────────────────────────────────────────

    def score(self, G: nx.DiGraph, region: set) -> float:
        if not region:
            return 0.0
        err_cover = sum(node_error(G, r) for r in region)
        kappa_mean = float(np.mean([self._kappa_cache.get(r, 0.0) for r in region]))
        rho_relief = self._rho_relief(G, region)
        cost = sum(node_cost(G, r) for r in region) + 1.0

        # Score = ErrCover · (1 + κ̄) · ρ_relief / Cost
        # All three factors are desirable:
        #   ErrCover: we want to fix high-error nodes
        #   (1+κ̄):   bonus for nodes that also reduce coupling
        #   ρ_relief: reduction in post-repair amplification (T2/T4 grounding)
        num = err_cover * (1.0 + kappa_mean) * max(rho_relief, 1e-6)
        return float(num / cost)

    # ── Public API ────────────────────────────────────────────────────────────

    def candidate_regions(self, G: nx.DiGraph) -> list[Region]:
        self._precompute(G)
        seed_nodes = self.seeds(G)
        raw: list[set] = []
        for s in seed_nodes:
            r = self.grow(G, s)
            r = self.prune(G, r)
            if r:
                raw.append(r)
        # Merge overlapping regions
        merged = self._merge(raw)
        out = []
        for r in merged:
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
        """Return the highest-score region within budget."""
        regions = self.candidate_regions(G)
        if not regions:
            return set()
        if budget is None:
            return set(regions[0].nodes)
        if budget < 0:
            raise ValueError("budget must be non-negative")

        # The method returns one connected repair region, never a disconnected
        # union of independently grown candidates.
        for region in regions:
            if region.cost(G) <= budget:
                return set(region.nodes)

        # If no complete region fits, return the best affordable singleton.
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
