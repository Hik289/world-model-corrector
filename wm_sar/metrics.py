"""Evaluation metrics and the per-method aggregating evaluator.

Metrics (Section 13):
    Recovery Success Rate
    Cost-Normalized Recovery   (recovery rate per 1k tokens)
    Tokens per Recovery
    Downstream Error Reduction
    Propagation Depth Reduction
    Region Localization (IoU vs. ground-truth corrupted region)
    Local inconsistency after repair
    Latency
    Spectral deltas: dGEAF, d rho(B), d ErrorSlope, d TargetAmplify
"""

from __future__ import annotations

from typing import Callable

import networkx as nx
import numpy as np

from . import amplification as amp
from .baselines import RepairPlan
from .repair_executor import apply_repair, measure_recovery


def spectral_deltas(G: nx.DiGraph, repaired: set[str], H: int = 4) -> dict:
    """Before/after spectral dynamics reduction for one graph + repair."""
    before = amp.spectral_summary(G, H)
    Grep = apply_repair(G, set(repaired))
    after = amp.spectral_summary(Grep, H)
    reach = amp.target_reachable(G)
    rho_before = amp.rho_B(G, reach)
    rho_after = amp.rho_B(Grep, reach)
    return {
        "dGEAF": before["GEAF"] - after["GEAF"],
        "dRhoB": rho_before - rho_after,
        "dErrorSlope": before["error_slope"] - after["error_slope"],
        "dTargetAmplify": before["target_amplify"] - after["target_amplify"],
        "GEAF_before": before["GEAF"],
        "GEAF_after": after["GEAF"],
    }


def evaluate_method(
    graphs: list[nx.DiGraph],
    plan_fn: Callable[[nx.DiGraph], RepairPlan],
    with_spectral: bool = True,
    H: int = 4,
) -> dict:
    """Run ``plan_fn`` on every graph and aggregate all metrics."""
    rec, costs, tpr_tokens, lat, edits = [], [], [], [], []
    down_red, pd_red, incon, iou = [], [], [], []
    is_sub = []
    dGEAF, dRhoB, dSlope, dTAmp = [], [], [], []

    for G in graphs:
        plan = plan_fn(G)
        m = measure_recovery(G, plan.nodes)
        rec.append(m["recovered"])
        costs.append(plan.token_cost)
        lat.append(plan.latency)
        edits.append(plan.n_edits)
        down_red.append(m["downstream_err_reduction"])
        pd_red.append(m["pd_reduction"])
        incon.append(m["local_inconsistency"])
        if not np.isnan(m["region_iou"]):
            iou.append(m["region_iou"])
        is_sub.append(plan.is_subgraph)
        if m["recovered"]:
            tpr_tokens.append(plan.token_cost)
        if with_spectral:
            sd = spectral_deltas(G, plan.nodes, H)
            dGEAF.append(sd["dGEAF"])
            dRhoB.append(sd["dRhoB"])
            dSlope.append(sd["dErrorSlope"])
            dTAmp.append(sd["dTargetAmplify"])

    n = len(graphs)
    n_rec = sum(rec)
    recovery = n_rec / n if n else 0.0
    mean_cost = float(np.mean(costs)) if costs else 0.0
    out = {
        "recovery": recovery,
        "n_recovered": int(n_rec),
        "n": n,
        "mean_token_cost": mean_cost,
        # cost-normalized recovery: recoveries per 1k tokens
        "cost_norm_recovery": float(recovery / (mean_cost / 1000.0)) if mean_cost else 0.0,
        "tokens_per_recovery": float(np.sum(costs) / n_rec) if n_rec else float("inf"),
        "mean_latency": float(np.mean(lat)) if lat else 0.0,
        "mean_edits": float(np.mean(edits)) if edits else 0.0,
        "downstream_err_reduction": float(np.mean(down_red)) if down_red else 0.0,
        "pd_reduction": float(np.mean(pd_red)) if pd_red else 0.0,
        "local_inconsistency": float(np.mean(incon)) if incon else 0.0,
        "region_iou": float(np.mean(iou)) if iou else float("nan"),
        "is_subgraph": bool(np.mean(is_sub) > 0.5),
    }
    if with_spectral:
        out.update({
            "dGEAF": float(np.mean(dGEAF)) if dGEAF else 0.0,
            "dRhoB": float(np.mean(dRhoB)) if dRhoB else 0.0,
            "dErrorSlope": float(np.mean(dSlope)) if dSlope else 0.0,
            "dTargetAmplify": float(np.mean(dTAmp)) if dTAmp else 0.0,
        })
    return out
