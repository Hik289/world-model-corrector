"""Experiment 2: Parametric GWM Failed-Case Repair (Table 2).

Repairs failed parametric graph-world-model rollouts (GCN/MPNN/GPS/ActionNode/
Error-Aware GWM). Reports NodeMSE-style residual error, edge-structure recovery,
return error, and the spectral reductions dGEAF / d rho(B). Expectation: WM-SAR
reduces graph rollout error and recovers failed parametric plans, approaching the
OracleRegion upper bound.
"""

from __future__ import annotations

import numpy as np

import _common as C

from wm_sar import baselines as bl
from wm_sar import metrics as me
from wm_sar.repair_executor import propagate_effective_error

COLUMNS = ["Method", "NodeMSE", "EdgeF1", "ReturnErr", "dGEAF", "dRhoB", "Recovery"]


def _node_mse_and_return(g, repaired):
    """Residual node error (proxy NodeMSE) and return/target error after repair."""
    eff = propagate_effective_error(g, set(repaired))
    vals = [e for v, e in eff.items() if v != g.graph["t_star"]]
    node_mse = float(np.mean(np.square(vals))) if vals else 0.0
    ret_err = float(eff[g.graph["t_star"]])
    return node_mse, ret_err


def run(verbose: bool = True) -> list[dict]:
    _, _, gwm_graphs = C.build_dataset()
    methods = {
        "BeforeRepair": lambda g: bl.RepairPlan("BeforeRepair", set(), 0, 0, False, 0.0),
        "Top-B-Nodes": bl.top_b_nodes,
        "Top-B-Edges": bl.top_b_edges,
        "kHop-Subgraph": lambda g: bl.khop_last_error(g, 2),
        "WM-SAR": bl.wm_sar,
        "OracleRegion": bl.oracle_region,
    }
    rows = []
    for name, fn in methods.items():
        r = me.evaluate_method(gwm_graphs, fn)
        node_mses, ret_errs, edge_f1s = [], [], []
        for g in gwm_graphs:
            plan = fn(g)
            nm, re = _node_mse_and_return(g, plan.nodes)
            node_mses.append(nm)
            ret_errs.append(re)
            # edge structure recovery proxy: fraction of corrupted region covered
            gt = set(g.graph.get("gt_region", set()))
            cov = len(plan.nodes & gt) / len(gt) if gt else 0.0
            prec = len(plan.nodes & gt) / len(plan.nodes) if plan.nodes else 0.0
            f1 = 2 * cov * prec / (cov + prec) if (cov + prec) else 0.0
            edge_f1s.append(f1)
        rows.append({
            "Method": name,
            "NodeMSE": C.fmt(float(np.mean(node_mses)), 4),
            "EdgeF1": C.fmt(float(np.mean(edge_f1s))),
            "ReturnErr": C.fmt(float(np.mean(ret_errs))),
            "dGEAF": C.fmt(r["dGEAF"], 1),
            "dRhoB": C.fmt(r["dRhoB"], 3),
            "Recovery": C.fmt(r["recovery"]),
        })
    if verbose:
        C.print_table("Table 2: Parametric GWM Failed-Case Repair", rows, COLUMNS)
    C.save_csv("table2_parametric_gwm_repair.csv", rows, COLUMNS)
    return rows


if __name__ == "__main__":
    run()
