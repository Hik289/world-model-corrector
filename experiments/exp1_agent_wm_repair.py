"""Experiment 1: Agent World-Model Failed-Case Repair (Table 1).

Repairs failed agent-world-model rollouts and reports recovery, cost-normalized
recovery, tokens-per-recovery, propagation-depth reduction, and downstream-error
reduction. Expectation: WM-SAR recovers more than pointwise / short-context
repair; TraceScan-Full is competitive but far more expensive.
"""

from __future__ import annotations

import _common as C

from wm_sar import baselines as bl
from wm_sar import metrics as me

COLUMNS = ["Method", "Recovery", "CostNorm", "Tokens/Rec", "PDred", "DownErrRed", "Tokens"]


def _methods():
    return {
        "LastError-Point": bl.last_error_point,
        "FirstFailedCall-Point": bl.first_failed_call_point,
        "RuleScanner-Point": bl.rule_scanner_point,
        "TraceScan-w1-Point": lambda g: bl.trace_scan_window(g, 1),
        "TraceScan-w2-Point": lambda g: bl.trace_scan_window(g, 2),
        "TraceScan-w4-Point": lambda g: bl.trace_scan_window(g, 4),
        "TraceScan-Full-Point": lambda g: bl.trace_scan_full(g),
        "Top-B-Nodes": bl.top_b_nodes,
        "kHop-Subgraph": lambda g: bl.khop_last_error(g, 2),
        "WM-SAR": bl.wm_sar,
        "FullReplan": bl.full_replan,
    }


def run(verbose: bool = True) -> list[dict]:
    _, agent_graphs, _ = C.build_dataset()
    rows = []
    for name, fn in _methods().items():
        r = me.evaluate_method(agent_graphs, fn)
        rows.append({
            "Method": name,
            "Recovery": C.fmt(r["recovery"]),
            "CostNorm": C.fmt(r["cost_norm_recovery"]),
            "Tokens/Rec": C.fmt(r["tokens_per_recovery"], 0),
            "PDred": C.fmt(r["pd_reduction"], 2),
            "DownErrRed": C.fmt(r["downstream_err_reduction"], 2),
            "Tokens": C.fmt(r["mean_token_cost"], 0),
        })
    if verbose:
        C.print_table("Table 1: Agent World-Model Failed-Case Repair", rows, COLUMNS)
    C.save_csv("table1_agent_wm_repair.csv", rows, COLUMNS)
    return rows


if __name__ == "__main__":
    run()
