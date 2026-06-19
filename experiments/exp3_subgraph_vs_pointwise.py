"""Experiment 3: Subgraph vs Pointwise Repair (Table 3).

Question: is repairing a connected amplification subgraph better than repairing
top nodes/edges one by one? Compares Top-B Nodes/Edges/Mixed and a k-hop
subgraph against WM-SAR's region. Key signal: local inconsistency after repair
(pointwise edits leave dangling corrupted neighbors; the subgraph repair does
not) and target-cone error reduction.
"""

from __future__ import annotations

import _common as C

from wm_sar import baselines as bl
from wm_sar import metrics as me

COLUMNS = ["Method", "Recovery", "TargetConeErrRed", "PDred",
           "LocalIncon", "dGEAF", "dRhoB", "RegionIoU"]


def run(verbose: bool = True) -> list[dict]:
    # combine both domains for a broad subgraph-vs-pointwise comparison
    _, agent_graphs, gwm_graphs = C.build_dataset()
    graphs = agent_graphs + gwm_graphs
    methods = {
        "Top-B-Nodes (pointwise)": bl.top_b_nodes,
        "Top-B-Edges (pointwise)": bl.top_b_edges,
        "Top-B-Mixed (pointwise)": bl.top_b_mixed,
        "kHop-Subgraph": lambda g: bl.khop_last_error(g, 2),
        "TargetCone-Subgraph": bl.target_cone_subgraph,
        "WM-SAR Region": bl.wm_sar,
        "OracleRegion": bl.oracle_region,
    }
    rows = []
    for name, fn in methods.items():
        r = me.evaluate_method(graphs, fn)
        rows.append({
            "Method": name,
            "Recovery": C.fmt(r["recovery"]),
            "TargetConeErrRed": C.fmt(r["downstream_err_reduction"], 2),
            "PDred": C.fmt(r["pd_reduction"], 2),
            "LocalIncon": C.fmt(r["local_inconsistency"], 2),
            "dGEAF": C.fmt(r["dGEAF"], 1),
            "dRhoB": C.fmt(r["dRhoB"], 3),
            "RegionIoU": C.fmt(r["region_iou"]),
        })
    if verbose:
        C.print_table("Table 3: Subgraph vs Pointwise Repair", rows, COLUMNS)
    C.save_csv("table3_subgraph_vs_pointwise.csv", rows, COLUMNS)
    return rows


if __name__ == "__main__":
    run()
