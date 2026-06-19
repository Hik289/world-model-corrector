"""Experiment 4: Spectral Dynamics Reduction (Table 4 + Figure 5).

Mandatory experiment proving WM-SAR is a world-model error-control method:
it should reduce the graph error-amplification field GEAF, the coupled
amplification rho(B), the temporal ErrorSlope, and TargetAmplify more than
scanner / pointwise baselines. Produces a grouped bar chart (Figure 5).
"""

from __future__ import annotations

import _common as C

from wm_sar import baselines as bl
from wm_sar import metrics as me

COLUMNS = ["Method", "dGEAF", "dRhoB", "dErrorSlope", "dTargetAmplify"]


def run(verbose: bool = True, make_fig: bool = True) -> list[dict]:
    _, agent_graphs, gwm_graphs = C.build_dataset()
    graphs = agent_graphs + gwm_graphs
    methods = {
        "TraceScan-w1": lambda g: bl.trace_scan_window(g, 1),
        "TraceScan-Full": lambda g: bl.trace_scan_full(g),
        "PointwiseGEA (Top-B)": bl.top_b_nodes,
        "WM-SAR": bl.wm_sar,
        "OracleRegion": bl.oracle_region,
    }
    rows = []
    for name, fn in methods.items():
        r = me.evaluate_method(graphs, fn)
        rows.append({
            "Method": name,
            "dGEAF": C.fmt(r["dGEAF"], 2),
            "dRhoB": C.fmt(r["dRhoB"], 3),
            "dErrorSlope": C.fmt(r["dErrorSlope"], 4),
            "dTargetAmplify": C.fmt(r["dTargetAmplify"], 2),
        })
    if verbose:
        C.print_table("Table 4: Spectral Dynamics Reduction", rows, COLUMNS)
    C.save_csv("table4_spectral_reduction.csv", rows, COLUMNS)
    if make_fig:
        _figure(rows)
    return rows


def _figure(rows: list[dict]) -> None:
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    metrics = ["dGEAF", "dRhoB", "dErrorSlope"]
    methods = [r["Method"] for r in rows]
    x = np.arange(len(metrics))
    w = 0.15
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, r in enumerate(rows):
        vals = [float(r[m]) for m in metrics]
        # normalize each metric column to [0,1] for visual comparability
        ax.bar(x + i * w, vals, w, label=r["Method"])
    ax.set_xticks(x + w * (len(rows) - 1) / 2)
    ax.set_xticklabels(["ΔGEAF", "Δρ(B)", "ΔErrorSlope"])
    ax.set_ylabel("Reduction after repair (higher = more error removed)")
    ax.set_title("Figure 5: Spectral Dynamics Reduction by Repair Method")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(C.FIG_DIR, "fig5_spectral_reduction.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    run()
