"""Experiment 5: Context-Limited Pointwise Repair vs WM-SAR (Table 5 + Figure 6).

Does a pure engineering repairer need longer context or many pointwise edits to
match WM-SAR? Reports recovery, region localization, token cost, tokens per
recovery, #edit attempts, post-repair local inconsistency, latency, and
cost-normalized recovery. Produces the token-cost-vs-recovery scatter (Figure 6).

Expected pattern:
    TraceScan-w1/w2/w4 -> repair the wrong local symptom (short context)
    TraceScan-Full     -> better but very high token cost, still pointwise
    LLMRepair-Full-Plan-> strong but expensive
    WM-SAR             -> matches/beats recovery with far lower token cost
"""

from __future__ import annotations

import _common as C

from wm_sar import baselines as bl
from wm_sar import metrics as me

COLUMNS = ["Method", "Recovery", "RegionIoU", "Tokens", "Tokens/Rec",
           "Edits", "LocalIncon", "Latency", "CostNorm"]


def run(verbose: bool = True, make_fig: bool = True) -> list[dict]:
    _, agent_graphs, _ = C.build_dataset()
    methods = {
        "TraceScan-w1-Point": lambda g: bl.trace_scan_window(g, 1),
        "TraceScan-w2-Point": lambda g: bl.trace_scan_window(g, 2),
        "TraceScan-w4-Point": lambda g: bl.trace_scan_window(g, 4),
        "TraceScan-Full-Point": lambda g: bl.trace_scan_full(g),
        "LLMRepair-Full-Plan": bl.llm_repair_full_plan,
        "Top-B-Node/Edge": bl.top_b_mixed,
        "WM-SAR": bl.wm_sar,
        "WM-SAR+ShortPrompt": lambda g: bl.wm_sar(g, short_prompt=True),
    }
    rows = []
    for name, fn in methods.items():
        r = me.evaluate_method(agent_graphs, fn, with_spectral=False)
        rows.append({
            "Method": name,
            "Recovery": C.fmt(r["recovery"]),
            "RegionIoU": C.fmt(r["region_iou"]),
            "Tokens": C.fmt(r["mean_token_cost"], 0),
            "Tokens/Rec": C.fmt(r["tokens_per_recovery"], 0),
            "Edits": C.fmt(r["mean_edits"], 1),
            "LocalIncon": C.fmt(r["local_inconsistency"], 2),
            "Latency": C.fmt(r["mean_latency"], 2),
            "CostNorm": C.fmt(r["cost_norm_recovery"], 3),
        })
    if verbose:
        C.print_table("Table 5: Context-Limited Pointwise Repair vs WM-SAR", rows, COLUMNS)
    C.save_csv("table5_context_limited.csv", rows, COLUMNS)
    if make_fig:
        _figure(rows)
    return rows


def _figure(rows: list[dict]) -> None:
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for r in rows:
        x = float(r["Tokens"])
        y = float(r["Recovery"])
        marker = "*" if "WM-SAR" in r["Method"] else "o"
        size = 320 if "WM-SAR" in r["Method"] else 110
        ax.scatter(x, y, s=size, marker=marker, label=r["Method"], zorder=3)
        ax.annotate(r["Method"], (x, y), fontsize=7,
                    xytext=(5, 5), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Mean token cost (log scale)")
    ax.set_ylabel("Recovery success rate")
    ax.set_title("Figure 6: Token Cost vs Recovery (Agent World-Model Repair)")
    fig.tight_layout()
    out = os.path.join(C.FIG_DIR, "fig6_token_cost_vs_recovery.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    run()
