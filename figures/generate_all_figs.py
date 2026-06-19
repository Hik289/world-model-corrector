"""
generate_all_figs.py — Regenerate all paper figures FROM REAL experiment results.

Reads JSON files from experiments/results/ and emits paper-ready PNGs to
figures/ (and they are also copied into paper/ by the caller).

Figures produced:
  fig_tradeoff.png        ← exp_agent.json           (Pareto: region size vs Rec-Exact-like)
  fig_llm_multiapi.png    ← exp_agent_llm.json
                           + exp_multiapi.json       (NEW: merged 2-panel)
  fig_llm_comparison.png  ← exp_agent_llm.json       (kept for back-compat; same content as
                                                       panel (a) of fig_llm_multiapi)
  fig_multiapi.png        ← exp_multiapi.json        (kept for back-compat)
  fig_budget.png          ← exp_budget.json
  fig_cascade_gain.png    ← exp_cascade_gain.json
  fig_rho_reduction.png   ← exp_agent.json           (bar chart)
  fig_ablation.png        ← (no real ablation experiment — kept disabled)

DPI ≥ 150, font sizes ≥ 9pt, all axes labelled.

This script REPLACES the previous one (which used hardcoded numbers).
"""

from __future__ import annotations
import json
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# ── Global typography (paper-ready) ─────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":         "DejaVu Sans",
    "font.size":           9,
    "axes.titlesize":      10,
    "axes.labelsize":      9,
    "xtick.labelsize":     8,
    "ytick.labelsize":     8,
    "legend.fontsize":     8,
    "figure.titlesize":    11,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.grid":           True,
    "grid.alpha":          0.25,
    "grid.linewidth":      0.5,
    "savefig.bbox":        "tight",
    "savefig.dpi":         200,
})

# Colour palette (colorblind-safe, ColorBrewer)
C_BLUE   = "#2166AC"
C_LIGHTB = "#92C5DE"
C_ORANGE = "#E08214"
C_GREEN  = "#4DAC26"
C_RED    = "#D01C8B"
C_WMSAR  = "#D6604D"
C_GRAY   = "#969696"
C_DARK   = "#252525"
C_ORACLE = "#4393C3"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "experiments", "results")
FIG_DIR     = os.path.join(ROOT, "figures")
PAPER_DIR   = os.path.join(ROOT, "paper")


def _load(name: str) -> dict:
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return json.load(f)


def _save(fig, base_name: str) -> None:
    out = os.path.join(FIG_DIR, base_name)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    # mirror into paper/
    paper_out = os.path.join(PAPER_DIR, base_name)
    import shutil
    shutil.copy(out, paper_out)
    print(f"  saved → {base_name}  (figures/ + paper/)")


# ═══════════════════════════════════════════════════════════════════════════
#  1. ρ(B) reduction bar chart  ──── exp_agent.json
# ═══════════════════════════════════════════════════════════════════════════
def fig_rho_reduction():
    d = _load("exp_agent.json")
    sums = d["summaries"]
    order = ["Greedy-Point(K=1)", "Window-4-Point", "TopK-Point(K=5)",
             "CascadeRepair", "Oracle", "LocalRepair-3Hop", "WM-SAR"]
    rho_red = [sums[m]["mean_rho_reduction"] for m in order]
    sizes   = [sums[m]["mean_region_size"]    for m in order]
    rho_B0  = d["dataset_stats"]["mean_rho_B"]
    short = {
        "Greedy-Point(K=1)":  "Greedy\nPoint",
        "Window-4-Point":     "Window-4\nPoint",
        "TopK-Point(K=5)":    "TopK-5\nPoint",
        "CascadeRepair":      "Cascade\nRepair",
        "Oracle":             "Oracle",
        "LocalRepair-3Hop":   "LocalRep\n3-Hop",
        "WM-SAR":             "WM-SAR",
    }
    labels = [short[m] for m in order]
    n = len(order)
    x = np.arange(n)
    w = 0.35
    sizes_n = [s / 26.0 for s in sizes]

    fig, ax1 = plt.subplots(figsize=(6.0, 3.2))
    ax2 = ax1.twinx()
    rho_colors  = [C_WMSAR if m == "WM-SAR" else C_BLUE for m in order]
    # Use consistent orange for region size to avoid colour-coding inconsistency.
    size_color = C_ORANGE

    b1 = ax1.bar(x - w/2, rho_red,  width=w, color=rho_colors,
                 edgecolor="white", linewidth=0.5, zorder=3)
    b2 = ax2.bar(x + w/2, sizes_n, width=w, color=size_color,
                 edgecolor="white", linewidth=0.5, zorder=3, alpha=0.85)

    ax1.axhline(rho_B0, color=C_GRAY, linewidth=0.8, linestyle="--", zorder=2,
                label=f"Unrepaired ρ(B)={rho_B0:.2f}")

    for i, (r, s) in enumerate(zip(rho_red, sizes)):
        ax1.text(i - w/2, r + 0.04, f"{r:.2f}",
                 ha="center", va="bottom", fontsize=7.5,
                 fontweight="bold" if order[i] == "WM-SAR" else "normal",
                 color=C_WMSAR if order[i] == "WM-SAR" else C_DARK)
        ax2.text(i + w/2, sizes_n[i] + 0.015, f"{s:.1f}",
                 ha="center", va="bottom", fontsize=7, color=C_ORANGE)

    ax1.set_ylabel("ρ(B) Reduction", fontsize=9)
    ax2.set_ylabel("Region Size / 26", fontsize=9, color=C_ORANGE)
    ax1.set_ylim(0, max(rho_B0 + 0.4, max(rho_red) + 0.4))
    ax2.set_ylim(0, 1.2)
    ax2.tick_params(axis="y", colors=C_ORANGE)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8)
    ax1.tick_params(axis="x", length=0)
    ax1.grid(axis="x", visible=False)
    ax2.grid(False)
    ax1.set_title(f"ρ(B) reduction vs. region size (n={d['n']} agent calling-trees)",
                  fontsize=9.5, pad=4)

    # Build a complete legend covering both axes
    rho_patch  = mpatches.Patch(color=C_BLUE,   label="ρ(B) reduction (other)")
    wmsar_patch= mpatches.Patch(color=C_WMSAR,  label="ρ(B) reduction (WM-SAR)")
    size_patch = mpatches.Patch(color=C_ORANGE, label="Region size / 26")
    base_line  = mlines.Line2D([], [], color=C_GRAY, linestyle="--", linewidth=0.8,
                                label=f"Unrepaired ρ(B)={rho_B0:.2f}")
    ax1.legend(handles=[rho_patch, wmsar_patch, size_patch, base_line],
               loc="upper left", fontsize=7.5, framealpha=0.85,
               handlelength=1.5, ncol=2)
    fig.tight_layout()
    _save(fig, "fig_rho_reduction.png")


# ═══════════════════════════════════════════════════════════════════════════
#  2. Efficiency–Quality trade-off scatter  ──── exp_agent.json
# ═══════════════════════════════════════════════════════════════════════════
def fig_tradeoff():
    d = _load("exp_agent.json")
    sums = d["summaries"]
    order = ["Greedy-Point(K=1)", "Window-2-Point", "Window-4-Point",
             "Window-8-Point", "TopK-Point(K=3)", "TopK-Point(K=5)",
             "LocalRepair-2Hop", "LocalRepair-3Hop", "CascadeRepair",
             "Oracle", "WM-SAR"]
    sizes  = [sums[m]["mean_region_size"]    for m in order]
    rho_red = [sums[m]["mean_rho_reduction"] for m in order]
    rho_B0 = d["dataset_stats"]["mean_rho_B"]
    quality = [r / rho_B0 for r in rho_red]

    fig, ax = plt.subplots(figsize=(7.0, 3.8))

    # Hand-tuned label placements that avoid overlap (xytext = absolute)
    # Coordinates are in (region_size, quality) data space
    label_pos = {
        "Greedy-Point(K=1)": (3.5,  0.05),
        "Window-2-Point":    (3.5,  0.20),
        "Window-4-Point":    (5.5,  0.30),
        "Window-8-Point":    (10.5, 0.55),
        "TopK-Point(K=3)":   (5.5,  0.40),
        "TopK-Point(K=5)":   (8.0,  0.48),
        "LocalRepair-2Hop":  (21.0, 0.78),
        "LocalRepair-3Hop":  (26.5, 0.93),
        "CascadeRepair":     (3.0,  0.70),
        "Oracle":            (10.5, 0.75),
        "WM-SAR":            (10.5, 0.99),
    }
    for m, s, q in zip(order, sizes, quality):
        if m == "WM-SAR":
            ax.scatter(s, q, s=260, marker="*", color=C_WMSAR,
                       edgecolors="#8B0000", linewidths=0.8, zorder=6,
                       label="WM-SAR (proposed)")
        elif m == "Oracle":
            ax.scatter(s, q, s=90, marker="D", color=C_ORACLE,
                       edgecolors=C_DARK, linewidths=0.5, zorder=5,
                       label="Oracle (GT region)")
        else:
            ax.scatter(s, q, s=55, marker="o", color=C_GRAY,
                       edgecolors=C_DARK, linewidths=0.5, zorder=4)
        tx, ty = label_pos[m]
        col = C_WMSAR if m == "WM-SAR" else (C_ORACLE if m == "Oracle" else "#444444")
        fw  = "bold" if m == "WM-SAR" else "normal"
        ax.annotate(m, xy=(s, q), xytext=(tx, ty),
                    fontsize=7.5, color=col, fontweight=fw,
                    arrowprops=dict(arrowstyle="-", color="#BBBBBB",
                                    lw=0.45, shrinkA=2, shrinkB=2))

    # Pareto frontier (smaller region & higher quality)
    pts = sorted(zip(sizes, quality), key=lambda p: p[0])
    front = []
    best = -1
    for sz, q in pts:
        if q > best:
            front.append((sz, q))
            best = q
    if front:
        px, py = zip(*front)
        ax.step(px, py, where="post", color=C_GREEN, linewidth=1.2,
                linestyle="--", zorder=3, label="Pareto frontier")

    ax.set_xlabel("Region Size (# nodes shown to repairer)", fontsize=9)
    ax.set_ylabel("Quality: ρ-reduction / ρ(B)", fontsize=9)
    ax.set_xlim(-1.0, 32)
    ax.set_ylim(0.0, 1.08)
    ax.set_title(f"Efficiency vs. Quality (n={d['n']} agent calling-trees)",
                 fontsize=10, pad=4)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    _save(fig, "fig_tradeoff.png")


# ═══════════════════════════════════════════════════════════════════════════
#  3. Budget sensitivity  ──── exp_budget.json
# ═══════════════════════════════════════════════════════════════════════════
def fig_budget():
    d = _load("exp_budget.json")
    Ks = d["K_values"]
    methods = ["Greedy-TopK", "Window-K", "Local-KHop", "Oracle-K", "WM-SAR"]
    # WM-SAR is budget-free → constant horizontal line
    wmsar_const = d["wmsar_default"]

    colors  = {"Greedy-TopK": C_BLUE,   "Window-K": C_ORANGE,
               "Local-KHop": C_GREEN,  "Oracle-K": C_ORACLE,
               "WM-SAR":     C_WMSAR}
    markers = {"Greedy-TopK":"o", "Window-K":"s", "Local-KHop":"^",
               "Oracle-K":"D", "WM-SAR":"*"}

    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    for m in methods:
        if m == "WM-SAR":
            ax.axhline(wmsar_const, color=C_WMSAR, linewidth=2.0,
                       linestyle="-", zorder=5, label=f"WM-SAR (no budget)  ρ-red={wmsar_const:.2f}")
            continue
        ys = [d["results"][str(K)][m]["mean_rho_reduction"] for K in Ks]
        ax.plot(Ks, ys, marker=markers[m], color=colors[m],
                linewidth=1.5, markersize=6, label=m, zorder=4)

    ax.axhline(d["oracle_full"], color=C_GRAY, linewidth=0.8,
               linestyle=":", zorder=2,
               label=f"Oracle (full region)  ρ-red={d['oracle_full']:.2f}")

    ax.set_xlabel("Repair budget K (# nodes)", fontsize=9)
    ax.set_ylabel("Mean ρ(B) reduction", fontsize=9)
    ax.set_title(f"Budget sensitivity (n={d['n']} agent calling-trees)",
                 fontsize=10, pad=4)
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)
    ax.set_xticks(Ks)
    fig.tight_layout()
    _save(fig, "fig_budget.png")


# ═══════════════════════════════════════════════════════════════════════════
#  4. Cascade-gain α sensitivity  ──── exp_cascade_gain.json
# ═══════════════════════════════════════════════════════════════════════════
def fig_cascade_gain():
    d = _load("exp_cascade_gain.json")
    alphas = d["alphas"]
    methods = ["Greedy-Point(K=1)", "Window-4-Point", "TopK-Point(K=5)",
               "LocalRepair-2Hop", "LocalRepair-3Hop", "WM-SAR"]
    colors  = {"Greedy-Point(K=1)": C_GRAY,
               "Window-4-Point":    C_ORANGE,
               "TopK-Point(K=5)":   C_BLUE,
               "LocalRepair-2Hop":  C_LIGHTB,
               "LocalRepair-3Hop":  C_GREEN,
               "WM-SAR":            C_WMSAR}
    markers = {"Greedy-Point(K=1)":"o", "Window-4-Point":"s",
               "TopK-Point(K=5)":"^", "LocalRepair-2Hop":"v",
               "LocalRepair-3Hop":"D", "WM-SAR":"*"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.4))

    # ── Left: ρ-reduction vs α  ─────────────────────────────────
    for m in methods:
        ys = [d["results"][str(a)]["summaries"][m]["mean_rho_reduction"]
              for a in alphas]
        lw = 2.0 if m == "WM-SAR" else 1.2
        ms = 8 if m == "WM-SAR" else 5
        ax1.plot(alphas, ys, marker=markers[m], color=colors[m],
                 linewidth=lw, markersize=ms, label=m, zorder=5 if m == "WM-SAR" else 3)

    rho_B = [d["results"][str(a)]["rho_before"] for a in alphas]
    ax1.plot(alphas, rho_B, color=C_DARK, linewidth=0.8, linestyle=":",
             label="Unrepaired ρ(B)", zorder=2)

    ax1.set_xlabel("Cascade gain α", fontsize=9)
    ax1.set_ylabel("Mean ρ(B) reduction", fontsize=9)
    ax1.set_title("(a) ρ-reduction vs cascade gain", fontsize=10, pad=4)
    ax1.axvline(1.0, color=C_DARK, linewidth=0.5, linestyle="-", alpha=0.4)

    # ── Right: NodeMSE@32 after repair vs α  ─────────────────────
    for m in methods:
        ys = [d["results"][str(a)]["summaries"][m]["NodeMSE_after"]["32"]
              for a in alphas]
        lw = 2.0 if m == "WM-SAR" else 1.2
        ms = 8 if m == "WM-SAR" else 5
        ax2.plot(alphas, ys, marker=markers[m], color=colors[m],
                 linewidth=lw, markersize=ms, label=m,
                 zorder=5 if m == "WM-SAR" else 3)
    ax2.set_xlabel("Cascade gain α", fontsize=9)
    ax2.set_ylabel("NodeMSE@H=32 (post-repair)", fontsize=9)
    ax2.set_yscale("symlog", linthresh=1e-2)
    ax2.set_title("(b) Residual error at long horizon", fontsize=10, pad=4)
    ax2.axvline(1.0, color=C_DARK, linewidth=0.5, linestyle="-", alpha=0.4)

    # Shared legend outside subplots, below
    handles, labels_ = ax1.get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=4, fontsize=7.5,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Cascade-gain robustness (n={d['n']} agent calling-trees)",
                 fontsize=10.5, y=1.02)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _save(fig, "fig_cascade_gain.png")


# ═══════════════════════════════════════════════════════════════════════════
#  5. LLM single-API comparison panel  ──── exp_agent_llm.json
#     and 6. Multi-API panel ──── exp_multiapi.json
#     ALSO merged: fig_llm_multiapi.png (2-panel as required by Director)
# ═══════════════════════════════════════════════════════════════════════════
def _plot_llm_panel(ax, d_llm, with_legend=True):
    """Single-API horizontal bar (Rec-Exact/Rec-Type/Rec-2Hop) + token text."""
    methods_in = ["Greedy-Point-LLM", "Window-4-LLM", "Window-8-LLM",
                  "TopK-5-LLM", "LocalRepair-2Hop-LLM", "Full-Graph-LLM",
                  "WM-SAR-LLM"]
    short = {
        "Greedy-Point-LLM":     "Greedy-Point",
        "Window-4-LLM":         "Window-4",
        "Window-8-LLM":         "Window-8",
        "TopK-5-LLM":           "TopK-5",
        "LocalRepair-2Hop-LLM": "LocalRep-2Hop",
        "Full-Graph-LLM":       "Full-Graph",
        "WM-SAR-LLM":           "WM-SAR",
    }
    sums = d_llm["summaries"]
    methods = [m for m in methods_in if m in sums]
    labels  = [short[m] for m in methods]
    rec_exact = [sums[m]["rec_exact"] for m in methods]
    rec_type  = [sums[m]["rec_type"]  for m in methods]
    rec_hop2  = [sums[m]["rec_hop2"]  for m in methods]
    tokens    = [sums[m]["mean_tokens"] for m in methods]

    n = len(methods)
    y = np.arange(n)
    h = 0.25
    C_2HOP, C_TYPE, C_EXACT = "#BDBDBD", "#64B5F6", "#1565C0"
    for i in range(n):
        is_wmsar = (methods[i] == "WM-SAR-LLM")
        ec = C_WMSAR if is_wmsar else "white"
        lw = 1.0     if is_wmsar else 0.5
        ax.barh(y[i] + h,   rec_hop2[i], height=h, color=C_2HOP, edgecolor=ec, linewidth=lw, zorder=3)
        ax.barh(y[i],       rec_type[i], height=h, color=C_TYPE, edgecolor=ec, linewidth=lw, zorder=3)
        ax.barh(y[i] - h,   rec_exact[i], height=h, color=C_EXACT, edgecolor=ec, linewidth=lw, zorder=3)
        # Annotate the three recall values to the right of each bar group
        max_val = max(rec_exact[i], rec_type[i], rec_hop2[i])
        ax.text(max_val + 0.02, y[i], f"E={rec_exact[i]:.2f}",
                va="center", fontsize=6.5,
                color=C_WMSAR if is_wmsar else C_DARK,
                fontweight="bold" if is_wmsar else "normal")
        # Token text on right column
        ax.text(1.32, y[i], f"{int(tokens[i])}", va="center", fontsize=7,
                color=C_WMSAR if is_wmsar else C_GRAY,
                fontweight="bold" if is_wmsar else "normal")

    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Recall score", fontsize=9)
    ax.set_xlim(0, 1.45)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.text(1.32, n - 0.4, "Tokens",
            fontsize=7, color=C_GRAY, ha="left", fontweight="bold")
    ax.axvline(1.0, color=C_DARK, linewidth=0.4, linestyle=":", alpha=0.3, zorder=1)
    ax.axhspan(n - 1 - 0.45, n - 1 + 0.45, color=C_WMSAR, alpha=0.07, zorder=1)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_title(f"(a) Single-API recall ({d_llm.get('model','gpt-4o-mini')}, n={d_llm['n']})",
                 fontsize=10, pad=4)
    if with_legend:
        p2 = mpatches.Patch(color=C_2HOP,  label="Rec-2Hop")
        pt = mpatches.Patch(color=C_TYPE,  label="Rec-Type")
        pe = mpatches.Patch(color=C_EXACT, label="Rec-Exact")
        ax.legend(handles=[p2, pt, pe], fontsize=7, loc="lower right",
                  framealpha=0.85, handlelength=1.0)


def _plot_multiapi_panel(ax_heat, ax_line, d_multi):
    methods_in = ["Greedy-Point", "Window-4", "TopK-5",
                  "LocalRepair-2Hop", "WM-SAR"]
    model_keys = d_multi["models"]
    short_model = {
        "gpt-4.1-nano":     "4.1-nano",
        "gpt-4o-mini":      "4o-mini",
        "gpt-4.1-mini":     "4.1-mini",
        "gemini-2.5-flash": "gemini\n2.5-flash",
    }
    methods = [m for m in methods_in
               if all(m in d_multi["results"][k] for k in model_keys)]
    M, N = len(methods), len(model_keys)
    data = np.zeros((M, N))
    for i, m in enumerate(methods):
        for j, k in enumerate(model_keys):
            data[i, j] = d_multi["results"][k][m]["rec_exact"]

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "wmsar_blue", ["#FFFFFF", "#DEEBF7", "#2166AC"], N=256)
    vmax = max(0.6, float(data.max()) + 0.02)
    im = ax_heat.imshow(data, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    ax_heat.set_xticks(range(N))
    ax_heat.set_xticklabels([short_model.get(k, k) for k in model_keys], fontsize=7.5)
    ax_heat.set_yticks(range(M))
    ax_heat.set_yticklabels(methods, fontsize=8)
    ax_heat.grid(False)
    ax_heat.set_title(f"(b) Rec-Exact heatmap (n={d_multi['n']} per cell)",
                      fontsize=10, pad=4)
    for i in range(M):
        for j in range(N):
            v = data[i, j]
            tc = "white" if v > 0.5 * vmax else "#333333"
            fw = "bold" if methods[i] == "WM-SAR" else "normal"
            ax_heat.text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=7.5, color=tc, fontweight=fw)
    # Highlight WM-SAR row
    if "WM-SAR" in methods:
        wi = methods.index("WM-SAR")
        for j in range(N):
            rect = plt.Rectangle((j - 0.5, wi - 0.5), 1, 1,
                                  edgecolor=C_WMSAR, facecolor="none",
                                  linewidth=1.5, zorder=5)
            ax_heat.add_patch(rect)

    # Line plot
    mx = np.arange(N)
    plot_methods = [m for m in ["WM-SAR", "LocalRepair-2Hop", "TopK-5", "Greedy-Point"]
                    if m in methods]
    style = {"WM-SAR": ("o-", C_WMSAR, 2.0, 7),
             "LocalRepair-2Hop": ("s--", C_BLUE, 1.2, 5),
             "TopK-5": ("^:", C_GRAY, 1.0, 5),
             "Greedy-Point": ("v:", "#888888", 0.8, 4)}
    for m in plot_methods:
        ys = data[methods.index(m), :]
        st, cl, lw, ms = style[m]
        ax_line.plot(mx, ys, st, color=cl, linewidth=lw, markersize=ms,
                     label=m, zorder=5 if m == "WM-SAR" else 3)
    ax_line.set_xticks(mx)
    ax_line.set_xticklabels([short_model.get(k, k) for k in model_keys], fontsize=7.5)
    ax_line.set_ylabel("Rec-Exact", fontsize=9)
    ax_line.set_ylim(-0.02, max(0.75, vmax + 0.05))
    ax_line.set_title("(c) Cross-model Rec-Exact", fontsize=10, pad=4)
    ax_line.legend(loc="upper left", fontsize=7, framealpha=0.9, handlelength=1.2)


def fig_llm_comparison():
    d_llm = _load("exp_agent_llm.json")
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    _plot_llm_panel(ax, d_llm)
    ax.set_title(f"LLM repair (model={d_llm.get('model','gpt-4o-mini')}, n={d_llm['n']})",
                 fontsize=10, pad=4)
    fig.tight_layout()
    _save(fig, "fig_llm_comparison.png")


def fig_multiapi():
    d_multi = _load("exp_multiapi.json")
    fig, (axh, axl) = plt.subplots(1, 2, figsize=(8.0, 3.3),
                                    gridspec_kw={"width_ratios": [1.3, 1.0]})
    _plot_multiapi_panel(axh, axl, d_multi)
    fig.suptitle(f"Multi-API comparison (n={d_multi['n']} per model)",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    _save(fig, "fig_multiapi.png")


def fig_llm_multiapi_merged():
    """NEW: merged 2-panel — required by Director for 8-page compaction.

    (a) Single-API recall barchart   (LEFT)
    (b) Multi-API Rec-Exact heatmap  (RIGHT-TOP)
    (c) Cross-model line             (RIGHT-BOTTOM)
    """
    d_llm   = _load("exp_agent_llm.json")
    d_multi = _load("exp_multiapi.json")
    fig = plt.figure(figsize=(13.5, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 1.55, 1.15], wspace=0.55)
    ax_llm  = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[0, 1])
    ax_line = fig.add_subplot(gs[0, 2])
    _plot_llm_panel(ax_llm, d_llm)
    _plot_multiapi_panel(ax_heat, ax_line, d_multi)
    fig.suptitle(
        f"LLM repair: single-API (n={d_llm['n']}, model={d_llm.get('model','gpt-4o-mini')})  "
        f"+ multi-API (n={d_multi['n']} × {len(d_multi['models'])} models)",
        fontsize=11, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, "fig_llm_multiapi.png")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Regenerating WM-SAR figures from real experiment results …")
    fig_rho_reduction()
    fig_tradeoff()
    fig_budget()
    fig_cascade_gain()
    fig_llm_comparison()
    fig_multiapi()
    fig_llm_multiapi_merged()
    print("\nAll figures regenerated successfully.")
