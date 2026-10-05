from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


HERE = Path(__file__).resolve().parent
TABLE_PATH = HERE / "main_table_rho.json"
OUTPUT_PATH = HERE / "fig_rho_reduction.png"


def load_table(path=TABLE_PATH):
    with Path(path).open(encoding="utf-8") as handle:
        table = json.load(handle)
    if table.get("source") != "user_provided_main_table":
        raise ValueError("Expected user-provided main-table data.")
    rows = table["rows"]
    if not rows or len({row["method"] for row in rows}) != len(rows):
        raise ValueError("Table methods must be nonempty and unique.")
    for row in rows:
        for key in ("size", "rho_reduction", "rho_uncertainty"):
            value = row[key]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {key} for {row['method']}.")
    return table


def render_rho_reduction(output=OUTPUT_PATH, data_path=TABLE_PATH, dpi=400):
    if dpi < 300:
        raise ValueError("DPI must be at least 300.")
    table = load_table(data_path)
    rows = table["rows"]
    positions = list(range(len(rows)))
    width = 0.35
    rho = [row["rho_reduction"] for row in rows]
    uncertainty = [row["rho_uncertainty"] for row in rows]
    sizes = [row["size"] for row in rows]
    colors = ["#D6604D" if row["method"] == "ReCore" else "#2166AC" for row in rows]
    style = {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.spines.top": False,
        "axes.grid": False,
        "savefig.facecolor": "white",
    }
    with matplotlib.rc_context(style):
        fig, ax_rho = plt.subplots(figsize=(9.4, 3.9))
        ax_size = ax_rho.twinx()
        ax_rho.bar(
            [x - width / 2 for x in positions], rho, width=width,
            color=colors, edgecolor="white", linewidth=0.5,
            yerr=uncertainty, capsize=2.0,
            error_kw={"ecolor": "black", "elinewidth": 0.75, "capthick": 0.75},
            zorder=3,
        )
        ax_size.bar(
            [x + width / 2 for x in positions], sizes, width=width,
            color="#E08214", alpha=0.85, edgecolor="white", linewidth=0.5,
            zorder=3,
        )
        for x, row in zip(positions, rows):
            ax_rho.text(
                x - width / 2, row["rho_reduction"] + row["rho_uncertainty"] + 0.025,
                f"{row['rho_reduction']:.2f}", ha="center", va="bottom",
                fontsize=8, color="black",
                fontweight="bold" if row["method"] == "ReCore" else "normal",
            )
            ax_size.text(
                x + width / 2, row["size"] + 0.3, f"{row['size']:.1f}",
                ha="center", va="bottom", fontsize=8, color="black",
            )
        ax_rho.set_ylabel("ρ(B) reduction", fontsize=10)
        ax_size.set_ylabel("Mean region size (nodes)", fontsize=10)
        ax_rho.set_ylim(0, 2.55)
        ax_rho.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])
        ax_size.set_ylim(0, 24)
        ax_size.set_yticks([0, 4, 8, 12, 16, 20, 24])
        ax_rho.set_xlim(-0.65, len(rows) - 0.35)
        ax_rho.set_xticks(positions)
        ax_rho.set_xticklabels([row["label"] for row in rows], fontsize=8.5)
        ax_rho.tick_params(axis="x", length=0)
        ax_rho.tick_params(axis="y", labelsize=8.5)
        ax_size.tick_params(axis="y", labelsize=8.5)
        ax_rho.set_axisbelow(True)
        ax_rho.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.7)
        ax_size.grid(False)
        ax_rho.spines["right"].set_visible(False)
        ax_size.spines["left"].set_visible(False)
        ax_rho.set_title(
            f"ρ(B) reduction vs. region size ({table['testbed']})",
            fontsize=11, pad=8,
        )
        ax_rho.legend(
            handles=[
                Patch(facecolor="#2166AC", label="ρ(B) reduction (other)"),
                Patch(facecolor="#D6604D", label="ρ(B) reduction (ReCore)"),
                Patch(facecolor="#E08214", alpha=0.85, label="Mean region size"),
            ],
            loc="upper left", ncol=3, fontsize=8, framealpha=1,
            facecolor="white", edgecolor="#D0D0D0", handlelength=1.5,
        )
        fig.tight_layout(pad=0.7)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi, bbox_inches="tight", metadata={
            "Source": "User-provided main table, 2026-10-05; not a rerun of experiments.",
            "Uncertainty": table["uncertainty"],
        })
        plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description="Plot spectral relief and region size from the supplied main table.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--data", type=Path, default=TABLE_PATH)
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()
    print(render_rho_reduction(args.output, args.data, args.dpi))


if __name__ == "__main__":
    main()
