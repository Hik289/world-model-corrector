"""Run the full WM-SAR experiment suite.

Generates (and persists) the synthetic failed-rollout corpus, runs Experiments
1-6, prints all result tables, and writes figures + CSVs. Deterministic under
seed=42.

    python experiments/run_all.py
"""

from __future__ import annotations

import json
import os
import time

import _common as C

import exp1_agent_wm_repair as e1
import exp2_parametric_gwm_repair as e2
import exp3_subgraph_vs_pointwise as e3
import exp4_spectral_reduction as e4
import exp5_context_limited as e5
import exp6_ablation as e6

from wm_sar import data_generator as dg


def _persist_dataset() -> None:
    ds, agent_graphs, gwm_graphs = C.build_dataset()
    dg.save_dataset(ds, C.DATA_DIR)
    ds["stats"]["mean_agent_graph_nodes"] = float(
        sum(g.number_of_nodes() for g in agent_graphs) / len(agent_graphs))
    ds["stats"]["mean_gwm_graph_nodes"] = float(
        sum(g.number_of_nodes() for g in gwm_graphs) / len(gwm_graphs))
    with open(os.path.join(C.DATA_DIR, "dataset_statistics.json"), "w") as f:
        json.dump(ds["stats"], f, indent=2)
    print("Dataset statistics:")
    print(json.dumps(ds["stats"], indent=2))


def _make_region_figure() -> None:
    """Figure 3: amplification field + selected WM-SAR region on one example."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    from wm_sar import amplification as amp
    from wm_sar import baselines as bl

    _, agent_graphs, _ = C.build_dataset()
    # pick a graph with a reasonably long amplification chain
    G = max(agent_graphs, key=lambda g: len(g.graph.get("gt_region", [])))
    tfield = amp.phi_H_target(G)
    region = bl.wm_sar(G).nodes

    pos = nx.spring_layout(G, seed=42, k=0.5)
    fig, ax = plt.subplots(figsize=(11, 7))
    vals = [tfield.get(n, 0.0) for n in G.nodes()]
    nodes = nx.draw_networkx_nodes(
        G, pos, node_color=vals, cmap="viridis", node_size=180, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax, arrowsize=6)
    # outline the WM-SAR region
    nx.draw_networkx_nodes(
        G, pos, nodelist=[n for n in region if G.has_node(n)],
        node_color="none", edgecolors="red", linewidths=2.5, node_size=260, ax=ax)
    tstar = G.graph["t_star"]
    nx.draw_networkx_nodes(
        G, pos, nodelist=[tstar], node_color="red", node_shape="X",
        node_size=320, ax=ax)
    fig.colorbar(nodes, ax=ax, label="target-aware amplification  φ_H^target(v)")
    ax.set_title("Figure 3: Target-Aware Amplification Field + WM-SAR Region (red)\n"
                 "X = final-failure target node")
    ax.axis("off")
    out = os.path.join(C.FIG_DIR, "fig3_amplification_region.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[saved] {out}")


def main() -> None:
    t0 = time.time()
    print("#" * 78)
    print("# WM-SAR: World-Model-Guided Subgraph Amplification Repair")
    print("# Full experiment suite (seed=42)")
    print("#" * 78)

    _persist_dataset()

    e1.run()
    e2.run()
    e3.run()
    e4.run()
    e5.run()
    e6.run()

    _make_region_figure()

    print(f"\nAll experiments complete in {time.time() - t0:.1f}s.")
    print(f"Tables (CSV): {C.ANALYSIS_DIR}")
    print(f"Figures:      {C.FIG_DIR}")


if __name__ == "__main__":
    main()
