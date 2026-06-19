"""Experiment 6: Region Extraction Ablation (Table 6).

Removes one WM-SAR component at a time (target-aware amplification, node-edge
coupling, uncertainty, region growing, region pruning) and a pointwise-GEA-only
variant. Reports recovery, dGEAF, dErrorSlope, and propagation-depth reduction.
Expectation: region growing and target-aware amplification are essential;
coupling matters most for dynamic-edge / parametric failures.
"""

from __future__ import annotations

import _common as C

from wm_sar import baselines as bl
from wm_sar import metrics as me
from wm_sar.region_extractor import WMSARConfig

COLUMNS = ["Variant", "Recovery", "dGEAF", "dErrorSlope", "PDred", "RegionIoU", "Tokens"]


def _variants():
    return {
        "w/o TargetAmplification": WMSARConfig(use_target=False),
        "w/o Coupling": WMSARConfig(use_coupling=False),
        "w/o Uncertainty": WMSARConfig(use_uncertainty=False),
        "w/o RegionGrowing": WMSARConfig(use_growing=False),
        "w/o RegionPruning": WMSARConfig(use_pruning=False),
        "PointwiseGEA": WMSARConfig(use_growing=False, n_seeds=4),
        "WM-SAR Full": WMSARConfig(),
    }


def run(verbose: bool = True) -> list[dict]:
    _, agent_graphs, gwm_graphs = C.build_dataset()
    graphs = agent_graphs + gwm_graphs
    rows = []
    for name, cfg in _variants().items():
        r = me.evaluate_method(graphs, lambda g, c=cfg: bl.wm_sar(g, c))
        rows.append({
            "Variant": name,
            "Recovery": C.fmt(r["recovery"]),
            "dGEAF": C.fmt(r["dGEAF"], 1),
            "dErrorSlope": C.fmt(r["dErrorSlope"], 4),
            "PDred": C.fmt(r["pd_reduction"], 2),
            "RegionIoU": C.fmt(r["region_iou"]),
            "Tokens": C.fmt(r["mean_token_cost"], 0),
        })
    if verbose:
        C.print_table("Table 6: Region Extraction Ablation", rows, COLUMNS)
    C.save_csv("table6_ablation.csv", rows, COLUMNS)
    return rows


if __name__ == "__main__":
    run()
