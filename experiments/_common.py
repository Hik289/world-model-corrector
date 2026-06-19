"""Shared utilities for the WM-SAR experiment scripts.

Centralizes: dataset construction (seeded), failure-graph building, table
printing/saving, and figure output paths. Importing this module makes the
``wm_sar`` package importable when scripts are run from anywhere.
"""

from __future__ import annotations

import csv
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# make the project root importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wm_sar import data_generator as dg  # noqa: E402
from wm_sar import failure_graph as fg  # noqa: E402

SEED = 42
ANALYSIS_DIR = os.path.join(ROOT, "analysis")
FIG_DIR = os.path.join(ROOT, "figures")
DATA_DIR = os.path.join(ROOT, "data")
for _d in (ANALYSIS_DIR, FIG_DIR, DATA_DIR):
    os.makedirs(_d, exist_ok=True)

_CACHE: dict = {}


def build_dataset(n_agent: int = 120, n_gwm: int = 80, seed: int = SEED):
    """Return (dataset, agent_graphs, gwm_graphs), cached per process."""
    key = (n_agent, n_gwm, seed)
    if key in _CACHE:
        return _CACHE[key]
    ds = dg.generate_dataset(n_agent=n_agent, n_gwm=n_gwm, seed=seed)
    agent_graphs = [fg.world_model_failure_to_graph(r) for r in ds["agent"]]
    gwm_graphs = [fg.world_model_failure_to_graph(r) for r in ds["gwm"]]
    _CACHE[key] = (ds, agent_graphs, gwm_graphs)
    return _CACHE[key]


def print_table(title: str, rows: list[dict], columns: list[str]) -> None:
    from tabulate import tabulate

    table = [[r.get(c, "") for c in columns] for r in rows]
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    print(tabulate(table, headers=columns, floatfmt=".3f", tablefmt="github"))


def save_csv(name: str, rows: list[dict], columns: list[str]) -> str:
    path = os.path.join(ANALYSIS_DIR, name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def fmt(x, nd: int = 3):
    if isinstance(x, float):
        if x == float("inf"):
            return "inf"
        return round(x, nd)
    return x
