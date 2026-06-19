"""WM-SAR: World-Model-Guided Subgraph Amplification Repair.

A research codebase for repairing failed world-model rollouts by locating and
correcting error-amplifying subgraphs, rather than scanning the last failure or
repairing individual nodes/edges pointwise.
"""

from . import (
    amplification,
    baselines,
    data_generator,
    failure_graph,
    metrics,
    region_extractor,
    repair_executor,
)

__all__ = [
    "data_generator",
    "failure_graph",
    "amplification",
    "region_extractor",
    "baselines",
    "repair_executor",
    "metrics",
]

__version__ = "0.1.0"

SEED = 42
