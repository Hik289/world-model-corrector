from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wm_sar import amplification as amp
from wm_sar.agent_calling_tree import FEAT_NAMES, generate_calling_trees


def distribution(values):
    values = np.asarray(list(values), dtype=float)
    if values.size == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    if not np.all(np.isfinite(values)):
        raise ValueError("dataset statistics contain non-finite values")
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def count_summary(values):
    counts = Counter(str(value) for value in values)
    total = sum(counts.values())
    return {
        "counts": dict(sorted(counts.items())),
        "percent": {key: 100.0 * count / total for key, count in sorted(counts.items())},
        "n": total,
    }


def failure_type(tree):
    recorded = tree.G.graph.get("failure_type")
    if recorded is not None:
        return str(recorded)
    match = re.match(r"Agent calling-tree failure: ([A-Za-z0-9_]+) injected at ",
                     tree.failure_desc)
    return match.group(1) if match else "unknown"


def describe_tree(tree, instance_id, H_max=32, weight_norm=0.9):
    if H_max < 1 or not np.isfinite(weight_norm) or weight_norm < 0:
        raise ValueError("H_max must be positive and weight_norm finite and non-negative")
    graph = tree.G
    if not graph or tree.root_cause_node not in graph:
        raise ValueError("each tree must have a nonempty graph and an observed root-cause ID")
    node_types = Counter(str(data.get("node_type", "unknown"))
                         for _, data in graph.nodes(data=True))
    edge_types = Counter(str(data.get("edge_type", "unknown"))
                         for _, _, data in graph.edges(data=True))
    dimensions = []
    for _, data in graph.nodes(data=True):
        state = np.asarray(data.get("state", []))
        if state.ndim != 1:
            raise ValueError("node states must be one-dimensional feature vectors")
        dimensions.append(int(state.size))
    L_X, L_A, M_X, M_A = amp.coupling_blocks_region(graph, set(graph), weight_norm)
    rho = float(amp.rho_B_from_blocks(L_X, L_A, M_X, M_A))
    profile = amp.simulate_error_propagation(graph, set(), H=H_max,
                                            weight_norm=weight_norm)
    slope_horizons = [horizon for horizon in profile if 4 <= horizon <= H_max]
    growth_slope = (amp.error_growth_slope(profile, h_start=4, h_end=H_max)
                    if len(slope_horizons) >= 2 else None)
    adjacency, _, _ = amp.adjacency_matrix(graph)
    local_amplification = amp.geaf_all(graph, H=4, weight_norm=weight_norm)
    gt_region = set(graph.graph.get("gt_region", set()))
    if not gt_region.issubset(graph.nodes):
        raise ValueError("ground-truth proxy contains nodes outside the graph")
    return {
        "instance_id": instance_id,
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "node_type_counts": dict(sorted(node_types.items())),
        "edge_type_counts": dict(sorted(edge_types.items())),
        "state_dimension_counts": dict(sorted(Counter(dimensions).items())),
        "root_cause_node": str(tree.root_cause_node),
        "root_cause_type": str(graph.nodes[tree.root_cause_node].get("node_type", "unknown")),
        "failure_type": failure_type(tree),
        "gt_region_size": len(gt_region),
        "is_dag": bool(nx.is_directed_acyclic_graph(graph)),
        "adjacency_spectral_radius": float(amp._spectral_radius(adjacency)),
        "geaf_all_zero": bool(all(value == 0.0 for value in local_amplification.values())),
        "rho_B": rho,
        "operator_blocks": {"L_X": float(L_X), "L_A": float(L_A),
                            "M_X": float(M_X), "M_A": float(M_A)},
        "strict_superadditivity": bool(rho > max(L_X, M_A) + 1e-6),
        "unrepaired_growth_slope": growth_slope,
        "unrepaired_NodeMSE": {str(horizon): float(value) for horizon, value in profile.items()},
    }


def summarize_trees(trees, seed=42, H_max=32, weight_norm=0.9):
    trees = list(trees)
    if not trees:
        raise ValueError("at least one calling tree is required")
    rows = [describe_tree(tree, f"s{seed}_i{index:03d}", H_max, weight_norm)
            for index, tree in enumerate(trees)]
    node_types, edge_types, state_dimensions = Counter(), Counter(), Counter()
    for row in rows:
        node_types.update(row["node_type_counts"])
        edge_types.update(row["edge_type_counts"])
        state_dimensions.update(row["state_dimension_counts"])
    summary = {
        "nodes_per_graph": distribution(row["n_nodes"] for row in rows),
        "edges_per_graph": distribution(row["n_edges"] for row in rows),
        "gt_region_size": distribution(row["gt_region_size"] for row in rows),
        "rho_B": distribution(row["rho_B"] for row in rows),
        "unrepaired_growth_slope": distribution(row["unrepaired_growth_slope"] for row in rows
                                                if row["unrepaired_growth_slope"] is not None),
        "node_type_counts": dict(sorted(node_types.items())),
        "edge_type_counts": dict(sorted(edge_types.items())),
        "state_dimension_counts": dict(sorted(state_dimensions.items())),
        "n_node_types_observed": len(node_types),
        "n_edge_types_observed": len(edge_types),
        "root_cause_types": count_summary(row["root_cause_type"] for row in rows),
        "failure_types": count_summary(row["failure_type"] for row in rows),
        "strict_superadditivity_percent": 100.0 * sum(row["strict_superadditivity"] for row in rows) / len(rows),
        "dag_count": sum(row["is_dag"] for row in rows),
        "zero_geaf_graph_count": sum(row["geaf_all_zero"] for row in rows),
    }
    return {
        "n": len(rows), "seed": seed, "H_max": H_max, "weight_norm": weight_norm,
        "feature_names": list(FEAT_NAMES),
        "dispersion": "population_standard_deviation",
        "superadditivity_tolerance": 1e-6,
        "geaf_neighborhood_horizon": 4,
        "growth_slope_horizon_start": 4,
        "growth_slope_unavailable_policy": "null_when_fewer_than_two_horizons",
        "rollout_model": "legacy_single_channel_proxy",
        "ground_truth_region_kind": "node_error_above_graph_mean_proxy",
        "failure_type_source": "graph_metadata_or_generator_description",
        "statistics": summary,
        "per_instance": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--H_max", type=int, default=32)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "results" / "exp_dataset_stats.json")
    args = parser.parse_args()
    if args.n < 1 or args.H_max < 1 or args.seed < 0:
        parser.error("n and H_max must be positive and seed non-negative")
    trees = generate_calling_trees(n=args.n, seed=args.seed)
    payload = summarize_trees(trees, args.seed, args.H_max)
    serialized = json.dumps(payload, indent=2, allow_nan=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(serialized + "\n", encoding="utf-8")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
