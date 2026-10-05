import unittest
from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx

from experiments.exp_dataset_stats import count_summary, distribution, failure_type, summarize_trees


def make_tree(root_type="planner", recorded_failure=None):
    graph = nx.DiGraph()
    graph.add_node("root", node_type=root_type, state=[0.0] * 8, err=1.0)
    graph.add_node("end", node_type="final_answer", state=[0.0] * 8, err=0.0)
    graph.add_edge("root", "end", edge_type="calls")
    graph.graph["gt_region"] = {"root"}
    if recorded_failure is not None:
        graph.graph["failure_type"] = recorded_failure
    return SimpleNamespace(G=graph, root_cause_node="root",
                           failure_desc="Agent calling-tree failure: tool_misfire injected at planner")


class DatasetStatisticsTests(unittest.TestCase):
    def test_distribution_is_computed_and_population_scaled(self):
        self.assertEqual(distribution([1, 3]), {"n": 2, "mean": 2.0, "std": 1.0,
                                               "min": 1.0, "max": 3.0})
        self.assertIsNone(distribution([])["mean"])
        with self.assertRaises(ValueError):
            distribution([float("nan")])

    def test_counts_use_actual_denominators(self):
        summary = count_summary(["planner", "executor", "executor"])
        self.assertEqual(summary["counts"], {"executor": 2, "planner": 1})
        self.assertAlmostEqual(summary["percent"]["planner"], 100 / 3)
        self.assertEqual(count_summary([])["percent"], {})

    def test_failure_metadata_precedes_description(self):
        self.assertEqual(failure_type(make_tree()), "tool_misfire")
        self.assertEqual(failure_type(make_tree(recorded_failure="prediction_drift")), "prediction_drift")
        tree = make_tree()
        tree.failure_desc = "unrecognized input"
        self.assertEqual(failure_type(tree), "unknown")

    def test_full_statistics_include_observed_types_and_diagnostics(self):
        trees = [make_tree(), make_tree(root_type="executor")]
        with patch("experiments.exp_dataset_stats.amp.simulate_error_propagation",
                   return_value={1: 2.0, 2: 3.0}) as simulate:
            output = summarize_trees(trees, seed=7, H_max=2)
        summary = output["statistics"]
        self.assertEqual(output["n"], 2)
        self.assertEqual(summary["n_node_types_observed"], 3)
        self.assertEqual(summary["n_edge_types_observed"], 1)
        self.assertEqual(summary["state_dimension_counts"], {8: 4})
        self.assertEqual(summary["root_cause_types"]["percent"]["planner"], 50.0)
        self.assertEqual(summary["zero_geaf_graph_count"], 2)
        self.assertEqual(summary["dag_count"], 2)
        self.assertEqual(output["per_instance"][1]["instance_id"], "s7_i001")
        self.assertEqual(output["per_instance"][0]["unrepaired_NodeMSE"], {"1": 2.0, "2": 3.0})
        self.assertIsNone(output["per_instance"][0]["unrepaired_growth_slope"])
        self.assertEqual(summary["unrepaired_growth_slope"]["n"], 0)
        self.assertIsNone(summary["unrepaired_growth_slope"]["mean"])
        self.assertEqual(simulate.call_count, 2)
        self.assertEqual(simulate.call_args.kwargs["H"], 2)

    def test_invalid_input_is_not_silently_counted(self):
        with self.assertRaises(ValueError):
            summarize_trees([])
        tree = make_tree()
        tree.root_cause_node = "missing"
        with self.assertRaises(ValueError):
            summarize_trees([tree])
        with self.assertRaises(ValueError):
            summarize_trees([make_tree()], H_max=0)


if __name__ == "__main__":
    unittest.main()
