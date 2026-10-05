import unittest
from unittest.mock import patch

import networkx as nx

from experiments.exp_agent_ablation import run_ablation, variants
from experiments.exp_cascade_gain import inject_with_gain
from wm_sar.engineering_baselines import (
    RepairResult, _aggregate, run_all_baselines, window_repair, wmsar_repair,
)
from wm_sar.region_extractor import WMSARConfig


def result_for(horizon):
    profile = {step: float(step) for step in range(1, horizon + 1)}
    return RepairResult(
        method="test", selected_nodes={"a"}, is_connected=True, err_cover=1.0,
        region_size=1, rho_before=2.0, rho_after_region=1.0, rho_reduction=1.0,
        mse_profile_before=profile, mse_profile_after=profile,
        growth_slope_before=0.0, growth_slope_after=0.0, iou_vs_gt=1.0,
        return_bound_before=2.0, return_bound_after=1.0, regret_reduction=1.0,
    )


class PaperExperimentTests(unittest.TestCase):
    def test_aggregation_does_not_invent_uncomputed_horizons(self):
        summary = _aggregate([result_for(5), result_for(5)])
        self.assertEqual(set(summary["NodeMSE_after"]), {1, 2, 4, 5})
        self.assertNotIn(32, summary["NodeMSE_after"])
        self.assertEqual(summary["std_rho_reduction"], 0.0)
        self.assertEqual(summary["NodeMSE_after_std"][5], 0.0)

    def test_default_methods_receive_requested_evaluation(self):
        graph = nx.DiGraph()
        seen = []

        def method(G, **evaluation):
            seen.append(evaluation)
            return result_for(evaluation["H_max"])

        with patch("wm_sar.engineering_baselines.ALL_BASELINES", {"test": method}):
            summaries = run_all_baselines([graph], H_max=5, weight_norm=0.7,
                                          gamma=0.8, verbose=False)
        self.assertEqual(seen, [{"H_max": 5, "weight_norm": 0.7, "gamma": 0.8}])
        self.assertIn(5, summaries["test"]["NodeMSE_after"])

    def test_failed_graph_is_not_silently_dropped(self):
        with patch("wm_sar.engineering_baselines.ALL_BASELINES",
                   {"test": lambda G, **kwargs: 1 / 0}):
            with self.assertRaises(RuntimeError):
                run_all_baselines([nx.DiGraph()], verbose=False)

    def test_selection_and_evaluation_use_same_weight(self):
        with patch("wm_sar.engineering_baselines.WMSAR") as selector:
            selector.return_value.repair_region.return_value = set()
            with patch("wm_sar.engineering_baselines._evaluate_repair") as evaluate:
                wmsar_repair(nx.DiGraph(), weight_norm=0.7, gamma=0.8)
        config = selector.call_args[0][0]
        self.assertEqual(config.weight_norm, 0.7)
        self.assertEqual(config.gamma, 0.8)
        self.assertEqual(evaluate.call_args[1]["weight_norm"], 0.7)
        with self.assertRaises(ValueError):
            wmsar_repair(nx.DiGraph(), WMSARConfig(), weight_norm=0.7)

    def test_zero_error_window_keeps_requested_size(self):
        graph = nx.DiGraph()
        graph.add_nodes_from((str(i), {"err": 0.0, "time_step": i}) for i in range(5))
        with patch("wm_sar.engineering_baselines._evaluate_repair") as evaluate:
            window_repair(graph, window=2)
        self.assertEqual(evaluate.call_args[0][1], {"0", "1"})

    def test_ablation_has_all_six_variants_on_identical_graphs(self):
        graphs = [nx.DiGraph(), nx.DiGraph()]
        with patch("experiments.exp_agent_ablation.wmsar_repair",
                   side_effect=lambda graph, config, H_max: result_for(H_max)) as evaluate:
            summaries = run_ablation(graphs, H_max=5)
        self.assertEqual(len(variants()), 6)
        self.assertEqual(len(summaries), 6)
        self.assertEqual(evaluate.call_count, 12)
        self.assertTrue(all(summary["n"] == 2 for summary in summaries.values()))
        self.assertTrue(all(summary["method"] == name for name, summary in summaries.items()))

    def test_cascade_starts_at_explicit_root_and_preserves_original(self):
        graph = nx.DiGraph([("root", "child"), ("child", "leaf")])
        graph.graph["gt_region"] = {"leaf"}
        repaired_input = inject_with_gain(graph, 1.0, "root")
        self.assertEqual(repaired_input.nodes["root"]["err"], 0.6)
        self.assertGreaterEqual(repaired_input.nodes["leaf"]["err"], 0.6)
        self.assertNotIn("err", graph.nodes["root"])
        self.assertEqual(repaired_input.graph["root_cause_node"], "root")
        with self.assertRaises(ValueError):
            inject_with_gain(graph, 1.0)


if __name__ == "__main__":
    unittest.main()
