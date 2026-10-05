import json
import math
import unittest
from unittest.mock import patch

import networkx as nx

from experiments.exp_agent_ablation import run_ablation, variants
from experiments.exp1_agent_wm_repair import (
    aggregate_results, evaluate_plan, main as full_baseline_main,
    method_specs, run_experiment,
)
from experiments.exp_cascade_gain import inject_with_gain
from wm_sar.baselines import RepairPlan
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


def measurement(recovered=True, before=10.0, reduction=6.0, iou=0.5):
    return {
        "recovered": recovered,
        "final_err_before": 1.0,
        "final_err_after": 0.2 if recovered else 0.8,
        "final_err_reduction": 0.8 if recovered else 0.2,
        "downstream_err_before": before,
        "downstream_err_after": before - reduction,
        "downstream_err_reduction": reduction,
        "pd_before": 4,
        "pd_after": 2,
        "pd_reduction": 2,
        "local_inconsistency": 1,
        "region_iou": iou,
        "region_size": 1,
    }


def result(recovered=True, tokens=100.0, before=10.0, reduction=6.0, iou=0.5):
    values = measurement(recovered, before, reduction, iou)
    values["downstream_err_reduction_pct"] = 100.0 * reduction / before if before > 0 else None
    return {
        "metrics": values,
        "token_cost": tokens,
        "latency_proxy": 1.0,
        "n_edits": 1,
        "is_subgraph": True,
    }


class FullBaselineTests(unittest.TestCase):
    def test_registry_matches_fourteen_table_methods(self):
        specs = method_specs()
        self.assertEqual(len(specs), 14)
        names = {name for name, _, _ in specs}
        self.assertEqual(names, {
            "ReCore", "Oracle-Region", "LLMRepair-Full-Plan", "TraceScan-Full-Point",
            "Top-B-Nodes (K=3)", "TraceScan-w4-Point", "Top-B-Edges (K=3)",
            "PageRank-Subgraph", "k-hop-k2", "Uncertainty-Subgraph",
            "TraceScan-w2-Point", "TraceScan-w1-Point", "LastError-Point",
            "FirstFailedCall-Point",
        })
        settings = {name: params for name, _, params in specs}
        self.assertEqual(settings["Top-B-Nodes (K=3)"]["budget"], 3)
        self.assertEqual(settings["Top-B-Edges (K=3)"]["budget"], 3)
        self.assertEqual(settings["ReCore"]["repair_cost_budget"], 14.0)

    def test_token_denominators_are_distinct(self):
        summary = aggregate_results([result(True, 100.0), result(False, 300.0)])
        self.assertEqual(summary["n_recovered"], 1)
        self.assertEqual(summary["recovery"], 0.5)
        self.assertEqual(summary["mean_token_cost"], 200.0)
        self.assertEqual(summary["mean_tokens_successful_only"], 100.0)
        self.assertEqual(summary["total_attempted_tokens_per_success"], 400.0)
        self.assertEqual(summary["table_tok_per_rec"], 100.0)
        self.assertFalse(summary["table_tok_per_rec_uses_attempt_mean_fallback"])

    def test_zero_success_uses_null_rates_and_explicit_table_fallback(self):
        summary = aggregate_results([result(False, 100.0), result(False, 300.0)])
        self.assertIsNone(summary["mean_tokens_successful_only"])
        self.assertIsNone(summary["total_attempted_tokens_per_success"])
        self.assertEqual(summary["table_tok_per_rec"], 200.0)
        self.assertTrue(summary["table_tok_per_rec_uses_attempt_mean_fallback"])
        json.dumps(summary, allow_nan=False)

    def test_percentage_reductions_use_explicit_denominators(self):
        rows = [result(before=10.0, reduction=5.0), result(before=90.0, reduction=9.0),
                result(before=0.0, reduction=0.0, iou=None)]
        summary = aggregate_results(rows)
        self.assertEqual(summary["mean_downstream_err_reduction_pct"], 30.0)
        self.assertEqual(summary["pooled_downstream_err_reduction_pct"], 14.0)
        self.assertEqual(summary["n_downstream_err_reduction_pct"], 2)
        self.assertEqual(summary["n_region_iou"], 2)

    def test_zero_before_error_does_not_create_a_fake_percentage(self):
        summary = aggregate_results([result(before=0.0, reduction=0.0)])
        self.assertIsNone(summary["mean_downstream_err_reduction_pct"])
        self.assertIsNone(summary["pooled_downstream_err_reduction_pct"])

    def test_plan_preserves_raw_selection_and_missing_iou(self):
        graph = nx.DiGraph([("a", "b")])
        plan = RepairPlan("legacy-name", {"b", "a"}, 100.0, 1, True, 0.5)
        with patch("experiments.exp1_agent_wm_repair.re.measure_recovery",
                   return_value=measurement(iou=math.nan)):
            row = evaluate_plan(graph, lambda _: plan, {"budget": 2})
        self.assertEqual(row["selected_nodes"], ["a", "b"])
        self.assertEqual(row["parameters"], {"budget": 2})
        self.assertEqual(row["plan_method"], "legacy-name")
        self.assertIsNone(row["metrics"]["region_iou"])
        json.dumps(row, allow_nan=False)

    def test_unknown_selected_node_is_rejected(self):
        plan = RepairPlan("test", {"missing"}, 100.0, 1, True, 0.5)
        with self.assertRaises(ValueError):
            evaluate_plan(nx.DiGraph(), lambda _: plan, {})

    def test_nonfinite_cost_is_rejected(self):
        graph = nx.DiGraph()
        graph.add_node("a")
        plan = RepairPlan("test", {"a"}, math.inf, 1, True, 0.5)
        with self.assertRaises(ValueError):
            evaluate_plan(graph, lambda _: plan, {})

    def test_single_domain_is_supported_without_silent_sample_loss(self):
        graph = nx.DiGraph([("a", "b")])
        graph.graph.update(t_star="b", gt_region={"a"}, rollout_id="r1", failure_type="test")
        data = {"agent": [object()], "gwm": [],
                "stats": {"n_total": 1, "mean_gwm_horizon": math.nan}}
        plan = RepairPlan("test", {"a"}, 100.0, 1, True, 0.5)
        specs = [("test", lambda _: plan, {})]
        with patch("experiments.exp1_agent_wm_repair.dg.generate_dataset", return_value=data) as generate:
            with patch("experiments.exp1_agent_wm_repair.fg.world_model_failure_to_graph", return_value=graph):
                with patch("experiments.exp1_agent_wm_repair.method_specs", return_value=specs):
                    with patch("experiments.exp1_agent_wm_repair.re.measure_recovery", return_value=measurement()):
                        output = run_experiment(n_agent=1, n_gwm=0, seed=7)
        generate.assert_called_once_with(n_agent=1, n_gwm=0, seed=7)
        self.assertEqual(output["n"], 1)
        self.assertEqual(set(output["summaries_by_domain"]), {"agent"})
        self.assertEqual(output["per_instance"][0]["instance_id"], "agent_s7_i0000")
        self.assertEqual(output["token_cost_kind"], "synthetic_proxy")
        self.assertEqual(output["repair_kind"], "simulated")
        self.assertEqual(output["llm_api_calls"], 0)
        self.assertIsNone(output["dataset_stats"]["mean_gwm_horizon"])
        json.dumps(output, allow_nan=False)

    def test_gwm_only_uses_separate_generation_seed(self):
        graph = nx.DiGraph([("a", "b")])
        graph.graph.update(t_star="b", gt_region={"a"})
        data = {"agent": [], "gwm": [object()], "stats": {"n_total": 1}}
        plan = RepairPlan("test", {"a"}, 100.0, 1, True, 0.5)
        with patch("experiments.exp1_agent_wm_repair.dg.generate_dataset", return_value=data):
            with patch("experiments.exp1_agent_wm_repair.fg.world_model_failure_to_graph", return_value=graph):
                with patch("experiments.exp1_agent_wm_repair.method_specs", return_value=[("test", lambda _: plan, {})]):
                    with patch("experiments.exp1_agent_wm_repair.re.measure_recovery", return_value=measurement()):
                        output = run_experiment(n_agent=0, n_gwm=1, seed=7)
        self.assertEqual(set(output["summaries_by_domain"]), {"gwm"})
        self.assertEqual(output["per_instance"][0]["instance_id"], "gwm_s8_i0000")

    def test_generator_shortfall_is_not_silently_accepted(self):
        data = {"agent": [], "gwm": [], "stats": {}}
        with patch("experiments.exp1_agent_wm_repair.dg.generate_dataset", return_value=data):
            with self.assertRaises(ValueError):
                run_experiment(n_agent=1, n_gwm=0)

    def test_failed_method_does_not_drop_instance(self):
        graph = nx.DiGraph([("a", "b")])
        data = {"agent": [object()], "gwm": [], "stats": {}}
        with patch("experiments.exp1_agent_wm_repair.dg.generate_dataset", return_value=data):
            with patch("experiments.exp1_agent_wm_repair.fg.world_model_failure_to_graph", return_value=graph):
                with patch("experiments.exp1_agent_wm_repair.method_specs", return_value=[("broken", lambda _: 1 / 0, {})]):
                    with self.assertRaisesRegex(RuntimeError, "broken.*agent_s42_i0000"):
                        run_experiment(n_agent=1, n_gwm=0)

    def test_invalid_counts_and_budgets_are_rejected(self):
        for kwargs in ({"n_agent": 0, "n_gwm": 0}, {"n_agent": -1},
                       {"n_gwm": True}, {"seed": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                run_experiment(**kwargs)
        for kwargs in ({"top_k": 0}, {"trace_budget": True}, {"repair_budget": math.nan}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                method_specs(**kwargs)
        with self.assertRaises(ValueError):
            aggregate_results([])

    def test_legacy_cli_keeps_original_run(self):
        with patch("experiments.exp1_agent_wm_repair.run", return_value=[]) as legacy:
            with patch("experiments.exp1_agent_wm_repair.run_experiment") as full:
                full_baseline_main([])
        legacy.assert_called_once_with()
        full.assert_not_called()

    def test_full_cli_forwards_all_configuration(self):
        with patch("experiments.exp1_agent_wm_repair.run_experiment",
                   return_value={"n": 3}) as full:
            with patch("experiments.exp1_agent_wm_repair.run") as legacy:
                with patch("pathlib.Path.mkdir") as mkdir:
                    with patch("pathlib.Path.write_text") as write:
                        with patch("builtins.print"):
                            full_baseline_main([
                                "--full-baselines", "--n-agent", "2", "--n-gwm", "1",
                                "--seed", "7", "--top-k", "5", "--trace-budget", "6",
                                "--repair-budget", "9", "--out", "temporary/full.json",
                            ])
        full.assert_called_once_with(2, 1, 7, 5, 6, 9.0)
        legacy.assert_not_called()
        mkdir.assert_called_once_with(parents=True, exist_ok=True)
        self.assertEqual(json.loads(write.call_args[0][0]), {"n": 3})
        self.assertEqual(write.call_args[1]["encoding"], "utf-8")


if __name__ == "__main__":
    unittest.main()
