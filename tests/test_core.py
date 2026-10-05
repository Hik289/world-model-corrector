import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import networkx as nx

from experiments import run_all as paper_runner
from experiments.exp_cascade_gain import inject_with_gain
from wm_sar import baselines
from wm_sar.agent_calling_tree import generate_calling_trees
from wm_sar.failure_graph import build_from_agent_calling_tree
from wm_sar.llm_client import LLMClient
from wm_sar.region_extractor import Region, WMSAR, WMSARConfig
from wm_sar.repair_executor import propagate_effective_error


def _path_graph():
    graph = nx.DiGraph()
    for i, err in enumerate((0.8, 0.6, 0.4, 0.2)):
        graph.add_node(
            f"n{i}",
            err=err,
            unc=0.1,
            cost=1.0,
            time_step=i,
        )
    graph.add_edges_from((f"n{i}", f"n{i + 1}") for i in range(3))
    graph.graph["t_star"] = "n3"
    graph.graph["gt_region"] = {"n0", "n1"}
    return graph


def _failure_graph_from_tree(tree):
    return build_from_agent_calling_tree(
        tree.G,
        dict(zip(tree.node_list, tree.node_states)),
        dict(zip(tree.node_list, tree.true_error)),
        tree.G.graph["t_star"],
    )


class WMSARCoreTests(unittest.TestCase):
    def test_config_and_repair_strength_validation(self):
        with self.assertRaises(ValueError):
            WMSARConfig(max_region_size=0)
        with self.assertRaises(ValueError):
            propagate_effective_error(_path_graph(), set(), strength=1.1)

    def test_budget_selects_one_region_without_overspending(self):
        graph = _path_graph()
        extractor = WMSAR()
        extractor.candidate_regions = lambda _: [
            Region(nodes={"n0", "n1"}, score=2.0),
            Region(nodes={"n2"}, score=1.0),
        ]
        selected = extractor.repair_region(graph, budget=2.0)
        self.assertEqual(selected, {"n0", "n1"})
        self.assertLessEqual(sum(graph.nodes[n]["cost"] for n in selected), 2.0)

    def test_budget_fallback_is_an_affordable_singleton(self):
        graph = _path_graph()
        graph.nodes["n0"]["cost"] = 5.0
        graph.nodes["n2"]["cost"] = 2.0
        extractor = WMSAR()
        extractor.candidate_regions = lambda _: [
            Region(nodes={"n0", "n1"}, score=2.0),
            Region(nodes={"n2"}, score=1.0),
        ]
        self.assertEqual(extractor.repair_region(graph, budget=1.0), {"n1"})
        self.assertEqual(extractor.repair_region(graph, budget=0.0), set())

    def test_extracted_region_is_connected_and_bounded(self):
        graph = _path_graph()
        config = WMSARConfig(max_region_size=3, n_seeds=2)
        region = WMSAR(config).repair_region(graph)
        self.assertLessEqual(len(region), config.max_region_size)
        if len(region) > 1:
            self.assertTrue(nx.is_connected(graph.subgraph(region).to_undirected()))

    def test_pipeline_smoke(self):
        tree = generate_calling_trees(n=1, seed=7)[0]
        graph = _failure_graph_from_tree(tree)
        plan = baselines.wm_sar(graph)
        self.assertNotIn(graph.graph["t_star"], plan.nodes)

    def test_cascade_gain_is_repeatable(self):
        tree = generate_calling_trees(n=1, seed=11)[0]
        graph = _failure_graph_from_tree(tree)
        first = inject_with_gain(graph, 1.1, tree.root_cause_node)
        second = inject_with_gain(graph, 1.1, tree.root_cause_node)
        self.assertEqual(
            nx.get_node_attributes(first, "err"),
            nx.get_node_attributes(second, "err"),
        )

    def test_llm_client_validates_configuration_and_exposes_chat(self):
        with patch.dict(
            os.environ,
            {
                "LLM_API_KEY": "",
                "OPENAI_API_KEY": "",
                "LLM_MODEL": "gpt-4o-mini",
            },
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "missing"):
                LLMClient(backend="openai")

        client = object.__new__(LLMClient)
        client._call = lambda *_: ("ok", 2, 1, 3.0)
        result = client.chat("system", "user")
        self.assertEqual((result.text, result.total_tokens), ("ok", 3))


class PaperRunnerTests(unittest.TestCase):
    def plan(self, *arguments):
        return paper_runner.build_plan(paper_runner.argument_parser().parse_args(arguments))

    def test_defaults_cover_seven_offline_experiments(self):
        plan = self.plan()
        self.assertEqual(len(plan["jobs"]), 7)
        self.assertEqual({job["experiment"] for job in plan["jobs"]},
                         set(paper_runner.OFFLINE_EXPERIMENTS))
        self.assertTrue(all(job["seed"] == 42 for job in plan["jobs"]))
        for job in plan["jobs"]:
            self.assertEqual(job["command"][0], paper_runner.sys.executable)
            self.assertEqual(job["status"], "planned")
        full = next(job for job in plan["jobs"] if job["experiment"] == "full_baselines")
        self.assertEqual(full["effective_parameters"],
                         {"seed": 42, "n_agent": 120, "n_gwm": 80})

    def test_merged_entries_use_existing_files_and_explicit_modes(self):
        jobs = self.plan("--experiments", "full_baselines", "dataset_stats")["jobs"]
        self.assertEqual(Path(jobs[0]["command"][1]).name, "exp1_agent_wm_repair.py")
        self.assertIn("--full-baselines", jobs[0]["command"])
        self.assertEqual(Path(jobs[1]["command"][1]).name, "exp_agent.py")
        self.assertIn("--dataset-stats", jobs[1]["command"])

    def test_no_arguments_preserve_legacy_suite(self):
        with patch("experiments.run_all._run_legacy_suite", return_value=0) as legacy:
            with patch("experiments.run_all.execute_plan") as execute:
                status = paper_runner.main([])
        self.assertEqual(status, 0)
        legacy.assert_called_once_with()
        execute.assert_not_called()

    def test_paper_options_require_explicit_mode(self):
        with patch("experiments.run_all._run_legacy_suite") as legacy:
            with patch("sys.stderr", new_callable=io.StringIO):
                with self.assertRaises(SystemExit) as exc:
                    paper_runner.main(["--dry-run"])
        self.assertEqual(exc.exception.code, 2)
        legacy.assert_not_called()

    def test_seeds_produce_independent_outputs(self):
        jobs = self.plan("--experiments", "main", "ablation", "--seeds", "42", "123")["jobs"]
        self.assertEqual(len(jobs), 4)
        self.assertEqual(len({job["output"] for job in jobs}), 4)

    def test_horizon_is_only_forwarded_to_compatible_entries(self):
        plan = self.plan("--H_max", "8")
        for job in plan["jobs"]:
            accepts = job["experiment"] in ("main", "ablation", "dataset_stats")
            self.assertEqual("--H_max" in job["command"], accepts)
            if accepts:
                self.assertEqual(job["effective_parameters"]["H_max"], 8)
            elif job["experiment"] in ("budget", "cascade", "topology"):
                self.assertEqual(job["effective_parameters"]["H_max"], 32)

    def test_api_requires_explicit_permission(self):
        for experiment in paper_runner.API_EXPERIMENTS:
            with self.assertRaises(ValueError):
                self.plan("--experiments", experiment)

    def test_api_counts_seeds_models_and_sidecars(self):
        jobs = self.plan("--experiments", "llm", "multiapi", "--allow-api",
                         "--llm-seeds", "42", "123", "--model", "selected-model",
                         "--models", "gpt-4o-mini,gpt-4o")["jobs"]
        self.assertEqual(len(jobs), 3)
        llm_jobs = [job for job in jobs if job["experiment"] == "llm"]
        self.assertEqual([job["seed"] for job in llm_jobs], [42, 123])
        for job in llm_jobs:
            self.assertEqual(job["effective_parameters"]["n"], 50)
            self.assertIn("selected-model", job["command"])
            self.assertEqual(len(job["artifacts"]), 3)
            self.assertTrue(job["artifacts"][-1].endswith(f"_seed{job['seed']}.jsonl"))
        multiapi = jobs[-1]
        self.assertEqual(multiapi["effective_parameters"]["n"], 30)
        self.assertEqual(multiapi["effective_parameters"]["models"], "gpt-4o-mini,gpt-4o")

    def test_invalid_parameters_are_rejected(self):
        for arguments in (("--n", "0"), ("--H_max", "0"),
                          ("--seeds", "42", "42"), ("--seeds", "-1"),
                          ("--experiments", "main", "main"),
                          ("--n-agent", "0", "--n-gwm", "0"),
                          ("--n-agent", "-1"),
                          ("--models", "unknown-model")):
            with self.assertRaises(ValueError):
                self.plan(*arguments)

    def test_full_comparison_accepts_a_single_domain(self):
        plan = self.plan("--experiments", "full_baselines", "--n-agent", "0")
        self.assertEqual(plan["jobs"][0]["effective_parameters"]["n_agent"], 0)

    def test_payload_must_match_requested_counts_seed_and_horizon(self):
        job = self.plan("--experiments", "main")["jobs"][0]
        payload = {"n": 50, "seed": 42, "H_max": 32,
                   "summaries": {"ReCore": {"n": 50}}}
        paper_runner._validate_payload(job, payload)
        for key, wrong in (("n", 49), ("seed", 123), ("H_max", 16)):
            with self.assertRaises(ValueError):
                paper_runner._validate_payload(job, dict(payload, **{key: wrong}))
        with self.assertRaises(ValueError):
            paper_runner._validate_payload(job, dict(payload, summaries={"ReCore": {"n": 49}}))

    def test_llm_payload_must_match_model_and_seed(self):
        job = self.plan("--experiments", "llm", "--allow-api", "--llm-n", "1",
                        "--model", "requested-model")["jobs"][0]
        payload = {"n": 1, "seed": 42, "seeds": [42], "model": "requested-model",
                   "run_metadata": {"model": "requested-model"},
                   "summaries": {"ReCore": {"n": 1}},
                   "per_instance": [{"instance_id": "s42_i000", "seed": 42}]}
        paper_runner._validate_payload(job, payload)
        with self.assertRaises(ValueError):
            paper_runner._validate_payload(job, dict(payload, model="other-model"))
        with self.assertRaises(ValueError):
            paper_runner._validate_payload(job, dict(payload, seeds=[123]))

    def test_multiapi_accepts_invalid_and_zero_valid_responses(self):
        job = self.plan("--experiments", "multiapi", "--allow-api", "--multiapi-n", "2",
                        "--models", "gpt-4o-mini")["jobs"][0]
        rows = [{"instance_id": f"s42_i{index:03d}", "seed": 42, "model": "gpt-4o-mini"}
                for index in range(2)]
        for n_valid in (0, 1, 2):
            summary = {"n": n_valid, "n_attempted": 2, "n_valid": n_valid,
                       "n_invalid": 2 - n_valid, "n_unverified": 0,
                       "n_api_errors": 0, "n_token_measured": 2,
                       "recall_denominator": "valid_response"}
            payload = {"n": 2, "seed": 42, "models": ["gpt-4o-mini"],
                       "results": {"gpt-4o-mini": {"ReCore": summary}},
                       "per_instance": rows}
            paper_runner._validate_payload(job, payload)
        with self.assertRaises(ValueError):
            paper_runner._validate_multiapi_summaries(
                {"ReCore": {"n": 1, "n_attempted": 1, "n_valid": 1}}, 2)
        with self.assertRaises(ValueError):
            paper_runner._validate_multiapi_summaries(
                {"ReCore": {"n": 1, "n_attempted": 2, "n_valid": 1,
                            "n_invalid": 2, "n_unverified": 0}}, 2)

    def test_dry_run_does_not_write_or_spawn(self):
        with patch("experiments.run_all.execute_plan") as execute:
            with patch("experiments.run_all.subprocess.run") as child:
                with patch("pathlib.Path.mkdir") as mkdir:
                    with patch("sys.stdout", new_callable=io.StringIO) as output:
                        status = paper_runner.main(["--paper-suite", "--dry-run", "--experiments", "main"])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "planned")
        execute.assert_not_called()
        child.assert_not_called()
        mkdir.assert_not_called()

    def test_existing_output_is_not_overwritten_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.plan("--experiments", "main", "--out-dir", directory)
            output = Path(plan["jobs"][0]["output"])
            output.parent.mkdir(parents=True)
            output.write_text('{"existing": true}', encoding="utf-8")
            with patch("experiments.run_all.subprocess.run") as child:
                with self.assertRaises(FileExistsError):
                    paper_runner.execute_plan(plan)
            child.assert_not_called()
            self.assertFalse(Path(plan["manifest"]).exists())
            self.assertEqual(json.loads(output.read_text())["existing"], True)

    def test_existing_manifest_is_not_overwritten_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.plan("--experiments", "main", "--out-dir", directory)
            Path(plan["manifest"]).write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                paper_runner.execute_plan(plan)

    def test_explicit_overwrite_tracks_success(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.plan("--experiments", "main", "--out-dir", directory)
            output = Path(plan["jobs"][0]["output"])
            output.parent.mkdir(parents=True)
            output.write_text('{"existing": true}', encoding="utf-8")

            def complete(command, **kwargs):
                self.assertFalse(kwargs["shell"])
                self.assertFalse(kwargs["check"])
                output.write_text(json.dumps({"n": 50, "seed": 42, "H_max": 32,
                                              "summaries": {"ReCore": {"n": 50}}}),
                                  encoding="utf-8")
                return subprocess.CompletedProcess(command, 0)

            with patch("experiments.run_all.subprocess.run", side_effect=complete):
                status = paper_runner.execute_plan(plan, overwrite=True)
            self.assertEqual(status, 0)
            saved = json.loads(Path(plan["manifest"]).read_text())
            self.assertEqual(saved["status"], "completed")
            self.assertEqual(saved["jobs"][0]["status"], "completed")
            self.assertEqual(saved["jobs"][0]["returncode"], 0)

    def test_failed_child_is_recorded_and_stops_following_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.plan("--experiments", "main", "ablation", "--out-dir", directory)
            with patch("experiments.run_all.subprocess.run",
                       return_value=subprocess.CompletedProcess([], 9)) as child:
                status = paper_runner.execute_plan(plan)
            self.assertEqual(status, 1)
            child.assert_called_once()
            saved = json.loads(Path(plan["manifest"]).read_text())
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(saved["jobs"][0]["returncode"], 9)
            self.assertEqual(saved["jobs"][0]["status"], "failed")
            self.assertEqual(saved["jobs"][1]["status"], "not_run")

    def test_zero_exit_without_json_output_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.plan("--experiments", "main", "--out-dir", directory)
            with patch("experiments.run_all.subprocess.run",
                       return_value=subprocess.CompletedProcess([], 0)):
                status = paper_runner.execute_plan(plan)
            self.assertEqual(status, 1)
            self.assertEqual(plan["jobs"][0]["error_type"], "FileNotFoundError")

    def test_existing_unmodified_output_is_not_reported_as_fresh_success(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.plan("--experiments", "main", "--out-dir", directory)
            output = Path(plan["jobs"][0]["output"])
            output.parent.mkdir(parents=True)
            output.write_text('{"seed": 42}', encoding="utf-8")
            with patch("experiments.run_all.subprocess.run",
                       return_value=subprocess.CompletedProcess([], 0)):
                status = paper_runner.execute_plan(plan, overwrite=True)
            self.assertEqual(status, 1)
            self.assertEqual(plan["jobs"][0]["status"], "failed")
            self.assertIn("refresh", plan["jobs"][0]["error"])

    def test_interruption_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.plan("--experiments", "main", "--out-dir", directory)
            with patch("experiments.run_all.subprocess.run", side_effect=KeyboardInterrupt):
                status = paper_runner.execute_plan(plan)
            self.assertEqual(status, 130)
            saved = json.loads(Path(plan["manifest"]).read_text())
            self.assertEqual(saved["status"], "interrupted")


if __name__ == "__main__":
    unittest.main()
