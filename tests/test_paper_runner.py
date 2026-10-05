import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from experiments import run_paper


class PaperRunnerTests(unittest.TestCase):
    def plan(self, *arguments):
        return run_paper.build_plan(run_paper.argument_parser().parse_args(arguments))

    def test_defaults_cover_seven_offline_experiments(self):
        plan = self.plan()
        self.assertEqual(len(plan["jobs"]), 7)
        self.assertEqual({job["experiment"] for job in plan["jobs"]},
                         set(run_paper.OFFLINE_EXPERIMENTS))
        self.assertTrue(all(job["seed"] == 42 for job in plan["jobs"]))
        for job in plan["jobs"]:
            self.assertEqual(job["command"][0], run_paper.sys.executable)
            self.assertEqual(job["status"], "planned")
        full = next(job for job in plan["jobs"] if job["experiment"] == "full_baselines")
        self.assertEqual(full["effective_parameters"],
                         {"seed": 42, "n_agent": 120, "n_gwm": 80})

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
        for experiment in run_paper.API_EXPERIMENTS:
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
        run_paper._validate_payload(job, payload)
        for key, wrong in (("n", 49), ("seed", 123), ("H_max", 16)):
            with self.assertRaises(ValueError):
                run_paper._validate_payload(job, dict(payload, **{key: wrong}))
        with self.assertRaises(ValueError):
            run_paper._validate_payload(job, dict(payload, summaries={"ReCore": {"n": 49}}))

    def test_llm_payload_must_match_model_and_seed(self):
        job = self.plan("--experiments", "llm", "--allow-api", "--llm-n", "1",
                        "--model", "requested-model")["jobs"][0]
        payload = {"n": 1, "seed": 42, "seeds": [42], "model": "requested-model",
                   "run_metadata": {"model": "requested-model"},
                   "summaries": {"ReCore": {"n": 1}},
                   "per_instance": [{"instance_id": "s42_i000", "seed": 42}]}
        run_paper._validate_payload(job, payload)
        with self.assertRaises(ValueError):
            run_paper._validate_payload(job, dict(payload, model="other-model"))
        with self.assertRaises(ValueError):
            run_paper._validate_payload(job, dict(payload, seeds=[123]))

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
            run_paper._validate_payload(job, payload)
        with self.assertRaises(ValueError):
            run_paper._validate_multiapi_summaries(
                {"ReCore": {"n": 1, "n_attempted": 1, "n_valid": 1}}, 2)
        with self.assertRaises(ValueError):
            run_paper._validate_multiapi_summaries(
                {"ReCore": {"n": 1, "n_attempted": 2, "n_valid": 1,
                            "n_invalid": 2, "n_unverified": 0}}, 2)

    def test_dry_run_does_not_write_or_spawn(self):
        with patch("experiments.run_paper.execute_plan") as execute:
            with patch("experiments.run_paper.subprocess.run") as child:
                with patch("pathlib.Path.mkdir") as mkdir:
                    with patch("sys.stdout", new_callable=io.StringIO) as output:
                        status = run_paper.main(["--dry-run", "--experiments", "main"])
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
            with patch("experiments.run_paper.subprocess.run") as child:
                with self.assertRaises(FileExistsError):
                    run_paper.execute_plan(plan)
            child.assert_not_called()
            self.assertFalse(Path(plan["manifest"]).exists())
            self.assertEqual(json.loads(output.read_text())["existing"], True)

    def test_existing_manifest_is_not_overwritten_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.plan("--experiments", "main", "--out-dir", directory)
            Path(plan["manifest"]).write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run_paper.execute_plan(plan)

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

            with patch("experiments.run_paper.subprocess.run", side_effect=complete):
                status = run_paper.execute_plan(plan, overwrite=True)
            self.assertEqual(status, 0)
            saved = json.loads(Path(plan["manifest"]).read_text())
            self.assertEqual(saved["status"], "completed")
            self.assertEqual(saved["jobs"][0]["status"], "completed")
            self.assertEqual(saved["jobs"][0]["returncode"], 0)

    def test_failed_child_is_recorded_and_stops_following_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.plan("--experiments", "main", "ablation", "--out-dir", directory)
            with patch("experiments.run_paper.subprocess.run",
                       return_value=subprocess.CompletedProcess([], 9)) as child:
                status = run_paper.execute_plan(plan)
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
            with patch("experiments.run_paper.subprocess.run",
                       return_value=subprocess.CompletedProcess([], 0)):
                status = run_paper.execute_plan(plan)
            self.assertEqual(status, 1)
            self.assertEqual(plan["jobs"][0]["error_type"], "FileNotFoundError")

    def test_existing_unmodified_output_is_not_reported_as_fresh_success(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.plan("--experiments", "main", "--out-dir", directory)
            output = Path(plan["jobs"][0]["output"])
            output.parent.mkdir(parents=True)
            output.write_text('{"seed": 42}', encoding="utf-8")
            with patch("experiments.run_paper.subprocess.run",
                       return_value=subprocess.CompletedProcess([], 0)):
                status = run_paper.execute_plan(plan, overwrite=True)
            self.assertEqual(status, 1)
            self.assertEqual(plan["jobs"][0]["status"], "failed")
            self.assertIn("refresh", plan["jobs"][0]["error"])

    def test_interruption_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.plan("--experiments", "main", "--out-dir", directory)
            with patch("experiments.run_paper.subprocess.run", side_effect=KeyboardInterrupt):
                status = run_paper.execute_plan(plan)
            self.assertEqual(status, 130)
            saved = json.loads(Path(plan["manifest"]).read_text())
            self.assertEqual(saved["status"], "interrupted")


if __name__ == "__main__":
    unittest.main()
