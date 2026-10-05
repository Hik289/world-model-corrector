import csv
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from experiments.analyze_llm_results import analyze_records, load_records, write_csv


def result(success, tokens=10, valid=True, api_error=""):
    return {
        "rec_exact": success,
        "rec_type": success,
        "rec_hop2": success,
        "tokens": tokens,
        "valid_json": valid,
        "valid_response": valid,
        "api_error": api_error,
    }


def record(seed=42, index=0, model="model-a", methods=None, max_tokens=512):
    return {
        "instance_id": f"s{seed}_i{index:03d}",
        "seed": seed,
        "true_root": "n0",
        "run_metadata": {
            "context_protocol": "test-protocol-v1",
            "model": model,
            "backend": "test-backend",
            "temperature": 0,
            "max_tokens": max_tokens,
        },
        "results": methods if methods is not None else {"WM-SAR-LLM": result(1)},
    }


class LLMResultAnalysisTests(unittest.TestCase):
    def load_payloads(self, directory, payloads):
        paths = []
        for index, (suffix, payload) in enumerate(payloads):
            path = Path(directory) / f"input{index}.{suffix}"
            if suffix == "jsonl":
                path.write_text("\n".join(json.dumps(row) for row in payload) + "\n", encoding="utf-8")
            else:
                path.write_text(json.dumps(payload), encoding="utf-8")
            paths.append(path)
        return load_records(paths)

    def test_json_and_jsonl_duplicates_are_removed(self):
        row = record()
        with tempfile.TemporaryDirectory() as directory:
            rows, provenance = self.load_payloads(directory, [
                ("json", {"model": "model-a", "per_instance": [row]}),
                ("jsonl", [row]),
            ])
        self.assertEqual(len(rows), 1)
        self.assertEqual(provenance["n_exact_duplicates_removed"], 1)

    def test_conflicting_duplicate_is_rejected(self):
        first = record()
        second = deepcopy(first)
        second["results"]["WM-SAR-LLM"]["tokens"] = 11
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
                self.load_payloads(directory, [("jsonl", [first, second])])

    def test_models_and_protocol_settings_are_never_pooled(self):
        source = [record(), record(model="model-b"), record(max_tokens=400)]
        with tempfile.TemporaryDirectory() as directory:
            rows, provenance = self.load_payloads(directory, [("json", {"per_instance": source})])
        report = analyze_records(rows, provenance)
        self.assertEqual(len(report["groups"]), 3)
        for group in report["groups"]:
            self.assertEqual(group["methods"]["WM-SAR-LLM"]["pooled"]["n_attempted"], 1)

    def test_every_method_has_cross_seed_statistics(self):
        source = [
            record(seed=42, methods={"WM-SAR-LLM": result(1), "TopK-5-LLM": result(0)}),
            record(seed=123, methods={"WM-SAR-LLM": result(0), "TopK-5-LLM": result(1)}),
        ]
        report = analyze_records(source)
        methods = report["groups"][0]["methods"]
        self.assertEqual(set(methods), {"WM-SAR-LLM", "TopK-5-LLM"})
        self.assertEqual(methods["WM-SAR-LLM"]["display_name"], "ReCore-LLM")
        for details in methods.values():
            stats = details["cross_seed"]["rec_exact_attempted"]
            self.assertEqual(stats["mean_ratio"], 0.5)
            self.assertEqual(stats["population_std_ratio"], 0.5)
            self.assertEqual(stats["coefficient_of_variation_ratio"], 1.0)

    def test_pooled_token_efficiency_is_not_a_mean_of_seed_ratios(self):
        source = [
            record(seed=42, methods={"M": result(1, 10)}),
            record(seed=123, methods={"M": result(1, 100)}),
            record(seed=123, index=1, methods={"M": result(1, 100)}),
            record(seed=123, index=2, methods={"M": result(0, 100)}),
        ]
        details = analyze_records(source)["groups"][0]["methods"]["M"]
        self.assertAlmostEqual(details["pooled"]["tokens_per_rec_exact_recovery"], 310 / 3)
        self.assertAlmostEqual(details["pooled"]["rec_exact_attempted_ratio"], 3 / 4)
        self.assertAlmostEqual(details["cross_seed"]["rec_exact_attempted"]["mean_ratio"], 5 / 6)

    def test_invalid_and_api_failures_remain_in_attempted_denominator(self):
        source = [
            record(index=0, methods={"M": result(1, 10)}),
            record(index=1, methods={"M": result(0, 20, False)}),
            record(index=2, methods={"M": result(0, None, False, "TimeoutError")}),
        ]
        pooled = analyze_records(source)["groups"][0]["methods"]["M"]["pooled"]
        self.assertEqual(pooled["n_attempted"], 3)
        self.assertEqual(pooled["n_valid"], 1)
        self.assertEqual(pooled["n_invalid"], 2)
        self.assertEqual(pooled["n_api_errors"], 1)
        self.assertEqual(pooled["n_token_unknown"], 1)
        self.assertAlmostEqual(pooled["rec_exact_attempted_ratio"], 1 / 3)
        self.assertEqual(pooled["rec_exact_valid_ratio"], 1)
        self.assertIsNone(pooled["total_tokens"])
        self.assertIsNone(pooled["tokens_per_rec_exact_recovery"])
        self.assertEqual(pooled["total_tokens_known"], 30)

    def test_zero_success_and_zero_mean_cv_are_null(self):
        report = analyze_records([record(methods={"M": result(0)})])
        details = report["groups"][0]["methods"]["M"]
        self.assertIsNone(details["pooled"]["tokens_per_rec_exact_recovery"])
        self.assertIsNone(details["cross_seed"]["rec_exact_attempted"]["coefficient_of_variation_ratio"])
        self.assertEqual(details["cross_seed"]["rec_exact_attempted"]["population_std_ratio"], 0)

    def test_no_valid_seed_is_reported_unavailable(self):
        source = [
            record(seed=42, methods={"M": result(1)}),
            record(seed=123, methods={"M": result(0, 10, False)}),
        ]
        details = analyze_records(source)["groups"][0]["methods"]["M"]
        stats = details["cross_seed"]["rec_exact_valid"]
        self.assertEqual(stats["n_seeds_total"], 2)
        self.assertEqual(stats["n_seeds_available"], 1)
        self.assertEqual(stats["n_seeds_unavailable"], 1)
        self.assertEqual(stats["mean_ratio"], 1)

    def test_missing_method_seeds_and_instances_are_not_imputed(self):
        source = [
            record(seed=42, index=0, methods={"M": result(1), "Other": result(1)}),
            record(seed=42, index=1, methods={"Other": result(0)}),
            record(seed=123, index=0, methods={"Other": result(1)}),
        ]
        report = analyze_records(source)
        details = report["groups"][0]["methods"]["M"]
        self.assertEqual(details["n_instances_available_in_group"], 3)
        self.assertEqual(details["n_missing_method_results"], 2)
        self.assertEqual(details["n_seeds"], 2)
        self.assertEqual(details["n_seeds_with_attempts"], 1)
        self.assertEqual(details["n_seeds_missing_method_results"], 1)
        self.assertEqual(details["pooled"]["n_attempted"], 1)
        self.assertEqual(details["pooled"]["rec_exact_attempted_ratio"], 1)
        present = details["per_seed"]["42"]
        self.assertEqual(present["n_instances_available_in_group"], 2)
        self.assertEqual(present["n_missing_method_results"], 1)
        self.assertEqual(present["n_attempted"], 1)
        missing = details["per_seed"]["123"]
        self.assertEqual(missing["n_instances_available_in_group"], 1)
        self.assertEqual(missing["n_missing_method_results"], 1)
        self.assertEqual(missing["n_attempted"], 0)
        self.assertIsNone(missing["rec_exact_attempted_ratio"])
        self.assertIsNone(missing["rec_exact_valid_ratio"])
        self.assertIsNone(missing["tokens_per_rec_exact_recovery"])
        stats = details["cross_seed"]["rec_exact_attempted"]
        self.assertEqual(stats["n_seeds_total"], 2)
        self.assertEqual(stats["n_seeds_available"], 1)
        self.assertEqual(stats["n_seeds_unavailable"], 1)
        self.assertEqual(stats["mean_ratio"], 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.csv"
            write_csv(report, path)
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
        csv_missing = next(row for row in rows if row["method"] == "M"
                           and row["scope"] == "seed" and row["seed"] == "123"
                           and row["metric"] == "rec_exact" and row["denominator"] == "attempted")
        self.assertEqual(csv_missing["n_instances_available_in_group"], "1")
        self.assertEqual(csv_missing["n_missing_method_results"], "1")
        self.assertEqual(csv_missing["n_attempted"], "0")
        self.assertEqual(csv_missing["recall_ratio"], "")

    def test_missing_protocol_is_rejected(self):
        row = record()
        del row["run_metadata"]["context_protocol"]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "context_protocol"):
                self.load_payloads(directory, [("jsonl", [row])])

    def test_summary_only_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Aggregate-only"):
                self.load_payloads(directory, [("json", {"summaries": {}})])

    def test_invalid_success_is_rejected(self):
        row = record(methods={"M": result(1, 10, False)})
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "invalid response"):
                self.load_payloads(directory, [("jsonl", [row])])

    def test_conflicting_top_level_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "model"):
                self.load_payloads(directory, [("json", {"model": "different", "per_instance": [record()]})])

    def test_token_aliases_are_normalized(self):
        first = record()
        second = deepcopy(first)
        method = second["results"]["WM-SAR-LLM"]
        method["token_cost"] = method.pop("tokens")
        with tempfile.TemporaryDirectory() as directory:
            rows, provenance = self.load_payloads(directory, [("jsonl", [first, second])])
        self.assertEqual(len(rows), 1)
        self.assertEqual(provenance["n_exact_duplicates_removed"], 1)

    def test_duplicate_json_keys_and_nonfinite_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            for text in ('{"per_instance": [], "per_instance": []}', '{"value": NaN}'):
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_records([path])

    def test_csv_contains_model_seed_and_all_metric_scopes(self):
        report = analyze_records([record()])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "analysis.csv"
            write_csv(report, path)
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual({row["scope"] for row in rows}, {"pooled", "seed", "cross_seed"})
        self.assertEqual({row["metric"] for row in rows}, {"rec_exact", "rec_type", "rec_hop2"})
        self.assertEqual({row["model"] for row in rows}, {"model-a"})
        self.assertEqual({row["method"] for row in rows}, {"WM-SAR-LLM"})


if __name__ == "__main__":
    unittest.main()
