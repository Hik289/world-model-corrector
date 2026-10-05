import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, mock_open, patch

import networkx as nx

from experiments.exp_agent_llm import LLM_METHODS, _call_llm_on_region, aggregate, load_resume_records
from experiments.exp_multiapi import aggregate_model, call_llm_region
from wm_sar.act_text import build_locate_prompt, parse_locate_response, tree_to_text
from wm_sar.llm_client import ChatResult, LLMClient


def _graph():
    graph = nx.DiGraph()
    for i in range(4):
        graph.add_node(
            f"n{i}", node_type="executor", time_step=i, err=0.1,
            state=[1.0, 0.2, 0.1, 0.2, 0.9, 0.8, 1.0, 1.0],
        )
    graph.add_edges_from((f"n{i}", f"n{i + 1}", {"edge_type": "calls"})
                         for i in range(3))
    graph.graph["t_star"] = "n3"
    return graph


def _response(nodes=None, **updates):
    data = {
        "root_cause_nodes": ["n1"] if nodes is None else nodes,
        "root_cause_type": "executor",
        "explanation": "The selected executor introduced the error.",
        "confidence": 0.8,
    }
    data.update(updates)
    return json.dumps(data)


def _row(valid, tokens, recovered=False, api_error=""):
    return {
        "valid_response": valid,
        "tokens": tokens,
        "token_cost": tokens,
        "rec_exact": recovered,
        "rec_type": recovered,
        "rec_hop2": recovered,
        "region_size": 2,
        "latency_ms": 1.0,
        "api_error": api_error,
    }


class PaperContextTests(unittest.TestCase):
    def test_selected_context_contains_state_and_boundary_dependencies(self):
        text, nodes = tree_to_text(_graph(), {"n1", "n2"})
        self.assertEqual(nodes, ["n1", "n2"])
        self.assertEqual(text.count("state=["), 2)
        self.assertEqual(text.count("observed_error=0.1"), 2)
        self.assertEqual(text.count("confidence=0.8"), 2)
        self.assertIn("n0 [external] --[calls]--> n1", text)
        self.assertIn("n1 --[calls]--> n2", text)
        self.assertIn("n2 --[calls]--> n3 [external]", text)
        self.assertNotIn("[EXECUTOR] 'n0'", text)
        self.assertNotIn("[EXECUTOR] 'n3'", text)

    def test_truncation_limits_internal_edges_and_allowed_ids(self):
        graph = _graph()
        text, nodes = tree_to_text(graph, {"n1", "n2", "n3"}, max_nodes=1)
        self.assertEqual(nodes, ["n1"])
        self.assertIn("n1 --[calls]--> n2 [external]", text)
        self.assertNotIn("n2 --[calls]--> n3", text)
        self.assertNotIn("[EXECUTOR] 'n2'", text)
        _, prompt = build_locate_prompt(text, nodes, graph)
        self.assertIn('Eligible root-cause node IDs: ["n1"]', prompt)

    def test_full_graph_and_trace_serialization(self):
        graph = _graph()
        text, nodes = tree_to_text(graph)
        self.assertEqual(nodes, list(graph))
        self.assertIn("n0 --[calls]--> n1", text)
        self.assertNotIn("[external]", text)
        trace, _ = tree_to_text(graph, {"n1", "n2"}, include_edges=False)
        self.assertNotIn("--[calls]-->", trace)
        self.assertNotIn("Boundary dependencies", trace)

    def test_default_context_is_not_silently_truncated(self):
        graph = nx.DiGraph()
        graph.add_nodes_from((f"n{i}", {"time_step": i}) for i in range(40))
        _, nodes = tree_to_text(graph)
        self.assertEqual(len(nodes), 40)

    def test_invalid_selection_and_limit_are_rejected(self):
        with self.assertRaises(ValueError):
            tree_to_text(_graph(), {"missing"})
        with self.assertRaises(ValueError):
            tree_to_text(_graph(), max_nodes=-1)

    def test_json_schema_and_selected_ids_are_validated(self):
        parsed = parse_locate_response(_response(), "n1", _graph(), {"n1", "n2"})
        self.assertTrue(parsed["valid_json"])
        self.assertTrue(parsed["valid_response"])
        self.assertTrue(parsed["recovered_exact"])
        self.assertTrue(parsed["recovered_type"])
        self.assertTrue(parsed["recovered_hop2"])
        self.assertEqual(parsed["identified_nodes"], ["n1"])

    def test_quoted_ids_in_invalid_output_do_not_count_as_recovery(self):
        parsed = parse_locate_response('Root cause is "n1".', "n1", _graph())
        self.assertFalse(parsed["valid_json"])
        self.assertFalse(parsed["valid_response"])
        self.assertFalse(parsed["recovered_exact"])
        self.assertEqual(parsed["identified_nodes"], [])

    def test_boundary_or_unknown_nodes_are_not_eligible_predictions(self):
        for nodes in (["n0"], ["missing"], ["n1", "n0"]):
            with self.subTest(nodes=nodes):
                parsed = parse_locate_response(_response(nodes), "n1", _graph(), {"n1"})
                self.assertTrue(parsed["valid_json"])
                self.assertFalse(parsed["valid_response"])
                self.assertFalse(parsed["recovered_exact"])
                self.assertEqual(parsed["invalid_reason"], "node_outside_selected_region")

    def test_malformed_schema_is_not_credited(self):
        payloads = [
            "[]", _response(nodes="n1"), _response(confidence=True),
            _response(confidence=float("nan")), _response(confidence=1.5),
            _response(explanation=None), _response(root_cause_type=None),
        ]
        for payload in payloads:
            with self.subTest(payload=payload):
                parsed = parse_locate_response(payload, "n1", _graph())
                self.assertFalse(parsed["valid_response"])
                self.assertFalse(parsed["recovered_exact"])

    def test_json_fence_and_empty_prediction_are_handled(self):
        parsed = parse_locate_response(f"```json\n{_response()}\n```", "n1", _graph())
        self.assertTrue(parsed["valid_response"])
        parsed = parse_locate_response(_response([]), "n1", _graph())
        self.assertTrue(parsed["valid_response"])
        self.assertFalse(parsed["recovered_exact"])

    def test_nonstandard_numbers_and_duplicate_json_keys_are_rejected(self):
        payloads = [
            _response(confidence=float("nan")),
            _response().replace('"confidence": 0.8', '"confidence": 0.1, "confidence": 0.8'),
        ]
        for payload in payloads:
            parsed = parse_locate_response(payload, "n1", _graph())
            self.assertFalse(parsed["valid_json"])
            self.assertFalse(parsed["valid_response"])

    def test_cross_model_valid_and_attempted_denominators_are_separate(self):
        rows = [_row(True, 100, True), _row(False, 200),
                _row(False, None, api_error="RuntimeError")]
        result = aggregate_model({"ReCore": rows})["ReCore"]
        self.assertEqual(result["n"], 1)
        self.assertEqual(result["n_valid"], 1)
        self.assertEqual(result["n_attempted"], 3)
        self.assertEqual(result["n_invalid"], 2)
        self.assertEqual(result["n_api_errors"], 1)
        self.assertEqual(result["rec_exact"], 1.0)
        self.assertAlmostEqual(result["rec_exact_attempted"], 1 / 3)
        self.assertEqual(result["mean_tokens"], 150)

    def test_no_valid_responses_produces_no_valid_recall_estimate(self):
        result = aggregate_model({"ReCore": [_row(False, None)]})["ReCore"]
        self.assertIsNone(result["rec_exact"])
        self.assertIsNone(result["mean_tokens"])
        self.assertEqual(result["rec_exact_attempted"], 0.0)

    def test_single_model_retains_attempted_denominator_and_legacy_status(self):
        rows = [_row(True, 100, True), _row(False, 200), _row(None, 300)]
        result = aggregate([{"ReCore": row} for row in rows])["ReCore"]
        self.assertEqual(result["n_attempted"], 3)
        self.assertEqual(result["n_valid"], 1)
        self.assertEqual(result["n_unverified"], 1)
        self.assertAlmostEqual(result["rec_exact"], 1 / 3)
        self.assertEqual(result["rec_exact_valid"], 1.0)

    def test_api_failure_remains_an_attempt_without_invented_tokens(self):
        client = Mock()
        client.chat.side_effect = RuntimeError("unavailable")
        for caller, token_key in ((_call_llm_on_region, "token_cost"),
                                  (call_llm_region, "tokens")):
            row = caller(_graph(), {"n1"}, "n1", client, "ReCore")
            self.assertFalse(row["valid_response"])
            self.assertFalse(row["rec_exact"])
            self.assertIsNone(row[token_key])
            self.assertEqual(row["invalid_reason"], "api_error")

    def test_callers_validate_prediction_against_serialized_region(self):
        client = Mock()
        client.chat.return_value = SimpleNamespace(
            text=_response(["n0"]), prompt_tokens=100, completion_tokens=20,
            total_tokens=120,
        )
        for caller in (_call_llm_on_region, call_llm_region):
            row = caller(_graph(), {"n1"}, "n0", client, "ReCore")
            self.assertFalse(row["valid_response"])
            self.assertFalse(row["rec_exact"])

    def test_missing_usage_does_not_invalidate_structured_response(self):
        client = Mock()
        client.chat.return_value = ChatResult(_response(), None, 20, 1.0)
        self.assertIsNone(client.chat.return_value.total_tokens)
        for caller, key in ((_call_llm_on_region, "token_cost"), (call_llm_region, "tokens")):
            row = caller(_graph(), {"n1"}, "n1", client, "ReCore")
            self.assertTrue(row["valid_response"])
            self.assertTrue(row["rec_exact"])
            self.assertIsNone(row[key])
            self.assertEqual(row["api_error"], "")

    def test_legacy_helpers_keep_numeric_tokens_and_mark_estimates(self):
        client = object.__new__(LLMClient)
        client._call = lambda *_: ('{"root_cause_steps": [1], "confidence": 0.5}', None, None, 1.0)
        result = client.locate_error([{"step": 1}])
        self.assertIsInstance(result.total_tokens, int)
        self.assertTrue(result.usage_estimated)
        chat = client.chat("system", "user")
        self.assertIsNone(chat.prompt_tokens)
        self.assertIsNone(chat.completion_tokens)
        self.assertIsNone(chat.total_tokens)

    def test_resume_loader_deduplicates_and_checks_protocol(self):
        metadata = {"context_protocol": "recore-context-v2", "model": "model-a"}
        result = dict(_row(True, 100, True), valid_json=True)
        record = {
            "instance_id": "s42_i000", "seed": 42, "true_root": "n1",
            "n_nodes": 4, "run_metadata": metadata,
            "results": {method: dict(result) for method in LLM_METHODS},
        }
        payload = json.dumps(record) + "\n"
        with patch("builtins.open", mock_open(read_data=payload + payload)):
            loaded = load_resume_records("unused.jsonl", metadata, 42, 1)
        self.assertEqual(loaded, [record])
        with patch("builtins.open", mock_open(read_data=payload)):
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                load_resume_records("unused.jsonl", dict(metadata, model="model-b"), 42, 1)
        legacy = dict(record)
        legacy.pop("run_metadata")
        with patch("builtins.open", mock_open(read_data=json.dumps(legacy))):
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                load_resume_records("unused.jsonl", metadata, 42, 1)

    def test_resume_loader_rejects_corrupt_or_conflicting_records(self):
        metadata = {"context_protocol": "recore-context-v2", "model": "model-a"}
        result = dict(_row(True, 100, True), valid_json=True)
        record = {
            "instance_id": "s42_i000", "seed": 42, "true_root": "n1",
            "n_nodes": 4, "run_metadata": metadata,
            "results": {method: dict(result) for method in LLM_METHODS},
        }
        payload = json.dumps(record) + "\n"
        with patch("builtins.open", mock_open(read_data=payload + '{"instance_id":')):
            with self.assertRaisesRegex(ValueError, "Invalid resume JSON"):
                load_resume_records("unused.jsonl", metadata, 42, 1)
        changed = dict(record, n_nodes=5)
        with patch("builtins.open", mock_open(read_data=payload + json.dumps(changed))):
            with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
                load_resume_records("unused.jsonl", metadata, 42, 1)


if __name__ == "__main__":
    unittest.main()
