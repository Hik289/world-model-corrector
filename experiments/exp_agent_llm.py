import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wm_sar.agent_calling_tree import generate_calling_trees
from wm_sar.engineering_baselines import (
    greedy_point, topk_point, window_repair, local_khop
)
from wm_sar.region_extractor import WMSAR, WMSARConfig
from wm_sar.llm_client import LLMClient
from wm_sar.act_text import (
    CONTEXT_PROTOCOL_VERSION, tree_to_text, build_locate_prompt, parse_locate_response,
)

import networkx as nx

LLM_METHODS = (
    "Greedy-Point-LLM", "TopK-5-LLM", "Window-4-LLM", "Window-8-LLM",
    "LocalRepair-2Hop-LLM", "Full-Graph-LLM", "WM-SAR-LLM",
    "TraceScan-w1-LLM", "TraceScan-w2-LLM", "TraceScan-Full-LLM",
    "LLMRepair-Full-Plan-LLM",
)


def load_resume_records(path, run_metadata, seed, n):
    records = {}
    expected_ids = {f"s{seed}_i{i:03d}" for i in range(n)}
    required = {
        "rec_exact", "rec_type", "rec_hop2", "tokens", "region_size",
        "latency_ms", "valid_json", "valid_response",
    }
    with open(path) as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid resume JSON at line {line_number}; use a new output path") from exc
            if not isinstance(record, dict) or record.get("run_metadata") != run_metadata:
                raise ValueError(f"Incompatible resume protocol or model at line {line_number}; use a new output path")
            instance_id = record.get("instance_id")
            results = record.get("results")
            if (not isinstance(instance_id, str) or instance_id not in expected_ids
                    or record.get("seed") != seed
                    or not isinstance(record.get("true_root"), str)
                    or not isinstance(record.get("n_nodes"), int)
                    or not isinstance(results, dict)
                    or set(results) != set(LLM_METHODS)):
                raise ValueError(f"Invalid resume instance at line {line_number}")
            for result in results.values():
                if (not isinstance(result, dict) or not required.issubset(result)
                        or not isinstance(result["valid_json"], bool)
                        or not isinstance(result["valid_response"], bool)
                        or any(result[k] not in (0, 1) for k in ("rec_exact", "rec_type", "rec_hop2"))):
                    raise ValueError(f"Invalid resume result at line {line_number}")
                if result["valid_response"] and not result["valid_json"]:
                    raise ValueError(f"Invalid resume validity flags at line {line_number}")
                region_size, latency = result["region_size"], result["latency_ms"]
                if (isinstance(region_size, bool) or not isinstance(region_size, int)
                        or region_size < 0 or isinstance(latency, bool)
                        or not isinstance(latency, (int, float))
                        or not math.isfinite(latency) or latency < 0):
                    raise ValueError(f"Invalid resume measurement at line {line_number}")
                if (not result["valid_response"]
                        and any(result[k] for k in ("rec_exact", "rec_type", "rec_hop2"))):
                    raise ValueError(f"Invalid resume recovery at line {line_number}")
                tokens = result["tokens"]
                if tokens is not None and (isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0):
                    raise ValueError(f"Invalid resume token count at line {line_number}")
            if instance_id in records and records[instance_id] != record:
                raise ValueError(f"Conflicting duplicate resume instance {instance_id}")
            records[instance_id] = record
    return list(records.values())


def _topo_order(G: nx.DiGraph) -> list[str]:
    try:
        return list(nx.topological_sort(G))
    except Exception:
        return sorted(G.nodes(),
                      key=lambda v: G.nodes[v].get("time_step", 0))


def _tracescan_window(G: nx.DiGraph, w: int) -> set[str]:

    topo = _topo_order(G)
    errs = [(float(G.nodes[v].get("err", 0.0)), i, v) for i, v in enumerate(topo)]
    if not errs:
        return set()
    _, ctr_i, _ = max(errs)
    lo = max(0, ctr_i - w // 2)
    hi = min(len(topo), lo + w)
    lo = max(0, hi - w)
    return set(topo[lo:hi])


def _build_full_plan_prompt(tree_text: str, node_list: list[str]) -> tuple[str, str]:


    system = (
        "You are an expert AI agent failure analyst and repair planner. "
        "You will receive a complete report of a failed multi-agent calling-tree. "
        "Your job has TWO parts: (1) identify the root-cause node that "
        "introduced the initial error; (2) propose a corrective action plan "
        "for every affected node along the cascade. Respond ONLY in valid JSON."
    )
    user = (
        f"{tree_text}\n\n"
        "Step 1 — identify the SINGLE root-cause node that started the cascade.\n"
        "  Use: high error_prob + low success_flag = direct error; "
        "dependency UNSATISFIED = cascade victim, not cause; "
        "low throughput at executors is a strong signal.\n\n"
        "Step 2 — list, in topological order, every affected downstream node and "
        "a one-sentence corrective action for each.\n\n"
        f"Eligible node IDs: {json.dumps(node_list)}.\n\n"
        "Respond ONLY with valid JSON of the form:\n"
        '{"root_cause_nodes": ["<node_id>"], '
        '"root_cause_type": "<node_type>", '
        '"repair_plan": {"<node_id>": "<action>", ...}, '
        '"explanation": "<one sentence>", '
        '"confidence": <0-1>}'
    )
    return system, user


def _call_llm_on_region(
    G: nx.DiGraph, region: set, true_root: str,
    client: LLMClient, method_name: str,
    prompt_builder=build_locate_prompt,
    include_edges: bool = True,
) -> dict:


    text, node_list = tree_to_text(G, selected_nodes=region,
                                    include_edges=include_edges,
                                    max_nodes=max(len(region), 30))
    if prompt_builder is _build_full_plan_prompt:
        system, user = prompt_builder(text, node_list)
    else:
        system, user = prompt_builder(text, node_list, G)
    t0 = time.time()
    api_error = ""
    token_cost = None
    try:
        resp = client.chat(system=system, user=user)
    except Exception as exc:
        parsed = parse_locate_response("", true_root, G,
                                       allowed_nodes=set(node_list))
        parsed["invalid_reason"] = "api_error"
        api_error = type(exc).__name__
    else:
        pt, ct = resp.prompt_tokens, resp.completion_tokens
        if all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in (pt, ct)):
            token_cost = pt + ct
        parsed = parse_locate_response(resp.text, true_root, G,
                                       allowed_nodes=set(node_list))
    latency_ms = (time.time() - t0) * 1000.0
    return {
        "method": method_name,
        "region_size": len(node_list),
        "token_cost": token_cost,
        "latency_ms": latency_ms,
        "rec_exact": parsed["recovered_exact"],
        "rec_type": parsed["recovered_type"],
        "rec_hop2": parsed["recovered_hop2"],
        "identified_nodes": parsed["identified_nodes"],
        "confidence": parsed["confidence"],
        "valid_json": parsed["valid_json"],
        "valid_response": parsed["valid_response"],
        "invalid_reason": parsed["invalid_reason"],
        "explanation": parsed["explanation"],
        "api_error": api_error,
    }


def run_instance(G: nx.DiGraph, true_root: str, client: LLMClient) -> dict:

    results = {}


    rr = greedy_point(G, K=1)
    results["Greedy-Point-LLM"] = _call_llm_on_region(
        G, rr.selected_nodes, true_root, client, "Greedy-Point-LLM")


    rr = topk_point(G, K=5)
    results["TopK-5-LLM"] = _call_llm_on_region(
        G, rr.selected_nodes, true_root, client, "TopK-5-LLM")


    rr = window_repair(G, window=4)
    results["Window-4-LLM"] = _call_llm_on_region(
        G, rr.selected_nodes, true_root, client, "Window-4-LLM")


    rr = window_repair(G, window=8)
    results["Window-8-LLM"] = _call_llm_on_region(
        G, rr.selected_nodes, true_root, client, "Window-8-LLM")


    rr = local_khop(G, k=2)
    results["LocalRepair-2Hop-LLM"] = _call_llm_on_region(
        G, rr.selected_nodes, true_root, client, "LocalRepair-2Hop-LLM")


    full_region = set(G.nodes())
    results["Full-Graph-LLM"] = _call_llm_on_region(
        G, full_region, true_root, client, "Full-Graph-LLM")


    extractor = WMSAR(WMSARConfig())
    region = extractor.repair_region(G)
    results["WM-SAR-LLM"] = _call_llm_on_region(
        G, region, true_root, client, "WM-SAR-LLM")


    for w in (1, 2):
        results[f"TraceScan-w{w}-LLM"] = _call_llm_on_region(
            G, _tracescan_window(G, w), true_root, client,
            f"TraceScan-w{w}-LLM",
            include_edges=False)

    results["TraceScan-Full-LLM"] = _call_llm_on_region(
        G, set(G.nodes()), true_root, client,
        "TraceScan-Full-LLM",
        include_edges=False)

    results["LLMRepair-Full-Plan-LLM"] = _call_llm_on_region(
        G, set(G.nodes()), true_root, client,
        "LLMRepair-Full-Plan-LLM",
        prompt_builder=_build_full_plan_prompt,
        include_edges=True)

    return results


def aggregate(all_results: list[dict]) -> dict[str, dict]:


    methods = sorted({m for r in all_results for m in r.keys()})
    summaries = {}
    for m in methods:
        rows = [r[m] for r in all_results if m in r]
        n = len(rows)
        if n == 0:
            continue
        valid_rows = [r for r in rows if r.get("valid_response") is True]
        token_values = [r["token_cost"] for r in rows if r.get("token_cost") is not None]
        summaries[m] = {
            "n": n,
            "n_attempted": n,
            "n_valid": len(valid_rows),
            "n_invalid": sum(r.get("valid_response") is False for r in rows),
            "n_unverified": sum(r.get("valid_response") is None for r in rows),
            "n_api_errors": sum(bool(r.get("api_error")) for r in rows),
            "n_token_measured": len(token_values),
            "recall_denominator": "attempted",
            "rec_exact":  float(np.mean([r["rec_exact"] for r in rows])),
            "rec_type":   float(np.mean([r["rec_type"] for r in rows])),
            "rec_hop2":   float(np.mean([r["rec_hop2"] for r in rows])),
            "mean_tokens": float(np.mean(token_values)) if token_values else None,
            "mean_region_size": float(np.mean([r["region_size"] for r in rows])),
            "mean_latency_ms": float(np.mean([r["latency_ms"] for r in rows])),
        }
        for metric in ("rec_exact", "rec_type", "rec_hop2"):
            summaries[m][f"{metric}_valid"] = (
                float(np.mean([r[metric] for r in valid_rows])) if valid_rows else None
            )
        summaries[m]["tok_per_rec_hop2"] = (
            sum(token_values) / sum(r["rec_hop2"] for r in rows)
            if len(token_values) == n and any(r["rec_hop2"] for r in rows) else None
        )
    return summaries


def build_per_instance(per_instance_rows: list[dict]) -> list[dict]:


    out = []
    for row in per_instance_rows:
        rec = {
            "instance_id":   row["instance_id"],
            "seed":          row["seed"],
            "true_root":     row["true_root"],
            "true_root_type": row["true_root_type"],
            "n_nodes":       row["n_nodes"],
            "results": {},
            "run_metadata": row.get("run_metadata"),
        }
        for method, r in row["results"].items():
            rec["results"][method] = {
                "rec_exact":    int(bool(r["rec_exact"])),
                "rec_type":     int(bool(r["rec_type"])),
                "rec_hop2":     int(bool(r["rec_hop2"])),
                "tokens":       int(r["token_cost"]) if r.get("token_cost") is not None else None,
                "region_size":  int(r["region_size"]),
                "latency_ms":   float(r["latency_ms"]),
                "identified_nodes": list(r.get("identified_nodes", [])),
                "confidence":   float(r.get("confidence", 0.0)),
                "valid_json": r.get("valid_json"),
                "valid_response": r.get("valid_response"),
                "invalid_reason": r.get("invalid_reason", ""),
                "explanation": r.get("explanation", ""),
                "api_error": r.get("api_error", ""),
            }
        out.append(rec)
    return out


_ANALYSIS_METRICS = ("rec_exact", "rec_type", "rec_hop2")


def _analysis_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _analysis_reject_constant(value):
    raise ValueError(f"Non-finite JSON value: {value}")


def _analysis_decode(text):
    return json.loads(text, object_pairs_hook=_analysis_object, parse_constant=_analysis_reject_constant)


def _analysis_canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _analysis_metadata(row, envelope):
    metadata = {}
    for source in (envelope, row):
        supplied = source.get("run_metadata")
        if supplied is not None and not isinstance(supplied, dict):
            raise ValueError("run_metadata must be an object")
        for key, value in (supplied or {}).items():
            if key in metadata and metadata[key] != value:
                raise ValueError(f"Conflicting run_metadata field: {key}")
            metadata[key] = value
        for key in ("context_protocol", "model"):
            value = source.get(key)
            if value is not None:
                if key in metadata and metadata[key] != value:
                    raise ValueError(f"Conflicting {key} metadata")
                metadata[key] = value
    for key in ("context_protocol", "model"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f"Each instance requires explicit {key} metadata")
    _analysis_canonical(metadata)
    return metadata


def _analysis_normalize_result(method, result):
    if not isinstance(method, str) or not method or not isinstance(result, dict):
        raise ValueError("Each method requires a nonempty name and a result object")
    normalized = dict(result)
    if "method" in normalized and normalized["method"] != method:
        raise ValueError(f"Conflicting method name for {method}")
    normalized.pop("method", None)
    for metric in _ANALYSIS_METRICS:
        value = normalized.get(metric)
        if not isinstance(value, (int, float)) or value not in (0, 1):
            raise ValueError(f"{method}.{metric} must be a binary observation")
        normalized[metric] = int(value)
    for flag in ("valid_response", "valid_json"):
        value = normalized.get(flag)
        if value is not None and not isinstance(value, bool):
            raise ValueError(f"{method}.{flag} must be boolean or null")
        normalized[flag] = value
    if normalized["valid_response"] is True and normalized["valid_json"] is not True:
        raise ValueError(f"{method} has inconsistent response validity")
    if normalized["valid_response"] is False and any(normalized[k] for k in _ANALYSIS_METRICS):
        raise ValueError(f"{method} reports recovery for an invalid response")
    if "tokens" in normalized and "token_cost" in normalized:
        if normalized["tokens"] != normalized["token_cost"]:
            raise ValueError(f"{method} has conflicting token counts")
    tokens = normalized.pop("token_cost", normalized.get("tokens"))
    if tokens is not None and (isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0):
        raise ValueError(f"{method}.tokens must be a nonnegative integer or null")
    normalized["tokens"] = tokens
    api_error = normalized.get("api_error", "")
    if api_error is None:
        api_error = ""
    if not isinstance(api_error, str):
        raise ValueError(f"{method}.api_error must be a string or null")
    if api_error and (normalized["valid_response"] is True or any(normalized[k] for k in _ANALYSIS_METRICS)):
        raise ValueError(f"{method} reports a valid response for an API error")
    normalized["api_error"] = api_error
    _analysis_canonical(normalized)
    return normalized


def _analysis_normalize_record(row, envelope):
    if not isinstance(row, dict):
        raise ValueError("Each per_instance entry must be an object")
    instance_id = row.get("instance_id")
    if not isinstance(instance_id, str) or not instance_id:
        raise ValueError("Each instance requires a nonempty instance_id")
    seed = row.get("seed", envelope.get("seed"))
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"{instance_id} requires an integer seed")
    if "seed" not in row and len(envelope.get("seeds", [])) > 1:
        raise ValueError(f"{instance_id} has an ambiguous seed")
    results = row.get("results")
    if not isinstance(results, dict) or not results:
        raise ValueError(f"{instance_id} requires per-method results")
    normalized = {
        key: value for key, value in row.items()
        if key not in ("run_metadata", "model", "context_protocol", "results", "seed")
    }
    normalized["seed"] = seed
    normalized["run_metadata"] = _analysis_metadata(row, envelope)
    normalized["results"] = {
        method: _analysis_normalize_result(method, result) for method, result in results.items()
    }
    _analysis_canonical(normalized)
    return normalized


def _analysis_document_records(document):
    if isinstance(document, list):
        return [_analysis_normalize_record(row, {}) for row in document]
    if not isinstance(document, dict):
        raise ValueError("Input must contain per-instance JSON records")
    if "per_instance" in document:
        if not isinstance(document["per_instance"], list):
            raise ValueError("per_instance must be an array")
        return [_analysis_normalize_record(row, document) for row in document["per_instance"]]
    if "instance_id" in document:
        return [_analysis_normalize_record(document, {})]
    raise ValueError("Aggregate-only input is insufficient; provide per_instance records")


def load_records(paths):
    records = {}
    duplicate_count = 0
    source_paths = []
    for value in paths:
        path = Path(value)
        source_paths.append(str(path.resolve()))
        text = path.read_text(encoding="utf-8")
        documents = []
        try:
            if path.suffix.lower() == ".jsonl":
                for number, line in enumerate(text.splitlines(), 1):
                    if line.strip():
                        try:
                            documents.append(_analysis_decode(line))
                        except ValueError as exc:
                            raise ValueError(f"line {number}: {exc}") from exc
            else:
                documents.append(_analysis_decode(text))
            for document in documents:
                for record in _analysis_document_records(document):
                    key = (_analysis_canonical(record["run_metadata"]), record["seed"], record["instance_id"])
                    if key in records:
                        if _analysis_canonical(records[key]) != _analysis_canonical(record):
                            raise ValueError(f"Conflicting duplicate instance: {record['instance_id']}")
                        duplicate_count += 1
                    else:
                        records[key] = record
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: {exc}") from exc
    if not records:
        raise ValueError("No per-instance records were provided")
    return list(records.values()), {
        "input_paths": source_paths,
        "n_unique_instances": len(records),
        "n_exact_duplicates_removed": duplicate_count,
    }


def summarize_rows(rows):
    n = len(rows)
    valid = [row for row in rows if row["valid_response"] is True]
    measured = [row["tokens"] for row in rows if row["tokens"] is not None]
    unknown = n - len(measured)
    summary = {
        "n_attempted": n,
        "n_valid": len(valid),
        "n_invalid": sum(row["valid_response"] is False for row in rows),
        "n_unverified": sum(row["valid_response"] is None for row in rows),
        "n_api_errors": sum(bool(row["api_error"]) for row in rows),
        "n_token_measured": len(measured),
        "n_token_unknown": unknown,
        "total_tokens_known": sum(measured),
        "total_tokens": sum(measured) if unknown == 0 else None,
        "mean_tokens_measured": statistics.mean(measured) if measured else None,
    }
    for metric in _ANALYSIS_METRICS:
        successes = sum(row[metric] for row in rows)
        valid_successes = sum(row[metric] for row in valid)
        summary[f"{metric}_successes_attempted"] = successes
        summary[f"{metric}_successes_valid"] = valid_successes
        summary[f"{metric}_attempted_ratio"] = successes / n if n else None
        summary[f"{metric}_valid_ratio"] = valid_successes / len(valid) if valid else None
        summary[f"tokens_per_{metric}_recovery"] = (
            sum(measured) / successes if unknown == 0 and successes > 0 else None
        )
    return summary


def _analysis_seed_statistics(values):
    available = [value for value in values if value is not None]
    mean = statistics.mean(available) if available else None
    std = statistics.pstdev(available) if available else None
    return {
        "n_seeds_total": len(values),
        "n_seeds_available": len(available),
        "n_seeds_unavailable": len(values) - len(available),
        "mean_ratio": mean,
        "population_std_ratio": std,
        "coefficient_of_variation_ratio": std / mean if mean is not None and mean != 0 else None,
    }


def analyze_records(records, provenance=None):
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    group_instance_counts = defaultdict(lambda: defaultdict(int))
    for record in records:
        protocol_key = _analysis_canonical(record["run_metadata"])
        group_instance_counts[protocol_key][record["seed"]] += 1
        for method, result in record["results"].items():
            grouped[protocol_key][method][record["seed"]].append(result)
    groups = []
    for protocol_key, methods in sorted(grouped.items()):
        metadata = _analysis_decode(protocol_key)
        group = {"model": metadata["model"], "run_metadata": metadata, "methods": {}}
        available_instances = group_instance_counts[protocol_key]
        n_available = sum(available_instances.values())
        for method, seeds in sorted(methods.items()):
            per_seed = {}
            for seed, n_instances in sorted(available_instances.items()):
                summary = summarize_rows(seeds.get(seed, []))
                summary["n_instances_available_in_group"] = n_instances
                summary["n_missing_method_results"] = n_instances - summary["n_attempted"]
                per_seed[str(seed)] = summary
            pooled_rows = [row for rows in seeds.values() for row in rows]
            pooled = summarize_rows(pooled_rows)
            pooled["n_instances_available_in_group"] = n_available
            pooled["n_missing_method_results"] = n_available - len(pooled_rows)
            cross_seed = {}
            for metric in _ANALYSIS_METRICS:
                for denominator in ("attempted", "valid"):
                    key = f"{metric}_{denominator}_ratio"
                    cross_seed[f"{metric}_{denominator}"] = _analysis_seed_statistics(
                        [summary[key] for summary in per_seed.values()]
                    )
            group["methods"][method] = {
                "display_name": method.replace("WM-SAR", "ReCore"),
                "n_seeds": len(per_seed),
                "n_seeds_with_attempts": len(seeds),
                "n_seeds_missing_method_results": len(per_seed) - len(seeds),
                "n_instances_available_in_group": n_available,
                "n_missing_method_results": n_available - len(pooled_rows),
                "pooled": pooled,
                "per_seed": per_seed,
                "cross_seed": cross_seed,
            }
        groups.append(group)
    if not groups:
        raise ValueError("No per-method observations were provided")
    return {
        "analysis_version": 1,
        "recall_units": "ratio",
        "cross_seed_weighting": "equal_weight_per_available_seed",
        "cross_seed_dispersion": "population_standard_deviation",
        "cv_units": "ratio",
        "token_efficiency": "total_attempt_tokens_divided_by_total_successes",
        "unknown_token_policy": "null_total_and_tokens_per_recovery",
        "provenance": provenance or {},
        "groups": groups,
    }


def write_csv(report, path):
    columns = [
        "model", "run_metadata", "method", "display_name", "scope", "seed", "metric",
        "n_instances_available_in_group", "n_missing_method_results",
        "denominator", "n_attempted", "n_valid", "n_invalid", "n_unverified", "n_api_errors",
        "n_token_measured", "n_token_unknown", "total_tokens_known", "total_tokens",
        "successes", "recall_ratio", "tokens_per_recovery", "n_seeds_total",
        "n_seeds_available", "n_seeds_unavailable", "mean_ratio", "population_std_ratio",
        "coefficient_of_variation_ratio",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for group in report["groups"]:
            for method, details in group["methods"].items():
                identity = {
                    "model": group["model"], "run_metadata": _analysis_canonical(group["run_metadata"]),
                    "method": method, "display_name": details["display_name"],
                    "n_instances_available_in_group": details["n_instances_available_in_group"],
                    "n_missing_method_results": details["n_missing_method_results"],
                }
                for seed, summary in [(None, details["pooled"]), *details["per_seed"].items()]:
                    counts = {key: value for key, value in summary.items() if key in columns}
                    for metric in _ANALYSIS_METRICS:
                        for denominator in ("attempted", "valid"):
                            writer.writerow({
                                **identity, **counts, "scope": "pooled" if seed is None else "seed",
                                "seed": seed, "metric": metric, "denominator": denominator,
                                "successes": summary[f"{metric}_successes_{denominator}"],
                                "recall_ratio": summary[f"{metric}_{denominator}_ratio"],
                                "tokens_per_recovery": (
                                    summary[f"tokens_per_{metric}_recovery"] if denominator == "attempted" else None
                                ),
                            })
                for metric in _ANALYSIS_METRICS:
                    for denominator in ("attempted", "valid"):
                        writer.writerow({
                            **identity, "scope": "cross_seed", "metric": metric,
                            "denominator": denominator,
                            **details["cross_seed"][f"{metric}_{denominator}"],
                        })


def analyze_result_files(args, parser):
    if not args.inputs:
        parser.error("--analyze-results requires --inputs")
    if not args.out:
        parser.error("--analyze-results requires --out")
    input_paths = {Path(path).resolve() for path in args.inputs}
    output_paths = [Path(args.out).resolve()]
    if args.csv:
        output_paths.append(Path(args.csv).resolve())
    if len(set(output_paths)) != len(output_paths) or input_paths.intersection(output_paths):
        parser.error("Output paths must be distinct from inputs and from each other")
    try:
        records, provenance = load_records(args.inputs)
        report = analyze_records(records, provenance)
        for path in output_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if args.csv:
            write_csv(report, args.csv)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42,
                        help="Single seed (legacy). Overridden by --seeds.")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated list of seeds, e.g. '42,123,456'. "
                             "If set, each seed is run separately and a merged "
                             "JSON is written to --out plus per-seed JSONs.")
    parser.add_argument("--model", type=str, default=os.environ.get("LLM_MODEL"))
    parser.add_argument("--resume", action="store_true",
                        help="Resume: skip instances already present in "
                             "<out>_seed<S>.jsonl. The JSONL is appended to "
                             "instead of truncated.")
    parser.add_argument("--out", type=str)
    parser.add_argument("--analyze-results", action="store_true")
    parser.add_argument("--inputs", nargs="+")
    parser.add_argument("--csv")
    args = parser.parse_args()
    if args.analyze_results:
        analyze_result_files(args, parser)
        return
    if args.inputs is not None or args.csv is not None:
        parser.error("--inputs and --csv require --analyze-results")
    if args.out is None:
        args.out = os.path.join(os.path.dirname(__file__), "results", "exp_agent_llm.json")
    if args.n < 1:
        parser.error("n must be at least 1")
    if not args.out.endswith(".json"):
        parser.error("out must end with .json")

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    if len(set(seeds)) != len(seeds):
        parser.error("seeds must be unique")

    print(f"\n{'='*60}")
    print("  Agent Calling-Tree LLM Experiment")
    model_label = args.model or os.environ.get("MODEL_NAME") or "configured default"
    print(f"  n={args.n} × seeds={seeds}, model={model_label}")
    print(f"{'='*60}\n")

    client = LLMClient(model=args.model, temperature=0.0, max_tokens=512)
    run_metadata = {
        "context_protocol": CONTEXT_PROTOCOL_VERSION,
        "model": client.model,
        "backend": client.backend,
        "temperature": client.temperature,
        "max_tokens": client.max_tokens,
    }

    all_per_instance = []
    all_results      = []
    per_seed_outputs = {}


    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    for seed in seeds:
        print(f"\n  ── seed={seed} ──")
        trees = generate_calling_trees(n=args.n, seed=seed)
        seed_per_instance = []
        seed_results = []
        jsonl_path = args.out.replace(".json", f"_seed{seed}.jsonl")


        done_ids: set[str] = set()
        resume_records = []
        if args.resume and os.path.exists(jsonl_path):
            resume_records = load_resume_records(jsonl_path, run_metadata, seed, args.n)
            done_ids = {rec["instance_id"] for rec in resume_records}
            if done_ids:
                print(f"    [resume] {len(done_ids)} instances already in "
                      f"{jsonl_path}; will skip those.")
        elif not args.resume:

            with open(jsonl_path, "w") as f:
                pass


        for rec in resume_records:
            pseudo_res = {}
            for meth, r in rec["results"].items():
                pseudo_res[meth] = dict(r, token_cost=r["tokens"])
            seed_results.append(pseudo_res)
            all_results.append(pseudo_res)
            row = {
                "instance_id": rec["instance_id"],
                "seed": rec["seed"],
                "true_root": rec["true_root"],
                "true_root_type": rec.get("true_root_type", ""),
                "n_nodes": rec["n_nodes"],
                "results": pseudo_res,
                "run_metadata": run_metadata,
            }
            seed_per_instance.append(row)
            all_per_instance.append(row)

        for i, tree in enumerate(trees):
            G = tree.G
            true_root = tree.root_cause_node
            instance_id = f"s{seed}_i{i:03d}"
            if instance_id in done_ids:
                continue
            print(f"  [s{seed} {i+1:2d}/{args.n}] root={true_root} "
                  f"({G.nodes[true_root].get('node_type')}), "
                  f"N={G.number_of_nodes()}", end="  ", flush=True)
            try:
                res = run_instance(G, true_root, client)
                seed_results.append(res)
                all_results.append(res)
                row = {
                    "instance_id":   instance_id,
                    "seed":          seed,
                    "true_root":     true_root,
                    "true_root_type": G.nodes[true_root].get("node_type", ""),
                    "n_nodes":       G.number_of_nodes(),
                    "results":       res,
                    "run_metadata": run_metadata,
                }
                seed_per_instance.append(row)
                all_per_instance.append(row)

                jsonl_row = build_per_instance([row])[0]
                with open(jsonl_path, "a") as f:
                    f.write(json.dumps(jsonl_row) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                wmsar_e = res.get("WM-SAR-LLM", {}).get("rec_exact", 0)
                ts_full_e = res.get("TraceScan-Full-LLM", {}).get("rec_exact", 0)
                lr_e = res.get("LLMRepair-Full-Plan-LLM", {}).get("rec_exact", 0)
                print(f"WM-SAR-E={wmsar_e:.0f}  TS-Full-E={ts_full_e:.0f}  "
                      f"LLMRep-E={lr_e:.0f}")
            except Exception as e:
                raise RuntimeError(f"Failed to complete or persist {instance_id}") from e


        if seed_results:
            seed_out = args.out.replace(".json", f"_seed{seed}.json")
            seed_payload = {
                "n": len(seed_results),
                "seed": seed,
                "model": client.model,
                "run_metadata": run_metadata,
                "summaries": aggregate(seed_results),
                "per_instance": build_per_instance(seed_per_instance),
            }
            with open(seed_out, "w") as f:
                json.dump(seed_payload, f, indent=2)
                f.flush(); os.fsync(f.fileno())
            print(f"    seed={seed} saved → {seed_out} (+ {jsonl_path})")
            per_seed_outputs[seed] = seed_out

    if not all_results:
        print("No results collected.")
        return

    summaries = aggregate(all_results)

    print(f"\n{'='*72}")
    print(f"  Merged summary (n={len(all_results)} = {args.n} × {len(seeds)} seeds)")
    print(f"{'='*72}")
    print(f"  {'Method':<28}  {'Rec-Hop2':>8}  {'Rec-Type':>8}  {'Rec-Exact':>9}  "
          f"{'Tokens':>7}  {'Size':>5}")
    print(f"  {'-'*78}")
    order = ["Greedy-Point-LLM", "TopK-5-LLM",
             "Window-4-LLM", "Window-8-LLM", "LocalRepair-2Hop-LLM",
             "Full-Graph-LLM",
             "TraceScan-w1-LLM", "TraceScan-w2-LLM", "TraceScan-Full-LLM",
             "LLMRepair-Full-Plan-LLM",
             "WM-SAR-LLM"]
    for m in order:
        s = summaries.get(m, {})
        if not s:
            continue
        mean_tokens = s.get("mean_tokens")
        tokens_text = f"{mean_tokens:>7.0f}" if mean_tokens is not None else f"{'n/a':>7}"
        print(f"  {m:<28}  {s.get('rec_hop2',0):>8.3f}  {s.get('rec_type',0):>8.3f}  "
              f"{s.get('rec_exact',0):>9.3f}  "
              f"{tokens_text}  {s.get('mean_region_size',0):>5.1f}")


    per_seed_wmsar = {}
    if len(seeds) > 1:
        print("\n  Per-seed Rec-Exact for WM-SAR-LLM:")
        for seed in seeds:
            vals = [p["results"]["WM-SAR-LLM"]["rec_exact"]
                    for p in all_per_instance if p["seed"] == seed
                    and "WM-SAR-LLM" in p["results"]]
            mean_v = float(np.mean(vals)) if vals else 0.0
            per_seed_wmsar[seed] = {"n": len(vals), "rec_exact_mean": mean_v}
            print(f"    seed={seed}  Rec-Exact (mean over {len(vals)}) = {mean_v:.3f}")
        per_seed_means = [v["rec_exact_mean"] for v in per_seed_wmsar.values()]
        cs_std = float(np.std(per_seed_means))
        print(f"    cross-seed std = {cs_std:.3f}  "
              f"({'HEALTHY' if cs_std <= 0.05 else 'FLAG: >0.05'})")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    output = {
        "n": len(all_results),
        "seeds": seeds,
        "seed": args.seed,
        "model": client.model,
        "run_metadata": run_metadata,
        "summaries": summaries,
        "per_seed_wmsar_rec_exact": per_seed_wmsar,
        "per_instance": build_per_instance(all_per_instance),
        "per_seed_files": per_seed_outputs,
    }
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Merged results saved to: {args.out}")


if __name__ == "__main__":
    main()
