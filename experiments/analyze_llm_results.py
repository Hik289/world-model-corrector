import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = ("rec_exact", "rec_type", "rec_hop2")


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON value: {value}")


def _decode(text):
    return json.loads(text, object_pairs_hook=_object, parse_constant=_reject_constant)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _metadata(row, envelope):
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
    _canonical(metadata)
    return metadata


def _normalize_result(method, result):
    if not isinstance(method, str) or not method or not isinstance(result, dict):
        raise ValueError("Each method requires a nonempty name and a result object")
    normalized = dict(result)
    if "method" in normalized and normalized["method"] != method:
        raise ValueError(f"Conflicting method name for {method}")
    normalized.pop("method", None)
    for metric in METRICS:
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
    if normalized["valid_response"] is False and any(normalized[k] for k in METRICS):
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
    if api_error and (normalized["valid_response"] is True or any(normalized[k] for k in METRICS)):
        raise ValueError(f"{method} reports a valid response for an API error")
    normalized["api_error"] = api_error
    _canonical(normalized)
    return normalized


def _normalize_record(row, envelope):
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
    normalized["run_metadata"] = _metadata(row, envelope)
    normalized["results"] = {
        method: _normalize_result(method, result) for method, result in results.items()
    }
    _canonical(normalized)
    return normalized


def _document_records(document):
    if isinstance(document, list):
        return [_normalize_record(row, {}) for row in document]
    if not isinstance(document, dict):
        raise ValueError("Input must contain per-instance JSON records")
    if "per_instance" in document:
        if not isinstance(document["per_instance"], list):
            raise ValueError("per_instance must be an array")
        return [_normalize_record(row, document) for row in document["per_instance"]]
    if "instance_id" in document:
        return [_normalize_record(document, {})]
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
                            documents.append(_decode(line))
                        except ValueError as exc:
                            raise ValueError(f"line {number}: {exc}") from exc
            else:
                documents.append(_decode(text))
            for document in documents:
                for record in _document_records(document):
                    key = (_canonical(record["run_metadata"]), record["seed"], record["instance_id"])
                    if key in records:
                        if _canonical(records[key]) != _canonical(record):
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
    for metric in METRICS:
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


def _seed_statistics(values):
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
        protocol_key = _canonical(record["run_metadata"])
        group_instance_counts[protocol_key][record["seed"]] += 1
        for method, result in record["results"].items():
            grouped[protocol_key][method][record["seed"]].append(result)
    groups = []
    for protocol_key, methods in sorted(grouped.items()):
        metadata = _decode(protocol_key)
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
            for metric in METRICS:
                for denominator in ("attempted", "valid"):
                    key = f"{metric}_{denominator}_ratio"
                    cross_seed[f"{metric}_{denominator}"] = _seed_statistics(
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
                    "model": group["model"], "run_metadata": _canonical(group["run_metadata"]),
                    "method": method, "display_name": details["display_name"],
                    "n_instances_available_in_group": details["n_instances_available_in_group"],
                    "n_missing_method_results": details["n_missing_method_results"],
                }
                for seed, summary in [(None, details["pooled"]), *details["per_seed"].items()]:
                    counts = {key: value for key, value in summary.items() if key in columns}
                    for metric in METRICS:
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
                for metric in METRICS:
                    for denominator in ("attempted", "valid"):
                        writer.writerow({
                            **identity, "scope": "cross_seed", "metric": metric,
                            "denominator": denominator,
                            **details["cross_seed"][f"{metric}_{denominator}"],
                        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv")
    args = parser.parse_args()
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


if __name__ == "__main__":
    main()
