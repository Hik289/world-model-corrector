from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import partial
import json
import math
from pathlib import Path
from statistics import mean, pstdev
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wm_sar import baselines as bl
from wm_sar import data_generator as dg
from wm_sar import failure_graph as fg
from wm_sar import metrics as me
from wm_sar import repair_executor as re
from wm_sar.region_extractor import ReCoreConfig


COLUMNS = ["Method", "Recovery", "CostNorm", "Tokens/Rec", "PDred", "DownErrRed", "Tokens"]


def _methods():
    return {
        "LastError-Point": bl.last_error_point,
        "FirstFailedCall-Point": bl.first_failed_call_point,
        "RuleScanner-Point": bl.rule_scanner_point,
        "TraceScan-w1-Point": lambda g: bl.trace_scan_window(g, 1),
        "TraceScan-w2-Point": lambda g: bl.trace_scan_window(g, 2),
        "TraceScan-w4-Point": lambda g: bl.trace_scan_window(g, 4),
        "TraceScan-Full-Point": lambda g: bl.trace_scan_full(g),
        "Top-B-Nodes": bl.top_b_nodes,
        "kHop-Subgraph": lambda g: bl.khop_last_error(g, 2),
        "WM-SAR": bl.wm_sar,
        "FullReplan": bl.full_replan,
    }


def run(verbose: bool = True) -> list[dict]:
    if __package__:
        from . import _common as C
    else:
        import _common as C

    _, agent_graphs, _ = C.build_dataset()
    rows = []
    for name, fn in _methods().items():
        r = me.evaluate_method(agent_graphs, fn)
        rows.append({
            "Method": name,
            "Recovery": C.fmt(r["recovery"]),
            "CostNorm": C.fmt(r["cost_norm_recovery"]),
            "Tokens/Rec": C.fmt(r["tokens_per_recovery"], 0),
            "PDred": C.fmt(r["pd_reduction"], 2),
            "DownErrRed": C.fmt(r["downstream_err_reduction"], 2),
            "Tokens": C.fmt(r["mean_token_cost"], 0),
        })
    if verbose:
        C.print_table("Table 1: Agent World-Model Failed-Case Repair", rows, COLUMNS)
    C.save_csv("table1_agent_wm_repair.csv", rows, COLUMNS)
    return rows


def method_specs(top_k=3, trace_budget=4, repair_budget=14.0):
    for name, value in (("top_k", top_k), ("trace_budget", trace_budget)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if (isinstance(repair_budget, bool) or not isinstance(repair_budget, (int, float))
            or not math.isfinite(repair_budget) or repair_budget <= 0):
        raise ValueError("repair_budget must be finite and positive")
    config = ReCoreConfig()
    return [
        ("ReCore", partial(bl.wm_sar, config=config, budget=repair_budget),
         {"config": asdict(config), "repair_cost_budget": repair_budget}),
        ("Oracle-Region", bl.oracle_region, {}),
        ("LLMRepair-Full-Plan", bl.llm_repair_full_plan, {}),
        ("TraceScan-Full-Point", partial(bl.trace_scan_full, budget=trace_budget),
         {"budget": trace_budget}),
        (f"Top-B-Nodes (K={top_k})", partial(bl.top_b_nodes, budget=top_k),
         {"budget": top_k}),
        ("TraceScan-w4-Point", partial(bl.trace_scan_window, w=4, budget=trace_budget),
         {"window": 4, "budget": trace_budget}),
        (f"Top-B-Edges (K={top_k})", partial(bl.top_b_edges, budget=top_k),
         {"budget": top_k}),
        ("PageRank-Subgraph", bl.pagerank_subgraph, {}),
        ("k-hop-k2", partial(bl.khop_last_error, k=2), {"k": 2}),
        ("Uncertainty-Subgraph", bl.uncertainty_subgraph, {}),
        ("TraceScan-w2-Point", partial(bl.trace_scan_window, w=2, budget=trace_budget),
         {"window": 2, "budget": trace_budget}),
        ("TraceScan-w1-Point", partial(bl.trace_scan_window, w=1, budget=trace_budget),
         {"window": 1, "budget": trace_budget}),
        ("LastError-Point", bl.last_error_point, {}),
        ("FirstFailedCall-Point", bl.first_failed_call_point, {}),
    ]


def _nonnegative_finite(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


def evaluate_plan(graph, selector, parameters):
    plan = selector(graph)
    selected = set(plan.nodes)
    if not selected.issubset(graph.nodes()):
        raise ValueError("repair selection contains nodes outside the graph")
    token_cost = _nonnegative_finite(plan.token_cost, "token_cost")
    latency = _nonnegative_finite(plan.latency, "latency")
    if isinstance(plan.n_edits, bool) or not isinstance(plan.n_edits, int) or plan.n_edits < 0:
        raise ValueError("n_edits must be a non-negative integer")
    measurements = re.measure_recovery(graph, selected)
    cleaned = {}
    for key, value in measurements.items():
        if key == "region_iou" and not math.isfinite(value):
            cleaned[key] = None
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"non-finite measurement: {key}")
        else:
            cleaned[key] = value
    before = cleaned["downstream_err_before"]
    cleaned["downstream_err_reduction_pct"] = (
        100.0 * cleaned["downstream_err_reduction"] / before if before > 0 else None
    )
    return {
        "plan_method": plan.method,
        "selected_nodes": sorted(selected, key=str),
        "parameters": parameters,
        "token_cost": token_cost,
        "latency_proxy": latency,
        "n_edits": plan.n_edits,
        "is_subgraph": bool(plan.is_subgraph),
        "window": plan.window,
        "metrics": cleaned,
    }


def aggregate_results(rows):
    rows = list(rows)
    if not rows:
        raise ValueError("at least one result is required")
    n = len(rows)
    successful = [row for row in rows if row["metrics"]["recovered"]]
    tokens = [row["token_cost"] for row in rows]
    total_tokens = sum(tokens)
    n_success = len(successful)
    mean_tokens = float(mean(tokens))
    summary = {
        "n": n,
        "n_attempted": n,
        "n_recovered": n_success,
        "n_failed": n - n_success,
        "recovery": n_success / n,
        "total_tokens": total_tokens,
        "mean_token_cost": mean_tokens,
        "std_token_cost": float(pstdev(tokens)),
        "mean_tokens_successful_only": (
            float(mean(row["token_cost"] for row in successful)) if successful else None
        ),
        "total_attempted_tokens_per_success": total_tokens / n_success if n_success else None,
        "table_tok_per_rec": (
            float(mean(row["token_cost"] for row in successful)) if successful else mean_tokens
        ),
        "table_tok_per_rec_uses_attempt_mean_fallback": not successful,
        "cost_norm_recovery": (n_success / n) / (mean_tokens / 1000.0) if mean_tokens else None,
        "mean_latency_proxy": float(mean(row["latency_proxy"] for row in rows)),
        "mean_edits": float(mean(row["n_edits"] for row in rows)),
        "subgraph_plan_fraction": sum(row["is_subgraph"] for row in rows) / n,
    }
    metric_names = (
        "final_err_before", "final_err_after", "final_err_reduction",
        "downstream_err_before", "downstream_err_after", "downstream_err_reduction",
        "downstream_err_reduction_pct", "pd_before", "pd_after", "pd_reduction",
        "local_inconsistency", "region_iou", "region_size",
    )
    for metric in metric_names:
        values = [row["metrics"][metric] for row in rows if row["metrics"][metric] is not None]
        summary[f"mean_{metric}"] = float(mean(values)) if values else None
        summary[f"std_{metric}"] = float(pstdev(values)) if values else None
        summary[f"n_{metric}"] = len(values)
    total_before = sum(row["metrics"]["downstream_err_before"] for row in rows)
    total_reduction = sum(row["metrics"]["downstream_err_reduction"] for row in rows)
    summary["pooled_downstream_err_reduction_pct"] = (
        100.0 * total_reduction / total_before if total_before > 0 else None
    )
    return summary


def run_experiment(n_agent=120, n_gwm=80, seed=42, top_k=3,
                   trace_budget=4, repair_budget=14.0):
    for name, value in (("n_agent", n_agent), ("n_gwm", n_gwm), ("seed", seed)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if n_agent + n_gwm == 0:
        raise ValueError("at least one instance is required")
    specs = method_specs(top_k, trace_budget, repair_budget)
    dataset = dg.generate_dataset(n_agent=n_agent, n_gwm=n_gwm, seed=seed)
    dataset_stats = dict(dataset["stats"])
    if not n_agent:
        dataset_stats["mean_agent_horizon"] = None
    if not n_gwm:
        dataset_stats["mean_gwm_horizon"] = None
    records = []
    domains = (("agent", n_agent, seed), ("gwm", n_gwm, seed + 1))
    for domain, expected, generation_seed in domains:
        rollouts = dataset[domain]
        if len(rollouts) != expected:
            raise ValueError(f"{domain} generated {len(rollouts)} of {expected} requested instances")
        for index, rollout in enumerate(rollouts):
            graph = fg.world_model_failure_to_graph(rollout)
            instance_id = f"{domain}_s{generation_seed}_i{index:04d}"
            record = {
                "instance_id": instance_id,
                "domain": domain,
                "generation_seed": generation_seed,
                "rollout_id": graph.graph.get("rollout_id"),
                "failure_type": graph.graph.get("failure_type"),
                "n_nodes": graph.number_of_nodes(),
                "n_edges": graph.number_of_edges(),
                "target_node": graph.graph.get("t_star"),
                "ground_truth_region": sorted(graph.graph.get("gt_region", set()), key=str),
                "results": {},
            }
            for name, selector, parameters in specs:
                try:
                    record["results"][name] = evaluate_plan(graph, selector, parameters)
                except Exception as exc:
                    raise RuntimeError(f"Failed {name} on {instance_id}") from exc
            records.append(record)
    method_names = [name for name, _, _ in specs]
    summaries = {
        name: aggregate_results(record["results"][name] for record in records)
        for name in method_names
    }
    by_domain = {
        domain: {
            name: aggregate_results(record["results"][name] for record in records
                                    if record["domain"] == domain)
            for name in method_names
        }
        for domain, expected, _ in domains if expected
    }
    return {
        "n": len(records),
        "n_agent": n_agent,
        "n_gwm": n_gwm,
        "seed": seed,
        "generation_seeds": {"agent": seed, "gwm": seed + 1},
        "evaluation_kind": "legacy_mixed_synthetic_full_baseline_comparison",
        "token_cost_kind": "synthetic_proxy",
        "latency_kind": "synthetic_proxy",
        "repair_kind": "simulated",
        "llm_api_calls": 0,
        "dispersion": "population_standard_deviation",
        "recovery_definition": "final_error_after <= recovery_frac * final_error_before and final_error_after <= recovery_abs",
        "repair_parameters": {
            "repair_strength": re.REPAIR_STRENGTH,
            "propagation_gain": re.PROP_GAIN,
            "active_error_threshold": re.ERR_EPS,
            "recovery_frac": re.RECOVERY_FRAC,
            "recovery_abs": re.RECOVERY_ABS,
        },
        "token_proxy_parameters": {
            "base_tokens": bl.BASE_TOKENS,
            "latency_per_ktok": bl.LATENCY_PER_KTOK,
            "latency_per_edit": bl.LATENCY_PER_EDIT,
        },
        "aggregation": {
            "mean_tokens_successful_only": "mean cost conditional on recovery; null for no successes",
            "total_attempted_tokens_per_success": "sum of all attempted costs divided by successful recoveries; null for no successes",
            "table_tok_per_rec": "conditional-success mean, or mean attempted cost when no recovery succeeds",
            "mean_downstream_err_reduction_pct": "unweighted mean of per-instance percentage reductions with positive before-error",
            "pooled_downstream_err_reduction_pct": "percentage reduction of the pooled downstream error",
        },
        "methods": {name: parameters for name, _, parameters in specs},
        "dataset_stats": dataset_stats,
        "summaries": summaries,
        "summaries_by_domain": by_domain,
        "per_instance": records,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-baselines", action="store_true")
    parser.add_argument("--n-agent", type=int, default=120)
    parser.add_argument("--n-gwm", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--trace-budget", type=int, default=4)
    parser.add_argument("--repair-budget", type=float, default=14.0)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "results" / "exp_full_baselines.json")
    args = parser.parse_args(argv)
    if not args.full_baselines:
        return run()
    try:
        output = run_experiment(args.n_agent, args.n_gwm, args.seed, args.top_k,
                                args.trace_budget, args.repair_budget)
    except ValueError as exc:
        parser.error(str(exc))
    serialized = json.dumps(output, indent=2, allow_nan=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(serialized + "\n", encoding="utf-8")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
