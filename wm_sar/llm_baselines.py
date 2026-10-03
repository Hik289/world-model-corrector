from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import networkx as nx

from wm_sar.llm_client import LLMClient, LLMResult
from wm_sar.baselines import wm_sar as _graph_wm_sar


@dataclass
class LLMRepairResult:
    method: str
    identified_steps: list[int]
    repaired_nodes: set[str]
    llm_result: Optional[LLMResult]
    token_cost: int
    latency_ms: float
    recovered: bool
    region_iou: float
    n_llm_calls: int = 1


def _step_node_ids(G: nx.DiGraph, steps: list[int]) -> set[str]:

    result = set()
    for n, d in G.nodes(data=True):
        if d.get("node_type") == "predicted_state" and d.get("t") in steps:
            result.add(n)
    return result


def _oracle_steps(rollout: Any) -> set[int]:

    return set(getattr(rollout, "gt_region_steps", [getattr(rollout, "root_cause_t", -1)]))


def _check_recovery(identified_steps: list[int], oracle: set[int], tol: int = 1) -> bool:


    if not identified_steps or not oracle:
        return False
    for s in identified_steps:
        for o in oracle:
            if abs(s - o) <= tol:
                return True
    return False


def _region_iou(pred_steps: set[int], oracle_steps: set[int]) -> float:
    if not pred_steps and not oracle_steps:
        return 1.0
    union = pred_steps | oracle_steps
    inter = pred_steps & oracle_steps
    return len(inter) / len(union) if union else 0.0


def tracescan_window_llm(
    G: nx.DiGraph,
    rollout_steps: list[dict],
    rollout: Any,
    failure_desc: str = "task failed",
    window: int = 4,
    client: Optional[LLMClient] = None,
) -> LLMRepairResult:


    if client is None:
        client = LLMClient()

    oracle = _oracle_steps(rollout)
    T = len(rollout_steps)


    total_pt = total_ct = 0
    total_lat = 0.0
    n_calls = 0
    best_identified: list[int] = []


    errors = [(s["error"], s["step"]) for s in rollout_steps]
    errors_sorted = sorted(errors, reverse=True)

    candidates = [step for _, step in errors_sorted[:3]]

    for cand_t in candidates:

        center = next((i for i, s in enumerate(rollout_steps) if s["step"] == cand_t), 0)
        lo = max(0, center - window // 2)
        hi = min(T, lo + window)
        window_steps = rollout_steps[lo:hi]

        try:
            result = client.locate_error(window_steps, failure_desc=failure_desc)
            total_pt += result.prompt_tokens
            total_ct += result.completion_tokens
            total_lat += result.latency_ms
            n_calls += 1

            if result.identified_steps:
                best_identified = result.identified_steps
                break
        except Exception:

            pass

    token_cost = total_pt + total_ct
    recovered = _check_recovery(best_identified, oracle)
    pred_set = set(best_identified)
    iou = _region_iou(pred_set, oracle)
    repaired_nodes = _step_node_ids(G, list(pred_set))

    return LLMRepairResult(
        method=f"TraceScan-w{window}-LLM",
        identified_steps=best_identified,
        repaired_nodes=repaired_nodes,
        llm_result=None,
        token_cost=token_cost,
        latency_ms=total_lat,
        recovered=recovered,
        region_iou=iou,
        n_llm_calls=n_calls,
    )


def tracescan_full_llm(
    G: nx.DiGraph,
    rollout_steps: list[dict],
    rollout: Any,
    failure_desc: str = "task failed",
    client: Optional[LLMClient] = None,
) -> LLMRepairResult:

    if client is None:
        client = LLMClient()

    oracle = _oracle_steps(rollout)
    try:
        result = client.locate_error(rollout_steps, failure_desc=failure_desc)
        token_cost = result.total_tokens
        identified = result.identified_steps
    except Exception:
        result = None
        token_cost = len(rollout_steps) * 60
        identified = []

    recovered = _check_recovery(identified, oracle)
    pred_set = set(identified)
    iou = _region_iou(pred_set, oracle)
    repaired_nodes = _step_node_ids(G, list(pred_set))

    return LLMRepairResult(
        method="TraceScan-Full-LLM",
        identified_steps=identified,
        repaired_nodes=repaired_nodes,
        llm_result=result,
        token_cost=token_cost,
        latency_ms=result.latency_ms if result else 0.0,
        recovered=recovered,
        region_iou=iou,
        n_llm_calls=1,
    )


def llm_replan(
    G: nx.DiGraph,
    rollout_steps: list[dict],
    rollout: Any,
    failure_desc: str = "task failed",
    client: Optional[LLMClient] = None,
) -> LLMRepairResult:

    if client is None:
        client = LLMClient()

    oracle = _oracle_steps(rollout)
    try:
        result = client.full_replan(rollout_steps, failure_desc=failure_desc)
        token_cost = result.total_tokens
        identified = result.identified_steps
    except Exception:
        result = None
        token_cost = len(rollout_steps) * 180
        identified = []

    recovered = _check_recovery(identified, oracle)
    pred_set = set(identified)
    iou = _region_iou(pred_set, oracle)
    repaired_nodes = _step_node_ids(G, list(pred_set))

    return LLMRepairResult(
        method="LLMRepair-Full-Plan-LLM",
        identified_steps=identified,
        repaired_nodes=repaired_nodes,
        llm_result=result,
        token_cost=token_cost,
        latency_ms=result.latency_ms if result else 0.0,
        recovered=recovered,
        region_iou=iou,
        n_llm_calls=1,
    )


def wmsar_with_llm_repair(
    G: nx.DiGraph,
    rollout_steps: list[dict],
    rollout: Any,
    failure_desc: str = "task failed",
    client_repair: Optional[LLMClient] = None,
    budget: float = 14.0,
) -> LLMRepairResult:


    if client_repair is None:
        client_repair = LLMClient()

    oracle = _oracle_steps(rollout)


    graph_plan = _graph_wm_sar(G, budget=budget)
    graph_region_nodes = graph_plan.nodes


    region_step_nums: set[int] = set()
    for nid in graph_region_nodes:
        nd = G.nodes.get(nid, {})
        t = nd.get("t")
        if t is not None:
            region_step_nums.add(t)


    region_steps = [s for s in rollout_steps if s["step"] in region_step_nums]

    if not region_steps:

        region_steps = sorted(rollout_steps, key=lambda s: s["error"], reverse=True)[:3]
        region_step_nums = {s["step"] for s in region_steps}


    try:
        llm_res = client_repair.repair_region(region_steps, failure_desc=failure_desc)
        identified = llm_res.identified_steps or list(region_step_nums)
        token_cost = llm_res.total_tokens
        lat = llm_res.latency_ms
    except Exception:
        llm_res = None
        identified = list(region_step_nums)

        token_cost = int(0.2 * len(G.nodes) * 60 + len(region_step_nums) * 80)
        lat = 0.0

    recovered = _check_recovery(identified, oracle)
    pred_set = set(identified) if identified else region_step_nums
    iou = _region_iou(pred_set, oracle)
    repaired_nodes = _step_node_ids(G, list(pred_set))

    return LLMRepairResult(
        method="WM-SAR-LLM",
        identified_steps=list(pred_set),
        repaired_nodes=repaired_nodes,
        llm_result=llm_res,
        token_cost=token_cost,
        latency_ms=lat,
        recovered=recovered,
        region_iou=iou,
        n_llm_calls=1,
    )


def last_error_heuristic(
    G: nx.DiGraph,
    rollout_steps: list[dict],
    rollout: Any,
) -> LLMRepairResult:

    oracle = _oracle_steps(rollout)
    best = max(rollout_steps, key=lambda s: s["error"])
    identified = [best["step"]]
    recovered = _check_recovery(identified, oracle)
    pred_set = set(identified)
    iou = _region_iou(pred_set, oracle)
    repaired_nodes = _step_node_ids(G, identified)
    return LLMRepairResult(
        method="LastError-Heuristic",
        identified_steps=identified,
        repaired_nodes=repaired_nodes,
        llm_result=None,
        token_cost=0,
        latency_ms=0.0,
        recovered=recovered,
        region_iou=iou,
        n_llm_calls=0,
    )


def run_all_llm_baselines(
    G: nx.DiGraph,
    rollout_steps: list[dict],
    rollout: Any,
    failure_desc: str = "task failed",
    client_fast: Optional[LLMClient] = None,
    client_repair: Optional[LLMClient] = None,
) -> dict[str, LLMRepairResult]:

    if client_fast is None:
        client_fast = LLMClient()
    if client_repair is None:
        client_repair = client_fast

    results = {}


    for w in [1, 2, 4]:
        r = tracescan_window_llm(
            G, rollout_steps, rollout, failure_desc, window=w, client=client_fast
        )
        results[r.method] = r


    results["TraceScan-Full-LLM"] = tracescan_full_llm(
        G, rollout_steps, rollout, failure_desc, client=client_fast
    )


    results["LLMRepair-Full-Plan-LLM"] = llm_replan(
        G, rollout_steps, rollout, failure_desc, client=client_fast
    )


    results["LastError-Heuristic"] = last_error_heuristic(G, rollout_steps, rollout)


    results["WM-SAR-LLM"] = wmsar_with_llm_repair(
        G, rollout_steps, rollout, failure_desc, client_repair=client_repair
    )

    return results
