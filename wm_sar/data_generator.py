from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


AGENT_FAILURE_TYPES = [
    "prediction_drift",
    "false_dependency",
    "tool_misfire",
    "belief_contradiction",
    "artifact_corruption",
    "cascading_subgoal",
]

GWM_MODEL_TYPES = ["GCN-GWM", "MPNN-GWM", "GPS-GWM", "ActionNode-GWM", "Error-Aware-GWM"]


@dataclass
class AgentStep:


    t: int
    action: str
    tool: str


    predicted_state: float
    observed_state: float
    prediction_confidence: float
    prediction_error: float
    observed_error: float
    uncertainty: float
    risk: float
    agent_identity: str
    status: str
    contradicts_observation: bool
    on_target_path: bool
    downstream_of_error: bool
    tool_response_ok: bool
    artifact_ok: bool
    subgoal_done: bool


@dataclass
class AgentRollout:
    rollout_id: str
    domain: str = "agent_wm"
    task_goal: str = ""
    failure_type: str = ""
    horizon: int = 0
    root_cause_t: int = -1
    root_cause_node_type: str = ""
    failed: bool = True
    steps: list[AgentStep] = field(default_factory=list)

    gt_region_steps: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class GWMRollout:


    rollout_id: str
    domain: str = "parametric_gwm"
    model_type: str = ""
    failure_type: str = ""
    horizon: int = 0
    n_nodes: int = 0
    root_cause_t: int = -1
    root_cause_node: int = -1
    failed: bool = True

    pred_node: list[list[float]] = field(default_factory=list)
    true_node: list[list[float]] = field(default_factory=list)
    pred_adj: list[list[list[float]]] = field(default_factory=list)
    true_adj: list[list[list[float]]] = field(default_factory=list)
    gt_region_steps: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_TOOLS = ["search", "fetch", "compute", "write", "verify", "route", "merge", "deploy"]
_AGENTS = ["planner", "executor", "critic", "retriever", "coder"]


def _gen_agent_rollout(rng: np.random.Generator, idx: int) -> AgentRollout:
    failure_type = AGENT_FAILURE_TYPES[idx % len(AGENT_FAILURE_TYPES)]
    T = int(rng.integers(8, 18))
    root_t = int(rng.integers(1, max(2, T - 3)))

    roll = AgentRollout(
        rollout_id=f"agent_{idx:04d}",
        task_goal=f"task_{idx % 20}",
        failure_type=failure_type,
        horizon=T,
        root_cause_t=root_t,
    )


    base = rng.normal(0.0, 0.05, size=T + 1).cumsum() * 0.1


    amp = rng.uniform(1.15, 1.45)
    inj = rng.uniform(0.4, 0.8)


    gt_region = list(range(root_t, T + 1))

    rc_node_type = {
        "prediction_drift": "predicted_state",
        "false_dependency": "dependency",
        "tool_misfire": "tool_response",
        "belief_contradiction": "belief_state",
        "artifact_corruption": "artifact",
        "cascading_subgoal": "subgoal",
    }[failure_type]
    roll.root_cause_node_type = rc_node_type
    roll.gt_region_steps = gt_region

    err = 0.0
    for t in range(T + 1):
        observed = base[t]
        downstream = t >= root_t

        if t < root_t:
            pred_err = abs(rng.normal(0.02, 0.01))
            err = pred_err
        elif t == root_t:
            err = inj
            pred_err = inj
        else:

            err = err * amp + abs(rng.normal(0.0, 0.03))
            pred_err = err

        predicted = observed + (err if downstream else rng.normal(0, 0.02))
        conf = float(np.clip(1.0 - pred_err, 0.02, 0.99))
        unc = float(np.clip(pred_err * rng.uniform(0.6, 1.1) + rng.normal(0, 0.02), 0.01, 1.5))
        risk = float(np.clip(pred_err * 0.8 + (0.3 if downstream else 0.0), 0.0, 2.0))

        status = "ok"
        if downstream:
            status = "degraded" if t < T else "failed"
        if t == root_t:
            status = "degraded"


        contradicts = False
        tool_ok = True
        artifact_ok = True
        subgoal_done = True
        if downstream:
            if failure_type == "belief_contradiction" and t == root_t:
                contradicts = True
            if failure_type == "tool_misfire" and t == root_t:
                tool_ok = False
            if failure_type == "artifact_corruption" and t >= root_t:
                artifact_ok = False
            if failure_type == "cascading_subgoal" and t >= root_t:
                subgoal_done = False

        step = AgentStep(
            t=t,
            action=f"a_{t}",
            tool=_TOOLS[(idx + t) % len(_TOOLS)],
            predicted_state=float(predicted),
            observed_state=float(observed),
            prediction_confidence=conf,
            prediction_error=float(pred_err),
            observed_error=float(abs(predicted - observed)),
            uncertainty=unc,
            risk=risk,
            agent_identity=_AGENTS[(idx + t) % len(_AGENTS)],
            status=status,
            contradicts_observation=contradicts,
            on_target_path=True,
            downstream_of_error=downstream,
            tool_response_ok=tool_ok,
            artifact_ok=artifact_ok,
            subgoal_done=subgoal_done,
        )
        roll.steps.append(step)

    return roll


def generate_agent_wm_rollouts(n: int = 120, seed: int = 42) -> list[AgentRollout]:
    rng = np.random.default_rng(seed)
    return [_gen_agent_rollout(rng, i) for i in range(n)]


def _gen_gwm_rollout(rng: np.random.Generator, idx: int) -> GWMRollout:
    model_type = GWM_MODEL_TYPES[idx % len(GWM_MODEL_TYPES)]
    failure_type = ["node_pred_error", "edge_pred_error", "dynamic_edge_drift"][idx % 3]
    T = int(rng.integers(6, 12))
    n = int(rng.integers(6, 12))
    root_t = int(rng.integers(1, max(2, T - 2)))
    root_node = int(rng.integers(0, n))


    true_node = np.zeros((T + 1, n))
    true_node[0] = rng.normal(0, 1, size=n)

    W = rng.normal(0, 1, size=(n, n))
    W /= (np.abs(np.linalg.eigvals(W)).max() + 1e-9)
    true_adj = np.zeros((T + 1, n, n))
    base_adj = (rng.random((n, n)) < 0.25).astype(float)
    np.fill_diagonal(base_adj, 0.0)
    for t in range(T + 1):
        true_adj[t] = base_adj
        if t > 0:
            true_node[t] = np.tanh(W @ true_node[t - 1]) + rng.normal(0, 0.02, n)

    pred_node = true_node.copy()
    pred_adj = true_adj.copy()


    amp = rng.uniform(1.1, 1.4)
    inj = rng.uniform(0.6, 1.2)
    gt_region = list(range(root_t, T + 1))
    err_vec = np.zeros(n)
    for t in range(root_t, T + 1):
        if t == root_t:
            err_vec[root_node] = inj
        else:

            err_vec = amp * (true_adj[t] @ err_vec) / (n ** 0.5)
            err_vec[root_node] += 0.05
        pred_node[t] = true_node[t] + err_vec
        if failure_type in ("edge_pred_error", "dynamic_edge_drift") and t >= root_t:

            flip = (rng.random(n) < 0.3).astype(float)
            pred_adj[t, root_node] = np.clip(true_adj[t, root_node] + flip, 0, 1)

    return GWMRollout(
        rollout_id=f"gwm_{idx:04d}",
        model_type=model_type,
        failure_type=failure_type,
        horizon=T,
        n_nodes=n,
        root_cause_t=root_t,
        root_cause_node=root_node,
        pred_node=pred_node.tolist(),
        true_node=true_node.tolist(),
        pred_adj=pred_adj.tolist(),
        true_adj=true_adj.tolist(),
        gt_region_steps=gt_region,
    )


def generate_parametric_gwm_rollouts(n: int = 80, seed: int = 43) -> list[GWMRollout]:
    rng = np.random.default_rng(seed)
    return [_gen_gwm_rollout(rng, i) for i in range(n)]


def generate_dataset(
    n_agent: int = 120, n_gwm: int = 80, seed: int = 42
) -> dict[str, Any]:

    agent = generate_agent_wm_rollouts(n_agent, seed)
    gwm = generate_parametric_gwm_rollouts(n_gwm, seed + 1)

    stats = {
        "n_total": n_agent + n_gwm,
        "n_agent_wm": n_agent,
        "n_parametric_gwm": n_gwm,
        "agent_failure_type_counts": {
            ft: sum(1 for r in agent if r.failure_type == ft)
            for ft in AGENT_FAILURE_TYPES
        },
        "gwm_model_type_counts": {
            mt: sum(1 for r in gwm if r.model_type == mt) for mt in GWM_MODEL_TYPES
        },
        "mean_agent_horizon": float(np.mean([r.horizon for r in agent])),
        "mean_gwm_horizon": float(np.mean([r.horizon for r in gwm])),
    }
    return {"agent": agent, "gwm": gwm, "stats": stats}


def save_dataset(dataset: dict[str, Any], outdir: str) -> None:
    import os

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "failed_cases_agent_wm.jsonl"), "w") as f:
        for r in dataset["agent"]:
            f.write(json.dumps(r.to_dict()) + "\n")
    with open(os.path.join(outdir, "failed_cases_parametric_gwm.jsonl"), "w") as f:
        for r in dataset["gwm"]:
            f.write(json.dumps(r.to_dict()) + "\n")
    with open(os.path.join(outdir, "dataset_statistics.json"), "w") as f:
        json.dump(dataset["stats"], f, indent=2)


if __name__ == "__main__":
    ds = generate_dataset()
    print(json.dumps(ds["stats"], indent=2))
