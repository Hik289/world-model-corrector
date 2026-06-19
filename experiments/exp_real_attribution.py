"""Real Multi-Agent Attribution Experiment — Wave B S4 (spec §3.3).

Dataset: Kevin355/Who_and_When (HuggingFace, arxiv 2505.00212).
  184 annotated multi-agent failure tasks with per-step root-cause labels.
  Each task: history = list of {role, name, content}; ground-truth
  mistake_agent + mistake_step.

Adapter: convert each task into a directed graph where:
  - One node per `history` message
  - Edge i → i+1 (sequential causal chain across agent calls)
  - Plus an edge from every prior message of the same agent name to the
    next message of that same agent (intra-agent dependency)
  - node_type   = the agent's role (e.g. 'planner', 'executor', or in
                  Who&When: the agent's `name` like 'Excel_Expert')
  - true_root  = node corresponding to `mistake_step` (0-indexed in history)
  - err        = 1.0 if step >= mistake_step else 0.0 (cascade visualisation)

Methods (Wave-B-S4 subset, per Director spec):
  - LastError-Point     : pick the last node with err>0 (= the failed final step)
  - TraceScan-w4-Point  : pick the 4-node window around the highest-error node
  - LocalRepair-2Hop    : pick the 2-hop neighbourhood of the highest-error node
  - WM-SAR              : run the WM-SAR region extractor

Metric: Rec-Exact = does the method's selected region include the true root-cause
node (i.e. the message authored at mistake_step)?
Optional: Rec-Agent = does the region include ANY message authored by the true
mistake_agent? (less strict, since multiple messages from the same agent exist)

INFEASIBLE caveat (spec): the Who&When dataset has `mistake_step` labels but
does not provide a ground-truth "repair region" with IoU semantics. We
therefore report Recovery (Rec-Exact / Rec-Agent / Region-Size) only, no IoU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from glob import glob

import numpy as np
import networkx as nx

# Make sure project root is importable
HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, PROJ)

from wm_sar.engineering_baselines import greedy_point, window_repair, local_khop
from wm_sar.region_extractor import WMSAR, WMSARConfig


# ─────────────────────────────────────────────────────────────────────────────
#  Adapter: Who&When task JSON  →  failure graph (nx.DiGraph)
# ─────────────────────────────────────────────────────────────────────────────
def task_to_graph(task: dict) -> tuple[nx.DiGraph, str, str]:
    """Build a failure graph from one Who&When task.

    Returns:
        G          : nx.DiGraph with per-node attrs
                     (node_type, err, state, step_idx, agent_name)
        true_root  : node ID corresponding to the mistake_step
        failure_desc: short text description (mistake_reason)
    """
    history = task.get("history", []) or []
    mistake_step_raw = task.get("mistake_step", -1)
    try:
        mistake_step = int(mistake_step_raw)
    except (ValueError, TypeError):
        mistake_step = -1
    mistake_agent = task.get("mistake_agent", "") or ""
    failure_desc = task.get("mistake_reason", "") or "Multi-agent task failed"

    # IMPORTANT (no-leakage policy):
    # The methods evaluated downstream MUST NOT see mistake_step / mistake_agent
    # in any node attribute. The ONLY observable signal we expose is that
    # the FINAL message in the conversation is the visible failure point
    # (success_flag = 0 at the sink). All upstream nodes look healthy from
    # an outside observer's perspective — that's the realistic deployment
    # setting and is what spec §3.3 INFEASIBLE caveat refers to.
    G = nx.DiGraph()
    node_ids: list[str] = []
    n_msgs = len(history)
    for i, msg in enumerate(history):
        agent_name = msg.get("name") or msg.get("role") or f"agent_{i}"
        role = msg.get("role", "assistant")
        node_type = (agent_name or role).lower().replace(" ", "_")
        nid = f"msg_{i:03d}_{agent_name}"
        node_ids.append(nid)

        is_sink = (i == n_msgs - 1)
        # Only the sink carries observable error (= the visible failure).
        # All upstream nodes start at err=0 (we do NOT use mistake_step here).
        err = 1.0 if is_sink else 0.0
        # Lightweight textual-error heuristic: any explicit "error"/"fail"
        # keyword in the message content adds a small bump. This gives a
        # *weak, observable* signal (not the GT label).
        content = (msg.get("content") or "").lower()
        if any(k in content for k in ("error", "exception", "traceback",
                                       "failed", "failure")):
            err = max(err, 0.3)

        # 8-dim state matching the synthetic schema:
        # [activation, load, latency, error_prob, throughput, confidence,
        #  dependency_ok, success_flag]
        state = np.array([
            1.0,                               # activation (always on)
            min(1.0, len(msg.get("content", "")) / 5000.0),  # load proxy
            0.2,                               # latency (no info)
            err,                               # error_prob = observable only
            max(0.1, 1.0 - err),               # throughput inversely
            max(0.1, 1.0 - err),               # confidence inversely
            1.0,                               # dep_ok (unknown observable)
            0.0 if is_sink else 1.0,           # success_flag = 0 at sink only
        ], dtype=float)

        G.add_node(nid,
                   node_type=node_type,
                   err=err,
                   state=state,
                   step_idx=i,
                   agent_name=agent_name,
                   role=role)

    # Sequential edges (i → i+1)
    for i in range(len(node_ids) - 1):
        G.add_edge(node_ids[i], node_ids[i + 1], edge_type="next")

    # Intra-agent edges (last msg by same agent → this msg)
    last_by_agent: dict[str, str] = {}
    for nid in node_ids:
        agent = G.nodes[nid]["agent_name"]
        if agent in last_by_agent and last_by_agent[agent] != nid:
            prev = last_by_agent[agent]
            # skip if already adjacent
            if not G.has_edge(prev, nid):
                G.add_edge(prev, nid, edge_type="intra_agent")
        last_by_agent[agent] = nid

    # Sink convention
    if node_ids:
        G.graph["t_star"] = node_ids[-1]
        G.graph["domain"] = "real_multi_agent_attribution"

    # true_root = node at mistake_step (if in range), else fallback
    if 0 <= mistake_step < len(node_ids):
        true_root = node_ids[mistake_step]
    elif mistake_agent:
        # fallback: first node authored by mistake_agent
        cand = [n for n in node_ids if G.nodes[n]["agent_name"] == mistake_agent]
        true_root = cand[0] if cand else (node_ids[0] if node_ids else "")
    else:
        true_root = node_ids[0] if node_ids else ""

    return G, true_root, failure_desc


# ─────────────────────────────────────────────────────────────────────────────
#  Methods (no LLM — pure structural; for spec §11.5 alignment)
# ─────────────────────────────────────────────────────────────────────────────
def method_last_error(G: nx.DiGraph) -> set[str]:
    """Pick the last node with err > 0 (the most-downstream error message)."""
    errs = [(G.nodes[n].get("err", 0.0), G.nodes[n].get("step_idx", 0), n)
            for n in G.nodes()]
    pos = [(e, i, n) for e, i, n in errs if e > 0]
    if not pos:
        # fallback to max-error node (= 0 here means no errors labelled)
        return {max(errs, key=lambda x: (x[0], x[1]))[2]}
    return {max(pos, key=lambda x: x[1])[2]}


def method_tracescan_w4(G: nx.DiGraph) -> set[str]:
    """4-node sequential window around the highest-error node."""
    rr = window_repair(G, window=4)
    return rr.selected_nodes


def method_local_2hop(G: nx.DiGraph) -> set[str]:
    """2-hop neighbourhood of the highest-error node."""
    rr = local_khop(G, k=2)
    return rr.selected_nodes


def method_wmsar(G: nx.DiGraph) -> set[str]:
    """WM-SAR region extractor."""
    extractor = WMSAR(WMSARConfig())
    return extractor.repair_region(G)


METHODS = {
    "LastError-Point":    method_last_error,
    "TraceScan-w4-Point": method_tracescan_w4,
    "LocalRepair-2Hop":   method_local_2hop,
    "WM-SAR":             method_wmsar,
}


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(G: nx.DiGraph, true_root: str, region: set[str]) -> dict:
    """Compute Rec-Exact (true_root in region) and Rec-Agent (any node by the
    same agent name in region)."""
    if not region:
        return {
            "rec_exact": 0, "rec_agent": 0,
            "region_size": 0,
        }
    rec_exact = int(true_root in region)
    true_agent = G.nodes[true_root].get("agent_name", "") if true_root in G else ""
    rec_agent = int(any(
        G.nodes[n].get("agent_name", "") == true_agent and true_agent
        for n in region
    ))
    return {
        "rec_exact":   rec_exact,
        "rec_agent":   rec_agent,
        "region_size": len(region),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Driver
# ─────────────────────────────────────────────────────────────────────────────
def load_tasks(data_dir: str, max_n: int = -1) -> list[dict]:
    """Load Who&When task JSONs from a local mirror."""
    files = sorted(glob(os.path.join(data_dir, "*.json")))
    tasks = []
    for fp in files:
        try:
            with open(fp) as f:
                t = json.load(f)
            t["__source_file"] = os.path.basename(fp)
            tasks.append(t)
        except Exception as e:
            print(f"  WARN: failed to load {fp}: {e}")
        if max_n > 0 and len(tasks) >= max_n:
            break
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing Who&When task JSONs "
                             "(e.g. /path/to/who_and_when_dataset/Algorithm-Generated)")
    parser.add_argument("--n", type=int, default=20,
                        help="Number of tasks to evaluate (default 20)")
    parser.add_argument("--out", type=str,
                        default=os.path.join(HERE, "results",
                                              "exp_real_attribution.json"))
    parser.add_argument("--seed", type=int, default=42,
                        help="(unused — methods deterministic; kept for log)")
    parser.add_argument("--min-history", type=int, default=2,
                        help="Skip tasks with shorter history than this")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Real Multi-Agent Attribution Experiment (Wave B S4)")
    print(f"  data_dir={args.data_dir}  n={args.n}")
    print(f"{'='*60}\n")

    tasks_all = load_tasks(args.data_dir, max_n=-1)
    print(f"  Loaded {len(tasks_all)} task JSONs from {args.data_dir}")

    # Filter usable
    usable = []
    for t in tasks_all:
        hist = t.get("history", []) or []
        try:
            ms = int(t.get("mistake_step", -1))
        except (ValueError, TypeError):
            ms = -1
        if len(hist) < args.min_history:
            continue
        if ms < 0 or ms >= len(hist):
            continue
        usable.append(t)
    print(f"  Usable (history ≥ {args.min_history}, mistake_step in range): "
          f"{len(usable)}")

    if not usable:
        print("  No usable tasks. Abort.")
        sys.exit(1)

    # Take first --n
    selected = usable[:args.n]
    print(f"  Selected first {len(selected)} for evaluation.\n")

    per_instance = []
    summaries = {m: {"correct_exact": 0, "correct_agent": 0,
                     "region_sizes": [], "n": 0}
                 for m in METHODS}

    for i, task in enumerate(selected):
        G, true_root, fdesc = task_to_graph(task)
        if not true_root or not G.number_of_nodes():
            continue
        true_agent = G.nodes[true_root].get("agent_name", "")
        N = G.number_of_nodes()
        E = G.number_of_edges()

        print(f"  [{i+1:2d}/{len(selected)}] {task.get('__source_file','?'):<14}  "
              f"N={N:>3} E={E:>3}  root={true_root[:20]:<22} ({true_agent})",
              end="  ", flush=True)

        inst_rec = {
            "instance_id":   f"who_and_when_{i:03d}",
            "source_file":   task.get("__source_file", ""),
            "question_ID":   task.get("question_ID", ""),
            "level":         task.get("level", ""),
            "true_root":     true_root,
            "true_agent":    true_agent,
            "n_nodes":       N,
            "n_edges":       E,
            "results":       {},
        }

        # Run each method
        marks = []
        for m_name, m_fn in METHODS.items():
            try:
                t0 = time.time()
                region = m_fn(G)
                lat = (time.time() - t0) * 1000.0
                ev = evaluate(G, true_root, region)
                ev["latency_ms"] = lat
                inst_rec["results"][m_name] = ev
                summaries[m_name]["n"] += 1
                summaries[m_name]["correct_exact"] += ev["rec_exact"]
                summaries[m_name]["correct_agent"] += ev["rec_agent"]
                summaries[m_name]["region_sizes"].append(ev["region_size"])
                marks.append(f"{m_name.split('-')[0][:5]}={ev['rec_exact']}")
            except Exception as e:
                inst_rec["results"][m_name] = {"error": str(e)[:200]}
                marks.append(f"{m_name.split('-')[0][:5]}=ERR")

        print("  " + "  ".join(marks))
        per_instance.append(inst_rec)

    # Aggregate
    final_summaries = {}
    for m, s in summaries.items():
        if s["n"] == 0:
            continue
        final_summaries[m] = {
            "n":               s["n"],
            "rec_exact":       s["correct_exact"] / s["n"],
            "rec_agent":       s["correct_agent"] / s["n"],
            "mean_region_size": float(np.mean(s["region_sizes"])),
            "std_region_size":  float(np.std(s["region_sizes"])),
        }

    print(f"\n{'='*60}")
    print(f"  Summary (n={len(per_instance)} usable instances)")
    print(f"{'='*60}")
    print(f"  {'Method':<22}  {'Rec-Exact':>9}  {'Rec-Agent':>9}  "
          f"{'Region Size':>12}")
    print(f"  {'-'*60}")
    for m in ["LastError-Point", "TraceScan-w4-Point", "LocalRepair-2Hop",
              "WM-SAR"]:
        s = final_summaries.get(m, {})
        if not s:
            continue
        print(f"  {m:<22}  {s['rec_exact']:>9.3f}  {s['rec_agent']:>9.3f}  "
              f"{s['mean_region_size']:>5.1f} ± {s['std_region_size']:.1f}")

    # Save
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    output = {
        "dataset":      "Kevin355/Who_and_When",
        "subset_dir":   args.data_dir,
        "n":            len(per_instance),
        "seed":         args.seed,
        "summaries":    final_summaries,
        "per_instance": per_instance,
        "notes": (
            "Rec-Exact = method's selected region contains the GT mistake_step node. "
            "Rec-Agent = method's selected region contains ANY message authored by "
            "the GT mistake_agent. No IoU reported (dataset has no GT repair region)."
        ),
    }
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
        f.flush(); os.fsync(f.fileno())
    print(f"\n  Saved: {args.out}")


if __name__ == "__main__":
    main()
