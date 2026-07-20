# Artifact Guide

This guide maps the public `world-model-corrector` repository to a reviewer-friendly artifact workflow for `Repair the Amplifier, Not the Symptom`. It is meant to make the release easier to inspect in the style of ICML, ICLR, NeurIPS, and similar artifact-review processes.

## What To Inspect First

- `wm_sar/`: Project-specific implementation subtree.
- `experiments/`: Experiment drivers, ablations, and benchmark-specific runners.
- `figures/`: README and paper-facing figures.

## Environment Files

- `requirements.txt`: Primary Python dependency list.

## Minimal Verification

Run these checks in a fresh environment before launching expensive jobs:

```bash
python -m compileall -q .
```

If a smoke command is not tracked, use the README Quick Start with the smallest available seed, sample, or task count.

## Reproduction And Analysis Entry Points

These are the main tracked files to inspect for paper-scale or benchmark-scale reproduction. Some require arguments, credentials, downloaded benchmarks, or local data paths described in the README.

- `python experiments/exp1_agent_wm_repair.py`
- `python experiments/exp2_parametric_gwm_repair.py`
- `python experiments/exp3_subgraph_vs_pointwise.py`
- `python experiments/exp4_spectral_reduction.py`
- `python experiments/exp5_context_limited.py`
- `python experiments/exp6_ablation.py`
- `python experiments/exp_agent.py`
- `python experiments/exp_agent_llm.py`
- `python experiments/exp_benchmarks.py`
- `python experiments/exp_budget.py`
- `python experiments/exp_cascade_gain.py`
- `python experiments/exp_llm.py`
- `python experiments/exp_multiapi.py`
- `python experiments/exp_real_attribution.py`

## Figure Assets

- `figures/intuition.png`
- `figures/pipeline.png`

## Data, Credentials, And Generated Outputs

- API-backed runs should read credentials from environment variables or local `.env` files only; never commit real keys or provider-specific secrets.
- Record provider endpoint, model/deployment name, sampling parameters, and execution date for every API-backed table or figure.
- Treat generated JSONL files, logs, caches, model checkpoints, and benchmark downloads as local artifacts unless explicitly tracked as fixtures.
- For stochastic experiments, record seeds, task counts, dataset splits, and the exact git commit used for the run.

## Reviewer Reporting Checklist

- `git rev-parse HEAD`
- Python version and dependency-install command
- Full command line for every table, figure, or benchmark cell
- Paths to raw outputs and aggregation scripts
- External data, benchmark, or API-backed steps that were intentionally skipped
