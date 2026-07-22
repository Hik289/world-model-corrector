# Artifact Guide

Operational notes for reproducing `Repair the Amplifier, Not the Symptom` from the public `world-model-corrector` repository.

## Review Path

- `wm_sar/`: Project-specific implementation subtree.
- `experiments/`: Experiment drivers, ablations, and benchmark-specific runners.
- `figures/`: README and paper-facing figures.

## Environment Files

- `requirements.txt`: Primary Python dependency list.

## Smoke Checks

Run these checks before long jobs:

```bash
python -m compileall -q .
```

If no smoke command is tracked, use the README Quick Start with the smallest seed, sample, or task count.

## Reproduction Entry Points

Main tracked entry points for paper-scale or benchmark-scale runs:

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

## Data And Outputs

- API-backed runs should read credentials from environment variables or local `.env` files only; never commit real keys or provider-specific secrets.
- Record provider endpoint, model/deployment name, sampling parameters, and execution date for every API-backed table or figure.
- Treat generated JSONL files, logs, caches, model checkpoints, and benchmark downloads as local artifacts unless explicitly tracked as fixtures.
- For stochastic experiments, record seeds, task counts, dataset splits, and the exact git commit used for the run.

## Reporting Checklist

- `git rev-parse HEAD`
- Python version and dependency-install command
- Full command line for every table, figure, or benchmark cell
- Paths to raw outputs and aggregation scripts
- External data, benchmark, or API-backed steps that were intentionally skipped
