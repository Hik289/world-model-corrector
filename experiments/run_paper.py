from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
OFFLINE_EXPERIMENTS = (
    "main", "ablation", "budget", "cascade", "topology",
    "full_baselines", "dataset_stats",
)
API_EXPERIMENTS = ("llm", "multiapi")
SCRIPTS = {
    "main": "exp_agent.py",
    "ablation": "exp_agent_ablation.py",
    "budget": "exp_budget.py",
    "cascade": "exp_cascade_gain.py",
    "topology": "exp_benchmarks.py",
    "full_baselines": "exp_full_baselines.py",
    "dataset_stats": "exp_dataset_stats.py",
    "llm": "exp_agent_llm.py",
    "multiapi": "exp_multiapi.py",
}


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", choices=tuple(SCRIPTS),
                        default=list(OFFLINE_EXPERIMENTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--llm-seeds", nargs="+", type=int)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--llm-n", type=int, default=50)
    parser.add_argument("--multiapi-n", type=int, default=30)
    parser.add_argument("--n-agent", type=int, default=120)
    parser.add_argument("--n-gwm", type=int, default=80)
    parser.add_argument("--H_max", "--H-max", dest="H_max", type=int, default=32,
                        help="Horizon for main, ablation, and dataset_stats only")
    parser.add_argument("--model")
    parser.add_argument("--models", default="all")
    parser.add_argument("--allow-api", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "experiments" / "results" / "paper")
    return parser


def validate_arguments(args):
    for name in ("n", "llm_n", "multiapi_n", "H_max"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be at least 1")
    if args.n_agent < 0 or args.n_gwm < 0 or args.n_agent + args.n_gwm == 0:
        raise ValueError("n_agent and n_gwm must be non-negative with a positive total")
    for name in ("seeds", "llm_seeds"):
        values = getattr(args, name)
        if values is not None and (not values or len(values) != len(set(values))
                                   or any(value < 0 for value in values)):
            raise ValueError(f"{name} must contain unique non-negative integers")
    if len(args.experiments) != len(set(args.experiments)):
        raise ValueError("experiments must be unique")
    if any(name in API_EXPERIMENTS for name in args.experiments) and not args.allow_api:
        raise ValueError("llm and multiapi require explicit --allow-api")
    model_keys = [value.strip() for value in args.models.split(",")]
    available = {"gpt-4o-mini", "gpt-4o", "gemini-2.5-flash"}
    if args.models != "all" and (len(model_keys) != len(set(model_keys))
                                 or any(value not in available for value in model_keys)):
        raise ValueError("models must be all or unique supported model keys")
    if args.model is not None and not args.model.strip():
        raise ValueError("model must not be empty")


def build_plan(args):
    validate_arguments(args)
    output_dir = args.out_dir.expanduser().resolve()
    jobs = []
    for experiment in args.experiments:
        seeds = args.llm_seeds if experiment == "llm" and args.llm_seeds else args.seeds
        for seed in seeds:
            output = output_dir / experiment / f"seed_{seed}.json"
            command = [sys.executable, str(ROOT / "experiments" / SCRIPTS[experiment])]
            parameters = {"seed": seed}
            if experiment == "full_baselines":
                command.extend(["--n-agent", str(args.n_agent), "--n-gwm", str(args.n_gwm)])
                parameters.update(n_agent=args.n_agent, n_gwm=args.n_gwm)
            else:
                n = args.llm_n if experiment == "llm" else (
                    args.multiapi_n if experiment == "multiapi" else args.n)
                command.extend(["--n", str(n)])
                parameters["n"] = n
            command.extend(["--seed", str(seed), "--out", str(output)])
            if experiment in ("main", "ablation", "dataset_stats"):
                command.extend(["--H_max", str(args.H_max)])
                parameters["H_max"] = args.H_max
            elif experiment in ("budget", "cascade", "topology"):
                parameters["H_max"] = 32
            if experiment == "llm":
                parameters["model"] = args.model
                parameters["model_resolution"] = "explicit" if args.model else "child_configuration"
                if args.model:
                    command.extend(["--model", args.model])
            if experiment == "multiapi":
                model_list = "all" if args.models == "all" else ",".join(
                    value.strip() for value in args.models.split(","))
                command.extend(["--models", model_list])
                parameters["models"] = model_list
            artifacts = [str(output)]
            if experiment == "llm":
                artifacts.extend([
                    str(output.with_name(f"{output.stem}_seed{seed}.json")),
                    str(output.with_name(f"{output.stem}_seed{seed}.jsonl")),
                ])
            jobs.append({
                "id": f"{experiment}:seed={seed}",
                "experiment": experiment,
                "seed": seed,
                "command": command,
                "output": str(output),
                "artifacts": artifacts,
                "effective_parameters": parameters,
                "status": "planned",
                "returncode": None,
            })
    requested = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    return {
        "schema_version": 1,
        "suite": "ReCore",
        "out_dir": str(output_dir),
        "manifest": str(output_dir / "manifest.json"),
        "requested_parameters": requested,
        "status": "planned",
        "jobs": jobs,
    }


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _file_revision(path):
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _reject_nonfinite(value):
    raise ValueError(f"non-finite JSON value: {value}")


def _require_integer(payload, key, expected):
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ValueError(f"output {key} does not match the requested value {expected}")


def _require_mapping(payload, key):
    value = payload.get(key)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"output {key} must be a nonempty JSON object")
    return value


def _validate_summaries(summaries, expected_n):
    if not isinstance(summaries, dict) or not summaries:
        raise ValueError("output summaries must be a nonempty JSON object")
    for summary in summaries.values():
        if not isinstance(summary, dict):
            raise ValueError("each output summary must be a JSON object")
        _require_integer(summary, "n", expected_n)


def _validate_multiapi_summaries(summaries, expected_n):
    if not isinstance(summaries, dict) or not summaries:
        raise ValueError("multiapi summaries must be a nonempty JSON object")
    for summary in summaries.values():
        if not isinstance(summary, dict):
            raise ValueError("each multiapi summary must be a JSON object")
        _require_integer(summary, "n_attempted", expected_n)
        n_valid = summary.get("n_valid")
        if (isinstance(n_valid, bool) or not isinstance(n_valid, int)
                or not 0 <= n_valid <= expected_n):
            raise ValueError("multiapi valid-response count is outside the attempted count")
        _require_integer(summary, "n", n_valid)
        for key in ("n_invalid", "n_unverified", "n_api_errors", "n_token_measured"):
            if key not in summary:
                continue
            count = summary[key]
            if (isinstance(count, bool) or not isinstance(count, int)
                    or not 0 <= count <= expected_n):
                raise ValueError(f"multiapi {key} is outside the attempted count")
        if all(key in summary for key in ("n_invalid", "n_unverified")):
            if n_valid + summary["n_invalid"] + summary["n_unverified"] != expected_n:
                raise ValueError("multiapi response counts do not sum to attempted responses")
        else:
            if n_valid + summary.get("n_invalid", 0) + summary.get("n_unverified", 0) > expected_n:
                raise ValueError("multiapi response counts exceed attempted responses")
        if ("n_invalid" in summary
                and summary.get("n_api_errors", 0) > summary["n_invalid"]):
            raise ValueError("multiapi API errors exceed invalid responses")
        if summary.get("recall_denominator", "valid_response") != "valid_response":
            raise ValueError("multiapi recall denominator must be valid_response")


def _validate_payload(job, payload):
    if not isinstance(payload, dict):
        raise ValueError("experiment output must be a JSON object")
    parameters = job["effective_parameters"]
    experiment = job["experiment"]
    _require_integer(payload, "seed", parameters["seed"])
    expected_n = parameters.get("n", parameters.get("n_agent", 0) + parameters.get("n_gwm", 0))
    _require_integer(payload, "n", expected_n)
    if experiment in ("main", "ablation", "dataset_stats"):
        _require_integer(payload, "H_max", parameters["H_max"])
    elif "H_max" in payload and "H_max" in parameters:
        _require_integer(payload, "H_max", parameters["H_max"])
    if experiment == "full_baselines":
        for key in ("n_agent", "n_gwm"):
            _require_integer(payload, key, parameters[key])
    if experiment in ("main", "ablation", "full_baselines", "llm"):
        _validate_summaries(payload.get("summaries"), expected_n)
    elif experiment == "budget":
        results = _require_mapping(payload, "results")
        for summaries in results.values():
            _validate_summaries(summaries, expected_n)
    elif experiment in ("cascade", "topology"):
        results = _require_mapping(payload, "results")
        for result in results.values():
            if not isinstance(result, dict):
                raise ValueError("each sweep result must be a JSON object")
            _validate_summaries(result.get("summaries"), expected_n)
    elif experiment == "dataset_stats":
        _require_mapping(payload, "statistics")
    if experiment == "llm":
        if payload.get("seeds") != [parameters["seed"]]:
            raise ValueError("LLM output seeds do not match the requested seed")
        model = payload.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError("LLM output must identify its actual model")
        if parameters["model"] is not None and model != parameters["model"]:
            raise ValueError("LLM output model does not match the requested model")
        metadata = _require_mapping(payload, "run_metadata")
        if metadata.get("model") != model:
            raise ValueError("LLM output model and run_metadata disagree")
    if experiment == "multiapi":
        expected_models = (set(("gpt-4o-mini", "gpt-4o", "gemini-2.5-flash"))
                           if parameters["models"] == "all"
                           else set(parameters["models"].split(",")))
        models = payload.get("models")
        if (not isinstance(models, list) or len(models) != len(expected_models)
                or any(not isinstance(model, str) for model in models)
                or set(models) != expected_models):
            raise ValueError("multiapi output models do not match the requested models")
        results = _require_mapping(payload, "results")
        if set(results) != expected_models:
            raise ValueError("multiapi output is missing model results")
        for summaries in results.values():
            _validate_multiapi_summaries(summaries, expected_n)
    if experiment in ("full_baselines", "dataset_stats", "llm", "multiapi"):
        rows = payload.get("per_instance")
        expected_rows = expected_n * (len(payload["models"]) if experiment == "multiapi" else 1)
        if not isinstance(rows, list) or len(rows) != expected_rows:
            raise ValueError("per-instance output count does not match the requested experiment")
        seen = set()
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("instance_id"), str):
                raise ValueError("per-instance output must include instance IDs")
            identity = row["instance_id"]
            if experiment in API_EXPERIMENTS:
                _require_integer(row, "seed", parameters["seed"])
            if experiment == "multiapi":
                if row.get("model") not in expected_models:
                    raise ValueError("per-instance model does not match the requested models")
                identity = (row["model"], identity)
            if identity in seen:
                raise ValueError("per-instance output contains duplicate records")
            seen.add(identity)


def _write_manifest(plan):
    destination = Path(plan["manifest"])
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                     dir=destination.parent,
                                     prefix=".manifest-", suffix=".json",
                                     delete=False) as handle:
        temporary_path = Path(handle.name)
        try:
            json.dump(plan, handle, indent=2, allow_nan=False)
            handle.write("\n")
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    try:
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate_destinations(plan, overwrite):
    outputs = [Path(plan["manifest"])]
    outputs.extend(Path(item) for job in plan["jobs"] for item in job["artifacts"])
    if len(outputs) != len(set(outputs)):
        raise ValueError("output paths must be unique")
    for output in outputs:
        if output.is_symlink() or output.is_dir():
            raise ValueError(f"refusing non-regular output target: {output}")
        if output.exists() and not overwrite:
            raise FileExistsError(f"output already exists; use --overwrite: {output}")
        for parent in output.parents:
            if parent.exists() and not parent.is_dir():
                raise ValueError(f"output parent is not a directory: {parent}")
    for job in plan["jobs"]:
        if not Path(job["command"][1]).is_file():
            raise FileNotFoundError(f"experiment entry does not exist: {job['command'][1]}")


def execute_plan(plan, overwrite=False):
    _validate_destinations(plan, overwrite)
    Path(plan["out_dir"]).mkdir(parents=True, exist_ok=True)
    plan["status"] = "running"
    plan["started_at"] = _timestamp()
    _write_manifest(plan)
    exit_code = 0
    for index, job in enumerate(plan["jobs"]):
        job["status"] = "running"
        job["started_at"] = _timestamp()
        _write_manifest(plan)
        try:
            output = Path(job["output"])
            output.parent.mkdir(parents=True, exist_ok=True)
            prior_revision = _file_revision(output)
            completed = subprocess.run(job["command"], cwd=str(ROOT),
                                       check=False, shell=False)
            job["returncode"] = completed.returncode
            if completed.returncode:
                raise RuntimeError(f"child process exited with status {completed.returncode}")
            with output.open(encoding="utf-8") as handle:
                payload = json.load(handle, parse_constant=_reject_nonfinite)
            if prior_revision is not None and _file_revision(output) == prior_revision:
                raise RuntimeError("child process did not refresh the existing output")
            _validate_payload(job, payload)
            for artifact in job["artifacts"]:
                if not Path(artifact).is_file():
                    raise FileNotFoundError(f"expected output artifact is missing: {artifact}")
            job["status"] = "completed"
        except KeyboardInterrupt:
            job["status"] = "interrupted"
            job["error_type"] = "KeyboardInterrupt"
            plan["status"] = "interrupted"
            exit_code = 130
        except Exception as exc:
            job["status"] = "failed"
            job["error_type"] = type(exc).__name__
            job["error"] = str(exc)
            plan["status"] = "failed"
            exit_code = 1
        job["finished_at"] = _timestamp()
        if exit_code:
            for pending in plan["jobs"][index + 1:]:
                pending["status"] = "not_run"
            break
        _write_manifest(plan)
    if not exit_code:
        plan["status"] = "completed"
    plan["finished_at"] = _timestamp()
    _write_manifest(plan)
    return exit_code


def main(argv=None):
    parser = argument_parser()
    args = parser.parse_args(argv)
    try:
        plan = build_plan(args)
        if args.dry_run:
            print(json.dumps(plan, indent=2, allow_nan=False))
            return 0
        return execute_plan(plan, overwrite=args.overwrite)
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
