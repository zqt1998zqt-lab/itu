#!/usr/bin/env python3
"""Shard frozen score-only runs by case, then merge through frozen helpers."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

from evaluation.common import (
    ASSETS_DIR,
    DEPS_DIR,
    SCORE_RUNNER,
    STAGE_TOOL,
    atomic_json,
    evaluation_env,
    require_file,
    run_managed_process,
    safe_name,
    sha256_file,
    utc_now,
)


SCHEMA_VERSION = 1


class ScoreParallelError(RuntimeError):
    """A shard or merge contract failed."""


def load_frozen_module(path: Path) -> Any:
    path = require_file(path, "frozen run_and_score.py")
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location("frozen_run_and_score", path)
    if spec is None or spec.loader is None:
        raise ScoreParallelError(f"cannot import frozen scorer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_runs(path: Path) -> dict[str, Any]:
    document = json.loads(require_file(path, "score input manifest").read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("runs"), list) or not document["runs"]:
        raise ScoreParallelError(f"score input must contain a non-empty runs list: {path}")
    return document


def shard_manifest(document: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, Any]:
    value = copy.deepcopy(document)
    value["runs"] = copy.deepcopy(runs)
    return value


def score_command(
    *,
    run_and_score: Path,
    input_manifest: Path,
    output_dir: Path,
    stage_tool: Path,
    assets_dir: Path,
    metric_stages: Sequence[str],
    baseline_variant: str,
) -> list[str]:
    return [
        sys.executable,
        str(run_and_score),
        "--score-only",
        str(input_manifest),
        "--output-dir",
        str(output_dir),
        "--assets-dir",
        str(assets_dir),
        "--stage-tool",
        str(stage_tool),
        "--metric-stages",
        *metric_stages,
        "--bootstrap-samples",
        "2",
        "--baseline-variant",
        baseline_variant,
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--baseline-variant", default="adaptive")
    parser.add_argument("--metric-stages", nargs="+", default=["tx_ns", "tx_output"])
    parser.add_argument("--assets-dir", type=Path, default=ASSETS_DIR)
    parser.add_argument("--stage-tool", type=Path, default=STAGE_TOOL)
    parser.add_argument("--run-and-score", type=Path, default=SCORE_RUNNER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.shards < 1 or args.workers < 1:
        parser.error("--shards and --workers must be >= 1")
    if args.bootstrap_samples < 2:
        parser.error("--bootstrap-samples must be >= 2")
    input_manifest = require_file(args.input_manifest, "score input manifest").resolve()
    run_and_score = require_file(args.run_and_score, "frozen run_and_score.py").resolve()
    stage_tool = require_file(args.stage_tool, "frozen stage tool").resolve()
    assets_dir = args.assets_dir.expanduser().resolve()
    if not assets_dir.is_dir():
        parser.error(f"--assets-dir is not a directory: {assets_dir}")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    document = load_runs(input_manifest)
    runs = document["runs"]
    shard_groups: list[list[dict[str, Any]]] = [[] for _ in range(args.shards)]
    for index, run in enumerate(runs):
        shard_groups[index % args.shards].append(run)

    status_path = output_dir / "score-parallel.json"
    status: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "started_utc": utc_now(),
        "input_manifest": str(input_manifest),
        "input_manifest_sha256": sha256_file(input_manifest),
        "output_dir": str(output_dir),
        "runs": len(runs),
        "shards": args.shards,
        "workers": args.workers,
        "stage_tool": str(stage_tool),
        "stage_tool_sha256": sha256_file(stage_tool),
        "processes": [],
    }
    atomic_json(status_path, status)

    shard_root = output_dir / "score-shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    tasks: list[tuple[int, Path, Path, list[str]]] = []
    for index, shard_runs in enumerate(shard_groups):
        if not shard_runs:
            continue
        shard_dir = shard_root / f"shard-{index:02d}"
        shard_dir.mkdir(parents=True, exist_ok=False)
        shard_input = shard_dir / "input.json"
        atomic_json(shard_input, shard_manifest(document, shard_runs))
        scored_dir = shard_dir / "scored"
        command = score_command(
            run_and_score=run_and_score,
            input_manifest=shard_input,
            output_dir=scored_dir,
            stage_tool=stage_tool,
            assets_dir=assets_dir,
            metric_stages=args.metric_stages,
            baseline_variant=args.baseline_variant,
        )
        tasks.append((index, shard_input, scored_dir, command))

    env = evaluation_env()
    process_records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        futures = {
            pool.submit(
                run_managed_process,
                name=f"score-shard-{index:02d}",
                command=command,
                step_dir=shard_root / f"shard-{index:02d}/steps",
                env=env,
            ): (index, shard_input, scored_dir, command)
            for index, shard_input, scored_dir, command in tasks
        }
        for future in as_completed(futures):
            index, _, _, _ = futures[future]
            try:
                record = future.result()
            except Exception as error:
                record = {
                    "name": f"score-shard-{index:02d}",
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "finished_utc": utc_now(),
                }
            process_records.append(record)
            status["processes"] = sorted(process_records, key=lambda item: item["name"])
            atomic_json(status_path, status)
            print(f"SHARD {index:02d} {record['status']}", flush=True)

    failures = [record for record in process_records if record.get("status") != "ok"]
    if failures:
        status.update(
            {
                "status": "failed",
                "finished_utc": utc_now(),
                "failures": failures,
            }
        )
        atomic_json(status_path, status)
        return 1

    frozen = load_frozen_module(run_and_score)
    merged_runs: list[dict[str, Any]] = []
    for index, _, scored_dir, _ in tasks:
        manifest_path = require_file(scored_dir / "run_manifest.json", f"shard {index} manifest")
        shard_manifest_document = json.loads(manifest_path.read_text(encoding="utf-8"))
        if shard_manifest_document.get("status") != "complete" or int(shard_manifest_document.get("failed_runs", 0)) != 0:
            raise ScoreParallelError(f"shard {index} did not complete: {manifest_path}")
        merged_runs.extend(shard_manifest_document.get("runs", []))
    if len(merged_runs) != len(runs):
        raise ScoreParallelError(f"merged runs {len(merged_runs)} != input runs {len(runs)}")
    merged_runs.sort(key=lambda run: (str(run.get("case_id")), str(run.get("variant"))))

    # Rehome only the small record path; audio remains in immutable shard dirs.
    for record in merged_runs:
        case = record.get("case")
        if not isinstance(case, dict):
            raise ScoreParallelError("merged run lacks embedded case metadata")
        run_dir = output_dir / "cases" / safe_name(str(case["id"])) / "variants" / safe_name(str(record["variant"]))
        run_dir.mkdir(parents=True, exist_ok=True)
        record["run_dir"] = str(run_dir)

    frozen.write_case_outputs(output_dir, merged_runs)
    summary, case_deltas = frozen.summary_rows(
        merged_runs, args.bootstrap_samples, args.baseline_variant
    )
    frozen.write_csv(output_dir / "summary.csv", summary)
    frozen.write_csv(output_dir / "case_deltas.csv", case_deltas)
    frozen.atomic_json(
        output_dir / "summary.json", {"summary": summary, "case_deltas": case_deltas}
    )
    for record in merged_runs:
        frozen.atomic_json(Path(record["run_dir"]) / "engine_record.json", record)
    failures_in_runs = [record for record in merged_runs if record.get("status") != "ok"]
    final_status = "complete" if not failures_in_runs else "partial"
    manifest = frozen.manifest_document(
        final_status, None, None, stage_tool, assets_dir, merged_runs, True
    )
    manifest["failed_runs"] = len(failures_in_runs)
    manifest["baseline_variant"] = args.baseline_variant
    frozen.atomic_json(output_dir / "run_manifest.json", manifest)

    status.update(
        {
            "status": "complete" if not failures_in_runs else "partial",
            "finished_utc": utc_now(),
            "merged_runs": len(merged_runs),
            "failed_runs": len(failures_in_runs),
            "final_manifest": str(output_dir / "run_manifest.json"),
            "final_manifest_sha256": sha256_file(output_dir / "run_manifest.json"),
            "summary": str(output_dir / "summary.csv"),
        }
    )
    atomic_json(status_path, status)
    print(json.dumps({"status": status["status"], "runs": len(merged_runs), "failed": len(failures_in_runs)}))
    return 0 if not failures_in_runs else 1


if __name__ == "__main__":
    raise SystemExit(main())

