#!/usr/bin/env python3
"""Parallel hybrid-v2 evaluation suite over frozen VCP groups."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Sequence

from evaluation.common import (
    ASSETS_DIR,
    BUILD_ROOT,
    DEPS_DIR,
    FROZEN_CLI,
    MAKE_SCORE_MANIFEST,
    PROJECT_ROOT,
    SCORE_RUNNER,
    STAGE_TOOL,
    append_text,
    atomic_json,
    evaluation_env,
    require_executable,
    require_file,
    run_managed_process,
    safe_name,
    sha256_file,
    utc_now,
)


SCHEMA_VERSION = 1
SCRIPT_DIR = Path(__file__).resolve().parent
RUN_CANDIDATE = SCRIPT_DIR / "run_candidate.py"
SCORE_PARALLEL = SCRIPT_DIR / "score_parallel.py"
NEW_PROBE = SCRIPT_DIR / "native/hybrid_v2_probe"
DPDF_RUN_EXTERNAL = PROJECT_ROOT / "benchmarks/ns/dpdfnet2-16k-v1/frozen/run_external.py"
PREVIOUS_STAT21 = PROJECT_ROOT / "build/ns-distilled-c64-20261116/evaluation/stat21-frozen"

V1_GROUPS = {
    "v1speech48": (PROJECT_ROOT / "benchmarks/ns/dpdfnet2-16k-v1/cases/cases-speech48.json", 48),
    "v1clean16": (PROJECT_ROOT / "benchmarks/ns/dpdfnet2-16k-v1/cases/cases-clean16.json", 16),
    "v1continuous4": (PROJECT_ROOT / "benchmarks/ns/dpdfnet2-16k-v1/cases/cases-continuous4.json", 4),
    "v1guards": (PROJECT_ROOT / "benchmarks/ns/dpdfnet2-16k-v1/cases/cases-guards.json", 6),
}
HIGHWAY_GROUPS = {
    "highway120-speech48": (PROJECT_ROOT / "build/ns-highway120-20261115/data/cases-speech48.json", 48),
    "highway120-continuous4": (PROJECT_ROOT / "build/ns-highway120-20261115/data/cases-continuous4.json", 4),
    "highway120-noise2": (PROJECT_ROOT / "build/ns-highway120-20261115/data/cases-noise2.json", 2),
}
GROUP_ORDER = (*V1_GROUPS, *HIGHWAY_GROUPS, "newtest48")
NEWTEST_CANDIDATES = (
    SCRIPT_DIR / "data/newtest16/cases.json",
    SCRIPT_DIR / "data/newtest16/cases-newtest16.json",
    SCRIPT_DIR / "data/newtest16/manifest.json",
    SCRIPT_DIR / "data/cases-newtest16.json",
)


class SuiteError(RuntimeError):
    """An evaluation contract or process failed."""


def load_json(path: Path) -> Any:
    return json.loads(require_file(path, "JSON file").read_text(encoding="utf-8"))


def load_list(path: Path, key: str) -> list[dict[str, Any]]:
    document = load_json(path)
    rows = document.get(key, document) if isinstance(document, dict) else document
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise SuiteError(f"{path} must contain a non-empty {key} list")
    return rows


def is_pure_noise(row: dict[str, Any]) -> bool:
    for key in ("pure_noise", "is_pure_noise"):
        if isinstance(row.get(key), bool):
            return bool(row[key])
    for key in ("speech_present", "has_speech"):
        if isinstance(row.get(key), bool):
            return not bool(row[key])
    kind = str(row.get("kind") or row.get("role") or "").strip().lower()
    return kind in {"noise", "noise_only", "pure_noise", "nonspeech", "non_speech", "interference_only"}


def resolve_path(value: Any, base: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def validate_group(path: Path, expected: int, group: str) -> list[dict[str, Any]]:
    path = require_file(path, f"cases manifest for {group}")
    rows = load_list(path, "cases")
    if len(rows) != expected:
        raise SuiteError(f"{group}: expected {expected} cases, found {len(rows)}")
    ids: set[str] = set()
    for index, row in enumerate(rows):
        case_id = str(row.get("id") or row.get("case_id") or "").strip()
        if not case_id or case_id in ids:
            raise SuiteError(f"{group}: missing/duplicate case id at index {index}: {case_id!r}")
        ids.add(case_id)
        mic = resolve_path(next((row[key] for key in ("mic_path", "mic", "input_path") if row.get(key)), None), path.parent)
        if mic is None:
            raise SuiteError(f"{group}/{case_id}: missing mic_path")
        require_file(mic, f"{group}/{case_id}.mic_path")
        clean_value = next(
            (
                row[key]
                for key in ("clean_path", "clean", "reference_path", "input_clean_path")
                if row.get(key)
            ),
            None,
        )
        if clean_value:
            require_file(resolve_path(clean_value, path.parent), f"{group}/{case_id}.clean_path")
        elif not is_pure_noise(row):
            raise SuiteError(f"{group}/{case_id}: speech case lacks clean_path")
    return rows


def selected_groups(values: Sequence[str], newtest_cases: Path | None) -> tuple[list[str], dict[str, tuple[Path, int]]]:
    requested = list(values or GROUP_ORDER)
    expanded: list[str] = []
    for value in requested:
        if value == "newtest16":
            value = "newtest48"
        if value in {"all", "v1"}:
            expanded.extend(V1_GROUPS if value == "v1" else GROUP_ORDER)
        elif value == "highway120":
            expanded.extend(HIGHWAY_GROUPS)
        else:
            expanded.append(value)
    groups: list[str] = []
    for group in expanded:
        if group not in groups:
            groups.append(group)
    mapping = dict(V1_GROUPS)
    mapping.update(HIGHWAY_GROUPS)
    if "newtest48" in groups:
        if newtest_cases is not None:
            path = newtest_cases.expanduser().resolve()
        else:
            path = next((candidate for candidate in NEWTEST_CANDIDATES if candidate.is_file()), None)
        if path is None:
            raise SuiteError(
                "newtest48 is pending: dataworker manifest not found; pass --newtest-cases <absolute path>"
            )
        mapping["newtest48"] = (path, 48)
    unknown = [group for group in groups if group not in mapping]
    if unknown:
        raise SuiteError(f"unknown groups: {unknown}")
    return groups, mapping


def load_variants(path: Path) -> list[dict[str, Any]]:
    path = require_file(path, "variants config")
    rows = load_list(path, "variants")
    result: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, raw in enumerate(rows):
        name = str(raw.get("name") or raw.get("id") or "").strip()
        if not name or safe_name(name) != name or name in names:
            raise SuiteError(f"variants[{index}] has unsafe/missing/duplicate name: {name!r}")
        names.add(name)
        model = resolve_path(raw.get("model"), path.parent)
        if model is not None:
            require_file(model, f"variant {name}.model")
        probe = resolve_path(raw.get("probe"), path.parent) if raw.get("probe") else NEW_PROBE
        probe = require_executable(probe, f"variant {name}.probe")
        floor_db = float(raw.get("floor_db", -9.0))
        absolute_floor_db = float(raw.get("absolute_floor_db", -21.0))
        mix = float(raw.get("mix", 1.0))
        pitch = float(raw.get("pitch_strength", 0.0))
        if not all(math.isfinite(value) for value in (floor_db, absolute_floor_db, mix, pitch)):
            raise SuiteError(f"variant {name} has a non-finite numeric setting")
        if not 0.0 <= mix <= 1.0 or not 0.0 <= pitch <= 1.0:
            raise SuiteError(f"variant {name}: mix and pitch_strength must be in [0, 1]")
        result.append(
            {
                "name": name,
                "model": str(model) if model else None,
                "probe": str(probe),
                "floor_db": floor_db,
                "absolute_floor_db": absolute_floor_db,
                "mix": mix,
                "pitch_strength": pitch,
                "raw": copy.deepcopy(raw),
            }
        )
    by_name = {variant["name"]: variant for variant in result}
    for variant in result:
        same_as = variant["raw"].get("same_weights_as")
        if not same_as:
            continue
        if not variant["model"]:
            raise SuiteError(f"variant {variant['name']}.same_weights_as requires a model")
        target = by_name.get(str(same_as))
        if target is not None:
            if not target["model"]:
                raise SuiteError(f"same_weights_as target has no model: {same_as!r}")
            reference_sha = sha256_file(Path(target["model"]))
        else:
            reference = resolve_path(same_as, path.parent)
            if reference is None or not reference.is_file():
                raise SuiteError(
                    f"variant {variant['name']}.same_weights_as must name a config variant or model.json/weights file: {same_as!r}"
                )
            if reference.suffix == ".json":
                metadata = json.loads(reference.read_text(encoding="utf-8"))
                reference_sha = str(metadata.get("weights_sha256") or metadata.get("sha256") or "")
                if len(reference_sha) != 64:
                    raise SuiteError(f"same_weights_as sidecar lacks weights SHA: {reference}")
            else:
                reference_sha = sha256_file(reference)
        if sha256_file(Path(variant["model"])) != reference_sha:
            raise SuiteError(f"variant {variant['name']} weights differ from {same_as}")
    return result


def validate_candidate_receipt(path: Path, expected: int, variant: dict[str, Any]) -> dict[str, Any]:
    document = load_json(path)
    if document.get("status") != "complete" or int(document.get("failed", 0)) != 0:
        raise SuiteError(f"candidate failed: {path}")
    runs = document.get("runs")
    if not isinstance(runs, list) or len(runs) != expected:
        raise SuiteError(f"candidate coverage {len(runs) if isinstance(runs, list) else 'invalid'} != {expected}: {path}")
    for run in runs:
        if run.get("status") != "ok" or run.get("upstream_input_exact") is not True:
            raise SuiteError(f"candidate run failed/replay mismatch: {run.get('case', {}).get('id')}")
        if run.get("raw_after_subtype") != "FLOAT":
            raise SuiteError("candidate output is not FLOAT")
        samples = int(run.get("input_samples", -1))
        if int(run.get("raw_after_samples", -1)) != samples + 160:
            raise SuiteError("candidate output length is not input+160")
        contract = run.get("probe_conversion_contract", {})
        if contract.get("extra_lookahead_samples") != 0 or contract.get("drain_hops") != 12:
            raise SuiteError("candidate drain/lookahead contract changed")
        if contract.get("replay_payload_preserved_exact") is not True or contract.get("double_quantization") is not False:
            raise SuiteError("candidate replay payload was not preserved exactly")
        if run.get("identity_stable") is not True:
            raise SuiteError("candidate identity changed during run")
        if run.get("native_sha256_before") != run.get("native_sha256_after"):
            raise SuiteError("native probe changed during run")
        probe_values = run.get("probe_values", {})
        pitch = probe_values.get("pitch", {})
        pitch_applied = int(pitch.get("applied_frames", 0))
        if variant["model"]:
            if probe_values.get("modelstate") != "ready" or run.get("model_sha256_before") != run.get("model_sha256_after"):
                raise SuiteError("model candidate identity/state is invalid")
        elif probe_values.get("modelstate") != "absent":
            raise SuiteError("no-model candidate did not report absent state")
    return document


def pitch_aggregate(variants: list[dict[str, Any]], artifacts: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variant in variants:
        if variant["pitch_strength"] <= 0:
            continue
        voiced_cases = 0
        active_frames = 0
        pure_noise_active = 0
        for record in artifacts["variants"].get(variant["name"], {}).values():
            receipt_path = record.get("candidate_receipt")
            if not receipt_path or not Path(receipt_path).is_file():
                continue
            receipt = load_json(Path(receipt_path))
            for run in receipt.get("runs", []):
                pitch = run.get("probe_values", {}).get("pitch", {})
                applied = int(pitch.get("applied_frames", 0))
                if run.get("pure_noise"):
                    pure_noise_active += applied
                else:
                    voiced_cases += 1
                    active_frames += applied
        result[variant["name"]] = {
            "voiced_cases": voiced_cases,
            "active_frames": active_frames,
            "pure_noise_active_frames": pure_noise_active,
            "pass": voiced_cases == 0 or active_frames > 0,
        }
    return result


def validate_scored_manifest(path: Path, expected: int) -> dict[str, Any]:
    document = load_json(path)
    if document.get("status") != "complete" or int(document.get("failed_runs", 0)) != 0:
        raise SuiteError(f"scored manifest incomplete: {path}")
    if document.get("stage_tool", {}).get("sha256") != sha256_file(STAGE_TOOL):
        raise SuiteError(f"scored manifest used wrong stage tool: {path}")
    runs = document.get("runs")
    if not isinstance(runs, list) or len(runs) != expected:
        raise SuiteError(f"scored coverage mismatch: {path}")
    for run in runs:
        if run.get("status") != "ok":
            raise SuiteError(f"scored run failed: {run.get('case_id')}")
        for stage in run.get("scored_stages", []):
            row = next((item for item in stage.get("channels_scored", []) if item.get("channel") == 0), None)
            if row is None:
                raise SuiteError(f"stage {stage.get('id')} lacks channel 0")
            for prefix in ("raw", "minus30"):
                for metric in ("G", "S", "N"):
                    if f"{prefix}_{metric}" not in row:
                        raise SuiteError(f"stage {stage.get('id')} omitted {prefix}_{metric}")
    return document


def candidate_command(variant: dict[str, Any], cases: Path, output: Path, jobs: int, store: Path) -> list[str]:
    command = [
        sys.executable,
        str(RUN_CANDIDATE),
        "--cases",
        str(cases),
        "--output-dir",
        str(output),
        "--variant",
        variant["name"],
        "--probe",
        variant["probe"],
        "--floor-db",
        format(variant["floor_db"], ".12g"),
        "--absolute-floor-db",
        format(variant["absolute_floor_db"], ".12g"),
        "--mix",
        format(variant["mix"], ".12g"),
        "--pitch-strength",
        format(variant["pitch_strength"], ".12g"),
        "--jobs",
        str(jobs),
        "--hardlink-store",
        str(store),
    ]
    if variant["model"]:
        command += ["--model", variant["model"]]
    return command


def score_parallel_command(
    score_input: Path,
    scored_dir: Path,
    shards: int,
    workers: int,
    bootstrap_samples: int,
) -> list[str]:
    return [
        sys.executable,
        str(SCORE_PARALLEL),
        "--input-manifest",
        str(score_input),
        "--output-dir",
        str(scored_dir),
        "--shards",
        str(shards),
        "--workers",
        str(workers),
        "--bootstrap-samples",
        str(bootstrap_samples),
        "--assets-dir",
        str(ASSETS_DIR),
        "--stage-tool",
        str(STAGE_TOOL),
        "--run-and-score",
        str(SCORE_RUNNER),
    ]


def run_group(
    *,
    variant: dict[str, Any],
    group: str,
    cases: Path,
    expected: int,
    output_root: Path,
    native_jobs: int,
    score_shards: int,
    score_workers: int,
    bootstrap_samples: int,
    env: dict[str, str],
    resume: bool,
    hardlink_store: Path,
) -> dict[str, Any]:
    group_root = output_root / "variants" / safe_name(variant["name"]) / safe_name(group)
    final_manifest = group_root / "scored/run_manifest.json"
    if resume and final_manifest.is_file():
        validate_scored_manifest(final_manifest, expected)
        return {
            "status": "reused",
            "group_root": str(group_root),
            "candidate_receipt": str(group_root / "candidate/receipt.json"),
            "scored_manifest": str(final_manifest),
        }
    if group_root.exists():
        raise SuiteError(f"incomplete existing group output without --resume reuse: {group_root}")
    group_root.mkdir(parents=True, exist_ok=False)

    candidate_dir = group_root / "candidate"
    candidate_process = run_managed_process(
        name=f"candidate:{variant['name']}:{group}",
        command=candidate_command(variant, cases, candidate_dir, native_jobs, hardlink_store),
        step_dir=group_root / "steps/candidate",
        env=env,
    )
    if candidate_process["status"] != "ok":
        raise SuiteError(f"candidate process failed: {group_root}/steps/candidate")
    validate_candidate_receipt(candidate_dir / "receipt.json", expected, variant)

    score_input = group_root / "score-input.json"
    manifest_process = run_managed_process(
        name=f"manifest:{variant['name']}:{group}",
        command=[
            sys.executable,
            str(MAKE_SCORE_MANIFEST),
            "--input-dir",
            str(candidate_dir),
            "--output",
            str(score_input),
        ],
        step_dir=group_root / "steps/manifest",
        env=env,
    )
    if manifest_process["status"] != "ok":
        raise SuiteError(f"score manifest failed: {group_root}/steps/manifest")

    scored_dir = group_root / "scored"
    score_process = run_managed_process(
        name=f"score:{variant['name']}:{group}",
        command=score_parallel_command(
            score_input, scored_dir, score_shards, score_workers, bootstrap_samples
        ),
        step_dir=group_root / "steps/score-parallel",
        env=env,
    )
    if score_process["status"] != "ok":
        raise SuiteError(f"parallel scoring failed: {group_root}/steps/score-parallel")
    validate_scored_manifest(scored_dir / "run_manifest.json", expected)
    return {
        "status": "complete",
        "group_root": str(group_root),
        "candidate_receipt": str(candidate_dir / "receipt.json"),
        "score_input": str(score_input),
        "scored_manifest": str(scored_dir / "run_manifest.json"),
        "processes": {
            "candidate": candidate_process,
            "manifest": manifest_process,
            "score": score_process,
        },
    }


def run_dpdf_reference(
    *, cases: Path, output_root: Path, score_shards: int, score_workers: int,
    bootstrap_samples: int, env: dict[str, str], resume: bool
) -> dict[str, Any]:
    cases = Path(cases).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    root = output_root / "references/dpdfnet2/newtest48"
    final_manifest = root / "scored/run_manifest.json"
    if resume and final_manifest.is_file():
        validate_scored_manifest(final_manifest, 48)
        return {"status": "reused", "scored_manifest": str(final_manifest)}
    if root.exists():
        raise SuiteError(f"incomplete existing DPDF reference output: {root}")
    root.mkdir(parents=True, exist_ok=False)
    normalized_cases = root / "cases-normalized.json"
    rows = copy.deepcopy(load_list(cases, "cases"))
    for row in rows:
        if not row.get("clean_path") and row.get("input_clean_path"):
            row["clean_path"] = row["input_clean_path"]
    atomic_json(normalized_cases, {"cases": rows})
    raw_dir = root / "raw"
    external = run_managed_process(
        name="dpdfnet2:newtest48",
        command=[
            sys.executable,
            str(DPDF_RUN_EXTERNAL),
            "--cases",
            str(normalized_cases),
            "--output-dir",
            str(raw_dir),
            "--cli",
            str(FROZEN_CLI),
            "--skip-score",
            "--model",
            "dpdfnet2",
        ],
        step_dir=root / "steps/external",
        env=env,
    )
    if external["status"] != "ok":
        raise SuiteError(f"frozen DPDF external run failed: {root}/steps/external")
    receipt = load_json(raw_dir / "receipt.json")
    if int(receipt.get("failed", 1)) != 0 or len(receipt.get("runs", [])) != 48:
        raise SuiteError("frozen DPDF external coverage failed")
    score_input = root / "score-input.json"
    manifest = run_managed_process(
        name="dpdfnet2:manifest:newtest48",
        command=[
            sys.executable,
            str(MAKE_SCORE_MANIFEST),
            "--input-dir",
            str(raw_dir),
            "--output",
            str(score_input),
        ],
        step_dir=root / "steps/manifest",
        env=env,
    )
    if manifest["status"] != "ok":
        raise SuiteError("frozen DPDF score manifest failed")
    score = run_managed_process(
        name="dpdfnet2:score:newtest48",
        command=score_parallel_command(
            score_input, root / "scored", score_shards, score_workers, bootstrap_samples
        ),
        step_dir=root / "steps/score-parallel",
        env=env,
    )
    if score["status"] != "ok":
        raise SuiteError("frozen DPDF score failed")
    validate_scored_manifest(root / "scored/run_manifest.json", 48)
    return {
        "status": "complete",
        "raw_receipt": str(raw_dir / "receipt.json"),
        "score_input": str(score_input),
        "scored_manifest": str(root / "scored/run_manifest.json"),
        "processes": {"external": external, "manifest": manifest, "score": score},
    }


def coverage_document(
    *,
    version: str,
    output_root: Path,
    variants: list[dict[str, Any]],
    groups: list[str],
    group_mapping: dict[str, tuple[Path, int]],
    artifacts: dict[str, Any],
    reference: dict[str, Any] | None,
    status: str,
) -> dict[str, Any]:
    source_quality_path = SCRIPT_DIR / "data/source-quality.json"
    source_quality = None
    if source_quality_path.is_file():
        quality_document = load_json(source_quality_path)
        source_quality = {
            "path": str(source_quality_path),
            "sha256": sha256_file(source_quality_path),
            "non_denoised_reference": True,
            "label": quality_document.get("label") if isinstance(quality_document, dict) else None,
            "clean_judgment": quality_document.get("clean_judgment") if isinstance(quality_document, dict) else None,
            "note": "source-quality reference is the human original, not a denoised teacher or clean guarantee",
        }
    previous = PREVIOUS_STAT21 / "receipt.json"
    previous_stat21 = None
    if previous.is_file():
        previous_stat21 = {
            "status": "reuse_existing_no_rerun",
            "receipt": str(previous),
            "sha256": sha256_file(previous),
            "groups": ["v1speech48", "v1clean16", "v1continuous4", "v1guards", "highway120-speech48", "highway120-continuous4", "highway120-noise2"],
            "protocol": "base_floor_db=-21, absolute_floor_db=-21; not a base-9 2x2",
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": utc_now(),
        "status": status,
        "version": version,
        "output_root": str(output_root),
        "protocol": {
            "frozen_cli": str(FROZEN_CLI),
            "profile": "vcp_adaptive",
            "no_aec": True,
            "delay_samples": 160,
            "drain_hops": 12,
            "extra_lookahead_samples": 0,
            "scoring": "frozen run_and_score --score-only, sharded by case",
            "score_stage_tool": str(STAGE_TOOL),
            "all_stage_gsn": True,
            "pure_noise_gsn": None,
        },
        "frozen_sha256": {
            "cli": sha256_file(FROZEN_CLI),
            "run_and_score": sha256_file(SCORE_RUNNER),
            "stage_tool": sha256_file(STAGE_TOOL),
            "make_external_score_manifest": sha256_file(MAKE_SCORE_MANIFEST),
        },
        "groups": {
            group: {
                "manifest": str(group_mapping[group][0]),
                "manifest_sha256": sha256_file(group_mapping[group][0]),
                "expected_cases": group_mapping[group][1],
                "scope": "historical_regression" if group.startswith(("v1", "highway120")) else "independent_zh_test",
            }
            for group in groups
        },
        "variants": {
            variant["name"]: {
                "model": variant["model"],
                "probe": variant["probe"],
                "probe_sha256": sha256_file(Path(variant["probe"])),
                "floor_db": variant["floor_db"],
                "absolute_floor_db": variant["absolute_floor_db"],
                "mix": variant["mix"],
                "pitch_strength": variant["pitch_strength"],
            }
            for variant in variants
        },
        "artifacts": artifacts,
        "newtest48_dpdf_reference": reference or {
            "status": "pending",
            "requirement": "same tx_ns_input, frozen dpdfnet2-16k-v1, full teacher run",
            "runner": str(DPDF_RUN_EXTERNAL),
        },
        "source_quality_reference": source_quality,
        "previous_stat21": previous_stat21,
        "rules": {
            "v1_and_120": "historical regression only; do not compare aggregate G across datasets",
            "newtest48": "independent Common Voice zh extension: 16 speakers x SNR 0/10/20; separate from frozen v1/120",
            "newtest48_source_quality": "non-denoised human-original reference; materialized VAD/loudness normalization is not clean evidence",
            "stat21": "base21/final21 legacy baseline; base9 is only the no-model negative control",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--groups", nargs="+", default=list(GROUP_ORDER))
    parser.add_argument("--newtest-cases", type=Path)
    parser.add_argument("--with-dpdf-reference", action="store_true")
    parser.add_argument("--dpdf-reference-manifest", type=Path)
    parser.add_argument("--native-jobs", type=int, default=4)
    parser.add_argument("--group-workers", type=int, default=4)
    parser.add_argument("--score-shards", type=int, default=4)
    parser.add_argument("--score-workers", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--hardlink-store", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.native_jobs < 1 or args.group_workers < 1 or args.score_shards < 1 or args.score_workers < 1:
        parser.error("jobs/workers/shards must be >= 1")
    if args.bootstrap_samples < 2:
        parser.error("--bootstrap-samples must be >= 2")
    version = args.version.strip()
    if not version or safe_name(version) != version:
        parser.error("--version must be a safe path component")
    groups, group_mapping = selected_groups(args.groups, args.newtest_cases)
    for group in groups:
        validate_group(*group_mapping[group], group)
    variants = load_variants(args.configs)
    for path, label in (
        (RUN_CANDIDATE, "run_candidate.py"),
        (SCORE_PARALLEL, "score_parallel.py"),
        (FROZEN_CLI, "frozen CLI"),
        (SCORE_RUNNER, "frozen run_and_score.py"),
        (MAKE_SCORE_MANIFEST, "frozen score manifest builder"),
        (STAGE_TOOL, "frozen stage tool"),
    ):
        require_file(path, label)
    require_executable(FROZEN_CLI, "frozen CLI")
    if not ASSETS_DIR.is_dir():
        raise SuiteError(f"perceptual assets missing: {ASSETS_DIR}")
    if args.with_dpdf_reference and "newtest48" not in groups:
        parser.error("--with-dpdf-reference requires newtest48 in --groups")
    if args.with_dpdf_reference and args.dpdf_reference_manifest is not None:
        parser.error("choose --with-dpdf-reference or --dpdf-reference-manifest, not both")

    output_root = BUILD_ROOT / "evaluation" / version
    if output_root.exists() and not args.resume and not args.preflight_only:
        parser.error(f"output version already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    hardlink_store = (
        args.hardlink_store.expanduser().resolve()
        if args.hardlink_store
        else output_root / "hardlinks"
    )
    hardlink_store.mkdir(parents=True, exist_ok=True)
    timestamp = utc_now().replace(":", "").replace("+", "Z")
    log_path = BUILD_ROOT / "logs" / f"eval-{version}-{os.getpid()}-{timestamp}.log"
    runlog = output_root / "run.log"
    progress_path = output_root / "progress.json"
    progress_jsonl = output_root / "progress.jsonl"
    coverage_path = BUILD_ROOT / "reports/eval-coverage.json"
    lock = threading.Lock()
    artifacts: dict[str, Any] = {
        "variants": {variant["name"]: {} for variant in variants},
        "reference": None,
        "pitch_aggregate": {},
    }
    progress: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "preflight_ok" if args.preflight_only else "running",
        "started_utc": utc_now(),
        "version": version,
        "output_root": str(output_root),
        "configs": str(args.configs.resolve()),
        "variants": [variant["name"] for variant in variants],
        "groups": groups,
        "hardlink_store": str(hardlink_store),
        "artifacts": artifacts,
        "failures": [],
    }
    if args.dpdf_reference_manifest is not None:
        reference_manifest = require_file(args.dpdf_reference_manifest, "DPDF reference manifest").resolve()
        validate_scored_manifest(reference_manifest, 48)
        artifacts["reference"] = {
            "status": "reused",
            "scored_manifest": str(reference_manifest),
            "scored_manifest_sha256": sha256_file(reference_manifest),
        }

    def publish(status: str = progress["status"]) -> None:
        with lock:
            progress["status"] = status
            progress["updated_utc"] = utc_now()
            progress["artifacts"] = artifacts
            atomic_json(progress_path, progress)
            atomic_json(
                coverage_path,
                coverage_document(
                    version=version,
                    output_root=output_root,
                    variants=variants,
                    groups=groups,
                    group_mapping=group_mapping,
                    artifacts=artifacts,
                    reference=artifacts["reference"],
                    status=status,
                ),
            )

    append_text(runlog, f"[{progress['started_utc']}] preflight {version}\n")
    append_text(log_path, f"[{progress['started_utc']}] preflight {version}\n")
    publish(progress["status"])
    if args.preflight_only:
        progress["finished_utc"] = utc_now()
        publish("preflight_ok")
        print(json.dumps({"status": "preflight_ok", "output_root": str(output_root), "groups": len(groups), "variants": len(variants)}))
        return 0

    env = evaluation_env()
    failures: list[str] = []
    try:
        if args.with_dpdf_reference:
            reference = run_dpdf_reference(
                cases=group_mapping["newtest48"][0],
                output_root=output_root,
                score_shards=args.score_shards,
                score_workers=args.score_workers,
                bootstrap_samples=args.bootstrap_samples,
                env=env,
                resume=args.resume,
            )
            artifacts["reference"] = reference
            publish()

        for variant in variants:
            with ThreadPoolExecutor(max_workers=min(args.group_workers, len(groups))) as pool:
                futures = {
                    pool.submit(
                        run_group,
                        variant=variant,
                        group=group,
                        cases=group_mapping[group][0],
                        expected=group_mapping[group][1],
                        output_root=output_root,
                        native_jobs=args.native_jobs,
                        score_shards=args.score_shards,
                        score_workers=args.score_workers,
                        bootstrap_samples=args.bootstrap_samples,
                        env=env,
                        resume=args.resume,
                        hardlink_store=hardlink_store,
                    ): group
                    for group in groups
                }
                for future in as_completed(futures):
                    group = futures[future]
                    try:
                        record = future.result()
                    except Exception as error:
                        record = {"status": "failed", "error": f"{type(error).__name__}: {error}", "finished_utc": utc_now()}
                        failures.append(f"{variant['name']}/{group}: {error}")
                    artifacts["variants"][variant["name"]][group] = record
                    with progress_jsonl.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({"variant": variant["name"], "group": group, **record}, ensure_ascii=False) + "\n")
                    append_text(
                        runlog,
                        f"[{utc_now()}] {variant['name']}/{group} {record['status']}\n",
                    )
                    append_text(
                        log_path,
                        f"[{utc_now()}] {variant['name']}/{group} {record['status']}\n",
                    )
                    publish("running" if not failures else "failed")
            if failures:
                break
        artifacts["pitch_aggregate"] = pitch_aggregate(variants, artifacts)
        for variant_name, check in artifacts["pitch_aggregate"].items():
            if not check["pass"]:
                failures.append(
                    f"{variant_name}: pitch aggregate failed ({check['active_frames']} active / "
                    f"{check['voiced_cases']} voiced; pure-noise {check['pure_noise_active_frames']})"
                )
        progress["failures"] = failures
        progress["finished_utc"] = utc_now()
        final_status = "complete" if not failures else "failed"
        publish(final_status)
        append_text(log_path, f"[{progress['finished_utc']}] {final_status}\n")
        print(json.dumps({"status": final_status, "output_root": str(output_root), "failures": len(failures), "coverage": str(coverage_path)}))
        return 0 if not failures else 1
    except Exception as error:
        failures.append(f"{type(error).__name__}: {error}")
        progress["failures"] = failures
        progress["finished_utc"] = utc_now()
        publish("failed")
        append_text(log_path, f"[{progress['finished_utc']}] failed: {error}\n")
        print(json.dumps({"status": "failed", "error": str(error), "output_root": str(output_root)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

