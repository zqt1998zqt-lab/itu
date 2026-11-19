#!/usr/bin/env python3
"""Run the frozen 74-case NS quality regression on optimized CoFi-Lite.

This driver imports the retained 2,406-line stream verifier and overrides only
the candidate execution identity.  The optimized runtime has a fixed 480-sample
physical delay, while the frozen DPDFNet reference retains its original
640-sample algorithm delay.  Candidate audio is always scored afresh.

The legacy verifier and every existing quality artifact remain immutable.  All
new evidence is written exclusively below:

  build/cofilite-opt-20261116/quality74
"""

from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True

from pathlib import Path

PROJECT_ROOT = Path("/home/zhangqingtian/project/itu")
EXPECTED_OUTPUT_ROOT = PROJECT_ROOT / "build/cofilite-opt-20261116/quality74"
LEGACY_VALIDATOR = PROJECT_ROOT / "scripts/verify-cofilite-stream-quality.py"
RUNTIME_SOURCE = PROJECT_ROOT / "server/cofilite_stream.py"
RUNTIME_FRAME_SOURCE = PROJECT_ROOT / "server/cofilite_frame.py"
DPDF_MANIFEST = PROJECT_ROOT / "benchmarks/ns/dpdfnet2-16k-v1/manifest.json"

DPDF_REFERENCE_DELAY_SAMPLES = 640
EXPECTED_TOTAL = 74
EXPECTED_SPEECH = 68
EXPECTED_NOISE = 6
EXPECTED_GROUPS = {
    "v1speech48": 48,
    "v1clean16": 16,
    "v1continuous4": 4,
    "v1guards": 6,
}
PARITY_MAX_ABS_THRESHOLD = 1e-5

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ["NUMBA_CACHE_DIR"] = str(EXPECTED_OUTPUT_ROOT / "cache/numba")
os.environ["MPLCONFIGDIR"] = str(EXPECTED_OUTPUT_ROOT / "cache/matplotlib")
os.environ["XDG_CACHE_HOME"] = str(EXPECTED_OUTPUT_ROOT / "cache/xdg")

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import multiprocessing
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from types import ModuleType
from typing import Any, Iterator, Sequence

import numpy as np


_LEGACY: ModuleType | None = None
_RUNTIME: ModuleType | None = None
_ORIGINAL_DPDF_REFERENCE_FOR: Any = None
_ACTIVE_OUTPUT_ROOT = EXPECTED_OUTPUT_ROOT


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise ValueError(f"missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_module_file(name: str, path: Path) -> ModuleType:
    if not path.is_file():
        raise ValueError(f"missing module: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def legacy_module() -> ModuleType:
    global _LEGACY
    if _LEGACY is None:
        _LEGACY = load_module_file(
            "_itu_verify_cofilite_stream_quality_legacy",
            LEGACY_VALIDATOR,
        )
    return _LEGACY


def runtime_module() -> ModuleType:
    global _RUNTIME
    if _RUNTIME is None:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        import server.cofilite_stream as module

        _RUNTIME = module
    return _RUNTIME


def candidate_constants() -> dict[str, int]:
    runtime = runtime_module()
    values = {
        "sample_rate": int(runtime.SAMPLE_RATE),
        "frame_samples": int(runtime.FRAME_SAMPLES),
        "delay_samples": int(runtime.DELAY_SAMPLES),
        "hop_samples": int(runtime.HOP_SAMPLES),
        "fft_size": int(runtime.FFT_SIZE),
        "lookahead_samples": int(runtime.LOOKAHEAD_SAMPLES),
    }
    expected = {
        "sample_rate": 16000,
        "frame_samples": 160,
        "delay_samples": 480,
        "hop_samples": 256,
        "fft_size": 512,
        "lookahead_samples": 511,
    }
    if values != expected:
        raise RuntimeError(
            f"optimized runtime constants changed: expected {expected}, got {values}"
        )
    return values


def apply_legacy_candidate_constants() -> ModuleType:
    legacy = legacy_module()
    constants = candidate_constants()
    legacy.DEFAULT_OUTPUT_ROOT = str(EXPECTED_OUTPUT_ROOT)
    legacy.PHYSICAL_DELAY_SAMPLES = constants["delay_samples"]
    legacy.TRIM_SAMPLES = constants["delay_samples"]
    legacy.STREAM_CHUNK_SIZE = constants["frame_samples"]
    legacy.SAMPLE_RATE = constants["sample_rate"]
    legacy.STREAM_SOURCE = RUNTIME_SOURCE
    return legacy


@contextmanager
def temporary_physical_delay(
    legacy: ModuleType, delay_samples: int
) -> Iterator[None]:
    previous = int(legacy.PHYSICAL_DELAY_SAMPLES)
    legacy.PHYSICAL_DELAY_SAMPLES = int(delay_samples)
    try:
        yield
    finally:
        legacy.PHYSICAL_DELAY_SAMPLES = previous


def _dpdf_reference_for_with_frozen_delay(
    case: dict[str, Any], reference_data: dict[str, Any]
) -> dict[str, Any]:
    if _ORIGINAL_DPDF_REFERENCE_FOR is None:
        raise RuntimeError("legacy DPDF reference resolver was not initialized")
    legacy = apply_legacy_candidate_constants()
    with temporary_physical_delay(legacy, DPDF_REFERENCE_DELAY_SAMPLES):
        reference = _ORIGINAL_DPDF_REFERENCE_FOR(case, reference_data)
    if int(reference["fixed_delay_samples"]) != DPDF_REFERENCE_DELAY_SAMPLES:
        raise RuntimeError("DPDFNet reference delay is not the frozen 640 samples")
    return reference


def configure_legacy() -> ModuleType:
    global _ORIGINAL_DPDF_REFERENCE_FOR
    legacy = apply_legacy_candidate_constants()
    if _ORIGINAL_DPDF_REFERENCE_FOR is None:
        _ORIGINAL_DPDF_REFERENCE_FOR = legacy.dpdf_reference_for
    legacy.dpdf_reference_for = _dpdf_reference_for_with_frozen_delay
    return legacy


def load_frozen_reference_data() -> dict[str, Any]:
    legacy = configure_legacy()
    with temporary_physical_delay(legacy, DPDF_REFERENCE_DELAY_SAMPLES):
        reference_data = legacy.load_reference_data()
    contract = reference_data.get("contract")
    if not isinstance(contract, dict):
        raise RuntimeError("DPDFNet reference contract is missing")
    if int(contract.get("algorithm_delay_samples", -1)) != 640:
        raise RuntimeError("DPDFNet reference delay changed from 640 samples")
    if contract.get("fixed_delay_search") is not False:
        raise RuntimeError("DPDFNet reference unexpectedly enables delay search")
    return reference_data


def execution_source_identity() -> dict[str, Any]:
    paths = {
        "driver": Path(__file__).resolve(),
        "legacy_validator": LEGACY_VALIDATOR,
        "runtime_stream": RUNTIME_SOURCE,
        "runtime_frame": RUNTIME_FRAME_SOURCE,
        "frozen_adapter": PROJECT_ROOT
        / "build/ns-cofilite-20261115/cofi_infer.py",
        "frozen_model_source": PROJECT_ROOT
        / "build/ns-cofilite-20261115/upstream/model/cofilite.py",
        "frozen_checkpoint": PROJECT_ROOT
        / "build/ns-cofilite-20261115/upstream/checkpoint/model_on_dns3.tar",
    }
    files: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"required execution source is missing: {path}")
        files[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {
        "created_utc": utc_now(),
        "candidate_constants": candidate_constants(),
        "dpdf_reference_delay_samples": DPDF_REFERENCE_DELAY_SAMPLES,
        "files": files,
    }


def ensure_source_lock(output_root: Path, *, create: bool) -> dict[str, Any]:
    path = output_root / "RUNTIME-SOURCE-LOCK.json"
    current = execution_source_identity()
    if path.is_file():
        locked = load_json(path)
        if locked.get("files") != current.get("files"):
            raise RuntimeError(
                "optimized runtime source changed after the parity source lock"
            )
        if (
            locked.get("candidate_constants")
            != current.get("candidate_constants")
        ):
            raise RuntimeError("optimized runtime constants changed after source lock")
        return locked
    if not create:
        raise FileNotFoundError(f"missing runtime source lock: {path}")
    legacy = configure_legacy()
    legacy.write_json_exclusive(path, current)
    return current


def worker_initialize_optimized() -> None:
    import torch

    torch.set_num_threads(1)
    apply_legacy_candidate_constants()
    legacy_module().worker_initialize()


def process_one_stream_case_optimized(
    payload: dict[str, Any],
) -> dict[str, Any]:
    legacy = apply_legacy_candidate_constants()
    if getattr(legacy, "_WORKER_MODEL", None) is None:
        worker_initialize_optimized()
    record = legacy.process_one_stream_case(payload)
    delay = int(candidate_constants()["delay_samples"])
    record["stream_output"]["physical_delay_samples"] = delay
    record["stream_output"]["trim_samples"] = delay
    record["stream_output"]["score_fixed_delay_samples"] = 0
    record["stream_output"]["delay_note"] = (
        f"optimized runtime physical delay {delay} samples was drained and "
        "trimmed; the aligned FLOAT file is scored with fixed delay 0"
    )
    record["candidate_execution_identity"] = payload["source_lock"]
    return record


def summarize_parity(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    legacy = apply_legacy_candidate_constants()
    constants = candidate_constants()
    delay = constants["delay_samples"]
    if len(records) != EXPECTED_TOTAL:
        raise ValueError(
            f"parity result count {len(records)} != {EXPECTED_TOTAL}"
        )
    groups: dict[str, dict[str, Any]] = {}
    for group, expected in EXPECTED_GROUPS.items():
        subset = [row for row in records if row["group"] == group]
        if len(subset) != expected:
            raise ValueError(
                f"{group}: parity count {len(subset)} != {expected}"
            )
        groups[group] = {
            "cases": len(subset),
            "expected_cases": expected,
            "pass_count": sum(bool(row["parity"]["pass"]) for row in subset),
            "max_abs": max(float(row["parity"]["max_abs"]) for row in subset),
            "max_rmse": max(float(row["parity"]["rmse"]) for row in subset),
            "mean_rmse": float(
                np.mean([float(row["parity"]["rmse"]) for row in subset])
            ),
            "rtf_median": float(
                np.median([float(row["runtime"]["rtf"]) for row in subset])
            ),
            "rtf_p95": float(
                np.percentile(
                    [float(row["runtime"]["rtf"]) for row in subset], 95
                )
            ),
            "state_bytes_values": sorted(
                {int(row["runtime"]["state_bytes"]) for row in subset}
            ),
            "process_rss_delta_max_bytes": max(
                int(row["runtime"]["process_rss_delta_bytes"])
                for row in subset
            ),
            "process_rss_after_max_bytes": max(
                int(row["runtime"]["process_rss_after_bytes"])
                for row in subset
            ),
        }
    all_rtf = [float(row["runtime"]["rtf"]) for row in records]
    state_values = sorted(
        {int(row["runtime"]["state_bytes"]) for row in records}
    )
    all_physical_delay_matches = all(
        int(row["stream_output"]["physical_delay_samples"]) == delay
        for row in records
    )
    all_trim_matches = all(
        int(row["stream_output"]["trim_samples"]) == delay for row in records
    )
    all_score_delay_zero = all(
        int(row["stream_output"]["score_fixed_delay_samples"]) == 0
        for row in records
    )
    cached_samples_max = max(
        int(row["stream_info"]["cached_samples"]) for row in records
    )
    return {
        "cases": len(records),
        "expected_cases": EXPECTED_TOTAL,
        "speech_cases": sum(not bool(row["pure_noise"]) for row in records),
        "noise_cases": sum(bool(row["pure_noise"]) for row in records),
        "pass_count": sum(bool(row["parity"]["pass"]) for row in records),
        "all_pass": all(bool(row["parity"]["pass"]) for row in records),
        "threshold": PARITY_MAX_ABS_THRESHOLD,
        "max_abs": max(float(row["parity"]["max_abs"]) for row in records),
        "max_rmse": max(float(row["parity"]["rmse"]) for row in records),
        "mean_rmse": float(
            np.mean([float(row["parity"]["rmse"]) for row in records])
        ),
        "candidate_execution": {
            **constants,
            "physical_delay_samples": delay,
            "trim_samples": delay,
            "score_fixed_delay_samples": 0,
            "all_physical_delay_match": all_physical_delay_matches,
            "all_trim_match": all_trim_matches,
            "all_score_delay_zero": all_score_delay_zero,
        },
        "reference_execution": {
            "dpdfnet2_fixed_delay_samples": DPDF_REFERENCE_DELAY_SAMPLES,
            "dynamic_delay_search": False,
        },
        "state_bytes_values": state_values,
        "state_bytes_bounded_observation": (
            len(state_values) == 1
            and cached_samples_max <= delay + constants["fft_size"]
        ),
        "cached_samples_max": cached_samples_max,
        "rtf_median": float(np.median(all_rtf)),
        "rtf_p95": float(np.percentile(all_rtf, 95)),
        "process_rss_delta_max_bytes": max(
            int(row["runtime"]["process_rss_delta_bytes"]) for row in records
        ),
        "process_rss_after_max_bytes": max(
            int(row["runtime"]["process_rss_after_bytes"]) for row in records
        ),
        "worker_model_load_s": sorted(
            {
                float(row["runtime"]["model_load_s"])
                for row in records
                if row["runtime"]["model_load_s"] is not None
            }
        ),
        "frame_executors": sorted(
            {str(row["stream_info"].get("frame_executor")) for row in records}
        ),
        "frame_executor_trace_sec_values": sorted(
            {
                float(row["stream_info"].get("frame_executor_trace_sec"))
                for row in records
                if row["stream_info"].get("frame_executor_trace_sec") is not None
            }
        ),
        "groups": groups,
    }


def parity_phase(
    cases: Sequence[dict[str, Any]], output_root: Path, jobs: int
) -> int:
    legacy = configure_legacy()
    if (output_root / "parity-results.jsonl").exists():
        raise FileExistsError(
            f"refusing to overwrite {output_root / 'parity-results.jsonl'}"
        )
    source_lock = ensure_source_lock(output_root, create=True)
    stream_root = legacy.next_attempt_dir(output_root, "stream")
    log_path = output_root / "run.log"
    legacy.append_log(
        log_path,
        f"optimized parity start cases={len(cases)} jobs={jobs} "
        f"stream_root={stream_root}",
    )
    payloads: list[dict[str, Any]] = []
    for case in cases:
        case_dir = (
            stream_root / case["dataset"] / legacy.safe_name(case["case_id"])
        )
        case_dir.mkdir(parents=True, exist_ok=False)
        payloads.append(
            {
                **case,
                "stream_output_path": str(
                    case_dir / "stream-output.float.wav"
                ),
                "source_lock": source_lock,
            }
        )

    records: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=jobs,
        mp_context=multiprocessing.get_context("fork"),
        initializer=worker_initialize_optimized,
    ) as pool:
        futures = {
            pool.submit(process_one_stream_case_optimized, payload): payload
            for payload in payloads
        }
        for future in concurrent.futures.as_completed(futures):
            payload = futures[future]
            record = future.result()
            records.append(record)
            legacy.append_log(
                log_path,
                "optimized parity "
                f"{record['group']}/{record['case_id']} "
                f"max_abs={record['parity']['max_abs']:.12g} "
                f"rmse={record['parity']['rmse']:.12g} "
                f"rtf={record['runtime']['rtf']:.9g} "
                f"state_bytes={record['runtime']['state_bytes']} "
                f"executor={record['stream_info'].get('frame_executor')}",
            )
            print(
                "PARITY "
                f"{record['group']}/{record['case_id']} "
                f"max_abs={record['parity']['max_abs']:.12g} "
                f"rmse={record['parity']['rmse']:.12g}",
                flush=True,
            )

    ensure_source_lock(output_root, create=False)
    records.sort(key=lambda row: (row["group"], row["case_id"]))
    legacy.write_jsonl_exclusive(
        output_root / "parity-results.jsonl", records
    )
    summary = summarize_parity(records)
    legacy.write_json_exclusive(
        output_root / "parity-summary.json", summary
    )
    legacy.append_log(
        log_path,
        "optimized parity complete "
        f"pass={summary['pass_count']}/{summary['cases']} "
        f"max_abs={summary['max_abs']:.12g} "
        f"max_rmse={summary['max_rmse']:.12g}",
    )
    print(
        json.dumps(
            legacy.json_safe(summary),
            sort_keys=True,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if summary["all_pass"] else 2


def validate_parity(
    output_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    legacy = configure_legacy()
    parity_path = output_root / "parity-results.jsonl"
    summary_path = output_root / "parity-summary.json"
    records = legacy.read_jsonl(parity_path)
    summary = load_json(summary_path)
    expected = summarize_parity(records)
    if summary.get("candidate_execution") != expected["candidate_execution"]:
        raise RuntimeError("parity summary candidate execution fields are stale")
    if summary.get("reference_execution") != expected["reference_execution"]:
        raise RuntimeError("parity summary DPDF reference fields are stale")
    if not expected["all_pass"]:
        raise RuntimeError(
            "optimized parity failed; scoring is not allowed because "
            f"max_abs={expected['max_abs']:.12g} must be < "
            f"{PARITY_MAX_ABS_THRESHOLD:g}"
        )
    candidate = expected["candidate_execution"]
    if not (
        candidate["all_physical_delay_match"]
        and candidate["all_trim_match"]
        and candidate["all_score_delay_zero"]
    ):
        raise RuntimeError("candidate delay/trim/score-delay contract failed")
    if (
        expected["reference_execution"]["dpdfnet2_fixed_delay_samples"]
        != DPDF_REFERENCE_DELAY_SAMPLES
    ):
        raise RuntimeError("DPDFNet reference delay contract failed")
    return records, expected


def score_one_case_optimized(
    case: dict[str, Any],
    parity_record: dict[str, Any],
    stage_tool: ModuleType,
    score_module: ModuleType,
    cache: Any,
    metric_cache: dict[str, Any],
    reference_data: dict[str, Any],
    c4eb_data: dict[str, Any],
    source_lock: dict[str, Any],
) -> dict[str, Any]:
    legacy = configure_legacy()
    record = legacy.score_one_case(
        case,
        parity_record,
        stage_tool,
        score_module,
        cache,
        metric_cache,
        reference_data,
        c4eb_data,
    )
    delay = candidate_constants()["delay_samples"]
    record["stream_output"]["physical_delay_samples"] = delay
    record["stream_output"]["trim_samples"] = delay
    record["stream_output"]["score_fixed_delay_samples"] = 0
    record["stream_output"]["delay_note"] = (
        f"optimized runtime physical delay {delay} samples was drained and "
        "trimmed; the aligned FLOAT file is scored with fixed delay 0"
    )
    record["scoring"]["candidate_physical_delay_samples"] = delay
    record["scoring"]["candidate_trim_samples"] = delay
    record["scoring"]["candidate_stage"]["measurement_domain"] = (
        f"optimized CoFi-Lite stateful stream output; physical delay {delay} "
        "samples already drained and trimmed; score delay 0"
    )
    if (
        int(record["dpdfnet2_reference"]["fixed_delay_samples"])
        != DPDF_REFERENCE_DELAY_SAMPLES
    ):
        raise RuntimeError("scored record changed the DPDF reference delay")
    record["candidate_execution_identity"] = source_lock
    return record


def load_scoring_runtime(
    output_root: Path,
) -> tuple[ModuleType, ModuleType, Any, dict[str, Any]]:
    legacy = configure_legacy()
    if sha256_file(legacy.STAGE_TOOL) != legacy.EXPECTED_STAGE_TOOL_SHA256:
        raise RuntimeError("frozen stage tool SHA changed")
    if sha256_file(legacy.RUN_AND_SCORE) != legacy.EXPECTED_RUN_AND_SCORE_SHA256:
        raise RuntimeError("frozen run_and_score SHA changed")
    if sha256_file(legacy.DNSMOS_MODEL) != legacy.EXPECTED_DNSMOS_SHA256:
        raise RuntimeError("DNSMOS ONNX SHA changed")

    reference_data = load_frozen_reference_data()
    c4eb_data = legacy.load_c4eb_references()
    stage_tool = legacy.load_module("vcp_stage_mos", legacy.STAGE_TOOL)
    score_module = legacy.load_module("run_and_score", legacy.RUN_AND_SCORE)
    if getattr(stage_tool, "MODEL_SHA", None) != legacy.EXPECTED_DNSMOS_SHA256:
        raise RuntimeError("stage tool DNSMOS model identity changed")

    scoring_root = legacy.next_attempt_dir(output_root, "scoring")
    log_path = output_root / "run.log"
    legacy.append_log(
        log_path, f"optimized scoring start scoring_root={scoring_root}"
    )
    scorer = stage_tool.DNSMOS(
        legacy.ASSETS_DIR, scoring_root / "cache/dnsmos"
    )
    try:
        pyloudnorm = stage_tool.import_pyloudnorm()
    except RuntimeError as error:
        cause = error.__cause__
        if cause is None or type(cause).__name__ != "PackageNotFoundError":
            raise
        deps_dir = legacy.PROJECT_ROOT + "/build/ns-next-20261115/deps"
        if deps_dir not in sys.path:
            sys.path.insert(0, deps_dir)
        import pyloudnorm

        source_dir = Path(pyloudnorm.__file__).resolve().parent
        source_hashes = {
            name: sha256_file(source_dir / name)
            for name in score_module.PYLOUDNORM_011_SHA256
        }
        if source_hashes != score_module.PYLOUDNORM_011_SHA256:
            raise RuntimeError(
                "unpacked pyloudnorm source is not the frozen 0.1.1 source"
            ) from error

    score_module.atomic_json = lambda *args, **kwargs: None
    cache = score_module.ScoreCache(
        scoring_root / "cache-store",
        stage_tool,
        scorer,
        pyloudnorm,
    )
    return stage_tool, score_module, cache, {
        "reference_data": reference_data,
        "c4eb_data": c4eb_data,
        "scoring_root": scoring_root,
        "log_path": log_path,
    }


def score_phase(
    cases: Sequence[dict[str, Any]], output_root: Path
) -> int:
    legacy = configure_legacy()
    source_lock = ensure_source_lock(output_root, create=False)
    parity_records, parity_summary = validate_parity(output_root)
    for path in (
        output_root / "case-results.jsonl",
        output_root / "summary.json",
        output_root / "REPORT-opt.md",
        output_root / "source-identities.json",
        output_root / "FINAL-VALIDATION.json",
    ):
        if path.exists():
            raise FileExistsError(
                f"refusing to overwrite optimized scoring output: {path}"
            )

    stage_tool, score_module, cache, state = load_scoring_runtime(output_root)
    scoring_root = state["scoring_root"]
    log_path = state["log_path"]
    reference_data = state["reference_data"]
    c4eb_data = state["c4eb_data"]
    metric_cache: dict[str, Any] = {}
    parity_by_key = {
        (row["dataset"], row["case_id"]): row for row in parity_records
    }
    if len(parity_by_key) != EXPECTED_TOTAL:
        raise RuntimeError("parity result keys are not unique")

    scored_records: list[dict[str, Any]] = []
    for index, case in enumerate(cases, 1):
        key = (case["dataset"], case["case_id"])
        parity_record = parity_by_key.get(key)
        if parity_record is None:
            raise RuntimeError(f"missing parity record for {key}")
        record = score_one_case_optimized(
            case,
            parity_record,
            stage_tool,
            score_module,
            cache,
            metric_cache,
            reference_data,
            c4eb_data,
            source_lock,
        )
        scored_records.append(record)
        legacy.append_log(
            log_path,
            f"optimized score {index}/{len(cases)} "
            f"{case['group']}/{case['case_id']}",
        )
        print(
            f"SCORE {index}/{len(cases)} "
            f"{case['group']}/{case['case_id']}",
            flush=True,
        )

    ensure_source_lock(output_root, create=False)
    if len(scored_records) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"scored record count {len(scored_records)} != {EXPECTED_TOTAL}"
        )
    legacy.write_jsonl_exclusive(
        scoring_root / "scoring-records.jsonl", scored_records
    )
    legacy.write_jsonl_exclusive(
        output_root / "case-results.jsonl", scored_records
    )
    clean_validation = clean_reference_validation(scored_records)
    scoring_validation = {
        "created_utc": utc_now(),
        "status": "pass",
        "case_count": len(scored_records),
        "speech_count": sum(
            not bool(row["pure_noise"]) for row in scored_records
        ),
        "noise_count": sum(bool(row["pure_noise"]) for row in scored_records),
        "candidate_scores_recomputed": True,
        "old_cofilite_output_role": "FP32 stream-parity reference only",
        "dpdfnet_scores_reused_from_frozen_reference": True,
        "clean_reference": clean_validation,
        "parity": parity_summary,
        "scoring_root": str(scoring_root),
    }
    legacy.write_json_exclusive(
        output_root / "SCORING-VALIDATION.json", scoring_validation
    )
    legacy.write_json_exclusive(
        scoring_root / "score-cache.snapshot.json",
        {"schema_version": 1, "values": cache.values},
    )
    legacy.append_log(
        log_path,
        "optimized scoring complete "
        f"cases={len(scored_records)} "
        f"clean_max_abs_delta="
        f"{clean_validation['max_abs_delta_vs_frozen_csv']:.12g} "
        f"clean_sha_mismatches="
        f"{clean_validation['common_audio_sha256_mismatches']}",
    )
    print(
        json.dumps(
            legacy.json_safe(scoring_validation),
            sort_keys=True,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


def clean_reference_validation(
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    clean_records = [
        row["scoring"]["clean_reference"]
        for row in records
        if row["scoring"]["clean_reference"] is not None
    ]
    if len(clean_records) != EXPECTED_SPEECH:
        raise RuntimeError(
            f"clean-reference record count {len(clean_records)} "
            f"!= {EXPECTED_SPEECH}"
        )
    max_delta = max(
        float(row["max_abs_delta_vs_frozen_csv"]) for row in clean_records
    )
    sha_mismatches = sum(
        not bool(row["common_audio_sha256_matches_frozen"])
        for row in clean_records
    )
    return {
        "cases": len(clean_records),
        "max_abs_delta_vs_frozen_csv": max_delta,
        "common_audio_sha256_mismatches": sha_mismatches,
        "exact_within_1e-12": max_delta <= 1e-12 and sha_mismatches == 0,
    }


def pyloudnorm_identity(score_module: ModuleType) -> dict[str, Any]:
    deps_dir = PROJECT_ROOT / "build/ns-next-20261115/deps"
    if str(deps_dir) not in sys.path:
        sys.path.insert(0, str(deps_dir))
    import pyloudnorm

    source_dir = Path(pyloudnorm.__file__).resolve().parent
    source_hashes = {
        name: sha256_file(source_dir / name)
        for name in score_module.PYLOUDNORM_011_SHA256
    }
    if source_hashes != score_module.PYLOUDNORM_011_SHA256:
        raise RuntimeError(
            "unpacked pyloudnorm source is not the frozen 0.1.1 source"
        )
    return {
        "version": "0.1.1",
        "source": "unpacked_source_hash_verified",
        "source_dir": str(source_dir),
        "sha256": source_hashes,
    }


def build_source_identities(
    output_root: Path,
    reference_data: dict[str, Any],
    c4eb_data: dict[str, Any],
    score_module: ModuleType,
) -> dict[str, Any]:
    legacy = configure_legacy()
    constants = candidate_constants()
    dpdf_manifest = reference_data["manifest"]
    dpdf_model_path = Path(dpdf_manifest["runtime_refs"]["model_path"])
    dpdf_model_expected = dpdf_manifest["source_code"]["model_sha256"]
    dpdf_model_actual = sha256_file(dpdf_model_path)
    if dpdf_model_actual != dpdf_model_expected:
        raise RuntimeError("frozen DPDFNet model SHA changed")
    lock_path = output_root / "RUNTIME-SOURCE-LOCK.json"
    return {
        "schema_version": 1,
        "created_utc": utc_now(),
        "driver": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "legacy_validator_path": str(LEGACY_VALIDATOR),
            "legacy_validator_sha256": sha256_file(LEGACY_VALIDATOR),
        },
        "runtime_source_lock": {
            "path": str(lock_path),
            "sha256": sha256_file(lock_path),
            "value": load_json(lock_path),
        },
        "stream": {
            "source_path": str(RUNTIME_SOURCE),
            "source_sha256": sha256_file(RUNTIME_SOURCE),
            "frame_source_path": str(RUNTIME_FRAME_SOURCE),
            "frame_source_sha256": sha256_file(RUNTIME_FRAME_SOURCE),
            "constants": constants,
            "model_adapter_path": str(
                PROJECT_ROOT / "build/ns-cofilite-20261115/cofi_infer.py"
            ),
            "model_adapter_sha256": sha256_file(
                PROJECT_ROOT / "build/ns-cofilite-20261115/cofi_infer.py"
            ),
            "model_source_path": str(
                PROJECT_ROOT
                / "build/ns-cofilite-20261115/upstream/model/cofilite.py"
            ),
            "model_source_sha256": sha256_file(
                PROJECT_ROOT
                / "build/ns-cofilite-20261115/upstream/model/cofilite.py"
            ),
            "checkpoint_path": str(
                PROJECT_ROOT
                / "build/ns-cofilite-20261115/upstream/checkpoint/"
                "model_on_dns3.tar"
            ),
            "checkpoint_sha256": sha256_file(
                PROJECT_ROOT
                / "build/ns-cofilite-20261115/upstream/checkpoint/"
                "model_on_dns3.tar"
            ),
        },
        "scorer": {
            "stage_tool_path": str(legacy.STAGE_TOOL),
            "stage_tool_sha256": sha256_file(legacy.STAGE_TOOL),
            "run_and_score_path": str(legacy.RUN_AND_SCORE),
            "run_and_score_sha256": sha256_file(legacy.RUN_AND_SCORE),
            "ecnr_perceptual_path": str(
                legacy.STAGE_TOOL.parent / "ecnr_perceptual.py"
            ),
            "ecnr_perceptual_sha256": sha256_file(
                legacy.STAGE_TOOL.parent / "ecnr_perceptual.py"
            ),
            "dnsmos_model_path": str(legacy.DNSMOS_MODEL),
            "dnsmos_model_sha256": sha256_file(legacy.DNSMOS_MODEL),
            "assets_dir": str(legacy.ASSETS_DIR),
            "score_rate": constants["sample_rate"],
            "normalization_lufs": -30.0,
            "raw_and_minus30": True,
            "pyloudnorm": pyloudnorm_identity(score_module),
        },
        "references": {
            "dpdfnet2": {
                **reference_data["source_sha256"],
                "model_path": str(dpdf_model_path),
                "model_sha256": dpdf_model_actual,
                "algorithm_delay_samples": DPDF_REFERENCE_DELAY_SAMPLES,
                "dynamic_delay_search": False,
            },
            "c4eb": c4eb_data["sources"],
            "cofilite_receipts": {
                group: {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
                for group, path in legacy.RECEIPTS.items()
            },
        },
        "protocol": {
            "case_count": EXPECTED_TOTAL,
            "speech_count": EXPECTED_SPEECH,
            "noise_count": EXPECTED_NOISE,
            "groups": EXPECTED_GROUPS,
            "ns_only": True,
            "full_vcp": False,
            "device_realtime": False,
            "hard_realtime_verified": False,
            "stream_chunk_samples": constants["frame_samples"],
            "candidate_physical_delay_samples": constants["delay_samples"],
            "candidate_physical_delay_ms": (
                constants["delay_samples"] * 1000.0 / constants["sample_rate"]
            ),
            "candidate_trim_samples": constants["delay_samples"],
            "score_fixed_delay_samples": 0,
            "dpdfnet2_reference_fixed_delay_samples": (
                DPDF_REFERENCE_DELAY_SAMPLES
            ),
            "dpdfnet2_reference_fixed_delay_ms": (
                DPDF_REFERENCE_DELAY_SAMPLES
                * 1000.0
                / constants["sample_rate"]
            ),
            "dynamic_delay_search": False,
            "same_window_s": True,
            "pcm16_lround_once": True,
            "torch_inference_mode": True,
            "processes_max": 4,
            "model_loaded_once_per_worker": True,
            "shared_model_threads": False,
            "gpu_used": False,
            "speech_weighting": (
                "language equal; SNR equal within language; speaker equal "
                "within stratum"
            ),
            "noise_mos": False,
            "noise_metric": (
                "energy ratio versus actual NS input on the same window"
            ),
        },
    }


def build_report(
    parity_summary: dict[str, Any],
    score_summary: dict[str, Any],
    source_identities: dict[str, Any],
    validation: dict[str, Any],
) -> str:
    legacy = configure_legacy()
    fmt = legacy.fmt
    md_table = legacy.md_table
    find_metric = legacy.find_metric
    find_clean = legacy.find_clean
    constants = candidate_constants()
    delay = constants["delay_samples"]
    candidate_ms = delay * 1000.0 / constants["sample_rate"]
    dpdf_ms = (
        DPDF_REFERENCE_DELAY_SAMPLES * 1000.0 / constants["sample_rate"]
    )
    speech = score_summary["v1speech48"]
    speech_section = "overall_language_snr_speaker_equal"
    speech_rows = []
    for metric, label in (
        ("raw_G", "raw G"),
        ("raw_S", "raw S"),
        ("raw_N", "raw N"),
        ("minus30_G", "-30 LUFS G"),
        ("minus30_S", "-30 LUFS S"),
        ("minus30_N", "-30 LUFS N"),
        ("pesq", "PESQ-WB"),
        ("stoi", "STOI"),
        ("weak_speech_energy_ratio_vs_input_db", "weak energy/input dB"),
    ):
        speech_rows.append(
            [
                label,
                fmt(
                    find_metric(
                        speech,
                        "v1speech48",
                        speech_section,
                        metric,
                        "candidate",
                    )
                ),
                fmt(
                    find_metric(
                        speech,
                        "v1speech48",
                        speech_section,
                        metric,
                        "dpdfnet2",
                    )
                ),
                fmt(
                    find_metric(
                        speech,
                        "v1speech48",
                        speech_section,
                        metric,
                        "delta_vs_dpdfnet2",
                    )
                ),
                fmt(
                    find_metric(
                        speech,
                        "v1speech48",
                        speech_section,
                        metric,
                        "c4eb",
                    )
                ),
                fmt(
                    find_metric(
                        speech,
                        "v1speech48",
                        speech_section,
                        metric,
                        "delta_vs_c4eb",
                    )
                ),
            ]
        )

    language_rows = []
    for language in sorted(speech["by_language"]):
        bundle = speech["by_language"][language]["snr_speaker_equal"][
            "metrics"
        ]
        for metric in ("raw_G", "raw_S", "raw_N"):
            item = bundle[metric]
            language_rows.append(
                [
                    language,
                    metric,
                    fmt(
                        item["candidate"]["mean"]
                        if item.get("candidate") is not None
                        else None
                    ),
                    fmt(
                        item["delta_vs_dpdfnet2"]["mean"]
                        if item.get("delta_vs_dpdfnet2") is not None
                        else None
                    ),
                ]
            )

    snr_rows = []
    for key in sorted(speech["by_language_snr"]):
        item = speech["by_language_snr"][key]
        metrics = item["speaker_equal"]["metrics"]
        snr_rows.append(
            [
                item["language"],
                "none" if item["snr_db"] is None else f"{item['snr_db']:g}",
                fmt(
                    metrics["raw_G"]["candidate"]["mean"]
                    if metrics["raw_G"].get("candidate") is not None
                    else None
                ),
                fmt(
                    metrics["raw_G"]["delta_vs_dpdfnet2"]["mean"]
                    if metrics["raw_G"].get("delta_vs_dpdfnet2") is not None
                    else None
                ),
                fmt(
                    metrics["raw_S"]["delta_vs_dpdfnet2"]["mean"]
                    if metrics["raw_S"].get("delta_vs_dpdfnet2") is not None
                    else None
                ),
                fmt(
                    metrics["raw_N"]["delta_vs_dpdfnet2"]["mean"]
                    if metrics["raw_N"].get("delta_vs_dpdfnet2") is not None
                    else None
                ),
            ]
        )

    clean_rows = []
    for metric in (
        "raw_G",
        "raw_S",
        "raw_N",
        "minus30_G",
        "minus30_S",
        "minus30_N",
    ):
        clean_rows.append(
            [
                metric,
                fmt(
                    find_clean(
                        speech,
                        "v1speech48",
                        speech_section,
                        metric,
                        "recomputed",
                    )
                ),
                fmt(
                    find_clean(
                        speech,
                        "v1speech48",
                        speech_section,
                        metric,
                        "frozen_csv",
                    )
                ),
            ]
        )

    def group_rows_for(group: str, section: str) -> list[list[Any]]:
        item = score_summary[group]
        rows = []
        for metric, label in (
            ("raw_G", "raw G"),
            ("raw_S", "raw S"),
            ("raw_N", "raw N"),
            ("minus30_G", "-30 LUFS G"),
            ("minus30_S", "-30 LUFS S"),
            ("minus30_N", "-30 LUFS N"),
            ("pesq", "PESQ-WB"),
            ("stoi", "STOI"),
            ("weak_speech_energy_ratio_vs_input_db", "weak energy/input dB"),
        ):
            rows.append(
                [
                    label,
                    fmt(
                        find_metric(
                            item, group, section, metric, "candidate"
                        )
                    ),
                    fmt(
                        find_metric(
                            item, group, section, metric, "dpdfnet2"
                        )
                    ),
                    fmt(
                        find_metric(
                            item,
                            group,
                            section,
                            metric,
                            "delta_vs_dpdfnet2",
                        )
                    ),
                ]
            )
        return rows

    clean16_rows = group_rows_for("v1clean16", "overall_speaker_equal")
    continuous_rows = group_rows_for(
        "v1continuous4", "overall_speaker_equal"
    )
    noise_rows = []
    for record in sorted(
        (
            row
            for row in score_summary["_records"]
            if row["dataset"] == "guards"
        ),
        key=lambda row: row["case_id"],
    ):
        noise_rows.append(
            [
                record["case_id"],
                fmt(
                    record["scoring"]["candidate_scores"][
                        "energy_ratio_vs_input_db"
                    ]
                ),
                fmt(
                    record["dpdfnet2_reference"][
                        "energy_ratio_vs_input_db"
                    ]
                ),
                fmt(
                    record["deltas"]["vs_dpdfnet2"][
                        "energy_ratio_vs_input_db"
                    ]
                ),
            ]
        )

    parity_rows = []
    for group in EXPECTED_GROUPS:
        item = parity_summary["groups"][group]
        parity_rows.append(
            [
                group,
                f"{item['pass_count']}/{item['expected_cases']}",
                f"{item['max_abs']:.3e}",
                f"{item['max_rmse']:.3e}",
                f"{item['rtf_median']:.6f}",
                f"{item['rtf_p95']:.6f}",
                ",".join(str(value) for value in item["state_bytes_values"]),
            ]
        )

    clean = validation["clean_reference"]
    return "\n".join(
        [
            "# Optimized CoFi-Lite — frozen 74-case NS-only regression",
            "",
            f"Generated: `{utc_now()}`",
            "",
            "## Result",
            "",
            f"- Full-case FP32 parity: **{parity_summary['pass_count']}/"
            f"{parity_summary['cases']} PASS** at `max_abs < "
            f"{PARITY_MAX_ABS_THRESHOLD:g}`.",
            f"- Global `max_abs={parity_summary['max_abs']:.12g}`; "
            f"`max_rmse={parity_summary['max_rmse']:.12g}`.",
            f"- Final validation status: **{validation['status']}**.",
            "- This is a frozen NS-only regression. It is not a full-VCP run, "
            "not a device real-time proof, and not a quality ceiling.",
            "",
            "## Delay and execution identity",
            "",
            f"- Candidate runtime: fixed `{delay}` samples "
            f"(`{candidate_ms:.6g} ms`) physical delay and trim; aligned "
            "candidate audio is scored with fixed delay `0`.",
            f"- Frozen DPDFNet reference: unchanged `"
            f"{DPDF_REFERENCE_DELAY_SAMPLES}` samples "
            f"(`{dpdf_ms:.6g} ms`) algorithm delay with no dynamic search.",
            "- The candidate and reference delays are separate fields; the "
            "reference delay was not rewritten to the candidate value.",
            f"- Optimized runtime source SHA: `"
            f"{source_identities['stream']['source_sha256']}`.",
            f"- Frame executor source SHA: `"
            f"{source_identities['stream']['frame_source_sha256']}`.",
            f"- Frame executor modes: `"
            f"{', '.join(parity_summary['frame_executors'])}`.",
            f"- Frozen CoFi checkpoint SHA: `"
            f"{source_identities['stream']['checkpoint_sha256']}`.",
            f"- Frozen DPDFNet model SHA: `"
            f"{source_identities['references']['dpdfnet2']['model_sha256']}`.",
            "",
            "## Parity and resources",
            "",
            md_table(
                (
                    "group",
                    "pass",
                    "max abs",
                    "max RMSE",
                    "RTF median",
                    "RTF p95",
                    "state bytes",
                ),
                parity_rows,
            ),
            "",
            f"Overall model RTF median `{parity_summary['rtf_median']:.6f}`, "
            f"p95 `{parity_summary['rtf_p95']:.6f}`. Max cached samples "
            f"`{parity_summary['cached_samples_max']}`. Max post-case process "
            f"RSS `{parity_summary['process_rss_after_max_bytes']}` bytes; max "
            f"within-case RSS delta "
            f"`{parity_summary['process_rss_delta_max_bytes']}` bytes.",
            "",
            "RTF here is a batch measurement over frozen files; it does not "
            "establish device hard-real-time behavior.",
            "",
            "## Speech48",
            "",
            "Equal weighting is applied over languages, nominal SNR strata, "
            "and speaker clusters.",
            "",
            md_table(
                (
                    "metric",
                    "CoFi optimized",
                    "DPDFNet",
                    "Δ vs DPDF",
                    "c4eb",
                    "Δ vs c4eb",
                ),
                speech_rows,
            ),
            "",
            "### By language",
            "",
            md_table(
                ("language", "metric", "CoFi optimized", "Δ vs DPDF"),
                language_rows,
            ),
            "",
            "### By language and nominal SNR",
            "",
            md_table(
                ("language", "SNR dB", "raw G", "ΔG", "ΔS", "ΔN"),
                snr_rows,
            ),
            "",
            "## Clean16",
            "",
            md_table(
                ("metric", "CoFi optimized", "DPDFNet", "Δ vs DPDF"),
                clean16_rows,
            ),
            "",
            "## Continuous4 (includes 60 s cases)",
            "",
            md_table(
                ("metric", "CoFi optimized", "DPDFNet", "Δ vs DPDF"),
                continuous_rows,
            ),
            "",
            "## Guards (pure noise, no MOS)",
            "",
            md_table(
                ("case", "CoFi energy/input dB", "DPDFNet dB", "Δ dB"),
                noise_rows,
            ),
            "",
            "## Clean reference validation",
            "",
            md_table(
                ("metric", "recomputed", "frozen CSV"),
                clean_rows,
            ),
            "",
            f"Clean-reference max absolute delta: "
            f"`{clean['max_abs_delta_vs_frozen_csv']:.12g}`; common-audio "
            f"SHA mismatches: `{clean['common_audio_sha256_mismatches']}`; "
            f"within `1e-12`: `{clean['exact_within_1e-12']}`.",
            "",
            "## Evidence",
            "",
            "- `parity-results.jsonl` — all 74 optimized parity/runtime records.",
            "- `case-results.jsonl` — all 74 parity plus fresh candidate scores.",
            "- `summary.json` — grouped, machine-readable final result.",
            "- `source-identities.json` — runtime, scorer, references, weights.",
            "- `FINAL-VALIDATION.json` — explicit gate checks.",
            "- `RUNTIME-SOURCE-LOCK.json` — source identity captured at parity start.",
            "",
            "The old offline CoFi output is used only as the FP32 stream-parity "
            "reference. Candidate NS scores are recomputed; frozen DPDFNet and "
            "clean-reference records are identity-checked and reused where "
            "specified.",
            "",
            "`deployment_approved=false`; `full_vcp=false`; "
            "`device_realtime=false`; `hard_realtime_verified=false`.",
            "",
        ]
    )


def final_validation(
    records: Sequence[dict[str, Any]],
    parity_summary: dict[str, Any],
    clean_validation: dict[str, Any],
) -> dict[str, Any]:
    constants = candidate_constants()
    delay = constants["delay_samples"]
    group_counts = {
        group: sum(row["group"] == group for row in records)
        for group in EXPECTED_GROUPS
    }
    checks = {
        "case_count_74": len(records) == EXPECTED_TOTAL,
        "speech_count_68": sum(not row["pure_noise"] for row in records)
        == EXPECTED_SPEECH,
        "noise_count_6": sum(row["pure_noise"] for row in records)
        == EXPECTED_NOISE,
        "group_counts_48_16_4_6": group_counts == EXPECTED_GROUPS,
        "parity_all_74_pass": parity_summary["all_pass"]
        and parity_summary["pass_count"] == EXPECTED_TOTAL,
        "parity_max_abs_lt_1e-5": parity_summary["max_abs"]
        < PARITY_MAX_ABS_THRESHOLD,
        "candidate_delay_480": all(
            int(row["stream_output"]["physical_delay_samples"]) == delay
            for row in records
        ),
        "candidate_trim_480": all(
            int(row["stream_output"]["trim_samples"]) == delay
            for row in records
        ),
        "candidate_score_delay_0": all(
            int(row["stream_output"]["score_fixed_delay_samples"]) == 0
            for row in records
        ),
        "dpdf_reference_delay_640": all(
            int(row["dpdfnet2_reference"]["fixed_delay_samples"])
            == DPDF_REFERENCE_DELAY_SAMPLES
            for row in records
        ),
        "dynamic_delay_search_false": all(
            row["scoring"]["dynamic_delay_search"] is False for row in records
        ),
        "candidate_scores_present": all(
            isinstance(row.get("scoring", {}).get("candidate_scores"), dict)
            for row in records
        ),
        "clean_reference_exact": clean_validation["exact_within_1e-12"],
        "full_74_scored": len(records) == EXPECTED_TOTAL,
    }
    return {
        "created_utc": utc_now(),
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "case_count": len(records),
        "group_counts": group_counts,
        "parity": {
            "pass_count": parity_summary["pass_count"],
            "threshold": PARITY_MAX_ABS_THRESHOLD,
            "max_abs": parity_summary["max_abs"],
            "max_rmse": parity_summary["max_rmse"],
        },
        "delay_contract": {
            "candidate_physical_delay_samples": delay,
            "candidate_trim_samples": delay,
            "candidate_score_fixed_delay_samples": 0,
            "dpdfnet2_reference_fixed_delay_samples": (
                DPDF_REFERENCE_DELAY_SAMPLES
            ),
            "dynamic_delay_search": False,
        },
        "clean_reference": clean_validation,
        "scope": {
            "ns_only": True,
            "full_vcp": False,
            "device_realtime": False,
            "hard_realtime_verified": False,
            "deployment_approved": False,
        },
    }


def finalize_phase(output_root: Path) -> int:
    legacy = configure_legacy()
    ensure_source_lock(output_root, create=False)
    _, parity_summary = validate_parity(output_root)
    records = legacy.read_jsonl(output_root / "case-results.jsonl")
    if len(records) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"case result count {len(records)} != {EXPECTED_TOTAL}"
        )
    for path in (
        output_root / "summary.json",
        output_root / "source-identities.json",
        output_root / "REPORT-opt.md",
        output_root / "FINAL-VALIDATION.json",
        output_root / "FINAL-MANIFEST.json",
    ):
        if path.exists():
            raise FileExistsError(
                f"refusing to overwrite finalized output: {path}"
            )

    reference_data = load_frozen_reference_data()
    c4eb_data = legacy.load_c4eb_references()
    score_module = legacy.load_module(
        "run_and_score_finalize_optimized", legacy.RUN_AND_SCORE
    )
    score_summary = legacy.build_score_summary(records)
    clean_validation = clean_reference_validation(records)
    source_identities = build_source_identities(
        output_root,
        reference_data,
        c4eb_data,
        score_module,
    )
    validation = final_validation(
        records, parity_summary, clean_validation
    )
    summary = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": (
            "complete" if validation["status"] == "pass"
            else "validation_failed"
        ),
        "execution_status": "complete",
        "scope": {
            "name": "Optimized CoFi-Lite stateful stream NS-only regression",
            "full_vcp": False,
            "device_realtime": False,
            "hard_realtime_verified": False,
            "deployment_approved": False,
            "notes": [
                "frozen actual tx_ns_input cases only",
                "old CoFi output is the FP32 stream-parity reference only",
                "candidate scores were recomputed from optimized stream output",
                "frozen DPDFNet and clean-reference records are reused by identity",
                "candidate physical delay/trim is 480 samples; score delay is 0",
                "frozen DPDFNet reference delay remains 640 samples",
                "speech48, clean16, continuous4, and guards are never pooled",
            ],
        },
        "counts": {
            "total": len(records),
            "speech": sum(not row["pure_noise"] for row in records),
            "noise": sum(row["pure_noise"] for row in records),
            "groups": {
                group: sum(row["group"] == group for row in records)
                for group in EXPECTED_GROUPS
            },
        },
        "parity": parity_summary,
        "clean_reference_validation": clean_validation,
        "scores": score_summary,
        "source_identities": source_identities,
        "final_validation": validation,
    }
    report = build_report(
        parity_summary,
        {**score_summary, "_records": records},
        source_identities,
        validation,
    )
    legacy.write_json_exclusive(
        output_root / "source-identities.json", source_identities
    )
    legacy.write_json_exclusive(output_root / "summary.json", summary)
    with (output_root / "REPORT-opt.md").open(
        "x", encoding="utf-8"
    ) as stream:
        stream.write(report)
    legacy.write_json_exclusive(
        output_root / "FINAL-VALIDATION.json", validation
    )
    manifest_paths = (
        "RUNTIME-SOURCE-LOCK.json",
        "parity-results.jsonl",
        "parity-summary.json",
        "case-results.jsonl",
        "SCORING-VALIDATION.json",
        "source-identities.json",
        "summary.json",
        "REPORT-opt.md",
        "FINAL-VALIDATION.json",
    )
    manifest = {
        "created_utc": utc_now(),
        "status": validation["status"],
        "files": {
            name: {
                "path": str(output_root / name),
                "bytes": (output_root / name).stat().st_size,
                "sha256": sha256_file(output_root / name),
            }
            for name in manifest_paths
        },
        "scope": validation["scope"],
    }
    legacy.write_json_exclusive(
        output_root / "FINAL-MANIFEST.json", manifest
    )
    legacy.append_log(
        output_root / "run.log",
        "optimized finalize complete "
        f"status={validation['status']} "
        f"parity={parity_summary['pass_count']}/{parity_summary['cases']} "
        f"max_abs={parity_summary['max_abs']:.12g} "
        f"clean_exact={clean_validation['exact_within_1e-12']}",
    )
    print(
        json.dumps(
            legacy.json_safe(
                {
                    "status": validation["status"],
                    "parity": parity_summary,
                    "final_validation": validation,
                    "summary": str(output_root / "summary.json"),
                    "report": str(output_root / "REPORT-opt.md"),
                }
            ),
            sort_keys=True,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if validation["status"] == "pass" else 2


def preflight_phase(output_root: Path) -> int:
    legacy = configure_legacy()
    output_root.mkdir(parents=True, exist_ok=True)
    attempt = legacy.next_attempt_dir(output_root, "preflight")
    cases = legacy.load_cases()
    reference_data = load_frozen_reference_data()
    c4eb_data = legacy.load_c4eb_references()
    constants = candidate_constants()
    if sha256_file(legacy.STAGE_TOOL) != legacy.EXPECTED_STAGE_TOOL_SHA256:
        raise RuntimeError("frozen stage tool SHA changed")
    if sha256_file(legacy.RUN_AND_SCORE) != legacy.EXPECTED_RUN_AND_SCORE_SHA256:
        raise RuntimeError("frozen run_and_score SHA changed")
    if sha256_file(legacy.DNSMOS_MODEL) != legacy.EXPECTED_DNSMOS_SHA256:
        raise RuntimeError("DNSMOS ONNX SHA changed")
    result = {
        "created_utc": utc_now(),
        "status": "ready",
        "model_execution_started": False,
        "numeric_scoring_started": False,
        "case_count": len(cases),
        "speech_count": sum(not row["pure_noise"] for row in cases),
        "noise_count": sum(row["pure_noise"] for row in cases),
        "group_counts": {
            group: sum(row["group"] == group for row in cases)
            for group in EXPECTED_GROUPS
        },
        "candidate_constants": constants,
        "dpdf_reference_delay_samples": DPDF_REFERENCE_DELAY_SAMPLES,
        "dynamic_delay_search": False,
        "source_identity": execution_source_identity(),
        "reference_identities": reference_data["source_sha256"],
        "c4eb_complete": bool(c4eb_data["complete"]),
        "scorer_hashes_verified": True,
        "required_launch_authorization": (
            "root must confirm the runtime worker benchmark is complete and "
            "the runtime source is frozen"
        ),
    }
    legacy.write_json_exclusive(attempt / "preflight.json", result)
    legacy.append_log(
        output_root / "run.log",
        f"optimized preflight ready cases={len(cases)} "
        f"candidate_delay={constants['delay_samples']} "
        f"dpdf_reference_delay={DPDF_REFERENCE_DELAY_SAMPLES} "
        "model_execution_started=False",
    )
    print(
        json.dumps(
            legacy.json_safe(result), sort_keys=True, ensure_ascii=False
        ),
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        choices=("preflight", "parity", "score", "finalize", "all"),
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=EXPECTED_OUTPUT_ROOT,
    )
    parser.add_argument("--jobs", type=int, default=4)
    return parser


def validate_output_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    expected = EXPECTED_OUTPUT_ROOT.resolve()
    if resolved != expected:
        raise SystemExit(
            f"--output-root must be exactly {expected}; got {resolved}"
        )
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    global _ACTIVE_OUTPUT_ROOT
    args = build_parser().parse_args(argv)
    if args.jobs < 1 or args.jobs > 4:
        raise SystemExit("--jobs must be between 1 and 4")
    output_root = validate_output_root(args.output_root)
    _ACTIVE_OUTPUT_ROOT = output_root
    if args.phase == "preflight":
        return preflight_phase(output_root)

    legacy = configure_legacy()
    output_root.mkdir(parents=True, exist_ok=True)
    cases = legacy.load_cases()
    if args.phase == "parity":
        return parity_phase(cases, output_root, args.jobs)
    if args.phase == "score":
        return score_phase(cases, output_root)
    if args.phase == "finalize":
        return finalize_phase(output_root)

    status = parity_phase(cases, output_root, args.jobs)
    if status != 0:
        print(
            "optimized parity threshold failed; scoring was not started",
            file=sys.stderr,
            flush=True,
        )
        return status
    status = score_phase(cases, output_root)
    if status != 0:
        return status
    return finalize_phase(output_root)


if __name__ == "__main__":
    raise SystemExit(main())
