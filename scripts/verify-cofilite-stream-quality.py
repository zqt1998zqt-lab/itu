#!/usr/bin/env python3
"""Verify CoFi-Lite streaming parity and NS-only quality on the frozen 74 cases.

This is intentionally not a full-VCP comparison.  It consumes the exact
``tx_ns_input`` boundary recorded by the frozen CoFi evaluation, applies the
same one-time PCM16 ``lround`` quantization, runs the stateful stream adapter,
and compares the aligned output with the old offline CoFi output.

The scoring phase uses the frozen NS scorer primitives without running the VCP
engine again.  A fresh stream output is already trimmed by 640 samples, so it is
scored with ``fixed_delay_samples=0``; the physical delay and trim remain
explicitly recorded as 640 samples.
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True

PROJECT_ROOT = "/home/zhangqingtian/project/itu"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT + "/build/cofilite-stream-20261116-v1/quality74"
)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", DEFAULT_OUTPUT_ROOT + "/cache/numba")
os.environ.setdefault("MPLCONFIGDIR", DEFAULT_OUTPUT_ROOT + "/cache/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", DEFAULT_OUTPUT_ROOT + "/cache/xdg")

import argparse
import concurrent.futures
import csv
import hashlib
import importlib.util
import json
import math
import multiprocessing
import platform
import re
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import soundfile as sf


SCHEMA_VERSION = 1
EXPECTED_TOTAL = 74
EXPECTED_SPEECH = 68
EXPECTED_NOISE = 6
PARITY_MAX_ABS_THRESHOLD = 1e-5
PHYSICAL_DELAY_SAMPLES = 640
TRIM_SAMPLES = 640
STREAM_CHUNK_SIZE = 160
SAMPLE_RATE = 16000

COFILITE_ROOT = Path(PROJECT_ROOT) / "build/ns-cofilite-20261115"
RECEIPTS = {
    "v1speech48": COFILITE_ROOT / "evaluation/v1speech48/candidate/receipt.json",
    "v1clean16": COFILITE_ROOT / "evaluation/v1clean16/candidate/receipt.json",
    "v1continuous4": COFILITE_ROOT / "evaluation/v1continuous4/candidate/receipt.json",
    "v1guards": COFILITE_ROOT / "evaluation/v1guards/candidate/receipt.json",
}
GROUP_SPECS = (
    ("v1speech48", "speech48", 48, False),
    ("v1clean16", "clean16", 16, False),
    ("v1continuous4", "continuous4", 4, False),
    ("v1guards", "guards", 6, True),
)
DATASET_BY_GROUP = {group: dataset for group, dataset, _, _ in GROUP_SPECS}
GROUP_BY_DATASET = {dataset: group for group, dataset, _, _ in GROUP_SPECS}

STREAM_SOURCE = Path(PROJECT_ROOT) / "server/cofilite_stream.py"
STAGE_TOOL = Path(PROJECT_ROOT) / "build/ns-next-20261115/vcp_stage_mos.py"
RUN_AND_SCORE = (
    Path(PROJECT_ROOT) / "build/ns-next-20261115/evaluation/run_and_score.py"
)
ASSETS_DIR = Path(PROJECT_ROOT) / "build/ecnr-v2-20261011/perceptual-assets"
DNSMOS_MODEL = ASSETS_DIR / "sig_bak_ovr.onnx"
EXPECTED_DNSMOS_SHA256 = (
    "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd"
)
EXPECTED_STAGE_TOOL_SHA256 = (
    "d9609b85efeffc8e6317596ae7e0c74488a7dbf16e6bc896f62c47f5ff265f2b"
)
EXPECTED_RUN_AND_SCORE_SHA256 = (
    "bb49e43f57e8736e24aa58c89b97f55db3873e15c045b7e4afd9e9b289a6eda7"
)

DPDF_ROOT = Path(PROJECT_ROOT) / "benchmarks/ns/dpdfnet2-16k-v1"
DPDF_MANIFEST = DPDF_ROOT / "manifest.json"
DPDF_CASES = DPDF_ROOT / "reference-cases.json"
DPDF_STAGES = DPDF_ROOT / "reference-stages.csv"
DPDF_CLEAN = DPDF_ROOT / "clean-reference.csv"

C4EB_MANIFESTS = {
    "speech48": (
        Path(PROJECT_ROOT)
        / "build/ns-next-20261115/evaluation/runs/"
        "speech48-final-20261114T163202Z/run_manifest.json"
    ),
    "clean16": (
        Path(PROJECT_ROOT)
        / "build/ns-next-20261115/evaluation/runs/"
        "clean16-final-20261114T163748Z/run_manifest.json"
    ),
    "continuous4": (
        Path(PROJECT_ROOT)
        / "build/ns-next-20261115/evaluation/runs/"
        "continuous4-final-20261114T163814Z/run_manifest.json"
    ),
    "guards": (
        Path(PROJECT_ROOT)
        / "build/ns-next-20261115/evaluation/runs/"
        "guards-final-20261114T163819Z/run_manifest.json"
    ),
}

SCORE_METRICS = (
    "raw_G",
    "raw_S",
    "raw_N",
    "minus30_G",
    "minus30_S",
    "minus30_N",
    "pesq",
    "stoi",
    "sisdr_db",
    "weak_speech_energy_ratio_vs_input_db",
    "energy_ratio_vs_input_db",
)

_WORKER_MODEL: Any | None = None
_WORKER_MODEL_LOAD_S: float | None = None
_WORKER_PID: int | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array_f8(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest()


def safe_name(value: Any) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return result or "unnamed"


def finite_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(child) for child in value]
    if isinstance(value, np.ndarray):
        return [json_safe(child) for child in value.tolist()]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(
            json_safe(value),
            stream,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        stream.write("\n")


def write_jsonl_exclusive(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            json.dump(
                json_safe(row),
                stream,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"[{utc_now()}] {message.rstrip()}\n")


def next_attempt_dir(root: Path, base: str) -> Path:
    for index in range(1, 10_000):
        name = base if index == 1 else f"{base}-{index:03d}"
        candidate = root / name
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"cannot allocate a new attempt directory below {root}")


def process_rss_bytes() -> int:
    for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return 0


def pcm16_lround(values: np.ndarray) -> np.ndarray:
    scaled = np.clip(
        np.asarray(values, dtype=np.float32) * np.float32(32768.0),
        -32768.0,
        32767.0,
    )
    rounded = np.copysign(
        np.floor(np.abs(scaled) + np.float32(0.5)),
        scaled,
    )
    return np.asarray(rounded, dtype=np.int16)


def pcm16_lround_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(
        pcm16_lround(values).astype("<i2", copy=False).tobytes()
    ).hexdigest()


def read_float_mono(path: Path, label: str) -> tuple[np.ndarray, dict[str, Any]]:
    info = sf.info(path)
    if info.samplerate != SAMPLE_RATE:
        raise ValueError(f"{label} rate {info.samplerate} != {SAMPLE_RATE}: {path}")
    if info.channels != 1:
        raise ValueError(f"{label} is not mono: {path}")
    if info.subtype != "FLOAT":
        raise ValueError(f"{label} subtype {info.subtype} != FLOAT: {path}")
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    values = np.ascontiguousarray(audio[:, 0])
    if rate != SAMPLE_RATE or values.size == 0:
        raise ValueError(f"{label} failed rate/shape validation: {path}")
    if not bool(np.isfinite(values).all()):
        raise ValueError(f"{label} contains NaN/Inf: {path}")
    return values, {
        "format": info.format,
        "subtype": info.subtype,
        "sample_rate": int(rate),
        "channels": int(audio.shape[1]),
        "samples": int(values.size),
        "dtype": "float32",
    }


def write_float_wav_new(
    path: Path, values: np.ndarray
) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite stream output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        path,
        np.asarray(values, dtype=np.float32),
        SAMPLE_RATE,
        subtype="FLOAT",
        format="WAV",
    )
    reread, rate = sf.read(path, dtype="float32", always_2d=True)
    reread_values = np.ascontiguousarray(reread[:, 0])
    if rate != SAMPLE_RATE or reread.shape != (values.size, 1):
        raise RuntimeError(f"stream FLOAT WAV changed shape/rate on reread: {path}")
    if not np.array_equal(reread_values, values):
        raise RuntimeError(f"stream FLOAT WAV changed payload on reread: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "sample_rate": int(rate),
        "channels": int(reread.shape[1]),
        "samples": int(reread_values.size),
        "subtype": "FLOAT",
        "finite": bool(np.isfinite(reread_values).all()),
    }


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def parse_window(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"invalid window: {value!r}")
    return [float(value[0]), float(value[1])]


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise ValueError(f"missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for group, dataset, expected, pure_noise_group in GROUP_SPECS:
        receipt_path = RECEIPTS[group]
        receipt = load_json(receipt_path)
        runs = receipt.get("runs")
        if not isinstance(runs, list) or len(runs) != expected:
            raise ValueError(
                f"{group}: expected {expected} runs, found "
                f"{len(runs) if isinstance(runs, list) else 'non-list'}"
            )
        if receipt.get("completed") != expected or receipt.get("failed") != 0:
            raise ValueError(
                f"{group}: receipt completed/failed mismatch: "
                f"{receipt.get('completed')}/{receipt.get('failed')}"
            )
        for run in runs:
            case_id = str(run.get("case_id") or run.get("case", {}).get("id") or "")
            if not case_id:
                raise ValueError(f"{group}: run lacks case_id")
            if run.get("status") != "ok":
                raise ValueError(f"{group}/{case_id}: old run status is not ok")
            case = run.get("case")
            if not isinstance(case, dict):
                raise ValueError(f"{group}/{case_id}: embedded case is missing")
            if bool(run.get("pure_noise")) != bool(pure_noise_group):
                raise ValueError(f"{group}/{case_id}: pure-noise receipt mismatch")
            input_path = Path(str(run.get("candidate_input_float_path", "")))
            old_output_path = Path(str(run.get("model_output_path", "")))
            if not input_path.is_file():
                raise ValueError(f"{group}/{case_id}: input missing: {input_path}")
            if not old_output_path.is_file():
                raise ValueError(
                    f"{group}/{case_id}: old model output missing: {old_output_path}"
                )
            if sha256_file(input_path) != run.get("candidate_input_float_sha256"):
                raise ValueError(f"{group}/{case_id}: input file SHA changed")
            if sha256_file(old_output_path) != run.get("model_output_sha256"):
                raise ValueError(f"{group}/{case_id}: old output SHA changed")
            cases.append(
                {
                    "group": group,
                    "dataset": dataset,
                    "case_id": case_id,
                    "case": case,
                    "run": run,
                    "receipt_path": str(receipt_path),
                    "receipt_sha256": sha256_file(receipt_path),
                    "pure_noise": bool(pure_noise_group),
                    "language": str(case.get("language") or "unknown"),
                    "snr_db": parse_float(case.get("snr_db")),
                    "speaker_cluster": str(
                        case.get("speaker_id")
                        or case.get("speaker_cluster")
                        or case_id
                    ),
                    "kind": str(case.get("kind") or "near_only"),
                    "window_s": parse_window(case.get("window_s")),
                    "clean_path": (
                        str(case.get("clean_path"))
                        if case.get("clean_path")
                        else None
                    ),
                    "input_path": str(input_path),
                    "input_sha256": run["candidate_input_float_sha256"],
                    "input_pcm16_lround_sha256": run[
                        "candidate_input_pcm16_lround_sha256"
                    ],
                    "old_output_path": str(old_output_path),
                    "old_output_sha256": run["model_output_sha256"],
                    "old_output_samples": int(run["model_output_samples"]),
                }
            )
    if len(cases) != EXPECTED_TOTAL:
        raise ValueError(f"expected {EXPECTED_TOTAL} cases, found {len(cases)}")
    ids = [row["case_id"] for row in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("case IDs are not unique across the frozen groups")
    return cases


def import_stream_module() -> ModuleType:
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    import server.cofilite_stream as stream_module

    return stream_module


def worker_initialize() -> None:
    global _WORKER_MODEL, _WORKER_MODEL_LOAD_S, _WORKER_PID
    import torch

    torch.set_num_threads(1)
    _WORKER_PID = os.getpid()
    stream_module = import_stream_module()
    started = time.perf_counter()
    _WORKER_MODEL = stream_module.load_model("cpu", 1)
    _WORKER_MODEL_LOAD_S = time.perf_counter() - started


def process_one_stream_case(payload: dict[str, Any]) -> dict[str, Any]:
    global _WORKER_MODEL, _WORKER_MODEL_LOAD_S, _WORKER_PID
    if _WORKER_MODEL is None:
        worker_initialize()
    import torch

    stream_module = import_stream_module()
    input_path = Path(payload["input_path"])
    old_output_path = Path(payload["old_output_path"])
    output_path = Path(payload["stream_output_path"])
    values, input_info = read_float_mono(
        input_path, f"{payload['group']}/{payload['case_id']} input"
    )
    if sha256_file(input_path) != payload["input_sha256"]:
        raise RuntimeError(f"{payload['case_id']}: input SHA changed in worker")
    actual_pcm_sha = pcm16_lround_sha256(values)
    if actual_pcm_sha != payload["input_pcm16_lround_sha256"]:
        raise RuntimeError(f"{payload['case_id']}: PCM16 lround SHA mismatch")

    quantized = (
        pcm16_lround(values).astype(np.float32) / np.float32(32768.0)
    )
    rss_before = process_rss_bytes()
    started = time.perf_counter()
    with torch.inference_mode():
        output, info = stream_module.enhance_stream(
            quantized,
            chunk_size=STREAM_CHUNK_SIZE,
            model=_WORKER_MODEL,
        )
    elapsed = time.perf_counter() - started
    rss_after = process_rss_bytes()
    output = np.ascontiguousarray(output, dtype=np.float32)
    if output.shape != values.shape:
        raise RuntimeError(
            f"{payload['case_id']}: stream output shape {output.shape} != {values.shape}"
        )
    if not bool(np.isfinite(output).all()):
        raise RuntimeError(f"{payload['case_id']}: stream output is non-finite")
    output_info = write_float_wav_new(output_path, output)

    old_output, old_info = read_float_mono(
        old_output_path, f"{payload['group']}/{payload['case_id']} old output"
    )
    if old_output.shape != output.shape:
        raise RuntimeError(
            f"{payload['case_id']}: old/new output shapes differ: "
            f"{old_output.shape}/{output.shape}"
        )
    difference = np.abs(
        output.astype(np.float64) - old_output.astype(np.float64)
    )
    max_abs = float(np.max(difference))
    rmse = float(np.sqrt(np.mean(difference * difference)))
    old_rms = float(
        np.sqrt(np.mean(old_output.astype(np.float64) ** 2))
    )
    relative_rmse = (
        rmse / old_rms if old_rms > 0.0 else (0.0 if rmse == 0.0 else None)
    )
    max_index = int(np.argmax(difference))
    if int(info["delay_samples"]) != PHYSICAL_DELAY_SAMPLES:
        raise RuntimeError(
            f"{payload['case_id']}: delay {info['delay_samples']} != "
            f"{PHYSICAL_DELAY_SAMPLES}"
        )
    if int(info["trim_samples"]) != TRIM_SAMPLES:
        raise RuntimeError(
            f"{payload['case_id']}: trim {info['trim_samples']} != {TRIM_SAMPLES}"
        )
    if int(info["chunk_size"]) != STREAM_CHUNK_SIZE:
        raise RuntimeError(f"{payload['case_id']}: chunk size changed")
    if int(info["aligned_samples"]) != int(values.size):
        raise RuntimeError(f"{payload['case_id']}: aligned sample count changed")

    duration_s = float(values.size) / SAMPLE_RATE
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "group": payload["group"],
        "dataset": payload["dataset"],
        "case_id": payload["case_id"],
        "language": payload["language"],
        "snr_db": payload["snr_db"],
        "speaker_cluster": payload["speaker_cluster"],
        "kind": payload["kind"],
        "pure_noise": payload["pure_noise"],
        "window_s": payload["window_s"],
        "clean_path": payload["clean_path"],
        "input": {
            "path": str(input_path),
            "sha256": payload["input_sha256"],
            "pcm16_lround_sha256": actual_pcm_sha,
            "expected_pcm16_lround_sha256": payload[
                "input_pcm16_lround_sha256"
            ],
            **input_info,
        },
        "old_offline_output": {
            "path": str(old_output_path),
            "sha256": payload["old_output_sha256"],
            **old_info,
        },
        "stream_output": {
            **output_info,
            "physical_delay_samples": PHYSICAL_DELAY_SAMPLES,
            "trim_samples": TRIM_SAMPLES,
            "score_fixed_delay_samples": 0,
            "delay_note": (
                "physical stream delay 640 samples was drained and trimmed; "
                "the new aligned FLOAT file is scored with fixed delay 0"
            ),
        },
        "stream_info": {
            **info,
            "torch_inference_mode": True,
            "model_loaded_once_per_worker": True,
            "stateful_streaming": True,
            "offline_whole_utterance": False,
        },
        "parity": {
            "samples": int(values.size),
            "duration_s": duration_s,
            "max_abs": max_abs,
            "rmse": rmse,
            "relative_rmse": relative_rmse,
            "max_abs_index": max_index,
            "threshold": PARITY_MAX_ABS_THRESHOLD,
            "pass": max_abs < PARITY_MAX_ABS_THRESHOLD,
        },
        "runtime": {
            "elapsed_s": elapsed,
            "rtf": elapsed / duration_s,
            "worker_pid": _WORKER_PID,
            "model_load_s": _WORKER_MODEL_LOAD_S,
            "process_rss_before_bytes": rss_before,
            "process_rss_after_bytes": rss_after,
            "process_rss_delta_bytes": rss_after - rss_before,
            "state_bytes": int(info["state_bytes"]),
            "cached_samples": int(info["cached_samples"]),
            "inferences": int(info["inferences"]),
        },
    }


def summarize_parity(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for group, _, expected, _ in GROUP_SPECS:
        subset = [row for row in records if row["group"] == group]
        groups[group] = {
            "cases": len(subset),
            "expected_cases": expected,
            "pass_count": sum(bool(row["parity"]["pass"]) for row in subset),
            "max_abs": max(row["parity"]["max_abs"] for row in subset),
            "max_rmse": max(row["parity"]["rmse"] for row in subset),
            "mean_rmse": float(
                np.mean([row["parity"]["rmse"] for row in subset])
            ),
            "rtf_median": float(
                np.median([row["runtime"]["rtf"] for row in subset])
            ),
            "rtf_p95": float(
                np.percentile([row["runtime"]["rtf"] for row in subset], 95)
            ),
            "state_bytes_values": sorted(
                {int(row["runtime"]["state_bytes"]) for row in subset}
            ),
            "process_rss_delta_max_bytes": max(
                int(row["runtime"]["process_rss_delta_bytes"]) for row in subset
            ),
            "process_rss_after_max_bytes": max(
                int(row["runtime"]["process_rss_after_bytes"]) for row in subset
            ),
        }
    all_rtf = [float(row["runtime"]["rtf"]) for row in records]
    state_values = sorted(
        {int(row["runtime"]["state_bytes"]) for row in records}
    )
    return {
        "cases": len(records),
        "expected_cases": EXPECTED_TOTAL,
        "pass_count": sum(bool(row["parity"]["pass"]) for row in records),
        "all_pass": all(bool(row["parity"]["pass"]) for row in records),
        "threshold": PARITY_MAX_ABS_THRESHOLD,
        "max_abs": max(float(row["parity"]["max_abs"]) for row in records),
        "max_rmse": max(float(row["parity"]["rmse"]) for row in records),
        "mean_rmse": float(
            np.mean([row["parity"]["rmse"] for row in records])
        ),
        "all_physical_delay_640": all(
            int(row["stream_output"]["physical_delay_samples"])
            == PHYSICAL_DELAY_SAMPLES
            for row in records
        ),
        "all_trim_640": all(
            int(row["stream_output"]["trim_samples"]) == TRIM_SAMPLES
            for row in records
        ),
        "all_score_delay_0": all(
            int(row["stream_output"]["score_fixed_delay_samples"]) == 0
            for row in records
        ),
        "state_bytes_values": state_values,
        "state_bytes_bounded_observation": (
            len(state_values) == 1
            and all(
                int(row["stream_info"]["cached_samples"])
                <= PHYSICAL_DELAY_SAMPLES + 512
                for row in records
            )
        ),
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
        "groups": groups,
    }


def parity_phase(cases: Sequence[dict[str, Any]], output_root: Path, jobs: int) -> int:
    if (output_root / "parity-results.jsonl").exists():
        raise FileExistsError(
            f"refusing to overwrite {output_root / 'parity-results.jsonl'}"
        )
    stream_root = next_attempt_dir(output_root, "stream")
    log_path = output_root / "run.log"
    append_log(
        log_path,
        f"parity start cases={len(cases)} jobs={jobs} stream_root={stream_root}",
    )
    payloads: list[dict[str, Any]] = []
    for case in cases:
        case_dir = (
            stream_root / case["dataset"] / safe_name(case["case_id"])
        )
        case_dir.mkdir(parents=True, exist_ok=False)
        payloads.append(
            {
                **case,
                "stream_output_path": str(
                    case_dir / "stream-output.float.wav"
                ),
            }
        )
    records: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=jobs,
        mp_context=multiprocessing.get_context("fork"),
        initializer=worker_initialize,
    ) as pool:
        futures = {
            pool.submit(process_one_stream_case, payload): payload
            for payload in payloads
        }
        for future in concurrent.futures.as_completed(futures):
            payload = futures[future]
            record = future.result()
            records.append(record)
            append_log(
                log_path,
                "parity "
                f"{record['group']}/{record['case_id']} "
                f"max_abs={record['parity']['max_abs']:.12g} "
                f"rmse={record['parity']['rmse']:.12g} "
                f"rtf={record['runtime']['rtf']:.9g} "
                f"state_bytes={record['runtime']['state_bytes']}",
            )
            print(
                "PARITY "
                f"{record['group']}/{record['case_id']} "
                f"max_abs={record['parity']['max_abs']:.12g} "
                f"rmse={record['parity']['rmse']:.12g}",
                flush=True,
            )
    records.sort(key=lambda row: (row["group"], row["case_id"]))
    write_jsonl_exclusive(output_root / "parity-results.jsonl", records)
    summary = summarize_parity(records)
    write_json_exclusive(output_root / "parity-summary.json", summary)
    append_log(
        log_path,
        "parity complete "
        f"pass={summary['pass_count']}/{summary['cases']} "
        f"max_abs={summary['max_abs']:.12g} "
        f"max_rmse={summary['max_rmse']:.12g}",
    )
    print(json.dumps(json_safe(summary), sort_keys=True), flush=True)
    return 0 if summary["all_pass"] else 2


def load_module(name: str, path: Path) -> ModuleType:
    if not path.is_file():
        raise ValueError(f"missing module: {path}")
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_reference_data() -> dict[str, Any]:
    dpdf_manifest = load_json(DPDF_MANIFEST)
    contract = dpdf_manifest.get("technical_contract")
    if not isinstance(contract, dict):
        raise ValueError("DPDFNet manifest lacks technical_contract")
    if int(contract.get("algorithm_delay_samples")) != PHYSICAL_DELAY_SAMPLES:
        raise ValueError("DPDFNet fixed delay is not 640 samples")
    if contract.get("fixed_delay_search") is not False:
        raise ValueError("DPDFNet reference unexpectedly enables delay search")
    if int(contract.get("hop_samples")) != 160:
        raise ValueError("DPDFNet reference hop is not 160")
    if contract.get("input_boundary_stage") != "tx_ns_input":
        raise ValueError("DPDFNet input boundary is not tx_ns_input")
    if contract.get("output_boundary_stage") != "tx_ns":
        raise ValueError("DPDFNet output boundary is not tx_ns")

    reference_cases = load_json(DPDF_CASES)
    if not isinstance(reference_cases, list) or len(reference_cases) != EXPECTED_TOTAL:
        raise ValueError("DPDFNet reference-cases.json count is not 74")
    cases_by_key = {
        (str(row["dataset"]), str(row["id"])): row
        for row in reference_cases
    }
    if len(cases_by_key) != EXPECTED_TOTAL:
        raise ValueError("DPDFNet reference case keys are not unique")

    with DPDF_STAGES.open("r", encoding="utf-8", newline="") as stream:
        stage_rows = list(csv.DictReader(stream))
    tx_ns_rows = [
        row for row in stage_rows if row.get("stage") == "tx_ns"
    ]
    if len(tx_ns_rows) != EXPECTED_TOTAL:
        raise ValueError(f"DPDFNet tx_ns row count {len(tx_ns_rows)} != 74")
    stages_by_key = {
        (str(row["dataset"]), str(row["case_id"])): row
        for row in tx_ns_rows
    }
    if len(stages_by_key) != EXPECTED_TOTAL:
        raise ValueError("DPDFNet tx_ns keys are not unique")

    with DPDF_CLEAN.open("r", encoding="utf-8", newline="") as stream:
        clean_rows = list(csv.DictReader(stream))
    if len(clean_rows) != EXPECTED_SPEECH:
        raise ValueError(f"DPDFNet clean reference count {len(clean_rows)} != 68")
    clean_by_key = {
        (str(row["dataset"]), str(row["case_id"])): row
        for row in clean_rows
    }
    if len(clean_by_key) != EXPECTED_SPEECH:
        raise ValueError("DPDFNet clean reference keys are not unique")

    return {
        "manifest": dpdf_manifest,
        "contract": contract,
        "cases_by_key": cases_by_key,
        "stages_by_key": stages_by_key,
        "clean_by_key": clean_by_key,
        "source_sha256": {
            "manifest": sha256_file(DPDF_MANIFEST),
            "reference_cases": sha256_file(DPDF_CASES),
            "reference_stages": sha256_file(DPDF_STAGES),
            "clean_reference": sha256_file(DPDF_CLEAN),
        },
    }


def score_fields(row: dict[str, Any]) -> dict[str, float | None]:
    return {
        "raw_G": parse_float(row.get("raw_G")),
        "raw_S": parse_float(row.get("raw_S")),
        "raw_N": parse_float(row.get("raw_N")),
        "minus30_G": parse_float(row.get("minus30_G")),
        "minus30_S": parse_float(row.get("minus30_S")),
        "minus30_N": parse_float(row.get("minus30_N")),
        "pesq": parse_float(row.get("pesq")),
        "stoi": parse_float(row.get("stoi")),
        "sisdr_db": parse_float(row.get("sisdr_db")),
        "weak_speech_energy_ratio_vs_input_db": parse_float(
            row.get("weak_speech_energy_ratio_vs_input_db")
        ),
        "energy_ratio_vs_input_db": parse_float(
            row.get("energy_ratio_vs_input_db")
        ),
    }


def load_c4eb_references() -> dict[str, Any]:
    references: dict[tuple[str, str], dict[str, Any]] = {}
    sources: dict[str, dict[str, Any]] = {}
    for dataset, path in C4EB_MANIFESTS.items():
        manifest = load_json(path)
        runs = manifest.get("runs")
        if not isinstance(runs, list) or not runs:
            raise ValueError(f"c4eb manifest has no runs: {path}")
        for run in runs:
            if run.get("status") != "ok":
                continue
            case_id = str(run.get("case_id") or run.get("case", {}).get("id"))
            stage = next(
                (
                    item
                    for item in (
                        run.get("scored_stages")
                        or run.get("score_stages")
                        or []
                    )
                    if item.get("id") == "tx_ns"
                ),
                None,
            )
            if stage is None:
                continue
            channel = next(
                (
                    item
                    for item in stage.get("channels_scored", [])
                    if int(item.get("channel", -1)) == 0
                ),
                None,
            )
            if channel is None:
                continue
            references[(dataset, case_id)] = {
                **score_fields(channel),
                "source": "c4eb_legacy_same_frozen_cases",
                "fixed_delay_samples": int(
                    stage.get("fixed_delay_samples", 0)
                ),
                "record_path": str(path),
            }
        sources[dataset] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "runs": len(runs),
            "scored_references": sum(
                1 for key in references if key[0] == dataset
            ),
        }
    return {
        "by_key": references,
        "sources": sources,
        "complete": len(references) == EXPECTED_TOTAL,
        "note": (
            "c4eb is an optional same-frozen-case reference. Candidate run "
            "manifests without embedded scored_stages are retained as provenance "
            "but are not treated as scored baselines."
        ),
    }


def dpdf_reference_for(
    case: dict[str, Any], reference_data: dict[str, Any]
) -> dict[str, Any]:
    key = (case["dataset"], case["case_id"])
    case_ref = reference_data["cases_by_key"].get(key)
    stage_ref = reference_data["stages_by_key"].get(key)
    if case_ref is None or stage_ref is None:
        raise ValueError(f"missing DPDFNet reference for {key}")
    expected_window = parse_window(case_ref.get("window_s"))
    if expected_window != case["window_s"]:
        raise ValueError(
            f"{case['dataset']}/{case['case_id']}: DPDFNet window "
            f"{expected_window} != candidate window {case['window_s']}"
        )
    reference = {
        **score_fields(stage_ref),
        "source": "dpdfnet2-16k-v1",
        "stage": "tx_ns",
        "fixed_delay_samples": PHYSICAL_DELAY_SAMPLES,
        "dynamic_delay_search": False,
        "window_s": expected_window,
        "record_path": stage_ref.get("record_path"),
    }
    if reference["raw_G"] is None and not case["pure_noise"]:
        raise ValueError(f"speech DPDFNet reference lacks G for {key}")
    return reference


def clean_frozen_for(
    case: dict[str, Any], reference_data: dict[str, Any]
) -> dict[str, Any]:
    key = (case["dataset"], case["case_id"])
    row = reference_data["clean_by_key"].get(key)
    if row is None:
        raise ValueError(f"missing frozen clean reference for {key}")
    window = parse_window(row.get("window_s"))
    if window != case["window_s"]:
        raise ValueError(
            f"{case['dataset']}/{case['case_id']}: clean window {window} "
            f"!= {case['window_s']}"
        )
    return {
        "window_s": window,
        "clean_path": row.get("clean_path"),
        "clean_sha256": row.get("clean_sha256"),
        "common_audio_sha256": row.get("clean_common_audio_sha256"),
        "score_input_sha256": row.get("clean_score_input_sha256"),
        "raw_G": parse_float(row.get("clean_raw_G")),
        "raw_S": parse_float(row.get("clean_raw_S")),
        "raw_N": parse_float(row.get("clean_raw_N")),
        "minus30_G": parse_float(row.get("clean_minus30_G")),
        "minus30_S": parse_float(row.get("clean_minus30_S")),
        "minus30_N": parse_float(row.get("clean_minus30_N")),
        "record_path": row.get("reference_record_path"),
    }


def score_one_case(
    case: dict[str, Any],
    parity_record: dict[str, Any],
    stage_tool: ModuleType,
    score_module: ModuleType,
    cache: Any,
    metric_cache: dict[str, Any],
    reference_data: dict[str, Any],
    c4eb_data: dict[str, Any],
) -> dict[str, Any]:
    stream_path = Path(parity_record["stream_output"]["path"])
    stream_sha = sha256_file(stream_path)
    if stream_sha != parity_record["stream_output"]["sha256"]:
        raise RuntimeError(f"{case['case_id']}: stream output SHA changed")
    input_path = Path(case["input_path"])
    input_sha = sha256_file(input_path)
    if input_sha != case["input_sha256"]:
        raise RuntimeError(f"{case['case_id']}: input SHA changed before scoring")

    kind = None if case["pure_noise"] else case["kind"]
    if kind is not None and kind not in stage_tool.ALLOWED_ROLES:
        kind = "near_only"
    candidate_stage = {
        "id": "tx_ns_stream",
        "direction": "tx",
        "order": 5,
        "file": str(stream_path),
        "file_sha256": stream_sha,
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "capture_channels": 1,
        "samples_per_channel": int(parity_record["stream_output"]["samples"]),
        "fixed_delay_samples": 0,
        "enabled": True,
        "capture_available": True,
        "diagnostic": False,
        "measurement_domain": (
            "CoFi-Lite stateful stream output; physical delay 640 samples "
            "already drained and trimmed; score delay 0"
        ),
        "channel_policy": "mono actual NS-only stream output",
    }
    candidate_scored, candidate_common_by_channel = score_module.score_stage_audio(
        stage_tool,
        candidate_stage,
        int(parity_record["stream_output"]["samples"]),
        SAMPLE_RATE,
        case["window_s"],
        kind,
        cache,
    )
    candidate_channel = next(
        (
            row
            for row in candidate_scored.get("channels_scored", [])
            if int(row.get("channel", -1)) == 0
        ),
        None,
    )
    if candidate_channel is None:
        raise RuntimeError(f"{case['case_id']}: candidate channel 0 is missing")
    candidate_common = candidate_common_by_channel[0]
    candidate_scores = score_fields(candidate_channel)

    input_stage = {
        "id": "tx_ns_input",
        "direction": "tx",
        "order": 3,
        "file": str(input_path),
        "file_sha256": input_sha,
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "capture_channels": 1,
        "samples_per_channel": int(parity_record["input"]["samples"]),
        "fixed_delay_samples": 0,
        "enabled": True,
        "capture_available": True,
        "diagnostic": True,
        "measurement_domain": "actual frozen NS input boundary",
        "channel_policy": "mono actual NS input",
    }
    input_scored, input_common_by_channel = score_module.score_stage_audio(
        stage_tool,
        input_stage,
        int(parity_record["input"]["samples"]),
        SAMPLE_RATE,
        case["window_s"],
        None,
        cache,
    )
    input_common = input_common_by_channel[0]

    clean_record: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    weak_speech: dict[str, Any] | None = None
    if not case["pure_noise"]:
        if not case["clean_path"]:
            raise ValueError(f"{case['case_id']}: speech case lacks clean path")
        clean_common, clean_meta = score_module.clean_window_audio(
            stage_tool,
            Path(case["clean_path"]),
            SAMPLE_RATE,
            int(parity_record["input"]["samples"]),
            case["window_s"],
        )
        clean_scores = cache.score(clean_common, kind or "near_only")
        frozen_clean = clean_frozen_for(case, reference_data)
        recomputed_clean = {
            "raw_G": clean_scores["original_scores"].get("G"),
            "raw_S": clean_scores["original_scores"].get("S"),
            "raw_N": clean_scores["original_scores"].get("N"),
            "minus30_G": clean_scores["normalized_scores"].get("G"),
            "minus30_S": clean_scores["normalized_scores"].get("S"),
            "minus30_N": clean_scores["normalized_scores"].get("N"),
        }
        clean_deltas = {
            key: (
                float(recomputed_clean[key]) - float(frozen_clean[key])
                if recomputed_clean[key] is not None
                and frozen_clean[key] is not None
                else None
            )
            for key in (
                "raw_G",
                "raw_S",
                "raw_N",
                "minus30_G",
                "minus30_S",
                "minus30_N",
            )
        }
        clean_max_abs_delta = max(
            abs(value)
            for value in clean_deltas.values()
            if value is not None
        )
        clean_audio_sha_matches = (
            clean_meta["common_audio_sha256"]
            == frozen_clean["common_audio_sha256"]
        )
        clean_record = {
            "window_s": case["window_s"],
            "source": clean_meta,
            "recomputed": recomputed_clean,
            "frozen_csv": frozen_clean,
            "delta_recomputed_minus_frozen": clean_deltas,
            "max_abs_delta_vs_frozen_csv": clean_max_abs_delta,
            "common_audio_sha256_matches_frozen": clean_audio_sha_matches,
            "normalization": clean_scores["normalization"],
            "raw_scores": clean_scores["original_scores"],
            "minus30_scores": clean_scores["normalized_scores"],
        }
        metrics = score_module.speech_metrics(
            clean_common,
            candidate_common,
            metric_cache,
        )
        weak_speech = score_module.weak_speech_diagnostics(
            clean_common,
            candidate_common,
            input_common,
        )
        candidate_scores.update(
            {
                "pesq": parse_float(metrics.get("pesq")),
                "stoi": parse_float(metrics.get("stoi")),
                "sisdr_db": parse_float(metrics.get("sisdr_db")),
                "weak_speech_energy_ratio_vs_input_db": parse_float(
                    weak_speech.get("energy_ratio_vs_input_db")
                ),
            }
        )
    else:
        size = min(candidate_common.size, input_common.size)
        candidate_energy = float(
            np.mean(candidate_common[:size] ** 2)
        ) if size else 0.0
        input_energy = float(np.mean(input_common[:size] ** 2)) if size else 0.0
        candidate_scores["energy_ratio_vs_input_db"] = 10.0 * math.log10(
            max(candidate_energy, 1e-12) / max(input_energy, 1e-12)
        )

    dpdf_reference = dpdf_reference_for(case, reference_data)
    c4eb_reference = c4eb_data["by_key"].get(
        (case["dataset"], case["case_id"])
    )

    def delta_map(reference: dict[str, Any]) -> dict[str, float | None]:
        result: dict[str, float | None] = {}
        for metric in SCORE_METRICS:
            candidate_value = candidate_scores.get(metric)
            reference_value = reference.get(metric)
            result[metric] = (
                float(candidate_value) - float(reference_value)
                if candidate_value is not None and reference_value is not None
                else None
            )
        return result

    return {
        **parity_record,
        "scoring": {
            "created_utc": utc_now(),
            "score_kind": kind,
            "dynamic_delay_search": False,
            "window_s": case["window_s"],
            "candidate_fixed_delay_samples": 0,
            "candidate_physical_delay_samples": PHYSICAL_DELAY_SAMPLES,
            "candidate_trim_samples": TRIM_SAMPLES,
            "candidate_stage": candidate_scored,
            "input_stage": input_scored,
            "candidate_scores": candidate_scores,
            "speech_metrics": metrics,
            "weak_speech": weak_speech,
            "clean_reference": clean_record,
        },
        "dpdfnet2_reference": dpdf_reference,
        "c4eb_reference": c4eb_reference,
        "deltas": {
            "vs_dpdfnet2": delta_map(dpdf_reference),
            "vs_c4eb": (
                delta_map(c4eb_reference)
                if c4eb_reference is not None
                else {metric: None for metric in SCORE_METRICS}
            ),
        },
    }


def case_value(record: dict[str, Any], source: str, metric: str) -> float | None:
    if source == "candidate":
        return finite_or_none(record["scoring"]["candidate_scores"].get(metric))
    if source == "dpdfnet2":
        return finite_or_none(record["dpdfnet2_reference"].get(metric))
    if source == "c4eb":
        reference = record.get("c4eb_reference")
        return (
            finite_or_none(reference.get(metric))
            if isinstance(reference, dict)
            else None
        )
    if source == "clean":
        clean = record["scoring"]["clean_reference"]
        if clean is None:
            return None
        return finite_or_none(clean["recomputed"].get(metric))
    if source == "clean_frozen":
        clean = record["scoring"]["clean_reference"]
        if clean is None:
            return None
        return finite_or_none(clean["frozen_csv"].get(metric))
    raise ValueError(f"unknown source: {source}")


def key_value(record: dict[str, Any], key: str) -> Any:
    if key == "language":
        return record["language"]
    if key == "snr_db":
        return record["snr_db"]
    if key == "speaker_cluster":
        return record["speaker_cluster"]
    if key == "case_id":
        return record["case_id"]
    raise ValueError(f"unknown grouping key: {key}")


def equal_weighted_stats(
    records: Sequence[dict[str, Any]],
    getter: Callable[[dict[str, Any]], float | None],
    levels: Sequence[str],
) -> dict[str, Any] | None:
    usable = [
        record for record in records if finite_or_none(getter(record)) is not None
    ]
    if not usable:
        return None
    leaf_values: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for record in usable:
        key = tuple(key_value(record, level) for level in levels)
        leaf_values[key].append(float(getter(record)))
    nodes = {
        key: float(np.mean(values)) for key, values in leaf_values.items()
    }
    remaining = list(levels)
    while len(remaining) > 1:
        collapsed: dict[tuple[Any, ...], list[float]] = defaultdict(list)
        for key, value in nodes.items():
            collapsed[key[:-1]].append(value)
        nodes = {
            key: float(np.mean(values)) for key, values in collapsed.items()
        }
        remaining.pop()
    return {
        "mean": float(np.mean(list(nodes.values()))),
        "n_cases": len({record["case_id"] for record in usable}),
        "n_speakers": len(
            {record["speaker_cluster"] for record in usable}
        ),
        "n_strata": len(nodes),
        "equal_weight_levels": list(levels),
    }


def metric_bundle(
    records: Sequence[dict[str, Any]],
    metric: str,
    levels: Sequence[str],
) -> dict[str, Any] | None:
    candidate = equal_weighted_stats(
        records, lambda row: case_value(row, "candidate", metric), levels
    )
    dpdf = equal_weighted_stats(
        records, lambda row: case_value(row, "dpdfnet2", metric), levels
    )
    c4eb = equal_weighted_stats(
        records, lambda row: case_value(row, "c4eb", metric), levels
    )
    delta_dpdf = equal_weighted_stats(
        records,
        lambda row: (
            float(case_value(row, "candidate", metric))
            - float(case_value(row, "dpdfnet2", metric))
            if case_value(row, "candidate", metric) is not None
            and case_value(row, "dpdfnet2", metric) is not None
            else None
        ),
        levels,
    )
    delta_c4eb = equal_weighted_stats(
        records,
        lambda row: (
            float(case_value(row, "candidate", metric))
            - float(case_value(row, "c4eb", metric))
            if case_value(row, "candidate", metric) is not None
            and case_value(row, "c4eb", metric) is not None
            else None
        ),
        levels,
    )
    if candidate is None and dpdf is None and c4eb is None:
        return None
    return {
        "candidate": candidate,
        "dpdfnet2": dpdf,
        "delta_vs_dpdfnet2": delta_dpdf,
        "c4eb": c4eb,
        "delta_vs_c4eb": delta_c4eb,
    }


def clean_bundle(
    records: Sequence[dict[str, Any]], levels: Sequence[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in (
        "raw_G",
        "raw_S",
        "raw_N",
        "minus30_G",
        "minus30_S",
        "minus30_N",
    ):
        result[metric] = {
            "recomputed": equal_weighted_stats(
                records, lambda row, m=metric: case_value(row, "clean", m), levels
            ),
            "frozen_csv": equal_weighted_stats(
                records,
                lambda row, m=metric: case_value(row, "clean_frozen", m),
                levels,
            ),
        }
    return result


def metrics_table(
    records: Sequence[dict[str, Any]], levels: Sequence[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in SCORE_METRICS:
        bundle = metric_bundle(records, metric, levels)
        if bundle is not None:
            result[metric] = bundle
    return result


def build_score_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_dataset[record["dataset"]].append(record)
    result: dict[str, Any] = {}
    for group, dataset, expected, pure_noise in GROUP_SPECS:
        subset = by_dataset[dataset]
        if len(subset) != expected:
            raise ValueError(f"{group}: scored case count {len(subset)} != {expected}")
        if pure_noise:
            result[group] = {
                "cases": len(subset),
                "overall_case_equal": metrics_table(subset, ("case_id",)),
            }
            continue
        if group == "v1speech48":
            overall_levels = ("language", "snr_db", "speaker_cluster")
            result[group] = {
                "cases": len(subset),
                "overall_language_snr_speaker_equal": {
                    "metrics": metrics_table(subset, overall_levels),
                    "clean_reference": clean_bundle(subset, overall_levels),
                },
                "by_language": {},
                "by_language_snr": {},
            }
            for language in sorted({row["language"] for row in subset}):
                language_rows = [
                    row for row in subset if row["language"] == language
                ]
                result[group]["by_language"][language] = {
                    "cases": len(language_rows),
                    "snr_speaker_equal": {
                        "metrics": metrics_table(
                            language_rows, ("snr_db", "speaker_cluster")
                        ),
                        "clean_reference": clean_bundle(
                            language_rows, ("snr_db", "speaker_cluster")
                        ),
                    },
                }
                for snr in sorted(
                    {row["snr_db"] for row in language_rows},
                    key=lambda value: -1e9 if value is None else value,
                ):
                    snr_rows = [
                        row for row in language_rows if row["snr_db"] == snr
                    ]
                    key = f"{language}:{snr:g}" if snr is not None else f"{language}:none"
                    result[group]["by_language_snr"][key] = {
                        "language": language,
                        "snr_db": snr,
                        "cases": len(snr_rows),
                        "speaker_equal": {
                            "metrics": metrics_table(
                                snr_rows, ("speaker_cluster",)
                            ),
                            "clean_reference": clean_bundle(
                                snr_rows, ("speaker_cluster",)
                            ),
                        },
                    }
        else:
            result[group] = {
                "cases": len(subset),
                "overall_speaker_equal": {
                    "metrics": metrics_table(subset, ("speaker_cluster",)),
                    "clean_reference": clean_bundle(
                        subset, ("speaker_cluster",)
                    ),
                },
                "by_language": {},
            }
            for language in sorted({row["language"] for row in subset}):
                language_rows = [
                    row for row in subset if row["language"] == language
                ]
                result[group]["by_language"][language] = {
                    "cases": len(language_rows),
                    "speaker_equal": {
                        "metrics": metrics_table(
                            language_rows, ("speaker_cluster",)
                        ),
                        "clean_reference": clean_bundle(
                            language_rows, ("speaker_cluster",)
                        ),
                    },
                }
    return result


def find_metric(
    summary: dict[str, Any],
    group: str,
    section: str,
    metric: str,
    source: str,
) -> float | None:
    try:
        container = summary[group] if group in summary else summary
        value = container[section]["metrics"][metric][source]
    except (KeyError, TypeError):
        return None
    if not isinstance(value, dict):
        return None
    return finite_or_none(value.get("mean"))


def find_clean(
    summary: dict[str, Any],
    group: str,
    section: str,
    metric: str,
    source: str,
) -> float | None:
    try:
        container = summary[group] if group in summary else summary
        value = container[section]["clean_reference"][metric][source]
    except (KeyError, TypeError):
        return None
    if not isinstance(value, dict):
        return None
    return finite_or_none(value.get("mean"))


def fmt(value: Any, digits: int = 6) -> str:
    number = finite_or_none(value)
    if number is None:
        return "—"
    return f"{number:.{digits}f}"


def md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def build_report(
    parity_summary: dict[str, Any],
    score_summary: dict[str, Any],
    source_identities: dict[str, Any],
) -> str:
    speech = score_summary["v1speech48"]
    speech_section = "overall_language_snr_speaker_equal"
    speech_rows = []
    for metric, label in (
        ("raw_G", "raw G"),
        ("raw_S", "raw S"),
        ("raw_N", "raw N"),
        ("minus30_G", "-30 G"),
        ("minus30_S", "-30 S"),
        ("minus30_N", "-30 N"),
        ("pesq", "PESQ-WB"),
        ("stoi", "STOI"),
        (
            "weak_speech_energy_ratio_vs_input_db",
            "weak energy/input dB",
        ),
    ):
        speech_rows.append(
            [
                label,
                fmt(find_metric(speech, "v1speech48", speech_section, metric, "candidate")),
                fmt(find_metric(speech, "v1speech48", speech_section, metric, "dpdfnet2")),
                fmt(find_metric(speech, "v1speech48", speech_section, metric, "delta_vs_dpdfnet2")),
                fmt(find_metric(speech, "v1speech48", speech_section, metric, "c4eb")),
                fmt(find_metric(speech, "v1speech48", speech_section, metric, "delta_vs_c4eb")),
            ]
        )

    language_rows = []
    for language in sorted(speech["by_language"]):
        for metric in ("raw_G", "raw_S", "raw_N"):
            language_rows.append(
                [
                    language,
                    metric,
                    fmt(
                        find_metric(
                            speech,
                            "v1speech48",
                            f"by_language.{language}",
                            metric,
                            "candidate",
                        )
                    ),
                    fmt(
                        find_metric(
                            speech,
                            "v1speech48",
                            f"by_language.{language}",
                            metric,
                            "delta_vs_dpdfnet2",
                        )
                    ),
                ]
            )

    # The report helper indexes nested language entries directly.
    for language in sorted(speech["by_language"]):
        section = speech["by_language"][language]["snr_speaker_equal"]
        for metric in ("raw_G", "raw_S", "raw_N"):
            bundle = section["metrics"][metric]
            candidate_mean = (
                bundle["candidate"]["mean"]
                if bundle.get("candidate") is not None
                else None
            )
            delta_mean = (
                bundle["delta_vs_dpdfnet2"]["mean"]
                if bundle.get("delta_vs_dpdfnet2") is not None
                else None
            )
            row = next(
                item
                for item in language_rows
                if item[0] == language and item[1] == metric
            )
            row[2] = fmt(candidate_mean)
            row[3] = fmt(delta_mean)

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
                    if metrics["raw_G"].get("candidate")
                    else None
                ),
                fmt(
                    metrics["raw_G"]["delta_vs_dpdfnet2"]["mean"]
                    if metrics["raw_G"].get("delta_vs_dpdfnet2")
                    else None
                ),
                fmt(
                    metrics["raw_S"]["delta_vs_dpdfnet2"]["mean"]
                    if metrics["raw_S"].get("delta_vs_dpdfnet2")
                    else None
                ),
                fmt(
                    metrics["raw_N"]["delta_vs_dpdfnet2"]["mean"]
                    if metrics["raw_N"].get("delta_vs_dpdfnet2")
                    else None
                ),
            ]
        )

    clean_rows = []
    for metric in ("raw_G", "raw_S", "raw_N", "minus30_G", "minus30_S", "minus30_N"):
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

    clean16 = score_summary["v1clean16"]
    clean16_section = "overall_speaker_equal"
    clean16_rows = []
    for metric, label in (
        ("raw_G", "raw G"),
        ("raw_S", "raw S"),
        ("raw_N", "raw N"),
        ("minus30_G", "-30 G"),
        ("minus30_S", "-30 S"),
        ("minus30_N", "-30 N"),
        ("pesq", "PESQ-WB"),
        ("stoi", "STOI"),
        ("weak_speech_energy_ratio_vs_input_db", "weak energy/input dB"),
    ):
        clean16_rows.append(
            [
                label,
                fmt(find_metric(clean16, "v1clean16", clean16_section, metric, "candidate")),
                fmt(find_metric(clean16, "v1clean16", clean16_section, metric, "dpdfnet2")),
                fmt(find_metric(clean16, "v1clean16", clean16_section, metric, "delta_vs_dpdfnet2")),
            ]
        )

    continuous = score_summary["v1continuous4"]
    continuous_section = "overall_speaker_equal"
    continuous_rows = []
    for metric, label in (
        ("raw_G", "raw G"),
        ("raw_S", "raw S"),
        ("raw_N", "raw N"),
        ("minus30_G", "-30 G"),
        ("minus30_S", "-30 S"),
        ("minus30_N", "-30 N"),
        ("pesq", "PESQ-WB"),
        ("stoi", "STOI"),
        ("weak_speech_energy_ratio_vs_input_db", "weak energy/input dB"),
    ):
        continuous_rows.append(
            [
                label,
                fmt(find_metric(continuous, "v1continuous4", continuous_section, metric, "candidate")),
                fmt(find_metric(continuous, "v1continuous4", continuous_section, metric, "dpdfnet2")),
                fmt(find_metric(continuous, "v1continuous4", continuous_section, metric, "delta_vs_dpdfnet2")),
            ]
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
        candidate = record["scoring"]["candidate_scores"][
            "energy_ratio_vs_input_db"
        ]
        reference = record["dpdfnet2_reference"][
            "energy_ratio_vs_input_db"
        ]
        noise_rows.append(
            [
                record["case_id"],
                fmt(candidate),
                fmt(reference),
                fmt(
                    record["deltas"]["vs_dpdfnet2"][
                        "energy_ratio_vs_input_db"
                    ]
                ),
            ]
        )

    group_rows = []
    for group, _, expected, _ in GROUP_SPECS:
        item = parity_summary["groups"][group]
        group_rows.append(
            [
                group,
                f"{item['pass_count']}/{expected}",
                f"{item['max_abs']:.3e}",
                f"{item['max_rmse']:.3e}",
                f"{item['rtf_median']:.4f}",
                f"{item['rtf_p95']:.4f}",
                ",".join(str(value) for value in item["state_bytes_values"]),
            ]
        )

    return "\n".join(
        [
            "# CoFi-Lite stateful stream — frozen 74-case NS-only regression",
            "",
            f"Generated: `{utc_now()}`",
            "",
            "## Scope",
            "",
            "- This is a CoFi-Lite NS model regression on the actual frozen `tx_ns_input` boundary.",
            "- It is **not** a new complete VCP run, not a device real-time measurement, and not a quality ceiling claim.",
            "- The old offline CoFi output is used only as the FP32 stream-parity reference.",
            "- The new stream output is FLOAT; PCM16 `lround` quantization is applied once before model input, matching the old path.",
            "- `enhance_stream` drains and trims the physical 640-sample delay. The aligned file is scored with fixed delay 0; physical delay and trim remain 640 samples.",
            "- No dynamic delay search or historical aggregate was used. Speech48, clean16, continuous4, and guards remain separate.",
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
                group_rows,
            ),
            "",
            f"All cases: `{parity_summary['pass_count']}/{parity_summary['cases']}` below "
            f"`{PARITY_MAX_ABS_THRESHOLD:g}`; global max abs "
            f"`{parity_summary['max_abs']:.12g}`, max RMSE "
            f"`{parity_summary['max_rmse']:.12g}`.",
            "",
            f"Overall RTF median `{parity_summary['rtf_median']:.6f}`, p95 "
            f"`{parity_summary['rtf_p95']:.6f}`. Max post-case process RSS "
            f"`{parity_summary['process_rss_after_max_bytes']}` bytes; max within-case "
            f"RSS delta `{parity_summary['process_rss_delta_max_bytes']}` bytes.",
            "",
            "## Speech48",
            "",
            "Equal weighting is applied over languages, nominal SNR strata, and speaker clusters.",
            "",
            md_table(
                ("metric", "CoFi stream", "DPDFNet", "Δ vs DPDF", "c4eb", "Δ vs c4eb"),
                speech_rows,
            ),
            "",
            "### By language",
            "",
            md_table(("language", "metric", "CoFi stream", "Δ vs DPDF"), language_rows),
            "",
            "### By language and nominal SNR",
            "",
            md_table(
                ("language", "SNR dB", "raw G", "ΔG", "ΔS", "ΔN"),
                snr_rows,
            ),
            "",
            "### Clean reference",
            "",
            md_table(("metric", "recomputed", "frozen CSV"), clean_rows),
            "",
            "## Clean16",
            "",
            md_table(("metric", "CoFi stream", "DPDFNet", "Δ vs DPDF"), clean16_rows),
            "",
            "## Continuous4",
            "",
            md_table(
                ("metric", "CoFi stream", "DPDFNet", "Δ vs DPDF"),
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
            "## Evidence",
            "",
            "- `parity-results.jsonl` — all 74 parity/runtime records.",
            "- `case-results.jsonl` — all 74 parity plus exact scoring/reference records.",
            "- `summary.json` — machine-readable grouped summaries.",
            "- `source-identities.json` — frozen inputs, scorer, model, and reference identities.",
            "",
            "## Provenance highlights",
            "",
            f"- DNSMOS ONNX: `{source_identities['scorer']['dnsmos_model_sha256']}`",
            f"- frozen stage tool: `{source_identities['scorer']['stage_tool_sha256']}`",
            f"- frozen score module: `{source_identities['scorer']['run_and_score_sha256']}`",
            f"- stream adapter source: `{source_identities['stream']['source_sha256']}`",
            "",
            "`deployment_approved=false`; this evidence only establishes the stated NS-only model/stream regression.",
            "",
        ]
    )


def score_phase(
    cases: Sequence[dict[str, Any]],
    output_root: Path,
) -> int:
    parity_path = output_root / "parity-results.jsonl"
    parity_records = read_jsonl(parity_path)
    if len(parity_records) != EXPECTED_TOTAL:
        raise ValueError(
            f"parity results count {len(parity_records)} != {EXPECTED_TOTAL}"
        )
    for path in (
        output_root / "case-results.jsonl",
        output_root / "summary.json",
        output_root / "REPORT.md",
    ):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite scoring output: {path}")

    if sha256_file(STAGE_TOOL) != EXPECTED_STAGE_TOOL_SHA256:
        raise RuntimeError("frozen stage tool SHA changed")
    if sha256_file(RUN_AND_SCORE) != EXPECTED_RUN_AND_SCORE_SHA256:
        raise RuntimeError("frozen run_and_score SHA changed")
    if sha256_file(DNSMOS_MODEL) != EXPECTED_DNSMOS_SHA256:
        raise RuntimeError("DNSMOS ONNX SHA changed")

    reference_data = load_reference_data()
    c4eb_data = load_c4eb_references()
    stage_tool = load_module("vcp_stage_mos", STAGE_TOOL)
    score_module = load_module("run_and_score", RUN_AND_SCORE)
    if getattr(stage_tool, "MODEL_SHA", None) != EXPECTED_DNSMOS_SHA256:
        raise RuntimeError("stage tool DNSMOS model identity changed")

    scoring_root = next_attempt_dir(output_root, "scoring")
    log_path = output_root / "run.log"
    append_log(log_path, f"scoring start scoring_root={scoring_root}")
    scorer = stage_tool.DNSMOS(ASSETS_DIR, scoring_root / "cache/dnsmos")
    try:
        pyloudnorm = stage_tool.import_pyloudnorm()
        loudness_identity: dict[str, Any] = {
            "version": "0.1.1",
            "source": "package_metadata",
        }
    except RuntimeError as error:
        cause = error.__cause__
        if (
            cause is None
            or type(cause).__name__ != "PackageNotFoundError"
        ):
            raise
        deps_dir = Path(PROJECT_ROOT) / "build/ns-next-20261115/deps"
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
            ) from error
        loudness_identity = {
            "version": "0.1.1",
            "source": "unpacked_source_hash_verified",
            "source_dir": str(source_dir),
            "sha256": source_hashes,
        }

    # ScoreCache normally rewrites its in-memory JSON after every score.  The
    # project's no-overwrite rule forbids that pattern, so retain its in-memory
    # cache semantics and persist one exclusive snapshot at the end instead.
    score_module.atomic_json = lambda *args, **kwargs: None
    cache = score_module.ScoreCache(
        scoring_root / "cache-store",
        stage_tool,
        scorer,
        pyloudnorm,
    )
    metric_cache: dict[str, Any] = {}

    parity_by_key = {
        (row["dataset"], row["case_id"]): row for row in parity_records
    }
    if len(parity_by_key) != EXPECTED_TOTAL:
        raise ValueError("parity result keys are not unique")
    scored_records: list[dict[str, Any]] = []
    for index, case in enumerate(cases, 1):
        key = (case["dataset"], case["case_id"])
        parity_record = parity_by_key.get(key)
        if parity_record is None:
            raise ValueError(f"missing parity record for {key}")
        record = score_one_case(
            case,
            parity_record,
            stage_tool,
            score_module,
            cache,
            metric_cache,
            reference_data,
            c4eb_data,
        )
        scored_records.append(record)
        append_log(
            log_path,
            f"score {index}/{len(cases)} {case['group']}/{case['case_id']}",
        )
        print(
            f"SCORE {index}/{len(cases)} {case['group']}/{case['case_id']}",
            flush=True,
        )

    write_jsonl_exclusive(
        scoring_root / "scoring-records.jsonl",
        scored_records,
    )
    write_jsonl_exclusive(output_root / "case-results.jsonl", scored_records)
    parity_summary = summarize_parity(scored_records)
    score_summary = build_score_summary(scored_records)
    clean_max_abs_delta = max(
        float(record["scoring"]["clean_reference"]["max_abs_delta_vs_frozen_csv"])
        for record in scored_records
        if record["scoring"]["clean_reference"] is not None
    )
    clean_sha_mismatches = sum(
        not bool(
            record["scoring"]["clean_reference"][
                "common_audio_sha256_matches_frozen"
            ]
        )
        for record in scored_records
        if record["scoring"]["clean_reference"] is not None
    )
    source_identities = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "stream": {
            "source_path": str(STREAM_SOURCE),
            "source_sha256": sha256_file(STREAM_SOURCE),
            "model_adapter_path": str(
                COFILITE_ROOT / "cofi_infer.py"
            ),
            "model_adapter_sha256": sha256_file(
                COFILITE_ROOT / "cofi_infer.py"
            ),
            "model_source_path": str(
                COFILITE_ROOT / "upstream/model/cofilite.py"
            ),
            "model_source_sha256": sha256_file(
                COFILITE_ROOT / "upstream/model/cofilite.py"
            ),
            "checkpoint_path": str(
                COFILITE_ROOT / "upstream/checkpoint/model_on_dns3.tar"
            ),
            "checkpoint_sha256": sha256_file(
                COFILITE_ROOT / "upstream/checkpoint/model_on_dns3.tar"
            ),
        },
        "scorer": {
            "stage_tool_path": str(STAGE_TOOL),
            "stage_tool_sha256": sha256_file(STAGE_TOOL),
            "run_and_score_path": str(RUN_AND_SCORE),
            "run_and_score_sha256": sha256_file(RUN_AND_SCORE),
            "ecnr_perceptual_path": str(
                STAGE_TOOL.parent / "ecnr_perceptual.py"
            ),
            "ecnr_perceptual_sha256": sha256_file(
                STAGE_TOOL.parent / "ecnr_perceptual.py"
            ),
            "dnsmos_model_path": str(DNSMOS_MODEL),
            "dnsmos_model_sha256": sha256_file(DNSMOS_MODEL),
            "assets_dir": str(ASSETS_DIR),
            "score_rate": SAMPLE_RATE,
            "normalization_lufs": -30.0,
            "raw_and_minus30": True,
            "pyloudnorm": loudness_identity,
        },
        "references": {
            "dpdfnet2": reference_data["source_sha256"],
            "c4eb": c4eb_data["sources"],
            "cofilite_receipts": {
                group: {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
                for group, path in RECEIPTS.items()
            },
        },
        "protocol": {
            "case_count": EXPECTED_TOTAL,
            "speech_count": EXPECTED_SPEECH,
            "noise_count": EXPECTED_NOISE,
            "ns_only": True,
            "full_vcp": False,
            "device_realtime": False,
            "stream_chunk_samples": STREAM_CHUNK_SIZE,
            "physical_delay_samples": PHYSICAL_DELAY_SAMPLES,
            "trim_samples": TRIM_SAMPLES,
            "score_fixed_delay_samples": 0,
            "dynamic_delay_search": False,
            "same_window_s": True,
            "pcm16_lround_once": True,
            "torch_inference_mode": True,
            "processes_max": 4,
            "model_loaded_once_per_worker": True,
            "shared_model_threads": False,
            "gpu_used": False,
            "speech_weighting": "language equal; SNR equal within language; speaker equal within stratum",
            "noise_mos": False,
            "noise_metric": "energy ratio versus actual NS input on the same window",
        },
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "status": "complete",
        "scope": {
            "name": "CoFi-Lite stateful stream NS-only regression",
            "full_vcp": False,
            "device_realtime": False,
            "deployment_approved": False,
            "notes": [
                "frozen actual tx_ns_input cases only",
                "new stream output scored independently; old CoFi scores were not reused",
                "physical delay and trim are 640 samples; scoring delay is 0 on the aligned file",
                "speech48, clean16, continuous4, and guards are never pooled into one MOS mean",
            ],
        },
        "counts": {
            "total": len(scored_records),
            "speech": sum(
                not bool(row["pure_noise"]) for row in scored_records
            ),
            "noise": sum(bool(row["pure_noise"]) for row in scored_records),
            "groups": {
                group: sum(
                    row["group"] == group for row in scored_records
                )
                for group, _, _, _ in GROUP_SPECS
            },
        },
        "parity": parity_summary,
        "clean_reference_validation": {
            "max_abs_delta_vs_frozen_csv": clean_max_abs_delta,
            "common_audio_sha256_mismatches": clean_sha_mismatches,
            "exact_within_1e-12": (
                clean_max_abs_delta <= 1e-12 and clean_sha_mismatches == 0
            ),
        },
        "scores": score_summary,
        "source_identities": source_identities,
    }
    report = build_report(
        parity_summary,
        {**score_summary, "_records": scored_records},
        source_identities,
    )
    write_json_exclusive(output_root / "summary.json", summary)
    write_json_exclusive(output_root / "source-identities.json", source_identities)
    with (output_root / "REPORT.md").open("x", encoding="utf-8") as stream:
        stream.write(report)
    write_json_exclusive(
        scoring_root / "score-cache.snapshot.json",
        {"schema_version": 1, "values": cache.values},
    )
    append_log(
        log_path,
        "scoring complete "
        f"parity={parity_summary['pass_count']}/{parity_summary['cases']} "
        f"clean_max_abs_delta={clean_max_abs_delta:.12g} "
        f"clean_sha_mismatches={clean_sha_mismatches}",
    )
    print(
        json.dumps(
            json_safe(
                {
                    "status": "complete",
                    "parity": parity_summary,
                    "clean_reference_validation": summary[
                        "clean_reference_validation"
                    ],
                    "case_results": str(output_root / "case-results.jsonl"),
                    "summary": str(output_root / "summary.json"),
                    "report": str(output_root / "REPORT.md"),
                }
            ),
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def finalize_existing_phase(output_root: Path) -> int:
    """Build missing final summaries from an already complete scored JSONL."""
    case_results_path = output_root / "case-results.jsonl"
    scored_records = read_jsonl(case_results_path)
    if len(scored_records) != EXPECTED_TOTAL:
        raise ValueError(
            f"case results count {len(scored_records)} != {EXPECTED_TOTAL}"
        )
    for path in (
        output_root / "summary.json",
        output_root / "REPORT.md",
        output_root / "source-identities.json",
    ):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite finalized output: {path}")

    if sha256_file(STAGE_TOOL) != EXPECTED_STAGE_TOOL_SHA256:
        raise RuntimeError("frozen stage tool SHA changed")
    if sha256_file(RUN_AND_SCORE) != EXPECTED_RUN_AND_SCORE_SHA256:
        raise RuntimeError("frozen run_and_score SHA changed")
    if sha256_file(DNSMOS_MODEL) != EXPECTED_DNSMOS_SHA256:
        raise RuntimeError("DNSMOS ONNX SHA changed")
    reference_data = load_reference_data()
    c4eb_data = load_c4eb_references()
    score_module = load_module("run_and_score_finalize", RUN_AND_SCORE)

    deps_dir = Path(PROJECT_ROOT) / "build/ns-next-20261115/deps"
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
    loudness_identity = {
        "version": "0.1.1",
        "source": "unpacked_source_hash_verified",
        "source_dir": str(source_dir),
        "sha256": source_hashes,
    }

    parity_summary = summarize_parity(scored_records)
    score_summary = build_score_summary(scored_records)
    clean_max_abs_delta = max(
        float(record["scoring"]["clean_reference"]["max_abs_delta_vs_frozen_csv"])
        for record in scored_records
        if record["scoring"]["clean_reference"] is not None
    )
    clean_sha_mismatches = sum(
        not bool(
            record["scoring"]["clean_reference"][
                "common_audio_sha256_matches_frozen"
            ]
        )
        for record in scored_records
        if record["scoring"]["clean_reference"] is not None
    )
    source_identities = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "stream": {
            "source_path": str(STREAM_SOURCE),
            "source_sha256": sha256_file(STREAM_SOURCE),
            "model_adapter_path": str(COFILITE_ROOT / "cofi_infer.py"),
            "model_adapter_sha256": sha256_file(
                COFILITE_ROOT / "cofi_infer.py"
            ),
            "model_source_path": str(
                COFILITE_ROOT / "upstream/model/cofilite.py"
            ),
            "model_source_sha256": sha256_file(
                COFILITE_ROOT / "upstream/model/cofilite.py"
            ),
            "checkpoint_path": str(
                COFILITE_ROOT / "upstream/checkpoint/model_on_dns3.tar"
            ),
            "checkpoint_sha256": sha256_file(
                COFILITE_ROOT / "upstream/checkpoint/model_on_dns3.tar"
            ),
        },
        "scorer": {
            "stage_tool_path": str(STAGE_TOOL),
            "stage_tool_sha256": sha256_file(STAGE_TOOL),
            "run_and_score_path": str(RUN_AND_SCORE),
            "run_and_score_sha256": sha256_file(RUN_AND_SCORE),
            "ecnr_perceptual_path": str(
                STAGE_TOOL.parent / "ecnr_perceptual.py"
            ),
            "ecnr_perceptual_sha256": sha256_file(
                STAGE_TOOL.parent / "ecnr_perceptual.py"
            ),
            "dnsmos_model_path": str(DNSMOS_MODEL),
            "dnsmos_model_sha256": sha256_file(DNSMOS_MODEL),
            "assets_dir": str(ASSETS_DIR),
            "score_rate": SAMPLE_RATE,
            "normalization_lufs": -30.0,
            "raw_and_minus30": True,
            "pyloudnorm": loudness_identity,
        },
        "references": {
            "dpdfnet2": reference_data["source_sha256"],
            "c4eb": c4eb_data["sources"],
            "cofilite_receipts": {
                group: {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
                for group, path in RECEIPTS.items()
            },
        },
        "protocol": {
            "case_count": EXPECTED_TOTAL,
            "speech_count": EXPECTED_SPEECH,
            "noise_count": EXPECTED_NOISE,
            "ns_only": True,
            "full_vcp": False,
            "device_realtime": False,
            "stream_chunk_samples": STREAM_CHUNK_SIZE,
            "physical_delay_samples": PHYSICAL_DELAY_SAMPLES,
            "trim_samples": TRIM_SAMPLES,
            "score_fixed_delay_samples": 0,
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
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "status": "complete",
        "scope": {
            "name": "CoFi-Lite stateful stream NS-only regression",
            "full_vcp": False,
            "device_realtime": False,
            "deployment_approved": False,
            "notes": [
                "frozen actual tx_ns_input cases only",
                "new stream output scored independently; old CoFi scores were not reused",
                "physical delay and trim are 640 samples; scoring delay is 0 on the aligned file",
                "speech48, clean16, continuous4, and guards are never pooled into one MOS mean",
            ],
        },
        "counts": {
            "total": len(scored_records),
            "speech": sum(
                not bool(row["pure_noise"]) for row in scored_records
            ),
            "noise": sum(bool(row["pure_noise"]) for row in scored_records),
            "groups": {
                group: sum(
                    row["group"] == group for row in scored_records
                )
                for group, _, _, _ in GROUP_SPECS
            },
        },
        "parity": parity_summary,
        "clean_reference_validation": {
            "max_abs_delta_vs_frozen_csv": clean_max_abs_delta,
            "common_audio_sha256_mismatches": clean_sha_mismatches,
            "exact_within_1e-12": (
                clean_max_abs_delta <= 1e-12 and clean_sha_mismatches == 0
            ),
        },
        "scores": score_summary,
        "source_identities": source_identities,
    }
    report = build_report(
        parity_summary,
        {**score_summary, "_records": scored_records},
        source_identities,
    )
    write_json_exclusive(output_root / "source-identities.json", source_identities)
    write_json_exclusive(output_root / "summary.json", summary)
    with (output_root / "REPORT.md").open("x", encoding="utf-8") as stream:
        stream.write(report)
    append_log(
        output_root / "run.log",
        "finalize complete "
        f"cases={len(scored_records)} "
        f"parity={parity_summary['pass_count']}/{parity_summary['cases']} "
        f"clean_max_abs_delta={clean_max_abs_delta:.12g} "
        f"clean_sha_mismatches={clean_sha_mismatches}",
    )
    print(
        json.dumps(
            json_safe(
                {
                    "status": "complete",
                    "parity": parity_summary,
                    "clean_reference_validation": summary[
                        "clean_reference_validation"
                    ],
                    "summary": str(output_root / "summary.json"),
                    "report": str(output_root / "REPORT.md"),
                }
            ),
            sort_keys=True,
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
        choices=("parity", "score", "finalize", "all"),
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(DEFAULT_OUTPUT_ROOT),
    )
    parser.add_argument("--jobs", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.jobs < 1 or args.jobs > 4:
        raise SystemExit("--jobs must be between 1 and 4")
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    if args.phase == "parity":
        return parity_phase(cases, output_root, args.jobs)
    if args.phase == "score":
        return score_phase(cases, output_root)
    if args.phase == "finalize":
        return finalize_existing_phase(output_root)
    status = parity_phase(cases, output_root, args.jobs)
    if status != 0:
        print(
            "parity threshold failed; parity evidence was retained and scoring "
            "was not started by --phase all",
            file=sys.stderr,
            flush=True,
        )
        return status
    return score_phase(cases, output_root)


if __name__ == "__main__":
    raise SystemExit(main())
