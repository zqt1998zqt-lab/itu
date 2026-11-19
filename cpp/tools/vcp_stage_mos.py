#!/usr/bin/env python3
"""Score one VCP run's diagnostic taps with DNSMOS P.835.

The runner executes the engine once, then consumes the ``pipeline_stages``
manifest written by that same run.  Every channel is scored independently at
its actual level and after a common 16 kHz loudness normalization.  This tool
does not train, upload audio, fit delays, or synthesize missing stages.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Iterable, Sequence

import librosa
import numpy as np
import soundfile as sf


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
from ecnr_perceptual import (  # noqa: E402
    ALLOWED_ROLES,
    DEFAULT_ASSETS,
    MODEL_SHA,
    RATE as SCORE_RATE,
    SCOPE as SCORE_SCOPE,
    DNSMOS,
)


SCHEMA_VERSION = 1
ALIASES = {"G": "OVRL", "S": "SIG", "N": "BAK"}
FORBIDDEN_ENGINE_OVERRIDES = frozenset(
    ("--mic", "--echo", "--rx-in", "--out", "--profile", "--diagnostic-stages")
)
LUFS_BLOCK_SIZE = 0.4
LUFS_CORRECTION_TOLERANCE = 1e-6
LUFS_REREAD_TOLERANCE = 1e-3
LUFS_MAX_CORRECTIONS = 8
ENGINE_TIMEOUT_SECONDS = 300
PEAK_PREP_CEILING = 0.7
CSV_COLUMNS = (
    "direction",
    "order",
    "stage",
    "channel",
    "raw_G",
    "raw_S",
    "raw_N",
    "minus30_G",
    "minus30_S",
    "minus30_N",
    "delta_raw_G",
    "delta_raw_S",
    "delta_raw_N",
    "delta_minus30_G",
    "delta_minus30_S",
    "delta_minus30_N",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    audio = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(str(audio.shape).encode() + audio.tobytes()).hexdigest()


def save_json(path: Path, document: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _finite(value: Any, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def split_engine_args(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    values = list(argv)
    separators = [index for index, value in enumerate(values) if value == "--"]
    if not separators:
        return values, []
    if len(separators) != 1:
        raise ValueError("use exactly one '--' before engine arguments")
    index = separators[0]
    return values[:index], values[index + 1 :]


def reject_engine_overrides(engine_args: Sequence[str]) -> None:
    for token in engine_args:
        name = token.split("=", 1)[0]
        if name in FORBIDDEN_ENGINE_OVERRIDES:
            raise ValueError(
                f"engine passthrough may not override {name}; use the runner argument instead"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, allow_abbrev=False, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cli", type=Path, required=True, help="itu_audio_cli executable")
    parser.add_argument("--mic", type=Path, help="microphone capture WAV")
    parser.add_argument("--rx-in", type=Path, help="raw downlink WAV")
    parser.add_argument("--echo", type=Path, help="actual playback-reference WAV")
    parser.add_argument("--profile", default="vcp", help="engine profile (default: vcp)")
    parser.add_argument("--output-dir", type=Path, required=True, help="new output directory")
    parser.add_argument("--kind", choices=sorted(ALLOWED_ROLES), default="near_only")
    parser.add_argument(
        "--window",
        nargs=2,
        type=float,
        metavar=("START", "END"),
        help="near-speech window in input-timeline seconds",
    )
    parser.add_argument("--assets-dir", type=Path, help="pinned DNSMOS asset directory")
    parser.add_argument(
        "--loudness-target", type=float, default=-30.0, help="normalized integrated LUFS (default: -30)"
    )
    return parser


def resolve_input_path(path: Path | None, field: str) -> Path | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"{field} does not exist or is not a file: {resolved}")
    return resolved


def validate_arguments(args: argparse.Namespace, engine_args: Sequence[str]) -> None:
    reject_engine_overrides(engine_args)
    args.cli = resolve_input_path(args.cli, "--cli")
    args.mic = resolve_input_path(args.mic, "--mic")
    args.rx_in = resolve_input_path(args.rx_in, "--rx-in")
    args.echo = resolve_input_path(args.echo, "--echo")
    if args.rx_in is not None and args.echo is not None:
        raise ValueError("--rx-in and --echo are mutually exclusive")
    if args.mic is None and args.rx_in is None:
        raise ValueError("at least one of --mic or --rx-in is required")
    if args.echo is not None and args.mic is None:
        raise ValueError("--echo requires --mic")
    if not args.profile or args.profile.startswith("-"):
        raise ValueError("--profile must name an engine profile")
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output_dir.exists():
        raise ValueError(f"--output-dir must be new; already exists: {args.output_dir}")
    if args.window is not None:
        start, end = (_finite(value, "--window") for value in args.window)
        if start < 0 or end <= start:
            raise ValueError("--window requires 0 <= START < END")
        args.window = [start, end]
    args.loudness_target = _finite(args.loudness_target, "--loudness-target")
    if args.loudness_target > 0:
        raise ValueError("--loudness-target must be <= 0 dB LUFS")
    if args.assets_dir is not None:
        args.assets_dir = args.assets_dir.expanduser().resolve()
        if not args.assets_dir.is_dir():
            raise ValueError(f"--assets-dir is not a directory: {args.assets_dir}")


def input_paths(args: argparse.Namespace) -> dict[str, Path]:
    values = {"cli": args.cli, "mic": args.mic, "rx_in": args.rx_in, "echo": args.echo}
    return {name: path for name, path in values.items() if path is not None}


def hash_paths(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path): sha256_file(path) for path in paths}


def verify_hashes(expected: dict[str, str], field: str) -> dict[str, str]:
    current = hash_paths(Path(path) for path in expected)
    changed = [path for path, digest in expected.items() if current.get(path) != digest]
    if changed:
        raise RuntimeError(f"{field} changed during measurement: {', '.join(changed)}")
    return current


def validate_input_wavs(
    args: argparse.Namespace, input_rate: int, input_samples: int
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in ("mic", "rx_in", "echo"):
        path = getattr(args, name)
        if path is None:
            continue
        info = sf.info(path)
        if info.samplerate != input_rate:
            raise ValueError(
                f"{name} sample rate {info.samplerate} differs from metadata {input_rate}"
            )
        if info.frames != input_samples:
            raise ValueError(
                f"{name} samples {info.frames} differ from metadata {input_samples}"
            )
        result[name] = {
            "path": str(path),
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "samples_per_channel": info.frames,
        }
    return result


def validate_input_stage_copy(stage: dict[str, Any], source_path: Path) -> dict[str, Any]:
    stage_path = Path(stage["file"])
    stage_audio, stage_rate = sf.read(stage_path, dtype="float64", always_2d=True)
    source_audio, source_rate = sf.read(source_path, dtype="float64", always_2d=True)
    if stage_rate != source_rate or stage_rate != stage["sample_rate"]:
        raise ValueError(f"source stage {stage['id']} sample rate does not match its input")
    if stage_audio.shape != source_audio.shape:
        raise ValueError(f"source stage {stage['id']} shape does not match its input")
    if stage_audio.shape != (stage["samples_per_channel"], stage["channels"]):
        raise ValueError(f"source stage {stage['id']} shape does not match metadata")
    expected = source_audio.astype(np.float32).astype(np.float64)
    exact = bool(np.array_equal(stage_audio, expected))
    max_error = float(np.max(np.abs(stage_audio - expected))) if stage_audio.size else 0.0
    if not exact:
        raise ValueError(
            f"source stage {stage['id']} is not the exact float32 source PCM "
            f"(max abs error {max_error})"
        )
    return {
        "stage": stage["id"],
        "source_path": str(source_path.resolve()),
        "source_sha256": sha256_file(source_path),
        "stage_path": str(stage_path.resolve()),
        "stage_sha256": sha256_file(stage_path),
        "source_kind": (
            "original_path" if stage_path.resolve() == source_path.resolve() else "float32_run_copy"
        ),
        "sample_rate": stage_rate,
        "channels": int(stage_audio.shape[1]),
        "samples_per_channel": int(stage_audio.shape[0]),
        "samples_exact_after_float32_conversion": exact,
        "max_abs_sample_error": max_error,
    }


def declared_duration(metadata: dict[str, Any]) -> float | None:
    for key in ("input_duration_s", "channel_duration_s", "input_duration_seconds"):
        if key in metadata:
            return _finite(metadata[key], key)
    return None


def normalize_stage_entries(
    metadata: dict[str, Any],
    output_dir: Path,
    input_files: dict[str, Path | None] | None = None,
) -> list[dict[str, Any]]:
    entries = metadata.get("pipeline_stages")
    if not isinstance(entries, list) or not entries:
        raise ValueError("stage metadata must contain a non-empty pipeline_stages array")
    normalized: list[dict[str, Any]] = []
    output_root = output_dir.resolve()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"pipeline_stages[{index}] must be an object")
        stage_id = _string(entry.get("id"), f"pipeline_stages[{index}].id")
        direction = _string(entry.get("direction"), f"pipeline_stages[{index}].direction")
        if direction not in ("tx", "rx"):
            raise ValueError(f"pipeline_stages[{index}].direction must be tx or rx")
        order = _integer(entry.get("order"), f"pipeline_stages[{index}].order")
        file_value = _string(entry.get("file"), f"pipeline_stages[{index}].file")
        file_path = Path(file_value)
        if not file_path.is_absolute():
            raise ValueError(f"pipeline_stages[{index}].file must be absolute: {file_value}")
        file_path = file_path.resolve()
        expected_input = None
        if stage_id == "tx_input":
            expected_input = (input_files or {}).get("tx_input")
            expected_direction = "tx"
        elif stage_id == "rx_input":
            expected_input = (input_files or {}).get("rx_input")
            expected_direction = "rx"
        else:
            expected_direction = None
        if expected_direction is not None:
            if direction != expected_direction:
                raise ValueError(f"pipeline_stages[{index}].{stage_id} has wrong direction")
            if expected_input is None:
                raise ValueError(f"pipeline_stages[{index}].{stage_id} has no matching CLI input")
            if file_path != expected_input.resolve() and not file_path.is_relative_to(output_root):
                raise ValueError(
                    f"pipeline_stages[{index}].file must match the {stage_id} CLI input "
                    "or be a source copy inside this run"
                )
        elif not file_path.is_relative_to(output_root):
            raise ValueError(f"pipeline_stages[{index}].file is outside this run: {file_path}")
        sample_rate = _integer(
            entry.get("sample_rate"), f"pipeline_stages[{index}].sample_rate", minimum=1
        )
        channels = _integer(
            entry.get("channels"), f"pipeline_stages[{index}].channels", minimum=1
        )
        capture_channels = _integer(
            entry.get("capture_channels", channels),
            f"pipeline_stages[{index}].capture_channels",
        )
        if capture_channels > channels:
            raise ValueError(
                f"pipeline_stages[{index}].capture_channels exceeds file channels"
            )
        samples = _integer(
            entry.get("samples_per_channel"),
            f"pipeline_stages[{index}].samples_per_channel",
            minimum=1,
        )
        delay = _integer(
            entry.get("fixed_delay_samples"),
            f"pipeline_stages[{index}].fixed_delay_samples",
        )
        if delay >= samples:
            raise ValueError(f"pipeline_stages[{index}] delay consumes the whole tap")
        enabled = _boolean(entry.get("enabled"), f"pipeline_stages[{index}].enabled")
        capture_available = _boolean(
            entry.get("capture_available", True),
            f"pipeline_stages[{index}].capture_available",
        )
        diagnostic = _boolean(
            entry.get("diagnostic", stage_id == "tx_ns_selected"),
            f"pipeline_stages[{index}].diagnostic",
        )
        if capture_available and capture_channels == 0:
            raise ValueError(
                f"pipeline_stages[{index}] is capture-available but has no physical channel"
            )
        measurement_domain = _string(
            entry.get("measurement_domain"), f"pipeline_stages[{index}].measurement_domain"
        )
        channel_policy = _string(
            entry.get("channel_policy"), f"pipeline_stages[{index}].channel_policy"
        )
        normalized.append(
            {
                "id": stage_id,
                "direction": direction,
                "order": order,
                "file": str(file_path),
                "sample_rate": sample_rate,
                "channels": channels,
                "capture_channels": capture_channels,
                "samples_per_channel": samples,
                "fixed_delay_samples": delay,
                "enabled": enabled,
                "capture_available": capture_available,
                "diagnostic": diagnostic,
                "measurement_domain": measurement_domain,
                "channel_policy": channel_policy,
                "metadata_index": index,
            }
        )
    return sorted(
        normalized,
        key=lambda row: (0 if row["direction"] == "tx" else 1, row["order"], row["id"], row["metadata_index"]),
    )


def resample_for_scoring(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("resample_for_scoring requires an explicit mono channel")
    if sample_rate == SCORE_RATE:
        return values.copy()
    return librosa.resample(
        values,
        orig_sr=sample_rate,
        target_sr=SCORE_RATE,
        res_type="soxr_hq",
        fix=True,
        scale=False,
    )


def audio_extrema(samples: np.ndarray) -> dict[str, Any]:
    values = np.asarray(samples, dtype=np.float64)
    if not len(values):
        return {
            "peak": None,
            "clipped_samples": None,
            "finite": False,
            "nonfinite_samples": 0,
        }
    finite_mask = np.isfinite(values)
    finite = bool(np.all(finite_mask))
    if not finite:
        return {
            "peak": None,
            "clipped_samples": None,
            "finite": False,
            "nonfinite_samples": int(values.size - np.count_nonzero(finite_mask)),
        }
    magnitude = np.abs(values)
    return {
        "peak": float(np.max(magnitude)),
        "clipped_samples": int(np.count_nonzero(magnitude > 1.0)),
        "finite": True,
        "nonfinite_samples": 0,
    }


def validate_score_numbers(score: dict[str, Any], field: str) -> None:
    def visit(value: Any, location: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{location}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{location}[{index}]")
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(float(value)):
                raise ValueError(f"non-finite score at {location}")

    for key in ("raw", "poly", "hops"):
        visit(score.get(key), f"{field}.{key}")
    for alias in ALIASES:
        visit(score.get(alias), f"{field}.{alias}")


def score_record(result: dict[str, Any]) -> dict[str, Any]:
    available = bool(result.get("available"))
    if available and result.get("model_sha256") != MODEL_SHA:
        raise ValueError("DNSMOS model hash changed from the pinned hash")
    poly = result.get("poly") if available else None
    record = {
        "available": available,
        "reason": result.get("reason"),
        "G": float(poly["OVRL"]) if poly is not None else None,
        "S": float(poly["SIG"]) if poly is not None else None,
        "N": float(poly["BAK"]) if poly is not None else None,
        "raw": result.get("raw") if available else None,
        "poly": poly,
        "hops": result.get("hops") if available else None,
        "short_repeat": bool(result.get("short_repeat")) if available else None,
        "doubling_count": result.get("doubling_count") if available else None,
        "resampled_original_samples": (
            result.get("resampled_original_samples") if available else None
        ),
        "aliases": dict(ALIASES),
        "meta": result,
    }
    validate_score_numbers(record, "score")
    return record


def unavailable_score(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
        "G": None,
        "S": None,
        "N": None,
        "raw": None,
        "poly": None,
        "hops": None,
        "short_repeat": None,
        "doubling_count": None,
        "resampled_original_samples": None,
        "aliases": dict(ALIASES),
        "meta": None,
    }


def aggregate_stage_availability(stage: dict[str, Any]) -> dict[str, Any]:
    channels = stage["channels_scored"]
    original_available = bool(channels) and all(
        channel["original_scores"]["available"] for channel in channels
    )
    normalized_available = bool(channels) and all(
        channel["normalized_scores"]["available"] for channel in channels
    )
    unavailable_scores = []
    for channel in channels:
        for score_name in ("original_scores", "normalized_scores"):
            score = channel[score_name]
            if not score["available"]:
                unavailable_scores.append(
                    {
                        "channel": channel["channel"],
                        "score": score_name.removesuffix("_scores"),
                        "reason": score.get("reason") or "unavailable",
                    }
                )
    expected = len(channels) * 2
    available_count = expected - len(unavailable_scores)
    available = original_available and normalized_available
    reason = None
    if not available:
        reason = (
            unavailable_scores[0]["reason"]
            if unavailable_scores
            else stage.get("reason") or "no_physical_channels"
        )
    stage.update(
        {
            "available": available,
            "original_available": original_available,
            "normalized_available": normalized_available,
            "partial": 0 < available_count < expected,
            "available_score_count": available_count,
            "expected_score_count": expected,
            "unavailable_scores": unavailable_scores,
            "reason": reason,
        }
    )
    return stage


def import_pyloudnorm() -> Any:
    try:
        installed = package_version("pyloudnorm")
    except PackageNotFoundError as error:
        raise RuntimeError("pyloudnorm==0.1.1 is required for normalized MOS") from error
    if installed != "0.1.1":
        raise RuntimeError(f"pyloudnorm 0.1.1 is required; found {installed}")
    import pyloudnorm

    return pyloudnorm


def measure_lufs(pyloudnorm: Any, samples: np.ndarray) -> float:
    meter = pyloudnorm.Meter(SCORE_RATE, filter_class="K-weighting", block_size=LUFS_BLOCK_SIZE)
    return float(meter.integrated_loudness(np.asarray(samples, dtype=np.float64)))


def normalize_to_target(
    samples: np.ndarray,
    target_lufs: float,
    pyloudnorm: Any,
    output_path: Path,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    source = np.asarray(samples, dtype=np.float64)
    meta: dict[str, Any] = {
        "available": False,
        "reason": None,
        "method": "pyloudnorm K-weighting, 0.4 s blocks, peak prep 0.7/peak, constant gain",
        "target_lufs": target_lufs,
        "source_peak": audio_extrema(source)["peak"],
        "source_clipped_samples": audio_extrema(source)["clipped_samples"],
        "peak_prep_gain": None,
        "peak_prep_ceiling": PEAK_PREP_CEILING,
        "correction_iterations": 0,
        "correction_tolerance_db": LUFS_CORRECTION_TOLERANCE,
        "correction_measurements_lufs": [],
        "applied_gain_db": None,
        "measured_lufs_before": None,
        "original_measured_lufs": None,
        "original_measurement_finite": None,
        "prepared_measured_lufs": None,
        "measured_lufs_after": None,
        "constant_gain_verification_max_abs_error": None,
        "output_peak": None,
        "output_clipped_samples": None,
        "reread_max_abs_error": None,
        "reread_error_bound": None,
        "reread_measured_lufs": None,
        "reread_tolerance_db": LUFS_REREAD_TOLERANCE,
        "output_path": None,
        "output_sha256": None,
        "output_shape": None,
    }
    if not len(source) or not np.all(np.isfinite(source)):
        meta["reason"] = "empty_or_nonfinite_audio"
        return None, meta
    if len(source) < int(math.ceil(LUFS_BLOCK_SIZE * SCORE_RATE)):
        meta["reason"] = "too_short_for_lufs"
        return None, meta
    peak = float(np.max(np.abs(source)))
    if peak <= 1e-10 or float(np.sqrt(np.mean(source * source))) <= 1e-10:
        meta["reason"] = "silent_audio"
        return None, meta
    original_measured = measure_lufs(pyloudnorm, source)
    meta["original_measurement_finite"] = math.isfinite(original_measured)
    meta["original_measured_lufs"] = original_measured if math.isfinite(original_measured) else None
    meta["measured_lufs_before"] = meta["original_measured_lufs"]
    prep_gain = PEAK_PREP_CEILING / peak
    prepared = source * prep_gain
    meta["peak_prep_gain"] = prep_gain
    measured = measure_lufs(pyloudnorm, prepared)
    meta["prepared_measured_lufs"] = measured if math.isfinite(measured) else None
    if not math.isfinite(measured) or measured < -70.0:
        meta["reason"] = "silent_or_below_loudness_gate"
        return None, meta
    working = prepared.copy()
    cumulative_gain = 1.0
    measurements = []
    for _ in range(LUFS_MAX_CORRECTIONS):
        gain = 10.0 ** ((target_lufs - measured) / 20.0)
        working *= gain
        cumulative_gain *= gain
        measured = measure_lufs(pyloudnorm, working)
        measurements.append(measured if math.isfinite(measured) else None)
        meta["correction_iterations"] += 1
        if math.isfinite(measured) and abs(measured - target_lufs) <= LUFS_CORRECTION_TOLERANCE:
            break
    meta["correction_measurements_lufs"] = measurements
    if not math.isfinite(measured) or abs(measured - target_lufs) > LUFS_CORRECTION_TOLERANCE:
        raise RuntimeError(
            f"LUFS correction did not converge within {LUFS_MAX_CORRECTIONS} passes: "
            f"measured={measured}, target={target_lufs}"
        )
    extrema = audio_extrema(working)
    reconstruction_error = float(np.max(np.abs(working - prepared * cumulative_gain)))
    meta["constant_gain_verification_max_abs_error"] = reconstruction_error
    if reconstruction_error > 1e-12:
        raise RuntimeError("normalization is not a single constant gain over the prepared signal")
    meta.update(
        {
            "available": True,
            "reason": None,
            "applied_gain_db": 20.0 * math.log10(cumulative_gain * meta["peak_prep_gain"]),
            "measured_lufs_after": measured,
            "output_peak": extrema["peak"],
            "output_clipped_samples": extrema["clipped_samples"],
        }
    )
    if extrema["peak"] >= 1.0 or extrema["clipped_samples"]:
        meta["reason"] = "normalization_clipping"
        meta["available"] = False
        return None, meta
    sf.write(output_path, working.reshape(-1, 1), SCORE_RATE, format="WAV", subtype="FLOAT")
    reread, reread_rate = sf.read(output_path, dtype="float64", always_2d=True)
    if reread_rate != SCORE_RATE or reread.shape != (len(working), 1):
        raise RuntimeError("normalized FLOAT WAV changed shape or sample rate on reread")
    reread_error = float(np.max(np.abs(reread[:, 0] - working)))
    reread_bound = float(np.finfo(np.float32).eps * np.max(np.abs(working)))
    meta["reread_max_abs_error"] = reread_error
    meta["reread_error_bound"] = reread_bound
    if reread_error > reread_bound:
        raise RuntimeError(
            f"normalized FLOAT WAV reread error {reread_error} exceeds {reread_bound}"
        )
    reread_lufs = measure_lufs(pyloudnorm, reread[:, 0])
    if not math.isfinite(reread_lufs) or abs(reread_lufs - target_lufs) >= LUFS_REREAD_TOLERANCE:
        raise RuntimeError(
            f"normalized WAV reread loudness is outside {LUFS_REREAD_TOLERANCE} dB: {reread_lufs}"
        )
    meta.update(
        {
            "reread_measured_lufs": reread_lufs,
            "output_path": str(output_path.resolve()),
            "output_sha256": sha256_file(output_path),
            "output_shape": [int(reread.shape[0]), int(reread.shape[1])],
        }
    )
    return reread[:, 0].copy(), meta


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def crop_window(
    data: np.ndarray,
    sample_rate: int,
    input_samples: int,
    input_rate: int,
    delay: int,
    window: Sequence[float] | None,
    input_duration: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    nominal = input_samples * sample_rate / input_rate
    aligned_samples = len(data) - delay
    if aligned_samples <= 0 or abs(aligned_samples - nominal) >= 1.0:
        raise ValueError(
            f"tap aligned length {aligned_samples} is not the input duration "
            f"({nominal:.9f} samples at {sample_rate} Hz)"
        )
    cropped = data[delay : delay + aligned_samples]
    if window is None:
        start, end = 0.0, input_duration
    else:
        start, end = float(window[0]), float(window[1])
        if end > input_duration + 1e-9:
            raise ValueError(f"window end {end} exceeds input duration {input_duration}")
    start_sample = int(round(start * sample_rate))
    end_sample = int(round(end * sample_rate))
    if end_sample > aligned_samples and end >= input_duration - 1e-9:
        end_sample = aligned_samples
    if not 0 <= start_sample < end_sample <= aligned_samples:
        raise ValueError(
            f"window {start}..{end}s maps to invalid samples {start_sample}..{end_sample} "
            f"for {aligned_samples} aligned samples"
        )
    windowed = cropped[start_sample:end_sample]
    return windowed, {
        "input_timeline_s": [start, end],
        "start_sample": start_sample,
        "end_sample": end_sample,
        "samples_per_channel": int(len(windowed)),
        "fixed_delay_samples": delay,
        "aligned_samples_per_channel": aligned_samples,
        "nominal_input_samples_at_stage_rate": nominal,
        "conversion_within_one_sample": abs(aligned_samples - nominal) < 1.0,
        "conversion_exact": abs(aligned_samples - nominal) == 0.0,
    }


def process_channel(
    data: np.ndarray,
    channel: int,
    stage: dict[str, Any],
    input_samples: int,
    input_rate: int,
    input_duration: float,
    window: Sequence[float] | None,
    kind: str,
    target_lufs: float,
    scorer: DNSMOS,
    pyloudnorm: Any,
    normalized_dir: Path,
) -> dict[str, Any]:
    values = np.asarray(data[:, channel], dtype=np.float64)
    windowed, crop_meta = crop_window(
        values,
        stage["sample_rate"],
        input_samples,
        input_rate,
        stage["fixed_delay_samples"],
        window,
        input_duration,
    )
    extrema = audio_extrema(windowed)
    if not np.all(np.isfinite(windowed)):
        original = unavailable_score("nonfinite_audio")
        normalized = unavailable_score("nonfinite_audio")
        return {
            "channel": channel,
            "channel_policy": stage["channel_policy"],
            "source_sample_rate": stage["sample_rate"],
            "source_shape": [int(values.size), 1],
            "crop": crop_meta,
            "window_peak": extrema["peak"],
            "window_clipped_samples": extrema["clipped_samples"],
            "window_finite": extrema["finite"],
            "window_nonfinite_samples": extrema["nonfinite_samples"],
            "original_scores": original,
            "normalized_scores": normalized,
            "normalization": {"available": False, "reason": "nonfinite_audio"},
        }
    common16 = resample_for_scoring(windowed, stage["sample_rate"])
    if not len(common16):
        original = unavailable_score("empty_after_resample")
        normalized = unavailable_score("empty_after_resample")
        return {
            "channel": channel,
            "channel_policy": stage["channel_policy"],
            "source_sample_rate": stage["sample_rate"],
            "source_shape": [int(values.size), 1],
            "crop": crop_meta,
            "window_peak": extrema["peak"],
            "window_clipped_samples": extrema["clipped_samples"],
            "window_finite": extrema["finite"],
            "window_nonfinite_samples": extrema["nonfinite_samples"],
            "common_scoring_domain": {"sample_rate": SCORE_RATE, "samples_per_channel": 0},
            "original_scores": original,
            "normalized_scores": normalized,
            "normalization": {"available": False, "reason": "empty_after_resample"},
        }
    common_meta = {
        "sample_rate": SCORE_RATE,
        "samples_per_channel": int(len(common16)),
        "duration_s": float(len(common16) / SCORE_RATE),
        "audio_sha256": sha256_array(common16),
        "resampler": "librosa.resample soxr_hq fix=True scale=False",
    }
    original = score_record(
        scorer.score_array(common16, SCORE_RATE, kind=kind, use_cache=False)
    )
    normalized_name = (
        f"{stage['direction']}-{stage['order']:04d}-{stage['metadata_index']:04d}-"
        f"{safe_name(stage['id'])}-ch{channel}.wav"
    )
    normalized_path = normalized_dir / normalized_name
    normalized_audio, normalization = normalize_to_target(
        common16, target_lufs, pyloudnorm, normalized_path
    )
    if normalized_audio is None:
        normalized = unavailable_score(normalization.get("reason") or "normalization_unavailable")
    else:
        normalized = score_record(
            scorer.score_array(normalized_audio, SCORE_RATE, kind=kind, use_cache=False)
        )
        normalization["score_input_sha256"] = sha256_array(normalized_audio)
    return {
        "channel": channel,
        "channel_policy": stage["channel_policy"],
        "source_sample_rate": stage["sample_rate"],
        "source_shape": [int(values.size), 1],
        "crop": crop_meta,
        "window_peak": extrema["peak"],
        "window_clipped_samples": extrema["clipped_samples"],
        "window_finite": extrema["finite"],
        "window_nonfinite_samples": extrema["nonfinite_samples"],
        "common_scoring_domain": common_meta,
        "original_scores": original,
        "normalized_scores": normalized,
        "normalization": normalization,
    }


def score_stage(
    stage: dict[str, Any],
    metadata: dict[str, Any],
    args: argparse.Namespace,
    scorer: DNSMOS,
    pyloudnorm: Any,
    normalized_dir: Path,
) -> dict[str, Any]:
    result = dict(stage)
    result["bypass"] = not stage["enabled"]
    result["status"] = (
        "capture_unavailable"
        if not stage["capture_available"]
        else "bypassed"
        if not stage["enabled"]
        else "captured"
    )
    result["tap_sha256_before"] = None
    result["tap_sha256_after"] = None
    result["channels_scored"] = []
    path = Path(stage["file"])
    if not path.is_file():
        raise ValueError(f"stage file does not exist: {path}")
    payload = path.read_bytes()
    result["tap_sha256_before"] = hashlib.sha256(payload).hexdigest()
    if not stage["capture_available"]:
        reason = "capture_unavailable"
        result["reason"] = reason
        for channel in range(stage["capture_channels"]):
            result["channels_scored"].append(
                {
                    "channel": channel,
                    "channel_policy": stage["channel_policy"],
                    "source_sample_rate": stage["sample_rate"],
                    "source_shape": None,
                    "crop": None,
                    "original_scores": unavailable_score(reason),
                    "normalized_scores": unavailable_score(reason),
                    "normalization": {"available": False, "reason": reason},
                }
            )
        return aggregate_stage_availability(result)
    try:
        data, sample_rate = sf.read(io.BytesIO(payload), dtype="float64", always_2d=True)
    except sf.LibsndfileError as error:
        raise ValueError(f"cannot decode stage tap {path}: {error}") from error
    if sample_rate != stage["sample_rate"]:
        raise ValueError(
            f"stage {stage['id']} sample rate {sample_rate} differs from metadata {stage['sample_rate']}"
        )
    if data.shape != (stage["samples_per_channel"], stage["channels"]):
        raise ValueError(
            f"stage {stage['id']} shape {data.shape} differs from metadata "
            f"({stage['samples_per_channel']}, {stage['channels']})"
        )
    file_extrema = audio_extrema(data.reshape(-1))
    result["file_peak"] = file_extrema["peak"]
    result["file_clipped_samples"] = file_extrema["clipped_samples"]
    result["file_finite"] = file_extrema["finite"]
    result["file_nonfinite_samples"] = file_extrema["nonfinite_samples"]
    input_samples = _integer(
        metadata["input_samples_per_channel"], "input_samples_per_channel", minimum=1
    )
    input_rate = _integer(metadata["input_sample_rate"], "input_sample_rate", minimum=1)
    input_duration = input_samples / input_rate
    if stage["capture_channels"] > data.shape[1]:
        raise ValueError(f"stage {stage['id']} has fewer file channels than capture_channels")
    for channel in range(stage["capture_channels"]):
        result["channels_scored"].append(
            process_channel(
                data,
                channel,
                stage,
                input_samples,
                input_rate,
                input_duration,
                args.window,
                args.kind,
                args.loudness_target,
                scorer,
                pyloudnorm,
                normalized_dir,
            )
        )
    return aggregate_stage_availability(result)


def same_index_mapping(previous: dict[str, Any], current: dict[str, Any], channel: int) -> tuple[bool, str]:
    if channel >= previous["capture_channels"]:
        return False, "channel_absent_from_previous_stage"
    if previous["capture_channels"] != current["capture_channels"]:
        if (
            previous["id"] == "tx_input"
            and current["id"] == "tx_bf"
            and previous["capture_channels"] > 1
            and current["capture_channels"] == 1
            and channel == 0
        ):
            return True, "selected_reference_channel_0"
        return False, "channel_count_changed_without_declared_one_to_one_mapping"
    return True, "same_channel_index"


def add_adjacent_deltas(stages: list[dict[str, Any]]) -> None:
    for stage in stages:
        if not stage["diagnostic"]:
            continue
        for channel_row in stage["channels_scored"]:
            channel_row["delta_vs_previous"] = {
                "previous_stage": None,
                "previous_channel": None,
                "channel_mapping": None,
                "raw": {"G": None, "S": None, "N": None},
                "minus30": {"G": None, "S": None, "N": None},
                "unavailable_reason": "diagnostic_substage_not_in_main_chain",
            }
    for direction in ("tx", "rx"):
        chain = sorted(
            (
                stage
                for stage in stages
                if stage["direction"] == direction and not stage["diagnostic"]
            ),
            key=lambda stage: (stage["order"], stage["id"], stage["metadata_index"]),
        )
        previous: dict[str, Any] | None = None
        for stage in chain:
            for channel_row in stage["channels_scored"]:
                channel = channel_row["channel"]
                delta: dict[str, Any] = {
                    "previous_stage": previous["id"] if previous else None,
                    "previous_channel": None,
                    "channel_mapping": None,
                    "raw": {"G": None, "S": None, "N": None},
                    "minus30": {"G": None, "S": None, "N": None},
                    "unavailable_reason": None,
                }
                if previous is None:
                    delta["unavailable_reason"] = "first_stage_in_direction"
                else:
                    mapped, mapping = same_index_mapping(previous, stage, channel)
                    delta["channel_mapping"] = mapping
                    if not mapped:
                        delta["unavailable_reason"] = mapping
                    else:
                        delta["previous_channel"] = channel
                        previous_channel = previous["channels_scored"][channel]
                        current_original = channel_row["original_scores"]
                        previous_original = previous_channel["original_scores"]
                        current_normalized = channel_row["normalized_scores"]
                        previous_normalized = previous_channel["normalized_scores"]
                        if current_original["available"] and previous_original["available"]:
                            delta["raw"] = {
                                key: current_original[key] - previous_original[key]
                                for key in ("G", "S", "N")
                            }
                        if current_normalized["available"] and previous_normalized["available"]:
                            delta["minus30"] = {
                                key: current_normalized[key] - previous_normalized[key]
                                for key in ("G", "S", "N")
                            }
                        if not all(value is not None for value in delta["raw"].values()) or not all(
                            value is not None for value in delta["minus30"].values()
                        ):
                            delta["unavailable_reason"] = "score_unavailable"
                channel_row["delta_vs_previous"] = delta
            previous = stage


def csv_value(value: Any) -> Any:
    return "" if value is None else value


def write_csv(path: Path, stages: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for stage in stages:
            for channel in stage["channels_scored"]:
                original = channel["original_scores"]
                normalized = channel["normalized_scores"]
                delta = channel.get("delta_vs_previous", {})
                raw_delta = delta.get("raw", {})
                minus30_delta = delta.get("minus30", {})
                writer.writerow(
                    {
                        "direction": stage["direction"],
                        "order": stage["order"],
                        "stage": stage["id"],
                        "channel": channel["channel"],
                        "raw_G": csv_value(original["G"]),
                        "raw_S": csv_value(original["S"]),
                        "raw_N": csv_value(original["N"]),
                        "minus30_G": csv_value(normalized["G"]),
                        "minus30_S": csv_value(normalized["S"]),
                        "minus30_N": csv_value(normalized["N"]),
                        "delta_raw_G": csv_value(raw_delta.get("G")),
                        "delta_raw_S": csv_value(raw_delta.get("S")),
                        "delta_raw_N": csv_value(raw_delta.get("N")),
                        "delta_minus30_G": csv_value(minus30_delta.get("G")),
                        "delta_minus30_S": csv_value(minus30_delta.get("S")),
                        "delta_minus30_N": csv_value(minus30_delta.get("N")),
                    }
                )


def format_score(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def write_markdown(path: Path, stages: list[dict[str, Any]], metadata_path: Path) -> None:
    lines = [
        "# VCP stage MOS",
        "",
        f"One-run stage metadata: `{metadata_path}`",
        "",
        "`raw` is the actual-level score; `-30` is the same 16 kHz segment after integrated-LUFS normalization.",
        "",
        "| Dir | Order | Stage | Ch | State | raw G/S/N | -30 G/S/N | Δraw G/S/N | Δ-30 G/S/N |",
        "| --- | ---: | --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for stage in stages:
        for channel in stage["channels_scored"]:
            original = channel["original_scores"]
            normalized = channel["normalized_scores"]
            delta = channel.get("delta_vs_previous", {})
            raw_delta = delta.get("raw", {})
            minus30_delta = delta.get("minus30", {})
            state = stage["status"]
            if original["available"] or normalized["available"]:
                state = "scored" if stage["status"] == "captured" else f"{stage['status']}: scored"
            elif original.get("reason"):
                state = f"{state}: {original['reason']}"
            lines.append(
                "| {direction} | {order} | {stage} | {channel} | {state} | {rg}/{rs}/{rn} | "
                "{ng}/{ns}/{nn} | {drg}/{drs}/{drn} | {dng}/{dns}/{dnn} |".format(
                    direction=stage["direction"],
                    order=stage["order"],
                    stage=stage["id"],
                    channel=channel["channel"],
                    state=state.replace("|", "\\|"),
                    rg=format_score(original["G"]),
                    rs=format_score(original["S"]),
                    rn=format_score(original["N"]),
                    ng=format_score(normalized["G"]),
                    ns=format_score(normalized["S"]),
                    nn=format_score(normalized["N"]),
                    drg=format_score(raw_delta.get("G")),
                    drs=format_score(raw_delta.get("S")),
                    drn=format_score(raw_delta.get("N")),
                    dng=format_score(minus30_delta.get("G")),
                    dns=format_score(minus30_delta.get("S")),
                    dnn=format_score(minus30_delta.get("N")),
                )
            )
    lines.extend(
        [
            "",
            "Deltas use only adjacent stages from the same direction and `order`; missing or non-one-to-one channels are null.",
            "",
            "DNSMOS P.835 is a model estimate, not human MOS or an ITU listening-test result.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace, engine_args: Sequence[str]) -> Path:
    pyloudnorm = import_pyloudnorm()
    runner_file = Path(__file__).resolve()
    runner_sha256 = sha256_file(runner_file)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    prefix = args.output_dir / "run"
    metadata_path = Path(f"{prefix}.stages.json")
    command = [str(args.cli), "--profile", args.profile]
    if args.mic is not None:
        command += ["--mic", str(args.mic)]
    if args.rx_in is not None:
        command += ["--rx-in", str(args.rx_in)]
    if args.echo is not None:
        command += ["--echo", str(args.echo)]
    command += ["--diagnostic-stages", "--out", str(prefix), *engine_args]

    sources = input_paths(args)
    source_hashes_before = hash_paths(sources.values())
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=ENGINE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        (args.output_dir / "engine.stdout.txt").write_text(error.stdout or "", encoding="utf-8")
        (args.output_dir / "engine.stderr.txt").write_text(error.stderr or "", encoding="utf-8")
        raise RuntimeError(
            f"engine timed out after {ENGINE_TIMEOUT_SECONDS} seconds"
        ) from error
    (args.output_dir / "engine.stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (args.output_dir / "engine.stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"engine failed with return code {completed.returncode}: {completed.stderr.strip()}"
        )
    source_hashes_after_engine = verify_hashes(source_hashes_before, "input or CLI")
    if not metadata_path.is_file():
        raise RuntimeError(f"engine did not create diagnostic metadata: {metadata_path}")
    metadata_sha_before = sha256_file(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("stage metadata root must be an object")
    input_rate = _integer(
        metadata.get("input_sample_rate"), "input_sample_rate", minimum=1
    )
    input_samples = _integer(
        metadata.get("input_samples_per_channel"),
        "input_samples_per_channel",
        minimum=1,
    )
    input_duration = input_samples / input_rate
    duration_declared = declared_duration(metadata)
    if duration_declared is not None and abs(duration_declared - input_duration) > 1e-9:
        raise ValueError(
            f"metadata duration {duration_declared} differs from {input_duration} "
            "from input samples/rate"
        )
    if args.window is not None and args.window[1] > input_duration + 1e-9:
        raise ValueError(f"window end {args.window[1]} exceeds input duration {input_duration}")
    inputs = validate_input_wavs(args, input_rate, input_samples)
    stages = normalize_stage_entries(
        metadata,
        args.output_dir,
        {"tx_input": args.mic, "rx_input": args.rx_in},
    )
    stage_ids = {stage["id"] for stage in stages}
    if args.mic is not None and "tx_input" not in stage_ids:
        raise ValueError("metadata is missing tx_input for --mic")
    if args.rx_in is not None and "rx_input" not in stage_ids:
        raise ValueError("metadata is missing rx_input for --rx-in")
    input_stage_validation: dict[str, Any] = {}
    for stage in stages:
        if stage["id"] == "tx_input":
            input_stage_validation["tx_input"] = validate_input_stage_copy(stage, args.mic)
        elif stage["id"] == "rx_input":
            input_stage_validation["rx_input"] = validate_input_stage_copy(stage, args.rx_in)
    tap_hashes_before: dict[str, str] = {}
    for stage in stages:
        path = Path(stage["file"])
        if not path.is_file():
            raise ValueError(f"stage file does not exist: {path}")
        tap_hashes_before[str(path)] = sha256_file(path)
    for source_stage in input_stage_validation.values():
        if source_hashes_after_engine[source_stage["source_path"]] != source_stage["source_sha256"]:
            raise RuntimeError(
                f"source input changed during source-stage validation: {source_stage['source_path']}"
            )
        if tap_hashes_before[source_stage["stage_path"]] != source_stage["stage_sha256"]:
            raise RuntimeError(
                f"source stage changed after PCM equivalence check: {source_stage['stage_path']}"
            )
    assets_dir = args.assets_dir if args.assets_dir is not None else DEFAULT_ASSETS
    scorer = DNSMOS(assets_dir, args.output_dir / "perceptual-cache")
    if scorer.preprocessing.get("model_sha256") != MODEL_SHA:
        raise RuntimeError("DNSMOS scorer does not expose the pinned model hash")
    normalized_dir = args.output_dir / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=False)
    stages_scored = [
        score_stage(stage, metadata, args, scorer, pyloudnorm, normalized_dir)
        for stage in stages
    ]
    for stage in stages_scored:
        if stage["tap_sha256_before"] != tap_hashes_before[stage["file"]]:
            raise RuntimeError(f"stage tap changed before scoring: {stage['file']}")
    add_adjacent_deltas(stages_scored)
    for stage in stages_scored:
        if stage["tap_sha256_before"] is not None:
            stage["tap_sha256_after"] = sha256_file(Path(stage["file"]))
            if stage["tap_sha256_after"] != stage["tap_sha256_before"]:
                raise RuntimeError(f"stage tap changed while scoring: {stage['file']}")
    metadata_sha_after = sha256_file(metadata_path)
    if metadata_sha_after != metadata_sha_before:
        raise RuntimeError("stage metadata changed while scoring")
    source_hashes_after = verify_hashes(source_hashes_before, "input or CLI")

    json_path = args.output_dir / "stage-mos.json"
    csv_path = args.output_dir / "stage-mos.csv"
    markdown_path = args.output_dir / "stage-mos.md"
    write_csv(csv_path, stages_scored)
    write_markdown(markdown_path, stages_scored, metadata_path)
    report = {
        "schema_version": SCHEMA_VERSION,
        "runner_file": str(runner_file),
        "runner_sha256": runner_sha256,
        "command": command,
        "engine_args": list(engine_args),
        "returncode": completed.returncode,
        "profile": args.profile,
        "kind": args.kind,
        "window_s": args.window,
        "loudness_target_lufs": args.loudness_target,
        "scope": SCORE_SCOPE,
        "training_or_upload": False,
        "protocol": {
            "one_engine_run": True,
            "tap_source": "pipeline_stages from the stage metadata produced by this run",
            "topology": "sort each direction by order; deltas only between adjacent stages",
            "channel_policy": "score capture_channels independently; copied file channels are not physical channels",
            "crop": "remove fixed_delay_samples, verify input-length conversion within one sample, then apply window",
            "common_scoring_domain": "librosa soxr_hq at 16 kHz before both original and normalized scoring",
            "short_policy": "ecnr_perceptual official repeated-doubling policy; short_repeat is retained",
            "normalization": "pyloudnorm 0.1.1 K-weighting, 0.4 s blocks, peak prep 0.7/peak, constant gain corrections",
            "invalid_policy": "capture-unavailable or invalid values are null; bypassed stages still score their captured PCM",
            "diagnostic_policy": "diagnostic substages are scored but excluded from main-chain deltas",
            "delay_policy": "fixed metadata delay only; no fitted lag",
            "pure_noise_policy": "the runner does not infer a near-speech role; select a speech-bearing kind/window",
        },
        "scorer": {
            "assets_dir": str(Path(assets_dir).resolve()),
            "model_sha256": MODEL_SHA,
            "preprocessing": scorer.preprocessing,
            "preprocessing_sha256": scorer.preprocessing_sha256,
            "use_cache": False,
        },
        "inputs": inputs,
        "input_stage_validation": input_stage_validation,
        "source_hashes": {
            "before": source_hashes_before,
            "after_engine": source_hashes_after_engine,
            "after_scoring": source_hashes_after,
            "unchanged": True,
        },
        "stage_metadata": {
            "path": str(metadata_path.resolve()),
            "sha256_before": metadata_sha_before,
            "sha256_after": metadata_sha_after,
            "unchanged": metadata_sha_before == metadata_sha_after,
            "input_sample_rate": input_rate,
            "input_samples_per_channel": input_samples,
            "input_duration_s": input_duration,
            "declared_duration_s": duration_declared,
            "original": metadata,
        },
        "tap_hashes_before": tap_hashes_before,
        "stages": stages_scored,
        "files": {
            "json": str(json_path.resolve()),
            "csv": str(csv_path.resolve()),
            "markdown": str(markdown_path.resolve()),
            "engine_stdout": str((args.output_dir / "engine.stdout.txt").resolve()),
            "engine_stderr": str((args.output_dir / "engine.stderr.txt").resolve()),
            "normalized_dir": str(normalized_dir.resolve()),
        },
    }
    save_json(json_path, report)
    return json_path.resolve()


def main(argv: Sequence[str] | None = None) -> int:
    runner_argv, engine_args = split_engine_args(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(runner_argv)
    validate_arguments(args, engine_args)
    output = run(args, engine_args)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
