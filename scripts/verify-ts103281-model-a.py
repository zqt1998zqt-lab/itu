#!/usr/bin/env python3
"""Verify the local Model A implementation against ETSI Annex C."""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from time import perf_counter
from typing import Any, Callable, Mapping
from urllib.request import Request, urlopen
import uuid

import numpy as np
import soundfile as sf

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from server.quality import ts103281
from server.quality.ts103281_validation import within_annex_c_tolerance


EXPECTED = (
    PROJECT
    / "server/quality/assets/ts103281/annex-c-model-a.csv"
)
DEFAULT_VECTOR_DIR = (
    PROJECT / "build/model-a-20261105/baseline/vectors"
)
BASE_URL = (
    "https://docbox.etsi.org/Reference/STQ/"
    "TS%20103%20281/Annex_C%20test%20vectors"
)
EXPECTED_CONDITION_COUNT = 48
EXPECTED_METRICS = ("S-MOS", "N-MOS", "G-MOS")
RMS_EXPECTED = ts103281.MODEL_DIR / "annex-c-rms.csv"
# A shared +15 dB gain reproduces all Table C.1 RMS values within rounding.
# This vector-specific pressure conversion is inferred, not explicitly stated
# by Annex C, and is never fitted to expected MOS scores.
ANNEX_C_PRESSURE_GAIN_DB = 15.0
RMS_ROUNDING_TOLERANCE_DB = .005 + 1e-9
VECTOR_NAME = re.compile(r"^C([0-9]{2})_f1s1\.wav$")


class VerificationError(ValueError):
    """Raised when evidence is incomplete, stale, or internally invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_rows(
    path: Path = EXPECTED,
) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            condition = row.get("condition", "")
            if condition in rows:
                raise VerificationError(
                    f"duplicate Annex C condition: {condition!r}"
                )
            try:
                values = {
                    metric: float(row[column])
                    for metric, column in (
                        ("S-MOS", "s_mos"),
                        ("N-MOS", "n_mos"),
                        ("G-MOS", "g_mos"),
                    )
                }
            except (KeyError, TypeError, ValueError) as error:
                raise VerificationError(
                    f"invalid Annex C expected row for {condition!r}"
                ) from error
            if not all(math.isfinite(value) for value in values.values()):
                raise VerificationError(
                    f"non-finite Annex C expected score for {condition!r}"
                )
            rows[condition] = values
    expected_names = {
        f"C{number:02d}" for number in range(1, EXPECTED_CONDITION_COUNT + 1)
    }
    if len(rows) != EXPECTED_CONDITION_COUNT or set(rows) != expected_names:
        missing = sorted(expected_names - set(rows))
        extra = sorted(set(rows) - expected_names)
        raise VerificationError(
            "Annex C must contain exactly C01..C48 independent conditions; "
            f"missing={missing}, extra={extra}"
        )
    return rows


def expected_rms_rows(path: Path = RMS_EXPECTED) -> dict[str, float]:
    result: dict[str, float] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            condition = row.get("condition", "")
            try:
                value = float(row["rms_dbpa"])
            except (KeyError, ValueError, TypeError) as error:
                raise VerificationError("invalid Annex C RMS calibration row") from error
            if condition in result or not math.isfinite(value):
                raise VerificationError("duplicate or nonfinite Annex C RMS calibration")
            result[condition] = value
    if set(result) != {f"C{i:02d}" for i in range(1, 49)}:
        raise VerificationError("RMS calibration must contain C01..C48")
    return result


def calibrate_vector_pressure(samples: np.ndarray, expected_rms_dbpa: float) -> tuple[np.ndarray, dict]:
    """Apply the shared pressure calibration and check full-file degraded RMS."""
    digital_rms = float(np.sqrt(np.mean(samples[:, 0] ** 2)))
    if digital_rms <= 0 or not math.isfinite(digital_rms):
        raise VerificationError("Annex C degraded RMS must be positive and finite")
    digital_db = 20 * math.log10(digital_rms)
    calibrated_db = digital_db + ANNEX_C_PRESSURE_GAIN_DB
    error = calibrated_db - expected_rms_dbpa
    if abs(error) > RMS_ROUNDING_TOLERANCE_DB:
        raise VerificationError(
            f"Annex C pressure calibration mismatch: measured {calibrated_db:.6f} dBPa "
            f"versus Table C.1 {expected_rms_dbpa:.2f} dBPa"
        )
    return samples * 10 ** (ANNEX_C_PRESSURE_GAIN_DB / 20), {
        "method": "shared PCM-to-Pascal gain verified against Annex C Table C.1 RMS",
        "digital_degraded_rms_dbfs": digital_db,
        "target_degraded_rms_dbpa": expected_rms_dbpa,
        "common_gain_db": ANNEX_C_PRESSURE_GAIN_DB,
        "calibrated_degraded_rms_dbpa": calibrated_db,
        "rms_error_db": error,
        "output_unit": "Pa",
        "applied_to_both_channels": True,
        "uses_mos_targets": False,
    }


def _validate_riff_container(path: Path) -> None:
    payload = Path(path).read_bytes()
    if len(payload) < 12 or payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        raise VerificationError(f"{path.name}: not a RIFF/WAVE file")
    declared_size = int.from_bytes(payload[4:8], "little") + 8
    if len(payload) < declared_size:
        raise VerificationError(
            f"{path.name}: truncated WAV, RIFF declares {declared_size} bytes, "
            f"file has {len(payload)}"
        )
    offset = 12
    chunks: set[bytes] = set()
    while offset + 8 <= declared_size:
        chunk_id = payload[offset : offset + 4]
        chunk_size = int.from_bytes(payload[offset + 4 : offset + 8], "little")
        chunk_end = offset + 8 + chunk_size
        if chunk_end > declared_size:
            raise VerificationError(
                f"{path.name}: truncated {chunk_id!r} chunk"
            )
        chunks.add(chunk_id)
        offset = chunk_end + (chunk_size & 1)
    if b"fmt " not in chunks or b"data" not in chunks:
        raise VerificationError(f"{path.name}: missing fmt or data chunk")


def validate_wav(path: Path) -> dict[str, Any]:
    path = Path(path)
    try:
        _validate_riff_container(path)
        info = sf.info(path)
        samples, rate = sf.read(
            path,
            always_2d=True,
            dtype="float64",
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise VerificationError(f"{path.name}: invalid or truncated WAV: {error}") from error
    if info.format != "WAV":
        raise VerificationError(f"{path.name}: expected WAV, got {info.format!r}")
    if info.channels != 2 or samples.shape[1] != 2:
        raise VerificationError(f"{path.name}: expected exactly two channels")
    if info.frames <= 0 or samples.shape[0] != info.frames:
        raise VerificationError(
            f"{path.name}: declared {info.frames} frames, read {samples.shape[0]}"
        )
    if rate != info.samplerate:
        raise VerificationError(
            f"{path.name}: inconsistent sample rate {rate} vs {info.samplerate}"
        )
    if not np.isfinite(samples).all():
        raise VerificationError(f"{path.name}: contains non-finite samples")
    return {
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration_seconds": float(info.frames / info.samplerate),
        "format": info.format,
        "subtype": info.subtype,
    }


def vector_name(condition: str) -> str:
    if condition not in {
        f"C{number:02d}" for number in range(1, EXPECTED_CONDITION_COUNT + 1)
    }:
        raise VerificationError(f"invalid Annex C condition: {condition!r}")
    return f"{condition}_f1s1.wav"


def download_vector(
    destination: Path,
    *,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    destination = Path(destination)
    request = Request(
        f"{BASE_URL}/{destination.name}",
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.etsi.org/",
        },
    )
    with opener(request, timeout=60) as response:
        content_length = response.headers.get("Content-Length")
        payload = response.read()
    if content_length is not None and int(content_length) != len(payload):
        raise VerificationError(
            f"{destination.name}: truncated download, expected "
            f"{content_length} bytes, received {len(payload)}"
        )
    if not payload:
        raise VerificationError(f"{destination.name}: empty download")
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.download"
    )
    with temporary.open("xb") as handle:
        handle.write(payload)
    record = validate_wav(temporary)
    if destination.exists():
        raise VerificationError(
            f"{destination}: appeared during download; refusing to overwrite it"
        )
    temporary.replace(destination)
    record["url"] = request.full_url
    return record


def _write_json_exclusive(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n"
        )


def ensure_vectors(
    directory: Path,
    conditions: list[str] | tuple[str, ...],
    *,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, dict[str, Any]]:
    directory = Path(directory)
    ordered = tuple(sorted(set(conditions)))
    if not ordered or len(ordered) != len(conditions):
        raise VerificationError("vector conditions must be non-empty and unique")
    directory.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, Any]] = {}
    for condition in ordered:
        path = directory / vector_name(condition)
        if path.exists():
            record = validate_wav(path)
            record["path"] = str(path)
        else:
            record = download_vector(path, opener=opener)
            record["path"] = str(path)
        record["url"] = f"{BASE_URL}/{path.name}"
        records[condition] = record

    manifest_path = directory / "vector-manifest.json"
    manifest = {
        "source": BASE_URL,
        "condition_count": len(records),
        "vectors": records,
    }
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise VerificationError(
                f"{manifest_path}: invalid vector manifest: {error}"
            ) from error
        if existing != manifest:
            raise VerificationError(
                f"{manifest_path}: vector hashes or metadata changed"
            )
    else:
        _write_json_exclusive(manifest_path, manifest)
    return records


def _local_module_path(
    project_root: Path,
    module_parts: list[str],
) -> Path | None:
    if not module_parts:
        return None
    module_path = project_root.joinpath(*module_parts).with_suffix(".py")
    if module_path.is_file():
        return module_path.resolve()
    package_path = project_root.joinpath(*module_parts, "__init__.py")
    if package_path.is_file():
        return package_path.resolve()
    return None


def _imported_local_paths(
    source_path: Path,
    project_root: Path,
) -> set[Path]:
    project_root = project_root.resolve()
    tree = ast.parse(
        source_path.read_text(encoding="utf-8"),
        filename=str(source_path),
    )
    package_parts = list(
        source_path.resolve().relative_to(project_root).parts[:-1]
    )
    imported: set[Path] = set()
    for node in ast.walk(tree):
        bases: list[list[str]] = []
        if isinstance(node, ast.Import):
            bases = [alias.name.split(".") for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = list(package_parts)
                if node.level > 1:
                    base = base[: -(node.level - 1)]
                if node.module:
                    base.extend(node.module.split("."))
                bases = [base]
                for alias in node.names:
                    if alias.name != "*":
                        candidate = _local_module_path(
                            project_root,
                            [*base, alias.name],
                        )
                        if candidate is not None:
                            imported.add(candidate)
            elif node.module:
                base = node.module.split(".")
                bases = [base]
                for alias in node.names:
                    if alias.name != "*":
                        candidate = _local_module_path(
                            project_root,
                            [*base, alias.name],
                        )
                        if candidate is not None:
                            imported.add(candidate)
        for parts in bases:
            candidate = _local_module_path(project_root, parts)
            if candidate is not None:
                imported.add(candidate)
    return imported


def _dependency_paths(
    main_path: Path,
    project_root: Path = PROJECT,
) -> tuple[Path, ...]:
    project_root = project_root.resolve()
    main_path = main_path.resolve()
    if not main_path.is_file():
        raise VerificationError(f"missing Model A implementation: {main_path}")
    pending = [main_path]
    dependencies: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in dependencies:
            continue
        if not path.is_file():
            raise VerificationError(f"missing local dependency: {path}")
        try:
            path.relative_to(project_root)
        except ValueError as error:
            raise VerificationError(
                f"local dependency escapes project root: {path}"
            ) from error
        dependencies.add(path)
        for imported in _imported_local_paths(path, project_root):
            if imported not in dependencies:
                pending.append(imported)
        for parent in path.parents:
            if parent == project_root:
                break
            initializer = parent / "__init__.py"
            if initializer.is_file() and initializer not in dependencies:
                pending.append(initializer)
    return tuple(sorted(dependencies))


def implementation_identity(
    *,
    project_root: Path = PROJECT,
    main_path: Path | None = None,
    model_dir: Path | None = None,
    verifier_path: Path | None = None,
) -> dict[str, Any]:
    """Hash the main implementation, local import closure, models, verifier."""
    project_root = Path(project_root).resolve()
    main_path = (
        Path(ts103281.__file__).resolve()
        if main_path is None
        else Path(main_path).resolve()
    )
    model_dir = (
        Path(ts103281.MODEL_DIR).resolve()
        if model_dir is None
        else Path(model_dir).resolve()
    )
    verifier_path = (
        Path(__file__).resolve()
        if verifier_path is None
        else Path(verifier_path).resolve()
    )
    model_paths = sorted(model_dir.glob("*.model"))
    if {path.name for path in model_paths} != {
        "S-MOS.model",
        "N-MOS.model",
        "G-MOS.model",
    }:
        raise VerificationError(
            f"{model_dir}: expected exactly the three Model A forest files"
        )
    dependency_paths = _dependency_paths(main_path, project_root)
    dependency_sha256 = {
        path.relative_to(project_root).as_posix(): sha256_file(path)
        for path in dependency_paths
    }
    model_sha256 = {
        path.relative_to(project_root).as_posix(): sha256_file(path)
        for path in model_paths
    }
    reference_assets_sha256 = {
        path.relative_to(project_root).as_posix(): sha256_file(path)
        for path in (model_dir / "annex-c-model-a.csv", model_dir / "annex-c-rms.csv")
        if path.is_file()
    }
    implementation_key = main_path.relative_to(project_root).as_posix()
    return {
        # Kept for compatibility with the existing evidence loader.
        "implementation_sha256": dependency_sha256[implementation_key],
        "dependency_sha256": dependency_sha256,
        "model_sha256": model_sha256,
        "reference_assets_sha256": reference_assets_sha256,
        "verifier_sha256": sha256_file(verifier_path),
        "identity_scope": (
            "main implementation plus transitive local imports, package "
            "initializers, Model A forests, and verifier"
        ),
    }


def _identity_differences(
    start: Any,
    end: Any,
    prefix: str = "",
) -> list[str]:
    if isinstance(start, Mapping) and isinstance(end, Mapping):
        differences = []
        for key in sorted(set(start) | set(end)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in start or key not in end:
                differences.append(child)
            else:
                differences.extend(
                    _identity_differences(start[key], end[key], child)
                )
        return differences
    return [] if start == end else [prefix]


def _require_stable_identity(
    start: Mapping[str, Any],
    end: Mapping[str, Any],
) -> None:
    if start == end:
        return
    changed = _identity_differences(start, end)
    raise VerificationError(
        "implementation identity changed during verification; "
        f"changed fields={changed}"
    )


def _finite_score(score: Any) -> float | None:
    value = score.get("value") if isinstance(score, Mapping) else score
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _metric_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "mae": None,
            "rmse": None,
            "maximum_absolute_error": None,
        }
    array = np.asarray(values, dtype=np.float64)
    absolute = np.abs(array)
    return {
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(array**2))),
        "maximum_absolute_error": float(np.max(absolute)),
    }


def verify(
    directory: Path,
    limit: int | None = None,
    *,
    expected_path: Path = EXPECTED,
    include_features: bool = False,
    evaluator: Callable[..., Mapping[str, Any]] | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    start_identity = implementation_identity()
    if limit is not None and not 1 <= limit <= EXPECTED_CONDITION_COUNT:
        raise VerificationError("limit must be in 1..48")
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 16:
        raise VerificationError("workers must be in 1..16")
    if workers > 1 and evaluator is not None:
        raise VerificationError("parallel verification cannot use a custom evaluator")
    expected = expected_rows(expected_path)
    expected_rms = expected_rms_rows()
    all_names = sorted(expected)
    vector_records = ensure_vectors(directory, all_names)
    names = all_names if limit is None else all_names[:limit]
    evaluator = ts103281.evaluate_model_a if evaluator is None else evaluator
    rows: list[dict[str, Any]] = []
    score_count = 0
    finite_score_count = 0
    started_all = perf_counter()
    calibrations = []
    def prepare(condition: str):
        path = Path(directory) / vector_name(condition)
        vector = validate_wav(path)
        samples, rate = sf.read(
            path,
            always_2d=True,
            dtype="float64",
        )
        samples, input_calibration = calibrate_vector_pressure(samples, expected_rms[condition])
        # ETSI files use channel 2 for the common reference and channel 1 for
        # the degraded signal, inferred from the vectors and Table C.1.
        return path, vector, input_calibration, samples, rate

    def serial_results():
        for condition in names:
            path, vector, calibration, samples, rate = prepare(condition)
            started = perf_counter()
            result = evaluator(samples[:, 1], samples[:, 0], rate)
            yield condition, path, vector, calibration, result, perf_counter() - started

    def parallel_results(pool):
        pending = []
        for condition in names:
            path, vector, calibration, samples, rate = prepare(condition)
            started = perf_counter()
            future = pool.submit(evaluator, samples[:, 1], samples[:, 0], rate)
            pending.append((condition, path, vector, calibration, started, future))
        for condition, path, vector, calibration, started, future in pending:
            yield condition, path, vector, calibration, future.result(), perf_counter() - started

    with ExitStack() as stack:
        results = serial_results()
        if workers > 1:
            pool = stack.enter_context(ProcessPoolExecutor(max_workers=workers))
            results = parallel_results(pool)
        for index, (condition, path, vector, input_calibration, result, duration_seconds) in enumerate(results, start=1):
            calibrations.append(input_calibration)
            row, finite_count = _verification_row(
                condition, path, vector, vector_records[condition], input_calibration, result,
                duration_seconds, expected, include_features,
                index, len(names), started_all,
            )
            if workers > 1:
                row["duration_includes_queue"] = True
            rows.append(row)
            score_count += len(EXPECTED_METRICS)
            finite_score_count += finite_count
            if on_progress is not None:
                on_progress(row["progress"])

    summary = {
        metric: _metric_summary(
            [
                row["difference"][metric]
                for row in rows
                if row["difference"][metric] is not None
            ]
        )
        for metric in EXPECTED_METRICS
    }
    passed_conditions = sum(row["within_0_1"] for row in rows)
    independent_conditions = len({row["condition"] for row in rows})
    full_run = limit is None
    conformant = bool(
        full_run
        and len(rows) == EXPECTED_CONDITION_COUNT
        and independent_conditions == EXPECTED_CONDITION_COUNT
        and score_count == EXPECTED_CONDITION_COUNT * len(EXPECTED_METRICS)
        and finite_score_count == score_count
        and passed_conditions == EXPECTED_CONDITION_COUNT
    )
    if conformant:
        failed_reason = None
    elif not full_run:
        failed_reason = (
            "Subset runs are diagnostic only and cannot claim Annex C "
            "conformance; run all 48 conditions without --limit."
        )
    elif finite_score_count != score_count:
        failed_reason = (
            f"Only {finite_score_count}/{score_count} original model scores "
            "were finite."
        )
    else:
        failed_reason = (
            f"{passed_conditions}/{EXPECTED_CONDITION_COUNT} official "
            "conditions reproduced every original score within +/-0.1 MOS."
        )
    report = {
        "standard": "ETSI TS 103 281 V1.3.2 Annex C",
        "condition_count": len(rows),
        "expected_condition_count": EXPECTED_CONDITION_COUNT,
        "independent_condition_count": independent_conditions,
        "score_count": score_count,
        "finite_score_count": finite_score_count,
        "tolerance_mos": 0.1,
        "conformance_scope": "full" if full_run else "subset",
        "requested_limit": limit,
        "conformant": conformant,
        "passed_conditions": passed_conditions,
        **start_identity,
        "failed_reason": failed_reason,
        "features_included": include_features,
        "original_scores_only": True,
        "pressure_calibration": {
            "method": "Annex C Table C.1 full-file unweighted RMS calibration",
            "source": "ETSI TS 103 281 V1.3.2 Annex C Table C.1, page 73",
            "common_gain_db": ANNEX_C_PRESSURE_GAIN_DB,
            "verified_conditions": len(calibrations),
            "max_rms_error_db": max(abs(c["rms_error_db"]) for c in calibrations),
            "uses_mos_targets": False,
            "basis": "vector-specific gain inferred from the Annex C RMS table",
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": perf_counter() - started_all,
        "environment": {
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "workers": workers,
        },
        "vector_manifest": {
            "path": str(Path(directory) / "vector-manifest.json"),
            "condition_count": len(vector_records),
        },
        "summary": summary,
        "rows": rows,
    }
    end_identity = implementation_identity()
    _require_stable_identity(start_identity, end_identity)
    report["identity_verified_during_run"] = True
    return report


def _verification_row(
    condition, path, vector, vector_record, input_calibration, result, duration_seconds,
    expected, include_features, index, total, started_all,
):
    if not isinstance(result, Mapping):
        raise VerificationError(
            f"{condition}: evaluator returned a non-mapping result"
        )
    scores = result.get("scores")
    if not isinstance(scores, Mapping):
        raise VerificationError(f"{condition}: evaluator returned no scores")
    actual = {
        metric: _finite_score(scores.get(metric))
        for metric in EXPECTED_METRICS
    }
    finite_count = sum(value is not None for value in actual.values())
    differences = {
        metric: (
            None
            if actual[metric] is None
            else actual[metric] - expected[condition][metric]
        )
        for metric in EXPECTED_METRICS
    }
    within = all(within_annex_c_tolerance(value) for value in differences.values())
    progress = {
        "completed": index,
        "total": total,
        "fraction": index / total,
        "elapsed_seconds": perf_counter() - started_all,
    }
    row = {
        "condition": condition,
        "expected": expected[condition],
        "actual": actual,
        "difference": differences,
        "score_count": len(EXPECTED_METRICS),
        "finite_score_count": finite_count,
        "within_0_1": within,
        "duration_seconds": duration_seconds,
        "progress": progress,
        "vector": {
            **vector,
            "path": str(path),
            "manifest": vector_record,
        },
        "input_calibration": input_calibration,
    }
    if include_features:
        row["features"] = result.get("features")
        row["diagnostics"] = result.get("diagnostics")
    return row, finite_count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vectors",
        type=Path,
        default=DEFAULT_VECTOR_DIR,
        help="directory containing C01_f1s1.wav through C48_f1s1.wav",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1, help="number of model processes (1..16)")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--include-features",
        action="store_true",
        help="include per-condition extracted features and diagnostics",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "write complete verification evidence JSON to this new path; "
            "existing evidence is never overwritten"
        ),
    )
    args = parser.parse_args()
    if (
        args.limit is not None
        and not 1 <= args.limit <= EXPECTED_CONDITION_COUNT
    ):
        parser.error("--limit must be in 1..48")
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be in 1..16")

    def report_progress(progress: dict[str, Any]) -> None:
        print(
            f"[{progress['completed']}/{progress['total']}] "
            f"elapsed={progress['elapsed_seconds']:.3f}s",
            file=sys.stderr,
            flush=True,
        )

    report = verify(
        args.vectors,
        args.limit,
        include_features=args.include_features,
        on_progress=report_progress,
        workers=args.workers,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        print(
            f"{report['passed_conditions']}/{report['condition_count']} "
            f"conditions within +/-{report['tolerance_mos']} MOS"
        )
        print(
            f"{report['finite_score_count']}/{report['score_count']} "
            f"original scores finite; scope={report['conformance_scope']}"
        )
        for metric, values in report["summary"].items():
            if values["mae"] is None:
                print(f"{metric}: no finite original scores")
            else:
                print(
                    f"{metric}: MAE={values['mae']:.4f}, "
                    f"RMSE={values['rmse']:.4f}, "
                    f"max={values['maximum_absolute_error']:.4f}"
                )
    if args.output is not None:
        _write_json_exclusive(args.output, report)
    return 0 if report["conformant"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
