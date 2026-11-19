#!/home/zhangqingtian/.venv/bin/python
"""Evaluate the local OM-LSA tuning probe on fixed dev/holdout evidence.

Run from the ITU project root, for example:
  PYTHONDONTWRITEBYTECODE=1 /home/zhangqingtian/.venv/bin/python \
    scripts/evaluate_omlsa_tuning.py --probe BIN --original-probe BIN \
    --output NEW_DIR --mode dev

This is deliberately a small experimental harness. It does not select a
winner, tune parameters from results, or modify project/cache files.
"""

from __future__ import annotations

import os

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import math
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


RATE = 16_000
DURATION_SECONDS = 12
SAMPLES = RATE * DURATION_SECONDS
HOP = 160
FRAME = 320
ACTIVITY_DBFS = -55.0
FLOOR_DB = -9.0
PROBE_TIMEOUT_SECONDS = 60

EXPECTED_MODEL_SHA256 = (
    "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd"
)
DEV_REFERENCE_SHA256 = (
    "dbe939a3c3fba3b9c8905455f34387448c5ca4a8cc0a64016d561485648660c5"
)
HOLDOUT_BATCH_ID = "e8c06dab3bf544be809070682954a806"
HOLDOUT_REFERENCES = {
    "02": (
        "54d93a7656b85cd1c5e5699727418072a35bac00a075f7d25858e208906069d5",
        (2026111402, 2026111403),
        (6, 12),
    ),
    "03": (
        "07c70f7c15159f6814660fae6485adb3e02b601a637ba08ce356ae324482dc6b",
        (2026111402, 2026111403),
        (12, 18),
    ),
    "04": (
        "ae18dc343e073cbad381d55785a323ed2fa5201b621828583a4d80ac6671f431",
        (2026111402, 2026111403),
        (6, 18),
    ),
}

DEV_MANIFEST = Path(
    "/tmp/itu-ns-near-single-20261114/20261113T165610Z/stationary-white/evidence.json"
)
DATASET_AUDIO_DIR = Path(
    "/home/zhangqingtian/project/itu/build/acoustic-qualification/"
    "acoustic-scene-condition-datasets/"
    "cf29475e343052eaffa73dfebad1b0b0/assets/audio"
)
HOLDOUT_BATCH_DIR = Path(
    "/home/zhangqingtian/project/itu/build/acoustic-qualification/"
    "scene-condition-batches"
) / HOLDOUT_BATCH_ID

DEFAULT_VARIANTS: dict[str, tuple[float, float, float]] = {
    "baseline": (1.0, 0.0, 60.0),
    "noise": (0.5, 0.0, 60.0),
    "speech": (1.0, 0.2, 60.0),
    "combined": (0.5, 0.2, 60.0),
}


class EvaluationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Variant:
    name: str
    noise_update_scale: float
    speech_recovery: float
    release_ms: float

    @property
    def params(self) -> dict[str, float]:
        return {
            "noise_update_scale": self.noise_update_scale,
            "speech_recovery": self.speech_recovery,
            "release_ms": self.release_ms,
        }


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    role: str
    source_case_id: str
    reference_path: Path | None
    reference_sha256: str
    assessment_path: Path | None
    noise_seed: int
    target_snr_db: float
    mos_eligible: bool
    synthetic: bool = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def finite(value: float | np.floating[Any]) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    payload = json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(payload, encoding="utf-8")


def safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")[:80] or "item"
    return f"{cleaned}-{sha256_bytes(value.encode())[:8]}"


def recursive_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from recursive_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from recursive_strings(item)


def recursive_wav_paths(value: Any) -> Iterable[str]:
    for text in recursive_strings(value):
        if text.lower().endswith(".wav") and "/" in text:
            yield text


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"cannot read JSON {path}: {error}") from error


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    if not path.is_file():
        raise EvaluationError(f"source audio does not exist: {path}")
    try:
        import soundfile as sf

        audio, rate = sf.read(path, dtype="float64", always_2d=True)
        audio = np.asarray(audio, dtype=np.float64)
    except ImportError:
        from scipy.io import wavfile

        rate, audio = wavfile.read(path, mmap=False)
        audio = np.asarray(audio)
        if audio.ndim == 1:
            audio = audio[:, None]
        if np.issubdtype(audio.dtype, np.integer):
            info = np.iinfo(audio.dtype)
            scale = float(max(abs(info.min), info.max))
            audio = audio.astype(np.float64) / scale
        else:
            audio = audio.astype(np.float64)
    except Exception as error:
        raise EvaluationError(f"cannot decode WAV {path}: {error}") from error
    if audio.ndim != 2 or audio.shape[1] != 1:
        raise EvaluationError(f"source must be mono: {path}, shape={audio.shape}")
    if int(rate) != RATE:
        raise EvaluationError(f"source must be {RATE} Hz: {path}, got {rate}")
    result = np.ascontiguousarray(audio[:, 0], dtype=np.float64)
    if result.size < SAMPLES:
        raise EvaluationError(
            f"source is shorter than 12 seconds: {path}, samples={result.size}"
        )
    if not np.isfinite(result).all():
        raise EvaluationError(f"source contains non-finite samples: {path}")
    return np.ascontiguousarray(result[:SAMPLES]), int(rate)


def validate_reference(path: Path, expected_sha256: str) -> tuple[np.ndarray, dict[str, Any]]:
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise EvaluationError(
            f"source SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual_sha256}"
        )
    audio, rate = read_wav_mono(path)
    return audio, {
        "path": str(path),
        "sha256": actual_sha256,
        "sample_rate": rate,
        "decoded_samples": int(audio.size),
        "selected_samples": SAMPLES,
        "selection": "first 12 seconds",
    }


def load_dev_spec() -> tuple[CaseSpec, dict[str, Any]]:
    manifest_record: dict[str, Any] = {
        "path": str(DEV_MANIFEST),
        "exists": DEV_MANIFEST.is_file(),
    }
    reference_path = DATASET_AUDIO_DIR / f"{DEV_REFERENCE_SHA256}.wav"
    if DEV_MANIFEST.is_file():
        manifest = read_json(DEV_MANIFEST)
        manifest_record["sha256"] = sha256_file(DEV_MANIFEST)
        declared_sha = manifest.get("clean_reference_sha256")
        if declared_sha and declared_sha != DEV_REFERENCE_SHA256:
            raise EvaluationError(
                "dev manifest clean_reference_sha256 does not match the fixed source"
            )
        if manifest.get("noise_seed") not in (None, 2026111401):
            raise EvaluationError("dev manifest noise seed is not 2026111401")
        if manifest.get("target_snr_db") not in (None, 12):
            raise EvaluationError("dev manifest target SNR is not 12 dB")
        if manifest.get("rate") not in (None, RATE):
            raise EvaluationError("dev manifest rate is not 16000 Hz")
        declared_path = manifest.get("clean_reference_path")
        if declared_path:
            reference_path = Path(declared_path)
        manifest_record["declared_reference_path"] = declared_path
        manifest_record["declared_noise_seed"] = manifest.get("noise_seed")
        manifest_record["declared_target_snr_db"] = manifest.get("target_snr_db")
    if not reference_path.is_file():
        raise EvaluationError(
            f"fixed dev source is missing: {reference_path}; refusing to substitute another source"
        )
    spec = CaseSpec(
        case_id="dev-near_single-01-white-seed2026111401-snr12",
        role="dev",
        source_case_id="sce1-wb-wired-near_single-01-conversation-v2",
        reference_path=reference_path,
        reference_sha256=DEV_REFERENCE_SHA256,
        assessment_path=None,
        noise_seed=2026111401,
        target_snr_db=12.0,
        mos_eligible=True,
    )
    return spec, manifest_record


def load_holdout_specs() -> list[CaseSpec]:
    specs: list[CaseSpec] = []
    for suffix, (reference_sha, seeds, snrs) in sorted(HOLDOUT_REFERENCES.items()):
        source_case_id = f"sce1-wb-wired-near_single-{suffix}-conversation-v2"
        assessment_path = HOLDOUT_BATCH_DIR / "cases" / source_case_id / "assessment.json"
        if not assessment_path.is_file():
            raise EvaluationError(
                f"fixed holdout assessment is missing for near_single-{suffix}: {assessment_path}"
            )
        reference_path = DATASET_AUDIO_DIR / f"{reference_sha}.wav"
        if not reference_path.is_file():
            raise EvaluationError(
                f"fixed holdout reference is missing for near_single-{suffix}: {reference_path}"
            )
        assessment = read_json(assessment_path)
        assessment_paths = list(recursive_wav_paths(assessment))
        matching_paths = [
            Path(path)
            for path in assessment_paths
            if Path(path).name == reference_path.name
        ]
        if matching_paths and not any(path == reference_path for path in matching_paths):
            raise EvaluationError(
                f"holdout assessment reference path disagrees with fixed source for {source_case_id}"
            )
        for seed, snr in zip(seeds, snrs, strict=True):
            specs.append(
                CaseSpec(
                    case_id=f"holdout-near_single-{suffix}-white-seed{seed}-snr{snr}",
                    role="holdout",
                    source_case_id=source_case_id,
                    reference_path=reference_path,
                    reference_sha256=reference_sha,
                    assessment_path=assessment_path,
                    noise_seed=seed,
                    target_snr_db=float(snr),
                    mos_eligible=True,
                )
            )
    if len(specs) != 6:
        raise EvaluationError(f"holdout invariant failed: expected 6 examples, got {len(specs)}")
    return specs


def make_synthetic_case() -> tuple[CaseSpec, np.ndarray]:
    time_axis = np.arange(SAMPLES, dtype=np.float64) / RATE
    signal = np.zeros(SAMPLES, dtype=np.float64)
    consonant_rng = np.random.default_rng(2026111410)
    for event_index, start_s in enumerate(np.arange(0.55, 11.2, 1.35)):
        start = int(round(start_s * RATE))
        consonant_length = int(0.085 * RATE)
        if start + consonant_length >= SAMPLES:
            break
        noise = consonant_rng.standard_normal(consonant_length)
        noise = np.diff(noise, prepend=noise[0])
        noise /= max(np.sqrt(np.mean(noise**2)), 1e-12)
        envelope = np.sin(np.pi * np.arange(consonant_length) / consonant_length) ** 2
        signal[start : start + consonant_length] += 10 ** (-48.0 / 20.0) * noise * envelope

        tail_start = start + consonant_length
        tail_length = min(int(0.36 * RATE), SAMPLES - tail_start)
        if tail_length <= 0:
            continue
        local_time = np.arange(tail_length, dtype=np.float64) / RATE
        fundamental = 115.0 + 7.0 * event_index
        tail = (
            np.sin(2 * np.pi * fundamental * local_time)
            + 0.45 * np.sin(2 * np.pi * 2.0 * fundamental * local_time + 0.3)
            + 0.2 * np.sin(2 * np.pi * 3.0 * fundamental * local_time + 0.8)
        )
        tail *= np.exp(-local_time / 0.11)
        tail *= np.minimum(1.0, local_time / 0.008)
        tail *= 0.018 / max(np.max(np.abs(tail)), 1e-12)
        signal[tail_start : tail_start + tail_length] += tail

    # A quiet deterministic room floor keeps the diagnostic waveform realistic.
    background_rng = np.random.default_rng(2026111411)
    background = background_rng.standard_normal(SAMPLES)
    signal_rms = float(np.sqrt(np.mean(signal**2)))
    background *= signal_rms / max(float(np.sqrt(np.mean(background**2))), 1e-12) / 10 ** (18 / 20)
    reference = np.ascontiguousarray(signal + background, dtype=np.float64)
    if not np.isfinite(reference).all() or float(np.max(np.abs(reference))) >= 1.0:
        raise EvaluationError("synthetic diagnostic source is non-finite or clipping")
    spec = CaseSpec(
        case_id="dev-synthetic-weak-consonant-decay-v1",
        role="dev_synthetic",
        source_case_id="synthetic-weak-consonant-decay-v1",
        reference_path=None,
        reference_sha256=sha256_bytes(np.asarray(reference, dtype="<f8").tobytes()),
        assessment_path=None,
        noise_seed=2026111411,
        target_snr_db=18.0,
        mos_eligible=False,
        synthetic=True,
    )
    return spec, reference


def parse_variants(raw: str | None) -> list[Variant]:
    if raw is None:
        return [Variant(name, *params) for name, params in DEFAULT_VARIANTS.items()]
    candidate = Path(raw)
    if candidate.is_file():
        payload = read_json(candidate)
    else:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise EvaluationError(f"--variants must be JSON or a JSON file path: {error}") from error
    if isinstance(payload, list):
        entries: list[tuple[str, Any]] = []
        for index, item in enumerate(payload):
            if not isinstance(item, Mapping) or "name" not in item:
                raise EvaluationError("--variants list entries require a name")
            entries.append((str(item["name"]), item))
    elif isinstance(payload, Mapping):
        entries = [(str(name), item) for name, item in payload.items()]
    else:
        raise EvaluationError("--variants must be a JSON object or list")
    variants: list[Variant] = []
    seen: set[str] = set()
    for name, item in entries:
        if not name or name in seen:
            raise EvaluationError(f"variant names must be non-empty and unique: {name!r}")
        seen.add(name)
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, Mapping)):
            if len(item) != 3:
                raise EvaluationError(f"variant {name} list must be [noise, speech, release]")
            values = item
        elif isinstance(item, Mapping):
            noise = item.get("noise_update_scale", item.get("noise"))
            speech = item.get("speech_recovery", item.get("speech"))
            release = item.get("release_ms", item.get("release"))
            values = (noise, speech, release)
        else:
            raise EvaluationError(f"variant {name} must be a 3-item list or object")
        try:
            numbers = tuple(float(value) for value in values)
        except (TypeError, ValueError) as error:
            raise EvaluationError(f"variant {name} has a non-numeric parameter") from error
        variant = Variant(name, *numbers)
        validate_variant(variant)
        variants.append(variant)
    if not variants:
        raise EvaluationError("--variants cannot be empty")
    return variants


def validate_variant(variant: Variant) -> None:
    values = variant.params
    if not all(math.isfinite(value) for value in values.values()):
        raise EvaluationError(f"variant {variant.name} contains a non-finite parameter")
    if not 0.1 <= variant.noise_update_scale <= 1.0:
        raise EvaluationError(f"variant {variant.name}: noise_update_scale must be [0.1, 1]")
    if not 0.0 <= variant.speech_recovery <= 1.0:
        raise EvaluationError(f"variant {variant.name}: speech_recovery must be [0, 1]")
    if not 0.0 <= variant.release_ms <= 200.0:
        raise EvaluationError(f"variant {variant.name}: release_ms must be [0, 200]")


def write_f32(path: Path, samples: np.ndarray) -> dict[str, Any]:
    values = np.ascontiguousarray(samples, dtype="<f4")
    if values.ndim != 1 or not np.isfinite(values).all():
        raise EvaluationError(f"invalid float32 input for {path}")
    values.tofile(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "samples": int(values.size),
        "dtype": "float32le",
    }


def read_f32(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype="<f4")


def rms_dbfs(samples: np.ndarray) -> float:
    values = np.asarray(samples, dtype=np.float64)
    if values.size == 0:
        return -math.inf
    rms = float(np.sqrt(np.mean(values * values)))
    return 20.0 * math.log10(max(rms, 1e-12))


def make_noise(reference: np.ndarray, seed: int, snr_db: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(SAMPLES)
    reference_f32 = np.asarray(reference, dtype="<f4")
    ref_rms = float(np.sqrt(np.mean(reference_f32.astype(np.float64) ** 2)))
    raw_rms = float(np.sqrt(np.mean(raw * raw)))
    scale = ref_rms / max(raw_rms, 1e-12) / 10 ** (snr_db / 20.0)
    return np.ascontiguousarray(raw * scale, dtype="<f4")


def actual_snr_db(reference: np.ndarray, noise: np.ndarray) -> float:
    ref = np.asarray(reference, dtype="<f4").astype(np.float64)
    nse = np.asarray(noise, dtype="<f4").astype(np.float64)
    ratio = float(np.sum(ref * ref) / max(np.sum(nse * nse), 1e-30))
    return 10.0 * math.log10(max(ratio, 1e-30))


def frame_activity(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frames = np.asarray(reference, dtype=np.float64).reshape(-1, FRAME)
    frame_db = 20.0 * np.log10(
        np.maximum(np.sqrt(np.mean(frames * frames, axis=1)), 1e-12)
    )
    active = frame_db > ACTIVITY_DBFS
    if int(np.count_nonzero(active)) < 4:
        raise EvaluationError("reference has fewer than four active 20 ms frames")
    return frame_db, active


def mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    result = float(np.mean(np.asarray(values, dtype=np.float64)))
    return finite(result)


def projection_metrics(
    reference: np.ndarray, mixture: np.ndarray, output: np.ndarray, active_frames: np.ndarray
) -> dict[str, Any]:
    active_samples = np.repeat(active_frames, FRAME)
    ref = np.asarray(reference, dtype=np.float64)[active_samples]
    mix = np.asarray(mixture, dtype=np.float64)[active_samples]
    out = np.asarray(output, dtype=np.float64)[active_samples]
    ref_energy = float(np.dot(ref, ref))
    if ref_energy <= 0.0:
        raise EvaluationError("active reference region has zero energy")
    alpha = float(np.dot(ref, out) / ref_energy)
    mix_alpha = float(np.dot(ref, mix) / ref_energy)
    target = alpha * ref
    residual = out - target
    sisdr = 10.0 * math.log10(
        max(float(np.dot(target, target)), 1e-30) / max(float(np.dot(residual, residual)), 1e-30)
    )
    corr = float(np.corrcoef(ref, out)[0, 1]) if np.std(ref) > 0 and np.std(out) > 0 else math.nan
    return {
        "active_frame_count": int(np.count_nonzero(active_frames)),
        "active_samples": int(np.count_nonzero(active_samples)),
        "input_projection_gain_db": finite(
            20.0 * math.log10(mix_alpha) if mix_alpha > 0 else -math.inf
        ),
        "projection_gain_db": finite(20.0 * math.log10(alpha) if alpha > 0 else -math.inf),
        "correlation": finite(corr),
        "si_sdr_db": finite(sisdr),
        "output_rms_dbfs": finite(rms_dbfs(out)),
        "output_peak": finite(float(np.max(np.abs(out)))),
        "clip_samples": int(np.count_nonzero(np.abs(out) > 1.0)),
        "finite": bool(np.isfinite(out).all()),
    }


def frame_diagnostics(
    reference: np.ndarray,
    mixture: np.ndarray,
    output: np.ndarray,
    reference_frame_db: np.ndarray,
    active: np.ndarray,
) -> dict[str, Any]:
    ref_frames = np.asarray(reference, dtype=np.float64).reshape(-1, FRAME)
    mix_frames = np.asarray(mixture, dtype=np.float64).reshape(-1, FRAME)
    out_frames = np.asarray(output, dtype=np.float64).reshape(-1, FRAME)
    mix_db = 20.0 * np.log10(
        np.maximum(np.sqrt(np.mean(mix_frames * mix_frames, axis=1)), 1e-12)
    )
    out_db = 20.0 * np.log10(
        np.maximum(np.sqrt(np.mean(out_frames * out_frames, axis=1)), 1e-12)
    )
    gain_db = out_db - reference_frame_db
    delta_db = out_db - mix_db
    active_indices = np.flatnonzero(active)
    ranked = active_indices[np.argsort(reference_frame_db[active_indices], kind="stable")]
    low_rank = int(math.floor(0.10 * ranked.size))
    high_rank = int(math.ceil(0.30 * ranked.size))
    high_rank = min(ranked.size, max(low_rank + 1, high_rank))
    weak_indices = np.sort(ranked[low_rank:high_rank])
    last_indices = active_indices[-3:]

    def rows(indices: Sequence[int]) -> list[dict[str, Any]]:
        return [
            {
                "frame_index": int(index),
                "start_s": float(index * FRAME / RATE),
                "reference_dbfs": finite(reference_frame_db[index]),
                "mixture_dbfs": finite(mix_db[index]),
                "output_dbfs": finite(out_db[index]),
                "gain_db": finite(gain_db[index]),
                "delta_db": finite(delta_db[index]),
            }
            for index in indices
        ]

    weak_rows = rows(weak_indices)
    last_rows = rows(last_indices)
    return {
        "activity_definition": "reference 20 ms RMS > -55 dBFS",
        "frame_samples": FRAME,
        "weak_definition": "10th through 30th percentile of active-frame reference level",
        "weak_rank": [low_rank, high_rank],
        "weak_frames": weak_rows,
        "weak_gain_db_mean": mean_or_none([row["gain_db"] for row in weak_rows]),
        "weak_delta_db_mean": mean_or_none([row["delta_db"] for row in weak_rows]),
        "last_active_frames": last_rows,
        "last_active_gain_db_mean": mean_or_none([row["gain_db"] for row in last_rows]),
        "last_active_delta_db_mean": mean_or_none([row["delta_db"] for row in last_rows]),
    }


def level_metrics(input_samples: np.ndarray, output_samples: np.ndarray) -> dict[str, Any]:
    input_values = np.asarray(input_samples, dtype=np.float64)
    output_values = np.asarray(output_samples, dtype=np.float64)
    input_rms = rms_dbfs(input_values)
    output_rms = rms_dbfs(output_values)
    return {
        "input_samples": int(input_values.size),
        "output_samples": int(output_values.size),
        "input_rms_dbfs": finite(input_rms),
        "output_rms_dbfs": finite(output_rms),
        "level_change_db": finite(output_rms - input_rms),
        "input_peak": finite(float(np.max(np.abs(input_values)))),
        "output_peak": finite(float(np.max(np.abs(output_values)))),
        "output_finite": bool(np.isfinite(output_values).all()),
        "output_clip_samples": int(np.count_nonzero(np.abs(output_values) > 1.0)),
    }


def command_record(
    command: Sequence[str],
    cwd: Path,
    probe_sha256: str,
    params: Mapping[str, Any],
    input_record: Mapping[str, Any],
) -> dict[str, Any]:
    full_parameters = {
        **dict(params),
        "rate": RATE,
        "floor_db": FLOOR_DB,
    }
    return {
        "argv": list(command),
        "argv_sha256": sha256_json(list(command)),
        "cwd": str(cwd),
        "probe_sha256": probe_sha256,
        "parameters": full_parameters,
        "parameters_sha256": sha256_json(full_parameters),
        "input_sha256": input_record["sha256"],
    }


def run_probe(
    *,
    probe: Path,
    probe_sha256: str,
    input_path: Path,
    input_record: Mapping[str, Any],
    output_path: Path,
    params: Mapping[str, float],
    cwd: Path,
    expected_samples: int = SAMPLES,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(probe),
        "--input-f32",
        str(input_path),
        "--output-f32",
        str(output_path),
        "--rate",
        str(RATE),
        "--noise-update-scale",
        format(float(params["noise_update_scale"]), ".9g"),
        "--speech-recovery",
        format(float(params["speech_recovery"]), ".9g"),
        "--release-ms",
        format(float(params["release_ms"]), ".9g"),
        "--floor-db",
        format(FLOOR_DB, ".9g"),
    ]
    record = command_record(command, cwd, probe_sha256, params, input_record)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise EvaluationError(f"probe timed out after {PROBE_TIMEOUT_SECONDS}s: {command}") from error
    record["seconds"] = time.monotonic() - started
    record["exit_code"] = completed.returncode
    record["stdout"] = completed.stdout
    record["stderr"] = completed.stderr
    record["stdout_sha256"] = sha256_bytes(completed.stdout.encode("utf-8"))
    if completed.returncode != 0:
        raise EvaluationError(
            f"probe failed with exit {completed.returncode}: {command}\n{completed.stderr}"
        )
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise EvaluationError(f"probe stdout is not JSON: {completed.stdout!r}") from error
    if not output_path.is_file():
        raise EvaluationError(f"probe did not create output: {output_path}")
    output_bytes = output_path.read_bytes()
    expected_bytes = (expected_samples + HOP) * 4
    if len(output_bytes) != expected_bytes:
        raise EvaluationError(
            f"probe output length mismatch: {output_path}, expected {expected_bytes}, got {len(output_bytes)}"
        )
    values = read_f32(output_path)
    if values.size != expected_samples + HOP:
        raise EvaluationError(f"probe output sample count mismatch: {output_path}")
    if not np.isfinite(values).all():
        raise EvaluationError(f"probe output contains non-finite samples: {output_path}")
    if int(report.get("delay", -1)) != HOP:
        raise EvaluationError(f"probe reported unexpected delay: {report.get('delay')}")
    if int(report.get("input_samples", -1)) != expected_samples:
        raise EvaluationError(f"probe reported unexpected input sample count: {report.get('input_samples')}")
    if int(report.get("output_samples", -1)) != expected_samples + HOP:
        raise EvaluationError(f"probe reported unexpected output sample count: {report.get('output_samples')}")
    record["report"] = report
    record["output"] = {
        "path": str(output_path),
        "sha256": sha256_bytes(output_bytes),
        "bytes": len(output_bytes),
        "samples": int(values.size),
    }
    report_path = output_path.with_suffix(".report.json")
    write_json(report_path, report)
    record["report_sha256"] = sha256_file(report_path)
    record["_samples"] = values
    return record


def delay_crop(samples: np.ndarray, delay: int = HOP) -> np.ndarray:
    if samples.size != SAMPLES + delay:
        raise EvaluationError(
            f"delay crop requires {SAMPLES + delay} samples, got {samples.size}"
        )
    cropped = np.ascontiguousarray(samples[delay : delay + SAMPLES], dtype=np.float64)
    if cropped.size != SAMPLES:
        raise EvaluationError("delay crop did not produce exactly 12 seconds")
    return cropped


def score_output(scorer: Any, output: np.ndarray) -> dict[str, Any]:
    scorer_input = np.ascontiguousarray(output, dtype=np.float64)
    if scorer_input.size != SAMPLES:
        raise EvaluationError("scorer input is not exactly 12 seconds")
    row = scorer.score_array(scorer_input, RATE, kind="near_only", use_cache=False)
    if not isinstance(row, Mapping) or not row.get("available"):
        raise EvaluationError(f"DNSMOS scorer unavailable: {row}")
    if row.get("model_sha256") != EXPECTED_MODEL_SHA256:
        raise EvaluationError(
            f"DNSMOS model SHA mismatch: expected {EXPECTED_MODEL_SHA256}, got {row.get('model_sha256')}"
        )
    if row.get("cache_hit"):
        raise EvaluationError("DNSMOS unexpectedly reported a cache hit")
    scores = row.get("poly")
    if not isinstance(scores, Mapping) or any(key not in scores for key in ("SIG", "BAK", "OVRL")):
        raise EvaluationError(f"DNSMOS result lacks poly SIG/BAK/OVRL: {row}")
    return {
        "input_samples": int(scorer_input.size),
        "input_seconds": scorer_input.size / RATE,
        "input_sha256": sha256_bytes(np.asarray(scorer_input, dtype="<f8").tobytes()),
        "use_cache": False,
        "cache_hit": False,
        "model_sha256": row.get("model_sha256"),
        "preprocessing_sha256": row.get("preprocessing_sha256"),
        "num_scored_hops": int(row.get("num_scored_hops", 0)),
        "scores": {key: finite(float(scores[key])) for key in ("SIG", "BAK", "OVRL")},
        "raw": json_safe(row.get("raw")),
        "poly": json_safe(scores),
    }


def exact_params(variant: Variant) -> dict[str, float]:
    return {
        "noise_update_scale": float(variant.noise_update_scale),
        "speech_recovery": float(variant.speech_recovery),
        "release_ms": float(variant.release_ms),
    }


def is_disabled(variant: Variant) -> bool:
    return (
        variant.noise_update_scale == 1.0
        and variant.speech_recovery == 0.0
        and variant.release_ms == 60.0
    )


def load_scorer(output: Path, project_root: Path) -> tuple[Any, dict[str, Any]]:
    sys.path.insert(0, str(project_root))
    try:
        from cpp.tools.ecnr_perceptual import DNSMOS
    except Exception as error:
        raise EvaluationError(f"cannot import DNSMOS scorer: {error}") from error
    unused_cache = output / "work" / "scorer-cache-not-written"
    try:
        scorer = DNSMOS(cache_dir=unused_cache)
    except Exception as error:
        raise EvaluationError(f"cannot initialize DNSMOS scorer: {error}") from error
    preprocessing = json_safe(getattr(scorer, "preprocessing", None))
    if not isinstance(preprocessing, Mapping):
        raise EvaluationError("DNSMOS scorer does not expose preprocessing identity")
    if preprocessing.get("model_sha256") != EXPECTED_MODEL_SHA256:
        raise EvaluationError(
            f"DNSMOS preprocessing model SHA mismatch: {preprocessing.get('model_sha256')}"
        )
    assets_dir = Path(getattr(scorer, "assets_dir"))
    model_path = assets_dir / "sig_bak_ovr.onnx"
    if not model_path.is_file() or sha256_file(model_path) != EXPECTED_MODEL_SHA256:
        raise EvaluationError(f"DNSMOS ONNX model SHA mismatch: {model_path}")
    model_record = {
        "class": f"{type(scorer).__module__}.{type(scorer).__qualname__}",
        "assets_dir": str(assets_dir),
        "unused_cache_dir": str(unused_cache),
        "unused_cache_dir_created": unused_cache.exists(),
        "expected_sha256": EXPECTED_MODEL_SHA256,
        "preprocessing": preprocessing,
        "assets": {
            name: {
                "path": str(assets_dir / name),
                "exists": (assets_dir / name).is_file(),
                "sha256": sha256_file(assets_dir / name) if (assets_dir / name).is_file() else None,
            }
            for name in ("manifest.json", "dnsmos_local.py", "sig_bak_ovr.onnx")
        },
    }
    return scorer, model_record


def variant_output_record(
    variant: Variant,
    run: Mapping[str, Any],
    reference: np.ndarray,
    mixture: np.ndarray,
    reference_frame_db: np.ndarray,
    active: np.ndarray,
    scorer: Any,
    mos_eligible: bool,
    aliases: Sequence[str] = (),
) -> dict[str, Any]:
    output = delay_crop(run["_samples"], int(run["report"]["delay"]))
    record: dict[str, Any] = {
        "name": variant.name,
        "aliases": list(aliases),
        "parameters": exact_params(variant),
        "parameters_sha256": sha256_json(exact_params(variant)),
        "delay_samples": int(run["report"]["delay"]),
        "delay_crop_samples": int(output.size),
        "delay_crop_seconds": output.size / RATE,
        "probe": {key: value for key, value in run.items() if key != "_samples"},
        "metrics": projection_metrics(reference, mixture, output, active),
        "frames": frame_diagnostics(
            reference, mixture, output, reference_frame_db, active
        ),
    }
    if mos_eligible:
        record["dnsmos"] = score_output(scorer, output)
    else:
        record["dnsmos"] = {
            "not_scored": True,
            "reason": "synthetic diagnostic source is not a MOS source",
        }
    return record


def run_case(
    spec: CaseSpec,
    reference: np.ndarray,
    *,
    probe: Path,
    probe_sha256: str,
    original_probe: Path,
    original_probe_sha256: str,
    output: Path,
    project_root: Path,
    variants: Sequence[Variant],
    scorer: Any,
) -> dict[str, Any]:
    case_dir = output / "work" / safe_slug(spec.case_id)
    case_dir.mkdir(parents=True, exist_ok=True)
    reference = np.ascontiguousarray(reference, dtype=np.float64)
    noise = make_noise(reference, spec.noise_seed, spec.target_snr_db)
    reference_f32 = np.asarray(reference, dtype="<f4")
    mixture = np.ascontiguousarray(reference_f32 + noise, dtype="<f4")
    if not np.isfinite(mixture).all():
        raise EvaluationError(f"mixture is non-finite for {spec.case_id}")
    reference_record = write_f32(case_dir / "reference.f32", reference_f32)
    noise_record = write_f32(case_dir / "noise.f32", noise)
    mixture_record = write_f32(case_dir / "mixture.f32", mixture)
    frame_db, active = frame_activity(reference)

    disabled = Variant("disabled", 1.0, 0.0, 60.0)
    original_run = run_probe(
        probe=original_probe,
        probe_sha256=original_probe_sha256,
        input_path=case_dir / "mixture.f32",
        input_record=mixture_record,
        output_path=case_dir / "original-baseline.f32",
        params=exact_params(disabled),
        cwd=project_root,
    )
    disabled_run = run_probe(
        probe=probe,
        probe_sha256=probe_sha256,
        input_path=case_dir / "mixture.f32",
        input_record=mixture_record,
        output_path=case_dir / "candidate-disabled.f32",
        params=exact_params(disabled),
        cwd=project_root,
    )
    original_bytes = Path(original_run["output"]["path"]).read_bytes()
    disabled_bytes = Path(disabled_run["output"]["path"]).read_bytes()
    parity = {
        "float_bytes_exact": original_bytes == disabled_bytes,
        "original_output_sha256": original_run["output"]["sha256"],
        "disabled_output_sha256": disabled_run["output"]["sha256"],
        "original_delay": int(original_run["report"]["delay"]),
        "disabled_delay": int(disabled_run["report"]["delay"]),
        "delay_equal": int(original_run["report"]["delay"])
        == int(disabled_run["report"]["delay"]),
    }
    if not parity["float_bytes_exact"]:
        raise EvaluationError(f"original/disabled float bytes differ for {spec.case_id}")
    if not parity["delay_equal"]:
        raise EvaluationError(f"original/disabled delays differ for {spec.case_id}")

    disabled_score_record = variant_output_record(
        disabled,
        disabled_run,
        reference,
        mixture,
        frame_db,
        active,
        scorer,
        spec.mos_eligible,
        aliases=["candidate baseline control"],
    )
    variant_records: list[dict[str, Any]] = []
    noise_control_records: list[dict[str, Any]] = []
    for variant in variants:
        if is_disabled(variant):
            run = disabled_run
            aliases = ["reuses candidate-disabled output"]
        else:
            run = run_probe(
                probe=probe,
                probe_sha256=probe_sha256,
                input_path=case_dir / "mixture.f32",
                input_record=mixture_record,
                output_path=case_dir / f"variant-{safe_slug(variant.name)}.f32",
                params=exact_params(variant),
                cwd=project_root,
            )
            aliases = []
        variant_records.append(
            variant_output_record(
                variant,
                run,
                reference,
                mixture,
                frame_db,
                active,
                scorer,
                spec.mos_eligible,
                aliases=aliases,
            )
        )

        noise_run = run_probe(
            probe=probe,
            probe_sha256=probe_sha256,
            input_path=case_dir / "noise.f32",
            input_record=noise_record,
            output_path=case_dir / f"noise-control-{safe_slug(variant.name)}.f32",
            params=exact_params(variant),
            cwd=project_root,
        )
        noise_output = delay_crop(noise_run["_samples"], int(noise_run["report"]["delay"]))
        noise_control_records.append(
            {
                "variant": variant.name,
                "parameters_sha256": sha256_json(exact_params(variant)),
                "dnsmos_scored": False,
                "delay_crop_samples": int(noise_output.size),
                "probe": {key: value for key, value in noise_run.items() if key != "_samples"},
                "levels": level_metrics(noise.astype(np.float64), noise_output),
            }
        )

    all_delays = {
        int(original_run["report"]["delay"]),
        int(disabled_run["report"]["delay"]),
        *(int(record["probe"]["report"]["delay"]) for record in variant_records),
        *(int(record["probe"]["report"]["delay"]) for record in noise_control_records),
    }
    if all_delays != {HOP}:
        raise EvaluationError(f"inconsistent probe delay for {spec.case_id}: {all_delays}")

    return {
        "case_id": spec.case_id,
        "role": spec.role,
        "source_case_id": spec.source_case_id,
        "mos_eligible": spec.mos_eligible,
        "synthetic": spec.synthetic,
        "source": {
            "reference_path": str(spec.reference_path) if spec.reference_path else None,
            "expected_reference_sha256": spec.reference_sha256,
            "assessment_path": str(spec.assessment_path) if spec.assessment_path else None,
            "assessment_sha256": sha256_file(spec.assessment_path) if spec.assessment_path else None,
            "noise_seed": spec.noise_seed,
            "noise_type": "deterministic iid zero-mean unit-Gaussian, full-clip RMS SNR",
            "target_snr_db": spec.target_snr_db,
            "actual_snr_db": actual_snr_db(reference_f32, noise),
            "noise_carrier_shared_across_variants": True,
        },
        "reference_f32": reference_record,
        "mixture_f32": mixture_record,
        "noise_f32": noise_record,
        "reference_levels": {
            "rms_dbfs": finite(rms_dbfs(reference)),
            "peak": finite(float(np.max(np.abs(reference)))),
            "clip_samples": int(np.count_nonzero(np.abs(reference) > 1.0)),
        },
        "activity": {
            "threshold_dbfs": ACTIVITY_DBFS,
            "frame_samples": FRAME,
            "frame_count": int(frame_db.size),
            "active_frame_count": int(np.count_nonzero(active)),
            "active_fraction": float(np.mean(active)),
        },
        "original_baseline": {key: value for key, value in original_run.items() if key != "_samples"},
        "candidate_disabled": {key: value for key, value in disabled_run.items() if key != "_samples"},
        "parity": parity,
        "disabled_control_score_and_metrics": disabled_score_record,
        "variants": variant_records,
        "noise_only_controls": noise_control_records,
    }


def make_steady_tone() -> np.ndarray:
    time_axis = np.arange(SAMPLES, dtype=np.float64) / RATE
    tone = 10 ** (-30.0 / 20.0) * np.sqrt(2.0) * np.sin(2 * np.pi * 1000.0 * time_axis)
    return np.ascontiguousarray(tone, dtype="<f4")


def run_steady_tone_controls(
    *,
    probe: Path,
    probe_sha256: str,
    output: Path,
    project_root: Path,
    variants: Sequence[Variant],
) -> list[dict[str, Any]]:
    tone = make_steady_tone()
    tone_dir = output / "work" / "steady-tone-control"
    tone_dir.mkdir(parents=True, exist_ok=True)
    tone_record = write_f32(tone_dir / "steady-tone-1khz-minus30dbfs.f32", tone)
    records: list[dict[str, Any]] = []
    for variant in variants:
        run = run_probe(
            probe=probe,
            probe_sha256=probe_sha256,
            input_path=tone_dir / "steady-tone-1khz-minus30dbfs.f32",
            input_record=tone_record,
            output_path=tone_dir / f"output-{safe_slug(variant.name)}.f32",
            params=exact_params(variant),
            cwd=project_root,
        )
        cropped = delay_crop(run["_samples"], int(run["report"]["delay"]))
        records.append(
            {
                "variant": variant.name,
                "parameters_sha256": sha256_json(exact_params(variant)),
                "dnsmos_scored": False,
                "probe": {key: value for key, value in run.items() if key != "_samples"},
                "levels": level_metrics(tone.astype(np.float64), cropped),
            }
        )
    return records


def hash_artifacts(paths: Iterable[Path]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in sorted({str(item) for item in paths}):
        item = Path(path)
        result[str(item)] = {
            "exists": item.is_file(),
            "sha256": sha256_file(item) if item.is_file() else None,
        }
    return result


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path.cwd().resolve()
    output = Path(args.output).expanduser().resolve()
    probe = Path(args.probe).expanduser().resolve()
    original_probe = Path(args.original_probe).expanduser().resolve()
    if output.exists():
        raise EvaluationError(f"--output must be a new path: {output}")
    for binary, label in ((probe, "--probe"), (original_probe, "--original-probe")):
        if not binary.is_file():
            raise EvaluationError(f"{label} does not exist: {binary}")
    variants = parse_variants(args.variants)
    output.mkdir(parents=True, exist_ok=False)
    work = output / "work"
    work.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(work / "matplotlib-cache"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(work / "numba-cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(work / "xdg-cache"))

    probe_sha = sha256_file(probe)
    original_probe_sha = sha256_file(original_probe)
    scorer, model_record = load_scorer(output, project_root)

    specs: list[tuple[CaseSpec, np.ndarray]] = []
    dev_manifest_record: dict[str, Any] = {
        "path": str(DEV_MANIFEST),
        "loaded": False,
        "reason": "not requested by mode",
    }
    holdout_specs: list[CaseSpec] = []
    if args.mode in ("dev", "all"):
        dev_spec, dev_manifest_record = load_dev_spec()
        dev_reference, _ = validate_reference(dev_spec.reference_path, dev_spec.reference_sha256)
        specs.append((dev_spec, dev_reference))
        synthetic_spec, synthetic_reference = make_synthetic_case()
        specs.append((synthetic_spec, synthetic_reference))
    if args.mode in ("holdout", "all"):
        holdout_specs = load_holdout_specs()
        loaded_holdouts: dict[Path, np.ndarray] = {}
        for spec in holdout_specs:
            if spec.reference_path not in loaded_holdouts:
                loaded_holdouts[spec.reference_path], _ = validate_reference(
                    spec.reference_path, spec.reference_sha256
                )
            specs.append((spec, loaded_holdouts[spec.reference_path]))

    dev_hashes = {DEV_REFERENCE_SHA256} if args.mode in ("dev", "all") else set()
    holdout_hashes = {spec.reference_sha256 for spec in holdout_specs} if args.mode in ("holdout", "all") else set()
    if dev_hashes & holdout_hashes:
        raise EvaluationError("dev and holdout reference sources overlap")

    tracked_inputs = {
        path
        for path in (
            probe,
            original_probe,
            DEV_MANIFEST,
            *(spec.assessment_path for spec in holdout_specs if spec.assessment_path),
            *(spec.reference_path for spec, _ in specs if spec.reference_path),
        )
        if path is not None
    }
    before_hashes = hash_artifacts(tracked_inputs)
    case_records: list[dict[str, Any]] = []
    for spec, reference in specs:
        case_records.append(
            run_case(
                spec,
                reference,
                probe=probe,
                probe_sha256=probe_sha,
                original_probe=original_probe,
                original_probe_sha256=original_probe_sha,
                output=output,
                project_root=project_root,
                variants=variants,
                scorer=scorer,
            )
        )
    tone_controls = run_steady_tone_controls(
        probe=probe,
        probe_sha256=probe_sha,
        output=output,
        project_root=project_root,
        variants=variants,
    )
    after_hashes = hash_artifacts(tracked_inputs)
    if before_hashes != after_hashes:
        raise EvaluationError("a tracked input/model/probe changed during evaluation")

    all_parity = all(record["parity"]["float_bytes_exact"] for record in case_records)
    all_delays = {
        int(record["parity"]["original_delay"])
        for record in case_records
    } | {
        int(record["parity"]["disabled_delay"])
        for record in case_records
    }
    all_finite_noise = all(
        control["levels"]["output_finite"]
        for record in case_records
        for control in record["noise_only_controls"]
    )
    all_scores_full_12s = all(
        variant["dnsmos"].get("input_samples") == SAMPLES
        for record in case_records
        for variant in record["variants"]
        if record["mos_eligible"]
    )
    model_cache_not_created = not Path(model_record["unused_cache_dir"]).exists()
    verification = {
        "original_disabled_float_bytes_exact_all_cases": all_parity,
        "delay_values": sorted(all_delays),
        "delay_equal_160_all_cases": all_delays == {HOP},
        "noise_only_outputs_finite": all_finite_noise,
        "all_dnsmos_inputs_exactly_12_seconds": all_scores_full_12s,
        "dnsmos_cache_not_created": model_cache_not_created,
        "project_inputs_unchanged": before_hashes == after_hashes,
        "dev_holdout_reference_hashes_disjoint": not (dev_hashes & holdout_hashes),
        "expected_model_sha256": EXPECTED_MODEL_SHA256,
        "python_dont_write_bytecode": os.environ.get("PYTHONDONTWRITEBYTECODE"),
    }
    required_true = (
        "original_disabled_float_bytes_exact_all_cases",
        "delay_equal_160_all_cases",
        "noise_only_outputs_finite",
        "all_dnsmos_inputs_exactly_12_seconds",
        "dnsmos_cache_not_created",
        "project_inputs_unchanged",
        "dev_holdout_reference_hashes_disjoint",
    )
    if not all(verification[key] for key in required_true):
        raise EvaluationError(f"verification gate failed: {verification}")

    return {
        "schema": "itu_omlsa_tuning_evaluation.v1",
        "created_at": utc_now(),
        "mode": args.mode,
        "project_root": str(project_root),
        "python_executable": sys.executable,
        "claim_policy": (
            "Report measurements only. No variant is declared a winner; noise and tone "
            "controls are level diagnostics and are not MOS-scored."
        ),
        "evidence_scope": (
            "One fixed dev reference plus one synthetic diagnostic; six fixed holdout "
            "examples from near_single 02/03/04, with two preassigned seed/SNR pairs each."
        ),
        "variants": [
            {
                "name": variant.name,
                "parameters": exact_params(variant),
                "parameters_sha256": sha256_json(exact_params(variant)),
            }
            for variant in variants
        ],
        "probe": {
            "path": str(probe),
            "sha256": probe_sha,
        },
        "original_probe": {
            "path": str(original_probe),
            "sha256": original_probe_sha,
        },
        "dev_manifest": dev_manifest_record,
        "model": model_record,
        "cases": case_records,
        "steady_tone_controls": tone_controls,
        "tracked_input_hashes_before": before_hashes,
        "tracked_input_hashes_after": after_hashes,
        "verification": verification,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, help="candidate probe binary")
    parser.add_argument("--original-probe", required=True, help="original-control probe binary")
    parser.add_argument("--output", required=True, help="new evidence directory")
    parser.add_argument("--mode", choices=("dev", "holdout", "all"), required=True)
    parser.add_argument(
        "--variants",
        help=(
            "JSON object/list or JSON file. Each value is [noise, speech, release] or "
            "an object with noise_update_scale/speech_recovery/release_ms."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    try:
        evidence = evaluate(args)
        write_json(output / "evidence.json", evidence)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "evidence": str(output / "evidence.json"),
                    "mode": args.mode,
                    "cases": len(evidence["cases"]),
                    "variants": [item["name"] for item in evidence["variants"]],
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        if output.is_dir():
            failure = {
                "schema": "itu_omlsa_tuning_evaluation.failure.v1",
                "created_at": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "mode": args.mode,
                "argv": sys.argv,
            }
            try:
                write_json(output / "failure.json", failure)
            except Exception:
                pass
        print(f"evaluate_omlsa_tuning: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
