#!/usr/bin/env python3
"""Pure signal repairs for the acoustic dataset revision.

The helpers in this module operate on caller-owned arrays and metadata.  They
do not render algorithm output, write dataset artifacts, or claim codec,
vehicle, or ITU-T conformance.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy import signal
import soundfile as sf


PROJECT = Path(__file__).resolve().parents[1]
TRANSPORT_DIRECTION = "far_end_to_rx_input"
PEAK_GUARD = 0.99
ZERO_EPSILON = 1e-15
HIGHWAY_SCENES = ("TBUS", "STRAFFIC")
EXCLUDED_NOISE_SCENES = ("TCAR",)
SOURCE_ROOTS = (
    PROJECT / "datasets/itu-eval-v4-quality-filtered-20261010/sources",
    PROJECT / "datasets/itu-eval-v4-no-passenger-20261011/sources",
    PROJECT / "datasets/itu-eval-v4-curated/sources",
    PROJECT / "datasets/itu-eval-v4-sweeps-pilot/sources",
)


def _array_1d(name: str, value: np.ndarray) -> np.ndarray:
    samples = np.asarray(value, dtype=np.float64)
    if samples.ndim != 1 or not len(samples):
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(samples).all():
        raise ValueError(f"{name} must contain only finite samples")
    return samples.copy()


def _positive_rate(rate: int) -> int:
    if isinstance(rate, (bool, np.bool_)) or not isinstance(
        rate,
        (int, np.integer),
    ):
        raise ValueError("rate must be a positive integer")
    canonical = int(rate)
    if canonical <= 0:
        raise ValueError("rate must be a positive integer")
    return canonical


def _integer_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed,
        (int, np.integer),
    ):
        raise ValueError("seed must be an integer")
    return int(seed)


def _stable_seed(seed: int, *parts: str) -> int:
    text = ":".join((str(_integer_seed(seed)), *parts))
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _rms(values: np.ndarray, mask: np.ndarray | None = None) -> float:
    samples = np.asarray(values, dtype=np.float64)
    if mask is not None:
        samples = samples[np.asarray(mask, dtype=bool)]
    if not samples.size:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def _db(numerator: float, denominator: float) -> float | None:
    if numerator <= ZERO_EPSILON or denominator <= ZERO_EPSILON:
        return None
    return float(20.0 * math.log10(numerator / denominator))


def _sha256_f32le(samples: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(samples, dtype="<f4").tobytes(order="C")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mode_parts(mode: str) -> tuple[str, str]:
    canonical = str(mode).strip().lower().replace("_", "-")
    parts = canonical.split("-")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"unsupported transport mode: {mode!r}")
    bandwidth, transport = parts
    if bandwidth not in {"nb", "wb", "swb"}:
        raise ValueError(f"unsupported bandwidth in mode: {mode!r}")
    if transport not in {"wired", "wireless", "bt"}:
        raise ValueError(f"unsupported transport in mode: {mode!r}")
    if transport == "bt" and bandwidth == "swb":
        raise ValueError("SWB-bt is not part of the NB/WB codec-proxy contract")
    return bandwidth, transport


def _safe_sosfiltfilt(sos: np.ndarray, values: np.ndarray) -> np.ndarray:
    samples = np.asarray(values, dtype=np.float64)
    if len(samples) <= 1:
        return samples.copy()
    padlen = min(max(3 * (2 * len(sos) + 1), 0), len(samples) - 1)
    return np.asarray(
        signal.sosfiltfilt(sos, samples, padlen=padlen),
        dtype=np.float64,
    )


def _band_limit(
    samples: np.ndarray,
    rate: int,
    low_hz: float,
    high_hz: float,
    *,
    order: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    nyquist = rate / 2.0
    applied_high = min(float(high_hz), 0.98 * nyquist)
    applied_low = max(20.0, min(float(low_hz), applied_high * 0.95))
    if applied_low >= applied_high:
        raise ValueError("transport band-limit edges do not fit the sample rate")
    sos = signal.butter(
        order,
        (applied_low, applied_high),
        btype="bandpass",
        fs=rate,
        output="sos",
    )
    filtered = _safe_sosfiltfilt(sos, samples)
    return filtered, {
        "filter": "butterworth_bandpass_sosfiltfilt",
        "order": order,
        "requested_edges_hz": [float(low_hz), float(high_hz)],
        "applied_edges_hz": [applied_low, applied_high],
    }


def _rms_match(
    original: np.ndarray,
    processed: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    input_rms = _rms(original)
    output_rms = _rms(processed)
    if input_rms <= ZERO_EPSILON:
        return np.zeros_like(processed), {
            "input_rms": input_rms,
            "before_rms_correction": output_rms,
            "rms_correction_linear": 1.0,
            "peak_guard_linear": 1.0,
            "output_rms": 0.0,
            "silence_preserved": True,
        }
    if output_rms <= ZERO_EPSILON:
        raise ValueError("transport impairment produced digital silence")

    correction = float(np.clip(input_rms / output_rms, 0.5, 2.0))
    corrected = np.asarray(processed, dtype=np.float64) * correction
    peak = float(np.max(np.abs(corrected)))
    peak_guard = 1.0
    if peak > PEAK_GUARD:
        peak_guard = PEAK_GUARD / peak
        corrected *= peak_guard
    return corrected, {
        "input_rms": input_rms,
        "before_rms_correction": output_rms,
        "rms_correction_linear": correction,
        "peak_guard_linear": float(peak_guard),
        "output_rms": _rms(corrected),
        "silence_preserved": False,
    }


def _plc_frames(
    samples: np.ndarray,
    frame_samples: int,
    lost_frames: set[int],
) -> np.ndarray:
    result = np.asarray(samples, dtype=np.float64).copy()
    frame_count = math.ceil(len(samples) / frame_samples)
    last_good: np.ndarray | None = None
    for frame_index in range(frame_count):
        start = frame_index * frame_samples
        end = min(len(samples), start + frame_samples)
        source_frame = np.asarray(samples[start:end], dtype=np.float64)
        if frame_index not in lost_frames:
            last_good = source_frame.copy()
            continue
        if not np.any(source_frame):
            result[start:end] = 0.0
            last_good = np.zeros_like(source_frame)
            continue

        next_good: np.ndarray | None = None
        for candidate in range(frame_index + 1, frame_count):
            candidate_start = candidate * frame_samples
            candidate_end = min(len(samples), candidate_start + frame_samples)
            if candidate in lost_frames:
                continue
            candidate_frame = np.asarray(
                samples[candidate_start:candidate_end],
                dtype=np.float64,
            )
            if np.any(candidate_frame):
                next_good = candidate_frame
                break

        if last_good is None and next_good is None:
            result[start:end] = 0.0
        elif last_good is None:
            result[start:end] = next_good
        elif next_good is None or len(last_good) != len(source_frame):
            result[start:end] = last_good
        else:
            fade = np.linspace(
                0.0,
                1.0,
                len(source_frame),
                endpoint=False,
                dtype=np.float64,
            )
            result[start:end] = (
                (1.0 - fade) * last_good + fade * next_good
            )
    return result


def _wireless_plc(
    samples: np.ndarray,
    rate: int,
    bandwidth: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    frame_ms = 10.0 if bandwidth == "swb" else 20.0
    frame_samples = max(1, int(round(rate * frame_ms / 1000.0)))
    frame_count = math.ceil(len(samples) / frame_samples)
    rng = np.random.default_rng(
        _stable_seed(seed, "wireless", bandwidth, "burst-plc")
    )
    lost: set[int] = set()
    cursor = 0
    while cursor < frame_count:
        if rng.random() < 0.035:
            burst = min(int(rng.integers(1, 4)), frame_count - cursor)
            lost.update(range(cursor, cursor + burst))
            cursor += burst
        else:
            cursor += 1
    if not lost:
        start = int(rng.integers(0, frame_count))
        lost.add(start)
    output = _plc_frames(samples, frame_samples, lost)
    return output, {
        "frame_ms": frame_ms,
        "frame_samples": frame_samples,
        "frame_count": frame_count,
        "lost_frame_indices": sorted(lost),
        "lost_frame_count": len(lost),
        "burst_packet_loss": True,
        "plc": "last_good_to_next_good_linear_concealment",
        "fixed_frame_jitter_samples": 0,
        "fixed_frame_jitter_applied": False,
    }


def _mu_law_quantize(samples: np.ndarray, mu: int) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(samples, dtype=np.float64)
    peak = float(np.max(np.abs(values)))
    if peak <= ZERO_EPSILON:
        return np.zeros_like(values), {
            "mu": int(mu),
            "levels": int(mu + 1),
            "input_peak": 0.0,
            "normalization_gain": 1.0,
        }
    normalized = values / peak
    magnitude = np.abs(normalized)
    companded = np.sign(normalized) * (
        np.log1p(float(mu) * magnitude) / math.log1p(float(mu))
    )
    levels = int(mu + 1)
    quantized = np.round((companded + 1.0) * 0.5 * (levels - 1))
    quantized = (quantized / (levels - 1)) * 2.0 - 1.0
    expanded_magnitude = np.expm1(np.abs(quantized) * math.log1p(float(mu)))
    expanded = np.sign(quantized) * expanded_magnitude / float(mu)
    return expanded * peak, {
        "mu": int(mu),
        "levels": levels,
        "input_peak": peak,
        "normalization_gain": peak,
    }


def _bluetooth_proxy(
    samples: np.ndarray,
    rate: int,
    bandwidth: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if bandwidth == "nb":
        low_hz, high_hz = 300.0, 3_400.0
        mu = 255
        frame_ms = 30.0
    else:
        low_hz, high_hz = 100.0, 7_000.0
        mu = 2_047
        frame_ms = 20.0

    filtered, filter_info = _band_limit(
        samples,
        rate,
        low_hz,
        high_hz,
        order=6,
    )
    frame_samples = max(1, int(round(rate * frame_ms / 1000.0)))
    frame_count = math.ceil(len(samples) / frame_samples)
    rng = np.random.default_rng(
        _stable_seed(seed, "bluetooth-proxy", bandwidth, "frame-repeat")
    )
    repeat_index = int(rng.integers(max(1, frame_count // 3), frame_count))
    impaired = filtered.copy()
    repeat_start = repeat_index * frame_samples
    repeat_end = min(len(filtered), repeat_start + frame_samples)
    source_start = max(0, repeat_start - frame_samples)
    source_end = source_start + (repeat_end - repeat_start)
    if np.any(filtered[source_start:source_end]):
        impaired[repeat_start:repeat_end] = filtered[source_start:source_end]
    quantized, quantization_info = _mu_law_quantize(impaired, mu)
    return quantized, {
        "band_limit": filter_info,
        "frame_ms": frame_ms,
        "frame_samples": frame_samples,
        "frame_count": frame_count,
        "repeated_frame_index": repeat_index,
        "frame_repeat_plc_proxy": True,
        "quantization": {
            **quantization_info,
            "method": "mu_law_companding_engineering_proxy",
        },
    }


def apply_transport(
    far_end: np.ndarray,
    rate: int,
    mode: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a transport-side far-end proxy without touching truth signals.

    ``mode`` is ``NB/WB/SWB`` crossed with ``wired/wireless/bt``.  ``wired`` is
    sample-for-sample identity.  ``wireless`` applies deterministic burst
    packet loss with simple PLC.  ``NB-bt`` and ``WB-bt`` are explicitly
    engineering proxies using band limiting, frame concealment, and mu-law
    quantization; they are not CVSD or mSBC implementations.
    """
    samples = _array_1d("far_end", far_end)
    canonical_rate = _positive_rate(rate)
    canonical_seed = _integer_seed(seed)
    bandwidth, transport = _mode_parts(mode)
    input_sha = _sha256_f32le(samples)
    base_metadata: dict[str, Any] = {
        "direction": TRANSPORT_DIRECTION,
        "input_role": "far_end",
        "output_role": "rx_input",
        "mode": f"{bandwidth.upper()}-{transport}",
        "bandwidth": bandwidth.upper(),
        "transport": transport,
        "sample_rate_hz": canonical_rate,
        "seed": canonical_seed,
        "input_length_samples": int(len(samples)),
        "output_length_samples": int(len(samples)),
        "input_sha256_f32le": input_sha,
        "codec_conformance": False,
        "actual_codec": False,
        "engineering_proxy": True,
        "not_actual_codec": ["CVSD", "mSBC"],
        "reference_or_noise_modified": False,
        "algorithm_output_modified": False,
        "hard_clipped": False,
        "digital_silence_preserved": bool(not np.any(samples)),
    }

    if not np.any(samples):
        base_metadata.update({
            "transformation_applied": False,
            "reason": "digital_silence",
            "output_sha256_f32le": input_sha,
            "peak": 0.0,
            "rms": 0.0,
        })
        return np.zeros_like(samples), base_metadata

    if transport == "wired":
        output = samples.copy()
        impairment: dict[str, Any] = {
            "transformation_applied": False,
            "impairments": [],
        }
    elif transport == "wireless":
        impaired, impairment = _wireless_plc(
            samples,
            canonical_rate,
            bandwidth,
            canonical_seed,
        )
        output, rms_info = _rms_match(samples, impaired)
        impairment.update(rms_info)
        impairment["transformation_applied"] = True
    else:
        impaired, impairment = _bluetooth_proxy(
            samples,
            canonical_rate,
            bandwidth,
            canonical_seed,
        )
        output, rms_info = _rms_match(samples, impaired)
        impairment.update(rms_info)
        impairment["transformation_applied"] = True
        impairment["proxy_name"] = (
            "narrowband_telephony_engineering_proxy"
            if bandwidth == "nb"
            else "wideband_telephony_engineering_proxy"
        )

    if len(output) != len(samples):
        raise ValueError("transport output length differs from input")
    if not np.isfinite(output).all():
        raise ValueError("transport output contains non-finite samples")
    peak = float(np.max(np.abs(output)))
    if peak > 1.0:
        raise ValueError("transport output exceeds full scale")

    base_metadata.update(impairment)
    base_metadata.update({
        "output_sha256_f32le": _sha256_f32le(output),
        "peak": peak,
        "rms": _rms(output),
        "input_rms": _rms(samples),
        "rms_ratio": (
            _rms(output) / _rms(samples)
            if _rms(samples) > ZERO_EPSILON
            else None
        ),
    })
    return np.asarray(output, dtype=np.float64), base_metadata


def _noise_rows(bank: Any) -> list[dict[str, Any]]:
    raw = getattr(bank, "noises", None)
    if raw is None:
        raise ValueError("source bank does not expose noises")
    rows: list[dict[str, Any]] = []
    if isinstance(raw, Mapping):
        iterator = raw.items()
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        iterator = ((None, row) for row in raw)
    else:
        raise ValueError("source bank noises must be a mapping or sequence")
    for key, raw_row in iterator:
        if not isinstance(raw_row, Mapping):
            raise ValueError("source bank noise rows must be mappings")
        row = dict(raw_row)
        if "path" not in row and key is not None:
            row["path"] = str(key)
        if "path" not in row:
            raise ValueError("source bank noise row lacks path")
        rows.append(row)
    return rows


def _row_scene(row: Mapping[str, Any]) -> str:
    scene = str(row.get("scene", "")).strip().upper()
    if scene:
        return scene
    path = str(row.get("path", "")).replace("\\", "/")
    basename = path.rsplit("/", 1)[-1].upper()
    for candidate in (*HIGHWAY_SCENES, *EXCLUDED_NOISE_SCENES):
        if candidate in basename:
            return candidate
    return ""


def _candidate_source_roots(bank: Any) -> list[Path]:
    candidates: list[Path] = []
    for attribute in ("speech_root", "source_root", "root", "dataset_root"):
        value = getattr(bank, attribute, None)
        if value is not None:
            candidates.append(Path(value))
    module = sys.modules.get(type(bank).__module__)
    if module is not None:
        value = getattr(module, "SPEECH_ROOT", None)
        if value is not None:
            candidates.append(Path(value))
    candidates.extend(SOURCE_ROOTS)
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def _resolve_source_path(bank: Any, row: Mapping[str, Any]) -> Path:
    for key in ("absolute_path", "file_path", "resolved_path"):
        raw = row.get(key)
        if raw is not None:
            candidate = Path(str(raw))
            if candidate.is_file():
                return candidate.resolve()
    relative = Path(str(row["path"]))
    if relative.is_absolute() and relative.is_file():
        return relative.resolve()
    for root in _candidate_source_roots(bank):
        candidate = root / relative
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"cannot resolve source-bank noise path: {row['path']}")


def _read_verified_noise(
    bank: Any,
    row: Mapping[str, Any],
) -> tuple[np.ndarray, int, Path, str]:
    expected = str(row.get("sha256", "")).strip().lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError(f"source-bank noise row lacks a valid sha256: {row.get('path')}")
    path = _resolve_source_path(bank, row)
    actual = _sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"source SHA mismatch for {path}: expected {expected}, got {actual}"
        )
    samples, source_rate = sf.read(path, always_2d=True, dtype="float64")
    waveform = np.asarray(samples[:, 0], dtype=np.float64)
    if not len(waveform) or not np.isfinite(waveform).all():
        raise ValueError(f"source noise is empty or non-finite: {path}")
    declared_rate = row.get("sample_rate")
    if declared_rate is not None and int(declared_rate) != int(source_rate):
        raise ValueError(
            f"source sample rate mismatch for {path}: "
            f"manifest {declared_rate}, WAV {source_rate}"
        )
    return waveform, int(source_rate), path, actual


def _crop_noise_component(
    recorded: np.ndarray,
    source_rate: int,
    target_rate: int,
    target_frames: int,
    *,
    scene: str,
    mode: str,
    seed: int,
) -> tuple[np.ndarray, int, int]:
    divisor = math.gcd(source_rate, target_rate)
    up = target_rate // divisor
    down = source_rate // divisor
    source_frames = int(math.ceil(target_frames * down / up)) + 64
    if source_frames > len(recorded):
        raise ValueError(
            f"{scene} source is too short for a non-looping crop: "
            f"need {source_frames}, have {len(recorded)}"
        )
    rng = np.random.default_rng(
        _stable_seed(seed, "highway-noise", mode, scene, "crop")
    )
    maximum_start = len(recorded) - source_frames
    start = int(rng.integers(0, maximum_start + 1))
    crop = np.asarray(
        recorded[start:start + source_frames],
        dtype=np.float64,
    )
    if source_rate != target_rate:
        resampled = signal.resample_poly(crop, up, down)
    else:
        resampled = crop.copy()
    padding = max(0, target_frames - len(resampled))
    if padding:
        resampled = np.pad(resampled, (0, padding), mode="constant")
    resampled = np.asarray(resampled[:target_frames], dtype=np.float64)
    return resampled, start, padding


def highway_noise(
    bank: Any,
    mode: str,
    rate: int,
    seed: int,
    duration_s: float = 12.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Mix hash-verified DEMAND TBUS and STRAFFIC recordings.

    The result is a non-looping, unit-scale engineering noise bed.  It never
    substitutes a white-noise fraction for vehicle noise and never claims a
    measured highway speed or vehicle condition.
    """
    canonical_rate = _positive_rate(rate)
    canonical_seed = _integer_seed(seed)
    bandwidth, transport = _mode_parts(mode)
    canonical_mode = f"{bandwidth.upper()}-{transport}"
    if (
        isinstance(duration_s, (bool, np.bool_))
        or not isinstance(duration_s, (int, float, np.integer, np.floating))
        or not math.isfinite(float(duration_s))
        or float(duration_s) <= 0.0
    ):
        raise ValueError("duration_s must be a positive finite number")
    target_frames = int(round(float(duration_s) * canonical_rate))
    if target_frames <= 0:
        raise ValueError("duration_s is too short at the requested rate")

    rows = _noise_rows(bank)
    by_scene: dict[str, dict[str, Any]] = {}
    for row in rows:
        scene = _row_scene(row)
        if scene in HIGHWAY_SCENES and scene not in by_scene:
            by_scene[scene] = row
    missing = [scene for scene in HIGHWAY_SCENES if scene not in by_scene]
    if missing:
        raise ValueError(
            "source bank lacks required independent highway-noise scenes: "
            + ", ".join(missing)
        )

    tcar_hashes = {
        str(row.get("sha256", "")).strip().lower()
        for row in rows
        if _row_scene(row) in EXCLUDED_NOISE_SCENES
    }
    components: list[dict[str, Any]] = []
    component_arrays: dict[str, np.ndarray] = {}
    component_paths: set[Path] = set()
    component_hashes: set[str] = set()
    for scene in HIGHWAY_SCENES:
        row = by_scene[scene]
        recorded, source_rate, path, digest = _read_verified_noise(bank, row)
        if "TCAR" in path.name.upper():
            raise ValueError(f"{scene} resolves to a TCAR source: {path}")
        if digest in tcar_hashes:
            raise ValueError(f"{scene} reuses the excluded TCAR source hash")
        if path in component_paths or digest in component_hashes:
            raise ValueError("TBUS and STRAFFIC components are not independent")
        component_paths.add(path)
        component_hashes.add(digest)

        crop, start, padding = _crop_noise_component(
            recorded,
            source_rate,
            canonical_rate,
            target_frames,
            scene=scene,
            mode=canonical_mode,
            seed=canonical_seed,
        )
        crop -= float(np.mean(crop))
        source_crop_rms = _rms(crop)
        if source_crop_rms <= ZERO_EPSILON:
            raise ValueError(f"{scene} crop has no observable AC RMS")
        unit = crop / source_crop_rms
        component_arrays[scene] = unit
        components.append({
            "scene": scene,
            "scenario": "sce3_highway_engineering_mix",
            "source_kind": "hash_verified_real_noise_component",
            "path": str(path),
            "sha256": digest,
            "source_sample_rate_hz": int(source_rate),
            "target_sample_rate_hz": canonical_rate,
            "start_sample": int(start),
            "source_samples_selected": int(
                math.ceil(
                    target_frames
                    * (source_rate // math.gcd(source_rate, canonical_rate))
                    / (canonical_rate // math.gcd(source_rate, canonical_rate))
                )
                + 64
            ),
            "target_samples_selected": int(target_frames),
            "resampling_padding_samples": int(padding),
            "source_crop_rms": source_crop_rms,
            "unit_rms_normalization_gain_linear": 1.0 / source_crop_rms,
            "repeated": False,
            "within_case_source_repetition": False,
            "actual_speed_recording": False,
            "actual_vehicle_recording": False,
            "license": row.get("license"),
            "dataset": row.get("dataset"),
            "attribution": row.get("attribution"),
            "source_page": row.get("source_page"),
        })

    used = getattr(bank, "used", None)
    if isinstance(used, dict):
        for component in components:
            used[str(component["path"])] = deepcopy(component)

    mix_rng = np.random.default_rng(
        _stable_seed(canonical_seed, "highway-noise", canonical_mode, "mix")
    )
    base_weights = {"TBUS": 0.62, "STRAFFIC": 0.38}
    jitter = float(mix_rng.uniform(-0.08, 0.08))
    raw_weights = {
        "TBUS": max(0.05, base_weights["TBUS"] + jitter),
        "STRAFFIC": max(0.05, base_weights["STRAFFIC"] - jitter),
    }
    weight_sum = sum(raw_weights.values())
    power_weights = {
        scene: value / weight_sum for scene, value in raw_weights.items()
    }
    mix = np.zeros(target_frames, dtype=np.float64)
    for scene, values in component_arrays.items():
        amplitude = math.sqrt(power_weights[scene])
        mix += amplitude * values
        for component in components:
            if component["scene"] == scene:
                component["power_weight"] = power_weights[scene]
                component["amplitude_weight"] = amplitude

    mix -= float(np.mean(mix))
    low_cutoff = min(180.0, 0.05 * canonical_rate)
    low_sos = signal.butter(2, low_cutoff, btype="lowpass", fs=canonical_rate, output="sos")
    low_component = _safe_sosfiltfilt(low_sos, mix)
    wind_gain = float(mix_rng.uniform(0.05, 0.10))
    shaped = mix + wind_gain * low_component
    dc_sos = signal.butter(2, 20.0, btype="highpass", fs=canonical_rate, output="sos")
    shaped = _safe_sosfiltfilt(dc_sos, shaped)
    shaped_rms = _rms(shaped)
    if shaped_rms <= ZERO_EPSILON:
        raise ValueError("highway-noise mix has no observable AC RMS")
    shaped /= shaped_rms
    peak = float(np.max(np.abs(shaped)))
    peak_guard = 1.0
    if peak > 0.98:
        peak_guard = 0.98 / peak
        shaped *= peak_guard
    if not np.isfinite(shaped).all():
        raise ValueError("highway-noise mix contains non-finite samples")
    if len(shaped) != target_frames:
        raise ValueError("highway-noise output length differs from requested duration")

    primary = components[0]
    output_rms = _rms(shaped)
    info: dict[str, Any] = {
        "source_kind": "hash_verified_real_noise_mix",
        "mode": canonical_mode,
        "sample_rate_hz": canonical_rate,
        "duration_s": round(target_frames / canonical_rate, 9),
        "target_frames": int(target_frames),
        "start_sample": int(primary["start_sample"]),
        "crop_offset_samples": int(primary["start_sample"]),
        "path": str(primary["path"]),
        "sha256": str(primary["sha256"]),
        "source_sample_rate_hz": int(primary["source_sample_rate_hz"]),
        "noise_crop_seed": canonical_seed,
        "source_scenes": list(HIGHWAY_SCENES),
        "components": components,
        "excluded_source_scenes": list(EXCLUDED_NOISE_SCENES),
        "tcar_excluded": True,
        "mix": {
            "method": "sqrt_power_weighted_sum_of_unit_rms_recorded_crops",
            "power_weights": power_weights,
            "transparent_component_metadata": True,
            "synthetic_noise_fraction": 0.0,
        },
        "wind_shaping": {
            "method": "bounded_low_frequency_reinforcement_and_dc_removal",
            "lowpass_cutoff_hz": low_cutoff,
            "lowpass_order": 2,
            "reinforcement_gain_linear": wind_gain,
            "maximum_reinforcement_gain_linear": 0.10,
            "highpass_cutoff_hz": 20.0,
            "synthetic_component_used": False,
        },
        "output_rms": output_rms,
        "peak": float(np.max(np.abs(shaped))),
        "peak_guard_linear": peak_guard,
        "repeated": False,
        "looped": False,
        "within_case_source_repetition": False,
        "source_reuse_allowed_across_cases": True,
        "actual_highway_recording": False,
        "actual_vehicle_speed_recording": False,
        "actual_speed_recording": False,
        "recorded_speed_known": False,
        "engineering_proxy": True,
        "evidence_scope": (
            "DEMAND TBUS plus STRAFFIC engineering mix; not a measured "
            "vehicle speed, cabin, or highway recording"
        ),
    }
    return np.asarray(shaped, dtype=np.float64), info


def _optional_number(value: Any, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field} must be numeric or null")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric or null") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite or null")
    return 0.0 if number == 0.0 else number


def add_ser_aliases(design: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve legacy ``ser_db_*`` values and add unambiguous sign aliases.

    Legacy fields retain their historical echo/near convention:
    ``20*log10(RMS(echo)/RMS(near))``.  Therefore the explicit near-to-echo
    SER aliases are the negated legacy values, while the echo-to-near ESR
    aliases preserve them.
    """
    if not isinstance(design, Mapping):
        raise ValueError("design must be a mapping")
    result = deepcopy(dict(design))
    applicable = design.get("ser_db_target_applicable")
    if applicable is False:
        target = effective = measured = None
    else:
        target = _optional_number(
            design.get("ser_db_nominal_target", design.get("ser_db_target")),
            "ser_db_nominal_target",
        )
        effective = _optional_number(
            design.get("ser_db_effective_target"),
            "ser_db_effective_target",
        )
        measured = _optional_number(
            design.get("ser_db_measured"),
            "ser_db_measured",
        )

    def inverted(value: float | None) -> float | None:
        return None if value is None else (0.0 if value == 0.0 else -value)

    result.update({
        "near_to_echo_ser_db_target": inverted(target),
        "near_to_echo_ser_db_effective": inverted(effective),
        "near_to_echo_ser_db_measured": inverted(measured),
        "echo_to_near_esr_db_target": target,
        "echo_to_near_esr_db_effective": effective,
        "echo_to_near_esr_db_measured": measured,
        "ser_db_legacy_semantics": "echo_to_near_esr_db",
        "near_to_echo_ser_db_definition": (
            "Sign inversion of legacy ser_db_* using the legacy masks; "
            "see ser_db_definition."
        ),
        "echo_to_near_esr_db_definition": (
            "Legacy ser_db_* using the legacy masks; see ser_db_definition."
        ),
        "ser_aliases_definition": (
            "Legacy ser_db_* values are preserved as echo/near; "
            "near_to_echo_ser_db_* is their exact sign inversion with the "
            "same legacy masks, not a newly measured common window."
        ),
    })
    return result


def _mask_from_segments(
    segments: Sequence[Mapping[str, Any]],
    frames: int,
    rate: int,
    labels: frozenset[str],
) -> np.ndarray:
    mask = np.zeros(frames, dtype=bool)
    for segment in segments:
        if not isinstance(segment, Mapping):
            raise ValueError("activity segments must be mappings")
        if str(segment.get("label")) not in labels:
            continue
        start = max(
            0,
            min(frames, int(round(float(segment["start_s"]) * rate))),
        )
        end = max(
            0,
            min(frames, int(round(float(segment["end_s"]) * rate))),
        )
        mask[start:end] = True
    return mask


def _double_talk_mask(
    design: Mapping[str, Any],
    frames: int,
    rate: int,
) -> np.ndarray | None:
    segments = design.get("activity_segments", design.get("segments"))
    if isinstance(segments, Sequence) and not isinstance(segments, (str, bytes)):
        mask = _mask_from_segments(
            segments,
            frames,
            rate,
            frozenset({"double_talk"}),
        )
        if np.any(mask):
            return mask
    masks = design.get("activity_masks")
    if isinstance(masks, Mapping):
        windows = masks.get("double_talk")
        if isinstance(windows, Sequence) and not isinstance(windows, (str, bytes)):
            mask = np.zeros(frames, dtype=bool)
            for window in windows:
                if not isinstance(window, Mapping):
                    raise ValueError("activity mask windows must be mappings")
                start = max(
                    0,
                    min(frames, int(round(float(window["start_s"]) * rate))),
                )
                end = max(
                    0,
                    min(frames, int(round(float(window["end_s"]) * rate))),
                )
                mask[start:end] = True
            if np.any(mask):
                return mask
    return None


def _short_overlap_mask(
    design: Mapping[str, Any],
    frames: int,
    rate: int,
) -> np.ndarray:
    category = str(
        design.get("workload_category", design.get("stimulus_kind", ""))
    )
    if category != "short_overlap":
        raise ValueError(
            "calibrate_short_overlap only accepts workload_category=short_overlap"
        )
    mask = _double_talk_mask(design, frames, rate)
    if mask is None:
        raise ValueError(
            "short_overlap design has no actual double_talk observation window"
        )
    return mask


def _noise_channel_from_recipe(
    noise0: np.ndarray,
    design: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    spatial = design.get("noise_spatial")
    if not isinstance(spatial, Mapping):
        raise ValueError("design lacks noise_spatial metadata")
    recipe = spatial.get("channel_1_recipe")
    if not isinstance(recipe, Mapping):
        raise ValueError("design lacks noise_spatial.channel_1_recipe")
    try:
        rho = float(recipe["rho"])
        seed2 = int(recipe["seed2"])
        normalizer = float(recipe["normalizer"])
        scale = float(recipe["noise_channel_scale_linear"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("dual-microphone noise recipe is incomplete") from error
    if (
        not math.isfinite(rho)
        or not -1.0 < rho < 1.0
        or seed2 != recipe["seed2"]
        or not math.isfinite(normalizer)
        or normalizer <= ZERO_EPSILON
        or not math.isfinite(scale)
        or scale <= ZERO_EPSILON
    ):
        raise ValueError("dual-microphone noise recipe is invalid")
    unit0 = np.asarray(noise0, dtype=np.float64) / scale
    independent = np.random.default_rng(seed2).normal(size=len(unit0))
    mixed = rho * unit0 + math.sqrt(1.0 - rho * rho) * independent
    return (mixed / normalizer * scale), dict(recipe)


def _fir_component(
    source: np.ndarray,
    path: Mapping[str, Any] | None,
) -> np.ndarray:
    component = np.zeros(len(source), dtype=np.float64)
    if path is None:
        return component
    offsets = path.get("tap_samples")
    gains = path.get("tap_gains")
    if (
        not isinstance(offsets, Sequence)
        or isinstance(offsets, (str, bytes))
        or not isinstance(gains, Sequence)
        or isinstance(gains, (str, bytes))
        or len(offsets) != len(gains)
    ):
        raise ValueError("echo path requires matching tap_samples and tap_gains")
    echo_gain = _optional_number(
        path.get("echo_gain_linear", 1.0),
        "echo_gain_linear",
    )
    assert echo_gain is not None
    for raw_offset, raw_gain in zip(offsets, gains, strict=True):
        offset = int(raw_offset)
        gain = float(raw_gain)
        if offset != raw_offset or offset < 0 or not math.isfinite(gain):
            raise ValueError("echo FIR taps are invalid")
        if offset < len(component):
            component[offset:] += source[:len(component) - offset] * gain
    return component * echo_gain


def _echo_components(
    far_end: np.ndarray,
    design: Mapping[str, Any],
    rate: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    echo0 = np.zeros(len(far_end), dtype=np.float64)
    echo1 = np.zeros(len(far_end), dtype=np.float64)
    segments = design.get("echo_path_segments")
    if isinstance(segments, Sequence) and not isinstance(segments, (str, bytes)) and segments:
        global_gain = float(design.get("echo_gain_linear", 1.0))
        if not math.isfinite(global_gain):
            raise ValueError("echo_gain_linear must be finite")
        for segment in segments:
            if not isinstance(segment, Mapping):
                raise ValueError("echo_path_segments entries must be mappings")
            paths = segment.get("paths")
            if not isinstance(paths, Sequence) or len(paths) != 2:
                raise ValueError("echo path segment requires two channel paths")
            mask = np.zeros(len(far_end), dtype=bool)
            start = max(
                0,
                min(
                    len(far_end),
                    int(round(float(segment["start_s"]) * rate)),
                ),
            )
            end = max(
                0,
                min(
                    len(far_end),
                    int(round(float(segment["end_s"]) * rate)),
                ),
            )
            mask[start:end] = True
            echo0 += _fir_component(far_end, paths[0]) * mask * global_gain
            echo1 += _fir_component(far_end, paths[1]) * mask * global_gain
        return echo0, echo1, {
            "representation": "echo_path_segments",
            "segment_count": len(segments),
            "echo_gain_linear": global_gain,
        }

    paths = design.get("echo_paths")
    if isinstance(paths, Sequence) and not isinstance(paths, (str, bytes)) and paths:
        if len(paths) != 2:
            raise ValueError("echo_paths must contain zero or two paths")
        echo0 = _fir_component(far_end, paths[0])
        echo1 = _fir_component(far_end, paths[1])
        return echo0, echo1, {
            "representation": "echo_paths",
            "segment_count": 1,
        }
    return echo0, echo1, {
        "representation": "none",
        "segment_count": 0,
    }


def _delayed(samples: np.ndarray, delay: int) -> np.ndarray:
    if delay < 0 or delay >= len(samples):
        raise ValueError("near arrival delay must be inside the signal")
    result = np.zeros_like(samples)
    result[delay:] = samples[:len(samples) - delay]
    return result


def _rebuild_microphone(
    near: np.ndarray,
    far_end: np.ndarray,
    noise0: np.ndarray,
    noise1: np.ndarray,
    design: Mapping[str, Any],
    rate: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    delays = design.get("near_arrival_delays_samples")
    if (
        not isinstance(delays, Sequence)
        or isinstance(delays, (str, bytes))
        or len(delays) != 2
        or any(int(value) != value or int(value) < 0 for value in delays)
    ):
        raise ValueError("design requires two non-negative near arrival delays")
    echo0, echo1, _ = _echo_components(far_end, design, rate)
    delayed_near1 = _delayed(near, int(delays[1]))
    microphone = np.column_stack((
        near + echo0 + noise0,
        delayed_near1 + echo1 + noise1,
    ))
    return microphone, echo0, echo1


def add_observed_ser(
    signals: Mapping[str, np.ndarray],
    design: Mapping[str, Any],
    rate: int,
) -> dict[str, Any]:
    """Measure near/echo SER on the actual common double-talk overlap.

    The echo component is recovered from
    ``microphone[:, 0] - reference - noise``.  This function adds only
    explicitly named overlap fields and leaves every legacy ``ser_db_*`` field
    unchanged.
    """
    if not isinstance(signals, Mapping):
        raise ValueError("signals must be a mapping")
    if not isinstance(design, Mapping):
        raise ValueError("design must be a mapping")
    canonical_rate = _positive_rate(rate)
    if "reference" in signals:
        near_key = "reference"
    elif "near_end" in signals:
        near_key = "near_end"
    else:
        raise ValueError("signals require reference or near_end")
    missing = sorted({"microphone", near_key, "noise"} - set(signals))
    if missing:
        raise ValueError("signals lack required roles: " + ", ".join(missing))
    near = _array_1d(near_key, signals[near_key])
    microphone = np.asarray(signals["microphone"], dtype=np.float64)
    noise_raw = np.asarray(signals["noise"], dtype=np.float64)
    if microphone.ndim != 2 or microphone.shape[1] != 2:
        raise ValueError("observed SER requires stereo microphone")
    if microphone.shape[0] != len(near):
        raise ValueError("microphone and reference lengths differ")
    if noise_raw.ndim == 1:
        noise0 = _array_1d("noise", noise_raw)
    elif noise_raw.ndim == 2 and noise_raw.shape == (len(near), 2):
        noise0 = _array_1d("noise channel 0", noise_raw[:, 0])
    else:
        raise ValueError("noise must be (frames,) or (frames, 2)")
    if len(noise0) != len(near):
        raise ValueError("noise and reference lengths differ")
    if not np.isfinite(microphone).all():
        raise ValueError("microphone contains non-finite samples")
    echo0 = microphone[:, 0] - near - noise0
    overlap = _double_talk_mask(design, len(near), canonical_rate)
    result = deepcopy(dict(design))
    if overlap is None:
        result["near_to_echo_overlap_ser_db_measured"] = None
        result["echo_to_near_overlap_esr_db_measured"] = None
        result["ser_observation_window_s"] = None
        result["ser_observation_window_definition"] = (
            "No activity_segments/activity_masks double_talk overlap exists."
        )
        return result

    near_rms = _rms(near, overlap)
    echo_rms = _rms(echo0, overlap)
    near_to_echo = _db(near_rms, echo_rms)
    result["near_to_echo_overlap_ser_db_measured"] = near_to_echo
    result["echo_to_near_overlap_esr_db_measured"] = (
        None if near_to_echo is None else -near_to_echo
    )
    result["ser_observation_window_s"] = round(
        float(np.count_nonzero(overlap) / canonical_rate),
        9,
    )
    result["ser_observation_window_definition"] = (
        "Same-mask RMS over activity label double_talk; "
        "echo0 = microphone[:,0] - reference - noise_channel_0."
    )
    return result


def calibrate_short_overlap(
    signals: Mapping[str, np.ndarray],
    design: Mapping[str, Any],
    rate: int,
    target_snr_db: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Calibrate both noise channels over the real short-overlap window.

    The near/reference and far-end truth signals are never changed.  Channel 0
    and the reconstructible channel 1 receive one common noise scale; the
    microphone is then rebuilt from the unchanged truth, echo recipe, and the
    scaled noise components.
    """
    canonical_rate = _positive_rate(rate)
    if (
        isinstance(target_snr_db, (bool, np.bool_))
        or not math.isfinite(float(target_snr_db))
    ):
        raise ValueError("target_snr_db must be finite")
    target_snr = float(target_snr_db)
    design_rate = design.get("sample_rate_hz")
    if design_rate is not None and int(design_rate) != canonical_rate:
        raise ValueError("design sample_rate_hz differs from calibration rate")

    if not isinstance(signals, Mapping):
        raise ValueError("signals must be a mapping")
    if "reference" in signals:
        near_key = "reference"
    elif "near_end" in signals:
        near_key = "near_end"
    else:
        raise ValueError("signals require reference or near_end")
    required = {"microphone", near_key, "far_end", "noise"}
    missing = sorted(required - set(signals))
    if missing:
        raise ValueError("signals lack required roles: " + ", ".join(missing))

    near = _array_1d(near_key, signals[near_key])
    far_end = _array_1d("far_end", signals["far_end"])
    microphone = np.asarray(signals["microphone"], dtype=np.float64)
    noise_raw = np.asarray(signals["noise"], dtype=np.float64)
    if microphone.ndim != 2 or microphone.shape[1] != 2:
        raise ValueError("short_overlap microphone must be stereo")
    frames = len(near)
    if len(far_end) != frames or microphone.shape[0] != frames:
        raise ValueError("short_overlap signal lengths differ")
    if not np.isfinite(microphone).all():
        raise ValueError("short_overlap microphone contains non-finite samples")
    if noise_raw.ndim == 1:
        if len(noise_raw) != frames:
            raise ValueError("short_overlap noise length differs")
        noise0 = _array_1d("noise", noise_raw)
        noise1, recipe = _noise_channel_from_recipe(noise0, design)
        noise_two_channel = False
    elif noise_raw.ndim == 2 and noise_raw.shape == (frames, 2):
        noise0 = _array_1d("noise channel 0", noise_raw[:, 0])
        noise1 = _array_1d("noise channel 1", noise_raw[:, 1])
        _, recipe = _noise_channel_from_recipe(noise0, design)
        noise_two_channel = True
    else:
        raise ValueError("short_overlap noise must be (frames,) or (frames, 2)")
    if not np.isfinite(noise0).all() or not np.isfinite(noise1).all():
        raise ValueError("short_overlap noise contains non-finite samples")

    overlap = _short_overlap_mask(design, frames, canonical_rate)
    near_rms = _rms(near, overlap)
    current_noise_rms = _rms(noise0, overlap)
    if near_rms <= ZERO_EPSILON:
        raise ValueError("short_overlap near reference is silent in the overlap window")
    if current_noise_rms <= ZERO_EPSILON:
        raise ValueError("short_overlap noise is silent in the overlap window")

    current_microphone, _, _ = _rebuild_microphone(
        near,
        far_end,
        noise0,
        noise1,
        design,
        canonical_rate,
    )
    current_residual = float(np.max(np.abs(microphone - current_microphone)))
    if current_residual > 1e-8:
        raise ValueError(
            "signals and design cannot reconstruct the current microphone "
            f"(max residual {current_residual:.12g})"
        )

    target_noise_rms = near_rms / (10.0 ** (target_snr / 20.0))
    scale_factor = target_noise_rms / current_noise_rms
    if not math.isfinite(scale_factor) or scale_factor <= 0.0:
        raise ValueError("short_overlap noise scale factor is invalid")
    scaled_noise0 = noise0 * scale_factor
    scaled_noise1 = noise1 * scale_factor
    calibrated_microphone, echo0, echo1 = _rebuild_microphone(
        near,
        far_end,
        scaled_noise0,
        scaled_noise1,
        design,
        canonical_rate,
    )
    peak = float(np.max(np.abs(calibrated_microphone)))
    if peak > PEAK_GUARD:
        raise ValueError(
            "short_overlap calibration would exceed the peak guard; "
            "choose a feasible target_snr_db"
        )

    measured_snr0 = _db(_rms(near, overlap), _rms(scaled_noise0, overlap))
    measured_snr1 = _db(_rms(near, overlap), _rms(scaled_noise1, overlap))
    if measured_snr0 is None or abs(measured_snr0 - target_snr) > 1e-9:
        raise ValueError("short_overlap channel-0 SNR calibration did not converge")
    residual = max(
        float(np.max(np.abs(
            calibrated_microphone[:, 0] - near - echo0 - scaled_noise0
        ))),
        float(np.max(np.abs(
            calibrated_microphone[:, 1]
            - _delayed(near, int(design["near_arrival_delays_samples"][1]))
            - echo1
            - scaled_noise1
        ))),
    )
    if residual > 1e-12:
        raise ValueError("short_overlap microphone component conservation failed")

    calibrated_design = deepcopy(dict(design))
    spatial = calibrated_design.get("noise_spatial")
    if not isinstance(spatial, Mapping):
        raise ValueError("design lacks noise_spatial metadata")
    calibrated_spatial = deepcopy(dict(spatial))
    calibrated_recipe = deepcopy(recipe)
    calibrated_recipe["noise_channel_scale_linear"] = (
        float(calibrated_recipe["noise_channel_scale_linear"]) * scale_factor
    )
    calibrated_spatial["channel_1_recipe"] = calibrated_recipe
    calibration_info: dict[str, Any] = {
        "applied": True,
        "category": "short_overlap",
        "observation_label": "double_talk",
        "observation_window_samples": int(np.count_nonzero(overlap)),
        "observation_window_s": round(
            float(np.count_nonzero(overlap) / canonical_rate),
            9,
        ),
        "target_snr_db": target_snr,
        "measured_snr_db_channel_0": measured_snr0,
        "measured_snr_db_channel_1": measured_snr1,
        "noise_scale_factor_linear": scale_factor,
        "near_reference_rms_before": near_rms,
        "noise_rms_before_channel_0": current_noise_rms,
        "noise_rms_after_channel_0": _rms(scaled_noise0, overlap),
        "uniform_scale_both_noise_channels": True,
        "reference_or_far_end_modified": False,
        "noise_two_channel_input": noise_two_channel,
    }
    calibrated_spatial["short_overlap_calibration"] = calibration_info
    calibrated_design["noise_spatial"] = calibrated_spatial
    calibrated_design["short_overlap_calibration"] = calibration_info
    calibrated_design["short_overlap_snr_observation_window_s"] = (
        calibration_info["observation_window_s"]
    )
    calibrated_design["noise_snr_db_target"] = target_snr
    calibrated_design["noise_snr_db_measured"] = measured_snr0
    calibrated_design["noise_snr_db_measured_channel_1"] = measured_snr1
    calibrated_design["snr_db_target_applicable"] = True
    calibrated_design["snr_db_target"] = target_snr
    calibrated_design["snr_db_measured"] = measured_snr0
    calibrated_design["snr_db_definition"] = (
        "20*log10(RMS(reference[double_talk observation window])"
        " / RMS(noise_channel_0[double_talk observation window]))"
    )
    calibrated_design["source_mix_check"] = {
        **(
            deepcopy(dict(calibrated_design.get("source_mix_check", {})))
            if isinstance(calibrated_design.get("source_mix_check"), Mapping)
            else {}
        ),
        "applicable": True,
        "residual_max_abs": residual,
        "formula": (
            "microphone[:,ch] = delayed_reference + echo_component_ch"
            " + reconstructed_noise_ch"
        ),
        "observation_window_s": calibration_info["observation_window_s"],
    }
    calibrated_design["mixture"] = {
        "category": "short_overlap",
        "observation_window_s": calibration_info["observation_window_s"],
        "noise_scale_factor_linear": scale_factor,
        "source_mix_residual_max_abs": residual,
        "channel_1_recipe_rebuild_consistent": True,
    }
    calibrated_design["peak"] = peak
    calibrated_design["clipped"] = False
    calibrated_design["hard_clipped"] = False

    output_signals: dict[str, np.ndarray] = {
        key: np.asarray(value, dtype=np.float64).copy()
        for key, value in signals.items()
    }
    output_signals["microphone"] = calibrated_microphone
    output_signals["noise"] = (
        np.column_stack((scaled_noise0, scaled_noise1))
        if noise_two_channel
        else scaled_noise0
    )
    output_signals[near_key] = near.copy()
    output_signals["far_end"] = far_end.copy()
    return output_signals, calibrated_design


__all__ = [
    "add_ser_aliases",
    "add_observed_ser",
    "apply_transport",
    "calibrate_short_overlap",
    "highway_noise",
]
