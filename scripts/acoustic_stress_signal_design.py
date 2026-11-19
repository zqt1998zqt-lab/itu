#!/usr/bin/env python3
"""Deterministic echo and double-talk stress-signal construction."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy import signal


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
sys.dont_write_bytecode = True


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SIGNAL_DESIGN = _load_module(
    SCRIPTS / "acceptance_signal_design.py",
    "itu_acceptance_signal_design_for_stress",
)
CONVERSATION = _load_module(
    SCRIPTS / "create-background-noise-conversation-dataset.py",
    "itu_background_conversation_for_stress",
)
MEASUREMENT = CONVERSATION.MEASUREMENT


SCOPE_CATEGORIES = {
    "echo": (
        "steady_echo",
        "convergence",
        "delay_change",
        "path_change",
        "noisy_echo",
    ),
    "double_talk": (
        "balanced",
        "near_dominant",
        "far_dominant",
        "transition",
        "short_overlap",
    ),
}
DOUBLE_TALK_SER_DB = {
    "balanced": 0.0,
    "near_dominant": -10.0,
    "far_dominant": 10.0,
    "transition": 0.0,
    "short_overlap": 0.0,
}
ECHO_GAIN_LEVELS = (0.22, 0.30, 0.40, 0.55, 0.72)
ECHO_NOISE_SNR_DB = (12.0, 15.0, 18.0, 21.0, 24.0)
NOISY_ECHO_SNR_DB = (0.0, 3.0, 6.0, 9.0, 12.0)
DOUBLE_TALK_SNR_DB = (10.0, 13.0, 16.0, 19.0, 22.0)
PRE_PCM_MIX_RESIDUAL_LIMIT = 1e-12
PEAK_TARGET = SIGNAL_DESIGN.PEAK_TARGET


def _array_1d(name: str, value: np.ndarray) -> np.ndarray:
    samples = np.asarray(value, dtype=np.float64)
    if samples.ndim != 1 or not len(samples):
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(samples).all():
        raise ValueError(f"{name} must contain only finite samples")
    return samples.copy()


def _rms(values: np.ndarray, mask: np.ndarray | None = None) -> float:
    samples = np.asarray(values, dtype=np.float64)
    if mask is not None:
        samples = samples[np.asarray(mask, dtype=bool)]
    if not samples.size:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def _db(numerator: float, denominator: float) -> float | None:
    if numerator <= 0.0 or denominator <= 0.0:
        return None
    return float(20.0 * math.log10(numerator / denominator))


def _window(frames: int, rate: int, start_s: float, end_s: float) -> np.ndarray:
    mask = np.zeros(frames, dtype=bool)
    start = max(0, min(frames, int(round(start_s * rate))))
    end = max(0, min(frames, int(round(end_s * rate))))
    mask[start:end] = True
    return mask


def _mask_from_segments(
    segments: list[dict[str, Any]],
    frames: int,
    rate: int,
    labels: frozenset[str],
) -> np.ndarray:
    mask = np.zeros(frames, dtype=bool)
    for segment in segments:
        if segment["label"] in labels:
            mask |= _window(
                frames,
                rate,
                float(segment["start_s"]),
                float(segment["end_s"]),
            )
    return mask


def _append_segment(
    result: list[dict[str, Any]],
    start_s: float,
    end_s: float,
    label: str,
    duration_s: float,
) -> None:
    start = max(0.0, min(duration_s, float(start_s)))
    end = max(0.0, min(duration_s, float(end_s)))
    if end > start:
        result.append({
            "start_s": round(start, 9),
            "end_s": round(end, 9),
            "label": label,
        })


def activity_segments(
    scope: str,
    category: str,
    duration_s: float,
) -> list[dict[str, Any]]:
    """Return the exact non-overlapping activity timeline for one workload."""
    if scope not in SCOPE_CATEGORIES or category not in SCOPE_CATEGORIES[scope]:
        raise ValueError("unknown acoustic stress scope/category")
    result: list[dict[str, Any]] = []
    if scope == "echo":
        if category == "convergence":
            _append_segment(result, 0.0, 0.5, "quiet", duration_s)
            _append_segment(result, 0.5, duration_s, "far_only", duration_s)
        else:
            _append_segment(result, 0.0, duration_s, "far_only", duration_s)
        return result

    if category in {"balanced", "near_dominant", "far_dominant"}:
        _append_segment(result, 0.0, duration_s, "double_talk", duration_s)
    elif category == "transition":
        _append_segment(result, 0.0, 0.5, "quiet", duration_s)
        _append_segment(result, 0.5, 2.0, "far_only", duration_s)
        _append_segment(result, 2.0, 5.0, "near_only", duration_s)
        _append_segment(result, 5.0, 8.0, "far_only", duration_s)
        _append_segment(result, 8.0, 11.0, "double_talk", duration_s)
        _append_segment(result, 11.0, duration_s, "noise_only", duration_s)
    elif category == "short_overlap":
        _append_segment(result, 0.0, 0.5, "quiet", duration_s)
        _append_segment(result, 0.5, 6.5, "near_only", duration_s)
        _append_segment(result, 6.5, 7.0, "double_talk", duration_s)
        _append_segment(result, 7.0, duration_s, "far_only", duration_s)
    else:
        raise ValueError(f"unsupported double-talk category: {category}")
    return result


def conversation_mode_for(scope: str, category: str) -> str:
    if scope == "echo":
        return "far_single"
    if scope == "double_talk":
        return "transition" if category == "transition" else "double_talk"
    raise ValueError("unknown acoustic stress scope")


def analysis_start_for(scope: str, category: str) -> float:
    if category in {"convergence", "transition", "short_overlap"}:
        return 0.5
    return 0.0


def _band_limits(bandwidth: str) -> tuple[float, float]:
    try:
        return SIGNAL_DESIGN.BAND_LIMITS_HZ[bandwidth]
    except KeyError as error:
        raise ValueError(f"unsupported bandwidth: {bandwidth}") from error


def band_limit(
    samples: np.ndarray,
    rate: int,
    bandwidth: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Band-limit one stress source according to its NB/WB/SWB profile."""
    values = _array_1d("band-limit source", samples)
    low_hz, requested_high_hz = _band_limits(bandwidth)
    nyquist_hz = rate / 2.0
    high_hz = min(requested_high_hz, 0.99 * nyquist_hz)
    if low_hz >= high_hz:
        raise ValueError("band-limit edges do not fit the sample rate")
    sos = signal.butter(
        4,
        (low_hz, high_hz),
        btype="bandpass",
        fs=rate,
        output="sos",
    )
    filtered = signal.sosfiltfilt(sos, values)
    return np.asarray(filtered, dtype=np.float64), {
        "bandwidth": bandwidth,
        "requested_edges_hz": [low_hz, requested_high_hz],
        "applied_edges_hz": [low_hz, high_hz],
        "sample_rate_hz": rate,
        "order": 4,
        "filter": "butterworth_bandpass_sosfiltfilt",
    }


def _path(
    rate: int,
    *,
    channel: int,
    delay_ms: float,
    variant: str,
) -> dict[str, Any]:
    offsets_by_variant = {
        "steady": {
            0: (0.0, 11.5, 27.0, 53.0),
            1: (0.0, 15.5, 34.0, 61.0),
        },
        "early": {
            0: (0.0, 7.0, 19.0, 41.0),
            1: (0.0, 9.0, 23.0, 47.0),
        },
        "late": {
            0: (0.0, 17.0, 39.0, 71.0),
            1: (0.0, 21.0, 46.0, 83.0),
        },
        "alternate": {
            0: (0.0, 5.0, 39.0, 67.0),
            1: (0.0, 8.0, 43.0, 79.0),
        },
    }
    gains_by_variant = {
        "steady": {
            0: (1.0, 0.45, 0.24, 0.11),
            1: (0.96, 0.36, 0.29, 0.15),
        },
        "early": {
            0: (0.92, 0.38, 0.20, 0.10),
            1: (0.88, 0.42, 0.18, 0.12),
        },
        "late": {
            0: (0.82, 0.51, 0.31, 0.17),
            1: (0.79, 0.47, 0.35, 0.21),
        },
        "alternate": {
            0: (0.78, 0.22, 0.61, 0.33),
            1: (0.71, 0.55, 0.18, 0.46),
        },
    }
    if variant not in offsets_by_variant:
        raise ValueError(f"unknown echo path variant: {variant}")
    if channel not in {0, 1}:
        raise ValueError("echo channel must be zero or one")
    delay_samples = int(round(float(delay_ms) * rate / 1000.0))
    tap_samples = [
        delay_samples + int(round(offset_ms * rate / 1000.0))
        for offset_ms in offsets_by_variant[variant][channel]
    ]
    raw_gains = list(gains_by_variant[variant][channel])
    normalizer = max(sum(abs(gain) for gain in raw_gains), 1e-15)
    tap_gains = [float(gain / normalizer) for gain in raw_gains]
    return {
        "channel": channel,
        "variant": variant,
        "delay_ms": float(delay_ms),
        "tap_samples": tap_samples,
        "tap_offsets_ms": [
            round(float(delay_ms) + offset_ms, 9)
            for offset_ms in offsets_by_variant[variant][channel]
        ],
        "tap_gains": tap_gains,
        "echo_gain_linear": 1.0,
        "tap_l1": float(sum(abs(gain) for gain in tap_gains)),
    }


def _echo_timing(
    scope: str,
    category: str,
    duration_s: float,
) -> list[dict[str, Any]]:
    if scope == "echo":
        start_s = 0.5 if category == "convergence" else 0.0
        if category == "delay_change":
            return [
                {"start_s": 0.0, "end_s": 6.0, "delays_ms": (20.0, 25.0), "variant": "steady"},
                {"start_s": 6.0, "end_s": duration_s, "delays_ms": (80.0, 90.0), "variant": "late"},
            ]
        if category == "path_change":
            return [
                {"start_s": 0.0, "end_s": 6.0, "delays_ms": (20.0, 25.0), "variant": "steady"},
                {"start_s": 6.0, "end_s": duration_s, "delays_ms": (20.0, 25.0), "variant": "alternate"},
            ]
        return [
            {"start_s": start_s, "end_s": duration_s, "delays_ms": (20.0, 25.0), "variant": "steady"}
        ]
    return [
        {"start_s": 0.0, "end_s": duration_s, "delays_ms": (20.0, 25.0), "variant": "steady"}
    ]


def _dynamic_echo(
    far_end: np.ndarray,
    rate: int,
    *,
    scope: str,
    category: str,
    duration_s: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    echo_0 = np.zeros(len(far_end), dtype=np.float64)
    echo_1 = np.zeros(len(far_end), dtype=np.float64)
    records: list[dict[str, Any]] = []
    for timing in _echo_timing(scope, category, duration_s):
        mask = _window(
            len(far_end),
            rate,
            float(timing["start_s"]),
            float(timing["end_s"]),
        )
        path_0 = _path(
            rate,
            channel=0,
            delay_ms=float(timing["delays_ms"][0]),
            variant=str(timing["variant"]),
        )
        path_1 = _path(
            rate,
            channel=1,
            delay_ms=float(timing["delays_ms"][1]),
            variant=str(timing["variant"]),
        )
        segment_0 = SIGNAL_DESIGN._apply_echo_taps(
            far_end,
            path_0["tap_samples"],
            path_0["tap_gains"],
        )
        segment_1 = SIGNAL_DESIGN._apply_echo_taps(
            far_end,
            path_1["tap_samples"],
            path_1["tap_gains"],
        )
        echo_0 += segment_0 * mask
        echo_1 += segment_1 * mask
        records.append({
            "start_s": float(timing["start_s"]),
            "end_s": float(timing["end_s"]),
            "paths": [path_0, path_1],
        })
    return echo_0, echo_1, records


def _noise_snr_for(
    scope: str,
    category: str,
    case_index: int,
    explicit: float | None,
) -> float:
    if explicit is not None:
        value = float(explicit)
    elif scope == "echo":
        levels = NOISY_ECHO_SNR_DB if category == "noisy_echo" else ECHO_NOISE_SNR_DB
        value = levels[(case_index - 1) % len(levels)]
    else:
        value = DOUBLE_TALK_SNR_DB[(case_index - 1) % len(DOUBLE_TALK_SNR_DB)]
    if not math.isfinite(value):
        raise ValueError("noise SNR must be finite")
    return value


def render_stress_signals(
    near: np.ndarray,
    far: np.ndarray,
    noise_unit: np.ndarray,
    rate: int,
    *,
    scope: str,
    category: str,
    volume_gain_db: float,
    seed: int,
    case_index: int,
    bandwidth: str,
    noise_snr_db: float | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Render one stress case with explicit activity and dynamic FIR segments."""
    if scope not in SCOPE_CATEGORIES or category not in SCOPE_CATEGORIES[scope]:
        raise ValueError("unknown acoustic stress scope/category")
    if isinstance(case_index, bool) or not isinstance(case_index, int) or case_index < 1:
        raise ValueError("case_index must be a positive integer")
    near_input = _array_1d("near", near)
    far_input = _array_1d("far", far)
    noise_input = _array_1d("noise_unit", noise_unit)
    if len(near_input) != len(far_input) or len(near_input) != len(noise_input):
        raise ValueError("near, far, and noise_unit lengths differ")
    if not math.isfinite(float(volume_gain_db)):
        raise ValueError("volume_gain_db must be finite")
    near_limited, near_filter = band_limit(near_input, rate, bandwidth)
    far_limited, far_filter = band_limit(far_input, rate, bandwidth)
    noise_limited, noise_filter = band_limit(noise_input, rate, bandwidth)
    duration_s = len(near_input) / rate
    segments = activity_segments(scope, category, duration_s)
    near_mask = _mask_from_segments(
        segments,
        len(near_input),
        rate,
        frozenset({"near_only", "double_talk"}),
    )
    far_mask = _mask_from_segments(
        segments,
        len(near_input),
        rate,
        frozenset({"far_only", "double_talk"}),
    )

    base_kind = "far_single" if scope == "echo" else (
        "duplex" if category == "transition" else "double_talk"
    )
    requested_snr = _noise_snr_for(scope, category, case_index, noise_snr_db)
    base_signals, base_metadata = SIGNAL_DESIGN.render_signals(
        near_limited,
        far_limited,
        noise_limited,
        rate,
        direction="send",
        kind=base_kind,
        snr_db=requested_snr,
        ser_db=DOUBLE_TALK_SER_DB.get(category, 0.0),
        echo_delay_ms=20.0,
        seed=seed,
        input_level_db=0.0,
        volume_gain_db=float(volume_gain_db),
        bandwidth=bandwidth,
    )
    reference = np.asarray(base_signals["reference"], dtype=np.float64) * near_mask
    far_end = np.asarray(base_signals["far_end"], dtype=np.float64) * far_mask
    noise_0 = np.asarray(base_signals["noise"], dtype=np.float64).copy()
    noise_1 = MEASUREMENT._reconstruct_noise_channel(
        noise_0,
        base_metadata,
        1,
    )

    target_mask = near_mask if np.any(near_mask) else far_mask
    target_signal = reference if np.any(near_mask) else far_end
    target_rms = _rms(target_signal, target_mask)
    current_noise_rms = _rms(noise_0, target_mask)
    if target_rms <= 1e-12 or current_noise_rms <= 1e-15:
        raise ValueError("stress noise SNR has no observable target/noise")
    target_noise_rms = target_rms / (10.0 ** (requested_snr / 20.0))
    noise_scale_factor = target_noise_rms / current_noise_rms
    noise_0 *= noise_scale_factor
    noise_1 *= noise_scale_factor

    raw_echo_0, raw_echo_1, echo_segments = _dynamic_echo(
        far_end,
        rate,
        scope=scope,
        category=category,
        duration_s=duration_s,
    )
    if scope == "double_talk":
        near_rms = _rms(reference, near_mask)
        raw_echo_rms = _rms(raw_echo_0, near_mask)
        if near_rms <= 1e-12 or raw_echo_rms <= 1e-15:
            raise ValueError("double-talk SER has no observable near/echo")
        ser_nominal = DOUBLE_TALK_SER_DB[category]
        ser_effective = ser_nominal + float(volume_gain_db)
        echo_target_rms = near_rms * (10.0 ** (ser_effective / 20.0))
        echo_gain = echo_target_rms / raw_echo_rms
    else:
        ser_nominal = None
        ser_effective = None
        echo_gain = ECHO_GAIN_LEVELS[(case_index - 1) % len(ECHO_GAIN_LEVELS)]
    echo_0 = raw_echo_0 * echo_gain
    echo_1 = raw_echo_1 * echo_gain

    delays = (0, 1)
    delayed_reference_0 = reference
    delayed_reference_1 = np.concatenate((np.zeros(1), reference[:-1]))
    microphone = np.column_stack((
        delayed_reference_0 + echo_0 + noise_0,
        delayed_reference_1 + echo_1 + noise_1,
    ))
    current_peak = max(
        float(np.max(np.abs(values)))
        for values in (reference, far_end, noise_0, noise_1, microphone)
    )
    common_gain = 1.0
    if current_peak > PEAK_TARGET:
        common_gain = PEAK_TARGET / current_peak
        reference *= common_gain
        far_end *= common_gain
        noise_0 *= common_gain
        noise_1 *= common_gain
        echo_0 *= common_gain
        echo_1 *= common_gain
        microphone *= common_gain

    recipe = dict(base_metadata["noise_spatial"]["channel_1_recipe"])
    recipe["noise_channel_scale_linear"] = (
        float(recipe["noise_channel_scale_linear"])
        * noise_scale_factor
        * common_gain
    )
    measured_snr = _db(
        _rms(reference if np.any(near_mask) else far_end, target_mask),
        _rms(noise_0, target_mask),
    )
    measured_ser = (
        _db(_rms(echo_0, near_mask), _rms(reference, near_mask))
        if scope == "double_talk"
        else None
    )
    reconstructed_0 = np.zeros(len(reference), dtype=np.float64)
    reconstructed_1 = np.zeros(len(reference), dtype=np.float64)
    for segment in echo_segments:
        mask = _window(
            len(reference),
            rate,
            segment["start_s"],
            segment["end_s"],
        )
        reconstructed_0 += MEASUREMENT._fir_component(
            far_end,
            segment["paths"][0],
        ) * mask * echo_gain
        reconstructed_1 += MEASUREMENT._fir_component(
            far_end,
            segment["paths"][1],
        ) * mask * echo_gain
    scaled_delayed_reference_0 = reference
    scaled_delayed_reference_1 = np.concatenate((np.zeros(1), reference[:-1]))
    source_mix_residual = max(
        float(np.max(np.abs(
            microphone[:, 0]
            - scaled_delayed_reference_0
            - reconstructed_0
            - noise_0
        ))),
        float(np.max(np.abs(
            microphone[:, 1]
            - scaled_delayed_reference_1
            - reconstructed_1
            - noise_1
        ))),
    )

    signals = {
        "microphone": microphone,
        "reference": reference,
        "far_end": far_end,
        "noise": noise_0,
    }
    metadata: dict[str, Any] = {
        "kind": "speech",
        "stimulus_kind": category,
        "workload_scope": scope,
        "workload_category": category,
        "direction": "send",
        "sample_rate_hz": int(rate),
        "duration_s": round(duration_s, 9),
        "bandwidth": bandwidth,
        "segments": segments,
        "activity_segments": segments,
        "analysis_start_s": analysis_start_for(scope, category),
        "beam_delays_ms": [1000.0 / rate, 0.0],
        "near_arrival_delays_samples": list(delays),
        "echo_path_segments": echo_segments,
        "echo_paths": [],
        "echo_gain_linear": float(echo_gain),
        "noise_spatial": {
            "returned_noise": "channel-0 noise; channel 1 is reconstructible",
            "channel_1_recipe": recipe,
        },
        "noise_snr_db_target": float(requested_snr),
        "noise_snr_db_measured": measured_snr,
        "ser_db_nominal_target": ser_nominal,
        "ser_db_effective_target": ser_effective,
        "ser_db_measured": measured_ser,
        "source_mix_check": {
            "applicable": True,
            "residual_max_abs": source_mix_residual,
            "formula": "microphone[:,ch] = delayed_reference + dynamic_echo + noise_ch",
        },
        "band_limit": {
            "near": near_filter,
            "far": far_filter,
            "noise": noise_filter,
        },
        "global_safety_gain": float(common_gain),
        "peak": float(np.max(np.abs(microphone))),
        "clipped": False,
        "engineering_preview": True,
        "standard_compliance": False,
        "evidence_scope": "synthetic acoustic stress engineering stimulus",
    }
    return signals, metadata


def _echo_component(
    delivered_far_end: np.ndarray,
    design: dict[str, Any],
    channel: int,
) -> np.ndarray:
    component = np.zeros(len(delivered_far_end), dtype=np.float64)
    rate = int(design["sample_rate_hz"])
    for segment in design["echo_path_segments"]:
        mask = _window(
            len(delivered_far_end),
            rate,
            float(segment["start_s"]),
            float(segment["end_s"]),
        )
        component += MEASUREMENT._fir_component(
            delivered_far_end,
            segment["paths"][channel],
        ) * mask * float(design["echo_gain_linear"])
    return component


def mixture_residuals(
    delivered: dict[str, np.ndarray],
    design: dict[str, Any],
) -> list[float]:
    microphone = np.asarray(delivered["microphone"], dtype=np.float64)
    reference = np.asarray(delivered["reference"], dtype=np.float64)
    far_end = np.asarray(delivered["far_end"], dtype=np.float64)
    noise = np.asarray(delivered["noise"], dtype=np.float64)
    if microphone.ndim != 2 or microphone.shape[1] != 2:
        raise ValueError("stress mixture requires stereo microphone")
    frames = len(microphone)
    if (
        reference.shape != (frames,)
        or far_end.shape != (frames,)
        or noise.shape != (frames,)
    ):
        raise ValueError("stress mixture component lengths differ")
    residuals: list[float] = []
    for channel, delay in enumerate(design["near_arrival_delays_samples"]):
        delayed = np.zeros(frames, dtype=np.float64)
        delayed[int(delay):] = reference[:frames - int(delay)]
        expected = (
            delayed
            + _echo_component(far_end, design, channel)
            + MEASUREMENT._reconstruct_noise_channel(noise, design, channel)
        )
        residuals.append(float(np.max(np.abs(microphone[:, channel] - expected))))
    return residuals


def quantized_mixture_bounds(design: dict[str, Any]) -> list[dict[str, Any]]:
    step = CONVERSATION.PCM16_MAX_QUANTIZATION_ERROR
    echo_gain = abs(float(design["echo_gain_linear"]))
    slack = 64.0 * np.finfo(np.float64).eps
    bounds: list[dict[str, Any]] = []
    for channel in range(2):
        echo_l1 = sum(
            echo_gain * float(segment["paths"][channel]["tap_l1"])
            for segment in design["echo_path_segments"]
        )
        noise_l1 = CONVERSATION._noise_mix_l1(design, channel)
        components = {
            "microphone": step,
            "reference": step,
            "noise": step * noise_l1,
            "far_end_echo": step * echo_l1,
        }
        bounds.append({
            "channel": channel,
            "pcm16_max_quantization_error": step,
            "noise_mix_l1": noise_l1,
            "echo_path_l1_with_gain": echo_l1,
            "quantization_components": components,
            "float_arithmetic_slack": slack,
            "theoretical_max_abs_residual": sum(components.values()) + slack,
        })
    return bounds


def verify_stress_mixture(
    pre_pcm: dict[str, np.ndarray],
    decoded: dict[str, np.ndarray],
    design: dict[str, Any],
) -> dict[str, Any]:
    source_residual = design.get("source_mix_check", {}).get("residual_max_abs")
    if (
        source_residual is None
        or not math.isfinite(float(source_residual))
        or float(source_residual) > PRE_PCM_MIX_RESIDUAL_LIMIT
    ):
        raise ValueError("stress pre-PCM source_mix_check exceeds 1e-12")
    pre_residuals = mixture_residuals(pre_pcm, design)
    if max(pre_residuals) > PRE_PCM_MIX_RESIDUAL_LIMIT:
        raise ValueError("stress pre-PCM two-channel reconstruction exceeds 1e-12")
    pcm_residuals = mixture_residuals(decoded, design)
    bounds = quantized_mixture_bounds(design)
    for channel, (residual, bound) in enumerate(
        zip(pcm_residuals, bounds, strict=True)
    ):
        theoretical = float(bound["theoretical_max_abs_residual"])
        if residual > theoretical:
            raise ValueError(
                "stress PCM mixture exceeds theoretical bound "
                f"(channel {channel}, residual {residual:.12g}, bound {theoretical:.12g})"
            )
    return {
        "source_mix_residual_max_abs": float(source_residual),
        "pre_pcm_residuals_max_abs": pre_residuals,
        "pcm_component_residuals_max_abs": pcm_residuals,
        "theoretical_bounds": bounds,
        "pre_pcm_limit": PRE_PCM_MIX_RESIDUAL_LIMIT,
    }


def assert_stress_activity(
    decoded: dict[str, np.ndarray],
    design: dict[str, Any],
    *,
    scope: str,
    category: str,
) -> dict[str, Any]:
    reference = np.asarray(decoded["reference"], dtype=np.float64)
    far_end = np.asarray(decoded["far_end"], dtype=np.float64)
    noise = np.asarray(decoded["noise"], dtype=np.float64)
    rate = int(design["sample_rate_hz"])
    duration_s = len(reference) / rate
    near_mask = _mask_from_segments(
        design["activity_segments"],
        len(reference),
        rate,
        frozenset({"near_only", "double_talk"}),
    )
    far_mask = _mask_from_segments(
        design["activity_segments"],
        len(reference),
        rate,
        frozenset({"far_only", "double_talk"}),
    )
    quiet_mask = _mask_from_segments(
        design["activity_segments"],
        len(reference),
        rate,
        frozenset({"quiet"}),
    )
    noise_only_mask = _mask_from_segments(
        design["activity_segments"],
        len(reference),
        rate,
        frozenset({"noise_only"}),
    )
    if scope == "echo":
        if np.count_nonzero(reference) != 0:
            raise ValueError("echo workload reference must be digital silence")
        if _rms(far_end, far_mask) <= 1e-5:
            raise ValueError("echo workload has no active far-end speech")
        if category == "convergence" and (
            _rms(far_end, quiet_mask) != 0.0
            or np.count_nonzero(far_end[quiet_mask]) != 0
        ):
            raise ValueError("convergence is not quiet before 0.5s")
    else:
        if _rms(reference, near_mask) <= 1e-5:
            raise ValueError("double-talk workload has no near-end speech")
        if _rms(far_end, far_mask) <= 1e-5:
            raise ValueError("double-talk workload has no far-end speech")
        if category == "short_overlap":
            overlap = _mask_from_segments(
                design["activity_segments"],
                len(reference),
                rate,
                frozenset({"double_talk"}),
            )
            overlap_s = float(np.count_nonzero(overlap) / rate)
            if abs(overlap_s - 0.5) > 1.0 / rate:
                raise ValueError("short_overlap actual overlap is not 0.5s")
        if np.any(noise_only_mask) and (
            np.count_nonzero(reference[noise_only_mask]) != 0
            or np.count_nonzero(far_end[noise_only_mask]) != 0
        ):
            raise ValueError("noise_only activity contains speech")
    return {
        "near_active_seconds": float(np.count_nonzero(near_mask) / rate),
        "far_active_seconds": float(np.count_nonzero(far_mask) / rate),
        "quiet_seconds": float(np.count_nonzero(quiet_mask) / rate),
        "noise_only_seconds": float(np.count_nonzero(noise_only_mask) / rate),
        "duration_s": duration_s,
        "noise_rms": _rms(noise),
        "reference_rms": _rms(reference),
        "far_end_rms": _rms(far_end),
    }


__all__ = [
    "DOUBLE_TALK_SER_DB",
    "ECHO_GAIN_LEVELS",
    "SCOPE_CATEGORIES",
    "activity_segments",
    "analysis_start_for",
    "assert_stress_activity",
    "band_limit",
    "conversation_mode_for",
    "mixture_residuals",
    "quantized_mixture_bounds",
    "render_stress_signals",
    "verify_stress_mixture",
]
