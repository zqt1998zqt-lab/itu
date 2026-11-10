#!/usr/bin/env python3
"""Deterministic engineering signal construction for acceptance datasets.

This module only prepares synthetic engineering stimuli. It does not establish
ITU-T, customer, or product compliance.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


FREQUENCY_RESPONSE_CENTERS_HZ = (
    125.0,
    160.0,
    200.0,
    250.0,
    315.0,
    400.0,
    500.0,
    630.0,
    800.0,
    1_000.0,
    1_250.0,
    1_600.0,
    2_000.0,
    2_500.0,
    3_150.0,
    4_000.0,
    5_000.0,
    6_300.0,
    8_000.0,
    10_000.0,
    12_500.0,
)
BAND_LIMITS_HZ = {
    "NB": (125.0, 4_000.0),
    "WB": (125.0, 8_000.0),
    "SWB": (125.0, 12_500.0),
}
OOB_DIAGNOSTIC_HZ = {
    "NB": 4_200.0,
    "WB": 8_500.0,
    "SWB": 14_500.0,
}
KINDS = {"speech", "duplex", "response", "oob", "idle", "cng"}
DIRECTIONS = {
    "send": "send",
    "tx": "send",
    "round_trip": "round_trip",
    "roundtrip": "round_trip",
    "receive": "receive",
    "rx": "receive",
}
PEAK_TARGET = 0.66
SPEECH_FADE_MS = 5.0
ECHO_COUPLING_GAIN = 0.35
IDLE_FLOOR_RMS = 10.0 ** (-75.0 / 20.0)


def _array_1d(name: str, value: np.ndarray) -> np.ndarray:
    samples = np.asarray(value, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if not len(samples):
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(samples).all():
        raise ValueError(f"{name} must contain only finite samples")
    return samples.copy()


def _rms(samples: np.ndarray, mask: np.ndarray | None = None) -> float:
    values = np.asarray(samples, dtype=np.float64)
    if mask is not None:
        values = values[np.asarray(mask, dtype=bool)]
    if not values.size:
        return 0.0
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def _db(numerator: float, denominator: float) -> float | None:
    if numerator <= 0.0 or denominator <= 0.0:
        return None
    return float(20.0 * math.log10(numerator / denominator))


def _correlation(
    first: np.ndarray,
    second: np.ndarray,
    mask: np.ndarray | None = None,
) -> float | None:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if mask is not None:
        selected = np.asarray(mask, dtype=bool)
        left = left[selected]
        right = right[selected]
    if not left.size:
        return None
    left = left - float(np.mean(left))
    right = right - float(np.mean(right))
    denominator = math.sqrt(
        float(np.dot(left, left)) * float(np.dot(right, right))
    )
    if denominator <= 1e-20:
        return None
    return float(np.dot(left, right) / denominator)


def _window(
    frames: int,
    rate: int,
    start_s: float,
    end_s: float,
) -> np.ndarray:
    start = max(0, min(frames, int(round(start_s * rate))))
    end = max(0, min(frames, int(round(end_s * rate))))
    result = np.zeros(frames, dtype=bool)
    if end > start:
        result[start:end] = True
    return result


def _delay(samples: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        return samples.copy()
    delayed = np.zeros_like(samples)
    if count < len(samples):
        delayed[count:] = samples[:-count]
    return delayed


def _fade_edges(samples: np.ndarray, rate: int) -> np.ndarray:
    faded = samples.copy()
    requested = int(round(SPEECH_FADE_MS * rate / 1000.0))
    edge = min(requested, len(faded) // 2)
    if edge <= 0:
        return faded
    ramp = np.sin(np.linspace(0.0, math.pi / 2.0, edge, dtype=np.float64)) ** 2
    faded[:edge] *= ramp
    faded[-edge:] *= ramp[::-1]
    return faded


def _mask_segments(mask: np.ndarray, rate: int) -> list[dict[str, float]]:
    values = np.asarray(mask, dtype=bool)
    changes = np.diff(np.r_[False, values, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [
        {
            "start_s": round(float(start) / rate, 9),
            "end_s": round(float(end) / rate, 9),
        }
        for start, end in zip(starts, ends, strict=True)
    ]


def _normalize_over_mask(samples: np.ndarray, mask: np.ndarray) -> np.ndarray:
    normalized = np.asarray(samples, dtype=np.float64).copy()
    level = _rms(normalized, mask)
    if level <= 1e-12:
        raise ValueError("noise mix has zero RMS over its normalization mask")
    normalized /= level
    return normalized


def _unit_noise_variants(
    noise_unit: np.ndarray,
    seed: int,
    rate: int,
    normalization_mask: np.ndarray,
    *,
    normalization_name: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    base = noise_unit - float(np.mean(noise_unit))
    base_rms = _rms(base)
    if base_rms <= 1e-12:
        raise ValueError("noise_unit must have non-zero AC RMS")
    base = base / base_rms

    rng = np.random.default_rng(seed)
    independent_0 = rng.normal(size=len(base))
    independent_0 -= float(np.mean(independent_0))
    independent_0 /= max(_rms(independent_0), 1e-15)

    theta_0 = rng.uniform(0.18, 0.42)
    channel_0 = math.cos(theta_0) * base + math.sin(theta_0) * independent_0
    channel_0 = _normalize_over_mask(channel_0, normalization_mask)

    seed_2 = int(rng.integers(0, 2**63 - 1, dtype=np.int64))
    rho = float(rng.uniform(0.25, 0.55))
    independent_2 = np.random.default_rng(seed_2).normal(size=len(base))
    mixed_1 = (
        rho * channel_0
        + math.sqrt(1.0 - rho * rho) * independent_2
    )
    normalizer = _rms(mixed_1, normalization_mask)
    if normalizer <= 1e-12:
        raise ValueError("channel-1 noise mix has zero RMS over normalization mask")
    channel_1 = mixed_1 / normalizer
    recipe = {
        "rho": rho,
        "seed2": seed_2,
        "independent_distribution": "standard_normal",
        "independent_draw": (
            "numpy.random.default_rng(seed2).normal(size=len(noise0))"
        ),
        "normalization_scope": normalization_name,
        "normalization_mask_segments": _mask_segments(normalization_mask, rate),
        "normalizer": float(normalizer),
        "formula": (
            "unit_noise0 = delivered_noise / noise_channel_scale_linear; "
            "noise1 = ((rho*unit_noise0 + sqrt(1-rho^2)*independent2)"
            " / normalizer) * noise_channel_scale_linear"
        ),
        "long_arrays_in_sidecar": False,
    }
    return channel_0, channel_1, recipe


def _append_segment(
    segments: list[dict[str, Any]],
    start_s: float,
    end_s: float,
    label: str,
    duration_s: float,
) -> None:
    start = max(0.0, min(duration_s, float(start_s)))
    end = max(0.0, min(duration_s, float(end_s)))
    if end <= start:
        return
    segments.append({
        "start_s": round(start, 9),
        "end_s": round(end, 9),
        "label": label,
    })


def _scene(
    kind: str,
    direction: str,
    frames: int,
    rate: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    duration_s = frames / rate
    zeros = np.zeros(frames, dtype=bool)
    ones = np.ones(frames, dtype=bool)
    masks = {
        "near": zeros.copy(),
        "far": zeros.copy(),
        "near_only": zeros.copy(),
        "far_only": zeros.copy(),
        "double_talk": zeros.copy(),
        "noise_only": zeros.copy(),
        "quiet": zeros.copy(),
    }
    segments: list[dict[str, Any]] = []

    if kind == "speech":
        if direction == "receive":
            masks["far"] = ones.copy()
        else:
            masks["near"] = ones.copy()
        _append_segment(segments, 0.0, duration_s, "speech", duration_s)
    elif kind == "duplex":
        speech_end_s = min(11.0, max(0.0, duration_s - 1.0))
        masks["quiet"] = _window(frames, rate, 0.0, 0.5)
        masks["far_only"] = (
            _window(frames, rate, 0.5, 2.0)
            | _window(frames, rate, 5.0, 8.0)
        )
        masks["near_only"] = _window(frames, rate, 2.0, 5.0)
        masks["double_talk"] = _window(frames, rate, 8.0, speech_end_s)
        masks["noise_only"] = _window(frames, rate, speech_end_s, duration_s)
        masks["near"] = (
            _window(frames, rate, 2.0, 5.0)
            | _window(frames, rate, 8.0, speech_end_s)
        )
        masks["far"] = (
            _window(frames, rate, 0.5, 2.0)
            | _window(frames, rate, 5.0, speech_end_s)
        )
        _append_segment(segments, 0.0, 0.5, "quiet", duration_s)
        _append_segment(segments, 0.5, 2.0, "far_only", duration_s)
        _append_segment(segments, 2.0, 5.0, "near_only", duration_s)
        _append_segment(segments, 5.0, 8.0, "far_only", duration_s)
        _append_segment(segments, 8.0, speech_end_s, "double_talk", duration_s)
        _append_segment(segments, speech_end_s, duration_s, "noise_only", duration_s)
    elif kind in {"response", "oob"}:
        if direction == "receive":
            masks["far"] = ones.copy()
        else:
            masks["near"] = ones.copy()
        _append_segment(segments, 0.0, duration_s, kind, duration_s)
    elif kind == "idle":
        _append_segment(segments, 0.0, duration_s, "idle", duration_s)
    elif kind == "cng":
        active_end_s = min(2.0, duration_s)
        measurement_start_s = min(4.0, duration_s)
        if direction == "receive":
            masks["far"] = _window(frames, rate, 0.0, active_end_s)
        else:
            masks["near"] = _window(frames, rate, 0.0, active_end_s)
        masks["noise_only"] = _window(frames, rate, active_end_s, duration_s)
        _append_segment(segments, 0.0, active_end_s, "cng_active", duration_s)
        _append_segment(
            segments,
            active_end_s,
            measurement_start_s,
            "cng_settling",
            duration_s,
        )
        _append_segment(
            segments,
            measurement_start_s,
            duration_s,
            "cng_noise_observation",
            duration_s,
        )
    else:
        raise ValueError(f"unsupported kind: {kind!r}")

    return masks, segments


def _multisine(
    frames: int,
    rate: int,
    bandwidth: str,
    seed: int,
    *,
    with_oob_tone: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    low_hz, high_hz = BAND_LIMITS_HZ[bandwidth]
    nyquist_hz = rate / 2.0
    if high_hz >= nyquist_hz:
        raise ValueError(
            f"{bandwidth} multisine upper edge {high_hz:g} Hz is not observable "
            f"at {rate} Hz (Nyquist {nyquist_hz:g} Hz)"
        )

    diagnostic_hz = OOB_DIAGNOSTIC_HZ[bandwidth]
    if with_oob_tone and diagnostic_hz >= nyquist_hz:
        raise ValueError(
            f"{bandwidth} OOB diagnostic {diagnostic_hz:g} Hz is not observable "
            f"at {rate} Hz (Nyquist {nyquist_hz:g} Hz)"
        )

    frequencies = np.asarray(
        [
            frequency
            for frequency in FREQUENCY_RESPONSE_CENTERS_HZ
            if frequency <= high_hz
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, 2.0 * math.pi, len(frequencies))
    time = np.arange(frames, dtype=np.float64) / rate
    signal = np.zeros(frames, dtype=np.float64)
    for frequency, phase in zip(frequencies, phases, strict=True):
        signal += np.sin(2.0 * math.pi * frequency * time + phase)
    signal /= max(_rms(signal), 1e-15)

    if with_oob_tone:
        diagnostic_phase = float(rng.uniform(0.0, 2.0 * math.pi))
        signal += 0.5 * np.sin(
            2.0 * math.pi * diagnostic_hz * time + diagnostic_phase
        )

    return signal, {
        "multisine_frequencies_hz": [float(value) for value in frequencies],
        "band_edges_hz": [low_hz, high_hz],
        "diagnostic_tone_hz": diagnostic_hz if with_oob_tone else None,
    }


def _echo_path(
    frames: int,
    rate: int,
    source: np.ndarray,
    echo_delay_ms: float,
    channel: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    delay_samples = int(round(echo_delay_ms * rate / 1000.0))
    if delay_samples >= frames:
        raise ValueError("echo_delay_ms places the direct echo beyond the signal")

    reflection_offsets_ms = (
        (0.0, 11.5, 27.0, 53.0),
        (0.0, 15.5, 34.0, 61.0),
    )[channel]
    reflection_gains = (
        (1.0, 0.45, 0.24, 0.11),
        (0.96, 0.36, 0.29, 0.15),
    )[channel]
    tap_samples: list[int] = []
    raw_tap_gains: list[float] = []
    for offset_ms, gain in zip(
        reflection_offsets_ms,
        reflection_gains,
        strict=True,
    ):
        position = delay_samples + int(round(offset_ms * rate / 1000.0))
        if position < frames:
            tap_samples.append(position)
            raw_tap_gains.append(gain)
    normalization = max(sum(abs(gain) for gain in raw_tap_gains), 1e-15)
    tap_gains = [float(gain / normalization) for gain in raw_tap_gains]
    echo = _apply_echo_taps(source, tap_samples, tap_gains)
    return echo, {
        "channel": channel,
        "direct_delay_samples": delay_samples,
        "tap_offsets_ms": [
            round(offset_ms + echo_delay_ms, 9)
            for offset_ms in reflection_offsets_ms
        ],
        "tap_samples": tap_samples,
        "tap_gains": tap_gains,
    }


def _apply_echo_taps(
    source: np.ndarray,
    tap_samples: list[int],
    tap_gains: list[float],
) -> np.ndarray:
    echo = np.zeros(len(source), dtype=np.float64)
    for position, gain in zip(tap_samples, tap_gains, strict=True):
        if position >= len(source):
            continue
        echo[position:] += source[:len(source) - position] * gain
    return echo


def _peak(*arrays: np.ndarray) -> float:
    return max(
        (float(np.max(np.abs(np.asarray(values)))) for values in arrays),
        default=0.0,
    )


def render_signals(
    near: np.ndarray,
    far: np.ndarray,
    noise_unit: np.ndarray,
    rate: int,
    *,
    direction: str,
    kind: str,
    snr_db: float,
    ser_db: float = -5,
    echo_delay_ms: float = 30,
    seed: int = 1,
    input_level_db: float = 0,
    volume_gain_db: float = 0,
    bandwidth: str = "WB",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Render one deterministic four-signal engineering scene.

    ``near``, ``far``, and ``noise_unit`` must have the same length. The
    returned microphone is stereo for send/round_trip and mono for receive.
    The send channel-0 microphone is exactly ``reference + echo + noise``;
    any peak protection is one common gain applied to every returned component.
    """
    near_input = _array_1d("near", near)
    far_input = _array_1d("far", far)
    noise_input = _array_1d("noise_unit", noise_unit)
    if len(near_input) != len(far_input) or len(near_input) != len(noise_input):
        raise ValueError("near, far, and noise_unit must have identical lengths")
    if isinstance(rate, bool) or not isinstance(rate, (int, np.integer)):
        raise ValueError("rate must be a positive integer")
    rate = int(rate)
    if rate <= 0:
        raise ValueError("rate must be a positive integer")

    canonical_direction = DIRECTIONS.get(str(direction).lower())
    if canonical_direction is None:
        raise ValueError(f"unsupported direction: {direction!r}")
    canonical_kind = str(kind).lower()
    if canonical_kind not in KINDS:
        raise ValueError(f"unsupported kind: {kind!r}")
    canonical_bandwidth = str(bandwidth).upper()
    if canonical_bandwidth not in BAND_LIMITS_HZ:
        raise ValueError(f"unsupported bandwidth: {bandwidth!r}")

    scalar_values = {
        "snr_db": snr_db,
        "ser_db": ser_db,
        "echo_delay_ms": echo_delay_ms,
        "input_level_db": input_level_db,
        "volume_gain_db": volume_gain_db,
    }
    for name, value in scalar_values.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if float(echo_delay_ms) < 0.0:
        raise ValueError("echo_delay_ms must be non-negative")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)

    frames = len(near_input)
    masks, segments = _scene(canonical_kind, canonical_direction, frames, rate)
    snr_applicable = (
        canonical_direction != "receive"
        and canonical_kind in {"speech", "duplex", "cng", "response", "oob"}
    )
    ser_applicable = (
        canonical_direction != "receive"
        and canonical_kind == "duplex"
    )
    noise_normalization_mask = (
        masks["near"]
        if snr_applicable
        else np.ones(frames, dtype=bool)
    )
    noise_0, noise_1, noise_recipe = _unit_noise_variants(
        noise_input,
        seed,
        rate,
        noise_normalization_mask,
        normalization_name="near_active" if snr_applicable else "full",
    )
    near_gain = (
        10.0 ** (float(input_level_db) / 20.0)
        if canonical_direction != "receive"
        else 1.0
    )
    far_input_gain = (
        10.0 ** (float(input_level_db) / 20.0)
        if canonical_direction == "receive"
        else 1.0
    )
    volume_gain = 10.0 ** (float(volume_gain_db) / 20.0)

    near_faded = _fade_edges(near_input, rate)
    far_faded = _fade_edges(far_input, rate)
    reference = near_faded * masks["near"] * near_gain

    multisine_metadata: dict[str, Any] | None = None
    if canonical_kind in {"response", "oob"}:
        excitation, multisine_metadata = _multisine(
            frames,
            rate,
            canonical_bandwidth,
            seed,
            with_oob_tone=canonical_kind == "oob",
        )
        excitation *= 0.055
        if canonical_direction == "receive":
            far_base = excitation * far_input_gain * volume_gain
            reference = np.zeros(frames, dtype=np.float64)
        else:
            reference = excitation * near_gain
            far_base = np.zeros(frames, dtype=np.float64)
    else:
        far_base = far_faded * masks["far"]
    far_end = (
        far_base
        if canonical_kind in {"response", "oob"}
        else far_base * volume_gain * far_input_gain
    )

    near_active = reference[masks["near"]]
    near_active_rms = _rms(near_active)
    if snr_applicable:
        if near_active_rms <= 1e-12:
            raise ValueError(
                f"{canonical_kind} requires non-silent active near excitation"
            )
        noise_scale = near_active_rms / (10.0 ** (float(snr_db) / 20.0))
    elif canonical_kind == "idle":
        noise_scale = IDLE_FLOOR_RMS
    else:
        noise_scale = 0.01 if canonical_kind in {"response", "oob"} else 0.005
    noise_0 = noise_0 * noise_scale
    noise_1 = noise_1 * noise_scale

    echo_0 = np.zeros(frames, dtype=np.float64)
    echo_1 = np.zeros(frames, dtype=np.float64)
    echo_gain_0 = 0.0
    echo_gain_1 = 0.0
    echo_paths: list[dict[str, Any]] = []
    if canonical_direction != "receive" and canonical_kind == "duplex":
        raw_echo_0, path_0 = _echo_path(
            frames, rate, far_end, float(echo_delay_ms), channel=0
        )
        raw_echo_1, path_1 = _echo_path(
            frames, rate, far_end, float(echo_delay_ms), channel=1
        )
        far_only_rms = _rms(raw_echo_0, masks["far_only"])
        if near_active_rms <= 1e-12 or far_only_rms <= 1e-12:
            raise ValueError(
                "duplex SER requires observable near-active and far-only windows"
            )
        echo_target_rms = (
            near_active_rms
            * (10.0 ** (float(ser_db) / 20.0))
            * volume_gain
        )
        echo_gain_0 = echo_target_rms / far_only_rms
        echo_gain_1 = echo_gain_0
        echo_0 = raw_echo_0 * echo_gain_0
        echo_1 = raw_echo_1 * echo_gain_1
        path_0["echo_gain_linear"] = float(echo_gain_0)
        path_1["echo_gain_linear"] = float(echo_gain_1)
        echo_paths = [path_0, path_1]

    if canonical_direction == "receive":
        microphone = noise_0.copy()
        if canonical_kind in {"response", "oob"}:
            reference = np.zeros(frames, dtype=np.float64)
        noise = microphone.copy()
    else:
        delayed_reference = _delay(reference, 1)
        microphone = np.column_stack((
            reference + echo_0 + noise_0,
            delayed_reference + echo_1 + noise_1,
        ))
        noise = noise_0

    returned_arrays = [reference, far_end, noise, microphone]
    common_gain = 1.0
    current_peak = _peak(*returned_arrays)
    if current_peak > PEAK_TARGET:
        common_gain = PEAK_TARGET / current_peak
        reference *= common_gain
        far_end *= common_gain
        noise *= common_gain
        microphone *= common_gain
        echo_0 *= common_gain
        echo_1 *= common_gain
        noise_1 *= common_gain

    noise_channel_scale_linear = noise_scale * common_gain
    noise_recipe.update({
        "noise_scale": float(noise_scale),
        "common_gain": float(common_gain),
        "noise_channel_scale_linear": float(noise_channel_scale_linear),
        "channel_1_used_for_microphone": canonical_direction != "receive",
    })

    signals = {
        "microphone": np.asarray(microphone, dtype=np.float64),
        "reference": np.asarray(reference, dtype=np.float64),
        "far_end": np.asarray(far_end, dtype=np.float64),
        "noise": np.asarray(noise, dtype=np.float64),
    }
    if canonical_direction == "receive":
        source_mix_residual = None
    else:
        if echo_paths:
            reconstructed_echo_0 = _apply_echo_taps(
                signals["far_end"],
                echo_paths[0]["tap_samples"],
                echo_paths[0]["tap_gains"],
            )
            reconstructed_echo_0 *= echo_gain_0
        else:
            reconstructed_echo_0 = np.zeros(frames, dtype=np.float64)
        source_mix_residual = float(np.max(np.abs(
            signals["microphone"][:, 0]
            - signals["reference"]
            - reconstructed_echo_0
            - signals["noise"]
        )))
    peak_by_signal = {
        name: float(np.max(np.abs(values)))
        for name, values in signals.items()
    }
    measured_snr = None
    if snr_applicable:
        measured_snr = _db(
            _rms(signals["reference"], masks["near"]),
            _rms(signals["noise"], masks["near"]),
        )
    measured_ser = None
    if (
        ser_applicable
        and np.any(masks["far_only"])
        and np.any(masks["near"])
    ):
        measured_ser = _db(
            _rms(echo_0, masks["far_only"]),
            _rms(signals["reference"], masks["near"]),
        )
    snr_db_target = float(snr_db) if snr_applicable else None
    ser_db_nominal_target = float(ser_db) if ser_applicable else None
    ser_db_effective_target = (
        float(ser_db) + float(volume_gain_db)
        if ser_applicable
        else None
    )
    noise_spatial_check = {
        "channel_0_rms_over_normalization_mask": _rms(
            signals["noise"],
            noise_normalization_mask,
        ),
        "channel_1_rms_over_normalization_mask": _rms(
            noise_1,
            noise_normalization_mask,
        ),
        "channel_correlation_over_normalization_mask": _correlation(
            signals["noise"],
            noise_1,
            noise_normalization_mask,
        ),
        "channel_1_used_for_microphone": canonical_direction != "receive",
    }

    beam_delays_ms = (
        [1000.0 / rate, 0.0]
        if canonical_direction != "receive"
        else [0.0]
    )
    cng_measurement_start_s = (
        round(min(4.0, frames / rate), 9) if canonical_kind == "cng" else None
    )
    metadata: dict[str, Any] = {
        "kind": canonical_kind,
        "direction": canonical_direction,
        "sample_rate_hz": rate,
        "duration_s": round(frames / rate, 9),
        "bandwidth": canonical_bandwidth,
        "segments": segments,
        "beam_delays_ms": beam_delays_ms,
        "near_arrival_delays_samples": (
            [0, 1] if canonical_direction != "receive" else [0]
        ),
        "snr_db_target_applicable": snr_applicable,
        "snr_db_target": snr_db_target,
        "snr_db_measured": measured_snr,
        "snr_db_definition": (
            "20*log10(RMS(reference[near_active])"
            " / RMS(noise[near_active]))"
            if snr_applicable
            else None
        ),
        "ser_db_target_applicable": ser_applicable,
        "ser_db_target": ser_db_nominal_target,
        "ser_db_nominal_target": ser_db_nominal_target,
        "ser_db_effective_target": ser_db_effective_target,
        "ser_db_measured": measured_ser,
        "ser_db_definition": (
            "20*log10(RMS(echo[far_only]) / RMS(reference[near_active])); "
            "effective target = ser_db + volume_gain_db"
            if ser_applicable
            else None
        ),
        "echo_delay_ms": float(echo_delay_ms),
        "echo_paths": echo_paths,
        "source_mix_check": {
            "applicable": canonical_direction != "receive",
            "formula": (
                "microphone[:,0] = reference + echo_component + noise"
                if canonical_direction != "receive"
                else None
            ),
            "echo_component_representation": (
                "truncate(convolve(far_end, echo_paths[ch].impulse_response)"
                " * echo_paths[ch].echo_gain_linear, input_length)"
                if canonical_direction != "receive"
                else None
            ),
            "echo_component_metadata": (
                "echo_paths[].tap_samples, tap_gains, echo_gain_linear"
                if canonical_direction != "receive"
                else None
            ),
            "residual_max_abs": source_mix_residual,
        },
        "noise_spatial": {
            "returned_noise": (
                "one-dimensional channel-0 noise; channel 1 is reconstructible "
                "from the compact recipe"
            ),
            "channel_1_recipe": noise_recipe,
            "check": noise_spatial_check,
        },
        "cng_measurement_start_s": cng_measurement_start_s,
        "idle_noise_floor_dbfs": (
            -75.0 if canonical_kind == "idle" else None
        ),
        "stimulus_source": "synthetic",
        "source_inputs": "caller-provided near/far/noise_unit",
        "global_safety_gain": float(common_gain),
        "global_safety_gain_db": float(20.0 * math.log10(common_gain)),
        "gain_scope": {
            "input_level_db": {
                "value": float(input_level_db),
                "targets": (
                    ["far_end"]
                    if canonical_direction == "receive"
                    else ["near", "reference", "near-driven noise"]
                ),
            },
            "volume_gain_db": {
                "value": float(volume_gain_db),
                "targets": ["far_end playback", "echo path"],
            },
            "common_safety_gain": {
                "linear": float(common_gain),
                "targets": ["microphone", "reference", "far_end", "noise"],
            },
        },
        "component_relation": (
            "microphone[:,0] = reference + echo_ch0 + noise"
            if canonical_direction != "receive"
            else "microphone is a receive-side noise dummy"
        ),
        "peak": float(max(peak_by_signal.values())),
        "peak_by_signal": peak_by_signal,
        "clipped": False,
        "standard_compliance": False,
        "evidence_scope": "synthetic engineering stimulus only",
    }
    if multisine_metadata is not None:
        metadata.update(multisine_metadata)

    return signals, metadata
