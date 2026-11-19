"""Runtime microphone synthesis from post-RX playback.

The dataset keeps the source components immutable.  For send/round-trip
execution, the microphone is derived after VCP RX has produced the actual
playback PCM.  This keeps the acoustic echo in the microphone and the AEC
reference on the same time base.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping

import numpy as np


class DynamicMicrophoneError(ValueError):
    """The source components or render metadata cannot form a safe mixture."""


@dataclass(frozen=True)
class DynamicMicrophoneMix:
    microphone: np.ndarray
    reference: np.ndarray
    noise: np.ndarray
    capture_gain: float
    peak_before_gain: float
    channels: int
    formula: str


def _one_dimensional(name: str, value: np.ndarray) -> np.ndarray:
    samples = np.asarray(value, dtype=np.float64).copy()
    if samples.ndim != 1 or not len(samples):
        raise DynamicMicrophoneError(f"{name} must be a non-empty 1-D signal")
    if not np.all(np.isfinite(samples)):
        raise DynamicMicrophoneError(f"{name} contains non-finite samples")
    return samples


def _delay(samples: np.ndarray, count: int) -> np.ndarray:
    if count < 0:
        raise DynamicMicrophoneError("near arrival delay must be non-negative")
    if count >= len(samples):
        raise DynamicMicrophoneError(
            "near arrival delay must be inside the delivered signal"
        )
    result = np.zeros_like(samples)
    if count < len(samples):
        result[count:] = samples[: len(samples) - count]
    return result


def _fir_component(source: np.ndarray, path: Mapping[str, Any] | None) -> np.ndarray:
    component = np.zeros(len(source), dtype=np.float64)
    if path is None:
        return component
    offsets = path.get("tap_samples")
    gains = path.get("tap_gains")
    echo_gain = path.get("echo_gain_linear")
    if (
        not isinstance(offsets, list)
        or not isinstance(gains, list)
        or len(offsets) != len(gains)
    ):
        raise DynamicMicrophoneError(
            "echo path requires matching tap_samples and tap_gains"
        )
    if echo_gain is None or not math.isfinite(float(echo_gain)):
        raise DynamicMicrophoneError("echo path requires finite echo_gain_linear")
    for raw_offset, raw_gain in zip(offsets, gains, strict=True):
        offset = int(raw_offset)
        gain = float(raw_gain)
        if offset != raw_offset or offset < 0 or not math.isfinite(gain):
            raise DynamicMicrophoneError("echo FIR taps are invalid")
        if offset < len(component):
            component[offset:] += source[: len(component) - offset] * gain
    return component * float(echo_gain)


def _scale_echo_path(
    path: Mapping[str, Any] | None,
    rate_scale: float,
) -> Mapping[str, Any] | None:
    if path is None or rate_scale == 1.0:
        return path
    offsets = path.get("tap_samples")
    if not isinstance(offsets, list):
        raise DynamicMicrophoneError("echo path tap_samples must be a list")
    scaled_offsets: list[int] = []
    for raw_offset in offsets:
        if (
            isinstance(raw_offset, bool)
            or not isinstance(raw_offset, int)
            or raw_offset < 0
        ):
            raise DynamicMicrophoneError("echo FIR taps are invalid")
        scaled_offsets.append(int(round(raw_offset * rate_scale)))
    return {
        **path,
        "tap_samples": scaled_offsets,
    }


def _window_mask(
    frames: int,
    sample_rate: int,
    start_s: float,
    end_s: float,
) -> np.ndarray:
    if not math.isfinite(start_s) or not math.isfinite(end_s) or end_s < start_s:
        raise DynamicMicrophoneError("echo segment window bounds are invalid")
    mask = np.zeros(frames, dtype=bool)
    start = max(0, min(frames, int(round(start_s * sample_rate))))
    end = max(0, min(frames, int(round(end_s * sample_rate))))
    mask[start:end] = True
    return mask


def _segmented_echo_components(
    playback: np.ndarray,
    segments: list[Any],
    echo_gain_linear: Any,
    *,
    sample_rate: int,
    rate_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    if (
        echo_gain_linear is None
        or isinstance(echo_gain_linear, bool)
        or not isinstance(echo_gain_linear, (int, float))
        or not math.isfinite(float(echo_gain_linear))
        or float(echo_gain_linear) < 0.0
    ):
        raise DynamicMicrophoneError(
            "segmented echo requires finite non-negative echo_gain_linear"
        )
    gain = float(echo_gain_linear)
    echo0 = np.zeros(len(playback), dtype=np.float64)
    echo1 = np.zeros(len(playback), dtype=np.float64)
    for segment in segments:
        if not isinstance(segment, Mapping):
            raise DynamicMicrophoneError(
                "echo_path_segments entry must be a mapping"
            )
        raw_start = segment.get("start_s")
        raw_end = segment.get("end_s")
        if (
            isinstance(raw_start, bool)
            or isinstance(raw_end, bool)
            or not isinstance(raw_start, (int, float))
            or not isinstance(raw_end, (int, float))
        ):
            raise DynamicMicrophoneError(
                "echo segment requires numeric start_s/end_s"
            )
        seg_paths = segment.get("paths")
        if not isinstance(seg_paths, list) or len(seg_paths) != 2:
            raise DynamicMicrophoneError(
                "echo segment requires two channel paths"
            )
        mask = _window_mask(
            len(playback),
            sample_rate,
            float(raw_start),
            float(raw_end),
        )
        echo0 += (
            _fir_component(
                playback,
                _scale_echo_path(seg_paths[0], rate_scale),
            )
            * mask
            * gain
        )
        echo1 += (
            _fir_component(
                playback,
                _scale_echo_path(seg_paths[1], rate_scale),
            )
            * mask
            * gain
        )
    return echo0, echo1


def _noise_channel(
    noise0: np.ndarray,
    metadata: Mapping[str, Any],
    channel: int,
) -> np.ndarray:
    if channel == 0:
        return np.asarray(noise0, dtype=np.float64).copy()
    recipe = metadata.get("noise_spatial", {}).get("channel_1_recipe")
    if not isinstance(recipe, Mapping):
        raise DynamicMicrophoneError(
            "dual-microphone synthesis requires noise_spatial.channel_1_recipe"
        )
    try:
        rho = float(recipe["rho"])
        seed2 = int(recipe["seed2"])
        normalizer = float(recipe["normalizer"])
        scale = float(recipe["noise_channel_scale_linear"])
    except (KeyError, TypeError, ValueError) as error:
        raise DynamicMicrophoneError("dual-microphone noise recipe is incomplete") from error
    if (
        not math.isfinite(rho)
        or not -1.0 < rho < 1.0
        or seed2 != recipe["seed2"]
        or not math.isfinite(normalizer)
        or normalizer <= 0.0
        or not math.isfinite(scale)
        or scale <= 0.0
    ):
        raise DynamicMicrophoneError("dual-microphone noise recipe is invalid")
    unit0 = np.asarray(noise0, dtype=np.float64) / scale
    independent2 = np.random.default_rng(seed2).normal(size=len(unit0))
    mixed = rho * unit0 + math.sqrt(1.0 - rho * rho) * independent2
    return mixed / normalizer * scale


def synthesize_post_rx_microphone(
    reference: np.ndarray,
    noise: np.ndarray,
    post_rx_playback: np.ndarray,
    render_metadata: Mapping[str, Any],
    *,
    sample_rate: int,
) -> DynamicMicrophoneMix:
    """Build the two capture channels from the actual post-RX playback.

    The returned ``reference`` and ``noise`` are capture-scaled together with
    the microphone.  ``post_rx_playback`` itself is never scaled: it is the
    actual playback/AEC reference and must remain byte-identical to the RX
    artifact consumed by TX.
    """
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise DynamicMicrophoneError("sample_rate must be a positive integer")
    near = _one_dimensional("reference", reference)
    noise0 = _one_dimensional("noise", noise)
    playback = _one_dimensional("post_rx_playback", post_rx_playback)
    if len(near) != len(noise0) or len(near) != len(playback):
        raise DynamicMicrophoneError(
            "reference, noise and post-RX playback lengths must match"
        )

    delays = render_metadata.get("near_arrival_delays_samples")
    if (
        not isinstance(delays, list)
        or len(delays) != 2
        or any(
            isinstance(value, bool)
            or int(value) != value
            or int(value) < 0
            for value in delays
        )
    ):
        raise DynamicMicrophoneError(
            "render metadata requires two non-negative near arrival delays"
        )
    raw_segments = render_metadata.get("echo_path_segments")
    if raw_segments is not None and not isinstance(raw_segments, list):
        raise DynamicMicrophoneError(
            "render metadata echo_path_segments must be a list"
        )
    use_segments = isinstance(raw_segments, list) and len(raw_segments) > 0
    paths = render_metadata.get("echo_paths", [])
    if not use_segments and (not isinstance(paths, list) or len(paths) not in (0, 2)):
        raise DynamicMicrophoneError(
            "render metadata echo_paths must contain zero or two paths"
        )

    meta_rate = render_metadata.get("sample_rate_hz")
    rate_scale = (
        sample_rate / float(meta_rate)
        if isinstance(meta_rate, int)
        and not isinstance(meta_rate, bool)
        and meta_rate > 0
        and meta_rate != sample_rate
        else 1.0
    )
    delay0 = int(round(int(delays[0]) * rate_scale))
    delay1 = int(round(int(delays[1]) * rate_scale))
    if use_segments:
        echo0, echo1 = _segmented_echo_components(
            playback,
            raw_segments,
            render_metadata.get("echo_gain_linear"),
            sample_rate=sample_rate,
            rate_scale=rate_scale,
        )
        formula = (
            "microphone[:,0] = delayed_reference_0 + segmented_FIR(post_rx_playback) + noise; "
            "microphone[:,1] = delayed_reference_1 + segmented_FIR(post_rx_playback) + "
            "reconstructed_noise_1"
        )
    else:
        scaled_paths = [_scale_echo_path(p, rate_scale) for p in paths]
        echo0 = _fir_component(playback, scaled_paths[0] if scaled_paths else None)
        echo1 = _fir_component(playback, scaled_paths[1] if scaled_paths else None)
        formula = (
            "microphone[:,0] = reference + FIR(post_rx_playback) + noise; "
            "microphone[:,1] = delayed_reference + FIR(post_rx_playback) + "
            "reconstructed_noise_1"
        )
    noise1 = _noise_channel(noise0, render_metadata, 1)
    mic0 = _delay(near, delay0) + echo0 + noise0
    mic1 = _delay(near, delay1) + echo1 + noise1
    microphone = np.column_stack((mic0, mic1))

    peak = float(np.max(np.abs(microphone)))
    if not math.isfinite(peak):
        raise DynamicMicrophoneError("synthesized microphone is non-finite")
    capture_gain = min(1.0, 0.94 / peak) if peak > 0.94 else 1.0
    if capture_gain < 1.0:
        near = near * capture_gain
        noise0 = noise0 * capture_gain
        microphone = microphone * capture_gain

    return DynamicMicrophoneMix(
        microphone=np.asarray(microphone, dtype=np.float64),
        reference=np.asarray(near, dtype=np.float64),
        noise=np.asarray(noise0, dtype=np.float64),
        capture_gain=float(capture_gain),
        peak_before_gain=peak,
        channels=2,
        formula=formula,
    )


_SCOPED_ECHO_OFFSETS_MS = (
    (20.0, 31.5, 47.0, 73.0),
    (25.0, 40.5, 59.0, 86.0),
)
_SCOPED_ECHO_GAINS = (
    (0.5555555555555556, 0.25, 0.13333333333333333, 0.06111111111111111),
    (0.5454545454545454, 0.20454545454545453, 0.16477272727272727, 0.08522727272727272),
)
_SCOPED_NOISE_RHO = 0.65


def _scoped_echo_path(
    channel: int,
    sample_rate: int,
    echo_gain_linear: float,
) -> dict[str, Any]:
    return {
        "tap_samples": [
            int(round(offset_ms * sample_rate / 1000.0))
            for offset_ms in _SCOPED_ECHO_OFFSETS_MS[channel]
        ],
        "tap_gains": list(_SCOPED_ECHO_GAINS[channel]),
        "echo_gain_linear": float(echo_gain_linear),
    }


def synthesize_scoped_post_rx_microphone(
    reference: np.ndarray,
    noise: np.ndarray,
    post_rx_playback: np.ndarray,
    *,
    sample_rate: int,
    channels: int = 1,
    beam_delays_ms: list[float] | None = None,
    case_id: str = "",
    echo_gain_linear: float = 0.25,
) -> DynamicMicrophoneMix:
    """Build a deterministic 1- or 2-channel runtime microphone for scoped datasets.

    Used for acceptance datasets (such as the 336-item background-noise suite)
    that do not carry per-case ``render_metadata``.  ``post_rx_playback`` is
    never scaled so it stays byte-identical to the RX artifact consumed by TX.
    """
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, int)
        or sample_rate <= 0
    ):
        raise DynamicMicrophoneError("sample_rate must be a positive integer")
    if isinstance(channels, bool) or channels not in (1, 2):
        raise DynamicMicrophoneError(
            "scoped runtime synthesis supports 1 or 2 channels"
        )
    if (
        isinstance(echo_gain_linear, bool)
        or not isinstance(echo_gain_linear, (int, float))
        or not math.isfinite(float(echo_gain_linear))
        or float(echo_gain_linear) < 0.0
    ):
        raise DynamicMicrophoneError(
            "echo_gain_linear must be finite and non-negative"
        )

    near = _one_dimensional("reference", reference)
    noise0 = _one_dimensional("noise", noise)
    playback = _one_dimensional("post_rx_playback", post_rx_playback)
    if len(near) != len(noise0) or len(near) != len(playback):
        raise DynamicMicrophoneError(
            "reference, noise and post-RX playback lengths must match"
        )

    echo0 = _fir_component(
        playback,
        _scoped_echo_path(0, sample_rate, float(echo_gain_linear)),
    )
    if channels == 1:
        microphone = (near + echo0 + noise0)[:, None]
        formula = "microphone[:,0] = reference + FIR(post_rx_playback) + noise"
    else:
        if (
            isinstance(beam_delays_ms, list)
            and len(beam_delays_ms) == 2
            and all(
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(float(value))
                and float(value) >= 0.0
                for value in beam_delays_ms
            )
        ):
            max_delay = max(float(value) for value in beam_delays_ms)
            delay0 = int(
                round((max_delay - float(beam_delays_ms[0])) * sample_rate / 1000.0)
            )
            delay1 = int(
                round((max_delay - float(beam_delays_ms[1])) * sample_rate / 1000.0)
            )
        else:
            delay0, delay1 = 0, 1
        echo1 = _fir_component(
            playback,
            _scoped_echo_path(1, sample_rate, float(echo_gain_linear)),
        )
        rms0 = float(np.sqrt(np.mean(np.square(noise0, dtype=np.float64))))
        if rms0 > 0.0:
            seed_bytes = hashlib.sha256(
                f"scoped-runtime-mic:{case_id}:{sample_rate}".encode("utf-8")
            ).digest()[:8]
            seed = int.from_bytes(seed_bytes, "little")
            independent = np.random.default_rng(seed).normal(size=len(noise0))
            unit0 = noise0 / rms0
            mixed = (
                _SCOPED_NOISE_RHO * unit0
                + math.sqrt(1.0 - _SCOPED_NOISE_RHO * _SCOPED_NOISE_RHO)
                * independent
            )
            mixed_rms = float(
                np.sqrt(np.mean(np.square(mixed, dtype=np.float64)))
            )
            noise1 = (
                (mixed / mixed_rms * rms0)
                if mixed_rms > 0.0
                else np.zeros_like(noise0)
            )
        else:
            noise1 = np.zeros_like(noise0)
        mic0 = _delay(near, delay0) + echo0 + noise0
        mic1 = _delay(near, delay1) + echo1 + noise1
        microphone = np.column_stack((mic0, mic1))
        formula = (
            "microphone[:,0] = delayed_reference_0 + FIR(post_rx_playback) + noise; "
            "microphone[:,1] = delayed_reference_1 + FIR(post_rx_playback) + "
            "spatial_noise_1"
        )

    peak = float(np.max(np.abs(microphone)))
    if not math.isfinite(peak):
        raise DynamicMicrophoneError("synthesized microphone is non-finite")
    capture_gain = min(1.0, 0.94 / peak) if peak > 0.94 else 1.0
    if capture_gain < 1.0:
        near = near * capture_gain
        noise0 = noise0 * capture_gain
        microphone = microphone * capture_gain

    return DynamicMicrophoneMix(
        microphone=np.asarray(microphone, dtype=np.float64),
        reference=np.asarray(near, dtype=np.float64),
        noise=np.asarray(noise0, dtype=np.float64),
        capture_gain=float(capture_gain),
        peak_before_gain=peak,
        channels=channels,
        formula=formula,
    )


__all__ = [
    "DynamicMicrophoneError",
    "DynamicMicrophoneMix",
    "synthesize_post_rx_microphone",
    "synthesize_scoped_post_rx_microphone",
]
