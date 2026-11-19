"""Runtime microphone synthesis from post-RX playback.

The dataset keeps the source components immutable.  For send/round-trip
execution, the microphone is derived after VCP RX has produced the actual
playback PCM.  This keeps the acoustic echo in the microphone and the AEC
reference on the same time base.
"""
from __future__ import annotations

from dataclasses import dataclass
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
        or any(int(value) != value or int(value) < 0 for value in delays)
    ):
        raise DynamicMicrophoneError(
            "render metadata requires two non-negative near arrival delays"
        )
    paths = render_metadata.get("echo_paths", [])
    if not isinstance(paths, list) or len(paths) not in (0, 2):
        raise DynamicMicrophoneError(
            "render metadata echo_paths must contain zero or two paths"
        )

    meta_rate = render_metadata.get("sample_rate_hz")
    rate_scale = (
        sample_rate / float(meta_rate)
        if isinstance(meta_rate, int) and not isinstance(meta_rate, bool) and meta_rate > 0 and meta_rate != sample_rate
        else 1.0
    )
    scaled_paths = [
        {
            **p,
            "tap_samples": [int(round(int(s) * rate_scale)) for s in p.get("tap_samples", [])],
        }
        for p in paths
    ] if rate_scale != 1.0 else paths
    delay1 = int(round(int(delays[1]) * rate_scale))
    echo0 = _fir_component(playback, scaled_paths[0] if scaled_paths else None)
    echo1 = _fir_component(playback, scaled_paths[1] if scaled_paths else None)
    noise1 = _noise_channel(noise0, render_metadata, 1)
    mic0 = near + echo0 + noise0
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
        formula=(
            "microphone[:,0] = reference + FIR(post_rx_playback) + noise; "
            "microphone[:,1] = delayed_reference + FIR(post_rx_playback) + "
            "reconstructed_noise_1"
        ),
    )


__all__ = [
    "DynamicMicrophoneError",
    "DynamicMicrophoneMix",
    "synthesize_post_rx_microphone",
]
