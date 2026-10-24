"""Create a traceable measurement bundle from existing measured DSP evidence.

This is deliberately an import/compiler, not an acoustic simulator.  It reads
already-recorded PCM traces, derives only metrics that those traces support,
and leaves all unsupported acoustic tests explicitly unmeasured.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf


SUPPORTED_FAMILIES = {
    "round_trip_delay": "loopback",
    "frequency_response": "transfer",
    "idle_noise_level": "noise",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mono(path: Path) -> tuple[np.ndarray, int]:
    data, rate = sf.read(path, always_2d=True)
    if data.shape[1] != 1 or not len(data) or not np.isfinite(data).all():
        raise ValueError(f"{path}: finite mono PCM required")
    return data[:, 0], rate


def _rms_db(x: np.ndarray) -> float:
    power = float(np.mean(x.astype(np.float64) ** 2))
    if power <= 0:
        raise ValueError("zero-power noise trace cannot produce a level")
    return 10 * math.log10(power)


def loopback_delay_ms(reference: Path, capture: Path, max_delay_ms: float) -> float:
    x, rate = _mono(reference)
    y, capture_rate = _mono(capture)
    if rate != capture_rate or max_delay_ms <= 0 or max_delay_ms > 5000:
        raise ValueError("matching rates and 0..5000 ms delay search required")
    max_lag = min(round(rate * max_delay_ms / 1000), min(len(x), len(y)) - 1)
    size = 1 << math.ceil(math.log2(len(x) + len(y)))
    correlation = np.fft.irfft(np.fft.rfft(y, size) *
                               np.conj(np.fft.rfft(x, size)), size)
    lag = int(np.argmax(correlation[:max_lag + 1]))
    return lag * 1000 / rate


def noise_level_dbfs(path: Path, start_s: float, end_s: float) -> float:
    x, rate = _mono(path)
    lo, hi = round(start_s * rate), round(end_s * rate)
    if not 0 <= lo < hi <= len(x):
        raise ValueError("noise interval lies outside the trace")
    return _rms_db(x[lo:hi])


def response_shape(
    reference: Path, capture: Path, mask: dict, delay_samples: int
) -> float:
    x, rate = _mono(reference)
    y, capture_rate = _mono(capture)
    if rate != capture_rate or delay_samples < 0 or delay_samples >= len(y):
        raise ValueError("matching rates and causal delay required")
    n = min(len(x), len(y) - delay_samples)
    if n < rate:
        raise ValueError("at least one second of aligned response required")
    x, y = x[:n], y[delay_samples:delay_samples + n]
    if np.max(np.abs(x)) >= .999 or np.max(np.abs(y)) >= .999:
        raise ValueError("clipped trace cannot be used")
    nperseg = rate // 4
    frequency = np.fft.rfftfreq(nperseg, 1 / rate)
    # Overlap-averaged spectra provide a coherence-gated H1 response.
    step, frames = nperseg // 2, []
    window = np.hanning(nperseg)
    pxx = np.zeros(len(frequency))
    pyy = np.zeros(len(frequency))
    pxy = np.zeros(len(frequency), dtype=complex)
    for start in range(0, n - nperseg + 1, step):
        xa = x[start:start + nperseg] * window
        ya = y[start:start + nperseg] * window
        X, Y = np.fft.rfft(xa), np.fft.rfft(ya)
        pxx += np.abs(X) ** 2
        pyy += np.abs(Y) ** 2
        pxy += Y * np.conj(X)
        frames.append(start)
    if len(frames) < 4:
        raise ValueError("insufficient response frames")
    coherence = np.clip(
        np.abs(pxy) ** 2 / np.maximum(pxx * pyy, 1e-30), 0, 1
    )
    gain = 20 * np.log10(
        np.maximum(np.abs(pxy) / np.maximum(pxx, 1e-30), 1e-15)
    )
    points = np.array(mask["frequency_hz"], dtype=float)
    lo, hi = np.array(mask["lower_db"]), np.array(mask["upper_db"])
    normalized_at_hz = mask.get("normalized_at_hz")
    if (
        isinstance(normalized_at_hz, bool)
        or not isinstance(normalized_at_hz, (int, float))
        or not math.isfinite(normalized_at_hz)
        or normalized_at_hz <= 0
    ):
        raise ValueError("response mask requires a positive normalized_at_hz")
    if not (
        np.all(np.diff(points) > 0)
        and np.all(np.isfinite(lo))
        and np.all(np.isfinite(hi))
        and points[0] <= normalized_at_hz <= points[-1]
    ):
        raise ValueError("invalid ordered mask")
    values = np.interp(points, frequency, gain)
    normalization = np.interp(
        np.array([normalized_at_hz]), frequency, gain
    )[0]
    if not math.isfinite(normalization):
        raise ValueError("response normalization is not finite")
    values -= normalization
    coherence_points = np.interp(points, frequency, coherence)
    if np.any(coherence_points < .6):
        raise ValueError("coherence below 0.6 at a response mask point")
    return float(np.minimum(values - lo, hi - values).min())


def build_measurement_bundle(
    *,
    item_id: str,
    unit: str,
    kind: str,
    reference: Path,
    capture: Path,
    max_delay_ms: float = 1000,
    start_s: float = 0,
    end_s: float | None = None,
    mask: dict | None = None,
    delay_samples: int = 0,
    trace_path: str = "",
) -> dict:
    """Return one bundle row plus the trace identity that supports it."""
    if kind == "loopback":
        value = loopback_delay_ms(reference, capture, max_delay_ms)
    elif kind == "noise":
        if end_s is None:
            raise ValueError("noise measurement requires an explicit end_s")
        value = noise_level_dbfs(capture, start_s, end_s)
    elif kind == "transfer":
        if mask is None:
            raise ValueError("transfer measurement requires an official mask object")
        value = response_shape(reference, capture, mask, delay_samples)
    else:
        raise ValueError(f"unsupported evidence compiler kind: {kind}")
    return {
        "item_id": item_id,
        "value": value,
        "unit": unit,
        "method_id": f"server-evidence-compiler-v1:{kind}",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "trace": {
            "path": trace_path,
            "sha256": hashlib.sha256(
                (_sha(reference) + ":" + _sha(capture)).encode()
            ).hexdigest(),
            "kind": "measurement_trace",
        },
        "notes": (
            "Derived from paired PCM input/output traces. "
            "Digital analysis is not a calibrated acoustic result."
        ),
        "source_identity": {
            "reference": str(reference),
            "reference_sha256": _sha(reference),
            "capture": str(capture),
            "capture_sha256": _sha(capture),
        },
    }
