"""Measured PCM diagnostics, with declared alignment and no lag optimisation."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import warnings

import numpy as np
import soundfile as sf


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def db(power: float) -> float:
    return float(10 * np.log10(max(float(power), 1e-30)))


@dataclass(frozen=True)
class Audio:
    samples: np.ndarray
    rate: int
    identity: dict


def load_audio(source: Path | bytes, *, channel: int | None = None,
               offset_samples: int = 0, pcm16: bool = False) -> Audio:
    payload = source if isinstance(source, bytes) else Path(source).read_bytes()
    try:
        data, rate = sf.read(io.BytesIO(payload), always_2d=True, dtype="float64")
    except (RuntimeError, ValueError) as error:
        raise ValueError("cannot decode audio") from error
    if data.shape[1] != 1 and channel is None:
        raise ValueError("multichannel audio requires an explicit channel")
    channel = 0 if channel is None else channel
    if isinstance(channel, bool) or not isinstance(channel, int) or not 0 <= channel < data.shape[1]:
        raise ValueError("invalid channel")
    if isinstance(offset_samples, bool) or not isinstance(offset_samples, int) or not 0 <= offset_samples < len(data):
        raise ValueError("invalid declared offset_samples")
    x = data[offset_samples:, channel].copy()
    if not np.isfinite(x).all():
        raise ValueError("nonfinite audio")
    if pcm16:
        scaled = x * 32768
        rounded = np.copysign(np.floor(np.abs(scaled) + .5), scaled)
        x = np.clip(rounded, -32768, 32767).astype("<i2").astype(np.float64) / 32768
    x.setflags(write=False)
    return Audio(x, int(rate), dict(source_file_sha256=sha(payload), channel=channel,
                 source_channels=data.shape[1], offset_samples=offset_samples,
                 pcm16_quantized=pcm16, pcm16_policy="C++ lround: half away from zero, then saturate" if pcm16 else None,
                 source_frames=len(data), aligned_frames=len(x),
                 aligned_float64_sha256=sha(np.asarray(x, dtype="<f8").tobytes()),
                 sample_rate=int(rate), resampled=False, level_normalized=False))


def physical(x: np.ndarray) -> dict:
    return dict(samples=len(x), rms_dbfs=db(np.mean(x ** 2)),
                peak_dbfs=db(np.max(x ** 2)), dc=float(np.mean(x)),
                clipping_samples=int(np.count_nonzero(np.abs(x) >= .999)),
                nonfinite_samples=int(np.count_nonzero(~np.isfinite(x))))


def reference_metrics(target: np.ndarray, observed: np.ndarray, rate: int) -> dict:
    if target.shape != observed.shape or target.ndim != 1:
        raise ValueError("reference and observed samples must align exactly")
    a, b = target - target.mean(), observed - observed.mean()
    energy = float(np.dot(a, a))
    if energy / max(1, len(a)) < 1e-15:
        return dict(available=False, reason="reference_has_no_ac_excitation")
    alpha = float(np.dot(a, b) / energy)
    projected = alpha * a
    pp = float(np.mean(projected ** 2))
    residual = float(np.mean((b - projected) ** 2))
    result = dict(available=True, sisdr_db=min(120., db(pp / max(residual, 1e-30))) if pp > 1e-30 else None,
                  sisdr_censored=bool(pp > 0 and residual <= pp * 1e-12),
                  projection_gain_db=db(alpha ** 2),
                  correlation=float(np.corrcoef(a, b)[0, 1]) if np.std(b) > 1e-15 else None,
                  stoi=None, stoi_reason=None, stoi_internal_sample_rate=10000,
                  alignment="declared offsets only; no cross-correlation search")
    try:
        from pystoi import stoi
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = float(stoi(a, b, rate, extended=False))
        if caught or not np.isfinite(value):
            result["stoi_reason"] = "insufficient_active_speech_or_metric_warning"
        else:
            result["stoi"] = value
    except ImportError:
        result["stoi_reason"] = "pystoi_not_installed"
    except ValueError:
        result["stoi_reason"] = "invalid_or_too_short_active_speech_for_stoi"
    return result


def attenuation(before: np.ndarray, after: np.ndarray) -> dict:
    p, q = float(np.mean(before ** 2)), float(np.mean(after ** 2))
    if p <= 1e-15:
        return dict(available=False, reason="input_has_no_measurable_excitation")
    return dict(available=True, attenuation_db=db(p / max(q, 1e-30)),
                censored=bool(q <= 1e-30), interpretation="level reduction; not a speech-quality score")
