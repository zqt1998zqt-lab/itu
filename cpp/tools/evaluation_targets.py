"""Independent clean-array timeline for the declared causal beam latency.

No measured output, lag search, speech label or score enters this calculation.
The source microphones are immutable. Interpolation is applied only to the
offline array target, using the same fixed geometry at the actual core rate.
"""
import math
import numpy as np
from scipy.signal import resample_poly


def calibrated_beam_target(clean_mics, delays, io_rate, source_rate=16000, core_rate=None):
    x = np.asarray(clean_mics, dtype=np.float64)
    d = np.asarray(delays, dtype=np.float64)
    if (x.ndim != 2 or not len(x) or x.shape[1] != len(d) or
            not np.isfinite(x).all() or not np.isfinite(d).all() or
            np.any(d < 0) or io_rate not in (16000, 24000, 48000)):
        raise ValueError("finite clean microphone array and calibrated causal delays required")
    core_rate = (48000 if io_rate == 24000 else io_rate) if core_rate is None else core_rate
    if core_rate not in (16000, 48000) or core_rate < io_rate:
        raise ValueError("unsupported processing core rate")
    gcd = math.gcd(core_rate, source_rate)
    core = resample_poly(x, core_rate // gcd, source_rate // gcd, axis=0) if core_rate != source_rate else x
    scaled = d * core_rate / source_rate
    core_bound = math.ceil(float(max(scaled)))
    # NativePipeline rounds total latency up to the I/O sample grid. Its other
    # fixed stages have integral 24 kHz delays; an odd 48 kHz beam bound rounds
    # to the next even sample. Retain that deterministic fractional offset.
    boundary = math.ceil(core_bound * io_rate / core_rate) * core_rate / io_rate
    time = np.arange(len(core), dtype=np.float64)
    target = sum(np.interp(time + boundary - delay, time, core[:, ch], left=0, right=0)
                 for ch, delay in enumerate(scaled)) / len(scaled)
    target = resample_poly(target, source_rate // gcd, core_rate // gcd) if core_rate != source_rate else target
    return target[:len(x)], dict(method="fixed geometry at processing rate; declared ceil latency; no lag fitting",
                                core_rate=core_rate, beam_bound_core_samples=core_bound,
                                target_origin_core_samples=boundary,
                                legacy_target_origin_source_samples=math.ceil(float(max(d))))
