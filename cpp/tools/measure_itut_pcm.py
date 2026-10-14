#!/usr/bin/env python3
"""Unweighted digital diagnostics with fixed delay and explicit frequency-mask provenance."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from tune_calibrated_eq import validate_response


def dbfs(power):
    return float(10 * np.log10(power)) if power > 0 else None


def noise_windows(pcm, rate, milliseconds):
    if milliseconds not in (50, 100) or pcm.ndim != 1 or not len(pcm) or not np.isfinite(pcm).all():
        raise ValueError("finite mono samples and 50/100 ms windows required")
    size = round(rate * milliseconds / 1000)
    count = len(pcm) // size
    if not count:
        raise ValueError("noise interval too short")
    powers = np.mean(pcm[:count * size].reshape(count, size) ** 2, axis=1)
    return dict(window_ms=milliseconds, complete_windows=count, trailing_samples=len(pcm) % size,
                rms_dbfs=[dbfs(value) for value in powers], max_rms_dbfs=dbfs(float(powers.max())),
                weighting="unweighted digital PCM; not dB[Pa], dBm0 or acoustic idle noise",
                standard_compliance=False)


def transfer_response(before, after, rate, mask, delay_samples=0):
    if rate not in (16000, 24000, 48000) or delay_samples < 0:
        raise ValueError("supported rate and causal fixed delay required")
    n = min(len(before), len(after) - delay_samples)
    if n < rate or before.ndim != 1 or after.ndim != 1 or not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError("at least one second of finite mono overlap required")
    x, y = before[:n], after[delay_samples:delay_samples + n]
    if np.max(np.abs(x)) >= .999 or np.max(np.abs(y)) >= .999:
        raise ValueError("clipped response cannot calibrate EQ")
    # Multiple Welch windows provide a coherence check; a single FFT cannot.
    size = rate // 4
    frequency, pxx = signal.welch(x, rate, nperseg=size)
    _, pyy = signal.welch(y, rate, nperseg=size)
    _, pxy = signal.csd(x, y, rate, nperseg=size)
    coherence = np.clip(abs(pxy) ** 2 / np.maximum(pxx * pyy, 1e-30), 0, 1)
    gain = 20 * np.log10(np.maximum(abs(pxy) / np.maximum(pxx, 1e-30), 1e-15))
    record = dict(sample_rate=rate, scope="digital_diagnostic", measurement_id=mask["measurement_id"],
                  mask_source=mask["mask_source"], normalized_at_hz=mask["normalized_at_hz"],
                  frequency_hz=mask["frequency_hz"], lower_db=mask["lower_db"], upper_db=mask["upper_db"],
                  measured_db=[0.] * len(mask["frequency_hz"]))
    _, points, _, lower, upper, reference = validate_response(record)
    required = np.unique(np.r_[points, reference])
    bins = set()
    for value in required:
        hi = np.searchsorted(frequency, value)
        bins.update((int(hi) - 1, int(hi)))
    valid = (pxx > max(float(pxx.max()) * 1e-4, 1e-20)) & (coherence >= .6)
    if any(not valid[index] for index in bins):
        raise ValueError("missing excitation or coherence at a required mask/normalization frequency")
    values = np.interp(points, frequency, gain) - np.interp(reference, frequency, gain)
    record.update(measured_db=values.tolist(), coherence=np.interp(points, frequency, coherence).tolist(),
                  fixed_delay_samples=delay_samples, absolute_gain_at_normalization_db=float(np.interp(reference, frequency, gain)),
                  worst_shape_margin_db=float(np.minimum(values - lower, upper - values).min()),
                  shape_mask_passed=bool(np.all((values >= lower) & (values <= upper))),
                  standard_compliance=False, calibration_id=None,
                  method="Welch H1, 250 ms Hann windows, 50% overlap; fixed delay only",
                  limitation="digital normalized transfer only; nonlinear speech adaptation is not a calibrated acoustic response")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--delay-samples", type=int, default=0)
    parser.add_argument("--start-s", type=float, default=0, help="explicit measurement settle interval; same input/output timeline")
    parser.add_argument("--noise-interval", type=float, nargs=2, metavar=("START_S", "END_S"), help="explicitly confirmed noise-only interval on input timeline")
    args = parser.parse_args()
    before, rate = sf.read(args.before, always_2d=True)
    after, output_rate = sf.read(args.after, always_2d=True)
    if before.shape[1] != 1 or after.shape[1] != 1 or rate != output_rate or not np.isfinite(args.start_s) or args.start_s < 0:
        parser.error("mono matching-rate files and finite nonnegative start required")
    start = round(args.start_s * rate)
    report = transfer_response(before[start:, 0], after[start:, 0], rate, json.loads(args.mask.read_text()), args.delay_samples)
    if args.noise_interval:
        lo, hi = args.noise_interval
        if not np.isfinite([lo, hi]).all() or lo < 0 or hi <= lo or hi > len(before) / rate or round(hi * rate) + args.delay_samples > len(after):
            parser.error("noise interval outside common measurement timeline")
        pcm = after[round(lo * rate) + args.delay_samples:round(hi * rate) + args.delay_samples, 0]
        report["noise_only"] = dict(interval_s=[lo, hi], windows=[noise_windows(pcm, rate, ms) for ms in (50, 100)])
    report.update(start_s=args.start_s, input_sha256=hashlib.sha256(args.before.read_bytes()).hexdigest(),
                  output_sha256=hashlib.sha256(args.after.read_bytes()).hexdigest(),
                  mask_sha256=hashlib.sha256(args.mask.read_bytes()).hexdigest())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps(dict(shape_mask_passed=report["shape_mask_passed"], standard_compliance=False, report=str(args.out))))


if __name__ == "__main__":
    main()
