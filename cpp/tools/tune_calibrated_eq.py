#!/usr/bin/env python3
"""Control-plane EQ fitting from a measured normalized response; never clean/SNR labels."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import freqz


def response_db(frequencies, centers, gains, rate, q=1.):
    result = np.zeros(len(frequencies))
    for center, gain in zip(centers, gains, strict=True):
        amplitude = 10 ** (gain / 40)
        w = 2 * np.pi * center / rate
        alpha = np.sin(w) / (2 * q)
        b = [1 + alpha * amplitude, -2 * np.cos(w), 1 - alpha * amplitude]
        a = [1 + alpha / amplitude, -2 * np.cos(w), 1 - alpha / amplitude]
        _, transfer = freqz(b, a, worN=2 * np.pi * np.asarray(frequencies) / rate)
        result += 20 * np.log10(np.maximum(np.abs(transfer), 1e-12))
    return result


def validate_response(record):
    rate = record["sample_rate"]
    if rate not in (16000, 24000, 48000) or record["scope"] not in ("digital_diagnostic", "acoustic_calibrated"):
        raise ValueError("supported rate and explicit measurement scope required")
    if record["scope"] == "acoustic_calibrated" and not (isinstance(record.get("calibration_id"), str) and record["calibration_id"].strip()):
        raise ValueError("acoustic curve requires calibration identity")
    if not all(isinstance(record.get(key), str) and record[key].strip() for key in ("measurement_id", "mask_source")):
        raise ValueError("measurement and mask source required")
    f, measured, lower, upper = [np.asarray(record[key], dtype=float) for key in
                                ("frequency_hz", "measured_db", "lower_db", "upper_db")]
    if (f.ndim != 1 or len(f) < 10 or any(v.shape != f.shape or not np.isfinite(v).all()
                                          for v in (measured, lower, upper)) or not np.isfinite(f).all()
            or np.any(np.diff(f) <= 0) or f[0] < 20 or f[-1] > .45 * rate or np.any(lower >= upper)):
        raise ValueError("finite ordered frequency curve with valid masks required")
    reference = record["normalized_at_hz"]
    if not f[0] <= reference <= f[-1] or abs(np.interp(np.log(reference), np.log(f), measured)) > .05:
        raise ValueError("measurement must already be normalized using its declared test method")
    return rate, f, measured, lower, upper, reference


def fit_response(record):
    rate, f, measured, lower, upper, reference = validate_response(record)
    centers = np.asarray([value for value in (120, 250, 500, 1000, 2000, 4000, 6000)
                          if f[0] <= value <= f[-1]])
    if not len(centers):
        raise ValueError("no supported EQ centers in curve")
    target = (lower + upper) / 2

    def shaped(gains):
        result = response_db(f, centers, gains, rate)
        return measured + result - response_db([reference], centers, gains, rate)[0]

    def residual(gains):
        return np.r_[shaped(gains) - target, .05 * gains]

    fit = least_squares(residual, np.zeros(len(centers)), bounds=(-6, 6), max_nfev=300)
    gains = np.round(fit.x, 3)
    after = shaped(gains)
    before_margin = float(np.min(np.minimum(measured - lower, upper - measured)))
    after_margin = float(np.min(np.minimum(after - lower, upper - after)))
    improved = fit.success and after_margin > before_margin + .01
    if not improved:
        gains = np.zeros(len(centers))
        after, after_margin = measured.copy(), before_margin
    preamp = -float(np.ceil(np.maximum(gains, 0).sum() * 1000) / 1000)
    if preamp < -24:
        gains = np.zeros(len(centers))
        preamp, improved, after, after_margin = 0., False, measured.copy(), before_margin
    bands = [dict(kind="peak", frequency_hz=int(center), gain_db=float(gain), q=1.)
             for center, gain in zip(centers, gains, strict=True) if abs(gain) >= .001]
    options = ["--eq-preamp-db", str(preamp)]
    for band in bands:
        options.extend(["--eq-band", f"peak,{band['frequency_hz']},{band['gain_db']},1"])
    return dict(sample_rate=rate, scope=record["scope"], measurement_id=record["measurement_id"],
                mask_source=record["mask_source"], calibration_id=record.get("calibration_id"),
                normalized_at_hz=reference, control_plane=True, online_adaptive_eq=False,
                preamp_db=preamp, bands=bands, cli_options=options, accepted_candidate=bool(improved),
                before_shape_margin_db=before_margin, after_shape_margin_db=after_margin,
                shape_mask_passed=after_margin >= 0, standard_compliance=False,
                frequency_hz=f.tolist(), after_normalized_db=after.tolist(),
                limitation="normalized response only; absolute SLR/RLR, peaks, speech quality and paced replay remain mandatory")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = fit_response(json.loads(args.response.read_text()))
    report["source_sha256"] = hashlib.sha256(args.response.read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("frequency_hz", "after_normalized_db")}, indent=2))


if __name__ == "__main__":
    main()
