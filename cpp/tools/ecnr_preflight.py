#!/usr/bin/env python3
"""Digital ECNR input calibration inspired by PDF P14/P15; no ITU acoustic claim."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy import fft, signal

SCOPE = ("Digital calibration inputs only. Results do not certify ITU-T acoustic "
         "compliance, real vehicle bulk delay, or reference reliability during double talk.")


def array_sha256(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(str(arr.shape).encode("ascii") + arr.tobytes()).hexdigest()


def _vector(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or len(arr) < 256 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name}: require a finite 1-D vector with at least 256 samples")
    if float(np.std(arr)) < 1e-8:
        raise ValueError(f"{name}: silent or constant calibration signal")
    return arr


def _broadband_quality(reference: np.ndarray, fs: float) -> dict[str, Any]:
    freq, power = signal.welch(reference, fs=fs, nperseg=min(2048, len(reference)))
    power = np.maximum(power, np.finfo(float).tiny)
    active = power >= np.max(power) * 1e-3
    cdf = np.cumsum(power) / np.sum(power)
    low = float(freq[np.searchsorted(cdf, 0.05)])
    high = float(freq[min(len(freq)-1, np.searchsorted(cdf, 0.95))])
    bins = int(np.count_nonzero(active))
    return {"occupied_bins_above_minus30db": bins,
            "bandwidth_5_to_95_hz": high-low,
            "broadband": bins >= 24 and high-low >= 0.15*fs}


def estimate_delay(reference: np.ndarray, recorded: np.ndarray, sample_rate: float,
                   *, max_delay_ms: float = 80.0, interpolation: int = 32,
                   context: str = "unspecified", min_correlation: float = 0.8,
                   min_peak_ratio: float = 1.8) -> dict[str, Any]:
    """Positive delay means recorded lags reference; signed correlation detects polarity.

    FFT cross correlation uses zero padding and frequency-domain oversampling.
    GCC-PHAT is also reported as a diagnostic; truncated narrow spectral supports
    can make PHAT select edge artifacts, so the energy-weighted linear correlation
    supplies the measurement. Parabolic interpolation refines the local peak.
    Quality uses broadband support,
    a separated competing peak, and independently aligned normalized correlation.
    A declaration of digital calibration is necessary, but not alone sufficient.
    """
    ref = _vector(reference, "reference")
    rec = _vector(recorded, "recorded")
    if ref.shape != rec.shape or not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("equal vector lengths and positive sample rate required")
    if interpolation < 1 or interpolation > 128 or max_delay_ms <= 0:
        raise ValueError("interpolation must be 1..128; max delay must be positive")
    ref = ref - np.mean(ref)
    rec = rec - np.mean(rec)
    nfft = fft.next_fast_len(2*len(ref)-1)
    cross = fft.rfft(rec, nfft)*np.conj(fft.rfft(ref, nfft))
    magnitude = np.abs(cross)
    # Avoid amplifying empty/unsupported bins in band-limited digital calibration.
    support = magnitude > np.max(magnitude)*1e-6
    phat = np.zeros_like(cross)
    phat[support] = cross[support]/magnitude[support]
    oversampled = fft.irfft(cross, n=nfft*interpolation)
    phat_oversampled = fft.irfft(phat, n=nfft*interpolation)
    limit = min(int(max_delay_ms*sample_rate*interpolation/1000),
                len(ref)*interpolation//3)
    lags = np.arange(-limit, limit+1, dtype=float)/interpolation
    corr = np.concatenate((oversampled[-limit:], oversampled[:limit+1]))
    phat_corr = np.concatenate((phat_oversampled[-limit:], phat_oversampled[:limit+1]))
    phat_delay_samples = float(lags[int(np.argmax(np.abs(phat_corr)))])
    absolute = np.abs(corr)
    peak_index = int(np.argmax(absolute))
    delta = 0.0
    if 0 < peak_index < len(absolute)-1:
        a, b, c = absolute[peak_index-1:peak_index+2]
        denominator = a-2*b+c
        if abs(denominator) > 1e-15:
            delta = float(np.clip(0.5*(a-c)/denominator, -1, 1))
    delay_samples = float(lags[peak_index]+delta/interpolation)
    guard_samples = max(2.0, sample_rate*0.001)
    alternatives = absolute[np.abs(lags-delay_samples) > guard_samples]
    competitor = float(np.max(alternatives)) if len(alternatives) else 0.0
    peak_ratio = float(absolute[peak_index]/max(competitor, 1e-15))
    # Independent time-domain interpolation validates the peak rather than PHAT height.
    indices = np.arange(len(ref), dtype=float)
    overlap = (indices+delay_samples >= 0) & (indices+delay_samples <= len(rec)-1)
    aligned_ref = ref[overlap]
    aligned_rec = np.interp(indices[overlap]+delay_samples, indices, rec)
    aligned_ref -= np.mean(aligned_ref)
    aligned_rec -= np.mean(aligned_rec)
    correlation = float(np.dot(aligned_ref, aligned_rec) /
                        max(np.linalg.norm(aligned_ref)*np.linalg.norm(aligned_rec), 1e-15))
    linear_gain = float(np.dot(aligned_ref, aligned_rec) /
                        max(np.dot(aligned_ref, aligned_ref), 1e-15))
    quality = _broadband_quality(ref, sample_rate)
    reasons = []
    if context != "digital_calibration":
        reasons.append("context_is_not_declared_digital_calibration")
    if not quality["broadband"]:
        reasons.append("insufficient_broadband_support")
    if peak_ratio < min_peak_ratio:
        reasons.append("ambiguous_separated_delay_peaks")
    if abs(correlation) < min_correlation:
        reasons.append("weak_aligned_correlation_or_unmodelled_content")
    if correlation < 0:
        reasons.append("polarity_reversal")
    if peak_index in (0, len(absolute)-1):
        reasons.append("peak_at_search_boundary")
    return {"accepted": not reasons, "rejection_reasons": reasons,
            "delay_samples": delay_samples,
            "gcc_phat_diagnostic_delay_samples": phat_delay_samples,
            "measurement_method": "energy_weighted_fft_linear_correlation",
            "delay_ms": delay_samples*1000/sample_rate,
            "signed_aligned_correlation": correlation,
            "linear_gain": linear_gain, "separated_peak_ratio": peak_ratio,
            "quality": quality,
            "parameters": {"sample_rate_hz": sample_rate,
                           "max_delay_ms": max_delay_ms, "interpolation": interpolation,
                           "min_correlation": min_correlation, "min_peak_ratio": min_peak_ratio,
                           "context": context},
            "input_sha256": {"reference": array_sha256(reference),
                             "recorded": array_sha256(recorded)}}


def audit_bulk_delay(reference: np.ndarray, recordings: np.ndarray, sample_rate: float,
                     *, context: str = "unspecified", window_ms: float = 250.0,
                     max_delay_ms: float = 80.0, within_tolerance_ms: float = 0.10,
                     between_tolerance_ms: float = 0.10) -> dict[str, Any]:
    """Audit exactly seven reconnect-like digital records, not physical connections.

    Reports bulk delay, within-record span/slope, and between-record spread separately.
    A jump is rejected by span; the slope alone is not used to hide abrupt changes.
    """
    ref = _vector(reference, "reference")
    recs = np.asarray(recordings, dtype=float)
    if recs.ndim != 2 or recs.shape[1] != len(ref):
        raise ValueError("recordings must have shape (records, reference_samples)")
    nwin = int(round(window_ms*sample_rate/1000))
    if nwin < 256 or len(ref) < 3*nwin:
        raise ValueError("need at least three calibration windows per record")
    records = []
    all_delays = []
    for number, rec in enumerate(recs):
        windows = []
        for start in range(0, len(ref)-nwin+1, nwin):
            # Matching subwindows retain local delay; reject unsuitable windows.
            estimate = estimate_delay(ref[start:start+nwin], rec[start:start+nwin],
                                      sample_rate, context=context, max_delay_ms=max_delay_ms)
            estimate["start_sample"] = start
            estimate["center_time_s"] = (start+nwin/2)/sample_rate
            windows.append(estimate)
        delays = np.array([item["delay_ms"] for item in windows])
        times = np.array([item["center_time_s"] for item in windows])
        median = float(np.median(delays))
        span = float(np.ptp(delays))
        slope = float(np.polyfit(times, delays, 1)[0])
        qualified = all(item["accepted"] for item in windows)
        records.append({"record_index": number+1, "window_count": len(windows),
                        "median_bulk_delay_ms": median, "within_record_span_ms": span,
                        "within_record_linear_drift_ms_per_s": slope,
                        "quality_accepted": qualified,
                        "within_record_stable": qualified and span <= within_tolerance_ms,
                        "windows": windows, "recorded_sha256": array_sha256(rec)})
        all_delays.append(median)
    spread = float(np.ptp(all_delays)) if all_delays else 0.0
    reasons = []
    if len(recs) != 7:
        reasons.append("requires_exactly_seven_records")
    if any(not record["quality_accepted"] for record in records):
        reasons.append("one_or_more_windows_rejected")
    if any(not record["within_record_stable"] for record in records):
        reasons.append("within_record_delay_not_stable")
    if spread > between_tolerance_ms:
        reasons.append("between_record_delay_not_consistent")
    return {"accepted": not reasons, "rejection_reasons": reasons,
            "records": records, "record_count": len(recs),
            "median_bulk_delay_ms": float(np.median(all_delays)),
            "between_record_delay_spread_ms": spread,
            "parameters": {"sample_rate_hz": sample_rate, "window_ms": window_ms,
                           "max_delay_ms": max_delay_ms,
                           "within_tolerance_ms": within_tolerance_ms,
                           "between_tolerance_ms": between_tolerance_ms, "context": context},
            "reference_sha256": array_sha256(reference), "scope": SCOPE}


def audit_array_phase(mic1: np.ndarray, mic2: np.ndarray, sample_rate: float,
                      *, geometry: str, spacing_mm: float = 20.0,
                      sound_speed_m_s: float = 343.0, context: str = "unspecified",
                      phase_tolerance_deg: float = 5.0,
                      gain_tolerance_db: float = 1.0,
                      min_tone_to_residual_db: float = 20.0) -> dict[str, Any]:
    """Fit declared 1/2 kHz calibration tones, with residual-content rejection.

    angle 0 deg = endfire; angle 90 deg = broadside. Mic2 is farther from
    the source in endfire, giving negative phase of mic2 relative to mic1.
    Phase does not establish gain consistency; amplitudes are checked separately.
    """
    x = _vector(mic1, "mic1")
    y = _vector(mic2, "mic2")
    if x.shape != y.shape or sample_rate <= 4000:
        raise ValueError("equal vectors and sample rate above 4 kHz required")
    if geometry not in ("endfire", "broadside") or spacing_mm < 0 or sound_speed_m_s <= 0:
        raise ValueError("valid geometry, spacing, and sound speed required")
    frequencies = np.array([1000.0, 2000.0])
    t = np.arange(len(x))/sample_rate
    design = np.column_stack([np.ones(len(t)),
                              np.cos(2*np.pi*1000*t), np.sin(2*np.pi*1000*t),
                              np.cos(2*np.pi*2000*t), np.sin(2*np.pi*2000*t)])
    fits = []
    tone_ratios = []
    for values in (x, y):
        coefficients, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
        fitted = design@coefficients
        residual = values-fitted
        tone_power = float(np.var(fitted))
        residual_power = float(np.mean(residual**2))
        tone_ratios.append(float(10*np.log10(max(tone_power, 1e-30)/
                                            max(residual_power, 1e-30))))
        fits.append(np.array([coefficients[1]-1j*coefficients[2],
                              coefficients[3]-1j*coefficients[4]]))
    expected_delay_s = spacing_mm/1000/sound_speed_m_s if geometry == "endfire" else 0.0
    expected = -360*frequencies*expected_delay_s
    reasons = []
    if context != "digital_calibration":
        reasons.append("context_is_not_declared_digital_calibration")
    if min(tone_ratios) < min_tone_to_residual_db:
        reasons.append("excess_residual_content_or_unmodelled_signal")
    tones = []
    for i, frequency in enumerate(frequencies):
        available = abs(fits[0][i]) > 1e-5 and abs(fits[1][i]) > 1e-5
        if not available:
            reasons.append(f"missing_{int(frequency)}hz_calibration_tone")
        transfer = fits[1][i]/fits[0][i] if available else 0j
        measured = float(np.angle(transfer, deg=True))
        error = float((measured-expected[i]+180)%360-180)
        gain_db = float(20*np.log10(max(abs(transfer), 1e-30)))
        tones.append({"frequency_hz": float(frequency),
                      "expected_phase_deg": float(expected[i]),
                      "measured_phase_deg": measured, "phase_error_deg": error,
                      "mic2_relative_gain_db": gain_db, "tone_present": bool(available)})
    polarity_reversed = all(abs(abs(tone["phase_error_deg"])-180) <= phase_tolerance_deg
                            for tone in tones)
    if polarity_reversed:
        reasons.append("polarity_reversal")
    elif any(abs(tone["phase_error_deg"]) > phase_tolerance_deg for tone in tones):
        reasons.append("phase_does_not_match_declared_geometry")
    if any(abs(tone["mic2_relative_gain_db"]) > gain_tolerance_db for tone in tones):
        reasons.append("gain_mismatch")
    return {"accepted": not reasons, "rejection_reasons": reasons,
            "polarity_reversed": polarity_reversed, "tones": tones,
            "tone_to_residual_db": tone_ratios,
            "parameters": {"sample_rate_hz": sample_rate, "geometry": geometry,
                           "spacing_mm": spacing_mm, "sound_speed_m_s": sound_speed_m_s,
                           "expected_delay_ms": expected_delay_s*1000,
                           "phase_tolerance_deg": phase_tolerance_deg,
                           "gain_tolerance_db": gain_tolerance_db,
                           "min_tone_to_residual_db": min_tone_to_residual_db,
                           "context": context},
            "input_sha256": {"mic1": array_sha256(mic1), "mic2": array_sha256(mic2)},
            "scope": SCOPE}


def _metadata() -> dict[str, Any]:
    return {"schema_version": 1, "scope": SCOPE, "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "algorithm": {"delay": "zero-padded FFT linear correlation, 32x frequency interpolation, "
                                  "parabolic peak and time-domain correlation check; "
                                  "GCC-PHAT diagnostic also reported",
                          "bulk": "within-record window median/span/slope separated from "
                                  "seven-record median spread",
                          "array": "joint DC/1kHz/2kHz sine-cosine least squares, "
                                   "circular phase error and independent gain check"},
            "threshold_note": "These are configurable digital calibration quality "
                              "tolerances, not existing simulation acceptance thresholds "
                              "or ITU-T limits."}


def _save_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False)+"\n",
                    encoding="utf-8")


def run_demo(output: Path) -> dict[str, Any]:
    """Generate paired normal/negative controls, retain all numeric inputs and hashes."""
    output.mkdir(parents=True, exist_ok=True)
    fs = 16000
    seconds = 1.25
    t = np.arange(int(seconds*fs))/fs
    rng = np.random.default_rng(20261011)
    frequencies = rng.uniform(300, 6000, 180)
    phases = rng.uniform(-np.pi, np.pi, len(frequencies))
    def source(time: np.ndarray) -> np.ndarray:
        return np.sum(np.sin(2*np.pi*frequencies[:, None]*time+phases[:, None]),
                      axis=0)*0.012
    reference = source(t)
    delays_ms = 13.372+np.array([-0.010, 0.0, 0.005, -0.005, 0.010, 0.0, 0.005])
    normal = np.stack([source(t-delay/1000) for delay in delays_ms])
    jumping = normal.copy()
    jumping[3, len(t)//2:] = source(t[len(t)//2:]-(delays_ms[3]+4)/1000)
    cases = []
    for name, recordings, expected in [("bulk_normal", normal, True),
                                       ("bulk_delay_jump", jumping, False)]:
        fixture = output/f"{name}.npz"
        np.savez_compressed(fixture, reference=reference, recordings=recordings,
                            sample_rate=fs, true_record_delays_ms=delays_ms,
                            injected_jump_ms=4 if not expected else 0,
                            jump_record_index=4, jump_time_s=seconds/2)
        result = audit_bulk_delay(reference, recordings, fs, context="digital_calibration")
        result.update({"fixture": str(fixture),
                       "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest()})
        _save_json(output/f"{name}.json", {**_metadata(), "result": result})
        cases.append({"case": name, "expected_accepted": expected,
                      "actual_accepted": result["accepted"],
                      "expectation_met": result["accepted"] == expected,
                      "rejection_reasons": result["rejection_reasons"]})
    tone_t = np.arange(fs)/fs
    def tone_source(time: np.ndarray) -> np.ndarray:
        return 0.15*np.sin(2*np.pi*1000*time+0.3)+0.12*np.sin(2*np.pi*2000*time-0.4)
    mic1 = tone_source(tone_t)
    for name, geometry, polarity, gain, expected in [
        ("array_endfire_normal", "endfire", 1, 1.0, True),
        ("array_broadside_normal", "broadside", 1, 1.0, True),
        ("array_polarity_reversed", "endfire", -1, 1.0, False),
        ("array_gain_mismatch", "endfire", 1, 2.0, False),
    ]:
        delay_s = 0.020/343 if geometry == "endfire" else 0.0
        mic2 = polarity*gain*tone_source(tone_t-delay_s)
        fixture = output/f"{name}.npz"
        np.savez_compressed(fixture, mic1=mic1, mic2=mic2, sample_rate=fs,
                            spacing_mm=20.0, sound_speed_m_s=343.0,
                            true_delay_ms=delay_s*1000, polarity=polarity, gain=gain,
                            geometry=geometry)
        result = audit_array_phase(mic1, mic2, fs, geometry=geometry,
                                  context="digital_calibration")
        result.update({"fixture": str(fixture),
                       "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest()})
        _save_json(output/f"{name}.json", {**_metadata(), "result": result})
        cases.append({"case": name, "expected_accepted": expected,
                      "actual_accepted": result["accepted"],
                      "expectation_met": result["accepted"] == expected,
                      "rejection_reasons": result["rejection_reasons"]})
    summary = {**_metadata(), "input_generation": {"seed": 20261011, "sample_rate_hz": fs,
               "bulk_duration_s": seconds, "broadband_tones": len(frequencies),
               "bulk_base_delay_ms": 13.372,
               "bulk_record_delays_ms": delays_ms.tolist(),
               "negative_jump_ms": 4.0, "array_spacing_mm": 20.0,
               "array_tones_hz": [1000, 2000], "negative_polarity": -1,
               "negative_gain": 2.0},
               "all_expectations_met": all(case["expectation_met"] for case in cases),
               "cases": cases}
    _save_json(output/"summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="paired digital calibration/negative controls")
    demo.add_argument("--output", type=Path, required=True)
    bulk = sub.add_parser("bulk", help="NPZ: reference, recordings, sample_rate")
    bulk.add_argument("input", type=Path)
    bulk.add_argument("--output", type=Path, required=True)
    bulk.add_argument("--digital-calibration", action="store_true")
    bulk.add_argument("--within-tolerance-ms", type=float, default=0.10)
    bulk.add_argument("--between-tolerance-ms", type=float, default=0.10)
    array = sub.add_parser("array", help="NPZ: mic1, mic2, sample_rate")
    array.add_argument("input", type=Path)
    array.add_argument("--output", type=Path, required=True)
    array.add_argument("--digital-calibration", action="store_true")
    array.add_argument("--geometry", choices=["endfire", "broadside"], required=True)
    array.add_argument("--spacing-mm", type=float, default=20.0)
    args = parser.parse_args()
    if args.command == "demo":
        result = run_demo(args.output)
        print(json.dumps({"summary": str(args.output/"summary.json"),
                          "all_expectations_met": result["all_expectations_met"],
                          "cases": result["cases"]}, sort_keys=True))
        return 0 if result["all_expectations_met"] else 1
    with np.load(args.input, allow_pickle=False) as data:
        context = "digital_calibration" if args.digital_calibration else "unspecified"
        if args.command == "bulk":
            result = audit_bulk_delay(data["reference"], data["recordings"],
                                      float(data["sample_rate"]), context=context,
                                      within_tolerance_ms=args.within_tolerance_ms,
                                      between_tolerance_ms=args.between_tolerance_ms)
        else:
            result = audit_array_phase(data["mic1"], data["mic2"], float(data["sample_rate"]),
                                       context=context, geometry=args.geometry,
                                       spacing_mm=args.spacing_mm)
    report = {**_metadata(), "source_path": str(args.input),
              "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
              "result": result}
    _save_json(args.output, report)
    print(json.dumps({"accepted": result["accepted"],
                      "rejection_reasons": result["rejection_reasons"],
                      "report": str(args.output)}))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
