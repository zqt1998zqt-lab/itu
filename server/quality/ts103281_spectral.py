"""Preprocessing and spectral activity from TS 103 281 clauses 6.3.2--6.3.3.

The Byrne combined LTASS is transcribed from Table II of Byrne et al., JASA
96(4), 2116 (1994), DOI 10.1121/1.410152. The paper and table provenance are
retained under build/model-a-20261105/prior/. No MOS targets enter this code.

The public standard leaves the Welch window/FFT size and LTASS interpolation
kernel unspecified. These numerical choices are exposed in diagnostics;
they are not evidence that the complete Model A has been reproduced.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from .ts103281_activity import active_speech_level


LTASS_FREQUENCIES_HZ = np.array([
    63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
    1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
    10000, 12500, 16000,
], dtype=np.float64)
LTASS_COMBINED_DB_SPL = np.array([
    38.6, 43.5, 54.4, 57.7, 56.8, 60.2, 60.3, 59.0, 62.1, 62.1,
    60.5, 56.8, 53.7, 53.0, 52.0, 48.7, 48.1, 46.8, 45.6, 44.5,
    44.3, 43.7, 43.4, 41.3, 40.7,
], dtype=np.float64)


def calibrate_reference(
    reference: np.ndarray,
    degraded: np.ndarray,
    sample_active: np.ndarray,
    centres: np.ndarray,
    rate: int = 48_000,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Estimate the active-segment H1 and initial reference level gain.

    Returns the scaled time reference, the magnitude H1 on auditory centres
    relative to that scaled reference, and the initial calibration evidence.
    Normalizing H by the initial gain preserves H_original*X2 = H_scaled*X3.
    """
    x = np.asarray(reference, dtype=np.float64)
    y = np.asarray(degraded, dtype=np.float64)
    active = np.asarray(sample_active, dtype=bool)
    frequencies = np.asarray(centres, dtype=np.float64)
    if x.ndim != 1 or x.shape != y.shape or x.shape != active.shape:
        raise ValueError("H1 requires matching mono signals and sample activity")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("H1 signals must be finite")
    if frequencies.ndim != 1 or not np.isfinite(frequencies).all():
        raise ValueError("auditory centres must be a finite vector")
    x_active, y_active = x[active], y[active]
    if x_active.size < 2 or not np.any(x_active):
        raise ValueError("H1 requires nonzero active reference samples")
    segment_length = min(2048, len(x_active))
    frequency, power = signal.welch(
        x_active, fs=rate, nperseg=segment_length, detrend=False,
    )
    _, cross_power = signal.csd(
        x_active, y_active, fs=rate, nperseg=segment_length, detrend=False,
    )
    h1 = np.divide(
        cross_power, power, out=np.zeros_like(cross_power), where=power > 0,
    )
    magnitude = np.abs(h1)
    initial_gain = float(np.mean(magnitude))
    if not np.isfinite(initial_gain) or initial_gain <= 0:
        raise ValueError("H1 does not provide a positive finite reference gain")
    scaled = x * initial_gain
    scaled_asl = active_speech_level(scaled, rate)
    band_gain = np.interp(frequencies, frequency, magnitude) / initial_gain
    return scaled, band_gain, {
        "method": "active-segment CSD/PSD H1; mean magnitude over full frequency range",
        "initial_gain_linear": initial_gain,
        "initial_gain_db": float(20 * np.log10(initial_gain)),
        "scaled_reference_asl_db": scaled_asl,
        "h1_magnitude_on_scaled_reference": band_gain.tolist(),
        "active_samples": int(np.count_nonzero(active)),
        "numerical_assumptions": {
            "welch_window": "periodic Hann", "welch_segment_samples": segment_length,
            "welch_overlap": "50 percent", "detrend": False,
            "frequency_interpolation": "linear magnitude",
        },
    }


def ltass_activity_mask(
    reference_spectrum: np.ndarray,
    centres: np.ndarray,
    scaled_reference_asl_db: float,
) -> tuple[np.ndarray, dict]:
    """Classify spectral bins and apply the five Table 6.1 mask values.

    A common amplitude scale preserves the interpolated Byrne spectral shape.
    The 33 basic bands establish the total energy, avoiding a factor of three
    from the 66 overlapping intermediate analysis bands.
    """
    spectrum = np.asarray(reference_spectrum, dtype=np.float64)
    frequencies = np.asarray(centres, dtype=np.float64)
    if spectrum.ndim != 2 or spectrum.shape[0] != 99 or frequencies.shape != (99,):
        raise ValueError("spectral activity requires 99 bands and matching centres")
    if not np.isfinite(spectrum).all() or np.any(spectrum < 0):
        raise ValueError("reference spectrum must be finite and nonnegative")
    if not np.isfinite(frequencies).all() or np.any(frequencies <= 0):
        raise ValueError("centres must be finite and positive")
    if not np.isfinite(scaled_reference_asl_db):
        raise ValueError("scaled reference ASL must be finite")
    interpolated_db = np.interp(
        np.log(frequencies), np.log(LTASS_FREQUENCIES_HZ), LTASS_COMBINED_DB_SPL,
    )
    shape_power = 10 ** (interpolated_db / 10)
    basic_power = np.mean(shape_power.reshape(33, 3), axis=1)
    levels = interpolated_db + scaled_reference_asl_db - 10 * np.log10(np.sum(basic_power))
    relative = 20 * np.log10(np.maximum(spectrum, np.finfo(float).tiny)) - levels[:, None]
    classes = np.full(spectrum.shape, "S", dtype="<U1")
    mask = np.ones(spectrum.shape, dtype=np.float64)
    for threshold, label, weight in [(-35, "U", .4), (-16, "L", .2), (-10, "M", 0), (-1, "H", 0)]:
        selected = relative > threshold
        classes[selected], mask[selected] = label, weight
    unique, counts = np.unique(classes, return_counts=True)
    return mask, {
        "source": "Byrne et al. (1994) Table II combined universal LTASS",
        "class_counts": {str(k): int(v) for k, v in zip(unique, counts, strict=True)},
        "ltass_band_levels_db": levels.tolist(),
        "mask_values": {"S": 1.0, "U": .4, "L": .2, "M": 0.0, "H": 0.0},
        "numerical_assumptions": {
            "interpolation": "linear dB over log frequency; endpoint extension",
            "normalization": "one common gain; quadratic energy on 33 basic bands",
        },
    }


def correct_reference_gain(
    reference: np.ndarray,
    speech: np.ndarray,
    active: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Bisection from 0 dB/3 dB until about 30% of active bins lie above S."""
    x, s = np.asarray(reference, dtype=np.float64), np.asarray(speech, dtype=np.float64)
    selected = np.asarray(active, dtype=bool)
    if x.ndim != 2 or x.shape != s.shape or selected.shape != (x.shape[1],):
        raise ValueError("iterative gain requires matching spectra and time mask")
    if not np.isfinite(x).all() or not np.isfinite(s).all() or np.any(x < 0) or np.any(s < 0):
        raise ValueError("spectral amplitudes must be finite and nonnegative")
    if not selected.any() or not np.any(x[:, selected] > 0):
        raise ValueError("iterative gain requires active nonzero reference bins")
    x_active, s_active = x[:, selected], s[:, selected]
    # A finite positive gain cannot move a zero speech bin below a nonzero
    # reference bin, or make a zero reference bin exceed speech. Reject an
    # unreachable target instead of silently returning the +/-600 dB guard.
    minimum_proportion = float(np.mean((x_active > 0) & (s_active == 0)))
    maximum_proportion = float(np.mean(x_active > 0))
    if minimum_proportion > .30 or maximum_proportion < .30:
        raise ValueError("30 percent reference-gain target is unreachable")

    def proportion(gain_db: float) -> float:
        return float(np.mean(x_active * 10 ** (gain_db / 20) > s_active))

    target, initial_step = .30, 3.0
    lower = upper = 0.0
    ratio = proportion(0.0)
    iterations = 0
    if ratio < target:
        upper = initial_step
        while proportion(upper) < target and upper < 600:
            lower, upper = upper, upper + initial_step
            iterations += 1
    else:
        lower = -initial_step
        while proportion(lower) >= target and lower > -600:
            upper, lower = lower, lower - initial_step
            iterations += 1
    # Finite bisection tolerance is a numerical stop, not a fitted parameter.
    while upper - lower > 1e-7:
        midpoint = (lower + upper) / 2
        if proportion(midpoint) < target:
            lower = midpoint
        else:
            upper = midpoint
        iterations += 1
    # Prefer the side closest to 30% for a discrete/tied bin distribution.
    gain_db = min([lower, upper], key=lambda value: abs(proportion(value) - target))
    return x * 10 ** (gain_db / 20), {
        "method": "active-bin proportion bisection",
        "initial_gain_db": 0.0, "initial_step_db": initial_step,
        "target_proportion_above": target, "proportion_above": proportion(gain_db),
        "gain_db": gain_db, "iterations": iterations,
        "numerical_tolerance_db": 1e-7,
    }
