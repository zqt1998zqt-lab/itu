"""Engineering H1 transfer estimate for a virtual binaural receiver.

The estimator uses the reference as the excitation and each receiver ear as a
response.  It reports only bands that pass excitation, coverage, Welch-segment,
output-energy, and coherence gates.  The result is a digital engineering
measurement, not a calibrated acoustic or standard-compliance result.
"""
from __future__ import annotations

import math
from numbers import Real
from typing import Any, Sequence

import numpy as np
from scipy import signal


METHOD = (
    "Welch/CSD H1=Pxy/Pxx, excitation-power-weighted per one-third-octave "
    "band, binaural power mean"
)
MINIMUM_COHERENCE = 0.6
MINIMUM_SEGMENTS = 4
MINIMUM_BINS_PER_BAND = 2
MINIMUM_INPUT_FRAMES = 256
MAX_NPERSEG = 65_536
ABSOLUTE_POWER_FLOOR = 1e-12
RELATIVE_BAND_POWER_FLOOR = 1e-4
_ZERO_POWER = 1e-30
_THIRD_OCTAVE_FACTOR = 2 ** (1 / 6)


def _as_real_signal(values: Any, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a one-dimensional numeric array") from error
    if array.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional, got shape {array.shape}"
        )
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric samples, not bool or complex")
    if array.size < MINIMUM_INPUT_FRAMES:
        raise ValueError(
            f"{name} is too short: {array.size} samples, "
            f"minimum {MINIMUM_INPUT_FRAMES}"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        numeric = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} contains non-finite samples")
    return numeric


def _as_sample_rate(rate: Any) -> int:
    if isinstance(rate, (bool, np.bool_)) or not isinstance(rate, Real):
        raise ValueError("rate must be a positive finite integer-valued number")
    numeric = float(rate)
    if (
        not math.isfinite(numeric)
        or numeric <= 0
        or not numeric.is_integer()
    ):
        raise ValueError("rate must be a positive finite integer-valued number")
    return int(numeric)


def _as_centers(centers_hz: Sequence[float]) -> np.ndarray:
    try:
        centers = np.asarray(centers_hz)
    except (TypeError, ValueError) as error:
        raise ValueError("centers_hz must be a one-dimensional real sequence") from error
    if centers.ndim == 0:
        centers = centers.reshape(1)
    if centers.ndim != 1 or not len(centers):
        raise ValueError("centers_hz must be a non-empty one-dimensional sequence")
    if centers.dtype.kind not in "iuf":
        raise ValueError("centers_hz must contain real numeric values, not bool or complex")
    values = centers.astype(np.float64, copy=False)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("centers_hz values must be finite and positive")
    if len(np.unique(values)) != len(values):
        raise ValueError("centers_hz values must be unique")
    return np.sort(values)


def _as_upper_hz(upper_hz: Any) -> float:
    if isinstance(upper_hz, (bool, np.bool_)) or not isinstance(upper_hz, Real):
        raise ValueError("upper_hz must be a positive finite number")
    value = float(upper_hz)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("upper_hz must be a positive finite number")
    return value


def _center_key(center: float) -> str:
    return f"{center:.17g}"


def _next_power_of_two(value: int) -> int:
    return 1 << max(0, int(value) - 1).bit_length()


def _segment_count(frames: int, nperseg: int) -> int:
    noverlap = nperseg // 2
    step = nperseg - noverlap
    if frames < nperseg or step <= 0:
        return 0
    return 1 + (frames - nperseg) // step


def _band_reason(
    *,
    center: float,
    nyquist_hz: float,
    input_power: float | None,
    total_power: float,
    left_coherence: float | None,
    right_coherence: float | None,
    left_transfer_power: float | None,
    right_transfer_power: float | None,
    left_output_power: float | None,
    right_output_power: float | None,
    bin_count: int,
) -> str | None:
    upper = center * _THIRD_OCTAVE_FACTOR
    if upper > nyquist_hz + np.finfo(float).eps * nyquist_hz:
        return (
            f"one-third-octave upper edge {upper:.3f} Hz exceeds "
            f"Nyquist {nyquist_hz:.3f} Hz"
        )
    if bin_count < MINIMUM_BINS_PER_BAND:
        return f"band coverage has {bin_count} Welch bins; at least 2 required"
    if input_power is None or not math.isfinite(input_power):
        return "reference excitation energy is not finite"
    if input_power <= ABSOLUTE_POWER_FLOOR:
        return "reference excitation is below the absolute power floor"
    if total_power > ABSOLUTE_POWER_FLOOR and (
        input_power < total_power * RELATIVE_BAND_POWER_FLOOR
    ):
        return "reference excitation is below the relative band-power floor"
    if left_output_power is None or not math.isfinite(left_output_power):
        return "left output energy is not finite"
    if left_output_power <= ABSOLUTE_POWER_FLOOR:
        return "left output energy is below the absolute power floor"
    if right_output_power is None or not math.isfinite(right_output_power):
        return "right output energy is not finite"
    if right_output_power <= ABSOLUTE_POWER_FLOOR:
        return "right output energy is below the absolute power floor"
    if (
        left_transfer_power is None
        or not math.isfinite(left_transfer_power)
        or left_transfer_power / input_power <= ABSOLUTE_POWER_FLOOR
    ):
        return "left coherent output energy is not finite or measurable"
    if (
        right_transfer_power is None
        or not math.isfinite(right_transfer_power)
        or right_transfer_power / input_power <= ABSOLUTE_POWER_FLOOR
    ):
        return "right coherent output energy is not finite or measurable"
    if (
        left_coherence is None
        or not math.isfinite(left_coherence)
        or left_coherence < MINIMUM_COHERENCE
    ):
        return "left coherence is below 0.6"
    if (
        right_coherence is None
        or not math.isfinite(right_coherence)
        or right_coherence < MINIMUM_COHERENCE
    ):
        return "right coherence is below 0.6"
    return None


def _invalid_band(center: float, reason: str) -> dict[str, Any]:
    return {
        "input_power": None,
        "left_gain_db": None,
        "right_gain_db": None,
        "binaural_gain_db": None,
        "left_coherence": None,
        "right_coherence": None,
        "valid": False,
        "reason": reason,
    }


def _result(
    *,
    available: bool,
    reason: str,
    span_db: float | None,
    bands: dict[str, dict[str, Any]],
    required_bands_hz: list[float],
    missing_bands_hz: list[float],
    sample_rate_hz: int,
    frames: int,
) -> dict[str, Any]:
    return {
        "available": available,
        "reason": reason,
        "method": METHOD,
        "span_db": span_db,
        "bands": bands,
        "required_bands_hz": required_bands_hz,
        "missing_bands_hz": missing_bands_hz,
        "sample_rate_hz": sample_rate_hz,
        "frames": frames,
        "standard_compliance": False,
        "calibrated": False,
    }


def estimate_binaural_transfer(
    reference: Any,
    left: Any,
    right: Any,
    rate: Any,
    *,
    centers_hz: Sequence[float],
    upper_hz: Any,
) -> dict[str, Any]:
    """Estimate gated H1 gains for left, right, and their power mean.

    Arrays may differ slightly at the tail; all analysis uses their common
    prefix.  Invalid measurements return ``available=False`` and ``span_db``
    null.  Invalid bands never receive a placeholder 0 dB gain.
    """
    sample_rate = _as_sample_rate(rate)
    centers = _as_centers(centers_hz)
    upper = _as_upper_hz(upper_hz)
    required = centers[centers <= upper]
    if not len(required):
        raise ValueError("upper_hz must include at least one centers_hz value")

    excitation = _as_real_signal(reference, name="reference")
    left_signal = _as_real_signal(left, name="left")
    right_signal = _as_real_signal(right, name="right")
    frames = min(len(excitation), len(left_signal), len(right_signal))
    excitation = excitation[:frames]
    left_signal = left_signal[:frames]
    right_signal = right_signal[:frames]

    required_list = [float(value) for value in required]
    bands = {
        _center_key(center): _invalid_band(center, "not evaluated")
        for center in required_list
    }

    narrowest_width = required_list[0] / _THIRD_OCTAVE_FACTOR * (
        _THIRD_OCTAVE_FACTOR - 1 / _THIRD_OCTAVE_FACTOR
    )
    target_nperseg = math.ceil(
        sample_rate * MINIMUM_BINS_PER_BAND * 2 / narrowest_width
    )
    nperseg = min(
        frames,
        MAX_NPERSEG,
        max(MINIMUM_INPUT_FRAMES, _next_power_of_two(target_nperseg)),
    )
    segments = _segment_count(frames, nperseg)
    if segments < MINIMUM_SEGMENTS:
        reason = (
            f"transfer unavailable: {segments} Welch segments; "
            f"at least {MINIMUM_SEGMENTS} required"
        )
        for center in required_list:
            bands[_center_key(center)] = _invalid_band(center, reason)
        return _result(
            available=False,
            reason=reason,
            span_db=None,
            bands=bands,
            required_bands_hz=required_list,
            missing_bands_hz=required_list.copy(),
            sample_rate_hz=sample_rate,
            frames=frames,
        )

    options = {
        "fs": sample_rate,
        "nperseg": nperseg,
        "noverlap": nperseg // 2,
        "detrend": "constant",
        "scaling": "density",
        "average": "mean",
    }
    frequencies, pxx = signal.welch(excitation, **options)
    _, pyy_left = signal.welch(left_signal, **options)
    _, pyy_right = signal.welch(right_signal, **options)
    _, pxy_left = signal.csd(excitation, left_signal, **options)
    _, pxy_right = signal.csd(excitation, right_signal, **options)

    total_input_power = float(np.sum(pxx))
    spectra = (pxx, pyy_left, pyy_right, pxy_left, pxy_right)
    if not math.isfinite(total_input_power) or not all(
        np.all(np.isfinite(spectrum)) for spectrum in spectra
    ):
        reason = "transfer unavailable: spectral estimate is non-finite"
        for center in required_list:
            bands[_center_key(center)] = _invalid_band(center, reason)
        return _result(
            available=False,
            reason=reason,
            span_db=None,
            bands=bands,
            required_bands_hz=required_list,
            missing_bands_hz=required_list.copy(),
            sample_rate_hz=sample_rate,
            frames=frames,
        )
    nyquist_hz = sample_rate / 2
    missing: list[float] = []
    gains: list[float] = []
    for center in required_list:
        lower = center / _THIRD_OCTAVE_FACTOR
        upper_edge = center * _THIRD_OCTAVE_FACTOR
        mask = (frequencies >= lower) & (frequencies <= upper_edge)
        bin_count = int(np.count_nonzero(mask))
        input_power = float(np.sum(pxx[mask])) if bin_count else None
        left_output_power = float(np.sum(pyy_left[mask])) if bin_count else None
        right_output_power = float(np.sum(pyy_right[mask])) if bin_count else None

        left_transfer_power: float | None = None
        right_transfer_power: float | None = None
        left_coherence: float | None = None
        right_coherence: float | None = None
        if bin_count:
            band_pxx = pxx[mask]
            coherent_left = np.zeros_like(band_pxx)
            coherent_right = np.zeros_like(band_pxx)
            np.divide(
                np.abs(pxy_left[mask]) ** 2,
                band_pxx,
                out=coherent_left,
                where=band_pxx > _ZERO_POWER,
            )
            np.divide(
                np.abs(pxy_right[mask]) ** 2,
                band_pxx,
                out=coherent_right,
                where=band_pxx > _ZERO_POWER,
            )
            left_transfer_power = float(np.sum(coherent_left))
            right_transfer_power = float(np.sum(coherent_right))
            if left_output_power is not None:
                left_coherence = float(
                    np.clip(
                        left_transfer_power / max(left_output_power, _ZERO_POWER),
                        0.0,
                        1.0,
                    )
                )
            if right_output_power is not None:
                right_coherence = float(
                    np.clip(
                        right_transfer_power / max(right_output_power, _ZERO_POWER),
                        0.0,
                        1.0,
                    )
                )

        reason = _band_reason(
            center=center,
            nyquist_hz=nyquist_hz,
            input_power=input_power,
            total_power=total_input_power,
            left_coherence=left_coherence,
            right_coherence=right_coherence,
            left_transfer_power=left_transfer_power,
            right_transfer_power=right_transfer_power,
            left_output_power=left_output_power,
            right_output_power=right_output_power,
            bin_count=bin_count,
        )
        if reason is not None:
            band = _invalid_band(center, reason)
            # Retain measurable diagnostics without exposing untrusted gains.
            band["input_power"] = input_power
            band["left_coherence"] = left_coherence
            band["right_coherence"] = right_coherence
            bands[_center_key(center)] = band
            missing.append(center)
            continue

        assert input_power is not None
        assert left_transfer_power is not None
        assert right_transfer_power is not None
        left_gain = 10 * math.log10(
            max(left_transfer_power / max(input_power, _ZERO_POWER), _ZERO_POWER)
        )
        right_gain = 10 * math.log10(
            max(right_transfer_power / max(input_power, _ZERO_POWER), _ZERO_POWER)
        )
        binaural_gain = 10 * math.log10(
            max(
                (left_transfer_power + right_transfer_power)
                / 2
                / max(input_power, _ZERO_POWER),
                _ZERO_POWER,
            )
        )
        bands[_center_key(center)] = {
            "input_power": input_power,
            "left_gain_db": left_gain,
            "right_gain_db": right_gain,
            "binaural_gain_db": binaural_gain,
            "left_coherence": left_coherence,
            "right_coherence": right_coherence,
            "valid": True,
            "reason": "ok",
        }
        gains.append(binaural_gain)

    available = not missing
    if available:
        reason = "all required one-third-octave bands passed quality gates"
        span_db = float(max(gains) - min(gains))
    else:
        reason = (
            f"transfer unavailable: {len(missing)} of {len(required_list)} "
            "required bands failed quality gates"
        )
        span_db = None
    return _result(
        available=available,
        reason=reason,
        span_db=span_db,
        bands=bands,
        required_bands_hz=required_list,
        missing_bands_hz=missing,
        sample_rate_hz=sample_rate,
        frames=frames,
    )
