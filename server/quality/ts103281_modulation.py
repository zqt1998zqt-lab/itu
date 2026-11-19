"""TS 103 281 Model A modulation features (clause 6.3.7.2).

Normative sources
-----------------
ETSI TS 103 281 V1.3.2, clause 6.3.7.2, Figure 6.10 and equations
(15)-(16). The public normative PDF is kept in the project's Model A
verification inputs.

Filter sources and licence
--------------------------
The second-order Falk 2010 modulation filter and the eight logarithmically
spaced centre frequencies are transcribed from the MIT-licensed SRMRToolbox:

* https://raw.githubusercontent.com/MuSAELab/SRMRToolbox/master/libs/makeModulationFilter.m
* https://raw.githubusercontent.com/MuSAELab/SRMRToolbox/master/libs/computeModulationCFs.m
* https://raw.githubusercontent.com/MuSAELab/SRMRToolbox/master/libs/modulationFilterBank.m

Copyright (c) 2014 Joao F. Santos, Tiago H. Falk. The toolbox is distributed
under the MIT licence; the licence text is available at:
https://raw.githubusercontent.com/MuSAELab/SRMRToolbox/master/LICENSE

Determined behaviour
--------------------
* SRMR forms a magnitude envelope before modulation filtering and applies a
  signed real second-order filter bank. It later forms windowed energy by
  squaring the signed filter output.
* TS 103 281 uses the continuous time axis, upsamples by four (125 Hz to
  500 Hz for the normative 8 ms frame step), aggregates with the 90th
  percentile over time, and selects the reference modulation maximum before
  forming MMER.
* The A-weighting is clipped at -10 dB and converted to a linear weight.

Documented uncertainty
----------------------
The TS 103 281 flow chart does not draw an explicit rectifier between the
real modulation filter and the 90th-percentile block, while the following
MMER branch explicitly uses magnitudes. This implementation therefore takes
``abs`` of each real filter output before the time P90. That placement is an
inference needed to satisfy the standard's magnitude semantics; it is not
stated as a separate block in the public flow chart.

The standard specifies a four-times time upsample but not its interpolation
kernel. This module uses SciPy's polyphase resampling, preserving a continuous
time axis without concatenating active frames.

No Annex C fitted constants are used. The only numerical floor is the IEEE-754
double machine epsilon, applied before logarithms so identical inputs remain
finite.
"""
from __future__ import annotations

from typing import Final

import numpy as np
from scipy import signal


MODULATION_CENTRES_HZ: Final[np.ndarray] = np.geomspace(4.0, 128.0, 8)
MODULATION_Q: Final[float] = 2.0
NOMINAL_MODULATION_SAMPLE_RATE_HZ: Final[float] = 500.0
TIME_UPSAMPLE_FACTOR: Final[int] = 4
A_WEIGHT_MINIMUM_DB: Final[float] = -10.0
NUMERIC_FLOOR: Final[float] = float(np.finfo(np.float64).eps)

SOURCE_URLS: Final[dict[str, str]] = {
    "normative_pdf": (
        "https://www.etsi.org/deliver/etsi_ts/103200_103299/"
        "103281/01.03.02_60/ts_103281v010302p.pdf"
    ),
    "srmr_filter": (
        "https://raw.githubusercontent.com/MuSAELab/SRMRToolbox/"
        "master/libs/makeModulationFilter.m"
    ),
    "srmr_centres": (
        "https://raw.githubusercontent.com/MuSAELab/SRMRToolbox/"
        "master/libs/computeModulationCFs.m"
    ),
    "srmr_bank": (
        "https://raw.githubusercontent.com/MuSAELab/SRMRToolbox/"
        "master/libs/modulationFilterBank.m"
    ),
    "srmr_license": (
        "https://raw.githubusercontent.com/MuSAELab/SRMRToolbox/"
        "master/LICENSE"
    ),
}

__all__ = [
    "A_WEIGHT_MINIMUM_DB",
    "MODULATION_CENTRES_HZ",
    "MODULATION_Q",
    "NOMINAL_MODULATION_SAMPLE_RATE_HZ",
    "NUMERIC_FLOOR",
    "SOURCE_URLS",
    "TIME_UPSAMPLE_FACTOR",
    "ModulationFeatureError",
    "modulation_features",
    "modulation_filter_bank",
    "modulation_filter_coefficients",
    "modulation_filter_response",
    "modulation_spectrum",
    "modified_a_weighting_linear",
    "upsample_modulation_time",
]


class ModulationFeatureError(ValueError):
    """The modulation-feature inputs violate the public contract."""


def _as_finite_vector(value: np.ndarray, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ModulationFeatureError(f"{name} must be numeric") from error
    if result.ndim != 1:
        raise ModulationFeatureError(f"{name} must be one-dimensional")
    if result.size == 0:
        raise ModulationFeatureError(f"{name} must not be empty")
    if not np.isfinite(result).all():
        raise ModulationFeatureError(f"{name} must contain only finite values")
    return result


def _as_nonnegative_matrix(value: np.ndarray, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ModulationFeatureError(f"{name} must be numeric") from error
    if result.ndim != 2:
        raise ModulationFeatureError(f"{name} must have shape (bands, time)")
    if 0 in result.shape:
        raise ModulationFeatureError(f"{name} must not be empty")
    if not np.isfinite(result).all():
        raise ModulationFeatureError(f"{name} must contain only finite values")
    if np.any(result < 0.0):
        raise ModulationFeatureError(f"{name} must be non-negative")
    return result


def _validate_frame_seconds(frame_seconds: float) -> float:
    try:
        value = float(frame_seconds)
    except (TypeError, ValueError) as error:
        raise ModulationFeatureError("frame_seconds must be numeric") from error
    if not np.isfinite(value) or value <= 0.0:
        raise ModulationFeatureError("frame_seconds must be finite and positive")
    modulation_rate = TIME_UPSAMPLE_FACTOR / value
    if modulation_rate <= 2.0 * float(np.max(MODULATION_CENTRES_HZ)):
        raise ModulationFeatureError(
            "frame_seconds produces a modulation rate at or below the "
            "128 Hz filter Nyquist limit"
        )
    return value


def _validate_auditory_centres(
    centres: np.ndarray,
    band_count: int,
) -> np.ndarray:
    result = _as_finite_vector(centres, "centres")
    if result.size != band_count:
        raise ModulationFeatureError(
            "centres must contain one frequency per auditory band"
        )
    if np.any(result <= 0.0):
        raise ModulationFeatureError("centres must be positive")
    return result


def modulation_filter_coefficients(
    centre_hz: float,
    sample_rate_hz: float = NOMINAL_MODULATION_SAMPLE_RATE_HZ,
    q: float = MODULATION_Q,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact SRMRToolbox second-order modulation filter.

    ``makeModulationFilter.m`` uses a bilinear transform with:
    ``W0 = tan(w0 / 2)``, ``B0 = W0 / Q``,
    ``b = [B0, 0, -B0]`` and
    ``a = [1+B0+W0^2, 2W0^2-2, 1-B0+W0^2]``.
    """

    try:
        centre = float(centre_hz)
        sample_rate = float(sample_rate_hz)
        quality = float(q)
    except (TypeError, ValueError) as error:
        raise ModulationFeatureError(
            "filter frequency, sample rate and Q must be numeric"
        ) from error
    if (
        not np.isfinite(centre)
        or not np.isfinite(sample_rate)
        or not np.isfinite(quality)
    ):
        raise ModulationFeatureError(
            "filter frequency, sample rate and Q must be finite"
        )
    if centre <= 0.0 or centre >= sample_rate / 2.0:
        raise ModulationFeatureError(
            "filter centre must be between DC and Nyquist"
        )
    if sample_rate <= 0.0 or quality <= 0.0:
        raise ModulationFeatureError(
            "sample rate and Q must be positive"
        )

    w0 = 2.0 * np.pi * centre / sample_rate
    w0_tangent = np.tan(w0 / 2.0)
    bandwidth = w0_tangent / quality
    numerator = np.array([bandwidth, 0.0, -bandwidth], dtype=np.float64)
    denominator = np.array(
        [
            1.0 + bandwidth + w0_tangent**2,
            2.0 * w0_tangent**2 - 2.0,
            1.0 - bandwidth + w0_tangent**2,
        ],
        dtype=np.float64,
    )
    numerator /= denominator[0]
    denominator /= denominator[0]
    return numerator, denominator


def modulation_filter_bank(
    centres_hz: np.ndarray = MODULATION_CENTRES_HZ,
    sample_rate_hz: float = NOMINAL_MODULATION_SAMPLE_RATE_HZ,
    q: float = MODULATION_Q,
) -> tuple[np.ndarray, np.ndarray]:
    """Return numerator and denominator rows for all modulation filters."""

    centres = _as_finite_vector(centres_hz, "modulation centres")
    if np.any(centres <= 0.0):
        raise ModulationFeatureError("modulation centres must be positive")
    numerators: list[np.ndarray] = []
    denominators: list[np.ndarray] = []
    for centre in centres:
        numerator, denominator = modulation_filter_coefficients(
            float(centre),
            sample_rate_hz,
            q,
        )
        numerators.append(numerator)
        denominators.append(denominator)
    return np.stack(numerators), np.stack(denominators)


def modulation_filter_response(
    centre_hz: float,
    frequencies_hz: np.ndarray,
    sample_rate_hz: float = NOMINAL_MODULATION_SAMPLE_RATE_HZ,
    q: float = MODULATION_Q,
) -> np.ndarray:
    """Evaluate one modulation filter at arbitrary non-negative frequencies."""

    frequencies = _as_finite_vector(frequencies_hz, "frequencies_hz")
    try:
        sample_rate = float(sample_rate_hz)
    except (TypeError, ValueError) as error:
        raise ModulationFeatureError("sample_rate_hz must be numeric") from error
    if not np.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ModulationFeatureError("sample_rate_hz must be finite and positive")
    if np.any(frequencies < 0.0) or np.any(frequencies > sample_rate / 2.0):
        raise ModulationFeatureError(
            "frequencies_hz must lie between DC and Nyquist"
        )
    numerator, denominator = modulation_filter_coefficients(
        centre_hz,
        sample_rate,
        q,
    )
    return signal.freqz(
        numerator,
        denominator,
        worN=frequencies,
        fs=sample_rate,
    )[1]


def modified_a_weighting_linear(frequencies_hz: np.ndarray) -> np.ndarray:
    """Return ``10 ** (max(A_dB, -10) / 20)`` for each frequency."""

    frequencies = _as_finite_vector(frequencies_hz, "frequencies_hz")
    if np.any(frequencies <= 0.0):
        raise ModulationFeatureError("frequencies_hz must be positive")
    frequency_squared = np.square(frequencies)
    numerator = (12_194.0**2) * np.square(frequency_squared)
    denominator = (
        (frequency_squared + 20.6**2)
        * np.sqrt(
            (frequency_squared + 107.7**2)
            * (frequency_squared + 737.9**2)
        )
        * (frequency_squared + 12_194.0**2)
    )
    a_weight_db = 2.0 + 20.0 * np.log10(
        np.maximum(numerator / denominator, NUMERIC_FLOOR)
    )
    clipped_db = np.maximum(a_weight_db, A_WEIGHT_MINIMUM_DB)
    return np.power(10.0, clipped_db / 20.0)


def upsample_modulation_time(
    values: np.ndarray,
    factor: int = TIME_UPSAMPLE_FACTOR,
) -> np.ndarray:
    """Upsample the final time axis while preserving continuous frame order."""

    matrix = _as_nonnegative_matrix(values, "values")
    if not isinstance(factor, (int, np.integer)) or int(factor) <= 0:
        raise ModulationFeatureError("factor must be a positive integer")
    return signal.resample_poly(
        matrix,
        up=int(factor),
        down=1,
        axis=-1,
    )


def _filtered_modulation_bands(
    values: np.ndarray,
    frame_seconds: float,
) -> np.ndarray:
    upsampled = upsample_modulation_time(values)
    sample_rate = TIME_UPSAMPLE_FACTOR / frame_seconds
    numerators, denominators = modulation_filter_bank(
        sample_rate_hz=sample_rate
    )
    return np.stack(
        [
            signal.lfilter(
                numerator,
                denominator,
                upsampled,
                axis=-1,
            )
            for numerator, denominator in zip(
                numerators,
                denominators,
                strict=True,
            )
        ],
        axis=1,
    )


def _p90_magnitude(filtered: np.ndarray) -> np.ndarray:
    return np.percentile(
        np.abs(filtered),
        90.0,
        axis=-1,
        method="linear",
    )


def modulation_spectrum(
    values: np.ndarray,
    centres: np.ndarray,
    frame_seconds: float = 0.008,
) -> np.ndarray:
    """Return the time-P90 modulation spectrum with shape ``(bands, 8)``."""

    matrix = _as_nonnegative_matrix(values, "values")
    _validate_auditory_centres(centres, matrix.shape[0])
    frame = _validate_frame_seconds(frame_seconds)
    return _p90_magnitude(_filtered_modulation_bands(matrix, frame))


def _to_db(linear: float) -> float:
    value = float(linear)
    if not np.isfinite(value):
        raise ModulationFeatureError("modulation feature overflowed")
    return float(20.0 * np.log10(max(value, NUMERIC_FLOOR)))


def modulation_features(
    reference: np.ndarray,
    speech: np.ndarray,
    centres: np.ndarray,
    frame_seconds: float = 0.008,
) -> dict[str, float]:
    """Compute TS 103 281 clause 6.3.7.2 DME and MMER.

    ``reference`` and ``speech`` are non-negative auditory-band spectra with
    shape ``(bands, time)``. ``centres`` contains the auditory-band centre
    frequencies in hertz. The time axis is resampled by four without removing
    or concatenating frames; P90 therefore addresses the continuous timeline.
    """

    reference_matrix = _as_nonnegative_matrix(reference, "reference")
    speech_matrix = _as_nonnegative_matrix(speech, "speech")
    if reference_matrix.shape != speech_matrix.shape:
        raise ModulationFeatureError(
            "reference and speech must have the same shape"
        )
    auditory_centres = _validate_auditory_centres(
        centres,
        reference_matrix.shape[0],
    )
    frame = _validate_frame_seconds(frame_seconds)

    reference_filtered = _filtered_modulation_bands(
        reference_matrix,
        frame,
    )
    speech_filtered = _filtered_modulation_bands(
        speech_matrix,
        frame,
    )
    reference_spectrum = _p90_magnitude(reference_filtered)
    speech_spectrum = _p90_magnitude(speech_filtered)
    difference_spectrum = _p90_magnitude(
        speech_filtered - reference_filtered
    )

    weights = modified_a_weighting_linear(auditory_centres)
    weighted_difference = difference_spectrum * weights[:, np.newaxis]
    dme_linear = float(
        np.linalg.norm(weighted_difference.reshape(-1), ord=2)
    )
    weighted_difference_db = _to_db(dme_linear)

    peak_modulation = np.argmax(reference_spectrum, axis=1)
    band_indices = np.arange(reference_spectrum.shape[0])
    reference_peak = reference_spectrum[band_indices, peak_modulation]
    speech_at_reference_peak = speech_spectrum[
        band_indices,
        peak_modulation,
    ]
    # A zero reference modulation band has no defined energy ratio. It must
    # not acquire an enormous ratio from dividing speech by machine epsilon.
    # Keep such bands out of the sum; an entirely unmodulated reference has
    # no MMER measurement and is explicitly rejected.
    valid = reference_peak > 0.0
    if not np.any(valid):
        raise ModulationFeatureError("MMER requires a nonzero reference modulation band")
    ratio = np.divide(
        speech_at_reference_peak, reference_peak,
        out=np.zeros_like(reference_peak), where=valid,
    )
    mmer_linear = float(np.sum(weights * ratio))
    mmer_db = _to_db(mmer_linear)

    return {
        "mmer": mmer_db,
        "weighted_difference": weighted_difference_db,
    }
