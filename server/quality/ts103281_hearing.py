"""Auditable TS 103 281 Model A hearing front-end (clause 6.3.3).

Normative signal path
---------------------
ETSI TS 103 281 V1.3.2 clause 6.3.3 specifies outer/middle-ear filtering,
an auditory filter bank, half-wave rectification, 3rd-order low-pass
filtering and downsampling to 8 ms. Equation (4) sets the auditory bandwidth
to ``50 + 0.14 * F`` Hz. Sottek (2016), equations (14)-(17), defines the
order-5 modulated low-pass filters:

``hT,F(t) = 1/(n-1)! * (1/tau) * (t/tau)^(n-1) * exp(-t/tau)``

``tau = comb(2*n-2, n-1) / (2^(2*n-1) * bandwidth)``

and recommends heterodyning, complex low-pass filtering, and shifting back.
The paper specifies half-wave rectification and a 3rd-order low-pass with a
120 Hz cutoff.

Exact, approximate and externally supplied parts
------------------------------------------------
Exact from the public TS/Sottek text:

* auditory bandwidth ``50 + 0.14 F``;
* order ``n = 5`` and the tau equation;
* heterodyne/complex-low-pass/up-shift band-pass construction;
* half-wave rectification and 120 Hz, 3rd-order low-pass cutoff;
* 8 ms output framing.

Externally supplied by contract:

* the 99 auditory centre frequencies. This module intentionally does not
  generate a centre grid.

Approximate:

* The public paper gives the outer/middle-ear chain as Zwicker's ``a0``
  characteristic followed by a 1st-order high-pass with 6.5 dB/octave
  attenuation below 500 Hz, but does not publish time-domain coefficients.
  This module uses the ISO 532-1 Zwicker ``a0`` transmission table exposed by
  MIT-licensed Phonometry, applies a 500 Hz first-order high-pass, normalizes
  at 1 kHz, and designs a causal minimum-phase FIR.
* The paper does not name the 3rd-order 120 Hz low-pass polynomial. This
  module uses a Butterworth 3rd-order section.
* The paper specifies downsampling but not the exact sample phase inside each
  8 ms frame. This module selects the frame-centre sample.
* A repeated first-order pole is the discrete implementation of the stated
  continuous low-pass. The pole placement uses ``exp(-1/(fs*tau))``.

The newer ECMA-418-2 outer/middle/inner-ear biquads are deliberately not used:
they differ from the Zwicker ``a0`` chain requested for Sottek 1993/2016.
If a caller substitutes an IEC A-weighting frequency formula for the ear
transfer, the traditional ``R_A`` form requires the +2 dB 1 kHz offset.

Sources retained under ``build/model-a-20261105/hearing-sources``
----------------------------------------------------------------
* Sottek, "A Hearing Model Approach to Time-Varying Loudness", Acta Acustica
  102 (2016), pp. 725-744, DOI 10.3813/AAA.918989.
* Phonometry 3.3.0, MIT License, Copyright (c) 2020-2026 Jose Manuel Requena
  Plens. Its ISO 532-1 ``A0_TRANSMISSION`` table and ECMA-418 implementation
  are retained for comparison; no GPL code is used.

No Annex C fitted constants are used.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Final

import numpy as np
from scipy import signal


SAMPLE_RATE_HZ: Final[float] = 48_000.0
FRAME_SECONDS: Final[float] = 0.008
AUDITORY_BAND_COUNT: Final[int] = 99
AUDITORY_FILTER_ORDER: Final[int] = 5
AUDITORY_BANDWIDTH_BASE_HZ: Final[float] = 50.0
AUDITORY_BANDWIDTH_SLOPE: Final[float] = 0.14
ENVELOPE_LOWPASS_HZ: Final[float] = 120.0
ENVELOPE_LOWPASS_ORDER: Final[int] = 3
EAR_HIGHPASS_HZ: Final[float] = 500.0
EAR_FIR_LINEAR_TAPS: Final[int] = 2049
EAR_FIR_N_FFT: Final[int] = 65_536
IEC_A_WEIGHTING_1KHZ_OFFSET_DB: Final[float] = 2.0
OUTER_MIDDLE_EAR_METHOD: Final[str] = (
    "zwicker_a0_plus_500hz_first_order_highpass_minimum_phase_fir"
)

# ISO 532-1 Zwicker a0 transmission corrections, exposed by the MIT
# Phonometry package. These are attenuations in dB; the linear ear response
# therefore uses their negative value. The first three entries correspond to
# the three low-frequency critical-band groups and are zero.
ZWICKER_A0_FREQUENCIES_HZ: Final[np.ndarray] = np.array(
    [
        25.0,
        100.0,
        200.0,
        315.0,
        400.0,
        500.0,
        630.0,
        800.0,
        1_000.0,
        1_250.0,
        1_600.0,
        2_000.0,
        2_500.0,
        3_150.0,
        4_000.0,
        5_000.0,
        6_300.0,
        8_000.0,
        10_000.0,
        12_500.0,
    ],
    dtype=np.float64,
)
ZWICKER_A0_TRANSMISSION_DB: Final[np.ndarray] = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.5,
        -1.6,
        -3.2,
        -5.4,
        -5.6,
        -4.0,
        -1.5,
        2.0,
        5.0,
        12.0,
    ],
    dtype=np.float64,
)

SOURCE_URLS: Final[dict[str, str]] = {
    "ts_103_281_v1_3_2": (
        "https://www.etsi.org/deliver/etsi_ts/103200_103299/"
        "103281/01.03.02_60/ts_103281v010302p.pdf"
    ),
    "sottek_2016_doi": "https://doi.org/10.3813/AAA.918989",
    "phonometry_source": "https://github.com/jmrplens/phonometry",
    "phonometry_ecma_frontend": (
        "https://github.com/jmrplens/phonometry/blob/main/src/"
        "phonometry/psychoacoustics/loudness/ecma.py"
    ),
    "phonometry_zwicker_a0": (
        "https://github.com/jmrplens/phonometry/blob/main/src/"
        "phonometry/psychoacoustics/loudness/_zwicker_data.py"
    ),
    "phonometry_license": (
        "https://github.com/jmrplens/phonometry/blob/main/LICENSE"
    ),
}

SOURCE_SHA256: Final[dict[str, str]] = {
    "sottek-2016-hearing-model.pdf": (
        "c644976b2a982d1928b92d26485f1d0c99bd8fab85c60e2d02d1438593c550e4"
    ),
    "phonometry-main.tar.gz": (
        "38182b30249f26d9b4d1e83a508808c02d87536b5b88770b10852a7f4f2381d1"
    ),
    "phonometry-LICENSE": (
        "b3b68372e2d85938ac48fe7ee0e8a3be9326adfc73c0fb01b25231118441d06b"
    ),
}

__all__ = [
    "AUDITORY_BANDWIDTH_BASE_HZ",
    "AUDITORY_BANDWIDTH_SLOPE",
    "AUDITORY_BAND_COUNT",
    "AUDITORY_FILTER_ORDER",
    "EAR_HIGHPASS_HZ",
    "ENVELOPE_LOWPASS_HZ",
    "ENVELOPE_LOWPASS_ORDER",
    "FRAME_SECONDS",
    "IEC_A_WEIGHTING_1KHZ_OFFSET_DB",
    "OUTER_MIDDLE_EAR_METHOD",
    "SAMPLE_RATE_HZ",
    "SOURCE_SHA256",
    "SOURCE_URLS",
    "ZWICKER_A0_FREQUENCIES_HZ",
    "ZWICKER_A0_TRANSMISSION_DB",
    "AuditoryFilterParameters",
    "HearingFrontendError",
    "apply_outer_middle_ear_filter",
    "auditory_bandpass_signal",
    "auditory_bandwidth",
    "auditory_filter_parameters",
    "auditory_filter_response",
    "auditory_spectrum",
    "auditory_time_constant",
    "iec_a_weighting_linear",
    "outer_middle_ear_filter_response",
    "outer_middle_ear_response",
]


class HearingFrontendError(ValueError):
    """The hearing-front-end input or configuration is invalid."""


@dataclass(frozen=True)
class AuditoryFilterParameters:
    centre_hz: float
    bandwidth_hz: float
    tau_seconds: float
    pole: float


def _positive_finite_scalar(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise HearingFrontendError(f"{name} must be numeric") from error
    if not np.isfinite(result) or result <= 0.0:
        raise HearingFrontendError(f"{name} must be finite and positive")
    return result


def _as_signal(samples: np.ndarray, name: str = "samples") -> np.ndarray:
    raw = np.asarray(samples)
    if np.iscomplexobj(raw):
        raise HearingFrontendError(f"{name} must be a real mono signal")
    try:
        result = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise HearingFrontendError(f"{name} must be numeric") from error
    if result.ndim != 1:
        raise HearingFrontendError(f"{name} must be one-dimensional")
    if result.size == 0:
        raise HearingFrontendError(f"{name} must not be empty")
    if not np.isfinite(result).all():
        raise HearingFrontendError(f"{name} must contain only finite samples")
    return result


def _as_frequency_vector(
    frequencies_hz: np.ndarray,
    name: str = "frequencies_hz",
) -> np.ndarray:
    raw = np.asarray(frequencies_hz)
    if np.iscomplexobj(raw):
        raise HearingFrontendError(f"{name} must be real")
    try:
        result = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise HearingFrontendError(f"{name} must be numeric") from error
    if result.ndim != 1:
        raise HearingFrontendError(f"{name} must be one-dimensional")
    if result.size == 0:
        raise HearingFrontendError(f"{name} must not be empty")
    if not np.isfinite(result).all():
        raise HearingFrontendError(f"{name} must contain only finite values")
    if np.any(result < 0.0):
        raise HearingFrontendError(f"{name} must be non-negative")
    return result


def _validate_centres(
    centres: np.ndarray,
    sample_rate_hz: float,
) -> np.ndarray:
    raw = np.asarray(centres)
    if np.iscomplexobj(raw):
        raise HearingFrontendError("centres must be real")
    try:
        result = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise HearingFrontendError("centres must be numeric") from error
    if result.ndim != 1:
        raise HearingFrontendError("centres must be one-dimensional")
    if result.size != AUDITORY_BAND_COUNT:
        raise HearingFrontendError(
            f"centres must contain exactly {AUDITORY_BAND_COUNT} values"
        )
    if not np.isfinite(result).all():
        raise HearingFrontendError("centres must contain only finite values")
    if np.any(result <= 0.0):
        raise HearingFrontendError("centres must be positive")
    if np.any(result >= sample_rate_hz / 2.0):
        raise HearingFrontendError(
            "centres must lie below the Nyquist frequency"
        )
    return result


def auditory_bandwidth(centre_hz: float) -> float:
    """TS 103 281 equation (4): ``50 + 0.14 * centre_hz`` Hz."""

    centre = _positive_finite_scalar(centre_hz, "centre_hz")
    return AUDITORY_BANDWIDTH_BASE_HZ + AUDITORY_BANDWIDTH_SLOPE * centre


def auditory_time_constant(centre_hz: float) -> float:
    """Sottek equation (17) for an order-5 auditory filter."""

    bandwidth = auditory_bandwidth(centre_hz)
    combinations = math.comb(
        2 * AUDITORY_FILTER_ORDER - 2,
        AUDITORY_FILTER_ORDER - 1,
    )
    return combinations / (
        2 ** (2 * AUDITORY_FILTER_ORDER - 1) * bandwidth
    )


def auditory_filter_parameters(
    centre_hz: float,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
) -> AuditoryFilterParameters:
    """Return bandwidth, tau and the stable repeated first-order pole."""

    centre = _positive_finite_scalar(centre_hz, "centre_hz")
    sample_rate = _positive_finite_scalar(sample_rate_hz, "sample_rate_hz")
    if centre >= sample_rate / 2.0:
        raise HearingFrontendError("centre_hz must be below Nyquist")
    bandwidth = auditory_bandwidth(centre)
    tau = auditory_time_constant(centre)
    pole = math.exp(-1.0 / (sample_rate * tau))
    if not 0.0 < pole < 1.0:
        raise HearingFrontendError("auditory-filter pole is not stable")
    return AuditoryFilterParameters(centre, bandwidth, tau, pole)


def auditory_filter_response(
    centre_hz: float,
    frequencies_hz: np.ndarray,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
) -> np.ndarray:
    """Complex base-band response of the order-5 auditory low-pass."""

    sample_rate = _positive_finite_scalar(sample_rate_hz, "sample_rate_hz")
    frequencies = _as_frequency_vector(frequencies_hz)
    if np.any(frequencies >= sample_rate / 2.0):
        raise HearingFrontendError(
            "frequencies_hz must lie below the Nyquist frequency"
        )
    parameters = auditory_filter_parameters(centre_hz, sample_rate)
    relative = frequencies - parameters.centre_hz
    delayed = np.exp(-2j * np.pi * relative / sample_rate)
    return (
        (1.0 - parameters.pole)
        / (1.0 - parameters.pole * delayed)
    ) ** AUDITORY_FILTER_ORDER


def auditory_bandpass_signal(
    samples: np.ndarray,
    centre_hz: float,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
) -> np.ndarray:
    """Apply the heterodyne order-5 auditory band-pass filter."""

    signal_values = _as_signal(samples)
    sample_rate = _positive_finite_scalar(sample_rate_hz, "sample_rate_hz")
    parameters = auditory_filter_parameters(centre_hz, sample_rate)
    time = np.arange(signal_values.size, dtype=np.float64)
    carrier = np.exp(
        -2j * np.pi * parameters.centre_hz * time / sample_rate
    )
    baseband = signal_values * carrier
    one_pole = np.array(
        [1.0 - parameters.pole],
        dtype=np.float64,
    )
    pole = np.array([1.0, -parameters.pole], dtype=np.float64)
    for _ in range(AUDITORY_FILTER_ORDER):
        baseband = signal.lfilter(one_pole, pole, baseband)
    shifted = 2.0 * np.real(baseband * np.conj(carrier))
    if not np.isfinite(shifted).all():
        raise HearingFrontendError("auditory band-pass produced non-finite output")
    return np.asarray(shifted, dtype=np.float64)


def _a0_transmission_db(frequencies_hz: np.ndarray) -> np.ndarray:
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    positive = frequencies > 0.0
    result = np.full(frequencies.shape, -200.0, dtype=np.float64)
    result[positive] = np.interp(
        np.log(frequencies[positive]),
        np.log(ZWICKER_A0_FREQUENCIES_HZ),
        ZWICKER_A0_TRANSMISSION_DB,
    )
    return result


def outer_middle_ear_response(frequencies_hz: np.ndarray) -> np.ndarray:
    """Approximate Sottek 1993/2016 outer/middle-ear linear response.

    The response combines the negated Zwicker ``a0`` attenuation table with a
    first-order 500 Hz high-pass, then normalizes at 1 kHz. This is an
    approximate response design, not a recovered original coefficient set.
    """

    frequencies = _as_frequency_vector(frequencies_hz)
    transmission_db = -_a0_transmission_db(frequencies)
    ratio = frequencies / EAR_HIGHPASS_HZ
    highpass = ratio / np.sqrt(1.0 + ratio**2)
    response = np.power(10.0, transmission_db / 20.0) * highpass
    normalization_db = -_a0_transmission_db(
        np.array([1_000.0], dtype=np.float64)
    )[0]
    normalization_highpass = (
        (1_000.0 / EAR_HIGHPASS_HZ)
        / np.sqrt(1.0 + (1_000.0 / EAR_HIGHPASS_HZ) ** 2)
    )
    normalization = (
        np.power(10.0, normalization_db / 20.0)
        * normalization_highpass
    )
    return response / normalization


def iec_a_weighting_linear(frequencies_hz: np.ndarray) -> np.ndarray:
    """Diagnostic IEC A weighting with the required +2 dB 1 kHz offset."""

    frequencies = _as_frequency_vector(frequencies_hz)
    if np.any(frequencies <= 0.0):
        raise HearingFrontendError(
            "IEC A weighting requires positive frequencies"
        )
    squared = np.square(frequencies)
    numerator = (12_194.0**2) * np.square(squared)
    denominator = (
        (squared + 20.6**2)
        * np.sqrt((squared + 107.7**2) * (squared + 737.9**2))
        * (squared + 12_194.0**2)
    )
    raw_db = 20.0 * np.log10(
        np.maximum(numerator / denominator, np.finfo(np.float64).tiny)
    )
    raw_db += IEC_A_WEIGHTING_1KHZ_OFFSET_DB
    anchor_squared = 1_000.0**2
    anchor_numerator = (12_194.0**2) * anchor_squared**2
    anchor_denominator = (
        (anchor_squared + 20.6**2)
        * np.sqrt(
            (anchor_squared + 107.7**2)
            * (anchor_squared + 737.9**2)
        )
        * (anchor_squared + 12_194.0**2)
    )
    anchor_db = 20.0 * np.log10(anchor_numerator / anchor_denominator)
    anchor_db += IEC_A_WEIGHTING_1KHZ_OFFSET_DB
    raw_db -= anchor_db
    return np.power(10.0, raw_db / 20.0)


@lru_cache(maxsize=4)
def _outer_middle_ear_fir(sample_rate_hz: float) -> np.ndarray:
    sample_rate = _positive_finite_scalar(sample_rate_hz, "sample_rate_hz")
    if sample_rate < 4_000.0:
        raise HearingFrontendError(
            "sample_rate_hz must be at least 4000 Hz for this ear model"
        )
    nyquist = sample_rate / 2.0
    interior = np.unique(
        np.r_[
            ZWICKER_A0_FREQUENCIES_HZ[
                ZWICKER_A0_FREQUENCIES_HZ < nyquist
            ],
            np.geomspace(20.0, nyquist * 0.98, 768),
        ]
    )
    grid = np.unique(np.r_[0.0, interior, nyquist])
    magnitudes = outer_middle_ear_response(grid)
    linear_phase = signal.firwin2(
        EAR_FIR_LINEAR_TAPS,
        grid,
        magnitudes,
        fs=sample_rate,
    )
    minimum_phase = signal.minimum_phase(
        linear_phase,
        method="homomorphic",
        n_fft=EAR_FIR_N_FFT,
    )
    _, response = signal.freqz(
        minimum_phase,
        worN=np.array([1_000.0]),
        fs=sample_rate,
    )
    magnitude = float(np.abs(response[0]))
    if not np.isfinite(magnitude) or magnitude <= 0.0:
        raise HearingFrontendError("ear filter cannot be normalized at 1 kHz")
    minimum_phase = minimum_phase / magnitude
    return np.asarray(minimum_phase, dtype=np.float64)


def outer_middle_ear_filter_response(
    frequencies_hz: np.ndarray,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
) -> np.ndarray:
    """Actual frequency response of the causal approximate ear FIR."""

    frequencies = _as_frequency_vector(frequencies_hz)
    sample_rate = _positive_finite_scalar(sample_rate_hz, "sample_rate_hz")
    if np.any(frequencies > sample_rate / 2.0):
        raise HearingFrontendError(
            "frequencies_hz must not exceed Nyquist"
        )
    coefficients = _outer_middle_ear_fir(sample_rate)
    return signal.freqz(
        coefficients,
        worN=frequencies,
        fs=sample_rate,
    )[1]


def apply_outer_middle_ear_filter(
    samples: np.ndarray,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
) -> np.ndarray:
    """Apply the causal approximate outer/middle-ear FIR."""

    signal_values = _as_signal(samples)
    sample_rate = _positive_finite_scalar(sample_rate_hz, "sample_rate_hz")
    coefficients = _outer_middle_ear_fir(sample_rate)
    result = signal.lfilter(coefficients, [1.0], signal_values)
    if not np.isfinite(result).all():
        raise HearingFrontendError("ear filter produced non-finite output")
    return np.asarray(result, dtype=np.float64)


@lru_cache(maxsize=4)
def _envelope_lowpass_sos(sample_rate_hz: float) -> np.ndarray:
    sample_rate = _positive_finite_scalar(sample_rate_hz, "sample_rate_hz")
    if ENVELOPE_LOWPASS_HZ >= sample_rate / 2.0:
        raise HearingFrontendError(
            "sample rate is too low for the 120 Hz envelope low-pass"
        )
    return signal.butter(
        ENVELOPE_LOWPASS_ORDER,
        ENVELOPE_LOWPASS_HZ,
        btype="low",
        fs=sample_rate,
        output="sos",
    )


def _downsample_envelope(
    envelope: np.ndarray,
    frame_size: int,
) -> np.ndarray:
    frame_count = envelope.size // frame_size
    if frame_count == 0:
        raise HearingFrontendError(
            "signal is too short for one complete output frame"
        )
    indices = (
        np.arange(frame_count, dtype=np.int64) * frame_size
        + frame_size // 2
    )
    return np.maximum(envelope[indices], 0.0)


def auditory_spectrum(
    samples: np.ndarray,
    centres: np.ndarray,
    sample_rate: float = SAMPLE_RATE_HZ,
    frame_seconds: float = FRAME_SECONDS,
) -> np.ndarray:
    """Return the non-negative Pa hearing spectrum with shape ``(99, frames)``.

    ``centres`` is deliberately mandatory. No centre-frequency grid is
    synthesized here because TS 103 281 requires 33 basic filters plus 66
    intermediate filters, and the caller owns that frequency construction.
    """

    signal_values = _as_signal(samples)
    sample_rate_hz = _positive_finite_scalar(sample_rate, "sample_rate")
    frame_seconds_value = _positive_finite_scalar(
        frame_seconds,
        "frame_seconds",
    )
    centre_values = _validate_centres(centres, sample_rate_hz)
    frame_size = int(round(sample_rate_hz * frame_seconds_value))
    if frame_size < 1:
        raise HearingFrontendError("frame size rounds to zero samples")
    if signal_values.size < frame_size:
        raise HearingFrontendError(
            "signal is too short for one complete output frame"
        )

    ear_filtered = apply_outer_middle_ear_filter(
        signal_values,
        sample_rate_hz,
    )
    lowpass = _envelope_lowpass_sos(sample_rate_hz)
    output = np.empty(
        (AUDITORY_BAND_COUNT, signal_values.size // frame_size),
        dtype=np.float64,
    )
    for band, centre in enumerate(centre_values):
        bandpass = auditory_bandpass_signal(
            ear_filtered,
            float(centre),
            sample_rate_hz,
        )
        rectified = np.maximum(bandpass, 0.0)
        envelope = signal.sosfilt(lowpass, rectified)
        output[band] = _downsample_envelope(envelope, frame_size)

    if not np.isfinite(output).all():
        raise HearingFrontendError("hearing spectrum contains non-finite values")
    return np.maximum(output, 0.0)
