"""Iterative noise-mask reconstruction for TS 103 281 Model A.

TS 103 281 V1.3.2 clause 6.3.2, equation (9), states that the time-domain
product of a fixed-frequency-band envelope and its reliability mask becomes a
convolution in the FFT domain. The standard then describes iterative spectral
deconvolution based on interpolation of dominant frequency lines. It does not
publish the original implementation's exact stopping condition.

This module implements an independently auditable approximation:

1. ``mask == 1`` everywhere is an exact identity path.
2. Otherwise each frequency band is processed independently.
3. The reliable weighted mean is fitted first as the DC line.
4. The FFT of the masked residual proposes the dominant new line.
5. All selected frequencies and amplitudes are refitted to every visible
   sample with weighted least squares; selected frequencies are then refined
   with bounded weighted nonlinear least squares.
6. Iteration stops at a numerical residual tolerance, an insignificant line,
   or a finite iteration cap.

No missing sample is inserted as 0 dB. No circular shift/``np.roll`` is used.
The reconstruction has no Annex C fitted constants.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

import numpy as np
from scipy.optimize import least_squares


METHOD: Final[str] = "iterative_fft_clean_weighted_least_squares"
MAX_LINES: Final[int] = 32
FFT_CANDIDATE_PEAKS: Final[int] = 32
STOP_RELATIVE_TOLERANCE: Final[float] = (
    100.0 * math.sqrt(float(np.finfo(np.float64).eps))
)
REFINEMENT_TOLERANCE: Final[float] = math.sqrt(
    float(np.finfo(np.float64).eps)
)
MIN_FREQUENCY_SEPARATION_BINS: Final[int] = 1

NUMERICAL_ASSUMPTIONS: Final[tuple[str, ...]] = (
    "mask values are used directly as weighted-least-squares weights",
    "masked/missing samples are never replaced by 0 dB",
    "FFT peak search uses a rectangular window over the masked residual",
    "new line frequencies are refined with bounded weighted nonlinear "
    "least squares",
    "analytic derivatives and iterative LSMR solve the unchanged least-squares objective",
    "the original Sottek iteration stop rule is unavailable; this module "
    "uses a machine-epsilon-derived relative tolerance and a finite cap",
    "no np.roll or other circular endpoint wrapping is used",
)

__all__ = [
    "MAX_LINES",
    "METHOD",
    "NUMERICAL_ASSUMPTIONS",
    "NoiseReconstructionError",
    "REFINEMENT_TOLERANCE",
    "STOP_RELATIVE_TOLERANCE",
    "reconstruct_noise_log_envelope",
]


class NoiseReconstructionError(ValueError):
    """The noise/mask contract is invalid or a band has no observations."""


@dataclass(frozen=True)
class _LinearFit:
    frequencies: np.ndarray
    coefficients: np.ndarray
    model: np.ndarray
    weighted_sse: float
    condition_number: float


def _as_real_matrix(value: np.ndarray, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise NoiseReconstructionError(f"{name} must be real-valued")
    try:
        result = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise NoiseReconstructionError(f"{name} must be numeric") from error
    if result.ndim != 2:
        raise NoiseReconstructionError(f"{name} must have shape (bands, time)")
    if 0 in result.shape:
        raise NoiseReconstructionError(f"{name} must not be empty")
    if not np.isfinite(result).all():
        raise NoiseReconstructionError(
            f"{name} must contain only finite values"
        )
    return result


def _design_matrix(
    time_index: np.ndarray,
    frequencies: np.ndarray,
) -> np.ndarray:
    columns = [np.ones(time_index.size, dtype=np.float64)]
    for frequency in frequencies:
        phase = 2.0 * np.pi * float(frequency) * time_index
        columns.append(np.cos(phase))
        columns.append(np.sin(phase))
    return np.column_stack(columns)


def _weighted_linear_fit(
    values: np.ndarray,
    mask: np.ndarray,
    frequencies: np.ndarray,
) -> _LinearFit:
    time_index = np.arange(values.size, dtype=np.float64)
    visible = mask > 0.0
    design = _design_matrix(time_index, frequencies)
    weighted_design = design[visible] * mask[visible, np.newaxis]
    weighted_values = values[visible] * mask[visible]
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        weighted_design,
        weighted_values,
        rcond=None,
    )
    if rank < weighted_design.shape[1]:
        raise NoiseReconstructionError(
            "visible samples cannot identify the requested spectral lines"
        )
    model = design @ coefficients
    residual = mask * (values - model)
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values.size and singular_values[-1] > 0.0
        else float("inf")
    )
    return _LinearFit(
        np.asarray(frequencies, dtype=np.float64),
        np.asarray(coefficients, dtype=np.float64),
        np.asarray(model, dtype=np.float64),
        float(np.dot(residual, residual)),
        condition,
    )


def _refine_frequencies(
    values: np.ndarray,
    mask: np.ndarray,
    frequencies: np.ndarray,
) -> _LinearFit:
    """Refine all selected lines while solving amplitudes by weighted LS."""

    if frequencies.size == 0:
        return _weighted_linear_fit(values, mask, frequencies)

    initial = _weighted_linear_fit(values, mask, frequencies)
    visible = mask > 0.0
    time_index = np.flatnonzero(visible).astype(np.float64) / values.size
    weights = mask[visible]
    observed = values[visible]
    line_count = frequencies.size
    # Frequency-bin units keep derivatives well-scaled as sample count grows.
    # The model and bounds are identical to cycles/sample after division by N.
    minimum_frequency = 1.0
    maximum_frequency = (0.5 - 1e-12) * values.size

    def unpack(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            parameters[:line_count],
            parameters[line_count:],
        )

    def residual(parameters: np.ndarray) -> np.ndarray:
        line_frequencies, coefficients = unpack(parameters)
        design = _design_matrix(time_index, line_frequencies)
        model = design @ coefficients
        return weights * (model - observed)

    def jacobian(parameters: np.ndarray) -> np.ndarray:
        line_frequencies, coefficients = unpack(parameters)
        design = _design_matrix(time_index, line_frequencies)
        cosine = design[:, 1::2]
        sine = design[:, 2::2]
        frequency_derivative = (2 * np.pi * time_index[:, None]) * (
            -sine * coefficients[1::2] + cosine * coefficients[2::2]
        )
        return weights[:, None] * np.column_stack((frequency_derivative, design))

    initial_parameters = np.r_[
        np.asarray(frequencies, dtype=np.float64) * values.size,
        initial.coefficients,
    ]
    lower = np.r_[
        np.full(line_count, minimum_frequency),
        np.full(initial.coefficients.size, -np.inf),
    ]
    upper = np.r_[
        np.full(line_count, maximum_frequency),
        np.full(initial.coefficients.size, np.inf),
    ]
    refined = least_squares(
        residual,
        initial_parameters,
        bounds=(lower, upper),
        jac=jacobian,
        tr_solver="lsmr",
        x_scale="jac",
        xtol=REFINEMENT_TOLERANCE,
        ftol=REFINEMENT_TOLERANCE,
        gtol=REFINEMENT_TOLERANCE,
        max_nfev=200,
    )
    if not np.isfinite(refined.x).all():
        return initial
    refined_frequencies = np.asarray(
        refined.x[:line_count] / values.size,
        dtype=np.float64,
    )
    refined_frequencies.sort()
    return _weighted_linear_fit(values, mask, refined_frequencies)


def _line_candidate(
    values: np.ndarray,
    model: np.ndarray,
    mask: np.ndarray,
    frequencies: np.ndarray,
) -> tuple[float, float]:
    time_index = np.arange(values.size, dtype=np.float64)
    maximum_bin = (values.size - 1) // 2
    if maximum_bin < 1:
        return 0.0, 0.0
    residual = mask * (values - model)
    fft_magnitude = np.abs(np.fft.rfft(residual))
    fft_magnitude[0] = 0.0
    if values.size % 2 == 0:
        fft_magnitude[-1] = 0.0
    candidate_count = min(
        FFT_CANDIDATE_PEAKS,
        maximum_bin,
    )
    fft_candidates = np.argsort(fft_magnitude[1 : maximum_bin + 1])[
        -candidate_count:
    ] + 1
    bins = np.sort(np.unique(fft_candidates))
    candidate_frequencies = bins / values.size
    allowed = np.ones(bins.size, dtype=bool)
    radius = MIN_FREQUENCY_SEPARATION_BINS
    for frequency in frequencies:
        centre = int(round(float(frequency) * values.size))
        # `bins` is a sparse top-K selection, not the full FFT-bin array.
        # Indexing `allowed` with an FFT bin number masked unrelated peaks.
        allowed &= np.abs(bins - centre) > radius
    if not np.any(allowed):
        return 0.0, 0.0

    denominator = float(np.sum(mask[mask > 0.0] ** 2))
    if denominator <= 0.0:
        return 0.0, 0.0
    scores = np.full(bins.size, -np.inf, dtype=np.float64)
    chunk_size = 128
    for start in range(0, bins.size, chunk_size):
        stop = min(bins.size, start + chunk_size)
        phases = (
            2.0
            * np.pi
            * candidate_frequencies[start:stop, np.newaxis]
            * time_index[np.newaxis, :]
        )
        cosine = np.cos(phases) @ residual
        sine = np.sin(phases) @ residual
        scores[start:stop] = np.hypot(cosine, sine) / denominator
    scores[~allowed] = -np.inf
    peak_index = int(np.argmax(scores))
    return (
        float(candidate_frequencies[peak_index]),
        float(scores[peak_index]),
    )


def _residual_statistics(
    values: np.ndarray,
    mask: np.ndarray,
    model: np.ndarray,
) -> tuple[float, float]:
    residual = mask * (values - model)
    visible = mask > 0.0
    denominator = float(np.sum(mask[visible] ** 2))
    rms = (
        float(np.sqrt(np.sum(residual[visible] ** 2) / denominator))
        if denominator > 0.0
        else float("inf")
    )
    maximum = float(np.max(np.abs(residual[visible])))
    return rms, maximum


def _reconstruct_band(
    values: np.ndarray,
    mask: np.ndarray,
    band_index: int,
) -> tuple[np.ndarray, dict]:
    visible = mask > 0.0
    observation_count = int(np.count_nonzero(visible))
    if observation_count == 0:
        raise NoiseReconstructionError(
            f"band {band_index} has no observations"
        )

    if np.all(mask == 1.0):
        return values.copy(), {
            "band": band_index,
            "identity": True,
            "usable": True,
            "observations": int(mask.size),
            "missing": 0,
            "iterations": 0,
            "dominant_frequencies_cycles_per_sample": [],
            "residual_rms_db": 0.0,
            "max_abs_residual_db": 0.0,
            "relative_residual": 0.0,
            "tolerance_db": 0.0,
            "condition_number": 1.0,
            "converged": True,
        }

    weighted_mean = float(
        np.average(values[visible], weights=mask[visible])
    )
    model = np.full(values.size, weighted_mean, dtype=np.float64)
    frequencies = np.asarray([], dtype=np.float64)
    scale = max(
        1.0,
        float(np.ptp(values[visible])),
        abs(float(np.mean(values[visible]))),
    )
    tolerance = STOP_RELATIVE_TOLERANCE * scale
    maximum_lines = min(
        MAX_LINES,
        max(0, (observation_count - 1) // 2),
    )
    condition_number = 1.0
    converged = False
    iteration = 0

    for _ in range(maximum_lines):
        residual_rms, _ = _residual_statistics(values, mask, model)
        if residual_rms <= tolerance:
            converged = True
            break
        candidate_frequency, candidate_peak = _line_candidate(
            values,
            model,
            mask,
            frequencies,
        )
        if (
            candidate_peak <= tolerance
            or candidate_frequency <= 0.0
        ):
            converged = True
            break
        proposed = np.append(frequencies, candidate_frequency)
        try:
            fitted = _refine_frequencies(values, mask, proposed)
        except NoiseReconstructionError:
            break
        previous_rms = residual_rms
        new_rms, _ = _residual_statistics(
            values,
            mask,
            fitted.model,
        )
        if new_rms > previous_rms * (1.0 + REFINEMENT_TOLERANCE):
            break
        frequencies = fitted.frequencies
        model = fitted.model
        condition_number = fitted.condition_number
        iteration += 1
        if new_rms <= tolerance:
            converged = True
            break
        if previous_rms - new_rms <= (
            REFINEMENT_TOLERANCE * max(1.0, previous_rms)
        ):
            converged = True
            break
    else:
        converged = False

    residual_rms, maximum_residual = _residual_statistics(
        values,
        mask,
        model,
    )
    if residual_rms <= tolerance:
        converged = True
    if not np.isfinite(model).all():
        raise NoiseReconstructionError(
            f"band {band_index} reconstruction is non-finite"
        )
    diagnostics = {
        "band": band_index,
        "identity": False,
        "usable": True,
        "observations": observation_count,
        "missing": int(mask.size - observation_count),
        "iterations": iteration,
        "dominant_frequencies_cycles_per_sample": (
            frequencies.tolist()
        ),
        "residual_rms_db": residual_rms,
        "max_abs_residual_db": maximum_residual,
        "relative_residual": residual_rms / scale,
        "tolerance_db": tolerance,
        "condition_number": condition_number,
        "converged": bool(converged),
    }
    return np.asarray(model, dtype=np.float64), diagnostics


def reconstruct_noise_log_envelope(
    noise_db: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Reconstruct a masked ``(bands, time)`` noise log envelope.

    ``mask`` uses the TS 103 281 convention: 1 means reliable, 0 means no
    observation, and intermediate values are weighted observations. At least
    one positive-mask sample is required in every band; otherwise this raises
    :class:`NoiseReconstructionError` instead of inventing a 0 dB fallback.
    """

    noise = _as_real_matrix(noise_db, "noise_db")
    reliability = _as_real_matrix(mask, "mask")
    if noise.shape != reliability.shape:
        raise NoiseReconstructionError(
            "noise_db and mask must have the same shape"
        )
    if np.any(reliability < 0.0) or np.any(reliability > 1.0):
        raise NoiseReconstructionError("mask values must lie in [0, 1]")
    empty_bands = np.flatnonzero(np.all(reliability == 0.0, axis=1))
    if empty_bands.size:
        raise NoiseReconstructionError(
            "bands with no observations: "
            + ", ".join(str(int(index)) for index in empty_bands)
        )

    reconstructed = np.empty_like(noise, dtype=np.float64)
    band_diagnostics: list[dict] = []
    for band in range(noise.shape[0]):
        reconstructed[band], diagnostics = _reconstruct_band(
            noise[band].copy(),
            reliability[band].copy(),
            band,
        )
        band_diagnostics.append(diagnostics)

    if not np.isfinite(reconstructed).all():
        raise NoiseReconstructionError(
            "reconstructed envelope contains non-finite values"
        )
    result = {
        "method": METHOD,
        "standard_reference": "ETSI TS 103 281 V1.3.2 equation (9)",
        "exact_sottek_iteration": False,
        "numerical_assumptions": list(NUMERICAL_ASSUMPTIONS),
        "stop_relative_tolerance": STOP_RELATIVE_TOLERANCE,
        "refinement_tolerance": REFINEMENT_TOLERANCE,
        "max_lines": MAX_LINES,
        "all_bands_usable": True,
        "input_unchanged": True,
        "bands": band_diagnostics,
    }
    return reconstructed, result
