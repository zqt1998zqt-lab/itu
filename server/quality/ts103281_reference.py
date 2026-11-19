"""Partial reference optimization for ETSI TS 103 281 clause 6.3.6.2.

The public entry point implements only the three reference-only operations
described in 6.3.6.2.  It does not implement the masking operation from
6.3.6.3 and therefore must not be presented as a complete Model A reference
preprocessor.

Array convention
----------------
Both inputs are finite, non-negative pressure-amplitude matrices with shape
``(bands, time)``.  All interpolation is linear.  The original discrete
support is ``[0, n-1]`` on each axis: exact sample-grid points are preserved,
while factor-four interpolation positions strictly between ``n-1`` and ``n``
are outside the support and explicitly zero.  Every shift copies only
in-range source samples.  No operation wraps the tail into the head.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import correlate


UPSAMPLE_FACTOR = 4
LOCAL_SHIFT = 3
_TIE_RELATIVE_TOLERANCE = 1e-12


def _validated_pair(
    reference: np.ndarray,
    degraded: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    reference_array = np.asarray(reference, dtype=np.float64)
    degraded_array = np.asarray(degraded, dtype=np.float64)
    if reference_array.ndim != 2 or degraded_array.ndim != 2:
        raise ValueError("reference and degraded must both be two-dimensional")
    if reference_array.shape != degraded_array.shape:
        raise ValueError("reference and degraded must have the same shape")
    if 0 in reference_array.shape:
        raise ValueError("reference and degraded must be non-empty")
    if not np.isfinite(reference_array).all():
        raise ValueError("reference must contain only finite amplitudes")
    if not np.isfinite(degraded_array).all():
        raise ValueError("degraded must contain only finite amplitudes")
    if np.any(reference_array < 0):
        raise ValueError("reference amplitudes must be non-negative")
    if np.any(degraded_array < 0):
        raise ValueError("degraded amplitudes must be non-negative")
    return reference_array, degraded_array


def _validate_parameters(
    *,
    factor: int,
    local_shift: int | None = None,
) -> None:
    if factor <= 0:
        raise ValueError("upsampling factor must be positive")
    if local_shift is not None and local_shift < 0:
        raise ValueError("local shift search radius must be non-negative")


def _linear_upsample(
    values: np.ndarray,
    factor: int,
    *,
    axis: int,
) -> np.ndarray:
    if factor <= 0:
        raise ValueError("upsampling factor must be positive")
    count = values.shape[axis]
    positions = np.arange(count * factor, dtype=np.float64) / factor
    left = np.floor(positions).astype(np.intp)
    fraction = positions - left
    right = np.minimum(left + 1, count - 1)
    left_values = np.take(values, left, axis=axis)
    right_values = np.take(values, right, axis=axis)
    weight_shape = [1] * values.ndim
    weight_shape[axis] = fraction.size
    weight = fraction.reshape(weight_shape)
    interpolated = (1.0 - weight) * left_values + weight * right_values
    valid = (left < count - 1) | (fraction == 0.0)
    valid_shape = [1] * values.ndim
    valid_shape[axis] = valid.size
    return np.where(valid.reshape(valid_shape), interpolated, 0.0)


def _downsample(values: np.ndarray, factor: int, *, axis: int) -> np.ndarray:
    return np.take(
        values,
        np.arange(values.shape[axis] // factor) * factor,
        axis=axis,
    )


def _shift_zero_fill(
    values: np.ndarray,
    shift: int,
    *,
    axis: int,
) -> np.ndarray:
    """Shift along one axis, filling the vacated samples with zero."""
    result = np.zeros_like(values)
    count = values.shape[axis]
    if abs(shift) >= count or shift == 0:
        if shift == 0:
            result[...] = values
        return result
    source = [slice(None)] * values.ndim
    destination = [slice(None)] * values.ndim
    if shift > 0:
        source[axis] = slice(0, count - shift)
        destination[axis] = slice(shift, count)
    else:
        source[axis] = slice(-shift, count)
        destination[axis] = slice(0, count + shift)
    result[tuple(destination)] = values[tuple(source)]
    return result


def _preferred_index(
    candidates: np.ndarray,
    scores: np.ndarray,
    *,
    maximize: bool,
) -> int:
    """Return a deterministic best candidate, preferring zero on ties."""
    best = np.max(scores) if maximize else np.min(scores)
    tied = np.isclose(
        scores,
        best,
        rtol=_TIE_RELATIVE_TOLERANCE,
        atol=np.finfo(np.float64).eps,
    )
    zero_indices = np.flatnonzero(candidates == 0)
    if zero_indices.size:
        zero_index = int(zero_indices[0])
        if tied[zero_index]:
            return zero_index
    tied_indices = np.flatnonzero(tied)
    return int(
        min(
            tied_indices,
            key=lambda index: (
                abs(int(candidates[index])),
                int(candidates[index]) < 0,
            ),
        )
    )


def _preferred_indices(
    candidates: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    """Vectorized minimum selection with deterministic zero-first tie-break."""
    best = np.min(scores, axis=0)
    tied = np.isclose(
        scores,
        best[None, :],
        rtol=_TIE_RELATIVE_TOLERANCE,
        atol=np.finfo(np.float64).eps,
    )
    preference = sorted(
        (int(candidate) for candidate in candidates),
        key=lambda candidate: (abs(candidate), candidate < 0),
    )
    candidate_indices = {
        int(candidate): index for index, candidate in enumerate(candidates)
    }
    selected = np.full(scores.shape[1], candidate_indices[0], dtype=np.intp)
    remaining = np.ones(scores.shape[1], dtype=bool)
    for candidate in preference:
        index = candidate_indices[candidate]
        choose = remaining & tied[index]
        selected[choose] = index
        remaining[choose] = False
    if remaining.any():
        raise RuntimeError("failed to select a local shift")
    return selected


def _normalized_correlation(
    reference: np.ndarray,
    degraded: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-aware mean-centred correlation normalized by full-series energy."""
    count = reference.size
    candidates = np.arange(-count + 1, count, dtype=np.intp)
    reference_centered = np.zeros_like(reference)
    reference_active = reference > 0
    if reference_active.any():
        reference_centered[reference_active] = (
            reference[reference_active]
            - np.mean(reference[reference_active])
        )
    degraded_centered = np.zeros_like(degraded)
    degraded_active = degraded > 0
    if degraded_active.any():
        degraded_centered[degraded_active] = (
            degraded[degraded_active]
            - np.mean(degraded[degraded_active])
        )
    correlation = correlate(
        degraded_centered,
        reference_centered,
        mode="full",
        method="fft",
    )
    denominator = float(
        np.sqrt(
            np.sum(reference_centered**2)
            * np.sum(degraded_centered**2)
        )
    )
    normalized = (
        correlation / denominator
        if denominator > np.finfo(np.float64).eps
        else np.zeros_like(correlation)
    )
    return candidates, normalized


def _group_delay_validated(
    reference: np.ndarray,
    degraded: np.ndarray,
    *,
    factor: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    band_count = reference.shape[0]
    lags = np.zeros(band_count, dtype=np.intp)
    peaks = np.zeros(band_count, dtype=np.float64)
    aligned = np.empty_like(reference)
    for band in range(band_count):
        reference_band = _linear_upsample(
            reference[band], factor, axis=0
        )
        degraded_band = _linear_upsample(
            degraded[band], factor, axis=0
        )
        if not np.any(reference_band) or not np.any(degraded_band):
            lag = 0
            peak = 0.0
        else:
            candidates, correlation = _normalized_correlation(
                reference_band,
                degraded_band,
            )
            index = _preferred_index(
                candidates,
                correlation,
                maximize=True,
            )
            lag = int(candidates[index])
            peak = float(correlation[index])
        lags[band] = lag
        peaks[band] = peak
        shifted = _shift_zero_fill(
            reference_band,
            lag,
            axis=0,
        )
        aligned[band] = _downsample(shifted, factor, axis=0)
    return aligned, {
        "factor": factor,
        "per_band_lag_upsampled": lags,
        "per_band_lag_samples": lags.astype(np.float64) / factor,
        "per_band_correlation_peak": peaks,
        "interpolation": "linear",
        "correlation_normalization": (
            "zero_aware_mean_centered_full_series_energy"
        ),
        "boundary": "zero_fill",
        "circular": False,
    }


def compensate_group_delay(
    reference: np.ndarray,
    degraded: np.ndarray,
    *,
    factor: int = UPSAMPLE_FACTOR,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compensate per-band group delay as specified in 6.3.6.2."""
    reference_array, degraded_array = _validated_pair(reference, degraded)
    _validate_parameters(factor=factor)
    return _group_delay_validated(
        reference_array,
        degraded_array,
        factor=factor,
    )


def _spectral_compression_validated(
    reference: np.ndarray,
    degraded: np.ndarray,
    *,
    factor: int,
    local_shift: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    reference_up = _linear_upsample(reference, factor, axis=0)
    degraded_up = _linear_upsample(degraded, factor, axis=0)
    candidates = np.arange(-local_shift, local_shift + 1, dtype=np.intp)
    errors = np.empty((candidates.size, reference.shape[1]), dtype=np.float64)
    for index, shift in enumerate(candidates):
        shifted = _shift_zero_fill(
            reference_up,
            int(shift),
            axis=0,
        )
        errors[index] = np.mean(
            (shifted - degraded_up) ** 2,
            axis=0,
        )
    selected = _preferred_indices(candidates, errors)
    shifts = candidates[selected]
    output = np.zeros_like(reference)
    band_rows = (
        np.arange(reference.shape[0], dtype=np.intp)[:, None] * factor
        - shifts[None, :]
    )
    columns = np.broadcast_to(
        np.arange(reference.shape[1], dtype=np.intp)[None, :],
        band_rows.shape,
    )
    valid = (band_rows >= 0) & (band_rows < reference_up.shape[0])
    output[valid] = reference_up[band_rows[valid], columns[valid]]
    return output, {
        "factor": factor,
        "local_shift_subbands": local_shift,
        "per_frame_shift_subbands": shifts,
        "per_frame_shift_bands": shifts.astype(np.float64) / factor,
        "per_frame_mse": errors[selected, np.arange(reference.shape[1])],
        "interpolation": "linear",
        "boundary": "zero_fill",
        "circular": False,
    }


def compensate_spectral_compression(
    reference: np.ndarray,
    degraded: np.ndarray,
    *,
    factor: int = UPSAMPLE_FACTOR,
    local_shift: int = LOCAL_SHIFT,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compensate local spectral compression/expansion in 6.3.6.2."""
    reference_array, degraded_array = _validated_pair(reference, degraded)
    _validate_parameters(factor=factor, local_shift=local_shift)
    return _spectral_compression_validated(
        reference_array,
        degraded_array,
        factor=factor,
        local_shift=local_shift,
    )


def _variable_delay_validated(
    reference: np.ndarray,
    degraded: np.ndarray,
    *,
    factor: int,
    local_shift: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    reference_up = _linear_upsample(reference, factor, axis=1)
    frame_count = reference.shape[1]
    base_indices = np.arange(frame_count, dtype=np.intp) * factor
    candidates = np.arange(-local_shift, local_shift + 1, dtype=np.intp)
    errors = np.empty((candidates.size, frame_count), dtype=np.float64)
    for index, shift in enumerate(candidates):
        indices = base_indices - shift
        valid = (indices >= 0) & (indices < reference_up.shape[1])
        candidate = np.zeros_like(reference)
        candidate[:, valid] = reference_up[:, indices[valid]]
        errors[index] = np.mean(
            (candidate - degraded) ** 2,
            axis=0,
        )
    selected = _preferred_indices(candidates, errors)
    shifts = candidates[selected]
    output = np.zeros_like(reference)
    indices = base_indices - shifts
    valid = (indices >= 0) & (indices < reference_up.shape[1])
    output[:, valid] = reference_up[:, indices[valid]]
    return output, {
        "factor": factor,
        "local_shift_subframes": local_shift,
        "per_frame_shift_subframes": shifts,
        "per_frame_shift_samples": shifts.astype(np.float64) / factor,
        "per_frame_mse": errors[selected, np.arange(frame_count)],
        "interpolation": "linear",
        "boundary": "zero_fill",
        "circular": False,
    }


def compensate_variable_delay(
    reference: np.ndarray,
    degraded: np.ndarray,
    *,
    factor: int = UPSAMPLE_FACTOR,
    local_shift: int = LOCAL_SHIFT,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compensate variable delay in the time axis as specified in 6.3.6.2."""
    reference_array, degraded_array = _validated_pair(reference, degraded)
    _validate_parameters(factor=factor, local_shift=local_shift)
    return _variable_delay_validated(
        reference_array,
        degraded_array,
        factor=factor,
        local_shift=local_shift,
    )


def optimize_reference(
    reference: np.ndarray,
    degraded: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return optimized reference, detached degraded copy, and diagnostics.

    The three 6.3.6.2 reference-only operations run in standard order.  The
    degraded values are unchanged and are returned as a memory-independent
    copy so callers cannot accidentally mutate the input through the result.
    """
    reference_array, degraded_array = _validated_pair(reference, degraded)
    group_delayed, group_diagnostics = _group_delay_validated(
        reference_array,
        degraded_array,
        factor=UPSAMPLE_FACTOR,
    )
    spectrally_compensated, spectrum_diagnostics = (
        _spectral_compression_validated(
            group_delayed,
            degraded_array,
            factor=UPSAMPLE_FACTOR,
            local_shift=LOCAL_SHIFT,
        )
    )
    optimized, delay_diagnostics = _variable_delay_validated(
        spectrally_compensated,
        degraded_array,
        factor=UPSAMPLE_FACTOR,
        local_shift=LOCAL_SHIFT,
    )
    diagnostics = {
        "status": "partial_reference_optimization",
        "complete_reference_model": False,
        "standard": "ETSI TS 103 281 V1.3.2",
        "clause": "6.3.6.2",
        "steps": {
            "group_delay": group_diagnostics,
            "spectral_compression": spectrum_diagnostics,
            "variable_delay": delay_diagnostics,
        },
        "unimplemented_steps": [
            {
                "clause": "6.3.6.3",
                "name": "Masking of inaudible differences",
                "implemented": False,
            }
        ],
        "interpolation": {
            "time": "linear",
            "frequency": "linear",
            "phase": "zero",
            "factor": UPSAMPLE_FACTOR,
        },
        "boundary": (
            "original discrete support is [0, n-1]; upsampled positions "
            "strictly between n-1 and n are zero; shifts are cropped and "
            "never wrapped"
        ),
        "input_shape": reference_array.shape,
        "degraded_unchanged": True,
    }
    return optimized, degraded_array.copy(), diagnostics


__all__ = [
    "UPSAMPLE_FACTOR",
    "LOCAL_SHIFT",
    "compensate_group_delay",
    "compensate_spectral_compression",
    "compensate_variable_delay",
    "optimize_reference",
]
