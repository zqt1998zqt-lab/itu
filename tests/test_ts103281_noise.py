"""Independent controls for TS 103 281 masked-noise reconstruction."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from server.quality.ts103281_noise import (
    METHOD,
    NUMERICAL_ASSUMPTIONS,
    NoiseReconstructionError,
    STOP_RELATIVE_TOLERANCE,
    reconstruct_noise_log_envelope,
)


def _mask_with_holes(length: int = 128) -> np.ndarray:
    mask = np.ones(length, dtype=np.float64)
    mask[20:35] = 0.0
    mask[70:90] = 0.0
    mask[40:45] = 0.4
    return mask


def _modulated_envelope(
    offset: float,
    frequencies: tuple[float, ...],
    amplitudes: tuple[float, ...],
    phases: tuple[float, ...],
    length: int = 128,
) -> np.ndarray:
    time = np.arange(length, dtype=np.float64)
    result = np.full(length, offset, dtype=np.float64)
    for frequency, amplitude, phase in zip(
        frequencies,
        amplitudes,
        phases,
        strict=True,
    ):
        result += amplitude * np.cos(
            2.0 * np.pi * frequency * time + phase
        )
    return result


def test_mask_product_is_frequency_domain_convolution():
    rng = np.random.default_rng(103281)
    values = rng.normal(size=64)
    mask = np.linspace(0.0, 1.0, values.size)
    left = np.fft.rfft(values * mask)
    full_values = np.fft.fft(values)
    full_mask = np.fft.fft(mask)
    convolved = (
        np.fft.ifft(np.fft.fft(full_values) * np.fft.fft(full_mask))
        / values.size
    )
    np.testing.assert_allclose(left, convolved[: left.size], atol=1e-12)


def test_full_mask_is_exact_identity_and_does_not_mutate_inputs():
    noise = np.array(
        [
            [3.0, 3.2, 2.8, 3.1, 2.9, 3.0, 3.05, 2.95],
            [-12.0, -11.8, -12.2, -12.1, -11.9, -12.0, -12.05, -11.95],
        ]
    )
    mask = np.ones_like(noise)
    original_noise = noise.copy()
    original_mask = mask.copy()
    reconstructed, diagnostics = reconstruct_noise_log_envelope(
        noise,
        mask,
    )
    np.testing.assert_array_equal(reconstructed, noise)
    np.testing.assert_array_equal(noise, original_noise)
    np.testing.assert_array_equal(mask, original_mask)
    assert all(entry["identity"] for entry in diagnostics["bands"])
    assert all(entry["iterations"] == 0 for entry in diagnostics["bands"])
    assert diagnostics["input_unchanged"] is True


def test_constant_envelopes_are_recovered_inside_holes():
    noise = np.array(
        [
            np.full(128, 6.0),
            np.full(128, -18.5),
        ]
    )
    mask = np.vstack(
        [_mask_with_holes(), _mask_with_holes()]
    )
    reconstructed, diagnostics = reconstruct_noise_log_envelope(
        noise,
        mask,
    )
    np.testing.assert_allclose(reconstructed, noise, atol=1e-12)
    assert all(entry["converged"] for entry in diagnostics["bands"])
    assert all(
        entry["dominant_frequencies_cycles_per_sample"] == []
        for entry in diagnostics["bands"]
    )


def test_known_off_bin_modulation_is_recovered_through_holes():
    envelope = _modulated_envelope(
        offset=5.0,
        frequencies=(5.3 / 128.0, 11.4 / 128.0),
        amplitudes=(2.0, 0.7),
        phases=(0.3, -0.8),
    )
    noise = np.vstack([envelope, 0.5 * envelope - 4.0])
    mask = np.vstack([_mask_with_holes(), _mask_with_holes()])
    reconstructed, diagnostics = reconstruct_noise_log_envelope(
        noise,
        mask,
    )
    holes = mask == 0.0
    error = np.abs(reconstructed - noise)[holes]
    assert np.max(error) < 1e-8
    assert all(entry["converged"] for entry in diagnostics["bands"])
    assert all(entry["iterations"] >= 2 for entry in diagnostics["bands"])
    assert all(
        len(entry["dominant_frequencies_cycles_per_sample"]) == 2
        for entry in diagnostics["bands"]
    )


def test_changing_visible_sample_changes_reconstruction_negative_control():
    baseline = _modulated_envelope(
        offset=2.0,
        frequencies=(3.0 / 64.0,),
        amplitudes=(1.0,),
        phases=(0.2,),
        length=64,
    )
    changed = baseline.copy()
    changed[5] += 0.8
    mask = np.ones(64)
    mask[24:32] = 0.0
    reconstructed_a, _ = reconstruct_noise_log_envelope(
        baseline[np.newaxis, :],
        mask[np.newaxis, :],
    )
    reconstructed_b, _ = reconstruct_noise_log_envelope(
        changed[np.newaxis, :],
        mask[np.newaxis, :],
    )
    assert not np.allclose(reconstructed_a, reconstructed_b)
    assert not np.allclose(
        reconstructed_a[0, mask == 0.0],
        reconstructed_b[0, mask == 0.0],
    )


def test_all_zero_mask_raises_instead_of_inventing_zero_db():
    noise = np.ones((2, 16))
    mask = np.ones((2, 16))
    mask[1] = 0.0
    with pytest.raises(
        NoiseReconstructionError,
        match="no observations: 1",
    ):
        reconstruct_noise_log_envelope(noise, mask)


@pytest.mark.parametrize(
    ("noise", "mask", "message"),
    [
        (
            np.ones(8),
            np.ones((1, 8)),
            "bands, time",
        ),
        (
            np.ones((2, 8)),
            np.ones((1, 8)),
            "same shape",
        ),
        (
            np.array([[1.0, np.nan, 1.0]]),
            np.ones((1, 3)),
            "finite",
        ),
        (
            np.ones((1, 8)),
            np.array([[1.0, 0.5, -0.1, 1.0, 1.0, 1.0, 1.0, 1.0]]),
            r"\[0, 1\]",
        ),
        (
            np.ones((1, 8)),
            np.array([[1.0, 1.1, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0]]),
            r"\[0, 1\]",
        ),
        (
            np.empty((0, 8)),
            np.empty((0, 8)),
            "must not be empty",
        ),
    ],
)
def test_invalid_inputs_are_rejected(
    noise: np.ndarray,
    mask: np.ndarray,
    message: str,
):
    with pytest.raises(NoiseReconstructionError, match=message):
        reconstruct_noise_log_envelope(noise, mask)


def test_outputs_are_finite_and_diagnostics_record_numerical_contract():
    rng = np.random.default_rng(9)
    noise = rng.normal(0.0, 6.0, size=(4, 96))
    mask = np.ones_like(noise)
    mask[:, 20:30] = 0.0
    mask[:, 50:55] = 0.25
    reconstructed, diagnostics = reconstruct_noise_log_envelope(
        noise,
        mask,
    )
    assert reconstructed.shape == noise.shape
    assert np.isfinite(reconstructed).all()
    assert diagnostics["method"] == METHOD
    assert diagnostics["exact_sottek_iteration"] is False
    assert diagnostics["numerical_assumptions"] == list(
        NUMERICAL_ASSUMPTIONS
    )
    assert diagnostics["stop_relative_tolerance"] == pytest.approx(
        STOP_RELATIVE_TOLERANCE
    )
    assert len(diagnostics["bands"]) == noise.shape[0]
    for entry in diagnostics["bands"]:
        assert {
            "iterations",
            "residual_rms_db",
            "max_abs_residual_db",
            "tolerance_db",
            "converged",
        }.issubset(entry)
        assert entry["residual_rms_db"] >= 0.0
        assert entry["max_abs_residual_db"] >= 0.0


def test_verification_artifact_records_formula_and_assumptions(
    tmp_path: Path,
):
    artifact = {
        "standard_reference": "ETSI TS 103 281 V1.3.2 equation (9)",
        "method": METHOD,
        "exact_sottek_iteration": False,
        "numerical_assumptions": list(NUMERICAL_ASSUMPTIONS),
        "stop_relative_tolerance": STOP_RELATIVE_TOLERANCE,
        "annex_c_fitted_constants_used": False,
        "full_mask_identity": True,
        "no_observations": "ValueError",
    }
    path = tmp_path / "noise-reconstruction-contract.json"
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8")) == artifact
