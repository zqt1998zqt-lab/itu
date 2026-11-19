"""Formula and signal controls, independent of Annex C MOS targets."""
from __future__ import annotations

import numpy as np
import pytest

from server.quality.ts103281 import (
    ModelAError, P0_PA, _nonlinear, _separate_spectra, _sharpness, _sharpness_weight,
)


def test_annex_k_pressure_is_normalized_to_twenty_micropascal():
    # K.1 defines x = p_eff / p_0.  At x=1, the lower branch is 1-a.
    a = 0.5 * (1.0 - 0.2) / (2.0 - 0.2) / 5.66
    result = _nonlinear(np.array([0.0, P0_PA]))
    np.testing.assert_allclose(result, [0.0, 1.0 - a], rtol=1e-13)


def test_annex_k_branches_are_twice_continuously_differentiable():
    exponent, threshold = 0.2, 5.66
    a = (1.0 - exponent) / (2.0 * (2.0 - exponent) * threshold)
    b = threshold ** (1.0 - exponent) / (exponent * (2.0 - exponent))
    c = (1.0 - exponent) * threshold / (2.0 * exponent)
    lower = threshold - a * threshold**2
    upper = b * threshold**exponent - c
    assert lower == pytest.approx(upper, abs=1e-12)
    assert 1 - 2 * a * threshold == pytest.approx(
        b * exponent * threshold ** (exponent - 1), abs=1e-12
    )
    assert -2 * a == pytest.approx(
        b * exponent * (exponent - 1) * threshold ** (exponent - 2),
        abs=1e-12,
    )
    pressure = P0_PA * threshold * np.array([1 - 1e-7, 1.0, 1 + 1e-7])
    values = _nonlinear(pressure)
    assert np.all(np.diff(values) > 0)
    assert values[1] == pytest.approx(lower, abs=1e-12)


def test_annex_k_compression_remains_nonnegative_monotonic_and_compressive():
    # Cover silence through 120 dB SPL without relying on validation vectors.
    pressure = P0_PA * np.r_[0.0, np.logspace(-3, 6, 200)]
    compressed = _nonlinear(pressure)
    assert np.all(np.isfinite(compressed))
    assert np.all(compressed >= 0)
    assert np.all(np.diff(compressed) > 0)
    assert compressed[-1] < pressure[-1] / P0_PA
    ratio = _nonlinear(np.array([1.0, 10.0]))
    assert 1 < ratio[1] / ratio[0] < 2


def test_wiener_separation_matches_known_amplitude_split_for_stationary_noise():
    reference = np.tile([[.01], [.02]], (1, 20))
    degraded = 2 * reference
    speech, noise, details = _separate_spectra(
        reference, degraded, np.ones(2), np.ones_like(reference),
    )
    # S_est=N_est gives W=1/2. The final S+N must equal observed Y,
    # rather than a clipped linear reference and an attenuated residual.
    np.testing.assert_allclose(speech, reference, rtol=1e-12)
    np.testing.assert_allclose(noise, reference, rtol=1e-12)
    np.testing.assert_allclose(speech + noise, degraded, rtol=1e-12)
    assert details["amplitude_identity_max_error"] == 0


def test_noise_reconstruction_interpolates_missing_stationary_observations():
    reference = np.full((2, 40), .01)
    degraded = np.full_like(reference, .02)
    reliability = np.ones_like(reference)
    reliability[:, 10:30] = 0
    speech, noise, details = _separate_spectra(
        reference, degraded, np.ones(2), reliability,
    )
    np.testing.assert_allclose(speech, .01, rtol=1e-10)
    np.testing.assert_allclose(noise, .01, rtol=1e-10)
    assert details["noise_reconstruction"]["all_bands_usable"] is True


def test_zero_residual_retains_exact_speech_and_no_noise_floor():
    reference = np.full((2, 20), .01)
    speech, noise, details = _separate_spectra(
        reference, reference.copy(), np.ones(2), np.ones_like(reference),
    )
    np.testing.assert_array_equal(speech, reference)
    np.testing.assert_array_equal(noise, np.zeros_like(reference))
    assert details["zero_residual_bands"] == 2


def test_noise_reconstruction_rejects_unobserved_bands():
    reference = np.full((1, 20), .01)
    with pytest.raises(ModelAError, match="no observations"):
        _separate_spectra(
            reference, 2 * reference, np.ones(1), np.zeros_like(reference),
        )


def test_sharpness_uses_first_bark_moment_and_is_level_invariant():
    centres = np.array([1000.0, 8000.0])
    bark = 13 * np.arctan(.00076 * centres) + 3.5 * np.arctan((centres / 7500)**2)
    spectrum = np.eye(2)
    expected = bark * _sharpness_weight(centres)
    np.testing.assert_allclose(_sharpness(spectrum, centres), expected)
    np.testing.assert_allclose(_sharpness(10 * spectrum, centres), expected)
    assert expected[1] > expected[0]
    np.testing.assert_array_equal(_sharpness(np.zeros((2, 3)), centres), np.zeros(3))
