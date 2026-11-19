from __future__ import annotations

import numpy as np
import pytest

from server.quality.ts103281_spectral import (
    LTASS_COMBINED_DB_SPL, LTASS_FREQUENCIES_HZ,
    calibrate_reference, correct_reference_gain, ltass_activity_mask,
)


def test_h1_calibrates_known_scalar_without_using_expected_mos():
    rng = np.random.default_rng(281)
    x = rng.normal(0, .04, 48_000)
    scaled, shape, details = calibrate_reference(
        x, .25 * x, np.ones(len(x), dtype=bool), np.geomspace(50, 20_000, 99),
    )
    np.testing.assert_allclose(scaled, .25 * x, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(shape, 1.0, atol=1e-12)
    assert details["initial_gain_linear"] == pytest.approx(.25)


def test_h1_ignores_inactive_output_corruption():
    rng = np.random.default_rng(9)
    x = rng.normal(0, .04, 48_000)
    active = np.arange(len(x)) < 24_000
    y = .5 * x
    y[~active] = 10
    _, _, details = calibrate_reference(x, y, active, np.geomspace(50, 20_000, 99))
    assert details["initial_gain_linear"] == pytest.approx(.5)


def test_ltass_uses_combined_table_and_preserves_common_energy():
    assert LTASS_FREQUENCIES_HZ.shape == (25,)
    assert LTASS_COMBINED_DB_SPL[21] == 43.7
    centres = np.geomspace(50, 20_000, 99)
    spectrum = np.ones((99, 5)) * .001
    _, details = ltass_activity_mask(spectrum, centres, -30.0)
    power = 10 ** (np.asarray(details["ltass_band_levels_db"]) / 10)
    assert np.mean(power.reshape(33, 3), axis=1).sum() == pytest.approx(.001)


def test_spectral_mask_has_all_five_normative_weights():
    centres = np.geomspace(50, 20_000, 99)
    _, details = ltass_activity_mask(np.zeros((99, 5)), centres, -30)
    level = np.asarray(details["ltass_band_levels_db"])
    spectrum = 10 ** ((level[:, None] + np.array([-50, -25, -13, -5, 0])) / 20)
    mask, _ = ltass_activity_mask(spectrum, centres, -30)
    np.testing.assert_allclose(mask, np.tile([1, .4, .2, 0, 0], (99, 1)))


def test_iterative_gain_reaches_thirty_percent_without_mutating_inputs():
    x = np.ones((10, 100)) * .01
    s = x * np.linspace(.1, .9, 100)[None, :]
    before = x.copy()
    scaled, details = correct_reference_gain(x, s, np.ones(100, dtype=bool))
    assert abs(details["proportion_above"] - .30) <= .01
    assert details["initial_step_db"] == 3
    np.testing.assert_array_equal(x, before)
    assert np.all(scaled < x)


def test_iterative_gain_ignores_inactive_time():
    x = np.ones((3, 100)) * .01
    s = x * np.linspace(.1, .9, 100)
    active = np.arange(100) < 50
    first, _ = correct_reference_gain(x, s, active)
    s[:, ~active] = 100
    second, _ = correct_reference_gain(x, s, active)
    np.testing.assert_array_equal(first, second)


def test_zero_activity_is_rejected_instead_of_generating_level():
    with pytest.raises(ValueError, match="active"):
        correct_reference_gain(np.ones((2, 5)), np.ones((2, 5)), np.zeros(5, bool))


def test_unreachable_gain_target_is_rejected_instead_of_hitting_guard():
    with pytest.raises(ValueError, match="unreachable"):
        correct_reference_gain(np.ones((2, 5)), np.zeros((2, 5)), np.ones(5, bool))
