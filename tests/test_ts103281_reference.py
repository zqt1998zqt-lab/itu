from __future__ import annotations

import numpy as np
import pytest

from server.quality import ts103281_reference as reference_model


def smooth_time_reference(bands: int = 4, frames: int = 12) -> np.ndarray:
    band_axis = np.arange(bands, dtype=np.float64)[:, None]
    time_axis = np.arange(frames, dtype=np.float64)[None, :]
    return 1.0 + 0.2 * band_axis + 0.1 * time_axis


def test_linear_upsample_zeroes_outside_support_but_preserves_grid_points():
    values = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ]
    )

    time_up = reference_model._linear_upsample(values, 4, axis=1)
    band_up = reference_model._linear_upsample(values, 4, axis=0)

    np.testing.assert_array_equal(time_up[:, 0::4], values)
    np.testing.assert_array_equal(band_up[0::4, :], values)
    np.testing.assert_array_equal(
        time_up[:, 13:],
        np.zeros((2, 3)),
    )
    np.testing.assert_array_equal(
        band_up[5:, :],
        np.zeros((3, 4)),
    )
    # The exact final sample remains at its original grid coordinate.
    np.testing.assert_array_equal(time_up[:, 12], values[:, -1])
    np.testing.assert_array_equal(band_up[4, :], values[-1, :])


def test_identity_is_exact_and_diagnostics_do_not_claim_masking(tmp_path):
    reference = np.array(
        [
            [0.1, 0.2, 0.4, 0.3, 0.2],
            [0.2, 0.3, 0.5, 0.4, 0.3],
            [0.4, 0.5, 0.7, 0.6, 0.5],
        ]
    )
    degraded = reference.copy()

    optimized, returned_degraded, diagnostics = (
        reference_model.optimize_reference(reference, degraded)
    )

    np.testing.assert_allclose(optimized, reference, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(returned_degraded, degraded)
    assert not np.shares_memory(optimized, reference)
    assert not np.shares_memory(returned_degraded, degraded)
    assert set(diagnostics["steps"]) == {
        "group_delay",
        "spectral_compression",
        "variable_delay",
    }
    assert np.all(
        diagnostics["steps"]["group_delay"]["per_band_lag_upsampled"]
        == 0
    )
    assert np.all(
        diagnostics["steps"]["spectral_compression"][
            "per_frame_shift_subbands"
        ]
        == 0
    )
    assert np.all(
        diagnostics["steps"]["variable_delay"][
            "per_frame_shift_subframes"
        ]
        == 0
    )
    assert diagnostics["complete_reference_model"] is False
    assert diagnostics["status"] == "partial_reference_optimization"
    assert diagnostics["unimplemented_steps"] == [
        {
            "clause": "6.3.6.3",
            "name": "Masking of inaudible differences",
            "implemented": False,
        }
    ]
    assert "never wrapped" in diagnostics["boundary"]
    assert diagnostics["interpolation"]["time"] == "linear"
    assert diagnostics["interpolation"]["frequency"] == "linear"


def test_group_delay_compensates_large_per_band_shift_without_wrap():
    rng = np.random.default_rng(103281)
    reference = 0.05 + rng.random((3, 24))
    delay = 7
    degraded = np.zeros_like(reference)
    degraded[:, delay:] = reference[:, :-delay]

    aligned, diagnostics = reference_model.compensate_group_delay(
        reference,
        degraded,
    )

    np.testing.assert_array_equal(
        diagnostics["per_band_lag_upsampled"],
        np.full(3, delay * 4),
    )
    np.testing.assert_allclose(aligned, degraded, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(
        aligned[:, :delay],
        np.zeros((3, delay)),
    )
    # The reference tail is cropped, not wrapped into the leading zeros.
    assert not np.any(np.isclose(aligned[:, 0], reference[:, -1]))
    assert diagnostics["circular"] is False
    assert diagnostics["boundary"] == "zero_fill"
    assert diagnostics["interpolation"] == "linear"


def test_group_delay_compensates_negative_band_shift():
    rng = np.random.default_rng(812)
    reference = 0.1 + rng.random((2, 20))
    shift = 4
    degraded = np.zeros_like(reference)
    degraded[:, :-shift] = reference[:, shift:]

    aligned, diagnostics = reference_model.compensate_group_delay(
        reference,
        degraded,
    )

    np.testing.assert_array_equal(
        diagnostics["per_band_lag_upsampled"],
        np.full(2, -shift * 4),
    )
    np.testing.assert_allclose(aligned, degraded, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(
        aligned[:, -shift:],
        np.zeros((2, shift)),
    )


def test_spectral_compression_selects_known_half_band_shift_without_wrap():
    profile = np.zeros(9, dtype=np.float64)
    profile[2:7] = [1.0, 1.2, 1.4, 1.6, 1.8]
    reference = np.repeat(profile[:, None], 6, axis=1)
    reference_up = reference_model._linear_upsample(
        reference,
        4,
        axis=0,
    )
    shifted_up = reference_model._shift_zero_fill(
        reference_up,
        2,
        axis=0,
    )
    degraded = reference_model._downsample(
        shifted_up,
        4,
        axis=0,
    )

    compensated, diagnostics = (
        reference_model.compensate_spectral_compression(
            reference,
            degraded,
        )
    )

    np.testing.assert_array_equal(
        diagnostics["per_frame_shift_subbands"],
        np.full(reference.shape[1], 2),
    )
    np.testing.assert_allclose(
        compensated,
        degraded,
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        compensated[:2],
        np.zeros((2, reference.shape[1])),
    )
    assert np.all(
        np.abs(diagnostics["per_frame_shift_subbands"]) <= 3
    )
    assert diagnostics["circular"] is False


def test_variable_delay_selects_known_half_sample_shift_without_wrap():
    reference = smooth_time_reference(bands=3, frames=12)
    degraded = np.zeros_like(reference)
    degraded[:, 1:] = 0.5 * (
        reference[:, 1:] + reference[:, :-1]
    )

    compensated, diagnostics = reference_model.compensate_variable_delay(
        reference,
        degraded,
    )

    np.testing.assert_array_equal(
        diagnostics["per_frame_shift_subframes"][1:],
        np.full(reference.shape[1] - 1, 2),
    )
    np.testing.assert_allclose(
        compensated,
        degraded,
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        compensated[:, 0],
        np.zeros(reference.shape[0]),
    )
    assert np.all(
        np.abs(diagnostics["per_frame_shift_subframes"]) <= 3
    )
    assert diagnostics["circular"] is False


def test_zero_input_uses_zero_shifts_and_never_wraps():
    reference = np.zeros((4, 9))
    degraded = np.zeros_like(reference)

    optimized, returned_degraded, diagnostics = (
        reference_model.optimize_reference(reference, degraded)
    )

    np.testing.assert_array_equal(optimized, np.zeros_like(reference))
    np.testing.assert_array_equal(returned_degraded, degraded)
    assert np.all(
        diagnostics["steps"]["group_delay"]["per_band_lag_upsampled"]
        == 0
    )
    assert np.all(
        diagnostics["steps"]["spectral_compression"][
            "per_frame_shift_subbands"
        ]
        == 0
    )
    assert np.all(
        diagnostics["steps"]["variable_delay"][
            "per_frame_shift_subframes"
        ]
        == 0
    )


@pytest.mark.parametrize(
    ("reference", "degraded"),
    [
        (np.ones((3, 4)), np.ones(12)),
        (np.ones((3, 4)), np.ones((4, 3))),
        (-np.ones((3, 4)), np.ones((3, 4))),
        (np.ones((3, 4)), -np.ones((3, 4))),
        (
            np.array([[1.0, np.nan], [1.0, 1.0]]),
            np.ones((2, 2)),
        ),
        (
            np.ones((2, 2)),
            np.array([[1.0, np.inf], [1.0, 1.0]]),
        ),
        (np.empty((0, 4)), np.empty((0, 4))),
    ],
)
def test_invalid_inputs_fail_before_optimization(reference, degraded):
    with pytest.raises(ValueError):
        reference_model.optimize_reference(reference, degraded)


def test_invalid_search_parameters_fail_explicitly():
    reference = np.ones((2, 4))
    with pytest.raises(ValueError, match="factor"):
        reference_model.compensate_group_delay(
            reference,
            reference,
            factor=0,
        )
    with pytest.raises(ValueError, match="radius"):
        reference_model.compensate_spectral_compression(
            reference,
            reference,
            local_shift=-1,
        )


def test_optimize_does_not_mutate_inputs_and_returns_detached_degraded():
    reference = smooth_time_reference(bands=5, frames=10)
    degraded = reference.copy()
    degraded[:, 1:] *= 0.97
    original_reference = reference.copy()
    original_degraded = degraded.copy()

    optimized, returned_degraded, _ = reference_model.optimize_reference(
        reference,
        degraded,
    )

    np.testing.assert_array_equal(reference, original_reference)
    np.testing.assert_array_equal(degraded, original_degraded)
    np.testing.assert_array_equal(returned_degraded, original_degraded)
    assert not np.shares_memory(optimized, reference)
    assert not np.shares_memory(returned_degraded, degraded)
    assert np.all(optimized >= 0)


def test_optimize_runs_the_three_steps_in_standard_order(monkeypatch):
    calls: list[str] = []
    reference = np.ones((2, 4))
    degraded = np.ones((2, 4))

    def group_step(values, comparison, *, factor):
        calls.append("group_delay")
        assert factor == 4
        return values.copy(), {"name": "group_delay"}

    def frequency_step(
        values,
        comparison,
        *,
        factor,
        local_shift,
    ):
        calls.append("spectral_compression")
        assert factor == 4
        assert local_shift == 3
        return values.copy(), {"name": "spectral_compression"}

    def variable_step(
        values,
        comparison,
        *,
        factor,
        local_shift,
    ):
        calls.append("variable_delay")
        assert factor == 4
        assert local_shift == 3
        return values.copy(), {"name": "variable_delay"}

    monkeypatch.setattr(
        reference_model,
        "_group_delay_validated",
        group_step,
    )
    monkeypatch.setattr(
        reference_model,
        "_spectral_compression_validated",
        frequency_step,
    )
    monkeypatch.setattr(
        reference_model,
        "_variable_delay_validated",
        variable_step,
    )

    optimized, returned_degraded, diagnostics = (
        reference_model.optimize_reference(reference, degraded)
    )

    assert calls == [
        "group_delay",
        "spectral_compression",
        "variable_delay",
    ]
    np.testing.assert_array_equal(optimized, reference)
    np.testing.assert_array_equal(returned_degraded, degraded)
    assert diagnostics["steps"]["group_delay"]["name"] == "group_delay"
    assert (
        diagnostics["steps"]["spectral_compression"]["name"]
        == "spectral_compression"
    )
    assert (
        diagnostics["steps"]["variable_delay"]["name"]
        == "variable_delay"
    )
