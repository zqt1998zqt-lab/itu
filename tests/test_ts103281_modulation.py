"""Independent controls for TS 103 281 modulation feature extraction."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from server.quality.ts103281_modulation import (
    MODULATION_CENTRES_HZ,
    NUMERIC_FLOOR,
    SOURCE_URLS,
    TIME_UPSAMPLE_FACTOR,
    ModulationFeatureError,
    modulation_features,
    modulation_filter_coefficients,
    modulation_filter_response,
    modulation_spectrum,
    modified_a_weighting_linear,
    upsample_modulation_time,
)


def _band_centres() -> np.ndarray:
    return np.linspace(100.0, 8_000.0, 5)


def _reference_spectrum() -> np.ndarray:
    time = np.arange(64, dtype=np.float64)
    bands = np.array([0.2, 0.4, 0.7, 0.5, 0.3])[:, None]
    modulation = 1.0 + 0.25 * np.sin(2.0 * np.pi * time / 12.0)
    return bands * modulation[None, :]


def test_default_centres_are_eight_logarithmic_falk2010_bands():
    assert MODULATION_CENTRES_HZ.shape == (8,)
    assert MODULATION_CENTRES_HZ[0] == pytest.approx(4.0)
    assert MODULATION_CENTRES_HZ[-1] == pytest.approx(128.0)
    np.testing.assert_allclose(
        MODULATION_CENTRES_HZ,
        np.geomspace(4.0, 128.0, 8),
        rtol=0.0,
        atol=0.0,
    )


def test_filter_has_unit_center_response_and_rejects_dc():
    frequencies = np.array([0.0, 4.0, 128.0])
    low_response = modulation_filter_response(4.0, frequencies)
    high_response = modulation_filter_response(128.0, frequencies)

    assert abs(low_response[0]) < 1e-14
    assert abs(low_response[1]) == pytest.approx(1.0, abs=1e-13)
    assert abs(high_response[0]) < 1e-14
    assert abs(high_response[2]) == pytest.approx(1.0, abs=1e-13)


def test_filter_coefficients_match_srmr_bilinear_formula():
    centre = 128.0
    sample_rate = 500.0
    q = 2.0
    w0 = 2.0 * np.pi * centre / sample_rate
    w0_tangent = np.tan(w0 / 2.0)
    bandwidth = w0_tangent / q
    expected_b = np.array([bandwidth, 0.0, -bandwidth])
    expected_a = np.array(
        [
            1.0 + bandwidth + w0_tangent**2,
            2.0 * w0_tangent**2 - 2.0,
            1.0 - bandwidth + w0_tangent**2,
        ]
    )
    expected_b /= expected_a[0]
    expected_a /= expected_a[0]

    numerator, denominator = modulation_filter_coefficients(
        centre,
        sample_rate,
        q,
    )
    np.testing.assert_allclose(numerator, expected_b, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(denominator, expected_a, rtol=0.0, atol=0.0)


def test_time_upsampling_preserves_factor_and_shape():
    values = _reference_spectrum()
    upsampled = upsample_modulation_time(values)
    assert upsampled.shape == (
        values.shape[0],
        TIME_UPSAMPLE_FACTOR * values.shape[1],
    )
    spectrum = modulation_spectrum(values, _band_centres())
    assert spectrum.shape == (values.shape[0], 8)
    assert np.isfinite(spectrum).all()
    assert np.all(spectrum >= 0.0)


def test_a_weighting_clips_at_minus_ten_db_and_is_linear():
    frequencies = np.array([4.0, 128.0, 1_000.0])
    weights = modified_a_weighting_linear(frequencies)
    assert weights[0] == pytest.approx(10 ** (-10.0 / 20.0))
    assert weights[1] == pytest.approx(10 ** (-10.0 / 20.0))
    # IEC 61672 A-weighting is normalized to 0 dB at 1 kHz.
    assert weights[2] == pytest.approx(1.0, abs=2e-4)


def test_zero_reference_band_cannot_manufacture_a_huge_mmer():
    reference = _reference_spectrum()
    reference[0] = 0
    first = reference.copy()
    second = reference.copy()
    second[0] = 100 * _reference_spectrum()[0]
    one = modulation_features(reference, first, _band_centres())
    two = modulation_features(reference, second, _band_centres())
    assert one["mmer"] == pytest.approx(two["mmer"], abs=1e-12)
    assert two["weighted_difference"] > one["weighted_difference"]


def test_no_reference_modulation_is_explicitly_unmeasurable():
    with pytest.raises(ModulationFeatureError, match="nonzero reference"):
        modulation_features(np.zeros((5, 64)), _reference_spectrum(), _band_centres())


def test_identical_inputs_use_summed_ratio_and_machine_epsilon_dme_floor():
    reference = _reference_spectrum()
    result = modulation_features(
        reference,
        reference.copy(),
        _band_centres(),
    )

    expected_mmer = float(
        20.0
        * np.log10(
            np.sum(modified_a_weighting_linear(_band_centres()))
        )
    )
    expected_floor = float(20.0 * np.log10(NUMERIC_FLOOR))
    assert result["mmer"] == pytest.approx(expected_mmer, abs=1e-12)
    assert result["mmer"] != pytest.approx(0.0, abs=1e-6)
    assert result["weighted_difference"] == pytest.approx(
        expected_floor,
        abs=1e-12,
    )


def test_output_change_is_a_negative_control_for_both_features():
    reference = _reference_spectrum()
    unchanged = modulation_features(
        reference,
        reference.copy(),
        _band_centres(),
    )
    changed = reference.copy()
    changed[2] = 0.55 * changed[2] + 0.03 * np.cos(
        np.arange(changed.shape[1]) * np.pi / 7.0
    )
    result = modulation_features(reference, changed, _band_centres())

    assert result["weighted_difference"] > unchanged["weighted_difference"]
    assert result["mmer"] != pytest.approx(unchanged["mmer"], abs=1e-9)
    assert np.isfinite(list(result.values())).all()


@pytest.mark.parametrize(
    ("reference", "speech", "centres", "frame_seconds", "message"),
    [
        (
            np.ones((3, 16)),
            np.ones((4, 16)),
            np.arange(1.0, 4.0),
            0.008,
            "same shape",
        ),
        (
            np.ones(16),
            np.ones(16),
            np.array([1.0]),
            0.008,
            "bands, time",
        ),
        (
            np.full((3, 16), -1.0),
            np.ones((3, 16)),
            np.arange(1.0, 4.0),
            0.008,
            "non-negative",
        ),
        (
            np.full((3, 16), np.nan),
            np.ones((3, 16)),
            np.arange(1.0, 4.0),
            0.008,
            "finite",
        ),
        (
            np.ones((3, 16)),
            np.ones((3, 16)),
            np.arange(1.0, 4.0)[:, None],
            0.008,
            "one-dimensional",
        ),
        (
            np.ones((3, 16)),
            np.ones((3, 16)),
            np.array([1.0, 2.0]),
            0.008,
            "one frequency",
        ),
        (
            np.ones((3, 16)),
            np.ones((3, 16)),
            np.arange(1.0, 4.0),
            0.0,
            "positive",
        ),
        (
            np.ones((3, 16)),
            np.ones((3, 16)),
            np.arange(1.0, 4.0),
            0.02,
            "Nyquist",
        ),
    ],
)
def test_invalid_contract_inputs_are_rejected(
    reference: np.ndarray,
    speech: np.ndarray,
    centres: np.ndarray,
    frame_seconds: float,
    message: str,
):
    with pytest.raises(ModulationFeatureError, match=message):
        modulation_features(
            reference,
            speech,
            centres,
            frame_seconds,
        )


def test_verification_artifact_records_sources_and_contract(tmp_path: Path):
    artifact = {
        "contract": {
            "function": "modulation_features",
            "return_keys": ["mmer", "weighted_difference"],
            "input_shape": ["bands", "time"],
            "time_upsample_factor": TIME_UPSAMPLE_FACTOR,
            "modulation_centres_hz": MODULATION_CENTRES_HZ.tolist(),
            "numeric_floor": NUMERIC_FLOOR,
        },
        "sources": SOURCE_URLS,
        "annex_c_fitted_constants_used": False,
    }
    path = tmp_path / "modulation-contract.json"
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8")) == artifact
