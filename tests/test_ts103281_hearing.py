"""Independent controls for the TS 103 281 hearing front-end."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from server.quality.ts103281_hearing import (
    AUDITORY_BAND_COUNT,
    AUDITORY_FILTER_ORDER,
    AUDITORY_BANDWIDTH_BASE_HZ,
    AUDITORY_BANDWIDTH_SLOPE,
    IEC_A_WEIGHTING_1KHZ_OFFSET_DB,
    OUTER_MIDDLE_EAR_METHOD,
    SAMPLE_RATE_HZ,
    SOURCE_SHA256,
    SOURCE_URLS,
    HearingFrontendError,
    apply_outer_middle_ear_filter,
    auditory_bandpass_signal,
    auditory_bandwidth,
    auditory_filter_parameters,
    auditory_filter_response,
    auditory_spectrum,
    auditory_time_constant,
    iec_a_weighting_linear,
    outer_middle_ear_filter_response,
    outer_middle_ear_response,
)


def _official_centres() -> np.ndarray:
    # TS 103 281 clause 6.3.3: 33 basic and 66 intermediate filters.
    z = np.arange(1.0, 100.0) / 3.0
    return np.minimum(
        (50.0 / 0.14) * np.sinh(0.14 * z),
        20_000.0,
    )


def _tone(
    frequency_hz: float,
    seconds: float,
    amplitude_pa: float = 1.0,
) -> np.ndarray:
    time = np.arange(
        round(SAMPLE_RATE_HZ * seconds),
        dtype=np.float64,
    ) / SAMPLE_RATE_HZ
    return amplitude_pa * np.sin(2.0 * np.pi * frequency_hz * time)


def test_auditory_bandwidth_and_tau_follow_ts_and_sottek():
    centre = 1_000.0
    bandwidth = AUDITORY_BANDWIDTH_BASE_HZ + (
        AUDITORY_BANDWIDTH_SLOPE * centre
    )
    assert auditory_bandwidth(centre) == pytest.approx(bandwidth)
    expected_tau = math.comb(8, 4) / (2**9 * bandwidth)
    assert auditory_time_constant(centre) == pytest.approx(expected_tau)
    parameters = auditory_filter_parameters(centre)
    assert parameters.bandwidth_hz == pytest.approx(bandwidth)
    assert parameters.tau_seconds == pytest.approx(expected_tau)
    assert 0.0 < parameters.pole < 1.0
    assert AUDITORY_FILTER_ORDER == 5


def test_auditory_filter_has_unit_center_gain_and_stable_impulse():
    centre = 1_000.0
    response = auditory_filter_response(centre, np.array([centre]))
    assert abs(response[0]) == pytest.approx(1.0, abs=1e-13)

    impulse = np.zeros(round(SAMPLE_RATE_HZ * 0.1))
    impulse[0] = 1.0
    filtered = auditory_bandpass_signal(impulse, centre)
    assert np.isfinite(filtered).all()
    early = np.sqrt(np.mean(filtered[:480] ** 2))
    late = np.sqrt(np.mean(filtered[-480:] ** 2))
    assert early > late


def test_outer_middle_ear_is_approximate_and_normalized_at_one_khz():
    frequencies = np.array([0.0, 100.0, 500.0, 1_000.0, 4_000.0])
    response = outer_middle_ear_response(frequencies)
    assert response[0] == 0.0
    assert response[3] == pytest.approx(1.0, abs=1e-12)
    assert response[1] < response[2] < response[3]
    actual = np.abs(
        outer_middle_ear_filter_response(np.array([1_000.0]))
    )
    assert actual[0] == pytest.approx(1.0, abs=1e-12)
    assert OUTER_MIDDLE_EAR_METHOD.startswith("zwicker_a0")


def test_one_khz_tone_keeps_rms_after_ear_transfer():
    tone = _tone(1_000.0, 1.0)
    filtered = apply_outer_middle_ear_filter(tone)
    steady_input = tone[round(0.2 * SAMPLE_RATE_HZ) :]
    steady_output = filtered[round(0.2 * SAMPLE_RATE_HZ) :]
    input_rms = float(np.sqrt(np.mean(steady_input**2)))
    output_rms = float(np.sqrt(np.mean(steady_output**2)))
    assert output_rms == pytest.approx(input_rms, rel=1e-10, abs=1e-12)


def test_iec_a_diagnostic_includes_required_plus_two_db_anchor():
    weights = iec_a_weighting_linear(
        np.array([100.0, 1_000.0, 10_000.0])
    )
    assert weights[1] == pytest.approx(1.0, abs=1e-12)
    assert weights[0] < weights[1]
    assert IEC_A_WEIGHTING_1KHZ_OFFSET_DB == 2.0


def test_silence_produces_exactly_zero_hearing_spectrum():
    samples = np.zeros(round(SAMPLE_RATE_HZ * 0.2))
    result = auditory_spectrum(samples, _official_centres())
    assert result.shape == (
        AUDITORY_BAND_COUNT,
        round(SAMPLE_RATE_HZ * 0.2) // round(SAMPLE_RATE_HZ * 0.008),
    )
    assert np.all(result == 0.0)


def test_one_khz_tone_has_nonnegative_finite_pa_output_near_center():
    tone = _tone(1_000.0, 0.4)
    centres = _official_centres()
    result = auditory_spectrum(tone, centres)
    assert result.shape[0] == AUDITORY_BAND_COUNT
    assert result.shape[1] == 50
    assert np.isfinite(result).all()
    assert np.all(result >= 0.0)
    steady_energy = np.mean(result[:, 12:] ** 2, axis=1)
    peak_band = int(np.argmax(steady_energy))
    nearest_1khz = int(np.argmin(np.abs(centres - 1_000.0)))
    assert abs(peak_band - nearest_1khz) <= 2
    assert steady_energy[peak_band] > 0.0


def test_impulse_has_finite_nonnegative_response_without_divergence():
    impulse = np.zeros(round(SAMPLE_RATE_HZ * 0.1))
    impulse[0] = 1.0
    result = auditory_spectrum(impulse, _official_centres())
    assert np.isfinite(result).all()
    assert np.all(result >= 0.0)
    assert np.max(result) < 10.0
    assert np.mean(result[:, -4:] ** 2) < np.mean(result[:, :4] ** 2)


@pytest.mark.parametrize("frequency_hz", [200.0, 8_000.0])
def test_long_low_and_high_tones_reach_steady_finite_output(
    frequency_hz: float,
):
    tone = _tone(frequency_hz, 0.8, amplitude_pa=0.1)
    centres = _official_centres()
    result = auditory_spectrum(tone, centres)
    assert result.shape == (
        AUDITORY_BAND_COUNT,
        round(SAMPLE_RATE_HZ * 0.8) // round(SAMPLE_RATE_HZ * 0.008),
    )
    assert np.isfinite(result).all()
    assert np.all(result >= 0.0)
    nearest = int(np.argmin(np.abs(centres - frequency_hz)))
    steady = result[nearest, -20:]
    assert np.max(steady) > 0.0


@pytest.mark.parametrize(
    ("samples", "centres", "sample_rate", "frame_seconds", "message"),
    [
        (
            np.ones((2, 384)),
            _official_centres(),
            48_000.0,
            0.008,
            "one-dimensional",
        ),
        (
            np.array([]),
            _official_centres(),
            48_000.0,
            0.008,
            "must not be empty",
        ),
        (
            np.array([0.0, np.nan, 0.0]),
            _official_centres(),
            48_000.0,
            0.008,
            "finite",
        ),
        (
            np.ones(384),
            np.arange(98.0),
            48_000.0,
            0.008,
            "exactly 99",
        ),
        (
            np.ones(384),
            np.ones((99, 1)),
            48_000.0,
            0.008,
            "one-dimensional",
        ),
        (
            np.ones(384),
            np.linspace(-1.0, 20_000.0, 99),
            48_000.0,
            0.008,
            "positive",
        ),
        (
            np.ones(384),
            _official_centres(),
            0.0,
            0.008,
            "positive",
        ),
        (
            np.ones(384),
            _official_centres(),
            48_000.0,
            0.0,
            "positive",
        ),
        (
            np.ones(383),
            _official_centres(),
            48_000.0,
            0.008,
            "too short",
        ),
    ],
)
def test_invalid_inputs_are_rejected(
    samples: np.ndarray,
    centres: np.ndarray,
    sample_rate: float,
    frame_seconds: float,
    message: str,
):
    with pytest.raises(HearingFrontendError, match=message):
        auditory_spectrum(
            samples,
            centres,
            sample_rate,
            frame_seconds,
        )


def test_verification_artifact_records_sources_and_approximations(
    tmp_path: Path,
):
    artifact = {
        "contract": {
            "function": "auditory_spectrum",
            "input": "mono pressure in Pa",
            "output_shape": [AUDITORY_BAND_COUNT, "8ms frames"],
            "centres": "externally supplied, exactly 99",
            "bandwidth_hz": "50 + 0.14 * centre_hz",
            "auditory_order": AUDITORY_FILTER_ORDER,
            "envelope_lowpass_hz": 120.0,
        },
        "outer_middle_ear": {
            "method": OUTER_MIDDLE_EAR_METHOD,
            "approximate": True,
            "normalization_hz": 1_000.0,
            "sottek_a0_and_ecma_418_table_are_identical": False,
            "iec_a_raw_formula_1khz_offset_db": (
                IEC_A_WEIGHTING_1KHZ_OFFSET_DB
            ),
        },
        "sources": SOURCE_URLS,
        "source_sha256": SOURCE_SHA256,
        "annex_c_fitted_constants_used": False,
    }
    path = tmp_path / "hearing-contract.json"
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8")) == artifact
