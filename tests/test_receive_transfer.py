from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import signal

from server.acoustic.receive_transfer import (
    METHOD,
    MINIMUM_COHERENCE,
    estimate_binaural_transfer,
)


CENTERS = (
    125,
    160,
    200,
    250,
    315,
    400,
    500,
    630,
    800,
    1000,
    1250,
    1600,
    2000,
    2500,
    3150,
    4000,
    5000,
    6300,
    8000,
    10000,
    12500,
    16000,
)
TOP_LEVEL_FIELDS = {
    "available",
    "reason",
    "method",
    "span_db",
    "bands",
    "required_bands_hz",
    "missing_bands_hz",
    "sample_rate_hz",
    "frames",
    "standard_compliance",
    "calibrated",
}
BAND_FIELDS = {
    "input_power",
    "left_gain_db",
    "right_gain_db",
    "binaural_gain_db",
    "left_coherence",
    "right_coherence",
    "valid",
    "reason",
}


def white_noise(rate: int, seconds: float = 2.0, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 0.1 * rng.standard_normal(round(rate * seconds))


def pink_noise(rate: int, seconds: float = 2.0, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = rng.standard_normal(round(rate * seconds))
    frequency = np.fft.rfftfreq(len(samples), 1 / rate)
    spectrum = np.fft.rfft(samples)
    spectrum[0] = 0
    spectrum[1:] /= np.sqrt(frequency[1:])
    pink = np.fft.irfft(spectrum, len(samples))
    return 0.1 * pink / np.std(pink)


def estimate(
    reference: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    rate: int,
    *,
    upper_hz: float,
) -> dict:
    return estimate_binaural_transfer(
        reference,
        left,
        right,
        rate,
        centers_hz=CENTERS,
        upper_hz=upper_hz,
    )


def assert_contract(result: dict) -> None:
    assert set(result) == TOP_LEVEL_FIELDS
    assert result["method"] == METHOD
    assert result["standard_compliance"] is False
    assert result["calibrated"] is False
    assert isinstance(result["bands"], dict)
    for key, band in result["bands"].items():
        assert key == f"{float(key):g}"
        assert set(band) == BAND_FIELDS
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("generator", [white_noise, pink_noise])
def test_identity_is_zero_db_with_near_unit_coherence(generator) -> None:
    reference = generator(48_000)
    result = estimate(reference, reference, reference, 48_000, upper_hz=4000)

    assert_contract(result)
    assert result["available"] is True
    assert result["missing_bands_hz"] == []
    assert result["span_db"] == pytest.approx(0.0, abs=1e-9)
    for band in result["bands"].values():
        assert band["valid"] is True
        assert band["reason"] == "ok"
        assert band["left_gain_db"] == pytest.approx(0.0, abs=1e-9)
        assert band["right_gain_db"] == pytest.approx(0.0, abs=1e-9)
        assert band["binaural_gain_db"] == pytest.approx(0.0, abs=1e-9)
        assert band["left_coherence"] >= 0.999
        assert band["right_coherence"] >= 0.999


def test_constant_half_gain_is_minus_six_db_and_has_zero_span() -> None:
    reference = white_noise(48_000)
    response = 0.5 * reference
    result = estimate(reference, response, response, 48_000, upper_hz=4000)

    assert result["available"] is True
    assert result["span_db"] == pytest.approx(0.0, abs=1e-9)
    for band in result["bands"].values():
        assert band["left_gain_db"] == pytest.approx(-6.020599913, abs=1e-9)
        assert band["right_gain_db"] == pytest.approx(-6.020599913, abs=1e-9)
        assert band["binaural_gain_db"] == pytest.approx(-6.020599913, abs=1e-9)


def test_known_low_pass_produces_a_frequency_span() -> None:
    reference = white_noise(48_000)
    sos = signal.butter(4, 3000, btype="lowpass", fs=48_000, output="sos")
    response = signal.sosfilt(sos, reference)
    result = estimate(reference, response, response, 48_000, upper_hz=8000)

    assert result["available"] is True
    assert result["span_db"] > 30
    assert result["bands"]["125"]["binaural_gain_db"] == pytest.approx(0.0, abs=0.1)
    assert result["bands"]["8000"]["binaural_gain_db"] < -30


def test_inverted_ears_do_not_cancel_in_the_power_mean() -> None:
    reference = white_noise(48_000)
    result = estimate(reference, reference, -reference, 48_000, upper_hz=4000)

    assert result["available"] is True
    assert result["span_db"] == pytest.approx(0.0, abs=1e-9)
    for band in result["bands"].values():
        assert band["left_gain_db"] == pytest.approx(0.0, abs=1e-9)
        assert band["right_gain_db"] == pytest.approx(0.0, abs=1e-9)
        assert band["binaural_gain_db"] == pytest.approx(0.0, abs=1e-9)
        assert band["left_coherence"] >= 0.999
        assert band["right_coherence"] >= 0.999


@pytest.mark.parametrize(
    ("rate", "upper_hz", "count"),
    [
        (16_000, 4000, 16),
        (24_000, 8000, 19),
        (32_000, 14_000, 21),
    ],
)
def test_nb_wb_and_swb_cover_the_expected_bands(
    rate: int,
    upper_hz: float,
    count: int,
) -> None:
    reference = white_noise(rate, seed=rate)
    result = estimate(reference, reference, reference, rate, upper_hz=upper_hz)

    assert result["available"] is True
    assert result["sample_rate_hz"] == rate
    assert len(result["required_bands_hz"]) == count
    assert len(result["bands"]) == count
    assert result["required_bands_hz"][-1] <= upper_hz
    assert result["missing_bands_hz"] == []


def test_independent_noise_is_rejected_by_coherence() -> None:
    reference = white_noise(48_000, seed=1)
    rng = np.random.default_rng(2)
    left = 0.1 * rng.standard_normal(len(reference))
    right = 0.1 * rng.standard_normal(len(reference))
    result = estimate(reference, left, right, 48_000, upper_hz=4000)

    assert_contract(result)
    assert result["available"] is False
    assert result["span_db"] is None
    assert result["missing_bands_hz"] == result["required_bands_hz"]
    for band in result["bands"].values():
        assert band["valid"] is False
        assert band["left_gain_db"] is None
        assert band["right_gain_db"] is None
        assert band["binaural_gain_db"] is None
        assert band["left_coherence"] < MINIMUM_COHERENCE
        assert band["right_coherence"] < MINIMUM_COHERENCE
        assert "coherence" in band["reason"]


def test_single_tone_does_not_fill_unexcited_bands() -> None:
    rate = 48_000
    time = np.arange(rate * 2) / rate
    reference = 0.1 * np.sin(2 * np.pi * 1000 * time)
    result = estimate(reference, reference, reference, rate, upper_hz=4000)

    assert result["available"] is False
    assert result["span_db"] is None
    assert result["bands"]["1000"]["valid"] is True
    assert result["bands"]["125"]["valid"] is False
    assert result["bands"]["125"]["left_gain_db"] is None
    assert result["bands"]["125"]["binaural_gain_db"] is None
    assert "excitation" in result["bands"]["125"]["reason"]


def test_silence_is_unavailable_without_placeholder_zero_gain() -> None:
    silence = np.zeros(48_000, dtype=np.float64)
    result = estimate(silence, silence, silence, 48_000, upper_hz=4000)

    assert_contract(result)
    assert result["available"] is False
    assert result["span_db"] is None
    assert result["missing_bands_hz"] == result["required_bands_hz"]
    assert all(band["valid"] is False for band in result["bands"].values())
    assert all(band["binaural_gain_db"] is None for band in result["bands"].values())


def test_audio_too_short_for_four_segments_is_unavailable() -> None:
    reference = white_noise(48_000, seconds=0.2)
    result = estimate(reference, reference, reference, 48_000, upper_hz=4000)

    assert result["available"] is False
    assert result["span_db"] is None
    assert "Welch segments" in result["reason"]
    assert result["missing_bands_hz"] == result["required_bands_hz"]


def test_band_above_nyquist_makes_the_measurement_unavailable() -> None:
    reference = white_noise(16_000)
    result = estimate(reference, reference, reference, 16_000, upper_hz=8000)

    assert result["available"] is False
    assert result["span_db"] is None
    assert 8000.0 in result["missing_bands_hz"]
    assert "Nyquist" in result["bands"]["8000"]["reason"]
    assert result["bands"]["4000"]["valid"] is True


def test_slightly_different_tail_lengths_use_the_common_prefix() -> None:
    reference = white_noise(48_000)
    left = reference[:-3]
    right = np.concatenate([reference, [0.0, 0.0]])
    result = estimate(reference, left, right, 48_000, upper_hz=4000)

    assert result["available"] is True
    assert result["frames"] == len(left)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reference", np.r_[white_noise(48_000)[:-1], np.nan], "non-finite"),
        ("left", np.r_[white_noise(48_000)[:-1], np.inf], "non-finite"),
        ("right", np.zeros((2, 24_000)), "one-dimensional"),
        ("reference", np.ones(48_000, dtype=bool), "real numeric"),
        ("left", np.ones(48_000, dtype=np.complex128), "real numeric"),
    ],
)
def test_invalid_signal_arrays_are_rejected(
    field: str,
    value: np.ndarray,
    message: str,
) -> None:
    valid = white_noise(48_000)
    arguments = {"reference": valid, "left": valid, "right": valid}
    arguments[field] = value

    with pytest.raises(ValueError, match=message):
        estimate_binaural_transfer(
            arguments["reference"],
            arguments["left"],
            arguments["right"],
            48_000,
            centers_hz=CENTERS,
            upper_hz=4000,
        )


@pytest.mark.parametrize("rate", [True, False, 0, -16_000, 16_000.5, np.nan, np.inf])
def test_invalid_sample_rates_are_rejected(rate) -> None:
    reference = white_noise(16_000)
    with pytest.raises(ValueError, match="rate"):
        estimate_binaural_transfer(
            reference,
            reference,
            reference,
            rate,
            centers_hz=CENTERS,
            upper_hz=4000,
        )


def test_hard_too_short_input_is_rejected() -> None:
    reference = np.ones(32, dtype=np.float64)
    with pytest.raises(ValueError, match="too short"):
        estimate_binaural_transfer(
            reference,
            reference,
            reference,
            48_000,
            centers_hz=CENTERS,
            upper_hz=4000,
        )
