"""Independent physical-signal controls for the digital preflight tool."""
from pathlib import Path
import importlib.util

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "ecnr_preflight", Path(__file__).resolve().parents[1]/"cpp/tools/ecnr_preflight.py")
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


def broadband_pair(fs=16000, delay_ms=11.371, seconds=1.0, seed=94):
    # Evaluate a continuous analytic waveform at shifted times; does not use
    # estimator FFT, resampling, or delay interpolation to create the truth.
    rng = np.random.default_rng(seed)
    frequencies = rng.uniform(250, fs*0.37, 173)
    phases = rng.uniform(-np.pi, np.pi, len(frequencies))
    times = np.arange(int(fs*seconds))/fs
    def evaluate(t):
        return np.sum(np.cos(2*np.pi*frequencies[:, None]*t+phases[:, None]),
                      axis=0)*0.01
    return evaluate(times), evaluate(times-delay_ms/1000), evaluate, times


def array_pair(geometry="endfire", fs=16000, gain=1.0, polarity=1):
    times = np.arange(fs)/fs
    def evaluate(t):
        return 0.16*np.cos(2*np.pi*1000*t+0.8)+0.09*np.sin(2*np.pi*2000*t-0.2)
    delay_s = 0.020/343.0 if geometry == "endfire" else 0.0
    return evaluate(times), polarity*gain*evaluate(times-delay_s)


@pytest.mark.parametrize("fs", [16000, 48000])
def test_known_fractional_delay_in_ms_and_samples(fs):
    x, y, _, _ = broadband_pair(fs=fs, delay_ms=11.371)
    result = preflight.estimate_delay(x, y, fs, context="digital_calibration")
    assert result["accepted"], result
    assert result["delay_ms"] == pytest.approx(11.371, abs=0.012)
    assert result["delay_samples"] == pytest.approx(11.371*fs/1000, abs=0.5)
    assert result["input_sha256"]["reference"] != result["input_sha256"]["recorded"]


def test_negative_delay_sign():
    x, y, _, _ = broadband_pair(delay_ms=-2.313)
    result = preflight.estimate_delay(x, y, 16000, context="digital_calibration")
    assert result["accepted"]
    assert result["delay_ms"] == pytest.approx(-2.313, abs=0.012)


def test_calibration_declaration_is_required():
    x, y, _, _ = broadband_pair()
    result = preflight.estimate_delay(x, y, 16000)
    assert not result["accepted"]
    assert "context_is_not_declared_digital_calibration" in result["rejection_reasons"]


def test_periodic_single_tone_is_not_a_reliable_reference():
    t = np.arange(16000)/16000
    x = 0.2*np.sin(2*np.pi*1000*t)
    y = 0.2*np.sin(2*np.pi*1000*(t-0.013))
    result = preflight.estimate_delay(x, y, 16000, context="digital_calibration")
    assert not result["accepted"]
    assert "insufficient_broadband_support" in result["rejection_reasons"]


def test_unrelated_content_and_double_talk_are_rejected():
    x, y, _, _ = broadband_pair()
    rng = np.random.default_rng(901)
    unrelated = rng.normal(0, np.std(y)*3, len(y))
    result = preflight.estimate_delay(x, y+unrelated, 16000,
                                      context="digital_calibration")
    assert not result["accepted"]
    assert "weak_aligned_correlation_or_unmodelled_content" in result["rejection_reasons"]
    declared_double_talk = preflight.estimate_delay(x, y, 16000, context="double_talk")
    assert not declared_double_talk["accepted"]


def test_delay_polarity_reversal_is_explicitly_rejected():
    x, y, _, _ = broadband_pair()
    result = preflight.estimate_delay(x, -y, 16000, context="digital_calibration")
    assert not result["accepted"]
    assert "polarity_reversal" in result["rejection_reasons"]
    assert result["signed_aligned_correlation"] < -0.8


def test_seven_records_keep_bulk_value_separate_from_drift_and_spread():
    x, _, source, times = broadband_pair(seconds=1.25)
    actual = np.array([12.331, 12.341, 12.336, 12.333, 12.338, 12.331, 12.336])
    recorded = np.stack([source(times-delay/1000) for delay in actual])
    result = preflight.audit_bulk_delay(x, recorded, 16000, context="digital_calibration")
    assert result["accepted"], result
    assert result["median_bulk_delay_ms"] == pytest.approx(np.median(actual), abs=0.02)
    assert result["between_record_delay_spread_ms"] < 0.04
    assert max(row["within_record_span_ms"] for row in result["records"]) < 0.05


def test_within_record_jump_rejected_even_when_seven_medians_match():
    x, y, source, times = broadband_pair(seconds=1.25, delay_ms=12.5)
    recorded = np.tile(y, (7, 1))
    recorded[2, len(times)//2:] = source(times[len(times)//2:]-0.0165)
    result = preflight.audit_bulk_delay(x, recorded, 16000, context="digital_calibration")
    assert not result["accepted"]
    assert "within_record_delay_not_stable" in result["rejection_reasons"]
    assert result["records"][2]["within_record_span_ms"] > 3.5
    assert result["records"][0]["within_record_stable"]


def test_between_reconnect_shift_is_not_within_record_drift():
    x, y, source, times = broadband_pair(seconds=1.25, delay_ms=12.5)
    recorded = np.tile(y, (7, 1))
    recorded[5] = source(times-0.0145)
    result = preflight.audit_bulk_delay(x, recorded, 16000, context="digital_calibration")
    assert not result["accepted"]
    assert "between_record_delay_not_consistent" in result["rejection_reasons"]
    assert all(row["within_record_stable"] for row in result["records"])
    assert result["between_record_delay_spread_ms"] == pytest.approx(2.0, abs=0.03)


def test_six_records_cannot_pass_the_seven_record_protocol():
    x, y, _, _ = broadband_pair(seconds=1.25)
    result = preflight.audit_bulk_delay(x, np.tile(y, (6, 1)), 16000,
                                        context="digital_calibration")
    assert not result["accepted"]
    assert "requires_exactly_seven_records" in result["rejection_reasons"]


@pytest.mark.parametrize("geometry", ["endfire", "broadside"])
def test_20mm_geometry_against_independent_physical_phase(geometry):
    x, y = array_pair(geometry=geometry)
    result = preflight.audit_array_phase(x, y, 16000, geometry=geometry,
                                         context="digital_calibration")
    assert result["accepted"], result
    expected_1k = -20.991253644314867 if geometry == "endfire" else 0
    assert result["tones"][0]["measured_phase_deg"] == pytest.approx(expected_1k, abs=0.01)
    assert result["tones"][1]["measured_phase_deg"] == pytest.approx(2*expected_1k, abs=0.01)
    assert all(abs(row["mic2_relative_gain_db"]) < 0.001 for row in result["tones"])


def test_array_polarity_is_distinguished_from_gain():
    x, y = array_pair(polarity=-1)
    result = preflight.audit_array_phase(x, y, 16000, geometry="endfire",
                                         context="digital_calibration")
    assert not result["accepted"]
    assert result["polarity_reversed"]
    assert "polarity_reversal" in result["rejection_reasons"]
    assert "gain_mismatch" not in result["rejection_reasons"]


def test_array_gain_is_distinguished_from_polarity():
    x, y = array_pair(gain=2)
    result = preflight.audit_array_phase(x, y, 16000, geometry="endfire",
                                         context="digital_calibration")
    assert not result["accepted"]
    assert not result["polarity_reversed"]
    assert "gain_mismatch" in result["rejection_reasons"]
    assert all(row["mic2_relative_gain_db"] == pytest.approx(6.0206, abs=0.001)
               for row in result["tones"])


def test_array_missing_tone_and_residual_content_are_rejected():
    t = np.arange(16000)/16000
    x = 0.2*np.sin(2*np.pi*1000*t)
    result = preflight.audit_array_phase(x, x, 16000, geometry="broadside",
                                         context="digital_calibration")
    assert not result["accepted"]
    assert "missing_2000hz_calibration_tone" in result["rejection_reasons"]
    a, b = array_pair()
    b = b+np.random.default_rng(91).normal(0, 0.1, len(b))
    noisy = preflight.audit_array_phase(a, b, 16000, geometry="endfire",
                                        context="digital_calibration")
    assert not noisy["accepted"]
    assert "excess_residual_content_or_unmodelled_signal" in noisy["rejection_reasons"]


@pytest.mark.parametrize("bad", [np.zeros(1024), np.full(1024, np.nan)])
def test_invalid_silent_nonfinite_inputs_do_not_produce_a_pass(bad):
    with pytest.raises(ValueError):
        preflight.estimate_delay(bad, bad, 16000, context="digital_calibration")
