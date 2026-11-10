from io import BytesIO
import json
import re
from pathlib import Path

import numpy as np
import pytest
from scipy import signal
import soundfile as sf
from fastapi.testclient import TestClient

from server.app import app
from server.acoustic import receive_simulation as simulation


def wav(
    samples: np.ndarray,
    rate: int = 16000,
    subtype: str = "PCM_16",
) -> bytes:
    stream = BytesIO()
    sf.write(stream, samples, rate, format="WAV", subtype=subtype)
    return stream.getvalue()


def impulse_signal(rate: int = 16000) -> np.ndarray:
    samples = np.zeros(rate // 10, dtype=np.float64)
    samples[80:84] = np.array([0.20, 0.12, -0.08, 0.05])
    return samples


@pytest.fixture(autouse=True)
def isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    store = tmp_path / "synthetic-receive"
    monkeypatch.setattr(simulation, "STORE", store)
    return store


def test_head_model_is_deterministic_and_right_source_is_right_louder(
    isolated_store: Path,
) -> None:
    audio = wav(impulse_signal())
    first = simulation.evaluate_synthetic_receive(
        "WB-wireless",
        "speaker.wav",
        audio,
        azimuth_deg=30.0,
    )
    second = simulation.evaluate_synthetic_receive(
        "WB-wireless",
        "speaker.wav",
        audio,
        azimuth_deg=30.0,
    )

    assert first["evidence_kind"] == "synthetic_head"
    assert first["simulation"]["head_model"]["source_side"] == "right"
    assert first["metrics"]["right_rms_dbfs"] > first["metrics"]["left_rms_dbfs"]
    assert first["metrics"]["interaural_level_difference_db"] < 0
    assert first["metrics"]["interaural_delay_ms"] > 0
    assert first["metrics"]["interaural_delay_ms"] == pytest.approx(
        (0.0875 / 343.0) * (np.deg2rad(30) + np.sin(np.deg2rad(30))) * 1000,
        abs=1e-6,
    )
    assert first["artifacts"]["left"]["sha256"] == second["artifacts"]["left"]["sha256"]
    assert first["artifacts"]["right"]["sha256"] == second["artifacts"]["right"]["sha256"]
    assert first["metrics"] == second["metrics"]

    left_path = simulation.get_synthetic_receive_file(
        first["evaluation_id"], "virtual_left.wav"
    )
    right_path = simulation.get_synthetic_receive_file(
        first["evaluation_id"], "virtual_right.wav"
    )
    assert left_path is not None and right_path is not None
    left, left_rate = sf.read(left_path, dtype="float64")
    right, right_rate = sf.read(right_path, dtype="float64")
    assert left_rate == right_rate == 16000
    assert not np.array_equal(left, right)
    assert np.max(np.abs(right)) > np.max(np.abs(left))


def test_rear_azimuth_folds_to_front_and_mirror_outputs_are_exact() -> None:
    rate = 16_000
    speaker = impulse_signal(rate)
    radius = 0.0875
    speed = 343.0

    left_60, right_60, metadata_60 = simulation._synthetic_head_outputs(
        speaker,
        rate,
        60.0,
        radius,
        speed,
    )
    left_minus_60, right_minus_60, metadata_minus_60 = (
        simulation._synthetic_head_outputs(
            speaker,
            rate,
            -60.0,
            radius,
            speed,
        )
    )
    assert np.array_equal(left_60, right_minus_60)
    assert np.array_equal(right_60, left_minus_60)
    assert metadata_60["source_side"] == "right"
    assert metadata_minus_60["source_side"] == "left"

    left_120, right_120, metadata_120 = simulation._synthetic_head_outputs(
        speaker,
        rate,
        120.0,
        radius,
        speed,
    )
    left_minus_120, right_minus_120, metadata_minus_120 = (
        simulation._synthetic_head_outputs(
            speaker,
            rate,
            -120.0,
            radius,
            speed,
        )
    )
    assert np.array_equal(left_120, right_minus_120)
    assert np.array_equal(right_120, left_minus_120)
    assert metadata_120["theta_rad"] == pytest.approx(np.pi / 3)
    assert metadata_minus_120["theta_rad"] == pytest.approx(np.pi / 3)

    left_zero, right_zero, metadata_zero = simulation._synthetic_head_outputs(
        speaker,
        rate,
        0.0,
        radius,
        speed,
    )
    left_180, right_180, metadata_180 = simulation._synthetic_head_outputs(
        speaker,
        rate,
        180.0,
        radius,
        speed,
    )
    left_minus_180, right_minus_180, metadata_minus_180 = (
        simulation._synthetic_head_outputs(
            speaker,
            rate,
            -180.0,
            radius,
            speed,
        )
    )
    assert np.array_equal(left_zero, right_zero)
    assert np.array_equal(left_180, right_180)
    assert np.array_equal(left_180, left_minus_180)
    assert np.array_equal(right_180, right_minus_180)
    for metadata in (metadata_zero, metadata_180, metadata_minus_180):
        assert metadata["source_side"] == "center"
        assert metadata["itd_ms"] == 0.0
        assert metadata["high_frequency_attenuation_db"] == 0.0


@pytest.mark.parametrize(
    ("source_rate", "target_rate"),
    [
        (8_000, 48_000),
        (48_000, 16_000),
        (16_000, 24_000),
    ],
)
@pytest.mark.parametrize("pulse_index", [0, 10])
def test_resampled_fir_preserves_actual_sine_convolution_gain(
    source_rate: int,
    target_rate: int,
    pulse_index: int,
) -> None:
    source_ir = np.zeros(32)
    source_ir[pulse_index] = 1.0
    kernel, processing = simulation._resample_ir(
        {"samples": source_ir, "sample_rate_hz": source_rate},
        target_rate,
    )

    assert processing["fir_gain_scale"] == pytest.approx(
        source_rate / target_rate
    )
    assert processing["boundary_extension"] == "zero"
    assert processing["ir_time_origin_samples"] > 0

    timeline = np.arange(target_rate // 2)
    margin = processing["ir_time_origin_samples"] + 32
    for frequency_hz in (100, 1000):
        sine = 0.1 * np.sin(
            2 * np.pi * frequency_hz * timeline / target_rate
        )
        convolved = signal.fftconvolve(
            sine,
            kernel,
            mode="full",
        )[:len(sine)]
        measured = convolved[margin:-margin]
        reference = sine[margin:-margin]
        gain_db = 20 * np.log10(
            np.sqrt(np.mean(measured * measured))
            / np.sqrt(np.mean(reference * reference))
        )
        assert gain_db == pytest.approx(0.0, abs=0.05)


def test_resampled_zero_dc_fir_is_not_sum_normalised() -> None:
    source_ir = np.zeros(64)
    source_ir[0] = 1.0
    source_ir[1] = -1.0
    kernel, processing = simulation._resample_ir(
        {"samples": source_ir, "sample_rate_hz": 48_000},
        16_000,
    )

    assert processing["fir_gain_scale"] == pytest.approx(3.0)
    assert abs(float(np.sum(kernel))) < 1e-3
    assert abs(float(np.sum(kernel))) != pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize(
    (
        "source_rate",
        "target_rate",
        "expected_crop_start",
        "expected_padding_origin",
        "expected_guard",
    ),
    [
        (48_000, 16_000, 5, 18, 13),
        (16_000, 48_000, 27, 60, 33),
    ],
)
def test_nonidentity_resampling_metadata_tracks_actual_crop(
    source_rate: int,
    target_rate: int,
    expected_crop_start: int,
    expected_padding_origin: int,
    expected_guard: int,
) -> None:
    source_ir = np.zeros(8)
    source_ir[0] = 1.0
    _, processing = simulation._resample_ir(
        {"samples": source_ir, "sample_rate_hz": source_rate},
        target_rate,
    )

    assert processing["source_sample_rate_hz"] != processing[
        "processing_sample_rate_hz"
    ]
    assert processing["input_padding_samples"] > 0
    assert processing["output_guard_samples"] == expected_guard
    assert processing["resampled_padding_origin_samples"] == (
        expected_padding_origin
    )
    assert processing["resampled_padding_origin_samples"] == (
        processing["input_padding_samples"] * processing["up"]
        // processing["down"]
    )
    assert processing["padding_removed_samples"] == expected_crop_start
    assert processing["padding_removed_samples"] == (
        processing["resampled_padding_origin_samples"]
        - processing["output_guard_samples"]
    )
    assert processing["padding_removed_samples"] != processing[
        "resampled_padding_origin_samples"
    ]


def test_mixed_rate_ir_origins_are_aligned_and_reported() -> None:
    left_ir = np.zeros(8)
    left_ir[0] = 1.0
    right_ir = np.zeros(8)
    right_ir[0] = 1.0
    left, left_processing = simulation._resample_ir(
        {"samples": left_ir, "sample_rate_hz": 16_000},
        48_000,
    )
    right, right_processing = simulation._resample_ir(
        {"samples": right_ir, "sample_rate_hz": 48_000},
        48_000,
    )
    left, right = simulation._align_ir_kernel_origins(
        left,
        right,
        left_processing,
        right_processing,
    )

    assert left_processing["aligned_ir_time_origin_samples"] == (
        right_processing["aligned_ir_time_origin_samples"]
    )
    assert left_processing["pair_alignment_prefix_samples"] == 0
    assert right_processing["pair_alignment_prefix_samples"] > 0
    assert int(np.argmax(np.abs(left))) == (
        left_processing["aligned_ir_time_origin_samples"]
    )
    assert int(np.argmax(np.abs(right))) == (
        right_processing["aligned_ir_time_origin_samples"]
    )


def test_kaiser_sinc_fractional_delay_has_exact_endpoints_and_flat_phase() -> None:
    samples = np.array([0.25, -0.5, 1.0, 0.0, -0.125])
    assert np.array_equal(simulation._fractional_delay(samples, 0.0), samples)

    expected_integer = np.array([0.0, 0.0, 0.25, -0.5, 1.0])
    assert np.array_equal(
        simulation._fractional_delay(samples, 2.0),
        expected_integer,
    )

    impulse = np.zeros(2048)
    impulse[256] = 1.0
    delay_samples = 0.37
    delayed = simulation._fractional_delay(impulse, delay_samples)
    response = np.fft.rfft(delayed) / np.fft.rfft(impulse)
    normalized_frequency = np.fft.rfftfreq(len(impulse))
    passband = (normalized_frequency > 0) & (normalized_frequency <= 0.45)
    magnitude_db = 20 * np.log10(np.abs(response[passband]))
    phase_error = np.angle(
        response[passband]
        * np.exp(1j * 2 * np.pi * normalized_frequency[passband] * delay_samples)
    )

    assert np.max(np.abs(magnitude_db)) < 0.05
    assert np.max(np.abs(phase_error)) < 1e-3
    metadata = simulation._fractional_delay_metadata(delay_samples)
    assert metadata["taps"] == 129
    assert metadata["boundary_extension"] == "zero"
    assert metadata["implementation"] == "kaiser_windowed_sinc_offline_interpolation"
    assert metadata["integer_delay_samples"] == 0
    assert metadata["fractional_delay_samples"] == pytest.approx(delay_samples)
    assert metadata["kernel_delay_samples"] == pytest.approx(delay_samples)
    assert metadata["version"] == "split_integer_kaiser_sinc_129_v2"


@pytest.mark.parametrize(
    "delay_samples",
    [0.178, 4.178, 12.534, 31.48, 61.7],
)
def test_split_integer_fractional_delay_matches_magnitude_and_phase(
    delay_samples: float,
) -> None:
    impulse = np.zeros(2048)
    impulse[256] = 1.0
    delayed = simulation._fractional_delay(impulse, delay_samples)
    response = np.fft.rfft(delayed) / np.fft.rfft(impulse)
    normalized_frequency = np.fft.rfftfreq(len(impulse))
    passband = (normalized_frequency > 0) & (normalized_frequency <= 0.45)
    magnitude_db = 20 * np.log10(np.abs(response[passband]))
    phase_error = np.angle(
        response[passband]
        * np.exp(1j * 2 * np.pi * normalized_frequency[passband] * delay_samples)
    )
    integer_delay, fractional_delay = simulation._split_fractional_delay(
        delay_samples
    )
    metadata = simulation._fractional_delay_metadata(delay_samples)

    assert integer_delay == int(np.floor(delay_samples))
    assert 0.0 <= fractional_delay < 1.0
    assert metadata["integer_delay_samples"] == integer_delay
    assert metadata["fractional_delay_samples"] == pytest.approx(
        fractional_delay
    )
    assert metadata["kernel_delay_samples"] == pytest.approx(fractional_delay)
    assert np.max(np.abs(magnitude_db)) < 0.05
    assert np.max(np.abs(phase_error)) < 1e-3


def test_fractional_delay_beyond_signal_length_is_zero_extension() -> None:
    samples = np.linspace(-1.0, 1.0, 32)
    delayed = simulation._fractional_delay(samples, len(samples) + 0.7)

    assert np.array_equal(delayed, np.zeros_like(samples))


def test_maximum_configured_woodworth_delay_uses_fractional_kernel() -> None:
    rate = 48_000
    metadata = simulation._fractional_delay_metadata(
        (0.15 / 300.0) * (np.pi / 2.0 + 1.0) * rate
    )

    assert metadata["delay_samples"] == pytest.approx(61.7, abs=1e-3)
    assert metadata["integer_delay_samples"] == 61
    assert 0.0 <= metadata["fractional_delay_samples"] < 1.0
    assert metadata["kernel_delay_samples"] == pytest.approx(
        metadata["fractional_delay_samples"]
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"head_radius_m": 0.039}, "head_radius_m must be between"),
        ({"head_radius_m": 0.151}, "head_radius_m must be between"),
        ({"speed_of_sound_mps": 299}, "speed_of_sound_mps must be between"),
        ({"speed_of_sound_mps": 381}, "speed_of_sound_mps must be between"),
    ],
)
def test_manual_api_head_geometry_matches_batch_range(
    kwargs: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(simulation.SyntheticReceiveError, match=message):
        simulation.evaluate_synthetic_receive(
            "WB-wireless",
            "speaker.wav",
            wav(impulse_signal()),
            **kwargs,
        )


def test_manual_api_keeps_positive_geometry_errors_and_accepts_batch_bounds() -> None:
    for kwargs, label in (
        ({"head_radius_m": 0.0}, "head_radius_m"),
        ({"speed_of_sound_mps": 0.0}, "speed_of_sound_mps"),
    ):
        with pytest.raises(
            simulation.SyntheticReceiveError,
            match=f"{label} must be positive",
        ):
            simulation.evaluate_synthetic_receive(
                "WB-wireless",
                "speaker.wav",
                wav(impulse_signal()),
                **kwargs,
            )

    for radius, speed in ((0.04, 300.0), (0.15, 380.0)):
        validated = simulation._validate_request(
            "WB-wireless",
            30.0,
            radius,
            speed,
            None,
            None,
            None,
            None,
        )
        assert validated[1] == radius
        assert validated[2] == speed


def test_head_shelf_is_unity_dc_with_high_frequency_asymptote() -> None:
    rate = 48_000
    azimuth_deg = 30.0
    radius = 0.0875
    speed = 343.0
    shadow = np.sin(np.deg2rad(azimuth_deg))
    impulse = np.zeros(4096)
    impulse[0] = 1.0
    response = simulation._first_order_head_shelf(
        impulse,
        rate,
        shadow,
        radius,
        speed,
    )
    frequencies = np.array([0.0, 125.0, 10000.0])
    _, transfer = signal.freqz(
        response,
        worN=2 * np.pi * frequencies / rate,
    )
    transfer_db = 20 * np.log10(np.abs(transfer))

    assert transfer_db[0] == pytest.approx(0.0, abs=1e-12)
    assert transfer_db[1] > -1.0
    assert transfer_db[2] == pytest.approx(-6.0, abs=0.2)

    timeline = np.arange(rate) / rate
    tone_125 = 0.1 * np.sin(2 * np.pi * 125.0 * timeline)
    far_125, near_125, metadata = simulation._synthetic_head_outputs(
        tone_125,
        rate,
        azimuth_deg,
        radius,
        speed,
    )
    steady = slice(256, -256)
    low_frequency_ild_db = 20 * np.log10(
        np.sqrt(np.mean(far_125[steady] ** 2))
        / np.sqrt(np.mean(near_125[steady] ** 2))
    )
    assert abs(low_frequency_ild_db) < 1.0
    assert metadata["high_frequency_attenuation_db"] == pytest.approx(6.0)
    assert metadata["far_ear_attenuation_db"] == pytest.approx(6.0)
    assert "high-frequency asymptotic" in (
        metadata["far_ear_attenuation_db_semantics"]
    )
    assert metadata["model_version"]
    assert metadata["version"] == metadata["model_version"]


def test_custom_ir_mode_performs_independent_convolutions(
    isolated_store: Path,
) -> None:
    speaker = np.zeros(64)
    speaker[8] = 0.2
    left_ir = np.array([0.5, 0.0, 0.0])
    right_ir = np.array([0.0, 0.0, 0.25])
    report = simulation.evaluate_synthetic_receive(
        "SWB-wired",
        "speaker.wav",
        wav(speaker),
        left_ir_name="left.wav",
        left_ir_bytes=wav(left_ir),
        right_ir_name="right.wav",
        right_ir_bytes=wav(right_ir),
    )

    assert report["evidence_kind"] == "synthetic_ir"
    assert report["simulation"]["head_model"] is None
    assert report["metrics"]["interaural_delay_ms"] == pytest.approx(
        2 / 16000 * 1000,
        abs=1e-6,
    )
    left_path = simulation.get_synthetic_receive_file(
        report["evaluation_id"], "virtual_left.wav"
    )
    right_path = simulation.get_synthetic_receive_file(
        report["evaluation_id"], "virtual_right.wav"
    )
    assert left_path is not None and right_path is not None
    left, _ = sf.read(left_path, dtype="float64")
    right, _ = sf.read(right_path, dtype="float64")
    assert left[8] == pytest.approx(0.1, abs=2e-4)
    assert abs(left[10]) < 2e-7
    assert abs(right[8]) < 2e-7
    assert right[10] == pytest.approx(0.05, abs=2e-4)
    assert len(right) >= 11


def test_invalid_mode_multichannel_clipping_and_one_sided_ir_are_rejected() -> None:
    mono = wav(impulse_signal())
    with pytest.raises(simulation.SyntheticReceiveError, match="mode"):
        simulation.evaluate_synthetic_receive("WB-invalid", "speaker.wav", mono)

    multichannel = wav(np.column_stack([impulse_signal(), impulse_signal()]))
    with pytest.raises(simulation.SyntheticReceiveError, match="mono"):
        simulation.evaluate_synthetic_receive("WB-wired", "speaker.wav", multichannel)

    clipped = impulse_signal()
    clipped[80] = 0.9999
    with pytest.raises(simulation.SyntheticReceiveError, match="clipped|ceiling"):
        simulation.evaluate_synthetic_receive(
            "WB-wired",
            "speaker.wav",
            wav(clipped),
        )

    with pytest.raises(simulation.SyntheticReceiveError, match="pair"):
        simulation.evaluate_synthetic_receive(
            "WB-wired",
            "speaker.wav",
            mono,
            left_ir_name="left.wav",
            left_ir_bytes=wav(np.array([1.0])),
        )


def test_persistence_list_get_file_and_path_traversal_guards(
    isolated_store: Path,
) -> None:
    audio = wav(impulse_signal())
    first = simulation.evaluate_synthetic_receive("NB-wired", "speaker.wav", audio)
    second = simulation.evaluate_synthetic_receive("NB-bt", "speaker.wav", audio)

    assert simulation.get_synthetic_receive_evaluation(first["evaluation_id"]) == first
    assert simulation.get_synthetic_receive_evaluation(second["evaluation_id"]) == second
    listed = simulation.list_synthetic_receive_evaluations(limit=100)
    assert {row["evaluation_id"] for row in listed} == {
        first["evaluation_id"],
        second["evaluation_id"],
    }
    assert listed == sorted(
        listed,
        key=lambda row: (row["created_at"], row["evaluation_id"]),
        reverse=True,
    )
    assert len(simulation.list_synthetic_receive_evaluations(limit=1)) == 1

    report_path = simulation.get_synthetic_receive_file(
        first["evaluation_id"], "report.json"
    )
    assert report_path is not None
    assert json.loads(report_path.read_text()) == first
    assert simulation.get_synthetic_receive_evaluation("../secret") is None
    assert simulation.get_synthetic_receive_evaluation("missing") is None
    assert simulation.get_synthetic_receive_file(first["evaluation_id"], "../report.json") is None
    assert simulation.get_synthetic_receive_file(first["evaluation_id"], "other.wav") is None
    assert simulation.get_synthetic_receive_file("../secret", "report.json") is None
    assert (isolated_store / first["evaluation_id"] / "virtual_left.wav").is_file()
    assert (isolated_store / first["evaluation_id"] / "virtual_right.wav").is_file()
    with pytest.raises(simulation.SyntheticReceiveError, match="limit"):
        simulation.list_synthetic_receive_evaluations(limit=0)
    with pytest.raises(simulation.SyntheticReceiveError, match="limit"):
        simulation.list_synthetic_receive_evaluations(limit=101)


def test_no_noise_interval_is_null_and_reports_are_explicitly_nonformal() -> None:
    report = simulation.evaluate_synthetic_receive(
        "NB-wireless",
        "speaker.wav",
        wav(impulse_signal()),
    )
    assert report["schema_version"] == 1
    assert report["standard_compliance"] is False
    assert report["formal_hats_evidence"] is False
    assert report["synthetic_proxy"] is True
    assert report["metrics"]["noise"] is None
    assert report["metrics"]["noise_interval"] is None
    assert "not a HATS/DRP measurement" in " ".join(report["limitations"])
    assert "P.57" in " ".join(report["limitations"])
    assert "P.58" in " ".join(report["limitations"])

    listed = simulation.list_synthetic_receive_evaluations()
    assert listed
    assert all(row["standard_compliance"] is False for row in listed)
    assert all(row["formal_hats_evidence"] is False for row in listed)


def test_noise_window_and_out_of_band_metrics_are_digital_proxies() -> None:
    rate = 16000
    time = np.arange(rate // 2) / rate
    signal_in = 0.08 * np.sin(2 * np.pi * 1000 * time)
    signal_in += 0.08 * np.sin(2 * np.pi * 6000 * time)
    report = simulation.evaluate_synthetic_receive(
        "WB-wired",
        "speaker.wav",
        wav(signal_in),
        upper_hz=4000.0,
        noise_start_s=0.1,
        noise_end_s=0.3,
    )
    metrics = report["metrics"]
    assert metrics["noise"] is not None
    assert metrics["noise"]["weighting"].startswith("none")
    assert metrics["noise"]["a_weighted"] is False
    assert metrics["noise"]["digital_rms_dbfs"] < 0
    assert metrics["noise"]["peak_prominence_db"] >= 0
    assert metrics["out_of_band_ratio_db"] < 0
    assert "125" in metrics["third_octave_band_levels_db"]["left"]
    assert "8000" in metrics["third_octave_band_levels_db"]["right"]
    assert metrics["third_octave_band_levels_db"] == metrics["third_octave_band_levels_dbfs"]


def test_six_khz_tone_is_not_squared_into_false_12khz_band_energy() -> None:
    rate = 48_000
    time = np.arange(rate) / rate
    signal_in = 0.08 * np.sin(2 * np.pi * 6000 * time)

    report = simulation.evaluate_synthetic_receive(
        "WB-wired",
        "speaker.wav",
        wav(signal_in, rate=rate, subtype="PCM_24"),
        upper_hz=8000.0,
    )

    assert report["metrics"]["out_of_band_ratio_db"] == simulation._DB_FLOOR


def test_out_of_band_ratio_merges_ear_power_without_phase_cancellation() -> None:
    rate = 48_000
    time = np.arange(rate) / rate
    mixed = (
        0.05 * np.sin(2 * np.pi * 6000 * time)
        + 0.05 * np.sin(2 * np.pi * 12000 * time)
    )
    pure = np.sin(2 * np.pi * 6000 * time)

    pure_antiphase = simulation._out_of_band_ratio_db(
        pure,
        -pure,
        rate,
        8000.0,
    )
    mixed_antiphase = simulation._out_of_band_ratio_db(
        mixed,
        -mixed,
        rate,
        8000.0,
    )

    assert pure_antiphase == simulation._DB_FLOOR
    assert mixed_antiphase == pytest.approx(
        10 * np.log10(0.5),
        abs=0.02,
    )


def test_out_of_band_ratio_is_none_when_no_bin_exceeds_upper_bound() -> None:
    rate = 48_000
    time = np.arange(rate // 2) / rate
    signal_in = 0.08 * np.sin(2 * np.pi * 6000 * time)

    assert (
        simulation._out_of_band_ratio_db(signal_in, signal_in, rate, rate / 2)
        is None
    )
    assert (
        simulation._out_of_band_ratio_db(
            signal_in,
            signal_in,
            rate,
            rate / 2 + 1,
        )
        is None
    )
    report = simulation.evaluate_synthetic_receive(
        "WB-wired",
        "speaker.wav",
        wav(signal_in, rate=rate, subtype="PCM_24"),
        upper_hz=rate / 2,
    )
    assert report["metrics"]["out_of_band_ratio_db"] is None
    json.dumps(report, allow_nan=False)


def test_binaural_third_octave_uses_psd_energy_average() -> None:
    rate = 16_000
    time = np.arange(rate) / rate
    left = 0.03 * np.sin(2 * np.pi * 1000 * time)
    right = 0.12 * np.sin(2 * np.pi * 1000 * time)

    levels = simulation._third_octave_levels(left, right, rate)
    left_db = levels["left"]["1000"]
    right_db = levels["right"]["1000"]
    expected = 10 * np.log10((10 ** (left_db / 10) + 10 ** (right_db / 10)) / 2)

    assert levels["binaural"]["1000"] == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize("center_hz", [10000, 12500, 16000])
def test_swb_third_octave_measures_high_frequency_power(center_hz: int) -> None:
    rate = 48_000
    time = np.arange(rate) / rate
    tone = 0.2 * np.sin(2 * np.pi * center_hz * time)

    levels = simulation._third_octave_levels(tone, -tone, rate)
    expected = 10 * np.log10(0.2 ** 2 / 2)

    assert levels["left"][str(center_hz)] == pytest.approx(expected, abs=0.02)
    assert levels["right"][str(center_hz)] == pytest.approx(expected, abs=0.02)
    assert levels["binaural"][str(center_hz)] == pytest.approx(expected, abs=0.02)


def test_swb_third_octave_does_not_invent_unobservable_bands() -> None:
    rate = 16_000
    time = np.arange(rate) / rate
    tone = 0.2 * np.sin(2 * np.pi * 1000 * time)

    levels = simulation._third_octave_levels(tone, tone, rate)

    for channel in levels.values():
        assert all(str(center_hz) not in channel for center_hz in (10000, 12500, 16000))


def test_atomic_json_write_retains_temporary_file_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "atomic" / "report.json"

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError(f"replace failed: {source} -> {destination}")

    monkeypatch.setattr(simulation.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        simulation._atomic_json_write(target, {"status": "failed"})

    temporary_files = list(target.parent.glob(f".{target.name}.*.tmp"))
    assert len(temporary_files) == 1
    assert json.loads(temporary_files[0].read_text()) == {"status": "failed"}
    assert not target.exists()


def test_failed_receive_staging_directory_is_retained(
    isolated_store: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_replace = simulation.os.replace

    def fail_directory_replace(source: Path, destination: Path) -> None:
        if Path(source).is_dir():
            raise OSError(f"directory replace failed: {source} -> {destination}")
        original_replace(source, destination)

    monkeypatch.setattr(simulation.os, "replace", fail_directory_replace)
    with pytest.raises(OSError, match="directory replace failed"):
        simulation.evaluate_synthetic_receive(
            "WB-wireless",
            "speaker.wav",
            wav(impulse_signal()),
        )

    staging_directories = list(isolated_store.glob(".synthetic-receive-*"))
    assert len(staging_directories) == 1
    staging = staging_directories[0]
    assert (staging / "virtual_left.wav").is_file()
    assert (staging / "virtual_right.wav").is_file()
    report_path = staging / "report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text())
    assert report["evaluation_id"]
    assert report["result_sha256"] == report["result"]["sha256"]


def test_report_result_identity_and_artifact_hashes_are_consistent() -> None:
    report = simulation.evaluate_synthetic_receive(
        "SWB-wireless",
        "speaker.wav",
        wav(impulse_signal()),
    )
    evaluation_id = report["evaluation_id"]
    assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", evaluation_id)
    assert report["result"]["path"] == report["result_path"]
    assert report["result"]["sha256"] == report["result_sha256"]
    assert report["artifacts"]["left"]["url"].endswith(
        f"/{evaluation_id}/files/virtual_left.wav"
    )
    assert report["artifacts"]["right"]["url"].endswith(
        f"/{evaluation_id}/files/virtual_right.wav"
    )
    left_path = simulation.get_synthetic_receive_file(evaluation_id, "virtual_left.wav")
    right_path = simulation.get_synthetic_receive_file(evaluation_id, "virtual_right.wav")
    assert left_path is not None and right_path is not None
    assert simulation._sha256_file(left_path) == report["artifacts"]["left"]["sha256"]
    assert simulation._sha256_file(right_path) == report["artifacts"]["right"]["sha256"]


def test_http_surface_persists_lists_and_serves_virtual_ear_files() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/acoustic/receive-simulation/evaluate",
        files={
            "speaker_output": (
                "speaker.wav",
                wav(impulse_signal()),
                "audio/wav",
            )
        },
        data={"mode": "WB-wireless", "azimuth_deg": "30"},
    )
    assert response.status_code == 201, response.text
    report = response.json()
    assert report["standard_compliance"] is False
    assert report["formal_hats_evidence"] is False

    latest = client.get(
        "/api/acoustic/receive-simulation/evaluations?limit=1"
    )
    assert latest.status_code == 200
    assert latest.json()[0]["evaluation_id"] == report["evaluation_id"]

    left = client.get(report["artifacts"]["left"]["url"])
    assert left.status_code == 200
    assert left.headers["content-type"].startswith("audio/")
    assert left.content[:4] == b"RIFF"
