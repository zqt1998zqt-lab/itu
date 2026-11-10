from __future__ import annotations

import json
import math

import numpy as np
import pytest

from scripts.acceptance_signal_design import render_signals
from server.acoustic.acceptance_models import FREQUENCY_RESPONSE_CENTERS_HZ


def _source(
    rate: int,
    seconds: float = 12.0,
    *,
    seed: int = 17,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = int(round(rate * seconds))
    time = np.arange(frames, dtype=np.float64) / rate
    near = (
        0.045 * np.sin(2.0 * math.pi * 310.0 * time)
        + 0.025 * np.sin(2.0 * math.pi * 733.0 * time + 0.4)
    )
    far = (
        0.040 * np.sin(2.0 * math.pi * 510.0 * time + 0.2)
        + 0.020 * np.sin(2.0 * math.pi * 1_170.0 * time)
    )
    noise = np.random.default_rng(seed).normal(size=frames)
    return near, far, noise


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def _slice(values: np.ndarray, rate: int, start_s: float, end_s: float) -> np.ndarray:
    return values[int(round(start_s * rate)):int(round(end_s * rate))]


def _mask_from_segments(
    segments: list[dict[str, float]],
    frames: int,
    rate: int,
) -> np.ndarray:
    mask = np.zeros(frames, dtype=bool)
    for segment in segments:
        start = int(round(segment["start_s"] * rate))
        end = int(round(segment["end_s"] * rate))
        mask[start:end] = True
    return mask


def _assert_contract_shapes(
    signals: dict[str, np.ndarray],
    metadata: dict,
    frames: int,
    channels: int,
) -> None:
    assert set(signals) == {"microphone", "reference", "far_end", "noise"}
    if channels == 1:
        assert signals["microphone"].shape in {(frames,), (frames, 1)}
    else:
        assert signals["microphone"].shape == (frames, channels)
    for name in ("reference", "far_end", "noise"):
        assert signals[name].shape == (frames,)
        assert signals[name].dtype == np.float64
        assert np.isfinite(signals[name]).all()
    assert signals["microphone"].dtype == np.float64
    assert np.isfinite(signals["microphone"]).all()
    assert metadata["peak"] < 0.7
    assert metadata["clipped"] is False
    assert metadata["standard_compliance"] is False


def test_duplex_has_real_far_near_double_talk_and_noise_only_windows():
    rate = 16_000
    near, far, noise = _source(rate)
    signals, metadata = render_signals(
        near,
        far,
        noise,
        rate,
        direction="send",
        kind="duplex",
        snr_db=10.0,
        ser_db=-5.0,
        seed=23,
    )
    _assert_contract_shapes(signals, metadata, len(near), 2)
    assert {segment["label"] for segment in metadata["segments"]} >= {
        "quiet",
        "far_only",
        "double_talk",
        "near_only",
        "noise_only",
    }

    far_only = _slice(signals["reference"], rate, 0.7, 1.8)
    far_only_far = _slice(signals["far_end"], rate, 0.7, 1.8)
    near_only = _slice(signals["reference"], rate, 2.1, 4.9)
    near_only_far = _slice(signals["far_end"], rate, 2.1, 4.9)
    double_talk_near = _slice(signals["reference"], rate, 8.1, 10.9)
    double_talk_far = _slice(signals["far_end"], rate, 8.1, 10.9)
    noise_only_near = _slice(signals["reference"], rate, 11.1, 11.9)
    noise_only_far = _slice(signals["far_end"], rate, 11.1, 11.9)
    noise_only_noise = _slice(signals["noise"], rate, 11.1, 11.9)

    assert _rms(far_only) < 1e-12
    assert _rms(far_only_far) > 0.01
    assert _rms(near_only) > 0.01
    assert np.array_equal(near_only_far, np.zeros_like(near_only_far))
    assert _rms(double_talk_near) > 0.01
    assert _rms(double_talk_far) > 0.01
    assert np.array_equal(noise_only_near, np.zeros_like(noise_only_near))
    assert np.array_equal(noise_only_far, np.zeros_like(noise_only_far))
    assert _rms(noise_only_noise) > 0.0

    echo_ch0 = signals["microphone"][:, 0] - signals["reference"] - signals["noise"]
    path = metadata["echo_paths"][0]
    reconstructed_echo = np.zeros(len(near), dtype=np.float64)
    for position, gain in zip(
        path["tap_samples"],
        path["tap_gains"],
        strict=True,
    ):
        reconstructed_echo[position:] += (
            signals["far_end"][:len(near) - position] * gain
        )
    reconstructed_echo *= path["echo_gain_linear"]
    assert np.allclose(
        signals["microphone"][:, 0],
        signals["reference"] + reconstructed_echo + signals["noise"],
        rtol=0.0,
        atol=1e-15,
    )
    assert metadata["source_mix_check"]["residual_max_abs"] <= 1e-15
    noise_recipe = metadata["noise_spatial"]["channel_1_recipe"]
    noise_scale = noise_recipe["noise_channel_scale_linear"]
    unit_noise_0 = signals["noise"] / noise_scale
    independent_2 = np.random.default_rng(noise_recipe["seed2"]).normal(
        size=len(near)
    )
    noise_1_reconstructed = (
        noise_recipe["rho"] * unit_noise_0
        + math.sqrt(1.0 - noise_recipe["rho"] ** 2) * independent_2
    )
    noise_1_reconstructed /= noise_recipe["normalizer"]
    noise_1_reconstructed *= noise_scale
    path_1 = metadata["echo_paths"][1]
    echo_1_reconstructed = np.zeros(len(near), dtype=np.float64)
    for position, gain in zip(
        path_1["tap_samples"],
        path_1["tap_gains"],
        strict=True,
    ):
        echo_1_reconstructed[position:] += (
            signals["far_end"][:len(near) - position] * gain
        )
    echo_1_reconstructed *= path_1["echo_gain_linear"]
    delayed_reference = np.concatenate((
        np.zeros(1),
        signals["reference"][:-1],
    ))
    assert np.allclose(
        signals["microphone"][:, 1],
        delayed_reference + echo_1_reconstructed + noise_1_reconstructed,
        rtol=0.0,
        atol=1e-15,
    )
    assert metadata["noise_spatial"]["check"][
        "channel_0_rms_over_normalization_mask"
    ] == pytest.approx(
        metadata["noise_spatial"]["check"][
            "channel_1_rms_over_normalization_mask"
        ],
        rel=1e-12,
    )
    assert abs(
        metadata["noise_spatial"]["check"][
            "channel_correlation_over_normalization_mask"
        ]
    ) < 0.999
    assert metadata["noise_spatial"]["channel_1_recipe"][
        "long_arrays_in_sidecar"
    ] is False
    assert noise_recipe["noise_channel_scale_linear"] == pytest.approx(
        noise_recipe["noise_scale"] * noise_recipe["common_gain"],
        rel=1e-15,
    )
    json.dumps(metadata, allow_nan=False)
    far_only_mask = np.zeros(len(near), dtype=bool)
    near_active_mask = np.zeros(len(near), dtype=bool)
    for segment in metadata["segments"]:
        start = int(round(segment["start_s"] * rate))
        end = int(round(segment["end_s"] * rate))
        if segment["label"] == "far_only":
            far_only_mask[start:end] = True
        elif segment["label"] in {"near_only", "double_talk"}:
            near_active_mask[start:end] = True
    measured = 20.0 * math.log10(
        _rms(echo_ch0[far_only_mask])
        / _rms(signals["reference"][near_active_mask])
    )
    assert metadata["ser_db_measured"] == pytest.approx(measured, abs=1e-12)
    assert metadata["snr_db_target_applicable"] is True
    assert metadata["snr_db_target"] == 10.0
    assert metadata["snr_db_measured"] == pytest.approx(10.0, abs=1e-9)
    assert metadata["ser_db_target_applicable"] is True
    assert metadata["ser_db_nominal_target"] == -5.0
    assert metadata["ser_db_effective_target"] == -5.0
    assert metadata["beam_delays_ms"] == pytest.approx([1_000.0 / rate, 0.0])
    assert not np.array_equal(echo_ch0, np.zeros_like(echo_ch0))


def test_speech_is_complete_and_channel_zero_is_exact_reference_plus_noise():
    rate = 16_000
    near, far, noise = _source(rate)
    signals, metadata = render_signals(
        near,
        far,
        noise,
        rate,
        direction="send",
        kind="speech",
        snr_db=8.0,
        seed=31,
    )
    _assert_contract_shapes(signals, metadata, len(near), 2)
    assert np.array_equal(
        signals["microphone"][:, 0],
        signals["reference"] + signals["noise"],
    )
    assert np.array_equal(signals["far_end"], np.zeros_like(signals["far_end"]))
    assert _rms(_slice(signals["reference"], rate, 0.0, 0.1)) > 0.005
    assert _rms(_slice(signals["reference"], rate, 11.9, 12.0)) > 0.005
    assert np.count_nonzero(signals["reference"]) > int(0.99 * len(near))
    assert [segment["label"] for segment in metadata["segments"]] == ["speech"]

    received, receive_metadata = render_signals(
        near,
        far,
        noise,
        rate,
        direction="receive",
        kind="speech",
        snr_db=8.0,
        seed=31,
    )
    _assert_contract_shapes(received, receive_metadata, len(near), 1)
    assert _rms(_slice(received["far_end"], rate, 0.0, 0.1)) > 0.005
    assert _rms(_slice(received["far_end"], rate, 11.9, 12.0)) > 0.005
    assert _rms(received["microphone"]) > 0.0


def test_swb_response_contains_native_high_frequency_and_oob_diagnostic():
    rate = 48_000
    near, far, noise = _source(rate, seconds=2.0)
    response, response_metadata = render_signals(
        near,
        far,
        noise,
        rate,
        direction="receive",
        kind="response",
        snr_db=10.0,
        bandwidth="SWB",
        seed=41,
    )
    _assert_contract_shapes(response, response_metadata, len(near), 1)
    assert np.array_equal(response["reference"], np.zeros_like(response["reference"]))
    spectrum = np.abs(np.fft.rfft(response["far_end"])) ** 2
    frequencies = np.fft.rfftfreq(len(near), d=1.0 / rate)
    high_energy = float(np.sum(spectrum[(frequencies >= 10_000.0)
                                        & (frequencies <= 12_500.0)]))
    total_energy = float(np.sum(spectrum[frequencies >= 125.0]))
    assert high_energy / total_energy > 0.05

    oob, oob_metadata = render_signals(
        near,
        far,
        noise,
        rate,
        direction="receive",
        kind="oob",
        snr_db=10.0,
        bandwidth="SWB",
        seed=41,
    )
    oob_spectrum = np.abs(np.fft.rfft(oob["far_end"])) ** 2
    diagnostic_energy = float(np.sum(
        oob_spectrum[(frequencies >= 14_000.0) & (frequencies < rate / 2.0)]
    ))
    assert diagnostic_energy / total_energy > 0.01
    assert oob_metadata["diagnostic_tone_hz"] == 14_500.0


@pytest.mark.parametrize(
    ("rate", "bandwidth", "direction"),
    [
        (16_000, "NB", "send"),
        (24_000, "WB", "send"),
        (48_000, "SWB", "receive"),
    ],
)
def test_response_excites_every_contract_frequency_center(
    rate: int,
    bandwidth: str,
    direction: str,
):
    near, far, noise = _source(rate, seconds=2.0)
    signals, metadata = render_signals(
        near,
        far,
        noise,
        rate,
        direction=direction,
        kind="response",
        snr_db=10.0,
        bandwidth=bandwidth,
        seed=43,
    )
    expected = [
        frequency
        for frequency in FREQUENCY_RESPONSE_CENTERS_HZ
        if frequency <= {"NB": 4_000.0, "WB": 8_000.0, "SWB": 12_500.0}[
            bandwidth
        ]
    ]
    assert metadata["multisine_frequencies_hz"] == expected
    source = signals["reference"] if direction == "send" else signals["far_end"]
    spectrum = np.abs(np.fft.rfft(source)) ** 2
    frequencies = np.fft.rfftfreq(len(source), d=1.0 / rate)
    strongest = float(np.max(spectrum))
    for center in expected:
        index = int(np.argmin(np.abs(frequencies - center)))
        assert spectrum[index] > strongest * 1e-3


def test_known_input_and_playback_gain_scopes_do_not_rescale_all_components():
    rate = 16_000
    near, far, noise = _source(rate)
    low, low_metadata = render_signals(
        near,
        far,
        noise,
        rate,
        direction="send",
        kind="speech",
        snr_db=12.0,
        seed=51,
        input_level_db=-6.0,
    )
    high, high_metadata = render_signals(
        near,
        far,
        noise,
        rate,
        direction="send",
        kind="speech",
        snr_db=12.0,
        seed=51,
        input_level_db=6.0,
    )
    expected_input_gain = 10.0 ** (12.0 / 20.0)
    assert _rms(high["reference"]) / _rms(low["reference"]) == pytest.approx(
        expected_input_gain,
        rel=1e-10,
    )
    assert _rms(high["noise"]) / _rms(low["noise"]) == pytest.approx(
        expected_input_gain,
        rel=1e-10,
    )
    assert np.array_equal(high["far_end"], low["far_end"])
    assert low_metadata["global_safety_gain"] == pytest.approx(1.0)
    assert high_metadata["global_safety_gain"] == pytest.approx(1.0)

    response_low, _ = render_signals(
        near,
        far,
        noise,
        rate,
        direction="send",
        kind="response",
        snr_db=10.0,
        bandwidth="NB",
        seed=52,
        input_level_db=-6.0,
    )
    response_high, response_metadata = render_signals(
        near,
        far,
        noise,
        rate,
        direction="send",
        kind="response",
        snr_db=10.0,
        bandwidth="NB",
        seed=52,
        input_level_db=6.0,
    )
    assert _rms(response_high["reference"]) / _rms(response_low["reference"]) == pytest.approx(
        expected_input_gain,
        rel=1e-10,
    )
    assert np.array_equal(
        response_high["far_end"],
        np.zeros_like(response_high["far_end"]),
    )
    assert response_metadata["gain_scope"]["input_level_db"]["targets"] == [
        "near",
        "reference",
        "near-driven noise",
    ]
    assert response_metadata["gain_scope"]["volume_gain_db"]["targets"] == [
        "far_end playback",
        "echo path",
    ]

    receive_low, _ = render_signals(
        near,
        far,
        noise,
        rate,
        direction="receive",
        kind="response",
        snr_db=10.0,
        bandwidth="NB",
        seed=52,
        volume_gain_db=-6.0,
    )
    receive_high, _ = render_signals(
        near,
        far,
        noise,
        rate,
        direction="receive",
        kind="response",
        snr_db=10.0,
        bandwidth="NB",
        seed=52,
        volume_gain_db=6.0,
    )
    expected_volume_gain = 10.0 ** (12.0 / 20.0)
    assert _rms(receive_high["far_end"]) / _rms(receive_low["far_end"]) == pytest.approx(
        expected_volume_gain,
        rel=1e-10,
    )
    assert np.array_equal(
        receive_high["reference"],
        np.zeros_like(receive_high["reference"]),
    )


def test_seed_is_reproducible_and_changes_spatial_noise_and_synthetic_tone():
    rate = 16_000
    near, far, noise = _source(rate)
    first, _ = render_signals(
        near,
        far,
        noise,
        rate,
        direction="send",
        kind="speech",
        snr_db=10.0,
        seed=61,
    )
    repeated, _ = render_signals(
        near,
        far,
        noise,
        rate,
        direction="send",
        kind="speech",
        snr_db=10.0,
        seed=61,
    )
    changed, _ = render_signals(
        near,
        far,
        noise,
        rate,
        direction="send",
        kind="speech",
        snr_db=10.0,
        seed=62,
    )
    for name in first:
        assert np.array_equal(first[name], repeated[name])
    assert np.array_equal(first["reference"], changed["reference"])
    assert not np.array_equal(first["noise"], changed["noise"])
    assert not np.array_equal(first["microphone"], changed["microphone"])

    tone_a, _ = render_signals(
        near,
        far,
        noise,
        rate,
        direction="receive",
        kind="response",
        snr_db=10.0,
        bandwidth="NB",
        seed=61,
    )
    tone_b, _ = render_signals(
        near,
        far,
        noise,
        rate,
        direction="receive",
        kind="response",
        snr_db=10.0,
        bandwidth="NB",
        seed=62,
    )
    assert not np.array_equal(tone_a["far_end"], tone_b["far_end"])


@pytest.mark.parametrize(
    ("rate", "bandwidth", "kind", "message"),
    [
        (8_000, "NB", "oob", "Nyquist"),
        (16_000, "WB", "oob", "Nyquist"),
        (16_000, "WB", "response", "Nyquist"),
    ],
)
def test_oob_rejects_a_diagnostic_tone_at_or_above_nyquist(
    rate: int,
    bandwidth: str,
    kind: str,
    message: str,
):
    near, far, noise = _source(rate, seconds=2.0)
    with pytest.raises(ValueError, match=message):
        render_signals(
            near,
            far,
            noise,
            rate,
            direction="receive",
            kind=kind,
            snr_db=10.0,
            bandwidth=bandwidth,
        )


def test_snr_and_ser_targets_are_null_or_effective_as_applicable():
    rate = 16_000
    near, far, noise = _source(rate)
    duplex, duplex_metadata = render_signals(
        near,
        far,
        noise,
        rate,
        direction="send",
        kind="duplex",
        snr_db=7.0,
        ser_db=-5.0,
        volume_gain_db=-6.0,
        seed=65,
    )
    _assert_contract_shapes(duplex, duplex_metadata, len(near), 2)
    assert duplex_metadata["snr_db_target"] == 7.0
    assert duplex_metadata["snr_db_measured"] == pytest.approx(7.0, abs=1e-9)
    assert duplex_metadata["ser_db_target"] == -5.0
    assert duplex_metadata["ser_db_nominal_target"] == -5.0
    assert duplex_metadata["ser_db_effective_target"] == -11.0
    assert duplex_metadata["ser_db_measured"] == pytest.approx(-11.0, abs=1e-9)

    response, response_metadata = render_signals(
        near,
        far,
        noise,
        rate,
        direction="send",
        kind="response",
        snr_db=12.0,
        bandwidth="NB",
        seed=66,
    )
    assert response_metadata["snr_db_target"] == 12.0
    assert response_metadata["snr_db_measured"] == pytest.approx(12.0, abs=1e-9)
    assert response_metadata["ser_db_target"] is None
    assert response_metadata["ser_db_effective_target"] is None
    assert response_metadata["ser_db_measured"] is None

    for direction, kind in (
        ("receive", "speech"),
        ("receive", "response"),
        ("send", "idle"),
        ("receive", "idle"),
    ):
        _, metadata = render_signals(
            near,
            far,
            noise,
            rate,
            direction=direction,
            kind=kind,
            snr_db=9.0,
                bandwidth="NB",
            seed=67,
        )
        assert metadata["snr_db_target_applicable"] is False
        assert metadata["snr_db_target"] is None
        assert metadata["snr_db_measured"] is None
        assert metadata["ser_db_target_applicable"] is False
        assert metadata["ser_db_target"] is None
        assert metadata["ser_db_effective_target"] is None
        assert metadata["ser_db_measured"] is None


def test_idle_receive_is_strictly_silent_and_cng_exposes_post_four_second_noise():
    rate = 16_000
    near, far, noise = _source(rate)
    idle, idle_metadata = render_signals(
        near,
        far,
        noise,
        rate,
        direction="receive",
        kind="idle",
        snr_db=10.0,
        seed=71,
    )
    _assert_contract_shapes(idle, idle_metadata, len(near), 1)
    assert np.array_equal(idle["far_end"], np.zeros_like(idle["far_end"]))
    assert _rms(idle["microphone"]) > 0.0
    assert idle_metadata["idle_noise_floor_dbfs"] == -75.0

    idle_tx, idle_tx_metadata = render_signals(
        near,
        far,
        noise,
        rate,
        direction="send",
        kind="idle",
        snr_db=10.0,
        seed=71,
    )
    _assert_contract_shapes(idle_tx, idle_tx_metadata, len(near), 2)
    assert np.array_equal(idle_tx["reference"], np.zeros_like(idle_tx["reference"]))
    assert np.array_equal(idle_tx["far_end"], np.zeros_like(idle_tx["far_end"]))
    idle_dbfs = 20.0 * math.log10(_rms(idle_tx["noise"]))
    assert idle_dbfs == pytest.approx(-75.0, abs=0.03)

    cng, cng_metadata = render_signals(
        near,
        far,
        noise,
        rate,
        direction="send",
        kind="cng",
        snr_db=10.0,
        seed=72,
    )
    _assert_contract_shapes(cng, cng_metadata, len(near), 2)
    assert _rms(_slice(cng["reference"], rate, 0.1, 1.9)) > 0.01
    assert np.array_equal(
        _slice(cng["reference"], rate, 2.0, 12.0),
        np.zeros(int(10.0 * rate)),
    )
    assert np.array_equal(cng["far_end"], np.zeros_like(cng["far_end"]))
    assert _rms(_slice(cng["noise"], rate, 4.0, 12.0)) > 0.0
    assert cng_metadata["cng_measurement_start_s"] == 4.0
