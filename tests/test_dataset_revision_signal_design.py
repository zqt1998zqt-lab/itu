from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf


PROJECT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT / "scripts/dataset_revision_signal_design.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "itu_dataset_revision_signal_design_test",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array, dtype=np.float64))))


def _mixed_tone(rate: int = 16_000, seconds: float = 0.5) -> np.ndarray:
    frames = int(round(rate * seconds))
    time = np.arange(frames, dtype=np.float64) / rate
    return (
        0.16 * np.sin(2.0 * math.pi * 280.0 * time)
        + 0.09 * np.sin(2.0 * math.pi * 1_100.0 * time + 0.2)
        + 0.04 * np.sin(2.0 * math.pi * 6_100.0 * time + 0.4)
    )


def _write_noise_wav(path: Path, seed: int, rate: int = 16_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    frames = rate * 4
    time = np.arange(frames, dtype=np.float64) / rate
    values = (
        0.05 * np.sin(2.0 * math.pi * (70.0 + seed) * time)
        + 0.02 * rng.standard_normal(frames)
    )
    sf.write(path, values, rate, subtype="PCM_16")


def _record(path: Path, scene: str, root: Path) -> dict:
    return {
        "path": str(path.relative_to(root)),
        "sha256": _sha(path),
        "sample_rate": 16_000,
        "scene": scene,
        "kind": "noise",
        "dataset": "DEMAND",
        "license": "CC-BY-SA-3.0",
    }


def test_transport_wired_identity_silence_and_proxy_modes_are_deterministic():
    module = _load()
    rate = 16_000
    source = _mixed_tone(rate)
    original = source.copy()

    wired, wired_info = module.apply_transport(source, rate, "WB-wired", 77)
    assert np.array_equal(wired, source)
    assert wired_info["direction"] == "far_end_to_rx_input"
    assert wired_info["codec_conformance"] is False
    assert wired_info["actual_codec"] is False
    assert wired_info["transformation_applied"] is False

    outputs = {}
    for mode in ("NB-wireless", "WB-wireless", "NB-bt", "WB-bt"):
        first, info = module.apply_transport(source, rate, mode, 91)
        second, second_info = module.apply_transport(source, rate, mode, 91)
        assert np.array_equal(first, second)
        assert info["output_sha256_f32le"] == second_info["output_sha256_f32le"]
        assert len(first) == len(source)
        assert np.isfinite(first).all()
        assert np.max(np.abs(first)) <= 1.0
        assert 0.9 <= _rms(first) / _rms(source) <= 1.1
        assert info["direction"] == "far_end_to_rx_input"
        assert info["codec_conformance"] is False
        assert info["not_actual_codec"] == ["CVSD", "mSBC"]
        outputs[mode] = first

    assert not np.array_equal(outputs["NB-wireless"], outputs["WB-wireless"])
    assert not np.array_equal(outputs["NB-bt"], outputs["WB-bt"])
    assert not np.array_equal(outputs["WB-wireless"], outputs["WB-bt"])
    assert np.array_equal(source, original)

    zeros = np.zeros(rate // 4, dtype=np.float64)
    for mode in ("NB-wired", "WB-wireless", "NB-bt", "WB-bt"):
        silence, info = module.apply_transport(zeros, rate, mode, 123)
        assert np.count_nonzero(silence) == 0
        assert info["digital_silence_preserved"] is True

    with pytest.raises(ValueError, match="finite"):
        module.apply_transport(np.array([0.0, np.nan]), rate, "WB-wired", 1)


def test_highway_noise_uses_verified_tbus_and_straffic_without_tcar(tmp_path):
    module = _load()
    root = tmp_path / "sources"
    tbus_path = root / "noise/TBUS-ch01.wav"
    straffic_path = root / "noise/STRAFFIC-ch01.wav"
    _write_noise_wav(tbus_path, 1)
    _write_noise_wav(straffic_path, 2)
    bank = SimpleNamespace(
        root=root,
        used={},
        noises={
            "noise/TBUS-ch01.wav": _record(tbus_path, "TBUS", root),
            "noise/STRAFFIC-ch01.wav": _record(straffic_path, "STRAFFIC", root),
        },
    )

    noise, info = module.highway_noise(bank, "WB-wired", 16_000, 2026, 1.0)
    repeated, repeated_info = module.highway_noise(
        bank, "WB-wired", 16_000, 2026, 1.0,
    )
    assert np.array_equal(noise, repeated)
    assert info["output_rms"] == repeated_info["output_rms"]
    assert len(noise) == 16_000
    assert np.isfinite(noise).all()
    assert np.max(np.abs(noise)) <= 1.0
    assert info["source_scenes"] == ["TBUS", "STRAFFIC"]
    assert info["tcar_excluded"] is True
    assert "TCAR" not in {row["scene"] for row in info["components"]}
    assert info["actual_highway_recording"] is False
    assert info["actual_vehicle_speed_recording"] is False
    assert info["mix"]["synthetic_noise_fraction"] == 0.0
    assert info["wind_shaping"]["maximum_reinforcement_gain_linear"] == 0.10
    assert info["repeated"] is False and info["looped"] is False
    assert {
        "start_sample",
        "path",
        "sha256",
        "source_sample_rate_hz",
    }.issubset(info)
    assert len({row["start_sample"] for row in info["components"]}) == 2
    assert set(bank.used) == {str(tbus_path.resolve()), str(straffic_path.resolve())}

    tampered_path = root / "noise/TBUS-tampered.wav"
    _write_noise_wav(tampered_path, 3)
    bad_bank = SimpleNamespace(
        root=root,
        used={},
        noises={
            "noise/TBUS-tampered.wav": {
                **_record(tbus_path, "TBUS", root),
                "path": "noise/TBUS-tampered.wav",
            },
            "noise/STRAFFIC-ch01.wav": _record(straffic_path, "STRAFFIC", root),
        },
    )
    with pytest.raises(ValueError, match="SHA mismatch"):
        module.highway_noise(bad_bank, "WB-wired", 16_000, 1, 0.5)

    missing_bank = SimpleNamespace(
        root=root,
        used={},
        noises={"noise/TBUS-ch01.wav": _record(tbus_path, "TBUS", root)},
    )
    with pytest.raises(ValueError, match="STRAFFIC"):
        module.highway_noise(missing_bank, "WB-wired", 16_000, 1, 0.5)


def _observed_fixture(rate: int = 16_000):
    frames = rate
    time = np.arange(frames, dtype=np.float64) / rate
    reference = 0.08 * np.sin(2.0 * math.pi * 220.0 * time)
    noise = 0.01 * np.sin(2.0 * math.pi * 43.0 * time)
    overlap = np.zeros(frames, dtype=bool)
    overlap[int(0.25 * rate):int(0.5 * rate)] = True
    echo0 = np.zeros(frames, dtype=np.float64)
    echo0[overlap] = 0.04 * np.sin(2.0 * math.pi * 500.0 * time[overlap])
    microphone = np.column_stack((reference + echo0 + noise, reference))
    design = {
        "ser_db_nominal_target": -6.0,
        "ser_db_effective_target": -5.0,
        "ser_db_measured": -7.0,
        "ser_db_definition": "legacy far_only over near_active",
        "activity_segments": [
            {"start_s": 0.0, "end_s": 0.25, "label": "near_only"},
            {"start_s": 0.25, "end_s": 0.5, "label": "double_talk"},
            {"start_s": 0.5, "end_s": 1.0, "label": "far_only"},
        ],
    }
    signals = {
        "reference": reference,
        "noise": noise,
        "microphone": microphone,
    }
    return signals, design, overlap


def test_ser_aliases_keep_legacy_sign_and_observed_ser_uses_one_overlap_mask():
    module = _load()
    signals, design, overlap = _observed_fixture()
    aliased = module.add_ser_aliases(design)
    assert aliased["near_to_echo_ser_db_target"] == 6.0
    assert aliased["near_to_echo_ser_db_effective"] == 5.0
    assert aliased["near_to_echo_ser_db_measured"] == 7.0
    assert aliased["echo_to_near_esr_db_target"] == -6.0
    assert aliased["echo_to_near_esr_db_effective"] == -5.0
    assert aliased["echo_to_near_esr_db_measured"] == -7.0
    assert "legacy masks" in aliased["near_to_echo_ser_db_definition"]
    assert design["ser_db_measured"] == -7.0

    observed = module.add_observed_ser(signals, aliased, 16_000)
    echo0 = signals["microphone"][:, 0] - signals["reference"] - signals["noise"]
    expected = 20.0 * math.log10(
        _rms(signals["reference"][overlap]) / _rms(echo0[overlap])
    )
    assert observed["near_to_echo_overlap_ser_db_measured"] == pytest.approx(
        expected,
        abs=1e-12,
    )
    assert observed["echo_to_near_overlap_esr_db_measured"] == pytest.approx(
        -expected,
        abs=1e-12,
    )
    assert observed["ser_observation_window_s"] == pytest.approx(0.25)
    assert observed["ser_db_nominal_target"] == -6.0
    assert observed["ser_db_measured"] == -7.0

    no_overlap = deepcopy(design)
    no_overlap["activity_segments"] = [
        {"start_s": 0.0, "end_s": 1.0, "label": "near_only"},
    ]
    unavailable = module.add_observed_ser(signals, no_overlap, 16_000)
    assert unavailable["near_to_echo_overlap_ser_db_measured"] is None
    assert unavailable["echo_to_near_overlap_esr_db_measured"] is None
    assert unavailable["ser_observation_window_s"] is None

    inapplicable = module.add_ser_aliases({
        "ser_db_target_applicable": False,
        "ser_db_nominal_target": 0.0,
        "ser_db_effective_target": 0.0,
        "ser_db_measured": 0.0,
    })
    assert inapplicable["near_to_echo_ser_db_target"] is None
    assert inapplicable["near_to_echo_ser_db_effective"] is None
    assert inapplicable["near_to_echo_ser_db_measured"] is None


def _short_overlap_fixture(rate: int = 16_000):
    frames = rate
    time = np.arange(frames, dtype=np.float64) / rate
    reference = 0.10 * np.sin(2.0 * math.pi * 230.0 * time)
    far_end = 0.06 * np.sin(2.0 * math.pi * 610.0 * time + 0.1)
    noise0 = 0.012 * np.sin(2.0 * math.pi * 47.0 * time)
    noise0[: int(0.5 * rate)] *= 5.0
    rho = 0.3
    seed2 = 17
    normalizer = 1.0
    scale = 0.02
    independent = np.random.default_rng(seed2).normal(size=frames)
    noise1 = (
        rho * (noise0 / scale)
        + math.sqrt(1.0 - rho * rho) * independent
    ) / normalizer * scale
    paths = [
        {"tap_samples": [0], "tap_gains": [0.20]},
        {"tap_samples": [0], "tap_gains": [0.15]},
    ]
    echo_gain = 0.8
    echo0 = far_end * 0.20 * echo_gain
    echo1 = far_end * 0.15 * echo_gain
    delayed_reference = np.r_[0.0, reference[:-1]]
    microphone = np.column_stack((
        reference + echo0 + noise0,
        delayed_reference + echo1 + noise1,
    ))
    design = {
        "workload_category": "short_overlap",
        "sample_rate_hz": rate,
        "activity_segments": [
            {"start_s": 0.0, "end_s": 0.5, "label": "near_only"},
            {"start_s": 0.5, "end_s": 0.6, "label": "double_talk"},
            {"start_s": 0.6, "end_s": 1.0, "label": "far_only"},
        ],
        "near_arrival_delays_samples": [0, 1],
        "echo_path_segments": [{
            "start_s": 0.0,
            "end_s": 1.0,
            "paths": paths,
        }],
        "echo_paths": [],
        "echo_gain_linear": echo_gain,
        "noise_snr_db_target": 5.0,
        "noise_snr_db_measured": 5.0,
        "noise_spatial": {
            "channel_1_recipe": {
                "rho": rho,
                "seed2": seed2,
                "normalizer": normalizer,
                "noise_channel_scale_linear": scale,
            },
        },
    }
    signals = {
        "microphone": microphone,
        "reference": reference,
        "far_end": far_end,
        "noise": noise0,
    }
    return signals, design, echo0, echo1, noise1


def test_short_overlap_calibration_uses_overlap_window_and_preserves_components():
    module = _load()
    signals, design, echo0, echo1, old_noise1 = _short_overlap_fixture()
    original_signals = {key: value.copy() for key, value in signals.items()}
    original_design = deepcopy(design)
    overlap = np.zeros(16_000, dtype=bool)
    overlap[8_000:9_600] = True

    calibrated, calibrated_design = module.calibrate_short_overlap(
        signals,
        design,
        16_000,
        12.0,
    )
    repeated, _ = module.calibrate_short_overlap(signals, design, 16_000, 12.0)
    assert np.array_equal(calibrated["noise"], repeated["noise"])
    assert np.array_equal(calibrated["reference"], signals["reference"])
    assert np.array_equal(calibrated["far_end"], signals["far_end"])
    assert np.array_equal(signals["reference"], original_signals["reference"])
    assert np.array_equal(signals["noise"], original_signals["noise"])
    assert design["noise_spatial"] == original_design["noise_spatial"]

    scale = calibrated_design["short_overlap_calibration"][
        "noise_scale_factor_linear"
    ]
    assert np.allclose(
        calibrated["noise"],
        signals["noise"] * scale,
        rtol=0.0,
        atol=1e-12,
    )
    measured = 20.0 * math.log10(
        _rms(calibrated["reference"][overlap])
        / _rms(calibrated["noise"][overlap])
    )
    assert measured == pytest.approx(12.0, abs=1e-9)
    assert calibrated_design["noise_snr_db_target"] == 12.0
    assert calibrated_design["noise_snr_db_measured"] == pytest.approx(12.0)
    assert calibrated_design["short_overlap_calibration"][
        "observation_window_s"
    ] == pytest.approx(0.1)

    near_only = np.zeros(16_000, dtype=bool)
    near_only[:8_000] = True
    near_only_snr = 20.0 * math.log10(
        _rms(calibrated["reference"][near_only])
        / _rms(calibrated["noise"][near_only])
    )
    assert near_only_snr != pytest.approx(12.0, abs=0.5)

    delayed_reference = np.r_[0.0, calibrated["reference"][:-1]]
    residual0 = np.max(np.abs(
        calibrated["microphone"][:, 0]
        - calibrated["reference"]
        - echo0
        - calibrated["noise"]
    ))
    residual1 = np.max(np.abs(
        calibrated["microphone"][:, 1]
        - delayed_reference
        - echo1
        - (
            calibrated["noise"] * 0.0
            + module._noise_channel_from_recipe(
                calibrated["noise"],
                calibrated_design,
            )[0]
        )
    ))
    assert residual0 <= 1e-12
    assert residual1 <= 1e-12
    rebuilt_noise1, _ = module._noise_channel_from_recipe(
        calibrated["noise"],
        calibrated_design,
    )
    assert np.allclose(rebuilt_noise1, old_noise1 * scale, atol=1e-12)

    non_short = deepcopy(design)
    non_short["workload_category"] = "balanced"
    with pytest.raises(ValueError, match="short_overlap"):
        module.calibrate_short_overlap(signals, non_short, 16_000, 12.0)

    silent_noise = signals["noise"].copy()
    silent_noise[overlap] = 0.0
    silent_mic = np.column_stack((
        signals["reference"] + echo0 + silent_noise,
        np.r_[0.0, signals["reference"][:-1]] + echo1,
    ))
    with pytest.raises(ValueError, match="noise is silent"):
        module.calibrate_short_overlap(
            {
                **signals,
                "noise": silent_noise,
                "microphone": silent_mic,
            },
            design,
            16_000,
            12.0,
        )
