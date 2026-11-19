from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.acoustic import acceptance_batch
from server.acoustic import scene_condition_batch_api
from server.acoustic.dynamic_microphone import (
    DynamicMicrophoneError,
    synthesize_post_rx_microphone,
)
from server import dpdfnet8, nb8_route


def _metadata(*, tap_gain: float = 1.0, echo_gain: float = 1.0) -> dict:
    return {
        "near_arrival_delays_samples": [0, 1],
        "echo_paths": [
            {
                "tap_samples": [0],
                "tap_gains": [tap_gain],
                "echo_gain_linear": echo_gain,
            },
            {
                "tap_samples": [0],
                "tap_gains": [tap_gain],
                "echo_gain_linear": echo_gain,
            },
        ],
        "noise_spatial": {
            "channel_1_recipe": {
                "rho": 0.0,
                "seed2": 7,
                "normalizer": 1.0,
                "noise_channel_scale_linear": 1.0,
            }
        },
    }


def test_post_rx_playback_is_used_for_microphone_echo():
    reference = np.array([0.10, 0.20, 0.00, 0.00], dtype=np.float64)
    noise = np.zeros(4, dtype=np.float64)
    pre_rx = np.array([0.00, 0.00, 0.00, 0.00], dtype=np.float64)
    post_rx = np.array([0.05, 0.10, 0.00, 0.00], dtype=np.float64)

    pre = synthesize_post_rx_microphone(
        reference, noise, pre_rx, _metadata(), sample_rate=16_000
    )
    post = synthesize_post_rx_microphone(
        reference, noise, post_rx, _metadata(), sample_rate=16_000
    )

    assert pre.microphone.shape == (4, 2)
    assert post.microphone.shape == (4, 2)
    assert not np.array_equal(pre.microphone[:, 0], post.microphone[:, 0])
    assert np.allclose(
        post.microphone[:, 0],
        reference + post_rx,
    )
    assert post.reference is not reference
    assert post.capture_gain == 1.0


def test_dual_channel_noise_is_reconstructed_and_reference_is_delayed():
    reference = np.array([0.10, 0.20, 0.30, 0.40], dtype=np.float64)
    noise = np.zeros(4, dtype=np.float64)
    playback = np.zeros(4, dtype=np.float64)
    metadata = _metadata()
    independent = np.random.default_rng(7).normal(size=4)

    mix = synthesize_post_rx_microphone(
        reference, noise, playback, metadata, sample_rate=16_000
    )

    assert np.allclose(mix.microphone[:, 0], reference)
    assert np.allclose(
        mix.microphone[:, 1],
        np.r_[0.0, reference[:-1]] + independent,
    )
    assert mix.channels == 2


def test_capture_gain_scales_microphone_reference_and_noise_not_playback():
    reference = np.array([0.90, 0.00], dtype=np.float64)
    noise = np.array([0.20, 0.00], dtype=np.float64)
    playback = np.array([0.30, 0.00], dtype=np.float64)

    mix = synthesize_post_rx_microphone(
        reference,
        noise,
        playback,
        _metadata(tap_gain=1.0, echo_gain=1.0),
        sample_rate=16_000,
    )

    assert mix.peak_before_gain == pytest.approx(1.4)
    assert mix.capture_gain == pytest.approx(0.94 / 1.4)
    assert np.max(np.abs(mix.microphone)) <= 0.94 + 1e-12
    assert np.allclose(
        mix.microphone[:, 0],
        (reference + playback + noise) * mix.capture_gain,
    )
    assert np.allclose(
        mix.reference,
        reference * mix.capture_gain,
    )
    assert np.allclose(mix.noise, noise * mix.capture_gain)


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"near_arrival_delays_samples": [0, 1]},
        {
            "near_arrival_delays_samples": [0, 1],
            "echo_paths": [],
            "noise_spatial": {},
        },
    ],
)
def test_invalid_metadata_is_rejected(metadata):
    with pytest.raises(DynamicMicrophoneError):
        synthesize_post_rx_microphone(
            np.zeros(4),
            np.zeros(4),
            np.zeros(4),
            metadata,
            sample_rate=16_000,
        )


def test_near_arrival_delay_must_stay_inside_signal():
    metadata = _metadata()
    metadata["near_arrival_delays_samples"] = [0, 4]
    with pytest.raises(
        DynamicMicrophoneError,
        match="inside the delivered signal",
    ):
        synthesize_post_rx_microphone(
            np.zeros(4),
            np.zeros(4),
            np.zeros(4),
            metadata,
            sample_rate=16_000,
        )


def test_run_cli_replaces_dataset_microphone_with_post_rx_mix(
    tmp_path, monkeypatch
):
    from server import alignment_runner

    rate = 16_000
    reference = np.array([0.10, 0.20, 0.00, 0.00], dtype=np.float32)
    noise = np.array([0.01, 0.00, 0.00, 0.00], dtype=np.float32)
    pre_rx = np.array([0.30, 0.00, 0.00, 0.00], dtype=np.float32)
    original_microphone = np.column_stack(
        (reference + noise, reference + noise)
    )
    paths: dict[str, Path] = {}
    for role, samples in {
        "reference": reference,
        "noise": noise,
        "far_end": pre_rx,
        "microphone": original_microphone,
    }.items():
        path = tmp_path / f"{role}.wav"
        sf.write(path, samples, rate, subtype="PCM_16")
        paths[role] = path

    cli = tmp_path / "fake-vcp"
    cli.write_bytes(b"fake executable identity")
    cli.chmod(0o755)
    monkeypatch.setattr(alignment_runner, "resolve_cli", lambda: cli)
    monkeypatch.setattr(
        alignment_runner,
        "resolve_algorithm_cluster",
        lambda cluster: (
            {"profile": "vcp"},
            [],
            {"profile": "vcp"},
            "c" * 64,
        ),
    )
    monkeypatch.setattr(
        alignment_runner,
        "validate_pipeline_artifacts",
        lambda path, output, profile: (
            json.loads(path.read_text()),
            [],
        ),
    )
    monkeypatch.setattr(
        acceptance_batch, "aec_route_requires_capability", lambda config: False
    )

    def fake_run(command, **kwargs):
        output_prefix = Path(command[command.index("--out") + 1])
        if "--rx-processing" in command:
            source = Path(command[command.index("--rx-in") + 1])
            samples, source_rate = sf.read(source, always_2d=True)
            sf.write(
                Path(f"{output_prefix}.rx-spk.wav"),
                samples,
                source_rate,
                subtype="PCM_16",
            )
            report = {
                "processing_profile": "vcp",
                "rx_processing": True,
                "rx_processing_frames": 1,
                "dsp_delay_samples": 0,
            }
        else:
            source = Path(command[command.index("--mic") + 1])
            samples, source_rate = sf.read(source, always_2d=True)
            sf.write(
                Path(f"{output_prefix}.after.wav"),
                samples[:, 0],
                source_rate,
                subtype="PCM_16",
            )
            report = {"dsp_delay_samples": 0}
        Path(f"{output_prefix}.json").write_text(json.dumps(report))
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(acceptance_batch.subprocess, "run", fake_run)
    destination = tmp_path / "out"
    destination.mkdir()
    case = {
        "conditions": {"mode": "WB-wired", "direction": "send"},
        "beam_delays_ms": [0.0, 0.0],
        "dynamic_microphone_from_post_rx": True,
        "provenance": {
            "render_metadata": {
                "near_arrival_delays_samples": [0, 0],
                "echo_paths": [
                    {
                        "tap_samples": [0],
                        "tap_gains": [1.0],
                        "echo_gain_linear": 1.0,
                    },
                    {
                        "tap_samples": [0],
                        "tap_gains": [1.0],
                        "echo_gain_linear": 1.0,
                    },
                ],
                "noise_spatial": {
                    "channel_1_recipe": {
                        "rho": 0.0,
                        "seed2": 7,
                        "normalizer": 1.0,
                        "noise_channel_scale_linear": 1.0,
                    }
                },
            }
        },
    }
    outputs, execution = acceptance_batch._run_cli(
        case, paths, destination, "vcp_default"
    )

    tx_command = execution["commands"][1]
    dynamic_mic_path = Path(tx_command[tx_command.index("--mic") + 1])
    assert dynamic_mic_path.name == "tx.dynamic-microphone.wav"
    assert dynamic_mic_path.is_file()
    assert dynamic_mic_path.resolve() == Path(
        execution["dynamic_microphone"]["path"]
    ).resolve()
    assert Path(tx_command[tx_command.index("--echo") + 1]).resolve() == Path(
        execution["dynamic_microphone"]["playback_path"]
    ).resolve()
    assert execution["dynamic_microphone"]["source"] == "post_rx_runtime_mix"
    assert execution["effective_input_sha256"]["microphone"] == (
        execution["dynamic_microphone"]["sha256"]
    )
    assert execution["effective_input_sha256"]["far_end"] == (
        execution["output_sha256"]["rx"]
    )
    assert paths["microphone"].is_file()
    assert outputs["tx"].is_file()


def test_native_nb8_dynamic_microphone_is_beamformed_before_speex(
    tmp_path,
):
    if (
        not nb8_route.DEFAULT_SPEEX_CLI.is_file()
        or not dpdfnet8.DEFAULT_MODEL_PATH.is_file()
    ):
        pytest.skip("native 8 kHz Speex CLI or DPDFNet model is missing")

    rate = 8_000
    time = np.arange(rate // 2) / rate
    reference = (0.03 * np.sin(2 * np.pi * 510 * time)).astype(np.float32)
    noise = np.zeros_like(reference)
    far_end = (0.02 * np.sin(2 * np.pi * 230 * time)).astype(np.float32)
    placeholder_microphone = np.column_stack(
        (reference + far_end, reference + far_end)
    )
    paths: dict[str, Path] = {}
    for role, samples in {
        "reference": reference,
        "noise": noise,
        "far_end": far_end,
        "microphone": placeholder_microphone,
    }.items():
        path = tmp_path / f"{role}.wav"
        sf.write(path, samples, rate, subtype="PCM_16")
        paths[role] = path

    destination = tmp_path / "out"
    destination.mkdir()
    case = {
        "conditions": {"mode": "NB-wired", "direction": "send"},
        "beam_delays_ms": [0.125, 0.0],
        "dynamic_microphone_from_post_rx": True,
        "provenance": {
            "render_metadata": {
                "near_arrival_delays_samples": [0, 0],
                "echo_paths": [
                    {
                        "tap_samples": [0],
                        "tap_gains": [1.0],
                        "echo_gain_linear": 1.0,
                    },
                    {
                        "tap_samples": [0],
                        "tap_gains": [1.0],
                        "echo_gain_linear": 1.0,
                    },
                ],
                "noise_spatial": {
                    "channel_1_recipe": {
                        "rho": 0.0,
                        "seed2": 7,
                        "normalizer": 1.0,
                        "noise_channel_scale_linear": 1.0,
                    }
                },
            }
        },
    }

    outputs, execution = acceptance_batch._run_cli(
        case,
        paths,
        destination,
        "vcp_default",
    )

    assert set(outputs) == {"rx", "tx"}
    assert execution["algorithm_cluster_id"] == "speex_dpdfnet2_8khz"
    assert execution["no_resample"] is True
    assert execution["fallback_frames"] == 0
    assert execution["dynamic_microphone"]["sample_rate"] == 8_000
    beamforming = execution["native_preprocessing"]["beamforming"]
    assert beamforming["source_channels"] == 2
    assert beamforming["output_channels"] == 1
    assert beamforming["source_path"].endswith("tx.dynamic-microphone.wav")
    assert execution["effective_input_sha256"]["microphone"] == (
        beamforming["output_sha256"]
    )
    assert execution["echo_reference"]["tx_echo_argument_sha256"] == (
        execution["output_sha256"]["rx"]
    )
    assert sf.info(outputs["tx"]).samplerate == 8_000
    assert sf.info(outputs["tx"]).channels == 1


def _dynamic_microphone_client() -> TestClient:
    app = FastAPI()
    app.include_router(scene_condition_batch_api.router)
    return TestClient(app)


def test_dynamic_microphone_endpoint_serves_verified_runtime_wav(
    tmp_path, monkeypatch
):
    path = tmp_path / "tx.dynamic-microphone.wav"
    sf.write(
        path,
        np.zeros((160, 2), dtype=np.float32),
        16_000,
        subtype="PCM_16",
    )
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    calls: list[tuple[str, str]] = []

    def resolve(identifier: str, case_id: str):
        calls.append((identifier, case_id))
        return path, digest

    monkeypatch.setattr(
        scene_condition_batch_api,
        "dynamic_microphone_path",
        resolve,
    )
    identifier = "a" * 32
    case_id = "sce1-wb-wired-double_talk-01-conversation-v2"
    response = _dynamic_microphone_client().get(
        "/api/alignment/acceptance/scene-condition-batches/"
        f"{identifier}/cases/{case_id}/inputs/dynamic-microphone"
    )

    assert response.status_code == 200
    assert response.content == payload
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.headers["etag"] == f'"{digest}"'
    assert "dynamic-microphone.wav" in response.headers[
        "content-disposition"
    ]
    assert calls == [(identifier, case_id)]


def test_dynamic_microphone_endpoint_fails_closed_on_hash_mismatch(
    tmp_path, monkeypatch
):
    path = tmp_path / "tx.dynamic-microphone.wav"
    sf.write(
        path,
        np.zeros((160, 2), dtype=np.float32),
        16_000,
        subtype="PCM_16",
    )
    monkeypatch.setattr(
        scene_condition_batch_api,
        "dynamic_microphone_path",
        lambda identifier, case_id: (path, "0" * 64),
    )

    response = _dynamic_microphone_client().get(
        "/api/alignment/acceptance/scene-condition-batches/"
        f"{'a' * 32}/cases/example/inputs/dynamic-microphone"
    )

    assert response.status_code == 409
    assert "动态麦克风哈希校验失败" in response.text
