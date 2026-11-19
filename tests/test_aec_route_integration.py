"""Real-CLI admission integration across HTTP and batch entry points."""
from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf

from server import synthesize as synth
from server import aec_routes, alignment_runner as alignment
from server.aec_routes import AEC_IDS
from server.alignment_runner import resolve_algorithm_cluster, resolve_cli
from server.app import app
from server.audio import AudioError
from server.ns_routes import resolve_ns_cli
from server.acoustic import acceptance_batch, acoustic_stress_batch, scene_condition_batch
from server.acoustic.acoustic_stress_profiles import stress_profile
from server.acoustic.acoustic_stress_store import SCENE as STRESS_SCENE


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "src/features/enhancement-chain-configs.json"
BATCH_PREFIX = "/api/alignment/acceptance/scene-condition-batches"
DATASET_ID = "cf29475e343052eaffa73dfebad1b0b0"
CASE_ID = "sce1-wb-wired-noise_only-01-conversation-v2"


def _localvqe_cli() -> Path:
    _, _, configuration, _ = resolve_algorithm_cluster("localvqe_aec200k")
    cli = resolve_ns_cli(resolve_cli(), configuration)
    assert cli is not None
    assert cli.is_file(), f"optimized LocalVQE CLI is missing: {cli}"
    return cli


def _must_not_run(label: str):
    def forbidden(*args, **kwargs):
        pytest.fail(f"{label} must not run before DSP capability admission")

    return forbidden


def _mono_pcm16(seconds: float = 0.2, frequency: float = 440.0) -> bytes:
    rate = 16_000
    time = np.arange(round(seconds * rate), dtype=np.float64) / rate
    samples = 0.1 * np.sin(2 * np.pi * frequency * time)
    stream = BytesIO()
    sf.write(stream, samples, rate, format="WAV", subtype="PCM_16")
    return stream.getvalue()


@pytest.mark.parametrize(
    "identifier",
    ["speex_fused_cpu", "speex_fused_dsp", "localvqe_aec200k"],
)
def test_direct_catalog_ids_resolve_explicit_aec_configuration_and_flags(identifier):
    preset, flags, configuration, catalog_sha256 = resolve_algorithm_cluster(
        identifier
    )
    cli = _localvqe_cli()
    _, _, localvqe_configuration, _ = resolve_algorithm_cluster(
        "localvqe_aec200k"
    )
    assert resolve_ns_cli(resolve_cli(), localvqe_configuration) == cli
    assert preset["id"] == identifier
    assert preset["profile"] == "vcp"
    assert configuration["profile"] == "vcp"
    assert configuration["selection"] == preset["selection"]
    assert configuration["selection"]["aec"] == identifier
    assert flags.count("--aec-backend") == 1
    assert flags[flags.index("--aec-backend") + 1] == identifier
    assert len(catalog_sha256) == 64


def test_real_capability_endpoint_reports_cpu_ready_and_dsp_blocked():
    response = TestClient(app).get("/api/alignment/aec-backends")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["schema"] == "itu-aec-capabilities-v1"
    assert len(payload["cli_sha256"]) == 64
    optimized = ROOT / "build/cofilite-opt-20261116/host-v3/cpp/itu_audio_cli"
    assert optimized.is_file()
    assert payload["cli_sha256"] == hashlib.sha256(
        optimized.read_bytes()
    ).hexdigest()

    backends = {entry["id"]: entry for entry in payload["backends"]}
    assert set(backends) == AEC_IDS
    for identifier in ("webrtc", "speex", "speex_fused_cpu"):
        assert backends[identifier]["available"] is True
        assert backends[identifier]["execution_provider"] == "cpu"
    dsp = backends["speex_fused_dsp"]
    assert dsp["available"] is False
    assert dsp["architecture"] == "mdf_complex_pf"
    assert "HiFi" in dsp["reason"]
    localvqe = backends["localvqe_aec200k"]
    assert localvqe["available"] is True
    assert localvqe["architecture"] == "localvqe_daf_neural_mask"
    assert localvqe["execution_provider"] == "cpu"
    assert localvqe["delay_samples"] == 480
    assert localvqe["model_codec_delay_samples_16k"] == 256
    assert localvqe["linear_frontend_delay_samples_16k"] == 0


def test_scene_batch_dsp_is_rejected_before_dataset_or_directory(
    monkeypatch, tmp_path
):
    batch_root = tmp_path / "scene-condition-batches"
    monkeypatch.setattr(scene_condition_batch, "_batch_root", lambda: batch_root)
    monkeypatch.setattr(
        scene_condition_batch,
        "_validated_dataset",
        _must_not_run("scene-condition dataset validation"),
    )
    monkeypatch.setattr(
        scene_condition_batch,
        "_create_batch_directories",
        _must_not_run("scene-condition directory creation"),
    )

    response = TestClient(app).post(
        BATCH_PREFIX,
        json={
            "dataset_id": DATASET_ID,
            "algorithm_cluster_id": "speex_fused_dsp",
            "mode": "WB-wired",
            "scene": "sce1",
            "test_scope": "background_noise",
            "case_id": CASE_ID,
        },
    )
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert "speex_fused_dsp" in detail
    assert "HiFi SDK/provider" in detail
    assert not batch_root.exists()


def test_basic_batch_dsp_is_rejected_before_dataset_lookup(monkeypatch):
    monkeypatch.setattr(
        acceptance_batch,
        "get_dataset",
        _must_not_run("basic batch dataset lookup"),
    )
    monkeypatch.setattr(
        acceptance_batch,
        "_save",
        _must_not_run("basic batch snapshot write"),
    )

    with pytest.raises(ValueError, match="speex_fused_dsp.*HiFi SDK/provider"):
        acceptance_batch.start_batch(
            "missing-basic-dataset",
            "speex_fused_dsp",
            test_scope="basic_uplink",
            mode="WB-wired",
        )


def test_stress_batch_dsp_is_rejected_before_dataset_or_directory(
    monkeypatch, tmp_path
):
    profile = stress_profile("echo", "WB-wired")
    batch_root = tmp_path / "scene-condition-batches"
    monkeypatch.setattr(scene_condition_batch, "_batch_root", lambda: batch_root)
    monkeypatch.setattr(
        acoustic_stress_batch,
        "validated_dataset",
        _must_not_run("stress dataset validation"),
    )
    monkeypatch.setattr(
        scene_condition_batch,
        "_create_batch_directories",
        _must_not_run("stress directory creation"),
    )

    with pytest.raises(ValueError, match="speex_fused_dsp.*HiFi SDK/provider"):
        acoustic_stress_batch.start_batch(
            profile.dataset_id,
            "speex_fused_dsp",
            mode=profile.mode,
            scene=STRESS_SCENE,
        )
    assert not batch_root.exists()


def test_synthesis_dsp_rejection_keeps_catalog_and_real_cli_but_writes_nothing(
    monkeypatch, tmp_path
):
    _, _, configuration, _ = resolve_algorithm_cluster("speex_fused_dsp")
    monkeypatch.setattr(synth, "ROOT", tmp_path)
    cli = _localvqe_cli()
    monkeypatch.setenv("ITU_CLI", str(cli))
    assert synth.CHAIN_CATALOG == CATALOG
    assert synth.cli_path() == cli

    with pytest.raises(AudioError, match="speex_fused_dsp.*HiFi SDK/provider"):
        synth.synthesize(
            "microphone.wav",
            _mono_pcm16(frequency=440.0),
            "reference.wav",
            _mono_pcm16(frequency=330.0),
            rate=16_000,
            profile="vcp",
            chain=configuration,
            microphone_mode="channel0",
        )
    assert not (tmp_path / "build" / "demo-runs").exists()


@pytest.mark.parametrize(
    ("rate", "capture_channels", "reference_channels", "bf", "reason"),
    [
        (24_000, 2, 1, "das", "sample_rate=16000"),
        (16_000, 2, 2, "das", "reference_channels=1"),
        (16_000, 2, 1, "off", "capture_channels"),
    ],
)
def test_alignment_admission_checks_real_audio_headers_before_capability(
    monkeypatch, tmp_path, rate, capture_channels, reference_channels, bf, reason
):
    _, _, configuration, _ = resolve_algorithm_cluster("speex_fused_dsp")
    configuration["selection"]["bf"] = bf
    microphone = tmp_path / "mic.wav"
    playback = tmp_path / "post-rx.wav"
    sf.write(microphone, np.full((rate // 5, capture_channels), 0.1), rate)
    sf.write(playback, np.full((rate // 5, reference_channels), 0.05), rate)
    monkeypatch.setattr(alignment, "DEFAULT_MIC", microphone)
    monkeypatch.setattr(alignment, "DEFAULT_REF", playback)
    monkeypatch.setattr(
        aec_routes, "read_aec_capabilities", _must_not_run("DSP capability query")
    )

    with pytest.raises(ValueError, match=reason):
        alignment._admit_alignment_aec(_localvqe_cli(), configuration)


def test_alignment_localvqe_admits_explicit_stereo_reference_channel(
    monkeypatch, tmp_path
):
    _, _, configuration, _ = resolve_algorithm_cluster("localvqe_aec200k")
    microphone = tmp_path / "mic.wav"
    playback = tmp_path / "stereo-playback.wav"
    sf.write(microphone, np.full(3200, 0.1), 16_000)
    sf.write(playback, np.full((3200, 2), 0.05), 16_000)
    monkeypatch.setattr(alignment, "DEFAULT_MIC", microphone)
    monkeypatch.setattr(alignment, "DEFAULT_REF", playback)

    admitted = alignment._admit_alignment_aec(_localvqe_cli(), configuration)
    assert admitted is not None
    assert admitted["id"] == "localvqe_aec200k"

    without_channel = {
        key: value for key, value in configuration.items() if key != "extra_flags"
    }
    with pytest.raises(ValueError, match="reference_channels=1"):
        alignment._admit_alignment_aec(_localvqe_cli(), without_channel)


def test_alignment_http_rejects_stereo_render_before_running_suite(monkeypatch, tmp_path):
    microphone = tmp_path / "mic.wav"
    playback = tmp_path / "post-rx.wav"
    sf.write(microphone, np.full((3200, 2), 0.1), 16_000)
    sf.write(playback, np.full((3200, 2), 0.05), 16_000)
    monkeypatch.setattr(alignment, "DEFAULT_MIC", microphone)
    monkeypatch.setattr(alignment, "DEFAULT_REF", playback)
    monkeypatch.setattr(
        alignment, "run_alignment_suite", _must_not_run("alignment suite")
    )
    monkeypatch.setattr(
        aec_routes, "read_aec_capabilities", _must_not_run("DSP capability query")
    )

    response = TestClient(app).post(
        "/api/alignment/run",
        json={"scope": "full", "algorithm_cluster_id": "speex_fused_dsp"},
    )
    assert response.status_code == 422, response.text
    assert "reference_channels=1" in response.json()["detail"]


@pytest.mark.parametrize(
    ("rate", "capture_channels", "reference_channels", "reason"),
    [
        (24_000, 1, 1, "sample_rate=16000"),
        (16_000, 1, 2, "reference_channels=1"),
        (16_000, 2, 1, "capture_channels"),
    ],
)
def test_synthesis_rejects_dsp_input_envelope_before_artifacts(
    monkeypatch, tmp_path, rate, capture_channels, reference_channels, reason
):
    _, _, configuration, _ = resolve_algorithm_cluster("speex_fused_dsp")
    configuration["selection"]["bf"] = "off"
    streams = []
    for channels, amplitude in ((capture_channels, 0.1), (reference_channels, 0.05)):
        stream = BytesIO()
        sf.write(stream, np.full((rate // 5, channels), amplitude), rate, format="WAV")
        streams.append(stream.getvalue())
    monkeypatch.setattr(synth, "ROOT", tmp_path)
    cli = _localvqe_cli()
    monkeypatch.setenv("ITU_CLI", str(cli))
    monkeypatch.setattr(
        aec_routes, "read_aec_capabilities", _must_not_run("DSP capability query")
    )

    with pytest.raises(AudioError, match=reason):
        synth.synthesize(
            "microphone.wav", streams[0], "post-rx.wav", streams[1],
            rate=rate, profile="vcp", chain=configuration, microphone_mode="channel0",
        )
    assert not (tmp_path / "build" / "demo-runs").exists()


@pytest.mark.parametrize(
    ("reference_channels", "bf", "reason"),
    [(2, "das", "reference_channels=1"), (1, "off", "capture_channels")],
)
def test_batch_rejects_dsp_input_envelope_before_rx_or_tx(
    monkeypatch, tmp_path, reference_channels, bf, reason
):
    _, _, configuration, _ = resolve_algorithm_cluster("speex_fused_dsp")
    configuration["selection"]["bf"] = bf
    flags, configuration, digest = synth._chain_flags("vcp", configuration)
    monkeypatch.setattr(
        alignment, "resolve_algorithm_cluster",
        lambda identifier: ({"profile": "vcp"}, flags, configuration, digest),
    )
    monkeypatch.setattr(
        aec_routes, "read_aec_capabilities", _must_not_run("DSP capability query")
    )
    paths = {}
    for role, channels in (("microphone", 2), ("far_end", reference_channels)):
        path = tmp_path / f"{role}.wav"
        sf.write(path, np.full((3200, channels), 0.1), 16_000)
        paths[role] = path
    destination = tmp_path / "result"
    destination.mkdir()
    case = {
        "conditions": {"direction": "send", "mode": "WB-wired"},
        "beam_delays_ms": [0, 0.1],
    }
    with pytest.raises(ValueError, match=reason):
        acceptance_batch._run_cli(case, paths, destination, "speex_fused_dsp")
    assert list(destination.iterdir()) == []

# Admission tests exercise the current dataset; retired IDs have a separate 410 contract.
from server.acoustic.dataset_revisions_v2 import revision_scene_profile
DATASET_ID = revision_scene_profile("sce1", "WB-wired").dataset_id
