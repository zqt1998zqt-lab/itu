"""Real CPU replay and negative controls for distinct CPU/HiFi AEC routes."""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import soundfile as sf
from server.alignment_runner import resolve_algorithm_cluster, resolve_cli
from server.ns_routes import resolve_ns_cli

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "build/vcp-unified-final-20261027/cpp/itu_audio_cli"
LOCALVQE_MODEL_SHA256 = (
    "b6e43138588a83bfe903ab5e143b4020b91c1e1629f5a575ac5855ff0003c731"
)
LOCALVQE_FRONTEND_SHA256 = (
    "76510c8250a3af232a6f241643683dad94d9b17f3c184da37870644005b77573"
)
LOCALVQE_PROVIDER_SHA256 = (
    "8a46866f562253a7dc7c17ec9a1d282b954c100e608287d044eac8c8890afe60"
)


def _localvqe_cli() -> Path:
    _, _, configuration, _ = resolve_algorithm_cluster("localvqe_aec200k")
    cli = resolve_ns_cli(resolve_cli(), configuration)
    assert cli is not None
    assert cli.is_file(), f"optimized LocalVQE CLI is missing: {cli}"
    return cli


@pytest.fixture
def pair(tmp_path):
    _localvqe_cli()
    rate = 16000
    rng = np.random.default_rng(54321)
    far = rng.normal(0, 0.04, rate)
    echo = np.r_[np.zeros(320), far[:-320]] * 0.6
    near = 0.025 * np.sin(np.arange(rate) * 2 * np.pi * 570 / rate)
    mic = near + echo + rng.normal(0, 0.0004, rate)
    paths = {"mic": tmp_path / "mic.wav", "far": tmp_path / "far.wav"}
    sf.write(paths["mic"], mic, rate, subtype="PCM_16")
    sf.write(paths["far"], far, rate, subtype="PCM_16")
    return paths


def run(binary, pair, prefix, backend="speex_fused_cpu"):
    return subprocess.run(
        [str(binary), "--profile", "vcp_unified", "--no-rx-processing",
         "--aec-backend", backend, "--mic", str(pair["mic"]),
         "--echo", str(pair["far"]), "--no-bf", "--out", str(prefix)],
        capture_output=True, text=True, timeout=30,
    )


def test_cpu_route_emits_actual_backend_identity(pair, tmp_path):
    prefix = tmp_path / "cpu"
    result = run(_localvqe_cli(), pair, prefix)
    assert result.returncode == 0, result.stderr
    report = json.loads(prefix.with_suffix(".json").read_text())
    assert report["aec_backend_requested"] == "speex_fused_cpu"
    assert report["aec_backend_effective"] == "speex_mdf_nlms_fused_cpu"
    assert report["aec_execution_provider"] == "cpu"
    assert report["aec_dsp_inferences"] == 0
    samples, rate = sf.read(prefix.with_suffix(".after.wav"))
    assert rate == 16000 and np.isfinite(samples).all()
    assert np.any(samples)


def test_localvqe_route_runs_streaming_cpu_model_and_emits_identity(pair, tmp_path):
    prefix = tmp_path / "localvqe"
    result = run(_localvqe_cli(), pair, prefix, "localvqe_aec200k")
    assert result.returncode == 0, result.stderr
    report = json.loads(prefix.with_suffix(".json").read_text())
    assert report["aec_backend_requested"] == "localvqe_aec200k"
    assert report["aec_backend_effective"] == "localvqe_aec200k_cpu"
    assert report["aec_execution_provider"] == "cpu"
    assert report["aec_device_id"] == "host_cpu"
    assert report["aec_model_id"] == "localvqe_aec200k"
    assert report["aec_model_sha256"] == LOCALVQE_MODEL_SHA256
    assert report["aec_frontend_sha256"] == LOCALVQE_FRONTEND_SHA256
    assert report["aec_provider_sha256"] == LOCALVQE_PROVIDER_SHA256
    assert report["aec_neural_delay_samples"] == 256
    assert report["aec_hop_adapter_delay_samples_16k"] == 224
    assert report["aec_algorithm_delay_samples_16k"] == 480
    assert report["aec_model_codec_delay_samples_16k"] == 256
    assert report["aec_linear_frontend_delay_samples_16k"] == 0
    assert report["aec_localvqe_inferences"] > 0
    processed_frames = report["aec_localvqe_processed_frames"]
    input_callbacks = (
        report["input_samples_per_channel"] + report["frame_size"] - 1
    ) // report["frame_size"]
    assert processed_frames >= input_callbacks
    assert processed_frames > input_callbacks
    assert report["aec_route_processed_frames"] == processed_frames
    expected_inferences = (
        report["input_samples_per_channel"]
        + report["aec_neural_delay_samples"]
        + 255
    ) // 256
    assert report["aec_localvqe_inferences"] == expected_inferences
    assert report["aec_declared_source_samples"] == report[
        "input_samples_per_channel"
    ]
    assert report["aec_native_hop_samples"] == 256
    assert report["aec_native_consumed_samples"] == expected_inferences * 256
    assert report["aec_native_consumed_samples"] >= (
        report["input_samples_per_channel"]
        + report["aec_neural_delay_samples"]
    )
    assert report["aec_native_consumed_samples"] < (
        report["input_samples_per_channel"]
        + report["aec_neural_delay_samples"]
        + 256
    )
    assert report["aec_localvqe_inferences"] < processed_frames
    assert report["aec_localvqe_fallback_frames"] == 0
    samples, rate = sf.read(prefix.with_suffix(".after.wav"))
    assert rate == 16000 and np.isfinite(samples).all()
    assert np.any(samples)


def test_localvqe_explicitly_selects_one_playback_channel(pair, tmp_path):
    far, rate = sf.read(pair["far"])
    stereo_far = tmp_path / "stereo-far.wav"
    sf.write(stereo_far, np.column_stack([far, far * 0.5]), rate, subtype="PCM_16")
    stereo_pair = {**pair, "far": stereo_far}
    prefix = tmp_path / "localvqe-stereo-reference"
    result = subprocess.run(
        [str(_localvqe_cli()), "--profile", "vcp", "--no-rx-processing",
         "--aec-backend", "localvqe_aec200k",
         "--localvqe-reference-channel", "0",
         "--mic", str(stereo_pair["mic"]), "--echo", str(stereo_pair["far"]),
         "--no-bf", "--out", str(prefix)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(prefix.with_suffix(".json").read_text())
    assert report["reference_channels"] == 1
    assert report["reference_channels_original"] == 2
    assert report["localvqe_reference_channel"] == 0


def test_route_extension_preserves_cpu_pcm_against_prior_binary(pair, tmp_path):
    if not PRIOR.is_file():
        pytest.skip("prior CPU CLI not present")
    for binary, name in ((PRIOR, "prior"), (_localvqe_cli(), "current")):
        result = run(binary, pair, tmp_path / name)
        assert result.returncode == 0, result.stderr
    prior = (tmp_path / "prior.after.wav").read_bytes()
    current = (tmp_path / "current.after.wav").read_bytes()
    assert hashlib.sha256(prior).hexdigest() == hashlib.sha256(current).hexdigest()


def test_dsp_selection_cannot_execute_cpu_or_create_success_artifacts(pair, tmp_path):
    cli = _localvqe_cli()
    capabilities = json.loads(subprocess.run(
        [str(cli), "--aec-capabilities"], capture_output=True,
        text=True, check=True, timeout=10,
    ).stdout)
    dsp = next(row for row in capabilities["backends"] if row["id"] == "speex_fused_dsp")
    if dsp["available"]:
        pytest.skip("test requires default host build without a HiFi provider")
    prefix = tmp_path / "dsp"
    result = run(cli, pair, prefix, "speex_fused_dsp")
    assert result.returncode != 0
    assert "speex_fused_dsp" in result.stderr and "HiFi" in result.stderr
    assert not prefix.with_suffix(".after.wav").exists()
    assert not prefix.with_suffix(".json").exists()


def test_rx_only_does_not_load_the_tx_dsp_model(pair, tmp_path):
    prefix = tmp_path / "rx"
    result = subprocess.run(
        [str(_localvqe_cli()), "--profile", "vcp", "--aec-backend", "speex_fused_dsp",
         "--rx-in", str(pair["far"]), "--rx-processing", "--out", str(prefix)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(prefix.with_suffix(".json").read_text())
    assert report["mode"] == "downlink_only"
    assert report["rx_processing_frames"] > 0
    assert prefix.with_suffix(".rx-spk.wav").is_file()


def test_dsp_v1_rejects_other_rates_without_cpu_fallback(pair, tmp_path):
    for role in pair:
        path = tmp_path / (role + "-48k.wav")
        sf.write(path, np.zeros(4800), 48000, subtype="PCM_16")
        pair[role] = path
    result = run(_localvqe_cli(), pair, tmp_path / "unsupported", "speex_fused_dsp")
    assert result.returncode != 0
    assert "speex_fused_dsp requires 16 kHz" in result.stderr
    assert not (tmp_path / "unsupported.after.wav").exists()
