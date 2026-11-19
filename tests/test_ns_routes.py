"""Contract tests for pinned external NS routes and replay validation."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from server import cofilite_stream, ns_model_worker, ns_routes


def _configuration(ns: str = "cofilite") -> dict:
    return {
        "profile": "vcp",
        # This module keeps the historical capture/replay contract as its
        # explicit diagnostic; production 48 kHz routes omit this key and use
        # the native callback worker.
        "ns_execution_mode": "offline_replay",
        "selection": {
            "bf": "off", "aec": "localvqe_aec200k", "ns": ns,
            "eq": "three_band", "agc": "adaptive", "drc": "speech", "limiter": "off",
        },
    }


def _command(
    tmp_path: Path, *, mic: bool = True, sample_rate: int = 16_000,
    aec_backend: str | None = None, no_aec: bool = False,
    include_aec: bool = True,
) -> list[str]:
    cli = tmp_path / "itu_audio_cli"
    if not cli.is_file():
        cli.write_bytes(b"#!/bin/sh\nexit 0\n")
        cli.chmod(0o755)
    selected_aec = aec_backend or (
        "webrtc" if sample_rate == 48_000 else "localvqe_aec200k"
    )
    command = [
        str(cli), "--profile", "vcp", "--ns-backend", "webrtc",
        "--echo", str(tmp_path / "ref.wav"), "--out", str(tmp_path / "out"),
    ]
    if include_aec:
        if no_aec:
            command.insert(1, "--no-aec")
        else:
            command[1:1] = ["--aec-backend", selected_aec]
    if mic:
        command[1:1] = ["--mic", str(tmp_path / "mic.wav")]
    return command


def _float_payload_sha(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f4").tobytes()).hexdigest()


def _write_float(path: Path, values: np.ndarray, rate: int = 16_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(values, dtype=np.float32), rate, subtype="FLOAT")


def _write_mic(tmp_path: Path, rate: int = 16_000, samples: int = 320) -> None:
    sf.write(tmp_path / "mic.wav", np.zeros(samples, dtype=np.float32), rate, subtype="PCM_16")


def _write_stages(path: Path, prefix: Path, input_values: np.ndarray,
                  tx_ns_values: np.ndarray | None = None,
                  sample_rate: int = 16_000) -> None:
    input_path = Path(str(prefix) + ".tx_ns_input.float.wav")
    _write_float(input_path, input_values, sample_rate)
    stages = [{
        "id": "tx_ns_input", "direction": "tx", "sample_rate": sample_rate,
        "channels": 1, "file": str(input_path), "samples_per_channel": len(input_values),
        "fixed_delay_samples": 0,
    }]
    if tx_ns_values is not None:
        tx_path = Path(str(prefix) + ".tx_ns.float.wav")
        _write_float(tx_path, tx_ns_values, sample_rate)
        stages.append({
            "id": "tx_ns", "direction": "tx", "sample_rate": sample_rate,
            "channels": 1, "file": str(tx_path), "samples_per_channel": len(tx_ns_values),
            "fixed_delay_samples": 0,
        })
    path.write_text(json.dumps({"schema_version": 2, "pipeline_stages": stages}), encoding="utf-8")


def _base_report(
    *, sample_rate: int = 16_000, frame_size: int = 160,
    input_samples_per_channel: int = 3200,
    aec_backend: str = "localvqe_aec200k",
) -> dict:
    effective = {
        "localvqe_aec200k": "localvqe_aec200k_cpu",
        "webrtc": "webrtc_aec3",
    }.get(aec_backend, aec_backend)
    report = {
        "sample_rate": sample_rate, "frame_size": frame_size,
        "microphones": 1, "reference_channels": 1,
        "input_samples_per_channel": input_samples_per_channel,
        "aec_backend": effective, "aec_backend_requested": aec_backend,
        "aec_backend_effective": effective,
        "aec_processing_mode": "causal_streaming",
    }
    if aec_backend == "localvqe_aec200k":
        report.update({
            "aec_model_sha256": "1" * 64,
            "aec_frontend_sha256": "2" * 64, "aec_provider_sha256": "3" * 64,
            "aec_execution_provider": "cpu", "aec_device_id": "host_cpu",
            "aec_model_id": "localvqe_aec200k",
            "aec_neural_delay_samples": 256,
            "aec_hop_adapter_delay_samples_16k": 224,
            "aec_algorithm_delay_samples_16k": 480,
            "aec_model_codec_delay_samples_16k": 256,
            "aec_linear_frontend_delay_samples_16k": 0,
        })
    return report


def _cofilite_report() -> dict:
    configuration = _configuration("cofilite")
    return {
        **ns_routes.ns_route_identity(configuration),
        "ns_backend": "external_stream",
        "ns_boundary_input_sha256": "a" * 64,
        "ns_output_sha256": "b" * 64,
        "ns_inference_calls": 1,
        "ns_external_stream_enabled": True,
        "ns_external_pcm_enabled": False,
        "ns_external_applied_frames": 20,
        "ns_external_fallback_frames": 0,
        "ns_external_delay_samples": ns_routes.NS_ROUTES["cofilite"].delay_samples,
        "ns_precomputed_replay": False,
        "ns_stateful": True,
        "ns_stream_chunk_samples": 160,
        "ns_execution_mode": "online_callback",
        "ns_worker_transport": "persistent_f32le_v1",
        "ns_state_resets": 1,
        "ns_realtime_verified": False,
        "ns_stream_callback_frames": 20,
    }


def _install_mock_runner(monkeypatch, tmp_path: Path, *, backend: str = "dpdfnet2",
                         boundary_sample_count: int | None = None,
                         fallback_frames: int = 0,
                         corrupt_boundary: bool = False, final_override: dict | None = None,
                         mutate_cli_on_replay: bool = False):
    spec = ns_routes.NS_ROUTES[backend]
    _write_mic(tmp_path, rate=spec.sample_rate, samples=spec.frame_samples)
    monkeypatch.setattr(ns_routes, "_verify_route_assets", lambda backend: None)
    monkeypatch.setattr(ns_routes, "_model_python", lambda: "/mock/python")
    sample_count = boundary_sample_count or 3200
    boundary_values = np.linspace(-0.2, 0.2, sample_count, dtype=np.float32)

    def fake_run(command, *, timeout, **kwargs):
        assert timeout > 0
        command = list(command)
        if "--backend" in command:
            backend = command[command.index("--backend") + 1]
            source = Path(command[command.index("--input") + 1])
            target = Path(command[command.index("--output") + 1])
            receipt_path = Path(command[command.index("--receipt") + 1])
            values, _ = sf.read(source, dtype="float32")
            aligned = (values * np.float32(0.5)).astype(np.float32)
            _write_float(target, aligned, spec.sample_rate)
            receipt_path.write_text(json.dumps({
                "backend": backend,
                "requested": backend,
                "effective": backend,
                "model_id": spec.model_id,
                "model_path": str(spec.model_path),
                "model_sha256": spec.model_sha256,
                "frontend_path": str(spec.frontend_path),
                "frontend_sha256": spec.frontend_sha256,
                "processing_mode": spec.processing_mode,
                "delay_samples": spec.delay_samples,
                "sample_rate": spec.sample_rate,
                "frame_samples": spec.frame_samples,
                "window_samples": spec.window_samples,
                "hop_samples": spec.hop_samples,
                "offline_whole_utterance": spec.offline_whole_utterance,
                "lookahead_samples": spec.lookahead_samples,
                "input": {
                    "samples": int(values.size),
                    "sample_rate": spec.sample_rate,
                    "channels": 1,
                    "subtype": "FLOAT",
                },
                "aligned_output": {
                    "payload_sha256": _float_payload_sha(aligned),
                    "samples": int(aligned.size),
                    "sample_rate": spec.sample_rate,
                    "channels": 1,
                    "subtype": "FLOAT",
                    "finite": True,
                },
                "inference": {
                    "elapsed_sec": 0.01,
                    "onnx_inferences": 1,
                },
                "details": {
                    "algorithm_latency_samples": spec.delay_samples,
                    "trim_samples": min(spec.delay_samples, int(aligned.size)),
                    "frames_processed": 1,
                },
            }), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "worker ok\n", "")

        prefix = Path(command[command.index("--out") + 1])
        aec_backend = (
            command[command.index("--aec-backend") + 1]
            if "--aec-backend" in command else "webrtc"
        )
        if "--post-ns-pcm" not in command:
            values = boundary_values
            if corrupt_boundary:
                values = boundary_values.copy()
                values[10] = np.nan
            _write_stages(
                Path(str(prefix) + ".stages.json"), prefix, values,
                sample_rate=spec.sample_rate,
            )
            Path(str(prefix) + ".json").write_text(json.dumps(_base_report(
                sample_rate=spec.sample_rate, frame_size=spec.frame_samples,
                input_samples_per_channel=spec.frame_samples,
                aec_backend=aec_backend,
            )), encoding="utf-8")
        else:
            preserved = Path(str(prefix) + ".ns-model/boundary.tx_ns_input.float.wav")
            boundary, _ = sf.read(preserved, dtype="float32")
            replay_path = Path(command[command.index("--post-ns-pcm") + 1])
            replay, _ = sf.read(replay_path, dtype="float32")
            delay = int(command[command.index("--post-ns-delay-samples") + 1])
            _write_stages(Path(str(prefix) + ".stages.json"), prefix, boundary,
                          replay[:len(boundary) + delay],
                          sample_rate=spec.sample_rate)
            report = _base_report(
                sample_rate=spec.sample_rate, frame_size=spec.frame_samples,
                input_samples_per_channel=spec.frame_samples,
                aec_backend=aec_backend,
            )
            report.update({
                "ns_backend": "external_pcm", "ns_external_pcm_enabled": True,
                "ns_external_applied_frames": 20, "ns_external_delay_samples": delay,
                "ns_external_fallback_frames": fallback_frames,
            })
            if final_override:
                report.update(final_override)
            Path(str(prefix) + ".json").write_text(json.dumps(report), encoding="utf-8")
            if mutate_cli_on_replay:
                cli = Path(command[0])
                cli.write_bytes(cli.read_bytes() + b"\n# changed during replay\n")
        return subprocess.CompletedProcess(command, 0, "real cli stdout\n", "")

    monkeypatch.setattr(ns_routes.subprocess, "run", fake_run)
    return boundary_values


def test_native_and_unknown_ns_pass_through_unchanged(monkeypatch):
    calls = []
    sentinel = subprocess.CompletedProcess(["x"], 0, "ok", "")

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return sentinel

    monkeypatch.setattr(ns_routes.subprocess, "run", fake_run)
    command = ("engine", "--mic", "mic.wav", "--out", "out")
    result = ns_routes.execute_ns_command(command, configuration=_configuration("webrtc"),
                                          timeout=7, capture_output=True)
    assert result is sentinel
    assert calls == [(command, {"timeout": 7, "capture_output": True})]
    assert ns_routes.ns_backend_for(_configuration("not_a_pinned_model")) is None


def test_rx_only_command_with_external_token_passes_through(monkeypatch):
    from server import ns_streaming

    calls = []
    sentinel = subprocess.CompletedProcess(["x"], 0, "rx", "")

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return sentinel

    monkeypatch.setattr(ns_routes.subprocess, "run", fake_run)
    monkeypatch.setattr(
        ns_streaming,
        "execute_stream_command",
        lambda *args, **kwargs: pytest.fail("RX-only command must not start a stream worker"),
    )
    command = ["engine", "--profile", "vcp", "--rx-in", "far.wav", "--out", "out"]
    assert ns_routes.execute_ns_command(command, configuration=_configuration(), timeout=9) is sentinel
    assert calls[0][0] is command
    assert calls[0][1] == {"timeout": 9}


def test_route_identity_and_pcm16_lround_are_frozen():
    assert ns_routes.ns_route_identity(None) is None
    assert ns_routes.ns_route_identity(_configuration("webrtc")) is None
    identity = ns_routes.ns_route_identity(_configuration("hybrid_v2_pitch"))
    assert identity["ns_model_sha256"] == "662377b8b2c011046226d2b457beb3af45486071fde6ae8273ee37be1fe0ac63"
    assert identity["ns_delay_samples"] == 160
    values = np.array([-1.1, -0.5 / 32768, 0.5 / 32768, 1.1], dtype=np.float32)
    quantized, pcm = ns_routes._quantize_pcm16_lround(values)
    assert np.array_equal(np.frombuffer(pcm, dtype="<i2"), np.array([-32768, -1, 1, 32767], dtype=np.int16))
    assert np.array_equal(quantized, np.array([-1.0, -1 / 32768, 1 / 32768, 32767 / 32768], dtype=np.float32))


def test_dpdfnet2_48k_hr_route_metadata_locks_the_native_high_rate_contract():
    spec = ns_routes.NS_ROUTES["dpdfnet2_48k_hr"]
    assert spec.backend == spec.model_id == "dpdfnet2_48k_hr"
    assert spec.processing_mode == "causal_streaming_precomputed"
    assert spec.model_path == (
        ns_routes.ROOT
        / "build/ns-dpdfnet2-48k-hr-20261118/assets/dpdfnet2_48khz_hr.onnx"
    )
    assert spec.model_sha256 == (
        "7f0575a5cec0ba4ffd8f8bd657e06d007e4ccdd955d76faab922b9d3291dc14b"
    )
    assert spec.frontend_path == ns_routes.ROOT / "server/dpdfnet2_48k_hr.py"
    assert spec.frontend_sha256 == (
        "481ac722385edfd99a880200a9cf58cbe21a457d38218634a637841f6b2777be"
    )
    assert spec.delay_samples == 1_920
    assert spec.offline_whole_utterance is False
    assert spec.lookahead_samples == 960
    assert (
        spec.sample_rate, spec.frame_samples, spec.window_samples,
        spec.hop_samples, spec.replay_drain_samples,
    ) == (48_000, 480, 960, 480, 48_000)

    identity = ns_routes.ns_route_identity(_configuration("dpdfnet2_48k_hr"))
    assert identity["ns_model_sha256"] == spec.model_sha256
    assert identity["ns_frontend_sha256"] == spec.frontend_sha256
    assert identity["ns_delay_samples"] == 1_920
    assert identity["ns_sample_rate"] == 48_000
    assert identity["ns_frame_samples"] == 480
    assert identity["ns_window_samples"] == 960
    assert identity["ns_hop_samples"] == 480


def test_cofilite_route_metadata_locks_the_stateful_stream_contract():
    spec = ns_routes.NS_ROUTES["cofilite"]
    assert spec.backend == spec.model_id == "cofilite"
    assert spec.processing_mode == "stateful_streaming"
    assert spec.model_path == (
        ns_routes.ROOT
        / "build/ns-cofilite-20261115/upstream/checkpoint/model_on_dns3.tar"
    )
    assert spec.model_sha256 == (
        "40d1bb69dcd553ce93c745c8bd136e1593397f529aa53609827c69df95ce2a30"
    )
    assert spec.delay_samples == cofilite_stream.DELAY_SAMPLES == 480
    assert spec.offline_whole_utterance is False
    assert spec.lookahead_samples == cofilite_stream.LOOKAHEAD_SAMPLES
    assert spec.frontend_path == ns_routes.ROOT / "server/cofilite_onnx.py"
    assert spec.frontend_sha256 == (
        "21d3072fbb0879912911c391534ae0e6d5d2276c6e440601c936f23a9733e682"
    )
    assert spec.frontend_sha256 == hashlib.sha256(
        spec.frontend_path.read_bytes()
    ).hexdigest()
    assert spec.frame_executor == "onnxruntime_cpu"
    assert spec.frame_model_path == (
        ns_routes.ROOT
        / "build/cofilite-ort-20261117/cofilite-frame-v1-opset18.onnx"
    )
    assert spec.frame_model_sha256 == (
        "f754898f6721c3180f53b602183db1275c09e08355b8ddff2b0b289181a7a2e0"
    )
    assert spec.frame_model_sha256 == hashlib.sha256(
        spec.frame_model_path.read_bytes()
    ).hexdigest()
    assert spec.extra_assets == (
        (
            ns_routes.ROOT / "build/ns-cofilite-20261115/upstream/model/cofilite.py",
            "e77db8c40fbeef33b2496aca3f258d47205a1453c1bfb44c6b644f543c0afb9c",
        ),
        (
            ns_routes.ROOT / "build/ns-cofilite-20261115/cofi_infer.py",
            "a885236e79a2df861d396f8d3cf0dafbf2e5d448513c96c3db30a68e3e0c274e",
        ),
        (
            ns_routes.ROOT / "server/cofilite_stream.py",
            "3b7e08a21460d1f9195db180c2868e075045d642564806f1af3fee1fe39d4ae1",
        ),
        (
            ns_routes.ROOT / "server/cofilite_frame.py",
            "aad70af318a73d1010b0e8f92708b5137e167645a6e0e731fdda0b3f2b236e5a",
        ),
        (
            ns_routes.ROOT
            / "build/cofilite-ort-20261117/cofilite-frame-v1-opset18.identity.json",
            "cb2dce2870eaa6035c5409c3743baf4eef950e9fb2ccc251171b95578792c29a",
        ),
        (
            ns_routes.ROOT
            / "build/cofilite-ort-20261117/cofilite-frame-v1-opset18.checker.json",
            "3c6acb110e95a0bd78073dbdb1dc97033476aa804413562bbbe307cfd99ea66b",
        ),
    )
    identity = ns_routes.ns_route_identity(_configuration("cofilite"))
    assert identity["ns_frame_executor"] == "onnxruntime_cpu"
    assert identity["ns_frame_model_sha256"] == spec.frame_model_sha256
    assert identity["ns_frame_execution_provider"] == "CPUExecutionProvider"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda spec: replace(spec, frame_executor="jit_freeze"), "执行身份缺失或非法"),
        (lambda spec: replace(spec, frame_executor="onnxruntime_cuda"), "执行身份缺失或非法"),
        (
            lambda spec: replace(spec, frame_model_path=Path("/missing/cofilite.onnx")),
            "模型文件不存在",
        ),
    ],
)
def test_cofilite_ort_asset_identity_fails_closed(monkeypatch, mutation, message):
    spec = mutation(ns_routes.NS_ROUTES["cofilite"])
    monkeypatch.setitem(ns_routes.NS_ROUTES, "cofilite", spec)
    with pytest.raises(ValueError, match=message):
        ns_routes._verify_route_assets("cofilite")


def test_cofilite_tampered_ort_graph_fails_closed(monkeypatch, tmp_path):
    graph = tmp_path / "cofilite-frame-v1-opset18.onnx"
    graph.write_bytes(b"not the frozen ONNX graph\n")
    spec = replace(
        ns_routes.NS_ROUTES["cofilite"],
        frame_model_path=graph,
    )
    monkeypatch.setitem(ns_routes.NS_ROUTES, "cofilite", spec)
    with pytest.raises(ValueError, match="SHA-256 不匹配"):
        ns_routes._verify_route_assets("cofilite")


def test_cofilite_dispatches_once_to_streaming_runner(monkeypatch, tmp_path):
    from server import ns_streaming

    command = _command(tmp_path)
    sentinel = subprocess.CompletedProcess(command, 0, "stream ok\n", "")
    calls = []

    def fake_stream(actual_command, **kwargs):
        calls.append((list(actual_command), kwargs))
        return sentinel

    monkeypatch.setattr(ns_streaming, "execute_stream_command", fake_stream)
    monkeypatch.setattr(
        ns_routes.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("CoFi-Lite must not use offline subprocess.run dispatch"),
    )

    result = ns_routes.execute_ns_command(
        command,
        configuration=_configuration("cofilite"),
        timeout=7,
        capture_output=True,
        text=True,
    )

    assert result is sentinel
    assert calls == [(
        command,
        {"configuration": _configuration("cofilite"), "timeout": 7,
         "capture_output": True, "text": True},
    )]


@pytest.mark.parametrize(
    "backend,delay",
    [("dpdfnet2", 640), ("hybrid_v2_pitch", 160)],
)
def test_offline_external_execute_merges_real_identity_and_validates_replay(
    monkeypatch, tmp_path, backend, delay,
):
    _install_mock_runner(monkeypatch, tmp_path, backend=backend)
    result = ns_routes.execute_ns_command(
        _command(tmp_path), configuration=_configuration(backend), timeout=20,
        capture_output=True, text=True,
    )
    prefix = tmp_path / "out"
    report = json.loads(Path(str(prefix) + ".json").read_text())
    stages = json.loads(Path(str(prefix) + ".stages.json").read_text())
    assert result.returncode == 0
    assert "NS model:" in result.stdout and result.stdout.startswith("real cli stdout")
    assert report["ns_backend"] == backend
    assert report["ns_backend_requested"] == backend
    assert report["ns_backend_effective"] == backend
    assert report["ns_delay_samples"] == delay
    assert report["ns_external_applied_frames"] == 20
    assert report["ns_external_fallback_frames"] == 0
    assert report["aec_backend"] == "localvqe_aec200k_cpu"
    assert stages["ns_model"]["ns_backend_effective"] == backend
    assert Path(str(prefix) + ".ns.json").is_file()
    assert Path(str(prefix) + ".ns-model/receipt.json").is_file()
    assert Path(str(prefix) + ".ns-model/replay.float.wav").is_file()
    tx_ns_stage = next(
        item for item in stages["pipeline_stages"] if item.get("id") == "tx_ns"
    )
    assert tx_ns_stage["channel_policy"] == "mono external model NS output"
    assert tx_ns_stage["measurement_domain"] == (
        "selected model output; before EQ, AGC, DRC and limiter"
    )
    assert "startup" not in tx_ns_stage["measurement_domain"].lower()
    tx_ns_values, _ = sf.read(tx_ns_stage["file"], dtype="float32")
    boundary_values = np.linspace(-0.2, 0.2, 3200, dtype=np.float32)
    quantized_boundary, _ = ns_routes._quantize_pcm16_lround(boundary_values)
    expected_tx_ns = np.concatenate((
        np.zeros(delay, dtype=np.float32),
        quantized_boundary * np.float32(0.5),
    ))
    assert np.array_equal(tx_ns_values, expected_tx_ns)
    assert ns_routes.validate_ns_route_report(report, _configuration(backend)) == []


def test_dpdfnet2_48k_hr_mock_capture_replay_and_identity(monkeypatch, tmp_path):
    _install_mock_runner(
        monkeypatch, tmp_path, backend="dpdfnet2_48k_hr",
        boundary_sample_count=9_601,
    )
    result = ns_routes.execute_ns_command(
        _command(tmp_path, sample_rate=48_000),
        configuration=_configuration("dpdfnet2_48k_hr"),
        timeout=20, capture_output=True, text=True,
    )
    prefix = tmp_path / "out"
    report = json.loads(Path(str(prefix) + ".json").read_text())
    receipt = json.loads(
        Path(str(prefix) + ".ns-model/receipt.json").read_text()
    )
    summary = json.loads(Path(str(prefix) + ".ns.json").read_text())
    stages = json.loads(Path(str(prefix) + ".stages.json").read_text())

    assert result.returncode == 0
    assert result.stdout.startswith("real cli stdout")
    assert "backend=dpdfnet2_48k_hr" in result.stdout
    assert "delay=1920" in result.stdout
    assert report["ns_backend"] == report["ns_backend_requested"] == (
        report["ns_backend_effective"]
    ) == "dpdfnet2_48k_hr"
    assert report["ns_model_id"] == "dpdfnet2_48k_hr"
    assert report["ns_model_sha256"] == (
        "7f0575a5cec0ba4ffd8f8bd657e06d007e4ccdd955d76faab922b9d3291dc14b"
    )
    assert report["ns_frontend_sha256"] == (
        "481ac722385edfd99a880200a9cf58cbe21a457d38218634a637841f6b2777be"
    )
    assert report["ns_sample_rate"] == receipt["sample_rate"] == 48_000
    assert report["ns_frame_samples"] == receipt["frame_samples"] == 480
    assert report["ns_window_samples"] == receipt["window_samples"] == 960
    assert report["ns_hop_samples"] == receipt["hop_samples"] == 480
    assert report["ns_delay_samples"] == receipt["delay_samples"] == 1_920
    assert report["aec_backend"] == "webrtc_aec3"
    assert report["frame_size"] == 480
    assert summary["worker_receipt"] == receipt
    assert summary["identity"]["ns_sample_rate"] == 48_000
    assert summary["execution"]["ns_replay_file_samples"] > 0
    assert stages["ns_model"]["ns_backend_effective"] == "dpdfnet2_48k_hr"
    assert Path(str(prefix) + ".ns-model/replay.float.wav").is_file()
    assert ns_routes.validate_ns_route_report(
        report, _configuration("dpdfnet2_48k_hr"),
    ) == []


def test_dpdfnet2_48k_hr_boundary_and_final_stages_are_native_48k(
    monkeypatch, tmp_path,
):
    _install_mock_runner(
        monkeypatch, tmp_path, backend="dpdfnet2_48k_hr",
        boundary_sample_count=9_601,
    )
    ns_routes.execute_ns_command(
        _command(tmp_path, sample_rate=48_000),
        configuration=_configuration("dpdfnet2_48k_hr"),
        timeout=20, capture_output=True, text=True,
    )
    prefix = tmp_path / "out"
    for stages_path in (
        Path(str(prefix) + ".ns-model/boundary.stages.json"),
        Path(str(prefix) + ".stages.json"),
    ):
        payload = json.loads(stages_path.read_text())
        stages = {
            item["id"]: item for item in payload["pipeline_stages"]
            if item.get("direction") == "tx"
        }
        assert stages["tx_ns_input"]["sample_rate"] == 48_000
        assert sf.info(stages["tx_ns_input"]["file"]).samplerate == 48_000
        if "tx_ns" in stages:
            assert stages["tx_ns"]["sample_rate"] == 48_000
            assert sf.info(stages["tx_ns"]["file"]).samplerate == 48_000


def test_dpdfnet2_48k_hr_delay_and_replay_are_frame_aligned(monkeypatch, tmp_path):
    boundary_samples = 9_601
    _install_mock_runner(
        monkeypatch, tmp_path, backend="dpdfnet2_48k_hr",
        boundary_sample_count=boundary_samples,
    )
    ns_routes.execute_ns_command(
        _command(tmp_path, sample_rate=48_000),
        configuration=_configuration("dpdfnet2_48k_hr"),
        timeout=20, capture_output=True, text=True,
    )
    prefix = tmp_path / "out"
    spec = ns_routes.NS_ROUTES["dpdfnet2_48k_hr"]
    report = json.loads(Path(str(prefix) + ".json").read_text())
    summary = json.loads(Path(str(prefix) + ".ns.json").read_text())
    replay, replay_rate = sf.read(
        Path(str(prefix) + ".ns-model/replay.float.wav"), dtype="float32",
    )

    expected_core = boundary_samples + spec.delay_samples
    expected_padded = int(np.ceil(expected_core / spec.frame_samples) * spec.frame_samples)
    expected_replay = expected_padded + spec.replay_drain_samples
    assert spec.delay_samples == 1_920
    assert report["ns_delay_samples"] == 1_920
    assert summary["execution"]["ns_replay_core_samples"] == expected_core == 11_521
    assert summary["execution"]["ns_replay_file_samples"] == expected_replay == 60_000
    assert replay_rate == spec.sample_rate == 48_000
    assert replay.size == expected_replay
    assert replay.size % spec.frame_samples == 0
    assert not np.any(replay[:spec.delay_samples])
    assert not np.any(replay[expected_padded:])

    stages = json.loads(Path(str(prefix) + ".stages.json").read_text())
    tx_ns_stage = next(
        item for item in stages["pipeline_stages"] if item.get("id") == "tx_ns"
    )
    tx_ns_values, tx_ns_rate = sf.read(tx_ns_stage["file"], dtype="float32")
    assert tx_ns_rate == 48_000
    assert tx_ns_values.size == expected_core
    assert not np.any(tx_ns_values[:spec.delay_samples])


def test_missing_model_asset_fails_before_cli(monkeypatch, tmp_path):
    _write_mic(tmp_path)
    spec = ns_routes.NS_ROUTES["cofilite"]
    monkeypatch.setitem(ns_routes.NS_ROUTES, "cofilite", replace(spec, model_path=tmp_path / "missing.bin"))
    calls = []
    monkeypatch.setattr(ns_routes.subprocess, "run", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match="模型文件不存在"):
        ns_routes.execute_ns_command(_command(tmp_path), configuration=_configuration(), timeout=5)
    assert calls == []


def test_corrupt_boundary_input_is_rejected(monkeypatch, tmp_path):
    _install_mock_runner(monkeypatch, tmp_path, corrupt_boundary=True)
    with pytest.raises(ValueError, match="NaN/Inf"):
        ns_routes.execute_ns_command(
            _command(tmp_path), configuration=_configuration("dpdfnet2"), timeout=5,
            capture_output=True, text=True,
        )


def test_forbidden_fallback_frames_raise(monkeypatch, tmp_path):
    _install_mock_runner(monkeypatch, tmp_path, fallback_frames=1)
    with pytest.raises(ValueError, match="fallback"):
        ns_routes.execute_ns_command(
            _command(tmp_path), configuration=_configuration("dpdfnet2"), timeout=5,
            capture_output=True, text=True,
        )


def test_cli_mutation_during_capture_replay_fails_closed(monkeypatch, tmp_path):
    _install_mock_runner(monkeypatch, tmp_path, mutate_cli_on_replay=True)
    with pytest.raises(ValueError, match="CLI 文件身份发生变化"):
        ns_routes.execute_ns_command(
            _command(tmp_path), configuration=_configuration("dpdfnet2"), timeout=5,
            capture_output=True, text=True,
        )


def test_timeout_is_forwarded_to_boundary(monkeypatch, tmp_path):
    _write_mic(tmp_path)
    monkeypatch.setattr(ns_routes, "_verify_route_assets", lambda backend: None)

    def timeout_run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(ns_routes.subprocess, "run", timeout_run)
    with pytest.raises(subprocess.TimeoutExpired):
        ns_routes.execute_ns_command(
            _command(tmp_path), configuration=_configuration("dpdfnet2"), timeout=2.5,
        )


def test_cofilite_stream_report_accepts_the_complete_live_contract():
    assert ns_routes.validate_ns_route_report(
        _cofilite_report(), _configuration("cofilite"),
    ) == []


def test_cofilite_rejects_legacy_replay_disguised_as_a_stream_report():
    report = _cofilite_report()
    report.update({
        "ns_backend": "external_pcm",
        "ns_external_stream_enabled": False,
        "ns_external_pcm_enabled": True,
        "ns_external_delay_samples": 0,
        "ns_precomputed_replay": True,
        "ns_stateful": False,
        "ns_execution_mode": "offline_replay",
        "ns_worker_transport": "captured_wav",
    })

    errors = ns_routes.validate_ns_route_report(report, _configuration("cofilite"))
    joined = "\n".join(errors)
    assert "ns_backend" in joined
    assert "ns_external_stream_enabled" in joined
    assert "ns_external_pcm_enabled" in joined
    assert "ns_external_delay_samples" in joined
    assert "ns_precomputed_replay" in joined
    assert "ns_stateful" in joined
    assert "ns_execution_mode" in joined
    assert "ns_worker_transport" in joined


@pytest.mark.parametrize(
    "field,value",
    [
        ("ns_external_stream_enabled", False),
        ("ns_external_pcm_enabled", True),
        ("ns_processing_mode", "causal_streaming_precomputed"),
        ("ns_offline_whole_utterance", True),
        ("ns_lookahead_samples", 0),
        ("ns_precomputed_replay", True),
        ("ns_stateful", False),
        ("ns_stream_chunk_samples", 159),
        ("ns_execution_mode", "offline_replay"),
        ("ns_worker_transport", "captured_f32le"),
        ("ns_state_resets", 0),
        ("ns_realtime_verified", True),
        ("ns_stream_callback_frames", 19),
        ("ns_external_fallback_frames", 1),
        ("ns_boundary_input_sha256", "A" * 64),
        ("ns_output_sha256", "b" * 63),
        ("ns_inference_calls", 0),
        ("ns_model_sha256", "0" * 64),
        ("ns_frontend_sha256", "0" * 64),
        ("ns_frame_executor", "jit_freeze"),
        ("ns_frame_model_sha256", "0" * 64),
        ("ns_frame_execution_provider", "CUDAExecutionProvider"),
    ],
)
def test_cofilite_report_validation_rejects_tampered_metadata(field, value):
    report = _cofilite_report()
    report[field] = value
    errors = ns_routes.validate_ns_route_report(report, _configuration("cofilite"))
    assert any(field in error for error in errors), (field, errors)


def test_cofilite_rejects_zero_applied_even_when_callback_counts_agree():
    report = _cofilite_report()
    report["ns_external_applied_frames"] = 0
    report["ns_stream_callback_frames"] = 0
    errors = ns_routes.validate_ns_route_report(report, _configuration("cofilite"))
    assert any("ns_external_applied_frames" in error for error in errors)


@pytest.mark.parametrize(
    "injected",
    [
        ["--post-ns-pcm", "x"],
        [
            "--post-ns-delay-samples",
            str(ns_routes.NS_ROUTES["cofilite"].delay_samples),
        ],
        ["--ns-stream-input-fd", "7"],
        ["--ns-stream-output-fd", "8"],
        [
            "--ns-stream-delay-samples",
            str(ns_routes.NS_ROUTES["cofilite"].delay_samples),
        ],
    ],
)
def test_external_command_rejects_preinjected_replay_and_stream_fds(tmp_path, injected):
    command = _command(tmp_path)
    with pytest.raises(ValueError, match="外部 NS 命令不能预置"):
        ns_routes._external_command(command + injected)


def test_external_command_rejects_wrong_profile(tmp_path):
    command = _command(tmp_path)
    wrong = ["--profile" if value == "vcp" else value for value in command]
    with pytest.raises(ValueError, match="VCP profile"):
        ns_routes._external_command(wrong)


def test_aec_drain_counters_may_grow_but_identity_must_not_change():
    boundary = {
        "sample_rate": 16_000, "microphones": 1, "reference_channels": 1,
        "input_samples_per_channel": 3200, "aec_backend_effective": "localvqe_aec200k_cpu",
        "aec_model_sha256": "1" * 64, "aec_localvqe_inferences": 20,
        "aec_neural_delay_samples": 256,
        "aec_hop_adapter_delay_samples_16k": 224,
        "aec_algorithm_delay_samples_16k": 480,
        "aec_model_codec_delay_samples_16k": 256,
        "aec_linear_frontend_delay_samples_16k": 0,
    }
    final = {**boundary, "aec_localvqe_inferences": 24, "aec_route_processed_frames": 24}
    ns_routes._validate_boundary_identity(boundary, final)
    final["aec_algorithm_delay_samples_16k"] = 512
    with pytest.raises(ValueError, match="aec_algorithm_delay_samples_16k"):
        ns_routes._validate_boundary_identity(boundary, final)
    final["aec_algorithm_delay_samples_16k"] = 480
    final["aec_model_sha256"] = "2" * 64
    with pytest.raises(ValueError, match="aec_model_sha256"):
        ns_routes._validate_boundary_identity(boundary, final)


@pytest.mark.parametrize(
    ("aec_backend", "no_aec", "include_aec"),
    [
        ("localvqe_aec200k", False, True),
        (None, True, True),
        (None, False, False),
    ],
)
def test_dpdfnet2_48k_hr_rejects_non_aec3_before_capture(
    monkeypatch, tmp_path, aec_backend, no_aec, include_aec,
):
    _write_mic(tmp_path, rate=48_000, samples=480)
    calls = []
    monkeypatch.setattr(ns_routes, "_verify_route_assets", lambda backend: None)
    monkeypatch.setattr(
        ns_routes.subprocess, "run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    command = _command(
        tmp_path, sample_rate=48_000, aec_backend=aec_backend,
        no_aec=no_aec, include_aec=include_aec,
    )
    with pytest.raises(
        ValueError,
        match="必须显式使用 --aec-backend webrtc",
    ):
        ns_routes.execute_ns_command(
            command, configuration=_configuration("dpdfnet2_48k_hr"), timeout=5,
        )
    assert calls == []


def test_non_16k_mic_header_is_rejected_before_capture(monkeypatch, tmp_path):
    mic = tmp_path / "mic.wav"
    sf.write(mic, np.zeros(800, dtype=np.float32), 8_000, subtype="PCM_16")
    calls = []
    monkeypatch.setattr(ns_routes, "_verify_route_assets", lambda backend: None)
    monkeypatch.setattr(ns_routes.subprocess, "run", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match=r"原生 16000 Hz.*实际 8000 Hz"):
        ns_routes.execute_ns_command(_command(tmp_path), configuration=_configuration(), timeout=5)
    assert calls == []


def test_cofilite_model_worker_uses_stateful_enhance_stream(monkeypatch, tmp_path):
    input_path = tmp_path / "input.q.float.wav"
    _write_float(input_path, np.array([0, 1, -1, 100, -100], dtype=np.float32) / 32768)
    observed = {}
    info = {
        "stateful_streaming": True,
        "whole_utterance_replay": False,
        "inferences": 1,
        "frames_processed": 1,
    }

    class Adapter:
        @staticmethod
        def enhance_stream(waveform, chunk_size):
            observed["shape"] = waveform.shape
            observed["chunk_size"] = chunk_size
            time.sleep(0.001)
            return waveform * np.float32(0.5), info

    monkeypatch.setattr(ns_model_worker, "_load_adapter", lambda path, name: Adapter)
    aligned, details, waveform = ns_model_worker._run_cofilite(
        ns_routes.NS_ROUTES["cofilite"], input_path,
    )
    assert aligned.shape == waveform.shape == (5,)
    assert observed == {"shape": (5,), "chunk_size": 160}
    assert details["offline_whole_utterance"] is False
    assert details["lookahead_samples_observed"] == 511
    assert details["lookahead_treated_as_zero"] is False
    assert details["adapter"] == "CoFiLiteStream/process(160)/flush"
    assert details["streaming_details"] is info
    assert details["elapsed_sec"] > 0


def test_dpdfnet2_48k_hr_worker_receipt_binds_native_geometry(
    monkeypatch, tmp_path,
):
    input_path = tmp_path / "input.q.float.wav"
    values = np.arange(961, dtype=np.float32) / np.float32(32768.0)
    _write_float(input_path, values, 48_000)
    observed = {}
    runtime = object()
    info = SimpleNamespace(
        latency_samples=1_920,
        trim_samples=1_920,
        frames_processed=1,
        inference_sec=0.002,
    )

    class Adapter:
        @staticmethod
        def load_runtime(model_path):
            observed["model_path"] = model_path
            return runtime

        @staticmethod
        def enhance_stream(waveform, loaded_runtime, *, chunk_size):
            observed["shape"] = waveform.shape
            observed["runtime"] = loaded_runtime
            observed["chunk_size"] = chunk_size
            return waveform * np.float32(0.5), info

    monkeypatch.setattr(ns_model_worker, "_load_adapter", lambda path, name: Adapter)
    monkeypatch.setattr(
        ns_model_worker, "_verify_assets",
        lambda spec: {"mock_asset": "0" * 64},
    )
    output_path = tmp_path / "aligned.float.wav"
    receipt_path = tmp_path / "receipt.json"
    receipt = ns_model_worker.run(
        "dpdfnet2_48k_hr", input_path, output_path, receipt_path,
    )

    assert observed == {
        "model_path": ns_routes.NS_ROUTES["dpdfnet2_48k_hr"].model_path,
        "shape": (961,),
        "runtime": runtime,
        "chunk_size": 480,
    }
    assert receipt["backend"] == receipt["requested"] == (
        receipt["effective"]
    ) == "dpdfnet2_48k_hr"
    assert receipt["model_sha256"] == (
        "7f0575a5cec0ba4ffd8f8bd657e06d007e4ccdd955d76faab922b9d3291dc14b"
    )
    assert receipt["frontend_sha256"] == (
        "481ac722385edfd99a880200a9cf58cbe21a457d38218634a637841f6b2777be"
    )
    assert receipt["sample_rate"] == receipt["input"]["sample_rate"] == 48_000
    assert receipt["frame_samples"] == 480
    assert receipt["window_samples"] == 960
    assert receipt["hop_samples"] == 480
    assert receipt["delay_samples"] == 1_920
    assert receipt["details"]["chunk_samples"] == 480
    assert receipt["details"]["window_samples"] == 960
    assert receipt["details"]["hop_samples"] == 480
    assert receipt["details"]["algorithm_latency_samples"] == 1_920
    assert receipt["aligned_output"]["samples"] == 961
    assert json.loads(receipt_path.read_text()) == receipt
