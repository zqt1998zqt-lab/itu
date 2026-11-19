"""Fail-closed lifecycle contracts for the CoFi-Lite ORT NS route."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
import soundfile as sf

from server import ns_routes, ns_stream_worker, ns_streaming


GRAPH_PATH = (
    ns_routes.ROOT
    / "build/cofilite-ort-20261117/cofilite-frame-v1-opset18.onnx"
)
GRAPH_SHA256 = (
    "f754898f6721c3180f53b602183db1275c09e08355b8ddff2b0b289181a7a2e0"
)
FRONTEND_SHA256 = (
    "21d3072fbb0879912911c391534ae0e6d5d2276c6e440601c936f23a9733e682"
)
WEIGHTS_SHA256 = (
    "40d1bb69dcd553ce93c745c8bd136e1593397f529aa53609827c69df95ce2a30"
)


def _configuration() -> dict:
    return {
        "profile": "vcp",
        "selection": {
            "bf": "off",
            "aec": "localvqe_aec200k",
            "ns": "cofilite",
            "eq": "three_band",
            "agc": "adaptive",
            "drc": "speech",
            "limiter": "off",
        },
    }


def _frame_identity() -> dict:
    return {
        "mode": "onnxruntime_cpu",
        "provider": "CPUExecutionProvider",
        "model_sha256": GRAPH_SHA256,
        "weights_sha256": WEIGHTS_SHA256,
        "intra_op_num_threads": 1,
        "inter_op_num_threads": 1,
        "state_count": 18,
        "session_load_count": 1,
    }


def _ready_payload() -> dict:
    spec = ns_routes.NS_ROUTES["cofilite"]
    return {
        "protocol": "itu-ns-f32le-v1",
        "backend": "cofilite",
        "sample_rate": 16_000,
        "frame_samples": 160,
        "delay_samples": spec.delay_samples,
        "model_sha256": spec.model_sha256,
        "frontend_sha256": spec.frontend_sha256,
        "processing_mode": spec.processing_mode,
        "frame_executor": spec.frame_executor,
        "frame_model_sha256": spec.frame_model_sha256,
        "frame_execution_provider": "CPUExecutionProvider",
    }


def _ready_line() -> bytes:
    return (json.dumps(_ready_payload(), separators=(",", ":")) + "\n").encode("ascii")


def _stream_report(callback_frames: int = 20) -> dict:
    spec = ns_routes.NS_ROUTES["cofilite"]
    return {
        "ns_backend": "external_stream",
        "ns_external_stream_enabled": True,
        "ns_external_pcm_enabled": False,
        "ns_external_applied_frames": callback_frames,
        "ns_external_fallback_frames": 0,
        "ns_external_delay_samples": spec.delay_samples,
    }


def _receipt(callback_frames: int = 20, frame_identity: dict | None = None) -> dict:
    samples = callback_frames * 160
    expected_inferences = 0 if samples < 257 else samples // 256
    timing = {
        field: {
            "count": callback_frames,
            "total_seconds": 0.02,
            "p50_upper_ms": 1.0,
            "p95_upper_ms": 1.2,
            "p99_upper_ms": 1.4,
            "max_ms": 1.5,
            "over_10ms": 0,
        }
        for field in (
            "read_wait",
            "quantize_hash_preprocess",
            "model_process",
            "postprocess_write",
            "callback",
        )
    }
    sample_count = min(callback_frames, 16)
    return {
        "input_payload_sha256": "1" * 64,
        "output_payload_sha256": "2" * 64,
        "quantized_input_pcm16_sha256": "3" * 64,
        "model_inferences": expected_inferences,
        "expected_model_inferences": expected_inferences,
        "model_hop_samples": 256,
        "model_ready_threshold_samples": 257,
        "inference_seconds": 0.02,
        "callback_frames": callback_frames,
        "samples": samples,
        "stateful": True,
        "precomputed_replay": False,
        "state_bytes": 4096,
        "callback_max_ms": 1.5,
        "callback_p50_upper_ms": 1.0,
        "callback_p95_upper_ms": 1.2,
        "callback_p99_upper_ms": 1.4,
        "callback_over_10ms": 0,
        "rtf_model": 0.1,
        "peak_rss_bytes": 64 * 1024 * 1024,
        "model_initializations": 1,
        "live_stream_initializations": 1,
        "live_state_reset_calls": 1,
        "warmup_calls": 8,
        "warmup_inferences": 5,
        "startup_load_seconds": 0.2,
        "warmup_seconds": 0.1,
        "startup_to_ready_seconds": 0.4,
        "startup_timing_scope": "module import through ready construction",
        "timing_clock": "time.perf_counter monotonic seconds",
        "timing_scope": "worker service time",
        "timing_sample_policy": "trailing bounded window",
        "timing_sample_capacity": 16,
        "timing_samples": [
            {
                "frame_index": callback_frames - sample_count + index + 1,
                "read_wait_ms": 0.1,
                "preprocess_ms": 0.2,
                "model_process_ms": 0.8,
                "postprocess_write_ms": 0.1,
                "callback_ms": 1.1,
            }
            for index in range(sample_count)
        ],
        "timing": timing,
        "worker_source_sha256": "4" * 64,
        "worker_command": ["/fake/python", "-m", "server.ns_stream_worker"],
        "worker_command_sha256": "5" * 64,
        "python_executable": "/fake/python",
        "frame_executor": dict(frame_identity or _frame_identity()),
    }


def _command(tmp_path: Path) -> tuple[list[str], Path]:
    cli = tmp_path / "itu_audio_cli"
    cli.write_bytes(b"#!/bin/sh\nexit 0\n")
    cli.chmod(0o755)
    mic = tmp_path / "mic.wav"
    sf.write(mic, np.zeros(320, dtype=np.float32), 16_000, subtype="PCM_16")
    prefix = tmp_path / "out"
    return [
        str(cli),
        "--profile",
        "vcp",
        "--aec-backend",
        "localvqe_aec200k",
        "--ns-backend",
        "webrtc",
        "--mic",
        str(mic),
        "--out",
        str(prefix),
    ], prefix


class _FakeWorker:
    def __init__(
        self,
        command: list[str],
        *,
        ready: bytes,
        stderr_text: bytes,
        receipt: dict | None,
        stderr,
    ):
        self.command = command
        self.receipt = receipt
        self.returncode: int | None = None

        stdin_read, stdin_write = os.pipe()
        os.close(stdin_read)
        stdout_read, stdout_write = os.pipe()
        self.stdin = os.fdopen(stdin_write, "wb", buffering=0)
        self.stdout = os.fdopen(stdout_read, "rb", buffering=0)
        self._stdout_write = os.fdopen(stdout_write, "wb", buffering=0)
        self.stdin_fd = self.stdin.fileno()
        self.stdout_fd = self.stdout.fileno()

        if ready:
            self._stdout_write.write(ready)
        if not ready:
            self._stdout_write.close()
        if stderr_text:
            stderr.write(stderr_text)
            stderr.flush()

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if not self._stdout_write.closed:
            self._stdout_write.close()
        self.returncode = 0
        if self.receipt is not None:
            receipt_path = Path(self.command[self.command.index("--receipt") + 1])
            receipt_path.write_text(
                json.dumps(self.receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return self.returncode

    def terminate(self):
        if not self._stdout_write.closed:
            self._stdout_write.close()
        self.returncode = -15

    def kill(self):
        if not self._stdout_write.closed:
            self._stdout_write.close()
        self.returncode = -9


def _install_fakes(
    monkeypatch,
    tmp_path: Path,
    *,
    ready: bytes,
    receipt: dict | None,
    stderr_text: bytes = b"",
):
    command, prefix = _command(tmp_path)
    workers: list[_FakeWorker] = []
    runs: list[tuple[list[str], dict]] = []

    monkeypatch.setattr(ns_routes, "_verify_route_assets", lambda backend: None)
    monkeypatch.setattr(ns_routes, "_model_python", lambda: "/fake/python")

    def fake_popen(worker_command, **kwargs):
        receipt_data = receipt
        if receipt is not None:
            receipt_data = dict(receipt)
            receipt_data["frame_executor"] = dict(
                receipt.get("frame_executor") or {}
            )
            if receipt_data.get("worker_source_sha256") == "4" * 64:
                receipt_data["worker_source_sha256"] = hashlib.sha256(
                    Path(ns_stream_worker.__file__).read_bytes()
                ).hexdigest()
            if receipt_data.get("worker_command_sha256") == "5" * 64:
                receipt_data["worker_command"] = list(worker_command)
                receipt_data["worker_command_sha256"] = (
                    ns_streaming._command_sha256(list(worker_command))
                )
        worker = _FakeWorker(
            list(worker_command),
            ready=ready,
            stderr_text=stderr_text,
            receipt=receipt_data,
            stderr=kwargs["stderr"],
        )
        workers.append(worker)
        return worker

    def fake_run(online_command, **kwargs):
        online = list(online_command)
        runs.append((online, kwargs))
        output_prefix = Path(online[online.index("--out") + 1])
        Path(str(output_prefix) + ".json").write_text(
            json.dumps(_stream_report(), indent=2) + "\n",
            encoding="utf-8",
        )
        Path(str(output_prefix) + ".stages.json").write_text(
            json.dumps({"pipeline_stages": []}) + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(online, 0, "native stream ok\n", "")

    monkeypatch.setattr(ns_streaming.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(ns_streaming.subprocess, "run", fake_run)
    return command, prefix, workers, runs


class _Model:
    pass


class _Stream:
    frame_samples = 160
    hop_samples = 256
    fft_size = 512

    def __init__(
        self,
        model: _Model,
        *,
        delay_samples: int,
        executor_info: dict,
        registry: list,
    ):
        self.model = model
        self.delay_samples = delay_samples
        self._executor_info = executor_info
        self.inferences = 0
        self.state_bytes = 4096
        self.total_samples = 0
        self.reset_calls = 1
        self.registry = registry
        registry.append(self)

    def _advance(self) -> None:
        self.total_samples += 160
        self.inferences = (
            0
            if self.total_samples < self.fft_size // 2 + 1
            else self.total_samples // self.hop_samples
        )

    def process(self, frame: np.ndarray) -> np.ndarray:
        assert frame.shape == (160,)
        assert frame.dtype == np.float32
        self._advance()
        return np.zeros(160, dtype=np.float32)

    def _consume_validated(self, frame: np.ndarray) -> np.ndarray:
        assert frame.shape == (160,)
        assert frame.dtype == np.float32
        assert frame.flags.c_contiguous
        self._advance()
        return np.ascontiguousarray(frame * np.float32(0.5), dtype=np.float32)


def _invoke_worker(
    tmp_path: Path,
    *,
    frame_identity: dict | None = None,
) -> tuple[dict | None, dict | None, bytes, Exception | None]:
    spec = ns_routes.NS_ROUTES["cofilite"]
    input_read, input_write = os.pipe()
    output_read, output_write = os.pipe()
    os.write(
        input_write,
        np.zeros((20, 160), dtype="<f4").tobytes(),
    )
    os.close(input_write)

    model = _Model()
    streams: list[_Stream] = []
    executor_info = dict(frame_identity or _frame_identity())
    verify_calls = 0

    def verify_assets():
        nonlocal verify_calls
        verify_calls += 1

    def load_backend():
        verify_assets()
        return (
            model,
            lambda: _Stream(
                model,
                delay_samples=spec.delay_samples,
                executor_info=executor_info,
                registry=streams,
            ),
            nullcontext,
        )

    receipt = None
    error = None
    receipt_path = tmp_path / "worker-receipt.json"
    try:
        receipt = ns_stream_worker.run(
            receipt_path,
            input_fd=input_read,
            output_fd=output_write,
            route_spec=spec,
            load_backend=load_backend,
            verify_assets=verify_assets,
            command_argv=["/fake/python", "-m", "server.ns_stream_worker"],
        )
    except Exception as caught:
        error = caught
    finally:
        os.close(input_read)
        os.close(output_write)

    chunks = []
    while True:
        block = os.read(output_read, 4096)
        if not block:
            break
        chunks.append(block)
    os.close(output_read)
    output = b"".join(chunks)
    ready = None
    if output:
        line, separator, _frames = output.partition(b"\n")
        assert separator == b"\n"
        ready = json.loads(line)
    if receipt is not None:
        assert verify_calls == 2
        assert len(streams) == 2
        assert streams[0].model is model
        assert streams[1].model is model
        assert streams[0]._executor_info is executor_info
        assert streams[1]._executor_info is executor_info
        assert [stream.reset_calls for stream in streams] == [1, 1]
    return receipt, ready, output, error


def test_global_cofilite_route_is_the_exact_ort_contract():
    spec = ns_routes.NS_ROUTES["cofilite"]
    assert spec.backend == spec.model_id == "cofilite"
    assert spec.model_path == (
        ns_routes.ROOT
        / "build/ns-cofilite-20261115/upstream/checkpoint/model_on_dns3.tar"
    )
    assert spec.model_sha256 == WEIGHTS_SHA256
    assert spec.frontend_path == ns_routes.ROOT / "server/cofilite_onnx.py"
    assert spec.frontend_sha256 == FRONTEND_SHA256
    assert spec.frontend_sha256 == hashlib.sha256(
        spec.frontend_path.read_bytes()
    ).hexdigest()
    assert spec.delay_samples == 480
    assert spec.lookahead_samples == 511
    assert spec.offline_whole_utterance is False
    assert spec.frame_executor == "onnxruntime_cpu"
    assert spec.frame_model_path == GRAPH_PATH
    assert spec.frame_model_sha256 == GRAPH_SHA256
    assert spec.frame_model_sha256 == hashlib.sha256(
        spec.frame_model_path.read_bytes()
    ).hexdigest()

    identity = ns_routes.ns_route_identity(_configuration())
    assert identity == {
        "ns_backend_requested": "cofilite",
        "ns_backend_effective": "cofilite",
        "ns_processing_mode": "stateful_streaming",
        "ns_model_id": "cofilite",
        "ns_model_sha256": WEIGHTS_SHA256,
        "ns_frontend_sha256": FRONTEND_SHA256,
        "ns_delay_samples": 480,
        "ns_sample_rate": 16000,
        "ns_frame_samples": 160,
        "ns_window_samples": 512,
        "ns_hop_samples": 256,
        "ns_offline_whole_utterance": False,
        "ns_lookahead_samples": 511,
        "ns_pcm16_quantization": "once_effective",
        "ns_frame_executor": "onnxruntime_cpu",
        "ns_frame_model_sha256": GRAPH_SHA256,
        "ns_frame_execution_provider": "CPUExecutionProvider",
    }


def test_route_validator_requires_exact_ort_identity():
    identity = ns_routes.ns_route_identity(_configuration())
    report = {
        **identity,
        **_stream_report(),
        "ns_boundary_input_sha256": "a" * 64,
        "ns_output_sha256": "b" * 64,
        "ns_inference_calls": 1,
        "ns_precomputed_replay": False,
        "ns_stateful": True,
        "ns_stream_chunk_samples": 160,
        "ns_execution_mode": "online_callback",
        "ns_worker_transport": "persistent_f32le_v1",
        "ns_state_resets": 1,
        "ns_realtime_verified": False,
        "ns_stream_callback_frames": 20,
    }
    assert ns_routes.validate_ns_route_report(report, _configuration()) == []

    for field, value in (
        ("ns_frame_executor", "jit_freeze"),
        ("ns_frame_model_sha256", "0" * 64),
        ("ns_frame_execution_provider", "CUDAExecutionProvider"),
    ):
        tampered = dict(report)
        tampered[field] = value
        errors = ns_routes.validate_ns_route_report(tampered, _configuration())
        assert any(field in error for error in errors)
        missing = dict(report)
        missing.pop(field)
        errors = ns_routes.validate_ns_route_report(missing, _configuration())
        assert any(field in error for error in errors)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda spec: replace(spec, frame_executor="jit_freeze"), "缺失或非法"),
        (lambda spec: replace(spec, frame_executor="onnxruntime_cuda"), "缺失或非法"),
        (
            lambda spec: replace(spec, frame_model_path=Path("/missing/cofilite.onnx")),
            "模型文件不存在",
        ),
    ],
)
def test_route_asset_verifier_rejects_bad_ort_spec(monkeypatch, mutation, message):
    spec = mutation(ns_routes.NS_ROUTES["cofilite"])
    monkeypatch.setitem(ns_routes.NS_ROUTES, "cofilite", spec)
    with pytest.raises(ValueError, match=message):
        ns_routes._verify_route_assets("cofilite")


def test_route_asset_verifier_rejects_tampered_graph(monkeypatch, tmp_path):
    graph = tmp_path / "tampered.onnx"
    graph.write_bytes(b"tampered graph\n")
    spec = replace(
        ns_routes.NS_ROUTES["cofilite"],
        frame_model_path=graph,
    )
    monkeypatch.setitem(ns_routes.NS_ROUTES, "cofilite", spec)
    with pytest.raises(ValueError, match="SHA-256 不匹配"):
        ns_routes._verify_route_assets("cofilite")


def test_missing_graph_fails_before_worker_or_cli(monkeypatch, tmp_path):
    command, _prefix = _command(tmp_path)
    spec = replace(
        ns_routes.NS_ROUTES["cofilite"],
        frame_model_path=tmp_path / "missing.onnx",
    )
    monkeypatch.setitem(ns_routes.NS_ROUTES, "cofilite", spec)
    calls = []
    monkeypatch.setattr(
        ns_streaming.subprocess,
        "Popen",
        lambda *args, **kwargs: calls.append(("worker", args, kwargs)),
    )
    monkeypatch.setattr(
        ns_streaming.subprocess,
        "run",
        lambda *args, **kwargs: calls.append(("cli", args, kwargs)),
    )

    with pytest.raises(ValueError, match="模型文件不存在"):
        ns_streaming.execute_stream_command(
            command,
            configuration=_configuration(),
            timeout=5,
        )

    assert calls == []


@pytest.mark.parametrize(
    "mutation",
    [
        lambda ready: ready.update(frame_executor="jit_freeze"),
        lambda ready: ready.update(frame_model_sha256="0" * 64),
        lambda ready: ready.update(
            frame_execution_provider="CUDAExecutionProvider"
        ),
        lambda ready: ready.pop("frame_executor"),
        lambda ready: ready.pop("frame_model_sha256"),
        lambda ready: ready.pop("frame_execution_provider"),
    ],
)
def test_ready_identity_is_exact_before_cli(monkeypatch, tmp_path, mutation):
    ready = _ready_payload()
    mutation(ready)
    command, _prefix, workers, runs = _install_fakes(
        monkeypatch,
        tmp_path,
        ready=(json.dumps(ready, separators=(",", ":")) + "\n").encode("ascii"),
        receipt=_receipt(),
        stderr_text=b"bad ORT ready\n",
    )

    with pytest.raises(ValueError, match="ready identity"):
        ns_streaming.execute_stream_command(
            command,
            configuration=_configuration(),
            timeout=5,
        )

    assert len(workers) == 1
    assert runs == []


def test_ort_receipt_passes_exact_identity_into_report(monkeypatch, tmp_path):
    frame_identity = _frame_identity()
    command, prefix, workers, runs = _install_fakes(
        monkeypatch,
        tmp_path,
        ready=_ready_line(),
        receipt=_receipt(frame_identity=frame_identity),
    )

    completed = ns_streaming.execute_stream_command(
        command,
        configuration=_configuration(),
        timeout=5,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert len(workers) == len(runs) == 1
    report = json.loads(Path(str(prefix) + ".json").read_text(encoding="utf-8"))
    stages = json.loads(Path(str(prefix) + ".stages.json").read_text(encoding="utf-8"))
    assert report["ns_frame_executor"] == "onnxruntime_cpu"
    assert report["ns_frame_model_sha256"] == GRAPH_SHA256
    assert report["ns_frame_execution_provider"] == "CPUExecutionProvider"
    assert report["ns_frame_executor_identity"] == frame_identity
    assert report["ns_execution"]["ns_frame_executor_identity"] == frame_identity
    assert stages["ns_model"] == report["ns_execution"]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("mode", "jit_freeze"),
        ("provider", "CUDAExecutionProvider"),
        ("model_sha256", "0" * 64),
        ("weights_sha256", "0" * 64),
        ("intra_op_num_threads", True),
        ("inter_op_num_threads", 2),
        ("state_count", 17),
        ("session_load_count", 2),
    ],
)
def test_receipt_identity_tamper_fails_closed(
    monkeypatch, tmp_path, key, value,
):
    frame_identity = _frame_identity()
    frame_identity[key] = value
    command, prefix, workers, runs = _install_fakes(
        monkeypatch,
        tmp_path,
        ready=_ready_line(),
        receipt=_receipt(frame_identity=frame_identity),
    )

    with pytest.raises(ValueError, match=f"frame executor identity mismatch: {key}"):
        ns_streaming.execute_stream_command(
            command,
            configuration=_configuration(),
            timeout=5,
        )

    assert len(workers) == len(runs) == 1
    assert not Path(str(prefix) + ".ns.json").exists()


def test_worker_ready_and_receipt_share_one_session_identity(tmp_path):
    receipt, ready, output, error = _invoke_worker(tmp_path)

    assert error is None
    assert receipt is not None
    assert ready == _ready_payload()
    assert output.count(b"\n") >= 1
    assert receipt["frame_executor"] == _frame_identity()
    assert receipt["model_initializations"] == 1
    assert receipt["live_stream_initializations"] == 1
    assert receipt["live_state_reset_calls"] == 1
    assert receipt["frame_executor"]["session_load_count"] == 1
    assert receipt["frame_executor"]["state_count"] == 18


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("mode", "jit_freeze"),
        ("provider", "CUDAExecutionProvider"),
        ("model_sha256", "0" * 64),
        ("weights_sha256", "0" * 64),
        ("intra_op_num_threads", True),
        ("inter_op_num_threads", 2),
        ("state_count", 17),
        ("session_load_count", 2),
    ],
)
def test_worker_rejects_bad_executor_before_ready(tmp_path, key, value):
    frame_identity = _frame_identity()
    frame_identity[key] = value
    receipt, ready, output, error = _invoke_worker(
        tmp_path,
        frame_identity=frame_identity,
    )

    assert receipt is None
    assert ready is None
    assert output == b""
    assert isinstance(error, ValueError)
    assert str(error) == f"stream frame executor not ready: {key}"


def test_worker_load_backend_creates_one_onnx_executor_for_both_streams(
    monkeypatch,
):
    from server import cofilite_onnx, cofilite_stream

    spec = ns_routes.NS_ROUTES["cofilite"]
    fake_model = object()
    executor = object()
    executor_calls = []
    stream_calls = []
    verify_calls = 0

    def verify_assets():
        nonlocal verify_calls
        verify_calls += 1

    def fake_get_executor(model, model_path, **kwargs):
        executor_calls.append((model, model_path, kwargs))
        return executor

    class FakeOnnxStream:
        def __init__(self, **kwargs):
            stream_calls.append(kwargs)
            self.frame_executor = kwargs["frame_executor"]
            self.model = kwargs["model"]

    monkeypatch.setattr(
        cofilite_stream,
        "load_model",
        lambda device, threads: fake_model,
    )
    monkeypatch.setattr(
        cofilite_onnx,
        "get_onnx_frame_executor",
        fake_get_executor,
    )
    monkeypatch.setattr(
        cofilite_onnx,
        "CoFiLiteOnnxStream",
        FakeOnnxStream,
    )

    model, factory, inference_mode = ns_stream_worker._load_cofilite_backend(
        spec,
        verify_assets,
    )
    warmup = factory()
    live = factory()

    assert model is fake_model
    assert verify_calls == 1
    assert executor_calls == [(
        fake_model,
        GRAPH_PATH,
        {
            "expected_model_sha256": GRAPH_SHA256,
            "expected_weights_sha256": WEIGHTS_SHA256,
        },
    )]
    assert len(stream_calls) == 2
    assert warmup.frame_executor is live.frame_executor is executor
    assert inference_mode.__name__ == "inference_mode"
