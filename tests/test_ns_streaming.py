"""Lifecycle tests for the persistent CoFi-Lite native streaming bridge."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
import soundfile as sf

from server import ns_routes, ns_stream_worker, ns_streaming


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


def _ready_payload(**updates) -> dict:
    spec = ns_routes.NS_ROUTES["cofilite"]
    payload = {
        "protocol": "itu-ns-f32le-v1",
        "backend": "cofilite",
        "sample_rate": 16_000,
        "frame_samples": 160,
        "delay_samples": spec.delay_samples,
        "model_sha256": spec.model_sha256,
        "frontend_sha256": spec.frontend_sha256,
        "processing_mode": spec.processing_mode,
    }
    if spec.frame_executor is not None:
        payload.update({
            "frame_executor": spec.frame_executor,
            "frame_model_sha256": spec.frame_model_sha256,
            "frame_execution_provider": "CPUExecutionProvider",
        })
    payload.update(updates)
    return payload


def _ready_line(**updates) -> bytes:
    return (json.dumps(_ready_payload(**updates), separators=(",", ":")) + "\n").encode("ascii")


def _frame_executor_identity() -> dict:
    spec = ns_routes.NS_ROUTES["cofilite"]
    if spec.frame_executor is None:
        return {}
    return {
        "mode": spec.frame_executor,
        "provider": "CPUExecutionProvider",
        "model_sha256": spec.frame_model_sha256,
        "weights_sha256": spec.model_sha256,
        "intra_op_num_threads": 1,
        "inter_op_num_threads": 1,
        "state_count": 18,
        "session_load_count": 1,
    }


def _receipt(callback_frames: int = 20) -> dict:
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
            "read_wait", "quantize_hash_preprocess", "model_process",
            "postprocess_write", "callback",
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
        "frame_executor": _frame_executor_identity(),
    }


def _native_report(callback_frames: int = 20) -> dict:
    spec = ns_routes.NS_ROUTES["cofilite"]
    return {
        "ns_backend": "external_stream",
        "ns_external_stream_enabled": True,
        "ns_external_pcm_enabled": False,
        "ns_external_applied_frames": callback_frames,
        "ns_external_fallback_frames": 0,
        "ns_external_delay_samples": spec.delay_samples,
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
        "--profile", "vcp",
        "--aec-backend", "localvqe_aec200k",
        "--ns-backend", "webrtc",
        "--mic", str(mic),
        "--out", str(prefix),
    ], prefix


class _FakeWorker:
    def __init__(
        self,
        command: list[str],
        *,
        ready: bytes,
        stderr_text: bytes,
        receipt: dict | None,
        close_stdout: bool,
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
        if close_stdout:
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
    close_stdout: bool | None = None,
    stderr_text: bytes = b"",
    receipt: dict | None = None,
    callback_frames: int = 20,
):
    command, prefix = _command(tmp_path)
    workers: list[_FakeWorker] = []
    runs: list[tuple[list[str], dict]] = []

    monkeypatch.setattr(ns_routes, "_verify_route_assets", lambda backend: None)
    monkeypatch.setattr(ns_routes, "_model_python", lambda: "/fake/python")

    def fake_popen(worker_command, **kwargs):
        worker_receipt_data = receipt
        if receipt is not None:
            worker_receipt_data = dict(receipt)
            if worker_receipt_data.get("worker_source_sha256") == "4" * 64:
                worker_receipt_data["worker_source_sha256"] = hashlib.sha256(
                    Path(ns_stream_worker.__file__).read_bytes()
                ).hexdigest()
            if worker_receipt_data.get("worker_command_sha256") == "5" * 64:
                worker_receipt_data["worker_command"] = list(worker_command)
                worker_receipt_data["worker_command_sha256"] = ns_streaming._command_sha256(
                    list(worker_command)
                )
        worker = _FakeWorker(
            list(worker_command),
            ready=ready,
            stderr_text=stderr_text,
            receipt=worker_receipt_data,
            close_stdout=not ready if close_stdout is None else close_stdout,
            stderr=kwargs["stderr"],
        )
        workers.append(worker)
        return worker

    def fake_run(online_command, **kwargs):
        online = list(online_command)
        runs.append((online, kwargs))
        output_prefix = Path(online[online.index("--out") + 1])
        Path(str(output_prefix) + ".json").write_text(
            json.dumps(_native_report(callback_frames), indent=2) + "\n",
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


def test_success_runs_cli_once_with_worker_fds(monkeypatch, tmp_path):
    receipt = _receipt()
    command, prefix, workers, runs = _install_fakes(
        monkeypatch,
        tmp_path,
        ready=_ready_line(),
        stderr_text=b"worker boot diagnostics retained\n",
        receipt=receipt,
    )

    completed = ns_streaming.execute_stream_command(
        command,
        configuration=_configuration(),
        timeout=5,
        capture_output=True,
        text=True,
    )

    assert len(workers) == 1
    assert len(runs) == 1
    online, options = runs[0]
    assert online[0] == command[0]
    assert online[online.index("--ns-stream-input-fd") + 1] == str(workers[0].stdin_fd)
    assert online[online.index("--ns-stream-output-fd") + 1] == str(workers[0].stdout_fd)
    spec = ns_routes.NS_ROUTES["cofilite"]
    assert online[online.index("--ns-stream-delay-samples") + 1] == str(spec.delay_samples)
    assert online.count("--diagnostic-stages") == 1
    assert {workers[0].stdin_fd, workers[0].stdout_fd}.issubset(set(options["pass_fds"]))
    assert options["capture_output"] is True
    assert options["text"] is True
    assert 0 < options["timeout"] <= 5

    report = json.loads(Path(str(prefix) + ".json").read_text(encoding="utf-8"))
    stages = json.loads(Path(str(prefix) + ".stages.json").read_text(encoding="utf-8"))
    assert report["ns_backend"] == "cofilite"
    assert report["ns_external_stream_enabled"] is True
    assert report["ns_external_pcm_enabled"] is False
    assert report["ns_stream_callback_frames"] == 20
    assert report["ns_execution"]["ns_stream_callback_frames"] == 20
    assert report["ns_execution"]["ns_worker_transport"] == "persistent_f32le_v1"
    assert report["ns_execution"]["ns_stream_callback_p50_upper_ms"] == 1.0
    assert report["ns_execution"]["ns_stream_callback_p99_upper_ms"] == 1.4
    assert report["ns_execution"]["ns_stream_startup_load_s"] == 0.2
    assert report["ns_execution"]["ns_stream_warmup_s"] == 0.1
    assert report["ns_execution"]["ns_worker_source_sha256"] == hashlib.sha256(
        Path(ns_stream_worker.__file__).read_bytes()
    ).hexdigest()
    assert len(report["ns_execution"]["ns_stream_timing_samples"]) == 16
    frame_identity = _frame_executor_identity()
    for field, value in {
        "ns_frame_executor": "onnxruntime_cpu",
        "ns_frame_model_sha256": spec.frame_model_sha256,
        "ns_frame_execution_provider": "CPUExecutionProvider",
    }.items():
        assert report[field] == value
        assert report["ns_execution"][field] == value
    assert report["ns_frame_executor_identity"] == frame_identity
    assert report["ns_execution"]["ns_frame_executor_identity"] == frame_identity
    assert stages["ns_model"] == report["ns_execution"]
    assert Path(str(prefix) + ".ns-stream.json").is_file()
    assert Path(str(prefix) + ".ns-stream.log").read_bytes() == b"worker boot diagnostics retained\n"
    assert completed.returncode == 0
    assert completed.stdout.endswith(
        "NS model: cofilite stateful live 160-sample callbacks\n"
    )


def test_ready_identity_mismatch_fails_before_cli_and_keeps_stderr(monkeypatch, tmp_path):
    command, prefix, workers, runs = _install_fakes(
        monkeypatch,
        tmp_path,
        ready=_ready_line(model_sha256="0" * 64),
        stderr_text=b"Traceback: stream worker ready identity mismatch\n",
        receipt=_receipt(),
    )

    with pytest.raises(ValueError, match="ready identity"):
        ns_streaming.execute_stream_command(
            command, configuration=_configuration(), timeout=5,
        )

    assert len(workers) == 1
    assert runs == []
    assert Path(str(prefix) + ".ns-stream.log").read_bytes() == (
        b"Traceback: stream worker ready identity mismatch\n"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda ready: ready.update(frame_executor="jit_freeze"),
        lambda ready: ready.update(frame_model_sha256="0" * 64),
        lambda ready: ready.update(frame_execution_provider="CUDAExecutionProvider"),
        lambda ready: ready.pop("frame_executor"),
        lambda ready: ready.pop("frame_model_sha256"),
        lambda ready: ready.pop("frame_execution_provider"),
    ],
)
def test_ready_ort_identity_mismatch_fails_before_cli(
    monkeypatch, tmp_path, mutation,
):
    ready = _ready_payload()
    mutation(ready)
    command, prefix, workers, runs = _install_fakes(
        monkeypatch,
        tmp_path,
        ready=(json.dumps(ready, separators=(",", ":")) + "\n").encode("ascii"),
        stderr_text=b"worker ORT ready identity mismatch\n",
        receipt=_receipt(),
    )

    with pytest.raises(ValueError, match="ready identity"):
        ns_streaming.execute_stream_command(
            command, configuration=_configuration(), timeout=5,
        )

    assert len(workers) == 1
    assert runs == []
    assert Path(str(prefix) + ".ns-stream.log").read_bytes() == (
        b"worker ORT ready identity mismatch\n"
    )


@pytest.mark.parametrize(
    "ready",
    [b"", b'{"protocol":"itu-ns-f32le-v1","backend":"cofilite"'],
)
def test_ready_eof_or_truncated_json_fails_closed(monkeypatch, tmp_path, ready):
    command, prefix, workers, runs = _install_fakes(
        monkeypatch,
        tmp_path,
        ready=ready,
        close_stdout=True,
        stderr_text=b"worker stderr evidence\n",
        receipt=_receipt(),
    )

    with pytest.raises(ValueError, match="ready"):
        ns_streaming.execute_stream_command(
            command, configuration=_configuration(), timeout=5,
        )

    assert len(workers) == 1
    assert runs == []
    assert Path(str(prefix) + ".ns-stream.log").read_bytes() == b"worker stderr evidence\n"


def test_ready_timeout_stops_before_cli(monkeypatch, tmp_path):
    command, prefix, workers, runs = _install_fakes(
        monkeypatch,
        tmp_path,
        ready=b"",
        stderr_text=b"worker still initializing\n",
        receipt=_receipt(),
    )
    monkeypatch.setattr(
        ns_streaming.select,
        "select",
        lambda read, write, error, timeout: ([], [], []),
    )

    with pytest.raises(TimeoutError, match="initialization timed out"):
        ns_streaming.execute_stream_command(
            command, configuration=_configuration(), timeout=5,
        )

    assert len(workers) == 1
    assert runs == []
    assert Path(str(prefix) + ".ns-stream.log").read_bytes() == b"worker still initializing\n"


def test_callback_count_mismatch_fails_after_single_cli_run(monkeypatch, tmp_path):
    command, prefix, workers, runs = _install_fakes(
        monkeypatch,
        tmp_path,
        ready=_ready_line(),
        stderr_text=b"worker completed\n",
        receipt=_receipt(callback_frames=19),
        callback_frames=20,
    )

    with pytest.raises(ValueError, match="callback counts"):
        ns_streaming.execute_stream_command(
            command, configuration=_configuration(), timeout=5,
        )

    assert len(workers) == 1
    assert len(runs) == 1
    assert Path(str(prefix) + ".ns-stream.json").is_file()
    assert Path(str(prefix) + ".ns-stream.log").read_bytes() == b"worker completed\n"


def test_worker_exact_read_distinguishes_clean_eof_and_truncated_frame():
    size = ns_stream_worker.FRAME_BYTES
    assert ns_stream_worker._exact_read(io.BytesIO(b""), size) == b""
    with pytest.raises(ValueError, match="truncated"):
        ns_stream_worker._exact_read(io.BytesIO(b"\0" * (size - 1)), size)


def test_worker_identity_accepts_cofi_hop_schedule_not_callback_equality():
    receipt = _receipt(callback_frames=1208)
    ns_streaming._validate_worker_identity(receipt, 1208)

    corrupt = dict(receipt, model_inferences=1208)
    with pytest.raises(ValueError, match="hop schedule"):
        ns_streaming._validate_worker_identity(corrupt, 1208)


def test_runtime_delay_comes_from_route_spec(monkeypatch, tmp_path):
    spec = ns_routes.NS_ROUTES["cofilite"]
    monkeypatch.setitem(
        ns_routes.NS_ROUTES,
        "cofilite",
        replace(spec, delay_samples=480),
    )
    command, _prefix, workers, runs = _install_fakes(
        monkeypatch,
        tmp_path,
        ready=_ready_line(),
        receipt=_receipt(),
    )

    ns_streaming.execute_stream_command(
        command, configuration=_configuration(), timeout=5,
    )

    online, _options = runs[0]
    assert online[online.index("--ns-stream-delay-samples") + 1] == "480"
    assert len(workers) == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda receipt: receipt.update(model_inferences=19), "inference count"),
        (lambda receipt: receipt.update(worker_source_sha256="not-a-sha"), "identity"),
        (lambda receipt: receipt.update(worker_command_sha256="0" * 64), "command identity"),
        (
            lambda receipt: receipt["timing_samples"].append(
                dict(receipt["timing_samples"][-1], frame_index=99)
            ),
            "unbounded",
        ),
        (lambda receipt: receipt.pop("timing_sample_policy"), "sample policy"),
    ],
)
def test_corrupt_worker_timing_or_identity_fails_closed(
    monkeypatch, tmp_path, mutation, message,
):
    receipt = _receipt()
    mutation(receipt)
    command, _prefix, workers, runs = _install_fakes(
        monkeypatch,
        tmp_path,
        ready=_ready_line(),
        receipt=receipt,
    )

    with pytest.raises(ValueError, match=message):
        ns_streaming.execute_stream_command(
            command, configuration=_configuration(), timeout=5,
        )

    assert len(workers) == 1
    assert len(runs) == 1


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
def test_receipt_ort_executor_identity_mismatch_fails_closed(
    monkeypatch, tmp_path, key, value,
):
    receipt = _receipt()
    receipt["frame_executor"][key] = value
    command, prefix, workers, runs = _install_fakes(
        monkeypatch,
        tmp_path,
        ready=_ready_line(),
        receipt=receipt,
    )

    with pytest.raises(ValueError, match=f"frame executor identity mismatch: {key}"):
        ns_streaming.execute_stream_command(
            command, configuration=_configuration(), timeout=5,
        )

    assert len(workers) == 1
    assert len(runs) == 1
    assert not Path(str(prefix) + ".ns.json").exists()


def test_receipt_ort_executor_identity_is_required(monkeypatch, tmp_path):
    receipt = _receipt()
    receipt.pop("frame_executor")
    command, prefix, workers, runs = _install_fakes(
        monkeypatch,
        tmp_path,
        ready=_ready_line(),
        receipt=receipt,
    )

    with pytest.raises(ValueError, match="omitted its frame executor identity"):
        ns_streaming.execute_stream_command(
            command, configuration=_configuration(), timeout=5,
        )

    assert len(workers) == 1
    assert len(runs) == 1
    assert not Path(str(prefix) + ".ns.json").exists()
