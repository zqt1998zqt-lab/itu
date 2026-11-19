"""Lifecycle contract for the native 48 kHz DPDFNet2 HR stream bridge."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
import soundfile as sf

from server import ns_routes, ns_stream_worker_48k_hr, ns_streaming_48k_hr


def _configuration() -> dict:
    return {
        "profile": "vcp",
        "selection": {
            "bf": "das",
            "aec": "webrtc",
            "ns": "dpdfnet2_48k_hr",
            "eq": "three_band",
            "agc": "guarded5",
            "drc": "default",
            "limiter": "on",
        },
    }


def _command(tmp_path: Path) -> list[str]:
    cli = tmp_path / "itu_audio_cli"
    cli.write_bytes(b"#!/bin/sh\nexit 0\n")
    cli.chmod(0o755)
    mic = tmp_path / "mic.wav"
    sf.write(mic, np.zeros(480, dtype=np.float32), 48_000, subtype="PCM_16")
    return [
        str(cli),
        "--profile", "vcp",
        "--aec-backend", "webrtc",
        "--ns-backend", "webrtc",
        "--mic", str(mic),
        "--out", str(tmp_path / "out"),
    ]


def _timing(callbacks: int) -> dict:
    summary = {
        "count": callbacks,
        "total_seconds": 0.1,
        "p50_upper_ms": 1.0,
        "p95_upper_ms": 1.2,
        "p99_upper_ms": 1.4,
        "max_ms": 1.5,
        "over_10ms": 0,
    }
    return {name: dict(summary) for name in (
        "read_wait", "quantize_hash_preprocess", "model_process",
        "postprocess_write", "callback",
    )}


def _receipt(command: list[str], callbacks: int = 12) -> dict:
    timing = _timing(callbacks)
    samples = [
        {
            "frame_index": index + 1,
            "read_wait_ms": 0.1,
            "preprocess_ms": 0.2,
            "model_process_ms": 0.8,
            "postprocess_write_ms": 0.1,
            "callback_ms": 1.1,
        }
        for index in range(callbacks)
    ]
    return {
        "protocol": "itu-ns-f32le-v1",
        "backend": "dpdfnet2_48k_hr",
        "sample_rate": 48_000,
        "frame_samples": 480,
        "window_samples": 960,
        "hop_samples": 480,
        "delay_samples": 2_400,
        "model_latency_samples": 1_920,
        "model_latency_frames": 4,
        "startup_delay_samples": 480,
        "processing_mode": "stateful_streaming",
        "offline_whole_utterance": False,
        "lookahead_samples": 960,
        "model_sha256": ns_routes.NS_ROUTES["dpdfnet2_48k_hr"].model_sha256,
        "frontend_sha256": ns_routes.NS_ROUTES["dpdfnet2_48k_hr"].frontend_sha256,
        "schema_version": 2,
        "stateful": True,
        "precomputed_replay": False,
        "callback_frames": callbacks,
        "samples": callbacks * 480,
        "input_payload_sha256": "1" * 64,
        "quantized_input_pcm16_sha256": "2" * 64,
        "output_payload_sha256": "3" * 64,
        "model_inferences": callbacks - 1,
        "live_model_inferences": callbacks - 1,
        "expected_model_inferences": callbacks - 1,
        "model_hop_samples": 480,
        "model_ready_threshold_samples": 960,
        "model_initializations": 1,
        "live_stream_initializations": 1,
        "live_state_reset_calls": 1,
        "warmup_calls": 0,
        "warmup_inferences": 0,
        "pre_roll_frames": 0,
        "pre_roll_output_frames": 0,
        "delay_frames": 5,
        "state_bytes": 240_000,
        "model_output_frames": callbacks - 1,
        "aligned_model_output_frames": max(0, callbacks - 5),
        "pending_output_frames": 0,
        "inference_seconds": 0.1,
        "model_process_seconds": 0.1,
        "rtf_model": 0.1,
        "callback_seconds": 0.1,
        "callback_max_ms": 1.5,
        "callback_p50_upper_ms": 1.0,
        "callback_p95_upper_ms": 1.2,
        "callback_p99_upper_ms": 1.4,
        "callback_over_10ms": 0,
        "startup_load_seconds": 0.1,
        "warmup_seconds": 0.1,
        "startup_to_ready_seconds": 0.2,
        "startup_timing_scope": "module import through ready",
        "timing_clock": "time.perf_counter monotonic seconds",
        "timing_scope": "worker service time",
        "timing_sample_policy": "trailing bounded window",
        "timing_sample_capacity": 16,
        "timing_samples": samples,
        "timing": timing,
        "worker_source_sha256": "4" * 64,
        "worker_helper_source_sha256": "5" * 64,
        "worker_command": command,
        "worker_command_sha256": hashlib.sha256(
            json.dumps(command, ensure_ascii=True, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "python_executable": "/fake/python",
        "peak_rss_bytes": 64 * 1024 * 1024,
        "stateful": True,
    }


class _FakeWorker:
    def __init__(self, command, receipt, ready, stderr):
        self.command = command
        self.receipt = receipt
        self.returncode = None
        input_read, input_write = os.pipe()
        os.close(input_read)
        output_read, output_write = os.pipe()
        self.stdin = os.fdopen(input_write, "wb", buffering=0)
        self.stdout = os.fdopen(output_read, "rb", buffering=0)
        self._stdout_write = os.fdopen(output_write, "wb", buffering=0)
        self.stdin_fd = self.stdin.fileno()
        self.stdout_fd = self.stdout.fileno()
        self._stdout_write.write(ready)
        stderr.write(b"worker log\n")
        stderr.flush()

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if not self._stdout_write.closed:
            self._stdout_write.close()
        self.returncode = 0
        receipt_path = Path(self.command[self.command.index("--receipt") + 1])
        receipt_path.write_text(json.dumps(self.receipt), encoding="utf-8")
        return self.returncode

    def terminate(self):
        if not self._stdout_write.closed:
            self._stdout_write.close()
        self.returncode = -15

    def kill(self):
        if not self._stdout_write.closed:
            self._stdout_write.close()
        self.returncode = -9


def test_stream_command_dispatches_native_48k_callback(monkeypatch, tmp_path):
    command = _command(tmp_path)
    prefix = Path(command[command.index("--out") + 1])
    callbacks = 12
    monkeypatch.setattr(ns_routes, "_verify_route_assets", lambda backend: None)
    monkeypatch.setattr(ns_routes, "_model_python", lambda: "/fake/python")
    fake_receipt = _receipt([], callbacks)
    workers = []
    runs = []

    def fake_popen(worker_command, **kwargs):
        receipt = dict(fake_receipt)
        receipt["worker_source_sha256"] = hashlib.sha256(
            Path(ns_stream_worker_48k_hr.__file__).read_bytes()
        ).hexdigest()
        receipt["worker_helper_source_sha256"] = hashlib.sha256(
            Path(__import__("server.ns_stream_worker", fromlist=["__file__"]).__file__).read_bytes()
        ).hexdigest()
        receipt["worker_command"] = list(worker_command)
        receipt["worker_command_sha256"] = hashlib.sha256(
            json.dumps(list(worker_command), ensure_ascii=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        worker = _FakeWorker(
            list(worker_command), receipt,
            (json.dumps({
                "protocol": "itu-ns-f32le-v1",
                "backend": "dpdfnet2_48k_hr",
                "sample_rate": 48_000,
                "frame_samples": 480,
                "window_samples": 960,
                "hop_samples": 480,
                "delay_samples": 2_400,
                "model_latency_samples": 1_920,
                "model_latency_frames": 4,
                "startup_delay_samples": 480,
                "model_sha256": ns_routes.NS_ROUTES["dpdfnet2_48k_hr"].model_sha256,
                "frontend_sha256": ns_routes.NS_ROUTES["dpdfnet2_48k_hr"].frontend_sha256,
                "processing_mode": "stateful_streaming",
                "offline_whole_utterance": False,
                "lookahead_samples": 960,
            }, separators=(",", ":")) + "\n").encode(),
            kwargs["stderr"],
        )
        workers.append(worker)
        return worker

    def fake_run(online, **kwargs):
        runs.append((list(online), kwargs))
        report = {
            "ns_backend": "external_stream",
            "ns_external_stream_enabled": True,
            "ns_external_pcm_enabled": False,
            "ns_external_applied_frames": callbacks,
            "ns_external_fallback_frames": 0,
            "ns_external_delay_samples": 2_400,
            "ns_stream_chunk_samples": 480,
            "ns_stream_frame_samples": 480,
            "ns_stream_sample_rate": 48_000,
            "sample_rate": 48_000,
            "frame_size": 480,
            "aec_backend_effective": "webrtc_aec3",
        }
        Path(str(prefix) + ".json").write_text(json.dumps(report), encoding="utf-8")
        Path(str(prefix) + ".stages.json").write_text(
            json.dumps({"pipeline_stages": []}), encoding="utf-8"
        )
        return subprocess.CompletedProcess(online, 0, "native stream ok\n", "")

    monkeypatch.setattr(ns_streaming_48k_hr.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(ns_streaming_48k_hr.subprocess, "run", fake_run)
    completed = ns_streaming_48k_hr.execute_stream_command(
        command,
        configuration=_configuration(),
        timeout=10,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert len(workers) == 1 and len(runs) == 1
    online = runs[0][0]
    assert online[online.index("--ns-stream-delay-samples") + 1] == "2400"
    report = json.loads(Path(str(prefix) + ".json").read_text(encoding="utf-8"))
    assert report["ns_backend"] == "dpdfnet2_48k_hr"
    assert report["ns_execution_mode"] == "online_callback"
    assert report["ns_stateful"] is True
    assert report["ns_precomputed_replay"] is False
    assert report["ns_stream_chunk_samples"] == 480
    assert report["ns_stream_callback_frames"] == 12
    assert report["ns_execution"]["ns_stream_callback_frames"] == 12
