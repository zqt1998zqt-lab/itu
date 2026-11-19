"""Control-thread lifecycle for native 48 kHz DPDFNet2 HR callbacks."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import select
import subprocess
import time


FRAME_SAMPLES = 480
MODEL_LATENCY_SAMPLES = 1_920
STARTUP_DELAY_SAMPLES = 480
DELAY_SAMPLES = MODEL_LATENCY_SAMPLES + STARTUP_DELAY_SAMPLES
DELAY_FRAMES = DELAY_SAMPLES // FRAME_SAMPLES
WINDOW_SAMPLES = 960
_SHA256_LENGTH = 64


def _finite_nonnegative(value) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
    )


def _ready(stream, deadline: float) -> dict:
    line = bytearray()
    while len(line) < 4096:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("48 kHz HR stream worker initialization timed out")
        try:
            selected = select.select([stream], [], [], remaining)[0]
        except InterruptedError:
            continue
        if not selected:
            raise TimeoutError("48 kHz HR stream worker initialization timed out")
        try:
            character = os.read(stream.fileno(), 1)
        except InterruptedError:
            continue
        if not character:
            raise ValueError("48 kHz HR stream worker exited before ready; inspect stream-worker.log")
        line.extend(character)
        if character == b"\n":
            return json.loads(line)
    raise ValueError("invalid 48 kHz HR streaming readiness message")


def _validate_worker_pipe_fd(fd: int, *, writable: bool) -> None:
    from server.ns_streaming import _validate_worker_pipe_fd as validate

    validate(fd, writable=writable)


def _validate_timing_receipt(receipt: dict, callbacks: int) -> None:
    from server.ns_streaming import _validate_timing_receipt as validate

    validate(receipt, callbacks)


def _validate_worker_identity(receipt: dict, callbacks: int) -> None:
    if not isinstance(receipt, dict):
        raise ValueError("48 kHz HR worker receipt must be an object")
    expected_warmup = 0
    expected_inferences = max(0, callbacks - 1)
    if receipt.get("samples") != callbacks * FRAME_SAMPLES:
        raise ValueError("48 kHz HR worker sample count does not match callbacks")
    if receipt.get("callback_frames") != callbacks:
        raise ValueError("48 kHz HR worker callback count does not match C++")
    if receipt.get("model_hop_samples") != FRAME_SAMPLES:
        raise ValueError("48 kHz HR worker hop geometry is invalid")
    if receipt.get("model_ready_threshold_samples") != WINDOW_SAMPLES:
        raise ValueError("48 kHz HR worker ready threshold is invalid")
    if receipt.get("model_inferences") != expected_inferences:
        raise ValueError("48 kHz HR worker inference count does not match hop schedule")
    if receipt.get("live_model_inferences") != expected_inferences:
        raise ValueError("48 kHz HR worker live inference count is invalid")
    if receipt.get("expected_model_inferences") != expected_inferences:
        raise ValueError("48 kHz HR worker expected inference count is invalid")
    if receipt.get("model_output_frames") != max(0, callbacks - 1):
        raise ValueError("48 kHz HR worker raw output frame count is invalid")
    if receipt.get("aligned_model_output_frames") != max(0, callbacks - DELAY_FRAMES):
        raise ValueError("48 kHz HR worker aligned output frame count is invalid")
    if receipt.get("pending_output_frames") != 0:
        raise ValueError("48 kHz HR worker ended with an incomplete callback queue")
    if receipt.get("warmup_calls") != 0 or receipt.get("warmup_inferences") != expected_warmup:
        raise ValueError("48 kHz HR worker pre-roll geometry is invalid")
    if (
        receipt.get("pre_roll_frames") != 0
        or receipt.get("pre_roll_output_frames") != expected_warmup
    ):
        raise ValueError("48 kHz HR worker pre-roll output proof is invalid")
    for field, value in {
        "model_initializations": 1,
        "live_stream_initializations": 1,
        "live_state_reset_calls": 1,
        "delay_frames": DELAY_FRAMES,
        "delay_samples": DELAY_SAMPLES,
        "model_latency_samples": MODEL_LATENCY_SAMPLES,
        "model_latency_frames": 4,
        "startup_delay_samples": STARTUP_DELAY_SAMPLES,
        "stateful": True,
        "precomputed_replay": False,
        "offline_whole_utterance": False,
    }.items():
        if receipt.get(field) != value:
            raise ValueError(f"48 kHz HR worker receipt {field} is invalid")
    if not isinstance(receipt.get("state_bytes"), int) or receipt["state_bytes"] < 1:
        raise ValueError("48 kHz HR worker state size is invalid")
    for field in ("startup_load_seconds", "warmup_seconds", "startup_to_ready_seconds"):
        if not _finite_nonnegative(receipt.get(field)):
            raise ValueError(f"48 kHz HR worker startup timing is invalid: {field}")
    for field in ("startup_timing_scope", "timing_clock", "timing_scope"):
        if not isinstance(receipt.get(field), str) or not receipt[field]:
            raise ValueError(f"48 kHz HR worker timing metadata is invalid: {field}")
    for field in ("worker_source_sha256", "worker_helper_source_sha256", "worker_command_sha256"):
        value = receipt.get(field)
        if not isinstance(value, str) or len(value) != _SHA256_LENGTH:
            raise ValueError(f"48 kHz HR worker identity is invalid: {field}")
    command = receipt.get("worker_command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(part, str) or not part for part in command)
    ):
        raise ValueError("48 kHz HR worker command identity is invalid")
    _validate_timing_receipt(receipt, callbacks)


def execute_stream_command(command, *, configuration: dict, timeout: float, **kwargs):
    from server import ns_routes as routes
    from server import ns_stream_worker_48k_hr

    spec = routes.NS_ROUTES["dpdfnet2_48k_hr"]
    prefix, original = routes._external_command(command)
    if prefix is None:
        raise ValueError("48 kHz HR online route requires a TX microphone command")
    aec_backends = routes._command_values(original, "--aec-backend")
    if routes._has_flag(original, "--no-aec") or aec_backends != ["webrtc"]:
        raise ValueError(
            "DPDFNet2 48 kHz HR 质量路线必须显式使用 --aec-backend webrtc"
        )
    routes._validate_mic_header(original, spec.sample_rate)
    routes._verify_route_assets("dpdfnet2_48k_hr")
    cli = Path(original[0]).resolve()
    cli_sha = routes._sha256_file(cli)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    worker_receipt = routes._artifact(prefix, ".ns-stream.json")
    worker_log = routes._artifact(prefix, ".ns-stream.log")
    model_dir = routes._artifact(prefix, ".ns-model")
    model_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + float(timeout)
    environment = dict(os.environ)
    environment.update({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "ORT_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
    })
    worker_command = [
        routes._model_python(),
        "-m",
        "server.ns_stream_worker_48k_hr",
        "--receipt",
        str(worker_receipt),
    ]
    worker_source_path = Path(ns_stream_worker_48k_hr.__file__).resolve()
    helper_source_path = Path(__import__("server.ns_stream_worker", fromlist=["__file__"]).__file__).resolve()
    worker_source_sha = routes._sha256_file(worker_source_path)
    helper_source_sha = routes._sha256_file(helper_source_path)
    worker_command_sha = hashlib.sha256(
        json.dumps(worker_command, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()

    with worker_log.open("xb") as stderr:
        worker = subprocess.Popen(
            worker_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            env=environment,
            cwd=routes.ROOT,
            bufsize=0,
        )
        try:
            tx_fd, rx_fd = worker.stdin.fileno(), worker.stdout.fileno()
            if tx_fd == rx_fd:
                raise ValueError("stream worker parent fds must be distinct")
            _validate_worker_pipe_fd(tx_fd, writable=True)
            _validate_worker_pipe_fd(rx_fd, writable=False)
            ready = _ready(worker.stdout, min(deadline, time.monotonic() + 60))
            expected_ready = {
                "protocol": "itu-ns-f32le-v1",
                "backend": "dpdfnet2_48k_hr",
                "sample_rate": 48_000,
                "frame_samples": FRAME_SAMPLES,
                "window_samples": WINDOW_SAMPLES,
                "hop_samples": FRAME_SAMPLES,
                "delay_samples": DELAY_SAMPLES,
                "model_latency_samples": MODEL_LATENCY_SAMPLES,
                "model_latency_frames": 4,
                "startup_delay_samples": STARTUP_DELAY_SAMPLES,
                "model_sha256": spec.model_sha256,
                "frontend_sha256": spec.frontend_sha256,
                "processing_mode": "stateful_streaming",
                "offline_whole_utterance": False,
                "lookahead_samples": WINDOW_SAMPLES,
            }
            if ready != expected_ready:
                raise ValueError("48 kHz HR worker ready identity does not match the selected route")

            online = [
                *original,
                "--ns-stream-input-fd",
                str(tx_fd),
                "--ns-stream-output-fd",
                str(rx_fd),
                "--ns-stream-delay-samples",
                str(DELAY_SAMPLES),
            ]
            if "--diagnostic-stages" not in online:
                online.append("--diagnostic-stages")
            options = dict(kwargs)
            options["pass_fds"] = tuple(set(options.get("pass_fds", ())) | {tx_fd, rx_fd})
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("48 kHz HR stream initialization exceeded execution timeout")
            completed = subprocess.run(online, timeout=remaining, **options)
            worker.stdin.close()
            if completed.returncode != 0:
                raise routes._process_error("48 kHz HR online VCP", completed)
            worker.wait(timeout=max(0.1, min(30, deadline - time.monotonic())))
            if worker.returncode != 0 or not worker_receipt.is_file():
                raise ValueError("48 kHz HR stream worker failed; inspect " + str(worker_log))

            receipt = json.loads(worker_receipt.read_text(encoding="utf-8"))
            report_path = routes._artifact(prefix, ".json")
            stages_path = routes._artifact(prefix, ".stages.json")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            for key, expected in {
                "ns_backend": "external_stream",
                "ns_external_stream_enabled": True,
                "ns_external_pcm_enabled": False,
                "ns_external_fallback_frames": 0,
                "ns_external_delay_samples": DELAY_SAMPLES,
                "ns_stream_chunk_samples": FRAME_SAMPLES,
                "ns_stream_frame_samples": FRAME_SAMPLES,
                "ns_stream_sample_rate": 48_000,
                "sample_rate": 48_000,
                "frame_size": FRAME_SAMPLES,
                "aec_backend_effective": "webrtc_aec3",
            }.items():
                if report.get(key) != expected:
                    raise ValueError(f"48 kHz HR native stream report mismatch: {key}")
            callbacks = report.get("ns_external_applied_frames")
            if (
                isinstance(callbacks, bool)
                or not isinstance(callbacks, int)
                or callbacks <= 0
                or callbacks != receipt.get("callback_frames")
            ):
                raise ValueError("48 kHz HR callback counts are invalid")
            _validate_worker_identity(receipt, callbacks)
            if receipt.get("worker_source_sha256") != worker_source_sha:
                raise ValueError("48 kHz HR worker source identity changed during processing")
            if receipt.get("worker_helper_source_sha256") != helper_source_sha:
                raise ValueError("48 kHz HR worker helper identity changed during processing")
            if receipt.get("worker_command_sha256") != worker_command_sha:
                raise ValueError("48 kHz HR worker command identity does not match the launched command")
            if routes._sha256_file(cli) != cli_sha:
                raise ValueError("48 kHz HR VCP executable identity changed during processing")
            routes._verify_route_assets("dpdfnet2_48k_hr")

            identity = routes.ns_route_identity(configuration)
            if identity is None:
                raise ValueError("48 kHz HR route identity is unexpectedly empty")
            identity = dict(identity)
            identity.update({
                "ns_processing_mode": "stateful_streaming",
                "ns_precomputed_replay": False,
                "ns_stateful": True,
                "ns_delay_samples": DELAY_SAMPLES,
                "ns_model_latency_samples": MODEL_LATENCY_SAMPLES,
                "ns_stream_startup_delay_samples": STARTUP_DELAY_SAMPLES,
                "ns_execution_mode": "online_callback",
                "ns_worker_transport": "persistent_f32le_v1",
                "ns_realtime_verified": False,
                "ns_stream_chunk_samples": FRAME_SAMPLES,
            })
            timing = receipt["timing"]
            execution = {
                **identity,
                "ns_cli_sha256": cli_sha,
                "ns_external_backend": "external_stream",
                "ns_boundary_input_sha256": receipt["input_payload_sha256"],
                "ns_output_sha256": receipt["output_payload_sha256"],
                "ns_input_payload_sha256": receipt["quantized_input_pcm16_sha256"],
                "ns_inference_calls": receipt["model_inferences"],
                "ns_model_inference_calls": receipt["model_inferences"],
                "ns_model_inference_s": receipt["inference_seconds"],
                "ns_model_process_s": receipt["model_process_seconds"],
                "ns_stream_callback_frames": callbacks,
                "ns_stream_chunk_samples": FRAME_SAMPLES,
                "ns_stream_delay_frames": DELAY_FRAMES,
                "ns_stream_model_latency_samples": MODEL_LATENCY_SAMPLES,
                "ns_stream_startup_delay_samples": STARTUP_DELAY_SAMPLES,
                "ns_stream_model_output_frames": receipt["model_output_frames"],
                "ns_stream_pending_output_frames": receipt["pending_output_frames"],
                "ns_precomputed_replay": False,
                "ns_stateful": True,
                "ns_execution_mode": "online_callback",
                "ns_worker_transport": "persistent_f32le_v1",
                "ns_state_resets": receipt["live_state_reset_calls"],
                "ns_realtime_verified": False,
                "ns_realtime_scope": "file inputs with native callback transport; device audio not connected",
                "ns_stream_state_bytes": receipt["state_bytes"],
                "ns_stream_callback_max_ms": receipt["callback_max_ms"],
                "ns_stream_callback_p50_upper_ms": receipt["callback_p50_upper_ms"],
                "ns_stream_callback_p95_upper_ms": receipt["callback_p95_upper_ms"],
                "ns_stream_callback_p99_upper_ms": receipt["callback_p99_upper_ms"],
                "ns_stream_callback_over_10ms": receipt["callback_over_10ms"],
                "ns_stream_quantize_hash_preprocess_s": timing["quantize_hash_preprocess"]["total_seconds"],
                "ns_stream_model_process_s": timing["model_process"]["total_seconds"],
                "ns_stream_postprocess_write_s": timing["postprocess_write"]["total_seconds"],
                "ns_stream_read_wait_s": timing["read_wait"]["total_seconds"],
                "ns_stream_startup_load_s": receipt["startup_load_seconds"],
                "ns_stream_warmup_s": receipt["warmup_seconds"],
                "ns_stream_startup_to_ready_s": receipt["startup_to_ready_seconds"],
                "ns_stream_startup_timing_scope": receipt["startup_timing_scope"],
                "ns_stream_timing_clock": receipt["timing_clock"],
                "ns_stream_timing_scope": receipt["timing_scope"],
                "ns_stream_timing_sample_policy": receipt["timing_sample_policy"],
                "ns_stream_timing_samples": receipt["timing_samples"],
                "ns_stream_timing_sample_capacity": receipt["timing_sample_capacity"],
                "ns_stream_timing": timing,
                "ns_worker_source_sha256": receipt["worker_source_sha256"],
                "ns_worker_helper_source_sha256": receipt["worker_helper_source_sha256"],
                "ns_worker_command_sha256": receipt["worker_command_sha256"],
                "ns_worker_command": receipt["worker_command"],
                "ns_worker_python_executable": receipt["python_executable"],
                "ns_stream_model_rtf": receipt["rtf_model"],
                "ns_stream_peak_rss_bytes": receipt["peak_rss_bytes"],
            }
            report.update(execution)
            report.update({
                "ns_backend": "dpdfnet2_48k_hr",
                "ns_backend_requested": "dpdfnet2_48k_hr",
                "ns_backend_effective": "dpdfnet2_48k_hr",
                "ns_processing_mode": "stateful_streaming",
                "ns_delay_samples": DELAY_SAMPLES,
                "ns_external_backend": "external_stream",
                "ns_execution": execution,
            })
            errors = routes.validate_ns_route_report(report, configuration)
            if errors:
                raise ValueError("48 kHz HR stream report validation failed: " + "; ".join(errors))
            routes._write_json(report_path, report)
            stages = json.loads(stages_path.read_text(encoding="utf-8"))
            stages["ns_model"] = execution
            routes._write_json(stages_path, stages)
            routes._publish_runtime_receipt(prefix, "dpdfnet2_48k_hr", {
                "schema_version": 2,
                "backend": "dpdfnet2_48k_hr",
                "execution": execution,
                "model_receipt": str(worker_receipt),
                "command": online,
            })
            return routes._append_stdout(
                completed,
                "NS model: dpdfnet2_48k_hr stateful live 480-sample callbacks",
            )
        except Exception as error:
            try:
                routes._write_json(model_dir / "failure.json", {
                    "schema_version": 1,
                    "backend": "dpdfnet2_48k_hr",
                    "error": {"type": type(error).__name__, "message": str(error)},
                })
            except Exception:
                pass
            raise
        finally:
            if worker.stdin and not worker.stdin.closed:
                worker.stdin.close()
            if worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait(timeout=5)
            if worker.stdout:
                worker.stdout.close()
