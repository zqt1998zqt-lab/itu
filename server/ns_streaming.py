"""Control-thread lifecycle for the native VCP's persistent NS frame worker."""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import select
import stat
import subprocess
import time


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FRAME_SAMPLES = 160
_TIMING_FIELDS = (
    "read_wait",
    "quantize_hash_preprocess",
    "model_process",
    "postprocess_write",
    "callback",
)


def _finite_nonnegative(value) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
    )


def _validate_worker_pipe_fd(fd: int, *, writable: bool) -> None:
    status = os.fstat(fd)
    if not stat.S_ISFIFO(status.st_mode):
        raise ValueError("stream worker parent fd must be a POSIX pipe")
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    mode = flags & os.O_ACCMODE
    allowed = (os.O_WRONLY, os.O_RDWR) if writable else (os.O_RDONLY, os.O_RDWR)
    if mode not in allowed:
        raise ValueError("stream worker parent fd access mode is invalid")


def _command_sha256(command: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(command, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _ready(stream, deadline: float) -> dict:
    line = bytearray()
    while len(line) < 4096:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("CoFi-Lite stream worker initialization timed out")
        try:
            selected = select.select([stream], [], [], remaining)[0]
        except InterruptedError:
            continue
        if not selected:
            raise TimeoutError("CoFi-Lite stream worker initialization timed out")
        try:
            character = os.read(stream.fileno(), 1)
        except InterruptedError:
            continue
        if not character:
            raise ValueError("CoFi-Lite stream worker exited before ready; inspect stream-worker.log")
        line.extend(character)
        if character == b"\n":
            return json.loads(line)
    raise ValueError("invalid streaming NS readiness message")


def _validate_timing_receipt(receipt: dict, callbacks: int) -> None:
    timing = receipt.get("timing")
    if not isinstance(timing, dict) or set(timing) != set(_TIMING_FIELDS):
        raise ValueError("stream timing receipt is incomplete")
    for field in _TIMING_FIELDS:
        summary = timing[field]
        if not isinstance(summary, dict):
            raise ValueError(f"stream timing summary is invalid: {field}")
        required = {
            "count", "total_seconds", "p50_upper_ms", "p95_upper_ms",
            "p99_upper_ms", "max_ms", "over_10ms",
        }
        if set(summary) != required or summary.get("count") != callbacks:
            raise ValueError(f"stream timing sample count mismatch: {field}")
        for key in required - {"count", "over_10ms"}:
            if not _finite_nonnegative(summary.get(key)):
                raise ValueError(f"stream timing value is invalid: {field}.{key}")
        if (
            isinstance(summary.get("over_10ms"), bool)
            or not isinstance(summary.get("over_10ms"), int)
            or summary["over_10ms"] < 0
        ):
            raise ValueError(f"stream over-10ms count is invalid: {field}")
        if not (
            summary["p50_upper_ms"] <= summary["p95_upper_ms"]
            <= summary["p99_upper_ms"] <= summary["max_ms"] + 0.1000001
        ):
            raise ValueError(f"stream timing quantiles are inconsistent: {field}")
    samples = receipt.get("timing_samples")
    capacity = receipt.get("timing_sample_capacity")
    if not isinstance(receipt.get("timing_sample_policy"), str) or not receipt["timing_sample_policy"]:
        raise ValueError("stream timing sample policy is invalid")
    if (
        isinstance(capacity, bool)
        or not isinstance(capacity, int)
        or capacity < 1
        or capacity > 64
        or not isinstance(samples, list)
        or len(samples) != min(callbacks, capacity)
    ):
        raise ValueError("stream timing sample buffer is invalid or unbounded")
    sample_keys = {
        "frame_index", "read_wait_ms", "preprocess_ms", "model_process_ms",
        "postprocess_write_ms", "callback_ms",
    }
    for sample in samples:
        if not isinstance(sample, dict) or set(sample) != sample_keys:
            raise ValueError("stream timing sample shape is invalid")
        if not isinstance(sample["frame_index"], int) or isinstance(sample["frame_index"], bool):
            raise ValueError("stream timing sample frame index is invalid")
        if not all(_finite_nonnegative(sample[key]) for key in sample_keys - {"frame_index"}):
            raise ValueError("stream timing sample value is invalid")


def _validate_worker_identity(receipt: dict, callbacks: int) -> None:
    samples = receipt.get("samples")
    hop_samples = receipt.get("model_hop_samples")
    ready_threshold = receipt.get("model_ready_threshold_samples")
    warmup_calls = receipt.get("warmup_calls")
    geometry = (samples, hop_samples, ready_threshold, warmup_calls)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in geometry
    ) or samples != callbacks * _FRAME_SAMPLES:
        raise ValueError("stream model inference geometry is invalid")
    expected_inferences = (
        0 if samples < ready_threshold else samples // hop_samples
    )
    expected_warmup = (
        0
        if warmup_calls * _FRAME_SAMPLES < ready_threshold
        else warmup_calls * _FRAME_SAMPLES // hop_samples
    )
    if (
        receipt.get("expected_model_inferences") != expected_inferences
        or receipt.get("model_inferences") != expected_inferences
    ):
        raise ValueError("stream model inference count does not match hop schedule")
    if receipt.get("model_initializations") != 1:
        raise ValueError("stream worker must initialize one model per session")
    if receipt.get("live_stream_initializations") != 1:
        raise ValueError("stream worker must initialize one live stateful stream")
    if receipt.get("live_state_reset_calls") != 1:
        raise ValueError("stream worker live state must reset only at construction")
    if receipt.get("warmup_inferences") != expected_warmup:
        raise ValueError("stream worker warmup inference count is invalid")
    for field in ("startup_load_seconds", "warmup_seconds", "startup_to_ready_seconds"):
        if not _finite_nonnegative(receipt.get(field)):
            raise ValueError(f"stream startup timing is invalid: {field}")
    for field in ("startup_timing_scope", "timing_clock", "timing_scope"):
        if not isinstance(receipt.get(field), str) or not receipt[field]:
            raise ValueError(f"stream timing metadata is invalid: {field}")
    for field in ("worker_source_sha256", "worker_command_sha256"):
        value = receipt.get(field)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError(f"stream worker identity is invalid: {field}")
    command = receipt.get("worker_command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(part, str) or not part for part in command)
    ):
        raise ValueError("stream worker command identity is invalid")
    _validate_timing_receipt(receipt, callbacks)


def execute_stream_command(command, *, configuration: dict, timeout: float, **kwargs):
    from server import ns_routes as routes

    spec = routes.NS_ROUTES["cofilite"]
    prefix, original = routes._external_command(command)
    routes._validate_mic_header(original, spec.sample_rate)
    routes._verify_route_assets("cofilite")
    cli = Path(original[0]).resolve()
    cli_sha = routes._sha256_file(cli)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    # Each caller uses a unique output prefix. Worker evidence is append/new
    # only and survives both success and failure.
    worker_receipt = routes._artifact(prefix, ".ns-stream.json")
    worker_log = routes._artifact(prefix, ".ns-stream.log")
    deadline = time.monotonic() + timeout
    environment = dict(os.environ)
    environment.update({name: "1" for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    )})
    worker_command = [routes._model_python(), "-m", "server.ns_stream_worker", "--receipt", str(worker_receipt)]
    worker_source_path = Path(__file__).resolve().with_name("ns_stream_worker.py")
    worker_source_sha256 = routes._sha256_file(worker_source_path)
    worker_command_sha256 = _command_sha256(worker_command)
    with worker_log.open("xb") as stderr:
        worker = subprocess.Popen(worker_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=stderr, env=environment, cwd=routes.ROOT, bufsize=0)
        try:
            tx_fd, rx_fd = worker.stdin.fileno(), worker.stdout.fileno()
            if tx_fd == rx_fd:
                raise ValueError("stream worker parent fds must be distinct")
            _validate_worker_pipe_fd(tx_fd, writable=True)
            _validate_worker_pipe_fd(rx_fd, writable=False)
            ready = _ready(worker.stdout, min(deadline, time.monotonic() + 45))
            expected = {
                "protocol": "itu-ns-f32le-v1", "backend": "cofilite",
                "sample_rate": spec.sample_rate,
                "frame_samples": spec.frame_samples, "delay_samples": spec.delay_samples,
                "model_sha256": spec.model_sha256, "frontend_sha256": spec.frontend_sha256,
                "processing_mode": spec.processing_mode,
            }
            if getattr(spec, "frame_executor", None) is not None:
                expected.update({
                    "frame_executor": spec.frame_executor,
                    "frame_model_sha256": spec.frame_model_sha256,
                    "frame_execution_provider": "CPUExecutionProvider",
                })
            if ready != expected:
                raise ValueError("stream worker ready identity does not match the selected route")
            online = [*original, "--ns-stream-input-fd", str(tx_fd),
                      "--ns-stream-output-fd", str(rx_fd),
                      "--ns-stream-delay-samples", str(spec.delay_samples)]
            if "--diagnostic-stages" not in online:
                online.append("--diagnostic-stages")
            options = dict(kwargs)
            options["pass_fds"] = tuple(set(options.get("pass_fds", ())) | {tx_fd, rx_fd})
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("CoFi-Lite stream initialization exceeded execution timeout")
            completed = subprocess.run(online, timeout=remaining, **options)
            worker.stdin.close()
            if completed.returncode != 0:
                raise routes._process_error("CoFi-Lite online VCP", completed)
            worker.wait(timeout=max(0.1, min(30, deadline - time.monotonic())))
            if worker.returncode != 0 or not worker_receipt.is_file():
                raise ValueError("CoFi-Lite stream worker failed; inspect " + str(worker_log))
            receipt = json.loads(worker_receipt.read_text(encoding="utf-8"))
            report_path = routes._artifact(prefix, ".json")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            for key, expected_value in {
                "ns_backend": "external_stream", "ns_external_stream_enabled": True,
                "ns_external_pcm_enabled": False, "ns_external_fallback_frames": 0,
                "ns_external_delay_samples": spec.delay_samples,
            }.items():
                if report.get(key) != expected_value:
                    raise ValueError(f"native streaming NS report mismatch: {key}")
            callbacks = report.get("ns_external_applied_frames")
            if (not routes._positive_int(callbacks) or callbacks != receipt.get("callback_frames")
                    or not routes._positive_int(receipt.get("model_inferences"))
                    or receipt.get("stateful") is not True
                    or receipt.get("precomputed_replay") is not False):
                raise ValueError("stream model callback counts or live state proof are invalid")
            _validate_worker_identity(receipt, callbacks)
            if getattr(spec, "frame_executor", None) is not None:
                frame_identity = receipt.get("frame_executor")
                if not isinstance(frame_identity, dict):
                    raise ValueError("stream worker omitted its frame executor identity")
                expected_frame = {
                    "mode": spec.frame_executor,
                    "provider": "CPUExecutionProvider",
                    "model_sha256": spec.frame_model_sha256,
                    "weights_sha256": spec.model_sha256,
                    "intra_op_num_threads": 1,
                    "inter_op_num_threads": 1,
                    "state_count": 18,
                    "session_load_count": 1,
                }
                for key, expected in expected_frame.items():
                    actual = frame_identity.get(key)
                    if type(actual) is not type(expected) or actual != expected:
                        raise ValueError(f"stream frame executor identity mismatch: {key}")
            if (
                receipt.get("worker_source_sha256") != worker_source_sha256
                or routes._sha256_file(worker_source_path) != worker_source_sha256
            ):
                raise ValueError("stream worker source identity changed during processing")
            if receipt.get("worker_command_sha256") != worker_command_sha256:
                raise ValueError("stream worker command identity does not match the launched command")
            if routes._sha256_file(cli) != cli_sha:
                raise ValueError("stream VCP executable identity changed during processing")
            routes._verify_route_assets("cofilite")
            identity = routes.ns_route_identity(configuration)
            timing = receipt["timing"]
            execution = {
                **identity, "ns_cli_sha256": cli_sha, "ns_external_backend": "external_stream",
                "ns_boundary_input_sha256": receipt["input_payload_sha256"],
                "ns_output_sha256": receipt["output_payload_sha256"],
                "ns_input_payload_sha256": receipt["quantized_input_pcm16_sha256"],
                "ns_inference_calls": receipt["model_inferences"],
                "ns_model_inference_calls": receipt["model_inferences"],
                "ns_model_inference_s": receipt["inference_seconds"],
                "ns_stream_callback_frames": callbacks,
                "ns_stream_chunk_samples": spec.frame_samples,
                "ns_precomputed_replay": False, "ns_stateful": True,
                "ns_execution_mode": "online_callback", "ns_worker_transport": "persistent_f32le_v1",
                "ns_state_resets": receipt["live_state_reset_calls"],
                "ns_realtime_verified": False,
                "ns_realtime_scope": "file inputs with online callbacks; device audio not connected",
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
                "ns_worker_command_sha256": receipt["worker_command_sha256"],
                "ns_worker_command": receipt["worker_command"],
                "ns_worker_python_executable": receipt["python_executable"],
                "ns_stream_model_rtf": receipt["rtf_model"],
                "ns_stream_peak_rss_bytes": receipt["peak_rss_bytes"],
            }
            report.update(execution)
            if getattr(spec, "frame_executor", None) is not None:
                execution["ns_frame_executor_identity"] = receipt["frame_executor"]
                report["ns_frame_executor_identity"] = receipt["frame_executor"]
            report["ns_backend"] = "cofilite"
            report["ns_execution"] = execution
            errors = routes.validate_ns_route_report(report, configuration)
            if errors:
                raise ValueError("stream NS route report validation failed: " + "; ".join(errors))
            routes._write_json(report_path, report)
            stages_path = routes._artifact(prefix, ".stages.json")
            stages = json.loads(stages_path.read_text(encoding="utf-8"))
            stages["ns_model"] = execution
            routes._write_json(stages_path, stages)
            routes._publish_runtime_receipt(prefix, "cofilite", {
                "schema_version": 2, "backend": "cofilite", "execution": execution,
                "model_receipt": str(worker_receipt), "command": online,
            })
            return routes._append_stdout(completed, "NS model: cofilite stateful live 160-sample callbacks")
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
