"""Persistent CoFi-Lite stream worker; binary FLOAT32 frames after a ready line.

One process owns one stream and one set of recurrent/convolution/OLA state.
The native VCP callback supplies the actual NS input and consumes the output
before advancing the next callback. No utterance capture or replay is used.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FRAME_SAMPLES = 160
FRAME_BYTES = FRAME_SAMPLES * np.dtype("<f4").itemsize
WARMUP_FRAMES = 8
TIMING_SAMPLE_CAPACITY = 16
_HISTOGRAM_BUCKETS = 1001
_MODULE_STARTED = time.perf_counter()


class LatencyStats:
    """Bounded histogram plus exact max for one per-frame phase."""

    __slots__ = ("_histogram", "count", "maximum", "total", "over_10ms")

    def __init__(self) -> None:
        self._histogram = np.zeros(_HISTOGRAM_BUCKETS, dtype=np.int64)
        self.count = 0
        self.maximum = 0.0
        self.total = 0.0
        self.over_10ms = 0

    def observe(self, seconds: float) -> None:
        value = float(seconds)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("stream timing sample must be finite and non-negative")
        self.count += 1
        self.maximum = max(self.maximum, value)
        self.total += value
        self.over_10ms += int(value > 0.010)
        self._histogram[min(_HISTOGRAM_BUCKETS - 1, int(value * 10_000))] += 1

    def _quantile_upper_ms(self, quantile: float) -> float:
        if not self.count:
            raise ValueError("stream timing quantile requested without samples")
        target = int(math.ceil(self.count * quantile))
        bucket = int(np.searchsorted(np.cumsum(self._histogram), target))
        return (min(_HISTOGRAM_BUCKETS - 1, bucket) + 1) * 0.1

    def summary(self) -> dict:
        if not self.count:
            raise ValueError("stream timing summary requested without samples")
        return {
            "count": self.count,
            "total_seconds": self.total,
            "p50_upper_ms": self._quantile_upper_ms(0.50),
            "p95_upper_ms": self._quantile_upper_ms(0.95),
            "p99_upper_ms": self._quantile_upper_ms(0.99),
            "max_ms": self.maximum * 1000.0,
            "over_10ms": self.over_10ms,
        }


class FrameProcessor:
    """Preallocated FLOAT32 read, exact PCM16 quantization, and hash views."""

    __slots__ = (
        "input_wire",
        "input_bytes",
        "values",
        "finite",
        "scaled",
        "magnitude",
        "rounded",
        "pcm16",
        "pcm16_bytes",
        "quantized",
    )

    def __init__(self) -> None:
        if sys.byteorder != "little":
            raise RuntimeError("streaming NS FLOAT32 protocol requires a little-endian host")
        self.input_wire = bytearray(FRAME_BYTES)
        self.input_bytes = memoryview(self.input_wire)
        self.values = np.frombuffer(self.input_wire, dtype=np.dtype("<f4"))
        if self.values.shape != (FRAME_SAMPLES,) or not self.values.dtype.isnative:
            raise RuntimeError("streaming NS FLOAT32 frame view is invalid")
        self.finite = np.empty(FRAME_SAMPLES, dtype=np.bool_)
        self.scaled = np.empty(FRAME_SAMPLES, dtype=np.float32)
        self.magnitude = np.empty(FRAME_SAMPLES, dtype=np.float32)
        self.rounded = np.empty(FRAME_SAMPLES, dtype=np.float32)
        self.pcm16 = np.empty(FRAME_SAMPLES, dtype="<i2")
        self.pcm16_bytes = memoryview(self.pcm16).cast("B")
        self.quantized = np.empty(FRAME_SAMPLES, dtype=np.float32)

    def read(self, fd: int) -> bool:
        return _exact_read_into(fd, self.input_wire)

    def quantize(self) -> np.ndarray:
        if not bool(np.isfinite(self.values, out=self.finite).all()):
            raise ValueError("stream input contains NaN or Inf")
        # Clipping before the power-of-two scale preserves the frozen
        # ns_routes._quantize_pcm16_lround result while avoiding float64 and
        # overflow warnings for finite out-of-range input.
        np.clip(
            self.values,
            np.float32(-1.0),
            np.float32(32767.0 / 32768.0),
            out=self.scaled,
        )
        np.multiply(self.scaled, np.float32(32768.0), out=self.scaled)
        np.abs(self.scaled, out=self.magnitude)
        np.add(self.magnitude, np.float32(0.5), out=self.rounded)
        np.floor(self.rounded, out=self.rounded)
        np.copysign(self.rounded, self.scaled, out=self.rounded)
        np.copyto(self.pcm16, self.rounded, casting="unsafe")
        np.divide(self.pcm16, np.float32(32768.0), out=self.quantized)
        return self.quantized


def _exact_read_into(fd: int, buffer) -> bool:
    """Fill ``buffer`` from a blocking pipe, distinguishing EOF and truncation."""
    view = memoryview(buffer).cast("B")
    offset = 0
    while offset < len(view):
        try:
            received = os.readv(fd, [view[offset:]])
        except InterruptedError:
            continue
        if received == 0:
            if offset:
                raise ValueError("truncated streaming NS FLOAT32 frame")
            return False
        offset += received
    return True


def _exact_write(fd: int, data) -> None:
    """Write all bytes, preserving short-write and EINTR handling."""
    view = memoryview(data).cast("B")
    offset = 0
    while offset < len(view):
        try:
            written = os.write(fd, view[offset:])
        except InterruptedError:
            continue
        except OSError as error:
            raise ValueError(f"stream output write failed: {error}") from error
        if written == 0:
            raise ValueError("stream output write returned zero bytes")
        offset += written


def _expected_model_inferences(
    samples: int,
    *,
    hop_samples: int,
    ready_threshold_samples: int,
) -> int:
    if samples < 0 or hop_samples < 1 or ready_threshold_samples < 1:
        raise ValueError("stream inference geometry is invalid")
    return 0 if samples < ready_threshold_samples else samples // hop_samples


def _exact_read(stream, size: int) -> bytes:
    chunks = bytearray(size)
    offset = 0
    while offset < size:
        block = stream.read(size - offset)
        if not block:
            if offset:
                raise ValueError("truncated streaming NS FLOAT32 frame")
            return b""
        chunks[offset : offset + len(block)] = block
        offset += len(block)
    return bytes(chunks)


def _validate_worker_fd(fd: int, *, writable: bool) -> None:
    import fcntl
    import stat

    if isinstance(fd, bool) or not isinstance(fd, int) or fd < 0:
        raise ValueError("stream worker fd must be a non-negative integer")
    status = os.fstat(fd)
    if not stat.S_ISFIFO(status.st_mode):
        raise ValueError("stream worker fd must be a POSIX pipe")
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    mode = flags & os.O_ACCMODE
    allowed = (os.O_WRONLY, os.O_RDWR) if writable else (os.O_RDONLY, os.O_RDWR)
    if mode not in allowed:
        raise ValueError("stream worker fd access mode is incompatible with its direction")


def _load_cofilite_backend(spec, verify_assets) -> tuple:
    # Torch remains outside the long-lived FastAPI process. Importing here lets
    # the receipt measure the complete import, asset verification, and weight
    # load path before the readiness handshake.
    import torch
    from server.cofilite_stream import CoFiLiteStream, load_model

    verify_assets()
    model = load_model("cpu", threads=1)
    if getattr(spec, "frame_executor", None) == "onnxruntime_cpu":
        from server.cofilite_onnx import CoFiLiteOnnxStream, get_onnx_frame_executor

        executor = get_onnx_frame_executor(
            model, spec.frame_model_path,
            expected_model_sha256=spec.frame_model_sha256,
            expected_weights_sha256=spec.model_sha256,
        )
        return model, lambda: CoFiLiteOnnxStream(
            model=model, model_path=spec.frame_model_path,
            expected_model_sha256=spec.frame_model_sha256,
            expected_weights_sha256=spec.model_sha256,
            frame_executor=executor,
        ), torch.inference_mode
    return model, lambda: CoFiLiteStream(model=model), torch.inference_mode


def run(
    receipt_path: Path,
    *,
    input_fd: int | None = None,
    output_fd: int | None = None,
    route_spec=None,
    load_backend=None,
    verify_assets=None,
    command_argv: list[str] | None = None,
) -> dict:
    run_started = time.perf_counter()
    if route_spec is None:
        from server.ns_routes import NS_ROUTES, _verify_route_assets

        spec = NS_ROUTES["cofilite"]
        verify_assets = lambda: _verify_route_assets("cofilite")
    else:
        spec = route_spec
        verify_assets = verify_assets or (lambda: None)
    input_fd = sys.stdin.fileno() if input_fd is None else input_fd
    output_fd = sys.stdout.fileno() if output_fd is None else output_fd
    if input_fd == output_fd:
        raise ValueError("stream worker input and output fds must differ")
    _validate_worker_fd(input_fd, writable=False)
    _validate_worker_fd(output_fd, writable=True)

    load_started = time.perf_counter()
    backend = load_backend or (lambda: _load_cofilite_backend(spec, verify_assets))
    model, stream_factory, inference_mode = backend()
    startup_load_seconds = time.perf_counter() - load_started
    live_factory_calls = 0
    with inference_mode():
        warmup_started = time.perf_counter()
        warmup = stream_factory()
        live_factory_calls += 1
        if getattr(warmup, "model", None) is not model:
            raise ValueError("warmup stream did not receive the one loaded model")
        hop_samples = int(getattr(warmup, "hop_samples", 0))
        fft_size = int(getattr(warmup, "fft_size", 0))
        ready_threshold_samples = fft_size // 2 + 1 if fft_size >= 2 else 0
        if hop_samples < 1 or ready_threshold_samples < 1:
            raise ValueError("warmup stream hop or FFT geometry is invalid")
        for _ in range(WARMUP_FRAMES):
            warmup.process(np.zeros(FRAME_SAMPLES, dtype=np.float32))
        warmup_seconds = time.perf_counter() - warmup_started
        expected_warmup_inferences = _expected_model_inferences(
            WARMUP_FRAMES * FRAME_SAMPLES,
            hop_samples=hop_samples,
            ready_threshold_samples=ready_threshold_samples,
        )
        if getattr(warmup, "inferences", None) != expected_warmup_inferences:
            raise ValueError("stream warmup inference count is invalid")
        # The live stream starts after warmup with its own zeroed state. No
        # warmup samples or recurrent history can enter the caller timeline.
        runtime = stream_factory()
        live_factory_calls += 1
        if getattr(runtime, "model", None) is not model:
            raise ValueError("live stream did not receive the one loaded model")
        if runtime.frame_samples != FRAME_SAMPLES or runtime.delay_samples != spec.delay_samples:
            raise ValueError("stream model geometry does not match the selected route")
        if runtime.hop_samples != hop_samples or runtime.fft_size != fft_size:
            raise ValueError("live and warmup stream geometries differ")
        ready = {
            "protocol": "itu-ns-f32le-v1", "backend": "cofilite",
            "model_sha256": spec.model_sha256, "frontend_sha256": spec.frontend_sha256,
            "frame_samples": FRAME_SAMPLES, "sample_rate": 16000,
            "delay_samples": runtime.delay_samples, "processing_mode": spec.processing_mode,
        }
        if getattr(spec, "frame_executor", None) is not None:
            frame_identity = dict(getattr(runtime, "_executor_info", {}))
            for key, expected in {
                "mode": spec.frame_executor,
                "model_sha256": spec.frame_model_sha256,
                "weights_sha256": spec.model_sha256,
                "provider": "CPUExecutionProvider",
                "intra_op_num_threads": 1,
                "inter_op_num_threads": 1,
                "state_count": 18,
                "session_load_count": 1,
            }.items():
                actual = frame_identity.get(key)
                if type(actual) is not type(expected) or actual != expected:
                    raise ValueError(f"stream frame executor not ready: {key}")
            ready.update({
                "frame_executor": spec.frame_executor,
                "frame_model_sha256": spec.frame_model_sha256,
                "frame_execution_provider": "CPUExecutionProvider",
            })
        startup_to_ready_seconds = time.perf_counter() - _MODULE_STARTED
        _exact_write(output_fd, json.dumps(ready).encode("ascii") + b"\n")

        processor = FrameProcessor()
        read_stats = LatencyStats()
        preprocess_stats = LatencyStats()
        model_stats = LatencyStats()
        postprocess_stats = LatencyStats()
        callback_stats = LatencyStats()
        timing_samples = deque(maxlen=TIMING_SAMPLE_CAPACITY)
        input_hash = hashlib.sha256()
        quantized_hash = hashlib.sha256()
        output_hash = hashlib.sha256()
        callbacks = 0
        started = time.perf_counter()
        while True:
            read_started = time.perf_counter()
            if not processor.read(input_fd):
                break
            read_seconds = time.perf_counter() - read_started
            frame_started = time.perf_counter()

            preprocess_started = frame_started
            quantized = processor.quantize()
            input_hash.update(processor.input_bytes)
            quantized_hash.update(processor.pcm16_bytes)
            preprocess_seconds = time.perf_counter() - preprocess_started

            model_started = time.perf_counter()
            # The frame processor has already checked shape, dtype, finiteness,
            # and contiguous input. Avoid repeating those NumPy passes inside
            # CoFiLiteStream.process while preserving its exact model body.
            enhanced = runtime._consume_validated(quantized)
            model_seconds = time.perf_counter() - model_started

            postprocess_started = time.perf_counter()
            if (
                not isinstance(enhanced, np.ndarray)
                or enhanced.shape != (FRAME_SAMPLES,)
                or enhanced.dtype != np.float32
                or not enhanced.flags.c_contiguous
                or not bool(np.isfinite(enhanced, out=processor.finite).all())
            ):
                raise ValueError("stream model must return 160 finite contiguous float32 samples")
            output_bytes = memoryview(enhanced).cast("B")
            output_hash.update(output_bytes)
            _exact_write(output_fd, output_bytes)
            postprocess_seconds = time.perf_counter() - postprocess_started

            callback_seconds = time.perf_counter() - frame_started
            callbacks += 1
            read_stats.observe(read_seconds)
            preprocess_stats.observe(preprocess_seconds)
            model_stats.observe(model_seconds)
            postprocess_stats.observe(postprocess_seconds)
            callback_stats.observe(callback_seconds)
            timing_samples.append({
                "frame_index": callbacks,
                "read_wait_ms": read_seconds * 1000.0,
                "preprocess_ms": preprocess_seconds * 1000.0,
                "model_process_ms": model_seconds * 1000.0,
                "postprocess_write_ms": postprocess_seconds * 1000.0,
                "callback_ms": callback_seconds * 1000.0,
            })

    if not callbacks:
        raise ValueError("stream ended without a single VCP callback")
    expected_model_inferences = _expected_model_inferences(
        callbacks * FRAME_SAMPLES,
        hop_samples=hop_samples,
        ready_threshold_samples=ready_threshold_samples,
    )
    if runtime.inferences != expected_model_inferences:
        raise ValueError("stream model inference count does not match hop schedule")
    verify_assets()
    read_timing = read_stats.summary()
    preprocess_timing = preprocess_stats.summary()
    model_timing = model_stats.summary()
    postprocess_timing = postprocess_stats.summary()
    callback_timing = callback_stats.summary()
    source_path = Path(__file__).resolve()
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    command = list(
        sys.orig_argv
        if command_argv is None and hasattr(sys, "orig_argv")
        else (sys.argv if command_argv is None else command_argv)
    )
    command_sha256 = hashlib.sha256(
        json.dumps(command, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    receipt = {
        **ready, "schema_version": 2, "stateful": True,
        "offline_whole_utterance": False, "precomputed_replay": False,
        "callback_frames": callbacks, "samples": callbacks * FRAME_SAMPLES,
        "input_payload_sha256": input_hash.hexdigest(),
        "quantized_input_pcm16_sha256": quantized_hash.hexdigest(),
        "output_payload_sha256": output_hash.hexdigest(),
        "model_inferences": runtime.inferences,
        "frame_executor": dict(getattr(runtime, "_executor_info", {})),
        "expected_model_inferences": expected_model_inferences,
        "model_hop_samples": hop_samples,
        "model_ready_threshold_samples": ready_threshold_samples,
        "model_initializations": 1,
        "live_stream_initializations": live_factory_calls - 1,
        "live_state_reset_calls": 1,
        "warmup_calls": WARMUP_FRAMES,
        "warmup_inferences": warmup.inferences,
        "state_bytes": runtime.state_bytes,
        "inference_seconds": model_timing["total_seconds"],
        "model_process_seconds": model_timing["total_seconds"],
        "rtf_model": model_timing["total_seconds"] / (callbacks * 0.010),
        "callback_seconds": callback_timing["total_seconds"],
        "callback_max_ms": callback_timing["max_ms"],
        "callback_p50_upper_ms": callback_timing["p50_upper_ms"],
        "callback_p95_upper_ms": callback_timing["p95_upper_ms"],
        "callback_p99_upper_ms": callback_timing["p99_upper_ms"],
        "callback_over_10ms": callback_timing["over_10ms"],
        "startup_load_seconds": startup_load_seconds,
        "warmup_seconds": warmup_seconds,
        "startup_to_ready_seconds": startup_to_ready_seconds,
        "startup_timing_scope": (
            "from ns_stream_worker module import through ready-line construction; "
            "interpreter startup is excluded"
        ),
        "timing_clock": "time.perf_counter monotonic seconds",
        "timing_scope": (
            "worker service time: complete FLOAT32 frame read through output write; "
            "read_wait is reported separately and callback excludes C++ encode/decode"
        ),
        "timing_sample_policy": (
            "trailing bounded window; frame_index is the 1-based worker callback ordinal"
        ),
        "timing_sample_capacity": TIMING_SAMPLE_CAPACITY,
        "timing_samples": list(timing_samples),
        "timing": {
            "read_wait": read_timing,
            "quantize_hash_preprocess": preprocess_timing,
            "model_process": model_timing,
            "postprocess_write": postprocess_timing,
            "callback": callback_timing,
        },
        "worker_source_sha256": source_sha256,
        "worker_command": command,
        "worker_command_sha256": command_sha256,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "elapsed_seconds": time.perf_counter() - started,
        "total_run_seconds": time.perf_counter() - run_started,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "pcm16_quantization": "once_effective",
        "clean_reference_read": False, "fallback_frames": 0,
        "eos": "native VCP supplies zero drain callbacks including its declared fixed delay",
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return receipt


def main() -> int:
    os.environ.update({name: "1" for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    )})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
