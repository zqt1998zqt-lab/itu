"""Persistent 48 kHz DPDFNet2 HR worker for native VCP callbacks.

The worker is deliberately separate from the 16 kHz CoFi-Lite worker.  It
consumes exactly 480 little-endian FLOAT32 samples per callback, keeps the
DPDFNet2 recurrent/OLA state alive for the whole session, and emits exactly
480 samples per callback.  The model's measured 1920-sample physical latency
is represented by four leading silence frames, matching the offline replay
contract without capturing or replaying a WAV file.
"""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import nullcontext
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

from server.ns_stream_worker import (  # noqa: E402
    LatencyStats,
    _exact_read_into,
    _exact_write,
    _validate_worker_fd,
)


FRAME_SAMPLES = 480
FRAME_BYTES = FRAME_SAMPLES * np.dtype("<f4").itemsize
SAMPLE_RATE = 48_000
WINDOW_SAMPLES = 960
HOP_SAMPLES = 480
MODEL_LATENCY_SAMPLES = 1_920
MODEL_LATENCY_FRAMES = MODEL_LATENCY_SAMPLES // FRAME_SAMPLES
STARTUP_DELAY_SAMPLES = FRAME_SAMPLES
DELAY_SAMPLES = MODEL_LATENCY_SAMPLES + STARTUP_DELAY_SAMPLES
DELAY_FRAMES = DELAY_SAMPLES // FRAME_SAMPLES
WARMUP_FRAMES = 0
TIMING_SAMPLE_CAPACITY = 16
_MODULE_STARTED = time.perf_counter()


class FrameProcessor:
    """Preallocated FLOAT32 reader and exact PCM16-lround quantizer."""

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
            raise RuntimeError("48 kHz streaming NS FLOAT32 frame view is invalid")
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


def _load_backend(spec, verify_assets):
    from server import dpdfnet2_48k_hr

    verify_assets()
    runtime = dpdfnet2_48k_hr.load_runtime(spec.model_path)
    if runtime.model_sha256 != spec.model_sha256:
        raise ValueError("DPDFNet2 48 kHz HR model identity differs from the route")
    return runtime, lambda: dpdfnet2_48k_hr.StreamingEnhancer(runtime)


def _expected_model_inferences(samples: int) -> int:
    if samples < 0:
        raise ValueError("stream sample count must be non-negative")
    # The first callback fills 480 samples, and every callback thereafter
    # completes one 960-sample window.  Thus N callbacks produce N-1 model
    # hops.  The first four hops are the model's latency prefix and are dropped
    # before the output FIFO; the fifth output-frame delay is the one-hop
    # startup buffer required by the 960-sample analysis window.
    return max(0, samples // FRAME_SAMPLES - 1)


def _state_bytes(stream) -> int:
    # Recurrent state plus the official analysis/OLA buffers owned by the
    # enhancer.  The input buffer is intentionally excluded because it varies
    # with callback phase and is not a model allocation.
    total = int(stream.runtime.init_state.nbytes)
    total += int(stream.window.nbytes)
    total += int(stream.ola_buffer.nbytes)
    return total


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

        spec = NS_ROUTES["dpdfnet2_48k_hr"]
        verify_assets = lambda: _verify_route_assets("dpdfnet2_48k_hr")
    else:
        spec = route_spec
        verify_assets = verify_assets or (lambda: None)
    if (
        getattr(spec, "sample_rate", SAMPLE_RATE) != SAMPLE_RATE
        or getattr(spec, "frame_samples", FRAME_SAMPLES) != FRAME_SAMPLES
        or getattr(spec, "window_samples", WINDOW_SAMPLES) != WINDOW_SAMPLES
        or getattr(spec, "hop_samples", HOP_SAMPLES) != HOP_SAMPLES
        or getattr(spec, "delay_samples", MODEL_LATENCY_SAMPLES) != MODEL_LATENCY_SAMPLES
    ):
        raise ValueError("48 kHz HR worker route geometry is invalid")

    input_fd = sys.stdin.fileno() if input_fd is None else input_fd
    output_fd = sys.stdout.fileno() if output_fd is None else output_fd
    if input_fd == output_fd:
        raise ValueError("stream worker input and output fds must differ")
    _validate_worker_fd(input_fd, writable=False)
    _validate_worker_fd(output_fd, writable=True)

    load_started = time.perf_counter()
    backend = load_backend or (lambda: _load_backend(spec, verify_assets))
    model, stream_factory = backend()
    startup_load_seconds = time.perf_counter() - load_started

    with nullcontext():
        runtime = stream_factory()
        if getattr(runtime, "runtime", None) is not model:
            raise ValueError("live stream did not receive the one loaded model")
        warmup_seconds = 0.0
        warmup_inferences = 0
        pre_roll_output_frames = 0
        ready = {
            "protocol": "itu-ns-f32le-v1",
            "backend": "dpdfnet2_48k_hr",
            "model_sha256": spec.model_sha256,
            "frontend_sha256": spec.frontend_sha256,
            "frame_samples": FRAME_SAMPLES,
            "sample_rate": SAMPLE_RATE,
            "delay_samples": DELAY_SAMPLES,
            "model_latency_samples": MODEL_LATENCY_SAMPLES,
            "model_latency_frames": MODEL_LATENCY_FRAMES,
            "startup_delay_samples": STARTUP_DELAY_SAMPLES,
            "processing_mode": "stateful_streaming",
            "window_samples": WINDOW_SAMPLES,
            "hop_samples": HOP_SAMPLES,
            "offline_whole_utterance": False,
            "lookahead_samples": WINDOW_SAMPLES,
        }
        startup_to_ready_seconds = time.perf_counter() - _MODULE_STARTED
        _exact_write(output_fd, json.dumps(ready, separators=(",", ":")).encode("ascii") + b"\n")

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
        output_queue = deque(
            np.zeros(FRAME_SAMPLES, dtype=np.float32) for _ in range(DELAY_FRAMES)
        )
        callbacks = 0
        model_output_frames = 0
        aligned_model_output_frames = 0
        model_trim_frames = MODEL_LATENCY_FRAMES
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
            raw = runtime.process(quantized)
            model_seconds = time.perf_counter() - model_started

            if not isinstance(raw, np.ndarray):
                raise ValueError("DPDFNet2 HR process must return a NumPy array")
            raw = np.asarray(raw, dtype=np.float32).reshape(-1)
            if raw.size:
                if raw.size % FRAME_SAMPLES or not bool(np.isfinite(raw).all()):
                    raise ValueError("DPDFNet2 HR output must contain finite 480-sample blocks")
                for offset in range(0, raw.size, FRAME_SAMPLES):
                    model_output_frames += 1
                    if model_trim_frames:
                        model_trim_frames -= 1
                        continue
                    output_queue.append(
                        np.ascontiguousarray(raw[offset : offset + FRAME_SAMPLES])
                    )
                    aligned_model_output_frames += 1
            if not output_queue:
                raise ValueError("DPDFNet2 HR output queue underrun")

            postprocess_started = time.perf_counter()
            enhanced = output_queue.popleft()
            if (
                enhanced.shape != (FRAME_SAMPLES,)
                or enhanced.dtype != np.float32
                or not enhanced.flags.c_contiguous
                or not bool(np.isfinite(enhanced).all())
            ):
                raise ValueError("stream model must return 480 finite contiguous float32 samples")
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
    expected_model_inferences = _expected_model_inferences(callbacks * FRAME_SAMPLES)
    if int(runtime.frames_processed) != expected_model_inferences:
        raise ValueError("stream model inference count does not match hop schedule")
    verify_assets()

    read_timing = read_stats.summary()
    preprocess_timing = preprocess_stats.summary()
    model_timing = model_stats.summary()
    postprocess_timing = postprocess_stats.summary()
    callback_timing = callback_stats.summary()
    source_path = Path(__file__).resolve()
    helper_path = Path(__import__("server.ns_stream_worker", fromlist=["__file__"]).__file__).resolve()
    command = list(
        sys.orig_argv
        if command_argv is None and hasattr(sys, "orig_argv")
        else (sys.argv if command_argv is None else command_argv)
    )
    receipt = {
        **ready,
        "schema_version": 2,
        "stateful": True,
        "offline_whole_utterance": False,
        "precomputed_replay": False,
        "callback_frames": callbacks,
        "samples": callbacks * FRAME_SAMPLES,
        "input_payload_sha256": input_hash.hexdigest(),
        "quantized_input_pcm16_sha256": quantized_hash.hexdigest(),
        "output_payload_sha256": output_hash.hexdigest(),
        "model_inferences": int(runtime.frames_processed),
        "live_model_inferences": int(runtime.frames_processed) - warmup_inferences,
        "expected_model_inferences": expected_model_inferences,
        "model_hop_samples": HOP_SAMPLES,
        "model_ready_threshold_samples": WINDOW_SAMPLES,
        "model_initializations": 1,
        "live_stream_initializations": 1,
        "live_state_reset_calls": 1,
        "warmup_calls": WARMUP_FRAMES,
        "warmup_inferences": warmup_inferences,
        "pre_roll_frames": WARMUP_FRAMES,
        "pre_roll_output_frames": pre_roll_output_frames,
        "state_bytes": _state_bytes(runtime),
        "model_output_frames": model_output_frames,
        "aligned_model_output_frames": aligned_model_output_frames,
        "model_latency_samples": MODEL_LATENCY_SAMPLES,
        "model_latency_frames": MODEL_LATENCY_FRAMES,
        "startup_delay_samples": STARTUP_DELAY_SAMPLES,
        "pending_output_frames": len(output_queue),
        "delay_frames": DELAY_FRAMES,
        "delay_samples": DELAY_SAMPLES,
        "inference_seconds": float(runtime.inference_sec),
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
            "from ns_stream_worker_48k_hr module import through ready-line construction; "
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
        "worker_source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "worker_helper_source_sha256": hashlib.sha256(helper_path.read_bytes()).hexdigest(),
        "worker_command": command,
        "worker_command_sha256": hashlib.sha256(
            json.dumps(command, ensure_ascii=True, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "elapsed_seconds": time.perf_counter() - started,
        "total_run_seconds": time.perf_counter() - run_started,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "pcm16_quantization": "once_effective",
        "clean_reference_read": False,
        "fallback_frames": 0,
        "eos": "native VCP supplies zero drain callbacks including its declared fixed delay",
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return receipt


def main() -> int:
    os.environ.update({name: "1" for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
