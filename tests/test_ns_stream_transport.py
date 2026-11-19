"""Transport-only tests for the persistent streaming NS worker."""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from server import ns_routes, ns_stream_worker


class _Model:
    def __init__(self) -> None:
        self.reset_calls = 0


class _Stream:
    frame_samples = 160
    hop_samples = 256
    fft_size = 512

    def __init__(self, model: _Model, delay_samples: int, registry: list) -> None:
        self.model = model
        self.delay_samples = delay_samples
        self.inferences = 0
        self.state_bytes = 1234
        self.process_calls = 0
        self.live_calls = 0
        self.total_samples = 0
        self.reset_calls = 1
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
        self.process_calls += 1
        self._advance()
        return np.zeros(160, dtype=np.float32)

    def _consume_validated(self, frame: np.ndarray) -> np.ndarray:
        assert frame.shape == (160,)
        assert frame.dtype == np.float32
        assert frame.flags.c_contiguous
        self.live_calls += 1
        self._advance()
        return np.ascontiguousarray(frame * np.float32(0.5), dtype=np.float32)


def _route_spec(delay_samples: int = 480):
    return SimpleNamespace(
        model_sha256="1" * 64,
        frontend_sha256="2" * 64,
        delay_samples=delay_samples,
        processing_mode="stateful_streaming",
    )


def _frames(count: int) -> np.ndarray:
    rng = np.random.default_rng(20261116)
    values = rng.uniform(-1.1, 1.1, size=(count, 160)).astype(np.float32)
    edge_count = min(9, count)
    values[:edge_count, 0] = np.array(
        [-2.0, -1.0, -0.5, -0.5 / 32768, -0.0, 0.5 / 32768, 0.5, 1.0, 2.0],
        dtype=np.float32,
    )[:edge_count]
    return values


def _read_all(fd: int) -> bytes:
    chunks = []
    while True:
        block = os.read(fd, 4096)
        if not block:
            return b"".join(chunks)
        chunks.append(block)


def _split_ready(payload: bytes) -> tuple[dict, bytes]:
    line, separator, frames = payload.partition(b"\n")
    assert separator == b"\n"
    return json.loads(line), frames


def _run_dummy(tmp_path, frames: np.ndarray, *, delay_samples: int = 480):
    spec = _route_spec(delay_samples)
    raw = np.ascontiguousarray(frames, dtype="<f4").tobytes()
    input_read, input_write = os.pipe()
    output_read, output_write = os.pipe()
    os.write(input_write, raw)
    os.close(input_write)

    model = _Model()
    streams: list[_Stream] = []
    backend_calls = 0
    asset_checks = 0

    def load_backend():
        nonlocal backend_calls
        backend_calls += 1
        verify_assets()
        return model, lambda: _Stream(model, delay_samples, streams), nullcontext

    def verify_assets():
        nonlocal asset_checks
        asset_checks += 1

    receipt_path = tmp_path / "transport-receipt.json"
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
    finally:
        os.close(input_read)
        os.close(output_write)
    output = _read_all(output_read)
    os.close(output_read)
    ready, frame_output = _split_ready(output)
    return receipt, ready, raw, frame_output, model, streams, backend_calls, asset_checks


def test_preallocated_quantization_is_bit_exact_to_frozen_route_helper():
    values = _frames(1)[0]
    values[:9] = np.array(
        [-2.0, -1.0, -0.5, -0.5 / 32768, -0.0, 0.5 / 32768, 0.5, 1.0, 2.0],
        dtype=np.float32,
    )
    expected_quantized, expected_pcm_bytes = ns_routes._quantize_pcm16_lround(values)

    processor = ns_stream_worker.FrameProcessor()
    processor.input_wire[:] = values.astype("<f4", copy=False).tobytes()
    actual_quantized = processor.quantize()

    assert processor.pcm16_bytes.tobytes() == expected_pcm_bytes
    np.testing.assert_array_equal(actual_quantized, expected_quantized)
    assert processor.pcm16.dtype == np.dtype("<i2")
    assert processor.quantized.dtype == np.float32


def test_reused_frame_processor_stays_bit_exact_across_rolling_frames():
    frames = _frames(257)
    processor = ns_stream_worker.FrameProcessor()
    for frame in frames:
        expected_quantized, expected_pcm_bytes = ns_routes._quantize_pcm16_lround(frame)
        processor.input_wire[:] = frame.astype("<f4", copy=False).tobytes()
        actual_quantized = processor.quantize()
        assert processor.pcm16_bytes.tobytes() == expected_pcm_bytes
        np.testing.assert_array_equal(actual_quantized, expected_quantized)


@pytest.mark.parametrize(
    ("samples", "expected"),
    [
        (256, 0),
        (257, 1),
        (512, 2),
        (8 * 160, 5),
        (1200 * 160, 750),
        (1208 * 160, 755),
    ],
)
def test_model_inference_expectation_matches_cofi_hop_schedule(samples, expected):
    assert ns_stream_worker._expected_model_inferences(
        samples,
        hop_samples=256,
        ready_threshold_samples=257,
    ) == expected


def test_dummy_stream_hashes_timing_and_one_model_lifecycle(tmp_path):
    frames = _frames(20)
    receipt, ready, raw, output, model, streams, backend_calls, asset_checks = _run_dummy(
        tmp_path, frames, delay_samples=480,
    )

    expected_quantized, expected_pcm_bytes = ns_routes._quantize_pcm16_lround(frames.reshape(-1))
    expected_output = np.ascontiguousarray(
        expected_quantized.reshape(-1, 160) * np.float32(0.5),
        dtype="<f4",
    ).tobytes()

    assert backend_calls == 1
    assert asset_checks == 2
    assert ready == {
        "protocol": "itu-ns-f32le-v1", "backend": "cofilite",
        "model_sha256": "1" * 64, "frontend_sha256": "2" * 64,
        "frame_samples": 160, "sample_rate": 16000,
        "delay_samples": 480, "processing_mode": "stateful_streaming",
    }
    assert model.reset_calls == 0
    assert len(streams) == 2
    assert streams[0].process_calls == 8
    assert streams[0].inferences == 5
    assert streams[1].live_calls == 20
    assert streams[1].inferences == 12
    assert streams[1].reset_calls == 1
    assert receipt["delay_samples"] == 480
    assert receipt["callback_frames"] == 20
    assert receipt["model_initializations"] == 1
    assert receipt["live_stream_initializations"] == 1
    assert receipt["live_state_reset_calls"] == 1
    assert receipt["model_inferences"] == 12
    assert receipt["expected_model_inferences"] == 12
    assert receipt["model_hop_samples"] == 256
    assert receipt["model_ready_threshold_samples"] == 257
    assert receipt["warmup_calls"] == 8
    assert receipt["warmup_inferences"] == 5
    assert receipt["input_payload_sha256"] == hashlib.sha256(raw).hexdigest()
    assert receipt["quantized_input_pcm16_sha256"] == hashlib.sha256(
        expected_pcm_bytes
    ).hexdigest()
    assert receipt["output_payload_sha256"] == hashlib.sha256(expected_output).hexdigest()
    assert output == expected_output
    assert len(receipt["timing_samples"]) == 16
    assert receipt["timing_sample_capacity"] == 16
    assert receipt["timing_sample_policy"].startswith("trailing bounded window")
    assert receipt["startup_timing_scope"].startswith("from ns_stream_worker module import")
    assert receipt["timing_clock"] == "time.perf_counter monotonic seconds"
    assert receipt["timing"]["callback"]["count"] == 20
    assert receipt["timing"]["quantize_hash_preprocess"]["count"] == 20
    assert receipt["timing"]["model_process"]["count"] == 20
    assert receipt["timing"]["postprocess_write"]["count"] == 20
    assert receipt["timing"]["callback"]["p50_upper_ms"] <= receipt["timing"]["callback"]["p95_upper_ms"]
    assert receipt["timing"]["callback"]["p95_upper_ms"] <= receipt["timing"]["callback"]["p99_upper_ms"]
    assert receipt["worker_source_sha256"] == hashlib.sha256(
        Path(ns_stream_worker.__file__).read_bytes()
    ).hexdigest()
    assert receipt["worker_command"] == ["/fake/python", "-m", "server.ns_stream_worker"]
    assert len(receipt["worker_command_sha256"]) == 64
    assert json.loads((tmp_path / "transport-receipt.json").read_text()) == receipt


def test_exact_read_into_handles_eintr_short_read_eof_and_truncation(monkeypatch):
    real_readv = os.readv
    input_read, input_write = os.pipe()
    os.write(input_write, b"abcdefghij")
    os.close(input_write)
    calls = 0

    def short_readv(fd, buffers):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise InterruptedError
        if calls == 2:
            return real_readv(fd, [buffers[0][:5]])
        return real_readv(fd, buffers)

    monkeypatch.setattr(ns_stream_worker.os, "readv", short_readv)
    buffer = bytearray(10)
    assert ns_stream_worker._exact_read_into(input_read, buffer) is True
    assert buffer == b"abcdefghij"
    assert ns_stream_worker._exact_read_into(input_read, bytearray(1)) is False
    os.close(input_read)

    truncated_read, truncated_write = os.pipe()
    os.write(truncated_write, b"abc")
    os.close(truncated_write)
    with pytest.raises(ValueError, match="truncated"):
        ns_stream_worker._exact_read_into(truncated_read, bytearray(4))
    os.close(truncated_read)


def test_exact_write_handles_eintr_short_write_and_broken_pipe(monkeypatch):
    real_write = os.write
    output_read, output_write = os.pipe()
    calls = 0

    def short_write(fd, data):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise InterruptedError
        if calls == 2:
            return real_write(fd, data[:3])
        return real_write(fd, data)

    monkeypatch.setattr(ns_stream_worker.os, "write", short_write)
    ns_stream_worker._exact_write(output_write, b"abcdefghij")
    os.close(output_write)
    assert _read_all(output_read) == b"abcdefghij"
    os.close(output_read)

    closed_read, closed_write = os.pipe()
    os.close(closed_read)
    with pytest.raises(ValueError, match="write failed"):
        ns_stream_worker._exact_write(closed_write, b"x")
    os.close(closed_write)


def test_nan_and_truncated_frames_fail_without_fallback_or_receipt(tmp_path):
    for raw, message in (
        (np.zeros(160, dtype="<f4").tobytes(), "NaN or Inf"),
        (b"\0" * 639, "truncated"),
    ):
        input_read, input_write = os.pipe()
        output_read, output_write = os.pipe()
        if message == "NaN or Inf":
            frame = np.zeros(160, dtype="<f4")
            frame[0] = np.nan
            os.write(input_write, frame.tobytes())
        else:
            os.write(input_write, raw)
        os.close(input_write)

        spec = _route_spec()
        model = _Model()
        streams: list[_Stream] = []

        def load_backend():
            return model, lambda: _Stream(model, spec.delay_samples, streams), nullcontext

        receipt_path = tmp_path / f"failed-{message.split()[0]}.json"
        with pytest.raises(ValueError, match=message):
            ns_stream_worker.run(
                receipt_path,
                input_fd=input_read,
                output_fd=output_write,
                route_spec=spec,
                load_backend=load_backend,
                command_argv=["/fake/python", "-m", "server.ns_stream_worker"],
            )
        os.close(input_read)
        os.close(output_write)
        _ready, frame_output = _split_ready(_read_all(output_read))
        assert frame_output == b""
        os.close(output_read)
        assert not receipt_path.exists()
        assert streams[1].live_calls == 0


def test_latency_stats_use_bounded_histogram_and_count_tail():
    stats = ns_stream_worker.LatencyStats()
    for seconds in (0.001, 0.002, 0.009, 0.011, 0.050):
        stats.observe(seconds)
    summary = stats.summary()

    assert summary["count"] == 5
    assert summary["over_10ms"] == 2
    assert summary["max_ms"] == pytest.approx(50.0)
    assert summary["p50_upper_ms"] <= summary["p95_upper_ms"]
    assert summary["p95_upper_ms"] <= summary["p99_upper_ms"]
    assert stats._histogram.size == ns_stream_worker._HISTOGRAM_BUCKETS
    with pytest.raises(ValueError, match="finite"):
        stats.observe(float("nan"))
