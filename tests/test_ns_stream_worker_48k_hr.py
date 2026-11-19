"""Transport and timeline tests for the native 48 kHz DPDFNet2 HR worker."""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from server import dpdfnet2_48k_hr, ns_stream_worker_48k_hr as worker


class _Model:
    def __init__(self) -> None:
        self.init_state = np.zeros(dpdfnet2_48k_hr.STATE_SIZE, dtype=np.float32)


class _Stream:
    def __init__(self, model: _Model) -> None:
        self.runtime = model
        self.window = np.zeros(worker.WINDOW_SAMPLES, dtype=np.float32)
        self.ola_buffer = np.zeros(worker.WINDOW_SAMPLES, dtype=np.float32)
        self.state = model.init_state.copy()
        self.frames_processed = 0
        self.inference_sec = 0.0
        self.calls = 0

    def process(self, frame: np.ndarray) -> np.ndarray:
        self.calls += 1
        if self.calls == 1:
            return np.zeros(0, dtype=np.float32)
        self.frames_processed += 1
        self.state[0] += np.float32(1.0)
        return np.ascontiguousarray(frame * np.float32(0.5), dtype=np.float32)


def _spec() -> SimpleNamespace:
    return SimpleNamespace(
        sample_rate=worker.SAMPLE_RATE,
        frame_samples=worker.FRAME_SAMPLES,
        window_samples=worker.WINDOW_SAMPLES,
        hop_samples=worker.HOP_SAMPLES,
        delay_samples=worker.MODEL_LATENCY_SAMPLES,
        processing_mode="stateful_streaming",
        model_sha256="1" * 64,
        frontend_sha256="2" * 64,
        model_path="/tmp/fake-48k.onnx",
    )


def _read_all(fd: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        block = os.read(fd, 8192)
        if not block:
            return b"".join(chunks)
        chunks.append(block)


def _run_dummy(tmp_path, frames: np.ndarray):
    raw = np.ascontiguousarray(frames, dtype="<f4").tobytes()
    input_read, input_write = os.pipe()
    output_read, output_write = os.pipe()
    os.write(input_write, raw)
    os.close(input_write)
    model = _Model()
    streams: list[_Stream] = []

    def make_stream():
        stream = _Stream(model)
        streams.append(stream)
        return stream

    def load_backend():
        return model, make_stream

    receipt_path = tmp_path / "receipt.json"
    try:
        receipt = worker.run(
            receipt_path,
            input_fd=input_read,
            output_fd=output_write,
            route_spec=_spec(),
            load_backend=load_backend,
            verify_assets=lambda: None,
            command_argv=["/fake/python", "-m", "server.ns_stream_worker_48k_hr"],
        )
    finally:
        os.close(input_read)
        os.close(output_write)
    payload = _read_all(output_read)
    os.close(output_read)
    ready_line, separator, frame_bytes = payload.partition(b"\n")
    assert separator == b"\n"
    return receipt, json.loads(ready_line), frame_bytes, raw, streams


def test_worker_emits_four_zero_frames_then_aligned_live_frames(tmp_path):
    values = np.linspace(-0.2, 0.2, 12 * worker.FRAME_SAMPLES, dtype=np.float32)
    frames = values.reshape(12, worker.FRAME_SAMPLES)
    receipt, ready, frame_bytes, raw, streams = _run_dummy(tmp_path, frames)

    quantized, _ = worker.np.frombuffer(raw, dtype="<f4"), None
    expected_quantized, _ = _quantize(quantized)
    expected_quantized = expected_quantized.reshape(-1, worker.FRAME_SAMPLES)
    expected = np.concatenate((
        np.zeros((worker.DELAY_FRAMES, worker.FRAME_SAMPLES), dtype=np.float32),
        expected_quantized[5:12] * np.float32(0.5),
    ), axis=0)
    assert np.frombuffer(frame_bytes, dtype="<f4").reshape(-1, worker.FRAME_SAMPLES).tobytes() == expected.astype("<f4").tobytes()
    assert ready["sample_rate"] == 48_000
    assert ready["frame_samples"] == 480
    assert ready["delay_samples"] == 2_400
    assert ready["processing_mode"] == "stateful_streaming"
    assert receipt["callback_frames"] == 12
    assert receipt["samples"] == 12 * 480
    assert receipt["warmup_calls"] == 0
    assert receipt["warmup_inferences"] == 0
    assert receipt["live_model_inferences"] == 11
    assert receipt["model_inferences"] == 11
    assert receipt["expected_model_inferences"] == 11
    assert receipt["delay_frames"] == 5
    assert receipt["pre_roll_output_frames"] == 0
    assert receipt["pending_output_frames"] == 0
    assert streams[0].state[0] == 11


def test_worker_rejects_nonfinite_and_truncated_frames(tmp_path):
    for payload, message in (
        (np.array([np.nan] + [0.0] * 479, dtype="<f4").tobytes(), "NaN or Inf"),
        (b"\0" * (worker.FRAME_BYTES - 1), "truncated"),
    ):
        input_read, input_write = os.pipe()
        output_read, output_write = os.pipe()
        os.write(input_write, payload)
        os.close(input_write)
        model = _Model()
        receipt_path = tmp_path / ("bad-" + message.split()[0] + ".json")
        with pytest.raises(ValueError, match=message):
            worker.run(
                receipt_path,
                input_fd=input_read,
                output_fd=output_write,
                route_spec=_spec(),
                load_backend=lambda: (model, lambda: _Stream(model)),
                verify_assets=lambda: None,
                command_argv=["/fake/python"],
            )
        os.close(input_read)
        os.close(output_write)
        os.close(output_read)
        assert not receipt_path.exists()


def _quantize(values: np.ndarray):
    values = np.asarray(values, dtype=np.float32)
    scaled = np.clip(values.astype(np.float64) * 32768.0, -32768.0, 32767.0)
    integers = np.copysign(np.floor(np.abs(scaled) + 0.5), scaled).astype(np.int16)
    return (integers.astype(np.float32) / np.float32(32768.0)).astype(np.float32), integers
