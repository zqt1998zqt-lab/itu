"""Real pinned-model smoke test for the 48 kHz HR callback worker."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from server import dpdfnet2_48k_hr, ns_routes, ns_stream_worker_48k_hr as worker


MODEL = (
    Path(__file__).resolve().parents[1]
    / "build/ns-dpdfnet2-48k-hr-20261118/assets/dpdfnet2_48khz_hr.onnx"
)


def _read_all(fd: int) -> bytes:
    chunks = []
    while True:
        block = os.read(fd, 8192)
        if not block:
            return b"".join(chunks)
        chunks.append(block)


@pytest.mark.skipif(not MODEL.is_file(), reason="pinned 48 kHz HR ONNX is unavailable")
def test_real_48k_worker_preserves_callback_count_and_state(tmp_path):
    count = 12
    live_callbacks = count + worker.DELAY_FRAMES
    time = np.arange(count * worker.FRAME_SAMPLES, dtype=np.float32) / np.float32(48_000)
    frames = (
        0.08 * np.sin(2 * np.pi * 997 * time)
        + 0.01 * np.cos(2 * np.pi * 231 * time)
    ).astype(np.float32).reshape(count, worker.FRAME_SAMPLES)
    live_frames = np.concatenate((
        frames,
        np.zeros((worker.DELAY_FRAMES, worker.FRAME_SAMPLES), dtype=np.float32),
    ))
    input_read, input_write = os.pipe()
    output_read, output_write = os.pipe()
    os.write(input_write, live_frames.astype("<f4", copy=False).tobytes())
    os.close(input_write)
    receipt_path = tmp_path / "real-receipt.json"
    try:
        receipt = worker.run(
            receipt_path,
            input_fd=input_read,
            output_fd=output_write,
            command_argv=["/venv/python", "-m", "server.ns_stream_worker_48k_hr"],
        )
    finally:
        os.close(input_read)
        os.close(output_write)
    payload = _read_all(output_read)
    os.close(output_read)
    ready_line, separator, frames_out = payload.partition(b"\n")
    assert separator == b"\n"
    ready = json.loads(ready_line)
    output = np.frombuffer(frames_out, dtype="<f4")

    assert ready["backend"] == "dpdfnet2_48k_hr"
    assert ready["sample_rate"] == 48_000
    assert ready["frame_samples"] == 480
    assert ready["delay_samples"] == 2_400
    assert ready["model_latency_samples"] == 1_920
    assert ready["startup_delay_samples"] == 480
    assert output.shape == (live_callbacks * worker.FRAME_SAMPLES,)
    assert np.isfinite(output).all()
    assert np.count_nonzero(output[: worker.DELAY_SAMPLES]) == 0
    assert receipt["callback_frames"] == live_callbacks
    assert receipt["model_inferences"] == live_callbacks - 1
    assert receipt["live_model_inferences"] == live_callbacks - 1
    assert receipt["fallback_frames"] == 0
    assert receipt["precomputed_replay"] is False
    assert receipt["worker_source_sha256"]
    assert receipt["state_bytes"] > 0

    runtime = dpdfnet2_48k_hr.load_runtime(MODEL)
    quantized, _ = ns_routes._quantize_pcm16_lround(frames.reshape(-1))
    aligned, _ = dpdfnet2_48k_hr.enhance_stream(
        quantized,
        runtime,
        chunk_size=worker.FRAME_SAMPLES,
        tail_frames=12,
    )
    assert np.array_equal(output[worker.DELAY_SAMPLES :], aligned)
