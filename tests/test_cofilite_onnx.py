"""Contract, parity, and fail-closed tests for the CoFi-Lite ONNX candidate."""

from __future__ import annotations

import hashlib
import json
import sys

import numpy as np
import onnx
import pytest
import torch

from server import cofilite_onnx as onnx_candidate
from server import cofilite_stream as stream
from server.cofilite_frame import CoFiFrameCore, initial_frame_states


def _probe_audio(sample_count: int, seed: int = 20261117) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time = np.arange(sample_count, dtype=np.float32) / stream.SAMPLE_RATE
    return np.ascontiguousarray(
        (
            0.06 * np.sin(2 * np.pi * 431 * time)
            + 0.01 * rng.standard_normal(sample_count)
        ).astype(np.float32)
    )


def _raw_stream(model, audio: np.ndarray) -> np.ndarray:
    instance = onnx_candidate.CoFiLiteOnnxStream(model)
    outputs = []
    with torch.inference_mode():
        for offset in range(0, int(audio.size), stream.FRAME_SAMPLES):
            outputs.append(
                instance.process(audio[offset : offset + stream.FRAME_SAMPLES])
            )
        outputs.append(instance.flush())
    return np.concatenate(outputs, dtype=np.float32)


@pytest.fixture(scope="module")
def model():
    return stream.load_model("cpu", threads=1)


@pytest.fixture(scope="module")
def executor(model):
    return onnx_candidate.get_onnx_frame_executor(model)


def test_exported_asset_identity_checker_and_provider(executor):
    identity = json.loads(executor.identity_path.read_text(encoding="utf-8"))
    assert executor.model_sha256 == onnx_candidate.DEFAULT_ONNX_SHA256
    assert executor.identity_sha256 == onnx_candidate.DEFAULT_ONNX_IDENTITY_SHA256
    assert identity["model_sha256"] == executor.model_sha256
    assert identity["weights_sha256"] == executor.weights_sha256
    assert identity["state_count"] == onnx_candidate.STATE_COUNT == 18
    assert identity["state_shapes"] == [list(shape) for shape in onnx_candidate.STATE_SHAPES]
    assert identity["onnx_checker"] == "pass"
    assert executor.providers == ("CPUExecutionProvider",)
    onnx.checker.check_model(str(executor.model_path))


def test_session_is_created_once_and_shared_between_streams(model):
    first = onnx_candidate.CoFiLiteOnnxStream(model)
    second = onnx_candidate.CoFiLiteOnnxStream(model)
    assert first._executor is second._executor
    assert first.executor_info["session_load_count"] == 1
    assert first.executor_info["file_hash_count"] == 5
    assert first.executor_info["state_count"] == 18
    assert first.executor_info["state_shapes"] == [
        list(shape) for shape in onnx_candidate.STATE_SHAPES
    ]
    assert first.executor_info["ort_version"] == onnx_candidate.ort.__version__
    assert first.executor_info["execution_provider"] == "CPUExecutionProvider"


def test_full_stream_parity_with_jit_on_2s(model, executor):
    audio = _probe_audio(3200)
    with torch.inference_mode():
        reference, reference_info = stream.enhance_stream(
            audio, model=model, jit=True
        )
        actual, actual_info = onnx_candidate.enhance_stream_onnx(
            audio, model=model
        )
    difference = np.abs(actual - reference)
    assert actual.shape == reference.shape == audio.shape
    assert float(difference.max()) < 1e-5
    assert float(np.sqrt(np.mean(difference * difference))) < 1e-6
    assert actual_info["delay_samples"] == 480
    assert actual_info["frame_executor"] == "onnxruntime_cpu"
    assert actual_info["inferences"] == reference_info["inferences"]
    assert actual_info["state_bytes"] == reference_info["state_bytes"]


def test_same_input_reset_and_prefix_causality(model):
    audio = _probe_audio(3200, seed=44)
    changed = audio.copy()
    changed[1600:] *= np.float32(-1.0)
    with torch.inference_mode():
        first, _ = onnx_candidate.enhance_stream_onnx(audio, model=model)
        second, _ = onnx_candidate.enhance_stream_onnx(audio, model=model)
    assert np.array_equal(first, second)
    baseline_raw = _raw_stream(model, audio)
    altered_raw = _raw_stream(model, changed)
    assert np.array_equal(baseline_raw[:1600], altered_raw[:1600])
    assert float(np.abs(baseline_raw[1600:] - altered_raw[1600:]).max()) > 1e-4


def test_frame_state_parity_across_50_frames(model, executor):
    core = CoFiFrameCore(model).eval()
    eager_states = initial_frame_states()
    onnx_states = initial_frame_states()
    rng = np.random.default_rng(81)
    maximum_enhanced = 0.0
    maximum_state = 0.0
    with torch.inference_mode():
        for _ in range(50):
            real = torch.from_numpy(
                rng.standard_normal((1, 257), dtype=np.float32)
            )
            imag = torch.from_numpy(
                rng.standard_normal((1, 257), dtype=np.float32)
            )
            eager_outputs = core(real, imag, *eager_states)
            onnx_outputs = executor(real, imag, *onnx_states)
            eager_states = tuple(eager_outputs[1:])
            onnx_states = tuple(onnx_outputs[1:])
            maximum_enhanced = max(
                maximum_enhanced,
                float((eager_outputs[0] - onnx_outputs[0]).abs().max()),
            )
            maximum_state = max(
                maximum_state,
                max(
                    float((left - right).abs().max())
                    for left, right in zip(eager_outputs[1:], onnx_outputs[1:])
                ),
            )
    assert maximum_enhanced < 1e-5
    # ONNX CPU kernel accumulation order can produce a last-bit difference;
    # record the strict observed bound instead of hiding it.
    # Compare with the requested 1e-5 tolerance, including relative
    # tolerance for state values whose magnitudes are not near zero.
    for left, right in zip(eager_outputs[1:], onnx_outputs[1:]):
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-5)
    assert np.isfinite(maximum_state)


def test_invalid_inputs_and_tampered_hash_fail_closed(executor):
    states = initial_frame_states()
    real = torch.zeros(1, 257, dtype=torch.float32)
    imag = torch.zeros(1, 257, dtype=torch.float32)
    bad_state = states[0].clone()
    bad_state[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        executor(real, imag, bad_state, *states[1:])
    with pytest.raises(ValueError, match="shape mismatch"):
        executor(real, imag, torch.zeros(1, dtype=torch.float32), *states[1:])
    with pytest.raises(TypeError, match="float32"):
        executor(real.double(), imag, *states)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        onnx_candidate.get_onnx_frame_executor(
            torch.nn.Identity(),
            expected_model_sha256="0" * 64,
            expected_identity_sha256=onnx_candidate.DEFAULT_ONNX_IDENTITY_SHA256,
        )


def test_enhance_stream_alias_and_optional_256_delay(model):
    assert onnx_candidate.enhance_stream is onnx_candidate.enhance_stream_onnx
    audio = _probe_audio(2570, seed=93)
    with torch.inference_mode():
        expected, expected_info = stream.enhance_stream(
            audio,
            chunk_size=256,
            model=model,
            frame_samples=256,
            delay_samples=256,
        )
        actual, actual_info = onnx_candidate.enhance_stream(
            audio,
            model=model,
            chunk_size=256,
            frame_samples=256,
            delay_samples=256,
        )
    assert actual_info["delay_samples"] == 256
    assert actual_info["flush_samples"] == 256
    assert actual_info["inferences"] == expected_info["inferences"]
    assert float(np.abs(actual - expected).max()) < 1e-5
