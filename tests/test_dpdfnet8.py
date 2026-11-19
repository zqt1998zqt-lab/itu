"""Contract tests for the official DPDFNet-2 8 kHz streaming adapter."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from server import dpdfnet8


pytestmark = pytest.mark.skipif(
    not dpdfnet8.DEFAULT_MODEL_PATH.is_file(),
    reason=f"missing pinned ONNX asset: {dpdfnet8.DEFAULT_MODEL_PATH}",
)


@pytest.fixture(scope="module")
def runtime():
    return dpdfnet8.load_runtime()


def _scratch_dir(prefix: str) -> Path:
    # Keep artifacts in /tmp: repository policy forbids deleting evidence.
    return Path(tempfile.mkdtemp(prefix=f"itu-dpdfnet8-{prefix}-"))


def _quantized_probe(samples: int = 4101, seed: int = 20261118) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time = np.arange(samples, dtype=np.float32) / np.float32(dpdfnet8.SAMPLE_RATE)
    waveform = (
        0.12 * np.sin(2 * np.pi * 437 * time)
        + 0.025 * rng.standard_normal(samples)
    )
    integers = np.clip(
        np.rint(waveform.astype(np.float64) * 32768.0),
        -32768.0,
        32767.0,
    ).astype(np.int16)
    return np.ascontiguousarray(
        integers.astype(np.float32) / np.float32(32768.0), dtype=np.float32
    )


def test_asset_metadata_state_layout_and_cpu_signature(runtime):
    digest = hashlib.sha256(dpdfnet8.DEFAULT_MODEL_PATH.read_bytes()).hexdigest()
    assert digest == dpdfnet8.MODEL_SHA256
    assert runtime.model_sha256 == dpdfnet8.MODEL_SHA256
    assert runtime.model_path == dpdfnet8.DEFAULT_MODEL_PATH.resolve()
    assert runtime.model_size_bytes == dpdfnet8.DEFAULT_MODEL_PATH.stat().st_size
    assert runtime.session.get_providers() == ["CPUExecutionProvider"]

    expected_io = (
        ("spec", "tensor(float)", [1, 1, 81, 2]),
        ("state_in", "tensor(float)", [37860]),
    )
    actual_io = tuple(
        (item.name, item.type, item.shape) for item in runtime.session.get_inputs()
    )
    assert actual_io == expected_io
    assert tuple(
        (item.name, item.type, item.shape) for item in runtime.session.get_outputs()
    ) == (
        ("spec_e", "tensor(float)", [1, 1, 81, 2]),
        ("state_out", "tensor(float)", [37860]),
    )

    for key, value in dpdfnet8.EXPECTED_METADATA.items():
        assert runtime.metadata[key] == value
    assert runtime.init_state.shape == (37860,)
    assert runtime.init_state.dtype == np.float32
    erb = np.asarray(
        runtime.metadata["erb_norm_init"].split(","), dtype=np.float32
    )
    spec = np.asarray(
        runtime.metadata["spec_norm_init"].split(","), dtype=np.float32
    )
    assert erb.shape == (81,)
    assert spec.shape == (80,)
    assert np.array_equal(runtime.init_state[:81], erb)
    assert np.array_equal(runtime.init_state[81:161], spec)
    assert np.count_nonzero(runtime.init_state[161:]) == 0
    assert dpdfnet8.OUTPUT_TRIM_SAMPLES == 320
    assert dpdfnet8.ALGORITHM_LATENCY_SAMPLES == 320


def test_pcm16_and_float_inputs_are_deterministic_and_emit_float(runtime):
    scratch = _scratch_dir("formats")
    waveform = _quantized_probe()
    pcm16_input = scratch / "input-pcm16.wav"
    float_input = scratch / "input-float.wav"
    pcm16_output = scratch / "output-pcm16-input.wav"
    float_output = scratch / "output-float-input.wav"
    sf.write(pcm16_input, waveform, dpdfnet8.SAMPLE_RATE, subtype="PCM_16")
    sf.write(float_input, waveform, dpdfnet8.SAMPLE_RATE, subtype="FLOAT")

    assert np.array_equal(
        dpdfnet8.read_8k_mono(pcm16_input), dpdfnet8.read_8k_mono(float_input)
    )
    first_info = dpdfnet8.enhance_file(
        pcm16_input,
        pcm16_output,
        runtime=runtime,
        chunk_size=80,
    )
    second_info = dpdfnet8.enhance_file(
        float_input,
        float_output,
        runtime=runtime,
        chunk_size=333,
    )

    first, first_rate = sf.read(
        pcm16_output, dtype="float32", always_2d=True
    )
    second, second_rate = sf.read(
        float_output, dtype="float32", always_2d=True
    )
    first = np.ascontiguousarray(first[:, 0], dtype=np.float32)
    second = np.ascontiguousarray(second[:, 0], dtype=np.float32)
    assert first_rate == second_rate == dpdfnet8.SAMPLE_RATE
    assert first.shape == second.shape == waveform.shape
    assert np.array_equal(first, second)
    assert np.all(np.isfinite(first))

    for output in (pcm16_output, float_output):
        info = sf.info(output)
        assert info.format == "WAV"
        assert info.samplerate == dpdfnet8.SAMPLE_RATE
        assert info.channels == 1
        assert info.subtype == "FLOAT"
        assert info.frames == waveform.size

    for info in (first_info, second_info):
        assert info["model_sha256"] == dpdfnet8.MODEL_SHA256
        assert info["model_sha"] == dpdfnet8.MODEL_SHA256
        assert info["metadata"]["profile"] == "dpdfnet2_8khz"
        assert info["input_samples"] == waveform.size
        assert info["output_samples"] == waveform.size
        assert info["frames"] == info["frames_processed"] > 0
        assert info["trim"] == info["trim_samples"] == 320
        assert info["latency"] == info["latency_samples"] == 320
        assert np.isfinite(info["final_state_norm"])
        assert info["state_norm"] == info["final_state_norm"] >= 0
        assert info.frames_processed == info.frames > 0
        assert info.trim_samples == info.latency_samples == 320
    assert first_info["frames_processed"] == second_info["frames_processed"]
    assert first_info["final_state_norm"] == second_info["final_state_norm"]


def test_deterministic_repeat_with_identical_chunking(runtime):
    scratch = _scratch_dir("determinism")
    waveform = _quantized_probe(samples=2003, seed=91)
    source = scratch / "input.wav"
    first_output = scratch / "first.wav"
    second_output = scratch / "second.wav"
    sf.write(source, waveform, dpdfnet8.SAMPLE_RATE, subtype="FLOAT")

    first_info = dpdfnet8.enhance_file(
        source, first_output, runtime=runtime, chunk_size=160
    )
    second_info = dpdfnet8.enhance_file(
        source, second_output, runtime=runtime, chunk_size=160
    )
    first = dpdfnet8.read_8k_mono(first_output)
    second = dpdfnet8.read_8k_mono(second_output)
    assert np.array_equal(first, second)
    assert first_info["frames_processed"] == second_info["frames_processed"]
    assert first_info["final_state_norm"] == second_info["final_state_norm"]


def test_rejects_non_8khz_input_before_inference(runtime):
    scratch = _scratch_dir("sample-rate")
    source = scratch / "input-16k.wav"
    target = scratch / "output.wav"
    sf.write(
        source,
        np.zeros(160, dtype=np.float32),
        16_000,
        subtype="FLOAT",
    )
    with pytest.raises(ValueError, match="requires 8000 Hz"):
        dpdfnet8.read_8k_mono(source)
    with pytest.raises(ValueError, match="requires 8000 Hz"):
        dpdfnet8.enhance_file(source, target, runtime=runtime)
    assert not target.exists()
