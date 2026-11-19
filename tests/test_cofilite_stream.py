"""Contract and real-checkpoint parity tests for the CoFi-Lite stream adapter."""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pytest

from server import cofilite_stream as stream


def _probe_audio(sample_count: int, seed: int = 20261116) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time = np.arange(sample_count, dtype=np.float32) / stream.SAMPLE_RATE
    return np.ascontiguousarray(
        (
            0.08 * np.sin(2 * np.pi * 437 * time)
            + 0.015 * rng.standard_normal(sample_count)
        ).astype(np.float32)
    )


def _run_raw_frames(
    model, audio: np.ndarray, *, chunk_size: int = stream.FRAME_SAMPLES
) -> np.ndarray:
    instance = stream.CoFiLiteStream(model)
    outputs = []
    full_samples = (int(audio.size) // chunk_size) * chunk_size
    for offset in range(0, full_samples, chunk_size):
        outputs.append(instance.process(audio[offset : offset + chunk_size]))
    tail = audio[full_samples:]
    if int(tail.size):
        tail_output, flush_output = instance._finish_with_tail(tail)
        outputs.append(tail_output)
    else:
        flush_output = instance.flush()
    return np.concatenate((*outputs, flush_output), dtype=np.float32)


@pytest.fixture(scope="module")
def model():
    return stream.load_model("cpu", threads=1)


@pytest.fixture(scope="module")
def non_hop_probe() -> np.ndarray:
    return _probe_audio(2570)


@pytest.fixture(scope="module")
def offline_parity(model, non_hop_probe):
    frozen = sys.modules["_itu_frozen_cofilite_infer"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        expected = frozen.enhance(model, non_hop_probe, "cpu")
    actual, info = stream.enhance_stream(
        non_hop_probe, chunk_size=stream.FRAME_SAMPLES, model=model
    )
    return expected, actual, info


def test_real_checkpoint_matches_frozen_offline_and_non_hop_tail(offline_parity):
    expected, actual, info = offline_parity
    difference = np.abs(actual - expected)

    assert actual.shape == expected.shape == (2570,)
    assert actual.dtype == expected.dtype == np.float32
    assert float(difference.max()) < 1e-4
    assert float(difference.mean()) < 1e-7
    # Upstream ISTFT emits floor(N / 256) * 256 samples and zero-pads the rest.
    assert np.array_equal(actual[-10:], np.zeros(10, dtype=np.float32))
    assert np.array_equal(expected[-10:], np.zeros(10, dtype=np.float32))

    assert info["sample_rate"] == 16_000
    assert info["frame_samples"] == 160
    assert info["delay_samples"] == 480
    assert info["lookahead_samples"] == 511
    assert info["stateful_streaming"] is True
    assert info["offline_whole_utterance"] is False
    assert info["whole_utterance_replay"] is False
    assert info["input_samples"] == 2570
    assert info["valid_model_samples"] == 2560
    assert info["zero_filled_tail_samples"] == 10
    assert info["flush_samples"] == 480
    assert info["raw_samples"] == 3050
    assert info["aligned_samples"] == 2570
    assert info["inferences"] == info["frames_processed"] == 11
    assert info["state_bytes"] > 0
    assert 0 <= info["cached_samples"] <= 2048


def test_process_is_fixed_160_and_flush_returns_physical_480(model, non_hop_probe):
    audio = non_hop_probe[:800]
    instance = stream.CoFiLiteStream(model)
    outputs = [
        instance.process(audio[offset : offset + stream.FRAME_SAMPLES])
        for offset in range(0, audio.size, stream.FRAME_SAMPLES)
    ]
    tail = instance.flush()
    raw = np.concatenate((*outputs, tail), dtype=np.float32)

    assert len(outputs) == 5
    assert all(output.shape == (160,) and output.dtype == np.float32 for output in outputs)
    assert np.array_equal(raw[:480], np.zeros(480, dtype=np.float32))
    assert raw.shape == (800 + 480,)
    assert tail.shape == (480,)
    assert raw[480:].shape == audio.shape
    assert np.isfinite(raw).all()

    with pytest.raises(RuntimeError, match="finished"):
        instance.process(np.zeros(160, dtype=np.float32))
    with pytest.raises(RuntimeError, match="finished"):
        instance.flush()


def test_chunking_is_independent_and_stream_never_replays_whole_forward(model):
    audio = _probe_audio(1000, seed=7)
    original_forward = model.forward

    def forbidden_whole_forward(*_args, **_kwargs):
        raise AssertionError("CoFiLiteStream must not call model.forward(audio)")

    model.forward = forbidden_whole_forward
    try:
        first, first_info = stream.enhance_stream(audio, chunk_size=160, model=model)
        second, second_info = stream.enhance_stream(audio, chunk_size=333, model=model)
    finally:
        model.forward = original_forward

    assert first.shape == second.shape == audio.shape
    assert np.array_equal(first, second)
    assert first_info["inferences"] == second_info["inferences"] == 4
    assert first_info["chunk_size"] == 160
    assert second_info["chunk_size"] == 333
    assert first_info["whole_utterance_replay"] is False
    assert second_info["whole_utterance_replay"] is False


def test_prefix_causality_negative_control(model):
    audio = _probe_audio(3200, seed=17)
    boundary = 1600
    changed = audio.copy()
    changed[boundary:] *= np.float32(-1.0)

    baseline_raw = _run_raw_frames(model, audio)
    changed_raw = _run_raw_frames(model, changed)

    # Raw output sample n is enhanced input n-480.  Thus changing input from
    # sample 1600 cannot change any delayed raw sample before sample 1600.
    assert np.array_equal(baseline_raw[:boundary], changed_raw[:boundary])
    assert float(np.abs(baseline_raw[boundary:] - changed_raw[boundary:]).max()) > 1e-4


def test_reset_isolates_stream_and_state_stays_bounded(model):
    audio = _probe_audio(960, seed=23)
    instance = stream.CoFiLiteStream(model)
    initial_bytes = instance.state_bytes
    outputs = []
    observed_bytes = []
    observed_samples = []
    for offset in range(0, audio.size, stream.FRAME_SAMPLES):
        outputs.append(instance.process(audio[offset : offset + stream.FRAME_SAMPLES]))
        observed_bytes.append(instance.state_bytes)
        observed_samples.append(instance.cached_samples)
    first_raw = np.concatenate((*outputs, instance.flush()), dtype=np.float32)

    assert instance.inferences == 4
    assert instance.state_bytes == initial_bytes
    assert max(observed_bytes) < 100_000
    assert max(observed_samples) <= 2048

    instance.reset()
    assert instance.inferences == 0
    assert instance.state_bytes == initial_bytes
    assert instance.cached_samples == 480
    second_outputs = [
        instance.process(audio[offset : offset + stream.FRAME_SAMPLES])
        for offset in range(0, audio.size, stream.FRAME_SAMPLES)
    ]
    second_raw = np.concatenate((*second_outputs, instance.flush()), dtype=np.float32)
    assert np.array_equal(first_raw, second_raw)


def test_silence_and_short_supported_boundary_are_finite(model):
    silence = np.zeros(257, dtype=np.float32)
    aligned, info = stream.enhance_stream(silence, chunk_size=160, model=model)

    assert aligned.shape == silence.shape
    assert aligned.dtype == np.float32
    assert np.isfinite(aligned).all()
    assert np.array_equal(aligned[-1:], np.zeros(1, dtype=np.float32))
    assert info["inferences"] == 2


def test_invalid_shapes_dtype_nonfinite_and_short_audio_are_rejected(model):
    instance = stream.CoFiLiteStream(model)
    with pytest.raises(ValueError, match="one-dimensional"):
        instance.process(np.zeros((160, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="exactly 160"):
        instance.process(np.zeros(159, dtype=np.float32))
    with pytest.raises(TypeError, match="float32"):
        instance.process(np.zeros(160, dtype=np.float64))
    bad = np.zeros(160, dtype=np.float32)
    bad[17] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        instance.process(bad)

    with pytest.raises(ValueError, match="more than 256"):
        stream.enhance_stream(np.zeros(256, dtype=np.float32), model=model)
    with pytest.raises(ValueError, match="more than 256"):
        stream.CoFiLiteStream(model).flush()


def test_training_model_is_rejected(model):
    instance = stream.CoFiLiteStream(model)
    model.train()
    try:
        with pytest.raises(RuntimeError, match="eval mode"):
            instance.process(np.zeros(160, dtype=np.float32))
    finally:
        model.eval()


def test_jit_construction_preserves_training_mode(model):
    model.train()
    try:
        instance = stream.CoFiLiteStream(model)
        assert model.training is True
        with pytest.raises(RuntimeError, match="eval mode"):
            instance.process(np.zeros(160, dtype=np.float32))
    finally:
        model.eval()


def test_jit_freezed_frame_core_matches_eager(model):
    audio = _probe_audio(1280, seed=31)
    eager, eager_info = stream.enhance_stream(audio, model=model, jit=False)
    jitted, jit_info = stream.enhance_stream(audio, model=model, jit=True)

    assert eager_info["frame_executor"] == "eager"
    assert jit_info["frame_executor"] == "jit_freeze"
    assert jit_info["frame_executor_trace_sec"] >= 0.0
    assert float(np.abs(eager - jitted).max()) < 1e-5


@pytest.mark.parametrize("sample_count", [257, 511, 767, 1280, 1281])
def test_non_hop_eos_phases_match_frozen_whole_utterance(model, sample_count):
    frozen = sys.modules["_itu_frozen_cofilite_infer"]
    audio = _probe_audio(sample_count, seed=10_000 + sample_count)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        expected = frozen.enhance(model, audio, "cpu")
    actual, info = stream.enhance_stream(audio, model=model)

    assert info["delay_samples"] == 480
    assert info["raw_samples"] == sample_count + 480
    assert float(np.abs(actual - expected).max()) < 1e-5
    remainder = sample_count % stream.HOP_SAMPLES
    if remainder:
        assert np.array_equal(
            actual[-remainder:],
            np.zeros(remainder, dtype=np.float32),
        )


def test_optional_256_sample_hop_and_256_delay_exact_blocks(model):
    sample_count = 2560
    frozen = sys.modules["_itu_frozen_cofilite_infer"]
    audio = _probe_audio(sample_count, seed=41)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        expected = frozen.enhance(model, audio, "cpu")
    actual, info = stream.enhance_stream(
        audio,
        chunk_size=256,
        model=model,
        frame_samples=256,
        delay_samples=256,
    )

    assert actual.shape == expected.shape == (sample_count,)
    assert info["frame_samples"] == 256
    assert info["delay_samples"] == info["trim_samples"] == 256
    assert info["flush_samples"] == 256
    assert info["raw_samples"] == sample_count + 256
    assert info["inferences"] == sample_count // stream.HOP_SAMPLES + 1
    assert float(np.abs(actual - expected).max()) < 1e-5


def test_configured_stream_rejects_unsafe_delay_shape(model):
    with pytest.raises(ValueError, match="integer in"):
        stream.CoFiLiteStream(model, frame_samples=0)
    with pytest.raises(ValueError, match="multiple"):
        stream.CoFiLiteStream(model, frame_samples=256, delay_samples=480)
