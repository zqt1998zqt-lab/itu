"""Contract tests for the causal DPDFNet2 48 kHz HR streaming adapter."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request

import numpy as np
import pytest
import soundfile as sf

from server import dpdfnet2_48k_hr


ROOT = Path(__file__).resolve().parents[1]
MODEL_SIZE_BYTES = 10_493_337
MODEL_URL = (
    "https://huggingface.co/Ceva-IP/DPDFNet/resolve/"
    "dd6818d00f50c836fed43a6243ebe49116de5964/onnx/"
    "dpdfnet2_48khz_hr.onnx"
)
MODEL_CACHE_DIR = Path(tempfile.gettempdir()) / (
    f"itu-dpdfnet2-48k-hr-model-{dpdfnet2_48k_hr.MODEL_SHA256}"
)


def _scratch_dir(prefix: str) -> Path:
    # Keep artifacts in /tmp: repository policy forbids deleting evidence.
    return Path(tempfile.mkdtemp(prefix=f"itu-dpdfnet2-48k-hr-{prefix}-"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_candidates() -> tuple[Path, ...]:
    candidates: list[Path] = []
    configured = os.environ.get("DPDFNET2_48K_HR_MODEL")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        (
            MODEL_CACHE_DIR / "model.onnx",
            ROOT
            / "build/ns-dpdfnet2-48k-hr-20261118/assets/dpdfnet2_48khz_hr.onnx",
            ROOT
            / "build/ns-dpdfnet2-48khz-20261118/assets/dpdfnet2_48khz_hr.onnx",
            ROOT
            / "build/dpdfnet2-48k-hr-20261118/assets/dpdfnet2_48khz_hr.onnx",
            ROOT / "models/dpdfnet2_48khz_hr.onnx",
        )
    )
    return tuple(candidates)


def _download_pinned_model() -> Path:
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    file_descriptor, name = tempfile.mkstemp(
        dir=MODEL_CACHE_DIR,
        prefix="model-",
        suffix=".onnx",
    )
    os.close(file_descriptor)
    path = Path(name)
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
            payload = response.read()
    except Exception as error:  # pragma: no cover - environment dependent
        pytest.skip(f"pinned 48 kHz HR model unavailable and download failed: {error}")
    path.write_bytes(payload)
    actual = _sha256_file(path)
    if actual != dpdfnet2_48k_hr.MODEL_SHA256:
        pytest.fail(
            "downloaded DPDFNet2 48 kHz HR model has the wrong SHA-256: "
            f"expected {dpdfnet2_48k_hr.MODEL_SHA256}, got {actual}"
        )
    return path


def _resolve_model_path() -> Path:
    for candidate in _model_candidates():
        if not candidate.is_file():
            continue
        actual = _sha256_file(candidate)
        if actual != dpdfnet2_48k_hr.MODEL_SHA256:
            pytest.fail(
                "DPDFNet2 48 kHz HR asset has the wrong SHA-256: "
                f"{candidate} expected {dpdfnet2_48k_hr.MODEL_SHA256}, "
                f"got {actual}"
            )
        return candidate.resolve()
    return _download_pinned_model()


@pytest.fixture(scope="module")
def model_path() -> Path:
    return _resolve_model_path()


@pytest.fixture(scope="module")
def runtime(model_path: Path):
    return dpdfnet2_48k_hr.load_runtime(model_path)


def _quantized_probe(samples: int = 1920, seed: int = 20261118) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time = (
        np.arange(samples, dtype=np.float32)
        / np.float32(dpdfnet2_48k_hr.SAMPLE_RATE)
    )
    waveform = (
        0.12 * np.sin(2 * np.pi * 997 * time)
        + 0.025 * rng.standard_normal(samples)
    )
    integers = np.clip(
        np.rint(waveform.astype(np.float64) * 32768.0),
        -32768.0,
        32767.0,
    ).astype(np.int16)
    return np.ascontiguousarray(
        integers.astype(np.float32) / np.float32(32768.0),
        dtype=np.float32,
    )


class _FakeNode:
    def __init__(self, name: str, shape: tuple[int, ...]) -> None:
        self.name = name
        self.type = "tensor(float)"
        self.shape = list(shape)


class _DeterministicSession:
    """Exercise the streaming schedule without paying for the 2.42 GMAC model."""

    def __init__(self) -> None:
        self._inputs = (
            _FakeNode(
                "spec",
                (1, 1, dpdfnet2_48k_hr.FREQ_BINS, 2),
            ),
            _FakeNode("state_in", (dpdfnet2_48k_hr.STATE_SIZE,)),
        )
        self._outputs = (
            _FakeNode(
                "spec_e",
                (1, 1, dpdfnet2_48k_hr.FREQ_BINS, 2),
            ),
            _FakeNode("state_out", (dpdfnet2_48k_hr.STATE_SIZE,)),
        )

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, output_names, feeds):
        assert output_names == ["spec_e", "state_out"]
        spec = np.asarray(feeds["spec"], dtype=np.float32).copy()
        state = np.asarray(feeds["state_in"], dtype=np.float32).copy()
        state[0] += np.float32(1.0)
        return [spec, state]


class _NaNSession(_DeterministicSession):
    def run(self, output_names, feeds):
        spec = np.asarray(feeds["spec"], dtype=np.float32).copy()
        state = np.asarray(feeds["state_in"], dtype=np.float32).copy()
        spec[0, 0, 0, 0] = np.nan
        return [spec, state]


def _fake_runtime():
    init_state = np.linspace(
        -1.0,
        1.0,
        dpdfnet2_48k_hr.STATE_SIZE,
        dtype=np.float32,
    )
    return dpdfnet2_48k_hr.RuntimeModel(
        session=_DeterministicSession(),
        model_path=Path("/tmp/fake-dpdfnet2-48k-hr.onnx"),
        init_state=np.ascontiguousarray(init_state),
        metadata={},
        model_sha256="0" * 64,
        model_size_bytes=0,
    )


def _stream_raw(
    waveform: np.ndarray,
    runtime,
    *,
    chunk_size: int,
    tail_frames: int,
) -> np.ndarray:
    enhancer = dpdfnet2_48k_hr.StreamingEnhancer(runtime)
    chunks = [
        enhancer.process(waveform[start : start + chunk_size])
        for start in range(0, waveform.size, chunk_size)
    ]
    chunks.append(enhancer.flush(tail_frames))
    return np.concatenate([chunk for chunk in chunks if chunk.size])


def test_rejects_wrong_model_sha_without_touching_pinned_asset():
    scratch = _scratch_dir("sha-mismatch")
    forged = scratch / "forged-model.onnx"
    forged.write_bytes(b"not the pinned DPDFNet2 48 kHz HR ONNX model")
    forged_sha = _sha256_file(forged)

    with pytest.raises(
        ValueError,
        match=(
            "DPDFNet2 48 kHz HR SHA mismatch: "
            f"expected {dpdfnet2_48k_hr.MODEL_SHA256}, got {forged_sha}"
        ),
    ):
        dpdfnet2_48k_hr.load_runtime(forged)

    assert forged.is_file()
    assert _sha256_file(forged) == forged_sha


def test_pinned_onnx_signature_metadata_and_initial_state(
    model_path: Path,
    runtime,
):
    assert _sha256_file(model_path) == dpdfnet2_48k_hr.MODEL_SHA256
    assert runtime.model_path == model_path.resolve()
    assert runtime.model_sha256 == dpdfnet2_48k_hr.MODEL_SHA256
    assert runtime.model_size_bytes == MODEL_SIZE_BYTES
    assert model_path.stat().st_size == MODEL_SIZE_BYTES
    assert runtime.session.get_providers() == ["CPUExecutionProvider"]

    assert tuple(
        (item.name, item.type, item.shape)
        for item in runtime.session.get_inputs()
    ) == (
        ("spec", "tensor(float)", [1, 1, 481, 2]),
        ("state_in", "tensor(float)", [56_436]),
    )
    assert tuple(
        (item.name, item.type, item.shape)
        for item in runtime.session.get_outputs()
    ) == (
        ("spec_e", "tensor(float)", [1, 1, 481, 2]),
        ("state_out", "tensor(float)", [56_436]),
    )

    expected_metadata = {
        "sample_rate": "48000",
        "n_fft": "960",
        "hop_length": "480",
        "window_length": "960",
        "freq_bins": "481",
        "window_type": "vorbis",
        "state_size": "56436",
        "erb_norm_state_size": "481",
        "spec_norm_state_size": "96",
        "spec_bins": "96",
        "erb_bins": "32",
        "profile": "dpdfnet2_48khz_hr",
        "model_type": "dpdfnet",
        "normalized": "0",
        "version": "1",
        "center": "1",
        "pad_mode": "reflect",
    }
    assert {
        key: runtime.metadata.get(key) for key in expected_metadata
    } == expected_metadata

    erb_init = np.fromstring(
        runtime.metadata["erb_norm_init"],
        sep=",",
        dtype=np.float32,
    )
    spec_init = np.fromstring(
        runtime.metadata["spec_norm_init"],
        sep=",",
        dtype=np.float32,
    )
    assert erb_init.shape == (dpdfnet2_48k_hr.ERB_NORM_STATE_SIZE,)
    assert spec_init.shape == (dpdfnet2_48k_hr.SPEC_NORM_STATE_SIZE,)
    assert runtime.init_state.shape == (dpdfnet2_48k_hr.STATE_SIZE,)
    assert runtime.init_state.dtype == np.float32
    assert np.isfinite(runtime.init_state).all()
    assert np.array_equal(
        runtime.init_state[: dpdfnet2_48k_hr.ERB_NORM_STATE_SIZE],
        erb_init,
    )
    assert np.array_equal(
        runtime.init_state[
            dpdfnet2_48k_hr.ERB_NORM_STATE_SIZE :
            dpdfnet2_48k_hr.ERB_NORM_STATE_SIZE
            + dpdfnet2_48k_hr.SPEC_NORM_STATE_SIZE
        ],
        spec_init,
    )
    normalizer_end = (
        dpdfnet2_48k_hr.ERB_NORM_STATE_SIZE
        + dpdfnet2_48k_hr.SPEC_NORM_STATE_SIZE
    )
    assert np.count_nonzero(runtime.init_state[normalizer_end:]) == 0
    assert dpdfnet2_48k_hr.OUTPUT_TRIM_SAMPLES == 1920
    assert dpdfnet2_48k_hr.ALGORITHM_LATENCY_SAMPLES == 1920


def test_48khz_mono_file_round_trip_preserves_length_and_finiteness(runtime):
    scratch = _scratch_dir("mono-io")
    waveform = _quantized_probe()
    source = scratch / "input-48k-mono.wav"
    target = scratch / "output-48k-mono.wav"
    sf.write(
        source,
        waveform,
        dpdfnet2_48k_hr.SAMPLE_RATE,
        subtype="FLOAT",
        format="WAV",
    )

    decoded = dpdfnet2_48k_hr.read_48k_mono(source)
    assert np.array_equal(decoded, waveform)

    enhanced, info = dpdfnet2_48k_hr.enhance_stream(
        decoded,
        runtime,
        chunk_size=dpdfnet2_48k_hr.HOP_SAMPLES,
        tail_frames=2,
    )
    assert enhanced.shape == waveform.shape
    assert enhanced.dtype == np.float32
    assert np.isfinite(enhanced).all()
    assert info.input_samples == waveform.size
    assert info.output_samples == waveform.size
    assert info.raw_samples == 2400
    assert info.frames_processed == 5
    assert info.trim_samples == 1920
    assert info.latency_samples == 1920
    assert np.isfinite(info.final_state_norm)

    dpdfnet2_48k_hr.write_48k_float(target, enhanced)
    written_info = sf.info(target)
    assert written_info.format == "WAV"
    assert written_info.samplerate == dpdfnet2_48k_hr.SAMPLE_RATE
    assert written_info.channels == 1
    assert written_info.subtype == "FLOAT"
    assert written_info.frames == waveform.size

    written, written_rate = sf.read(
        target,
        dtype="float32",
        always_2d=True,
    )
    assert written_rate == dpdfnet2_48k_hr.SAMPLE_RATE
    assert written.shape == (waveform.size, 1)
    assert np.isfinite(written).all()
    assert np.array_equal(written[:, 0], enhanced)


def test_flush_preserves_total_length_and_1920_sample_trim_alignment(runtime):
    del runtime
    fake_runtime = _fake_runtime()
    waveform = _quantized_probe(samples=960, seed=91)
    raw = _stream_raw(
        waveform,
        fake_runtime,
        chunk_size=dpdfnet2_48k_hr.HOP_SAMPLES,
        tail_frames=5,
    )
    aligned, info = dpdfnet2_48k_hr.enhance_stream(
        waveform,
        fake_runtime,
        chunk_size=dpdfnet2_48k_hr.HOP_SAMPLES,
        tail_frames=5,
    )

    assert raw.shape == (2880,)
    assert raw.dtype == np.float32
    assert np.isfinite(raw).all()
    assert info.raw_samples == raw.size
    assert info.trim_samples == 1920
    assert info.output_samples == waveform.size
    assert aligned.shape == waveform.shape
    assert np.isfinite(aligned).all()
    np.testing.assert_array_equal(
        aligned,
        raw[
            dpdfnet2_48k_hr.OUTPUT_TRIM_SAMPLES :
            dpdfnet2_48k_hr.OUTPUT_TRIM_SAMPLES + waveform.size
        ],
    )


def test_streaming_enhancer_instances_have_independent_reset_state():
    runtime = _fake_runtime()
    initial = runtime.init_state.copy()
    first = dpdfnet2_48k_hr.StreamingEnhancer(runtime)
    second = dpdfnet2_48k_hr.StreamingEnhancer(runtime)
    assert first.state is not second.state
    assert first.state is not runtime.init_state
    assert np.array_equal(first.state, initial)
    assert np.array_equal(second.state, initial)

    waveform = _quantized_probe(samples=dpdfnet2_48k_hr.WINDOW_SAMPLES)
    assert first.process(waveform).shape == (dpdfnet2_48k_hr.HOP_SAMPLES,)
    assert second.process(waveform).shape == (dpdfnet2_48k_hr.HOP_SAMPLES,)
    assert first.frames_processed == second.frames_processed == 1
    assert not np.array_equal(first.state, initial)
    assert not np.array_equal(second.state, initial)

    second_state_before_first_reset = second.state.copy()
    first.reset()
    assert np.array_equal(first.state, initial)
    assert np.array_equal(second.state, second_state_before_first_reset)
    assert first.input_buffer.size == 0
    assert first.emitted_samples == first.frames_processed == 0
    assert np.count_nonzero(first.ola_buffer) == 0

    first.state[0] += np.float32(7.0)
    assert second.state[0] == second_state_before_first_reset[0]

    second.reset()
    assert np.array_equal(first.state, runtime.init_state) is False
    assert np.array_equal(second.state, initial)
    assert np.array_equal(runtime.init_state, initial)


def test_stereo_input_is_downmixed_before_streaming():
    runtime = _fake_runtime()
    mono = _quantized_probe(samples=960, seed=77)
    stereo = np.stack([mono, -mono], axis=1)
    aligned, info = dpdfnet2_48k_hr.enhance_stream(
        stereo,
        runtime,
        chunk_size=dpdfnet2_48k_hr.HOP_SAMPLES,
        tail_frames=5,
    )
    assert info.input_samples == mono.size
    assert aligned.shape == mono.shape
    assert np.isfinite(aligned).all()


def test_nonfinite_model_output_is_rejected():
    runtime = _fake_runtime()
    runtime.session = _NaNSession()
    with pytest.raises(ValueError, match="NaN or infinity"):
        dpdfnet2_48k_hr.enhance_stream(
            _quantized_probe(samples=960, seed=78),
            runtime,
            chunk_size=dpdfnet2_48k_hr.HOP_SAMPLES,
        )
