"""Official DPDFNet-2 8 kHz causal streaming adapter.

The ONNX asset exposes one streaming frame at a time.  This module applies the
official Vorbis analysis/synthesis path, maintains the model state and overlap
buffer, and returns an aligned IEEE FLOAT WAV without any fallback path.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnxruntime as ort
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = (
    ROOT / "build/ns-dpdfnet2-8khz-20261118/assets/dpdfnet2_8khz.onnx"
)
MODEL_PATH = DEFAULT_MODEL_PATH
MODEL_SHA256 = "6218f1dbd6e4bac5768c63b7d899fe7b84b3788f2a35c4e246d4ab0946165c5d"

SAMPLE_RATE = 8_000
MODEL_SR = SAMPLE_RATE
N_FFT = 160
WIN_LEN = 160
HOP = 80
HOP_LENGTH = HOP
FREQ_BINS = 81

MODEL_STATE_SIZE = 37_860
STATE_SIZE = MODEL_STATE_SIZE
MODEL_ERB_NORM_STATE_SIZE = 81
ERB_NORM_STATE_SIZE = MODEL_ERB_NORM_STATE_SIZE
MODEL_SPEC_NORM_STATE_SIZE = 80
SPEC_NORM_STATE_SIZE = MODEL_SPEC_NORM_STATE_SIZE

MODEL_DF_LOOKAHEAD_FRAMES = 2
DF_LOOKAHEAD_FRAMES = MODEL_DF_LOOKAHEAD_FRAMES
ALGORITHM_LATENCY_SAMPLES = WIN_LEN + MODEL_DF_LOOKAHEAD_FRAMES * HOP
OUTPUT_TRIM_SAMPLES = ALGORITHM_LATENCY_SAMPLES
TAIL_GUARD_FRAMES = 12

EXPECTED_METADATA = {
    "sample_rate": "8000",
    "n_fft": "160",
    "hop_length": "80",
    "window_length": "160",
    "freq_bins": "81",
    "window_type": "vorbis",
    "state_size": "37860",
    "erb_norm_state_size": "81",
    "spec_norm_state_size": "80",
    "spec_bins": "80",
    "erb_bins": "32",
    "profile": "dpdfnet2_8khz",
    "model_type": "dpdfnet",
    "normalized": "0",
    "version": "1",
    "center": "1",
    "pad_mode": "reflect",
}


@dataclass(frozen=True)
class RuntimeModel:
    session: ort.InferenceSession
    model_path: Path
    init_state: np.ndarray
    metadata: dict[str, str]
    model_sha256: str
    model_size_bytes: int


class StreamInfo(dict[str, Any]):
    """Dictionary result with the attribute access used by the 16 kHz reference."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_float_metadata(value: str, *, name: str) -> np.ndarray:
    try:
        parsed = np.asarray([float(item) for item in value.split(",")], dtype=np.float32)
    except ValueError as exc:
        raise ValueError(f"Model metadata {name!r} is not a float list") from exc
    if not parsed.size or not np.all(np.isfinite(parsed)):
        raise ValueError(f"Model metadata {name!r} is empty or non-finite")
    return parsed


def vorbis_window(window_len: int = WIN_LEN) -> np.ndarray:
    """Return the official DPDFNet Vorbis analysis/synthesis window."""
    if window_len <= 0:
        raise ValueError("window_len must be positive")
    half = window_len / 2.0
    indices = np.arange(window_len, dtype=np.float64)
    sine = np.sin(0.5 * np.pi * (indices + 0.5) / half)
    return np.sin(0.5 * np.pi * sine * sine).astype(np.float32)


def load_runtime(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    *,
    expected_sha256: str = MODEL_SHA256,
) -> RuntimeModel:
    """Load and fully validate the pinned 8 kHz streaming ONNX model."""
    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"DPDFNet-2 8 kHz ONNX model not found: {path}")

    model_sha256 = _sha256_file(path)
    if model_sha256 != expected_sha256:
        raise RuntimeError(
            "DPDFNet-2 8 kHz model SHA-256 mismatch: "
            f"expected {expected_sha256}, got {model_sha256}"
        )

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if hasattr(options, "use_deterministic_compute"):
        options.use_deterministic_compute = True

    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    providers = session.get_providers()
    if providers != ["CPUExecutionProvider"]:
        raise RuntimeError(f"Unexpected ONNX providers: {providers}")

    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 2 or len(outputs) != 2:
        raise ValueError(
            "Expected two streaming ONNX inputs and outputs; "
            f"got {len(inputs)} inputs and {len(outputs)} outputs"
        )
    expected_io = (
        (inputs[0].name, inputs[0].type, tuple(inputs[0].shape)),
        (inputs[1].name, inputs[1].type, tuple(inputs[1].shape)),
        (outputs[0].name, outputs[0].type, tuple(outputs[0].shape)),
        (outputs[1].name, outputs[1].type, tuple(outputs[1].shape)),
    )
    required_io = (
        ("spec", "tensor(float)", (1, 1, FREQ_BINS, 2)),
        ("state_in", "tensor(float)", (MODEL_STATE_SIZE,)),
        ("spec_e", "tensor(float)", (1, 1, FREQ_BINS, 2)),
        ("state_out", "tensor(float)", (MODEL_STATE_SIZE,)),
    )
    if expected_io != required_io:
        raise ValueError(
            "Unexpected DPDFNet-2 8 kHz ONNX signature: "
            f"expected {required_io!r}, got {expected_io!r}"
        )

    metadata = dict(session.get_modelmeta().custom_metadata_map)
    for key, expected in EXPECTED_METADATA.items():
        actual = metadata.get(key)
        if actual != expected:
            raise ValueError(
                f"Model metadata mismatch for {key}: "
                f"expected {expected!r}, got {actual!r}"
            )

    try:
        erb_init = _parse_float_metadata(
            metadata["erb_norm_init"], name="erb_norm_init"
        )
        spec_init = _parse_float_metadata(
            metadata["spec_norm_init"], name="spec_norm_init"
        )
    except KeyError as exc:
        raise ValueError(f"Model is missing required metadata key: {exc}") from exc
    if erb_init.size != MODEL_ERB_NORM_STATE_SIZE:
        raise ValueError(f"Unexpected ERB norm init size: {erb_init.size}")
    if spec_init.size != MODEL_SPEC_NORM_STATE_SIZE:
        raise ValueError(f"Unexpected spec norm init size: {spec_init.size}")

    init_state = np.zeros(MODEL_STATE_SIZE, dtype=np.float32)
    erb_end = MODEL_ERB_NORM_STATE_SIZE
    spec_end = erb_end + MODEL_SPEC_NORM_STATE_SIZE
    init_state[:erb_end] = erb_init
    init_state[erb_end:spec_end] = spec_init
    init_state = np.ascontiguousarray(init_state, dtype=np.float32)

    return RuntimeModel(
        session=session,
        model_path=path,
        init_state=init_state,
        metadata=metadata,
        model_sha256=model_sha256,
        model_size_bytes=path.stat().st_size,
    )


def read_8k_mono(path: str | Path) -> np.ndarray:
    """Read a non-empty finite mono 8 kHz WAV as contiguous float32 samples."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Input WAV not found: {resolved}")

    info = sf.info(str(resolved))
    if info.format != "WAV":
        raise ValueError(f"DPDFNet-2 8 kHz input must be a WAV file: {resolved}")
    if int(info.samplerate) != SAMPLE_RATE:
        raise ValueError(
            f"DPDFNet-2 8 kHz requires {SAMPLE_RATE} Hz input; "
            f"{resolved} is {info.samplerate} Hz"
        )
    if int(info.channels) != 1:
        raise ValueError(
            f"DPDFNet-2 8 kHz requires mono input; "
            f"{resolved} has {info.channels} channels"
        )
    if int(info.frames) <= 0:
        raise ValueError(f"Input WAV is empty: {resolved}")

    audio, sample_rate = sf.read(str(resolved), dtype="float32", always_2d=True)
    if int(sample_rate) != SAMPLE_RATE:
        raise ValueError(
            f"DPDFNet-2 8 kHz requires {SAMPLE_RATE} Hz input; "
            f"{resolved} is {sample_rate} Hz"
        )
    if audio.ndim != 2 or audio.shape[1] != 1 or audio.shape[0] == 0:
        raise ValueError(f"Input WAV has invalid mono geometry: {resolved}")
    waveform = np.ascontiguousarray(audio[:, 0], dtype=np.float32)
    if not np.all(np.isfinite(waveform)):
        raise ValueError(f"Input WAV contains NaN or infinity: {resolved}")
    return waveform


def write_float_wav(path: str | Path, waveform: np.ndarray) -> None:
    """Write a mono 8 kHz IEEE FLOAT WAV and verify the serialized contract."""
    resolved = Path(path).expanduser().resolve()
    output = np.asarray(waveform, dtype=np.float32)
    if output.ndim != 1 or output.size == 0:
        raise ValueError("Refusing to write an empty or non-1-D output WAV")
    if not np.all(np.isfinite(output)):
        raise ValueError("Enhanced output contains NaN or infinity")
    output = np.ascontiguousarray(output, dtype=np.float32)

    resolved.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(resolved), output, SAMPLE_RATE, format="WAV", subtype="FLOAT")

    written = sf.info(str(resolved))
    if (
        written.format != "WAV"
        or int(written.samplerate) != SAMPLE_RATE
        or int(written.channels) != 1
        or written.subtype != "FLOAT"
        or int(written.frames) != int(output.size)
    ):
        raise RuntimeError(f"Written output WAV violates the FLOAT contract: {resolved}")


def _validate_waveform(waveform: np.ndarray) -> np.ndarray:
    samples = np.asarray(waveform, dtype=np.float32)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("DPDFNet-2 8 kHz input must be a non-empty 1-D waveform")
    if not np.all(np.isfinite(samples)):
        raise ValueError("DPDFNet-2 8 kHz input contains NaN or infinity")
    return np.ascontiguousarray(samples, dtype=np.float32)


class StreamingEnhancer:
    """Stateful causal DPDFNet-2 8 kHz streaming core."""

    def __init__(self, runtime: RuntimeModel) -> None:
        if not isinstance(runtime, RuntimeModel):
            raise TypeError("runtime must be a RuntimeModel")
        self.runtime = runtime
        self.window = vorbis_window(WIN_LEN)
        self.state = np.ascontiguousarray(runtime.init_state.copy(), dtype=np.float32)
        self.input_buffer = np.zeros(0, dtype=np.float32)
        self.ola_buffer = np.zeros(WIN_LEN, dtype=np.float32)
        self.inference_sec = 0.0
        self.frames_processed = 0
        self.emitted_samples = 0
        self._in_spec = runtime.session.get_inputs()[0].name
        self._in_state = runtime.session.get_inputs()[1].name
        self._out_spec = runtime.session.get_outputs()[0].name
        self._out_state = runtime.session.get_outputs()[1].name

    def reset(self) -> None:
        self.state = np.ascontiguousarray(
            self.runtime.init_state.copy(), dtype=np.float32
        )
        self.input_buffer = np.zeros(0, dtype=np.float32)
        self.ola_buffer = np.zeros(WIN_LEN, dtype=np.float32)
        self.inference_sec = 0.0
        self.frames_processed = 0
        self.emitted_samples = 0

    def process(self, chunk: np.ndarray) -> np.ndarray:
        samples = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return np.zeros(0, dtype=np.float32)
        if not np.all(np.isfinite(samples)):
            raise ValueError("Streaming input contains NaN or infinity")

        self.input_buffer = np.concatenate([self.input_buffer, samples])
        committed: list[np.ndarray] = []

        while self.input_buffer.size >= WIN_LEN:
            frame = self.input_buffer[:WIN_LEN] * self.window
            spec_complex = np.fft.rfft(frame, n=WIN_LEN)
            spec_ri = np.stack(
                [
                    spec_complex.real.astype(np.float32),
                    spec_complex.imag.astype(np.float32),
                ],
                axis=-1,
            )
            spec_tensor = np.ascontiguousarray(
                spec_ri[np.newaxis, np.newaxis, :, :], dtype=np.float32
            )

            started = time.perf_counter()
            spec_e_tensor, state_out = self.runtime.session.run(
                [self._out_spec, self._out_state],
                {self._in_spec: spec_tensor, self._in_state: self.state},
            )
            self.inference_sec += time.perf_counter() - started

            next_state = np.asarray(state_out, dtype=np.float32)
            if next_state.shape != (MODEL_STATE_SIZE,) or not np.all(
                np.isfinite(next_state)
            ):
                raise FloatingPointError("ONNX model returned an invalid state")
            self.state = np.ascontiguousarray(next_state, dtype=np.float32)
            self.frames_processed += 1

            spec_e = np.asarray(spec_e_tensor, dtype=np.float32)
            if spec_e.shape != (1, 1, FREQ_BINS, 2) or not np.all(
                np.isfinite(spec_e)
            ):
                raise FloatingPointError("ONNX model returned an invalid spectrum")
            complex_e = spec_e[0, 0, :, 0] + 1j * spec_e[0, 0, :, 1]
            time_frame = (
                np.fft.irfft(complex_e, n=WIN_LEN) * self.window
            ).astype(np.float32, copy=False)
            self.ola_buffer += time_frame
            committed.append(self.ola_buffer[:HOP].copy())
            self.ola_buffer[:-HOP] = self.ola_buffer[HOP:]
            self.ola_buffer[-HOP:] = 0.0
            self.input_buffer = self.input_buffer[HOP:]

        if not committed:
            return np.zeros(0, dtype=np.float32)
        output = np.concatenate(committed).astype(np.float32, copy=False)
        self.emitted_samples += int(output.size)
        return output

    def flush_guard(self, extra_frames: int = TAIL_GUARD_FRAMES) -> np.ndarray:
        if isinstance(extra_frames, bool) or not isinstance(extra_frames, int):
            raise ValueError("extra_frames must be a non-negative integer")
        if extra_frames < 0:
            raise ValueError("extra_frames must be a non-negative integer")
        if self.input_buffer.size == 0 or extra_frames == 0:
            return np.zeros(0, dtype=np.float32)
        return self.process(np.zeros(extra_frames * HOP, dtype=np.float32))


def _concat_chunks(chunks: Iterable[np.ndarray]) -> np.ndarray:
    nonempty = [
        np.asarray(chunk, dtype=np.float32).reshape(-1)
        for chunk in chunks
        if np.asarray(chunk).size
    ]
    if not nonempty:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(nonempty).astype(np.float32, copy=False)


def stream_raw(
    waveform: np.ndarray,
    runtime: RuntimeModel,
    *,
    chunk_size: int = 4096,
    tail_frames: int = TAIL_GUARD_FRAMES,
) -> tuple[np.ndarray, StreamingEnhancer]:
    samples = _validate_waveform(waveform)
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if isinstance(tail_frames, bool) or not isinstance(tail_frames, int) or tail_frames < 0:
        raise ValueError("tail_frames must be a non-negative integer")

    enhancer = StreamingEnhancer(runtime)
    chunks: list[np.ndarray] = []
    for start in range(0, samples.size, chunk_size):
        chunks.append(enhancer.process(samples[start : start + chunk_size]))
    chunks.append(enhancer.flush_guard(tail_frames))
    raw = _concat_chunks(chunks)
    if raw.size == 0 or not np.all(np.isfinite(raw)):
        raise FloatingPointError("DPDFNet-2 8 kHz streaming output is empty or non-finite")
    return raw, enhancer


def _fit_length(samples: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    if values.size >= length:
        return np.ascontiguousarray(values[:length], dtype=np.float32)
    return np.pad(values, (0, length - values.size)).astype(np.float32, copy=False)


def enhance_stream(
    waveform: np.ndarray,
    runtime: RuntimeModel,
    *,
    chunk_size: int = 4096,
    tail_frames: int = TAIL_GUARD_FRAMES,
) -> tuple[np.ndarray, StreamInfo]:
    """Return a finite aligned 8 kHz waveform and complete streaming metadata."""
    samples = _validate_waveform(waveform)
    wall_started = time.perf_counter()
    raw, enhancer = stream_raw(
        samples, runtime, chunk_size=chunk_size, tail_frames=tail_frames
    )
    trim = min(OUTPUT_TRIM_SAMPLES, int(raw.size))
    available_after_trim = int(raw.size) - trim
    padded_samples = max(0, int(samples.size) - available_after_trim)
    cropped_samples = max(0, available_after_trim - int(samples.size))
    aligned = _fit_length(raw[trim:], int(samples.size))
    if aligned.shape != samples.shape or not np.all(np.isfinite(aligned)):
        raise FloatingPointError("DPDFNet-2 8 kHz aligned output is invalid")
    wall_sec = time.perf_counter() - wall_started
    state_norm = float(np.linalg.norm(enhancer.state))

    info = StreamInfo(
        {
            "sample_rate": SAMPLE_RATE,
            "model_path": str(runtime.model_path),
            "model_sha256": runtime.model_sha256,
            "model_sha": runtime.model_sha256,
            "model_size_bytes": runtime.model_size_bytes,
            "metadata": dict(runtime.metadata),
            "input_samples": int(samples.size),
            "raw_samples": int(raw.size),
            "output_samples": int(aligned.size),
            "aligned_samples": int(aligned.size),
            "frames_processed": int(enhancer.frames_processed),
            "frames": int(enhancer.frames_processed),
            "trim_samples": int(trim),
            "trim": int(trim),
            "padded_samples": padded_samples,
            "cropped_samples": cropped_samples,
            "remaining_delay_samples": 0,
            "latency_samples": ALGORITHM_LATENCY_SAMPLES,
            "latency": ALGORITHM_LATENCY_SAMPLES,
            "final_state_norm": state_norm,
            "state_norm": state_norm,
            "inference_sec": float(enhancer.inference_sec),
            "wall_sec": float(wall_sec),
            "chunk_size": int(chunk_size),
            "tail_frames": int(tail_frames),
            "output_subtype": "FLOAT",
        }
    )
    return aligned, info


def enhance_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    runtime: RuntimeModel | None = None,
    chunk_size: int = 4096,
    tail_frames: int = TAIL_GUARD_FRAMES,
) -> dict[str, Any]:
    """Enhance one 8 kHz mono WAV and write an equal-length FLOAT WAV."""
    source = Path(input_path).expanduser().resolve()
    target = Path(output_path).expanduser().resolve()
    if source == target:
        raise ValueError("input_path and output_path must be different files")

    waveform = read_8k_mono(source)
    selected_runtime = (
        load_runtime(model_path) if runtime is None else runtime
    )
    if not isinstance(selected_runtime, RuntimeModel):
        raise TypeError("runtime must be a RuntimeModel or None")

    aligned, info = enhance_stream(
        waveform,
        selected_runtime,
        chunk_size=chunk_size,
        tail_frames=tail_frames,
    )
    write_float_wav(target, aligned)
    written = read_8k_mono(target)
    if written.shape != aligned.shape or not np.array_equal(written, aligned):
        raise RuntimeError("written FLOAT WAV differs from the aligned model output")
    if _sha256_file(selected_runtime.model_path) != selected_runtime.model_sha256:
        raise RuntimeError("DPDFNet-2 8 kHz model asset changed during inference")

    info = StreamInfo(info)
    info.update(
        {
            "input_path": str(source),
            "output_path": str(target),
            "output_format": "WAV",
            "output_subtype": "FLOAT",
            "model_sha": selected_runtime.model_sha256,
        }
    )
    return info


__all__ = [
    "ALGORITHM_LATENCY_SAMPLES",
    "DEFAULT_MODEL_PATH",
    "DF_LOOKAHEAD_FRAMES",
    "ERB_NORM_STATE_SIZE",
    "EXPECTED_METADATA",
    "FREQ_BINS",
    "HOP",
    "HOP_LENGTH",
    "MODEL_DF_LOOKAHEAD_FRAMES",
    "MODEL_ERB_NORM_STATE_SIZE",
    "MODEL_PATH",
    "MODEL_SHA256",
    "MODEL_SPEC_NORM_STATE_SIZE",
    "MODEL_SR",
    "MODEL_STATE_SIZE",
    "N_FFT",
    "OUTPUT_TRIM_SAMPLES",
    "RuntimeModel",
    "SAMPLE_RATE",
    "SPEC_NORM_STATE_SIZE",
    "STATE_SIZE",
    "StreamInfo",
    "StreamingEnhancer",
    "TAIL_GUARD_FRAMES",
    "WIN_LEN",
    "enhance_file",
    "enhance_stream",
    "load_runtime",
    "read_8k_mono",
    "stream_raw",
    "vorbis_window",
    "write_float_wav",
]
