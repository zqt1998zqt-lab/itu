"""Causal DPDFNet2 48 kHz HR adapter.

The upstream model is a stateful single-frame ONNX network:

    spec[1,1,481,2] + state[56436] -> spec_e[1,1,481,2] + state_out[56436]

This module implements the official causal 20 ms / 10 ms streaming schedule:
one 960-sample Vorbis window, a 480-sample hop, and a fresh state per stream.
It deliberately does not resample: the model consumes native 48 kHz PCM.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import onnxruntime as ort
import soundfile as sf


SAMPLE_RATE = 48_000
N_FFT = 960
WINDOW_SAMPLES = 960
HOP_SAMPLES = 480
FREQ_BINS = 481
STATE_SIZE = 56_436
ERB_NORM_STATE_SIZE = 481
SPEC_NORM_STATE_SIZE = 96
DF_LOOKAHEAD_FRAMES = 2
# Measured on the pinned model with an impulse and broadband-noise
# cross-correlation: one 960-sample analysis window plus two 480-sample
# internal lookahead frames.
ALGORITHM_LATENCY_SAMPLES = WINDOW_SAMPLES + DF_LOOKAHEAD_FRAMES * HOP_SAMPLES
OUTPUT_TRIM_SAMPLES = ALGORITHM_LATENCY_SAMPLES
TAIL_GUARD_FRAMES = 12
MODEL_SHA256 = "7f0575a5cec0ba4ffd8f8bd657e06d007e4ccdd955d76faab922b9d3291dc14b"


@dataclass
class RuntimeModel:
    session: ort.InferenceSession
    model_path: Path
    init_state: np.ndarray
    metadata: dict[str, str]
    model_sha256: str
    model_size_bytes: int


@dataclass
class StreamInfo:
    input_samples: int
    raw_samples: int
    output_samples: int
    frames_processed: int
    inference_sec: float
    wall_sec: float
    trim_samples: int
    latency_samples: int
    final_state_norm: float


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vorbis_window(window_len: int = WINDOW_SAMPLES) -> np.ndarray:
    half = window_len / 2.0
    indices = np.arange(window_len, dtype=np.float64)
    s = np.sin(0.5 * np.pi * (indices + 0.5) / half)
    return np.sin(0.5 * np.pi * s * s).astype(np.float32)


def _mono(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 2:
        if array.shape[1] < 1:
            raise ValueError("DPDFNet2 HR input has no channels")
        array = np.mean(array, axis=1, dtype=np.float32)
    elif array.ndim != 1:
        raise ValueError(f"DPDFNet2 HR input rank must be 1 or 2, got {array.ndim}")
    return np.ascontiguousarray(array.reshape(-1), dtype=np.float32)


def load_runtime(model_path: Path) -> RuntimeModel:
    model_path = model_path.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"DPDFNet2 48 kHz HR model not found: {model_path}")
    actual_sha = _sha256_file(model_path)
    if actual_sha != MODEL_SHA256:
        raise ValueError(
            f"DPDFNet2 48 kHz HR SHA mismatch: expected {MODEL_SHA256}, got {actual_sha}"
        )
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 2 or len(outputs) != 2:
        raise ValueError("DPDFNet2 HR must expose two inputs and two outputs")
    if tuple(inputs[0].shape) != (1, 1, FREQ_BINS, 2):
        raise ValueError(f"Unexpected HR spec shape: {inputs[0].shape}")
    if tuple(inputs[1].shape) != (STATE_SIZE,):
        raise ValueError(f"Unexpected HR state shape: {inputs[1].shape}")
    metadata = dict(session.get_modelmeta().custom_metadata_map)
    expected = {
        "sample_rate": str(SAMPLE_RATE),
        "n_fft": str(N_FFT),
        "hop_length": str(HOP_SAMPLES),
        "window_length": str(WINDOW_SAMPLES),
        "freq_bins": str(FREQ_BINS),
        "window_type": "vorbis",
        "state_size": str(STATE_SIZE),
        "erb_norm_state_size": str(ERB_NORM_STATE_SIZE),
        "spec_norm_state_size": str(SPEC_NORM_STATE_SIZE),
    }
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            raise ValueError(
                f"DPDFNet2 HR metadata mismatch for {key}: "
                f"expected {expected_value!r}, got {metadata.get(key)!r}"
            )
    try:
        erb_init = np.fromstring(
            metadata["erb_norm_init"], sep=",", dtype=np.float32
        )
        spec_init = np.fromstring(
            metadata["spec_norm_init"], sep=",", dtype=np.float32
        )
    except KeyError as error:
        raise ValueError(f"DPDFNet2 HR metadata missing {error}") from error
    if erb_init.size != ERB_NORM_STATE_SIZE or spec_init.size != SPEC_NORM_STATE_SIZE:
        raise ValueError("DPDFNet2 HR normalizer state metadata has wrong size")
    init_state = np.zeros(STATE_SIZE, dtype=np.float32)
    init_state[:ERB_NORM_STATE_SIZE] = erb_init
    init_state[
        ERB_NORM_STATE_SIZE : ERB_NORM_STATE_SIZE + SPEC_NORM_STATE_SIZE
    ] = spec_init
    return RuntimeModel(
        session=session,
        model_path=model_path,
        init_state=np.ascontiguousarray(init_state),
        metadata=metadata,
        model_sha256=actual_sha,
        model_size_bytes=model_path.stat().st_size,
    )


def read_48k_mono(path: Path) -> np.ndarray:
    path = path.expanduser().resolve()
    values, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if rate != SAMPLE_RATE or values.shape[1] != 1:
        raise ValueError(f"DPDFNet2 HR requires 48 kHz mono input: {path}")
    samples = np.ascontiguousarray(values[:, 0], dtype=np.float32)
    if not samples.size or not np.isfinite(samples).all():
        raise ValueError(f"DPDFNet2 HR input is empty or non-finite: {path}")
    return samples


def write_48k_float(path: Path, samples: np.ndarray) -> None:
    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("DPDFNet2 HR output is empty or non-finite")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), values, SAMPLE_RATE, subtype="FLOAT", format="WAV")


class StreamingEnhancer:
    """Official causal 48 kHz HR schedule with per-stream recurrent state."""

    def __init__(self, runtime: RuntimeModel) -> None:
        self.runtime = runtime
        self.window = vorbis_window()
        self.reset()

    def reset(self) -> None:
        self.state = np.ascontiguousarray(self.runtime.init_state.copy())
        self.input_buffer = np.zeros(0, dtype=np.float32)
        self.ola_buffer = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        self.inference_sec = 0.0
        self.frames_processed = 0
        self.emitted_samples = 0
        self._in_spec = self.runtime.session.get_inputs()[0].name
        self._in_state = self.runtime.session.get_inputs()[1].name
        self._out_spec = self.runtime.session.get_outputs()[0].name
        self._out_state = self.runtime.session.get_outputs()[1].name

    def process(self, chunk: np.ndarray) -> np.ndarray:
        chunk = _mono(chunk)
        if not chunk.size:
            return np.zeros(0, dtype=np.float32)
        if not np.isfinite(chunk).all():
            raise ValueError("DPDFNet2 HR input contains NaN or infinity")
        self.input_buffer = np.concatenate([self.input_buffer, chunk])
        committed: list[np.ndarray] = []
        while self.input_buffer.size >= WINDOW_SAMPLES:
            frame = self.input_buffer[:WINDOW_SAMPLES] * self.window
            spec_complex = np.fft.rfft(frame, n=WINDOW_SAMPLES)
            spec_ri = np.stack(
                [spec_complex.real.astype(np.float32),
                 spec_complex.imag.astype(np.float32)],
                axis=-1,
            )
            spec_t = np.ascontiguousarray(spec_ri[np.newaxis, np.newaxis, :, :])
            started = time.perf_counter()
            spec_e_t, state_out = self.runtime.session.run(
                [self._out_spec, self._out_state],
                {self._in_spec: spec_t, self._in_state: self.state},
            )
            self.inference_sec += time.perf_counter() - started
            if tuple(np.shape(spec_e_t)) != (1, 1, FREQ_BINS, 2):
                raise ValueError(f"Unexpected HR output spec shape: {np.shape(spec_e_t)}")
            if tuple(np.shape(state_out)) != (STATE_SIZE,):
                raise ValueError(f"Unexpected HR output state shape: {np.shape(state_out)}")
            if not np.isfinite(spec_e_t).all() or not np.isfinite(state_out).all():
                raise ValueError("DPDFNet2 HR model produced NaN or infinity")
            self.state = np.ascontiguousarray(state_out.astype(np.float32, copy=False))
            self.frames_processed += 1
            spec_e = spec_e_t[0, 0].astype(np.float32, copy=False)
            complex_e = spec_e[:, 0] + 1j * spec_e[:, 1]
            time_frame = (
                np.fft.irfft(complex_e, n=WINDOW_SAMPLES) * self.window
            ).astype(np.float32, copy=False)
            self.ola_buffer += time_frame
            committed.append(self.ola_buffer[:HOP_SAMPLES].copy())
            self.ola_buffer[:-HOP_SAMPLES] = self.ola_buffer[HOP_SAMPLES:]
            self.ola_buffer[-HOP_SAMPLES:] = 0.0
            self.input_buffer = self.input_buffer[HOP_SAMPLES:]
        if not committed:
            return np.zeros(0, dtype=np.float32)
        output = np.concatenate(committed).astype(np.float32, copy=False)
        self.emitted_samples += int(output.size)
        return output

    def flush(self, extra_frames: int = TAIL_GUARD_FRAMES) -> np.ndarray:
        if extra_frames < 0:
            raise ValueError("extra_frames must be non-negative")
        if not self.input_buffer.size or not extra_frames:
            return np.zeros(0, dtype=np.float32)
        return self.process(np.zeros(extra_frames * HOP_SAMPLES, dtype=np.float32))


def _fit_length(values: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size >= length:
        return values[:length].copy()
    return np.pad(values, (0, length - values.size)).astype(np.float32)


def enhance_stream(
    waveform: np.ndarray,
    runtime: RuntimeModel,
    *,
    chunk_size: int = HOP_SAMPLES,
    tail_frames: int = TAIL_GUARD_FRAMES,
) -> tuple[np.ndarray, StreamInfo]:
    values = _mono(waveform)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("DPDFNet2 HR input is empty or non-finite")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    wall_start = time.perf_counter()
    enhancer = StreamingEnhancer(runtime)
    chunks: list[np.ndarray] = []
    for start in range(0, values.size, chunk_size):
        chunks.append(enhancer.process(values[start : start + chunk_size]))
    chunks.append(enhancer.flush(tail_frames))
    raw = np.concatenate([chunk for chunk in chunks if chunk.size]).astype(np.float32)
    if not raw.size or not np.isfinite(raw).all():
        raise ValueError("DPDFNet2 HR raw output is empty or non-finite")
    trim = min(OUTPUT_TRIM_SAMPLES, raw.size)
    aligned = _fit_length(raw[trim:], values.size)
    if aligned.size != values.size or not np.isfinite(aligned).all():
        raise ValueError("DPDFNet2 HR aligned output is invalid")
    wall_sec = float(time.perf_counter() - wall_start)
    return aligned, StreamInfo(
        input_samples=int(values.size),
        raw_samples=int(raw.size),
        output_samples=int(aligned.size),
        frames_processed=int(enhancer.frames_processed),
        inference_sec=float(enhancer.inference_sec),
        wall_sec=wall_sec,
        trim_samples=int(trim),
        latency_samples=ALGORITHM_LATENCY_SAMPLES,
        final_state_norm=float(np.linalg.norm(enhancer.state)),
    )

