"""Stateful, fixed-frame streaming adapter for the frozen CoFi-Lite model.

The model weights and upstream implementation remain unchanged.  This module
reimplements only the execution schedule: a 512-point centered STFT is emitted
from bounded causal history, recurrent state is carried one frame at a time,
and the centered ISTFT is drained with a two-frame rolling WOLA window.

``process()`` always consumes and returns exactly 160 mono float32 samples.
The output has a fixed 480-sample physical delay: the first three 10 ms
callbacks are zero, after which output sample ``n`` is the enhanced sample
``n - 480``.  ``flush()`` returns the final 480 delayed samples at EOS.
Consequently, ``concatenate(process outputs, flush())[480:]`` restores the
N-sample offline timeline.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Iterator

import numpy as np
import torch

from server.cofilite_frame import (
    DelayRingBuffer,
    InputRingBuffer,
    get_frame_executor,
    initial_frame_states,
)


ROOT = Path(__file__).resolve().parents[1]
FROZEN_ADAPTER_PATH = ROOT / "build/ns-cofilite-20261115/cofi_infer.py"
FROZEN_MODEL_PATH = ROOT / "build/ns-cofilite-20261115/upstream/model/cofilite.py"
FROZEN_CHECKPOINT_PATH = (
    ROOT / "build/ns-cofilite-20261115/upstream/checkpoint/model_on_dns3.tar"
)

FROZEN_ADAPTER_SHA256 = (
    "a885236e79a2df861d396f8d3cf0dafbf2e5d448513c96c3db30a68e3e0c274e"
)
FROZEN_MODEL_SHA256 = (
    "e77db8c40fbeef33b2496aca3f258d47205a1453c1bfb44c6b644f543c0afb9c"
)
FROZEN_CHECKPOINT_SHA256 = (
    "40d1bb69dcd553ce93c745c8bd136e1593397f529aa53609827c69df95ce2a30"
)

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 160
DELAY_SAMPLES = 480
HOP_SAMPLES = 256
FFT_SIZE = 512
LOOKAHEAD_SAMPLES = 511
DELAY_RING_CAPACITY = DELAY_SAMPLES + HOP_SAMPLES

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(path: Path, expected: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"frozen CoFi-Lite asset is missing: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"frozen CoFi-Lite asset SHA-256 mismatch: {path} "
            f"(expected {expected}, got {actual})"
        )


def _load_frozen_adapter() -> ModuleType:
    _require_sha256(FROZEN_ADAPTER_PATH, FROZEN_ADAPTER_SHA256)
    module_name = "_itu_frozen_cofilite_infer"
    spec = importlib.util.spec_from_file_location(module_name, FROZEN_ADAPTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen CoFi-Lite adapter: {FROZEN_ADAPTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _validate_model(model: torch.nn.Module) -> None:
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    for name, tensor in model.named_parameters():
        if tensor.device.type != "cpu":
            raise ValueError(f"model parameter {name} must be on CPU")
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            raise ValueError(f"model parameter {name} must be float32")
    for name, tensor in model.named_buffers():
        if tensor.device.type != "cpu":
            raise ValueError(f"model buffer {name} must be on CPU")
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            raise ValueError(f"model buffer {name} must be float32")


def load_model(device: str | torch.device = "cpu", threads: int = 1) -> torch.nn.Module:
    """Load the SHA-locked CPU CoFi-Lite checkpoint through the frozen adapter."""
    resolved = torch.device(device)
    if resolved.type != "cpu":
        raise ValueError(f"CoFi-Lite streaming supports CPU only, got {resolved}")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer")

    _require_sha256(FROZEN_MODEL_PATH, FROZEN_MODEL_SHA256)
    _require_sha256(FROZEN_CHECKPOINT_PATH, FROZEN_CHECKPOINT_SHA256)
    adapter = _load_frozen_adapter()
    model = adapter.load_model("cpu", threads=threads)
    _validate_model(model)
    return model


def _validate_samples(samples: np.ndarray, *, name: str = "samples") -> np.ndarray:
    array = np.asarray(samples)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if array.dtype != np.float32:
        raise TypeError(f"{name} must have dtype float32, got {array.dtype}")
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"{name} contains NaN or Inf")
    return np.ascontiguousarray(array)


class CoFiLiteStream:
    """One independent 16 kHz mono CoFi-Lite stream.

    A process should own one instance and never share it between concurrent
    audio streams.  The instance owns recurrent/convolution history, bounded
    STFT state, and the fixed-delay PCM queue; model weights are shared.
    """

    sample_rate = SAMPLE_RATE
    frame_samples = FRAME_SAMPLES
    delay_samples = DELAY_SAMPLES
    hop_samples = HOP_SAMPLES
    fft_size = FFT_SIZE
    lookahead_samples = LOOKAHEAD_SAMPLES

    def __init__(
        self,
        model: torch.nn.Module | None = None,
        *,
        jit: bool = True,
        frame_samples: int = FRAME_SAMPLES,
        delay_samples: int = DELAY_SAMPLES,
    ):
        if not isinstance(jit, bool):
            raise TypeError("jit must be a boolean")
        if (
            isinstance(frame_samples, bool)
            or not isinstance(frame_samples, int)
            or frame_samples < 1
            or frame_samples > HOP_SAMPLES
        ):
            raise ValueError(
                f"frame_samples must be an integer in [1, {HOP_SAMPLES}]"
            )
        if (
            isinstance(delay_samples, bool)
            or not isinstance(delay_samples, int)
            or delay_samples < 1
            or delay_samples % frame_samples
        ):
            raise ValueError(
                "delay_samples must be a positive integer multiple of frame_samples"
            )
        self.frame_samples = frame_samples
        self.delay_samples = delay_samples
        self.delay_ring_capacity = delay_samples + HOP_SAMPLES
        self.model = load_model("cpu", threads=1) if model is None else model
        _validate_model(self.model)
        self._input_ring = InputRingBuffer(FFT_SIZE)
        self._delay_ring = DelayRingBuffer(delay_samples, self.delay_ring_capacity)
        self._frame_scratch = torch.empty(FFT_SIZE, dtype=torch.float32)
        self._callback_output = torch.empty(frame_samples, dtype=torch.float32)
        self._flush_output = torch.empty(delay_samples, dtype=torch.float32)
        self._flush_zeros = torch.zeros(HOP_SAMPLES, dtype=torch.float32)
        self._istft_pair = torch.empty(
            1, FFT_SIZE // 2 + 1, 2, 2, dtype=torch.float32
        )
        self._previous_spectrum = torch.zeros(
            1, FFT_SIZE // 2 + 1, 1, 2, dtype=torch.float32
        )
        self._executor, executor_info = get_frame_executor(self.model, jit=jit)
        self._executor_info = dict(executor_info)
        self.reset()

    def reset(self) -> None:
        """Reset all audio and recurrent state for a new independent stream."""
        self._input_ring.reset()
        self._delay_ring.reset()
        self._previous_spectrum.zero_()
        self._has_previous_spectrum = False
        self._frame_states = initial_frame_states()
        self._total_input_samples = 0
        self._next_frame_index = 0
        self._inferences = 0
        self._finished = False

    @property
    def inferences(self) -> int:
        """Number of one-frame neural forward executions since reset."""
        return self._inferences

    @property
    def frames_processed(self) -> int:
        return self._inferences

    @property
    def cached_samples(self) -> int:
        """Audio samples retained in bounded input and delayed-output queues."""
        return int(self._input_ring.valid_samples + self._delay_ring.available)

    def _state_tensors(self) -> Iterator[torch.Tensor]:
        yield self._input_ring.data
        yield self._delay_ring.data
        yield self._previous_spectrum
        yield from self._frame_states

    @property
    def state_bytes(self) -> int:
        """Bytes of bounded per-stream state, excluding shared model weights."""
        return int(
            sum(tensor.numel() * tensor.element_size() for tensor in self._state_tensors())
        )

    @property
    def cache_bytes(self) -> int:
        return self.state_bytes

    def _check_runnable(self) -> None:
        if self._finished:
            raise RuntimeError("stream is finished; call reset() before processing more audio")
        if self.model.training:
            raise RuntimeError("model must be in eval mode for streaming inference")

    def _append_input_history(self, samples: torch.Tensor) -> None:
        self._input_ring.append(samples)

    def _advance_input(self, samples: torch.Tensor) -> None:
        offset = 0
        sample_count = int(samples.numel())
        while offset < sample_count:
            # Frame 0's left reflection contains input sample 256.  Later
            # frames need the right edge of their 512-sample centered window.
            frame_ready_at = (
                FFT_SIZE // 2 + 1
                if self._next_frame_index == 0
                else (self._next_frame_index + 1) * HOP_SAMPLES
            )
            needed = frame_ready_at - self._total_input_samples
            if needed > 0:
                take = min(needed, sample_count - offset)
                self._append_input_history(samples[offset : offset + take])
                self._total_input_samples += take
                offset += take
            if self._total_input_samples == frame_ready_at:
                self._process_normal_frame(self._next_frame_index)
                self._next_frame_index += 1

    def _spectrum(self, frame: torch.Tensor) -> torch.Tensor:
        if frame.ndim != 1 or int(frame.numel()) != FFT_SIZE:
            raise RuntimeError(f"internal STFT frame shape changed: {tuple(frame.shape)}")
        waveform = frame.to(dtype=torch.float32).unsqueeze(0)
        spectrum = torch.stft(
            waveform,
            n_fft=FFT_SIZE,
            hop_length=HOP_SAMPLES,
            window=self.model.stft_encoder.window,
            center=False,
            return_complex=True,
        )
        if tuple(spectrum.shape) != (1, FFT_SIZE // 2 + 1, 1):
            raise RuntimeError(f"unexpected one-frame STFT shape: {tuple(spectrum.shape)}")
        return spectrum[..., 0]

    def _process_normal_frame(self, frame_index: int) -> None:
        if frame_index == 0:
            if self._input_ring.valid_samples != FFT_SIZE // 2 + 1:
                raise RuntimeError("frame 0 input history is incomplete")
            history = self._input_ring.view(FFT_SIZE // 2 + 1)
            self._frame_scratch[:HOP_SAMPLES].copy_(
                torch.flip(history[1 : FFT_SIZE // 2 + 1], dims=(0,))
            )
            self._frame_scratch[HOP_SAMPLES:].copy_(history[:HOP_SAMPLES])
            frame = self._frame_scratch
        else:
            if self._input_ring.valid_samples != FFT_SIZE:
                raise RuntimeError(f"frame {frame_index} input history is incomplete")
            frame = self._input_ring.view(FFT_SIZE)
        self._process_frame(frame)

    def _append_delayed(self, samples: torch.Tensor) -> None:
        if samples.ndim != 1 or int(samples.numel()) != HOP_SAMPLES:
            raise RuntimeError(f"internal WOLA segment shape changed: {tuple(samples.shape)}")
        if not bool(torch.isfinite(samples).all()):
            raise FloatingPointError("streaming ISTFT produced NaN or Inf")
        self._delay_ring.append(samples)

    def _accept_spectrum(self, spectrum: torch.Tensor) -> None:
        if self._has_previous_spectrum:
            self._istft_pair[..., 0, :].copy_(self._previous_spectrum[..., 0, :])
            self._istft_pair[..., 1, :].copy_(spectrum[..., 0, :])
            segment = torch.istft(
                torch.view_as_complex(self._istft_pair),
                n_fft=FFT_SIZE,
                hop_length=HOP_SAMPLES,
                window=self.model.stft_decoder.window,
                center=True,
                length=HOP_SAMPLES,
            )
            if tuple(segment.shape) != (1, HOP_SAMPLES):
                raise RuntimeError(f"unexpected rolling ISTFT shape: {tuple(segment.shape)}")
            self._append_delayed(segment[0])
        self._previous_spectrum.copy_(spectrum)
        self._has_previous_spectrum = True

    def _process_frame(self, frame: torch.Tensor) -> None:
        """Run the fixed-state neural frame core for exactly one STFT frame."""
        with torch.inference_mode():
            spectrum = self._spectrum(frame)
            outputs = self._executor(
                spectrum.real, spectrum.imag, *self._frame_states
            )
            enhanced = outputs[0]
            if not bool(torch.isfinite(enhanced).all()):
                raise FloatingPointError(
                    "streaming model produced a non-finite spectrum"
                )
            self._frame_states = tuple(outputs[1:])
            self._inferences += 1
            self._accept_spectrum(enhanced)

    def _pop_delay_into(self, count: int, destination: torch.Tensor) -> None:
        if count < 1:
            raise ValueError("output count must be positive")
        self._delay_ring.pop_into(count, destination)
        if destination.dtype != torch.float32 or not bool(
            torch.isfinite(destination).all()
        ):
            raise FloatingPointError("streaming PCM output is not finite float32")

    def _pop_delay(self, count: int) -> np.ndarray:
        if count == self.frame_samples:
            destination = self._callback_output
        elif count == self.delay_samples:
            destination = self._flush_output
        else:
            output = np.empty(count, dtype=np.float32)
            self._pop_delay_into(count, torch.from_numpy(output))
            return output
        self._pop_delay_into(count, destination)
        return np.ascontiguousarray(destination.numpy().copy())

    def _consume_validated(self, samples: np.ndarray) -> np.ndarray:
        self._check_runnable()
        if int(samples.size) > self.frame_samples:
            raise ValueError(
                f"internal consume chunk exceeds configured frame size "
                f"{self.frame_samples}: {int(samples.size)}"
            )
        if int(samples.size) == self.frame_samples:
            self._advance_input(torch.from_numpy(samples))
            return self._pop_delay(self.frame_samples)
        raise ValueError(
            "partial frame input must be passed together with EOS to flush"
        )

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Consume and return exactly one configured fixed-size float32 frame."""
        samples = _validate_samples(frame, name="frame")
        if int(samples.size) != self.frame_samples:
            raise ValueError(
                f"frame must contain exactly {self.frame_samples} samples, "
                f"got {int(samples.size)}"
            )
        return self._consume_validated(samples)

    def _flush_impl(self, leading_samples: int = 0) -> tuple[np.ndarray, np.ndarray]:
        sample_count = self._total_input_samples
        if sample_count <= FFT_SIZE // 2:
            raise ValueError(
                "CoFi-Lite requires more than 256 samples before EOS "
                "(upstream centered reflect STFT constraint)"
            )

        final_frame_index = self._next_frame_index
        valid_model_samples = final_frame_index * HOP_SAMPLES
        if self._input_ring.valid_samples < HOP_SAMPLES + 1:
            raise RuntimeError("EOF reflect history is incomplete")
        remainder = sample_count - valid_model_samples
        actual_samples = HOP_SAMPLES + remainder
        actual = self._input_ring.view(actual_samples)
        self._frame_scratch[:actual_samples].copy_(actual)
        last_257 = self._input_ring.view(HOP_SAMPLES + 1)
        right_length = HOP_SAMPLES - remainder
        right = torch.flip(last_257, dims=(0,))[1 : right_length + 1]
        self._frame_scratch[actual_samples:].copy_(right)
        if int(self._frame_scratch.numel()) != FFT_SIZE:
            raise RuntimeError(
                f"EOF reflect frame length mismatch: {int(self._frame_scratch.numel())}"
            )
        self._process_frame(self._frame_scratch)
        self._next_frame_index += 1
        if remainder:
            self._delay_ring.append(self._flush_zeros[:remainder])

        if self._delay_ring.available != self.delay_samples + leading_samples:
            raise RuntimeError(
                "flush invariant failed: expected "
                f"{self.delay_samples + leading_samples} queued samples, "
                f"got {self._delay_ring.available}"
            )
        if leading_samples:
            leading = np.empty(leading_samples, dtype=np.float32)
            self._pop_delay_into(
                leading_samples, torch.from_numpy(leading)
            )
        else:
            leading = np.empty(0, dtype=np.float32)
        output = self._pop_delay(self.delay_samples)
        self._finished = True
        return leading, output

    def _finish_with_tail(self, tail: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Consume a final partial frame and return its PCM plus the EOS delay."""
        self._check_runnable()
        samples = _validate_samples(tail, name="tail")
        if int(samples.size) >= self.frame_samples:
            raise ValueError(
                f"tail must contain fewer than {self.frame_samples} samples, "
                f"got {int(samples.size)}"
            )
        if int(samples.size):
            self._advance_input(torch.from_numpy(samples))
        return self._flush_impl(int(samples.size))

    def flush(self) -> np.ndarray:
        """EOS-drain exactly the configured delay; reset is required afterwards."""
        self._check_runnable()
        _, output = self._flush_impl()
        return output


def enhance_stream(
    audio: np.ndarray,
    chunk_size: int = FRAME_SAMPLES,
    *,
    model: torch.nn.Module | None = None,
    jit: bool = True,
    frame_samples: int = FRAME_SAMPLES,
    delay_samples: int = DELAY_SAMPLES,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run stateful streaming on an in-memory waveform without whole replay.

    This helper exists for offline numerical comparison and benchmarks.  It
    feeds the same bounded stream in caller-sized chunks, never calls
    ``model.forward(audio)``, flushes the fixed 480-sample physical tail, and
    trims that delay to return the aligned N-sample output.
    """
    samples = _validate_samples(audio, name="audio")
    if int(samples.size) <= FFT_SIZE // 2:
        raise ValueError(
            "CoFi-Lite requires more than 256 input samples "
            "(upstream centered reflect STFT constraint)"
        )
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")

    stream = CoFiLiteStream(
        model=model,
        jit=jit,
        frame_samples=frame_samples,
        delay_samples=delay_samples,
    )
    outputs: list[np.ndarray] = []
    full_samples = (int(samples.size) // stream.frame_samples) * stream.frame_samples
    for offset in range(0, full_samples, stream.frame_samples):
        outputs.append(
            stream.process(samples[offset : offset + stream.frame_samples])
        )

    tail = samples[full_samples:]
    if int(tail.size):
        tail_output, flush_samples = stream._finish_with_tail(tail)
        outputs.append(tail_output)
    else:
        flush_samples = stream.flush()
    raw = np.concatenate((*outputs, flush_samples), dtype=np.float32)
    expected_raw = int(samples.size) + delay_samples
    if raw.shape != (expected_raw,):
        raise RuntimeError(
            f"stream raw length mismatch: expected {expected_raw}, got {raw.shape}"
        )
    aligned = np.ascontiguousarray(raw[delay_samples : delay_samples + samples.size])
    if aligned.shape != samples.shape or not bool(np.isfinite(aligned).all()):
        raise FloatingPointError("stream aligned output is invalid")

    valid_model_samples = (int(samples.size) // HOP_SAMPLES) * HOP_SAMPLES
    info = {
        "sample_rate": SAMPLE_RATE,
        "frame_samples": stream.frame_samples,
        "hop_samples": HOP_SAMPLES,
        "fft_size": FFT_SIZE,
        "delay_samples": stream.delay_samples,
        "trim_samples": stream.delay_samples,
        "lookahead_samples": LOOKAHEAD_SAMPLES,
        "stateful_streaming": True,
        "offline_whole_utterance": False,
        "whole_utterance_replay": False,
        "chunk_size": chunk_size,
        "input_samples": int(samples.size),
        "valid_model_samples": valid_model_samples,
        "zero_filled_tail_samples": int(samples.size) - valid_model_samples,
        "flush_samples": stream.delay_samples,
        "raw_samples": expected_raw,
        "aligned_samples": int(aligned.size),
        "inferences": stream.inferences,
        "frames_processed": stream.frames_processed,
        "state_bytes": stream.state_bytes,
        "cached_samples": stream.cached_samples,
        "input_ring_capacity": FFT_SIZE,
        "delay_ring_capacity": stream.delay_ring_capacity,
        "frame_executor": stream._executor_info["mode"],
        "frame_executor_trace_sec": stream._executor_info["trace_sec"],
    }
    return aligned, info
