"""CPU ONNX Runtime frame executor for the frozen CoFi-Lite model.

This module is an experimental candidate.  It does not modify the existing
stream implementation and does not silently fall back to Torch when ONNX
Runtime fails.
"""

from __future__ import annotations

import hashlib
import json
import threading
import weakref
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnxruntime as ort
import torch

from server.cofilite_frame import STATE_SHAPES, initial_frame_states
from server.cofilite_stream import (
    DELAY_SAMPLES,
    FFT_SIZE,
    FROZEN_CHECKPOINT_PATH,
    FROZEN_CHECKPOINT_SHA256,
    FRAME_SAMPLES,
    HOP_SAMPLES,
    LOOKAHEAD_SAMPLES,
    SAMPLE_RATE,
    CoFiLiteStream,
    _sha256,
    _validate_samples,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONNX_PATH = (
    ROOT / "build/cofilite-ort-20261117/cofilite-frame-v1-opset18.onnx"
)
DEFAULT_ONNX_SHA256 = (
    "f754898f6721c3180f53b602183db1275c09e08355b8ddff2b0b289181a7a2e0"
)
DEFAULT_ONNX_IDENTITY_SHA256 = (
    "cb2dce2870eaa6035c5409c3743baf4eef950e9fb2ccc251171b95578792c29a"
)
FROZEN_STREAM_SOURCE_SHA256 = (
    "3b7e08a21460d1f9195db180c2868e075045d642564806f1af3fee1fe39d4ae1"
)
FROZEN_FRAME_SOURCE_SHA256 = (
    "aad70af318a73d1010b0e8f92708b5137e167645a6e0e731fdda0b3f2b236e5a"
)


STATE_COUNT = len(STATE_SHAPES)
SPECTRUM_SHAPE = (1, FFT_SIZE // 2 + 1)
ENHANCED_SHAPE = (1, FFT_SIZE // 2 + 1, 1, 2)
INPUT_NAMES = (
    "spectrum_real",
    "spectrum_imag",
    *(f"state_{index:02d}" for index in range(STATE_COUNT)),
)
OUTPUT_NAMES = (
    "enhanced",
    *(f"next_state_{index:02d}" for index in range(STATE_COUNT)),
)
_EXECUTOR_CACHE: weakref.WeakKeyDictionary[
    torch.nn.Module, dict[tuple[Any, ...], OnnxFrameExecutor]
] = weakref.WeakKeyDictionary()
_EXECUTOR_CACHE_LOCK = threading.Lock()


def _sha256_path(path: Path) -> str:
    return _sha256(path)


def _shape_tuple(value: Iterable[Any]) -> tuple[int, ...]:
    result: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool):
            raise RuntimeError(f"ONNX shape is not fixed: {tuple(value)}")
        result.append(item)
    return tuple(result)


class OnnxFrameExecutor:
    """One fixed-shape ORT session exposed as a Torch-compatible callable."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        expected_model_sha256: str | None = DEFAULT_ONNX_SHA256,
        expected_identity_sha256: str | None = DEFAULT_ONNX_IDENTITY_SHA256,
        expected_weights_sha256: str | None = FROZEN_CHECKPOINT_SHA256,
    ):
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"ONNX frame model is missing: {path}")
        self.model_path = path
        self.model_sha256 = _sha256_path(path)
        if (
            expected_model_sha256 is not None
            and self.model_sha256 != expected_model_sha256
        ):
            raise RuntimeError(
                "ONNX frame model SHA-256 mismatch: "
                f"expected {expected_model_sha256}, got {self.model_sha256}"
            )
        self.weights_sha256 = _sha256_path(FROZEN_CHECKPOINT_PATH)
        if (
            expected_weights_sha256 is not None
            and self.weights_sha256 != expected_weights_sha256
        ):
            raise RuntimeError(
                "frozen CoFi-Lite checkpoint SHA-256 mismatch: "
                f"expected {expected_weights_sha256}, got {self.weights_sha256}"
            )
        self.identity_path = path.with_name(path.stem + ".identity.json")
        if not self.identity_path.is_file():
            raise FileNotFoundError(
                f"ONNX frame identity manifest is missing: {self.identity_path}"
            )
        self.identity_sha256 = _sha256_path(self.identity_path)
        identity = json.loads(self.identity_path.read_text(encoding="utf-8"))
        self._validate_identity(
            identity,
            expected_identity_sha256=expected_identity_sha256,
        )

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self.session = ort.InferenceSession(
            str(path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.providers = tuple(self.session.get_providers())
        if self.providers != ("CPUExecutionProvider",):
            raise RuntimeError(
                f"ONNX session did not use CPUExecutionProvider only: {self.providers}"
            )
        self._validate_session_contract()
        self._session_load_count = 1
        self._file_hash_count = 5
        self._call_count = 0
        self._noncontiguous_copy_count = 0
        self._lock = threading.Lock()
        self.metadata: dict[str, Any] = {
            "mode": "onnxruntime_cpu",
            "model_path": str(path),
            "model_sha256": self.model_sha256,
            "identity_path": str(self.identity_path),
            "identity_sha256": self.identity_sha256,
            "weights_sha256": self.weights_sha256,
            "stream_source_sha256": self.stream_source_sha256,
            "frame_source_sha256": self.frame_source_sha256,
            "provider": "CPUExecutionProvider",
            "execution_provider": self.providers[0],
            "ort_version": ort.__version__,
            "intra_op_num_threads": 1,
            "inter_op_num_threads": 1,
            "execution_mode": "ORT_SEQUENTIAL",
            "state_count": STATE_COUNT,
            "state_shapes": [list(shape) for shape in STATE_SHAPES],
            "input_names": list(INPUT_NAMES),
            "output_names": list(OUTPUT_NAMES),
            "session_load_count": self._session_load_count,
            "file_hash_count": self._file_hash_count,
        }

    def _validate_identity(
        self,
        identity: dict[str, Any],
        *,
        expected_identity_sha256: str | None,
    ) -> None:
        if (
            expected_identity_sha256 is not None
            and self.identity_sha256 != expected_identity_sha256
        ):
            raise RuntimeError(
                "ONNX identity SHA-256 mismatch: "
                f"expected {expected_identity_sha256}, got {self.identity_sha256}"
            )
        expected_model_path = str(self.model_path.resolve())
        if Path(identity.get("model_path", "")).resolve() != self.model_path.resolve():
            raise RuntimeError(
                f"ONNX identity model path mismatch: {identity.get('model_path')}"
            )
        if identity.get("model_sha256") != self.model_sha256:
            raise RuntimeError("ONNX identity model SHA-256 mismatch")
        if identity.get("weights_sha256") != self.weights_sha256:
            raise RuntimeError("ONNX identity weights SHA-256 mismatch")
        stream_path = ROOT / "server/cofilite_stream.py"
        frame_path = ROOT / "server/cofilite_frame.py"
        self.stream_source_sha256 = _sha256_path(stream_path)
        self.frame_source_sha256 = _sha256_path(frame_path)
        if identity.get("frozen_stream_source_sha256") != self.stream_source_sha256:
            raise RuntimeError("ONNX identity stream source SHA-256 mismatch")
        if identity.get("frozen_frame_source_sha256") != self.frame_source_sha256:
            raise RuntimeError("ONNX identity frame source SHA-256 mismatch")
        if identity.get("state_count") != STATE_COUNT:
            raise RuntimeError("ONNX identity state count mismatch")
        if identity.get("state_shapes") != [list(shape) for shape in STATE_SHAPES]:
            raise RuntimeError("ONNX identity state shapes mismatch")
        if identity.get("input_names") != list(INPUT_NAMES):
            raise RuntimeError("ONNX identity input names mismatch")
        if identity.get("output_names") != list(OUTPUT_NAMES):
            raise RuntimeError("ONNX identity output names mismatch")
        if identity.get("onnx_checker") != "pass":
            raise RuntimeError("ONNX identity does not record a passing checker")

    def _validate_session_contract(self) -> None:
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 2 + STATE_COUNT or len(outputs) != 1 + STATE_COUNT:
            raise RuntimeError(
                "unexpected ONNX input/output count: "
                f"inputs={len(inputs)}, outputs={len(outputs)}, "
                f"expected={2 + STATE_COUNT}/{1 + STATE_COUNT}"
            )
        if tuple(item.name for item in inputs) != INPUT_NAMES:
            raise RuntimeError(
                "unexpected ONNX input names: "
                f"{tuple(item.name for item in inputs)}"
            )
        if tuple(item.name for item in outputs) != OUTPUT_NAMES:
            raise RuntimeError(
                "unexpected ONNX output names: "
                f"{tuple(item.name for item in outputs)}"
            )
        expected_shapes = (SPECTRUM_SHAPE, SPECTRUM_SHAPE, *STATE_SHAPES)
        for item, expected_shape in zip(inputs, expected_shapes):
            actual_shape = _shape_tuple(item.shape)
            if actual_shape != expected_shape or item.type != "tensor(float)":
                raise RuntimeError(
                    f"unexpected ONNX input {item.name}: "
                    f"shape={actual_shape}, type={item.type}"
                )
        expected_output_shapes = (ENHANCED_SHAPE, *STATE_SHAPES)
        for item, expected_shape in zip(outputs, expected_output_shapes):
            actual_shape = _shape_tuple(item.shape)
            if actual_shape != expected_shape or item.type != "tensor(float)":
                raise RuntimeError(
                    f"unexpected ONNX output {item.name}: "
                    f"shape={actual_shape}, type={item.type}"
                )

    def _torch_to_numpy(
        self,
        tensor: torch.Tensor,
        *,
        name: str,
        expected_shape: tuple[int, ...],
    ) -> np.ndarray:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.device.type != "cpu":
            raise ValueError(f"{name} must be CPU, got {tensor.device}")
        if tensor.dtype != torch.float32:
            raise TypeError(f"{name} must be float32, got {tensor.dtype}")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{name} shape mismatch: expected {expected_shape}, "
                f"got {tuple(tensor.shape)}"
            )
        array = tensor.detach().numpy()
        if not array.flags.c_contiguous:
            self._noncontiguous_copy_count += 1
            array = np.ascontiguousarray(array)
        if not bool(np.isfinite(array).all()):
            raise ValueError(f"{name} contains NaN or Inf")
        return array

    def __call__(
        self,
        spectrum_real: torch.Tensor,
        spectrum_imag: torch.Tensor,
        *states: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if len(states) != STATE_COUNT:
            raise ValueError(
                f"expected {STATE_COUNT} state tensors, got {len(states)}"
            )
        with self._lock:
            feeds = {
                INPUT_NAMES[0]: self._torch_to_numpy(
                    spectrum_real,
                    name=INPUT_NAMES[0],
                    expected_shape=SPECTRUM_SHAPE,
                ),
                INPUT_NAMES[1]: self._torch_to_numpy(
                    spectrum_imag,
                    name=INPUT_NAMES[1],
                    expected_shape=SPECTRUM_SHAPE,
                ),
            }
            for index, (state, shape) in enumerate(zip(states, STATE_SHAPES)):
                name = INPUT_NAMES[index + 2]
                feeds[name] = self._torch_to_numpy(
                    state,
                    name=name,
                    expected_shape=shape,
                )
            self._call_count += 1
            try:
                raw_outputs = self.session.run(list(OUTPUT_NAMES), feeds)
            except Exception as exc:
                raise RuntimeError(
                    f"ONNX Runtime frame execution failed: {type(exc).__name__}: {exc}"
                ) from exc
            expected_shapes = (ENHANCED_SHAPE, *STATE_SHAPES)
            outputs: list[torch.Tensor] = []
            for name, shape, raw in zip(OUTPUT_NAMES, expected_shapes, raw_outputs):
                array = np.asarray(raw)
                if array.dtype != np.float32:
                    raise RuntimeError(
                        f"ONNX output {name} dtype mismatch: {array.dtype}"
                    )
                if tuple(array.shape) != shape:
                    raise RuntimeError(
                        f"ONNX output {name} shape mismatch: "
                        f"expected {shape}, got {tuple(array.shape)}"
                    )
                if not bool(np.isfinite(array).all()):
                    raise FloatingPointError(
                        f"ONNX output {name} contains NaN or Inf"
                    )
                outputs.append(torch.from_numpy(array))
            return tuple(outputs)


def get_onnx_frame_executor(
    model: torch.nn.Module,
    model_path: str | Path = DEFAULT_ONNX_PATH,
    *,
    expected_model_sha256: str | None = DEFAULT_ONNX_SHA256,
    expected_identity_sha256: str | None = DEFAULT_ONNX_IDENTITY_SHA256,
    expected_weights_sha256: str | None = FROZEN_CHECKPOINT_SHA256,
) -> OnnxFrameExecutor:
    """Load one ONNX Runtime session and return a Torch-compatible frame callable."""
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    key = (
        str(Path(model_path).resolve()),
        expected_model_sha256,
        expected_identity_sha256,
        expected_weights_sha256,
    )
    with _EXECUTOR_CACHE_LOCK:
        cached_by_key = _EXECUTOR_CACHE.get(model)
        if cached_by_key is None:
            cached_by_key = {}
            _EXECUTOR_CACHE[model] = cached_by_key
        cached = cached_by_key.get(key)
        if cached is not None:
            cached.metadata["cache_hit"] = True
            return cached
        executor = OnnxFrameExecutor(
            model_path,
            expected_model_sha256=expected_model_sha256,
            expected_identity_sha256=expected_identity_sha256,
            expected_weights_sha256=expected_weights_sha256,
        )
        executor.metadata["cache_hit"] = False
        cached_by_key[key] = executor
        return executor


class CoFiLiteOnnxStream(CoFiLiteStream):
    """Experimental CoFiLiteStream with only the frame executor replaced."""

    def __init__(
        self,
        model: torch.nn.Module | None = None,
        *,
        model_path: str | Path = DEFAULT_ONNX_PATH,
        expected_model_sha256: str | None = DEFAULT_ONNX_SHA256,
        expected_identity_sha256: str | None = DEFAULT_ONNX_IDENTITY_SHA256,
        expected_weights_sha256: str | None = FROZEN_CHECKPOINT_SHA256,
        frame_executor: OnnxFrameExecutor | None = None,
        **stream_kwargs: Any,
    ):
        super().__init__(model=model, jit=False, **stream_kwargs)
        if frame_executor is None:
            self._executor = get_onnx_frame_executor(
                self.model,
                model_path,
                expected_model_sha256=expected_model_sha256,
                expected_identity_sha256=expected_identity_sha256,
                expected_weights_sha256=expected_weights_sha256,
            )
        else:
            if not isinstance(frame_executor, OnnxFrameExecutor):
                raise TypeError("frame_executor must be an OnnxFrameExecutor")
            self._executor = frame_executor
        self._executor_info = dict(self._executor.metadata)

    @property
    def executor_info(self) -> dict[str, Any]:
        return dict(self._executor_info)


def enhance_stream_onnx(
    audio: np.ndarray,
    model_path: str | Path = DEFAULT_ONNX_PATH,
    chunk_size: int = FRAME_SAMPLES,
    *,
    model: torch.nn.Module | None = None,
    expected_model_sha256: str | None = DEFAULT_ONNX_SHA256,
    expected_identity_sha256: str | None = DEFAULT_ONNX_IDENTITY_SHA256,
    expected_weights_sha256: str | None = FROZEN_CHECKPOINT_SHA256,
    frame_samples: int = FRAME_SAMPLES,
    delay_samples: int = DELAY_SAMPLES,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the ONNX frame executor through the existing streaming schedule."""
    samples = _validate_samples(audio, name="audio")
    if int(samples.size) <= FFT_SIZE // 2:
        raise ValueError(
            "CoFi-Lite requires more than 256 input samples "
            "(upstream centered reflect STFT constraint)"
        )
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    stream = CoFiLiteOnnxStream(
        model=model,
        model_path=model_path,
        expected_model_sha256=expected_model_sha256,
        expected_identity_sha256=expected_identity_sha256,
        expected_weights_sha256=expected_weights_sha256,
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
        raise FloatingPointError("ONNX stream aligned output is invalid")
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
        "frame_executor": "onnxruntime_cpu",
        "frame_executor_model_path": str(stream._executor.model_path),
        "frame_executor_model_sha256": stream._executor.model_sha256,
        "frame_executor_session_load_count": stream._executor.metadata[
            "session_load_count"
        ],
        "frame_executor_file_hash_count": stream._executor.metadata[
            "file_hash_count"
        ],
    }
    return aligned, info


enhance_stream = enhance_stream_onnx
