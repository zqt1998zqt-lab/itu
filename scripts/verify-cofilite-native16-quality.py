#!/usr/bin/env python3
"""Run the frozen 74-case NS quality matrix on native 256-sample CoFi-Lite.

The existing optimized driver remains the orchestration and scoring source of
truth.  This wrapper only supplies a versioned candidate runtime:

* the original ``server/cofilite_stream.py`` source is loaded into an isolated
  module namespace with exactly two constants changed:
  ``FRAME_SAMPLES 160 -> 256`` and ``DELAY_SAMPLES 480 -> 256``;
* the original ``server/cofilite_onnx.py`` source is loaded into another
  namespace and redirected to the transformed stream/frame modules;
* the original files remain byte-for-byte untouched and SHA-locked;
* the existing ONNX graph and checkpoint are the only model artifacts used.

Each case is aligned against both the frozen old offline CoFi output and the
already frozen current 480-sample ORT output.  The native candidate passes only
when both aligned comparisons have maximum absolute difference below ``1e-5``.
The frozen DPDFNet v1 reference keeps its original 640-sample delay.  Candidate
audio is already trimmed, so its scoring fixed delay is zero.

All new evidence is restricted to:

  build/native16-20261117/quality74

This is an NS-only regression.  It is not a full-VCP run and does not establish
device real-time behavior.
"""

from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True

from pathlib import Path

PROJECT_ROOT = Path("/home/zhangqingtian/project/itu")
EXPECTED_OUTPUT_ROOT = PROJECT_ROOT / "build/native16-20261117/quality74"
DRIVER_TEST_ROOT = (
    PROJECT_ROOT / "build/native16-20261117/quality74-driver-tests"
)

BASE_DRIVER = PROJECT_ROOT / "scripts/verify-cofilite-optimized-quality.py"
LEGACY_VALIDATOR = PROJECT_ROOT / "scripts/verify-cofilite-stream-quality.py"
ONNX_REFERENCE_DRIVER = (
    PROJECT_ROOT / "scripts/verify-cofilite-onnx-quality.py"
)

STREAM_SOURCE = PROJECT_ROOT / "server/cofilite_stream.py"
FRAME_SOURCE = PROJECT_ROOT / "server/cofilite_frame.py"
ONNX_SOURCE = PROJECT_ROOT / "server/cofilite_onnx.py"

ONNX_GRAPH = (
    PROJECT_ROOT
    / "build/cofilite-ort-20261117/cofilite-frame-v1-opset18.onnx"
)
ONNX_IDENTITY = ONNX_GRAPH.with_suffix(".identity.json")
ONNX_CHECKER = ONNX_GRAPH.with_suffix(".checker.json")
FROZEN_CHECKPOINT = (
    PROJECT_ROOT
    / "build/ns-cofilite-20261115/upstream/checkpoint/model_on_dns3.tar"
)
FROZEN_ADAPTER = PROJECT_ROOT / "build/ns-cofilite-20261115/cofi_infer.py"
FROZEN_MODEL_SOURCE = (
    PROJECT_ROOT
    / "build/ns-cofilite-20261115/upstream/model/cofilite.py"
)

CURRENT_480_ROOT = (
    PROJECT_ROOT / "build/cofilite-ort-20261117/quality74"
)
CURRENT_480_PARITY_RESULTS = CURRENT_480_ROOT / "parity-results.jsonl"

EXPECTED_BASE_DRIVER_SHA256 = (
    "06be78424a5df760e38fa1ac177c9ad94e8569ba590b693d3aa14a35c952796d"
)
EXPECTED_LEGACY_VALIDATOR_SHA256 = (
    "5c997394a24dfda5f5f558557c433d0b02f453b8f124b331b7453ae841f3dac2"
)
EXPECTED_ONNX_REFERENCE_DRIVER_SHA256 = (
    "21032ade28108269d20a58f94ddc249c35d6a55d31f52389357d5a063551ac96"
)
EXPECTED_STREAM_SOURCE_SHA256 = (
    "3b7e08a21460d1f9195db180c2868e075045d642564806f1af3fee1fe39d4ae1"
)
EXPECTED_FRAME_SOURCE_SHA256 = (
    "aad70af318a73d1010b0e8f92708b5137e167645a6e0e731fdda0b3f2b236e5a"
)
EXPECTED_ONNX_SOURCE_SHA256 = (
    "21d3072fbb0879912911c391534ae0e6d5d2276c6e440601c936f23a9733e682"
)
EXPECTED_GRAPH_SHA256 = (
    "f754898f6721c3180f53b602183db1275c09e08355b8ddff2b0b289181a7a2e0"
)
EXPECTED_GRAPH_IDENTITY_SHA256 = (
    "cb2dce2870eaa6035c5409c3743baf4eef950e9fb2ccc251171b95578792c29a"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "40d1bb69dcd553ce93c745c8bd136e1593397f529aa53609827c69df95ce2a30"
)
EXPECTED_ADAPTER_SHA256 = (
    "a885236e79a2df861d396f8d3cf0dafbf2e5d448513c96c3db30a68e3e0c274e"
)
EXPECTED_MODEL_SOURCE_SHA256 = (
    "e77db8c40fbeef33b2496aca3f258d47205a1453c1bfb44c6b644f543c0afb9c"
)

NATIVE_FRAME_SAMPLES = 256
NATIVE_DELAY_SAMPLES = 256
NATIVE_TRIM_SAMPLES = 256
NATIVE_SCORE_FIXED_DELAY_SAMPLES = 0
EXPECTED_HOP_SAMPLES = 256
EXPECTED_FFT_SIZE = 512
EXPECTED_LOOKAHEAD_SAMPLES = 511
EXPECTED_STATE_COUNT = 18
DPDF_REFERENCE_DELAY_SAMPLES = 640
EXPECTED_TOTAL = 74
EXPECTED_SPEECH = 68
EXPECTED_NOISE = 6
EXPECTED_GROUPS = {
    "v1speech48": 48,
    "v1clean16": 16,
    "v1continuous4": 4,
    "v1guards": 6,
}
EXPECTED_C4EB_PRESENT = 58
EXPECTED_C4EB_MISSING = 16
PARITY_MAX_ABS_THRESHOLD = 1e-5

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ["NUMBA_CACHE_DIR"] = str(EXPECTED_OUTPUT_ROOT / "cache/numba")
os.environ["MPLCONFIGDIR"] = str(EXPECTED_OUTPUT_ROOT / "cache/matplotlib")
os.environ["XDG_CACHE_HOME"] = str(EXPECTED_OUTPUT_ROOT / "cache/xdg")

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
from types import ModuleType
from typing import Any, Sequence

import numpy as np


_BASE: ModuleType | None = None
_ONNX_REFERENCE: ModuleType | None = None
_RUNTIME_BUNDLE: dict[str, Any] | None = None
_CONFIGURED = False

_ORIGINAL_WORKER_INITIALIZE: Any = None
_ORIGINAL_PROCESS_ONE_CASE: Any = None
_ORIGINAL_SUMMARIZE_PARITY: Any = None
_ORIGINAL_VALIDATE_PARITY: Any = None
_ORIGINAL_SCORE_ONE_CASE: Any = None
_ORIGINAL_BUILD_SOURCE_IDENTITIES: Any = None
_ORIGINAL_BUILD_REPORT: Any = None
_ORIGINAL_FINAL_VALIDATION: Any = None
_ORIGINAL_FINALIZE_PHASE: Any = None

_CURRENT_480_REFERENCES: dict[tuple[str, str], dict[str, Any]] | None = None
_CURRENT_480_SOURCE_SHA256: str | None = None


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise ValueError(f"missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def file_identity(path: Path, *, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{role} is missing: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "role": role,
    }


def require_file_sha(path: Path, expected: str, *, role: str) -> dict[str, Any]:
    identity = file_identity(path, role=role)
    if identity["sha256"] != expected:
        raise RuntimeError(
            f"{role} SHA mismatch: expected {expected}, "
            f"got {identity['sha256']}"
        )
    return identity


def load_module_file(name: str, path: Path) -> ModuleType:
    if not path.is_file():
        raise RuntimeError(f"module is missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_base_driver() -> ModuleType:
    global _BASE
    if _BASE is None:
        require_file_sha(
            BASE_DRIVER,
            EXPECTED_BASE_DRIVER_SHA256,
            role="base_driver",
        )
        require_file_sha(
            LEGACY_VALIDATOR,
            EXPECTED_LEGACY_VALIDATOR_SHA256,
            role="legacy_validator",
        )
        _BASE = load_module_file(
            "_itu_verify_cofilite_optimized_quality_native16_base",
            BASE_DRIVER,
        )
    return _BASE


def load_onnx_reference_driver() -> ModuleType:
    global _ONNX_REFERENCE
    if _ONNX_REFERENCE is None:
        require_file_sha(
            ONNX_REFERENCE_DRIVER,
            EXPECTED_ONNX_REFERENCE_DRIVER_SHA256,
            role="onnx_reference_driver",
        )
        _ONNX_REFERENCE = load_module_file(
            "_itu_verify_cofilite_onnx_quality_native16_reference",
            ONNX_REFERENCE_DRIVER,
        )
    return _ONNX_REFERENCE


def replace_exact(
    source: str,
    *,
    old: str,
    new: str,
    expected_count: int,
    label: str,
) -> tuple[str, dict[str, Any]]:
    count = source.count(old)
    if count != expected_count:
        raise RuntimeError(
            f"{label}: expected {expected_count} exact replacement(s), "
            f"found {count}"
        )
    return source.replace(old, new), {
        "label": label,
        "old": old,
        "new": new,
        "count": count,
    }


def execute_transformed_source(
    *,
    module_name: str,
    source_path: Path,
    source: str,
) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None:
        raise RuntimeError(f"cannot create module spec for {source_path}")
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(source_path)
    sys.modules[module_name] = module
    exec(compile(source, str(source_path), "exec"), module.__dict__)
    return module


def build_runtime_bundle() -> dict[str, Any]:
    """Load SHA-locked sources and create the native-256 runtime in isolation."""
    global _RUNTIME_BUNDLE
    if _RUNTIME_BUNDLE is not None:
        return _RUNTIME_BUNDLE

    stream_identity = require_file_sha(
        STREAM_SOURCE,
        EXPECTED_STREAM_SOURCE_SHA256,
        role="cofilite_stream_source",
    )
    frame_identity = require_file_sha(
        FRAME_SOURCE,
        EXPECTED_FRAME_SOURCE_SHA256,
        role="cofilite_frame_source",
    )
    onnx_identity = require_file_sha(
        ONNX_SOURCE,
        EXPECTED_ONNX_SOURCE_SHA256,
        role="cofilite_onnx_source",
    )

    frame_source_text = FRAME_SOURCE.read_text(encoding="utf-8")
    frame_module = execute_transformed_source(
        module_name="_itu_native16_cofilite_frame",
        source_path=FRAME_SOURCE,
        source=frame_source_text,
    )

    stream_source_text = STREAM_SOURCE.read_text(encoding="utf-8")
    stream_source_text, stream_import = replace_exact(
        stream_source_text,
        old="from server.cofilite_frame import (",
        new="from _itu_native16_cofilite_frame import (",
        expected_count=1,
        label="stream frame import",
    )
    stream_source_text, stream_frame = replace_exact(
        stream_source_text,
        old="FRAME_SAMPLES = 160\n",
        new="FRAME_SAMPLES = 256\n",
        expected_count=1,
        label="FRAME_SAMPLES",
    )
    stream_source_text, stream_delay = replace_exact(
        stream_source_text,
        old="DELAY_SAMPLES = 480\n",
        new="DELAY_SAMPLES = 256\n",
        expected_count=1,
        label="DELAY_SAMPLES",
    )
    stream_module = execute_transformed_source(
        module_name="_itu_native16_cofilite_stream",
        source_path=STREAM_SOURCE,
        source=stream_source_text,
    )

    onnx_source_text = ONNX_SOURCE.read_text(encoding="utf-8")
    onnx_source_text, onnx_frame_import = replace_exact(
        onnx_source_text,
        old="from server.cofilite_frame import STATE_SHAPES, initial_frame_states",
        new=(
            "from _itu_native16_cofilite_frame import "
            "STATE_SHAPES, initial_frame_states"
        ),
        expected_count=1,
        label="onnx frame import",
    )
    onnx_source_text, onnx_stream_import = replace_exact(
        onnx_source_text,
        old="from server.cofilite_stream import (",
        new="from _itu_native16_cofilite_stream import (",
        expected_count=1,
        label="onnx stream import",
    )
    onnx_module = execute_transformed_source(
        module_name="_itu_native16_cofilite_onnx",
        source_path=ONNX_SOURCE,
        source=onnx_source_text,
    )

    expected_stream_constants = {
        "SAMPLE_RATE": 16000,
        "FRAME_SAMPLES": NATIVE_FRAME_SAMPLES,
        "DELAY_SAMPLES": NATIVE_DELAY_SAMPLES,
        "HOP_SAMPLES": EXPECTED_HOP_SAMPLES,
        "FFT_SIZE": EXPECTED_FFT_SIZE,
        "LOOKAHEAD_SAMPLES": EXPECTED_LOOKAHEAD_SAMPLES,
    }
    actual_stream_constants = {
        name: int(getattr(stream_module, name))
        for name in expected_stream_constants
    }
    if actual_stream_constants != expected_stream_constants:
        raise RuntimeError(
            "transformed stream constants changed: "
            f"{actual_stream_constants} != {expected_stream_constants}"
        )
    expected_onnx_constants = {
        "SAMPLE_RATE": 16000,
        "FRAME_SAMPLES": NATIVE_FRAME_SAMPLES,
        "DELAY_SAMPLES": NATIVE_DELAY_SAMPLES,
        "HOP_SAMPLES": EXPECTED_HOP_SAMPLES,
        "FFT_SIZE": EXPECTED_FFT_SIZE,
        "LOOKAHEAD_SAMPLES": EXPECTED_LOOKAHEAD_SAMPLES,
        "STATE_COUNT": EXPECTED_STATE_COUNT,
    }
    actual_onnx_constants = {
        name: int(getattr(onnx_module, name))
        for name in expected_onnx_constants
    }
    if actual_onnx_constants != expected_onnx_constants:
        raise RuntimeError(
            "transformed ONNX constants changed: "
            f"{actual_onnx_constants} != {expected_onnx_constants}"
        )
    if int(frame_module.__dict__.get("STATE_COUNT", -1)) != EXPECTED_STATE_COUNT:
        frame_state_shapes = frame_module.STATE_SHAPES
        if len(frame_state_shapes) != EXPECTED_STATE_COUNT:
            raise RuntimeError("transformed frame state count is not 18")
    if len(onnx_module.STATE_SHAPES) != EXPECTED_STATE_COUNT:
        raise RuntimeError("transformed ONNX state count is not 18")

    enhance_signature = inspect.signature(onnx_module.enhance_stream_onnx)
    expected_defaults = {
        "chunk_size": NATIVE_FRAME_SAMPLES,
        "frame_samples": NATIVE_FRAME_SAMPLES,
        "delay_samples": NATIVE_DELAY_SAMPLES,
    }
    actual_defaults = {
        name: enhance_signature.parameters[name].default
        for name in expected_defaults
    }
    if actual_defaults != expected_defaults:
        raise RuntimeError(
            "transformed enhance_stream_onnx defaults changed: "
            f"{actual_defaults} != {expected_defaults}"
        )

    adapter = ModuleType("_itu_native16_runtime_adapter")
    adapter.__file__ = str(ONNX_SOURCE)
    for name in (
        "SAMPLE_RATE",
        "FRAME_SAMPLES",
        "DELAY_SAMPLES",
        "HOP_SAMPLES",
        "FFT_SIZE",
        "LOOKAHEAD_SAMPLES",
        "STATE_SHAPES",
        "STATE_COUNT",
        "CoFiLiteOnnxStream",
        "OnnxFrameExecutor",
    ):
        setattr(adapter, name, getattr(onnx_module, name))
    adapter._executor = "ort_cpu"

    def load_model(device: str = "cpu", threads: int = 1) -> Any:
        return stream_module.load_model(device, threads)

    def enhance_stream(
        audio: Any,
        chunk_size: int = NATIVE_FRAME_SAMPLES,
        *,
        model: Any = None,
    ) -> tuple[Any, dict[str, Any]]:
        if int(chunk_size) != NATIVE_FRAME_SAMPLES:
            raise ValueError(
                f"native-16 requires chunk_size={NATIVE_FRAME_SAMPLES}, "
                f"got {chunk_size}"
            )
        return onnx_module.enhance_stream_onnx(
            audio,
            ONNX_GRAPH,
            chunk_size=chunk_size,
            model=model,
            expected_model_sha256=EXPECTED_GRAPH_SHA256,
            expected_identity_sha256=EXPECTED_GRAPH_IDENTITY_SHA256,
            expected_weights_sha256=EXPECTED_CHECKPOINT_SHA256,
            frame_samples=NATIVE_FRAME_SAMPLES,
            delay_samples=NATIVE_DELAY_SAMPLES,
        )

    adapter.load_model = load_model
    adapter.enhance_stream = enhance_stream

    replacements = [
        stream_import,
        stream_frame,
        stream_delay,
        onnx_frame_import,
        onnx_stream_import,
    ]
    transformation = {
        "method": "exact_source_text_replacements_in_isolated_module_namespace",
        "replacements": replacements,
        "original_sources": {
            "stream": stream_identity,
            "frame": frame_identity,
            "onnx": onnx_identity,
        },
        "transformed_sources": {
            "stream_sha256": sha256_text(stream_source_text),
            "frame_sha256": sha256_text(frame_source_text),
            "onnx_sha256": sha256_text(onnx_source_text),
        },
        "stream_constants": actual_stream_constants,
        "onnx_constants": actual_onnx_constants,
        "enhance_stream_defaults": actual_defaults,
        "lookahead_note": (
            "LOOKAHEAD_SAMPLES remains the original FFT_SIZE/2-1 value 511"
        ),
    }
    _RUNTIME_BUNDLE = {
        "frame": frame_module,
        "stream": stream_module,
        "onnx": onnx_module,
        "adapter": adapter,
        "transformation": transformation,
    }
    return _RUNTIME_BUNDLE


def runtime_module() -> ModuleType:
    return build_runtime_bundle()["adapter"]


def import_candidate_stream_module() -> ModuleType:
    return runtime_module()


def candidate_constants() -> dict[str, int]:
    runtime = runtime_module()
    values = {
        "sample_rate": int(runtime.SAMPLE_RATE),
        "frame_samples": int(runtime.FRAME_SAMPLES),
        "delay_samples": int(runtime.DELAY_SAMPLES),
        "hop_samples": int(runtime.HOP_SAMPLES),
        "fft_size": int(runtime.FFT_SIZE),
        "lookahead_samples": int(runtime.LOOKAHEAD_SAMPLES),
    }
    expected = {
        "sample_rate": 16000,
        "frame_samples": NATIVE_FRAME_SAMPLES,
        "delay_samples": NATIVE_DELAY_SAMPLES,
        "hop_samples": EXPECTED_HOP_SAMPLES,
        "fft_size": EXPECTED_FFT_SIZE,
        "lookahead_samples": EXPECTED_LOOKAHEAD_SAMPLES,
    }
    if values != expected:
        raise RuntimeError(
            f"native-16 runtime constants changed: {values} != {expected}"
        )
    return values


def graph_identity() -> dict[str, Any]:
    graph = require_file_sha(
        ONNX_GRAPH,
        EXPECTED_GRAPH_SHA256,
        role="onnx_graph",
    )
    identity = require_file_sha(
        ONNX_IDENTITY,
        EXPECTED_GRAPH_IDENTITY_SHA256,
        role="onnx_graph_identity",
    )
    checker = file_identity(ONNX_CHECKER, role="onnx_graph_checker")
    declared = load_json(ONNX_IDENTITY)
    if declared.get("model_sha256") != EXPECTED_GRAPH_SHA256:
        raise RuntimeError("ONNX identity sidecar model SHA mismatch")
    if declared.get("weights_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("ONNX identity sidecar weight SHA mismatch")
    if int(declared.get("state_count", -1)) != EXPECTED_STATE_COUNT:
        raise RuntimeError("ONNX identity sidecar state count is not 18")
    if declared.get("ort_provider") != ["CPUExecutionProvider"]:
        raise RuntimeError("ONNX identity sidecar is not CPU-only")
    if checker.get("sha256") is None:
        raise RuntimeError("ONNX checker identity is incomplete")
    return {
        "graph": graph,
        "identity": identity,
        "checker": checker,
        "declared_state_count": EXPECTED_STATE_COUNT,
        "execution_provider": "CPUExecutionProvider",
    }


def checkpoint_identity() -> dict[str, Any]:
    return require_file_sha(
        FROZEN_CHECKPOINT,
        EXPECTED_CHECKPOINT_SHA256,
        role="frozen_checkpoint",
    )


def current_480_reference_data() -> tuple[
    dict[tuple[str, str], dict[str, Any]], str
]:
    global _CURRENT_480_REFERENCES
    global _CURRENT_480_SOURCE_SHA256
    if _CURRENT_480_REFERENCES is not None:
        if _CURRENT_480_SOURCE_SHA256 is None:
            raise RuntimeError("current-480 reference source SHA is missing")
        return _CURRENT_480_REFERENCES, _CURRENT_480_SOURCE_SHA256
    if not CURRENT_480_PARITY_RESULTS.is_file():
        raise RuntimeError(
            "current-480 parity reference is missing: "
            f"{CURRENT_480_PARITY_RESULTS}"
        )
    source_sha = sha256_file(CURRENT_480_PARITY_RESULTS)
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    with CURRENT_480_PARITY_RESULTS.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise RuntimeError(
                    f"{CURRENT_480_PARITY_RESULTS}:{line_number} "
                    "is not a JSON object"
                )
            key = (str(record["dataset"]), str(record["case_id"]))
            if key in rows:
                raise RuntimeError(f"duplicate current-480 reference key: {key}")
            info = record.get("stream_info")
            output = record.get("stream_output")
            if not isinstance(info, dict) or not isinstance(output, dict):
                raise RuntimeError(f"current-480 reference {key} lacks metadata")
            expected = {
                "frame_samples": 160,
                "delay_samples": 480,
                "trim_samples": 480,
                "chunk_size": 160,
                "hop_samples": EXPECTED_HOP_SAMPLES,
                "fft_size": EXPECTED_FFT_SIZE,
                "lookahead_samples": EXPECTED_LOOKAHEAD_SAMPLES,
            }
            actual = {name: int(info.get(name, -1)) for name in expected}
            if actual != expected:
                raise RuntimeError(
                    f"current-480 reference {key} runtime changed: "
                    f"{actual} != {expected}"
                )
            if int(output.get("physical_delay_samples", -1)) != 480:
                raise RuntimeError(
                    f"current-480 reference {key} physical delay is not 480"
                )
            if int(output.get("trim_samples", -1)) != 480:
                raise RuntimeError(
                    f"current-480 reference {key} trim is not 480"
                )
            if int(output.get("score_fixed_delay_samples", -1)) != 0:
                raise RuntimeError(
                    f"current-480 reference {key} score delay is not 0"
                )
            path = Path(str(output.get("path", "")))
            if not path.is_file():
                raise RuntimeError(
                    f"current-480 reference WAV is missing: {path}"
                )
            declared_sha = str(output.get("sha256", ""))
            actual_sha = sha256_file(path)
            if declared_sha != actual_sha:
                raise RuntimeError(
                    f"current-480 reference {key} WAV SHA mismatch"
                )
            rows[key] = record
    if len(rows) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"current-480 reference count {len(rows)} != {EXPECTED_TOTAL}"
        )
    _CURRENT_480_REFERENCES = rows
    _CURRENT_480_SOURCE_SHA256 = source_sha
    return rows, source_sha


def current_480_reference_for(
    dataset: str, case_id: str
) -> dict[str, Any]:
    rows, _ = current_480_reference_data()
    record = rows.get((str(dataset), str(case_id)))
    if record is None:
        raise RuntimeError(
            f"missing current-480 reference for {dataset}/{case_id}"
        )
    return record


def current_480_source_identity() -> dict[str, Any]:
    rows, source_sha = current_480_reference_data()
    return {
        "path": str(CURRENT_480_PARITY_RESULTS),
        "sha256": source_sha,
        "cases": len(rows),
        "frame_samples": 160,
        "delay_samples": 480,
        "trim_samples": 480,
        "score_fixed_delay_samples": 0,
    }


def runtime_execution_identity() -> dict[str, Any]:
    bundle = build_runtime_bundle()
    onnx = bundle["onnx"]
    return {
        "backend": "onnxruntime",
        "device": "cpu",
        "torch_jit_used": False,
        "gpu_used": False,
        "frame_executor": "onnxruntime_cpu",
        "executor_attribute": str(getattr(runtime_module(), "_executor", "")),
        "state_count": int(onnx.STATE_COUNT),
        "state_shapes": [list(shape) for shape in onnx.STATE_SHAPES],
        "candidate_constants": candidate_constants(),
    }


def execution_source_identity_native16() -> dict[str, Any]:
    bundle = build_runtime_bundle()
    files = {
        "driver": file_identity(Path(__file__).resolve(), role="driver"),
        "base_driver": require_file_sha(
            BASE_DRIVER,
            EXPECTED_BASE_DRIVER_SHA256,
            role="base_driver",
        ),
        "legacy_validator": require_file_sha(
            LEGACY_VALIDATOR,
            EXPECTED_LEGACY_VALIDATOR_SHA256,
            role="legacy_validator",
        ),
        "onnx_reference_driver": require_file_sha(
            ONNX_REFERENCE_DRIVER,
            EXPECTED_ONNX_REFERENCE_DRIVER_SHA256,
            role="onnx_reference_driver",
        ),
        "runtime_stream": require_file_sha(
            STREAM_SOURCE,
            EXPECTED_STREAM_SOURCE_SHA256,
            role="cofilite_stream_source",
        ),
        "runtime_frame": require_file_sha(
            FRAME_SOURCE,
            EXPECTED_FRAME_SOURCE_SHA256,
            role="cofilite_frame_source",
        ),
        "runtime_onnx": require_file_sha(
            ONNX_SOURCE,
            EXPECTED_ONNX_SOURCE_SHA256,
            role="cofilite_onnx_source",
        ),
        "frozen_adapter": require_file_sha(
            FROZEN_ADAPTER,
            EXPECTED_ADAPTER_SHA256,
            role="frozen_adapter",
        ),
        "frozen_model_source": require_file_sha(
            FROZEN_MODEL_SOURCE,
            EXPECTED_MODEL_SOURCE_SHA256,
            role="frozen_model_source",
        ),
        "frozen_checkpoint": checkpoint_identity(),
        "onnx_graph": graph_identity()["graph"],
        "onnx_graph_identity": graph_identity()["identity"],
        "onnx_graph_checker": graph_identity()["checker"],
    }
    return {
        "created_utc": utc_now(),
        "candidate_constants": candidate_constants(),
        "dpdf_reference_delay_samples": DPDF_REFERENCE_DELAY_SAMPLES,
        "files": files,
        "native16": {
            "transformation": bundle["transformation"],
            "runtime": runtime_execution_identity(),
            "onnx_graph": graph_identity(),
            "checkpoint": checkpoint_identity(),
            "current_480_reference": current_480_source_identity(),
        },
    }


def source_lock_signature(lock: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_constants": lock.get("candidate_constants"),
        "files": lock.get("files"),
        "native16": lock.get("native16"),
        "dpdf_reference_delay_samples": lock.get(
            "dpdf_reference_delay_samples"
        ),
    }


def ensure_source_lock_native16(
    output_root: Path, *, create: bool
) -> dict[str, Any]:
    path = output_root / "RUNTIME-SOURCE-LOCK.json"
    current = execution_source_identity_native16()
    if path.is_file():
        locked = load_json(path)
        if source_lock_signature(locked) != source_lock_signature(current):
            raise RuntimeError(
                "native-16 source or runtime changed after the parity source lock"
            )
        return locked
    if not create:
        raise FileNotFoundError(f"missing runtime source lock: {path}")
    legacy = configure_legacy()
    legacy.write_json_exclusive(path, current)
    return current


def validate_runtime_info(
    info: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    required = (
        "sample_rate",
        "frame_samples",
        "hop_samples",
        "fft_size",
        "delay_samples",
        "trim_samples",
        "lookahead_samples",
        "chunk_size",
        "input_samples",
        "valid_model_samples",
        "zero_filled_tail_samples",
        "flush_samples",
        "raw_samples",
        "aligned_samples",
        "inferences",
        "frames_processed",
        "state_bytes",
        "cached_samples",
        "input_ring_capacity",
        "delay_ring_capacity",
        "frame_executor",
        "frame_executor_model_path",
        "frame_executor_model_sha256",
    )
    missing = [name for name in required if name not in info]
    if missing:
        raise RuntimeError(f"{label}: runtime info lacks {missing}")

    input_samples = int(info["input_samples"])
    expected_inferences = (
        input_samples // NATIVE_FRAME_SAMPLES
    ) + 1
    expected = {
        "sample_rate": 16000,
        "frame_samples": NATIVE_FRAME_SAMPLES,
        "hop_samples": EXPECTED_HOP_SAMPLES,
        "fft_size": EXPECTED_FFT_SIZE,
        "delay_samples": NATIVE_DELAY_SAMPLES,
        "trim_samples": NATIVE_TRIM_SAMPLES,
        "lookahead_samples": EXPECTED_LOOKAHEAD_SAMPLES,
        "chunk_size": NATIVE_FRAME_SAMPLES,
        "aligned_samples": input_samples,
        "flush_samples": NATIVE_DELAY_SAMPLES,
        "raw_samples": input_samples + NATIVE_DELAY_SAMPLES,
        "inferences": expected_inferences,
        "frames_processed": expected_inferences,
        "valid_model_samples": (
            input_samples // EXPECTED_HOP_SAMPLES
        )
        * EXPECTED_HOP_SAMPLES,
        "input_ring_capacity": EXPECTED_FFT_SIZE,
        "delay_ring_capacity": (
            NATIVE_DELAY_SAMPLES + EXPECTED_HOP_SAMPLES
        ),
    }
    expected["zero_filled_tail_samples"] = (
        input_samples - expected["valid_model_samples"]
    )
    actual = {name: int(info[name]) for name in expected}
    if actual != expected:
        raise RuntimeError(
            f"{label}: runtime info mismatch: {actual} != {expected}"
        )
    if int(info["state_bytes"]) <= 0:
        raise RuntimeError(f"{label}: runtime state_bytes is not positive")
    if int(info["cached_samples"]) > (
        NATIVE_DELAY_SAMPLES + EXPECTED_FFT_SIZE
    ):
        raise RuntimeError(f"{label}: runtime cached_samples is unbounded")
    if info["frame_executor"] != "onnxruntime_cpu":
        raise RuntimeError(
            f"{label}: unexpected frame executor {info['frame_executor']!r}"
        )
    if Path(str(info["frame_executor_model_path"])).resolve() != (
        ONNX_GRAPH.resolve()
    ):
        raise RuntimeError(f"{label}: runtime reported an unexpected ONNX path")
    if info["frame_executor_model_sha256"] != EXPECTED_GRAPH_SHA256:
        raise RuntimeError(f"{label}: runtime reported an unexpected ONNX SHA")

    state_count = int(runtime_module().STATE_COUNT)
    if state_count != EXPECTED_STATE_COUNT:
        raise RuntimeError(f"{label}: runtime state count is not 18")
    return {
        "runtime_info_source": "enhance_stream_onnx_return_value",
        "frame_samples": actual["frame_samples"],
        "delay_samples": actual["delay_samples"],
        "trim_samples": actual["trim_samples"],
        "chunk_size": actual["chunk_size"],
        "lookahead_samples": actual["lookahead_samples"],
        "input_samples": input_samples,
        "aligned_samples": actual["aligned_samples"],
        "inferences": actual["inferences"],
        "state_count": state_count,
        "state_bytes": int(info["state_bytes"]),
        "frame_executor": info["frame_executor"],
        "onnx_graph_sha256": info["frame_executor_model_sha256"],
        "score_fixed_delay_samples": NATIVE_SCORE_FIXED_DELAY_SAMPLES,
        "all_runtime_fields_verified": True,
    }


def compare_aligned_files(
    *,
    candidate_path: Path,
    candidate_sha256: str,
    reference_path: Path,
    reference_sha256: str,
    label: str,
) -> dict[str, Any]:
    legacy = configure_legacy()
    if sha256_file(candidate_path) != candidate_sha256:
        raise RuntimeError(f"{label}: candidate WAV SHA changed")
    if sha256_file(reference_path) != reference_sha256:
        raise RuntimeError(f"{label}: reference WAV SHA changed")
    candidate, candidate_info = legacy.read_float_mono(
        candidate_path, f"{label} native-256"
    )
    reference, reference_info = legacy.read_float_mono(
        reference_path, f"{label} reference"
    )
    if candidate.shape != reference.shape:
        raise RuntimeError(
            f"{label}: aligned shapes differ "
            f"{candidate.shape} != {reference.shape}"
        )
    difference = np.abs(
        candidate.astype(np.float64) - reference.astype(np.float64)
    )
    max_abs = float(np.max(difference))
    rmse = float(np.sqrt(np.mean(difference * difference)))
    reference_rms = float(
        np.sqrt(np.mean(reference.astype(np.float64) ** 2))
    )
    relative_rmse = (
        rmse / reference_rms
        if reference_rms > 0.0
        else (0.0 if rmse == 0.0 else None)
    )
    return {
        "reference_path": str(reference_path),
        "reference_sha256": reference_sha256,
        "reference_info": reference_info,
        "samples": int(candidate.size),
        "max_abs": max_abs,
        "rmse": rmse,
        "relative_rmse": relative_rmse,
        "max_abs_index": int(np.argmax(difference)),
        "threshold": PARITY_MAX_ABS_THRESHOLD,
        "pass": max_abs < PARITY_MAX_ABS_THRESHOLD,
        "candidate_info": candidate_info,
    }


def worker_initialize_native16() -> None:
    import torch

    torch.set_num_threads(1)
    base = load_base_driver()
    base.apply_legacy_candidate_constants()
    legacy = base.legacy_module()
    legacy.import_stream_module = import_candidate_stream_module
    legacy.worker_initialize()


def process_one_stream_case_native16(
    payload: dict[str, Any],
) -> dict[str, Any]:
    base = load_base_driver()
    legacy = base.apply_legacy_candidate_constants()
    legacy.import_stream_module = import_candidate_stream_module
    if getattr(legacy, "_WORKER_MODEL", None) is None:
        worker_initialize_native16()

    record = legacy.process_one_stream_case(payload)
    info = record.get("stream_info")
    if not isinstance(info, dict):
        raise RuntimeError("native-16 stream info is missing")
    runtime_validation = validate_runtime_info(
        info,
        label=f"{record['group']}/{record['case_id']}",
    )
    info["driver_native16_runtime_verified"] = True
    info["driver_native16_runtime_source"] = (
        "transformed cofilite_stream/cofilite_onnx runtime info"
    )
    info["driver_native16_frame_samples"] = int(info["frame_samples"])
    info["driver_native16_delay_samples"] = int(info["delay_samples"])
    info["driver_native16_trim_samples"] = int(info["trim_samples"])
    info["driver_native16_score_fixed_delay_samples"] = (
        NATIVE_SCORE_FIXED_DELAY_SAMPLES
    )
    info["driver_native16_state_count"] = EXPECTED_STATE_COUNT
    info["driver_native16_onnx_graph_sha256"] = EXPECTED_GRAPH_SHA256
    info["driver_native16_torch_jit_used"] = False
    info["driver_native16_gpu_used"] = False
    record["native16_runtime_validation"] = runtime_validation

    stream_output = record.get("stream_output")
    if not isinstance(stream_output, dict):
        raise RuntimeError("native-16 stream output metadata is missing")
    if int(stream_output.get("samples", -1)) != int(info["aligned_samples"]):
        raise RuntimeError("native-16 output sample count does not match runtime")
    stream_output["physical_delay_samples"] = int(info["delay_samples"])
    stream_output["trim_samples"] = int(info["trim_samples"])
    stream_output["score_fixed_delay_samples"] = (
        NATIVE_SCORE_FIXED_DELAY_SAMPLES
    )
    stream_output["delay_note"] = (
        f"runtime reported frame={info['frame_samples']}, "
        f"physical delay={info['delay_samples']}, trim={info['trim_samples']}; "
        "the aligned FLOAT file is scored with fixed delay 0"
    )
    stream_output["runtime_observed_frame_samples"] = int(
        info["frame_samples"]
    )
    stream_output["runtime_observed_delay_samples"] = int(
        info["delay_samples"]
    )
    stream_output["runtime_observed_trim_samples"] = int(
        info["trim_samples"]
    )

    offline_parity = dict(record["parity"])
    offline_reference = record["old_offline_output"]
    offline_parity["reference_role"] = "frozen old offline CoFi output"
    offline_parity["reference_path"] = offline_reference["path"]
    offline_parity["reference_sha256"] = offline_reference["sha256"]
    offline_parity["pass"] = (
        float(offline_parity["max_abs"]) < PARITY_MAX_ABS_THRESHOLD
    )
    offline_parity["threshold"] = PARITY_MAX_ABS_THRESHOLD

    current_480 = current_480_reference_for(
        str(record["dataset"]), str(record["case_id"])
    )
    current_480_output = current_480["stream_output"]
    current_480_parity = compare_aligned_files(
        candidate_path=Path(stream_output["path"]),
        candidate_sha256=str(stream_output["sha256"]),
        reference_path=Path(current_480_output["path"]),
        reference_sha256=str(current_480_output["sha256"]),
        label=f"{record['group']}/{record['case_id']} current-480",
    )
    current_480_parity["reference_role"] = (
        "frozen current 480-sample ONNX Runtime output"
    )
    current_480_parity["reference_runtime"] = {
        "frame_samples": int(current_480["stream_info"]["frame_samples"]),
        "delay_samples": int(current_480["stream_info"]["delay_samples"]),
        "trim_samples": int(current_480["stream_info"]["trim_samples"]),
        "chunk_size": int(current_480["stream_info"]["chunk_size"]),
        "score_fixed_delay_samples": int(
            current_480_output["score_fixed_delay_samples"]
        ),
    }
    current_480_parity["pass"] = (
        float(current_480_parity["max_abs"]) < PARITY_MAX_ABS_THRESHOLD
    )

    selected = (
        offline_parity
        if float(offline_parity["max_abs"])
        >= float(current_480_parity["max_abs"])
        else current_480_parity
    )
    record["parity"] = {
        "samples": int(info["aligned_samples"]),
        "duration_s": float(info["aligned_samples"]) / 16000.0,
        "max_abs": max(
            float(offline_parity["max_abs"]),
            float(current_480_parity["max_abs"]),
        ),
        "rmse": max(
            float(offline_parity["rmse"]),
            float(current_480_parity["rmse"]),
        ),
        "relative_rmse": max(
            (
                float(offline_parity["relative_rmse"])
                if offline_parity.get("relative_rmse") is not None
                else 0.0
            ),
            (
                float(current_480_parity["relative_rmse"])
                if current_480_parity.get("relative_rmse") is not None
                else 0.0
            ),
        ),
        "max_abs_index": int(selected["max_abs_index"]),
        "threshold": PARITY_MAX_ABS_THRESHOLD,
        "pass": (
            float(offline_parity["max_abs"]) < PARITY_MAX_ABS_THRESHOLD
            and float(current_480_parity["max_abs"])
            < PARITY_MAX_ABS_THRESHOLD
        ),
        "old_offline": offline_parity,
        "current_480": current_480_parity,
    }
    record["candidate_execution_identity"] = payload["source_lock"]
    return record


def _nested_parity_summary(
    records: Sequence[dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    rows = [record["parity"][key] for record in records]
    return {
        "cases": len(rows),
        "pass_count": sum(bool(row["pass"]) for row in rows),
        "all_pass": all(bool(row["pass"]) for row in rows),
        "max_abs": max(float(row["max_abs"]) for row in rows),
        "max_rmse": max(float(row["rmse"]) for row in rows),
        "threshold": PARITY_MAX_ABS_THRESHOLD,
    }


def summarize_parity_native16(
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if _ORIGINAL_SUMMARIZE_PARITY is None:
        raise RuntimeError("base parity summarizer was not initialized")
    summary = _ORIGINAL_SUMMARIZE_PARITY(records)
    old_offline = _nested_parity_summary(records, "old_offline")
    current_480 = _nested_parity_summary(records, "current_480")
    runtime_checks = []
    for record in records:
        runtime_checks.append(
            validate_runtime_info(
                record["stream_info"],
                label=f"{record['group']}/{record['case_id']}",
            )
        )
    runtime_contract = {
        "cases": len(runtime_checks),
        "all_runtime_info_verified": all(
            row["all_runtime_fields_verified"] for row in runtime_checks
        ),
        "frame_samples_values": sorted(
            {int(row["frame_samples"]) for row in runtime_checks}
        ),
        "delay_samples_values": sorted(
            {int(row["delay_samples"]) for row in runtime_checks}
        ),
        "trim_samples_values": sorted(
            {int(row["trim_samples"]) for row in runtime_checks}
        ),
        "chunk_size_values": sorted(
            {int(row["chunk_size"]) for row in runtime_checks}
        ),
        "lookahead_samples_values": sorted(
            {int(row["lookahead_samples"]) for row in runtime_checks}
        ),
        "inferences_match_floor_input_over_256_plus_eos": all(
            int(row["inferences"])
            == (
                int(row["input_samples"]) // NATIVE_FRAME_SAMPLES
            )
            + 1
            for row in runtime_checks
        ),
        "state_count_values": sorted(
            {int(row["state_count"]) for row in runtime_checks}
        ),
        "score_fixed_delay_values": sorted(
            {
                int(row["score_fixed_delay_samples"])
                for row in runtime_checks
            }
        ),
    }
    candidate = dict(summary["candidate_execution"])
    candidate.update(
        {
            "all_runtime_info_verified": runtime_contract[
                "all_runtime_info_verified"
            ],
            "all_frame_256": runtime_contract["frame_samples_values"]
            == [NATIVE_FRAME_SAMPLES],
            "all_delay_256": runtime_contract["delay_samples_values"]
            == [NATIVE_DELAY_SAMPLES],
            "all_trim_256": runtime_contract["trim_samples_values"]
            == [NATIVE_TRIM_SAMPLES],
            "all_chunk_256": runtime_contract["chunk_size_values"]
            == [NATIVE_FRAME_SAMPLES],
            "all_lookahead_511": runtime_contract[
                "lookahead_samples_values"
            ]
            == [EXPECTED_LOOKAHEAD_SAMPLES],
            "all_state_count_18": runtime_contract["state_count_values"]
            == [EXPECTED_STATE_COUNT],
        }
    )
    summary["candidate_execution"] = candidate
    summary["all_pass"] = (
        bool(summary["all_pass"])
        and old_offline["all_pass"]
        and current_480["all_pass"]
        and all(candidate[name] for name in (
            "all_runtime_info_verified",
            "all_frame_256",
            "all_delay_256",
            "all_trim_256",
            "all_chunk_256",
            "all_lookahead_511",
            "all_state_count_18",
            "all_score_delay_zero",
        ))
    )
    summary["old_offline_parity"] = old_offline
    summary["current_480_parity"] = current_480
    summary["runtime_contract"] = runtime_contract
    summary["native16"] = {
        "callback_samples": NATIVE_FRAME_SAMPLES,
        "frame_samples": NATIVE_FRAME_SAMPLES,
        "delay_samples": NATIVE_DELAY_SAMPLES,
        "trim_samples": NATIVE_TRIM_SAMPLES,
        "score_fixed_delay_samples": NATIVE_SCORE_FIXED_DELAY_SAMPLES,
        "lookahead_samples": EXPECTED_LOOKAHEAD_SAMPLES,
        "old_offline_parity_max_abs": old_offline["max_abs"],
        "current_480_parity_max_abs": current_480["max_abs"],
        "runtime_metadata_source": "per-case enhance_stream_onnx info",
    }
    return summary


def validate_parity_native16(
    output_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if _ORIGINAL_VALIDATE_PARITY is None:
        raise RuntimeError("base parity validator was not initialized")
    records, summary = _ORIGINAL_VALIDATE_PARITY(output_root)
    if summary.get("old_offline_parity", {}).get("all_pass") is not True:
        raise RuntimeError("old-offline parity failed")
    if summary.get("current_480_parity", {}).get("all_pass") is not True:
        raise RuntimeError("current-480 parity failed")
    runtime_contract = summary.get("runtime_contract")
    if not isinstance(runtime_contract, dict):
        raise RuntimeError("native-16 runtime contract is missing")
    if runtime_contract.get("all_runtime_info_verified") is not True:
        raise RuntimeError("native-16 per-case runtime validation failed")
    ensure_source_lock_native16(output_root, create=False)
    return records, summary


def score_one_case_native16(
    case: dict[str, Any],
    parity_record: dict[str, Any],
    stage_tool: ModuleType,
    score_module: ModuleType,
    cache: Any,
    metric_cache: dict[str, Any],
    reference_data: dict[str, Any],
    c4eb_data: dict[str, Any],
    source_lock: dict[str, Any],
) -> dict[str, Any]:
    if _ORIGINAL_SCORE_ONE_CASE is None:
        raise RuntimeError("base scoring function was not initialized")
    runtime_validation = validate_runtime_info(
        parity_record["stream_info"],
        label=f"{parity_record['group']}/{parity_record['case_id']} score",
    )
    record = _ORIGINAL_SCORE_ONE_CASE(
        case,
        parity_record,
        stage_tool,
        score_module,
        cache,
        metric_cache,
        reference_data,
        c4eb_data,
    )
    info = parity_record["stream_info"]
    record["stream_output"]["physical_delay_samples"] = int(
        info["delay_samples"]
    )
    record["stream_output"]["trim_samples"] = int(info["trim_samples"])
    record["stream_output"]["score_fixed_delay_samples"] = (
        NATIVE_SCORE_FIXED_DELAY_SAMPLES
    )
    record["stream_output"]["delay_note"] = (
        f"runtime reported frame={info['frame_samples']}, "
        f"physical delay={info['delay_samples']}, trim={info['trim_samples']}; "
        "aligned candidate audio is scored with fixed delay 0"
    )
    record["scoring"]["candidate_physical_delay_samples"] = int(
        info["delay_samples"]
    )
    record["scoring"]["candidate_trim_samples"] = int(info["trim_samples"])
    record["scoring"]["candidate_fixed_delay_samples"] = (
        NATIVE_SCORE_FIXED_DELAY_SAMPLES
    )
    record["scoring"]["candidate_stage"]["measurement_domain"] = (
        "native-256 CoFi-Lite stateful stream output; runtime reported "
        f"frame={info['frame_samples']}, delay={info['delay_samples']}, "
        f"trim={info['trim_samples']}; score delay 0"
    )
    if int(record["dpdfnet2_reference"]["fixed_delay_samples"]) != (
        DPDF_REFERENCE_DELAY_SAMPLES
    ):
        raise RuntimeError("scored record changed the frozen DPDF delay")
    record["native16_runtime_validation"] = runtime_validation
    record["candidate_execution_identity"] = source_lock
    return record


def write_parity_supplement(
    output_root: Path,
    parity_summary: dict[str, Any],
) -> None:
    legacy = configure_legacy()
    path = output_root / "NATIVE16-PARITY-VALIDATION.json"
    value = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "pass" if parity_summary["all_pass"] else "fail",
        "scope": {
            "ns_only": True,
            "not_full_vcp": True,
            "device_realtime": False,
            "deployment_approved": False,
        },
        "native16": parity_summary["native16"],
        "runtime_contract": parity_summary["runtime_contract"],
        "old_offline_parity": parity_summary["old_offline_parity"],
        "current_480_parity": parity_summary["current_480_parity"],
        "groups": parity_summary["groups"],
        "source_identity": execution_source_identity_native16(),
    }
    legacy.write_json_exclusive(path, value)


def write_scoring_supplement(output_root: Path) -> None:
    legacy = configure_legacy()
    records = legacy.read_jsonl(output_root / "case-results.jsonl")
    if len(records) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"native-16 scored case count {len(records)} != {EXPECTED_TOTAL}"
        )
    present = sum(
        isinstance(record.get("c4eb_reference"), dict) for record in records
    )
    missing = len(records) - present
    checks = {
        "case_count_74": len(records) == EXPECTED_TOTAL,
        "speech_count_68": sum(not row["pure_noise"] for row in records)
        == EXPECTED_SPEECH,
        "noise_count_6": sum(bool(row["pure_noise"]) for row in records)
        == EXPECTED_NOISE,
        "candidate_scores_all_present": all(
            isinstance(row.get("scoring", {}).get("candidate_scores"), dict)
            for row in records
        ),
        "runtime_contract_all_present": all(
            isinstance(row.get("native16_runtime_validation"), dict)
            for row in records
        ),
        "dpdf_reference_delay_640": all(
            int(row["dpdfnet2_reference"]["fixed_delay_samples"])
            == DPDF_REFERENCE_DELAY_SAMPLES
            for row in records
        ),
        "score_fixed_delay_0": all(
            int(row["stream_output"]["score_fixed_delay_samples"]) == 0
            for row in records
        ),
        "c4eb_missing_16_not_fabricated": (
            present == EXPECTED_C4EB_PRESENT
            and missing == EXPECTED_C4EB_MISSING
        ),
    }
    path = output_root / "NATIVE16-SCORING-VALIDATION.json"
    value = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "counts": {
            "total": len(records),
            "speech": sum(not row["pure_noise"] for row in records),
            "noise": sum(bool(row["pure_noise"]) for row in records),
            "c4eb_present": present,
            "c4eb_missing": missing,
        },
        "scope": {
            "ns_only": True,
            "not_full_vcp": True,
            "device_realtime": False,
            "hard_realtime_verified": False,
            "deployment_approved": False,
        },
    }
    legacy.write_json_exclusive(path, value)


def build_source_identities_native16(
    output_root: Path,
    reference_data: dict[str, Any],
    c4eb_data: dict[str, Any],
    score_module: ModuleType,
) -> dict[str, Any]:
    if _ORIGINAL_BUILD_SOURCE_IDENTITIES is None:
        raise RuntimeError("base source identity builder was not initialized")
    identities = _ORIGINAL_BUILD_SOURCE_IDENTITIES(
        output_root,
        reference_data,
        c4eb_data,
        score_module,
    )
    bundle = build_runtime_bundle()
    constants = candidate_constants()
    lock_path = output_root / "RUNTIME-SOURCE-LOCK.json"
    identities["driver"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": sha256_file(Path(__file__).resolve()),
        "base_driver_path": str(BASE_DRIVER),
        "base_driver_sha256": sha256_file(BASE_DRIVER),
        "legacy_validator_path": str(LEGACY_VALIDATOR),
        "legacy_validator_sha256": sha256_file(LEGACY_VALIDATOR),
        "onnx_reference_driver_path": str(ONNX_REFERENCE_DRIVER),
        "onnx_reference_driver_sha256": sha256_file(
            ONNX_REFERENCE_DRIVER
        ),
    }
    identities["runtime_source_lock"] = {
        "path": str(lock_path),
        "sha256": sha256_file(lock_path),
        "value": load_json(lock_path),
    }
    identities["stream"] = {
        "source_path": str(STREAM_SOURCE),
        "source_sha256": sha256_file(STREAM_SOURCE),
        "frame_source_path": str(FRAME_SOURCE),
        "frame_source_sha256": sha256_file(FRAME_SOURCE),
        "onnx_source_path": str(ONNX_SOURCE),
        "onnx_source_sha256": sha256_file(ONNX_SOURCE),
        "transformed_stream_sha256": bundle["transformation"][
            "transformed_sources"
        ]["stream_sha256"],
        "transformed_onnx_sha256": bundle["transformation"][
            "transformed_sources"
        ]["onnx_sha256"],
        "constants": constants,
        "model_adapter_path": str(FROZEN_ADAPTER),
        "model_adapter_sha256": sha256_file(FROZEN_ADAPTER),
        "model_source_path": str(FROZEN_MODEL_SOURCE),
        "model_source_sha256": sha256_file(FROZEN_MODEL_SOURCE),
        "checkpoint_path": str(FROZEN_CHECKPOINT),
        "checkpoint_sha256": sha256_file(FROZEN_CHECKPOINT),
    }
    identities["native16"] = {
        "transformation": bundle["transformation"],
        "runtime": runtime_execution_identity(),
        "onnx_graph": graph_identity(),
        "checkpoint": checkpoint_identity(),
        "current_480_reference": current_480_source_identity(),
        "delay_contract": {
            "candidate_physical_delay_samples": NATIVE_DELAY_SAMPLES,
            "candidate_trim_samples": NATIVE_TRIM_SAMPLES,
            "candidate_score_fixed_delay_samples": (
                NATIVE_SCORE_FIXED_DELAY_SAMPLES
            ),
            "dpdfnet2_reference_fixed_delay_samples": (
                DPDF_REFERENCE_DELAY_SAMPLES
            ),
            "dynamic_delay_search": False,
        },
    }
    identities["protocol"] = {
        "case_count": EXPECTED_TOTAL,
        "speech_count": EXPECTED_SPEECH,
        "noise_count": EXPECTED_NOISE,
        "groups": EXPECTED_GROUPS,
        "ns_only": True,
        "not_full_vcp": True,
        "full_vcp": False,
        "device_realtime": False,
        "hard_realtime_verified": False,
        "deployment_approved": False,
        "stream_chunk_samples": NATIVE_FRAME_SAMPLES,
        "callback_samples": NATIVE_FRAME_SAMPLES,
        "candidate_physical_delay_samples": NATIVE_DELAY_SAMPLES,
        "candidate_physical_delay_ms": (
            NATIVE_DELAY_SAMPLES * 1000.0 / 16000.0
        ),
        "candidate_trim_samples": NATIVE_TRIM_SAMPLES,
        "score_fixed_delay_samples": NATIVE_SCORE_FIXED_DELAY_SAMPLES,
        "dpdfnet2_reference_fixed_delay_samples": (
            DPDF_REFERENCE_DELAY_SAMPLES
        ),
        "dynamic_delay_search": False,
        "same_window_s": True,
        "pcm16_lround_once": True,
        "torch_inference_mode": True,
        "processes_max": 4,
        "model_loaded_once_per_worker": True,
        "shared_model_threads": False,
        "gpu_used": False,
        "speech_weighting": (
            "language equal; SNR equal within language; speaker equal "
            "within stratum"
        ),
        "raw_and_minus30_lufs": True,
        "noise_mos": False,
        "noise_metric": (
            "energy ratio versus actual NS input on the same window"
        ),
        "c4eb_present_cases": EXPECTED_C4EB_PRESENT,
        "c4eb_missing_cases": EXPECTED_C4EB_MISSING,
        "c4eb_missing_values_fabricated": False,
    }
    return identities


def build_report_native16(
    parity_summary: dict[str, Any],
    score_summary: dict[str, Any],
    source_identities: dict[str, Any],
    validation: dict[str, Any],
) -> str:
    if _ORIGINAL_BUILD_REPORT is None:
        raise RuntimeError("base report builder was not initialized")
    report = _ORIGINAL_BUILD_REPORT(
        parity_summary,
        score_summary,
        source_identities,
        validation,
    )
    report = report.replace(
        "# Optimized CoFi-Lite — frozen 74-case NS-only regression",
        "# Native-256 CoFi-Lite ONNX CPU — frozen 74-case NS-only regression",
    )
    report = report.replace(
        "Optimized runtime source SHA",
        "Native-256 runtime source SHA",
    )
    report = report.replace("Optimized CoFi-Lite", "Native-256 CoFi-Lite")
    report = report.replace("optimized parity", "native-256 parity")
    report = report.replace("optimized runtime", "native-256 runtime")
    first_line, separator, body = report.partition("\n")
    details = [
        "",
        "- Runtime: exact source transform `FRAME_SAMPLES 160 -> 256`, "
        "`DELAY_SAMPLES 480 -> 256`; callback/frame/trim are 256 samples.",
        "- `LOOKAHEAD_SAMPLES` remains the original definition `511`.",
        "- Parity: native-256 output is aligned against both the frozen old "
        "offline output and the frozen current-480 ORT output; both must be "
        "strictly below `1e-5` max absolute difference.",
        "- Candidate aligned audio uses score fixed delay `0`; frozen DPDFNet "
        "v1 keeps its original `640`-sample delay.",
        "- Scope: `ns_only=true`, `not_full_vcp=true`, `device_realtime=false`, "
        "`deployment_approved=false`.",
        "",
    ]
    return first_line + separator + "\n".join(details) + body


def final_validation_native16(
    records: Sequence[dict[str, Any]],
    parity_summary: dict[str, Any],
    clean_validation: dict[str, Any],
) -> dict[str, Any]:
    if _ORIGINAL_FINAL_VALIDATION is None:
        raise RuntimeError("base final validator was not initialized")
    result = _ORIGINAL_FINAL_VALIDATION(
        records,
        parity_summary,
        clean_validation,
    )
    checks: dict[str, bool] = {}
    for name, value in result["checks"].items():
        renamed = name.replace("_480", "_256")
        checks[renamed] = bool(value)
    runtime_contract = parity_summary["runtime_contract"]
    c4eb_present = sum(
        isinstance(record.get("c4eb_reference"), dict)
        for record in records
    )
    c4eb_missing = len(records) - c4eb_present
    checks.update(
        {
            "native16_runtime_info_all_verified": runtime_contract[
                "all_runtime_info_verified"
            ],
            "native16_frame_256_all_cases": runtime_contract[
                "frame_samples_values"
            ]
            == [NATIVE_FRAME_SAMPLES],
            "native16_delay_256_all_cases": runtime_contract[
                "delay_samples_values"
            ]
            == [NATIVE_DELAY_SAMPLES],
            "native16_trim_256_all_cases": runtime_contract[
                "trim_samples_values"
            ]
            == [NATIVE_TRIM_SAMPLES],
            "native16_chunk_256_all_cases": runtime_contract[
                "chunk_size_values"
            ]
            == [NATIVE_FRAME_SAMPLES],
            "native16_lookahead_511_all_cases": runtime_contract[
                "lookahead_samples_values"
            ]
            == [EXPECTED_LOOKAHEAD_SAMPLES],
            "native16_state_count_18_all_cases": runtime_contract[
                "state_count_values"
            ]
            == [EXPECTED_STATE_COUNT],
            "old_offline_parity_all_74": parity_summary[
                "old_offline_parity"
            ]["all_pass"],
            "current_480_parity_all_74": parity_summary[
                "current_480_parity"
            ]["all_pass"],
            "onnx_graph_sha_locked": graph_identity()["graph"]["sha256"]
            == EXPECTED_GRAPH_SHA256,
            "checkpoint_sha_locked": checkpoint_identity()["sha256"]
            == EXPECTED_CHECKPOINT_SHA256,
            "c4eb_present_58_missing_16": (
                c4eb_present == EXPECTED_C4EB_PRESENT
                and c4eb_missing == EXPECTED_C4EB_MISSING
            ),
            "c4eb_missing_values_not_fabricated": all(
                record.get("c4eb_reference") is None
                or isinstance(record.get("c4eb_reference"), dict)
                for record in records
            ),
            "scope_ns_only": all(
                row["stream_info"].get("driver_native16_runtime_verified")
                is True
                for row in records
            ),
        }
    )
    result["checks"] = checks
    result["status"] = "pass" if all(checks.values()) else "fail"
    result["native16"] = {
        "runtime_contract": runtime_contract,
        "old_offline_parity": parity_summary["old_offline_parity"],
        "current_480_parity": parity_summary["current_480_parity"],
        "onnx_graph": graph_identity(),
        "checkpoint": checkpoint_identity(),
        "c4eb": {
            "present": c4eb_present,
            "missing": c4eb_missing,
            "missing_values_fabricated": False,
        },
    }
    result["delay_contract"] = {
        "candidate_frame_samples": NATIVE_FRAME_SAMPLES,
        "candidate_physical_delay_samples": NATIVE_DELAY_SAMPLES,
        "candidate_trim_samples": NATIVE_TRIM_SAMPLES,
        "candidate_score_fixed_delay_samples": (
            NATIVE_SCORE_FIXED_DELAY_SAMPLES
        ),
        "lookahead_samples": EXPECTED_LOOKAHEAD_SAMPLES,
        "dpdfnet2_reference_fixed_delay_samples": (
            DPDF_REFERENCE_DELAY_SAMPLES
        ),
        "dynamic_delay_search": False,
    }
    result["scope"] = {
        "ns_only": True,
        "not_full_vcp": True,
        "full_vcp": False,
        "device_realtime": False,
        "hard_realtime_verified": False,
        "deployment_approved": False,
    }
    return result


def write_c4eb_coverage_native16(
    output_root: Path,
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    legacy = configure_legacy()
    present_by_dataset = {group: 0 for group in EXPECTED_GROUPS}
    missing_by_dataset = {group: 0 for group in EXPECTED_GROUPS}
    missing_ids: list[str] = []
    for record in records:
        dataset = str(record["dataset"])
        if isinstance(record.get("c4eb_reference"), dict):
            present_by_dataset[dataset] += 1
        else:
            missing_by_dataset[dataset] += 1
            missing_ids.append(f"{dataset}/{record['case_id']}")
    present = sum(present_by_dataset.values())
    missing = sum(missing_by_dataset.values())
    value = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "expected_cases": EXPECTED_TOTAL,
        "present_cases": present,
        "missing_cases": missing,
        "present_by_dataset": present_by_dataset,
        "missing_by_dataset": missing_by_dataset,
        "missing_case_ids": sorted(missing_ids),
        "complete": missing == 0,
        "handling": (
            "Missing c4eb values remain null; no score or reference was "
            "inferred, borrowed, or fabricated."
        ),
    }
    path = output_root / "C4EB-COVERAGE.json"
    legacy.write_json_exclusive(path, value)
    return value


def finalize_phase_native16(output_root: Path) -> int:
    base = load_base_driver()
    legacy = configure_legacy()
    ensure_source_lock_native16(output_root, create=False)
    _, parity_summary = validate_parity_native16(output_root)
    records = legacy.read_jsonl(output_root / "case-results.jsonl")
    if len(records) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"case result count {len(records)} != {EXPECTED_TOTAL}"
        )
    output_paths = {
        "summary": output_root / "summary.json",
        "source_identities": output_root / "source-identities.json",
        "report": output_root / "REPORT-native16.md",
        "final_validation": output_root / "FINAL-VALIDATION.json",
        "final_manifest": output_root / "FINAL-MANIFEST.json",
        "c4eb_coverage": output_root / "C4EB-COVERAGE.json",
    }
    for name, path in output_paths.items():
        if path.exists():
            raise FileExistsError(
                f"refusing to overwrite finalized native-16 {name}: {path}"
            )

    reference_data = base.load_frozen_reference_data()
    c4eb_data = legacy.load_c4eb_references()
    score_module = legacy.load_module(
        "run_and_score_finalize_native16",
        legacy.RUN_AND_SCORE,
    )
    score_summary = legacy.build_score_summary(records)
    clean_validation = base.clean_reference_validation(records)
    source_identities = build_source_identities_native16(
        output_root,
        reference_data,
        c4eb_data,
        score_module,
    )
    validation = final_validation_native16(
        records,
        parity_summary,
        clean_validation,
    )
    c4eb_coverage = write_c4eb_coverage_native16(output_root, records)
    summary = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": (
            "complete" if validation["status"] == "pass"
            else "validation_failed"
        ),
        "execution_status": "complete",
        "scope": {
            "name": "Native-256 CoFi-Lite NS-only regression",
            "ns_only": True,
            "not_full_vcp": True,
            "full_vcp": False,
            "device_realtime": False,
            "hard_realtime_verified": False,
            "deployment_approved": False,
            "notes": [
                "frozen actual tx_ns_input cases only",
                "old offline and current-480 outputs are aligned parity references only",
                "all native-256 outputs were reprocessed",
                "all 74 candidate outputs were freshly rescored",
                "frozen DPDFNet v1 reference delay remains 640 samples",
                "candidate physical delay and trim are 256 samples",
                "candidate aligned scoring delay is 0",
                "speech48, clean16, continuous4, and guards are never pooled",
                "noise cases are not assigned MOS values",
                "16 missing c4eb references remain null and are not fabricated",
            ],
        },
        "counts": {
            "total": len(records),
            "speech": sum(not row["pure_noise"] for row in records),
            "noise": sum(bool(row["pure_noise"]) for row in records),
            "groups": {
                group: sum(row["group"] == group for row in records)
                for group in EXPECTED_GROUPS
            },
            "c4eb": {
                "present": c4eb_coverage["present_cases"],
                "missing": c4eb_coverage["missing_cases"],
            },
        },
        "parity": parity_summary,
        "clean_reference_validation": clean_validation,
        "scores": score_summary,
        "source_identities": source_identities,
        "final_validation": validation,
    }
    report = build_report_native16(
        parity_summary,
        {**score_summary, "_records": records},
        source_identities,
        validation,
    )
    legacy.write_json_exclusive(output_paths["source_identities"], source_identities)
    legacy.write_json_exclusive(output_paths["summary"], summary)
    with output_paths["report"].open("x", encoding="utf-8") as stream:
        stream.write(report)
    legacy.write_json_exclusive(
        output_paths["final_validation"], validation
    )
    manifest_names = (
        "RUNTIME-SOURCE-LOCK.json",
        "parity-results.jsonl",
        "parity-summary.json",
        "NATIVE16-PARITY-VALIDATION.json",
        "case-results.jsonl",
        "SCORING-VALIDATION.json",
        "NATIVE16-SCORING-VALIDATION.json",
        "C4EB-COVERAGE.json",
        "source-identities.json",
        "summary.json",
        "REPORT-native16.md",
        "FINAL-VALIDATION.json",
    )
    manifest_files: dict[str, Any] = {}
    for name in manifest_names:
        path = output_root / name
        if not path.is_file():
            raise RuntimeError(f"final native-16 evidence is missing: {path}")
        manifest_files[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest = {
        "created_utc": utc_now(),
        "status": validation["status"],
        "files": manifest_files,
        "scope": validation["scope"],
    }
    legacy.write_json_exclusive(output_paths["final_manifest"], manifest)
    legacy.append_log(
        output_root / "run.log",
        "native-16 finalize complete "
        f"status={validation['status']} "
        f"parity={parity_summary['pass_count']}/{parity_summary['cases']} "
        f"old_offline_max_abs="
        f"{parity_summary['old_offline_parity']['max_abs']:.12g} "
        f"current_480_max_abs="
        f"{parity_summary['current_480_parity']['max_abs']:.12g} "
        f"clean_exact={clean_validation['exact_within_1e-12']}",
    )
    print(
        json.dumps(
            legacy.json_safe(
                {
                    "status": validation["status"],
                    "parity": parity_summary,
                    "final_validation": validation,
                    "summary": str(output_paths["summary"]),
                    "report": str(output_paths["report"]),
                }
            ),
            sort_keys=True,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if validation["status"] == "pass" else 2


def configure_legacy() -> ModuleType:
    base = load_base_driver()
    return base.configure_legacy()


def configure_candidate() -> ModuleType:
    global _CONFIGURED
    global _ORIGINAL_WORKER_INITIALIZE
    global _ORIGINAL_PROCESS_ONE_CASE
    global _ORIGINAL_SUMMARIZE_PARITY
    global _ORIGINAL_VALIDATE_PARITY
    global _ORIGINAL_SCORE_ONE_CASE
    global _ORIGINAL_BUILD_SOURCE_IDENTITIES
    global _ORIGINAL_BUILD_REPORT
    global _ORIGINAL_FINAL_VALIDATION
    global _ORIGINAL_FINALIZE_PHASE

    base = load_base_driver()
    load_onnx_reference_driver()
    build_runtime_bundle()
    graph_identity()
    checkpoint_identity()
    current_480_reference_data()

    base.EXPECTED_OUTPUT_ROOT = EXPECTED_OUTPUT_ROOT
    base.RUNTIME_SOURCE = ONNX_SOURCE
    base.RUNTIME_FRAME_SOURCE = FRAME_SOURCE
    base._ACTIVE_OUTPUT_ROOT = EXPECTED_OUTPUT_ROOT
    base._RUNTIME = None

    if not _CONFIGURED:
        _ORIGINAL_WORKER_INITIALIZE = base.worker_initialize_optimized
        _ORIGINAL_PROCESS_ONE_CASE = base.process_one_stream_case_optimized
        _ORIGINAL_SUMMARIZE_PARITY = base.summarize_parity
        _ORIGINAL_VALIDATE_PARITY = base.validate_parity
        _ORIGINAL_SCORE_ONE_CASE = base.score_one_case_optimized
        _ORIGINAL_BUILD_SOURCE_IDENTITIES = base.build_source_identities
        _ORIGINAL_BUILD_REPORT = base.build_report
        _ORIGINAL_FINAL_VALIDATION = base.final_validation
        _ORIGINAL_FINALIZE_PHASE = base.finalize_phase

    base.runtime_module = runtime_module
    base.candidate_constants = candidate_constants
    base.execution_source_identity = execution_source_identity_native16
    base.ensure_source_lock = ensure_source_lock_native16
    base.worker_initialize_optimized = worker_initialize_native16
    base.process_one_stream_case_optimized = process_one_stream_case_native16
    base.summarize_parity = summarize_parity_native16
    base.validate_parity = validate_parity_native16
    base.score_one_case_optimized = score_one_case_native16
    base.build_source_identities = build_source_identities_native16
    base.build_report = build_report_native16
    base.final_validation = final_validation_native16
    base.finalize_phase = finalize_phase_native16

    legacy = base.configure_legacy()
    legacy.import_stream_module = import_candidate_stream_module
    constants = candidate_constants()
    if constants["frame_samples"] != NATIVE_FRAME_SAMPLES:
        raise RuntimeError("native-16 frame_samples was not applied")
    if constants["delay_samples"] != NATIVE_DELAY_SAMPLES:
        raise RuntimeError("native-16 delay_samples was not applied")
    _CONFIGURED = True
    return base


def validate_output_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    expected = EXPECTED_OUTPUT_ROOT.resolve()
    if resolved != expected:
        raise SystemExit(
            f"native-16 output root is fixed at {expected}; got {resolved}"
        )
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        choices=("parity", "score", "finalize", "all"),
        required=True,
        help=(
            "parity: reprocess and align all 74 cases; score: score the "
            "completed parity run; finalize: build summaries/manifest; "
            "all: run all three phases"
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="parallel parity workers, 1..4 (default: 4)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=EXPECTED_OUTPUT_ROOT,
        help=argparse.SUPPRESS,
    )
    return parser


def parity_phase_native16(
    cases: Sequence[dict[str, Any]],
    output_root: Path,
    jobs: int,
) -> int:
    base = load_base_driver()
    status = base.parity_phase(cases, output_root, jobs)
    summary_path = output_root / "parity-summary.json"
    if summary_path.is_file():
        summary = load_json(summary_path)
        if not (output_root / "NATIVE16-PARITY-VALIDATION.json").exists():
            write_parity_supplement(output_root, summary)
    return status


def score_phase_native16(
    cases: Sequence[dict[str, Any]],
    output_root: Path,
) -> int:
    base = load_base_driver()
    status = base.score_phase(cases, output_root)
    if status == 0:
        if not (output_root / "NATIVE16-SCORING-VALIDATION.json").exists():
            write_scoring_supplement(output_root)
    return status


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.jobs < 1 or args.jobs > 4:
        raise SystemExit("--jobs must be between 1 and 4")
    output_root = validate_output_root(args.output_root)
    configure_candidate()
    legacy = configure_legacy()
    output_root.mkdir(parents=True, exist_ok=True)
    cases = legacy.load_cases()
    if args.phase == "parity":
        return parity_phase_native16(cases, output_root, args.jobs)
    if args.phase == "score":
        return score_phase_native16(cases, output_root)
    if args.phase == "finalize":
        return finalize_phase_native16(output_root)
    status = parity_phase_native16(cases, output_root, args.jobs)
    if status != 0:
        print(
            "native-16 parity threshold failed; scoring was not started",
            file=sys.stderr,
            flush=True,
        )
        return status
    status = score_phase_native16(cases, output_root)
    if status != 0:
        return status
    return finalize_phase_native16(output_root)


if __name__ == "__main__":
    raise SystemExit(main())
