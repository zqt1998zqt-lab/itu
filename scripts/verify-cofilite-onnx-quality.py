#!/usr/bin/env python3
"""Run the frozen 74-case NS quality matrix on the CoFi-Lite ORT CPU runtime.

This is a thin wrapper around verify-cofilite-optimized-quality.py.  It reuses
the already verified case loading, stream parity, scorer, grouping, clean
reference, DPDFNet comparison, and c4eb handling.  It replaces only:

* the candidate runtime import, from server.cofilite_stream to
  server.cofilite_onnx;
* the candidate output root;
* source-lock identities, adding the versioned ONNX graph;
* executor/report wording so ORT CPU is never mislabeled as torch JIT.

No legacy script or runtime source is modified.  New evidence is restricted to:

  build/cofilite-ort-20261117/quality74
"""

from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True

from pathlib import Path

PROJECT_ROOT = Path("/home/zhangqingtian/project/itu")
BASE_DRIVER = PROJECT_ROOT / "scripts/verify-cofilite-optimized-quality.py"
LEGACY_VALIDATOR = PROJECT_ROOT / "scripts/verify-cofilite-stream-quality.py"
DEFAULT_RUNTIME_SOURCE = PROJECT_ROOT / "server/cofilite_onnx.py"
ORT_ROOT = PROJECT_ROOT / "build/cofilite-ort-20261117"
EXPECTED_OUTPUT_ROOT = ORT_ROOT / "quality74"

EXPECTED_WEIGHT_SHA256 = (
    "40d1bb69dcd553ce93c745c8bd136e1593397f529aa53609827c69df95ce2a30"
)
EXPECTED_STATE_COUNT = 18
EXPECTED_TOTAL = 74
EXPECTED_SPEECH = 68
EXPECTED_NOISE = 6
EXPECTED_GROUPS = {
    "v1speech48": 48,
    "v1clean16": 16,
    "v1continuous4": 4,
    "v1guards": 6,
}

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
import importlib
import importlib.util
import json
from types import ModuleType
from typing import Any, Sequence


_BASE: ModuleType | None = None
_RUNTIME_SOURCE = DEFAULT_RUNTIME_SOURCE
_GRAPH_ASSET: Path | None = None
_EXPECTED_RUNTIME_SHA256: str | None = None
_EXPECTED_GRAPH_SHA256: str | None = None
_ORIGINAL_EXECUTION_SOURCE_IDENTITY: Any = None
_ORIGINAL_BUILD_SOURCE_IDENTITIES: Any = None
_ORIGINAL_BUILD_REPORT: Any = None
_ORIGINAL_FINAL_VALIDATION: Any = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise ValueError(f"missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_base_driver() -> ModuleType:
    global _BASE
    if _BASE is not None:
        return _BASE
    if not BASE_DRIVER.is_file():
        raise RuntimeError(f"base quality driver is missing: {BASE_DRIVER}")
    spec = importlib.util.spec_from_file_location(
        "_itu_verify_cofilite_optimized_quality_base",
        BASE_DRIVER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import base driver: {BASE_DRIVER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _BASE = module
    return module


def file_identity(path: Path, *, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{role} is missing: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "role": role,
    }


def graph_identity() -> dict[str, Any]:
    if _GRAPH_ASSET is None:
        raise RuntimeError("ONNX graph asset was not configured")
    if _GRAPH_ASSET.suffix.lower() != ".onnx":
        raise RuntimeError(f"graph asset must be .onnx: {_GRAPH_ASSET}")
    resolved = _GRAPH_ASSET.resolve()
    if not resolved.is_relative_to(ORT_ROOT.resolve()):
        raise RuntimeError(
            f"graph asset must be versioned below {ORT_ROOT}: {resolved}"
        )
    if resolved.is_relative_to(EXPECTED_OUTPUT_ROOT.resolve()):
        raise RuntimeError(
            f"graph asset cannot live inside quality output root: {resolved}"
        )
    identity = file_identity(resolved, role="onnx_graph")
    expected = _EXPECTED_GRAPH_SHA256
    if expected is not None and identity["sha256"] != expected:
        raise RuntimeError(
            f"ONNX graph SHA mismatch: expected {expected}, "
            f"got {identity['sha256']}"
        )
    sidecars = []
    for suffix, role in (
        (".identity.json", "onnx_graph_identity"),
        (".checker.json", "onnx_graph_checker"),
    ):
        sidecar = resolved.with_suffix(suffix)
        if sidecar.is_file():
            sidecars.append(file_identity(sidecar, role=role))
    identity["sidecars"] = sidecars
    identity_path = resolved.with_suffix(".identity.json")
    if identity_path.is_file():
        declared = load_json(identity_path)
        if declared.get("model_sha256") != identity["sha256"]:
            raise RuntimeError("ONNX identity sidecar model SHA mismatch")
        if declared.get("weights_sha256") != EXPECTED_WEIGHT_SHA256:
            raise RuntimeError("ONNX identity sidecar weight SHA mismatch")
        if int(declared.get("state_count", -1)) != EXPECTED_STATE_COUNT:
            raise RuntimeError("ONNX identity sidecar state count is not 18")
        if declared.get("ort_provider") != ["CPUExecutionProvider"]:
            raise RuntimeError("ONNX identity sidecar is not CPU-only")
    return identity


def runtime_identity() -> dict[str, Any]:
    identity = file_identity(_RUNTIME_SOURCE, role="onnx_runtime")
    expected = _EXPECTED_RUNTIME_SHA256
    if expected is not None and identity["sha256"] != expected:
        raise RuntimeError(
            f"ONNX runtime SHA mismatch: expected {expected}, "
            f"got {identity['sha256']}"
        )
    return identity


def base_runtime_module() -> ModuleType:
    base = load_base_driver()
    if base._RUNTIME is None:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        candidate = importlib.import_module("server.cofilite_onnx")
        module_path = Path(candidate.__file__).resolve()
        if module_path != _RUNTIME_SOURCE.resolve():
            raise RuntimeError(
                "imported server.cofilite_onnx path does not match source lock: "
                f"{module_path} != {_RUNTIME_SOURCE.resolve()}"
            )
        base._RUNTIME = build_runtime_adapter(candidate)
    return base._RUNTIME


def build_runtime_adapter(candidate: ModuleType) -> ModuleType:
    """Expose the legacy driver API without changing the worker implementation."""
    if callable(getattr(candidate, "load_model", None)) and callable(
        getattr(candidate, "enhance_stream", None)
    ):
        return candidate
    if not callable(getattr(candidate, "enhance_stream_onnx", None)):
        raise RuntimeError(
            "server.cofilite_onnx exposes neither enhance_stream nor "
            "enhance_stream_onnx"
        )
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    import server.cofilite_stream as base_stream

    adapter = ModuleType("_itu_cofilite_onnx_candidate_adapter")
    adapter.__file__ = str(_RUNTIME_SOURCE)
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
        if hasattr(candidate, name):
            setattr(adapter, name, getattr(candidate, name))
    adapter._executor = "ort_cpu"

    def load_model(device: str = "cpu", threads: int = 1) -> Any:
        return base_stream.load_model(device, threads)

    def enhance_stream(
        audio: Any,
        chunk_size: int = int(candidate.FRAME_SAMPLES),
        *,
        model: Any = None,
    ) -> tuple[Any, dict[str, Any]]:
        graph = graph_identity()
        return candidate.enhance_stream_onnx(
            audio,
            graph["path"],
            chunk_size=chunk_size,
            model=model,
            expected_model_sha256=graph["sha256"],
            expected_weights_sha256=EXPECTED_WEIGHT_SHA256,
        )

    adapter.load_model = load_model
    adapter.enhance_stream = enhance_stream
    return adapter


def import_candidate_stream_module() -> ModuleType:
    return base_runtime_module()


def candidate_executor_label() -> str:
    runtime = base_runtime_module()
    executor = getattr(runtime, "_executor", None)
    label = None
    for name in ("EXECUTOR_NAME", "EXECUTOR_LABEL", "_executor_name"):
        value = getattr(runtime, name, None)
        if isinstance(value, str) and value.strip():
            label = value.strip()
            break
    if label is None and isinstance(executor, str) and executor.strip():
        label = executor.strip()
    if label is None:
        label = "ort_cpu"
    lowered = label.lower()
    if any(token in lowered for token in ("jit", "torchscript", "cuda", "gpu")):
        raise RuntimeError(f"candidate executor is not CPU ORT: {label}")
    if "ort" not in lowered and "onnx" not in lowered:
        label = f"ort_cpu:{label}"
    return label


def runtime_executor_identity() -> dict[str, Any]:
    runtime = base_runtime_module()
    executor = getattr(runtime, "_executor", None)
    state_shapes = getattr(runtime, "STATE_SHAPES", None)
    state_count = None
    if state_shapes is not None:
        try:
            state_count = len(state_shapes)
        except TypeError:
            state_count = None
    if state_count is not None and int(state_count) != EXPECTED_STATE_COUNT:
        raise RuntimeError(
            f"ONNX runtime state count {state_count} != {EXPECTED_STATE_COUNT}"
        )
    return {
        "label": candidate_executor_label(),
        "backend": "onnxruntime",
        "device": "cpu",
        "torch_jit_used": False,
        "executor_attribute_type": type(executor).__qualname__,
        "state_count": state_count,
        "expected_state_count": EXPECTED_STATE_COUNT,
    }


def stream_state_count(record: dict[str, Any]) -> int:
    info = record.get("stream_info")
    if not isinstance(info, dict):
        raise RuntimeError("stream info is missing from ONNX runtime record")
    for name in ("state_count", "num_states"):
        value = info.get(name)
        if value is not None:
            return int(value)
    for name in ("state_shapes", "states"):
        value = info.get(name)
        if isinstance(value, (list, tuple)):
            return len(value)
    value = runtime_executor_identity().get("state_count")
    if value is None:
        raise RuntimeError(
            "cannot verify the expected 18 ONNX recurrent states from "
            "runtime module or stream info"
        )
    return int(value)


def worker_initialize_onnx() -> None:
    import torch

    torch.set_num_threads(1)
    base = load_base_driver()
    base.apply_legacy_candidate_constants()
    legacy = base.legacy_module()
    legacy.import_stream_module = import_candidate_stream_module
    legacy.worker_initialize()


def process_one_stream_case_onnx(
    payload: dict[str, Any],
) -> dict[str, Any]:
    base = load_base_driver()
    legacy = base.apply_legacy_candidate_constants()
    legacy.import_stream_module = import_candidate_stream_module
    if getattr(legacy, "_WORKER_MODEL", None) is None:
        worker_initialize_onnx()
    record = legacy.process_one_stream_case(payload)
    delay = int(base.candidate_constants()["delay_samples"])
    executor = runtime_executor_identity()
    graph = graph_identity()
    reported_executor = record["stream_info"].get("frame_executor")
    if reported_executor != "onnxruntime_cpu":
        raise RuntimeError(
            "ONNX runtime reported unexpected frame_executor metadata: "
            f"{reported_executor!r}"
        )
    record["stream_info"]["driver_candidate_executor_label"] = executor["label"]
    record["stream_info"]["driver_expected_frame_executor"] = (
        "onnxruntime_cpu"
    )
    record["stream_info"]["driver_frame_executor_verified"] = True
    record["stream_info"]["driver_execution_backend"] = "onnxruntime_cpu"
    record["stream_info"]["driver_torch_jit_used"] = False
    record["stream_info"]["driver_onnx_graph_sha256"] = graph["sha256"]
    record["stream_info"]["driver_state_count"] = stream_state_count(record)
    reported_graph_path = record["stream_info"].get(
        "frame_executor_model_path"
    )
    reported_graph_sha = record["stream_info"].get(
        "frame_executor_model_sha256"
    )
    if (
        reported_graph_path is None
        or Path(str(reported_graph_path)).resolve()
        != Path(str(graph["path"])).resolve()
    ):
        raise RuntimeError(
            "ONNX runtime did not report the source-locked graph path"
        )
    if reported_graph_sha != graph["sha256"]:
        raise RuntimeError(
            "ONNX runtime did not report the source-locked graph SHA"
        )
    record["stream_output"]["physical_delay_samples"] = delay
    record["stream_output"]["trim_samples"] = delay
    record["stream_output"]["score_fixed_delay_samples"] = 0
    record["stream_output"]["delay_note"] = (
        f"ONNX Runtime CPU stream physical delay {delay} samples was drained "
        "and trimmed; the aligned FLOAT file is scored with fixed delay 0"
    )
    record["candidate_execution_identity"] = payload["source_lock"]
    return record


def execution_source_identity_onnx() -> dict[str, Any]:
    if _ORIGINAL_EXECUTION_SOURCE_IDENTITY is None:
        raise RuntimeError("base source identity function was not initialized")
    base = load_base_driver()
    current = _ORIGINAL_EXECUTION_SOURCE_IDENTITY()
    files = dict(current["files"])
    files.pop("runtime_frame", None)
    files["driver"] = file_identity(Path(__file__).resolve(), role="driver")
    files["runtime_stream"] = runtime_identity()
    files["onnx_graph"] = graph_identity()
    files["legacy_validator"] = file_identity(
        LEGACY_VALIDATOR, role="legacy_validator"
    )
    files["frozen_checkpoint"] = file_identity(
        PROJECT_ROOT
        / "build/ns-cofilite-20261115/upstream/checkpoint/model_on_dns3.tar",
        role="frozen_checkpoint",
    )
    if (
        files["frozen_checkpoint"]["sha256"] != EXPECTED_WEIGHT_SHA256
    ):
        raise RuntimeError("frozen CoFi checkpoint SHA changed")
    return {
        **current,
        "files": files,
        "onnx_runtime": runtime_executor_identity(),
        "onnx_graph": graph_identity(),
        "base_driver": file_identity(BASE_DRIVER, role="base_driver"),
        "candidate_constants": base.candidate_constants(),
        "dpdf_reference_delay_samples": base.DPDF_REFERENCE_DELAY_SAMPLES,
    }


def build_source_identities_onnx(
    output_root: Path,
    reference_data: dict[str, Any],
    c4eb_data: dict[str, Any],
    score_module: ModuleType,
) -> dict[str, Any]:
    if _ORIGINAL_BUILD_SOURCE_IDENTITIES is None:
        raise RuntimeError("base source identities function was not initialized")
    identities = _ORIGINAL_BUILD_SOURCE_IDENTITIES(
        output_root,
        reference_data,
        c4eb_data,
        score_module,
    )
    identities["driver"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": sha256_file(Path(__file__).resolve()),
        "base_driver_path": str(BASE_DRIVER),
        "base_driver_sha256": sha256_file(BASE_DRIVER),
        "legacy_validator_path": str(LEGACY_VALIDATOR),
        "legacy_validator_sha256": sha256_file(LEGACY_VALIDATOR),
    }
    identities["onnx"] = {
        "graph": graph_identity(),
        "runtime": runtime_identity(),
        "executor": runtime_executor_identity(),
        "weight_checkpoint": file_identity(
            PROJECT_ROOT
            / "build/ns-cofilite-20261115/upstream/checkpoint/"
            "model_on_dns3.tar",
            role="frozen_checkpoint",
        ),
        "execution_provider": "CPUExecutionProvider",
        "gpu_used": False,
    }
    if (
        identities["onnx"]["weight_checkpoint"]["sha256"]
        != EXPECTED_WEIGHT_SHA256
    ):
        raise RuntimeError("ONNX identity changed the frozen weight SHA")
    identities["stream"]["runtime_kind"] = "onnxruntime_cpu"
    identities["stream"]["candidate_executor"] = candidate_executor_label()
    identities["protocol"]["onnxruntime_cpu_only"] = True
    identities["protocol"]["torch_jit_used"] = False
    identities["protocol"]["gpu_used"] = False
    return identities


def build_report_onnx(
    parity_summary: dict[str, Any],
    score_summary: dict[str, Any],
    source_identities: dict[str, Any],
    validation: dict[str, Any],
) -> str:
    if _ORIGINAL_BUILD_REPORT is None:
        raise RuntimeError("base report function was not initialized")
    report = _ORIGINAL_BUILD_REPORT(
        parity_summary,
        score_summary,
        source_identities,
        validation,
    )
    report = report.replace(
        "# Optimized CoFi-Lite — frozen 74-case NS-only regression",
        "# CoFi-Lite ONNX Runtime CPU — frozen 74-case NS-only regression",
    )
    report = report.replace(
        "Optimized runtime source SHA",
        "ONNX runtime source SHA",
    )
    report = report.replace(
        "Frame executor source SHA",
        "ONNX Runtime executor source SHA",
    )
    report = report.replace("CoFi optimized", "CoFi ORT CPU")
    graph_sha = source_identities["onnx"]["graph"]["sha256"]
    report = report.replace(
        f"- Frozen DPDFNet model SHA: `"
        f"{source_identities['references']['dpdfnet2']['model_sha256']}`.",
        f"- Frozen DPDFNet model SHA: `"
        f"{source_identities['references']['dpdfnet2']['model_sha256']}`.\n"
        f"- Versioned ONNX graph SHA: `{graph_sha}`.\n"
        "- Candidate execution provider: `CPUExecutionProvider`; "
        "torch JIT is not used.",
    )
    return report


def final_validation_onnx(
    records: Sequence[dict[str, Any]],
    parity_summary: dict[str, Any],
    clean_validation: dict[str, Any],
) -> dict[str, Any]:
    if _ORIGINAL_FINAL_VALIDATION is None:
        raise RuntimeError("base final validation function was not initialized")
    result = _ORIGINAL_FINAL_VALIDATION(
        records, parity_summary, clean_validation
    )
    executor = candidate_executor_label()
    graph = graph_identity()
    lock_path = EXPECTED_OUTPUT_ROOT / "RUNTIME-SOURCE-LOCK.json"
    locked_graph = None
    if lock_path.is_file():
        locked_graph = load_json(lock_path).get("files", {}).get("onnx_graph")
    checks = result["checks"]
    checks.update(
        {
            "runtime_metadata_onnxruntime_cpu": all(
                row["stream_info"].get("frame_executor")
                == "onnxruntime_cpu"
                for row in records
            ),
            "driver_candidate_executor_ort_cpu": all(
                row["stream_info"].get("driver_candidate_executor_label")
                == executor
                for row in records
            ),
            "driver_frame_executor_verified": all(
                row["stream_info"].get("driver_frame_executor_verified")
                is True
                for row in records
            ),
            "onnx_graph_locked": (
                isinstance(locked_graph, dict)
                and locked_graph == graph
            ),
            "onnx_state_count_18": all(
                int(row["stream_info"].get("driver_state_count", -1))
                == EXPECTED_STATE_COUNT
                for row in records
            ),
            "frozen_weight_sha": (
                file_identity(
                    PROJECT_ROOT
                    / "build/ns-cofilite-20261115/upstream/checkpoint/"
                    "model_on_dns3.tar",
                    role="frozen_checkpoint",
                )["sha256"]
                == EXPECTED_WEIGHT_SHA256
            ),
        }
    )
    result["status"] = "pass" if all(checks.values()) else "fail"
    result["onnx"] = {
        "graph": graph,
        "executor": runtime_executor_identity(),
        "weight_sha256": EXPECTED_WEIGHT_SHA256,
        "source_lock": str(lock_path),
    }
    return result


def write_c4eb_coverage(output_root: Path) -> dict[str, Any]:
    base = load_base_driver()
    legacy = base.configure_legacy()
    records = legacy.read_jsonl(output_root / "case-results.jsonl")
    if len(records) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"case results count {len(records)} != {EXPECTED_TOTAL}"
        )
    present_by_dataset = {group: 0 for group in EXPECTED_GROUPS}
    missing_by_dataset = {group: 0 for group in EXPECTED_GROUPS}
    missing_ids: list[str] = []
    for record in records:
        dataset = str(record["dataset"])
        if isinstance(record.get("c4eb_reference"), dict):
            present_by_dataset[dataset] = present_by_dataset.get(dataset, 0) + 1
        else:
            missing_by_dataset[dataset] = missing_by_dataset.get(dataset, 0) + 1
            missing_ids.append(f"{dataset}/{record['case_id']}")
    present_count = sum(present_by_dataset.values())
    missing_count = sum(missing_by_dataset.values())
    coverage = {
        "schema_version": 1,
        "created_utc": base.utc_now(),
        "source": "case-results.jsonl",
        "source_sha256": sha256_file(output_root / "case-results.jsonl"),
        "expected_cases": EXPECTED_TOTAL,
        "present_cases": present_count,
        "missing_cases": missing_count,
        "complete": missing_count == 0,
        "present_by_dataset": present_by_dataset,
        "missing_by_dataset": missing_by_dataset,
        "missing_case_ids": sorted(missing_ids),
        "handling": (
            "Missing c4eb values remain null; no score or reference was "
            "inferred, borrowed, or fabricated."
        ),
        "candidate_scores_recomputed_for_all_74": True,
        "dpdfnet2_reference_complete_for_all_74": True,
        "clean_reference_complete_for_all_68_speech_cases": True,
    }
    legacy.write_json_exclusive(
        output_root / "C4EB-COVERAGE.json", coverage
    )
    return coverage


def write_ort_manifest(output_root: Path) -> dict[str, Any]:
    base = load_base_driver()
    legacy = base.configure_legacy()
    core_path = output_root / "FINAL-MANIFEST.json"
    core = load_json(core_path)
    coverage_path = output_root / "C4EB-COVERAGE.json"
    graph = graph_identity()
    runtime = runtime_identity()
    manifest = {
        "schema_version": 1,
        "created_utc": base.utc_now(),
        "status": core["status"],
        "core_manifest": {
            "path": str(core_path),
            "sha256": sha256_file(core_path),
            "files": core["files"],
        },
        "supplemental_files": {
            "C4EB-COVERAGE.json": file_identity(
                coverage_path, role="c4eb_coverage"
            ),
        },
        "onnx": {
            "graph": graph,
            "runtime": runtime,
            "executor": runtime_executor_identity(),
            "weight_checkpoint": file_identity(
                PROJECT_ROOT
                / "build/ns-cofilite-20261115/upstream/checkpoint/"
                "model_on_dns3.tar",
                role="frozen_checkpoint",
            ),
        },
        "scope": core["scope"],
    }
    legacy.write_json_exclusive(
        output_root / "FINAL-MANIFEST-ORT.json", manifest
    )
    return manifest


def finalize_phase_onnx(output_root: Path) -> int:
    base = load_base_driver()
    status = base.finalize_phase(output_root)
    write_c4eb_coverage(output_root)
    write_ort_manifest(output_root)
    return status


def configure_candidate(
    *,
    runtime_source: Path,
    graph_asset: Path,
    expected_runtime_sha256: str | None,
    expected_graph_sha256: str | None,
) -> ModuleType:
    global _RUNTIME_SOURCE
    global _GRAPH_ASSET
    global _EXPECTED_RUNTIME_SHA256
    global _EXPECTED_GRAPH_SHA256
    global _ORIGINAL_EXECUTION_SOURCE_IDENTITY
    global _ORIGINAL_BUILD_SOURCE_IDENTITIES
    global _ORIGINAL_BUILD_REPORT
    global _ORIGINAL_FINAL_VALIDATION

    _RUNTIME_SOURCE = runtime_source.expanduser().resolve()
    _GRAPH_ASSET = graph_asset.expanduser().resolve()
    _EXPECTED_RUNTIME_SHA256 = expected_runtime_sha256
    _EXPECTED_GRAPH_SHA256 = expected_graph_sha256
    runtime_identity()
    graph_identity()

    base = load_base_driver()
    base.EXPECTED_OUTPUT_ROOT = EXPECTED_OUTPUT_ROOT
    base.RUNTIME_SOURCE = _RUNTIME_SOURCE
    base.RUNTIME_FRAME_SOURCE = _RUNTIME_SOURCE
    base._RUNTIME = None
    base._ACTIVE_OUTPUT_ROOT = EXPECTED_OUTPUT_ROOT

    if _ORIGINAL_EXECUTION_SOURCE_IDENTITY is None:
        _ORIGINAL_EXECUTION_SOURCE_IDENTITY = base.execution_source_identity
    if _ORIGINAL_BUILD_SOURCE_IDENTITIES is None:
        _ORIGINAL_BUILD_SOURCE_IDENTITIES = base.build_source_identities
    if _ORIGINAL_BUILD_REPORT is None:
        _ORIGINAL_BUILD_REPORT = base.build_report
    if _ORIGINAL_FINAL_VALIDATION is None:
        _ORIGINAL_FINAL_VALIDATION = base.final_validation

    base.runtime_module = base_runtime_module
    base.execution_source_identity = execution_source_identity_onnx
    base.worker_initialize_optimized = worker_initialize_onnx
    base.process_one_stream_case_optimized = process_one_stream_case_onnx
    base.build_source_identities = build_source_identities_onnx
    base.build_report = build_report_onnx
    base.final_validation = final_validation_onnx

    legacy = base.configure_legacy()
    legacy.import_stream_module = import_candidate_stream_module
    constants = base.candidate_constants()
    expected_constants = {
        "sample_rate": 16000,
        "frame_samples": 160,
        "delay_samples": 480,
        "hop_samples": 256,
        "fft_size": 512,
        "lookahead_samples": 511,
    }
    if constants != expected_constants:
        raise RuntimeError(
            f"ONNX runtime constants changed: {constants} != "
            f"{expected_constants}"
        )
    runtime_executor_identity()
    return base


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        choices=("preflight", "parity", "score", "finalize", "all"),
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=EXPECTED_OUTPUT_ROOT,
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--runtime-source",
        type=Path,
        default=DEFAULT_RUNTIME_SOURCE,
    )
    parser.add_argument(
        "--graph-asset",
        type=Path,
        required=True,
        help="versioned CoFi frame ONNX graph below build/cofilite-ort-20261117",
    )
    parser.add_argument("--expected-runtime-sha256")
    parser.add_argument("--expected-graph-sha256")
    return parser


def validate_output_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    expected = EXPECTED_OUTPUT_ROOT.resolve()
    if resolved != expected:
        raise SystemExit(
            f"--output-root must be exactly {expected}; got {resolved}"
        )
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.jobs < 1 or args.jobs > 4:
        raise SystemExit("--jobs must be between 1 and 4")
    output_root = validate_output_root(args.output_root)
    base = configure_candidate(
        runtime_source=args.runtime_source,
        graph_asset=args.graph_asset,
        expected_runtime_sha256=args.expected_runtime_sha256,
        expected_graph_sha256=args.expected_graph_sha256,
    )
    if args.phase == "preflight":
        return base.preflight_phase(output_root)

    legacy = base.configure_legacy()
    output_root.mkdir(parents=True, exist_ok=True)
    cases = legacy.load_cases()
    if args.phase == "parity":
        return base.parity_phase(cases, output_root, args.jobs)
    if args.phase == "score":
        return base.score_phase(cases, output_root)
    if args.phase == "finalize":
        return finalize_phase_onnx(output_root)

    status = base.parity_phase(cases, output_root, args.jobs)
    if status != 0:
        print(
            "ONNX parity threshold failed; scoring was not started",
            file=sys.stderr,
            flush=True,
        )
        return status
    status = base.score_phase(cases, output_root)
    if status != 0:
        return status
    return finalize_phase_onnx(output_root)


if __name__ == "__main__":
    raise SystemExit(main())
