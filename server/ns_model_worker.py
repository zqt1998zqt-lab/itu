"""Isolated allowlisted NS model worker.

This process loads exactly one pinned adapter/model (including the native
48 kHz DPDFNet2 HR route), emits one aligned FLOAT utterance, and exits.
Torch and ONNX Runtime state never enters the shared server process.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.ns_routes import EXTERNAL_NS_IDS, NS_ROUTES, NSRouteSpec  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_sha(samples: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(samples, dtype="<f4").tobytes()).hexdigest()


def _read_float(path: Path, *, sample_rate: int = 16_000,
                require_quantized: bool = False) -> np.ndarray:
    info = sf.info(path)
    if info.samplerate != sample_rate or info.channels != 1 or info.subtype != "FLOAT":
        raise ValueError(
            f"model input must be mono {sample_rate} Hz IEEE FLOAT32 WAV"
        )
    values, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != sample_rate or values.shape[1] != 1 or not len(values):
        raise ValueError("model input geometry is invalid")
    samples = np.ascontiguousarray(values[:, 0], dtype=np.float32)
    if not np.isfinite(samples).all():
        raise ValueError("model input contains NaN or Inf")
    if require_quantized:
        scaled = samples.astype(np.float64) * 32768.0
        if np.max(np.abs(scaled - np.rint(scaled)), initial=0.0) > 1e-5:
            raise ValueError("model input is not the frozen PCM16-lround float grid")
    return samples


def _write_float(path: Path, samples: np.ndarray, *,
                 sample_rate: int = 16_000) -> None:
    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("model output is empty or non-finite")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, values, sample_rate, subtype="FLOAT", format="WAV")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _assets(spec: NSRouteSpec) -> list[tuple[Path, str]]:
    return [(spec.frontend_path, spec.frontend_sha256), (spec.model_path, spec.model_sha256), *spec.extra_assets]


def _verify_assets(spec: NSRouteSpec) -> dict[str, str]:
    result: dict[str, str] = {}
    for path, expected in _assets(spec):
        if not path.is_file():
            raise FileNotFoundError(f"pinned NS asset is missing: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"pinned NS asset SHA-256 mismatch: {path} ({actual})")
        result[str(path)] = actual
    return result


def _load_adapter(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned NS adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves postponed annotations through sys.modules during
    # class creation, so the isolated adapter name must be registered first.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _run_cofilite(spec: NSRouteSpec, input_path: Path) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    adapter = _load_adapter(spec.frontend_path, "itu_ns_cofilite_adapter")
    waveform = _read_float(
        input_path, sample_rate=spec.sample_rate, require_quantized=True
    )
    started = time.perf_counter()
    aligned, info = adapter.enhance_stream(waveform, chunk_size=spec.frame_samples)
    aligned = np.asarray(aligned, dtype=np.float32)
    elapsed = time.perf_counter() - started
    return aligned, {
        "offline_whole_utterance": False,
        "streaming_precomputed_offline_replay": True,
        "lookahead_samples_observed": spec.lookahead_samples,
        "lookahead_treated_as_zero": False,
        "adapter": f"CoFiLiteStream/process({spec.frame_samples})/flush",
        "streaming_details": info,
        "elapsed_sec": float(elapsed),
    }, waveform


def _run_dpdfnet2(spec: NSRouteSpec, input_path: Path) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    adapter = _load_adapter(spec.frontend_path, "itu_ns_dpdfnet2_adapter")
    waveform = _read_float(
        input_path, sample_rate=spec.sample_rate, require_quantized=True
    )
    runtime = adapter.load_runtime(spec.model_path)
    started = time.perf_counter()
    aligned, info = adapter.enhance_stream(
        waveform, runtime, chunk_size=spec.frame_samples
    )
    elapsed = time.perf_counter() - started
    aligned = np.asarray(aligned, dtype=np.float32)
    return aligned, {
        "offline_whole_utterance": False,
        "streaming_precomputed_offline_replay": True,
        "chunk_samples": spec.frame_samples,
        "algorithm_latency_samples": int(info.latency_samples),
        "trim_samples": int(info.trim_samples),
        "frames_processed": int(info.frames_processed),
        "elapsed_sec": float(elapsed),
        "frontend": "pinned infer.py/load_runtime/enhance_stream",
    }, waveform


def _run_dpdfnet2_48k_hr(
    spec: NSRouteSpec, input_path: Path
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    adapter = _load_adapter(spec.frontend_path, "itu_ns_dpdfnet2_48k_hr_adapter")
    waveform = _read_float(
        input_path, sample_rate=spec.sample_rate, require_quantized=True
    )
    runtime = adapter.load_runtime(spec.model_path)
    started = time.perf_counter()
    aligned, info = adapter.enhance_stream(
        waveform, runtime, chunk_size=spec.frame_samples
    )
    elapsed = time.perf_counter() - started
    aligned = np.asarray(aligned, dtype=np.float32)
    return aligned, {
        "offline_whole_utterance": False,
        "streaming_precomputed_offline_replay": True,
        "sample_rate": spec.sample_rate,
        "chunk_samples": spec.frame_samples,
        "window_samples": spec.window_samples,
        "hop_samples": spec.hop_samples,
        "algorithm_latency_samples": int(info.latency_samples),
        "trim_samples": int(info.trim_samples),
        "frames_processed": int(info.frames_processed),
        "inference_sec": float(info.inference_sec),
        "elapsed_sec": float(elapsed),
        "frontend": "pinned DPDFNet2 48 kHz HR StreamingEnhancer",
    }, waveform


def _run_hybrid_v2(spec: NSRouteSpec, input_path: Path, receipt_dir: Path) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    waveform = _read_float(
        input_path, sample_rate=spec.sample_rate, require_quantized=True
    )
    probe_prefix = receipt_dir / "v2-probe"
    command = [
        str(spec.frontend_path), "--mic", str(input_path), "--out", str(probe_prefix),
        "--model", str(spec.model_path), "--floor-db", "-9", "--absolute-floor-db", "-21",
        "--mix", "1", "--pitch-strength", ".5",
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, timeout=240, check=False)
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()
        raise RuntimeError("hybrid_v2_probe failed: " + (detail[-1] if detail else str(completed.returncode)))
    report_path = Path(str(probe_prefix) + ".json")
    output_path = Path(str(probe_prefix) + ".after.wav")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("modelstate") != "ready" or int(report.get("applied") or 0) <= 0 or int(report.get("fallback") or 0) != 0:
        raise ValueError("hybrid_v2_probe did not prove ready/applied/no-fallback execution")
    if int(report.get("delay") or 0) != spec.delay_samples:
        raise ValueError("hybrid_v2_probe delay changed from the pinned route delay")
    raw = _read_float(output_path, sample_rate=spec.sample_rate)
    if raw.size != waveform.size + spec.delay_samples:
        raise ValueError("hybrid_v2_probe output must be N+delay samples")
    aligned = np.ascontiguousarray(
        raw[spec.delay_samples:spec.delay_samples + waveform.size],
        dtype=np.float32,
    )
    return aligned, {
        "offline_whole_utterance": False,
        "streaming_precomputed_offline_replay": True,
        "raw_output_samples": int(raw.size),
        "physical_delay_samples": spec.delay_samples,
        "probe_report": report,
        "elapsed_sec": float(elapsed),
        "command": command,
    }, waveform


def run(backend: str, input_path: Path, output_path: Path, receipt_path: Path) -> dict[str, Any]:
    if backend not in EXTERNAL_NS_IDS:
        raise ValueError(f"unsupported NS backend: {backend}")
    spec = NS_ROUTES[backend]
    before = _verify_assets(spec)
    receipt_dir = receipt_path.parent
    receipt_dir.mkdir(parents=True, exist_ok=True)
    if backend == "cofilite":
        aligned, details, waveform = _run_cofilite(spec, input_path)
        raw = aligned
    elif backend == "dpdfnet2":
        aligned, details, waveform = _run_dpdfnet2(spec, input_path)
        raw = aligned
    elif backend == "dpdfnet2_48k_hr":
        aligned, details, waveform = _run_dpdfnet2_48k_hr(spec, input_path)
        raw = aligned
    else:
        aligned, details, waveform = _run_hybrid_v2(spec, input_path, receipt_dir)
        raw = _read_float(
            Path(str(receipt_dir / "v2-probe") + ".after.wav"),
            sample_rate=spec.sample_rate,
        )
    if aligned.shape != waveform.shape or not np.isfinite(aligned).all():
        raise ValueError("model output is not finite and time-aligned to the input")
    _write_float(output_path, aligned, sample_rate=spec.sample_rate)
    written = _read_float(output_path, sample_rate=spec.sample_rate)
    if not np.array_equal(written, aligned):
        raise ValueError("written aligned FLOAT WAV differs from model output")
    after = _verify_assets(spec)
    if before != after:
        raise ValueError("NS model assets changed during inference")
    input_sha = _sha256(input_path)
    output_sha = _sha256(output_path)
    receipt = {
        "schema_version": 1,
        "backend": backend,
        "requested": backend,
        "effective": backend,
        "model_id": spec.model_id,
        "model_path": str(spec.model_path),
        "model_sha256": spec.model_sha256,
        "frontend_path": str(spec.frontend_path),
        "frontend_sha256": spec.frontend_sha256,
        "processing_mode": spec.processing_mode,
        "delay_samples": spec.delay_samples,
        "sample_rate": spec.sample_rate,
        "frame_samples": spec.frame_samples,
        "window_samples": spec.window_samples,
        "hop_samples": spec.hop_samples,
        "offline_whole_utterance": spec.offline_whole_utterance,
        "lookahead_samples": spec.lookahead_samples,
        "input": {
            "path": str(input_path), "wav_sha256": input_sha,
            "payload_sha256": _payload_sha(waveform), "samples": int(waveform.size),
            "sample_rate": spec.sample_rate, "channels": 1, "subtype": "FLOAT",
        },
        "raw_output": {
            "payload_sha256": _payload_sha(raw), "samples": int(raw.size),
            "sample_rate": spec.sample_rate,
            "finite": bool(np.isfinite(raw).all()),
            "kind": "aligned_only" if raw is aligned else "physical_delay_output",
        },
        "aligned_output": {
            "path": str(output_path), "wav_sha256": output_sha,
            "payload_sha256": _payload_sha(aligned), "samples": int(aligned.size),
            "sample_rate": spec.sample_rate, "channels": 1, "subtype": "FLOAT",
            "finite": True,
        },
        "inference": {
            "worker_invocations": 1,
            "onnx_inferences": int(details.get("frames_processed") or 1),
            "device": "cpu", "threads": 1,
            "elapsed_sec": float(details.get("elapsed_sec") or 0.0),
            "fallback_frames": 0, "clean_reference_read": False,
        },
        "pcm16_quantization": {
            "worker_input_applied": True,
            "probe_internal_applied": backend == "hybrid_v2_pitch",
            "effective_applications": 1,
            "idempotent_qfloat": True,
            "clip": "[-32768,32767]",
            "scale": 32768,
            "rounding": "lround_half_away_from_zero",
        },
        "assets_before": before,
        "assets_after": after,
        "details": details,
    }
    _write_json(receipt_path, receipt)
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=sorted(EXTERNAL_NS_IDS), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    os.environ.update({
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "TORCH_NUM_THREADS": "1", "ORT_NUM_THREADS": "1",
    })
    args = parse_args(argv)
    receipt = run(args.backend, args.input, args.output, args.receipt)
    print(json.dumps({
        "backend": receipt["backend"], "model_sha256": receipt["model_sha256"],
        "frontend_sha256": receipt["frontend_sha256"], "delay_samples": receipt["delay_samples"],
        "aligned_samples": receipt["aligned_output"]["samples"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
