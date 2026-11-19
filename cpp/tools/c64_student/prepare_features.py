#!/usr/bin/env python3
"""Prepare C64 native features and frozen DPDFNet-2 teacher targets."""

from __future__ import annotations

import os

# Set thread limits before numpy / onnxruntime import in every process.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import shutil
import socket
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


SR = 16_000
SAMPLES = 640_000
HOP = 160
WINDOW = 320
N_FFT = 512
BINS = 257
FRAMES = 4000
FEATURE_WIDTH = 641
FEATURE_EPS = 1e-8
EXPECTED_PROBE_SHA256 = "4d1b3f139214bfad11a6c80e635635026dd2b0c3ce98fc5c4992c8ea4ce5209a"
EXPECTED_MODEL_SHA256 = "4f0ee28935b4a32abecc717d745416976565834d839601acf43031094b4dc94c"

_teacher_module = None
_teacher_runtime = None
_worker_config: dict[str, Any] = {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def pcm16_lround(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match C++ clamp(x*32768,-32768,32767), std::lround, /32768."""
    value = np.asarray(samples, dtype=np.float32)
    scaled = np.clip(value * np.float32(32768.0), np.float32(-32768.0), np.float32(32767.0))
    # C++ std::lround rounds halfway cases away from zero, unlike np.rint.
    quantized = np.copysign(np.floor(np.abs(scaled) + np.float32(0.5)), scaled).astype(np.int16)
    decoded = quantized.astype(np.float32) / 32768.0
    return quantized, decoded


def read_float_wav(path: Path) -> np.ndarray:
    samples, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if samples.ndim != 2 or samples.shape[1] != 1:
        raise ValueError(f"expected strict mono WAV: {path} has shape {samples.shape}")
    samples = np.asarray(samples[:, 0], dtype=np.float32).reshape(-1)
    if int(rate) != SR:
        raise ValueError(f"expected 16 kHz WAV: {path} ({rate})")
    if samples.size != SAMPLES:
        raise ValueError(f"expected {SAMPLES} samples: {path} ({samples.size})")
    if not np.all(np.isfinite(samples)):
        raise ValueError(f"non-finite WAV: {path}")
    return samples


def project_band_power(spectrum_power: np.ndarray) -> np.ndarray:
    """Match ResidualFeatureBands::project_into for a [frames,257] power matrix."""
    power = np.asarray(spectrum_power, dtype=np.float64)
    if power.ndim != 2 or power.shape[1] != BINS:
        raise ValueError(f"unexpected spectrum power shape: {power.shape}")
    nyquist = SR * 0.5
    centers = np.empty(64, dtype=np.float64)
    for band in range(64):
        if band < 24:
            centers[band] = 1000.0 * band / 23.0
        elif band < 40:
            centers[band] = 1000.0 + 3000.0 * (band - 23) / 16.0
        else:
            centers[band] = 4000.0 + (nyquist - 4000.0) * (band - 39) / 24.0
    left = np.zeros(BINS, dtype=np.int64)
    right = np.zeros(BINS, dtype=np.float64)
    normalization = np.zeros(64, dtype=np.float64)
    for band_index in range(BINS):
        hz = band_index * SR / N_FFT
        band = 0
        while band + 1 < 63 and hz > centers[band + 1]:
            band += 1
        weight = float(np.clip((hz - centers[band]) / (centers[band + 1] - centers[band]), 0.0, 1.0))
        left[band_index] = band
        right[band_index] = weight
        normalization[band] += 1.0 - weight
        normalization[band + 1] += weight
    output = np.zeros((power.shape[0], 64), dtype=np.float64)
    for band_index in range(BINS):
        band = int(left[band_index])
        weight = right[band_index]
        output[:, band] += (1.0 - weight) * power[:, band_index]
        output[:, band + 1] += weight * power[:, band_index]
    output /= np.maximum(normalization[np.newaxis, :], 1e-12)
    return output.astype(np.float32)


def pcm16_self_test() -> None:
    values = np.array([-1.5, -0.5, 0.0, 0.5, 1.5], dtype=np.float32) / np.float32(32768.0)
    expected = np.array([-2, -1, 0, 1, 2], dtype=np.int16)
    actual, _ = pcm16_lround(values)
    if not np.array_equal(actual, expected):
        raise AssertionError(f"PCM16 ties-away fixture failed: {actual.tolist()} != {expected.tolist()}")


def contract_stft(samples: np.ndarray) -> np.ndarray:
    """[previous 160,current 160]*sine, zero-pad to 512, rfft; no centering."""
    value = np.asarray(samples, dtype=np.float32).reshape(-1)
    if value.size != SAMPLES:
        raise ValueError(f"STFT contract requires {SAMPLES} samples, got {value.size}")
    padded = np.concatenate([np.zeros(HOP, dtype=np.float32), value])
    frames = np.lib.stride_tricks.sliding_window_view(padded, WINDOW)[::HOP][:FRAMES]
    if frames.shape != (FRAMES, WINDOW):
        raise AssertionError(f"unexpected STFT frame shape: {frames.shape}")
    n = np.arange(WINDOW, dtype=np.float64)
    sine = np.sin(math.pi * (n + 0.5) / WINDOW).astype(np.float32)
    padded_frames = np.zeros((FRAMES, N_FFT), dtype=np.float32)
    padded_frames[:, :WINDOW] = frames * sine[np.newaxis, :]
    spectrum = np.fft.rfft(padded_frames, n=N_FFT, axis=1).astype(np.complex64)
    if spectrum.shape != (FRAMES, BINS):
        raise AssertionError(f"unexpected STFT spectrum shape: {spectrum.shape}")
    return spectrum


def feature_from_power_noise(power: np.ndarray, noise: np.ndarray) -> np.ndarray:
    power = np.asarray(power, dtype=np.float64)
    noise = np.asarray(noise, dtype=np.float64)
    power_db = 10.0 * np.log10(np.maximum(power, FEATURE_EPS))
    noise_db = 10.0 * np.log10(np.maximum(noise, FEATURE_EPS))
    delta = np.empty_like(power_db)
    delta[0] = 0.0
    delta[1:] = power_db[1:] - power_db[:-1]
    return np.concatenate([power_db, noise_db, power_db - noise_db, delta], axis=1).astype(np.float32)


def load_native_vector(path: Path) -> np.ndarray:
    expected_bytes = FRAMES * FEATURE_WIDTH * 4
    size = path.stat().st_size
    if size != expected_bytes:
        raise ValueError(f"native feature byte count mismatch: {path} has {size}, expected {expected_bytes}")
    vector = np.fromfile(path, dtype="<f4")
    if vector.size != FRAMES * FEATURE_WIDTH:
        raise ValueError(f"native feature count mismatch: {vector.size}")
    matrix = vector.reshape(FRAMES, FEATURE_WIDTH)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("native feature output contains NaN or infinity")
    return matrix


def import_frozen_module(infer_py: Path):
    spec = importlib.util.spec_from_file_location("c64_frozen_dpdfnet_infer", str(infer_py))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import frozen infer module: {infer_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def init_worker(config: dict[str, Any]) -> None:
    global _teacher_module, _teacher_runtime, _worker_config
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    _worker_config = config
    _teacher_module = import_frozen_module(Path(config["infer_py"]))
    _teacher_runtime = _teacher_module.load_runtime(Path(config["model_path"]))
    if _teacher_runtime.model_sha256 != config["model_sha256"]:
        raise RuntimeError(f"teacher model hash changed: {_teacher_runtime.model_sha256}")


def process_row(row: dict[str, Any]) -> dict[str, Any]:
    if _teacher_module is None or _teacher_runtime is None:
        raise RuntimeError("teacher worker was not initialized")
    row_id = row["id"]
    features_dir = Path(_worker_config["features_dir"])
    native_input_dir = features_dir / "native-inputs"
    native_dir = features_dir / "native"
    teacher_dir = features_dir / "teacher"
    npz_dir = features_dir / "npz"
    for directory in (native_input_dir, native_dir, teacher_dir, npz_dir):
        directory.mkdir(parents=True, exist_ok=True)

    mix = read_float_wav(Path(row["mix_path"]))
    clean = read_float_wav(Path(row["clean_path"]))
    noise = read_float_wav(Path(row["noise_path"]))
    for key, path_key in (("mix_sha256", "mix_path"), ("clean_sha256", "clean_path"), ("noise_sha256", "noise_path")):
        expected = row.get(key)
        if expected and sha256_file(Path(row[path_key])) != expected:
            raise AssertionError(f"{key} mismatch for {row_id}")
    max_sum_error = float(np.max(np.abs(mix - (clean + noise))))
    if max_sum_error > 1e-7:
        raise AssertionError(f"mix != clean + noise for {row_id}: {max_sum_error}")

    quantized, decoded_mix = pcm16_lround(mix)
    quantized_path = native_input_dir / f"{row_id}.mix-pcm16.wav"
    sf.write(str(quantized_path), quantized, SR, subtype="PCM_16")
    decoded_from_file, rate = sf.read(str(quantized_path), dtype="float32", always_2d=False)
    if int(rate) != SR:
        raise AssertionError("PCM16 input sample rate changed")
    decoded_from_file = np.asarray(decoded_from_file, dtype=np.float32).reshape(-1)
    if decoded_from_file.size != SAMPLES:
        raise AssertionError("PCM16 input length changed")
    if not np.array_equal(decoded_from_file, decoded_mix):
        raise AssertionError("PCM16 WAV roundtrip differs from C++ lround decode")

    probe = Path(_worker_config["probe"])
    probe_hash = sha256_file(probe)
    if probe_hash != _worker_config["probe_sha256"]:
        raise RuntimeError(f"native probe hash changed: {probe_hash}")
    native_features = native_dir / f"{row_id}.raw.f32"
    probe_prefix = native_dir / f"{row_id}.probe"
    command = [
        str(probe),
        "--mic", str(quantized_path),
        "--features", str(native_features),
        "--floor-db", "-9",
        "--absolute-floor-db", "-9",
        "--out", str(probe_prefix),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, env=os.environ.copy())
    if completed.returncode != 0:
        raise RuntimeError(f"native probe failed for {row_id}: {completed.stderr or completed.stdout}")
    if not native_features.is_file():
        raise FileNotFoundError(f"native probe did not write {native_features}")
    native = load_native_vector(native_features)
    x = np.ascontiguousarray(native[:, :256], dtype=np.float32)
    power = np.ascontiguousarray(native[:, 256:320], dtype=np.float32)
    native_noise = np.ascontiguousarray(native[:, 320:384], dtype=np.float32)
    base_gain = np.ascontiguousarray(native[:, 384:641], dtype=np.float32)
    if np.min(power) < 0.0 or np.min(native_noise) < 0.0:
        raise AssertionError(f"negative projected power for {row_id}")
    x_recomputed = feature_from_power_noise(power, native_noise)
    x_max_error = float(np.max(np.abs(x - x_recomputed)))
    if x_max_error > 1e-4:
        raise AssertionError(f"native feature formula mismatch for {row_id}: {x_max_error}")
    if np.min(base_gain) < -1e-6 or np.max(base_gain) > 1.000001:
        raise AssertionError(f"base_gain outside [0,1] for {row_id}")

    teacher_wave, stream_info = _teacher_module.enhance_stream(
        decoded_from_file,
        _teacher_runtime,
        chunk_size=4096,
        tail_frames=_teacher_module.TAIL_GUARD_FRAMES,
    )
    teacher_wave = np.ascontiguousarray(teacher_wave, dtype=np.float32)
    if teacher_wave.size != SAMPLES:
        raise AssertionError(f"teacher alignment failed for {row_id}: {teacher_wave.size}")
    if not np.all(np.isfinite(teacher_wave)):
        raise AssertionError(f"teacher output is non-finite for {row_id}")
    teacher_path = teacher_dir / f"{row_id}.teacher.wav"
    sf.write(str(teacher_path), teacher_wave, SR, subtype="FLOAT")

    mix_spec = contract_stft(decoded_from_file)
    clean_spec = contract_stft(clean)
    teacher_spec = contract_stft(teacher_wave)
    mix_band_power = project_band_power(np.abs(mix_spec.astype(np.complex64)) ** 2)
    if not np.allclose(mix_band_power, power, rtol=5e-4, atol=5e-5):
        absolute = np.abs(mix_band_power.astype(np.float64) - power.astype(np.float64))
        relative = absolute / np.maximum(np.abs(power.astype(np.float64)), 1e-12)
        raise AssertionError(
            f"STFT/projected-power alignment failed for {row_id}: "
            f"max_abs={absolute.max()} max_rel={relative.max()}"
        )
    padding_mask = np.ones((FRAMES, 2), dtype=np.float32)
    padding_mask[0, 0] = 0.0
    power_mask = np.isfinite(power).astype(np.float32)
    noise_mask = np.isfinite(native_noise).astype(np.float32)
    stage_metadata = np.zeros((FRAMES, 5), dtype=np.float32)
    stage_metadata[:, 0] = np.arange(FRAMES, dtype=np.float32)
    stage_metadata[:, 1] = np.arange(FRAMES, dtype=np.float32) * HOP / SR
    stage_metadata[:, 2] = np.arange(FRAMES, dtype=np.float32) * HOP
    stage_metadata[:, 3] = padding_mask[:, 0]
    stage_metadata[:, 4] = -9.0

    for name, array in {
        "x": x,
        "base_gain": base_gain,
        "power": power,
        "noise": native_noise,
        "stage_metadata": stage_metadata,
        "padding_mask": padding_mask,
        "power_mask": power_mask,
        "noise_mask": noise_mask,
    }.items():
        if not np.all(np.isfinite(array)):
            raise AssertionError(f"non-finite {name} for {row_id}")
    for name, array in {"mix_spec": mix_spec, "clean_spec": clean_spec, "teacher_spec": teacher_spec}.items():
        if not np.all(np.isfinite(array.real)) or not np.all(np.isfinite(array.imag)):
            raise AssertionError(f"non-finite {name} for {row_id}")
    if x.shape != (FRAMES, 256) or base_gain.shape != (FRAMES, BINS):
        raise AssertionError(f"runtime feature shape mismatch for {row_id}")
    if power.shape != (FRAMES, 64) or native_noise.shape != (FRAMES, 64):
        raise AssertionError(f"projected feature shape mismatch for {row_id}")
    for array in (mix_spec, clean_spec, teacher_spec):
        if array.shape != (FRAMES, BINS) or array.dtype != np.complex64:
            raise AssertionError(f"spectrum contract mismatch for {row_id}: {array.shape} {array.dtype}")

    npz_path = npz_dir / f"{row_id}.npz"
    np.savez(
        npz_path,
        x=x,
        base_gain=base_gain,
        mix_spec=mix_spec,
        clean_spec=clean_spec,
        teacher_spec=teacher_spec,
        power=power,
        noise=native_noise,
        stage_metadata=stage_metadata,
        padding_mask=padding_mask,
        power_mask=power_mask,
        noise_mask=noise_mask,
    )
    npz_hash = sha256_file(npz_path)
    teacher_hash = sha256_file(teacher_path)
    return {
        "id": row_id,
        "npz_path": str(npz_path),
        "npz_sha256": npz_hash,
        "teacher_path": str(teacher_path),
        "teacher_sha256": teacher_hash,
        "native_input_path": str(quantized_path),
        "native_input_sha256": sha256_file(quantized_path),
        "native_feature_path": str(native_features),
        "native_feature_sha256": sha256_file(native_features),
        "native_probe_prefix": str(probe_prefix),
        "frames": FRAMES,
        "samples": SAMPLES,
        "x_max_formula_error": x_max_error,
        "mix_clean_noise_max_abs_error": max_sum_error,
        "stft_native_power_max_abs_error": float(np.max(np.abs(mix_band_power - power))),
        "teacher_alignment": {
            "output_samples": int(teacher_wave.size),
            "trim_samples": int(stream_info.trim_samples),
            "latency_samples": int(stream_info.latency_samples),
            "tail_guard_frames": int(_teacher_module.TAIL_GUARD_FRAMES),
            "tail_guard_samples": int(_teacher_module.TAIL_GUARD_FRAMES * HOP),
            "frames_processed": int(stream_info.frames_processed),
            "raw_samples": int(stream_info.raw_samples),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--probe-sha256", default=EXPECTED_PROBE_SHA256)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--frozen-infer", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive")
    pcm16_self_test()
    started = time.time()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_manifest_path = output_dir / "manifest.json"
    if feature_manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"feature manifest exists; pass --overwrite only after explicit review: {feature_manifest_path}")
    project_root = Path(__file__).resolve().parents[2]
    frozen_infer = (args.frozen_infer or (project_root / "benchmarks/ns/dpdfnet2-16k-v1/frozen/infer.py")).resolve()
    model_path = (args.model or (project_root / "build/ns-next-20261115/dpdfnet/assets/dpdfnet2.onnx")).resolve()
    probe = args.probe.resolve()
    if not probe.is_file():
        raise FileNotFoundError(f"native probe not found: {probe}")
    if not model_path.is_file():
        raise FileNotFoundError(f"teacher model not found: {model_path}")
    probe_hash = sha256_file(probe)
    if probe_hash != args.probe_sha256:
        raise ValueError(f"native probe SHA mismatch: got {probe_hash}, expected {args.probe_sha256}")
    model_hash = sha256_file(model_path)
    if model_hash != EXPECTED_MODEL_SHA256:
        raise ValueError(f"teacher model SHA mismatch: got {model_hash}, expected {EXPECTED_MODEL_SHA256}")
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_manifest_sha = sha256_file(manifest_path)
    rows = source_manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != 152:
        raise ValueError(f"expected 152 manifest rows, got {len(rows) if isinstance(rows, list) else 'invalid'}")
    split_counts: dict[str, int] = {}
    language_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        split = row["split"]
        language = row["language"]
        split_counts[split] = split_counts.get(split, 0) + 1
        language_counts.setdefault(split, {})
        language_counts[split][language] = language_counts[split].get(language, 0) + 1
    if split_counts != {"train": 128, "dev": 24}:
        raise ValueError(f"unexpected split counts: {split_counts}")
    if language_counts.get("train") != {"zh": 64, "en": 64} or language_counts.get("dev") != {"zh": 12, "en": 12}:
        raise ValueError(f"unexpected language balance: {language_counts}")
    log_dir = project_root / "build/ns-distilled-c64-20261116/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"features-{stamp}.log"

    def log(message: str) -> None:
        line = f"[{utc_now()}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    disk_before = shutil.disk_usage(output_dir)
    config = {
        "probe": str(probe),
        "probe_sha256": probe_hash,
        "infer_py": str(frozen_infer),
        "model_path": str(model_path),
        "model_sha256": model_hash,
        "features_dir": str(output_dir),
    }
    log(f"start rows={len(rows)} jobs={args.jobs} probe_sha256={probe_hash} model_sha256={model_hash}")
    results: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=init_worker, initargs=(config,)) as executor:
        future_to_id = {executor.submit(process_row, row): row["id"] for row in rows}
        completed_count = 0
        for future in as_completed(future_to_id):
            row_id = future_to_id[future]
            result = future.result()
            results[row_id] = result
            completed_count += 1
            if completed_count == 1 or completed_count % 8 == 0 or completed_count == len(rows):
                log(f"progress {completed_count}/{len(rows)} last={row_id}")
    if len(results) != len(rows):
        raise AssertionError("missing feature results")
    feature_rows = []
    for row in rows:
        result = results[row["id"]]
        feature_rows.append({**row, **result})
    feature_contract = {
        "schema_version": 1,
        "sample_rate": SR,
        "samples": SAMPLES,
        "frames": FRAMES,
        "hop_length": HOP,
        "window_length": WINDOW,
        "n_fft": N_FFT,
        "bins": BINS,
        "window": "sine",
        "center": False,
        "frame_policy": "[previous 160 PCM,current 160 PCM] * sine; zero-pad to 512; rfft",
        "first_frame_previous_half": "zero_padded",
        "mix_quantization": "clip(x*32768,-32768,32767); C++ std::lround; divide by 32768",
        "runtime_inputs": ["x", "base_gain"],
        "targets": ["mix_spec", "clean_spec", "teacher_spec"],
        "npz_path_key": "npz_path",
        "npz_hash_key": "npz_sha256",
        "teacher_hash_key": "teacher_sha256",
        "npz_fields": {
            "x": {"shape": [FRAMES, 256], "dtype": "float32"},
            "base_gain": {"shape": [FRAMES, BINS], "dtype": "float32", "semantics": "OM-LSA base gain; never called SPP"},
            "mix_spec": {"shape": [FRAMES, BINS], "dtype": "complex64"},
            "clean_spec": {"shape": [FRAMES, BINS], "dtype": "complex64"},
            "teacher_spec": {"shape": [FRAMES, BINS], "dtype": "complex64"},
            "power": {"shape": [FRAMES, 64], "dtype": "float32"},
            "noise": {"shape": [FRAMES, 64], "dtype": "float32"},
            "stage_metadata": {"shape": [FRAMES, 5], "dtype": "float32", "columns": ["frame_index", "time_s", "current_sample", "previous_half_valid", "residual_floor_db"]},
            "padding_mask": {"shape": [FRAMES, 2], "dtype": "float32", "columns": ["previous_160_valid", "current_160_valid"]},
            "power_mask": {"shape": [FRAMES, 64], "dtype": "float32"},
            "noise_mask": {"shape": [FRAMES, 64], "dtype": "float32"},
        },
        "native_vector_slices": {"x": [0, 256], "power": [256, 320], "noise": [320, 384], "base_gain": [384, 641]},
        "teacher_alignment": "frozen enhance_stream trims 640 samples and returns exactly 640000 aligned samples; 12-hop tail guard",
    }
    feature_manifest = {
        "schema_version": 1,
        "dataset_id": source_manifest.get("dataset_id"),
        "created_utc": utc_now(),
        "source_manifest": {"path": str(manifest_path), "sha256": source_manifest_sha},
        "native_probe": {"path": str(probe), "sha256": probe_hash},
        "teacher": {"frozen_infer": str(frozen_infer), "frozen_infer_sha256": sha256_file(frozen_infer), "model_path": str(model_path), "model_sha256": model_hash},
        "feature_contract": feature_contract,
        "rows": feature_rows,
        "validation": {
            "rows": len(feature_rows),
            "all_frames_4000": all(row["frames"] == FRAMES for row in feature_rows),
            "all_samples_640000": all(row["samples"] == SAMPLES for row in feature_rows),
            "max_x_formula_error": max(row["x_max_formula_error"] for row in feature_rows),
            "max_mix_clean_noise_error": max(row["mix_clean_noise_max_abs_error"] for row in feature_rows),
            "all_teacher_aligned": all(row["teacher_alignment"]["output_samples"] == SAMPLES for row in feature_rows),
        },
    }
    atomic_json(feature_manifest_path, feature_manifest)
    # Keep data/manifest.json as the shared source of truth while also
    # publishing the feature manifest requested by the trainer.
    updated_source = json.loads(json.dumps(source_manifest))
    by_id = {row["id"]: row for row in feature_rows}
    for row in updated_source["rows"]:
        extension = by_id[row["id"]]
        for key in ("npz_path", "npz_sha256", "teacher_path", "teacher_sha256", "native_feature_path", "native_feature_sha256", "frames"):
            row[key] = extension[key]
    updated_source["features"] = {
        "manifest_path": str(feature_manifest_path),
        "manifest_sha256": sha256_file(feature_manifest_path),
        "native_probe_sha256": probe_hash,
        "teacher_model_sha256": model_hash,
    }
    atomic_json(manifest_path, updated_source)
    disk_after = shutil.disk_usage(output_dir)
    receipt = {
        "schema_version": 1,
        "status": "ok",
        "created_utc": utc_now(),
        "elapsed_seconds": time.time() - started,
        "training_source_manifest": {"path": str(manifest_path), "before_sha256": source_manifest_sha, "after_sha256": sha256_file(manifest_path)},
        "feature_manifest": {"path": str(feature_manifest_path), "sha256": sha256_file(feature_manifest_path)},
        "teacher": {"frozen_infer": str(frozen_infer), "frozen_infer_sha256": sha256_file(frozen_infer), "model_path": str(model_path), "model_sha256": model_hash},
        "native_probe": {"path": str(probe), "sha256": probe_hash},
        "feature_contract": feature_contract,
        "validation": feature_manifest["validation"],
        "space": {"before_free": disk_before.free, "after_free": disk_after.free, "delta_used": disk_before.free - disk_after.free},
        "environment": {"hostname": socket.gethostname(), "python": sys.version, "numpy": np.__version__, "soundfile": sf.__version__, "platform": platform.platform(), "jobs": args.jobs, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")},
        "log": {"path": str(log_path), "sha256": sha256_file(log_path)},
    }
    atomic_json(output_dir / "receipt.json", receipt)
    log(f"complete rows={len(feature_rows)} feature_manifest_sha256={receipt['feature_manifest']['sha256']} elapsed={receipt['elapsed_seconds']:.1f}s")
    print(json.dumps(feature_manifest["validation"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
