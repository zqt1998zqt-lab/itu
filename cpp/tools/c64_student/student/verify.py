#!/usr/bin/env python3
"""Structure and optional native-parity verification for the C64 student."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch

if __package__:
    from .export import (
        FORMAT_MAGIC,
        HEADER_FIELDS,
        HEADER_STRUCT,
        WEIGHT_ORDER,
        export_checkpoint,
        fnv1a64,
        sha256_file,
    )
    from .model import (
        CausalFeatureState,
        FEATURE_DIM,
        FFT_SIZE,
        HIDDEN_SIZE,
        HOP_SAMPLES,
        MAC_PER_FRAME,
        NUM_BANDS,
        NUM_FFT_BINS,
        PARAMETER_COUNT,
        RESIDUAL_LIMIT_DB,
        ResidualStudent,
        SAMPLE_RATE_HZ,
        compose_causal_frame_features_with_state,
        project_fft_power_to_bands,
    )
else:
    from export import (
        FORMAT_MAGIC,
        HEADER_FIELDS,
        HEADER_STRUCT,
        WEIGHT_ORDER,
        export_checkpoint,
        fnv1a64,
        sha256_file,
    )
    from model import (
        CausalFeatureState,
        FEATURE_DIM,
        FFT_SIZE,
        HIDDEN_SIZE,
        HOP_SAMPLES,
        MAC_PER_FRAME,
        NUM_BANDS,
        NUM_FFT_BINS,
        PARAMETER_COUNT,
        RESIDUAL_LIMIT_DB,
        ResidualStudent,
        SAMPLE_RATE_HZ,
        compose_causal_frame_features_with_state,
        project_fft_power_to_bands,
    )


STEP_TOLERANCE = 1e-5
FEATURE_TOLERANCE = 1e-4
NATIVE_DB_TOLERANCE = 1e-3
NATIVE_HIDDEN_TOLERANCE = 1e-3


def _max_abs(left: torch.Tensor | np.ndarray, right: torch.Tensor | np.ndarray) -> float:
    left_t = left if isinstance(left, torch.Tensor) else torch.from_numpy(left)
    right_t = right if isinstance(right, torch.Tensor) else torch.from_numpy(right)
    return float((left_t - right_t).abs().max().item())


def _all_finite(*values: torch.Tensor) -> bool:
    return all(bool(torch.isfinite(value).all().item()) for value in values)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _fixed_raw_db_features(
    frames: int,
    batch_size: int = 1,
    *,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    base = torch.randn(batch_size, frames, FEATURE_DIM, generator=generator) * 18.0
    channel_offsets = torch.tensor([-55.0, -25.0, 0.0, 18.0]).repeat_interleave(64)
    frame_ramp = torch.linspace(-8.0, 8.0, frames).view(1, frames, 1)
    features = base + channel_offsets.view(1, 1, -1) + frame_ramp
    return features.clamp(min=-120.0, max=60.0).contiguous()


def _load_model_from_checkpoint(checkpoint_path: Path) -> ResidualStudent:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = ResidualStudent(HIDDEN_SIZE).eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


def _make_native_fixture_checkpoint(source: Path, output_dir: Path) -> Path:
    """Create an explicit trained-flag=1 parity fixture without touching source."""

    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    source_metadata = checkpoint.get("metadata", {})
    source_sha = sha256_file(source)
    fixture_manifest = hashlib.sha256(
        f"native-parity-fixture:{source_sha}".encode("utf-8")
    ).hexdigest()
    run_id = str(source_metadata.get("run_id", "model-structure-verification"))
    fixture_path = output_dir / "native-fixture-checkpoint.pt"
    torch.save(
        {
            "state_dict": checkpoint["state_dict"],
            "metadata": {
                "steps": 1,
                "training_manifest_sha256": fixture_manifest,
                "run_id": run_id + "-native-fixture",
                "trained": True,
                "fixture": True,
                "feature_contract": "dpdf-frame-v1-db",
                "source_checkpoint_sha256": source_sha,
            },
        },
        fixture_path,
    )
    return fixture_path


def _feature_chunk_probe() -> dict[str, Any]:
    generator = torch.Generator().manual_seed(71)
    frames = 11
    batch_size = 2
    fft_power = torch.rand(batch_size, frames, NUM_FFT_BINS, generator=generator) + 0.01
    fft_noise = torch.rand(batch_size, frames, NUM_FFT_BINS, generator=generator) * 0.2 + 0.001
    band_power = project_fft_power_to_bands(fft_power)
    band_noise = project_fft_power_to_bands(fft_noise)
    sequence_features, next_state = compose_causal_frame_features_with_state(
        band_power, band_noise, CausalFeatureState.initial()
    )

    chunk1_state = CausalFeatureState.initial()
    chunk1 = []
    for frame in range(frames):
        value, chunk1_state = compose_causal_frame_features_with_state(
            band_power[:, frame : frame + 1, :],
            band_noise[:, frame : frame + 1, :],
            chunk1_state,
        )
        chunk1.append(value)
    chunk1_features = torch.cat(chunk1, dim=1)

    chunk3_state = CausalFeatureState.initial()
    chunk3 = []
    for start in range(0, frames, 3):
        value, chunk3_state = compose_causal_frame_features_with_state(
            band_power[:, start : start + 3, :],
            band_noise[:, start : start + 3, :],
            chunk3_state,
        )
        chunk3.append(value)
    chunk3_features = torch.cat(chunk3, dim=1)

    chunk1_error = _max_abs(sequence_features, chunk1_features)
    chunk3_error = _max_abs(sequence_features, chunk3_features)
    state_finite = next_state.previous_log_power is not None and _all_finite(
        next_state.previous_log_power
    )
    result = {
        "frames": frames,
        "chunk1_max_abs_diff": chunk1_error,
        "chunk3_max_abs_diff": chunk3_error,
        "tolerance": FEATURE_TOLERANCE,
        "state_finite": state_finite,
        "pass": chunk1_error <= FEATURE_TOLERANCE
        and chunk3_error <= FEATURE_TOLERANCE
        and state_finite,
    }
    if not result["pass"]:
        raise AssertionError("feature chunk parity failed")
    return result


def _model_sequence_probe(model: ResidualStudent) -> dict[str, Any]:
    frames = 17
    batch_size = 2
    features = _fixed_raw_db_features(frames, batch_size, seed=101)
    generator = torch.Generator().manual_seed(102)
    hidden = torch.randn(batch_size, HIDDEN_SIZE, generator=generator) * 0.2
    with torch.no_grad():
        sequence_residual, sequence_hidden = model(features, hidden)
        stepped_residual = []
        state = hidden
        for frame in range(frames):
            residual, state = model.step(features[:, frame : frame + 1, :], state)
            stepped_residual.append(residual[:, 0, :])
        stepped_residual_tensor = torch.stack(stepped_residual, dim=1)

        changed = features.clone()
        split = frames // 2
        changed[:, split:, :] += 1000.0
        changed_residual, _ = model(changed, hidden)

    step_error = _max_abs(sequence_residual, stepped_residual_tensor)
    hidden_error = _max_abs(sequence_hidden, state)
    prefix_error = _max_abs(
        sequence_residual[:, :split, :], changed_residual[:, :split, :]
    )
    finite = _all_finite(sequence_residual, sequence_hidden)
    bounds = bool(
        ((sequence_residual >= -RESIDUAL_LIMIT_DB) & (sequence_residual <= RESIDUAL_LIMIT_DB))
        .all()
        .item()
    )
    result = {
        "sequence_shape": list(sequence_residual.shape),
        "hidden_shape": list(sequence_hidden.shape),
        "step_max_abs_diff": step_error,
        "step_hidden_max_abs_diff": hidden_error,
        "future_prefix_max_abs_diff": prefix_error,
        "tolerance": STEP_TOLERANCE,
        "finite": finite,
        "residual_bounds_ok": bounds,
        "pass": step_error <= STEP_TOLERANCE
        and hidden_error <= STEP_TOLERANCE
        and prefix_error <= STEP_TOLERANCE
        and finite
        and bounds,
    }
    if not result["pass"]:
        raise AssertionError("sequence/step or causal-prefix verification failed")
    return result


def _onnx_state_probe(model: ResidualStudent, onnx_path: Path) -> dict[str, Any]:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    frames = 32
    features = _fixed_raw_db_features(frames, 1, seed=8802)
    initial_hidden = torch.linspace(-0.25, 0.25, HIDDEN_SIZE).view(1, HIDDEN_SIZE)
    shared_hidden = initial_hidden.clone()
    step_errors = []
    hidden_errors = []
    with torch.no_grad():
        for frame in range(frames):
            current = features[:, frame : frame + 1, :]
            torch_residual, torch_hidden = model.step(current, shared_hidden)
            native_residual, ort_hidden = session.run(
                ["residual_db", "hidden_out"],
                {"features": current.numpy(), "hidden_in": shared_hidden.numpy()},
            )
            step_errors.append(_max_abs(torch_residual, native_residual))
            hidden_errors.append(_max_abs(torch_hidden, ort_hidden))
            shared_hidden = torch.from_numpy(ort_hidden.copy())

    silence = torch.zeros(1, 1, FEATURE_DIM)
    silence[..., :NUM_BANDS] = -160.0
    silence[..., NUM_BANDS : 2 * NUM_BANDS] = -160.0
    shared_silent_hidden = torch.linspace(-0.2, 0.2, HIDDEN_SIZE).view(1, HIDDEN_SIZE)
    silence_finite = True
    silence_errors = []
    with torch.no_grad():
        for _ in range(32):
            torch_residual, torch_silent_hidden = model.step(silence, shared_silent_hidden)
            native_residual, ort_silent_hidden = session.run(
                ["residual_db", "hidden_out"],
                {"features": silence.numpy(), "hidden_in": shared_silent_hidden.numpy()},
            )
            silence_finite = silence_finite and np.isfinite(native_residual).all()
            silence_errors.append(_max_abs(torch_residual, native_residual))
            shared_silent_hidden = torch.from_numpy(ort_silent_hidden.copy())

    nonzero_error = max(step_errors)
    nonzero_hidden_error = max(hidden_errors)
    silence_error = max(silence_errors)
    silence_state_nonzero = bool(np.abs(ort_silent_hidden).max() > 0.0)
    result = {
        "nonzero_state_frames": frames,
        "nonzero_state_max_abs_diff": nonzero_error,
        "nonzero_state_hidden_max_abs_diff": nonzero_hidden_error,
        "silent_state_frames": 32,
        "silent_state_max_abs_diff": silence_error,
        "silent_state_finite": bool(silence_finite),
        "silence_does_not_implicitly_reset": silence_state_nonzero,
        "metric": "same_state_operator_parity",
        "continuous_independent_state": False,
        "state_feedback": "ort_hidden_out_feeds_both_next_torch_and_ort_steps",
        "tolerance": STEP_TOLERANCE,
        "pass": nonzero_error <= STEP_TOLERANCE
        and nonzero_hidden_error <= STEP_TOLERANCE
        and silence_error <= STEP_TOLERANCE
        and bool(silence_finite)
        and silence_state_nonzero,
    }
    if not result["pass"]:
        raise AssertionError("ONNX continuous-state verification failed")
    return result


def run_independent_state_parity(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    frames: int = 32,
    seeds: tuple[int, ...] = (8802, 8810),
) -> dict[str, Any]:
    """Compare PyTorch and ORT while each runtime feeds back its own state."""

    checkpoint_path = Path(checkpoint_path).resolve()
    output_dir = Path(output_dir).resolve()
    onnx_path = output_dir / "step.onnx"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)
    model = _load_model_from_checkpoint(checkpoint_path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    initial_hidden = torch.linspace(-0.25, 0.25, HIDDEN_SIZE).view(1, HIDDEN_SIZE)
    results: dict[str, Any] = {}
    all_same_state_pass = True
    all_same_state_runtime_pass = True
    all_independent_pass = True
    all_finite = True

    for seed in seeds:
        features = _fixed_raw_db_features(frames, 1, seed=seed)
        torch_hidden = initial_hidden.clone()
        ort_hidden = initial_hidden.numpy().copy()
        independent_residual_errors = []
        independent_hidden_errors = []
        finite = True
        with torch.no_grad():
            for frame in range(frames):
                current = features[:, frame : frame + 1, :]
                torch_residual, torch_hidden = model.step(current, torch_hidden)
                ort_residual, ort_hidden = session.run(
                    ["residual_db", "hidden_out"],
                    {
                        "features": current.numpy(),
                        "hidden_in": ort_hidden,
                    },
                )
                independent_residual_errors.append(
                    _max_abs(torch_residual, ort_residual)
                )
                independent_hidden_errors.append(_max_abs(torch_hidden, ort_hidden))
                finite = finite and bool(
                    np.isfinite(ort_residual).all() and np.isfinite(ort_hidden).all()
                )

        shared_hidden = initial_hidden.clone()
        same_state_residual_errors = []
        same_state_hidden_errors = []
        with torch.no_grad():
            for frame in range(frames):
                current = features[:, frame : frame + 1, :]
                torch_residual, torch_hidden = model.step(current, shared_hidden)
                ort_residual, ort_hidden = session.run(
                    ["residual_db", "hidden_out"],
                    {
                        "features": current.numpy(),
                        "hidden_in": shared_hidden.numpy(),
                    },
                )
                same_state_residual_errors.append(_max_abs(torch_residual, ort_residual))
                same_state_hidden_errors.append(_max_abs(torch_hidden, ort_hidden))
                shared_hidden = torch.from_numpy(ort_hidden.copy())

        independent_residual_max = max(independent_residual_errors)
        independent_hidden_max = max(independent_hidden_errors)
        same_state_residual_max = max(same_state_residual_errors)
        same_state_hidden_max = max(same_state_hidden_errors)
        same_state_pass = same_state_residual_max <= STEP_TOLERANCE
        same_state_runtime_pass = same_state_residual_max <= NATIVE_DB_TOLERANCE
        independent_pass = independent_residual_max <= NATIVE_DB_TOLERANCE
        all_same_state_pass = all_same_state_pass and same_state_pass
        all_same_state_runtime_pass = (
            all_same_state_runtime_pass and same_state_runtime_pass
        )
        all_independent_pass = all_independent_pass and independent_pass
        all_finite = all_finite and finite
        results[str(seed)] = {
            "same_state_operator": {
                "residual_max_abs": same_state_residual_max,
                "hidden_max_abs": same_state_hidden_max,
                "tolerance": STEP_TOLERANCE,
                "pass": same_state_pass,
                "runtime_tolerance": NATIVE_DB_TOLERANCE,
                "runtime_pass": same_state_runtime_pass,
            },
            "independent_state": {
                "residual_max_abs": independent_residual_max,
                "hidden_max_abs": independent_hidden_max,
                "tolerance": NATIVE_DB_TOLERANCE,
                "pass": independent_pass,
                "residual_per_frame_max_abs": independent_residual_errors,
                "hidden_per_frame_max_abs": independent_hidden_errors,
            },
            "finite": finite,
        }

    native_receipt = output_dir / "native-parity.json"
    native_reference = None
    if native_receipt.is_file():
        receipt = json.loads(native_receipt.read_text(encoding="utf-8"))
        native_reference = {
            "path": str(native_receipt),
            "residual_db_max_abs": receipt.get("residual_db_max_abs"),
            "hidden_max_abs": receipt.get("hidden_max_abs"),
            "tolerance": NATIVE_DB_TOLERANCE,
        }

    status = (
        "pass"
        if all_same_state_runtime_pass and all_independent_pass and all_finite
        else "fail"
    )
    report = {
        "status": status,
        "metric_definitions": {
            "same_state_operator": (
                "At each frame both runtimes start from the same ORT hidden_out; "
                "this isolates one-step float operator error and uses 1e-5."
            ),
            "independent_state": (
                "Torch and ORT each feed their own hidden_out into their own next "
                "step; this includes accumulated runtime floating-point drift and "
                "uses 1e-3 as a runtime-parity bound, not a model-quality failure."
            ),
        },
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "onnx": str(onnx_path),
        "onnx_sha256": sha256_file(onnx_path),
        "frames": frames,
        "seeds": list(seeds),
        "same_state_tolerance": STEP_TOLERANCE,
        "independent_state_tolerance": NATIVE_DB_TOLERANCE,
        "same_state_strict_all_pass": all_same_state_pass,
        "same_state_runtime_tolerance_all_pass": all_same_state_runtime_pass,
        "results": results,
        "native_reference": native_reference,
        "all_finite": all_finite,
    }
    _write_json(output_dir / "independent-state-parity.json", report)
    if status != "pass":
        raise AssertionError("independent-state parity exceeded declared tolerances")
    return report


def _read_weights_bin(path: Path, model: ResidualStudent) -> dict[str, Any]:
    raw = path.read_bytes()
    if len(raw) < HEADER_STRUCT.size + 8:
        raise AssertionError("weights.bin is too short")
    values = HEADER_STRUCT.unpack(raw[: HEADER_STRUCT.size])
    magic = values[0]
    fields = dict(zip(HEADER_FIELDS, values[1:]))
    expected_bytes = HEADER_STRUCT.size + PARAMETER_COUNT * 4 + 8
    if magic != FORMAT_MAGIC or len(raw) != expected_bytes:
        raise AssertionError("weights.bin magic/size mismatch")
    if fields["parameter_count"] != PARAMETER_COUNT:
        raise AssertionError("weights.bin parameter count mismatch")
    stored_checksum = struct.unpack("<Q", raw[-8:])[0]
    actual_checksum = fnv1a64(raw[:-8])
    if stored_checksum != actual_checksum:
        raise AssertionError("weights.bin FNV1A64 mismatch")
    weights = np.frombuffer(raw[HEADER_STRUCT.size : -8], dtype="<f4")
    state = model.state_dict()
    expected = torch.cat(
        [state[name].detach().cpu().to(torch.float32).reshape(-1) for name in WEIGHT_ORDER]
    ).numpy()
    weight_error = float(np.max(np.abs(weights - expected))) if weights.size else 0.0
    if weight_error != 0.0:
        raise AssertionError("weights.bin order/value mismatch")
    return {
        "magic": magic.decode("ascii"),
        "fields": fields,
        "stored_checksum": stored_checksum,
        "actual_checksum": actual_checksum,
        "weight_count": int(weights.size),
        "weight_roundtrip_max_abs_diff": weight_error,
        "sha256": sha256_file(path),
        "bytes": len(raw),
    }


def run_structure_verification(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(20261116)
    model = ResidualStudent(HIDDEN_SIZE).eval()
    stats = model.stats_dict()
    if stats["parameters"] != PARAMETER_COUNT or stats["macs_per_frame"] != MAC_PER_FRAME:
        raise AssertionError("architecture accounting mismatch")

    checkpoint_path = output_dir / "structure-checkpoint.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "metadata": {
                "steps": 0,
                "training_manifest_sha256": "0" * 64,
                "run_id": "model-structure-verification",
                "trained": False,
                "fixture": True,
                "feature_contract": "dpdf-frame-v1-db",
            },
        },
        checkpoint_path,
    )
    model_json = export_checkpoint(checkpoint_path, output_dir, trained=False)
    weights = _read_weights_bin(output_dir / "weights.bin", model)
    feature_check = _feature_chunk_probe()
    model_check = _model_sequence_probe(model)
    onnx_check = _onnx_state_probe(model, output_dir / "step.onnx")

    if weights["fields"]["trained"] != 0:
        raise AssertionError("structural fixture must have trained=0 in header")
    if model_json["trained"] or not model_json["fixture"]:
        raise AssertionError("structural fixture metadata is incorrect")
    if weights["sha256"] != model_json["sha256"]:
        raise AssertionError("weights.bin SHA differs from model.json")

    result = {
        "status": "pass",
        "trained": False,
        "fixture": True,
        "quality_use": "forbidden",
        "mos_evaluated": False,
        "architecture": stats,
        "export": {
            "checkpoint": str(checkpoint_path),
            "weights": str(output_dir / "weights.bin"),
            "onnx": str(output_dir / "step.onnx"),
            "model_json": str(output_dir / "model.json"),
            "model_json_sha256": sha256_file(output_dir / "model.json"),
            "onnx_sha256": sha256_file(output_dir / "step.onnx"),
        },
        "weights_readback": weights,
        "feature_chunks": feature_check,
        "sequence_step_and_causality": model_check,
        "onnx_state": onnx_check,
    }
    _write_json(output_dir / "verification.json", result)
    return result


def _relative_error(left: np.ndarray, right: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.max(np.abs(left - right) / (np.abs(right) + eps)))


def _load_f32(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4")
    if values.size != int(np.prod(shape)):
        raise AssertionError(f"{path.name} has {values.size} values, expected {np.prod(shape)}")
    return values.reshape(shape)


def run_native_parity(
    native_probe: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    fixture: bool = False,
) -> dict[str, Any]:
    """Run an external native probe against the PyTorch checkpoint.

    Native input/output file protocol is documented in README.md. The executable
    invokes the native ``--band-input`` and ``--gru-input`` modes separately,
    then compares projected bands, four feature groups, and 128 recurrent steps.
    """

    native_probe = Path(native_probe).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not native_probe.is_file():
        raise FileNotFoundError(native_probe)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    native_probe_sha_before = sha256_file(native_probe)
    source_checkpoint = checkpoint_path
    if fixture:
        checkpoint_path = _make_native_fixture_checkpoint(source_checkpoint, output_dir)
    model_json = export_checkpoint(checkpoint_path, output_dir, trained=True)
    model = _load_model_from_checkpoint(checkpoint_path)
    trained_onnx_state = _onnx_state_probe(model, output_dir / "step.onnx")

    inputs_dir = output_dir / "native-inputs"
    native_dir = output_dir / "native-output"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    native_dir.mkdir(parents=True, exist_ok=True)
    frames = 128
    generator = torch.Generator().manual_seed(8801)
    fft_power = torch.pow(10.0, torch.empty(frames, NUM_FFT_BINS).uniform_(-10.0, 6.0, generator=generator))
    fft_noise = torch.pow(10.0, torch.empty(frames, NUM_FFT_BINS).uniform_(-12.0, 2.0, generator=generator))
    raw_features = _fixed_raw_db_features(frames, 1, seed=8802)[0]
    band_input = torch.cat((fft_power, fft_noise), dim=1).contiguous()
    band_input.numpy().astype("<f4", copy=False).tofile(inputs_dir / "band_input.f32")
    raw_features.numpy().astype("<f4", copy=False).tofile(inputs_dir / "features.f32")

    band_power = project_fft_power_to_bands(fft_power)
    band_noise = project_fft_power_to_bands(fft_noise)
    band_features, _ = compose_causal_frame_features_with_state(
        band_power.unsqueeze(0),
        band_noise.unsqueeze(0),
        CausalFeatureState.initial(),
    )
    band_features = band_features[0]
    torch_residual = []
    torch_hidden = []
    state = model.initial_state(1)
    with torch.no_grad():
        for frame in range(frames):
            residual, state = model.step(
                raw_features[frame].view(1, 1, FEATURE_DIM), state
            )
            torch_residual.append(residual[0, 0].numpy())
            torch_hidden.append(state[0].numpy())
    torch_residual_array = np.stack(torch_residual)
    torch_hidden_array = np.stack(torch_hidden)

    band_output = native_dir / "band-output.f32"
    gru_output = native_dir / "gru-output.f32"
    commands = [
        [
            str(native_probe),
            "--band-input",
            str(inputs_dir / "band_input.f32"),
            "--out",
            str(band_output),
        ],
        [
            str(native_probe),
            "--gru-input",
            str(inputs_dir / "features.f32"),
            "--model",
            str(output_dir / "weights.bin"),
            "--out",
            str(gru_output),
        ],
    ]
    completed_runs = [
        subprocess.run(command, capture_output=True, text=True, check=False)
        for command in commands
    ]
    if any(completed.returncode != 0 for completed in completed_runs):
        failure = {
            "status": "fail",
            "commands": commands,
            "returncodes": [completed.returncode for completed in completed_runs],
            "stdout": [completed.stdout for completed in completed_runs],
            "stderr": [completed.stderr for completed in completed_runs],
            "native_probe_sha256_before": native_probe_sha_before,
            "native_probe_sha256_after": sha256_file(native_probe),
        }
        _write_json(output_dir / "native-parity.json", failure)
        raise RuntimeError(
            "native probe failed: "
            + repr([completed.returncode for completed in completed_runs])
        )

    negative_band_input = band_input.clone()
    negative_band_input[1, 0] = -1.0
    negative_band_path = inputs_dir / "band_input_negative.f32"
    negative_band_input.numpy().astype("<f4", copy=False).tofile(negative_band_path)
    negative_band_command = [
        str(native_probe),
        "--band-input",
        str(negative_band_path),
        "--out",
        str(native_dir / "band-negative-output.f32"),
    ]
    negative_band_run = subprocess.run(
        negative_band_command, capture_output=True, text=True, check=False
    )
    negative_band_rejected = negative_band_run.returncode != 0

    untrained_control = None
    untrained_rejected = None
    untrained_run = None
    if fixture:
        untrained_weights = source_checkpoint.parent / "weights.bin"
        untrained_command = [
            str(native_probe),
            "--gru-input",
            str(inputs_dir / "features.f32"),
            "--model",
            str(untrained_weights),
            "--out",
            str(native_dir / "untrained-negative-control.f32"),
        ]
        untrained_run = subprocess.run(
            untrained_command, capture_output=True, text=True, check=False
        )
        untrained_control = untrained_command
        untrained_rejected = untrained_run.returncode != 0

    native_probe_sha_after = sha256_file(native_probe)
    native_probe_sha_consistent = native_probe_sha_before == native_probe_sha_after

    native_band = _load_f32(band_output, (frames, FEATURE_DIM + 2 * NUM_BANDS))
    native_features = native_band[:, :FEATURE_DIM]
    native_band_power = native_band[:, FEATURE_DIM : FEATURE_DIM + NUM_BANDS]
    native_band_noise = native_band[:, FEATURE_DIM + NUM_BANDS :]
    native_gru = _load_f32(gru_output, (frames, NUM_BANDS + HIDDEN_SIZE))
    native_residual = native_gru[:, :NUM_BANDS]
    native_hidden = native_gru[:, NUM_BANDS:]

    band_power_error = _relative_error(native_band_power, band_power.numpy())
    band_noise_error = _relative_error(native_band_noise, band_noise.numpy())
    feature_error = _max_abs(native_features, band_features.numpy())
    residual_error = _max_abs(native_residual, torch_residual_array)
    hidden_error = _max_abs(native_hidden, torch_hidden_array)
    feature_group_errors = [
        _max_abs(
            native_features[:, group * NUM_BANDS : (group + 1) * NUM_BANDS],
            band_features.numpy()[:, group * NUM_BANDS : (group + 1) * NUM_BANDS],
        )
        for group in range(4)
    ]
    residual_per_frame = np.max(np.abs(native_residual - torch_residual_array), axis=1)

    passed = (
        band_power_error <= FEATURE_TOLERANCE
        and band_noise_error <= FEATURE_TOLERANCE
        and feature_error <= FEATURE_TOLERANCE
        and residual_error <= NATIVE_DB_TOLERANCE
        and hidden_error <= NATIVE_HIDDEN_TOLERANCE
        and negative_band_rejected
        and (untrained_rejected is None or untrained_rejected)
        and native_probe_sha_consistent
    )
    result = {
        "status": "pass" if passed else "fail",
        "native_probe": str(native_probe),
        "native_probe_sha256": native_probe_sha_before,
        "native_probe_sha256_before": native_probe_sha_before,
        "native_probe_sha256_after": native_probe_sha_after,
        "native_probe_sha256_consistent": native_probe_sha_consistent,
        "checkpoint": str(checkpoint_path),
        "frames": frames,
        "feature_seed": 8801,
        "raw_feature_range_db": [float(raw_features.min()), float(raw_features.max())],
        "raw_feature_finite": bool(torch.isfinite(raw_features).all().item()),
        "band_feature_range_db": [float(band_features.min()), float(band_features.max())],
        "band_feature_finite": bool(torch.isfinite(band_features).all().item()),
        "band_power_relative_max": band_power_error,
        "band_noise_relative_max": band_noise_error,
        "feature_group_max_abs": feature_group_errors,
        "feature_max_abs": feature_error,
        "feature_tolerance": FEATURE_TOLERANCE,
        "residual_db_max_abs": residual_error,
        "residual_db_tolerance": NATIVE_DB_TOLERANCE,
        "hidden_max_abs": hidden_error,
        "hidden_tolerance": NATIVE_HIDDEN_TOLERANCE,
        "residual_per_frame_max_abs": residual_per_frame.tolist(),
        "weights_sha256": model_json["sha256"],
        "onnx_sha256": model_json["onnx_sha256"],
        "fixture": model_json["fixture"],
        "quality_use": "forbidden" if model_json["fixture"] else "not_evaluated",
        "native_stdout": [completed.stdout for completed in completed_runs],
        "native_stderr": [completed.stderr for completed in completed_runs],
        "trained_onnx_state": trained_onnx_state,
        "negative_raw_band_bin_rejected": negative_band_rejected,
        "negative_raw_band_bin_returncode": negative_band_run.returncode,
        "negative_raw_band_bin_stderr": negative_band_run.stderr,
        "untrained_weights_negative_control": untrained_control,
        "untrained_weights_rejected": untrained_rejected,
        "untrained_weights_returncode": (
            untrained_run.returncode if untrained_run is not None else None
        ),
        "untrained_weights_stderr": (
            untrained_run.stderr if untrained_run is not None else None
        ),
    }
    _write_json(output_dir / "native-parity.json", result)
    if not passed:
        raise AssertionError("native parity exceeded hard tolerances")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--native-probe", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--independent-state-parity", action="store_true")
    parser.add_argument("--independent-state-frames", type=int, default=32)
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="Create an explicit trained-flag=1 test fixture in output-dir",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.independent_state_parity:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for independent-state parity")
        result = run_independent_state_parity(
            args.checkpoint,
            args.output_dir,
            frames=args.independent_state_frames,
        )
        print("INDEPENDENT STATE PARITY PASS")
        for seed in ("8802", "8810"):
            values = result["results"][seed]
            print(
                f"seed={seed} same_state={values['same_state_operator']['residual_max_abs']:.3e} "
                f"independent={values['independent_state']['residual_max_abs']:.3e}"
            )
        print(f"report={args.output_dir / 'independent-state-parity.json'}")
        return
    if args.native_probe or args.checkpoint:
        if not args.native_probe or not args.checkpoint:
            raise SystemExit("--native-probe and --checkpoint must be supplied together")
        result = run_native_parity(
            args.native_probe,
            args.checkpoint,
            args.output_dir,
            fixture=args.fixture,
        )
        print("NATIVE PARITY PASS")
        print(f"feature_max_abs={result['feature_max_abs']:.3e}")
        print(f"residual_db_max_abs={result['residual_db_max_abs']:.3e}")
        print(f"hidden_max_abs={result['hidden_max_abs']:.3e}")
        print(f"report={args.output_dir / 'native-parity.json'}")
        return

    result = run_structure_verification(args.output_dir)
    print("MODEL STRUCTURE PASS")
    print(f"parameters={result['architecture']['parameters']}")
    print(f"macs_per_second_at_100fps={result['architecture']['macs_per_second_at_100fps']}")
    print(f"step_max_abs={result['sequence_step_and_causality']['step_max_abs_diff']:.3e}")
    print(f"onnx_state_max_abs={result['onnx_state']['nonzero_state_max_abs_diff']:.3e}")
    print(f"weights_sha256={result['weights_readback']['sha256']}")
    print(f"report={args.output_dir / 'verification.json'}")


if __name__ == "__main__":
    main()
