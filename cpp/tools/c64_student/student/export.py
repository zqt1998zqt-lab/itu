"""Versioned deployment export for :class:`ResidualStudent`."""

from __future__ import annotations

import hashlib
import json
import re
import struct
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import onnx
import torch
from torch import nn

if __package__:
    from .model import (
        FEATURE_CONTRACT_ID,
        FEATURE_CONTRACT_VERSION,
        FEATURE_DIM,
        FFT_SIZE,
        HEAD_UNITS,
        HIDDEN_SIZE,
        HOP_SAMPLES,
        MAC_PER_FRAME,
        MAC_PER_SECOND_AT_100FPS,
        NUM_BANDS,
        PARAMETER_COUNT,
        RAW_FEATURE_SCALING,
        ResidualStudent,
        SAMPLE_RATE_HZ,
        WINDOW_SAMPLES,
    )
else:
    from model import (
        FEATURE_CONTRACT_ID,
        FEATURE_CONTRACT_VERSION,
        FEATURE_DIM,
        FFT_SIZE,
        HEAD_UNITS,
        HIDDEN_SIZE,
        HOP_SAMPLES,
        MAC_PER_FRAME,
        MAC_PER_SECOND_AT_100FPS,
        NUM_BANDS,
        PARAMETER_COUNT,
        RAW_FEATURE_SCALING,
        ResidualStudent,
        SAMPLE_RATE_HZ,
        WINDOW_SAMPLES,
    )


FORMAT_MAGIC = b"ITUNSR01"
FORMAT_VERSION = 1
MAX_ABS_WEIGHT = 100.0
HEADER_STRUCT = struct.Struct("<8s14I")
CHECKSUM_STRUCT = struct.Struct("<Q")
WEIGHT_ORDER = (
    "projection.weight",
    "projection.bias",
    "gru.weight_ih_l0",
    "gru.weight_hh_l0",
    "gru.bias_ih_l0",
    "gru.bias_hh_l0",
    "head.weight",
    "head.bias",
)
HEADER_FIELDS = (
    "format_version",
    "sample_rate_hz",
    "bands",
    "feature_dim",
    "projection_units",
    "hidden_size",
    "head_units",
    "parameter_count",
    "trained",
    "feature_version",
    "fft_size",
    "hop_samples",
    "window_samples",
    "reserved",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class _StepExportWrapper(nn.Module):
    def __init__(self, student: ResidualStudent) -> None:
        super().__init__()
        self.student = student

    def forward(self, features: torch.Tensor, hidden_in: torch.Tensor):
        return self.student.step(features, hidden_in)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fnv1a64(data: bytes | bytearray) -> int:
    value = 14695981039346656037
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def _normalise_sha256(value: Any, *, name: str, allow_zero: bool) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a SHA-256 hex string")
    normalized = value.removeprefix("sha256:").lower()
    if not SHA256_RE.fullmatch(normalized):
        raise ValueError(f"{name} must be 64 lowercase/uppercase hex characters")
    if not allow_zero and normalized == "0" * 64:
        raise ValueError(f"{name} cannot be the zero placeholder for a trained export")
    return normalized


def _metadata_value(
    metadata: Mapping[str, Any], checkpoint: Mapping[str, Any], *names: str
) -> Any:
    for name in names:
        if name in metadata:
            return metadata[name]
    for name in names:
        if name in checkpoint:
            return checkpoint[name]
    return None


def _ordered_weights(model: ResidualStudent) -> torch.Tensor:
    state = model.state_dict()
    missing = [name for name in WEIGHT_ORDER if name not in state]
    if missing:
        raise ValueError(f"missing deployment weights: {missing}")
    return torch.cat([state[name].detach().cpu().to(torch.float32).reshape(-1) for name in WEIGHT_ORDER])


def _write_weights(
    model: ResidualStudent,
    path: Path,
    *,
    trained: bool,
) -> tuple[str, dict[str, int]]:
    weights = _ordered_weights(model)
    if weights.numel() != PARAMETER_COUNT:
        raise ValueError(
            f"deployment parameter count mismatch: {weights.numel()} != {PARAMETER_COUNT}"
        )
    if not bool(torch.isfinite(weights).all().item()):
        raise ValueError("deployment weights contain non-finite values")
    max_abs = float(weights.abs().max().item()) if weights.numel() else 0.0
    if max_abs > MAX_ABS_WEIGHT:
        raise ValueError(
            f"deployment weight exceeds |{MAX_ABS_WEIGHT}| boundary: {max_abs}"
        )

    header_values = (
        FORMAT_VERSION,
        SAMPLE_RATE_HZ,
        NUM_BANDS,
        FEATURE_DIM,
        64,
        HIDDEN_SIZE,
        HEAD_UNITS,
        PARAMETER_COUNT,
        1 if trained else 0,
        FEATURE_CONTRACT_ID,
        FFT_SIZE,
        HOP_SAMPLES,
        WINDOW_SAMPLES,
        0,
    )
    header = HEADER_STRUCT.pack(FORMAT_MAGIC, *header_values)
    weight_bytes = np.asarray(weights.numpy(), dtype="<f4", order="C").tobytes()
    payload = bytearray(header)
    payload.extend(weight_bytes)
    checksum = fnv1a64(payload)
    payload.extend(CHECKSUM_STRUCT.pack(checksum))
    path.write_bytes(payload)

    header_map = dict(zip(HEADER_FIELDS, header_values))
    header_map["checksum"] = checksum
    header_map["file_bytes"] = len(payload)
    return sha256_file(path), header_map


def _export_onnx(
    model: ResidualStudent,
    path: Path,
    *,
    trained: bool,
    fixture: bool,
    steps: int,
    run_id: str,
) -> None:
    wrapper = _StepExportWrapper(model).eval()
    sample_features = torch.zeros(1, 1, FEATURE_DIM)
    sample_hidden = model.initial_state(1)
    torch.onnx.export(
        wrapper,
        (sample_features, sample_hidden),
        str(path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
        input_names=["features", "hidden_in"],
        output_names=["residual_db", "hidden_out"],
        dynamic_axes={
            "features": {0: "batch"},
            "hidden_in": {0: "batch"},
            "residual_db": {0: "batch"},
            "hidden_out": {0: "batch"},
        },
    )
    model_proto = onnx.load(str(path))
    metadata = {
        "trained": str(bool(trained)).lower(),
        "fixture": str(bool(fixture)).lower(),
        "quality_use": "forbidden" if fixture or not trained else "requires_quality_gates",
        "model": "ResidualStudent",
        "steps": str(steps),
        "runid": run_id,
        "feature_contract": FEATURE_CONTRACT_VERSION,
        "feature_contract_id": str(FEATURE_CONTRACT_ID),
        "raw_feature_scaling": RAW_FEATURE_SCALING,
        "bands": str(NUM_BANDS),
        "feature_dim": str(FEATURE_DIM),
        "hidden_size": str(HIDDEN_SIZE),
        "head_units": str(HEAD_UNITS),
        "fft_size": str(FFT_SIZE),
        "hop_samples": str(HOP_SAMPLES),
        "window_samples": str(WINDOW_SAMPLES),
        "lookahead_frames": "0",
        "state_policy": "explicit_no_silence_reset",
    }
    del model_proto.metadata_props[:]
    for key, value in metadata.items():
        entry = model_proto.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.checker.check_model(model_proto)
    onnx.save(model_proto, str(path))
    onnx.checker.check_model(str(path))


def export_checkpoint(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    trained: bool = True,
) -> dict[str, Any]:
    """Export ``weights.bin``, ``step.onnx`` and ``model.json``.

    Expected checkpoint schema::

        {
          "state_dict": { ... ResidualStudent.state_dict() ... },
          "metadata": {
            "steps": int,
            "training_manifest_sha256": 64-char hex,
            "run_id": str,
            "trained": bool,
            "fixture": bool
          }
        }

    ``trained=True`` requires positive steps, a non-zero manifest SHA and a run
    id. Structural fixtures set ``trained=False``; their binary is intentionally
    rejected by a deployment loader but remains usable for parity tests.
    """

    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping) or "state_dict" not in checkpoint:
        raise ValueError("checkpoint must contain state_dict")
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint must contain metadata mapping")

    steps_raw = _metadata_value(metadata, checkpoint, "steps", "training_steps")
    if steps_raw is None:
        steps = 0 if not trained else -1
    else:
        steps = int(steps_raw)
    if trained and steps <= 0:
        raise ValueError("trained export requires metadata steps > 0")
    if steps < 0:
        raise ValueError("metadata steps cannot be negative")

    manifest_raw = _metadata_value(
        metadata,
        checkpoint,
        "training_manifest_sha256",
        "manifest_sha256",
    )
    manifest_sha = _normalise_sha256(
        manifest_raw if manifest_raw is not None else ("0" * 64),
        name="training_manifest_sha256",
        allow_zero=not trained,
    )
    run_id = _metadata_value(metadata, checkpoint, "run_id", "runid")
    if run_id is None:
        run_id = "structural-verification" if not trained else None
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("metadata run_id must be a non-empty string")
    run_id = run_id.strip()

    metadata_trained = bool(metadata.get("trained", trained))
    if metadata_trained != bool(trained):
        raise ValueError("requested trained flag conflicts with checkpoint metadata")
    fixture = bool(metadata.get("fixture", not trained))
    feature_contract = str(metadata.get("feature_contract", FEATURE_CONTRACT_VERSION))
    if feature_contract != FEATURE_CONTRACT_VERSION:
        raise ValueError(
            f"feature contract mismatch: {feature_contract} != {FEATURE_CONTRACT_VERSION}"
        )

    model = ResidualStudent(HIDDEN_SIZE).eval()
    try:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    except RuntimeError as exc:
        raise ValueError(f"checkpoint state_dict does not match ResidualStudent: {exc}") from exc
    stats = model.architecture_stats()
    if stats.parameters != PARAMETER_COUNT or stats.macs_per_frame != MAC_PER_FRAME:
        raise ValueError("model architecture accounting drifted from deployment contract")

    weights_path = output_dir / "weights.bin"
    onnx_path = output_dir / "step.onnx"
    weights_sha, header = _write_weights(model, weights_path, trained=trained)
    _export_onnx(
        model,
        onnx_path,
        trained=trained,
        fixture=fixture,
        steps=steps,
        run_id=run_id,
    )
    onnx_sha = sha256_file(onnx_path)

    model_json: dict[str, Any] = {
        "format": FORMAT_MAGIC.decode("ascii"),
        "format_version": FORMAT_VERSION,
        "model": "ResidualStudent",
        "featurecontract": FEATURE_CONTRACT_VERSION,
        "feature_contract": FEATURE_CONTRACT_VERSION,
        "feature_contract_id": FEATURE_CONTRACT_ID,
        "trained": bool(trained),
        "fixture": fixture,
        "quality_use": "forbidden" if fixture or not trained else "requires_quality_gates",
        "steps": steps,
        "headunits": HEAD_UNITS,
        "head_units": HEAD_UNITS,
        "training_manifest_sha256": manifest_sha,
        "runid": run_id,
        "run_id": run_id,
        "sha256": weights_sha,
        "weights_sha256": weights_sha,
        "onnx_sha256": onnx_sha,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "weights_file": weights_path.name,
        "onnx_file": onnx_path.name,
        "parameter_count": stats.parameters,
        "mac_per_frame": stats.macs_per_frame,
        "mac_per_second_at_100fps": stats.macs_per_second_at_100fps,
        "bands": NUM_BANDS,
        "feature_dim": FEATURE_DIM,
        "hidden_size": HIDDEN_SIZE,
        "fft_size": FFT_SIZE,
        "hop_samples": HOP_SAMPLES,
        "window_samples": WINDOW_SAMPLES,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "raw_feature_scaling": RAW_FEATURE_SCALING,
        "weight_order": list(WEIGHT_ORDER),
        "header": header,
    }
    (output_dir / "model.json").write_text(
        json.dumps(model_json, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return model_json


__all__ = (
    "FORMAT_MAGIC",
    "FORMAT_VERSION",
    "HEADER_FIELDS",
    "HEADER_STRUCT",
    "MAX_ABS_WEIGHT",
    "WEIGHT_ORDER",
    "export_checkpoint",
    "fnv1a64",
    "sha256_file",
)
