"""Versioned ITUNSR02 export for the C64 v1/v2 residual student."""

from __future__ import annotations

import hashlib
import json
import math
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
        FEATURE_V1, FFT_SIZE, HIDDEN, HOP, RAW_FEATURE_SCALING, ResidualStudent,
        SAMPLE_RATE, WINDOW, feature_contract, mac_per_frame,
    )
else:
    from model import (
        FEATURE_V1, FFT_SIZE, HIDDEN, HOP, RAW_FEATURE_SCALING, ResidualStudent,
        SAMPLE_RATE, WINDOW, feature_contract, mac_per_frame,
    )


MAGIC = b"ITUNSR02"
FORMAT_VERSION = 2
MAX_ABS_WEIGHT = 100.0
HEADER = struct.Struct("<8s14I")
WEIGHT_ORDER = (
    "projection.weight", "projection.bias", "gru.weight_ih_l0",
    "gru.weight_hh_l0", "gru.bias_ih_l0", "gru.bias_hh_l0",
    "head.weight", "head.bias",
)
FIELDS = (
    "version", "sample_rate", "bands", "input_dim", "projection", "hidden",
    "head", "parameter_count", "trained", "feature_version", "fft", "hop",
    "window", "reserved",
)
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class Step(nn.Module):
    def __init__(self, model: ResidualStudent):
        super().__init__()
        self.model = model

    def forward(self, x, h):
        return self.model.step(x, h)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fnv1a64(data: bytes | bytearray) -> int:
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def _get(meta: Mapping[str, Any], checkpoint: Mapping[str, Any], *keys: str, default=None):
    for key in keys:
        if key in meta:
            return meta[key]
    for key in keys:
        if key in checkpoint:
            return checkpoint[key]
    return default


def _sha(value: Any, name: str, *, zero_ok: bool) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be SHA-256 hex")
    value = value.removeprefix("sha256:").lower()
    if not SHA_RE.fullmatch(value) or (not zero_ok and value == "0" * 64):
        raise ValueError(f"invalid {name}")
    return value


def _floor(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number) or not -40.0 <= number <= 0.0:
        raise ValueError(f"{name} must be finite in [-40,0]")
    return number


def _weights(model: ResidualStudent) -> Tensor:
    state = model.state_dict()
    if set(state) != set(WEIGHT_ORDER):
        raise ValueError("state_dict keys do not match ITUNSR02 weight order")
    return torch.cat([state[k].detach().cpu().to(torch.float32).flatten() for k in WEIGHT_ORDER])


def _write_binary(model: ResidualStudent, path: Path, *, trained: bool) -> tuple[str, dict[str, int]]:
    weights = _weights(model)
    if not bool(torch.isfinite(weights).all().item()) or float(weights.abs().max()) > MAX_ABS_WEIGHT:
        raise ValueError("weights are non-finite or exceed |100|")
    contract, version, count = feature_contract(model.input_dim)
    if weights.numel() != count:
        raise ValueError(f"parameter count {weights.numel()} != {count}")
    values = (FORMAT_VERSION, SAMPLE_RATE, 64, model.input_dim, 64, HIDDEN, 64,
              count, int(trained), version, FFT_SIZE, HOP, WINDOW, 0)
    payload = bytearray(HEADER.pack(MAGIC, *values))
    payload.extend(np.asarray(weights.numpy(), dtype="<f4").tobytes())
    checksum = fnv1a64(payload)
    payload.extend(struct.pack("<Q", checksum))
    path.write_bytes(payload)
    header = dict(zip(FIELDS, values))
    header["checksum"] = checksum
    header["bytes"] = len(payload)
    return sha256(path), header


def _write_onnx(model: ResidualStudent, path: Path, meta: dict[str, Any]) -> None:
    wrapper = Step(model.eval())
    x = torch.zeros(1, 1, model.input_dim)
    h = model.initial_state(1)
    torch.onnx.export(
        wrapper, (x, h), str(path), opset_version=17, dynamo=False,
        input_names=["features", "hidden_in"],
        output_names=["delta_db", "hidden_out"],
        dynamic_axes={"features": {0: "batch"}, "hidden_in": {0: "batch"},
                      "delta_db": {0: "batch"}, "hidden_out": {0: "batch"}},
    )
    proto = onnx.load(str(path))
    del proto.metadata_props[:]
    for key, value in meta.items():
        item = proto.metadata_props.add()
        item.key = key
        item.value = str(value).lower() if isinstance(value, bool) else str(value)
    onnx.checker.check_model(proto)
    onnx.save(proto, str(path))
    onnx.checker.check_model(str(path))


def export_checkpoint(
    checkpoint_path: str | Path, output_dir: str | Path, trained: bool = True
) -> dict[str, Any]:
    """Export ITUNSR02 ``weights.bin``, ``step.onnx`` and ``model.json``."""

    checkpoint_path, output_dir = Path(checkpoint_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, Mapping) or "state_dict" not in ckpt or not isinstance(ckpt.get("metadata"), Mapping):
        raise ValueError("checkpoint requires state_dict and metadata mapping")
    meta = ckpt["metadata"]
    input_dim = int(_get(meta, ckpt, "input_dim"))
    contract, feature_version, count = feature_contract(input_dim)
    steps = int(_get(meta, ckpt, "steps", "training_steps", default=-1))
    if steps < 0:
        raise ValueError("metadata.steps cannot be negative")
    if trained and steps <= 0:
        raise ValueError("trained export requires metadata.steps > 0")
    manifest = _sha(
        _get(meta, ckpt, "training_manifest_sha256", "dataset_sha256", "manifest_sha256",
             default="0" * 64), "training_manifest_sha256", zero_ok=not trained)
    runid = _get(meta, ckpt, "run_id", "runid", default="structure-fixture" if not trained else None)
    if not isinstance(runid, str) or not runid.strip():
        raise ValueError("metadata.run_id is required")
    if bool(meta.get("trained", trained)) != bool(trained):
        raise ValueError("checkpoint trained flag conflicts with export request")
    fixture = bool(meta.get("fixture", not trained))
    if trained:
        if meta.get("trained") is not True or not isinstance(meta.get("fixture"), bool):
            raise ValueError("trained checkpoint requires metadata.trained=true and explicit boolean fixture")
        for key in ("feature_contract", "gain_mode"):
            if key not in meta:
                raise ValueError(f"trained checkpoint metadata requires {key}")
        if not any(k in meta for k in ("floor_db", "absolute_floor_db")):
            raise ValueError("trained checkpoint metadata requires floor_db")
        if not any(k in meta for k in ("base_floor_db", "base_frontend_floor_db")):
            raise ValueError("trained checkpoint metadata requires base_floor_db")
    declared_contract = str(meta.get("feature_contract", contract))
    if declared_contract != contract:
        raise ValueError(f"feature contract {declared_contract} != {contract}")
    gain_mode = str(meta.get("gain_mode", "residual_db"))
    if gain_mode != "residual_db":
        raise ValueError("only gain_mode=residual_db is supported")
    floor_db = _floor(_get(meta, ckpt, "floor_db", "absolute_floor_db", default=-9.0), "floor_db")
    base_floor_db = _floor(_get(meta, ckpt, "base_floor_db", "base_frontend_floor_db", default=-9.0), "base_floor_db")
    model_hash = sha256(Path(__file__).with_name("model.py"))
    declared_model = _get(meta, ckpt, "model_sha256", "model_hash")
    if declared_model is not None and str(declared_model) != model_hash:
        raise ValueError("checkpoint model hash differs from deployed model.py")

    model = ResidualStudent(input_dim, HIDDEN).eval()
    try:
        model.load_state_dict(ckpt["state_dict"], strict=True)
    except RuntimeError as exc:
        raise ValueError(f"state_dict shape/key mismatch: {exc}") from exc
    if model.stats_dict()["parameters"] != count:
        raise ValueError("parameter accounting mismatch")

    weights_path, onnx_path = output_dir / "weights.bin", output_dir / "step.onnx"
    weights_sha, header = _write_binary(model, weights_path, trained=trained)
    onnx_meta = {
        "trained": trained, "fixture": fixture, "input_dim": input_dim,
        "feature_contract": contract, "feature_version": feature_version,
        "gain_mode": gain_mode, "steps": steps, "runid": runid,
        "floor_db": floor_db, "base_floor_db": base_floor_db,
        "modelhash": model_hash, "datasetsha": manifest,
        "runtimeheadunits": 64, "residual_limit_db": 12.0,
        "state_policy": "explicit_no_silence_reset",
    }
    _write_onnx(model, onnx_path, onnx_meta)
    onnx_sha = sha256(onnx_path)
    model_json = {
        "format": MAGIC.decode(), "version": FORMAT_VERSION,
        "model": "ResidualStudent", "modelhash": model_hash, "model_sha256": model_hash,
        "featurecontract": contract, "feature_contract": contract,
        "feature_version": feature_version, "inputdim": input_dim, "input_dim": input_dim,
        "gainmode": gain_mode, "gain_mode": gain_mode, "raw_feature_scaling": RAW_FEATURE_SCALING,
        "floor_db": floor_db, "base_floor_db": base_floor_db,
        "datasetsha": manifest, "dataset_sha256": manifest,
        "training_manifest_sha256": manifest, "runid": runid, "run_id": runid,
        "trained": trained, "fixture": fixture, "quality_use": "forbidden" if fixture or not trained else "requires_quality_gates",
        "steps": steps, "headunits": 64, "head_units": 64,
        "runtimeheadunits": 64, "residual_limit_db": 12.0, "residuallimit_db": 12.0,
        "state_policy": "explicit_no_silence_reset",
        "parameter_count": count, "mac_per_frame": mac_per_frame(input_dim),
        "mac_per_second_at_100fps": mac_per_frame(input_dim) * 100,
        "sha256": weights_sha, "weights_sha256": weights_sha, "onnx_sha256": onnx_sha,
        "checkpoint_sha256": sha256(checkpoint_path), "weight_order": list(WEIGHT_ORDER),
        "header": header,
    }
    (output_dir / "model.json").write_text(json.dumps(model_json, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return model_json


__all__ = ("FIELDS", "FORMAT_VERSION", "HEADER", "MAGIC", "WEIGHT_ORDER", "export_checkpoint", "fnv1a64", "sha256")
