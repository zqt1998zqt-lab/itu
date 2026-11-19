#!/usr/bin/env python3
"""ITUNSR02 structure and future native-parity verification."""

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
    from .export import FIELDS, HEADER, MAGIC, WEIGHT_ORDER, export_checkpoint, fnv1a64, sha256
    from .model import (BANDS, CausalFeatureState, FEATURE_V1, FEATURE_V2, FFT_BINS,
                        HIDDEN, ResidualStudent, compose_causal_frame_features,
                        compose_causal_frame_features_with_state, feature_contract,
                        project_fft_to_bands)
else:
    from export import FIELDS, HEADER, MAGIC, WEIGHT_ORDER, export_checkpoint, fnv1a64, sha256
    from model import (BANDS, CausalFeatureState, FEATURE_V1, FEATURE_V2, FFT_BINS,
                       HIDDEN, ResidualStudent, compose_causal_frame_features,
                       compose_causal_frame_features_with_state, feature_contract,
                       project_fft_to_bands)


STRICT = 1e-5
RUNTIME = 1e-3
FEATURE_TOL = 1e-4
BAND_RUNTIME_TOL = 1e-3
BAND_LINEAR_REL_TOL = 1e-4
BAND_GAIN_ABS_TOL = 2e-6
SEEDS = (8802, 8810)


def _json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _max(a: torch.Tensor | np.ndarray, b: torch.Tensor | np.ndarray) -> float:
    aa = a if isinstance(a, torch.Tensor) else torch.from_numpy(a)
    bb = b if isinstance(b, torch.Tensor) else torch.from_numpy(b)
    return float((aa - bb).abs().max())


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)))


def _band_replica(values: torch.Tensor) -> torch.Tensor:
    x = values.detach().cpu().numpy().astype(np.float32, copy=False)
    centers = np.empty(64, dtype=np.float32)
    for b in range(64):
        centers[b] = np.float32(1000.0 * b / 23.0) if b < 24 else np.float32(
            1000.0 + 3000.0 * (b - 23) / 16.0) if b < 40 else np.float32(
            4000.0 + 4000.0 * (b - 39) / 24.0)
    out = np.zeros((*x.shape[:-1], 64), dtype=np.float32)
    norm = np.zeros(64, dtype=np.float32)
    for k in range(257):
        hz = np.float32(k * 16000.0 / 512.0); b = 0
        while b + 1 < 63 and hz > centers[b + 1]: b += 1
        w = np.float32(np.clip((hz - centers[b]) / (centers[b + 1] - centers[b]), 0.0, 1.0))
        out[..., b] += (np.float32(1.0) - w) * x[..., k]
        out[..., b + 1] += w * x[..., k]
        norm[b] += np.float32(1.0) - w; norm[b + 1] += w
    out /= np.maximum(norm, np.float32(1e-12))
    return torch.from_numpy(out)


def _rejects(call) -> bool:
    try:
        call()
    except (ValueError, TypeError):
        return True
    return False


def _raw(input_dim: int, frames: int, batch: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, frames, input_dim, generator=g) * 16.0
    offsets = torch.tensor([-45.0, -20.0, 0.0, 12.0, -15.0])
    x += offsets.repeat_interleave(64)[:input_dim].view(1, 1, -1)
    return x.clamp(-120.0, 60.0)


def _fft_fixture(input_dim: int, frames: int = 11, batch: int = 2):
    g = torch.Generator().manual_seed(900 + input_dim)
    power = torch.rand(batch, frames, FFT_BINS, generator=g) + 1e-6
    noise = torch.rand(batch, frames, FFT_BINS, generator=g) * 0.2 + 1e-7
    gain = torch.rand(batch, frames, FFT_BINS, generator=g) * 0.98 + 0.01
    bp, bn, bg = project_fft_to_bands(power), project_fft_to_bands(noise), project_fft_to_bands(gain)
    return power, noise, gain, bp, bn, bg


def _feature_check(input_dim: int) -> dict[str, Any]:
    power, noise, gain, bp, bn, bg = _fft_fixture(input_dim)
    base = bg if input_dim == 320 else None
    full, state = compose_causal_frame_features_with_state(bp, bn, base, CausalFeatureState.initial())
    expected = feature_contract(input_dim)[0]
    contract = FEATURE_V2 if input_dim == 320 else FEATURE_V1
    assert expected == contract and full.shape[-1] == input_dim and bool(torch.isfinite(full).all())
    if base is not None:
        assert _max(full[..., 256:], 20.0 * torch.log10(base.clamp_min(1e-8))) <= 1e-5
    chunks, chunk_state = [], CausalFeatureState.initial()
    for i in range(11):
        value, chunk_state = compose_causal_frame_features_with_state(
            bp[:, i:i+1], bn[:, i:i+1], None if base is None else base[:, i:i+1], chunk_state)
        chunks.append(value)
    chunk_error = _max(full, torch.cat(chunks, 1))
    bad_power = power.clone(); bad_power[0, 0, 0] = -1.0
    bad_band = bp.clone(); bad_band[0, 0, 0] = float("nan")
    eps_rejected = all(_rejects(lambda e=e: compose_causal_frame_features(bp, bn, base, eps=e))
                       for e in (0.0, -1.0, float("nan"), float("inf")))
    dtype_rejected = _rejects(lambda: compose_causal_frame_features(bp, bn.to(torch.float64), base))
    prev = torch.zeros(bp.shape[0], BANDS, dtype=torch.float64)
    previous_rejected = _rejects(lambda: compose_causal_frame_features(bp, bn, base, prev))
    if torch.cuda.is_available():
        previous_rejected &= _rejects(lambda: compose_causal_frame_features(bp, bn, base, prev.cuda()))
        dtype_rejected &= _rejects(lambda: compose_causal_frame_features(bp, bn.cuda(), base))
    gain_rejected = base is None or all(_rejects(
        lambda v=v: compose_causal_frame_features(bp, bn, torch.full_like(base, v))) for v in (-0.1, 1.1, float("nan")))
    negative_rejected = _rejects(lambda: project_fft_to_bands(bad_power))
    nonfinite_rejected = _rejects(lambda: compose_causal_frame_features(bad_band, bn, base))
    if (chunk_error > FEATURE_TOL or not all((negative_rejected, nonfinite_rejected, eps_rejected,
            dtype_rejected, previous_rejected, gain_rejected))):
        raise AssertionError("feature contract verification failed")
    return {"input_dim": input_dim, "contract": contract, "shape": list(full.shape),
            "finite": True, "chunk1_max_abs": chunk_error, "negative_rejected": negative_rejected,
            "nonfinite_rejected": True,
            "eps_rejected": eps_rejected, "dtype_rejected": dtype_rejected,
            "previous_state_rejected": previous_rejected, "basegain_invalid_rejected": gain_rejected,
            "basegain_log_formula_checked": input_dim == 320}


def _sequence_check(model: ResidualStudent) -> dict[str, Any]:
    x = _raw(model.input_dim, 17, 2, 700 + model.input_dim)
    h = torch.linspace(-0.2, 0.2, 2 * HIDDEN).view(2, HIDDEN)
    with torch.no_grad():
        seq, hs = model(x, h)
        out, state = [], h
        for i in range(17):
            y, state = model.step(x[:, i:i+1], state); out.append(y[:, 0])
        step = torch.stack(out, 1)
        changed = x.clone(); changed[:, 8:] += 1000
        future, _ = model(changed, h)
    err, herr = _max(seq, step), _max(hs, state)
    prefix = _max(seq[:, :8], future[:, :8])
    if max(err, herr, prefix) > STRICT or not bool(torch.isfinite(seq).all()):
        raise AssertionError("sequence/step/causality failed")
    return {"shape": list(seq.shape), "step_max_abs": err, "hidden_max_abs": herr,
            "future_prefix_max_abs": prefix, "finite": True}


def _ort_check(model: ResidualStudent, onnx_path: Path) -> dict[str, Any]:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    initial = torch.linspace(-0.25, 0.25, HIDDEN).view(1, HIDDEN)
    reports = {}
    for seed in SEEDS:
        x = _raw(model.input_dim, 32, 1, seed)
        th, oh = initial.clone(), initial.numpy().copy()
        independent = []
        with torch.no_grad():
            for i in range(32):
                frame = x[:, i:i+1]
                ty, th = model.step(frame, th)
                oy, oh = session.run(["delta_db", "hidden_out"],
                                     {"features": frame.numpy(), "hidden_in": oh})
                independent.append(_max(ty, oy))
        shared = initial.clone(); same = []
        with torch.no_grad():
            for i in range(32):
                frame = x[:, i:i+1]
                ty, th = model.step(frame, shared)
                oy, oh2 = session.run(["delta_db", "hidden_out"],
                                      {"features": frame.numpy(), "hidden_in": shared.numpy()})
                same.append(_max(ty, oy)); shared = torch.from_numpy(oh2.copy())
        reports[str(seed)] = {"independent_max_abs": max(independent),
                              "independent_pass_1e-3": max(independent) <= RUNTIME,
                              "same_state_max_abs": max(same),
                              "same_state_strict_pass_1e-5": max(same) <= STRICT}
    silence = torch.zeros(1, 1, model.input_dim)
    if model.input_dim == 320:
        silence[..., -BANDS:] = 20.0 * np.log10(1e-8)
    th, oh = initial.clone(), initial.numpy().copy(); finite = True
    with torch.no_grad():
        for _ in range(32):
            ty, th = model.step(silence, th)
            oy, oh = session.run(["delta_db", "hidden_out"],
                                 {"features": silence.numpy(), "hidden_in": oh})
            finite &= bool(np.isfinite(oy).all() and np.isfinite(oh).all())
    all_independent = all(v["independent_pass_1e-3"] for v in reports.values())
    all_same_state = all(v["same_state_strict_pass_1e-5"] for v in reports.values())
    if not finite or not all_independent:
        raise AssertionError("ORT independent-state parity failed")
    return {"seeds": reports, "silence_32_finite": finite,
            "state_feedback": "each_runtime_own_hidden_out",
            "all_independent_state_pass_1e-3": all_independent,
            "all_same_state_strict_pass_1e-5": all_same_state,
            "strict_tolerance": STRICT, "runtime_tolerance": RUNTIME}


def _read_bin(path: Path, model: ResidualStudent) -> dict[str, Any]:
    raw = path.read_bytes(); values = HEADER.unpack(raw[:64]); fields = dict(zip(FIELDS, values[1:]))
    contract, version, count = feature_contract(model.input_dim)
    if values[0] != MAGIC or fields["version"] != 2 or fields["input_dim"] != model.input_dim:
        raise AssertionError("ITUNSR02 header mismatch")
    if fields["feature_version"] != version or fields["parameter_count"] != count:
        raise AssertionError("ITUNSR02 feature/count mismatch")
    if len(raw) != 64 + 4 * count + 8 or struct.unpack("<Q", raw[-8:])[0] != fnv1a64(raw[:-8]):
        raise AssertionError("ITUNSR02 size/checksum mismatch")
    got = np.frombuffer(raw[64:-8], dtype="<f4")
    state = model.state_dict(); expected = torch.cat([state[k].flatten() for k in WEIGHT_ORDER]).numpy()
    if not np.array_equal(got, expected):
        raise AssertionError("ITUNSR02 weight order mismatch")
    return {"magic": MAGIC.decode(), "fields": fields, "sha256": sha256(path), "bytes": len(raw)}


def _export_boundary(model: ResidualStudent, root: Path) -> dict[str, Any]:
    bad = root / "bad-export"; bad.mkdir(exist_ok=True)
    base = {"input_dim": model.input_dim, "steps": 0, "feature_contract": model.feature_contract,
            "training_manifest_sha256": "0" * 64, "run_id": "bad-export", "trained": False,
            "fixture": True, "gain_mode": "residual_db", "floor_db": -9.0, "base_floor_db": -9.0}
    rejected = {}
    for name, patch in {
        "steps_negative": {"steps": -1}, "floor_nan": {"floor_db": float("nan")},
        "floor_inf": {"floor_db": float("inf")}, "floor_positive": {"floor_db": 1.0},
        "floor_too_deep": {"floor_db": -41.0}, "basefloor_nan": {"base_floor_db": float("nan")},
        "basefloor_positive": {"base_floor_db": 1.0},
    }.items():
        ckpt = bad / "bad.pt"
        torch.save({"state_dict": model.state_dict(), "metadata": {**base, **patch}}, ckpt)
        rejected[name] = _rejects(lambda c=ckpt: export_checkpoint(c, bad / "out", trained=False))
    rejected["unknown_input_dim"] = _rejects(lambda: ResidualStudent(288))
    if not all(rejected.values()):
        raise AssertionError("export boundary validation failed")
    return rejected


def run_structure(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True); torch.set_num_threads(1)
    summary = {"status": "pass", "trained": False, "fixture": True, "models": {}}
    for input_dim in (256, 320):
        root = output / f"model-{input_dim}"; root.mkdir(exist_ok=True)
        torch.manual_seed(20261115 + input_dim)
        model = ResidualStudent(input_dim).eval()
        contract, _, _ = feature_contract(input_dim)
        ckpt = root / "structure-checkpoint.pt"
        torch.save({"state_dict": model.state_dict(), "metadata": {
            "input_dim": input_dim, "steps": 0, "feature_contract": contract,
            "training_manifest_sha256": "0" * 64, "run_id": f"structure-{input_dim}",
            "trained": False, "fixture": True, "gain_mode": "residual_db",
            "floor_db": -9.0, "base_floor_db": -9.0}}, ckpt)
        manifest = export_checkpoint(ckpt, root, trained=False)
        result = {"architecture": model.stats_dict(), "features": _feature_check(input_dim),
                  "sequence": _sequence_check(model), "weights": _read_bin(root / "weights.bin", model),
                  "ort": _ort_check(model, root / "step.onnx"),
                  "export_boundaries": _export_boundary(model, root),
                  "model_json": manifest}
        _json(root / "verification.json", {"status": "pass", **result})
        summary["models"][str(input_dim)] = result
    _json(output / "verification.json", summary)
    return summary


def _native_fixture(source: Path, output: Path) -> Path:
    ckpt = torch.load(source, map_location="cpu", weights_only=True); meta = ckpt.get("metadata", {})
    dim = int(meta["input_dim"]); contract = feature_contract(dim)[0]
    path = output / "native-fixture-checkpoint.pt"
    torch.save({"state_dict": ckpt["state_dict"], "metadata": {
        **meta, "steps": 1, "trained": True, "fixture": True,
        "training_manifest_sha256": hashlib.sha256(sha256(source).encode()).hexdigest(),
        "run_id": str(meta.get("run_id", "fixture")) + "-native", "feature_contract": contract,
        "gain_mode": "residual_db"}}, path)
    return path


def run_native(native: Path, checkpoint: Path, output: Path, fixture: bool = False) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True); checkpoint = _native_fixture(checkpoint, output) if fixture else checkpoint
    native_sha_before = sha256(native)
    model_json = export_checkpoint(checkpoint, output, trained=True); model = ResidualStudent(model_json["input_dim"]).eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"])
    ort_report = _ort_check(model, output / "step.onnx")
    dim = model.input_dim; frames = 128; g = torch.Generator().manual_seed(8802)
    power = torch.pow(10, torch.empty(frames, FFT_BINS).uniform_(-10, 6, generator=g))
    noise = torch.pow(10, torch.empty(frames, FFT_BINS).uniform_(-12, 2, generator=g))
    gain = torch.rand(frames, FFT_BINS, generator=g) * 0.98 + 0.01
    raw = _raw(dim, frames, 1, 8802)[0]
    (output / "features.f32").write_bytes(raw.numpy().astype("<f4").tobytes())
    bp, bn, bg = project_fft_to_bands(power), project_fft_to_bands(noise), project_fft_to_bands(gain)
    band = torch.cat((power, noise, gain), 1)
    (output / "band-input.f32").write_bytes(band.numpy().astype("<f4").tobytes())
    gru_cmd = [str(native), "--gru-input", str(output / "features.f32"), "--model", str(output / "weights.bin"), "--out", str(output / "gru-output.f32")]
    run1 = subprocess.run(gru_cmd, capture_output=True, text=True)
    if run1.returncode: raise RuntimeError(run1.stderr)
    gru = np.fromfile(output / "gru-output.f32", "<f4").reshape(frames, 128)
    state = model.initial_state(1); ref = []; hidden_ref = []
    with torch.no_grad():
        for i in range(frames):
            y, state = model.step(raw[i].view(1, 1, -1), state)
            ref.append(y[0, 0].numpy()); hidden_ref.append(state[0].numpy())
    ref = np.stack(ref); hidden_ref = np.stack(hidden_ref)
    delta_err = _max(gru[:, :64], ref); hidden_err = _max(gru[:, 64:], hidden_ref)
    final_hidden_err = _max(gru[-1:, 64:], hidden_ref[-1:])
    delta_abs = np.abs(gru[:, :64] - ref); delta_index = tuple(int(x) for x in np.unravel_index(delta_abs.argmax(), delta_abs.shape))
    hidden_abs = np.abs(gru[:, 64:] - hidden_ref); hidden_index = tuple(int(x) for x in np.unravel_index(hidden_abs.argmax(), hidden_abs.shape))
    native_sha_after = sha256(native)
    result = {"status": "pass", "native_sha256": sha256(native), "input_dim": dim,
              "native_sha256_before": native_sha_before, "native_sha256_after": native_sha_after,
              "native_sha256_consistent": native_sha_before == native_sha_after,
              "frames": frames, "delta_max_abs": delta_err, "hidden_trajectory_max_abs": hidden_err,
              "hidden_final_max_abs": final_hidden_err,
              "first_delta_diff": {"index": delta_index, "native": float(gru[delta_index]), "python": float(ref[delta_index])},
              "first_hidden_diff": {"index": hidden_index, "native": float(gru[hidden_index[0], 64 + hidden_index[1]]), "python": float(hidden_ref[hidden_index])},
              "ort": ort_report}
    if dim == 320:
        band_cmd = [str(native), "--band-input", str(output / "band-input.f32"), "--out", str(output / "band-output.f32")]
        run2 = subprocess.run(band_cmd, capture_output=True, text=True)
        if run2.returncode: raise RuntimeError(run2.stderr)
        bandout = np.fromfile(output / "band-output.f32", "<f4").reshape(frames, 512)
        rbp, rbn, rbg = _band_replica(power), _band_replica(noise), _band_replica(gain)
        feat = compose_causal_frame_features(rbp.unsqueeze(0), rbn.unsqueeze(0), rbg.unsqueeze(0))[0].numpy()
        result["band_features_max_abs"] = _max(bandout[:, :320], feat)
        result["band_power_relative_max"] = _rel(bandout[:, 320:384], rbp.numpy())
        result["band_noise_relative_max"] = _rel(bandout[:, 384:448], rbn.numpy())
        result["band_gain_max_abs"] = _max(bandout[:, 448:], rbg.numpy())
        result["band_features_strict_1e-4_pass"] = result["band_features_max_abs"] <= FEATURE_TOL
        result["band_features_runtime_1e-3_pass"] = result["band_features_max_abs"] <= BAND_RUNTIME_TOL
        result["band_power_relative_pass_1e-4"] = result["band_power_relative_max"] <= BAND_LINEAR_REL_TOL
        result["band_noise_relative_pass_1e-4"] = result["band_noise_relative_max"] <= BAND_LINEAR_REL_TOL
        result["band_gain_abs_pass_2e-6"] = result["band_gain_max_abs"] <= BAND_GAIN_ABS_TOL
    band_ok = dim != 320 or (
        result["band_features_runtime_1e-3_pass"] and
        result["band_power_relative_pass_1e-4"] and
        result["band_noise_relative_pass_1e-4"] and
        result["band_gain_abs_pass_2e-6"])
    if (result["delta_max_abs"] > RUNTIME or result["hidden_trajectory_max_abs"] > RUNTIME or
            result["hidden_final_max_abs"] > RUNTIME or not result["native_sha256_consistent"] or
            result.get("band_feature_max_abs", 0) > FEATURE_TOL or
            not band_ok or
            not ort_report["all_independent_state_pass_1e-3"]):
        result["status"] = "fail"
    _json(output / "native-parity.json", result)
    if result["status"] != "pass": raise AssertionError("native parity failed")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--native-probe", type=Path); p.add_argument("--checkpoint", type=Path); p.add_argument("--fixture", action="store_true")
    a = p.parse_args()
    if a.native_probe or a.checkpoint:
        if not a.native_probe or not a.checkpoint: raise SystemExit("--native-probe and --checkpoint required together")
        r = run_native(a.native_probe, a.checkpoint, a.output_dir, a.fixture)
        print("NATIVE PARITY PASS"); print(f"delta_max_abs={r['delta_max_abs']:.3e}"); print(f"hidden_max_abs={r['hidden_final_max_abs']:.3e}")
    else:
        r = run_structure(a.output_dir); print("MODEL STRUCTURE PASS")
        for dim in ("256", "320"):
            x = r["models"][dim]; print(f"dim={dim} params={x['architecture']['parameters']} mmac={x['architecture']['mmacs_per_second_at_100fps']}")
        print(f"report={a.output_dir / 'verification.json'}")


if __name__ == "__main__":
    main()
