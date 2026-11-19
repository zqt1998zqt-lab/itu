#!/usr/bin/env python3
"""C64 NS v2 trainer: isolated old/new supervised objectives.

Formal runs use only manifest ``train`` and ``dev`` rows.  The v2 input is
the frozen 256-D x plus a 64-D, column-normalized base-gain projection in dB.
Teacher spectra are never read in this round.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

for _name in ("CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, ":4096:8" if _name == "CUBLAS_WORKSPACE_CONFIG" else "2")

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
NUM_BANDS = 64
NUM_BINS = 257
X_DIM = 256
EPS = 1e-8
BASE_FLOOR_DB = -9.0
FINAL_FLOOR_DB = -21.0
RESIDUAL_LIMIT_DB = 12.0
SEED = 20261117
STEP_BUDGET = 0  # Zero uses every stream in all requested epochs.
OLD_CLEAN_LOSS_SOURCE_SHA256 = "68eb4595c760d70fefd50752499314fc177ea43fafbcf48bd38ef7e3fa59418f"
FEATURE_V2 = "dpdf-frame-v2-basegain-db"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(name: str, *, self_test: bool) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if not self_test and device.type != "cuda":
        raise RuntimeError("formal training requires --device cuda")
    return device


def band_centers() -> np.ndarray:
    b = np.arange(NUM_BANDS, dtype=np.float64)
    return np.where(
        b < 24, 1000.0 * b / 23.0,
        np.where(b < 40, 1000.0 + 3000.0 * (b - 23.0) / 16.0,
                 4000.0 + 4000.0 * (b - 39.0) / 24.0),
    )


def band_matrices(device: torch.device) -> tuple[Tensor, Tensor]:
    centers = torch.from_numpy(band_centers()).to(device=device, dtype=torch.float32)
    hz = torch.arange(NUM_BINS, device=device, dtype=torch.float32) * (16000.0 / 512.0)
    left = (torch.searchsorted(centers, hz, right=False) - 1).clamp(0, NUM_BANDS - 2)
    w = ((hz - centers[left]) / (centers[left + 1] - centers[left])).clamp(0.0, 1.0)
    interp = torch.zeros((NUM_BINS, NUM_BANDS), device=device, dtype=torch.float32)
    rows = torch.arange(NUM_BINS, device=device)
    interp[rows, left] = 1.0 - w
    interp[rows, left + 1] += w
    projection = interp / interp.sum(0).clamp_min(1e-12).unsqueeze(0)
    return projection, interp


def interpolate_bands(values: Tensor, matrix: Tensor) -> Tensor:
    return values @ matrix.T


def final_gain(base_gain: Tensor, residual_db: Tensor, floor: float, matrix: Tensor) -> Tensor:
    return torch.clamp(
        base_gain * torch.pow(10.0, interpolate_bands(residual_db, matrix) / 20.0),
        floor, 1.0,
    )


def ls_projection(target: Tensor, mixture: Tensor, floor: float) -> Tensor:
    numerator = target.real * mixture.real + target.imag * mixture.imag
    denominator = mixture.real.square() + mixture.imag.square() + EPS
    return (numerator / denominator).clamp(floor, 1.0)


def _compress_complex(value: Tensor) -> Tensor:
    magnitude = value.abs().clamp_min(EPS)
    return value * (magnitude.pow(0.3) / magnitude)


def clean_loss(
    residual_db: Tensor, base_gain: Tensor, mixture: Tensor, clean: Tensor,
    valid_frames: Tensor, floor: float, matrix: Tensor, *, reduction: str = "mean",
) -> tuple[Tensor, Tensor, Tensor]:
    """Exact copy of the prior trainer's supervised clean_loss formula."""
    predicted_gain = final_gain(base_gain, residual_db, floor, matrix)
    target_gain = ls_projection(clean, mixture, floor)
    predicted = mixture * predicted_gain
    complex_error = (_compress_complex(predicted) - _compress_complex(clean)).abs().square()
    magnitude_error = (
        predicted.abs().clamp_min(EPS).pow(0.3) - clean.abs().clamp_min(EPS).pow(0.3)
    ).square()
    target_residual = (
        20.0 * torch.log10(target_gain.clamp_min(EPS))
        - 20.0 * torch.log10(base_gain.clamp_min(EPS))
    ).clamp(-RESIDUAL_LIMIT_DB, RESIDUAL_LIMIT_DB)
    raw_residual = interpolate_bands(residual_db, matrix)
    residual_error = F.smooth_l1_loss(raw_residual, target_residual, beta=1.0, reduction="none")
    clean_power = clean.abs().square()
    frame_peak = clean_power.amax(dim=-1, keepdim=True).clamp_min(EPS)
    relative = clean_power / frame_peak
    weak = (clean_power > EPS) & (relative <= 10.0 ** (-30.0 / 10.0))
    weight = relative.clamp_min(1e-3).pow(0.3) + 0.5 * weak.float()
    mask = valid_frames.to(weight.dtype).unsqueeze(-1).expand_as(weight) * weight
    per_element = complex_error + magnitude_error + 0.5 * residual_error
    total = (per_element * mask).sum()
    count = mask.sum().clamp_min(1.0)
    return (total / count if reduction == "mean" else total), count, target_gain


def make_input(x: Tensor, base_gain: Tensor, input_dim: int, projection: Tensor) -> Tensor:
    x = x.to(projection.device)
    base_gain = base_gain.to(projection.device)
    if input_dim == X_DIM:
        return x
    if input_dim != 320:
        raise ValueError("input_dim must be 256 or 320")
    projected = base_gain @ projection
    base_db = 20.0 * torch.log10(projected.clamp_min(EPS))
    return torch.cat((x, base_db), dim=-1)


def frame_weights(clean_spec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute label-only weights over the complete stream."""
    clean_power = np.abs(clean_spec).astype(np.float64) ** 2
    energy = clean_power.sum(axis=1)
    p95 = float(np.percentile(energy, 95)) if energy.size else 0.0
    active_threshold = max(p95 * 1e-5, EPS)
    active = energy > active_threshold
    weak = active & (energy < p95 * 0.03)
    onset = np.zeros_like(active, dtype=bool)
    tail = np.zeros_like(active, dtype=bool)
    for i in np.flatnonzero(active):
        onset[i] = i == 0 or bool(np.any(~active[max(0, i - 20):i]))
        tail[i] = i == len(active) - 1 or bool(np.any(~active[i + 1:min(len(active), i + 21)]))
    weights = np.zeros(energy.shape, dtype=np.float32)
    weights[active] = (1.0 + 2.0 * (weak | onset | tail)[active]).astype(np.float32)
    return weights, active, weak, onset, tail, energy


@dataclass
class Stream:
    row_id: str
    split: str
    x: Tensor
    base_gain: Tensor
    mix_spec: Tensor
    clean_spec: Tensor
    weights: Tensor
    active: Tensor
    weak: Tensor
    speech_coeff: float
    clean_to_noise_db: float

    @property
    def frames(self) -> int:
        return int(self.x.shape[0])


def _row_path(row: dict[str, Any], manifest: Path) -> Path:
    value = row.get("npzpath", row.get("npz_path"))
    if not isinstance(value, str) or not value:
        raise ValueError(f"manifest row {row.get('id')} lacks npzpath")
    path = Path(value)
    return path if path.is_absolute() else manifest.parent / path


def _stream_from_arrays(row_id: str, split: str, x: np.ndarray, base: np.ndarray,
                        mix: np.ndarray, clean: np.ndarray) -> Stream:
    if x.ndim != 2 or x.shape[1] != X_DIM or base.shape != (x.shape[0], NUM_BINS):
        raise ValueError(f"bad x/base_gain shape for {row_id}")
    if mix.shape != base.shape or clean.shape != base.shape:
        raise ValueError(f"bad spectrum shape for {row_id}")
    if not np.isfinite(x).all() or not np.isfinite(base).all() or not np.isfinite(mix).all() or not np.isfinite(clean).all():
        raise ValueError(f"non-finite data in {row_id}")
    weights, active, weak, _, _, energy = frame_weights(clean)
    clean_energy = float(energy[active].sum())
    noise_energy = float((np.abs(mix - clean).astype(np.float64) ** 2).sum(axis=1)[active].sum())
    ratio_db = 10.0 * math.log10((clean_energy + EPS) / (noise_energy + EPS))
    coeff = 4.0 if ratio_db > 20.0 else 2.0
    return Stream(
        row_id, split, torch.from_numpy(x.copy()), torch.from_numpy(base.copy()),
        torch.from_numpy(mix.copy()), torch.from_numpy(clean.copy()),
        torch.from_numpy(weights), torch.from_numpy(active), torch.from_numpy(weak),
        coeff, ratio_db,
    )


def load_streams(rows: Iterable[dict[str, Any]], manifest: Path) -> list[Stream]:
    streams: list[Stream] = []
    for row in rows:
        path = _row_path(row, manifest)
        expected = row.get("npz_sha256", row.get("hash", row.get("sha256")))
        if not isinstance(expected, str) or sha256_file(path) != expected.lower():
            raise ValueError(f"NPZ identity mismatch: {path}")
        with np.load(path, allow_pickle=False) as z:
            required = ("x", "base_gain", "mix_spec", "clean_spec")
            if any(key not in z.files for key in required):
                raise ValueError(f"missing fields in {path}")
            x = np.asarray(z["x"], dtype=np.float32)
            base = np.asarray(z["base_gain"], dtype=np.float32)
            mix = np.asarray(z["mix_spec"], dtype=np.complex64)
            clean = np.asarray(z["clean_spec"], dtype=np.complex64)
        streams.append(_stream_from_arrays(str(row["id"]), str(row["split"]), x, base, mix, clean))
    return streams


def read_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("rows")
    if not isinstance(rows, list):
        raise ValueError("manifest must contain rows")
    ids = [r.get("id") for r in rows]
    if any(not isinstance(v, str) or not v for v in ids) or len(ids) != len(set(ids)):
        raise ValueError("manifest ids must be unique non-empty strings")
    train = [r for r in rows if r.get("split") == "train"]
    dev = [r for r in rows if r.get("split") == "dev"]
    if not train or not dev:
        raise ValueError("manifest needs non-empty train and dev splits")
    return raw, train, dev


def _pad(value: Tensor, amount: int) -> Tensor:
    return value if amount == 0 else torch.cat((value, value.new_zeros((amount, *value.shape[1:]))), 0)


def collate(streams: list[Stream], start: int, frames: int, device: torch.device,
            input_dim: int, projection: Tensor):
    xs, bases, mixes, cleans, weights, weaks, coeffs, lengths = [], [], [], [], [], [], [], []
    for s in streams:
        length = min(frames, s.frames - start)
        if length <= 0:
            continue
        end, pad = start + length, frames - length
        xs.append(_pad(make_input(s.x[start:end], s.base_gain[start:end], input_dim, projection), pad))
        bases.append(_pad(s.base_gain[start:end], pad))
        mixes.append(_pad(s.mix_spec[start:end], pad))
        cleans.append(_pad(s.clean_spec[start:end], pad))
        weights.append(_pad(s.weights[start:end], pad))
        weaks.append(_pad(s.weak[start:end], pad))
        coeffs.append(torch.full((1,), s.speech_coeff))
        lengths.append(length)
    if not lengths:
        raise ValueError("no active stream in chunk")
    valid = torch.arange(frames)[None, :] < torch.tensor(lengths)[:, None]
    out = [torch.stack(v).to(device) for v in (xs, bases, mixes, cleans, weights, weaks)]
    speech_coeff = torch.cat(coeffs).to(device)
    return (*out, valid.to(device), speech_coeff, sum(lengths))


def speech_preserving_loss(residual_db: Tensor, base_gain: Tensor, mixture: Tensor, clean: Tensor,
                           weights: Tensor, weak: Tensor, valid: Tensor, coeff: Tensor,
                           floor: float, matrix: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
    predicted_gain = final_gain(base_gain, residual_db, floor, matrix)
    noise = mixture - clean
    clean_power = clean.abs().square()
    noise_power = noise.abs().square()
    # All real frames contribute noise loss, including speech pauses and
    # noise-only streams. Clean activity controls speech protection only.
    energy = (clean_power + noise_power) * valid.unsqueeze(-1).to(clean_power.dtype)
    denominator = energy.sum(dim=(1, 2)) + EPS
    speech_terms = (
        weights.unsqueeze(-1) * (1.0 - predicted_gain).square() * clean_power
    ).sum(dim=(1, 2)) / denominator
    noise_terms = (
        predicted_gain.square() * noise_power * valid.unsqueeze(-1).to(noise_power.dtype)
    ).sum(dim=(1, 2)) / denominator
    distortion = (coeff.reshape(-1) * speech_terms).mean()
    residual_noise = noise_terms.mean()
    main = distortion + residual_noise

    irm = torch.sqrt(clean_power / (clean_power + noise_power + EPS))
    target = (
        20.0 * torch.log10(irm.clamp_min(EPS))
        - 20.0 * torch.log10(base_gain.clamp_min(EPS))
    ).clamp(-RESIDUAL_LIMIT_DB, RESIDUAL_LIMIT_DB)
    raw = interpolate_bands(residual_db, matrix)
    aux_element = F.smooth_l1_loss(raw / RESIDUAL_LIMIT_DB, target / RESIDUAL_LIMIT_DB,
                                   beta=0.1, reduction="none")
    defined = valid.unsqueeze(-1) & ((clean_power + noise_power) > EPS)
    aux = 0.01 * (aux_element * defined.to(aux_element.dtype)).sum() / defined.sum().clamp_min(1)
    weight_sum = (weights * valid.to(weights.dtype)).sum().clamp_min(1.0)
    weak_ratio = (weights * weak.to(weights.dtype) * valid.to(weights.dtype)).sum() / weight_sum
    parts = {
        "main": main,
        "component_distortion": speech_terms.mean(),
        "distortion_weighted": distortion,
        "residual_noise": residual_noise,
        "aux_loss": aux,
        "aux_main_ratio": aux / main.clamp_min(EPS),
        "weighted_weak_active_ratio": weak_ratio,
    }
    return main + aux, parts


def evaluate(model: nn.Module, streams: list[Stream], args: argparse.Namespace,
             device: torch.device, projection: Tensor, matrix: Tensor, floor: float) -> dict[str, float]:
    model.eval()
    totals = {k: 0.0 for k in ("objective", "main", "component_distortion", "residual_noise", "aux_loss", "weighted_weak_active_ratio")}
    count = 0
    with torch.inference_mode():
        for offset in range(0, len(streams), args.batch_size):
            group = streams[offset:offset + args.batch_size]
            hidden = model.initial_state(len(group), device=device, dtype=torch.float32)
            for start in range(0, max(s.frames for s in group), args.chunk_frames):
                active = [i for i, s in enumerate(group) if start < s.frames]
                if not active:
                    continue
                x, base, mix, clean, weights, weak, valid, coeff, _ = collate(
                    [group[i] for i in active], start, args.chunk_frames, device, args.input_dim, projection)
                residual, hidden_out = model(x, hidden[active])
                objective, parts = speech_preserving_loss(
                    residual, base, mix, clean, weights, weak, valid, coeff, floor, matrix)
                n = len(active)
                totals["objective"] += float(objective) * n
                totals["main"] += float(parts["main"]) * n
                totals["component_distortion"] += float(parts["component_distortion"]) * n
                totals["residual_noise"] += float(parts["residual_noise"]) * n
                totals["aux_loss"] += float(parts["aux_loss"]) * n
                totals["weighted_weak_active_ratio"] += float(parts["weighted_weak_active_ratio"]) * n
                count += n
                hidden = hidden.detach().clone()
                hidden[active] = hidden_out.detach()
    model.train()
    return {k: v / max(count, 1) for k, v in totals.items()}


def checkpoint_metadata(args: argparse.Namespace, *, epoch: int, steps: int, manifest_sha: str,
                        run_id: str, best_metric: float) -> dict[str, Any]:
    contract, feature_version, _ = __import__("student.model", fromlist=["feature_contract"]).feature_contract(args.input_dim)
    recipe = "old_clean_loss_v1" if args.loss == "old" else "speech_preserving_cn_v1"
    return {
        "input_dim": args.input_dim, "inputdim": args.input_dim,
        "feature_version": feature_version, "featurever": feature_version,
        "feature_contract": contract, "trained": True, "fixture": False,
        "steps": steps, "optimizer_steps": steps, "epoch": epoch, "epochs": args.epochs,
        "training_manifest_sha256": manifest_sha, "run_id": run_id, "seed": args.seed,
        "loss_recipe": recipe, "lossrecipe": recipe,
        "base_floor_db": BASE_FLOOR_DB, "basefloor": BASE_FLOOR_DB,
        "final_floor_db": FINAL_FLOOR_DB, "finalfloor": FINAL_FLOOR_DB,
        "floor_db": FINAL_FLOOR_DB, "absolute_floor_db": FINAL_FLOOR_DB,
        "choice": "dev", "selection_metric": "dev_objective", "best_dev_objective": best_metric,
        "gain_mode": "residual_db",
        "model_sha256": sha256_file(HERE / "student/model.py"),
        "learning_rate": args.learning_rate, "batch_size": args.batch_size,
        "chunk_frames": args.chunk_frames, "continuous_state": "carried_across_40s_detach_no_silence_reset",
        "old_clean_loss_source_sha256": OLD_CLEAN_LOSS_SOURCE_SHA256,
    }


def save_checkpoint(path: Path, model: nn.Module, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "metadata": metadata}, path)


def _write_run(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def train(args: argparse.Namespace) -> None:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = resolve_device(args.device, self_test=False)
    seed_all(args.seed)
    manifest = args.manifest.resolve()
    raw, train_rows, dev_rows = read_manifest(manifest)
    manifest_sha = sha256_file(manifest)
    projection, matrix = band_matrices(device)
    train_streams = load_streams(train_rows, manifest)
    dev_streams = load_streams(dev_rows, manifest)
    if args.max_steps == 0:
        per_epoch = sum(
            math.ceil(max(s.frames for s in train_streams[i:i + args.batch_size]) / args.chunk_frames)
            for i in range(0, len(train_streams), args.batch_size)
        )
        args.max_steps = args.epochs * per_epoch
    model_sha = sha256_file(HERE / "student/model.py")
    run_id = f"c64v2-{args.loss}-{args.input_dim}-{args.seed}-{manifest_sha[:12]}"
    reserved = ("best.pt", "last.pt", "run.json", "training.csv", "training.jsonl", "training-manifest.json")
    if any((args.output_dir / name).exists() for name in reserved):
        raise FileExistsError(f"refusing to overwrite existing training run: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_manifest = dict(raw)
    training_manifest["v2_trainer"] = {
        "input_dim": args.input_dim, "loss": args.loss, "base_floor_db": BASE_FLOOR_DB,
        "final_floor_db": FINAL_FLOOR_DB, "step_budget": args.max_steps,
        "old_clean_loss_source_sha256": OLD_CLEAN_LOSS_SOURCE_SHA256,
    }
    _write_run(args.output_dir / "training-manifest.json", training_manifest)
    jsonl = (args.output_dir / "training.jsonl").open("w", encoding="utf-8")
    csv_file = (args.output_dir / "training.csv").open("w", newline="", encoding="utf-8")
    fields = ["epoch", "loss_recipe", "full_train_loss", "train_main_loss", "component_distortion",
              "residual_noise", "aux_loss", "aux_main_ratio", "weighted_weak_active_ratio",
              "grad_norm", "first_batch_grad_norm", "valid_frames", "step_count", "wall_seconds",
              "gpu_peak_mb", "dev_objective", "dev_component_distortion", "dev_residual_noise",
              "dev_aux_loss", "dev_weighted_weak_active_ratio"]
    writer = csv.DictWriter(csv_file, fieldnames=fields)
    writer.writeheader()
    model = __import__("student.model", fromlist=["ResidualStudent"]).ResidualStudent(
        input_dim=args.input_dim, hidden=64).to(device=device, dtype=torch.float32)
    seed_all(args.seed)
    with torch.no_grad():
        nn.init.zeros_(model.head.weight)
        nn.init.zeros_(model.head.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed)
    total_steps, best, first_grad, started = 0, math.inf, None, time.monotonic()
    completed_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        order = torch.randperm(len(train_streams), generator=generator).tolist()
        groups = [order[i:i + args.batch_size] for i in range(0, len(order), args.batch_size)]
        sums = {k: 0.0 for k in ("full", "main", "distortion", "noise", "aux", "weak", "grad", "frames")}
        epoch_start, epoch_steps = time.monotonic(), 0
        step_target = min(args.max_steps, ((epoch * args.max_steps) + args.epochs - 1) // args.epochs)
        for group in groups:
            if total_steps >= step_target:
                break
            group_streams = [train_streams[i] for i in group]
            hidden = model.initial_state(len(group_streams), device=device, dtype=torch.float32)
            for start in range(0, max(s.frames for s in group_streams), args.chunk_frames):
                if total_steps >= step_target:
                    break
                active = [i for i, s in enumerate(group_streams) if start < s.frames]
                if not active:
                    continue
                x, base, mix, clean, weights, weak, valid, coeff, count = collate(
                    [group_streams[i] for i in active], start, args.chunk_frames, device, args.input_dim, projection)
                residual, hidden_out = model(x, hidden[active])
                if args.loss == "old":
                    main, _, _ = clean_loss(residual, base, mix, clean, valid, floor=10.0 ** (FINAL_FLOOR_DB / 20.0), matrix=matrix)
                    full = main
                    parts = {"main": main, "component_distortion": torch.zeros_like(main),
                             "residual_noise": torch.zeros_like(main), "aux_loss": torch.zeros_like(main),
                             "aux_main_ratio": torch.zeros_like(main), "weighted_weak_active_ratio": torch.zeros_like(main)}
                else:
                    full, parts = speech_preserving_loss(
                        residual, base, mix, clean, weights, weak, valid, coeff,
                        floor=10.0 ** (FINAL_FLOOR_DB / 20.0), matrix=matrix)
                if not bool(torch.isfinite(full)):
                    raise FloatingPointError("non-finite training loss")
                optimizer.zero_grad(set_to_none=True)
                full.backward()
                grad = nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total_steps += 1
                epoch_steps += 1
                if first_grad is None:
                    first_grad = float(grad.detach())
                sums["full"] += float(full.detach()) * count
                sums["main"] += float(parts["main"].detach()) * count
                sums["distortion"] += float(parts["component_distortion"].detach()) * count
                sums["noise"] += float(parts["residual_noise"].detach()) * count
                sums["aux"] += float(parts["aux_loss"].detach()) * count
                sums["weak"] += float(parts["weighted_weak_active_ratio"].detach()) * count
                sums["grad"] += float(grad.detach()) * count
                sums["frames"] += count
                hidden = hidden.detach().clone()
                hidden[active] = hidden_out.detach()
        dev = evaluate(model, dev_streams, args, device, projection, matrix, 10.0 ** (FINAL_FLOOR_DB / 20.0))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        frames = max(sums["frames"], 1.0)
        row = {
            "epoch": epoch, "loss_recipe": "old_clean_loss_v1" if args.loss == "old" else "speech_preserving_cn_v1",
            "full_train_loss": sums["full"] / frames, "train_main_loss": sums["main"] / frames,
            "component_distortion": sums["distortion"] / frames, "residual_noise": sums["noise"] / frames,
            "aux_loss": sums["aux"] / frames, "aux_main_ratio": (sums["aux"] / max(sums["main"], EPS)),
            "weighted_weak_active_ratio": sums["weak"] / frames, "grad_norm": sums["grad"] / frames,
            "first_batch_grad_norm": first_grad, "valid_frames": int(sums["frames"]),
            "step_count": epoch_steps, "wall_seconds": time.monotonic() - epoch_start,
            "gpu_peak_mb": (torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else 0.0,
            "dev_objective": dev["objective"], "dev_component_distortion": dev["component_distortion"],
            "dev_residual_noise": dev["residual_noise"], "dev_aux_loss": dev["aux_loss"],
            "dev_weighted_weak_active_ratio": dev["weighted_weak_active_ratio"],
        }
        writer.writerow(row)
        csv_file.flush()
        jsonl.write(json.dumps(row, sort_keys=True) + "\n")
        jsonl.flush()
        print(json.dumps(row, sort_keys=True), flush=True)
        if dev["objective"] < best:
            best = dev["objective"]
            save_checkpoint(args.output_dir / "best.pt", model,
                            checkpoint_metadata(args, epoch=epoch, steps=total_steps, manifest_sha=manifest_sha,
                                                run_id=run_id, best_metric=best))
        save_checkpoint(args.output_dir / "last.pt", model,
                        checkpoint_metadata(args, epoch=epoch, steps=total_steps, manifest_sha=manifest_sha,
                                            run_id=run_id, best_metric=best))
        completed_epochs = epoch
        if total_steps >= args.max_steps:
            break
    summary = {
        "run_id": run_id, "trained": True, "fixture": False, "loss_recipe": "old_clean_loss_v1" if args.loss == "old" else "speech_preserving_cn_v1",
        "input_dim": args.input_dim, "seed": args.seed, "epochs_requested": args.epochs,
        "epochs_completed": completed_epochs, "steps": total_steps, "step_budget": args.max_steps,
        "best_dev_objective": best, "training_manifest_sha256": manifest_sha,
        "train_rows": len(train_rows), "dev_rows": len(dev_rows), "model_sha256": model_sha,
        "base_floor_db": BASE_FLOOR_DB, "final_floor_db": FINAL_FLOOR_DB,
        "old_clean_loss_source_sha256": OLD_CLEAN_LOSS_SOURCE_SHA256,
        "checkpoints": {"best": str(args.output_dir / "best.pt"), "last": str(args.output_dir / "last.pt")},
    }
    jsonl.close()
    csv_file.close()
    _write_run(args.output_dir / "run.json", summary)


def _synthetic_streams(device: torch.device) -> list[Stream]:
    streams = []
    for i in range(2):
        x = torch.randn(40, X_DIM, device=device) * 0.1
        base = torch.full((40, NUM_BINS), 0.5, device=device)
        clean = torch.zeros((40, NUM_BINS), dtype=torch.complex64, device=device)
        noise = torch.zeros_like(clean)
        clean[5, 10 + i] = 1.0
        clean[25, 20 + i] = 0.01
        clean[39, 30 + i] = 0.5
        noise[5, 40 + i] = 0.05
        noise[25, 50 + i] = 0.02
        mix = clean + noise
        weights, active, weak, _, _, energy = frame_weights(clean.cpu().numpy())
        clean_energy = float(energy[active].sum())
        noise_energy = float((np.abs(mix.cpu().numpy() - clean.cpu().numpy()).astype(np.float64) ** 2).sum(axis=1)[active].sum())
        ratio = 10.0 * math.log10((clean_energy + EPS) / (noise_energy + EPS))
        streams.append(Stream(
            f"fixture-{i}", "train", x.cpu(), base.cpu(), mix.cpu(), clean.cpu(),
            torch.from_numpy(weights), torch.from_numpy(active), torch.from_numpy(weak),
            4.0 if ratio > 20.0 else 2.0, ratio,
        ))
    return streams


def run_self_test(args: argparse.Namespace) -> None:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = resolve_device(args.device, self_test=True)
    seed_all(args.seed)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    projection, matrix = band_matrices(device)
    torch.testing.assert_close(matrix.sum(dim=1), torch.ones(NUM_BINS, device=device), atol=1e-6, rtol=0)
    streams = _synthetic_streams(device)
    weights = streams[0].weights
    assert float(weights[5]) == 3.0 and float(weights[25]) == 3.0 and float(weights[39]) == 3.0
    c0 = collate([streams[0]], 0, 20, device, 256, projection)
    c1 = collate([streams[0]], 20, 20, device, 256, projection)
    assert torch.equal(c0[4][0, :20].cpu(), streams[0].weights[:20])
    assert torch.equal(c1[4][0, :20].cpu(), streams[0].weights[20:])
    zero_mix = torch.zeros(1, 1, NUM_BINS, dtype=torch.complex64, device=device)
    zero_clean = torch.zeros_like(zero_mix)
    zero_base = torch.ones(1, 1, NUM_BINS, device=device)
    zero_weights = torch.zeros(1, 1, device=device)
    zero_weak = torch.zeros(1, 1, dtype=torch.bool, device=device)
    zero_valid = torch.ones(1, 1, dtype=torch.bool, device=device)
    zero_coeff = torch.tensor([2.0], device=device)
    old_zero, _, _ = clean_loss(torch.zeros(1, 1, NUM_BANDS, device=device), zero_base, zero_mix,
                                zero_clean, zero_valid, 10.0 ** (FINAL_FLOOR_DB / 20.0), matrix)
    new_zero, zero_parts = speech_preserving_loss(
        torch.zeros(1, 1, NUM_BANDS, device=device), zero_base, zero_mix, zero_clean,
        zero_weights, zero_weak, zero_valid, zero_coeff, 10.0 ** (FINAL_FLOOR_DB / 20.0), matrix)
    assert torch.isfinite(old_zero) and torch.isfinite(new_zero)
    # Noise-only streams must carry a learning signal; speech activity is
    # not a mask for the noise objective.
    noise_delta = torch.zeros(1, 1, NUM_BANDS, device=device, requires_grad=True)
    noise_only = torch.ones_like(zero_mix) * 0.1
    noise_value, _ = speech_preserving_loss(
        noise_delta, zero_base, noise_only, zero_clean, zero_weights,
        zero_weak, zero_valid, zero_coeff,
        10.0 ** (FINAL_FLOOR_DB / 20.0), matrix)
    noise_gradient, = torch.autograd.grad(noise_value, noise_delta)
    assert float(noise_value.detach()) > 0 and float(noise_gradient.abs().max()) > 0
    assert torch.allclose(final_gain(zero_base, torch.zeros(1, 1, NUM_BANDS, device=device),
                                     10.0 ** (FINAL_FLOOR_DB / 20.0), matrix), zero_base)
    checks = []
    for input_dim in (256, 320):
        for loss_name in ("old", "speech_preserving"):
            model = __import__("student.model", fromlist=["ResidualStudent"]).ResidualStudent(input_dim=input_dim, hidden=64).to(device)
            seed_all(args.seed)
            with torch.no_grad():
                nn.init.zeros_(model.head.weight)
                nn.init.zeros_(model.head.bias)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
            before = [p.detach().clone() for p in model.parameters()]
            hidden = model.initial_state(len(streams), device=device, dtype=torch.float32)
            first_state = hidden.detach().clone()
            for start in (0, 20):
                x, base, mix, clean, w, weak, valid, coeff, _ = collate(streams, start, 20, device, input_dim, projection)
                residual, hidden_out = model(x, hidden)
                if loss_name == "old":
                    loss, _, _ = clean_loss(residual, base, mix, clean, valid, 10.0 ** (FINAL_FLOOR_DB / 20.0), matrix)
                else:
                    loss, _ = speech_preserving_loss(residual, base, mix, clean, w, weak, valid, coeff,
                                                      10.0 ** (FINAL_FLOOR_DB / 20.0), matrix)
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                hidden = hidden.detach().clone()
                hidden[:] = hidden_out.detach()
            assert not torch.equal(first_state, hidden)
            assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
            model.eval()
            with torch.inference_mode():
                full_x = make_input(torch.stack([s.x for s in streams]), torch.stack([s.base_gain for s in streams]), input_dim, projection)
                full, full_h = model(full_x, model.initial_state(2, device=device, dtype=torch.float32))
                state, chunks = model.initial_state(2, device=device, dtype=torch.float32), []
                for start in (0, 20):
                    part, state = model(full_x[:, start:start + 20], state)
                    chunks.append(part)
                torch.testing.assert_close(torch.cat(chunks, dim=1), full, atol=1e-3, rtol=1e-3)
                torch.testing.assert_close(state, full_h, atol=1e-3, rtol=1e-3)
            manifest = out / "self-test-manifest.json"
            manifest.write_text(json.dumps({"rows": [{"id": "fixture", "split": "dev"}]}, indent=2) + "\n", encoding="utf-8")
            fixture_args = copy(args)
            fixture_args.input_dim, fixture_args.loss = input_dim, loss_name
            metadata = checkpoint_metadata(fixture_args, epoch=1, steps=2, manifest_sha=sha256_file(manifest),
                                           run_id=f"selftest-{input_dim}-{loss_name}", best_metric=0.0)
            metadata["model_sha256"] = sha256_file(HERE / "student/model.py")
            checkpoint = out / f"self-test-{input_dim}-{loss_name}.pt"
            save_checkpoint(checkpoint, model, metadata)
            restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
            loaded = __import__("student.model", fromlist=["ResidualStudent"]).ResidualStudent(input_dim=input_dim, hidden=64)
            loaded.load_state_dict(restored["state_dict"], strict=True)
            assert restored["metadata"]["input_dim"] == input_dim
            from student.export import export_checkpoint
            exported = export_checkpoint(checkpoint, out / f"self-test-export-{input_dim}-{loss_name}", trained=True)
            assert exported["trained"] is True and exported["fixture"] is False
            assert exported["input_dim"] == input_dim and exported["steps"] == 2
            checks.append({"input_dim": input_dim, "loss": loss_name, "state_carry": True,
                           "updated": True, "export_metadata": True})
    result = {"status": "pass", "checks": checks, "device": str(device), "testdata_loaded": False,
              "weight_cross_chunk": True, "zero_energy_finite": True}
    (out / "self-test.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--loss", choices=("old", "speech_preserving"), default="speech_preserving")
    p.add_argument("--input-dim", type=int, choices=(256, 320), default=256)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--chunk-frames", type=int, default=200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--max-steps", type=int, default=STEP_BUDGET)
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.chunk_frames < 1 or args.max_steps < 0:
        p.error("epochs, batch-size and chunk-frames must be positive; max-steps must be nonnegative")
    if not args.self_test and args.manifest is None:
        p.error("formal training requires --manifest")
    return args


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(HERE))
    if args.self_test:
        run_self_test(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
