#!/usr/bin/env python3
"""Train the causal C64 single-head residual student.

The trainer never reads eval/holdout rows.  It trains on ``train`` and selects
checkpoints only on ``dev`` with the clean supervised objective.
"""
from __future__ import annotations

import os

for _name in ("CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, ":4096:8" if _name == "CUBLAS_WORKSPACE_CONFIG" else "2")

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
NUM_BANDS = 64
NUM_BINS = 257
FEATURE_DIM = 256
FEATURE_CONTRACT = "dpdf-frame-v1-db"
EPS = 1e-8
BASE_FRONTEND_FLOOR_DB = -9.0
RESIDUAL_LIMIT_DB = 12.0
EXPECTED_NATIVE_PROBE_SHA256 = "4d1b3f139214bfad11a6c80e635635026dd2b0c3ce98fc5c4992c8ea4ce5209a"


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
        b < 24,
        1000.0 * b / 23.0,
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
    # Projection is for target-band aggregation; interpolation keeps row sum 1.
    projection = interp / interp.sum(0).clamp_min(1e-12).unsqueeze(0)
    return projection, interp


def interpolate_bands(values: Tensor, matrix: Tensor) -> Tensor:
    return values @ matrix.T


def final_gain(base_gain: Tensor, residual_db: Tensor, floor: float, matrix: Tensor) -> Tensor:
    return torch.clamp(base_gain * torch.pow(10.0, interpolate_bands(residual_db, matrix) / 20.0), floor, 1.0)


def ls_projection(target: Tensor, mixture: Tensor, floor: float) -> Tensor:
    numerator = (target.real * mixture.real + target.imag * mixture.imag)
    denominator = mixture.real.square() + mixture.imag.square() + EPS
    return (numerator / denominator).clamp(floor, 1.0)


def _compress_complex(value: Tensor) -> Tensor:
    magnitude = value.abs().clamp_min(EPS)
    return value * (magnitude.pow(0.3) / magnitude)


def clean_loss(
    residual_db: Tensor,
    base_gain: Tensor,
    mixture: Tensor,
    clean: Tensor,
    valid_frames: Tensor,
    floor: float,
    matrix: Tensor,
    *,
    reduction: str = "mean",
) -> tuple[Tensor, Tensor, Tensor]:
    """Clean-reference loss: compressed complex + magnitude + log-gain Huber.

    The clean LS/oracle mask is a label only.  It is never a runtime input or a
    quality score.  A power-derived weight plus a fixed weak-clean boost keeps
    quiet clean components in the objective.
    """
    predicted_gain = final_gain(base_gain, residual_db, floor, matrix)
    target_gain = ls_projection(clean, mixture, floor)
    predicted = mixture * predicted_gain
    complex_error = (_compress_complex(predicted) - _compress_complex(clean)).abs().square()
    magnitude_error = (predicted.abs().clamp_min(EPS).pow(0.3) - clean.abs().clamp_min(EPS).pow(0.3)).square()
    target_residual = (20.0 * torch.log10(target_gain.clamp_min(EPS)) -
                       20.0 * torch.log10(base_gain.clamp_min(EPS))).clamp(-RESIDUAL_LIMIT_DB, RESIDUAL_LIMIT_DB)
    raw_residual = interpolate_bands(residual_db, matrix)
    residual_error = F.smooth_l1_loss(raw_residual, target_residual, beta=1.0, reduction="none")
    clean_power = clean.abs().square()
    frame_peak = clean_power.amax(dim=-1, keepdim=True).clamp_min(EPS)
    relative = clean_power / frame_peak
    weak = (clean_power > EPS) & (relative <= 10.0 ** (-30.0 / 10.0))
    weight = relative.clamp_min(1e-3).pow(0.3) + 0.5 * weak.float()
    mask = valid_frames.to(weight.dtype).unsqueeze(-1).expand_as(weight) * weight
    # This aux remains differentiable when the clamped final gain is at a floor.
    per_element = complex_error + magnitude_error + 0.5 * residual_error
    total = (per_element * mask).sum()
    count = mask.sum().clamp_min(1.0)
    return (total / count if reduction == "mean" else total), count, target_gain


def kd_loss(
    residual_db: Tensor,
    base_gain: Tensor,
    mixture: Tensor,
    clean: Tensor,
    teacher: Tensor,
    valid_frames: Tensor,
    floor: float,
    matrix: Tensor,
) -> Tensor:
    teacher_gain = ls_projection(teacher, mixture, floor)
    predicted_gain = final_gain(base_gain, residual_db, floor, matrix)
    teacher_power = teacher.abs().square()
    clean_power = clean.abs().square()
    # Labels only: attenuated weak teacher events get less KD authority.
    weak_teacher = (teacher_power < 0.5 * clean_power) & (clean_power > EPS)
    weight = torch.where(weak_teacher, 0.25, 1.0)
    weight = weight * valid_frames.to(weight.dtype).unsqueeze(-1)
    error = F.smooth_l1_loss(
        20.0 * torch.log10(predicted_gain.clamp_min(EPS)),
        20.0 * torch.log10(teacher_gain.clamp_min(EPS)),
        beta=1.0,
        reduction="none",
    )
    return (error * weight).sum() / weight.sum().clamp_min(1.0)


@dataclass
class Stream:
    row_id: str
    split: str
    x: Tensor
    base_gain: Tensor
    mix_spec: Tensor
    clean_spec: Tensor
    teacher_spec: Tensor | None

    @property
    def frames(self) -> int:
        return int(self.x.shape[0])


def _row_path(row: dict[str, Any], manifest: Path) -> Path:
    value = row.get("npzpath", row.get("npz_path"))
    if not isinstance(value, str) or not value:
        raise ValueError(f"manifest row {row.get('id')} lacks npzpath")
    path = Path(value)
    return path if path.is_absolute() else manifest.parent / path


def load_streams(rows: Iterable[dict[str, Any]], manifest: Path, *, teacher: bool) -> list[Stream]:
    streams: list[Stream] = []
    for row in rows:
        path = _row_path(row, manifest)
        expected = row.get("npz_sha256", row.get("hash", row.get("sha256")))
        if not isinstance(expected, str) or sha256_file(path) != expected.lower():
            raise ValueError(f"NPZ identity mismatch: {path}")
        with np.load(path, allow_pickle=False) as z:
            x = np.asarray(z["x"], dtype=np.float32)
            base = np.asarray(z["base_gain"], dtype=np.float32)
            mix = np.asarray(z["mix_spec"], dtype=np.complex64)
            clean = np.asarray(z["clean_spec"], dtype=np.complex64)
            teacher_spec = np.asarray(z["teacher_spec"], dtype=np.complex64) if teacher else None
            if "power" in z.files:
                assert z["power"].shape == (x.shape[0], NUM_BANDS)
            if "noise" in z.files:
                assert z["noise"].shape == (x.shape[0], NUM_BANDS)
        if x.ndim != 2 or x.shape[1] != FEATURE_DIM or base.shape != (x.shape[0], NUM_BINS):
            raise ValueError(f"bad x/base_gain shape in {path}")
        if mix.shape != base.shape or clean.shape != base.shape or not np.isfinite(x).all():
            raise ValueError(f"bad spectra or nonfinite features in {path}")
        if teacher and (teacher_spec is None or teacher_spec.shape != base.shape):
            raise ValueError(f"bad teacher_spec shape in {path}")
        streams.append(Stream(
            str(row["id"]), str(row["split"]), torch.from_numpy(x.copy()),
            torch.from_numpy(base.copy()), torch.from_numpy(mix.copy()),
            torch.from_numpy(clean.copy()),
            torch.from_numpy(teacher_spec.copy()) if teacher_spec is not None else None,
        ))
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


def collate(streams: list[Stream], start: int, frames: int, device: torch.device, teacher: bool):
    xs, bases, mixes, cleans, teachers, lengths = [], [], [], [], [], []
    for s in streams:
        length = min(frames, s.frames - start)
        if length <= 0:
            continue
        end = start + length
        pad = frames - length
        xs.append(_pad(s.x[start:end], pad))
        bases.append(_pad(s.base_gain[start:end], pad))
        mixes.append(_pad(s.mix_spec[start:end], pad))
        cleans.append(_pad(s.clean_spec[start:end], pad))
        if teacher:
            assert s.teacher_spec is not None
            teachers.append(_pad(s.teacher_spec[start:end], pad))
        lengths.append(length)
    valid = torch.arange(frames)[None, :] < torch.tensor(lengths)[:, None]
    out = [torch.stack(v).to(device) for v in (xs, bases, mixes, cleans)]
    if teacher:
        out.append(torch.stack(teachers).to(device))
    return (*out, valid.to(device), sum(lengths))


def checkpoint_payload(model: nn.Module, metadata: dict[str, Any]) -> dict[str, Any]:
    return {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "metadata": metadata}


def save_checkpoint(path: Path, model: nn.Module, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, metadata), path)


def initialize_student(model: nn.Module, seed: int) -> None:
    """Same seeded initialization for both modes; zero head starts at 0 dB."""
    seed_all(seed)
    with torch.no_grad():
        nn.init.zeros_(model.head.weight)
        nn.init.zeros_(model.head.bias)


@torch.inference_mode()
def evaluate(model: nn.Module, streams: list[Stream], args: argparse.Namespace, device: torch.device, matrix: Tensor, floor: float) -> float:
    model.eval()
    total, weight = 0.0, 0.0
    for offset in range(0, len(streams), args.batch_size):
        group = streams[offset:offset + args.batch_size]
        hidden = model.initial_state(len(group), device=device, dtype=torch.float32)
        for start in range(0, max(s.frames for s in group), args.chunk_frames):
            active = [i for i, s in enumerate(group) if start < s.frames]
            if not active:
                continue
            x, base, mix, clean, valid, count = collate([group[i] for i in active], start, args.chunk_frames, device, False)
            residual, hidden_out = model(x, hidden[active])
            value, bins, _ = clean_loss(residual, base, mix, clean, valid, floor, matrix, reduction="sum")
            total += float(value)
            weight += float(bins)
            hidden = hidden.detach()
            hidden[active] = hidden_out
    return total / max(weight, 1.0)


def _metadata(args: argparse.Namespace, *, epoch: int, steps: int, stage: str, manifest_sha: str,
              model_sha: str, teacher_sha: str, native_sha: str | None, floor: float,
              run_id: str, best_metric: float, selection: str) -> dict[str, Any]:
    reachable = BASE_FRONTEND_FLOOR_DB - RESIDUAL_LIMIT_DB
    return {
        "trained": True, "fixture": False, "steps": steps, "optimizer_steps": steps,
        "total_epochs": args.kd_epochs + args.clean_epochs, "epoch": epoch, "stage": stage,
        "selection": selection, "selection_metric": "dev_clean_loss", "best_metric": best_metric,
        "training_manifest_sha256": manifest_sha, "featuresmanifestsha": manifest_sha,
        "model_sha256": model_sha, "teacher_sha256": teacher_sha,
        "native_probe_sha256": native_sha,
        "feature_contract": FEATURE_CONTRACT, "featurecontract": FEATURE_CONTRACT,
        "architecture": "ResidualStudent(hidden=64):256->64ReLU->GRU64->head64->12*tanh",
        "seed": args.seed, "mode": args.mode, "run_id": run_id,
        "absolute_floor_db": args.absolute_floor_db, "absolute_floor": floor,
        "base_frontend_floor_db": BASE_FRONTEND_FLOOR_DB,
        "reachable_floor_db": reachable, "residual_limit_db": RESIDUAL_LIMIT_DB,
        "learning_rate": args.learning_rate, "batch_size": args.batch_size,
        "chunk_frames": args.chunk_frames, "kd_epochs": args.kd_epochs,
        "clean_epochs": args.clean_epochs, "initialization": "torch_default+zero_head",
        "eval_loaded": False,
    }


def train(args: argparse.Namespace) -> None:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = resolve_device(args.device, self_test=False)
    seed_all(args.seed)
    previous_jsonl = args.output_dir / "training.jsonl"
    if (args.output_dir / "run.json").exists() or (previous_jsonl.exists() and previous_jsonl.stat().st_size > 0):
        raise FileExistsError(f"refusing to overwrite existing training run: {args.output_dir}")
    manifest = args.manifest.resolve()
    raw, train_rows, dev_rows = read_manifest(manifest)
    manifest_sha = sha256_file(manifest)
    teacher_sha = str(raw.get("teacher_sha256") or raw.get("teacher", {}).get("model_sha256") or raw.get("teacher", {}).get("sha256") or "not_declared")
    native_sha = sha256_file(HERE / "native/residual_ns_probe")
    if native_sha != EXPECTED_NATIVE_PROBE_SHA256:
        raise ValueError(f"native probe hash mismatch: {native_sha}")
    manifest_native = raw.get("native_probe_sha256") or raw.get("native_probe", {}).get("sha256")
    if manifest_native not in (None, native_sha):
        raise ValueError("manifest native_probe_sha256 does not match binary")
    floor = 10.0 ** (args.absolute_floor_db / 20.0)
    _, matrix = band_matrices(device)
    train_streams = load_streams(train_rows, manifest, teacher=args.mode == "kd_then_clean")
    dev_streams = load_streams(dev_rows, manifest, teacher=False)
    model_sha = sha256_file(HERE / "student/model.py")
    run_id = f"c64-{args.mode}-{args.seed}-floor{args.absolute_floor_db:g}-{manifest_sha[:12]}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_manifest = dict(raw)
    training_manifest["native_probe_sha256"] = native_sha
    (args.output_dir / "training-manifest.json").write_text(json.dumps(training_manifest, indent=2) + "\n", encoding="utf-8")
    jsonl = (args.output_dir / "training.jsonl").open("w", encoding="utf-8")
    csv_file = (args.output_dir / "training.csv").open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=["epoch", "stage", "loss", "kd_loss", "clean_loss", "lr", "grad_norm", "valid_frames", "step_count", "wall_seconds", "gpu_peak_mb", "dev_clean_loss"])
    writer.writeheader()
    model = ResidualStudent(hidden=64).to(device=device, dtype=torch.float32)
    initialize_student(model, args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed)
    total_steps = 0
    best_clean = math.inf
    best_kd = math.inf
    started = time.monotonic()
    total_epochs = args.kd_epochs + args.clean_epochs
    for epoch in range(1, total_epochs + 1):
        stage = "clean" if args.mode == "supervised" or epoch > args.kd_epochs else "kd"
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        order = torch.randperm(len(train_streams), generator=generator).tolist()
        groups = [order[i:i + args.batch_size] for i in range(0, len(order), args.batch_size)]
        sums = {"loss": 0.0, "kd": 0.0, "clean": 0.0, "grad": 0.0, "frames": 0.0, "steps": 0}
        epoch_start = time.monotonic()
        for group in groups:
            group_streams = [train_streams[i] for i in group]
            hidden = model.initial_state(len(group_streams), device=device, dtype=torch.float32)
            for start in range(0, max(s.frames for s in group_streams), args.chunk_frames):
                active = [i for i, s in enumerate(group_streams) if start < s.frames]
                if not active:
                    continue
                collated = collate([group_streams[i] for i in active], start, args.chunk_frames, device, stage == "kd")
                if stage == "kd":
                    x, base, mix, clean, teacher, valid, count = collated
                else:
                    x, base, mix, clean, valid, count = collated
                residual, hidden_out = model(x, hidden[active])
                clean_value, _, _ = clean_loss(residual, base, mix, clean, valid, floor, matrix)
                if stage == "kd":
                    kd_value = kd_loss(residual, base, mix, clean, teacher, valid, floor, matrix)
                    loss = kd_value + 0.25 * clean_value
                else:
                    kd_value = torch.zeros((), device=device)
                    loss = clean_value
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("non-finite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad = nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total_steps += 1
                sums["loss"] += float(loss.detach()) * count
                sums["kd"] += float(kd_value.detach()) * count
                sums["clean"] += float(clean_value.detach()) * count
                sums["grad"] += float(grad) * count
                sums["frames"] += count
                sums["steps"] += 1
                hidden = hidden.detach()
                hidden[active] = hidden_out.detach()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dev_metric = evaluate(model, dev_streams, args, device, matrix, floor)
        wall = time.monotonic() - epoch_start
        frames = max(sums["frames"], 1.0)
        row = {
            "epoch": epoch, "stage": stage, "loss": sums["loss"] / frames,
            "kd_loss": sums["kd"] / frames, "clean_loss": sums["clean"] / frames,
            "lr": optimizer.param_groups[0]["lr"], "grad_norm": sums["grad"] / frames,
            "valid_frames": int(sums["frames"]), "step_count": sums["steps"],
            "wall_seconds": wall,
            "gpu_peak_mb": (torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else 0.0,
            "dev_clean_loss": dev_metric,
        }
        writer.writerow(row)
        csv_file.flush()
        jsonl.write(json.dumps(row, sort_keys=True) + "\n")
        jsonl.flush()
        print(json.dumps(row, sort_keys=True), flush=True)
        if stage == "kd" and dev_metric < best_kd:
            best_kd = dev_metric
            save_checkpoint(args.output_dir / "best-kd.pt", model, _metadata(args, epoch=epoch, steps=total_steps, stage=stage, manifest_sha=manifest_sha, model_sha=model_sha, teacher_sha=teacher_sha, native_sha=native_sha, floor=floor, run_id=run_id, best_metric=best_kd, selection="best-kd"))
        if stage == "clean" and dev_metric < best_clean:
            best_clean = dev_metric
            save_checkpoint(args.output_dir / "best.pt", model, _metadata(args, epoch=epoch, steps=total_steps, stage=stage, manifest_sha=manifest_sha, model_sha=model_sha, teacher_sha=teacher_sha, native_sha=native_sha, floor=floor, run_id=run_id, best_metric=best_clean, selection="best"))
        save_checkpoint(args.output_dir / "last.pt", model, _metadata(args, epoch=epoch, steps=total_steps, stage=stage, manifest_sha=manifest_sha, model_sha=model_sha, teacher_sha=teacher_sha, native_sha=native_sha, floor=floor, run_id=run_id, best_metric=min(best_clean, best_kd), selection="last"))
        if args.mode == "kd_then_clean" and epoch == args.kd_epochs:
            for stream in train_streams:
                stream.teacher_spec = None
    summary = {
        "run_id": run_id, "mode": args.mode, "trained": True, "total_epochs": total_epochs,
        "steps": total_steps, "best_dev_clean_loss": best_clean,
        "best_kd_dev_clean_loss": best_kd if math.isfinite(best_kd) else None,
        "wall_seconds": time.monotonic() - started, "manifest_sha256": manifest_sha,
        "native_probe_sha256": native_sha, "model_sha256": model_sha,
        "absolute_floor_db": args.absolute_floor_db,
        "base_frontend_floor_db": BASE_FRONTEND_FLOOR_DB,
        "reachable_floor_db": BASE_FRONTEND_FLOOR_DB - RESIDUAL_LIMIT_DB,
        "eval_loaded": False, "checkpoints": {"best": str(args.output_dir / "best.pt"), "last": str(args.output_dir / "last.pt")},
    }
    (args.output_dir / "run.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    jsonl.close()
    csv_file.close()


def run_self_test(args: argparse.Namespace) -> None:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = resolve_device(args.device, self_test=True)
    seed_all(args.seed)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    floor = 10.0 ** (args.absolute_floor_db / 20.0)
    _, matrix = band_matrices(device)
    torch.testing.assert_close(matrix.sum(dim=1), torch.ones(NUM_BINS, device=device), atol=1e-6, rtol=0)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    x = torch.randn(2, 37, FEATURE_DIM, generator=generator, device=device)
    base = torch.rand(2, 37, NUM_BINS, generator=generator, device=device).clamp_min(floor)
    mix = torch.complex(torch.randn(2, 37, NUM_BINS, generator=generator, device=device), torch.randn(2, 37, NUM_BINS, generator=generator, device=device)) * 0.1
    clean = mix * torch.rand(2, 37, NUM_BINS, generator=generator, device=device)
    teacher = mix * torch.rand(2, 37, NUM_BINS, generator=generator, device=device)
    fixture = out / "selftest-fixtures"
    fixture.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in ("train", "dev"):
        path = fixture / f"{split}.npz"
        np.savez_compressed(path, x=x[0].cpu().numpy(), base_gain=base[0].cpu().numpy(), mix_spec=mix[0].cpu().numpy(), clean_spec=clean[0].cpu().numpy())
        rows.append({"id": split, "split": split, "language": "en", "npzpath": path.name, "npz_sha256": sha256_file(path)})
    rows.append({"id": "poison-test", "split": "test", "language": "en", "npzpath": "MUST_NOT_OPEN.npz", "hash": "0" * 64})
    manifest = fixture / "manifest.json"
    manifest.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    _, train_rows, dev_rows = read_manifest(manifest)
    loaded = load_streams(train_rows, manifest, teacher=False) + load_streams(dev_rows, manifest, teacher=False)
    assert [s.split for s in loaded] == ["train", "dev"]
    train_streams = [Stream(f"train{i}", "train", x[i].cpu(), base[i].cpu(), mix[i].cpu(), clean[i].cpu(), teacher[i].cpu()) for i in range(2)]
    dev_streams = [Stream(f"dev{i}", "dev", x[i].cpu(), base[i].cpu(), mix[i].cpu(), clean[i].cpu(), None) for i in range(2)]
    model = ResidualStudent(hidden=64).to(device=device, dtype=torch.float32)
    initialize_student(model, args.seed)
    with torch.inference_mode():
        initial, _ = model(x, model.initial_state(2, device=device, dtype=torch.float32))
        assert float(initial.abs().max()) == 0.0
        base_probe = torch.full((2, 1, NUM_BINS), 0.5, device=device)
        assert torch.allclose(final_gain(base_probe, torch.zeros(2, 1, 64, device=device), floor, matrix), base_probe)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    before = [p.detach().clone() for p in model.parameters()]
    args.batch_size, args.chunk_frames = 2, 20
    for stage in ("kd", "clean"):
        hidden = model.initial_state(2, device=device, dtype=torch.float32)
        for start in (0, 20):
            collated = collate(train_streams, start, args.chunk_frames, device, stage == "kd")
            if stage == "kd":
                xb, bb, mb, cb, tb, valid, _ = collated
            else:
                xb, bb, mb, cb, valid, _ = collated
            residual, hidden_out = model(xb, hidden)
            clean_value, _, _ = clean_loss(residual, bb, mb, cb, valid, floor, matrix)
            if stage == "kd":
                loss = kd_loss(residual, bb, mb, cb, tb, valid, floor, matrix) + 0.25 * clean_value
            else:
                loss = clean_value
            assert bool(torch.isfinite(loss))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            hidden = hidden.detach()
            hidden[:] = hidden_out.detach()
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    dev_metric = evaluate(model, dev_streams, args, device, matrix, floor)
    assert math.isfinite(dev_metric)
    model.eval()
    with torch.inference_mode():
        full, full_hidden = model(x, model.initial_state(2, device=device, dtype=torch.float32))
        state = model.initial_state(2, device=device, dtype=torch.float32)
        chunks = []
        for start in (0, 20):
            part, state = model(x[:, start:start + args.chunk_frames], state)
            chunks.append(part)
        chunked, chunk_hidden = torch.cat(chunks, dim=1), state
        torch.testing.assert_close(chunked, full, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(chunk_hidden, full_hidden, atol=1e-3, rtol=1e-3)
    manifest_sha = sha256_file(manifest)
    metadata = _metadata(args, epoch=1, steps=4, stage="kd", manifest_sha=manifest_sha,
                         model_sha=sha256_file(HERE / "student/model.py"), teacher_sha="selftest",
                         native_sha=None, floor=floor, run_id="selftest-run", best_metric=dev_metric,
                         selection="last")
    checkpoint = out / "self-test-checkpoint.pt"
    save_checkpoint(checkpoint, model, metadata)
    restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert restored["metadata"]["steps"] == 4 and len(restored["metadata"]["training_manifest_sha256"]) == 64
    from student.export import export_checkpoint
    exported = export_checkpoint(checkpoint, out / "self-test-export", trained=True)
    assert exported["trained"] is True and exported["steps"] == 4
    result = {"status": "pass", "checks": ["finite clean+KD loss", "TBPTT two chunks with carried hidden", "dev clean metric", "backward+parameter change", "zero-head init", "chunk hidden carry", "test split not loaded", "checkpoint save/load", "trained export metadata"], "device": str(device), "testdata_loaded": False}
    (out / "self-test.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=20261116)
    p.add_argument("--mode", choices=("kd_then_clean", "supervised"), default="kd_then_clean")
    p.add_argument("--kd-epochs", type=int, default=8)
    p.add_argument("--clean-epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--chunk-frames", type=int, default=200)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--absolute-floor-db", type=float, default=-9.0)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.kd_epochs < 0 or args.clean_epochs < 0 or args.kd_epochs + args.clean_epochs != 20:
        p.error("kd-epochs + clean-epochs must equal 20")
    if args.batch_size < 1 or args.chunk_frames < 1:
        p.error("batch-size and chunk-frames must be positive")
    if args.absolute_floor_db >= 0.0:
        p.error("absolute-floor-db must be negative")
    if not args.self_test and args.manifest is None:
        p.error("formal training requires --manifest")
    return args


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test(args)
    else:
        train(args)


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    from student.model import ResidualStudent  # noqa: E402
    main()
