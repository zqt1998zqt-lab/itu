#!/usr/bin/env python3
"""Train a local, experimental Complex_PF model from raw Speex-MDF residuals.

This is an offline experiment for the ``computer_pf_cpu`` route.  It does not
load an upstream checkpoint, does not use the CPU fused/NS output as training
target, and does not establish formal ITU compliance or general speech quality.

The data path is deliberately explicit:

1. Read real speech only from the two bounded LibriSpeech source directories.
2. Synthesize near/far/noise scenes with a deterministic seed.
3. Run the new native first-pass export with raw Speex MDF only.
4. Align clean near speech to ``linear_aec_delay_samples``.
5. Train Complex_PF with waveform and complex-spectral losses.
6. Reload both checkpoints with ``weights_only=True`` and evaluate on CPU.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16_000
CASE_SECONDS = 3.0
CASE_SAMPLES = int(SAMPLE_RATE * CASE_SECONDS)
SCHEMA = "computer-pf-training-v1"
VARIANT = "complex_pf_local_offline_v1"
TRAINING_STATUS = "locally_trained_experimental"
MODEL_SOURCE = PROJECT / "server" / "complex_pf_model.py"
SPEECH_ROOTS = (
    PROJECT / "datasets" / "itu-eval-v4-curated" / "sources" / "speech",
    PROJECT / "datasets" / "itu-eval-v1" / "sources" / "speech",
)
SCENARIOS = ("near_only", "far_only", "double_talk", "transition")
MIN_FREE_BYTES = 100 * 1024**3
MIN_MEM_AVAILABLE_BYTES = 5 * 1024**3
MAX_FILES_PER_ROOT = 256
MAX_SOURCE_FILES = 512


class ResourceSafetyStop(RuntimeError):
    """The mandatory disk or memory floor was reached."""


@dataclass(frozen=True)
class SourceFile:
    path: str
    speaker: str
    sha256: str
    dataset_root: str


@dataclass(frozen=True)
class CaseSpec:
    split: str
    index: int
    case_id: str
    scenario: str
    near_source: SourceFile
    far_source: SourceFile
    second_near_source: SourceFile | None = None


@dataclass
class PreparedCase:
    spec: CaseSpec
    residual: Any
    reference: Any
    target: Any
    metadata: dict[str, Any]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _existing_ancestor(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            raise OSError(f"no existing ancestor for {path}")
        candidate = candidate.parent
    return candidate


def _mem_available_bytes() -> int:
    with Path("/proc/meminfo").open(encoding="ascii") as stream:
        for line in stream:
            if line.startswith("MemAvailable:"):
                fields = line.split()
                if len(fields) >= 2:
                    return int(fields[1]) * 1024
                break
    raise OSError("MemAvailable is missing from /proc/meminfo")


def _resource_snapshot(path: Path) -> dict[str, int]:
    usage = shutil.disk_usage(_existing_ancestor(path))
    return {
        "disk_free_bytes": int(usage.free),
        "mem_available_bytes": _mem_available_bytes(),
    }


def _resource_preflight(path: Path) -> dict[str, int]:
    snapshot = _resource_snapshot(path)
    if snapshot["disk_free_bytes"] < MIN_FREE_BYTES:
        raise ResourceSafetyStop(
            "disk free is below the 100 GiB floor: "
            f"{snapshot['disk_free_bytes']} bytes"
        )
    if snapshot["mem_available_bytes"] < MIN_MEM_AVAILABLE_BYTES:
        raise ResourceSafetyStop(
            "MemAvailable is below the 5 GiB floor: "
            f"{snapshot['mem_available_bytes']} bytes"
        )
    return snapshot


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_json_once(path: Path, value: Any) -> None:
    _ensure_parent(path)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _write_text_once(path: Path, value: str) -> None:
    _ensure_parent(path)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value)


def _append_event(output: Path, event: dict[str, Any]) -> None:
    with (output / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")


def _write_wav_once(path: Path, samples: Any, sample_rate: int = SAMPLE_RATE) -> None:
    import numpy as np
    import soundfile as sf

    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    _ensure_parent(path)
    data = np.asarray(samples, dtype=np.float32).reshape(-1)
    if not np.isfinite(data).all():
        raise ValueError(f"non-finite audio for {path}")
    sf.write(path, data, sample_rate, subtype="PCM_16")


def _save_npy_once(path: Path, value: Any) -> None:
    import numpy as np

    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    _ensure_parent(path)
    with path.open("xb") as stream:
        np.save(stream, value, allow_pickle=False)


def _read_wav(path: Path) -> tuple[Any, int, int]:
    import numpy as np
    import soundfile as sf

    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if data.shape[1] < 1:
        raise ValueError(f"empty channel layout: {path}")
    mono = data.mean(axis=1, dtype=np.float32)
    if not np.isfinite(mono).all():
        raise ValueError(f"non-finite audio: {path}")
    return mono, int(sample_rate), int(data.shape[1])


def _resample(mono: Any, source_rate: int, target_rate: int = SAMPLE_RATE) -> Any:
    import numpy as np
    from scipy.signal import resample_poly

    if source_rate == target_rate:
        return np.asarray(mono, dtype=np.float32)
    divisor = math.gcd(source_rate, target_rate)
    return resample_poly(
        np.asarray(mono, dtype=np.float64),
        target_rate // divisor,
        source_rate // divisor,
    ).astype(np.float32)


def _fit_three_seconds(mono: Any, rng: random.Random) -> tuple[Any, int]:
    import numpy as np

    values = np.asarray(mono, dtype=np.float32)
    if values.size == 0:
        raise ValueError("empty speech source")
    if values.size >= CASE_SAMPLES:
        start = rng.randint(0, values.size - CASE_SAMPLES)
        return values[start : start + CASE_SAMPLES].copy(), start
    repeats = int(math.ceil(CASE_SAMPLES / values.size))
    tiled = np.tile(values, repeats)[:CASE_SAMPLES].copy()
    fade = min(SAMPLE_RATE // 100, values.size // 4)
    if fade > 1:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        tiled[:fade] *= ramp
        tiled[-fade:] *= ramp[::-1]
    return tiled, 0


def _load_speech_window(source: SourceFile, rng: random.Random) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    mono, sample_rate, channels = _read_wav(Path(source.path))
    mono = _resample(mono, sample_rate)
    window, start = _fit_three_seconds(mono, rng)
    rms = float(np.sqrt(np.mean(np.square(window, dtype=np.float64)) + 1e-12))
    if rms < 1e-6:
        raise ValueError(f"speech source is silent: {source.path}")
    peak = float(np.max(np.abs(window)))
    return window, {
        "path": source.path,
        "sha256": source.sha256,
        "speaker": source.speaker,
        "source_sample_rate": sample_rate,
        "source_channels": channels,
        "resampled_16k": sample_rate != SAMPLE_RATE,
        "crop_start_sample_16k": start,
        "source_window_rms": rms,
        "source_window_peak": peak,
    }


def _normalize_peak(samples: Any, peak: float) -> Any:
    import numpy as np

    values = np.asarray(samples, dtype=np.float32)
    current = float(np.max(np.abs(values)))
    if current <= 1e-12:
        return values
    return (values * (peak / current)).astype(np.float32)


def _speaker_from_name(path: Path) -> str:
    match = re.match(r"^(\d+)-", path.name)
    return match.group(1) if match else path.stem.split("-", 1)[0]


def _bounded_flac_paths(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    paths: list[Path] = []
    with os.scandir(root) as entries:
        for entry in entries:
            if len(paths) >= MAX_FILES_PER_ROOT:
                break
            if entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(".flac"):
                paths.append(Path(entry.path))
    return sorted(paths, key=lambda item: item.name)


def _enumerate_sources() -> tuple[list[SourceFile], dict[str, Any]]:
    sources: list[SourceFile] = []
    seen_hashes: dict[str, str] = {}
    duplicates: list[dict[str, str]] = []
    root_records: list[dict[str, Any]] = []
    for root in SPEECH_ROOTS:
        paths = _bounded_flac_paths(root)
        accepted = 0
        for path in paths:
            if len(sources) >= MAX_SOURCE_FILES:
                break
            digest = _sha256_file(path)
            if digest in seen_hashes:
                duplicates.append(
                    {"path": str(path), "duplicate_of": seen_hashes[digest], "sha256": digest}
                )
                continue
            seen_hashes[digest] = str(path)
            sources.append(
                SourceFile(
                    path=str(path),
                    speaker=_speaker_from_name(path),
                    sha256=digest,
                    dataset_root=str(root),
                )
            )
            accepted += 1
        root_records.append(
            {
                "root": str(root),
                "bounded_file_limit": MAX_FILES_PER_ROOT,
                "files_considered": len(paths),
                "unique_sources_accepted": accepted,
            }
        )
    if not sources:
        raise FileNotFoundError("no FLAC speech sources found in the approved roots")
    manifest = {
        "roots": root_records,
        "unique_source_count": len(sources),
        "speaker_count": len({source.speaker for source in sources}),
        "duplicate_file_count": len(duplicates),
        "duplicates": duplicates,
        "source_hashes": sorted(source.sha256 for source in sources),
    }
    return sources, manifest


def _split_sources(
    sources: Sequence[SourceFile], seed: int
) -> tuple[dict[str, list[SourceFile]], dict[str, Any]]:
    rng = random.Random(f"{seed}:source-split")
    by_speaker: dict[str, list[SourceFile]] = {}
    for source in sources:
        by_speaker.setdefault(source.speaker, []).append(source)
    speakers = sorted(by_speaker)
    rng.shuffle(speakers)

    if len(speakers) >= 9:
        test_speakers = speakers[:3]
        validation_speakers = speakers[3:6]
        train_speakers = speakers[6:]
        splits = {
            "train": [source for name in train_speakers for source in by_speaker[name]],
            "validation": [
                source for name in validation_speakers for source in by_speaker[name]
            ],
            "test": [source for name in test_speakers for source in by_speaker[name]],
        }
        strategy = "speaker_hash_disjoint"
        assigned_speakers = {
            "train": sorted(train_speakers),
            "validation": sorted(validation_speakers),
            "test": sorted(test_speakers),
        }
    else:
        shuffled = list(sources)
        rng.shuffle(shuffled)
        if len(shuffled) < 12:
            raise ValueError("at least 12 unique speech files are required for fallback splitting")
        test_count = max(4, len(shuffled) // 5)
        validation_count = max(4, len(shuffled) // 5)
        splits = {
            "test": shuffled[:test_count],
            "validation": shuffled[test_count : test_count + validation_count],
            "train": shuffled[test_count + validation_count :],
        }
        strategy = "file_hash_fallback"
        assigned_speakers = {
            name: sorted({source.speaker for source in rows}) for name, rows in splits.items()
        }

    for name, rows in splits.items():
        if len(rows) < 3:
            raise ValueError(f"{name} split has fewer than three unique source files")
        if len({source.sha256 for source in rows}) != len(rows):
            raise ValueError(f"{name} split contains duplicate source hashes")
    hashes = {
        name: {source.sha256 for source in rows} for name, rows in splits.items()
    }
    hash_disjoint = all(
        not (hashes[left] & hashes[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    )
    speaker_disjoint = all(
        not (set(assigned_speakers[left]) & set(assigned_speakers[right]))
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    )
    if not hash_disjoint:
        raise ValueError("source file hashes are not disjoint across train/validation/test")
    if strategy == "speaker_hash_disjoint" and not speaker_disjoint:
        raise ValueError("speaker groups are not disjoint across train/validation/test")
    manifest = {
        "strategy": strategy,
        "seed": seed,
        "speakers": assigned_speakers,
        "speaker_disjoint": speaker_disjoint,
        "file_sha256_disjoint": hash_disjoint,
        "files": {
            name: [
                {
                    "path": source.path,
                    "speaker": source.speaker,
                    "sha256": source.sha256,
                    "dataset_root": source.dataset_root,
                }
                for source in sorted(rows, key=lambda item: item.path)
            ]
            for name, rows in splits.items()
        },
        "file_counts": {name: len(rows) for name, rows in splits.items()},
        "speaker_counts": {
            name: len(set(assigned_speakers[name])) for name in ("train", "validation", "test")
        },
    }
    manifest["manifest_sha256"] = _sha256_json(manifest)
    return splits, manifest


def _choose_case_sources(
    split_sources: Sequence[SourceFile], scenario: str, rng: random.Random
) -> tuple[SourceFile, SourceFile, SourceFile | None]:
    first = rng.choice(list(split_sources))
    far_candidates = [
        source
        for source in split_sources
        if source.sha256 != first.sha256 and source.speaker != first.speaker
    ]
    if not far_candidates:
        far_candidates = [source for source in split_sources if source.sha256 != first.sha256]
    if not far_candidates:
        raise ValueError("cannot choose distinct near/far files in split")
    far = rng.choice(far_candidates)
    second = None
    if scenario == "double_talk":
        used = {first.sha256, far.sha256}
        candidates = [
            source
            for source in split_sources
            if source.sha256 not in used and source.speaker != first.speaker
        ]
        if not candidates:
            candidates = [source for source in split_sources if source.sha256 not in used]
        if not candidates:
            raise ValueError("double-talk case needs three distinct source files")
        second = rng.choice(candidates)
    return first, far, second


def _build_case_specs(
    splits: dict[str, list[SourceFile]], counts: dict[str, int], seed: int
) -> list[CaseSpec]:
    specs: list[CaseSpec] = []
    for split in ("train", "validation", "test"):
        rng = random.Random(f"{seed}:{split}:cases")
        cycle = list(SCENARIOS)
        rng.shuffle(cycle)
        for index in range(counts[split]):
            if index and index % len(SCENARIOS) == 0:
                rng.shuffle(cycle)
            scenario = cycle[index % len(SCENARIOS)]
            near, far, second = _choose_case_sources(splits[split], scenario, rng)
            used = {near.sha256, far.sha256}
            if second is not None:
                used.add(second.sha256)
            if len(used) != (3 if second is not None else 2):
                raise ValueError("case source files are not distinct")
            specs.append(
                CaseSpec(
                    split=split,
                    index=index,
                    case_id=f"{split}-{index:04d}-{scenario}",
                    scenario=scenario,
                    near_source=near,
                    far_source=far,
                    second_near_source=second,
                )
            )
    return specs


def _ramp(length: int, ascending: bool) -> Any:
    import numpy as np

    values = np.linspace(0.0, 1.0, max(length, 1), dtype=np.float32)
    return values if ascending else values[::-1].copy()


def _scenario_envelopes(scenario: str) -> tuple[Any, Any, dict[str, Any]]:
    import numpy as np

    near = np.ones(CASE_SAMPLES, dtype=np.float32)
    far = np.ones(CASE_SAMPLES, dtype=np.float32)
    if scenario == "near_only":
        near[:] = 1.0
        far[:] = 1.0
    elif scenario == "far_only":
        near[:] = 0.0
        far[:] = 1.0
    elif scenario == "double_talk":
        near[:] = 1.0
        far[:] = 1.0
    elif scenario == "transition":
        ramp_length = int(0.18 * SAMPLE_RATE)
        near[:] = 0.0
        far[:] = 0.0
        near[: int(0.75 * SAMPLE_RATE)] = 1.0
        near[int(0.75 * SAMPLE_RATE) : int(0.75 * SAMPLE_RATE) + ramp_length] = _ramp(
            ramp_length, False
        )
        far[int(0.90 * SAMPLE_RATE) : int(0.90 * SAMPLE_RATE) + ramp_length] = _ramp(
            ramp_length, True
        )
        far[int(0.90 * SAMPLE_RATE) + ramp_length :] = 1.0
    else:
        raise ValueError(f"unknown scenario: {scenario}")
    fade = int(0.01 * SAMPLE_RATE)
    near[:fade] *= _ramp(fade, True)
    near[-fade:] *= _ramp(fade, False)
    far[:fade] *= _ramp(fade, True)
    far[-fade:] *= _ramp(fade, False)
    return near, far, {
        "scenario": scenario,
        "near_active_fraction": float(np.mean(near > 1e-4)),
        "far_active_fraction": float(np.mean(far > 1e-4)),
    }


def _random_rir(rng: random.Random) -> Any:
    import numpy as np

    direct_delay = rng.randint(0, int(0.010 * SAMPLE_RATE))
    tail = rng.randint(int(0.045 * SAMPLE_RATE), int(0.140 * SAMPLE_RATE))
    length = direct_delay + tail
    rir = np.zeros(length, dtype=np.float32)
    rir[direct_delay] = rng.uniform(0.65, 1.0)
    reflection_count = rng.randint(12, 40)
    decay_seconds = rng.uniform(0.035, 0.110)
    for _ in range(reflection_count):
        position = rng.randint(direct_delay + 1, length - 1)
        offset = (position - direct_delay) / SAMPLE_RATE
        amplitude = rng.uniform(-0.7, 0.7) * math.exp(-offset / decay_seconds)
        rir[position] += amplitude
    energy = float(np.sqrt(np.mean(np.square(rir, dtype=np.float64))) + 1e-12)
    rir = (rir / energy * 0.35).astype(np.float32)
    return rir


def _echo_nonlinearity(samples: Any, kind: str, rng: random.Random) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    values = np.asarray(samples, dtype=np.float32)
    peak = float(np.max(np.abs(values)))
    if peak <= 1e-12:
        return values, {"kind": kind, "peak_before": peak, "peak_after": peak}
    normalized = values / peak
    if kind == "clip":
        drive = rng.uniform(1.15, 3.0)
        threshold = rng.uniform(0.35, 0.82)
        shaped = np.clip(normalized * drive, -threshold, threshold)
        parameters = {"drive": drive, "threshold": threshold}
    elif kind == "tanh":
        drive = rng.uniform(1.0, 3.2)
        shaped = np.tanh(normalized * drive)
        parameters = {"drive": drive}
    else:
        raise ValueError(f"unknown echo nonlinearity: {kind}")
    shaped_peak = float(np.max(np.abs(shaped)))
    shaped = (shaped / max(shaped_peak, 1e-12)).astype(np.float32)
    return shaped, {
        "kind": kind,
        "parameters": parameters,
        "peak_before": peak,
        "peak_after": float(np.max(np.abs(shaped))),
    }


def _colored_noise(rng: random.Random, length: int) -> Any:
    import numpy as np
    from scipy.signal import lfilter

    generator = np.random.default_rng(rng.getrandbits(64))
    white = generator.standard_normal(length).astype(np.float64)
    low = lfilter([0.05], [1.0, -0.95], white)
    broadband = lfilter([1.0, -0.97], [1.0], white)
    mixed = 0.55 * low + 0.45 * broadband
    mixed -= np.mean(mixed)
    rms = float(np.sqrt(np.mean(np.square(mixed)) + 1e-12))
    return (mixed / rms).astype(np.float32)


def _rms(samples: Any) -> float:
    import numpy as np

    return float(np.sqrt(np.mean(np.square(np.asarray(samples, dtype=np.float64))) + 1e-12))


def _synthesize_case(
    spec: CaseSpec, output: Path, seed: int
) -> tuple[Path, dict[str, Any]]:
    import numpy as np
    from scipy.signal import fftconvolve

    case_dir = output / "cases" / spec.split / spec.case_id
    case_dir.mkdir(parents=True, exist_ok=False)
    _resource_preflight(output)
    rng = random.Random(f"{seed}:synthesis:{spec.split}:{spec.index}")
    near_a, near_a_meta = _load_speech_window(spec.near_source, rng)
    far_clean, far_meta = _load_speech_window(spec.far_source, rng)
    near_a = _normalize_peak(near_a, rng.uniform(0.30, 0.72))
    near_mix = near_a
    near_sources = [near_a_meta]
    if spec.second_near_source is not None:
        near_b, near_b_meta = _load_speech_window(spec.second_near_source, rng)
        near_b = _normalize_peak(near_b, rng.uniform(0.20, 0.62))
        near_mix = near_a + near_b * rng.uniform(0.45, 0.95)
        near_mix = _normalize_peak(near_mix, rng.uniform(0.35, 0.78))
        near_sources.append(near_b_meta)
    near_envelope, far_envelope, envelope_meta = _scenario_envelopes(spec.scenario)
    target = (near_mix * near_envelope).astype(np.float32)
    far_playback = _normalize_peak(far_clean, rng.uniform(0.20, 0.72)) * far_envelope
    far_playback = far_playback.astype(np.float32)

    rir = _random_rir(rng)
    dry_echo = fftconvolve(far_playback, rir, mode="full")[:CASE_SAMPLES].astype(np.float32)
    nonlinearity_kind = "clip" if rng.random() < 0.5 else "tanh"
    shaped_echo, nonlinearity_meta = _echo_nonlinearity(
        dry_echo, nonlinearity_kind, rng
    )
    target_rms = _rms(target)
    if spec.scenario == "near_only":
        echo = np.zeros(CASE_SAMPLES, dtype=np.float32)
        echo_relative_db = None
    else:
        echo_relative_db = rng.uniform(-8.0, 4.0)
        anchor_rms = target_rms if target_rms > 1e-5 else rng.uniform(0.025, 0.075)
        desired_echo_rms = anchor_rms * 10.0 ** (echo_relative_db / 20.0)
        shaped_rms = _rms(shaped_echo)
        echo = (shaped_echo * (desired_echo_rms / max(shaped_rms, 1e-12))).astype(
            np.float32
        )

    noise_snr_db = rng.uniform(18.0, 35.0)
    mixture_anchor = _rms(target + echo)
    if mixture_anchor < 1e-5:
        mixture_anchor = 0.04
    noise = _colored_noise(rng, CASE_SAMPLES)
    noise = (
        noise * (mixture_anchor / 10.0 ** (noise_snr_db / 20.0))
    ).astype(np.float32)
    microphone = (target + echo + noise).astype(np.float32)
    microphone_clip = None
    if rng.random() < 0.25:
        threshold = rng.uniform(0.82, 1.15)
        microphone_clip = {
            "threshold": threshold,
            "clipped_fraction": float(np.mean(np.abs(microphone) > threshold)),
        }
        microphone = np.clip(microphone, -threshold, threshold).astype(np.float32)

    _write_wav_once(case_dir / "clean-near.wav", target)
    _write_wav_once(case_dir / "far.wav", far_playback)
    _write_wav_once(case_dir / "noise.wav", noise)
    _write_wav_once(case_dir / "mic.wav", microphone)
    _save_npy_once(case_dir / "echo-rir.npy", rir)
    metadata = {
        "case_id": spec.case_id,
        "split": spec.split,
        "scenario": spec.scenario,
        "sample_rate": SAMPLE_RATE,
        "samples": CASE_SAMPLES,
        "seed": seed,
        "near_sources": near_sources,
        "far_source": far_meta,
        "near_far_files_distinct": spec.near_source.sha256 != spec.far_source.sha256,
        "all_case_source_files_distinct": len(
            {
                source["sha256"]
                for source in [*near_sources, far_meta]
            }
        )
        == len([*near_sources, far_meta]),
        "envelopes": envelope_meta,
        "rir": {
            "file": str(case_dir / "echo-rir.npy"),
            "samples": int(len(rir)),
            "direct_delay_samples": int(np.argmax(np.abs(rir))),
            "energy_rms": _rms(rir),
        },
        "echo": {
            "relative_to_near_db": echo_relative_db,
            "nonlinearity": nonlinearity_meta,
            "rms": _rms(echo),
        },
        "noise": {
            "kind": "deterministic_colored_gaussian",
            "snr_db": noise_snr_db,
            "rms": _rms(noise),
        },
        "microphone_clip": microphone_clip,
        "levels": {
            "clean_near_rms": target_rms,
            "far_playback_rms": _rms(far_playback),
            "microphone_rms": _rms(microphone),
            "microphone_peak": float(np.max(np.abs(microphone))),
        },
    }
    _write_json_once(case_dir / "synthesis.json", metadata)
    return case_dir, metadata


def _nested_int(report: dict[str, Any], key: str) -> int:
    value = report.get(key)
    if value is None and isinstance(report.get("engine"), dict):
        value = report["engine"].get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"first-pass report has invalid {key}: {value!r}")
    return value


def _nested_bool(report: dict[str, Any], key: str) -> bool | None:
    value = report.get(key)
    if value is None and isinstance(report.get("engine"), dict):
        value = report["engine"].get(key)
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    return None


def _fit_length(samples: Any, length: int) -> Any:
    import numpy as np

    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    if values.size >= length:
        return values[:length].copy()
    return np.pad(values, (0, length - values.size)).astype(np.float32)


def _align_target(clean_near: Any, delay_samples: int, length: int) -> Any:
    import numpy as np

    clean = np.asarray(clean_near, dtype=np.float32).reshape(-1)
    if delay_samples >= length:
        raise ValueError(
            f"linear AEC delay {delay_samples} is not shorter than exported length {length}"
        )
    aligned = np.zeros(length, dtype=np.float32)
    count = min(clean.size, length - delay_samples)
    aligned[delay_samples : delay_samples + count] = clean[:count]
    return aligned


def _run_native_first_pass(
    native_cli: Path,
    case_dir: Path,
    output: Path,
    timeout_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    _resource_preflight(output)
    native_dir = case_dir / "native"
    native_dir.mkdir(parents=True, exist_ok=False)
    prefix = native_dir / "firstpass"
    command = [
        str(native_cli),
        "--profile",
        "vcp",
        "--aec-backend",
        "speex",
        "--no-speex-coherence-residual",
        "--no-speex-multirender-context",
        "--no-bf",
        "--mic",
        str(case_dir / "mic.wav"),
        "--echo",
        str(case_dir / "far.wav"),
        "--out",
        str(prefix),
        "--complex-pf-export-inputs",
    ]
    _write_json_once(
        native_dir / "command.json",
        {
            "argv": command,
            "cwd": str(PROJECT),
            "native_cli_sha256": _sha256_file(native_cli),
            "profile_contract": "raw_speex_mdf_no_cpu_fusion_no_bf",
        },
    )
    try:
        run = subprocess.run(
            command,
            cwd=PROJECT,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        _write_text_once(native_dir / "stdout.log", stdout)
        _write_text_once(native_dir / "stderr.log", stderr)
        _write_json_once(
            native_dir / "exit.json",
            {
                "returncode": None,
                "timeout_seconds": timeout_seconds,
                "status": "timeout",
            },
        )
        raise RuntimeError(f"native first-pass timed out for {case_dir.name}") from error
    _write_text_once(native_dir / "stdout.log", run.stdout)
    _write_text_once(native_dir / "stderr.log", run.stderr)
    _write_json_once(
        native_dir / "exit.json",
        {"returncode": run.returncode, "status": "completed" if run.returncode == 0 else "failed"},
    )
    if run.returncode != 0:
        tail = (run.stderr or run.stdout or "").strip().splitlines()
        raise RuntimeError(
            f"native first-pass failed for {case_dir.name}: "
            f"exit {run.returncode}: {tail[-1] if tail else 'no output'}"
        )
    _resource_preflight(output)

    report_path = prefix.with_suffix(".json")
    residual_path = prefix.with_suffix(".pf-residual.wav")
    reference_path = prefix.with_suffix(".pf-reference.wav")
    for required in (report_path, residual_path, reference_path):
        if not required.is_file():
            raise FileNotFoundError(f"native export did not create {required}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    backend = report.get("aec_backend_effective") or report.get("aec_backend")
    if backend != "speex_mdf":
        raise ValueError(f"native export is not raw Speex MDF: aec_backend={backend!r}")
    linear_available = _nested_bool(report, "linear_aec_available")
    if linear_available is not True:
        raise ValueError("native report says linear AEC tap is unavailable")
    delay = _nested_int(report, "linear_aec_delay_samples")

    residual, residual_rate, residual_channels = _read_wav(residual_path)
    reference, reference_rate, reference_channels = _read_wav(reference_path)
    clean_near, clean_rate, clean_channels = _read_wav(case_dir / "clean-near.wav")
    if residual_rate != SAMPLE_RATE or reference_rate != SAMPLE_RATE:
        raise ValueError("PF residual/reference are not 16 kHz")
    if clean_rate != SAMPLE_RATE:
        raise ValueError("clean near target is not 16 kHz")
    if residual_channels != 1 or reference_channels != 1 or clean_channels != 1:
        raise ValueError("PF export and target must be mono")
    target = _align_target(clean_near, delay, int(residual.size))
    reference = _fit_length(reference, int(residual.size))
    if not (
        np.isfinite(residual).all()
        and np.isfinite(reference).all()
        and np.isfinite(target).all()
    ):
        raise ValueError("aligned training tensors contain non-finite samples")
    aligned_path = native_dir / "aligned-target.wav"
    _write_wav_once(aligned_path, target)
    alignment = {
        "linear_aec_delay_samples": delay,
        "alignment_rule": "left-pad clean near by linear_aec_delay_samples; crop/pad to residual length",
        "clean_near_samples": int(clean_near.size),
        "residual_samples": int(residual.size),
        "reference_samples": int(reference.size),
        "aligned_target_samples": int(target.size),
        "residual_sha256": _sha256_file(residual_path),
        "reference_sha256": _sha256_file(reference_path),
        "aligned_target_sha256": _sha256_file(aligned_path),
    }
    _write_json_once(native_dir / "alignment.json", alignment)
    metadata = {
        "native_cli": str(native_cli),
        "native_cli_sha256": _sha256_file(native_cli),
        "firstpass_report": str(report_path),
        "firstpass_report_sha256": _sha256_file(report_path),
        "residual_path": str(residual_path),
        "reference_path": str(reference_path),
        "aligned_target_path": str(aligned_path),
        "alignment": alignment,
        "aec_backend": backend,
        "cpu_fused_or_ns_output_used": False,
    }
    return metadata, {"residual": residual, "reference": reference, "target": target}


def _state_dict_cpu(model: Any) -> dict[str, Any]:
    return {
        name: tensor.detach().to(device="cpu").contiguous().clone()
        for name, tensor in model.state_dict().items()
    }


def _state_dict_sha256(state_dict: dict[str, Any]) -> str:
    import torch

    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _checkpoint_payload(
    state_dict: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {"state_dict": state_dict, "metadata": metadata}


def _save_checkpoint_once(path: Path, payload: dict[str, Any]) -> None:
    import torch

    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    _ensure_parent(path)
    torch.save(payload, path)


def _load_checkpoint_weights_only(path: Path, *, expected_phase: str) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "state_dict" not in payload or "metadata" not in payload:
        raise ValueError(f"invalid checkpoint structure: {path}")
    metadata = payload["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError(f"invalid checkpoint metadata: {path}")
    if metadata.get("schema") != SCHEMA or metadata.get("variant") != VARIANT:
        raise ValueError(f"checkpoint schema/variant mismatch: {path}")
    if metadata.get("phase") != expected_phase:
        raise ValueError(f"checkpoint phase mismatch: {path}")
    return payload


def _pairs_to_spectrum(prediction: Any, batch_size: int, frame_count: int) -> Any:
    import torch
    import torch.nn.functional as F
    from server.complex_pf_model import FFT_SIZE

    bins = FFT_SIZE // 2 + 1
    pairs = prediction.reshape(batch_size, frame_count, bins, 2)
    first, second = pairs[..., 0], pairs[..., 1]
    spectrum = F.pad(first, (0, 0, 0, 1)) + F.pad(second, (0, 0, 1, 0))
    denominator = torch.full(
        (frame_count,), 2.0, dtype=spectrum.dtype, device=spectrum.device
    )
    denominator[0] = denominator[-1] = 1.0
    return (spectrum / denominator.view(1, frame_count, 1)).transpose(1, 2)


def _collate_waveforms(arrays: Sequence[Any], device: Any) -> tuple[Any, Any]:
    import numpy as np
    import torch

    lengths = [int(np.asarray(array).size) for array in arrays]
    maximum = max(lengths)
    padded = np.zeros((len(arrays), maximum), dtype=np.float32)
    for row, array in enumerate(arrays):
        values = np.asarray(array, dtype=np.float32).reshape(-1)
        padded[row, : values.size] = values
    return torch.from_numpy(padded).to(device), torch.tensor(lengths, device=device)


def _training_loss(
    model: Any,
    residual: Any,
    reference: Any,
    target: Any,
    lengths: Any,
    spectral_weight: float,
) -> tuple[Any, dict[str, float]]:
    import torch
    import torch.nn.functional as F
    from server.complex_pf_model import (
        FFT_SIZE,
        HOP_SIZE,
        reconstruct_pairs,
        spectral_pairs,
    )

    batch_size = residual.shape[0]
    pairs, scale, frame_count = spectral_pairs(residual, reference)
    out_r, out_i, _ = model(
        pairs.reshape(-1, 2, FFT_SIZE // 2 + 1, 2).real,
        pairs.reshape(-1, 2, FFT_SIZE // 2 + 1, 2).imag,
        batch_size=batch_size,
    )
    prediction = torch.stack((out_r[:, 0], out_i[:, 0]), dim=-1)
    predicted_spectrum = _pairs_to_spectrum(
        prediction, batch_size=batch_size, frame_count=frame_count
    )
    predicted_wave = reconstruct_pairs(
        prediction,
        batch_size=batch_size,
        frame_count=frame_count,
        samples=residual.shape[-1],
    ) * scale
    target_spectrum = torch.stft(
        target / scale,
        FFT_SIZE,
        HOP_SIZE,
        window=torch.hann_window(FFT_SIZE, device=target.device),
        center=True,
        pad_mode="constant",
        return_complex=True,
    )
    if target_spectrum.shape[-1] != frame_count:
        raise ValueError("target and residual STFT frame counts differ")
    sample_mask = (
        torch.arange(residual.shape[-1], device=residual.device).view(1, -1)
        < lengths.view(-1, 1)
    )
    error_power = (predicted_wave - target).square() * sample_mask
    target_power = target.square() * sample_mask
    waveform_relative_mse = (
        error_power.sum(dim=1) / target_power.sum(dim=1).clamp_min(1e-8)
    )
    spectral_l1 = (predicted_spectrum - target_spectrum).abs().mean(dim=(1, 2))
    per_sample = waveform_relative_mse + spectral_weight * spectral_l1
    loss = per_sample.mean()
    return loss, {
        "loss": float(loss.detach().item()),
        "waveform_relative_mse": float(waveform_relative_mse.detach().mean().item()),
        "complex_spectral_l1": float(spectral_l1.detach().mean().item()),
    }


def _train_model(
    model: Any,
    cases: Sequence[PreparedCase],
    *,
    steps: int,
    seed: int,
    device: Any,
    batch_size: int,
    learning_rate: float,
    spectral_weight: float,
) -> dict[str, Any]:
    import torch

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    rng = random.Random(f"{seed}:training")
    train_indices = list(range(len(cases)))
    order: list[int] = []
    step_records: list[dict[str, Any]] = []
    started = time.monotonic()
    model.train()
    for step in range(1, steps + 1):
        if len(order) < batch_size:
            next_order = train_indices.copy()
            rng.shuffle(next_order)
            order.extend(next_order)
        indices = [order.pop(0) for _ in range(batch_size)]
        batch = [cases[index] for index in indices]
        residual, lengths = _collate_waveforms(
            [case.residual for case in batch], device
        )
        reference, _ = _collate_waveforms([case.reference for case in batch], device)
        target, _ = _collate_waveforms([case.target for case in batch], device)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = _training_loss(
            model,
            residual,
            reference,
            target,
            lengths,
            spectral_weight,
        )
        if not math.isfinite(metrics["loss"]):
            raise FloatingPointError(f"non-finite training loss at step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        record = {
            "step": step,
            "case_ids": [case.spec.case_id for case in batch],
            **metrics,
            "gradient_norm": float(gradient_norm.detach().item()),
            "learning_rate": float(scheduler.get_last_lr()[0]),
        }
        step_records.append(record)
        if step == 1 or step % 25 == 0 or step == steps:
            print(json.dumps(record, ensure_ascii=False), flush=True)
    elapsed = time.monotonic() - started
    segments = []
    segment_size = 10
    for start in range(0, len(step_records), segment_size):
        rows = step_records[start : start + segment_size]
        segments.append(
            {
                "start_step": rows[0]["step"],
                "end_step": rows[-1]["step"],
                "mean_loss": sum(row["loss"] for row in rows) / len(rows),
                "mean_waveform_relative_mse": sum(
                    row["waveform_relative_mse"] for row in rows
                )
                / len(rows),
                "mean_complex_spectral_l1": sum(
                    row["complex_spectral_l1"] for row in rows
                )
                / len(rows),
            }
        )
    return {
        "elapsed_seconds": elapsed,
        "step_records": step_records,
        "loss_segments": segments,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": 1e-4,
            "scheduler": "CosineAnnealingLR",
            "gradient_clip_norm": 1.0,
            "batch_size": batch_size,
            "spectral_weight": spectral_weight,
        },
    }


def _waveform_metrics(prediction: Any, target: Any) -> dict[str, float]:
    import numpy as np

    predicted = np.asarray(prediction, dtype=np.float64).reshape(-1)
    expected = np.asarray(target, dtype=np.float64).reshape(-1)
    if predicted.size != expected.size:
        raise ValueError("prediction/target length mismatch")
    error = predicted - expected
    mse = float(np.mean(np.square(error)))
    target_power = float(np.sum(np.square(expected)))
    nmse = float(np.sum(np.square(error)) / max(target_power, 1e-12))
    numerator = float(np.dot(predicted, expected))
    denominator = float(
        math.sqrt(np.dot(predicted, predicted) * np.dot(expected, expected))
    )
    correlation = numerator / denominator if denominator > 1e-12 else None
    return {
        "waveform_mse": mse,
        "waveform_nmse": nmse,
        "target_rms": float(np.sqrt(np.mean(np.square(expected)) + 1e-12)),
        "prediction_rms": float(np.sqrt(np.mean(np.square(predicted)) + 1e-12)),
        "correlation": correlation,
        "error_sum_square": float(np.sum(np.square(error))),
        "target_sum_square": target_power,
    }


def _waveform_spectral_l1(prediction: Any, target: Any, residual: Any) -> float:
    import numpy as np
    import torch
    from server.complex_pf_model import FFT_SIZE, HOP_SIZE

    scale = float(np.max(np.abs(residual)))
    scale = max(scale, 1e-5)
    predicted = torch.from_numpy(
        np.asarray(prediction, dtype=np.float32).reshape(1, -1)
    )
    expected = torch.from_numpy(np.asarray(target, dtype=np.float32).reshape(1, -1))
    window = torch.hann_window(FFT_SIZE)
    predicted_spectrum = torch.stft(
        predicted / scale,
        FFT_SIZE,
        HOP_SIZE,
        window=window,
        center=True,
        pad_mode="constant",
        return_complex=True,
    )
    target_spectrum = torch.stft(
        expected / scale,
        FFT_SIZE,
        HOP_SIZE,
        window=window,
        center=True,
        pad_mode="constant",
        return_complex=True,
    )
    return float((predicted_spectrum - target_spectrum).abs().mean().item())


def _aggregate_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total_error = sum(row["error_sum_square"] for row in rows)
    total_target = sum(row["target_sum_square"] for row in rows)
    return {
        "case_count": len(rows),
        "waveform_mse_mean": sum(row["waveform_mse"] for row in rows) / len(rows),
        "waveform_nmse_global": total_error / max(total_target, 1e-12),
        "waveform_nmse_case_mean": sum(row["waveform_nmse"] for row in rows) / len(rows),
        "complex_spectral_l1_mean": sum(row["complex_spectral_l1"] for row in rows)
        / len(rows),
    }


def _evaluate_checkpoint_cpu(
    checkpoint: Path,
    cases: Sequence[PreparedCase],
    *,
    phase: str,
    prediction_dir: Path | None = None,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from server.complex_pf_model import ComplexPF, predict_waveform

    payload = _load_checkpoint_weights_only(checkpoint, expected_phase=phase)
    model = ComplexPF()
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device="cpu")
    model.eval()
    rows: list[dict[str, Any]] = []
    for case in cases:
        prediction, frame_count = predict_waveform(
            model, case.residual, case.reference
        )
        metrics = _waveform_metrics(prediction, case.target)
        metrics["complex_spectral_l1"] = _waveform_spectral_l1(
            prediction, case.target, case.residual
        )
        metrics["case_id"] = case.spec.case_id
        metrics["scenario"] = case.spec.scenario
        metrics["frame_count"] = frame_count
        rows.append(metrics)
        if prediction_dir is not None:
            target_dir = prediction_dir / case.spec.case_id
            target_dir.mkdir(parents=True, exist_ok=False)
            _write_wav_once(target_dir / f"{phase}.wav", prediction)
    return {"aggregate": _aggregate_metrics(rows), "cases": rows}


def _evaluate_mdf_baseline(cases: Sequence[PreparedCase]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        metrics = _waveform_metrics(case.residual, case.target)
        metrics["complex_spectral_l1"] = _waveform_spectral_l1(
            case.residual, case.target, case.residual
        )
        metrics["case_id"] = case.spec.case_id
        metrics["scenario"] = case.spec.scenario
        rows.append(metrics)
    return {"aggregate": _aggregate_metrics(rows), "cases": rows}


def _relative_improvement(baseline: float, candidate: float) -> float | None:
    if not math.isfinite(baseline) or abs(baseline) <= 1e-12:
        return None
    return (baseline - candidate) / baseline


def _resolve_device(value: str) -> Any:
    import torch

    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return device


def _configure_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--native-cli",
        required=True,
        type=Path,
        help="new itu_audio_cli binary that supports --complex-pf-export-inputs",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="new output directory; existing paths are rejected",
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--cases", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--spectral-weight", type=float, default=0.25)
    parser.add_argument("--native-timeout", type=float, default=180.0)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.cases < 1:
        raise ValueError("--cases must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.spectral_weight < 0:
        raise ValueError("--spectral-weight must be non-negative")
    if args.native_timeout <= 0:
        raise ValueError("--native-timeout must be positive")


def _run(args: argparse.Namespace) -> int:
    _validate_args(args)
    native_cli = args.native_cli.expanduser().resolve()
    if not native_cli.is_file():
        raise FileNotFoundError(f"native CLI not found: {native_cli}")
    if not os.access(native_cli, os.X_OK):
        raise PermissionError(f"native CLI is not executable: {native_cli}")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"output parent does not exist: {output.parent}")
    _resource_preflight(output)
    output.mkdir(mode=0o755, exist_ok=False)
    _append_event(output, {"time": _now(), "event": "output_created", "path": str(output)})

    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))
    import numpy as np
    import torch
    from server.complex_pf_model import ComplexPF

    _configure_seed(args.seed)
    device = _resolve_device(args.device)
    model = ComplexPF()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 1_856_578:
        raise ValueError(
            f"unexpected ComplexPF parameter count: {parameter_count}, expected 1856578"
        )
    initial_state = _state_dict_cpu(model)
    initial_weight_sha = _state_dict_sha256(initial_state)
    initial_checkpoint = output / "initial.pt"
    initial_metadata = {
        "schema": SCHEMA,
        "variant": VARIANT,
        "phase": "initial_random_control",
        "model_training_status": "random_initialization_negative_control",
        "official_pretrained_weights": False,
        "seed": args.seed,
        "parameter_count": parameter_count,
        "weight_sha256": initial_weight_sha,
        "model_source": str(MODEL_SOURCE),
        "model_source_sha256": _sha256_file(MODEL_SOURCE),
    }
    _save_checkpoint_once(
        initial_checkpoint,
        _checkpoint_payload(initial_state, initial_metadata),
    )
    initial_checkpoint_sha = _sha256_file(initial_checkpoint)
    _append_event(
        output,
        {
            "time": _now(),
            "event": "initial_checkpoint_saved",
            "path": str(initial_checkpoint),
            "sha256": initial_checkpoint_sha,
            "weight_sha256": initial_weight_sha,
        },
    )

    sources, source_manifest = _enumerate_sources()
    splits, split_manifest = _split_sources(sources, args.seed)
    counts = {"train": args.cases, "validation": 12, "test": 12}
    specs = _build_case_specs(splits, counts, args.seed)
    _write_json_once(
        output / "source-split.json",
        {
            "source_catalog": source_manifest,
            "split": split_manifest,
            "case_counts": counts,
            "scenario_counts": {
                split: {
                    scenario: sum(
                        1
                        for spec in specs
                        if spec.split == split and spec.scenario == scenario
                    )
                    for scenario in SCENARIOS
                }
                for split in ("train", "validation", "test")
            },
            "existing_double_talk_test_used": False,
        },
    )
    _append_event(
        output,
        {
            "time": _now(),
            "event": "sources_split",
            "file_counts": split_manifest["file_counts"],
            "speaker_counts": split_manifest["speaker_counts"],
            "file_sha256_disjoint": split_manifest["file_sha256_disjoint"],
            "speaker_disjoint": split_manifest["speaker_disjoint"],
        },
    )

    prepared: list[PreparedCase] = []
    native_started = time.monotonic()
    for position, spec in enumerate(specs, 1):
        _resource_preflight(output)
        case_dir, synthesis_metadata = _synthesize_case(spec, output, args.seed)
        native_metadata, arrays = _run_native_first_pass(
            native_cli, case_dir, output, args.native_timeout
        )
        metadata = {
            "synthesis": synthesis_metadata,
            "native": native_metadata,
        }
        prepared.append(
            PreparedCase(
                spec=spec,
                residual=arrays["residual"],
                reference=arrays["reference"],
                target=arrays["target"],
                metadata=metadata,
            )
        )
        if position == 1 or position % 10 == 0 or position == len(specs):
            event = {
                "time": _now(),
                "event": "native_case_completed",
                "completed": position,
                "total": len(specs),
                "case_id": spec.case_id,
                "linear_aec_delay_samples": metadata["native"]["alignment"][
                    "linear_aec_delay_samples"
                ],
            }
            _append_event(output, event)
            print(json.dumps(event, ensure_ascii=False), flush=True)
    native_elapsed = time.monotonic() - native_started

    train_cases = [case for case in prepared if case.spec.split == "train"]
    validation_cases = [
        case for case in prepared if case.spec.split == "validation"
    ]
    test_cases = [case for case in prepared if case.spec.split == "test"]
    if len(train_cases) != counts["train"] or len(validation_cases) != 12 or len(test_cases) != 12:
        raise ValueError("prepared case counts differ from the frozen split contract")

    initial_validation = _evaluate_checkpoint_cpu(
        initial_checkpoint, validation_cases, phase="initial_random_control"
    )
    model.to(device)
    training = _train_model(
        model,
        train_cases,
        steps=args.steps,
        seed=args.seed,
        device=device,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        spectral_weight=args.spectral_weight,
    )
    final_state = _state_dict_cpu(model)
    final_weight_sha = _state_dict_sha256(final_state)
    checkpoint = output / "checkpoint.pt"
    final_metadata = {
        "schema": SCHEMA,
        "variant": VARIANT,
        "phase": "final_trained_experimental",
        "model_training_status": TRAINING_STATUS,
        "official_pretrained_weights": False,
        "seed": args.seed,
        "steps": args.steps,
        "parameter_count": parameter_count,
        "weight_sha256": final_weight_sha,
        "initial_weight_sha256": initial_weight_sha,
        "training_device": str(device),
        "cpu_final_inference_required": True,
        "model_source": str(MODEL_SOURCE),
        "model_source_sha256": _sha256_file(MODEL_SOURCE),
    }
    _save_checkpoint_once(
        checkpoint,
        _checkpoint_payload(final_state, final_metadata),
    )
    checkpoint_sha = _sha256_file(checkpoint)
    _append_event(
        output,
        {
            "time": _now(),
            "event": "final_checkpoint_saved",
            "path": str(checkpoint),
            "sha256": checkpoint_sha,
            "weight_sha256": final_weight_sha,
            "steps": args.steps,
        },
    )

    final_validation = _evaluate_checkpoint_cpu(
        checkpoint, validation_cases, phase="final_trained_experimental"
    )
    validation_baseline = initial_validation["aggregate"]["waveform_nmse_global"]
    validation_candidate = final_validation["aggregate"]["waveform_nmse_global"]
    validation_improvement = _relative_improvement(
        validation_baseline, validation_candidate
    )

    test_prediction_dir = output / "evaluation" / "test-predictions"
    test_prediction_dir.mkdir(parents=True, exist_ok=False)
    test_mdf = _evaluate_mdf_baseline(test_cases)
    test_initial = _evaluate_checkpoint_cpu(
        initial_checkpoint,
        test_cases,
        phase="initial_random_control",
        prediction_dir=test_prediction_dir,
    )
    test_final = _evaluate_checkpoint_cpu(
        checkpoint,
        test_cases,
        phase="final_trained_experimental",
        prediction_dir=test_prediction_dir,
    )
    test_comparison = {
        "mdf": test_mdf["aggregate"],
        "random_initial": test_initial["aggregate"],
        "trained_final": test_final["aggregate"],
        "trained_vs_mdf_waveform_nmse_relative_improvement": _relative_improvement(
            test_mdf["aggregate"]["waveform_nmse_global"],
            test_final["aggregate"]["waveform_nmse_global"],
        ),
        "trained_vs_random_waveform_nmse_relative_improvement": _relative_improvement(
            test_initial["aggregate"]["waveform_nmse_global"],
            test_final["aggregate"]["waveform_nmse_global"],
        ),
    }
    final_report = {
        "schema": SCHEMA,
        "variant": VARIANT,
        "model_training_status": TRAINING_STATUS,
        "official_pretrained_weights": False,
        "formal_compliance": False,
        "general_speech_quality_validated": False,
        "deployment_approved": False,
        "cpu_final_inference": True,
        "seed": args.seed,
        "steps": args.steps,
        "device": {
            "requested": args.device,
            "training": str(device),
            "validation_and_test_inference": "cpu",
            "cuda_available": torch.cuda.is_available(),
        },
        "model": {
            "source": str(MODEL_SOURCE),
            "source_sha256": _sha256_file(MODEL_SOURCE),
            "parameter_count": parameter_count,
            "initial_weight_sha256": initial_weight_sha,
            "final_weight_sha256": final_weight_sha,
            "initial_checkpoint": str(initial_checkpoint),
            "initial_checkpoint_sha256": initial_checkpoint_sha,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_loaded_with_weights_only": True,
        },
        "native_export_contract": {
            "profile": "vcp",
            "aec_backend": "speex",
            "no_speex_coherence_residual": True,
            "no_speex_multirender_context": True,
            "no_bf": True,
            "complex_pf_export_inputs": True,
            "inputs": ["prefix.pf-residual.wav", "prefix.pf-reference.wav"],
            "cpu_fused_or_ns_output_used": False,
            "native_cli": str(native_cli),
            "native_cli_sha256": _sha256_file(native_cli),
        },
        "data": {
            "source_catalog": source_manifest,
            "split": split_manifest,
            "case_counts": counts,
            "total_cases": len(prepared),
            "scenario_counts": {
                split: {
                    scenario: sum(
                        1
                        for case in prepared
                        if case.spec.split == split and case.spec.scenario == scenario
                    )
                    for scenario in SCENARIOS
                }
                for split in ("train", "validation", "test")
            },
            "existing_double_talk_test_used": False,
            "test_used_for_training_or_selection": False,
            "validation_used_for_model_selection": False,
            "near_far_same_file_case_count": sum(
                1
                for case in prepared
                if case.spec.near_source.sha256 == case.spec.far_source.sha256
            ),
        },
        "cases": [
            {
                "case_id": case.spec.case_id,
                "split": case.spec.split,
                "scenario": case.spec.scenario,
                "near_sources": [
                    {
                        "path": case.spec.near_source.path,
                        "speaker": case.spec.near_source.speaker,
                        "sha256": case.spec.near_source.sha256,
                    }
                ]
                + (
                    [
                        {
                            "path": case.spec.second_near_source.path,
                            "speaker": case.spec.second_near_source.speaker,
                            "sha256": case.spec.second_near_source.sha256,
                        }
                    ]
                    if case.spec.second_near_source is not None
                    else []
                ),
                "far_source": {
                    "path": case.spec.far_source.path,
                    "speaker": case.spec.far_source.speaker,
                    "sha256": case.spec.far_source.sha256,
                },
                "metadata": case.metadata,
            }
            for case in prepared
        ],
        "timing": {
            "native_case_generation_seconds": native_elapsed,
            "training_seconds": training["elapsed_seconds"],
        },
        "training": {
            "optimizer": training["optimizer"],
            "loss_definition": (
                "mean(target-energy-normalized waveform MSE + "
                "spectral_weight * complex STFT L1)"
            ),
            "step_records": training["step_records"],
            "loss_segments": training["loss_segments"],
        },
        "validation": {
            "initial_random_control": initial_validation,
            "final_trained_experimental": final_validation,
            "waveform_nmse_global_improvement": validation_improvement,
            "improved": (
                validation_improvement is not None and validation_improvement > 0
            ),
        },
        "test": {
            "mdf_residual_baseline": test_mdf,
            "random_initial_control": test_initial,
            "trained_final_cpu": test_final,
            "comparison": test_comparison,
        },
        "limitations": [
            "No official or upstream Complex_PF checkpoint was used.",
            "Training and evaluation data are synthetic mixtures from a bounded local speech subset.",
            "Same-distribution test improvement, even if positive, is not formal ITU compliance.",
            "Generalization, hardware latency, HiFi integration, and production deployment are not established.",
        ],
        "started_at": None,
        "finished_at": _now(),
    }
    # Preserve a single stable start timestamp without rewriting an existing file.
    created = json.loads((output / "source-split.json").read_text(encoding="utf-8"))
    del created
    final_report["started_at"] = datetime.fromtimestamp(
        output.stat().st_ctime, timezone.utc
    ).isoformat()
    _write_json_once(output / "training.json", final_report)
    _append_event(
        output,
        {
            "time": _now(),
            "event": "training_completed",
            "training_json": str(output / "training.json"),
            "validation_improved": final_report["validation"]["improved"],
        },
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "output": str(output),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha,
                "initial_weight_sha256": initial_weight_sha,
                "final_weight_sha256": final_weight_sha,
                "validation_improved": final_report["validation"]["improved"],
                "validation_nmse_improvement": validation_improvement,
                "test": test_comparison,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return _run(args)
    except ResourceSafetyStop as error:
        print(f"RESOURCE SAFETY STOP: {error}", file=sys.stderr, flush=True)
        return 3
    except OSError as error:
        if error.errno == errno.ENOSPC:
            print(f"RESOURCE SAFETY STOP: ENOSPC: {error}", file=sys.stderr, flush=True)
            return 3
        raise


if __name__ == "__main__":
    raise SystemExit(main())
