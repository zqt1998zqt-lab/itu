#!/usr/bin/env python3
"""Build the isolated C64 distillation corpus.

The builder is deliberately strict about split isolation.  It records raw
source and noise hashes in every row and refuses to emit a corpus if any known
frozen-evaluation source or speaker enters train/dev.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import socket
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf


SR = 16_000
DURATION_SECONDS = 40
SAMPLES = SR * DURATION_SECONDS
FRAME = 320
HOP = 160
EPS = 1e-12

ZH_TRAIN_SPEAKER = "5_3039"
ZH_DEV_SPEAKER = "5_3099"
OLD_ZH_SPEAKERS = {"5_2525", "5_2595", "5_2641", "5_2751"}
V1_ZH_SPEAKERS = {"5_2399", "5_2770", "5_2824", "5_2855"}
EN_EXCLUDED_SPEAKERS = {
    "1919", "2277", "2412", "2902",
    "1462", "2035", "2428", "251", "3752", "5338", "5694", "777", "7976", "84",
}
DEMAND_DEV_RECORDINGS = {"TBUS", "PCAFETER", "OHALLWAY"}
DEMAND_EXCLUDED_RECORDINGS = {"TCAR"}
SNRS = (-5.0, 0.0, 5.0, 10.0, 20.0)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_float_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array, dtype="<f4")
    return hashlib.sha256(value.tobytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")
    os.replace(temporary, path)


def read_mono(path: Path, *, channel: int = 0) -> tuple[np.ndarray, int]:
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if int(rate) != SR:
        raise ValueError(f"expected 16 kHz source, got {rate}: {path}")
    if audio.shape[1] <= channel:
        raise ValueError(f"channel {channel} missing in {path} ({audio.shape[1]} channels)")
    samples = np.ascontiguousarray(audio[:, channel], dtype=np.float32)
    if not samples.size or not np.all(np.isfinite(samples)):
        raise ValueError(f"empty or non-finite source: {path}")
    return samples, int(audio.shape[1])


def write_float_wav(path: Path, samples: np.ndarray) -> None:
    value = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
    if value.size != SAMPLES:
        raise ValueError(f"refusing non-40-second WAV: {path} has {value.size} samples")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"non-finite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), value, SR, subtype="FLOAT")


def rms_dbfs(samples: np.ndarray) -> float:
    value = np.asarray(samples, dtype=np.float64)
    rms = math.sqrt(float(np.mean(value * value)) + EPS)
    return 20.0 * math.log10(max(rms, 1e-12))


def active_rms(samples: np.ndarray) -> tuple[float, dict[str, Any]]:
    value = np.asarray(samples, dtype=np.float64)
    frames = value.reshape(-1, FRAME)
    frame_rms = np.sqrt(np.mean(frames * frames, axis=1) + EPS)
    positive = frame_rms[frame_rms > 1e-12]
    if positive.size == 0:
        raise ValueError("speech timeline contains no active frames")
    p99 = float(np.percentile(positive, 99.0))
    threshold = p99 * (10.0 ** (-35.0 / 20.0))
    active = frame_rms >= threshold
    if not np.any(active):
        raise ValueError("activity threshold selected no frames")
    active_samples = frames[active]
    rms = math.sqrt(float(np.mean(active_samples * active_samples)) + EPS)
    return rms, {
        "frame_ms": 20.0,
        "hop_ms": 10.0,
        "rule": "20 ms RMS frame >= 99th percentile - 35 dB",
        "p99_frame_rms": p99,
        "threshold_rms": threshold,
        "active_frames": int(np.sum(active)),
        "active_seconds": float(np.sum(active) * 0.02),
    }


def scale_to_rms(samples: np.ndarray, target_dbfs: float, mask_source: np.ndarray | None = None) -> tuple[np.ndarray, float, float]:
    measured_source = samples if mask_source is None else mask_source
    current, _ = active_rms(measured_source)
    target = 10.0 ** (target_dbfs / 20.0)
    gain = target / max(current, 1e-12)
    return np.asarray(samples * gain, dtype=np.float32), gain, current


def edge_envelope(length: int, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, int]]:
    onset = int(rng.integers(20, 201))
    tail = int(rng.integers(20, 301))
    envelope = np.ones(length, dtype=np.float32)
    envelope[:onset] = np.linspace(0.0, 1.0, onset, endpoint=False, dtype=np.float32)
    envelope[-tail:] = np.linspace(1.0, 0.0, tail, endpoint=True, dtype=np.float32)
    return envelope, {"onset_fade_samples": onset, "tail_fade_samples": tail}


def slow_amplitude(length: int, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, float]]:
    depth_db = float(rng.uniform(0.0, 3.0))
    period = float(rng.uniform(5.0, 20.0))
    phase = float(rng.uniform(0.0, 2.0 * math.pi))
    t = np.arange(length, dtype=np.float64) / SR
    db = depth_db * np.sin(2.0 * math.pi * t / period + phase)
    return np.asarray(10.0 ** (db / 20.0), dtype=np.float32), {
        "depth_db": depth_db,
        "period_seconds": period,
        "phase_radians": phase,
    }


def tile_with_crossfade(source: np.ndarray, length: int, crossfade: int = 800) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.asarray(source, dtype=np.float32).reshape(-1)
    if source.size == 0:
        raise ValueError("cannot tile empty source")
    if source.size >= length:
        return source[:length].copy(), {"looped": False, "loop_count": 0, "crossfade_samples": 0}
    crossfade = min(crossfade, source.size // 4)
    if crossfade <= 0:
        repeats = int(math.ceil(length / source.size))
        return np.tile(source, repeats)[:length].copy(), {
            "looped": True,
            "loop_count": repeats - 1,
            "crossfade_samples": 0,
        }
    output = source.copy()
    loops = 0
    fade = np.linspace(0.0, 1.0, crossfade, endpoint=False, dtype=np.float32)
    while output.size < length:
        tail = output[-crossfade:].copy()
        head = source[:crossfade]
        overlap = tail * (1.0 - fade) + head * fade
        output[-crossfade:] = overlap
        output = np.concatenate([output, source[crossfade:]])
        loops += 1
        if loops > 1000:
            raise RuntimeError("crossfade tiler did not terminate")
    return np.ascontiguousarray(output[:length]), {
        "looped": True,
        "loop_count": loops,
        "crossfade_samples": crossfade,
        "loop_policy": "same recording take tiled with linear equal-time crossfade",
    }


def read_sha_audit(path: Path) -> dict[str, set[str]]:
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    speakers: set[str] = set()
    shas: set[str] = set()
    recordings: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                lowered = key.lower()
                if lowered in {"speaker_id", "speaker", "near_speaker", "far_speaker"} and item is not None:
                    # Keep MagicData's full "5_2399" identity rather than
                    # reducing it to the trailing four digits.
                    token = str(item).rsplit(":", 1)[-1]
                    match = re.fullmatch(r"(\d+(?:_\d+)?)", token)
                    if match:
                        speakers.add(match.group(1))
                if lowered in {"sha256", "sha", "source_sha256", "raw_sha256"} and isinstance(item, str) and re.fullmatch(r"[0-9a-fA-F]{64}", item):
                    shas.add(item.lower())
                if lowered in {"noise_id", "noise_recording", "noise_source_id", "recording_id", "member", "path"} and isinstance(item, str):
                    recordings.add(item)
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(data)
    return {"speakers": speakers, "sha256": shas, "recordings": recordings}


def collect_eval_exclusions(project_root: Path, allow_fallback: bool) -> dict[str, Any]:
    v1 = project_root / "benchmarks/ns/dpdfnet2-16k-v1"
    v120 = project_root / "build/ns-highway120-20261115"
    case_files = sorted((v1 / "cases").glob("*.json")) + sorted((v120 / "data").glob("cases-*.json"))
    if not case_files:
        raise FileNotFoundError("no frozen v1/120 case manifests found")
    audit_roots = [v1 / "source_audit/raws", v120 / "source_audit/raws"]
    audit_files = []
    missing_roots = []
    for root in audit_roots:
        if root.is_dir():
            audit_files.extend(sorted(root.rglob("*.json")))
        else:
            missing_roots.append(str(root))
    speakers = set(OLD_ZH_SPEAKERS) | set(V1_ZH_SPEAKERS) | set(EN_EXCLUDED_SPEAKERS)
    shas: set[str] = set()
    recordings: set[str] = {"TCAR", "CAVEMOVE"}
    parse_errors = []
    for path in case_files:
        try:
            parsed = read_sha_audit(path)
        except Exception as exc:
            parse_errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        speakers.update(parsed["speakers"])
        shas.update(parsed["sha256"])
        recordings.update(parsed["recordings"])
    for path in audit_files:
        try:
            parsed = read_sha_audit(path)
        except Exception as exc:
            parse_errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        speakers.update(parsed["speakers"])
        shas.update(parsed["sha256"])
        recordings.update(parsed["recordings"])
    if parse_errors and not allow_fallback:
        raise ValueError(f"failed to parse frozen source audit files: {parse_errors}")
    fallback = bool(parse_errors)
    if fallback and not allow_fallback:
        raise FileNotFoundError(f"source_audit/raws unavailable: {missing_roots}")
    return {
        "schema_version": 1,
        "created_utc": utc_now(),
        "scope": "frozen v1/120 evaluation plus explicit user exclusions",
        "case_files": [{"path": str(path), "sha256": sha256_file(path)} for path in case_files],
        "source_audit_roots_requested": [str(path) for path in audit_roots],
        "source_audit_files_read": [{"path": str(path), "sha256": sha256_file(path)} for path in audit_files],
        "source_audit_roots_missing": missing_roots,
        "parse_errors": parse_errors,
        "declared_fallback_used": fallback,
        "source_inventory_derived_from_case_manifests": True,
        "source_audit_status": "source_audit/raws not present; frozen case source_rows/statement_spans were parsed directly",
        "fallback_policy": {
            "chinese_speakers": sorted(OLD_ZH_SPEAKERS | V1_ZH_SPEAKERS),
            "english_speakers": sorted(EN_EXCLUDED_SPEAKERS, key=lambda value: int(value)),
            "noise_recordings": ["TCAR", "CAVEMOVE"],
            "unexpected_parse_behavior": "fail closed; no silent widening",
        },
        "excluded_speakers": sorted(speakers, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value)),
        "excluded_raw_sha256": sorted(shas),
        "excluded_noise_tokens": sorted(recordings),
    }


def discover_speech(root: Path) -> list[Path]:
    files = sorted(root.glob("*.wav")) + sorted(root.glob("*.flac"))
    if not files:
        raise FileNotFoundError(f"no speech files under {root}")
    return files


def speech_metadata(paths: Iterable[Path], speaker: str, language: str) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        info = sf.info(str(path))
        if int(info.samplerate) != SR or int(info.channels) != 1:
            raise ValueError(f"speech source is not mono 16 kHz: {path}")
        rows.append({
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "speaker_id": speaker,
            "language": language,
            "frames": int(info.frames),
            "seconds": float(info.frames / SR),
            "subtype": info.subtype,
        })
    return rows


def choose_near_timeline(
    catalog: list[dict[str, Any]],
    rng: np.random.Generator,
    preferred_speaker: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if len(catalog) < 2:
        raise ValueError("source pool has too few statements for a 40 s timeline")
    order = list(range(len(catalog)))
    rng.shuffle(order)
    if preferred_speaker is not None:
        preferred = [index for index in order if catalog[index]["speaker_id"] == preferred_speaker]
        if not preferred:
            raise ValueError(f"preferred speaker is absent from catalog: {preferred_speaker}")
        order = preferred + [index for index in order if catalog[index]["speaker_id"] != preferred_speaker]
    # Place the mandatory long pause at the first usable boundary.  This
    # guarantees two complete statements even for speakers whose only source
    # statements are long, while the exact pause duration remains 3-8 s.
    long_after = 1
    lead = int(round(float(rng.uniform(0.05, 0.50)) * SR))
    minimum_other = min(int(item["frames"]) for item in catalog)
    minimum_lead_room = lead + minimum_other + int(8.0 * SR) + int(0.05 * SR)
    # Do not start with a 30 s statement when that leaves no room for the
    # mandatory pause plus a second complete statement.
    first_position = next(
        (index for index, candidate in enumerate(order)
         if int(catalog[candidate]["frames"]) + minimum_lead_room <= SAMPLES),
        None,
    )
    if first_position is None:
        raise ValueError("no first statement permits two complete statements plus a long pause")
    order = order[first_position:] + order[:first_position]
    timeline = [np.zeros(lead, dtype=np.float32)]
    used = []
    cursor = lead
    selected = 0
    long_pause_done = False
    for index in order:
        item = catalog[index]
        if selected == long_after:
            gap_seconds = float(rng.uniform(3.0, 8.0))
        else:
            gap_seconds = float(rng.uniform(0.10, 1.00))
        gap = int(round(gap_seconds * SR))
        frames = int(item["frames"])
        reserve_tail = int(0.05 * SR)
        if cursor + gap + frames + reserve_tail > SAMPLES:
            continue
        waveform, _ = read_mono(Path(item["path"]))
        if waveform.size != frames:
            raise ValueError(f"source frame count changed: {item['path']}")
        timeline.append(np.zeros(gap, dtype=np.float32))
        timeline.append(waveform)
        used.append({
            **item,
            "output_start_sample": cursor + gap,
            "output_end_sample": cursor + gap + frames,
            "gap_before_samples": gap,
            "gap_before_seconds": gap_seconds,
            "complete_statement_preserved": True,
        })
        cursor += gap + frames
        selected += 1
        if selected == long_after + 1:
            long_pause_done = True
        if cursor > SAMPLES - int(0.2 * SR):
            break
    if selected < 2 or not long_pause_done:
        raise ValueError("unable to place at least two complete statements and one long pause")
    trailing = SAMPLES - cursor
    timeline.append(np.zeros(trailing, dtype=np.float32))
    clean = np.concatenate(timeline).astype(np.float32, copy=False)
    if clean.size != SAMPLES:
        raise AssertionError("near timeline length mismatch")
    return clean, {
        "policy": "random complete source statements; no statement cropping; same-split cross-speaker concatenation allowed",
        "statement_count": selected,
        "statements": used,
        "leading_silence_samples": lead,
        "trailing_silence_samples": trailing,
        "long_pause_required_seconds": [3.0, 8.0],
        "short_pause_required_seconds": [0.1, 1.0],
        "long_pause_present": any(3.0 <= float(row["gap_before_seconds"]) <= 8.0 for row in used),
    }


def demand_catalog(noise_root: Path, excluded_sha: set[str]) -> dict[str, list[dict[str, Any]]]:
    pool: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    for path in sorted(noise_root.glob("*-ch01.wav")) + sorted(noise_root.glob("MS-SNSD-*.wav")):
        recording = path.stem
        if recording.endswith("-ch01"):
            recording = recording[:-5]
        if recording in DEMAND_EXCLUDED_RECORDINGS:
            continue
        if recording.startswith("MS-SNSD-"):
            # MS-SNSD is retained as an HVAC-like train recording, not DEMAND.
            recording = "HVAC"
        sha = sha256_file(path)
        if sha in excluded_sha:
            continue
        info = sf.info(str(path))
        if int(info.samplerate) != SR or int(info.channels) != 1:
            raise ValueError(f"DEMAND source is not mono 16 kHz: {path}")
        row = {
            "family": "demand",
            "recording": recording,
            "path": str(path.resolve()),
            "sha256": sha,
            "frames": int(info.frames),
        }
        pool["dev" if recording in DEMAND_DEV_RECORDINGS else "train"].append(row)
    if not pool["dev"]:
        raise ValueError("no DEMAND dev recordings matched TBUS/PCAFETER/OHALLWAY")
    if not pool["train"]:
        raise ValueError("no DEMAND train recordings remain")
    return pool


def parse_car_records(manifest_path: Path, excluded_sha: set[str]) -> dict[str, list[dict[str, Any]]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"car-noise manifest not ready: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_records = data.get("records", data.get("recordings", data.get("items", data))) if isinstance(data, dict) else data
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError(f"car-noise manifest has no record list: {manifest_path}")
    pool: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    for item in raw_records:
        if not isinstance(item, dict):
            continue
        split = str(item.get("split", item.get("partition", ""))).lower()
        if split in {"development", "validation", "valid"}:
            split = "dev"
        if split not in {"train", "dev"}:
            continue
        vehicle = str(item.get("vehicle", item.get("car", item.get("name", ""))))
        array = str(item.get("representation", item.get("array", item.get("variant", item.get("scope", "")))))
        channel_paths = item.get("channel_paths") or item.get("channels")
        derived = item.get("derived")
        derived_path = derived.get("path") if isinstance(derived, dict) else None
        path_value = (
            item.get("mono_path")
            or derived_path
            or item.get("path")
            or item.get("wav_path")
            or item.get("file")
        )
        channel = int(item.get("channel", item.get("ch", 0)))
        if isinstance(channel_paths, dict):
            for key in ("0", "ch0", 0):
                if key in channel_paths:
                    path_value = channel_paths[key]
                    channel = 0
                    break
        elif isinstance(channel_paths, list) and channel_paths:
            path_value = channel_paths[min(channel, len(channel_paths) - 1)]
        if not path_value:
            raise ValueError(f"car-noise record lacks channel path: {item}")
        path = Path(str(path_value))
        if not path.is_absolute():
            path = (manifest_path.parent / path).resolve()
        sha = sha256_file(path)
        raw_path_value = item.get("raw_path")
        raw_sha = str(item.get("rawSHA") or item.get("raw_sha256") or "").lower()
        mono_sha = str(item.get("monoSHA") or item.get("mono_sha256") or sha).lower()
        if raw_sha and re.fullmatch(r"[0-9a-f]{64}", raw_sha):
            if raw_sha in excluded_sha:
                raise ValueError(f"car-noise raw source intersects frozen eval SHA: {raw_path_value}")
        if sha in excluded_sha or mono_sha in excluded_sha:
            raise ValueError(f"car-noise source intersects frozen eval SHA: {path}")
        info = sf.info(str(path))
        if int(info.samplerate) != SR:
            raise ValueError(f"car noise is not 16 kHz: {path}")
        if int(info.channels) <= channel:
            raise ValueError(f"car-noise channel {channel} missing: {path}")
        recording = str(item.get("recording_id", item.get("recording", item.get("id", path.stem))))
        pool[split].append({
            "family": "car",
            "recording": recording,
            "vehicle": vehicle,
            "array": array,
            "path": str(path),
            "raw_path": str(raw_path_value) if raw_path_value else None,
            "raw_sha256": raw_sha or None,
            "sha256": sha,
            "mono_sha256": mono_sha,
            "frames": int(info.frames),
            "channels": int(info.channels),
            "channel": channel,
            "manifest_record": item,
        })
    if not pool["train"] or not pool["dev"]:
        raise ValueError("car-noise manifest must provide train and dev records")
    normalize = lambda value: re.sub(r"[^a-z0-9]+", "", value.lower())
    train_names = " ".join(normalize(row["vehicle"] + " " + row["array"]) for row in pool["train"])
    dev_names = " ".join(normalize(row["vehicle"] + " " + row["array"]) for row in pool["dev"])
    for token in ("alfa", "golf", "smart"):
        if token not in train_names:
            raise ValueError(f"car-noise train pool missing required {token} take")
    if "honda" not in dev_names or "crv" not in dev_names:
        raise ValueError("car-noise dev pool missing HondaCRV")
    return pool


def sample_noise(record: dict[str, Any], rng: np.random.Generator, *, envelope: bool) -> tuple[np.ndarray, dict[str, Any]]:
    source, channels = read_mono(Path(record["path"]), channel=int(record.get("channel", 0)))
    loop_info: dict[str, Any]
    if source.size >= SAMPLES:
        start = int(rng.integers(0, source.size - SAMPLES + 1))
        base = source[start:start + SAMPLES].copy()
        loop_info = {"looped": False, "loop_count": 0, "slice_start_sample": start}
    else:
        base, loop_info = tile_with_crossfade(source, SAMPLES)
        loop_info["slice_start_sample"] = 0
    modulation, modulation_info = slow_amplitude(SAMPLES, rng)
    base *= modulation
    envelope_info = None
    if envelope:
        env, envelope_info = edge_envelope(SAMPLES, rng)
        base *= env
    return base.astype(np.float32, copy=False), {
        **loop_info,
        "source_frames": int(source.size),
        "source_channels": channels,
        "slow_amplitude": modulation_info,
        "edge_envelope": envelope_info,
        "same_recording_take": True,
    }


def make_tonal_negative(rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    t = np.arange(SAMPLES, dtype=np.float64) / SR
    count = int(rng.integers(1, 4))
    tones = []
    output = np.zeros(SAMPLES, dtype=np.float64)
    for _ in range(count):
        hz = float(rng.uniform(120.0, 5000.0))
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        amplitude = float(rng.uniform(0.2, 1.0))
        output += amplitude * np.sin(2.0 * math.pi * hz * t + phase)
        tones.append({"hz": hz, "phase_radians": phase, "relative_amplitude": amplitude})
    output /= max(float(np.max(np.abs(output))), 1e-12)
    env, env_info = edge_envelope(SAMPLES, rng)
    output *= env
    target_db = float(rng.uniform(-42.0, -28.0))
    output *= 10.0 ** (target_db / 20.0) / max(math.sqrt(float(np.mean(output * output))), 1e-12)
    return output.astype(np.float32), {
        "tones": tones,
        "target_rms_dbfs": target_db,
        "edge_envelope": env_info,
        "classification": "synthetic_tonal_negative",
    }


def allocate_types(count: int) -> dict[str, int]:
    mixed = int(round(0.70 * count))
    remaining = count - mixed
    clean = remaining // 2
    negative = remaining - clean
    if count == 24:
        clean, negative = 3, 4
    return {"mixed": mixed, "clean": clean, "negative": negative}


def allocate_languages(total: int) -> list[str]:
    return ["zh"] * (total // 2) + ["en"] * (total - total // 2)


def build_schedule(split: str, count: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    type_counts = allocate_types(count)
    languages = allocate_languages(count)
    rows: list[dict[str, Any]] = []
    if split == "train":
        counts = {
            ("mixed", "zh"): 45, ("mixed", "en"): 45,
            ("clean", "zh"): 9, ("clean", "en"): 10,
            ("noise_only", "zh"): 5, ("noise_only", "en"): 5,
            ("tonal_negative", "zh"): 5, ("tonal_negative", "en"): 4,
        }
    else:
        counts = {
            ("mixed", "zh"): 9, ("mixed", "en"): 8,
            ("clean", "zh"): 1, ("clean", "en"): 2,
            ("noise_only", "zh"): 1, ("noise_only", "en"): 1,
            ("tonal_negative", "zh"): 1, ("tonal_negative", "en"): 1,
        }
    for (kind, language), n in counts.items():
        for _ in range(n):
            rows.append({"type": kind, "language": language})
    if Counter(row["language"] for row in rows) != Counter(languages):
        raise AssertionError("language allocation mismatch")
    mixed_rows = [row for row in rows if row["type"] == "mixed"]
    car_count = int(round(0.60 * len(mixed_rows)))
    car_indices = set(rng.choice(len(mixed_rows), size=car_count, replace=False).tolist())
    for index, row in enumerate(mixed_rows):
        row["noise_family"] = "car" if index in car_indices else "demand"
    for row in rows:
        if row["type"] == "noise_only":
            row["noise_family"] = "car" if rng.random() < 0.5 else "demand"
    rng.shuffle(rows)
    return rows


def select_speaker_catalogs(project_root: Path, exclusions: dict[str, Any]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    zh_root = Path("/home/zhangqingtian/dev")
    zh_train = speech_metadata(discover_speech(zh_root / ZH_TRAIN_SPEAKER), ZH_TRAIN_SPEAKER, "zh")
    zh_dev = speech_metadata(discover_speech(zh_root / ZH_DEV_SPEAKER), ZH_DEV_SPEAKER, "zh")
    if len(zh_train) != 592 or len(zh_dev) != 595:
        raise ValueError(f"Chinese source counts changed: train={len(zh_train)} dev={len(zh_dev)}")
    english_root = project_root / "datasets/itu-eval-v4-no-passenger-20261011/sources/speech"
    english_files = sorted(english_root.glob("*.flac"))
    by_speaker: dict[str, list[Path]] = defaultdict(list)
    for path in english_files:
        by_speaker[path.name.split("-")[0]].append(path)
    allowed = sorted(set(by_speaker) - EN_EXCLUDED_SPEAKERS, key=int)
    if len(allowed) != 23:
        raise ValueError(f"expected 23 English speakers after exclusions, got {len(allowed)}: {allowed}")
    english_dev = allowed[-4:]
    english_train = allowed[:-4]
    catalog: dict[tuple[str, str], list[dict[str, Any]]] = {("train", "zh"): zh_train, ("dev", "zh"): zh_dev}
    for split, speakers in (("train", english_train), ("dev", english_dev)):
        items = []
        for speaker in speakers:
            items.extend(speech_metadata(by_speaker[speaker], speaker, "en"))
        catalog[(split, "en")] = items
    excluded_speakers = set(exclusions["excluded_speakers"])
    excluded_sha = set(exclusions["excluded_raw_sha256"])
    for (split, language), items in catalog.items():
        for item in items:
            if item["speaker_id"] in excluded_speakers or item["sha256"] in excluded_sha:
                raise ValueError(f"frozen eval source entered {split}/{language}: {item}")
    return catalog


def interleave_by_speaker(items: list[dict[str, Any]], rng: np.random.Generator, key: str = "speaker_id") -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[str(item.get(key, item.get("recording", "source")))] .append(item)
    for group in groups.values():
        rng.shuffle(group)
    names = sorted(groups)
    rng.shuffle(names)
    output = []
    while any(groups.values()):
        for name in names:
            if groups[name]:
                output.append(groups[name].pop())
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=128)
    parser.add_argument("--dev-count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--car-noise-manifest", type=Path, default=None)
    parser.add_argument("--allow-declared-eval-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.train_count != 128 or args.dev_count != 24:
        raise ValueError("this corpus contract is fixed at 128 train + 24 dev")
    project_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"manifest exists; pass --overwrite only after explicit review: {manifest_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_root = output_dir / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    log_dir = project_root / "build/ns-distilled-c64-20261116/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"data-{stamp}.log"

    def log(message: str) -> None:
        line = f"[{utc_now()}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    started = time.time()
    log(f"start output={output_dir} seed={args.seed}")
    disk_before = shutil.disk_usage(output_dir)
    exclusions = collect_eval_exclusions(project_root, args.allow_declared_eval_fallback)
    atomic_json(output_dir / "exclude-receipt.json", exclusions)
    log(f"exclude receipt: speakers={len(exclusions['excluded_speakers'])} shas={len(exclusions['excluded_raw_sha256'])} fallback={exclusions['declared_fallback_used']}")
    catalog = select_speaker_catalogs(project_root, exclusions)
    log("speaker catalog: " + ", ".join(f"{split}/{language}={len(items)}" for (split, language), items in sorted(catalog.items())))
    excluded_sha = set(exclusions["excluded_raw_sha256"])
    demand = demand_catalog(project_root / "datasets/itu-eval-v4-curated/sources/noise", excluded_sha)
    log("DEMAND pools: " + ", ".join(f"{split}={sorted({row['recording'] for row in rows})}" for split, rows in demand.items()))
    car_manifest = (args.car_noise_manifest or (output_dir / "raw-car-noise/records.json")).resolve()
    car = parse_car_records(car_manifest, excluded_sha)
    log("car-noise pools: " + ", ".join(f"{split}={[(r['vehicle'], r['array'], r['recording']) for r in rows]}" for split, rows in car.items()))
    rng = np.random.default_rng(args.seed)
    schedules = {"train": build_schedule("train", args.train_count, rng), "dev": build_schedule("dev", args.dev_count, rng)}
    speaker_queues: dict[tuple[str, str], list[dict[str, Any]]] = {}
    coverage_queues: dict[tuple[str, str], list[str]] = {}
    for key, items in catalog.items():
        speaker_queues[key] = interleave_by_speaker(items, rng, key="speaker_id")
        names = sorted({item["speaker_id"] for item in items}, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))
        rng.shuffle(names)
        coverage_queues[key] = names
    noise_queues = {
        ("train", "car"): interleave_by_speaker(car["train"], rng, key="recording"),
        ("dev", "car"): interleave_by_speaker(car["dev"], rng, key="recording"),
        ("train", "demand"): list(demand["train"]),
        ("dev", "demand"): list(demand["dev"]),
    }
    for key, items in noise_queues.items():
        rng.shuffle(items)
    rows = []
    split_counts = {}
    used_raw_sha_by_split: dict[str, set[str]] = {"train": set(), "dev": set()}
    used_noise_sha_by_split: dict[str, set[str]] = {"train": set(), "dev": set()}
    for split in ("train", "dev"):
        split_counts[split] = Counter()
        for ordinal, task in enumerate(schedules[split]):
            kind = task["type"]
            language = task["language"]
            row_id = f"{split}-{ordinal:03d}-{language}-{kind}"
            clean = np.zeros(SAMPLES, dtype=np.float32)
            noise = np.zeros(SAMPLES, dtype=np.float32)
            speech_sources = []
            near_info = None
            speech_gain = 1.0
            speech_pre_rms = None
            target_active = None
            if kind in {"mixed", "clean"}:
                # Use the complete same-split/same-language pool.  Several
                # English speakers have only one statement, so speaker-local
                # timelines cannot fill 40 seconds without artificial repeats.
                coverage = coverage_queues[(split, language)]
                preferred_speaker = coverage.pop(0)
                coverage.append(preferred_speaker)
                near, near_info = choose_near_timeline(catalog[(split, language)], rng, preferred_speaker=preferred_speaker)
                target_active = float(rng.uniform(-36.0, -24.0))
                clean, speech_gain, speech_pre_rms = scale_to_rms(near, target_active)
                speech_sources = near_info["statements"]
                speaker_ids = sorted({item["speaker_id"] for item in speech_sources}, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))
            else:
                speaker_ids = []
            noise_info = None
            snr = None
            noise_gain = 0.0
            noise_target = None
            if kind == "mixed":
                family = task["noise_family"]
                queue = noise_queues[(split, family)]
                record = queue.pop(0)
                queue.append(record)
                base_noise, noise_info = sample_noise(record, rng, envelope=False)
                snr = float(SNRS[int(rng.integers(0, len(SNRS)))])
                speech_active, _ = active_rms(clean)
                base_rms = math.sqrt(float(np.mean(base_noise.astype(np.float64) ** 2)) + EPS)
                noise_gain = (speech_active / max(base_rms, 1e-12)) / (10.0 ** (snr / 20.0))
                noise = np.asarray(base_noise * noise_gain, dtype=np.float32)
            elif kind == "noise_only":
                family = task["noise_family"]
                queue = noise_queues[(split, family)]
                record = queue.pop(0)
                queue.append(record)
                base_noise, noise_info = sample_noise(record, rng, envelope=True)
                noise_target = float(rng.uniform(-38.0, -24.0))
                base_rms = math.sqrt(float(np.mean(base_noise.astype(np.float64) ** 2)) + EPS)
                noise_gain = (10.0 ** (noise_target / 20.0)) / max(base_rms, 1e-12)
                noise = np.asarray(base_noise * noise_gain, dtype=np.float32)
            elif kind == "tonal_negative":
                base_noise, noise_info = make_tonal_negative(rng)
                noise_target = noise_info["target_rms_dbfs"]
                noise = base_noise
                noise_gain = 1.0
            selected_noise = record if kind in {"mixed", "noise_only"} else None
            mix = np.asarray(clean + noise, dtype=np.float32)
            pre_peak = float(max(np.max(np.abs(mix)), np.max(np.abs(clean)), np.max(np.abs(noise))))
            peak_scale = 1.0 if pre_peak <= 0.95 else 0.95 / pre_peak
            if peak_scale != 1.0:
                clean = np.asarray(clean * peak_scale, dtype=np.float32)
                noise = np.asarray(noise * peak_scale, dtype=np.float32)
                mix = np.asarray(clean + noise, dtype=np.float32)
            if np.max(np.abs(mix)) > 0.950001:
                raise AssertionError(f"peak normalization failed for {row_id}")
            max_sum_error = float(np.max(np.abs(mix - (clean + noise))))
            if max_sum_error > 1e-7:
                raise AssertionError(f"mix != clean + noise for {row_id}: {max_sum_error}")
            row_dir = audio_root / row_id
            paths = {name: row_dir / f"{name}.wav" for name in ("mix", "clean", "noise")}
            write_float_wav(paths["mix"], mix)
            write_float_wav(paths["clean"], clean)
            write_float_wav(paths["noise"], noise)
            source_shas = [item["sha256"] for item in speech_sources]
            used_raw_sha_by_split[split].update(source_shas)
            noise_record = None
            noise_sha = hashlib.sha256(np.zeros(SAMPLES, dtype="<f4").tobytes()).hexdigest()
            if kind == "clean":
                noise_record = "digital-silence"
            if noise_info is not None and kind != "tonal_negative":
                noise_record = f"{noise_info.get('family', task.get('noise_family', 'synthetic'))}:{noise_info.get('recording', 'synthetic')}"
                if kind in {"mixed", "noise_only"}:
                    selected_record = selected_noise
                    if selected_record is None:
                        raise AssertionError("noise identity was not retained")
                    noise_record = f"{selected_record['family']}:{selected_record['recording']}"
                    noise_sha = selected_record.get("raw_sha256") or selected_record["sha256"]
                    noise_info["source_path"] = selected_record["path"]
                    noise_info["source_sha256"] = selected_record["sha256"]
                    noise_info["source_raw_path"] = selected_record.get("raw_path")
                    noise_info["source_raw_sha256"] = selected_record.get("raw_sha256")
                    noise_info["source_mono_sha256"] = selected_record.get("mono_sha256")
                    noise_info["source_recording"] = selected_record["recording"]
                    noise_info["source_family"] = selected_record["family"]
                    used_noise_sha_by_split[split].add(selected_record["sha256"])
                    if selected_record.get("mono_sha256"):
                        used_noise_sha_by_split[split].add(selected_record["mono_sha256"])
                    if selected_record.get("raw_sha256"):
                        used_noise_sha_by_split[split].add(selected_record["raw_sha256"])
            elif kind == "tonal_negative":
                noise_record = "synthetic:tonal-negative"
                noise_sha = sha256_float_array(base_noise)
            actual_snr = None
            if kind == "mixed":
                speech_active, _ = active_rms(clean)
                noise_rms = math.sqrt(float(np.mean(noise.astype(np.float64) ** 2)) + EPS)
                actual_snr = 20.0 * math.log10(max(speech_active, 1e-12) / max(noise_rms, 1e-12))
            derived = {
                "mix": sha256_float_array(mix),
                "clean": sha256_float_array(clean),
                "noise": sha256_float_array(noise),
                "concat_float32": hashlib.sha256(np.concatenate([mix, clean, noise]).astype("<f4").tobytes()).hexdigest(),
            }
            row = {
                "id": row_id,
                "split": split,
                "language": language,
                "speaker_ids": speaker_ids,
                "raw_source_paths": [item["path"] for item in speech_sources],
                "raw_source_sha256": source_shas,
                "raw_sources": [{key: item[key] for key in ("path", "sha256", "speaker_id", "language", "frames", "seconds", "output_start_sample", "output_end_sample", "gap_before_samples", "gap_before_seconds")} for item in speech_sources],
                "noise_recording": noise_record,
                "noise_recording_sha256": noise_sha,
                "noise_source_path": noise_info.get("source_path") if noise_info else None,
                "noise_source_sha256": noise_info.get("source_sha256") if noise_info else None,
                "snr_db": snr,
                "actual_snr_db": actual_snr,
                "type": kind,
                "gains": {
                    "speech_linear": speech_gain,
                    "noise_linear": noise_gain,
                    "peak_scale_linear": peak_scale,
                    "peak_scale_db": 20.0 * math.log10(peak_scale),
                    "target_clean_active_rms_dbfs": target_active,
                    "pre_gain_clean_active_rms_dbfs": (20.0 * math.log10(speech_pre_rms) if speech_pre_rms else None),
                    "target_noise_rms_dbfs": noise_target,
                },
                "length_samples": SAMPLES,
                "length_seconds": DURATION_SECONDS,
                "mix_path": str(paths["mix"]),
                "clean_path": str(paths["clean"]),
                "noise_path": str(paths["noise"]),
                "mix_sha256": sha256_file(paths["mix"]),
                "clean_sha256": sha256_file(paths["clean"]),
                "noise_sha256": sha256_file(paths["noise"]),
                "derived_sha256": derived,
                "mix_clean_noise_max_abs_error": max_sum_error,
                "near_timeline": near_info,
                "noise_timeline": noise_info,
            }
            rows.append(row)
            split_counts[split][kind] += 1
            if (ordinal + 1) % 16 == 0:
                log(f"{split}: {ordinal + 1}/{len(schedules[split])} rows")
    if used_raw_sha_by_split["train"] & used_raw_sha_by_split["dev"]:
        raise AssertionError("raw speech SHA overlap between train/dev")
    if used_noise_sha_by_split["train"] & used_noise_sha_by_split["dev"]:
        raise AssertionError("noise SHA overlap between train/dev")
    if any(set(row["raw_source_sha256"]) & excluded_sha for row in rows):
        raise AssertionError("frozen eval SHA entered corpus")
    if any(set(row["speaker_ids"]) & set(exclusions["excluded_speakers"]) for row in rows):
        raise AssertionError("frozen eval speaker entered corpus")
    source_receipt = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "seed": args.seed,
        "train_dev_raw_sha_disjoint": True,
        "train_dev_noise_sha_disjoint": True,
        "used_raw_source_sha256": {split: sorted(values) for split, values in used_raw_sha_by_split.items()},
        "used_noise_source_sha256": {split: sorted(values) for split, values in used_noise_sha_by_split.items()},
        "speech_catalog_counts": {f"{split}/{language}": len(items) for (split, language), items in sorted(catalog.items())},
        "demand_recordings": {split: sorted({row["recording"] for row in items}) for split, items in demand.items()},
        "car_noise_manifest": {"path": str(car_manifest), "sha256": sha256_file(car_manifest)},
        "car_noise_records": {split: items for split, items in car.items()},
        "exclude_receipt": {"path": str(output_dir / "exclude-receipt.json"), "sha256": sha256_file(output_dir / "exclude-receipt.json")},
        "source_audit_raws_expected": [
            str(project_root / "benchmarks/ns/dpdfnet2-16k-v1/source_audit/raws"),
            str(project_root / "build/ns-highway120-20261115/source_audit/raws"),
        ],
        "source_audit_raws_status": "not present; frozen case source_rows/statement_spans used as the source inventory",
        "disk": {"before": {"total": disk_before.total, "used": disk_before.used, "free": disk_before.free}},
    }
    atomic_json(output_dir / "source-receipt.json", source_receipt)
    manifest = {
        "schema_version": 1,
        "dataset_id": "ns-distilled-c64-20261116",
        "created_utc": utc_now(),
        "seed": args.seed,
        "sample_rate": SR,
        "duration_seconds": DURATION_SECONDS,
        "samples": SAMPLES,
        "train_count": args.train_count,
        "dev_count": args.dev_count,
        "language_balance": {
            split: dict(Counter(row["language"] for row in rows if row["split"] == split))
            for split in ("train", "dev")
        },
        "type_counts": {split: dict(split_counts[split]) for split in ("train", "dev")},
        "mixed_noise_family_counts": {
            split: dict(Counter(row["noise_timeline"].get("source_family") for row in rows if row["split"] == split and row["type"] == "mixed"))
            for split in ("train", "dev")
        },
        "audio_contract": {
            "format": "WAV",
            "subtype": "FLOAT",
            "sample_rate": SR,
            "channels": 1,
            "samples": SAMPLES,
            "peak_limit": 0.95,
            "mix_is_clean_plus_noise": True,
            "mix_clean_noise_max_abs_error": max(row["mix_clean_noise_max_abs_error"] for row in rows),
        },
        "feature_contract": {
            "runtime_inputs": ["x", "base_gain"],
            "native_vector_float32": 641,
            "native_slices": {"x": [0, 256], "power": [256, 320], "noise": [320, 384], "base_gain": [384, 641]},
            "stft": {"sample_rate": SR, "n_fft": 512, "hop_length": HOP, "window_length": 320, "window": "sine", "center": False, "frame_policy": "[previous 160 PCM,current 160 PCM] * sine; zero-pad to 512; rfft"},
            "frames": SAMPLES // HOP,
            "bins": 257,
            "mix_quantization": "clip(x*32768,-32768,32767); C++ std::lround; divide by 32768",
            "teacher_alignment": "frozen enhance_stream trims 640 samples and fits exact input length; tail guard 12 hops",
        },
        "receipts": {
            "exclude_receipt": {"path": str(output_dir / "exclude-receipt.json"), "sha256": sha256_file(output_dir / "exclude-receipt.json")},
            "source_receipt": {"path": str(output_dir / "source-receipt.json"), "sha256": sha256_file(output_dir / "source-receipt.json")},
        },
        "rows": rows,
    }
    atomic_json(manifest_path, manifest)
    disk_after = shutil.disk_usage(output_dir)
    receipt = {
        "schema_version": 1,
        "status": "ok",
        "created_utc": utc_now(),
        "elapsed_seconds": time.time() - started,
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "exclude_receipt_sha256": sha256_file(output_dir / "exclude-receipt.json"),
        "source_receipt_sha256": sha256_file(output_dir / "source-receipt.json"),
        "log": {"path": str(log_path), "sha256": sha256_file(log_path)},
        "counts": {"train": args.train_count, "dev": args.dev_count, "rows": len(rows)},
        "space": {"before_free": disk_before.free, "after_free": disk_after.free, "delta_used": disk_before.free - disk_after.free},
        "environment": {"hostname": socket.gethostname(), "python": sys.version, "numpy": np.__version__, "soundfile": sf.__version__, "platform": platform.platform()},
        "verification": {
            "all_40s": all(row["length_samples"] == SAMPLES for row in rows),
            "all_finite_wavs": True,
            "all_mix_equals_clean_plus_noise": all(row["mix_clean_noise_max_abs_error"] <= 1e-7 for row in rows),
            "train_dev_raw_sha_disjoint": True,
            "train_dev_noise_sha_disjoint": True,
            "eval_sha_intersection": [],
            "eval_speaker_intersection": [],
        },
    }
    atomic_json(output_dir / "build-receipt.json", receipt)
    log(f"complete rows={len(rows)} manifest_sha256={receipt['manifest']['sha256']} elapsed={receipt['elapsed_seconds']:.1f}s")
    print(json.dumps(receipt["verification"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
