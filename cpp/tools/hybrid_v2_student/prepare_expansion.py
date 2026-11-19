#!/usr/bin/env python3
"""Build the v2 Chinese supervised NS expansion without touching frozen v1.

The new train/dev corpus uses only Common Voice zh CC0 human originals from
Qwen3-ASR-Car-50h.  A separate 16-speaker test timeline is emitted as cases and
never enters the training manifest.  New CAVEMOVE s110_w0 recordings are fetched
with ZIP HTTP Range reads; the full archive is never downloaded.
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
import struct
import subprocess
import sys
import tempfile
import time
import unicodedata
import uuid
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np
import soundfile as sf


SR = 16_000
TRAIN_SECONDS = 40
TEST_SECONDS = 24
FRAME = 320
HOP = 160
WINDOW = 320
N_FFT = 512
BINS = 257
FEATURE_WIDTH = 641
FRAMES = TRAIN_SECONDS * SR // HOP
EPS = 1e-12
SEED = 20261115
SNRS = (0.0, 10.0, 20.0)
EXPECTED_PROBE_SHA256 = "4d1b3f139214bfad11a6c80e635635026dd2b0c3ce98fc5c4992c8ea4ce5209a"
EXPECTED_OLD_FEATURE_MANIFEST_SHA256 = (
    "ae676c55783a561f409cabe2f686e0f954e4e6373f826529af3eac3beb84408c"
)

TARGET_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TARGET_ROOT.parents[1]
OLD_ROOT = PROJECT_ROOT / "build/ns-distilled-c64-20261116"
OLD_FEATURE_MANIFEST = OLD_ROOT / "features/manifest.json"
OLD_SOURCE_RECEIPT = OLD_ROOT / "data/source-receipt.json"
OLD_CAR_RECORDS = OLD_ROOT / "data/raw-car-noise/records.json"
OLD_PROBE = OLD_ROOT / "native/residual_ns_probe"
V1_ROOT = PROJECT_ROOT / "benchmarks/ns/dpdfnet2-16k-v1"
V120_ROOT = PROJECT_ROOT / "build/ns-highway120-20261115"
DETAIL_ROOT = Path("/shared/asr/Qwen3-ASR-Car-50h/qwen3_asr")
DEMAND_ROOT = PROJECT_ROOT / "datasets/itu-eval-v4-curated/sources/noise"
CAVEMOVE_URL = (
    "https://zenodo.org/api/records/17404917/files/cavemove_dataset.zip/content"
)
CAVEMOVE_ARCHIVE_LENGTH = 7_613_945_771
CAVEMOVE_CENTRAL_DIRECTORY = (
    PROJECT_ROOT
    / "build/dataset-quality-remediation-20261017/source-evidence/cavemove/zip-members.json"
)

DEMAND_DEV_RECORDINGS = {"TBUS", "PCAFETER", "OHALLWAY"}
DEMAND_EXCLUDED_RECORDINGS = {"TCAR"}

# One representative representation per vehicle, all at a new s110_w0 take.
# Honda and Golf are fetched first because together they are sufficient for the
# 16-case balanced test; Alfa and Smart are then attempted as extra coverage.
S110_TARGETS = (
    {
        "vehicle": "Honda_CR-V",
        "member": "cavemove_dataset/Honda_CR-V/distributed/noise/s110_w0.wav",
        "representation": "distributed",
    },
    {
        "vehicle": "Volkswagen_Golf",
        "member": "cavemove_dataset/Volkswagen_Golf/distributed/noise/s110_w0.wav",
        "representation": "distributed",
    },
    {
        "vehicle": "AlfaRomeo_146",
        "member": "cavemove_dataset/AlfaRomeo_146/distributed/noise/s110_w0.wav",
        "representation": "distributed",
    },
    {
        "vehicle": "Smart_forfour",
        "member": "cavemove_dataset/Smart_forfour/array/noise/s110_w0.wav",
        "representation": "array",
    },
)

_feature_worker_config: dict[str, Any] = {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_float_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype="<f4").tobytes()).hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    atomic_write_bytes(path, payload)


def json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_mono(path: Path, *, channel: int = 0) -> tuple[np.ndarray, int, int]:
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if int(rate) != SR:
        raise ValueError(f"expected 16 kHz source, got {rate}: {path}")
    if audio.shape[1] <= channel:
        raise ValueError(f"channel {channel} missing in {path} ({audio.shape[1]} channels)")
    samples = np.ascontiguousarray(audio[:, channel], dtype=np.float32)
    if not samples.size or not np.all(np.isfinite(samples)):
        raise ValueError(f"empty or non-finite source: {path}")
    return samples, int(audio.shape[1]), int(rate)


def write_float_wav(path: Path, samples: np.ndarray, expected_samples: int | None = None) -> None:
    value = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
    if expected_samples is not None and value.size != expected_samples:
        raise ValueError(f"unexpected WAV length {value.size}, expected {expected_samples}: {path}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"non-finite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), value, SR, subtype="FLOAT")


def pcm16_lround(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(samples, dtype=np.float32)
    scaled = np.clip(value * np.float32(32768.0), np.float32(-32768.0), np.float32(32767.0))
    quantized = np.copysign(np.floor(np.abs(scaled) + np.float32(0.5)), scaled).astype(np.int16)
    decoded = quantized.astype(np.float32) / 32768.0
    return quantized, decoded


def pcm16_self_test() -> None:
    values = np.array([-1.5, -0.5, 0.0, 0.5, 1.5], dtype=np.float32) / np.float32(32768.0)
    expected = np.array([-2, -1, 0, 1, 2], dtype=np.int16)
    actual, _ = pcm16_lround(values)
    if not np.array_equal(actual, expected):
        raise AssertionError(f"PCM16 ties-away fixture failed: {actual.tolist()}")


def rms_dbfs(samples: np.ndarray) -> float:
    value = np.asarray(samples, dtype=np.float64)
    rms = math.sqrt(float(np.mean(value * value)) + EPS)
    return 20.0 * math.log10(max(rms, 1e-12))


def active_rms(samples: np.ndarray) -> tuple[float, dict[str, Any]]:
    value = np.asarray(samples, dtype=np.float64)
    if value.size % FRAME:
        raise ValueError(f"active RMS requires a multiple of {FRAME} samples")
    frames = value.reshape(-1, FRAME)
    frame_rms = np.sqrt(np.mean(frames * frames, axis=1) + EPS)
    positive = frame_rms[frame_rms > 1e-12]
    if not positive.size:
        raise ValueError("timeline contains no active frames")
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
        "threshold_dbfs": 20.0 * math.log10(max(threshold, 1e-12)),
        "active_frames": int(np.sum(active)),
        "active_seconds": float(np.sum(active) * 0.02),
    }


def normalize_clean(samples: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float, float, dict[str, Any]]:
    base = np.asarray(samples, dtype=np.float32)
    targets = [float(rng.uniform(-32.0, -26.0))] + [-30.0, -28.0, -26.0, -34.0]
    last: tuple[np.ndarray, float, float, dict[str, Any]] | None = None
    for target in targets:
        current, activity = active_rms(base)
        gain = (10.0 ** (target / 20.0)) / max(current, 1e-12)
        output = np.asarray(base * gain, dtype=np.float32)
        peak = float(np.max(np.abs(output)))
        if peak > 0.95:
            peak_scale = 0.95 / peak
            output = np.asarray(output * peak_scale, dtype=np.float32)
            gain *= peak_scale
        final_active, final_activity = active_rms(output)
        final_db = 20.0 * math.log10(max(final_active, 1e-12))
        last = (output, gain, final_db, {**final_activity, "target_dbfs": target, "peak": float(np.max(np.abs(output)))})
        if -36.0 <= final_db <= -24.0:
            return last
    assert last is not None
    raise ValueError(f"could not place clean active RMS in [-36,-24] dBFS: {last[2]:.3f}")


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
    if not source.size:
        raise ValueError("cannot tile empty noise")
    if source.size >= length:
        return source[:length].copy(), {
            "looped": False,
            "loop_count": 0,
            "crossfade_samples": 0,
            "same_recording_take": True,
        }
    crossfade = min(crossfade, source.size // 4)
    output = source.copy()
    loops = 0
    if crossfade <= 0:
        repeats = int(math.ceil(length / source.size))
        return np.tile(source, repeats)[:length].copy(), {
            "looped": True,
            "loop_count": repeats - 1,
            "crossfade_samples": 0,
            "same_recording_take": True,
        }
    fade = np.linspace(0.0, 1.0, crossfade, endpoint=False, dtype=np.float32)
    while output.size < length:
        overlap = output[-crossfade:] * (1.0 - fade) + source[:crossfade] * fade
        output[-crossfade:] = overlap
        output = np.concatenate([output, source[crossfade:]])
        loops += 1
        if loops > 1000:
            raise RuntimeError("crossfade tiler did not terminate")
    return np.ascontiguousarray(output[:length]), {
        "looped": True,
        "loop_count": loops,
        "crossfade_samples": crossfade,
        "same_recording_take": True,
        "loop_policy": "same recording take tiled with linear equal-time crossfade",
    }


def sample_noise(record: dict[str, Any], length: int, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    source, channels, rate = read_mono(Path(record["mono_path"]), channel=0)
    if source.size >= length:
        start = int(rng.integers(0, source.size - length + 1))
        base = source[start:start + length].copy()
        loop_info = {"looped": False, "loop_count": 0, "slice_start_sample": start}
    else:
        base, loop_info = tile_with_crossfade(source, length)
        loop_info["slice_start_sample"] = 0
    modulation, modulation_info = slow_amplitude(length, rng)
    base *= modulation
    return np.asarray(base, dtype=np.float32), {
        **loop_info,
        "source_frames": int(source.size),
        "source_channels": channels,
        "source_rate": rate,
        "slow_amplitude": modulation_info,
        "same_recording_take": True,
        "synthetic_loop_is_not_a_new_recording": True,
    }


def normalized_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value or "").lower()
    return "".join(char for char in text if char.isalnum())


def source_row(split: str, detail: dict[str, Any]) -> dict[str, Any]:
    audio_path = Path(detail["output_audio"]).resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    text = detail.get("text_original") or detail.get("text_raw") or ""
    return {
        "split": split,
        "id": detail["id"],
        "dataset": detail["dataset"],
        "language": detail["language"],
        "usage": detail["usage"],
        "license": detail["license"],
        "speaker_id": detail["speaker_id"],
        "session_id": detail.get("session_id"),
        "source_id": detail["source_id"],
        "source_origin": detail.get("source"),
        "text": text,
        "text_normalized": normalized_text(text),
        "audio_path": str(audio_path),
        "audio_cleaning": detail.get("audio_cleaning", {}),
        "cross_asr": detail.get("cross_asr", {}),
        "text_qc_flags": detail.get("text_qc_flags", []),
        "subset_version": detail.get("subset_version"),
        "audio_materialization": detail.get("audio_materialization"),
    }


def load_speech_sources() -> dict[str, list[dict[str, Any]]]:
    expected = {"train": (765, 456, "original"), "dev": (87, 78, "evaluation"), "test": (74, 74, "evaluation")}
    output: dict[str, list[dict[str, Any]]] = {}
    for split, (count, speakers, usage) in expected.items():
        path = DETAIL_ROOT / f"{split}_detail.jsonl"
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            detail = json.loads(line)
            if (
                detail.get("language") == "zh"
                and detail.get("dataset") == "CV-zh"
                and detail.get("license") == "CC0-1.0"
                and detail.get("usage") == usage
                and detail.get("target_split") == split
            ):
                rows.append(source_row(split, detail))
        if len(rows) != count or len({row["speaker_id"] for row in rows}) != speakers:
            raise ValueError(
                f"CV-zh source contract changed for {split}: rows={len(rows)}, "
                f"speakers={len({row['speaker_id'] for row in rows})}"
            )
        output[split] = rows
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        if {row["speaker_id"] for row in output[left]} & {row["speaker_id"] for row in output[right]}:
            raise AssertionError(f"CV-zh speaker overlap between {left}/{right}")
        if {row["text_normalized"] for row in output[left]} & {row["text_normalized"] for row in output[right]}:
            raise AssertionError(f"CV-zh normalized text overlap between {left}/{right}")
    return output


def walk_json(path: Path) -> Iterable[Any]:
    try:
        yield json_file(path)
    except Exception:
        return


def collect_source_exclusions() -> dict[str, Any]:
    speakers: set[str] = set()
    shas: set[str] = set()
    tokens: set[str] = set()
    evidence: list[dict[str, str]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                lowered = key.lower()
                if lowered in {"speaker_id", "speaker", "near_speaker", "far_speaker"} and item is not None:
                    token = str(item).rsplit(":", 1)[-1]
                    if re.fullmatch(r"[A-Za-z0-9_.:-]+", token):
                        speakers.add(str(item))
                        speakers.add(token)
                if lowered in {"sha256", "sha", "source_sha256", "raw_sha256", "rawsha", "monosha"} and isinstance(item, str):
                    if re.fullmatch(r"[0-9a-fA-F]{64}", item):
                        shas.add(item.lower())
                if lowered in {"noise_id", "noise_recording", "recording_id", "member", "path", "filename"} and isinstance(item, str):
                    tokens.add(item)
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    old_feature = json_file(OLD_FEATURE_MANIFEST)
    if sha256_file(OLD_FEATURE_MANIFEST) != EXPECTED_OLD_FEATURE_MANIFEST_SHA256:
        raise ValueError("old frozen feature manifest SHA changed")
    visit(old_feature)
    evidence.append({"path": str(OLD_FEATURE_MANIFEST), "sha256": sha256_file(OLD_FEATURE_MANIFEST)})
    for root, pattern in ((V1_ROOT / "cases", "*.json"), (V120_ROOT / "data", "*.json")):
        for path in sorted(root.glob(pattern)):
            for document in walk_json(path):
                visit(document)
            evidence.append({"path": str(path), "sha256": sha256_file(path)})
    extra = [
        V1_ROOT / "manifest.json",
        V1_ROOT / "provenance.json",
        V120_ROOT / "sources/source-manifest.json",
        OLD_SOURCE_RECEIPT,
    ]
    for path in extra:
        if path.is_file():
            for document in walk_json(path):
                visit(document)
            evidence.append({"path": str(path), "sha256": sha256_file(path)})
    return {
        "schema_version": 1,
        "created_utc": utc_now(),
        "scope": "all frozen v1/v120 source speakers and raw SHA plus legacy MagicData rows",
        "evidence": evidence,
        "excluded_speakers": sorted(speakers),
        "excluded_raw_sha256": sorted(shas),
        "excluded_tokens": sorted(tokens),
        "policy": "fail closed on any selected speaker or materialized-original SHA intersection",
    }


def assess_source_quality(source: dict[str, Any]) -> dict[str, Any]:
    info = sf.info(source["audio_path"])
    if int(info.samplerate) != SR or int(info.channels) != 1:
        raise ValueError(f"CV-zh materialized source is not mono 16 kHz: {source['audio_path']}")
    samples, channels, rate = read_mono(Path(source["audio_path"]))
    frame_count = samples.size // FRAME
    if frame_count:
        frames = samples[: frame_count * FRAME].astype(np.float64).reshape(frame_count, FRAME)
        frame_rms = np.sqrt(np.mean(frames * frames, axis=1) + EPS)
        positive = frame_rms[frame_rms > 1e-12]
        noise_floor = float(np.percentile(positive, 10.0)) if positive.size else 0.0
        p50 = float(np.percentile(positive, 50.0)) if positive.size else 0.0
        p90 = float(np.percentile(positive, 90.0)) if positive.size else 0.0
    else:
        noise_floor = p50 = p90 = 0.0
    spectrum = np.abs(np.fft.rfft(samples[: min(samples.size, SR * 10)].astype(np.float64))) ** 2
    freqs = np.fft.rfftfreq(min(samples.size, SR * 10), d=1.0 / SR)
    centroid = float(np.sum(freqs * spectrum) / max(float(np.sum(spectrum)), EPS))
    cleaning = source.get("audio_cleaning", {})
    original_attribution = (
        source["dataset"] == "CV-zh"
        and source["license"] == "CC0-1.0"
        and source["usage"] in {"original", "evaluation"}
        and isinstance(source.get("source_origin"), dict)
        and source["source_origin"].get("kind") == "file"
    )
    return {
        "id": source["id"],
        "split": source["split"],
        "path": source["audio_path"],
        "sha256": sha256_file(Path(source["audio_path"])),
        "header": {
            "format": info.format,
            "subtype": info.subtype,
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
            "frames": int(info.frames),
            "seconds": float(info.frames / info.samplerate),
        },
        "descriptive_acoustics": {
            "peak": float(np.max(np.abs(samples))),
            "rms_dbfs": rms_dbfs(samples),
            "dc_offset": float(np.mean(samples, dtype=np.float64)),
            "clipped_fraction": float(np.mean(np.abs(samples) >= 0.999)),
            "frame_rms_p10_dbfs": 20.0 * math.log10(max(noise_floor, 1e-12)),
            "frame_rms_p50_dbfs": 20.0 * math.log10(max(p50, 1e-12)),
            "frame_rms_p90_dbfs": 20.0 * math.log10(max(p90, 1e-12)),
            "spectral_centroid_hz_first_10s": centroid,
            "diagnostic_only": True,
        },
        "source_attribution": {
            "dataset": source["dataset"],
            "usage": source["usage"],
            "license": source["license"],
            "source_kind": source["source_origin"].get("kind"),
            "original_file_path": source["source_origin"].get("path"),
            "human_original_confirmed": original_attribution,
            "augmentation_or_replay": False,
            "car_noise_mix_claimed_by_name": False,
        },
        "materialization": {
            "cleaning": cleaning,
            "audio_materialization": source.get("audio_materialization"),
            "judgment": (
                "CV-zh human original/evaluation source attributed by detail metadata; "
                "materialization contains VAD/loudness normalization, not denoising, "
                "speech enhancement, augmentation, replay simulation, or car-noise mixing"
            ),
        },
        "cross_asr": source.get("cross_asr"),
        "text_qc_flags": source.get("text_qc_flags"),
    }


def choose_timeline(
    catalog: list[dict[str, Any]],
    rng: np.random.Generator,
    *,
    seconds: int,
    primary_speakers: list[str],
    require_all_primary: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    samples_total = seconds * SR
    if not catalog:
        raise ValueError("empty speech catalog")
    primary_set = set(primary_speakers)
    if not primary_set or not any(row["speaker_id"] in primary_set for row in catalog):
        raise ValueError("primary speaker is absent from catalog")
    order = list(catalog)
    rng.shuffle(order)
    order.sort(key=lambda row: 0 if row["speaker_id"] in primary_set else 1)
    lead = int(round(float(rng.uniform(0.05, 0.50)) * SR))
    minimum_frames = min(int(sf.info(row["audio_path"]).frames) for row in order)
    first = next(
        (
            row
            for row in order
            if row["speaker_id"] in primary_set
            and int(sf.info(row["audio_path"]).frames) + lead + int(8.0 * SR) + minimum_frames <= samples_total
        ),
        None,
    )
    if first is None:
        raise ValueError("no primary statement permits a second complete statement and long pause")
    order.remove(first)
    order.insert(0, first)
    timeline = [np.zeros(lead, dtype=np.float32)]
    used: list[dict[str, Any]] = []
    cursor = lead
    used_speakers: set[str] = set()
    for index, row in enumerate(order):
        gap_seconds = float(rng.uniform(3.0, 8.0)) if index == 0 else float(rng.uniform(0.10, 1.00))
        gap = int(round(gap_seconds * SR))
        info = sf.info(row["audio_path"])
        frames = int(info.frames)
        if cursor + gap + frames + int(0.05 * SR) > samples_total:
            continue
        waveform, _, _ = read_mono(Path(row["audio_path"]))
        if waveform.size != frames:
            raise ValueError(f"source frame count changed: {row['audio_path']}")
        start = cursor + gap
        timeline.extend((np.zeros(gap, dtype=np.float32), waveform))
        used.append(
            {
                "id": row["id"],
                "path": row["audio_path"],
                "speaker_id": row["speaker_id"],
                "source_id": row["source_id"],
                "text": row["text"],
                "text_normalized": row["text_normalized"],
                "frames": frames,
                "seconds": frames / SR,
                "sha256": sha256_file(Path(row["audio_path"])),
                "output_start_sample": start,
                "output_end_sample": start + frames,
                "gap_before_samples": gap,
                "gap_before_seconds": gap_seconds,
                "complete_statement_preserved": True,
            }
        )
        used_speakers.add(row["speaker_id"])
        cursor += gap + frames
        if cursor >= samples_total - int(0.20 * SR):
            break
    if len(used) < 2:
        raise ValueError("timeline could not fit two complete statements")
    if not any(3.0 <= float(row["gap_before_seconds"]) <= 8.0 for row in used):
        raise AssertionError("mandatory 3-8 s pause is missing")
    if not primary_set & used_speakers:
        raise AssertionError("primary speaker was not included")
    if require_all_primary and not primary_set.issubset(used_speakers):
        raise ValueError(f"timeline omitted required primary speakers: {sorted(primary_set - used_speakers)}")
    trailing = samples_total - cursor
    if trailing < 0:
        raise AssertionError("timeline overflow")
    timeline.append(np.zeros(trailing, dtype=np.float32))
    clean = np.concatenate(timeline).astype(np.float32, copy=False)
    if clean.size != samples_total:
        raise AssertionError("timeline length mismatch")
    return clean, {
        "policy": "complete CV-zh statements; no statement cropping; same-split cross-speaker concatenation allowed",
        "statement_count": len(used),
        "speaker_count": len(used_speakers),
        "statements": used,
        "leading_silence_samples": lead,
        "trailing_silence_samples": trailing,
        "short_gap_seconds": [0.10, 1.00],
        "long_gap_seconds": [3.0, 8.0],
        "long_pause_present": True,
        "weak_onsets_and_tails": "source waveform retained in full; no hard onset/tail crop",
    }


def load_old_car_catalog() -> dict[str, list[dict[str, Any]]]:
    document = json_file(OLD_CAR_RECORDS)
    recordings = document.get("recordings")
    if not isinstance(recordings, list) or len(recordings) != 4:
        raise ValueError("old CAVEMOVE s100 records contract changed")
    pool: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    for item in recordings:
        split = str(item["split"])
        mono_path = Path(item["mono_path"])
        if not mono_path.is_file():
            raise FileNotFoundError(mono_path)
        if sha256_file(mono_path) != item["monoSHA"]:
            raise AssertionError(f"old car mono SHA changed: {mono_path}")
        raw_path = Path(item["raw_path"])
        if sha256_file(raw_path) != item["rawSHA"]:
            raise AssertionError(f"old car raw SHA changed: {raw_path}")
        pool[split].append(
            {
                "family": "car",
                "recording": item["recording_id"],
                "vehicle": item["vehicle"],
                "representation": item["representation"],
                "take_id": item["take_id"],
                "mono_path": str(mono_path),
                "raw_path": str(raw_path),
                "raw_sha256": item["rawSHA"],
                "mono_sha256": item["monoSHA"],
                "compressed_sha256": item.get("compressedSHA"),
                "crc32": item.get("crc32"),
            }
        )
    if len(pool["train"]) != 3 or len(pool["dev"]) != 1:
        raise ValueError(f"unexpected old car split counts: {pool}")
    return pool


def load_demand_catalog(excluded_sha: set[str]) -> dict[str, list[dict[str, Any]]]:
    pool: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    for path in sorted(DEMAND_ROOT.glob("*-ch01.wav")) + sorted(DEMAND_ROOT.glob("MS-SNSD-*.wav")):
        recording = path.stem[:-5] if path.stem.endswith("-ch01") else path.stem
        if recording in DEMAND_EXCLUDED_RECORDINGS:
            continue
        if recording.startswith("MS-SNSD-"):
            recording = "HVAC"
        sha = sha256_file(path)
        if sha in excluded_sha:
            continue
        info = sf.info(str(path))
        if int(info.samplerate) != SR or int(info.channels) != 1:
            raise ValueError(f"DEMAND source is not mono 16 kHz: {path}")
        split = "dev" if recording in DEMAND_DEV_RECORDINGS else "train"
        pool[split].append(
            {
                "family": "demand",
                "recording": recording,
                "vehicle": None,
                "representation": "ambient",
                "take_id": recording,
                "mono_path": str(path.resolve()),
                "raw_path": str(path.resolve()),
                "raw_sha256": sha,
                "mono_sha256": sha,
                "compressed_sha256": None,
                "crc32": None,
            }
        )
    if not pool["train"] or not pool["dev"]:
        raise ValueError(f"DEMAND split contract failed: {pool}")
    return pool


def noise_pools(excluded_sha: set[str]) -> dict[str, list[dict[str, Any]]]:
    old_car = load_old_car_catalog()
    # The requested expansion deliberately reuses the old fixed DEMAND train/dev
    # recordings; their hashes are present in the legacy manifest and must not
    # be mistaken for forbidden speech-source hashes.
    demand = load_demand_catalog(set())
    output = {
        split: sorted(old_car[split] + demand[split], key=lambda row: (row["family"], row["recording"]))
        for split in ("train", "dev")
    }
    train_sha = {row["raw_sha256"] for row in output["train"]} | {row["mono_sha256"] for row in output["train"]}
    dev_sha = {row["raw_sha256"] for row in output["dev"]} | {row["mono_sha256"] for row in output["dev"]}
    if train_sha & dev_sha:
        raise AssertionError("train/dev noise SHA overlap")
    return output


def schedule_specs(split: str) -> list[dict[str, Any]]:
    if split == "train":
        kinds = ["mixed"] * 26 + ["clean"] * 4 + ["noise_only"] * 2
    elif split == "dev":
        kinds = ["mixed"] * 6 + ["clean"] + ["noise_only"]
    else:
        raise ValueError(f"unsupported train/dev split: {split}")
    return [{"type": kind} for kind in kinds]


def primary_assignments(split: str, speakers: list[str]) -> list[list[str]]:
    if split == "train":
        return [[speaker] for speaker in speakers[:26]] + [
            [speakers[26], speakers[27]],
            [speakers[28], speakers[29]],
            [speakers[30]],
            [speakers[31]],
            [],
            [],
        ]
    if split == "dev":
        return [[speaker] for speaker in speakers[:6]] + [[speakers[6], speakers[7]], []]
    raise ValueError(split)


def choose_primary_speakers(catalog: list[dict[str, Any]], count: int, rng: np.random.Generator) -> list[str]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in catalog:
        groups[row["speaker_id"]].append(row)
    speakers = sorted(groups, key=lambda speaker: hashlib.sha256(speaker.encode()).hexdigest())
    rng.shuffle(speakers)
    if len(speakers) < count:
        raise ValueError(f"need {count} distinct speakers, only {len(speakers)} available")
    return speakers[:count]


def build_train_dev_rows(
    sources: dict[str, list[dict[str, Any]]],
    noise: dict[str, list[dict[str, Any]]],
    audio_root: Path,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("train", "dev"):
        catalog = sources[split]
        needed_speakers = 32 if split == "train" else 8
        selected_primary = choose_primary_speakers(catalog, needed_speakers, rng)
        specs = schedule_specs(split)
        assignments = primary_assignments(split, selected_primary)
        if len(specs) != len(assignments):
            raise AssertionError("schedule and speaker assignment length mismatch")
        noise_queue = list(noise[split])
        rng.shuffle(noise_queue)
        noise_cursor = 0
        for ordinal, (spec, primaries) in enumerate(zip(specs, assignments)):
            kind = spec["type"]
            row_id = f"v2-{split}-{ordinal:03d}-zh-{kind}"
            samples_total = TRAIN_SECONDS * SR
            clean = np.zeros(samples_total, dtype=np.float32)
            noise_wave = np.zeros(samples_total, dtype=np.float32)
            near_info = None
            speaker_ids: list[str] = []
            if kind in {"mixed", "clean"}:
                clean_raw, near_info = choose_timeline(
                    catalog,
                    rng,
                    seconds=TRAIN_SECONDS,
                    primary_speakers=primaries,
                    require_all_primary=len(primaries) > 1,
                )
                clean, speech_gain, clean_active_db, activity = normalize_clean(clean_raw, rng)
                speaker_ids = sorted({statement["speaker_id"] for statement in near_info["statements"]})
            else:
                speech_gain = 1.0
                clean_active_db = None
                activity = None
            selected_noise = None
            snr_db = None
            measured_snr = None
            noise_gain = 1.0
            noise_info = None
            if kind in {"mixed", "noise_only"}:
                selected_noise = noise_queue[noise_cursor % len(noise_queue)]
                noise_cursor += 1
                base_noise, noise_info = sample_noise(selected_noise, samples_total, rng)
                noise_rms = math.sqrt(float(np.mean(base_noise.astype(np.float64) ** 2)) + EPS)
                if kind == "mixed":
                    snr_db = float(SNRS[int(rng.integers(0, len(SNRS)))])
                    clean_active, _ = active_rms(clean)
                    noise_gain = (clean_active / max(noise_rms, 1e-12)) / (10.0 ** (snr_db / 20.0))
                else:
                    target_noise_db = float(rng.uniform(-34.0, -26.0))
                    noise_gain = (10.0 ** (target_noise_db / 20.0)) / max(noise_rms, 1e-12)
                    noise_info["target_noise_rms_dbfs"] = target_noise_db
                noise_wave = np.asarray(base_noise * noise_gain, dtype=np.float32)
            elif kind == "clean":
                noise_info = {"family": "digital_silence", "same_recording_take": False}
            mix = np.asarray(clean + noise_wave, dtype=np.float32)
            peak = float(max(np.max(np.abs(clean)), np.max(np.abs(noise_wave)), np.max(np.abs(mix))))
            common_scale = 1.0 if peak <= 0.95 else 0.95 / peak
            if common_scale != 1.0:
                clean = np.asarray(clean * common_scale, dtype=np.float32)
                noise_wave = np.asarray(noise_wave * common_scale, dtype=np.float32)
                mix = np.asarray(clean + noise_wave, dtype=np.float32)
            final_clean_active = None
            final_activity = None
            if kind in {"mixed", "clean"}:
                final_clean_active, final_activity = active_rms(clean)
                final_clean_active_db = 20.0 * math.log10(max(final_clean_active, 1e-12))
                if not -36.0 <= final_clean_active_db <= -24.0:
                    raise AssertionError(f"clean active RMS out of range for {row_id}: {final_clean_active_db}")
            else:
                final_clean_active_db = None
            if kind == "mixed":
                noise_rms = math.sqrt(float(np.mean(noise_wave.astype(np.float64) ** 2)) + EPS)
                measured_snr = 20.0 * math.log10(max(final_clean_active, 1e-12) / max(noise_rms, 1e-12))
                if abs(measured_snr - snr_db) > 1e-5:
                    raise AssertionError(f"SNR mismatch for {row_id}: {measured_snr} vs {snr_db}")
            sum_error = float(np.max(np.abs(mix - (clean + noise_wave))))
            if sum_error > 1e-7:
                raise AssertionError(f"mix != clean + noise for {row_id}: {sum_error}")
            row_dir = audio_root / split / row_id
            paths = {name: row_dir / f"{name}.wav" for name in ("mix", "clean", "noise")}
            write_float_wav(paths["mix"], mix, samples_total)
            write_float_wav(paths["clean"], clean, samples_total)
            write_float_wav(paths["noise"], noise_wave, samples_total)
            statements = near_info["statements"] if near_info else []
            source_shas = [statement["sha256"] for statement in statements]
            noise_take_hash = None
            noise_recording = "digital-silence"
            noise_recording_sha = sha256_float_array(np.zeros(samples_total, dtype=np.float32))
            if selected_noise is not None:
                noise_recording = selected_noise["recording"]
                noise_recording_sha = selected_noise["raw_sha256"]
                noise_take_hash = hashlib.sha256(
                    json.dumps(
                        {
                            "recording": selected_noise["recording"],
                            "take_id": selected_noise["take_id"],
                            "raw_sha256": selected_noise["raw_sha256"],
                            "mono_sha256": selected_noise["mono_sha256"],
                        },
                        sort_keys=True,
                    ).encode()
                ).hexdigest()
                noise_info.update(
                    {
                        "source_path": selected_noise["mono_path"],
                        "source_raw_path": selected_noise["raw_path"],
                        "source_raw_sha256": selected_noise["raw_sha256"],
                        "source_mono_sha256": selected_noise["mono_sha256"],
                        "source_recording": selected_noise["recording"],
                        "source_family": selected_noise["family"],
                        "take_id": selected_noise["take_id"],
                    }
                )
            derived = {
                "mix_float32_sha256": sha256_float_array(mix),
                "clean_float32_sha256": sha256_float_array(clean),
                "noise_float32_sha256": sha256_float_array(noise_wave),
                "concat_float32_sha256": hashlib.sha256(
                    np.concatenate([mix, clean, noise_wave]).astype("<f4").tobytes()
                ).hexdigest(),
            }
            rows.append(
                {
                    "id": row_id,
                    "split": split,
                    "language": "zh",
                    "speaker_ids": speaker_ids,
                    "primary_speakers": primaries,
                    "raw_source_paths": [statement["path"] for statement in statements],
                    "raw_source_sha256": source_shas,
                    "raw_sources": statements,
                    "raw_sha_scope": "materialized CV-zh CC0 human-original WAV; upstream MP3 not present",
                    "noise_recording": noise_recording,
                    "noise_recording_sha256": noise_recording_sha,
                    "noise_take_hash": noise_take_hash,
                    "noise_source_path": selected_noise["mono_path"] if selected_noise else None,
                    "noise_source_raw_path": selected_noise["raw_path"] if selected_noise else None,
                    "noise_source_sha256": selected_noise["mono_sha256"] if selected_noise else None,
                    "snr_db": snr_db,
                    "measured_snr_db": measured_snr,
                    "type": kind,
                    "length_samples": samples_total,
                    "length_seconds": TRAIN_SECONDS,
                    "mix_path": str(paths["mix"].resolve()),
                    "clean_path": str(paths["clean"].resolve()),
                    "noise_path": str(paths["noise"].resolve()),
                    "mix_sha256": sha256_file(paths["mix"]),
                    "clean_sha256": sha256_file(paths["clean"]),
                    "noise_sha256": sha256_file(paths["noise"]),
                    "derived_sha256": derived,
                    "mix_clean_noise_max_abs_error": sum_error,
                    "gains": {
                        "speech_linear": speech_gain,
                        "noise_linear": noise_gain,
                        "common_peak_scale": common_scale,
                        "common_peak_scale_db": 20.0 * math.log10(common_scale),
                    },
                    "clean_active_rms_dbfs": final_clean_active_db,
                    "clean_activity": final_activity or activity,
                    "near_timeline": near_info,
                    "noise_timeline": noise_info,
                    "test_used_for_training": False,
                    "newtest_used_for_training": False,
                }
            )
    used_by_split = {
        split: {
            "speakers": {speaker for row in rows if row["split"] == split for speaker in row["speaker_ids"]},
            "source_sha": {sha for row in rows if row["split"] == split for sha in row["raw_source_sha256"]},
            "noise_sha": {
                row["noise_recording_sha256"]
                for row in rows
                if row["split"] == split and row["noise_recording"] != "digital-silence"
            },
        }
        for split in ("train", "dev")
    }
    if used_by_split["train"]["speakers"] & used_by_split["dev"]["speakers"]:
        raise AssertionError("train/dev selected speaker overlap")
    if used_by_split["train"]["source_sha"] & used_by_split["dev"]["source_sha"]:
        raise AssertionError("train/dev source SHA overlap")
    if used_by_split["train"]["noise_sha"] & used_by_split["dev"]["noise_sha"]:
        raise AssertionError("train/dev noise SHA overlap")
    if len({speaker for row in rows if row["split"] == "train" for speaker in row["speaker_ids"]}) < 32:
        raise AssertionError("train expansion has fewer than 32 distinct speakers")
    if len({speaker for row in rows if row["split"] == "dev" for speaker in row["speaker_ids"]}) < 8:
        raise AssertionError("dev expansion has fewer than 8 distinct speakers")
    return rows


def project_band_power(spectrum_power: np.ndarray) -> np.ndarray:
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


def contract_stft(samples: np.ndarray, samples_total: int) -> np.ndarray:
    value = np.asarray(samples, dtype=np.float32).reshape(-1)
    if value.size != samples_total:
        raise ValueError(f"STFT contract requires {samples_total} samples, got {value.size}")
    padded = np.concatenate([np.zeros(HOP, dtype=np.float32), value])
    frame_total = samples_total // HOP
    frames = np.lib.stride_tricks.sliding_window_view(padded, WINDOW)[::HOP][:frame_total]
    if frames.shape != (frame_total, WINDOW):
        raise AssertionError(f"unexpected STFT frame shape: {frames.shape}")
    n = np.arange(WINDOW, dtype=np.float64)
    sine = np.sin(math.pi * (n + 0.5) / WINDOW).astype(np.float32)
    padded_frames = np.zeros((frame_total, N_FFT), dtype=np.float32)
    padded_frames[:, :WINDOW] = frames * sine[np.newaxis, :]
    spectrum = np.fft.rfft(padded_frames, n=N_FFT, axis=1).astype(np.complex64)
    if spectrum.shape != (frame_total, BINS):
        raise AssertionError(f"unexpected STFT shape: {spectrum.shape}")
    return spectrum


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


def feature_worker_init(config: dict[str, Any]) -> None:
    global _feature_worker_config
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    _feature_worker_config = config


def process_feature_row(row: dict[str, Any]) -> dict[str, Any]:
    config = _feature_worker_config
    if not config:
        raise RuntimeError("feature worker was not initialized")
    row_id = row["id"]
    features_dir = Path(config["features_dir"])
    native_input_dir = features_dir / "native-inputs"
    native_dir = features_dir / "native"
    npz_dir = features_dir / "npz"
    for directory in (native_input_dir, native_dir, npz_dir):
        directory.mkdir(parents=True, exist_ok=True)
    mix, _, _ = read_mono(Path(row["mix_path"]))
    clean, _, _ = read_mono(Path(row["clean_path"]))
    noise, _, _ = read_mono(Path(row["noise_path"]))
    samples_total = TRAIN_SECONDS * SR
    if mix.size != samples_total or clean.size != samples_total or noise.size != samples_total:
        raise ValueError(f"feature source length mismatch: {row_id}")
    for key, path_key in (("mix_sha256", "mix_path"), ("clean_sha256", "clean_path"), ("noise_sha256", "noise_path")):
        if sha256_file(Path(row[path_key])) != row[key]:
            raise AssertionError(f"{key} mismatch for {row_id}")
    sum_error = float(np.max(np.abs(mix - (clean + noise))))
    if sum_error > 1e-7:
        raise AssertionError(f"mix != clean + noise for {row_id}: {sum_error}")
    quantized, decoded_mix = pcm16_lround(mix)
    quantized_path = native_input_dir / f"{row_id}.mix-pcm16.wav"
    sf.write(str(quantized_path), quantized, SR, subtype="PCM_16")
    decoded_from_file, rate = sf.read(str(quantized_path), dtype="float32", always_2d=False)
    decoded_from_file = np.asarray(decoded_from_file, dtype=np.float32).reshape(-1)
    if int(rate) != SR or decoded_from_file.size != samples_total:
        raise AssertionError("PCM16 input contract changed")
    if not np.array_equal(decoded_from_file, decoded_mix):
        raise AssertionError("PCM16 WAV roundtrip differs from C++ lround decode")
    probe = Path(config["probe"])
    if sha256_file(probe) != EXPECTED_PROBE_SHA256:
        raise RuntimeError("native probe SHA changed")
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
    native = load_native_vector(native_features)
    x = np.ascontiguousarray(native[:, :256], dtype=np.float32)
    power = np.ascontiguousarray(native[:, 256:320], dtype=np.float32)
    native_noise = np.ascontiguousarray(native[:, 320:384], dtype=np.float32)
    base_gain = np.ascontiguousarray(native[:, 384:641], dtype=np.float32)
    if np.min(power) < 0.0 or np.min(native_noise) < 0.0:
        raise AssertionError(f"negative projected power for {row_id}")
    if np.min(base_gain) < -1e-6 or np.max(base_gain) > 1.000001:
        raise AssertionError(f"base_gain outside [0,1] for {row_id}")
    mix_spec = contract_stft(decoded_from_file, samples_total)
    clean_spec = contract_stft(clean, samples_total)
    mix_band_power = project_band_power(np.abs(mix_spec.astype(np.complex64)) ** 2)
    power_delta = np.abs(mix_band_power.astype(np.float64) - power.astype(np.float64))
    max_power_error = float(np.max(power_delta))
    max_power_relative_error = float(
        np.max(power_delta / np.maximum(np.abs(power.astype(np.float64)), 1.0))
    )
    if max_power_relative_error > 1e-4:
        raise AssertionError(
            f"STFT/native power alignment failed for {row_id}: "
            f"abs={max_power_error} relative={max_power_relative_error}"
        )
    for name, array in {"x": x, "base_gain": base_gain, "power": power, "noise": native_noise}.items():
        if not np.all(np.isfinite(array)):
            raise AssertionError(f"non-finite {name} for {row_id}")
    for name, array in {"mix_spec": mix_spec, "clean_spec": clean_spec}.items():
        if not np.all(np.isfinite(array.real)) or not np.all(np.isfinite(array.imag)):
            raise AssertionError(f"non-finite {name} for {row_id}")
    if x.shape != (FRAMES, 256) or base_gain.shape != (FRAMES, BINS):
        raise AssertionError(f"runtime feature shape mismatch for {row_id}")
    if power.shape != (FRAMES, 64) or native_noise.shape != (FRAMES, 64):
        raise AssertionError(f"projected feature shape mismatch for {row_id}")
    npz_path = npz_dir / f"{row_id}.npz"
    np.savez(
        npz_path,
        x=x,
        base_gain=base_gain,
        mix_spec=mix_spec,
        clean_spec=clean_spec,
        power=power,
        noise=native_noise,
    )
    return {
        **row,
        "npz_path": str(npz_path.resolve()),
        "npz_sha256": sha256_file(npz_path),
        "native_input_path": str(quantized_path.resolve()),
        "native_input_sha256": sha256_file(quantized_path),
        "native_feature_path": str(native_features.resolve()),
        "native_feature_sha256": sha256_file(native_features),
        "frames": FRAMES,
        "samples": samples_total,
        "feature_validation": {
            "mix_clean_noise_max_abs_error": sum_error,
            "stft_native_power_max_abs_error": max_power_error,
            "stft_native_power_max_relative_error": max_power_relative_error,
            "pcm16_lround_roundtrip_exact": True,
            "finite_inputs": True,
            "finite_features": True,
            "teacher_spec_generated": False,
            "supervised_targets": ["mix_spec", "clean_spec"],
        },
    }


def build_new_features(
    rows: list[dict[str, Any]],
    features_dir: Path,
    jobs: int,
    log: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if jobs <= 0:
        raise ValueError("--jobs must be positive")
    pcm16_self_test()
    if sha256_file(OLD_PROBE) != EXPECTED_PROBE_SHA256:
        raise ValueError("old native v1 probe SHA mismatch")
    config = {"probe": str(OLD_PROBE), "features_dir": str(features_dir)}
    results: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=jobs, initializer=feature_worker_init, initargs=(config,)) as executor:
        future_to_id = {executor.submit(process_feature_row, row): row["id"] for row in rows}
        completed_count = 0
        for future in as_completed(future_to_id):
            row_id = future_to_id[future]
            results[row_id] = future.result()
            completed_count += 1
            if completed_count == 1 or completed_count % 4 == 0 or completed_count == len(rows):
                log(f"features {completed_count}/{len(rows)} last={row_id}")
    if len(results) != len(rows):
        raise AssertionError("missing feature results")
    ordered = [results[row["id"]] for row in rows]
    validation = {
        "new_rows": len(ordered),
        "new_train": sum(row["split"] == "train" for row in ordered),
        "new_dev": sum(row["split"] == "dev" for row in ordered),
        "all_frames_4000": all(row["frames"] == FRAMES for row in ordered),
        "all_samples_640000": all(row["samples"] == TRAIN_SECONDS * SR for row in ordered),
        "max_stft_native_power_error": max(row["feature_validation"]["stft_native_power_max_abs_error"] for row in ordered),
        "max_stft_native_power_relative_error": max(
            row["feature_validation"]["stft_native_power_max_relative_error"] for row in ordered
        ),
        "max_mix_clean_noise_error": max(row["feature_validation"]["mix_clean_noise_max_abs_error"] for row in ordered),
        "newtest_rows_in_manifest": 0,
    }
    if validation["new_train"] != 32 or validation["new_dev"] != 8:
        raise AssertionError(f"new feature split counts are wrong: {validation}")
    return ordered, validation


def cachebusted_url(url: str) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["cachebuster"] = uuid.uuid4().hex
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def parse_content_range(value: str | None) -> tuple[int, int, int]:
    if not value:
        raise ValueError("missing Content-Range")
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", value.strip())
    if not match:
        raise ValueError(f"unexpected Content-Range: {value!r}")
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def fetch_range(start: int, end: int, *, purpose: str, temp_dir: Path) -> tuple[bytes, dict[str, Any]]:
    if start < 0 or end < start or end >= CAVEMOVE_ARCHIVE_LENGTH:
        raise ValueError(f"invalid archive Range {start}-{end}")
    requested_size = end - start + 1
    request_url = cachebusted_url(CAVEMOVE_URL)
    requested_range = f"bytes={start}-{end}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".range-", dir=temp_dir) as temporary:
        root = Path(temporary)
        header_path = root / "headers.txt"
        body_path = root / "body.bin"
        command = [
            "curl", "-L", "-sS", "--connect-timeout", "30", "--max-time", "180",
            "--max-filesize", str(requested_size + 1), "--range", f"{start}-{end}",
            "--dump-header", str(header_path), "--output", str(body_path),
            "-w", "%{http_code} %{size_download} %{time_total} %{speed_download}", request_url,
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        header_text = header_path.read_text(errors="replace").replace("\r\n", "\n") if header_path.exists() else ""
        blocks: list[list[str]] = []
        current: list[str] = []
        for line in header_text.splitlines():
            if line.startswith("HTTP/"):
                if current:
                    blocks.append(current)
                current = [line]
            elif current:
                current.append(line)
        if current:
            blocks.append(current)
        status: int | None = None
        headers: dict[str, str] = {}
        for block in reversed(blocks):
            match = re.match(r"HTTP/\d(?:\.\d)? (\d+)", block[0]) if block else None
            if match and int(match.group(1)) == 206:
                status = 206
                for line in block[1:]:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        headers[key.strip().lower()] = value.strip()
                break
        if completed.returncode != 0:
            raise RuntimeError(f"{purpose}: curl exit {completed.returncode}: {completed.stderr.strip()}")
        if status != 206:
            raise RuntimeError(f"{purpose}: required HTTP 206, got {completed.stdout.strip()!r}")
        content_range = headers.get("content-range")
        if parse_content_range(content_range) != (start, end, CAVEMOVE_ARCHIVE_LENGTH):
            raise RuntimeError(f"{purpose}: Content-Range mismatch: {content_range!r}")
        declared_length = headers.get("content-length")
        if declared_length is not None and int(declared_length) != requested_size:
            raise RuntimeError(f"{purpose}: Content-Length {declared_length} != {requested_size}")
        if headers.get("content-encoding") not in (None, "identity"):
            raise RuntimeError(f"{purpose}: unexpected Content-Encoding {headers.get('content-encoding')!r}")
        payload = body_path.read_bytes()
        if len(payload) != requested_size:
            raise RuntimeError(f"{purpose}: read {len(payload)} bytes, expected {requested_size}")
        metrics = completed.stdout.split()
        event = {
            "purpose": purpose,
            "requested_url": request_url,
            "requested_range": requested_range,
            "status": 206,
            "content_range": content_range,
            "content_length": declared_length,
            "content_encoding": headers.get("content-encoding"),
            "bytes_received": int(float(metrics[1])),
            "elapsed_seconds": round(float(metrics[2]), 6),
        }
    return payload, event


def parse_local_header(header: bytes) -> dict[str, int]:
    if len(header) < 30 or header[:4] != b"PK\x03\x04":
        raise ValueError(f"not a ZIP local header: {header[:8]!r}")
    _, version, flags, method, _, _, crc, comp_size, file_size, name_len, extra_len = struct.unpack("<IHHHHHIIIHH", header[:30])
    return {
        "version_needed": int(version),
        "flags": int(flags),
        "compression_method": int(method),
        "local_crc32": int(crc),
        "local_compressed_size": int(comp_size),
        "local_file_size": int(file_size),
        "name_length": int(name_len),
        "extra_length": int(extra_len),
        "data_offset": 30 + int(name_len) + int(extra_len),
    }


def fetch_s110_recordings(output_dir: Path, log: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    central = json_file(CAVEMOVE_CENTRAL_DIRECTORY)
    if not isinstance(central, list):
        raise ValueError("CAVEMOVE central directory evidence is not a list")
    by_name = {str(item["filename"]): item for item in central if isinstance(item, dict) and "filename" in item}
    raw_dir = output_dir / "raw-noise"
    raw_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / ".range-cache"
    existing_records_path = raw_dir / "records.json"
    if existing_records_path.is_file():
        existing = json_file(existing_records_path)
        recordings = existing.get("recordings")
        if isinstance(recordings, list) and {"Honda_CR-V", "Volkswagen_Golf"}.issubset(
            {str(row.get("vehicle")) for row in recordings}
        ):
            validated = []
            for row in recordings:
                raw_path = raw_dir / Path(row["raw_path"]).name
                mono_path = raw_dir / Path(row["mono_path"]).name
                if not raw_path.is_file() or not mono_path.is_file():
                    raise FileNotFoundError(f"staged s110 file missing: {raw_path} / {mono_path}")
                if sha256_file(raw_path) != row["raw_sha256"] or sha256_file(mono_path) != row["mono_sha256"]:
                    raise AssertionError(f"staged s110 SHA mismatch: {raw_path}")
                raw_payload = raw_path.read_bytes()
                crc = f"{zlib.crc32(raw_payload) & 0xffffffff:08x}"
                if crc.lower() != str(row["crc32"]).lower():
                    raise AssertionError(f"staged s110 CRC mismatch: {raw_path}")
                info = sf.info(str(raw_path))
                if int(info.samplerate) != SR or int(info.channels) != 8:
                    raise AssertionError(f"staged s110 header mismatch: {raw_path}")
                row["raw_path"] = str(raw_path.resolve())
                row["mono_path"] = str(mono_path.resolve())
                validated.append(row)
            existing["source"]["central_directory"] = str(CAVEMOVE_CENTRAL_DIRECTORY)
            existing["source"]["central_directory_sha256"] = sha256_file(CAVEMOVE_CENTRAL_DIRECTORY)
            existing["source"]["full_zip_downloaded"] = False
            existing["status"] = "complete"
            existing["reuse"] = "validated staged raw/mono SHA and raw CRC; no network refetch"
            atomic_json(existing_records_path, existing)
            log(f"s110 reuse staged takes={[row['vehicle'] for row in validated]}")
            return validated, existing
    records: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for target in S110_TARGETS:
        member = target["member"]
        try:
            central_item = by_name[member]
            offset = int(central_item["local_header_offset"])
            log(f"s110 range header {member} offset={offset}")
            header_bytes, header_event = fetch_range(offset, offset + 29, purpose=f"{target['vehicle']} local header", temp_dir=temp_dir)
            events.append(header_event)
            local = parse_local_header(header_bytes)
            if local["compression_method"] != 8 or int(central_item["compression_method"]) != 8:
                raise ValueError("expected raw-deflate ZIP member")
            compressed_size = int(central_item["compress_size"])
            data_start = offset + local["data_offset"]
            data_end = data_start + compressed_size - 1
            compressed, payload_event = fetch_range(
                data_start,
                data_end,
                purpose=f"{target['vehicle']} s110 payload",
                temp_dir=temp_dir,
            )
            events.append(payload_event)
            raw = zlib.decompress(compressed, -15)
            expected_size = int(central_item["file_size"])
            if len(raw) != expected_size:
                raise ValueError(f"inflated size {len(raw)} != central size {expected_size}")
            crc = f"{zlib.crc32(raw) & 0xffffffff:08x}"
            if crc.lower() != str(central_item["crc32"]).lower():
                raise ValueError(f"CRC mismatch: {crc} != {central_item['crc32']}")
            basename = Path(member).name
            raw_path = raw_dir / f"CAVEMOVE_{target['vehicle']}_{target['representation']}_noise_{basename}"
            atomic_write_bytes(raw_path, raw)
            info = sf.info(str(raw_path))
            if int(info.samplerate) != SR or int(info.channels) != 8:
                raise ValueError(f"unexpected CAVEMOVE raw format: {info}")
            audio, rate = sf.read(str(raw_path), dtype="float32", always_2d=True)
            if int(rate) != SR or audio.shape[1] != 8:
                raise ValueError("CAVEMOVE raw channel contract changed")
            mono = np.ascontiguousarray(audio[:, 0], dtype=np.float32)
            if not np.all(np.isfinite(mono)):
                raise ValueError("CAVEMOVE channel 0 is non-finite")
            mono_path = raw_dir / f"CAVEMOVE_{target['vehicle']}_{target['representation']}_noise_{basename[:-4]}_ch0_mono_f32.wav"
            sf.write(str(mono_path), mono, SR, subtype="FLOAT")
            raw_sha = sha256_file(raw_path)
            mono_sha = sha256_file(mono_path)
            record = {
                "split": "test",
                "family": "car",
                "recording_id": f"cavemove:{member}",
                "archive_member": member,
                "vehicle": target["vehicle"],
                "representation": target["representation"],
                "take_id": "s110_w0",
                "speed_kmh": 110,
                "windows_state": "w0",
                "raw_path": str(raw_path.resolve()),
                "mono_path": str(mono_path.resolve()),
                "raw_sha256": raw_sha,
                "mono_sha256": mono_sha,
                "compressed_sha256": sha256_bytes(compressed),
                "crc32": crc,
                "actual_format": info.format,
                "actual_subtype": info.subtype,
                "actual_channels": int(info.channels),
                "actual_rate": int(info.samplerate),
                "actual_frames": int(info.frames),
                "actual_duration": float(info.frames / info.samplerate),
                "channel_selection": "fixed physical channel 0; no downmix; no resampling",
                "new_take_vs_train_dev": True,
                "independent_vehicle_generalization": False,
            }
            records.append(record)
            log(f"s110 fetched {target['vehicle']} raw={raw_sha[:12]} mono={mono_sha[:12]}")
        except Exception as exc:
            failures.append({"vehicle": target["vehicle"], "member": member, "error": f"{type(exc).__name__}: {exc}"})
            log(f"s110 fetch failed {target['vehicle']}: {type(exc).__name__}: {exc}")
    if len(records) < 2 or not {"Honda_CR-V", "Volkswagen_Golf"}.issubset({row["vehicle"] for row in records}):
        raise RuntimeError(f"need Honda and Golf s110 takes for the held-out test: failures={failures}")
    receipt = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "complete" if not failures else "complete_with_optional_take_failures",
        "source": {
            "dataset": "CAVEMOVE",
            "zenodo_record_id": 17404917,
            "url": CAVEMOVE_URL,
            "archive_length": CAVEMOVE_ARCHIVE_LENGTH,
            "central_directory": str(CAVEMOVE_CENTRAL_DIRECTORY),
            "central_directory_sha256": sha256_file(CAVEMOVE_CENTRAL_DIRECTORY),
            "full_zip_downloaded": False,
            "method": "HTTP 206 Range local header plus exact compressed member; raw-deflate; CRC32",
        },
        "selected_take_id": "s110_w0",
        "recordings": records,
        "network_events": events,
        "network_bytes": sum(int(event["bytes_received"]) for event in events),
        "failures": failures,
        "split_fact": {
            "old_train_dev_take": "s100_w0",
            "new_test_take": "s110_w0",
            "same_vehicle_different_take": True,
            "independent_vehicle_generalization": False,
            "statement": "The test uses new recordings/takes relative to train/dev but repeats vehicle families; do not claim independent-vehicle generalization.",
        },
    }
    atomic_json(raw_dir / "records.json", receipt)
    return records, receipt


def build_newtest_cases(
    sources: list[dict[str, Any]],
    s110_records: list[dict[str, Any]],
    audio_root: Path,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    speakers = choose_primary_speakers(sources, 16, rng)
    cases: list[dict[str, Any]] = []
    base_rows: list[dict[str, Any]] = []
    for ordinal, speaker in enumerate(speakers):
        base_id = f"v2-newtest-{ordinal:02d}"
        clean_raw, timeline = choose_timeline(
            sources,
            rng,
            seconds=TEST_SECONDS,
            primary_speakers=[speaker],
        )
        clean, speech_gain, clean_active_db, activity = normalize_clean(clean_raw, rng)
        if not -36.0 <= clean_active_db <= -24.0:
            raise AssertionError(f"newtest clean RMS out of range: {base_id}")
        noise_record = s110_records[ordinal % len(s110_records)]
        base_noise, noise_info = sample_noise(noise_record, TEST_SECONDS * SR, rng)
        base_dir = audio_root / base_id
        clean_path = base_dir / "clean.wav"
        write_float_wav(clean_path, clean, TEST_SECONDS * SR)
        clean_sha = sha256_file(clean_path)
        cluster = "cv-zh-test-" + hashlib.sha256(speaker.encode()).hexdigest()[:12]
        noise_take_hash = hashlib.sha256(
            json.dumps(
                {
                    "recording": noise_record["recording_id"],
                    "take_id": noise_record["take_id"],
                    "raw_sha256": noise_record["raw_sha256"],
                    "mono_sha256": noise_record["mono_sha256"],
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        for snr_db in SNRS:
            noise_rms = math.sqrt(float(np.mean(base_noise.astype(np.float64) ** 2)) + EPS)
            clean_active, _ = active_rms(clean)
            noise_gain = (clean_active / max(noise_rms, 1e-12)) / (10.0 ** (snr_db / 20.0))
            noise = np.asarray(base_noise * noise_gain, dtype=np.float32)
            mix = np.asarray(clean + noise, dtype=np.float32)
            peak = float(max(np.max(np.abs(clean)), np.max(np.abs(noise)), np.max(np.abs(mix))))
            common_scale = 1.0 if peak <= 0.95 else 0.95 / peak
            if common_scale != 1.0:
                clean = np.asarray(clean * common_scale, dtype=np.float32)
                noise = np.asarray(noise * common_scale, dtype=np.float32)
                mix = np.asarray(clean + noise, dtype=np.float32)
                clean_active, _ = active_rms(clean)
                clean_sha = sha256_float_array(clean)
                write_float_wav(clean_path, clean, TEST_SECONDS * SR)
                clean_sha = sha256_file(clean_path)
            noise_rms = math.sqrt(float(np.mean(noise.astype(np.float64) ** 2)) + EPS)
            measured_snr = 20.0 * math.log10(max(clean_active, 1e-12) / max(noise_rms, 1e-12))
            if abs(measured_snr - snr_db) > 1e-5:
                raise AssertionError(f"newtest SNR mismatch {base_id}: {measured_snr} vs {snr_db}")
            snr_dir = base_dir / f"snr{int(snr_db):02d}"
            mic_path = snr_dir / "mic.wav"
            noise_path = snr_dir / "noise.wav"
            write_float_wav(mic_path, mix, TEST_SECONDS * SR)
            write_float_wav(noise_path, noise, TEST_SECONDS * SR)
            source_sha = [statement["sha256"] for statement in timeline["statements"]]
            cases.append(
                {
                    "id": f"{base_id}-snr{int(snr_db):02d}",
                    "base_id": base_id,
                    "language": "zh",
                    "speaker_id": speaker,
                    "speakercluster": cluster,
                    "mic": "synthetic mono clean speech plus fixed-channel-0 CAVEMOVE s110_w0 noise",
                    "kind": "near_only",
                    "window_s": [0.0, 12.0],
                    "snr_db": int(snr_db),
                    "measured_snr_db": measured_snr,
                    "snr_definition": "10log10(sum(clean_active^2)/sum(noise^2))",
                    "source_ids": [statement["source_id"] for statement in timeline["statements"]],
                    "source_sha256": source_sha,
                    "input_clean_path": str(clean_path.resolve()),
                    "input_clean_sha256": clean_sha,
                    "input_noise_path": str(noise_path.resolve()),
                    "input_noise_sha256": sha256_file(noise_path),
                    "mic_path": str(mic_path.resolve()),
                    "mic_sha256": sha256_file(mic_path),
                    "noise_id": noise_record["recording_id"],
                    "noise_take_hash": noise_take_hash,
                    "noise_source_path": noise_record["raw_path"],
                    "noise_source_sha256": noise_record["raw_sha256"],
                    "noise_vehicle": noise_record["vehicle"],
                    "noise_representation": noise_record["representation"],
                    "noise_take_id": noise_record["take_id"],
                    "noise_timeline": noise_info,
                    "clean_active_rms_dbfs": 20.0 * math.log10(max(clean_active, 1e-12)),
                    "activity_definition": activity,
                    "near_timeline": timeline,
                    "common_peak_scale": common_scale,
                    "freeze_protocol": "frozen deterministic case; never used to select a checkpoint or tune on test",
                    "same_vehicle_different_take_not_independent_vehicle": True,
                }
            )
        base_rows.append(
            {
                "base_id": base_id,
                "speaker_id": speaker,
                "speakercluster": cluster,
                "clean_path": str(clean_path.resolve()),
                "clean_sha256": clean_sha,
                "noise_take_hash": noise_take_hash,
                "source_ids": [statement["source_id"] for statement in timeline["statements"]],
            }
        )
    if len({case["speaker_id"] for case in cases}) != 16:
        raise AssertionError("newtest does not contain 16 distinct speakers")
    if len(cases) != 48:
        raise AssertionError(f"expected 48 newtest SNR cases, got {len(cases)}")
    receipt = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "frozen",
        "base_cases": 16,
        "cases": len(cases),
        "speakers": 16,
        "snr_values": list(SNRS),
        "window_s": [0.0, 12.0],
        "language": "zh",
        "source": "CV-zh test evaluation originals only",
        "noise": "new CAVEMOVE s110_w0 takes; not s100 train/dev",
        "training_manifest_inclusion": False,
        "checkpoint_selection_allowed": False,
        "base_rows": base_rows,
    }
    return cases, receipt


def write_source_quality(
    sources: dict[str, list[dict[str, Any]]],
    rows: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    selected_ids = {
        statement["id"]
        for row in rows
        if row.get("near_timeline")
        for statement in row["near_timeline"]["statements"]
    }
    selected_ids.update(statement["id"] for case in cases for statement in case["near_timeline"]["statements"])
    by_id = {row["id"]: row for split in sources.values() for row in split}
    assessments = [assess_source_quality(by_id[source_id]) for source_id in sorted(selected_ids)]
    if not all(item["source_attribution"]["human_original_confirmed"] for item in assessments):
        raise AssertionError("a selected source is not an attributed human original")
    representatives = {}
    for split in ("train", "dev", "test"):
        candidates = [item for item in assessments if item["split"] == split]
        if candidates:
            representatives[split] = candidates[0]
    receipt = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "source_root": str(DETAIL_ROOT),
        "detail_manifests": {
            split: {"path": str(DETAIL_ROOT / f"{split}_detail.jsonl"), "sha256": sha256_file(DETAIL_ROOT / f"{split}_detail.jsonl")}
            for split in ("train", "dev", "test")
        },
        "selection_rule": (
            "CV-zh CC0 only; train usage=original, dev/test usage=evaluation; "
            "human-original attribution from detail source.kind=file; no augmentation or replay rows"
        ),
        "source_name_is_not_clean_evidence": True,
        "label": (
            "CV-zh reference is the attributed human original, but it is not guaranteed noise-free; "
            "existing VAD/loudness normalization is materialization, not evidence of a clean reference."
        ),
        "clean_judgment": (
            "The materialized WAVs are VAD/loudness-normalized originals with possible real-world residual noise; "
            "they are not enhanced or car-noise mixtures. "
            "Header and descriptive acoustics are diagnostic only and were not used to relabel enhancement."
        ),
        "selected_source_count": len(assessments),
        "human_original_confirmed_count": sum(item["source_attribution"]["human_original_confirmed"] for item in assessments),
        "representatives": representatives,
        "selected_sources": assessments,
    }
    atomic_json(output_dir / "source-quality.json", receipt)
    return receipt


def write_newtest_quality(
    sources: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    by_id = {row["id"]: row for row in sources}
    selected_ids = {
        statement["id"] for case in cases for statement in case["near_timeline"]["statements"]
    }
    assessments = [assess_source_quality(by_id[source_id]) for source_id in sorted(selected_ids)]
    receipt = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "scope": "newtest16 only; does not replace or alter the frozen train/dev source-quality receipt",
        "selected_source_count": len(assessments),
        "human_original_confirmed_count": sum(
            item["source_attribution"]["human_original_confirmed"] for item in assessments
        ),
        "label": "reference CV-zh original is not guaranteed noise-free; VAD/loudness normalization is materialization, not denoising",
        "selected_sources": assessments,
    }
    atomic_json(output_dir / "source-quality-newtest.json", receipt)
    return receipt


def build_data_manifest(
    rows: list[dict[str, Any]],
    output_dir: Path,
    noise_pool: dict[str, list[dict[str, Any]]],
    noise_receipt: dict[str, Any],
    exclusions: dict[str, Any],
    quality: dict[str, Any],
    cases_path: Path,
    newtest_receipt_path: Path,
    source_detail_hashes: dict[str, Any],
) -> dict[str, Any]:
    train_sha = {sha for row in rows if row["split"] == "train" for sha in row["raw_source_sha256"]}
    dev_sha = {sha for row in rows if row["split"] == "dev" for sha in row["raw_source_sha256"]}
    train_speakers = {speaker for row in rows if row["split"] == "train" for speaker in row["speaker_ids"]}
    dev_speakers = {speaker for row in rows if row["split"] == "dev" for speaker in row["speaker_ids"]}
    excluded_sha = set(exclusions["excluded_raw_sha256"])
    excluded_speakers = set(exclusions["excluded_speakers"])
    if train_sha & dev_sha:
        raise AssertionError("train/dev selected source SHA overlap")
    if train_speakers & dev_speakers:
        raise AssertionError("train/dev selected speaker overlap")
    if (train_sha | dev_sha) & excluded_sha:
        raise AssertionError("frozen v1/v120 source SHA entered train/dev")
    if (train_speakers | dev_speakers) & excluded_speakers:
        raise AssertionError("frozen v1/v120 speaker entered train/dev")
    source_receipt = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "source_dataset": "CV-zh (Common Voice Chinese)",
        "license": "CC0-1.0",
        "detail_manifests": source_detail_hashes,
        "speech_catalog_counts": {"train": 765, "dev": 87, "test": 74},
        "speech_catalog_speakers": {"train": 456, "dev": 78, "test": 74},
        "train_dev_source_sha_disjoint": True,
        "train_dev_speaker_disjoint": True,
        "v1_v120_source_sha_intersection": [],
        "v1_v120_speaker_intersection": [],
        "used_raw_source_sha256": {
            split: sorted({sha for row in rows if row["split"] == split for sha in row["raw_source_sha256"]})
            for split in ("train", "dev")
        },
        "used_speakers": {
            split: sorted({speaker for row in rows if row["split"] == split for speaker in row["speaker_ids"]})
            for split in ("train", "dev")
        },
        "noise_split": {
            "train": [{"family": row["family"], "recording": row["recording"], "take_id": row["take_id"]} for row in noise_pool["train"]],
            "dev": [{"family": row["family"], "recording": row["recording"], "take_id": row["take_id"]} for row in noise_pool["dev"]],
        },
        "demand_split": {
            "train": sorted({row["recording"] for row in noise_pool["train"] if row["family"] == "demand"}),
            "dev": sorted({row["recording"] for row in noise_pool["dev"] if row["family"] == "demand"}),
        },
        "newtest_noise": {
            "take": "s110_w0",
            "receipt": str(output_dir / "raw-noise/records.json"),
            "receipt_sha256": sha256_file(output_dir / "raw-noise/records.json"),
            "recordings": [row["recording_id"] for row in noise_receipt["recordings"]],
            "same_vehicle_different_take_not_independent_vehicle": True,
        },
        "newtest_excluded_from_training": True,
        "test_must_not_select_checkpoint": True,
        "raw_sha_scope": "materialized CV-zh CC0 WAV; original upstream MP3 is not present locally",
    }
    atomic_json(output_dir / "source-receipt.json", source_receipt)
    manifest = {
        "schema_version": 2,
        "dataset_id": "ns-hybrid-v2-20261115-expansion",
        "created_utc": utc_now(),
        "seed": SEED,
        "sample_rate": SR,
        "duration_seconds": TRAIN_SECONDS,
        "samples": TRAIN_SECONDS * SR,
        "train_count": 32,
        "dev_count": 8,
        "newtest_base_count": 16,
        "newtest_case_count": 48,
        "language_balance": {split: dict(Counter(row["language"] for row in rows if row["split"] == split)) for split in ("train", "dev")},
        "type_counts": {split: dict(Counter(row["type"] for row in rows if row["split"] == split)) for split in ("train", "dev")},
        "speaker_counts": {
            "train": len(train_speakers),
            "dev": len(dev_speakers),
            "newtest_base": 16,
        },
        "audio_contract": {
            "format": "WAV",
            "subtype": "FLOAT",
            "sample_rate": SR,
            "channels": 1,
            "samples": TRAIN_SECONDS * SR,
            "frames": FRAMES,
            "mix_is_clean_plus_noise": True,
            "max_mix_clean_noise_abs_error": max(row["mix_clean_noise_max_abs_error"] for row in rows),
            "clean_active_rms_dbfs_range": [-36.0, -24.0],
            "statement_cropping": False,
            "short_gap_seconds": [0.10, 1.00],
            "long_gap_seconds": [3.0, 8.0],
        },
        "feature_contract": {
            "native_probe_sha256": EXPECTED_PROBE_SHA256,
            "native_vector_float32": FEATURE_WIDTH,
            "native_slices": {"x": [0, 256], "power": [256, 320], "noise": [320, 384], "base_gain": [384, 641]},
            "stft": {"sample_rate": SR, "n_fft": N_FFT, "hop_length": HOP, "window_length": WINDOW, "window": "sine", "center": False, "frames": FRAMES, "bins": BINS},
            "mix_quantization": "clip(x*32768,-32768,32767); C++ std::lround; divide by 32768",
            "supervised_targets": ["mix_spec", "clean_spec"],
            "teacher_required": False,
        },
        "isolation": {
            "train_dev_source_sha_disjoint": True,
            "train_dev_speaker_disjoint": True,
            "train_dev_noise_sha_disjoint": True,
            "v1_v120_source_intersection": [],
            "v1_v120_speaker_intersection": [],
            "newtest_in_training_manifest": False,
            "test_checkpoint_selection_allowed": False,
        },
        "receipts": {
            "exclude_receipt": {"path": str(output_dir / "exclude-receipt.json"), "sha256": sha256_file(output_dir / "exclude-receipt.json")},
            "source_receipt": {"path": str(output_dir / "source-receipt.json"), "sha256": sha256_file(output_dir / "source-receipt.json")},
            "source_quality": {"path": str(output_dir / "source-quality.json"), "sha256": sha256_file(output_dir / "source-quality.json")},
            "noise_receipt": {"path": str(output_dir / "raw-noise/records.json"), "sha256": sha256_file(output_dir / "raw-noise/records.json")},
            "newtest_cases": {"path": str(cases_path), "sha256": sha256_file(cases_path)},
            "newtest_receipt": {"path": str(newtest_receipt_path), "sha256": sha256_file(newtest_receipt_path)},
        },
        "rows": rows,
        "newtest_cases": {"path": str(cases_path), "sha256": sha256_file(cases_path), "included_in_rows": False},
    }
    return manifest


def merge_feature_manifest(
    new_rows: list[dict[str, Any]],
    features_dir: Path,
    source_manifest_cache: Path,
    validation: dict[str, Any],
) -> dict[str, Any]:
    old = json_file(OLD_FEATURE_MANIFEST)
    old_rows = old.get("rows")
    if not isinstance(old_rows, list) or len(old_rows) != 152:
        raise ValueError("old frozen feature manifest must contain 152 rows")
    if Counter(row["split"] for row in old_rows) != Counter({"train": 128, "dev": 24}):
        raise ValueError("old feature split counts changed")
    merged_rows = old_rows + new_rows
    if Counter(row["split"] for row in merged_rows) != Counter({"train": 160, "dev": 32}):
        raise AssertionError("merged feature split counts are not 160 train / 32 dev")
    if any(row.get("split") == "test" for row in merged_rows):
        raise AssertionError("newtest leaked into feature manifest")
    ids = [row["id"] for row in merged_rows]
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate feature row IDs")
    contract = {
        "schema_version": 2,
        "sample_rate": SR,
        "samples": TRAIN_SECONDS * SR,
        "frames": FRAMES,
        "hop_length": HOP,
        "window_length": WINDOW,
        "n_fft": N_FFT,
        "bins": BINS,
        "window": "sine",
        "center": False,
        "frame_policy": "[previous 160 PCM,current 160 PCM] * sine; zero-pad to 512; rfft",
        "mix_quantization": "clip(x*32768,-32768,32767); C++ std::lround; divide by 32768",
        "runtime_inputs": ["x", "base_gain"],
        "targets": ["mix_spec", "clean_spec"],
        "teacher_required": False,
        "npz_path_key": "npz_path",
        "npz_hash_key": "npz_sha256",
        "npz_fields": {
            "x": {"shape": [FRAMES, 256], "dtype": "float32"},
            "base_gain": {"shape": [FRAMES, BINS], "dtype": "float32"},
            "mix_spec": {"shape": [FRAMES, BINS], "dtype": "complex64"},
            "clean_spec": {"shape": [FRAMES, BINS], "dtype": "complex64"},
            "power": {"shape": [FRAMES, 64], "dtype": "float32"},
            "noise": {"shape": [FRAMES, 64], "dtype": "float32"},
        },
        "native_vector_slices": {"x": [0, 256], "power": [256, 320], "noise": [320, 384], "base_gain": [384, 641]},
        "legacy_rows": "frozen 128 train + 24 dev rows retained byte-for-byte as manifest metadata and old cache paths",
    }
    manifest = {
        "schema_version": 2,
        "dataset_id": "ns-hybrid-v2-20261115-expansion",
        "created_utc": utc_now(),
        "source_manifest": {"path": str(source_manifest_cache), "sha256": sha256_file(source_manifest_cache)},
        "legacy_feature_manifest": {"path": str(OLD_FEATURE_MANIFEST), "sha256": sha256_file(OLD_FEATURE_MANIFEST)},
        "native_probe": {"path": str(OLD_PROBE), "sha256": EXPECTED_PROBE_SHA256},
        "teacher": {"required": False, "generated_for_new_rows": False, "legacy_teacher_specs_preserved_in_old_rows": True},
        "feature_contract": contract,
        "counts": {"train": 160, "dev": 32, "total": 192, "new_train": 32, "new_dev": 8, "newtest": 0},
        "rows": merged_rows,
        "validation": {
            **validation,
            "merged_rows": 192,
            "merged_train": 160,
            "merged_dev": 32,
            "all_ids_unique": True,
            "newtest_rows_in_manifest": 0,
        },
    }
    atomic_json(features_dir / "manifest.json", manifest)
    atomic_json(features_dir / "new-rows.json", {"rows": new_rows, "count": len(new_rows)})
    return manifest


def build_pending_train_dev_manifest(
    rows: list[dict[str, Any]],
    output_dir: Path,
    noise_pool: dict[str, list[dict[str, Any]]],
    exclusions: dict[str, Any],
    quality: dict[str, Any],
    detail_hashes: dict[str, Any],
) -> dict[str, Any]:
    train_sha = {sha for row in rows if row["split"] == "train" for sha in row["raw_source_sha256"]}
    dev_sha = {sha for row in rows if row["split"] == "dev" for sha in row["raw_source_sha256"]}
    train_speakers = {speaker for row in rows if row["split"] == "train" for speaker in row["speaker_ids"]}
    dev_speakers = {speaker for row in rows if row["split"] == "dev" for speaker in row["speaker_ids"]}
    if train_sha & dev_sha or train_speakers & dev_speakers:
        raise AssertionError("train/dev isolation failed")
    pending_path = output_dir / "newtest-pending.json"
    atomic_json(
        pending_path,
        {
            "status": "pending",
            "created_utc": utc_now(),
            "reason": "train/dev features must not wait for the independent s110 Range fetch",
            "cases_expected": 48,
            "base_cases_expected": 16,
            "complete": False,
        },
    )
    source_receipt = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "source_dataset": "CV-zh",
        "license": "CC0-1.0",
        "detail_manifests": detail_hashes,
        "train_dev_source_sha_disjoint": True,
        "train_dev_speaker_disjoint": True,
        "v1_v120_source_sha_intersection": [],
        "v1_v120_speaker_intersection": [],
        "used_raw_source_sha256": {
            split: sorted({sha for row in rows if row["split"] == split for sha in row["raw_source_sha256"]})
            for split in ("train", "dev")
        },
        "used_speakers": {
            split: sorted({speaker for row in rows if row["split"] == split for speaker in row["speaker_ids"]})
            for split in ("train", "dev")
        },
        "noise_split": {
            split: [{"family": row["family"], "recording": row["recording"], "take_id": row["take_id"]} for row in noise_pool[split]]
            for split in ("train", "dev")
        },
        "raw_sha_scope": "materialized CV-zh CC0 WAV; upstream MP3 unavailable",
    }
    atomic_json(output_dir / "source-receipt.json", source_receipt)
    return {
        "schema_version": 2,
        "dataset_id": "ns-hybrid-v2-20261115-expansion",
        "created_utc": utc_now(),
        "status": "train_dev_complete_newtest_pending",
        "newtest_complete": False,
        "train_count": 32,
        "dev_count": 8,
        "speaker_counts": {"train": len(train_speakers), "dev": len(dev_speakers)},
        "type_counts": {split: dict(Counter(row["type"] for row in rows if row["split"] == split)) for split in ("train", "dev")},
        "rows": rows,
        "receipts": {
            "exclude_receipt": {"path": str(output_dir / "exclude-receipt.json"), "sha256": sha256_file(output_dir / "exclude-receipt.json")},
            "source_receipt": {"path": str(output_dir / "source-receipt.json"), "sha256": sha256_file(output_dir / "source-receipt.json")},
            "source_quality": {"path": str(output_dir / "source-quality.json"), "sha256": sha256_file(output_dir / "source-quality.json")},
            "newtest_pending": {"path": str(pending_path), "sha256": sha256_file(pending_path)},
        },
        "isolation": {
            "train_dev_source_sha_disjoint": True,
            "train_dev_speaker_disjoint": True,
            "newtest_in_training_manifest": False,
        },
        "feature_contract": {
            "native_probe_sha256": EXPECTED_PROBE_SHA256,
            "frames": FRAMES,
            "native_vector_float32": FEATURE_WIDTH,
            "targets": ["mix_spec", "clean_spec"],
            "teacher_required": False,
        },
        "newtest": {"status": "pending", "complete": False},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip-newtest", action="store_true", help="finish 32+8 caches while leaving s110 test explicitly pending")
    parser.add_argument("--finish-newtest", action="store_true", help="freeze s110 test cases without touching the 192-row feature manifest")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    features_dir = args.features_dir.resolve()
    if TARGET_ROOT not in output_dir.parents and output_dir != TARGET_ROOT:
        raise ValueError(f"output-dir escapes target root: {output_dir}")
    if TARGET_ROOT not in features_dir.parents and features_dir != TARGET_ROOT:
        raise ValueError(f"features-dir escapes target root: {features_dir}")
    if output_dir.name != "data" or features_dir.name != "features":
        raise ValueError("this task is fixed to data/ and features/")
    if shutil.disk_usage(TARGET_ROOT).free < 6_000_000_000:
        raise RuntimeError("less than 6 GB free; refusing the 40-row feature build")
    output_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    log_dir = TARGET_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"data-{stamp}-{os.getpid()}.log"

    def log(message: str) -> None:
        line = f"[{utc_now()}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    started = time.time()
    log(f"start output={output_dir} features={features_dir} seed={args.seed} jobs={args.jobs}")
    if args.finish_newtest:
        feature_manifest_path = features_dir / "manifest.json"
        feature_sha_before = sha256_file(feature_manifest_path)
        data_manifest_path = output_dir / "manifest.json"
        data_manifest = json_file(data_manifest_path)
        if len(data_manifest.get("rows", [])) != 40:
            raise ValueError("finish-newtest requires the frozen 40-row train/dev data manifest")
        sources = load_speech_sources()
        exclusions = collect_source_exclusions()
        s110_records, noise_receipt = fetch_s110_recordings(output_dir, log)
        rng = np.random.default_rng(args.seed)
        cases, newtest_receipt = build_newtest_cases(sources["test"], s110_records, output_dir / "audio-newtest", rng)
        cases_path = output_dir / "cases-newtest16.json"
        newtest_receipt_path = output_dir / "newtest-receipt.json"
        atomic_json(cases_path, cases)
        atomic_json(newtest_receipt_path, newtest_receipt)
        newtest_quality = write_newtest_quality(sources["test"], cases, output_dir)
        train_dev_speakers = {
            speaker for row in data_manifest["rows"] for speaker in row.get("speaker_ids", [])
        }
        train_dev_source_sha = {
            sha for row in data_manifest["rows"] for sha in row.get("raw_source_sha256", [])
        }
        test_speakers = {case["speaker_id"] for case in cases}
        test_source_sha = {sha for case in cases for sha in case["source_sha256"]}
        if train_dev_speakers & test_speakers:
            raise AssertionError("newtest speaker intersects frozen train/dev")
        if train_dev_source_sha & test_source_sha:
            raise AssertionError("newtest source SHA intersects frozen train/dev")
        if test_source_sha & set(exclusions["excluded_raw_sha256"]):
            raise AssertionError("newtest source SHA intersects frozen v1/v120")
        data_manifest["status"] = "complete"
        data_manifest["newtest_complete"] = True
        data_manifest["newtest_base_count"] = 16
        data_manifest["newtest_case_count"] = 48
        data_manifest["speaker_counts"]["newtest_base"] = 16
        data_manifest["newtest_cases"] = {
            "path": str(cases_path),
            "sha256": sha256_file(cases_path),
            "included_in_rows": False,
            "in_feature_manifest": False,
        }
        data_manifest.setdefault("receipts", {})["newtest_cases"] = {
            "path": str(cases_path), "sha256": sha256_file(cases_path)
        }
        pending_receipt = data_manifest["receipts"].pop("newtest_pending", None)
        if pending_receipt:
            data_manifest.setdefault("superseded_artifacts", []).append(
                {**pending_receipt, "status": "superseded_by_complete_newtest", "complete": False}
            )
        data_manifest["receipts"]["newtest_receipt"] = {
            "path": str(newtest_receipt_path), "sha256": sha256_file(newtest_receipt_path)
        }
        data_manifest["receipts"]["newtest_noise_receipt"] = {
            "path": str(output_dir / "raw-noise/records.json"), "sha256": sha256_file(output_dir / "raw-noise/records.json")
        }
        data_manifest["receipts"]["source_quality_newtest"] = {
            "path": str(output_dir / "source-quality-newtest.json"),
            "sha256": sha256_file(output_dir / "source-quality-newtest.json"),
        }
        data_manifest.setdefault("isolation", {}).update(
            {
                "newtest_speaker_intersection_train_dev": [],
                "newtest_source_sha_intersection_train_dev": [],
                "newtest_in_training_manifest": False,
                "test_checkpoint_selection_allowed": False,
                "newtest_noise_take": "s110_w0",
                "same_vehicle_different_take_not_independent_vehicle": True,
            }
        )
        atomic_json(data_manifest_path, data_manifest)
        feature_sha_after = sha256_file(feature_manifest_path)
        if feature_sha_after != feature_sha_before:
            raise AssertionError("finish-newtest changed the frozen feature manifest")
        receipt = {
            "schema_version": 1,
            "status": "complete",
            "created_utc": utc_now(),
            "elapsed_seconds": time.time() - started,
            "feature_manifest": {
                "path": str(feature_manifest_path),
                "sha256_before": feature_sha_before,
                "sha256_after": feature_sha_after,
                "unchanged": True,
            },
            "data_manifest": {"path": str(data_manifest_path), "sha256": sha256_file(data_manifest_path)},
            "cases": {"path": str(cases_path), "sha256": sha256_file(cases_path), "base": 16, "rows": 48},
            "newtest_receipt": {"path": str(newtest_receipt_path), "sha256": sha256_file(newtest_receipt_path)},
            "noise_receipt": {"path": str(output_dir / "raw-noise/records.json"), "sha256": sha256_file(output_dir / "raw-noise/records.json")},
            "source_quality": {"path": str(output_dir / "source-quality-newtest.json"), "sha256": sha256_file(output_dir / "source-quality-newtest.json"), "selected_sources": newtest_quality["selected_source_count"]},
            "validation": {
                "speakers": 16,
                "cases": 48,
                "snr_values": [0, 10, 20],
                "window_s": [0.0, 12.0],
                "speaker_intersection_train_dev": [],
                "source_sha_intersection_train_dev": [],
                "feature_manifest_unchanged": True,
            },
            "log": {"path": str(log_path), "sha256": sha256_file(log_path)},
        }
        atomic_json(output_dir / "newtest-stage-receipt.json", receipt)
        log(f"newtest complete cases=48 feature_manifest_unchanged={feature_sha_after} data_manifest={receipt['data_manifest']['sha256']}")
        print(json.dumps(receipt, indent=2, ensure_ascii=False))
        return 0
    sources = load_speech_sources()
    log("speech sources: " + ", ".join(f"{split}={len(rows)}" for split, rows in sources.items()))
    exclusions = collect_source_exclusions()
    atomic_json(output_dir / "exclude-receipt.json", exclusions)
    log(f"exclusions: speakers={len(exclusions['excluded_speakers'])} raw_sha={len(exclusions['excluded_raw_sha256'])}")
    excluded_sha = set(exclusions["excluded_raw_sha256"])
    noise = noise_pools(excluded_sha)
    log("noise pools: " + ", ".join(f"{split}={len(rows)}" for split, rows in noise.items()))
    rng = np.random.default_rng(args.seed)
    rows = build_train_dev_rows(sources, noise, output_dir / "audio", rng)
    log(f"train/dev audio complete rows={len(rows)}")
    detail_hashes = {
        split: {"path": str(DETAIL_ROOT / f"{split}_detail.jsonl"), "sha256": sha256_file(DETAIL_ROOT / f"{split}_detail.jsonl")}
        for split in ("train", "dev", "test")
    }
    if args.skip_newtest:
        quality = write_source_quality(sources, rows, [], output_dir)
        pending_manifest = build_pending_train_dev_manifest(rows, output_dir, noise, exclusions, quality, detail_hashes)
        source_manifest_cache = output_dir / "manifest-at-feature-cache.json"
        atomic_json(source_manifest_cache, pending_manifest)
        new_feature_rows, feature_validation = build_new_features(rows, features_dir, args.jobs, log)
        feature_manifest = merge_feature_manifest(new_feature_rows, features_dir, source_manifest_cache, feature_validation)
        pending_manifest["rows"] = new_feature_rows
        pending_manifest["features"] = {
            "manifest_path": str(features_dir / "manifest.json"),
            "manifest_sha256": sha256_file(features_dir / "manifest.json"),
        }
        pending_manifest["source_manifest_at_feature_cache"] = {
            "path": str(source_manifest_cache),
            "sha256": sha256_file(source_manifest_cache),
        }
        atomic_json(output_dir / "manifest.json", pending_manifest)
        receipt = {
            "schema_version": 2,
            "status": "train_dev_complete_newtest_pending",
            "complete": False,
            "elapsed_seconds": time.time() - started,
            "manifest": {"path": str(output_dir / "manifest.json"), "sha256": sha256_file(output_dir / "manifest.json")},
            "feature_manifest": {"path": str(features_dir / "manifest.json"), "sha256": sha256_file(features_dir / "manifest.json")},
            "counts": {"train": 32, "dev": 8, "new_features": 40, "merged_features": 192},
            "validation": feature_manifest["validation"],
            "newtest": {"status": "pending", "complete": False},
            "log": {"path": str(log_path), "sha256": sha256_file(log_path)},
        }
        atomic_json(output_dir / "train-dev-receipt.json", receipt)
        log(f"train/dev complete manifest={receipt['manifest']['sha256']} features={receipt['feature_manifest']['sha256']} elapsed={receipt['elapsed_seconds']:.1f}s")
        print(json.dumps(receipt, indent=2, ensure_ascii=False))
        return 0
    # Build s110 before the test cases.  The two required takes are fetched first.
    s110_records, noise_receipt = fetch_s110_recordings(output_dir, log)
    cases, newtest_receipt = build_newtest_cases(sources["test"], s110_records, output_dir / "audio-newtest", rng)
    cases_path = output_dir / "cases-newtest16.json"
    newtest_receipt_path = output_dir / "newtest-receipt.json"
    atomic_json(cases_path, cases)
    atomic_json(newtest_receipt_path, newtest_receipt)
    quality = write_source_quality(sources, rows, cases, output_dir)
    log(f"newtest frozen base=16 cases={len(cases)} noise_takes={len(s110_records)}")
    # A pre-feature source manifest avoids a circular hash reference.
    provisional_rows = [{key: value for key, value in row.items() if key not in {"npz_path", "npz_sha256"}} for row in rows]
    provisional = build_data_manifest(
        provisional_rows,
        output_dir,
        noise,
        noise_receipt,
        exclusions,
        quality,
        cases_path,
        newtest_receipt_path,
        detail_hashes,
    )
    source_manifest_cache = output_dir / "manifest-at-feature-cache.json"
    atomic_json(source_manifest_cache, provisional)
    new_feature_rows, feature_validation = build_new_features(rows, features_dir, args.jobs, log)
    feature_manifest = merge_feature_manifest(new_feature_rows, features_dir, source_manifest_cache, feature_validation)
    final_rows = new_feature_rows
    manifest = build_data_manifest(
        final_rows,
        output_dir,
        noise,
        noise_receipt,
        exclusions,
        quality,
        cases_path,
        newtest_receipt_path,
        detail_hashes,
    )
    manifest["features"] = {
        "manifest_path": str(features_dir / "manifest.json"),
        "manifest_sha256": sha256_file(features_dir / "manifest.json"),
        "native_probe_sha256": EXPECTED_PROBE_SHA256,
        "teacher_required": False,
    }
    manifest["source_manifest_at_feature_cache"] = {"path": str(source_manifest_cache), "sha256": sha256_file(source_manifest_cache)}
    atomic_json(output_dir / "manifest.json", manifest)
    receipt = {
        "schema_version": 2,
        "status": "ok",
        "created_utc": utc_now(),
        "elapsed_seconds": time.time() - started,
        "manifest": {"path": str(output_dir / "manifest.json"), "sha256": sha256_file(output_dir / "manifest.json")},
        "feature_manifest": {"path": str(features_dir / "manifest.json"), "sha256": sha256_file(features_dir / "manifest.json")},
        "newtest_cases": {"path": str(cases_path), "sha256": sha256_file(cases_path)},
        "counts": {"train": 32, "dev": 8, "new_features": 40, "merged_features": 192, "newtest_base": 16, "newtest_cases": 48},
        "validation": {
            "train_speakers": manifest["speaker_counts"]["train"],
            "dev_speakers": manifest["speaker_counts"]["dev"],
            "newtest_speakers": 16,
            "train_dev_source_sha_disjoint": True,
            "train_dev_speaker_disjoint": True,
            "train_dev_noise_sha_disjoint": True,
            "v1_v120_source_intersection": [],
            "newtest_in_feature_manifest": False,
            "feature_manifest_train": feature_manifest["counts"]["train"],
            "feature_manifest_dev": feature_manifest["counts"]["dev"],
            "stft_native_power_max_abs_error": feature_validation["max_stft_native_power_error"],
        },
        "log": {"path": str(log_path), "sha256": sha256_file(log_path)},
        "environment": {
            "hostname": socket.gethostname(),
            "python": sys.version,
            "numpy": np.__version__,
            "soundfile": sf.__version__,
            "platform": platform.platform(),
        },
    }
    atomic_json(output_dir / "build-receipt.json", receipt)
    log(f"complete manifest={receipt['manifest']['sha256']} features={receipt['feature_manifest']['sha256']} elapsed={receipt['elapsed_seconds']:.1f}s")
    print(json.dumps(receipt["validation"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
