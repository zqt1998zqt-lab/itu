#!/usr/bin/env python3
"""Create the 20-group Sce1 background-noise engineering preview sidecar.

This is deliberately separate from the formal 336-item acceptance dataset. It
uses clean speech and controlled noise sources, writes exclusively to the
scene-condition preview store, and never claims ITU-T compliance.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
from io import BytesIO
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np
from scipy import signal
import soundfile as sf


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
sys.dont_write_bytecode = True

SOURCE_PATHS = {
    "generator_sha256": Path(__file__),
    "legacy_generator_sha256": SCRIPTS / "create-acceptance-measurement-dataset.py",
    "signal_design_sha256": SCRIPTS / "acceptance_signal_design.py",
    "scene_contract_sha256": PROJECT / "server/acoustic/acceptance_scene_conditions.py",
}
IMPORTED_SOURCE_HASHES = {
    key: hashlib.sha256(path.read_bytes()).hexdigest()
    for key, path in SOURCE_PATHS.items()
}

from acceptance_signal_design import render_signals
from server.acoustic import acceptance_scene_conditions as scene_contract
from server.acoustic.acceptance_visualization import (
    cache_root as visualization_cache_root,
    precompute_visualizations,
)


DATASET_ID = scene_contract.DATASET_ID
DATASET_NAME = scene_contract.DATASET_NAME
SCHEMA_VERSION = scene_contract.SCHEMA_VERSION
MODE = scene_contract.MODE
SCENE = scene_contract.SCENE
ROLES = scene_contract.ROLES
CONDITIONS = dict(scene_contract.CONDITIONS)
OUTPUT_ROOT = scene_contract.dataset_root(DATASET_ID)
MANIFEST_PATH = OUTPUT_ROOT / "manifest.json"
ASSETS_PATH = OUTPUT_ROOT / "assets.json"
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"
QUALITY_PATH = OUTPUT_ROOT / "quality.json"

RATE = 16_000
DURATION_S = 12.0
FRAMES = int(round(RATE * DURATION_S))
BASE_SEED = 20261109
SPEECH_GROUPS = 10
CNG_GROUPS = 10
SNR_DB = 15.0
NEAR_CONDITIONED_PEAK = 0.12
MAX_SOURCE_SEGMENT_S = 4.2
MIN_NON_SILENT_SECONDS = 9.01
NON_SILENT_FRAME_RMS = 1e-5
MIN_FREE_BYTES = 100 * 1024**3
MIN_MEM_AVAILABLE_BYTES = 5 * 1024**3

SPEECH_ITEM_IDS = ["WB-wired:B24", "WB-wired:B25", "WB-wired:B26"]
SPEECH_FAMILIES = ["g_mos", "n_mos", "s_mos"]
CNG_ITEM_IDS = ["WB-wired:B27", "WB-wired:B28"]
CNG_FAMILIES = ["cng_level", "cng_spectrum"]


class ResourceSafetyStop(RuntimeError):
    """Raised before continuing when the project safety floor is crossed."""


def _load_legacy_generator():
    path = SCRIPTS / "create-acceptance-measurement-dataset.py"
    spec = importlib.util.spec_from_file_location(
        "acceptance_measurement_dataset_generator",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load legacy generator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LEGACY = _load_legacy_generator()
SourceBank = LEGACY.SourceBank
encode = LEGACY.encode
validate_case = LEGACY.validate_case
stable_seed = LEGACY.stable_seed
sha = LEGACY.sha
file_sha = LEGACY.file_sha
SPEECH_ROOT = LEGACY.SPEECH_ROOT


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _mem_available_bytes() -> int:
    with Path("/proc/meminfo").open(encoding="ascii") as stream:
        for line in stream:
            if line.startswith("MemAvailable:"):
                fields = line.split()
                return int(fields[1]) * 1024
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def _existing_ancestor(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            raise RuntimeError(f"no existing ancestor for {path}")
        candidate = candidate.parent
    return candidate


def _resource_snapshot() -> dict[str, int]:
    usage = shutil.disk_usage(_existing_ancestor(OUTPUT_ROOT))
    return {
        "disk_free_bytes": int(usage.free),
        "mem_available_bytes": _mem_available_bytes(),
    }


def _assert_resource_floor() -> dict[str, int]:
    snapshot = _resource_snapshot()
    if snapshot["disk_free_bytes"] < MIN_FREE_BYTES:
        raise ResourceSafetyStop(
            f"disk free {snapshot['disk_free_bytes']} is below {MIN_FREE_BYTES}"
        )
    if snapshot["mem_available_bytes"] <= MIN_MEM_AVAILABLE_BYTES:
        raise ResourceSafetyStop(
            f"MemAvailable {snapshot['mem_available_bytes']} is not above "
            f"{MIN_MEM_AVAILABLE_BYTES}"
        )
    return snapshot


def _exclusive_create_or_verify(path: Path, payload: bytes) -> bool:
    """Create a new file exclusively, or verify an identical partial artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError:
        existing = path.read_bytes()
        if existing != payload:
            raise FileExistsError(
                f"refusing to overwrite different existing artifact: {path}"
            )
        return False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        raise
    return True


def _clean_speech_records(bank: SourceBank) -> list[dict[str, Any]]:
    records = [
        record
        for speaker_records in bank.speakers.values()
        for record in speaker_records
    ]
    records.sort(key=lambda row: (str(row["speaker"]), str(row["path"])))
    if len(records) < 193:
        raise ValueError(f"expected at least 193 clean speech files, got {len(records)}")
    return records


def _read_record(bank: SourceBank, record: dict[str, Any]) -> tuple[np.ndarray, int]:
    source_path = SPEECH_ROOT / str(record["path"])
    samples, source_rate = bank._read(str(source_path), str(record["sha256"]))
    if source_rate != RATE:
        divisor = math.gcd(RATE, source_rate)
        samples = signal.resample_poly(
            samples,
            RATE // divisor,
            source_rate // divisor,
        )
    return np.asarray(samples, dtype=np.float64), int(source_rate)


def _fade_segment(samples: np.ndarray) -> np.ndarray:
    faded = np.asarray(samples, dtype=np.float64).copy()
    edge = min(int(round(0.005 * RATE)), len(faded) // 2)
    if edge > 0:
        ramp = np.sin(np.linspace(0.0, math.pi / 2.0, edge, dtype=np.float64)) ** 2
        faded[:edge] *= ramp
        faded[-edge:] *= ramp[::-1]
    return faded


def _crop_signature(segments: list[dict[str, Any]]) -> str:
    return sha(_json_bytes(segments))


def _build_speech_crop(
    bank: SourceBank,
    records: list[dict[str, Any]],
    *,
    group_id: int,
    role: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(
        stable_seed(f"{DATASET_ID}:{group_id}:{role}:crop", BASE_SEED)
    )
    order = rng.permutation(len(records))
    parts: list[np.ndarray] = []
    segments: list[dict[str, Any]] = []
    remaining = FRAMES
    max_segment_frames = int(round(MAX_SOURCE_SEGMENT_S * RATE))

    for raw_index in order:
        if remaining <= 0:
            break
        record = records[int(raw_index)]
        samples, source_rate = _read_record(bank, record)
        requested = min(remaining, max_segment_frames, len(samples))
        if requested < int(0.1 * RATE) and remaining >= int(0.1 * RATE):
            continue

        selected: np.ndarray | None = None
        selected_start = 0
        for _ in range(8):
            maximum_start = len(samples) - requested
            start = int(rng.integers(0, maximum_start + 1)) if maximum_start else 0
            candidate = samples[start : start + requested]
            candidate_rms = float(np.sqrt(np.mean(np.square(candidate))))
            if candidate_rms > NON_SILENT_FRAME_RMS:
                selected = candidate
                selected_start = start
                break
        if selected is None:
            continue

        faded = _fade_segment(selected)
        parts.append(faded)
        remaining -= len(faded)
        segment_sha = sha(np.asarray(selected, dtype="<f4").tobytes())
        segments.append(
            {
                "path": str((SPEECH_ROOT / str(record["path"])).relative_to(PROJECT)),
                "sha256": str(record["sha256"]),
                "speaker": str(record["speaker"]),
                "dataset": str(record["dataset"]),
                "license": str(record["license"]),
                "attribution": record.get("attribution"),
                "source_page": record.get("source_page"),
                "source_sample_rate_hz": source_rate,
                "target_sample_rate_hz": RATE,
                "start_sample_at_target_rate": selected_start,
                "selected_samples_at_target_rate": int(len(selected)),
                "selected_audio_sha256_f32le": segment_sha,
                "repeated": False,
            }
        )

    if remaining != 0:
        raise ValueError(
            f"could not assemble {DURATION_S:g}s non-repeated {role} speech crop"
        )

    waveform = np.concatenate(parts)
    waveform -= float(np.mean(waveform))
    peak = float(np.max(np.abs(waveform)))
    if peak <= 1e-8:
        raise ValueError(f"{role} speech crop is silent")
    normalization_gain = NEAR_CONDITIONED_PEAK / peak
    waveform *= normalization_gain
    return waveform, {
        "source_kind": "clean_speech_crop",
        "source_file_count": len(records),
        "selected_segment_count": len(segments),
        "segments": segments,
        "crop_recipe_sha256": _crop_signature(segments),
        "conditioned_peak": NEAR_CONDITIONED_PEAK,
        "peak_normalization_gain_linear": float(normalization_gain),
        "repeated": False,
    }


def _non_silent_seconds(samples: np.ndarray) -> float:
    values = np.asarray(samples, dtype=np.float64)
    frame = RATE // 100
    usable = len(values) - (len(values) % frame)
    if usable <= 0:
        return 0.0
    frames = values[:usable].reshape(-1, frame)
    frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))
    return float(np.count_nonzero(frame_rms > NON_SILENT_FRAME_RMS) / 100.0)


def _assert_delivered_wav(
    blob: bytes,
    *,
    channels: int,
    case_id: str,
    role: str,
) -> None:
    info = sf.info(BytesIO(blob))
    if (
        info.format != "WAV"
        or info.subtype != "PCM_16"
        or info.samplerate != RATE
        or info.frames != FRAMES
        or info.channels != channels
    ):
        raise ValueError(
            f"{case_id} {role} WAV identity mismatch: "
            f"{info.format}/{info.subtype}/{info.samplerate}/"
            f"{info.frames}/{info.channels}"
        )


def _build_cases(bank: SourceBank) -> tuple[list[dict[str, Any]], dict[str, bytes], dict[str, Any]]:
    records = _clean_speech_records(bank)
    cases: list[dict[str, Any]] = []
    blobs: dict[str, bytes] = {}
    quality_cases: list[dict[str, Any]] = []
    case_seeds: set[int] = set()
    near_crop_hashes: set[str] = set()

    for group_id in range(1, SPEECH_GROUPS + CNG_GROUPS + 1):
        if group_id <= SPEECH_GROUPS:
            kind = "speech"
            families = SPEECH_FAMILIES
            item_ids = SPEECH_ITEM_IDS
            analysis_start_s = 0.5
        else:
            kind = "cng"
            families = CNG_FAMILIES
            item_ids = CNG_ITEM_IDS
            analysis_start_s = 4.0

        kind_index = group_id if kind == "speech" else group_id - SPEECH_GROUPS
        case_id = f"sce1-wb-wired-{kind}-{kind_index:02d}-engineering-preview"
        case_seed = stable_seed(f"{DATASET_ID}:{group_id}:{kind}", BASE_SEED)
        if case_seed in case_seeds:
            raise ValueError(f"duplicate case seed for group {group_id}")
        case_seeds.add(case_seed)

        near, near_info = _build_speech_crop(
            bank,
            records,
            group_id=group_id,
            role="near",
        )
        far_input, far_input_info = _build_speech_crop(
            bank,
            records,
            group_id=group_id,
            role="far_input",
        )
        noise, noise_info = bank.noise(MODE, SCENE, RATE, case_seed)

        signals, design = render_signals(
            near,
            far_input,
            noise,
            RATE,
            direction=CONDITIONS["direction"],
            kind=kind,
            bandwidth="WB",
            snr_db=SNR_DB,
            ser_db=-5.0,
            echo_delay_ms=20.0 + group_id,
            seed=case_seed,
            input_level_db=float(CONDITIONS["input_level_db"]),
            volume_gain_db=0.0,
        )
        validation_case = {
            "id": case_id,
            "conditions": dict(CONDITIONS),
            "analysis_start_s": analysis_start_s,
        }
        plan = {"kind": kind, "rate": RATE}
        before_encode_checks = validate_case(signals, validation_case, plan)

        decoded: dict[str, np.ndarray] = {}
        signal_paths: dict[str, str] = {}
        signal_hashes: dict[str, str] = {}
        for role in ROLES:
            blob, decoded[role] = encode(signals[role], RATE)
            _assert_delivered_wav(
                blob,
                channels=2 if role == "microphone" else 1,
                case_id=case_id,
                role=role,
            )
            digest = sha(blob)
            relative = f"assets/audio/{digest}.wav"
            existing = blobs.get(relative)
            if existing is not None and existing != blob:
                raise ValueError(f"SHA collision for {relative}")
            blobs[relative] = blob
            signal_paths[role] = relative
            signal_hashes[role] = digest

        after_encode_checks = validate_case(decoded, validation_case, plan)
        if np.count_nonzero(decoded["far_end"]) != 0:
            raise ValueError(f"{case_id} send far_end is not digital silence")
        if np.asarray(decoded["microphone"]).shape != (FRAMES, 2):
            raise ValueError(f"{case_id} microphone is not stereo")
        for role in ("reference", "far_end", "noise"):
            if np.asarray(decoded[role]).shape != (FRAMES,):
                raise ValueError(f"{case_id} {role} is not mono")

        speech_active_s: float | None = None
        cng_quiet_start_s: float | None = None
        if kind == "speech":
            speech_active_s = _non_silent_seconds(decoded["reference"])
            if speech_active_s < MIN_NON_SILENT_SECONDS:
                raise ValueError(
                    f"{case_id} reference has only {speech_active_s:.2f}s "
                    "non-silent speech"
                )
        else:
            cng_quiet_start_s = 4.0
            start = int(round(cng_quiet_start_s * RATE))
            if (
                np.count_nonzero(decoded["reference"][start:]) != 0
                or np.count_nonzero(decoded["far_end"][start:]) != 0
            ):
                raise ValueError(f"{case_id} CNG quiet window is contaminated")

        peak = max(
            float(np.max(np.abs(values)))
            for values in decoded.values()
        )
        if peak >= 0.7:
            raise ValueError(f"{case_id} decoded peak is {peak}")

        near_crop_hash = str(near_info["crop_recipe_sha256"])
        if near_crop_hash in near_crop_hashes:
            raise ValueError(f"duplicate near crop for group {group_id}")
        near_crop_hashes.add(near_crop_hash)

        provenance = {
            "group_id": group_id,
            "group_key": f"sce1-wb-wired-preview-{group_id:02d}",
            "seed": case_seed,
            "base_seed": BASE_SEED,
            "engineering_preview": True,
            "signal_kind": kind,
            "analysis_start_s": analysis_start_s,
            "near_speech": near_info,
            "far_input": far_input_info,
            "noise": noise_info,
            "far_end_delivery": {
                "digital_silence": True,
                "nonzero_sample_count": 0,
                "reason": "send speech/CNG scenes carry no far-end excitation",
            },
            "render_metadata": design,
            "actual_vehicle_recording": False,
        }
        case = {
            "case_id": case_id,
            "conditions": dict(CONDITIONS),
            "families": list(families),
            "contract_item_ids": list(item_ids),
            "signals": signal_paths,
            "signals_sha256": signal_hashes,
            "provenance": provenance,
        }
        if set(case) != set(scene_contract._CASE_FIELDS):
            raise ValueError(f"{case_id} case fields differ from the v1 contract")
        cases.append(case)
        quality_cases.append(
            {
                "case_id": case_id,
                "group_id": group_id,
                "kind": kind,
                "seed": case_seed,
                "peak": peak,
                "speech_non_silent_seconds": speech_active_s,
                "cng_quiet_window_s": (
                    [cng_quiet_start_s, DURATION_S]
                    if cng_quiet_start_s is not None
                    else None
                ),
                "far_end_nonzero_samples": 0,
                "before_encode_checks": before_encode_checks,
                "after_encode_checks": after_encode_checks,
                "signals_sha256": signal_hashes,
            }
        )
        _assert_resource_floor()

    if len(cases) != scene_contract.PREVIEW_CONDITION_GROUP_COUNT:
        raise ValueError("preview condition group count differs from the contract")
    if Counter(case["provenance"]["signal_kind"] for case in cases) != {
        "speech": SPEECH_GROUPS,
        "cng": CNG_GROUPS,
    }:
        raise ValueError("speech/CNG group split differs from 10/10")
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("case ids are not unique")
    if len({case["provenance"]["seed"] for case in cases}) != len(cases):
        raise ValueError("case seeds are not unique")
    if len({case["provenance"]["near_speech"]["crop_recipe_sha256"] for case in cases}) != len(cases):
        raise ValueError("near speech crops are not unique")

    referenced_assets = {
        case["signals"][role]
        for case in cases
        for role in ROLES
    }
    if set(blobs) != referenced_assets:
        raise ValueError("generated WAV blobs differ from case references")
    far_assets = {case["signals"]["far_end"] for case in cases}
    if len(far_assets) != 1:
        raise ValueError("all send cases must share one digital-silence far_end WAV")
    non_far_assets = {
        case["signals"][role]
        for case in cases
        for role in ROLES
        if role != "far_end"
    }
    if len(non_far_assets) < 54:
        raise ValueError(
            f"expected at least 54 unique non-far WAVs, got {len(non_far_assets)}"
        )

    return cases, blobs, {
        "quality_cases": quality_cases,
        "far_assets": far_assets,
        "non_far_assets": non_far_assets,
    }


def _write_assets(blobs: dict[str, bytes]) -> dict[str, dict[str, Any]]:
    assets: dict[str, dict[str, Any]] = {}
    for relative, blob in sorted(blobs.items()):
        path = OUTPUT_ROOT / relative
        _exclusive_create_or_verify(path, blob)
        digest = sha(blob)
        if path.stat().st_size != len(blob) or file_sha(path) != digest:
            raise ValueError(f"written asset identity mismatch: {relative}")
        assets[relative] = {
            "sha256": digest,
            "size_bytes": len(blob),
        }
    return assets


def _visualization_counts(before: set[str], after: set[str]) -> tuple[int, int]:
    return len(after), len(after - before)


def _cache_names() -> set[str]:
    root = visualization_cache_root()
    if not root.exists():
        return set()
    return {path.name for path in root.glob("*.json")}


def _build_manifest(cases: list[dict[str, Any]]) -> dict[str, Any]:
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "dataset_name": DATASET_NAME,
        "dataset_kind": scene_contract.DATASET_KIND,
        "test_scope": scene_contract.TEST_SCOPE,
        "mode": MODE,
        "scene": SCENE,
        "contract_item_count": scene_contract.CONTRACT_ITEM_COUNT,
        "preview_condition_group_count": scene_contract.PREVIEW_CONDITION_GROUP_COUNT,
        "standard_compliance": False,
        "formal_test_status": scene_contract.FORMAL_TEST_STATUS,
        "cases": cases,
    }
    if set(manifest) != set(scene_contract._MANIFEST_FIELDS):
        raise ValueError("manifest fields differ from the scene-condition contract")
    return manifest


def build() -> dict[str, Any]:
    if MANIFEST_PATH.exists() or MANIFEST_PATH.is_symlink():
        from server.acoustic.acceptance_scene_condition_api import _load_dataset

        _load_dataset(DATASET_ID)
        return {
            "status": "SKIPPED",
            "reason": "existing dataset and runtime binding verified",
            "dataset_id": DATASET_ID,
            "manifest": str(MANIFEST_PATH),
        }

    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    initial_resources = _assert_resource_floor()
    source_hashes = dict(IMPORTED_SOURCE_HASHES)
    if any(file_sha(SOURCE_PATHS[key]) != value for key, value in source_hashes.items()):
        raise ValueError("source code changed after import; no dataset generated")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if OUTPUT_ROOT.is_symlink() or not OUTPUT_ROOT.is_dir():
        raise ValueError("scene-condition output root is not a regular directory")
    (OUTPUT_ROOT / "assets" / "audio").mkdir(parents=True, exist_ok=True)

    bank = SourceBank()
    cases, blobs, counts = _build_cases(bank)
    assets = _write_assets(blobs)
    manifest = _build_manifest(cases)

    referenced_assets = {
        case["signals"][role]
        for case in cases
        for role in ROLES
    }
    if set(assets) != referenced_assets:
        raise ValueError("assets.json set differs from case references")
    for case in cases:
        for role in ROLES:
            relative = case["signals"][role]
            if assets[relative]["sha256"] != case["signals_sha256"][role]:
                raise ValueError(f"{case['case_id']} {role} hash differs from assets")

    _assert_resource_floor()
    cache_before = _cache_names()
    visualizations_processed = precompute_visualizations(OUTPUT_ROOT, assets)
    cache_after = _cache_names()
    _, visualizations_created = _visualization_counts(cache_before, cache_after)
    if visualizations_processed != len(assets):
        raise ValueError("visualization precompute count differs from unique assets")

    completed_at = datetime.now(timezone.utc).isoformat()
    assets_blob = _json_bytes(assets)
    manifest_blob = _json_bytes(manifest)
    far_assets = sorted(counts["far_assets"])
    non_far_assets = sorted(counts["non_far_assets"])

    quality = {
        "schema_version": "scene-condition-preview-quality-v1",
        "dataset_id": DATASET_ID,
        "dataset_name": DATASET_NAME,
        "generated_at": completed_at,
        "generation_started_at": started_at,
        "generation_elapsed_seconds": round(time.monotonic() - started, 6),
        "engineering_preview": True,
        "standard_compliance": False,
        "formal_itu_t_compliance_status": "not_measured",
        "condition_group_count": len(cases),
        "kind_counts": dict(
            Counter(case["provenance"]["signal_kind"] for case in cases)
        ),
        "signal_roles": list(ROLES),
        "signal_count_per_case": len(ROLES),
        "audio": {
            "duration_s": DURATION_S,
            "sample_rate_hz": RATE,
            "subtype": "PCM_16",
            "microphone_channels": 2,
            "reference_channels": 1,
            "far_end_channels": 1,
            "noise_channels": 1,
            "peak_bound": 0.7,
        },
        "unique_wav_count": len(assets),
        "unique_non_far_wav_count": len(non_far_assets),
        "shared_far_end_wavs": far_assets,
        "all_cases_passed_validate_case": True,
        "all_send_far_end_digital_silence": True,
        "speech_reference_minimum_non_silent_seconds": MIN_NON_SILENT_SECONDS,
        "cng_quiet_window_start_s": 4.0,
        "visualizations_processed": visualizations_processed,
        "visualizations_created": visualizations_created,
        "resource_preflight": initial_resources,
        "source_hashes": source_hashes,
        "cases": counts["quality_cases"],
        "limitations": [
            "engineering preview only; not an actual-vehicle recording",
            "controlled hybrid noise proxy with synthetic broadband component",
            "does not establish ITU-T, customer, or product compliance",
            "formal ITU-T compliance was not measured",
        ],
    }
    quality_blob = _json_bytes(quality)
    quality_sha = sha(quality_blob)
    summary = {
        "schema_version": "scene-condition-preview-summary-v1",
        "dataset_id": DATASET_ID,
        "dataset_name": DATASET_NAME,
        "generated_at": completed_at,
        "generation_started_at": started_at,
        "engineering_preview": True,
        "standard_compliance": False,
        "formal_itu_t_compliance_status": "not_measured",
        "condition_group_count": len(cases),
        "kind_counts": quality["kind_counts"],
        "contract_item_count": scene_contract.CONTRACT_ITEM_COUNT,
        "signal_roles": list(ROLES),
        "duration_s": DURATION_S,
        "sample_rate_hz": RATE,
        "subtype": "PCM_16",
        "unique_wav_count": len(assets),
        "unique_non_far_wav_count": len(non_far_assets),
        "shared_far_end_wav_count": len(far_assets),
        "visualizations_processed": visualizations_processed,
        "visualizations_created": visualizations_created,
        "manifest_sha256": sha(manifest_blob),
        "assets_sha256": sha(assets_blob),
        "quality_sha256": quality_sha,
        "source_hashes": source_hashes,
        "scope": (
            "20 independent engineering condition groups for the idle-page "
            "background-noise preview; not formal certification evidence"
        ),
    }
    summary_blob = _json_bytes(summary)

    runtime_binding = {
        "schema_version": "scene-condition-runtime-binding-v1",
        "dataset_id": DATASET_ID,
        "status": "reconciled",
        "generation_contract_sha256": source_hashes["scene_contract_sha256"],
        "runtime_contract_sha256": source_hashes["scene_contract_sha256"],
        "difference": {"scope": "generation_contract_equals_runtime"},
        "artifact_sha256": {
            "manifest.json": sha(manifest_blob),
            "assets.json": sha(assets_blob),
            "summary.json": sha(summary_blob),
            "quality.json": sha(quality_blob),
        },
        "standard_compliance": False,
        "formal_test_status": "not_measured",
    }

    if any(file_sha(SOURCE_PATHS[key]) != value for key, value in source_hashes.items()):
        raise ValueError("source code changed during generation")

    _exclusive_create_or_verify(ASSETS_PATH, assets_blob)
    _exclusive_create_or_verify(QUALITY_PATH, quality_blob)
    _exclusive_create_or_verify(SUMMARY_PATH, summary_blob)
    _exclusive_create_or_verify(
        OUTPUT_ROOT / "runtime-contract-binding.json", _json_bytes(runtime_binding)
    )
    _exclusive_create_or_verify(MANIFEST_PATH, manifest_blob)

    loaded = scene_contract.load_scene_condition_dataset(DATASET_ID)
    if (
        loaded["contract_item_count"] != scene_contract.CONTRACT_ITEM_COUNT
        or loaded["preview_condition_group_count"] != len(cases)
        or len(loaded["referenced_assets"]) != len(assets)
    ):
        raise ValueError("scene-condition contract load returned inconsistent counts")

    return {
        "status": "CREATED",
        "dataset_id": DATASET_ID,
        "dataset_name": DATASET_NAME,
        "root": str(OUTPUT_ROOT),
        "condition_groups": len(cases),
        "kind_counts": quality["kind_counts"],
        "signal_roles": list(ROLES),
        "unique_wavs": len(assets),
        "unique_non_far_wavs": len(non_far_assets),
        "shared_far_end_wavs": len(far_assets),
        "visualizations_processed": visualizations_processed,
        "visualizations_created": visualizations_created,
        "standard_compliance": False,
        "formal_itu_t_compliance_status": "not_measured",
        "manifest_sha256": summary["manifest_sha256"],
        "assets_sha256": summary["assets_sha256"],
        "quality_sha256": summary["quality_sha256"],
        "resource_preflight": initial_resources,
    }


def main() -> int:
    try:
        result = build()
    except ResourceSafetyStop as error:
        print(
            json.dumps(
                {
                    "status": "STOPPED_RESOURCE_SAFETY_FLOOR",
                    "error": str(error),
                    **_resource_snapshot(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 75
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
