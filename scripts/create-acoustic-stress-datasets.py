#!/usr/bin/env python3
"""Create one 100-workload echo or double-talk engineering stress dataset."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import errno
import importlib.util
import json
import math
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


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONVERSATION = _load_module(
    SCRIPTS / "create-background-noise-conversation-dataset.py",
    "itu_background_conversation_for_stress_generator",
)
STRESS = _load_module(
    SCRIPTS / "acoustic_stress_signal_design.py",
    "itu_acoustic_stress_signal_design",
)

from server.acoustic import acceptance_scene_conditions as legacy_contract  # noqa: E402
from server.acoustic import service  # noqa: E402
from server.acoustic.acceptance_scope import ACCEPTANCE_MODES  # noqa: E402
from server.acoustic.acceptance_contract import required_conditions  # noqa: E402
from server.acoustic.acceptance_visualization import (  # noqa: E402
    cache_root as visualization_cache_root,
    precompute_visualizations,
)
from server.acoustic.acoustic_stress_profiles import (  # noqa: E402
    CATEGORIES,
    stress_profile,
)


SOURCE_PATHS = {
    "server/acoustic/acoustic_stress_profiles.py": (
        PROJECT / "server/acoustic/acoustic_stress_profiles.py"
    ),
    "server/acoustic/acceptance_scope.py": (
        PROJECT / "server/acoustic/acceptance_scope.py"
    ),
    "server/acoustic/acceptance_contract.py": (
        PROJECT / "server/acoustic/acceptance_contract.py"
    ),
    "server/acoustic/acceptance_scene_conditions.py": (
        PROJECT / "server/acoustic/acceptance_scene_conditions.py"
    ),
    "scripts/acoustic_stress_signal_design.py": (
        PROJECT / "scripts/acoustic_stress_signal_design.py"
    ),
    "scripts/create-acoustic-stress-datasets.py": Path(__file__),
    "scripts/create-background-noise-conversation-dataset.py": (
        PROJECT / "scripts/create-background-noise-conversation-dataset.py"
    ),
    "scripts/create-background-noise-scene-condition-dataset.py": (
        PROJECT / "scripts/create-background-noise-scene-condition-dataset.py"
    ),
    "scripts/create-acceptance-measurement-dataset.py": (
        PROJECT / "scripts/create-acceptance-measurement-dataset.py"
    ),
    "scripts/acceptance_signal_design.py": (
        PROJECT / "scripts/acceptance_signal_design.py"
    ),
    "cpp/evaluation/nv200-test-contract.json": (
        PROJECT / "cpp/evaluation/nv200-test-contract.json"
    ),
}
IMPORTED_SOURCE_HASHES = {
    relative: CONVERSATION.file_sha(path)
    for relative, path in SOURCE_PATHS.items()
}

RATE_16K = 16_000
RATE_SWB = 48_000
DURATION_S = 12.0
BASE_SEED = 20261110
CATEGORY_CASE_COUNT = 20
MIN_FREE_BYTES = 100 * 1024**3
MIN_MEM_AVAILABLE_BYTES = 5 * 1024**3
VOLUME_GAIN_DB = {
    "nominal": 0.0,
    "recommended": -2.0,
    "maximum": 6.0,
}
ROLES = legacy_contract.ROLES


class ResourceSafetyStop(RuntimeError):
    """Raised before further writes when a hard safety floor is crossed."""


def configure_dataset(scope: str, mode: str) -> None:
    global PROFILE, DATASET_ID, DATASET_NAME, DATASET_KIND, SCHEMA_VERSION
    global SCOPE, MODE, SCENE, BANDWIDTH, RATE, FRAMES, OUTPUT_ROOT
    global MANIFEST_PATH, ASSETS_PATH, SUMMARY_PATH, QUALITY_PATH, BINDING_PATH

    profile = stress_profile(scope, mode)
    if profile.case_count != 100 or profile.observation_count != 100:
        raise ValueError("stress profile must expose 100 cases and observations")
    if profile.category_counts != {
        category: CATEGORY_CASE_COUNT
        for category in STRESS.SCOPE_CATEGORIES[scope]
    }:
        raise ValueError("stress profile category counts differ from five-by-twenty")
    if not profile.contract_rows or any(
        len([row for row in profile.contract_rows if row["id"] == item_id]) != 1
        for item_id in profile.contract_item_ids
    ):
        raise ValueError("stress profile contract rows are not unique")
    if scope == "echo" and mode.startswith("SWB-"):
        if profile.contract_item_count != 3:
            raise ValueError("SWB echo must expose three contract rows")
    elif profile.contract_item_count != 2:
        raise ValueError("non-SWB stress datasets must expose two contract rows")

    PROFILE = profile
    DATASET_ID = profile.dataset_id
    DATASET_NAME = profile.dataset_name
    DATASET_KIND = profile.dataset_kind
    SCHEMA_VERSION = profile.schema_version
    SCOPE = profile.scope
    MODE = profile.mode
    SCENE = profile.scene
    BANDWIDTH = mode.split("-", 1)[0]
    RATE = RATE_SWB if BANDWIDTH == "SWB" else RATE_16K
    FRAMES = int(round(DURATION_S * RATE))
    OUTPUT_ROOT = service.BUILD / "acoustic-stress-datasets" / DATASET_ID
    MANIFEST_PATH = OUTPUT_ROOT / "manifest.json"
    ASSETS_PATH = OUTPUT_ROOT / "assets.json"
    SUMMARY_PATH = OUTPUT_ROOT / "summary.json"
    QUALITY_PATH = OUTPUT_ROOT / "quality.json"
    BINDING_PATH = OUTPUT_ROOT / "runtime-contract-binding.json"


configure_dataset("echo", "WB-wired")


def _json_bytes(value: Any) -> bytes:
    return CONVERSATION._json_bytes(value)


def _exclusive_create_or_verify(path: Path, payload: bytes) -> bool:
    return CONVERSATION._exclusive_create_or_verify(path, payload)


def _resource_snapshot() -> dict[str, int]:
    usage = shutil.disk_usage(CONVERSATION._existing_ancestor(OUTPUT_ROOT))
    return {
        "disk_free_bytes": int(usage.free),
        "mem_available_bytes": CONVERSATION._mem_available_bytes(),
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


def _source_hash_snapshot() -> dict[str, str]:
    current = {
        relative: CONVERSATION.file_sha(path)
        for relative, path in SOURCE_PATHS.items()
    }
    if current != IMPORTED_SOURCE_HASHES:
        changed = sorted(
            relative
            for relative in current
            if current[relative] != IMPORTED_SOURCE_HASHES.get(relative)
        )
        raise ValueError(
            "stress source code changed after import; no dataset generated "
            f"(changed={changed})"
        )
    return current


def _case_plan() -> list[dict[str, Any]]:
    rows = PROFILE.contract_rows
    row_by_id = {row["id"]: row for row in rows}
    plan: list[dict[str, Any]] = []
    group_id = 0
    for category in STRESS.SCOPE_CATEGORIES[SCOPE]:
        for category_index in range(1, CATEGORY_CASE_COUNT + 1):
            group_id += 1
            row = rows[(category_index - 1) % len(rows)]
            item_id = row["id"]
            conditions = PROFILE.conditions_for_item(item_id)
            if conditions != required_conditions(row):
                raise ValueError("stress condition derivation differs from contract")
            volume = conditions["volume"]
            if volume not in VOLUME_GAIN_DB:
                raise ValueError(f"unsupported contract volume: {volume}")
            plan.append({
                "group_id": group_id,
                "category": category,
                "category_index": category_index,
                "case_id": (
                    f"{SCOPE}-{MODE}-{category}-{category_index:02d}-stress-v1"
                ),
                "item_id": item_id,
                "contract_row": row_by_id[item_id],
                "conditions": conditions,
                "volume_gain_db": VOLUME_GAIN_DB[volume],
            })
    if Counter(row["category"] for row in plan) != {
        category: CATEGORY_CASE_COUNT
        for category in STRESS.SCOPE_CATEGORIES[SCOPE]
    }:
        raise ValueError("stress case plan does not expose 20 cases per category")
    if len({row["case_id"] for row in plan}) != len(plan):
        raise ValueError("stress case IDs are not unique")
    if any(len(row["contract_row"]) == 0 for row in plan):
        raise ValueError("stress case plan contains an empty contract row")
    return plan


def stress_matrix_targets() -> list[tuple[str, str]]:
    return [
        (scope, mode)
        for scope in ("echo", "double_talk")
        for mode in sorted(ACCEPTANCE_MODES)
    ]


def _crop_16k(
    bank,
    records: list[dict[str, Any]],
    *,
    group_id: int,
    role: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    CONVERSATION.OLD.DATASET_ID = DATASET_ID
    waveform, raw_info = CONVERSATION.OLD._build_speech_crop(
        bank,
        records,
        group_id=group_id,
        role=role,
    )
    info = deepcopy(raw_info)
    segments = info.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"{role} crop has no source segments")
    paths = [str(segment["path"]) for segment in segments]
    if len(paths) != len(set(paths)) or info.get("repeated") is not False:
        raise ValueError(f"{role} crop repeats a source inside one case")
    samples_16k = np.asarray(waveform, dtype=np.float64)
    if RATE == RATE_SWB:
        samples = signal.resample_poly(samples_16k, 3, 1)
        resampling = {
            "method": "scipy.signal.resample_poly",
            "up": 3,
            "down": 1,
            "source_rate_hz": RATE_16K,
            "target_rate_hz": RATE_SWB,
        }
    else:
        samples = samples_16k.copy()
        resampling = {
            "method": "none",
            "source_rate_hz": RATE_16K,
            "target_rate_hz": RATE,
        }
    if len(samples) != FRAMES:
        raise ValueError(f"{role} resampled crop length differs from 12 seconds")
    info.update({
        "speaker_pool": role,
        "target_sample_rate_hz": RATE,
        "resampling": resampling,
        "source_reuse_allowed_across_cases": True,
        "within_case_source_repetition": False,
    })
    return np.asarray(samples, dtype=np.float64), info


def _silence_info(role: str, reason: str) -> dict[str, Any]:
    return {
        "source_kind": "digital_silence",
        "role": role,
        "reason": reason,
        "repeated": False,
    }


def _noise_crop(
    bank,
    *,
    seed: int,
    group_id: int,
    used_offsets: set[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    for attempt in range(64):
        noise_seed = CONVERSATION.stable_seed(
            f"{DATASET_ID}:{group_id}:stress-noise:{attempt}",
            seed,
        )
        noise, raw_info = bank.noise(MODE, SCENE, RATE, noise_seed)
        info = deepcopy(raw_info)
        offset = int(info["start_sample"])
        if offset in used_offsets:
            continue
        used_offsets.add(offset)
        source_path = Path(str(info["path"]))
        source_info = sf.info(source_path)
        info.update({
            "noise_crop_seed": noise_seed,
            "crop_offset_samples": offset,
            "source_duration_s": round(
                source_info.frames / source_info.samplerate,
                9,
            ),
            "source_reuse_allowed_across_cases": True,
            "within_case_source_repetition": False,
        })
        return np.asarray(noise, dtype=np.float64), info
    raise ValueError("could not choose a unique stress noise crop offset")


def _assert_wav(
    blob: bytes,
    *,
    channels: int,
    case_id: str,
    role: str,
) -> None:
    from io import BytesIO

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


def _source_reuse_policy(inventory: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_definition": "100_engineering_workloads",
        "independent_source_recordings_claimed": False,
        "source_audio_may_be_reused_across_cases": True,
        "within_case_source_looping": False,
        "noise_crop_may_be_reused_across_cases": False,
        "speaker_pools_disjoint_for_near_and_far": True,
        "source_inventory": deepcopy(inventory),
    }


def _build_cases(
    bank,
) -> tuple[list[dict[str, Any]], dict[str, bytes], dict[str, Any]]:
    near_records, far_records, speaker_inventory = CONVERSATION._speaker_pools(bank)
    inventory = {
        **speaker_inventory,
        "noise_scene": SCENE,
        "noise_mode": MODE,
    }
    reuse_policy = _source_reuse_policy(inventory)
    cases: list[dict[str, Any]] = []
    blobs: dict[str, bytes] = {}
    quality_cases: list[dict[str, Any]] = []
    used_noise_offsets: set[int] = set()
    used_seeds: set[int] = set()
    source_files_used: set[str] = set()

    for plan in _case_plan():
        category = str(plan["category"])
        group_id = int(plan["group_id"])
        case_id = str(plan["case_id"])
        case_seed = CONVERSATION.stable_seed(
            f"{DATASET_ID}:{group_id}:{category}:stress-case",
            BASE_SEED,
        )
        if case_seed in used_seeds:
            raise ValueError("stress case seed is not unique")
        used_seeds.add(case_seed)

        if SCOPE == "echo":
            near_input = np.zeros(FRAMES, dtype=np.float64)
            near_info = _silence_info(
                "near",
                "echo workloads require strict zero near reference",
            )
        else:
            near_input, near_info = _crop_16k(
                bank,
                near_records,
                group_id=group_id,
                role="near",
            )
        far_input, far_info = _crop_16k(
            bank,
            far_records,
            group_id=group_id,
            role="far_input",
        )
        for info in (near_info, far_info):
            for segment in info.get("segments", []):
                source_files_used.add(str(segment["path"]))
        noise, noise_info = _noise_crop(
            bank,
            seed=case_seed,
            group_id=group_id,
            used_offsets=used_noise_offsets,
        )
        signals, design = STRESS.render_stress_signals(
            near_input,
            far_input,
            noise,
            RATE,
            scope=SCOPE,
            category=category,
            volume_gain_db=float(plan["volume_gain_db"]),
            seed=case_seed,
            case_index=int(plan["category_index"]),
            bandwidth=BANDWIDTH,
        )
        if design["kind"] != "speech":
            raise ValueError(f"{case_id} stress render kind is not speech")
        if design["analysis_start_s"] != STRESS.analysis_start_for(SCOPE, category):
            raise ValueError(f"{case_id} analysis start differs from category")
        if design["source_mix_check"]["residual_max_abs"] > 1e-12:
            raise ValueError(f"{case_id} pre-PCM stress mixture exceeds 1e-12")

        decoded: dict[str, np.ndarray] = {}
        signal_paths: dict[str, str] = {}
        signal_hashes: dict[str, str] = {}
        for role in ROLES:
            blob, delivered = CONVERSATION.encode(signals[role], RATE)
            _assert_wav(
                blob,
                channels=2 if role == "microphone" else 1,
                case_id=case_id,
                role=role,
            )
            digest = CONVERSATION.sha(blob)
            relative = f"assets/audio/{digest}.wav"
            existing = blobs.get(relative)
            if existing is not None and existing != blob:
                raise ValueError(f"SHA collision for {relative}")
            blobs[relative] = blob
            decoded[role] = np.asarray(delivered)
            signal_paths[role] = relative
            signal_hashes[role] = digest

        mixture_check = STRESS.verify_stress_mixture(signals, decoded, design)
        activity_check = STRESS.assert_stress_activity(
            decoded,
            design,
            scope=SCOPE,
            category=category,
        )
        peak = max(
            float(np.max(np.abs(values)))
            for values in decoded.values()
        )
        if peak >= 0.7:
            raise ValueError(f"{case_id} decoded peak is {peak}")
        conversation_mode = STRESS.conversation_mode_for(SCOPE, category)
        provenance = {
            "group_id": group_id,
            "group_key": f"{SCOPE}-{MODE}-stress-{group_id:03d}",
            "seed": case_seed,
            "base_seed": BASE_SEED,
            "scope": SCOPE,
            "mode": MODE,
            "scene": SCENE,
            "workload_category": category,
            "category_index": int(plan["category_index"]),
            "conversation_mode": conversation_mode,
            "contract_item_id": str(plan["item_id"]),
            "volume": str(plan["conditions"]["volume"]),
            "volume_gain_db": float(plan["volume_gain_db"]),
            "conditions": dict(plan["conditions"]),
            "analysis_start_s": float(design["analysis_start_s"]),
            "beam_delays_ms": list(design["beam_delays_ms"]),
            "activity_segments": deepcopy(design["activity_segments"]),
            "near_speech": near_info,
            "far_input": far_info,
            "noise": noise_info,
            "source_reuse_policy": deepcopy(reuse_policy),
            "render_metadata": design,
            "engineering_preview": True,
            "engineering_combination": True,
            "independent_source_recordings": False,
            "actual_vehicle_recording": False,
            "within_case_source_looping": False,
        }
        case = {
            "case_id": case_id,
            "conditions": dict(plan["conditions"]),
            "families": [str(plan["contract_row"]["family"])],
            "contract_item_ids": [str(plan["item_id"])],
            "signals": signal_paths,
            "signals_sha256": signal_hashes,
            "provenance": provenance,
        }
        if set(case) != set(legacy_contract._CASE_FIELDS):
            raise ValueError(f"{case_id} case fields differ from legacy contract")
        cases.append(case)
        quality_cases.append({
            "case_id": case_id,
            "workload_category": category,
            "category_index": int(plan["category_index"]),
            "contract_item_id": str(plan["item_id"]),
            "volume": str(plan["conditions"]["volume"]),
            "volume_gain_db": float(plan["volume_gain_db"]),
            "analysis_start_s": float(design["analysis_start_s"]),
            "noise_snr_db_target": design["noise_snr_db_target"],
            "noise_snr_db_measured": design["noise_snr_db_measured"],
            "ser_db_effective_target": design["ser_db_effective_target"],
            "ser_db_measured": design["ser_db_measured"],
            "noise_crop_offset_samples": int(noise_info["crop_offset_samples"]),
            "mixture_check": mixture_check,
            "activity_check": activity_check,
            "signals_sha256": signal_hashes,
            "peak": peak,
        })
        _assert_resource_floor()

    if len(cases) != 100:
        raise ValueError("stress dataset must contain exactly 100 cases")
    category_counts = Counter(
        case["provenance"]["workload_category"] for case in cases
    )
    if category_counts != {
        category: CATEGORY_CASE_COUNT
        for category in STRESS.SCOPE_CATEGORIES[SCOPE]
    }:
        raise ValueError("stress category counts differ from 20 each")
    item_counts = Counter(
        case["contract_item_ids"][0] for case in cases
    )
    if set(item_counts) != set(PROFILE.contract_item_ids):
        raise ValueError("stress cases do not cover every contract row")
    referenced_assets = {
        case["signals"][role]
        for case in cases
        for role in ROLES
    }
    if set(blobs) != referenced_assets:
        raise ValueError("stress generated assets differ from references")
    if len(used_noise_offsets) != len(cases):
        raise ValueError("stress noise crop offsets are not unique")
    return cases, blobs, {
        "quality_cases": quality_cases,
        "inventory": inventory,
        "reuse_policy": reuse_policy,
        "category_counts": dict(category_counts),
        "item_counts": dict(item_counts),
        "source_files_used_count": len(source_files_used),
        "noise_crop_offset_count": len(used_noise_offsets),
    }


def _write_assets(blobs: dict[str, bytes]) -> dict[str, dict[str, Any]]:
    assets: dict[str, dict[str, Any]] = {}
    for relative, blob in sorted(blobs.items()):
        path = OUTPUT_ROOT / relative
        _exclusive_create_or_verify(path, blob)
        digest = CONVERSATION.sha(blob)
        if (
            path.stat().st_size != len(blob)
            or CONVERSATION.file_sha(path) != digest
        ):
            raise ValueError(f"written stress asset identity mismatch: {relative}")
        assets[relative] = {
            "sha256": digest,
            "size_bytes": len(blob),
        }
        _assert_resource_floor()
    return assets


def _visualization_status(
    assets: dict[str, dict[str, Any]],
) -> tuple[int, int]:
    root = visualization_cache_root()
    hashes = {str(metadata["sha256"]) for metadata in assets.values()}
    before = {
        digest: (root / f"{digest}.json").exists()
        for digest in hashes
    }
    processed = precompute_visualizations(OUTPUT_ROOT, assets)
    after = {
        digest: (root / f"{digest}.json").exists()
        for digest in hashes
    }
    created = sum(
        1
        for digest in hashes
        if not before[digest] and after[digest]
    )
    return processed, created


def _build_manifest(cases: list[dict[str, Any]]) -> dict[str, Any]:
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "dataset_name": DATASET_NAME,
        "dataset_kind": DATASET_KIND,
        "test_scope": SCOPE,
        "mode": MODE,
        "scene": SCENE,
        "contract_item_count": PROFILE.contract_item_count,
        "preview_condition_group_count": 100,
        "standard_compliance": False,
        "formal_test_status": legacy_contract.FORMAL_TEST_STATUS,
        "cases": cases,
    }
    if set(manifest) != set(legacy_contract._MANIFEST_FIELDS):
        raise ValueError("stress manifest differs from legacy field set")
    return manifest


def _runtime_binding(
    *,
    artifact_sha256: dict[str, str],
    source_sha256: dict[str, str],
) -> dict[str, Any]:
    required_artifacts = {
        "manifest.json",
        "assets.json",
        "summary.json",
        "quality.json",
    }
    if set(artifact_sha256) != required_artifacts:
        raise ValueError("stress runtime binding artifact set is incomplete")
    required_sources = {
        "server/acoustic/acoustic_stress_profiles.py",
        "scripts/create-acoustic-stress-datasets.py",
        "scripts/acoustic_stress_signal_design.py",
        "scripts/create-background-noise-conversation-dataset.py",
        "scripts/create-background-noise-scene-condition-dataset.py",
        "scripts/acceptance_signal_design.py",
        "server/acoustic/acceptance_scope.py",
        "cpp/evaluation/nv200-test-contract.json",
    }
    if not required_sources.issubset(source_sha256):
        raise ValueError("stress runtime binding source set is incomplete")
    return {
        "schema_version": "acoustic-stress-runtime-binding-v1",
        "dataset_id": DATASET_ID,
        "status": "reconciled",
        "artifact_sha256": dict(artifact_sha256),
        "source_sha256": dict(source_sha256),
    }


def _assert_no_partial_json_target() -> None:
    partial = [
        path
        for path in (ASSETS_PATH, SUMMARY_PATH, QUALITY_PATH, BINDING_PATH)
        if path.exists() or path.is_symlink()
    ]
    if partial:
        raise FileExistsError(
            "refusing to overwrite partial stress artifacts: "
            + ", ".join(str(path) for path in partial)
        )


def _verify_existing_dataset() -> dict[str, Any]:
    required = (
        MANIFEST_PATH,
        ASSETS_PATH,
        SUMMARY_PATH,
        QUALITY_PATH,
        BINDING_PATH,
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileExistsError(
            "existing stress target is incomplete: "
            + ", ".join(str(path) for path in missing)
        )
    binding = json.loads(BINDING_PATH.read_text(encoding="utf-8"))
    sources = _source_hash_snapshot()
    if (
        not isinstance(binding, dict)
        or binding.get("schema_version")
        != "acoustic-stress-runtime-binding-v1"
        or binding.get("dataset_id") != DATASET_ID
        or binding.get("status") != "reconciled"
        or binding.get("source_sha256") != sources
    ):
        raise ValueError("existing stress runtime binding is stale")
    artifact_sha256 = binding.get("artifact_sha256")
    if (
        not isinstance(artifact_sha256, dict)
        or set(artifact_sha256) != {
            "manifest.json",
            "assets.json",
            "summary.json",
            "quality.json",
        }
    ):
        raise ValueError("existing stress binding artifacts are invalid")
    for name, expected in artifact_sha256.items():
        if CONVERSATION.file_sha(OUTPUT_ROOT / name) != expected:
            raise ValueError(f"existing stress artifact changed: {name}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        set(manifest) != set(legacy_contract._MANIFEST_FIELDS)
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("dataset_id") != DATASET_ID
        or manifest.get("dataset_kind") != DATASET_KIND
        or manifest.get("preview_condition_group_count") != 100
        or not isinstance(manifest.get("cases"), list)
        or len(manifest["cases"]) != 100
    ):
        raise ValueError("existing stress manifest identity is invalid")
    return {
        "status": "SKIPPED",
        "reason": "existing stress dataset and runtime binding verified",
        "dataset_id": DATASET_ID,
        "manifest": str(MANIFEST_PATH),
    }


def _verify_store() -> dict[str, Any]:
    from server.acoustic.acoustic_stress_store import load_stress_dataset

    loaded = load_stress_dataset(DATASET_ID)
    cases = loaded.get("cases")
    if (
        loaded.get("dataset_id") != DATASET_ID
        or not isinstance(cases, list)
        or len(cases) != 100
    ):
        raise ValueError("stress store verification returned invalid identity")
    return {
        "store_verified": True,
        "store_case_count": len(cases),
        "store_referenced_asset_count": len(
            loaded.get("referenced_assets", set())
        ),
    }


def build(*, verify_store: bool = True) -> dict[str, Any]:
    if MANIFEST_PATH.exists() or MANIFEST_PATH.is_symlink():
        result = _verify_existing_dataset()
        if verify_store:
            result.update(_verify_store())
        return result
    _assert_no_partial_json_target()
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    initial_resources = _assert_resource_floor()
    source_sha256 = _source_hash_snapshot()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if OUTPUT_ROOT.is_symlink() or not OUTPUT_ROOT.is_dir():
        raise ValueError("stress output root is not a regular directory")
    (OUTPUT_ROOT / "assets" / "audio").mkdir(parents=True, exist_ok=True)

    bank = CONVERSATION.SourceBank()
    cases, blobs, counts = _build_cases(bank)
    if _source_hash_snapshot() != source_sha256:
        raise ValueError("stress source code changed while building cases")
    assets = _write_assets(blobs)
    manifest = _build_manifest(cases)
    referenced_assets = {
        case["signals"][role]
        for case in cases
        for role in ROLES
    }
    if set(assets) != referenced_assets:
        raise ValueError("stress assets set differs from case references")
    for case in cases:
        for role in ROLES:
            relative = case["signals"][role]
            if assets[relative]["sha256"] != case["signals_sha256"][role]:
                raise ValueError(f"{case['case_id']} {role} asset hash differs")

    _assert_resource_floor()
    visualizations_processed, visualizations_created = _visualization_status(assets)
    if visualizations_processed != len(assets):
        raise ValueError("stress visualization count differs from assets")

    completed_at = datetime.now(timezone.utc).isoformat()
    assets_blob = _json_bytes(assets)
    manifest_blob = _json_bytes(manifest)
    quality = {
        "schema_version": "acoustic-stress-quality-v1",
        "dataset_id": DATASET_ID,
        "dataset_name": DATASET_NAME,
        "dataset_kind": DATASET_KIND,
        "scope": SCOPE,
        "mode": MODE,
        "scene": SCENE,
        "bandwidth": BANDWIDTH,
        "sample_rate_hz": RATE,
        "duration_s": DURATION_S,
        "generated_at": completed_at,
        "generation_started_at": started_at,
        "generation_elapsed_seconds": round(time.monotonic() - started, 6),
        "case_count": len(cases),
        "observation_count": len(cases),
        "category_counts": counts["category_counts"],
        "contract_item_counts": counts["item_counts"],
        "contract_item_count": PROFILE.contract_item_count,
        "volume_counts": dict(Counter(
            case["conditions"]["volume"] for case in cases
        )),
        "source_reuse_policy": counts["reuse_policy"],
        "source_inventory": counts["inventory"],
        "source_files_used_count": counts["source_files_used_count"],
        "noise_crop_offset_count": counts["noise_crop_offset_count"],
        "unique_wav_count": len(assets),
        "all_cases_passed_activity_checks": True,
        "all_cases_passed_pcm_mixture_checks": True,
        "visualizations_processed": visualizations_processed,
        "visualizations_created_for_dataset": visualizations_created,
        "resource_preflight": initial_resources,
        "source_sha256": source_sha256,
        "engineering_preview": True,
        "standard_compliance": False,
        "formal_itu_t_compliance_status": "not_measured",
        "cases": counts["quality_cases"],
        "limitations": [
            "100 engineering workloads, not 100 independent source recordings",
            "source recordings may recur across cases; no within-case looping",
            "synthetic FIR paths are engineering stress stimuli",
            "does not establish ITU-T, customer, or product compliance",
        ],
    }
    quality_blob = _json_bytes(quality)
    summary = {
        "schema_version": "acoustic-stress-summary-v1",
        "dataset_id": DATASET_ID,
        "dataset_name": DATASET_NAME,
        "dataset_kind": DATASET_KIND,
        "scope": SCOPE,
        "mode": MODE,
        "scene": SCENE,
        "bandwidth": BANDWIDTH,
        "sample_rate_hz": RATE,
        "duration_s": DURATION_S,
        "generated_at": completed_at,
        "case_count": len(cases),
        "observation_count": len(cases),
        "category_counts": counts["category_counts"],
        "contract_item_counts": counts["item_counts"],
        "volume_counts": quality["volume_counts"],
        "unique_wav_count": len(assets),
        "source_reuse_policy": counts["reuse_policy"],
        "source_sha256": source_sha256,
        "manifest_sha256": CONVERSATION.sha(manifest_blob),
        "assets_sha256": CONVERSATION.sha(assets_blob),
        "quality_sha256": CONVERSATION.sha(quality_blob),
        "engineering_preview": True,
        "standard_compliance": False,
        "formal_itu_t_compliance_status": "not_measured",
    }
    summary_blob = _json_bytes(summary)
    binding = _runtime_binding(
        artifact_sha256={
            "manifest.json": CONVERSATION.sha(manifest_blob),
            "assets.json": CONVERSATION.sha(assets_blob),
            "summary.json": CONVERSATION.sha(summary_blob),
            "quality.json": CONVERSATION.sha(quality_blob),
        },
        source_sha256=source_sha256,
    )
    if _source_hash_snapshot() != source_sha256:
        raise ValueError("stress source code changed during generation")
    _exclusive_create_or_verify(ASSETS_PATH, assets_blob)
    _exclusive_create_or_verify(QUALITY_PATH, quality_blob)
    _exclusive_create_or_verify(SUMMARY_PATH, summary_blob)
    _exclusive_create_or_verify(BINDING_PATH, _json_bytes(binding))
    _exclusive_create_or_verify(MANIFEST_PATH, manifest_blob)

    store_result: dict[str, Any] = {
        "store_verified": False,
        "store_verification": "deferred",
    }
    if verify_store:
        store_result = _verify_store()
    return {
        "status": "CREATED",
        "dataset_id": DATASET_ID,
        "dataset_name": DATASET_NAME,
        "root": str(OUTPUT_ROOT),
        "case_count": len(cases),
        "observation_count": len(cases),
        "category_counts": counts["category_counts"],
        "contract_item_counts": counts["item_counts"],
        "unique_wavs": len(assets),
        "visualizations_processed": visualizations_processed,
        "visualizations_created_for_dataset": visualizations_created,
        "manifest_sha256": summary["manifest_sha256"],
        "assets_sha256": summary["assets_sha256"],
        "quality_sha256": summary["quality_sha256"],
        "resource_preflight": initial_resources,
        **store_result,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create one 100-workload acoustic stress preview dataset."
    )
    parser.add_argument("--scope", choices=("echo", "double_talk"), required=True)
    parser.add_argument(
        "--mode",
        choices=(*sorted(ACCEPTANCE_MODES), "all"),
        required=True,
    )
    parser.add_argument(
        "--defer-store-verification",
        action="store_true",
        help="Defer store verification while the stress store is being aligned.",
    )
    args = parser.parse_args(argv)
    try:
        if args.mode == "all":
            targets = stress_matrix_targets()
            results = []
            for scope, mode in targets:
                configure_dataset(scope, mode)
                results.append(build(verify_store=False))
            store_results = []
            if not args.defer_store_verification:
                for scope, mode in targets:
                    configure_dataset(scope, mode)
                    store_results.append({
                        "scope": scope,
                        "mode": mode,
                        **_verify_store(),
                    })
            result = {
                "status": "COMPLETED_ALL_16",
                "dataset_count": len(results),
                "datasets": results,
                "store_verified": not args.defer_store_verification,
                "store_results": store_results,
            }
        else:
            configure_dataset(args.scope, args.mode)
            result = build(verify_store=not args.defer_store_verification)
    except ResourceSafetyStop as error:
        print(json.dumps({
            "status": "STOPPED_RESOURCE_SAFETY_FLOOR",
            "error": str(error),
            **_resource_snapshot(),
        }, ensure_ascii=False, indent=2))
        return 75
    except OSError as error:
        if error.errno != errno.ENOSPC:
            raise
        print(json.dumps({
            "status": "STOPPED_ENOSPC",
            "error": str(error),
            **_resource_snapshot(),
        }, ensure_ascii=False, indent=2))
        return 75
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
