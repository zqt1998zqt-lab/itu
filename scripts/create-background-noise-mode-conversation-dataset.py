#!/usr/bin/env python3
"""Create one per-scene 100-combination conversation preview dataset.

The 100 cases are engineering combinations assembled from a finite licensed
source bank. Source recordings and offsets may recur across cases; no recording
is looped inside one case. This script does not claim formal ITU-T compliance.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
from scipy import signal as sp_signal
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


OLD_GENERATOR_PATH = SCRIPTS / "create-background-noise-scene-condition-dataset.py"
OLD = _load_module(OLD_GENERATOR_PATH, "itu_scene_condition_generator_20")
MEASUREMENT = OLD.LEGACY

SourceBank = OLD.SourceBank
encode = OLD.encode
file_sha = OLD.file_sha
sha = OLD.sha
stable_seed = OLD.stable_seed

from acceptance_signal_design import (  # noqa: E402
    CONVERSATION_STIMULUS_KINDS,
    render_signals,
)
from server.acoustic import acceptance_scene_conditions as legacy_contract  # noqa: E402
from server.acoustic.acceptance_visualization import (  # noqa: E402
    cache_root as visualization_cache_root,
    precompute_visualizations,
)
from server.acoustic.acceptance_scope import (  # noqa: E402
    background_noise_scene_items,
)
from server.acoustic.acceptance_scope import ACCEPTANCE_MODES, background_noise_scene_items  # noqa: E402
from server.acoustic.scene_mode_conversation_profiles import (  # noqa: E402
    CONVERSATION_MODES,
    profile_for_scene_and_mode,
    profile_mode,
    profile_sample_rate_hz,
)


SOURCE_PATHS = {
    "server/acoustic/scene_conversation_profiles.py": (
        PROJECT / "server/acoustic/scene_conversation_profiles.py"
    ),
    "server/acoustic/scene_mode_conversation_profiles.py": (
        PROJECT / "server/acoustic/scene_mode_conversation_profiles.py"
    ),
    "scripts/create-background-noise-mode-conversation-dataset.py": Path(__file__),
    "server/acoustic/acceptance_scene_conditions.py": (
        PROJECT / "server/acoustic/acceptance_scene_conditions.py"
    ),
    "server/acoustic/acceptance_scope.py": (
        PROJECT / "server/acoustic/acceptance_scope.py"
    ),
    "server/acoustic/acceptance_contract.py": (
        PROJECT / "server/acoustic/acceptance_contract.py"
    ),
    "scripts/acceptance_signal_design.py": (
        PROJECT / "scripts/acceptance_signal_design.py"
    ),
    "scripts/create-background-noise-scene-condition-dataset.py": (
        OLD_GENERATOR_PATH
    ),
    "scripts/create-background-noise-conversation-dataset.py": (
        SCRIPTS / "create-background-noise-conversation-dataset.py"
    ),
    "scripts/create-acceptance-measurement-dataset.py": (
        SCRIPTS / "create-acceptance-measurement-dataset.py"
    ),
    "cpp/evaluation/nv200-test-contract.json": (
        PROJECT / "cpp/evaluation/nv200-test-contract.json"
    ),
}
IMPORTED_SOURCE_HASHES = {
    relative: file_sha(path)
    for relative, path in SOURCE_PATHS.items()
}

RATE = 16_000
DURATION_S = 12.0
FRAMES = int(round(RATE * DURATION_S))
BASE_SEED = 20261109
CASES_PER_MODE = 20
SPEECH_MODES = (
    "near_single",
    "far_single",
    "double_talk",
    "transition",
)
NOISE_MODE = "noise_only"
ECHO_MODES = frozenset({"far_single", "double_talk", "transition"})
SER_MODES = frozenset({"double_talk", "transition"})
SNR_OFFSETS_DB = (-3.0, 0.0, 3.0)
SER_DB_LEVELS = (-10.0, -5.0, 0.0)
ECHO_DELAY_LEVELS_MS = (8.0, 17.0, 29.0, 47.0, 73.0)
MIN_NON_SILENT_SECONDS = 9.01
MIN_DOUBLE_TALK_OVERLAP_SECONDS = 9.01
NON_SILENT_FRAME_RMS = 1e-5
MEASUREMENT_DB_TOLERANCE = 0.05
PCM16_MAX_QUANTIZATION_ERROR = 1.0 / 32768.0
PRE_PCM_MIX_RESIDUAL_LIMIT = 1e-12
MIN_FREE_BYTES = 100 * 1024**3
MIN_MEM_AVAILABLE_BYTES = 5 * 1024**3
ROLES = legacy_contract.ROLES
MODE = legacy_contract.MODE
SPEECH_FAMILIES = ("g_mos", "n_mos", "s_mos")
CNG_FAMILIES = ("cng_level", "cng_spectrum")


class ResourceSafetyStop(RuntimeError):
    """Raised before further writes when the project safety floor is crossed."""


def configure_scene(scene: str, mode: str = "WB-wired") -> None:
    """Bind this process to one immutable conversation dataset profile."""
    global SCENE_PROFILE, MODE, RATE, FRAMES
    global DATASET_ID, DATASET_NAME, DATASET_KIND, SCHEMA_VERSION, SCENE
    global CONDITIONS, OUTPUT_ROOT, MANIFEST_PATH, ASSETS_PATH
    global SUMMARY_PATH, QUALITY_PATH, BINDING_PATH, SNR_BASELINE_DB

    profile = profile_for_scene_and_mode(scene, mode)
    resolved_mode = profile_mode(profile)
    if profile.case_count != len(CONVERSATION_MODES) * CASES_PER_MODE:
        raise ValueError("conversation profile must expose exactly 100 cases")
    if profile.speech_case_count != len(SPEECH_MODES) * CASES_PER_MODE:
        raise ValueError("conversation profile must expose exactly 80 speech cases")
    if profile.cng_case_count != CASES_PER_MODE:
        raise ValueError("conversation profile must expose exactly 20 CNG cases")
    if profile.observation_count != (
        len(SPEECH_MODES) * CASES_PER_MODE * 3 + CASES_PER_MODE * 2
    ):
        raise ValueError("conversation profile must expose exactly 280 observations")
    expected_contract_ids = {
        row["id"]
        for row in background_noise_scene_items(resolved_mode, scene)
    }
    if (
        len(profile.contract_item_ids) != legacy_contract.CONTRACT_ITEM_COUNT
        or set(profile.contract_item_ids)
        != expected_contract_ids
    ):
        raise ValueError(
            f"conversation profile contract rows differ from {scene}/{resolved_mode}"
        )

    SCENE_PROFILE = profile
    MODE = resolved_mode
    RATE = profile_sample_rate_hz(profile)
    FRAMES = int(round(RATE * DURATION_S))
    DATASET_ID = profile.dataset_id
    DATASET_NAME = profile.dataset_name
    DATASET_KIND = profile.dataset_kind
    SCHEMA_VERSION = profile.schema_version
    SCENE = profile.scene
    CONDITIONS = dict(profile.conditions)
    OUTPUT_ROOT = legacy_contract.dataset_root(DATASET_ID)
    MANIFEST_PATH = OUTPUT_ROOT / "manifest.json"
    ASSETS_PATH = OUTPUT_ROOT / "assets.json"
    SUMMARY_PATH = OUTPUT_ROOT / "summary.json"
    QUALITY_PATH = OUTPUT_ROOT / "quality.json"
    BINDING_PATH = OUTPUT_ROOT / "runtime-contract-binding.json"
    SNR_BASELINE_DB = float(profile.snr_db)

    OLD.configure_scene(scene)
    OLD.DATASET_ID = DATASET_ID
    OLD.DATASET_NAME = DATASET_NAME


configure_scene("sce1", "WB-wired")


def _json_bytes(value: Any) -> bytes:
    return OLD._json_bytes(value)


def _exclusive_create_or_verify(path: Path, payload: bytes) -> bool:
    return OLD._exclusive_create_or_verify(path, payload)


def _mem_available_bytes() -> int:
    return OLD._mem_available_bytes()


def _existing_ancestor(path: Path) -> Path:
    return OLD._existing_ancestor(path)


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


def _source_hash_snapshot() -> dict[str, str]:
    current = {
        relative: file_sha(path)
        for relative, path in SOURCE_PATHS.items()
    }
    if current != IMPORTED_SOURCE_HASHES:
        changed = sorted(
            relative
            for relative in current
            if current[relative] != IMPORTED_SOURCE_HASHES.get(relative)
        )
        raise ValueError(
            "source code changed after import; no dataset generated "
            f"(changed={changed})"
        )
    return current


def analysis_start_for_mode(mode: str) -> float:
    if mode == NOISE_MODE:
        return 4.0
    if mode == "transition":
        return 0.5
    return 0.0


def case_plan() -> list[dict[str, Any]]:
    """Return the deterministic 20-per-mode engineering combination plan."""
    plan: list[dict[str, Any]] = []
    for mode_index in range(1, CASES_PER_MODE + 1):
        for mode_position, mode in enumerate(CONVERSATION_MODES):
            group_id = (
                (mode_index - 1) * len(CONVERSATION_MODES)
                + mode_position
                + 1
            )
            snr_offset_db = SNR_OFFSETS_DB[
                (mode_index - 1) % len(SNR_OFFSETS_DB)
            ]
            echo_delay_ms = (
                ECHO_DELAY_LEVELS_MS[
                    (mode_index - 1) % len(ECHO_DELAY_LEVELS_MS)
                ]
                if mode in ECHO_MODES
                else None
            )
            ser_db = (
                SER_DB_LEVELS[(mode_index - 1) % len(SER_DB_LEVELS)]
                if mode in SER_MODES
                else None
            )
            plan.append({
                "group_id": group_id,
                "mode": mode,
                "mode_index": mode_index,
                "case_id": (
                    f"{SCENE}-{MODE.lower()}-{mode}-{mode_index:02d}-"
                    "conversation-v2"
                ),
                "snr_db": SNR_BASELINE_DB + snr_offset_db,
                "snr_baseline_db": SNR_BASELINE_DB,
                "snr_offset_db": snr_offset_db,
                "ser_db": ser_db,
                "echo_delay_ms": echo_delay_ms,
                "analysis_start_s": analysis_start_for_mode(mode),
                "contract_item_ids": (
                    list(SCENE_PROFILE.contract_item_ids[3:])
                    if mode == NOISE_MODE
                    else list(SCENE_PROFILE.contract_item_ids[:3])
                ),
                "families": (
                    list(CNG_FAMILIES)
                    if mode == NOISE_MODE
                    else list(SPEECH_FAMILIES)
                ),
            })
    if Counter(row["mode"] for row in plan) != {
        mode: CASES_PER_MODE for mode in CONVERSATION_MODES
    }:
        raise ValueError("case plan does not expose 20 cases per mode")
    if len({row["case_id"] for row in plan}) != len(plan):
        raise ValueError("case plan IDs are not unique")
    return plan


def _record_duration_s(record: dict[str, Any]) -> float:
    """Read the real source-manifest duration field without silent fallback."""
    if "seconds" not in record:
        raise ValueError(
            f"source record {record.get('path')!r} lacks required seconds field"
        )
    try:
        duration_s = float(record["seconds"])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"source record {record.get('path')!r} has invalid seconds"
        ) from error
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError(
            f"source record {record.get('path')!r} has non-positive seconds"
        )
    return duration_s


def _speaker_pools(
    bank: SourceBank,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Validate the full bank, then split sorted speakers into disjoint pools."""
    records = OLD._clean_speech_records(bank)
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_speaker[str(record["speaker"])].append(record)
    speakers = sorted(by_speaker)
    if len(speakers) < 2:
        raise ValueError("conversation generator needs at least two speakers")
    near_speakers = speakers[::2]
    far_speakers = speakers[1::2]
    if set(near_speakers) & set(far_speakers):
        raise ValueError("near/far speaker pools overlap")
    near_records = [
        record
        for speaker in near_speakers
        for record in by_speaker[speaker]
    ]
    far_records = [
        record
        for speaker in far_speakers
        for record in by_speaker[speaker]
    ]
    near_records.sort(key=lambda row: (str(row["speaker"]), str(row["path"])))
    far_records.sort(key=lambda row: (str(row["speaker"]), str(row["path"])))
    total_duration_s = round(
        sum(_record_duration_s(record) for record in records),
        9,
    )
    if total_duration_s <= 0.0:
        raise ValueError("source bank has no positive recorded duration")
    metadata = {
        "available_source_file_count": len(records),
        "available_speaker_count": len(speakers),
        "available_source_duration_s": total_duration_s,
        "near": {
            "speaker_count": len(near_speakers),
            "source_file_count": len(near_records),
            "source_duration_s": round(
                sum(_record_duration_s(record) for record in near_records),
                9,
            ),
            "speakers": near_speakers,
        },
        "far": {
            "speaker_count": len(far_speakers),
            "source_file_count": len(far_records),
            "source_duration_s": round(
                sum(_record_duration_s(record) for record in far_records),
                9,
            ),
            "speakers": far_speakers,
        },
        "speaker_pools_disjoint": True,
        "assignment": "sorted speaker list split by even/odd index",
    }
    return near_records, far_records, metadata


def _speech_crop(
    bank: SourceBank,
    records: list[dict[str, Any]],
    *,
    group_id: int,
    role: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    waveform, raw_info = OLD._build_speech_crop(
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
    if len(paths) != len(set(paths)):
        raise ValueError(f"{role} crop repeats a source recording within one case")
    if info.get("repeated") is not False or any(
        segment.get("repeated") is not False for segment in segments
    ):
        raise ValueError(f"{role} crop is marked as repeated")
    samples_16k = np.asarray(waveform, dtype=np.float64)
    if RATE == 8_000:
        samples = sp_signal.resample_poly(samples_16k, 1, 2)
        cutoff = 3400.0
        samples = sp_signal.sosfiltfilt(
            sp_signal.butter(6, cutoff, fs=RATE, output="sos"),
            samples,
        )
        resampling = {
            "method": "scipy.signal.resample_poly",
            "up": 1,
            "down": 2,
            "source_rate_hz": 16_000,
            "target_rate_hz": 8_000,
            "band_limit_hz": cutoff,
        }
    elif RATE == 48_000:
        samples = sp_signal.resample_poly(samples_16k, 3, 1)
        resampling = {
            "method": "scipy.signal.resample_poly",
            "up": 3,
            "down": 1,
            "source_rate_hz": 16_000,
            "target_rate_hz": 48_000,
        }
    else:
        samples = samples_16k.copy()
        resampling = {
            "method": "none",
            "source_rate_hz": 16_000,
            "target_rate_hz": RATE,
        }
    samples = np.asarray(samples, dtype=np.float64)
    if len(samples) != FRAMES:
        raise ValueError(f"{role} crop length {len(samples)} differs from {FRAMES}")
    if RATE != 16_000:
        samples -= float(np.mean(samples))
        peak = float(np.max(np.abs(samples)))
        if peak <= 1e-8:
            raise ValueError(f"{role} resampled speech crop is silent")
        samples *= OLD.NEAR_CONDITIONED_PEAK / peak
    info.update({
        "speaker_pool": role,
        "target_sample_rate_hz": RATE,
        "resampling": resampling,
        "source_reuse_allowed_across_cases": True,
        "within_case_source_repetition": False,
    })
    return samples, info


def _digital_silence_info(role: str, reason: str) -> dict[str, Any]:
    return {
        "source_kind": "digital_silence",
        "role": role,
        "reason": reason,
        "repeated": False,
        "within_case_source_repetition": False,
    }


def _noise_crop(
    bank: SourceBank,
    *,
    case_seed: int,
    group_id: int,
    used_offsets: set[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    for attempt in range(64):
        noise_seed = stable_seed(
            f"{DATASET_ID}:{group_id}:noise:{attempt}",
            case_seed,
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
    raise ValueError("could not choose a unique noise crop offset")


def _rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return 0.0
    return float(np.sqrt(np.mean(np.square(array, dtype=np.float64))))


def _rms_window(
    samples: np.ndarray,
    start_s: float,
    end_s: float,
) -> float:
    start = max(0, min(len(samples), int(round(start_s * RATE))))
    end = max(0, min(len(samples), int(round(end_s * RATE))))
    return _rms(samples[start:end])


def _non_silent_seconds(samples: np.ndarray) -> float:
    values = np.asarray(samples, dtype=np.float64)
    frame = RATE // 100
    usable = len(values) - (len(values) % frame)
    if usable <= 0:
        return 0.0
    frames = values[:usable].reshape(-1, frame)
    frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))
    return float(np.count_nonzero(frame_rms > NON_SILENT_FRAME_RMS) / 100.0)


def _joint_non_silent_seconds(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    frame = RATE // 100
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    usable = min(len(left), len(right)) // frame * frame
    left_frames = left[:usable].reshape(-1, frame)
    right_frames = right[:usable].reshape(-1, frame)
    left_rms = np.sqrt(np.mean(np.square(left_frames), axis=1))
    right_rms = np.sqrt(np.mean(np.square(right_frames), axis=1))
    active = (left_rms > NON_SILENT_FRAME_RMS) & (
        right_rms > NON_SILENT_FRAME_RMS
    )
    return float(np.count_nonzero(active) / 100.0)


def _mask_from_segments(
    segments: list[dict[str, Any]],
    frames: int,
) -> np.ndarray:
    mask = np.zeros(frames, dtype=bool)
    for segment in segments:
        start = max(0, min(frames, int(round(float(segment["start_s"]) * RATE))))
        end = max(0, min(frames, int(round(float(segment["end_s"]) * RATE))))
        mask[start:end] = True
    return mask


def _db(numerator: float, denominator: float) -> float | None:
    if numerator <= 0.0 or denominator <= 0.0:
        return None
    return float(20.0 * math.log10(numerator / denominator))


def _mix_residuals(
    delivered: dict[str, np.ndarray],
    design: dict[str, Any],
) -> list[float]:
    """Reconstruct both microphone channels using upstream component helpers."""
    if design.get("direction") != "send":
        raise ValueError("mixture verification requires send direction")
    microphone = np.asarray(delivered["microphone"], dtype=np.float64)
    reference = np.asarray(delivered["reference"], dtype=np.float64)
    far_end = np.asarray(delivered["far_end"], dtype=np.float64)
    noise = np.asarray(delivered["noise"], dtype=np.float64)
    if microphone.ndim != 2 or microphone.shape[1] != 2:
        raise ValueError("mixture verification requires two microphone channels")
    frames = len(microphone)
    if (
        reference.shape != (frames,)
        or far_end.shape != (frames,)
        or noise.shape != (frames,)
    ):
        raise ValueError("mixture component lengths differ from microphone frames")
    paths = design.get("echo_paths")
    delays = design.get("near_arrival_delays_samples")
    if not isinstance(paths, list):
        raise ValueError("echo_paths must be a list")
    if paths and len(paths) != 2:
        raise ValueError("two echo paths are required for dual microphones")
    if not isinstance(delays, list) or len(delays) != 2:
        raise ValueError("two near arrival delays are required")

    residuals: list[float] = []
    for channel in range(2):
        raw_delay = delays[channel]
        delay = int(raw_delay)
        if delay != raw_delay or delay < 0 or delay >= frames:
            raise ValueError("near arrival delay is outside the delivered timeline")
        delayed_reference = np.zeros(frames, dtype=np.float64)
        delayed_reference[delay:] = reference[:frames - delay]
        echo = MEASUREMENT._fir_component(
            far_end,
            paths[channel] if paths else None,
        )
        channel_noise = MEASUREMENT._reconstruct_noise_channel(
            noise,
            design,
            channel,
        )
        expected = delayed_reference + echo + channel_noise
        residuals.append(
            float(np.max(np.abs(microphone[:, channel] - expected)))
        )
    return residuals


def _noise_mix_l1(design: dict[str, Any], channel: int) -> float:
    if channel == 0:
        return 1.0
    recipe = design.get("noise_spatial", {}).get("channel_1_recipe")
    if not isinstance(recipe, dict):
        raise ValueError("dual-microphone noise recipe is missing")
    rho = float(recipe["rho"])
    normalizer = float(recipe["normalizer"])
    if not math.isfinite(rho) or not -1.0 < rho < 1.0:
        raise ValueError("noise channel correlation is invalid")
    if not math.isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError("noise channel normalizer is invalid")
    return abs(rho) / normalizer


def _echo_l1_and_gain(
    design: dict[str, Any],
    channel: int,
) -> tuple[float, float]:
    paths = design.get("echo_paths", [])
    if not paths:
        return 0.0, 0.0
    path = paths[channel]
    taps = path.get("tap_gains")
    gain = path.get("echo_gain_linear")
    if not isinstance(taps, list) or gain is None:
        raise ValueError("echo path is missing taps or gain")
    tap_l1 = sum(abs(float(value)) for value in taps)
    echo_gain = abs(float(gain))
    if not math.isfinite(tap_l1) or not math.isfinite(echo_gain):
        raise ValueError("echo path contains non-finite metrics")
    return tap_l1, echo_gain


def _quantized_mixture_bounds(
    design: dict[str, Any],
) -> list[dict[str, Any]]:
    """Bound decoded residuals from four independent PCM16 quantizers."""
    bounds: list[dict[str, Any]] = []
    arithmetic_slack = 64.0 * np.finfo(np.float64).eps
    for channel in range(2):
        tap_l1, echo_gain = _echo_l1_and_gain(design, channel)
        noise_l1 = _noise_mix_l1(design, channel)
        components = {
            "microphone": PCM16_MAX_QUANTIZATION_ERROR,
            "reference": PCM16_MAX_QUANTIZATION_ERROR,
            "noise": PCM16_MAX_QUANTIZATION_ERROR * noise_l1,
            "far_end_echo": (
                PCM16_MAX_QUANTIZATION_ERROR * tap_l1 * echo_gain
            ),
        }
        bound = sum(components.values()) + arithmetic_slack
        bounds.append({
            "channel": channel,
            "pcm16_max_quantization_error": (
                PCM16_MAX_QUANTIZATION_ERROR
            ),
            "noise_mix_l1": noise_l1,
            "echo_tap_l1": tap_l1,
            "echo_gain_abs": echo_gain,
            "quantization_components": components,
            "float_arithmetic_slack": arithmetic_slack,
            "theoretical_max_abs_residual": bound,
        })
    return bounds


def _verify_mixture_with_pcm_bound(
    pre_pcm: dict[str, np.ndarray],
    decoded: dict[str, np.ndarray],
    design: dict[str, Any],
) -> dict[str, Any]:
    """Validate float source identity, then decoded PCM16 mixture within theory."""
    source_mix = design.get("source_mix_check")
    if not isinstance(source_mix, dict) or source_mix.get("applicable") is not True:
        raise ValueError("pre-PCM source_mix_check is not applicable")
    reported_source_residual = source_mix.get("residual_max_abs")
    if (
        reported_source_residual is None
        or not math.isfinite(float(reported_source_residual))
        or float(reported_source_residual) > PRE_PCM_MIX_RESIDUAL_LIMIT
    ):
        raise ValueError("pre-PCM source_mix_check exceeds 1e-12")
    pre_pcm_residuals = _mix_residuals(pre_pcm, design)
    if max(pre_pcm_residuals) > PRE_PCM_MIX_RESIDUAL_LIMIT:
        raise ValueError(
            "pre-PCM dual-microphone reconstruction exceeds 1e-12: "
            f"{pre_pcm_residuals}"
        )

    pcm_residuals = _mix_residuals(decoded, design)
    bounds = _quantized_mixture_bounds(design)
    for channel, (residual, bound) in enumerate(
        zip(pcm_residuals, bounds, strict=True)
    ):
        theoretical = float(bound["theoretical_max_abs_residual"])
        if residual > theoretical:
            raise ValueError(
                "delivered PCM microphone exceeds theoretical quantization bound "
                f"(channel {channel}, residual {residual:.12g}, "
                f"bound {theoretical:.12g})"
            )
    return {
        "applicable": True,
        "source_mix_residual_max_abs": float(reported_source_residual),
        "pre_pcm_residuals_max_abs": pre_pcm_residuals,
        "pre_pcm_limit": PRE_PCM_MIX_RESIDUAL_LIMIT,
        "pcm_component_residuals_max_abs": pcm_residuals,
        "theoretical_bounds": bounds,
        "method": (
            "upstream FIR/noise reconstruction; independent one-LSB PCM16 "
            "errors propagated through echo FIR L1 and noise-mix L1"
        ),
    }


def _post_pcm_measurements(
    mode: str,
    decoded: dict[str, np.ndarray],
    design: dict[str, Any],
) -> dict[str, Any]:
    masks = design["activity_masks"]
    near_mask = _mask_from_segments(masks["near"], FRAMES)
    measured_snr = None
    if design["snr_db_target_applicable"]:
        measured_snr = _db(
            _rms(decoded["reference"][near_mask]),
            _rms(decoded["noise"][near_mask]),
        )
        if measured_snr is None or abs(
            measured_snr - float(design["snr_db_target"])
        ) > MEASUREMENT_DB_TOLERANCE:
            raise ValueError(f"{mode} post-PCM SNR differs from its target")

    measured_ser = None
    if design["ser_db_target_applicable"]:
        ser_scope = "near" if mode == "double_talk" else "far_only"
        echo_mask = _mask_from_segments(masks[ser_scope], FRAMES)
        echo_0 = (
            decoded["microphone"][:, 0]
            - decoded["reference"]
            - decoded["noise"]
        )
        measured_ser = _db(
            _rms(echo_0[echo_mask]),
            _rms(decoded["reference"][near_mask]),
        )
        if measured_ser is None or abs(
            measured_ser - float(design["ser_db_effective_target"])
        ) > MEASUREMENT_DB_TOLERANCE:
            raise ValueError(f"{mode} post-PCM SER differs from its target")
    return {
        "snr_db_measured_post_pcm": measured_snr,
        "ser_db_measured_post_pcm": measured_ser,
        "tolerance_db": MEASUREMENT_DB_TOLERANCE,
    }


def _assert_pcm_activity(
    mode: str,
    decoded: dict[str, np.ndarray],
    design: dict[str, Any],
) -> dict[str, Any]:
    reference = decoded["reference"]
    far_end = decoded["far_end"]
    noise = decoded["noise"]
    echo_count = len(design["echo_paths"])
    metrics: dict[str, Any] = {
        "reference_non_silent_seconds": _non_silent_seconds(reference),
        "far_end_non_silent_seconds": _non_silent_seconds(far_end),
        "reference_nonzero_samples": int(np.count_nonzero(reference)),
        "far_end_nonzero_samples": int(np.count_nonzero(far_end)),
        "echo_path_count": echo_count,
    }

    if mode == "near_single":
        if np.count_nonzero(far_end) != 0 or echo_count != 0:
            raise ValueError("near_single must contain no far-end speech or echo")
        if metrics["reference_non_silent_seconds"] < MIN_NON_SILENT_SECONDS:
            raise ValueError("near_single has less than 9.01s active near speech")
    elif mode == "far_single":
        if np.count_nonzero(reference) != 0:
            raise ValueError("far_single reference must be strict digital silence")
        if echo_count != 2:
            raise ValueError("far_single requires two echo paths")
        if metrics["far_end_non_silent_seconds"] < MIN_NON_SILENT_SECONDS:
            raise ValueError("far_single has less than 9.01s active far speech")
    elif mode == "double_talk":
        overlap_s = _joint_non_silent_seconds(reference, far_end)
        metrics["double_talk_overlap_seconds"] = overlap_s
        if echo_count != 2:
            raise ValueError("double_talk requires two echo paths")
        if metrics["reference_non_silent_seconds"] < MIN_NON_SILENT_SECONDS:
            raise ValueError("double_talk has less than 9.01s active near speech")
        if metrics["far_end_non_silent_seconds"] < MIN_NON_SILENT_SECONDS:
            raise ValueError("double_talk has less than 9.01s active far speech")
        if overlap_s < MIN_DOUBLE_TALK_OVERLAP_SECONDS:
            raise ValueError("double_talk actual overlap is less than 9.01s")
    elif mode == "transition":
        windows = {
            "far_only_reference_rms": _rms_window(reference, 0.7, 1.8),
            "far_only_far_rms": _rms_window(far_end, 0.7, 1.8),
            "near_only_reference_rms": _rms_window(reference, 2.1, 4.9),
            "near_only_far_rms": _rms_window(far_end, 2.1, 4.9),
            "double_talk_reference_rms": _rms_window(reference, 8.1, 10.9),
            "double_talk_far_rms": _rms_window(far_end, 8.1, 10.9),
            "noise_only_reference_rms": _rms_window(reference, 11.1, 11.9),
            "noise_only_far_rms": _rms_window(far_end, 11.1, 11.9),
            "noise_only_noise_rms": _rms_window(noise, 11.1, 11.9),
        }
        metrics["transition_windows"] = windows
        if echo_count != 2:
            raise ValueError("transition requires two echo paths")
        if not (
            windows["far_only_reference_rms"] == 0.0
            and windows["far_only_far_rms"] > NON_SILENT_FRAME_RMS
            and windows["near_only_reference_rms"] > NON_SILENT_FRAME_RMS
            and windows["near_only_far_rms"] == 0.0
            and windows["double_talk_reference_rms"] > NON_SILENT_FRAME_RMS
            and windows["double_talk_far_rms"] > NON_SILENT_FRAME_RMS
            and windows["noise_only_reference_rms"] == 0.0
            and windows["noise_only_far_rms"] == 0.0
            and windows["noise_only_noise_rms"] > 0.0
        ):
            raise ValueError("transition does not contain all required activities")
    elif mode == "noise_only":
        start = int(round(4.0 * RATE))
        windows = {
            "preheat_0_2_reference_rms": _rms_window(reference, 0.0, 2.0),
            "preheat_0_2_far_rms": _rms_window(far_end, 0.0, 2.0),
            "settling_2_4_reference_rms": _rms_window(reference, 2.0, 4.0),
            "settling_2_4_far_rms": _rms_window(far_end, 2.0, 4.0),
            "settling_2_4_noise_rms": _rms_window(noise, 2.0, 4.0),
            "observation_4_12_reference_rms": _rms_window(reference, 4.0, 12.0),
            "observation_4_12_far_rms": _rms_window(far_end, 4.0, 12.0),
            "observation_4_12_noise_rms": _rms_window(noise, 4.0, 12.0),
        }
        metrics["noise_only_windows"] = windows
        if echo_count != 0:
            raise ValueError("noise_only must not contain an echo path")
        if windows["preheat_0_2_reference_rms"] <= NON_SILENT_FRAME_RMS:
            raise ValueError("noise_only is missing 0-2s near preheat")
        if (
            np.count_nonzero(reference[start:]) != 0
            or np.count_nonzero(far_end[start:]) != 0
        ):
            raise ValueError("noise_only 4-12s observation is not double-silent")
        if windows["observation_4_12_noise_rms"] <= 0.0:
            raise ValueError("noise_only 4-12s observation has no noise")
    else:
        raise ValueError(f"unsupported conversation mode: {mode}")
    return metrics


def _source_reuse_policy(
    source_inventory: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset_definition": "100_engineering_combinations",
        "independent_source_recordings_claimed": False,
        "source_audio_may_be_reused_across_cases": True,
        "within_case_source_looping": False,
        "noise_crop_may_be_reused_across_cases": False,
        "speaker_pools_disjoint_for_near_and_far": True,
        "source_inventory": deepcopy(source_inventory),
    }


def _build_cases(
    bank: SourceBank,
) -> tuple[list[dict[str, Any]], dict[str, bytes], dict[str, Any]]:
    near_records, far_records, speaker_inventory = _speaker_pools(bank)
    source_inventory = {
        **speaker_inventory,
        "noise_scene": SCENE,
        "noise_mode": MODE,
    }
    reuse_policy = _source_reuse_policy(source_inventory)
    cases: list[dict[str, Any]] = []
    blobs: dict[str, bytes] = {}
    quality_cases: list[dict[str, Any]] = []
    near_crop_hashes: set[str] = set()
    far_crop_hashes: set[str] = set()
    used_noise_offsets: set[int] = set()
    used_case_seeds: set[int] = set()
    used_case_ids: set[str] = set()
    source_files_used: set[str] = set()
    source_segment_hashes: set[str] = set()

    for plan in case_plan():
        mode = str(plan["mode"])
        group_id = int(plan["group_id"])
        case_id = str(plan["case_id"])
        case_seed = stable_seed(
            f"{DATASET_ID}:{group_id}:{mode}:conversation-case",
            BASE_SEED,
        )
        if case_seed in used_case_seeds or case_id in used_case_ids:
            raise ValueError("conversation case seed or ID is not unique")
        used_case_seeds.add(case_seed)
        used_case_ids.add(case_id)

        near_input = np.zeros(FRAMES, dtype=np.float64)
        far_input = np.zeros(FRAMES, dtype=np.float64)
        near_info = _digital_silence_info(
            "near",
            "far_single contains no near-end speech",
        )
        far_info = _digital_silence_info(
            "far_input",
            "near_single and noise_only contain no far-end speech",
        )
        if mode != "far_single":
            near_input, near_info = _speech_crop(
                bank,
                near_records,
                group_id=group_id,
                role="near",
            )
            crop_hash = str(near_info["crop_recipe_sha256"])
            if crop_hash in near_crop_hashes:
                raise ValueError("near crop recipe is not unique across cases")
            near_crop_hashes.add(crop_hash)
        if mode in {"far_single", "double_talk", "transition"}:
            far_input, far_info = _speech_crop(
                bank,
                far_records,
                group_id=group_id,
                role="far_input",
            )
            crop_hash = str(far_info["crop_recipe_sha256"])
            if crop_hash in far_crop_hashes:
                raise ValueError("far crop recipe is not unique across cases")
            far_crop_hashes.add(crop_hash)

        for crop_info in (near_info, far_info):
            for segment in crop_info.get("segments", []):
                source_files_used.add(str(segment["path"]))
                source_segment_hashes.add(
                    str(segment["selected_audio_sha256_f32le"])
                )

        noise, noise_info = _noise_crop(
            bank,
            case_seed=case_seed,
            group_id=group_id,
            used_offsets=used_noise_offsets,
        )
        signals, design = render_signals(
            near_input,
            far_input,
            noise,
            RATE,
            direction=str(CONDITIONS["direction"]),
            kind=mode,
            snr_db=float(plan["snr_db"]),
            ser_db=float(plan["ser_db"] if plan["ser_db"] is not None else -5.0),
            echo_delay_ms=float(
                plan["echo_delay_ms"]
                if plan["echo_delay_ms"] is not None
                else 30.0
            ),
            seed=case_seed,
            input_level_db=float(CONDITIONS["input_level_db"]),
            volume_gain_db=0.0,
            bandwidth=MODE.split("-", 1)[0],
        )
        expected_metadata_kind = "cng" if mode == NOISE_MODE else "speech"
        if (
            design.get("kind") != expected_metadata_kind
            or design.get("stimulus_kind") != mode
        ):
            raise ValueError(f"{case_id} render metadata kind mapping is invalid")
        if mode in ECHO_MODES and len(design.get("echo_paths", [])) != 2:
            raise ValueError(f"{case_id} does not expose two echo paths")
        if mode not in ECHO_MODES and design.get("echo_paths"):
            raise ValueError(f"{case_id} unexpectedly exposes echo paths")

        decoded: dict[str, np.ndarray] = {}
        signal_paths: dict[str, str] = {}
        signal_hashes: dict[str, str] = {}
        for role in ROLES:
            blob, delivered = encode(signals[role], RATE)
            from io import BytesIO
            wav_info = sf.info(BytesIO(blob))
            expected_ch = 2 if role == "microphone" else 1
            if (
                wav_info.format != "WAV"
                or wav_info.subtype != "PCM_16"
                or wav_info.samplerate != RATE
                or wav_info.frames != FRAMES
                or wav_info.channels != expected_ch
            ):
                raise ValueError(
                    f"{case_id} {role} WAV identity mismatch: "
                    f"{wav_info.format}/{wav_info.subtype}/{wav_info.samplerate}/"
                    f"{wav_info.frames}/{wav_info.channels}"
                )
            digest = sha(blob)
            relative = f"assets/audio/{digest}.wav"
            existing = blobs.get(relative)
            if existing is not None and existing != blob:
                raise ValueError(f"SHA collision for {relative}")
            blobs[relative] = blob
            decoded[role] = np.asarray(delivered)
            signal_paths[role] = relative
            signal_hashes[role] = digest

        if np.asarray(decoded["microphone"]).shape != (FRAMES, 2):
            raise ValueError(f"{case_id} microphone is not stereo PCM16")
        for role in ("reference", "far_end", "noise"):
            if np.asarray(decoded[role]).shape != (FRAMES,):
                raise ValueError(f"{case_id} {role} is not mono PCM16")

        mixture_check = _verify_mixture_with_pcm_bound(
            signals,
            decoded,
            design,
        )
        activity_check = _assert_pcm_activity(mode, decoded, design)
        measurement_check = _post_pcm_measurements(mode, decoded, design)
        peak = max(
            float(np.max(np.abs(values)))
            for values in decoded.values()
        )
        if peak >= 0.7:
            raise ValueError(f"{case_id} decoded peak is {peak}")

        provenance = {
            "group_id": group_id,
            "group_key": f"{SCENE}-{MODE.lower()}-conversation-{group_id:03d}",
            "seed": case_seed,
            "base_seed": BASE_SEED,
            "scene": SCENE,
            "conversation_mode": mode,
            "stimulus_kind": mode,
            "metadata_kind": expected_metadata_kind,
            "engineering_preview": True,
            "engineering_combination": True,
            "independent_source_recordings": False,
            "source_reuse_policy": deepcopy(reuse_policy),
            "snr_db": float(plan["snr_db"]),
            "snr_baseline_db": float(plan["snr_baseline_db"]),
            "snr_offset_db": float(plan["snr_offset_db"]),
            "snr_db_target_applicable": design["snr_db_target_applicable"],
            "snr_db_target": design["snr_db_target"],
            "ser_db": plan["ser_db"],
            "ser_db_target_applicable": design["ser_db_target_applicable"],
            "ser_db_target": design["ser_db_target"],
            "echo_delay_ms": plan["echo_delay_ms"],
            "analysis_start_s": float(plan["analysis_start_s"]),
            "near_speech": near_info,
            "far_input": far_info,
            "noise": noise_info,
            "far_end_delivery": {
                "digital_silence": mode in {"near_single", NOISE_MODE},
                "nonzero_sample_count": int(np.count_nonzero(decoded["far_end"])),
                "reason": (
                    "near_single/noise_only carry no far-end excitation"
                    if mode in {"near_single", NOISE_MODE}
                    else "far-end speech is present and drives both echo paths"
                ),
            },
            "render_metadata": design,
            "actual_vehicle_recording": False,
            "within_case_source_looping": False,
        }
        case = {
            "case_id": case_id,
            "conditions": dict(CONDITIONS),
            "families": list(plan["families"]),
            "contract_item_ids": list(plan["contract_item_ids"]),
            "signals": signal_paths,
            "signals_sha256": signal_hashes,
            "provenance": provenance,
        }
        if set(case) != set(legacy_contract._CASE_FIELDS):
            raise ValueError(f"{case_id} case fields differ from the legacy contract")
        cases.append(case)
        quality_cases.append({
            "case_id": case_id,
            "group_id": group_id,
            "conversation_mode": mode,
            "seed": case_seed,
            "snr_db": float(plan["snr_db"]),
            "snr_offset_db": float(plan["snr_offset_db"]),
            "ser_db": plan["ser_db"],
            "echo_delay_ms": plan["echo_delay_ms"],
            "analysis_start_s": float(plan["analysis_start_s"]),
            "noise_crop_offset_samples": int(noise_info["crop_offset_samples"]),
            "near_crop_recipe_sha256": near_info.get("crop_recipe_sha256"),
            "far_crop_recipe_sha256": far_info.get("crop_recipe_sha256"),
            "peak": peak,
            "mixture_check": mixture_check,
            "activity_check": activity_check,
            "measurement_check": measurement_check,
            "signals_sha256": signal_hashes,
        })
        _assert_resource_floor()

    if len(cases) != SCENE_PROFILE.case_count:
        raise ValueError("conversation case count differs from its profile")
    mode_counts = Counter(
        case["provenance"]["conversation_mode"] for case in cases
    )
    if mode_counts != {
        mode: CASES_PER_MODE for mode in CONVERSATION_MODES
    }:
        raise ValueError("conversation mode counts differ from 20 each")
    metadata_kind_counts = Counter(
        case["provenance"]["metadata_kind"] for case in cases
    )
    if metadata_kind_counts != {
        "speech": SCENE_PROFILE.speech_case_count,
        "cng": SCENE_PROFILE.cng_case_count,
    }:
        raise ValueError("conversation speech/CNG split differs from 80/20")

    referenced_assets = {
        case["signals"][role]
        for case in cases
        for role in ROLES
    }
    if set(blobs) != referenced_assets:
        raise ValueError("generated WAV blobs differ from case references")
    if len(used_noise_offsets) != len(cases):
        raise ValueError("noise crop offsets are not unique across cases")

    return cases, blobs, {
        "quality_cases": quality_cases,
        "source_inventory": source_inventory,
        "source_reuse_policy": reuse_policy,
        "source_files_used_count": len(source_files_used),
        "source_segment_sha256_count": len(source_segment_hashes),
        "near_crop_recipe_count": len(near_crop_hashes),
        "far_crop_recipe_count": len(far_crop_hashes),
        "noise_crop_offset_count": len(used_noise_offsets),
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
        _assert_resource_floor()
    return assets


def _visualization_status(
    assets: dict[str, dict[str, Any]],
) -> tuple[int, int]:
    root = visualization_cache_root()
    source_hashes = {
        str(metadata["sha256"])
        for metadata in assets.values()
    }
    before = {
        digest: (root / f"{digest}.json").exists()
        for digest in source_hashes
    }
    processed = precompute_visualizations(OUTPUT_ROOT, assets)
    after = {
        digest: (root / f"{digest}.json").exists()
        for digest in source_hashes
    }
    created_for_dataset = sum(
        1
        for digest in source_hashes
        if not before[digest] and after[digest]
    )
    return processed, created_for_dataset


def _build_manifest(cases: list[dict[str, Any]]) -> dict[str, Any]:
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "dataset_name": DATASET_NAME,
        "dataset_kind": DATASET_KIND,
        "test_scope": legacy_contract.TEST_SCOPE,
        "mode": MODE,
        "scene": SCENE,
        "contract_item_count": SCENE_PROFILE.contract_item_count,
        "preview_condition_group_count": SCENE_PROFILE.case_count,
        "standard_compliance": False,
        "formal_test_status": legacy_contract.FORMAL_TEST_STATUS,
        "cases": cases,
    }
    if set(manifest) != set(legacy_contract._MANIFEST_FIELDS):
        raise ValueError("manifest fields differ from the legacy top-level contract")
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
        raise ValueError("runtime binding artifact set is incomplete")
    required_sources = {
        "server/acoustic/scene_conversation_profiles.py",
        "scripts/acceptance_signal_design.py",
        "scripts/create-background-noise-conversation-dataset.py",
        "scripts/create-background-noise-scene-condition-dataset.py",
        "server/acoustic/acceptance_scope.py",
        "cpp/evaluation/nv200-test-contract.json",
    }
    if not required_sources.issubset(source_sha256):
        raise ValueError("runtime binding source set is incomplete")
    return {
        "schema_version": "scene-condition-runtime-binding-v2",
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
            "refusing to overwrite partial conversation artifacts: "
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
            "existing conversation target is incomplete: "
            + ", ".join(str(path) for path in missing)
        )
    binding = json.loads(BINDING_PATH.read_text(encoding="utf-8"))
    current_sources = _source_hash_snapshot()
    if (
        not isinstance(binding, dict)
        or binding.get("schema_version") != "scene-condition-runtime-binding-v2"
        or binding.get("dataset_id") != DATASET_ID
        or binding.get("status") != "reconciled"
        or binding.get("source_sha256") != current_sources
    ):
        raise ValueError("existing conversation runtime binding is stale")
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
        raise ValueError("existing conversation binding has invalid artifacts")
    for name, expected in artifact_sha256.items():
        if file_sha(OUTPUT_ROOT / name) != expected:
            raise ValueError(f"existing conversation artifact changed: {name}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        set(manifest) != set(legacy_contract._MANIFEST_FIELDS)
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("dataset_id") != DATASET_ID
        or manifest.get("dataset_kind") != DATASET_KIND
        or manifest.get("preview_condition_group_count")
        != SCENE_PROFILE.case_count
        or not isinstance(manifest.get("cases"), list)
        or len(manifest["cases"]) != SCENE_PROFILE.case_count
    ):
        raise ValueError("existing conversation manifest identity is invalid")
    return {
        "status": "SKIPPED",
        "reason": "existing conversation dataset and runtime binding verified",
        "dataset_id": DATASET_ID,
        "manifest": str(MANIFEST_PATH),
    }


def _verify_store() -> dict[str, Any]:
    from server.acoustic.acceptance_scene_condition_store import (
        load_scene_condition_dataset,
    )

    loaded = load_scene_condition_dataset(DATASET_ID)
    cases = loaded.get("cases")
    referenced_assets = loaded.get("referenced_assets")
    if (
        loaded.get("dataset_id") != DATASET_ID
        or not isinstance(cases, list)
        or len(cases) != SCENE_PROFILE.case_count
        or not isinstance(referenced_assets, set)
    ):
        raise ValueError("conversation store verification returned invalid identity")
    return {
        "store_verified": True,
        "store_case_count": len(cases),
        "store_referenced_asset_count": len(referenced_assets),
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
        raise ValueError("conversation output root is not a regular directory")
    (OUTPUT_ROOT / "assets" / "audio").mkdir(parents=True, exist_ok=True)

    bank = SourceBank()
    cases, blobs, counts = _build_cases(bank)
    _assert_source_stable = _source_hash_snapshot()
    if _assert_source_stable != source_sha256:
        raise ValueError("source code changed while building conversation cases")
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
    visualizations_processed, visualizations_created = _visualization_status(assets)
    if visualizations_processed != len(assets):
        raise ValueError("visualization precompute count differs from unique assets")

    completed_at = datetime.now(timezone.utc).isoformat()
    assets_blob = _json_bytes(assets)
    manifest_blob = _json_bytes(manifest)
    mode_counts = Counter(
        case["provenance"]["conversation_mode"] for case in cases
    )
    metadata_kind_counts = Counter(
        case["provenance"]["metadata_kind"] for case in cases
    )
    observation_counts = {
        mode: (
            len(SCENE_PROFILE.contract_item_ids[3:])
            if mode == NOISE_MODE
            else len(SCENE_PROFILE.contract_item_ids[:3])
        )
        for mode in CONVERSATION_MODES
    }
    quality = {
        "schema_version": "scene-condition-conversation-quality-v2",
        "dataset_id": DATASET_ID,
        "dataset_name": DATASET_NAME,
        "dataset_kind": DATASET_KIND,
        "scene": SCENE,
        "mode": MODE,
        "snr_baseline_db": SNR_BASELINE_DB,
        "generated_at": completed_at,
        "generation_started_at": started_at,
        "generation_elapsed_seconds": round(time.monotonic() - started, 6),
        "engineering_preview": True,
        "standard_compliance": False,
        "formal_itu_t_compliance_status": "not_measured",
        "case_count": len(cases),
        "speech_case_count": metadata_kind_counts["speech"],
        "cng_case_count": metadata_kind_counts["cng"],
        "conversation_mode_counts": dict(mode_counts),
        "metadata_kind_counts": dict(metadata_kind_counts),
        "observation_counts_by_mode": observation_counts,
        "observation_count": sum(
            mode_counts[mode] * count
            for mode, count in observation_counts.items()
        ),
        "signal_roles": list(ROLES),
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
        "variation": {
            "snr_baseline_db": SNR_BASELINE_DB,
            "snr_offsets_db": list(SNR_OFFSETS_DB),
            "snr_levels_db": sorted({
                float(case["provenance"]["snr_db"]) for case in cases
            }),
            "ser_levels_db": list(SER_DB_LEVELS),
            "echo_delay_levels_ms": list(ECHO_DELAY_LEVELS_MS),
            "noise_crop_offset_count": counts["noise_crop_offset_count"],
        },
        "source_reuse_policy": counts["source_reuse_policy"],
        "source_inventory": counts["source_inventory"],
        "source_files_used_count": counts["source_files_used_count"],
        "source_segment_sha256_count": counts["source_segment_sha256_count"],
        "near_crop_recipe_count": counts["near_crop_recipe_count"],
        "far_crop_recipe_count": counts["far_crop_recipe_count"],
        "unique_wav_count": len(assets),
        "all_cases_passed_activity_checks": True,
        "all_cases_passed_echo_mixture_checks": True,
        "minimum_non_silent_speech_seconds": MIN_NON_SILENT_SECONDS,
        "minimum_double_talk_overlap_seconds": (
            MIN_DOUBLE_TALK_OVERLAP_SECONDS
        ),
        "noise_only_windows_s": {
            "preheat": [0.0, 2.0],
            "settling": [2.0, 4.0],
            "observation": [4.0, 12.0],
        },
        "visualizations_processed": visualizations_processed,
        "visualizations_created_for_dataset": visualizations_created,
        "resource_preflight": initial_resources,
        "source_sha256": source_sha256,
        "cases": counts["quality_cases"],
        "limitations": [
            "100 engineering combinations, not 100 independent source recordings",
            "source recordings and offsets may recur across cases",
            "no source recording is looped inside one case",
            "controlled hybrid noise proxy with synthetic broadband component",
            "does not establish ITU-T, customer, or product compliance",
            "formal ITU-T compliance was not measured",
        ],
    }
    quality_blob = _json_bytes(quality)
    quality_sha = sha(quality_blob)
    summary = {
        "schema_version": "scene-condition-conversation-summary-v2",
        "dataset_id": DATASET_ID,
        "dataset_name": DATASET_NAME,
        "dataset_kind": DATASET_KIND,
        "scene": SCENE,
        "mode": MODE,
        "generated_at": completed_at,
        "generation_started_at": started_at,
        "engineering_preview": True,
        "standard_compliance": False,
        "formal_itu_t_compliance_status": "not_measured",
        "case_count": len(cases),
        "speech_case_count": quality["speech_case_count"],
        "cng_case_count": quality["cng_case_count"],
        "conversation_mode_counts": dict(mode_counts),
        "observation_count": quality["observation_count"],
        "contract_item_count": SCENE_PROFILE.contract_item_count,
        "signal_roles": list(ROLES),
        "duration_s": DURATION_S,
        "sample_rate_hz": RATE,
        "subtype": "PCM_16",
        "unique_wav_count": len(assets),
        "source_reuse_policy": counts["source_reuse_policy"],
        "source_inventory": counts["source_inventory"],
        "visualizations_processed": visualizations_processed,
        "visualizations_created_for_dataset": visualizations_created,
        "manifest_sha256": sha(manifest_blob),
        "assets_sha256": sha(assets_blob),
        "quality_sha256": quality_sha,
        "source_sha256": source_sha256,
        "scope": (
            f"100 deterministic engineering combinations for the {SCENE} "
            "conversation preview; not formal certification evidence"
        ),
    }
    summary_blob = _json_bytes(summary)
    runtime_binding = _runtime_binding(
        artifact_sha256={
            "manifest.json": sha(manifest_blob),
            "assets.json": sha(assets_blob),
            "summary.json": sha(summary_blob),
            "quality.json": sha(quality_blob),
        },
        source_sha256=source_sha256,
    )

    if _source_hash_snapshot() != source_sha256:
        raise ValueError("source code changed during conversation generation")

    _exclusive_create_or_verify(ASSETS_PATH, assets_blob)
    _exclusive_create_or_verify(QUALITY_PATH, quality_blob)
    _exclusive_create_or_verify(SUMMARY_PATH, summary_blob)
    _exclusive_create_or_verify(BINDING_PATH, _json_bytes(runtime_binding))
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
        "speech_case_count": quality["speech_case_count"],
        "cng_case_count": quality["cng_case_count"],
        "conversation_mode_counts": dict(mode_counts),
        "observation_count": quality["observation_count"],
        "signal_roles": list(ROLES),
        "unique_wavs": len(assets),
        "visualizations_processed": visualizations_processed,
        "visualizations_created_for_dataset": visualizations_created,
        "standard_compliance": False,
        "formal_itu_t_compliance_status": "not_measured",
        "manifest_sha256": summary["manifest_sha256"],
        "assets_sha256": summary["assets_sha256"],
        "quality_sha256": summary["quality_sha256"],
        "resource_preflight": initial_resources,
        **store_result,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create or verify one per-scene 100-combination background-noise "
            "conversation engineering-preview dataset."
        )
    )
    parser.add_argument(
        "--scene",
        choices=("sce1", "sce2", "sce3", "sce4"),
        default="sce1",
    )
    parser.add_argument(
        "--mode",
        choices=tuple(sorted(ACCEPTANCE_MODES)),
        default="WB-wired",
    )
    parser.add_argument(
        "--defer-store-verification",
        action="store_true",
        help=(
            "Skip final store loading while the v2 profile/store integration "
            "is still being finalized. Final generation must omit this flag."
        ),
    )
    args = parser.parse_args(argv)
    configure_scene(args.scene, args.mode)
    try:
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
