#!/usr/bin/env python3
"""Create 100-workload echo or double-talk datasets with true 8/16/48 kHz rates and overlap-calibrated SER."""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
from scipy import signal

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


BASE_GEN = _load_module(
    SCRIPTS / "create-acoustic-stress-datasets.py",
    "itu_base_acoustic_stress_generator",
)
STRESS = BASE_GEN.STRESS
CONVERSATION = BASE_GEN.CONVERSATION

from server.acoustic import service  # noqa: E402
from server.acoustic.acceptance_scope import ACCEPTANCE_MODES  # noqa: E402
from server.acoustic.acoustic_stress_profiles import stress_profile  # noqa: E402

RATE_NB = 8_000
RATE_16K = 16_000
RATE_SWB = 48_000


def _mode_rate(bandwidth: str) -> int:
    if bandwidth == "NB":
        return RATE_NB
    if bandwidth == "SWB":
        return RATE_SWB
    return RATE_16K


def _patched_render_stress_signals(
    near: np.ndarray,
    far: np.ndarray,
    noise_unit: np.ndarray,
    rate: int,
    *,
    scope: str,
    category: str,
    volume_gain_db: float,
    seed: int,
    case_index: int,
    bandwidth: str,
    noise_snr_db: float | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Render one stress case with true 8/16/48 kHz support and overlap-window SER calibration."""
    if scope not in STRESS.SCOPE_CATEGORIES or category not in STRESS.SCOPE_CATEGORIES[scope]:
        raise ValueError("unknown acoustic stress scope/category")
    if isinstance(case_index, bool) or not isinstance(case_index, int) or case_index < 1:
        raise ValueError("case_index must be a positive integer")
    near_input = STRESS._array_1d("near", near)
    far_input = STRESS._array_1d("far", far)
    noise_input = STRESS._array_1d("noise_unit", noise_unit)
    if len(near_input) != len(far_input) or len(near_input) != len(noise_input):
        raise ValueError("near, far, and noise_unit lengths differ")
    if not math.isfinite(float(volume_gain_db)):
        raise ValueError("volume_gain_db must be finite")
    near_limited, near_filter = STRESS.band_limit(near_input, rate, bandwidth)
    far_limited, far_filter = STRESS.band_limit(far_input, rate, bandwidth)
    noise_limited, noise_filter = STRESS.band_limit(noise_input, rate, bandwidth)
    duration_s = len(near_input) / rate
    segments = STRESS.activity_segments(scope, category, duration_s)
    near_mask = STRESS._mask_from_segments(
        segments,
        len(near_input),
        rate,
        frozenset({"near_only", "double_talk"}),
    )
    far_mask = STRESS._mask_from_segments(
        segments,
        len(near_input),
        rate,
        frozenset({"far_only", "double_talk"}),
    )
    overlap_mask = STRESS._mask_from_segments(
        segments,
        len(near_input),
        rate,
        frozenset({"double_talk"}),
    )

    base_kind = "far_single" if scope == "echo" else (
        "duplex" if category == "transition" else "double_talk"
    )
    requested_snr = STRESS._noise_snr_for(scope, category, case_index, noise_snr_db)
    base_signals, base_metadata = STRESS.SIGNAL_DESIGN.render_signals(
        near_limited,
        far_limited,
        noise_limited,
        rate,
        direction="send",
        kind=base_kind,
        snr_db=requested_snr,
        ser_db=STRESS.DOUBLE_TALK_SER_DB.get(category, 0.0),
        echo_delay_ms=20.0,
        seed=seed,
        input_level_db=0.0,
        volume_gain_db=float(volume_gain_db),
        bandwidth=bandwidth,
    )
    reference = np.asarray(base_signals["reference"], dtype=np.float64) * near_mask
    far_end = np.asarray(base_signals["far_end"], dtype=np.float64) * far_mask
    noise_0 = np.asarray(base_signals["noise"], dtype=np.float64).copy()
    noise_1 = STRESS.MEASUREMENT._reconstruct_noise_channel(
        noise_0,
        base_metadata,
        1,
    )

    target_mask = near_mask if np.any(near_mask) else far_mask
    target_signal = reference if np.any(near_mask) else far_end
    target_rms = STRESS._rms(target_signal, target_mask)
    current_noise_rms = STRESS._rms(noise_0, target_mask)
    if target_rms <= 1e-12 or current_noise_rms <= 1e-15:
        raise ValueError("stress noise SNR has no observable target/noise")
    target_noise_rms = target_rms / (10.0 ** (requested_snr / 20.0))
    noise_scale_factor = target_noise_rms / current_noise_rms
    noise_0 *= noise_scale_factor
    noise_1 *= noise_scale_factor

    raw_echo_0, raw_echo_1, echo_segments = STRESS._dynamic_echo(
        far_end,
        rate,
        scope=scope,
        category=category,
        duration_s=duration_s,
    )
    if scope == "double_talk":
        ser_mask = overlap_mask if np.any(overlap_mask) else near_mask
        near_rms = STRESS._rms(reference, ser_mask)
        raw_echo_rms = STRESS._rms(raw_echo_0, ser_mask)
        if near_rms <= 1e-12 or raw_echo_rms <= 1e-15:
            raise ValueError("double-talk SER has no observable near/echo")
        ser_nominal = STRESS.DOUBLE_TALK_SER_DB[category]
        ser_effective = ser_nominal + float(volume_gain_db)
        echo_target_rms = near_rms * (10.0 ** (ser_effective / 20.0))
        echo_gain = echo_target_rms / raw_echo_rms
    else:
        ser_mask = near_mask
        ser_nominal = None
        ser_effective = None
        echo_gain = STRESS.ECHO_GAIN_LEVELS[(case_index - 1) % len(STRESS.ECHO_GAIN_LEVELS)]
    echo_0 = raw_echo_0 * echo_gain
    echo_1 = raw_echo_1 * echo_gain

    delays = (0, 1)
    delayed_reference_0 = reference
    delayed_reference_1 = np.concatenate((np.zeros(1), reference[:-1]))
    microphone = np.column_stack((
        delayed_reference_0 + echo_0 + noise_0,
        delayed_reference_1 + echo_1 + noise_1,
    ))
    current_peak = max(
        float(np.max(np.abs(values)))
        for values in (reference, far_end, noise_0, noise_1, microphone)
    )
    common_gain = 1.0
    if current_peak > STRESS.PEAK_TARGET:
        common_gain = STRESS.PEAK_TARGET / current_peak
        reference *= common_gain
        far_end *= common_gain
        noise_0 *= common_gain
        noise_1 *= common_gain
        echo_0 *= common_gain
        echo_1 *= common_gain
        microphone *= common_gain

    recipe = dict(base_metadata["noise_spatial"]["channel_1_recipe"])
    recipe["noise_channel_scale_linear"] = (
        float(recipe["noise_channel_scale_linear"])
        * noise_scale_factor
        * common_gain
    )
    measured_snr = STRESS._db(
        STRESS._rms(reference if np.any(near_mask) else far_end, target_mask),
        STRESS._rms(noise_0, target_mask),
    )
    measured_ser = (
        STRESS._db(STRESS._rms(echo_0, ser_mask), STRESS._rms(reference, ser_mask))
        if scope == "double_talk"
        else None
    )
    reconstructed_0 = np.zeros(len(reference), dtype=np.float64)
    reconstructed_1 = np.zeros(len(reference), dtype=np.float64)
    for segment in echo_segments:
        mask = STRESS._window(
            len(reference),
            rate,
            segment["start_s"],
            segment["end_s"],
        )
        reconstructed_0 += STRESS.MEASUREMENT._fir_component(
            far_end,
            segment["paths"][0],
        ) * mask * echo_gain
        reconstructed_1 += STRESS.MEASUREMENT._fir_component(
            far_end,
            segment["paths"][1],
        ) * mask * echo_gain
    scaled_delayed_reference_0 = reference
    scaled_delayed_reference_1 = np.concatenate((np.zeros(1), reference[:-1]))
    source_mix_residual = max(
        float(np.max(np.abs(
            microphone[:, 0]
            - scaled_delayed_reference_0
            - reconstructed_0
            - noise_0
        ))),
        float(np.max(np.abs(
            microphone[:, 1]
            - scaled_delayed_reference_1
            - reconstructed_1
            - noise_1
        ))),
    )
    if source_mix_residual > STRESS.PRE_PCM_MIX_RESIDUAL_LIMIT:
        raise ValueError(
            f"stress pre-PCM mixture residual {source_mix_residual} exceeds 1e-12"
        )
    signals = {
        "microphone": np.asarray(microphone, dtype=np.float64),
        "reference": np.asarray(reference, dtype=np.float64),
        "far_end": np.asarray(far_end, dtype=np.float64),
        "noise": np.asarray(noise_0, dtype=np.float64),
    }
    metadata = {
        "kind": "speech",
        "workload_scope": scope,
        "workload_category": category,
        "case_index": int(case_index),
        "conversation_mode": STRESS.conversation_mode_for(scope, category),
        "direction": "send",
        "bandwidth": bandwidth,
        "sample_rate_hz": int(rate),
        "duration_s": round(duration_s, 6),
        "analysis_start_s": STRESS.analysis_start_for(scope, category),
        "activity_segments": segments,
        "volume_gain_db": float(volume_gain_db),
        "near_arrival_delays_samples": list(delays),
        "beam_delays_ms": [1000.0 / rate, 0.0],
        "echo_path_segments": echo_segments,
        "echo_paths": [],
        "echo_gain_linear": float(echo_gain),
        "noise_spatial": {
            "returned_noise": "channel-0 noise; channel 1 is reconstructible",
            "channel_1_recipe": recipe,
        },
        "noise_snr_db_target": float(requested_snr),
        "noise_snr_db_measured": measured_snr,
        "ser_db_nominal_target": ser_nominal,
        "ser_db_effective_target": ser_effective,
        "ser_db_measured": measured_ser,
        "source_mix_check": {
            "applicable": True,
            "residual_max_abs": source_mix_residual,
            "formula": "microphone[:,ch] = delayed_reference + dynamic_echo + noise_ch",
        },
        "band_limit": {
            "near": near_filter,
            "far": far_filter,
            "noise": noise_filter,
        },
        "global_safety_gain": float(common_gain),
        "peak": float(np.max(np.abs(microphone))),
        "clipped": False,
        "engineering_preview": True,
        "standard_compliance": False,
        "evidence_scope": "synthetic acoustic stress engineering stimulus",
    }
    return signals, metadata


def _patched_crop_16k(
    bank,
    records: list[dict[str, Any]],
    *,
    group_id: int,
    role: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    CONVERSATION.OLD.DATASET_ID = BASE_GEN.DATASET_ID
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
    rate = BASE_GEN.RATE
    if rate == RATE_NB:
        samples = signal.resample_poly(samples_16k, 1, 2)
        cutoff = 3400.0
        samples = signal.sosfiltfilt(
            signal.butter(6, cutoff, fs=rate, output="sos"),
            samples,
        )
        resampling = {
            "method": "scipy.signal.resample_poly",
            "up": 1,
            "down": 2,
            "source_rate_hz": RATE_16K,
            "target_rate_hz": RATE_NB,
            "band_limit_hz": cutoff,
        }
    elif rate == RATE_SWB:
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
            "target_rate_hz": rate,
        }
    if len(samples) != BASE_GEN.FRAMES:
        raise ValueError(f"{role} resampled crop length differs from 12 seconds")
    info.update({
        "speaker_pool": role,
        "target_sample_rate_hz": rate,
        "resampling": resampling,
        "source_reuse_allowed_across_cases": True,
        "within_case_source_repetition": False,
    })
    return np.asarray(samples, dtype=np.float64), info


STRESS.render_stress_signals = _patched_render_stress_signals
BASE_GEN._crop_16k = _patched_crop_16k


def configure_mode_dataset(scope: str, mode: str) -> None:
    BASE_GEN.configure_dataset(scope, mode)
    bandwidth = mode.split("-", 1)[0]
    rate = _mode_rate(bandwidth)
    BASE_GEN.RATE = rate
    BASE_GEN.FRAMES = int(round(BASE_GEN.DURATION_S * rate))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create 100-workload acoustic stress dataset with 8/16/48 kHz rates."
    )
    parser.add_argument("--scope", choices=("echo", "double_talk"), required=True)
    parser.add_argument("--mode", choices=(*sorted(ACCEPTANCE_MODES),), required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    configure_mode_dataset(args.scope, args.mode)
    if args.overwrite and BASE_GEN.OUTPUT_ROOT.exists():
        shutil.rmtree(BASE_GEN.OUTPUT_ROOT)
    result = BASE_GEN.build(verify_store=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
