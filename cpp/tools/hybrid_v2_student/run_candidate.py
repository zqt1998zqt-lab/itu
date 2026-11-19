#!/usr/bin/env python3
"""Replay hybrid-v2 at the frozen VCP NS boundary.

The previous round's driver remains the behavioral reference.  This script
reuses its exact boundary/replay helpers, while validating the ITUNSR02 model,
the hybrid-v2 probe report, fixed 160-sample delay, and causal pitch contract.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import struct
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf

from evaluation.common import (
    BUILD_ROOT,
    FROZEN_CLI,
    OLD_RUN_CANDIDATE,
    append_text,
    atomic_json,
    materialize,
    require_executable,
    require_file,
    run_managed_process,
    safe_name,
    sha256_file,
    utc_now,
)


SCHEMA_VERSION = 1
SAMPLE_RATE = 16000
FRAME_SAMPLES = 160
NS_DELAY_SAMPLES = 160
DRAIN_HOPS = 12
EXTRA_LOOKAHEAD_SAMPLES = 0
REPLAY_TAIL_SAMPLES = 1600
DEFAULT_PROBE = BUILD_ROOT / "native/hybrid_v2_probe"
EXPECTED_PARAMS = {256: 45568, 320: 49664}
EXPECTED_FEATURE_VERSION = {256: 1, 320: 2}
EXPECTED_FEATURE_CONTRACT = {
    256: "dpdf-frame-v1-db",
    320: "dpdf-frame-v2-basegain-db",
}
HEADER = struct.Struct("<8s14I")


class CandidateError(RuntimeError):
    """A contract violation in the candidate boundary or replay."""


def load_previous_driver() -> Any:
    path = require_file(OLD_RUN_CANDIDATE, "previous-round run_candidate.py")
    spec = importlib.util.spec_from_file_location("itu_previous_candidate", path)
    if spec is None or spec.loader is None:
        raise CandidateError(f"cannot import previous driver: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


legacy = load_previous_driver()


def case_is_pure_noise(case: dict[str, Any]) -> bool:
    raw = case.get("raw", case)
    for key in ("pure_noise", "is_pure_noise"):
        if isinstance(raw.get(key), bool):
            return bool(raw[key])
    for key in ("speech_present", "has_speech"):
        if isinstance(raw.get(key), bool):
            return not bool(raw[key])
    kind = str(raw.get("kind") or raw.get("role") or "").strip().lower()
    return kind in {"noise", "noise_only", "pure_noise", "nonspeech", "non_speech", "interference_only"}


def first_value(document: dict[str, Any], names: Sequence[str], label: str) -> Any:
    present = [name for name in names if name in document]
    if not present:
        raise CandidateError(f"probe JSON is missing {label}: {', '.join(names)}")
    values = [document[name] for name in present]
    if any(value != values[0] for value in values[1:]):
        raise CandidateError(f"probe JSON has conflicting {label} aliases: {present}")
    return values[0]


def validate_model_metadata(
    model: Path,
    *,
    floor_db: float,
    absolute_floor_db: float,
) -> dict[str, Any]:
    model = require_file(model, "model weights")
    metadata_path = require_file(model.parent / "model.json", "model.json sidecar")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise CandidateError("model.json root is not an object")
    if metadata.get("format") != "ITUNSR02":
        raise CandidateError(f"model.json format must be ITUNSR02, got {metadata.get('format')!r}")
    if int(metadata.get("version", -1)) != 2:
        raise CandidateError(f"model.json version must be 2, got {metadata.get('version')!r}")
    input_dim = int(first_value(metadata, ("inputdim", "input_dim"), "inputdim/input_dim"))
    if input_dim not in EXPECTED_PARAMS:
        raise CandidateError(f"unsupported input_dim: {input_dim}")
    steps = int(metadata.get("steps", -1))
    if steps <= 0:
        raise CandidateError(f"model.json steps must be > 0, got {steps}")
    if metadata.get("trained") is not True:
        raise CandidateError("model.json trained must be true")
    if metadata.get("fixture") is not False:
        raise CandidateError("model.json fixture must be false")
    if metadata.get("quality_use") == "forbidden":
        raise CandidateError("model.json marks this export quality_use=forbidden")
    for key in ("newfeature", "new_feature"):
        if key in metadata and metadata[key] is not True:
            raise CandidateError(f"model.json {key} must be true when present")
    feature_version = int(metadata.get("feature_version", -1))
    if feature_version != EXPECTED_FEATURE_VERSION[input_dim]:
        raise CandidateError(
            f"feature_version {feature_version} != {EXPECTED_FEATURE_VERSION[input_dim]} for input_dim {input_dim}"
        )
    feature_contract = first_value(
        metadata, ("featurecontract", "feature_contract"), "featurecontract/feature_contract"
    )
    if str(feature_contract) != EXPECTED_FEATURE_CONTRACT[input_dim]:
        raise CandidateError(f"unexpected feature contract: {feature_contract!r}")
    gain_mode = first_value(metadata, ("gainmode", "gain_mode"), "gainmode/gain_mode")
    if str(gain_mode) != "residual_db":
        raise CandidateError(f"gain_mode must be residual_db, got {gain_mode!r}")
    if metadata.get("weights_file") not in (None, model.name):
        raise CandidateError("model.json weights_file disagrees with supplied weights")
    weights_sha = sha256_file(model)
    declared = metadata.get("weights_sha256") or metadata.get("sha256")
    if not isinstance(declared, str) or declared.lower() != weights_sha.lower():
        raise CandidateError("model.json weights SHA does not match weights.bin")

    header = metadata.get("header")
    if not isinstance(header, dict):
        raise CandidateError("model.json header is missing")
    if int(header.get("input_dim", -1)) != input_dim:
        raise CandidateError("model.json header input_dim disagrees")
    if int(header.get("trained", 0)) != 1:
        raise CandidateError("model.json header trained must be 1")
    if int(header.get("feature_version", -1)) != feature_version:
        raise CandidateError("model.json header feature_version disagrees")
    if int(header.get("parameter_count", -1)) != EXPECTED_PARAMS[input_dim]:
        raise CandidateError("model.json header parameter_count disagrees")

    with model.open("rb") as stream:
        raw_header = stream.read(HEADER.size)
    if len(raw_header) != HEADER.size:
        raise CandidateError("weights.bin is shorter than the ITUNSR02 header")
    magic, *values = HEADER.unpack(raw_header)
    if magic != b"ITUNSR02":
        raise CandidateError(f"weights.bin magic is not ITUNSR02: {magic!r}")
    version, rate, bands, binary_dim, projection, hidden, head, params, trained, feature, fft, hop, window, reserved = values
    if (version, rate, bands, binary_dim, projection, hidden, head, params) != (
        2,
        16000,
        64,
        input_dim,
        64,
        64,
        64,
        EXPECTED_PARAMS[input_dim],
    ):
        raise CandidateError("weights.bin header contract disagrees with model.json")
    if (trained, feature, fft, hop, window, reserved) != (1, feature_version, 512, 160, 320, 0):
        raise CandidateError("weights.bin trained/feature/DSP header is invalid")

    declared_base = metadata.get("base_floor_db", metadata.get("base_frontend_floor_db"))
    declared_final = metadata.get("absolute_floor_db", metadata.get("floor_db"))
    if declared_base is not None and abs(float(declared_base) - floor_db) > 1e-6:
        raise CandidateError(f"model base floor {declared_base} != requested {floor_db}")
    if declared_final is not None and abs(float(declared_final) - absolute_floor_db) > 1e-6:
        raise CandidateError(f"model final floor {declared_final} != requested {absolute_floor_db}")
    return {
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "weights_sha256": weights_sha,
        "input_dim": input_dim,
        "steps": steps,
        "trained": True,
        "fixture": False,
        "feature_version": feature_version,
        "feature_contract": str(feature_contract),
        "new_feature": feature_version in {1, 2},
        "parameter_count": EXPECTED_PARAMS[input_dim],
    }


def parse_pitch_stats(report: dict[str, Any], pitch_strength: float) -> dict[str, Any]:
    nested = report.get("pitch_stats")
    stats = copy.deepcopy(nested) if isinstance(nested, dict) else {}
    if not stats:
        for target, names in {
            "available": ("pitch_available",),
            "applied_frames": ("pitch_applied_frames", "pitch_active_frames"),
            "modified_bins": ("pitch_modified_bins",),
        }.items():
            for name in names:
                if name in report:
                    stats[target] = report[name]
                    break
    reported_strength = report.get("pitch_strength", 0.0 if pitch_strength == 0.0 else None)
    if not isinstance(reported_strength, (int, float)) or isinstance(reported_strength, bool):
        raise CandidateError("probe JSON is missing numeric pitch_strength")
    if abs(float(reported_strength) - pitch_strength) > 1e-6:
        raise CandidateError(f"probe pitch_strength {reported_strength} != requested {pitch_strength}")
    applied = int(stats.get("applied_frames", stats.get("active_frames", 0)))
    bins = int(stats["modified_bins"]) if "modified_bins" in stats else None
    fallback = int(stats.get("fallback_frames", stats.get("fallback", 0)))
    if applied < 0 or fallback < 0 or (bins is not None and bins < 0):
        raise CandidateError("pitch statistics must be non-negative")
    return {
        "available": bool(stats.get("available", False)),
        "applied_frames": applied,
        "modified_bins": bins,
        "fallback_frames": fallback,
        "lag_samples": int(stats.get("lag_samples", 0)),
        "periodicity": float(stats.get("periodicity", 0.0)),
        "requested_strength": pitch_strength,
    }


def parse_probe_report(
    report: dict[str, Any],
    *,
    input_samples: int,
    model_enabled: bool,
    mix: float,
    pitch_strength: float,
    pure_noise: bool,
    expected_input_dim: int | None,
    floor_db: float,
    absolute_floor_db: float,
) -> dict[str, Any]:
    delay = int(first_value(report, ("delay_samples", "delay"), "delay_samples/delay"))
    if delay != NS_DELAY_SAMPLES:
        raise CandidateError(f"probe delay must be {NS_DELAY_SAMPLES}, got {delay}")
    frames = int(first_value(report, ("frames",), "frames"))
    expected_frames = int(math.ceil(input_samples / FRAME_SAMPLES))
    if frames != expected_frames:
        raise CandidateError(f"probe frames {frames} != expected {expected_frames}")
    if "drain_hops" in report and int(report["drain_hops"]) != DRAIN_HOPS:
        raise CandidateError(f"probe drain_hops must be {DRAIN_HOPS}")
    for key in ("extra_lookahead_samples", "extra_lookahead", "lookahead_samples"):
        if key in report and int(report[key]) != EXTRA_LOOKAHEAD_SAMPLES:
            raise CandidateError(f"probe {key} must be {EXTRA_LOOKAHEAD_SAMPLES}")

    mean_ms = float(
        first_value(
            report,
            (
                "mean_process_ms",
                "mean_ms",
                "ns_scope_mean_process_ms",
                "process_ms_mean",
            ),
            "mean_process_ms",
        )
    )
    if not math.isfinite(mean_ms) or mean_ms < 0:
        raise CandidateError(f"probe mean process time is invalid: {mean_ms}")
    process_scope = report.get("process_ms_scope")
    accepted_scopes = {
        "omlsa_imcra_engine_process",
        "omlsa_features_gru_and_optional_pitch_excludes_io_quantize_drain",
    }
    if process_scope is not None and str(process_scope) not in accepted_scopes:
        raise CandidateError(f"unexpected probe process_ms_scope: {process_scope!r}")

    state_values: list[int] = []
    for key in ("modelstate", "student_state", "state"):
        if key not in report:
            continue
        value = report[key]
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"ready", "loaded", "true"}:
                parsed = 1
            elif text in {"absent", "none", "false"}:
                parsed = 0
            else:
                parsed = int(text)
        else:
            parsed = int(value)
        state_values.append(parsed)
    if not state_values:
        raise CandidateError("probe JSON is missing modelstate/state")
    if any(value != state_values[0] for value in state_values[1:]):
        raise CandidateError(f"probe state aliases disagree: {state_values}")
    state = state_values[0]
    inferences = int(first_value(report, ("inferences", "student_inferences"), "inferences"))
    fallback = int(first_value(report, ("fallback", "fallback_frames", "student_fallback_frames"), "fallback"))
    applied = int(first_value(report, ("applied", "applied_frames", "student_applied_frames"), "applied"))
    if min(inferences, fallback, applied) < 0:
        raise CandidateError("probe student counters must be non-negative")
    loaded_raw = report.get("model_loaded", report.get("student_loaded", report.get("loaded")))
    loaded = bool(loaded_raw) if loaded_raw is not None else state == 1
    if isinstance(loaded_raw, str):
        loaded = loaded_raw.strip().lower() in {"1", "true", "ready", "loaded", "yes"}

    if model_enabled:
        if state != 1 or not loaded:
            raise CandidateError(f"model-enabled probe is not ready: state={state}, loaded={loaded}")
        if inferences < frames:
            raise CandidateError(f"model-enabled probe inferences {inferences} < frames {frames}")
        if fallback != 0:
            raise CandidateError(f"model-enabled probe fallback must be 0, got {fallback}")
        if mix > 0 and applied <= 0:
            raise CandidateError("model-enabled probe applied no student frames")
        if mix == 0 and applied != 0:
            raise CandidateError("mix=0 negative control applied student frames")
    else:
        if state != 0 or loaded or inferences != 0 or applied != 0:
            raise CandidateError("no-model baseline must report absent state, inferences=0, applied=0")

    input_dim = int(report.get("inputdim", report.get("input_dim", 0)))
    paramcount = int(report.get("paramcount", report.get("parameter_count", 0)))
    if model_enabled:
        if expected_input_dim is None or input_dim != expected_input_dim:
            raise CandidateError(f"probe inputdim {input_dim} != model inputdim {expected_input_dim}")
        if paramcount != EXPECTED_PARAMS[expected_input_dim]:
            raise CandidateError(f"probe paramcount {paramcount} is invalid for inputdim {expected_input_dim}")

    reported_floor = first_value(
        report,
        ("base_floor_db", "floor_db", "effective_floor_db", "residual_floor_db"),
        "base_floor_db",
    )
    reported_final = first_value(
        report,
        ("final_floor_db", "absolute_floor_db", "effective_absolute_floor_db"),
        "final_floor_db",
    )
    if abs(float(reported_floor) - floor_db) > 1e-6:
        raise CandidateError(f"probe base floor {reported_floor} != requested {floor_db}")
    if abs(float(reported_final) - absolute_floor_db) > 1e-6:
        raise CandidateError(f"probe final floor {reported_final} != requested {absolute_floor_db}")
    if report.get("output_subtype") not in (None, "FLOAT"):
        raise CandidateError(f"probe output_subtype must be FLOAT, got {report.get('output_subtype')!r}")
    if "rate" in report and int(report["rate"]) != SAMPLE_RATE:
        raise CandidateError(f"probe rate {report['rate']} != {SAMPLE_RATE}")
    reported_mix = float(report.get("mix", mix))
    if abs(reported_mix - mix) > 1e-6:
        raise CandidateError(f"probe mix {reported_mix} != requested {mix}")

    pitch = parse_pitch_stats(report, pitch_strength)
    if pitch_strength == 0.0:
        if pitch["applied_frames"] != 0 or pitch["fallback_frames"] != 0 or pitch["modified_bins"] not in (None, 0):
            raise CandidateError("pitch-disabled run modified pitch")
    else:
        if pitch["modified_bins"] is not None and pitch["modified_bins"] <= 0:
            if pitch["applied_frames"] > 0:
                raise CandidateError("pitch-enabled voiced case applied frames without modified bins")

    runtime = report.get("runtime")
    runtime = None if runtime is None else float(runtime)
    if runtime is not None and (not math.isfinite(runtime) or runtime < 0):
        raise CandidateError("probe runtime is invalid")
    inference_seconds = None
    inference_source = None
    for key, scale in (
        ("student_inference_sec", 1.0),
        ("inference_sec", 1.0),
        ("student_inference_ms", 0.001),
        ("inference_ms", 0.001),
        ("total_inference_ms", 0.001),
    ):
        value = report.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            inference_seconds = float(value) * scale
            inference_source = key
            break
    if inference_seconds is None:
        inference_seconds = mean_ms * frames / 1000.0
        inference_source = "derived_from_mean_process_ms"
    return {
        "delay_samples": delay,
        "frames": frames,
        "expected_input_frames": expected_frames,
        "modelstate": "ready" if state == 1 else "absent",
        "model_loaded": loaded,
        "inferences": inferences,
        "fallback": fallback,
        "applied": applied,
        "inputdim": input_dim,
        "paramcount": paramcount,
        "base_floor_db": float(reported_floor),
        "final_floor_db": float(reported_final),
        "mix": reported_mix,
        "pitch": pitch,
        "output_subtype": report.get("output_subtype"),
        "runtime_s": runtime,
        "mean_process_ms": mean_ms,
        "ns_inference_s": inference_seconds,
        "ns_inference_s_source": inference_source,
        "process_ms_scope": process_scope,
        "drain_hops": int(report.get("drain_hops", DRAIN_HOPS)),
        "extra_lookahead_samples": EXTRA_LOOKAHEAD_SAMPLES,
        "student_fields_present": True,
    }


def bundle_stages(
    metadata: dict[str, Any], destination: Path, store: Path | None
) -> tuple[dict[str, Any], Path]:
    stages = metadata.get("pipeline_stages")
    if not isinstance(stages, list) or not stages:
        raise CandidateError(f"stage metadata has no pipeline stages: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    copied = copy.deepcopy(metadata)
    for index, (source_entry, target_entry) in enumerate(zip(stages, copied["pipeline_stages"])):
        if not isinstance(source_entry, dict):
            raise CandidateError(f"pipeline_stages[{index}] is not an object")
        stage_id = str(source_entry.get("id") or f"stage-{index}")
        source_file = require_file(Path(str(source_entry.get("file", ""))), f"stage {stage_id} audio")
        target_file = destination / f"{index:02d}-{safe_name(stage_id)}.wav"
        digest = materialize(source_file, target_file, store)
        target_entry["file"] = str(target_file)
        target_entry["file_sha256"] = digest
    metadata_path = destination / "stages.json"
    atomic_json(metadata_path, copied)
    return copied, metadata_path


def execute_case(
    raw_case: dict[str, Any],
    cases_base: Path,
    output_dir: Path,
    *,
    cli: Path,
    probe: Path,
    model: Path | None,
    variant: str,
    floor_db: float,
    absolute_floor_db: float,
    mix: float,
    pitch_strength: float,
    store: Path | None,
) -> dict[str, Any]:
    normalized_raw = copy.deepcopy(raw_case)
    if not normalized_raw.get("clean_path") and normalized_raw.get("input_clean_path"):
        normalized_raw["clean_path"] = normalized_raw["input_clean_path"]
    case = legacy.normalize_case(normalized_raw, cases_base)
    case_id = case["id"]
    pure_noise = case_is_pure_noise(case)
    case_dir = output_dir / "cases" / safe_name(case_id) / safe_name(variant)
    case_dir.mkdir(parents=True, exist_ok=False)
    receipt: dict[str, Any] = {
        "case": case,
        "variant": variant,
        "status": "running",
        "started_utc": utc_now(),
        "cli_path": str(cli),
        "cli_sha256": sha256_file(cli),
        "probe_path": str(probe),
        "native_sha256": sha256_file(probe),
        "native_sha256_before": sha256_file(probe),
        "input_sha256": sha256_file(Path(case["mic_path"])),
        "model_path": str(model) if model else None,
        "model_sha256": sha256_file(model) if model else None,
        "model_sha256_before": sha256_file(model) if model else None,
        "floor_db": floor_db,
        "absolute_floor_db": absolute_floor_db,
        "mix": mix,
        "pitch_strength": pitch_strength,
        "pure_noise": pure_noise,
        "ns_delay_samples": NS_DELAY_SAMPLES,
        "drain_hops": DRAIN_HOPS,
        "extra_lookahead_samples": EXTRA_LOOKAHEAD_SAMPLES,
        "mode": "hybrid_v2_student" if model else "independent_stat_baseline",
        "not_c4eb": True,
        "commands": {},
        "processes": {},
    }
    try:
        model_metadata = validate_model_metadata(
            model, floor_db=floor_db, absolute_floor_db=absolute_floor_db
        ) if model is not None else None
        receipt["model_metadata"] = model_metadata
        receipt["model_metadata_sha256_before"] = model_metadata["metadata_sha256"] if model_metadata else None
        receipt["native_hash_role"] = "external identity only; native loader uses FNV for corruption checks"
        receipt["native_identity_only"] = True

        boundary_prefix = case_dir / "boundary" / "boundary"
        boundary_prefix.parent.mkdir(parents=True, exist_ok=True)
        boundary_command = [
            str(cli),
            "--profile",
            "vcp_adaptive",
            "--no-aec",
            "--mic",
            str(case["mic_path"]),
            "--diagnostic-stages",
            "--out",
            str(boundary_prefix),
        ]
        receipt["commands"]["boundary"] = boundary_command
        boundary_process = run_managed_process(
            name=f"boundary:{variant}:{case_id}",
            command=boundary_command,
            step_dir=case_dir / "steps/boundary",
        )
        receipt["processes"]["boundary"] = boundary_process
        receipt["boundary_exit"] = boundary_process["returncode"]
        receipt["boundary_wall_s"] = boundary_process["elapsed_s"]
        if boundary_process["returncode"] != 0:
            raise CandidateError(f"frozen CLI boundary failed with exit {boundary_process['returncode']}")
        boundary_metadata_file = require_file(
            legacy.stage_metadata_path(boundary_prefix), "boundary stage metadata"
        )
        boundary_metadata = json.loads(boundary_metadata_file.read_text(encoding="utf-8"))
        boundary_stage = legacy.find_stage(boundary_metadata, "tx_ns_input")
        boundary_stage_file = require_file(Path(str(boundary_stage["file"])), "boundary tx_ns_input")
        first_input, first_rate = legacy.read_mono_float(boundary_stage_file, "boundary tx_ns_input")
        if first_rate != SAMPLE_RATE:
            raise CandidateError(f"tx_ns_input rate {first_rate} != {SAMPLE_RATE}")
        if int(boundary_metadata.get("input_samples_per_channel", -1)) != len(first_input):
            raise CandidateError("tx_ns_input length disagrees with stage metadata")
        if sf.info(boundary_stage_file).subtype != "FLOAT":
            raise CandidateError(f"tx_ns_input must be FLOAT: {boundary_stage_file}")

        candidate_input = case_dir / "input/boundaryfloat.wav"
        input_float_sha = materialize(boundary_stage_file, candidate_input, store)
        input_pcm16_sha = legacy.pcm16_lround_sha256(first_input)
        (candidate_input.parent / "tx_ns_input.pcm16-lround.sha256").write_text(
            input_pcm16_sha + "\n", encoding="utf-8"
        )
        receipt.update(
            {
                "input_samples": int(len(first_input)),
                "input_sample_rate": first_rate,
                "input_stage_path": str(boundary_stage_file),
                "candidate_input_float_path": str(candidate_input),
                "candidate_input_float_sha256": input_float_sha,
                "candidate_input_pcm16_lround_sha256": input_pcm16_sha,
                "pcm16_quantization": "probe performs one C++ lround; runner writes FLOAT only",
            }
        )

        probe_prefix = case_dir / "probe/candidate"
        probe_prefix.parent.mkdir(parents=True, exist_ok=True)
        probe_command = [
            str(probe),
            "--mic",
            str(candidate_input),
            "--out",
            str(probe_prefix),
            "--floor-db",
            format(floor_db, ".12g"),
            "--absolute-floor-db",
            format(absolute_floor_db, ".12g"),
            "--mix",
            format(mix, ".12g"),
        ]
        if pitch_strength > 0.0:
            probe_command += ["--pitch-strength", format(pitch_strength, ".12g")]
        if model is not None:
            probe_command += ["--model", str(model)]
        receipt["commands"]["probe"] = probe_command
        probe_process = run_managed_process(
            name=f"probe:{variant}:{case_id}",
            command=probe_command,
            step_dir=case_dir / "steps/probe",
        )
        receipt["processes"]["probe"] = probe_process
        receipt["probe_exit"] = probe_process["returncode"]
        receipt["probe_wall_s"] = probe_process["elapsed_s"]
        if probe_process["returncode"] != 0:
            raise CandidateError(f"native hybrid-v2 probe failed with exit {probe_process['returncode']}")
        raw_probe_after = require_file(
            probe_prefix.with_name(probe_prefix.name + ".after.wav"), "probe after.wav"
        )
        raw_probe_report = require_file(
            probe_prefix.with_name(probe_prefix.name + ".json"), "probe JSON"
        )
        after_path = case_dir / "raw.after.wav"
        report_path = case_dir / "probe.json"
        after_sha = materialize(raw_probe_after, after_path, store)
        materialize(raw_probe_report, report_path, store)
        raw_after, after_rate = legacy.read_mono_float(after_path, "probe after.wav")
        if after_rate != SAMPLE_RATE:
            raise CandidateError(f"probe output rate {after_rate} != {SAMPLE_RATE}")
        after_subtype = sf.info(after_path).subtype
        if after_subtype != "FLOAT":
            raise CandidateError(f"probe output must be FLOAT, got {after_subtype}")
        if len(raw_after) != len(first_input) + NS_DELAY_SAMPLES:
            raise CandidateError(
                f"probe output length {len(raw_after)} != input+160 ({len(first_input) + NS_DELAY_SAMPLES})"
            )
        probe_report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(probe_report, dict):
            raise CandidateError("probe JSON root is not an object")
        probe_values = parse_probe_report(
            probe_report,
            input_samples=len(first_input),
            model_enabled=model is not None,
            mix=mix,
            pitch_strength=pitch_strength,
            pure_noise=pure_noise,
            expected_input_dim=model_metadata["input_dim"] if model_metadata else None,
            floor_db=floor_db,
            absolute_floor_db=absolute_floor_db,
        )
        receipt.update(
            {
                "raw_after_path": str(after_path),
                "raw_after_sha256": after_sha,
                "raw_after_subtype": after_subtype,
                "raw_after_samples": int(len(raw_after)),
                "probe_json_path": str(report_path),
                "probe_json_sha256": sha256_file(report_path),
                "probe_values": probe_values,
                "ns_inferences": probe_values["inferences"],
                "ns_wall_s": probe_process["elapsed_s"],
                "ns_inference_s": probe_values["ns_inference_s"],
                "ns_inference_s_source": probe_values["ns_inference_s_source"],
                "ns_process_s": probe_values["ns_inference_s"],
                "ns_inference_scope": "native hybrid-v2 NS engine.process incl OMLSA, features, GRU, causal pitch; excludes IO/quantize/drain",
                "rtf_ns_process": probe_values["ns_inference_s"] / (len(first_input) / SAMPLE_RATE),
                "floor_db_effective": probe_values["base_floor_db"],
                "absolute_floor_db_effective": probe_values["final_floor_db"],
                "pitch_enabled": pitch_strength > 0,
                "pitch_effective": probe_values["pitch"],
            }
        )

        replay_input = case_dir / "replay/ns_replay.float.wav"
        replay_values = np.concatenate(
            [raw_after, np.zeros(REPLAY_TAIL_SAMPLES, dtype=np.float32)]
        )
        legacy.write_float_wav(replay_input, replay_values, SAMPLE_RATE)
        replay_check, replay_check_rate = legacy.read_mono_float(replay_input, "replay input")
        if replay_check_rate != SAMPLE_RATE or not np.array_equal(replay_check, replay_values):
            raise CandidateError("replay FLOAT WAV changed payload")
        receipt["replay_input_path"] = str(replay_input)
        receipt["replay_input_sha256"] = sha256_file(replay_input)
        receipt["replay_tail_samples"] = REPLAY_TAIL_SAMPLES
        receipt["replay_tail_policy"] = "append zero tail; never left-pad the 160-sample algorithm delay"
        receipt["probe_conversion_contract"] = {
            "input_float_sha256": input_float_sha,
            "input_pcm16_lround_sha256": input_pcm16_sha,
            "probe_output_subtype": after_subtype,
            "probe_output_samples": int(len(raw_after)),
            "expected_output_samples": int(len(first_input) + NS_DELAY_SAMPLES),
            "replay_payload_preserved_exact": True,
            "double_quantization": False,
            "current_and_past_input_only": True,
            "drain_hops": DRAIN_HOPS,
            "extra_lookahead_samples": EXTRA_LOOKAHEAD_SAMPLES,
        }

        replay_prefix = case_dir / "replay/processed"
        replay_command = [
            str(cli),
            "--profile",
            "vcp_adaptive",
            "--no-aec",
            "--mic",
            str(case["mic_path"]),
            "--diagnostic-stages",
            "--experiment-ns-pcm",
            str(replay_input),
            "--experiment-ns-delay-samples",
            str(NS_DELAY_SAMPLES),
            "--out",
            str(replay_prefix),
        ]
        receipt["commands"]["replay"] = replay_command
        receipt["command"] = replay_command
        replay_process = run_managed_process(
            name=f"replay:{variant}:{case_id}",
            command=replay_command,
            step_dir=case_dir / "steps/replay",
        )
        receipt["processes"]["replay"] = replay_process
        receipt["replay_exit"] = replay_process["returncode"]
        receipt["tail_runtime_s"] = replay_process["elapsed_s"]
        if replay_process["returncode"] != 0:
            raise CandidateError(f"frozen CLI replay failed with exit {replay_process['returncode']}")
        replay_metadata_file = require_file(
            legacy.stage_metadata_path(replay_prefix), "replay stage metadata"
        )
        replay_metadata = json.loads(replay_metadata_file.read_text(encoding="utf-8"))
        replay_stage = legacy.find_stage(replay_metadata, "tx_ns_input")
        replay_stage_file = require_file(Path(str(replay_stage["file"])), "replay tx_ns_input")
        second_input, second_rate = legacy.read_mono_float(replay_stage_file, "replay tx_ns_input")
        if second_rate != first_rate or not np.array_equal(first_input, second_input):
            raise CandidateError("upstream tx_ns_input changed between boundary and replay")
        receipt["upstream_input_exact"] = True
        receipt["upstream_input_sha256"] = sha256_file(replay_stage_file)
        receipt["upstream_stage_file_sha_equal"] = sha256_file(replay_stage_file) == sha256_file(boundary_stage_file)

        boundary_copied, boundary_copied_metadata = bundle_stages(
            boundary_metadata, case_dir / "boundary/stages", store
        )
        replay_copied, replay_copied_metadata = bundle_stages(
            replay_metadata, case_dir / "replay/stages", store
        )
        receipt["boundary_stages_path"] = str(boundary_copied_metadata)
        receipt["stages_path"] = str(replay_copied_metadata)
        receipt["stagespath"] = str(replay_copied_metadata)
        receipt["cli_sha"] = receipt["cli_sha256"]
        receipt["input_sha"] = receipt["input_sha256"]
        receipt["nativehash"] = receipt["native_sha256"]
        receipt["modelhash"] = receipt["model_sha256"]
        receipt["floor"] = floor_db
        receipt["absolute_floor"] = absolute_floor_db
        receipt["external_stages"] = replay_copied.get("pipeline_stages", [])
        receipt["boundary_stages"] = boundary_copied.get("pipeline_stages", [])

        input_saved = case_dir / "input.wav"
        materialize(boundary_stage_file, input_saved, store)
        receipt["input_saved_path"] = str(input_saved)
        if case.get("clean_path"):
            clean_saved = case_dir / "clean.wav"
            materialize(Path(case["clean_path"]), clean_saved, store)
            receipt["clean_saved_path"] = str(clean_saved)
        else:
            receipt["clean_saved_path"] = None
        receipt["clean_used_for_processing"] = False
        receipt["clean_policy"] = "clean is copied after processing and is never read as a feature"

        model_after = sha256_file(model) if model is not None else None
        native_after = sha256_file(probe)
        if model_after != receipt["model_sha256_before"]:
            raise CandidateError("model weights changed during candidate run")
        if native_after != receipt["native_sha256_before"]:
            raise CandidateError("native probe changed during candidate run")
        receipt["model_sha256_after"] = model_after
        receipt["native_sha256_after"] = native_after
        if model_metadata is not None:
            metadata_after = sha256_file(Path(model_metadata["metadata_path"]))
            if metadata_after != receipt["model_metadata_sha256_before"]:
                raise CandidateError("model.json changed during candidate run")
            receipt["model_metadata_sha256_after"] = metadata_after
        else:
            receipt["model_metadata_sha256_after"] = None
        receipt["identity_stable"] = True
        receipt["nativehash"] = native_after
        receipt["status"] = "ok"
        receipt["finished_utc"] = utc_now()
        receipt["runtime_s"] = replay_process["elapsed_s"]
        receipt["all_wall_s"] = (
            boundary_process["elapsed_s"] + probe_process["elapsed_s"] + replay_process["elapsed_s"]
        )
        return receipt
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(error).__name__}: {error}"
        receipt["finished_utc"] = utc_now()
        atomic_json(case_dir / "receipt.json", receipt)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--floor-db", type=float, default=-9.0)
    parser.add_argument("--absolute-floor-db", type=float, default=-21.0)
    parser.add_argument("--mix", type=float, default=1.0)
    parser.add_argument("--pitch-strength", type=float, default=0.0)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--hardlink-store", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    if not 0.0 <= args.mix <= 1.0:
        parser.error("--mix must be in [0, 1]")
    if not 0.0 <= args.pitch_strength <= 1.0:
        parser.error("--pitch-strength must be in [0, 1]")
    if not math.isfinite(args.floor_db) or not math.isfinite(args.absolute_floor_db):
        parser.error("floors must be finite")
    cli = require_executable(FROZEN_CLI, "frozen CLI")
    probe = require_executable(args.probe, "hybrid-v2 native probe")
    model = require_file(args.model, "model") if args.model else None
    cases_path = require_file(args.cases, "cases manifest")
    rows = legacy.load_case_rows(cases_path)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    store = args.hardlink_store.expanduser().resolve() if args.hardlink_store else None
    progress_path = output_dir / "progress.json"
    atomic_json(
        progress_path,
        {
            "status": "running",
            "variant": args.variant,
            "cases": len(rows),
            "completed": 0,
            "failed": 0,
            "started_utc": utc_now(),
        },
    )
    runs: list[dict[str, Any]] = []
    failures = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(
                execute_case,
                row,
                cases_path.parent,
                output_dir,
                cli=cli,
                probe=probe,
                model=model,
                variant=args.variant,
                floor_db=args.floor_db,
                absolute_floor_db=args.absolute_floor_db,
                mix=args.mix,
                pitch_strength=args.pitch_strength,
                store=store,
            ): row
            for row in rows
        }
        for future in as_completed(futures):
            row = futures[future]
            case_id = str(row.get("id") or row.get("case_id") or "unknown")
            try:
                receipt = future.result()
                print(f"OK {case_id} {args.variant}", flush=True)
            except Exception as error:
                failures += 1
                path = output_dir / "cases" / safe_name(case_id) / safe_name(args.variant) / "receipt.json"
                receipt = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {
                    "case": copy.deepcopy(row),
                    "variant": args.variant,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "finished_utc": utc_now(),
                }
                print(f"FAILED {case_id} {error}", flush=True)
            atomic_json(
                output_dir / "cases" / safe_name(case_id) / safe_name(args.variant) / "receipt.json",
                receipt,
            )
            runs.append(receipt)
            atomic_json(
                progress_path,
                {
                    "status": "running",
                    "variant": args.variant,
                    "cases": len(rows),
                    "completed": len(runs),
                    "failed": failures,
                    "last_case": case_id,
                    "updated_utc": utc_now(),
                },
            )
    runs.sort(key=lambda item: str(item.get("case", {}).get("id", "")))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "status": "complete" if failures == 0 else "failed",
        "variant": args.variant,
        "cases_manifest": str(cases_path),
        "cases_manifest_sha256": sha256_file(cases_path),
        "cli_path": str(cli),
        "cli_sha256": sha256_file(cli),
        "probe_path": str(probe),
        "native_sha256": sha256_file(probe),
        "model_path": str(model) if model else None,
        "model_sha256": sha256_file(model) if model else None,
        "floor_db": args.floor_db,
        "absolute_floor_db": args.absolute_floor_db,
        "mix": args.mix,
        "pitch_strength": args.pitch_strength,
        "ns_delay_samples": NS_DELAY_SAMPLES,
        "drain_hops": DRAIN_HOPS,
        "extra_lookahead_samples": EXTRA_LOOKAHEAD_SAMPLES,
        "replay_tail_samples": REPLAY_TAIL_SAMPLES,
        "mode": "hybrid_v2_student" if model else "independent_stat_baseline",
        "not_c4eb": True,
        "runs": runs,
        "failed": failures,
        "completed": len(runs) - failures,
    }
    atomic_json(output_dir / "receipt.json", summary)
    atomic_json(
        progress_path,
        {
            "status": summary["status"],
            "variant": args.variant,
            "cases": len(rows),
            "completed": len(runs),
            "failed": failures,
            "finished_utc": utc_now(),
        },
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

