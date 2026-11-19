#!/usr/bin/env python3
"""Paired Speex fused CPU / LocalVQE comparison on a verified frozen A/B set.

No product configuration is changed. All candidates consume identical post-RX
playback and microphone PCM; metrics use the exact prior activity windows.
The fused candidate is measured at the post-fusion AEC tap with NS/AGC/DRC off.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("paired_ab_engine", ROOT / "scripts/run_mdf_localvqe_mask_ab_v2.py")
assert SPEC and SPEC.loader
ab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ab)
BASELINE = ROOT / "build/mdf-localvqe-ab-verified-20261112-r3-v0JjLL/result.json"
BASELINE_SHA = "d4a2189c143c7e7a25ede1eb27e625ec41ab2955fb47917038971d5577b1e3d9"


def assert_fused_identity(report: dict) -> None:
    expected = {
        "processing_profile": "vcp", "aec_backend_requested": "speex_fused_cpu",
        "aec_backend": "speex_mdf_nlms_fused_cpu", "aec_backend_effective": "speex_mdf_nlms_fused_cpu",
        "aec_execution_provider": "cpu", "aec_enabled": True,
        "speex_echo_tail_ms": 500, "speex_native_conditioning": True,
        "speex_coherence_residual_suppression": True, "speex_residual_suppression": False,
        "speex_post_context": True, "reference_channels": 1, "microphones": 1,
        "ns_enabled": False, "agc2_enabled": False, "tx_drc_enabled": False,
    }
    mismatches = {key: report.get(key) for key, expected_value in expected.items() if report.get(key) != expected_value}
    if mismatches:
        raise ValueError(f"Speex fused route/configuration not proven: {mismatches}")


def measurement_mask(row: dict, near: np.ndarray, playback: np.ndarray) -> np.ndarray:
    near_activity, far_activity = ab.activity_mask(near), ab.activity_mask(playback)
    group = row["category"]
    if group == "near_single":
        selected = near_activity
    elif group == "double_talk":
        selected = near_activity & far_activity
        if row["scene"]["source_category"] == "transition":
            designed = np.zeros(len(near), dtype=bool)
            designed[8 * ab.RATE:11 * ab.RATE] = True
            selected &= designed
        elif row["scene"]["source_category"] == "short_overlap":
            designed = np.zeros(len(near), dtype=bool)
            designed[round(6.5 * ab.RATE):7 * ab.RATE] = True
            selected &= designed
    else:
        selected = far_activity
    if int(np.sum(selected)) != round(row["activity"]["measurement_s"] * ab.RATE):
        raise ValueError("paired activity window differs from frozen baseline")
    return selected


def means(rows: list[dict]) -> dict:
    result = {}
    names = ("speex_fused_cpu", "localvqe_full", "mdf_residual_mask")
    for group in ab.GROUPS:
        cases = [row for row in rows if row["category"] == group]
        valid = [row for row in cases if row["measurement_status"] == "measured"]
        routes = {}
        for route in names:
            values = [row["routes"][route] for row in valid]
            stats = {}
            for field in ("output_rms", "output_attenuation_db", "near_si_sdr_db", "near_target_gain_db", "near_total_error_rms"):
                numbers = [value["metrics"][field] for value in values]
                stats[field] = float(np.mean(numbers)) if all(number is not None for number in numbers) else None
            stats["process_rtf"] = float(np.mean([value["timing"]["process_rtf"] for value in values]))
            stats["process_ms"] = float(np.mean([value["timing"]["process_ms"] for value in values]))
            if group == "path_change":
                stats["path_windows"] = {
                    window: float(np.mean([value["metrics"]["path_windows"][window]["output_attenuation_db"] for value in values]))
                    for window in ("before_switch", "after_switch_0_1s", "after_switch_2_6s")}
            routes[route] = stats
        result[group] = {"case_count": len(cases), "measured_case_count": len(valid), "routes": routes}
    return result


def run_case(row: dict, output: Path, engine) -> dict:
    ab.resource_check()
    source = BASELINE.parent / row["category"] / row["case_id"]
    work = output / row["category"] / row["case_id"]
    work.mkdir(parents=True, exist_ok=False)
    microphone, playback, clean, residual, yhat = [ab.read_mono(source / name) for name in
        ("mic.wav", "playback.wav", "clean.wav", "mdf-e.wav", "mdf-yhat.wav")]
    if len(microphone) != 12 * ab.RATE or len({len(x) for x in (microphone, playback, clean, residual, yhat)}) != 1:
        raise ValueError("all paired candidates require full-length identical inputs")
    for role, samples in (("mic", microphone), ("playback", playback), ("clean", clean)):
        if ab.digest(source / f"{role}.wav") != row["scene"]["artifacts"][role]["sha256"]:
            raise ValueError("source audio changed since the baseline")
    selected = measurement_mask(row, clean, playback)
    near_present = row["category"] in {"near_single", "double_talk"}
    prefix = work / "fused"
    command = [str(ab.CLI), "--profile", "vcp", "--aec-backend", "speex_fused_cpu", "--no-bf",
               "--no-ns", "--no-agc", "--no-tx-drc", "--no-limiter",
               "--mic", str(source / "mic.wav"), "--echo", str(source / "playback.wav"),
               "--out", str(prefix)]
    report, wall_ms = ab.call_cli(command, work / "fused.log")
    assert_fused_identity(report)
    delay = report.get("aec_output_delay_samples")
    if not isinstance(delay, int) or isinstance(delay, bool) or not 0 <= delay <= ab.RATE:
        raise ValueError("invalid fused AEC tap delay")
    tap = ab.read_mono(work / "fused.aec.wav")
    if len(tap) != len(microphone) + delay:
        raise ValueError("fused AEC tap duration mismatch")
    fused = tap[delay:delay + len(microphone)]
    routes = {}
    candidates = {"speex_fused_cpu": (fused, {
        "process_rtf": report["process_rtf"], "process_ms": report["process_rtf"] * 12000,
        "process_p95_ms": report["process_p95_ms"], "process_max_ms": report["process_max_ms"],
        "invocation_wall_ms": wall_ms,
        "scope": "C++ 10 ms callbacks, AEC and residual tail, NS/AGC/DRC/limiter disabled; no subprocess I/O"})}
    for name, first, second, mode in (
            ("localvqe_full", microphone, playback, "full"),
            ("mdf_residual_mask", residual, yhat, "mask")):
        aligned, raw, timing = engine.run(first, second, mode=mode)
        prior = row["routes"][name]["artifact"]
        if ab.pcm_digest(aligned) != prior["pcm_sha256"]:
            raise ValueError(f"LocalVQE replay differs from the frozen candidate: {name}")
        timing["scope"] = "CPU per-hop inference + FFI + flush; excludes model loading; mask route excludes MDF"
        candidates[name] = aligned, timing
    for name, (samples, timing) in candidates.items():
        metrics = ab.measure(samples, microphone, clean, selected, near_present=near_present) if row["measurement_status"] == "measured" else None
        if metrics is not None and row["category"] == "path_change":
            windows = {}
            for window, start, stop in (("before_switch", 3, 5.5), ("after_switch_0_1s", 6, 7), ("after_switch_2_6s", 8, 12)):
                mask = selected & (np.arange(len(samples)) >= round(start * ab.RATE)) & (np.arange(len(samples)) < round(stop * ab.RATE))
                windows[window] = ab.measure(samples, microphone, clean, mask, near_present=False)
            metrics["path_windows"] = windows
        artifact = ab.write_audio(work / f"{name}.aligned.wav", samples)
        routes[name] = {"artifact": artifact, "metrics": metrics, "timing": timing}
    record = {"case_id": row["case_id"], "category": row["category"], "measurement_status": row["measurement_status"],
              "measurement_reason": row.get("measurement_reason"), "activity": row["activity"],
              "inputs": {"folder": str(source), "mic_sha256": ab.digest(source / "mic.wav"),
                         "playback_sha256": ab.digest(source / "playback.wav"),
                         "reference_mode": "same matched post-RX as frozen A/B"},
              "fused_command": command, "fused_report_sha256": ab.digest(work / "fused.json"),
              "fused_aec_delay_samples": delay, "fused_report": report, "routes": routes,
              "all_candidates_pcm_replay_verified": True}
    ab.write_json(work / "case.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new empty evidence directory")
    args = parser.parse_args()
    ab.resource_check()
    if ab.digest(BASELINE) != BASELINE_SHA or ab.digest(ab.CLI) != ab.CLI_SHA or ab.digest(ab.MODEL) != ab.MODEL_SHA:
        raise ValueError("pinned comparison identities differ")
    baseline = json.loads(BASELINE.read_text())
    library = Path(baseline["identity"]["library"]["path"])
    if ab.digest(library) != baseline["identity"]["library"]["sha256"]:
        raise ValueError("LocalVQE bridge changed")
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("preserve old evidence; use a new empty directory")
    out.mkdir(parents=True, exist_ok=True)
    engine = ab.Engine(library)
    rows = []
    try:
        for row in baseline["cases"]:
            rows.append(run_case(row, out, engine))
            print(json.dumps({"finished": len(rows), "total": len(baseline["cases"]), "case": row["case_id"]}), flush=True)
    finally:
        engine.close()
    if ab.digest(BASELINE) != BASELINE_SHA or ab.digest(ab.CLI) != ab.CLI_SHA:
        raise ValueError("binary/report identity changed during the run")
    result = {"schema": "itu-speex-fused-localvqe-paired-v1", "created_at": datetime.now(timezone.utc).isoformat(),
              "source_result": {"path": str(BASELINE), "sha256": BASELINE_SHA},
              "runner_sha256": ab.digest(Path(__file__)),
              "measurement_runner_sha256": ab.digest(Path(ab.__file__)),
              "cli": {"path": str(ab.CLI), "sha256": ab.CLI_SHA},
              "library": baseline["identity"]["library"], "model": baseline["identity"]["model"],
              "case_count": len(rows), "case_duration_s": 12, "categories": means(rows), "cases": rows,
              "boundary": "paired synthetic mono WB-wired comparison at AEC stage; one invalid short-overlap stimulus excluded from quality means; timing scopes differ as recorded; no formal ITU-T/board or full-VCP claim",
              "formal_compliance": False}
    ab.write_json(out / "result.json", result)
    ab.write_json(out / "artifacts.json", {"files": {str(p.relative_to(out)): {"sha256": ab.digest(p), "bytes": p.stat().st_size}
        for p in sorted(out.rglob("*")) if p.is_file()}})
    print(json.dumps({"result": str(out / "result.json"), "sha256": ab.digest(out / "result.json"), "cases": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
