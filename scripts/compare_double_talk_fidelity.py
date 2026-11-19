#!/usr/bin/env python3
"""Five-route double-talk fidelity on hash-frozen matched post-RX inputs.

Reuse three proven route outputs, execute missing Speex/AEC3 routes, and score
all outputs on the PCM16 grid. STOI always receives a contiguous time window.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import warnings

import numpy as np
from pystoi import stoi
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("speex_fidelity_support", ROOT / "scripts/compare_speex_echo_erle.py")
assert SPEC and SPEC.loader
speex = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(speex)
four, ab = speex.four, speex.ab
paired = four.base.paired
PAIRED = ROOT / "build/speex-fused-localvqe-paired-20261112-74pnNM/result.json"
PAIRED_SHA = "5bdfb6ab07fa6bae3636bf52b101418f160b8fef1a46dfce5eecfa01caa088d1"
ERLE = ROOT / "build/speex-echo-erle-20261112-3nd6pz2c/result.json"
ERLE_SHA = "c18c83707d1862e00d478bd67e74900ea8fc0be0be895c6ff2016942b1a57993"
ROUTES = ("speex_mdf_coherence", "speex_fused_cpu", "webrtc_aec3", "localvqe_full", "mdf_residual_mask")


def pcm16(audio: np.ndarray) -> np.ndarray:
    x = np.clip(audio.astype(np.float64), -1, 32767 / 32768)
    return np.asarray(np.sign(x) * np.floor(np.abs(x) * 32768 + .5) / 32768, dtype=np.float32)


def contiguous_stoi(target: np.ndarray, observed: np.ndarray) -> dict:
    if target.shape != observed.shape or target.ndim != 1 or not np.isfinite(target).all() or not np.isfinite(observed).all():
        raise ValueError("STOI requires equally sized finite contiguous mono windows")
    if not np.any(observed):
        return {"value": None, "available": False, "reason": "zero output is not an intelligibility score"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = float(stoi(target, observed, ab.RATE, extended=False))
    if caught or not np.isfinite(value):
        return {"value": None, "available": False, "reason": "; ".join(str(w.message) for w in caught) or "nonfinite score"}
    return {"value": value, "available": True, "reason": ""}


def continuous_metrics(audio, microphone, clean, start_s, end_s) -> dict:
    lo, hi = round(start_s * ab.RATE), round(end_s * ab.RATE)
    if not 0 <= lo < hi <= len(clean):
        raise ValueError("continuous fidelity window outside original timeline")
    y, m, target = audio[lo:hi], microphone[lo:hi], clean[lo:hi]
    values = ab.measure(y, m, target, np.ones(len(y), dtype=bool), near_present=True)
    values["stoi"] = contiguous_stoi(target, y)
    return {"start_s": start_s, "end_s": end_s, "original_samples": hi - lo, **values}


def controls() -> dict:
    n = 32000
    t = np.arange(n, dtype=np.float64) / ab.RATE
    target = (.08 * np.sin(2 * np.pi * 217 * t) + .04 * np.sin(2 * np.pi * 743 * t)) * (.7 + .3 * np.sin(2 * np.pi * 4 * t))
    mask = np.ones(n, dtype=bool)
    perfect = ab.measure(target, target, target, mask, near_present=True)
    quiet = ab.measure(target * .01, target, target, mask, near_present=True)
    muted = ab.measure(np.zeros(n), target, target, mask, near_present=True)
    if not (perfect["near_si_sdr_db"] > 100 and abs(perfect["near_target_gain_db"]) < 1e-10
            and quiet["near_si_sdr_db"] > 100 and abs(quiet["near_target_gain_db"] + 40) < 1e-10
            and quiet["near_retention_probe"] is False and muted["near_si_sdr_db"] is None
            and muted["near_retention_probe"] is False):
        raise ValueError("fidelity scorer cannot detect a near-speech mute/gain failure")
    score = contiguous_stoi(target.astype(np.float32), target.astype(np.float32))
    if not score["available"] or abs(score["value"] - 1) > 1e-6:
        raise ValueError("continuous STOI identity control failed")
    return {"perfect_si_sdr_db": perfect["near_si_sdr_db"], "perfect_target_gain_db": perfect["near_target_gain_db"],
            "quiet_si_sdr_db": quiet["near_si_sdr_db"], "quiet_target_gain_db": quiet["near_target_gain_db"],
            "quiet_retention_rejected": True, "muted_retention_rejected": True, "stoi_identity": score["value"], "passed": True}


def run_case(row: dict, prior: dict, output: Path) -> dict:
    ab.resource_check()
    source = paired.BASELINE.parent / row["category"] / row["case_id"]
    previous = PAIRED.parent / row["category"] / row["case_id"]
    work = output / row["case_id"]
    work.mkdir(parents=True, exist_ok=False)
    arrays = {}
    for role in ("mic", "playback", "clean"):
        path = source / f"{role}.wav"
        if ab.digest(path) != row["scene"]["artifacts"][role]["sha256"]:
            raise ValueError("frozen double-talk input changed")
        arrays[role] = ab.read_mono(path)
    microphone, playback, clean = [arrays[k] for k in ("mic", "playback", "clean")]
    if any(v.shape != (192000,) for v in arrays.values()) or prior["inputs"]["folder"] != str(source):
        raise ValueError("matched full 12 second input chain not proven")
    mask = paired.measurement_mask(row, clean, playback)
    is_valid = row["measurement_status"] == "measured"
    if is_valid != (np.count_nonzero(mask) >= ab.RATE // 20):
        raise ValueError("frozen joint-activity validity mismatch")
    design = (8, 11) if row["scene"]["source_category"] == "transition" else (6.5, 7) if not is_valid else (0, 12)
    candidates, execution = {}, {}
    for route, backend in (("speex_mdf_coherence", "speex"), ("webrtc_aec3", "webrtc")):
        prefix = work / route
        command = [str(ab.CLI), "--profile", "vcp", "--aec-backend", backend,
                   "--no-bf", "--no-ns", "--no-agc", "--no-tx-drc", "--no-limiter",
                   "--mic", str(source / "mic.wav"), "--echo", str(source / "playback.wav"), "--out", str(prefix)]
        report, _ = ab.call_cli(command, work / f"{route}.log")
        (speex.assert_identity if backend == "speex" else four.assert_identity)(report)
        delay = report.get("aec_output_delay_samples")
        if delay != (0 if backend == "speex" else 128):
            raise ValueError("route-declared AEC tap delay changed")
        tap_path = work / f"{route}.aec.wav"
        tap = ab.read_mono(tap_path)
        if len(tap) != len(microphone) + delay or sf.info(tap_path).subtype != "PCM_16":
            raise ValueError("native AEC tap incomplete or precision changed")
        candidates[route] = tap[delay:delay + len(microphone)]
        execution[route] = {"kind": "actual_native_execution", "command": command, "report": report,
                            "report_sha256": ab.digest(work / f"{route}.json"), "aec_output_delay_samples": delay}
    paired.assert_fused_identity(prior["fused_report"])
    for route in ("speex_fused_cpu", "localvqe_full", "mdf_residual_mask"):
        record = prior["routes"][route]["artifact"]
        path = previous / record["path"]
        samples = ab.read_mono(path)
        if ab.digest(path) != record["sha256"] or ab.pcm_digest(samples) != record["pcm_sha256"] or len(samples) != len(clean):
            raise ValueError("previous route PCM identity mismatch")
        candidates[route] = samples
        execution[route] = {"kind": "hash_verified_frozen_output", "source": str(path), "sha256": record["sha256"],
                            "pcm_sha256": record["pcm_sha256"], "alignment": "already aligned by frozen declared delay; no fitted lag"}
    quantized_mic = pcm16(microphone)
    outputs = {}
    for route, samples in {"input_mic": quantized_mic, **candidates}.items():
        aligned = pcm16(samples)
        metrics = None
        if is_valid:
            metrics = {"joint_active": ab.measure(aligned, quantized_mic, clean, mask, near_present=True),
                       "continuous_full": continuous_metrics(aligned, quantized_mic, clean, 0, 12),
                       "continuous_dt_design": continuous_metrics(aligned, quantized_mic, clean, *design)}
        outputs[route] = {"artifact": ab.write_audio(work / f"{route}.common-pcm16.wav", aligned), "metrics": metrics}
    result = {"case_id": row["case_id"], "source_category": row["scene"]["source_category"],
              "measurement_status": row["measurement_status"], "measurement_reason": row.get("measurement_reason"),
              "sample_rate": ab.RATE, "samples": len(clean), "activity": row["activity"],
              "joint_mask_samples": int(np.count_nonzero(mask)), "continuous_design_window_s": list(design),
              "inputs": {k: {"path": str(source / f"{k}.wav"), "sha256": ab.digest(source / f"{k}.wav")} for k in arrays},
              "execution": execution, "routes": outputs}
    ab.write_json(work / "case.json", result)
    return result


def summarize(rows: list[dict]) -> dict:
    valid = [r for r in rows if r["measurement_status"] == "measured"]
    result = {}
    for route in ("input_mic", *ROUTES):
        metrics = [r["routes"][route]["metrics"] for r in valid]
        result[route] = {"measured_case_count": len(valid)}
        for name, values in (
                ("joint_si_sdr_db", [m["joint_active"]["near_si_sdr_db"] for m in metrics]),
                ("joint_target_gain_db", [m["joint_active"]["near_target_gain_db"] for m in metrics]),
                ("joint_error_rms", [m["joint_active"]["near_total_error_rms"] for m in metrics]),
                ("continuous_dt_stoi", [m["continuous_dt_design"]["stoi"]["value"] for m in metrics]),
                ("continuous_full_stoi", [m["continuous_full"]["stoi"]["value"] for m in metrics])):
            result[route][name] = float(np.mean(values)) if all(v is not None for v in values) else None
        result[route]["retention_probe_cases"] = sum(m["joint_active"]["near_retention_probe"] for m in metrics)
    for route in result:
        result[route]["joint_si_sdr_delta_db"] = result[route]["joint_si_sdr_db"] - result["input_mic"]["joint_si_sdr_db"]
        result[route]["continuous_dt_stoi_delta"] = (result[route]["continuous_dt_stoi"] - result["input_mic"]["continuous_dt_stoi"]
                                                   if result[route]["continuous_dt_stoi"] is not None else None)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new empty evidence directory")
    args = parser.parse_args()
    ab.resource_check()
    scorer_controls = controls()
    assets = {paired.BASELINE: paired.BASELINE_SHA, PAIRED: PAIRED_SHA, ERLE: ERLE_SHA, ab.CLI: ab.CLI_SHA}
    for path, expected in assets.items():
        if ab.digest(path) != expected:
            raise ValueError(f"frozen asset changed: {path}")
    baseline, prior, echo_report = [json.loads(p.read_text()) for p in (paired.BASELINE, PAIRED, ERLE)]
    selected = [r for r in baseline["cases"] if r["category"] == "double_talk"]
    if len(selected) != 5:
        raise ValueError("expected original five double-talk scenes")
    previous = {r["case_id"]: r for r in prior["cases"] if r["category"] == "double_talk"}
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("preserve previous evidence; choose a new empty directory")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in selected:
        rows.append(run_case(row, previous[row["case_id"]], output))
        print(json.dumps({"finished": len(rows), "total": 5, "case": row["case_id"], "status": row["measurement_status"]}), flush=True)
    for path, expected in assets.items():
        if ab.digest(path) != expected:
            raise ValueError("frozen asset changed during execution")
    result = {"schema": "itu-five-route-double-talk-fidelity-v1", "created_at": datetime.now(timezone.utc).isoformat(),
              "identity": {"runner_sha256": ab.digest(Path(__file__)), "assets": {str(p): v for p, v in assets.items()},
                           "pystoi_version": importlib.metadata.version("pystoi")},
              "case_count": len(rows), "measured_case_count": sum(r["measurement_status"] == "measured" for r in rows),
              "case_duration_s": 12, "sample_rate": ab.RATE, "cases": rows, "summary": summarize(rows),
              "echo_only_erle_categories": echo_report["categories"], "scorer_controls": scorer_controls,
              "fitted_delay_for_scoring": False, "formal_compliance": False,
              "boundary": "five frozen synthetic double-talk cases, four valid; all route outputs PCM16; joint-active SI-SDR/gain and original contiguous DT design-window STOI separately; invalid short overlap preserved but excluded; no formal compliance, listening or CPU/hardware claim"}
    ab.write_json(output / "result.json", result)
    ab.write_json(output / "artifacts.json", {"files": {str(p.relative_to(output)): {"sha256": ab.digest(p), "bytes": p.stat().st_size}
        for p in sorted(output.rglob("*")) if p.is_file()}})
    print(json.dumps({"result": str(output / "result.json"), "sha256": ab.digest(output / "result.json")}), flush=True)


if __name__ == "__main__":
    main()
