#!/usr/bin/env python3
"""Add actual WebRTC AEC3 to the frozen pure-echo, common-PCM16 ERLE set."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("echo_erle_base", ROOT / "scripts/compare_echo_only_erle.py")
assert SPEC and SPEC.loader
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)
ab = base.ab
PREVIOUS = ROOT / "build/echo-only-erle-20261112-wwcncyd6"
RESULT_SHA = "3e4cd187329cb19e9916ec5e1b095a4c15c446f929b50a6d0e00988737191044"
PCM_SHA = "32a6d64ee39fb2faa6fbea2660e6ef8cdf7e4403a8a26c2dcfe13827835a479c"
NAME = "webrtc_aec3"


def assert_identity(report: dict) -> None:
    expected = {"processing_profile": "vcp", "aec_backend_requested": "webrtc", "aec_backend": NAME,
                "aec_backend_effective": NAME, "aec_execution_provider": "cpu", "aec_enabled": True,
                "speex_coherence_residual_suppression": False, "ns_enabled": False, "agc2_enabled": False,
                "tx_drc_enabled": False, "microphones": 1, "reference_channels": 1}
    bad = {key: report.get(key) for key, value in expected.items() if report.get(key) != value}
    if bad:
        raise ValueError(f"WebRTC AEC3 identity/configuration mismatch: {bad}")


def common_windows(echo: np.ndarray, output: np.ndarray) -> dict:
    clipped = np.clip(output.astype(np.float64), -1, 32767 / 32768)
    quantized = np.sign(clipped) * np.floor(np.abs(clipped) * 32768 + .5) / 32768
    if not np.array_equal(quantized, output):
        raise ValueError("native AEC output is not on the required PCM16 grid")
    windows = {}
    for name, (start, stop) in base.WINDOWS.items():
        measured = base.erle(echo, output, start, stop)
        windows[name] = {"erle_db": measured["erle_db"], "zero_pcm_output": measured["erle_db"] is None,
                         "erle_lower_bound_db": float(10 * np.log10(measured["input_echo_power"] / (.5 / 32768)**2))
                         if measured["erle_db"] is None else None}
    return windows


def summarize(cases: list[dict]) -> dict:
    summary = {}
    for category in ("far_single", "path_change"):
        rows = [row for row in cases if row["category"] == category]
        summary[category] = {}
        for route in rows[0]["routes"]:
            windows = {}
            for name in base.WINDOWS:
                values = [row["routes"][route][name] for row in rows]
                finite = [v["erle_db"] for v in values if v["erle_db"] is not None]
                limits = [v["erle_db"] if v["erle_db"] is not None else v["erle_lower_bound_db"] for v in values]
                windows[name] = {"mean_erle_db": float(np.mean(finite)) if len(finite) == len(rows) else None,
                                 "measured_cases": len(finite), "censored_cases": len(rows) - len(finite),
                                 "mean_lower_bound_db": float(np.mean(limits)) if len(finite) < len(rows) else None}
            summary[category][route] = windows
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new empty output directory")
    args = parser.parse_args()
    ab.resource_check()
    paths = {PREVIOUS / "result.json": RESULT_SHA, PREVIOUS / "common-pcm16-erle.json": PCM_SHA, ab.CLI: ab.CLI_SHA}
    for path, expected in paths.items():
        if ab.digest(path) != expected:
            raise ValueError(f"frozen experiment asset changed: {path}")
    previous = json.loads((PREVIOUS / "result.json").read_text())
    common = json.loads((PREVIOUS / "common-pcm16-erle.json").read_text())
    if len(previous["cases"]) != 10 or len(common["cases"]) != 10:
        raise ValueError("expected the original ten pure-echo cases")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("preserve historical evidence; choose a new empty directory")
    output.mkdir(parents=True, exist_ok=True)
    cases = deepcopy(common["cases"])
    identities_rejected = 0
    for row, joined in zip(previous["cases"], cases, strict=True):
        ab.resource_check()
        if (row["case_id"], row["category"]) != (joined["case_id"], joined["category"]):
            raise ValueError("case identity/order mismatch")
        source = PREVIOUS / row["category"] / row["case_id"]
        for name, key in (("mic.wav", "mic"), ("playback.wav", "reference")):
            if ab.digest(source / name) != row["input"][key]["sha256"]:
                raise ValueError("pure echo input/reference changed")
        if row["input"]["near_power"] != 0 or row["input"]["noise_power"] != 0:
            raise ValueError("ERLE comparison requires pure echo inputs")
        echo = ab.read_mono(source / "mic.wav")
        if echo.shape != (192000,):
            raise ValueError("expected full 12 seconds at 16 kHz")
        work = output / row["category"] / row["case_id"]
        work.mkdir(parents=True, exist_ok=False)
        command = [str(ab.CLI), "--profile", "vcp", "--aec-backend", "webrtc", "--no-bf", "--no-ns",
                   "--no-agc", "--no-tx-drc", "--no-limiter", "--mic", str(source / "mic.wav"),
                   "--echo", str(source / "playback.wav"), "--out", str(work / "webrtc")]
        report, wall_ms = ab.call_cli(command, work / "webrtc.log")
        assert_identity(report)
        wrong = dict(report, aec_backend_effective="speex_mdf")
        try:
            assert_identity(wrong)
        except ValueError:
            identities_rejected += 1
        else:
            raise ValueError("wrong backend negative control accepted")
        delay = report.get("aec_output_delay_samples")
        if isinstance(delay, bool) or not isinstance(delay, int) or not 0 <= delay <= ab.RATE:
            raise ValueError("invalid declared AEC output delay")
        tap_path = work / "webrtc.aec.wav"
        tap = ab.read_mono(tap_path)
        if sf.info(tap_path).subtype != "PCM_16" or len(tap) != len(echo) + delay:
            raise ValueError("native tap length/precision is not proven")
        aligned = tap[delay:delay + len(echo)]
        joined["routes"][NAME] = common_windows(echo, aligned)
        joined["execution"] = {"command": command, "report": report, "report_sha256": ab.digest(work / "webrtc.json"),
                               "aec_output_delay_samples": delay, "invocation_wall_ms": wall_ms,
                               "input_folder": str(source), "input_sha256": row["input"]["mic"]["sha256"],
                               "reference_sha256": row["input"]["reference"]["sha256"],
                               "path_switch_times_s": row["input"]["path_switch_times_s"],
                               "artifact": ab.write_audio(work / "webrtc.aligned.wav", aligned)}
        ab.write_json(work / "case.json", joined)
        print(json.dumps({"finished": len([r for r in cases if "execution" in r]), "total": 10, "case": row["case_id"], "delay": delay}), flush=True)
    summary = summarize(cases)
    for group, routes in common["categories"].items():
        for route, values in routes.items():
            if summary[group][route] != values:
                raise ValueError("original three-route comparison was changed")
    for path, expected in paths.items():
        if ab.digest(path) != expected:
            raise ValueError("asset changed during execution")
    result = {"schema": "itu-four-route-echo-erle-v1", "created_at": datetime.now(timezone.utc).isoformat(),
              "case_count": len(cases), "case_duration_s": 12, "sample_rate": ab.RATE,
              "identity": {"runner_sha256": ab.digest(Path(__file__)), "base_runner_sha256": ab.digest(Path(base.__file__)),
                           "cli": {"path": str(ab.CLI), "sha256": ab.CLI_SHA},
                           "previous_result": {"path": str(PREVIOUS / "result.json"), "sha256": RESULT_SHA},
                           "previous_common_pcm16": {"path": str(PREVIOUS / "common-pcm16-erle.json"), "sha256": PCM_SHA}},
              "cases": cases, "categories": summary, "scorer_controls": base.controls(),
              "wrong_backend_negative_controls_rejected": identities_rejected,
              "state_reset_at_path_switch": False, "fitted_delay_for_scoring": False,
              "boundary": "ten identical pure-echo inputs; terminal ERLE including AEC residual suppression, common PCM16 output; original three routes reused with fixed identity; WebRTC actually executed with declared tap delay and flush; no formal compliance claim",
              "formal_compliance": False}
    ab.write_json(output / "result.json", result)
    ab.write_json(output / "artifacts.json", {"files": {str(p.relative_to(output)): {"sha256": ab.digest(p), "bytes": p.stat().st_size}
        for p in sorted(output.rglob("*")) if p.is_file()}})
    print(json.dumps({"result": str(output / "result.json"), "sha256": ab.digest(output / "result.json")}), flush=True)


if __name__ == "__main__":
    main()
