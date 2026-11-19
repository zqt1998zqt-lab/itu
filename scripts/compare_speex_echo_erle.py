#!/usr/bin/env python3
"""Add VCP Speex MDF + Coherence RES to the frozen four-route ERLE set."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("four_route_erle", ROOT / "scripts/compare_webrtc_echo_erle.py")
assert SPEC and SPEC.loader
four = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(four)
ab = four.ab
PREVIOUS = ROOT / "build/webrtc-echo-erle-20261112-raptf3dm"
PREVIOUS_SHA = "2b7aedf52c95af344e4f138f204388f3a7d28ec2f4b10b53263e23f0497a0ddc"
NAME = "speex_mdf_coherence"


def assert_identity(report: dict) -> None:
    expected = {"processing_profile": "vcp", "aec_backend_requested": "speex", "aec_backend": "speex_mdf",
                "aec_backend_effective": "speex_mdf", "aec_execution_provider": "cpu", "aec_device_id": "host_cpu",
                "aec_enabled": True, "speex_echo_tail_ms": 500, "speex_native_conditioning": True,
                "speex_post_context": True, "speex_coherence_residual_suppression": True,
                "speex_residual_suppression": False, "ns_enabled": False, "agc2_enabled": False,
                "tx_drc_enabled": False, "microphones": 1, "reference_channels": 1}
    bad = {key: report.get(key) for key, value in expected.items() if report.get(key) != value}
    if bad:
        raise ValueError(f"ordinary Speex VCP route/configuration not proven: {bad}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new empty directory; historical evidence is preserved")
    args = parser.parse_args()
    ab.resource_check()
    assets = {PREVIOUS / "result.json": PREVIOUS_SHA, ab.CLI: ab.CLI_SHA}
    for path, expected in assets.items():
        if ab.digest(path) != expected:
            raise ValueError(f"frozen comparison identity changed: {path}")
    previous = json.loads((PREVIOUS / "result.json").read_text())
    if len(previous["cases"]) != 10:
        raise ValueError("expected ten frozen pure-echo scenes")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use a new empty directory; never overwrite previous evidence")
    output.mkdir(parents=True, exist_ok=True)
    cases = deepcopy(previous["cases"])
    rejections = 0
    for index, row in enumerate(cases):
        ab.resource_check()
        prior = row["execution"]
        source = Path(prior["input_folder"])
        if ab.digest(source / "mic.wav") != prior["input_sha256"] or ab.digest(source / "playback.wav") != prior["reference_sha256"]:
            raise ValueError("same pure-echo/post-RX input identity not proven")
        echo = ab.read_mono(source / "mic.wav")
        if echo.shape != (192000,):
            raise ValueError("full 12 s / 16 kHz mono required")
        work = output / row["category"] / row["case_id"]
        work.mkdir(parents=True, exist_ok=False)
        command = [str(ab.CLI), "--profile", "vcp", "--aec-backend", "speex", "--speex-coherence-residual",
                   "--no-bf", "--no-ns", "--no-agc", "--no-tx-drc", "--no-limiter",
                   "--mic", str(source / "mic.wav"), "--echo", str(source / "playback.wav"), "--out", str(work / "speex")]
        report, wall_ms = ab.call_cli(command, work / "speex.log")
        assert_identity(report)
        for field, bad_value in (("aec_backend_effective", "speex_mdf_nlms_fused_cpu"), ("speex_coherence_residual_suppression", False),
                                 ("ns_enabled", True), ("agc2_enabled", True), ("speex_echo_tail_ms", 200)):
            bad = dict(report)
            bad[field] = bad_value
            try:
                assert_identity(bad)
            except ValueError:
                rejections += 1
            else:
                raise ValueError("incorrect route/configuration negative control accepted")
        delay = report.get("aec_output_delay_samples")
        if isinstance(delay, bool) or not isinstance(delay, int) or not 0 <= delay <= ab.RATE:
            raise ValueError("invalid declared AEC output delay")
        tap_path = work / "speex.aec.wav"
        tap = ab.read_mono(tap_path)
        if sf.info(tap_path).subtype != "PCM_16" or len(tap) != len(echo) + delay:
            raise ValueError("native AEC tap precision/complete timeline not proven")
        aligned = tap[delay:delay + len(echo)]
        row["routes"][NAME] = four.common_windows(echo, aligned)
        capture_delay, linear_delay = report.get("aec_input_delay_samples"), report.get("linear_aec_delay_samples")
        if capture_delay != 0 or linear_delay != 0:
            raise ValueError("16 kHz conditioned/raw MDF diagnostic delay changed")
        capture, linear = ab.read_mono(work / "speex.aec-input.wav"), ab.read_mono(work / "speex.linear-aec.wav")
        if capture.shape != echo.shape or linear.shape != echo.shape:
            raise ValueError("raw MDF diagnostic taps incomplete")
        row["speex_execution"] = {"command": command, "report": report, "report_sha256": ab.digest(work / "speex.json"),
                                  "input_folder": str(source), "input_sha256": prior["input_sha256"],
                                  "reference_sha256": prior["reference_sha256"], "aec_output_delay_samples": delay,
                                  "invocation_wall_ms": wall_ms, "path_switch_times_s": prior["path_switch_times_s"],
                                  "artifact": ab.write_audio(work / "speex.aligned.wav", aligned),
                                  "linear_mdf_diagnostic": {"definition": "conditioned PCM16 capture / raw MDF error; excludes RES/dry protection",
                                      "windows": four.common_windows(capture, linear)}}
        ab.write_json(work / "case.json", row)
        print(json.dumps({"finished": index + 1, "total": 10, "case": row["case_id"], "delay": delay}), flush=True)
    summary = four.summarize(cases)
    for group, routes in previous["categories"].items():
        for route, values in routes.items():
            if summary[group][route] != values:
                raise ValueError("previous four-route result was changed")
    for path, expected in assets.items():
        if ab.digest(path) != expected:
            raise ValueError("asset changed during execution")
    result = {"schema": "itu-five-route-echo-erle-v1", "created_at": datetime.now(timezone.utc).isoformat(),
              "case_count": 10, "case_duration_s": 12, "sample_rate": ab.RATE,
              "identity": {"runner_sha256": ab.digest(Path(__file__)), "four_route_runner_sha256": ab.digest(Path(four.__file__)),
                           "cli": {"path": str(ab.CLI), "sha256": ab.CLI_SHA},
                           "previous_result": {"path": str(PREVIOUS / "result.json"), "sha256": PREVIOUS_SHA}},
              "cases": cases, "categories": summary, "scorer_controls": four.base.controls(),
              "configuration_negative_controls_rejected": rejections,
              "state_reset_at_path_switch": False, "fitted_delay_for_scoring": False,
              "boundary": "same ten pure-echo inputs; ordinary VCP Speex MDF with Coherence RES enabled and no NLMS fusion; terminal ERLE with common PCM16 precision; separate conditioned/raw linear MDF diagnostic; previous four routes preserved with fixed identity",
              "formal_compliance": False}
    ab.write_json(output / "result.json", result)
    ab.write_json(output / "artifacts.json", {"files": {str(p.relative_to(output)): {"sha256": ab.digest(p), "bytes": p.stat().st_size}
        for p in sorted(output.rglob("*")) if p.is_file()}})
    print(json.dumps({"result": str(output / "result.json"), "sha256": ab.digest(output / "result.json")}), flush=True)


if __name__ == "__main__":
    main()
