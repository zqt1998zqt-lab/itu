#!/usr/bin/env python3
"""Echo-only ERLE for the frozen Speex fused / LocalVQE comparison.

This is a new isolation experiment: preserve post-RX playback and its rendered
echo, remove near speech and independent noise, and reset only at case start.
Never reinterpret the previous noisy output attenuation as pure echo ERLE.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("paired_erle_support", ROOT / "scripts/compare_speex_fused_localvqe.py")
assert SPEC and SPEC.loader
paired = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(paired)
ab = paired.ab
ROUTES = ("speex_fused_cpu", "localvqe_full", "mdf_residual_mask")
WINDOWS = {"whole_0_12s": (0, 12), "steady_8_12s": (8, 12),
           "before_switch": (3, 5.5), "after_switch_0_1s": (6, 7), "after_switch_2_6s": (8, 12)}


def erle(echo: np.ndarray, residual: np.ndarray, start_s: float, end_s: float) -> dict:
    if echo.ndim != 1 or residual.shape != echo.shape or not np.isfinite(echo).all() or not np.isfinite(residual).all():
        raise ValueError("ERLE requires finite, aligned mono input/output with identical length")
    start, stop = round(start_s * ab.RATE), round(end_s * ab.RATE)
    if not 0 <= start < stop <= len(echo):
        raise ValueError("ERLE window outside complete input timeline")
    source, out = echo[start:stop].astype(np.float64), residual[start:stop].astype(np.float64)
    source_power, residual_power = float(np.mean(source * source)), float(np.mean(out * out))
    if source_power <= 1e-20:
        raise ValueError("no measurable echo excitation in ERLE window")
    return {"start_s": start_s, "end_s": end_s, "samples": stop - start,
            "input_echo_power": source_power, "residual_power": residual_power,
            "erle_db": float(10 * np.log10(source_power / residual_power)) if residual_power > 0 else None,
            "status": "measured" if residual_power > 0 else "zero_output_no_finite_erle"}


def controls() -> dict:
    x = np.linspace(-.2, .2, 16000, dtype=np.float32)
    bypass = erle(x, x, 0, 1)["erle_db"]
    attenuated = erle(x, x * .1, 0, 1)["erle_db"]
    zero = erle(x, np.zeros_like(x), 0, 1)
    rejected = 0
    for a, b, lo, hi in ((x, x[:-1], 0, 1), (x, x, 0, 2), (np.zeros_like(x), x, 0, 1)):
        try:
            erle(a, b, lo, hi)
        except ValueError:
            rejected += 1
    if bypass != 0 or abs(attenuated - 20) > 1e-5 or zero["erle_db"] is not None or rejected != 3:
        raise ValueError("ERLE scorer negative controls failed")
    return {"bypass_erle_db": bypass, "known_20db_gain_erle_db": attenuated,
            "zero_output_not_given_finite_score": True, "bad_inputs_rejected": rejected, "passed": True}


def case(row: dict, output: Path, engine) -> dict:
    ab.resource_check()
    source = paired.BASELINE.parent / row["category"] / row["case_id"]
    work = output / row["category"] / row["case_id"]
    work.mkdir(parents=True, exist_ok=False)
    samples = {}
    for role in ("echo", "playback", "clean", "noise"):
        path = source / f"{role}.wav"
        if ab.digest(path) != row["scene"]["artifacts"][role]["sha256"]:
            raise ValueError(f"changed frozen source: {path}")
        samples[role] = ab.read_mono(path)
    echo, playback = samples["echo"], samples["playback"]
    if echo.shape != (12 * ab.RATE,) or any(v.shape != echo.shape for v in samples.values()) or np.any(samples["clean"]):
        raise ValueError("echo-only source requires full 12 s mono and originally silent near end")
    if ab.pcm_digest(playback) != row["scene"]["rx_speaker_pcm_sha256"]:
        raise ValueError("reference is not the previously measured post-RX playback")
    rebuilt = np.asarray(ab.reconstruct_echo(playback, row["scene"]["echo_path_segments"], row["scene"]["effective_echo_gain"]), dtype=np.float32)
    reconstruction_error = float(np.max(np.abs(rebuilt.astype(np.float64) - echo)))
    if reconstruction_error > 1e-8:
        raise ValueError("echo PCM does not match declared post-RX FIR path")
    if row["category"] == "path_change" and ab.verify_path_coverage(len(echo), row["scene"]["echo_path_segments"]) != [6.0]:
        raise ValueError("expected actual six-second path switch")
    mic_record = ab.write_audio(work / "mic.wav", echo)
    reference_record = ab.write_audio(work / "playback.wav", playback)
    command = [str(ab.CLI), "--profile", "vcp", "--aec-backend", "speex_fused_cpu", "--no-bf",
               "--no-ns", "--no-agc", "--no-tx-drc", "--no-limiter", "--mic", str(work / "mic.wav"),
               "--echo", str(work / "playback.wav"), "--out", str(work / "fused")]
    report, wall_ms = ab.call_cli(command, work / "fused.log")
    paired.assert_fused_identity(report)
    delay = report["aec_output_delay_samples"]
    if not isinstance(delay, int) or isinstance(delay, bool) or delay < 0:
        raise ValueError("invalid AEC output tap delay")
    tap = ab.read_mono(work / "fused.aec.wav")
    if len(tap) != len(echo) + delay:
        raise ValueError("AEC tap not complete")
    mdf, mdf_record = ab.mdf_inputs({"mic": echo}, work)
    routes = {}
    candidates = {"speex_fused_cpu": (tap[delay:delay + len(echo)], {"aec_tap_delay_samples": delay, "invocation_wall_ms": wall_ms})}
    for name, first, second, mode in (("localvqe_full", echo, playback, "full"),
                                     ("mdf_residual_mask", mdf["e"], mdf["yhat"], "mask")):
        aligned, raw, timing = engine.run(first, second, mode=mode)
        ab.write_audio(work / f"{name}.raw.wav", raw)
        candidates[name] = aligned, timing
    for name, (audio, timing) in candidates.items():
        routes[name] = {"artifact": ab.write_audio(work / f"{name}.aligned.wav", audio), "timing": timing,
                        "windows": {label: erle(echo, audio, *window) for label, window in WINDOWS.items()}}
    result = {"case_id": row["case_id"], "category": row["category"], "samples": len(echo),
              "input": {"kind": "pure_rendered_echo; independent noise and near speech removed", "mic": mic_record,
                        "reference": reference_record, "source_folder": str(source), "echo_pcm_sha256": ab.pcm_digest(echo),
                        "post_rx_pcm_sha256": ab.pcm_digest(playback), "near_power": 0.0, "noise_power": 0.0,
                        "removed_noise_power": ab.power(samples["noise"]), "echo_fir_max_error": reconstruction_error,
                        "path_switch_times_s": row["scene"]["path_change_times_s"]},
              "fused_command": command, "fused_report": report, "mdf": mdf_record, "routes": routes,
              "bypass_erle_db": erle(echo, echo, 0, 12)["erle_db"]}
    ab.write_json(work / "case.json", result)
    return result


def aggregates(rows: list[dict]) -> dict:
    output = {}
    for category in ("far_single", "path_change"):
        cases = [r for r in rows if r["category"] == category]
        output[category] = {"case_count": len(cases), "routes": {}}
        for route in ROUTES:
            windows = {}
            for window in WINDOWS:
                records = [r["routes"][route]["windows"][window] for r in cases]
                if any(r["status"] != "measured" for r in records):
                    raise ValueError("cannot average censored/zero-output ERLE as finite results")
                windows[window] = {"mean_erle_db": float(np.mean([r["erle_db"] for r in records])),
                                   "min_erle_db": min(r["erle_db"] for r in records),
                                   "max_erle_db": max(r["erle_db"] for r in records)}
            output[category]["routes"][route] = windows
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new empty directory; existing evidence is preserved")
    args = parser.parse_args()
    ab.resource_check()
    scorer_controls = controls()
    if ab.digest(paired.BASELINE) != paired.BASELINE_SHA or ab.digest(ab.CLI) != ab.CLI_SHA or ab.digest(ab.MODEL) != ab.MODEL_SHA:
        raise ValueError("frozen result/CLI/model identity changed")
    baseline = json.loads(paired.BASELINE.read_text())
    library = Path(baseline["identity"]["library"]["path"])
    if ab.digest(library) != baseline["identity"]["library"]["sha256"]:
        raise ValueError("frozen LocalVQE bridge changed")
    selected = [row for row in baseline["cases"] if row["category"] in ("far_single", "path_change")]
    if len(selected) != 10 or any(sum(r["category"] == group for r in selected) != 5 for group in ("far_single", "path_change")):
        raise ValueError("expected five far-only and five path-change scenes")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("preserve previous evidence; choose a new empty directory")
    output.mkdir(parents=True, exist_ok=True)
    engine = ab.Engine(library)
    rows = []
    try:
        for row in selected:
            rows.append(case(row, output, engine))
            print(json.dumps({"finished": len(rows), "total": len(selected), "case": row["case_id"]}), flush=True)
    finally:
        engine.close()
    for path, expected in ((paired.BASELINE, paired.BASELINE_SHA), (ab.CLI, ab.CLI_SHA), (ab.MODEL, ab.MODEL_SHA), (library, baseline["identity"]["library"]["sha256"])):
        if ab.digest(path) != expected:
            raise ValueError("asset changed during experiment")
    result = {"schema": "itu-echo-only-erle-v1", "created_at": datetime.now(timezone.utc).isoformat(),
              "identity": {"runner_sha256": ab.digest(Path(__file__)), "paired_runner_sha256": ab.digest(Path(paired.__file__)),
                           "measurement_runner_sha256": ab.digest(Path(ab.__file__)), "cli": {"path": str(ab.CLI), "sha256": ab.CLI_SHA},
                           "library": baseline["identity"]["library"], "model": baseline["identity"]["model"],
                           "frozen_source": {"path": str(paired.BASELINE), "sha256": paired.BASELINE_SHA}},
              "case_count": len(rows), "case_duration_s": 12, "sample_rate": ab.RATE, "cases": rows,
              "categories": aggregates(rows), "scorer_controls": scorer_controls,
              "definition": "10*log10(mean(pure_input_echo^2)/mean(aligned_output_residual^2)); unweighted full window including silence and convergence; per-case dB arithmetic group means",
              "boundary": "new echo-only isolation experiment, not the previous noisy run; post-RES terminal ERLE, not raw linear MDF ERLE; no near-end quality or formal compliance claim",
              "formal_compliance": False, "state_reset_at_path_switch": False, "fitted_delay_for_scoring": False}
    ab.write_json(output / "result.json", result)
    ab.write_json(output / "artifacts.json", {"files": {str(p.relative_to(output)): {"sha256": ab.digest(p), "bytes": p.stat().st_size}
        for p in sorted(output.rglob("*")) if p.is_file()}})
    print(json.dumps({"result": str(output / "result.json"), "sha256": ab.digest(output / "result.json"), "cases": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
