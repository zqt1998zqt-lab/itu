#!/usr/bin/env python3
"""Independent simulated playback clock, paired real-speech PCM, fixed scoring windows."""
import argparse
import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import run_audio_eval as ev


def simulate_echo(playback, rate, ppm):
    position = np.arange(len(playback), dtype=np.float64)
    ratio = 1 + ppm * 1e-6
    return sum(gain * np.interp((position - delay * rate) * ratio, position,
                               playback, left=0, right=0)
               for delay, gain in ((.080, .60), (.085, .18)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--pair", type=int, default=0)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--pace", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        raise ValueError("workers must be 1..6")
    manifest = json.loads((args.dataset / "inputs-manifest.json").read_text())
    candidates = [s for s in manifest["scenes"] if s["pair"] == args.pair and
                  s["mics"] == 1 and s["family"] == "aec"]
    if not candidates:
        raise ValueError("real far/near speech source missing")
    scene = next((s for s in candidates if s.get("playback_source") == "radio"), candidates[0])
    folder = args.dataset / "cases" / scene["id"]
    for name in ("ref.wav", "clean.wav"):
        if ev.digest(folder / name) != scene["files"][name]["sha256"]:
            raise ValueError("frozen source changed")
    # Real far-end speech continues for 90 s; near-end speech is independently scheduled.
    far = ev.read(folder / "ref.wav")[2*ev.RATE:10*ev.RATE, 0]
    near_source = ev.read(folder / "clean.wav")[12*ev.RATE:15*ev.RATE, 0]
    if ev.power(far) <= 1e-8 or ev.power(near_source) <= 1e-8:
        raise ValueError("nonzero real speech sources required")
    args.root.mkdir(parents=True, exist_ok=False)
    thresholds = dict(echo_attenuation_min_db=50, correction_improvement_min_db=20,
                      clock_error_max_ppm=4, fifo_errors_max=0, process_p95_max_ms=1)
    jobs = []
    for rate in (16000, 24000, 48000):
        divisor = math.gcd(rate, ev.RATE)
        original = np.resize(far, 90*ev.RATE)
        original_near = np.zeros_like(original)
        for start in (12, 40):
            original_near[start*ev.RATE:(start+8)*ev.RATE] = np.resize(near_source, 8*ev.RATE)
        playback = resample_poly(original, rate//divisor, ev.RATE//divisor)
        near = resample_poly(original_near, rate//divisor, ev.RATE//divisor)
        for ppm in (-300, 300):
            directory = args.root / f"{rate}-{ppm:+d}ppm"
            directory.mkdir()
            capture = near + simulate_echo(playback, rate, ppm)
            if max(np.max(np.abs(capture)), np.max(np.abs(playback))) >= .98:
                raise ValueError("simulated capture would clip")
            sf.write(directory / "ref.wav", playback, rate, subtype="PCM_16")
            sf.write(directory / "before.wav", capture, rate, subtype="PCM_16")
            for correction in (False, True):
                jobs.append((rate, ppm, directory, correction))

    def run(job):
        rate, ppm, directory, correction = job
        prefix = directory / ("corrected" if correction else "uncorrected")
        command = [str(args.cli.resolve()), "--profile", "vehicle_stream",
                   "--mic", str(directory / "before.wav"), "--echo", str(directory / "ref.wav"),
                   "--out", str(prefix), "--simulate-clock-ppm", str(ppm)]
        if not correction:
            command.append("--disable-clock-correction")
        if args.pace:
            command.append("--pace-10ms")
        result = subprocess.run(command, capture_output=True, text=True, timeout=240)
        prefix.with_suffix(".invocation.json").write_text(json.dumps(dict(
            argv=command, exit_code=result.returncode, stderr=result.stderr), indent=2)+"\n")
        result.check_returncode()
        engine = json.loads(prefix.with_suffix(".json").read_text())
        output, actual_rate = sf.read(prefix.with_suffix(".after.wav"))
        before, _ = sf.read(directory / "before.wav")
        if actual_rate != rate or len(output) != len(before)+engine["dsp_delay_samples"]:
            raise ValueError("simulated-clock output contract changed")
        output = output[engine["dsp_delay_samples"]:]
        attenuation = ev.db(ev.power(before[70*rate:85*rate]) /
                            max(ev.power(output[70*rate:85*rate]), ev.EPS))
        return dict(sample_rate=rate, injected_ppm=ppm, correction=correction,
                    echo_attenuation_db=attenuation, engine=engine,
                    mic_sha256=ev.digest(directory / "before.wav"),
                    ref_sha256=ev.digest(directory / "ref.wav"),
                    output_sha256=ev.digest(prefix.with_suffix(".after.wav")))

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(run, jobs))
    failures = []
    for row in rows:
        if not row["correction"]:
            continue
        negative = next(r for r in rows if not r["correction"] and
                        (r["sample_rate"], r["injected_ppm"]) == (row["sample_rate"], row["injected_ppm"]))
        if any(row[name] != negative[name] for name in ("mic_sha256", "ref_sha256")):
            raise ValueError("paired clock inputs changed")
        row["improvement_db"] = row["echo_attenuation_db"] - negative["echo_attenuation_db"]
        engine = row["engine"]
        row["passed"] = (engine["clock_locked"] and
                         abs(engine["clock_estimated_ppm"]-row["injected_ppm"]) < thresholds["clock_error_max_ppm"] and
                         engine["clock_underruns"]+engine["clock_overruns"] == 0 and
                         row["echo_attenuation_db"] >= thresholds["echo_attenuation_min_db"] and
                         row["improvement_db"] >= thresholds["correction_improvement_min_db"] and
                         engine["process_p95_ms"] < thresholds["process_p95_max_ms"])
        if not row["passed"]:
            failures.append(dict(sample_rate=row["sample_rate"], ppm=row["injected_ppm"]))
    report = dict(scope="simulated independent clock with real playback speech; no hardware or original highband certification",
                  source_scene=scene["id"], source_manifest_sha256=ev.digest(args.dataset / "inputs-manifest.json"),
                  cli_sha256=ev.digest(args.cli), thresholds=thresholds,
                  scoring_window_s=[70, 85], input_seconds=90, rows=rows, failures=failures,
                  passed=not failures, final_acceptance_eligible=False)
    (args.root / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(passed=report["passed"], failures=failures,
                         corrected=[{k: r[k] for k in ("sample_rate", "injected_ppm", "echo_attenuation_db", "improvement_db")}
                                    for r in rows if r["correction"]]), indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
