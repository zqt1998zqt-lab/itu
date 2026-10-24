#!/usr/bin/env python3
"""Verify speech-band protection through the real general/VCP CLI path."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import soundfile as sf

P = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def token(run, scene):
    return hashlib.sha256((run + "/" + scene).encode()).hexdigest()[:20]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-root", type=Path, required=True)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-recoveries", type=int, default=8)
    parser.add_argument("--transient-dry-guard", type=float, default=1)
    parser.add_argument("--transient-spectral-guard-db", type=float, default=18)
    args = parser.parse_args()
    root, cli, out = args.quality_root.resolve(), args.cli.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(line) for line in (root / "results.jsonl").open()]
    chosen = []
    for row in rows:
        if not row["baseline_passed"] and row["passed"]:
            chosen.append(row)
    if len(chosen) != args.expected_recoveries:
        raise ValueError("recovered target scene count differs")
    cases = []
    for row in chosen:
        folder = out / token(row["run"], row["id"])
        folder.mkdir(parents=True, exist_ok=False)
        source = P / "build" / row["run"] / "cases" / row["id"] / "before.wav"
        x, rate = sf.read(source)
        x = x.astype(np.float32).astype(np.float64)
        q = np.copysign(np.floor(np.abs(x) * 32768 + .5), x)
        q = np.clip(q, -32768, 32767) / 32768
        entry = folder / "input-pcm16.wav"
        sf.write(entry, q, rate, subtype="PCM_16")
        component = (
            root / "recovery-1p0" / token(row["run"], row["id"])
            / "result.after.wav"
        )
        reference, reference_rate = sf.read(component)
        for profile in ("general", "vcp"):
            prefix = folder / profile
            command = [
                str(cli), "--profile", profile, "--no-aec", "--no-agc",
                "--no-tx-drc", "--no-limiter", "--no-low-cut", "--no-external-vad",
                "--vehicle-ns-control", "--mic", str(entry), "--out", str(prefix),
                "--ns-backend", "omlsa_imcra", "--ns-level", "0",
                "--vehicle-ns-mild-strength", "0",
                "--vehicle-ns-repair-mode", "4",
                "--vehicle-rumble-hz", "30", "--vehicle-ns-policy", "parallel_start",
                "--vehicle-ns-dry-max", "1", "--vehicle-ns-spectral-dry-guard", "1",
                "--vehicle-ns-rumble-recovery", "1",
                "--vehicle-ns-rumble-adaptive-hz", "60",
                "--vehicle-ns-weak-speech-recovery", "0",
                "--vehicle-ns-speech-band-recovery", "1",
                "--vehicle-ns-transient-dry-guard",
                str(args.transient_dry_guard),
                "--vehicle-ns-transient-spectral-guard-db",
                str(args.transient_spectral_guard_db),
                "--vehicle-ns-noise-step-recovery", "0.5",
                "--vehicle-ns-snr-threshold", "6", "--vehicle-ns-noise-gain", "1",
                "--vehicle-ns-startup-ms", "500", "--vehicle-ns-slew-ms", "50",
                "--vehicle-ns-release-ms", "0", "--vehicle-ns-quiet-floor-db", "-65",
                "--vehicle-ns-floor-guard", "1", "--vehicle-adaptive-rumble",
            ]
            proc = subprocess.run(command, capture_output=True, text=True)
            (folder / (profile + ".invocation.json")).write_text(
                json.dumps(dict(argv=command, returncode=proc.returncode,
                                stderr=proc.stderr), indent=2) + "\n")
            proc.check_returncode()
            audio, actual_rate = sf.read(prefix.with_suffix(".after.wav"))
            info = json.loads(prefix.with_suffix(".json").read_text())
            if actual_rate != reference_rate or audio.shape != reference.shape:
                raise ValueError("engine rate/shape differs from component")
            maximum = float(np.max(np.abs(audio - reference)))
            if maximum != 0 or info["dsp_delay_samples"] != 480:
                raise ValueError("engine output/timebase differs from component")
            if (info["vehicle_ns_speech_band_recovery"] != 1
                    or info["vehicle_ns_transient_dry_guard"]
                    != args.transient_dry_guard
                    or info["vehicle_ns_transient_spectral_guard_db"]
                    != args.transient_spectral_guard_db
                    or info["external_vad_enabled"]):
                raise ValueError("speech-band option did not reach engine or VAD enabled")
            with prefix.with_suffix(".frames.csv").open() as stream:
                if any(int(item["external_vad_calls"]) for item in csv.DictReader(stream)):
                    raise ValueError("external VAD was called")
            cases.append(dict(run=row["run"], id=row["id"], profile=profile,
                              rate=actual_rate, delay_samples=info["dsp_delay_samples"],
                              max_abs_difference=maximum, bitexact=True,
                              external_vad_calls=0))
    report = dict(state="completed", comparisons=cases, all_bitexact=True,
                  cli_sha256=sha(cli), native_rate=16000,
                  qualification_scope="isolated NS in actual CLI; not full duplex quality")
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("speech-band engine equivalence:", len(cases), "bitexact comparisons")


if __name__ == "__main__":
    main()
