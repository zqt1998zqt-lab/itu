#!/usr/bin/env python3
"""Run the full speech-band regression with an immutable result schema."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import soundfile as sf

from evaluate_gtcrn_direct import score_aligned

P = Path(__file__).resolve().parents[2]
FROZEN = P / "build/ns-rumble-adaptive-20261019/full-r4"
RESULTS = FROZEN / "results.jsonl"
IDENTITY = FROZEN / "identity.json"
PROFILE = "omlsa_seq16_rumbleadaptive"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def token(row):
    return hashlib.sha256((row["run"] + "/" + row["id"]).encode()).hexdigest()[:20]


def execute(task):
    root, binary, recovery, transient_dry_guard, spectral_guard_db, baseline, scene = task
    folder = root / ("recovery-" + str(recovery).replace(".", "p")) / token(baseline)
    folder.mkdir(parents=True, exist_ok=False)
    original = json.loads((FROZEN / PROFILE / token(baseline) / "invocation.json").read_text())["argv"]
    command = list(original)
    command[0], command[4] = str(binary), str(folder / "result")
    command.append("--speech-band-recovery=" + str(recovery))
    if recovery > 0:
        command.extend([
            "--transient-dry-guard=" + str(transient_dry_guard),
            "--transient-spectral-guard-db=" + str(spectral_guard_db),
        ])
    proc = subprocess.run(command, capture_output=True, text=True)
    save(folder / "invocation.json", dict(argv=command, returncode=proc.returncode, stderr=proc.stderr))
    proc.check_returncode()
    origin = P / "build" / baseline["run"] / "cases" / baseline["id"]
    clean, clean_rate = sf.read(origin / "clean.wav")
    before, before_rate = sf.read(origin / "before.wav")
    before = np.clip(np.rint(before * 32768), -32768, 32767) / 32768
    engine = json.loads((folder / "result.json").read_text())
    audio, rate = sf.read(folder / "result.after.wav")
    delay = engine["delay_samples"]
    if (clean_rate != before_rate or rate != 16000 or engine["core_rate"] != 16000
            or engine["src_delay_samples"] != 0
            or delay != baseline["engine"]["delay_samples"]
            or len(audio) != len(clean) + delay or not np.isfinite(audio).all()):
        raise ValueError("candidate changed native rate, timebase, or finite output")
    output_sha = sha(folder / "result.after.wav")
    if recovery == 0 and output_sha != baseline["output_sha256"]:
        raise ValueError("disabled protection is not sample-identical to frozen baseline")
    scored = score_aligned(clean, before, audio[delay:], scene, "full",
                           engine["clipped_samples_before_write"])
    row = dict(recovery=recovery, run=baseline["run"], id=baseline["id"],
               noise=baseline["noise"], noise_class=baseline["noise_class"],
               snr_db=baseline["snr_db"], split=scene["split"],
               baseline_passed=baseline["passed"], baseline_failures=baseline["failures"],
               output_sha256=output_sha,
               output_changed=output_sha != baseline["output_sha256"],
               speech_band_frames=engine["speech_band_frames"],
               speech_band_floor=engine["speech_band_floor"],
               speech_band_gain_delta=engine["speech_band_gain_delta"],
               transient_dry_guard=engine["transient_dry_guard"],
               transient_spectral_guard_db=engine["transient_spectral_guard_db"],
               transient_spectral_guard_applied=(
                   engine["transient_spectral_guard_applied"]),
               delay_samples=delay, engine=engine, **scored)
    save(folder / "metrics.json", row)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--recovery", type=float, nargs="+", default=[0, 1])
    parser.add_argument("--transient-dry-guard", type=float, default=1)
    parser.add_argument("--transient-spectral-guard-db", type=float, default=18)
    args = parser.parse_args()
    if not 0 <= args.transient_dry_guard <= 1:
        raise ValueError("transient dry guard must be in [0, 1]")
    if not 0 <= args.transient_spectral_guard_db <= 24:
        raise ValueError("transient spectral guard must be in [0, 24] dB")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binary = args.frozen_binary.resolve()
    manifest = P / "build/ns-gtcrn-direct-20261018/full/identity.json"
    scenes = {(r["run"], r["scene"]["id"]): r["scene"]
              for r in json.loads(manifest.read_text())["selection"]}
    rows = [json.loads(line) for line in RESULTS.open()
            if json.loads(line)["profile"] == PROFILE]
    rows.sort(key=lambda row: (row["run"], row["id"]))
    if len(rows) != 692 or len({(r["run"], r["id"]) for r in rows}) != 692:
        raise ValueError("incomplete frozen OM-LSA selection")
    bindings = {str(p): sha(p) for p in (binary, RESULTS, IDENTITY, manifest,
                Path(__file__), Path(__file__).with_name("evaluate_gtcrn_direct.py"))}
    save(out / "identity.json", dict(bindings=bindings, binary_sha256=sha(binary),
         selection=[dict(run=r["run"], id=r["id"], baseline_passed=r["passed"])
                    for r in rows],
         recoveries=args.recovery, thresholds_changed=False,
         transient_dry_guard=args.transient_dry_guard,
         transient_spectral_guard_db=args.transient_spectral_guard_db,
         no_fitted_alignment=True, scope="full"))
    tasks = [(out, binary, recovery, args.transient_dry_guard,
              args.transient_spectral_guard_db, row,
              scenes[(row["run"], row["id"])])
             for recovery in args.recovery for row in rows]
    results = []
    with ProcessPoolExecutor(args.jobs) as pool, (out / "results.jsonl").open("x") as stream:
        for index, future in enumerate(as_completed([pool.submit(execute, task) for task in tasks]), 1):
            row = future.result()
            results.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            if index % 50 == 0 or index == len(tasks):
                save(out / "report.json", dict(state="running", completed=index, total=len(tasks)))
                print(index, "/", len(tasks), flush=True)
    changed = [path for path, digest in bindings.items() if sha(Path(path)) != digest]
    if changed:
        raise ValueError("experiment dependency changed during run")
    summary = {}
    for recovery in args.recovery:
        group = [r for r in results if r["recovery"] == recovery]
        summary[str(recovery)] = dict(
            total=len(group),
            baseline_passed=sum(r["baseline_passed"] for r in group),
            passed=sum(r["passed"] for r in group),
            recovered=sum(not r["baseline_passed"] and r["passed"] for r in group),
            regressed=sum(r["baseline_passed"] and not r["passed"] for r in group),
            changed_outputs=sum(r["output_changed"] for r in group),
            gate_failed_cases={gate: sum(
                any(failure.split(":")[-1] == gate for failure in r["failures"])
                for r in group)
                for gate in ("attenuation", "projection_gain", "raw_sisdr", "stoi", "clipping")})
    save(out / "summary.json", summary)
    save(out / "report.json", dict(state="completed", completed=len(results),
         total=len(tasks), dependencies_verified=True, thresholds_changed=False,
         realtime_validation=False))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
