#!/usr/bin/env python3
"""Screen the causal speech-band protection against frozen OM-LSA outputs."""
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


def key(row):
    return row["run"], row["id"]


def token(row):
    return hashlib.sha256((row["run"] + "/" + row["id"]).encode()).hexdigest()[:20]


def selection(rows, scenes, scope):
    if scope == "full":
        return rows
    chosen = {}
    for row in rows:
        key = row["run"], row["id"]
        if row["run"] == "controlled-cavemove-20261017" and any(
                failure.endswith(":projection_gain") for failure in row["failures"]):
            chosen[key] = row
        elif row["passed"] and row["run"] == "controlled-native-full-20261017" and row["snr_db"] <= -9:
            chosen[key] = row
        elif row["run"] == "controlled-native-full-20261017" and row["noise_class"] == "transient":
            chosen[key] = row
        elif row["passed"] and row["id"] in {
                "coverage-16000-development-84118526-snr+10",
                "coverage-16000-development-84118526-snr+30",
                "sweep-16000-development-1a2ad3b0-tcar-snr+10"}:
            chosen[key] = row
    return list(chosen.values())


def execute(task):
    root, binary, recovery, baseline, scene = task
    folder = root / ("recovery-" + str(recovery).replace(".", "p")) / token(baseline)
    folder.mkdir(parents=True, exist_ok=False)
    original = json.loads((FROZEN / PROFILE / token(baseline) / "invocation.json").read_text())["argv"]
    command = list(original)
    command[0], command[4] = str(binary), str(folder / "result")
    command.append("--speech-band-recovery=" + str(recovery))
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
               delay_samples=delay, engine=engine, **scored)
    save(folder / "metrics.json", row)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--recovery", type=float, nargs="+", default=[0, 1])
    parser.add_argument("--scope", choices=("screen", "full"), default="screen")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binary = args.frozen_binary.resolve()
    manifest = P / "build/ns-gtcrn-direct-20261018/full/identity.json"
    scenes = {(r["run"], r["scene"]["id"]): r["scene"]
              for r in json.loads(manifest.read_text())["selection"]}
    rows = [json.loads(line) for line in RESULTS.open()
            if json.loads(line)["profile"] == PROFILE]
    if len(rows) != 692 or len({(r["run"], r["id"]) for r in rows}) != 692:
        raise ValueError("incomplete frozen OM-LSA selection")
    selected = selection(rows, scenes, args.scope)
    selected.sort(key=lambda row: (row["run"], row["id"]))
    bindings = {str(p): sha(p) for p in (binary, RESULTS, IDENTITY, manifest,
                Path(__file__), Path(__file__).with_name("evaluate_gtcrn_direct.py"))}
    save(out / "identity.json", dict(bindings=bindings, binary_sha256=sha(binary),
         selection=[dict(run=r["run"], id=r["id"], baseline_passed=r["passed"])
                    for r in selected],
         recoveries=args.recovery, thresholds_changed=False,
         no_fitted_alignment=True, scope=args.scope,
         retrospective_screen=args.scope == "screen"))
    tasks = [(out, binary, recovery, row, scenes[key(row)])
             for recovery in args.recovery for row in selected]
    results = []
    with ProcessPoolExecutor(args.jobs) as pool, (out / "results.jsonl").open("x") as stream:
        for index, future in enumerate(as_completed([pool.submit(execute, task) for task in tasks]), 1):
            row = future.result()
            results.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            if index % 25 == 0 or index == len(tasks):
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
