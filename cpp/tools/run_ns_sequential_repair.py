#!/usr/bin/env python3
"""Paired sequential C++ NS repairs on the unchanged native-16-kHz manifest."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess

import numpy as np
import soundfile as sf

from diagnose_traditional_ns_remaining import BASE, BRANCHES, P, inputs, original_folder, save, sha, token
from evaluate_gtcrn_direct import score_aligned


def execute(task):
    root, binary, profile, baseline, scene = task
    folder = root / profile["name"] / token(baseline)
    folder.mkdir(parents=True, exist_ok=False)
    command = json.loads((original_folder(baseline) / "invocation.json").read_text())["argv"]
    command[0], command[4] = str(binary), str(folder / "result")
    command.extend(profile.get("options", []))
    proc = subprocess.run(command, capture_output=True, text=True)
    save(folder / "invocation.json", dict(argv=command, returncode=proc.returncode, stderr=proc.stderr))
    proc.check_returncode()
    before, clean = inputs(baseline, scene)
    engine = json.loads((folder / "result.json").read_text())
    audio, rate = sf.read(folder / "result.after.wav")
    delay = engine["delay_samples"]
    if (rate != 16000 or engine["core_rate"] != 16000 or engine["src_delay_samples"] != 0
            or delay != baseline["engine"]["delay_samples"]
            or len(audio) != len(clean) + delay or not np.isfinite(audio).all()):
        raise ValueError("candidate changed native rate, timebase, or finite output")
    output_sha = sha(folder / "result.after.wav")
    if profile.get("expect_identical") and output_sha != baseline["output_sha256"]:
        raise ValueError("disabled repair is not sample-identical to frozen baseline")
    scored = score_aligned(clean, before, audio[delay:], scene, "full", engine["clipped_samples_before_write"])
    if profile.get("expect_identical") and scored["failures"] != baseline["failures"]:
        raise ValueError("disabled repair changed gate decisions")
    row = dict(profile=profile["name"], variant=baseline["variant"],
        run=baseline["run"], id=baseline["id"], noise=baseline["noise"],
        noise_class=baseline["noise_class"], snr_db=baseline["snr_db"], split=scene["split"],
        baseline_passed=baseline["passed"], baseline_failures=baseline["failures"],
        output_sha256=output_sha, output_changed=output_sha != baseline["output_sha256"], engine=engine, **scored)
    save(folder / "metrics.json", row)
    return row


def screen(rows, scenes):
    # Retrospective development screening, never a claim of blind validation.
    # Include each noise/SNR regime and the largest failures; also retain passing
    # controls so tuning cannot silently trade speech fidelity for attenuation.
    rows = [r for r in rows if scenes[(r["run"], r["id"])]["split"] == "development"]
    buckets = collections.defaultdict(list)
    for row in rows:
        band = "low" if row["snr_db"] < 0 else "mid" if row["snr_db"] < 20 else "high"
        buckets[(row["noise"], band, row["passed"])].append(row)
    selected = {}
    for items in buckets.values():
        for row in sorted(items, key=lambda r: (r["snr_db"], r["id"]))[::max(1, len(items)-1)]:
            selected[(row["run"], row["id"])] = row
    return list(selected.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scope", choices=("screen", "full"), default="screen")
    parser.add_argument("--jobs", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    out, frozen = args.out.resolve(), args.frozen_root.resolve()
    out.mkdir(parents=True, exist_ok=False)
    plan = json.loads(args.plan.read_text())
    if not plan or len({p["name"] for p in plan}) != len(plan):
        raise ValueError("empty or duplicate profile plan")
    binary = frozen / "ns_vehicle_probe"
    if sha(binary) != json.loads((frozen / "identity.json").read_text())["binary_sha256"]:
        raise ValueError("candidate build identity differs")
    source = BASE / "full-r3-fixed-results.jsonl"
    baseline = [r for line in source.read_text().splitlines() if (r := json.loads(line))["treatment"] == "off"]
    manifest = P / "build/ns-gtcrn-direct-20261018/full/identity.json"
    scenes = {(r["run"], r["scene"]["id"]): r["scene"]
              for r in json.loads(manifest.read_text())["selection"]}
    tasks, selections = [], {}
    for profile in plan:
        if profile["variant"] not in BRANCHES or Path(profile["name"]).name != profile["name"]:
            raise ValueError("invalid profile")
        rows = [r for r in baseline if r["variant"] == profile["variant"]]
        if len(rows) != 692 or len({(r["run"], r["id"]) for r in rows}) != 692:
            raise ValueError("incomplete or duplicate source branch")
        if args.scope == "screen":
            rows = screen(rows, scenes)
        selections[profile["name"]] = [dict(run=r["run"], id=r["id"]) for r in rows]
        tasks.extend((out, binary, profile, r, scenes[(r["run"], r["id"])]) for r in rows)
    bindings = {str(p): sha(p) for p in (binary, source, manifest, args.plan.resolve(), Path(__file__),
                Path(__file__).with_name("evaluate_gtcrn_direct.py"),
                Path(__file__).with_name("diagnose_traditional_ns_remaining.py"))}
    save(out / "identity.json", dict(bindings=bindings, plan=plan, scope=args.scope,
        selection=selections, thresholds_changed=False, no_fitted_alignment=True,
        tuning_is_retrospective=True, realtime_validation=False))
    results = []
    with ProcessPoolExecutor(args.jobs) as pool, (out / "results.jsonl").open("x") as stream:
        for i, future in enumerate(as_completed([pool.submit(execute, task) for task in tasks]), 1):
            row = future.result()
            results.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            if i % 50 == 0 or i == len(tasks):
                save(out / "report.json", dict(state="running", completed=i, total=len(tasks)))
                print(i, "/", len(tasks), flush=True)
    for path, digest in bindings.items():
        if sha(Path(path)) != digest:
            raise ValueError("experiment dependency changed during run")
    summary = {}
    for profile in plan:
        group = [r for r in results if r["profile"] == profile["name"]]
        if len(group) != len(selections[profile["name"]]):
            raise ValueError("incomplete profile results")
        summary[profile["name"]] = dict(total=len(group),
            baseline_passed=sum(r["baseline_passed"] for r in group), passed=sum(r["passed"] for r in group),
            recovered=sum(not r["baseline_passed"] and r["passed"] for r in group),
            regressed=sum(r["baseline_passed"] and not r["passed"] for r in group),
            changed_outputs=sum(r["output_changed"] for r in group),
            gate_failed_cases={g: sum(any(f.split(":")[-1] == g for f in r["failures"]) for r in group)
                               for g in ("attenuation", "projection_gain", "raw_sisdr", "stoi", "clipping")},
            noise_classes={c: dict(total=sum(r["noise_class"] == c for r in group),
                passed=sum(r["noise_class"] == c and r["passed"] for r in group)) for c in ("steady", "transient")})
    save(out / "summary.json", summary)
    save(out / "report.json", dict(state="completed", completed=len(results), total=len(tasks),
        thresholds_changed=False, realtime_validation=False))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
