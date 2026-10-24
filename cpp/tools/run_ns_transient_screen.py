#!/usr/bin/env python3
"""Screen the causal transient dry-mix guard with paired 52-scene runs."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import soundfile as sf

from evaluate_gtcrn_direct import score_aligned
from run_ns_speechband_screen import (
    FROZEN, IDENTITY, P, PROFILE, RESULTS, key, save, selection, sha, token)

DEFAULT_REFERENCE = P / "build/ns-speechband-20261020/screen-r33"


def trace_summary(path):
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream))
    evidence = [float(row["transient_noise_evidence"]) for row in rows]
    applied = [float(row["transient_guard_applied"]) for row in rows]
    spectral_applied = [
        float(row["transient_spectral_guard_applied"]) for row in rows
    ]
    periodic_100 = [float(row["transient_periodicity_100ms"]) for row in rows]
    periodic_500 = [float(row["transient_periodicity_500ms"]) for row in rows]
    return dict(
        trace_frames=len(rows),
        evidence_frames=sum(value > 0 for value in evidence),
        guard_frames=sum(value < .999 for value in applied),
        spectral_guard_frames=sum(value < .999 for value in spectral_applied),
        evidence_max=max(evidence, default=0),
        guard_min=min(applied, default=1),
        spectral_guard_min=min(spectral_applied, default=1),
        periodicity_100ms_max=max(periodic_100, default=0),
        periodicity_500ms_max=max(periodic_500, default=0),
    )


def execute(task):
    root, binary, guard, spectral_db, baseline, scene = task
    folder = root / f"guard-{guard}-spectral-{spectral_db}" / token(baseline)
    folder.mkdir(parents=True, exist_ok=False)
    original = json.loads(
        (FROZEN / PROFILE / token(baseline) / "invocation.json").read_text()
    )["argv"]
    command = list(original)
    command[0], command[4] = str(binary), str(folder / "result")
    command.extend([
        "--speech-band-recovery=1.0",
        f"--transient-dry-guard={guard}",
        f"--transient-spectral-guard-db={spectral_db}",
    ])
    proc = subprocess.run(command, capture_output=True, text=True)
    save(folder / "invocation.json", dict(
        argv=command, returncode=proc.returncode, stderr=proc.stderr))
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
    scored = score_aligned(
        clean, before, audio[delay:], scene, "full",
        engine["clipped_samples_before_write"])
    row = dict(
        guard=guard, spectral_guard_db=spectral_db,
        run=baseline["run"], id=baseline["id"],
        noise=baseline["noise"], noise_class=baseline["noise_class"],
        snr_db=baseline["snr_db"], split=scene["split"],
        baseline_passed=baseline["passed"],
        baseline_failures=baseline["failures"],
        output_sha256=output_sha,
        transient_dry_guard=engine["transient_dry_guard"],
        transient_spectral_guard_db=engine["transient_spectral_guard_db"],
        speech_band_recovery=engine["speech_band_recovery"],
        delay_samples=delay, engine=engine,
        **trace_summary(folder / "result.csv"), **scored)
    save(folder / "metrics.json", row)
    return row


def failure_set(rows, gate):
    return {
        (row["run"], row["id"])
        for row in rows
        if any(failure.split(":")[-1] == gate for failure in row["failures"])
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference-screen", type=Path,
                        default=DEFAULT_REFERENCE)
    parser.add_argument("--jobs", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--spectral-guard-db", type=float, default=18)
    args = parser.parse_args()
    if not 0 < args.spectral_guard_db <= 24:
        raise ValueError("spectral guard must be in (0, 24] dB")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binary = args.frozen_binary.resolve()
    reference = args.reference_screen.resolve()

    manifest = P / "build/ns-gtcrn-direct-20261018/full/identity.json"
    scenes = {
        (row["run"], row["scene"]["id"]): row["scene"]
        for row in json.loads(manifest.read_text())["selection"]
    }
    rows = [
        json.loads(line) for line in RESULTS.open()
        if json.loads(line)["profile"] == PROFILE
    ]
    if len(rows) != 692 or len({(row["run"], row["id"]) for row in rows}) != 692:
        raise ValueError("incomplete frozen OM-LSA selection")
    selected = selection(rows, scenes, "screen")
    selected.sort(key=lambda row: (row["run"], row["id"]))
    if len(selected) != 52:
        raise ValueError("screen selection changed")

    reference_identity = json.loads((reference / "identity.json").read_text())
    reference_summary = json.loads((reference / "summary.json").read_text())
    if (len(reference_identity["selection"]) != 52
            or reference_summary.get("1.0", {}).get("recovered") != 5
            or reference_summary.get("1.0", {}).get("regressed") != 0):
        raise ValueError("reference screen is not the 5-recovery speech-band baseline")
    reference_files = {
        token(row): reference / "recovery-1p0" / token(row) / "result.after.wav"
        for row in selected
    }
    missing_reference = [str(path) for path in reference_files.values()
                         if not path.is_file()]
    if missing_reference:
        raise ValueError(f"reference screen is incomplete: {missing_reference[:3]}")

    dependencies = (
        binary, RESULTS, IDENTITY, manifest, reference / "identity.json",
        reference / "summary.json", Path(__file__),
        Path(__file__).with_name("run_ns_speechband_screen.py"),
        Path(__file__).with_name("evaluate_gtcrn_direct.py"),
    )
    bindings = {str(path): sha(path) for path in dependencies}
    save(out / "identity.json", dict(
        bindings=bindings, binary_sha256=sha(binary),
        selection=[
            dict(run=row["run"], id=row["id"], baseline_passed=row["passed"])
            for row in selected
        ],
        guards=[0.0, 1.0], speech_band_recovery=1.0,
        spectral_guard_db=args.spectral_guard_db,
        reference_screen=str(reference), thresholds_changed=False,
        no_fitted_alignment=True, scope="screen"))

    variants = (
        (0, 0),
        (1, 0),
        (1, args.spectral_guard_db),
    )
    tasks = [
        (out, binary, guard, spectral_db, row, scenes[key(row)])
        for guard, spectral_db in variants for row in selected
    ]
    results = []
    with ProcessPoolExecutor(args.jobs) as pool, (
            out / "results.jsonl").open("x") as stream:
        futures = [pool.submit(execute, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            results.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            if index % 25 == 0 or index == len(tasks):
                save(out / "report.json", dict(
                    state="running", completed=index, total=len(tasks)))
                print(index, "/", len(tasks), flush=True)

    changed_dependencies = [
        path for path, digest in bindings.items() if sha(Path(path)) != digest
    ]
    if changed_dependencies:
        raise ValueError(f"experiment dependency changed: {changed_dependencies}")

    by_variant = {
        (guard, spectral_db): [
            row for row in results
            if row["guard"] == guard
            and row["spectral_guard_db"] == spectral_db
        ]
        for guard, spectral_db in variants
    }
    reference_mismatches = [
        f'{row["run"]}/{row["id"]}'
        for row in by_variant[(0, 0)]
        if row["output_sha256"] != sha(reference_files[token(row)])
    ]
    summary = {}
    for variant, group in by_variant.items():
        summary["/".join(map(str, variant))] = dict(
            total=len(group),
            passed=sum(row["passed"] for row in group),
            recovered=sum(
                not row["baseline_passed"] and row["passed"] for row in group),
            regressed=sum(
                row["baseline_passed"] and not row["passed"] for row in group),
            changed_outputs=sum(
                row["output_sha256"] != sha(reference_files[token(row)])
                for row in group),
            triggered_cases=sum(row["evidence_frames"] > 0 for row in group),
            applied_cases=sum(row["guard_frames"] > 0 for row in group),
            spectral_applied_cases=sum(
                row["spectral_guard_frames"] > 0 for row in group),
            total_evidence_frames=sum(row["evidence_frames"] for row in group),
            total_guard_frames=sum(row["guard_frames"] for row in group),
            total_spectral_guard_frames=sum(
                row["spectral_guard_frames"] for row in group),
            gate_failed_cases={
                gate: sum(
                    any(failure.split(":")[-1] == gate
                        for failure in row["failures"])
                    for row in group)
                for gate in ("attenuation", "projection_gain", "raw_sisdr",
                             "stoi", "clipping")
            })

    dry_only = {
        (row["run"], row["id"]): row for row in by_variant[(1, 0)]
    }
    guarded = {
        (row["run"], row["id"]): row
        for row in by_variant[(1, args.spectral_guard_db)]
    }
    pairs = sorted(dry_only)
    # Compare failures by case key, not aggregate counts.
    stoi_off = failure_set(by_variant[(1, 0)], "stoi")
    stoi_on = failure_set(by_variant[(1, args.spectral_guard_db)], "stoi")
    clipping_off = failure_set(by_variant[(1, 0)], "clipping")
    clipping_on = failure_set(
        by_variant[(1, args.spectral_guard_db)], "clipping")
    new_stoi = sorted(stoi_on - stoi_off)
    new_clipping = sorted(clipping_on - clipping_off)
    regressions = sorted(
        key_value for key_value in pairs
        if dry_only[key_value]["passed"] and not guarded[key_value]["passed"])
    recoveries = sorted(
        key_value for key_value in pairs
        if not dry_only[key_value]["passed"]
        and guarded[key_value]["passed"])
    triggered = sorted(
        key_value for key_value in pairs
        if guarded[key_value]["guard_frames"] > 0
        and guarded[key_value]["spectral_guard_frames"] > 0)
    checks = {
        "disabled_guard_bit_identical": not reference_mismatches,
        "no_regression": not regressions,
        "new_recovery_positive": bool(recoveries),
        "no_new_stoi": not new_stoi,
        "no_new_clipping": not new_clipping,
        "guards_actually_triggered": bool(triggered),
    }
    gate = dict(
        passed=all(checks.values()), checks=checks,
        reference_mismatches=reference_mismatches,
        regressions=[f"{run}/{case}" for run, case in regressions],
        newly_recovered=[f"{run}/{case}" for run, case in recoveries],
        new_stoi=[f"{run}/{case}" for run, case in new_stoi],
        new_clipping=[f"{run}/{case}" for run, case in new_clipping],
        triggered_cases=[f"{run}/{case}" for run, case in triggered])
    save(out / "summary.json", summary)
    save(out / "gate.json", gate)
    save(out / "report.json", dict(
        state="completed", completed=len(results), total=len(tasks),
        dependencies_verified=True, thresholds_changed=False,
        realtime_validation=False, gate_passed=gate["passed"]))
    print(json.dumps(dict(summary=summary, gate=gate), indent=2), flush=True)
    if not gate["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
