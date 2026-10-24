#!/usr/bin/env python3
"""Audit frozen sequential NS replays against unchanged inputs and gates.

Gate decisions are independently reconstructed from retained measurements.
This verifies PCM identities and headers, but does not re-estimate STOI/SI-SDR.
"""
import argparse
import collections
import copy
import json
from pathlib import Path

import soundfile as sf

from audit_traditional_ns_residual import failures, finite, read_rows, sha, unique
from audit_traditional_ns_remaining import check_before_metrics, branch_summary
from diagnose_traditional_ns_remaining import BASE, P, original_folder, token


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    identity = json.loads((root / "identity.json").read_text())
    for name, digest in identity["bindings"].items():
        if sha(Path(name)) != digest:
            raise ValueError("bound input drift: " + name)
    status = json.loads((root / "report.json").read_text())
    rows = read_rows(root / "results.jsonl")
    if (identity["scope"] != "full" or status["state"] != "completed"
            or status["completed"] != len(rows) or status["total"] != len(rows)
            or identity["thresholds_changed"] or status["thresholds_changed"]):
        raise ValueError("incomplete or changed-threshold full comparison")
    baseline = unique([r for r in read_rows(BASE / "full-r3-fixed-results.jsonl")
                       if r["treatment"] == "off"])
    manifest = json.loads((P / "build/ns-gtcrn-direct-20261018/full/identity.json").read_text())
    scenes = {(r["run"], r["scene"]["id"]): r["scene"] for r in manifest["selection"]}
    profiles = {p["name"]: p for p in identity["plan"]}
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row["profile"]].append(row)
    if set(grouped) != set(profiles):
        raise ValueError("profile coverage differs")
    lengths, checked, before_differences = {}, collections.Counter(), collections.defaultdict(float)
    changes, summaries = [], {}
    for name, group in grouped.items():
        profile = profiles[name]
        paired = unique(group)
        expected = {k for k in baseline if k[0] == profile["variant"]}
        if len(paired) != 692 or set(paired) != expected:
            raise ValueError("incomplete or unpaired profile")
        for key, row in paired.items():
            previous = baseline[key]
            sk = row["run"], row["id"]
            scene = scenes[sk]
            if sk not in lengths:
                sizes = []
                for filename in ("before.wav", "clean.wav"):
                    path = P / "build" / row["run"] / "cases" / row["id"] / filename
                    if sha(path) != scene["files"][filename]["sha256"]:
                        raise ValueError("frozen PCM input drift")
                    info = sf.info(path)
                    if info.samplerate != 16000 or info.channels != 1:
                        raise ValueError("input is not native 16 kHz mono")
                    sizes.append(info.frames)
                    checked["input_hashes_and_headers"] += 1
                if sizes[0] != sizes[1]:
                    raise ValueError("clean/input shape mismatch")
                lengths[sk] = sizes[0]
            folder = root / name / token(row)
            invocation = json.loads((folder / "invocation.json").read_text())
            original = json.loads((original_folder(previous) / "invocation.json").read_text())["argv"]
            command = original[:]
            command[0] = invocation["argv"][0]
            command[4] = str(folder / "result")
            command.extend(profile.get("options", []))
            if invocation["returncode"] or invocation["argv"] != command:
                raise ValueError("unexpected invocation mutation")
            if command[0] not in identity["bindings"]:
                raise ValueError("unbound executable")
            engine = json.loads((folder / "result.json").read_text())
            path = folder / "result.after.wav"
            info = sf.info(path)
            if (sha(path) != row["output_sha256"] or engine != row["engine"]
                    or info.samplerate != 16000 or info.channels != 1
                    or info.frames != lengths[sk] + engine["delay_samples"]
                    or engine["delay_samples"] != previous["engine"]["delay_samples"]
                    or engine["core_rate"] != 16000 or engine["src_delay_samples"] != 0):
                raise ValueError("output identity, timebase or native rate mismatch")
            if (not finite(row) or failures(row) != row["failures"]
                    or row["passed"] != (not row["failures"])
                    or row["baseline_passed"] != previous["passed"]
                    or row["baseline_failures"] != previous["failures"]):
                raise ValueError("gate or baseline evidence mismatch")
            for field in ("noise", "noise_class", "snr_db", "variant"):
                if row[field] != previous[field]:
                    raise ValueError("paired metadata differs")
            for label, segment in row["segments"].items():
                if label.startswith("near"):
                    check_before_metrics(segment["before"], previous["segments"][label]["before"], before_differences)
            if profile.get("expect_identical") and row["output_sha256"] != previous["output_sha256"]:
                raise ValueError("disabled repairs changed PCM")
            if previous["passed"] != row["passed"]:
                changes.append(dict(profile=name, run=row["run"], id=row["id"],
                    noise=row["noise"], snr_db=row["snr_db"],
                    change="recovered" if row["passed"] else "regressed",
                    previous_failures=previous["failures"], failures=row["failures"]))
            checked["invocations_outputs_and_gates"] += 1
        summaries[name] = dict(**branch_summary(group),
            recovered=sum(not r["baseline_passed"] and r["passed"] for r in group),
            regressed=sum(r["baseline_passed"] and not r["passed"] for r in group),
            splits={split: branch_summary([r for r in group if r["split"] == split])
                    for split in sorted({r["split"] for r in group})})
    control = copy.deepcopy(next(r for r in rows if r["passed"]))
    label = next(k for k in control["segments"] if k.startswith("near"))
    control["segments"][label]["after"]["projection_gain_db"] = -4
    if label + ":projection_gain" not in failures(control):
        raise ValueError("known-bad gate negative control was not detected")
    report = dict(state="completed", source=str(root), checked=dict(checked),
        thresholds_changed=False, fitted_alignment=False, realtime_validation=False,
        waveform_metrics_recomputed=False, known_bad_gate_detected=True,
        before_metric_max_roundoff=dict(before_differences), summary=summaries, changes=changes,
        bindings=identity["bindings"], audit_sha256=sha(Path(__file__)))
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(checked=dict(checked), summary=summaries), indent=2))


if __name__ == "__main__":
    main()
