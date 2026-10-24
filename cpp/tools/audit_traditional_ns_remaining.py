#!/usr/bin/env python3
"""Audit the retained remaining-failure and fixed-rumble experiments.

This checks stored metrics against unchanged gates and verifies the actual
invocations, input/output hashes and declared delays. It does not recompute
STOI/SI-SDR from PCM or turn diagnostic per-scene choices into a runtime policy.
"""
import argparse
import collections
import copy
import json
import math
from pathlib import Path
import statistics

import soundfile as sf

from audit_traditional_ns_residual import failures, finite, read_rows, sha, unique
from diagnose_traditional_ns_remaining import BASE, BRANCHES, FROZEN, P, original_folder, token


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def scene_key(row):
    return row["run"], row["id"]


def check_before_metrics(actual, expected, differences):
    if actual.keys() != expected.keys():
        raise ValueError("paired before-metric structure mismatch")
    for name, value in actual.items():
        previous = expected[name]
        if value == previous:
            continue
        if (type(value) is float and type(previous) is float
                and math.isclose(value, previous, rel_tol=0, abs_tol=1e-10)):
            differences[name] = max(differences[name], abs(value - previous))
        else:
            raise ValueError("paired speech before-metric mismatch: " + name)


def branch_summary(rows):
    return dict(total=len(rows), passed=sum(r["passed"] for r in rows),
        failed=sum(not r["passed"] for r in rows),
        # Count cases per gate, rather than failing windows per gate.
        failed_by_gate={g: sum(any(f.split(":")[-1] == g for f in r["failures"]) for r in rows)
                        for g in ("attenuation", "projection_gain", "raw_sisdr", "stoi", "clipping")},
        noise_classes={c: dict(total=sum(r["noise_class"] == c for r in rows),
            passed=sum(r["noise_class"] == c and r["passed"] for r in rows))
            for c in ("steady", "transient")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--diagnostic-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    candidate, diagnostic, out = (p.resolve() for p in
        (args.candidate_root, args.diagnostic_root, args.out))
    out.mkdir(parents=True, exist_ok=False)
    source = BASE / "full-r3-fixed-results.jsonl"
    off = unique([r for r in read_rows(source) if r["treatment"] == "off"])
    trial = unique(read_rows(candidate / "results.jsonl"))
    expected = {k for k in off if k[0] != "webrtc_v5_bounded09"}
    if len(off) != 2768 or len(trial) != 2076 or set(trial) != expected:
        raise ValueError("missing/extra comparison case")
    by_branch = {v: {scene_key(r): r for r in off.values() if r["variant"] == v}
                 for v in BRANCHES}
    keys = set(next(iter(by_branch.values())))
    if len(keys) != 692 or any(set(group) != keys for group in by_branch.values()):
        raise ValueError("four-way paired case sets differ")
    identity = json.loads((candidate / "identity.json").read_text())
    bindings = {FROZEN / "ns_vehicle_probe": identity["binary_sha256"],
                source: identity["baseline_sha256"],
                Path(__file__).with_name("diagnose_traditional_ns_remaining.py"):
                    identity["diagnostic_script_sha256"],
                Path(__file__).with_name("validate_traditional_rumble.py"): identity["script_sha256"]}
    for path, digest in bindings.items():
        if sha(path) != digest:
            raise ValueError("experiment input drift: " + str(path))
    report = json.loads((candidate / "report.json").read_text())
    if report["state"] != "completed" or report["completed"] != 2076 or report["thresholds_changed"]:
        raise ValueError("incomplete/changed-threshold experiment")
    selection = P / "build/ns-gtcrn-direct-20261018/full/identity.json"
    scenes = {(r["run"], r["scene"]["id"]): r["scene"]
              for r in json.loads(selection.read_text())["selection"]}
    lengths, checked = {}, collections.Counter()
    before_differences = collections.defaultdict(float)
    changed = []
    for k, row in sorted(trial.items()):
        old = off[k]
        sk = scene_key(row)
        scene = scenes[sk]
        if sk not in lengths:
            n = []
            for name in ("before.wav", "clean.wav"):
                path = P / "build" / row["run"] / "cases" / row["id"] / name
                if sha(path) != scene["files"][name]["sha256"]:
                    raise ValueError("frozen input drift")
                info = sf.info(path)
                if info.samplerate != 16000 or info.channels != 1:
                    raise ValueError("non-native-rate/mono input")
                n.append(info.frames)
                checked["input_hashes_and_headers"] += 1
            if n[0] != n[1]:
                raise ValueError("input and clean lengths differ")
            lengths[sk] = n[0]
        folder = candidate / "counterfactuals" / BRANCHES[row["variant"]] / token(row) / "fixed_rumble"
        invocation = json.loads((folder / "invocation.json").read_text())
        argv = json.loads((original_folder(old) / "invocation.json").read_text())["argv"]
        argv[0], argv[4], argv[18] = str(FROZEN / "ns_vehicle_probe"), str(folder / "result"), "linear"
        if invocation["returncode"] or invocation["argv"] != argv:
            raise ValueError("treatment changed additional invocation arguments")
        engine = json.loads((folder / "result.json").read_text())
        wav = folder / "result.after.wav"
        info = sf.info(wav)
        if sha(wav) != row["output_sha256"]:
            raise ValueError("scored output hash mismatch")
        if (info.samplerate != 16000 or info.channels != 1
                or info.frames != lengths[sk] + row["delay_samples"]
                or engine["delay_samples"] != row["delay_samples"]
                or row["delay_samples"] != old["engine"]["delay_samples"]
                or engine["core_rate"] != 16000 or engine["src_delay_samples"] != 0):
            raise ValueError("output shape, rate, or declared delay mismatch")
        audited = dict(row, engine=engine)
        if (not finite(row) or failures(audited) != row["failures"]
                or row["passed"] != (not row["failures"])
                or row["baseline_passed"] != old["passed"]
                or row["baseline_failures"] != old["failures"]):
            raise ValueError("nonfinite metrics or gate/baseline inconsistency")
        for name, segment in row["segments"].items():
            if name.startswith("near"):
                check_before_metrics(segment["before"], old["segments"][name]["before"], before_differences)
        checked["candidate_invocations"] += 1
        checked["output_hashes_and_headers"] += 1
        checked["stored_gate_recomputations"] += 1
        if row["passed"] != old["passed"]:
            changed.append(dict(variant=row["variant"], run=row["run"], id=row["id"],
                noise=row["noise"], noise_class=row["noise_class"], snr_db=row["snr_db"],
                change="recovered" if row["passed"] else "regressed",
                before_failures=old["failures"], after_failures=row["failures"],
                before_segments=old["segments"], after_segments=row["segments"]))
    for row in off.values():
        if not finite(row) or failures(row) != row["failures"] or row["passed"] != (not row["failures"]):
            raise ValueError("baseline gate inconsistency")
    # Verify the audit detects a deliberately introduced projection-gain failure.
    control = copy.deepcopy(next(r for r in off.values() if r["passed"]))
    label = next(n for n in control["segments"] if n.startswith("near"))
    control["segments"][label]["after"]["projection_gain_db"] = -4
    if label + ":projection_gain" not in failures(control):
        raise ValueError("gate audit negative control failed")
    common = []
    for sk in sorted(keys):
        group = [by_branch[v][sk] for v in BRANCHES]
        if all(not r["passed"] for r in group):
            first = group[0]
            noise_only = all(all(f.endswith(":attenuation") for f in r["failures"]) for r in group)
            common.append(dict(run=sk[0], id=sk[1], noise=first["noise"],
                noise_class=first["noise_class"], snr_db=first["snr_db"],
                all_failures_noise_only=noise_only,
                failures={r["variant"]: r["failures"] for r in group}))
    diagnostic_rows = json.loads((diagnostic / "failure-waveform-diagnostics.json").read_text())
    diag_keys = {(r["variant"], r["run"], r["id"]) for r in diagnostic_rows}
    if len(diagnostic_rows) != 274 or diag_keys != {k for k, r in off.items() if not r["passed"]}:
        raise ValueError("failure diagnostic coverage mismatch")
    diagnostics = {}
    for v in BRANCHES:
        group = [r for r in diagnostic_rows if r["variant"] == v]
        windows = [w for r in group for w in r["windows"].values()]
        noise = [w for w in windows if "frequency_bands" in w]
        speech = [w for w in windows if any(f.endswith(":projection_gain") for f in w["gates"])]
        diagnostics[v] = dict(
            failed_noise_windows=len(noise),
            median_output_power_fraction_0_80_hz=statistics.median(
                w["frequency_bands"]["0-80"]["output_power_fraction"] for w in noise),
            projection_failed_windows=len(speech),
            median_clean_energy_fraction_in_frames_below_minus6db_projection=statistics.median(
                w["clean_energy_fraction_under_minus6"] for w in speech),
            projection_failed_window_control_means_median={name: statistics.median(
                w["output_time_control_means"][name] for w in speech)
                for name in ("confidence", "snr_db", "input_mix_target", "quiet_mix_target")})
    summaries = {}
    for v in BRANCHES:
        summaries[v] = dict(baseline=branch_summary(list(by_branch[v].values())))
        group = [r for r in trial.values() if r["variant"] == v]
        if group:
            changes = [r for r in changed if r["variant"] == v]
            summaries[v]["fixed_rumble"] = dict(**branch_summary(group),
                recovered=sum(r["change"] == "recovered" for r in changes),
                regressed=sum(r["change"] == "regressed" for r in changes))
    save(out / "summary.json", summaries)
    save(out / "changed-cases.json", changed)
    save(out / "common-failures.json", common)
    save(out / "failure-diagnostic-summary.json", diagnostics)
    save(out / "report.json", dict(state="completed", checked=dict(checked),
        baseline_cases=len(off), candidate_cases=len(trial), failure_diagnostics=len(diagnostic_rows),
        common_failed_cases=len(common), common_steady=sum(r["noise_class"] == "steady" for r in common),
        common_transient_noise_only=sum(r["noise_class"] == "transient" and r["all_failures_noise_only"] for r in common),
        gate_negative_control_passed=True, stored_metrics_recomputed_from_pcm=False,
        before_metric_max_abs_roundoff=dict(before_differences), before_metric_abs_tolerance=1e-10,
        gate_tolerance=0,
        thresholds_changed=False, default_configuration_changed=False, realtime_verified=False))
    paths = [*bindings, Path(__file__), selection, diagnostic / "failure-waveform-diagnostics.json",
             candidate / "results.jsonl", candidate / "identity.json", candidate / "report.json",
             Path(__file__).with_name("audit_traditional_ns_residual.py")]
    save(out / "identity.json", dict(bindings={str(p): sha(p) for p in paths},
        scope="stored gate decisions, invocations, hashes, headers and paired before metrics"))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
