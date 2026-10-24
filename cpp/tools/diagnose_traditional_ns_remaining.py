#!/usr/bin/env python3
"""Measure remaining original-profile NS failures and bounded counterfactuals.

All reference labels are used by this offline diagnostic only. The frozen C++
executable receives capture PCM and explicit configuration, never clean audio.
Counterfactual success is diagnostic and is not a deployable per-case selector.
"""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_gtcrn_direct import score_aligned

P = Path(__file__).resolve().parents[2]
BASE = P / "build/ns-traditional-residual-20261019-v5"
FROZEN = P / "build/ns-traditional-residual-20261019-v6"
BRANCHES = {"webrtc_v5_bounded09": "webrtc", "speex_v5_steadydual": "speex",
            "omlsa_imcra_v2_mild": "omlsa", "super_gaussian_balanced": "super"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def key(row):
    return row["run"], row["id"]


def token(row):
    return hashlib.sha256((row["run"] + "/" + row["id"]).encode()).hexdigest()[:20]


def original_folder(row):
    return BASE / "full-r3-fixed" / BRANCHES[row["variant"]] / "off" / token(row)


def inputs(row, scene):
    folder = P / "build" / row["run"] / "cases" / row["id"]
    for name in ("before.wav", "clean.wav"):
        if sha(folder / name) != scene["files"][name]["sha256"]:
            raise ValueError("input identity drift")
    before, br = sf.read(folder / "before.wav")
    clean, cr = sf.read(folder / "clean.wav")
    if br != cr or br != 16000 or before.shape != clean.shape or before.ndim != 1:
        raise ValueError("non-native/unequal diagnostic inputs")
    before = np.clip(np.rint(before * 32768), -32768, 32767) / 32768
    return before, clean


def quantiles(values):
    return {"p05": float(np.quantile(values, .05)), "median": float(np.median(values)),
            "p95": float(np.quantile(values, .95))} if len(values) else None


def spectral(before, after):
    window = np.hanning(len(before))
    b, a = (np.abs(np.fft.rfft(x * window)) ** 2 for x in (before, after))
    hz = np.fft.rfftfreq(len(before), 1 / 16000)
    result = {}
    for lo, hi in ((0, 80), (80, 300), (300, 1000), (1000, 3400), (3400, 8001)):
        mask = (hz >= lo) & (hz < hi)
        bp, ap = float(b[mask].sum()), float(a[mask].sum())
        result[f"{lo}-{hi}"] = dict(input_power_fraction=bp / max(float(b.sum()), 1e-30),
            output_power_fraction=ap / max(float(a.sum()), 1e-30),
            attenuation_db=10 * np.log10(max(bp, 1e-30) / max(ap, 1e-30)))
    return result


def waveform_diagnostic(task):
    row, scene = task
    before, clean = inputs(row, scene)
    folder = original_folder(row)
    output = folder / "result.after.wav"
    if sha(output) != row["output_sha256"]:
        raise ValueError("baseline output drift")
    after, rate = sf.read(output)
    delay = row["engine"]["delay_samples"]
    if rate != 16000 or len(after) != len(clean) + delay or not np.isfinite(after).all():
        raise ValueError("baseline waveform shape/finite check failed")
    after = after[delay:]
    h = 160
    n = len(clean) // h
    c, b, a = (x[:n * h].reshape(n, h) for x in (clean, before, after))
    energy = np.sum(c * c, axis=1)
    valid = energy / h >= 10 ** (-65 / 10)
    alpha = np.divide(np.sum(c * a, axis=1), energy,
                      out=np.zeros(n), where=energy > 1e-15)
    pg = 20 * np.log10(np.maximum(np.abs(alpha), 1e-12))
    with (folder / "result.csv").open() as stream:
        observations = list(csv.DictReader(stream))
    windows = {}
    for label in sorted({f.split(":")[0] for f in row["failures"] if ":" in f}):
        lo, hi = scene["windows"][label]
        lo_s, hi_s = round(lo * rate), min(len(clean), round(hi * rate))
        lo_f, hi_f = round(lo * 100), min(n, round(hi * 100))
        item = dict(limits_s=[lo, hi], gates=[f for f in row["failures"] if f.startswith(label + ":")],
                    scored_segment=row["segments"][label])
        if label.startswith("near"):
            active = valid[lo_f:hi_f]
            selected = pg[lo_f:hi_f][active]
            weights = energy[lo_f:hi_f][active]
            item.update(active_frames=int(active.sum()), projection_db=quantiles(selected),
                frames_under_minus3=int((selected < -3).sum()),
                frames_under_minus6=int((selected < -6).sum()),
                clean_energy_fraction_under_minus6=float(weights[selected < -6].sum() / max(weights.sum(), 1e-30)))
        else:
            item["frequency_bands"] = spectral(before[lo_s:hi_s], after[lo_s:hi_s])
        # CSV is indexed by processing call, so include only recorded calls
        # corresponding to the declared-delay-aligned output window.
        first_call = (lo_s + delay) // h
        last_call = min(len(observations), (hi_s + delay + h - 1) // h)
        calls = observations[first_call:last_call]
        item["output_time_control_means"] = {
            name: float(np.mean([float(x[name]) for x in calls]))
            for name in ("confidence", "snr_db", "input_mix_target", "quiet_mix_target",
                         "rumble_fraction", "rumble_mix_applied", "steady_mix_target")}
        windows[label] = item
    return dict(variant=row["variant"], run=row["run"], id=row["id"], noise=row["noise"],
                noise_class=row["noise_class"], snr_db=row["snr_db"], failures=row["failures"], windows=windows)


def margin(row, failure):
    label, metric = failure.split(":", 1)
    segment = row["segments"][label]
    if metric == "attenuation":
        value = segment.get("total_attenuation_db")
        if value is None:
            value = segment["attenuation_lower_bound_db"]
        return 6 - value
    before, after = segment["before"], segment["after"]
    if metric == "projection_gain":
        return max(-3 - after["projection_gain_db"], after["projection_gain_db"] - 18)
    if metric == "raw_sisdr":
        return min(before["sisdr_db"], 20) - 1 - after["sisdr_db"]
    return (before["stoi"] - .02 - after["stoi"]) * 100


def selection(rows, per_branch):
    result = []
    for variant in BRANCHES:
        failed = [r for r in rows if r["variant"] == variant and not r["passed"]]
        passed = [r for r in rows if r["variant"] == variant and r["passed"]]
        groups = collections.defaultdict(list)
        for r in failed:
            for f in r["failures"]:
                if ":" in f:
                    groups[(r["noise_class"], r["noise"], f.split(":")[1])].append((margin(r, f), r))
        ordered = sorted(groups, key=lambda g: (-len(groups[g]), g))
        chosen = {}
        # Reserve representation for each noise class, then cover the largest
        # distinct failure groups. Selection is fixed before counterfactuals.
        for kind in ("steady", "transient"):
            available = [g for g in ordered if g[0] == kind]
            if available:
                r = max(groups[available[0]], key=lambda pair: pair[0])[1]
                chosen[key(r)] = r
        for g in ordered:
            if len(chosen) >= per_branch:
                break
            r = max(groups[g], key=lambda pair: pair[0])[1]
            chosen[key(r)] = r
        for r in chosen.values():
            result.append((r, "baseline_failed"))
        controls = {}
        for r in chosen.values():
            control = min(passed, key=lambda p: (p["noise_class"] != r["noise_class"],
                          p["noise"] != r["noise"], abs(p["snr_db"] - r["snr_db"]),
                          p["run"] != r["run"], p["id"]))
            controls[key(control)] = control
        result.extend((r, "baseline_passed_control") for r in controls.values())
    return result


def counterfactual(task):
    out, row, scene, role, treatment = task
    folder = out / "counterfactuals" / BRANCHES[row["variant"]] / token(row) / treatment
    folder.mkdir(parents=True, exist_ok=False)
    command = json.loads((original_folder(row) / "invocation.json").read_text())["argv"]
    command[0] = str(FROZEN / "ns_vehicle_probe")
    command[4] = str(folder / "result")
    if treatment == "wet_only":
        for position, value in {6: "0", 9: "0", 10: "aligned", 14: "-300", 16: "0"}.items():
            command[position] = value
    elif treatment == "fixed_rumble":
        command[18] = "linear"
    elif treatment == "no_rumble":
        command[17] = "0"
        command[18] = "linear"
    elif treatment == "stronger_ns":
        command[2] = str(int(command[2]) - 6 if BRANCHES[row["variant"]] == "speex" else min(3, int(command[2]) + 1))
    elif treatment == "longer_release":
        command[13] = "200"
    elif treatment != "replay":
        raise ValueError("unknown counterfactual")
    proc = subprocess.run(command, capture_output=True, text=True)
    save(folder / "invocation.json", dict(argv=command, returncode=proc.returncode, stderr=proc.stderr))
    proc.check_returncode()
    engine = json.loads((folder / "result.json").read_text())
    before, clean = inputs(row, scene)
    after, ar = sf.read(folder / "result.after.wav")
    if ar != 16000 or len(after) != len(clean) + engine["delay_samples"] or not np.isfinite(after).all():
        raise ValueError("counterfactual output shape/finite check failed")
    output_hash = sha(folder / "result.after.wav")
    if treatment == "replay" and output_hash != row["output_sha256"]:
        raise ValueError("unmodified replay is not baseline-identical")
    if engine["src_delay_samples"] != 0 or engine["core_rate"] != 16000:
        raise ValueError("counterfactual changed native sample rate")
    scored = score_aligned(clean, before, after[engine["delay_samples"]:], scene, "full", engine["clipped_samples_before_write"])
    if treatment == "replay" and scored["failures"] != row["failures"]:
        raise ValueError("unmodified replay score differs")
    result = dict(variant=row["variant"], run=row["run"], id=row["id"], role=role,
                  treatment=treatment, noise=row["noise"], noise_class=row["noise_class"], snr_db=row["snr_db"],
                  baseline_passed=row["passed"], baseline_failures=row["failures"],
                  output_sha256=output_hash, delay_samples=engine["delay_samples"], **scored)
    save(folder / "metrics.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4, choices=range(1, 5))
    parser.add_argument("--per-branch", type=int, default=8)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    baseline = BASE / "full-r3-fixed-results.jsonl"
    rows = [json.loads(x) for x in baseline.read_text().splitlines() if json.loads(x)["treatment"] == "off"]
    scenes = {(r["run"], r["scene"]["id"]): r["scene"] for r in
              json.loads((P / "build/ns-gtcrn-direct-20261018/full/identity.json").read_text())["selection"]}
    binary_hash = sha(FROZEN / "ns_vehicle_probe")
    if binary_hash != json.loads((FROZEN / "identity.json").read_text())["binary_sha256"]:
        raise ValueError("frozen executable changed")
    failed = [r for r in rows if not r["passed"]]
    selected = selection(rows, args.per_branch)
    treatments = ("replay", "wet_only", "fixed_rumble", "no_rumble", "stronger_ns", "longer_release")
    save(out / "identity.json", dict(binary_sha256=binary_hash, baseline_sha256=sha(baseline),
         script_sha256=sha(Path(__file__)), selection=[dict(variant=r["variant"], run=r["run"], id=r["id"], role=role)
         for r, role in selected], treatments=treatments, thresholds_changed=False, diagnostic_only=True))
    with ProcessPoolExecutor(args.jobs) as pool:
        diagnostics = list(pool.map(waveform_diagnostic, [(r, scenes[key(r)]) for r in failed]))
        save(out / "failure-waveform-diagnostics.json", diagnostics)
        print("failed waveform diagnostics", len(diagnostics), flush=True)
        tasks = [(out, r, scenes[key(r)], role, t) for r, role in selected for t in treatments]
        results = []
        with (out / "counterfactual-results.jsonl").open("x") as stream:
            for i, future in enumerate(as_completed([pool.submit(counterfactual, t) for t in tasks]), 1):
                r = future.result()
                results.append(r)
                stream.write(json.dumps(r, allow_nan=False) + "\n")
                stream.flush()
                if i % 50 == 0 or i == len(tasks):
                    print(i, "/", len(tasks), flush=True)
    summary = {}
    for variant in BRANCHES:
        summary[variant] = {}
        for treatment in treatments:
            group = [r for r in results if r["variant"] == variant and r["treatment"] == treatment]
            summary[variant][treatment] = dict(total=len(group),
                failed_cases=sum(r["role"] == "baseline_failed" for r in group),
                recovered=sum(not r["baseline_passed"] and r["passed"] for r in group),
                controls=sum(r["baseline_passed"] for r in group),
                regressed=sum(r["baseline_passed"] and not r["passed"] for r in group),
                changed_outputs=sum(r["output_sha256"] != next(x["output_sha256"] for x in rows
                     if x["variant"] == variant and key(x) == key(r)) for r in group))
    if sha(FROZEN / "ns_vehicle_probe") != binary_hash:
        raise ValueError("executable changed during experiment")
    save(out / "summary.json", summary)
    save(out / "report.json", dict(state="completed", failures_analyzed=len(diagnostics),
         counterfactual_runs=len(results), selected_profiles=len(selected), diagnostic_only=True,
         thresholds_changed=False, binary_sha256=binary_hash))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
