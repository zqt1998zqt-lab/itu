#!/usr/bin/env python3
"""Paired frozen r5 Speex/WebRTC NS analysis by continuous/event noise class."""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

import run_audio_eval as meter

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "build/ns-recorded-repair-v4-20261018"
TAXONOMY = PROJECT / "build/ns-noise-taxonomy-20261018/samples.jsonl"
VARIANTS = ("speex_v4_balanced", "webrtc_v4_balanced")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def key(row):
    return row["original_run"], row["scene"]["id"]


def stats(values):
    a = np.array([v for v in values if v is not None], dtype=float)
    if not len(a):
        return dict(n=0, median=None, mean=None, p05=None, p95=None, minimum=None, maximum=None)
    if not np.isfinite(a).all():
        raise ValueError("nonfinite metric")
    return dict(n=len(a), median=float(np.median(a)), mean=float(a.mean()),
                p05=float(np.percentile(a, 5)), p95=float(np.percentile(a, 95)),
                minimum=float(a.min()), maximum=float(a.max()))


def non_attenuation_pass(row):
    # A diagnostic axis only; historical acceptance is not rewritten.
    return not any(not f.endswith(":attenuation") for f in row["failures"] + row["raw_failures"])


def snr_band(snr):
    return "negative" if snr < 0 else "0_to_9" if snr < 10 else "10_to_19" if snr < 20 else "20_to_30"


def summarize(rows):
    result = dict(n=len(rows), original_pass=sum(r["ns_tradeoff_pass"] for r in rows),
                  original_raw_pass=sum(r["raw_quality_pass"] for r in rows),
                  non_attenuation_checks_pass=sum(non_attenuation_pass(r) for r in rows),
                  failures=dict(Counter(f for r in rows for f in set(r["failures"] + r["raw_failures"]))))
    fields = ("mean_stoi_improvement", "mean_sisdr_improvement_db", "mean_noise_attenuation_db",
              "onset_0_50ms_projection_db", "pesq_first")
    result["metrics"] = {f: stats([r[f] for r in rows]) for f in fields}
    for window in ("0_50ms", "50_150ms", "150_500ms"):
        # Equal weight per case, averaging its available onsets first.
        values = []
        for r in rows:
            a = [o[window] for o in r["onsets"] if o[window] is not None]
            values.append(float(np.mean(a)) if a else None)
        result["metrics"]["onset_" + window] = stats(values)
    result["clipped_cases"] = sum(r["engine"]["clipped_samples_before_write"] > 0 for r in rows)
    result["runtime_note"] = "Frozen unpaced batch timings are not real-time acceptance."
    return result


def frame_slices(before, after, rate):
    hop = rate // 100
    n = len(before) // hop
    b, a = (x[:n * hop].reshape(n, hop) for x in (before, after))
    p, q = (np.mean(x * x, axis=1) for x in (b, a))
    top = np.argsort(p, kind="stable")[-max(1, int(np.ceil(.1 * n))):]
    remaining = np.ones(n, dtype=bool)
    remaining[top] = False
    exact = (p > 1e-12) & (q > 1e-12)
    per_frame = 10 * np.log10(p[exact] / q[exact])
    return dict(frames=n, top10_input_energy_percent=float(100 * p[top].sum() / p.sum()),
                top10=meter.attenuation_measurement(b[top].ravel(), a[top].ravel(), pcm16=True),
                other90=meter.attenuation_measurement(b[remaining].ravel(), a[remaining].ravel(), pcm16=True),
                all_frames=meter.attenuation_measurement(before, after, pcm16=True),
                exact_per_frame_attenuation=stats(per_frame),
                exact_per_frame_count=int(exact.sum()),
                frame_count_without_exact_attenuation=int(n - exact.sum()))


def analyze_frames(task):
    parent, paired, legacy = task
    source = PROJECT / "build" / parent["original_run"] / "cases" / parent["scene_id"]
    before_path, clean_path = source / "before.wav", source / "clean.wav"
    if sha(before_path) != parent["files"]["before.wav"]["sha256"] or sha(clean_path) != parent["files"]["clean.wav"]["sha256"]:
        raise ValueError("source PCM changed")
    before, rate = sf.read(before_path, dtype="float64")
    clean, cr = sf.read(clean_path, dtype="float64")
    if cr != rate or rate != parent["sample_rate"] or before.shape != clean.shape:
        raise ValueError("source geometry differs")
    before = np.clip(np.rint(before * 32768), -32768, 32767) / 32768
    result = dict(original_run=parent["original_run"], scene_id=parent["scene_id"],
                  noise_class=parent["noise_class"], noise=parent["noise"],
                  sample_rate=rate, split=parent["split"], snr_db=parent["snr_db"], variants={})
    for variant in VARIANTS:
        r = paired[variant]
        old = legacy[(variant, key(r))]
        if any(r[f] != old[f] for f in ("input_sha256", "clean_sha256", "output_sha256")) or r["engine"]["delay_samples"] != old["engine"]["delay_samples"]:
            raise ValueError("retained WAV is not the frozen r5 signal")
        wav = ROOT / "final-vehicle" / variant / r["original_run"] / r["scene"]["id"] / "result.after.wav"
        if sha(wav) != r["output_sha256"]:
            raise ValueError("retained WAV hash differs")
        output, ar = sf.read(wav, dtype="float64")
        delay = r["engine"]["delay_samples"]
        if ar != rate or len(output) != len(before) + delay:
            raise ValueError("output fixed-delay geometry differs")
        output = output[delay:]
        windows = {}
        for label, (lo, hi) in r["scene"]["windows"].items():
            if not label.startswith("noise_only"):
                continue
            sl = slice(round(lo * rate), round(hi * rate))
            if np.count_nonzero(clean[sl]):
                raise ValueError("target speech in pure noise window")
            item = frame_slices(before[sl], output[sl], rate)
            expected = r["segments"][label]
            for field, value in item["all_frames"].items():
                if (value is None) != (expected[field] is None) or (value is not None and abs(value - expected[field]) > 1e-9):
                    raise ValueError("frozen noise measurement was not reproduced")
            windows[label] = item
        result["variants"][variant] = dict(windows=windows, output_sha256=r["output_sha256"])
    return result


def frame_summary(records):
    result = {}
    for variant in VARIANTS:
        # One common 22-24 s window per case; extra step windows stay in the ledger.
        windows = [r["variants"][variant]["windows"]["noise_only"] for r in records]
        result[variant] = dict(cases=len(windows),
            top10_input_energy_percent=stats([w["top10_input_energy_percent"] for w in windows]),
            top10_attenuation_db=stats([w["top10"]["total_attenuation_db"] for w in windows]),
            top10_lower_bound_db=stats([w["top10"]["attenuation_lower_bound_db"] for w in windows]),
            other90_attenuation_db=stats([w["other90"]["total_attenuation_db"] for w in windows]),
            other90_lower_bound_db=stats([w["other90"]["attenuation_lower_bound_db"] for w in windows]))
    return result


def paired_deltas(rows):
    result = {}
    for field, tie in (("mean_sisdr_improvement_db", .1), ("mean_stoi_improvement", .001),
                       ("mean_noise_attenuation_db", .1)):
        values = [r[VARIANTS[0]][field] - r[VARIANTS[1]][field] for r in rows
                  if r[VARIANTS[0]][field] is not None and r[VARIANTS[1]][field] is not None]
        result[field] = dict(speex_minus_webrtc=stats(values), speex_greater=sum(v > tie for v in values),
                             webrtc_greater=sum(v < -tie for v in values), within_tolerance=sum(abs(v) <= tie for v in values),
                             comparison_tolerance=tie)
    return result


def self_check():
    t = np.arange(16000) / 16000
    before = .1 * np.sin(2 * np.pi * 300 * t)
    a = frame_slices(before, before * .5, 16000)
    assert abs(a["top10"]["total_attenuation_db"] - 6.020599913279624) < 1e-9
    assert abs(a["other90"]["total_attenuation_db"] - 6.020599913279624) < 1e-9
    quiet = frame_slices(before, np.zeros_like(before), 16000)
    assert quiet["top10"]["total_attenuation_db"] is None
    assert quiet["top10"]["attenuation_lower_bound_db"] is not None
    bad = dict(failures=["noise_only:attenuation"], raw_failures=["near_only:raw_sisdr"])
    assert not non_attenuation_pass(bad)
    assert non_attenuation_pass(dict(failures=["noise_only:attenuation"], raw_failures=[]))
    return dict(half_gain_6db_control=True, zero_output_keeps_quantization_bound=True,
                speech_failure_not_hidden_by_diagnostic=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, choices=range(1, 9), default=4)
    args = parser.parse_args()
    checks = self_check()
    paths = [TAXONOMY, Path(__file__), Path(meter.__file__)]
    for prefix in ("final-r5-", "final-"):
        for scope in ("vehicle", "clean"):
            paths.extend(ROOT / (prefix + scope) / name for name in ("results.jsonl", "identity.json", "report.json"))
    binary = ROOT / "artifacts/final-r5/ns_vehicle_probe"
    paths.append(binary)
    before_hashes = {str(path): sha(path) for path in paths}
    labels = {(p["original_run"], p["scene_id"]): p for p in load(TAXONOMY)}
    all_rows = {}
    pairs = {}
    for scope in ("vehicle", "clean"):
        report = json.loads((ROOT / ("final-r5-" + scope) / "report.json").read_text())
        if report["state"] != "completed" or not report["identity_verified"]:
            raise ValueError("r5 batch incomplete or identity unverified")
        selected = [r for r in load(ROOT / ("final-r5-" + scope) / "results.jsonl") if r["variant"] in VARIANTS]
        grouped = defaultdict(dict)
        for r in selected:
            if r["variant"] in grouped[key(r)]:
                raise ValueError("duplicate case/variant")
            if r["binary_sha256"] != before_hashes[str(binary)]:
                raise ValueError("r5 processor identity differs")
            grouped[key(r)][r["variant"]] = r
        for k, pair in grouped.items():
            if set(pair) != set(VARIANTS):
                raise ValueError("missing paired algorithm")
            a, b = (pair[v] for v in VARIANTS)
            if any(a[f] != b[f] for f in ("input_sha256", "clean_sha256", "scene")):
                raise ValueError("paired inputs differ")
            p = labels[k]
            if a["clean_sha256"] != p["files"]["clean.wav"]["sha256"]:
                raise ValueError("taxonomy target differs")
            if scope == "vehicle" and a["input_sha256"] != p["files"]["before.wav"]["sha256"]:
                raise ValueError("taxonomy input differs")
        all_rows[scope], pairs[scope] = selected, grouped
    if set(pairs["vehicle"]) != set(labels):
        raise ValueError("taxonomy does not cover the entire vehicle matrix")
    args.out.mkdir(parents=True, exist_ok=False)
    old = {(r["variant"], key(r)): r for r in load(ROOT / "final-vehicle/results.jsonl")}
    tasks = [(p, pairs["vehicle"][k], old) for k, p in sorted(labels.items())]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        frame_rows = list(pool.map(analyze_frames, tasks))
    with (args.out / "noise-frames.jsonl").open("w") as stream:
        for r in frame_rows:
            stream.write(json.dumps(r, allow_nan=False) + "\n")
    groups = {}
    for scope, selected in all_rows.items():
        groups[scope] = {}
        group_keys = ("clean_control",) if scope == "clean" else ("steady", "transient")
        for c in group_keys:
            rr = [r for r in selected if scope == "clean" or labels[key(r)]["noise_class"] == c]
            pp = [p for k, p in pairs[scope].items() if scope == "clean" or labels[k]["noise_class"] == c]
            groups[scope][c] = dict(algorithms={v: summarize([r for r in rr if r["variant"] == v]) for v in VARIANTS},
                                    paired=paired_deltas(pp))
            if scope == "vehicle":
                groups[scope][c]["pure_noise_frames"] = frame_summary([r for r in frame_rows if r["noise_class"] == c])
    strata = {}
    selected = all_rows["vehicle"]
    for axis in ("noise", "sample_rate", "snr_band", "split"):
        axis_key = lambda r: snr_band(r["scene"]["snr_db"]) if axis == "snr_band" else str(r["scene"][axis])
        strata[axis] = {value: {v: summarize([r for r in selected if axis_key(r) == value and r["variant"] == v])
                                for v in VARIANTS} for value in sorted({axis_key(r) for r in selected})}
    fields = ("variant", "original_run", "scene_id", "noise_class", "noise", "sample_rate", "snr_db", "split",
              "original_pass", "original_raw_pass", "non_attenuation_checks_pass", "stoi_delta", "sisdr_delta_db", "noise_attenuation_db", "onset_0_50_db")
    with (args.out / "paired-cases.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for k, pair in sorted(pairs["vehicle"].items()):
            for v in VARIANTS:
                r = pair[v]
                writer.writerow(dict(variant=v, original_run=k[0], scene_id=k[1], noise_class=labels[k]["noise_class"],
                    **{f:r["scene"][f] for f in ("noise", "sample_rate", "snr_db", "split")},
                    original_pass=r["ns_tradeoff_pass"], original_raw_pass=r["raw_quality_pass"],
                    non_attenuation_checks_pass=non_attenuation_pass(r), stoi_delta=r["mean_stoi_improvement"],
                    sisdr_delta_db=r["mean_sisdr_improvement_db"], noise_attenuation_db=r["mean_noise_attenuation_db"], onset_0_50_db=r["onset_0_50ms_projection_db"]))
    after_hashes = {str(path): sha(path) for path in paths}
    if before_hashes != after_hashes:
        raise ValueError("frozen bindings changed during analysis")
    report = dict(state="completed", generated_at=datetime.now(timezone.utc).isoformat(), groups=groups,
        strata=strata, unique_vehicle_pairs=len(labels), actual_pcm_verified=2 * len(frame_rows),
        self_checks=checks, input_and_target_identities_match=True, frozen_files_unchanged=True,
        variants=VARIANTS, variant_configs={v:next(r["config"] for r in selected if r["variant"] == v) for v in VARIANTS},
        limitations=["Complete optimized NS branches, including different controllers/high-pass settings; not isolated backend comparison.",
                     "Historical >=6 dB pure-noise gates retained; non-attenuation checks are a diagnostic axis, not new acceptance.",
                     "Steady and transient source/SNR/rate distributions differ; compare algorithms within a class.",
                     "24 kHz shared path bridges through 48 kHz for both branches; not a native 24 kHz backend claim.",
                     "Retrospective frozen simulations, not new blind data, subjective listening or ITU certification.",
                     "Loudest 10% frames are energy-ranked within a pure-noise window, not semantic event ground truth."])
    save(args.out / "report.json", report)
    save(args.out / "identity.json", dict(input_sha256=before_hashes,
         output_sha256={p.name:sha(p) for p in args.out.iterdir() if p.is_file()}, binary_sha256=sha(binary)))
    for scope in groups:
        for c, g in groups[scope].items():
            for v, s in g["algorithms"].items():
                print(scope, c, v, "n", s["n"], "pass/raw/speech", s["original_pass"], s["original_raw_pass"], s["non_attenuation_checks_pass"],
                      "median", {f:round(m["median"], 6) if m["median"] is not None else None for f,m in s["metrics"].items()}, flush=True)


if __name__ == "__main__":
    main()
