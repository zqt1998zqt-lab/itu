#!/usr/bin/env python3
"""Add a two-class, source-traceable noise taxonomy to frozen NS evaluations.

This is an evaluation sidecar. It never changes WAVs, frozen manifests, DSP
settings, acceptance gates, or historical results. The two classes describe
continuous background versus event-like noise, not strict statistical stationarity.
"""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import median_filter

PROJECT = Path(__file__).resolve().parents[2]
SCOPES = ("vehicle", "clean", "cold", "native24")
LABELS = {"steady": "稳态噪声", "transient": "瞬时噪声"}
FAMILIES = {
    "tcar": ("steady", "recorded_continuous_background"),
    "cavemove_alfaromeo_146_s60_w0": ("steady", "recorded_continuous_background"),
    "cavemove_smart_forfour_s60_w0": ("steady", "recorded_continuous_background"),
    "cavemove_volkswagen_golf_s60_w0": ("steady", "recorded_continuous_background"),
    "sim_engine_ramp": ("steady", "slowly_varying_engine_background"),
    "sim_wind_gust": ("transient", "recurring_wind_gusts"),
    "sim_rain_impulse": ("transient", "dense_impulses_over_background"),
    "sim_indicator": ("transient", "periodic_short_clicks_over_background"),
}
EVENT_RULE = dict(frame_ms=10, local_median_ms=1010, excess_db=6, minimum_peak_excess_db=12,
                  minimum_duration_ms=10, maximum_duration_ms=500,
                  interpretation="candidate short energy excursions, not semantic ground truth")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_rows(path, rows):
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def key(row):
    return row["original_run"], row["scene"]["id"]


def waveform_features(audio, rate):
    if rate not in (16000, 24000, 48000) or audio.ndim != 1:
        raise ValueError("expected mono noise at a supported native sample rate")
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("invalid noise PCM")
    hop = rate // 100
    count = len(audio) // hop
    if count < 101:
        raise ValueError("noise fixture too short for the declared 1.01 s analysis")
    frames = audio[:count * hop].reshape(count, hop)
    powers = np.mean(frames * frames, axis=1)
    total = float(powers.sum())
    if total <= 1e-25:
        raise ValueError("silent noise component cannot be assigned a noise class")
    levels = 10 * np.log10(np.maximum(powers, 1e-30))
    baseline = median_filter(levels, size=101, mode="nearest")
    excess = levels - baseline
    candidate = excess >= EVENT_RULE["excess_db"]
    padded = np.r_[False, candidate, False].astype(np.int8)
    starts = np.flatnonzero(np.diff(padded) == 1)
    stops = np.flatnonzero(np.diff(padded) == -1)
    events = []
    for start, stop in zip(starts, stops):
        if 1 <= stop - start <= 50 and excess[start:stop].max() >= EVENT_RULE["minimum_peak_excess_db"]:
            events.append(dict(start_s=int(start) / 100, end_s=int(stop) / 100,
                               duration_ms=int(stop - start) * 10,
                               peak_excess_db=float(excess[start:stop].max())))
    top = max(1, int(np.ceil(count * .1)))
    return dict(frame_count=count, rms_dbfs=float(10 * np.log10(total / count)),
                frame_level_p10_dbfs=float(np.percentile(levels, 10)),
                frame_level_p90_dbfs=float(np.percentile(levels, 90)),
                top10_frame_energy_percent=float(100 * np.sort(powers)[-top:].sum() / total),
                candidate_short_event_count=len(events), candidate_short_events=events,
                candidate_detection=EVENT_RULE)


def classify_parent(row):
    run, scene_id = key(row)
    origin = PROJECT / "build" / run / "cases" / scene_id
    scene_path = origin / "scene.json"
    scene_hash = digest(scene_path)
    scene = json.loads(scene_path.read_text())
    for field in ("id", "noise", "sample_rate", "snr_db", "split"):
        if scene[field] != row["scene"][field]:
            raise ValueError(f"scene metadata differs: {scene_id}:{field}")
    if scene["mics"] != 1:
        raise ValueError("this taxonomy version audits the mono NS evaluation only")
    source_noise = scene["noise"]
    if source_noise not in FAMILIES:
        raise ValueError("unreviewed noise family: " + source_noise)
    bindings = {}
    for name, row_hash in (("before.wav", row["input_sha256"]),
                           ("clean.wav", row["clean_sha256"]),
                           ("noise-component.wav", None)):
        path = origin / name
        actual = digest(path)
        if actual != scene["files"][name]["sha256"] or (row_hash and actual != row_hash):
            raise ValueError("input hash mismatch: " + str(path))
        bindings[name] = dict(path=str(path), sha256=actual)
    audio, rate = sf.read(origin / "noise-component.wav", dtype="float64", always_2d=True)
    if rate != scene["sample_rate"] or audio.shape != (round(scene["seconds"] * rate), 1):
        raise ValueError("noise rate/duration/channel geometry differs")
    features = waveform_features(audio[:, 0], rate)
    noise_class, subtype = FAMILIES[source_noise]
    transitions = []
    if scene.get("noise_steps"):
        subtype = "continuous_background_with_level_step"
        transitions.extend(dict(kind="persistent_noise_level_step", at_s=step["time_s"],
                                gain_db=step["gain_db"],
                                interpretation="a changed background level persists; not a short impulse")
                           for step in scene["noise_steps"])
    if digest(scene_path) != scene_hash:
        raise ValueError("scene changed during classification")
    return dict(original_run=run, scene_id=scene_id, noise=source_noise,
                noise_class=noise_class, noise_class_zh=LABELS[noise_class], noise_subtype=subtype,
                sample_rate=rate, split=scene["split"], snr_db=scene["snr_db"],
                near_speaker=scene["near_speaker"], duration_s=scene["seconds"],
                near_schedule=scene["near_schedule"], windows=scene["windows"],
                scene_path=str(scene_path), scene_sha256=scene_hash, files=bindings,
                noise_source=scene.get("noise_source"), transitions=transitions,
                classification_basis="reviewed source construction and provenance; waveform excursions are auxiliary",
                has_candidate_short_events=bool(features["candidate_short_event_count"]),
                waveform=features)


def taxonomy_for(scope, row, parent):
    # clean.wav is the actual input in this lane: never inherit its noisy parent's class.
    if scope == "clean":
        return dict(version=1, noise_present=False, noise_class=None, noise_class_zh=None,
                    sample_role="clean_speech_control", parent_noise_class=parent["noise_class"])
    crop = 0.0
    if scope == "cold":
        path = PROJECT / "build" / row["original_run"] / "cases" / row["scene"]["id"] / "clean.wav"
        audio, rate = sf.read(path, dtype="float64")
        hop = rate // 100
        p = np.mean(audio[:len(audio) // hop * hop].reshape(-1, hop) ** 2, axis=1)
        active = np.flatnonzero(p >= 10 ** (-65 / 10))
        if not len(active):
            raise ValueError("cold input has no valid target onset")
        crop = int(active[0]) / 100
    events = [dict(event, start_s=max(0., event["start_s"] - crop),
                   end_s=event["end_s"] - crop,
                   duration_ms=round(1000 * (event["end_s"] - max(crop, event["start_s"]))))
              for event in parent["waveform"]["candidate_short_events"] if event["end_s"] > crop]
    return dict(version=1, noise_present=True, noise_class=parent["noise_class"],
                noise_class_zh=parent["noise_class_zh"], noise_subtype=parent["noise_subtype"],
                sample_role="cold_start_noisy_speech" if scope == "cold" else "noisy_speech",
                parent_key=list(key(row)), source_time_offset_s=crop,
                candidate_short_events=events, candidate_detection=EVENT_RULE,
                transitions=[dict(t, at_s=t["at_s"] - crop) for t in parent["transitions"] if t["at_s"] >= crop])


def self_check():
    rate = 16000
    t = np.arange(3 * rate) / rate
    steady = .02 * np.sin(2 * np.pi * 100 * t)
    burst = steady.copy()
    start, end = rate, rate + rate // 50
    burst[start:end] += .8 * np.sin(2 * np.pi * 1800 * t[start:end])
    assert waveform_features(steady, rate)["candidate_short_event_count"] == 0
    assert waveform_features(burst, rate)["candidate_short_event_count"] > 0
    assert waveform_features(steady, rate)["top10_frame_energy_percent"] < 11
    assert waveform_features(burst, rate)["top10_frame_energy_percent"] > 80
    for native_rate in (24000, 48000):
        tt = np.arange(3 * native_rate) / native_rate
        assert waveform_features(.02 * np.sin(2 * np.pi * 100 * tt), native_rate)["candidate_short_event_count"] == 0
    try:
        waveform_features(np.zeros(3 * rate), rate)
    except ValueError:
        pass
    else:
        raise AssertionError("silent negative control unexpectedly accepted")
    assert FAMILIES["sim_engine_ramp"][0] == "steady"
    assert FAMILIES["sim_indicator"][0] == "transient"
    return dict(steady_control_pass=True, injected_20ms_event_detected=True,
                silent_negative_control_rejected=True, native_rates_checked=[16000, 24000, 48000])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path,
                        default=PROJECT / "build/ns-recorded-repair-v4-20261018")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, choices=range(1, 9), default=4)
    args = parser.parse_args()
    checks = self_check()
    inputs = {scope: args.results_root / ("final-" + scope) / "results.jsonl" for scope in SCOPES}
    protected = [args.results_root / ("final-" + scope) / name
                 for scope in SCOPES for name in ("identity.json", "report.json", "results.jsonl")]
    before_hashes = {str(p): digest(p) for p in protected}
    rows = {scope: read_rows(path) for scope, path in inputs.items()}
    unique = {}
    for row in rows["vehicle"]:
        k = key(row)
        if k in unique:
            previous = unique[k]
            if any(previous[field] != row[field] for field in ("scene", "input_sha256", "clean_sha256")):
                raise ValueError("variant input/scene differs; cannot share a label")
        else:
            unique[k] = row
    if not unique:
        raise ValueError("empty vehicle inventory")
    args.out.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        parents = list(pool.map(classify_parent, unique.values()))
    lookup = {(p["original_run"], p["scene_id"]): p for p in parents}
    cold_cache = {}
    coverage = {}
    for scope in SCOPES:
        enriched = []
        for row in rows[scope]:
            p = lookup[key(row)]  # Missing parents are fatal, never silently omitted.
            if row["clean_sha256"] != p["files"]["clean.wav"]["sha256"]:
                raise ValueError("scope clean target differs")
            cache_key = (scope, key(row))
            if cache_key not in cold_cache:
                cold_cache[cache_key] = taxonomy_for(scope, row, p)
            if scope == "clean":
                expected = p["files"]["clean.wav"]["sha256"]
            elif scope != "cold":
                expected = p["files"]["before.wav"]["sha256"]
            else:
                cold_path = args.results_root / "final-cold" / row["variant"] / row["original_run"] / row["scene"]["id"] / "cold-input.wav"
                expected = digest(cold_path)
                info = sf.info(cold_path)
                offset_s = cold_cache[cache_key]["source_time_offset_s"]
                if info.samplerate != p["sample_rate"] or info.channels != 1 or info.frames != round((p["duration_s"] - offset_s) * info.samplerate):
                    raise ValueError("cold crop geometry differs")
            if row["input_sha256"] != expected:
                raise ValueError("scope input differs")
            enriched.append(dict(row, noise_taxonomy=cold_cache[cache_key]))
        write_rows(args.out / f"results-{scope}.jsonl", enriched)
        single = {key(r): r for r in enriched}
        coverage[scope] = dict(result_records=len(enriched), unique_parent_scenes=len(single),
                               classes=dict(Counter(r["noise_taxonomy"]["noise_class"] or "clean_control" for r in single.values())))
    parents.sort(key=lambda p: (p["original_run"], p["scene_id"]))
    for noise_class in LABELS:
        write_rows(args.out / f"{noise_class}.jsonl", [p for p in parents if p["noise_class"] == noise_class])
    write_rows(args.out / "samples.jsonl", parents)
    fields = ("original_run", "scene_id", "noise_class", "noise_class_zh", "noise_subtype",
              "noise", "sample_rate", "split", "snr_db", "near_speaker", "has_candidate_short_events")
    with (args.out / "samples.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: p[field] for field in fields} for p in parents)
    after_hashes = {str(p): digest(p) for p in protected}
    if before_hashes != after_hashes:
        raise ValueError("frozen evaluation bindings changed while classifying")
    report = dict(state="completed", generated_at=datetime.now(timezone.utc).isoformat(),
                  primary_vehicle_scenes=len(parents),
                  primary_classes=dict(Counter(p["noise_class"] for p in parents)),
                  counts_by_noise={name: dict(noise_class=FAMILIES[name][0],
                                             scenes=sum(p["noise"] == name for p in parents)) for name in FAMILIES},
                  source_family_rules={name: dict(noise_class=v[0], subtype=v[1]) for name, v in FAMILIES.items()},
                  class_definition="steady = continuous/slowly varying background; transient = event-like source; not strict statistical stationarity",
                  candidate_rule=EVENT_RULE,
                  parents_with_candidate_short_events=sum(p["has_candidate_short_events"] for p in parents),
                  noise_step_parents=sum(bool(p["transitions"]) for p in parents),
                  per_scope=coverage, self_check=checks, frozen_bindings_unchanged=True,
                  input_audio_modified=False, acceptance_gates_modified=False,
                  scope="Latest 847-scene mono NS comparison and its clean/cold/native24 lanes; not the historical full-project dataset")
    save(args.out / "report.json", report)
    save(args.out / "identity.json", dict(script_sha256=digest(Path(__file__)),
         frozen_evaluation_files=before_hashes,
         source_generator_sha256=digest(PROJECT / "cpp/tools/run_controlled_native_scenes.py"),
         output_sha256={p.name: digest(p) for p in sorted(args.out.iterdir()) if p.is_file()},
         parent_scene_audio_bindings="samples.jsonl: files and scene_sha256"))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
