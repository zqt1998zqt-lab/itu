#!/usr/bin/env python3
"""Audit all active input datasets and retained source inventories; never read enhanced audio."""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "cpp/tools"))
from server.quality.dataset import aggregate_assets, inspect_asset
from server.quality.evaluators import create_evaluators
from server.quality.metrics import sha
from cpp.tools.ecnr_perceptual import save_json
from audit_dataset_quality_deep import check_scene, negative_controls, read_inventory

SCORERS = []


def file_sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def object_sha(value):
    return sha(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def read(path):
    return json.loads(Path(path).read_text())


def inventory():
    scenes, bindings = read_inventory(PROJECT)
    for name in ("controlled-native-full-20261017", "controlled-cavemove-20261017"):
        run = PROJECT / "build" / name
        for filename in ("results.jsonl", "identity.json"):
            bindings[str(run / filename)] = file_sha(run / filename)
        for line in (run / "results.jsonl").open():
            # Existing result files provide frozen input metadata only. No output metrics are read.
            scene = json.loads(line)["scene"]
            scenes.append((name, run / "cases" / scene["id"], scene))
    assets, links, sources = {}, [], []

    def add(path, identity, role, rate, intervals, use, channels=None, samples=None):
        key = dict(sha256=identity, role=role, sample_rate=rate, speech_intervals=intervals)
        identifier = object_sha(key)[:32]
        if identifier not in assets:
            assets[identifier] = dict(asset_id=identifier, path=str(Path(path).resolve()),
                                     channels=channels, samples=samples, **key)
        links.append(dict(asset_id=identifier, path=str(Path(path).resolve()), **use))

    for label, folder, scene in scenes:
        for name in ("clean.wav", "clean-mics.wav", "clean-beam.wav"):
            if name not in scene["files"]:
                continue
            add(folder / name, scene["files"][name]["sha256"], "target_reference", scene["sample_rate"],
                scene["near_schedule"], dict(dataset=label, scene_id=scene["id"], view=name,
                    near_speaker=scene["near_speaker"], split=scene.get("split"),
                    family=scene["family"], noise=scene["noise"], snr_db=scene.get("snr_db"),
                    near_rt60_s=scene.get("near_rt60_s"), track=scene.get("quality_track")))
    for label, path in (
        ("legacy_source_inventory", PROJECT / "datasets/itu-eval-current/sources/manifest.json"),
        ("native_source_inventory_v5", PROJECT / "datasets/itu-native-rate-identity-v5-20261017/manifest.json")):
        path = path.resolve()
        bindings[str(path)] = file_sha(path)
        for source in read(path)["sources"]:
            source_path = (path.parent / source["path"]).resolve()
            role = {"noise": "noise_source", "speech_candidate": "candidate_speech_inventory"}.get(source["kind"], "speech_source")
            use = dict(dataset=label, speaker=source.get("canonical_speaker_id", source.get("speaker")),
                       recording_id=source.get("recording_id"), split=source.get("split"),
                       eligibility=source.get("eligibility"), vehicle_scope=source.get("vehicle_scope"),
                       language=source.get("language"), historical_resampling=source.get("sample_rate_conversion"))
            add(source_path, source["sha256"], role, source["sample_rate"], None, use,
                channels=source.get("channels"), samples=source.get("samples"))
            sources.append(dict(path=str(source_path), sha256=source["sha256"], **use))
    path = PROJECT / "datasets/independent-fullband-20261012/dataset-manifest.json"
    bindings[str(path)] = file_sha(path)
    for source in read(path)["files"]:
        for prefix, label, role, rate_key in (
            ("native", "independent_fullband_original", "speech_source", "native_sample_rate_hz"),
            ("derived_24k", "historical_fullband_24k_derived", "historical_derived_speech", None)):
            source_path = path.parent / source[prefix + "_path"]
            rate = source[rate_key] if rate_key else 24000
            use = dict(dataset=label, speaker=source.get("speaker_id"), language=source.get("language"),
                       historical_resampling=prefix != "native", source_id=source["id"])
            add(source_path, source[prefix + "_sha256"], role, rate, None, use,
                channels=source.get("native_channels", 1) if prefix == "native" else 1,
                samples=source.get(prefix + "_samples"))
            sources.append(dict(path=str(source_path.resolve()), sha256=source[prefix + "_sha256"], **use))
    return scenes, bindings, list(assets.values()), links, sources


def verify_source(item):
    path = Path(item["path"])
    return dict(**item, error=None if path.is_file() and file_sha(path) == item["sha256"] else "missing_or_file_sha_mismatch")


def initialize(cache):
    global SCORERS
    SCORERS = create_evaluators(["dnsmos", "scoreq"], Path(cache))


def score_asset(task):
    spec, folder, model_identity = task
    destination = Path(folder) / (spec["asset_id"] + ".json")
    binding = dict(asset_spec_sha256=object_sha(spec), evaluator_identities_sha256=model_identity)
    if destination.exists():
        row = read(destination)
        actual = object_sha({k: v for k,v in row.items() if k != "evidence_sha256"})
        if row.get("binding") != binding or row.get("evidence_sha256") != actual:
            raise ValueError("existing dataset evidence identity differs: " + str(destination))
        return row
    try:
        row = inspect_asset(spec, SCORERS)
    except Exception as error:
        row = dict(asset_id=spec["asset_id"], role=spec["role"], sample_rate=spec["sample_rate"],
                   state="error", error=type(error).__name__ + ": " + str(error), source_path=spec["path"])
    row["binding"] = binding
    row["evidence_sha256"] = object_sha(row)
    save_json(destination, row)
    return row


def distributions(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row["state"] != "completed":
            continue
        values = defaultdict(list)
        for channel in row["channels"]:
            for window in channel["windows"]:
                for model, score in window["perceptual"].items():
                    if score["available"]:
                        for dimension, value in score["values"].items():
                            values[f"{model}:{dimension}"].append(value)
        for name, items in values.items():
            grouped[f"{row['role']}:{row['sample_rate']}:{name}"].append(
                dict(asset_id=row["asset_id"], source_path=row["source_path"], mean_score=float(np.mean(items))))
    result = {}
    for key, values in grouped.items():
        values.sort(key=lambda v: v["mean_score"])
        scores = [v["mean_score"] for v in values]
        result[key] = dict(assets=len(values), quantiles=dict(zip(("min","p05","median","p95","max"),
                           map(float, np.percentile(scores, [0,5,50,95,100])))),
                           lowest_20=values[:20], interpretation="same-model relative review priorities, not a deletion or ITU pass threshold")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, choices=range(1,13), default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if not args.resume:
        root.mkdir(parents=True, exist_ok=False)
    elif not (root / "identity.json").is_file():
        raise ValueError("resume needs an existing identity")
    start = time.monotonic()
    scenes, bindings, assets, links, sources = inventory()
    for path in (Path(__file__), PROJECT / "server/quality/dataset.py", PROJECT / "server/quality/evaluators.py",
                 PROJECT / "server/quality/metrics.py", PROJECT / "cpp/tools/audit_dataset_quality_deep.py",
                 PROJECT / "cpp/tools/ecnr_perceptual.py"):
        bindings[str(path)] = file_sha(path)
    scorers = create_evaluators(["dnsmos", "scoreq"], PROJECT / "build/quality-evaluations/cache")
    model_identity = {s.name:s.identity for s in scorers}
    identity = dict(bindings=bindings, evaluator_identities=model_identity, assets_sha256=object_sha(assets),
                    asset_uses_sha256=object_sha(links), scene_counts=dict(Counter(label for label,_,_ in scenes)),
                    source_entries=len(sources), assets=len(assets), enhanced_audio_evaluated=False)
    if args.resume:
        if read(root / "identity.json") != identity:
            raise ValueError("dataset batch identity differs; preserve prior evidence")
    else:
        save_json(root / "identity.json", identity)
        save_json(root / "assets-manifest.json", dict(assets=assets, uses=links, sources=sources))
    print(json.dumps(dict(phase="inventory", scenes=len(scenes), assets=len(assets), source_entries=len(sources),
                          scene_counts=identity["scene_counts"])), flush=True)
    progress = dict(state="running", phase="integrity", total_scenes=len(scenes), total_assets=len(assets), jobs=args.jobs)
    save_json(root / "progress.json", progress)
    controls = negative_controls(next(item for item in scenes if item[2]["mics"] == 1 and item[2]["family"] == "ns"))
    scene_rows = []
    with (root / "scenes.jsonl").open("w") as output, ThreadPoolExecutor(args.jobs) as pool:
        for i,row in enumerate(pool.map(check_scene, scenes), 1):
            output.write(json.dumps(row, allow_nan=False) + "\n")
            scene_rows.append(row)
            if i % 500 == 0 or i == len(scenes):
                output.flush()
                progress.update(completed_scenes=i, elapsed_seconds=time.monotonic()-start)
                save_json(root / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    with ThreadPoolExecutor(args.jobs) as pool:
        source_checks = list(pool.map(verify_source, sources))
    save_json(root / "source-path-checks.json", dict(entries=source_checks))
    progress.update(phase="source_and_target_quality", completed_assets=0)
    save_json(root / "progress.json", progress)
    rows = []
    (root / "assets").mkdir(exist_ok=True)
    tasks = [(a, str(root / "assets"), object_sha(model_identity)) for a in assets]
    with ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize, initargs=(str(PROJECT / "build/quality-evaluations/cache"),)) as pool:
        futures = [pool.submit(score_asset, task) for task in tasks]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 25 == 0 or len(rows) == len(assets):
                progress.update(completed_assets=len(rows), asset_errors=sum(r["state"] == "error" for r in rows),
                                elapsed_seconds=time.monotonic()-start)
                save_json(root / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    rows.sort(key=lambda r:r["asset_id"])
    groups = {}
    for name in identity["scene_counts"]:
        selected = [r for r in scene_rows if r["dataset"] == name]
        groups[name] = dict(scenes=len(selected), files_checked=sum(r.get("files_checked", 0) for r in selected),
            scenes_with_errors=sum(bool(r["errors"]) for r in selected),
            error_counts=dict(Counter(error for r in selected for error in r["errors"])),
            warning_counts=dict(Counter(error for r in selected for error in r["warnings"])),
            sample_rates=dict(Counter(r["sample_rate"] for r in selected)))
    changed = [path for path,expected in bindings.items() if file_sha(path) != expected]
    errors = [r for r in rows if r["state"] == "error"]
    source_errors = [r for r in source_checks if r["error"]]
    report = dict(schema_version=1, state="completed" if not (changed or errors or source_errors) else "completed_with_errors",
        completed_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-start,
        identity_verified=not changed, changed_bindings=changed, scope="Input datasets only; no enhanced output inspected or algorithm evaluated",
        scenes=len(scenes), scene_groups=groups, source_path_entries=len(sources), source_errors=source_errors,
        asset_errors=errors, assets=len(rows), **aggregate_assets(rows), perceptual_distributions=distributions(rows),
        negative_controls=controls, rules=dict(mixture_error_lsb_max=2, nominal_mono_snr_tolerance_db=.05,
            recorded_snr_tolerance_db=.001, clipping_abs=.999, speech_sub60hz_review_percent=5,
            speech_dc_review_dbfs=-50, neural_low_score_rule="relative tail only; no automatic deletion",
            reference_self_similarity_is_not_quality=False, native_rates_only=True,
            high_rate_neural_quality_supported=False, external_vad=False,
            model_scores_averaged=False, noise_speech_mos_applied=False,
            historical_derived_24k="existing diagnostic inventory only; no new resampling or generation"),
        artifacts=dict(identity=str(root / "identity.json"), manifest=str(root / "assets-manifest.json"),
                       scene_rows=str(root / "scenes.jsonl"), asset_folder=str(root / "assets")),
        report_bindings={str(root / "scenes.jsonl"):file_sha(root / "scenes.jsonl"),
                         str(root / "assets-manifest.json"):file_sha(root / "assets-manifest.json")})
    save_json(root / "report.json", report)
    save_json(root / "progress.json", dict(state=report["state"], phase="completed", scenes=len(scenes),
              assets=len(rows), elapsed_seconds=report["elapsed_seconds"], report=str(root / "report.json")))
    print(json.dumps(dict(state=report["state"], scenes=len(scenes), assets=len(rows),
                          elapsed_seconds=report["elapsed_seconds"], report=str(root / "report.json"))), flush=True)
    return 0 if report["state"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
