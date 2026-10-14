#!/usr/bin/env python3
"""Versioned, auditable CRUD for evaluation inputs; parents are never written."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import shutil
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf
import numpy as np

from audit_dataset_quality import audio_stats, digest, audit_scene

DERIVED = {"quality_track", "quality_flags", "interference_class", "split", "notes"}


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def folder(scene):
    return Path(scene["source_dataset"]) / "cases" / scene.get("source_case_id", scene["id"])


def verify_scene(scene):
    for name, identity in scene["files"].items():
        path = folder(scene) / name
        if Path(name).name != name or path.stat().st_size != identity["bytes"] or digest(path) != identity["sha256"]:
            raise ValueError(f"input identity mismatch: {scene['id']} {name}")


@contextmanager
def locked(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def load(root):
    path = root / "catalog.json"
    state = json.loads(path.read_text())
    operation_hash = root / "operations.sha256"
    if operation_hash.exists() and operation_hash.read_text().strip() != digest(root / "operations.jsonl"):
        raise ValueError("operation log content changed")
    records = [json.loads(line) for line in (root / "operations.jsonl").read_text().splitlines()]
    previous = None
    for index, record in enumerate(records):
        if record["revision"] != index + 1 or record["before_sha256"] != previous:
            raise ValueError("operation history chain broken")
        previous = record["after_sha256"]
    if not records or previous != digest(path) or state["revision"] != len(records):
        raise ValueError("catalog changed outside CRUD history")
    return state


def save(root, state, action, evidence):
    path = root / "catalog.json"
    before = digest(path) if path.exists() else None
    state["revision"] += 1
    payload = encoded(state)
    record = dict(revision=state["revision"], utc=datetime.now(timezone.utc).isoformat(),
                  action=action, before_sha256=before, after_sha256=sha(payload), evidence=evidence)
    temporary = root / "catalog.tmp"
    temporary.write_bytes(payload)
    temporary.replace(path)
    with (root / "operations.jsonl").open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    (root / "operations.sha256").write_text(digest(root / "operations.jsonl") + "\n")
    return record


def initialize(root, datasets):
    if (root / "catalog.json").exists() or (root / "operations.jsonl").exists():
        raise ValueError("use a new workspace")
    state = dict(schema_version=1, revision=0, parents=[], scenes={}, sources=[], speaker_metadata={})
    seen_sources = set()
    for dataset in datasets:
        dataset = dataset.resolve()
        manifest = dataset / "inputs-manifest.json"
        source_path = dataset / "sources/manifest.json"
        content = json.loads(manifest.read_text())
        if digest(source_path) != content["source_manifest_sha256"]:
            raise ValueError("source manifest identity mismatch")
        state["parents"].append(dict(path=str(dataset), manifest_sha256=digest(manifest),
                                     source_manifest_sha256=digest(source_path)))
        source_content = json.loads(source_path.read_text())
        state["speaker_metadata"].update(source_content.get("speaker_metadata", {}))
        for source in source_content["sources"]:
            path = (dataset / "sources" / source["path"]).resolve()
            if digest(path) != source["sha256"]:
                raise ValueError(f"source changed: {path}")
            if source["sha256"] not in seen_sources:
                state["sources"].append(dict(source, absolute_path=str(path)))
                seen_sources.add(source["sha256"])
        for original in content["scenes"]:
            scene = copy.deepcopy(original)
            key = f"{dataset.name}--{scene['id']}"
            if key in state["scenes"]:
                raise ValueError("duplicate scene")
            scene.update(id=key, source_case_id=original["id"], source_dataset=str(dataset),
                         quality_track="unreviewed", split="unassigned")
            state["scenes"][key] = dict(scene=scene, excluded=False, exclusion_reason=None)
    return state


def parse_filters(filters):
    conditions = []
    for expression in filters:
        key, separator, text = expression.partition("=")
        if not separator:
            raise ValueError("filter requires field=JSON_value or field=text")
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = text
        conditions.append((key, value))
    return conditions


def select(state, filters=(), include_excluded=False):
    conditions = parse_filters(filters)
    return [record for record in state["scenes"].values()
            if (include_excluded or not record["excluded"]) and
            all(record["scene"].get(key) == value for key, value in conditions)]


def select_sources(state, filters=()):
    conditions = parse_filters(filters)
    return [row for row in state["sources"]
            if all(row.get(key) == value for key, value in conditions)]


def curate(state, audit_path):
    report = json.loads(audit_path.read_text())
    if not report["input_integrity_passed"]:
        raise ValueError("cannot curate from a failed input audit")
    parent_hashes = {parent["manifest_sha256"] for parent in state["parents"]}
    if not {item["manifest_sha256"] for item in report["inventories"]}.issubset(parent_hashes):
        raise ValueError("audit does not match catalog parents")
    flags = {}
    for row in report["cold_target_spectrum"]:
        current = []
        if row["ac_below_60hz_percent"] > 5:
            current.append("low_frequency_over_5_percent")
        if row["dc_dbfs"] > -50:
            current.append("dc_over_minus_50_dbfs")
        flags[str(row["speaker"])] = current
    if not {str(r["scene"]["near_speaker"]) for r in state["scenes"].values()}.issubset(flags):
        raise ValueError("audit must cover every target speaker")
    # Reciprocal near/far pairs stay together. This split is retrospective.
    pairs = sorted({tuple(sorted((str(r["scene"]["near_speaker"]), str(r["scene"]["far_speaker"]))))
                    for r in state["scenes"].values()})
    assignment = {}
    for index, pair in enumerate(pairs):
        partition = "validation" if index % 5 == 0 else "holdout" if index % 5 == 1 else "development"
        for speaker in pair:
            if speaker in assignment and assignment[speaker] != partition:
                raise ValueError("overlapping near/far groups need explicit partitioning")
            assignment[speaker] = partition
    for record in state["scenes"].values():
        s = record["scene"]
        current = list(flags[str(s["near_speaker"])])
        if s["family"] == "ns":
            current.append("legacy_sweep_target_level_not_controlled")
        s["interference_class"] = ("interfering_speech" if s["noise"] in ("car_passenger", "babble")
                                   else "referenced_playback" if s.get("playback_source") == "radio"
                                   else "environmental_noise")
        if s["noise"] == "car_radio":
            record.update(excluded=True, exclusion_reason="historical radio without playback reference; retained in parent")
            current.append("historical_radio_label")
        split = assignment[str(s["near_speaker"])]
        if any(assignment.get(str(speaker)) != split for speaker in s.get("noise_speakers", [])):
            split = "cross_split_stress"
            current.append("interfering_source_crosses_split")
        s.update(quality_flags=current, quality_track="stress" if current else "quality", split=split)
    state["split_scope"] = "speaker-disjoint in all recorded roles except cross_split_stress; retrospective, not an unseen blind test"
    state["speaker_splits"] = assignment


def add_scene(state, scene_path):
    s = json.loads(scene_path.read_text())
    s.update(source_dataset=str(scene_path.parent.parent.parent.resolve()), source_case_id=s["id"])
    if s["id"] in state["scenes"]:
        raise ValueError("duplicate scene id")
    if not s.get("quality_track") or not s.get("split"):
        raise ValueError("new scenes require quality_track and split")
    verify_scene(s)
    errors = audit_scene((Path(s["source_dataset"]), s))["errors"]
    if errors:
        raise ValueError(f"new scene failed input audit: {errors}")
    state["scenes"][s["id"]] = dict(scene=s, excluded=False, exclusion_reason=None)
    return s


def validate_patch(patch, rows):
    if not isinstance(patch, dict) or not patch or set(patch) - DERIVED:
        raise ValueError("only derived metadata can be updated; regenerate PCM/SNR/windows")
    if "quality_track" in patch and patch["quality_track"] not in ("quality", "stress", "unreviewed"):
        raise ValueError("unknown quality track")
    if "split" in patch and patch["split"] not in ("development", "validation", "holdout", "cross_split_stress", "unassigned"):
        raise ValueError("unknown split")
    if "interference_class" in patch:
        for record in rows:
            s = record["scene"]
            expected = ("interfering_speech" if s["noise"] in ("car_passenger", "babble") else
                        "referenced_playback" if s.get("playback_source") == "radio" else "environmental_noise")
            if patch["interference_class"] != expected:
                raise ValueError("interference label contradicts known source roles")


def add_source(state, path, kind, speaker, language, origin, license_name, noise_name=None):
    if not origin or not license_name or (kind in ("clean_speech", "speech_candidate") and (not speaker or not language)):
        raise ValueError("source provenance and speech identity required")
    if kind == "noise" and not noise_name:
        raise ValueError("noise additions require a noise name")
    if kind == "noise" and any(source["kind"] == "noise" and
                               source.get("scene", "").lower() == noise_name.lower()
                               for source in state["sources"]):
        raise ValueError("duplicate noise name; use a distinct source key")
    data, rate = sf.read(path, always_2d=True)
    if not len(data) or not np.isfinite(data).all():
        raise ValueError("nonempty finite source required")
    if kind != "speech_candidate" and rate != 16000:
        raise ValueError("current mixture generator requires 16 kHz clean/noise; preserve native audio as a candidate")
    identity = digest(path)
    if any(source["sha256"] == identity for source in state["sources"]):
        raise ValueError("duplicate source hash")
    row = dict(absolute_path=str(path.resolve()), path=f"added/{identity[:16]}-{path.name}", sha256=identity,
               kind=kind, speaker=speaker, language=language, origin=origin, license=license_name,
               quality_status="unreviewed_not_clean_ground_truth" if kind == "speech_candidate" else "unreviewed",
               sample_rate=rate, **audio_stats(data, rate))
    if kind == "noise":
        row["scene"] = noise_name
    state["sources"].append(row)
    return row


def export(state, root, destination, filters=(), include_excluded=False, storage="external"):
    rows = select(state, filters, include_excluded)
    if not rows:
        raise ValueError("nonempty export required")
    if destination.exists():
        raise ValueError("export is frozen; use a new destination")
    for parent in state["parents"]:
        if digest(Path(parent["path"]) / "inputs-manifest.json") != parent["manifest_sha256"]:
            raise ValueError("parent manifest changed")
    for record in rows:
        verify_scene(record["scene"])
    destination.mkdir(parents=True)
    source_rows = []
    for row in state["sources"]:
        source = Path(row["absolute_path"])
        if digest(source) != row["sha256"]:
            raise ValueError("source changed before export")
        output = destination / "sources" / row["path"]
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, output)
        source_rows.append({key: value for key, value in row.items() if key != "absolute_path"})
    source_manifest = destination / "sources/manifest.json"
    source_manifest.write_bytes(encoded(dict(sources=source_rows, speaker_metadata=state["speaker_metadata"])))
    scenes = []
    for record in rows:
        original = copy.deepcopy(record["scene"])
        original["catalog_excluded"] = bool(record["excluded"])
        source_folder = folder(original)
        if storage == "copy":
            target_folder = destination / "cases" / original["id"]
            target_folder.mkdir(parents=True, exist_ok=True)
            for name in original["files"]:
                shutil.copy2(source_folder / name, target_folder / name)
            original["source_dataset"] = str(destination)
            original["source_case_id"] = original["id"]
            (target_folder / "scene.json").write_bytes(encoded(original))
        scenes.append(original)
    split_speakers = {}
    for scene in scenes:
        if scene["split"] in ("cross_split_stress", "unassigned"):
            continue
        for speaker in [scene["near_speaker"], scene["far_speaker"], *scene.get("noise_speakers", [])]:
            previous = split_speakers.setdefault(str(speaker), scene["split"])
            if previous != scene["split"]:
                raise ValueError("speaker leakage across split")
    manifest = dict(schema_version=4, profile="curated-v4", scenes=scenes,
                    source_manifest_sha256=digest(source_manifest), parents=state["parents"],
                    catalog_sha256=digest(root / "catalog.json"),
                    operations_sha256=digest(root / "operations.jsonl"),
                    manager_sha256=digest(Path(__file__)), split_scope=state.get("split_scope", "unassigned"),
                    source_audio_storage="self-contained copied scene files" if storage == "copy" else "external frozen scene files; SHA256 verified at export and replay",
                    counts=dict(scenes=len(scenes), tracks=dict(Counter(s["quality_track"] for s in scenes)),
                                splits=dict(Counter(s["split"] for s in scenes)),
                    excluded_in_catalog=sum(r["excluded"] for r in state["scenes"].values())),
                    exclusion_list=[dict(id=s["scene"]["id"], reason=s["exclusion_reason"])
                                    for s in state["scenes"].values() if s["excluded"]])
    manifest_path = destination / "inputs-manifest.json"
    manifest_path.write_bytes(encoded(manifest))
    (destination / "inputs-manifest.sha256").write_text(digest(manifest_path) + "\n")
    shutil.copy2(root / "operations.jsonl", destination / "operations.jsonl")
    if (root / "operations.sha256").exists():
        shutil.copy2(root / "operations.sha256", destination / "operations.sha256")
    return dict(path=str(destination), manifest_sha256=digest(manifest_path), **manifest["counts"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("init")
    p.add_argument("--dataset", type=Path, action="append", required=True)
    p = sub.add_parser("query")
    p.add_argument("--where", action="append", default=[])
    p.add_argument("--include-excluded", action="store_true")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--sources", action="store_true")
    p = sub.add_parser("curate")
    p.add_argument("--audit", type=Path, required=True)
    p = sub.add_parser("add")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--scene", type=Path, action="append")
    g.add_argument("--audio", type=Path)
    p.add_argument("--kind", choices=("clean_speech", "speech_candidate", "noise"))
    p.add_argument("--noise-name", help="Noise source key used by mixture generation")
    for key in ("speaker", "language", "origin", "license"):
        p.add_argument(f"--{key}")
    for action in ("update", "remove"):
        p = sub.add_parser(action)
        p.add_argument("--where", action="append", required=True)
        p.add_argument("--reason", required=True)
        if action == "update":
            p.add_argument("--set", dest="patch", type=json.loads, required=True)
    p = sub.add_parser("export")
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--where", action="append", default=[])
    p.add_argument("--include-excluded", action="store_true")
    p.add_argument("--storage", choices=("external", "copy"), default="external")
    args = parser.parse_args()
    root = args.workspace.resolve()
    with locked(root):
        if args.action == "init":
            state = initialize(root, args.dataset)
            result = save(root, state, "init", state["parents"])
        else:
            state = load(root)
            if args.action == "query":
                if args.sources:
                    rows = select_sources(state, args.where)
                    result = dict(count=len(rows), sources=rows[:args.limit])
                else:
                    rows = select(state, args.where, args.include_excluded)
                    result = dict(count=len(rows), scenes=rows[:args.limit])
            elif args.action == "export":
                result = export(state, root, args.destination.resolve(), args.where, args.include_excluded, args.storage)
            elif args.action == "curate":
                curate(state, args.audit)
                result = save(root, state, "curate", dict(audit_sha256=digest(args.audit)))
            elif args.action == "add":
                if args.scene:
                    added = [add_scene(state, p) for p in args.scene]
                    evidence = [dict(id=s["id"], files=s["files"]) for s in added]
                else:
                    if not args.kind:
                        raise ValueError("audio additions require --kind")
                    evidence = add_source(state, args.audio, args.kind, args.speaker, args.language,
                                          args.origin, args.license, args.noise_name)
                result = save(root, state, "add", evidence)
            else:
                rows = select(state, args.where)
                if not rows:
                    raise ValueError("mutation must match nonempty active selection")
                if args.action == "update":
                    validate_patch(args.patch, rows)
                for record in rows:
                    if args.action == "remove":
                        record.update(excluded=True, exclusion_reason=args.reason)
                    else:
                        record["scene"].update(args.patch)
                result = save(root, state, args.action, dict(reason=args.reason,
                    ids=[r["scene"]["id"] for r in rows], patch=getattr(args, "patch", None)))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
