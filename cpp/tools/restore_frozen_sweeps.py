#!/usr/bin/env python3
"""Reconstruct only frozen curated WAVs, refusing any nonidentical bytes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import soundfile as sf

import run_audio_eval as ev

NAMES = {"before.wav", "clean.wav", "clean-mics.wav", "clean-beam.wav",
         "noise-component.wav", "echo-component.wav", "ref.wav", "playback.wav"}
ROOTS = ("itu-eval-v4-curated-sweeps", "itu-eval-v4-curated-sweeps-rest")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def destination(project, scene):
    root = Path(scene["source_dataset"])
    allowed = [project / "datasets" / name for name in ROOTS]
    if root not in allowed or root.resolve() != root:
        raise ValueError(f"unapproved restore root: {root}")
    case = scene["source_case_id"]
    if case != scene["id"] or Path(case).name != case or case in (".", ".."):
        raise ValueError("unsafe case ID")
    folder = root / "cases" / case
    if folder.resolve() != folder:
        raise ValueError("symlinked restore destination")
    return folder


def checked_payloads(scene, target, background):
    mask = ev.mask_windows(scene["seconds"], scene["near_schedule"])
    clean = ev.level(target, -36, mask) * scene["common_gain"]
    noise = ev.level(background, -36, mask) * scene["common_gain"] * 10 ** (-scene["snr_db"] / 20)
    zeros = np.zeros_like(clean)
    arrays = {"before.wav": clean + noise, "clean.wav": clean,
              "clean-mics.wav": clean, "clean-beam.wav": clean,
              "noise-component.wav": noise, "echo-component.wav": zeros,
              "ref.wav": zeros, "playback.wav": zeros}
    if set(scene["files"]) != NAMES or scene["sample_rate"] != 16000 or scene["mics"] != 1:
        raise ValueError("unsupported frozen recipe")
    payloads = {}
    cache = {}
    for name, values in arrays.items():
        if id(values) not in cache:
            buffer = io.BytesIO()
            sf.write(buffer, values, 16000, format="WAV", subtype="PCM_16")
            cache[id(values)] = buffer.getvalue()
        payload = cache[id(values)]
        expected = scene["files"][name]
        if len(payload) != expected["bytes"] or sha(payload) != expected["sha256"]:
            raise ValueError(f"reconstruction differs: {scene['id']}/{name}")
        payloads[name] = payload
    return payloads


def write_identical(path, payload, execute):
    # Refuse changed existing files and symlinks; never replace user data.
    if path.is_symlink():
        raise ValueError(f"symlink destination: {path}")
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ValueError(f"existing file differs: {path}")
        return "existing_verified"
    if not execute:
        return "would_restore"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".restore-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # link is atomic and cannot overwrite a concurrently created target.
        os.link(temporary, path)
        if path.read_bytes() != payload:
            raise ValueError(f"post-write identity mismatch: {path}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return "restored"


def run(project, report, execute, jobs):
    started = time.monotonic()
    manifest_path = project / "datasets/itu-eval-current/inputs-manifest.json"
    original = manifest_path.read_bytes()
    scenes = json.loads(original)["scenes"]
    index = {scene["id"]: scene for scene in scenes}
    groups = defaultdict(list)
    for scene in scenes:
        if Path(scene["source_dataset"]).name in ROOTS:
            destination(project, scene)
            parents = scene["parent_components"]
            groups[(parents["clean_scene"], parents["noise_scene"])].append(scene)
    if sum(map(len, groups.values())) != 9758 or len(groups) != 238:
        raise ValueError("unexpected frozen restore inventory")
    report.mkdir(parents=True, exist_ok=False)
    (report / "tool-frozen.py").write_bytes(Path(__file__).read_bytes())
    counts = defaultdict(int)
    parent_hashes = {}

    def restore_group(rows):
        parents = rows[0]["parent_components"]
        arrays = []
        verified = {}
        for field, filename, hash_field in (("clean_scene", "clean.wav", "clean_sha256"),
                                             ("noise_scene", "noise-component.wav", "noise_sha256")):
            parent = index[parents[field]]
            path = Path(parent["source_dataset"]) / "cases" / parent["source_case_id"] / filename
            actual = ev.digest(path)
            if any(row["parent_components"][hash_field] != actual for row in rows):
                raise ValueError(f"parent differs: {path}")
            audio, rate = sf.read(path, dtype="float64")
            if rate != 16000 or audio.ndim != 1 or not np.isfinite(audio).all():
                raise ValueError(f"invalid parent PCM: {path}")
            arrays.append(audio)
            verified[str(path)] = actual
        records = []
        for scene in rows:
            folder = destination(project, scene)
            payloads = checked_payloads(scene, *arrays)
            # Verify all preexisting destinations before creating any file.
            for name, payload in payloads.items():
                if (folder / name).exists() or (folder / name).is_symlink():
                    write_identical(folder / name, payload, False)
            statuses = {name: write_identical(folder / name, payload, execute)
                        for name, payload in payloads.items()}
            records.append(dict(id=scene["id"], path=str(folder), files=statuses,
                                frozen_hashes_verified=len(payloads)))
        return records, verified

    state = "failed"
    try:
        with (report / "scenes.jsonl").open("x") as output, ThreadPoolExecutor(max_workers=jobs) as pool:
            for i, (records, parents) in enumerate(pool.map(restore_group, groups.values()), 1):
                parent_hashes.update(parents)
                for row in records:
                    output.write(json.dumps(row) + "\n")
                    counts["scenes_verified"] += 1
                    for status in row["files"].values():
                        counts[status] += 1
                output.flush()
                if i % 10 == 0 or i == len(groups):
                    print(json.dumps(dict(groups=i, total_groups=len(groups), counts=dict(counts),
                                          elapsed_s=round(time.monotonic()-started, 2))), flush=True)
        if manifest_path.read_bytes() != original:
            raise ValueError("frozen manifest changed during recovery")
        state = "complete" if execute else "verified_dry_run"
    finally:
        summary = dict(state=state, created_at=datetime.now(timezone.utc).isoformat(),
                       execute=execute, manifest_sha256=sha(original),
                       manifest_unchanged=manifest_path.read_bytes() == original,
                       tool_sha256=ev.digest(report / "tool-frozen.py"),
                       verified_parent_files=len(parent_hashes), counts=dict(counts),
                       elapsed_s=round(time.monotonic()-started, 2))
        (report / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        (report / "parent-hashes.json").write_text(json.dumps(parent_hashes, indent=2)+"\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--jobs", type=int, choices=(1, 2, 3, 4), default=4)
    args = parser.parse_args()
    run(args.project.resolve(), args.report.resolve(), args.execute, args.jobs)
