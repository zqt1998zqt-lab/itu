#!/usr/bin/env python3
"""Copy verified dataset metadata into an independent history archive; delete nothing."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.dont_write_bytecode = True


def guard():
    available = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                         if line.startswith("MemAvailable:"))) * 1024
    if shutil.disk_usage(PROJECT).free < 100 * 1024**3 or available < 5 * 1024**3:
        raise RuntimeError("resource safety floor; preserve all files")


def create_or_verify(path: Path, payload: bytes):
    guard()
    if path.is_symlink():
        raise ValueError(f"refuse symlink: {path}")
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"refuse to overwrite changed archive: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def main():
    from server.acoustic.dataset_dependency_archive import ARCHIVE_ROOT, SCHEMA, encode_metadata
    from server.acoustic import acceptance_dataset, scene_condition_batch, service
    from server.acoustic.dataset_revisions_v2 import install_from_activation, revision_scene_profile, revision_stress_profile
    from server.acoustic.dataset_revisions import install_revision_profiles as register_first_revision

    guard()
    register_first_revision(activate=False)
    install_from_activation()
    plan = json.loads((PROJECT / "build/dataset-revision-20261117/cleanup-inventory-20261117-readonly/review-manifest.json").read_text())
    entries = {}; replacements = {}; old_ids = []
    for row in plan["entries"]:
        dataset_id = row.get("dataset_id")
        if not dataset_id:
            continue
        guard()
        if row.get("kind") == "contract":
            dataset = acceptance_dataset.get_dataset(dataset_id)
            if dataset is None:
                raise ValueError(f"old contract dataset missing: {dataset_id}")
            contracts = None
            kind = "contract"
            replacement = "fa11d83bbeb6480c98a9eb3bae35045f"
        else:
            dataset, contracts = scene_condition_batch._validated_dataset(dataset_id)
            kind = "stress" if row["group"] in {"r1_stress", "old_stress_v1"} else "scene"
            replacement = (revision_stress_profile(row["condition"], row["mode"]).dataset_id
                           if kind == "stress" else revision_scene_profile(row["condition"], row["mode"]).dataset_id)
        payload = canonical(encode_metadata({
            "dataset_id": dataset_id, "kind": kind, "dataset": dataset,
            "contracts": contracts, "original_store": row["path"],
            "execution_allowed": False,
        }))
        relative = f"datasets/{dataset_id}.json"
        create_or_verify(ARCHIVE_ROOT / relative, payload)
        entries[dataset_id] = {"kind": kind, "path": relative,
                               "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
        old_ids.append(dataset_id); replacements[dataset_id] = replacement
        print(f"ARCHIVED {len(entries)}/103 {kind}/{dataset_id}", flush=True)
    manifest = canonical({"schema": SCHEMA, "datasets": entries,
                          "purpose": "hash-locked history metadata only; no new execution permission"})
    create_or_verify(ARCHIVE_ROOT / "manifest.json", manifest)
    receipt = {"schema": "itu-dataset-retirement-v1", "active_revision": "measurement-20261117-r2",
               "retired_dataset_ids": sorted(old_ids), "replacement_dataset_ids": replacements,
               "metadata_archive_manifest_sha256": hashlib.sha256(manifest).hexdigest()}
    create_or_verify(ARCHIVE_ROOT / "retirement.json", canonical(receipt))
    print(json.dumps({"metadata_archived": len(entries), "audio_copied": False, "files_deleted": 0,
                      "root": str(ARCHIVE_ROOT), "manifest_sha256": receipt["metadata_archive_manifest_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
