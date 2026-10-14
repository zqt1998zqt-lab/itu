#!/usr/bin/env python3
"""Remove only audited, reproducible historical DSP outputs; preview by default."""
import argparse
import hashlib
import json
import os
import stat
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "datasets/itu-eval-v4-quality-filtered-20261010/inputs-manifest.json"
FROZEN_SHA = "7da67fff6217aea11a4914c26c15e43f559cf1d53888b7378923ab51c1c3d7c4"
INPUT_CASES = ROOT / "build/dataset-quality-filtered-full-inputs-20261010/cases"
KEEP = {
    "build/radio-optimization/verified-radio-v1",
    "build/radio-optimization/verified-radio-v3",
    "build/radio-optimization/verified-general-v3",
    "build/quality-repair/baseline-v1",
    "build/quality-repair/final-verified-v1",
    "build/data-algorithm-v4/continuous-delivery",
    "build/data-algorithm-v4/final-binary-sample-verification",
    "build/data-algorithm-v4/baseline",
}
OUTPUT_SUFFIXES = (".after.wav", ".aec.wav", ".bf.wav", ".linear-aec.wav")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def signature(path):
    info = path.stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def files(directory):
    for current, directories, names in os.walk(directory, followlinks=False):
        directories[:] = [name for name in directories
                          if not (Path(current) / name).is_symlink()]
        for name in names:
            yield Path(current) / name


def protected_inventory():
    if digest(FROZEN) != FROZEN_SHA:
        raise ValueError("frozen manifest changed")
    manifest = json.loads(FROZEN.read_text())
    if len(manifest["scenes"]) != 21998:
        raise ValueError("unexpected frozen scene count")
    records = {str(FROZEN): signature(FROZEN)}
    samples = []
    for index, scene in enumerate(manifest["scenes"]):
        for name, record in scene["files"].items():
            path = INPUT_CASES / scene["id"] / name
            resolved = path.resolve(strict=True)
            identity = signature(resolved)
            if identity[2] != record["bytes"]:
                raise ValueError(f"input size mismatch: {path}")
            records[str(path)] = identity
            if index % 347 == 0 and name in ("before.wav", "clean.wav", "ref.wav"):
                samples.append((path, record["sha256"]))
    for path in files(ROOT / "public"):
        if path.suffix.lower() == ".wav":
            records[str(path)] = signature(path)
    for path in (ROOT / "cpp/evaluation/acceptance-v1.json",
                 ROOT / "build/quality-filtered-full-20261010/acceptance.json",
                 ROOT / "build/quality-filtered-full-20261010/inputs-manifest.json"):
        records[str(path)] = signature(path)
    for path, expected in samples:
        if digest(path) != expected:
            raise ValueError(f"sampled frozen PCM changed: {path}")
    return records, samples, len(manifest["scenes"])


def candidate_roots():
    yield ROOT / "datasets/itu-eval-v2-fixed/cases"
    for name in ("radio-optimization", "quality-repair", "data-algorithm-v4"):
        for child in (ROOT / "build" / name).iterdir():
            relative = child.relative_to(ROOT).as_posix()
            if (child.is_dir() and not child.is_symlink() and relative not in KEEP
                    and not child.name.endswith("-final")):
                yield child
    for child in (ROOT / "build").iterdir():
        if child.is_dir() and not child.is_symlink() and child.name.startswith(
                ("optimization-", "startup-", "multirate-startup-local-")):
            yield child


def output_candidate(path):
    relative = path.relative_to(ROOT)
    if path.suffix.lower() != ".wav" or path.is_symlink():
        return False
    return ("runs" in relative.parts or path.name == "denoise.wav"
            or path.name.endswith(OUTPUT_SUFFIXES))


def build_plan(protected):
    protected_inodes = {identity[:2] for identity in protected.values()}
    rows = []
    skipped = 0
    for directory in candidate_roots():
        for path in files(directory):
            if not output_candidate(path):
                continue
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode):
                continue
            inode = (info.st_dev, info.st_ino)
            if inode in protected_inodes:
                skipped += 1
                continue
            rows.append(dict(path=path.relative_to(ROOT).as_posix(),
                             signature=signature(path), nlink=info.st_nlink,
                             allocated_bytes=info.st_blocks * 512))
    counts = Counter(tuple(row["signature"][:2]) for row in rows)
    # Preserve any inode with links outside the complete deletion plan.
    removable = [row for row in rows if row["nlink"] == counts[tuple(row["signature"][:2])]]
    inodes = {}
    groups = Counter()
    for row in removable:
        key = tuple(row["signature"][:2])
        if key not in inodes:
            inodes[key] = row["allocated_bytes"]
            group = "/".join(Path(row["path"]).parts[:2])
            groups[group] += row["allocated_bytes"]
    return dict(files=sorted(removable, key=lambda row: row["path"]),
                skipped_protected_links=skipped,
                skipped_external_hardlinks=len(rows) - len(removable),
                estimated_allocated_bytes=sum(inodes.values()),
                allocated_bytes_by_group=dict(groups))


def available_bytes():
    usage = os.statvfs(ROOT)
    return usage.f_bavail * usage.f_frsize


def validate_preserved(protected, samples):
    for name, identity in protected.items():
        if signature(Path(name)) != identity:
            raise ValueError(f"protected file changed: {name}")
    if digest(FROZEN) != FROZEN_SHA:
        raise ValueError("frozen manifest changed")
    for path, expected in samples:
        if digest(path) != expected:
            raise ValueError(f"sampled frozen PCM changed: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="new audit directory")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    args.log.mkdir(parents=True, exist_ok=False)
    protected, samples, scenes = protected_inventory()
    plan = build_plan(protected)
    plan.update(root=str(ROOT), frozen_sha256=FROZEN_SHA,
                protected_file_count=len(protected), sampled_hash_count=len(samples),
                frozen_scenes=scenes, apply=args.apply, available_bytes_before=available_bytes())
    (args.log / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({key: value for key, value in plan.items() if key != "files"}, indent=2), flush=True)
    print(f"planned output files: {len(plan['files'])}", flush=True)
    if not args.apply:
        return
    validate_preserved(protected, samples)
    for row in plan["files"]:
        path = ROOT / row["path"]
        if path.is_symlink() or list(signature(path)) != list(row["signature"]):
            raise ValueError(f"candidate changed since planning: {path}")
    deleted = 0
    try:
        with (args.log / "deleted.jsonl").open("x") as log:
            for row in plan["files"]:
                path = ROOT / row["path"]
                if path.is_symlink() or list(signature(path)) != list(row["signature"]):
                    raise ValueError(f"candidate changed during cleanup: {path}")
                path.unlink()
                log.write(json.dumps(dict(path=row["path"], status="deleted")) + "\n")
                deleted += 1
    finally:
        result = dict(deleted_files=deleted, available_bytes_after=available_bytes(),
                      available_bytes_before=plan["available_bytes_before"])
        result["available_bytes_increase"] = result["available_bytes_after"] - result["available_bytes_before"]
        (args.log / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    validate_preserved(protected, samples)
    result.update(protected_files_unchanged=len(protected),
                  sampled_hashes_verified=len(samples), frozen_scenes=scenes,
                  frozen_manifest_sha256=digest(FROZEN))
    (args.log / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
