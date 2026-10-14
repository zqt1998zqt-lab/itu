#!/usr/bin/env python3
"""Replay every mode against immutable inputs in a new, separately audited batch."""
import argparse
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import run_audio_eval as ev
import run_expanded_eval as expanded


def initialize(root, cli, processing_profile):
    expanded.ROOT, expanded.CLI = Path(root), Path(cli)
    expanded.PROCESSING_PROFILE = processing_profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs-from", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cli", type=Path, default=Path("build/webrtc-host/cpp/itu_audio_cli"))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--processing-profile", choices=ev.PROCESSING_PROFILES, default="general")
    args = parser.parse_args()
    source, root = args.inputs_from.resolve(), args.root.resolve()
    if source == root or args.jobs < 1:
        raise ValueError("a separate output directory and positive jobs are required")
    inventory = source / "inputs-manifest.json"
    if not inventory.exists():
        inventory = source / "manifest.json"
    manifest = json.loads(inventory.read_text())
    ids = [scene["id"] for scene in manifest["scenes"]]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("empty or duplicate frozen scene inventory")
    for scene in manifest["scenes"]:
        for name, record in scene["files"].items():
            if ev.digest(source / "cases" / scene["id"] / name) != record["sha256"]:
                raise ValueError(f"frozen input changed: {scene['id']} {name}")
    root.mkdir(parents=True, exist_ok=False)
    artifacts = root / "artifacts"
    artifacts.mkdir()
    cli = artifacts / "itu_audio_cli"
    shutil.copy2(args.cli, cli)
    for path in (Path(__file__), Path(ev.__file__), Path(expanded.__file__),
                 Path(__file__).with_name("eval_profiles.py")):
        shutil.copy2(path, artifacts / path.name)
    if (source / "artifacts/source-verification.json").exists():
        shutil.copy2(source / "artifacts/source-verification.json", artifacts / "source-verification.json")
    (root / "sources").symlink_to(source / "sources", target_is_directory=True)
    if (source / "ATTRIBUTION.md").exists():
        shutil.copy2(source / "ATTRIBUTION.md", root / "ATTRIBUTION.md")
    shutil.copy2(inventory, root / "inputs-manifest.json")
    (root / "inputs-manifest.sha256").write_text(ev.digest(inventory) + "\n")
    for scene in manifest["scenes"]:
        folder = root / "cases" / scene["id"]
        folder.mkdir(parents=True)
        for name in scene["files"]:
            (folder / name).symlink_to(source / "cases" / scene["id"] / name)
        (folder / "scene.json").write_text(json.dumps(scene, indent=2) + "\n")
    manifest["environment"] = dict(manifest["environment"], cli_sha256=ev.digest(cli))
    manifest["processing_profile"] = args.processing_profile
    cpp_root = Path(__file__).resolve().parents[1]
    cpp_snapshot = artifacts / "cpp-source"
    for name in ("src", "include", "tests", "cmake"):
        shutil.copytree(cpp_root / name, cpp_snapshot / name)
    shutil.copy2(cpp_root / "CMakeLists.txt", cpp_snapshot / "CMakeLists.txt")
    (cpp_snapshot / "tools").mkdir()
    shutil.copy2(cpp_root / "tools/itu_audio_cli.cpp", cpp_snapshot / "tools/itu_audio_cli.cpp")
    manifest["replay_identity"] = dict(input_manifest_sha256=ev.digest(inventory),
        source_root=str(source), cli_sha256=ev.digest(cli),
        processing_profile=args.processing_profile,
        sources={path.name: ev.digest(path) for path in artifacts.glob("*.py")},
        cpp_sources={str(path.relative_to(cpp_snapshot)): ev.digest(path)
                     for path in sorted(cpp_snapshot.rglob("*")) if path.is_file()})
    started = time.time()
    scenes, results = [], []
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=initialize, initargs=(root, cli, args.processing_profile)) as pool:
        for scene, rows in pool.map(expanded.process, manifest["scenes"], chunksize=1):
            scenes.append(scene)
            results.extend(rows)
            if len(scenes) % 100 == 0:
                print(f"replay {len(scenes)}/{len(ids)}; {len(results)} runs; {time.time()-started:.1f}s", flush=True)
    manifest["scenes"] = scenes
    manifest["elapsed_seconds"] = time.time()-started
    manifest["parallel_jobs"] = args.jobs
    manifest["timing_scope"] = "host C++ timing during concurrent offline scenes; not Android certification"
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary = ev.summarize(root, scenes, results)
    print(json.dumps(dict(scenes=summary["scenes"], runs=summary["runs"], elapsed_seconds=time.time()-started)), flush=True)


if __name__ == "__main__":
    main()
