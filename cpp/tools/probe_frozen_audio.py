#!/usr/bin/env python3
"""Paired diagnostics on frozen audio; never claims full-matrix acceptance."""
import argparse
import json
import shlex
import shutil
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import check_audio_acceptance as ac
import run_audio_eval as ev


def accepted_inventory(source_root, acceptance):
    inventory = source_root / "inputs-manifest.json"
    manifest = json.loads(inventory.read_text())
    if ev.digest(inventory) != acceptance["frozen_manifest_sha256"]:
        frozen_path = Path(acceptance["frozen_manifest_path"])
        if not frozen_path.is_absolute():
            frozen_path = Path(__file__).resolve().parents[2] / frozen_path
        frozen, _, _ = ac.frozen_inventory(frozen_path, acceptance["frozen_manifest_sha256"])
        ac.verify_manifest(manifest, frozen)
    return manifest


def variant_invocation(scene, cli, directory, flags):
    flags = list(flags)
    profile = "general"
    if flags.count("--profile") > 1:
        raise ValueError("a variant must select only one processing profile")
    if "--profile" in flags:
        index = flags.index("--profile")
        if index + 1 == len(flags):
            raise ValueError("processing profile value missing")
        profile = flags[index + 1]
        del flags[index:index + 2]
    return ev.invocation(scene, "full", cli, directory, profile) + flags, profile


def select_scenes(manifest, acceptance):
    selected = set()
    failing = {}
    for item in acceptance["details"]:
        for gate in item["gates"]:
            if gate["status"] == "fail":
                key = gate["segment"] + "." + gate["metric"]
                margin = gate["value"] - gate["threshold"] if gate["operator"] == ">=" else gate["threshold"] - gate["value"]
                failing.setdefault(key, []).append((margin, item["id"]))
    for rows in failing.values():
        selected.update(item[1] for item in sorted(rows)[:2])
    passed = {item["id"] for item in acceptance["details"] if item["passed"]}
    for family in sorted({scene["family"] for scene in manifest["scenes"]}):
        candidates = [scene for scene in manifest["scenes"] if scene["id"] in passed and scene["family"] == family]
        selected.update(scene["id"] for scene in candidates[::max(1, len(candidates) // 3)][:3])
    return [scene for scene in manifest["scenes"] if scene["id"] in selected]


def process(arguments):
    scene, variant, flags, source_root, output_root, cli, rules = arguments
    source = source_root / "cases" / scene["id"]
    directory = output_root / variant / scene["id"]
    directory.mkdir(parents=True)
    for name, record in scene["files"].items():
        if ev.digest(source / name) != record["sha256"]:
            raise ValueError(f"frozen input changed: {scene['id']} {name}")
        (directory / name).symlink_to(source / name)
    prefix = directory / "runs/full/result"
    prefix.parent.mkdir(parents=True)
    command, profile = variant_invocation(scene, cli, directory, flags)
    run = subprocess.run(command, capture_output=True, text=True, timeout=60)
    (prefix.parent / "invocation.json").write_text(json.dumps(dict(argv=command,
        cli_sha256=ev.digest(cli), exit_code=run.returncode, stdout=run.stdout, stderr=run.stderr), indent=2) + "\n")
    run.check_returncode()
    row = ev.measure(scene, "full", directory)
    ev.verify_processing_profile(row["engine"], profile)
    checked = ac.check_scene(scene, row, rules)
    if variant == "baseline":
        original = source / "runs/full/result.after.wav"
        if ev.digest(original) != row["output_sha256"]:
            raise ValueError("baseline replay differs from frozen PCM output")
    return dict(variant=variant, measurements=row, acceptance=checked)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cli", type=Path, default=Path("build/webrtc-host/cpp/itu_audio_cli"))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--ids", help="comma-separated exact scene IDs; default is failure representatives plus passed controls")
    selection.add_argument("--all", action="store_true", help="all frozen scenes; remains diagnostic without module controls")
    parser.add_argument("--variant", action="append", required=True, help="name=CLI flags; baseline requires frozen output equality")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    source, root = args.dataset.resolve(), args.output.resolve()
    acceptance = json.loads((source / "acceptance.json").read_text())
    manifest = accepted_inventory(source, acceptance)
    if args.ids:
        ids = set(args.ids.split(","))
        scenes = [scene for scene in manifest["scenes"] if scene["id"] in ids]
        if ids != {scene["id"] for scene in scenes}:
            raise ValueError("requested scenes missing from frozen inputs")
    elif args.all:
        scenes = manifest["scenes"]
    else:
        scenes = select_scenes(manifest, acceptance)
    if not scenes:
        raise ValueError("empty diagnostic inventory")
    variants = {}
    for value in args.variant:
        name, separator, flags = value.partition("=")
        if not separator or not name or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_" for char in name) or name in variants:
            raise ValueError("variants need unique simple names and '='")
        variants[name] = shlex.split(flags)
        variant_invocation(scenes[0], args.cli, root, variants[name])
    root.mkdir(parents=True, exist_ok=False)
    cli = root / "itu_audio_cli"
    shutil.copy2(args.cli, cli)
    rules = json.loads(ac.RULES.read_text())
    if ev.digest(ac.RULES) != acceptance["rules_sha256"]:
        raise ValueError("rules changed")
    (root / "identity.json").write_text(json.dumps(dict(scope="diagnostic paired subset; not final acceptance",
        cli_sha256=ev.digest(cli), input_manifest_sha256=ev.digest(source / "inputs-manifest.json"),
        acceptance_frozen_manifest_sha256=acceptance["frozen_manifest_sha256"],
        rules_sha256=ev.digest(ac.RULES), variants=variants, scenes=[scene["id"] for scene in scenes]), indent=2) + "\n")
    arguments = [(scene, variant, flags, source, root, cli, rules)
                 for variant, flags in variants.items() for scene in scenes]
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for row in pool.map(process, arguments, chunksize=1):
            rows.append(row)
            if len(rows) % len(scenes) == 0:
                current = [item for item in rows if item["variant"] == row["variant"]]
                failures = Counter(gate["segment"] + "." + gate["metric"] for item in current
                                   for gate in item["acceptance"]["gates"] if gate["status"] == "fail")
                print(json.dumps(dict(variant=row["variant"], scenes=len(current),
                    passed=sum(item["acceptance"]["passed"] for item in current), failures=dict(failures))), flush=True)
    (root / "probe.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
