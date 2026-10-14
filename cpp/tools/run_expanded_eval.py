#!/usr/bin/env python3
"""Freeze the expanded inputs, then run every C++ mode with resumable evidence."""
import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import run_audio_eval as ev
from eval_profiles import expanded_matrix, playback_check_matrix, coverage

SPEAKERS = NOISES = CLI = ROOT = None
PROCESSING_PROFILE = "general"
CACHE = {}


def pin_generator(root, source_sha):
    paths = [Path(__file__), Path(ev.__file__), Path(__file__).with_name("eval_profiles.py")]
    identity = dict(source_manifest_sha256=source_sha,
                    generator_sha256={path.name: ev.digest(path) for path in paths})
    lock = root / "generator-lock.json"
    if lock.exists():
        if json.loads(lock.read_text()) != identity:
            raise ValueError("generation sources changed; use a new batch root")
    else:
        if (root / "inputs-manifest.json").exists() or any((root / "cases").glob("*/scene.json")):
            raise ValueError("existing batch has no pre-generation version lock; use a new batch root")
        lock.write_text(json.dumps(identity, indent=2) + "\n")
        artifacts = root / "artifacts"
        artifacts.mkdir(exist_ok=True)
        for path in paths:
            shutil.copyfile(path, artifacts / path.name)
    return identity


def initialize(root, cli, processing_profile="general"):
    global ROOT, CLI, SPEAKERS, NOISES, PROCESSING_PROFILE
    ROOT, CLI = Path(root), Path(cli)
    PROCESSING_PROFILE = processing_profile
    SPEAKERS, NOISES = ev.sources(ROOT)


def prepare(case):
    folder = ROOT / "cases" / case["id"]
    if (folder / "scene.json").exists():
        scene = json.loads((folder / "scene.json").read_text())
        if any(scene.get(k) != v for k, v in case.items()):
            raise ValueError(f"prepared scene conditions changed: {case['id']}")
        for name, record in scene["files"].items():
            if ev.digest(folder / name) != record["sha256"]:
                raise ValueError(f"prepared scene hash changed: {case['id']} {name}")
        return scene
    return ev.make_case(case, SPEAKERS, NOISES, folder, CACHE)


def process(scene):
    folder = ROOT / "cases" / scene["id"]
    results, cache = [], {}
    for mode in ev.modes(scene):
        prefix = folder / "runs" / mode / "result"
        prefix.parent.mkdir(parents=True, exist_ok=True)
        command = ev.invocation(scene, mode, CLI, folder, PROCESSING_PROFILE)
        log_path = prefix.parent / "invocation.json"
        if log_path.exists() and prefix.with_suffix(".json").exists():
            log = json.loads(log_path.read_text())
            if log["argv"] != command or log["cli_sha256"] != ev.digest(CLI) or log["exit_code"] != 0:
                raise ValueError(f"resume binary or invocation changed: {scene['id']} {mode}")
        else:
            run = subprocess.run(command, text=True, capture_output=True, timeout=60)
            log_path.write_text(json.dumps(dict(argv=command, cli_sha256=ev.digest(CLI),
                exit_code=run.returncode, stdout=run.stdout, stderr=run.stderr), indent=2) + "\n")
            run.check_returncode()
        measurement = ev.measure(scene, mode, folder, quality_cache=cache)
        ev.verify_processing_profile(measurement["engine"], PROCESSING_PROFILE)
        results.append(measurement)
    full = next(row for row in results if row["mode"] == "full")
    output = folder / "runs/full/result.after.wav"
    denoise = folder / "denoise.wav"
    if not denoise.exists():
        os.link(output, denoise)
    if ev.digest(denoise) != full["output_sha256"]:
        raise ValueError("denoise audio differs from evaluated full output")
    (folder / "website-options.json").write_text(json.dumps(dict(sample_rate=ev.RATE,
        speech=dict(zip(("start", "end"), scene["windows"]["near_only"])),
        noise=dict(zip(("start", "end"), scene["windows"]["noise_only"])),
        after_offset_ms=full["engine"]["dsp_delay_samples"] / 16,
        common_clock=scene["stress"] != "clock_300ppm", speech_confirmed=True,
        far_end_only=False, aec_only=False, stationary_noise=scene["noise"] == "white"), indent=2) + "\n")
    return dict(scene, denoise_sha256=ev.digest(denoise)), results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/itu-eval-v3"))
    parser.add_argument("--cli", type=Path, default=Path("build/webrtc-host/cpp/itu_audio_cli"))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--processing-profile", choices=ev.PROCESSING_PROFILES, default="general")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--playback-check", action="store_true", help="480 scenes: matched radio AEC plus cabin NS controls")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads((root / "sources/manifest.json").read_text())
    cases = (playback_check_matrix if args.playback_check else expanded_matrix)(source_manifest["speaker_metadata"])
    source_sha = ev.digest(root / "sources/manifest.json")
    generation = pin_generator(root, source_sha)
    artifact = root / "artifacts/itu_audio_cli"
    artifact.parent.mkdir(exist_ok=True)
    if not artifact.exists():
        shutil.copy2(args.cli, artifact)
    if ev.digest(artifact) != ev.digest(args.cli):
        raise ValueError("expanded batch binary already frozen; use a new root")
    manifest_path, sha_path = root / "inputs-manifest.json", root / "inputs-manifest.sha256"
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=initialize,
                             initargs=(root, artifact, args.processing_profile)) as pool:
        if manifest_path.exists():
            if not sha_path.exists() or ev.digest(manifest_path) != sha_path.read_text().strip():
                raise ValueError("pre-run frozen input manifest changed")
            manifest = json.loads(manifest_path.read_text())
            if manifest["source_manifest_sha256"] != source_sha:
                raise ValueError("frozen source inventory changed")
            if len(manifest["scenes"]) != len(cases) or [{k: s[k] for k in c} for s, c in zip(manifest["scenes"], cases)] != cases:
                raise ValueError("expanded matrix changed after input freeze")
        else:
            scenes = []
            for scene in pool.map(prepare, cases, chunksize=1):
                scenes.append(scene)
                if len(scenes) % 200 == 0:
                    print(f"prepare {len(scenes)}/{len(cases)} {time.time()-started:.1f}s", flush=True)
            manifest = dict(schema_version=3, profile="playback-check-v3" if args.playback_check else "expanded-v3",
                evaluation_scope="radio AEC classification correction; not the full expanded matrix" if args.playback_check else "full expanded matrix",
                seed=ev.SEED, scenes=scenes,
                coverage=coverage(scenes), source_manifest_sha256=source_sha,
                generator_sha256=ev.digest(Path(__file__)),
                generation_identity=generation,
                environment=dict(machine=platform.machine(), system=platform.platform(),
                    cli_sha256=ev.digest(artifact), packages={name: importlib.metadata.version(name)
                    for name in ("numpy", "scipy", "soundfile", "pystoi", "rir-generator")}))
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            sha_path.write_text(ev.digest(manifest_path) + "\n")
            print(f"FROZEN {len(cases)} scenes {ev.digest(manifest_path)}", flush=True)
        if args.prepare_only:
            return
        results, scenes = [], []
        for scene, rows in pool.map(process, manifest["scenes"], chunksize=1):
            scenes.append(scene)
            results.extend(rows)
            if len(scenes) % 100 == 0:
                print(f"process {len(scenes)}/{len(cases)} {len(results)} runs {time.time()-started:.1f}s", flush=True)
        manifest["scenes"] = scenes
        manifest["processing_profile"] = args.processing_profile
        manifest["elapsed_seconds"] = time.time()-started
        manifest["parallel_jobs"] = args.jobs
        manifest["timing_scope"] = "host C++ timing during independent concurrent offline scenes; not Android or isolated performance certification"
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        summary = ev.summarize(root, scenes, results)
        print(json.dumps(dict(scenes=summary["scenes"], runs=summary["runs"], seconds=time.time()-started)), flush=True)


if __name__ == "__main__":
    main()
