#!/usr/bin/env python3
"""Generate one-dB SNR sweeps with byte-identical targets, from frozen components."""
import argparse
import copy
import json
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

import run_audio_eval as ev
import manage_eval_dataset as manager
import audit_dataset_quality as audit


def prepare_group(rows, target_dbfs, grid):
    # Use the least-quantized target and noise separately, preserving their identity.
    target_scene = max(rows, key=lambda s: s["snr_db"])
    noise_scene = min(rows, key=lambda s: s["snr_db"])
    for scene in (target_scene, noise_scene):
        manager.verify_scene(scene)
    target = ev.read(manager.folder(target_scene) / "clean.wav")[:, 0]
    noise = ev.read(manager.folder(noise_scene) / "noise-component.wav")[:, 0]
    mask = ev.mask_windows(target_scene["seconds"], target_scene["near_schedule"])
    target = ev.level(target, target_dbfs, mask)
    noise = ev.level(noise, target_dbfs, mask)
    bound = max(max(np.max(np.abs(target + noise * 10 ** (-snr / 20))),
                    np.max(np.abs(noise * 10 ** (-snr / 20)))) for snr in grid)
    return target_scene, noise_scene, target, noise, mask, float(bound)


def render_group(prepared, grid, gain, root):
    original, noise_scene, target, background, mask, _ = prepared
    target = target * gain
    scenes = []
    for snr in grid:
        noise = background * gain * 10 ** (-snr / 20)
        scene = copy.deepcopy(original)
        scene_id = f"v4-p{scene['pair']:02d}-{scene['noise']}-snr{snr:+03d}"
        destination = root / "cases" / scene_id
        destination.mkdir(parents=True)
        zeros = np.zeros_like(target)
        signals = {"before.wav": target + noise, "clean.wav": target, "clean-mics.wav": target,
                   "clean-beam.wav": target, "noise-component.wav": noise, "echo-component.wav": zeros,
                   "ref.wav": zeros, "playback.wav": zeros}
        for name, values in signals.items():
            ev.write(destination / name, values)
        clean = ev.read(destination / "clean.wav")[:, 0]
        actual_noise = ev.read(destination / "noise-component.wav")[:, 0]
        effective_level = ev.db(ev.power(clean[mask]))
        flags = [flag for flag in original.get("quality_flags", []) if flag != "legacy_sweep_target_level_not_controlled"]
        scene.update(id=scene_id, profile="curated-v4", snr_db=snr, common_gain=gain,
                     near_dbfs=effective_level - ev.db(gain * gain), effective_near_dbfs=effective_level,
                     measured_input_snr_db=audit.ratio_db(clean[mask], actual_noise[mask]),
                     source_dataset=str(root), source_case_id=scene_id, quality_flags=flags,
                     quality_track="stress" if flags else "quality",
                     snr_scope="scheduled-window mean power, not VAD-active power",
                     target_level_scope="identical PCM target across entire sweep; shared headroom across all generated groups",
                     parent_components=dict(clean_scene=original["id"], noise_scene=noise_scene["id"],
                        clean_sha256=original["files"]["clean.wav"]["sha256"],
                        noise_sha256=noise_scene["files"]["noise-component.wav"]["sha256"]),
                     files={name:dict(sha256=ev.digest(destination / name), bytes=(destination / name).stat().st_size)
                            for name in signals})
        (destination / "scene.json").write_bytes(manager.encoded(scene))
        result = audit.audit_scene((root, scene))
        if result["errors"]:
            raise ValueError(f"new scene input audit failed: {scene_id} {result['errors']}")
        scenes.append(scene)
    if len({s["files"]["clean.wav"]["sha256"] for s in scenes}) != 1:
        raise ValueError("SNR sweep changed the clean target")
    return scenes


def build(workspace, destination, pairs, noises, minimum=-10, maximum=30, step=1, target_dbfs=-36, jobs=2):
    if step < 1 or maximum < minimum or (maximum - minimum) % step or jobs not in (1, 2, 3, 4):
        raise ValueError("nonempty evenly spaced SNR grid and 1..4 jobs required")
    state = manager.load(workspace)
    grid = list(range(minimum, maximum + 1, step))
    groups = defaultdict(list)
    for record in state["scenes"].values():
        s = record["scene"]
        if not record["excluded"] and s["family"] == "ns" and s["mics"] == 1 and s["pair"] in pairs and s["noise"] in noises:
            groups[(s["pair"], s["noise"])].append(s)
    if set(groups) != {(pair, noise) for pair in pairs for noise in noises}:
        raise ValueError("requested frozen speaker/noise coverage missing")
    prepared = [prepare_group(rows, target_dbfs, grid) for _, rows in sorted(groups.items())]
    gain = min(1., .88 / max(row[-1] for row in prepared))
    destination.mkdir(parents=True, exist_ok=False)
    scenes = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for rows in pool.map(lambda p: render_group(p, grid, gain, destination), prepared):
            scenes.extend(rows)
            print(f"generated {len(scenes)}/{len(prepared)*len(grid)}", flush=True)
    with manager.locked(workspace):
        if manager.load(workspace)["revision"] != state["revision"]:
            raise ValueError("catalog changed during generation")
        for scene in scenes:
            manager.add_scene(state, destination / "cases" / scene["id"] / "scene.json")
        manager.save(workspace, state, "add_constant_target_sweeps", dict(
            generator_sha256=ev.digest(Path(__file__)), grid=grid, shared_gain=gain, target_dbfs=target_dbfs,
            scenes=[dict(id=s["id"], parents=s["parent_components"], files=s["files"]) for s in scenes]))
    (destination / "generation.json").write_bytes(manager.encoded(dict(
        scenes=len(scenes), groups=len(prepared), snr_db=grid, common_gain=gain,
        effective_target_dbfs=target_dbfs + ev.db(gain * gain),
        parent_manifest_sha256=[p["manifest_sha256"] for p in state["parents"]],
        generator_sha256=ev.digest(Path(__file__)), all_input_audits_passed=True,
        clean_target_identical_within_sweep=True)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--pairs", default="0,2,4,10")
    parser.add_argument("--noises", default="tcar,car_hvac,car_window_wind,car_rain,car_horn,car_indicator,car_engine_ramp")
    parser.add_argument("--snr-min", type=int, default=-10)
    parser.add_argument("--snr-max", type=int, default=30)
    parser.add_argument("--snr-step", type=int, default=1)
    parser.add_argument("--target-dbfs", type=float, default=-36)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    build(args.workspace.resolve(), args.destination.resolve(), list(map(int, args.pairs.split(","))),
          args.noises.split(","), args.snr_min, args.snr_max, args.snr_step, args.target_dbfs, args.jobs)


if __name__ == "__main__":
    main()
