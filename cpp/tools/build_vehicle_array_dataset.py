#!/usr/bin/env python3
"""Derive new spatial vehicle inputs without modifying the frozen mono corpus."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import run_audio_eval as ev
from run_vehicle_stream import select_timeline, sequence_beam_delays


def source_folder(dataset, scene):
    return Path(scene.get("source_dataset", dataset)) / "cases" / scene.get("source_case_id", scene["id"])


def spatial_case(scene, microphones):
    if microphones not in (2, 4) or scene["mics"] != 1:
        raise ValueError("array derivation requires a mono source and 2/4 microphones")
    center = scene.get("microphone_positions_m", [[2.0, 2.0, 1.2]])
    if len(center) != 1 or len(center[0]) != 3:
        raise ValueError("mono source must have exactly one microphone position")
    case = {key: value for key, value in scene.items()
            if key not in ("source_dataset", "source_case_id", "files", "microphone_positions_m",
                           "near_arrival_samples", "beam_delays_samples")}
    return dict(case, id=f"{scene['id']}-m{microphones}", mics=microphones,
                microphone_positions_m=[[center[0][0] + (ch - (microphones - 1) / 2) * .035,
                                         center[0][1], center[0][2]] for ch in range(microphones)],
                derived_from_id=scene["id"], derived_from_files=scene["files"],
                spatial_generation="new simulated array; not copied mono channels")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pairs", type=int, nargs="+")
    parser.add_argument("--radio-dataset", type=Path, help="Referenced radio scenes required by the vehicle timeline")
    parser.add_argument("--jobs", type=int, choices=(1, 2, 3, 4), default=4)
    args = parser.parse_args()
    dataset, root = args.dataset.resolve(), args.root.resolve()
    if root == dataset or root.is_relative_to(dataset):
        parser.error("new array dataset must be outside the frozen dataset")
    manifest = json.loads((dataset / "inputs-manifest.json").read_text())
    radio = args.radio_dataset.resolve() if args.radio_dataset else None
    if radio:
        manifest["scenes"] += [dict(scene, source_dataset=str(radio)) for scene in
                               json.loads((radio / "inputs-manifest.json").read_text())["scenes"]
                               if scene.get("playback_source") == "radio"]
    pairs = sorted(set(args.pairs) if args.pairs is not None else {s["pair"] for s in manifest["scenes"]})
    timelines = {pair: select_timeline(manifest["scenes"], pair) for pair in pairs}
    if not timelines:
        parser.error("nonempty speaker selection required")
    speakers, noises = ev.sources(dataset)
    root.mkdir(parents=True, exist_ok=False)
    (root / "cases").mkdir()
    (root / "sources").symlink_to((dataset / "sources").resolve(), target_is_directory=True)
    selected = []
    work = []
    for pair, timeline in timelines.items():
        for _, scene, _, _ in timeline:
            folder = source_folder(dataset, scene)
            for name, record in scene["files"].items():
                if ev.digest(folder / name) != record["sha256"]:
                    raise ValueError(f"frozen input changed: {scene['id']} {name}")
            (root / "cases" / scene["id"]).symlink_to(folder, target_is_directory=True)
            selected.append({key: value for key, value in scene.items()
                             if key not in ("source_dataset", "source_case_id")})
            for microphones in (2, 4):
                work.append(spatial_case(scene, microphones))

    def generate(case):
        # A per-task RIR cache avoids concurrent mutation and preserves geometry.
        result = ev.make_case(case, speakers, noises, root / "cases" / case["id"], {})
        print(json.dumps(dict(id=result["id"], pair=result["pair"], mics=result["mics"])), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        selected.extend(pool.map(generate, work))
    for pair in pairs:
        for microphones in (1, 2, 4):
            sequence_beam_delays(select_timeline(selected, pair, microphones), microphones)
    identity = dict(base_manifest_sha256=ev.digest(dataset / "inputs-manifest.json"),
                    source_manifest_sha256=ev.digest(dataset / "sources/manifest.json"),
                    radio_manifest_sha256=ev.digest(radio / "inputs-manifest.json") if radio else None,
                    generator_sha256=ev.digest(Path(__file__)),
                    acoustic_generator_sha256=ev.digest(Path(ev.__file__)),
                    base_dataset=str(dataset), pairs=pairs, microphone_options=[1, 2, 4],
                    scope="recorded sources with simulated spatial delays, echo RIR and independent sensor noise; not a measured cabin array",
                    mono_scope="symlinks to immutable original mono inputs",
                    spatial_noise_scope="directional combined road/interference source plus per-channel sensor noise; no diffuse-field model",
                    beam_scope="fixed geometry supplied as calibration; no signal-derived or target-derived steering")
    result = dict(identity=identity, scenes=selected)
    (root / "inputs-manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(scenes=len(selected), speakers=len(pairs),
                         manifest_sha256=ev.digest(root / "inputs-manifest.json")), indent=2), flush=True)


if __name__ == "__main__":
    main()
