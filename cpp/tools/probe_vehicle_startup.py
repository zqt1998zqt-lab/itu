#!/usr/bin/env python3
"""Cold-start ablations from frozen capture/render, without a silence preamble."""
import argparse
import csv
import json
import shlex
import shutil
import subprocess
from pathlib import Path

import numpy as np
import run_audio_eval as ev
import check_audio_acceptance as ac


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--pairs", default="0,2")
    parser.add_argument("--variant", action="append", required=True)
    args = parser.parse_args()
    dataset, root = args.dataset.resolve(), args.root.resolve()
    manifest_path = dataset / "inputs-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rules = json.loads(ac.RULES.read_text())
    if ev.digest(ac.RULES) != ac.RULES_SHA256:
        raise ValueError("frozen thresholds changed")
    pairs = [int(value) for value in args.pairs.split(",")]
    if not pairs or len(set(pairs)) != len(pairs):
        raise ValueError("unique nonempty pair selection required")
    variants = {}
    for value in args.variant:
        name, separator, flags = value.partition("=")
        if not separator or not name or name in variants or any(
                char not in "abcdefghijklmnopqrstuvwxyz0123456789_" for char in name):
            raise ValueError("unique simple variant name required")
        variants[name] = shlex.split(flags)
    root.mkdir(parents=True, exist_ok=False)
    cli = root / "itu_audio_cli"
    shutil.copy2(args.cli, cli)
    identity = dict(scope="cold-start paired diagnostic; no real-time acceptance",
                    cli_sha256=ev.digest(cli), manifest_sha256=ev.digest(manifest_path),
                    rules_sha256=ev.digest(ac.RULES), variants=variants, pairs=pairs)
    rows = []
    for pair in pairs:
        candidates = [scene for scene in manifest["scenes"] if scene["pair"] == pair
                      and scene["mics"] == 1 and scene["family"] == "ns"
                      and scene["noise"] == "tcar" and scene["snr_db"] == 30]
        if len(candidates) != 1:
            raise ValueError("unique frozen cold-start condition required")
        scene, begin, end = candidates[0], 2, 9
        folder = dataset / "cases" / scene["id"]
        inputs = root / f"pair-{pair:02d}"
        inputs.mkdir()
        for name in ("before.wav", "ref.wav", "clean.wav"):
            if ev.digest(folder / name) != scene["files"][name]["sha256"]:
                raise ValueError("frozen input changed")
            ev.write(inputs / name, ev.read(folder / name)[round(begin * ev.RATE):round(end * ev.RATE)])
        clean = ev.read(inputs / "clean.wav")[:, 0]
        before = ev.read(inputs / "before.wav")[:, 0]
        for name, flags in variants.items():
            run_root = inputs / name
            run_root.mkdir()
            prefix = run_root / "result"
            command = [str(cli), "--profile", "vehicle_stream", "--mic", str(inputs / "before.wav"),
                       "--echo", str(inputs / "ref.wav"), "--out", str(prefix), *flags]
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            (run_root / "invocation.json").write_text(json.dumps(dict(
                argv=command, exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr), indent=2) + "\n")
            result.check_returncode()
            engine = json.loads(prefix.with_suffix(".json").read_text())
            output = ev.read(prefix.with_suffix(".after.wav"))[:, 0]
            after = output[engine["dsp_delay_samples"]:engine["dsp_delay_samples"] + len(clean)]
            if len(after) != len(clean) or engine["stream_resets"] != 0:
                raise ValueError("output alignment or stream reset mismatch")
            qb, qa = ev.quality(clean, before), ev.quality(clean, after)
            frames = list(csv.DictReader(prefix.with_suffix(".frames.csv").open()))[:len(clean) // 160]
            diagnostics = {key: dict(mean=float(np.mean([float(row[key]) for row in frames])),
                                     minimum=float(min(float(row[key]) for row in frames)),
                                     maximum=float(max(float(row[key]) for row in frames)))
                           for key in ("speech_probability", "fullband_ns_mix", "fullband_noise_power")}
            measured = dict(mode="full", segments=dict(near_only=dict(
                before=qb, after=qa, stoi_improvement=qa["stoi"] - qb["stoi"])), engine=engine,
                process_frames_over_10ms=engine["process_frames_over_10ms"],
                output_levels=dict(peak_dbfs=qa["peak_dbfs"], clipping_percent=qa["clipping_percent"]))
            condition = dict(scene, windows=dict(near_only=[0, end - begin]))
            verdict = ac.check_scene(condition, measured, rules)
            rows.append(dict(pair=pair, id=scene["id"], variant=name, quality=qa,
                             stoi_delta=qa["stoi"] - qb["stoi"], acceptance=verdict, diagnostics=diagnostics,
                             output_sha256=ev.digest(prefix.with_suffix(".after.wav"))))
            print(json.dumps(dict(pair=pair, variant=name, sisdr_db=qa["sisdr_db"],
                                  stoi_delta=qa["stoi"] - qb["stoi"])), flush=True)
    (root / "probe.json").write_text(json.dumps(dict(identity=identity, rows=rows), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
