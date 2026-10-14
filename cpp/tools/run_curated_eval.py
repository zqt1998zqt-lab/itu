#!/usr/bin/env python3
"""Paired C++ NS experiments on immutable v4 inputs and fixed scoring windows."""
import argparse
import csv
import json
import math
import shlex
import shutil
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

import run_audio_eval as ev
import check_audio_acceptance as ac
import manage_eval_dataset as manager


def convert(x, source, target):
    divisor = math.gcd(source, target)
    return x if source == target else resample_poly(x, target // divisor, source // divisor, axis=0)


def run_case(arguments):
    scene, rate, variants, cli, root, cold, rules = arguments
    manager.verify_scene(scene)
    source = manager.folder(scene)
    before = ev.read(source / "before.wav")[:, 0]
    target = ev.read(source / "clean.wav")[:, 0]
    windows = scene["windows"]
    if cold:
        lo, hi = windows["near_only"]
        before, target = (ev.window(value, [lo, hi]) for value in (before, target))
        windows = dict(near_only=[0, hi-lo])
    directory = root / scene["id"] / str(rate)
    directory.mkdir(parents=True)
    input_path = directory / "before.wav"
    sf.write(input_path, convert(before, ev.RATE, rate), rate, subtype="PCM_16")
    input_sha = ev.digest(input_path)
    rows = []
    for variant, flags in variants.items():
        prefix = directory / variant
        command = [str(cli), "--profile", "vehicle_stream", "--no-aec", "--mic", str(input_path),
                   "--out", str(prefix), *flags]
        run = subprocess.run(command, capture_output=True, text=True, timeout=120)
        prefix.with_suffix(".invocation.json").write_bytes(manager.encoded(dict(argv=command,
            exit_code=run.returncode, stdout=run.stdout, stderr=run.stderr, input_sha256=input_sha)))
        run.check_returncode()
        engine = json.loads(prefix.with_suffix(".json").read_text())
        output, actual_rate = sf.read(prefix.with_suffix(".after.wav"))
        delay = engine["dsp_delay_samples"]
        if actual_rate != rate or len(output) != len(before)*rate//ev.RATE + delay or engine["stream_resets"] != 0:
            raise ValueError("rate/length/fixed delay/stream reset mismatch")
        after = convert(output[delay:], rate, ev.RATE)
        segments = {}
        for name, limits in windows.items():
            b, a, t = (ev.window(value, limits) for value in (before, after, target))
            if name.startswith("near"):
                qb, qa = ev.quality(t, b), ev.quality(t, a)
                segments[name] = dict(before=qb, after=qa, stoi_improvement=(qa["stoi"]-qb["stoi"])
                    if qa["stoi"] is not None and qb["stoi"] is not None else None)
            else:
                segments[name] = ev.attenuation_measurement(b, a, pcm16=True)
        frames = list(csv.DictReader(prefix.with_suffix(".frames.csv").open()))
        timing, over = ev.frame_timing(frames, len(before)//160)
        row = dict(id=scene["id"], family=scene["family"], mode="full", segments=segments,
                   engine=dict(engine, **timing), process_frames_over_10ms=over,
                   output_levels=dict(peak_dbfs=ev.db(float(np.max(np.abs(output)))**2),
                                      clipping_percent=float(np.mean(np.abs(output)>=.999)*100)))
        verdict = ac.check_scene(dict(scene, windows=windows), row, rules)
        applicable = [gate for gate in verdict["gates"] if not
            (scene.get("interference_class") == "interfering_speech" and gate["segment"] == "noise_only")]
        startup = None
        if cold:
            correct, wrong = (ev.quality(target[:4800], value[:4800]) for value in (after, convert(output, rate, ev.RATE)))
            startup = dict(first_300ms=correct, wrong_zero_delay=wrong,
                negative_control_detected=correct["sisdr_db"] is not None and wrong["sisdr_db"] is not None
                    and correct["sisdr_db"]-wrong["sisdr_db"]>3)
        rows.append(dict(scene_id=scene["id"], pair=scene["pair"], near_speaker=scene["near_speaker"],
            noise=scene["noise"], snr_db=scene["snr_db"], split=scene.get("split"), track=scene.get("quality_track"),
            interference_class=scene.get("interference_class"), sample_rate=rate, variant=variant,
            measurements=row, historical_threshold_verdict=verdict,
            applicable_gates_passed=all(g["status"] == "pass" or (g["status"] == "unavailable" and g["optional"]) for g in applicable),
            not_applicable_gates=[g for g in verdict["gates"] if g not in applicable], startup=startup,
            input_sha256=input_sha, output_sha256=ev.digest(prefix.with_suffix(".after.wav"))))
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["variant"], row["sample_rate"], row["track"])].append(row)
    summary = []
    for (variant, rate, track), selected in sorted(groups.items()):
        summary.append(dict(variant=variant, sample_rate=rate, track=track, cases=len(selected),
            historical_passed=sum(row["historical_threshold_verdict"]["passed"] for row in selected),
            applicable_gates_passed=sum(row["applicable_gates_passed"] for row in selected),
            not_applicable_gates=sum(len(row["not_applicable_gates"]) for row in selected),
            failed_gates=dict(Counter(f"{gate['segment']}/{gate['metric']}" for row in selected
                for gate in row["historical_threshold_verdict"]["gates"] if gate["status"] == "fail"))))
    return summary


def select_cases(scenes, pairs, snrs, noises=None, splits=None):
    selected = [scene for scene in scenes if not scene.get("catalog_excluded", False)
                and (splits is None or scene.get("split") in splits)
                and scene.get("profile") == "curated-v4" and scene["family"] == "ns" and scene["mics"] == 1
                and scene["pair"] in pairs and scene["snr_db"] in snrs
                and (noises is None or scene["noise"] in noises)]
    requested_noises = set(noises) if noises is not None else {s["noise"] for s in selected}
    keys = [(s["pair"], s["snr_db"], s["noise"]) for s in selected]
    expected = {(pair, snr, noise) for pair in pairs for snr in snrs for noise in requested_noises}
    if not selected or set(keys) != expected or len(keys) != len(set(keys)):
        raise ValueError("nonempty complete pair/SNR/noise selection required")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--pairs", default="0,2,4,10")
    parser.add_argument("--snr", default="-10,0,10,30")
    parser.add_argument("--noises")
    parser.add_argument("--variant", action="append", required=True)
    parser.add_argument("--cold-start", action="store_true")
    parser.add_argument("--split", action="append", choices=("development", "validation", "holdout", "cross_split_stress"))
    parser.add_argument("--jobs", type=int, choices=(1,2,3,4), default=4)
    args = parser.parse_args()
    manifest_path = args.dataset / "inputs-manifest.json"
    manifest_sha = ev.digest(manifest_path)
    if manifest_sha != (args.dataset / "inputs-manifest.sha256").read_text().strip():
        raise ValueError("input manifest freeze changed")
    manifest = json.loads(manifest_path.read_text())
    pairs, snrs = set(map(int, args.pairs.split(","))), set(map(int, args.snr.split(",")))
    selected = select_cases(manifest["scenes"], pairs, snrs,
                            args.noises.split(",") if args.noises else None, args.split)
    variants = {}
    for text in args.variant:
        name, separator, flags = text.partition("=")
        if not separator or not name.isidentifier() or name in variants:
            raise ValueError("unique variant name required")
        variants[name] = shlex.split(flags)
    if ev.digest(ac.RULES) != ac.RULES_SHA256:
        raise ValueError("frozen rules changed")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    cli = root / "itu_audio_cli"
    shutil.copy2(args.cli, cli)
    identity = dict(source_manifest_sha256=manifest_sha, cli_sha256=ev.digest(cli),
        rules_sha256=ev.digest(ac.RULES), runner_sha256=ev.digest(Path(__file__)), variants=variants,
        jobs=args.jobs, cold_start=args.cold_start, input_scenes=len(selected),
        scope="paired host NS diagnostic; no native highband, paced callback, Android or blind holdout acceptance",
        final_acceptance_eligible=False)
    (root / "identity.json").write_bytes(manager.encoded(identity))
    rows = []
    rules = json.loads(ac.RULES.read_text())
    tasks = [(scene,rate,variants,cli,root,args.cold_start,rules) for scene in selected for rate in (16000,24000,48000)]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for results in pool.map(run_case, tasks):
            rows.extend(results)
            if len(rows) % (24*len(variants)) == 0:
                print(f"processed {len(rows)}/{len(tasks)*len(variants)}", flush=True)
    for task in tasks:
        input_path = root / task[0]["id"] / str(task[1]) / "before.wav"
        if not input_path.is_file() or ev.digest(input_path) != next(
                row["input_sha256"] for row in rows if row["scene_id"] == task[0]["id"] and row["sample_rate"] == task[1]):
            raise ValueError("input changed after variant processing")
    (root / "report.json").write_bytes(manager.encoded(dict(identity=identity, summary=summarize(rows), rows=rows)))
    print(json.dumps(summarize(rows), indent=2), flush=True)


if __name__ == "__main__":
    main()
