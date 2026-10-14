#!/usr/bin/env python3
"""Frozen speaker coverage with one persistent engine per vehicle sequence."""
import argparse
import collections
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import run_audio_eval as ev
import check_audio_acceptance as ac
from run_vehicle_stream import select_timeline, VEHICLE_TIMELINE


def summarize(reports):
    if not reports or any(not report["details"] for _, _, report in reports):
        raise ValueError("nonempty continuous measurements required")
    by_label = {}
    failures = []
    quality_phases = 0
    for pair, speaker, report in reports:
        for segment in report["details"]:
            stats = by_label.setdefault(segment["label"], dict(sequences=0, passed=0, failed_gates=0, unavailable_gates=0))
            stats["sequences"] += 1
            stats["passed"] += int(segment["passed"])
            stats["failed_gates"] += sum(g["status"] == "fail" for g in segment["gates"])
            stats["unavailable_gates"] += sum(g["status"] == "unavailable" for g in segment["gates"])
            quality_phases += all(g["status"] == "pass" or (g["status"] == "unavailable" and g["optional"])
                                  for g in segment["gates"] if g["segment"] != "runtime")
        failures.extend(dict(pair=pair, near_speaker=speaker, **gate) for gate in report["failures"])
    return dict(sequences=len(reports), phases=sum(len(r["details"]) for _, _, r in reports),
                passed_phases=sum(s["passed"] for s in by_label.values()),
                quality_passed_phases=quality_phases,
                passed_sequences=sum(r["all_available_passed"] for _, _, r in reports),
                required_measurements_complete=all(r["required_measurements_complete"] for _, _, r in reports),
                all_available_passed=all(r["all_available_passed"] for _, _, r in reports),
                by_label=by_label, failed_gates=len(failures), failures=failures,
                failed_gate_counts=dict(collections.Counter(f"{g['label']}/{g['segment']}/{g['metric']}" for g in failures)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--microphones", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--sample-rate", type=int, choices=(16000, 24000, 48000), default=16000)
    parser.add_argument("--radio-dataset", type=Path)
    parser.add_argument("--pairs", help="Optional comma-separated frozen speaker pairs")
    parser.add_argument("--simulate-clock", action="store_true")
    parser.add_argument("--cli-option", action="append", default=[])
    parser.add_argument("--pace", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1 or args.jobs > 4 or (args.pace and args.jobs != 1):
        parser.error("jobs must be 1..4; paced timing requires a single worker")
    dataset, root, cli = args.dataset.resolve(), args.root.resolve(), args.cli.resolve()
    manifest = json.loads((dataset / "inputs-manifest.json").read_text())
    radio_manifest_sha256 = None
    if args.radio_dataset:
        radio = args.radio_dataset.resolve()
        radio_manifest = json.loads((radio / "inputs-manifest.json").read_text())
        radio_manifest_sha256 = ev.digest(radio / "inputs-manifest.json")
        manifest["scenes"] += [dict(scene, source_dataset=str(radio))
                               for scene in radio_manifest["scenes"] if scene.get("playback_source") == "radio"]
    pairs = sorted({s["pair"] for s in manifest["scenes"] if s["mics"] == args.microphones})
    if args.pairs is not None:
        requested = [int(value) for value in args.pairs.split(",")]
        if not requested or len(set(requested)) != len(requested) or not set(requested).issubset(pairs):
            raise ValueError("unique existing speaker pairs required")
        pairs = sorted(requested)
    if not pairs:
        raise ValueError("no speaker pairs")
    speakers = {}
    for pair in pairs:
        timeline = select_timeline(manifest["scenes"], pair, args.microphones)
        selected = {s["near_speaker"] for _, s, _, _ in timeline}
        if len(selected) != 1:
            raise ValueError("one speaker required per continuous sequence")
        speakers[pair] = selected.pop()
    root.mkdir(parents=True, exist_ok=args.resume)
    artifact = root / "itu_audio_cli"
    if not artifact.exists():
        shutil.copy2(cli, artifact)
    if ev.digest(cli) != ev.digest(artifact):
        raise ValueError("resume CLI identity changed")
    identity = dict(source_manifest_sha256=ev.digest(dataset / "inputs-manifest.json"),
                    phase_labels=[label for label, _, _, _ in VEHICLE_TIMELINE],
                    seconds_per_sequence=sum(end - begin for _, _, begin, end in VEHICLE_TIMELINE),
                    cli_sha256=ev.digest(artifact), rules_sha256=ev.digest(ac.RULES),
                    runner_sha256=ev.digest(Path(__file__).with_name("run_vehicle_stream.py")),
                    regression_runner_sha256=ev.digest(Path(__file__)),
                    pairs=pairs, near_speakers=speakers, jobs=args.jobs, paced=args.pace,
                    microphones=args.microphones,
                    sample_rate=args.sample_rate, simulated_clock=args.simulate_clock,
                    diagnostic_overrides=args.cli_option, radio_source_manifest_sha256=radio_manifest_sha256,
                    scope="host continuous replay; unpaced runs validate quality only",
                    final_acceptance_eligible=args.pace and not args.cli_option and not args.simulate_clock
                    and args.sample_rate == ev.RATE)
    lock = root / "identity.json"
    if lock.exists():
        previous = json.loads(lock.read_text())
        # The driver may gain resumability; the actual stream runner is frozen.
        for key in ("source_manifest_sha256", "cli_sha256", "rules_sha256", "runner_sha256", "pairs", "jobs", "paced"):
            if previous[key] != identity[key]:
                raise ValueError(f"resume identity changed: {key}")
        if previous.get("microphones", 1) != args.microphones:
            raise ValueError("resume microphone selection changed")
        for key, default in (("sample_rate", 16000), ("simulated_clock", False),
                             ("diagnostic_overrides", []), ("radio_source_manifest_sha256", None)):
            if previous.get(key, default) != identity[key]:
                raise ValueError(f"resume processing options changed: {key}")
    else:
        lock.write_text(json.dumps(identity, indent=2) + "\n")

    def execute(pair):
        folder = root / f"pair-{pair:02d}"
        argv = [sys.executable, str(Path(__file__).with_name("run_vehicle_stream.py")),
                "--dataset", str(dataset), "--root", str(folder), "--cli", str(artifact), "--pair", str(pair),
                "--microphones", str(args.microphones), "--sample-rate", str(args.sample_rate)]
        if args.radio_dataset:
            argv.extend(["--radio-dataset", str(args.radio_dataset.resolve())])
        if args.simulate_clock:
            argv.append("--simulate-clock")
        argv.extend(f"--cli-option={option}" for option in args.cli_option)
        if args.pace:
            argv.append("--pace")
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        if not (folder / "stream-evaluation.json").exists():
            if folder.exists():
                raise ValueError(f"incomplete sequence retained; inspect before rerunning: {folder}")
            run = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=600)
            (root / f"pair-{pair:02d}.log").write_text(run.stdout + run.stderr)
            run.check_returncode()
        report = json.loads((folder / "stream-evaluation.json").read_text())
        for key in ("source_manifest_sha256", "cli_sha256", "rules_sha256"):
            if report["identity"][key] != identity[key]:
                raise ValueError(f"changed input or artifact identity: {key}")
        if (report["identity"]["processing_profile"] != "vehicle_stream" or
                report["identity"]["diagnostic_overrides"] != args.cli_option or
                report["identity"]["sample_rate"] != args.sample_rate or
                ("simulated_clock" in report["identity"]) != args.simulate_clock or
                report["identity"].get("radio_source_manifest_sha256") != radio_manifest_sha256):
            raise ValueError("sequence processing options changed")
        if report["identity"]["engine_resets"] != 0 or report["engine"]["paced_10ms"] != args.pace:
            raise ValueError("sequence reset or pacing changed")
        if report["identity"].get("microphones", 1) != args.microphones:
            raise ValueError("sequence microphone selection changed")
        print(json.dumps(dict(pair=pair, near_speaker=speakers[pair], failed_gates=len(report["failures"]))), flush=True)
        return pair, speakers[pair], report

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        reports = list(pool.map(execute, pairs))
    result = dict(identity=identity, **summarize(reports))
    (root / "regression-summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("failures", "identity")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
