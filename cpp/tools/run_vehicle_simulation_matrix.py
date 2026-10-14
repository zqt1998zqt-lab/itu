#!/usr/bin/env python3
"""Serial, identified host streams for all rates and microphone counts.

Quality replays, clock negative controls and paced timing stay separate. A
successful process exit means the batch completed, not that every gate passed.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import run_audio_eval as ev
import check_audio_acceptance as ac
from run_vehicle_stream import VEHICLE_TIMELINE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--cli", type=Path, required=True)
    p.add_argument("--paced", action="store_true", help="Also replay one 48k/4mic sequence at 10 ms intervals")
    a = p.parse_args()
    root, dataset = a.root.resolve(), a.dataset.resolve()
    root.mkdir(parents=True, exist_ok=False)
    binary = root / "itu_audio_cli"
    shutil.copy2(a.cli, binary)
    if ev.digest(ac.RULES) != ac.RULES_SHA256:
        raise ValueError("frozen engineering thresholds changed")
    manifest = json.loads((dataset / "inputs-manifest.json").read_text())
    pairs = sorted({s["pair"] for s in manifest["scenes"]})
    if not pairs or any({s["pair"] for s in manifest["scenes"] if s["mics"] == m} != set(pairs) for m in (1, 2, 4)):
        raise ValueError("same nonempty speaker coverage required for 1/2/4 microphones")
    tools = Path(__file__).resolve().parent
    identity = dict(cli_sha256=ev.digest(binary), manifest_sha256=ev.digest(dataset / "inputs-manifest.json"),
                    rules_sha256=ev.digest(ac.RULES), pairs=pairs, rates=[16000, 24000, 48000], microphones=[1, 2, 4],
                    runner_sha256={n: ev.digest(tools / n) for n in
                                   (Path(__file__).name, "run_vehicle_regression.py", "run_vehicle_stream.py", "evaluation_targets.py")},
                    phase_labels=[label for label, _, _, _ in VEHICLE_TIMELINE],
                    seconds_per_sequence=sum(end - begin for _, _, begin, end in VEHICLE_TIMELINE),
                    standard_compliance=False, scope="serial host simulation; no Android build or device claim")
    (root / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    rows = []

    def execute(name, command, artifact):
        with (root / f"{name}.log").open("w") as log:
            proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=1800)
        (root / f"{name}.invocation.json").write_text(json.dumps(dict(argv=command, exit_code=proc.returncode), indent=2) + "\n")
        proc.check_returncode()
        result = json.loads(artifact.read_text())
        print(json.dumps(dict(batch=name, artifact=str(artifact), completed=True)), flush=True)
        return result

    for rate in identity["rates"]:
        for mics in identity["microphones"]:
            for variant in (["control"] if mics == 1 else ["control", "spatial"]):
                name = f"r{rate}-m{mics}-{variant}"
                folder = root / name
                flags = ["--noise-transition-control"]
                if variant == "spatial":
                    flags += ["--spatial-mvdr", "--wind-control"]
                command = [sys.executable, str(tools / "run_vehicle_regression.py"), "--dataset", str(dataset),
                           "--root", str(folder), "--cli", str(binary), "--jobs", "1", "--sample-rate", str(rate),
                           "--microphones", str(mics), "--simulate-clock"] + [f"--cli-option={flag}" for flag in flags]
                result = execute(name, command, folder / "regression-summary.json")
                if (result["sequences"] != len(pairs) or result["phases"] != len(VEHICLE_TIMELINE) * len(pairs) or
                        result["identity"]["cli_sha256"] != identity["cli_sha256"] or
                        result["identity"]["source_manifest_sha256"] != identity["manifest_sha256"]):
                    raise ValueError("incomplete or changed stream matrix")
                rows.append(dict(rate=rate, microphones=mics, variant=variant,
                                 report=str((folder / "regression-summary.json").relative_to(root)),
                                 **{k: result[k] for k in ("sequences", "phases", "quality_passed_phases", "passed_phases", "failed_gate_counts")}))

    controls = []
    for rate in identity["rates"]:
        name = f"clock-negative-r{rate}-m4"
        folder = root / name
        command = [sys.executable, str(tools / "run_vehicle_stream.py"), "--dataset", str(dataset), "--root", str(folder),
                   "--cli", str(binary), "--pair", str(pairs[0]), "--microphones", "4", "--sample-rate", str(rate),
                   "--simulate-clock", "--disable-clock-correction", "--cli-option=--noise-transition-control"]
        result = execute(name, command, folder / "stream-evaluation.json")
        if result["identity"]["simulated_clock"]["correction"]:
            raise ValueError("clock negative control silently enabled correction")
        controls.append(dict(rate=rate, report=str((folder / "stream-evaluation.json").relative_to(root)),
                             failures=len(result["failures"])))
    paced = None
    if a.paced:
        name = "paced-r48000-m4-control"
        folder = root / name
        command = [sys.executable, str(tools / "run_vehicle_stream.py"), "--dataset", str(dataset), "--root", str(folder),
                   "--cli", str(binary), "--pair", str(pairs[0]), "--microphones", "4", "--sample-rate", "48000",
                   "--simulate-clock", "--pace", "--cli-option=--noise-transition-control"]
        result = execute(name, command, folder / "stream-evaluation.json")
        paced = dict(report=str((folder / "stream-evaluation.json").relative_to(root)),
                     **{k: result["engine"][k] for k in ("process_p95_ms", "process_max_ms", "process_frames_over_10ms", "completion_deadline_misses")})
    if ev.digest(binary) != identity["cli_sha256"]:
        raise ValueError("frozen CLI changed")
    report = dict(identity=identity, complete=True, rows=rows, clock_negative_controls=controls, paced=paced,
                  quality_phases=sum(r["phases"] for r in rows),
                  quality_passed_phases=sum(r["quality_passed_phases"] for r in rows))
    (root / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
