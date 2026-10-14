#!/usr/bin/env python3
"""Replay identified mono vehicle timelines without resetting a model per phase.

Reuses frozen PCM from an earlier timeline, but does not simulate device
timestamps or ASRC. Thus it is an algorithm diagnostic, not a paired clock
correction comparison with the earlier WebRTC run.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess

import check_audio_acceptance as ac
import run_audio_eval as ev
import run_localvqe_ab as ab


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("matrix", "root", "probe", "library", "models"):
        p.add_argument("--"+name, type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    artifacts = root / "artifacts"
    artifacts.mkdir()
    snapshots = {}
    for key, path in (("probe", a.probe), ("library", a.library),
                      ("model", a.models/ab.MODELS["compact"][0]), ("frontend", a.models/ab.MODELS["frontend"][0])):
        target = artifacts/path.name
        shutil.copy2(path, target)
        snapshots[key] = target.resolve()
    for key, expected in (("model", ab.MODELS["compact"][1]), ("frontend", ab.MODELS["frontend"][1])):
        if ev.digest(snapshots[key]) != expected:
            raise ValueError("pinned model hash mismatch")
    if ev.digest(ac.RULES) != ac.RULES_SHA256:
        raise ValueError("frozen thresholds changed")
    rules = json.loads(ac.RULES.read_text())
    # Source PCM is validated at each rate; score target stays the original
    # 16k channel-0 signal. It is never sent to the C++ model.
    sources = sorted((a.matrix/"r16000-m1-control").glob("pair-*/timeline.json"))
    if not sources:
        raise ValueError("nonempty mono timeline inventory required")
    rows = []
    identity = dict(probe_sha256=ev.digest(snapshots["probe"]), library_sha256=ev.digest(snapshots["library"]),
        model_sha256=ev.digest(snapshots["model"]), rules_sha256=ev.digest(ac.RULES),
        runner_sha256=ev.digest(Path(__file__)), measure_runner_sha256=ev.digest(Path(ab.__file__)),
        scope="mono compact neural diagnostic, one state per continuous sequence; no timestamp clock correction; no phase resets",
        native_highband=False, standard_compliance=False, sequences=[])
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    for timeline_path in sources:
        base = timeline_path.parent
        base_timeline = json.loads(timeline_path.read_text())
        for rate in (16000,24000,48000):
            input_root = a.matrix/f"r{rate}-m1-control"/base.name
            timeline = json.loads((input_root/"timeline.json").read_text())
            if timeline["microphones"] != 1 or timeline["segments"] != base_timeline["segments"]:
                raise ValueError("timeline identities differ")
            for name, sha in timeline["files"].items():
                if ev.digest(input_root/name) != sha:
                    raise ValueError("frozen continuous PCM changed")
            folder = root/f"{base.name}-r{rate}"
            folder.mkdir()
            for name in ("before.wav","ref.wav"):
                shutil.copy2(input_root/name,folder/name)
            run = folder/"compact"
            run.mkdir()
            command = [str(snapshots["probe"]),"--library",str(snapshots["library"]),"--model",str(snapshots["model"]),
                "--mic",str(folder/"before.wav"),"--echo",str(folder/"ref.wav"),"--out",str(run/"result")]
            proc = subprocess.run(command,capture_output=True,text=True,timeout=180,env=env)
            (run/"process.log").write_text(proc.stdout+proc.stderr)
            (run/"invocation.json").write_text(json.dumps(dict(argv=command,exit_code=proc.returncode),indent=2)+"\n")
            proc.check_returncode()
            prefix = run/"result"
            engine = json.loads(prefix.with_suffix(".json").read_text())
            if engine["input_samples_per_channel"] != round(timeline["seconds"]*rate):
                raise ValueError("continuous input length changed")
            seq = dict(pair=base.name,rate=rate,timeline_sha256=ev.digest(input_root/"timeline.json"),
                input_sha256={name:ev.digest(folder/name) for name in ("before.wav","ref.wav")},
                output_sha256=ev.digest(prefix.with_suffix(".after.wav")),
                frames_sha256=ev.digest(prefix.with_suffix(".frames.csv")),runner_explicit_resets=0,
                sdk_internal_reset_count=None,sdk_internal_reset_count_measured=False,engine=engine)
            identity["sequences"].append(seq)
            for segment in timeline["segments"]:
                scene = dict(segment["scene"], windows=segment["windows"])
                checked, measured = ab.measure(scene,base,prefix,rate)
                verdict = ac.check_scene(checked,measured,rules)
                gates = [g for g in verdict["gates"] if g["segment"] != "runtime"]
                rows.append(dict(pair=base.name,rate=rate,label=segment["label"],measurements=measured,acceptance=verdict,
                    quality_passed=all(g["status"]=="pass" or g["status"]=="unavailable" and g["optional"] for g in gates)))
            print(json.dumps(dict(pair=base.name,rate=rate,phases=len(timeline["segments"]),resets=0)),flush=True)
    report = dict(identity=identity,rows=rows,phases=len(rows),quality_pass=sum(r["quality_passed"] for r in rows),
        failed_metrics=dict(Counter(g["metric"] for r in rows for g in r["acceptance"]["gates"] if g["status"]!="pass" and not g["optional"])))
    (root/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
