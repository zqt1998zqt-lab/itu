#!/usr/bin/env python3
"""Derive transparent spectrum-screen tracks without changing frozen cases or gates."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil

import numpy as np
import soundfile as sf

import run_audio_eval as ev
from audit_dataset_quality import target_spectrum


def inspect_target(item):
    run,scene=item
    path=run/"cases"/scene["id"]/"clean.wav"
    audio,rate=sf.read(path,dtype="float64",always_2d=True)
    if rate!=scene["sample_rate"] or not np.isfinite(audio).all():raise ValueError(f"invalid clean: {path}")
    near=np.concatenate([audio[round(a*rate):round(b*rate)] for a,b in scene["near_schedule"]])
    return dict(sha256=scene["files"]["clean.wav"]["sha256"],sample_rate=rate,
                near_schedule_seconds=scene["near_schedule"],full=target_spectrum(audio,rate),near=target_spectrum(near,rate))


def screen(metrics):
    if metrics["ac_below_60hz_percent"] is None:return ["no_ac_target_power"]
    return [name for name,hit in (("ac_below_60hz_percent",metrics["ac_below_60hz_percent"]>5),
                                 ("dc_dbfs",metrics["dc_dbfs"]>-50)) if hit]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--evidence",type=Path)
    args=parser.parse_args();run=args.run.resolve();out=args.out.resolve()
    results=run/"results.jsonl";snapshot=ev.digest(results)
    rows=[json.loads(line) for line in results.read_text().splitlines()]
    if any("scene" not in row for row in rows):raise ValueError("execution-error rows require manual review")
    if len({r["scene"]["id"] for r in rows})!=len(rows):raise ValueError("duplicate scene IDs")
    targets={}
    for row in rows:
        scene=row["scene"];sha=scene["files"]["clean.wav"]["sha256"]
        if ev.digest(run/"cases"/scene["id"]/"clean.wav")!=sha:raise ValueError("clean SHA changed")
        if sha in targets and (targets[sha]["sample_rate"],targets[sha]["near_schedule"])!=(scene["sample_rate"],scene["near_schedule"]):
            raise ValueError("identical file with inconsistent schedule/rate")
        targets[sha]=scene
    if args.evidence:
        evidence=json.loads(args.evidence.read_text())
        if evidence["input_snapshot"]["sha256"]!=snapshot:raise ValueError("spectrum evidence results snapshot differs")
        metrics={r["sha256"]:r for r in evidence["targets"]}
        if set(metrics)!=set(targets):raise ValueError("spectrum evidence target set differs")
        for sha,scene in targets.items():
            record=metrics[sha]
            if record["sample_rate"]!=scene["sample_rate"] or record["near_schedule_seconds"]!=scene["near_schedule"]:
                raise ValueError("spectrum evidence alignment differs")
    else:
        with ThreadPoolExecutor(max_workers=4) as pool:
            metrics={r["sha256"]:r for r in pool.map(inspect_target,[(run,s) for s in targets.values()])}
    entries=[];counts=defaultdict(Counter)
    for row in rows:
        scene=row["scene"];record=metrics[scene["files"]["clean.wav"]["sha256"]]
        reasons=screen(record["near"]);track="diagnostic_spectrum_review" if reasons else "spectrum_screened_core"
        entries.append(dict(id=scene["id"],scene_path=str(run/"cases"/scene["id"]/"scene.json"),
            scene_sha256=ev.digest(run/"cases"/scene["id"]/"scene.json"),clean_sha256=record["sha256"],
            sample_rate=scene["sample_rate"],near_speaker=scene["near_speaker"],family=scene["family"],
            noise=scene["noise"],split=scene["split"],track=track,spectrum_reasons=reasons,
            existing_acceptance_status=row["acceptance"]["status"]))
        counts[track][row["acceptance"]["status"]]+=1
    if ev.digest(results)!=snapshot:raise ValueError("source results changed during classification")
    out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,out/"classifier-frozen.py")
    result=dict(schema_version=1,source_run=str(run),source_results_sha256=snapshot,
        source_identity_sha256=ev.digest(run/"identity.json"),thresholds=dict(ac_below_60hz_percent_max=5,dc_dbfs_max=-50),
        selection_scope="near_schedule engineering spectrum/DC screen only; not global clean certification; no selection by algorithm gate or speaker history",
        result_scope="all original results preserved; track changes are data classification, not DSP improvements",
        counts={k:dict(v) for k,v in counts.items()},scenes=entries,standard_compliance=False)
    (out/"manifest.json").write_text(json.dumps(result,indent=2)+"\n")
    (out/"target-spectrum.json").write_text(json.dumps(list(metrics.values()),indent=2)+"\n")
    print(json.dumps(dict(scenes=len(entries),unique_clean_container_hashes=len(targets),counts=result["counts"])))


if __name__=="__main__":main()
