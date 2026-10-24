#!/usr/bin/env python3
"""Independent input audit and clustered summary for a completed native run."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
from pathlib import Path
import shutil

import numpy as np
import soundfile as sf

import audit_dataset_quality_deep as audit


def target_counts(root, rows):
    container={};pcm={}
    for name in ("clean.wav","clean-mics.wav","clean-beam.wav"):
        unique={r["scene"]["files"][name]["sha256"]:r for r in rows}
        signatures=set()
        for row in unique.values():
            audio,rate=sf.read(root/"cases"/row["scene"]["id"]/name,dtype="float64",always_2d=True)
            signatures.add((rate,audio.shape,hashlib.sha256(audio.tobytes()).hexdigest()))
        container[name]=len(unique);pcm[name]=len(signatures)
    return container,pcm


def expected_groups(groups):
    expected={}
    for group in groups:
        for snr in group["grid"]:
            key=group["id"]+(f"-snr{snr:+03d}" if snr is not None else "")
            if key in expected:raise ValueError("duplicate planned scene ID")
            expected[key]=group["id"]
    return expected


def paired_waveforms(item):
    root,group,rows=item
    findings=[]
    ordered=sorted(rows,key=lambda r:r["scene"]["snr_db"])
    if len(ordered)<2 or group["family"]!="ns" or group.get("noise_steps"):
        return None
    containers,targets=target_counts(root,rows)
    levels=[r["input_audit"]["near_scheduled_rms_dbfs"] for r in rows]
    seeds={r["scene"]["seed"] for r in rows}
    starts={r["scene"]["noise_source_start_sample"] for r in rows}
    reference=None;maximum=0.
    for row in ordered:
        scene=row["scene"]
        noise,rate=sf.read(root/"cases"/scene["id"]/"noise-component.wav",dtype="float64",always_2d=True)
        unit=noise*10**(scene["snr_db"]/20)
        if reference is None:reference=unit
        else:
            error=float(np.linalg.norm(unit-reference)/max(np.linalg.norm(reference),1e-20))
            maximum=max(maximum,error)
    if any(v!=1 for v in targets.values()):findings.append("target_pcm_drift")
    if max(levels)-min(levels)>1e-8:findings.append("target_level_drift")
    if maximum>1e-6:findings.append("noise_waveform_changed")
    if len(seeds)!=1 or len(starts)!=1:findings.append("seed_or_noise_offset_changed")
    return dict(group_id=group["id"],kind=group["grid_kind"],scenes=len(rows),target_container_hash_counts=containers,target_pcm_hash_counts=targets,
                target_level_span_db=max(levels)-min(levels),noise_relative_error_max=maximum,
                snr_grid=[r["scene"]["snr_db"] for r in ordered],errors=findings)


def safe_paired_waveforms(item):
    try:return paired_waveforms(item)
    except Exception as error:
        _,group,rows=item
        return dict(group_id=group["id"],kind=group["grid_kind"],scenes=len(rows),
                    target_level_span_db=0.,noise_relative_error_max=0.,
                    errors=["pair_inputs_unreadable:"+type(error).__name__+":"+str(error)])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--jobs",type=int,choices=range(1,9),default=4)
    args=parser.parse_args()
    root=args.run.resolve();out=args.out.resolve()
    report=json.loads((root/"report.json").read_text())
    if not report["state"].startswith("completed"):
        raise ValueError("run must finish before independent audit")
    plan=json.loads((root/"plan.json").read_text())["groups"]
    identity=json.loads((root/"identity.json").read_text())
    changed=[p for p,h in identity["dependencies"].items() if not Path(p).is_file() or audit.digest(p)!=h]
    if audit.digest(root/"artifacts/itu_audio_cli")!=identity["cli_sha256"]:
        changed.append(str(root/"artifacts/itu_audio_cli"))
    rows=[json.loads(line) for line in (root/"results.jsonl").read_text().splitlines()]
    valid=[r for r in rows if "scene" in r]
    by_id={r["scene"]["id"]:r for r in valid}
    if len(by_id)!=len(valid):raise ValueError("duplicate result IDs")
    expected=expected_groups(plan)
    missing=sorted(set(expected)-set(by_id));extra=sorted(set(by_id)-set(expected))
    planned_rows=[r for r in valid if r["scene"]["id"] in expected]
    out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,out/"analyzer-frozen.py")
    work=[("controlled_native",root/"cases"/r["scene"]["id"],r["scene"]) for r in valid]
    input_rows=[]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool,(out/"input-scenes.jsonl").open("x") as stream:
        for i,row in enumerate(pool.map(audit.check_scene,work),1):
            input_rows.append(row);stream.write(json.dumps(row)+"\n")
            if i%500==0:print(json.dumps(dict(input_rechecked=i,total=len(work))),flush=True)
    grouped=defaultdict(list)
    for r in valid:
        if r["scene"]["id"] in expected:grouped[expected[r["scene"]["id"]]].append(r)
    pairwork=[(root,g,grouped[g["id"]]) for g in plan]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        pairs=[p for p in pool.map(safe_paired_waveforms,pairwork) if p is not None]
    (out/"paired-sweeps.json").write_text(json.dumps(pairs,indent=2)+"\n")
    dimensions={}
    for key in ("sample_rate","split","language","mics","family","noise","snr_db","grid_kind"):
        counts=defaultdict(Counter)
        for row in planned_rows:
            counts[str(row["scene"].get(key))][row["acceptance"]["status"]]+=1
        dimensions[key]={k:dict(v) for k,v in sorted(counts.items())}
    speakers=defaultdict(list);files={};noise_files={};onsets=defaultdict(list)
    for row in planned_rows:
        scene=row["scene"]
        speakers[scene["near_speaker"]].append(float(row["acceptance"]["passed"]))
        for speaker,records in scene["source_speech_files"].items():
            for source in records:files[source["sha256"]]=dict(source,speaker=speaker)
        noise=scene.get("noise_source")
        if noise and noise.get("sha256"):noise_files[noise["sha256"]]=noise
        for name,value in row["onset_diagnostics"].items():
            if value.get("available"):
                onsets[name].append(value)
    means=np.array([np.mean(v) for v in speakers.values()])
    rng=np.random.default_rng(20261017)
    draws=np.mean(rng.choice(means,size=(5000,len(means)),replace=True),axis=1) if len(means) else np.array([])
    result=dict(state="completed",run=str(root),scenes=len(valid),planned_scenes=len(expected),
        missing=missing,extra=extra,changed_dependencies=changed,execution_errors=sum("error" in r for r in rows),
        input_integrity=dict(scenes=len(input_rows),files_checked=sum(r.get("files_checked",0) for r in input_rows),
                             errors=[r for r in input_rows if r["errors"]]),
        pair_integrity=dict(groups=len(pairs),full_one_db_groups=sum(p["kind"]=="one_db_fixed_target_sweep" for p in pairs),
                            maximum_target_span_db=max((p["target_level_span_db"] for p in pairs),default=0),
                            maximum_noise_shape_relative_error=max((p["noise_relative_error_max"] for p in pairs),default=0),
                            errors=[p for p in pairs if p["errors"]]),
        statistics_scope="only planned scene IDs; extra results retained as audit errors",statistics_scenes=len(planned_rows),
        status_counts=dict(Counter(r["acceptance"]["status"] for r in planned_rows)),by=dimensions,
        actual_source_usage=dict(near_speakers=len(speakers),speech_files=len(files),real_noise_files=len(noise_files),
                                 speech=files,noise=noise_files),
        speaker_cluster_summary=dict(speakers=len(means),speaker_macro_pass_rate=float(np.mean(means)),
            bootstrap_percentile_95=[float(v) for v in np.percentile(draws,[2.5,97.5])] if len(draws) else [],
            repetitions=5000,scope="speaker-level resampling; SNR combinations are correlated, not independent utterances"),
        onset_diagnostics={k:dict(available=len(v),projection_gain_db=audit.quantiles([r.get("projection_gain_db") for r in v]),
                                   nmse_db=audit.quantiles([r.get("nmse_db") for r in v])) for k,v in onsets.items()},
        snr_scope="scheduled-window calibration plus 20 ms energy-activity proxy; not ITU P.56 and not VAD labels",
        measurement_scope="same frozen CLI and engineering gates, changed matrix; before/after pass rates are not an algorithm improvement estimate",
        standard_compliance=False,negative_controls=report.get("negative_controls"),paced_checks=report.get("paced_checks"),
        run_identity_verified=report.get("identity_verified"))
    result["data_quality_scope"]="PCM integrity, mix/window calibration and paired sweeps; clean-spectrum eligibility is a separate audit"
    result["data_quality_verified"]=not (missing or extra or changed or result["execution_errors"] or result["input_integrity"]["errors"] or result["pair_integrity"]["errors"])
    (out/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("actual_source_usage","by","paced_checks","negative_controls")}),flush=True)


if __name__=="__main__":main()
