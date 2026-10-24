#!/usr/bin/env python3
"""Replay five frozen SNR conditions in one uninterrupted 120-second engine."""
import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np

import run_controlled_native_scenes as controlled
import run_audio_eval as ev
import run_native_rate_scenes_v3 as legacy
import audit_dataset_quality_deep as audit


def select_chunks(rows):
    selected={}
    for rate in (16000,24000,48000):
        for mics in (1,2,4):
            groups={}
            for row in rows:
                scene=row["scene"]
                if scene["sample_rate"]==rate and scene["mics"]==mics and scene["grid_kind"] in ("one_db_fixed_target_sweep","array_coverage_probe"):
                    group_id=scene["id"].rsplit("-snr",1)[0]
                    groups.setdefault(group_id,{})[scene["snr_db"]]=scene
            complete=[(name,group) for name,group in groups.items() if all(s in group for s in (-10,0,10,20,30))]
            if not complete:raise ValueError(f"missing complete continuous input group: {rate} Hz / {mics} microphones")
            name,group=sorted(complete)[0]
            selected[(rate,mics)]=[group[s] for s in (-10,0,10,20,30)]
    return selected


def check_source_identity(source,identity,status):
    if status["state"]!="completed" or not status.get("identity_verified") or status.get("execution_errors",0):
        raise ValueError("source run has incomplete execution or unverified identity")
    changed=[p for p,h in identity["dependencies"].items() if not Path(p).is_file() or ev.digest(Path(p))!=h]
    if changed:raise ValueError(f"source dependency identity differs: {changed}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();source=args.run.resolve();root=args.out.resolve()
    status=json.loads((source/"report.json").read_text())
    identity=json.loads((source/"identity.json").read_text())
    check_source_identity(source,identity,status)
    cli=source/"artifacts/itu_audio_cli";cli_sha=ev.digest(cli)
    if cli_sha!=identity["cli_sha256"] or ev.digest(legacy.acceptance.RULES)!=legacy.RULES_SHA:
        raise ValueError("binary or gates differ")
    results=[json.loads(line) for line in (source/"results.jsonl").read_text().splitlines()]
    rows=[r for r in results if "scene" in r]
    selected=select_chunks(rows)
    root.mkdir(parents=True,exist_ok=False)
    (root/"artifacts").mkdir();shutil.copy2(cli,root/"artifacts/itu_audio_cli")
    cli=root/"artifacts/itu_audio_cli"
    dependencies=[Path(__file__),Path(controlled.__file__),Path(ev.__file__),Path(legacy.__file__),
                  Path(audit.__file__),Path(legacy.acceptance.__file__),legacy.acceptance.RULES,
                  source/"identity.json",source/"results.jsonl",source/"report.json",source/"plan.json"]
    bindings={str(p.resolve()):ev.digest(p) for p in dependencies}
    bindings.update(identity["dependencies"])
    legacy.save(root/"identity.json",dict(source_run=str(source),source_identity_sha256=ev.digest(source/"identity.json"),
                source_results_sha256=ev.digest(source/"results.jsonl"),cli_sha256=cli_sha,dependencies=bindings))
    output=[]
    for rate in (16000,24000,48000):
        ev.RATE=rate
        for mics in (1,2,4):
            chunks=selected[(rate,mics)];first=chunks[0]
            scene=copy.deepcopy(first);scene["id"]=f"continuous-{rate}-m{mics}"
            scene["seconds"]=sum(c["seconds"] for c in chunks)
            scene["grid_kind"]="continuous_variable_snr"
            scene["near_schedule"]=[];scene["windows"]={};scene["snr_schedule"]=[]
            components={name:[] for name in ("before.wav","clean.wav","clean-mics.wav","clean-beam.wav",
                                              "noise-component.wav","echo-component.wav","ref.wav","playback.wav")}
            offset=0
            for i,chunk in enumerate(chunks):
                folder=source/"cases"/chunk["id"]
                for name in components:
                    path=folder/name
                    if ev.digest(path)!=chunk["files"][name]["sha256"]:
                        raise ValueError(f"input changed: {path}")
                    components[name].append(ev.read(path))
                scene["near_schedule"].extend([[a+offset,b+offset] for a,b in chunk["near_schedule"]])
                for key in ("near_only","near_late"):
                    a,b=chunk["windows"][key]
                    scene["windows"][f"near_chunk{i}_{key}"]=[a+offset,b+offset]
                if i==len(chunks)-1:
                    a,b=chunk["windows"]["noise_only"];scene["windows"]["noise_only"]=[a+offset,b+offset]
                scene["snr_schedule"].append(dict(start_s=offset,stop_s=offset+chunk["seconds"],snr_db=chunk["snr_db"],source_case=chunk["id"]))
                offset+=chunk["seconds"]
            folder=root/"cases"/scene["id"];folder.mkdir(parents=True)
            for name,pieces in components.items():ev.write(folder/name,np.concatenate(pieces),"FLOAT")
            mask=ev.mask_windows(scene["seconds"],scene["near_schedule"])
            clean=ev.read(folder/"clean-mics.wav")[:,0];noise=ev.read(folder/"noise-component.wav")[:,0]
            scene["snr_db"]=ev.db(ev.power(clean[mask])/ev.power(noise[mask]))
            scene["measured_input_snr_db"]=scene["snr_db"]
            scene["snr_scope"]="variable piecewise schedule; nominal scalar is overall scheduled-window ratio"
            scene["effective_near_dbfs"]=ev.db(ev.power(clean[mask]))
            scene["continuous_scope"]="one uninterrupted C++ engine, 120 s offline 10 ms callbacks; deterministic repeated target, not five independent utterances"
            scene["files"]={p.name:dict(sha256=ev.digest(p),bytes=p.stat().st_size) for p in folder.iterdir() if p.is_file()}
            legacy.save(folder/"scene.json",scene)
            checked=audit.check_scene(("continuous",folder,scene))
            if checked["errors"]:raise ValueError(str(checked["errors"]))
            prefix=folder/"runs/full/result"
            legacy.invoke(controlled.command_for(scene,folder,prefix,str(cli)),prefix,cli_sha)
            row=legacy.measure(scene,root,"full")
            row["onset_diagnostics"]=controlled.early_diagnostics(scene,root,"full")
            row["input_audit"]=checked
            output.append(row)
            with (root/"results.jsonl").open("a") as stream:stream.write(json.dumps(row)+"\n")
            print(json.dumps(dict(completed=len(output),id=scene["id"],status=row["acceptance"]["status"])),flush=True)
    changed=[p for p,h in bindings.items() if ev.digest(p)!=h]
    report=dict(state="completed",cases=len(output),seconds_per_case=120,continuous_callback_frames_per_case=12000,
                identity_verified=not changed,source_dependencies_verified=not changed,changed_dependencies=changed,summary=legacy.summary(output),
                standard_compliance=False,scope="supplementary offline state-continuity test; paced realtime tests are in the source run")
    legacy.save(root/"report.json",report);print(json.dumps(report))


if __name__=="__main__":main()
