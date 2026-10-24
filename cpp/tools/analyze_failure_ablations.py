#!/usr/bin/env python3
"""Diagnostic ablations of frozen failed PCM, without changing acceptance results."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import shutil

import numpy as np

import run_controlled_native_scenes as ctl

PROJECT=Path(__file__).resolve().parents[2]
OUT=PROJECT/"build/failure-cause-analysis-20261017/ablations"
SOURCES=(("controlled-native-full-20261017","itu-controlled-quality-v1-20261017"),
         ("controlled-cavemove-20261017","itu-cavemove-quality-v1-20261017"))


def select():
    rows=[]
    for run,manifest in SOURCES:
        root=PROJECT/"build"/run
        core={s["id"] for s in json.loads((PROJECT/"datasets"/manifest/"manifest.json").read_text())["scenes"] if s["track"]=="spectrum_screened_core"}
        for line in (root/"results.jsonl").read_text().splitlines():
            row=json.loads(line)
            if row["scene"]["id"] in core:rows.append((root,row))
    choices=[]
    for rate in (16000,24000,48000):
        for kind in ("high_snr_ns","low_snr_gain","aec","agc"):
            candidates=[]
            for root,row in rows:
                s=row["scene"];snr=s.get("snr_db");gates=[g for g in row["acceptance"]["gates"] if g["status"]=="fail"]
                if s["sample_rate"]!=rate or s["mics"]!=1 or not gates:continue
                relevant=[]
                if kind=="high_snr_ns" and s["family"]=="ns" and snr==30:
                    relevant=[g for g in gates if g["metric"] in ("raw_sisdr_db","stoi_change")]
                elif kind=="low_snr_gain" and s["family"]=="ns" and snr==-10:
                    relevant=[g for g in gates if g["metric"]=="projection_gain_min_db"]
                elif kind=="aec" and s["family"]=="aec" and s["noise"]=="none" and s["stress"]=="none":
                    relevant=[g for g in gates if g["segment"]=="far_only"]
                elif kind=="agc" and s["family"]=="agc" and s["near_dbfs"]<=-40:
                    relevant=[g for g in gates if g["metric"]=="projection_gain_min_db"]
                if relevant:
                    deficit=max((g["threshold"]-g["value"])/(0.02 if g["metric"]=="stoi_change" else 1) for g in relevant)
                    candidates.append((deficit,str(root),s["id"],root,row))
            if candidates:
                _,_,_,root,row=max(candidates,key=lambda x:x[:3])
                choices.append((kind,root,row))
    return choices


def frame_summary(prefix,scene):
    with prefix.with_suffix(".frames.csv").open() as stream:frames=list(csv.DictReader(stream))
    result={}
    keys=("speech_probability","applied_gain_db","pre_ns_rms","post_ns_rms","gain_control_noise_power",
          "gain_speech_qualified","gain_control_state","noise_speech_mix","fullband_ns_mix",
          "ns_residual_gain","input_excess_score","linear_speech_probability","estimated_delay_ms","erle_db",
          "startup_dry_mix","fullband_noise_power","nearend_mix")
    for name,(a,b) in scene["windows"].items():
        selected=[r for r in frames if a<=float(r["time_s"])<b]
        values={}
        for key in keys:
            numbers=np.array([float(r[key]) for r in selected if key in r and r[key] not in ("","nan")])
            if len(numbers):values[key]=dict(zip(("min","p05","median","p95","max"),map(float,np.percentile(numbers,[0,5,50,95,100]))))
        result[name]=dict(frames=len(selected),columns=values)
    return result


def replay(arguments):
    kind,source,row=arguments
    ev,legacy=ctl.ev,ctl.legacy
    scene=row["scene"];folder=OUT/"cases"/scene["id"];folder.mkdir(parents=True)
    origin=source/"cases"/scene["id"]
    for path in origin.iterdir():
        if path.is_file() and path.name not in ("complete.json",):
            if path.name in scene["files"] and ev.digest(path)!=scene["files"][path.name]["sha256"]:raise ValueError(f"input changed: {path}")
            (folder/path.name).symlink_to(path.resolve())
        elif path.name=="oracle":(folder/path.name).symlink_to(path.resolve(),target_is_directory=True)
    cli=OUT/"artifacts/itu_audio_cli";sha=ev.digest(cli)
    variants={"full":[],"no_agc":["--no-agc"],"no_ns":["--no-ns"],
              "no_ns_agc":["--no-ns","--no-agc"],"no_lowcut":["--no-low-cut"],
              "frontend_only":["--no-ns","--no-agc","--no-low-cut"]}
    if kind=="aec":
        variants.update(no_reference_guard=["--no-reference-guard"],no_linear_protection=["--no-linear-speech-protection"],
                        long_filter=["--aec-filter-blocks","50"],
                        known_delay_diagnostic=["--external-delay-ms",str(round(scene["echo_delay_ms"]))])
    output=[]
    for name,flags in variants.items():
        mode="diagnostic_"+name;prefix=folder/"runs"/mode/"result"
        command=ctl.command_for(scene,folder,prefix,str(cli))+flags
        if name=="full":command+=["--diagnostic-stages"]
        legacy.invoke(command,prefix,sha)
        result=legacy.measure(scene,OUT,mode)
        result["diagnostic_kind"]=kind;result["variant"]=name;result["original_run"]=str(source)
        result["frame_summary"]=frame_summary(prefix,scene)
        result["baseline_output_matches_frozen"]=(result["metrics"]["output_sha256"]==row["metrics"]["output_sha256"]) if name=="full" else None
        result["scope"]="diagnostic ablation only; disabled modules and known delay are not candidate acceptance"
        output.append(result)
    return output


def main():
    choices=select()
    OUT.mkdir(parents=True,exist_ok=False);(OUT/"artifacts").mkdir()
    cli=PROJECT/"build/vcp-topology-20261016/cpp/itu_audio_cli"
    shutil.copy2(cli,OUT/"artifacts/itu_audio_cli")
    dependencies=[Path(__file__),Path(ctl.__file__),Path(ctl.legacy.__file__),Path(ctl.ev.__file__),
                  Path(ctl.legacy.acceptance.__file__),ctl.legacy.acceptance.RULES]
    bindings={str(p):ctl.ev.digest(p) for p in dependencies}
    input_bindings={}
    for run,manifest in SOURCES:
        root=PROJECT/"build"/run;identity=json.loads((root/"identity.json").read_text())
        if ctl.ev.digest(cli)!=identity["cli_sha256"]:raise ValueError("CLI differs")
        for p,h in identity["dependencies"].items():
            if ctl.ev.digest(Path(p))!=h:raise ValueError(f"source identity differs: {p}")
        for p in (root/"identity.json",root/"results.jsonl",PROJECT/"datasets"/manifest/"manifest.json"):
            input_bindings[str(p)]=ctl.ev.digest(p)
    ctl.legacy.save(OUT/"identity.json",dict(dependencies=bindings,inputs=input_bindings,cli_sha256=ctl.ev.digest(cli),
                     scope="frozen input diagnostic; no certification or replacement of original results"))
    ctl.legacy.save(OUT/"selection.json",[dict(kind=k,run=str(p),id=r["scene"]["id"]) for k,p,r in choices])
    results=[]
    with ProcessPoolExecutor(max_workers=2) as pool,(OUT/"results.jsonl").open("x") as stream:
        for future in as_completed([pool.submit(replay,args) for args in choices]):
            rows=future.result();results.extend(rows)
            for row in rows:stream.write(json.dumps(row)+"\n")
            stream.flush();print(json.dumps(dict(completed_variants=len(results),last_case=rows[0]["scene"]["id"])),flush=True)
    changed=[p for p,h in {**bindings,**input_bindings}.items() if ctl.ev.digest(Path(p))!=h]
    controls=[r for r in results if r["variant"]=="full"]
    verified=not changed and all(r["baseline_output_matches_frozen"] for r in controls) and len(controls)==len(choices)
    ctl.legacy.save(OUT/"report.json",dict(state="completed",cases=len(choices),invocations=len(results),
        frozen_full_reproduced=sum(r["baseline_output_matches_frozen"] for r in controls),
        identity_verified=verified,changed_dependencies=changed,standard_compliance=False))
    print(json.dumps(dict(completed_variants=len(results),identity_verified=verified)),flush=True)


if __name__=="__main__":main()
