#!/usr/bin/env python3
"""Serial C++ paired runs on the independently audited cabin-field extension."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import subprocess
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import run_audio_eval as ev
import check_audio_acceptance as ac
from evaluation_targets import calibrated_beam_target


def main():
    p=argparse.ArgumentParser(); p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--root",type=Path,required=True); p.add_argument("--cli",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True); p.add_argument("--remeasure",action="store_true")
    p.add_argument("--scene",action="append",default=[],help="Exact scene id; repeat for focused paired diagnostics")
    p.add_argument("--rates",type=int,nargs="+",choices=(16000,24000,48000),default=[16000,24000,48000])
    p.add_argument("--variants",nargs="+",choices=("baseline","control","spatial","rnnoise","ulunas","single_render_negative"))
    p.add_argument("--cli-option",action="append",default=[],help="Common diagnostic override, retained in identity")
    a=p.parse_args(); a.root.mkdir(parents=True,exist_ok=a.remeasure)
    manifest=json.loads((a.dataset/"inputs-manifest.json").read_text()); rules=json.loads(ac.RULES.read_text())
    assert ev.digest(ac.RULES)==ac.RULES_SHA256
    identity=dict(cli_sha256=ev.digest(a.cli),model_sha256=ev.digest(a.model),
        manifest_sha256=ev.digest(a.dataset/"inputs-manifest.json"),rules_sha256=ev.digest(ac.RULES),
        scope="serial host diagnostic; unchanged engineering thresholds; not formal ITU or paced device acceptance",
        rates=[16000,24000,48000],native_highband=False,
        target_scope="output uses fixed geometry at actual core rate and declared latency; input baseline uses channel-0 clean")
    identity.update(runner_sha256=ev.digest(Path(__file__)),scene_filter=a.scene,
                    rate_filter=a.rates,variant_filter=a.variants,diagnostic_overrides=a.cli_option,
                    target_runner_sha256=ev.digest(Path(__file__).with_name("evaluation_targets.py")))
    scenes=[s for s in manifest["scenes"] if not a.scene or s["id"] in a.scene]
    if not scenes or set(a.scene)-{s["id"] for s in scenes}: p.error("unknown or empty scene selection")
    snapshot=a.root/"itu_audio_cli"
    import shutil
    if not snapshot.exists(): shutil.copy2(a.cli,snapshot)
    assert ev.digest(snapshot)==identity["cli_sha256"]
    cli=snapshot.resolve()
    rows=[]
    for scene in scenes:
        source=a.dataset/"cases"/scene["id"]
        for name,record in scene["files"].items(): assert ev.digest(source/name)==record["sha256"]
        for rate in a.rates:
            if rate!=16000 and scene["snr_db"]!=15: continue
            root=a.root/f"{scene['id']}-r{rate}"; root.mkdir(exist_ok=a.remeasure)
            for name in ("before.wav","ref.wav"):
                if a.remeasure: continue
                x=ev.read(source/name)
                gcd=math.gcd(rate,16000)
                if rate!=16000: x=resample_poly(x,rate//gcd,16000//gcd,axis=0)
                sf.write(root/name,x,rate,subtype="FLOAT")
            before=ev.read(source/"before.wav")[:,0]; target=ev.read(source/"clean-beam.wav")[:,0]
            baseline_target=ev.read(source/"clean.wav")[:,0]
            clean_mics=ev.read(source/"clean-mics.wav") if scene["mics"]>1 else None
            gcd=math.gcd(rate,16000)
            variants={"baseline":[],"control":["--noise-transition-control"]}
            if scene["mics"]>1:
                variants["spatial"]=["--noise-transition-control","--spatial-mvdr","--wind-control"]
            if scene["snr_db"]==15:
                variants["rnnoise"]=["--noise-transition-control","--ns-backend","rnnoise"]
                variants["ulunas"]=["--noise-transition-control","--ns-backend","ulunas","--neural-model",str(a.model.resolve())]
            if scene.get("reference_channels",1)>1:
                variants["single_render_negative"]=["--no-reference-guard","--no-linear-speech-protection",
                    "--analyze-nonlinear-noise","--no-coherence-dtd","--internal-delay"]
                if not a.remeasure:
                    reference,sr=sf.read(root/"ref.wav",always_2d=True)
                    sf.write(root/"ref-channel0.wav",reference[:,0],sr,subtype="FLOAT")
            for variant,flags in variants.items():
                if a.variants and variant not in a.variants: continue
                target_alignment=None
                if clean_mics is not None:
                    core_rate=48000 if rate==24000 or rate==16000 and variant=="rnnoise" else rate
                    target,target_alignment=calibrated_beam_target(clean_mics,scene["beam_delays_samples"],rate,core_rate=core_rate)
                run=root/variant; run.mkdir(exist_ok=a.remeasure); prefix=run/"result"
                command=[str(cli),"--profile","vehicle_stream","--mic",str(root/"before.wav"),
                    "--echo",str(root/("ref-channel0.wav" if variant=="single_render_negative" else "ref.wav")),"--out",str(prefix)]
                if scene["mics"]>1:
                    command += ["--beam-delays",",".join(str(v*rate/16000) for v in scene["beam_delays_samples"])]
                command+=flags+a.cli_option
                if not a.remeasure:
                    completed=subprocess.run(command,capture_output=True,text=True,timeout=120)
                    (run/"invocation.json").write_text(json.dumps(dict(argv=command,exit_code=completed.returncode,stderr=completed.stderr),indent=2)+"\n")
                    completed.check_returncode()
                else:
                    invocation=json.loads((run/"invocation.json").read_text())
                    assert invocation["exit_code"]==0 and invocation["argv"][1:]==command[1:]
                engine=json.loads(prefix.with_suffix(".json").read_text())
                output,sr=sf.read(prefix.with_suffix(".after.wav")); assert sr==rate
                delay=engine["dsp_delay_samples"]
                assert len(output)==len(before)*rate//16000+delay
                after_io=output[delay:]
                before_io,br=sf.read(root/"before.wav",always_2d=True)
                if br!=rate or len(before_io)!=len(after_io): raise ValueError("native attenuation contract mismatch")
                before_io=before_io[:,0]
                after=after_io
                if rate!=16000: after=resample_poly(after,16000//gcd,rate//gcd)
                assert len(after)==len(before) and np.isfinite(after).all()
                measurements={}; windows=dict(scene["windows"])
                if "cold_start" in windows: windows["near_cold_start"]=windows.pop("cold_start")
                for name,bounds in windows.items():
                    b,z=ev.window(before,bounds),ev.window(after,bounds)
                    if name.startswith("near") or name=="double_talk":
                        clean=ev.window(target,bounds); qb,qa=ev.quality(ev.window(baseline_target,bounds),b),ev.quality(clean,z)
                        measurements[name]=dict(before=qb,after=qa,stoi_improvement=qa["stoi"]-qb["stoi"] if qa["stoi"] is not None and qb["stoi"] is not None else None)
                    else: measurements[name]=ev.attenuation_measurement(ev.window(before_io,bounds,rate),ev.window(after_io,bounds,rate),True)
                levels=dict(peak_dbfs=ev.db(float(np.max(np.abs(output)))**2),clipping_percent=float(np.mean(np.abs(output)>=.999)*100))
                measured=dict(mode="full",segments=measurements,engine=engine,
                    process_frames_over_10ms=engine["process_frames_over_10ms"],output_levels=levels)
                verdict=ac.check_scene(dict(scene,windows=windows),measured,rules)
                quality_gates=[g for g in verdict["gates"] if g["segment"]!="runtime"]
                row=dict(id=scene["id"],pair=scene["pair"],label=scene["field_label"],snr=scene["snr_db"],negative_control=variant=="single_render_negative",
                    microphones=scene["mics"],rate=rate,variant=variant,measurements=measured,target_alignment=target_alignment,
                    quality_passed=all(g["status"]=="pass" or g["status"]=="unavailable" and g["optional"] for g in quality_gates),acceptance=verdict)
                row["pcm_sha256"]={path.name:ev.digest(path) for path in [root/"before.wav",root/"ref.wav",prefix.with_suffix(".after.wav")]}
                rows.append(row); print(json.dumps(dict(id=row["id"],rate=rate,variant=variant,passed=verdict["passed"],quality=row["quality_passed"])),flush=True)
                (run/"measurements.json").write_text(json.dumps(row,indent=2,allow_nan=False)+"\n")
                (a.root/"progress.json").write_text(json.dumps(dict(done=len(rows),last=dict(id=row["id"],rate=rate,variant=variant)))+"\n")
    groups={}
    if not rows: raise ValueError("no applicable scene-rate-variant runs")
    for key in sorted({(r["rate"],r["variant"]) for r in rows}):
        selected=[r for r in rows if (r["rate"],r["variant"])==key]
        groups[f"{key[0]}-{key[1]}"]=dict(rows=len(selected),quality_pass=sum(r["quality_passed"] for r in selected),
            all_pass=sum(r["acceptance"]["passed"] for r in selected),
            failed_metrics=dict(Counter(g["metric"] for r in selected for g in r["acceptance"]["gates"] if g["status"]=="fail")))
    (a.root/"report.json").write_text(json.dumps(dict(identity=identity,groups=groups,rows=rows),indent=2,allow_nan=False)+"\n")
    print(json.dumps(groups,indent=2))


if __name__=="__main__": main()
