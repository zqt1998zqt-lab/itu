#!/usr/bin/env python3
"""Attach objective evidence to an identified complete matrix; publish original-amplitude stages."""
from __future__ import annotations
import argparse, copy, json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import soundfile as sf
from ecnr_campaign_calibration import save,sha
from ecnr_perceptual import DNSMOS,DEFAULT_ASSETS
from ecnr_window_audit import audit_stream
ALLOWED={"near_only","near_late","double_talk"}

def verify_sequence(item):
    for side in ("baseline","candidate"):
        report=Path(item[side+"_report"])
        if sha(report)!=item[side+"_report_sha256"]:raise ValueError("sequence report changed")
        data=json.loads(report.read_text())
        for name,expected in {**data["identity"]["files"],**data["identity"]["output_files"]}.items():
            if sha(report.parent/name)!=expected:raise ValueError("sequence audio/log changed")
    return True
def score_summary(pairs,expected):
    available=[x for x in pairs if x["baseline"]["available"] and x["candidate"]["available"]]
    scores=[]
    for metric in ("SIG","BAK","OVRL"):
        b=np.array([x["baseline"]["poly"][metric] for x in available])
        c=np.array([x["candidate"]["poly"][metric] for x in available])
        scores.append(dict(metric=metric,baseline_median=float(np.median(b)) if len(b) else None,
            candidate_median=float(np.median(c)) if len(c) else None,paired_delta_median=float(np.median(c-b)) if len(c) else None,windows=len(b)))
    return dict(status="completed" if len(available)==expected and expected else "unavailable",
        scope="Predicted perceptual DNSMOS P.835; no absolute MOS acceptance line; fixed-delay near speech windows",
        expected_windows=expected,paired_windows=len(available),scores=scores,
        short_repeated_count=sum(x[side].get("short_repeat",False) for x in pairs for side in ("baseline","candidate")),
        windows=pairs)
def envelope(path,bins=2400):
    values,rate=sf.read(path,dtype="float64",always_2d=True)
    if not len(values) or not np.isfinite(values).all():raise ValueError("invalid stage audio")
    channel=values[:,0];edges=np.linspace(0,len(channel),min(bins,len(channel))+1,dtype=int)
    return dict(min=[float(np.min(channel[a:b])) for a,b in zip(edges[:-1],edges[1:])],
                max=[float(np.max(channel[a:b])) for a,b in zip(edges[:-1],edges[1:])],channel=0),rate,values.shape
def stage_record(name,path,delay,web,sequence_id,side,domain):
    path=Path(path);info=sf.info(path)
    if not 0<=delay<info.frames:raise ValueError("invalid declared delay for stage")
    directory=web/"audio"/sequence_id/side;directory.mkdir(parents=True,exist_ok=True)
    link=directory/path.name
    if link.is_symlink() and link.resolve()!=path.resolve():raise ValueError("published stage link identity differs")
    if not link.exists():link.symlink_to(path.resolve())
    elif not link.is_symlink() and sha(link)!=sha(path):raise ValueError("published ordinary stage content differs")
    wave,rate,shape=envelope(path)
    return dict(name=side+"/"+name,file="/evaluation/ecnr-v2/"+str(link.relative_to(web)),
        sample_rate=rate,channels=shape[1],fixed_delay_samples=delay,trimmed_leading_samples=0,
        timeline_offset_ms=-delay/rate*1000,duration_s=shape[0]/rate,measurement_domain=domain,
        sha256=sha(path),waveform=wave)
def publish(report,web):
    web=Path(web).resolve();web.mkdir(parents=True,exist_ok=True);result=copy.deepcopy(report)
    result["identity_sequences"]=result.pop("sequences",[])
    result["sequences"]=[]
    for item in result["identity_sequences"]:
        stage_list=[]
        for side in ("baseline","candidate"):
            folder=Path(item[side+"_report"]).parent;engine=json.loads((folder/"stream.json").read_text())
            standard=[("reference","ref.wav",0,"declared digital playback reference"),
                ("microphones","before.wav",0,"raw microphone channels; displayed envelope channel0"),
                ("aec-input","stream.aec-input.wav",engine["aec_input_delay_samples"],"shared conditioned capture input"),
                ("linear-aec","stream.linear-aec.wav",engine["linear_aec_delay_samples"],"linear AEC mono 16k before nonlinear protection"),
                ("protected-aec","stream.aec.wav",engine["aec_output_delay_samples"],"protected AEC includes AEC3 residual suppression"),
                ("final-output","stream.after.wav",engine["dsp_delay_samples"],"final I/O-rate mono output")]
            for name,file,delay,domain in standard:
                if (folder/file).exists():stage_list.append(stage_record(name,folder/file,delay,web,item["id"],side,domain))
            tapfile=folder/"stream.stages.json"
            if tapfile.exists():
                for tap in json.loads(tapfile.read_text())["stages"]:
                    tap_path=Path(tap["file"])
                    if not tap_path.is_absolute():tap_path=Path(__file__).resolve().parents[2]/tap_path
                    rec=stage_record(tap["name"],tap_path,tap["fixed_delay_samples"],web,item["id"],side,tap["measurement_domain"])
                    if tap.get("samples_per_channel")!=sf.info(tap_path).frames:raise ValueError("stage frame count differs")
                    if tap.get("sha256") and rec["sha256"]!=tap["sha256"]:raise ValueError("stage hash differs")
                    for key in ("sample_rate","channels","trimmed_leading_samples","timeline_offset_ms"):
                        if rec[key]!=tap[key] and not isinstance(rec[key],float):raise ValueError("stage metadata disagrees")
                    if not math.isclose(rec["timeline_offset_ms"],tap["timeline_offset_ms"],abs_tol=1e-7):raise ValueError("stage timeline differs")
                    stage_list.append(rec)
        result["sequences"].append(dict(id=item["id"],stages=stage_list))
    result["failures"]=[dict(sequence_id=p["sequence_id"],label=p["label"],**g) for p in result["pairs"] for g in p["candidate_failures"]]
    groups=defaultdict(lambda:dict(baseline_pass=0,candidate_pass=0,phases=0))
    for pair in result["pairs"]:
        key=(pair["rate"],pair["microphones"],pair["variant"])
        groups[key]["baseline_pass"]+=pair["baseline_pass"];groups[key]["candidate_pass"]+=pair["candidate_pass"];groups[key]["phases"]+=1
    result["configurations"]=[dict(id=f"r{k[0]}-m{k[1]}-{k[2]}",rate=k[0],microphones=k[1],variant=k[2],**v) for k,v in groups.items()]
    save(web/"report.json",result);return result
def run(args):
    root=args.matrix.resolve();report=json.loads((root/"report.json").read_text())
    if not report["complete"] or not report["identity_verified"]:raise ValueError("complete identified matrix required for import")
    scorer=DNSMOS(args.assets,args.cache if args.cache else root/"dnsmos-cache")
    paired=[];diagnostic=[];tails=[];expected=0;model_id=None
    for item in report["sequences"]:
        verify_sequence(item)
        records={}
        for side in ("baseline","candidate"):
            folder=Path(item[side+"_report"]).parent
            score=scorer.score_windows(folder);audit=audit_stream(folder)
            output=root/"objective"/item["id"];save(output/(side+"-perceptual.json"),score);save(output/(side+"-audit.json"),audit)
            records[side]={(x["label"],x["kind"],tuple(x["window_s"])):x for x in score["windows"]}
            model_id=dict(preprocessing=score["preprocessing"],preprocessing_sha256=score["preprocessing_sha256"])
            if side=="candidate":
                for tail in audit.get("post_playback_tails",{}).get("events",[]):
                    tails.append(dict(sequence_id=item["id"],side=side,details=tail))
                for window in audit["windows"]:
                    diagnostic.append(dict(sequence_id=item["id"],label=window["label"],segment=window["kind"],
                        input_dbfs=window["before"]["rms_dbfs"],output_dbfs=window["after"]["rms_dbfs"],
                        total_attenuation_db=(float(20*np.log10(window["before"]["rms"]/window["after"]["rms"]))
                            if window["before"]["rms"]>0 and window["after"]["rms"]>0 else None),
                        ratio_scope="raw level ratio diagnostic; near-zero denominator remains unavailable to physical attenuation interpretation",
                        absolute_tail_floor_dbfs=window["after"]["rms_dbfs"],
                        reason=window["level_change"].get("reason"),window_s=window["window_s"],
                        details=window))
        if set(records["baseline"])!=set(records["candidate"]):raise ValueError("paired perceptual window inventory differs")
        expected+=len(records["baseline"])
        for key in records["baseline"]:
            b,c=records["baseline"][key]["after"],records["candidate"][key]["after"]
            paired.append(dict(sequence_id=item["id"],label=key[0],kind=key[1],window_s=key[2],baseline=b,candidate=c))
        print(json.dumps(dict(sequence=item["id"],paired_windows=len(paired))),flush=True)
    report["perceptual"]={**score_summary(paired,expected),**model_id}
    report["audits"]=dict(nearzero_windows=[x for x in diagnostic if x["details"]["before"]["near_measurement_floor"] or x["reason"]=="near_zero_denominator"],windows=diagnostic,post_playback_tails=tails)
    report["recommendation"]=dict(eligible=False,reason="20,298 scene paired regression and isolated paced repetitions remain required.")
    save(root/"report.json",report)
    if args.web:publish(report,args.web)
    print(json.dumps({k:v for k,v in report["perceptual"].items() if k not in ("windows","preprocessing")}),flush=True)
    return 0
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--matrix",type=Path,required=True)
    p.add_argument("--assets",type=Path,default=DEFAULT_ASSETS);p.add_argument("--web",type=Path);p.add_argument("--cache",type=Path)
    return run(p.parse_args())
if __name__=="__main__":raise SystemExit(main())
