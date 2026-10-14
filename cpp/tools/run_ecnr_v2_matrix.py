#!/usr/bin/env python3
"""Resumable v2 continuous pairing against the immutable 184/270 ECNR baseline."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
import check_audio_acceptance as ac
from ecnr_campaign_calibration import save, sha
from ecnr_gate_contract import paired_contract

PILOT=((16000,1,"control",0),(16000,1,"control",2),(16000,2,"control",0),
       (48000,4,"control",0),(48000,4,"control",2),(24000,4,"spatial",2))
PAIR_KEYS=("files","source_manifest_sha256","sample_rate","microphones","engine_resets",
           "target_alignment","simulated_clock","processing_profile","beam_delays_samples")
def quality(gates):
    return all(g["status"]=="pass" or g["optional"] and g["status"]=="unavailable"
               for g in gates if g["segment"]!="runtime")
def compare(before,after):
    for key in PAIR_KEYS:
        if before["identity"].get(key)!=after["identity"].get(key):raise ValueError("pair input identity mismatch: "+key)
    if [s["label"] for s in before["details"]]!=[s["label"] for s in after["details"]]:raise ValueError("phase order differs")
    paired=[]
    for b,c in zip(before["details"],after["details"]):
        bg,cg,branch_changes=paired_contract(b["gates"],c["gates"])
        failed=lambda g:g["status"]=="fail" or g["status"]=="unavailable" and not g["optional"]
        paired.append(dict(label=b["label"],baseline_pass=quality(b["gates"]),candidate_pass=quality(c["gates"]),
            new_failures=[cg[k] for k in cg if failed(cg[k]) and not failed(bg[k])],
            repaired_gates=[cg[k] for k in cg if cg[k]["status"]=="pass" and failed(bg[k])],
            candidate_failures=[g for g in c["gates"] if failed(g) and g["segment"]!="runtime"],attenuation_branch_changes=branch_changes))
    return paired
def calibration_identity(path,dataset):
    report=json.loads(Path(path).read_text())
    if not report.get("complete") or report["status"]!="passed":raise ValueError("digital calibration normal/negative controls required")
    if report["identity"]["dataset_manifest_sha256"]!=sha(dataset/"inputs-manifest.json"):raise ValueError("calibration dataset mismatch")
    if report["identity"]["tool_sha256"]!=sha(Path(__file__).with_name("ecnr_campaign_calibration.py")):raise ValueError("calibration campaign changed")
    if report["identity"]["preflight_sha256"]!=sha(Path(__file__).with_name("ecnr_preflight.py")):raise ValueError("calibration implementation changed")
    for case in report["cases"]:
        if not case["expectation_met"]:raise ValueError("failed calibration expectation")
        if case.get("fixture") and sha(case["fixture"])!=case["fixture_sha256"]:raise ValueError("calibration fixture changed")
    return dict(report_sha256=sha(path),status="passed",summary=report["summary"],report=str(path))

def run(args):
    dataset=args.dataset.resolve();baseline=args.baseline.resolve();root=args.root.resolve()
    old=json.loads((baseline/"report.json").read_text())
    if not old["complete"] or old["candidate_pass"]!=184 or old["total_phases"]!=270:raise ValueError("require frozen 184/270 baseline")
    if sha(dataset/"inputs-manifest.json")!=old["identity"]["manifest_sha256"]:raise ValueError("frozen input manifest differs")
    if sha(baseline/"itu_audio_cli")!=old["identity"]["cli_sha256"]:raise ValueError("frozen baseline binary changed")
    if sha(ac.RULES)!=old["identity"]["rules_sha256"]:raise ValueError("quality rules differ")
    calibration=calibration_identity(args.calibration,dataset)
    frozen_hashes={str(Path(x["candidate_report"]).relative_to(baseline)):x["candidate_report_sha256"] for x in old["pairs"]}
    for relative,expected_sha in frozen_hashes.items():
        frozen=baseline/relative
        if sha(frozen)!=expected_sha:raise ValueError("frozen baseline report changed: "+relative)
        prior=json.loads(frozen.read_text())
        for file,hash_value in {**prior["identity"]["files"],**prior["identity"]["output_files"]}.items():
            if sha(frozen.parent/file)!=hash_value:raise ValueError("frozen baseline artifact changed: "+file)
    array_manifest=json.loads((dataset/"inputs-manifest.json").read_text())
    independent_manifest_sha256=array_manifest["identity"]["base_manifest_sha256"]
    configurations=list(PILOT) if args.pilot else [(g["rate"],g["microphones"],g["variant"],p) for g in old["groups"] for p in (0,2)]
    flags=args.cli_option
    if args.no_coord: v1_flags=["--noise-transition-control"]
    else:v1_flags=["--noise-transition-control","--ecnr-coordinated-control"]
    expected=dict(baseline_report_sha256=sha(baseline/"report.json"),baseline_cli_sha256=old["identity"]["cli_sha256"],
        cli_sha256=sha(args.cli),manifest_sha256=sha(dataset/"inputs-manifest.json"),rules_sha256=sha(ac.RULES),
        runner_sha256={name:sha(Path(__file__).with_name(name)) for name in ("run_ecnr_v2_matrix.py","run_vehicle_stream.py","evaluation_targets.py","ecnr_gate_contract.py")},
        independent_manifest_sha256=independent_manifest_sha256,calibration=calibration,candidate_flags=flags,v1_flags=v1_flags,configurations=[list(x) for x in configurations],
        scope="HSAE host simulation; quality excludes runtime; explicit optional candidate; no Android/vehicle certification")
    if root.exists():
        if not args.resume:raise ValueError("new batch root required; use --resume only with unchanged identity")
        if json.loads((root/"identity.json").read_text())!=expected:raise ValueError("resume identity changed")
    else:
        root.mkdir(parents=True);shutil.copy2(args.cli,root/"itu_audio_cli");save(root/"identity.json",expected)
        artifact=root/"artifacts";artifact.mkdir()
        for name,h in expected["runner_sha256"].items():
            shutil.copy2(Path(__file__).with_name(name),artifact/name)
            if sha(artifact/name)!=h:raise ValueError("frozen runner copy changed")
    binary=root/"itu_audio_cli"
    if sha(binary)!=expected["cli_sha256"]:raise ValueError("frozen candidate changed")
    pairs=[];sequences=[];started=time.time()
    def report(state,reason=None):
        result=dict(schema_version=1,state=state,complete=state=="completed",identity_verified=True,identity=expected,
            baseline=dict(id="ecnr-v1-184-of-270",quality_passed=184,total_phases=270),
            calibration=calibration,coverage=dict(completed_sequences=len(sequences),expected_sequences=len(configurations)),
            quality=dict(baseline_pass=sum(p["baseline_pass"] for p in pairs),candidate_pass=sum(p["candidate_pass"] for p in pairs),
                total_phases=len(pairs),phase_wins=sum(not p["baseline_pass"] and p["candidate_pass"] for p in pairs),
                phase_regressions=sum(p["baseline_pass"] and not p["candidate_pass"] for p in pairs),
                repaired_gates=sum(len(p["repaired_gates"]) for p in pairs),new_failed_gates=sum(len(p["new_failures"]) for p in pairs)),
            pairs=pairs,sequences=sequences,elapsed_s=time.time()-started,
            recommendation=dict(eligible=False,reason=reason or "Complete perceptual, 20,298 scene pairing and isolated paced repetitions remain required."))
        save(root/"report.json",result);return result
    report("running")
    for rate,mics,variant,pair in configurations:
        if sha(ac.RULES)!=expected["rules_sha256"] or any(sha(Path(__file__).with_name(n))!=h for n,h in expected["runner_sha256"].items()):
            raise ValueError("runner or original rules changed during batch")
        if shutil.disk_usage(root).free<3*1024**3:
            report("blocked","Disk reserve reached; batch paused without deleting history or reducing coverage.");return 2
        name=f"r{rate}-m{mics}-{variant}";folder=root/name/f"pair-{pair:02}";folder.parent.mkdir(exist_ok=True)
        options=v1_flags+flags+(["--spatial-mvdr","--wind-control"] if variant=="spatial" else [])
        if args.stages:options+=["--diagnostic-stages"]
        argv=[sys.executable,str(Path(__file__).with_name("run_vehicle_stream.py")),"--dataset",str(dataset),
            "--root",str(folder),"--cli",str(binary),"--pair",str(pair),"--microphones",str(mics),
            "--sample-rate",str(rate),"--simulate-clock"]+["--cli-option="+x for x in options]
        invocation=root/(name+f"-pair-{pair:02}.invocation.json")
        if folder.exists():
            if not args.resume or not invocation.exists():raise ValueError("unidentified partial sequence; preserve and use a new root")
            previous=json.loads(invocation.read_text())
            if previous["argv"]!=argv or previous["exit_code"]!=0:raise ValueError("resume invocation differs or failed")
        else:
            save(invocation,dict(argv=argv,exit_code=None,cli_sha256=sha(binary),state="running"))
            try:
                with (root/(name+f"-pair-{pair:02}.log")).open("w") as log:
                    completed=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=300,
                        env=dict(os.environ,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1"))
            except subprocess.TimeoutExpired:
                save(invocation,dict(argv=argv,exit_code=None,cli_sha256=sha(binary),state="timeout"));raise
            save(invocation,dict(argv=argv,exit_code=completed.returncode,cli_sha256=sha(binary),state="completed"))
            if completed.returncode:
                report("blocked","Sequence execution failed; see retained invocation/log.");return completed.returncode
        before_path=baseline/name/f"pair-{pair:02}"/"stream-evaluation.json";after_path=folder/"stream-evaluation.json"
        before=json.loads(before_path.read_text());after=json.loads(after_path.read_text())
        matched=compare(before,after)
        sid=name+f"-pair-{pair:02}"
        for item in matched:item.update(sequence_id=sid,rate=rate,microphones=mics,variant=variant,pair=pair)
        pairs+=matched
        sequences.append(dict(id=sid,baseline_report=str(before_path),candidate_report=str(after_path),
            baseline_report_sha256=sha(before_path),candidate_report_sha256=sha(after_path),
            baseline_output_sha256=before["identity"]["output_files"]["stream.after.wav"],candidate_output_sha256=after["identity"]["output_files"]["stream.after.wav"]))
        current=report("running");print(json.dumps(dict(sequence=sid,quality=current["quality"])),flush=True)
    if sha(binary)!=expected["cli_sha256"]:raise ValueError("candidate mutated while processing")
    result=report("running")
    if not args.pilot and (result["quality"]["total_phases"]!=270 or result["quality"]["baseline_pass"]!=184):raise ValueError("incomplete or altered frozen baseline")
    if sha(ac.RULES)!=expected["rules_sha256"] or any(sha(Path(__file__).with_name(n))!=h for n,h in expected["runner_sha256"].items()):raise ValueError("runner or original rules changed")
    result=report("completed")
    print(json.dumps(result["quality"]),flush=True);return 0
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("dataset","baseline","root","cli","calibration"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--pilot",action="store_true");p.add_argument("--resume",action="store_true");p.add_argument("--stages",action="store_true")
    p.add_argument("--no-coord",action="store_true");p.add_argument("--cli-option",action="append",default=[])
    args=p.parse_args()
    try:return run(args)
    except Exception as error:
        report_path=args.root/"report.json"
        if report_path.exists():
            report=json.loads(report_path.read_text());report.update(state="blocked",complete=False,identity_verified=False)
            report["recommendation"]=dict(eligible=False,reason=str(error));save(report_path,report)
        save(args.root/"failure.json",dict(error_type=type(error).__name__,reason=str(error)))
        raise
if __name__=="__main__":raise SystemExit(main())
