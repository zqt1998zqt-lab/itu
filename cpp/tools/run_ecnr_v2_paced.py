#!/usr/bin/env python3
"""Serial real 10ms cadence repetitions; preserve all trials and historical 37 misses."""
from __future__ import annotations
import argparse,csv,json,math,os,shutil,subprocess,time
from pathlib import Path
from ecnr_campaign_calibration import save,sha
from ecnr_v2_evidence import verify_sequence
import check_audio_acceptance as ac
import run_audio_eval as ev
def host_snapshot():
    return dict(loadavg=Path("/proc/loadavg").read_text().strip(),uptime=Path("/proc/uptime").read_text().strip(),
      processes=subprocess.run(["ps","-eo","pid,comm,pcpu","--sort=-pcpu"],capture_output=True,text=True,check=True).stdout.splitlines()[:16])
def trial_metrics(folder,trial_id):
    engine=json.loads((folder/"stream.json").read_text())
    with (folder/"stream.frames.csv").open(newline="") as f:frames=list(csv.DictReader(f))
    count=engine["input_samples_per_channel"]//engine["frame_size"];rows=frames[:count]
    metrics,over=ev.frame_timing(frames,count)
    if len(rows)!=count or any(not math.isfinite(float(x["release_lateness_ms"])) or float(x["release_lateness_ms"])<0 or int(x["completion_late"]) not in (0,1) for x in rows):raise ValueError("paced timing unavailable")
    misses=[x for x in rows if int(x["completion_late"])]
    if not engine["paced_10ms"] or len(misses)!=engine["completion_deadline_misses"]:raise ValueError("deadline count differs")
    late_before=sum(float(x["release_lateness_ms"])>=10 for x in misses)
    return dict(id=trial_id,rate=engine["sample_rate"],microphones=engine["microphones"],**metrics,
        process_frames_over_10ms=over,completion_deadline_misses=len(misses),
        release_lateness_max_ms=max(float(x["release_lateness_ms"]) for x in rows),
        already_late_before_processing=late_before,remaining_misses=len(misses)-late_before,
        dsp_delay_ms=engine["dsp_delay_samples"]/engine["sample_rate"]*1000,
        frames=count,missed_frames=misses,engine_sha256=sha(folder/"stream.json"),csv_sha256=sha(folder/"stream.frames.csv"),
        output_sha256=sha(folder/"stream.after.wav"))
def runtime_gates(metric,rules):
    values=(("process_p95_ms","process_p95_max_ms"),("process_max_ms","process_max_ms"),
            ("process_frames_over_10ms","frames_over_10ms"),("process_rtf","process_rtf_max"))
    return [dict(metric=name,value=metric[name],operator="<=",threshold=rules["runtime"][key],
                 status="pass" if metric[name]<=rules["runtime"][key] else "fail") for name,key in values]
def verify_independent(path,matrix):
    report=json.loads(Path(path).read_text());identity=report["identity"]
    if not (report["complete"] and report["identity_verified"] and report["processed_scenes"]==20298
            and report["total_scenes"]==20298 and identity["expected_scenes"]==20298 and not identity["smoke_first"]):
        raise ValueError("full identified 20,298 scene report required before paced validation")
    expected=matrix["identity"]
    if (identity["baseline_cli_sha256"]!=expected["baseline_cli_sha256"] or identity["candidate_cli_sha256"]!=expected["cli_sha256"]
            or identity["rules_sha256"]!=expected["rules_sha256"] or identity["source_manifest_sha256"]!=expected["independent_manifest_sha256"]
            or identity["candidate_flags"]!=expected["v1_flags"]+expected["candidate_flags"]):
        raise ValueError("independent/continuous paced identity differs")
    return report
def run(args):
    matrix=args.matrix.resolve();report=json.loads((matrix/"report.json").read_text())
    if not report["complete"] or not report["identity_verified"]:raise ValueError("completed identified matrix required")
    verify_independent(args.independent_proof,report)
    if sha(ac.RULES)!=report["identity"]["rules_sha256"]:raise ValueError("original timing rules changed")
    rules=json.loads(ac.RULES.read_text())
    selected=next(x for x in report["sequences"] if x["id"]==args.sequence)
    verify_sequence(selected)
    root=args.root.resolve();root.mkdir(parents=True,exist_ok=False)
    original=args.historical.resolve();historical=trial_metrics(original,"historical-v1-37-misses")
    if historical["completion_deadline_misses"]!=37:raise ValueError("historical count changed")
    trials=[historical];identity=dict(matrix_report_sha256=sha(matrix/"report.json"),sequence=args.sequence,
        historical_folder=str(original),historical_engine_sha256=historical["engine_sha256"],
        independent_report_sha256=sha(args.independent_proof),rules_sha256=sha(ac.RULES),runner_sha256=sha(__file__),
        baseline_cli_sha256=sha(args.baseline_cli),candidate_cli_sha256=sha(matrix/"itu_audio_cli"),
        repetitions=args.repetitions,scope="serial host 10ms cadence with original 235s history; shared host state recorded; not board realtime proof")
    save(root/"identity.json",identity)
    artifact=root/"artifacts";artifact.mkdir()
    shutil.copy2(__file__,artifact/Path(__file__).name);shutil.copy2(ac.RULES,artifact/ac.RULES.name)
    for repetition in range(args.repetitions):
        for side in ("baseline","candidate"):
            if shutil.disk_usage(root).free<3*1024**3:
                save(root/"report.json",dict(complete=False,state="blocked",identity=identity,trials=trials,reason="disk reserve reached"));return 2
            binary=args.baseline_cli.resolve() if side=="baseline" else matrix/"itu_audio_cli"
            expected=identity[side+"_cli_sha256"]
            if sha(binary)!=expected:raise ValueError("frozen paced binary changed")
            if sha(ac.RULES)!=identity["rules_sha256"] or sha(__file__)!=identity["runner_sha256"]:
                raise ValueError("paced runner/original rules changed")
            verify_sequence(selected)
            source=Path(selected[side+"_report"]).parent
            prior=json.loads((source/"invocation.json").read_text())
            if prior["exit_code"]!=0:raise ValueError("source CLI execution failed")
            folder=root/(side+f"-repeat-{repetition+1:02}");folder.mkdir()
            argv=list(prior["argv"]);argv[0]=str(binary)
            argv[argv.index("--out")+1]=str(folder/"stream")
            if "--diagnostic-stages" in argv:argv.remove("--diagnostic-stages")
            if "--pace-10ms" not in argv:argv+=["--pace-10ms"]
            save(folder/"host-before.json",host_snapshot());started=time.time()
            save(folder/"invocation.json",dict(argv=argv,exit_code=None,state="running",cli_sha256=expected))
            try:completed=subprocess.run(argv,text=True,capture_output=True,timeout=360,
                    env=dict(os.environ,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1"))
            except subprocess.TimeoutExpired as error:
                save(folder/"invocation.json",dict(argv=argv,exit_code=None,state="timeout",cli_sha256=expected));raise
            save(folder/"invocation.json",dict(argv=argv,exit_code=completed.returncode,stdout=completed.stdout,stderr=completed.stderr,
                cli_sha256=expected,elapsed_s=time.time()-started))
            completed.check_returncode();save(folder/"host-after.json",host_snapshot())
            metric=trial_metrics(folder,folder.name);metric.update(side=side,repetition=repetition+1)
            expected_output=selected[side+"_output_sha256"]
            metric["offline_audio_parity"]=metric["output_sha256"]==expected_output
            metric["runtime_gates"]=runtime_gates(metric,rules)
            metric["processing_pass"]=all(g["status"]=="pass" for g in metric["runtime_gates"])
            trials.append(metric)
            save(root/"report.json",dict(complete=False,state="running",identity=identity,trials=trials))
            print(json.dumps({k:v for k,v in metric.items() if k!="missed_frames"}),flush=True)
    if len(trials)!=1+2*args.repetitions:raise ValueError("paced trial coverage differs")
    result=dict(complete=True,state="completed",identity=identity,trials=trials,
        candidate_processing_pass=all(t["processing_pass"] for t in trials[1:] if t["side"]=="candidate"),
        candidate_deadline_pass=all(t["completion_deadline_misses"]==0 for t in trials[1:] if t["side"]=="candidate"),
        audio_parity_pass=all(t["offline_audio_parity"] for t in trials[1:]),
        historical_misses_preserved=37)
    save(root/"report.json",result);return 0
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("matrix","root","baseline-cli","historical","independent-proof"):p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--sequence",default="r48000-m4-control-pair-00");p.add_argument("--repetitions",type=int,default=2)
    args=p.parse_args()
    if args.repetitions<2:raise ValueError("at least two repetitions must all be reported")
    try:return run(args)
    except Exception as error:
        path=args.root/"report.json";report=json.loads(path.read_text()) if path.exists() else {}
        report.update(complete=False,state="blocked",reason=str(error));save(path,report);raise
if __name__=="__main__":raise SystemExit(main())
