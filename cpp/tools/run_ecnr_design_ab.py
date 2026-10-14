#!/usr/bin/env python3
"""Pair an identified ECNR candidate with the frozen current continuous matrix."""
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def quality(gates):
    return all(g["status"]=="pass" or g["optional"] and g["status"]=="unavailable"
               for g in gates if g["segment"]!="runtime")
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--cli",type=Path,required=True)
    a=p.parse_args()
    dataset,base,root=a.dataset.resolve(),a.baseline.resolve(),a.root.resolve()
    old=json.loads((base/"report.json").read_text())
    if not old["complete"] or old["identity"]["standard_compliance"]: raise ValueError("completed engineering baseline required")
    if digest(dataset/"inputs-manifest.json")!=old["identity"]["manifest_sha256"]: raise ValueError("same frozen input required")
    tools=Path(__file__).resolve().parent
    import check_audio_acceptance as ac
    if digest(ac.RULES)!=old["identity"]["rules_sha256"]: raise ValueError("frozen threshold mismatch")
    if digest(base/"itu_audio_cli")!=old["identity"]["cli_sha256"]: raise ValueError("frozen baseline CLI changed")
    root.mkdir(parents=True,exist_ok=False)
    binary=root/"itu_audio_cli";shutil.copy2(a.cli,binary)
    identity=dict(baseline_report_sha256=digest(base/"report.json"),baseline_cli_sha256=old["identity"]["cli_sha256"],
        cli_sha256=digest(binary),manifest_sha256=old["identity"]["manifest_sha256"],rules_sha256=old["identity"]["rules_sha256"],
        runner_sha256={n:digest(tools/n) for n in ("run_ecnr_design_ab.py","run_vehicle_regression.py","run_vehicle_stream.py","evaluation_targets.py")},
        standard_compliance=False,scope="host simulation; calibrated digital input; optional candidate; no Android build",
        seconds_per_sequence=235,engine_resets_per_sequence=0,ecnr_gain_limit_db=3,
        baseline_measurements="frozen prior 30-stream matrix; not rerun here")
    (root/"identity.json").write_text(json.dumps(identity,indent=2)+"\n")
    groups=[];paired=[]
    for row in old["rows"]:
        rate,mics,variant=row["rate"],row["microphones"],row["variant"]
        name=f"r{rate}-m{mics}-{variant}";folder=root/name
        flags=["--noise-transition-control","--ecnr-coordinated-control"]
        if variant=="spatial":flags+=["--spatial-mvdr","--wind-control"]
        argv=[sys.executable,str(tools/"run_vehicle_regression.py"),"--dataset",str(dataset),"--root",str(folder),
            "--cli",str(binary),"--jobs","1","--sample-rate",str(rate),"--microphones",str(mics),"--simulate-clock"]
        argv += ["--cli-option="+flag for flag in flags]
        with (root/(name+".log")).open("w") as log:
            run=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=900,
                env=dict(os.environ,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1"))
        (root/(name+".invocation.json")).write_text(json.dumps(dict(argv=argv,exit_code=run.returncode),indent=2))
        run.check_returncode()
        report=json.loads((folder/"regression-summary.json").read_text())
        if report["phases"]!=row["phases"] or report["identity"]["pairs"]!=old["identity"]["pairs"]:raise ValueError("incomplete pair coverage")
        if report["identity"]["rules_sha256"]!=identity["rules_sha256"]:raise ValueError("threshold drift")
        for path in sorted(folder.glob("pair-*/stream-evaluation.json")):
            before_path=base/name/path.parent.name/path.name
            before=json.loads(before_path.read_text());after=json.loads(path.read_text())
            for k in ("files","source_manifest_sha256","sample_rate","microphones","engine_resets","target_alignment","simulated_clock","processing_profile","beam_delays_samples"):
                if before["identity"].get(k)!=after["identity"].get(k):raise ValueError("pair identity mismatch: "+k)
            if [s["label"] for s in before["details"]]!=[s["label"] for s in after["details"]]:raise ValueError("phase labels changed")
            for b,c in zip(before["details"],after["details"]):
                bg={(g["segment"],g["metric"]):g for g in b["gates"] if g["segment"]!="runtime"}
                cg={(g["segment"],g["metric"]):g for g in c["gates"] if g["segment"]!="runtime"}
                if set(bg)!=set(cg):raise ValueError("gate inventory changed")
                for k in bg:
                    if any(bg[k][f]!=cg[k][f] for f in ("operator","threshold","optional")):raise ValueError("gate contract changed")
                paired.append(dict(rate=rate,microphones=mics,variant=variant,pair=path.parent.name,label=b["label"],
                    baseline_pass=quality(b["gates"]),candidate_pass=quality(c["gates"]),
                    new_failures=[cg[k] for k in cg if cg[k]["status"]=="fail" and bg[k]["status"]!="fail"],
                    repaired_gates=[cg[k] for k in cg if cg[k]["status"]=="pass" and bg[k]["status"]=="fail"],
                    candidate_failures=[g for g in c["gates"] if g["status"]=="fail"],
                    baseline_report=str(before_path),candidate_report=str(path),
                    baseline_report_sha256=digest(before_path),candidate_report_sha256=digest(path)))
        groups.append(dict(rate=rate,microphones=mics,variant=variant,phases=report["phases"],
            baseline_pass=row["quality_passed_phases"],candidate_pass=report["quality_passed_phases"],
            failed_gate_counts=report["failed_gate_counts"]))
        print(json.dumps(groups[-1]),flush=True)
    if digest(binary)!=identity["cli_sha256"]:raise ValueError("CLI mutated during evaluation")
    report=dict(identity=identity,complete=True,groups=groups,pairs=paired,total_phases=len(paired),
        baseline_pass=sum(x["baseline_pass"] for x in paired),candidate_pass=sum(x["candidate_pass"] for x in paired),
        phase_wins=sum(not x["baseline_pass"] and x["candidate_pass"] for x in paired),
        phase_regressions=sum(x["baseline_pass"] and not x["candidate_pass"] for x in paired),
        repaired_gates=sum(len(x["repaired_gates"]) for x in paired),new_failed_gates=sum(len(x["new_failures"]) for x in paired))
    (root/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("groups","pairs","identity")}),flush=True)
if __name__=="__main__":main()
