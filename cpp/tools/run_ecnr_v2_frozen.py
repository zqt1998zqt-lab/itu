#!/usr/bin/env python3
"""Frozen 20,298-scene baseline/candidate replay with retained failures and resumable identity."""
from __future__ import annotations
import argparse, ast, concurrent.futures, gzip, hashlib, json, math, os, shutil, subprocess, sys, time
from collections import Counter
from pathlib import Path
import run_audio_eval as ev
import check_audio_acceptance as ac
from ecnr_campaign_calibration import save, sha
from run_ecnr_v2_matrix import quality
import ecnr_perceptual as perceptual
import ecnr_gate_contract as contract
_SCORER=None

EXPECTED_SCENES=20298
EXPECTED_MANIFEST_SHA256="7bea97f2a997c3a09a50eb2b944a6e22566828260072b8948dfc7d63ac4b9be3"

EXPECTED_ARRAY_SHA256="c10bec8ee8806a0e78b21cc5b8f21b074ee5bff6edaa0a10cfe58bca43e5981a"
V1_FLAGS=["--noise-transition-control","--ecnr-coordinated-control"]
CONFIGURATIONS=[(r,m,v,p) for r in (16000,24000,48000) for m in (1,2,4)
                for v in (("control",) if m==1 else ("control","spatial")) for p in (0,2)]
PHASES=("cold_start_speech","engine_change","window_wind","radio_and_road","long_radio_path",
        "reference_clock_drift","reference_mismatch","path_change","radio_restart")
_PROOF_FILES={}
def json_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
def checked_file(path,expected):
    path=Path(path)
    if not path.is_file() or sha(path)!=expected:raise ValueError("evidence file SHA mismatch: "+str(path))
    return path
def proof_file(key,path,expected=None):
    path=Path(path).resolve()
    if expected is not None:checked_file(path,expected)
    _PROOF_FILES[key]=path
    return path
def load_json(path):return json.loads(Path(path).read_text())
def failure(g):
    return g["status"]=="fail" or g["status"]=="unavailable" and not g["optional"]
def gate_summary(gates):
    gb,gc,changes=contract.paired_contract(gates["baseline"],gates["candidate"])
    return dict(baseline_pass=quality(gates["baseline"]),candidate_pass=quality(gates["candidate"]),
        new_failures=[gc[k] for k in gc if failure(gc[k]) and not failure(gb[k])],
        repaired_gates=[gc[k] for k in gc if gc[k]["status"]=="pass" and failure(gb[k])],
        attenuation_branch_changes=changes)
def summarize_pairs(pairs,continuous=False):
    result=dict(baseline_pass=sum(x["baseline_pass"] for x in pairs),candidate_pass=sum(x["candidate_pass"] for x in pairs),
        phase_wins=sum(not x["baseline_pass"] and x["candidate_pass"] for x in pairs),
        phase_regressions=sum(x["baseline_pass"] and not x["candidate_pass"] for x in pairs),
        new_failed_gates=sum(len(x["new_failures"]) for x in pairs),repaired_gates=sum(len(x["repaired_gates"]) for x in pairs))
    if continuous:result["total_phases"]=len(pairs)
    return result
def verify_calibration(identity):
    meta=identity["calibration"];path=Path(meta["report"])
    if not path.is_absolute():path=Path(__file__).resolve().parents[2]/path
    report=load_json(proof_file("calibration/report.json",path,meta["report_sha256"]))
    if not report["complete"] or report["status"]!="passed" or report["identity"]["dataset_manifest_sha256"]!=identity["manifest_sha256"]:
        raise ValueError("calibration is incomplete or belongs to another array manifest")
    ids={f"r{r}-reference-{kind}" for r in (16000,24000,48000) for kind in ("normal","delay_jump","polarity","gain_mismatch")}
    ids|={f"r{r}-m1-p{p:02}-geometry" for r in (16000,24000,48000) for p in (0,2)}
    ids|={f"r{r}-m{m}-p{p:02}-{kind}" for r in (16000,24000,48000) for m in (2,4) for p in (0,2) for kind in ("normal","polarity","gain_mismatch")}
    cases=report["cases"]
    if len(cases)!=54 or {c["id"] for c in cases}!=ids:raise ValueError("54 unique calibration normal/negative controls required")
    for case in cases:
        expected=case["id"].endswith(("-normal","-geometry"))
        if case["expected_accepted"]!=expected or case["actual_accepted"]!=expected or not case["expectation_met"]:
            raise ValueError("calibration control expectation changed")
        if case.get("fixture"):proof_file("calibration/"+Path(case["fixture"]).name,case["fixture"],case["fixture_sha256"])
    for key,name in (("tool_sha256","ecnr_campaign_calibration.py"),("preflight_sha256","ecnr_preflight.py")):
        proof_file("calibration/code/"+name,Path(__file__).with_name(name),report["identity"][key])
def score_key(w):return (w["sequence_id"],w["label"],w["kind"],tuple(w["window_s"]))
def verify_score(score,audio,rate,kind,prep_sha):
    import numpy as np
    if (not score.get("available") or score.get("role")!=kind or score.get("model_sha256")!=perceptual.MODEL_SHA
        or score.get("preprocessing_sha256")!=prep_sha or score.get("official_commit")!=perceptual.PINNED_COMMIT
        or score.get("personalized") is not False or score.get("source_sample_rate")!=rate or score.get("source_samples")!=len(audio)):
        raise ValueError("perceptual score identity or availability differs")
    audio=np.ascontiguousarray(audio,dtype="<f8")
    digest=hashlib.sha256(str(audio.shape).encode()+audio.tobytes()).hexdigest()
    if score.get("audio_sha256")!=digest:raise ValueError("DNSMOS window audio SHA differs after fixed-delay alignment")
    for metric in ("SIG","BAK","OVRL"):
        if not math.isfinite(score["poly"][metric]) or not math.isfinite(score["raw"][metric]):raise ValueError("nonfinite perceptual score")
def require_proof(path):
    try:return _require_proof(path)
    except (KeyError,TypeError,AttributeError) as error:
        raise ValueError("incomplete quality proof: missing or invalid required evidence fields") from error
def _require_proof(path):
    import numpy as np
    import soundfile as sf
    _PROOF_FILES.clear()
    path=Path(path).resolve();report=load_json(path);q=report["quality"];p=report.get("perceptual",{});identity=report["identity"]
    if not (report["complete"] and report["identity_verified"] and q["total_phases"]==270 and q["baseline_pass"]==184
            and q["candidate_pass"]>184 and q["new_failed_gates"]==0 and q["phase_regressions"]==0
            and p.get("status")=="completed" and p.get("expected_windows")==510 and p.get("paired_windows")==510):
        raise ValueError("qualified 30-stream/270-phase/510-window quality and perceptual proof required")
    expected_ids={f"r{r}-m{m}-{v}-pair-{pair:02}":(r,m,v,pair) for r,m,v,pair in CONFIGURATIONS}
    sequences=report["sequences"]
    if len(sequences)!=30 or {s["id"] for s in sequences}!=set(expected_ids) or identity["configurations"]!=[list(x) for x in CONFIGURATIONS]:
        raise ValueError("30 unique complete rate/microphone/variant/pair identities required")
    if report["coverage"]!={"completed_sequences":30,"expected_sequences":30}:raise ValueError("continuous coverage differs")
    if identity["manifest_sha256"]!=EXPECTED_ARRAY_SHA256 or identity["independent_manifest_sha256"]!=EXPECTED_MANIFEST_SHA256:
        raise ValueError("continuous proof source manifest differs")
    if identity["rules_sha256"]!=sha(ac.RULES) or identity["v1_flags"]!=V1_FLAGS:raise ValueError("continuous proof rules or v1 flags differ")
    verify_calibration(identity)
    baseline_root=Path(sequences[0]["baseline_report"]).parents[2]
    proof_file("baseline-report.json",baseline_root/"report.json",identity["baseline_report_sha256"])
    proof_file("quality-proof.json",path)
    for name,expected in identity["runner_sha256"].items():
        proof_file("code/"+name,path.parent/"artifacts"/name,expected)
    prep=p["preprocessing"];prep_sha=json_digest(prep)
    if p["preprocessing_sha256"]!=prep_sha or prep["implementation_sha256"]!=sha(perceptual.__file__) or prep["model_sha256"]!=perceptual.MODEL_SHA:
        raise ValueError("perceptual preprocessing implementation identity differs")
    model_dir=perceptual.DEFAULT_ASSETS
    for name,expected in (("sig_bak_ovr.onnx",perceptual.MODEL_SHA),("dnsmos_local.py",perceptual.SCRIPT_SHA)):
        proof_file("perceptual-assets/"+name,model_dir/name,expected)
    proof_file("perceptual-assets/manifest.json",model_dir/"manifest.json")
    windows=p["windows"]
    if len(windows)!=510 or len({score_key(w) for w in windows})!=510:raise ValueError("510 unique speech windows required")
    by_window={score_key(w):w for w in windows};expected_windows=set();pairs=[]
    rules=load_json(ac.RULES)
    for item in sequences:
        sid=item["id"];rate,mics,variant,pair=expected_ids[sid];streams={}
        outer=load_json(proof_file("sequences/"+sid+"/runner-invocation.json",path.parent/(sid+".invocation.json")))
        if outer["exit_code"]!=0 or outer["cli_sha256"]!=identity["cli_sha256"]:raise ValueError("continuous runner invocation failed or differs")
        dataset=Path(outer["argv"][outer["argv"].index("--dataset")+1])
        proof_file("array-inputs-manifest.json",dataset/"inputs-manifest.json",identity["manifest_sha256"])
        for side in ("baseline","candidate"):
            folder=Path(item[side+"_report"]).parent
            stream=load_json(proof_file("sequences/"+sid+"/"+side+"/stream-evaluation.json",item[side+"_report"],item[side+"_report_sha256"]))
            si=stream["identity"];engine=stream["engine"]
            if (si["processing_profile"]!="vehicle_stream" or si["sample_rate"]!=rate or si["microphones"]!=mics
                or si["source_manifest_sha256"]!=identity["manifest_sha256"] or si["rules_sha256"]!=identity["rules_sha256"]
                or si["engine_resets"]!=0 or engine["stream_resets"]!=0 or si["cli_sha256"]!=identity["baseline_cli_sha256" if side=="baseline" else "cli_sha256"]):
                raise ValueError("continuous stream profile/data/binary identity differs")
            ev.verify_processing_profile(engine,"vehicle_stream")
            if engine!=load_json(folder/"stream.json"):raise ValueError("continuous engine differs from actual CLI metadata")
            for name,expected in {**si["files"],**si["output_files"]}.items():checked_file(folder/name,expected)
            if si["output_files"]["stream.after.wav"]!=item[side+"_output_sha256"]:raise ValueError("continuous output SHA differs")
            inv=load_json(proof_file("sequences/"+sid+"/"+side+"/invocation.json",folder/"invocation.json"))
            flags=identity["v1_flags"]+(identity["candidate_flags"] if side=="candidate" else [])+(["--spatial-mvdr","--wind-control"] if variant=="spatial" else [])
            if engine.get("diagnostic_stages"):flags+=["--diagnostic-stages"]
            argv=[inv["argv"][0],"--profile","vehicle_stream","--mic",str(folder/"before.wav"),"--echo",str(folder/"ref.wav"),"--out",str(folder/"stream"),"--microphones",str(mics)]
            if mics>1:argv+=["--beam-delays",",".join(f"{x*rate/16000:.9f}" for x in si["beam_delays_samples"])]
            argv+=["--clock-events",str(folder/"simulated-clock-events.csv")]+flags
            if inv["argv"]!=argv or inv["exit_code"]!=0 or inv["cli_sha256"]!=si["cli_sha256"]:raise ValueError("actual continuous CLI flags/inputs differ")
            checked_file(argv[0],si["cli_sha256"])
            if si["diagnostic_overrides"]!=flags:raise ValueError("continuous effective flags differ")
            proof_file("sequences/"+sid+"/"+side+"/timeline.json",folder/"timeline.json")
            proof_file("sequences/"+sid+"/"+side+"/simulated-clock-events.csv",folder/"simulated-clock-events.csv",si["simulated_clock"]["events_sha256"])
            if [s["label"] for s in si["segments"]]!=list(PHASES) or [s["label"] for s in stream["details"]]!=list(PHASES) or [s["label"] for s in stream["measurements"]]!=list(PHASES):
                raise ValueError("nine complete continuous phase identities required")
            for segment,measurement,detail in zip(si["segments"],stream["measurements"],stream["details"]):
                condition=dict(segment["scene"],windows=segment["windows"])
                verdict=ac.check_scene(condition,measurement["row"],rules);verdict["label"]=segment["label"]
                if verdict!=detail:raise ValueError("continuous original gates do not match retained measurements")
            streams[side]=stream
            objective=load_json(proof_file("objective/"+sid+"/"+side+"-perceptual.json",path.parent/"objective"/sid/(side+"-perceptual.json")))
            osi=objective["source_identity"];alignment=osi["alignment"]
            if objective["preprocessing_sha256"]!=prep_sha or objective["preprocessing"]!=prep or osi["cli_sha256"]!=si["cli_sha256"]:
                raise ValueError("per-sequence perceptual identity differs")
            for name,info in osi["files"].items():
                checked_file(folder/name,info["sha256"])
                if (folder/name).stat().st_size!=info["bytes"]:raise ValueError("perceptual source size differs")
            info=sf.info(folder/"stream.after.wav");before_info=sf.info(folder/"before.wav");delay=engine["dsp_delay_samples"]
            if (alignment!={"policy":"declared DSP delay; no lag optimization","after_trim_samples":delay,"aligned_samples":before_info.frames,"sample_rate":rate}
                or info.samplerate!=rate or before_info.samplerate!=rate or info.frames!=before_info.frames+delay or info.channels!=1):
                raise ValueError("perceptual fixed DSP delay/sample-rate/length alignment differs")
            records={(w["label"],w["kind"],tuple(w["window_s"])):w for w in objective["windows"]}
            expected={(s["label"],kind,tuple(limits)) for s in si["segments"] for kind,limits in s["windows"].items() if kind in perceptual.ALLOWED_ROLES}
            if len(records)!=len(objective["windows"]) or set(records)!=expected:raise ValueError("per-sequence speech inventory differs")
            with sf.SoundFile(folder/"stream.after.wav") as audio:
                for label,kind,limits in expected:
                    key=(sid,label,kind,limits);expected_windows.add(key)
                    if key not in by_window:raise ValueError("missing continuous speech score")
                    score=by_window[key][side]
                    if score!=records[(label,kind,limits)]["after"]:raise ValueError("perceptual summary differs from per-stream evidence")
                    a,b=[round(x*rate) for x in limits];audio.seek(delay+a)
                    values=audio.read(b-a,dtype="float64",always_2d=True)[:,0]
                    verify_score(score,values,rate,kind,prep_sha)
        before,after=streams["baseline"],streams["candidate"]
        for key in ("files","source_manifest_sha256","sample_rate","microphones","engine_resets","target_alignment","simulated_clock","processing_profile","beam_delays_samples","segments"):
            if before["identity"].get(key)!=after["identity"].get(key):raise ValueError("continuous paired input differs: "+key)
        for b,c in zip(before["details"],after["details"]):
            entry=dict(label=b["label"],**gate_summary(dict(baseline=b["gates"],candidate=c["gates"])),
                candidate_failures=[g for g in c["gates"] if failure(g) and g["segment"]!="runtime"],sequence_id=sid,rate=rate,microphones=mics,variant=variant,pair=pair)
            pairs.append(entry)
    if set(by_window)!=expected_windows or report["pairs"]!=pairs or summarize_pairs(pairs,True)!=q:
        raise ValueError("continuous proof phase/window identity or aggregate quality differs")
    score_rows=p["scores"]
    if len(score_rows)!=3 or {s["metric"] for s in score_rows}!={"SIG","BAK","OVRL"}:raise ValueError("three perceptual metrics required")
    for row in score_rows:
        metric=row["metric"];b=np.array([w["baseline"]["poly"][metric] for w in windows]);c=np.array([w["candidate"]["poly"][metric] for w in windows])
        expected=dict(metric=metric,baseline_median=float(np.median(b)),candidate_median=float(np.median(c)),paired_delta_median=float(np.median(c-b)),windows=510)
        if row!=expected:raise ValueError("perceptual aggregate does not match all 510 scores")
        if metric in ("SIG","OVRL") and row["paired_delta_median"]<0:raise ValueError("SIG/OVRL paired medians declined")
    return report
def predicted_bytes(scene):
    import soundfile as sf
    # Conservative reservation for both sides, all modes, taps, CSV and measurements.
    return int(scene["seconds"]*scene["sample_rate"]*(scene.get("mics",1)+12)*4*len(ev.modes(scene))*2)+16*1024**2
def bind_proof(report,baseline_cli,candidate_cli,manifest_sha,candidate_flags):
    identity=report["identity"]
    if identity["cli_sha256"]!=sha(candidate_cli) or identity["baseline_cli_sha256"]!=sha(baseline_cli):
        raise ValueError("quality proof belongs to another baseline/candidate binary")
    if identity["rules_sha256"]!=sha(ac.RULES) or identity["independent_manifest_sha256"]!=manifest_sha:
        raise ValueError("quality proof threshold or independent data differs")
    if identity["v1_flags"]+identity["candidate_flags"]!=candidate_flags:
        raise ValueError("quality proof flags differ")
def stable_scene(scene):return {k:v for k,v in scene.items() if k!="denoise_sha256"}
def owned_unlink(path,root):
    path=Path(path);root=Path(root).resolve()
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError("refuse to remove non-owned output")
    path.unlink()
def source_folder(dataset,scene):
    return Path(scene.get("source_dataset",dataset))/"cases"/scene.get("source_case_id",scene["id"])
def check_source(dataset,scene):
    folder=source_folder(dataset,scene);inventory={}
    for name,info in scene["files"].items():
        p=folder/name
        if not p.is_file() or p.stat().st_size!=info["bytes"] or sha(p)!=info["sha256"]:
            raise ValueError("frozen input changed: "+scene["id"]+" "+name)
        inventory[name]=info["sha256"]
    return inventory
def flags_for(side,identity):
    return identity["baseline_flags"] if side=="baseline" else identity["candidate_flags"]
def verify_programs(identity):
    for name,expected in identity.get("dependency_sha256",{}).items():
        checked_file(Path(__file__).resolve().parents[2]/name,expected)
    if sha(contract.__file__)!=identity["gate_contract_sha256"]:raise ValueError("gate comparison implementation changed")
    if sha(perceptual.__file__)!=identity["perceptual_implementation_sha256"]:raise ValueError("perceptual implementation changed")
    if any(sha(path)!=expected for path,expected in ((Path(__file__),identity["runner_sha256"]),(Path(ev.__file__),identity["evaluator_sha256"]),(Path(ac.__file__),identity["acceptance_sha256"]),(ac.RULES,identity["rules_sha256"]))):
        raise ValueError("frozen evaluator/runner/rules changed during batch")



def verify_mode_config(scene,mode,argv,engine):
    for key,flag in (("aec_enabled","--no-aec"),("ns_enabled","--no-ns"),("agc2_enabled","--no-agc")):
        if engine.get(key) is not (flag not in argv):raise ValueError("actual mode module differs: "+mode+" "+key)
    expected_bf=scene.get("mics",1)>1 and "--beam-delays" in argv and "--no-bf" not in argv
    if engine.get("beamforming_enabled") is not expected_bf:raise ValueError("actual mode BF differs: "+mode)
    if mode=="correct_reference":
        if "--echo" not in argv or Path(argv[argv.index("--echo")+1]).name!="playback.wav":raise ValueError("correct_reference input differs")
def negative_controls(scene,rows,bypass_pcm_error=None):
    by_mode={r["mode"]:r for r in rows};rules=load_json(ac.RULES);records={}
    candidates=[("bypass","bypass","noise_only",rules["suppression"]["noise_only_min_attenuation_db"])]
    if scene["family"]=="ns":candidates.append(("NS disabled","no_ns","noise_only",rules["suppression"]["noise_only_min_attenuation_db"]))
    if scene["family"]=="aec":candidates.append(("AEC disabled","no_aec","far_only",rules["suppression"]["far_only_min_attenuation_db"]))
    for name,mode,segment,threshold in candidates:
        if mode not in by_mode:continue
        value=by_mode[mode]["segments"].get(segment,{}).get("total_attenuation_db")
        if value is not None and not math.isfinite(value):raise ValueError("nonfinite control measurement")
        records[name]=dict(mode=mode,segment=segment,value=value,threshold=threshold,
            input_sha256=scene["files"]["before.wav"]["sha256"],output_sha256=by_mode[mode]["output_sha256"],dsp_delay_samples=by_mode[mode]["engine"]["dsp_delay_samples"],
            status="unavailable" if value is None else "expected_failure" if value<threshold else "unexpected_pass")
    if "bypass" in by_mode:
        maximum=rules["runtime"]["bypass_max_pcm16_lsb_error"]
        if bypass_pcm_error is None or not math.isfinite(bypass_pcm_error) or not 0<=bypass_pcm_error<=maximum:
            raise ValueError("bypass PCM identity error unavailable or exceeds original bound")
        records["bypass"].update(max_pcm16_lsb_error=bypass_pcm_error,pcm16_lsb_limit=maximum,pcm_identity_verified=True,
            module_readback="AEC/NS/AGC/BF disabled",limiter_scope="exact --no-limiter invocation; no limiter-enabled JSON field")
    return records
def summarize_controls(done):
    result={}
    for side in ("baseline","candidate"):
        categories={}
        for case in done:
            for name,record in case.get("negative_controls",{}).get(side,{}).items():
                bucket=categories.setdefault(name,dict(counts={},max_pcm16_lsb_error=None))
                bucket["counts"][record["status"]]=bucket["counts"].get(record["status"],0)+1
                if "max_pcm16_lsb_error" in record:bucket["max_pcm16_lsb_error"]=max(bucket["max_pcm16_lsb_error"] or 0,record["max_pcm16_lsb_error"])
        for name,bucket in categories.items():bucket["verified"]=bucket["counts"].get("expected_failure",0)>0
        result[side]=categories
    return result
def commit_case(path,result):
    save(path,result)
    sidecar=Path(str(path)+".sha256");temporary=Path(str(sidecar)+".tmp")
    temporary.write_text(sha(path)+"\n");temporary.replace(sidecar)
def committed_sha(path):
    sidecar=Path(str(path)+".sha256")
    if not sidecar.is_file():raise ValueError("paired commit SHA sidecar missing; preserve unidentified prior case")
    value=sidecar.read_text().strip()
    if len(value)!=64 or any(c not in "0123456789abcdef" for c in value) or sha(path)!=value:
        raise ValueError("paired commit SHA mismatch")
    return value
def validate_case(root,scene,identity,result=None):
    root=Path(root);folder=root/"cases"/scene["id"];path=folder/"paired.json"
    committed_sha(path);result=load_json(path) if result is None else result
    if (not result.get("complete") or result["identity_sha256"]!=identity["identity_sha256"]
        or stable_scene(result["scene"])!=stable_scene(scene)):
        raise ValueError("resume case identity or completion changed")
    expected_modes=list(ev.modes(scene))
    if result["modes"]!=expected_modes or set(result["measurements"])!={"baseline","candidate"}:raise ValueError("complete paired mode inventory differs")
    for name,expected in result["retained_files"].items():
        artifact=root/name
        if artifact.is_symlink() or not artifact.resolve().is_relative_to(folder.resolve()):raise ValueError("retained output escapes owned case")
        checked_file(artifact,expected)
    full={}
    for side in ("baseline","candidate"):
        rows=result["measurements"][side]
        if [r["mode"] for r in rows]!=expected_modes:raise ValueError("missing or reordered measured modes")
        binary=root/"artifacts"/(side+"-cli");checked_file(binary,identity[side+"_cli_sha256"])
        side_folder=folder/side
        for name in scene["files"]:
            target=source_folder(root,scene)/name
            # Legacy datasets require the original manifest source_dataset; normal run check_source already verifies it.
            link=side_folder/name
            if not link.is_symlink() or not link.resolve().is_file():raise ValueError("frozen source link missing")
            if scene.get("source_dataset") and link.resolve()!=target.resolve():raise ValueError("source link identity differs")
        for row in rows:
            mode=row["mode"];run_folder=side_folder/"runs"/mode
            measurement=run_folder/"measurement.json"
            if load_json(measurement)!=row or result["retained_files"].get(str(measurement.relative_to(root)))!=sha(measurement):
                raise ValueError("paired measurement differs from retained measurement")
            invpath=run_folder/"invocation.json";inv=load_json(invpath)
            expected_argv=ev.invocation(scene,mode,binary,side_folder,identity["processing_profile"])+flags_for(side,identity)
            if inv["argv"]!=expected_argv or inv["cli_sha256"]!=identity[side+"_cli_sha256"] or inv["exit_code"]!=0:
                raise ValueError("paired invocation flags/binary/input binding differs")
            ev.verify_processing_profile(row["engine"],identity["processing_profile"])
            verify_mode_config(scene,mode,expected_argv,row["engine"])
            if row["id"]!=scene["id"] or row["family"]!=scene["family"] or row["engine"]["sample_rate"]!=scene["sample_rate"] or row["engine"]["microphones"]!=scene.get("mics",1):
                raise ValueError("paired measured scene or engine identity differs")
            for flag,key in (("--noise-transition-control","noise_transition_control"),("--ecnr-coordinated-control","ecnr_coordinated_control"),("--ecnr-v2-control","ecnr_v2_control")):
                if bool(row["engine"].get(key,False))!=(flag in flags_for(side,identity)):raise ValueError("effective engine control differs: "+key)
            raw_engine=load_json(run_folder/"result.json")
            for key,value in raw_engine.items():
                if key not in ("process_p95_ms","process_max_ms","process_rtf") and row["engine"].get(key)!=value:
                    raise ValueError("measured engine differs from CLI metadata: "+key)
            output_key=str((run_folder/"result.after.wav").relative_to(root))
            if result["output_hashes"].get(output_key)!=row["output_sha256"]:raise ValueError("output SHA not bound to measurement")
            for relative,expected in row["files"].items():
                key=str((side_folder/relative).relative_to(root))
                if result["output_hashes"].get(key)!=expected:raise ValueError("original output inventory differs from measurements")
            if mode=="full":full[side]=row
    gates={side:ac.check_scene(scene,full[side],load_json(ac.RULES))["gates"] for side in full}
    if result["gates"]!=gates:raise ValueError("paired original gates differ from retained full measurements")
    summary=gate_summary(gates)
    for key,value in summary.items():
        if result.get(key)!=value:raise ValueError("paired gate summary differs: "+key)
    failed_case=not summary["baseline_pass"] or not summary["candidate_pass"]
    if failed_case:
        for side in ("baseline","candidate"):
            key=str((folder/side/"runs/full/result.after.wav").relative_to(root))
            if key not in result["retained_audio"] or key not in result["retained_files"]:raise ValueError("failed pair full audio missing")
    expected_csv={str((folder/side/"runs"/mode/"result.frames.csv.gz").relative_to(root)) for side in ("baseline","candidate") for mode in expected_modes}
    if not expected_csv.issubset(result["compressed_timing_logs"]):raise ValueError("complete timing log inventory missing")
    for name,record in result["compressed_timing_logs"].items():
        if result["retained_files"].get(name)!=record["compressed_sha256"] or result["output_hashes"].get(name[:-3])!=record["original_sha256"]:
            raise ValueError("gzip log does not bind original full timing log")
    if set(result.get("negative_controls",{}))!={"baseline","candidate"}:raise ValueError("paired control evidence missing")
    for side in ("baseline","candidate"):
        stored=result["negative_controls"][side]
        error=stored.get("bypass",{}).get("max_pcm16_lsb_error")
        if negative_controls(scene,result["measurements"][side],error)!=stored:raise ValueError("paired negative-control summary differs")
    windows=result["perceptual"]["windows"]
    expected={(kind,tuple(limits)) for kind,limits in scene["windows"].items() if kind in perceptual.ALLOWED_ROLES}
    actual={(w["kind"],tuple(w["window_s"])) for w in windows}
    if len(windows)!=len(actual) or actual!=expected:raise ValueError("paired perceptual inventory differs")
    for w in windows:
        available=all(w[side]["available"] for side in ("baseline","candidate"))
        expected_delta={k:w["candidate"]["poly"][k]-w["baseline"]["poly"][k] for k in ("SIG","BAK","OVRL")} if available else None
        if w["paired_delta"]!=expected_delta:raise ValueError("paired perceptual delta differs")
        if expected_delta and not all(math.isfinite(v) for v in expected_delta.values()):raise ValueError("nonfinite paired perceptual delta")
    return result
def dependency_sources():
    # Resolve only local Python imports; third-party/runtime versions are recorded by perceptual preprocessing.
    folder=Path(__file__).parent;todo=[Path(__file__),"ecnr_v2_evidence.py","ecnr_window_audit.py"];found={}
    while todo:
        path=todo.pop();path=path if isinstance(path,Path) else folder/path
        if path.name in found or not path.is_file():continue
        found[path.name]=path
        for node in ast.walk(ast.parse(path.read_text())):
            names=[a.name.split(".")[0] for a in node.names] if isinstance(node,ast.Import) else [node.module.split(".")[0]] if isinstance(node,ast.ImportFrom) and node.module else []
            for name in names:
                q=folder/(name+".py")
                if q.is_file():todo.append(q)
    return {"cpp/tools/"+name:path for name,path in found.items()}
def evidence_sources(args,proof):
    sources=dependency_sources();sources["cpp/evaluation/"+ac.RULES.name]=ac.RULES
    sources.update({side+"-cli":Path(binary) for side,binary in (("baseline",args.baseline_cli),("candidate",args.candidate_cli))})
    hpf=Path(__file__).resolve().parents[2]/"build/webrtc-src/modules/audio_processing/high_pass_filter.cc"
    if hpf.is_file():sources["build/webrtc-src/modules/audio_processing/high_pass_filter.cc"]=hpf
    if proof:
        for name,path in _PROOF_FILES.items():sources["proof/"+name]=path
    assets=perceptual.DEFAULT_ASSETS
    for name in ("manifest.json","sig_bak_ovr.onnx","dnsmos_local.py"):
        if (assets/name).is_file():sources["perceptual-assets/"+name]=assets/name
    return sources
def verify_artifacts(root,identity):
    for name,expected in identity["artifact_files"].items():checked_file(Path(root)/"artifacts"/name,expected)
    checked_file(Path(root)/"inputs-manifest.json",identity["source_manifest_sha256"])

def measure_mode(scene,mode,side_folder,quality_cache=None):
    # A partial case may already have measurement.json. It cannot hash its own prior version.
    measured=ev.measure(scene,mode,side_folder,write_metrics=False,quality_cache=quality_cache)
    own=str((Path("runs")/mode/"measurement.json"))
    measured["files"].pop(own,None)
    return measured
def require_snapshot_space(root,sources,manifest,reserve_bytes):
    root=Path(root);anchor=root.parent
    while not anchor.exists():anchor=anchor.parent
    snapshot_bytes=sum(Path(p).stat().st_size for p in sources.values())+Path(manifest).stat().st_size
    # Also reserve metadata/identity and copy overlap before the first output directory is created.
    required=reserve_bytes+snapshot_bytes+16*1024**2
    available=shutil.disk_usage(anchor).free
    if available<required:raise ValueError(f"snapshot disk reserve: available={available} required={required}")
    return dict(available_bytes=available,required_bytes=required,snapshot_bytes=snapshot_bytes,reserve_bytes=reserve_bytes)
def run_case(arguments):
    dataset,root,scene,identity,representative,reserve_bytes=arguments
    dataset=Path(dataset);root=Path(root);folder=root/"cases"/scene["id"]
    complete=folder/"paired.json"
    verify_programs(identity)
    check_source(dataset,scene)
    if complete.exists():return validate_case(root,scene,identity)
    modes=list(ev.modes(scene))
    # Includes float/PCM taps, CSV, log and peak simultaneous writes. Never consume the reserve.
    predicted=predicted_bytes(scene)
    if shutil.disk_usage(root).free<reserve_bytes+predicted:
        return dict(blocked=True,id=scene["id"],reason="disk reserve",available_bytes=shutil.disk_usage(root).free,required_bytes=reserve_bytes+predicted)
    folder.mkdir(parents=True,exist_ok=True)
    rows_by_side={};retained={};all_outputs={};mode_invocations=[];bypass_pcm_errors={}
    for side in ("baseline","candidate"):
        side_folder=folder/side;side_folder.mkdir(exist_ok=True)
        for name in scene["files"]:
            link=side_folder/name;target=source_folder(dataset,scene)/name
            if not link.exists():link.symlink_to(target)
            if link.resolve()!=target.resolve():raise ValueError("input link changed")
        binary=root/"artifacts"/(side+"-cli")
        if sha(binary)!=identity[side+"_cli_sha256"]:raise ValueError("frozen binary changed")
        cache={};rows=[]
        for mode in modes:
            prefix=side_folder/"runs"/mode/"result";prefix.parent.mkdir(parents=True,exist_ok=True)
            command=ev.invocation(scene,mode,binary,side_folder,identity["processing_profile"])+flags_for(side,identity)
            invocation=prefix.parent/"invocation.json"
            metric_path=prefix.parent/"measurement.json"
            if invocation.exists():
                old=json.loads(invocation.read_text())
                if old["argv"]!=command or old["cli_sha256"]!=identity[side+"_cli_sha256"] or old["exit_code"]!=0:
                    raise ValueError("partial run identity changed or execution failed")
            else:
                started=time.perf_counter()
                save(invocation,dict(argv=command,cli_sha256=identity[side+"_cli_sha256"],exit_code=None,state="running"))
                try:
                    run=subprocess.run(command,text=True,capture_output=True,timeout=180,
                        env=dict(os.environ,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1"))
                except subprocess.TimeoutExpired as error:
                    output=lambda v:v.decode(errors="replace") if isinstance(v,bytes) else v
                    save(invocation,dict(argv=command,cli_sha256=identity[side+"_cli_sha256"],exit_code=None,state="timeout",
                        stdout=output(error.stdout),stderr=output(error.stderr),elapsed_s=time.perf_counter()-started));raise
                save(invocation,dict(argv=command,cli_sha256=identity[side+"_cli_sha256"],exit_code=run.returncode,state="completed",
                     stdout=run.stdout,stderr=run.stderr,elapsed_s=time.perf_counter()-started))
                if run.returncode:raise RuntimeError(f"{scene['id']}/{side}/{mode} failed; retained invocation")
            measured=measure_mode(scene,mode,side_folder,quality_cache=cache);ev.verify_processing_profile(measured["engine"],identity["processing_profile"])
            verify_mode_config(scene,mode,command,measured["engine"])
            save(metric_path,measured);rows.append(measured)
            hashes={str(p.relative_to(root)):sha(p) for p in sorted(prefix.parent.glob("*")) if p.is_file() and not p.is_symlink()}
            all_outputs.update(hashes);mode_invocations.append(dict(side=side,mode=mode,files=hashes))
            if mode=="bypass":
                import soundfile as sf
                import numpy as np
                before,rate=sf.read(side_folder/"before.wav",always_2d=True)
                output,_=sf.read(prefix.with_suffix(".after.wav"),always_2d=True)
                if output.shape!=(len(before),1) or np.max(np.abs(output[:,0]-before[:,0]))>1/32768:
                    raise ValueError("bypass changed microphone channel zero")
                bypass_pcm_errors[side]=float(np.max(np.abs(output[:,0]-before[:,0]))*32768)
        rows_by_side[side]=rows
    full={side:next(x for x in rows_by_side[side] if x["mode"]=="full") for side in rows_by_side}
    global _SCORER
    if _SCORER is None:_SCORER=perceptual.DNSMOS(perceptual.DEFAULT_ASSETS,root/"perceptual-cache")
    import soundfile as sf
    score_by_side={}
    for side in ("baseline","candidate"):
        values,rate=sf.read(folder/side/"runs/full/result.after.wav",dtype="float64",always_2d=True)
        delay=full[side]["engine"]["dsp_delay_samples"];values=values[delay:,0]
        records=[]
        for kind,limits in scene["windows"].items():
            if kind not in perceptual.ALLOWED_ROLES:continue
            a,b=(round(x*rate) for x in limits)
            records.append(dict(kind=kind,window_s=limits,score=_SCORER.score_array(values[a:b],rate,kind=kind)))
        score_by_side[side]=records
    if [(x["kind"],x["window_s"]) for x in score_by_side["baseline"]]!=[(x["kind"],x["window_s"]) for x in score_by_side["candidate"]]:
        raise ValueError("perceptual window contract differs")
    paired_scores=[]
    for b,c in zip(score_by_side["baseline"],score_by_side["candidate"]):
        available=b["score"]["available"] and c["score"]["available"]
        paired_scores.append(dict(kind=b["kind"],window_s=b["window_s"],baseline=b["score"],candidate=c["score"],
            paired_delta={k:c["score"]["poly"][k]-b["score"]["poly"][k] for k in ("SIG","BAK","OVRL")} if available else None))

    rules=json.loads(ac.RULES.read_text())
    gates={side:ac.check_scene(scene,full[side],rules)["gates"] for side in full}
    gb,gc,branch_changes=contract.paired_contract(gates["baseline"],gates["candidate"])
    failed=lambda g:g["status"]=="fail" or g["status"]=="unavailable" and not g["optional"]
    bpass,cpass=quality(gates["baseline"]),quality(gates["candidate"])
    failed_case=not bpass or not cpass
    retained_audio=[];release_audio=[];release_logs=[];compressed_logs={}
    for side in rows_by_side:
        for csv_log in (folder/side/"runs").glob("*/*.csv"):
            packed=csv_log.with_suffix(csv_log.suffix+".gz")
            with csv_log.open("rb") as src, gzip.GzipFile(filename=str(packed),mode="wb",mtime=0) as dst:shutil.copyfileobj(src,dst)
            with gzip.open(packed,"rb") as src:
                if hashlib.file_digest(src,"sha256").hexdigest()!=sha(csv_log):raise ValueError("lossless timing log verification failed")
            compressed_logs[str(packed.relative_to(root))]=dict(compressed_sha256=sha(packed),original_sha256=sha(csv_log))
            release_logs.append(csv_log)
    for side in rows_by_side:
        for wave in (folder/side/"runs").glob("*/*.wav"):
            key=str(wave.relative_to(root))
            if representative or failed_case and wave.parent.name=="full" and wave.name=="result.after.wav":retained_audio.append(key)
            else:release_audio.append(wave)
    # Commit measurements and hashes before releasing any owned successful audio.
    for p in folder.rglob("*"):
        if p.is_file() and not p.is_symlink() and p.name not in ("paired.json","paired.json.sha256") and p not in release_audio and p not in release_logs:retained[str(p.relative_to(root))]=sha(p)
    result=dict(schema_version=1,identity_sha256=identity["identity_sha256"],scene=scene,modes=modes,
        measurements=rows_by_side,negative_controls={side:negative_controls(scene,rows,bypass_pcm_errors.get(side)) for side,rows in rows_by_side.items()},gates=gates,attenuation_branch_changes=branch_changes,baseline_pass=bpass,candidate_pass=cpass,
        new_failures=[gc[k] for k in gc if failed(gc[k]) and not failed(gb[k])],
        repaired_gates=[gc[k] for k in gc if gc[k]["status"]=="pass" and failed(gb[k])],
        perceptual=dict(scope="predicted DNSMOS P.835, full-mode fixed-delay near speech only",windows=paired_scores,
            preprocessing=_SCORER.preprocessing,preprocessing_sha256=_SCORER.preprocessing_sha256),
        output_hashes=all_outputs,retained_files=retained,retained_audio=retained_audio,compressed_timing_logs=compressed_logs,
        audio_retention_reason=("all mode audio for deterministic representative" if representative else
            "both full final outputs for failed case; frozen inputs remain linked" if failed_case else
            "all metrics/lossless timing logs/hashes retained; owned successful audio released"),
        complete=True)
    verify_programs(identity);commit_case(complete,result)
    validate_case(root,scene,identity,result)
    for wave in release_audio+release_logs:owned_unlink(wave,root)
    return result


def case_index_entry(root,result):
    path=Path(root)/"cases"/result["scene"]["id"]/"paired.json"
    return dict(id=result["scene"]["id"],baseline_pass=result["baseline_pass"],candidate_pass=result["candidate_pass"],
        modes=result["modes"],negative_controls=result["negative_controls"],new_failures=result["new_failures"],repaired_gates=result["repaired_gates"],
        perceptual_deltas=[x["paired_delta"] for x in result["perceptual"]["windows"]],result_path=str(path),paired_sha256=committed_sha(path))
def verify_prior_index(root,identity,scenes):
    root=Path(root);path=root/"paired-index.json"
    if not path.exists():return
    index=load_json(path);entries=index["scenes"];expected={s["id"] for s in scenes};ids=[e["id"] for e in entries]
    if index["identity_sha256"]!=identity["identity_sha256"] or len(ids)!=len(set(ids)) or not set(ids).issubset(expected):
        raise ValueError("resume paired index identity or scene inventory changed")
    for entry in entries:
        case=root/"cases"/entry["id"]/"paired.json"
        if entry!=case_index_entry(root,load_json(case)):raise ValueError("resume paired index commit/summary differs")
    prior=load_json(root/"report.json")
    if (prior["identity"]!=identity or prior["processed_scenes"]!=len(entries) or prior["quality"]!=summarize_pairs(entries)
        or prior["runs"]!=sum(2*len(e["modes"]) for e in entries) or prior["negative_controls"]!=summarize_controls(entries)
        or prior["complete"]!=index["complete"]):
        raise ValueError("resume report aggregate differs from committed paired index")
def run(args):
    dataset=args.dataset.resolve();root=args.root.resolve()
    if sha(dataset/"inputs-manifest.json")!=EXPECTED_MANIFEST_SHA256:raise ValueError("current frozen 20,298 manifest changed")
    source_manifest=json.loads((dataset/"inputs-manifest.json").read_text())
    scenes=source_manifest["scenes"]
    if len(scenes)!=EXPECTED_SCENES or len({x["id"] for x in scenes})!=EXPECTED_SCENES:raise ValueError("full frozen 20,298 scene inventory required")
    if any(x.get("catalog_excluded") for x in scenes):raise ValueError("current denominator includes excluded inputs")
    proof=None
    if not args.smoke_first:proof=require_proof(args.quality_proof)
    elif args.smoke_first<1:raise ValueError("positive smoke inventory required")
    baseline_flags=["--noise-transition-control","--ecnr-coordinated-control"]
    candidate_flags=baseline_flags+args.cli_option
    if proof:bind_proof(proof,args.baseline_cli,args.candidate_cli,sha(dataset/"inputs-manifest.json"),candidate_flags)
    sources=evidence_sources(args,proof)
    dependency_hashes={name:sha(path) for name,path in dependency_sources().items()}
    artifact_hashes={name:sha(path) for name,path in sources.items()}
    identity=dict(schema_version=2,dependency_sha256=dependency_hashes,artifact_files=artifact_hashes,source_manifest_sha256=sha(dataset/"inputs-manifest.json"),expected_scenes=EXPECTED_SCENES,
        baseline_cli_sha256=sha(args.baseline_cli),candidate_cli_sha256=sha(args.candidate_cli),
        baseline_flags=baseline_flags,candidate_flags=candidate_flags,processing_profile="vehicle_stream",rules_sha256=sha(ac.RULES),
        gate_contract_sha256=sha(contract.__file__),perceptual_implementation_sha256=sha(perceptual.__file__),perceptual_model_sha256=perceptual.MODEL_SHA,
        runner_sha256=sha(__file__),evaluator_sha256=sha(ev.__file__),acceptance_sha256=sha(ac.__file__),
        smoke_first=args.smoke_first,quality_proof_sha256=sha(args.quality_proof) if proof else None,
        mode_contract="all ev.modes(scene), same profile; full quality gates and diagnostic controls reported separately",
        audio_retention="all original gates, lossless gzip timing logs and output hash; failures retain both full final outputs, family representatives all mode audio; frozen inputs preserved",
        scope="independent host simulation; no Android/vehicle or ITU-T certification")
    identity["identity_sha256"]=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    if root.exists():
        if not args.resume or json.loads((root/"identity.json").read_text())!=identity:raise ValueError("batch identity differs; preserve previous batch")
    else:
        require_snapshot_space(root,sources,dataset/"inputs-manifest.json",args.reserve_gib*1024**3)
        root.mkdir(parents=True);(root/"artifacts").mkdir()
        for side,source in (("baseline",args.baseline_cli),("candidate",args.candidate_cli)):shutil.copy2(source,root/"artifacts"/(side+"-cli"))
        shutil.copy2(dataset/"inputs-manifest.json",root/"inputs-manifest.json");save(root/"identity.json",identity)
        for name,source in sources.items():
            target=root/"artifacts"/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    verify_artifacts(root,identity)
    verify_prior_index(root,identity,scenes)
    if args.smoke_first:
        strata={}
        for scene in scenes:strata.setdefault((scene["family"],scene.get("stress","none")),[]).append(scene)
        selected=[]
        for index in range(args.smoke_first):
            keys=list(strata);key=keys[index%len(keys)];round_number=index//len(keys)
            if round_number<len(strata[key]):selected.append(strata[key][round_number])
    else:selected=scenes
    representatives=set()
    seen=Counter()
    for scene in selected:
        key=(scene["family"],scene.get("stress","none"))
        if seen[key]<2:representatives.add(scene["id"])
        seen[key]+=1
    done=[];blocked=[];started=time.time()
    def write(state):
        result=dict(schema_version=1,state=state,complete=state=="completed" and not args.smoke_first,
            identity_verified=not blocked,identity=identity,total_scenes=EXPECTED_SCENES,processed_scenes=len(done),
            scope="smoke only" if args.smoke_first else "full independent scene paired regression",
            quality=dict(baseline_pass=sum(x["baseline_pass"] for x in done),candidate_pass=sum(x["candidate_pass"] for x in done),
                phase_wins=sum(not x["baseline_pass"] and x["candidate_pass"] for x in done),
                phase_regressions=sum(x["baseline_pass"] and not x["candidate_pass"] for x in done),
                new_failed_gates=sum(len(x["new_failures"]) for x in done),repaired_gates=sum(len(x["repaired_gates"]) for x in done)),
            runs=sum(2*len(x["modes"]) for x in done),elapsed_s=time.time()-started,
            timing_scope="concurrent offline timing, not isolated paced performance",
            negative_controls=summarize_controls(done),blocked=blocked,recommendation=dict(eligible=False,reason="Continuous quality/perceptual and isolated timing are separate gates."))
        save(root/"report.json",result);return result
    write("running")
    arguments=[(str(dataset),str(root),s,identity,s["id"] in representatives,args.reserve_gib*1024**3) for s in selected]
    # Bound in-flight work rather than scheduling all 20k before disk checks can stop the batch.
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
        pending={};iterator=iter(arguments);queued=None
        def submit_next():
            nonlocal queued
            if queued is None:
                try:queued=next(iterator)
                except StopIteration:return False
            reservation=predicted_bytes(queued[2])
            in_flight=sum(value[1] for value in pending.values())
            required=args.reserve_gib*1024**3+in_flight+reservation
            if shutil.disk_usage(root).free<required:
                if pending:return False
                blocked.append(dict(id=queued[2]["id"],reason="parent disk reservation",required_bytes=required,available_bytes=shutil.disk_usage(root).free))
                return False
            pending[pool.submit(run_case,queued)]=(queued[2]["id"],reservation);queued=None;return True
        while len(pending)<args.jobs and submit_next():pass
        while pending:
            finished,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in finished:
                sid,reservation=pending.pop(future)
                try:result=future.result()
                except Exception as error:blocked.append(dict(id=sid,reason=str(error)));result=None
                if result and result.get("blocked"):blocked.append(result)
                elif result:done.append(case_index_entry(root,result))
            if not blocked:
                while len(pending)<args.jobs and submit_next():pass
            current=write("blocked" if blocked else "running")
            if len(done)%100==0 or blocked:print(json.dumps({k:current[k] for k in ("state","processed_scenes","runs","quality","elapsed_s","blocked")}),flush=True)
    verify_programs(identity);verify_artifacts(root,identity)
    ids=[x["id"] for x in done]
    if len(ids)!=len(set(ids)) or not set(ids).issubset({s["id"] for s in selected}):raise ValueError("paired index scene identity differs")
    complete=len(done)==len(selected) and not blocked
    import numpy as np
    deltas=[x for scene in done for x in scene["perceptual_deltas"] if x is not None]
    available_count=len(deltas);expected_count=sum(len(scene["perceptual_deltas"]) for scene in done)
    perceptual_summary=dict(status="completed" if available_count==expected_count and complete else "unavailable",
        paired_windows=available_count,expected_windows=expected_count,
        paired_delta_median={k:float(np.median([x[k] for x in deltas])) if deltas else None for k in ("SIG","BAK","OVRL")})
    report=write("completed" if complete else "blocked");report["perceptual"]=perceptual_summary;save(root/"report.json",report)
    save(root/"paired-index.json",dict(identity_sha256=identity["identity_sha256"],complete=report["complete"],scenes=done))
    print(json.dumps({k:report[k] for k in ("state","complete","processed_scenes","runs","quality","elapsed_s","blocked")}),flush=True)
    return 0 if complete else 2
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("dataset","root","baseline-cli","candidate-cli"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--quality-proof",type=Path);p.add_argument("--cli-option",action="append",default=[])
    p.add_argument("--jobs",type=int,default=4);p.add_argument("--reserve-gib",type=int,default=3)
    p.add_argument("--resume",action="store_true");p.add_argument("--smoke-first",type=int)
    args=p.parse_args()
    if args.jobs<1 or args.jobs>8 or args.reserve_gib<2:raise ValueError("jobs 1..8 and >=2GiB reserve required")
    try:return run(args)
    except Exception as error:
        if (args.root/"report.json").exists():
            report=json.loads((args.root/"report.json").read_text());report.update(state="blocked",complete=False,identity_verified=False)
            report["blocked"]=[dict(reason=str(error),error_type=type(error).__name__)];save(args.root/"report.json",report)
        save(args.root/"failure.json",dict(reason=str(error),error_type=type(error).__name__))
        raise
if __name__=="__main__":raise SystemExit(main())
