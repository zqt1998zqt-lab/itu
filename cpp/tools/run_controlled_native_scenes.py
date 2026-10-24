#!/usr/bin/env python3
"""Controlled native-rate sweeps and vehicle stress replay; frozen DSP gates."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
from scipy import signal
import soundfile as sf

import run_audio_eval as ev
import run_native_rate_scenes_v3 as legacy
import audit_dataset_quality_deep as audit

CACHE = {}


def stable_seed(value):
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16)


def load_sources(manifest, rate, split):
    key = (str(manifest), rate, split)
    if key in CACHE:
        return CACHE[key]
    rows = [e for e in json.loads(manifest.read_text())["sources"]
            if e["sample_rate"] == rate and e["split"] == split]
    speech, evidence, noise, noise_evidence = defaultdict(list), defaultdict(list), {}, {}
    for entry in rows:
        path = Path(entry["path"])
        if ev.digest(path) != entry["sha256"]:
            raise ValueError(f"source hash mismatch: {path}")
        audio, actual = sf.read(path, dtype="float64", always_2d=True)
        if actual != rate or not len(audio) or not np.isfinite(audio).all():
            raise ValueError(f"invalid source: {path}")
        if entry["kind"] == "clean_speech" and entry.get("canonical_speaker_id"):
            speaker = entry["canonical_speaker_id"]
            evidence[speaker].append(dict(path=str(path), sha256=entry["sha256"],
                samples=len(audio), rate=rate, language=entry.get("language", "en"),
                recording_id=entry["recording_id"], split=split,
                pool_start_sample=sum(len(v) for v in speech[speaker])))
            speech[speaker].append(audio[:, 0])
        elif entry["kind"] == "noise":
            name = entry["scene"].lower()
            if name in noise:
                raise ValueError("multiple recordings must have distinct noise keys")
            noise[name] = audio[:, 0]
            noise_evidence[name] = dict(path=str(path), sha256=entry["sha256"],
                recording_id=entry["recording_id"], vehicle_scope=entry.get("vehicle_scope"), split=split)
    result = ({k: np.concatenate(v) for k, v in speech.items()}, noise, evidence, noise_evidence)
    CACHE[key] = result
    return result


def source_context(case, manifest):
    rate = case["sample_rate"]
    speech, noise, evidence, noise_evidence = load_sources(manifest, rate, case["split"])
    if case["noise"].startswith("sim_"):
        noise = dict(noise)
        noise_evidence = dict(noise_evidence)
        rng=np.random.default_rng(stable_seed(case["split"]+case["noise"]+str(rate)))
        t=np.arange(rate*40)/rate
        if case["noise"]=="sim_wind_gust":
            generated=signal.sosfilt(signal.butter(2,250,fs=rate,output="sos"),rng.normal(size=len(t)))
            generated*=.15+.85*np.sin(2*np.pi*.4*t)**2
        elif case["noise"]=="sim_engine_ramp":
            phase=2*np.pi*(40*t+25*t*t/40)
            generated=sum(np.sin(h*phase)/h for h in (1,2,3,4))+.1*rng.normal(size=len(t))
        elif case["noise"]=="sim_rain_impulse":
            drops=rng.exponential(size=len(t))*(rng.random(len(t))<.02)
            generated=.2*rng.normal(size=len(t))+signal.lfilter([1,.6,.3],[1],drops-np.mean(drops))
        elif case["noise"]=="sim_indicator":
            generated=np.exp(-(t%.5)*120)*np.sin(2*np.pi*1800*t)+.02*rng.normal(size=len(t))
        else:
            raise ValueError("unreviewed synthetic vehicle condition")
        noise[case["noise"]]=generated
        noise_evidence[case["noise"]]=dict(recording_id=None,vehicle_scope="procedural_stress_model_not_a_vehicle_recording",
            split=case["split"],generation_seed=stable_seed(case["split"]+case["noise"]+str(rate)))
    rolled, used = {}, {}
    for speaker in (case["near_speaker"], case["far_speaker"]):
        audio = speech[speaker]
        offset = stable_seed(case["source_phase_key"] + speaker) % len(audio)
        rolled[speaker] = np.roll(audio, -offset)
        # schedule consumes 20 s for NS, 13 s for acoustic near and 20 s far.
        samples = round((20 if case["family"] in ("ns", "agc") or speaker == case["far_speaker"]
                         or case["stress"]=="wrong_reference" else 13) * rate)
        intervals = []
        remaining, cursor = min(samples, len(audio)), offset
        while remaining:
            size = min(remaining, len(audio) - cursor)
            intervals.append((cursor, cursor + size))
            remaining -= size
            cursor = 0
        used[speaker] = [dict(e, onset_phase_offset_samples=offset)
                         for e in evidence[speaker]
                         if any(e["pool_start_sample"] < hi and e["pool_start_sample"]+e["samples"] > lo for lo, hi in intervals)]
    return rolled, noise, used, noise_evidence


def matrix(manifest):
    inventory = json.loads(manifest.read_text())["sources"]
    groups = []
    selected = []
    for rate in (16000, 24000, 48000):
        for split in ("development", "validation", "holdout"):
            records = [e for e in inventory if e["sample_rate"] == rate and e["split"] == split]
            speakers = sorted({e["canonical_speaker_id"] for e in records if e.get("canonical_speaker_id")})
            if len(speakers) < 2:
                continue
            full = set()
            for language in ("en", "zh"):
                candidates = sorted({e["canonical_speaker_id"] for e in records if e.get("language") == language
                                     and e.get("canonical_speaker_id")}, key=lambda v: stable_seed("selection:"+v))
                # Two verified speakers per language/split where available; all others get coverage probes.
                full.update(candidates[:2])
            real = sorted({e["scene"].lower() for e in records if e["kind"] == "noise"})
            for pair, speaker in enumerate(speakers):
                near_entries = [e for e in records if e.get("canonical_speaker_id") == speaker]
                language = near_entries[0].get("language", "en")
                far = speakers[(speakers.index(speaker)+1) % len(speakers)]
                label = f"{rate}-{split}-{stable_seed(speaker):08x}"
                base = dict(sample_rate=rate, split=split, pair=pair, near_speaker=speaker, far_speaker=far,
                    language=language, near_dbfs=-32., mics=1, noise="none", snr_db=None, ser_db=-5,
                    rt60_s=.15, echo_delay_ms=30, stress="none", profile="controlled-native-v1", pcm_subtype="FLOAT",
                    room_dimensions_m=[3.6, 1.8, 1.5], near_position_m=[.8,.65,1.1],
                    render_positions_m=[[.2,.3,.8],[.2,1.5,.8]], field_label="natural_onset",
                    source_phase_key=label)
                noises = ["white", "pink"] + real
                if speaker not in full:
                    groups.append(dict(base, id="coverage-"+label, family="ns", noise=(real or ["pink"])[0],
                                       grid=[10,30], seed=stable_seed(label+":coverage"), grid_kind="speaker_coverage_probe"))
                    continue
                selected.append(dict(sample_rate=rate, split=split, speaker=speaker, language=language))
                for noise in noises:
                    group_id = f"sweep-{label}-{noise}"
                    groups.append(dict(base, id=group_id, family="ns", noise=noise,
                        grid=list(range(-10,31)), seed=stable_seed(group_id), grid_kind="one_db_fixed_target_sweep"))
                # Sparse array probes complement the full one-dB mono scans.
                for mics in (2,4):
                    group_id = f"array-{label}-m{mics}"
                    groups.append(dict(base, id=group_id, family="ns", mics=mics, noise=(real or ["pink"])[0],
                        grid=[-10,0,10,20,30], seed=stable_seed(group_id), grid_kind="array_coverage_probe",
                        spatial_noise_model="spherical"))
    # One independently scheduled acoustic/stress pair per rate/split, all mic counts.
    for anchor in [s for i,s in enumerate(selected) if not any(p["sample_rate"]==s["sample_rate"] and p["split"]==s["split"] for p in selected[:i])]:
        rate, split, speaker = anchor["sample_rate"], anchor["split"], anchor["speaker"]
        template = next(g for g in groups if g["sample_rate"]==rate and g["split"]==split and g["near_speaker"]==speaker)
        real = sorted({e["scene"].lower() for e in inventory if e["sample_rate"]==rate and e["split"]==split and e["kind"]=="noise"})
        for mics in (1,2,4):
            base = dict(template, mics=mics, grid=[None], grid_kind="vehicle_stress_probe", field_label="acoustic",
                        source_phase_key=f"stress-{rate}-{split}")
            specs = [dict(family="aec",noise="none",snr_db=None,rt60_s=rt,echo_delay_ms=delay)
                     for rt,delay in ((.15,20),(.45,80),(.6,180))]
            specs += [dict(family="combined",noise=(real or ["pink"])[0],snr_db=15,rt60_s=.3)]
            specs += [dict(family="stress",noise=(real or ["pink"])[0],snr_db=15,rt60_s=.3,
                           stress="clock_drift",clock_ppm=ppm) for ppm in (-300,-100,100,300)]
            specs += [dict(family="stress",noise=(real or ["pink"])[0],snr_db=15,stress=stress)
                      for stress in ("nonlinear_speaker","wrong_reference")]
            if mics>1:
                specs += [dict(family="combined",noise=(real or ["pink"])[0],snr_db=10,near_rt60_s=.2,
                               steering_error_samples=rate*.00015,field_label="steering_error")]
            specs += [dict(family="agc",noise=(real or ["pink"])[0],snr_db=20,near_dbfs=level)
                      for level in (-45,-30,-15)]
            for index,spec in enumerate(specs):
                group_id=f"vehicle-{rate}-{split}-m{mics}-{index:02d}"
                groups.append(dict(base,**spec,id=group_id,seed=stable_seed(group_id)))
            for kind in ("sim_wind_gust","sim_engine_ramp","sim_rain_impulse","sim_indicator"):
                group_id=f"vehicle-noise-{rate}-{split}-m{mics}-{kind}"
                condition=dict(base,id=group_id,family="ns",noise=kind,grid=[-10,10,30],
                               grid_kind="synthetic_vehicle_probe",seed=stable_seed(group_id))
                if mics>1:
                    condition["spatial_noise_model"]="corcos" if kind=="sim_wind_gust" else "spherical"
                groups.append(condition)
            group_id=f"noise-step-{rate}-{split}-m{mics}"
            groups.append(dict(base,id=group_id,family="ns",noise=(real or ["pink"])[0],grid=[20],
                grid_kind="transient_probe",seed=stable_seed(group_id),field_label="noise_step",
                noise_steps=[dict(time_s=11,gain_db=9)]))
    for group in groups:
        mics=group["mics"]
        group["microphone_positions_m"]=[[1.4+(ch-(mics-1)/2)*.035,.9,1.3] for ch in range(mics)]
        group["noise_origin"] = "none" if group["noise"]=="none" else "synthetic at requested rate" if group["noise"].startswith("sim_") or group["noise"] in ("white","pink") else "published-rate real recording"
        group["sample_rate_conversion"] = False
        if group["family"] == "ns":
            group["near_windows"] = [[0,10],[12,22]]
    return groups


def fixed_components(clean, noise, mask, grid, target_dbfs=-32.):
    clean = clean * math.sqrt(10**(target_dbfs/10) / ev.power(clean[mask,0]))
    noise = noise * math.sqrt(ev.power(clean[mask,0]) / ev.power(noise[mask,0]))
    # A single conservative bound protects every grid point, including components.
    peak = max(max(float(np.max(np.abs(clean+noise*10**(-snr/20)))),
                   float(np.max(np.abs(noise*10**(-snr/20)))), float(np.max(np.abs(clean)))) for snr in grid)
    gain = min(1., .88/max(peak,ev.EPS))
    return clean*gain, noise*gain, gain


def write_components(folder, scene, clean, noise, echo, reference=None, playback=None, stable_payloads=None):
    n, rate = len(clean), scene["sample_rate"]
    reference = np.zeros(n) if reference is None else reference
    playback = reference if playback is None else playback
    compensation = np.asarray(scene["beam_delays_samples"])
    beam_delay = int(np.ceil(compensation.max()))
    beam = np.mean([ev.shift(np.pad(clean[:,ch],(0,beam_delay)), compensation[ch])
                    for ch in range(scene["mics"])],axis=0)[beam_delay:beam_delay+n]
    for name,data in {"before.wav":clean+noise+echo,"clean.wav":clean[:,0],"clean-mics.wav":clean,
                      "clean-beam.wav":beam,"noise-component.wav":noise,"echo-component.wav":echo,
                      "ref.wav":reference,"playback.wav":playback}.items():
        if stable_payloads is not None and name in stable_payloads:
            (folder/name).write_bytes(stable_payloads[name])
        else:
            ev.write(folder/name,data,"FLOAT")
            if stable_payloads is not None and name not in ("before.wav","noise-component.wav"):
                stable_payloads[name]=(folder/name).read_bytes()
    mask=ev.mask_windows(scene["seconds"],scene["near_schedule"])
    scene["effective_near_dbfs"]=ev.db(ev.power(clean[mask,0]))
    scene["measured_input_snr_db"]=ev.db(ev.power(clean[mask,0])/ev.power(noise[mask,0])) if scene["noise"]!="none" else None
    scene["files"]={p.name:dict(sha256=ev.digest(p),bytes=p.stat().st_size) for p in sorted(folder.iterdir())
                    if p.is_file() and p.name!="scene.json"}
    legacy.save(folder/"scene.json",scene)


def early_diagnostics(scene,root,mode):
    folder=root/"cases"/scene["id"]
    prefix=folder/"runs"/mode/"result"
    engine=json.loads(prefix.with_suffix(".json").read_text())
    rate=scene["sample_rate"];delay=engine["dsp_delay_samples"]
    target=ev.read(folder/("clean-beam.wav" if scene["mics"]>1 else "clean.wav"))[:,0]
    before=ev.read(folder/"before.wav")[:,0]
    noise=ev.read(folder/"noise-component.wav")[:,0]
    echo=ev.read(folder/"echo-component.wav")[:,0]
    out=ev.read(prefix.with_suffix(".after.wav"))[:,0][delay:delay+len(before)]
    rows=[];hop=rate//100
    for i in range(len(target)//hop):
        t,b,a=(v[i*hop:(i+1)*hop] for v in (target,before,out))
        tp=ev.power(t)
        npow=ev.power(noise[i*hop:(i+1)*hop])
        projection=float(np.dot(t,a)/np.dot(t,t)) if tp>1e-15 else None
        rows.append(dict(frame=i,time_s=i/100,target_dbfs=ev.db(tp),input_dbfs=ev.db(ev.power(b)),
            noise_dbfs=ev.db(npow),echo_dbfs=ev.db(ev.power(echo[i*hop:(i+1)*hop])),
            input_snr_db=ev.db(tp/npow) if tp>1e-15 and npow>1e-15 else None,
            output_dbfs=ev.db(ev.power(a)),projection_gain_db=20*math.log10(abs(projection)) if projection is not None and abs(projection)>1e-15 else None,
            target_present=tp>1e-15,not_a_vad_label=True))
    (prefix.parent/"signal-frames.jsonl").write_text("".join(json.dumps(v)+"\n" for v in rows))
    segments={}
    onset=scene["near_schedule"][0][0]
    for name,lo,hi in (("0_50ms",0,.05),("50_150ms",.05,.15),("150_500ms",.15,.5)):
        sl=slice(round((onset+lo)*rate),round((onset+hi)*rate))
        t,b,a=(v[sl] for v in (target,before,out))
        tp=ev.power(t);den=float(np.dot(t,t))
        projection=float(np.dot(t,a)/den) if den>1e-15 else None
        segments[name]=dict(target_dbfs=ev.db(tp),input_dbfs=ev.db(ev.power(b)),output_dbfs=ev.db(ev.power(a)),
            projection_gain_db=20*math.log10(abs(projection)) if projection is not None and abs(projection)>1e-15 else None,
            nmse_db=ev.db(ev.power(a-t)/tp) if tp>1e-15 else None,
            available=tp>1e-15,scope="natural source phase; short-window diagnostics, no invented STOI or VAD labels")
    legacy.save(prefix.parent/"onset-diagnostics.json",segments)
    return segments


def command_for(scene,folder,prefix,cli,mode="full",paced=False):
    command=legacy.command_for(scene,folder,prefix,cli,mode,paced)
    if scene.get("steering_error_samples") and scene["mics"]>1:
        i=command.index("--beam-delays")+1
        delays=np.asarray(scene["beam_delays_samples"])+np.linspace(0,scene["steering_error_samples"],scene["mics"])
        command[i]=",".join(f"{d:.9f}" for d in delays)
    return command


def run_group(args):
    group,root,manifest,cli,cli_sha=args
    ev.RATE=group["sample_rate"]
    rows=[]
    try:
        if shutil.disk_usage(root).free < 50*1024**3:
            raise ValueError("disk reserve below 50 GiB; preserve completed evidence")
        speech,noises,used,noise_evidence=source_context(group,manifest)
        # The old evaluator is reused for unchanged acoustic/calibration topology only.
        legacy.source_pool=lambda _manifest,_rate:(speech,noises,used)
        def transport(playback,scene):
            if scene["stress"]=="clock_drift":
                return np.interp(np.arange(len(playback))*(1+scene["clock_ppm"]*1e-6),np.arange(len(playback)),playback,left=0,right=0)
            if scene["stress"]=="nonlinear_speaker":
                common=scene["common_gain"]
                return np.tanh(playback/common*6)/6*common
            return playback
        legacy.transport=transport
        prototype=None;clean=unit_noise=None;gain=None;stable_payloads={}
        for snr in group["grid"]:
            case=copy.deepcopy(group)
            case["id"]=group["id"]+(f"-snr{snr:+03d}" if snr is not None else "")
            case["snr_db"]=snr if snr is not None else group.get("snr_db")
            case.pop("grid")
            folder=root/"cases"/case["id"]
            if case["family"]=="ns" and not case.get("noise_steps"):
                if prototype is None:
                    prototype=ev.make_case(case,speech,noises,folder,{})
                    mask=ev.mask_windows(prototype["seconds"],prototype["near_schedule"])
                    raw_clean=ev.read(folder/"clean-mics.wav")
                    raw_noise=ev.read(folder/"noise-component.wav")
                    clean,unit_noise,gain=fixed_components(raw_clean,raw_noise,mask,group["grid"],case["near_dbfs"])
                else:
                    folder.mkdir(parents=True,exist_ok=False)
                scene=dict(copy.deepcopy(prototype),**case)
                scene.update(common_gain=gain,capture_headroom_gain=gain,reference_mode="no_playback",
                             source_speech_files={case["near_speaker"]:used[case["near_speaker"]]},geometry_scope="synthetic vehicle geometry; not measured SU7 RIR",
                             target_level_scope="identical target PCM and one shared headroom gain across this complete SNR sweep",
                             noise_source= noise_evidence.get(case["noise"]),onset_policy="unselected deterministic natural source phase",
                             steering_scope="oracle geometry" if not case.get("steering_error_samples") else "perturbed steering")
                write_components(folder,scene,clean,unit_noise*10**(-snr/20),np.zeros_like(clean),stable_payloads=stable_payloads)
            else:
                scene=legacy.prepare(case,root,manifest,str(cli),cli_sha)
                scene.update(noise_source=noise_evidence.get(case["noise"]),onset_policy="unselected deterministic natural source phase",
                             simulated_clock_time_warp=case["stress"]=="clock_drift",
                             steering_scope="perturbed steering" if case.get("steering_error_samples") else "oracle geometry")
                if case["family"] == "agc":
                    scene["source_speech_files"]={case["near_speaker"]:used[case["near_speaker"]]}
                legacy.save(folder/"scene.json",scene)
            scene["noise_rendering_scope"]="mono source plus simulated spatial field/geometry; not a recorded microphone array" if case["mics"]>1 else "native mono source or procedural model at the requested rate"
            legacy.save(folder/"scene.json",scene)
            checked=audit.check_scene(("controlled_native",folder,scene))
            # Quantization metadata tolerance and target-window problems are retained, never skipped.
            if checked["errors"]:
                raise ValueError(f"input quality check failed: {checked['errors']}")
            prefix=folder/"runs/full/result"
            legacy.invoke(command_for(scene,folder,prefix,str(cli)),prefix,cli_sha)
            result=legacy.measure(scene,root,"full")
            result["metrics"]["stoi_scope"]="standard STOI on natural source phase; unavailable is retained, no onset selection"
            result["onset_diagnostics"]=early_diagnostics(scene,root,"full")
            result["input_audit"]=checked
            legacy.save(folder/"complete.json",dict(scene=scene,metrics=result["metrics"],acceptance=result["acceptance"]))
            rows.append(result)
        if group["family"]=="ns":
            for name in ("clean.wav","clean-mics.wav","clean-beam.wav"):
                if len({r["scene"]["files"][name]["sha256"] for r in rows})!=1:
                    raise ValueError(f"target PCM drift inside paired sweep: {name}")
        return rows
    except Exception:
        return rows+[dict(id=group["id"],sample_rate=group["sample_rate"],error=traceback.format_exc(),
                          uncompleted_scenes=len(group["grid"])-len(rows))]


def validate_plan(groups,manifest):
    if not groups or len({g["id"] for g in groups})!=len(groups):
        raise ValueError("empty/nonunique groups")
    source=json.loads(manifest.read_text())
    entries=source["sources"]
    for group in groups:
        for speaker in (group["near_speaker"],group["far_speaker"]):
            match=[e for e in entries if e["sample_rate"]==group["sample_rate"] and e.get("canonical_speaker_id")==speaker]
            if not match or {e["split"] for e in match}!={group["split"]}:
                raise ValueError("speaker split contamination")
        if group["grid_kind"]=="one_db_fixed_target_sweep" and group["grid"]!=list(range(-10,31)):
            raise ValueError("nonlinear SNR grid")
    return dict(groups=len(groups),scenes=sum(len(g["grid"]) for g in groups),
                source_split_overlap=source["cross_split_overlaps"],
                near_speakers=len({g["near_speaker"] for g in groups}),
                by_rate=dict(Counter(str(g["sample_rate"]) for g in groups)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources",type=Path,required=True)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--cli",type=Path,required=True)
    parser.add_argument("--jobs",type=int,choices=range(1,9),default=8)
    parser.add_argument("--pilot",action="store_true")
    parser.add_argument("--plan-only",action="store_true")
    parser.add_argument("--paced",action="store_true")
    args=parser.parse_args()
    root,manifest=args.root.resolve(),args.sources.resolve()
    if ev.digest(legacy.acceptance.RULES)!=legacy.RULES_SHA:
        raise ValueError("frozen gates changed")
    groups=matrix(manifest)
    if args.pilot:
        selected=[]
        for rate in (16000,24000,48000):
            for mics in (1,2,4):
                for family in ("ns","combined"):
                    group=copy.deepcopy(next(g for g in groups if g["sample_rate"]==rate and g["mics"]==mics and g["family"]==family))
                    if family=="ns":group["grid"]=[-10,10,30];group["grid_kind"]="pilot_fixed_target_probe"
                    selected.append(group)
                for condition in ("wind", "clock", "noise_step"):
                    group=copy.deepcopy(next(g for g in groups if g["sample_rate"]==rate and g["mics"]==mics
                        and (g["noise"]=="sim_wind_gust" if condition=="wind" else
                             g.get("clock_ppm")==-300 if condition=="clock" else bool(g.get("noise_steps")))))
                    if condition=="wind":group["grid"]=[10]
                    selected.append(group)
        groups=selected
    plan_summary=validate_plan(groups,manifest)
    root.mkdir(parents=True,exist_ok=False)
    (root/"artifacts").mkdir()
    cli=root/"artifacts/itu_audio_cli"
    shutil.copy2(args.cli.resolve(),cli)
    cli_sha=ev.digest(cli)
    dependencies=[Path(__file__),Path(ev.__file__),Path(legacy.__file__),Path(audit.__file__),
                  Path(__file__).with_name("spatial_noise.py"),Path(legacy.acceptance.__file__),legacy.acceptance.RULES,manifest]
    for path in dependencies:
        shutil.copy2(path,root/"artifacts"/path.name)
    identity=dict(cli_sha256=cli_sha,source_manifest=str(manifest),rules_sha256=legacy.RULES_SHA,
                  dependencies={str(p.resolve()):ev.digest(p) for p in dependencies},
                  external_vad_enabled=False,sample_rate_conversion=False)
    legacy.save(root/"identity.json",identity)
    legacy.save(root/"plan.json",dict(groups=groups,summary=plan_summary))
    if args.plan_only:
        print(json.dumps(plan_summary));return
    started=time.monotonic();rows=[];errors=[]
    legacy.save(root/"report.json",dict(state="running",completed=0,total=plan_summary["scenes"],standard_compliance=False))
    with (root/"results.jsonl").open("x") as out,ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures=[pool.submit(run_group,(g,root,manifest,cli,cli_sha)) for g in groups]
        for future in as_completed(futures):
            for row in future.result():
                out.write(json.dumps(row,allow_nan=False)+"\n")
                rows.append(row)
                if "error" in row:errors.append(row)
            out.flush()
            report=dict(state="running",completed=sum("scene" in r for r in rows),total=plan_summary["scenes"],
                        elapsed_s=round(time.monotonic()-started,2),summary=legacy.summary(rows),standard_compliance=False,
                        execution_errors=len(errors))
            legacy.save(root/"report.json",report)
            print(json.dumps({k:v for k,v in report.items() if k!="summary"}),flush=True)
    controls=[];paced=[]
    for rate in (16000,24000,48000):
        for family,mode,segment in (("ns","negative_no_ns","noise_only"),("aec","negative_no_aec","far_only")):
            selected=next((r for r in rows if "scene" in r and r["scene"]["sample_rate"]==rate
                           and r["scene"]["mics"]==1 and r["scene"]["family"]==family),None)
            if selected is None:continue
            scene=selected["scene"];folder=root/"cases"/scene["id"];prefix=folder/"runs"/mode/"result"
            legacy.invoke(command_for(scene,folder,prefix,str(cli),mode),prefix,cli_sha)
            result=legacy.measure(scene,root,mode)
            gate=next(g for g in result["acceptance"]["gates"] if g["segment"]==segment and g["metric"]=="total_attenuation_db")
            controls.append(dict(id=scene["id"],mode=mode,rate=rate,verified=gate["status"]=="fail",value=gate["value"],threshold=gate["threshold"]))
        if args.paced:
            for mics in (1,2,4):
                row=next((r for r in rows if "scene" in r and r["scene"]["sample_rate"]==rate
                          and r["scene"]["mics"]==mics and r["scene"]["family"]=="combined"),None)
                if row is None:continue
                scene=row["scene"];folder=root/"cases"/scene["id"];prefix=folder/"runs/paced/result"
                legacy.invoke(command_for(scene,folder,prefix,str(cli),paced=True),prefix,cli_sha)
                result=legacy.measure(scene,root,"paced")
                paced.append(dict(id=scene["id"],sample_rate=rate,mics=mics,
                    output_matches_offline=result["metrics"]["output_sha256"]==row["metrics"]["output_sha256"],
                    runtime_gates=[g for g in result["acceptance"]["gates"] if g["segment"]=="runtime"]))
    changed=[str(p) for p in dependencies if ev.digest(p)!=identity["dependencies"][str(p.resolve())]]
    if ev.digest(cli)!=cli_sha:changed.append(str(cli))
    expected_controls=sum(any(g["sample_rate"]==rate and g["family"]==family and g["mics"]==1 for g in groups)
                          for rate in (16000,24000,48000) for family in ("ns","aec"))
    controls_verified=len(controls)==expected_controls and all(c["verified"] for c in controls)
    paced_verified=not args.paced or (len(paced)==9 and all(p["output_matches_offline"] and
                    all(g["status"]=="pass" for g in p["runtime_gates"]) for p in paced))
    report.update(state="completed_with_errors" if errors or changed else "completed" if controls_verified and paced_verified else "completed_with_verification_failures",
                  completed=sum("scene" in r for r in rows),elapsed_s=round(time.monotonic()-started,2),
                  summary=legacy.summary(rows),negative_controls=controls,paced_checks=paced,
                  controls_verified=controls_verified,paced_verified=paced_verified,
                  identity_verified=not changed,changed_dependencies=changed,plan=plan_summary,
                  created_utc=datetime.now(timezone.utc).isoformat())
    legacy.save(root/"report.json",report)
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    main()
