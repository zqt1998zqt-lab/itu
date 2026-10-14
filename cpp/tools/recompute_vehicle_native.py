#!/usr/bin/env python3
"""Recompute physical PCM gates without changing prior reports or rerunning DSP.

Speech quality is independently recomputed against frozen source targets.
Suppression, PCM16 censoring and output levels use decoded native-rate PCM.
The independent destination keeps measurement fixes separate from DSP.
"""
import argparse
import csv
import copy
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import math

import check_audio_acceptance as ac
import run_audio_eval as ev
from evaluation_targets import calibrated_beam_target
from run_vehicle_stream import select_timeline, segment_timing


def levels(signal):
    return dict(peak_dbfs=ev.db(float(np.max(np.abs(signal)))**2),
                clipping_percent=float(np.mean(np.abs(signal)>=.999)*100))


def passed(verdict):
    return all(g["status"] == "pass" or g["optional"] and g["status"] == "unavailable"
               for g in verdict["gates"] if g["segment"] != "runtime")


def update(row, before, after, rate, bounds):
    result = copy.deepcopy(row)
    for name, window in bounds.items():
        if not name.startswith("near") and name != "double_talk":
            result["segments"][name].update(ev.attenuation_measurement(
                ev.window(before, window, rate), ev.window(after, window, rate), pcm16=True))
    return result


def speech(row, before, target, baseline_target, after, windows):
    for name, bounds in windows.items():
        if name.startswith("near") or name == "double_talk":
            qb = ev.quality(ev.window(baseline_target,bounds),ev.window(before,bounds))
            qa = ev.quality(ev.window(target,bounds),ev.window(after,bounds))
            row["segments"][name] = dict(before=qb,after=qa,stoi_improvement=
                qa["stoi"]-qb["stoi"] if qa["stoi"] is not None and qb["stoi"] is not None else None)


def source_signal(scene, name, dataset):
    folder = Path(scene.get("source_dataset",dataset))/"cases"/scene.get("source_case_id",scene["id"])
    if ev.digest(folder/name) != scene["files"][name]["sha256"]:
        raise ValueError("source target identity changed")
    return ev.read(folder/name)


def source_targets(identity, pair, dataset):
    selections = select_timeline([s["scene"] for s in identity["segments"]],pair,identity["microphones"])
    before,baseline,target,array = [],[],[],[]
    for (label,scene,lo,hi),segment in zip(selections,identity["segments"]):
        if label != segment["label"] or scene["id"] != segment["source_id"]:
            raise ValueError("frozen timeline selection changed")
        cut = slice(round(lo*16000),round(hi*16000))
        before.append(source_signal(scene,"before.wav",dataset)[cut,0])
        baseline.append(source_signal(scene,"clean.wav",dataset)[cut,0])
        if identity["microphones"]>1:
            array.append(source_signal(scene,"clean-mics.wav",dataset)[cut])
        else:
            target.append(baseline[-1])
    if array:
        target,_ = calibrated_beam_target(np.concatenate(array),identity["beam_delays_samples"],
            identity["sample_rate"],core_rate=identity["target_alignment"]["core_rate"])
    else:
        target=np.concatenate(target)
    return np.concatenate(before),target,np.concatenate(baseline)


def pcm(folder, prefix, engine):
    before, br = sf.read(folder/"before.wav", dtype="float64", always_2d=True)
    output_path = prefix.with_suffix(".after.wav")
    output, rate = sf.read(output_path, dtype="float64", always_2d=True)
    delay = engine["dsp_delay_samples"]
    if (br != rate or rate != engine["sample_rate"] or output.shape[1] != 1 or
            len(output) != len(before)+delay or not np.isfinite(output).all()):
        raise ValueError("native PCM contract mismatch")
    if sf.info(output_path).subtype != "PCM_16":
        raise ValueError("PCM16 bound requires actual PCM16 output")
    hashes = {str(p): ev.digest(p) for p in (folder/"before.wav", output_path,
                                          prefix.with_suffix(".json"), prefix.with_suffix(".frames.csv"))}
    return before[:, 0], output[delay:, 0], output[:, 0], rate, hashes


def erle_signals(prefix, engine, io_length):
    if not engine["linear_aec_available"]:
        return None,{}
    lp,cp=prefix.with_suffix(".linear-aec.wav"),prefix.with_suffix(".aec-input.wav")
    linear,lr=sf.read(lp,dtype="float64",always_2d=True)
    capture,cr=sf.read(cp,dtype="float64",always_2d=True)
    if lr!=16000 or cr!=engine["sample_rate"]:
        raise ValueError("ERLE tap rate mismatch")
    count=io_length*16000//cr
    ld,cd=engine["linear_aec_delay_samples"],engine["aec_input_delay_samples"]
    linear=linear[ld:ld+count,0];capture=capture[cd:cd+io_length,0]
    gcd=math.gcd(cr,16000)
    if cr!=16000: capture=resample_poly(capture,16000//gcd,cr//gcd)
    if len(linear)!=count or len(capture)!=count:
        raise ValueError("ERLE tap length mismatch")
    return (capture,linear),{str(p):ev.digest(p) for p in (lp,cp)}


def stream_rows(root, rules, dataset):
    rows = []
    paths = [root/"stream-evaluation.json"] if (root/"stream-evaluation.json").is_file() else sorted(root.glob("r*/pair-*/stream-evaluation.json"))
    if not paths:
        raise ValueError("nonempty continuous inventory required")
    for path in paths:
        original = json.loads(path.read_text())
        folder = path.parent
        identity = original["identity"]
        if ev.digest(dataset/"inputs-manifest.json") != identity["source_manifest_sha256"]:
            raise ValueError("stream manifest identity mismatch")
        if identity["rules_sha256"] != ac.RULES_SHA256:
            raise ValueError("stream thresholds changed")
        for name, expected in identity["files"].items():
            if ev.digest(folder/name) != expected:
                raise ValueError("stream input changed")
        for name,expected in identity.get("output_files",{}).items():
            if ev.digest(folder/name)!=expected:
                raise ValueError("stream output identity changed")
        invocation = json.loads((folder/"invocation.json").read_text())
        if invocation["exit_code"] or invocation["cli_sha256"] != identity["cli_sha256"]:
            raise ValueError("stream invocation identity mismatch")
        binary = Path(invocation["argv"][0])
        if not binary.is_file() or ev.digest(binary) != identity["cli_sha256"]:
            raise ValueError("stream binary identity mismatch")
        before, after, _, rate, hashes = pcm(folder, folder/"stream", original["engine"])
        clean_before,target,baseline_target = source_targets(identity,identity["segments"][0]["scene"]["pair"],dataset)
        erle,tap_hashes=erle_signals(folder/"stream",original["engine"],len(before))
        hashes.update(tap_hashes)
        gcd=math.gcd(rate,16000)
        quality_after=resample_poly(after,16000//gcd,rate//gcd) if rate!=16000 else after
        if len(quality_after)!=len(target):
            raise ValueError("independent quality target length mismatch")
        with (folder/"stream.frames.csv").open() as file:
            frames=list(csv.DictReader(file))
        segment_map = {s["label"]: s for s in identity["segments"]}
        detail_map = {d["label"]: d for d in original["details"]}
        for item in original["measurements"]:
            segment = segment_map[item["label"]]
            row = update(item["row"], before, after, rate, segment["windows"])
            speech(row,clean_before,target,baseline_target,quality_after,segment["windows"])
            if segment["scene"]["noise"]=="none":
                for name,bounds in segment["windows"].items():
                    if name.startswith("far") and erle is not None:
                        p,q=ev.power(ev.window(erle[0],bounds)),ev.power(ev.window(erle[1],bounds))
                        row["segments"][name]["linear_erle_engineering_db"]=ev.db(p/q) if p>1e-12 and q>1e-12 else None
            selected,timing,over=segment_timing(frames,segment["start_s"],segment["end_s"],original["engine"]["paced_10ms"])
            row["engine"].update(timing)
            row["engine"]["reference_unreliable_frames"]=sum(int(r["reference_state"])==2 for r in selected)
            row["process_frames_over_10ms"]=over
            row["output_levels"] = levels(ev.window(after, [segment["start_s"], segment["end_s"]], rate))
            verdict = ac.check_scene(dict(segment["scene"], windows=segment["windows"]), row, rules)
            rows.append(dict(key=[rate, identity["microphones"], folder.name, item["label"],folder.parent.name],
                original_quality_passed=passed(detail_map[item["label"]]), quality_passed=passed(verdict),
                acceptance=verdict, measurements=row, source_report=str(path),
                source_report_sha256=ev.digest(path), pcm_sha256=hashes,
                cli_sha256=identity["cli_sha256"], native_highband=False))
    return rows


def field_rows(path, rules, scenes, dataset):
    original = json.loads(path.read_text())
    if original["identity"]["rules_sha256"] != ac.RULES_SHA256:
        raise ValueError("field thresholds changed")
    binary = path.parent/"itu_audio_cli"
    if ev.digest(binary) != original["identity"]["cli_sha256"]:
        raise ValueError("field binary identity mismatch")
    rows = []
    for item in original["rows"]:
        if item["variant"] != "spatial":
            continue
        scene = dict(scenes[item["id"]])
        source = Path(scene.get("source_dataset", dataset))/"cases"/scene.get("source_case_id",scene["id"])
        for name, record in scene["files"].items():
            if ev.digest(source/name) != record["sha256"]:
                raise ValueError("field source input changed")
        windows = dict(scene["windows"])
        if "cold_start" in windows:
            windows["near_cold_start"] = windows.pop("cold_start")
        run = path.parent/f"{item['id']}-r{item['rate']}"/item["variant"]
        invocation = json.loads((run/"invocation.json").read_text())
        if invocation["exit_code"]:
            raise ValueError("field invocation failed")
        prefix = run/"result"
        for name,expected in item.get("pcm_sha256",{}).items():
            actual = run.parent/name if name in ("before.wav","ref.wav") else run/name
            if ev.digest(actual)!=expected:
                raise ValueError("field run PCM changed")
        before, after, output, rate, hashes = pcm(run.parent, prefix, item["measurements"]["engine"])
        row = update(item["measurements"], before, after, rate, windows)
        clean_before=source_signal(scene,"before.wav",dataset)[:,0]
        baseline_target=source_signal(scene,"clean.wav",dataset)[:,0]
        core_rate=row["engine"]["internal_sample_rate"]
        target,_=calibrated_beam_target(source_signal(scene,"clean-mics.wav",dataset),
            scene["beam_delays_samples"],rate,core_rate=core_rate)
        gcd=math.gcd(rate,16000)
        quality_after=resample_poly(after,16000//gcd,rate//gcd) if rate!=16000 else after
        speech(row,clean_before,target,baseline_target,quality_after,windows)
        with prefix.with_suffix(".frames.csv").open() as file:
            frames=list(csv.DictReader(file))
        timing,over=ev.frame_timing(frames,math.ceil(len(before)/(rate//100)))
        row["engine"].update(timing)
        row["process_frames_over_10ms"]=over
        row["output_levels"] = levels(output)
        scene["windows"] = windows
        verdict = ac.check_scene(scene, row, rules)
        rows.append(dict(key=[item["id"], rate],original_quality_passed=item["quality_passed"],
            quality_passed=passed(verdict),acceptance=verdict,measurements=row,
            source_report=str(path),source_report_sha256=ev.digest(path),pcm_sha256=hashes,
            cli_sha256=original["identity"]["cli_sha256"],native_highband=False))
    if not rows:
        raise ValueError("nonempty spatial field inventory required")
    return rows


def main():
    code_paths=[Path(__file__),Path(ev.__file__),Path(ac.__file__),
                Path(__file__).with_name("evaluation_targets.py"),Path(__file__).with_name("run_vehicle_stream.py")]
    code_hashes={str(p):ev.digest(p) for p in code_paths}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stream", type=Path, action="append", default=[])
    parser.add_argument("--stream-dataset", type=Path)
    parser.add_argument("--field-report", type=Path, action="append", default=[])
    parser.add_argument("--field-dataset", type=Path)
    args = parser.parse_args()
    if ev.digest(ac.RULES) != ac.RULES_SHA256:
        raise ValueError("frozen thresholds changed")
    scenes = {}
    if args.stream and args.stream_dataset is None:
        parser.error("stream remeasurement requires the frozen dataset")
    if args.field_report:
        if args.field_dataset is None:
            parser.error("field remeasurement requires the frozen dataset")
        manifest_path = args.field_dataset/"inputs-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for source in args.field_report:
            if json.loads(source.read_text())["identity"]["manifest_sha256"] != ev.digest(manifest_path):
                raise ValueError("field manifest identity mismatch")
        scenes = {s["id"]: s for s in manifest["scenes"]}
    args.root.mkdir(parents=True, exist_ok=False)
    rules = json.loads(ac.RULES.read_text())
    batches = []
    for source, kind in [(p, "stream") for p in args.stream]+[(p, "field") for p in args.field_report]:
        rows = stream_rows(source, rules, args.stream_dataset) if kind == "stream" else field_rows(source, rules, scenes, args.field_dataset)
        # Detect mutation while remeasurement was in progress.
        snapshots = {p: digest for row in rows for p, digest in row["pcm_sha256"].items()}
        for path, expected in snapshots.items():
            if ev.digest(Path(path)) != expected:
                raise ValueError("PCM/report changed during remeasurement")
        batches.append(dict(source=str(source),kind=kind,rows=rows,total=len(rows),
            quality_passed=sum(r["quality_passed"] for r in rows),
            measurement_repairs=sum(not r["original_quality_passed"] and r["quality_passed"] for r in rows),
            measurement_regressions=sum(r["original_quality_passed"] and not r["quality_passed"] for r in rows)))
    if not batches:
        parser.error("nonempty stream or field sources required")
    if any(ev.digest(Path(p))!=sha for p,sha in code_hashes.items()):
        raise ValueError("measurement source changed during remeasurement")
    result = dict(rules_sha256=ac.RULES_SHA256,runner_sha256=ev.digest(Path(__file__)),
        measurement_code_sha256=code_hashes,
        scope="native suppression/censoring/levels and frozen-source speech quality independently recomputed; runtime from actual frame CSV; no DSP rerun or lag fit",
        standard_compliance=False,batches=batches)
    (args.root/"report.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps([{k:v for k,v in b.items() if k!="rows"} for b in batches],indent=2))


if __name__ == "__main__":
    main()
