#!/usr/bin/env python3
"""Execute frozen, paired BF→AEC→NS→EQ→AGC→DRC chains, never add module scores."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np
import soundfile as sf

from evaluation_targets import calibrated_beam_target
from run_bf_algorithm_matrix import digest, quality, read_rate
from run_audio_eval import attenuation_measurement


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "src/features/enhancement-chain-configs.json"
DATASET = ROOT / "datasets/itu-vehicle-fields-float-20261010"
MANIFEST = DATASET / "inputs-manifest.json"
GATES = {
    "speech_sisdr_delta_min_db": -0.1,
    "speech_stoi_delta_min": -0.02,
    "noise_attenuation_delta_min_db": 0.0,
    "far_attenuation_delta_min_db": 6.0,
    "projection_gain_min_db": -6.0,
    "projection_gain_max_db": 18.0,
    "peak_max_dbfs": -0.1,
    "clipping_max_percent": 0.0,
}


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def flags_for(config, preset):
    flags = ["--profile", preset["profile"], "--no-rx-processing"]
    for stage in config["stages"]:
        choice = next(x for x in stage["options"] if x["id"] == preset["selection"][stage["id"]])
        flags.extend(choice["flags"])
    return flags + preset.get("extra_flags", [])


def attenuation(before, after):
    result = attenuation_measurement(before, after, pcm16=True)
    if result["before_dbfs"] <= -120:
        raise ValueError("measurement window has no input excitation")
    exact = result["total_attenuation_db"]
    value = exact if exact is not None else result["attenuation_lower_bound_db"]
    if value is None:
        raise ValueError("attenuation is neither measurable nor bounded")
    return float(value), exact is None


def gate(measurements, baseline):
    failures = []
    speech = {}
    for window in measurements["speech_windows"]:
        actual = measurements["speech_windows"][window]
        control = baseline["speech_windows"][window]
        values = [actual["sisdr_db"], control["sisdr_db"], actual["stoi"], control["stoi"]]
        if any(v is None or not np.isfinite(v) for v in values):
            failures.append(f"{window}:invalid_speech")
            speech[window] = {"sisdr_delta_db": None, "stoi_delta": None}
            continue
        sd, st = actual["sisdr_db"] - control["sisdr_db"], actual["stoi"] - control["stoi"]
        speech[window] = {"sisdr_delta_db": sd, "stoi_delta": st}
        if sd < GATES["speech_sisdr_delta_min_db"]:
            failures.append(f"{window}:sisdr")
        if st < GATES["speech_stoi_delta_min"]:
            failures.append(f"{window}:stoi")
        gain = actual["projection_gain_db"]
        if gain is None or not GATES["projection_gain_min_db"] <= gain <= GATES["projection_gain_max_db"]:
            failures.append(f"{window}:gain")
    noise_delta = measurements["noise_attenuation_db"] - baseline["noise_attenuation_db"]
    if noise_delta < GATES["noise_attenuation_delta_min_db"]:
        failures.append("noise_attenuation")
    far_delta = None
    if measurements["far_attenuation_db"] is not None:
        far_delta = measurements["far_attenuation_db"] - baseline["far_attenuation_db"]
        if far_delta < GATES["far_attenuation_delta_min_db"]:
            failures.append("far_attenuation")
    if measurements["peak_dbfs"] > GATES["peak_max_dbfs"]:
        failures.append("peak")
    if measurements["clipping_percent"] > GATES["clipping_max_percent"]:
        failures.append("clipping")
    return {
        "pass": not failures,
        "failures": failures,
        "speech": speech,
        "noise_delta_db": noise_delta,
        "far_delta_db": far_delta,
    }


def measure(scene, target, before, output):
    rate = scene["sample_rate"]
    primary = "near_late" if "near_late" in scene["windows"] else "near_only"
    names = [primary] + (["double_talk"] if "double_talk" in scene["windows"] else [])
    windows = scene["windows"]

    def cut(data, name):
        start, end = [round(t * rate) for t in windows[name]]
        return data[start:end]

    noise, noise_bound = attenuation(cut(before, "noise_only"), cut(output, "noise_only"))
    far, far_bound = attenuation(cut(before, "far_only"), cut(output, "far_only")) if "far_only" in windows else (None, False)
    return {
        "primary_window": primary,
        "speech_windows": {name: quality(cut(target, name), cut(output, name), rate) for name in names},
        "noise_attenuation_db": noise, "noise_attenuation_lower_bound": noise_bound,
        "far_attenuation_db": far, "far_attenuation_lower_bound": far_bound,
        "peak_dbfs": float(20 * np.log10(max(float(np.max(np.abs(output))), 1e-20))),
        "clipping_percent": float(np.mean(np.abs(output) >= .999) * 100),
    }


def process_scene(task):
    root, scene, config, cli = task
    source, rate = DATASET / "cases" / scene["id"], scene["sample_rate"]
    before = read_rate(source / "before.wav", rate)[:, 0]
    clean = read_rate(source / "clean-mics.wav", rate)
    target, alignment = calibrated_beam_target(clean, scene["beam_delays_samples"], rate, core_rate=rate)
    baseline, records = None, []
    for preset in config["presets"]:
        prefix = root / preset["id"] / "cases" / scene["id"] / "runs/full/result"
        prefix.parent.mkdir(parents=True, exist_ok=False)
        flags = flags_for(config, preset)
        command = [str(cli)] + flags + [
            "--mic", str(source / "before.wav"), "--echo", str(source / "ref.wav"),
            "--beam-delays", ",".join(f"{d:.9f}" for d in scene["beam_delays_samples"]),
            "--out", str(prefix),
        ]
        if "far_only" in scene["windows"] and preset["selection"]["aec"] != "off":
            command += ["--far-start", str(scene["windows"]["far_only"][0]), "--far-end", str(scene["windows"]["far_only"][1])]
        started = time.monotonic()
        completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
        (prefix.parent / "invocation.json").write_text(json.dumps({
            "argv": command, "exit_code": completed.returncode, "stdout": completed.stdout,
            "stderr": completed.stderr, "wall_seconds": time.monotonic() - started,
        }, ensure_ascii=False, indent=2) + "\n")
        completed.check_returncode()
        engine = json.loads(prefix.with_suffix(".json").read_text())
        delay = int(engine["dsp_delay_samples"])
        all_output = read_rate(prefix.with_suffix(".after.wav"), rate)[:, 0]
        if sf.info(prefix.with_suffix(".after.wav")).subtype != "PCM_16":
            raise ValueError("quantization bound assumes the actual CLI PCM_16 output")
        if len(all_output) != len(before) + delay or not np.isfinite(all_output).all():
            raise ValueError(f"invalid output timeline: {scene['id']}/{preset['id']}")
        if not engine["beamforming_before_aec"] or (not engine["eq_before_agc"] and preset["profile"] == "vcp"):
            raise ValueError("executed chain topology differs from the declared graph")
        output = all_output[delay:delay + len(before)]
        measured = measure(scene, target, before, output)
        measured["peak_dbfs"] = float(20 * np.log10(max(float(np.max(np.abs(all_output))), 1e-20)))
        measured["clipping_percent"] = float(np.mean(np.abs(all_output) >= .999) * 100)
        record = {
            "variant": preset["id"], "scene_id": scene["id"], "mics": scene["mics"],
            "field": scene["field_label"], "snr_db": scene["snr_db"], "sample_rate": rate,
            "measurements": measured, "output_file": relative(prefix.with_suffix(".after.wav")),
            "before_file": relative(source / "before.wav"), "output_sha256": digest(prefix.with_suffix(".after.wav")),
            "delay_samples": delay, "target_alignment": alignment,
            "process_rtf": engine["process_rtf"], "limited_samples": engine["limited_samples_in_input_frames"],
        }
        if baseline is None:
            if preset["id"] != "array_control":
                raise ValueError("DAS paired control must execute first")
            baseline = record
            if measured["noise_attenuation_lower_bound"] or measured["far_attenuation_lower_bound"]:
                raise ValueError("paired DAS control must have measurable, uncensored noise/echo power")
        record["gate"] = gate(measured, baseline["measurements"])
        record["output_changed"] = record["output_sha256"] != baseline["output_sha256"]
        records.append(record)
    mute = measure(scene, target, before, np.zeros_like(before))
    mute_gate = gate(mute, baseline["measurements"])
    if mute_gate["pass"]:
        raise ValueError("known-bad mute passed the engineering gate")
    return records, {"scene_id": scene["id"], "rejected": True, "failures": mute_gate["failures"]}


def summary(rows, preset, scenes):
    primary_delta = [r["gate"]["speech"][r["measurements"]["primary_window"]] for r in rows]
    dt = [r["gate"]["speech"]["double_talk"] for r in rows if "double_talk" in r["gate"]["speech"]]
    far = [r["gate"]["far_delta_db"] for r in rows if r["gate"]["far_delta_db"] is not None]
    case_groups = []
    for passed, status, group_id in [(True, "通过", "pass"), (False, "失败", "fail")]:
        candidates = sorted([r for r in rows if r["gate"]["pass"] == passed], key=lambda r: r["gate"]["speech"][r["measurements"]["primary_window"]]["sisdr_delta_db"])
        if not candidates:
            continue
        row = candidates[len(candidates) // 2]
        scene = scenes[row["scene_id"]]
        window = row["measurements"]["primary_window"]
        if not passed:
            failed_speech = [name for name in row["measurements"]["speech_windows"] if any(reason.startswith(name + ":") for reason in row["gate"]["failures"])]
            window = failed_speech[0] if failed_speech else "noise_only" if "noise_attenuation" in row["gate"]["failures"] else "far_only" if "far_attenuation" in row["gate"]["failures"] else window
        if window in row["gate"]["speech"]:
            delta = row["gate"]["speech"][window]
            headline = f"ΔSI-SDR {delta['sisdr_delta_db']:+.3f} dB · ΔSTOI {delta['stoi_delta']:+.4f}"
        else:
            attenuation_delta = row["gate"]["noise_delta_db" if window == "noise_only" else "far_delta_db"]
            headline = f"{'纯噪声' if window == 'noise_only' else '远端单讲'}衰减增量 {attenuation_delta:+.3f} dB"
        asset = f"chain-{preset['id']}-{group_id}"
        case_groups.append({
            "id": group_id, "status": status,
            "headline": headline,
            "note": "所有适用工程门限通过。" if passed else "失败项：" + ", ".join(row["gate"]["failures"]),
            "gate": row["gate"],
            "caseAudio": {
                "id": row["scene_id"], "detail": f"{row['mics']} 麦 · {row['field']} · SNR {row['snr_db']} dB · {window}",
                "sampleRate": row["sample_rate"], "windowSeconds": scene["windows"][window],
                "delaySamples": row["delay_samples"], "beforeLabel": "原始麦克风 · 通道 0", "afterLabel": preset["label"] + " · TX 输出",
                "beforeFile": row["before_file"], "afterFile": row["output_file"],
                "waveformSrc": f"/evaluation/pipeline-cases/{asset}-waveform.svg",
                "melSrc": f"/evaluation/pipeline-cases/{asset}-mel.svg",
                "analysis": "图为原始通道 0 与整链输出；分数相对同案例 DAS 对照，非图中两路直接相减。",
            },
        })
    return {
        "id": preset["id"], "label": preset["label"], "description": preset["description"],
        "profile": preset["profile"], "selection": preset["selection"],
        "cases": len(rows), "pass": sum(r["gate"]["pass"] for r in rows),
        "changed": sum(r["output_changed"] for r in rows),
        "mean_sisdr_delta_db": float(np.mean([d["sisdr_delta_db"] for d in primary_delta])),
        "mean_stoi_delta": float(np.mean([d["stoi_delta"] for d in primary_delta])),
        "mean_noise_delta_db": float(np.mean([r["gate"]["noise_delta_db"] for r in rows])),
        "noise_censored_cases": sum(r["measurements"]["noise_attenuation_lower_bound"] for r in rows),
        "mean_double_sisdr_delta_db": float(np.mean([d["sisdr_delta_db"] for d in dt])) if dt else None,
        "mean_double_stoi_delta": float(np.mean([d["stoi_delta"] for d in dt])) if dt else None,
        "double_cases": len(dt), "far_cases": len(far),
        "mean_far_delta_db": float(np.mean(far)) if far else None,
        "far_censored_cases": sum(r["measurements"]["far_attenuation_lower_bound"] for r in rows),
        "clipped_cases": sum(r["measurements"]["clipping_percent"] > 0 for r in rows),
        "limited_samples": sum(r["limited_samples"] for r in rows),
        "mean_rtf": float(np.mean([r["process_rtf"] for r in rows])),
        "failure_counts": dict(Counter(reason for row in rows for reason in row["gate"]["failures"])),
        "caseGroups": case_groups,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--cli", type=Path, default=ROOT / "build/host/cpp/itu_audio_cli")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--scene", action="append", help="pilot only; omitted for the complete 72-case publication")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    artifacts = root / "artifacts"
    artifacts.mkdir()
    for source in [args.cli.resolve(), CONFIG, Path(__file__).resolve(), MANIFEST]:
        shutil.copy2(source, artifacts / source.name)
    cli = artifacts / "itu_audio_cli"
    config = json.loads((artifacts / CONFIG.name).read_text())
    manifest = json.loads(MANIFEST.read_text())
    scenes = [s for s in manifest["scenes"] if s["mics"] > 1 and (not args.scene or s["id"] in args.scene)]
    if not scenes or len({s["id"] for s in scenes}) != len(scenes) or not args.scene and len(scenes) != 72:
        raise ValueError("expected nonempty, unique, complete paired scenes")
    started = time.monotonic()
    # Validate every frozen input before scheduling any engine run.
    for scene in scenes:
        for name, entry in scene["files"].items():
            if digest(DATASET / "cases" / scene["id"] / name) != entry["sha256"]:
                raise ValueError(f"frozen input hash changed: {scene['id']}/{name}")
    rows, controls = [], []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool, (root / "results.jsonl").open("x") as stream:
        tasks = [pool.submit(process_scene, (root, scene, config, cli)) for scene in scenes]
        for count, task in enumerate(as_completed(tasks), 1):
            records, control = task.result()
            rows.extend(records)
            controls.append(control)
            for row in records:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            print(f"scenes {count}/{len(scenes)} · CLI runs {len(rows)}/{len(scenes) * len(config['presets'])}", flush=True)
    by_scene = {s["id"]: s for s in scenes}
    variants = []
    for preset in config["presets"]:
        selected = [r for r in rows if r["variant"] == preset["id"]]
        if {r["scene_id"] for r in selected} != set(by_scene) or len(selected) != len(scenes):
            raise ValueError(f"incomplete paired variant: {preset['id']}")
        item = summary(selected, preset, by_scene)
        item["flags"] = flags_for(config, preset)
        variants.append(item)
    report = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "source": relative(root / "report.json"), "case_count": len(scenes), "cli_runs": len(rows),
        "scope": "16 kHz · 2/4 microphones · 2 speech pairs · 0/15/30 dB SNR · 6 simulated field families",
        "target": "All routes use the same calibrated clean delay-sum target; declared DSP delay only, no lag fitting.",
        "rx_scope": "TX replay uses the frozen actual-playback PCM as --echo; RX is not reprocessed and no new acoustic echo is synthesized.",
        "standard_compliance": False, "gates": GATES,
        "attenuation_scope": "CLI output is PCM_16; sub-resolution output uses the half-LSB RMS upper bound from attenuation_measurement. Mixed means are conservative lower bounds if any case is censored.",
        "host": {"architecture": platform.machine(), "workers": args.jobs, "paced": False},
        "cli_sha256": digest(cli), "config_sha256": digest(artifacts / CONFIG.name),
        "results_sha256": digest(root / "results.jsonl"), "manifest_sha256": digest(MANIFEST),
        "negative_control": {"kind": "muted output, scored with the identical gates", "cases": len(controls), "rejected": sum(c["rejected"] for c in controls)},
        "variants": variants, "wall_seconds": time.monotonic() - started,
    }
    for path in [root / "report.json", root / "negative-controls.json"]:
        path.write_text(json.dumps(report if path.name == "report.json" else controls, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"variants": [{"id": v["id"], "pass": v["pass"], "cases": v["cases"]} for v in variants], "wall_seconds": report["wall_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
