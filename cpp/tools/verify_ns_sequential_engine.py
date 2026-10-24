#!/usr/bin/env python3
"""Compare scored component PCM with the actual general/VCP CLI NS path."""
import argparse
import csv
import json
from pathlib import Path
import subprocess

import numpy as np
import soundfile as sf

from audit_traditional_ns_residual import read_rows, sha
from diagnose_traditional_ns_remaining import P, token


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True,
                        help="Complete public configurations; replay names/options must match quality identity")
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    identity = json.loads((root / "identity.json").read_text())
    rows = read_rows(root / "results.jsonl")
    if json.loads((root / "report.json").read_text())["state"] != "completed":
        raise ValueError("quality regression is incomplete")
    profiles = json.loads(args.profiles.read_text())
    if ([(p["name"],p["variant"],p["options"]) for p in profiles]
            != [(p["name"],p["variant"],p["options"]) for p in identity["plan"]]):
        raise ValueError("public profile changed the frozen treatments")
    results = []
    binary = None
    for profile in profiles:
        group = [r for r in rows if r["profile"] == profile["name"]]
        ordered = sorted((r for r in group if r["noise_class"] == "steady"),
                         key=lambda r: (r["snr_db"], r["run"], r["id"]))
        chosen = [ordered[0], ordered[-1], next(r for r in group if r["id"].startswith("noise-step"))]
        c = profile["vehicle_ns"]
        for row in chosen:
            folder = out / profile["name"] / token(row)
            folder.mkdir(parents=True, exist_ok=False)
            source = P / "build" / row["run"] / "cases" / row["id"] / "before.wav"
            x, rate = sf.read(source)
            # C++ quantizes the FLOAT input once, with lround's half-away rule.
            x = x.astype(np.float32).astype(np.float64)
            q = np.copysign(np.floor(np.abs(x)*32768 + .5), x)
            q = np.clip(q,-32768,32767)/32768
            entry = folder / "input-pcm16.wav"
            sf.write(entry,q,rate,subtype="PCM_16")
            reference_path = root / row["profile"] / token(row) / "result.after.wav"
            if sha(reference_path) != row["output_sha256"]:
                raise ValueError("scored PCM drift")
            reference, ref_rate = sf.read(reference_path)
            invocation = json.loads((reference_path.parent / "invocation.json").read_text())
            binary = Path(invocation["argv"][0]).with_name("itu_audio_cli")
            for engine_profile in ("general", "vcp"):
                prefix = folder / engine_profile
                command = [str(binary),"--profile",engine_profile,"--no-aec","--no-agc",
                    "--no-tx-drc","--no-limiter","--no-low-cut","--no-external-vad",
                    "--vehicle-ns-control","--mic",str(entry),"--out",str(prefix),
                    "--ns-backend",c["backend"]]
                for field, flag in (("mild_strength","mild-strength"),("repair_mode","repair-mode"),
                        ("policy","policy"),("dry_max","dry-max"),("snr_threshold_db","snr-threshold"),
                        ("noise_gain","noise-gain"),("startup_ms","startup-ms"),("slew_ms","slew-ms"),
                        ("release_ms","release-ms"),("quiet_floor_db","quiet-floor-db"),("floor_guard","floor-guard"),
                        ("spectral_dry_guard","spectral-dry-guard"),
                        ("rumble_recovery","rumble-recovery"),("rumble_adaptive_hz","rumble-adaptive-hz"),
                        ("weak_speech_recovery","weak-speech-recovery"),
                        ("noise_step_recovery","noise-step-recovery")):
                    command += ["--vehicle-ns-"+flag,str(c[field])]
                command += ["--vehicle-rumble-hz",str(c["rumble_hz"]),
                    "--speex-suppression-db" if c["backend"]=="speex" else "--ns-level",str(c["strength"])]
                if c["adaptive_rumble"]:
                    command.append("--vehicle-adaptive-rumble")
                proc = subprocess.run(command,capture_output=True,text=True)
                (folder/(engine_profile+".invocation.json")).write_text(json.dumps(dict(
                    argv=command,returncode=proc.returncode,stderr=proc.stderr),indent=2)+"\n")
                proc.check_returncode()
                audio, actual_rate = sf.read(prefix.with_suffix(".after.wav"))
                info = json.loads(prefix.with_suffix(".json").read_text())
                if actual_rate!=ref_rate or audio.shape!=reference.shape:
                    raise ValueError("engine rate/shape differs from the scored component")
                delta = float(np.max(np.abs(audio-reference)))
                if (delta!=0 or info["dsp_delay_samples"]!=row["engine"]["delay_samples"]
                        or info["external_vad_enabled"]):
                    raise ValueError("actual engine differs from scored component: "+str(delta))
                for field in ("spectral_dry_guard","rumble_recovery","rumble_adaptive_hz",
                              "weak_speech_recovery","noise_step_recovery"):
                    if info["vehicle_ns_"+field]!=c[field]:
                        raise ValueError("new option did not reach the engine CLI")
                with prefix.with_suffix(".frames.csv").open() as stream:
                    if any(int(r["external_vad_calls"]) for r in csv.DictReader(stream)):
                        raise ValueError("external VAD was called")
                results.append(dict(profile=profile["name"],engine_profile=engine_profile,
                    run=row["run"],id=row["id"],rate=rate,delay_samples=info["dsp_delay_samples"],
                    max_abs_difference=delta,bitexact=True,external_vad_calls=0))
    report = dict(state="completed",comparisons=results,all_bitexact=True,
        engine_sha256=sha(binary),native_rate=16000,qualification_scope="isolated NS in actual CLI; not full duplex quality")
    report["public_profiles_sha256"] = sha(args.profiles)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("actual engine CLI equivalence:",len(results),"bitexact comparisons")


if __name__ == "__main__":
    main()
