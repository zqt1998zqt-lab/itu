#!/usr/bin/env python3
"""Matched, uninterrupted C++ streams for P.1120-inspired digital diagnostics."""
import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import run_audio_eval as ev
from itu_method_metrics import noise_continuity, double_talk_attenuation, cng_difference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--cli", type=Path, required=True)
    p.add_argument("--speakers", type=int, default=2)
    p.add_argument("--cli-option", action="append", default=[])
    p.add_argument("--signal", type=Path, help="optional caller-supplied prescribed stimulus; identity retained")
    a = p.parse_args()
    a.root.mkdir(parents=True, exist_ok=False)
    binary = a.root / "itu_audio_cli"; shutil.copy2(a.cli, binary)
    manifest = json.loads((a.dataset / "inputs-manifest.json").read_text())
    candidates = [s for s in manifest["scenes"] if s["mics"] == 1 and s["field_label"] == "diffuse" and s["snr_db"] == 15]
    if not 1 <= a.speakers <= len(candidates): p.error("invalid source speaker count")
    identity = dict(cli_sha256=ev.digest(binary), input_manifest_sha256=ev.digest(a.dataset / "inputs-manifest.json"),
                    runner_sha256=ev.digest(Path(__file__)), metric_sha256=ev.digest(Path(__file__).with_name("itu_method_metrics.py")),
                    source_pdf_sha256=ev.digest(Path(__file__).resolve().parents[2] / "build/itut-standards-20261010/P.1120.pdf"),
                    adaptation_s=5, input_native_highband=False, p501_signal_verified=False,
                    stimulus_scope="natural speech bursts with known gaps; not certified P.501 CSS",
                    options=a.cli_option, standard_compliance=False)
    rows = []
    for scene in candidates[:a.speakers]:
        folder = a.dataset / "cases" / scene["id"]
        speech, sr = sf.read(a.signal or folder / "clean.wav")
        if speech.ndim != 1: raise ValueError("mono stimulus required")
        raw, noise_rate = sf.read(folder / "noise-component.wav")
        # Preserve the same sample realization across all four chain invocations.
        noise_source = raw / max(np.sqrt(np.mean(raw*raw)), 1e-12)
        for rate in (16000, 24000, 48000):
            gcd = math.gcd(rate, sr)
            voice = resample_poly(speech, rate//gcd, sr//gcd) if rate != sr else speech.copy()
            voice /= max(np.sqrt(np.mean(voice*voice)), 1e-12)
            gcd = math.gcd(rate, noise_rate)
            road = resample_poly(noise_source, rate//gcd, noise_rate//gcd) if rate != noise_rate else noise_source
            for kind in ("road", "white"):
                case = a.root / f"p{scene['pair']}-{rate}-{kind}"; case.mkdir()
                count = rate*22
                noise = (np.resize(road, count) if kind == "road" else np.random.default_rng(10+scene['pair']).normal(size=count)) * .004
                clean = np.zeros(count); playback = np.zeros(count)
                windows = [[5.5,7.5],[8.5,10.5],[11.5,13.5],[14.5,16.5]]
                for lo, hi in windows:
                    start,end=round(lo*rate),round(hi*rate)
                    piece=np.resize(voice, end-start)*.04
                    fade=round(.02*rate); piece[:fade]*=np.linspace(0,1,fade); piece[-fade:]*=np.linspace(1,0,fade)
                    clean[start:end]=piece
                    playback[start:end]=np.roll(piece, rate//2)
                echo=np.r_[np.zeros(round(.03*rate)),playback][:(count)]*.3
                waves={"noise":(noise,playback*0),"near":(noise+clean,playback*0),
                       "far":(noise+echo,playback),"double":(noise+clean+echo,playback)}
                outputs={}; source_hashes={}
                for mode,(mic,ref) in waves.items():
                    if max(abs(mic).max(),abs(ref).max())>=.95: raise ValueError("clipped generated input")
                    mic_path=case/f"{mode}.mic.wav"; ref_path=case/f"{mode}.ref.wav"
                    sf.write(mic_path,mic,rate,subtype="FLOAT"); sf.write(ref_path,ref,rate,subtype="FLOAT")
                    source_hashes[mode]={"mic":ev.digest(mic_path),"ref":ev.digest(ref_path)}
                    prefix=case/mode
                    command=[str(binary.resolve()),"--profile","vehicle_stream","--mic",str(mic_path),"--echo",str(ref_path),"--out",str(prefix)]+a.cli_option
                    proc=subprocess.run(command,capture_output=True,text=True,timeout=120)
                    prefix.with_suffix(".invocation.json").write_text(json.dumps(dict(argv=command,exit_code=proc.returncode,stderr=proc.stderr))+"\n")
                    proc.check_returncode()
                    engine=json.loads(prefix.with_suffix(".json").read_text())
                    out,actual_rate=sf.read(prefix.with_suffix(".after.wav"));delay=engine["dsp_delay_samples"]
                    if actual_rate!=rate or len(out)!=count+delay:raise ValueError("output timeline mismatch")
                    outputs[mode]=out[delay:]
                gaps=[[7.8,8.4],[10.8,11.4],[13.8,14.4],[16.8,20.5]]
                row=dict(id=case.name,source_scene=scene['id'],sample_rate=rate,source_hashes=source_hashes,
                         near_noise=noise_continuity(outputs['noise'],outputs['near'],rate,gaps),
                         far_noise=noise_continuity(outputs['noise'],outputs['far'],rate,[[5.5,20.5]]),
                         double_talk=double_talk_attenuation(clean,outputs['near'],outputs['double'],rate,windows),
                         cng=cng_difference(outputs['noise'][:rate*10],outputs['far'][rate*10:],rate,False))
                rows.append(row); print(json.dumps(row),flush=True)
    if ev.digest(binary)!=identity['cli_sha256']:raise ValueError("frozen executable changed")
    report=dict(identity=identity,rows=rows,summary={name:{'passed':sum(r[name]['passed'] is True for r in rows),
                    'failed':sum(r[name]['passed'] is False for r in rows),'unavailable':sum(r[name]['passed'] is None for r in rows)}
                    for name in ('near_noise','far_noise','double_talk','cng')})
    (a.root/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report['summary'],indent=2))


if __name__=='__main__': main()
