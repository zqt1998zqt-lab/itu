#!/usr/bin/env python3
"""Native 48 kHz paired VoiceBank recordings; local, rate-aware C++ simulation."""
import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, welch
import run_audio_eval as ev
import check_audio_acceptance as ac


def band_power(x, rate, lo, hi):
    f, p = welch(x, rate, nperseg=min(len(x), 8192))
    return float(np.sum(p[(f >= lo) & (f < hi)]) * (f[1] - f[0]))


def native_sisdr(clean, output):
    x, y = clean-clean.mean(), output-output.mean()
    projected = x*(float(x@y)/max(float(x@x),1e-20))
    return ev.db(float(projected@projected)/max(float((y-projected)@(y-projected)),1e-20))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--cli", type=Path, required=True)
    p.add_argument("--per-speaker", type=int, default=4)
    a = p.parse_args()
    a.root.mkdir(parents=True, exist_ok=False)
    binary = a.root / "itu_audio_cli"; shutil.copy2(a.cli, binary)
    rules = json.loads(ac.RULES.read_text())
    if ev.digest(ac.RULES) != ac.RULES_SHA256: raise ValueError("frozen thresholds changed")
    selected, counts = [], {}
    for clean_path in sorted((a.source / "clean_testset_wav").glob("*.wav")):
        speaker = clean_path.stem.split("_")[0]
        if counts.get(speaker, 0) >= a.per_speaker: continue
        noisy_path = a.source / "noisy_testset_wav" / clean_path.name
        c, rate = sf.read(clean_path); b, r = sf.read(noisy_path)
        if rate != 48000 or r != rate or c.ndim != 1 or c.shape != b.shape: raise ValueError("source mismatch")
        if len(c) < rate*3 or not np.isfinite(c).all() or not np.isfinite(b).all() or max(abs(c).max(), abs(b).max()) >= .999: continue
        selected.append((clean_path, noisy_path, c, b, speaker)); counts[speaker] = counts.get(speaker, 0)+1
    if len(counts) < 2 or any(n != a.per_speaker for n in counts.values()): raise ValueError("incomplete speaker selection")
    identity = dict(cli_sha256=ev.digest(binary), rules_sha256=ev.digest(ac.RULES),
                    runner_sha256=ev.digest(Path(__file__)), native_source_rate=48000,
                    source_url="https://datashare.ed.ac.uk/handle/10283/2791", source_license="CC BY 4.0",
                    selection="first sorted >=3s finite unclipped pair per speaker; independent of algorithm scores",
                    native_highband_available=True, standard_compliance=False,
                    scope="native recorded pair plus controlled SNR; mono; noise=noisy-clean, not isolated microphone recording")
    rows, assets = [], []
    for cp, bp, c48, b48, speaker in selected:
        asset = dict(id=cp.stem, speaker=speaker, clean_path=str(cp.resolve()), noisy_path=str(bp.resolve()),
                     clean_sha256=ev.digest(cp), noisy_sha256=ev.digest(bp), seconds=len(c48)/48000,
                     clean_8_12k_power=band_power(c48,48000,8000,12000), clean_12_14k_power=band_power(c48,48000,12000,14000))
        assets.append(asset)
        for rate in (16000,24000,48000):
            gcd=math.gcd(rate,48000)
            clean=resample_poly(c48,rate//gcd,48000//gcd) if rate != 48000 else c48.copy()
            noise=resample_poly(b48-c48,rate//gcd,48000//gcd) if rate != 48000 else b48-c48
            for snr in (0,15,30):
                n=noise*np.sqrt(np.mean(clean*clean)/max(np.mean(noise*noise),1e-20)/10**(snr/10))
                gain=min(1.,.9/max(abs(clean+n).max(),abs(clean).max(),1e-12))
                c,b=clean*gain,(clean+n)*gain
                folder=a.root/f"{cp.stem}-r{rate}-snr{snr}";folder.mkdir()
                mic=folder/'before.wav';ref=folder/'ref.wav'
                sf.write(mic,b,rate,subtype="FLOAT");sf.write(ref,np.zeros(len(b)),rate,subtype="FLOAT")
                for variant, flags in (("baseline",[]),("control",["--noise-transition-control"])):
                    prefix=folder/variant
                    command=[str(binary.resolve()),"--profile","vehicle_stream","--mic",str(mic),"--echo",str(ref),"--out",str(prefix)]+flags
                    proc=subprocess.run(command,capture_output=True,text=True,timeout=120)
                    prefix.with_suffix('.invocation.json').write_text(json.dumps(dict(argv=command,exit_code=proc.returncode,stderr=proc.stderr))+"\n");proc.check_returncode()
                    engine=json.loads(prefix.with_suffix('.json').read_text());out,sr=sf.read(prefix.with_suffix('.after.wav'))
                    after=out[engine['dsp_delay_samples']:]
                    if sr != rate or len(after)!=len(c):raise ValueError("timeline mismatch")
                    to16=lambda x: resample_poly(x,16000//math.gcd(16000,rate),rate//math.gcd(16000,rate)) if rate!=16000 else x
                    qb,qa=ev.quality(to16(c),to16(b)),ev.quality(to16(c),to16(after))
                    segment=dict(before=qb,after=qa,stoi_improvement=qa['stoi']-qb['stoi'] if qb['stoi'] is not None and qa['stoi'] is not None else None)
                    measured=dict(mode="full",segments=dict(near_only=segment),engine=engine,
                                  process_frames_over_10ms=engine['process_frames_over_10ms'],output_levels=dict(peak_dbfs=ev.db(float(max(abs(out)))**2),clipping_percent=float(np.mean(abs(out)>=.999)*100)))
                    verdict=ac.check_scene(dict(id=cp.stem,family="ns",mics=1,noise="native_recorded_residual",snr_db=snr,stress="none",windows=dict(near_only=[0,len(c)/rate])),dict(measured,mode="full"),rules)
                    highband=None
                    if rate>=24000:
                        hi=11000 if rate==24000 else 14000
                        pc,po=band_power(c,rate,8000,hi),band_power(after,rate,8000,hi)
                        highband=dict(band_hz=[8000,hi],clean_power=pc,output_power=po,
                                      output_to_clean_power_db=ev.db(po/pc) if pc>1e-20 and po>1e-20 else None,
                                      scope="observed band power including residual noise; not target-only gain or standardized frequency response")
                    row=dict(id=cp.stem,speaker=speaker,rate=rate,snr_db=snr,variant=variant,measurements=measured,
                             native_sisdr_db=native_sisdr(c,after),highband=highband,acceptance=verdict)
                    rows.append(row);print(json.dumps(dict(id=cp.stem,rate=rate,snr=snr,variant=variant,passed=verdict['passed'])),flush=True)
    if ev.digest(binary)!=identity['cli_sha256']:raise ValueError("frozen CLI changed")
    (a.root/'report.json').write_text(json.dumps(dict(identity=identity,assets=assets,rows=rows,
        summary=dict(runs=len(rows),speakers=counts,all_pass=sum(r['acceptance']['passed'] for r in rows))),indent=2,allow_nan=False)+'\n')


if __name__=='__main__': main()
