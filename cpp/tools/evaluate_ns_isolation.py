#!/usr/bin/env python3
"""Isolated NS strength sweeps, fixed-delay quality and 10 ms diagnostics."""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import soundfile as sf
from scipy import signal
import compare_ns_backends as paired


VARIANTS = {**{f'webrtc{i}': ['webrtc', i, 'shared'] for i in range(4)},
            **{f'speex{db}': ['speex', -db, 'shared'] for db in (6, 12, 18, 24)},
            'bypass': ['bypass', 0, 'shared'],
            'filterbank': ['filterbank', 0, 'shared']}


def projection(clean, observed):
    denominator = float(np.dot(clean, clean))
    if denominator < 1e-12:
        return dict(projection_gain_db=None, nmse_db=None)
    alpha = float(np.dot(clean, observed)) / denominator
    return dict(projection_gain_db=float(20*np.log10(max(abs(alpha), 1e-12))),
                nmse_db=float(10*np.log10(max(float(np.dot(observed-clean, observed-clean))/denominator, 1e-12))))


def quality(clean, observed, rate):
    paired.replay.ctl.ev.RATE = rate
    return paired.replay.ctl.ev.quality(clean, observed)


def spectral_distance(clean, observed, rate):
    # Diagnostic speech-bin LSD; target excitation threshold is explicit.
    _, _, c = signal.stft(clean, rate, nperseg=rate//50, noverlap=rate//100, boundary=None)
    _, _, a = signal.stft(observed, rate, nperseg=rate//50, noverlap=rate//100, boundary=None)
    p = abs(c)**2
    active = p > max(float(p.max()) * 1e-4, 1e-12)
    if not active.any():
        return None
    distance = 10*np.log10(np.maximum(abs(a)**2, 1e-12)) - 10*np.log10(np.maximum(p, 1e-12))
    return float(np.sqrt(np.mean(distance[active]**2)))


def task(args):
    root, probe, run, original, name, variant = args
    scene = original['scene']; rate = scene['sample_rate']; hop = rate//100
    origin = paired.replay.PROJECT/'build'/run/'cases'/scene['id']
    folder = root/name/run/scene['id']; folder.mkdir(parents=True)
    prefix = folder/'result'
    for filename in ('before.wav', 'clean.wav'):
        if paired.replay.ctl.ev.digest(origin/filename) != scene['files'][filename]['sha256']:
            raise ValueError('input identity differs')
    command = [str(probe), str(variant[0]), str(variant[1]), str(origin/'before.wav'), str(prefix), variant[2]]
    result = subprocess.run(command, capture_output=True, text=True)
    paired.save(folder/'invocation.json', dict(argv=command, exit_code=result.returncode,
                                             stderr=result.stderr, stdout=result.stdout))
    result.check_returncode()
    engine = json.loads(prefix.with_suffix('.json').read_text())
    if engine['backend'] != variant[0] or engine['strength'] != variant[1]:
        raise ValueError('runtime backend/strength differs')
    clean, sr = sf.read(origin/'clean.wav', dtype='float64')
    before, br = sf.read(origin/'before.wav', dtype='float64')
    # The actual probe input is quantized identically for both backends.
    before = np.clip(np.rint(before*32768), -32768, 32767)/32768
    out, ar = sf.read(prefix.with_suffix('.after.wav'), dtype='float64')
    delay = engine['delay_samples']
    if sr != rate or br != rate or ar != rate or len(out) != len(clean)+delay:
        raise ValueError('output shape/rate/delay differs')
    after = out[delay:delay+len(clean)]
    segments = {}
    for window, (lo, hi) in scene['windows'].items():
        sl = slice(round(lo*rate), round(hi*rate)); c,b,a = clean[sl],before[sl],after[sl]
        if window.startswith('near') or window == 'double_talk':
            qb, qa = quality(c,b,rate), quality(c,a,rate)
            segments[window] = dict(before=qb, after=qa,
                lsd_excited_bins_db=spectral_distance(c,a,rate),
                sisdr_improvement_db=qa['sisdr_db']-qb['sisdr_db'] if qa['sisdr_db'] is not None and qb['sisdr_db'] is not None else None,
                stoi_improvement=qa['stoi']-qb['stoi'] if qa['stoi'] is not None and qb['stoi'] is not None else None)
        else:
            segments[window] = paired.replay.ctl.ev.attenuation_measurement(b,a,pcm16=True)
    count=len(clean)//hop
    c,b,a=(v[:count*hop].reshape(count,hop) for v in (clean,before,after))
    cp=np.mean(c*c,axis=1); bp=np.mean(b*b,axis=1); ap=np.mean(a*a,axis=1)
    dot=np.sum(c*a,axis=1); denom=np.sum(c*c,axis=1)
    alpha=np.divide(dot,denom,out=np.zeros(count),where=denom>1e-12)
    active=cp>=10**(-65/10)
    gain=np.where(active,20*np.log10(np.maximum(abs(alpha),1e-12)),np.nan)
    before_alpha=np.divide(np.sum(c*b,axis=1),denom,out=np.zeros(count),where=denom>1e-12)
    before_gain=np.where(active,20*np.log10(np.maximum(abs(before_alpha),1e-12)),np.nan)
    nmse=10*np.log10(np.maximum(np.mean((a-c)**2,axis=1),1e-15)/np.maximum(cp,1e-15))
    gain_step=abs(np.diff(gain)); valid_pair=active[1:] & active[:-1]
    np.savez_compressed(folder/'frames.npz',time_s=np.arange(count)/100,
        target_power=cp,input_power=bp,output_power=ap,oracle_speech_active=active,
        projection_gain_db=gain,input_projection_gain_db=before_gain,nmse_db=nmse)
    onsets=[]
    for lo,hi in scene['near_schedule']:
        candidates=np.flatnonzero(active & (np.arange(count)/100>=lo) & (np.arange(count)/100<hi))
        if not len(candidates):
            continue
        onset=int(candidates[0])*hop
        windows={}
        for label,start,end in (('0_50ms',0,.05),('50_150ms',.05,.15),('150_500ms',.15,.5)):
            sl=slice(onset+round(start*rate),min(len(clean),onset+round(end*rate)))
            windows[label]=projection(clean[sl],after[sl])
        onsets.append(dict(schedule_start_s=lo,oracle_onset_s=onset/rate,windows=windows))
    row=dict(variant=name,original_run=run,scene={k:scene[k] for k in
        ('id','sample_rate','noise','snr_db','split','near_speaker','near_dbfs','family','seconds')},
        engine=engine,segments=segments,onsets=onsets,
        frames=dict(count=count,oracle_active_count=int(active.sum()),oracle_threshold_dbfs=-65,
            active_projection_p05_db=float(np.percentile(gain[active],5)) if active.any() else None,
            active_projection_median_db=float(np.median(gain[active])) if active.any() else None,
            active_projection_below_minus6_percent=float(np.mean(gain[active]<-6)*100) if active.any() else None,
            projection_step_p95_db=float(np.percentile(gain_step[valid_pair],95)) if valid_pair.any() else None),
        output_sha256=paired.replay.ctl.ev.digest(prefix.with_suffix('.after.wav')),
        input_sha256=scene['files']['before.wav']['sha256'],standard_compliance=False)
    paired.save(folder/'metrics.json',row)
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--jobs',type=int,default=8)
    args=p.parse_args()
    if not 1<=args.jobs<=12:p.error('jobs must be 1..12')
    selected=[(run,row) for run,row in paired.selected_cases('sweep') if row['scene']['family']=='ns']
    root=args.root.resolve();root.mkdir(parents=True,exist_ok=False)
    artifacts=root/'artifacts';artifacts.mkdir();probe=artifacts/'ns_backend_probe';shutil.copy2(args.probe,probe)
    deps=[Path(__file__),Path(paired.__file__),Path(paired.replay.__file__),Path(paired.replay.ctl.ev.__file__)]
    for run,manifest in paired.replay.SOURCES:
        deps.extend([paired.replay.PROJECT/'build'/run/'results.jsonl',paired.replay.PROJECT/'datasets'/manifest/'manifest.json'])
    bindings={str(x):paired.replay.ctl.ev.digest(x) for x in deps}
    for i,path in enumerate(deps):shutil.copy2(path,artifacts/f'{i:02d}-{path.name}')
    paired.save(root/'selection.json',[dict(run=run,id=row['scene']['id']) for run,row in selected])
    paired.save(root/'identity.json',dict(probe_sha256=paired.replay.ctl.ev.digest(probe),dependencies=bindings,
        variants=VARIANTS,scope='isolated mono NS; common PCM16 input; matched SRC at 24k',
        frame_oracle='evaluation only; never passed to DSP',alignment='declared fixed delay only'))
    tasks=[]
    for run,row in selected:
        tasks.extend((root,probe,run,row,name,variant) for name,variant in VARIANTS.items())
        if row['scene']['sample_rate']==24000:
            tasks.append((root,probe,run,row,'speex12native24',['speex',-12,'native24']))
    started=time.monotonic();completed=0
    paired.save(root/'report.json',dict(state='running',total=len(tasks),cases=len(selected)))
    with ProcessPoolExecutor(max_workers=args.jobs) as pool,(root/'results.jsonl').open('x') as stream:
        for future in as_completed([pool.submit(task,t) for t in tasks]):
            row=future.result();completed+=1;stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            if completed%100==0 or completed==len(tasks):
                report=dict(state='running',completed=completed,total=len(tasks),cases=len(selected),elapsed_seconds=time.monotonic()-started)
                paired.save(root/'report.json',report);print(json.dumps(report),flush=True)
    changed=[path for path,sha in bindings.items() if paired.replay.ctl.ev.digest(Path(path))!=sha]
    report.update(state='completed',identity_verified=not changed,changed_dependencies=changed,standard_compliance=False)
    paired.save(root/'report.json',report)
    if changed:raise ValueError(f'evaluator dependencies changed: {changed}')


if __name__=='__main__':main()
