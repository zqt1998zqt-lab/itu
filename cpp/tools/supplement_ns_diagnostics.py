#!/usr/bin/env python3
"""Clean-only preservation and sustained noise-step diagnostics."""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import soundfile as sf
import compare_ns_backends as paired


def clean_task(args):
    root,run,original,phase,name=args;s=original['scene'];rate=s['sample_rate']
    origin=paired.replay.PROJECT/'build'/run/'cases'/s['id'];input_path=origin/'clean.wav'
    if paired.replay.ctl.ev.digest(input_path)!=s['files']['clean.wav']['sha256']:raise ValueError('clean input changed')
    folder=root/'supplement/clean-only'/phase/name/s['files']['clean.wav']['sha256'];folder.mkdir(parents=True)
    prefix=folder/'result';backend='webrtc' if name=='webrtc0' else 'speex'
    if phase=='raw':
        command=[str(root/'build-tree/ns_backend_probe'),backend,'0' if backend=='webrtc' else '-12',str(input_path),str(prefix),'shared']
    else:
        flags=['--ns-backend',backend]+(['--ns-level','0'] if backend=='webrtc' else ['--speex-suppression-db','-12'])
        command=[str(root/'build-tree/cpp/itu_audio_cli'),'--profile','vcp','--mic',str(input_path),'--no-aec','--out',str(prefix)]+flags
    result=subprocess.run(command,capture_output=True,text=True);result.check_returncode()
    paired.save(folder/'invocation.json',dict(argv=command,exit_code=result.returncode,stderr=result.stderr))
    m=json.loads(prefix.with_suffix('.json').read_text());delay=m['delay_samples'] if phase=='raw' else m['dsp_delay_samples']
    c,sr=sf.read(input_path);a,ar=sf.read(prefix.with_suffix('.after.wav'));a=a[delay:delay+len(c)]
    if sr!=rate or ar!=rate or len(a)!=len(c):raise ValueError('clean output shape differs')
    paired.replay.ctl.ev.RATE=rate;windows={}
    for label in ('near_only','near_late'):
        lo,hi=s['windows'][label];sl=slice(round(lo*rate),round(hi*rate));windows[label]=paired.replay.ctl.ev.quality(c[sl],a[sl])
    hop=rate//100;n=len(c)//hop;cframes=c[:n*hop].reshape(n,hop);aframes=a[:n*hop].reshape(n,hop)
    p=np.mean(cframes*cframes,axis=1);active=p>=10**(-65/10)
    alpha=np.sum(cframes*aframes,axis=1)/np.maximum(np.sum(cframes*cframes,axis=1),1e-20)
    gain=20*np.log10(np.maximum(abs(alpha),1e-12))
    return dict(phase=phase,variant=name,rate=rate,source_sha256=s['files']['clean.wav']['sha256'],
        original_run=run,id=s['id'],windows=windows,active_frame_projection_below_minus6_percent=float(np.mean(gain[active]<-6)*100),
        mean_speech_sisdr_db=float(np.mean([x['sisdr_db'] for x in windows.values()])),
        mean_speech_stoi=float(np.mean([x['stoi'] for x in windows.values() if x['stoi'] is not None])),
        mean_projection_gain_db=float(np.mean([x['projection_gain_db'] for x in windows.values()])))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--jobs',type=int,default=8)
    args=p.parse_args();root=args.root.resolve();out=root/'supplement';out.mkdir(exist_ok=False)
    selected=[(run,row) for run,row in paired.selected_cases('sweep') if row['scene']['family']=='ns'];unique={}
    for run,row in selected:unique.setdefault(row['scene']['files']['clean.wav']['sha256'],(run,row))
    tasks=[(root,run,row,phase,name) for run,row in unique.values() for phase in ('raw','vcp') for name in ('webrtc0','speex12')]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool,(out/'clean-results.jsonl').open('x') as stream:
        for row in pool.map(clean_task,tasks):stream.write(json.dumps(row)+'\n')
    rows=[json.loads(l) for l in (out/'clean-results.jsonl').read_text().splitlines()];clean_summary=[]
    for phase in ('raw','vcp'):
        for name in ('webrtc0','speex12'):
            for rate in (16000,24000,48000):
                group=[r for r in rows if r['phase']==phase and r['variant']==name and r['rate']==rate]
                clean_summary.append(dict(phase=phase,variant=name,rate=rate,cases=len(group),
                    mean_sisdr_db=float(np.mean([r['mean_speech_sisdr_db'] for r in group])),
                    mean_stoi=float(np.mean([r['mean_speech_stoi'] for r in group])),
                    mean_projection_gain_db=float(np.mean([r['mean_projection_gain_db'] for r in group])),
                    mean_active_frame_below_minus6_percent=float(np.mean([r['active_frame_projection_below_minus6_percent'] for r in group]))))
    dynamics=[]
    for line in (root/'isolation-strengths/results.jsonl').open():
        r=json.loads(line)
        if r['variant'] not in ('webrtc0','speex12') or 'noise_only_after_step' not in r['segments']:continue
        scene=next(row['scene'] for run,row in selected if run==r['original_run'] and row['scene']['id']==r['scene']['id'])
        frame=np.load(root/'isolation-strengths'/r['variant']/r['original_run']/scene['id']/'frames.npz')
        lo,hi=scene['windows']['noise_only_after_step'];start,end=round(lo*100),round(hi*100)
        attenuation=10*np.log10(np.maximum(frame['input_power'][start:end],1e-20)/np.maximum(frame['output_power'][start:end],1e-20))
        qualifying=np.flatnonzero(np.convolve((attenuation>=6).astype(int),np.ones(5,dtype=int),'valid')==5)
        dynamics.append(dict(variant=r['variant'],rate=scene['sample_rate'],original_run=r['original_run'],id=scene['id'],
            window_s=[lo,hi],first_50ms_at_6db_s=float(qualifying[0]/100) if len(qualifying) else None,
            window_attenuation_db=r['segments']['noise_only_after_step']['total_attenuation_db']))
    paired.save(out/'report.json',dict(state='completed',clean_unique_sources=len(unique),clean_invocations=len(tasks),
        clean_summary=clean_summary,noise_step=dynamics,
        clean_input_scope='unaltered existing clean.wav, deduplicated by content SHA256; no added noise',
        step_scope='first five consecutive 10ms frames >=6dB, within the declared post-step noise-only window; unavailable means not reached',
        script_sha256=paired.replay.ctl.ev.digest(Path(__file__)),standard_compliance=False))
    shutil.copy2(__file__,out/Path(__file__).name)


if __name__=='__main__':main()
