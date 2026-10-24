#!/usr/bin/env python3
"""Diagnostic NS effect with independently measured neutral frontend targets.

Does not replace raw waveform quality or any engineering acceptance gate.
WebRTC's QMF reference and Speex's SRC-only reference are measured separately.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import soundfile as sf
import compare_ns_backends as paired


def task(args):
    root,run,original=args;s=original['scene'];rate=s['sample_rate']
    origin=paired.replay.PROJECT/'build'/run/'cases'/s['id']
    folder=root/'frontend-diagnostic'/run/s['id'];folder.mkdir(parents=True)
    rows={}
    for name,neutral in (('webrtc0','filterbank'),('speex12','bypass')):
        prefix=folder/neutral
        command=[str(root/'build-tree/ns_backend_probe'),neutral,'0',str(origin/'clean.wav'),str(prefix),'shared']
        result=subprocess.run(command,capture_output=True,text=True);result.check_returncode()
        m=json.loads(prefix.with_suffix('.json').read_text());target,sr=sf.read(prefix.with_suffix('.after.wav'))
        n=m['samples'];target=target[m['delay_samples']:m['delay_samples']+n]
        p=root/'isolation-strengths'/neutral/run/s['id']/'result';bmeta=json.loads(p.with_suffix('.json').read_text());before,br=sf.read(p.with_suffix('.after.wav'));before=before[bmeta['delay_samples']:bmeta['delay_samples']+n]
        p=root/'isolation-strengths'/name/run/s['id']/'result';ameta=json.loads(p.with_suffix('.json').read_text());after,ar=sf.read(p.with_suffix('.after.wav'));after=after[ameta['delay_samples']:ameta['delay_samples']+n]
        if sr!=rate or br!=rate or ar!=rate:raise ValueError('rate mismatch')
        segments={};paired.replay.ctl.ev.RATE=rate
        for label in ('near_only','near_late'):
            lo,hi=s['windows'][label];sl=slice(round(lo*rate),round(hi*rate))
            qb=paired.replay.ctl.ev.quality(target[sl],before[sl]);qa=paired.replay.ctl.ev.quality(target[sl],after[sl])
            segments[label]=dict(before=qb,after=qa,sisdr_improvement_db=qa['sisdr_db']-qb['sisdr_db'],stoi_improvement=qa['stoi']-qb['stoi'])
        rows[name]=segments
    return dict(original_run=run,id=s['id'],rate=rate,snr_db=s['snr_db'],noise=s['noise'],measurements=rows,
        scope='diagnostic only: frontend-matched before and clean target; fixed declared delay, no lag search')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--jobs',type=int,default=4)
    args=p.parse_args();root=args.root.resolve();out=root/'frontend-diagnostic';out.mkdir(exist_ok=False)
    selected=[(run,row) for run,row in paired.selected_cases('sweep') if row['scene']['family']=='ns']
    with ProcessPoolExecutor(max_workers=args.jobs) as pool,(out/'results.jsonl').open('x') as stream:
        for row in pool.map(task,[(root,run,row) for run,row in selected]):stream.write(json.dumps(row)+'\n')
    rows=[json.loads(l) for l in (out/'results.jsonl').read_text().splitlines()];summary=[]
    for rate in (16000,24000,48000):
        for band in ('low','mid','high'):
            group=[r for r in rows if r['rate']==rate and ('low' if r['snr_db']<0 else 'mid' if r['snr_db']<=15 else 'high')==band]
            for name in ('webrtc0','speex12'):
                summary.append(dict(rate=rate,snr_band=band,variant=name,cases=len(group),
                    mean_sisdr_improvement_db=float(np.mean([np.mean([s['sisdr_improvement_db'] for s in r['measurements'][name].values()]) for r in group])),
                    mean_stoi_improvement=float(np.mean([np.mean([s['stoi_improvement'] for s in r['measurements'][name].values()]) for r in group]))))
    shutil.copy2(__file__,out/Path(__file__).name)
    paired.save(out/'report.json',dict(state='completed',cases=len(rows),summary=summary,
        diagnostic_only=True,original_acceptance_unchanged=True,standard_compliance=False,
        script_sha256=paired.replay.ctl.ev.digest(Path(__file__)),probe_sha256=paired.replay.ctl.ev.digest(root/'build-tree/ns_backend_probe')))


if __name__=='__main__':main()
