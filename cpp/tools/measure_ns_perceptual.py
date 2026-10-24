#!/usr/bin/env python3
"""Paired DNSMOS/PESQ diagnostics on the first scheduled speech utterance.

Model/metric resampling is scoring-only; source audio and DSP inputs stay native.
These metrics have no new pass threshold and are not POLQA or listening tests.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
from pesq import pesq
from scipy import signal
import soundfile as sf
import compare_ns_backends as paired
from ecnr_perceptual import DNSMOS,DEFAULT_ASSETS

SCORER=None


def evaluate(task):
    global SCORER
    root,run,original=task;s=original['scene'];rate=s['sample_rate']
    if SCORER is None:SCORER=DNSMOS(DEFAULT_ASSETS,root/'perceptual/cache')
    folder=paired.replay.PROJECT/'build'/run/'cases'/s['id']
    clean,sr=sf.read(folder/'clean.wav');before,br=sf.read(folder/'before.wav')
    if sr!=rate or br!=rate:raise ValueError('reference rate differs')
    lo,hi=s['near_schedule'][0];sl=slice(round(lo*rate),round(hi*rate));target=clean[sl]
    signals={'input':before[sl]}
    for name in ('webrtc0','speex12'):
        for phase in ('full-defaults','isolation-strengths'):
            prefix=root/phase/name/run/('cases' if phase=='full-defaults' else '')/s['id']
            prefix=prefix/'runs/full/result' if phase=='full-defaults' else prefix/'result'
            m=json.loads(prefix.with_suffix('.json').read_text());a,ar=sf.read(prefix.with_suffix('.after.wav'))
            if ar!=rate:raise ValueError('output rate differs')
            delay=m['dsp_delay_samples'] if phase=='full-defaults' else m['delay_samples']
            signals[('vcp_' if phase=='full-defaults' else 'raw_')+name]=a[delay:delay+len(clean)][sl]
    def narrow(x):
        if rate==16000:return x
        divisor=math.gcd(rate,16000)
        return signal.resample_poly(x,16000//divisor,rate//divisor)
    target16=narrow(target);scores={}
    for name,audio in signals.items():
        if len(audio)!=len(target):raise ValueError('scoring window differs')
        mos=SCORER.score_array(audio,rate,kind='near_only')
        try:
            p=float(pesq(16000,target16,narrow(audio),'wb'));error=None
        except Exception as exc:p=None;error=type(exc).__name__+': '+str(exc)
        scores[name]=dict(dnsmos=mos,pesq_wb_16k=p,pesq_unavailable_reason=error)
    return dict(original_run=run,id=s['id'],sample_rate=rate,noise=s['noise'],snr_db=s['snr_db'],
        split=s['split'],window_s=[lo,hi],scores=scores,
        scope='same first scheduled speech utterance; fixed declared delay; no oracle lag or gain fit')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--jobs',type=int,default=6);args=p.parse_args();root=args.root.resolve()
    if not 1<=args.jobs<=8:p.error('jobs must be 1..8')
    out=root/'perceptual';out.mkdir(exist_ok=False)
    selected=[(run,row) for run,row in paired.selected_cases('sweep') if row['scene']['family']=='ns']
    artifacts=out/'artifacts';artifacts.mkdir()
    import ecnr_perceptual
    deps=[Path(__file__),Path(ecnr_perceptual.__file__),DEFAULT_ASSETS/'manifest.json',DEFAULT_ASSETS/'sig_bak_ovr.onnx',
          DEFAULT_ASSETS/'dnsmos_local.py',root/'full-defaults/identity.json',root/'isolation-strengths/identity.json']
    bindings={str(path):paired.replay.ctl.ev.digest(path) for path in deps}
    for i,path in enumerate(deps):shutil.copy2(path,artifacts/f'{i:02d}-{path.name}')
    started=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool,(out/'results.jsonl').open('x') as stream:
        for n,future in enumerate(as_completed([pool.submit(evaluate,(root,run,row)) for run,row in selected]),1):
            stream.write(json.dumps(future.result(),allow_nan=False)+'\n');stream.flush()
            if n%25==0 or n==len(selected):
                paired.save(out/'report.json',dict(state='running',completed=n,total=len(selected),elapsed_seconds=time.monotonic()-started))
    changed=[path for path,sha in bindings.items() if paired.replay.ctl.ev.digest(Path(path))!=sha]
    paired.save(out/'identity.json',dict(dependencies=bindings,metric_scope='DNSMOS predicted P.835; PESQ WB at 16k scoring band',
        source_and_dsp_resampling=False,scoring_resampling='DNSMOS official 16k; PESQ common 16k bandwidth; original sources unchanged'))
    paired.save(out/'report.json',dict(state='completed',cases=len(selected),scores=len(selected)*5,identity_verified=not changed,
        changed_dependencies=changed,elapsed_seconds=time.monotonic()-started,standard_compliance=False))
    if changed:raise ValueError('perceptual dependencies changed')


if __name__=='__main__':main()
