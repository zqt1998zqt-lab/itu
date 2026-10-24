#!/usr/bin/env python3
"""Calibrate IMCRA B_min on seeded WGN for this exact causal analysis chain."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--frames',type=int,default=4000)
    args=p.parse_args()
    if args.frames<1000:raise ValueError('at least 1000 calibration frames required')
    rows=[]
    for rate in (16000,24000,48000):
        hop=rate//100;nfft=1<<(2*hop-1).bit_length()
        window=np.sin(np.pi*(np.arange(2*hop)+.5)/(2*hop))
        seeds=[]
        for seed in (718,719,720):
            rng=np.random.default_rng(seed);past=np.zeros(hop)
            history=None;s=None;values=[]
            for f in range(args.frames):
                current=rng.normal(size=hop)
                power=np.abs(np.fft.rfft(np.concatenate((past,current))*window,n=nfft))**2
                past=current
                sf=.25*np.r_[power[0],power[:-1]]+.5*power+.25*np.r_[power[1:],power[-1]]
                if s is None:
                    s=sf.copy();minimum=s.copy();sub=s.copy();history=np.tile(s,(8,1));head=0
                s=.9*s+.1*sf;minimum=np.minimum(minimum,s);sub=np.minimum(sub,s)
                if (f+1)%12==0:
                    history[head]=sub;minimum=history.min(axis=0);sub=s.copy();head=(head+1)%8
                if f>=500:values.append(float(np.mean(minimum[2:-2])/np.sum(window**2)))
            seeds.append(dict(seed=seed,b_min=1/np.mean(values),frames=len(values)))
        rows.append(dict(rate=rate,nfft=nfft,hop=hop,b_min=float(np.mean([r['b_min'] for r in seeds])),seeds=seeds))
    source=Path(__file__).resolve().parents[1]/'src/omlsa_imcra.hpp'
    data=dict(purpose='unconditional stage-1 minima calibration, not speech-quality tuning',
              analysis_window='20 ms sine',frame_ms=10,frequency_kernel=[.25,.5,.25],alpha_s=.9,
              subwindows=8,subwindow_frames=12,frames=args.frames,rows=rows,
              source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    args.out.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(rows))

if __name__=='__main__':main()
