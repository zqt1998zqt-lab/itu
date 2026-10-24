#!/usr/bin/env python3
"""Standalone measured tradeoff figure; no metric threshold adjustments."""
import os
os.environ['MPLBACKEND']='Agg'
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import refine_ns_streams as opt


def main():
    old=opt.PROJECT/'build/ns-optimization-20261018';root=opt.ROOT
    names=['webrtc_best','webrtc_v2_balanced','webrtc_v2_startup','speex_best','speex_v2_balanced','speex_v2_quality']
    labels=['W previous','W balanced','W startup','S previous','S balanced','S quality'];colors=['#8c9aa8','#3987d3','#005a9c','#8c9aa8','#e8a34b','#a75c0b']
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained');data={}
    for ax,(phase,domain,title) in zip(axes.flat,[('final-vehicle','recorded_car','Recorded car noise: simulated mixtures'),('final-vehicle','synthetic_vehicle','Synthetic vehicle probes'),('final-clean',None,'Clean speech protection'),('final-cold',None,'Speech at stream start')]):
        rows=[]
        for base,ph in [(old,'final-cold-v2' if phase=='final-cold' else phase),(root,phase)]:rows += [json.loads(l) for l in (base/ph/'results.jsonl').open()]
        counts=[]
        for name in names:
            group=[r for r in rows if r['variant']==name and (domain is None or r['domain']==domain)];counts.append((sum(r['ns_tradeoff_pass'] for r in group),len(group)))
        bars=ax.bar(labels,[100*p/n for p,n in counts],color=colors)
        for bar,(p,n) in zip(bars,counts):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1.5,f'{p}/{n}',ha='center',fontsize=9)
        ax.set(ylim=(0,112),title=title,ylabel='Engineering pass (%)');ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2);ax.tick_params(axis='x',labelrotation=15)
        data[title]=dict(zip(names,[dict(passed=p,cases=n) for p,n in counts]))
    fig.suptitle('NS-only frozen replay: measured tradeoffs',fontsize=15)
    fig.savefig(root/'frozen-replay.png',dpi=160)
    opt.save(root/'frozen-replay-plot-data.json',data)

if __name__=='__main__':main()
