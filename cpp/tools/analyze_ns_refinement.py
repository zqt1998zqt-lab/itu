#!/usr/bin/env python3
"""Transparent split-only ranking and paired NS refinement audit."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import refine_ns_backends as opt


def mean(v):
    a=[x for x in v if x is not None];return float(np.mean(a)) if a else None


def summary(rows):
    cells=defaultdict(list)
    for r in rows:cells[(r['scene']['sample_rate'],r['scene']['noise'])].append(r)
    return dict(cases=len(rows),passed=sum(r['ns_tradeoff_pass'] for r in rows),
        raw_passed=sum(r['raw_quality_pass'] for r in rows),
        balanced_pass=mean([mean([r['ns_tradeoff_pass'] for r in g]) for g in cells.values()]),
        pesq=mean([r['pesq_first'] for r in rows]),pesq_available=sum(r['pesq_first'] is not None for r in rows),
        stoi_delta=mean([r['mean_stoi_improvement'] for r in rows]),
        noise_attenuation_db=mean([r['mean_noise_attenuation_db'] for r in rows]),
        onset_0_50_db=mean([r['onset_0_50ms_projection_db'] for r in rows]),
        failures=dict(Counter(f for r in rows for f in r['failures'])))


def analyze(phases,output):
    rows=[]
    for phase in phases:
        report=json.loads((opt.ROOT/phase/'report.json').read_text())
        if report['state']!='completed' or not report['identity_verified']:raise ValueError('incomplete or drifting phase')
        rows.extend(json.loads(x) for x in (opt.ROOT/phase/'results.jsonl').open())
    seen=set()
    for r in rows:
        key=(r['variant'],r['original_run'],r['scene']['id'],r['scope'])
        if key in seen:raise ValueError('duplicate measurement')
        seen.add(key)
    result={}
    for split in ('development','validation','holdout'):
        result[split]={}
        for name in sorted({r['variant'] for r in rows}):
            g=[r for r in rows if r['variant']==name and r['scene']['split']==split]
            if not g:continue
            result[split][name]=dict(all=summary(g),**{d:summary([r for r in g if r['domain']==d]) for d in ('recorded_car','synthetic_vehicle','diagnostic') if any(r['domain']==d for r in g)})
    ranking={}
    for backend in ('webrtc','speex'):
        static=result['validation'][backend+'_static_best'];prev=result['validation'][backend+'_best'];candidates=[]
        for name,stats in result['validation'].items():
            if not name.startswith(backend+'_'):continue
            if not all(d in stats for d in ('recorded_car','synthetic_vehicle')):continue
            guards=dict(recorded_pass=stats['recorded_car']['balanced_pass']>=static['recorded_car']['balanced_pass']-1e-12,
                recorded_pesq=stats['recorded_car']['pesq']>=static['recorded_car']['pesq']-.02,
                synthetic_pesq=stats['synthetic_vehicle']['pesq']>=static['synthetic_vehicle']['pesq']-.02,
                stoi=stats['all']['stoi_delta']>=prev['all']['stoi_delta']-.003,
                all_pesq_available=stats['all']['pesq_available']==stats['all']['cases'])
            candidates.append(dict(variant=name,eligible=all(guards.values()),guards=guards,
                score=mean([stats[d]['balanced_pass'] for d in ('recorded_car','synthetic_vehicle')]),summary=stats))
        candidates.sort(key=lambda r:(r['eligible'],r['score'],r['summary']['all']['pesq'],-abs(r['summary']['all']['onset_0_50_db'])),reverse=True)
        ranking[backend]=candidates
    opt.save(output,dict(phases=phases,measurements=len(rows),summary=result,ranking=ranking,selection_split='validation only',standard_compliance=False))
    for b,entries in ranking.items():
        print(b)
        for e in entries[:8]:
            s=e['summary'];print(e['variant'],e['eligible'],round(e['score'],4),'recorded',s['recorded_car']['passed'],'PESQ',round(s['recorded_car']['pesq'],4),'synthetic',s['synthetic_vehicle']['passed'],'PESQ',round(s['synthetic_vehicle']['pesq'],4),'guards',e['guards'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phases',nargs='+');p.add_argument('--output',type=Path,required=True);a=p.parse_args();analyze(a.phases,a.output)
