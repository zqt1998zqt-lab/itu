#!/usr/bin/env python3
"""Freeze NS winners using validation only, before final replay is started."""
from datetime import datetime, timezone
import json
import optimize_ns_backends as opt


def key(item):
    name, r = item
    return (-r['balanced_pass']['recorded_car'], -r['balanced_pass']['synthetic_vehicle'],
            -r['mean_stoi_improvement'], abs(r['onset_0_50ms_projection_db']), name)


def main():
    phases=['round1','refine-speex','refine-webrtc']
    for phase in phases:
        report=json.loads((opt.ROOT/phase/'report.json').read_text())
        if report['state']!='completed' or not report['identity_verified']:raise ValueError('unverified tuning phase')
    if (opt.ROOT/'final-vehicle').exists() or (opt.ROOT/'final-variants.json').exists():raise ValueError('winner selection must precede final replay')
    summary=json.loads((opt.ROOT/'summary-round1-refine-speex-refine-webrtc.json').read_text())
    if summary['holdout']:raise ValueError('holdout present in tuning')
    validation=summary['validation'];chosen=[];ranking={}
    for backend in ('webrtc','speex'):
        ranked=sorted([(n,r) for n,r in validation.items() if r['config']['backend']==backend],key=key)
        if len(ranked)!=32 or any(r['cases']!=178 for _,r in ranked):raise ValueError('unbalanced candidate budget/validation coverage')
        raw=[x for x in ranked if x[1]['config']['dry_max']==0 and x[1]['config']['noise_gain']==1]
        best_name,best=ranked[0];static_name,static=raw[0]
        baseline_name='webrtc_0_raw' if backend=='webrtc' else 'speex_12_raw'
        for label,source,r in [('baseline',baseline_name,validation[baseline_name]),('static_best',static_name,static),('best',best_name,best)]:
            c=dict(r['config']);c.update(name=backend+'_'+label,source_variant=source);chosen.append(c)
        ranking[backend]=dict(winner=best_name,best_static=static_name,top6=[dict(variant=n,**r) for n,r in ranked[:6]])
    opt.save(opt.ROOT/'final-variants.json',chosen)
    opt.save(opt.ROOT/'winner-selection.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),selection_split='validation only',
        cases_per_candidate=262,validation_per_candidate=178,candidates_per_backend=32,historical_holdout_used=False,ranking=ranking,
        rule='recorded-car rate/noise-cell pass; synthetic cell pass; STOI delta; onset projection closest to 0; lexical name for exact ties'))
    print(json.dumps(chosen,indent=2))


if __name__=='__main__':main()
