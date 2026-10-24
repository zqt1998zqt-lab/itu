#!/usr/bin/env python3
"""Paired frozen replay summary; never used to retune a winner."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import refine_ns_streams as opt
from analyze_ns_refinement import summary


def main():
    root=opt.ROOT;old=opt.PROJECT/'build/ns-optimization-20261018';out=root/'analysis';out.mkdir(exist_ok=False)
    configs=json.loads((root/'comparison-variants.json').read_text());names=[c['name'] for c in configs]
    allrows={}
    for phase in ('final-vehicle','final-diagnostic','final-clean','final-cold'):
        previous='final-cold-v2' if phase=='final-cold' else phase
        rows=[json.loads(l) for l in (root/phase/'results.jsonl').open()]
        rows += [json.loads(l) for l in (old/previous/'results.jsonl').open()]
        assert len({(r['variant'],r['original_run'],r['scene']['id']) for r in rows})==len(rows)
        allrows[phase]=rows
    # Old frozen measurements acquire their already measured selection PESQ
    # only where available. Perceptual comparisons below always use paired
    # identical scenes, not these uneven historical score availabilities.
    for rows in allrows.values():
        for r in rows:r.setdefault('pesq_first',None)
    tables={}
    for phase,rows in allrows.items():
        tables[phase]={name:summary([r for r in rows if r['variant']==name]) for name in names}
    detailed={}
    for dimension in ('sample_rate','noise','split','snr_db'):
        detailed[dimension]={}
        for key in sorted({r['scene'][dimension] for r in allrows['final-vehicle']},key=str):
            detailed[dimension][str(key)]={name:summary([r for r in allrows['final-vehicle'] if r['variant']==name and r['scene'][dimension]==key]) for name in names}
    recorded={name:summary([r for r in allrows['final-vehicle'] if r['variant']==name and r['domain']=='recorded_car']) for name in names}
    synthetic={name:summary([r for r in allrows['final-vehicle'] if r['variant']==name and r['domain']=='synthetic_vehicle']) for name in names}
    paired={}
    for config in configs:
        if config.get('evidence_root'):continue
        b=config['backend'];n=config['name'];paired[n]={}
        for phase,rows in allrows.items():
            a={(r['original_run'],r['scene']['id']):r for r in rows if r['variant']==n}
            for baseline in (b+'_best',b+'_static_best'):
                z={(r['original_run'],r['scene']['id']):r for r in rows if r['variant']==baseline};assert a.keys()==z.keys()
                rescue=[k for k in a if a[k]['ns_tradeoff_pass'] and not z[k]['ns_tradeoff_pass']]
                regress=[k for k in a if not a[k]['ns_tradeoff_pass'] and z[k]['ns_tradeoff_pass']]
                paired[n][phase+':vs:'+baseline]=dict(rescued=len(rescue),regressed=len(regress),regression_cases=[dict(run=k[0],id=k[1],failures=a[k]['failures']) for k in regress])
    perceptual_rows=[json.loads(l) for l in (root/'perceptual/results.jsonl').open()]
    perceptual={}
    for split in ('validation','holdout'):
        perceptual[split]={}
        for domain in ('recorded_car','synthetic_vehicle'):
            g=[r for r in perceptual_rows if r['split']==split and r['domain']==domain];table={}
            for name in ['input']+names:
                vals=[r['scores'][name] for r in g]
                def avg(v):return float(np.mean(v)) if v else None
                table[name]=dict(cases=len(vals),pesq_available=sum(v['pesq_wb_16k'] is not None for v in vals),pesq=avg([v['pesq_wb_16k'] for v in vals if v['pesq_wb_16k'] is not None]),dnsmos={k:avg([v['dnsmos']['poly'][k] for v in vals if v['dnsmos']['available']]) for k in ('SIG','BAK','OVRL')})
            perceptual[split][domain]=table
    stream=json.loads((root/'streams/report.json').read_text());stream_summary=[]
    for r in stream['rows']:
        stream_summary.append({k:r[k] for k in ('variant','rate','mode','paced_pcm_bit_exact','continuous_vs_reset_different_samples','frames','transitions') } | dict(delay_ms=r['continuous']['delay_samples']*1000/r['rate'],p99_ms=r['continuous']['p99_process_ms'],over_10ms=r['continuous']['over_10ms'],paced_deadline_misses=r['paced']['paced_deadline_misses']))
    output=dict(recorded_car=recorded,synthetic_vehicle=synthetic,tables=tables,by_dimension=detailed,paired=paired,perceptual=perceptual,streams=stream_summary,standard_compliance=False)
    opt.save(out/'summary.json',output)
    print('recorded car:')
    for n,t in recorded.items():print(n,t['passed'],'/',t['cases'])
    print('holdout recorded perception:')
    for n,t in perceptual['holdout']['recorded_car'].items():print(n,t)

if __name__=='__main__':main()
