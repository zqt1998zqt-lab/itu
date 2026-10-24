#!/usr/bin/env python3
"""Layered, paired reporting for the frozen NS optimization experiment."""
import json
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import optimize_ns_backends as opt
from verify_optimized_ns import frame_summary


def mean(values):
    a=[x for x in values if x is not None]
    return float(np.mean(a)) if a else None


def summarize(rows):
    result=opt.aggregate(rows)
    for name, group in ((v,[r for r in rows if r['variant']==v]) for v in result):
        noise=[r['mean_noise_attenuation_db'] for r in group if r['mean_noise_attenuation_db'] is not None]
        onsets=[o['0_50ms'] for r in group for o in r['onsets'] if o['0_50ms'] is not None]
        result[name].update(noise_p10_db=float(np.percentile(noise,10)) if noise else None,
            onset_events=len(onsets),onset_p10_db=float(np.percentile(onsets,10)) if onsets else None,
            strict_onset_case_pass=sum(all(o['0_50ms'] is not None and o['0_50ms']>=-3 for o in r['onsets']) for r in group),
            stoi_loss_case_count=sum(any(':stoi' in f for f in r['failures']) for r in group),
            noise_failure_case_count=sum(any(':attenuation' in f for f in r['failures']) for r in group),
            projection_failure_case_count=sum(any(':projection_gain' in f for f in r['failures']) for r in group))
    return result


def main():
    out=opt.ROOT/'analysis-v2';out.mkdir(exist_ok=False)
    cold_identity=json.loads((opt.ROOT/'cold-input-identity.json').read_text())
    if not cold_identity['passed'] or not cold_identity['all_decoded_float32_pcm_exact']:raise ValueError('cold PCM identity failed')
    canonical={(x['variant'],x['run'],x['id']):x['canonical_float32_pcm_sha256'] for x in cold_identity['entries']}
    def same_input(phase,x,y):
        if x['clean_sha256']!=y['clean_sha256']:return False
        if x['input_sha256']==y['input_sha256']:return True
        if phase!='final-cold-v2':return False
        # Container timestamps differ; every other byte and decoded float32
        # sample is independently verified in cold-input-identity.json.
        return canonical[(x['variant'],x['original_run'],x['scene']['id'])]==canonical[(y['variant'],y['original_run'],y['scene']['id'])]
    all_rows=[];summaries={};paired={}
    for phase in ('final-vehicle','final-diagnostic','final-clean','final-cold-v2'):
        rows=[json.loads(l) for l in (opt.ROOT/phase/'results.jsonl').open()]
        report=json.loads((opt.ROOT/phase/'report.json').read_text())
        if report['state']!='completed' or not report['identity_verified']:raise ValueError('phase not frozen/completed')
        all_rows+=rows
        groups={'all':rows}
        for r in rows:
            s=r['scene'];snr=s['snr_db'];band='lt0' if snr<0 else '0to9' if snr<10 else '10to19' if snr<20 else 'ge20'
            for label in (r['domain'],r['domain']+':'+s['split'],r['domain']+':'+str(s['sample_rate']),r['domain']+':'+s['noise'],r['domain']+':snr:'+band):
                groups.setdefault(label,[]).append(r)
        summaries[phase]={label:summarize(group) for label,group in groups.items()}
        index={(r['variant'],r['original_run'],r['scene']['id']):r for r in rows}
        for backend in ('webrtc','speex'):
            best=backend+'_best';baseline=backend+'_baseline'
            a=[r for r in rows if r['variant']==best]
            for domain in ('all','recorded_car','synthetic_vehicle','diagnostic'):
                pairs=[(index[(baseline,r['original_run'],r['scene']['id'])],r) for r in a if domain=='all' or r['domain']==domain]
                if not pairs:continue
                if not all(same_input(phase,x,y) for x,y in pairs):raise ValueError('paired input identity differs')
                paired[phase+':'+backend+':'+domain]=dict(cases=len(pairs),
                    passed_before=sum(x['ns_tradeoff_pass'] for x,y in pairs),passed_after=sum(y['ns_tradeoff_pass'] for x,y in pairs),
                    rescued=sum(not x['ns_tradeoff_pass'] and y['ns_tradeoff_pass'] for x,y in pairs),regressed=sum(x['ns_tradeoff_pass'] and not y['ns_tradeoff_pass'] for x,y in pairs),
                    delta_stoi=mean(y['mean_stoi_improvement']-x['mean_stoi_improvement'] for x,y in pairs),
                    delta_noise_db=mean(y['mean_noise_attenuation_db']-x['mean_noise_attenuation_db'] for x,y in pairs if y['mean_noise_attenuation_db'] is not None and x['mean_noise_attenuation_db'] is not None),
                    delta_onset_projection_db=mean(y['onset_0_50ms_projection_db']-x['onset_0_50ms_projection_db'] for x,y in pairs))
    # The two selected winners are scored on the same sources and schedules.
    vehicle=[r for r in all_rows if r['scope']=='vehicle'];names={r['variant'] for r in vehicle}
    keys={n:{(r['original_run'],r['scene']['id']) for r in vehicle if r['variant']==n} for n in names}
    if len({frozenset(k) for k in keys.values()})!=1:raise ValueError('final variants do not cover identical cases')
    opt.save(out/'summary.json',dict(scopes=summaries,paired_improvements=paired,identical_final_case_keys=True,standard_compliance=False))
    frame_rows=[]
    with (out/'frames.jsonl').open('x') as stream:
        for row in vehicle:
            if row['variant'] not in ('webrtc_best','speex_best','webrtc_baseline','speex_baseline'):continue
            run=row['original_run'];s=row['scene'];rate=s['sample_rate']
            origin=opt.PROJECT/'build'/run/'cases'/s['id']
            clean,_=sf.read(origin/'clean.wav');before,_=sf.read(origin/'before.wav')
            p=opt.ROOT/'final-vehicle'/row['variant']/run/s['id']/'result.after.wav'
            if opt.paired.replay.ctl.ev.digest(p)!=row['output_sha256']:raise ValueError('frame output changed')
            a,_=sf.read(p);d=row['engine']['delay_samples'];a=a[d:d+len(clean)]
            f=dict(variant=row['variant'],run=run,id=s['id'],rate=rate,domain=row['domain'],split=s['split'],
                frame_metrics=frame_summary(clean,before,a,rate))
            frame_rows.append(f);stream.write(json.dumps(f,allow_nan=False)+'\n')
    frame_groups=defaultdict(list)
    for f in frame_rows:
        for scope in (f['domain'],f['domain']+':'+str(f['rate']),f['domain']+':'+f['split']):
            frame_groups[(scope,f['variant'])].append(f['frame_metrics'])
    opt.save(out/'frames-summary.json',[dict(scope=scope,variant=variant,cases=len(g),**{k:mean(x[k] for x in g) for k in ('projection_p05_db','projection_median_db','active_below_minus6_percent','projection_step_p95_db')}) for (scope,variant),g in sorted(frame_groups.items())])
    perceptual=[json.loads(l) for l in (opt.ROOT/'perceptual/results.jsonl').open()]
    p_summary={}
    for domain in ('recorded_car','synthetic_vehicle'):
        group=[r for r in perceptual if r['domain']==domain]
        p_summary[domain]={}
        for name in group[0]['scores']:
            scores=[r['scores'][name] for r in group];available=[s for s in scores if s['dnsmos']['available']]
            p_summary[domain][name]=dict(cases=len(group),dnsmos_available=len(available),pesq_available=sum(s['pesq_wb_16k'] is not None for s in scores),
                dnsmos={k:mean(s['dnsmos']['poly'][k] for s in available) for k in ('SIG','BAK','OVRL')},pesq=mean(s['pesq_wb_16k'] for s in scores))
    opt.save(out/'perceptual-summary.json',p_summary)
    opt.save(out/'report.json',dict(state='completed',final_variants=len(names),vehicle_cases_per_variant=len(next(iter(keys.values()))),frame_rows=len(frame_rows),historical_holdout_cases=len(perceptual),standard_compliance=False))
    print(json.dumps(paired,indent=2))


if __name__=='__main__':main()
