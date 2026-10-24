#!/usr/bin/env python3
"""Summarize paired NS runs without modifying stored gates or measurements."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path

import numpy as np


def load(path):
    return [json.loads(line) for line in path.open()]


def stats(values):
    a=np.asarray([x for x in values if x is not None and np.isfinite(x)],dtype=float)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),p05=float(np.percentile(a,5)),p95=float(np.percentile(a,95))) if len(a) else dict(n=0)


def quality_pass(row):
    return all(g['status']=='pass' or (g['status']=='unavailable' and g['optional'])
               for g in row['acceptance']['gates'] if g['segment']!='runtime')


def features(row):
    segments=row.get('metrics',row)['segments']
    names=[n for n in ('near_only','near_late','double_talk') if n in segments]
    result={}
    for key in ('sisdr_db','stoi','projection_gain_db'):
        a=[segments[n]['after'].get(key) for n in names];a=[v for v in a if v is not None]
        result[key]=float(np.mean(a)) if a else None
    for key in ('sisdr_improvement_db','stoi_improvement'):
        a=[segments[n].get(key) for n in names];a=[v for v in a if v is not None]
        result[key]=float(np.mean(a)) if a else None
    # The legacy field spelling is sisdr_improvement_db in isolated runs,
    # and sisdr_improvement_db in the V5 segment records as well.
    for name in ('noise_only','far_only','noise_only_before_step','noise_only_after_step'):
        s=segments.get(name,{})
        result[name+'_attenuation_db']=s.get('total_attenuation_db') if s.get('total_attenuation_db') is not None else s.get('attenuation_lower_bound_db')
    return result


def cluster_ci(pairs,key):
    clusters=defaultdict(list)
    for a,b in pairs:
        av,bv=features(a).get(key),features(b).get(key)
        if av is None or bv is None:continue
        s=a['scene'];group=(s['sample_rate'],s['noise'],s['split'],s['near_speaker'])
        clusters[group].append(bv-av)
    if not clusters:return None
    sums=np.array([sum(v) for v in clusters.values()]);counts=np.array([len(v) for v in clusters.values()])
    rng=np.random.default_rng(20261018);draw=rng.integers(len(sums),size=(2000,len(sums)))
    boot=sums[draw].sum(axis=1)/counts[draw].sum(axis=1)
    return dict(mean_speex_minus_webrtc=float(sums.sum()/counts.sum()),clusters=len(sums),
                paired_cases=int(counts.sum()),bootstrap_95ci=np.percentile(boot,[2.5,97.5]).tolist(),
                cluster='rate/noise/split/near_speaker; 2000 seeded cluster resamples')


def summary(rows,full):
    result={}
    for variant in sorted({r['variant'] for r in rows}):
        selected=[r for r in rows if r['variant']==variant]
        f=[features(r) for r in selected]
        item=dict(cases=len(selected),metrics={k:stats(v[k] for v in f) for k in f[0]})
        if full:
            item.update(all_gate_pass=sum(r['acceptance']['passed'] for r in selected),
                        quality_level_pass=sum(quality_pass(r) for r in selected))
            failed=Counter((g['segment'],g['metric']) for r in selected for g in r['acceptance']['gates'] if g['status']=='fail')
            item['failed_gates']=[dict(segment=a,metric=b,count=n) for (a,b),n in failed.most_common()]
        else:
            item['frames']={k:stats(r['frames'][k] for r in selected) for k in
                ('active_projection_p05_db','active_projection_median_db','active_projection_below_minus6_percent','projection_step_p95_db')}
            item['onset_projection_db']={label:stats(o['windows'][label]['projection_gain_db'] for r in selected for o in r['onsets'])
                for label in ('0_50ms','50_150ms','150_500ms')}
        result[variant]=item
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    root=args.root.resolve();out=root/'analysis';out.mkdir(exist_ok=False)
    full=load(root/'full-defaults/results.jsonl');isolated=load(root/'isolation-strengths/results.jsonl');sweep=load(root/'vcp-strengths/results.jsonl')
    for name in ('full-defaults','isolation-strengths','vcp-strengths'):
        report=json.loads((root/name/'report.json').read_text())
        if report['state']!='completed' or not report['identity_verified']:raise ValueError('incomplete or unverified run')
    indexed={}
    for row in full:
        key=(row['original_run'],row['scene']['id']);indexed.setdefault(key,{})[row['variant']]=row
    if len(indexed)!=4412 or any(set(v)!= {'webrtc0','speex12'} for v in indexed.values()):raise ValueError('incomplete pairing')
    pairs=[(v['webrtc0'],v['speex12']) for v in indexed.values()]
    transitions=Counter(('pass' if quality_pass(a) else 'fail')+'->'+('pass' if quality_pass(b) else 'fail') for a,b in pairs)
    grouped={}
    for field in ('sample_rate','noise','snr_db','split','mics','family'):
        grouped[field]={str(value):summary([r for r in full if r['scene'].get(field)==value],True)
                        for value in sorted({r['scene'].get(field) for r in full},key=str)}
    feature_keys=list(features(full[0]));deltas={k:cluster_ci(pairs,k) for k in feature_keys}
    ranked=[]
    for a,b in pairs:
        af,bf=features(a),features(b)
        delta=bf['sisdr_db']-af['sisdr_db'] if af['sisdr_db'] is not None and bf['sisdr_db'] is not None else None
        ranked.append(dict(run=a['original_run'],id=a['scene']['id'],scene={k:a['scene'].get(k) for k in
            ('sample_rate','noise','snr_db','mics','family','near_speaker')},webrtc=af,speex=bf,delta_sisdr_db=delta,
            webrtc_quality_pass=quality_pass(a),speex_quality_pass=quality_pass(b)))
    ranked.sort(key=lambda r: r['delta_sisdr_db'] if r['delta_sisdr_db'] is not None else 0.)
    report=dict(full=summary(full,True),quality_level_transitions_webrtc_to_speex=transitions,
                grouped=grouped,paired_cluster_differences=deltas,isolation=summary(isolated,False),
                vcp_strengths=summary(sweep,True),largest_speex_losses=ranked[:20],largest_speex_gains=ranked[-20:],
                quality_level_scope='all frozen non-runtime gates; original all-gate results retained separately',
                primary_quality_aggregation='equal mean of near_only/near_late/double_talk per case, then equal mean of cases',
                caveat='exploratory strength sweep includes historical holdout; not a new blind validation',standard_compliance=False)
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(14,10));colors={'webrtc0':'#2563eb','speex12':'#e46a24'}
    rates=(16000,24000,48000);x=np.arange(3)
    for i,v in enumerate(colors):
        values=[100*grouped['sample_rate'][str(rate)][v]['quality_level_pass']/grouped['sample_rate'][str(rate)][v]['cases'] for rate in rates]
        axes[0,0].bar(x+(i-.5)*.32,values,.32,label=v,color=colors[v])
    axes[0,0].set(xticks=x,xticklabels=['16k','24k','48k'],ylabel='Quality + level pass (%)',title='Full VCP: fixed engineering gates');axes[0,0].legend()
    for v in colors:
        snrs=sorted(int(k) for k in grouped['snr_db'] if k!='None')
        y=[grouped['snr_db'][str(k)][v]['metrics']['sisdr_improvement_db'].get('mean',np.nan) for k in snrs]
        axes[0,1].plot(snrs,y,label=v,color=colors[v])
    axes[0,1].set(xlabel='Input SNR (dB)',ylabel='Mean speech SI-SDR improvement (dB)',title='Full VCP: all available SNRs');axes[0,1].legend()
    for v,item in report['isolation'].items():
        if v in ('bypass','filterbank','speex12native24'):continue
        a=item['metrics']['noise_only_attenuation_db'].get('mean');b=item['metrics']['sisdr_improvement_db'].get('mean')
        if a is not None and b is not None:
            axes[1,0].scatter(a,b,c='#2563eb' if v.startswith('webrtc') else '#e46a24');axes[1,0].annotate(v,(a,b),fontsize=8)
    axes[1,0].set(xlabel='Mean noise-only attenuation (dB)',ylabel='Mean speech SI-SDR improvement (dB)',title='Isolated NS: strength/quality tradeoff')
    for v in colors:
        sel=[r for r in isolated if r['variant']==v]
        vals=[np.mean([o['windows'][label]['projection_gain_db'] for r in sel for o in r['onsets'] if o['windows'][label]['projection_gain_db'] is not None])
              for label in ('0_50ms','50_150ms','150_500ms')]
        axes[1,1].plot(range(3),vals,'o-',label=v,color=colors[v])
    axes[1,1].set(xticks=range(3),xticklabels=['0-50 ms','50-150 ms','150-500 ms'],ylabel='Mean target projection (dB)',title='Isolated NS: oracle-evaluated speech onset');axes[1,1].legend()
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle('Speex NS vs WebRTC NS | same frozen PCM | simulation only');fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(full=report['full'],transitions=transitions),allow_nan=False))


if __name__=='__main__':main()
