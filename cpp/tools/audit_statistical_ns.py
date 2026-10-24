#!/usr/bin/env python3
"""Paired audit of statistical NS experiments and retained frozen baselines."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

P=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return [json.loads(line) for line in p.open()]
def key(r):return r['original_run'],r['scene']['id']
def speech_pass(r):return not any(not f.endswith(':attenuation') for f in r['failures']+r['raw_failures'])
def mean(rs,k):
    v=[r[k] for r in rs if r.get(k) is not None]
    return statistics.mean(v) if v else None
def summary(rs):
    result=dict(n=len(rs),raw_pass=sum(r['raw_quality_pass'] for r in rs),
        speech_pass=sum(speech_pass(r) for r in rs),
        failure_counts=dict(Counter(f for r in rs for f in r['failures']+r['raw_failures'])),
        clipped_cases=sum(r['engine']['clipped_samples_before_write']>0 for r in rs))
    for k in ('mean_noise_attenuation_db','mean_sisdr_improvement_db','mean_stoi_improvement','onset_0_50ms_projection_db','pesq_first'):
        result[k]=mean(rs,k)
    for name,get in [('rate',lambda r:str(r['scene']['sample_rate'])),('noise',lambda r:r['scene']['noise']),
                     ('split',lambda r:r['scene']['split']),('snr',lambda r:str(r['scene']['snr_db']))]:
        result['by_'+name]={v:dict(n=len(g),raw_pass=sum(r['raw_quality_pass'] for r in g),speech_pass=sum(speech_pass(r) for r in g))
            for v in sorted({get(r) for r in rs}) if (g:=[r for r in rs if get(r)==v])}
    result['stoi_unavailable_windows']=sum(s.get('after',{}).get('stoi') is None for r in rs for k,s in r['segments'].items() if k.startswith('near'))
    result['delay_samples_by_rate']={str(rate):sorted({r['engine']['delay_samples'] for r in rs if r['scene']['sample_rate']==rate})
        for rate in sorted({r['scene']['sample_rate'] for r in rs})}
    return result

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--root',type=Path,required=True)
    a.add_argument('--plan',type=Path,required=True);args=a.parse_args();root=args.root.resolve()
    plan=json.loads(args.plan.read_text());baseline=Path(plan['baseline_root'])
    if not baseline.is_absolute():baseline=P/baseline
    baseline_index=json.loads((baseline/'audit-report.json').read_text())
    bindings={str(args.plan.resolve()):sha(args.plan)};scopes={};total=0;verified=0;negative_controls=0
    for scope,n in (('steady',766),('transient',81),('clean',88),('cold',88)):
        old=baseline/f'release-{scope}'/'results.jsonl'
        assert baseline_index['bindings'][str(old.resolve())]==sha(old)
        combined=rows(old);bindings[str(old)]=sha(old)
        phase_names=plan['scopes'][scope];new_names=set()
        for phase in phase_names:
            folder=root/phase;report=json.loads((folder/'report.json').read_text())
            assert report['state']=='completed' and report['completed']==report['total']>0
            identity=json.loads((folder/'identity.json').read_text())
            for path,h in identity['dependencies'].items():assert sha(Path(path))==h,(phase,path)
            rr=rows(folder/'results.jsonl');assert len(rr)==report['total']
            assert len({key(r) for r in rr})==n
            binary={r['binary_sha256'] for r in rr};assert len(binary)==1
            probe=Path(next(p for p in identity['dependencies'] if p.endswith('/ns_vehicle_probe')))
            assert binary=={sha(probe)}
            new_names.update(r['variant'] for r in rr);total+=len(rr);combined+=rr
            bindings[str(folder/'results.jsonl')]=sha(folder/'results.jsonl')
        groups={v:{key(r):r for r in combined if r['variant']==v} for v in {r['variant'] for r in combined}}
        assert all(len(g)==n for g in groups.values()),'duplicate or missing rows'
        keys=set(next(iter(groups.values())))
        assert all(set(g)==keys for g in groups.values())
        for k in keys:
            rr=[g[k] for g in groups.values()]
            for field in ('input_pcm16_sha256','input_crop_samples','clean_sha256'):
                assert len({r[field] for r in rr})==1,(scope,k,field)
            assert len({r['scene']['sample_rate'] for r in rr})==1
            verified+=1
        pairs={}
        for name in sorted(new_names):
            g=groups[name]
            for base in ('omlsa_imcra_balanced','omlsa_imcra_mild','webrtc_v5_bounded09','speex_v5_steadydual'):
                b=groups[base];changed=sum(g[k]['output_sha256']!=b[k]['output_sha256'] for k in keys)
                negative_controls+=changed
                pairs[name+'__'+base]=dict(n=n,new_pass=sum(not b[k]['raw_quality_pass'] and g[k]['raw_quality_pass'] for k in keys),
                    new_fail=sum(b[k]['raw_quality_pass'] and not g[k]['raw_quality_pass'] for k in keys),different_output_hashes=changed)
            for r in g.values():
                e=r['engine'];rate=r['scene']['sample_rate']
                assert e['io_rate']==e['core_rate']==rate and e['src_delay_samples']==0 and e['core_ns_delay_samples']==rate//100
                assert e['backend']==r['config']['backend']
                assert e['imcra_noise_updates']>0 and e['imcra_minima_rotations']>0
        scopes[scope]=dict(unique_parent_cases=n,variants={v:summary(list(g.values())) for v,g in sorted(groups.items())},paired=pairs)
    assert negative_controls>0
    result=dict(state='completed',new_comparison_runs=total,verified_pcm_parent_scope_pairs=verified,scopes=scopes,
        bindings=bindings,engineering_gates_unchanged=True,standard_compliance=False,default_backend_changed=False,
        corpus='847 unique noisy parent scenes; clean/cold are 88+88 derived diagnostics; known regression, not blind validation',
        transient_policy='speech_pass excludes attenuation-only failures; raw_quality_pass retains all legacy gates')
    (root/'audit-report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(state='completed',new_runs=total,pcm_pairs=verified)))

if __name__=='__main__':main()
