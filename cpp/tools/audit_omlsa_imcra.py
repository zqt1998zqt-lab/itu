#!/usr/bin/env python3
"""Audit four-way native NS comparison without changing any quality gate."""
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
def average(rs,k):
    values=[r[k] for r in rs if r[k] is not None]
    return statistics.mean(values) if values else None
def speech_pass(r):
    # Transient attenuation is informational; speech gates remain mandatory.
    return not [f for f in r['failures']+r['raw_failures'] if not f.endswith(':attenuation')]

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--root',type=Path,required=True)
    args=a.parse_args();root=args.root.resolve()
    probe=root/'artifacts/release/ns_vehicle_probe';identity=json.loads((root/'artifacts/release/identity.json').read_text())
    assert sha(probe)==identity['binary_sha256']
    for file,h in identity['source_sha256'].items():
        assert sha(root/'artifacts/release/source'/file)==h
        assert sha(P/file)==h
    scopes={};bound={str(probe):sha(probe)};old_total=0
    for scope,count in (('steady',766),('transient',81),('clean',88),('cold',88)):
        folder=root/f'release-{scope}';report=json.loads((folder/'report.json').read_text())
        assert report['state']=='completed' and report['completed']==4*count==report['total']
        batches=json.loads((folder/'identity.json').read_text())
        for p,h in batches['dependencies'].items():assert sha(Path(p))==h
        result=rows(folder/'results.jsonl');bound[str(folder/'results.jsonl')]=sha(folder/'results.jsonl')
        groups={v:{key(r):r for r in result if r['variant']==v} for v in {r['variant'] for r in result}}
        assert len(groups)==4 and all(len(g)==count for g in groups.values())
        keys=set(next(iter(groups.values())))
        assert all(set(g)==keys for g in groups.values())
        for k in keys:
            rr=[g[k] for g in groups.values()]
            for field in ('input_pcm16_sha256','input_crop_samples','clean_sha256'):
                assert len({r[field] for r in rr})==1,(scope,k,field)
            assert len({r['scene']['sample_rate'] for r in rr})==1
            if scope!='cold':assert len({r['input_sha256'] for r in rr})==1
        old=rows(P/f'build/ns-steady-opt-20261018/release-{scope}/results.jsonl')
        old_map={(r['variant'],*key(r)):r for r in old}
        variants={};pairs={}
        for v,g in sorted(groups.items()):
            rr=list(g.values());rates={}
            for rate in (16000,24000,48000):
                rs=[r for r in rr if r['scene']['sample_rate']==rate]
                rates[rate]=dict(n=len(rs),raw_pass=sum(r['raw_quality_pass'] for r in rs),
                                 delay_samples=sorted({r['engine']['delay_samples'] for r in rs}),
                                 core_rate=sorted({r['engine']['core_rate'] for r in rs}))
            failure_noise=Counter();failure_snr=Counter()
            for r in rr:
                if not r['raw_quality_pass']:
                    failure_noise[r['scene']['noise']]+=1;failure_snr[str(r['scene']['snr_db'])]+=1
                assert r['binary_sha256']==sha(probe)
                if v.startswith('omlsa'):
                    e=r['engine'];rate=r['scene']['sample_rate']
                    assert e['core_rate']==rate and e['src_delay_samples']==0 and e['core_ns_delay_samples']==rate//100
                    assert e['imcra_excluded_bins']>0 and e['imcra_noise_updates']>0 and e['imcra_minima_rotations']>0
            variants[v]=dict(n=len(rr),raw_pass=sum(r['raw_quality_pass'] for r in rr),speech_pass=sum(speech_pass(r) for r in rr),
                mean_stoi_improvement=average(rr,'mean_stoi_improvement'),mean_sisdr_improvement_db=average(rr,'mean_sisdr_improvement_db'),
                mean_noise_attenuation_db=average(rr,'mean_noise_attenuation_db'),onset_projection_db=average(rr,'onset_0_50ms_projection_db'),
                clipped_cases=sum(r['engine']['clipped_samples_before_write']>0 for r in rr),rates=rates,
                failure_counts=dict(Counter(f for r in rr for f in r['failures']+r['raw_failures'])),
                failed_by_noise=dict(failure_noise),failed_by_snr=dict(failure_snr))
            if v.startswith('omlsa'):
                for baseline in ('speex_v5_steadydual','webrtc_v5_bounded09'):
                    b=groups[baseline]
                    pairs[v+'__'+baseline]=dict(new_pass=sum(not b[k]['raw_quality_pass'] and g[k]['raw_quality_pass'] for k in keys),
                        new_fail=sum(b[k]['raw_quality_pass'] and not g[k]['raw_quality_pass'] for k in keys),
                        pcm_pairs=len(keys))
            else:
                assert all(old_map[(v,*k)]['output_sha256']==r['output_sha256'] for k,r in g.items()),('legacy output regression',v,scope)
                old_total+=len(g)
        scopes[scope]=dict(unique_parent_cases=count,variants=variants,paired=pairs,input_pcm_verified=True)
    data=dict(state='completed',scopes=scopes,total_runs=sum(4*s['unique_parent_cases'] for s in scopes.values()),
              legacy_identical_outputs=old_total,binary_sha256=sha(probe),bindings=bound,
              engineering_gates_unchanged=True,standard_compliance=False,
              generalization='known regression corpus; development-only parameter selection; not a fresh blind test',
              selection='optional OM-LSA profiles; existing default unchanged',
              transient_policy='raw legacy pass retained; use separate speech_pass; short-event attenuation informational')
    (root/'audit-report.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(state=data['state'],runs=data['total_runs'],legacy_identical_outputs=old_total)))

if __name__=='__main__':main()
