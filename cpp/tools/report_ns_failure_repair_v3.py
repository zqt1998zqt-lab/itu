#!/usr/bin/env python3
"""Full frozen NS-only results; separate rank winners from Pareto profiles."""
import json,statistics
from collections import defaultdict,Counter
from pathlib import Path
P=Path(__file__).resolve().parents[2];R=P/'build/ns-failure-repair-v3-20261018';D=R/'analysis';D.mkdir(exist_ok=True)
phases=['final-vehicle','final-clean','final-cold','final-diagnostic'];result={};pairs={};bad={};coverage={}
def mean(x):
 z=[v for v in x if v is not None];return statistics.mean(z) if z else None
for phase in phases:
 m=json.loads((R/phase/'report.json').read_text());assert m['state']=='completed' and m['identity_verified'];rows=[json.loads(l) for l in (R/phase/'results.jsonl').open()];vs=sorted({x['variant'] for x in rows});result[phase]={};bad[phase]={}
 for v in vs:
  rr=[x for x in rows if x['variant']==v]
  def summarize(g):return dict(n=len(g),passed=sum(x['ns_tradeoff_pass'] for x in g),raw_passed=sum(x['raw_quality_pass'] for x in g),pesq=mean([x['pesq_first'] for x in g]),stoi_delta=mean([x['mean_stoi_improvement'] for x in g]),noise_attenuation_db=mean([x['mean_noise_attenuation_db'] for x in g]),failures=dict(Counter(f for x in g for f in x['failures'])))
  result[phase][v]=dict(all=summarize(rr),**{d:summarize([x for x in rr if x['domain']==d]) for d in sorted({x['domain'] for x in rr})},rates={str(rate):summarize([x for x in rr if x['scene']['sample_rate']==rate]) for rate in (16000,24000,48000)},splits={split:summarize([x for x in rr if x['scene']['split']==split]) for split in ('development','validation','holdout')})
  bad[phase][v]=[dict(run=x['original_run'],id=x['scene']['id'],sample_rate=x['scene']['sample_rate'],split=x['scene']['split'],noise=x['scene']['noise'],snr_db=x['scene']['snr_db'],failures=x['failures'],segments=x['segments']) for x in rr if x['failures']]
  if v in ('webrtc_v2_startup','speex_v2_quality'):continue
  b='webrtc_v2_startup' if v.startswith('webrtc') else 'speex_v2_quality';base={(x['original_run'],x['scene']['id']):x for x in rows if x['variant']==b};better=[];worse=[];changed=0
  for x in rr:
   a=base[(x['original_run'],x['scene']['id'])];assert a['scene']==x['scene'] and a['clean_sha256']==x['clean_sha256'];changed+=a['output_sha256']!=x['output_sha256']
   if phase!='final-cold':assert a['input_sha256']==x['input_sha256']
   if x['ns_tradeoff_pass'] and not a['ns_tradeoff_pass']:better.append(dict(run=x['original_run'],id=x['scene']['id']))
   if a['ns_tradeoff_pass'] and not x['ns_tradeoff_pass']:worse.append(dict(run=x['original_run'],id=x['scene']['id'],failures=x['failures']))
  pairs.setdefault(phase,{})[v]=dict(paired=len(rr),new_pass=better,new_fail=worse,different_output_hashes=changed)
 coverage[phase]=dict(executions=len(rows),variants=len(vs))
(R/'failure-attribution.json').write_text(json.dumps(bad,indent=2)+'\n');(D/'summary.json').write_text(json.dumps(dict(results=result,paired=pairs,coverage=coverage,standard_compliance=False),indent=2)+'\n')
for phase,vs in result.items():
 print(phase)
 for v,ss in vs.items():print(v,[(d,s['passed'],s['n'],round(s['pesq'],4) if s['pesq'] is not None else None) for d,s in ss.items() if d in ('all','recorded_car','synthetic_vehicle','diagnostic')])
