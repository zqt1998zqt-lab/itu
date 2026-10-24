#!/usr/bin/env python3
"""Apply the predeclared split/domain gates to the NS failure-repair candidates."""
import json,statistics,datetime,hashlib
from collections import defaultdict
from pathlib import Path
P=Path(__file__).resolve().parents[2];R=P/'build/ns-failure-repair-v3-20261018'
phases={k:[f'{k}-{x}' for x in ('tune','clean','cold')] for k in ('ablation','repair','floor','gain')}
data={}; identities={}; all_config={}; duplicated=[]
for family,pp in phases.items():
 for phase in pp:
  meta=json.loads((R/phase/'report.json').read_text());assert meta['state']=='completed' and meta['identity_verified'],phase
  identities[phase]=hashlib.sha256((R/phase/'results.jsonl').read_bytes()).hexdigest()
  rows=[json.loads(l) for l in (R/phase/'results.jsonl').open()];scope=phase.split('-')[-1]
  assert rows and all(r['scene']['split'] in ('development','validation') for r in rows)
  for row in rows:
   v=row['variant'];all_config[v]=row['config'];key=(scope,v,row['original_run'],row['scene']['id'])
   if key in data:
    prev=data[key];assert row['failures']==prev['failures'] and row['segments']==prev['segments'] and row['output_sha256']==prev['output_sha256'],key
    duplicated.append(key)
   data[key]=row

def mean(v):
 x=[y for y in v if y is not None];return statistics.mean(x) if x else None

def stats(v,scope,domain=None):
 rr=[row for (sc,n,_,_),row in data.items() if sc==scope and n==v and row['scene']['split']=='validation' and (domain is None or row['domain']==domain)]
 return dict(n=len(rr),passed=sum(x['ns_tradeoff_pass'] for x in rr),pesq=mean([x['pesq_first'] for x in rr]),stoi=mean([x['mean_stoi_improvement'] for x in rr]),pesq_available=sum(x['pesq_first'] is not None for x in rr))
summary={v:dict(recorded=stats(v,'tune','recorded_car'),synthetic=stats(v,'tune','synthetic_vehicle'),all=stats(v,'tune'),clean=stats(v,'clean'),cold=stats(v,'cold')) for v in all_config}
rank={};chosen=[];alternatives=[];status={}
for backend in ('webrtc','speex'):
 base=backend+('_v2_startup' if backend=='webrtc' else '_v2_quality');baseline=summary[base]
 static=[backend+'_static_best']+(['speex_baseline'] if backend=='speex' else [])
 best_pesq={d:max(summary[n][d]['pesq'] for n in static) for d in ('recorded','synthetic')}
 rr=[]
 for v,s in summary.items():
  if not v.startswith(backend+'_'):continue
  gates=dict(recorded_pass=s['recorded']['passed']>=baseline['recorded']['passed'],clean_pass=s['clean']['passed']>=baseline['clean']['passed'],cold_pass=s['cold']['passed']>=baseline['cold']['passed'],recorded_pesq=s['recorded']['pesq']>=best_pesq['recorded']-.02,synthetic_pesq=s['synthetic']['pesq']>=best_pesq['synthetic']-.02,stoi=s['all']['stoi']>=baseline['all']['stoi']-.003,pesq_available=s['all']['pesq_available']==s['all']['n'],nonempty_expected_counts=s['recorded']['n']==142 and s['synthetic']['n']==36 and s['clean']['n']==22 and s['cold']['n']==22)
  rr.append(dict(name=v,eligible=all(gates.values()),guards=gates,summary=s))
 rr.sort(key=lambda x:(x['eligible'],x['summary']['recorded']['passed'],x['summary']['cold']['passed'],x['summary']['clean']['passed'],x['summary']['synthetic']['passed'],x['summary']['all']['pesq']),reverse=True);rank[backend]=rr
 best=next(x for x in rr if x['eligible']);c=dict(all_config[best['name']]);c['source_variant']=c['name'];c['name']=backend+'_v3';c['selection_reason']='Registered v3 selection rule winner; all validation guards passed';c.setdefault('floor_guard',0);c.setdefault('release_ms',0);c.setdefault('quiet_floor_db',-300);c['evidence_root']=str(R);chosen.append(c)
 print(backend,'chosen',best['name'],'recorded',best['summary']['recorded']['passed'],'clean',best['summary']['clean']['passed'],'cold',best['summary']['cold']['passed'],'PESQ',best['summary']['recorded']['pesq'])
 status[backend]=dict(scalar_winner=best['name'],kept_v2_baseline=best['name']==base)
 # Explicitly secondary Pareto profile: measured using validation only; this
 # does not rewrite or override the registered scalar winner.
 clean_best=max([x for x in rr if x['eligible']],key=lambda x:x['summary']['clean']['pesq'])
 if clean_best['name'] not in (base,best['name']):
  c=dict(all_config[clean_best['name']]);c['source_variant']=c['name'];c['name']=backend+'_v3_clean';c.setdefault('floor_guard',0);c.setdefault('release_ms',0);c.setdefault('quiet_floor_db',-300);c['evidence_root']=str(R);c['selection_reason']='Secondary validation clean-PESQ Pareto alternative; not the registered scalar rank winner';alternatives.append(c)
  status[backend]['secondary_clean_alternative']=clean_best['name']
 for row in rr[:5]:print(' ',row['name'],row['eligible'],row['summary']['recorded']['passed'],row['guards'])
baseconfigs=json.loads((P/'build/ns-refinement-v2-20261018/comparison-variants.json').read_text());final=[dict(c,release_ms=0,quiet_floor_db=-300,floor_guard=0,evidence_root=str(R)) for c in baseconfigs if c['name'] in ('webrtc_v2_startup','speex_v2_quality')]+[c for c in chosen if not status[c['backend']]['kept_v2_baseline']]+alternatives
(R/'selection-analysis.json').write_text(json.dumps(dict(ranking=rank,summary=summary,status=status,duplicate_exact_measurements=len(duplicated),unique_measurements=len(data),phase_results_sha256=identities,selection_scope='development and validation only; no holdout ranking'),indent=2)+'\n')
(R/'final-variants.json').write_text(json.dumps(final,indent=2)+'\n');(R/'comparison-variants.json').write_text(json.dumps(final,indent=2)+'\n')
(R/'selection-freeze.json').write_text(json.dumps(dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),contract_sha256=hashlib.sha256((R/'contract.json').read_bytes()).hexdigest(),variants_sha256=hashlib.sha256((R/'final-variants.json').read_bytes()).hexdigest(),binary_sha256=hashlib.sha256((R/'ns_floor_repair_probe').read_bytes()).hexdigest(),selected=chosen,secondary_alternatives=alternatives,status=status,rank_source='development and validation only',default_engine_changed=False,standard_compliance=False),indent=2)+'\n')
