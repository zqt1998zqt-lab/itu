#!/usr/bin/env python3
"""Independent final output/config/hash and baseline-equivalence audit."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,hashlib,csv
from pathlib import Path
import numpy as np
import soundfile as sf
import repair_ns_floor_v3 as opt
R=opt.ROOT;P=opt.PROJECT;old=P/'build/ns-refinement-v2-20261018';configs={c['name']:c for c in json.loads((R/'final-variants.json').read_text())};baseline={'webrtc_v2_startup','speex_v2_quality'};result=[];baseline_checks=[];rescored=[]
for phase,scope in [('final-vehicle','vehicle'),('final-clean','clean'),('final-cold','cold'),('final-diagnostic','diagnostic')]:
 meta=json.loads((R/phase/'report.json').read_text());identity=json.loads((R/phase/'identity.json').read_text());assert meta['state']=='completed' and meta['identity_verified'];assert meta['completed']==len(configs)*len(opt.sources(scope))
 assert all(opt.paired.replay.ctl.ev.digest(Path(p))==h for p,h in identity['dependencies'].items())
 rows=[json.loads(l) for l in (R/phase/'results.jsonl').open()];seen=set();orig={(run,r['scene']['id']):r for run,r in opt.sources(scope)}
 for row in rows:
  run,sid=row['original_run'],row['scene']['id'];c=configs[row['variant']];key=(row['variant'],run,sid);assert key not in seen;seen.add(key);f=R/phase/row['variant']/run/sid;e=json.loads((f/'result.json').read_text());iv=json.loads((f/'invocation.json').read_text());assert iv['exit_code']==0
  for field,param in [('backend','backend'),('strength','strength'),('dry_max','dry_max'),('snr_threshold_db','snr_threshold_db'),('noise_gain','noise_gain'),('startup_ms','startup_ms'),('control_policy','policy'),('mild_strength','mild_strength'),('slew_ms','slew_ms'),('repair_release_ms','release_ms'),('quiet_floor_db','quiet_floor_db'),('floor_guard','floor_guard')]:assert e[field]==c[param],(key,field)
  op=f/'result.after.wav';ip=Path(iv['argv'][3]);assert opt.paired.replay.ctl.ev.digest(op)==row['output_sha256'];assert opt.paired.replay.ctl.ev.digest(ip)==row['input_sha256'];assert opt.paired.replay.ctl.ev.digest(opt.PROBE)==row['binary_sha256'];assert row['wav_retained'];a,sr=sf.read(op);assert sr==row['scene']['sample_rate'] and len(a)==e['samples']+e['delay_samples'] and np.isfinite(a).all()
  if row['variant'] in baseline:
   oldf=old/phase/row['variant']/run/sid;o=json.loads((oldf/'metrics.json').read_text());assert row['output_sha256']==o['output_sha256'];assert row['failures']==o['failures'] and row['segments']==o['segments'];pcm_equal=True
   if scope=='cold':
    assert opt.paired.replay.ctl.ev.digest(oldf/'cold-input.wav')==o['input_sha256'];x,xr=sf.read(ip,dtype='float32');y,yr=sf.read(oldf/'cold-input.wav',dtype='float32');pcm_equal=xr==yr and np.array_equal(x,y);assert pcm_equal
   baseline_checks.append(dict(phase=phase,variant=row['variant'],run=run,id=sid,output_hash_equal=True,input_pcm_equal=pcm_equal))
  # Rescore one metadata-selected scene/rate/split/variant/phase, including nulls.
  token=(phase,row['variant'],sr,row['scene']['split'])
  if token not in {tuple(x['token']) for x in rescored}:
   source=P/'build'/run/'cases'/sid;clean,_=sf.read(source/'clean.wav');before,_=sf.read(ip)
   if scope=='cold':
    hop=sr//100;pwr=np.mean(clean[:len(clean)//hop*hop].reshape(-1,hop)**2,axis=1);offset=int(np.flatnonzero(pwr>=10**(-65/10))[0])*hop;clean=clean[offset:]
   after=a[e['delay_samples']:];before=np.clip(np.rint(before*32768),-32768,32767)/32768
   for label,(lo,hi) in row['scene']['windows'].items():
    if label not in row['segments']:continue
    sl=slice(round(lo*sr),min(len(clean),round(hi*sr)));v=row['segments'][label]
    if label.startswith('near'):
     qa=opt.quality(clean[sl],after[sl],sr);qb=opt.quality(clean[sl],before[sl],sr)
     for got,stored in ((qa,v['after']),(qb,v['before'])):
      for k,val in got.items():assert (val is None and stored[k] is None) or (val is not None and stored[k] is not None and abs(val-stored[k])<1e-10),(token,label,k)
    elif label.startswith('noise_only'):
     got=opt.paired.replay.ctl.ev.attenuation_measurement(before[sl],after[sl],pcm16=True);assert got==v,(token,label)
   rescored.append(dict(token=token,run=run,id=sid,passed=True))
 result.append(dict(phase=phase,rows=len(rows),unique_keys=len(seen),identity_verified=True))
# Required independent post-freeze stages.
for n in ('perceptual','streams','native24','adc-stress'):
 m=json.loads((R/n/'report.json').read_text());assert m['state']=='completed'
 if 'identity_verified' in m:assert m['identity_verified']
 if n=='streams':assert all(x['paced_pcm_bit_exact'] and x['continuous']['over_10ms']==0 and x['paced']['paced_deadline_misses']==0 for x in m['rows'])
opt.save(R/'final-audit.json',dict(passed=True,phases=result,baseline_exact_pairs=len(baseline_checks),rescored=rescored,scope='all final retained WAV hashes/config/shape, exact frozen-baseline reproduction and metadata-stratified independent metric recomputation; NS-only, not full VCP',standard_compliance=False))
opt.save(R/'baseline-reproduction.json',dict(passed=True,pairs=len(baseline_checks),checks=baseline_checks));print('AUDIT PASS',result,'baseline_pairs',len(baseline_checks),'independent_rescores',len(rescored))
