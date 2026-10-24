#!/usr/bin/env python3
"""Post-freeze ADC-noise stress: no labels or targets enter the C++ binary."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,hashlib
import numpy as np
import soundfile as sf
import verify_ns_failure_repair_v3 as verify
opt=verify.opt;R=opt.ROOT;D=R/'adc-stress';D.mkdir(exist_ok=False)
configs=json.loads((R/'comparison-variants.json').read_text());selected={};out=[]
for run,r in opt.sources('clean-tune'):
 if r['scene']['split']=='development':selected.setdefault(r['scene']['sample_rate'],(run,r))
for rate,(run,r) in selected.items():
 s=r['scene'];p=opt.PROJECT/'build'/run/'cases'/s['id']/'clean.wav';assert opt.paired.replay.ctl.ev.digest(p)==s['files']['clean.wav']['sha256'];clean,sr=sf.read(p);assert sr==rate
 rng=np.random.default_rng(77400+rate);noise=rng.normal(size=len(clean));noise/=np.sqrt(np.mean(noise**2))
 for db in (-90,-80,-70,-60):
  before=clean+noise*10**(db/20);input_path=D/f'input-{rate}-{abs(db)}.wav';sf.write(input_path,before,rate,subtype='FLOAT');before=np.clip(np.rint(before*32768),-32768,32767)/32768
  for c in configs:
   prefix=D/c['name']/f'{rate}-{abs(db)}';a,ar,em=verify.call(c,input_path,prefix);assert ar==rate and len(a)==len(clean)
   failures=[];segments={}
   for label,(lo,hi) in s['windows'].items():
    sl=slice(round(lo*rate),round(hi*rate))
    if label.startswith('near'):
     qb=opt.quality(clean[sl],before[sl],rate);qa=opt.quality(clean[sl],a[sl],rate);delta=qa['stoi']-qb['stoi'] if qa['stoi'] is not None and qb['stoi'] is not None else None
     segments[label]=dict(before=qb,after=qa,stoi_delta=delta)
     if delta is not None and delta<-.02:failures.append(label+':stoi')
     if qa['projection_gain_db'] is not None and not -3<=qa['projection_gain_db']<=18:failures.append(label+':projection_gain')
    elif label.startswith('noise_only'):
     m=opt.paired.replay.ctl.ev.attenuation_measurement(before[sl],a[sl],pcm16=True);segments[label]=m;v=m['total_attenuation_db'] if m['total_attenuation_db'] is not None else m['attenuation_lower_bound_db']
     if v is not None and v<6:failures.append(label+':attenuation')
   out.append(dict(variant=c['name'],rate=rate,adc_noise_dbfs=db,source=dict(run=run,id=s['id'],clean_sha256=s['files']['clean.wav']['sha256']),input_sha256=opt.paired.replay.ctl.ev.digest(input_path),output_sha256=opt.paired.replay.ctl.ev.digest(prefix.with_suffix('.after.wav')),segments=segments,pass_gates=not failures,failures=failures,engine=em))
opt.save(D/'report.json',dict(state='completed',rows=out,post_freeze=True,used_for_selection=False,standard_compliance=False))
for c in configs:
 g=[x for x in out if x['variant']==c['name']];print(c['name'],sum(x['pass_gates'] for x in g),'/',len(g))
