#!/usr/bin/env python3
"""NS repair fork: real PCM compatibility, causality and control checks."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,subprocess,pathlib,numpy as np,soundfile as sf
import repair_ns_branches_v3 as opt
P=opt.PROJECT;R=opt.ROOT;D=R/'probe-validation';D.mkdir(exist_ok=False)
old=P/'build/ns-refinement-v2-20261018/ns_refinement_stream_probe';configs=json.loads((R/'repair-variants.json').read_text());checks=[]
def call(c,x,out,legacy=False,mode='shared'):
 argv=[str(old if legacy else opt.PROBE),c['backend'],str(c['strength']),str(x),str(out),mode,*[str(c[k]) for k in ('dry_max','snr_threshold_db','noise_gain','startup_ms','policy','mild_strength','slew_ms')]]
 if not legacy:argv += [str(c.get('release_ms',0)),str(c.get('quiet_floor_db',-300))]
 proc=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True);proc.check_returncode()
 return sf.read(out.with_suffix('.after.wav'))[0],json.loads(out.with_suffix('.json').read_text())
selected={}
for run,r in opt.sources('tune'):selected.setdefault(r['scene']['sample_rate'],(run,r))
for rate,(run,r) in sorted(selected.items()):
 origin=P/'build'/run/'cases'/r['scene']['id'];x,_=sf.read(origin/'before.wav');x=x[:2*rate];z=x.copy();z[rate:]=.01-.5*z[rate:]
 a=D/f'in-{rate}.wav';b=D/f'changed-{rate}.wav';sf.write(a,x,rate,subtype='PCM_16');sf.write(b,z,rate,subtype='PCM_16')
 for c in configs[:4]:
  base,_=call(c,a,D/f'old-{rate}-{c["name"]}',True);out,em=call(c,a,D/f'base-{rate}-{c["name"]}')
  checks.append(dict(name=f'baseline-{rate}-{c["name"]}',bit_exact=bool(np.array_equal(base,out)),finite=bool(np.isfinite(out).all())))
 for backend in ('webrtc','speex'):
  c=next(c for c in configs if c['name']==backend+'_q6');stem=f'causal-{rate}-{backend}';out,em=call(c,a,D/stem);changed,_=call(c,b,D/(stem+'-changed'));baseline,_=call(dict(c,release_ms=0,quiet_floor_db=-300),a,D/(stem+'-off'))
  checks.append(dict(name=stem,prefix_equal=bool(np.array_equal(out[:rate],changed[:rate])),negative_suffix_changed=bool(np.any(out[rate:]!=changed[rate:])),real_configuration_effect=bool(np.any(out!=baseline)),delay_unchanged=em['delay_samples']==(96 if rate==16000 and backend=='webrtc' else 328 if rate==24000 and backend=='webrtc' else 400 if rate==24000 else 336 if backend=='webrtc' else rate//100),shape=len(out)==len(x)+em['delay_samples'],finite=bool(np.isfinite(out).all())))
 if rate==24000:
  c=next(c for c in configs if c['name']=='speex_q6');out,em=call(c,a,D/'native24',mode='native24');checks.append(dict(name='native24',delay=em['delay_samples']==240,core_rate=em['core_rate']==24000,shape=len(out)==len(x)+240))
# Two independent physical controls: low constant noise must not look like quiet
# speech, and speech at multiple levels must not be protected via a label.
for rate in (16000,24000,48000):
 rng=np.random.default_rng(808+rate);t=np.arange(4*rate)/rate
 for label,x in [('stationary',rng.normal(0,10**(-55/20),len(t))),('quiet-speech',np.sin(2*np.pi*220*t)*.03*(.5+.5*np.sin(2*np.pi*3*t))*(t>1))]:
  path=D/f'{label}-{rate}.wav';sf.write(path,x,rate,subtype='PCM_16')
  for backend in ('webrtc','speex'):
   c=next(c for c in configs if c['name']==backend+'_q6');out,em=call(c,path,D/f'control-{label}-{rate}-{backend}')
   trace=np.genfromtxt(D/f'control-{label}-{rate}-{backend}.csv',delimiter=',',names=True);q=trace['quiet_mix_target']
   checks.append(dict(name=f'{label}-{rate}-{backend}',passed=bool(np.max(q[200:])<.05) if label=='stationary' else bool(np.max(q[120:])>.5)))
invalid=[str(opt.PROBE),'webrtc','1',str(a),str(D/'invalid'),'shared','1','9','1','60','parallel','0','50','nan','-85'];p=subprocess.run(invalid,capture_output=True)
checks.append(dict(name='nonfinite_config_rejected',passed=p.returncode!=0))
passed=all(all(v for k,v in x.items() if k!='name') for x in checks);opt.save(D/'report.json',dict(passed=passed,checks=checks));print(json.dumps(dict(passed=passed,checks=len(checks),failed=[x for x in checks if not all(v for k,v in x.items() if k!='name')]),indent=2))
if not passed:raise SystemExit(1)
