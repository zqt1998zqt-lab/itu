#!/usr/bin/env python3
"""Non-vacuous reproducibility, causality, branch-state and alignment checks."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import csv,json,subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
import refine_ns_streams as opt


def main():
    root=opt.ROOT/'stream-probe-validation';root.mkdir(exist_ok=False)
    old=opt.PROJECT/'build/ns-optimization-20261018/ns_tuning_probe'
    configs=json.loads((opt.ROOT/'baseline-variants.json').read_text())
    selected={}
    for run,r in opt.sources('tune'):selected.setdefault(r['scene']['sample_rate'],(run,r))
    checks=[]
    def invoke(config,source,prefix,old_binary=False):
        cmd=[str(old if old_binary else opt.PROBE),config['backend'],str(config['strength']),str(source),str(prefix),'shared',*[str(config[k]) for k in ('dry_max','snr_threshold_db','noise_gain','startup_ms')]]
        if not old_binary:cmd += [config['policy'],str(config['mild_strength']),str(config['slew_ms'])]
        proc=subprocess.run(cmd,capture_output=True,text=True);proc.check_returncode()
        return sf.read(prefix.with_suffix('.after.wav'))[0],json.loads(prefix.with_suffix('.json').read_text())
    for rate,(run,r) in sorted(selected.items()):
        folder=opt.PROJECT/'build'/run/'cases'/r['scene']['id'];x,sr=sf.read(folder/'before.wav');x=x[:2*rate]
        a=root/f'input-{rate}.wav';b=root/f'changed-{rate}.wav';sf.write(a,x,rate,subtype='PCM_16')
        y=x.copy();y[rate:]=-.5*y[rate:]+.01;sf.write(b,y,rate,subtype='PCM_16')
        for cfg in configs:
            name=f'{rate}-{cfg["name"]}';before,_=invoke(cfg,a,root/(name+'-old'),True);after,meta=invoke(cfg,a,root/(name+'-new'))
            checks.append(dict(name=name,legacy_bit_exact=bool(np.array_equal(before,after)),same_shape=before.shape==after.shape))
        for backend,strength in [('webrtc',1),('speex',-24)]:
            for policy in ['aligned','parallel','parallel_start']+(['dynamic'] if backend=='speex' else []):
                cfg=dict(backend=backend,strength=strength,dry_max=1,snr_threshold_db=9,noise_gain=.85,startup_ms=60,policy=policy,mild_strength=0 if backend=='webrtc' else -6,slew_ms=50)
                stem=f'{backend}-{rate}-{policy}';out,em=invoke(cfg,a,root/stem);changed,_=invoke(cfg,b,root/(stem+'-changed'))
                raw,_=invoke(dict(cfg,dry_max=0,noise_gain=1),a,root/(stem+'-raw'))
                checks.append(dict(name=stem,future_prefix_equal=bool(np.array_equal(out[:rate],changed[:rate])),suffix_negative_control=bool(np.any(out[rate:]!=changed[rate:])),actual_policy_changes_output=bool(np.any(out!=raw)),same_shape=out.shape==raw.shape,finite=bool(np.isfinite(out).all()),configuration_bound=em['control_policy']==policy and em['mild_strength']==cfg['mild_strength'],delay_unchanged=em['delay_samples']==(96 if rate==16000 and backend=='webrtc' else 328 if rate==24000 and backend=='webrtc' else 400 if rate==24000 else 336 if backend=='webrtc' else rate//100)))
                if policy=='parallel':
                    equal,_=invoke(dict(cfg,mild_strength=strength,noise_gain=1),a,root/(stem+'-equal-state'))
                    checks[-1]['equal_strength_parallel_equals_raw']=bool(np.array_equal(equal,raw))
                if policy=='dynamic':
                    trace=list(csv.DictReader((root/stem).with_suffix('.csv').open()));v=np.array([float(row['effective_suppression_db']) for row in trace]);checks[-1]['suppression_really_varies']=bool(np.ptp(v)>.1 and v.min()>=strength and v.max()<=-6)
    # Exact known control-ring impulse; cannot pass merely from audio equality.
    source=root/'alignment.cpp';source.write_text('#include "ns_control_alignment.hpp"\nint main(){itu::NsControlAlignment s(3,4,0);s.begin(1,1);float expected[]={0,0,0,.25f};for(unsigned i=0;i<4;++i){float m,g;s.sample(i,m,g);if(m!=expected[i]||g!=1)return 1;}return 0;}\n')
    subprocess.run(['c++','-std=c++20','-I'+str(opt.PROJECT/'cpp/src'),str(source),'-o',str(root/'alignment')],check=True)
    unit=subprocess.run([str(root/'alignment')]);checks.append(dict(name='known_control_delay',passed=unit.returncode==0))
    invalid_cfg=dict(configs[0],policy='unsupported');cmd=[str(opt.PROBE),'webrtc','1',str(a),str(root/'invalid'),'shared','1','9','1','60','unsupported','0','50']
    invalid=subprocess.run(cmd,capture_output=True,text=True)
    checks.append(dict(name='invalid_rejected',passed=invalid.returncode!=0))
    passed=all(all(v for k,v in row.items() if k!='name') for row in checks)
    opt.save(root/'report.json',dict(passed=passed,checks=checks))
    print(json.dumps(dict(passed=passed,checks=len(checks),failed=[c for c in checks if not all(v for k,v in c.items() if k!='name')]),indent=2))
    if not passed:raise SystemExit(1)

if __name__=='__main__':main()
