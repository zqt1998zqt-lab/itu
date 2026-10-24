#!/usr/bin/env python3
"""Paired real-audio baseline/student evaluation at fixed declared delay.

No threshold tuning, fitted alignment, gain normalization or runtime clean input.
Parallel runs measure quality. Serial runs separately measure streaming cost.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import soundfile as sf
import repair_ns_vehicle_v4 as ev

P=Path(__file__).resolve().parents[2]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path,data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def execute(task):
    root,probe,config,run,scene,scope,keep=task
    rate=scene['sample_rate'];h=rate//100
    origin=P/'build'/run/'cases'/scene['id']
    token=hashlib.sha256((run+'/'+scene['id']).encode()).hexdigest()[:20]
    folder=root/config['name']/token;folder.mkdir(parents=True)
    before,br=sf.read(origin/'before.wav');clean,cr=sf.read(origin/'clean.wav')
    if before.ndim!=1 or before.shape!=clean.shape or br!=rate or cr!=rate:
        raise ValueError('native mono input/target mismatch')
    for name in ('before.wav','clean.wav'):
        if sha(origin/name)!=scene['files'][name]['sha256']:
            raise ValueError('source identity drift')
    source=origin/'before.wav';windows=scene['windows'];schedules=scene['near_schedule'];crop=0
    if scope=='clean':
        source=origin/'clean.wav';before=clean.copy()
    if scope=='cold':
        cp=np.mean(clean[:len(clean)//h*h].reshape(-1,h)**2,axis=1)
        active=np.flatnonzero(cp>=10**(-65/10))
        if not len(active):raise ValueError('cold start has no speech')
        crop=int(active[0])*h;clean=clean[crop:];before=before[crop:]
        source=folder/'cold-input.wav';sf.write(source,before,rate,subtype='FLOAT')
        windows={key:(max(0,a-crop/rate),max(0,b-crop/rate)) for key,(a,b) in windows.items()}
        schedules=[(max(0,a-crop/rate),max(0,b-crop/rate)) for a,b in schedules]
    before=np.clip(np.rint(before*32768),-32768,32767)/32768
    prefix=folder/'result'
    command=[str(probe),'--mic',str(source),'--out',str(prefix),'--mode','ecnr','--floor-db','-18']
    model=config.get('models',{}).get(str(rate))
    if model:
        command+=['--model',model,'--spp-mix',str(config['spp_mix']),'--snr-mix',str(config['snr_mix'])]
    process=subprocess.run(command,capture_output=True,text=True)
    save(folder/'invocation.json',dict(argv=command,returncode=process.returncode,stderr=process.stderr))
    process.check_returncode()
    engine=json.loads(prefix.with_suffix('.json').read_text())
    audio,ar=sf.read(prefix.with_suffix('.after.wav'));delay=engine['delay_samples']
    if ar!=rate or delay!=h or engine['core_rate']!=rate or engine['src_delay_samples']!=0 or len(audio)!=len(clean)+delay or not np.isfinite(audio).all():
        raise ValueError('native rate, fixed delay or finite output failed')
    if model and (engine['student_state']!=1 or engine['student_inferences']<engine['frames'] or engine['student_fallback_frames']):
        raise ValueError('student did not actually run')
    after=audio[delay:delay+len(clean)]
    failures=[];segments={};sisdr=[];stoi=[];noise=[];gains=[]
    for label,(lo,hi) in windows.items():
        sl=slice(round(lo*rate),min(len(clean),round(hi*rate)))
        c,b,a=clean[sl],before[sl],after[sl]
        if len(c)<rate//20:continue
        if label.startswith('near'):
            qb,qa=ev.quality(c,b,rate),ev.quality(c,a,rate)
            segments[label]=dict(before=qb,after=qa)
            if qa['sisdr_db'] is None:continue
            sisdr.append(qa['sisdr_db']-qb['sisdr_db']);gains.append(qa['projection_gain_db'])
            if qa['sisdr_db']<min(qb['sisdr_db'],20)-1:failures.append(label+':raw_sisdr')
            if not -3<=qa['projection_gain_db']<=18:failures.append(label+':projection_gain')
            if qa['stoi'] is not None and qb['stoi'] is not None:
                stoi.append(qa['stoi']-qb['stoi'])
                if qa['stoi']<qb['stoi']-.02:failures.append(label+':stoi')
        elif label.startswith('noise_only') and scope!='clean':
            item=ev.paired.replay.ctl.ev.attenuation_measurement(b,a,pcm16=True);segments[label]=item
            attenuation=item.get('total_attenuation_db')
            if attenuation is None:attenuation=item.get('attenuation_lower_bound_db')
            if attenuation is not None:
                noise.append(attenuation)
                if attenuation<6:failures.append(label+':attenuation')
    if not sisdr:raise ValueError('no scored speech segment')
    if engine['clipped_samples']:failures.append('clipping')
    n=len(clean)//h;c=clean[:n*h].reshape(-1,h);a=after[:n*h].reshape(-1,h)
    cp=np.mean(c*c,axis=1);active=cp>=10**(-65/10)
    projection=np.divide(np.sum(c*a,axis=1),np.sum(c*c,axis=1),out=np.zeros(n),where=np.sum(c*c,axis=1)>1e-12)
    pg=20*np.log10(np.maximum(abs(projection),1e-12))
    onsets=[]
    for lo,hi in schedules:
        positions=np.flatnonzero(active & (np.arange(n)/100>=lo) & (np.arange(n)/100<hi))
        if len(positions):
            start=int(positions[0])*h
            onsets.append(ev.isolation.projection(clean[start:start+5*h],after[start:start+5*h])['projection_gain_db'])
    row=dict(variant=config['name'],run=run,id=scene['id'],rate=rate,split=scene['split'],noise=scene['noise'],snr_db=scene['snr_db'],noise_class=scene.get('noise_class','unclassified'),
         scope=scope,crop_samples=crop,engine=engine,segments=segments,failures=failures,passed=not failures,
         mean_sisdr_improvement_db=float(np.mean(sisdr)),mean_stoi_improvement=float(np.mean(stoi)) if stoi else None,
         mean_noise_attenuation_db=float(np.mean(noise)) if noise else None,
         mean_projection_gain_db=float(np.mean(gains)),onset_0_50ms_projection_db=float(np.mean([x for x in onsets if x is not None])) if any(x is not None for x in onsets) else None,
         frame_projection_p05_db=float(np.percentile(pg[active],5)) if active.any() else None,
         frame_under_minus6_fraction=float(np.mean(pg[active]<-6)) if active.any() else None,
         input_pcm16_sha256=hashlib.sha256(before.astype('<f4').tobytes()).hexdigest(),
         output_sha256=sha(prefix.with_suffix('.after.wav')),standard_compliance=False)
    save(folder/'metrics.json',row)
    if not keep:
        prefix.with_suffix('.after.wav').unlink();prefix.with_suffix('.csv').unlink()
    if scope=='cold':source.unlink()
    return row

def aggregate(rows):
    out={}
    for name in sorted({r['variant'] for r in rows}):
        group=[r for r in rows if r['variant']==name]
        def summary(g):
            result=dict(cases=len(g),passed=sum(r['passed'] for r in g),failures=dict(Counter(f for r in g for f in r['failures'])))
            for key in ('mean_sisdr_improvement_db','mean_stoi_improvement','mean_noise_attenuation_db','onset_0_50ms_projection_db','frame_under_minus6_fraction'):
                values=[r[key] for r in g if r[key] is not None]
                result[key]=float(np.mean(values)) if values else None
            return result
        out[name]=dict(total=summary(group),by_rate={str(rate):summary([r for r in group if r['rate']==rate]) for rate in sorted({r['rate'] for r in group})},
                       by_noise_class={kind:summary([r for r in group if r['noise_class']==kind]) for kind in sorted({r['noise_class'] for r in group})},
                       by_split={split:summary([r for r in group if r['split']==split]) for split in sorted({r['split'] for r in group})})
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--configs',type=Path,required=True);p.add_argument('--split',choices=('all','development','validation','holdout'),default='all')
    p.add_argument('--scope',choices=('vehicle','clean','cold'),default='vehicle');p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--keep-wav',action='store_true')
    p.add_argument('--rates',nargs='+',type=int,choices=(16000,24000,48000),default=[16000,24000,48000])
    args=p.parse_args()
    if not 1<=args.jobs<=8:p.error('jobs must be 1..8')
    root=args.root.resolve();root.mkdir(parents=True,exist_ok=False);(root/'artifacts').mkdir()
    configs=json.loads(args.configs.read_text());bindings={str(args.probe.resolve()):sha(args.probe),str(Path(__file__).resolve()):sha(__file__)}
    for config in configs:
        if not config['name'].replace('_','').isalnum():raise ValueError('unsafe variant name')
        for rate,path in config.get('models',{}).items():
            source=Path(path).resolve();bindings[str(source)]=sha(source)
            dest=root/'artifacts'/(config['name']+'-'+str(rate)+'.bin');shutil.copy2(source,dest)
            config['models'][rate]=str(dest)
    selected=ev.sources(args.scope)
    if args.split!='all':selected=[(r,x) for r,x in selected if x['scene']['split']==args.split]
    selected=[(r,x) for r,x in selected if x['scene']['sample_rate'] in args.rates]
    if not configs or not selected:raise ValueError('empty evaluation')
    tags={(x['original_run'],x['scene_id']):x['noise_class'] for x in map(json.loads,(P/'build/ns-noise-taxonomy-20261018/samples.jsonl').open())}
    for run,row in selected:row['scene']['noise_class']=tags[(run,row['scene']['id'])]
    for c in configs:
        if c.get('models') and any(str(row['scene']['sample_rate']) not in c['models'] for _,row in selected):
            raise ValueError('missing requested native-rate model in benchmark config')
    save(root/'identity.json',dict(bindings=bindings,configs=configs,selection=[dict(run=r,id=x['scene']['id']) for r,x in selected],
         thresholds=dict(stoi_delta_min=-.02,projection_gain_db=[-3,18],sisdr_rule='after >= min(before,20)-1',noise_only_attenuation_db_min=6),
         scope=args.scope,split=args.split,rates=args.rates,native_rates=True,teacher_in_runtime=False,standard_compliance=False))
    tasks=[(root,args.probe.resolve(),c,r,x['scene'],args.scope,args.keep_wav) for c in configs for r,x in selected]
    rows=[];start=time.monotonic()
    with ProcessPoolExecutor(args.jobs) as pool,(root/'results.jsonl').open('x') as stream:
        for f in as_completed([pool.submit(execute,t) for t in tasks]):
            row=f.result();rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            if len(rows)%25==0 or len(rows)==len(tasks):
                save(root/'report.json',dict(state='running',completed=len(rows),total=len(tasks),elapsed_seconds=time.monotonic()-start))
                print(len(rows),'/',len(tasks),flush=True)
    drift=[path for path,value in bindings.items() if sha(path)!=value]
    save(root/'summary.json',aggregate(rows))
    save(root/'report.json',dict(state='completed',completed=len(rows),total=len(tasks),cases=len(selected),identity_verified=not drift,
         changed=drift,elapsed_seconds=time.monotonic()-start,parallel_runtime_not_acceptance=True,standard_compliance=False))
    if drift:raise ValueError('evaluation identity drift')

if __name__=='__main__':main()
