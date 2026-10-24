#!/usr/bin/env python3
"""Counterfactual configuration probes on measured common NS failures."""
import os
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import numpy as np
import soundfile as sf
import repair_ns_vehicle_v4 as ev
from analyze_statistical_ns_common_failures import P,rows,key,save,sha,projection

VARIANTS = {
    'linear_rumble': {18:'linear'},
    'no_rumble': {17:'0',18:'linear'},
    'no_main_dry': {6:'0'},
    'no_quiet_raw': {14:'-300',16:'2'},
    'no_startup': {9:'0'},
    'core_only': {6:'0',9:'0',10:'aligned',14:'-300',16:'0',17:'0',18:'linear'},
}

def measure(task):
    out,frame_row,row,name = task
    rate = row['scene']['sample_rate']; config = row['config']
    base = P/frame_row['folder']; folder = out/frame_row['scope']/row['variant']/name/row['original_run']/row['scene']['id']
    folder.mkdir(parents=True,exist_ok=False); prefix = folder/'result'
    argv = json.loads((base/'invocation.json').read_text())['argv']
    argv[4] = str(prefix)
    for position,value in VARIANTS[name].items():
        argv[position] = value
    result = subprocess.run(argv,capture_output=True,text=True)
    save(folder/'invocation.json',dict(argv=argv,exit_code=result.returncode,stderr=result.stderr))
    result.check_returncode()
    engine = json.loads(prefix.with_suffix('.json').read_text())
    origin = P/'build'/row['original_run']/'cases'/row['scene']['id']
    clean,cr = sf.read(origin/'clean.wav'); before,br = sf.read(origin/'before.wav')
    crop = row['input_crop_samples']; clean = clean[crop:]; before = before[crop:]
    canonical = np.clip(np.rint(before*32768),-32768,32767).astype('<i2')
    import hashlib
    assert hashlib.sha256(canonical.tobytes()).hexdigest() == row['input_pcm16_sha256']
    before = canonical.astype(float)/32768
    after,ar = sf.read(prefix.with_suffix('.after.wav')); delay = engine['delay_samples']
    assert ar == br == cr == rate and len(after) == len(clean)+delay
    after = after[delay:delay+len(clean)]
    segments = {}; failures = []
    for label,(lo,hi) in row['scene']['windows'].items():
        sl = slice(round(lo*rate),min(len(clean),round(hi*rate)));c,b,a=clean[sl],before[sl],after[sl]
        if len(c) < rate//20:
            continue
        if label.startswith('near'):
            qb,qa = ev.quality(c,b,rate),ev.quality(c,a,rate)
            if qa['sisdr_db'] is None:
                continue
            segments[label] = dict(before=qb,after=qa)
            if qa['stoi'] is not None and qb['stoi'] is not None and qa['stoi'] < qb['stoi']-.02:
                failures.append(label+':stoi')
            if not -3 <= qa['projection_gain_db'] <= 18:
                failures.append(label+':projection_gain')
            if qa['sisdr_db'] < min(qb['sisdr_db'],20)-1:
                failures.append(label+':raw_sisdr')
        elif label.startswith('noise_only'):
            q = ev.paired.replay.ctl.ev.attenuation_measurement(b,a,pcm16=True)
            attenuation = q['total_attenuation_db']
            if attenuation is None:
                attenuation = q['attenuation_lower_bound_db']
            segments[label] = q
            if attenuation is not None and attenuation < 6:
                failures.append(label+':attenuation')
    if engine['clipped_samples_before_write']:
        failures.append('clipping')
    first_end = min(len(clean),round(.05*rate))
    record = dict(scope=frame_row['scope'],variant=row['variant'],ablation=name,
        original_run=row['original_run'],scene=row['scene'],segments=segments,failures=failures,
        raw_quality_pass=not failures,speech_pass=not any(not f.endswith(':attenuation') for f in failures),
        baseline_failures=row['failures']+row['raw_failures'],
        baseline_segments=row['segments'],delay_samples=delay,
        first50ms_projection_db=projection(clean[:first_end],after[:first_end]),
        output_sha256=sha(prefix.with_suffix('.after.wav')),folder=str(folder.relative_to(P)),
        thresholds_unchanged=True,diagnostic_only=True)
    save(folder/'metrics.json',record)
    return record

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--frames',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--jobs',type=int,default=4)
    args=parser.parse_args()
    if not 1 <= args.jobs <= 4:
        parser.error('jobs must be 1..4')
    root=args.root.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    frame_rows=rows(args.frames/'results.jsonl');om=[r for r in frame_rows if r['variant']=='omlsa_imcra_v2_balanced']
    selected={}
    def choose(scope,noise,window,field,rate=None):
        g=[r for r in om if r['scope']==scope and r['scene']['noise']==noise and window in r['windows']
           and r['windows'][window].get(field) is not None and (rate is None or r['scene']['sample_rate']==rate)]
        if not g:
            raise ValueError(('empty diagnostic selection',scope,noise,window,rate))
        r=min(g,key=lambda r:(r['windows'][window][field],key(r)))
        selected[(scope,*key(r))]=r
    choose('steady','tcar','noise_only','attenuation_db')
    choose('steady','cavemove_alfaromeo_146_s60_w0','noise_only_after_step','attenuation_db')
    choose('steady','cavemove_alfaromeo_146_s60_w0','near_only','output_projection_db')
    choose('steady','sim_engine_ramp','near_late','output_projection_db',24000)
    choose('steady','sim_engine_ramp','near_late','output_projection_db',48000)
    choose('transient','sim_wind_gust','near_only','output_projection_db')
    choose('transient','sim_rain_impulse','near_only','output_projection_db')
    choose('cold','cavemove_alfaromeo_146_s60_w0','near_startup','output_projection_db')
    groups={}
    bindings={str((args.frames/'results.jsonl').resolve()):sha(args.frames/'results.jsonl')}
    for scope in ('steady','transient','cold'):
        for phase in (f'omlsa-final-{scope}-r2',f'sg-final-{scope}-r1'):
            path=root/phase/'results.jsonl';bindings[str(path)]=sha(path)
            for row in rows(path):
                if row['variant'] in ('omlsa_imcra_v2_balanced','super_gaussian_balanced'):
                    groups[(scope,row['variant'],*key(row))]=row
    tasks=[]
    for (scope,run,scene),_ in sorted(selected.items()):
        for variant in ('omlsa_imcra_v2_balanced','super_gaussian_balanced'):
            frame_row=next(r for r in frame_rows if (r['scope'],r['variant'],*key(r))==(scope,variant,run,scene))
            row=groups[(scope,variant,run,scene)]
            tasks.extend((out,frame_row,row,name) for name in VARIANTS)
    save(out/'identity.json',dict(selection=[dict(scope=s,run=r,id=i) for s,r,i in selected],
        variants=VARIANTS,tasks=len(tasks),dependencies=bindings,script_sha256=sha(Path(__file__))))
    records=[]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool,(out/'results.jsonl').open('x') as stream:
        for future in as_completed([pool.submit(measure,t) for t in tasks]):
            record=future.result();records.append(record);stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
    assert len(records)==len(tasks)>0
    assert all(sha(Path(p))==h for p,h in bindings.items())
    save(out/'report.json',dict(state='completed',cases=len(selected),executions=len(records),
        changed_output_negative_controls=sum(r['output_sha256']!=groups[(r['scope'],r['variant'],*key(r))]['output_sha256'] for r in records),
        thresholds_unchanged=True,diagnostic_only=True))
    print('completed',len(records),'configuration counterfactuals')

if __name__=='__main__':
    main()
