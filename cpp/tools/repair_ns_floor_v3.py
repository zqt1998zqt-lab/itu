#!/usr/bin/env python3
"""Cold-start scoring fork: preserve unavailable short-window STOI as null."""
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time
import math
from pesq import pesq
from scipy import signal

import numpy as np
import soundfile as sf
import compare_ns_backends as paired
import evaluate_ns_isolation as isolation

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / 'build/ns-failure-repair-v3-20261018'
PROBE = Path(os.environ.get('NS_V3_PROBE', PROJECT / 'build/ns-failure-repair-v3-20261018/ns_floor_repair_probe'))


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def domain(scene):
    n = scene['noise']
    return 'recorded_car' if n == 'tcar' or n.startswith('cavemove_') else 'synthetic_vehicle' if n.startswith('sim_') else 'diagnostic'


def sources(scope):
    base = paired.selected_cases('core')
    rows = [(run, r) for run, r in base if r['scene']['family'] == 'ns' and r['scene']['mics'] == 1]
    if scope == 'tune':
        # All recorded-car validation cases; development is selected by metadata
        # only, with four examples per rate/noise/SNR-band cell. The synthetic
        # grid is small enough to retain all development/validation cases.
        selected = []; groups = defaultdict(list)
        for run, r in rows:
            s = r['scene']; d = domain(s)
            if s['split'] == 'holdout' or d == 'diagnostic':
                continue
            if d == 'synthetic_vehicle' or s['split'] == 'validation':
                selected.append((run, r)); continue
            snr = s['snr_db']
            band = 'lt0' if snr < 0 else '0to9' if snr < 10 else '10to19' if snr < 20 else 'ge20'
            groups[(s['sample_rate'], s['noise'], band)].append((run, r))
        for key in sorted(groups):
            selected.extend(sorted(groups[key], key=lambda x: hashlib.sha256((x[0]+'/'+x[1]['scene']['id']).encode()).hexdigest())[:4])
        return sorted(selected, key=lambda x: (x[0], x[1]['scene']['id']))
    if scope == 'vehicle':
        return [(run, r) for run, r in rows if domain(r['scene']) != 'diagnostic']
    if scope == 'diagnostic':
        return [(run, r) for run, r in paired.selected_cases('sweep') if r['scene']['family'] == 'ns' and domain(r['scene']) == 'diagnostic']
    if scope in ('clean', 'cold', 'clean-tune', 'cold-tune'):
        unique = {}
        for run, r in rows:
            if domain(r['scene']) != 'diagnostic' and (not scope.endswith('-tune') or r['scene']['split'] != 'holdout'):
                unique.setdefault((r['scene']['sample_rate'], r['scene']['files']['clean.wav']['sha256']), (run, r))
        return list(unique.values())
    raise ValueError(scope)


def prepare():
    contract = json.loads((ROOT / 'contract.json').read_text())
    variants = []
    for backend, levels in contract['strengths'].items():
        for strength in levels:
            for profile in contract['common_profiles']:
                variants.append(dict(name=f'{backend}_{abs(strength)}_{profile["name"]}', backend=backend, strength=strength, **{k:v for k,v in profile.items() if k != 'name'}))
    save(ROOT / 'round1-variants.json', variants)
    save(ROOT / 'selection-coverage.json', {scope:dict(cases=len(sources(scope)), cells=dict(Counter(f'{domain(r["scene"])}:{r["scene"]["sample_rate"]}:{r["scene"]["split"]}' for _,r in sources(scope)))) for scope in ('tune','vehicle','diagnostic','clean','cold')})
    print('prepared', len(variants), 'variants; tune cases', len(sources('tune')), flush=True)


def quality(clean, audio, rate):
    paired.replay.ctl.ev.RATE = rate
    return paired.replay.ctl.ev.quality(clean, audio)


def execute(task):
    out, config, run, original, scope = task
    scene = copy.deepcopy(original['scene']); rate = scene['sample_rate']
    folder = out / config['name'] / run / scene['id']; folder.mkdir(parents=True)
    origin = PROJECT / 'build' / run / 'cases' / scene['id']
    for name in ('before.wav','clean.wav'):
        if paired.replay.ctl.ev.digest(origin/name) != scene['files'][name]['sha256']:
            raise ValueError('input identity changed')
    clean, sr = sf.read(origin/'clean.wav'); before, br = sf.read(origin/'before.wav')
    if sr != rate or br != rate or before.ndim != 1 or clean.shape != before.shape:
        raise ValueError('mono native input/target shape mismatch')
    input_path = origin / 'before.wav'
    if scope.startswith('clean'):
        input_path = origin / 'clean.wav'; before = clean.copy()
    if scope.startswith('cold'):
        hop=rate//100; power=np.mean(clean[:len(clean)//hop*hop].reshape(-1,hop)**2,axis=1)
        positions=np.flatnonzero(power>=10**(-65/10))
        if not len(positions):raise ValueError('cold-start fixture has no speech')
        offset=int(positions[0])*hop; clean=clean[offset:];before=before[offset:]
        scene['windows']={k:[max(0,lo-offset/rate),hi-offset/rate] for k,(lo,hi) in scene['windows'].items() if hi-offset/rate>=.25}
        scene['windows']['near_startup']=[0., min(.5,len(clean)/rate)]
        scene['near_schedule']=[[max(0,lo-offset/rate),hi-offset/rate] for lo,hi in scene['near_schedule'] if hi>offset/rate]
        input_path=folder/'cold-input.wav';sf.write(input_path,before,rate,subtype='FLOAT')
        scene['cold_fixture_crop_samples']=offset
    prefix=folder/'result'
    command=[str(PROBE),config['backend'],str(config['strength']),str(input_path),str(prefix),config.get('mode','shared'),str(config['dry_max']),str(config['snr_threshold_db']),str(config['noise_gain']),str(config['startup_ms']),config['policy'],str(config['mild_strength']),str(config['slew_ms']),str(config.get('release_ms',0)),str(config.get('quiet_floor_db',-300)),str(config.get('floor_guard',0))]
    result=subprocess.run(command,capture_output=True,text=True)
    save(folder/'invocation.json',dict(argv=command,exit_code=result.returncode,stderr=result.stderr))
    result.check_returncode()
    engine=json.loads(prefix.with_suffix('.json').read_text())
    if engine['backend']!=config['backend'] or engine['strength']!=config['strength'] or engine['floor_guard']!=config.get('floor_guard',0) or engine['repair_release_ms']!=config.get('release_ms',0) or engine['quiet_floor_db']!=config.get('quiet_floor_db',-300) or engine['mode']!=config.get('mode','shared'):
        raise ValueError('actual backend/config differs')
    after,ar=sf.read(prefix.with_suffix('.after.wav'));delay=engine['delay_samples']
    if ar!=rate or len(after)!=len(clean)+delay or not np.isfinite(after).all():raise ValueError('output shape/finite failure')
    after=after[delay:delay+len(clean)]; before=np.clip(np.rint(before*32768),-32768,32767)/32768
    # Perception is a selection metric this round, measured on the same first
    # scheduled utterance at fixed declared delay. Never a fitted lag/gain.
    lo,hi=scene['near_schedule'][0];sl=slice(round(lo*rate),round(hi*rate))
    divisor=math.gcd(rate,16000)
    def scoring_band(x):
        return x if rate==16000 else signal.resample_poly(x,16000//divisor,rate//divisor)
    try:
        pesq_first=float(pesq(16000,scoring_band(clean[sl]),scoring_band(after[sl]),'wb'));pesq_error=None
    except Exception as exc:
        pesq_first=None;pesq_error=type(exc).__name__+': '+str(exc)
    segments={}; failures=[];raw_failures=[];speech=[];noise=[]
    for label,(lo,hi) in scene['windows'].items():
        sl=slice(round(lo*rate),min(len(clean),round(hi*rate)))
        c,b,a=clean[sl],before[sl],after[sl]
        if len(c)<rate//20:continue
        if label.startswith('near'):
            qb,qa=quality(c,b,rate),quality(c,a,rate)
            if qa['sisdr_db'] is None:continue
            item=dict(before=qb,after=qa,sisdr_improvement_db=qa['sisdr_db']-qb['sisdr_db'],stoi_improvement=qa['stoi']-qb['stoi'] if qa['stoi'] is not None and qb['stoi'] is not None else None, stoi_unavailable_reason='insufficient valid STOI frames in this window' if qa['stoi'] is None or qb['stoi'] is None else None)
            segments[label]=item;speech.append(item)
            if qa['stoi'] is not None and qb['stoi'] is not None and qa['stoi']<qb['stoi']-.02:failures.append(label+':stoi')
            if not -3<=qa['projection_gain_db']<=18:failures.append(label+':projection_gain')
            if qa['sisdr_db']<min(qb['sisdr_db'],20)-1:raw_failures.append(label+':raw_sisdr')
        elif label.startswith('noise_only'):
            item=paired.replay.ctl.ev.attenuation_measurement(b,a,pcm16=True);segments[label]=item
            attenuation=item.get('total_attenuation_db')
            if attenuation is None:attenuation=item.get('attenuation_lower_bound_db')
            if attenuation is not None:
                noise.append(attenuation)
                if attenuation<6:failures.append(label+':attenuation')
    if not speech:raise ValueError('no available speech metrics')
    if engine['clipped_samples_before_write']>0:failures.append('clipping')
    onsets=[]
    hop=rate//100;n=len(clean)//hop;power=np.mean(clean[:n*hop].reshape(n,hop)**2,axis=1)
    active=power>=10**(-65/10)
    for lo,hi in scene['near_schedule']:
        positions=np.flatnonzero(active & (np.arange(n)/100>=lo) & (np.arange(n)/100<hi))
        if not len(positions):continue
        start=int(positions[0])*hop
        onsets.append({label:isolation.projection(clean[start+round(lo2*rate):start+round(hi2*rate)],after[start+round(lo2*rate):start+round(hi2*rate)])['projection_gain_db'] for label,lo2,hi2 in (('0_50ms',0,.05),('50_150ms',.05,.15),('150_500ms',.15,.5))})
    gain=[]
    for o in onsets:
        if o['0_50ms'] is not None:gain.append(o['0_50ms'])
    row=dict(pesq_first=pesq_first,pesq_unavailable_reason=pesq_error,variant=config['name'],config=config,original_run=run,scene={k:scene[k] for k in ('id','sample_rate','split','noise','snr_db','near_speaker','near_schedule','windows')},domain=domain(scene),scope=scope,engine=engine,segments=segments,onsets=onsets,ns_tradeoff_pass=not failures,raw_quality_pass=not failures and not raw_failures,failures=failures,raw_failures=raw_failures,mean_stoi_improvement=float(np.mean([s['stoi_improvement'] for s in speech if s['stoi_improvement'] is not None])) if any(s['stoi_improvement'] is not None for s in speech) else None,mean_sisdr_improvement_db=float(np.mean([s['sisdr_improvement_db'] for s in speech])),mean_noise_attenuation_db=float(np.mean(noise)) if noise else None,onset_0_50ms_projection_db=float(np.mean(gain)) if gain else None,output_sha256=paired.replay.ctl.ev.digest(prefix.with_suffix('.after.wav')),input_sha256=paired.replay.ctl.ev.digest(input_path),clean_sha256=scene['files']['clean.wav']['sha256'],standard_compliance=False)
    row['binary_sha256']=paired.replay.ctl.ev.digest(PROBE)
    row['wav_retained']=os.environ.get('NS_V3_KEEP_WAV')=='1'
    save(folder/'metrics.json',row)
    if not row['wav_retained']:
        prefix.with_suffix('.after.wav').unlink()
        if scope.startswith('cold'): input_path.unlink()
    return row


def run(args):
    configs=json.loads(args.variants.read_text());out=ROOT/args.phase;out.mkdir(exist_ok=False)
    selected=sources(args.scope)
    if not selected or not configs:raise ValueError('nonempty matrix required')
    dependencies=[Path(__file__),Path(paired.__file__),Path(isolation.__file__),Path(paired.replay.__file__),Path(paired.replay.ctl.ev.__file__),PROJECT/'cpp/src/ns_tuning_controller.hpp',PROJECT/'cpp/tools/ns_refinement_stream_probe.cpp',PROJECT/'cpp/src/ns_control_alignment.hpp',ROOT/'contract.json',PROBE,args.variants.resolve()]
    for run_name, manifest in paired.replay.SOURCES:
        dependencies.extend([PROJECT/'build'/run_name/'results.jsonl',PROJECT/'datasets'/manifest/'manifest.json'])
    dependencies += [PROJECT/'cpp/tools/ns_floor_repair_probe.cpp',PROJECT/'cpp/src/ns_floor_repair_controller.hpp',PROJECT/'cpp/src/ns_repair_alignment.hpp']
    bindings={str(p):paired.replay.ctl.ev.digest(p) for p in dependencies}
    save(out/'identity.json',dict(dependencies=bindings,scope=args.scope,configs=configs,selection=[dict(run=run,id=r['scene']['id']) for run,r in selected]))
    tasks=[(out,c,run,r,args.scope) for c in configs for run,r in selected];start=time.monotonic();completed=0
    save(out/'report.json',dict(state='running',completed=0,total=len(tasks)))
    with ProcessPoolExecutor(max_workers=args.jobs) as pool,(out/'results.jsonl').open('x') as stream:
        futures=[pool.submit(execute,t) for t in tasks]
        for future in as_completed(futures):
            row=future.result();stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();completed+=1
            if completed%100==0 or completed==len(tasks):
                save(out/'report.json',dict(state='running',completed=completed,total=len(tasks),elapsed_seconds=time.monotonic()-start))
                print(args.phase,completed,'/',len(tasks),flush=True)
    changed=[p for p,h in bindings.items() if paired.replay.ctl.ev.digest(Path(p))!=h]
    save(out/'report.json',dict(state='completed',completed=completed,total=len(tasks),cases=len(selected),configs=len(configs),identity_verified=not changed,changed_dependencies=changed,elapsed_seconds=time.monotonic()-start,standard_compliance=False))
    if changed:raise ValueError('identity drift')


def aggregate(rows):
    result={}
    for name in sorted({r['variant'] for r in rows}):
        group=[r for r in rows if r['variant']==name];cells=defaultdict(list)
        for r in group:cells[(r['domain'],r['scene']['sample_rate'],r['scene']['noise'])].append(r)
        result[name]=dict(config=group[0]['config'],cases=len(group),passed=sum(r['ns_tradeoff_pass'] for r in group),raw_passed=sum(r['raw_quality_pass'] for r in group),balanced_pass={d:float(np.mean([np.mean([x['ns_tradeoff_pass'] for x in v]) for k,v in cells.items() if k[0]==d])) if any(k[0]==d for k in cells) else None for d in ('recorded_car','synthetic_vehicle','diagnostic')},mean_stoi_improvement=float(np.mean([r['mean_stoi_improvement'] for r in group])),mean_noise_attenuation_db=float(np.mean([r['mean_noise_attenuation_db'] for r in group if r['mean_noise_attenuation_db'] is not None])) if any(r['mean_noise_attenuation_db'] is not None for r in group) else None,onset_0_50ms_projection_db=float(np.mean([r['onset_0_50ms_projection_db'] for r in group if r['onset_0_50ms_projection_db'] is not None])),failure_counts=dict(Counter(f for r in group for f in r['failures'])))
    return result


def summarize(phases):
    rows=[]
    for phase in phases:
        rows += [json.loads(l) for l in (ROOT/phase/'results.jsonl').open()]
    result={split:aggregate([r for r in rows if r['scene']['split']==split]) for split in ('development','validation','holdout')}
    save(ROOT/('summary-'+'-'.join(phases)+'.json'),result)
    print('summary saved',phases)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare',action='store_true');p.add_argument('--summarize',nargs='+')
    p.add_argument('--phase');p.add_argument('--variants',type=Path);p.add_argument('--scope',choices=('tune','vehicle','diagnostic','clean','cold','clean-tune','cold-tune'),default='tune');p.add_argument('--jobs',type=int,default=8)
    args=p.parse_args()
    if args.prepare:prepare()
    elif args.summarize:summarize(args.summarize)
    else:
        if not args.phase or not args.variants or not 1<=args.jobs<=8:p.error('phase, variants and jobs 1..8 required')
        run(args)


if __name__=='__main__':main()
