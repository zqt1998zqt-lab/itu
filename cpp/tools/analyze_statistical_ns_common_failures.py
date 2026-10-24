#!/usr/bin/env python3
"""Replay paired statistical-NS failures without changing the frozen scoring."""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import soundfile as sf

P = Path(__file__).resolve().parents[2]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def rows(path):
    return [json.loads(line) for line in path.open()]

def key(row):
    return row['original_run'], row['scene']['id']

def projection(c, x):
    denominator = float(np.dot(c, c))
    if denominator < 1e-12:
        return None
    return float(20*np.log10(max(abs(float(np.dot(c, x))/denominator), 1e-12)))

def power_db(x):
    return float(10*np.log10(max(float(np.mean(x*x)), 1e-30)))

def band_powers(x, rate):
    p = abs(np.fft.rfft(x))**2
    if len(p) > 2:
        p[1:-1] *= 2
    frequency = np.fft.rfftfreq(len(x), 1/rate)
    return {name:float(p[(frequency >= lo) & (frequency < hi)].sum()/len(x)**2)
            for name,lo,hi in (('below80',0,80),('80to300',80,300),
                               ('300to3400',300,3400),('above3400',3400,rate/2+1))}

def longest(mask):
    best = current = 0
    for valid in mask:
        current = current+1 if valid else 0
        best = max(best, current)
    return best

def replay(task):
    root, out, scope, row = task
    scene = row['scene']; rate = scene['sample_rate']; hop = rate//100
    origin = P/'build'/row['original_run']/'cases'/scene['id']
    c, cr = sf.read(origin/'clean.wav'); b, br = sf.read(origin/'before.wav')
    assert cr == br == rate and c.shape == b.shape and c.ndim == 1
    assert sha(origin/'clean.wav') == row['clean_sha256']
    offset = row['input_crop_samples']; c = c[offset:]; b = b[offset:]
    canonical = np.clip(np.rint(b*32768), -32768, 32767).astype('<i2')
    assert hashlib.sha256(canonical.tobytes()).hexdigest() == row['input_pcm16_sha256']
    folder = out/'replay'/scope/row['variant']/row['original_run']/scene['id']
    folder.mkdir(parents=True, exist_ok=False)
    source = origin/'before.wav'
    if offset:
        source = folder/'cropped-input.wav'
        sf.write(source, b, rate, subtype='FLOAT')
    config = row['config']; candidate = 'candidate-r2' if config['backend'] == 'omlsa_imcra' else 'candidate-r1'
    probe = root/'artifacts'/candidate/'ns_vehicle_probe'
    assert sha(probe) == row['binary_sha256']
    prefix = folder/'result'
    argv = [str(probe),config['backend'],str(config['strength']),str(source),str(prefix),'shared']
    argv += [str(config[k]) for k in ('dry_max','snr_threshold_db','noise_gain','startup_ms',
             'policy','mild_strength','slew_ms','release_ms','quiet_floor_db','floor_guard',
             'repair_mode','rumble_hz','rumble_kind')]
    if 'spectral_dry_guard' in config:
        argv += ['--spectral-dry-guard='+str(config['spectral_dry_guard'])]
    invocation = subprocess.run(argv, capture_output=True, text=True)
    save(folder/'invocation.json', dict(argv=argv,exit_code=invocation.returncode,
         stdout=invocation.stdout,stderr=invocation.stderr))
    invocation.check_returncode()
    assert sha(prefix.with_suffix('.after.wav')) == row['output_sha256'], (scope,key(row),'output mismatch')
    engine = json.loads(prefix.with_suffix('.json').read_text())
    a, ar = sf.read(prefix.with_suffix('.after.wav')); delay = engine['delay_samples']
    assert ar == rate and len(a) == len(c)+delay
    a = a[delay:delay+len(c)]; b = canonical.astype(float)/32768
    frames = len(c)//hop
    cf,bf,af = [v[:frames*hop].reshape(frames,hop) for v in (c,b,a)]
    cp,bp,ap = [np.mean(v*v,axis=1) for v in (cf,bf,af)]
    denominator = np.sum(cf*cf,axis=1)
    active = cp >= 10**(-65/10)
    gain = []
    for data in (bf,af):
        alpha = np.divide(np.sum(cf*data,axis=1),denominator,
                          out=np.zeros(frames),where=denominator>1e-12)
        gain.append(20*np.log10(np.maximum(abs(alpha),1e-12)))
    attenuation = 10*np.log10(np.maximum(bp,1e-30)/np.maximum(ap,1e-30))
    np.savez_compressed(folder/'frames.npz',time_s=np.arange(frames)/100,
        target_power=cp,input_power=bp,output_power=ap,oracle_speech_active=active,
        input_projection_db=gain[0],output_projection_db=gain[1],attenuation_db=attenuation)
    stats = np.atleast_1d(np.genfromtxt(prefix.with_suffix('.csv'),delimiter=',',names=True))
    assert len(stats) == frames
    labels = sorted(set(row['failures']+row['raw_failures']))
    windows = {}
    for label,(lo,hi) in scene['windows'].items():
        if label not in {f.split(':')[0] for f in labels}:
            continue
        start,stop = max(0,round(lo*rate)),min(len(c),round(hi*rate))
        x,y,z = b[start:stop],a[start:stop],c[start:stop]
        mask = (np.arange(frames)/100 >= lo) & (np.arange(frames)/100 < hi)
        valid = mask & active
        bad = valid & (gain[1] < -3)
        indices = np.flatnonzero(bad)
        common = dict(start_s=lo,end_s=hi,target_rms_dbfs=power_db(z),
            input_projection_db=projection(z,x),output_projection_db=projection(z,y),
            active_frames=int(valid.sum()),output_projection_below_minus3_frames=int(bad.sum()),
            input_projection_below_minus3_frames=int((valid & (gain[0]<-3)).sum()),
            first_bad_active_frame=int(indices[0]) if len(indices) else None,
            first_bad_active_time_s=float(indices[0]/100) if len(indices) else None,
            longest_bad_active_run_ms=10*longest(bad),
            input_bands=band_powers(x,rate),output_bands=band_powers(y,rate),
            frozen_metrics=row['segments'].get(label))
        if label.startswith('noise_only'):
            common.update(attenuation_db=power_db(x)-power_db(y),
                frames_below_6db=int((mask & (attenuation<6)).sum()),
                frame_count=int(mask.sum()),longest_below_6db_run_ms=10*longest(mask & (attenuation<6)))
        common['input_time_control_target_means'] = {
            name:float(np.mean(stats[name][mask])) for name in
            ('confidence','snr_db','input_mix_target','quiet_mix_target','rumble_fraction',
             'rumble_mix_target','omlsa_probability','omlsa_gain') if name in stats.dtype.names and mask.any()}
        windows[label] = common
    record = dict(scope=scope,variant=row['variant'],original_run=row['original_run'],scene=scene,
        failures=labels,delay_samples=delay,input_crop_samples=offset,
        input_pcm16_sha256=row['input_pcm16_sha256'],output_sha256=row['output_sha256'],
        replay_hash_matches=True,windows=windows,folder=str(folder.relative_to(P)),
        frame_diagnostic='10ms projection and power; no frame STOI; oracle active is evaluation only',
        control_attribution='CSV is current input-time target, not reconstructed effective sample mix; waveform uses declared delay only')
    save(folder/'analysis.json',record)
    return record

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--jobs',type=int,default=4)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 4:
        parser.error('jobs must be 1..4')
    root = args.root.resolve(); out = args.out.resolve(); out.mkdir(parents=True,exist_ok=False)
    tasks = []; bindings = {}; counts = {}
    for scope in ('steady','transient','clean','cold'):
        groups = []
        for phase,variant in ((f'omlsa-final-{scope}-r2','omlsa_imcra_v2_balanced'),
                              (f'sg-final-{scope}-r1','super_gaussian_balanced')):
            path = root/phase/'results.jsonl'; bindings[str(path)] = sha(path)
            group = {key(r):r for r in rows(path) if r['variant'] == variant}
            groups.append(group)
        assert set(groups[0]) == set(groups[1])
        shared = sorted(k for k in groups[0] if all(not g[k]['raw_quality_pass'] for g in groups))
        counts[scope] = len(shared)
        for k in shared:
            pair = [g[k] for g in groups]
            for field in ('input_pcm16_sha256','input_crop_samples','clean_sha256'):
                assert pair[0][field] == pair[1][field]
            tasks.extend((root,out,scope,r) for r in pair)
    save(out/'identity.json',dict(dependencies=bindings,script_sha256=sha(Path(__file__)),
         common_case_counts=counts,tasks=len(tasks),thresholds_unchanged=True))
    complete = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool, (out/'results.jsonl').open('x') as stream:
        for result in as_completed([pool.submit(replay,t) for t in tasks]):
            record = result.result(); complete.append(record)
            stream.write(json.dumps(record,allow_nan=False)+'\n'); stream.flush()
            if len(complete)%25 == 0 or len(complete) == len(tasks):
                print(len(complete),'/',len(tasks),flush=True)
    assert all(sha(Path(p)) == h for p,h in bindings.items())
    assert len(complete) == len(tasks) > 0
    save(out/'report.json',dict(state='completed',replays=len(complete),common_case_counts=counts,
         exact_output_hash_matches=sum(r['replay_hash_matches'] for r in complete),
         dependencies_verified=True,standard_compliance=False))

if __name__ == '__main__':
    main()
