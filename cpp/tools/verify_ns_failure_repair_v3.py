#!/usr/bin/env python3
"""Independent perceptual and continuous-stream checks of frozen NS winners."""
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
from pesq import pesq
from scipy import signal
import soundfile as sf
import repair_ns_floor_v3 as opt
from ecnr_perceptual import DNSMOS, DEFAULT_ASSETS

SCORER = None


def call(config, source, prefix, mode='shared', paced=False):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(opt.PROBE), config['backend'], str(config['strength']), str(source), str(prefix), mode,
           str(config['dry_max']), str(config['snr_threshold_db']), str(config['noise_gain']), str(config['startup_ms']),config['policy'],str(config['mild_strength']),str(config['slew_ms']),str(config.get('release_ms',0)),str(config.get('quiet_floor_db',-300)),str(config.get('floor_guard',0))]
    if paced: cmd.append('--pace-10ms')
    p = subprocess.run(cmd, capture_output=True, text=True)
    opt.save(prefix.with_suffix('.invocation.json'), dict(argv=cmd, exit_code=p.returncode, stderr=p.stderr))
    p.check_returncode()
    em = json.loads(prefix.with_suffix('.json').read_text())
    out, sr = sf.read(prefix.with_suffix('.after.wav'))
    return out[em['delay_samples']:], sr, em


def perceptual_task(task):
    global SCORER
    root, configs, run, original = task
    if SCORER is None: SCORER = DNSMOS(DEFAULT_ASSETS, root/'cache')
    s = original['scene']; rate = s['sample_rate']
    origin = opt.PROJECT/'build'/run/'cases'/s['id']
    for name in ('clean.wav','before.wav'):
        if opt.paired.replay.ctl.ev.digest(origin/name)!=s['files'][name]['sha256']:raise ValueError('source identity differs')
    clean, sr = sf.read(origin/'clean.wav'); before, br = sf.read(origin/'before.wav')
    if sr != rate or br != rate: raise ValueError('input rate differs')
    lo, hi = s['near_schedule'][0]; sl = slice(round(lo*rate), round(hi*rate)); clean = clean[sl]
    before = np.clip(np.rint(before*32768), -32768, 32767)/32768
    signals = {'input': before[sl]}; hashes = {}
    for c in configs:
        folder = Path(c.get('evidence_root',str(opt.ROOT)))/'final-vehicle'/c['name']/run/s['id']
        metrics = json.loads((folder/'metrics.json').read_text())
        path = folder/'result.after.wav'; sha = opt.paired.replay.ctl.ev.digest(path)
        if sha != metrics['output_sha256']: raise ValueError('final output identity differs')
        audio, ar = sf.read(path); d = metrics['engine']['delay_samples']
        if ar != rate: raise ValueError('output rate differs')
        signals[c['name']] = audio[d:d+len(before)][sl]; hashes[c['name']] = sha
    divisor = math.gcd(rate, 16000)
    def narrow(x):
        return x if rate == 16000 else signal.resample_poly(x, 16000//divisor, rate//divisor)
    target16 = narrow(clean); scores = {}
    for name, audio in signals.items():
        if len(audio) != len(clean): raise ValueError('window shape differs')
        mos = SCORER.score_array(audio, rate, kind='near_only')
        try: p = float(pesq(16000, target16, narrow(audio), 'wb')); err = None
        except Exception as exc: p = None; err = str(exc)
        scores[name] = dict(dnsmos=mos, pesq_wb_16k=p, unavailable_reason=err)
    return dict(run=run, id=s['id'], rate=rate, split=s['split'], domain=opt.domain(s),
                noise=s['noise'], snr_db=s['snr_db'], scores=scores, output_sha256=hashes,
                scope='fixed first scheduled speech window; no best-lag/gain search; metric-only common 16k scoring band')


def perceptual(configs, jobs):
    root = opt.ROOT/'perceptual'; root.mkdir(exist_ok=False)
    # Winner configurations are already frozen. These historical holdouts are
    # retrospective generalization checks, never a new blind test.
    selected = [(run,r) for run,r in opt.sources('vehicle') if r['scene']['split'] in ('validation','holdout')]
    deps=[Path(__file__),DEFAULT_ASSETS/'manifest.json',DEFAULT_ASSETS/'sig_bak_ovr.onnx',DEFAULT_ASSETS/'dnsmos_local.py',opt.ROOT/'comparison-variants.json',opt.ROOT/'final-vehicle/identity.json']
    bindings={str(p):opt.paired.replay.ctl.ev.digest(p) for p in deps}
    started=time.monotonic()
    with ProcessPoolExecutor(max_workers=jobs) as pool,(root/'results.jsonl').open('x') as stream:
        futures=[pool.submit(perceptual_task,(root,configs,run,r)) for run,r in selected]
        for i,future in enumerate(as_completed(futures),1):
            stream.write(json.dumps(future.result(),allow_nan=False)+'\n');stream.flush()
            if i%25==0 or i==len(selected):
                opt.save(root/'report.json',dict(state='running',completed=i,total=len(selected),elapsed_seconds=time.monotonic()-started))
    changed=[p for p,h in bindings.items() if opt.paired.replay.ctl.ev.digest(Path(p))!=h]
    opt.save(root/'identity.json',dict(dependencies=bindings))
    opt.save(root/'report.json',dict(state='completed',cases=len(selected),roles_per_case=len(configs)+1,identity_verified=not changed,changed_dependencies=changed,elapsed_seconds=time.monotonic()-started,standard_compliance=False))
    if changed: raise ValueError('perceptual dependency drift')


def frame_summary(clean, before, after, rate):
    hop=rate//100; n=len(clean)//hop
    c,b,a=[x[:n*hop].reshape(n,hop) for x in (clean,before,after)]
    power=np.mean(c*c,axis=1); active=power>=10**(-65/10); den=np.sum(c*c,axis=1)
    alpha=np.divide(np.sum(c*a,axis=1),den,out=np.zeros(n),where=den>1e-12)
    gain=20*np.log10(np.maximum(abs(alpha),1e-12)); steps=abs(np.diff(gain)); pair=active[1:]&active[:-1]
    return dict(frames=n,active_frames=int(active.sum()),oracle_scope='evaluation only',
        projection_p05_db=float(np.percentile(gain[active],5)),projection_median_db=float(np.median(gain[active])),
        active_below_minus6_percent=float(np.mean(gain[active]<-6)*100),
        projection_step_p95_db=float(np.percentile(steps[pair],95)))


def streams(configs, paced_seconds):
    root=opt.ROOT/'streams';root.mkdir(exist_ok=False)
    previous=opt.PROJECT/'build/ns-backend-comparison-20261018/streams'
    old_identity=json.loads((previous/'identity.json').read_text())
    bindings={str(p):opt.paired.replay.ctl.ev.digest(p) for p in (Path(__file__),opt.PROBE,opt.ROOT/'comparison-variants.json')}
    rows=[];started=time.monotonic()
    for meta in old_identity['sources']:
        rate=meta['rate'];source=previous/f'input-{rate}.wav';target=previous/f'clean-{rate}.wav'
        if opt.paired.replay.ctl.ev.digest(source)!=meta['input_sha256'] or opt.paired.replay.ctl.ev.digest(target)!=meta['clean_sha256']:raise ValueError('stream source identity differs')
        before,sr=sf.read(source);clean,cr=sf.read(target)
        if sr!=rate or cr!=rate or len(before)!=120*rate:raise ValueError('native stream shape differs')
        before=np.clip(np.rint(before*32768),-32768,32767)/32768
        # Include two SNR changes in the paced fixture. Sources remain at their
        # original rate; this is a stated edit of three native PCM intervals.
        take=paced_seconds*rate//3
        paced_input=np.concatenate([before[start*rate:start*rate+take] for start in (0,24,48)])
        paced_source=root/f'paced-input-{rate}.wav';sf.write(paced_source,paced_input,rate,subtype='FLOAT')
        for config in configs:
            modes=['shared']+(['native24'] if rate==24000 and config['backend']=='speex' else [])
            for mode in modes:
                prefix=root/config['name']/f'continuous-{rate}-{mode}'
                after,ar,em=call(config,source,prefix,mode)
                if ar!=rate or len(after)!=len(before):raise ValueError('continuous length differs')
                windows=[];resets=[];transitions=[]
                for i,snr in enumerate((30,-10,10,30,-10)):
                    sl=slice((i*24+2)*rate,(i*24+9)*rate)
                    windows.append(dict(block=i,snr_db=snr,before=opt.quality(clean[sl],before[sl],rate),after=opt.quality(clean[sl],after[sl],rate)))
                    block=root/f'block-{rate}-{i}.wav'
                    if not block.exists():sf.write(block,before[i*24*rate:(i+1)*24*rate],rate,subtype='FLOAT')
                    a,_,_=call(config,block,root/config['name']/f'reset-{rate}-{mode}-{i}',mode)
                    resets.append(a)
                    if i:
                        sl=slice(i*24*rate,(i*24+1)*rate)
                        transitions.append(dict(at_s=i*24,output_rms_dbfs=float(10*np.log10(max(np.mean(after[sl]**2),1e-15))),
                            error_vs_target_dbfs=float(10*np.log10(max(np.mean((after[sl]-clean[sl])**2),1e-15)))))
                reset=np.concatenate(resets)
                paced,_,pm=call(config,paced_source,root/config['name']/f'paced-{rate}-{mode}',mode,True)
                # A changed process/reset is intentionally expected to differ;
                # paced/offline equality is checked with the same source length.
                offline,_,_=call(config,paced_source,root/config['name']/f'paced-control-{rate}-{mode}',mode)
                rows.append(dict(variant=config['name'],backend=config['backend'],rate=rate,mode=mode,continuous=em,paced=pm,
                    paced_pcm_bit_exact=bool(np.array_equal(paced,offline)),windows=windows,transitions=transitions,
                    frames=frame_summary(clean,before,after,rate),
                    continuous_vs_reset_different_samples=int(np.count_nonzero(after!=reset)),continuous_vs_reset_rms_error=float(np.sqrt(np.mean((after-reset)**2)))))
                opt.save(root/'progress.json',dict(completed=len(rows),elapsed_seconds=time.monotonic()-started))
    changed=[p for p,h in bindings.items() if opt.paired.replay.ctl.ev.digest(Path(p))!=h]
    opt.save(root/'identity.json',dict(dependencies=bindings,sources=old_identity['sources']))
    opt.save(root/'report.json',dict(state='completed',rows=rows,identity_verified=not changed,changed_dependencies=changed,elapsed_seconds=time.monotonic()-started,
        paced_seconds=len(paced_input)/rate,paced_fixture='three equal native-rate slices starting at 0s/24s/48s; SNR 30/-10/10dB',performance_scope='sequential local aarch64 CPU; not Android or hardware certification',standard_compliance=False))
    if changed or not all(r['paced_pcm_bit_exact'] for r in rows):raise ValueError('stream identity or pacing mismatch')


def native24(configs):
    root=opt.ROOT/'native24';root.mkdir(exist_ok=False);rows=[]
    selected=[(run,r) for scope in ('vehicle','diagnostic') for run,r in opt.sources(scope) if r['scene']['sample_rate']==24000]
    for config in configs:
        if config['backend']!='speex':continue
        for run,original in selected:
            s=original['scene'];origin=opt.PROJECT/'build'/run/'cases'/s['id'];before,sr=sf.read(origin/'before.wav');clean,cr=sf.read(origin/'clean.wav')
            if sr!=24000 or cr!=24000:raise ValueError('native source differs')
            after,ar,em=call(config,origin/'before.wav',root/config['name']/run/s['id']/'result','native24')
            if ar!=24000 or em['core_rate']!=24000 or em['src_delay_samples']!=0 or len(after)!=len(clean):raise ValueError('not genuine native24')
            before=np.clip(np.rint(before*32768),-32768,32767)/32768;segments={};failures=[]
            for name,(lo,hi) in s['windows'].items():
                sl=slice(round(lo*24000),round(hi*24000))
                if name.startswith('near'):
                    qb=opt.quality(clean[sl],before[sl],24000);qa=opt.quality(clean[sl],after[sl],24000)
                    segments[name]=dict(before=qb,after=qa,stoi_improvement=qa['stoi']-qb['stoi'])
                    if qa['stoi']<qb['stoi']-.02:failures.append(name+':stoi')
                    if not -3<=qa['projection_gain_db']<=18:failures.append(name+':projection_gain')
                elif name.startswith('noise_only'):
                    att=opt.paired.replay.ctl.ev.attenuation_measurement(before[sl],after[sl],pcm16=True);segments[name]=att
                    a=att['total_attenuation_db'] if att['total_attenuation_db'] is not None else att['attenuation_lower_bound_db']
                    if a is not None and a<6:failures.append(name+':attenuation')
            if em['clipped_samples_before_write']:failures.append('clipping')
            phase='final-vehicle' if opt.domain(s)!='diagnostic' else 'final-diagnostic'
            shared=json.loads((Path(config.get('evidence_root',str(opt.ROOT)))/phase/config['name']/run/s['id']/'metrics.json').read_text())
            rows.append(dict(variant=config['name'],run=run,id=s['id'],domain=opt.domain(s),split=s['split'],engine=em,segments=segments,
                ns_tradeoff_pass=not failures,failures=failures,shared_pass=shared['ns_tradeoff_pass'],shared_segments=shared['segments']))
    opt.save(root/'report.json',dict(state='completed',rows=rows,cases_per_variant=len(selected),standard_compliance=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['perceptual','streams','native24']);p.add_argument('--jobs',type=int,default=6);p.add_argument('--paced-seconds',type=int,default=12)
    args=p.parse_args()
    if not 1<=args.jobs<=8 or args.paced_seconds<1:p.error('invalid jobs/paced duration')
    configs=json.loads((opt.ROOT/'comparison-variants.json').read_text())
    # Extra best-static checkpoints belong in quality screening; the live
    # benchmark uses the same two original baselines and two frozen winners.
    if args.phase=='perceptual':perceptual(configs,args.jobs)
    elif args.phase=='streams':streams([c for c in configs if c['name'] not in ('webrtc_baseline','speex_baseline')],args.paced_seconds)
    else:native24(configs)


if __name__=='__main__':main()
