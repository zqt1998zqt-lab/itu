#!/usr/bin/env python3
"""Add native-16-kHz, direct streaming GTCRN to the frozen Hybrid NS comparison."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import soundfile as sf
import evaluate_hybrid_ns as reference
from verify_gtcrn_direct import pcm16

PROJECT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def score_aligned(clean, before, after, scene, scope, clipped):
    """Same windows, target and gates as evaluate_hybrid_ns; no fitted alignment."""
    rate = scene['sample_rate']; h = rate//100
    failures = []; segments = {}; sisdr = []; stoi = []; noise = []; gains = []
    for label, (lo, hi) in scene['windows'].items():
        sl = slice(round(lo*rate), min(len(clean), round(hi*rate)))
        c, b, a = clean[sl], before[sl], after[sl]
        if len(c) < rate//20:
            continue
        if label.startswith('near'):
            qb, qa = reference.ev.quality(c, b, rate), reference.ev.quality(c, a, rate)
            segments[label] = dict(before=qb, after=qa)
            if qa['sisdr_db'] is None:
                continue
            sisdr.append(qa['sisdr_db']-qb['sisdr_db']); gains.append(qa['projection_gain_db'])
            if qa['sisdr_db'] < min(qb['sisdr_db'], 20)-1:
                failures.append(label+':raw_sisdr')
            if not -3 <= qa['projection_gain_db'] <= 18:
                failures.append(label+':projection_gain')
            if qa['stoi'] is not None and qb['stoi'] is not None:
                stoi.append(qa['stoi']-qb['stoi'])
                if qa['stoi'] < qb['stoi']-.02:
                    failures.append(label+':stoi')
        elif label.startswith('noise_only') and scope != 'clean':
            item = reference.ev.paired.replay.ctl.ev.attenuation_measurement(b, a, pcm16=True)
            segments[label] = item
            attenuation = item.get('total_attenuation_db')
            if attenuation is None:
                attenuation = item.get('attenuation_lower_bound_db')
            if attenuation is not None:
                noise.append(attenuation)
                if attenuation < 6:
                    failures.append(label+':attenuation')
    if not sisdr:
        raise ValueError('no scored speech segment')
    if clipped:
        failures.append('clipping')
    n = len(clean)//h; c = clean[:n*h].reshape(-1, h); a = after[:n*h].reshape(-1, h)
    power = np.sum(c*c, axis=1); active = np.mean(c*c, axis=1) >= 10**(-65/10)
    projection = np.divide(np.sum(c*a, axis=1), power, out=np.zeros(n), where=power > 1e-12)
    pg = 20*np.log10(np.maximum(abs(projection), 1e-12)); onsets = []
    for lo, hi in scene['near_schedule']:
        positions = np.flatnonzero(active & (np.arange(n)/100 >= lo) & (np.arange(n)/100 < hi))
        if len(positions):
            start = int(positions[0])*h
            onsets.append(reference.ev.isolation.projection(clean[start:start+5*h], after[start:start+5*h])['projection_gain_db'])
    return dict(segments=segments, failures=failures, passed=not failures,
        mean_sisdr_improvement_db=float(np.mean(sisdr)),
        mean_stoi_improvement=float(np.mean(stoi)) if stoi else None,
        mean_noise_attenuation_db=float(np.mean(noise)) if noise else None,
        mean_projection_gain_db=float(np.mean(gains)),
        onset_0_50ms_projection_db=float(np.mean([x for x in onsets if x is not None])) if any(x is not None for x in onsets) else None,
        frame_projection_p05_db=float(np.percentile(pg[active], 5)) if active.any() else None,
        frame_under_minus6_fraction=float(np.mean(pg[active] < -6)) if active.any() else None)


def execute(task):
    root, probe, model, run, original, scope, old = task
    scene = copy.deepcopy(original); rate = scene['sample_rate']; h = rate//100
    if rate != 16000:
        raise ValueError('GTCRN comparison forbids SRC and non-16-kHz sources')
    token = hashlib.sha256((run+'/'+scene['id']).encode()).hexdigest()[:20]
    folder = root/'gtcrn_dns3_direct'/token; folder.mkdir(parents=True, exist_ok=False)
    origin = PROJECT/'build'/run/'cases'/scene['id']
    for name in ('before.wav', 'clean.wav'):
        if sha(origin/name) != scene['files'][name]['sha256']:
            raise ValueError('input identity drift')
    before, br = sf.read(origin/'before.wav'); clean, cr = sf.read(origin/'clean.wav')
    if br != rate or cr != rate or before.ndim != 1 or before.shape != clean.shape:
        raise ValueError('input shape/rate differs')
    source = origin/'before.wav'; crop = 0
    if scope == 'clean':
        source = origin/'clean.wav'; before = clean.copy()
    if scope == 'cold':
        active = np.flatnonzero(np.mean(clean[:len(clean)//h*h].reshape(-1, h)**2, axis=1) >= 10**(-65/10))
        if not len(active):
            raise ValueError('cold fixture lacks speech')
        crop = int(active[0])*h; clean = clean[crop:]; before = before[crop:]
        source = folder/'cold-input.wav'; sf.write(source, before, rate, subtype='FLOAT')
        scene['windows'] = {k: (max(0, a-crop/rate), max(0, b-crop/rate)) for k, (a, b) in scene['windows'].items()}
        scene['near_schedule'] = [(max(0, a-crop/rate), max(0, b-crop/rate)) for a, b in scene['near_schedule']]
    actual_input = pcm16(before)
    before = np.clip(np.rint(before*32768), -32768, 32767)/32768
    input_hash = hashlib.sha256(before.astype('<f4').tobytes()).hexdigest()
    if input_hash != old['input_pcm16_sha256'] or crop != old['crop_samples']:
        raise ValueError('frozen Hybrid input or cold-crop mismatch')
    prefix = folder/'result'
    command = [str(probe), '--mic', str(source), '--out', str(prefix), '--model', str(model)]
    proc = subprocess.run(command, capture_output=True, text=True)
    save(folder/'invocation.json', dict(argv=command, returncode=proc.returncode, stderr=proc.stderr))
    proc.check_returncode()
    engine = json.loads(prefix.with_suffix('.json').read_text())
    audio, ar = sf.read(prefix.with_suffix('.after.wav'))
    if (ar != rate or engine['core_rate'] != rate or engine['src_delay_samples'] != 0
            or engine['delay_samples'] != 480 or engine['identity_model'] or engine['inferences'] <= 0
            or len(audio) != len(clean)+480 or not np.isfinite(audio).all()):
        raise ValueError('GTCRN runtime identity, shape or declared delay failed')
    after = audio[480:480+len(clean)]
    scored = score_aligned(clean, before, after, scene, scope, engine['clipped_samples'])
    # A before-metric mismatch would invalidate a paired comparison even if
    # file hashes match. This independently checks target, crop and windows.
    for label, metrics in scored['segments'].items():
        if label.startswith('near') and metrics['before'] != old['segments'][label]['before']:
            raise ValueError('frozen before-metric parity failed: '+label)
    row = dict(variant='gtcrn_dns3_direct', run=run, id=scene['id'], rate=rate,
        split=scene['split'], noise=scene['noise'], snr_db=scene['snr_db'], noise_class=old['noise_class'],
        scope=scope, crop_samples=crop, engine=engine, **scored,
        input_pcm16_sha256=input_hash,
        actual_input_pcm16_sha256=hashlib.sha256(actual_input.astype('<f4').tobytes()).hexdigest(),
        scoring_rounding_difference_samples=int(np.count_nonzero(actual_input != before)),
        output_sha256=sha(prefix.with_suffix('.after.wav')), before_metrics_bit_equal=True,
        standard_compliance=False)
    save(folder/'metrics.json', row)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--scope', choices=('vehicle','clean','cold'), default='vehicle')
    p.add_argument('--jobs', type=int, default=4)
    args = p.parse_args()
    if not 1 <= args.jobs <= 8:
        p.error('jobs must be 1..8')
    root = args.root.resolve(); root.mkdir(parents=True, exist_ok=False)
    previous = [json.loads(line) for line in args.reference.open()]
    previous = [r for r in previous if r['rate'] == 16000]
    old = {(r['run'], r['id']): r for r in previous if r['variant'] == 'legacy_gru32'}
    selected = [(run, r['scene']) for run, r in reference.ev.sources(args.scope) if r['scene']['sample_rate'] == 16000]
    if not selected or set(old) != {(run, r['id']) for run, r in selected}:
        raise ValueError('exact frozen selection differs or is empty')
    paths = [args.probe.resolve(), args.model.resolve(), Path(__file__).resolve(), Path(reference.__file__),
             Path(reference.ev.__file__), Path(reference.ev.paired.replay.ctl.ev.__file__), args.reference.resolve(),
             Path(__file__).with_name('verify_gtcrn_direct.py'), PROJECT/'cpp/tools/gtcrn_direct_probe.cpp', PROJECT/'cpp/src/fft.hpp', PROJECT/'cpp/src/wav.cpp']
    bindings = {str(path): sha(path) for path in paths}
    save(root/'identity.json', dict(bindings=bindings,
        selection=[dict(run=r, scene=s) for r, s in selected],
        native_rate=16000, scope=args.scope, delay_samples=480, sample_lag_search=False,
        thresholds=dict(noise_attenuation_min_db=6, projection_gain_db=[-3,18], stoi_delta_min=-.02,
                        sisdr='after>=min(before,20)-1', clipping=0),
        standard_compliance=False, reference_models_replayed=False))
    tasks = [(root, args.probe.resolve(), args.model.resolve(), r, s, args.scope, old[(r,s['id'])]) for r, s in selected]
    rows = []; start = time.monotonic()
    with ProcessPoolExecutor(args.jobs) as pool, (root/'results.jsonl').open('x') as stream:
        for future in as_completed([pool.submit(execute, task) for task in tasks]):
            row = future.result(); rows.append(row)
            stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush()
            if len(rows) % 25 == 0 or len(rows) == len(tasks):
                save(root/'report.json', dict(state='running', completed=len(rows), total=len(tasks),
                     elapsed_seconds=time.monotonic()-start, timing_not_acceptance=True))
                print(len(rows), '/', len(tasks), flush=True)
    drift = [path for path, expected in bindings.items() if sha(path) != expected]
    with (root/'paired-results.jsonl').open('x') as stream:
        for row in previous+rows:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
    save(root/'summary.json', reference.aggregate(previous+rows))
    save(root/'report.json', dict(state='completed', completed=len(rows), total=len(tasks),
         identity_verified=not drift, changed=drift, elapsed_seconds=time.monotonic()-start,
         before_metrics_verified=len(rows), standard_compliance=False))
    if drift:
        raise ValueError('identity drift after evaluation')


if __name__ == '__main__':
    main()
