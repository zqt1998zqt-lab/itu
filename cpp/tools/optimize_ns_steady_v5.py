#!/usr/bin/env python3
"""Paired steady-first NS regression; existing scoring gates remain unchanged."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import repair_ns_vehicle_v4 as ev

P = Path(__file__).resolve().parents[2]
R = P / 'build/ns-steady-opt-20261018'
BASE = P / 'build/ns-recorded-repair-v4-20261018'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save(p, x):
    p.write_text(json.dumps(x, indent=2, allow_nan=False) + '\n')

def init(probe):
    ev.PROBE = Path(probe)

def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--probe', type=Path, required=True)
    a.add_argument('--variants', type=Path, required=True)
    a.add_argument('--phase', required=True)
    a.add_argument('--scope', choices=('screen', 'steady', 'transient', 'clean', 'cold'), required=True)
    a.add_argument('--jobs', type=int, default=8)
    args = a.parse_args()
    if not 1 <= args.jobs <= 8:
        raise ValueError('jobs must be 1..8')
    out = R / args.phase
    out.mkdir(parents=True, exist_ok=False)
    configs = json.loads(args.variants.read_text())
    tags = {(x['original_run'], x['scene_id']): x for x in
            map(json.loads, (P/'build/ns-noise-taxonomy-20261018/samples.jsonl').open())
            if x.get('noise_class') is not None}
    baseline = {(x['variant'], x['original_run'], x['scene']['id']): x for x in
                map(json.loads, (BASE/'final-r5-vehicle/results.jsonl').open())}
    selected = ev.sources(args.scope if args.scope in ('clean', 'cold') else 'vehicle')
    if args.scope not in ('clean', 'cold'):
        selected = [(run, row) for run, row in selected if
                    tags[(run, row['scene']['id'])]['noise_class'] ==
                    ('transient' if args.scope == 'transient' else 'steady')]
    if args.scope == 'screen':
        selected = [(run, row) for run, row in selected if row['scene']['split'] == 'development']
        chosen, cells = {}, defaultdict(list)
        for run, row in selected:
            s = row['scene']; key = run, s['id']
            if any(not baseline[(v, *key)]['raw_quality_pass'] for v in
                   ('speex_v4_balanced', 'webrtc_v4_balanced')):
                chosen[key] = run, row
            band = 'negative' if s['snr_db'] < 0 else 'low' if s['snr_db'] < 10 else 'mid' if s['snr_db'] < 20 else 'high'
            cells[(s['noise'], s['sample_rate'], band)].append((run, row))
        for cell in cells.values():
            run, row = sorted(cell, key=lambda v: sha_text(v[0]+'/'+v[1]['scene']['id']))[0]
            chosen[(run, row['scene']['id'])] = run, row
        selected = [chosen[k] for k in sorted(chosen)]
    if not selected or not configs:
        raise ValueError('empty experiment')
    dependencies = [Path(__file__), Path(ev.__file__), args.probe.resolve(), args.variants.resolve(),
                    P/'build/ns-noise-taxonomy-20261018/samples.jsonl', BASE/'final-r5-vehicle/results.jsonl']
    bindings = {str(p): sha(p) for p in dependencies}
    tasks = [(out, c, run, row, args.scope) for c in configs for run, row in selected]
    save(out/'identity.json', dict(dependencies=bindings, configs=configs,
         selection=[dict(run=run, id=row['scene']['id']) for run, row in selected],
         scope=args.scope, thresholds_unchanged=True, standard_compliance=False))
    started = time.monotonic()
    def report(state, n):
        save(out/'report.json', dict(state=state, completed=n, total=len(tasks), cases=len(selected),
             elapsed_seconds=time.monotonic()-started, standard_compliance=False))
    report('running', 0)
    try:
        with ProcessPoolExecutor(args.jobs, initializer=init, initargs=(str(args.probe.resolve()),)) as pool, (out/'results.jsonl').open('x') as stream:
            for n, future in enumerate(as_completed([pool.submit(ev.execute, t) for t in tasks]), 1):
                row = future.result()
                stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush()
                if n % 25 == 0 or n == len(tasks):
                    report('running', n)
                    print(args.phase, n, '/', len(tasks), flush=True)
        changed = [p for p, h in bindings.items() if sha(Path(p)) != h]
        if changed:
            raise ValueError('identity drift: '+str(changed))
        report('completed', len(tasks))
    except Exception:
        report('failed', sum(1 for _ in (out/'results.jsonl').open()) if (out/'results.jsonl').exists() else 0)
        raise

def sha_text(s):
    return hashlib.sha256(s.encode()).hexdigest()

if __name__ == '__main__':
    main()
