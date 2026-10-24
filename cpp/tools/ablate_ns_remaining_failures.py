#!/usr/bin/env python3
"""Diagnostic component ablations of frozen failures. Never select or tune a profile."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import analyze_ns_remaining_frames as audit
import repair_ns_vehicle_v4 as evaluation
import run_audio_eval as meter
import verify_ns_vehicle_v4 as replay

P, R = replay.P, replay.R


def score(row, clean, before, after, engine):
    rate = row['scene']['sample_rate']
    segments, failures, raw_failures = {}, [], []
    f = audit.frames(clean, before, after, rate)
    damage = {}
    for label, (lo, hi) in row['scene']['windows'].items():
        sl = slice(round(lo * rate), min(len(clean), round(hi * rate)))
        c, b, a = (x[sl] for x in (clean, before, after))
        if len(c) < rate // 20:
            continue
        if label.startswith('near'):
            qb, qa = evaluation.quality(c, b, rate), evaluation.quality(c, a, rate)
            if qa['sisdr_db'] is None:
                continue
            if qa['stoi'] is not None and qb['stoi'] is not None and qa['stoi'] < qb['stoi'] - .02:
                failures.append(label + ':stoi')
            if not -3 <= qa['projection_gain_db'] <= 18:
                failures.append(label + ':projection_gain')
            if qa['sisdr_db'] < min(qb['sisdr_db'], 20) - 1:
                raw_failures.append(label + ':raw_sisdr')
            segments[label] = dict(before=qb, after=qa)
            start, stop = round(lo * 100), min(len(f['time_s']), round(hi * 100))
            active = f['target_active'][start:stop]
            mask = active & (f['input_projection_db'][start:stop] >= -3) & (f['output_projection_db'][start:stop] < -6)
            cp = f['clean_power'][start:stop]
            damage[label] = dict(newly_damaged_frames=int(mask.sum()),
                                 newly_damaged_target_energy_percent=float(100 * cp[mask].sum() / max(cp.sum(), 1e-30)))
        elif label.startswith('noise_only'):
            item = meter.attenuation_measurement(b, a, pcm16=True)
            segments[label] = item
            attenuation = item['total_attenuation_db']
            if attenuation is None:
                attenuation = item['attenuation_lower_bound_db']
            if attenuation is not None and attenuation < 6:
                failures.append(label + ':attenuation')
    if engine['clipped_samples_before_write']:
        failures.append('clipping')
    return dict(segments=segments, failures=failures, raw_failures=raw_failures, frame_damage=damage,
                ns_tradeoff_pass=not failures, raw_quality_pass=not failures and not raw_failures)


def execute(task):
    out, scope, row, kind, config = task
    clean, before, original, source, offset = audit.source_arrays(scope, row)
    folder = out / kind / scope / row['variant'] / row['original_run'] / row['scene']['id']
    prefix = folder / 'result'
    after, rate, engine = replay.call(config, source, prefix)
    if rate != row['scene']['sample_rate'] or after.shape != clean.shape:
        raise ValueError('ablation output shape differs')
    result = score(row, clean, before, after, engine)
    if kind == 'reproduce':
        if audit.sha(prefix.with_suffix('.after.wav')) != row['output_sha256']:
            raise ValueError('reproduced output hash differs')
        if result['failures'] != row['failures'] or result['raw_failures'] != row['raw_failures']:
            raise ValueError('reproduced gates differ')
        for label, item in result['segments'].items():
            if label.startswith('near'):
                audit.identity_check(item['before'], row['segments'][label]['before'])
                audit.identity_check(item['after'], row['segments'][label]['after'])
    old = set(row['failures'] + row['raw_failures'])
    new = set(result['failures'] + result['raw_failures'])
    result.update(scope=scope, variant=row['variant'], original_run=row['original_run'], scene=row['scene'],
                  domain=row['domain'], ablation=kind, original_config=row['config'], config=config,
                  original_failures=row['failures'], original_raw_failures=row['raw_failures'],
                  removed_failures=sorted(old - new), added_failures=sorted(new - old),
                  input_sha256=audit.sha(source), clean_sha256=row['clean_sha256'],
                  original_output_sha256=row['output_sha256'], output_sha256=audit.sha(prefix.with_suffix('.after.wav')),
                  engine=engine, original_delay_samples=row['engine']['delay_samples'],
                  alignment='each real path declared delay; no fitted lag or gain', crop_samples=offset,
                  invocation=str(prefix.with_suffix('.invocation.json').relative_to(P)),
                  analysis_only=True, gates_unchanged=True)
    audit.save(folder / 'metrics.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=6)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    if not 1 <= args.jobs <= 8:
        raise ValueError('jobs must be 1..8')
    replay.bindings()
    rows, dependencies = [], [Path(__file__), Path(audit.__file__), Path(evaluation.__file__), Path(replay.__file__), Path(meter.__file__), replay.B]
    for scope in ('vehicle', 'clean', 'cold', 'native24'):
        path = R / ('final-' + scope) / 'results.jsonl'
        dependencies.append(path)
        rows.extend((scope, json.loads(line)) for line in path.open() if not json.loads(line)['raw_quality_pass'])
    if args.limit:
        rows = rows[:args.limit]
    tasks, groups = [], set()
    for scope, row in rows:
        c = row['config']
        tasks.append((out, scope, row, 'no_rumble', dict(c, rumble_hz=0, rumble_kind='linear')))
        # The two WebRTC settings disable one feature: the steady fullband raw
        # protection. Startup protection and the dual NS states stay enabled.
        tasks.append((out, scope, row, 'steady_raw_off', dict(c, quiet_floor_db=-300, repair_mode=min(2, c['repair_mode']))))
        if scope == 'cold':
            tasks.append((out, scope, row, 'startup_off', dict(c, startup_ms=0)))
        if scope == 'clean':
            base = dict(c, backend='bypass', strength=0, dry_max=0, noise_gain=1,
                        startup_ms=0, policy='aligned', mild_strength=0, repair_mode=0,
                        release_ms=0, quiet_floor_db=-300, floor_guard=0)
            tasks.append((out, scope, row, 'filter_only', dict(base, rumble_kind='linear')))
            tasks.append((out, scope, row, 'bypass', dict(base, rumble_hz=0, rumble_kind='linear')))
        group = scope, row['variant']
        if group not in groups:
            groups.add(group)
            tasks.append((out, scope, row, 'reproduce', c))
    bindings = {str(path): audit.sha(path) for path in dependencies}
    audit.save(out / 'identity.json', dict(dependencies=bindings, failure_records=len(rows),
                                         operations=dict(Counter(t[3] for t in tasks)), thresholds_unchanged=True,
                                         no_profile_selection=True, self_check=audit.self_check()))
    started, results = time.monotonic(), []
    audit.save(out / 'report.json', dict(state='running', completed=0, total=len(tasks)))
    with ProcessPoolExecutor(args.jobs) as pool, (out / 'results.jsonl').open('x') as stream:
        for future in as_completed([pool.submit(execute, t) for t in tasks]):
            result = future.result()
            stream.write(json.dumps(result, allow_nan=False) + '\n')
            stream.flush()
            results.append(result)
            if len(results) % 50 == 0 or len(results) == len(tasks):
                audit.save(out / 'report.json', dict(state='running', completed=len(results), total=len(tasks),
                                                    elapsed_seconds=time.monotonic() - started))
                print('ablation', len(results), '/', len(tasks), flush=True)
    summary = {}
    for kind in sorted({r['ablation'] for r in results}):
        selected = [r for r in results if r['ablation'] == kind]
        summary[kind] = dict(rows=len(selected), main_pass=sum(r['ns_tradeoff_pass'] for r in selected),
                             raw_pass=sum(r['raw_quality_pass'] for r in selected),
                             removed_failure_records=sum(bool(r['removed_failures']) for r in selected),
                             added_failure_records=sum(bool(r['added_failures']) for r in selected),
                             removed_failures=dict(Counter(f for r in selected for f in r['removed_failures'])),
                             added_failures=dict(Counter(f for r in selected for f in r['added_failures'])))
    replay.bindings()
    changed = [path for path, value in bindings.items() if audit.sha(Path(path)) != value]
    report = dict(state='completed', completed=len(results), total=len(tasks), identity_verified=not changed,
                  changed_dependencies=changed, operations=summary, elapsed_seconds=time.monotonic() - started,
                  gates_unchanged=True, standard_compliance=False)
    audit.save(out / 'report.json', report)
    if changed:
        raise ValueError('ablation dependency drift')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
