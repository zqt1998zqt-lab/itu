#!/usr/bin/env python3
"""Re-meter frozen outputs in the pre-AEC beam domain; never rewrite v3 results."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import copy
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf

import run_controlled_native_scenes as ctl


def remeasure(task):
    source, original = task
    scene = original['scene']
    folder = source / 'cases' / scene['id']
    if not folder.is_dir():
        folder = source / original['original_run'] / 'cases' / scene['id']
    prefix = folder / 'runs/full/result'
    row = copy.deepcopy(original['metrics'])
    engine = row['engine']
    rate = scene['sample_rate']
    ctl.ev.RATE = rate
    for name, info in scene['files'].items():
        if ctl.ev.digest(folder / name) != info['sha256']:
            raise ValueError(f'input identity differs: {folder / name}')
    if ctl.ev.digest(prefix.with_suffix('.after.wav')) != row['output_sha256']:
        raise ValueError('output identity differs')
    target_name = 'clean-beam.wav' if scene['mics'] > 1 else 'clean.wav'
    target, sr = sf.read(folder / target_name, dtype='float64', always_2d=True)
    if sr != rate or target.shape[1] != 1:
        raise ValueError('target format differs')
    target = target[:, 0]
    if scene['mics'] > 1:
        if not engine['beamforming_before_aec']:
            raise ValueError('expected pre-AEC beam')
        before_path = prefix.with_suffix('.bf.wav')
        delay = engine['bf_delay_samples']
    else:
        before_path = folder / 'before.wav'
        delay = 0
    before, sr = sf.read(before_path, dtype='float64', always_2d=True)
    if sr != rate or before.shape[1] != 1 or len(before) != len(target) + delay:
        raise ValueError(f'pre-AEC baseline length/rate/channels differ: {before_path}')
    before = before[delay:delay + len(target), 0]
    after, sr = sf.read(prefix.with_suffix('.after.wav'), dtype='float64', always_2d=True)
    output_delay = engine['dsp_delay_samples']
    if sr != rate or len(after) != len(target) + output_delay:
        raise ValueError('output length/rate differs')
    after = after[output_delay:output_delay + len(target), 0]
    for name, limits in scene['windows'].items():
        b, a, clean = (ctl.ev.window(x, limits, rate=rate) for x in (before, after, target))
        if name.startswith('near') or name == 'double_talk':
            qb = ctl.ev.quality(clean, b)
            qa = row['segments'][name]['after']
            row['segments'][name]['before'] = qb
            row['segments'][name]['sisdr_improvement_db'] = (
                qa['sisdr_db'] - qb['sisdr_db'] if qa['sisdr_db'] is not None and qb['sisdr_db'] is not None else None)
            row['segments'][name]['stoi_improvement'] = (
                qa['stoi'] - qb['stoi'] if qa['stoi'] is not None and qb['stoi'] is not None else None)
        else:
            row['segments'][name].update(ctl.ev.attenuation_measurement(b, a, pcm16=True))
    row['measurement_version'] = 'beam-domain-v4'
    row['input_baseline_domain'] = 'same-run pre-AEC beam' if scene['mics'] > 1 else 'raw mono'
    row['baseline_sha256'] = ctl.ev.digest(before_path)
    row['baseline_delay_samples'] = delay
    row['target_domain'] = 'nominal geometric clean beam' if scene['mics'] > 1 else 'raw clean mono'
    row['scope'] = 'post-beam DSP comparison; BF benefit and steering error require a separate spatial evaluation'
    gates = ctl.legacy.acceptance.check_scene(scene, row, json.loads(ctl.legacy.acceptance.RULES.read_text()))
    for name in ('noise_only_before_step', 'noise_only_after_step'):
        if name in row['segments']:
            segment = row['segments'][name]
            value = segment['total_attenuation_db']
            if value is None:
                value = segment['attenuation_lower_bound_db']
            gates['gates'].append(dict(segment=name, metric='total_attenuation_db', value=value,
                operator='>=', threshold=6., status='unavailable' if value is None else 'pass' if value >= 6 else 'fail',
                reason='unchanged v3 noise-step threshold', optional=False))
    gates['counts'] = dict(Counter(g['status'] for g in gates['gates']))
    gates['passed'] = all(g['status'] == 'pass' or (g['status'] == 'unavailable' and g['optional']) for g in gates['gates'])
    gates['status'] = 'pass' if gates['passed'] else 'fail' if gates['counts'].get('fail') else 'incomplete_required_measurement'
    result = copy.deepcopy(original)
    result.update(metrics=row, acceptance=gates, v3_acceptance=original['acceptance'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=4, choices=range(1, 9))
    args = parser.parse_args()
    source, out = args.run.resolve(), args.out.resolve()
    rows = [json.loads(l) for l in (source / 'results.jsonl').read_text().splitlines()]
    keys = [(r.get('original_run', str(source)), r['scene']['id']) for r in rows]
    if not rows or len(keys) != len(set(keys)):
        raise ValueError('empty or duplicate run-scoped cases')
    out.mkdir(parents=True, exist_ok=False)
    deps = [Path(__file__), Path(ctl.ev.__file__), Path(ctl.legacy.acceptance.__file__),
            ctl.legacy.acceptance.RULES, source / 'results.jsonl', source / 'identity.json']
    bindings = {str(p.resolve()): ctl.ev.digest(p) for p in deps}
    ctl.legacy.save(out / 'identity.json', dict(dependencies=bindings, measurement_version='beam-domain-v4', source=str(source)))
    started = time.monotonic()
    counts, transitions = Counter(), Counter()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool, (out / 'results.jsonl').open('x') as stream:
        for row in pool.map(remeasure, [(source, r) for r in rows], chunksize=8):
            stream.write(json.dumps(row) + '\n')
            counts[row['acceptance']['status']] += 1
            transitions[row['v3_acceptance']['status'] + '->' + row['acceptance']['status']] += 1
    changed = [p for p, h in bindings.items() if ctl.ev.digest(Path(p)) != h]
    ctl.legacy.save(out / 'report.json', dict(state='completed', cases=len(rows), counts=counts,
        v3_to_v4_transitions=transitions, elapsed_seconds=time.monotonic() - started,
        identity_verified=not changed, changed_dependencies=changed, standard_compliance=False))
    if changed:
        raise ValueError(f'dependency changed: {changed}')
    print(json.dumps(dict(cases=len(rows), counts=counts, transitions=transitions)))


if __name__ == '__main__':
    main()
