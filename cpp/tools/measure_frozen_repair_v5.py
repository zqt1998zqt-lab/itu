#!/usr/bin/env python3
"""Re-meter immutable PCM in the actual static beam domain, with tap verification."""
import os

# One thread per worker. numpy/OpenBLAS otherwise spawns one thread per core in
# every worker process: 8 workers on a 20-core host became 160 threads fighting
# for 20 cores and 4,412 cases took 729 s instead of 90 s. Keep these defaulted
# so a bare invocation cannot reproduce that.
for _thread_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

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


def static_beam(audio, delays):
    audio = np.asarray(audio)
    delays = np.asarray(delays, dtype=np.float64)
    if audio.ndim != 2 or len(delays) != audio.shape[1] or not np.isfinite(delays).all() or (delays < 0).any():
        raise ValueError('invalid static beam input/delays')
    delay = int(np.ceil(delays.max()))
    n = len(audio)
    time = np.arange(n + delay)
    beam = sum(np.interp(time - offset, np.arange(n + delay + 1), np.pad(audio[:, channel], (0, delay + 1)), left=0., right=0.)
               for channel, offset in enumerate(delays)) / audio.shape[1]
    return beam[delay:delay + n], delay


def verify_static_tap(before, measured, delays):
    expected, delay = static_beam(before, delays)
    # The CLI writes PCM16; interpolation and channel summation use float32.
    tolerance = 2. / 32768.
    error = float(np.max(np.abs(expected - measured)))
    if error > tolerance:
        raise ValueError(f'BF tap differs from actual capture/steering: max_abs_error={error}')
    return dict(max_abs_error=error, tolerance=tolerance, verified=True, delay=delay)


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
    target_domain = 'raw clean mono'
    tap_audit = None
    matched_target = True
    if scene['mics'] > 1:
        invocation = json.loads((prefix.parent / 'invocation.json').read_text())
        argv = invocation['argv']
        spatial = engine['spatial_mvdr'] or engine['wind_noise_control'] or engine['spatial_delay_samples'] != 0
        if spatial:
            matched_target = False
            target_domain = 'nominal clean beam; adaptive spatial transfer not identifiable from scalar metadata'
        else:
            if '--beam-delays' not in argv:
                raise ValueError('missing actual beam steering invocation')
            delays = [float(v) for v in argv[argv.index('--beam-delays') + 1].split(',')]
            capture, sr = sf.read(folder / 'before.wav', dtype='float64', always_2d=True)
            if sr != rate or capture.shape != (len(target), scene['mics']):
                raise ValueError('capture format differs')
            tap_audit = verify_static_tap(capture, before, delays)
            if delay != tap_audit['delay']:
                raise ValueError('BF delay differs from actual static steering')
            clean_mics, sr = sf.read(folder / 'clean-mics.wav', dtype='float64', always_2d=True)
            if sr != rate or clean_mics.shape != capture.shape:
                raise ValueError('clean microphone format differs')
            target, _ = static_beam(clean_mics, delays)
            target_domain = 'clean microphones through actual static steering; nominal steering evaluated separately'
    after, sr = sf.read(prefix.with_suffix('.after.wav'), dtype='float64', always_2d=True)
    output_delay = engine['dsp_delay_samples']
    if sr != rate or after.shape[1] != 1 or len(after) != len(target) + output_delay:
        raise ValueError('output length/rate differs')
    after = after[output_delay:output_delay + len(target), 0]
    for name, limits in scene['windows'].items():
        b, a, clean = (ctl.ev.window(x, limits, rate=rate) for x in (before, after, target))
        if name.startswith('near') or name == 'double_talk':
            qb = ctl.ev.quality(clean, b)
            qa = ctl.ev.quality(clean, a)
            row['segments'][name]['after'] = qa
            row['segments'][name]['before'] = qb
            row['segments'][name]['sisdr_improvement_db'] = (
                qa['sisdr_db'] - qb['sisdr_db'] if qa['sisdr_db'] is not None and qb['sisdr_db'] is not None else None)
            row['segments'][name]['stoi_improvement'] = (
                qa['stoi'] - qb['stoi'] if qa['stoi'] is not None and qb['stoi'] is not None else None)
        else:
            row['segments'][name].update(ctl.ev.attenuation_measurement(b, a, pcm16=True))
    row['measurement_version'] = 'actual-static-beam-v5'
    row['input_baseline_domain'] = 'same-run pre-AEC beam' if scene['mics'] > 1 else 'raw mono'
    row['baseline_sha256'] = ctl.ev.digest(before_path)
    row['baseline_delay_samples'] = delay
    row['target_domain'] = target_domain
    row['matched_static_target'] = matched_target
    row['bf_tap_verification'] = tap_audit
    row['output_channels_verified'] = True
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
    parser.add_argument('--jobs', type=int, default=8, choices=range(1, 33))
    args = parser.parse_args()
    cpu = os.cpu_count() or 8
    if args.jobs > cpu:
        print(f"warning: --jobs {args.jobs} exceeds {cpu} CPUs; BLAS threads are already pinned to 1 per worker")
    source, out = args.run.resolve(), args.out.resolve()
    rows = [json.loads(l) for l in (source / 'results.jsonl').read_text().splitlines()]
    keys = [(r.get('original_run', str(source)), r['scene']['id']) for r in rows]
    if not rows or len(keys) != len(set(keys)):
        raise ValueError('empty or duplicate run-scoped cases')
    out.mkdir(parents=True, exist_ok=False)
    deps = [Path(__file__), Path(ctl.ev.__file__), Path(ctl.legacy.acceptance.__file__),
            ctl.legacy.acceptance.RULES, source / 'results.jsonl', source / 'identity.json']
    bindings = {str(p.resolve()): ctl.ev.digest(p) for p in deps}
    ctl.legacy.save(out / 'identity.json', dict(dependencies=bindings, measurement_version='actual-static-beam-v5', source=str(source)))
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
