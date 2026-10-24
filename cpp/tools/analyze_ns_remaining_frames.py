#!/usr/bin/env python3
"""Audit every frozen remaining NS failure; diagnostics never change gates."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
from scipy import signal

import repair_ns_vehicle_v4 as frozen
import run_audio_eval as meter

P = Path(__file__).resolve().parents[2]
R = P / 'build/ns-recorded-repair-v4-20261018'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def quantiles(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return dict(n=len(values), p05=float(np.percentile(values, 5)), median=float(np.median(values)),
                p95=float(np.percentile(values, 95))) if len(values) else dict(n=0, p05=None, median=None, p95=None)


def db(x):
    return 10 * np.log10(np.maximum(x, 1e-30))


def longest(mask):
    padded = np.r_[False, mask, False].astype(int)
    starts = np.flatnonzero(np.diff(padded) == 1)
    stops = np.flatnonzero(np.diff(padded) == -1)
    if not len(starts):
        return dict(frames=0, start_frame=None, end_frame_exclusive=None)
    j = int(np.argmax(stops - starts))
    return dict(frames=int(stops[j] - starts[j]), start_frame=int(starts[j]), end_frame_exclusive=int(stops[j]))


def frames(clean, before, after, rate):
    hop = rate // 100
    n = len(clean) // hop
    c, b, a = (x[:n * hop].reshape(n, hop) for x in (clean, before, after))
    cp, bp, ap = (np.mean(x * x, axis=1) for x in (c, b, a))
    noise = b - c
    npower = np.mean(noise * noise, axis=1)
    denominator = np.sum(c * c, axis=1)
    alpha_b = np.divide(np.sum(c * b, axis=1), denominator, out=np.zeros(n), where=denominator > 1e-15)
    alpha_a = np.divide(np.sum(c * a, axis=1), denominator, out=np.zeros(n), where=denominator > 1e-15)
    return dict(time_s=np.arange(n) / 100, clean_dbfs=db(cp), input_dbfs=db(bp), output_dbfs=db(ap),
                clean_power=cp, input_power=bp, output_power=ap, noise_power=npower,
                input_frame_snr_db=db(cp / np.maximum(npower, 1e-30)),
                input_projection_alpha=alpha_b, output_projection_alpha=alpha_a,
                input_projection_db=db(alpha_b ** 2), output_projection_db=db(alpha_a ** 2),
                input_nmse_db=db(np.mean((b - c) ** 2, axis=1) / np.maximum(cp, 1e-30)),
                output_nmse_db=db(np.mean((a - c) ** 2, axis=1) / np.maximum(cp, 1e-30)),
                noise_attenuation_db=db(bp / np.maximum(ap, 1e-30)),
                target_active=cp >= 10 ** (-65 / 10))


def identity_check(a, b):
    for key in ('projection_gain_db', 'sisdr_db', 'stoi'):
        if a[key] is None or b[key] is None:
            if a[key] != b[key]:
                raise ValueError('metric availability differs: ' + key)
        elif abs(a[key] - b[key]) > 1e-9:
            raise ValueError('metric differs: ' + key)


def source_arrays(scope, row):
    origin = P / 'build' / row['original_run'] / 'cases' / row['scene']['id']
    folder = R / ('final-' + scope) / row['variant'] / row['original_run'] / row['scene']['id']
    source = origin / ('clean.wav' if scope == 'clean' else 'before.wav')
    if scope == 'cold':
        source = folder / 'cold-input.wav'
    output = folder / 'result.after.wav'
    if sha(source) != row['input_sha256'] or sha(origin / 'clean.wav') != row['clean_sha256'] or sha(output) != row['output_sha256']:
        raise ValueError('frozen WAV identity differs')
    clean, cr = sf.read(origin / 'clean.wav')
    before, br = sf.read(source)
    after, ar = sf.read(output)
    rate = row['scene']['sample_rate']
    if cr != rate or br != rate or ar != rate or before.ndim != 1:
        raise ValueError('native-rate mono shape differs')
    offset = len(clean) - len(before)
    if scope != 'cold' and offset:
        raise ValueError('unexpected input crop')
    clean = clean[offset:]
    delay = row['engine']['delay_samples']
    if len(after) != len(clean) + delay:
        raise ValueError('fixed-delay output length differs')
    after = after[delay:]
    before = np.clip(np.rint(before * 32768), -32768, 32767) / 32768
    if not all(np.isfinite(x).all() for x in (clean, before, after)):
        raise ValueError('nonfinite PCM')
    return clean, before, after, source, offset


def common_band_diagnostic(c, b, a, rate):
    # Secondary common 100 Hz high-pass only. No lag fitting or gate changes.
    sos = signal.butter(4, 100, fs=rate, btype='highpass', output='sos')
    cc, bb, aa = (signal.sosfilt(sos, x) for x in (c, b, a))
    return dict(before=frozen.quality(cc, bb, rate), after=frozen.quality(cc, aa, rate),
                scope='identical causal 100Hz HPF on target/input/output; diagnostic, not an ITU measurement or revised gate')


def analyze(task):
    root, scope, row = task
    clean, before, after, source, crop = source_arrays(scope, row)
    rate = row['scene']['sample_rate']
    hop = rate // 100
    f = frames(clean, before, after, rate)
    folder = root / scope / row['variant'] / row['original_run'] / row['scene']['id']
    folder.mkdir(parents=True)
    np.savez_compressed(folder / 'frames.npz', **f)
    labels = sorted(set(x.split(':')[0] for x in row['failures'] + row['raw_failures']))
    windows = {}
    for label in labels:
        if label not in row['scene']['windows']:
            continue
        lo, hi = row['scene']['windows'][label]
        start, stop = round(lo * rate), min(len(clean), round(hi * rate))
        c, b, a = (x[start:stop] for x in (clean, before, after))
        first, last = round(lo * 100), min(len(f['time_s']), round(hi * 100))
        valid = f['target_active'][first:last]
        alpha_before = f['input_projection_db'][first:last]
        alpha_after = f['output_projection_db'][first:last]
        added_damage = valid & (alpha_before >= -3) & (alpha_after < -6)
        target_energy = f['clean_power'][first:last]
        item = dict(limits_s=[lo, hi], frame_start=first, frame_stop=last,
                    triggered=[x for x in row['failures'] + row['raw_failures'] if x.startswith(label + ':')],
                    active_frames=int(valid.sum()), total_frames=last - first,
                    input_snr_db=quantiles(f['input_frame_snr_db'][first:last][valid]),
                    target_dbfs=quantiles(f['clean_dbfs'][first:last][valid]),
                    before_projection_db=quantiles(alpha_before[valid]), after_projection_db=quantiles(alpha_after[valid]),
                    input_below_minus6_frames=int(np.sum(valid & (alpha_before < -6))),
                    output_below_minus6_frames=int(np.sum(valid & (alpha_after < -6))),
                    newly_damaged_frames=int(added_damage.sum()),
                    newly_damaged_target_energy_percent=float(100 * target_energy[added_damage].sum() / max(target_energy.sum(), 1e-30)),
                    worst_damage_run=longest(added_damage))
        if label.startswith('near'):
            qb, qa = frozen.quality(c, b, rate), frozen.quality(c, a, rate)
            identity_check(qb, row['segments'][label]['before'])
            identity_check(qa, row['segments'][label]['after'])
            centered_c, centered_b, centered_a = (x - x.mean() for x in (c, b, a))
            denominator = float(np.dot(centered_c, centered_c))
            signed_b = float(np.dot(centered_c, centered_b) / max(denominator, 1e-30))
            signed_a = float(np.dot(centered_c, centered_a) / max(denominator, 1e-30))
            item.update(before=qb, after=qa, signed_window_projection_before=signed_b, signed_window_projection_after=signed_a,
                        before_projection_already_outside_gate=not -3 <= qb['projection_gain_db'] <= 18,
                        common_above100_diagnostic=common_band_diagnostic(c, b, a, rate))
            if label + ':stoi' in row['failures']:
                rolling = []
                width, stride = rate, rate // 4
                for position in range(start, stop - width + 1, stride):
                    sl = slice(position, position + width)
                    cb, ca = frozen.quality(clean[sl], before[sl], rate), frozen.quality(clean[sl], after[sl], rate)
                    rolling.append(dict(start_s=position / rate, duration_s=1, before_stoi=cb['stoi'], after_stoi=ca['stoi'],
                                        delta=ca['stoi'] - cb['stoi'] if ca['stoi'] is not None and cb['stoi'] is not None else None))
                item['rolling_stoi_1s_hop250ms'] = rolling
        elif label.startswith('noise_only'):
            measurement = meter.attenuation_measurement(b, a, pcm16=True)
            expected = row['segments'][label]
            for k, v in measurement.items():
                if v is None:
                    if expected[k] is not None:
                        raise ValueError('attenuation availability differs')
                elif abs(v - expected[k]) > 1e-9:
                    raise ValueError('attenuation differs')
            p = f['input_power'][first:last]
            q = f['output_power'][first:last]
            top = np.argsort(p)[-max(1, int(np.ceil(.1 * len(p)))):]
            mask = np.ones(len(p), dtype=bool)
            mask[top] = False
            per_frame = f['noise_attenuation_db'][first:last]
            tail = {}
            for margin_ms in (20, 50, 100, 250, 500):
                trim = margin_ms * rate // 1000
                if len(b) > trim:
                    tail[str(margin_ms)] = meter.attenuation_measurement(b[trim:], a[trim:], pcm16=True)
            item.update(attenuation=measurement, frame_attenuation_db=quantiles(per_frame),
                        below6db_frames=int(np.sum(per_frame < 6)),
                        low_suppression_run=longest(per_frame < 6),
                        top10_input_energy_percent=float(100 * p[top].sum() / max(p.sum(), 1e-30)),
                        top10_output_energy_percent=float(100 * q[top].sum() / max(q.sum(), 1e-30)),
                        top10_attenuation_db=float(db(p[top].sum() / max(q[top].sum(), 1e-30))),
                        other90_attenuation_db=float(db(p[mask].sum() / max(q[mask].sum(), 1e-30))),
                        clean_nonzero_samples=int(np.count_nonzero(c)),
                        input_power_below_1e_minus12=bool(np.mean(b * b) <= 1e-12),
                        trim_boundary_diagnostic=tail)
        windows[label] = item
    onset = []
    for lo, hi in row['scene']['near_schedule']:
        indices = np.flatnonzero(f['target_active'] & (f['time_s'] >= lo) & (f['time_s'] < hi))
        if not len(indices):
            continue
        index = int(indices[0])
        parts = {}
        for name, begin, end in (('0_50ms', 0, 5), ('50_150ms', 5, 15), ('150_500ms', 15, 50)):
            sl = slice(index + begin, min(len(f['time_s']), index + end))
            parts[name] = dict(input=quantiles(f['input_projection_db'][sl]), output=quantiles(f['output_projection_db'][sl]))
        onset.append(dict(onset_s=index / 100, parts=parts))
    result = dict(scope=scope, variant=row['variant'], original_run=row['original_run'], scene=row['scene'], domain=row['domain'],
                  failures=row['failures'], raw_failures=row['raw_failures'], windows=windows, onset=onset,
                  frame_count=len(f['time_s']), input_sha256=row['input_sha256'], clean_sha256=row['clean_sha256'],
                  output_sha256=row['output_sha256'], source=str(source), crop_samples=crop,
                  alignment='frozen declared delay only; no fitted lag/gain', frame_ms=10,
                  activity='clean frame RMS >= -65dBFS; evaluation only',
                  projection_scope='10ms raw dot products for diagnostics; frozen whole-window centered metrics independently reproduced',
                  algorithm_or_dataset_cause='not assigned from correlation alone')
    save(folder / 'analysis.json', result)
    return result


def self_check():
    rate = 16000
    t = np.arange(rate) / rate
    c = .1 * np.sin(2 * np.pi * 300 * t)
    good, bad = frames(c, c, c, rate), frames(c, c, .5 * c, rate)
    assert np.max(abs(good['output_projection_db'])) < 1e-10
    assert np.max(abs(bad['output_projection_db'] + 6.020599913279624)) < 1e-10
    assert longest(np.array([False, True, True, False]))['frames'] == 2
    assert np.all(bad['output_projection_db'] < -6) and np.all(good['output_projection_db'] >= -3)
    return dict(unchanged_signal_projection_db=0, half_gain_negative_db=-6.020599913279624, negative_control_detected=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=6)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    selected, dependencies = [], [Path(__file__), Path(frozen.__file__), Path(meter.__file__)]
    for scope in ('vehicle', 'clean', 'cold', 'native24'):
        path = R / ('final-' + scope) / 'results.jsonl'
        dependencies.append(path)
        for line in path.open():
            row = json.loads(line)
            if not row['raw_quality_pass']:
                selected.append((root, scope, row))
    if args.limit:
        selected = selected[:args.limit]
    if not selected or not 1 <= args.jobs <= 8:
        raise ValueError('nonempty selection and jobs 1..8 required')
    bindings = {str(path): sha(path) for path in dependencies}
    save(root / 'identity.json', dict(dependencies=bindings, rows=len(selected), no_threshold_changes=True, self_check=self_check()))
    started, results = time.monotonic(), []
    with ProcessPoolExecutor(args.jobs) as pool, (root / 'results.jsonl').open('x') as stream:
        for future in as_completed([pool.submit(analyze, task) for task in selected]):
            row = future.result()
            results.append(row)
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
            if len(results) % 50 == 0 or len(results) == len(selected):
                print('frame-audit', len(results), '/', len(selected), flush=True)
                save(root / 'progress.json', dict(completed=len(results), total=len(selected)))
    changed = [path for path, value in bindings.items() if sha(Path(path)) != value]
    report = dict(state='completed', rows=len(results), unique_parent_scenes=len({(r['original_run'], r['scene']['id']) for r in results}),
                  frames=sum(r['frame_count'] for r in results), windows=sum(len(r['windows']) for r in results),
                  scope_variant_counts=dict(Counter(r['scope'] + ':' + r['variant'] for r in results)),
                  identity_verified=not changed, changed_dependencies=changed, elapsed_seconds=time.monotonic() - started,
                  gates_unchanged=True, standard_compliance=False)
    save(root / 'report.json', report)
    if changed:
        raise ValueError('analysis dependency drift')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
