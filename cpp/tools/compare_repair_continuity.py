#!/usr/bin/env python3
"""Compare 45 independent chunks with nine uninterrupted frozen-input streams."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil

import numpy as np
import soundfile as sf

import run_frozen_repair_replay as replay
import run_repair_continuous as continuous
import measure_frozen_repair_v5 as metering


def one(task):
    root, run, row, cli = task
    result = replay.replay((root, run, row, cli, [], False, False))
    return metering.remeasure((root, result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--continuous', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    source = args.continuous.resolve()
    report = json.loads((source / 'report.json').read_text())
    if report['state'] != 'completed' or not report['identity_verified'] or report['cases'] != 9:
        raise ValueError('expected nine completed continuous cases')
    streams = [json.loads(line) for line in (source / 'results.jsonl').open()]
    run = 'controlled-native-full-20261017'
    original = replay.PROJECT / 'build' / run
    original_rows = {r['scene']['id']: r for r in map(json.loads, (original / 'results.jsonl').open())}
    requested = {part['source_case'] for row in streams for part in row['scene']['snr_schedule']}
    if len(requested) != 45 or not requested <= original_rows.keys():
        raise ValueError('expected 45 distinct source chunks')
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / 'artifacts').mkdir()
    cli = root / 'artifacts/itu_audio_cli'
    shutil.copy2(source / 'artifacts/itu_audio_cli', cli)
    deps = [Path(__file__), Path(replay.__file__), Path(continuous.__file__), Path(metering.__file__),
            Path(replay.ctl.__file__), Path(replay.ctl.legacy.__file__), Path(replay.ctl.ev.__file__),
            Path(replay.ctl.legacy.acceptance.__file__), replay.ctl.legacy.acceptance.RULES,
            source / 'identity.json', source / 'results.jsonl', source / 'report.json', original / 'results.jsonl', cli]
    bindings = {str(p.resolve()): replay.ctl.ev.digest(p) for p in deps}
    replay.ctl.legacy.save(root / 'identity.json', dict(dependencies=bindings))
    tasks = [(root, run, original_rows[key], cli) for key in sorted(requested)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        chunks = list(pool.map(one, tasks))
    with (root / 'results.jsonl').open('x') as output:
        for row in chunks:
            output.write(json.dumps(row)+'\n')
    by_id = {row['scene']['id']: row for row in chunks}
    comparisons = []
    for stream in streams:
        scene = stream['scene']
        folder = source / 'cases' / scene['id']
        rate = scene['sample_rate']
        for i, part in enumerate(scene['snr_schedule']):
            chunk = by_id[part['source_case']]
            chunk_folder = original / 'cases' / chunk['scene']['id']
            for name in ('before.wav', 'clean-mics.wav'):
                joined, sr = sf.read(folder / name, start=round(part['start_s']*rate), stop=round(part['stop_s']*rate), always_2d=True)
                separate, sr2 = sf.read(chunk_folder / name, always_2d=True)
                if sr != sr2 or not np.array_equal(joined, separate):
                    raise ValueError(f'continuous input differs: {scene["id"]}/{part["source_case"]}/{name}')
            for key in ('near_only', 'near_late'):
                combined = stream['metrics']['segments'][f'near_chunk{i}_{key}']
                independent = chunk['metrics']['segments'][key]
                before_errors = {name:abs(combined['before'][name]-independent['before'][name])
                    for name in combined['before'] if isinstance(combined['before'][name], (int,float)) and isinstance(independent['before'][name], (int,float))}
                if not before_errors or max(before_errors.values()) > 1e-8:
                    raise ValueError('before metrics differ between matched windows')
                deltas = {name:combined['after'][name]-independent['after'][name]
                    for name in ('sisdr_db','stoi','projection_gain_db')
                    if isinstance(combined['after'].get(name),(int,float)) and isinstance(independent['after'].get(name),(int,float))}
                comparisons.append(dict(id=scene['id'], source_case=part['source_case'], segment=key,
                    snr_db=part['snr_db'], before_errors=before_errors, after_continuous_minus_independent=deltas))
    changed = [p for p, sha in bindings.items() if replay.ctl.ev.digest(Path(p)) != sha]
    replay.ctl.legacy.save(root / 'comparison.json', comparisons)
    result = dict(state='completed', streams=len(streams), chunks=len(chunks), paired_windows=len(comparisons),
                  identical_input_pcm=True, max_before_metric_error=max(v for r in comparisons for v in r['before_errors'].values()),
                  after_delta_ranges={name:[min(r['after_continuous_minus_independent'][name] for r in comparisons),
                                            max(r['after_continuous_minus_independent'][name] for r in comparisons)]
                    for name in ('sisdr_db','stoi','projection_gain_db') if all(name in r['after_continuous_minus_independent'] for r in comparisons)},
                  identity_verified=not changed, changed_dependencies=changed, standard_compliance=False,
                  scope='state continuity comparison; differences are measured, not assumed to vanish')
    replay.ctl.legacy.save(root / 'report.json', result)
    print(json.dumps(result), flush=True)
    if changed:
        raise ValueError('dependencies changed')


if __name__ == '__main__':
    main()
