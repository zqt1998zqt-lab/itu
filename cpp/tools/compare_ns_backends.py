#!/usr/bin/env python3
"""Paired NS backend replay on immutable, run-scoped vehicle PCM and V5 gates.

This compares complete integrations. Use ns_backend_probe.cpp for isolated NS.
No thresholds, source recordings, manifests or DSP defaults are changed.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import time

import run_frozen_repair_replay as replay
import measure_frozen_repair_v5 as meter


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def selected_cases(scope):
    selected = replay.select_cases('core')
    if scope == 'core':
        return selected
    # Select one deterministic source per rate/noise/SNR band/split/family.
    # Selection uses metadata only, never candidate scores.
    groups = {}
    for run, row in selected:
        s = row['scene']
        if s['mics'] != 1 or s.get('snr_db') not in (-10, 0, 10, 20, 30):
            continue
        key = (s['sample_rate'], s['noise'], s['snr_db'], s['split'], s['family'])
        token = hashlib.sha256((run + '/' + s['id']).encode()).hexdigest()
        if key not in groups or token < groups[key][0]:
            groups[key] = token, (run, row)
    return [entry[1] for _, entry in sorted(groups.items())]


def invoke(task):
    root, run, original, cli, flags, name, stages = task
    row = replay.replay((root, run, original, cli, flags, stages, False))
    row = meter.remeasure((root, row))
    row['variant'] = name
    expected = flags[flags.index('--ns-backend') + 1]
    if row['metrics']['engine']['ns_backend'] != expected:
        raise ValueError('runtime NS backend differs from requested backend')
    if row['metrics']['measurement_version'] != 'actual-static-beam-v5':
        raise ValueError('wrong measurement version')
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--cli', type=Path, required=True)
    p.add_argument('--variants', type=Path, required=True)
    p.add_argument('--scope', choices=('core', 'sweep'), default='core')
    p.add_argument('--jobs', type=int, default=8)
    p.add_argument('--stages', action='store_true')
    args = p.parse_args()
    if not 1 <= args.jobs <= 12:
        p.error('jobs must be 1..12; performance validation runs separately')
    selected = selected_cases(args.scope)
    variants = json.loads(args.variants.read_text())
    if not selected or not variants:
        raise ValueError('nonempty cases and variants required')
    for name, flags in variants.items():
        if not name.replace('_', '').isalnum() or '--ns-backend' not in flags:
            raise ValueError('unsafe variant name or missing explicit backend')
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    artifacts = root / 'artifacts'
    artifacts.mkdir()
    cli = artifacts / 'itu_audio_cli'
    shutil.copy2(args.cli, cli)
    deps = [Path(__file__), Path(replay.__file__), Path(meter.__file__),
            Path(replay.ctl.__file__), Path(replay.ctl.legacy.__file__),
            Path(replay.ctl.ev.__file__), Path(replay.ctl.legacy.acceptance.__file__),
            replay.ctl.legacy.acceptance.RULES, args.variants.resolve()]
    for run, manifest in replay.SOURCES:
        deps.extend([replay.PROJECT / 'build' / run / 'results.jsonl',
                     replay.PROJECT / 'datasets' / manifest / 'manifest.json'])
    bindings = {str(x): replay.ctl.ev.digest(x) for x in deps}
    for i, path in enumerate(deps):
        shutil.copy2(path, artifacts / f'{i:02d}-{path.name}')
    save(root / 'selection.json', [dict(run=run, id=row['scene']['id']) for run, row in selected])
    save(root / 'identity.json', dict(cli_sha256=replay.ctl.ev.digest(cli),
        dependencies=bindings, variants=variants, scope=args.scope,
        measurement_version='actual-static-beam-v5', metadata_only_selection=True))
    total = len(selected) * len(variants)
    started = time.monotonic()
    counts = {name: Counter() for name in variants}
    tasks = [(root / name, run, row, cli, flags, name, args.stages)
             for run, row in selected for name, flags in variants.items()]
    save(root / 'report.json', dict(state='running', total=total, completed=0))
    try:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool, (root / 'results.jsonl').open('x') as stream:
            futures = [pool.submit(invoke, t) for t in tasks]
            for n, future in enumerate(as_completed(futures), 1):
                row = future.result()
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
                counts[row['variant']][row['acceptance']['status']] += 1
                if n % 100 == 0 or n == total:
                    report = dict(state='running', completed=n, total=total, counts=counts,
                                  elapsed_seconds=time.monotonic() - started)
                    save(root / 'report.json', report)
                    print(json.dumps(report), flush=True)
        changed = [path for path, sha in bindings.items() if replay.ctl.ev.digest(Path(path)) != sha]
        report.update(state='completed', identity_verified=not changed, changed_dependencies=changed,
                      cases=len(selected), standard_compliance=False)
        save(root / 'report.json', report)
        if changed:
            raise ValueError(f'dependencies changed: {changed}')
    except BaseException as exc:
        save(root / 'failure.json', dict(error=repr(exc), elapsed_seconds=time.monotonic() - started))
        raise


if __name__ == '__main__':
    main()
