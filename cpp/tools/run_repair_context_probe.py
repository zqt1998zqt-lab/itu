#!/usr/bin/env python3
"""Run-scoped post-AEC diagnostics on immutable core inputs and V5 gates."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil
import time

import run_frozen_repair_replay as replay
import measure_frozen_repair_v5 as meter

PROJECT = Path(__file__).resolve().parents[2]
EVIDENCE = PROJECT / 'build/repair-sequential-20261017'


def invoke(task):
    root, run, original, cli, flags, variant, stages = task
    row = replay.replay((root, run, original, cli, flags, stages, False))
    row = meter.remeasure((root, row))
    row['variant'] = variant
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cli', type=Path, required=True)
    parser.add_argument('--subset', choices=('probes', 'regressions', 'aec'), default='probes')
    parser.add_argument('--variants', type=json.loads, required=True)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--stages', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.jobs <= 8 or not args.variants:
        raise ValueError('nonempty variants and 1..8 workers required')
    regressions = json.loads((EVIDENCE / 'final-evidence-r19.json').read_text())['regression_cases']
    keys = {(r['original_run'], r['id']) for r in regressions}
    probes = {
        ('controlled-native-full-20261017', s) for s in (
            'vehicle-16000-development-m2-10', 'vehicle-16000-development-m4-09',
            'vehicle-24000-development-m1-08', 'vehicle-24000-validation-m1-09',
            'vehicle-48000-development-m1-03', 'vehicle-48000-holdout-m1-00',
            'vehicle-48000-validation-m1-03', 'vehicle-16000-development-m1-00')}
    selected = [(run, row) for run, row in replay.select_cases('core')
                if (args.subset == 'aec' and row['scene']['far_schedule'])
                or (args.subset == 'regressions' and (run, row['scene']['id']) in keys)
                or (args.subset == 'probes' and (run, row['scene']['id']) in probes)]
    expected = {'probes': len(probes), 'regressions': len(keys), 'aec': 347}[args.subset]
    if len(selected) != expected:
        raise ValueError(f'selection incomplete: {len(selected)} != {expected}')
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / 'artifacts').mkdir()
    cli = root / 'artifacts/itu_audio_cli'
    shutil.copy2(args.cli, cli)
    dependencies = [Path(__file__), Path(replay.__file__), Path(meter.__file__),
                    Path(replay.ctl.__file__), Path(replay.ctl.legacy.__file__),
                    Path(replay.ctl.ev.__file__), Path(replay.ctl.legacy.acceptance.__file__),
                    replay.ctl.legacy.acceptance.RULES, EVIDENCE / 'final-evidence-r19.json']
    for run, manifest in replay.SOURCES:
        dependencies += [PROJECT / 'build' / run / 'results.jsonl',
                         PROJECT / 'datasets' / manifest / 'manifest.json']
    bindings = {str(p): replay.ctl.ev.digest(p) for p in dependencies}
    replay.ctl.legacy.save(root / 'identity.json', dict(dependencies=bindings,
        cli_sha256=replay.ctl.ev.digest(cli), variants=args.variants, subset=args.subset,
        scope='diagnostic disabled-module variants are not acceptance candidates'))
    replay.ctl.legacy.save(root / 'selection.json', [dict(run=run, id=row['scene']['id']) for run, row in selected])
    tasks = [(root / variant, run, row, cli, flags, variant, args.stages)
             for variant, flags in args.variants.items() for run, row in selected]
    started = time.monotonic()
    counts = {variant: Counter() for variant in args.variants}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool, (root / 'results.jsonl').open('x') as stream:
        for row in pool.map(invoke, tasks):
            stream.write(json.dumps(row) + '\n')
            stream.flush()
            counts[row['variant']][row['acceptance']['status']] += 1
    changed = [p for p, sha in bindings.items() if replay.ctl.ev.digest(Path(p)) != sha]
    report = dict(state='completed', cases=len(selected), invocations=len(tasks), counts=counts,
                  elapsed_seconds=time.monotonic() - started, identity_verified=not changed,
                  changed_dependencies=changed, standard_compliance=False)
    replay.ctl.legacy.save(root / 'report.json', report)
    print(json.dumps(report), flush=True)
    if changed:
        raise ValueError(f'dependencies changed: {changed}')


if __name__ == '__main__':
    main()
