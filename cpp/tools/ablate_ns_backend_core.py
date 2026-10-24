#!/usr/bin/env python3
"""Isolate each frozen backend from the shared protective frontend (diagnostic)."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import json
from pathlib import Path
import time
import ablate_ns_remaining_failures as ablation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=6)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    ablation.replay.bindings()
    tasks, dependencies = [], [Path(__file__), Path(ablation.__file__), ablation.replay.B]
    for scope in ('vehicle', 'clean', 'cold', 'native24'):
        path = ablation.R / ('final-' + scope) / 'results.jsonl'
        dependencies.append(path)
        for line in path.open():
            row = json.loads(line)
            if row['raw_quality_pass']:
                continue
            config = dict(row['config'], dry_max=0, noise_gain=1, startup_ms=0,
                          policy='aligned', release_ms=0, quiet_floor_db=-300,
                          floor_guard=0, repair_mode=0, rumble_hz=0, rumble_kind='linear')
            tasks.append((out, scope, row, 'bare_backend', config))
    dependencies += [Path(m.__file__) for m in (ablation.audit, ablation.evaluation, ablation.replay, ablation.meter)]
    hashes = {str(p): ablation.audit.sha(p) for p in dependencies}
    ablation.audit.save(out / 'identity.json', dict(dependencies=hashes,
                         scope='backend with original strength; shared controls and rumble filter all disabled',
                         analysis_only=True, thresholds_unchanged=True, total=len(tasks)))
    started, results = time.monotonic(), []
    with ProcessPoolExecutor(args.jobs) as pool, (out / 'results.jsonl').open('x') as stream:
        for future in as_completed([pool.submit(ablation.execute, task) for task in tasks]):
            result = future.result()
            results.append(result)
            stream.write(json.dumps(result, allow_nan=False) + '\n')
            stream.flush()
            if len(results) % 50 == 0 or len(results) == len(tasks):
                print('backend-core', len(results), '/', len(tasks), flush=True)
    ablation.replay.bindings()
    changed = [p for p, value in hashes.items() if ablation.audit.sha(Path(p)) != value]
    report = dict(state='completed', rows=len(results), main_pass=sum(r['ns_tradeoff_pass'] for r in results),
                  raw_pass=sum(r['raw_quality_pass'] for r in results), identity_verified=not changed,
                  changed_dependencies=changed, elapsed_seconds=time.monotonic() - started,
                  gates_unchanged=True, standard_compliance=False)
    ablation.audit.save(out / 'report.json', report)
    if changed:
        raise ValueError('backend-core dependency drift')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
