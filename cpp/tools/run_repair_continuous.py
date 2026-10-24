#!/usr/bin/env python3
"""Replay the existing nine uninterrupted streams with a frozen candidate."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import time

import run_controlled_native_scenes as ctl
import measure_frozen_repair_v5 as metering

SOURCE = Path(__file__).resolve().parents[2] / 'build/controlled-native-continuous-v2-20261017'


def replay(task):
    root, row, cli, paced = task
    scene = row['scene']
    origin = SOURCE / 'cases' / scene['id']
    folder = root / 'cases' / scene['id']
    folder.mkdir(parents=True)
    for name, meta in scene['files'].items():
        path = origin / name
        if ctl.ev.digest(path) != meta['sha256']:
            raise ValueError(f'input identity differs: {path}')
        (folder / name).symlink_to(path.resolve())
    ctl.legacy.save(folder / 'scene.json', scene)
    prefix = folder / 'runs/full/result'
    ctl.legacy.invoke(ctl.command_for(scene, folder, prefix, str(cli), paced=paced),
                      prefix, ctl.ev.digest(cli))
    result = ctl.legacy.measure(scene, root, 'full')
    result['original_status'] = row['acceptance']['status']
    result['original_run'] = SOURCE.name
    return metering.remeasure((root, result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli', required=True, type=Path)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--paced', action='store_true', help='single 24k/2-mic real-time stream')
    args = parser.parse_args()
    rows = [json.loads(line) for line in (SOURCE / 'results.jsonl').open()]
    if len(rows) != 9 or len({(r['scene']['sample_rate'], r['scene']['mics']) for r in rows}) != 9:
        raise ValueError('expected all nine rate/microphone combinations')
    if args.paced:
        rows = [r for r in rows if r['scene']['sample_rate'] == 24000 and r['scene']['mics'] == 2]
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / 'artifacts').mkdir()
    cli = root / 'artifacts/itu_audio_cli'
    shutil.copy2(args.cli, cli)
    dependencies = [Path(__file__), Path(metering.__file__), Path(ctl.__file__),
                    Path(ctl.legacy.__file__), Path(ctl.ev.__file__),
                    Path(ctl.legacy.acceptance.__file__), ctl.legacy.acceptance.RULES,
                    SOURCE / 'results.jsonl', SOURCE / 'identity.json', cli]
    bindings = {str(p.resolve()): ctl.ev.digest(p) for p in dependencies}
    ctl.legacy.save(root / 'identity.json', dict(dependencies=bindings, paced=args.paced,
                    source=str(SOURCE), metering='actual-static-beam-v5'))
    started = time.monotonic()
    results = []
    with ProcessPoolExecutor(max_workers=1 if args.paced else 3) as pool, (root / 'results.jsonl').open('x') as stream:
        for future in as_completed([pool.submit(replay, (root, row, cli, args.paced)) for row in rows]):
            result = future.result()
            results.append(result)
            stream.write(json.dumps(result) + '\n')
            stream.flush()
            print(json.dumps(dict(completed=len(results), total=len(rows), id=result['scene']['id'])), flush=True)
    changed = [p for p, sha in bindings.items() if ctl.ev.digest(Path(p)) != sha]
    report = dict(state='completed', cases=len(results), counts=Counter(r['acceptance']['status'] for r in results),
                  paced=args.paced, seconds_per_case=120, identity_verified=not changed,
                  changed_dependencies=changed, elapsed_seconds=time.monotonic() - started,
                  standard_compliance=False, scope='causal stream continuity; fixed original PCM; unchanged engineering gates')
    ctl.legacy.save(root / 'report.json', report)
    print(json.dumps(report), flush=True)
    if changed:
        raise ValueError('dependencies changed')


if __name__ == '__main__':
    main()
