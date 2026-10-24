#!/usr/bin/env python3
"""Compare a frozen AEC candidate on the original fixed-input factorial."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import time

import probe_repair_aec_candidates as probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / 'artifacts').mkdir()
    cli = root / 'artifacts/itu_audio_cli'
    shutil.copy2(args.cli, cli)
    rows = probe.read_jsonl(probe.SOURCE / 'results.jsonl')
    if len(rows) != 27 or len({r['scene']['id'] for r in rows}) != 27:
        raise ValueError('expected original 27-case factorial')
    source_identity = json.loads((probe.SOURCE / 'identity.json').read_text())
    for name, sha in source_identity['dependencies'].items():
        if probe.digest(Path(name)) != sha:
            raise ValueError(f'factorial dependency changed: {name}')
    before = probe.audit_scene_inputs(rows, phase='before')
    variants = tuple((f'initial_tail_{blocks}', ['--aec-filter-blocks', str(blocks)])
                     for blocks in (20, 64, 128))
    dependencies = [Path(__file__), Path(probe.__file__), Path(probe.ctl.__file__),
                    Path(probe.ctl.legacy.__file__), Path(probe.ctl.ev.__file__),
                    Path(probe.ctl.legacy.acceptance.__file__), probe.ctl.legacy.acceptance.RULES,
                    probe.SOURCE / 'results.jsonl', probe.SOURCE / 'identity.json', cli]
    bindings = {str(p.resolve()): probe.digest(p) for p in dependencies}
    probe.save_json(root / 'identity.json', dict(dependencies=bindings, variants=variants,
                    scope='diagnostic fixed-input AEC; protections retained; no known delay'))
    tasks = []
    for name, flags in variants:
        variant_root = root / name
        for row in rows:
            scene = row['scene']
            probe.prepare_case(probe.SOURCE / 'cases' / scene['id'],
                               variant_root / 'cases' / scene['id'], scene)
            tasks.append((name, flags, scene, row['acceptance'], row['metrics'],
                          variant_root, cli, probe.digest(cli)))
    started = time.monotonic()
    records = []
    with ProcessPoolExecutor(max_workers=4) as pool, (root / 'results.jsonl').open('x') as stream:
        for future in as_completed([pool.submit(probe.run_one, task) for task in tasks]):
            row = future.result()
            records.append(row)
            stream.write(json.dumps(row) + '\n')
            stream.flush()
    after = probe.audit_scene_inputs(rows, phase='after')
    if any(before[k]['input_files'] != after[k]['input_files'] for k in before):
        raise ValueError('PCM changed during experiment')
    changed = [p for p, sha in bindings.items() if probe.digest(Path(p)) != sha]
    summary = {}
    for name, _ in variants:
        group = [r for r in records if r['variant'] == name]
        valid = [r for r in group if not r.get('error')]
        summary[name] = dict(cases=len(group), errors=len(group) - len(valid),
            far_ge_15=sum(r['candidate_values']['far_only_total_attenuation_db'] >= 15 for r in valid),
            linear_erle_ge_6=sum(r['candidate_values']['far_only_linear_erle_engineering_db'] >= 6 for r in valid),
            new_near_dt_failures=sum(len(r['new_near_dt_failures']) for r in valid),
            passed=sum(r['candidate_acceptance']['status'] == 'pass' for r in valid))
    report = dict(state='completed', cases=len(records), variants=summary,
                  identity_verified=not changed, changed_dependencies=changed,
                  elapsed_seconds=time.monotonic() - started, standard_compliance=False)
    probe.save_json(root / 'report.json', report)
    print(json.dumps(report), flush=True)
    if changed or any(v['errors'] for v in summary.values()):
        raise ValueError('identity or execution failure')


if __name__ == '__main__':
    main()
