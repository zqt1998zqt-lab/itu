#!/usr/bin/env python3
"""Replay an explicit configuration matrix on the original fixed AEC PCM."""
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
    parser.add_argument('--variants', required=True, help='JSON object of variant names to CLI argument arrays')
    parser.add_argument(
        '--allow-dataset-manifest-drift', action='store_true',
        help='allow only the external dataset manifest to differ; all PCM files still require their frozen hashes')
    args = parser.parse_args()
    variants = json.loads(args.variants)
    if not isinstance(variants, dict) or not variants:
        raise ValueError('expected a nonempty variant object')
    for name, flags in variants.items():
        if not name or Path(name).name != name or name in ('.', '..') or not isinstance(flags, list) or not all(isinstance(v, str) for v in flags):
            raise ValueError('invalid variant name/flags')
    rows = probe.read_jsonl(probe.SOURCE / 'results.jsonl')
    if len(rows) != 27 or len({r['scene']['id'] for r in rows}) != 27:
        raise ValueError('expected 27 original cases')
    source_identity = json.loads((probe.SOURCE / 'identity.json').read_text())
    source_drift = [
        name for name, sha in source_identity['dependencies'].items()
        if probe.digest(Path(name)) != sha
    ]
    if args.allow_dataset_manifest_drift:
        unexpected_drift = [
            name for name in source_drift
            if Path(name).name != 'manifest.json'
            or 'datasets/' not in name
        ]
    else:
        unexpected_drift = source_drift
    if unexpected_drift:
        raise ValueError(
            f'factorial dependency changed: {unexpected_drift}')
    before = probe.audit_scene_inputs(rows, phase='before')
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / 'artifacts').mkdir()
    cli = root / 'artifacts/itu_audio_cli'
    shutil.copy2(args.cli, cli)
    dependencies = [Path(__file__), Path(probe.__file__), Path(probe.ctl.__file__),
                    Path(probe.ctl.legacy.__file__), Path(probe.ctl.ev.__file__),
                    Path(probe.ctl.legacy.acceptance.__file__), probe.ctl.legacy.acceptance.RULES,
                    probe.SOURCE / 'results.jsonl', probe.SOURCE / 'identity.json', cli]
    bindings = {str(p.resolve()): probe.digest(p) for p in dependencies}
    probe.save_json(root / 'identity.json', dict(
        dependencies=bindings, variants=variants,
        allowed_source_dependency_drift=(
            source_drift if args.allow_dataset_manifest_drift else []),
        scope='fixed original PCM; unchanged engineering gates; explicit configurations'))
    tasks = []
    for name, flags in variants.items():
        for row in rows:
            scene = row['scene']
            probe.prepare_case(probe.SOURCE / 'cases' / scene['id'], root / name / 'cases' / scene['id'], scene)
            tasks.append((name, flags, scene, row['acceptance'], row['metrics'], root / name, cli, probe.digest(cli)))
    records = []
    started = time.monotonic()
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
    for name in variants:
        group = [r for r in records if r['variant'] == name]
        valid = [r for r in group if not r.get('error')]
        summary[name] = dict(cases=len(group), errors=len(group)-len(valid),
            far_ge_15=sum((r['candidate_values']['far_only_total_attenuation_db'] or -1e9) >= 15 for r in valid),
            linear_erle_ge_6=sum((r['candidate_values']['far_only_linear_erle_engineering_db'] or -1e9) >= 6 for r in valid),
            unavailable_linear=sum(r['candidate_values']['far_only_linear_erle_engineering_db'] is None for r in valid),
            new_near_dt_failures=sum(len(r['new_near_dt_failures']) for r in valid),
            passed=sum(r['candidate_acceptance']['status'] == 'pass' for r in valid))
    complete = len(records) == len(tasks) and not changed and not any(v['errors'] for v in summary.values())
    report = dict(state='completed' if complete else 'failed', cases=len(records), variants=summary,
                  identity_verified=not changed, changed_dependencies=changed,
                  elapsed_seconds=time.monotonic()-started, standard_compliance=False)
    probe.save_json(root / 'report.json', report)
    print(json.dumps(report), flush=True)
    if not complete:
        raise ValueError('identity or execution failure')


if __name__ == '__main__':
    main()
