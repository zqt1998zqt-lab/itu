#!/usr/bin/env python3
"""Audit and prune completed, explicitly listed historical DSP output files.

Source PCM, symlinks, hardlinks, metadata, frozen binaries, current runs and
final/streaming results are outside the deletion allowlist. Preview is default.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import stat
import time

ROOT = Path(__file__).resolve().parents[2]
RELATIVE_ROOTS = [
    'build/ns-optimization-20261018/round1',
    'build/ns-optimization-20261018/refine-speex',
    'build/ns-optimization-20261018/refine-webrtc',
    'build/ns-refinement-v2-20261018/round1',
    'build/ns-refinement-v2-20261018/round2',
    'build/ns-refinement-v2-20261018/round2-clean',
    'build/ns-refinement-v2-20261018/round2-cold',
    'build/ns-refinement-v2-20261018/development-clean',
    'build/ns-refinement-v2-20261018/development-cold',
    'build/ns-backend-comparison-20261018/aec-strengths',
    'build/ns-backend-comparison-20261018/vcp-strengths',
    'build/ns-backend-comparison-20261018/isolation-strengths',
    'build/repair-sequential-20261017/context-r35-lowgain',
    'build/repair-sequential-20261017/context-r36-lowfloor',
    'build/repair-sequential-20261017/context-r37-hold',
    'build/repair-sequential-20261017/context-r42-peakveto',
]
OUTPUT_NAMES = {
    'result.after.wav', 'result.aec.wav', 'result.bf.wav',
    'result.linear-aec.wav',
}


def walk(root):
    for current, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = [n for n in dirs if not (Path(current) / n).is_symlink()]
        for name in names:
            yield Path(current) / name


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def signature(path):
    s = path.lstat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_nlink]


def available():
    s = os.statvfs(ROOT)
    return s.f_bavail * s.f_frsize


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    keep = set()
    retained = {}
    completed = {}
    roots = [ROOT / r for r in RELATIVE_ROOTS]
    # Retain one pass and one failure at each rate for every old variant.
    for root in roots:
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f'root missing or symlink: {root}')
        report = json.loads((root / 'report.json').read_text())
        if report.get('state') != 'completed' or not report.get('identity_verified'):
            raise ValueError(f'not an identity-verified completed batch: {root}')
        completed[str(root.relative_to(ROOT))] = report
        cells = set()
        with (root / 'results.jsonl').open() as stream:
            for line in stream:
                row = json.loads(line)
                scene = row['scene']
                variant = row.get('variant', 'candidate')
                passed = row.get('raw_quality_pass', row.get('acceptance', {}).get('passed'))
                cell = variant, scene['sample_rate'], passed
                if cell in cells:
                    continue
                cells.add(cell)
                direct = root / variant / row['original_run'] / scene['id']
                nested = root / row['original_run'] / 'cases' / scene['id'] / 'runs/full'
                nested_variant = root / variant / row['original_run'] / 'cases' / scene['id'] / 'runs/full'
                for folder in (direct, nested, nested_variant):
                    if folder.is_dir():
                        keep.add(folder)
    # Snapshot all non-output files in the allowed roots: metadata and frozen
    # artifacts remain measurable after cleanup, without rewriting old reports.
    rows = []
    skipped = Counter()
    for root in roots:
        for path in walk(root):
            if path.is_symlink():
                continue
            s = path.lstat()
            if not stat.S_ISREG(s.st_mode):
                continue
            rel = path.relative_to(ROOT).as_posix()
            if path.name not in OUTPUT_NAMES or 'artifacts' in path.relative_to(root).parts:
                retained[rel] = signature(path)
            elif path.parent in keep:
                retained[rel] = signature(path)
                skipped['representative_files'] += 1
            elif s.st_nlink != 1:
                retained[rel] = signature(path)
                skipped['hardlinked_files'] += 1
            else:
                rows.append({'path': rel, 'signature': signature(path),
                             'allocated_bytes': s.st_blocks * 512,
                             'root': root.relative_to(ROOT).as_posix()})
    candidates = {r['path']: r for r in rows}
    # A downstream symlink is an actual dependency, including symlinked dirs.
    linked = set()
    root_prefix = str(ROOT) + os.sep
    target_prefixes = tuple(str(root) + os.sep for root in roots)
    pending = [str(ROOT / 'build')]
    while pending:
        current = pending.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                if entry.is_symlink():
                    target = os.path.normpath(os.path.join(current, os.readlink(entry.path)))
                    if not target.startswith(root_prefix):
                        continue
                    rel = target[len(root_prefix):]
                    if rel in candidates:
                        linked.add(rel)
                    elif target.startswith(target_prefixes) or any(str(r).startswith(target + os.sep) for r in roots):
                        prefix = rel + '/'
                        linked.update(r for r in candidates if r.startswith(prefix))
                elif entry.is_dir(follow_symlinks=False):
                    pending.append(entry.path)
    for rel in linked:
        retained[rel] = candidates[rel]['signature']
    rows = [r for r in rows if r['path'] not in linked]
    skipped['referenced_by_symlink'] = len(linked)
    with (out / 'plan.jsonl').open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row) + '\n')
    write(out / 'retained-signatures.json', retained)
    write(out / 'representative-folders.json', sorted(str(p.relative_to(ROOT)) for p in keep))
    write(out / 'completed-batches.json', completed)
    groups = Counter()
    for row in rows:
        groups[row['root']] += row['allocated_bytes']
    summary = {'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'files': len(rows), 'allocated_bytes': sum(groups.values()),
               'by_root': dict(groups), 'skipped': dict(skipped),
               'retained_files': len(retained), 'available_before': available(),
               'script_sha256': sha(Path(__file__)), 'plan_sha256': sha(out / 'plan.jsonl')}
    write(out / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def apply(out):
    summary = json.loads((out / 'summary.json').read_text())
    if summary['plan_sha256'] != sha(out / 'plan.jsonl') or summary['script_sha256'] != sha(Path(__file__)):
        raise ValueError('plan or cleaner changed')
    retained = json.loads((out / 'retained-signatures.json').read_text())
    for rel, expected in retained.items():
        if signature(ROOT / rel) != expected:
            raise ValueError(f'retained evidence changed: {rel}')
    count = allocated = 0
    start = available()
    with (out / 'deleted.jsonl').open('x') as log:
        for line in (out / 'plan.jsonl').open():
            row = json.loads(line)
            path = ROOT / row['path']
            if row['root'] not in RELATIVE_ROOTS or not path.is_relative_to(ROOT / row['root']):
                raise ValueError('outside explicit allowlist')
            if path.name not in OUTPUT_NAMES or 'artifacts' in path.parts or path.is_symlink():
                raise ValueError('protected path in plan')
            if signature(path) != row['signature'] or row['signature'][-1] != 1:
                raise ValueError(f'candidate changed: {path}')
            path.unlink()
            log.write(json.dumps(row) + '\n')
            count += 1
            allocated += row['allocated_bytes']
    for rel, expected in retained.items():
        if signature(ROOT / rel) != expected:
            raise ValueError(f'retained evidence changed: {rel}')
    result = {'state': 'completed', 'deleted_files': count, 'deleted_allocated_bytes': allocated,
              'available_before': start, 'available_after': available(),
              'retained_files_verified': len(retained),
              'plan_sha256': summary['plan_sha256'], 'script_sha256': summary['script_sha256']}
    write(out / 'result.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    apply(args.audit) if args.apply else prepare(args.audit)
