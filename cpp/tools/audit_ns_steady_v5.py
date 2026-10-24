#!/usr/bin/env python3
"""Read-only paired metric audit against r5; never changes historical gates."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import statistics

P = Path(__file__).resolve().parents[2]
BASE = P/'build/ns-recorded-repair-v4-20261018'

def rows(path):
    return [json.loads(line) for line in path.open()]

def key(r):
    return r['config']['backend'], r['original_run'], r['scene']['id']

def attenuation(x):
    a = x.get('total_attenuation_db')
    return a if a is not None else x.get('attenuation_lower_bound_db')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder', type=Path)
    args = p.parse_args()
    folder = args.folder.resolve()
    state = json.loads((folder/'report.json').read_text())
    if state['state'] != 'completed':
        raise ValueError('audit requires completed batch')
    scope = json.loads((folder/'identity.json').read_text())['scope']
    old_scope = scope if scope in ('clean', 'cold') else 'vehicle'
    baseline = {key(r): r for r in rows(BASE/f'final-r5-{old_scope}'/'results.jsonl')
                if r['variant'] in ('speex_v4_balanced', 'webrtc_v4_balanced')}
    groups = defaultdict(list)
    for r in rows(folder/'results.jsonl'):
        groups[r['variant']].append(r)
    report = {}
    for name, rs in groups.items():
        old = [baseline[key(r)] for r in rs]
        pairs = list(zip(old, rs))
        if len({key(r) for r in rs}) != len(rs):
            raise ValueError('duplicate paired case')
        if any(o['clean_sha256'] != r['clean_sha256'] or
               (scope != 'cold' and o['input_sha256'] != r['input_sha256']) or
               o['scene'] != r['scene'] for o, r in pairs):
            raise ValueError('unpaired input, target or scene')
        noise_delta, stoi_delta, si_delta, onset_delta = [], [], [], []
        noise_regressions, detail = [], []
        for o, r in pairs:
            for label, v in r['segments'].items():
                ov = o['segments'][label]
                if label.startswith('noise_only'):
                    a, b = attenuation(v), attenuation(ov)
                    if a is not None and b is not None:
                        noise_delta.append(a-b)
                        if b >= 6 and a < 6:
                            noise_regressions.append(dict(run=r['original_run'], id=r['scene']['id'], window=label, before=b, after=a))
                elif label.startswith('near'):
                    for field, target in (('stoi', stoi_delta), ('sisdr_db', si_delta)):
                        a, b = v['after'][field], ov['after'][field]
                        if a is not None and b is not None:
                            target.append(a-b)
            a, b = r['onset_0_50ms_projection_db'], o['onset_0_50ms_projection_db']
            if a is not None and b is not None:
                onset_delta.append(a-b)
            detail.append(dict(run=r['original_run'], id=r['scene']['id'],
                 snr=r['scene']['snr_db'], rate=r['scene']['sample_rate'], noise=r['scene']['noise'],
                 before_raw=o['raw_quality_pass'], after_raw=r['raw_quality_pass'],
                 added_failures=sorted(set(r['failures']+r['raw_failures'])-set(o['failures']+o['raw_failures']))))
        def spread(values):
            return dict(min=min(values), median=statistics.median(values), max=max(values)) if values else None
        def median_field(field):
            return {k: statistics.median(r[field] for r in xs if r[field] is not None)
                    for k, xs in (('before', old), ('after', rs))
                    if any(r[field] is not None for r in xs)}
        losses = [d for d in detail if d['before_raw'] and not d['after_raw']]
        max_delay = max(r['engine']['delay_samples']-o['engine']['delay_samples'] for o, r in pairs)
        guards = dict(no_new_raw_failures=not losses, no_new_noise_window_failures=not noise_regressions,
             noise_drop_le_05=not noise_delta or min(noise_delta) >= -.5,
             stoi_drop_le_001=not stoi_delta or min(stoi_delta) >= -.01,
             sisdr_drop_le_1=not si_delta or min(si_delta) >= -1,
             no_added_delay=max_delay <= 0,
             no_clipping=not any(r['engine']['clipped_samples_before_write'] for r in rs))
        report[name] = dict(cases=len(rs), config=rs[0]['config'],
             main_pass_before=sum(o['ns_tradeoff_pass'] for o in old), main_pass_after=sum(r['ns_tradeoff_pass'] for r in rs),
             raw_pass_before=sum(o['raw_quality_pass'] for o in old), raw_pass_after=sum(r['raw_quality_pass'] for r in rs),
             gained=sum(not d['before_raw'] and d['after_raw'] for d in detail), lost=len(losses), losses=losses,
             noise_window_regressions=noise_regressions,
             deltas=dict(noise_attenuation_db=spread(noise_delta), stoi=spread(stoi_delta), sisdr_db=spread(si_delta), onset_projection_db=spread(onset_delta)),
             medians={f: median_field(f) for f in ('mean_noise_attenuation_db', 'mean_stoi_improvement', 'mean_sisdr_improvement_db', 'onset_0_50ms_projection_db')},
             remaining_failures=dict(Counter(f for r in rs for f in r['failures']+r['raw_failures'])),
             max_added_delay_samples=max_delay, guards=guards, guards_pass=all(guards.values()),
             output_identical=sum(o['output_sha256']==r['output_sha256'] for o, r in pairs),
             process_p99_ms=statistics.median(r['engine']['p99_process_ms'] for r in rs),
             standard_compliance=False)
        with (folder/f'{name}-paired.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(detail[0])); writer.writeheader(); writer.writerows(detail)
    (folder/'paired-audit.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    for name, x in report.items():
        print(json.dumps(dict(variant=name, cases=x['cases'], raw=[x['raw_pass_before'],x['raw_pass_after']],
             gained=x['gained'], lost=x['lost'], guards=x['guards'], deltas=x['deltas']), allow_nan=False))

if __name__ == '__main__':
    main()
