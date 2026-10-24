#!/usr/bin/env python3
"""Non-vacuous raw-equivalence, causality and paced-state checks."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import soundfile as sf
import optimize_ns_backends as opt


def main():
    root = opt.ROOT / 'probe-validation-v2'
    root.mkdir(exist_ok=False)
    old = opt.PROJECT / 'build/ns-backend-comparison-20261018/isolation-strengths/artifacts/ns_backend_probe'
    chosen = {}
    for run, row in opt.sources('tune'):
        chosen.setdefault(row['scene']['sample_rate'], (run, row))
    checks = []
    for rate, (run, row) in sorted(chosen.items()):
        source = opt.PROJECT / 'build' / run / 'cases' / row['scene']['id'] / 'before.wav'
        x, sr = sf.read(source)
        x = x[:2*rate]
        base = root / f'input-{rate}.wav'
        altered = root / f'altered-{rate}.wav'
        sf.write(base, x, rate, subtype='FLOAT')
        changed = x.copy(); changed[rate:] = -.5*changed[rate:] + .01
        sf.write(altered, changed, rate, subtype='FLOAT')
        for backend, strength in [('webrtc', 0), ('speex', -12)]:
            def invoke(label, source, params=None, pace=False):
                prefix = root / f'{backend}-{rate}-{label}'
                command = [str(old if params is None else opt.PROBE), backend, str(strength), str(source), str(prefix), 'shared']
                if params is not None: command += list(map(str, params))
                if pace: command += ['--pace-10ms']
                p = subprocess.run(command, capture_output=True, text=True)
                p.check_returncode()
                return sf.read(prefix.with_suffix('.after.wav'))[0], json.loads(prefix.with_suffix('.json').read_text())
            a, _ = invoke('old', base)
            b, meta = invoke('raw', base, [0,12,1,0])
            c, policy = invoke('protected', base, [1,6,.6,60])
            d, _ = invoke('suffix-changed', altered, [1,6,.6,60])
            checks.append(dict(rate=rate, backend=backend, raw_bit_exact=bool(np.array_equal(a,b)),
                policy_changes_pcm=bool(np.any(b!=c)), future_suffix_does_not_change_prefix=bool(np.array_equal(c[:rate],d[:rate])),
                suffix_control_nonvacuous=bool(np.any(c[rate:]!=d[rate:])), shape_finite=bool(a.shape==b.shape==c.shape and np.isfinite(c).all()),
                delay_unchanged=meta['delay_samples']==policy['delay_samples']))
            if rate == 16000:
                e, paced = invoke('paced', base, [1,6,.6,60], True)
                checks[-1].update(paced_pcm_bit_exact=bool(np.array_equal(c,e)), actual_pacing_enabled=paced['paced'], paced_deadline_misses=paced['paced_deadline_misses'])
    invalid = subprocess.run([str(opt.PROBE),'webrtc','0',str(base),str(root/'invalid'),'shared','2','12','1','0'],capture_output=True,text=True)
    report=dict(checks=checks, invalid_rejected=invalid.returncode!=0, invalid_stderr=invalid.stderr)
    boolean_keys=['raw_bit_exact','policy_changes_pcm','future_suffix_does_not_change_prefix','suffix_control_nonvacuous','shape_finite','delay_unchanged']
    report['passed']=report['invalid_rejected'] and all(all(c[k] for k in boolean_keys) and c.get('paced_pcm_bit_exact',True) and c.get('actual_pacing_enabled',True) for c in checks)
    opt.save(root/'report.json',report)
    print(json.dumps(report,indent=2))
    if not report['passed']: sys.exit(1)


if __name__ == '__main__': main()
