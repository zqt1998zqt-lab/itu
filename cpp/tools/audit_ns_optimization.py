#!/usr/bin/env python3
"""Audit parameter identity and exact tuning-to-final reproducibility."""
import json
from pathlib import Path
import optimize_ns_backends as opt


def main():
    configs=json.loads((opt.ROOT/'final-variants.json').read_text());by_name={c['name']:c for c in configs}
    phases=['round1','refine-speex','refine-webrtc','final-vehicle','final-diagnostic','final-clean','final-cold-v2']
    checks=[];drift=[];tuning={};final=[]
    for phase in phases:
        folder=opt.ROOT/phase;report=json.loads((folder/'report.json').read_text());identity=json.loads((folder/'identity.json').read_text())
        if report['state']!='completed' or not report['identity_verified']:raise ValueError('phase failed '+phase)
        for path,sha in identity['dependencies'].items():
            if opt.paired.replay.ctl.ev.digest(Path(path))!=sha:drift.append(dict(phase=phase,path=path))
        rows=[json.loads(l) for l in (folder/'results.jsonl').open()]
        if len(rows)!=report['total']:raise ValueError('row count differs')
        seen=set()
        for r in rows:
            k=(r['variant'],r['original_run'],r['scene']['id'])
            if k in seen:raise ValueError('duplicate row')
            seen.add(k);em=r['engine'];c=r['config']
            for name in ('strength','dry_max','snr_threshold_db','noise_gain','startup_ms'):
                if abs(em[name]-c[name])>1e-5:raise ValueError('runtime parameter differs '+name)
            if em['aec'] or em['agc'] or em['hpf'] or em['backend']!=c['backend']:raise ValueError('NS-only scope violated')
            if phase.startswith('final-') and c!=by_name[r['variant']]:raise ValueError('winner config differs')
            if not phase.startswith('final-'):
                if r['scene']['split']=='holdout':raise ValueError('holdout in search')
                tuning[k]=r
            elif phase=='final-vehicle':final.append(r)
        checks.append(dict(phase=phase,rows=len(rows),identity_verified=True,unique_keys=True,actual_parameters_verified=True))
    exact=[]
    for c in configs:
        pairs=[]
        for r in final:
            if r['variant']!=c['name']:continue
            t=tuning.get((c['source_variant'],r['original_run'],r['scene']['id']))
            if t is not None:pairs.append((t,r))
        if len(pairs)!=262:raise ValueError('tuning overlap incomplete')
        if not all(a['output_sha256']==b['output_sha256'] and a['ns_tradeoff_pass']==b['ns_tradeoff_pass'] and a['segments']==b['segments'] for a,b in pairs):raise ValueError('replay is not exact')
        exact.append(dict(variant=c['name'],pairs=len(pairs),output_hashes_and_metrics_exact=True))
    for phase in ('perceptual','streams'):
        ident=json.loads((opt.ROOT/phase/'identity.json').read_text())
        for path,sha in ident['dependencies'].items():
            if opt.paired.replay.ctl.ev.digest(Path(path))!=sha:drift.append(dict(phase=phase,path=path))
    report=dict(state='completed',passed=not drift,dependency_drift=drift,phase_checks=checks,tuning_final_equivalence=exact,
        scope='NS-only inputs/configs/outputs verified; timings excluded from bit-exact comparisons',standard_compliance=False)
    opt.save(opt.ROOT/'final-audit.json',report)
    if drift:raise ValueError('dependencies changed')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
