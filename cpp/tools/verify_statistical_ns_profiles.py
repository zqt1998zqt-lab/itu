#!/usr/bin/env python3
"""Verify selected shared protection profiles through CLI and direct C++ NS."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from verify_omlsa_cli import run

def main():
    a=argparse.ArgumentParser(description=__doc__)
    for key in ('cli','probe','profiles','inputs','out'):a.add_argument('--'+key,type=Path,required=True)
    args=a.parse_args();args.cli=args.cli.resolve();args.probe=args.probe.resolve()
    out=args.out.resolve();out.mkdir(exist_ok=False);configs=json.loads(args.profiles.read_text());results=[]
    for rate in (16000,24000,48000):
        source=(args.inputs/f'input-{rate}.wav').resolve()
        for c in configs:
            direct=out/f'direct-{c["name"]}-{rate}'
            argv=[str(args.probe),c['backend'],str(c['strength']),str(source),str(direct),'shared']
            argv += [str(c[k]) for k in ('dry_max','snr_threshold_db','noise_gain','startup_ms','policy','mild_strength','slew_ms','release_ms','quiet_floor_db','floor_guard','repair_mode','rumble_hz','rumble_kind')]
            if 'spectral_dry_guard' in c:argv+=['--spectral-dry-guard='+str(c['spectral_dry_guard'])]
            core=run(argv,direct);reference,_=sf.read(direct.with_suffix('.after.wav'))
            for profile in ('general','vcp'):
                prefix=out/f'{profile}-{c["name"]}-{rate}'
                argv=[str(args.cli),'--profile',profile,'--mic',str(source),'--ns-backend',c['backend'],
                    '--ns-level',str(c['strength']),'--vehicle-ns-control','--no-aec','--no-agc','--no-tx-drc','--no-low-cut','--no-limiter','--out',str(prefix)]
                for key,flag in [('dry_max','dry-max'),('snr_threshold_db','snr-threshold'),('noise_gain','noise-gain'),
                    ('startup_ms','startup-ms'),('policy','policy'),('mild_strength','mild-strength'),('slew_ms','slew-ms'),
                    ('release_ms','release-ms'),('quiet_floor_db','quiet-floor-db'),('floor_guard','floor-guard'),('repair_mode','repair-mode')]:
                    argv+=['--vehicle-ns-'+flag,str(c[key])]
                argv+=['--vehicle-rumble-hz',str(c['rumble_hz'])]
                if c['rumble_kind']=='adaptive':argv+=['--vehicle-adaptive-rumble']
                if 'spectral_dry_guard' in c:argv+=['--vehicle-ns-spectral-dry-guard',str(c['spectral_dry_guard'])]
                report=run(argv,prefix);rendered,ar=sf.read(prefix.with_suffix('.after.wav'))
                assert ar==rate and rendered.shape==reference.shape
                difference=float(np.max(np.abs(reference-rendered)));assert difference==0,(profile,c['name'],rate,difference)
                assert report['ns_backend']==c['backend'] and report['ns_internal_sample_rate']==rate
                assert not report['tx_drc_enabled']
                assert report['vehicle_ns_spectral_dry_guard']==c.get('spectral_dry_guard',0)
                assert report['ns_delay_samples']==core['delay_samples'] and not report['external_vad_enabled']
                results.append(dict(profile=profile,variant=c['name'],rate=rate,maximum_sample_difference=difference,
                                    ns_delay_samples=core['delay_samples']))
    report=dict(state='completed',cases=len(results),results=results,cli_sha256=hashlib.sha256(args.cli.read_bytes()).hexdigest(),
        probe_sha256=hashlib.sha256(args.probe.read_bytes()).hexdigest(),profile_sha256=hashlib.sha256(args.profiles.read_bytes()).hexdigest())
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
