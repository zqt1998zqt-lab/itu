#!/usr/bin/env python3
"""Sequential 10 ms paced native PCM checks for frozen statistical NS profiles."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--probe',type=Path,required=True);a.add_argument('--profiles',type=Path,required=True)
    a.add_argument('--inputs',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    args=a.parse_args();args.probe=args.probe.resolve();out=args.out.resolve();out.mkdir(exist_ok=False)
    configs=[c for c in json.loads(args.profiles.read_text()) if c['backend'] in ('omlsa_imcra','super_gaussian')]
    if not configs:raise ValueError('no statistical NS profiles')
    results=[];start=time.monotonic();input_hashes={}
    for rate in (16000,24000,48000):
        source=(args.inputs/f'input-{rate}.wav').resolve();input_hashes[str(source)]=sha(source)
        for c in configs:
            prefix=out/f'{c["name"]}-{rate}'
            argv=[str(args.probe),c['backend'],str(c['strength']),str(source),str(prefix),'shared']
            argv += [str(c[k]) for k in ('dry_max','snr_threshold_db','noise_gain','startup_ms','policy','mild_strength','slew_ms','release_ms','quiet_floor_db','floor_guard','repair_mode','rumble_hz','rumble_kind')]
            if 'spectral_dry_guard' in c:argv+=['--spectral-dry-guard='+str(c['spectral_dry_guard'])]
            argv+=['--pace-10ms']
            r=subprocess.run(argv,capture_output=True,text=True)
            prefix.with_suffix('.invocation.json').write_text(json.dumps(dict(argv=argv,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
            r.check_returncode();report=json.loads(prefix.with_suffix('.json').read_text())
            assert report['backend']==c['backend']
            assert report.get('spectral_dry_guard',0)==c.get('spectral_dry_guard',0)
            assert report['io_rate']==report['core_rate']==rate and report['src_delay_samples']==0
            assert report['core_ns_delay_samples']==rate//100
            assert report['frames']==600 and report['paced']
            results.append(dict(variant=c['name'],rate=rate,report=report))
    report=dict(state='completed',sequential=True,results=results,
                tested_frames=sum(r['report']['frames'] for r in results),
                over_10ms=sum(r['report']['over_10ms'] for r in results),
                deadline_misses=sum(r['report']['paced_deadline_misses'] for r in results),
                probe_sha256=sha(args.probe),profile_sha256=sha(args.profiles),input_sha256=input_hashes,
                elapsed_seconds=time.monotonic()-start,scope='local aarch64 CPU; native 6-second real PCM crops; no Android claim')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
