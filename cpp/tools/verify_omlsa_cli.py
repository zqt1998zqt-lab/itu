#!/usr/bin/env python3
"""Verify real CLI selection, native PCM timebase and active AEC/AGC routes."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import soundfile as sf
import repair_ns_vehicle_v4 as ev

P=Path(__file__).resolve().parents[2]

def run(argv,prefix):
    prefix.parent.mkdir(parents=True,exist_ok=True)
    r=subprocess.run(argv,capture_output=True,text=True)
    (prefix.parent/(prefix.name+'.invocation.json')).write_text(json.dumps(dict(argv=argv,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
    r.check_returncode()
    return json.loads(prefix.with_suffix('.json').read_text())

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--cli',type=Path,required=True);a.add_argument('--probe',type=Path,required=True)
    a.add_argument('--out',type=Path,required=True);a.add_argument('--backend',choices=('omlsa_imcra','super_gaussian'),default='omlsa_imcra');args=a.parse_args()
    args.cli=args.cli.resolve();args.probe=args.probe.resolve();out=args.out.resolve();out.mkdir(exist_ok=False)
    results=[]
    for rate in (16000,24000,48000):
        original,row=next((r,s) for r,s in ev.sources('vehicle') if s['scene']['sample_rate']==rate)
        source=P/'build'/original/'cases'/row['scene']['id']/'before.wav'
        assert hashlib.sha256(source.read_bytes()).hexdigest()==row['scene']['files']['before.wav']['sha256']
        pcm,sr=sf.read(source);assert sr==rate
        # Same-rate crop only; no rate conversion or synthetic speech.
        path=out/f'input-{rate}.wav';sf.write(path,pcm[:6*rate],rate,subtype='FLOAT')
        direct=out/f'direct-{rate}'
        raw=run([str(args.probe),args.backend,'1',str(path),str(direct),'shared','0','3','1','0','aligned','0','50','0','-300','0','0','0','linear'],direct)
        for profile,agc,aec in (('general',False,False),('vcp',False,False),('vcp',True,False),('vcp',True,True)):
            prefix=out/f'{profile}-{rate}-agc{int(agc)}-aec{int(aec)}'
            command=[str(args.cli),'--profile',profile,'--mic',str(path),'--ns-backend',args.backend,'--ns-level','1','--no-low-cut','--no-limiter','--diagnostic-stages','--out',str(prefix)]
            command+=['--echo',str(path)] if aec else ['--no-aec']
            if not agc:command+=['--no-agc']
            report=run(command,prefix)
            after,ar=sf.read(prefix.with_suffix('.after.wav'));assert ar==rate and np.isfinite(after).all()
            assert report['ns_backend']==args.backend and report['ns_internal_sample_rate']==rate
            assert not report['external_vad_enabled']
            assert report['aec_internal_sample_rate']==(rate if aec else 0)
            assert report['ns_delay_samples']==rate//100
            maximum=None
            if profile=='general' and not agc:
                before,_=sf.read(direct.with_suffix('.after.wav'))
                assert before.shape==after.shape
                maximum=float(np.max(np.abs(before-after)));assert maximum==0
            results.append(dict(rate=rate,profile=profile,agc=agc,aec=aec,
                                declared_delay_samples=report['dsp_delay_samples'],ns_rate=report['ns_internal_sample_rate'],
                                direct_max_difference=maximum,p99_process_ms=report.get('p99_process_ms')))
    report=dict(state='completed',backend=args.backend,cases=len(results),results=results,
                cli_sha256=hashlib.sha256(args.cli.read_bytes()).hexdigest(),
                probe_sha256=hashlib.sha256(args.probe.read_bytes()).hexdigest(),standard_compliance=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
