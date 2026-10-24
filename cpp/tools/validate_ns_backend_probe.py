#!/usr/bin/env python3
"""Validate NS probe delay independently, retaining QMF phase diagnostics."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import soundfile as sf


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();probe=args.probe.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    rows=[];checks=[]
    for rate in (16000,24000,48000):
        x=np.zeros(rate*3);positions=[round(.7*rate),round(1.7*rate)];x[positions]=.5
        path=out/f'impulse-{rate}.wav';sf.write(path,x,rate,subtype='FLOAT')
        variants=[('webrtc0','webrtc',0,'shared'),('speex12','speex',-12,'shared'),
                  ('bypass','bypass',0,'shared'),('filterbank','filterbank',0,'shared')]
        if rate==24000:variants.append(('speexnative24','speex',-12,'native24'))
        measured={}
        for name,backend,strength,mode in variants:
            prefix=out/f'{name}-{rate}'
            command=[str(probe),backend,str(strength),str(path),str(prefix),mode]
            result=subprocess.run(command,capture_output=True,text=True);result.check_returncode()
            metadata=json.loads(prefix.with_suffix('.json').read_text());y,sr=sf.read(prefix.with_suffix('.after.wav'))
            peaks=[]
            for pos in positions:
                start=pos-round(.1*rate);end=pos+round(.1*rate)
                peaks.append(start+int(np.argmax(abs(y[start:end])))-pos)
            row=dict(rate=rate,backend=name,declared=metadata['delay_samples'],measured_peaks=peaks,
                     impulse_peak_minus_declared=[v-metadata['delay_samples'] for v in peaks],
                     nonzero=int(np.count_nonzero(y)),stderr=result.stderr)
            rows.append(row);measured[name]=row
            checks.append(dict(check=f'{rate}:{name}:finite_nonzero_length',passed=bool(
                sr==rate and len(y)==len(x)+metadata['delay_samples'] and np.isfinite(y).all() and np.max(abs(y))>.001)))
        # A QMF synthesis filter is not linear phase: its impulse maximum need
        # not coincide with the declared stream offset. Compare to an independent
        # split/merge-only probe plus the known 96-sample subband STFT offset.
        extra=round(96*rate/16000)
        checks.append(dict(check=f'{rate}:webrtc_vs_filterbank_plus_stft',passed=all(
            abs(a-b-extra)<=1 for a,b in zip(measured['webrtc0']['measured_peaks'],measured['filterbank']['measured_peaks']))))
        for name in ('speex12','bypass')+ (('speexnative24',) if rate==24000 else ()):
            checks.append(dict(check=f'{rate}:{name}:declared_impulse_delay',passed=all(
                abs(v-measured[name]['declared'])<=1 for v in measured[name]['measured_peaks'])))
    rate=16000;rng=np.random.default_rng(20261018);x=rng.normal(0,.03,rate*4)
    path=out/'stationary-noise.wav';sf.write(path,x,rate,subtype='FLOAT');attenuations={}
    for name,backend,strength in (('webrtc0','webrtc',0),('speex12','speex',-12),('bypass','bypass',0)):
        prefix=out/f'noise-{name}'
        result=subprocess.run([str(probe),backend,str(strength),str(path),str(prefix),'shared'],capture_output=True,text=True);result.check_returncode()
        m=json.loads(prefix.with_suffix('.json').read_text());y,_=sf.read(prefix.with_suffix('.after.wav'));y=y[m['delay_samples']:m['delay_samples']+len(x)]
        q=np.clip(np.rint(x*32768),-32768,32767)/32768
        attenuation=float(10*np.log10(np.mean(q[rate:]**2)/max(np.mean(y[rate:]**2),1e-20)))
        attenuations[name]=attenuation
        if name=='bypass':checks.append(dict(check='known_bad_bypass_fails_6db_noise_gate',passed=attenuation<6))
        else:checks.append(dict(check=f'{name}:actually_changes_noise',passed=attenuation>.5))
    invalid=subprocess.run([str(probe),'webrtc','9',str(path),str(out/'invalid'),'shared'],capture_output=True,text=True)
    checks.append(dict(check='invalid_strength_rejected',passed=invalid.returncode!=0))
    report=dict(passed=all(c['passed'] for c in checks),checks=checks,impulse_diagnostics=rows,
        noise_attenuation_db=attenuations,alignment='all quality measurements retain declared fixed delay; no per-case lag search')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
