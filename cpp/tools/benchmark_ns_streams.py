#!/usr/bin/env python3
"""Sequential, same-rate NS stream/performance checks after parallel screening."""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import soundfile as sf
import compare_ns_backends as paired


def run(command,prefix):
    prefix.parent.mkdir(parents=True,exist_ok=True)
    result=subprocess.run(command,capture_output=True,text=True)
    paired.save(prefix.parent/(prefix.name+'.invocation.json'),dict(argv=command,exit_code=result.returncode,
        stdout=result.stdout,stderr=result.stderr))
    result.check_returncode()
    return json.loads(prefix.with_suffix('.json').read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--cli',type=Path,required=True)
    p.add_argument('--probe',type=Path,required=True);p.add_argument('--paced-seconds',type=int,default=12)
    args=p.parse_args();root=args.root.resolve();root.mkdir(parents=True,exist_ok=False)
    artifacts=root/'artifacts';artifacts.mkdir()
    cli=artifacts/'itu_audio_cli';probe=artifacts/'ns_backend_probe'
    shutil.copy2(args.cli,cli);shutil.copy2(args.probe,probe);shutil.copy2(__file__,artifacts/Path(__file__).name)
    selected=paired.selected_cases('core');sources=[];results=[];started=time.monotonic()
    for rate in (16000,24000,48000):
        groups={}
        for original_run,row in selected:
            s=row['scene']
            if s['sample_rate']!=rate or s['family']!='ns' or s['mics']!=1 or s['snr_db'] not in (-10,10,30):continue
            key=(original_run,s['noise'],s['near_speaker'],s['split'])
            groups.setdefault(key,{})[s['snr_db']]=(original_run,row)
        choices=[(key,rows) for key,rows in groups.items() if len(rows)==3]
        choices.sort(key=lambda v:(v[0][1] in ('white','pink'),v[0]))
        if not choices:raise ValueError('missing complete same-rate SNR stream group')
        key,rows=choices[0];blocks=[];targets=[];block_sources=[]
        for snr in (30,-10,10,30,-10):
            original_run,row=rows[snr];s=row['scene'];folder=paired.replay.PROJECT/'build'/original_run/'cases'/s['id']
            for name in ('before.wav','clean.wav'):
                if paired.replay.ctl.ev.digest(folder/name)!=s['files'][name]['sha256']:raise ValueError('source differs')
            b,sr=sf.read(folder/'before.wav');c,cr=sf.read(folder/'clean.wav')
            if sr!=rate or cr!=rate or b.ndim!=1 or len(b)!=24*rate:raise ValueError('not native 24-second mono')
            blocks.append(b);targets.append(c);block_sources.append(dict(run=original_run,id=s['id'],snr=snr,
                input_sha256=s['files']['before.wav']['sha256']))
        source=root/f'input-{rate}.wav';target_path=root/f'clean-{rate}.wav'
        before=np.concatenate(blocks);clean=np.concatenate(targets)
        sf.write(source,before,rate,subtype='FLOAT');sf.write(target_path,clean,rate,subtype='FLOAT')
        paced_source=root/f'paced-input-{rate}.wav';sf.write(paced_source,before[:args.paced_seconds*rate],rate,subtype='FLOAT')
        source_meta=dict(rate=rate,seconds=len(before)/rate,sources=block_sources,conversion=False,
            input_sha256=paired.replay.ctl.ev.digest(source),clean_sha256=paired.replay.ctl.ev.digest(target_path))
        sources.append(source_meta)
        for name,backend,strength in (('webrtc0','webrtc',0),('speex12','speex',-12)):
            flags=['--ns-backend',backend]+(['--ns-level',str(strength)] if backend=='webrtc' else ['--speex-suppression-db',str(strength)])
            raw_prefix=root/name/f'raw-{rate}'
            raw=run([str(probe),backend,str(strength),str(source),str(raw_prefix),'shared'],raw_prefix)
            prefix=root/name/f'continuous-{rate}'
            command=[str(cli),'--profile','vcp','--mic',str(source),'--no-aec','--out',str(prefix)]+flags
            engine=run(command,prefix)
            out,sr=sf.read(prefix.with_suffix('.after.wav'));delay=engine['dsp_delay_samples'];after=out[delay:delay+len(before)]
            if sr!=rate or len(after)!=len(before) or not np.isfinite(after).all():raise ValueError('continuous output shape/finite differs')
            windows=[]
            for i,snr in enumerate((30,-10,10,30,-10)):
                sl=slice((i*24+2)*rate,(i*24+9)*rate)
                paired.replay.ctl.ev.RATE=rate
                windows.append(dict(block=i,snr=snr,before=paired.replay.ctl.ev.quality(clean[sl],before[sl]),
                    after=paired.replay.ctl.ev.quality(clean[sl],after[sl])))
            block_outputs=[]
            # These are real process/state resets, compared against one live
            # 120-second state, never used as a substitute for continuous data.
            for i,b in enumerate(blocks):
                block_path=root/f'block-{rate}-{i}.wav'
                if not block_path.exists():sf.write(block_path,b,rate,subtype='FLOAT')
                bp=root/name/f'reset-{rate}-{i}'
                em=run([str(cli),'--profile','vcp','--mic',str(block_path),'--no-aec','--out',str(bp)]+flags,bp)
                bo,_=sf.read(bp.with_suffix('.after.wav'));bd=em['dsp_delay_samples'];block_outputs.append(bo[bd:bd+len(b)])
            reset=np.concatenate(block_outputs)
            pp=root/name/f'paced-{rate}'
            paced=run([str(cli),'--profile','vcp','--mic',str(paced_source),'--no-aec','--out',str(pp),'--pace-10ms']+flags,pp)
            results.append(dict(variant=name,rate=rate,continuous_seconds=120,raw=raw,engine=engine,paced=paced,
                windows=windows,continuous_vs_reset_rms_error=float(np.sqrt(np.mean((after-reset)**2))),
                continuous_vs_reset_different_samples=int(np.count_nonzero(after!=reset))))
            paired.save(root/'progress.json',dict(completed=len(results),total=6,elapsed_seconds=time.monotonic()-started))
    dependencies={str(p):paired.replay.ctl.ev.digest(p) for p in
                  (Path(__file__),Path(paired.__file__),Path(paired.replay.ctl.ev.__file__))}
    paired.save(root/'identity.json',dict(dependencies=dependencies,cli_sha256=paired.replay.ctl.ev.digest(cli),
        probe_sha256=paired.replay.ctl.ev.digest(probe),sources=sources,
        performance_scope='sequential local aarch64 CPU; unrelated host processes retained; no Android claim'))
    paired.save(root/'report.json',dict(state='completed',results=results,seconds_per_continuous_stream=120,
        paced_seconds_per_stream=args.paced_seconds,sources=sources,standard_compliance=False,
        elapsed_seconds=time.monotonic()-started))


if __name__=='__main__':main()
