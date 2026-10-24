#!/usr/bin/env python3
"""Verify the real AudioEngine CLI integration and exact fallback on native PCM."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import soundfile as sf
from train_hybrid_ns import save,sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cli',type=Path,required=True);p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--models',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--expected-hidden-size',type=int,choices=(32,64))
    p.add_argument('--expected-contract',type=int,choices=(1,2,3))
    args=p.parse_args()
    root=args.root.resolve();root.mkdir(parents=True,exist_ok=False)
    data=json.loads(args.manifest.read_text());checks=[]
    for rate in (16000,24000,48000):
        scene=next(r for r in data['rows'] if r['rate']==rate and r['split']=='validation')
        source=Path(scene['origin'])/'before.wav';model=(args.models/f'student-{rate}.bin').resolve()
        results={}
        for name,extra in [('baseline',[]),('student',['--hybrid-ns-model',str(model)]),
                           ('missing',['--hybrid-ns-model',str(root/'missing.bin')])]:
            prefix=root/f'{rate}-{name}'
            command=[str(args.cli.resolve()),'--profile','general','--mic',str(source),'--out',str(prefix),
                     '--no-aec','--no-agc','--no-limiter','--no-low-cut','--no-external-vad','--ns-backend','omlsa_imcra','--ns-level','2']+extra
            result=subprocess.run(command,capture_output=True,text=True)
            save(prefix.with_suffix('.invocation.json'),dict(argv=command,code=result.returncode,stdout=result.stdout,stderr=result.stderr))
            result.check_returncode()
            report=json.loads(prefix.with_suffix('.json').read_text());audio,sr=sf.read(prefix.with_suffix('.after.wav'))
            if sr!=rate or not np.isfinite(audio).all():raise ValueError('invalid engine audio')
            results[name]=(report,audio)
        baseline=results['baseline'][1];student=results['student'][0];missing=results['missing'][0]
        if not np.array_equal(baseline,results['missing'][1]):raise ValueError('engine fallback changed PCM')
        if student['hybrid_ns_state']!=1 or student['hybrid_ns_inferences']==0 or student['hybrid_ns_fallback_frames']!=0:
            raise ValueError('engine ignored student model')
        if missing['hybrid_ns_state']!=2 or missing['hybrid_ns_fallback_frames']==0:
            raise ValueError('missing model fallback not visible')
        prefix=root/f'{rate}-core'
        subprocess.run([str(args.probe.resolve()),'--mic',str(source),'--out',str(prefix),'--model',str(model)],check=True,capture_output=True)
        core,sr=sf.read(prefix.with_suffix('.after.wav'))
        core_report=json.loads(prefix.with_suffix('.json').read_text())
        if args.expected_hidden_size and core_report['student_hidden_size']!=args.expected_hidden_size:
            raise ValueError('unexpected model width actually loaded')
        if args.expected_contract and core_report['student_contract']!=args.expected_contract:
            raise ValueError('unexpected model architecture actually loaded')
        if not np.array_equal(core,results['student'][1]):raise ValueError('engine vs isolated core PCM differs')
        checks.append(dict(rate=rate,core_engine_pcm_bit_equal=True,missing_model_pcm_bit_equal=True,
             student_state=1,fallback_state=2,input_sha256=sha(source),model_sha256=sha(model)))
    # Never silently ignore a model under another backend or resample its rate.
    for label,flags in [('wrong_backend',['--ns-backend','webrtc']),('aec3_native24',['--ns-backend','omlsa_imcra'])]:
        r=next(x for x in data['rows'] if x['rate']==24000)
        cmd=[str(args.cli.resolve()),'--mic',str(Path(r['origin'])/'before.wav'),'--out',str(root/label),
             '--hybrid-ns-model',str((args.models/'student-24000.bin').resolve())]+flags
        result=subprocess.run(cmd,capture_output=True,text=True)
        if result.returncode==0:raise ValueError('incompatible model/rate silently accepted')
        save(root/(label+'.json'),dict(returncode=result.returncode,stderr=result.stderr))
    save(root/'report.json',dict(passed=True,checks=checks,cli_sha256=sha(args.cli),
         scope='AudioEngine integration with HPF/AEC/AGC disabled for exact isolated-core equality',c_api_model_loader_implemented=False))
    print('native AudioEngine integration/fallback checks passed')

if __name__=='__main__':main()
