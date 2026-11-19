"""Evidence checks and matched output comparisons for isolated VCP trials."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import wave
import numpy as np


def require(value: bool, message: str) -> None:
    if not value:raise RuntimeError(message)


def audio(p: Path) -> np.ndarray:
    with wave.open(str(p)) as w:
        require(w.getframerate()==16000 and w.getsampwidth()==2,'WAV format differs')
        return np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').reshape(-1,w.getnchannels())


def summarize(path: Path, reference: Path|None=None, exact: bool=False) -> dict:
    path=path.resolve();command=json.loads((path/'COMMAND.json').read_text());out=path/'collected';d=json.loads((out/'out.json').read_text());seconds=command['seconds'];n=seconds*16000;input_calls=n//256
    require(int((out/'exitcode').read_text())==0,'Process failed')
    require(d['sample_rate']==16000 and d['frame_size']==256 and d['input_samples_per_channel']==n,'Input geometry changed')
    require(d['aec_model_sha256']=='b6e43138588a83bfe903ab5e143b4020b91c1e1629f5a575ac5855ff0003c731','Model changed')
    require(d['aec_provider_sha256']==command['variant']['library_sha256'],'Wrong candidate library executed')
    require(d['ns_frame_model_sha256']=='f754898f6721c3180f53b602183db1275c09e08355b8ddff2b0b289181a7a2e0','CoFi changed')
    require(d['ns_external_fallback_frames']==d['aec_localvqe_fallback_frames']==0,'Fallback occurred')
    require(d['aec_localvqe_inferences']==input_calls+2 and d['ns_inference_calls']==input_calls+3,'Inference count incorrect')
    require(d['dsp_delay_samples']==513 and d['stream_resets']==0,'PCM delay/reset changed')
    rows=list(csv.DictReader((out/'out.timing.csv').open()));frames=list(csv.DictReader((out/'out.frames.csv').open()))
    require(len(rows)==len(frames)==input_calls+3,'Incomplete callback log')
    require(all(int(row['frame'])==i for i,row in enumerate(frames)),'Frame sequence incomplete')
    metrics={}
    for key in ['process_ms','localvqe_process_ms','ns_stream_roundtrip_ms','localvqe_post_ms']:
        x=np.array([float(row[key]) for row in rows[:input_calls]])
        require(np.isfinite(x).all(),'Nonfinite time')
        metrics[key]={'mean':float(x.mean()),'p50':float(np.percentile(x,50)),'p95':float(np.percentile(x,95)),'p99':float(np.percentile(x,99)),'max':float(x.max())}
    over=sum(float(x['process_ms'])>16 for x in rows[:input_calls]);miss=sum(x['completion_late']=='1' for x in frames[:input_calls]) if command['paced'] else None
    wavs={};comparisons={};passed=True
    for name in ['after','aec','aec-input','bf']:
        p=out/('out.'+name+'.wav');x=audio(p);wavs[name]={'samples':len(x),'channels':x.shape[1],'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        require(np.isfinite(x).all(),'Nonfinite PCM')
        if name=='after':require(x.shape==(n+513,1),'Output/EOS sample count incorrect')
        if reference:
            q=reference/'collected'/p.name;y=audio(q);require(x.shape==y.shape,'PCM shape mismatch')
            diff=x.astype(np.int32)-y.astype(np.int32);maximum=int(np.max(np.abs(diff)));first=np.argwhere(diff!=0)
            item={'max_lsb':maximum,'rmse_lsb':float(np.sqrt(np.mean(diff.astype(np.float64)**2))),'bit_exact':not np.any(diff),'first_different_sample':int(first[0,0]) if len(first) else None,'passed':maximum==0 if exact else maximum<=1}
            passed=passed and item['passed'];comparisons[name]=item
    dispatch=None
    if (out/'sgemm-dispatch.csv').is_file():
        raw=(out/'sgemm-dispatch.csv').read_text();dispatch={'path':str(out/'sgemm-dispatch.csv'),'raw':raw}
    return {'status':'valid','variant':command['variant']['id'],'scene':command['scene'],'seconds':seconds,'paced':command['paced'],'directory':str(path),'expected_library_sha256':command['variant']['library_sha256'],'input_callbacks':input_calls,'total_callbacks':len(rows),'timings':metrics,'process_over16ms':over,'completion_misses':miss,'window_deadline_pass':over==0 and miss==0 if command['paced'] else None,'process_rtf':float(d['process_rtf']),'audio':wavs,'output_comparisons':comparisons,'pcm_comparison_passed':passed if reference else None,'reference_directory':str(reference) if reference else None,'fine_profiling':False,'dispatch':dispatch}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);parser.add_argument('--reference',type=Path);parser.add_argument('--exact',action='store_true');parser.add_argument('--out',type=Path,required=True);a=parser.parse_args();r=summarize(a.run,a.reference,a.exact)
    with a.out.open('x') as f:json.dump(r,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in r.items() if k not in ['dispatch','audio','output_comparisons']},ensure_ascii=False,indent=2))
