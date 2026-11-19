from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np

def compare(a:Path,b:Path,exact:bool)->dict:
    meta_a=list(csv.DictReader(Path(str(a)+'.meta.csv').open()));meta_b=list(csv.DictReader(Path(str(b)+'.meta.csv').open()))
    if meta_a!=meta_b:raise ValueError('State metadata differs')
    size=sum(int(r['count'])*(4 if r['dtype']=='f32' else 8) for r in meta_a)
    p=Path(str(a)+'.states.bin');q=Path(str(b)+'.states.bin')
    if p.stat().st_size!=q.stat().st_size or not size or p.stat().st_size%size:raise ValueError('State file size differs or truncated')
    frames=p.stat().st_size//size;x=np.memmap(p,dtype=np.uint8,mode='r',shape=(frames,size));y=np.memmap(q,dtype=np.uint8,mode='r',shape=(frames,size));fail=[];first=None;maxdiff=0.;exact_all=True
    for row in meta_a:
        count=int(row['count']);offset=int(row['byte_offset']);dtype='<f4' if row['dtype']=='f32' else '<i8';width=np.dtype(dtype).itemsize
        xx=np.ndarray((frames,count),dtype=dtype,buffer=x,offset=offset,strides=(size,width));yy=np.ndarray((frames,count),dtype=dtype,buffer=y,offset=offset,strides=(size,width))
        if not np.isfinite(xx).all() or not np.isfinite(yy).all():raise ValueError('Nonfinite state '+row['name'])
        bits=xx.tobytes()==yy.tobytes();exact_all=exact_all and bits
        delta=np.abs(xx.astype(np.float64)-yy.astype(np.float64));mx=float(delta.max(initial=0));maxdiff=max(maxdiff,mx)
        mismatch=xx.view('<u4')!=yy.view('<u4') if exact and row['dtype']=='f32' else xx!=yy if row['dtype']=='i64' else delta>(1e-5+1e-4*np.abs(xx))
        if np.any(mismatch):
            where=np.argwhere(mismatch)[0];item={'name':row['name'],'frame':int(where[0]),'index':int(where[1]),'maxabs':mx,'expected':float(xx[tuple(where)]),'actual':float(yy[tuple(where)])};fail.append(item)
            if first is None or item['frame']<first['frame']:first=item
    pcm_a=np.fromfile(str(a)+'.pcm.f32',dtype='<f4');pcm_b=np.fromfile(str(b)+'.pcm.f32',dtype='<f4')
    if pcm_a.shape!=pcm_b.shape or not len(pcm_a) or not np.isfinite(pcm_a).all() or not np.isfinite(pcm_b).all():raise ValueError('PCM geometry/nonfinite')
    pcmmax=float(np.max(np.abs(pcm_a.astype(float)-pcm_b)));pcmbits=pcm_a.tobytes()==pcm_b.tobytes();passpcm=pcmbits if exact else pcmmax<=1e-5
    return {'passed':not fail and passpcm,'state_frames':frames,'state_frame_bytes':size,'state_tensors':len(meta_a),'state_all_bit_exact':exact_all,'max_state_absdiff':maxdiff,'failing_tensors':fail,'first_state_failure':first,'pcm_bit_exact':pcmbits,'pcm_maxabs':pcmmax,'pcm_rmse':float(np.sqrt(np.mean((pcm_a.astype(float)-pcm_b)**2))),'pcm_samples':len(pcm_a),'pcm_passed':passpcm,'exact_required':exact,'state_atol':0 if exact else 1e-5,'state_rtol':0 if exact else 1e-4}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path);p.add_argument('--exact',action='store_true');p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=compare(a.baseline,a.candidate,a.exact)
    with a.out.open('x') as f:json.dump(r,f,indent=2)
    print(json.dumps(r,indent=2));raise SystemExit(0 if r['passed'] else 1)
