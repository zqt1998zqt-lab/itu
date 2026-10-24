#!/usr/bin/env python3
"""Add real native clean utterances to development/validation, without holdout."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from train_hybrid_ns import save,sha

def task(args):
    root,probe,row=args
    clean=Path(row['origin'])/'clean.wav'
    if sha(clean)!=row['clean_sha256']:raise ValueError('clean identity drift')
    token=hashlib.sha256((row['run']+'/'+row['id']+'-clean').encode()).hexdigest()[:20]
    prefix=root/'features'/token;raw=prefix.with_suffix('.f32')
    subprocess.run([str(probe),'--mic',str(clean),'--clean',str(clean),'--features',str(raw),'--out',str(prefix)],
                   check=True,capture_output=True)
    z=np.fromfile(raw,dtype='<f4').reshape(-1,448)
    dest=prefix.with_suffix('.npz')
    np.savez_compressed(dest,x=z[:,:128],p_stat=z[:,128:192],xi=z[:,192:256],gamma=z[:,256:320],cp=z[:,320:384],np=z[:,384:448])
    for ext in ('.f32','.after.wav','.csv','.json'):prefix.with_suffix(ext).unlink()
    return dict(row,id=row['id']+'-clean',features=str(dest),features_sha256=sha(dest),
                input_path=str(clean),input_sha256=sha(clean),noise='clean_augmentation',snr_db=None)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--probe',type=Path,required=True);p.add_argument('--jobs',type=int,default=4)
    args=p.parse_args();root=args.root.resolve();root.mkdir(parents=True,exist_ok=False);(root/'features').mkdir()
    manifest=json.loads(args.manifest.read_text());unique={}
    for row in manifest['rows']:
        if row['split'] not in ('development','validation'):raise ValueError('holdout in augmentation manifest')
        key=row['rate'],row['split'],row['clean_sha256'];unique.setdefault(key,row)
    with ProcessPoolExecutor(args.jobs) as pool:
        added=list(pool.map(task,[(root,args.probe.resolve(),r) for r in unique.values()]))
    manifest['rows']+=added
    manifest['augmentation']=dict(kind='real native clean inputs',added=len(added),source_manifest_sha256=sha(args.manifest),
         resampled=False,holdout_used=False)
    save(root/'manifest.json',manifest);print('added clean examples',len(added))

if __name__=='__main__':main()
