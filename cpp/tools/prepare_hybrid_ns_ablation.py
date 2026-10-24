#!/usr/bin/env python3
"""Re-extract enriched inputs from the exact old development/validation rows."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from train_hybrid_ns import save, sha


def task(args):
    root, probe, row = args
    source = Path(row.get('input_path', str(Path(row['origin'])/'before.wav')))
    clean = Path(row['origin'])/'clean.wav'
    if sha(source)!=row['input_sha256'] or sha(clean)!=row['clean_sha256'] or sha(row['features'])!=row['features_sha256']:
        raise ValueError('source or feature identity drift')
    token = hashlib.sha256((row['run']+'/'+row['id']).encode()).hexdigest()[:20]
    prefix = root/'features'/token;raw = prefix.with_suffix('.f32')
    subprocess.run([str(probe),'--mic',str(source),'--clean',str(clean),'--features',str(raw),
                    '--feature-version','2','--out',str(prefix)],check=True,capture_output=True)
    z = np.fromfile(raw,dtype='<f4').reshape(-1,704)
    if len(z)!=row['frames'] or not np.isfinite(z).all():
        raise ValueError('invalid enriched feature extraction')
    arrays = dict(x=z[:,:256],p_stat=z[:,256:320],xi=z[:,320:384],gamma=z[:,384:448],
                  cp=z[:,448:512],np=z[:,512:576],power=z[:,576:640],noise=z[:,640:704])
    with np.load(row['features']) as old:
        if not np.array_equal(arrays['x'][:,:128],old['x']) or any(not np.array_equal(arrays[k],old[k]) for k in ('p_stat','xi','gamma','cp','np')):
            raise ValueError('re-extraction changed historical features/labels')
    dest=prefix.with_suffix('.npz');np.savez_compressed(dest,**arrays)
    for ext in ('.f32','.after.wav','.csv','.json'):
        prefix.with_suffix(ext).unlink()
    return dict(row,features=str(dest),features_sha256=sha(dest),legacy_features=row['features'],legacy_features_bit_equal=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--probe',type=Path,required=True);p.add_argument('--jobs',type=int,default=4)
    args=p.parse_args();root=args.root.resolve();root.mkdir(parents=True,exist_ok=False);(root/'features').mkdir()
    manifest=json.loads(args.manifest.read_text())
    if not manifest['rows'] or any(r['split'] not in ('development','validation') for r in manifest['rows']):
        raise ValueError('empty selection or holdout prohibited')
    rows=[]
    with ProcessPoolExecutor(args.jobs) as pool:
        for row in pool.map(task,[(root,args.probe.resolve(),r) for r in manifest['rows']]):
            rows.append(row)
            if len(rows)%25==0:print('prepare',len(rows),'/',len(manifest['rows']),flush=True)
    save(root/'manifest.json',dict(manifest,rows=rows,feature_columns=704,feature_inputs=256,
         feature_channels=['log_power_noise_ratio/6','delta_log_power/6','log_power/12','log_noise/12'],
         source_manifest_sha256=sha(args.manifest),probe_sha256=sha(args.probe),
         preparation_sha256=sha(__file__),legacy_features_bit_equal=True,holdout_loaded=False))
    print('prepared',len(rows),flush=True)
