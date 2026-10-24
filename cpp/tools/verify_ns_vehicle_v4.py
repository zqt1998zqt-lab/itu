#!/usr/bin/env python3
"""Frozen NS component reproduction, state continuity and sequential paced checks."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[name]='1'
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
import repair_ns_vehicle_v4 as evaluation
P=Path(__file__).resolve().parents[2]
R=P/'build/ns-recorded-repair-v4-20261018'
B=R/'artifacts/final-r5/ns_vehicle_probe'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def bindings():
 x=json.loads((R/'selection-freeze.json').read_text())
 expected=dict(x['source_sha256'])
 overlay_path=R/'engine-integration-freeze-v2.json'
 if overlay_path.exists():
  overlay=json.loads(overlay_path.read_text())
  if overlay['selection_sha256']!=sha(R/'selection-freeze.json'):raise RuntimeError('integration overlay belongs to another selection')
  if set(overlay['changes'])!={'cpp/src/audio_engine.cpp','cpp/src/native_pipeline.cpp','cpp/src/vehicle_ns.cpp','cpp/include/itu/vehicle_ns.hpp'}:raise RuntimeError('unexpected integration patch scope')
  for path,change in overlay['changes'].items():
   if path in expected and expected[path]!=change['before_sha256']:raise RuntimeError('integration baseline differs')
   expected[path]=change['after_sha256']
 changed=[p for p,h in expected.items() if sha(P/p)!=h]
 if changed: raise RuntimeError('frozen identity changed: '+repr(changed))
 artifact=json.loads((B.parent/'identity.json').read_text())
 if sha(B)!=artifact['binary_sha256']:raise RuntimeError('processor binary differs from build identity')
 for path,digest in artifact['source_sha256'].items():
  if sha(P/path)!=digest or sha(B.parent/'source'/path)!=digest:raise RuntimeError('processor source snapshot differs: '+path)
 for path,digest in artifact['dependency_sha256'].items():
  if sha(Path(path))!=digest:raise RuntimeError('processor link input differs: '+path)
 if overlay_path.exists() and P/overlay['processor_binary']!=B:raise RuntimeError('integration overlay selects another binary')
 return x

def call(c,source,prefix,paced=False):
 prefix.parent.mkdir(parents=True,exist_ok=True)
 argv=[str(B),c['backend'],str(c['strength']),str(source),str(prefix),c.get('mode','shared'),str(c['dry_max']),str(c['snr_threshold_db']),str(c['noise_gain']),str(c['startup_ms']),c['policy'],str(c['mild_strength']),str(c['slew_ms']),str(c.get('release_ms',0)),str(c.get('quiet_floor_db',-300)),str(c.get('floor_guard',0)),str(c.get('repair_mode',0)),str(c.get('rumble_hz',0)),c.get('rumble_kind','iir')]
 if paced:argv.append('--pace-10ms')
 result=subprocess.run(argv,capture_output=True,text=True)
 save(prefix.with_suffix('.invocation.json'),dict(argv=argv,exit_code=result.returncode,stderr=result.stderr));result.check_returncode()
 info=json.loads(prefix.with_suffix('.json').read_text());x,sr=sf.read(prefix.with_suffix('.after.wav'))
 if not np.isfinite(x).all():raise RuntimeError('nonfinite stream output')
 return x[info['delay_samples']:],sr,info

def baseline(suffix=''):
 bindings();root=R/('baseline-reproduction'+suffix);root.mkdir(exist_ok=False)
 old=P/'build/ns-failure-repair-v3-20261018';rows=[json.loads(l) for l in (old/'final-vehicle/results.jsonl').open()];results=[]
 for name in ('speex_v3','webrtc_v2_startup'):
  for rate in (16000,24000,48000):
   row=next(r for r in rows if r['variant']==name and r['scene']['sample_rate']==rate)
   c=dict(row['config'],repair_mode=0,rumble_hz=0,rumble_kind='iir')
   source=P/'build'/row['original_run']/'cases'/row['scene']['id']/'before.wav'
   original=old/'final-vehicle'/name/row['original_run']/row['scene']['id']/'result.after.wav'
   if sha(source)!=row['input_sha256'] or sha(original)!=row['output_sha256']:raise RuntimeError('old replay identity differs')
   prefix=root/f'{name}-{rate}';call(c,source,prefix)
   h=sha(prefix.with_suffix('.after.wav'))
   results.append(dict(variant=name,rate=rate,source_sha256=sha(source),old_output_sha256=row['output_sha256'],new_output_sha256=h,bitexact=h==row['output_sha256']))
 bindings();save(root/'report.json',dict(rows=results,all_bitexact=all(r['bitexact'] for r in results)))
 if not all(r['bitexact'] for r in results):raise RuntimeError('baseline reproduction differs')
 print('baseline bitexact',len(results),flush=True)

def streams(suffix=''):
 frozen=bindings();root=R/('streams'+suffix);root.mkdir(exist_ok=False)
 previous=P/'build/ns-backend-comparison-20261018/streams';ident=json.loads((previous/'identity.json').read_text());rows=[]
 for meta in ident['sources']:
  rate=meta['rate'];source=previous/f'input-{rate}.wav';target=previous/f'clean-{rate}.wav'
  if sha(source)!=meta['input_sha256'] or sha(target)!=meta['clean_sha256']:raise RuntimeError('long stream identity differs')
  before,sr=sf.read(source);clean,cr=sf.read(target)
  if sr!=rate or cr!=rate or len(before)!=len(clean) or len(clean)!=120*rate:raise RuntimeError('stream shape differs')
  # Three 2-second native slices: SNR 30, -10, +10 dB. No rate conversion.
  paced_input=np.concatenate([before[t*rate:(t+2)*rate] for t in (0,24,48)])
  paced_source=root/f'paced-input-{rate}.wav';sf.write(paced_source,paced_input,rate,subtype='FLOAT')
  for c in frozen['variants']:
   modes=['shared']+(['native24'] if rate==24000 and c['backend']=='speex' and c['name'].endswith('balanced') else [])
   for mode in modes:
    cfg=dict(c,mode=mode);folder=root/c['name']/f'{rate}-{mode}';folder.mkdir(parents=True)
    after,ar,engine=call(cfg,source,folder/'continuous')
    if ar!=rate or len(after)!=len(clean):raise RuntimeError('long output shape differs')
    windows=[]
    for i,snr in enumerate((30,-10,10,30,-10)):
     sl=slice((i*24+2)*rate,(i*24+9)*rate)
     windows.append(dict(snr_db=snr,before=evaluation.quality(clean[sl],before[sl],rate),after=evaluation.quality(clean[sl],after[sl],rate)))
    reset_source=folder/'reset-input.wav';sf.write(reset_source,before[24*rate:48*rate],rate,subtype='FLOAT')
    reset,_,_=call(cfg,reset_source,folder/'reset-negative')
    different=int(np.count_nonzero(reset!=after[24*rate:48*rate]))
    offline,_,_=call(cfg,paced_source,folder/'offline-control')
    paced,_,pm=call(cfg,paced_source,folder/'paced',True)
    equal=bool(np.array_equal(offline,paced))
    row=dict(variant=c['name'],rate=rate,mode=mode,windows=windows,continuous=engine,paced=pm,paced_pcm_bitexact=equal,reset_negative_different_samples=different,input_sha256=sha(source),paced_input_sha256=sha(paced_source),output_sha256=sha(folder/'continuous.after.wav'))
    rows.append(row);save(root/'progress.json',dict(completed=len(rows),latest=dict(variant=c['name'],rate=rate,mode=mode)))
    print('stream',c['name'],rate,mode,'p99',pm['p99_process_ms'],'max',pm['max_process_ms'],'deadline_misses',pm['paced_deadline_misses'],flush=True)
    if not equal or not different:raise RuntimeError('pacing equality or reset negative control failed')
 bindings();save(root/'report.json',dict(state='completed',rows=rows,identity_verified=True,scope='sequential local aarch64 NS component; 120s single-state streams and 6s paced fixtures; not Android/end-to-end call RTD',standard_compliance=False))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('baseline','streams'));parser.add_argument('--suffix',default='');args=parser.parse_args()
 if args.suffix and (not args.suffix.startswith('-') or '/' in args.suffix or '\\' in args.suffix):parser.error('suffix must start with - and contain no directory separators')
 baseline(args.suffix) if args.phase=='baseline' else streams(args.suffix)
