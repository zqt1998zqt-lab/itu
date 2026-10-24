#!/usr/bin/env python3
"""Exercise the actual AudioEngine CLI with frozen NS options and ADC-like PCM16."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
from verify_ns_vehicle_v4 import bindings,call,save,R,P

def main():
 freeze=bindings();out=R/'engine-cli-equivalence-final';out.mkdir(exist_ok=False)
 binary=R/'engine-build/cpp/itu_audio_cli';rows=[json.loads(l) for l in (R/'final-vehicle/results.jsonl').open()];results=[]
 for c in freeze['variants']:
  for rate in (16000,24000,48000):
   r=next(x for x in rows if x['variant']==c['name'] and x['scene']['sample_rate']==rate and x['scene']['split']=='development')
   origin=P/'build'/r['original_run']/'cases'/r['scene']['id']/'before.wav';x,sr=sf.read(origin)
   folder=out/c['name']/str(rate);folder.mkdir(parents=True);source=folder/'input.wav'
   sf.write(source,np.clip(np.rint(x*32768),-32768,32767)/32768,rate,subtype='PCM_16')
   reference,ar,em=call(c,source,folder/'component')
   prefix=folder/'engine'
   argv=[str(binary),'--profile','general','--no-aec','--no-agc','--no-limiter','--no-low-cut','--vehicle-ns-control','--mic',str(source),'--out',str(prefix),'--ns-backend',c['backend'],'--vehicle-ns-mild-strength',str(c['mild_strength']),'--vehicle-ns-repair-mode',str(c['repair_mode']),'--vehicle-rumble-hz',str(c['rumble_hz']),'--vehicle-ns-policy',c['policy'],'--vehicle-ns-dry-max',str(c['dry_max']),'--vehicle-ns-snr-threshold',str(c['snr_threshold_db']),'--vehicle-ns-noise-gain',str(c['noise_gain']),'--vehicle-ns-startup-ms',str(c['startup_ms']),'--vehicle-ns-slew-ms',str(c['slew_ms']),'--vehicle-ns-release-ms',str(c['release_ms']),'--vehicle-ns-quiet-floor-db',str(c['quiet_floor_db']),'--vehicle-ns-floor-guard',str(c['floor_guard'])]
   argv += ['--speex-suppression-db' if c['backend']=='speex' else '--ns-level',str(c['strength'])]
   if c['rumble_kind']=='adaptive':argv.append('--vehicle-adaptive-rumble')
   p=subprocess.run(argv,capture_output=True,text=True);save(folder/'invocation.json',dict(argv=argv,exit_code=p.returncode,stderr=p.stderr));p.check_returncode()
   info=json.loads(prefix.with_suffix('.json').read_text());audio,actual_rate=sf.read(prefix.with_suffix('.after.wav'));delay=info['dsp_delay_samples'];audio=audio[delay:]
   if actual_rate!=rate or audio.shape!=reference.shape or delay!=em['delay_samples']:raise RuntimeError('integrated CLI shape/timebase mismatch')
   difference=audio-reference;maximum=float(np.max(abs(difference)));rms=float(np.sqrt(np.mean(difference*difference)))
   passed=maximum==0
   results.append(dict(variant=c['name'],rate=rate,delay_samples=delay,max_abs_difference=maximum,rms_difference=rms,bitexact=maximum==0,passed=passed,comparison='same native-rate PCM16 entry conversion once before any SRC'))
   if not passed:raise RuntimeError('engine/component PCM mismatch '+repr(results[-1]))
 bindings();save(out/'report.json',dict(state='completed',rows=results,engine_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),all_passed=all(x['passed'] for x in results),standard_compliance=False));print('engine CLI equivalence passed',len(results))

if __name__=='__main__':main()
