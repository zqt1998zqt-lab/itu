#!/usr/bin/env python3
"""Serial native-rate complete NS wall time, with baseline and optional student."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from train_hybrid_ns import save,sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--configs',type=Path,required=True)
    p.add_argument('--repeat',type=int,default=3)
    args=p.parse_args();root=args.root.resolve();root.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(args.manifest.read_text());configs=json.loads(args.configs.read_text());rows=[]
    for rate in (16000,24000,48000):
        # Metadata selection only; never choose the fastest audio after timing.
        scene=sorted((r for r in manifest['rows'] if r['rate']==rate and r['split']=='validation'),key=lambda r:r['id'])[0]
        source=Path(scene['origin'])/'before.wav'
        for config in configs:
            series=[]
            for run in range(args.repeat):
                out=root/f"{config['name']}-{rate}-{run}"
                command=[str(args.probe.resolve()),'--mic',str(source),'--out',str(out),'--mode','ecnr']
                model=config.get('models',{}).get(str(rate))
                if model:command+=['--model',model,'--spp-mix',str(config['spp_mix']),'--snr-mix',str(config['snr_mix'])]
                subprocess.run(command,check=True,capture_output=True)
                report=json.loads(out.with_suffix('.json').read_text());series.append(report)
                out.with_suffix('.after.wav').unlink();out.with_suffix('.csv').unlink()
            rows.append(dict(rate=rate,variant=config['name'],source_id=scene['id'],input_sha256=sha(source),
                median_p50_ms=float(np.median([r['p50_process_ms'] for r in series])),
                worst_p99_ms=max(r['p99_process_ms'] for r in series),worst_max_ms=max(r['max_process_ms'] for r in series),
                over_10ms=sum(r['over_10ms'] for r in series),frames=sum(r['frames'] for r in series),
                model_state=series[-1]['student_state'],native_core_rate=series[-1]['core_rate'],
                teacher_in_runtime=False,process_budget_ms=10,
                passed=all(r['over_10ms']==0 for r in series)))
    save(root/'report.json',dict(checks=rows,passed=all(r['passed'] for r in rows),serial=True,
         scope='complete OM-LSA + features + GRU + synthesis, not whole AEC/AGC engine',
         model_loading_excluded=True,host_arch='aarch64',probe_sha256=sha(args.probe)))
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
