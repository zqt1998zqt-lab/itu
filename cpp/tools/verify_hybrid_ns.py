#!/usr/bin/env python3
"""Cross-runtime GRU numerical/causal verification; random fixtures are not models."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from train_hybrid_ns import Student, sha, save
from train_hybrid_ns_ablation import ExtendedStudent, load_student, export

def verify(args):
    root=args.root.resolve();root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(773)
    rng=np.random.default_rng(773)
    results=[]
    for rate in (16000,24000,48000):
        checkpoint=args.models/f'student-{rate}.pt' if args.models else None
        state=torch.load(checkpoint,map_location='cpu',weights_only=True) if checkpoint else None
        hidden=state['gru.weight_hh_l0'].shape[1] if state else args.hidden_size
        net=(load_student(state) if checkpoint else
             Student(hidden) if args.architecture=='gru' else ExtendedStudent(hidden,args.architecture=='conv')).eval()
        inputs=net.gru.input_size
        x=rng.uniform(-2,2,(511,inputs)).astype('<f4');x.tofile(root/'input.f32')
        model=root/f'fixture-{rate}.bin'
        export(net,rate,model)
        with torch.inference_mode():
            p,d,_=net(torch.from_numpy(x)[None]);expected=torch.cat((p,d),dim=-1)[0].numpy()
        command=[str(args.probe.resolve()),'--gru-input',str(root/'input.f32'),
                 '--model',str(model),'--rate',str(rate),'--out',str(root/f'cpp-{rate}.f32')]
        subprocess.run(command,check=True,capture_output=True)
        actual=np.fromfile(root/f'cpp-{rate}.f32',dtype='<f4').reshape(-1,128)
        error=abs(expected-actual);at=np.unravel_index(np.argmax(error),error.shape)
        if not np.isfinite(actual).all() or error.max()>3e-6:
            raise ValueError('PyTorch/C++ GRU mismatch: '+str((rate,error.max(),at)))
        # Prefix length changed, but preceding hidden-state evolution may not.
        x[:173].tofile(root/'prefix.f32')
        cmd=command.copy();cmd[cmd.index('--gru-input')+1]=str(root/'prefix.f32');cmd[-1]=str(root/'prefix-out.f32')
        subprocess.run(cmd,check=True,capture_output=True)
        prefix=np.fromfile(root/'prefix-out.f32',dtype='<f4').reshape(-1,128)
        if not np.array_equal(prefix,actual[:173]):
            raise ValueError('C++ output depends on future input')
        results.append(dict(rate=rate,hidden_size=hidden,inputs=inputs,contract=getattr(net,'contract',1),maxabs=float(error.max()),meanabs=float(error.mean()),
             worst_frame=int(at[0]),worst_output=int(at[1]),prefix_bit_equal=True,
             deployment_bytes=model.stat().st_size,model_sha256=sha(model)))
    save(root/'report.json',dict(passed=True,checks=results,tolerance=3e-6,frames=511,
         fixture_only=args.models is None,trained_model_directory=str(args.models) if args.models else None,
         probe_sha256=sha(args.probe)))
    print(json.dumps(results,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--models',type=Path)
    p.add_argument('--hidden-size',type=int,choices=(32,64),default=32)
    p.add_argument('--architecture',choices=('gru','enriched','conv'),default='gru')
    verify(p.parse_args())
