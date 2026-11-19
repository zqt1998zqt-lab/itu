"""Run bounded, meaningful component controls; all evidence is retained."""
from __future__ import annotations
from datetime import datetime
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def main() -> None:
    out=ROOT/'build/localvqe-fp32-opt'/('verification-'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S%f'));out.mkdir(parents=True)
    events=[]
    def run(label,argv):
        p=subprocess.run(argv,capture_output=True,text=True)
        record={'label':label,'argv':argv,'exitcode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
        with (out/(label+'.json')).open('x') as f:json.dump(record,f,indent=2)
        events.append(record)
        if p.returncode:raise RuntimeError(label+' failed; '+str(out))
        return p
    run('layout',[sys.executable,'-B',str(HERE/'layout/verify_layout.py')])
    exe=out/'gru-reference'
    run('compile-gru',['/usr/bin/clang++-17','-std=c++17','-O3','-ffp-contract=off','-fno-fast-math','-march=armv8-a+simd','-I'+str(HERE/'daf'),str(HERE/'tests/verify_gru.cpp'),'-o',str(exe)])
    p=run('gru-reference',[str(exe)])
    if 'RESULT PASS' not in p.stdout or 'wrongstate expected_failure detected=1' not in p.stdout or 'shift expected_failure detected=1' not in p.stdout:raise RuntimeError('GRU controls incomplete')
    g=ROOT/'build/deps/ggml-c044a8eeae2591faa0950c8b5e514cbc4bbfc4ca';stub=out/'dispatch-stub'
    run('compile-dispatch',['cc','-std=gnu11','-O0','-DGGML_USE_LLAMAFILE','-I'+str(HERE/'sgemm-root-v2'),'-I'+str(g/'src'),'-I'+str(g/'src/ggml-cpu'),'-I'+str(g/'include'),str(HERE/'tests/sgemm_stub.c'),'-o',str(stub)])
    for mode in ['success','decline','zero','off','invalid_nth','overflow','existing']:
        path=out/(mode+'.csv')
        if mode=='existing':
            with path.open('xb') as f:f.write(b'existing-evidence\n')
        run('dispatch-'+mode,[str(stub),mode,str(path)])
        if mode in ['zero','off']:
            if path.exists():raise RuntimeError('Empty/disabled dispatch created false evidence')
        elif mode=='existing':
            if path.read_bytes()!=b'existing-evidence\n':raise RuntimeError('Existing evidence overwritten')
        else:
            rows=list(csv.DictReader(path.open()));summary=next(r for r in rows if r['row_type']=='summary')
            if None in summary or any(v is None for v in summary.values()):raise RuntimeError('Dispatch CSV columns disagree')
            want={'success':(2,2,0,0,0),'decline':(2,0,2,0,0),'invalid_nth':(1,1,0,1,0),'overflow':(257,257,0,0,1)}[mode]
            actual=tuple(int(summary[k]) for k in ['attempt','success','decline','invalid_nth','unrecorded_attempts'])
            if actual!=want:raise RuntimeError('Dispatch count mismatch')
    spec=importlib.util.spec_from_file_location('lvq_comparator',HERE/'validation-root/compare.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    meta='name,dtype,count,byte_offset\nfloats,f32,3,0\ninteger,i64,1,12\n';row=np.array([1.,2.,3.],dtype='<f4').tobytes()+np.array([7],dtype='<i8').tobytes();raw=row*2;pcm=np.arange(512,dtype='<f4')*.0001
    def fixture(name,state=raw,description=meta,signal=pcm):
        prefix=out/name
        for suffix,data in [('.states.bin',state),('.meta.csv',description.encode()),('.pcm.f32',signal.tobytes())]:
            with Path(str(prefix)+suffix).open('xb') as f:f.write(data)
        return prefix
    baseline=fixture('baseline');same=fixture('same');wrong=bytearray(raw);wrong[:4]=np.array([1.25],dtype='<f4').tobytes();nan=pcm.copy();nan[3]=np.nan
    controls=[('identical',same,True),('wrong-state',fixture('wrong',bytes(wrong)),False),('truncated',fixture('short',raw[:-1]),False),('wrong-meta',fixture('meta',description=meta.replace('floats','other')),False),('nan',fixture('nan',signal=nan),False),('shifted',fixture('shift',signal=np.roll(pcm,1)),False)]
    results=[]
    for name,path,expected in controls:
        try:r=m.compare(baseline,path,True);passed=r['passed']
        except ValueError as e:r={'rejected':str(e)};passed=False
        if passed!=expected:raise RuntimeError('Comparator control failed: '+name)
        results.append({'name':name,'expected':expected,'accepted':passed,'result':r})
    with (out/'RESULT.json').open('x') as f:json.dump({'status':'passed','component_commands':len(events),'state_comparator_controls':results,'layout':'independent direct convolution oracle and dtype/batch/layout negative controls','gru':'128 steps/random/extreme/bin129/partition128 bit exact and wrong-state/shift rejected','dispatch':'7 positive/negative/lifecycle/overflow/overwrite controls'},f,indent=2)
    print(json.dumps({'status':'passed','evidence':str(out/'RESULT.json'),'component_commands':len(events),'comparator_controls':len(results)},ensure_ascii=False))


if __name__=='__main__':main()
