#!/usr/bin/env python3
"""C++ 10ms PCM vs independently assembled ONNX/NumPy causal oracle.

Also changes the future suffix: prefix output must remain bit-identical. This
checks real state continuity and PCM dependency, rather than finite output only.
"""
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
import onnxruntime as ort
import soundfile as sf
import run_audio_eval as ev


def reference(x,model,reset_each=False):
    options=ort.SessionOptions(); options.intra_op_num_threads=options.inter_op_num_threads=1
    session=ort.InferenceSession(str(model),options,providers=["CPUExecutionProvider"])
    states=[np.zeros((1,n),np.float32) for n in (5358,402,1056)]
    window=.5-.5*np.cos(2*np.pi*np.arange(512)/512)
    # Exactly mirror the CLI's four 160-sample zero drain callbacks (512 delay).
    padded=np.pad(x,(256,1024)); overlap=np.zeros(len(x)+1536); norm=np.zeros_like(overlap)
    for start in range(0,len(x)+640,256):
        spec=np.fft.rfft(padded[start:start+512]*window)
        mix=np.stack((spec.real,spec.imag),axis=-1)[None,:,None,:].astype(np.float32)
        if reset_each: states=[np.zeros_like(s) for s in states]
        out,*states=session.run(None,dict(zip(("mix","conv_cache","tfa_cache","inter_cache"),(mix,*states))))
        enhanced=np.fft.irfft(out[0,:,0,0]+1j*out[0,:,0,1],n=512)
        overlap[start:start+512]+=enhanced*window; norm[start:start+512]+=window**2
    return (overlap/np.maximum(norm,1e-9))[256:256+len(x)]


def main():
    p=argparse.ArgumentParser(); p.add_argument("--cli",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True); p.add_argument("--root",type=Path,required=True)
    p.add_argument("--input",type=Path,required=True); a=p.parse_args(); a.root.mkdir(parents=True,exist_ok=False)
    x,rate=sf.read(a.input); assert rate==16000
    if x.ndim>1: x=x[:,0]
    x=x[:16000*3]; sf.write(a.root/"input.wav",x,rate,subtype="PCM_16")
    def run(name,pcm):
        sf.write(a.root/(name+".wav"),pcm,rate,subtype="PCM_16")
        prefix=a.root/name
        command=[str(a.cli.resolve()),"--mic",str(a.root/(name+".wav")),"--out",str(prefix),
            "--no-aec","--no-agc","--no-low-cut","--ns-backend","ulunas","--neural-model",str(a.model.resolve())]
        subprocess.run(command,check=True,capture_output=True)
        info=json.loads(prefix.with_suffix(".json").read_text()); y,_=sf.read(prefix.with_suffix(".after.wav"))
        assert info["dsp_delay_samples"]==512 and info["neural_inferences"]>0
        return y[512:512+len(pcm)],info
    actual,info=run("actual",x); expected=reference(x,a.model); bad=reference(x,a.model,True)
    mutated=x.copy(); split=16000; mutated[split:]*=-1
    future,_=run("future",mutated)
    error=np.abs(actual-expected); control=float(np.sqrt(np.mean((expected-bad)**2)))
    raw,_=sf.read(a.root/"actual.after.wav"); raw_future,_=sf.read(a.root/"future.after.wav")
    result=dict(cli_sha256=ev.digest(a.cli),model_sha256=ev.digest(a.model),delay_samples=512,
        max_abs_error=float(error.max()),rms_error=float(np.sqrt(np.mean(error**2))),
        pcm16_tolerance=3/32768,state_reset_negative_control_rms=control,
        # Causality is checked on the emitted time axis, before delay removal.
        future_prefix_bit_identical=bool(np.array_equal(raw[:split],raw_future[:split])),engine=info)
    (a.root/"report.json").write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))
    assert result["max_abs_error"]<3/32768 and control>1e-4 and result["future_prefix_bit_identical"]


if __name__=="__main__": main()
