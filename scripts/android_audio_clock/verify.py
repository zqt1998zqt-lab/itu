"""Offline acceptance: fake platform contracts, real builds and full DSP parity."""
from __future__ import annotations
import argparse
import ast
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import csv
from zoneinfo import ZoneInfo

ROOT=Path('/home/zhangqingtian/project/itu')
HERE=Path(__file__).resolve().parent


def require(value,message):
    if not value:raise RuntimeError(message)


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,required=True);a=parser.parse_args();m=json.loads(a.manifest.read_text());host=m['targets']['host']
    out=ROOT/'build/android-audio-clock-20261117'/('validation-'+datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%dT%H%M%S%f'));out.mkdir()
    commands=[]
    def run(label,argv,env=None,success=True):
        p=subprocess.run(argv,cwd=ROOT,capture_output=True,text=True,env=env)
        record={'label':label,'argv':argv,'exitcode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
        with (out/(label+'.json')).open('x') as f:json.dump(record,f,ensure_ascii=False,indent=2)
        commands.append(record);require((p.returncode==0)==success,label+' unexpected exit, evidence '+str(out));return p
    for p in HERE.rglob('*.py'):ast.parse(p.read_text())
    fake=out/'aaudio-contract'
    run('compile-fake-aaudio',['/usr/bin/clang++-17','-std=c++17','-O2','-Wall','-Wextra','-Werror','-pthread','-DITU_AUDIO_CLOCK_FAKE_ANDROID',str(HERE/'tests/aaudio_clock_contract.cpp'),'-o',str(fake)])
    test=run('fake-aaudio',[str(fake)]);reports={}
    for line in test.stdout.splitlines():
        if line.startswith('JSON '):_,name,data=line.split(' ',2);reports[name]=json.loads(data)
    require(len(reports)==5 and 'RESULT PASS cases=5' in test.stdout,'Fake platform scenarios incomplete')
    require(reports['rate_reject']['actual_sample_rate']==44100 and not reports['rate_reject']['completed'],'Wrong-rate acceptance')
    require(reports['fallback_silence_endstop']['sharing_fallback'] and reports['fallback_silence_endstop']['completed'],'Requested/actual modes not distinct')
    run('adpf-component-verification',[sys.executable,'-B',str(HERE/'tests/verify_performance_hint.py')])
    elf=run('android-elf',['/usr/bin/llvm-readelf-17','-h','-l','-d',m['targets']['android']['binary']])
    require('AArch64' in elf.stdout and '/system/bin/linker64' in elf.stdout and 'libaaudio.so' in elf.stdout and 'libandroid.so' in elf.stdout,'Not a linked Android AAudio executable')
    sources=json.loads((ROOT/'build/localvqe-opt-20261117/INPUTS.json').read_text())['cases'];cases=[]
    env=dict(os.environ,ITU_LOCALVQE_LIBRARY=host['D_library'],ITU_LOCALVQE_MODEL=m['model_paths']['localvqe'])
    require(hashlib.sha256(Path(host['D_library']).read_bytes()).hexdigest()==host['D_library_sha256'],'Host provider drift')
    for scene in sources:
        name=scene['condition'];paths={};reports_out={}
        for variant,binary,extras in [('classic',host['binaries']['reference'],[]),('extracted',host['binaries']['callback'],[]),('mock',host['binaries']['callback'],['--audio-clock-test'])]:
            directory=out/(name+'-'+variant);directory.mkdir();prefix=directory/'out';paths[variant]=prefix
            argv=[binary,'--profile','vcp','--no-rx-processing','--aec-backend','localvqe_aec200k','--ns-backend','webrtc','--eq-preamp-db','-2','--eq-band','low_shelf,180,-2,0.707','--eq-band','peak,1800,2,1','--eq-band','high_shelf,4500,-1,0.707','--tx-drc','--localvqe-reference-channel','0','--beam-delays','1.0,0.0','--native16-cofi-model',m['model_paths']['cofi'],'--native16-cofi-window',m['model_paths']['window'],'--mic',scene['files']['mic']['path'],'--echo',scene['files']['echo']['path'],'--profile-stages','--out',str(prefix),*extras]
            run(name+'-'+variant,argv,env);report=json.loads(Path(str(prefix)+'.json').read_text());reports_out[variant]=report
            require(report['aec_provider_sha256']==host['D_library_sha256'] and report['aec_model_sha256']==m['model_sha256']['localvqe'],'Wrong provider/model in host replay')
            require(report['aec_localvqe_inferences']==752 and report['ns_inference_calls']==753 and report['aec_localvqe_fallback_frames']==report['ns_external_fallback_frames']==0,'Inference/EOS/fallback differs')
            require(report['dsp_delay_samples']==513 and report['input_samples_per_channel']==192000,'PCM timeline changed')
        wavs={}
        for suffix in ['after.wav','aec.wav','aec-input.wav','bf.wav']:
            payloads=[Path(str(paths[v])+'.'+suffix).read_bytes() for v in paths]
            require(all(p==payloads[0] for p in payloads),name+' audio changed: '+suffix)
            wavs[suffix]=hashlib.sha256(payloads[0]).hexdigest()
        for v in ['extracted','mock']:
            sidecar=json.loads(Path(str(paths[v])+'.audio-clock.json').read_text());trace=list(csv.DictReader(Path(str(paths[v])+'.audio-clock.frames.csv').open()))
            require(sidecar['callbacks']==753 and sidecar['clock_errors']==0 and len(trace)==753,'Callback trace incomplete/invalid')
            require(sidecar['mock_clock']==(v=='mock') and not sidecar['hardware_audio_clock_executed'],'Offline path falsely claimed hardware execution')
            require(not sidecar['microphone_capture_verified'] and not sidecar['processed_audio_playback_verified'],'End-to-end claim unsupported')
            require(all(row['clocks_valid']=='1' and int(row['frame'])==i for i,row in enumerate(trace)),'Trace clock/order invalid')
        cases.append({'scene':name,'four_wavs_bit_exact_across_three_paths':True,'wav_sha256':wavs,'input_samples':192000,'AEC_inferences':752,'CoFi_inferences':753,'output_samples':192513,'DSP_delay_samples':513,'fallback':0})
        print(json.dumps({'scene':name,'four_wavs_bit_exact':True,'callback_and_EOS_trace':'passed'},ensure_ascii=False),flush=True)
    # Unsupported host platform must not silently execute a timer instead.
    scene=sources[0];directory=out/'unsupported-host';directory.mkdir();prefix=directory/'out'
    argv=[host['binaries']['callback'],'--profile','vcp','--no-rx-processing','--aec-backend','localvqe_aec200k','--ns-backend','webrtc','--native16-cofi-model',m['model_paths']['cofi'],'--native16-cofi-window',m['model_paths']['window'],'--mic',scene['files']['mic']['path'],'--echo',scene['files']['echo']['path'],'--beam-delays','1.0,0.0','--out',str(prefix),'--aaudio-clock']
    run('host-aaudio-unavailable',argv,env,False)
    failed=json.loads(Path(str(prefix)+'.audio-clock.json').read_text());require(not failed['hardware_audio_clock_executed'] and not failed['audio_stream']['completed'] and not Path(str(prefix)+'.json').exists(),'Unavailable clock became false success')
    result={'status':'passed_offline','manifest':str(a.manifest),'source_files_AST':'passed','fake_aaudio_scenarios':5,'adpf_fake_scenarios':13,'host_cases':cases,'four_wav_comparisons':5*4*2,'host_unavailable_not_faked':True,'android_binary_linked':True,'phone_connected_or_executed':False,'actual_device_audio_configuration':'not_measured','latency_or_scheduling_improvement':'not_measured','end_to_end_capture_playback':'not_verified','hardware_performance_hint':'not_measured','evidence_directory':str(out),'commands':commands}
    with (out/'RESULT.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ['commands','host_cases']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
