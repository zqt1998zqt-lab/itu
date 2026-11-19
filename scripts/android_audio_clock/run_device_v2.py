"""Prepare/run v2 phone AAudio trials with writable Perfetto trace output.

--plan-only needs no device and saves exact model/CLI identities and commands.
No APK, microphone permission, governor, priority, or system settings are changed.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time
from zoneinfo import ZoneInfo

ROOT=Path('/home/zhangqingtian/project/itu')
HERE=Path(__file__).resolve().parent
DEVICE='/data/local/tmp/itu-native16-20261117'


def digest(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()


def main()->None:
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--serial',default='cb8145db');p.add_argument('--clock',choices=['timer','aaudio'],default='aaudio');p.add_argument('--hint',choices=['off','on'],default='off');p.add_argument('--sharing',choices=['exclusive','shared'],default='exclusive');p.add_argument('--seconds',type=int,choices=[60,600],default=60);p.add_argument('--trace',action='store_true');p.add_argument('--plan-only',action='store_true');a=p.parse_args()
    if a.clock=='timer' and a.hint=='on':raise ValueError('ADPF must bind a real audio callback TID')
    mem=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
    if shutil.disk_usage(ROOT).free<100*1024**3 or mem<5*1024**3:raise RuntimeError('Host resource stop line; no cleanup')
    m=json.loads(a.manifest.read_text());android=m['targets']['android'];binary=Path(android['binary']);library=Path(android['D_library'])
    if digest(binary)!=android['binary_sha256'] or digest(library)!=android['D_library_sha256']:raise RuntimeError('Build identity changed')
    stamp=datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%dT%H%M%S%f');out=ROOT/'build/android-audio-clock-20261117'/('device-'+a.clock+'-'+a.hint+'-'+stamp);out.mkdir(parents=True)
    remote='/data/local/tmp/itu-audio-clock-'+stamp
    inputs=json.loads((ROOT/'build/android-native16-20261117/device/inputs/INPUT-MANIFEST.json').read_text())['probe' if a.seconds==60 else 'long']
    expected={DEVICE+'/models/localvqe.gguf':m['model_sha256']['localvqe'],DEVICE+'/models/cofi.onnx':m['model_sha256']['cofi'],DEVICE+'/models/cofi-window.f32':m['model_sha256']['window'],DEVICE+f'/inputs/double-talk-frozen-loop-{a.seconds}s.mic.wav':inputs['microphone']['sha256'],DEVICE+f'/inputs/double-talk-frozen-loop-{a.seconds}s.echo.wav':inputs['echo']['sha256']}
    command=['env','LD_LIBRARY_PATH='+DEVICE+'/lib','ITU_LOCALVQE_LIBRARY='+remote+'/liblocalvqe.so.0.1.0','ITU_LOCALVQE_MODEL='+DEVICE+'/models/localvqe.gguf',remote+'/itu_audio_clock','--profile','vcp','--no-rx-processing','--aec-backend','localvqe_aec200k','--ns-backend','webrtc','--eq-preamp-db','-2','--eq-band','low_shelf,180,-2,0.707','--eq-band','peak,1800,2,1','--eq-band','high_shelf,4500,-1,0.707','--tx-drc','--localvqe-reference-channel','0','--beam-delays','1.0,0.0','--native16-cofi-model',DEVICE+'/models/cofi.onnx','--native16-cofi-window',DEVICE+'/models/cofi-window.f32','--mic',DEVICE+f'/inputs/double-talk-frozen-loop-{a.seconds}s.mic.wav','--echo',DEVICE+f'/inputs/double-talk-frozen-loop-{a.seconds}s.echo.wav','--profile-stages','--out',remote+'/out']
    if a.clock=='timer':command+=['--pace-16ms']
    else:command+=['--aaudio-clock']
    if a.hint=='on':command+=['--performance-hint']
    if a.sharing=='shared':command+=['--audio-clock-shared']
    # Deliberately no taskset: allow AAudio to select its callback thread placement.
    # A controlled affinity comparison, if desired, is a separately declared trial.
    plan={'schema':'itu-aaudio-device-trial-v1','serial':a.serial,'manifest':str(a.manifest.resolve()),'mode':a.clock,'hint_requested':a.hint,'sharing_requested':a.sharing,'seconds':a.seconds,'remote':remote,'local':str(out),'argv':command,'expected_existing_phone_assets':expected,'new_assets':{'binary':{'path':str(binary),'sha256':digest(binary)},'library':{'path':str(library),'sha256':digest(library)}},'silent_clock_stream':a.clock=='aaudio','frozen_file_input':True,'microphone_capture':False,'processed_audio_playback':False,'affinity_policy':'OS-managed; no taskset/governor/priority changes','hardware_validation':'pending until this script executes on a connected phone','trace_requested':a.trace}
    with (out/'PLAN.json').open('x') as f:json.dump(plan,f,ensure_ascii=False,indent=2)
    if a.plan_only:print(json.dumps({'status':'plan_only_no_device_actions','plan':str(out/'PLAN.json'),'command':shlex.join(command)},ensure_ascii=False));return
    lock=(ROOT/'build/android-audio-clock-20261117/phone-exclusive.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);index=0
    def adb(label,args,timeout=30,required=True):
        nonlocal index;index+=1;r=subprocess.run(['adb','-s',a.serial,*args],capture_output=True,text=True,timeout=timeout)
        with (out/(f'{index:03d}-'+label+'.json')).open('x') as f:json.dump({'argv':args,'exitcode':r.returncode,'stdout':r.stdout,'stderr':r.stderr},f,indent=2)
        if required and r.returncode:raise RuntimeError('ADB '+label+' failed; no cleanup or silent retry')
        return r
    pre=adb('preflight',['shell','getprop ro.soc.model\ndf -k /data/local/tmp\nhead -n 3 /proc/meminfo\nps -A -o PID,NAME | awk \'index($2,"itu_audio")==1 || $2=="state-probe" {print}\'']).stdout
    if pre.splitlines()[0]!='SM8650':raise RuntimeError('Unexpected phone identity')
    if any('itu_audio' in l or 'state-probe' in l for l in pre.splitlines()):raise RuntimeError('Another audio test is active; preserve exclusive resource')
    disk=int(next(l.split()[3] for l in pre.splitlines() if l.startswith('/dev/')))*1024;phone_mem=int(next(l.split()[1] for l in pre.splitlines() if l.startswith('MemAvailable:')))*1024
    if disk<100*1024**3 or phone_mem<1024**3:raise RuntimeError('Phone capacity insufficient; retain all existing files')
    identity=adb('existing-asset-identities',['shell','sha256sum '+shlex.join(list(expected))]).stdout
    actual={l.split(maxsplit=1)[1].strip():l.split(maxsplit=1)[0] for l in identity.splitlines() if len(l.split(maxsplit=1))==2}
    if any(actual.get(k)!=v for k,v in expected.items()):raise RuntimeError('Frozen model/input mismatch')
    adb('mkdir',['shell','mkdir '+shlex.quote(remote)])
    for label,path,destination in [('binary',binary,remote+'/itu_audio_clock'),('library',library,remote+'/liblocalvqe.so.0.1.0')]:
        adb('push-'+label,['push',str(path),destination]);r=adb('sha-'+label,['shell','sha256sum '+shlex.quote(destination)])
        if r.stdout.split()[0]!=digest(path):raise RuntimeError('Transferred asset mismatch')
    adb('chmod',['shell','chmod 700 '+shlex.quote(remote+'/itu_audio_clock')])
    trace_command=None
    remote_trace_config=None
    remote_trace=None
    if a.trace:
        # Permission and feature availability are tested by actual return status.
        trace_source=(HERE/'perfetto-scheduler.pbtxt').read_text().replace('duration_ms: 65000','duration_ms: '+str((a.seconds+5)*1000))
        with (out/'perfetto.pbtxt').open('x') as f:f.write(trace_source)
        remote_trace_config='/data/misc/perfetto-configs/itu-'+stamp+'.pbtxt'
        adb('push-trace-config',['push',str(out/'perfetto.pbtxt'),remote_trace_config])
        # The perfetto SELinux domain cannot read /data/local/tmp on this
        # Android 16 build. Use the platform config and trace spools.
        remote_trace='/data/misc/perfetto-traces/itu-'+stamp+'.pftrace'
        trace_command=shlex.join(['perfetto','--txt','-c',remote_trace_config,'-o',remote_trace])
    script=['#!/system/bin/sh','set -C','cd '+shlex.quote(remote),'date +%s%3N > started_ms']
    if trace_command:script += [trace_command+' > perfetto.stdout 2> perfetto.stderr &','trace_pid=$!']
    script += [shlex.join(command)+' > stdout.log 2> stderr.log &','runner=$!','printf "%s\\n" "$runner" > runner.pid','wait "$runner"','result=$?','printf "%s\\n" "$result" > algorithm_exitcode','date +%s%3N > algorithm_finished_ms']
    if trace_command:script += ['wait "$trace_pid"','printf "%s\\n" "$?" > perfetto_exitcode']
    script += ['printf "%s\\n" "$result" > exitcode','date +%s%3N > finished_ms']
    with (out/'runner.sh').open('x') as f:f.write('\n'.join(script)+'\n')
    adb('push-script',['push',str(out/'runner.sh'),remote+'/runner.sh']);adb('launch',['shell','nohup setsid sh '+shlex.quote(remote+'/runner.sh')+' > '+shlex.quote(remote+'/supervisor.log')+' 2>&1 < /dev/null &'])
    print(json.dumps({'status':'running','plan':str(out/'PLAN.json')},ensure_ascii=False),flush=True)
    start=time.monotonic();done=False
    while time.monotonic()-start<a.seconds+60:
        r=adb('poll',['shell','if test -f '+shlex.quote(remote+'/exitcode')+'; then cat '+shlex.quote(remote+'/exitcode')+'; else echo RUNNING; fi'],required=False)
        if not r.returncode and r.stdout.strip()!='RUNNING':done=True;break
        time.sleep(5)
    if not done:raise RuntimeError('No terminal marker; persistent process and evidence retained; do not start a parallel trial')
    adb('pull',['pull',remote,str(out/'collected')],timeout=120)
    if remote_trace:
        try:
            adb('pull-trace',['pull',remote_trace,str(out/'collected/scheduler.pftrace')],timeout=120)
        except Exception as error:
            with (out/'pull-trace-error.json').open('x') as f:
                json.dump({'error':str(error),'remote_trace':remote_trace},f,indent=2)
    code=int((out/'collected/exitcode').read_text());result={'status':'process_complete' if code==0 else 'process_failed','exitcode':code,'directory':str(out),'actual_hardware_acceptance':'requires analysis; process exit is not a deadline or audio-quality pass'}
    with (out/'RESULT.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))
    if code:raise RuntimeError('Device trial failed; evidence retained')


if __name__=='__main__':main()
