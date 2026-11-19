"""Serialize persistent Android candidate replays without replacing old assets."""
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

SOURCE_DIR=Path(__file__).resolve().parent
HERE=SOURCE_DIR.parents[1]/'build/localvqe-fp32-opt'
HERE.mkdir(parents=True,exist_ok=True)
if not (HERE/'INPUTS.json').exists():
    with (HERE/'INPUTS.json').open('xb') as seed:seed.write((SOURCE_DIR/'inputs.json').read_bytes())
OLD=HERE.parent/'android-native16-20261117'
DEVICE='/data/local/tmp/itu-native16-20261117'
SERIAL='cb8145db'


def host_guard() -> None:
    mem=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
    if shutil.disk_usage(HERE).free<100*1024**3 or mem<5*1024**3:
        raise RuntimeError('Host resource stop line; all files retained')


def run_case(manifest: Path, variant: str, label: str, scene: str='double_talk', seconds: int=60, paced: bool=True, audit: bool=False) -> dict:
    host_guard()
    lock=(HERE/'phone-exclusive.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=json.loads(manifest.read_text());candidate=source['variants'][variant]
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S%f')
    local=HERE/'runs'/(label+'-'+variant+'-'+stamp);local.mkdir(parents=True)
    remote='/data/local/tmp/itu-localvqe-opt-20261117/'+local.name
    count=0

    def adb(name: str, command: list[str], timeout: int=45, require: bool=True) -> subprocess.CompletedProcess:
        nonlocal count
        count+=1
        result=subprocess.run(['adb','-s',SERIAL,*command],capture_output=True,text=True,timeout=timeout)
        with (local/(f'{count:03d}-'+name+'.json')).open('x') as f:
            json.dump({'argv':command,'exitcode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()},f,ensure_ascii=False,indent=2)
        if require and result.returncode:raise RuntimeError(f'ADB {name} failed; preserve {remote}')
        return result

    pre=adb('preflight',['shell','getprop ro.soc.model\ndf -k /data/local/tmp\ncat /proc/meminfo\ncat /proc/loadavg\nps -A -o PID,NAME | awk \'index($2,"itu_audio_cli")==1 || $2=="simpleperf" || $2=="state-probe" {print}\'']).stdout
    if pre.splitlines()[0]!='SM8650':raise RuntimeError('Wrong target phone')
    mem=int(next(x.split()[1] for x in pre.splitlines() if x.startswith('MemAvailable:')))*1024
    disk=int(next(x.split()[3] for x in pre.splitlines() if x.startswith('/dev/')))*1024
    if mem<1024**3 or disk<100*1024**3:raise RuntimeError('Phone capacity insufficient; no cleanup')
    if any('itu_audio_cli' in line or 'state-probe' in line or 'simpleperf' in line for line in pre.splitlines()):
        raise RuntimeError('Phone already executing a test; do not compete')
    adb('mkdir-parent',['shell','mkdir -p '+shlex.quote(str(Path(remote).parent))])
    adb('mkdir',['shell','mkdir '+shlex.quote(remote)])
    def push(name: str, path: Path, destination: str, expected: str|None=None) -> str:
        current=hashlib.sha256(path.read_bytes()).hexdigest()
        if expected and current!=expected:raise RuntimeError('Host artifact changed: '+str(path))
        adb('push-'+name,['push',str(path),destination])
        check=adb('sha-'+name,['shell','sha256sum '+shlex.quote(destination)]).stdout.split()[0]
        if check!=current:raise RuntimeError('Phone upload SHA mismatch')
        return destination
    binary=push('binary',Path(candidate['binary']),remote+'/itu_audio_cli',candidate['binary_sha256'])
    library=push('library',Path(candidate['library']),remote+'/liblocalvqe.so.0.1.0',candidate['library_sha256'])
    adb('chmod',['shell','chmod 700 '+shlex.quote(binary)])
    if seconds in [60,600] and scene=='double_talk':
        mic=DEVICE+f'/inputs/double-talk-frozen-loop-{seconds}s.mic.wav'
        echo=DEVICE+f'/inputs/double-talk-frozen-loop-{seconds}s.echo.wav'
    elif seconds==12:
        cases=json.loads((HERE/'INPUTS.json').read_text())['cases'];case=next(x for x in cases if x['condition']==scene)
        mic=push('mic',Path(case['files']['mic']['path']),remote+'/mic.wav',case['files']['mic']['sha256'])
        echo=push('echo',Path(case['files']['echo']['path']),remote+'/echo.wav',case['files']['echo']['sha256'])
    else:raise ValueError('Unsupported scene/duration: use five 12s scenes or frozen double-talk60/600')
    identity_paths=[DEVICE+'/models/localvqe.gguf',DEVICE+'/models/cofi.onnx',mic,echo]
    model_identity=adb('frozen-input-identity',['shell','sha256sum '+shlex.join(identity_paths)]).stdout
    actual={line.split(maxsplit=1)[1].strip():line.split(maxsplit=1)[0] for line in model_identity.splitlines() if len(line.split(maxsplit=1))==2}
    if seconds in [60,600] and scene=='double_talk':
        frozen=json.loads((OLD/'device/inputs/INPUT-MANIFEST.json').read_text())['probe' if seconds==60 else 'long']
        input_expected={mic:frozen['microphone']['sha256'],echo:frozen['echo']['sha256']}
    else:
        input_expected={mic:case['files']['mic']['sha256'],echo:case['files']['echo']['sha256']}
    expected={DEVICE+'/models/localvqe.gguf':'b6e43138588a83bfe903ab5e143b4020b91c1e1629f5a575ac5855ff0003c731',DEVICE+'/models/cofi.onnx':'f754898f6721c3180f53b602183db1275c09e08355b8ddff2b0b289181a7a2e0',**input_expected}
    if any(actual.get(path)!=digest for path,digest in expected.items()):raise RuntimeError('Per-path frozen model or input SHA mismatch')
    env=['env','LD_LIBRARY_PATH='+DEVICE+'/lib','ITU_LOCALVQE_LIBRARY='+library,'ITU_LOCALVQE_MODEL='+DEVICE+'/models/localvqe.gguf']
    if audit:env+=['ITU_LVQ_SGEMM_REPORT='+remote+'/sgemm-dispatch.csv']
    command=env+['taskset','80',binary,'--profile','vcp','--no-rx-processing','--aec-backend','localvqe_aec200k','--ns-backend','webrtc','--eq-preamp-db','-2','--eq-band','low_shelf,180,-2,0.707','--eq-band','peak,1800,2,1','--eq-band','high_shelf,4500,-1,0.707','--tx-drc','--localvqe-reference-channel','0','--mic',mic,'--echo',echo,'--beam-delays','1.0,0.0','--native16-cofi-model',DEVICE+'/models/cofi.onnx','--native16-cofi-window',DEVICE+'/models/cofi-window.f32','--profile-stages','--out',remote+'/out']
    if paced:command+=['--pace-16ms']
    shell='#!/system/bin/sh\nset -C\ncd '+shlex.quote(remote)+'\nprintf "%s\\n" "$$" > supervisor.pid\ndate +%s%3N > started_ms\n'+shlex.join(command)+' > stdout.log 2> stderr.log &\nrunner=$!\nprintf "%s\\n" "$runner" > runner.pid\nwait "$runner"\nstatus=$?\nprintf "%s\\n" "$status" > exitcode\ndate +%s%3N > finished_ms\n'
    with (local/'runner.sh').open('x') as f:f.write(shell)
    config={'serial':SERIAL,'remote':remote,'variant':candidate,'seconds':seconds,'scene':scene,'paced':paced,'dispatch_audit':audit,'argv':command,'manifest':str(manifest)}
    with (local/'COMMAND.json').open('x') as f:json.dump(config,f,ensure_ascii=False,indent=2)
    adb('push-runner',['push',str(local/'runner.sh'),remote+'/runner.sh'])
    adb('launch',['shell','nohup setsid sh '+shlex.quote(remote+'/runner.sh')+' > '+shlex.quote(remote+'/supervisor.log')+' 2>&1 < /dev/null &'])
    print(json.dumps({'state':'running','variant':variant,'seconds':seconds,'scene':scene,'local':str(local)},ensure_ascii=False),flush=True)
    started=time.monotonic();done=False;poll=0
    while time.monotonic()-started<seconds+180:
        host_guard();poll+=1
        probe=adb('poll',['shell','if test -f '+shlex.quote(remote+'/exitcode')+'; then cat '+shlex.quote(remote+'/exitcode')+'; else echo RUNNING; fi'],timeout=20,require=False)
        if probe.returncode==0 and probe.stdout.strip()!='RUNNING':done=True;break
        if poll%5==0 or poll==2:
            telemetry=adb('telemetry',['shell','pid=$(cat '+shlex.quote(remote+'/runner.pid')+'); cat /proc/"$pid"/status; cat /proc/loadavg; head -n 3 /proc/meminfo; if test ! -f '+shlex.quote(remote+'/maps')+'; then cat /proc/"$pid"/maps > '+shlex.quote(remote+'/maps')+'; fi'],require=False)
            if 'No space left' in telemetry.stderr or 'Out of memory' in telemetry.stderr:raise RuntimeError('Phone resource failure; preserved')
            print(json.dumps({'state':'running','variant':variant,'elapsed_s':round(time.monotonic()-started,1)},ensure_ascii=False),flush=True)
        time.sleep(2)
    if not done:raise RuntimeError('Phone job not terminal; keep persistent evidence and do not overlap a new test')
    adb('pull',['pull',remote,str(local/'collected')],timeout=120)
    exitcode=int((local/'collected/exitcode').read_text())
    result={'state':'process_complete' if exitcode==0 else 'process_failed','exitcode':exitcode,'local':str(local),'remote':remote,'variant':variant,'seconds':seconds,'scene':scene}
    with (local/'RESULT.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    print(json.dumps(result,ensure_ascii=False),flush=True)
    if exitcode:raise RuntimeError('Phone process failed; see '+str(local))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--variant',required=True);parser.add_argument('--label',required=True);parser.add_argument('--scene',default='double_talk');parser.add_argument('--seconds',type=int,default=60);parser.add_argument('--unpaced',action='store_true');parser.add_argument('--audit',action='store_true');a=parser.parse_args()
    run_case(a.manifest.resolve(),a.variant,a.label,a.scene,a.seconds,not a.unpaced,a.audit)
