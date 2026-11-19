"""Build the frozen quality-preserving D chain with opt-in AAudio pacing.

Android links the exact tested D archive/library. A separate host D build only
checks callback extraction and numerical invariants; its timing is not a phone
measurement. Every invocation creates new output paths and retains failures.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT=Path('/home/zhangqingtian/project/itu')
HERE=Path(__file__).resolve().parent
ANDROID=ROOT/'build/android-native16-20261117/root'
ABASE=ANDROID/'localvqe-build-v4'
HBASE=ROOT/'build/localvqe-host-20261010'
DROOT=ROOT/'build/localvqe-opt-20261117/build-20261117T173046192276'
HENGINE=ROOT/'build/native16-20261117/compile-gz6ek6n1'
ORT_HOST=ROOT/'build/deps/onnxruntime-linux-aarch64-1.23.2'


def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()


def resources()->None:
    mem=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    if shutil.disk_usage(ROOT).free<100*1024**3 or mem<5*1024**3:raise RuntimeError('Resource stop line; no cleanup allowed')


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument('--target',choices=['all','host','android'],default='all');args=parser.parse_args();resources()
    root=ROOT/'build/android-audio-clock-20261117'/('build-'+datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%dT%H%M%S%f'));root.mkdir(parents=True)
    operations=[]
    def run(label,argv,cwd):
        resources()
        with (root/(label+'.log')).open('xb') as log:p=subprocess.run(argv,cwd=cwd,stdout=log,stderr=subprocess.STDOUT)
        record={'label':label,'argv':argv,'cwd':str(cwd),'exitcode':p.returncode};operations.append(record)
        with (root/(label+'.json')).open('x') as f:json.dump(record,f,indent=2)
        if p.returncode:raise RuntimeError(label+' failed; evidence '+str(root/(label+'.log')))
    generated=root/'generated/itu_audio_cli.cpp'
    run('generate-callback-cli',[sys.executable,'-B',str(HERE/'prepare_cli_v3.py'),'--out',str(generated)],ROOT)

    def fixed_link(path:Path,cwd:Path,output:Path,replacements:dict)->list[str]:
        command=shlex.split(path.read_text().strip());out_index=command.index('-o')+1
        for i,token in enumerate(command):
            if i==out_index:command[i]=str(output)
            elif token in replacements:command[i]=str(replacements[token])
            elif i and not token.startswith('-') and token.endswith(('.a','.o','.so')):command[i]=str((cwd/token).resolve())
        return command

    def archive(label,cwd,path,replacements,extras=[]):
        command=shlex.split(path.read_text().splitlines()[0]);out=root/(label+'.a');files=[str(replacements.get(Path(x).name,(cwd/x).resolve())) for x in command[3:]]
        run('archive-'+label,[command[0],'qc',str(out),*files,*map(str,extras)],root);run('index-'+label,['/usr/bin/ranlib',str(out)],root);return out

    def make_flags(path):
        flags={}
        for line in path.read_text().splitlines():
            if line.startswith(('CXX_DEFINES =','CXX_INCLUDES =','CXX_FLAGS =')):
                key,value=line.split('=',1);flags[key.strip()]=shlex.split(value)
        return flags

    results={}
    if args.target in ['all','android']:
        target=root/'android';target.mkdir();provider=DROOT/'D/overlay';library=DROOT/'D/liblocalvqe.so.0.1.0';manifest=json.loads((DROOT/'MANIFEST.json').read_text())['variants']['D']
        if sha(library)!=manifest['library_sha256']:raise RuntimeError('Tested Android D provider changed')
        flags=make_flags(ANDROID/'itu-build-v2/cpp/CMakeFiles/itu_audio_cli.dir/flags.make')
        obj=target/'clock_cli.o'
        command=['/usr/bin/clang++-17','--target=aarch64-linux-android26',*flags['CXX_DEFINES'],'-I'+str(HERE),'-I'+str(provider),'-I'+str(ROOT/'build/deps/speexdsp-1.2.1/include'),'-I'+str(ANDROID/'itu-build-v2/speex-generated/speex'),*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],'-c',str(generated),'-o',str(obj)]
        run('compile-android-clock-cli',command,ROOT)
        link_file=ANDROID/'itu-build-v2/cpp/CMakeFiles/itu_audio_cli.dir/link.txt';original=next(x for x in shlex.split(link_file.read_text()) if x.endswith('/itu_audio_cli.cpp.o'));binary=target/'itu_audio_clock'
        command=fixed_link(link_file,ANDROID/'itu-build-v2/cpp',binary,{original:obj,'libitu_audio.a':DROOT/'vcp-D.a'})+['-laaudio','-landroid']
        run('link-android-clock-cli',command,ROOT)
        results['android']={'binary':str(binary),'binary_sha256':sha(binary),'D_library':str(library),'D_library_sha256':sha(library),'uses_exact_tested_D_archive':True,'hardware_execution':'pending_phone_offline','mock_clock_compiled':False,'models_changed':False}

    if args.target in ['all','host']:
        target=root/'host';target.mkdir();commands=json.loads((HBASE/'compile_commands.json').read_text())
        def compile_part(name,source,template,extras=[]):
            matches=[x for x in commands if Path(x['file']).name==template]
            if len(matches)!=1:raise RuntimeError('Host compile record not unique')
            item=matches[0];argv=shlex.split(item['command']);out=target/(name+'.o');argv[argv.index('-c')+1]=str(source);argv[argv.index('-o')+1]=str(out);argv[1:1]=extras
            run('host-'+name,argv,Path(item['directory']));return out
        with ThreadPoolExecutor(max_workers=2) as pool:
            f1=pool.submit(compile_part,'layout',ROOT/'scripts/localvqe_fp32_opt/layout/localvqe_graph.cpp','localvqe_graph.cpp')
            f2=pool.submit(compile_part,'sgemm-cpu',ROOT/'scripts/localvqe_fp32_opt/sgemm-root-v2/ggml-cpu.c','ggml-cpu.c',['-DGGML_USE_LLAMAFILE','-I'+str(ROOT/'scripts/localvqe_fp32_opt/sgemm-root-v2')]);layout=f1.result();cpu=f2.result()
        vec=next(x for x in commands if x['file'].endswith('/vec.cpp'));argv=shlex.split(vec['command']);sgemm_source=Path(vec['file']).parent/'llamafile/sgemm.cpp';sgemm=target/'sgemm-kernel.o';argv[argv.index('-c')+1]=str(sgemm_source);argv[argv.index('-o')+1]=str(sgemm);argv[1:1]=['-DGGML_USE_LLAMAFILE'];run('host-sgemm-kernel',argv,Path(vec['directory']))
        graph=archive('host-d-graph',HBASE,HBASE/'CMakeFiles/dvqe_graph.dir/link.txt',{'localvqe_graph.cpp.o':layout})
        cpu_archive=archive('host-d-cpu',HBASE/'vendor/ggml/src',HBASE/'vendor/ggml/src/CMakeFiles/ggml-cpu.dir/link.txt',{'ggml-cpu.c.o':cpu},[sgemm])
        library=target/'liblocalvqe.so.0.1.0';command=fixed_link(HBASE/'CMakeFiles/localvqe_shared.dir/link.txt',HBASE,library,{'libdvqe_graph.a':graph,'vendor/ggml/src/libggml-cpu.a':cpu_archive});run('host-link-localvqe',command,HBASE)
        overlay=target/'overlay';overlay.mkdir();provider=(ANDROID/'overlay/localvqe_aec.hpp').read_text();old_hash='182c64ecee04e611537f8e6d5ea20f706526cb90ddb33bce33b58f8efddb3de9'
        if provider.count(old_hash)!=1:raise RuntimeError('Provider binding not unique')
        with (overlay/'localvqe_aec.hpp').open('x') as f:f.write(provider.replace(old_hash,sha(library),1))
        with (overlay/'audio_engine.cpp').open('xb') as f:f.write((ANDROID/'overlay/audio_engine.cpp').read_bytes())
        with (overlay/'classic_cli.cpp').open('xb') as f:f.write((ANDROID/'overlay/itu_audio_cli.cpp').read_bytes())
        host_recipe=json.loads((HENGINE/'manifest.json').read_text());engine_template=host_recipe['commands'][0]
        def compile_cli_part(label,path,extras=[]):
            argv=list(engine_template);out=target/(label+'.o');argv[argv.index('-c')+1]=str(path);argv[argv.index('-o')+1]=str(out);argv[1:1]=['-I'+str(overlay),'-I'+str(HERE),'-I'+str(ANDROID/'overlay'),'-I'+str(ORT_HOST/'include'),*extras];run('host-'+label,argv,ROOT);return out
        engine=compile_cli_part('engine',overlay/'audio_engine.cpp')
        classic_obj=compile_cli_part('classic-cli',overlay/'classic_cli.cpp')
        new_obj=compile_cli_part('clock-cli',generated,['-DITU_AUDIO_CLOCK_TESTING=1'])
        outputs={}
        for label,cli_obj in [('reference',classic_obj),('callback',new_obj)]:
            command=list(host_recipe['commands'][-1]);command[command.index('-o')+1]=str(target/('itu_audio_'+label))
            for i,x in enumerate(command):
                if x==str(HENGINE/'audio_engine.o'):command[i]=str(engine)
                elif x==str(HENGINE/'itu_audio_cli.o'):command[i]=str(cli_obj)
            command += [str(ORT_HOST/'lib/libonnxruntime.so'),'-Wl,-rpath,'+str(ORT_HOST/'lib'),'-pthread']
            run('host-link-'+label,command,ROOT);outputs[label]=str(target/('itu_audio_'+label))
        results['host']={'binaries':outputs,'binaries_sha256':{k:sha(Path(v)) for k,v in outputs.items()},'D_library':str(library),'D_library_sha256':sha(library),'ORT_version':'1.23.2 host; Android uses1.26.0','scope':'same-host callback extraction/PCM invariance only; never Android timing evidence','mock_clock_compiled':True}

    manifest={'status':'built_offline','created_at':datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(),'build_directory':str(root),'targets':results,'model_paths':{'localvqe':str(ROOT/'build/deps/localvqe-models/localvqe-v1.4-aec-200K-f32.gguf'),'cofi':str(ROOT/'build/cofilite-ort-20261117/cofilite-frame-v1-opset18.onnx'),'window':str(ANDROID/'cofi-window.f32')},'model_sha256':{'localvqe':'b6e43138588a83bfe903ab5e143b4020b91c1e1629f5a575ac5855ff0003c731','cofi':'f754898f6721c3180f53b602183db1275c09e08355b8ddff2b0b289181a7a2e0','window':'9c16ead03d25f18137be55c43d5540e86aaaf15cee31434d7494c9912d9403c0'},'source_files':{str(p):sha(p) for p in [*(HERE.glob('*.hpp')),HERE/'prepare_cli_v3.py',generated]},'phone_tests_performed':False,'commands':operations}
    with (root/'MANIFEST.json').open('x') as f:json.dump(manifest,f,ensure_ascii=False,indent=2)
    print(json.dumps({'status':'built_offline','manifest':str(root/'MANIFEST.json'),'targets':results},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
