"""Build isolated LocalVQE candidates from frozen Android compilation records."""
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
from zoneinfo import ZoneInfo

ROOT = Path('/home/zhangqingtian/project/itu')
HERE = Path(__file__).resolve().parent
OLD = ROOT / 'build/android-native16-20261117/root'
BASE = OLD / 'localvqe-build-v4'
VCP = OLD / 'itu-build-v2/cpp'
SRC = ROOT / 'build/deps/localvqe-f53063c9eb2a85f96479867d1dd911dc3bf6319b/ggml'


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def guard() -> None:
    mem = next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    if shutil.disk_usage(HERE).free < 100*1024**3 or mem < 5*1024**3:
        raise RuntimeError('Resource stop line: preserve all files; no cleanup')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--variants', nargs='+', default=['A','B','C','D','E','F'])
    parser.add_argument('--graph-for-f', choices=['A','B','C','D'], default='D')
    parser.add_argument('--without-probe', action='store_true')
    args = parser.parse_args()
    guard()
    original = BASE / 'liblocalvqe.so.0.1.0'
    original_sha = sha(original)
    if original_sha != '182c64ecee04e611537f8e6d5ea20f706526cb90ddb33bce33b58f8efddb3de9':
        raise RuntimeError('Frozen library identity changed')
    output_root = ROOT / 'build/localvqe-fp32-opt'
    output_root.mkdir(parents=True, exist_ok=True)
    directory = output_root / ('build-' + datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S%f'))
    directory.mkdir()
    objects = directory / 'objects'
    objects.mkdir()
    commands = json.loads((BASE / 'compile_commands.json').read_text())
    receipts = []

    def execute(label: str, command: list[str], cwd: Path) -> None:
        guard()
        with (directory/(label+'.log')).open('xb') as log:
            p = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
        receipt = {'label':label,'argv':command,'cwd':str(cwd),'exitcode':p.returncode}
        receipts.append(receipt)
        with (directory/(label+'.json')).open('x') as handle:
            json.dump(receipt,handle,indent=2)
        if p.returncode:
            raise RuntimeError(f'{label} failed; preserved {directory/(label+".log")}')

    def compile_object(name: str, source: Path, filename: str, extra: list[str] = []) -> Path:
        matching = [c for c in commands if Path(c['file']).name==filename and (filename!='localvqe_api.cpp' or 'localvqe_shared.dir' in c['command'])]
        if len(matching)!=1 or not source.is_file():
            raise RuntimeError(f'Candidate source/compile record not ready: {source}')
        entry=matching[0]
        command=shlex.split(entry['command'])
        output=objects/(name+'.o')
        command[command.index('-c')+1]=str(source)
        command[command.index('-o')+1]=str(output)
        command[1:1]=extra
        execute('compile-'+name,command,Path(entry['directory']))
        return output

    # Diagnostics ABI changes no existing frame API or mathematical operation.
    jobs=[] if args.without_probe else [('probe-api',HERE/'validation-root/probe_api.cpp','localvqe_api.cpp',[])]
    if not args.without_probe:jobs.append(('diagnostic-graph',HERE/'validation-root/original-lifetime-graph.cpp','localvqe_graph.cpp',[]))
    variants=set(args.variants)
    if variants & {'B','D'} or 'F' in variants and args.graph_for_f in ['B','D']:
        jobs.append(('layout',HERE/('layout/localvqe_graph.cpp' if args.without_probe else 'validation-root/layout-lifetime-graph.cpp'),'localvqe_graph.cpp',[]))
    if variants & {'C','D'} or 'F' in variants and args.graph_for_f in ['C','D']:
        jobs.append(('sgemm-cpu',HERE/'sgemm-root-v2/ggml-cpu.c','ggml-cpu.c',['-DGGML_USE_LLAMAFILE','-I'+str(HERE/'sgemm')]))
    if variants & {'E','F'}:
        jobs.append(('daf-batch',HERE/'daf-root-v2/daf_frontend.cpp','daf_frontend.cpp',['-I'+str(HERE/'daf')]))
    compiled={}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={j[0]:pool.submit(compile_object,*j) for j in jobs}
        for key,future in futures.items():
            compiled[key]=future.result()

    if 'sgemm-cpu' in compiled:
        e=next(c for c in commands if c['file'].endswith('/vec.cpp'))
        command=shlex.split(e['command']);llama=SRC/'vendor/ggml/src/ggml-cpu/llamafile/sgemm.cpp'
        command[command.index('-c')+1]=str(llama)
        compiled['sgemm-kernel']=objects/'sgemm-kernel.o'
        command[command.index('-o')+1]=str(compiled['sgemm-kernel'])
        command[1:1]=['-DGGML_USE_LLAMAFILE']
        execute('compile-sgemm-kernel',command,Path(e['directory']))

    def archive(name: str, cwd: Path, link_file: Path, substitutions: dict[str,Path], additions: list[Path]=[]) -> Path:
        command=shlex.split(link_file.read_text().splitlines()[0]);out=directory/(name+'.a')
        inputs=[str(substitutions.get(Path(x).name,(cwd/x).resolve())) for x in command[3:]]
        execute('archive-'+name,[command[0],'qc',str(out),*inputs,*map(str,additions)],directory)
        execute('index-'+name,['/usr/bin/llvm-ranlib-17',str(out)],directory)
        return out

    def link_command(record: Path, cwd: Path, output: Path, substitutions: dict[str,Path]) -> list[str]:
        command=shlex.split(record.read_text().strip());out_index=command.index('-o')+1
        for i,x in enumerate(command):
            if i==out_index:command[i]=str(output)
            elif x in substitutions:command[i]=str(substitutions[x])
            elif i and not x.startswith('-') and x.endswith(('.a','.o','.so')):command[i]=str((cwd/x).resolve())
        return command

    cpu_archive=BASE/'vendor/ggml/src/libggml-cpu.a'
    if 'sgemm-cpu' in compiled:
        cpu_archive=archive('cpu-sgemm',BASE/'vendor/ggml/src',BASE/'vendor/ggml/src/CMakeFiles/ggml-cpu.dir/link.txt',{'ggml-cpu.c.o':compiled['sgemm-cpu']},[compiled['sgemm-kernel']])
    matrix={}
    flags={}
    for line in (VCP/'CMakeFiles/itu_audio.dir/flags.make').read_text().splitlines():
        if line.startswith(('CXX_DEFINES =','CXX_INCLUDES =','CXX_FLAGS =')):
            key,value=line.split('=',1);flags[key.strip()]=shlex.split(value)
    for variant in args.variants:
        graph_id=args.graph_for_f if variant=='F' else variant
        has_layout=graph_id in ['B','D']
        has_sgemm=graph_id in ['C','D']
        has_daf=variant in ['E','F']
        sub={}
        if not args.without_probe:sub['localvqe_graph.cpp.o']=compiled['diagnostic-graph']
        if has_layout:sub['localvqe_graph.cpp.o']=compiled['layout']
        if has_daf:sub['daf_frontend.cpp.o']=compiled['daf-batch']
        graph=BASE/'libdvqe_graph.a'
        if sub:graph=archive('graph-'+variant,BASE,BASE/'CMakeFiles/dvqe_graph.dir/link.txt',sub)
        target=directory/variant;target.mkdir()
        library=target/'liblocalvqe.so.0.1.0'
        sub={'libdvqe_graph.a':graph}
        if not args.without_probe:sub['CMakeFiles/localvqe_shared.dir/localvqe_api.cpp.o']=compiled['probe-api']
        if has_sgemm:sub['vendor/ggml/src/libggml-cpu.a']=cpu_archive
        execute('link-library-'+variant,link_command(BASE/'CMakeFiles/localvqe_shared.dir/link.txt',BASE,library,sub),BASE)
        library_sha=sha(library)
        overlay=target/'overlay';overlay.mkdir()
        provider=(OLD/'overlay/localvqe_aec.hpp').read_text()
        if provider.count(original_sha)!=1:raise RuntimeError('Provider SHA binding is ambiguous')
        with (overlay/'localvqe_aec.hpp').open('x') as f:f.write(provider.replace(original_sha,library_sha,1))
        with (overlay/'audio_engine.cpp').open('xb') as f:f.write((OLD/'overlay/audio_engine.cpp').read_bytes())
        engine_object=target/'audio_engine.cpp.o'
        command=['/usr/bin/clang++-17','--target=aarch64-linux-android26',*flags['CXX_DEFINES'],'-I'+str(overlay),*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],'-c',str(overlay/'audio_engine.cpp'),'-o',str(engine_object)]
        execute('compile-provider-'+variant,command,VCP)
        engine=archive('vcp-'+variant,VCP,VCP/'CMakeFiles/itu_audio.dir/link.txt',{'audio_engine.cpp.o':engine_object})
        link_file=VCP/'CMakeFiles/itu_audio_cli.dir/link.txt'
        original_cli=next(x for x in shlex.split(link_file.read_text()) if x.endswith('/itu_audio_cli.cpp.o'))
        binary=target/'itu_audio_cli'
        execute('link-cli-'+variant,link_command(link_file,VCP,binary,{'libitu_audio.a':engine,original_cli:OLD/'native16-cli-r3.o'}),VCP)
        matrix[variant]={'id':variant,'layout':has_layout,'sgemm':has_sgemm,'daf_batch':has_daf,'library':str(library),'library_sha256':library_sha,'binary':str(binary),'binary_sha256':sha(binary)}
        print(json.dumps({'status':'candidate_built',**matrix[variant]}),flush=True)

    if not args.without_probe:
        # Reuse the frozen Android compiler/sysroot rather than host libc.
        entry=next(c for c in commands if c['file'].endswith('/localvqe_api.cpp') and 'localvqe_shared.dir' in c['command'])
        command=shlex.split(entry['command']);output=directory/'probe-runner.o'
        command[command.index('-c')+1]=str(HERE/'validation-root/probe.cpp');command[command.index('-o')+1]=str(output)
        execute('compile-state-runner',command,Path(entry['directory']))
        link=link_command(BASE/'CMakeFiles/localvqe_shared.dir/link.txt',BASE,directory/'state-probe',{})
        core=[x for x in link if not x.endswith(('.a','.o')) and x not in ['-shared','-Wl,-soname,liblocalvqe.so.0']]
        core[core.index('-o')+1]=str(directory/'state-probe')
        execute('link-state-runner',[*core,str(output),'-ldl','-unwindlib=libunwind'],directory)
    receipt={'status':'built','directory':str(directory),'variants':matrix,'state_probe':str(directory/'state-probe') if not args.without_probe else None,'state_probe_sha256':sha(directory/'state-probe') if not args.without_probe else None,'source_hashes':{str(p):sha(p) for p in [p for p in [HERE/'validation-root/probe_api.cpp',HERE/'validation-root/probe.cpp',*[j[1] for j in jobs]] if p.is_file()]},'original_library_unchanged':sha(original)==original_sha,'original_library_sha256':original_sha,'commands':receipts}
    with (directory/'MANIFEST.json').open('x') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
    print(json.dumps({'status':'matrix_built','manifest':str(directory/'MANIFEST.json'),'variants':list(matrix)}),flush=True)


if __name__=='__main__':main()
