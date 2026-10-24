#!/usr/bin/env python3
"""Build from retained C++ sources and bind every reused link input by hash."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

P = Path(__file__).resolve().parents[2]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    previous = json.loads((P/'build/ns-failure-repair-v3-20261018/floor-build-identity.json').read_text())
    files = [P/n for n in ('cpp/src/vehicle_ns_controller.hpp', 'cpp/src/vehicle_rumble_filter.hpp',
             'cpp/tools/ns_vehicle_probe.cpp', 'cpp/tools/ns_vehicle_stream.cpp',
             'cpp/src/vehicle_ns.cpp', 'cpp/include/itu/vehicle_ns.hpp', 'cpp/src/residual_noise_suppressor.hpp', 'cpp/src/ns_repair_alignment.hpp',
             'cpp/src/speex_ns.hpp', 'cpp/src/omlsa_imcra.hpp', 'cpp/src/super_gaussian.hpp', 'cpp/src/fft.hpp', 'cpp/src/rate_converter.hpp',
             'cpp/src/wav.hpp', 'cpp/tools/repair_ns_vehicle_v4.py')]
    source = {str(f.relative_to(P)): sha(f) for f in files}
    for f in files:
        target = out/'source'/f.relative_to(P)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
    compile_ = previous['compile'][:]
    compile_[1:1] = ['-I'+str(out/'source/cpp/src'), '-I'+str(out/'source/cpp/include')]
    compile_[compile_.index('-o')+1] = str(out/'probe.o')
    compile_[-1] = str(out/'source/cpp/tools/ns_vehicle_stream.cpp')
    link = previous['link'][:]
    link = [str(out/'probe.o') if x.endswith('/ns_floor_repair_probe.o') else x for x in link]
    link[link.index('-o')+1] = str(out/'ns_vehicle_probe')
    module_compile = compile_[:]
    module_compile[module_compile.index('-o')+1] = str(out/'vehicle_ns.o')
    module_compile[-1] = str(out/'source/cpp/src/vehicle_ns.cpp')
    link.insert(link.index('-o'), str(out/'vehicle_ns.o'))
    cwd = Path(previous['cwd'])
    dependencies = {}
    for x in compile_ + link:
        p = Path(x)
        p = p if p.is_absolute() else cwd/p
        if p.is_file() and (p.suffix in ('.a', '.o') or p.name == 'c++') and p.resolve() != (out/'probe.o').resolve():
            dependencies[str(p.resolve())] = sha(p)
    with (out/'build.log').open('x') as log:
        for argv in (module_compile, compile_, link):
            subprocess.run(argv, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)
    assert all(sha(P/n) == h for n,h in source.items()), 'source changed during build'
    assert all(sha(Path(n)) == h for n,h in dependencies.items()), 'link input changed'
    identity = dict(compile=compile_, module_compile=module_compile, link=link, cwd=str(cwd), source_sha256=source,
                    dependency_sha256=dependencies, binary_sha256=sha(out/'ns_vehicle_probe'))
    (out/'identity.json').write_text(json.dumps(identity, indent=2)+'\n')
    print(json.dumps(dict(binary=str(out/'ns_vehicle_probe'), sha256=identity['binary_sha256'])))

if __name__ == '__main__':
    main()
