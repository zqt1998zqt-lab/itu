"""Build an isolated native-256 variant without changing historical sources.

Every transformation is explicit and counted; the generated source and original
hashes are archived in a unique attempt. No existing build output is overwritten.
Pinned WebRTC/Speex/RNNoise static dependencies are reused from the tested host.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'build/cofilite-opt-20261116/host-v3'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resources():
    if shutil.disk_usage(ROOT).free < 100 * 1024**3:
        raise RuntimeError('disk safety threshold: stop without cleanup')
    available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    if available < 5 * 1024**3:
        raise RuntimeError('memory safety threshold: stop without cleanup')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'build/native16-20261117/host/cpp/itu_audio_cli')
    args = parser.parse_args()
    resources()
    if args.output.exists():
        raise FileExistsError(f'choose a new output; existing evidence is preserved: {args.output}')
    work = ROOT / 'build/native16-20261117'
    work.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix='compile-', dir=work))
    overlay = attempt / 'src'
    overlay.mkdir()
    changes = []
    originals = {}

    def transform(relative, replacements):
        source = ROOT / relative
        text = source.read_text()
        originals[relative] = {'sha256': sha(source), 'bytes': source.stat().st_size}
        for before, after, expected in replacements:
            count = text.count(before)
            if count != expected:
                raise RuntimeError(f'{relative}: replacement count {count} != {expected}: {before!r}')
            text = text.replace(before, after)
            changes.append({'source': relative, 'before': before, 'after': after, 'count': count})
        output = overlay / source.name
        with output.open('x') as stream:
            stream.write(text)
        return output

    transform('cpp/src/localvqe_aec.hpp', [
        ('#include "continuous_hop_adapter.hpp"', '#include "native16_hop.hpp"', 1),
        ('ContinuousHopAdapter', 'Native16HopAdapter', 2),
        ('mic16_(160)', 'mic16_(c_.frame_size)', 1),
        ('ref16_(160)', 'ref16_(c_.frame_size)', 1),
        ('aec16_(160)', 'aec16_(c_.frame_size)', 1),
        ('        render_hold_ = 100', '        render_hold_ = 63', 1),
        ('post_render_hold_ = 100', 'post_render_hold_ = 63', 1),
    ])
    transform('cpp/src/stream_ns_pipe.hpp', [
        ('std::array<unsigned char, 160 * sizeof(float)>', 'std::array<unsigned char, 256 * sizeof(float)>', 1),
    ])
    engine = transform('cpp/src/audio_engine.cpp', [
        ('#include "localvqe_aec.hpp"', '#include "localvqe_aec.hpp"\n#include "native16_post_pipeline.hpp"', 1),
        ('c.frame_size != c.sample_rate/100 || !c.microphones', '(c.frame_size != c.sample_rate/100 && !(c.sample_rate==16000 && c.frame_size==256)) || !c.microphones', 1),
        ('if(c.sample_rate!=16000 || c.frame_size!=160)\n      throw std::invalid_argument(\n          "external streaming NS', 'if(c.sample_rate!=16000 || c.frame_size!=256)\n      throw std::invalid_argument(\n          "external streaming NS', 1),
        ('if(c.ns_stream_delay_samples>1600 || c.ns_stream_delay_samples%160)', 'if(c.ns_stream_delay_samples!=256)', 1),
        ('if(c.sample_rate!=16000 || c.frame_size!=160)\n      throw std::invalid_argument("external post-NS PCM', 'if(c.sample_rate!=16000 || (c.frame_size!=160 && c.frame_size!=256))\n      throw std::invalid_argument("external post-NS PCM', 1),
        ('if(c.post_ns_delay_samples>1600 || c.post_ns_delay_samples%160)', 'if(c.post_ns_delay_samples>1600 || c.post_ns_delay_samples%c.frame_size)', 1),
        ('std::unique_ptr<LocalVqeAecPipeline> localvqe_aec;', 'std::unique_ptr<LocalVqeAecPipeline> localvqe_aec;\n  std::unique_ptr<Native16PostPipeline> native16_post;', 1),
        ('    // The standalone OM-LSA path', '    if(c.frame_size==256) {\n      native16_post=std::make_unique<Native16PostPipeline>(c); return;\n    }\n    // The standalone OM-LSA path', 1),
        ('  if(s.pre_aec) {\n    DenormalScope denormals;', '  if(s.native16_post) {\n    const auto result=s.native16_post->process(mic,out,info,context);\n    if(aec_tap) std::copy_n(mic,h,aec_tap);\n    if(aec_input_tap) std::copy_n(mic,h,aec_input_tap);\n    if(bf_tap) std::fill_n(bf_tap,h,0.f);\n    if(linear_tap) std::fill_n(linear_tap,160,0.f);\n    if(result!=Status::ok) s.failed_backend_error=info.backend_error ? info.backend_error : -1016;\n    return result;\n  }\n  if(s.pre_aec) {\n    DenormalScope denormals;', 1),
    ])
    cli = transform('cpp/tools/itu_audio_cli.cpp', [
        ('c.frame_size = c.sample_rate / 100;', 'c.frame_size = c.sample_rate==16000 ? 256 : c.sample_rate/100;', 2),
        ('else if(a=="--pace-10ms") paced=true;', 'else if(a=="--pace-16ms") paced=true;', 1),
        ('c.frame_size!=160 || c.microphones!=1 ||', 'c.frame_size!=256 || c.microphones!=1 ||', 1),
        ('if(value>1600 || value%160)', 'if(value!=256)', 2),
        ('std::chrono::milliseconds(k*10)', 'std::chrono::milliseconds(k*16)', 1),
        ('std::chrono::milliseconds(10)', 'std::chrono::milliseconds(16)', 1),
        ('process_misses+=ms>10', 'process_misses+=ms>16', 1),
        ('total/(frames*10.)', 'total/(frames*16.)', 1),
        ('paced_10ms', 'paced_16ms', 1),
        ('process_frames_over_10ms', 'process_frames_over_16ms', 1),
        ('float32[160]', 'float32[256]', 2),
    ])
    make = (BASE / 'cpp/CMakeFiles/itu_audio.dir/flags.make').read_text()
    definitions = next(row.split(' = ', 1)[1] for row in make.splitlines() if row.startswith('CXX_DEFINES = '))
    includes = next(row.split(' = ', 1)[1] for row in make.splitlines() if row.startswith('CXX_INCLUDES = '))
    flags = next(row.split(' = ', 1)[1] for row in make.splitlines() if row.startswith('CXX_FLAGS = '))
    # flags.make quotes the preprocessor string for a shell, so shlex restores
    # one actual argument containing a C++ quoted ITU_PROJECT_ROOT string.
    options = [*shlex.split(definitions), *shlex.split(includes), *shlex.split(flags), '-I' + str(overlay), '-I' + str(ROOT / 'cpp/src')]
    objects = []
    commands = []
    sources = [engine, cli, ROOT / 'cpp/src/native_pipeline.cpp', ROOT / 'cpp/src/eq_processor.cpp', ROOT / 'cpp/src/reference_clock.cpp', ROOT / 'cpp/src/spatial_frontend.cpp', ROOT / 'cpp/src/neural_ns.cpp', ROOT / 'cpp/src/vehicle_ns.cpp', ROOT / 'cpp/src/wav.cpp']
    with (attempt / 'compile.log').open('x') as log:
        for index, source in enumerate(sources):
            resources()
            target = attempt / (source.stem + '.o')
            command = ['/usr/bin/c++', *options, '-c', str(source), '-o', str(target)]
            commands.append(command)
            print(f'compile {index+1}/{len(sources)} {source.name}', flush=True)
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            objects.append(target)
        original_link = shlex.split((BASE / 'cpp/CMakeFiles/itu_audio_cli.dir/link.txt').read_text())
        library_start = original_link.index('libitu_webrtc.a')
        libraries = [str((BASE / 'cpp' / part).resolve()) if part.endswith('.a') else part for part in original_link[library_start:]]
        binary = attempt / 'itu_audio_cli'
        link = ['/usr/bin/c++', '-O3', '-DNDEBUG', *(str(item) for item in objects), '-o', str(binary), *libraries]
        commands.append(link)
        subprocess.run(link, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with binary.open('rb') as source, args.output.open('xb') as destination:
        shutil.copyfileobj(source, destination)
    args.output.chmod(0o755)
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'originals': originals, 'transformations': changes, 'commands': commands, 'binary': str(args.output), 'binary_sha256': sha(args.output), 'generated': {str(path): sha(path) for path in overlay.iterdir()}, 'new_headers': {str(path): sha(path) for path in [ROOT / 'cpp/src/native16_hop.hpp', ROOT / 'cpp/src/native16_post_pipeline.hpp']}}
    with (attempt / 'manifest.json').open('x') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'attempt': str(attempt), 'binary': str(args.output), 'sha256': manifest['binary_sha256']}), flush=True)


if __name__ == '__main__':
    main()
