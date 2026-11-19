"""Create auditable, source-locked native16 transport variants, new files only."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]

def main():
    names=['ns_routes','ns_streaming','ns_stream_worker','aec_routes']
    identities={name:{'bytes':(ROOT/f'server/{name}.py').stat().st_size,'sha256':hashlib.sha256((ROOT/f'server/{name}.py').read_bytes()).hexdigest()} for name in names}
    helper='''"""Counted transformations of immutable 10ms sources into native16 variants."""
from pathlib import Path
import hashlib
import re
ROOT=Path(__file__).resolve().parents[1]
BASE_IDENTITIES = '''+repr(identities)+'''

def load_base(name, namespace, *, replace=(), patterns=()):
    path=ROOT/f"server/{name}.py"
    lock=BASE_IDENTITIES[name]
    data=path.read_bytes()[:lock['bytes']]
    if hashlib.sha256(data).hexdigest()!=lock['sha256']:
        raise RuntimeError(f"native16 base source changed: {name}")
    text=data.decode()
    guard='if __name__ == "__main__":'
    if guard in text:text=text[:text.rindex(guard)]
    for old,new,count in replace:
        actual=text.count(old)
        if actual!=count:raise RuntimeError(f"native16 replacement count mismatch {name}: {old!r} {actual}!={count}")
        text=text.replace(old,new)
    for pattern,new,count in patterns:
        text,actual=re.subn(pattern,new,text)
        if actual!=count:raise RuntimeError(f"native16 regex count mismatch {name}: {pattern!r} {actual}!={count}")
    exec(compile(text,str(path)+":native16",'exec'),namespace)
'''
    outputs={'native16_variants':helper}
    ns_source=(ROOT/'server/ns_routes.py').read_text()
    worker_source=(ROOT/'server/ns_stream_worker.py').read_text()
    parent_source=(ROOT/'server/ns_streaming.py').read_text()
    import re
    def patterns(source):return [(r'\b160\b','256',len(re.findall(r'\b160\b',source)))]
    ns_replacements=[('"cofilite"','"cofilite_native16"',ns_source.count('"cofilite"')),('from server.ns_streaming import','from server.ns_streaming_native16 import',1)]
    outputs['ns_routes_native16']='''"""Independent 256-point CoFi route; legacy route remains unchanged."""
from server.native16_variants import load_base
load_base('ns_routes',globals(),replace='''+repr(ns_replacements)+''',patterns='''+repr(patterns(ns_source))+''')
from dataclasses import replace as _replace
NATIVE16_CLI=ROOT/'build/native16-20261117/host/cpp/itu_audio_cli'
EXTERNAL_NS_IDS=frozenset({'cofilite_native16'})
_spec=NS_ROUTES['cofilite_native16']
_worker=ROOT/'server/ns_stream_worker_native16.py'
_extra=tuple((ROOT/'server'/name,_sha256_file(ROOT/'server'/name)) for name in ['native16_variants.py','ns_routes_native16.py','ns_streaming_native16.py','ns_stream_worker_native16.py','aec_routes_native16.py'])
NS_ROUTES={'cofilite_native16':_replace(_spec,backend='cofilite_native16',model_id='cofilite',delay_samples=256,frontend_path=_worker,frontend_sha256=_sha256_file(_worker),extra_assets=(*_spec.extra_assets,*_extra))}

def is_native16_configuration(configuration):
    return isinstance(configuration,dict) and isinstance(configuration.get('selection'),dict) and configuration['selection'].get('ns')=='cofilite_native16'

def resolve_ns_cli(default_cli,configuration):
    if not is_native16_configuration(configuration) or configuration['selection'].get('aec')!='localvqe_aec200k':
        raise ValueError('native16 requires explicit LocalVQE/CoFi native16 selection')
    if not NATIVE16_CLI.is_file() or not os.access(NATIVE16_CLI,os.X_OK):raise ValueError('native16 CLI missing')
    return NATIVE16_CLI

_validate_native_base=validate_ns_route_report
def validate_ns_route_report(report,configuration):
    errors=_validate_native_base(report,configuration)
    if not isinstance(report,dict):return errors
    for field,expected in {'frame_size':256,'sample_rate':16000,'aec_hop_adapter_delay_samples_16k':0,'aec_algorithm_delay_samples_16k':256,'ns_stream_chunk_samples':256}.items():
        value=report.get(field)
        if type(value) is not type(expected) or value!=expected:errors.append(f'native16 {field} must equal {expected}')
    return errors
'''
    replacements=[('"cofilite"','"cofilite_native16"',worker_source.count('"cofilite"')),('from server.ns_routes import','from server.ns_routes_native16 import',1),('"itu-ns-f32le-v1"','"itu-ns-f32le-native16-v1"',1),('frame_executor=executor,','frame_executor=executor, frame_samples=256, delay_samples=256,',1),('over_10ms','over_16ms',worker_source.count('over_10ms')),('0.010','0.016',worker_source.count('0.010'))]
    outputs['ns_stream_worker_native16']='''"""Persistent native16 worker; inherited protocol checks plus distinct geometry."""
from server.native16_variants import load_base
load_base('ns_stream_worker',globals(),replace='''+repr(replacements)+''',patterns='''+repr(patterns(worker_source))+''')
if __name__=='__main__':raise SystemExit(main())
'''
    replacements=[('"cofilite"','"cofilite_native16"',parent_source.count('"cofilite"')),('from server import ns_routes as routes','from server import ns_routes_native16 as routes',1),('"server.ns_stream_worker"','"server.ns_stream_worker_native16"',1),('with_name("ns_stream_worker.py")','with_name("ns_stream_worker_native16.py")',1),('"itu-ns-f32le-v1"','"itu-ns-f32le-native16-v1"',1),('persistent_f32le_v1','persistent_f32le_native16_v1',1),('over_10ms','over_16ms',parent_source.count('over_10ms'))]
    outputs['ns_streaming_native16']='''"""Control lifecycle for explicit 256-point stateful native16 callbacks."""
from server.native16_variants import load_base
load_base('ns_streaming',globals(),replace='''+repr(replacements)+''',patterns='''+repr(patterns(parent_source))+''')
'''
    # Transport string is checked both in the report validator and in wrapper.
    outputs['ns_routes_native16']=outputs['ns_routes_native16'].replace("(\"from server.ns_streaming import\", \"from server.ns_streaming_native16 import\", 1)","(\"from server.ns_streaming import\", \"from server.ns_streaming_native16 import\", 1)")
    ns_replacements.extend([('persistent_f32le_v1','persistent_f32le_native16_v1',1)])
    # Replace the serialized replacement-list literal, preserving all counted operations.
    old=repr(ns_replacements[:-1]);new=repr(ns_replacements)
    assert old in outputs['ns_routes_native16']
    outputs['ns_routes_native16']=outputs['ns_routes_native16'].replace(old,new,1)
    outputs['aec_routes_native16']='''"""LocalVQE identity variant with native hop and zero adapter PCM shift."""
from server.native16_variants import load_base
load_base('aec_routes',globals(),replace=[('_LOCALVQE_ADAPTER_DELAY = 224','_LOCALVQE_ADAPTER_DELAY = 0',1),('"hybrid_v2_pitch"}','"hybrid_v2_pitch", "cofilite_native16"}',1)])
_base_admit=admit_aec_route
def admit_aec_route(cli,configuration,**kwargs):
    expected=Path(__file__).resolve().parents[1]/'build/native16-20261117/host/cpp/itu_audio_cli'
    if Path(cli).resolve()!=expected.resolve():raise ValueError('native16 CLI identity mismatch')
    return _base_admit(cli,configuration,**kwargs)
'''
    for name,source in outputs.items():
        path=ROOT/f'server/{name}.py'
        with path.open('x') as f:f.write(source)
    # Import after all mutually referenced source paths exist.
    import sys
    sys.path.insert(0,str(ROOT))
    from server import ns_routes_native16 as routes
    configuration={'selection':{'aec':'localvqe_aec200k','ns':'cofilite_native16'}}
    print(json.dumps({'files':list(outputs),'route':routes.ns_route_identity(configuration),'originals':identities},ensure_ascii=False))

if __name__=='__main__':main()
