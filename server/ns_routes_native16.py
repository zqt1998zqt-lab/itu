"""Independent 256-point CoFi route; legacy route remains unchanged."""
from server.native16_variants import load_base
load_base('ns_routes',globals(),replace=[('"cofilite"', '"cofilite_native16"', 8), ('from server.ns_streaming import', 'from server.ns_streaming_native16 import', 1), ('persistent_f32le_v1', 'persistent_f32le_native16_v1', 1)],patterns=[('\\b160\\b', '256', 7)])
from dataclasses import replace as _replace
NATIVE16_CLI=ROOT/'build/native16-20261117/host/cpp/itu_audio_cli'
EXTERNAL_NS_IDS=frozenset({'cofilite_native16'})
_spec=NS_ROUTES['cofilite_native16']
_worker=ROOT/'server/ns_stream_worker_native16.py'
_extra=tuple((ROOT/'server'/name,_sha256_file(ROOT/'server'/name)) for name in ['native16_variants.py','ns_routes_native16.py','ns_streaming_native16.py','ns_stream_worker_native16.py','aec_routes_native16.py'])
NS_ROUTES={'cofilite_native16':_replace(_spec,backend='cofilite_native16',model_id='cofilite',delay_samples=256,sample_rate=16000,frame_samples=256,window_samples=512,hop_samples=256,replay_drain_samples=16000,frontend_path=_worker,frontend_sha256=_sha256_file(_worker),extra_assets=(*_spec.extra_assets,*_extra))}

def is_native16_configuration(configuration):
    return isinstance(configuration,dict) and isinstance(configuration.get('selection'),dict) and configuration['selection'].get('ns')=='cofilite_native16'

def ns_backend_for(configuration):
    return _native16_original_ns_backend_for(configuration)

def ns_route_identity(configuration):
    return _native16_original_ns_route_identity(configuration)

def execute_ns_command(command,*,configuration,timeout=300,**kwargs):
    if not is_native16_configuration(configuration):
        raise ValueError('native16 route requires explicit native16 configuration')
    from server.ns_streaming_native16 import execute_stream_command
    return execute_stream_command(command,configuration=configuration,timeout=timeout,**kwargs)

def resolve_ns_cli(default_cli,configuration):
    if not is_native16_configuration(configuration) or configuration['selection'].get('aec')!='localvqe_aec200k':
        raise ValueError('native16 requires explicit LocalVQE/CoFi native16 selection')
    if not NATIVE16_CLI.is_file() or not os.access(NATIVE16_CLI,os.X_OK):raise ValueError('native16 CLI missing')
    return NATIVE16_CLI

_validate_native_base=_native16_original_validate_ns_route_report
def validate_ns_route_report(report,configuration):
    if not isinstance(report,dict):
        return _validate_native_base(report,configuration)
    # Frozen native16 smoke evidence predates the generic geometry fields.
    # Validate its explicit native16 identity while supplying only the
    # independently pinned route constants that this wrapper already owns.
    candidate=dict(report)
    identity=ns_route_identity(configuration)
    if identity:
        for key,value in identity.items():
            candidate.setdefault(key,value)
    errors=_validate_native_base(candidate,configuration)
    for field,expected in {'frame_size':256,'sample_rate':16000,'aec_hop_adapter_delay_samples_16k':0,'aec_algorithm_delay_samples_16k':256,'ns_stream_chunk_samples':256}.items():
        value=report.get(field)
        if type(value) is not type(expected) or value!=expected:errors.append(f'native16 {field} must equal {expected}')
    return errors

# A separately built diagnostic-length fix; prior binary remains preserved.
NATIVE16_CLI=ROOT/'build/native16-20261117/host-v2/cpp/itu_audio_cli'
