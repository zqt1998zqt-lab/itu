"""LocalVQE identity variant with native hop and zero adapter PCM shift."""
from server.native16_variants import load_base
load_base('aec_routes',globals(),replace=[('_LOCALVQE_ADAPTER_DELAY = 224','_LOCALVQE_ADAPTER_DELAY = 0',1),('"dpdfnet2_48k_hr", "hybrid_v2_pitch",','"dpdfnet2_48k_hr", "hybrid_v2_pitch", "cofilite_native16",',1)])
_base_admit=_native16_original_admit_aec_route
_base_read_aec_capabilities=_native16_original_read_aec_capabilities
_base_validate_aec_route_report=_native16_original_validate_aec_route_report
_base_validate_admitted_aec_identity=_native16_original_validate_admitted_aec_identity

def read_aec_capabilities(cli):
    return _base_read_aec_capabilities(cli)

def validate_aec_route_report(report,configuration):
    return _base_validate_aec_route_report(report,configuration)

def validate_admitted_aec_identity(report,capability):
    return _base_validate_admitted_aec_identity(report,capability)

def admit_aec_route(cli,configuration,**kwargs):
    expected=Path(__file__).resolve().parents[1]/'build/native16-20261117/host/cpp/itu_audio_cli'
    if Path(cli).resolve()!=expected.resolve():raise ValueError('native16 CLI identity mismatch')
    return _base_admit(cli,configuration,**kwargs)

def admit_aec_route(cli,configuration,**kwargs):
    expected=Path(__file__).resolve().parents[1]/'build/native16-20261117/host-v2/cpp/itu_audio_cli'
    if Path(cli).resolve()!=expected.resolve():raise ValueError('native16 CLI identity mismatch')
    return _base_admit(cli,configuration,**kwargs)
