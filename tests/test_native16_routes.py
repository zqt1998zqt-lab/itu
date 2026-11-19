"""Native16 has independent geometry and cannot borrow legacy receipts."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from server import aec_routes, ns_routes
from server import ns_routes_native16 as native
from server import ns_streaming_native16 as parent
from server.alignment_runner import resolve_algorithm_cluster

ROOT=Path(__file__).resolve().parents[1]


def test_native16_catalog_and_legacy_selection_stay_distinct():
    preset,flags,config,digest=resolve_algorithm_cluster('localvqe_cofilite_native16')
    assert config['selection']['ns']=='cofilite_native16'
    assert config['selection']['eq']=='three_band'
    assert flags.count('--eq-band')==3
    assert config['selection']['agc']=='guarded5' and '--tx-drc' in flags
    assert ns_routes.resolve_ns_cli(None,config).is_relative_to(ROOT/'build/native16-20261117/host-v2')
    old=resolve_algorithm_cluster('localvqe_cofilite')[2]
    assert ns_routes.ns_route_identity(old)['ns_delay_samples']==480
    assert ns_routes.ns_route_identity(config)['ns_delay_samples']==256
    assert ns_routes.resolve_ns_cli(None,old).is_relative_to(ROOT/'build/cofilite-opt-20261116/host-v3')


def test_real_native_capability_and_wrong_cli_negative():
    config=resolve_algorithm_cluster('localvqe_cofilite_native16')[2]
    cli=ns_routes.resolve_ns_cli(None,config)
    cap=aec_routes.admit_aec_route(cli,config,sample_rate=16000,reference_channels=1)
    assert cap['delay_samples']==256
    old=ns_routes.resolve_ns_cli(None,resolve_algorithm_cluster('localvqe_cofilite')[2])
    with pytest.raises(ValueError,match='identity mismatch'):
        aec_routes.admit_aec_route(old,config,sample_rate=16000,reference_channels=1)
    with pytest.raises(ValueError):
        aec_routes.admit_aec_route(cli,config,sample_rate=48000,reference_channels=1)


@pytest.fixture(scope='module')
def actual():
    report=json.loads((ROOT/'build/native16-20261117/smoke-v2-zo_z8ewv/tx.json').read_text())
    return report,resolve_algorithm_cluster('localvqe_cofilite_native16')[2]


def test_actual_native16_geometry_and_strict_receipt(actual):
    report,config=actual
    assert report['frame_size']==256
    assert report['dsp_delay_samples']==513 and report['dsp_delay_ms']==32.0625
    assert report['aec_localvqe_inferences']==752
    assert report['ns_stream_callback_frames']==753
    assert ns_routes.validate_ns_route_report(report,config)==[]
    assert aec_routes.validate_aec_route_report(report,config)==[]
    receipt=json.loads((ROOT/'build/native16-20261117/smoke-v2-zo_z8ewv/tx.ns-stream.json').read_text())
    parent._validate_worker_identity(receipt,753)
    wrong=deepcopy(receipt);wrong['samples']=753*160
    with pytest.raises(ValueError,match='geometry'):
        parent._validate_worker_identity(wrong,753)


@pytest.mark.parametrize('field,value',[
    ('frame_size',160),('ns_delay_samples',480),('ns_stream_chunk_samples',160),
    ('aec_hop_adapter_delay_samples_16k',224),('ns_external_fallback_frames',1),
    ('ns_frame_model_sha256','0'*64),('ns_worker_transport','persistent_f32le_v1'),
    ('aec_algorithm_delay_samples_16k',480),
])
def test_native16_rejects_forged_or_legacy_report(actual,field,value):
    report,config=actual
    wrong=deepcopy(report);wrong[field]=value
    assert ns_routes.validate_ns_route_report(wrong,config)


def test_new_route_assets_and_original_prefixes_remain_bound():
    native._verify_route_assets('cofilite_native16')
    from server.native16_variants import BASE_IDENTITIES
    import hashlib
    for name,lock in BASE_IDENTITIES.items():
        data=(ROOT/f'server/{name}.py').read_bytes()[:lock['bytes']]
        assert hashlib.sha256(data).hexdigest()==lock['sha256']
