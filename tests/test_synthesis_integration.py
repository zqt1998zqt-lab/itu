"""Behavior tests for the catalog-bound synthesis and generated graphics."""
from io import BytesIO
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from server.app import app
from server.audio import AudioError, decode

synth = importlib.import_module('server.synthesize')
ROOT = Path(__file__).resolve().parents[1]
client = TestClient(app)


def wav(values, rate=16000):
    stream = BytesIO()
    sf.write(stream, values, rate, format='WAV', subtype='PCM_16')
    return stream.getvalue()


def joint_chain():
    catalog = json.loads((ROOT / 'src/features/enhancement-chain-configs.json').read_text())
    preset = next(p for p in catalog['presets'] if p['id'] == 'vcp_joint')
    return {key: preset[key] for key in ('profile', 'selection')}


def request(microphone, reference, options=None):
    return client.post('/api/synthesize', files={
        'microphone': ('microphone.wav', microphone, 'audio/wav'),
        'reference': ('reference.wav', reference, 'audio/wav'),
    }, data={'options': json.dumps(options or {})})


def signals(channels=2):
    rng = np.random.default_rng(4242)
    # Independent references with deliberately silent first reference: only
    # preserving channel 2 can expose its echo to the multirender engine.
    right = rng.normal(0, .035, 16000 * 2)
    reference = np.column_stack([np.zeros_like(right), right])
    echo = np.r_[np.zeros(400), right[:-400]] * .65
    near = np.sin(np.arange(len(right)) * 2 * np.pi * 350 / 16000) * .025
    microphone = np.column_stack([echo + near, echo * .9 + near])
    return wav(microphone[:, :channels]), wav(reference)


def test_decoder_preserves_every_channel_and_resamples_time_axis():
    rate = 24000
    t = np.arange(rate) / rate
    values = np.column_stack([np.sin(2 * np.pi * 300 * t) * .1,
                              np.cos(2 * np.pi * 700 * t) * .2])
    payload = wav(values, rate)
    decoded = decode('echo', 'reference.wav', payload, channel=None)
    ordinary = decode('echo', 'reference.wav', payload)
    assert decoded.data.shape == (24000, 2)
    assert np.array_equal(decoded.data[:, 0], ordinary.data)
    converted = synth._resample(decoded.data, rate, 16000)
    assert converted.shape == (16000, 2)
    assert np.max(np.abs(converted[:, 1])) > .19


def test_catalog_rejects_untrusted_flags_and_invalid_joint_donors():
    chain = joint_chain()
    flags, canonical, digest = synth._chain_flags('vcp_joint', chain)
    assert canonical == chain and len(digest) == 64
    assert '--vehicle-ns-legacy-budget' not in flags
    with pytest.raises(AudioError):
        synth._chain_flags('vcp_joint', {**chain, 'extra_flags': ['--no-ns']})
    with pytest.raises(AudioError, match='Super-Gaussian'):
        synth._chain_flags('vcp_joint', {**chain, 'selection': {**chain['selection'], 'ns': 'webrtc'}})
    with pytest.raises(AudioError):
        synth._chain_flags('vcp', chain)
    for delays in (None, [0], [0, float('nan')], [-1, 0], [True, 0]):
        with pytest.raises(AudioError):
            synth._microphone_flags(2, 'beamform', delays, chain, 16000)
    with pytest.raises(AudioError):
        synth._microphone_flags(2, 'channel0', None, chain, 16000)


def test_api_rejects_options_before_starting_cli():
    mic, ref = signals()
    for options in ({'sample_rate': True}, {'profile': '--no-ns'}, {'flags': ['--no-aec']},
                    {'chain': {**joint_chain(), 'selection': {'bf': 'das'}}},
                    {'profile': 'vcp_joint', 'microphone_mode': 'beamform', 'beam_delays_samples': [0]}):
        response = request(mic, ref, options)
        assert response.status_code == 422, response.text


def test_real_api_joint_preserves_stereo_inputs_uses_host_cli_and_binds_graphics(monkeypatch):
    monkeypatch.delenv('ITU_CLI', raising=False)
    mic, ref = signals()
    response = request(mic, ref, {'profile': 'vcp_joint', 'chain': joint_chain(),
                                 'microphone_mode': 'beamform', 'beam_delays_samples': [1.25, 0]})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['binary'] == str(ROOT / 'build/host/cpp/itu_audio_cli')
    assert result['binary_sha256'] == '4760e6a1b705fc8f5bba72debe5bb12b82bb6bf0a73281ca817f6adc68eab0f7'
    assert result['engine_report']['reference_channels'] == 2
    assert result['engine_report']['microphones'] == 2
    assert result['engine_report']['beamforming_enabled'] is True
    assert result['engine_report']['speex_multirender_context'] is True
    assert result['engine_report']['vehicle_ns_unified_raw_budget'] is True
    assert result['engine_report']['tx_drc_config']['threshold_dbfs'] == -6
    assert result['inputs']['reference']['processed_channels'] == 2
    assert result['inputs']['reference']['selected_channel'] is None
    assert result['joint_control']['budget_violations'] == 0
    assert result['joint_control']['raw_mix_applied_max'] <= .250001
    work = Path(result['diagnostics_dir'])
    request_info = json.loads((work / 'request.json').read_text())
    assert request_info['command'][1:3] == ['--profile', 'vcp_joint']
    saved_reference, rate = sf.read(work / 'ref.wav', always_2d=True)
    original_reference, _ = sf.read(BytesIO(ref), always_2d=True)
    assert rate == 16000 and np.array_equal(saved_reference, original_reference)
    assert np.all(saved_reference[:, 0] == 0) and np.any(saved_reference[:, 1] != 0)
    downloaded = client.get(result['download_url'])
    assert downloaded.status_code == 200
    output, _ = sf.read(BytesIO(downloaded.content), dtype='int16')
    produced, _ = sf.read(work / 'out.after.wav', dtype='int16')
    assert np.array_equal(output, produced)
    report = client.post(result['graphics_url'])
    assert report.status_code == 200, report.text
    graphics = report.json()
    assert graphics['alignment']['method'] == 'declared_dsp_delay'
    assert graphics['synthesis']['fixed_delay_samples'] == result['engine_report']['dsp_delay_samples']
    assert graphics['synthesis']['best_lag_search'] is False
    assert graphics['synthesis']['level_normalized'] is False
    assert graphics['synthesis']['run_id'] == result['run_id']
    assert len(graphics['figures']) == 8
    image = client.get(graphics['figures'][0]['url'])
    assert image.status_code == 200 and image.content.startswith(b'\x89PNG\r\n\x1a\n')


def test_direct_default_profile_remains_vcp(monkeypatch):
    monkeypatch.delenv('ITU_CLI', raising=False)
    monkeypatch.delenv('ITU_SYNTH_PROFILE', raising=False)
    mic, ref = signals(1)
    response = request(mic, ref)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['profile'] == 'vcp'
    assert result['engine_report']['speex_multirender_context'] is False
    assert result['engine_report']['vehicle_ns_unified_raw_budget'] is False
    assert result['engine_report']['tx_drc_config']['threshold_dbfs'] == -12
