from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf

from server.acoustic import ns_quality_metrics as nsq
from server.acoustic import scene_condition_batch_api as api


def example():
    rate = 16000
    time = np.arange(rate * 3) / rate
    speech = 0.1 * (np.sin(2 * np.pi * 187 * time) + 0.4 * np.sin(2 * np.pi * 391 * time))
    speech *= ((time >= 0.5) & (time < 2.0))
    noise = np.random.default_rng(83).normal(0.006, 0.003, len(time))
    return speech, noise, rate


def calculate(ref, mic, tx, far=None):
    return nsq.calculate_metrics(ref, mic, tx, np.zeros_like(ref) if far is None else far,
                                 sample_rate=16000, residual_alignment=False)


def test_identity_and_pure_noise_attenuation():
    ref, noise, _ = example()
    result = calculate(ref, ref + noise, ref + noise * 0.1)
    assert result["analysis"]["noise_observation_s"] > 1
    assert result["after"]["noise_attenuation_db"]["value"] == pytest.approx(20, abs=0.02)
    expected = 20 * np.log10(result["after"]["residual_noise_rms_dbfs"]["linear_rms"])
    assert result["after"]["residual_noise_rms_dbfs"]["value"] == pytest.approx(expected)
    identity = calculate(ref, ref, ref)
    assert identity["after"]["si_sdr_db"]["display"] == "+∞"
    assert identity["after"]["speech_truncation_rate_pct"]["value"] == 0
    assert identity["after"]["onset_loss_db"]["value"] == pytest.approx(0, abs=1e-10)


def test_dc_distinguishes_si_sdr_from_zero_mean_si_snr():
    ref, _, _ = example()
    result = calculate(ref, ref, ref + 0.03)
    assert result["after"]["si_sdr_db"]["value"] is not None
    assert result["after"]["si_snr_db"]["display"] == "+∞"


def test_far_end_activity_excludes_noise_observation_windows():
    ref, noise, _ = example()
    far = 0.05 * np.sin(2 * np.pi * 229 * np.arange(len(ref)) / 16000)
    result = calculate(ref, ref + noise, ref + noise * 0.1, far)
    assert result["analysis"]["noise_observation_s"] == 0
    assert result["after"]["noise_attenuation_db"]["value"] is None
    assert "观察窗" in result["after"]["noise_attenuation_db"]["reason"]


def test_silence_is_not_a_good_speech_score_or_perfect_retention():
    ref, noise, _ = example()
    result = calculate(ref, ref + noise, np.zeros_like(ref))
    assert result["after"]["si_sdr_db"]["status"] == "undefined"
    assert result["after"]["speech_truncation_rate_pct"]["value"] == 100
    assert result["after"]["onset_loss_db"]["display"] == "+∞"
    only_noise = calculate(np.zeros_like(ref), noise, noise * 0.1)
    assert only_noise["after"]["pesq"]["status"] == "not_applicable"
    assert only_noise["after"]["stoi"]["status"] == "not_applicable"
    assert only_noise["after"]["speech_truncation_rate_pct"]["value"] is None


def test_onset_and_tail_erasure_are_separate_from_waveform_clipping():
    ref, _, _ = example()
    out = ref.copy()
    out[8000:9000] = 0
    out[31000:32000] = 0
    result = calculate(ref, ref, out)
    assert result["after"]["onset_loss_db"]["value"] > 3
    assert result["after"]["tail_loss_db"]["value"] > 3
    assert result["after"]["speech_truncation_rate_pct"]["value"] > 0
    assert result["after"]["clipping_rate_pct"]["value"] == 0


def test_weak_speech_retention_is_relative_to_whole_speech_gain():
    ref, _, _ = example()
    ref[8000:11200] *= 0.05
    output = ref.copy()
    output[8000:11200] = 0
    result = calculate(ref, ref, output)
    assert result["after"]["weak_speech_retention_pct"]["eligible_frames"] > 0
    assert result["after"]["weak_speech_retention_pct"]["value"] < 10


def test_polqa_never_uses_the_pesq_value():
    ref, _, _ = example()
    result = calculate(ref, ref, ref)
    assert result["after"]["polqa"]["status"] == "unavailable"
    assert result["after"]["polqa"]["value"] is None


def test_input_and_output_use_identical_reference_overlap_after_alignment():
    ref, noise, _ = example()
    delayed = np.pad(ref, (160, 0))[:len(ref)] + noise * 0.01
    result = nsq.calculate_metrics(ref, ref + noise, delayed, np.zeros_like(ref), sample_rate=16000)
    assert abs(result["analysis"]["residual_reference_lag_samples"] - 160) <= 1
    assert result["analysis"]["common_reference_samples"] < len(ref)
    assert result["before"]["speech_truncation_rate_pct"]["eligible_frames"] == result["after"]["speech_truncation_rate_pct"]["eligible_frames"]


def test_nonfinite_and_wrong_length_rejected():
    ref, _, _ = example()
    out = ref.copy()
    out[20] = np.nan
    with pytest.raises(ValueError, match="有限"):
        calculate(ref, ref, out)
    with pytest.raises(ValueError, match="等长"):
        calculate(ref, ref, ref[:-1])


def test_cached_audio_reader_rejects_changed_payload(tmp_path):
    path = tmp_path / "reference.wav"
    sf.write(path, np.zeros(1600), 16000, subtype="FLOAT")
    digest = nsq._sha(path)
    sf.write(path, np.ones(1600) * 0.1, 16000, subtype="FLOAT")
    with pytest.raises(ValueError, match="哈希"):
        nsq._verified_audio(path, digest)


def test_api_preserves_identity_errors_and_missing_cases(monkeypatch):
    app = FastAPI()
    app.include_router(api.router)
    client = TestClient(app)
    url = "/api/alignment/acceptance/scene-condition-batches/batch/cases/case/ns-quality-metrics"
    monkeypatch.setattr(nsq, "case_ns_metrics", lambda *_: {"schema_version": nsq.SCHEMA, "status": "complete"})
    assert client.get(url).json()["schema_version"] == nsq.SCHEMA

    def missing(*_):
        raise FileNotFoundError("该组 TX 尚未完成")

    monkeypatch.setattr(nsq, "case_ns_metrics", missing)
    assert client.get(url).status_code == 404

    def invalid(*_):
        raise api.ArtifactIntegrityError("音频 SHA 不一致")

    monkeypatch.setattr(nsq, "case_ns_metrics", invalid)
    assert client.get(url).status_code == 409
