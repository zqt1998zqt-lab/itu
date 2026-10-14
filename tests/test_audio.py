import io
import json

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from scipy import signal

from server.app import app
from server.audio import ROLES, demo_wav

client = TestClient(app)


def wav(values, rate=16000, format="WAV", subtype="FLOAT"):
    stream = io.BytesIO()
    sf.write(stream, values, rate, format=format, subtype=subtype)
    return stream.getvalue()


def request(blobs, options=None):
    return client.post("/api/analyze", files={role: (name, value, "audio/wav") for role, (name, value) in blobs.items()}, data={"options": json.dumps(options or {})})


def rows(result):
    return {m["id"]: m for m in result["metrics"]}


@pytest.fixture
def demo_files():
    return {role: (role + ".wav", demo_wav(role)) for role in ROLES}


@pytest.fixture
def config():
    return {"sample_rate": 16000, "speech": {"start": 0.4, "end": 3.1}, "noise": {"start": 5, "end": 6}, "far_end": {"start": 3.4, "end": 4.7}, "common_clock": True, "far_end_only": True}


@pytest.mark.parametrize("sample_rate", [16000, 24000, 48000])
def test_actual_multipart_known_delay_noise_and_echo(demo_files, config, sample_rate):
    config["sample_rate"] = sample_rate
    response = request(demo_files, config)
    assert response.status_code == 200
    result, m = response.json(), rows(response.json())
    assert len(result["files"]) == 4
    assert {f["duration"] for f in result["files"]} == {6.0}
    assert m["delay_before"]["value"] == m["delay_after"]["value"] == 80
    assert m["noise_attenuation"]["value"] == pytest.approx(20 * np.log10(4), abs=0.01)
    assert m["echo_attenuation"]["value"] == pytest.approx(20, abs=0.5)
    assert m["erle"]["value"] is None
    assert m["sisdr_improvement"]["value"] > 10
    assert m["stoi_after"]["value"] is None
    assert len(result["standards"]) == 17
    assert all(s["status"] == "unavailable" for s in result["standards"])
    a, b = [p["alignment"] for p in result["pairs"]]
    assert a["common_reference_start"] == b["common_reference_start"]
    assert a["common_reference_end"] == b["common_reference_end"]


def test_channel_metadata_and_real_full_scale_units():
    rate = 48000
    t = np.arange(rate) / rate
    sine = 0.5 * np.sin(2 * np.pi * 1000 * t)
    stereo = np.column_stack([sine, -sine])
    result = request({"after": ("stereo.wav", wav(stereo, rate))}, {"channels": {"after": 1}}).json()
    f, m = result["files"][0], rows(result)
    assert (f["channels"], f["selected_channel"], f["sample_rate"]) == (2, 2, 48000)
    assert m["rms_dbfs_after"]["value"] == pytest.approx(-9.0309, abs=0.0001)
    assert m["peak_dbfs_after"]["value"] == pytest.approx(-6.0206, abs=0.0001)
    assert f["spectrum"]["peak_hz"] == pytest.approx(1000, abs=12)
    assert m["reference_missing"]["value"] is None


def test_clipping_and_dc_are_measured_from_samples():
    values = np.ones(16000) * 0.1
    values[:160] = 1
    m = rows(request({"after": ("clip.wav", wav(values))}).json())
    assert m["clipping_percent_after"]["value"] == pytest.approx(1)
    assert m["clipping_percent_after"]["status"] == "warning"
    assert m["dc_offset_after"]["value"] == pytest.approx(0.109, abs=1e-8)


def test_silence_cannot_produce_delay_quality_or_finite_erle(config):
    blank = wav(np.zeros(96000))
    result = request({role: (role + ".wav", blank) for role in ROLES}, config).json()
    m = rows(result)
    for id in ["delay_after", "sisdr_after", "noise_attenuation", "erle"]:
        assert m[id]["value"] is None
    assert m["rms_dbfs_after"]["status"] == "warning"
    assert m["rms_dbfs_after"]["value"] == -150
    assert result["files"][0]["spectrum"]["peak_hz"] is None


def test_unrelated_recordings_do_not_get_quality_score(demo_files, config):
    demo_files["after"] = ("wrong.wav", wav(np.random.default_rng(3).normal(0, 0.1, 96000)))
    m = rows(request(demo_files, config).json())
    assert m["delay_after"]["value"] is None
    assert m["sisdr_after"]["value"] is None
    assert m["sisdr_improvement"]["value"] is None


def test_periodic_ambiguous_delay_is_refused():
    t = np.arange(32000) / 16000
    x = np.sin(2 * np.pi * 400 * t) * 0.1
    m = rows(request({"reference": ("ref.wav", wav(x)), "after": ("out.wav", wav(np.roll(x, 200)))}).json())
    assert m["delay_after"]["value"] is None
    assert "多个相关峰" in m["delay_after"]["note"]


def test_delay_at_search_boundary_is_not_accepted(demo_files, config):
    config["max_delay_ms"] = 80
    m = rows(request(demo_files, config).json())
    assert m["delay_after"]["value"] is None
    assert "搜索边界" in m["delay_after"]["note"]


@pytest.mark.parametrize("confirmation", ["common_clock", "far_end_only"])
def test_erle_requires_actual_far_end_conditions(demo_files, config, confirmation):
    config[confirmation] = False
    m = rows(request(demo_files, config).json())
    assert m["erle"]["value"] is None
    assert m["delay_after"]["value"] == 80


def test_noise_gain_reduction_cannot_hide_speech_damage(demo_files, config):
    values, rate = sf.read(io.BytesIO(demo_files["after"][1]))
    values[:int(3.1 * rate)] += np.random.default_rng(4).normal(0, 0.03, int(3.1 * rate))
    demo_files["after"] = ("damaged.wav", wav(values, rate))
    m = rows(request(demo_files, config).json())
    assert m["noise_attenuation"]["value"] > 10
    assert m["sisdr_improvement"]["value"] < -5


def test_different_rates_and_delays_share_reference_samples(demo_files, config):
    source, rate = sf.read(io.BytesIO(demo_files["reference"][1]))
    high_rate = signal.resample_poly(source, 3, 1)
    demo_files["reference"] = ("ref48k.wav", wav(high_rate, 48000))
    output, _ = sf.read(io.BytesIO(demo_files["after"][1]))
    demo_files["after"] = ("out120ms.wav", wav(np.r_[np.zeros(640), output[:-640]]))
    result = request(demo_files, config).json()
    m = rows(result)
    assert m["delay_after"]["value"] == pytest.approx(120, abs=0.1)
    assert m["delay_before"]["value"] == pytest.approx(80, abs=0.1)
    bounds = [(p["alignment"]["common_reference_start"], p["alignment"]["common_reference_end"]) for p in result["pairs"]]
    assert bounds[0] == bounds[1]
    assert bounds[0][1] == pytest.approx(2.98)
    assert result["files"][0]["sample_rate"] == 48000


def test_stoi_is_explicit_and_short_frames_are_unavailable(demo_files, config):
    config["speech_confirmed"] = True
    m = rows(request(demo_files, config).json())
    assert m["stoi_after"]["status"] == "ok"
    assert -1 <= m["stoi_after"]["value"] <= 1
    config["speech"] = {"start": 0.5, "end": 0.85}
    m = rows(request(demo_files, config).json())
    assert m["stoi_after"]["value"] is None
    assert "有效结果" in m["stoi_after"]["note"]


def test_stationary_snr_subtracts_noise_in_linear_domain(demo_files, config):
    config["stationary_noise"] = True
    result = request(demo_files, config).json()
    m = rows(result)
    data, rate = sf.read(io.BytesIO(demo_files["before"][1]))
    pn = np.mean(data[5 * rate:6 * rate] ** 2)
    pm = np.mean(data[round(.4 * rate):round(3.1 * rate)] ** 2)
    assert m["snr_before"]["value"] == pytest.approx(10 * np.log10((pm - pn) / pn), abs=0.001)


def test_invalid_noise_power_is_not_a_fake_snr():
    values = np.r_[np.ones(16000) * 0.01, np.ones(16000) * 0.1]
    m = rows(request({"after": ("inverse.wav", wav(values))}, {"speech": {"start": 0, "end": 1}, "noise": {"start": 1, "end": 2}, "stationary_noise": True}).json())
    assert m["snr_after"]["value"] is None
    assert "混合功率" in m["snr_after"]["note"]


def test_out_of_bounds_windows_preserve_single_file_results(demo_files):
    m = rows(request(demo_files, {"speech": {"start": 7, "end": 8}}).json())
    assert m["delay_after"]["value"] is None
    assert m["rms_dbfs_after"]["value"] is not None


@pytest.mark.parametrize("name,data", [("test.xlsx", b"not an audio"), ("broken.wav", b"not an audio"), ("empty.wav", b"")])
def test_non_audio_and_corrupt_files_are_rejected(name, data):
    assert request({"after": (name, data)}).status_code == 422


def test_missing_required_file_invalid_channel_and_configuration(demo_files):
    assert client.post("/api/analyze").status_code == 422
    assert request(demo_files, {"channels": {"after": 1}}).status_code == 422
    assert request(demo_files, {"speech": {"start": 3, "end": 2}}).status_code == 422


def test_lossless_flac_is_decoded_with_original_metadata():
    x = np.random.default_rng(9).normal(0, .01, 22050)
    result = request({"after": ("actual.flac", wav(x, 22050, "FLAC", "PCM_16"))})
    assert result.status_code == 200
    assert result.json()["files"][0]["sample_rate"] == 22050
    assert result.json()["files"][0]["format"] == "FLAC"


def test_negative_delay_preserves_sign_and_alignment():
    x = np.random.default_rng(11).normal(0, .1, 48000)
    reference = np.r_[np.zeros(800), x[:-800]]
    m = rows(request({"reference": ("late-reference.wav", wav(reference)), "after": ("earlier.wav", wav(x))}).json())
    assert m["delay_after"]["value"] == -50
    assert m["sisdr_after"]["value"] > 110


def test_erle_and_noise_allow_explicit_processing_delay_compensation(demo_files, config):
    output, rate = sf.read(io.BytesIO(demo_files["after"][1]))
    # Reserve 40 ms at the end so the requested noise interval remains in bounds.
    config["noise"]["end"] = 5.9
    demo_files["after"] = ("extra40ms.wav", wav(np.r_[np.zeros(640), output[:-640]], rate))
    config["after_offset_ms"] = 40
    m = rows(request(demo_files, config).json())
    assert m["delay_after"]["value"] == 120
    assert m["noise_attenuation"]["value"] == pytest.approx(12.0412, abs=.01)
    assert m["echo_attenuation"]["value"] == pytest.approx(20, abs=.5)


def test_filter_erle_requires_explicit_measurement_points(demo_files, config):
    assert rows(request(demo_files, config).json())["erle"]["value"] is None
    config["aec_only"] = True
    m = rows(request(demo_files, config).json())
    assert m["erle"]["value"] == m["echo_attenuation"]["value"]
    assert m["erle"]["status"] == "ok"
    config["far_end_only"] = False
    assert rows(request(demo_files, config).json())["erle"]["value"] is None


def test_low_nonzero_energy_is_not_called_digital_zero():
    x = np.random.default_rng(12).normal(0, 1e-7, 16000)
    result = request({"after": ("very-quiet.wav", wav(x))}).json()
    assert result["files"][0]["silent"] is False
    assert result["files"][0]["low_energy"] is True
    assert "数字零" not in rows(result)["rms_dbfs_after"]["note"]
    assert rows(result)["rms_dbfs_after"]["value"] == pytest.approx(-140, abs=.2)


def test_too_short_shared_reference_cannot_produce_improvement():
    rng = np.random.default_rng(13)
    reference = rng.normal(0, .1, 9600)
    before = np.r_[np.zeros(3040), reference[:-3040]] + rng.normal(0, .002, 9600)
    after = np.r_[reference[3040:], np.zeros(3040)] + rng.normal(0, .002, 9600)
    result = request({"reference": ("ref.wav", wav(reference)), "before": ("pre.wav", wav(before)), "after": ("post.wav", wav(after))}, {"common_clock": True}).json()
    m = rows(result)
    assert m["delay_before"]["value"] == 190
    assert m["delay_after"]["value"] == -190
    assert m["sisdr_before"]["status"] == m["sisdr_after"]["status"] == "ok"
    assert m["sisdr_improvement"]["value"] is None
    assert "共同参考" in m["sisdr_improvement"]["note"]
