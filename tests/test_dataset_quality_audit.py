"""The input audit must catch corrupt inputs independently of DSP scores."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cpp/tools"))
import audit_dataset_quality as audit
import run_audio_eval as ev


def fixture(tmp_path):
    rng = np.random.default_rng(567)
    case = dict(id="qa", family="ns", pair=0, near_speaker="1", far_speaker="2", seed=1,
                near_dbfs=-26, noise="white", snr_db=10, ser_db=0, rt60_s=.15,
                echo_delay_ms=80, stress="none", profile="expanded-v3", mics=1)
    scene = ev.make_case(case, {"1":rng.normal(size=384000), "2":rng.normal(size=384000)}, {},
                         tmp_path / "cases/qa", {})
    return scene


def test_valid_input_and_corrupt_hash_negative_control(tmp_path):
    scene = fixture(tmp_path)
    actual = audit.audit_scene((tmp_path, scene))
    assert actual["errors"] == []
    assert actual["max_reconstruction_error_lsb"] <= 2
    assert actual["snr_error_db"] < .01
    path = tmp_path / "cases/qa/before.wav"
    pcm = ev.read(path)
    pcm[50000] += .01
    ev.write(path, pcm)
    broken = audit.audit_scene((tmp_path, scene))
    assert "identity:before.wav" in broken["errors"]
    assert "mixture_reconstruction_over_2_lsb" in broken["errors"]


def test_wrong_window_negative_control(tmp_path):
    scene = fixture(tmp_path)
    scene["windows"]["noise_only"] = [2, 3]
    assert "target_leaks_into:noise_only" in audit.audit_scene((tmp_path, scene))["errors"]


def test_bad_snr_label_negative_control(tmp_path):
    scene = fixture(tmp_path)
    scene["snr_db"] = 0
    assert "measured_snr_outside_0.05_db" in audit.audit_scene((tmp_path, scene))["errors"]


def test_energy_proxy_does_not_hide_silence():
    data = np.zeros((16000, 1))
    result = audit.audio_stats(data, 16000)
    assert result["low_energy_20ms_percent"] == 100
    assert result["zero_lsb_percent"] == 100
    with pytest.raises(ValueError, match="nonempty finite"):
        audit.audio_stats(data[:0], 16000)


def test_target_spectrum_detects_dc_and_subsonic_content():
    t = np.arange(16000) / 16000
    data = (.01 + .03 * np.sin(2 * np.pi * 10 * t) + .03 * np.sin(2 * np.pi * 400 * t))[:, None]
    result = audit.target_spectrum(data, 16000)
    assert abs(result["ac_below_60hz_percent"] - 50) < 1e-8
    assert abs(result["dc_dbfs"] + 40) < 1e-8
