"""Guard against accepting silence or ignoring double-talk and quantization bounds."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "cpp/tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("chain_matrix", TOOLS / "run_enhancement_chain_matrix.py")
chain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chain)


def measurement():
    return {
        "speech_windows": {"near_only": {"sisdr_db": 10., "stoi": .9, "projection_gain_db": 0.}, "double_talk": {"sisdr_db": 6., "stoi": .85, "projection_gain_db": 0.}},
        "noise_attenuation_db": 2., "far_attenuation_db": 3., "peak_dbfs": -2., "clipping_percent": 0.,
    }


def test_valid_joint_chain_and_far_only_negative_control():
    control = measurement()
    actual = deepcopy(control)
    actual["far_attenuation_db"] += 6
    assert chain.gate(actual, control)["pass"]
    result = chain.gate(control, control)
    assert not result["pass"]
    assert result["failures"] == ["far_attenuation"]


def test_muting_cannot_pass_by_reducing_noise_and_echo_power():
    control = measurement()
    actual = deepcopy(control)
    actual["noise_attenuation_db"] = 80.
    actual["far_attenuation_db"] = 80.
    for values in actual["speech_windows"].values():
        values.update(sisdr_db=None, stoi=None, projection_gain_db=None)
    result = chain.gate(actual, control)
    assert not result["pass"]
    assert result["failures"] == ["near_only:invalid_speech", "double_talk:invalid_speech"]


def test_double_talk_damage_fails_even_when_primary_speech_is_unchanged():
    control = measurement()
    actual = deepcopy(control)
    actual["far_attenuation_db"] += 10
    actual["speech_windows"]["double_talk"]["sisdr_db"] -= .2
    assert chain.gate(actual, control)["failures"] == ["double_talk:sisdr"]


def test_pcm16_digital_zero_uses_a_finite_quantization_bound():
    before = np.full(320, .001)
    value, bounded = chain.attenuation(before, np.zeros_like(before))
    assert bounded
    assert np.isclose(value, 20 * np.log10(.001 / (.5 / 32768)))
    measured, bounded = chain.attenuation(before, before / 2)
    assert not bounded
    assert np.isclose(measured, 20 * np.log10(2))
