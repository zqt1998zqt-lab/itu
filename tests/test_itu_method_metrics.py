import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cpp/tools"))
from itu_method_metrics import noise_continuity, cng_difference, double_talk_attenuation


@pytest.mark.parametrize("rate", [16000, 24000, 48000])
def test_noise_pumping_detected_and_unavailable_never_passes(rate):
    x = np.random.default_rng(51).normal(0, .01, rate * 8)
    good = noise_continuity(x, x.copy(), rate, [[5, 8]])
    assert good["passed"] and good["max_absolute_variation_db"] == 0
    bad = x.copy(); bad[rate*6:] *= .1
    result = noise_continuity(x, bad, rate, [[5, 8]])
    assert not result["passed"] and result["max_absolute_variation_db"] > 19
    assert noise_continuity(x*0, x*0, rate, [[5, 8]])["passed"] is None
    with pytest.raises(ValueError): noise_continuity(x, x, rate, [[4, 8]])


def test_double_talk_mute_is_not_full_duplex():
    x = np.sin(np.arange(16000)*.31)*.02
    assert double_talk_attenuation(x, x, x, 16000, [[0, 1]])["category"] == "1"
    bad = double_talk_attenuation(x, x, x*.1, 16000, [[0, 1]])
    assert bad["category"] == "3" and not bad["passed"]
    assert double_talk_attenuation(x, x, x*0, 16000, [[0, 1]])["passed"] is None


def test_cng_scope_level_and_high_frequency_mask():
    x = np.random.default_rng(1).normal(0, .01, 48000*5)
    assert cng_difference(x, x, 48000, True)["passed"]
    assert not cng_difference(x, x*2, 48000, True)["passed"]
    assert not cng_difference(x, x*.1, 48000, True)["passed"]
    assert cng_difference(x, x, 48000, False)["passed"] is None
    assert cng_difference(x[:16000*5], x[:16000*5], 16000, True)["passed"] is None
