"""Non-vacuous checks for the direct-GTCRN comparison and its shared gates."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'cpp/tools'))
from evaluate_gtcrn_direct import score_aligned
from verify_gtcrn_direct import pcm16


def fixture():
    rate = 16000; t = np.arange(4*rate)/rate
    clean = np.zeros(len(t)); clean[:2*rate] = (.1*np.sin(2*np.pi*183*t[:2*rate])
        +.035*np.sin(2*np.pi*529*t[:2*rate]))*(.8+.2*np.cos(2*np.pi*3*t[:2*rate]))
    noise = np.random.default_rng(41).normal(0, .01, len(t))
    before = clean+noise; after = clean+noise*.1
    scene = dict(sample_rate=rate, windows=dict(near_only=(.1,1.9), noise_only=(2.1,3.9)), near_schedule=[(0,2)])
    return clean, before, after, scene


def test_canonical_pcm_matches_cpp_ties_and_saturation():
    x = np.array([-.5/32768, .5/32768, -1.5/32768, 1.5/32768, -2., 2.], dtype=np.float32)
    np.testing.assert_array_equal(pcm16(x)*32768, [-1,1,-2,2,-32768,32767])


def test_same_gates_accept_protected_speech_and_attenuated_noise():
    clean, before, after, scene = fixture()
    row = score_aligned(clean, before, after, scene, 'vehicle', 0)
    assert row['passed'], row['failures']
    assert 19 < row['mean_noise_attenuation_db'] < 21


def test_negative_controls_reject_passthrough_silence_and_clipping():
    clean, before, after, scene = fixture()
    bypass = score_aligned(clean, before, before, scene, 'vehicle', 0)
    assert not bypass['passed'] and 'noise_only:attenuation' in bypass['failures']
    # The inherited metric contract marks an entirely silent speech output
    # unscorable and aborts the case. It must never count as a passing result.
    with pytest.raises(ValueError, match='no scored speech segment'):
        score_aligned(clean, before, np.zeros_like(before), scene, 'vehicle', 0)
    clipping = score_aligned(clean, before, after, scene, 'vehicle', 1)
    assert not clipping['passed'] and 'clipping' in clipping['failures']
