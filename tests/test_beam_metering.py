"""Check beam-domain metering alignment and a deliberately corrupted tap."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'cpp/tools'))
from measure_frozen_repair_v5 import static_beam, verify_static_tap


def test_fractional_steering_and_single_delay_removal():
    capture = np.zeros((16, 2))
    capture[4, 0] = .8
    capture[6, 1] = .4
    beam, delay = static_beam(capture, [2.5, .5])
    # Both impulses land at 6.5 before removing the single ceil(2.5) delay.
    expected = np.zeros(16)
    expected[3:5] = .3
    assert delay == 3
    np.testing.assert_allclose(beam, expected, atol=1e-12)
    quantized = np.clip(np.floor(beam * 32768), -32768, 32767) / 32768
    assert verify_static_tap(capture, quantized, [2.5, .5])['verified']
    corrupted = quantized.copy()
    corrupted[3] += .001
    with pytest.raises(ValueError, match='BF tap differs'):
        verify_static_tap(capture, corrupted, [2.5, .5])
    with pytest.raises(ValueError, match='BF tap differs'):
        verify_static_tap(capture, quantized, [0., 0.])


def test_actual_steering_changes_the_clean_target():
    rng = np.random.default_rng(17)
    clean = rng.normal(0, .05, (1000, 4))
    nominal, _ = static_beam(clean, [1., 0., 2., 3.])
    actual, delay = static_beam(clean, [1., .8, 3.6, 5.4])
    assert delay == 6
    assert np.max(np.abs(actual - nominal)) > .01
    assert verify_static_tap(clean, actual, [1., .8, 3.6, 5.4])['verified']


def test_frozen_real_steering_case():
    """Use the existing immutable input and BF tap without executing DSP."""
    import json
    import soundfile as sf
    root = Path(__file__).resolve().parents[1]
    folder = root / 'build/controlled-native-full-20261017/cases/vehicle-16000-development-m2-10'
    capture, rate = sf.read(folder / 'before.wav', always_2d=True)
    tap, actual_rate = sf.read(folder / 'runs/full/result.bf.wav')
    report = json.loads((folder / 'runs/full/result.json').read_text())
    argv = json.loads((folder / 'runs/full/invocation.json').read_text())['argv']
    delays = [float(v) for v in argv[argv.index('--beam-delays') + 1].split(',')]
    delay = report['bf_delay_samples']
    assert rate == actual_rate == 16000
    assert len(tap) == len(capture) + delay
    assert verify_static_tap(capture, tap[delay:], delays)['verified']
