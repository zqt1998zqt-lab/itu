"""Simulated playback time must predict the capture echo without target feedback."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cpp/tools"))
from run_simulated_clock import simulate_echo


@pytest.mark.parametrize("rate", [16000, 24000, 48000])
def test_independent_playback_clock_moves_echo_in_correct_direction(rate):
    source = np.zeros(4*rate)
    source[3*rate] = .5
    slow, nominal, fast = [simulate_echo(source, rate, ppm) for ppm in (-300, 0, 300)]
    assert np.count_nonzero(nominal) == 2
    assert nominal[round(3.080*rate)] == pytest.approx(.3)
    assert nominal[round(3.085*rate)] == pytest.approx(.09)
    assert np.argmax(fast) < np.argmax(nominal) < np.argmax(slow)
    assert np.max(fast) > .1 and np.max(slow) > .1
    assert not np.any(slow[:3*rate])
