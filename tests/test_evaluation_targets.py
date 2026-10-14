import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cpp/tools"))
from evaluation_targets import calibrated_beam_target


@pytest.mark.parametrize("rate,boundary", [(16000, 5), (24000, 14), (48000, 13)])
def test_fixed_geometry_origin_and_old_target_negative_control(rate, boundary):
    # The four channels contain identical 2 kHz waves; known fractional delay
    # creates a predictable complex transfer. This catches silent reuse of the
    # old 16 kHz ceil timestamp in either higher-rate path.
    t = np.arange(16000) / 16000
    x = np.tile(np.sin(2 * np.pi * 2000 * t)[:, None], (1, 4))
    delays = np.array([4.31839738, 2.89806713, 1.45775238, 0.])
    y, identity = calibrated_beam_target(x, delays, rate)
    assert identity["target_origin_core_samples"] == boundary
    assert len(y) == len(x) and np.isfinite(y).all()
    if rate != 16000:
        old, _ = calibrated_beam_target(x, delays, 16000)
        # It need only be observably different; the threshold is deliberately
        # below the measured 24 kHz difference so this remains a negative check
        # against silently reusing the legacy 16 kHz target.
        assert np.mean((old[100:-100] - y[100:-100]) ** 2) > 1e-4
    else:
        n = np.arange(len(x))
        known = np.mean([np.interp(n + 5 - d, n, x[:, ch], left=0, right=0)
                         for ch, d in enumerate(delays)], axis=0)
        np.testing.assert_allclose(y, known, atol=1e-14)


def test_target_rejects_missing_calibration():
    with pytest.raises(ValueError):
        calibrated_beam_target(np.zeros((160, 4)), [0], 48000)


def test_rnnoise_16k_uses_48k_geometry_and_io_rounding():
    t = np.arange(16000) / 16000
    x = np.tile(np.sin(2 * np.pi * 3000 * t)[:, None], (1, 4))
    delays = [4.31839738, 2.89806713, 1.45775238, 0.]
    y, identity = calibrated_beam_target(x, delays, 16000, core_rate=48000)
    old, _ = calibrated_beam_target(x, delays, 16000)
    assert identity["core_rate"] == 48000
    assert identity["beam_bound_core_samples"] == 13
    assert identity["target_origin_core_samples"] == 15
    assert len(y) == len(x) and np.isfinite(y).all()
    # The negative control reuses the 16k interpolation. Both origins equal
    # five source samples here, so only the interpolation response differs.
    assert np.mean((y[100:-100] - old[100:-100]) ** 2) > 1e-6
