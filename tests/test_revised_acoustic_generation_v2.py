from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from server.acoustic import dataset_revisions as first


def _configure(mode):
    path = Path(__file__).resolve().parents[1] / "scripts/create-revised-acoustic-datasets-v2.py"
    spec = importlib.util.spec_from_file_location("itu_r2_generation_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generator = module.configure("stress", "double_talk", mode)
    assert first.DATASET_REVISION == "measurement-20261117-r1"
    return generator


def _render(generator, *, quiet_far_overlap=False):
    rate = generator.RATE
    clock = np.arange(rate * 12) / rate
    near = 0.08 * np.sin(2 * np.pi * 467 * clock)
    far = 0.07 * np.sin(2 * np.pi * 721 * clock)
    if quiet_far_overlap:
        far[round(6.5 * rate):round(7.0 * rate)] *= 1e-3
    noise = np.random.default_rng(713).normal(size=len(clock))
    return generator.STRESS.render_stress_signals(
        near, far, noise, rate, scope="double_talk", category="short_overlap",
        volume_gain_db=6, seed=419, case_index=5, bandwidth=generator.BANDWIDTH,
    )


@pytest.mark.parametrize("mode", ("WB-wired", "SWB-wired"))
def test_r2_delivered_pcm_guard_measures_the_actual_noise_window(mode):
    generator = _configure(mode)
    _, design = _render(generator)
    assert design["noise_snr_db_error_post_pcm_overlap"] < 0.1
    assert design["noise_snr_db_measured_post_pcm_overlap"] == pytest.approx(
        design["noise_snr_db_target"], abs=0.1,
    )
    sources = generator._source_hash_snapshot()
    assert "scripts/dataset_revision_observation.py" in sources
    assert "scripts/create-revised-acoustic-datasets-v2.py" in sources


def test_r2_rejects_a_float_correct_but_pcm16_unresolvable_overlap():
    generator = _configure("SWB-wired")
    with pytest.raises(ValueError, match="PCM16"):
        _render(generator, quiet_far_overlap=True)
