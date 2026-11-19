"""Check the revised entry point against the real immutable generation contracts."""
from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]


def _generator():
    path = PROJECT / "scripts/create-revised-acoustic-datasets.py"
    spec = importlib.util.spec_from_file_location("itu_revised_generator_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sources(rate):
    clock = np.arange(rate * 12) / rate
    near = 0.08 * np.sin(2 * np.pi * 517 * clock)
    far = 0.07 * np.sin(2 * np.pi * 683 * clock)
    if rate == 48_000:
        near += 0.007 * np.sin(2 * np.pi * 10_100 * clock)
        far += 0.006 * np.sin(2 * np.pi * 11_300 * clock)
    noise = np.random.default_rng(17).normal(size=len(clock))
    return near, far, noise


@pytest.mark.parametrize("mode", ("NB-wired", "WB-wireless", "SWB-wired"))
def test_scene_revision_covers_both_ser_signs_and_pcm_mixture(mode):
    entry = _generator()
    generator = entry.configure_scene("sce3", mode)
    legacy_id = hashlib.sha256(f"background-noise-conversation-v2:sce3:{mode}".encode()).hexdigest()[:32]
    assert generator.DATASET_ID != legacy_id
    plan = generator.case_plan()
    for conversation_mode in ("double_talk", "transition"):
        levels = [row["ser_db"] for row in plan if row["mode"] == conversation_mode]
        assert min(levels) == -10 and max(levels) == 10
    near, far, noise = _sources(generator.RATE)
    signals, design = generator.render_signals(
        near, far, noise, generator.RATE,
        direction="send", kind="double_talk", bandwidth=mode.split("-")[0],
        snr_db=6, ser_db=10, seed=19, volume_gain_db=0, input_level_db=0,
    )
    assert design["near_to_echo_ser_db_measured"] == pytest.approx(-10, abs=1e-8)
    assert design["transport"]["codec_conformance"] is False
    decoded = {role: generator.encode(audio, generator.RATE)[1] for role, audio in signals.items()}
    generator._verify_mixture_with_pcm_bound(signals, decoded, design)
    generator._assert_pcm_activity("double_talk", decoded, design)
    generator._post_pcm_measurements("double_talk", decoded, design)


@pytest.mark.parametrize("mode", ("NB-bt", "WB-wired", "SWB-wireless"))
def test_stress_short_overlap_recalibration_preserves_real_store_contract(mode):
    generator = _generator().configure_stress("double_talk", mode)
    near, far, noise = _sources(generator.RATE)
    # A quiet syllable exactly at the 500 ms overlap used to create accidental
    # extreme local SNR despite a benign full-utterance SNR target.
    selection = slice(round(6.5 * generator.RATE), 7 * generator.RATE)
    near[selection] *= 0.08
    signals, design = generator.STRESS.render_stress_signals(
        near, far, noise, generator.RATE,
        scope="double_talk", category="short_overlap",
        volume_gain_db=0, seed=23, case_index=1, bandwidth=mode.split("-")[0],
    )
    ratio = np.sqrt(np.mean(signals["reference"][selection] ** 2)) / np.sqrt(np.mean(signals["noise"][selection] ** 2))
    assert 20 * np.log10(ratio) == pytest.approx(design["noise_snr_db_target"], abs=1e-7)
    decoded = {role: generator.CONVERSATION.encode(audio, generator.RATE)[1] for role, audio in signals.items()}
    generator.STRESS.verify_stress_mixture(signals, decoded, design)
    generator.STRESS.assert_stress_activity(decoded, design, scope="double_talk", category="short_overlap")


def test_revision_sources_are_bound_without_editing_legacy_generators():
    generator = _generator().configure_scene("sce1", "WB-wired")
    expected = {
        "scripts/create-revised-acoustic-datasets.py",
        "scripts/dataset_revision_sources.py",
        "scripts/dataset_revision_signal_design.py",
        "server/acoustic/dataset_revisions.py",
    }
    assert expected <= set(generator._source_hash_snapshot())
    assert generator.SCHEMA_VERSION == "scene-condition-preview-v2"
