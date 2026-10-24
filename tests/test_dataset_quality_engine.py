from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from server.quality.dataset import aggregate_assets, inspect_asset, speech_windows
from server.quality.evaluators import Evaluator
from server.quality.metrics import sha


class Probe(Evaluator):
    name = "probe"
    minimum_seconds = 9.01

    def __init__(self, path):
        super().__init__(path)
        self.identity = {"name": "fixture"}
        self.calls = 0

    def predict(self, x):
        self.calls += 1
        return {"MOS": 3.}, {}


def asset(tmp_path, x, rate=16000, role="target_reference", intervals=None):
    path = tmp_path / "source.wav"
    sf.write(path, x, rate, subtype="FLOAT")
    return dict(asset_id="fixture", path=str(path), sha256=sha(path.read_bytes()), sample_rate=rate,
                role=role, speech_intervals=intervals)


def test_true_schedule_cannot_be_joined_or_padded_for_mos(tmp_path):
    assert speech_windows([[1,10],[12,22]], 384000, 16000) == [(16000,160000),(192000,352000)]
    x = np.random.default_rng(5).normal(0, .03, 384000)
    probe = Probe(tmp_path / "cache")
    row = inspect_asset(asset(tmp_path, x, intervals=[[1,10],[12,22]]), [probe])
    scores = [w["perceptual"]["probe"] for w in row["channels"][0]["windows"]]
    assert not scores[0]["available"] and scores[0]["reason"] == "window_too_short"
    assert scores[1]["available"] and probe.calls == 1


def test_noise_and_empty_target_are_not_scored_as_speech(tmp_path):
    probe = Probe(tmp_path / "cache")
    row = inspect_asset(asset(tmp_path, np.zeros(160000), role="noise_source"), [probe])
    assert not row["channels"][0]["windows"] and probe.calls == 0
    row = inspect_asset(asset(tmp_path, np.zeros(160000), intervals=[]), [probe])
    assert not row["channels"][0]["windows"] and probe.calls == 0


@pytest.mark.parametrize("rate", [24000,48000])
def test_high_rate_physical_checks_without_implicit_src(tmp_path, rate):
    x = np.random.default_rng(5).normal(0, .03, rate * 10)
    probe = Probe(tmp_path / "cache")
    row = inspect_asset(asset(tmp_path, x, rate), [probe])
    assert row["sample_rate"] == rate and not row["policy"]["resampled"]
    assert row["channels"][0]["windows"][0]["perceptual"]["probe"]["reason"] == "unsupported_native_sample_rate"
    assert probe.calls == 0


def test_source_hash_negative_control_and_role_aware_low_frequency(tmp_path):
    x = .1 + .05 * np.sin(2 * np.pi * 30 * np.arange(160000) / 16000)
    spec = asset(tmp_path, x)
    row = inspect_asset(spec, [])
    flags = row["channels"][0]["review_flags"]
    assert "scheduled_dc_above_minus50_dbfs" in flags and "scheduled_sub60hz_above_5_percent" in flags
    spec["role"] = "noise_source"
    assert not inspect_asset(spec, [])["channels"][0]["review_flags"]
    spec["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA mismatch"):
        inspect_asset(spec, [])
    with pytest.raises(ValueError, match="overlapping"):
        speech_windows([[0,2],[1,3]], 48000, 16000)


def test_pcm_dedup_preserves_usage_counts_and_model_coverage(tmp_path):
    row = inspect_asset(asset(tmp_path, np.zeros(160000)), [Probe(tmp_path / "cache")])
    other = {**row, "asset_id": "copy"}
    stats = aggregate_assets([row,other])
    assert stats["unique_pcm_files"] == 1 and stats["role_rate_counts"] == {"target_reference:16000": 2}
    assert stats["exact_pcm_duplicate_groups"] == [["fixture","copy"]]
    assert stats["coverage"]["target_reference:16000:probe"]["unavailable_reasons"] == {"silence_or_no_ac_excitation": 2}
