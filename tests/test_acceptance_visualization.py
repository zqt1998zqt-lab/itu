from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from server.acoustic import acceptance_visualization, service


@pytest.fixture
def visual_environment(tmp_path, monkeypatch):
    build = tmp_path / "build"
    assets = tmp_path / "assets"
    assets.mkdir()
    monkeypatch.setattr(service, "BUILD", build)
    return assets, build


def _write_wav(
    path: Path,
    samples: np.ndarray,
    rate: int = 16_000,
    *,
    subtype: str = "PCM_16",
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples, rate, format="WAV", subtype=subtype)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _i16(value: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(value, validate=True), dtype="<i2")


def test_visualization_shape_base64_lengths_and_parameters(visual_environment):
    assets, _ = visual_environment
    source = assets / "audio" / "tone.wav"
    rate = 16_000
    time = np.arange(rate // 2, dtype=np.float64) / rate
    digest = _write_wav(source, 0.5 * np.sin(2 * np.pi * 1_000 * time), rate)

    value = acceptance_visualization.load_or_build_visualization(source, digest)

    assert set(value) == {
        "schema",
        "source_sha256",
        "waveform_bins",
        "mel_bands",
        "mel_frames",
        "mel_fft_size",
        "mel_hop_ms",
        "mel_dynamic_range_db",
        "mel_db_offset_db",
        "mel_db_step_db",
        "duration_s",
        "sample_rate",
        "channels",
        "peak",
        "wave_scale",
        "wave_min_i16",
        "wave_max_i16",
        "mel_i16",
        "mel_ceiling_db",
    }
    assert value["schema"] == acceptance_visualization.SCHEMA
    assert value["source_sha256"] == digest
    assert value["waveform_bins"] == 720
    assert value["mel_bands"] == 40
    assert value["mel_frames"] == 240
    assert value["mel_fft_size"] == 512
    assert value["mel_hop_ms"] == 10
    assert value["mel_dynamic_range_db"] == 80
    assert value["mel_db_offset_db"] == -200.0
    assert value["mel_db_step_db"] == 0.1
    assert value["sample_rate"] == rate
    assert value["channels"] == 1
    assert value["duration_s"] == pytest.approx(0.5)
    assert value["peak"] == pytest.approx(0.5, abs=1 / 32768)
    assert value["wave_scale"] == pytest.approx(1.0)
    assert len(base64.b64decode(value["wave_min_i16"])) == 720 * 2
    assert len(base64.b64decode(value["wave_max_i16"])) == 720 * 2
    assert len(base64.b64decode(value["mel_i16"])) == 40 * 240 * 2
    assert _i16(value["wave_min_i16"]).shape == (720,)
    assert _i16(value["wave_max_i16"]).shape == (720,)
    assert _i16(value["mel_i16"]).shape == (40 * 240,)
    assert value["mel_ceiling_db"] > -200.0


def test_cache_is_reused_without_reading_source_again(
    visual_environment,
    monkeypatch,
):
    assets, _ = visual_environment
    source = assets / "tone.wav"
    digest = _write_wav(source, np.full(1_600, 0.25), 16_000)
    first = acceptance_visualization.load_or_build_visualization(source, digest)
    cache_path = acceptance_visualization.cache_root() / f"{digest}.json"
    cached_bytes = cache_path.read_bytes()

    def must_not_compute(*_args, **_kwargs):
        raise AssertionError("cache miss")

    monkeypatch.setattr(
        acceptance_visualization,
        "_compute_visualization",
        must_not_compute,
    )
    second = acceptance_visualization.load_or_build_visualization(source, digest)

    assert second == first
    assert cache_path.read_bytes() == cached_bytes


@pytest.mark.parametrize(
    "mutation",
    ["schema", "field", "base64", "parameter"],
)
def test_corrupt_cache_fails_closed_without_overwrite(
    visual_environment,
    mutation,
):
    assets, _ = visual_environment
    source = assets / "tone.wav"
    digest = _write_wav(source, np.full(1_600, 0.125), 16_000)
    acceptance_visualization.load_or_build_visualization(source, digest)
    cache_path = acceptance_visualization.cache_root() / f"{digest}.json"
    value = json.loads(cache_path.read_text(encoding="utf-8"))
    if mutation == "schema":
        value["schema"] = "other-schema"
    elif mutation == "field":
        value.pop("mel_ceiling_db")
    elif mutation == "base64":
        value["wave_min_i16"] = ""
    else:
        value["mel_fft_size"] = 256
    corrupt = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    cache_path.write_bytes(corrupt)

    with pytest.raises(acceptance_visualization.VisualizationCacheError):
        acceptance_visualization.load_or_build_visualization(source, digest)

    assert cache_path.read_bytes() == corrupt


def test_silence_keeps_absolute_sentinel_and_zero_waveform(visual_environment):
    assets, _ = visual_environment
    source = assets / "silence.wav"
    digest = _write_wav(source, np.zeros(1_600, dtype=np.float64), 16_000)

    value = acceptance_visualization.load_or_build_visualization(source, digest)

    assert value["peak"] == 0.0
    assert value["wave_scale"] == 1.0
    assert value["mel_ceiling_db"] == -200.0
    assert np.all(_i16(value["wave_min_i16"]) == 0)
    assert np.all(_i16(value["wave_max_i16"]) == 0)
    assert np.all(_i16(value["mel_i16"]) == 0)


def test_identical_stereo_channels_match_mono(visual_environment):
    assets, _ = visual_environment
    rate = 16_000
    time = np.arange(rate // 4, dtype=np.float64) / rate
    mono = 0.4 * np.sin(2 * np.pi * 440 * time)
    stereo = np.column_stack((mono, mono))
    mono_path = assets / "mono.wav"
    stereo_path = assets / "stereo.wav"
    mono_digest = _write_wav(mono_path, mono, rate)
    stereo_digest = _write_wav(stereo_path, stereo, rate)

    mono_value = acceptance_visualization.load_or_build_visualization(
        mono_path,
        mono_digest,
    )
    stereo_value = acceptance_visualization.load_or_build_visualization(
        stereo_path,
        stereo_digest,
    )

    assert mono_value["channels"] == 1
    assert stereo_value["channels"] == 2
    assert stereo_value["wave_min_i16"] == mono_value["wave_min_i16"]
    assert stereo_value["wave_max_i16"] == mono_value["wave_max_i16"]
    assert stereo_value["mel_i16"] == mono_value["mel_i16"]
    assert stereo_value["mel_ceiling_db"] == pytest.approx(
        mono_value["mel_ceiling_db"]
    )


def test_stereo_waveform_scans_channels_and_mel_averages_equal_weight(
    visual_environment,
):
    assets, _ = visual_environment
    source = assets / "antiphase.wav"
    stereo = np.array(
        [
            [0.8, -0.8],
            [-0.8, 0.8],
        ],
        dtype=np.float64,
    )
    digest = _write_wav(source, stereo, 16_000, subtype="FLOAT")

    value = acceptance_visualization.load_or_build_visualization(source, digest)

    minimum = _i16(value["wave_min_i16"])
    maximum = _i16(value["wave_max_i16"])
    assert minimum[0] < 0
    assert maximum[0] > 0
    assert value["peak"] == pytest.approx(0.8, abs=1 / 32768)
    assert value["mel_ceiling_db"] == -200.0
    assert np.all(_i16(value["mel_i16"]) == 0)


def test_precompute_counts_unique_assets_and_reuses_cache(
    visual_environment,
    monkeypatch,
):
    assets, _ = visual_environment
    first = assets / "first.wav"
    second = assets / "nested" / "second.wav"
    first_digest = _write_wav(first, np.full(800, 0.1), 16_000)
    second_digest = _write_wav(second, np.full(800, -0.1), 16_000)
    index = {
        "first.wav": {"sha256": first_digest},
        "nested/second.wav": {"sha256": second_digest},
    }

    assert acceptance_visualization.precompute_visualizations(assets, index) == 2
    assert len(list(acceptance_visualization.cache_root().glob("*.json"))) == 2

    def must_not_compute(*_args, **_kwargs):
        raise AssertionError("cache miss")

    monkeypatch.setattr(
        acceptance_visualization,
        "_compute_visualization",
        must_not_compute,
    )
    assert acceptance_visualization.precompute_visualizations(assets, index) == 2


def test_source_hash_mismatch_fails_before_cache_creation(visual_environment):
    assets, _ = visual_environment
    source = assets / "tone.wav"
    _write_wav(source, np.full(800, 0.1), 16_000)

    with pytest.raises(
        acceptance_visualization.VisualizationCacheError,
        match="SHA-256 differs",
    ):
        acceptance_visualization.load_or_build_visualization(
            source,
            "0" * 64,
        )

    assert not acceptance_visualization.cache_root().exists()
