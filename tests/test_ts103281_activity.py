from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pytest

from server.quality import ts103281_activity as activity


PROJECT = Path(__file__).resolve().parents[1]
SOURCES = PROJECT / "build/model-a-20261105/activity-sources"


def generated_signal(
    rate: int,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    seconds = 2.5
    time = np.arange(round(rate * seconds), dtype=np.float64) / rate
    carrier = 0.05 * np.sin(2.0 * np.pi * 440.0 * time)
    envelope = 0.7 + 0.3 * np.sin(2.0 * np.pi * 2.3 * time)
    signal = carrier * envelope
    signal[: round(rate * 0.12)] = 0.0
    signal[round(rate * 0.65) : round(rate * 0.90)] = 0.0
    signal[round(rate * 1.60) : round(rate * 1.82)] = 0.0
    rng = np.random.default_rng(103281 + rate)
    signal += rng.normal(0.0, 0.0002, signal.size)
    return (scale * signal).astype(np.float32)


def run_official_oracle(
    executable: Path,
    samples: np.ndarray,
    rate: int,
) -> float:
    payload = np.asarray(samples, dtype="<f4").tobytes()
    process = subprocess.run(
        [str(executable), str(rate)],
        input=payload,
        capture_output=True,
        check=True,
    )
    return float(process.stdout.decode("ascii").strip())


@pytest.fixture(scope="session")
def official_oracle(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("official-p56-oracle")
    executable = directory / "p56_oracle"
    subprocess.run(
        [
            "gcc",
            "-O2",
            "-std=c99",
            "-Wall",
            "-Wextra",
            "-o",
            str(executable),
            str(SOURCES / "p56_oracle.c"),
            str(SOURCES / "sv-p56.c"),
            "-lm",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return executable


def framed_signal(
    levels_db: list[float],
    *,
    rate: int = 1_000,
) -> np.ndarray:
    frame_samples = int(round(rate * activity.FRAME_SECONDS))
    amplitudes = np.power(10.0, np.asarray(levels_db) / 20.0)
    return np.repeat(amplitudes, frame_samples).astype(np.float64)


def test_downloaded_official_sources_and_license_are_pinned():
    source = SOURCES / "sv-p56.c"
    header = SOURCES / "sv-p56.h"
    license_text = SOURCES / "LICENSE.md"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == (
        activity.STL_SOURCE_SHA256
    )
    assert hashlib.sha256(header.read_bytes()).hexdigest() == (
        activity.STL_HEADER_SHA256
    )
    assert hashlib.sha256(license_text.read_bytes()).hexdigest() == (
        activity.STL_LICENSE_SHA256
    )
    text = license_text.read_text(encoding="utf-8")
    assert "ITU-T SOFTWARE TOOLS' GENERAL PUBLIC LICENSE" in text
    assert "g160app2.c" not in source.read_text(encoding="utf-8")


def test_python_method_b_matches_official_c_oracle_at_three_rates(
    official_oracle: Path,
):
    errors = []
    for rate in (8_000, 16_000, 48_000):
        samples = generated_signal(rate)
        python_level = activity.active_speech_level(samples, rate)
        official_level = run_official_oracle(
            official_oracle,
            samples,
            rate,
        )
        errors.append(abs(python_level - official_level))
        assert python_level == pytest.approx(
            official_level,
            abs=1e-9,
        )
    assert max(errors) < 1e-9


def test_six_db_amplitude_scaling_raises_asl_by_six_db():
    rate = 16_000
    original = generated_signal(rate)
    doubled = (2.0 * original).astype(np.float32)

    original_level = activity.active_speech_level(original, rate)
    doubled_level = activity.active_speech_level(doubled, rate)

    assert doubled_level - original_level == pytest.approx(
        20.0 * np.log10(2.0),
        abs=1e-9,
    )


def test_silence_and_insufficient_activity_raise_value_error():
    rate = 16_000
    with pytest.raises(ValueError, match="zero input"):
        activity.active_speech_level(np.zeros(rate, dtype=np.float64), rate)
    with pytest.raises(ValueError, match="shorter"):
        activity.classify_activity(
            np.zeros(rate // 1_000, dtype=np.float64),
            rate,
        )
    with pytest.raises(ValueError, match="undefined for zero input"):
        activity.classify_activity(np.zeros(rate, dtype=np.float64), rate)


def test_thresholds_are_strict_and_masks_separate_pause_from_ulmh(monkeypatch):
    monkeypatch.setattr(
        activity,
        "active_speech_level",
        lambda samples, rate: 0.0,
    )
    levels = [
        -0.5,
        -9.5,
        -15.5,
        -34.5,
        -1.5,
        -10.5,
        -16.5,
        -35.5,
    ] + [-60.0] * 40
    labels, diagnostics = activity.classify_activity(
        framed_signal(levels),
        rate=1_000,
    )

    assert labels[:8].tolist() == [
        "H",
        "M",
        "L",
        "U",
        "M",
        "L",
        "U",
        "S",
    ]
    assert np.all(labels[8:] == "S")
    assert diagnostics["frame_ms"] == 10.0
    assert diagnostics["asl_db"] == 0.0
    assert diagnostics["active_mask"].tolist() == (
        [True] * 7 + [False] * 41
    )
    assert diagnostics["ulmh_mask"].tolist() == (
        [True] * 7 + [False] * 41
    )


def test_only_internal_short_silence_runs_become_pause(monkeypatch):
    monkeypatch.setattr(
        activity,
        "active_speech_level",
        lambda samples, rate: 0.0,
    )
    # Use explicit -60 dB frames for silence while retaining an active frame
    # on both sides of the short gap.
    levels = [-60.0, -60.0, 0.0] + [-60.0] * 39 + [0.0, -60.0, -60.0]
    labels, diagnostics = activity.classify_activity(
        framed_signal(levels),
        rate=1_000,
    )

    assert labels[:2].tolist() == ["S", "S"]
    assert np.all(labels[3:42] == "P")
    assert labels[42] == "H"
    assert labels[-2:].tolist() == ["S", "S"]
    assert diagnostics["counts"] == {"H": 2, "P": 39, "S": 4}
    assert diagnostics["active_frames"] == 41
    assert np.count_nonzero(diagnostics["active_mask"]) == 41
    assert np.count_nonzero(diagnostics["ulmh_mask"]) == 2
    assert not np.any(
        diagnostics["ulmh_mask"] & (labels == "P")
    )


def test_exactly_40_internal_silence_frames_remain_s(monkeypatch):
    monkeypatch.setattr(
        activity,
        "active_speech_level",
        lambda samples, rate: 0.0,
    )
    levels = [0.0] + [-60.0] * 40 + [0.0]
    labels, diagnostics = activity.classify_activity(
        framed_signal(levels),
        rate=1_000,
    )

    assert labels[0] == "H"
    assert np.all(labels[1:41] == "S")
    assert labels[41] == "H"
    assert diagnostics["counts"] == {"H": 2, "S": 40}


def test_partial_tail_is_reported_and_not_classified(monkeypatch):
    monkeypatch.setattr(
        activity,
        "active_speech_level",
        lambda samples, rate: 0.0,
    )
    frame_samples = int(round(1_000 * activity.FRAME_SECONDS))
    samples = np.concatenate(
        [
            np.full(frame_samples, 0.1),
            np.full(frame_samples // 2, 0.1),
        ]
    )
    samples[:frame_samples] = 1.0
    labels, diagnostics = activity.classify_activity(samples, rate=1_000)

    assert labels.tolist() == ["H"]
    assert diagnostics["used_samples"] == frame_samples
    assert diagnostics["discarded_samples"] == frame_samples // 2


@pytest.mark.parametrize(
    ("samples", "rate"),
    [
        (np.zeros((2, 10)), 1_000),
        (np.array([0.1, np.nan]), 1_000),
        (np.array([0.1, 0.2]), 0),
        (np.array([0.1, 0.2]), 1_000.5),
    ],
)
def test_invalid_inputs_raise_value_error(samples, rate):
    with pytest.raises(ValueError):
        activity.active_speech_level(samples, rate)
