#!/usr/bin/env python3
"""Protocol tests for the VCP stage-MOS runner; no ONNX inference is used."""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf


CPP_ROOT = Path(__file__).resolve().parents[1]
TOOLS = CPP_ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import ecnr_perceptual as perceptual  # noqa: E402
import vcp_stage_mos as stage_mos  # noqa: E402


def _fake_pyloudnorm():
    class Meter:
        def __init__(self, rate, filter_class, block_size):
            assert rate == stage_mos.SCORE_RATE
            assert filter_class == "K-weighting"
            assert block_size == stage_mos.LUFS_BLOCK_SIZE

        def integrated_loudness(self, samples):
            values = np.asarray(samples, dtype=np.float64)
            rms = float(np.sqrt(np.mean(values * values)))
            return 20.0 * math.log10(rms) if rms > 0.0 else -math.inf

    return SimpleNamespace(Meter=Meter)


class FakeScorer:
    """Return fixed, recognizably different SIG/BAK/OVRL values."""

    def __init__(self, mutate_path=None, mutate_on_call=None):
        self.preprocessing = {"model_sha256": stage_mos.MODEL_SHA}
        self.preprocessing_sha256 = "fake-preprocessing-for-protocol-tests"
        self.calls = []
        self.mutate_path = Path(mutate_path) if mutate_path is not None else None
        self.mutate_on_call = mutate_on_call

    def score_array(self, samples, sample_rate, *, kind, use_cache=True):
        values = np.asarray(samples, dtype=np.float64)
        self.calls.append(values.copy())
        if self.mutate_path is not None and len(self.calls) == self.mutate_on_call:
            with self.mutate_path.open("ab") as stream:
                stream.write(b"changed")
        return {
            "available": True,
            "reason": None,
            "model_sha256": stage_mos.MODEL_SHA,
            "raw": {"SIG": 2.0, "BAK": 3.0, "OVRL": 4.0},
            "poly": {"SIG": 2.25, "BAK": 3.25, "OVRL": 4.25},
            "hops": [],
            "short_repeat": False,
            "doubling_count": 0,
            "resampled_original_samples": int(len(values)),
            "source_sample_rate": int(sample_rate),
            "role": kind,
            "use_cache_requested": use_cache,
        }


class SilenceAwareFakeScorer(FakeScorer):
    def score_array(self, samples, sample_rate, *, kind, use_cache=True):
        values = np.asarray(samples, dtype=np.float64)
        if not len(values) or not np.any(values):
            return {
                "available": False,
                "reason": "silence_or_no_ac_excitation",
                "raw": None,
                "poly": None,
                "hops": None,
                "short_repeat": False,
                "doubling_count": 0,
                "resampled_original_samples": int(len(values)),
            }
        return super().score_array(
            values, sample_rate, kind=kind, use_cache=use_cache
        )


FAKE_CLI = r'''#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--mic", required=True)
parser.add_argument("--out", required=True)
args, _ = parser.parse_known_args()

source = Path(args.mic).resolve()
prefix = Path(args.out)
rate = 16000
source_audio, source_rate = sf.read(source, dtype="float64", always_2d=True)
if source_rate != rate:
    raise SystemExit(f"unexpected source rate {source_rate}")
frames = source_audio.shape[0]
seconds = np.arange(frames, dtype=np.float64) / rate


def write_tap(name, values):
    path = Path(f"{prefix}.{name}.wav")
    sf.write(path, np.asarray(values, dtype=np.float64), rate, format="WAV", subtype="FLOAT")
    return path


early = write_tap("tx_early", 0.08 * np.sin(2.0 * np.pi * 220.0 * seconds))
ch0 = 0.10 * np.sin(2.0 * np.pi * 180.0 * seconds)
ch1 = 0.01 * np.sin(2.0 * np.pi * 310.0 * seconds)
bypass = write_tap("tx_bypass", np.column_stack((ch0, ch1, np.zeros_like(ch0))))
unavailable = Path(f"{prefix}.tx_unavailable.wav")
unavailable.write_bytes(b"capture unavailable; intentionally not a WAV")
rx_after = write_tap("rx_after", 0.06 * np.sin(2.0 * np.pi * 170.0 * seconds))

if os.environ.get("FAKE_SOURCE_COPY") == "1":
    input_path = Path(f"{prefix}.tx_input.wav")
    sf.write(input_path, source_audio, rate, format="WAV", subtype="FLOAT")
else:
    input_path = source


def row(stage_id, direction, order, path, *, channels=1, capture_channels=None,
        enabled=True, capture_available=True, diagnostic=False):
    value = {
        "id": stage_id,
        "direction": direction,
        "order": order,
        "file": str(Path(path).resolve()),
        "sample_rate": rate,
        "channels": channels,
        "samples_per_channel": frames,
        "fixed_delay_samples": 0,
        "enabled": enabled,
        "capture_available": capture_available,
        "diagnostic": diagnostic,
        "measurement_domain": f"fake {stage_id}",
        "channel_policy": "each physical channel independently",
    }
    if capture_channels is not None:
        value["capture_channels"] = capture_channels
    return value


metadata = {
    "schema_version": 1,
    "input_sample_rate": rate,
    "input_samples_per_channel": frames,
    "input_duration_s": frames / rate,
    "pipeline_stages": [
        row("rx_after", "rx", 40, rx_after),
        row("tx_bypass", "tx", 20, bypass, channels=3, capture_channels=2,
            enabled=False),
        row("tx_unavailable", "tx", 25, unavailable, channels=2,
            capture_channels=0, capture_available=False),
        row("tx_early", "tx", 10, early),
        row("tx_input", "tx", 0, input_path, channels=source_audio.shape[1]),
    ],
}
Path(f"{prefix}.stages.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print("fake VCP engine completed one run")
'''


def _write_fake_cli(tmp_path: Path) -> Path:
    path = tmp_path / "fake_vcp_cli.py"
    script = FAKE_CLI.replace("#!/usr/bin/env python3", f"#!{sys.executable}", 1)
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)
    return path


def _patch_fake_measurement(monkeypatch, scorer=None):
    scorer = scorer or FakeScorer()
    monkeypatch.setattr(stage_mos, "import_pyloudnorm", _fake_pyloudnorm)
    monkeypatch.setattr(stage_mos, "DNSMOS", lambda *_args, **_kwargs: scorer)
    return scorer


def _stage_entry(stage_id, direction, order, path, **overrides):
    entry = {
        "id": stage_id,
        "direction": direction,
        "order": order,
        "file": str(Path(path).resolve()),
        "sample_rate": 16000,
        "channels": 1,
        "samples_per_channel": 16000,
        "fixed_delay_samples": 0,
        "enabled": True,
        "capture_available": True,
        "diagnostic": False,
        "measurement_domain": "test tap",
        "channel_policy": "no downmix",
    }
    entry.update(overrides)
    return entry


def _score_direct_tap(tmp_path, tap, *, channels, scorer):
    stage = stage_mos.normalize_stage_entries(
        {"pipeline_stages": [_stage_entry("tx_stage", "tx", 10, tap, channels=channels)]},
        tmp_path,
    )[0]
    normalized_dir = tmp_path / "normalized"
    normalized_dir.mkdir(exist_ok=True)
    return stage_mos.score_stage(
        stage,
        {"input_sample_rate": 16000, "input_samples_per_channel": 16000},
        SimpleNamespace(window=None, kind="near_only", loudness_target=-30.0),
        scorer,
        _fake_pyloudnorm(),
        normalized_dir,
    )


def test_fixed_delay_crop_and_rates_keep_input_timeline_duration():
    input_rate = 16000
    input_samples = 16000
    duration = input_samples / input_rate
    scored_lengths = []
    for rate, delay in ((24000, 120), (48000, 240)):
        nominal = input_samples * rate // input_rate
        data = np.linspace(-0.1, 0.1, nominal + delay, endpoint=False)
        cropped, crop_meta = stage_mos.crop_window(
            data, rate, input_samples, input_rate, delay, (0.2, 0.7), duration
        )
        assert cropped[0] == pytest.approx(data[delay + round(0.2 * rate)])
        assert crop_meta["start_sample"] == round(0.2 * rate)
        assert crop_meta["end_sample"] == round(0.7 * rate)
        assert len(cropped) == round(0.5 * rate)
        common = stage_mos.resample_for_scoring(cropped, rate)
        assert len(common) == round(0.5 * stage_mos.SCORE_RATE)
        scored_lengths.append(len(common))

    assert scored_lengths == [8000, 8000]
    with pytest.raises(ValueError, match="is not the input duration"):
        stage_mos.crop_window(
            np.zeros(48000 + 240 + 480),
            48000,
            input_samples,
            input_rate,
            240,
            None,
            duration,
        )


def test_minus30_is_constant_gain_and_float_wav_roundtrip(tmp_path):
    rate = stage_mos.SCORE_RATE
    seconds = np.arange(rate, dtype=np.float64) / rate
    source = 0.9 * np.sin(2.0 * np.pi * 173.0 * seconds)
    source += 0.05 * np.sin(2.0 * np.pi * 431.0 * seconds)
    snapshot = source.copy()
    output_path = tmp_path / "normalized.wav"

    normalized, meta = stage_mos.normalize_to_target(
        source, -30.0, _fake_pyloudnorm(), output_path
    )

    assert normalized is not None
    assert meta["available"] is True
    assert meta["reason"] is None
    assert meta["peak_prep_gain"] == pytest.approx(0.7 / np.max(np.abs(source)))
    assert meta["original_measurement_finite"] is True
    assert meta["prepared_measured_lufs"] is not None
    assert meta["constant_gain_verification_max_abs_error"] <= 1e-12
    assert meta["output_peak"] < 1.0
    assert meta["output_clipped_samples"] == 0
    assert meta["reread_max_abs_error"] <= meta["reread_error_bound"]
    assert abs(meta["reread_measured_lufs"] + 30.0) < 1e-3
    assert np.array_equal(source, snapshot)

    gain = normalized[np.abs(source) > 1e-5] / source[np.abs(source) > 1e-5]
    np.testing.assert_allclose(gain, np.median(gain), rtol=0.0, atol=2e-7)
    reread, reread_rate = sf.read(output_path, dtype="float64", always_2d=True)
    assert reread_rate == rate
    assert reread.shape == (rate, 1)
    assert np.max(np.abs(reread[:, 0] - normalized)) <= meta["reread_error_bound"]


def test_normalization_clipping_is_explicitly_unavailable(tmp_path, monkeypatch):
    rate = stage_mos.SCORE_RATE
    source = 0.5 * np.sin(2.0 * np.pi * 211.0 * np.arange(rate) / rate)
    output_path = tmp_path / "must-not-exist.wav"
    measured = iter((-40.0, -40.0, 0.0))
    monkeypatch.setattr(stage_mos, "measure_lufs", lambda *_args: next(measured))

    normalized, meta = stage_mos.normalize_to_target(
        source, 0.0, _fake_pyloudnorm(), output_path
    )

    assert normalized is None
    assert meta["available"] is False
    assert meta["reason"] == "normalization_clipping"
    assert meta["output_peak"] >= 1.0 or meta["output_clipped_samples"] > 0
    assert not output_path.exists()


def test_silence_has_no_scores_and_short_audio_marks_repeat():
    silent, silent_meta = perceptual.prepare_audio(
        np.zeros(stage_mos.SCORE_RATE, dtype=np.float64), stage_mos.SCORE_RATE
    )
    assert silent is None
    assert silent_meta["available"] is False
    silent_record = stage_mos.score_record({**silent_meta, "poly": None, "raw": None})
    assert silent_record["available"] is False
    assert silent_record["reason"] == "silence_or_no_ac_excitation"
    assert [silent_record[key] for key in ("G", "S", "N")] == [None, None, None]
    assert silent_record["raw"] is None
    assert silent_record["poly"] is None
    assert silent_record["hops"] is None

    short_seconds = np.arange(1600, dtype=np.float64) / stage_mos.SCORE_RATE
    short_audio, short_meta = perceptual.prepare_audio(
        0.1 * np.sin(2.0 * np.pi * 300.0 * short_seconds), stage_mos.SCORE_RATE
    )
    assert short_audio is not None
    assert short_meta["available"] is True
    assert short_meta["short_repeat"] is True
    assert short_meta["doubling_count"] >= 1
    short_record = stage_mos.score_record(
        {
            **short_meta,
            "model_sha256": stage_mos.MODEL_SHA,
            "raw": {"SIG": 2.0, "BAK": 3.0, "OVRL": 4.0},
            "poly": {"SIG": 2.0, "BAK": 3.0, "OVRL": 4.0},
            "hops": [],
        }
    )
    assert short_record["short_repeat"] is True
    assert short_record["doubling_count"] == short_meta["doubling_count"]
    assert short_record["G"] == 4.0
    assert short_record["S"] == 2.0
    assert short_record["N"] == 3.0


def test_source_input_exception_and_float32_copy_are_exact(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    source_dir = tmp_path / "capture"
    source_dir.mkdir()
    source = source_dir / "mic.wav"
    values = np.linspace(-0.25, 0.25, 16000, dtype=np.float64)
    sf.write(source, values.reshape(-1, 1), 16000, format="WAV", subtype="FLOAT")
    tap = output_dir / "bf.wav"
    metadata = {
        "pipeline_stages": [
            _stage_entry("tx_input", "tx", 0, source),
            _stage_entry("tx_bf", "tx", 10, tap),
        ]
    }

    stages = stage_mos.normalize_stage_entries(
        metadata, output_dir, {"tx_input": source, "rx_input": None}
    )
    assert [stage["id"] for stage in stages] == ["tx_input", "tx_bf"]
    assert Path(stages[0]["file"]) == source.resolve()

    outside = tmp_path / "not-this-run.wav"
    metadata["pipeline_stages"][1] = _stage_entry("tx_bf", "tx", 10, outside)
    with pytest.raises(ValueError, match="outside this run"):
        stage_mos.normalize_stage_entries(
            metadata, output_dir, {"tx_input": source, "rx_input": None}
        )

    metadata["pipeline_stages"] = [
        _stage_entry(
            "tx_unavailable",
            "tx",
            0,
            tap,
            channels=2,
            capture_channels=0,
            capture_available=False,
        )
    ]
    assert stage_mos.normalize_stage_entries(metadata, output_dir)[0]["capture_channels"] == 0
    metadata["pipeline_stages"][0]["capture_available"] = True
    with pytest.raises(ValueError, match="capture-available but has no physical channel"):
        stage_mos.normalize_stage_entries(metadata, output_dir)
    with pytest.raises(ValueError, match="pipeline_stages array"):
        stage_mos.normalize_stage_entries({}, output_dir)

    copy = output_dir / "tx-input-copy.wav"
    sf.write(copy, values.reshape(-1, 1), 16000, format="WAV", subtype="FLOAT")
    copy_stage = stage_mos.normalize_stage_entries(
        {"pipeline_stages": [_stage_entry("tx_input", "tx", 0, copy)]},
        output_dir,
        {"tx_input": source},
    )[0]
    validation = stage_mos.validate_input_stage_copy(copy_stage, source)
    assert validation["source_kind"] == "float32_run_copy"
    assert validation["samples_exact_after_float32_conversion"] is True
    assert validation["max_abs_sample_error"] == 0.0

    changed, changed_rate = sf.read(copy, dtype="float64", always_2d=True)
    changed[100, 0] += 1e-4
    sf.write(copy, changed, changed_rate, format="WAV", subtype="FLOAT")
    with pytest.raises(ValueError, match="exact float32 source PCM"):
        stage_mos.validate_input_stage_copy(copy_stage, source)

    missing = output_dir / "missing.wav"
    missing_stage = stage_mos.normalize_stage_entries(
        {"pipeline_stages": [_stage_entry("tx_bf", "tx", 0, missing)]},
        output_dir,
    )[0]
    args = SimpleNamespace(window=None, kind="near_only", loudness_target=-30.0)
    with pytest.raises(ValueError, match="stage file does not exist"):
        stage_mos.score_stage(
            missing_stage,
            {"input_sample_rate": 16000, "input_samples_per_channel": 16000},
            args,
            FakeScorer(),
            _fake_pyloudnorm(),
            output_dir / "normalized",
        )


def test_main_reports_alias_order_channels_bypass_and_unavailable(tmp_path, monkeypatch):
    cli = _write_fake_cli(tmp_path)
    assets = tmp_path / "assets"
    assets.mkdir()
    mic = tmp_path / "mic.wav"
    rate = stage_mos.SCORE_RATE
    seconds = np.arange(rate, dtype=np.float64) / rate
    sf.write(
        mic,
        0.12 * np.sin(2.0 * np.pi * 160.0 * seconds),
        rate,
        format="WAV",
        subtype="FLOAT",
    )
    output_dir = tmp_path / "result"
    scorer = _patch_fake_measurement(monkeypatch)

    assert stage_mos.main(
        [
            "--cli",
            str(cli),
            "--mic",
            str(mic),
            "--output-dir",
            str(output_dir),
            "--assets-dir",
            str(assets),
        ]
    ) == 0

    report = json.loads((output_dir / "stage-mos.json").read_text(encoding="utf-8"))
    stages = report["stages"]
    assert [stage["id"] for stage in stages] == [
        "tx_input",
        "tx_early",
        "tx_bypass",
        "tx_unavailable",
        "rx_after",
    ]
    assert report["input_stage_validation"]["tx_input"]["source_kind"] == "original_path"
    assert report["input_stage_validation"]["tx_input"]["samples_exact_after_float32_conversion"]

    bypass = next(stage for stage in stages if stage["id"] == "tx_bypass")
    assert bypass["enabled"] is False
    assert bypass["bypass"] is True
    assert bypass["status"] == "bypassed"
    assert len(bypass["channels_scored"]) == 2
    assert [row["channel"] for row in bypass["channels_scored"]] == [0, 1]
    assert [row["window_peak"] for row in bypass["channels_scored"]] == pytest.approx(
        [0.1, 0.01], rel=1e-4
    )
    for row in bypass["channels_scored"]:
        assert row["original_scores"]["available"] is True
        assert row["original_scores"]["G"] == 4.25
        assert row["original_scores"]["S"] == 2.25
        assert row["original_scores"]["N"] == 3.25
        assert row["original_scores"]["aliases"] == {"G": "OVRL", "S": "SIG", "N": "BAK"}

    unavailable = next(stage for stage in stages if stage["id"] == "tx_unavailable")
    assert unavailable["status"] == "capture_unavailable"
    assert unavailable["available"] is False
    assert unavailable["reason"] == "capture_unavailable"
    assert unavailable["channels_scored"] == []
    assert unavailable["tap_sha256_before"] == unavailable["tap_sha256_after"]

    with (output_dir / "stage-mos.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [(row["direction"], int(row["order"]), row["stage"], int(row["channel"])) for row in rows] == [
        ("tx", 0, "tx_input", 0),
        ("tx", 10, "tx_early", 0),
        ("tx", 20, "tx_bypass", 0),
        ("tx", 20, "tx_bypass", 1),
        ("rx", 40, "rx_after", 0),
    ]
    assert rows[2]["raw_G"] == "4.25"
    assert rows[2]["raw_S"] == "2.25"
    assert rows[2]["raw_N"] == "3.25"
    assert rows[2]["minus30_G"] == "4.25"
    assert len(scorer.calls) == 10  # five captured physical channels, raw + -30 each


def test_tap_rewrite_before_its_turn_is_rejected(tmp_path, monkeypatch):
    cli = _write_fake_cli(tmp_path)
    assets = tmp_path / "assets"
    assets.mkdir()
    mic = tmp_path / "mic.wav"
    sf.write(mic, 0.1 * np.sin(np.linspace(0.0, 40.0, 16000)), 16000, format="WAV", subtype="FLOAT")
    output_dir = tmp_path / "result"
    scorer = FakeScorer(output_dir / "run.tx_bypass.wav", mutate_on_call=1)
    _patch_fake_measurement(monkeypatch, scorer)

    with pytest.raises(RuntimeError, match="stage tap changed before scoring"):
        stage_mos.main(
            [
                "--cli",
                str(cli),
                "--mic",
                str(mic),
                "--output-dir",
                str(output_dir),
                "--assets-dir",
                str(assets),
            ]
        )
    assert not (output_dir / "stage-mos.json").exists()


def test_nonfinite_and_silent_taps_fail_closed_with_json_safe_nulls(tmp_path):
    rate = stage_mos.SCORE_RATE
    nonfinite = np.zeros(rate, dtype=np.float64)
    nonfinite[100] = np.nan
    nonfinite[200] = np.inf
    nonfinite_path = tmp_path / "nonfinite.wav"
    sf.write(
        nonfinite_path,
        nonfinite.reshape(-1, 1),
        rate,
        format="WAV",
        subtype="FLOAT",
    )

    nonfinite_result = _score_direct_tap(
        tmp_path, nonfinite_path, channels=1, scorer=SilenceAwareFakeScorer()
    )
    extrema = stage_mos.audio_extrema(nonfinite)
    assert extrema["peak"] is None and extrema["clipped_samples"] is None
    assert extrema["finite"] is False and extrema["nonfinite_samples"] == 2
    assert nonfinite_result["status"] == "captured"
    assert nonfinite_result["available"] is False
    assert nonfinite_result["original_available"] is False
    assert nonfinite_result["normalized_available"] is False
    assert nonfinite_result["reason"] == "nonfinite_audio"
    assert nonfinite_result["file_peak"] is None
    assert nonfinite_result["file_clipped_samples"] is None
    row = nonfinite_result["channels_scored"][0]
    assert row["window_peak"] is None
    assert row["window_clipped_samples"] is None
    for field in ("original_scores", "normalized_scores"):
        score = row[field]
        assert score["available"] is False
        assert score["reason"] == "nonfinite_audio"
        assert [score[key] for key in ("G", "S", "N")] == [None, None, None]
        assert score["raw"] is None
        assert score["poly"] is None
        assert score["hops"] is None
    json.dumps(nonfinite_result, allow_nan=False)

    silent_path = tmp_path / "silent.wav"
    sf.write(
        silent_path,
        np.zeros((rate, 1), dtype=np.float64),
        rate,
        format="WAV",
        subtype="FLOAT",
    )
    silent_result = _score_direct_tap(
        tmp_path, silent_path, channels=1, scorer=SilenceAwareFakeScorer()
    )
    assert silent_result["status"] == "captured"
    assert silent_result["available"] is False
    assert silent_result["original_available"] is False
    assert silent_result["normalized_available"] is False
    assert "silence" in silent_result["reason"] or "silent" in silent_result["reason"]
    silent_row = silent_result["channels_scored"][0]
    assert silent_row["window_peak"] == 0.0
    assert silent_row["original_scores"]["available"] is False
    assert silent_row["normalized_scores"]["available"] is False
    for field in ("original_scores", "normalized_scores"):
        score = silent_row[field]
        assert [score[key] for key in ("G", "S", "N")] == [None, None, None]
        assert score["raw"] is None
        assert score["poly"] is None
        assert score["hops"] is None
    json.dumps(silent_result, allow_nan=False)


def test_mixed_channel_availability_fails_stage_aggregate(tmp_path):
    rate = stage_mos.SCORE_RATE
    seconds = np.arange(rate, dtype=np.float64) / rate
    mixed = np.column_stack(
        (
            0.1 * np.sin(2.0 * np.pi * 220.0 * seconds),
            np.zeros(rate, dtype=np.float64),
        )
    )
    mixed_path = tmp_path / "mixed.wav"
    sf.write(mixed_path, mixed, rate, format="WAV", subtype="FLOAT")

    result = _score_direct_tap(
        tmp_path, mixed_path, channels=2, scorer=SilenceAwareFakeScorer()
    )
    assert result["status"] == "captured"
    assert result["available"] is False
    assert result["original_available"] is False
    assert result["normalized_available"] is False
    assert result["reason"] is not None
    assert [row["channel"] for row in result["channels_scored"]] == [0, 1]

    available, unavailable = result["channels_scored"]
    assert available["original_scores"]["available"] is True
    assert available["normalized_scores"]["available"] is True
    assert available["original_scores"]["G"] == 4.25
    assert available["normalized_scores"]["G"] == 4.25
    assert unavailable["original_scores"]["available"] is False
    assert unavailable["normalized_scores"]["available"] is False
    for score in (unavailable["original_scores"], unavailable["normalized_scores"]):
        assert [score[key] for key in ("G", "S", "N")] == [None, None, None]
        assert score["raw"] is None
        assert score["poly"] is None
        assert score["hops"] is None
    json.dumps(result, allow_nan=False)
