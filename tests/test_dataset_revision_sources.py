from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest
from scipy import signal
import soundfile as sf

from scripts.dataset_revision_sources import (
    RevisionSourceBank,
    revision_speech_crop,
)


PROJECT = Path(__file__).resolve().parents[1]
CONVERSATION_GENERATOR = (
    PROJECT / "scripts/create-background-noise-mode-conversation-dataset.py"
)
NATIVE_MANIFEST = (
    PROJECT / "datasets/independent-fullband-20261012/dataset-manifest.json"
)
RATE = 48_000


def _sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_signal(
    path: Path,
    *,
    rate: int,
    seconds: float,
    low_frequency_hz: float,
    high_frequency_hz: float,
    leading_silence_s: float = 0.0,
    trailing_silence_s: float = 0.0,
) -> dict[str, Any]:
    frames = int(round(seconds * rate))
    samples = np.zeros(frames, dtype=np.float64)
    start = int(round(leading_silence_s * rate))
    end = frames - int(round(trailing_silence_s * rate))
    time = np.arange(end - start, dtype=np.float64) / rate
    active = (
        0.07 * np.sin(2.0 * math.pi * low_frequency_hz * time)
        + 0.03
        * np.sin(
            2.0 * math.pi * high_frequency_hz * time
            + 0.17 * low_frequency_hz / 100.0
        )
    )
    edge = min(int(round(0.01 * rate)), len(active) // 2)
    if edge:
        ramp = np.sin(
            np.linspace(0.0, math.pi / 2.0, edge, dtype=np.float64)
        ) ** 2
        active[:edge] *= ramp
        active[-edge:] *= ramp[::-1]
    samples[start:end] = active
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples, rate, format="WAV", subtype="PCM_16")
    return {
        "path": path,
        "sha256": _file_sha(path),
        "frames": frames,
        "seconds": seconds,
    }


def _write_manifest(root: Path, rows: list[dict[str, Any]]) -> Path:
    path = root / "dataset-manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_id": "revision-source-test",
                "files": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


class _LegacyBank:
    def __init__(
        self,
        *,
        fullband_manifest: Path | None = None,
        speech_manifest: Path | None = None,
    ):
        self.fullband_manifest = fullband_manifest
        self.speech_manifest = speech_manifest
        self.speakers = {
            "legacy-near": [
                {
                    "path": "legacy/near.wav",
                    "sha256": "a" * 64,
                    "speaker": "legacy-near",
                    "seconds": 12.0,
                }
            ],
            "legacy-far": [
                {
                    "path": "legacy/far.wav",
                    "sha256": "b" * 64,
                    "speaker": "legacy-far",
                    "seconds": 12.0,
                }
            ],
        }
        self.used: dict[str, dict[str, Any]] = {}
        self.marker = "legacy-bank"
        self._cache: dict[tuple[str, str], tuple[np.ndarray, int]] = {}

    def _read(self, path: str, expected: str) -> tuple[np.ndarray, int]:
        key = (path, expected)
        if key not in self._cache:
            source = Path(path)
            actual = _file_sha(source)
            if actual != expected:
                raise ValueError(f"source SHA mismatch: {source}")
            samples, rate = sf.read(source, always_2d=True)
            self._cache[key] = (
                np.asarray(samples[:, 0], dtype=np.float64),
                int(rate),
            )
        cached, rate = self._cache[key]
        return cached.copy(), rate

    def noise(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        return np.arange(8, dtype=np.float64), {
            "args": args,
            "kwargs": kwargs,
            "marker": self.marker,
        }


@pytest.fixture(scope="session")
def native_bank(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[RevisionSourceBank, _LegacyBank, Path]:
    root = tmp_path_factory.mktemp("native48k-source-bank")
    rows: list[dict[str, Any]] = []
    for speaker_index, speaker in enumerate(("p232", "p257")):
        for index in range(97):
            if index == 0:
                seconds = 6.0
                leading_silence_s = 0.5
                trailing_silence_s = 0.5
            elif index == 1:
                seconds = 4.8
                leading_silence_s = 0.4
                trailing_silence_s = 0.4
            else:
                seconds = 0.25
                leading_silence_s = 0.0
                trailing_silence_s = 0.0
            source = root / "native" / speaker / f"{speaker}_{index:03d}.wav"
            low_frequency_hz = (
                220.0 + 17.0 * speaker_index + 11.0 * index
            )
            high_frequency_hz = (
                9_000.0 + 300.0 * (index % 10) + 100.0 * speaker_index
            )
            written = _write_signal(
                source,
                rate=RATE,
                seconds=seconds,
                low_frequency_hz=low_frequency_hz,
                high_frequency_hz=high_frequency_hz,
                leading_silence_s=leading_silence_s,
                trailing_silence_s=trailing_silence_s,
            )
            rows.append(
                {
                    "id": f"en_{speaker}_{index:03d}",
                    "native_path": str(source.relative_to(root)),
                    "native_sha256": written["sha256"],
                    "speaker_id": speaker,
                    "duration_s": written["seconds"],
                    "native_samples": written["frames"],
                    "native_sample_rate_hz": RATE,
                    "language": "en",
                    "activity_proxy": {
                        "active_cut_dbfs": -35.0,
                        "method": "test fixture",
                    },
                }
            )

    # A metadata-only 16 kHz row must be ignored before any file inspection.
    rows.append(
        {
            "id": "metadata-16k",
            "native_path": "native/p232/p232_000.wav",
            "native_sha256": rows[0]["native_sha256"],
            "speaker_id": "ignored",
            "duration_s": rows[0]["duration_s"],
            "native_samples": rows[0]["native_samples"],
            "native_sample_rate_hz": 16_000,
        }
    )
    manifest = _write_manifest(root, rows)
    legacy = _LegacyBank(fullband_manifest=manifest)
    bank = RevisionSourceBank(legacy, "SWB-wired")
    assert bank.native48k_record_count == 194
    return bank, legacy, root


def _load_conversation_generator():
    name = "itu_revision_sources_conversation_generator_test"
    spec = importlib.util.spec_from_file_location(name, CONVERSATION_GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _crop_signature(segments: list[dict[str, Any]]) -> str:
    payload = (
        json.dumps(
            segments,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return _sha(payload)


def test_native_bank_keeps_only_verified_48k_records_and_normalizes_metadata(
    native_bank,
):
    bank, legacy, _ = native_bank
    assert bank.speakers is not legacy.speakers
    assert bank.native_manifest_path == str(
        Path(legacy.fullband_manifest).resolve()
    )
    assert bank.native_manifest_sha256 == _file_sha(
        Path(legacy.fullband_manifest)
    )
    assert bank.native_manifest_dataset_id == "revision-source-test"
    assert bank.native48k_record_count == 194
    assert set(bank.speakers) == {"p232", "p257"}
    assert {speaker: len(rows) for speaker, rows in bank.speakers.items()} == {
        "p232": 97,
        "p257": 97,
    }

    for record in bank.native48k_records:
        source = Path(record["path"])
        assert source.is_absolute()
        assert source.is_file()
        info = sf.info(source)
        assert info.samplerate == RATE
        assert info.frames == record["native_samples"]
        assert _file_sha(source) == record["sha256"]
        assert record["seconds"] == pytest.approx(
            info.frames / info.samplerate,
            abs=1 / RATE,
        )
        assert record["license"] == "CC-BY-4.0"
        assert record["dataset"] == "VoiceBank/VCTK clean testset"
        assert "Valentini-Botinhao" in record["attribution"]
        assert record["source_page"].startswith("https://datashare.ed.ac.uk/")


def test_native_48k_bank_rejects_real_16k_file_that_claims_swb(tmp_path):
    valid = _write_signal(
        tmp_path / "native/valid.wav",
        rate=RATE,
        seconds=0.25,
        low_frequency_hz=250.0,
        high_frequency_hz=10_000.0,
    )
    fake = _write_signal(
        tmp_path / "native/fake_swb.wav",
        rate=16_000,
        seconds=0.25,
        low_frequency_hz=300.0,
        high_frequency_hz=6_000.0,
    )
    rows = [
        {
            "id": "valid",
            "native_path": str(valid["path"].relative_to(tmp_path)),
            "native_sha256": valid["sha256"],
            "speaker_id": "p1",
            "duration_s": valid["seconds"],
            "native_samples": valid["frames"],
            "native_sample_rate_hz": RATE,
        },
        {
            "id": "fake-swb",
            "native_path": str(fake["path"].relative_to(tmp_path)),
            "native_sha256": fake["sha256"],
            "speaker_id": "p2",
            "duration_s": fake["seconds"],
            "native_samples": fake["frames"],
            "native_sample_rate_hz": RATE,
        },
    ]
    legacy = _LegacyBank(fullband_manifest=_write_manifest(tmp_path, rows))
    with pytest.raises(ValueError, match="samplerate 16000"):
        RevisionSourceBank(legacy, "SWB-wireless")


def test_native_48k_bank_rejects_source_sha_mismatch(tmp_path):
    written = _write_signal(
        tmp_path / "native/hash.wav",
        rate=RATE,
        seconds=0.25,
        low_frequency_hz=250.0,
        high_frequency_hz=10_000.0,
    )
    rows = [
        {
            "id": "wrong-sha",
            "native_path": str(written["path"].relative_to(tmp_path)),
            "native_sha256": "0" * 64,
            "speaker_id": "p1",
            "duration_s": written["seconds"],
            "native_samples": written["frames"],
            "native_sample_rate_hz": RATE,
        }
    ]
    legacy = _LegacyBank(fullband_manifest=_write_manifest(tmp_path, rows))
    with pytest.raises(ValueError, match="source SHA mismatch"):
        RevisionSourceBank(legacy, "SWB-bt")


def test_speaker_pools_are_disjoint_and_both_sides_are_activity_rich(native_bank):
    bank, _, _ = native_bank
    module = _load_conversation_generator()
    near_records, far_records, metadata = module._speaker_pools(bank)

    assert metadata["available_source_file_count"] == 194
    assert metadata["available_speaker_count"] == 2
    assert metadata["speaker_pools_disjoint"] is True
    near_speakers = {row["speaker"] for row in near_records}
    far_speakers = {row["speaker"] for row in far_records}
    assert near_speakers.isdisjoint(far_speakers)

    near, near_info = revision_speech_crop(
        bank,
        near_records,
        rate=RATE,
        dataset_id="revision-test",
        group_id=31,
        role="near_input",
        base_seed=20261117,
    )
    far, far_info = revision_speech_crop(
        bank,
        far_records,
        rate=RATE,
        dataset_id="revision-test",
        group_id=31,
        role="far_input",
        base_seed=20261117,
    )
    assert len(near) == len(far) == 12 * RATE
    assert near_info["non_silent_seconds"] >= 9.01
    assert far_info["non_silent_seconds"] >= 9.01
    assert set(near_info["speakers"]).isdisjoint(far_info["speakers"])


def test_crop_is_stable_per_group_traceable_non_repeating_and_native_fullband(
    native_bank,
):
    bank, _, _ = native_bank
    records = bank.speakers["p232"]
    waveform, info = revision_speech_crop(
        bank,
        records,
        rate=RATE,
        dataset_id="revision-test",
        group_id=7,
        role="near_input",
        base_seed=20261117,
    )
    repeated_waveform, repeated_info = revision_speech_crop(
        bank,
        records,
        rate=RATE,
        dataset_id="revision-test",
        group_id=7,
        role="near_input",
        base_seed=20261117,
    )
    other_waveform, other_info = revision_speech_crop(
        bank,
        records,
        rate=RATE,
        dataset_id="revision-test",
        group_id=8,
        role="near_input",
        base_seed=20261117,
    )

    assert np.array_equal(waveform, repeated_waveform)
    assert info == repeated_info
    assert not np.array_equal(waveform, other_waveform)
    assert info["crop_recipe_sha256"] != other_info["crop_recipe_sha256"]
    assert len(waveform) == 12 * RATE
    assert np.isfinite(waveform).all()
    assert np.max(np.abs(waveform)) == pytest.approx(0.12, abs=1e-12)

    assert info["crop_recipe_sha256"] == _crop_signature(info["segments"])
    assert info["repeated"] is False
    assert info["target_sample_rate_hz"] == RATE
    assert info["speaker_pool"] == "near_input"
    assert info["resampling"] == {
        "method": "none",
        "source_rate_hz": RATE,
        "target_rate_hz": RATE,
    }
    assert info["band_limit_hz"] == 14_000.0
    assert info["non_silent_seconds"] >= 9.01
    assert info["minimum_non_silent_seconds"] == 9.01
    assert info["source_reuse_allowed_across_cases"] is True
    assert info["within_case_source_repetition"] is False
    assert info["native_manifest_path"] == bank.native_manifest_path
    assert info["native_manifest_sha256"] == bank.native_manifest_sha256
    assert info["native_manifest_dataset_id"] == "revision-source-test"
    assert info["native_verified_source_count"] == 194
    assert (
        info["native_source_scope"]
        == "actual_manifest_sha256_current_verified_subset"
    )

    paths = [segment["path"] for segment in info["segments"]]
    assert len(paths) == len(set(paths))
    assert all(segment["repeated"] is False for segment in info["segments"])
    for segment in info["segments"]:
        source = Path(segment["path"])
        assert source.is_absolute()
        assert _file_sha(source) == segment["sha256"]
        assert segment["source_sample_rate_hz"] == RATE
        assert segment["target_sample_rate_hz"] == RATE
        assert segment["license"] == "CC-BY-4.0"
        assert segment["dataset"] == "VoiceBank/VCTK clean testset"
        assert segment["attribution"]
        assert segment["source_page"]

        source_samples, source_rate = sf.read(source, always_2d=True)
        start = segment["start_sample_at_source_rate"]
        count = segment["selected_samples_at_source_rate"]
        selected = np.asarray(
            source_samples[start : start + count, 0],
            dtype=np.float64,
        )
        assert source_rate == RATE
        assert len(selected) == count
        assert segment["selected_source_audio_sha256_f32le"] == _sha(
            np.asarray(selected, dtype="<f4").tobytes()
        )
        assert segment["selected_samples_at_target_rate"] > 0
        assert segment["start_sample_at_target_rate"] >= 0

    frequencies = np.fft.rfftfreq(len(waveform), 1.0 / RATE)
    spectrum = np.abs(np.fft.rfft(waveform))
    fullband_8_to_14k = np.sum(
        np.square(spectrum[(frequencies >= 8_000.0) & (frequencies < 14_000.0)])
    )
    synthetic_15_to_22k = np.sum(
        np.square(spectrum[(frequencies >= 15_000.0) & (frequencies < 22_000.0)])
    )
    assert fullband_8_to_14k > 100.0 * max(synthetic_15_to_22k, 1e-30)


def test_nb_wb_keep_the_exact_legacy_speakers_and_proxy_legacy_accessors():
    legacy = _LegacyBank()
    legacy._read = lambda path, expected: (np.zeros(12 * 16_000), 16_000)  # type: ignore[method-assign]
    for mode in ("NB-wired", "NB-wireless", "WB-wired", "WB-bt"):
        bank = RevisionSourceBank(legacy, mode)
        assert bank.is_swb is False
        assert bank.speakers is legacy.speakers
        assert bank.used is legacy.used
        assert bank.marker == "legacy-bank"
        samples, rate = bank._read("legacy/path.wav", "expected")
        assert rate == 16_000
        assert samples.shape == (12 * 16_000,)
        noise, metadata = bank.noise("WB-wired", "scene", 16_000, 9)
        assert noise.shape == (8,)
        assert metadata["args"] == ("WB-wired", "scene", 16_000, 9)


def test_repository_manifest_exposes_only_821_real_native_48k_rows():
    actual_manifest_sha256 = _file_sha(NATIVE_MANIFEST)
    payload = json.loads(NATIVE_MANIFEST.read_text(encoding="utf-8"))
    native = [
        row
        for row in payload["files"]
        if row.get("native_sample_rate_hz") == RATE
    ]
    assert actual_manifest_sha256 == _sha(NATIVE_MANIFEST.read_bytes())
    assert payload["dataset_id"] == "ecnr-independent-fullband-20261012"
    assert len(native) == 821
    assert {
        speaker: sum(row["speaker_id"] == speaker for row in native)
        for speaker in ("p232", "p257")
    } == {"p232": 393, "p257": 428}
    for speaker in ("p232", "p257"):
        row = next(item for item in native if item["speaker_id"] == speaker)
        source = NATIVE_MANIFEST.parent / row["native_path"]
        info = sf.info(source)
        assert info.samplerate == RATE
        assert info.frames == row["native_samples"]
        assert _file_sha(source) == row["native_sha256"]
