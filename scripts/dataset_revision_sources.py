#!/usr/bin/env python3
"""Revision-safe speech source access for NB, WB, and native-SWB datasets.

The acceptance generators historically read 16 kHz speech and resampled it to
48 kHz for SWB.  That is useful for engineering regression stimuli, but it
cannot establish native-fullband behaviour.  This module keeps the legacy bank
for NB/WB and replaces only the SWB speaker bank with hash-verified native
48 kHz recordings.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal
import soundfile as sf


NATIVE_SWB_RATE = 48_000
NATIVE_SWB_CUTOFF_HZ = 14_000.0
MAX_SOURCE_SEGMENT_S = 4.2
ACTIVITY_FRAME_S = 0.01
MIN_ACTIVE_SEGMENT_RATIO = 0.85
MIN_NON_SILENT_SECONDS = 9.01
NON_SILENT_FRAME_RMS = 1e-5
FADE_S = 0.005

VOICEBANK_LICENSE = "CC-BY-4.0"
VOICEBANK_DATASET = "VoiceBank/VCTK clean testset"
VOICEBANK_ATTRIBUTION = (
    "VoiceBank corpus; Cassia Valentini-Botinhao; "
    "source speech from CSTR VCTK"
)
VOICEBANK_SOURCE_PAGE = "https://datashare.ed.ac.uk/handle/10283/2791"


def _sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(text: str, seed: int) -> int:
    payload = f"{seed}:{text}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _crop_signature(segments: list[dict[str, Any]]) -> str:
    return _sha(_json_bytes(segments))


def _native_header(path: Path) -> sf._SoundFileInfo:
    try:
        info = sf.info(path)
    except Exception as error:
        raise ValueError(f"cannot inspect native SWB source: {path}") from error
    if int(info.samplerate) != NATIVE_SWB_RATE:
        raise ValueError(
            f"native SWB source {path} has samplerate {info.samplerate}, "
            f"expected {NATIVE_SWB_RATE}"
        )
    if int(info.channels) != 1:
        raise ValueError(
            f"native SWB source {path} has {info.channels} channels, expected 1"
        )
    return info


class RevisionSourceBank:
    """Proxy a legacy source bank and override SWB speakers with native 48 kHz.

    NB/WB modes retain the exact legacy ``speakers`` mapping.  For SWB modes,
    only manifest rows explicitly declaring ``native_sample_rate_hz == 48000``
    are considered.  Every selected row is verified against its real file
    header, declared frame count, and SHA-256 before it enters the bank.
    """

    def __init__(self, legacy_bank: Any, mode: str):
        self.legacy_bank = legacy_bank
        self.mode = str(mode)
        self.is_swb = self.mode.startswith("SWB-")
        self.used = legacy_bank.used
        self.native_manifest_path: str | None = None
        self.native_manifest_sha256: str | None = None
        self.native_manifest_dataset_id: str | None = None
        if not self.is_swb:
            self.native48k_records: tuple[dict[str, Any], ...] = ()
            self.speakers = legacy_bank.speakers
            return

        records = self._load_native48k_records()
        self.native48k_records = tuple(records)
        self.speakers: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            self.speakers[str(record["speaker"])].append(record)
        if len(self.speakers) < 2:
            raise ValueError(
                "native 48 kHz SWB bank needs at least two speakers for "
                "disjoint near/far pools"
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.legacy_bank, name)

    @property
    def native48k_record_count(self) -> int:
        return len(self.native48k_records)

    def _read(
        self,
        path: str,
        expected: str,
    ) -> tuple[np.ndarray, int]:
        if self.is_swb:
            _native_header(Path(path))
        return self.legacy_bank._read(path, expected)

    def noise(self, *args: Any, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        return self.legacy_bank.noise(*args, **kwargs)

    def _load_native48k_records(self) -> list[dict[str, Any]]:
        manifest = Path(self.legacy_bank.fullband_manifest).resolve()
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception as error:
            raise ValueError(
                f"cannot read native fullband manifest: {manifest}"
            ) from error
        rows = payload.get("files")
        if not isinstance(rows, list):
            raise ValueError(f"native fullband manifest has no files list: {manifest}")

        self.native_manifest_path = str(manifest)
        self.native_manifest_sha256 = _file_sha(manifest)
        self.native_manifest_dataset_id = str(payload.get("dataset_id", "")) or None
        root = manifest.parent
        records: list[dict[str, Any]] = []
        seen_paths: set[Path] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("native fullband manifest contains a non-object row")
            if row.get("native_sample_rate_hz") != NATIVE_SWB_RATE:
                continue
            required = (
                "native_path",
                "native_sha256",
                "speaker_id",
                "duration_s",
                "native_samples",
            )
            missing = [key for key in required if key not in row]
            if missing:
                raise ValueError(
                    f"native 48 kHz manifest row lacks {missing}: {row!r}"
                )

            source_path = (root / str(row["native_path"])).resolve()
            if source_path in seen_paths:
                raise ValueError(
                    f"native 48 kHz manifest repeats source path: {source_path}"
                )
            if not source_path.is_file():
                raise ValueError(
                    f"native 48 kHz source does not exist: {source_path}"
                )
            seen_paths.add(source_path)

            info = _native_header(source_path)
            expected_frames = int(row["native_samples"])
            if int(info.frames) != expected_frames:
                raise ValueError(
                    f"native SWB source {source_path} has {info.frames} frames, "
                    f"manifest declares {expected_frames}"
                )
            seconds = float(row["duration_s"])
            if not math.isfinite(seconds) or seconds <= 0.0:
                raise ValueError(
                    f"native SWB source {source_path} has invalid duration_s"
                )
            actual_seconds = float(info.frames) / float(info.samplerate)
            if not math.isclose(
                seconds,
                actual_seconds,
                rel_tol=0.0,
                abs_tol=max(1.0 / NATIVE_SWB_RATE, 1e-6),
            ):
                raise ValueError(
                    f"native SWB source {source_path} duration {seconds} "
                    f"differs from real duration {actual_seconds}"
                )

            expected_sha256 = str(row["native_sha256"]).lower()
            actual_sha256 = _file_sha(source_path)
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"source SHA mismatch: {source_path}: "
                    f"expected {expected_sha256}, got {actual_sha256}"
                )

            activity_cut_dbfs = None
            activity_proxy = row.get("activity_proxy")
            if isinstance(activity_proxy, dict):
                value = activity_proxy.get("active_cut_dbfs")
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    activity_cut_dbfs = float(value)

            records.append(
                {
                    "path": str(source_path),
                    "sha256": actual_sha256,
                    "speaker": str(row["speaker_id"]),
                    "seconds": seconds,
                    "license": VOICEBANK_LICENSE,
                    "dataset": VOICEBANK_DATASET,
                    "attribution": VOICEBANK_ATTRIBUTION,
                    "source_page": VOICEBANK_SOURCE_PAGE,
                    "language": str(row.get("language", "en")),
                    "source_sample_rate_hz": NATIVE_SWB_RATE,
                    "native_samples": expected_frames,
                    "activity_cut_dbfs": activity_cut_dbfs,
                    "manifest_id": row.get("id"),
                    "manifest_record_sha256": _sha(_json_bytes(row)),
                }
            )

        if not records:
            raise ValueError(
                f"native fullband manifest has no {NATIVE_SWB_RATE} Hz rows: "
                f"{manifest}"
            )
        records.sort(key=lambda item: (str(item["speaker"]), str(item["path"])))
        return records


def _source_path(bank: Any, record: dict[str, Any]) -> Path:
    path = Path(str(record["path"]))
    if path.is_absolute():
        return path
    speech_manifest = getattr(bank, "speech_manifest", None)
    if speech_manifest is not None:
        return (Path(speech_manifest).resolve().parent / path).resolve()
    source_root = getattr(bank, "source_root", None)
    if source_root is not None:
        return (Path(source_root).resolve() / path).resolve()
    raise ValueError(f"relative source path has no bank root: {path}")


def _frame_activity(
    samples: np.ndarray,
    rate: int,
) -> tuple[np.ndarray, int]:
    frame = max(1, int(round(ACTIVITY_FRAME_S * rate)))
    total = len(samples) // frame
    if total <= 0:
        return np.empty(0, dtype=bool), frame
    framed = samples[: total * frame].reshape(total, frame)
    frame_rms = np.sqrt(np.mean(np.square(framed, dtype=np.float64), axis=1))
    return frame_rms > NON_SILENT_FRAME_RMS, frame


def _non_silent_seconds(samples: np.ndarray, rate: int) -> float:
    active, frame = _frame_activity(samples, rate)
    if not len(active):
        return 0.0
    return float(np.count_nonzero(active) * frame / rate)


def _select_active_window(
    samples: np.ndarray,
    source_rate: int,
    requested_target_frames: int,
    target_rate: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, float] | None:
    source_frames = min(
        len(samples),
        max(
            1,
            int(
                math.ceil(
                    requested_target_frames * source_rate / target_rate
                )
            ),
        ),
    )
    if source_frames <= 0:
        return None

    active, frame = _frame_activity(samples, source_rate)
    if not len(active):
        start = (
            int(rng.integers(0, len(samples) - source_frames + 1))
            if len(samples) > source_frames
            else 0
        )
        selected = samples[start : start + source_frames]
        return selected, start, 0.0

    window_frames = max(1, int(math.ceil(source_frames / frame)))
    max_start_frame = max(0, (len(samples) - source_frames) // frame)
    last_start = min(len(active) - window_frames, max_start_frame)
    if last_start < 0:
        start = 0
        selected = samples[:source_frames]
        active_ratio = _non_silent_seconds(selected, source_rate) / (
            len(selected) / source_rate
        )
        return selected, start, float(active_ratio)

    cumulative = np.concatenate(
        (np.array([0], dtype=np.int64), np.cumsum(active, dtype=np.int64))
    )
    starts = np.arange(last_start + 1, dtype=np.int64)
    active_counts = cumulative[starts + window_frames] - cumulative[starts]
    active_ratios = active_counts.astype(np.float64) / window_frames
    eligible = np.flatnonzero(active_ratios >= MIN_ACTIVE_SEGMENT_RATIO)
    if len(eligible):
        selected_start = int(eligible[int(rng.integers(0, len(eligible)))])
    else:
        selected_start = int(np.argmax(active_ratios))

    start = selected_start * frame
    selected = samples[start : start + source_frames]
    active_ratio = _non_silent_seconds(selected, source_rate) / (
        len(selected) / source_rate
    )
    return selected, start, float(active_ratio)


def _resample_segment(
    samples: np.ndarray,
    source_rate: int,
    target_rate: int,
) -> np.ndarray:
    if source_rate == target_rate:
        return np.asarray(samples, dtype=np.float64).copy()
    divisor = math.gcd(source_rate, target_rate)
    return np.asarray(
        signal.resample_poly(
            samples,
            target_rate // divisor,
            source_rate // divisor,
        ),
        dtype=np.float64,
    )


def _fade_segment(samples: np.ndarray, rate: int) -> np.ndarray:
    faded = np.asarray(samples, dtype=np.float64).copy()
    edge = min(int(round(FADE_S * rate)), len(faded) // 2)
    if edge > 0:
        ramp = np.sin(
            np.linspace(0.0, math.pi / 2.0, edge, dtype=np.float64)
        ) ** 2
        faded[:edge] *= ramp
        faded[-edge:] *= ramp[::-1]
    return faded


def _resampling_info(
    segments: list[dict[str, Any]],
    target_rate: int,
) -> dict[str, Any]:
    source_rates = sorted(
        {int(segment["source_sample_rate_hz"]) for segment in segments}
    )
    if len(source_rates) != 1:
        operations = []
        for source_rate in source_rates:
            divisor = math.gcd(source_rate, target_rate)
            operations.append(
                {
                    "source_rate_hz": source_rate,
                    "up": target_rate // divisor,
                    "down": source_rate // divisor,
                }
            )
        return {
            "method": "per-segment scipy.signal.resample_poly",
            "source_rate_hz": source_rates,
            "target_rate_hz": target_rate,
            "operations": operations,
        }

    source_rate = source_rates[0]
    if source_rate == target_rate:
        return {
            "method": "none",
            "source_rate_hz": source_rate,
            "target_rate_hz": target_rate,
        }
    divisor = math.gcd(source_rate, target_rate)
    return {
        "method": "scipy.signal.resample_poly",
        "up": target_rate // divisor,
        "down": source_rate // divisor,
        "source_rate_hz": source_rate,
        "target_rate_hz": target_rate,
    }


def revision_speech_crop(
    bank: Any,
    records: list[dict[str, Any]],
    *,
    rate: int,
    dataset_id: str,
    group_id: int,
    role: str,
    base_seed: int,
    duration_s: float = 12.0,
    conditioned_peak: float = 0.12,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Assemble a deterministic, non-repeating, activity-rich speech crop.

    Native SWB sources are never upsampled from 16 kHz.  Each source file is
    used at most once per crop, and a group-specific stable seed chooses an
    active window from every selected file.
    """

    if not records:
        raise ValueError(f"{role} crop has no source records")
    if not isinstance(rate, int) or rate <= 0:
        raise ValueError(f"invalid target sample rate: {rate!r}")
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError(f"invalid crop duration: {duration_s!r}")
    if not math.isfinite(conditioned_peak) or conditioned_peak <= 0.0:
        raise ValueError(f"invalid conditioned peak: {conditioned_peak!r}")

    frames = int(round(duration_s * rate))
    if frames <= 0:
        raise ValueError("crop duration rounds to zero frames")
    recipe_seed = _stable_seed(
        f"{dataset_id}:{group_id}:{role}:crop",
        int(base_seed),
    )
    rng = np.random.default_rng(recipe_seed)
    order = rng.permutation(len(records))
    max_segment_frames = max(1, int(round(MAX_SOURCE_SEGMENT_S * rate)))

    parts: list[np.ndarray] = []
    segments: list[dict[str, Any]] = []
    remaining = frames
    for raw_index in order:
        if remaining <= 0:
            break
        record = records[int(raw_index)]
        path = _source_path(bank, record)
        expected_sha256 = str(record["sha256"])
        samples, source_rate = bank._read(str(path), expected_sha256)
        source_rate = int(source_rate)
        if rate == NATIVE_SWB_RATE and source_rate != NATIVE_SWB_RATE:
            raise ValueError(
                f"48 kHz revision crop requires native 48 kHz sources; "
                f"{path} is {source_rate} Hz"
            )
        samples = np.asarray(samples, dtype=np.float64).reshape(-1)
        if not len(samples):
            continue
        if not np.isfinite(samples).all():
            raise ValueError(f"non-finite speech source: {path}")

        available_target_frames = int(
            math.floor(len(samples) * rate / source_rate)
        )
        minimum_segment_frames = min(
            remaining,
            max(1, int(round(0.1 * rate))),
        )
        if available_target_frames < minimum_segment_frames:
            continue

        requested_limit = min(
            remaining,
            max_segment_frames,
            available_target_frames,
        )
        selected: tuple[np.ndarray, int, float] | None = None
        selected_frames = 0
        fallback: tuple[np.ndarray, int, float, int] | None = None
        attempted: set[int] = set()
        for fraction in (1.0, 0.85, 0.7):
            requested = min(
                requested_limit,
                max(
                    minimum_segment_frames,
                    int(round(requested_limit * fraction)),
                ),
            )
            requested = min(requested, remaining, available_target_frames)
            if requested in attempted:
                continue
            attempted.add(requested)
            candidate = _select_active_window(
                samples,
                source_rate,
                requested,
                rate,
                rng,
            )
            if candidate is None:
                continue
            if fallback is None or candidate[2] > fallback[2]:
                fallback = (*candidate, requested)
            if candidate[2] >= MIN_ACTIVE_SEGMENT_RATIO:
                selected = candidate
                selected_frames = requested
                break

        if selected is None and fallback is not None:
            selected = fallback[:3]
            selected_frames = fallback[3]
        if selected is None:
            continue

        source_segment, source_start, _ = selected
        target_segment = _resample_segment(
            source_segment,
            source_rate,
            rate,
        )
        if len(target_segment) < selected_frames:
            continue
        target_segment = target_segment[:selected_frames]
        faded = _fade_segment(target_segment, rate)
        parts.append(faded)
        remaining -= selected_frames

        target_start = int(round(source_start * rate / source_rate))
        segment = {
            "path": str(path),
            "sha256": expected_sha256,
            "speaker": str(record["speaker"]),
            "seconds": float(record["seconds"]),
            "license": record.get("license"),
            "dataset": record.get("dataset"),
            "attribution": record.get("attribution"),
            "source_page": record.get("source_page"),
            "source_sample_rate_hz": source_rate,
            "target_sample_rate_hz": rate,
            "start_sample_at_source_rate": int(source_start),
            "start_sample_at_target_rate": target_start,
            "selected_samples_at_source_rate": int(len(source_segment)),
            "selected_samples_at_target_rate": int(selected_frames),
            "selected_source_audio_sha256_f32le": _sha(
                np.asarray(source_segment, dtype="<f4").tobytes()
            ),
            "selected_audio_sha256_f32le": _sha(
                np.asarray(faded, dtype="<f4").tobytes()
            ),
            "repeated": False,
        }
        segments.append(segment)
        bank.used[str(path)] = segment

    if remaining != 0:
        raise ValueError(
            f"could not assemble {duration_s:g}s non-repeating {role} speech crop"
        )

    waveform = np.concatenate(parts)
    band_limit_hz: float | None = None
    if rate == NATIVE_SWB_RATE:
        band_limit_hz = NATIVE_SWB_CUTOFF_HZ
        waveform = signal.sosfiltfilt(
            signal.butter(
                6,
                band_limit_hz,
                fs=rate,
                output="sos",
            ),
            waveform,
        )

    waveform -= float(np.mean(waveform))
    peak = float(np.max(np.abs(waveform)))
    if peak <= 1e-8:
        raise ValueError(f"{role} speech crop is silent")
    normalization_gain = float(conditioned_peak) / peak
    waveform *= normalization_gain

    non_silent_seconds = _non_silent_seconds(waveform, rate)
    required_non_silent_seconds = min(MIN_NON_SILENT_SECONDS, float(duration_s))
    if non_silent_seconds + 1e-9 < required_non_silent_seconds:
        raise ValueError(
            f"{role} crop has {non_silent_seconds:.6f}s non-silent activity, "
            f"below required {required_non_silent_seconds:.6f}s"
        )

    info = {
        "source_kind": "clean_speech_crop",
        "source_file_count": len(records),
        "selected_segment_count": len(segments),
        "segments": segments,
        "crop_recipe_sha256": _crop_signature(segments),
        "conditioned_peak": float(conditioned_peak),
        "peak_normalization_gain_linear": normalization_gain,
        "source_gain_linear": normalization_gain,
        "repeated": False,
        "speaker_pool": role,
        "speakers": sorted({str(segment["speaker"]) for segment in segments}),
        "target_sample_rate_hz": rate,
        "resampling": _resampling_info(segments, rate),
        "band_limit_hz": band_limit_hz,
        "native_fullband": rate == NATIVE_SWB_RATE,
        "transport_simulated": False,
        "non_silent_seconds": non_silent_seconds,
        "minimum_non_silent_seconds": required_non_silent_seconds,
        "source_reuse_allowed_across_cases": True,
        "within_case_source_repetition": False,
        "dataset_id": str(dataset_id),
        "group_id": int(group_id),
        "role": str(role),
        "base_seed": int(base_seed),
        "recipe_seed": int(recipe_seed),
        "native_manifest_path": getattr(bank, "native_manifest_path", None),
        "native_manifest_sha256": getattr(bank, "native_manifest_sha256", None),
        "native_manifest_dataset_id": getattr(
            bank,
            "native_manifest_dataset_id",
            None,
        ),
        "native_verified_source_count": int(
            getattr(bank, "native48k_record_count", 0)
        ),
        "native_source_scope": "actual_manifest_sha256_current_verified_subset",
    }
    return np.asarray(waveform, dtype=np.float64), info
