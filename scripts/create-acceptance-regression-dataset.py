#!/usr/bin/env python3
"""Create a reproducible 106-case/336-item software regression ZIP.

This dataset is engineering regression material only. It is deliberately named
``software-regression-not-certification`` so it cannot be mistaken for formal
ITU-T or customer certification evidence.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.dont_write_bytecode = True

from server.acoustic import service
from server.acoustic.acceptance_contract import contract_items
from server.acoustic.acceptance_dataset import (
    MAX_ARCHIVE_BYTES,
    MAX_ASSET_BYTES,
    MAX_TOTAL_UNCOMPRESSED_BYTES,
    get_dataset,
    import_dataset,
    template_manifest,
)


SOURCE_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "itu-vehicle-fields-float-20261010"
    / "cases"
    / "field-p0-m2-multirender-snr15"
)
SOURCE_SCENE = SOURCE_DIR / "scene.json"
REFERENCE_SOURCE = SOURCE_DIR / "clean.wav"
FAR_END_SOURCE = SOURCE_DIR / "playback.wav"
SOURCE_RATE = 16_000
SOURCE_START_SECONDS = 12.0
DURATION_SECONDS = 11.0
ANALYSIS_START_SECONDS = 0.5
DATASET_NAME = "software-regression-not-certification"
SIGNAL_NAMES = ("microphone", "reference", "far_end", "noise")
EXPECTED_CASES = 106
EXPECTED_ITEMS = 336
EXPECTED_ASSETS = EXPECTED_CASES * len(SIGNAL_NAMES)
EXPECTED_OUT_OF_BAND_CASES = 18
EXPECTED_RECEIVE_LEVEL_ALIGNMENT_CASES = 18
EXPECTED_UPLIFTED_CASES = 30
PEAK_LIMIT = 0.7
DISK_RESERVE_BYTES = 2 * 1024**3
CONTRACT_ROWS_BY_ID = {str(row["id"]): row for row in contract_items()}
OUT_OF_BAND_MIN_RATE = {
    "NB": 16_000,
    "WB": 48_000,
    "SWB": 48_000,
}
@dataclass(frozen=True)
class SceneProfile:
    """Deterministic spectral recipe for one regression scene."""

    seed: int
    alpha: float
    low_cut_hz: float
    high_cut_hz: float
    band_center_hz: float
    band_width: float
    band_gain: float
    tones_hz: tuple[float, ...]
    tone_gain: float
    modulation_hz: float
    modulation_depth: float
    snr_db: float


# baseline is a quiet regression floor. sce1..sce4 correspond to the four
# catalog scenarios C1..C4: idle, city, highway, and highway with HVAC.
SCENE_PROFILES = {
    "baseline": SceneProfile(
        0x49545501,
        1.55,
        32.0,
        1_000.0,
        240.0,
        0.62,
        0.18,
        (),
        0.0,
        0.07,
        0.03,
        18.0,
    ),
    "sce1": SceneProfile(
        0x49545511,
        1.38,
        35.0,
        1_300.0,
        180.0,
        0.50,
        0.28,
        (72.0,),
        0.04,
        0.13,
        0.05,
        15.0,
    ),
    "sce2": SceneProfile(
        0x49545522,
        1.02,
        45.0,
        4_500.0,
        720.0,
        0.78,
        0.20,
        (),
        0.0,
        0.21,
        0.12,
        10.0,
    ),
    "sce3": SceneProfile(
        0x49545533,
        0.72,
        42.0,
        7_200.0,
        1_500.0,
        0.82,
        0.24,
        (),
        0.0,
        0.09,
        0.10,
        6.0,
    ),
    "sce4": SceneProfile(
        0x49545544,
        0.88,
        38.0,
        6_200.0,
        900.0,
        0.64,
        0.34,
        (120.0, 240.0, 360.0),
        0.055,
        0.27,
        0.08,
        3.0,
    ),
}

VOLUME_GAIN_DB = {
    "nominal": 0.0,
    "maximum": 6.0,
    "recommended": -2.0,
}
REQUIRED_LEVEL_VARIANTS = {
    ("volume", "maximum"),
    ("volume", "recommended"),
    ("input_level_db", -5),
    ("input_level_db", 5),
}


class RegressionDatasetError(RuntimeError):
    """The source material or generated regression dataset is invalid."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def _dbfs(value: float) -> float:
    return 20.0 * math.log10(max(value, 1e-15))


def _peak(samples: np.ndarray) -> float:
    return float(np.max(np.abs(samples)))


def _existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            raise RegressionDatasetError(f"no existing parent for {path}")
        candidate = candidate.parent
    return candidate


def _estimated_uncompressed_bytes(manifest: dict) -> int:
    frames_by_rate: dict[int, int] = defaultdict(int)
    for case in manifest["cases"]:
        rate = _sample_rate_for_case(case)
        frames_by_rate[rate] += int(round(DURATION_SECONDS * rate))
    # Four mono PCM16 signals per case, plus a conservative manifest allowance.
    return (
        sum(frames * len(SIGNAL_NAMES) * 2 for frames in frames_by_rate.values())
        + 2 * 1024 * 1024
    )


def _preflight_disk(output: Path, estimated_bytes: int) -> None:
    # ZIP worst case plus the importer's extracted copy, with a fixed reserve.
    required = 2 * estimated_bytes + DISK_RESERVE_BYTES
    paths = [output.parent, _existing_parent(Path(service.BUILD))]
    for path in paths:
        free = shutil.disk_usage(path).free
        if free < required:
            raise RegressionDatasetError(
                "insufficient disk space before writing: "
                f"{path} free={free} required={required}; stopped without cleanup"
            )


def _sample_rate_for_mode(mode: str) -> int:
    bandwidth = mode.split("-", 1)[0].upper()
    rates = {"NB": 8_000, "WB": 16_000, "SWB": 48_000}
    if bandwidth not in rates:
        raise RegressionDatasetError(f"unsupported mode bandwidth: {mode!r}")
    return rates[bandwidth]


def _case_families(case: dict) -> set[str]:
    families = set()
    for item_id in case["item_ids"]:
        row = CONTRACT_ROWS_BY_ID.get(str(item_id))
        if row is None:
            raise RegressionDatasetError(
                f"case {case.get('id')!r} references unknown item {item_id!r}"
            )
        families.add(str(row.get("family")))
    return families


def _sample_rate_for_case(case: dict) -> int:
    mode = str(case["conditions"]["mode"])
    bandwidth = mode.split("-", 1)[0].upper()
    base_rate = _sample_rate_for_mode(mode)
    needs_uplift = (
        "out_of_band" in _case_families(case)
        or _requires_receive_level_alignment(case)
    )
    if not needs_uplift:
        return base_rate
    minimum = OUT_OF_BAND_MIN_RATE.get(bandwidth)
    if minimum is None:
        raise RegressionDatasetError(
            f"unsupported uplift bandwidth {bandwidth!r} for case {case['id']}"
        )
    return max(base_rate, minimum)


def _requires_receive_level_alignment(case: dict) -> bool:
    conditions = case["conditions"]
    bandwidth = str(conditions["mode"]).split("-", 1)[0].upper()
    return (
        bandwidth in {"NB", "WB"}
        and conditions["direction"] == "receive"
        and conditions["scene"] == "baseline"
        and conditions["volume"] == "nominal"
        and conditions["window_ms"] is None
        and conditions["input_level_db"] in {-5, 0, 5}
    )


def _validate_receive_level_alignment(manifest: dict) -> dict:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for case in manifest["cases"]:
        if not _requires_receive_level_alignment(case):
            continue
        conditions = case["conditions"]
        grouped[
            (
                str(conditions["mode"]),
                str(conditions["scene"]),
                str(conditions["window_ms"]),
            )
        ].append(case)

    case_count = sum(len(cases) for cases in grouped.values())
    if case_count != EXPECTED_RECEIVE_LEVEL_ALIGNMENT_CASES:
        raise RegressionDatasetError(
            f"found {case_count} receive level-alignment cases, "
            f"expected {EXPECTED_RECEIVE_LEVEL_ALIGNMENT_CASES}"
        )

    groups = []
    for (mode, scene, window), cases in sorted(grouped.items()):
        levels = {case["conditions"]["input_level_db"] for case in cases}
        if levels != {-5, 0, 5}:
            raise RegressionDatasetError(
                f"{mode} receive level group {scene}/{window} has levels "
                f"{sorted(levels)}, expected [-5, 0, 5]"
            )
        rates = {_sample_rate_for_case(case) for case in cases}
        bandwidth = mode.split("-", 1)[0].upper()
        required_rate = OUT_OF_BAND_MIN_RATE[bandwidth]
        if rates != {required_rate}:
            raise RegressionDatasetError(
                f"{mode} receive level group {scene}/{window} has rates "
                f"{sorted(rates)}, expected [{required_rate}]"
            )
        groups.append(
            {
                "mode": mode,
                "scene": scene,
                "window_ms": None if window == "None" else int(window),
                "sample_rate": required_rate,
                "case_ids_by_level": {
                    str(case["conditions"]["input_level_db"]): case["id"]
                    for case in sorted(
                        cases,
                        key=lambda row: row["conditions"]["input_level_db"],
                    )
                },
            }
        )
    return {
        "case_count": case_count,
        "group_count": len(groups),
        "groups": groups,
    }


def _validate_sources() -> tuple[np.ndarray, np.ndarray, dict]:
    try:
        scene = json.loads(SOURCE_SCENE.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RegressionDatasetError(f"cannot read source scene: {error}") from error

    expected_hashes = {
        name: scene.get("files", {}).get(name, {}).get("sha256")
        for name in (REFERENCE_SOURCE.name, FAR_END_SOURCE.name)
    }
    actual_hashes = {}
    for path in (REFERENCE_SOURCE, FAR_END_SOURCE):
        if not path.is_file():
            raise RegressionDatasetError(f"missing real source asset: {path}")
        actual = _sha256_file(path)
        expected = expected_hashes[path.name]
        if not isinstance(expected, str) or actual != expected:
            raise RegressionDatasetError(
                f"source identity mismatch for {path.name}: "
                f"expected={expected!r} actual={actual}"
            )
        actual_hashes[path.name] = actual

    start = int(round(SOURCE_START_SECONDS * SOURCE_RATE))
    frames = int(round(DURATION_SECONDS * SOURCE_RATE))
    sources = []
    for path in (REFERENCE_SOURCE, FAR_END_SOURCE):
        try:
            with sf.SoundFile(path) as audio:
                if (
                    audio.samplerate != SOURCE_RATE
                    or audio.channels < 1
                    or audio.frames < start + frames
                ):
                    raise RegressionDatasetError(
                        f"{path.name} must be >=16 kHz, mono/stereo, and "
                        f"cover [{SOURCE_START_SECONDS}, "
                        f"{SOURCE_START_SECONDS + DURATION_SECONDS}) s"
                    )
                audio.seek(start)
                samples = audio.read(frames, dtype="float64", always_2d=True)[:, 0]
        except (OSError, RuntimeError) as error:
            raise RegressionDatasetError(f"cannot read {path}: {error}") from error
        if len(samples) != frames or not np.isfinite(samples).all():
            raise RegressionDatasetError(f"{path.name} excerpt is invalid")
        if _peak(samples) <= 1e-8 or _rms(samples) <= 1e-10:
            raise RegressionDatasetError(f"{path.name} excerpt is silent")
        sources.append(np.asarray(samples, dtype=np.float64))

    source_summary = {
        "reference": str(REFERENCE_SOURCE),
        "far_end": str(FAR_END_SOURCE),
        "sha256": actual_hashes,
        "excerpt_start_s": SOURCE_START_SECONDS,
        "duration_s": DURATION_SECONDS,
    }
    return sources[0], sources[1], source_summary


def _fade(samples: np.ndarray, rate: int, seconds: float = 0.02) -> np.ndarray:
    result = np.asarray(samples, dtype=np.float64).copy()
    count = min(int(round(seconds * rate)), len(result) // 2)
    if count > 0:
        edge = np.linspace(0.0, 1.0, count, endpoint=False)
        result[:count] *= edge
        result[-count:] *= edge[::-1]
    return result


def _normalize_peak(samples: np.ndarray, rate: int, target: float) -> np.ndarray:
    result = _fade(np.asarray(samples, dtype=np.float64) - float(np.mean(samples)), rate)
    peak = _peak(result)
    if peak <= 1e-12:
        raise RegressionDatasetError("cannot normalize silent source material")
    return result * (target / peak)


def _fit_frames(samples: np.ndarray, frames: int) -> np.ndarray:
    result = np.asarray(samples, dtype=np.float64)
    if len(result) > frames:
        return result[:frames]
    if len(result) < frames:
        return np.pad(result, (0, frames - len(result)))
    return result


def _resample(samples: np.ndarray, target_rate: int) -> np.ndarray:
    if target_rate == SOURCE_RATE:
        return _fit_frames(samples, int(round(DURATION_SECONDS * target_rate)))
    divisor = math.gcd(SOURCE_RATE, target_rate)
    resampled = resample_poly(
        np.asarray(samples, dtype=np.float64),
        target_rate // divisor,
        SOURCE_RATE // divisor,
        padtype="line",
    )
    return _fit_frames(resampled, int(round(DURATION_SECONDS * target_rate)))


def _spectral_noise(
    rate: int,
    frames: int,
    scene: str,
    profile: SceneProfile,
) -> np.ndarray:
    rng = np.random.default_rng(profile.seed ^ (rate << 12))
    white = rng.standard_normal(frames)
    spectrum = np.fft.rfft(white)
    frequencies = np.fft.rfftfreq(frames, d=1.0 / rate)
    effective = np.maximum(frequencies, 20.0)

    envelope = np.power(effective / 1_000.0, -profile.alpha / 2.0)
    envelope *= np.power(effective / profile.low_cut_hz, 2.0) / np.sqrt(
        1.0 + np.power(effective / profile.low_cut_hz, 4.0)
    )
    envelope *= 1.0 / np.sqrt(
        1.0 + np.power(effective / profile.high_cut_hz, 4.0)
    )
    envelope += profile.band_gain * np.exp(
        -0.5
        * np.square(
            np.log(effective / profile.band_center_hz) / profile.band_width
        )
    )
    envelope[0] = 0.0
    noise = np.fft.irfft(spectrum * envelope, n=frames)
    noise -= float(np.mean(noise))

    time = np.arange(frames, dtype=np.float64) / rate
    phase = float(rng.uniform(0.0, 2.0 * math.pi))
    modulation = 1.0 + profile.modulation_depth * np.sin(
        2.0 * math.pi * profile.modulation_hz * time + phase
    )
    noise *= modulation

    for index, frequency in enumerate(profile.tones_hz):
        tone_phase = float(rng.uniform(0.0, 2.0 * math.pi))
        noise += (
            profile.tone_gain
            * np.sin(2.0 * math.pi * frequency * time + tone_phase)
        )

    noise = _fade(noise, rate)
    current = _rms(noise)
    if current <= 1e-12:
        raise RegressionDatasetError(f"scene {scene} generated silent noise")
    return noise / current


def _scale_noise_to_snr(
    noise: np.ndarray,
    reference: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    reference_rms = _rms(reference)
    if reference_rms <= 1e-12:
        raise RegressionDatasetError("reference RMS is too low for noise scaling")
    return noise * (reference_rms / (10.0 ** (snr_db / 20.0)))


def _delay(samples: np.ndarray, rate: int, seconds: float) -> np.ndarray:
    count = int(round(seconds * rate))
    if count <= 0:
        return samples.copy()
    return np.concatenate((np.zeros(count), samples[:-count]))


def _base_material(
    reference_16k: np.ndarray,
    far_end_16k: np.ndarray,
    rate: int,
    scene: str,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    reference = _normalize_peak(_resample(reference_16k, rate), rate, 0.24)
    far_end = _normalize_peak(_resample(far_end_16k, rate), rate, 0.20)
    profile = SCENE_PROFILES[scene]
    noise = _spectral_noise(rate, len(reference), scene, profile)
    noise = _scale_noise_to_snr(noise, reference, profile.snr_db)
    echo = _delay(far_end, rate, 0.030) * 0.35
    microphone = _normalize_peak(reference + echo + noise, rate, 0.28)
    return (
        {
            "microphone": microphone,
            "reference": reference,
            "far_end": far_end,
            "noise": noise,
        },
        {
            "snr_db": profile.snr_db,
            "noise_rms_dbfs": _dbfs(_rms(noise)),
        },
    )


def _condition_gain_db(conditions: dict) -> float:
    volume = conditions["volume"]
    if volume not in VOLUME_GAIN_DB:
        raise RegressionDatasetError(f"unsupported volume condition: {volume!r}")
    return VOLUME_GAIN_DB[volume] + float(conditions["input_level_db"])


def _wav_bytes(samples: np.ndarray, rate: int) -> bytes:
    values = np.asarray(samples, dtype=np.float64)
    if not np.isfinite(values).all():
        raise RegressionDatasetError("generated signal contains non-finite samples")
    if _peak(values) >= PEAK_LIMIT:
        raise RegressionDatasetError(
            f"generated signal peak {_peak(values):.9f} is not below {PEAK_LIMIT}"
        )
    stream = BytesIO()
    sf.write(stream, values, rate, format="WAV", subtype="PCM_16")
    return stream.getvalue()


def _manifest() -> dict:
    manifest = template_manifest()
    manifest["name"] = DATASET_NAME
    if len(manifest["cases"]) != EXPECTED_CASES:
        raise RegressionDatasetError(
            f"template has {len(manifest['cases'])} cases, expected {EXPECTED_CASES}"
        )
    item_ids = [
        item_id
        for case in manifest["cases"]
        for item_id in case["item_ids"]
    ]
    if len(item_ids) != EXPECTED_ITEMS or len(set(item_ids)) != EXPECTED_ITEMS:
        raise RegressionDatasetError(
            f"template has {len(item_ids)} items / {len(set(item_ids))} unique ids, "
            f"expected {EXPECTED_ITEMS}"
        )
    for case in manifest["cases"]:
        if case["conditions"]["scene"] not in SCENE_PROFILES:
            raise RegressionDatasetError(
                f"unsupported scene {case['conditions']['scene']!r}"
            )
        case["analysis_start_s"] = ANALYSIS_START_SECONDS
    return manifest


def _write_archive(
    output: Path,
    manifest: dict,
    reference_16k: np.ndarray,
    far_end_16k: np.ndarray,
) -> dict:
    manifest_blob = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    base_cache: dict[tuple[int, str], tuple[dict[str, np.ndarray], dict[str, float]]] = {}
    with zipfile.ZipFile(
        output,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        archive.writestr("manifest.json", manifest_blob)
        for case in manifest["cases"]:
            conditions = case["conditions"]
            rate = _sample_rate_for_case(case)
            scene = str(conditions["scene"])
            key = (rate, scene)
            if key not in base_cache:
                base_cache[key] = _base_material(
                    reference_16k,
                    far_end_16k,
                    rate,
                    scene,
                )
            base, _ = base_cache[key]
            gain = 10.0 ** (_condition_gain_db(conditions) / 20.0)
            for role in SIGNAL_NAMES:
                relative = case["signals"][role]
                payload = _wav_bytes(base[role] * gain, rate)
                if len(payload) > MAX_ASSET_BYTES:
                    raise RegressionDatasetError(
                        f"asset {relative} exceeds {MAX_ASSET_BYTES} bytes"
                    )
                archive.writestr(relative, payload)

    size = output.stat().st_size
    if size > MAX_ARCHIVE_BYTES:
        raise RegressionDatasetError(
            f"archive is {size} bytes, above {MAX_ARCHIVE_BYTES}"
        )
    return {
        "manifest_sha256": _sha256_bytes(manifest_blob),
        "zip_sha256": _sha256_file(output),
        "zip_bytes": size,
    }


def _scan_archive(output: Path, manifest: dict) -> dict:
    role_ranges = {
        role: {
            "count": 0,
            "min_peak": math.inf,
            "max_peak": 0.0,
            "min_rms_dbfs": math.inf,
            "max_rms_dbfs": -math.inf,
            "sample_rates": set(),
        }
        for role in SIGNAL_NAMES
    }
    case_stats: dict[str, dict[str, dict[str, float]]] = {}
    noise_hashes: dict[int, dict[str, str]] = defaultdict(dict)
    out_of_band_rates: dict[int, int] = defaultdict(int)
    receive_alignment_rates: dict[int, int] = defaultdict(int)
    uplifted_rates: dict[int, int] = defaultdict(int)
    base_rate_counts: dict[int, int] = defaultdict(int)
    out_of_band_cases = []
    uplifted_cases = []
    uncompressed_bytes = 0
    asset_count = 0

    with zipfile.ZipFile(output, "r") as archive:
        names = set(archive.namelist())
        expected_names = {
            case["signals"][role]
            for case in manifest["cases"]
            for role in SIGNAL_NAMES
        }
        if names != expected_names | {"manifest.json"}:
            raise RegressionDatasetError("ZIP member set differs from the manifest")
        for info in archive.infolist():
            if info.is_dir():
                raise RegressionDatasetError("regression ZIP must not contain directories")
            uncompressed_bytes += info.file_size
            if info.file_size > MAX_ASSET_BYTES:
                raise RegressionDatasetError(
                    f"ZIP member {info.filename} exceeds {MAX_ASSET_BYTES} bytes"
                )

        for case in manifest["cases"]:
            conditions = case["conditions"]
            base_rate = _sample_rate_for_mode(str(conditions["mode"]))
            expected_rate = _sample_rate_for_case(case)
            is_out_of_band = "out_of_band" in _case_families(case)
            is_receive_alignment = _requires_receive_level_alignment(case)
            if is_out_of_band:
                bandwidth = str(conditions["mode"]).split("-", 1)[0].upper()
                required_rate = OUT_OF_BAND_MIN_RATE[bandwidth]
                if expected_rate < required_rate:
                    raise RegressionDatasetError(
                        f"{case['id']} out-of-band rate {expected_rate} "
                        f"is below required {required_rate}"
                    )
                out_of_band_rates[expected_rate] += 1
                out_of_band_cases.append(
                    {
                        "case_id": case["id"],
                        "mode": conditions["mode"],
                        "sample_rate": expected_rate,
                    }
                )
            if is_receive_alignment:
                bandwidth = str(conditions["mode"]).split("-", 1)[0].upper()
                required_rate = OUT_OF_BAND_MIN_RATE[bandwidth]
                if expected_rate != required_rate:
                    raise RegressionDatasetError(
                        f"{case['id']} receive level-alignment rate "
                        f"{expected_rate} != required {required_rate}"
                    )
                receive_alignment_rates[expected_rate] += 1
            if not is_out_of_band and not is_receive_alignment:
                if expected_rate != base_rate:
                    raise RegressionDatasetError(
                        f"{case['id']} unchanged-rate case moved from "
                        f"{base_rate} to {expected_rate}"
                    )
            if expected_rate != base_rate:
                uplifted_rates[expected_rate] += 1
                reasons = []
                if is_out_of_band:
                    reasons.append("out_of_band")
                if is_receive_alignment:
                    reasons.append("receive_level_alignment")
                uplifted_cases.append(
                    {
                        "case_id": case["id"],
                        "mode": conditions["mode"],
                        "base_sample_rate": base_rate,
                        "sample_rate": expected_rate,
                        "reasons": reasons,
                    }
                )
            else:
                base_rate_counts[expected_rate] += 1
            stats_for_case: dict[str, dict[str, float]] = {}
            for role in SIGNAL_NAMES:
                relative = case["signals"][role]
                payload = archive.read(relative)
                try:
                    with sf.SoundFile(BytesIO(payload)) as audio:
                        rate = int(audio.samplerate)
                        channels = int(audio.channels)
                        frames = int(audio.frames)
                        values = audio.read(dtype="float64", always_2d=True)[:, 0]
                except (OSError, RuntimeError) as error:
                    raise RegressionDatasetError(
                        f"cannot decode generated WAV {relative}: {error}"
                    ) from error
                duration = frames / rate
                peak = _peak(values)
                rms = _rms(values)
                if channels != 1:
                    raise RegressionDatasetError(
                        f"{relative} must be mono, got {channels} channels"
                    )
                if rate != expected_rate:
                    raise RegressionDatasetError(
                        f"{relative} rate {rate} != expected {expected_rate}"
                    )
                if duration <= 10.0:
                    raise RegressionDatasetError(
                        f"{relative} duration {duration:.6f} is not above 10 s"
                    )
                if not np.isfinite(values).all() or peak >= PEAK_LIMIT:
                    raise RegressionDatasetError(
                        f"{relative} is non-finite or peak {peak:.9f} >= {PEAK_LIMIT}"
                    )
                if rms <= 1e-10:
                    raise RegressionDatasetError(f"{relative} is silent")

                row = role_ranges[role]
                row["count"] += 1
                row["min_peak"] = min(row["min_peak"], peak)
                row["max_peak"] = max(row["max_peak"], peak)
                row["min_rms_dbfs"] = min(row["min_rms_dbfs"], _dbfs(rms))
                row["max_rms_dbfs"] = max(row["max_rms_dbfs"], _dbfs(rms))
                row["sample_rates"].add(rate)
                stats_for_case[role] = {"peak": peak, "rms": rms}
                asset_count += 1

                if (
                    role == "noise"
                    and conditions["volume"] == "nominal"
                    and conditions["input_level_db"] == 0
                ):
                    noise_hashes[expected_rate].setdefault(
                        str(conditions["scene"]),
                        _sha256_bytes(payload),
                    )
            case_stats[case["id"]] = stats_for_case

    if asset_count != EXPECTED_ASSETS:
        raise RegressionDatasetError(
            f"scanned {asset_count} assets, expected {EXPECTED_ASSETS}"
        )
    if len(out_of_band_cases) != EXPECTED_OUT_OF_BAND_CASES:
        raise RegressionDatasetError(
            f"found {len(out_of_band_cases)} out-of-band cases, "
            f"expected {EXPECTED_OUT_OF_BAND_CASES}"
        )
    if sum(receive_alignment_rates.values()) != EXPECTED_RECEIVE_LEVEL_ALIGNMENT_CASES:
        raise RegressionDatasetError(
            f"found {sum(receive_alignment_rates.values())} receive "
            "level-alignment cases, expected "
            f"{EXPECTED_RECEIVE_LEVEL_ALIGNMENT_CASES}"
        )
    if len(uplifted_cases) != EXPECTED_UPLIFTED_CASES:
        raise RegressionDatasetError(
            f"found {len(uplifted_cases)} uplifted cases, "
            f"expected {EXPECTED_UPLIFTED_CASES}"
        )
    if uncompressed_bytes > MAX_TOTAL_UNCOMPRESSED_BYTES:
        raise RegressionDatasetError(
            f"uncompressed payload {uncompressed_bytes} exceeds "
            f"{MAX_TOTAL_UNCOMPRESSED_BYTES}"
        )
    for rate, hashes in sorted(noise_hashes.items()):
        if set(hashes) != set(SCENE_PROFILES):
            raise RegressionDatasetError(
                f"rate {rate} lacks one or more scene noise fingerprints"
            )
        if len(set(hashes.values())) != len(SCENE_PROFILES):
            raise RegressionDatasetError(
                f"rate {rate} reuses noise across scenes"
            )

    normalized_ranges = {
        role: {
            "count": values["count"],
            "min_peak": values["min_peak"],
            "max_peak": values["max_peak"],
            "min_rms_dbfs": values["min_rms_dbfs"],
            "max_rms_dbfs": values["max_rms_dbfs"],
            "sample_rates": sorted(values["sample_rates"]),
        }
        for role, values in role_ranges.items()
    }
    return {
        "asset_count": asset_count,
        "uncompressed_bytes": uncompressed_bytes,
        "signal_ranges": normalized_ranges,
        "case_stats": case_stats,
        "scene_noise_sha256": {
            str(rate): dict(sorted(hashes.items()))
            for rate, hashes in sorted(noise_hashes.items())
        },
        "sample_rate_plan": {
            "out_of_band_case_count": len(out_of_band_cases),
            "out_of_band_rate_counts": {
                str(rate): count
                for rate, count in sorted(out_of_band_rates.items())
            },
            "receive_level_alignment_case_count": sum(
                receive_alignment_rates.values()
            ),
            "receive_level_alignment_rate_counts": {
                str(rate): count
                for rate, count in sorted(receive_alignment_rates.items())
            },
            "uplifted_case_count": len(uplifted_cases),
            "uplifted_rate_counts": {
                str(rate): count
                for rate, count in sorted(uplifted_rates.items())
            },
            "base_rate_case_counts": {
                str(rate): count
                for rate, count in sorted(base_rate_counts.items())
            },
            "out_of_band_cases": out_of_band_cases,
            "uplifted_cases": uplifted_cases,
        },
    }


def _validate_level_variants(manifest: dict, case_stats: dict) -> dict:
    nominal_peaks: dict[tuple[int, str], dict[str, float]] = {}
    nominal_cases: dict[tuple[int, str], str] = {}
    for case in manifest["cases"]:
        if _condition_gain_db(case["conditions"]) != 0.0:
            continue
        key = (_sample_rate_for_case(case), str(case["conditions"]["scene"]))
        peaks = {
            role: case_stats[case["id"]][role]["peak"]
            for role in SIGNAL_NAMES
        }
        existing = nominal_peaks.get(key)
        if existing is None:
            nominal_peaks[key] = peaks
            nominal_cases[key] = case["id"]
            continue
        for role in SIGNAL_NAMES:
            difference = abs(20.0 * math.log10(peaks[role] / existing[role]))
            if difference > 0.02:
                raise RegressionDatasetError(
                    f"nominal peak mismatch for rate={key[0]} scene={key[1]} "
                    f"role={role}: {difference:.4f} dB"
                )

    seen: set[tuple[str, object]] = set()
    measurements = []
    for case in manifest["cases"]:
        for field in ("volume", "input_level_db"):
            value = case["conditions"][field]
            nominal_value = "nominal" if field == "volume" else 0
            if value == nominal_value:
                continue
            rate = _sample_rate_for_case(case)
            scene = str(case["conditions"]["scene"])
            key = (rate, scene)
            base_peaks = nominal_peaks.get(key)
            if base_peaks is None:
                raise RegressionDatasetError(
                    f"{field}={value!r} has no nominal peak baseline at "
                    f"rate={rate} scene={scene}"
                )
            expected_db = _condition_gain_db(case["conditions"])
            role_measurements = {}
            for role in SIGNAL_NAMES:
                measured_db = 20.0 * math.log10(
                    case_stats[case["id"]][role]["peak"] / base_peaks[role]
                )
                if abs(measured_db - expected_db) > 0.20:
                    raise RegressionDatasetError(
                        f"{field}={value!r} {role} gain is {measured_db:.4f} dB, "
                        f"expected {expected_db:.4f} dB"
                    )
                role_measurements[role] = measured_db
            seen.add((field, value))
            measurements.append(
                {
                    "field": field,
                    "value": value,
                    "case_id": case["id"],
                    "nominal_case_id": nominal_cases[key],
                    "sample_rate": rate,
                    "expected_db": expected_db,
                    "measured_db": role_measurements,
                }
            )

    missing = REQUIRED_LEVEL_VARIANTS - seen
    if missing:
        raise RegressionDatasetError(
            "level variation coverage is incomplete: "
            + ", ".join(f"{field}={value!r}" for field, value in sorted(missing))
        )
    return {
        "verified_variants": sorted(
            ({"field": field, "value": value} for field, value in seen),
            key=lambda row: (row["field"], str(row["value"])),
        ),
        "measurements": measurements,
    }


def _default_output() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return PROJECT_ROOT / "build" / f"{DATASET_NAME}-{stamp}.zip"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create and import-validate the deterministic 106-case/336-item "
            "software regression dataset."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "exclusive ZIP output path; defaults to "
            "build/software-regression-not-certification-<UTC timestamp>.zip"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output = (args.output or _default_output()).expanduser().resolve()
    if output.exists() or output.is_symlink():
        print(f"error: output already exists: {output}", file=sys.stderr)
        return 2
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        manifest = _manifest()
        receive_level_alignment = _validate_receive_level_alignment(manifest)
        estimated_bytes = _estimated_uncompressed_bytes(manifest)
        _preflight_disk(output, estimated_bytes)
        reference_16k, far_end_16k, source_summary = _validate_sources()
        archive_summary = _write_archive(
            output,
            manifest,
            reference_16k,
            far_end_16k,
        )
        scan = _scan_archive(output, manifest)
        level_verification = _validate_level_variants(
            manifest,
            scan.pop("case_stats"),
        )
        imported = import_dataset(output.read_bytes())
        if imported["case_count"] != EXPECTED_CASES:
            raise RegressionDatasetError(
                f"importer returned {imported['case_count']} cases"
            )
        if imported["item_count"] != EXPECTED_ITEMS:
            raise RegressionDatasetError(
                f"importer returned {imported['item_count']} items"
            )
        if imported["name"] != DATASET_NAME:
            raise RegressionDatasetError(
                f"importer returned unexpected name {imported['name']!r}"
            )
        stored = get_dataset(imported["dataset_id"])
        if stored is None:
            raise RegressionDatasetError(
                "imported dataset failed the persisted index integrity check"
            )
        assets_sha256 = stored.get("assets_sha256")
        if (
            not isinstance(assets_sha256, str)
            or len(assets_sha256) != 64
            or any(character not in "0123456789abcdef" for character in assets_sha256)
        ):
            raise RegressionDatasetError(
                f"persisted dataset has invalid assets_sha256: {assets_sha256!r}"
            )
        if stored.get("integrity", {}).get("complete") is not True:
            raise RegressionDatasetError(
                "persisted dataset integrity is not complete"
            )

        report = {
            "status": "ok",
            "output": str(output),
            "zip_bytes": archive_summary["zip_bytes"],
            "zip_sha256": archive_summary["zip_sha256"],
            "manifest_sha256": archive_summary["manifest_sha256"],
            "assets_sha256": assets_sha256,
            "dataset_name": DATASET_NAME,
            "scope": "software regression; not certification",
            "source": source_summary,
            "contract": {
                "case_count": len(manifest["cases"]),
                "item_count": sum(
                    len(case["item_ids"]) for case in manifest["cases"]
                ),
                "asset_count": scan["asset_count"],
                "uncompressed_bytes": scan["uncompressed_bytes"],
                "duration_s": DURATION_SECONDS,
                "peak_limit": PEAK_LIMIT,
            },
            "signal_ranges": scan["signal_ranges"],
            "sample_rate_plan": scan["sample_rate_plan"],
            "receive_level_alignment": receive_level_alignment,
            "scene_noise_sha256": scan["scene_noise_sha256"],
            "level_verification": level_verification,
            "import_summary": imported,
            "stored_dataset": {
                "asset_count": stored["asset_count"],
                "total_bytes": stored["total_bytes"],
                "integrity": stored["integrity"],
            },
        }
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except OSError as error:
        if error.errno == errno.ENOSPC:
            print(
                "error: ENOSPC while creating the regression dataset; "
                "stopped immediately and preserved every existing/partial file: "
                f"{error}",
                file=sys.stderr,
            )
            return 3
        raise
    except RegressionDatasetError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
