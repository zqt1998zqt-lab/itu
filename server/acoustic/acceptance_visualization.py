"""Content-addressed waveform and Mel visualization caches."""
from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any

import numpy as np
import soundfile as sf

from . import service


SCHEMA = "scoped-acoustic-visual-v1"
WAVEFORM_BINS = 720
MEL_BANDS = 40
MEL_FRAMES = 240
MEL_FFT_SIZE = 512
MEL_HOP_MS = 10
MEL_DYNAMIC_RANGE_DB = 80
MEL_DB_OFFSET_DB = -200.0
MEL_DB_STEP_DB = 0.1

_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_BASE64 = re.compile(
    r"^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$"
)
_FIELDS = frozenset({
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
})
_WAVEFORM_I16_BYTES = WAVEFORM_BINS * 2
_MEL_I16_BYTES = MEL_BANDS * MEL_FRAMES * 2


class VisualizationCacheError(ValueError):
    """A visualization source or cache entry failed validation."""


def cache_root() -> Path:
    """Return the cache directory for the currently configured build root."""
    return Path(service.BUILD) / "acceptance-visual-cache" / SCHEMA


def _safe_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise VisualizationCacheError("asset path must be a non-empty relative path")
    if "\x00" in value or "\\" in value:
        raise VisualizationCacheError("asset path contains an invalid character")
    if value.startswith("/"):
        raise VisualizationCacheError("asset path must not be absolute")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise VisualizationCacheError("asset path contains an unsafe component")
    if any(any(ord(character) < 32 for character in part) for part in parts):
        raise VisualizationCacheError("asset path contains a control character")
    path = PurePosixPath(value)
    if path.is_absolute() or path.suffix.lower() != ".wav":
        raise VisualizationCacheError("asset path must name a relative WAV file")
    return path.as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _base64_i16(values: np.ndarray) -> str:
    payload = np.ascontiguousarray(values, dtype="<i2").tobytes()
    return base64.b64encode(payload).decode("ascii")


def _waveform_envelope(
    samples: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    frames, channels = samples.shape
    minimum = np.full(WAVEFORM_BINS, np.inf, dtype=np.float64)
    maximum = np.full(WAVEFORM_BINS, -np.inf, dtype=np.float64)
    source_bins = np.minimum(
        WAVEFORM_BINS - 1,
        (np.arange(frames, dtype=np.int64) * WAVEFORM_BINS) // frames,
    )
    for channel in range(channels):
        channel_samples = samples[:, channel]
        np.minimum.at(minimum, source_bins, channel_samples)
        np.maximum.at(maximum, source_bins, channel_samples)
    filled = np.isfinite(minimum)
    minimum[~filled] = 0.0
    maximum[~filled] = 0.0
    peak = float(max(
        np.max(np.abs(minimum), initial=0.0),
        np.max(np.abs(maximum), initial=0.0),
    ))
    scale = max(1.0, peak)
    quantized_minimum = np.clip(
        np.rint(minimum / scale * 32767.0),
        -32768,
        32767,
    ).astype("<i2")
    quantized_maximum = np.clip(
        np.rint(maximum / scale * 32767.0),
        -32768,
        32767,
    ).astype("<i2")
    return peak, scale, quantized_minimum, quantized_maximum


def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (np.power(10.0, mel / 2595.0) - 1.0)


def _mel_filterbank(sample_rate: int) -> np.ndarray:
    maximum_hz = min(8_000.0, sample_rate / 2.0)
    mel_start = float(_hz_to_mel(20.0))
    mel_end = float(_hz_to_mel(maximum_hz))
    edges = _mel_to_hz(
        mel_start
        + (mel_end - mel_start)
        * np.arange(MEL_BANDS + 2, dtype=np.float64)
        / (MEL_BANDS + 1)
    )
    bin_hz = sample_rate / MEL_FFT_SIZE
    bank = np.zeros((MEL_FFT_SIZE // 2 + 1, MEL_BANDS), dtype=np.float64)
    for band in range(MEL_BANDS):
        left, center, right = edges[band : band + 3]
        for bin_index in range(1, MEL_FFT_SIZE // 2 + 1):
            frequency = bin_index * bin_hz
            if frequency < left or frequency > right:
                continue
            if frequency <= center:
                weight = (
                    (frequency - left)
                    / max(center - left, np.finfo(np.float64).eps)
                )
            else:
                weight = (
                    (right - frequency)
                    / max(right - center, np.finfo(np.float64).eps)
                )
            if weight > 0.0:
                bank[bin_index, band] = weight
    return bank


def _pool_mel(values: np.ndarray) -> np.ndarray:
    frame_count = values.shape[1]
    pooled = np.empty((MEL_BANDS, MEL_FRAMES), dtype=np.float64)
    for column in range(MEL_FRAMES):
        start = math.floor(column * frame_count / MEL_FRAMES)
        end = max(
            start + 1,
            math.floor((column + 1) * frame_count / MEL_FRAMES),
        )
        pooled[:, column] = np.max(values[:, start:end], axis=1)
    return pooled


def _mel_spectrogram(
    samples: np.ndarray,
    sample_rate: int,
) -> tuple[float, np.ndarray]:
    hop = max(1, math.floor(sample_rate * MEL_HOP_MS / 1000.0 + 0.5))
    frame_count = max(1, math.ceil(len(samples) / hop))
    window = (
        0.5
        - 0.5
        * np.cos(
            2.0
            * np.pi
            * np.arange(MEL_FFT_SIZE, dtype=np.float64)
            / (MEL_FFT_SIZE - 1)
        )
    )
    window_energy = float(np.sum(window * window))
    mixed = np.mean(samples, axis=1, dtype=np.float64)
    bank = _mel_filterbank(sample_rate)
    values = np.empty((MEL_BANDS, frame_count), dtype=np.float64)
    sample_offsets = np.arange(MEL_FFT_SIZE, dtype=np.int64)

    for start_frame in range(0, frame_count, 256):
        stop_frame = min(frame_count, start_frame + 256)
        starts = np.arange(start_frame, stop_frame, dtype=np.int64) * hop
        indices = starts[:, None] + sample_offsets[None, :]
        valid = indices < len(mixed)
        block = np.zeros((stop_frame - start_frame, MEL_FFT_SIZE))
        block[valid] = mixed[indices[valid]]
        spectrum = np.fft.rfft(block * window[None, :], axis=1)
        power = (
            spectrum.real * spectrum.real
            + spectrum.imag * spectrum.imag
        ) @ bank
        values[:, start_frame:stop_frame] = (
            10.0
            * np.log10(np.maximum(power / window_energy, 1e-20)).T
        )

    pooled = _pool_mel(values)
    ceiling = float(np.max(pooled))
    quantized = np.clip(
        np.rint(
            (pooled - MEL_DB_OFFSET_DB)
            / MEL_DB_STEP_DB
        ),
        -32768,
        32767,
    ).astype("<i2")
    return ceiling, quantized


def _compute_visualization(path: Path, source_sha256: str) -> dict[str, Any]:
    try:
        with sf.SoundFile(path) as audio:
            if audio.format != "WAV":
                raise VisualizationCacheError(f"{path}: source is not a WAV file")
            sample_rate = int(audio.samplerate)
            channels = int(audio.channels)
            frames = int(audio.frames)
            if (
                not 8_000 <= sample_rate <= 384_000
                or channels < 1
                or frames < 1
            ):
                raise VisualizationCacheError(
                    f"{path}: WAV sample rate, channels, or duration is invalid"
                )
            samples = audio.read(dtype="float64", always_2d=True)
    except VisualizationCacheError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise VisualizationCacheError(f"cannot decode WAV {path}: {error}") from error

    if samples.shape != (frames, channels):
        raise VisualizationCacheError(f"{path}: decoded WAV shape is inconsistent")
    if not np.isfinite(samples).all():
        raise VisualizationCacheError(f"{path}: WAV contains non-finite samples")

    peak, scale, wave_min, wave_max = _waveform_envelope(samples)
    mel_ceiling, mel_values = _mel_spectrogram(samples, sample_rate)
    return {
        "schema": SCHEMA,
        "source_sha256": source_sha256,
        "waveform_bins": WAVEFORM_BINS,
        "mel_bands": MEL_BANDS,
        "mel_frames": MEL_FRAMES,
        "mel_fft_size": MEL_FFT_SIZE,
        "mel_hop_ms": MEL_HOP_MS,
        "mel_dynamic_range_db": MEL_DYNAMIC_RANGE_DB,
        "mel_db_offset_db": MEL_DB_OFFSET_DB,
        "mel_db_step_db": MEL_DB_STEP_DB,
        "duration_s": frames / sample_rate,
        "sample_rate": sample_rate,
        "channels": channels,
        "peak": peak,
        "wave_scale": scale,
        "wave_min_i16": _base64_i16(wave_min),
        "wave_max_i16": _base64_i16(wave_max),
        "mel_i16": _base64_i16(mel_values),
        "mel_ceiling_db": mel_ceiling,
    }


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VisualizationCacheError(
                f"cache contains duplicate JSON key {key!r}"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise VisualizationCacheError(f"cache contains non-finite number {value}")


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise VisualizationCacheError(f"cache field {label} must be an integer")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VisualizationCacheError(f"cache field {label} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise VisualizationCacheError(f"cache field {label} must be finite")
    return numeric


def _decode_i16(value: Any, expected_bytes: int, label: str) -> np.ndarray:
    if not isinstance(value, str) or _BASE64.fullmatch(value) is None:
        raise VisualizationCacheError(f"cache field {label} is not valid base64")
    try:
        payload = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as error:
        raise VisualizationCacheError(
            f"cache field {label} is not valid base64"
        ) from error
    if len(payload) != expected_bytes:
        raise VisualizationCacheError(
            f"cache field {label} has {len(payload)} bytes; "
            f"expected {expected_bytes}"
        )
    return np.frombuffer(payload, dtype="<i2")


def _validate_cache(value: Any, expected_sha256: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise VisualizationCacheError("cache JSON must contain an object")
    if set(value) != _FIELDS:
        missing = sorted(_FIELDS - set(value))
        extra = sorted(set(value) - _FIELDS)
        raise VisualizationCacheError(
            f"cache fields differ (missing={missing}, extra={extra})"
        )
    if value["schema"] != SCHEMA:
        raise VisualizationCacheError("cache schema differs")
    if value["source_sha256"] != expected_sha256:
        raise VisualizationCacheError("cache source SHA-256 differs")

    expected_integers = {
        "waveform_bins": WAVEFORM_BINS,
        "mel_bands": MEL_BANDS,
        "mel_frames": MEL_FRAMES,
        "mel_fft_size": MEL_FFT_SIZE,
        "mel_hop_ms": MEL_HOP_MS,
    }
    for field, expected in expected_integers.items():
        if _integer(value[field], field) != expected:
            raise VisualizationCacheError(f"cache parameter {field} differs")
    expected_numbers = {
        "mel_dynamic_range_db": float(MEL_DYNAMIC_RANGE_DB),
        "mel_db_offset_db": MEL_DB_OFFSET_DB,
        "mel_db_step_db": MEL_DB_STEP_DB,
    }
    for field, expected in expected_numbers.items():
        if _finite_number(value[field], field) != expected:
            raise VisualizationCacheError(f"cache parameter {field} differs")

    duration = _finite_number(value["duration_s"], "duration_s")
    sample_rate = _integer(value["sample_rate"], "sample_rate")
    channels = _integer(value["channels"], "channels")
    peak = _finite_number(value["peak"], "peak")
    scale = _finite_number(value["wave_scale"], "wave_scale")
    ceiling = _finite_number(value["mel_ceiling_db"], "mel_ceiling_db")
    if duration <= 0.0:
        raise VisualizationCacheError("cache duration_s must be positive")
    if not 8_000 <= sample_rate <= 384_000:
        raise VisualizationCacheError("cache sample_rate is outside the WAV contract")
    if channels < 1:
        raise VisualizationCacheError("cache channels must be positive")
    if peak < 0.0 or scale < 1.0:
        raise VisualizationCacheError("cache waveform scale is invalid")
    if ceiling < MEL_DB_OFFSET_DB:
        raise VisualizationCacheError("cache Mel ceiling is below its sentinel")

    _decode_i16(value["wave_min_i16"], _WAVEFORM_I16_BYTES, "wave_min_i16")
    _decode_i16(value["wave_max_i16"], _WAVEFORM_I16_BYTES, "wave_max_i16")
    _decode_i16(value["mel_i16"], _MEL_I16_BYTES, "mel_i16")
    return value


def _load_cache(path: Path, expected_sha256: str) -> dict[str, Any]:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise VisualizationCacheError(f"cannot inspect cache {path}: {error}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise VisualizationCacheError(f"cache entry is not a regular file: {path}")
    try:
        payload = path.read_text(encoding="utf-8")
        value = json.loads(
            payload,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except VisualizationCacheError:
        raise
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise VisualizationCacheError(f"cannot read cache {path}: {error}") from error
    return _validate_cache(value, expected_sha256)


def _canonical_cache_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise VisualizationCacheError(
            f"cannot serialize visualization cache: {error}"
        ) from error


def _create_cache(path: Path, value: Mapping[str, Any]) -> bool:
    """Create a new cache entry without replacing an existing file."""
    payload = _canonical_cache_bytes(value)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError:
        return False
    except OSError as error:
        raise VisualizationCacheError(f"cannot create cache {path}: {error}") from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        raise VisualizationCacheError(f"cannot write cache {path}: {error}") from error
    return True


def load_or_build_visualization(
    path: Path,
    expected_sha256: str,
    *,
    verify_source: bool = True,
) -> dict:
    """Load one strict cache entry, or create it from a validated WAV source."""
    source = Path(path)
    if not isinstance(expected_sha256, str) or _SHA256.fullmatch(expected_sha256) is None:
        raise VisualizationCacheError("expected source SHA-256 must be 64 lowercase hex digits")

    root = cache_root()
    destination = root / f"{expected_sha256}.json"
    if destination.exists() or destination.is_symlink():
        return _load_cache(destination, expected_sha256)

    if verify_source:
        try:
            actual_sha256 = _sha256_file(source)
        except OSError as error:
            raise VisualizationCacheError(
                f"cannot hash visualization source {source}: {error}"
            ) from error
        if actual_sha256 != expected_sha256:
            raise VisualizationCacheError(
                f"visualization source SHA-256 differs for {source}"
            )

    value = _compute_visualization(source, expected_sha256)
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise VisualizationCacheError(f"cannot create cache root {root}: {error}") from error
    if root.is_symlink() or not root.is_dir():
        raise VisualizationCacheError(f"cache root is not a regular directory: {root}")

    created = _create_cache(destination, value)
    if not created:
        return _load_cache(destination, expected_sha256)
    return _load_cache(destination, expected_sha256)


def precompute_visualizations(
    assets_root: Path,
    assets: Mapping[str, Mapping[str, Any]],
) -> int:
    """Precompute one cache entry for every unique declared WAV asset."""
    try:
        root = Path(assets_root).resolve(strict=True)
    except OSError as error:
        raise VisualizationCacheError(
            f"cannot resolve visualization assets root {assets_root}: {error}"
        ) from error
    if root.is_symlink() or not root.is_dir():
        raise VisualizationCacheError(
            f"visualization assets root is not a regular directory: {root}"
        )
    if not isinstance(assets, Mapping):
        raise VisualizationCacheError("visualization assets must be a mapping")

    count = 0
    for relative in sorted(assets):
        try:
            normalized = _safe_relative_path(relative)
            metadata = assets[relative]
            if not isinstance(metadata, Mapping):
                raise VisualizationCacheError(
                    f"{normalized}: asset metadata must be a mapping"
                )
            source_sha256 = metadata.get("sha256")
            if (
                not isinstance(source_sha256, str)
                or _SHA256.fullmatch(source_sha256) is None
            ):
                raise VisualizationCacheError(
                    f"{normalized}: asset SHA-256 is invalid"
                )
            source = (root / normalized).resolve(strict=True)
            if not source.is_relative_to(root) or not source.is_file():
                raise VisualizationCacheError(
                    f"{normalized}: asset escapes or is not a regular file"
                )
            load_or_build_visualization(
                source,
                source_sha256,
                verify_source=False,
            )
        except VisualizationCacheError as error:
            raise VisualizationCacheError(
                f"cannot precompute visualization for {relative!r}: {error}"
            ) from error
        except OSError as error:
            raise VisualizationCacheError(
                f"cannot precompute visualization for {relative!r}: {error}"
            ) from error
        count += 1
    return count


__all__ = [
    "MEL_BANDS",
    "MEL_DB_OFFSET_DB",
    "MEL_DB_STEP_DB",
    "MEL_DYNAMIC_RANGE_DB",
    "MEL_FFT_SIZE",
    "MEL_FRAMES",
    "MEL_HOP_MS",
    "SCHEMA",
    "WAVEFORM_BINS",
    "VisualizationCacheError",
    "cache_root",
    "load_or_build_visualization",
    "precompute_visualizations",
]
