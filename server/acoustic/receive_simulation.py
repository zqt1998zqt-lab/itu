"""Software-only binaural receive-path simulation.

The outputs from this module are deterministic engineering proxies. They are
not HATS/DRP measurements, an HRTF implementation, or evidence of ITU-T
P.57/P.58 or P.1100/P.1110/P.1120 compliance.
"""
from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import re
import tempfile
from threading import Lock
from typing import Any
import uuid

import numpy as np
from scipy import signal
import soundfile as sf


STORE = Path(__file__).resolve().parents[2] / "build" / "synthetic-receive"
MAX_BYTES = 32 * 1024 * 1024
MAX_DURATION_S = 120.0
MAX_IR_DURATION_S = 10.0

ALLOWED_MODES = frozenset({
    "NB-wired",
    "NB-wireless",
    "NB-bt",
    "WB-wired",
    "WB-wireless",
    "WB-bt",
    "SWB-wired",
    "SWB-wireless",
})
SPEAKER_SAMPLE_RATES = frozenset({16000, 24000, 48000})
IR_SAMPLE_RATES = frozenset({
    8000, 11025, 12000, 16000, 22050, 24000, 32000,
    44100, 48000, 88200, 96000, 176400, 192000,
})
ALLOWED_FILE_NAMES = frozenset({
    "virtual_left.wav",
    "virtual_right.wav",
    "report.json",
})

_LINEAR_WAV_SUBTYPES = frozenset({
    "PCM_S8",
    "PCM_U8",
    "PCM_16",
    "PCM_24",
    "PCM_32",
    "FLOAT",
    "DOUBLE",
})
_THIRD_OCTAVE_CENTERS_HZ = (
    125,
    160,
    200,
    250,
    315,
    400,
    500,
    630,
    800,
    1000,
    1250,
    1600,
    2000,
    2500,
    3150,
    4000,
    5000,
    6300,
    8000,
    10000,
    12500,
    16000,
)
_DB_FLOOR = -120.0
_ZERO_POWER = 1e-30
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_WRITE_LOCK = Lock()
_IR_RESAMPLING_VERSION = "fir_coefficient_resample_poly_v1"
_FRACTIONAL_DELAY_TAPS = 129
_FRACTIONAL_DELAY_KAISER_BETA = 8.6
_HEAD_MODEL_VERSION = "woodworth_folded_split_delay_unity_dc_shelf_v2"
_HEAD_HIGH_FREQUENCY_ATTENUATION_DB = 12.0
_MIN_HEAD_RADIUS_M = 0.04
_MAX_HEAD_RADIUS_M = 0.15
_MIN_SPEED_OF_SOUND_MPS = 300.0
_MAX_SPEED_OF_SOUND_MPS = 380.0


class SyntheticReceiveError(ValueError):
    """The requested software-only receive simulation is unsafe or invalid."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: dict[str, Any]) -> str:
    data = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(data)


def _clean_name(name: str | None, fallback: str) -> str:
    if name is None:
        return fallback
    if not isinstance(name, str):
        raise SyntheticReceiveError("audio names must be strings")
    candidate = re.split(r"[/\\]", name.strip())[-1].replace("\x00", "")
    if not candidate:
        return fallback
    if len(candidate) > 255:
        raise SyntheticReceiveError("audio file name is too long")
    return candidate


def _coerce_bytes(blob: bytes, label: str) -> bytes:
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise SyntheticReceiveError(f"{label} must be supplied as bytes")
    data = bytes(blob)
    if not data:
        raise SyntheticReceiveError(f"{label} is empty")
    if len(data) > MAX_BYTES:
        raise SyntheticReceiveError(
            f"{label} exceeds the {MAX_BYTES}-byte upload limit"
        )
    return data


def _decode_linear_wav(
    *,
    name: str | None,
    blob: bytes,
    fallback_name: str,
    label: str,
    allowed_rates: frozenset[int] | None,
    max_duration_s: float,
    reject_clipping: bool,
) -> dict[str, Any]:
    data = _coerce_bytes(blob, label)
    try:
        info = sf.info(BytesIO(data))
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise SyntheticReceiveError(f"{label} is not a readable audio file: {error}") from error

    if info.format != "WAV":
        raise SyntheticReceiveError(f"{label} must use the WAV container")
    if info.subtype not in _LINEAR_WAV_SUBTYPES:
        raise SyntheticReceiveError(
            f"{label} must use uncompressed linear PCM or IEEE-float WAV samples"
        )
    if info.channels != 1:
        raise SyntheticReceiveError(
            f"{label} must be mono; received {info.channels} channels"
        )
    rate = int(info.samplerate)
    if rate <= 0:
        raise SyntheticReceiveError(f"{label} has an invalid sample rate")
    if allowed_rates is not None and rate not in allowed_rates:
        raise SyntheticReceiveError(
            f"{label} sample rate must be one of 16000, 24000, or 48000 Hz"
        )
    frames = int(info.frames)
    if frames <= 0:
        raise SyntheticReceiveError(f"{label} contains no audio frames")
    duration_s = frames / rate
    if duration_s > max_duration_s:
        raise SyntheticReceiveError(
            f"{label} exceeds the {max_duration_s:g}-second duration limit"
        )

    try:
        samples, decoded_rate = sf.read(
            BytesIO(data),
            dtype="float64",
            always_2d=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise SyntheticReceiveError(f"{label} could not be decoded: {error}") from error
    if int(decoded_rate) != rate or samples.shape[1] != 1:
        raise SyntheticReceiveError(f"{label} metadata changed while decoding")
    mono = np.asarray(samples[:, 0], dtype=np.float64)
    if not len(mono):
        raise SyntheticReceiveError(f"{label} contains no audio frames")
    if not np.all(np.isfinite(mono)):
        raise SyntheticReceiveError(f"{label} contains non-finite samples")
    peak = float(np.max(np.abs(mono)))
    quantisation_tolerance = {
        "PCM_S8": 1.0 / 128.0,
        "PCM_U8": 1.0 / 128.0,
        "PCM_16": 1.0 / 32768.0,
        "PCM_24": 1.0 / 8388608.0,
        "PCM_32": 1.0 / 2147483648.0,
    }.get(info.subtype, 0.0)
    clip_threshold = 0.999 - quantisation_tolerance
    if reject_clipping and peak >= clip_threshold:
        raise SyntheticReceiveError(
            f"{label} is clipped or reaches the digital ceiling "
            f"(peak={peak:.9f} FS; threshold={clip_threshold:.9f})"
        )

    return {
        "name": _clean_name(name, fallback_name),
        "bytes": len(data),
        "sha256": _sha256_bytes(data),
        "sample_rate_hz": rate,
        "channels": 1,
        "frames": len(mono),
        "duration_s": len(mono) / rate,
        "format": info.format,
        "subtype": info.subtype,
        "samples": mono,
    }


def _public_identity(decoded: dict[str, Any]) -> dict[str, Any]:
    return {
        key: decoded[key]
        for key in (
            "name",
            "bytes",
            "sha256",
            "sample_rate_hz",
            "channels",
            "frames",
            "duration_s",
            "format",
            "subtype",
        )
    }


def _resample_ir(ir: dict[str, Any], target_rate: int) -> tuple[np.ndarray, dict[str, Any]]:
    samples = np.asarray(ir["samples"], dtype=np.float64)
    source_rate = int(ir["sample_rate_hz"])
    if source_rate <= 0 or target_rate <= 0:
        raise SyntheticReceiveError("IR sample rates must be positive")
    if source_rate not in IR_SAMPLE_RATES or target_rate not in SPEAKER_SAMPLE_RATES:
        raise SyntheticReceiveError("unsupported IR/processing sample rate; polyphase ratio is bounded")
    if not len(samples):
        raise SyntheticReceiveError("IR contains no samples")

    gain_scale = source_rate / target_rate
    input_padding_samples = 0
    padding_removed_samples = 0
    resampled_padding_origin_samples = 0
    output_guard_samples = 0
    up = 1
    down = 1
    if source_rate != target_rate:
        common = math.gcd(target_rate, source_rate)
        up = target_rate // common
        down = source_rate // common

        # resample_poly's polyphase FIR spans roughly
        # 10 * max(up, down) samples before downsampling. Preserve both
        # edges so a unit IR at sample zero keeps the full filtering tail.
        output_guard_samples = math.ceil(10 * max(up, down) / down) + 3
        required_padding = math.ceil(
            (output_guard_samples + 2) * down / up
        ) + 8
        # A multiple of ``down`` makes the physical origin land exactly on
        # an output sample, so no fractional trim is silently introduced.
        input_padding_samples = (
            (required_padding + down - 1) // down
        ) * down
        padded = np.pad(
            samples,
            (input_padding_samples, input_padding_samples),
            mode="constant",
        )
        try:
            resampled = signal.resample_poly(
                padded,
                up,
                down,
                padtype="constant",
            )
        except (TypeError, ValueError) as error:
            raise SyntheticReceiveError(f"IR resampling failed: {error}") from error

        # Treat the supplied IR as FIR coefficients whose convolution gain
        # must not change with the processing rate. resample_poly changes
        # the discrete coefficient density by target/source; apply the
        # inverse factor. Do not normalise by sum(h): that would erase a
        # deliberately zero-DC or high-pass IR.
        resampled *= gain_scale
        padded_origin = input_padding_samples * up // down
        crop_start = padded_origin - output_guard_samples
        nominal_frames = math.ceil(len(samples) * up / down)
        output_frames = nominal_frames + 2 * output_guard_samples
        if crop_start < 0:
            raise SyntheticReceiveError("IR resampling padding is insufficient")
        crop_end = crop_start + output_frames
        if len(resampled) < crop_end:
            resampled = np.pad(
                resampled,
                (0, crop_end - len(resampled)),
                mode="constant",
            )
        samples = np.asarray(
            resampled[crop_start:crop_end],
            dtype=np.float64,
        )
        padding_removed_samples = crop_start
        resampled_padding_origin_samples = padded_origin

    samples = np.asarray(samples, dtype=np.float64)
    if not len(samples) or not np.all(np.isfinite(samples)):
        raise SyntheticReceiveError("resampled IR is empty or non-finite")
    return samples, {
        "source_sample_rate_hz": source_rate,
        "processing_sample_rate_hz": target_rate,
        "resampled_frames": len(samples),
        "resampled_duration_s": len(samples) / target_rate,
        "resampling": "none" if source_rate == target_rate else "scipy.signal.resample_poly",
        "resampling_version": _IR_RESAMPLING_VERSION,
        "up": up,
        "down": down,
        "fir_gain_scale": _rounded(gain_scale),
        "fir_gain_contract": (
            "discrete FIR convolution gain preserved; coefficient density "
            "compensated by source_rate/target_rate"
        ),
        "input_padding_samples": input_padding_samples,
        "padding_removed_samples": padding_removed_samples,
        "resampled_padding_origin_samples": resampled_padding_origin_samples,
        "output_guard_samples": output_guard_samples,
        "boundary_extension": "zero",
        "ir_time_origin_samples": output_guard_samples,
        "ir_time_origin_s": _rounded(output_guard_samples / target_rate),
        "delay_compensation": (
            "explicit IR time origin; filter-tail latency is reported and "
            "is not silently removed or used as a measurement origin"
        ),
    }


def _align_ir_kernel_origins(
    left: np.ndarray,
    right: np.ndarray,
    left_processing: dict[str, Any],
    right_processing: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Put both supplied IR time origins on the same processing timeline."""
    left_origin = int(left_processing["ir_time_origin_samples"])
    right_origin = int(right_processing["ir_time_origin_samples"])
    aligned_origin = max(left_origin, right_origin)
    left_prefix = aligned_origin - left_origin
    right_prefix = aligned_origin - right_origin
    if left_prefix:
        left = np.pad(left, (left_prefix, 0), mode="constant")
    if right_prefix:
        right = np.pad(right, (right_prefix, 0), mode="constant")
    for processing, prefix in (
        (left_processing, left_prefix),
        (right_processing, right_prefix),
    ):
        processing["pair_alignment_prefix_samples"] = prefix
        processing["aligned_ir_time_origin_samples"] = aligned_origin
        processing["aligned_ir_time_origin_s"] = _rounded(
            aligned_origin / int(processing["processing_sample_rate_hz"])
        )
    return left, right


def _rms_dbfs(samples: np.ndarray) -> float:
    values = np.asarray(samples, dtype=np.float64)
    if not len(values):
        return _DB_FLOOR
    power = float(np.mean(values * values))
    if not math.isfinite(power) or power <= _ZERO_POWER:
        return _DB_FLOOR
    return max(_DB_FLOOR, float(10.0 * math.log10(power)))


def _binaural_signal(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    length = max(len(left), len(right))
    result = np.zeros(length, dtype=np.float64)
    if len(left):
        result[:len(left)] += left * left
    if len(right):
        result[:len(right)] += right * right
    return np.sqrt(result / 2.0)


def _rounded(value: float | None) -> float | None:
    if value is None:
        return None
    rounded = round(float(value), 9)
    return 0.0 if rounded == 0 else rounded


def _woodworth_itd_s(
    azimuth_deg: float,
    head_radius_m: float,
    speed_of_sound_mps: float,
) -> tuple[float, float, float]:
    """Return folded (ITD seconds, lateral angle radians, shadow factor)."""
    absolute_azimuth = min(abs(float(azimuth_deg)), 180.0)
    folded_azimuth = min(absolute_azimuth, 180.0 - absolute_azimuth)
    lateral_rad = math.radians(max(0.0, min(folded_azimuth, 90.0)))
    shadow = math.sin(lateral_rad)
    itd_s = head_radius_m / speed_of_sound_mps * (lateral_rad + shadow)
    return itd_s, lateral_rad, shadow


def _fractional_delay(samples: np.ndarray, delay_samples: float) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float64)
    delay = float(delay_samples)
    if not math.isfinite(delay) or delay < 0:
        raise SyntheticReceiveError(
            "fractional delay must be finite and non-negative"
        )
    if delay_samples <= 1e-12:
        return values.copy()
    if not len(values):
        return values.copy()

    integer_delay, fractional_delay = _split_fractional_delay(delay)
    delayed = np.zeros_like(values)
    if integer_delay >= len(values):
        return delayed

    if fractional_delay <= 1e-12:
        delayed[integer_delay:] = values[:len(values) - integer_delay]
        return delayed

    # Keep the fractional interpolation kernel centred in a fixed causal
    # buffer, then crop the buffer's base delay. This is offline
    # interpolation with explicit zero extension outside the input. The
    # kernel only carries the sub-sample remainder; the integer part is an
    # exact shift so delays beyond one sample never move the window edge.
    centre = _FRACTIONAL_DELAY_TAPS // 2
    taps = np.arange(_FRACTIONAL_DELAY_TAPS, dtype=np.float64)
    kernel = np.sinc(taps - centre - fractional_delay)
    kernel *= np.kaiser(
        _FRACTIONAL_DELAY_TAPS,
        _FRACTIONAL_DELAY_KAISER_BETA,
    )
    kernel /= float(np.sum(kernel))
    convolved = signal.fftconvolve(values, kernel, mode="full")
    interpolated = np.asarray(
        convolved[centre:centre + len(values)],
        dtype=np.float64,
    )
    delayed[integer_delay:] = interpolated[
        :len(values) - integer_delay
    ]
    return delayed


def _split_fractional_delay(delay_samples: float) -> tuple[int, float]:
    delay = float(delay_samples)
    nearest_integer = int(round(delay))
    if abs(delay - nearest_integer) <= 1e-12:
        return nearest_integer, 0.0
    integer_delay = int(math.floor(delay))
    fractional_delay = delay - integer_delay
    if fractional_delay >= 1.0 - 1e-12:
        return integer_delay + 1, 0.0
    return integer_delay, fractional_delay


def _fractional_delay_metadata(delay_samples: float) -> dict[str, Any]:
    integer_delay, fractional_delay = _split_fractional_delay(delay_samples)
    return {
        "implementation": "kaiser_windowed_sinc_offline_interpolation",
        "version": "split_integer_kaiser_sinc_129_v2",
        "taps": _FRACTIONAL_DELAY_TAPS,
        "kaiser_beta": _FRACTIONAL_DELAY_KAISER_BETA,
        "boundary_extension": "zero",
        "delay_samples": _rounded(delay_samples),
        "integer_delay_samples": integer_delay,
        "fractional_delay_samples": _rounded(fractional_delay),
        "kernel_delay_samples": _rounded(fractional_delay),
        "kernel_offset_policy": "fraction_only_0_to_lt_1",
    }


def _head_shelf_parameters(
    shadow: float,
    head_radius_m: float,
    speed_of_sound_mps: float,
) -> tuple[float, float, float]:
    """Return (high-frequency gain, centre Hz, pole Hz)."""
    high_frequency_gain = 10.0 ** (
        -_HEAD_HIGH_FREQUENCY_ATTENUATION_DB * shadow / 20.0
    )
    centre_hz = speed_of_sound_mps / (2.0 * math.pi * head_radius_m)
    square_root_gain = math.sqrt(high_frequency_gain)
    pole_hz = centre_hz * square_root_gain
    return high_frequency_gain, centre_hz, pole_hz


def _first_order_head_shelf(
    samples: np.ndarray,
    rate: int,
    shadow: float,
    head_radius_m: float,
    speed_of_sound_mps: float,
) -> np.ndarray:
    """Apply an uncalibrated unity-DC first-order high-frequency shelf."""
    values = np.asarray(samples, dtype=np.float64)
    if shadow <= 1e-12:
        return values.copy()
    high_frequency_gain, centre_hz, pole_hz = _head_shelf_parameters(
        shadow,
        head_radius_m,
        speed_of_sound_mps,
    )
    zero_hz = centre_hz / math.sqrt(high_frequency_gain)
    zero_rad = 2.0 * math.pi * zero_hz
    pole_rad = 2.0 * math.pi * pole_hz
    numerator, denominator = signal.bilinear(
        np.array([1.0 / zero_rad, 1.0], dtype=np.float64),
        np.array([1.0 / pole_rad, 1.0], dtype=np.float64),
        fs=rate,
    )
    return signal.lfilter(numerator, denominator, values)


def _first_order_head_lowpass(
    samples: np.ndarray,
    rate: int,
    shadow: float,
) -> np.ndarray:
    if shadow <= 1e-12:
        return np.asarray(samples, dtype=np.float64).copy()
    nyquist = rate / 2.0
    cutoff_hz = min(0.95 * nyquist, max(1500.0, 8000.0 - 6000.0 * shadow))
    alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff_hz / rate)
    return signal.lfilter(
        np.array([alpha], dtype=np.float64),
        np.array([1.0, alpha - 1.0], dtype=np.float64),
        np.asarray(samples, dtype=np.float64),
    )


def _synthetic_head_outputs(
    speaker: np.ndarray,
    rate: int,
    azimuth_deg: float,
    head_radius_m: float,
    speed_of_sound_mps: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    speaker = np.asarray(speaker, dtype=np.float64)
    itd_s, lateral_rad, shadow = _woodworth_itd_s(
        azimuth_deg,
        head_radius_m,
        speed_of_sound_mps,
    )
    delay_samples = itd_s * rate
    delay_metadata = _fractional_delay_metadata(delay_samples)
    if shadow <= 1e-12:
        return speaker.copy(), speaker.copy(), {
            "name": "woodworth_folded_shelf_engineering_proxy",
            "model_version": _HEAD_MODEL_VERSION,
            "version": _HEAD_MODEL_VERSION,
            "formula": (
                "theta=min(abs(azimuth), 180-abs(azimuth)); "
                "ITD=(a/c)*(theta+sin(theta)), theta in [0, pi/2]"
            ),
            "approximation": "uncalibrated engineering proxy",
            "source_side": "center",
            "theta_rad": 0.0,
            "head_shadow_fraction": 0.0,
            "itd_ms": 0.0,
            "far_ear": "none",
            "far_ear_attenuation_db": 0.0,
            "far_ear_attenuation_db_semantics": (
                "high-frequency asymptotic attenuation; not broadband"
            ),
            "high_frequency_attenuation_db": 0.0,
            "far_ear_lowpass_hz": None,
            "far_ear_shelf_cutoff_hz": None,
            "filter_type": "identity",
            "fractional_delay": delay_metadata,
        }

    delayed = _fractional_delay(speaker, delay_samples)
    far = _first_order_head_shelf(
        delayed,
        rate,
        shadow,
        head_radius_m,
        speed_of_sound_mps,
    )
    attenuation_db = _HEAD_HIGH_FREQUENCY_ATTENUATION_DB * shadow
    _, cutoff_hz, _ = _head_shelf_parameters(
        shadow,
        head_radius_m,
        speed_of_sound_mps,
    )

    if azimuth_deg > 0:
        left = far
        right = speaker.copy()
        source_side = "right"
        far_ear = "left"
    else:
        left = speaker.copy()
        right = far
        source_side = "left"
        far_ear = "right"
    return left, right, {
        "name": "woodworth_folded_shelf_engineering_proxy",
        "model_version": _HEAD_MODEL_VERSION,
        "version": _HEAD_MODEL_VERSION,
        "formula": (
            "theta=min(abs(azimuth), 180-abs(azimuth)); "
            "ITD=(a/c)*(theta+sin(theta)); unity-DC first-order "
            "high-frequency shelf with -12*sin(theta) dB asymptote"
        ),
        "approximation": "uncalibrated engineering proxy",
        "source_side": source_side,
        "theta_rad": lateral_rad,
        "head_shadow_fraction": shadow,
        "itd_ms": itd_s * 1000.0,
        "far_ear": far_ear,
        "far_ear_attenuation_db": attenuation_db,
        "far_ear_attenuation_db_semantics": (
            "high-frequency asymptotic attenuation; not broadband"
        ),
        "high_frequency_attenuation_db": attenuation_db,
        "far_ear_lowpass_hz": cutoff_hz,
        "far_ear_shelf_cutoff_hz": cutoff_hz,
        "filter_type": "first_order_unity_dc_high_frequency_shelf_bilinear",
        "fractional_delay": delay_metadata,
    }


def _pad_pair(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    length = max(len(left), len(right))
    shaped: list[np.ndarray] = []
    for values in (left, right):
        padded = np.zeros(length, dtype=np.float64)
        padded[:len(values)] = values
        shaped.append(padded)
    return shaped[0], shaped[1]


def _normalise_output_pair(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    peak = max(
        float(np.max(np.abs(left))) if len(left) else 0.0,
        float(np.max(np.abs(right))) if len(right) else 0.0,
    )
    if not math.isfinite(peak):
        raise SyntheticReceiveError("simulation produced non-finite samples")
    gain = 1.0
    if peak >= 0.999:
        gain = 0.999 / peak
    return left * gain, right * gain, {
        "pre_normalisation_peak_fs": _rounded(peak),
        "gain_db": _rounded(20.0 * math.log10(gain)) if gain != 1.0 else 0.0,
        "normalised": gain != 1.0,
    }


def _interaural_delay_ms(left: np.ndarray, right: np.ndarray, rate: int) -> float:
    if len(left) < 2 or len(right) < 2:
        return 0.0
    left_centered = np.asarray(left, dtype=np.float64) - float(np.mean(left))
    right_centered = np.asarray(right, dtype=np.float64) - float(np.mean(right))
    if (
        float(np.dot(left_centered, left_centered)) <= _ZERO_POWER
        or float(np.dot(right_centered, right_centered)) <= _ZERO_POWER
    ):
        return 0.0
    correlation = signal.correlate(
        right_centered,
        left_centered,
        mode="full",
        method="fft",
    )
    lags = signal.correlation_lags(len(right_centered), len(left_centered), mode="full")
    peak_index = int(np.argmax(np.abs(correlation)))
    return abs(float(lags[peak_index])) / rate * 1000.0


def _welch_spectrum(samples: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray, float]:
    values = np.asarray(samples, dtype=np.float64)
    if len(values) < 8:
        values = np.pad(values, (0, 8 - len(values)))
    nperseg = min(8192, len(values))
    frequency, psd = signal.welch(
        values,
        fs=rate,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
        return_onesided=True,
    )
    delta_hz = float(frequency[1] - frequency[0]) if len(frequency) > 1 else float(rate)
    return frequency, np.asarray(psd, dtype=np.float64), delta_hz


def _third_octave_levels(
    left: np.ndarray,
    right: np.ndarray,
    rate: int,
) -> dict[str, dict[str, float]]:
    frequency, left_psd, delta_hz = _welch_spectrum(left, rate)
    _, right_psd, _ = _welch_spectrum(right, rate)
    binaural_psd = 0.5 * (left_psd + right_psd)
    nyquist = rate / 2.0
    result: dict[str, dict[str, float]] = {
        "left": {},
        "right": {},
        "binaural": {},
    }
    for center_hz in _THIRD_OCTAVE_CENTERS_HZ:
        if center_hz > nyquist + 1e-9:
            continue
        lower_hz = center_hz / (2.0 ** (1.0 / 6.0))
        upper_hz = min(nyquist, center_hz * (2.0 ** (1.0 / 6.0)))
        mask = (frequency >= lower_hz) & (frequency <= upper_hz)
        for channel, psd in (
            ("left", left_psd),
            ("right", right_psd),
            ("binaural", binaural_psd),
        ):
            power = float(np.sum(psd[mask]) * delta_hz)
            level = (
                max(_DB_FLOOR, 10.0 * math.log10(power))
                if power > _ZERO_POWER
                else _DB_FLOOR
            )
            result[channel][str(center_hz)] = _rounded(level)  # type: ignore[assignment]
    return result


def _out_of_band_ratio_db(
    left: np.ndarray,
    right: np.ndarray,
    rate: int,
    upper_hz: float,
) -> float | None:
    nyquist = rate / 2.0
    if upper_hz >= nyquist:
        return None
    frequency, left_psd, delta_hz = _welch_spectrum(left, rate)
    _, right_psd, _ = _welch_spectrum(right, rate)
    binaural_psd = 0.5 * (left_psd + right_psd)
    if not len(binaural_psd):
        return _DB_FLOOR
    total = float(np.sum(binaural_psd) * delta_hz)
    if total <= _ZERO_POWER:
        return _DB_FLOOR
    out_of_band = float(
        np.sum(binaural_psd[frequency > upper_hz]) * delta_hz
    )
    ratio = out_of_band / total
    if ratio <= _ZERO_POWER:
        return _DB_FLOOR
    return max(_DB_FLOOR, 10.0 * math.log10(ratio))


def _metrics(
    left: np.ndarray,
    right: np.ndarray,
    rate: int,
    *,
    evidence_kind: str,
    head_metrics: dict[str, Any] | None,
    upper_hz: float | None,
    noise_start_s: float | None,
    noise_end_s: float | None,
) -> dict[str, Any]:
    left_rms = _rms_dbfs(left)
    right_rms = _rms_dbfs(right)
    binaural_rms = _rms_dbfs(_binaural_signal(left, right))
    if head_metrics is not None:
        delay_ms = float(head_metrics["itd_ms"])
    elif evidence_kind == "synthetic_ir":
        delay_ms = _interaural_delay_ms(left, right, rate)
    else:
        delay_ms = 0.0

    band_levels = _third_octave_levels(left, right, rate)
    out_of_band = (
        _out_of_band_ratio_db(left, right, rate, upper_hz)
        if upper_hz is not None
        else None
    )

    noise_metrics: dict[str, Any] | None = None
    if noise_start_s is not None and noise_end_s is not None:
        start_sample = max(0, int(round(noise_start_s * rate)))
        end_sample = min(len(left), int(round(noise_end_s * rate)))
        left_window = left[start_sample:end_sample]
        right_window = right[start_sample:end_sample]
        window_rms = _rms_dbfs(_binaural_signal(left_window, right_window))
        window_peak = max(
            float(np.max(np.abs(left_window))) if len(left_window) else 0.0,
            float(np.max(np.abs(right_window))) if len(right_window) else 0.0,
        )
        peak_dbfs = (
            max(_DB_FLOOR, 20.0 * math.log10(window_peak))
            if window_peak > 0.0
            else _DB_FLOOR
        )
        noise_metrics = {
            "start_s": _rounded(noise_start_s),
            "end_s": _rounded(noise_end_s),
            "digital_rms_dbfs": _rounded(window_rms),
            "rms_dbfs": _rounded(window_rms),
            "binaural_digital_rms_dbfs": _rounded(window_rms),
            "left_digital_rms_dbfs": _rounded(_rms_dbfs(left_window)),
            "right_digital_rms_dbfs": _rounded(_rms_dbfs(right_window)),
            "peak_dbfs": _rounded(peak_dbfs),
            "peak_prominence_db": _rounded(max(0.0, peak_dbfs - window_rms)),
            "weighting": "none (unweighted digital dBFS proxy)",
            "calibration": "none",
            "digital_proxy": True,
            "a_weighted": False,
        }

    return {
        "left_rms_dbfs": _rounded(left_rms),
        "right_rms_dbfs": _rounded(right_rms),
        "binaural_rms_dbfs": _rounded(binaural_rms),
        "interaural_level_difference_db": _rounded(left_rms - right_rms),
        "interaural_delay_ms": _rounded(delay_ms),
        "third_octave_band_levels_db": band_levels,
        "third_octave_band_levels_dbfs": band_levels,
        "out_of_band_ratio_db": _rounded(out_of_band),
        "noise": noise_metrics,
        "noise_interval": noise_metrics,
        "definitions": {
            "dbfs": "20*log10(digital RMS), floor -120 dBFS; uncalibrated digital proxy",
            "interaural_level_difference_db": "left RMS dBFS minus right RMS dBFS",
            "interaural_delay_ms": "non-negative magnitude; head mode uses Woodworth model",
            "out_of_band_ratio_db": (
                "10*log10(mean left/right Welch out-of-band power / "
                "mean left/right Welch total power); null when "
                "upper_hz >= Nyquist"
            ),
        },
    }


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(
            payload,
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    # A failed write or replace retains the temporary file as failure evidence.
    # A successful replace atomically moves it to the requested path.
    os.replace(temporary, path)


def _validate_number(
    value: float,
    label: str,
    *,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SyntheticReceiveError(f"{label} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise SyntheticReceiveError(f"{label} must be finite")
    if positive and numeric <= 0:
        raise SyntheticReceiveError(f"{label} must be positive")
    return numeric


def _validate_request(
    mode: str,
    azimuth_deg: float,
    head_radius_m: float,
    speed_of_sound_mps: float,
    upper_hz: float | None,
    noise_start_s: float | None,
    noise_end_s: float | None,
    output_duration_s: float | None,
) -> tuple[float, float, float, float | None, float | None, float | None]:
    if not isinstance(mode, str) or mode not in ALLOWED_MODES:
        raise SyntheticReceiveError(
            "mode must be one of: " + ", ".join(sorted(ALLOWED_MODES))
        )
    azimuth = _validate_number(azimuth_deg, "azimuth_deg")
    if not -180.0 <= azimuth <= 180.0:
        raise SyntheticReceiveError("azimuth_deg must be between -180 and 180")
    radius = _validate_number(head_radius_m, "head_radius_m", positive=True)
    if not _MIN_HEAD_RADIUS_M <= radius <= _MAX_HEAD_RADIUS_M:
        raise SyntheticReceiveError(
            "head_radius_m must be between "
            f"{_MIN_HEAD_RADIUS_M:g} and {_MAX_HEAD_RADIUS_M:g}"
        )
    speed = _validate_number(
        speed_of_sound_mps,
        "speed_of_sound_mps",
        positive=True,
    )
    if not _MIN_SPEED_OF_SOUND_MPS <= speed <= _MAX_SPEED_OF_SOUND_MPS:
        raise SyntheticReceiveError(
            "speed_of_sound_mps must be between "
            f"{_MIN_SPEED_OF_SOUND_MPS:g} and {_MAX_SPEED_OF_SOUND_MPS:g}"
        )
    upper = (
        _validate_number(upper_hz, "upper_hz", positive=True)
        if upper_hz is not None
        else None
    )
    if (noise_start_s is None) != (noise_end_s is None):
        raise SyntheticReceiveError(
            "noise_start_s and noise_end_s must be supplied together"
        )
    start = (
        _validate_number(noise_start_s, "noise_start_s")
        if noise_start_s is not None
        else None
    )
    end = (
        _validate_number(noise_end_s, "noise_end_s")
        if noise_end_s is not None
        else None
    )
    if start is not None and end is not None:
        if start < 0 or end <= start:
            raise SyntheticReceiveError(
                "noise interval must satisfy 0 <= noise_start_s < noise_end_s"
            )
        if output_duration_s is not None and end > output_duration_s + 1e-9:
            raise SyntheticReceiveError(
                "noise interval extends beyond the simulated output duration"
            )
    return azimuth, radius, speed, upper, start, end


def _safe_evaluation_id(value: str) -> bool:
    return isinstance(value, str) and bool(_SAFE_COMPONENT.fullmatch(value))


def _artifact_url(evaluation_id: str, filename: str) -> str:
    return (
        f"/api/acoustic/receive-simulation/evaluations/"
        f"{evaluation_id}/files/{filename}"
    )


def _artifact_record(evaluation_id: str, filename: str, path: Path) -> dict[str, Any]:
    return {
        "filename": filename,
        "path": f"{evaluation_id}/{filename}",
        "sha256": _sha256_file(path),
        "url": _artifact_url(evaluation_id, filename),
    }


def evaluate_synthetic_receive(
    mode: str,
    speaker_name: str,
    speaker_bytes: bytes,
    left_ir_name: str | None = None,
    left_ir_bytes: bytes | None = None,
    right_ir_name: str | None = None,
    right_ir_bytes: bytes | None = None,
    azimuth_deg: float = 30.0,
    head_radius_m: float = 0.0875,
    speed_of_sound_mps: float = 343.0,
    upper_hz: float | None = None,
    noise_start_s: float | None = None,
    noise_end_s: float | None = None,
) -> dict:
    """Evaluate a speaker output through a software-only binaural proxy."""
    (
        azimuth,
        radius,
        speed,
        upper,
        noise_start,
        noise_end,
    ) = _validate_request(
        mode,
        azimuth_deg,
        head_radius_m,
        speed_of_sound_mps,
        upper_hz,
        noise_start_s,
        noise_end_s,
        None,
    )
    speaker = _decode_linear_wav(
        name=speaker_name,
        blob=speaker_bytes,
        fallback_name="speaker.wav",
        label="speaker output",
        allowed_rates=SPEAKER_SAMPLE_RATES,
        max_duration_s=MAX_DURATION_S,
        reject_clipping=True,
    )
    rate = int(speaker["sample_rate_hz"])

    if left_ir_name is not None and left_ir_bytes is None:
        raise SyntheticReceiveError("left IR name was supplied without IR bytes")
    if right_ir_name is not None and right_ir_bytes is None:
        raise SyntheticReceiveError("right IR name was supplied without IR bytes")

    has_left = left_ir_bytes is not None
    has_right = right_ir_bytes is not None
    if has_left != has_right:
        raise SyntheticReceiveError(
            "left and right custom IRs must be supplied as a pair"
        )

    left_ir: dict[str, Any] | None = None
    right_ir: dict[str, Any] | None = None
    left_ir_processing: dict[str, Any] | None = None
    right_ir_processing: dict[str, Any] | None = None
    if has_left and has_right:
        left_ir = _decode_linear_wav(
            name=left_ir_name,
            blob=left_ir_bytes,
            fallback_name="left_ir.wav",
            label="left IR",
            allowed_rates=IR_SAMPLE_RATES,
            max_duration_s=MAX_IR_DURATION_S,
            reject_clipping=False,
        )
        right_ir = _decode_linear_wav(
            name=right_ir_name,
            blob=right_ir_bytes,
            fallback_name="right_ir.wav",
            label="right IR",
            allowed_rates=IR_SAMPLE_RATES,
            max_duration_s=MAX_IR_DURATION_S,
            reject_clipping=False,
        )
        left_kernel, left_ir_processing = _resample_ir(left_ir, rate)
        right_kernel, right_ir_processing = _resample_ir(right_ir, rate)
        left_kernel, right_kernel = _align_ir_kernel_origins(
            left_kernel,
            right_kernel,
            left_ir_processing,
            right_ir_processing,
        )
        left, right = _pad_pair(
            signal.fftconvolve(speaker["samples"], left_kernel, mode="full"),
            signal.fftconvolve(speaker["samples"], right_kernel, mode="full"),
        )
        evidence_kind = "synthetic_ir"
        head_metrics = None
    else:
        left, right, head_metrics = _synthetic_head_outputs(
            speaker["samples"],
            rate,
            azimuth,
            radius,
            speed,
        )
        evidence_kind = "synthetic_head"

    output_duration_s = len(left) / rate
    if noise_end is not None and noise_end > output_duration_s + 1e-9:
        raise SyntheticReceiveError(
            "noise interval extends beyond the simulated output duration"
        )
    left, right, normalisation = _normalise_output_pair(left, right)
    left, right = _pad_pair(left, right)

    metrics = _metrics(
        left,
        right,
        rate,
        evidence_kind=evidence_kind,
        head_metrics=head_metrics,
        upper_hz=upper,
        noise_start_s=noise_start,
        noise_end_s=noise_end,
    )

    evaluation_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + "-"
        + uuid.uuid4().hex[:12]
    )
    store = Path(STORE)
    store.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(
        tempfile.mkdtemp(prefix=".synthetic-receive-", dir=store)
    )
    try:
        left_path = temporary_directory / "virtual_left.wav"
        right_path = temporary_directory / "virtual_right.wav"
        sf.write(left_path, left, rate, format="WAV", subtype="PCM_24")
        sf.write(right_path, right, rate, format="WAV", subtype="PCM_24")

        created_at = _utc_now()
        report_path = f"{evaluation_id}/report.json"
        report_without_result: dict[str, Any] = {
            "schema_version": 1,
            "evaluation_id": evaluation_id,
            "created_at": created_at,
            "mode": mode,
            "scope": (
                "software-only deterministic HATS/DRP binaural receive proxy; "
                "diagnostic simulation only"
            ),
            "synthetic_proxy": True,
            "standard_compliance": False,
            "formal_hats_evidence": False,
            "evidence_kind": evidence_kind,
            "input_identity": {
                "speaker": _public_identity(speaker),
                "left_ir": _public_identity(left_ir) if left_ir else None,
                "right_ir": _public_identity(right_ir) if right_ir else None,
                "left_ir_processing": left_ir_processing,
                "right_ir_processing": right_ir_processing,
                "azimuth_deg": _rounded(azimuth),
                "head_radius_m": _rounded(radius),
                "speed_of_sound_mps": _rounded(speed),
                "upper_hz": _rounded(upper),
                "noise_interval_s": (
                    [_rounded(noise_start), _rounded(noise_end)]
                    if noise_start is not None and noise_end is not None
                    else None
                ),
            },
            "simulation": {
                "analysis_sample_rate_hz": rate,
                "output_subtype": "PCM_24",
                "output_frames": len(left),
                "output_duration_s": _rounded(len(left) / rate),
                "output_normalisation": normalisation,
                "head_model": head_metrics,
            },
            "artifacts": {
                "left": _artifact_record(
                    evaluation_id,
                    "virtual_left.wav",
                    left_path,
                ),
                "right": _artifact_record(
                    evaluation_id,
                    "virtual_right.wav",
                    right_path,
                ),
            },
            "metrics": metrics,
            "limitations": [
                "This is a software-only synthetic proxy, not a HATS/DRP measurement.",
                "It is not an HRTF implementation and has no pinna, torso, room, or source-directivity model.",
                (
                    "The no-IR path uses an uncalibrated engineering proxy: "
                    "folded Woodworth ITD plus a unity-DC first-order "
                    "high-frequency shelf; it is not an exact rigid-sphere model."
                ),
                "Custom IR results are only as valid as the supplied, unverified IR files and their measurement conditions.",
                "RMS, band, and noise values are uncalibrated digital dBFS proxies; no A-weighting or acoustic SPL is claimed.",
                "The result does not establish P.57, P.58, or any other standard compliance.",
            ],
            "result_path": report_path,
        }
        result_digest = _canonical_sha256(report_without_result)
        report = {
            **report_without_result,
            "result": {
                "path": report_path,
                "sha256": result_digest,
                "url": _artifact_url(evaluation_id, "report.json"),
                "hash_basis": (
                    "canonical JSON of report.json excluding result "
                    "and result_sha256"
                ),
            },
            "result_sha256": result_digest,
        }
        _atomic_json_write(temporary_directory / "report.json", report)

        final_directory = store / evaluation_id
        with _WRITE_LOCK:
            if final_directory.exists():
                raise SyntheticReceiveError("generated evaluation id already exists")
            os.replace(temporary_directory, final_directory)
    except Exception:
        # Preserve the complete failed staging directory for postmortem review.
        # Successful runs leave this path via the atomic directory replace above.
        raise
    return report


def _load_report(evaluation_id: str) -> dict | None:
    if not _safe_evaluation_id(evaluation_id):
        return None
    store = Path(STORE)
    directory = store / evaluation_id
    if directory.is_symlink() or not directory.is_dir():
        return None
    path = directory / "report.json"
    if path.is_symlink() or not path.is_file():
        return None
    try:
        resolved_store = store.resolve()
        resolved_path = path.resolve()
        if (
            resolved_path.parent != directory.resolve()
            or not resolved_path.is_relative_to(resolved_store)
        ):
            return None
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != 1
        or report.get("evaluation_id") != evaluation_id
    ):
        return None
    return report


def list_synthetic_receive_evaluations(limit: int = 20) -> list[dict]:
    """Return persisted reports in reverse chronological order."""
    if isinstance(limit, bool) or not isinstance(limit, Integral):
        raise SyntheticReceiveError("limit must be an integer from 1 to 100")
    limit = int(limit)
    if not 1 <= limit <= 100:
        raise SyntheticReceiveError("limit must be an integer from 1 to 100")
    store = Path(STORE)
    if not store.is_dir():
        return []
    reports: list[dict] = []
    for directory in store.iterdir():
        if directory.is_symlink() or not directory.is_dir():
            continue
        report = _load_report(directory.name)
        if report is not None:
            reports.append(report)
    reports.sort(
        key=lambda row: (str(row.get("created_at", "")), str(row.get("evaluation_id", ""))),
        reverse=True,
    )
    return reports[:limit]


def get_synthetic_receive_evaluation(evaluation_id: str) -> dict | None:
    """Read a persisted report, returning None for unsafe or unknown IDs."""
    return _load_report(evaluation_id)


def get_synthetic_receive_file(evaluation_id: str, filename: str) -> Path | None:
    """Resolve one allow-listed artifact without permitting path traversal."""
    if not _safe_evaluation_id(evaluation_id) or filename not in ALLOWED_FILE_NAMES:
        return None
    if _load_report(evaluation_id) is None:
        return None
    store = Path(STORE)
    directory = store / evaluation_id
    candidate = directory / filename
    if candidate.is_symlink() or not candidate.is_file():
        return None
    try:
        resolved_store = store.resolve()
        resolved = candidate.resolve()
        if (
            resolved.parent != directory.resolve()
            or not resolved.is_relative_to(resolved_store)
        ):
            return None
    except OSError:
        return None
    return resolved
