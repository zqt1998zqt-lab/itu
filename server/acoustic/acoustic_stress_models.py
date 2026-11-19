"""Engineering diagnostics and contract rows for acoustic stress workloads."""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .acoustic_stress_profiles import StressProfile
from .acoustic_stress_store import ROLES, validate_activity_provenance


SCHEMA_VERSION = "acoustic-stress-diagnostics-v2"
LEGACY_SCHEMA_VERSION = "acoustic-stress-diagnostics-v1"
EVIDENCE_GRADE = "engineering_proxy"
FORMAL_COMPLIANCE = False
FORMAL_TEST_STATUS = "not_measured"
SOURCE_DURATION_S = 12.0
_SILENCE_RMS = 1e-9
_MAX_ALIGNMENT_MS = 100.0
_TIME_TOLERANCE_S = 1e-6
ACTIVITY_FRAME_MS = 20
ACTIVITY_HOP_MS = 10
ACTIVITY_RMS_THRESHOLD = 1e-5
MIN_ACTIVITY_SECONDS = 0.05
# A declared long segment must not be 99% silence. This is a label-integrity
# guard in addition to the 50 ms minimum; it is not a speech detector.
MIN_ACTIVITY_COVERAGE = 0.05
ECHO_DELAY_GUARD_MS = 100.0
ECHO_CONVERGENCE_WINDOWS_S = ((0.5, 1.5), (10.0, 12.0))
_ACTIVITY_NUMERIC_TOLERANCE = 1e-9

METRIC_KEYS = (
    "primary",
    "tx_residual_reduction_db",
    "convergence_db",
    "cleanref_projection_gain_db",
    "cleanref_correlation",
    "tx_level_dbfs",
    "mic_level_dbfs",
    "far_level_dbfs",
    "near_level_dbfs",
)
SEGMENT_MODE = {
    "quiet": "not_applicable",
    "far_only": "far",
    "near_only": "near",
    "double_talk": "double_talk",
    "noise_only": "noise",
}


class StressModelError(ValueError):
    """A stress case or its audio cannot support a trustworthy proxy."""


def stress_rate(mode: str) -> int:
    if mode.startswith("SWB-"):
        return 48_000
    if mode.startswith("NB-"):
        return 8_000
    if mode.startswith("WB-"):
        return 16_000
    raise StressModelError(f"unsupported stress mode: {mode}")


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(float(value))
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rms(samples: np.ndarray) -> float:
    values = np.asarray(samples, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(values))))


def _power(samples: np.ndarray) -> float:
    return float(np.mean(np.square(np.asarray(samples, dtype=np.float64)))) if len(samples) else 0.0


def _dbfs(power: float) -> float | None:
    if not math.isfinite(power) or power <= _SILENCE_RMS**2:
        return None
    return float(10.0 * math.log10(power))


def _ratio_db(numerator: float, denominator: float) -> float | None:
    if (
        not math.isfinite(numerator)
        or not math.isfinite(denominator)
        or numerator <= _SILENCE_RMS**2
        or denominator <= _SILENCE_RMS**2
    ):
        return None
    return float(10.0 * math.log10(numerator / denominator))


def _load_mono(path: Path, label: str) -> tuple[np.ndarray, int]:
    try:
        samples, rate = sf.read(
            Path(path),
            dtype="float64",
            always_2d=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise StressModelError(f"{label} could not be decoded: {error}") from error
    if (
        not isinstance(rate, (int, np.integer))
        or int(rate) <= 0
        or samples.ndim != 2
        or samples.shape[1] < 1
        or len(samples) == 0
        or not np.isfinite(samples).all()
    ):
        raise StressModelError(f"{label} has invalid audio shape or samples")
    return np.asarray(samples[:, 0], dtype=np.float64), int(rate)


def validate_activity_segments(
    provenance: Mapping[str, Any],
    *,
    scope: str,
    mode: str,
) -> list[dict[str, Any]]:
    """Validate exact generator timelines and source provenance."""
    try:
        return validate_activity_provenance(
            provenance,
            scope=scope,
            mode=mode,
            require_source_flags=True,
        )
    except ValueError as error:
        raise StressModelError(str(error)) from error


def _segment_bounds(
    segment: Mapping[str, Any],
    rate: int,
) -> tuple[int, int]:
    return (
        int(round(float(segment["start_s"]) * rate)),
        int(round(float(segment["end_s"]) * rate)),
    )


def _segment_audio(
    samples: np.ndarray,
    segment: Mapping[str, Any],
    rate: int,
) -> np.ndarray:
    start, end = _segment_bounds(segment, rate)
    if start < 0 or end > len(samples) or end <= start:
        return np.asarray([], dtype=np.float64)
    return np.asarray(samples[start:end], dtype=np.float64)


def _frame_activity(
    samples: np.ndarray,
    segment: Mapping[str, Any],
    rate: int,
) -> tuple[
    dict[str, Any],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
]:
    values = np.asarray(samples, dtype=np.float64)
    start, end = _segment_bounds(segment, rate)
    start = max(0, min(len(values), start))
    end = max(0, min(len(values), end))
    frame_samples = max(1, int(round(ACTIVITY_FRAME_MS * rate / 1000.0)))
    hop_samples = max(1, int(round(ACTIVITY_HOP_MS * rate / 1000.0)))
    length = max(0, end - start)
    mask = np.zeros(len(values), dtype=bool)
    if length < frame_samples:
        return {
            "active_samples": 0,
            "active_seconds": 0.0,
            "coverage": 0.0,
            "frame_count": 0,
        }, mask, np.asarray([], dtype=np.int64), np.asarray([], dtype=bool), hop_samples
    frame_starts = np.arange(
        start,
        end - frame_samples + 1,
        hop_samples,
        dtype=np.int64,
    )
    frame_values = values[
        frame_starts[:, None] + np.arange(frame_samples, dtype=np.int64)[None, :]
    ]
    frame_rms = np.sqrt(np.mean(np.square(frame_values), axis=1))
    active_frames = frame_rms > ACTIVITY_RMS_THRESHOLD
    for frame_start in frame_starts[active_frames]:
        start_sample = int(frame_start)
        end_sample = start_sample + frame_samples
        mask[start_sample:end_sample] |= (
            np.abs(values[start_sample:end_sample]) > ACTIVITY_RMS_THRESHOLD
        )
    # Count one hop per active 20 ms frame. This keeps a 50 ms rectangular
    # burst rate-aware without losing it to sine zero crossings.
    active_samples = min(length, int(np.count_nonzero(active_frames)) * hop_samples)
    return {
        "active_samples": active_samples,
        "active_seconds": active_samples / rate,
        "coverage": active_samples / length if length else 0.0,
        "frame_count": int(len(frame_starts)),
        "active_frame_count": int(np.count_nonzero(active_frames)),
    }, mask, frame_starts, active_frames, hop_samples


def _shift_mask(mask: np.ndarray, samples: int) -> np.ndarray:
    shifted = np.zeros_like(mask)
    if samples <= 0:
        shifted[:] = mask
    elif samples < len(mask):
        shifted[samples:] = mask[:-samples]
    return shifted


def _segment_activity(
    *,
    near: np.ndarray,
    far_end: np.ndarray,
    segment: Mapping[str, Any],
    rate: int,
    scope: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    label = segment["label"]
    near_view, near_mask, near_starts, near_frames, near_hop = _frame_activity(
        near,
        segment,
        rate,
    )
    far_view, far_mask, far_starts, far_frames, far_hop = _frame_activity(
        far_end,
        segment,
        rate,
    )
    joint_mask = near_mask & far_mask
    start, end = _segment_bounds(segment, rate)
    segment_length = max(0, end - start)
    if (
        near_hop == far_hop
        and np.array_equal(near_starts, far_starts)
    ):
        joint_frames = near_frames & far_frames
        joint_samples = min(
            segment_length,
            int(np.count_nonzero(joint_frames)) * near_hop,
        )
    else:
        joint_samples = int(np.count_nonzero(joint_mask))
    min_samples = int(round(MIN_ACTIVITY_SECONDS * rate))
    near_valid = (
        near_view["active_samples"] >= min_samples
        and near_view["coverage"] >= MIN_ACTIVITY_COVERAGE
    )
    far_valid = (
        far_view["active_samples"] >= min_samples
        and far_view["coverage"] >= MIN_ACTIVITY_COVERAGE
    )
    joint_valid = (
        joint_samples >= min_samples
        and segment_length > 0
        and joint_samples / segment_length >= MIN_ACTIVITY_COVERAGE
    )
    if label == "far_only":
        valid = far_valid and near_view["active_samples"] == 0
        reason = (
            "declared far_only lacks sufficient far activity or contains near activity"
            if not valid
            else "far_only has sufficient far activity and no near activity"
        )
    elif label == "near_only":
        valid = near_valid and far_view["active_samples"] == 0
        reason = (
            "declared near_only lacks sufficient near activity or contains far activity"
            if not valid
            else "near_only has sufficient near activity and no far activity"
        )
    elif label == "double_talk":
        valid = near_valid and far_valid and joint_valid
        reason = (
            "declared double_talk lacks 50 ms of same-frame near/far overlap"
            if not valid
            else "double_talk has sufficient same-frame near/far overlap"
        )
    else:
        valid = near_view["active_samples"] == 0 and far_view["active_samples"] == 0
        reason = (
            f"declared {label} contains near/far activity"
            if not valid
            else f"{label} has no near/far activity"
        )
    public = {
        "frame_ms": ACTIVITY_FRAME_MS,
        "hop_ms": ACTIVITY_HOP_MS,
        "rms_threshold": ACTIVITY_RMS_THRESHOLD,
        "min_active_s": MIN_ACTIVITY_SECONDS,
        "min_coverage": MIN_ACTIVITY_COVERAGE,
        "near_active_samples": near_view["active_samples"],
        "far_active_samples": far_view["active_samples"],
        "joint_active_samples": joint_samples,
        "near_coverage": near_view["coverage"],
        "far_coverage": far_view["coverage"],
        "joint_coverage": (
            joint_samples / segment_length if segment_length else 0.0
        ),
        "valid": bool(valid),
        "reason": reason,
    }
    masks = {
        "near": near_mask,
        "far": far_mask,
        "joint": joint_mask,
    }
    if scope == "echo" and label == "far_only":
        segment_mask = np.zeros(len(near), dtype=bool)
        segment_mask[start:end] = True
        guard_samples = int(round(ECHO_DELAY_GUARD_MS * rate / 1000.0))
        echo_mask = _shift_mask(far_mask, guard_samples) & segment_mask & ~near_mask
        masks["echo_primary"] = echo_mask
        public["echo_delay_guard_ms"] = ECHO_DELAY_GUARD_MS
        public["echo_guarded_active_samples"] = int(np.count_nonzero(echo_mask))
        public["echo_guarded_active_s"] = (
            int(np.count_nonzero(echo_mask)) / rate
        )
    return public, masks


def _concat_masked(
    samples: np.ndarray,
    masks: list[np.ndarray],
) -> np.ndarray:
    pieces = [
        np.asarray(samples, dtype=np.float64)[mask]
        for mask in masks
        if np.count_nonzero(mask)
    ]
    return np.concatenate(pieces) if pieces else np.asarray([])


def _bounded_alignment(
    reference: np.ndarray,
    transmitted: np.ndarray,
    *,
    rate: int,
) -> dict[str, Any] | None:
    length = min(len(reference), len(transmitted))
    if length < 2:
        return None
    ref = np.asarray(reference[:length], dtype=np.float64)
    tx = np.asarray(transmitted[:length], dtype=np.float64)
    if _rms(ref) <= _SILENCE_RMS or _rms(tx) <= _SILENCE_RMS:
        return None
    ref = ref - float(np.mean(ref))
    tx = tx - float(np.mean(tx))
    max_lag = min(length - 1, int(round(_MAX_ALIGNMENT_MS * rate / 1000.0)))
    if max_lag < 1:
        return None
    fft_size = 1 << (2 * length - 1).bit_length()
    spectrum = np.fft.rfft(tx, n=fft_size) * np.conj(
        np.fft.rfft(ref, n=fft_size)
    )
    corr = np.fft.irfft(spectrum, n=fft_size)
    ref_square = np.empty(length + 1, dtype=np.float64)
    tx_square = np.empty(length + 1, dtype=np.float64)
    ref_square[0] = 0.0
    tx_square[0] = 0.0
    np.cumsum(np.square(ref), out=ref_square[1:])
    np.cumsum(np.square(tx), out=tx_square[1:])
    lags = np.arange(-max_lag, max_lag + 1, dtype=np.int64)
    dots = np.empty(len(lags), dtype=np.float64)
    counts = np.empty(len(lags), dtype=np.int64)
    ref_energy = np.empty(len(lags), dtype=np.float64)
    tx_energy = np.empty(len(lags), dtype=np.float64)
    for index, lag in enumerate(lags):
        if lag >= 0:
            overlap = length - int(lag)
            dots[index] = corr[int(lag)]
            ref_energy[index] = ref_square[overlap]
            tx_energy[index] = tx_square[length] - tx_square[int(lag)]
        else:
            overlap = length + int(lag)
            dots[index] = corr[fft_size + int(lag)]
            ref_energy[index] = ref_square[length] - ref_square[-int(lag)]
            tx_energy[index] = tx_square[overlap]
        counts[index] = overlap
    denominator = np.sqrt(ref_energy * tx_energy)
    valid = (
        (counts >= max(2, int(round(0.05 * rate))))
        & np.isfinite(denominator)
        & (denominator > 0.0)
    )
    if not valid.any():
        return None
    normalized = np.full(len(lags), np.nan)
    normalized[valid] = dots[valid] / denominator[valid]
    # Prefer a positive reference projection for a transfer-gain proxy.
    masked = np.where(valid, normalized, -np.inf)
    peak = int(np.argmax(masked))
    lag = int(lags[peak])
    overlap = int(counts[peak])
    if lag >= 0:
        aligned_ref = ref[:overlap]
        aligned_tx = tx[lag:lag + overlap]
    else:
        aligned_ref = ref[-lag:-lag + overlap]
        aligned_tx = tx[:overlap]
    denominator_value = float(
        np.linalg.norm(aligned_ref) * np.linalg.norm(aligned_tx)
    )
    if denominator_value <= 0.0 or not math.isfinite(denominator_value):
        return None
    correlation = float(np.dot(aligned_ref, aligned_tx) / denominator_value)
    reference_energy = float(np.dot(aligned_ref, aligned_ref))
    projection = (
        float(np.dot(aligned_tx, aligned_ref)) / reference_energy
        if reference_energy > 0.0
        else None
    )
    gain_db = (
        20.0 * math.log10(projection)
        if projection is not None and projection > 0.0 and math.isfinite(projection)
        else None
    )
    return {
        "lag_samples": lag,
        "lag_ms": lag * 1000.0 / rate,
        "correlation": max(-1.0, min(1.0, correlation)),
        "projection_gain_db": gain_db,
        "overlap_samples": overlap,
    }


def _segment_metrics(
    *,
    scope: str,
    segment: Mapping[str, Any],
    reference: np.ndarray,
    microphone: np.ndarray,
    far_end: np.ndarray,
    transmitted: np.ndarray,
    rate: int,
    activity: Mapping[str, Any],
    masks: Mapping[str, np.ndarray],
    tx_valid: bool,
) -> tuple[dict[str, Any], str, str]:
    label = segment["label"]
    metrics = {key: None for key in METRIC_KEYS}
    if label in {"quiet", "noise_only"}:
        if not activity["valid"]:
            return metrics, "not_measurable", str(activity["reason"])
        return (
            metrics,
            "not_applicable",
            "静音/噪声段不构造 stress 主指标或替代门限；"
            "该结论要求 near/far 实际无活动",
        )
    if not activity["valid"]:
        return metrics, "not_measurable", str(activity["reason"])
    if scope == "double_talk" and not tx_valid:
        return (
            metrics,
            "not_measurable",
            "TX mute/short makes double-talk TX-dependent proxies unavailable; "
            "the near/far activity map is retained but is not a passing proxy",
        )
    min_samples = int(round(MIN_ACTIVITY_SECONDS * rate))
    min_guard_samples = max(
        1,
        min_samples - int(round(ACTIVITY_HOP_MS * rate / 1000.0)) // 2,
    )
    if label == "far_only" and scope == "echo":
        mask = masks["echo_primary"]
        if int(np.count_nonzero(mask)) < min_guard_samples:
            return (
                metrics,
                "not_measurable",
                "echo far activity does not survive the 100 ms far->mic delay guard",
            )
        mic = np.asarray(microphone, dtype=np.float64)[mask]
        far = np.asarray(far_end, dtype=np.float64)[mask]
        tx = np.asarray(transmitted, dtype=np.float64)[mask]
        mic_power = _power(mic)
        far_power = _power(far)
        metrics.update(
            mic_level_dbfs=_dbfs(mic_power),
            far_level_dbfs=_dbfs(far_power),
            tx_level_dbfs=_dbfs(_power(tx)) if tx_valid else None,
        )
        metrics["primary"] = _ratio_db(far_power, mic_power)
        if tx_valid:
            metrics["tx_residual_reduction_db"] = _ratio_db(
                mic_power,
                _power(tx),
            )
        reason = (
            "echo input coupling on far-active frames after a 100 ms delay guard; "
            "far/mic only, not TX-dependent, and not formal TCLw or ERLE"
        )
        status = (
            "review_required"
            if metrics["primary"] is not None
            else "not_measurable"
        )
        return metrics, status, reason

    if label == "double_talk" and scope == "double_talk":
        mask = masks["joint"]
        mic = np.asarray(microphone, dtype=np.float64)[mask]
        tx = np.asarray(transmitted, dtype=np.float64)[mask]
        far = np.asarray(far_end, dtype=np.float64)[mask]
        near = np.asarray(reference, dtype=np.float64)[mask]
        mic_power = _power(mic)
        tx_power = _power(tx)
        far_power = _power(far)
        near_power = _power(near)
        metrics.update(
            tx_level_dbfs=_dbfs(tx_power),
            mic_level_dbfs=_dbfs(mic_power),
            far_level_dbfs=_dbfs(far_power),
            near_level_dbfs=_dbfs(near_power),
        )
        if (
            tx_valid
            and far_power > _SILENCE_RMS**2
            and near_power > _SILENCE_RMS**2
            and mic_power > _SILENCE_RMS**2
            and tx_power > _SILENCE_RMS**2
        ):
            metrics["primary"] = float(
                np.clip(100.0 * tx_power / mic_power, 0.0, 150.0)
            )
        if tx_valid and far_power > _SILENCE_RMS**2 and near_power > _SILENCE_RMS**2:
            alignment = _bounded_alignment(near, tx, rate=rate)
            if alignment is not None:
                metrics["cleanref_correlation"] = alignment["correlation"]
                metrics["cleanref_projection_gain_db"] = (
                    alignment["projection_gain_db"]
                )
        if tx_valid:
            metrics["tx_residual_reduction_db"] = _ratio_db(mic_power, tx_power)
        reason = (
            "双讲同帧实际重叠段（20 ms 帧/10 ms hop，至少 50 ms）的 "
            "100*Ptx/Pmic 工程比；附 cleanref 有界对齐投影，不是 A1/A2"
        )
    elif label == "near_only":
        mask = masks["near"]
        mic = np.asarray(microphone, dtype=np.float64)[mask]
        far = np.asarray(far_end, dtype=np.float64)[mask]
        near = np.asarray(reference, dtype=np.float64)[mask]
        tx = np.asarray(transmitted, dtype=np.float64)[mask]
        metrics.update(
            mic_level_dbfs=_dbfs(_power(mic)),
            far_level_dbfs=_dbfs(_power(far)),
            near_level_dbfs=_dbfs(_power(near)),
            tx_level_dbfs=_dbfs(_power(tx)) if tx_valid else None,
        )
        if tx_valid:
            alignment = _bounded_alignment(near, tx, rate=rate)
            if alignment is not None:
                metrics["cleanref_correlation"] = alignment["correlation"]
                metrics["cleanref_projection_gain_db"] = (
                    alignment["projection_gain_db"]
                )
        reason = "近端实际活动段的 cleanref 对齐投影诊断"
    elif label == "far_only" and scope == "double_talk":
        mask = masks["far"]
        mic = np.asarray(microphone, dtype=np.float64)[mask]
        far = np.asarray(far_end, dtype=np.float64)[mask]
        tx = np.asarray(transmitted, dtype=np.float64)[mask]
        metrics.update(
            mic_level_dbfs=_dbfs(_power(mic)),
            far_level_dbfs=_dbfs(_power(far)),
            tx_level_dbfs=_dbfs(_power(tx)) if tx_valid else None,
        )
        reason = "远端实际活动段的输入电平证据；不把含噪能量差当 ERLE"
    else:
        reason = "stress engineering segment; no formal compliance claim"
    metrics.update(
        convergence_db=None,
    )
    if metrics["primary"] is not None or any(
        metrics[key] is not None
        for key in (
            "tx_residual_reduction_db",
            "cleanref_projection_gain_db",
            "cleanref_correlation",
        )
    ):
        status = "review_required"
    else:
        status = "not_measurable"
    return metrics, status, reason


def diagnose_stress_case(
    case: Mapping[str, Any],
    input_paths: Mapping[str, Path],
    tx_path: Path,
    *,
    profile: StressProfile,
    tx_sha256: str | None = None,
) -> dict[str, Any]:
    """Compute the dedicated echo/double-talk engineering diagnostic."""
    if not isinstance(case, Mapping):
        raise StressModelError("stress case must be an object")
    provenance = case.get("provenance")
    if not isinstance(provenance, Mapping):
        raise StressModelError("stress case provenance is invalid")
    category = provenance.get("workload_category")
    segments = validate_activity_segments(
        provenance,
        scope=profile.scope,
        mode=profile.mode,
    )
    if set(input_paths) != set(ROLES):
        raise StressModelError(
            "stress input paths must contain microphone/reference/far_end/noise"
        )
    audio: dict[str, np.ndarray] = {}
    rates: set[int] = set()
    for role in ROLES:
        samples, rate = _load_mono(Path(input_paths[role]), role)
        audio[role] = samples
        rates.add(rate)
    tx, tx_rate = _load_mono(Path(tx_path), "tx")
    rates.add(tx_rate)
    if len(rates) != 1:
        raise StressModelError("stress four inputs and TX rates differ")
    rate = next(iter(rates))
    if rate != stress_rate(profile.mode):
        raise StressModelError("stress audio rate differs from profile mode")
    if any(len(samples) != len(audio["microphone"]) for samples in audio.values()):
        raise StressModelError("stress four input lengths differ")
    tx_hash = (
        tx_sha256
        if _is_sha256(tx_sha256)
        else _sha256_file(Path(tx_path))
    )
    tx_duration_s = len(tx) / rate
    tx_muted = _rms(tx) <= _SILENCE_RMS
    short_tx = tx_duration_s < 0.95 * SOURCE_DURATION_S
    tx_valid = not (short_tx or tx_muted)
    segment_rows: list[dict[str, Any]] = []
    segment_evidence: list[
        tuple[Mapping[str, Any], dict[str, Any], dict[str, np.ndarray]]
    ] = []
    for segment in segments:
        activity, masks = _segment_activity(
            near=audio["reference"],
            far_end=audio["far_end"],
            segment=segment,
            rate=rate,
            scope=profile.scope,
        )
        metrics, status, reason = _segment_metrics(
            scope=profile.scope,
            segment=segment,
            reference=audio["reference"],
            microphone=audio["microphone"],
            far_end=audio["far_end"],
            transmitted=tx,
            rate=rate,
            activity=activity,
            masks=masks,
            tx_valid=tx_valid,
        )
        segment_evidence.append((segment, activity, masks))
        segment_rows.append(
            {
                "label": segment["label"],
                "start_s": segment["start_s"],
                "end_s": segment["end_s"],
                "mode": SEGMENT_MODE[segment["label"]],
                "status": status,
                "metrics": metrics,
                "reason": reason,
                "activity": activity,
            }
        )
    primary_name = (
        "echo_primary_proxy_db"
        if profile.scope == "echo"
        else "double_talk_energy_ratio_proxy_percent"
    )
    primary_unit = "dB" if profile.scope == "echo" else "percent"
    far_segments = [
        segment for segment in segments if segment["label"] == "far_only"
    ]
    double_segments = [
        segment for segment in segments if segment["label"] == "double_talk"
    ]
    primary: float | None = None
    convergence: float | None = None
    if profile.scope == "echo" and far_segments:
        echo_evidence = [
            evidence
            for evidence in segment_evidence
            if evidence[0]["label"] == "far_only"
        ]
        if echo_evidence and all(
            evidence[1]["valid"] for evidence in echo_evidence
        ):
            masks = [
                evidence[2]["echo_primary"]
                for evidence in echo_evidence
            ]
            if sum(int(np.count_nonzero(mask)) for mask in masks) >= max(
                1,
                int(round(MIN_ACTIVITY_SECONDS * rate))
                - int(round(ACTIVITY_HOP_MS * rate / 1000.0)) // 2,
            ):
                mic_far = _concat_masked(audio["microphone"], masks)
                far_far = _concat_masked(audio["far_end"], masks)
                primary = _ratio_db(_power(far_far), _power(mic_far))
        if tx_valid and echo_evidence and all(
            evidence[1]["valid"] for evidence in echo_evidence
        ):
            if len(far_segments) >= 2:
                first = _segment_audio(tx, far_segments[0], rate)
                last = _segment_audio(tx, far_segments[-1], rate)
                convergence = _ratio_db(_power(last), _power(first))
            elif category == "convergence":
                early = _segment_audio(
                    tx,
                    {
                        "start_s": ECHO_CONVERGENCE_WINDOWS_S[0][0],
                        "end_s": ECHO_CONVERGENCE_WINDOWS_S[0][1],
                    },
                    rate,
                )
                late = _segment_audio(
                    tx,
                    {
                        "start_s": ECHO_CONVERGENCE_WINDOWS_S[1][0],
                        "end_s": ECHO_CONVERGENCE_WINDOWS_S[1][1],
                    },
                    rate,
                )
                convergence = _ratio_db(_power(late), _power(early))
    elif profile.scope == "double_talk" and double_segments:
        double_evidence = [
            evidence
            for evidence in segment_evidence
            if evidence[0]["label"] == "double_talk"
        ]
        if (
            tx_valid
            and double_evidence
            and all(evidence[1]["valid"] for evidence in double_evidence)
        ):
            masks = [evidence[2]["joint"] for evidence in double_evidence]
            if sum(
                evidence[1]["joint_active_samples"]
                for evidence in double_evidence
            ) >= int(round(MIN_ACTIVITY_SECONDS * rate)):
                mic_double = _concat_masked(audio["microphone"], masks)
                tx_double = _concat_masked(tx, masks)
                ref_double = _concat_masked(audio["reference"], masks)
                far_double = _concat_masked(audio["far_end"], masks)
                if (
                    _power(ref_double) > _SILENCE_RMS**2
                    and _power(far_double) > _SILENCE_RMS**2
                    and _power(mic_double) > _SILENCE_RMS**2
                    and _power(tx_double) > _SILENCE_RMS**2
                ):
                    primary = float(
                        np.clip(
                            100.0 * _power(tx_double) / _power(mic_double),
                            0.0,
                            150.0,
                        )
                    )
    status = "review_required" if primary is not None else "not_measurable"
    if status == "not_measurable":
        if profile.scope == "echo" and tx_valid is False:
            reason = (
                "echo input coupling is not measurable from the gated "
                "far/mic activity; TX is also invalid, so TX residual and "
                "convergence are unavailable"
            )
        else:
            reason = (
                "stress primary is not measurable: the declared activity "
                "map has no sufficient 50 ms same-frame valid evidence"
            )
    else:
        if profile.scope == "echo" and tx_valid is False:
            reason = (
                "echo far/mic input coupling is measurable independently of "
                "TX; TX residual and convergence are not measured, and this "
                "does not establish AEC performance"
            )
        else:
            reason = (
                "engineering stress proxy only; primary value requires review, "
                "not formal TCLw/ERLE or A1/A2"
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case.get("case_id") or case.get("id"),
        "tx_sha256": tx_hash,
        "scope": profile.scope,
        "mode": profile.mode,
        "category": category,
        "evidence_grade": EVIDENCE_GRADE,
        "formal_compliance": FORMAL_COMPLIANCE,
        "standard_compliance": False,
        "formal_test_status": FORMAL_TEST_STATUS,
        "status": status,
        "reason": reason,
        "review_required": status == "review_required",
        "sample_rate_hz": rate,
        "source_duration_s": len(audio["microphone"]) / rate,
        "tx_duration_s": tx_duration_s,
        "tx_muted": tx_muted,
        "activity_gate": {
            "frame_ms": ACTIVITY_FRAME_MS,
            "hop_ms": ACTIVITY_HOP_MS,
            "rms_threshold": ACTIVITY_RMS_THRESHOLD,
            "min_active_s": MIN_ACTIVITY_SECONDS,
            "min_coverage": MIN_ACTIVITY_COVERAGE,
        },
        "primary": {
            "name": primary_name,
            "value": primary,
            "unit": primary_unit,
            "status": "review_required" if primary is not None else "not_measured",
        },
        "convergence_db": convergence,
        "convergence_note": (
            "supplemental TX power ratio for convergence: "
            "10-12 s versus 0.5-1.5 s; signal-level confound, not an "
            "adaptive-filter convergence measurement"
            if category == "convergence" and convergence is not None
            else "not measured or not applicable"
        ),
        "segments": segment_rows,
        "limitations": [
            "engineering proxy only",
            "not formal ITU-T compliance evidence",
            "echo metric is a noisy coupling proxy, not formal TCLw or ERLE",
            "double-talk metric is a power ratio, not A1/A2",
            "activity uses 20 ms frames, 10 ms hop, and 1e-5 RMS",
            "echo input coupling is independent of TX; TX residual is not AEC proof",
            "mute, silence, no overlap, and short TX are not passing evidence",
        ],
    }


_ACTIVITY_FIELDS = frozenset(
    {
        "frame_ms",
        "hop_ms",
        "rms_threshold",
        "min_active_s",
        "min_coverage",
        "near_active_samples",
        "far_active_samples",
        "joint_active_samples",
        "near_coverage",
        "far_coverage",
        "joint_coverage",
        "valid",
        "reason",
    }
)
_ECHO_ACTIVITY_FIELDS = _ACTIVITY_FIELDS | {
    "echo_delay_guard_ms",
    "echo_guarded_active_samples",
    "echo_guarded_active_s",
}


def _metrics_all_null(metrics: Mapping[str, Any]) -> bool:
    return all(value is None for value in metrics.values())


def _validate_activity_evidence(
    activity: Any,
    *,
    label: str,
    scope: str,
    rate: int,
    segment_samples: int,
) -> None:
    expected_fields = (
        _ECHO_ACTIVITY_FIELDS
        if scope == "echo" and label == "far_only"
        else _ACTIVITY_FIELDS
    )
    if not isinstance(activity, Mapping) or set(activity) != set(expected_fields):
        raise StressModelError("stress segment activity fields are invalid")
    if (
        activity.get("frame_ms") != ACTIVITY_FRAME_MS
        or activity.get("hop_ms") != ACTIVITY_HOP_MS
        or activity.get("rms_threshold") != ACTIVITY_RMS_THRESHOLD
        or activity.get("min_active_s") != MIN_ACTIVITY_SECONDS
        or activity.get("min_coverage") != MIN_ACTIVITY_COVERAGE
        or not isinstance(activity.get("valid"), bool)
        or not isinstance(activity.get("reason"), str)
        or not activity["reason"]
    ):
        raise StressModelError("stress segment activity envelope is invalid")
    for field in (
        "near_active_samples",
        "far_active_samples",
        "joint_active_samples",
    ):
        value = activity.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise StressModelError("stress segment activity count is invalid")
    for field in ("near_coverage", "far_coverage", "joint_coverage"):
        value = activity.get(field)
        if (
            not _finite_number(value)
            or float(value) < 0.0
            or float(value) > 1.0
        ):
            raise StressModelError("stress segment activity coverage is invalid")
    counts = {
        field: int(activity[field])
        for field in (
            "near_active_samples",
            "far_active_samples",
            "joint_active_samples",
        )
    }
    if segment_samples <= 0:
        raise StressModelError("stress segment sample length is invalid")
    if any(value > segment_samples for value in counts.values()):
        raise StressModelError("stress segment activity exceeds segment length")
    expected_coverage = {
        "near_coverage": counts["near_active_samples"] / segment_samples,
        "far_coverage": counts["far_active_samples"] / segment_samples,
        "joint_coverage": counts["joint_active_samples"] / segment_samples,
    }
    for field, expected in expected_coverage.items():
        if abs(float(activity[field]) - expected) > _ACTIVITY_NUMERIC_TOLERANCE:
            raise StressModelError(
                f"stress segment {field} is not active_samples/segment_samples"
            )
    if counts["joint_active_samples"] > min(
        counts["near_active_samples"],
        counts["far_active_samples"],
    ):
        raise StressModelError("stress joint activity exceeds a source activity")
    min_samples = int(round(MIN_ACTIVITY_SECONDS * rate))
    if activity["valid"]:
        if label == "far_only":
            if (
                counts["far_active_samples"] < min_samples
                or activity["far_coverage"] < MIN_ACTIVITY_COVERAGE
                or counts["near_active_samples"] != 0
            ):
                raise StressModelError("valid far_only activity rule is false")
        elif label == "near_only":
            if (
                counts["near_active_samples"] < min_samples
                or activity["near_coverage"] < MIN_ACTIVITY_COVERAGE
                or counts["far_active_samples"] != 0
            ):
                raise StressModelError("valid near_only activity rule is false")
        elif label == "double_talk":
            if (
                counts["near_active_samples"] < min_samples
                or counts["far_active_samples"] < min_samples
                or counts["joint_active_samples"] < min_samples
                or activity["near_coverage"] < MIN_ACTIVITY_COVERAGE
                or activity["far_coverage"] < MIN_ACTIVITY_COVERAGE
                or activity["joint_coverage"] < MIN_ACTIVITY_COVERAGE
            ):
                raise StressModelError("valid double_talk activity rule is false")
        else:
            if any(counts.values()):
                raise StressModelError("valid quiet/noise activity has source activity")
    if scope == "echo" and label == "far_only":
        samples = activity.get("echo_guarded_active_samples")
        seconds = activity.get("echo_guarded_active_s")
        if (
            activity.get("echo_delay_guard_ms") != ECHO_DELAY_GUARD_MS
            or not isinstance(samples, int)
            or isinstance(samples, bool)
            or samples < 0
            or samples > activity["far_active_samples"]
            or not _finite_number(seconds)
            or abs(float(seconds) - samples / rate)
            > _ACTIVITY_NUMERIC_TOLERANCE
            or samples > segment_samples
        ):
            raise StressModelError("stress echo delay-guard evidence is invalid")


def _validate_stress_diagnostics_v2(
    diagnostics: Mapping[str, Any],
    *,
    expected_case_id: str,
    expected_tx_sha256: str,
    expected_scope: str,
    expected_mode: str,
    expected_category: str,
    expected_segments: list[dict[str, Any]],
) -> None:
    expected_rate = stress_rate(expected_mode)
    expected_unit = "dB" if expected_scope == "echo" else "percent"
    activity_gate = (
        diagnostics.get("activity_gate")
        if isinstance(diagnostics, Mapping)
        else None
    )
    if (
        not isinstance(diagnostics, Mapping)
        or diagnostics.get("schema_version") != SCHEMA_VERSION
        or diagnostics.get("case_id") != expected_case_id
        or diagnostics.get("tx_sha256") != expected_tx_sha256
        or not _is_sha256(diagnostics.get("tx_sha256"))
        or diagnostics.get("scope") != expected_scope
        or diagnostics.get("mode") != expected_mode
        or diagnostics.get("category") != expected_category
        or diagnostics.get("evidence_grade") != EVIDENCE_GRADE
        or diagnostics.get("formal_compliance") is not False
        or diagnostics.get("standard_compliance") is not False
        or diagnostics.get("formal_test_status") != FORMAL_TEST_STATUS
        or diagnostics.get("status")
        not in {"review_required", "not_measurable", "not_applicable"}
        or not isinstance(diagnostics.get("reason"), str)
        or not diagnostics["reason"]
        or not _finite_number(diagnostics.get("source_duration_s"))
        or abs(float(diagnostics["source_duration_s"]) - SOURCE_DURATION_S)
        > _TIME_TOLERANCE_S
        or not _finite_number(diagnostics.get("tx_duration_s"))
        or diagnostics.get("sample_rate_hz") != expected_rate
        or not isinstance(diagnostics.get("tx_muted"), bool)
        or not isinstance(diagnostics.get("review_required"), bool)
        or diagnostics.get("review_required")
        != (diagnostics.get("status") == "review_required")
        or not isinstance(diagnostics.get("limitations"), list)
        or not diagnostics["limitations"]
        or any(
            not isinstance(item, str) or not item
            for item in diagnostics["limitations"]
        )
        or (
            diagnostics.get("convergence_db") is not None
            and not _finite_number(diagnostics.get("convergence_db"))
        )
        or not isinstance(activity_gate, Mapping)
        or set(activity_gate)
        != {"frame_ms", "hop_ms", "rms_threshold", "min_active_s", "min_coverage"}
        or activity_gate.get("frame_ms") != ACTIVITY_FRAME_MS
        or activity_gate.get("hop_ms") != ACTIVITY_HOP_MS
        or activity_gate.get("rms_threshold") != ACTIVITY_RMS_THRESHOLD
        or activity_gate.get("min_active_s") != MIN_ACTIVITY_SECONDS
        or activity_gate.get("min_coverage") != MIN_ACTIVITY_COVERAGE
        or not isinstance(diagnostics.get("segments"), list)
    ):
        raise StressModelError("stress diagnostics identity or envelope is invalid")
    tx_duration = float(diagnostics["tx_duration_s"])
    source_duration = float(diagnostics["source_duration_s"])
    tx_invalid = tx_duration < 0.95 * source_duration or bool(
        diagnostics["tx_muted"]
    )
    if expected_scope != "echo" and tx_invalid and any(
        segment.get("status") == "review_required"
        for segment in diagnostics["segments"]
        if isinstance(segment, Mapping)
    ):
        raise StressModelError(
            "short stress TX cannot carry reviewable finite segment metrics"
        )
    if len(diagnostics["segments"]) != len(expected_segments):
        raise StressModelError("stress diagnostics segment count differs")
    for index, (segment, expected) in enumerate(
        zip(diagnostics["segments"], expected_segments, strict=True)
    ):
        if (
            not isinstance(segment, Mapping)
            or segment.get("label") != expected["label"]
            or segment.get("start_s") != expected["start_s"]
            or segment.get("end_s") != expected["end_s"]
            or segment.get("mode") != SEGMENT_MODE[expected["label"]]
            or segment.get("status")
            not in {"review_required", "not_measurable", "not_applicable"}
            or not isinstance(segment.get("metrics"), Mapping)
            or set(segment["metrics"]) != set(METRIC_KEYS)
            or not isinstance(segment.get("reason"), str)
            or not segment["reason"]
        ):
            raise StressModelError(
                f"stress diagnostics segment {index} differs from activity"
            )
        for value in segment["metrics"].values():
            if value is not None and not _finite_number(value):
                raise StressModelError("stress segment metric is not finite/null")
        activity = segment.get("activity")
        segment_samples = (
            int(round(float(expected["end_s"]) * expected_rate))
            - int(round(float(expected["start_s"]) * expected_rate))
        )
        _validate_activity_evidence(
            activity,
            label=expected["label"],
            scope=expected_scope,
            rate=expected_rate,
            segment_samples=segment_samples,
        )
        if not activity["valid"]:
            if segment["status"] != "not_measurable" or not _metrics_all_null(
                segment["metrics"]
            ):
                raise StressModelError(
                    f"stress segment {index} has invalid activity with metrics"
                )
        elif expected["label"] in {"quiet", "noise_only"}:
            if segment["status"] != "not_applicable" or not _metrics_all_null(
                segment["metrics"]
            ):
                raise StressModelError(
                    f"stress segment {index} has activity for a quiet label"
                )
        elif segment["status"] == "not_applicable":
            raise StressModelError(
                f"stress segment {index} cannot be not_applicable"
            )
        if (
            expected_scope == "echo"
            and expected["label"] == "far_only"
            and activity["echo_guarded_active_samples"]
            < max(
                1,
                int(round(MIN_ACTIVITY_SECONDS * expected_rate))
                - int(round(ACTIVITY_HOP_MS * expected_rate / 1000.0)) // 2,
            )
            and segment["metrics"].get("primary") is not None
        ):
            raise StressModelError(
                f"stress segment {index} promotes primary before delay guard"
            )
        if tx_invalid and expected_scope == "double_talk":
            if expected["label"] in {
                "near_only",
                "far_only",
                "double_talk",
            } and (
                segment["status"] != "not_measurable"
                or not _metrics_all_null(segment["metrics"])
            ):
                raise StressModelError(
                    f"stress segment {index} promotes metrics with invalid TX"
                )
        if tx_invalid and expected_scope == "echo" and expected["label"] == "far_only":
            if (
                segment["metrics"].get("tx_residual_reduction_db") is not None
                or segment["metrics"].get("tx_level_dbfs") is not None
            ):
                raise StressModelError(
                    f"stress segment {index} exposes TX metrics with invalid TX"
                )
        if segment["status"] == "review_required" and not any(
            _finite_number(value)
            for value in segment["metrics"].values()
        ):
            raise StressModelError(
                f"stress segment {index} is reviewable without a finite metric"
            )
    primary = diagnostics.get("primary")
    expected_primary_name = (
        "echo_primary_proxy_db"
        if expected_scope == "echo"
        else "double_talk_energy_ratio_proxy_percent"
    )
    if (
        not isinstance(primary, Mapping)
        or primary.get("name") != expected_primary_name
        or primary.get("unit") != expected_unit
        or primary.get("status")
        != (
            "review_required"
            if primary.get("value") is not None
            else "not_measured"
        )
        or (
            primary.get("value") is not None
            and not _finite_number(primary.get("value"))
        )
    ):
        raise StressModelError("stress primary diagnostic is invalid")
    if (
        expected_scope == "double_talk"
        and primary.get("value") is not None
        and not 0.0 <= float(primary["value"]) <= 150.0
    ):
        raise StressModelError("double-talk primary is outside its proxy range")
    if (
        (primary.get("value") is not None)
        != (diagnostics.get("status") == "review_required")
    ):
        raise StressModelError(
            "stress primary value and aggregate status differ"
        )
    if tx_invalid and expected_scope == "double_talk" and primary.get("value") is not None:
        raise StressModelError("double-talk primary cannot survive invalid TX")
    if tx_invalid and expected_scope == "echo" and diagnostics.get("convergence_db") is not None:
        raise StressModelError("echo convergence cannot survive invalid TX")
    if primary.get("value") is not None:
        review_label = "far_only" if expected_scope == "echo" else "double_talk"
        if not any(
            segment.get("label") == review_label
            and segment.get("status") == "review_required"
            and isinstance(segment.get("metrics"), Mapping)
            and _finite_number(segment["metrics"].get("primary"))
            for segment in diagnostics["segments"]
            if isinstance(segment, Mapping)
        ):
            raise StressModelError(
                "stress primary lacks a corresponding reviewable segment"
            )


def _validate_stress_diagnostics_v1(
    diagnostics: Mapping[str, Any],
    *,
    expected_case_id: str,
    expected_tx_sha256: str,
    expected_scope: str,
    expected_mode: str,
    expected_category: str,
    expected_segments: list[dict[str, Any]],
) -> None:
    """Preserve the original v1 semantics without claiming activity gating."""
    if not isinstance(diagnostics, Mapping):
        raise StressModelError("legacy stress diagnostics must be an object")
    expected_rate = stress_rate(expected_mode)
    expected_unit = "dB" if expected_scope == "echo" else "percent"
    if (
        diagnostics.get("schema_version") != LEGACY_SCHEMA_VERSION
        or diagnostics.get("case_id") != expected_case_id
        or diagnostics.get("tx_sha256") != expected_tx_sha256
        or not _is_sha256(diagnostics.get("tx_sha256"))
        or diagnostics.get("scope") != expected_scope
        or diagnostics.get("mode") != expected_mode
        or diagnostics.get("category") != expected_category
        or diagnostics.get("evidence_grade") != EVIDENCE_GRADE
        or diagnostics.get("formal_compliance") is not False
        or diagnostics.get("standard_compliance") is not False
        or diagnostics.get("formal_test_status") != FORMAL_TEST_STATUS
        or diagnostics.get("status")
        not in {"review_required", "not_measurable", "not_applicable"}
        or not isinstance(diagnostics.get("reason"), str)
        or not diagnostics["reason"]
        or not _finite_number(diagnostics.get("source_duration_s"))
        or abs(float(diagnostics["source_duration_s"]) - SOURCE_DURATION_S)
        > _TIME_TOLERANCE_S
        or not _finite_number(diagnostics.get("tx_duration_s"))
        or diagnostics.get("sample_rate_hz") != expected_rate
        or not isinstance(diagnostics.get("tx_muted"), bool)
        or not isinstance(diagnostics.get("review_required"), bool)
        or diagnostics.get("review_required")
        != (diagnostics.get("status") == "review_required")
        or not isinstance(diagnostics.get("limitations"), list)
        or not diagnostics["limitations"]
        or any(
            not isinstance(item, str) or not item
            for item in diagnostics["limitations"]
        )
        or (
            diagnostics.get("convergence_db") is not None
            and not _finite_number(diagnostics.get("convergence_db"))
        )
        or not isinstance(diagnostics.get("segments"), list)
    ):
        raise StressModelError("legacy stress diagnostics envelope is invalid")
    tx_duration = float(diagnostics["tx_duration_s"])
    source_duration = float(diagnostics["source_duration_s"])
    tx_invalid = tx_duration < 0.95 * source_duration or bool(
        diagnostics["tx_muted"]
    )
    if tx_invalid and expected_scope == "double_talk" and any(
        segment.get("status") == "review_required"
        for segment in diagnostics["segments"]
        if isinstance(segment, Mapping)
    ):
        raise StressModelError(
            "legacy double-talk TX cannot carry reviewable segment metrics"
        )
    if len(diagnostics["segments"]) != len(expected_segments):
        raise StressModelError("legacy stress diagnostics segment count differs")
    for index, (segment, expected) in enumerate(
        zip(diagnostics["segments"], expected_segments, strict=True)
    ):
        if (
            not isinstance(segment, Mapping)
            or segment.get("label") != expected["label"]
            or segment.get("start_s") != expected["start_s"]
            or segment.get("end_s") != expected["end_s"]
            or segment.get("mode") != SEGMENT_MODE[expected["label"]]
            or segment.get("status")
            not in {"review_required", "not_measurable", "not_applicable"}
            or not isinstance(segment.get("metrics"), Mapping)
            or set(segment["metrics"]) != set(METRIC_KEYS)
            or not isinstance(segment.get("reason"), str)
            or not segment["reason"]
        ):
            raise StressModelError(
                f"legacy stress diagnostics segment {index} differs from activity"
            )
        for value in segment["metrics"].values():
            if value is not None and not _finite_number(value):
                raise StressModelError(
                    "legacy stress segment metric is not finite/null"
                )
        if tx_invalid and expected_scope == "double_talk" and (
            expected["label"] in {"near_only", "far_only", "double_talk"}
            and (
                segment["status"] != "not_measurable"
                or not _metrics_all_null(segment["metrics"])
            )
        ):
            raise StressModelError(
                f"legacy stress segment {index} promotes metrics with invalid TX"
            )
    primary = diagnostics.get("primary")
    expected_primary_name = (
        "echo_primary_proxy_db"
        if expected_scope == "echo"
        else "double_talk_energy_ratio_proxy_percent"
    )
    if (
        not isinstance(primary, Mapping)
        or primary.get("name") != expected_primary_name
        or primary.get("unit") != expected_unit
        or primary.get("status")
        != (
            "review_required"
            if primary.get("value") is not None
            else "not_measured"
        )
        or (
            primary.get("value") is not None
            and not _finite_number(primary.get("value"))
        )
    ):
        raise StressModelError("legacy stress primary diagnostic is invalid")
    if (
        expected_scope == "double_talk"
        and primary.get("value") is not None
        and not 0.0 <= float(primary["value"]) <= 150.0
    ):
        raise StressModelError("legacy double-talk primary is outside its proxy range")
    if (
        (primary.get("value") is not None)
        != (diagnostics.get("status") == "review_required")
    ):
        raise StressModelError(
            "legacy stress primary value and aggregate status differ"
        )
    if tx_invalid and expected_scope == "double_talk" and primary.get("value") is not None:
        raise StressModelError("legacy double-talk primary cannot survive invalid TX")
    if tx_invalid and diagnostics.get("convergence_db") is not None:
        raise StressModelError("legacy convergence cannot survive invalid TX")
    if primary.get("value") is not None:
        review_label = "far_only" if expected_scope == "echo" else "double_talk"
        if not any(
            segment.get("label") == review_label
            and segment.get("status") == "review_required"
            and isinstance(segment.get("metrics"), Mapping)
            and _finite_number(segment["metrics"].get("primary"))
            for segment in diagnostics["segments"]
            if isinstance(segment, Mapping)
        ):
            raise StressModelError(
                "legacy stress primary lacks a corresponding reviewable segment"
            )


def validate_stress_diagnostics(
    diagnostics: Mapping[str, Any],
    **expected: Any,
) -> None:
    """Dispatch by schema; v2 can never fall back to legacy v1 validation."""
    if not isinstance(diagnostics, Mapping):
        raise StressModelError("stress diagnostics must be an object")
    schema_version = diagnostics.get("schema_version")
    if schema_version == LEGACY_SCHEMA_VERSION:
        _validate_stress_diagnostics_v1(diagnostics, **expected)
        return
    if schema_version == SCHEMA_VERSION:
        _validate_stress_diagnostics_v2(diagnostics, **expected)
        return
    raise StressModelError("unsupported stress diagnostics schema version")


def calculate_stress_case(
    case: Mapping[str, Any],
    input_paths: Mapping[str, Path],
    outputs: Mapping[str, Path],
    execution: Mapping[str, Any],
    *,
    report_id: str,
    profile: StressProfile,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return one stress contract row and its supplemental diagnostic."""
    del execution
    if not isinstance(report_id, str) or not report_id:
        raise StressModelError("stress report_id is invalid")
    item_ids = case.get("item_ids")
    if not isinstance(item_ids, list) or len(item_ids) != 1:
        raise StressModelError("stress case must request exactly one contract row")
    tx_path = outputs.get("tx")
    if tx_path is None:
        raise StressModelError("stress calculation requires outputs['tx']")
    diagnostics = diagnose_stress_case(
        case,
        input_paths,
        Path(tx_path),
        profile=profile,
    )
    primary = diagnostics["primary"]
    value = primary.get("value")
    if value is None:
        row = {
            "item_id": item_ids[0],
            "value": None,
            "status": "not_measured",
            "metric": None,
            "reason": diagnostics["reason"],
        }
    else:
        row = {
            "item_id": item_ids[0],
            "value": float(value),
            "status": "review_required",
            "metric": primary["name"],
            "reason": diagnostics["reason"],
        }
    return [row], diagnostics


__all__ = [
    "ACTIVITY_FRAME_MS",
    "ACTIVITY_HOP_MS",
    "ACTIVITY_RMS_THRESHOLD",
    "EVIDENCE_GRADE",
    "ECHO_CONVERGENCE_WINDOWS_S",
    "ECHO_DELAY_GUARD_MS",
    "FORMAL_COMPLIANCE",
    "FORMAL_TEST_STATUS",
    "LEGACY_SCHEMA_VERSION",
    "METRIC_KEYS",
    "MIN_ACTIVITY_COVERAGE",
    "MIN_ACTIVITY_SECONDS",
    "SCHEMA_VERSION",
    "SEGMENT_MODE",
    "SOURCE_DURATION_S",
    "StressModelError",
    "calculate_stress_case",
    "diagnose_stress_case",
    "stress_rate",
    "validate_activity_segments",
    "validate_stress_diagnostics",
]
