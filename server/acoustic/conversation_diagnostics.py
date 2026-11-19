"""Engineering-only diagnostics for the conversation scene-condition datasets.

The metrics in this module are deliberately supplemental.  They never change
the five contract rows and are never promoted to formal ITU-T evidence.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


SCHEMA_VERSION = "scene-condition-conversation-diagnostics-v1"
EVIDENCE_GRADE = "engineering_proxy"
FORMAL_COMPLIANCE = False
FORMAL_TEST_STATUS = "not_measured"

CONVERSATION_MODES = (
    "near_single",
    "far_single",
    "double_talk",
    "transition",
    "noise_only",
)
EXPECTED_SEGMENT_LABELS = {
    "near_single": ("speech",),
    "far_single": ("far_speech",),
    "double_talk": ("double_talk",),
    "transition": (
        "quiet",
        "far_only",
        "near_only",
        "far_only",
        "double_talk",
        "noise_only",
    ),
    "noise_only": (
        "cng_active",
        "cng_settling",
        "cng_noise_observation",
    ),
}
EXPECTED_RENDER_KIND = {
    "near_single": "speech",
    "far_single": "speech",
    "double_talk": "speech",
    "transition": "speech",
    "noise_only": "cng",
}
EXPECTED_SEGMENT_MODE = {
    "speech": "near",
    "far_speech": "far",
    "double_talk": "double_talk",
    "quiet": "not_applicable",
    "far_only": "far",
    "near_only": "near",
    "noise_only": "noise",
    "cng_active": "not_applicable",
    "cng_settling": "not_applicable",
    "cng_noise_observation": "noise",
}

_SILENCE_RMS = 1e-9
_MAX_ALIGNMENT_MS = 100.0
_TIME_TOLERANCE_S = 1e-6


class ConversationDiagnosticsError(ValueError):
    """The conversation metadata or one of its audio inputs is invalid."""


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
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def conversation_mode_from_provenance(provenance: Mapping[str, Any]) -> str:
    mode = provenance.get("conversation_mode")
    if not isinstance(mode, str) or mode not in CONVERSATION_MODES:
        raise ConversationDiagnosticsError(
            "provenance.conversation_mode must be one of "
            + ", ".join(CONVERSATION_MODES)
        )
    return mode


def validate_activity_metadata(
    provenance: Mapping[str, Any],
    *,
    expected_mode: str | None = None,
    expected_kind: str | None = None,
    expected_duration_s: float = 12.0,
    expected_sample_rate_hz: int | None = None,
) -> dict[str, Any]:
    """Validate one v2 conversation provenance and return its activity view."""
    if not isinstance(provenance, Mapping):
        raise ConversationDiagnosticsError("case provenance must be an object")
    mode = conversation_mode_from_provenance(provenance)
    if expected_mode is not None and mode != expected_mode:
        raise ConversationDiagnosticsError(
            f"conversation_mode {mode!r} differs from {expected_mode!r}"
        )
    render_metadata = provenance.get("render_metadata")
    if not isinstance(render_metadata, Mapping):
        raise ConversationDiagnosticsError(
            "provenance.render_metadata must be an object"
        )
    kind = render_metadata.get("kind")
    expected_kind_value = EXPECTED_RENDER_KIND[mode]
    if kind != expected_kind_value:
        raise ConversationDiagnosticsError(
            f"{mode} render kind must be {expected_kind_value!r}"
        )
    if expected_kind is not None and kind != expected_kind:
        raise ConversationDiagnosticsError(
            f"render kind {kind!r} differs from {expected_kind!r}"
        )

    duration_s = render_metadata.get("duration_s")
    if (
        not _finite_number(duration_s)
        or abs(float(duration_s) - float(expected_duration_s))
        > _TIME_TOLERANCE_S
    ):
        raise ConversationDiagnosticsError(
            f"render duration must be {expected_duration_s:g} s"
        )
    sample_rate_hz = render_metadata.get("sample_rate_hz")
    if expected_sample_rate_hz is None:
        bw = render_metadata.get("bandwidth")
        target_sample_rate_hz = (
            {"NB": 8_000, "WB": 16_000, "SWB": 48_000}[str(bw)]
            if bw in {"NB", "WB", "SWB"}
            else 16_000
        )
    else:
        target_sample_rate_hz = int(expected_sample_rate_hz)
    if (
        not isinstance(sample_rate_hz, (int, np.integer))
        or isinstance(sample_rate_hz, (bool, np.bool_))
        or int(sample_rate_hz) != target_sample_rate_hz
    ):
        raise ConversationDiagnosticsError(
            f"render sample rate must be {target_sample_rate_hz} Hz"
        )
    analysis_start_s = provenance.get("analysis_start_s")
    if (
        not _finite_number(analysis_start_s)
        or float(analysis_start_s) < 0.0
        or float(analysis_start_s) >= float(expected_duration_s)
    ):
        raise ConversationDiagnosticsError(
            "provenance.analysis_start_s must be inside the 12 s activity"
        )

    raw_segments = render_metadata.get("segments")
    if not isinstance(raw_segments, Sequence) or isinstance(
        raw_segments,
        (str, bytes, bytearray),
    ):
        raise ConversationDiagnosticsError(
            "render_metadata.segments must be a list"
        )
    expected_labels = EXPECTED_SEGMENT_LABELS[mode]
    if len(raw_segments) != len(expected_labels):
        raise ConversationDiagnosticsError(
            f"{mode} must expose {len(expected_labels)} activity segments"
        )
    segments: list[dict[str, Any]] = []
    previous_end = 0.0
    for index, (raw_segment, expected_label) in enumerate(
        zip(raw_segments, expected_labels, strict=True)
    ):
        if not isinstance(raw_segment, Mapping):
            raise ConversationDiagnosticsError(
                f"{mode} segment {index} must be an object"
            )
        if set(raw_segment) != {"start_s", "end_s", "label"}:
            raise ConversationDiagnosticsError(
                f"{mode} segment {index} has unexpected fields"
            )
        start_s = raw_segment.get("start_s")
        end_s = raw_segment.get("end_s")
        label = raw_segment.get("label")
        if (
            label != expected_label
            or not _finite_number(start_s)
            or not _finite_number(end_s)
        ):
            raise ConversationDiagnosticsError(
                f"{mode} segment {index} identity or bounds are invalid"
            )
        start = float(start_s)
        end = float(end_s)
        if (
            start < 0.0
            or end > float(expected_duration_s)
            or end <= start
            or abs(start - previous_end) > _TIME_TOLERANCE_S
        ):
            raise ConversationDiagnosticsError(
                f"{mode} segments must be contiguous and inside 0..12 s"
            )
        if index == 0 and abs(start) > _TIME_TOLERANCE_S:
            raise ConversationDiagnosticsError(
                f"{mode} activity must start at 0 s"
            )
        segments.append(
            {
                "label": label,
                "start_s": start,
                "end_s": end,
            }
        )
        previous_end = end
    if (
        abs(previous_end - float(expected_duration_s))
        > _TIME_TOLERANCE_S
    ):
        raise ConversationDiagnosticsError(
            f"{mode} activity must end at {expected_duration_s:g} s"
        )
    return {
        "conversation_mode": mode,
        "kind": kind,
        "duration_s": float(duration_s),
        "sample_rate_hz": int(sample_rate_hz),
        "analysis_start_s": float(analysis_start_s),
        "segments": segments,
    }


def activity_segments(provenance: Mapping[str, Any]) -> list[dict[str, Any]]:
    return validate_activity_metadata(provenance)["segments"]


def _load_mono(path: Path, label: str) -> tuple[np.ndarray, int]:
    path = Path(path)
    try:
        samples, rate = sf.read(
            path,
            dtype="float64",
            always_2d=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ConversationDiagnosticsError(
            f"{label} could not be decoded: {error}"
        ) from error
    if (
        not isinstance(rate, (int, np.integer))
        or int(rate) <= 0
        or samples.ndim != 2
        or samples.shape[1] < 1
        or len(samples) == 0
        or not np.isfinite(samples).all()
    ):
        raise ConversationDiagnosticsError(
            f"{label} has invalid samples, channels, or sample rate"
        )
    return (
        np.asarray(samples[:, 0], dtype=np.float64),
        int(rate),
    )


def _rms(samples: np.ndarray) -> float:
    values = np.asarray(samples, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(values))))


def _dbfs(value: float) -> float | None:
    if not math.isfinite(value) or value <= _SILENCE_RMS:
        return None
    return float(20.0 * math.log10(value))


def _prefix_square_sum(values: np.ndarray) -> np.ndarray:
    result = np.empty(len(values) + 1, dtype=np.float64)
    result[0] = 0.0
    np.cumsum(np.square(values), out=result[1:])
    return result


def _bounded_alignment(
    reference: np.ndarray,
    transmitted: np.ndarray,
    rate: int = 16_000,
) -> dict[str, Any] | None:
    """Return bounded FFT cross-correlation alignment and Pearson correlation."""
    length = min(len(reference), len(transmitted))
    if length < 2:
        return None
    ref = np.asarray(reference[:length], dtype=np.float64)
    tx = np.asarray(transmitted[:length], dtype=np.float64)
    if _rms(ref) <= _SILENCE_RMS or _rms(tx) <= _SILENCE_RMS:
        return None
    ref = ref - float(np.mean(ref))
    tx = tx - float(np.mean(tx))
    if not np.isfinite(ref).all() or not np.isfinite(tx).all():
        return None

    max_lag = min(
        length - 1,
        int(round(_MAX_ALIGNMENT_MS * rate / 1000.0)),
    )
    if max_lag < 1:
        return None
    fft_size = 1 << (2 * length - 1).bit_length()
    spectrum = np.fft.rfft(tx, n=fft_size) * np.conj(
        np.fft.rfft(ref, n=fft_size)
    )
    correlation = np.fft.irfft(spectrum, n=fft_size)
    ref_energy = _prefix_square_sum(ref)
    tx_energy = _prefix_square_sum(tx)

    lags = np.arange(-max_lag, max_lag + 1, dtype=np.int64)
    dots = np.empty(len(lags), dtype=np.float64)
    overlap_counts = np.empty(len(lags), dtype=np.int64)
    ref_energies = np.empty(len(lags), dtype=np.float64)
    tx_energies = np.empty(len(lags), dtype=np.float64)
    for index, lag in enumerate(lags):
        if lag >= 0:
            overlap = length - int(lag)
            dots[index] = correlation[int(lag)]
            ref_energies[index] = ref_energy[overlap] - ref_energy[0]
            tx_energies[index] = tx_energy[length] - tx_energy[int(lag)]
        else:
            overlap = length + int(lag)
            dots[index] = correlation[fft_size + int(lag)]
            ref_energies[index] = ref_energy[length] - ref_energy[-int(lag)]
            tx_energies[index] = tx_energy[overlap] - tx_energy[0]
        overlap_counts[index] = overlap

    minimum_overlap = max(2, int(round(0.05 * rate)))
    denominators = np.sqrt(ref_energies * tx_energies)
    valid = (
        (overlap_counts >= minimum_overlap)
        & np.isfinite(denominators)
        & (denominators > 0.0)
    )
    if not valid.any():
        return None
    normalized = np.full(len(lags), np.nan, dtype=np.float64)
    normalized[valid] = dots[valid] / denominators[valid]
    if not np.isfinite(normalized[valid]).any():
        return None
    masked = np.where(valid, np.abs(normalized), -np.inf)
    peak_index = int(np.argmax(masked))
    lag = int(lags[peak_index])
    corr = float(normalized[peak_index])
    overlap = int(overlap_counts[peak_index])
    if lag >= 0:
        aligned_ref = ref[:overlap]
        aligned_tx = tx[lag:lag + overlap]
    else:
        aligned_ref = ref[-lag:-lag + overlap]
        aligned_tx = tx[:overlap]
    denominator = float(
        np.linalg.norm(aligned_ref) * np.linalg.norm(aligned_tx)
    )
    if denominator <= 0.0 or not math.isfinite(denominator):
        return None
    corr = float(np.dot(aligned_ref, aligned_tx) / denominator)
    if not math.isfinite(corr):
        return None
    return {
        "lag_samples": lag,
        "lag_ms": float(lag * 1000.0 / float(rate)),
        "correlation": max(-1.0, min(1.0, corr)),
        "overlap_samples": overlap,
        "aligned_reference": aligned_ref,
        "aligned_transmitted": aligned_tx,
    }


def _sample_bounds(
    start_s: float,
    end_s: float,
    rate: int,
) -> tuple[int, int]:
    start = int(round(start_s * rate))
    end = int(round(end_s * rate))
    return start, end


def _alignment_metrics(
    reference: np.ndarray,
    transmitted: np.ndarray,
    *,
    rate: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    alignment = _bounded_alignment(reference, transmitted, rate=rate)
    tx_dbfs = _dbfs(_rms(transmitted))
    if alignment is None:
        metrics = {
            "near_gain_db": None,
            "near_correlation": None,
            "echo_reduction_db": None,
            "tx_level_dbfs": tx_dbfs,
        }
        return metrics, {
            "lag_samples": None,
            "lag_ms": None,
            "overlap_samples": 0,
        }

    aligned_reference = alignment["aligned_reference"]
    aligned_transmitted = alignment["aligned_transmitted"]
    reference_energy = float(
        np.dot(aligned_reference, aligned_reference)
    )
    projection = (
        float(np.dot(aligned_transmitted, aligned_reference))
        / reference_energy
        if reference_energy > 0.0
        else float("nan")
    )
    gain_db = (
        20.0 * math.log10(projection)
        if math.isfinite(projection) and projection > 0.0
        else None
    )
    metrics = {
        "near_gain_db": gain_db,
        "near_correlation": alignment["correlation"],
        "echo_reduction_db": None,
        "tx_level_dbfs": tx_dbfs,
    }
    return metrics, {
        "lag_samples": alignment["lag_samples"],
        "lag_ms": alignment["lag_ms"],
        "overlap_samples": alignment["overlap_samples"],
        "rate_hz": rate,
    }


def _energy_reduction_metrics(
    microphone: np.ndarray,
    transmitted: np.ndarray,
    *,
    metric_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    length = min(len(microphone), len(transmitted))
    mic = np.asarray(microphone[:length], dtype=np.float64)
    tx = np.asarray(transmitted[:length], dtype=np.float64)
    mic_rms = _rms(mic)
    tx_rms = _rms(tx)
    muted = bool(length > 0 and tx_rms <= _SILENCE_RMS)
    mic_dbfs = _dbfs(mic_rms)
    tx_dbfs = _dbfs(tx_rms)
    reduction = None
    if length == 0:
        pass
    elif mic_rms <= _SILENCE_RMS:
        pass
    elif muted:
        pass
    else:
        reduction = float(20.0 * math.log10(mic_rms / tx_rms))
    metrics = {
        "near_gain_db": None,
        "near_correlation": None,
        "echo_reduction_db": reduction,
        "tx_level_dbfs": tx_dbfs,
    }
    return metrics, {
        "mic_rms_linear": mic_rms,
        "tx_rms_linear": tx_rms,
        "tx_muted": muted,
        "overlap_samples": length,
        "contains_noise": True,
        "pure_erle": False,
        "mic_rms_dbfs": mic_dbfs,
        "metric_name": metric_name,
    }


def _segment_audio(
    samples: np.ndarray,
    *,
    start_s: float,
    end_s: float,
    rate: int,
) -> tuple[np.ndarray, dict[str, int]]:
    start, end = _sample_bounds(start_s, end_s, rate)
    if start < 0 or end > len(samples) or end <= start:
        return np.asarray([], dtype=np.float64), {
            "start_sample": start,
            "end_sample": end,
            "selected_samples": 0,
        }
    return (
        np.asarray(samples[start:end], dtype=np.float64),
        {
            "start_sample": start,
            "end_sample": end,
            "selected_samples": end - start,
        },
    )


def _segment_mode(label: str) -> str:
    return EXPECTED_SEGMENT_MODE.get(label, "not_applicable")


def _segment_diagnostic(
    segment: Mapping[str, Any],
    *,
    mode: str,
    reference: np.ndarray,
    microphone: np.ndarray,
    transmitted: np.ndarray,
    rate: int,
) -> dict[str, Any]:
    label = str(segment["label"])
    start_s = float(segment["start_s"])
    end_s = float(segment["end_s"])
    ref, ref_bounds = _segment_audio(
        reference, start_s=start_s, end_s=end_s, rate=rate
    )
    mic, mic_bounds = _segment_audio(
        microphone, start_s=start_s, end_s=end_s, rate=rate
    )
    tx, tx_bounds = _segment_audio(
        transmitted, start_s=start_s, end_s=end_s, rate=rate
    )
    result: dict[str, Any] = {
        "label": label,
        "start_s": start_s,
        "end_s": end_s,
        "mode": _segment_mode(label),
        "source_bounds": {
            "reference": ref_bounds,
            "microphone": mic_bounds,
            "tx": tx_bounds,
        },
        "metrics": {},
    }
    expected_start, expected_end = _sample_bounds(start_s, end_s, rate)
    expected_samples = max(0, expected_end - expected_start)
    tx_overlap_samples = (
        max(0, min(expected_end, len(transmitted)) - expected_start)
        if expected_start < len(transmitted)
        else 0
    )
    overlap_ratio = (
        tx_overlap_samples / expected_samples if expected_samples else 0.0
    )
    result["tx_overlap_ratio"] = float(overlap_ratio)

    near_labels = {"speech", "near_only", "double_talk"}
    far_labels = {"far_speech", "far_only"}
    noise_labels = {"noise_only", "cng_noise_observation"}
    if (
        label in near_labels | far_labels | noise_labels
        and overlap_ratio < 0.95
    ):
        result["metrics"] = {
            "near_gain_db": None,
            "near_correlation": None,
            "echo_reduction_db": None,
            "tx_level_dbfs": _dbfs(_rms(tx)),
        }
        result["status"] = "not_measurable"
        result["reason"] = (
            "TX 与活动段重叠不足 95%，不按截短/错位索引伪造指标"
        )
        return result
    if label in near_labels:
        metrics, alignment = _alignment_metrics(
            ref,
            tx,
            rate=rate,
        )
        result["metrics"] = metrics
        result["alignment"] = alignment
        result["reason"] = (
            "clean reference 与真实 TX 的有界对齐相关度和参考投影增益；"
            "均为需复核的工程代理"
        )
    elif label in far_labels:
        metrics, details = _energy_reduction_metrics(
            mic,
            tx,
            metric_name="mic_to_tx_energy_reduction_proxy_db",
        )
        result["metrics"] = metrics
        result.update(details)
        result["reason"] = (
            "远端活动段 mic 到 TX 的含噪能量降低代理；"
            "不得解释为纯 ERLE 或正式回声门"
        )
        if details.get("tx_muted"):
            result["reason"] = (
                "远端活动段 TX 为静音；mute 不是衰减通过，"
                "echo_reduction_db 保持 null"
            )
    elif label in noise_labels:
        metrics, details = _energy_reduction_metrics(
            mic,
            tx,
            metric_name="mic_to_tx_noise_energy_reduction_proxy_db",
        )
        result["metrics"] = metrics
        result.update(details)
        result["reason"] = (
            "噪声活动段的含噪能量降低代理，仅作补充诊断"
        )
        if details.get("tx_muted"):
            result["reason"] = (
                "噪声活动段 TX 为静音；mute 不是通过，"
                "echo_reduction_db 保持 null"
            )
    else:
        result["reason"] = (
            f"{mode}/{label} 不是近端、远端或噪声适用段，"
            "不构造替代指标"
        )
        result["metrics"] = {
            "near_gain_db": None,
            "near_correlation": None,
            "echo_reduction_db": None,
            "tx_level_dbfs": _dbfs(_rms(tx)),
        }
    if result["mode"] in {"near", "double_talk"}:
        primary_metrics = (
            result["metrics"]["near_gain_db"],
            result["metrics"]["near_correlation"],
        )
    elif result["mode"] in {"far", "noise"}:
        primary_metrics = (result["metrics"]["echo_reduction_db"],)
    else:
        primary_metrics = ()
    if result["mode"] == "not_applicable":
        result["status"] = "not_applicable"
    elif any(_finite_number(value) for value in primary_metrics):
        result["status"] = "review_required"
    else:
        result["status"] = "not_measurable"
    return result


def diagnose_conversation_case(
    case: Mapping[str, Any],
    input_paths: Mapping[str, Path],
    tx_path: Path,
    *,
    tx_sha256: str | None = None,
) -> dict[str, Any]:
    """Return supplemental metrics for one real VCP TX and four inputs."""
    if not isinstance(case, Mapping):
        raise ConversationDiagnosticsError("case must be an object")
    provenance = case.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ConversationDiagnosticsError("case.provenance must be an object")
    activity = validate_activity_metadata(provenance)
    required_roles = ("microphone", "reference", "far_end", "noise")
    if set(input_paths) != set(required_roles):
        raise ConversationDiagnosticsError(
            "input_paths must contain exactly microphone/reference/far_end/noise"
        )
    audio: dict[str, np.ndarray] = {}
    rates: set[int] = set()
    for role in required_roles:
        samples, rate = _load_mono(Path(input_paths[role]), role)
        audio[role] = samples
        rates.add(rate)
    tx, tx_rate = _load_mono(Path(tx_path), "tx")
    rates.add(tx_rate)
    if len(rates) != 1:
        raise ConversationDiagnosticsError(
            "four inputs and TX must share one sample rate"
        )
    rate = next(iter(rates))
    if rate != activity["sample_rate_hz"]:
        raise ConversationDiagnosticsError(
            "activity metadata sample rate differs from decoded audio"
        )
    if any(len(samples) != len(audio["microphone"]) for samples in audio.values()):
        raise ConversationDiagnosticsError(
            "four input WAVs must have equal source lengths"
        )
    tx_hash = (
        tx_sha256
        if isinstance(tx_sha256, str) and len(tx_sha256) == 64
        else _sha256_file(Path(tx_path))
    )
    segments = [
        _segment_diagnostic(
            segment,
            mode=activity["conversation_mode"],
            reference=audio["reference"],
            microphone=audio["microphone"],
            transmitted=tx,
            rate=rate,
        )
        for segment in activity["segments"]
    ]
    segment_statuses = {segment["status"] for segment in segments}
    if "review_required" in segment_statuses:
        status = "review_required"
    elif "not_measurable" in segment_statuses:
        status = "not_measurable"
    else:
        status = "not_applicable"
    if activity["conversation_mode"] == "far_single":
        reason = (
            "far_single 没有持续近端语音，DNSMOS MOS 行明确不适用；"
            "保留真实 TX 与含噪 echo_reduction_db 工程代理"
        )
    elif activity["conversation_mode"] == "transition":
        reason = (
            "transition 由多个短活动段组成，DNSMOS MOS 行明确不适用；"
            "逐段保留 near/far/double-talk/noise 工程诊断"
        )
    else:
        reason = (
            "补充工程诊断，不改变正式五行门限；"
            "所有有限值均需复核"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case.get("case_id") or case.get("id"),
        "conversation_mode": activity["conversation_mode"],
        "evidence_grade": EVIDENCE_GRADE,
        "formal_compliance": FORMAL_COMPLIANCE,
        "standard_compliance": False,
        "formal_test_status": FORMAL_TEST_STATUS,
        "status": status,
        "reason": reason,
        "review_required": status == "review_required",
        "sample_rate_hz": rate,
        "source_duration_s": len(audio["microphone"]) / rate,
        "tx_duration_s": len(tx) / rate,
        "tx_sha256": tx_hash,
        "alignment_method": {
            "name": "bounded_fft_cross_correlation",
            "max_lag_ms": _MAX_ALIGNMENT_MS,
            "fixed_index_assumption": False,
        },
        "segments": segments,
        "limitations": [
            "engineering proxy only",
            "not formal ITU-T compliance evidence",
            "all finite values require review",
            "mic-to-TX energy reduction includes noise and is not pure ERLE",
            "mute or silence is not interpreted as a passing result",
        ],
    }


def validate_diagnostics(
    diagnostics: Mapping[str, Any],
    *,
    expected_mode: str,
    expected_case_id: str | None = None,
    expected_tx_sha256: str | None = None,
    expected_activity: Mapping[str, Any] | None = None,
) -> None:
    allowed_statuses = {
        "review_required",
        "not_measurable",
        "not_applicable",
    }
    if (
        not isinstance(diagnostics, Mapping)
        or diagnostics.get("schema_version") != SCHEMA_VERSION
        or diagnostics.get("conversation_mode") != expected_mode
        or diagnostics.get("evidence_grade") != EVIDENCE_GRADE
        or diagnostics.get("formal_compliance") is not False
        or diagnostics.get("standard_compliance") is not False
        or diagnostics.get("formal_test_status") != FORMAL_TEST_STATUS
        or diagnostics.get("status") not in allowed_statuses
        or not isinstance(diagnostics.get("reason"), str)
        or not diagnostics["reason"]
        or not isinstance(diagnostics.get("case_id"), str)
        or not diagnostics["case_id"]
        or not _is_sha256(diagnostics.get("tx_sha256"))
        or not _finite_number(diagnostics.get("source_duration_s"))
        or not _finite_number(diagnostics.get("tx_duration_s"))
        or not isinstance(diagnostics.get("sample_rate_hz"), (int, np.integer))
        or isinstance(
            diagnostics.get("sample_rate_hz"),
            (bool, np.bool_),
        )
        or not isinstance(diagnostics.get("segments"), list)
    ):
        raise ConversationDiagnosticsError(
            "conversation diagnostics schema, mode, or evidence grade is invalid"
        )
    if expected_case_id is not None and (
        diagnostics.get("case_id") != expected_case_id
    ):
        raise ConversationDiagnosticsError(
            "conversation diagnostics case_id differs from its case"
        )
    if expected_tx_sha256 is not None and (
        not _is_sha256(expected_tx_sha256)
        or diagnostics.get("tx_sha256") != expected_tx_sha256
    ):
        raise ConversationDiagnosticsError(
            "conversation diagnostics tx_sha256 differs from its output"
        )
    source_duration_s = float(diagnostics["source_duration_s"])
    tx_duration_s = float(diagnostics["tx_duration_s"])
    if (
        abs(source_duration_s - 12.0) > _TIME_TOLERANCE_S
        or tx_duration_s < 0.0
    ):
        raise ConversationDiagnosticsError(
            "conversation diagnostics source/TX duration is invalid"
        )
    expected_segments: Sequence[Mapping[str, Any]] | None = None
    if expected_activity is not None:
        if (
            not isinstance(expected_activity, Mapping)
            or expected_activity.get("conversation_mode") != expected_mode
            or not isinstance(expected_activity.get("segments"), Sequence)
        ):
            raise ConversationDiagnosticsError(
                "conversation diagnostics expected activity is invalid"
            )
        expected_segments = expected_activity["segments"]
        if (
            int(diagnostics["sample_rate_hz"])
            != int(expected_activity.get("sample_rate_hz", -1))
        ):
            raise ConversationDiagnosticsError(
                "conversation diagnostics sample rate differs from activity"
            )
    expected_labels = EXPECTED_SEGMENT_LABELS[expected_mode]
    labels = tuple(
        segment.get("label")
        for segment in diagnostics["segments"]
        if isinstance(segment, Mapping)
    )
    if labels != expected_labels:
        raise ConversationDiagnosticsError(
            "conversation diagnostics segment identities are invalid"
        )
    expected_metric_keys = {
        "near_gain_db",
        "near_correlation",
        "echo_reduction_db",
        "tx_level_dbfs",
    }
    for index, segment in enumerate(diagnostics["segments"]):
        if (
            not isinstance(segment, Mapping)
            or segment.get("status") not in allowed_statuses
            or not isinstance(segment.get("mode"), str)
            or not isinstance(segment.get("reason"), str)
            or not segment["reason"]
            or not _finite_number(segment.get("start_s"))
            or not _finite_number(segment.get("end_s"))
            or float(segment["end_s"]) <= float(segment["start_s"])
            or not isinstance(segment.get("metrics"), Mapping)
            or set(segment["metrics"]) != expected_metric_keys
        ):
            raise ConversationDiagnosticsError(
                "conversation diagnostics segment shape is invalid"
            )
        label = segment.get("label")
        if segment.get("mode") != EXPECTED_SEGMENT_MODE.get(
            str(label),
            "not_applicable",
        ):
            raise ConversationDiagnosticsError(
                "conversation diagnostics segment mode differs from its label"
            )
        if expected_segments is not None:
            if index >= len(expected_segments):
                raise ConversationDiagnosticsError(
                    "conversation diagnostics has extra segments"
                )
            expected_segment = expected_segments[index]
            if (
                not isinstance(expected_segment, Mapping)
                or segment.get("label") != expected_segment.get("label")
                or segment.get("start_s") != expected_segment.get("start_s")
                or segment.get("end_s") != expected_segment.get("end_s")
            ):
                raise ConversationDiagnosticsError(
                    "conversation diagnostics segment bounds differ from provenance"
                )
        for value in segment["metrics"].values():
            if value is not None and not _finite_number(value):
                raise ConversationDiagnosticsError(
                    "conversation diagnostics metric is not finite or null"
                )
    segment_statuses = {
        segment["status"] for segment in diagnostics["segments"]
    }
    expected_status = (
        "review_required"
        if "review_required" in segment_statuses
        else "not_measurable"
        if "not_measurable" in segment_statuses
        else "not_applicable"
    )
    if (
        diagnostics.get("status") != expected_status
        or diagnostics.get("review_required")
        is not (expected_status == "review_required")
    ):
        raise ConversationDiagnosticsError(
            "conversation diagnostics aggregate status is inconsistent"
        )
    if (
        expected_segments is not None
        and len(diagnostics["segments"]) != len(expected_segments)
    ):
        raise ConversationDiagnosticsError(
            "conversation diagnostics segment count differs from provenance"
        )
    if (
        tx_duration_s < 0.95 * source_duration_s
        and any(
            segment["status"] == "review_required"
            for segment in diagnostics["segments"]
        )
    ):
        raise ConversationDiagnosticsError(
            "short TX cannot carry reviewable finite diagnostics"
        )


__all__ = [
    "CONVERSATION_MODES",
    "ConversationDiagnosticsError",
    "EVIDENCE_GRADE",
    "EXPECTED_RENDER_KIND",
    "EXPECTED_SEGMENT_MODE",
    "EXPECTED_SEGMENT_LABELS",
    "FORMAL_COMPLIANCE",
    "FORMAL_TEST_STATUS",
    "SCHEMA_VERSION",
    "activity_segments",
    "conversation_mode_from_provenance",
    "diagnose_conversation_case",
    "validate_activity_metadata",
    "validate_diagnostics",
]
