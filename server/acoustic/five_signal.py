"""Five-signal engineering assessment for the NV200 contract.

This module deliberately separates an engineering result from formal ITU-T
compliance.  It derives only values supported by the supplied PCM and maps them
onto the customer workbook rows.  Rows that require calibrated acoustic paths,
approved MOS algorithms, or a different signal direction remain not measured.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from threading import Lock
from typing import Literal
import uuid
import warnings

import numpy as np
from scipy import signal

from server.audio import AudioError, decode, power, resample
from . import service


ROLE_ORDER = ("near_end", "noise", "microphone", "far_end", "reference")
ROLE_LABELS = {
    "near_end": "近端音频",
    "noise": "噪音",
    "microphone": "麦克音",
    "far_end": "远端音频",
    "reference": "参考音",
}
DECODE_ROLE = {
    "near_end": "after",
    "noise": "reference",
    "microphone": "before",
    "far_end": "echo",
    "reference": "reference",
}
MODE_RATES = {
    "NB": 8000,
    "WB": 16000,
    "SWB": 48000,
}
MAX_BYTES = 32 * 1024 * 1024
_WRITE_LOCK = Lock()


class FiveSignalError(ValueError):
    """The uploaded five-signal set cannot be evaluated safely."""


@dataclass(frozen=True)
class InputTrace:
    role: str
    name: str
    samples: np.ndarray
    rate: int
    sha256: str
    duration_s: float


def _db(value: float) -> float:
    return float(10 * np.log10(max(float(value), 1e-30)))


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _mode_spec(mode: str) -> tuple[str, str, dict]:
    if not re.fullmatch(r"(NB|WB|SWB)-(wired|wireless|bt)", mode):
        raise FiveSignalError("mode must be one of the eight NV200 contract modes")
    bandwidth, transport = mode.split("-", 1)
    rows = service._items_for(bandwidth, transport)
    if len(rows) != 42:
        raise FiveSignalError(f"{mode} must resolve to exactly 42 contract rows")
    return bandwidth, transport, rows


def _decode_trace(role: str, name: str, blob: bytes) -> InputTrace:
    if not blob or len(blob) > MAX_BYTES:
        raise FiveSignalError(f"{ROLE_LABELS[role]}：文件为空或超过 32 MB")
    try:
        audio = decode(DECODE_ROLE[role], name or f"{role}.wav", blob, 0)
    except AudioError as error:
        raise FiveSignalError(f"{ROLE_LABELS[role]}：{error}") from error
    return InputTrace(
        role=role,
        name=Path(name or f"{role}.wav").name,
        samples=np.asarray(audio.data, dtype=np.float64),
        rate=int(audio.rate),
        sha256=_sha(blob),
        duration_s=len(audio.data) / audio.rate,
    )


def _resample_trace(trace: InputTrace, rate: int) -> np.ndarray:
    if trace.rate == rate:
        return trace.samples.copy()
    common = math.gcd(rate, trace.rate)
    return signal.resample_poly(
        trace.samples, rate // common, trace.rate // common
    )


def _trim_to_common(traces: dict[str, InputTrace], rate: int) -> dict[str, np.ndarray]:
    values = {role: _resample_trace(trace, rate) for role, trace in traces.items()}
    timeline_roles = ("near_end", "microphone", "far_end", "reference")
    length = min(len(values[role]) for role in timeline_roles)
    if length < round(rate * 0.25):
        raise FiveSignalError("五路音频的共同有效长度必须至少为 0.25 秒")
    result = {
        role: np.asarray(values[role][:length], dtype=np.float64)
        for role in timeline_roles
    }
    # A stand-alone noise recording need not share the speech-file duration.
    result["noise"] = np.asarray(values["noise"], dtype=np.float64)
    return result


def _alignment_delay(
    reference: np.ndarray, target: np.ndarray, rate: int, max_delay_ms: float = 2000
) -> dict:
    from server.audio import align

    try:
        _, _, row = align(reference, target, rate, max_delay_ms)
        return {
            **row,
            "available": bool(row.get("reliable")),
            "note": row.get("note") or "",
        }
    except AudioError as error:
        return {
            "available": False,
            "delay_ms": None,
            "correlation": None,
            "reliable": False,
            "note": str(error),
        }


def _shift_aligned(
    target: np.ndarray, lag_samples: int, length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Place target on the reference timeline; lag means target is later."""
    result = np.zeros(length, dtype=np.float64)
    valid = np.zeros(length, dtype=bool)
    if abs(lag_samples) >= length:
        return result, valid
    if lag_samples >= 0:
        usable = length - lag_samples
        result[:usable] = target[lag_samples:lag_samples + usable]
        valid[:usable] = True
    else:
        usable = length + lag_samples
        result[-lag_samples:] = target[:usable]
        valid[-lag_samples:] = True
    return result, valid


def _frame_power(x: np.ndarray, rate: int) -> tuple[np.ndarray, int, int]:
    window = max(1, round(rate * 0.02))
    hop = max(1, round(rate * 0.01))
    if len(x) < window:
        raise FiveSignalError("音频过短，无法建立 20 ms 分析帧")
    kernel = np.ones(window, dtype=np.float64) / window
    energies = np.convolve(x * x, kernel, mode="valid")[::hop]
    return np.asarray(energies, dtype=np.float64), window, hop


def _activity(x: np.ndarray, rate: int) -> tuple[np.ndarray, int, int, float]:
    energies, window, hop = _frame_power(x, rate)
    finite = energies[np.isfinite(energies)]
    if not len(finite) or float(np.max(finite)) <= 1e-20:
        return np.zeros(len(energies), dtype=bool), window, hop, -150.0
    levels = 10 * np.log10(np.maximum(finite, 1e-30))
    floor_db = float(np.percentile(levels, 20))
    dynamic_range = float(np.percentile(levels, 90) - floor_db)
    threshold = max(-64.0, floor_db + min(10.0, max(3.0, dynamic_range / 4)))
    return levels > threshold, window, hop, floor_db


def _frame_mask(active: np.ndarray, window: int, hop: int, length: int) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for index in np.flatnonzero(active):
        start = int(index) * hop
        mask[start:min(length, start + window)] = True
    return mask


def _masked(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return x[mask & np.isfinite(x)]


def _sisdr(reference: np.ndarray, observed: np.ndarray) -> dict:
    if reference.shape != observed.shape or not len(reference):
        return {
            "available": False,
            "sisdr_db": None,
            "projection_gain_db": None,
            "speech_power": 0.0,
            "residual_power": 0.0,
            "reason": "signals have no common support",
        }
    x = reference - float(np.mean(reference))
    y = observed - float(np.mean(observed))
    energy = float(np.dot(x, x))
    if energy / max(1, len(x)) < 1e-15:
        return {
            "available": False,
            "sisdr_db": None,
            "projection_gain_db": None,
            "speech_power": 0.0,
            "residual_power": 0.0,
            "reason": "reference has no AC speech excitation",
        }
    alpha = float(np.dot(x, y) / energy)
    speech = alpha * x
    residual = y - speech
    speech_power = float(np.mean(speech * speech))
    residual_power = float(np.mean(residual * residual))
    return {
        "available": True,
        "sisdr_db": min(120.0, _db(speech_power / max(residual_power, 1e-30))),
        "projection_gain_db": _db(alpha * alpha),
        "speech_power": speech_power,
        "residual_power": residual_power,
        "reason": "",
    }


def _stoi(reference: np.ndarray, observed: np.ndarray, rate: int) -> dict:
    try:
        from pystoi import stoi

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = float(stoi(reference, observed, rate, extended=False))
        if caught or not math.isfinite(value):
            return {"available": False, "value": None, "reason": "STOI warning or non-finite value"}
        return {"available": True, "value": value, "reason": ""}
    except ImportError:
        return {"available": False, "value": None, "reason": "pystoi is not installed"}
    except (TypeError, ValueError) as error:
        return {"available": False, "value": None, "reason": f"STOI unavailable: {error}"}


def _mask_power(x: np.ndarray, mask: np.ndarray) -> float | None:
    values = _masked(x, mask)
    return power(values) if len(values) >= 32 else None


def _levels(x: np.ndarray) -> dict:
    if not len(x) or not np.isfinite(x).all():
        return {"rms_dbfs": None, "peak_dbfs": None}
    return {
        "rms_dbfs": _db(power(x)),
        "peak_dbfs": float(20 * np.log10(max(float(np.max(np.abs(x))), 1e-15))),
    }


def _spectrum_metrics(x: np.ndarray, rate: int, upper_hz: float) -> dict:
    if len(x) < max(256, round(rate * 0.1)):
        return {
            "oob_suppression_db": None,
            "noise_peak_prominence_db": None,
            "flatness_db": None,
        }
    nperseg = min(4096, len(x))
    frequency, psd = signal.welch(x, rate, nperseg=nperseg, detrend="constant")
    usable = frequency > 20
    in_band = usable & (frequency <= min(upper_hz, rate / 2))
    out_band = usable & (frequency > min(upper_hz, rate / 2))
    if not np.any(in_band) or not np.any(out_band):
        oob = None
    else:
        oob = _db(float(np.sum(psd[in_band]) / max(float(np.sum(psd[out_band])), 1e-30)))
    band = psd[in_band & (psd > 1e-30)]
    if len(band) >= 3:
        peak = float(np.max(10 * np.log10(band)))
        median = float(np.median(10 * np.log10(band)))
        prominence = peak - median
        flatness = peak - float(np.min(10 * np.log10(band)))
    else:
        prominence = None
        flatness = None
    return {
        "oob_suppression_db": oob,
        "noise_peak_prominence_db": prominence,
        "flatness_db": flatness,
    }


def _transfer_margin(
    reference: np.ndarray,
    observed: np.ndarray,
    rate: int,
    mask: dict | None,
) -> tuple[float | None, float | None]:
    if not len(reference) or len(reference) != len(observed):
        return None, None
    nperseg = min(rate // 4, len(reference))
    if nperseg < 256:
        return None, None
    frequency, pxx = signal.welch(reference, rate, nperseg=nperseg, detrend="constant")
    _, pyy = signal.welch(observed, rate, nperseg=nperseg, detrend="constant")
    _, pxy = signal.csd(reference, observed, rate, nperseg=nperseg, detrend="constant")
    coherence = np.abs(pxy) ** 2 / np.maximum(pxx * pyy, 1e-30)
    gain = 20 * np.log10(
        np.maximum(np.abs(pxy) / np.maximum(pxx, 1e-30), 1e-15)
    )
    valid = (coherence >= 0.6) & (frequency > 20)
    if not np.any(valid):
        return None, None
    flatness = float(np.max(gain[valid]) - np.min(gain[valid]))
    if not mask:
        return None, flatness
    points = np.asarray(mask["frequency_hz"], dtype=float)
    lower = np.asarray(mask["lower_db"], dtype=float)
    upper = np.asarray(mask["upper_db"], dtype=float)
    normalized_at = float(mask["normalized_at_hz"])
    if (
        not len(points)
        or np.any(np.diff(points) <= 0)
        or points[0] > normalized_at
        or points[-1] < normalized_at
    ):
        return None, flatness
    values = np.interp(points, frequency, gain)
    values -= float(np.interp(normalized_at, frequency, gain))
    margin = float(np.min(np.minimum(values - lower, upper - values)))
    return margin, flatness


def _compare(value: float | None, lower, upper) -> str:
    if value is None or isinstance(value, bool) or not math.isfinite(value):
        return "review_required"
    lower_number = float(lower) if isinstance(lower, (int, float)) else None
    upper_number = float(upper) if isinstance(upper, (int, float)) else None
    if lower_number is None and upper_number is None:
        return "review_required"
    if lower_number is not None and value < lower_number:
        return "fail"
    if upper_number is not None and value > upper_number:
        return "fail"
    return "pass"


def _response_mask(bandwidth: str, direction: str) -> dict | None:
    if bandwidth != "SWB" or direction != "send":
        return None
    catalog = json.loads(service.CATALOG.read_text())
    return catalog.get("official_p1120_masks", {}).get("SWB_send")


def _persist_inputs(
    evaluation_id: str, traces: dict[str, InputTrace], blobs: dict[str, bytes]
) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for role in ROLE_ORDER:
        content_type = "audio/wav"
        record = service.save_evidence(
            traces[role].name, content_type, blobs[role]
        )
        record = service.evidence_with_namespace(
            record, f"five-signal-{evaluation_id[:12]}-{role}"
        )
        record = service.classify_evidence(record.artifact, "measurement_trace")
        records[role] = record.model_dump(mode="json")
    return records


def _persist_report(report: dict) -> dict:
    destination = service.BUILD / "five-signal-evaluations" / report["evaluation_id"]
    with _WRITE_LOCK:
        destination.mkdir(parents=True, exist_ok=False)
        path = destination / "result.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return {
        "path": str(path.relative_to(service.BUILD)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def get_evaluation(evaluation_id: str) -> dict | None:
    if not re.fullmatch(r"[a-f0-9]{32}", evaluation_id):
        return None
    path = service.BUILD / "five-signal-evaluations" / evaluation_id / "result.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def list_evaluations(limit: int = 20) -> list[dict]:
    root = service.BUILD / "five-signal-evaluations"
    if not root.is_dir():
        return []
    paths = sorted(
        root.glob("*/result.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    rows = []
    for path in paths[:max(1, min(limit, 100))]:
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(value, dict):
            continue
        rows.append({
            "evaluation_id": value.get("evaluation_id"),
            "created_at": value.get("created_at"),
            "mode": value.get("mode"),
            "summary": value.get("summary"),
            "standard_compliance": False,
            "formal_test_status": "not_measured",
        })
    return rows


def _family_source(
    *,
    family: str,
    direction: str,
    values: dict[str, float | None],
    lower,
    upper,
) -> tuple[str | None, float | None, str, str]:
    """Return metric id, value, status, reason."""
    if family == "round_trip_delay":
        return (
            "reference_to_output_delay_ms",
            values.get("reference_to_output_delay_ms"),
            "review_required",
            "相对时间峰不是完整网络/编解码往返 RTD；需端到端时间戳或物理回环",
        )
    if family == "slr" and direction == "send":
        return (
            "send_gain_db",
            values.get("send_gain_db"),
            "review_required",
            "五路数字 PCM 的发送增益不是校准后的 SLR",
        )
    if family in {"rlr", "rlr_linearity"}:
        return None, None, "not_measured", "当前五路输入是上行路径，没有校准的接收耳/扬声器输出"
    if family == "frequency_response":
        if direction != "send":
            return None, None, "not_measured", "缺少下行扬声器至人工耳的接收响应"
        metric = (
            "frequency_response_margin_db"
            if values.get("frequency_response_margin_db") is not None
            else "frequency_flatness_db"
        )
        return (
            metric,
            values.get(metric),
            "review_required",
            "发送频响由五路 PCM 计算；非 SWB 或缺少官方掩模时不自动判通过",
        )
    if family == "idle_noise_level":
        if direction != "send":
            return None, None, "not_measured", "没有下行扬声器空闲噪声采集"
        return (
            "output_noise_floor_dbfs",
            values.get("output_noise_floor_dbfs"),
            "review_required",
            "数字 dBFS 噪声电平不能直接等同于合同 dBPa/dBm0",
        )
    if family == "idle_noise_peaks":
        if direction != "send":
            return None, None, "not_measured", "没有下行空闲噪声频谱"
        return (
            "output_noise_peak_prominence_db",
            values.get("output_noise_peak_prominence_db"),
            "review_required",
            "已检测发送端谱峰突出度，但合同曲线/量纲尚未自动对齐",
        )
    if family == "listening_quality":
        if direction != "send":
            return None, None, "not_measured", "没有接收端 MOS-LQO 输入"
        return (
            "output_stoi",
            values.get("output_stoi"),
            "review_required",
            "STOI 是可懂度诊断，不能替代 TMOS/POLQA MOS 门限",
        )
    if family == "out_of_band":
        if direction != "send":
            return None, None, "not_measured", "没有接收端带外响应采集"
        return (
            "out_of_band_suppression_db",
            values.get("out_of_band_suppression_db"),
            "review_required",
            "数字带内/带外功率比不是合同规定激励下的 Corrected value",
        )
    if family == "double_talk_classification":
        value = values.get("double_talk_transparent_percent")
        return (
            "double_talk_transparent_percent",
            value,
            _compare(value, lower, upper),
            "按双讲有效帧输出衰减 ≤3 dB 估计透传比例；不是 A1/A2 标准分类器",
        )
    if family == "tclw":
        value = values.get("echo_loss_db")
        return (
            "echo_loss_db",
            value,
            _compare(value, lower, upper),
            "参考播放至麦克风的工程耦合损耗；未执行 TCLw 计权和正式声学标定",
        )
    if family == "hats_snr":
        value = values.get("output_snr_db")
        return (
            "output_snr_db",
            value,
            _compare(value, lower, upper),
            "由近端语音和麦克风噪声帧估计；没有 HATS 声压标定",
        )
    if family == "cng_level":
        return (
            "noise_level_change_db",
            values.get("noise_level_change_db"),
            "review_required",
            "由处理前后噪声段得到，未识别 CNG 接通/断开状态",
        )
    if family == "cng_spectrum":
        return (
            "noise_spectrum_change_db",
            values.get("noise_spectrum_change_db"),
            "review_required",
            "处理前后噪声谱差异不是合同 CNG 频谱掩模余量",
        )
    if family in {"g_mos", "n_mos", "s_mos"}:
        return None, None, "not_measured", "五路 PCM 未提供批准的 G/N/S-MOS 模型或主观评分"
    return None, None, "not_measured", "该测项无法由当前五路信号唯一映射"


def evaluate_five_signal(
    *,
    mode: str,
    uploads: dict[str, tuple[str, bytes]],
    channel: int = 0,
    max_delay_ms: float = 2000,
) -> dict:
    bandwidth, transport, rows = _mode_spec(mode)
    if set(uploads) != set(ROLE_ORDER):
        missing = sorted(set(ROLE_ORDER) - set(uploads))
        extra = sorted(set(uploads) - set(ROLE_ORDER))
        raise FiveSignalError(f"五路输入不完整：missing={missing}, extra={extra}")
    if isinstance(channel, bool) or not isinstance(channel, int) or not 0 <= channel <= 7:
        raise FiveSignalError("channel must be 0..7")
    if not math.isfinite(max_delay_ms) or not 0 < max_delay_ms <= 5000:
        raise FiveSignalError("max_delay_ms must be in 0..5000")
    traces = {
        role: _decode_trace(role, uploads[role][0], uploads[role][1])
        for role in ROLE_ORDER
    }
    rate = min(48000, max(trace.rate for trace in traces.values()))
    samples = _trim_to_common(traces, rate)
    length = len(samples["reference"])

    reference_activity, window, hop, reference_floor = _activity(samples["reference"], rate)
    if np.count_nonzero(reference_activity) < 3:
        raise FiveSignalError("参考音没有足够的有效语音活动")
    far_activity, _, _, far_floor = _activity(samples["far_end"], rate)

    reference_mask = _frame_mask(reference_activity, window, hop, length)
    far_mask = _frame_mask(far_activity, window, hop, length)
    near_only_mask = reference_mask & ~far_mask
    far_only_mask = far_mask & ~reference_mask
    double_talk_mask = reference_mask & far_mask
    noise_only_mask = ~reference_mask & ~far_mask
    if np.count_nonzero(noise_only_mask) < round(rate * 0.05):
        energies, _, _ = _frame_power(samples["microphone"], rate)
        quiet = np.argsort(energies)[:max(3, len(energies) // 10)]
        active = np.zeros(len(energies), dtype=bool)
        active[quiet] = True
        noise_only_mask = _frame_mask(active, window, hop, length)

    mic_alignment = _alignment_delay(
        samples["reference"], samples["microphone"], rate, max_delay_ms
    )
    output_alignment = _alignment_delay(
        samples["reference"], samples["near_end"], rate, max_delay_ms
    )
    mic_lag = round((mic_alignment.get("delay_ms") or 0) * rate / 1000)
    output_lag = round((output_alignment.get("delay_ms") or 0) * rate / 1000)
    mic, mic_valid = _shift_aligned(samples["microphone"], mic_lag, length)
    output, output_valid = _shift_aligned(samples["near_end"], output_lag, length)

    fidelity_mask = near_only_mask & mic_valid & output_valid
    mic_sisdr = _sisdr(samples["reference"][fidelity_mask], mic[fidelity_mask])
    output_sisdr = _sisdr(samples["reference"][fidelity_mask], output[fidelity_mask])
    mic_stoi = _stoi(samples["reference"][fidelity_mask], mic[fidelity_mask], rate)
    output_stoi = _stoi(samples["reference"][fidelity_mask], output[fidelity_mask], rate)

    mic_noise = _mask_power(mic, noise_only_mask & mic_valid)
    output_noise = _mask_power(output, noise_only_mask & output_valid)
    noise_reference = power(samples["noise"])
    mic_speech = _mask_power(mic, near_only_mask & mic_valid)
    output_speech = _mask_power(output, near_only_mask & output_valid)
    reference_speech = _mask_power(samples["reference"], near_only_mask)
    far_power = _mask_power(samples["far_end"], far_only_mask)
    mic_echo = _mask_power(mic, far_only_mask & mic_valid)
    output_echo = _mask_power(output, far_only_mask & output_valid)

    values: dict[str, float | None] = {
        "reference_to_microphone_delay_ms": mic_alignment.get("delay_ms"),
        "reference_to_output_delay_ms": output_alignment.get("delay_ms"),
        "microphone_correlation": mic_alignment.get("correlation"),
        "output_correlation": output_alignment.get("correlation"),
        "microphone_sisdr_db": mic_sisdr["sisdr_db"],
        "output_sisdr_db": output_sisdr["sisdr_db"],
        "sisdr_delta_db": (
            output_sisdr["sisdr_db"] - mic_sisdr["sisdr_db"]
            if output_sisdr["sisdr_db"] is not None and mic_sisdr["sisdr_db"] is not None
            else None
        ),
        "microphone_stoi": mic_stoi["value"],
        "output_stoi": output_stoi["value"],
        "stoi_delta": (
            output_stoi["value"] - mic_stoi["value"]
            if output_stoi["value"] is not None and mic_stoi["value"] is not None
            else None
        ),
        "send_gain_db": (
            _db(output_speech / reference_speech)
            if output_speech not in (None, 0.0) and reference_speech not in (None, 0.0)
            else None
        ),
        "microphone_noise_floor_dbfs": _db(mic_noise) if mic_noise is not None else None,
        "output_noise_floor_dbfs": _db(output_noise) if output_noise is not None else None,
        "noise_reference_dbfs": _db(noise_reference) if noise_reference is not None else None,
        "noise_level_change_db": (
            _db(output_noise / mic_noise)
            if output_noise not in (None, 0.0) and mic_noise not in (None, 0.0)
            else None
        ),
        "output_snr_db": (
            _db(output_sisdr["speech_power"] / output_noise)
            if output_sisdr["speech_power"] > 0 and output_noise not in (None, 0.0)
            else None
        ),
        "input_snr_db": (
            _db(mic_sisdr["speech_power"] / mic_noise)
            if mic_sisdr["speech_power"] > 0 and mic_noise not in (None, 0.0)
            else None
        ),
        "echo_loss_db": (
            _db(far_power / mic_echo)
            if far_power not in (None, 0.0) and mic_echo not in (None, 0.0)
            else None
        ),
        "echo_reduction_db": (
            _db(mic_echo / output_echo)
            if mic_echo not in (None, 0.0) and output_echo not in (None, 0.0)
            else None
        ),
    }

    if np.count_nonzero(double_talk_mask) >= round(rate * 0.05):
        mic_dt = _mask_power(mic, double_talk_mask & mic_valid)
        output_dt = _mask_power(output, double_talk_mask & output_valid)
        values["double_talk_transparent_percent"] = (
            float(np.clip(100.0 * output_dt / mic_dt, 0.0, 150.0))
            if mic_dt not in (None, 0.0) and output_dt is not None
            else None
        )
    else:
        values["double_talk_transparent_percent"] = None

    analysis_mask = near_only_mask & output_valid
    response, flatness = _transfer_margin(
        samples["reference"][analysis_mask],
        output[analysis_mask],
        rate,
        _response_mask(bandwidth, "send"),
    )
    values["frequency_response_margin_db"] = response
    values["frequency_flatness_db"] = flatness

    upper_hz = {"NB": 4000.0, "WB": 8000.0, "SWB": 14000.0}[bandwidth]
    spectrum_values = _spectrum_metrics(
        output[analysis_mask], rate, upper_hz
    )
    values.update(spectrum_values)
    noise_only_output = output[noise_only_mask & output_valid]
    noise_values = _spectrum_metrics(noise_only_output, rate, upper_hz)
    values["output_noise_peak_prominence_db"] = noise_values[
        "noise_peak_prominence_db"
    ]
    if mic_noise not in (None, 0.0) and output_noise not in (None, 0.0):
        values["noise_spectrum_change_db"] = abs(
            _db(output_noise / mic_noise)
        )
    else:
        values["noise_spectrum_change_db"] = None

    input_metadata = {
        role: {
            "name": trace.name,
            "sha256": trace.sha256,
            "sample_rate": trace.rate,
            "duration_s": trace.duration_s,
            **_levels(trace.samples),
        }
        for role, trace in traces.items()
    }
    segment_counts = {
        "near_only_seconds": float(np.count_nonzero(near_only_mask) / rate),
        "far_only_seconds": float(np.count_nonzero(far_only_mask) / rate),
        "double_talk_seconds": float(np.count_nonzero(double_talk_mask) / rate),
        "noise_only_seconds": float(np.count_nonzero(noise_only_mask) / rate),
    }

    items = []
    for row in rows:
        metric, value, status, reason = _family_source(
            family=row["family"],
            direction=row["direction"],
            values=values,
            lower=row.get("lower_limit"),
            upper=row.get("upper_limit"),
        )
        if value is not None and not math.isfinite(value):
            value = None
            status = "invalid"
            reason = "计算值不是有限数"
        items.append({
            "item_id": row["id"],
            "source_cell": row["source_cell"],
            "family": row["family"],
            "direction": row["direction"],
            "test_name": row["test_name"],
            "unit": row["unit"],
            "lower_limit": row.get("lower_limit"),
            "upper_limit": row.get("upper_limit"),
            "metric": metric,
            "value": value,
            "status": status,
            "reason": reason,
            "evidence_grade": (
                "not_measured" if status == "not_measured" else
                "engineering_proxy"
            ),
            "standard_compliance": False,
        })

    counts = {
        status: sum(item["status"] == status for item in items)
        for status in ("pass", "fail", "review_required", "not_measured", "invalid")
    }
    measured = len(items) - counts["not_measured"] - counts["invalid"]
    evaluation_id = uuid.uuid4().hex
    report = {
        "schema_version": 1,
        "evaluation_id": evaluation_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "bandwidth": bandwidth,
        "transport": transport,
        "scope": (
            "five-signal engineering assessment against NV200 customer rows; "
            "not an ITU-T formal acoustic qualification"
        ),
        "standard_compliance": False,
        "formal_test_status": "not_measured",
        "input_rate_hz": rate,
        "channel": channel,
        "inputs": input_metadata,
        "analysis": {
            "reference_floor_dbfs": reference_floor,
            "far_end_floor_dbfs": far_floor,
            "segments": segment_counts,
            "alignments": {
                "microphone": mic_alignment,
                "near_end": output_alignment,
            },
            "values": values,
        },
        "summary": {
            "total": len(items),
            "measured": measured,
            **counts,
        },
        "items": items,
        "limitations": [
            "五路 PCM 只覆盖可从给定上行/回声/噪声时间线推导的工程指标。",
            "数字 dBFS、STOI、相对时延和工程回声损耗不能替代校准 SLR/RLR/TCLw/TMOS/POLQA。",
            "NB/WB/SWB 是业务配置，不能从采样率或同一组音频自动推断另外七个模式。",
            "接收方向、HATS 标定、POLQA/TMOS、G/N/S-MOS 和正式 CNG 判据仍需对应测量或方法包。",
        ],
    }
    report["evidence"] = _persist_inputs(evaluation_id, traces, {
        role: uploads[role][1] for role in ROLE_ORDER
    })
    report["result"] = _persist_report(report)
    return report
