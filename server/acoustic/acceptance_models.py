"""Contract-row dispatch for measurements derived from real VCP outputs.

This module is deliberately an engineering-evidence layer.  It never promotes
an uncalibrated proxy, a model estimate, or a customer-workbook threshold match
to ITU-T compliance.  Every returned row carries the exact case conditions and
the observed input/output hashes used to compute it.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from functools import lru_cache
import errno
import hashlib
from io import BytesIO
import math
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal
import soundfile as sf

from server.quality.evaluators import PROJECT as QUALITY_PROJECT, create_evaluators

from .acceptance_contract import matching_conditions, required_conditions
from .five_signal import evaluate_five_signal
from .receive_configuration import normalize_receive_configuration
from .receive_transfer import estimate_binaural_transfer
from .receive_quality_proxy import (
    MINIMUM_SECONDS as DNSMOS_MINIMUM_SECONDS,
    NATIVE_SAMPLE_RATE as DNSMOS_NATIVE_RATE,
    evaluate_binaural_dnsmos_with_evaluator,
)
from .receive_simulation import (
    evaluate_synthetic_receive,
    get_synthetic_receive_file,
)


SEND_INPUT_ROLES = ("reference", "noise", "microphone", "far_end")
INPUT_ROLE_ALIASES = {
    "reference": ("reference",),
    "noise": ("noise",),
    "microphone": ("microphone", "before"),
    "far_end": ("far_end", "echo"),
}
MODE_UPPER_HZ = {
    "NB": 4000.0,
    "WB": 8000.0,
    "SWB": 14000.0,
}
FREQUENCY_RESPONSE_CENTERS_HZ = (
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
DNSMOS_SEND_METRICS = {
    "g_mos": ("dnsmos_gmos_ovrl_proxy", "OVRL", "G-MOS"),
    "n_mos": ("dnsmos_nmos_bak_proxy", "BAK", "N-MOS"),
    "s_mos": ("dnsmos_smos_sig_proxy", "SIG", "S-MOS"),
    "listening_quality": (
        "dnsmos_listening_quality_ovrl_proxy",
        "OVRL",
        "TMOS/POLQA",
    ),
}
DNSMOS_REQUIRED_VALUES = ("SIG", "BAK", "OVRL")
SEND_WINDOW_FAMILIES = {
    "idle_noise_level",
    "idle_noise_peaks",
    "cng_level",
    "cng_spectrum",
    "out_of_band",
}
_ZERO_POWER = 1e-30


def _raise_storage_failure(error: Exception) -> None:
    if isinstance(error, OSError) and error.errno == errno.ENOSPC:
        raise error


class AcceptanceModelError(ValueError):
    """The case, output identity, or requested measurement is invalid."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_hashes(paths: Mapping[str, Path | None]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for role, raw_path in paths.items():
        if raw_path is None:
            continue
        path = Path(raw_path)
        if path.is_file():
            hashes[role] = _sha256_file(path)
    return hashes


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("value contains a non-finite number")
        return number
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise ValueError(f"value contains unsupported type {type(value).__name__}")


def _finite_value(value: Any, label: str = "measurement") -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _validate_window(
    start_s: Any,
    window_ms: Any,
) -> tuple[float, float | None]:
    if isinstance(start_s, bool) or not isinstance(start_s, Real):
        raise AcceptanceModelError("analysis_start_s must be numeric")
    start = float(start_s)
    if not math.isfinite(start) or start < 0:
        raise AcceptanceModelError("analysis_start_s must be finite and non-negative")
    if window_ms is None:
        return start, None
    if isinstance(window_ms, bool) or not isinstance(window_ms, Real):
        raise AcceptanceModelError("window_ms must be numeric or null")
    duration_ms = float(window_ms)
    if not math.isfinite(duration_ms) or duration_ms <= 0:
        raise AcceptanceModelError("window_ms must be finite and positive")
    return start, duration_ms


def _load_window(
    path: Path,
    *,
    start_s: float,
    window_ms: float | None,
    label: str,
) -> dict[str, Any]:
    path = Path(path)
    try:
        samples, rate = sf.read(
            path,
            dtype="float64",
            always_2d=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        _raise_storage_failure(error)
        raise AcceptanceModelError(f"{label} could not be decoded: {error}") from error
    if rate <= 0 or samples.ndim != 2 or samples.shape[1] < 1:
        raise AcceptanceModelError(f"{label} has invalid audio dimensions")
    mono = np.asarray(samples[:, 0], dtype=np.float64)
    if not len(mono) or not np.all(np.isfinite(mono)):
        raise AcceptanceModelError(f"{label} is empty or contains non-finite samples")

    start_sample = int(round(start_s * rate))
    if window_ms is None:
        end_sample = len(mono)
    else:
        requested_samples = int(round(window_ms * rate / 1000.0))
        if requested_samples < 1:
            raise AcceptanceModelError(f"{label} selected window is shorter than one sample")
        end_sample = start_sample + requested_samples
    if start_sample >= len(mono) or end_sample > len(mono):
        raise AcceptanceModelError(
            f"{label} selected window [{start_s:.6f}, "
            f"{start_s + (end_sample - start_sample) / rate:.6f}) s "
            f"exceeds {len(mono) / rate:.6f} s"
        )
    selected = np.asarray(mono[start_sample:end_sample], dtype=np.float64)
    if not len(selected):
        raise AcceptanceModelError(f"{label} selected window contains no samples")

    buffer = BytesIO()
    sf.write(buffer, selected, rate, format="WAV", subtype="FLOAT")
    return {
        "path": path,
        "name": path.name,
        "samples": selected,
        "rate": int(rate),
        "bytes": buffer.getvalue(),
        "sha256": _sha256_file(path),
        "window": {
            "analysis_start_s": start_s,
            "window_ms": window_ms,
            "source_sample_rate_hz": int(rate),
            "source_frames": len(mono),
            "selected_start_sample": start_sample,
            "selected_end_sample": end_sample,
            "selected_frames": len(selected),
            "selected_duration_s": len(selected) / rate,
            "cropped": window_ms is not None or start_sample != 0,
            "source_channel": 0,
        },
    }


def _wav_bytes(samples: np.ndarray, rate: int) -> bytes:
    buffer = BytesIO()
    sf.write(
        buffer,
        np.asarray(samples, dtype=np.float64),
        int(rate),
        format="WAV",
        subtype="FLOAT",
    )
    return buffer.getvalue()


def _window_view(
    audio: Mapping[str, Any],
    *,
    start_s: float,
    window_ms: float | None,
    label: str,
) -> dict[str, Any]:
    samples = np.asarray(audio["samples"], dtype=np.float64)
    rate = int(audio["rate"])
    start_sample = int(round(start_s * rate))
    if window_ms is None:
        end_sample = len(samples)
    else:
        end_sample = start_sample + int(round(window_ms * rate / 1000.0))
    if start_sample >= len(samples) or end_sample > len(samples):
        raise AcceptanceModelError(
            f"{label} selected window exceeds {len(samples) / rate:.6f} s"
        )
    selected = samples[start_sample:end_sample]
    if not len(selected):
        raise AcceptanceModelError(f"{label} selected window contains no samples")
    return {
        **audio,
        "samples": selected,
        "bytes": _wav_bytes(selected, rate),
        "window": {
            "analysis_start_s": start_s,
            "window_ms": window_ms,
            "source_sample_rate_hz": rate,
            "source_frames": len(samples),
            "selected_start_sample": start_sample,
            "selected_end_sample": end_sample,
            "selected_frames": len(selected),
            "selected_duration_s": len(selected) / rate,
            "cropped": window_ms is not None or start_sample != 0,
            "source_channel": 0,
        },
    }


def _find_input_path(
    input_paths: Mapping[str, Any],
    canonical_role: str,
) -> tuple[str | None, Path | None]:
    for key in INPUT_ROLE_ALIASES[canonical_role]:
        value = input_paths.get(key)
        if value is not None:
            return key, Path(value)
    return None, None


def _prepare_inputs(
    input_paths: Mapping[str, Any],
    *,
    start_s: float,
    window_ms: float | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, str]]:
    prepared: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    errors: dict[str, str] = {}
    for role in SEND_INPUT_ROLES:
        key, path = _find_input_path(input_paths, role)
        if path is None:
            errors[role] = f"missing input_paths role {role!r}"
            continue
        try:
            item = _load_window(
                path,
                start_s=0.0,
                window_ms=None,
                label=f"input_paths[{key}]",
            )
        except Exception as error:
            _raise_storage_failure(error)
            errors[role] = str(error)
            continue
        prepared[role] = item
        hashes[role] = item["sha256"]
    return prepared, hashes, errors


def _prepare_output(
    outputs: Mapping[str, Any],
    role: str,
    *,
    start_s: float,
    window_ms: float | None,
) -> dict[str, Any]:
    raw_path = outputs.get(role)
    if raw_path is None:
        raise AcceptanceModelError(f"missing outputs[{role!r}]")
    return _load_window(
        Path(raw_path),
        start_s=0.0,
        window_ms=None,
        label=f"outputs[{role!r}]",
    )


def _declared_output_mismatch(
    execution: Mapping[str, Any],
    outputs: Mapping[str, Any],
    output_hashes: Mapping[str, str],
) -> str | None:
    declared = execution.get("output_sha256")
    if not isinstance(declared, Mapping):
        return "execution.output_sha256 must be a complete role-to-hash mapping"
    if set(declared) != set(output_hashes) or set(output_hashes) != set(outputs):
        return "execution.output_sha256 role identities are missing or inconsistent"
    for role, actual in output_hashes.items():
        if declared[role] != actual:
            return f"declared output_sha256 does not match outputs[{role!r}]"
    return None


def _empty_result(
    contract: Mapping[str, Any] | None,
    *,
    item_id: str,
    reason: str,
    binding: Mapping[str, Any],
    status: str = "not_measured",
) -> dict[str, Any]:
    source = {"id": item_id} if contract is None else deepcopy(dict(contract))
    return {
        **source,
        "item_id": item_id,
        "metric": None,
        "value": None,
        "status": status,
        "reason": reason,
        "evidence_grade": "not_measured" if status == "not_measured" else status,
        "standard_compliance": False,
        "evidence_binding": deepcopy(dict(binding)),
        "model_details": {},
    }


def _set_engineering_value(
    result: dict[str, Any],
    *,
    metric: str,
    value: Any,
    reason: str,
    diagnostics: Mapping[str, Any] | None = None,
    window_metadata: Mapping[str, Any] | None = None,
) -> None:
    try:
        normalized = _json_value(value)
    except (TypeError, ValueError) as error:
        _raise_storage_failure(error)
        result.update(
            metric=metric,
            value=None,
            status="invalid",
            reason=f"measurement was not finite or JSON-safe: {error}",
            evidence_grade="invalid",
            standard_compliance=False,
        )
        return
    if isinstance(normalized, (int, float)) and normalized <= -120 and ("dbfs" in metric or "out_of_band" in metric):
        result.update(metric=metric, value=None, status="not_measured", evidence_grade="bound",
                      reason=reason + "；仅能确定不高于 -120 dB 的数值下限，不计入有限数值测量",
                      bound={"relation": "<=", "value": -120.0}, standard_compliance=False)
        return
    result.update(
        metric=metric,
        value=normalized,
        status="review_required",
        reason=reason,
        evidence_grade="engineering_proxy",
        standard_compliance=False,
    )
    if diagnostics is not None:
        result["diagnostics"] = _json_value(diagnostics)
    if window_metadata is not None:
        result["window"] = _json_value(window_metadata)


def _set_not_measured(
    result: dict[str, Any],
    *,
    reason: str,
    status: str = "not_measured",
    diagnostics: Mapping[str, Any] | None = None,
    window_metadata: Mapping[str, Any] | None = None,
) -> None:
    result.update(
        metric=None,
        value=None,
        status=status,
        reason=reason,
        evidence_grade="not_measured" if status == "not_measured" else status,
        standard_compliance=False,
    )
    if diagnostics is not None:
        result["diagnostics"] = _json_value(diagnostics)
    if window_metadata is not None:
        result["window"] = _json_value(window_metadata)


def _merge_diagnostics(
    result: dict[str, Any],
    additions: Mapping[str, Any],
) -> None:
    current = result.get("diagnostics")
    diagnostics = deepcopy(current) if isinstance(current, dict) else {}
    diagnostics.update(deepcopy(dict(additions)))
    result["diagnostics"] = diagnostics


def _merge_model_details(
    result: dict[str, Any],
    additions: Mapping[str, Any],
) -> None:
    current = result.get("model_details")
    details = deepcopy(current) if isinstance(current, dict) else {}
    details.update(_json_value(additions))
    result["model_details"] = details


def _full_context_scope(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "scope": "full_clip",
        "analysis_start_s_applied": False,
        "window_ms_applied": False,
        "note": (
            "analysis_start_s/window_ms 仅用于明确 selected-window 家族；"
            "五路信号与 DNSMOS 使用未截短的完整片段"
        ),
    }


def _model_a_diagnostics(report: Mapping[str, Any]) -> dict[str, Any]:
    analysis = report.get("analysis")
    if not isinstance(analysis, Mapping):
        return {}
    model_a = analysis.get("model_a")
    return {"model_a": deepcopy(model_a)} if model_a is not None else {}


def _send_report(
    *,
    mode: str,
    prepared_inputs: Mapping[str, Mapping[str, Any]],
    tx: Mapping[str, Any],
) -> dict[str, Any]:
    missing = [
        role for role in SEND_INPUT_ROLES if role not in prepared_inputs
    ]
    if missing:
        raise AcceptanceModelError(
            "five-signal evaluation requires roles: " + ", ".join(missing)
        )
    uploads = {
        role: (
            prepared_inputs[role]["name"],
            prepared_inputs[role]["bytes"],
        )
        for role in SEND_INPUT_ROLES
    }
    uploads["near_end"] = (tx["name"], tx["bytes"])
    return evaluate_five_signal(
        mode=mode,
        uploads=uploads,
        channel=0,
        max_delay_ms=2000,
        persist=False,
    )


def _apply_send_frequency_response(
    result: dict[str, Any],
    *,
    reference: Mapping[str, Any] | None,
    noise: Mapping[str, Any] | None,
    tx: Mapping[str, Any] | None,
    mode: str,
) -> None:
    """Measure a stationary excitation with H1, without a speech VAD gate."""
    scope: dict[str, Any] = {
        **_full_context_scope("send_frequency_response_h1"),
        "method": "Welch cross-spectrum H1 magnitude, weighted per third-octave band",
        "engineering_proxy": True,
        "speech_activity_gate": False,
        "minimum_input_power_fraction": .001,
        "minimum_input_snr_db": 10.0,
        "minimum_band_coherence": .6,
        "minimum_coherent_output_power": 1e-12,
        "included_bands_hz": [],
        "missing_bands_hz": [],
        "required_bands_hz": [f for f in FREQUENCY_RESPONSE_CENTERS_HZ
                              if f <= MODE_UPPER_HZ[_bandwidth(mode)]],
        "bands": {},
    }
    details = {"send_frequency_response": scope, "analysis_scope": scope}
    def unavailable(reason: str) -> None:
        scope["reason"] = reason
        _merge_model_details(result, details)
        _set_not_measured(result, reason=reason, diagnostics=details,
                          window_metadata=tx.get("window", {}) if tx else {})

    if reference is None or tx is None:
        unavailable("发送频响需要独立参考激励和 VCP tx 输出")
        return
    rate = int(reference["rate"])
    scope.update(sample_rate_hz=rate, frames=len(reference["samples"]),
                 reference_sha256=reference.get("sha256"), tx_sha256=tx.get("sha256"))
    if int(tx["rate"]) != rate:
        unavailable("发送频响输入/输出采样率不一致，不能直接估计传函")
        return
    if scope["required_bands_hz"][-1] * 2 ** (1 / 6) >= rate / 2:
        unavailable("发送频响最高必需频带不可完整观测，需要更高采样率")
        return
    x, y = np.asarray(reference["samples"]), np.asarray(tx["samples"])
    if len(x) != len(y) or len(x) < rate * 2.5:
        unavailable("发送频响需要至少 2.5 秒且长度一致的输入/输出，保证至少四个 Welch 分段")
        return
    if noise is None or int(noise["rate"]) != rate or len(noise["samples"]) != len(x):
        unavailable("发送频响需要与参考激励同采样率、同长度的噪声参考，不能跳过逐频带 SNR 校验")
        return
    options = {"fs": rate, "nperseg": rate, "noverlap": rate // 2,
               "detrend": False, "scaling": "density"}
    frequencies, pxx = signal.welch(x, **options)
    _, pyy = signal.welch(y, **options)
    _, pxy = signal.csd(x, y, **options)
    _, pnn = signal.welch(noise["samples"], **options)
    total = float(np.sum(pxx))
    gains = []
    for center in scope["required_bands_hz"]:
        mask = ((frequencies >= center / 2 ** (1 / 6)) &
                (frequencies <= center * 2 ** (1 / 6)))
        power = float(np.sum(pxx[mask]))
        band: dict[str, Any] = {"input_power": power,
                               "input_power_fraction": power / max(total, _ZERO_POWER)}
        scope["bands"][str(center)] = band
        if total <= 1e-12 or power < max(1e-12, total * .001):
            scope["missing_bands_hz"].append(center)
            unavailable(f"发送频响缺少有效参考激励频带 {center} Hz，不能用噪声补齐")
            return
        snr = 10 * math.log10(power / max(float(np.sum(pnn[mask])), _ZERO_POWER))
        band["input_snr_db"] = snr
        if snr < 10:
            unavailable(f"发送频响 {center} Hz 参考激励相对噪声不足 10 dB")
            return
        coherent_power = float(np.sum(np.abs(pxy[mask]) ** 2 / np.maximum(pxx[mask], _ZERO_POWER)))
        output_power = float(np.sum(pyy[mask]))
        coherence = min(1.0, coherent_power / max(output_power, _ZERO_POWER))
        band["coherent_output_power"] = coherent_power
        band["coherence"] = coherence
        if coherent_power <= 1e-12 or coherence < .6:
            unavailable(f"发送频响 {center} Hz 输出低于可测下限或与激励缺乏相干性，不能当作精确响应")
            return
        gain = 10 * math.log10(coherent_power / power)
        band["transfer_gain_db"] = gain
        scope["included_bands_hz"].append(center)
        gains.append(gain)
    _merge_model_details(result, details)
    _set_engineering_value(
        result, metric="send_h1_frequency_response_span_db",
        value=max(gains) - min(gains),
        reason="已知激励到 VCP 发送输出的 H1 频响跨度工程代理；不等于标准频响容差曲线，不代表正式合规",
        diagnostics=details, window_metadata=tx.get("window", {}),
    )


def _resample_for_dnsmos(
    samples: np.ndarray,
    rate: int,
    *,
    label: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(samples, dtype=np.float64)
    if not len(values) or not np.all(np.isfinite(values)):
        raise AcceptanceModelError(f"{label} is empty or non-finite")
    metadata: dict[str, Any] = {
        "source_sample_rate_hz": int(rate),
        "target_sample_rate_hz": DNSMOS_NATIVE_RATE,
        "source_samples": len(values),
        "source_duration_s": len(values) / rate,
        "resampled": int(rate) != DNSMOS_NATIVE_RATE,
        "resampling": "none",
        "repeated": False,
    }
    if int(rate) == DNSMOS_NATIVE_RATE:
        scored = values
    else:
        common = math.gcd(DNSMOS_NATIVE_RATE, int(rate))
        scored = signal.resample_poly(
            values,
            DNSMOS_NATIVE_RATE // common,
            int(rate) // common,
        )
        metadata["resampling"] = "scipy.signal.resample_poly"
    metadata.update(
        scored_samples=len(scored),
        scored_duration_s=len(scored) / DNSMOS_NATIVE_RATE,
    )
    return np.asarray(scored, dtype=np.float64), metadata


@lru_cache(maxsize=1)
def _dnsmos_evaluator():
    evaluators = create_evaluators(
        ["dnsmos"],
        QUALITY_PROJECT / "build/quality-evaluations/cache",
    )
    if not evaluators:
        raise AcceptanceModelError("DNSMOS evaluator factory returned no evaluator")
    return evaluators[0]


def _validated_dnsmos_values(row: Any) -> dict[str, float]:
    if not isinstance(row, Mapping):
        raise AcceptanceModelError(
            f"DNSMOS returned {type(row).__name__}, expected a mapping"
        )
    if "available" in row and not bool(row["available"]):
        raise AcceptanceModelError(
            "DNSMOS model unavailable: " + str(row.get("reason", "unspecified"))
        )
    values = row.get("values", row)
    if not isinstance(values, Mapping):
        raise AcceptanceModelError("DNSMOS values must be a mapping")
    result: dict[str, float] = {}
    for metric in DNSMOS_REQUIRED_VALUES:
        if metric not in values:
            raise AcceptanceModelError(f"DNSMOS output is missing {metric}")
        score = _finite_value(values[metric], f"DNSMOS {metric}")
        if not 0.0 <= score <= 5.0:
            raise AcceptanceModelError(
                f"DNSMOS {metric} score {score} is outside [0, 5]"
            )
        result[metric] = score
    return result


def _score_dnsmos(
    samples: np.ndarray,
    rate: int,
    *,
    label: str,
) -> dict[str, Any]:
    source = np.asarray(samples, dtype=np.float64)
    if len(source) / rate < DNSMOS_MINIMUM_SECONDS:
        return {
            "available": False,
            "reason": (
                f"{label} is too short for DNSMOS P.835: "
                f"{len(source) / rate:.3f} s < {DNSMOS_MINIMUM_SECONDS:.2f} s; "
                "audio was not repeated"
            ),
        }
    try:
        scored, resampling = _resample_for_dnsmos(source, rate, label=label)
    except Exception as error:
        _raise_storage_failure(error)
        return {"available": False, "reason": f"{label} resampling failed: {error}"}
    if len(scored) / DNSMOS_NATIVE_RATE < DNSMOS_MINIMUM_SECONDS:
        return {
            "available": False,
            "reason": (
                f"{label} is too short for DNSMOS after explicit 16 kHz "
                "resampling; audio was not repeated"
            ),
        }
    try:
        evaluator = _dnsmos_evaluator()
        evaluator_rate = int(getattr(evaluator, "rate", DNSMOS_NATIVE_RATE))
        if evaluator_rate != DNSMOS_NATIVE_RATE:
            raise AcceptanceModelError(
                f"DNSMOS evaluator rate is {evaluator_rate}, expected "
                f"{DNSMOS_NATIVE_RATE}"
            )
        row = evaluator.score(scored, DNSMOS_NATIVE_RATE)
        values = _validated_dnsmos_values(row)
    except Exception as error:
        _raise_storage_failure(error)
        return {
            "available": False,
            "reason": f"DNSMOS model unavailable: {error}",
            "resampling": resampling,
        }
    return {
        "available": True,
        "values": values,
        "resampling": resampling,
        "model": getattr(evaluator, "name", "dnsmos"),
        "identity": deepcopy(getattr(evaluator, "identity", None)),
        "cache_hit": bool(row.get("cache_hit", False))
        if isinstance(row, Mapping)
        else False,
        "model_row": deepcopy(row) if isinstance(row, dict) else None,
    }


def _bandwidth(mode: str) -> str:
    return str(mode).split("-", 1)[0]


def _receive_report(
    *,
    mode: str,
    rx: Mapping[str, Any],
    start_s: float,
    window_ms: float | None,
    reference: Mapping[str, Any] | None = None,
    receive_config: Mapping[str, Any] | None = None,
    receive_ir_paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    upper_hz = MODE_UPPER_HZ.get(_bandwidth(mode))
    samples = np.asarray(rx["samples"], dtype=np.float64)
    source_rate = int(rx["rate"])
    target_rate = source_rate
    if source_rate not in {16_000, 24_000, 48_000}:
        target_rate = 16_000
        common = math.gcd(source_rate, target_rate)
        samples = signal.resample_poly(
            samples,
            target_rate // common,
            source_rate // common,
        )
    input_processing = {
        "source_sample_rate_hz": source_rate,
        "simulation_sample_rate_hz": target_rate,
        "resampled": source_rate != target_rate,
        "resampling": (
            "none"
            if source_rate == target_rate
            else "scipy.signal.resample_poly"
        ),
        "source_frames": len(rx["samples"]),
        "simulation_frames": len(samples),
        "reason": (
            "receive_simulation accepts 16/24/48 kHz; engineering input was "
            "explicitly resampled without changing the source output identity"
            if source_rate != target_rate
            else "source rate accepted by receive_simulation"
        ),
    }
    noise_end_s = (
        start_s + float(window_ms) / 1000.0
        if window_ms is not None
        else None
    )
    config = normalize_receive_configuration(receive_config)
    acoustic_arguments: dict[str, Any] = {}
    if config["model"] == "head":
        if receive_ir_paths:
            raise AcceptanceModelError("head simulation must not receive IR assets")
        acoustic_arguments = {key: config[key] for key in (
            "azimuth_deg", "head_radius_m", "speed_of_sound_mps"
        )}
    else:
        if not isinstance(receive_ir_paths, Mapping) or set(receive_ir_paths) != {"left_ir", "right_ir"}:
            raise AcceptanceModelError("paired_ir requires resolved left/right dataset assets")
        for role in ("left_ir", "right_ir"):
            path = Path(receive_ir_paths[role])
            if not path.is_file() or path.is_symlink():
                raise AcceptanceModelError(f"{role} must be a regular dataset asset")
            acoustic_arguments[f"{role}_name"] = config[role]
            acoustic_arguments[f"{role}_bytes"] = path.read_bytes()
    report = evaluate_synthetic_receive(
        mode,
        rx["name"],
        _wav_bytes(samples, target_rate),
        upper_hz=upper_hz,
        noise_start_s=start_s if noise_end_s is not None else None,
        noise_end_s=noise_end_s,
        **acoustic_arguments,
    )
    report["acceptance_input_processing"] = input_processing
    # These additions are bound by assessment.json, not the immutable synthetic
    # report's digest. Never rewrite that report or reuse its digest for them.
    report["acceptance_acoustic_configuration"] = config
    report["acceptance_transfer_response"] = {
        "available": False, "span_db": None,
        "reason": "缺少已知远端激励，不能将输出频谱当作传递函数",
        "standard_compliance": False, "calibrated": False,
    }
    if reference is not None:
        try:
            reference_rate = int(reference["rate"])
            reference_samples = np.asarray(reference["samples"], dtype=np.float64)
            if reference_rate != target_rate:
                divisor = math.gcd(reference_rate, target_rate)
                reference_samples = signal.resample_poly(
                    reference_samples, target_rate // divisor, reference_rate // divisor,
                )
            ears = []
            for role, filename in (("left", "virtual_left.wav"), ("right", "virtual_right.wav")):
                path = get_synthetic_receive_file(report["evaluation_id"], filename)
                if path is None or _sha256_file(path) != report["artifacts"][role]["sha256"]:
                    raise AcceptanceModelError("虚拟耳产物缺失或身份已改变")
                ear, ear_rate = sf.read(path, dtype="float64", always_2d=False)
                if int(ear_rate) != target_rate:
                    raise AcceptanceModelError("虚拟耳产物采样率不一致")
                ears.append(ear)
            transfer = estimate_binaural_transfer(
                reference_samples, ears[0], ears[1], target_rate,
                centers_hz=FREQUENCY_RESPONSE_CENTERS_HZ,
                upper_hz=upper_hz,
            )
            transfer["reference_sample_rate_hz"] = reference_rate
            transfer["reference_resampled"] = reference_rate != target_rate
            transfer["reference_sha256"] = reference.get("sha256")
            transfer["signal_path"] = "far_end -> VCP RX -> virtual binaural outputs"
            transfer["evidence_digest_scope"] = "acceptance assessment, not synthetic result_sha256"
            required = transfer.get("required_bands_hz", [])
            if required and max(required) * 2 ** (1 / 6) >= reference_rate / 2:
                transfer.update(available=False, span_db=None,
                                reason="原始参考采样率无法完整覆盖必需频带，重采样不能补出证据")
            if report["simulation"]["output_normalisation"]["normalised"]:
                transfer.update(available=False, span_db=None,
                                reason="虚拟耳输出经过峰值归一化，不能据此评估原始传递增益")
            report["acceptance_transfer_response"] = transfer
        except (ValueError, TypeError, KeyError, OSError) as error:
            _raise_storage_failure(error)
            report["acceptance_transfer_response"]["reason"] = str(error)
    return report


def _receive_diagnostics(report: Mapping[str, Any]) -> dict[str, Any]:
    metrics = report.get("metrics")
    return {
        "synthetic_receive": {
            "evaluation_id": report.get("evaluation_id"),
            "evidence_kind": report.get("evidence_kind"),
            "synthetic_proxy": report.get("synthetic_proxy"),
            "formal_hats_evidence": report.get("formal_hats_evidence"),
            "metrics": deepcopy(metrics) if isinstance(metrics, Mapping) else None,
            "simulation": deepcopy(report.get("simulation")),
            "input_processing": deepcopy(report.get("acceptance_input_processing")),
            "acoustic_configuration": deepcopy(report.get("acceptance_acoustic_configuration")),
            "transfer_response": deepcopy(report.get("acceptance_transfer_response")),
            "limitations": deepcopy(report.get("limitations")),
        }
    }


def _receive_frequency_response_bands(
    report: Mapping[str, Any],
    *,
    mode: str,
) -> tuple[dict[str, float] | None, dict[str, Any]]:
    report_mode = report.get("mode")
    if not isinstance(report_mode, str) or not report_mode:
        return None, {
            "scope": "mode_limited_third_octave_span",
            "reason": "接收报告缺少 mode，无法限定频响带宽",
        }
    if report_mode != mode:
        return None, {
            "scope": "mode_limited_third_octave_span",
            "report_mode": report_mode,
            "requested_mode": mode,
            "reason": "接收报告 mode 与合同 mode 不一致",
        }
    upper_hz = MODE_UPPER_HZ.get(_bandwidth(report_mode))
    if upper_hz is None:
        return None, {
            "scope": "mode_limited_third_octave_span",
            "report_mode": report_mode,
            "reason": f"不支持的接收频响带宽: {report_mode}",
        }
    simulation = report.get("simulation")
    raw_rate = (
        simulation.get("analysis_sample_rate_hz")
        if isinstance(simulation, Mapping)
        else None
    )
    try:
        rate = _finite_value(raw_rate, "receive sample rate")
    except ValueError:
        return None, {
            "scope": "mode_limited_third_octave_span",
            "report_mode": report_mode,
            "upper_hz": upper_hz,
            "reason": "接收报告缺少有效采样率，无法确认频响频带可观测性",
        }
    metrics = report.get("metrics")
    levels = (
        metrics.get("third_octave_band_levels_dbfs")
        if isinstance(metrics, Mapping)
        else None
    )
    binaural = levels.get("binaural") if isinstance(levels, Mapping) else None
    if not isinstance(binaural, Mapping):
        return None, {
            "scope": "mode_limited_third_octave_span",
            "report_mode": report_mode,
            "upper_hz": upper_hz,
            "nyquist_hz": rate / 2.0,
            "reason": "接收报告缺少双耳第三倍频程频带",
        }
    available: dict[float, float] = {}
    try:
        for raw_frequency, raw_level in binaural.items():
            if isinstance(raw_frequency, bool):
                raise ValueError("third-octave frequency must be a real number")
            try:
                frequency = float(raw_frequency)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "third-octave frequency must be a real number"
                ) from error
            if not math.isfinite(frequency):
                raise ValueError("third-octave frequency must be finite")
            if frequency <= 0.0 or frequency in available:
                raise ValueError(f"invalid third-octave frequency {raw_frequency!r}")
            available[frequency] = _finite_value(
                raw_level,
                f"third-octave band {frequency:g}",
            )
    except ValueError as error:
        return None, {
            "scope": "mode_limited_third_octave_span",
            "report_mode": report_mode,
            "upper_hz": upper_hz,
            "nyquist_hz": rate / 2.0,
            "reason": str(error),
        }
    required = [
        float(frequency)
        for frequency in FREQUENCY_RESPONSE_CENTERS_HZ
        if float(frequency) <= upper_hz
    ]
    missing = [
        frequency for frequency in required if frequency not in available
    ]
    included = [
        frequency for frequency in required if frequency in available
    ]
    scope = {
        "scope": "mode_limited_third_octave_span",
        "report_mode": report_mode,
        "upper_hz": upper_hz,
        "nyquist_hz": rate / 2.0,
        "required_bands_hz": required,
        "included_bands_hz": included,
        "missing_bands_hz": missing,
        "engineering_proxy": True,
    }
    if missing:
        scope["reason"] = (
            "接收频响缺少必需可观测第三倍频程: "
            + ", ".join(f"{frequency:g}" for frequency in missing)
            + " Hz"
        )
        return None, scope
    return {
        str(int(frequency)): available[frequency]
        for frequency in required
    }, scope


def _receive_family_value(
    family: str,
    report: Mapping[str, Any],
    *,
    mode: str,
) -> tuple[str, Any, str] | None:
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    if family == "rlr":
        return (
            "binaural_rms_dbfs",
            metrics.get("binaural_rms_dbfs"),
            "software binaural digital RMS 是工程代理，不是校准后的 RLR",
        )
    if family == "frequency_response":
        transfer = report.get("acceptance_transfer_response")
        if not isinstance(transfer, Mapping) or transfer.get("available") is not True:
            reason = transfer.get("reason") if isinstance(transfer, Mapping) else None
            if reason is None:
                _, legacy_scope = _receive_frequency_response_bands(report, mode=mode)
                reason = legacy_scope.get("reason")
            return ("receive_h1_frequency_response_span_db", None,
                    str(reason or "缺少有效输入/输出传递函数证据，输出频谱跨度仅保留为诊断"))
        if report.get("mode") != mode:
            return ("receive_h1_frequency_response_span_db", None,
                    "接收报告 mode 与合同 mode 不一致")
        return (
            "receive_h1_frequency_response_span_db",
            transfer.get("span_db"),
            "已知远端激励到 VCP RX 与虚拟双耳的 H1 频响跨度；未校准，不等于合同频响掩模",
        )
    if family in {"idle_noise_level", "idle_noise_peaks"}:
        noise = metrics.get("noise")
        if not isinstance(noise, Mapping):
            return None
        if family == "idle_noise_level":
            return (
                "selected_window_binaural_rms_dbfs",
                noise.get("binaural_digital_rms_dbfs"),
                "选定时间窗的软件接收数字 RMS 是工程代理，不是 dBPa/A 计权空闲噪声",
            )
        noise_level = noise.get("binaural_digital_rms_dbfs")
        if isinstance(noise_level, Real) and float(noise_level) <= -120:
            return (
                "selected_window_peak_prominence_db", None,
                "选定窗静音或低于数字可测下限，不能将显示下限之差记作零噪声峰",
            )
        return (
            "selected_window_peak_prominence_db",
            noise.get("peak_prominence_db"),
            "选定时间窗数字峰值突出度是工程诊断，不是合同噪声曲线",
        )
    if family == "out_of_band":
        return (
            "out_of_band_ratio_db",
            metrics.get("out_of_band_ratio_db"),
            "软件接收带外/总能量比是工程诊断，不是合同 Corrected value",
        )
    return None


def _resample_psd_pair(
    first: np.ndarray,
    first_rate: int,
    second: np.ndarray,
    second_rate: int,
) -> tuple[np.ndarray, np.ndarray, int, dict[str, Any]]:
    target_rate = min(int(first_rate), int(second_rate))
    common = math.gcd(int(first_rate), target_rate)
    first_values = (
        np.asarray(first, dtype=np.float64)
        if int(first_rate) == target_rate
        else signal.resample_poly(
            np.asarray(first, dtype=np.float64),
            target_rate // common,
            int(first_rate) // common,
        )
    )
    common = math.gcd(int(second_rate), target_rate)
    second_values = (
        np.asarray(second, dtype=np.float64)
        if int(second_rate) == target_rate
        else signal.resample_poly(
            np.asarray(second, dtype=np.float64),
            target_rate // common,
            int(second_rate) // common,
        )
    )
    length = min(len(first_values), len(second_values))
    if length < 8:
        raise AcceptanceModelError("PSD comparison requires at least 8 common samples")
    return (
        np.asarray(first_values[:length], dtype=np.float64),
        np.asarray(second_values[:length], dtype=np.float64),
        target_rate,
        {
            "input_rate_hz": int(first_rate),
            "output_rate_hz": int(second_rate),
            "comparison_rate_hz": target_rate,
            "input_resampled": int(first_rate) != target_rate,
            "output_resampled": int(second_rate) != target_rate,
            "comparison_frames": length,
            "resampling": (
                "none"
                if int(first_rate) == int(second_rate) == target_rate
                else "scipy.signal.resample_poly"
            ),
        },
    )


def _normalized_psd_shape_distance(
    input_noise: np.ndarray,
    input_rate: int,
    output_noise: np.ndarray,
    output_rate: int,
    *,
    upper_hz: float,
) -> tuple[float, dict[str, Any]]:
    """Return Jensen-Shannon distance between gain-normalized PSD shapes."""
    first, second, rate, resampling = _resample_psd_pair(
        input_noise,
        input_rate,
        output_noise,
        output_rate,
    )
    nperseg = min(4096, len(first))
    if nperseg < 8:
        raise AcceptanceModelError("PSD comparison window is too short")
    frequencies, first_psd = signal.welch(
        first,
        fs=rate,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
    )
    _, second_psd = signal.welch(
        second,
        fs=rate,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
    )
    nyquist = rate / 2.0
    mask = (frequencies >= 20.0) & (
        frequencies <= min(float(upper_hz), nyquist) + 1e-9
    )
    if np.count_nonzero(mask) < 8:
        raise AcceptanceModelError("PSD comparison band has fewer than 8 bins")
    p = np.asarray(first_psd[mask], dtype=np.float64)
    q = np.asarray(second_psd[mask], dtype=np.float64)
    p = np.maximum(p, 0.0)
    q = np.maximum(q, 0.0)
    p_total = float(np.sum(p))
    q_total = float(np.sum(q))
    if p_total <= _ZERO_POWER or q_total <= _ZERO_POWER:
        raise AcceptanceModelError("input or output noise PSD has no usable power")
    p /= p_total
    q /= q_total
    midpoint = 0.5 * (p + q)
    positive_p = p > 0.0
    positive_q = q > 0.0
    kl_p = float(np.sum(p[positive_p] * np.log(p[positive_p] / midpoint[positive_p])))
    kl_q = float(np.sum(q[positive_q] * np.log(q[positive_q] / midpoint[positive_q])))
    divergence_bits = max(0.0, 0.5 * (kl_p + kl_q) / math.log(2.0))
    distance_bits = math.sqrt(divergence_bits)
    return distance_bits, {
        "method": "Jensen-Shannon distance between gain-normalized Welch PSD shapes",
        "normalized_psd": True,
        "uses_total_power_difference": False,
        "js_divergence_bits": divergence_bits,
        "shape_distance_bits": distance_bits,
        "band_hz": [20.0, min(float(upper_hz), nyquist)],
        "frequency_bins": int(np.count_nonzero(mask)),
        "welch_nperseg": nperseg,
        **resampling,
    }


def _rms_dbfs(samples: np.ndarray) -> float:
    values = np.asarray(samples, dtype=np.float64)
    if not len(values):
        raise AcceptanceModelError("RMS requires at least one sample")
    power = float(np.mean(np.square(values)))
    if not math.isfinite(power) or power <= _ZERO_POWER:
        return -120.0
    return max(-120.0, float(10.0 * math.log10(power)))


def _out_of_band_ratio_db(
    samples: np.ndarray,
    rate: int,
    upper_hz: float,
) -> float:
    values = np.asarray(samples, dtype=np.float64)
    if upper_hz >= rate / 2:
        raise AcceptanceModelError("带外频带不可观测：采样率 Nyquist 必须高于带宽上限")
    if not len(values):
        raise AcceptanceModelError("out-of-band ratio requires audio samples")
    window = signal.windows.hann(len(values), sym=False)
    spectrum = np.fft.rfft(values * window)
    power = np.abs(spectrum) ** 2
    frequency = np.fft.rfftfreq(len(values), d=1.0 / int(rate))
    total = float(np.sum(power))
    if total <= _ZERO_POWER:
        raise AcceptanceModelError("out-of-band ratio input has no usable power")
    outside = float(np.sum(power[frequency > upper_hz]))
    if outside <= _ZERO_POWER:
        return -120.0
    return max(-120.0, 10.0 * math.log10(outside / total))


def _selected_window_metric(
    family: str,
    *,
    noise: Mapping[str, Any] | None,
    tx: Mapping[str, Any],
    mode: str,
    start_s: float,
    window_ms: float | None,
) -> tuple[str, float, str, dict[str, Any], dict[str, Any]]:
    tx_view = _window_view(
        tx,
        start_s=start_s,
        window_ms=window_ms,
        label="VCP tx noise",
    )
    tx_rms = _rms_dbfs(tx_view["samples"])
    if family == "idle_noise_level":
        return (
            "selected_window_output_rms_dbfs",
            tx_rms,
            "选定时间窗的 VCP 输出数字 RMS 是工程代理，不是 dBPa/A 计权空闲噪声",
            {"tx_rms_dbfs": tx_rms},
            tx_view["window"],
        )
    if family == "idle_noise_peaks":
        peak = float(np.max(np.abs(tx_view["samples"])))
        peak_dbfs = (
            max(-120.0, 20.0 * math.log10(peak))
            if peak > 0.0
            else -120.0
        )
        prominence = max(0.0, peak_dbfs - tx_rms)
        return (
            "selected_window_peak_prominence_db",
            prominence,
            "选定时间窗的数字峰值突出度是工程诊断，不是合同噪声曲线",
            {"tx_rms_dbfs": tx_rms, "tx_peak_dbfs": peak_dbfs},
            tx_view["window"],
        )
    if family == "cng_level":
        if noise is None:
            raise AcceptanceModelError("CNG level requires input noise")
        noise_view = _window_view(
            noise,
            start_s=start_s,
            window_ms=window_ms,
            label="input noise",
        )
        noise_rms = _rms_dbfs(noise_view["samples"])
        change = tx_rms - noise_rms
        return (
            "selected_window_noise_level_change_db",
            change,
            "选定时间窗输入/输出噪声 RMS 差是工程代理，未识别 CNG 状态或校准量纲",
            {"input_noise_rms_dbfs": noise_rms, "output_noise_rms_dbfs": tx_rms},
            tx_view["window"],
        )
    if family == "out_of_band":
        upper_hz = MODE_UPPER_HZ.get(_bandwidth(mode))
        if upper_hz is None:
            raise AcceptanceModelError(f"unsupported mode bandwidth: {mode}")
        ratio = _out_of_band_ratio_db(tx_view["samples"], tx_view["rate"], upper_hz)
        return (
            "selected_window_out_of_band_ratio_db",
            ratio,
            "选定时间窗输出带外/总能量比是工程诊断，不是合同 Corrected value",
            {"upper_hz": upper_hz},
            tx_view["window"],
        )
    raise AcceptanceModelError(f"no selected-window mapping for {family}")


def _apply_send_evaluation(
    result: dict[str, Any],
    evaluation_item: Mapping[str, Any] | None,
    *,
    model_a: Mapping[str, Any],
    tx_window: Mapping[str, Any],
) -> None:
    if model_a:
        _merge_model_details(result, model_a)
    _merge_model_details(
        result,
        {"analysis_scope": _full_context_scope("five_signal")},
    )
    if evaluation_item is None:
        _set_not_measured(
            result,
            reason="five-signal result did not contain this contract row",
            diagnostics=model_a,
            window_metadata=tx_window,
        )
        return
    value = evaluation_item.get("value")
    if value is None:
        reason = str(evaluation_item.get("reason") or "五路信号没有产生该测项")
        _set_not_measured(
            result,
            reason=reason,
            diagnostics=model_a,
            window_metadata=tx_window,
        )
        return
    try:
        numeric = _finite_value(value, "five-signal value")
    except ValueError as error:
        _raise_storage_failure(error)
        _set_not_measured(
            result,
            reason=str(error),
            status="invalid",
            diagnostics=model_a,
            window_metadata=tx_window,
        )
        return
    _set_engineering_value(
        result,
        metric=str(evaluation_item.get("metric") or result["family"]),
        value=numeric,
        reason=(
            f"{evaluation_item.get('reason') or '五路信号工程计算'}；"
            "数值统一标记 review_required，不套用合同门限，不代表标准符合性"
        ),
        diagnostics=model_a,
        window_metadata=tx_window,
    )


def _apply_send_dnsmos(
    result: dict[str, Any],
    dnsmos: Mapping[str, Any],
    *,
    model_a: Mapping[str, Any],
    tx_window: Mapping[str, Any],
) -> None:
    family = result["family"]
    metric, score_key, target = DNSMOS_SEND_METRICS[family]
    if model_a:
        _merge_model_details(result, model_a)
    _merge_model_details(
        result,
        {"analysis_scope": _full_context_scope("dnsmos_send")},
    )
    dnsmos_details = {
        "model": dnsmos.get("model"),
        "identity": dnsmos.get("identity"),
        "values": dnsmos.get("values"),
        "resampling": dnsmos.get("resampling"),
        "cache_hit": dnsmos.get("cache_hit"),
        "model_row": dnsmos.get("model_row"),
        "available": dnsmos.get("available"),
        "reason": dnsmos.get("reason"),
    }
    _merge_model_details(result, {"dnsmos": dnsmos_details})
    if not dnsmos.get("available"):
        _set_not_measured(
            result,
            reason=str(dnsmos.get("reason") or "DNSMOS unavailable"),
            diagnostics={**model_a, "dnsmos": deepcopy(dict(dnsmos))},
            window_metadata=tx_window,
        )
        return
    values = dnsmos.get("values")
    if not isinstance(values, Mapping) or score_key not in values:
        _set_not_measured(
            result,
            reason=f"DNSMOS output did not contain {score_key}",
            diagnostics={**model_a, "dnsmos": deepcopy(dict(dnsmos))},
            window_metadata=tx_window,
        )
        return
    _set_engineering_value(
        result,
        metric=metric,
        value=values[score_key],
        reason=(
            f"DNSMOS P.835 {score_key} 是工程模型代理，不是合同 {target}；"
            "不重复短片，不套用合同门限"
        ),
        diagnostics={**model_a, "dnsmos": deepcopy(dict(dnsmos))},
        window_metadata=tx_window,
    )


def _apply_cng_spectrum(
    result: dict[str, Any],
    *,
    noise: Mapping[str, Any] | None,
    tx: Mapping[str, Any] | None,
    mode: str,
    model_a: Mapping[str, Any],
) -> None:
    if noise is None or tx is None:
        _set_not_measured(
            result,
            reason="CNG spectrum requires both the input noise segment and outputs['tx']",
            diagnostics=model_a,
        )
        return
    upper_hz = MODE_UPPER_HZ.get(_bandwidth(mode))
    if upper_hz is None:
        _set_not_measured(
            result,
            reason=f"unsupported bandwidth for CNG spectrum: {mode}",
            diagnostics=model_a,
        )
        return
    try:
        distance, diagnostics = _normalized_psd_shape_distance(
            noise["samples"],
            noise["rate"],
            tx["samples"],
            tx["rate"],
            upper_hz=upper_hz,
        )
    except Exception as error:
        _raise_storage_failure(error)
        _set_not_measured(
            result,
            reason=f"CNG PSD shape calculation failed: {error}",
            status="invalid",
            diagnostics=model_a,
            window_metadata=tx.get("window"),
        )
        return
    _set_engineering_value(
        result,
        metric="normalized_psd_shape_distance_bits",
        value=distance,
        reason=(
            "工程 CNG 频谱代理：比较输入/输出噪声段归一化 PSD 形状的 "
            "Jensen-Shannon 距离；未使用总功率差，不是合同 CNG 频谱掩模，"
            "不套用 0–120 门限"
        ),
        diagnostics={
            **model_a,
            "cng_spectrum": diagnostics,
            "input_noise_window": noise.get("window"),
            "output_noise_window": tx.get("window"),
        },
        window_metadata=tx.get("window"),
    )


def _binaural_listening_quality(
    report: Mapping[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    evaluation_id = report.get("evaluation_id")
    if not isinstance(evaluation_id, str):
        return {
            "available": False,
            "reason": "synthetic receive report has no evaluation_id",
        }
    left_path = get_synthetic_receive_file(evaluation_id, "virtual_left.wav")
    right_path = get_synthetic_receive_file(evaluation_id, "virtual_right.wav")
    if left_path is None or right_path is None:
        return {
            "available": False,
            "reason": "synthetic receive report did not expose both virtual ear files",
        }
    try:
        left, left_rate = sf.read(left_path, dtype="float64")
        right, right_rate = sf.read(right_path, dtype="float64")
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        _raise_storage_failure(error)
        return {
            "available": False,
            "reason": f"virtual ear audio could not be read: {error}",
        }
    if int(left_rate) != int(right_rate):
        return {
            "available": False,
            "reason": "virtual ear files have different sample rates",
        }
    if len(left) / left_rate < DNSMOS_MINIMUM_SECONDS:
        return {
            "available": False,
            "reason": (
                f"virtual ears are too short for DNSMOS P.835: "
                f"{len(left) / left_rate:.3f} s < "
                f"{DNSMOS_MINIMUM_SECONDS:.2f} s; audio was not repeated"
            ),
        }
    try:
        left_16k, left_resampling = _resample_for_dnsmos(
            left,
            int(left_rate),
            label="virtual left ear",
        )
        right_16k, right_resampling = _resample_for_dnsmos(
            right,
            int(right_rate),
            label="virtual right ear",
        )
        evaluator = _dnsmos_evaluator()
        proxy = evaluate_binaural_dnsmos_with_evaluator(
            left_16k,
            right_16k,
            evaluator,
            {
                "evaluation_id": evaluation_id,
                "left_resampling": left_resampling,
                "right_resampling": right_resampling,
                "evidence_binding": "real VCP rx -> synthetic receive -> DNSMOS",
            },
            sample_rate=DNSMOS_NATIVE_RATE,
            mode=mode,
        )
    except Exception as error:
        _raise_storage_failure(error)
        return {
            "available": False,
            "reason": f"binaural DNSMOS P.835 proxy unavailable: {error}",
        }
    return {
        "available": True,
        "proxy": proxy,
        "identity": deepcopy(getattr(evaluator, "identity", None)),
        "left_resampling": left_resampling,
        "right_resampling": right_resampling,
    }


def _apply_receive_listening(
    result: dict[str, Any],
    *,
    receive_report: Mapping[str, Any],
    rx_window: Mapping[str, Any],
    mode: str,
) -> None:
    proxy = _binaural_listening_quality(receive_report, mode=mode)
    diagnostics = _receive_diagnostics(receive_report)
    diagnostics["binaural_dnsmos"] = deepcopy(proxy)
    _merge_model_details(
        result,
        {"analysis_scope": _full_context_scope("dnsmos_receive")},
    )
    _merge_model_details(
        result,
        {
            "dnsmos": {
                "identity": proxy.get("identity"),
                "proxy": proxy.get("proxy"),
                "left_resampling": proxy.get("left_resampling"),
                "right_resampling": proxy.get("right_resampling"),
                "available": proxy.get("available"),
                "reason": proxy.get("reason"),
            }
        },
    )
    if not proxy.get("available"):
        _set_not_measured(
            result,
            reason=str(proxy.get("reason") or "binaural DNSMOS unavailable"),
            diagnostics=diagnostics,
            window_metadata=rx_window,
        )
        return
    value = proxy["proxy"]["OVRL_proxy"]
    _set_engineering_value(
        result,
        metric="dnsmos_p835_binaural_ovrl_proxy",
        value=value,
        reason=(
            "双耳 DNSMOS P.835 最差耳 OVRL 是工程代理，不是 TMOS 或 POLQA；"
            "不重复短片，不套用合同 MOS 门限"
        ),
        diagnostics=diagnostics,
        window_metadata=rx_window,
    )


def _apply_receive_family(
    result: dict[str, Any],
    *,
    receive_report: Mapping[str, Any],
    rx_window: Mapping[str, Any],
    mode: str,
) -> None:
    family = result["family"]
    simulation = receive_report.get("simulation")
    normalisation = simulation.get("output_normalisation") if isinstance(simulation, Mapping) else None
    if family == "rlr" and isinstance(normalisation, Mapping) and normalisation.get("normalised"):
        _set_not_measured(
            result, reason="虚拟耳输出经过峰值归一化，原始接收电平不可直接评估",
            diagnostics=_receive_diagnostics(receive_report), window_metadata=rx_window,
        )
        return
    if family == "rlr_linearity":
        _set_not_measured(
            result,
            reason=(
                "RLR linearity requires a matched nominal/-5 dB/+5 dB group; "
                "single-case dispatch does not borrow another level"
            ),
            diagnostics=_receive_diagnostics(receive_report),
            window_metadata=rx_window,
        )
        return
    if family == "listening_quality":
        _apply_receive_listening(
            result,
            receive_report=receive_report,
            rx_window=rx_window,
            mode=mode,
        )
        return
    if family in {"rlr", "frequency_response", "out_of_band"}:
        _merge_model_details(
            result,
            {"analysis_scope": _full_context_scope(f"receive_{family}")},
        )
    if family == "frequency_response":
        filtered_bands, response_scope = _receive_frequency_response_bands(
            receive_report,
            mode=mode,
        )
        _merge_model_details(
            result,
            {
                "receive_frequency_bands_dbfs": filtered_bands,
                "receive_frequency_response_scope": response_scope,
                "receive_transfer_response": deepcopy(receive_report.get("acceptance_transfer_response")),
            },
        )
    try:
        mapped = _receive_family_value(
            family,
            receive_report,
            mode=mode,
        )
    except (TypeError, ValueError) as error:
        _raise_storage_failure(error)
        _set_not_measured(
            result,
            reason=str(error),
            status="invalid",
            diagnostics=_receive_diagnostics(receive_report),
            window_metadata=rx_window,
        )
        return
    if mapped is None:
        _set_not_measured(
            result,
            reason=f"no receive engineering mapping for family {family!r}",
            diagnostics=_receive_diagnostics(receive_report),
            window_metadata=rx_window,
        )
        return
    metric, value, reason = mapped
    if value is None:
        _set_not_measured(
            result,
            reason=(
                f"{metric} was not available from the selected receive window; "
                f"{reason}"
            ),
            diagnostics=_receive_diagnostics(receive_report),
            window_metadata=rx_window,
        )
        return
    if isinstance(value, Mapping):
        if not value:
            _set_not_measured(
                result,
                reason=f"{metric} contained no frequency bands",
                diagnostics=_receive_diagnostics(receive_report),
                window_metadata=rx_window,
            )
            return
        try:
            for band, level in value.items():
                _finite_value(level, f"{metric} band {band}")
        except ValueError as error:
            _raise_storage_failure(error)
            _set_not_measured(
                result,
                reason=str(error),
                status="invalid",
                diagnostics=_receive_diagnostics(receive_report),
                window_metadata=rx_window,
            )
            return
    else:
        try:
            value = _finite_value(value, metric)
        except ValueError as error:
            _raise_storage_failure(error)
            _set_not_measured(
                result,
                reason=str(error),
                status="invalid",
                diagnostics=_receive_diagnostics(receive_report),
                window_metadata=rx_window,
            )
            return
    _set_engineering_value(
        result,
        metric=metric,
        value=value,
        reason=reason + "；统一 review_required，不代表标准符合性",
        diagnostics=_receive_diagnostics(receive_report),
        window_metadata=rx_window,
    )


def calculate_case_items(
    case: dict,
    input_paths: dict[str, Path],
    outputs: dict[str, Path],
    execution: dict,
    *,
    report_id: str,
    receive_config: Mapping[str, Any] | None = None,
    receive_ir_paths: Mapping[str, Path] | None = None,
) -> list[dict]:
    """Calculate exactly the requested contract rows for one acquisition case.

    ``outputs['tx']`` is the real VCP ``after.wav`` and ``outputs['rx']`` is the
    real VCP ``rx-spk.wav``.  Only the output needed by the case direction is
    required.  The function returns one row per requested item id and isolates
    failures to the affected model/family.
    """
    if not isinstance(case, dict):
        raise AcceptanceModelError("case must be a dictionary")
    if not isinstance(input_paths, dict):
        raise AcceptanceModelError("input_paths must be a dictionary")
    if not isinstance(outputs, dict):
        raise AcceptanceModelError("outputs must be a dictionary")
    if not isinstance(execution, dict):
        raise AcceptanceModelError("execution must be a dictionary")

    conditions = case.get("conditions")
    if not isinstance(conditions, dict):
        raise AcceptanceModelError("case.conditions must be a dictionary")
    item_ids = case.get("item_ids")
    if not isinstance(item_ids, list):
        raise AcceptanceModelError("case.item_ids must be a list")
    case_id = case.get("id")
    if not isinstance(case_id, str) or not case_id:
        raise AcceptanceModelError("case.id must be a non-empty string")
    if not isinstance(report_id, str) or not report_id:
        raise AcceptanceModelError("report_id must be a non-empty string")

    start_s, window_ms = _validate_window(
        case.get("analysis_start_s", 0),
        conditions.get("window_ms"),
    )
    all_contract_items = _contract_lookup()
    prepared_inputs, input_hashes, input_errors = _prepare_inputs(
        input_paths,
        start_s=start_s,
        window_ms=window_ms,
    )
    output_hashes: dict[str, str] = {}
    for role in ("tx", "rx"):
        raw_path = outputs.get(role)
        if raw_path is not None and Path(raw_path).is_file():
            output_hashes[role] = _sha256_file(Path(raw_path))

    binding = {
        "mode": conditions.get("mode"),
        "scene": conditions.get("scene"),
        "volume": conditions.get("volume"),
        "window_ms": conditions.get("window_ms"),
        "input_level_db": conditions.get("input_level_db"),
        "case_id": case_id,
        "report_id": report_id,
        "batch_id": execution.get("batch_id"),
        "dataset_id": execution.get("dataset_id"),
        "input_sha256": input_hashes,
        "output_sha256": output_hashes,
    }

    entries: list[dict[str, Any]] = []
    for raw_item_id in item_ids:
        if not isinstance(raw_item_id, str):
            raise AcceptanceModelError("case.item_ids entries must be strings")
        contract = all_contract_items.get(raw_item_id)
        result = _empty_result(
            contract,
            item_id=raw_item_id,
            reason="not calculated",
            binding=binding,
        )
        entry = {
            "item_id": raw_item_id,
            "contract": contract,
            "result": result,
            "matched": False,
        }
        entries.append(entry)
        if contract is None:
            _set_not_measured(
                result,
                reason=f"unknown contract item id: {raw_item_id}",
            )
            continue
        if not matching_conditions(contract, conditions):
            expected = required_conditions(contract)
            _set_not_measured(
                result,
                reason=(
                    "case conditions do not match the contract row; "
                    f"expected={expected!r}, supplied={conditions!r}"
                ),
            )
            continue
        entry["matched"] = True

    integrity_error = _declared_output_mismatch(
        execution,
        outputs,
        output_hashes,
    )
    if integrity_error is not None:
        for entry in entries:
            if entry["matched"]:
                _set_not_measured(
                    entry["result"],
                    reason=integrity_error,
                    status="invalid",
                )
        return [entry["result"] for entry in entries]

    tx: dict[str, Any] | None = None
    rx: dict[str, Any] | None = None
    tx_error: str | None = None
    rx_error: str | None = None
    matched_entries = [entry for entry in entries if entry["matched"]]
    needs_tx = any(
        entry["contract"]["direction"] in {"send", "round_trip"}
        for entry in matched_entries
    )
    needs_rx = any(
        entry["contract"]["direction"] in {"receive", "round_trip"}
        for entry in matched_entries
    )
    if needs_tx:
        try:
            tx = _prepare_output(
                outputs,
                "tx",
                start_s=start_s,
                window_ms=window_ms,
            )
        except Exception as error:
            _raise_storage_failure(error)
            tx_error = str(error)
    if needs_rx:
        try:
            rx = _prepare_output(
                outputs,
                "rx",
                start_s=start_s,
                window_ms=window_ms,
            )
        except Exception as error:
            _raise_storage_failure(error)
            rx_error = str(error)

    send_entries = [
        entry
        for entry in matched_entries
        if entry["contract"]["direction"] in {"send", "round_trip"}
    ]
    send_evaluation: dict[str, Any] | None = None
    send_error: str | None = None
    # A stationary frequency sweep is not speech. Its dedicated H1 path below
    # must not inherit the five-signal evaluator's speech-activity gate.
    if any(entry["contract"]["family"] != "frequency_response" for entry in send_entries):
        if tx is None:
            send_error = tx_error or "outputs['tx'] is required for send/round_trip"
        else:
            try:
                send_evaluation = _send_report(
                    mode=str(conditions.get("mode")),
                    prepared_inputs=prepared_inputs,
                    tx=tx,
                )
            except Exception as error:
                _raise_storage_failure(error)
                missing = "; ".join(
                    f"{role}: {reason}"
                    for role, reason in sorted(input_errors.items())
                )
                send_error = str(error)
                if missing:
                    send_error += f" ({missing})"

    send_items: dict[str, Mapping[str, Any]] = {}
    model_a: dict[str, Any] = {}
    if send_evaluation is not None:
        raw_items = send_evaluation.get("items")
        if isinstance(raw_items, list):
            send_items = {
                str(item.get("item_id")): item
                for item in raw_items
                if isinstance(item, Mapping) and item.get("item_id") is not None
            }
        model_a = _model_a_diagnostics(send_evaluation)

    dnsmos_send: dict[str, Any] | None = None
    needs_send_dnsmos = any(
        entry["contract"]["family"] in DNSMOS_SEND_METRICS
        for entry in send_entries
    )
    if needs_send_dnsmos:
        if tx is None:
            dnsmos_send = {
                "available": False,
                "reason": tx_error or "outputs['tx'] is required for DNSMOS",
            }
        else:
            dnsmos_send = _score_dnsmos(
                tx["samples"],
                tx["rate"],
                label="outputs['tx'] selected window",
            )

    for entry in send_entries:
        result = entry["result"]
        family = entry["contract"]["family"]
        if model_a:
            _merge_model_details(result, model_a)
        if entry["contract"]["direction"] == "round_trip":
            continue
        tx_window = tx.get("window") if tx is not None else None
        if family == "frequency_response":
            _apply_send_frequency_response(
                result, reference=prepared_inputs.get("reference"),
                noise=prepared_inputs.get("noise"), tx=tx,
                mode=str(conditions.get("mode")),
            )
            continue
        if family in DNSMOS_SEND_METRICS:
            assert dnsmos_send is not None
            _apply_send_dnsmos(
                result,
                dnsmos_send,
                model_a=model_a,
                tx_window=tx_window or {},
            )
            continue
        if family == "cng_spectrum":
            noise_view = (
                _window_view(
                    prepared_inputs["noise"],
                    start_s=start_s,
                    window_ms=window_ms,
                    label="input noise",
                )
                if "noise" in prepared_inputs
                else None
            )
            tx_view = (
                _window_view(
                    tx,
                    start_s=start_s,
                    window_ms=window_ms,
                    label="VCP tx noise",
                )
                if tx is not None
                else None
            )
            _apply_cng_spectrum(
                result,
                noise=noise_view,
                tx=tx_view,
                mode=str(conditions.get("mode")),
                model_a=model_a,
            )
            _merge_model_details(
                result,
                {"cng_spectrum": result.get("diagnostics", {}).get("cng_spectrum")},
            )
            continue
        if family in SEND_WINDOW_FAMILIES:
            if tx is None:
                _set_not_measured(
                    result,
                    reason=tx_error or "outputs['tx'] is required for a windowed metric",
                    status="invalid",
                    diagnostics=model_a,
                )
                continue
            if family == "out_of_band" and tx["rate"] / 2 <= MODE_UPPER_HZ[_bandwidth(str(conditions["mode"]))]:
                _set_not_measured(result, reason="带外频带不可观测：需要更高采样率的 VCP 输出，不能以数值下限代替测量")
                continue
            try:
                metric, value, reason, details, selected_window = _selected_window_metric(
                    family,
                    noise=prepared_inputs.get("noise"),
                    tx=tx,
                    mode=str(conditions.get("mode")),
                    start_s=start_s,
                    window_ms=window_ms,
                )
            except Exception as error:
                _raise_storage_failure(error)
                _set_not_measured(
                    result,
                    reason=f"selected-window {family} calculation failed: {error}",
                    status="invalid",
                    diagnostics=model_a,
                )
                continue
            _set_engineering_value(
                result,
                metric=metric,
                value=value,
                reason=reason + "；统一 review_required，不代表标准符合性",
                diagnostics={**model_a, "selected_window": details},
                window_metadata=selected_window,
            )
            continue
        if send_error is not None:
            _set_not_measured(
                result,
                reason=send_error,
                status="invalid",
                diagnostics=model_a,
                window_metadata=tx_window or {},
            )
            continue
        _apply_send_evaluation(
            result,
            send_items.get(entry["item_id"]),
            model_a=model_a,
            tx_window=tx_window or {},
        )

    round_trip_entries = [
        entry
        for entry in matched_entries
        if entry["contract"]["direction"] == "round_trip"
    ]
    for entry in round_trip_entries:
        result = entry["result"]
        if tx is None or rx is None:
            _set_not_measured(
                result,
                reason=(
                    "round-trip proxy requires both outputs['tx'] and outputs['rx']: "
                    + "; ".join(
                        reason
                        for reason in (tx_error, rx_error)
                        if reason is not None
                    )
                ),
                status="invalid",
                diagnostics=model_a,
            )
            continue
        analysis = (
            send_evaluation.get("analysis")
            if isinstance(send_evaluation, Mapping)
            else None
        )
        values = analysis.get("values") if isinstance(analysis, Mapping) else None
        delay = values.get("reference_to_output_delay_ms") if isinstance(values, Mapping) else None
        if delay is None:
            _set_not_measured(
                result,
                reason=send_error or "five-signal analysis did not produce a relative delay",
                diagnostics=model_a,
            )
            continue
        try:
            numeric = _finite_value(delay, "relative output delay")
        except ValueError as error:
            _raise_storage_failure(error)
            _set_not_measured(
                result,
                reason=str(error),
                status="invalid",
                diagnostics=model_a,
            )
            continue
        _set_engineering_value(
            result,
            metric="reference_to_output_relative_delay_ms",
            value=numeric,
            reason=(
                "tx 相对输入参考的时延代理；不是同步端到端往返时延，"
                "不得套用 RTD 合同门限或声明正式通过"
            ),
            diagnostics=model_a,
        )
        _merge_model_details(
            result,
            {
                "analysis_scope": _full_context_scope(
                    "five_signal_round_trip_delay"
                )
            },
        )

    receive_entries = [
        entry
        for entry in matched_entries
        if entry["contract"]["direction"] == "receive"
    ]
    receive_report: dict[str, Any] | None = None
    receive_error: str | None = None
    if receive_entries:
        if rx is None:
            receive_error = rx_error or "outputs['rx'] is required for a receive case"
        else:
            try:
                receive_report = _receive_report(
                    mode=str(conditions.get("mode")),
                    rx=rx,
                    start_s=start_s,
                    window_ms=window_ms,
                    reference=prepared_inputs.get("far_end"),
                    receive_config=receive_config,
                    receive_ir_paths=receive_ir_paths,
                )
            except Exception as error:
                _raise_storage_failure(error)
                receive_error = str(error)
    for entry in receive_entries:
        result = entry["result"]
        rx_window = rx.get("window") if rx is not None else None
        if receive_report is not None:
            receive_id = receive_report.get("evaluation_id")
            _merge_model_details(
                result,
                {
                    "receive_report_id": receive_id,
                    "receive_report": {
                        "evaluation_id": receive_id,
                        "evidence_kind": receive_report.get("evidence_kind"),
                        "synthetic_proxy": receive_report.get("synthetic_proxy"),
                        "input_processing": receive_report.get(
                            "acceptance_input_processing"
                        ),
                    },
                },
            )
        if (
            entry["contract"]["family"]
            in {"idle_noise_level", "idle_noise_peaks"}
            and rx is not None
        ):
            try:
                rx_window = _window_view(
                    rx,
                    start_s=start_s,
                    window_ms=window_ms,
                    label="VCP rx noise",
                )["window"]
            except Exception as error:
                _raise_storage_failure(error)
                pass
        if entry["contract"]["family"] == "rlr_linearity":
            _apply_receive_family(
                result,
                receive_report=receive_report or {},
                rx_window=rx_window or {},
                mode=str(conditions.get("mode")),
            )
            continue
        if receive_report is None:
            _set_not_measured(
                result,
                reason=receive_error or "synthetic receive evaluation failed",
                status="invalid",
                window_metadata=rx_window or {},
            )
            continue
        _apply_receive_family(
            result,
            receive_report=receive_report,
            rx_window=rx_window or {},
            mode=str(conditions.get("mode")),
        )

    return [entry["result"] for entry in entries]


@lru_cache(maxsize=1)
def _contract_lookup() -> dict[str, dict[str, Any]]:
    from .acceptance_contract import contract_items

    lookup: dict[str, dict[str, Any]] = {}
    for row in contract_items():
        lookup[row["id"]] = row
        lookup.setdefault(row["source_cell"], row)
        lookup.setdefault(f"{row['sheet']}:{row['source_cell']}", row)
    return lookup


__all__ = [
    "AcceptanceModelError",
    "calculate_case_items",
]
