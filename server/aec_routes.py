"""Central AEC route selection, capability admission, and report identity checks."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
import subprocess
from typing import Any


AEC_IDS = frozenset(
    {
        "webrtc",
        "speex",
        "speex_fused_cpu",
        "speex_fused_dsp",
        "computer_pf_cpu",
        "localvqe_aec200k",
    }
)

_SELECTION_TO_BACKEND = {
    "webrtc": "webrtc",
    "aec3": "webrtc",
    "speex": "speex",
    "speex_fused_cpu": "speex_fused_cpu",
    "speex_fused_dsp": "speex_fused_dsp",
    "computer_pf_cpu": "computer_pf_cpu",
    "localvqe_aec200k": "localvqe_aec200k",
}
_EXPECTED_REPORT_BACKEND = {
    "webrtc": "webrtc_aec3",
    "speex": "speex_mdf",
    "speex_fused_cpu": "speex_mdf_nlms_fused_cpu",
    "speex_fused_dsp": "speex_mdf_complex_pf_dsp",
    "computer_pf_cpu": "speex_mdf_complex_pf_cpu",
    "localvqe_aec200k": "localvqe_aec200k_cpu",
}
_LEGACY_CPU_FUSED_REPORT_BACKEND = "speex_mdf_nlms_fused"
_CAPABILITY_SCHEMA = "itu-aec-capabilities-v1"
_DSP_ROUTE = "speex_fused_dsp"
_DSP_ARCHITECTURE = "mdf_complex_pf"
_DSP_REPORT_BACKEND = "speex_mdf_complex_pf_dsp"
_COMPUTER_PF_CPU_ROUTE = "computer_pf_cpu"
_COMPUTER_PF_CPU_ARCHITECTURE = "mdf_complex_pf_cpu"
_COMPUTER_PF_CPU_REPORT_BACKEND = "speex_mdf_complex_pf_cpu"
_LOCALVQE_ROUTE = "localvqe_aec200k"
_LOCALVQE_ARCHITECTURE = "localvqe_daf_neural_mask"
_LOCALVQE_REPORT_BACKEND = "localvqe_aec200k_cpu"
_LOCALVQE_NEURAL_DELAY = 256
_LOCALVQE_ADAPTER_DELAY = 224
_LOCALVQE_DELAY = _LOCALVQE_NEURAL_DELAY + _LOCALVQE_ADAPTER_DELAY
_CAPABILITY_ROUTES = frozenset(
    {_DSP_ROUTE, _COMPUTER_PF_CPU_ROUTE, _LOCALVQE_ROUTE}
)
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")
_MODEL_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,63}\Z")
_DSP_DELAY_MIN = 212
_DSP_DELAY_MAX = 16000
_DSP_DELAY_STEP = 160
_ADMITTED_IDENTITY_FIELDS = (
    ("aec_execution_provider", "execution_provider", "text"),
    ("aec_device_id", "device_id", "text"),
    ("aec_model_id", "model_id", "model_id"),
    ("aec_model_sha256", "model_sha256", "sha256"),
    ("aec_frontend_sha256", "frontend_sha256", "sha256"),
    ("aec_provider_sha256", "provider_sha256", "sha256"),
    ("aec_neural_delay_samples", "delay_samples", "delay"),
)


def aec_backend_for(configuration: dict | None) -> str | None:
    """Return the canonical AEC backend selected by a chain configuration.

    ``aec3`` is the user-facing name for WebRTC AEC3, while ``off`` disables
    AEC. Malformed configurations are rejected instead of silently selecting a
    default backend.
    """
    if configuration is None:
        return None
    if not isinstance(configuration, dict):
        raise ValueError("AEC 配置必须是对象")
    selection = configuration.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("AEC 配置缺少 selection 对象")
    if "aec" not in selection:
        raise ValueError("AEC selection 缺少 aec 字段")
    selected = selection["aec"]
    if not isinstance(selected, str):
        raise ValueError("AEC selection.aec 必须是字符串")
    if selected == "off":
        return None
    backend = _SELECTION_TO_BACKEND.get(selected)
    if backend is None:
        supported = ", ".join(sorted({*_SELECTION_TO_BACKEND, "off"}))
        raise ValueError(f"不支持的 AEC 后端 {selected!r}；可选值：{supported}")
    return backend


def aec_route_requires_capability(configuration: dict | None) -> bool:
    """Return whether the selected route needs a capability identity check."""
    return aec_backend_for(configuration) in _CAPABILITY_ROUTES


def effective_aec_reference_channels(
    configuration: dict | None, reference_channels: int
) -> int:
    """Return the channel count consumed by the selected AEC route.

    LocalVQE is mono-reference. The catalog preset may explicitly select one
    playback channel; that selection is a deterministic adapter, not a
    multi-render downmix.
    """
    if aec_backend_for(configuration) != _LOCALVQE_ROUTE:
        return reference_channels
    extra_flags = configuration.get("extra_flags") if isinstance(configuration, dict) else None
    if not isinstance(extra_flags, list):
        return reference_channels
    for index, flag in enumerate(extra_flags[:-1]):
        if flag != "--localvqe-reference-channel":
            continue
        value = extra_flags[index + 1]
        if isinstance(value, str) and value.isdigit():
            selected = int(value)
            if 0 <= selected < reference_channels:
                return 1
    return reference_channels


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _is_model_id(value: Any) -> bool:
    return isinstance(value, str) and _MODEL_ID_RE.fullmatch(value) is not None


def _is_dsp_delay(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int)
        and _DSP_DELAY_MIN <= value <= _DSP_DELAY_MAX
        and value % _DSP_DELAY_STEP == 0
    )


def _is_computer_pf_cpu_delay(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int)
        and value == 0
    )


def _is_localvqe_delay(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int)
        and value == _LOCALVQE_DELAY
    )


def _identity_value_valid(value: Any, kind: str) -> bool:
    if kind == "sha256":
        return _is_sha256(value)
    if kind == "model_id":
        return _is_model_id(value)
    if kind == "delay":
        return _is_dsp_delay(value)
    if kind == "cpu_pf_delay":
        return _is_computer_pf_cpu_delay(value)
    if kind == "localvqe_delay":
        return _is_localvqe_delay(value)
    return _is_nonempty_string(value)


def _identity_requirement(kind: str) -> str:
    if kind == "sha256":
        return "64 位十六进制字符串"
    if kind == "model_id":
        return "匹配 [A-Za-z0-9_-]{1,63} 的字符串"
    if kind == "delay":
        return "212 到 16000 之间且能被 160 整除的整数"
    if kind == "cpu_pf_delay":
        return "整数 0"
    if kind == "localvqe_delay":
        return f"整数 {_LOCALVQE_DELAY}"
    return "非空字符串"


def validate_admitted_aec_identity(
    report: dict, capability: dict | None
) -> list[str]:
    """Bind a run report to the capability identity admitted before execution."""
    if capability is None:
        return []
    if not isinstance(report, dict):
        return ["AEC 报告必须是 JSON 对象"]
    if not isinstance(capability, dict):
        return ["AEC capability 必须是对象"]

    errors: list[str] = []
    for report_field, capability_field, kind in _ADMITTED_IDENTITY_FIELDS:
        if (
            kind == "delay"
            and capability.get("id") == _COMPUTER_PF_CPU_ROUTE
        ):
            kind = "cpu_pf_delay"
        if (
            kind == "delay"
            and capability.get("id") == _LOCALVQE_ROUTE
        ):
            kind = "localvqe_delay"
            report_field = "aec_algorithm_delay_samples_16k"
        report_present = report_field in report
        capability_present = capability_field in capability
        report_value = report.get(report_field)
        capability_value = capability.get(capability_field)

        report_valid = _identity_value_valid(report_value, kind)
        capability_valid = _identity_value_valid(capability_value, kind)
        requirement = _identity_requirement(kind)
        if not report_present or not report_valid:
            errors.append(f"{report_field} 必须存在且为{requirement}")
        if not capability_present or not capability_valid:
            errors.append(f"capability.{capability_field} 必须存在且为{requirement}")
        if (
            report_present
            and capability_present
            and report_valid
            and capability_valid
            and report_value != capability_value
        ):
            errors.append(
                f"{report_field} 必须与 capability.{capability_field} 完全一致"
            )
    return errors


def _validate_dsp_report(report: dict) -> list[str]:
    errors: list[str] = []

    if "aec_enabled" in report and report["aec_enabled"] is not True:
        errors.append(
            f"aec_enabled 存在时必须为 true（实际为 {report['aec_enabled']!r}）"
        )

    requested = report.get("aec_backend_requested")
    if requested != _DSP_ROUTE:
        errors.append(
            "aec_backend_requested 必须是 speex_fused_dsp"
            f"（实际为 {requested!r}）"
        )

    effective = report.get("aec_backend_effective")
    if effective != _DSP_REPORT_BACKEND:
        errors.append(
            "aec_backend_effective 必须是 speex_mdf_complex_pf_dsp"
            f"（实际为 {effective!r}）"
        )

    actual = report.get("aec_backend")
    if actual != _DSP_REPORT_BACKEND:
        errors.append(
            "aec_backend 必须与 aec_backend_effective 同为 "
            f"speex_mdf_complex_pf_dsp（实际为 {actual!r}）"
        )

    provider = report.get("aec_execution_provider")
    if provider != "hifi_dsp":
        errors.append(f"aec_execution_provider 必须是 hifi_dsp（实际为 {provider!r}）")

    for field in ("aec_device_id", "aec_model_id"):
        if field == "aec_model_id":
            if not _is_model_id(report.get(field)):
                errors.append(
                    f"{field} 必须匹配 [A-Za-z0-9_-]{{1,63}}"
                    f"（实际为 {report.get(field)!r}）"
                )
        elif not _is_nonempty_string(report.get(field)):
            errors.append(f"{field} 必须是非空字符串")

    for field in (
        "aec_model_sha256",
        "aec_frontend_sha256",
        "aec_provider_sha256",
    ):
        if not _is_sha256(report.get(field)):
            errors.append(f"{field} 必须是 64 位十六进制字符串")

    inferences = report.get("aec_dsp_inferences")
    if (
        isinstance(inferences, bool)
        or not isinstance(inferences, int)
        or inferences <= 0
    ):
        errors.append("aec_dsp_inferences 必须是正整数且不能是布尔值")

    processed_frames = report.get("aec_dsp_processed_frames")
    if (
        isinstance(processed_frames, bool)
        or not isinstance(processed_frames, int)
        or processed_frames <= 0
    ):
        errors.append(
            "aec_dsp_processed_frames 必须是正整数且不能是布尔值"
        )

    fallback_frames = report.get("aec_dsp_fallback_frames")
    if (
        isinstance(fallback_frames, bool)
        or not isinstance(fallback_frames, int)
        or fallback_frames != 0
    ):
        errors.append("aec_dsp_fallback_frames 必须是整数 0")

    neural_delay = report.get("aec_neural_delay_samples")
    if not _is_dsp_delay(neural_delay):
        errors.append(
            "aec_neural_delay_samples 必须是 212 到 16000 之间"
            f"且能被 160 整除的整数（实际为 {neural_delay!r}）"
        )

    return errors


def _validate_computer_pf_cpu_report(report: dict) -> list[str]:
    errors: list[str] = []

    if "aec_enabled" in report and report["aec_enabled"] is not True:
        errors.append(
            f"aec_enabled 存在时必须为 true（实际为 {report['aec_enabled']!r}）"
        )

    requested = report.get("aec_backend_requested")
    if requested != _COMPUTER_PF_CPU_ROUTE:
        errors.append(
            "aec_backend_requested 必须是 computer_pf_cpu"
            f"（实际为 {requested!r}）"
        )

    effective = report.get("aec_backend_effective")
    if effective != _COMPUTER_PF_CPU_REPORT_BACKEND:
        errors.append(
            "aec_backend_effective 必须是 speex_mdf_complex_pf_cpu"
            f"（实际为 {effective!r}）"
        )

    actual = report.get("aec_backend")
    if actual != _COMPUTER_PF_CPU_REPORT_BACKEND:
        errors.append(
            "aec_backend 必须与 aec_backend_effective 同为 "
            f"speex_mdf_complex_pf_cpu（实际为 {actual!r}）"
        )

    provider = report.get("aec_execution_provider")
    if provider != "cpu":
        errors.append(f"aec_execution_provider 必须是 cpu（实际为 {provider!r}）")

    device = report.get("aec_device_id")
    if device != "host_cpu":
        errors.append(f"aec_device_id 必须是 host_cpu（实际为 {device!r}）")

    if not _is_model_id(report.get("aec_model_id")):
        errors.append(
            "aec_model_id 必须匹配 [A-Za-z0-9_-]{1,63}"
            f"（实际为 {report.get('aec_model_id')!r}）"
        )

    for field in (
        "aec_model_sha256",
        "aec_frontend_sha256",
        "aec_provider_sha256",
    ):
        if not _is_sha256(report.get(field)):
            errors.append(f"{field} 必须是 64 位十六进制字符串")

    for field in ("aec_cpu_inferences", "aec_cpu_processed_frames"):
        value = report.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            errors.append(f"{field} 必须是正整数且不能是布尔值")

    fallback_frames = report.get("aec_cpu_fallback_frames")
    if (
        isinstance(fallback_frames, bool)
        or not isinstance(fallback_frames, int)
        or fallback_frames != 0
    ):
        errors.append("aec_cpu_fallback_frames 必须是整数 0")

    neural_delay = report.get("aec_neural_delay_samples")
    if not _is_computer_pf_cpu_delay(neural_delay):
        errors.append(
            "aec_neural_delay_samples 必须是整数 0"
            f"（实际为 {neural_delay!r}）"
        )

    processing_mode = report.get("aec_processing_mode")
    if processing_mode != "offline_noncausal":
        errors.append(
            "aec_processing_mode 必须是 offline_noncausal"
            f"（实际为 {processing_mode!r}）"
        )

    training_status = report.get("model_training_status")
    if training_status != "locally_trained_experimental":
        errors.append(
            "model_training_status 必须是 locally_trained_experimental"
            f"（实际为 {training_status!r}）"
        )

    return errors


def _validate_localvqe_report(report: dict) -> list[str]:
    errors: list[str] = []
    if "aec_enabled" in report and report["aec_enabled"] is not True:
        errors.append(
            f"aec_enabled 存在时必须为 true（实际为 {report['aec_enabled']!r}）"
        )
    if report.get("aec_backend_requested") != _LOCALVQE_ROUTE:
        errors.append(
            "aec_backend_requested 必须是 localvqe_aec200k"
            f"（实际为 {report.get('aec_backend_requested')!r}）"
        )
    for field in ("aec_backend_effective", "aec_backend"):
        if report.get(field) != _LOCALVQE_REPORT_BACKEND:
            errors.append(
                f"{field} 必须是 localvqe_aec200k_cpu"
                f"（实际为 {report.get(field)!r}）"
            )
    if report.get("aec_execution_provider") != "cpu":
        errors.append(
            "aec_execution_provider 必须是 cpu"
            f"（实际为 {report.get('aec_execution_provider')!r}）"
        )
    if report.get("aec_device_id") != "host_cpu":
        errors.append(
            "aec_device_id 必须是 host_cpu"
            f"（实际为 {report.get('aec_device_id')!r}）"
        )
    if not _is_model_id(report.get("aec_model_id")):
        errors.append(
            "aec_model_id 必须匹配 [A-Za-z0-9_-]{1,63}"
            f"（实际为 {report.get('aec_model_id')!r}）"
        )
    for field in (
        "aec_model_sha256",
        "aec_frontend_sha256",
        "aec_provider_sha256",
    ):
        if not _is_sha256(report.get(field)):
            errors.append(f"{field} 必须是 64 位十六进制字符串")
    for field in (
        "aec_localvqe_inferences",
        "aec_localvqe_processed_frames",
    ):
        value = report.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            errors.append(f"{field} 必须是正整数且不能是布尔值")
    fallback_frames = report.get("aec_localvqe_fallback_frames")
    if (
        isinstance(fallback_frames, bool)
        or not isinstance(fallback_frames, int)
        or fallback_frames != 0
    ):
        errors.append("aec_localvqe_fallback_frames 必须是整数 0")
    neural_delay = report.get("aec_neural_delay_samples")
    if type(neural_delay) is not int or neural_delay != _LOCALVQE_NEURAL_DELAY:
        errors.append(
            "aec_neural_delay_samples 必须是整数 256"
            f"（实际为 {report.get('aec_neural_delay_samples')!r}）"
        )
    for field, expected_delay in (
        ("aec_hop_adapter_delay_samples_16k", _LOCALVQE_ADAPTER_DELAY),
        ("aec_algorithm_delay_samples_16k", _LOCALVQE_DELAY),
    ):
        actual_delay = report.get(field)
        if type(actual_delay) is not int or actual_delay != expected_delay:
            errors.append(f"{field} 必须是整数 {expected_delay}（实际为 {actual_delay!r}）")
    for field, expected_delay in (
        ("aec_model_codec_delay_samples_16k", 256),
        ("aec_linear_frontend_delay_samples_16k", 0),
    ):
        if type(report.get(field)) is not int or report.get(field) != expected_delay:
            errors.append(
                f"{field} 必须是整数 {expected_delay}"
                f"（实际为 {report.get(field)!r}）"
            )
    if report.get("aec_processing_mode") != "causal_streaming":
        errors.append(
            "aec_processing_mode 必须是 causal_streaming"
            f"（实际为 {report.get('aec_processing_mode')!r}）"
        )
    return errors


def _validate_cpu_report(report: dict, backend: str) -> list[str]:
    errors: list[str] = []
    expected = _EXPECTED_REPORT_BACKEND[backend]

    if "aec_enabled" in report and report["aec_enabled"] is not True:
        errors.append(
            f"aec_enabled 存在时必须为 true（实际为 {report['aec_enabled']!r}）"
        )

    if (
        "aec_backend_requested" in report
        and report["aec_backend_requested"] != backend
    ):
        errors.append(
            f"aec_backend_requested 存在时必须为 {backend!r}"
            f"（实际为 {report['aec_backend_requested']!r}）"
        )

    if (
        "aec_backend_effective" in report
        and report["aec_backend_effective"] != expected
    ):
        errors.append(
            f"aec_backend_effective 存在时必须为 {expected!r}"
            f"（实际为 {report['aec_backend_effective']!r}）"
        )

    accepted = {expected}
    if backend == "speex_fused_cpu":
        accepted.add(_LEGACY_CPU_FUSED_REPORT_BACKEND)
    actual = report.get("aec_backend")
    if actual not in accepted:
        names = " 或 ".join(repr(name) for name in sorted(accepted))
        errors.append(f"aec_backend 必须是 {names}（实际为 {actual!r}）")
    return errors


def validate_aec_route_report(
    report: dict | None, configuration: dict | None
) -> list[str]:
    """Validate that a pipeline report proves the selected AEC route ran.

    CPU routes accept only their exact current report names. The historical
    ``speex_mdf_nlms_fused`` name is accepted only for the CPU fused route; it
    can never be interpreted as the DSP route.
    """
    backend = aec_backend_for(configuration)
    if backend is None:
        return []
    if not isinstance(report, dict):
        return ["AEC 报告必须是 JSON 对象"]
    if backend == _DSP_ROUTE:
        return _validate_dsp_report(report)
    if backend == _COMPUTER_PF_CPU_ROUTE:
        return _validate_computer_pf_cpu_report(report)
    if backend == _LOCALVQE_ROUTE:
        return _validate_localvqe_report(report)

    return _validate_cpu_report(report, backend)


def _capability_error(detail: str) -> ValueError:
    return ValueError(f"AEC capabilities 无效：{detail}")


def read_aec_capabilities(cli: Path) -> dict:
    """Read and validate the CLI's AEC capability document."""
    try:
        completed = subprocess.run(
            [str(cli), "--aec-capabilities"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired as error:
        raise ValueError("AEC capabilities 查询超时（10 秒）") from error
    except OSError as error:
        raise ValueError(f"AEC capabilities 查询失败：{error}") from error

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        suffix = f"：{detail.splitlines()[-1]}" if detail else ""
        raise ValueError(
            f"AEC capabilities 查询失败（退出码 {completed.returncode}）{suffix}"
        )
    if not isinstance(completed.stdout, str):
        raise _capability_error("stdout 不是文本")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise _capability_error(f"stdout 不是合法 JSON：{error.msg}") from error
    if not isinstance(payload, dict):
        raise _capability_error("顶层必须是对象")
    if payload.get("schema") != _CAPABILITY_SCHEMA:
        raise _capability_error(
            f"schema 必须是 {_CAPABILITY_SCHEMA!r}"
            f"（实际为 {payload.get('schema')!r}）"
        )

    backends = payload.get("backends")
    if not isinstance(backends, list):
        raise _capability_error("backends 必须是列表")
    seen_ids: set[str] = set()
    for index, entry in enumerate(backends):
        prefix = f"backends[{index}]"
        if not isinstance(entry, dict):
            raise _capability_error(f"{prefix} 必须是对象")
        identifier = entry.get("id")
        if not _is_nonempty_string(identifier):
            raise _capability_error(f"{prefix}.id 必须是非空字符串")
        if identifier not in AEC_IDS:
            raise _capability_error(
                f"{prefix}.id 必须使用规范 AEC 后端名"
                f"（实际为 {identifier!r}）"
            )
        if identifier in seen_ids:
            raise _capability_error(f"backend id 重复：{identifier!r}")
        seen_ids.add(identifier)
        if "available" not in entry or not isinstance(entry["available"], bool):
            raise _capability_error(f"{prefix}.available 必须是布尔值")
        if not _is_nonempty_string(entry.get("architecture")):
            raise _capability_error(f"{prefix}.architecture 必须是非空字符串")
        if "reason" not in entry or not isinstance(entry["reason"], str):
            raise _capability_error(f"{prefix}.reason 必须是字符串")
    return payload


def _dsp_capability_identity_errors(entry: dict) -> list[str]:
    errors: list[str] = []
    if entry.get("architecture") != _DSP_ARCHITECTURE:
        errors.append(
            "architecture 必须是 mdf_complex_pf"
            f"（实际为 {entry.get('architecture')!r}）"
        )
    if entry.get("execution_provider") != "hifi_dsp":
        errors.append(
            "execution_provider 必须是 hifi_dsp"
            f"（实际为 {entry.get('execution_provider')!r}）"
        )
    if not _is_nonempty_string(entry.get("device_id")):
        errors.append("device_id 必须是非空字符串")
    if not _is_model_id(entry.get("model_id")):
        errors.append(
            "model_id 必须匹配 [A-Za-z0-9_-]{1,63}"
            f"（实际为 {entry.get('model_id')!r}）"
        )
    for field in ("model_sha256", "frontend_sha256", "provider_sha256"):
        if not _is_sha256(entry.get(field)):
            errors.append(f"{field} 必须是 64 位十六进制字符串")
    delay = entry.get("delay_samples")
    if not _is_dsp_delay(delay):
        errors.append(
            "delay_samples 必须是 212 到 16000 之间"
            f"且能被 160 整除的整数（实际为 {delay!r}）"
        )
    return errors


def _computer_pf_cpu_capability_identity_errors(entry: dict) -> list[str]:
    errors: list[str] = []
    if entry.get("architecture") != _COMPUTER_PF_CPU_ARCHITECTURE:
        errors.append(
            "architecture 必须是 mdf_complex_pf_cpu"
            f"（实际为 {entry.get('architecture')!r}）"
        )
    if entry.get("execution_provider") != "cpu":
        errors.append(
            "execution_provider 必须是 cpu"
            f"（实际为 {entry.get('execution_provider')!r}）"
        )
    if entry.get("device_id") != "host_cpu":
        errors.append(
            "device_id 必须是 host_cpu"
            f"（实际为 {entry.get('device_id')!r}）"
        )
    if not _is_model_id(entry.get("model_id")):
        errors.append(
            "model_id 必须匹配 [A-Za-z0-9_-]{1,63}"
            f"（实际为 {entry.get('model_id')!r}）"
        )
    for field in ("model_sha256", "frontend_sha256", "provider_sha256"):
        if not _is_sha256(entry.get(field)):
            errors.append(f"{field} 必须是 64 位十六进制字符串")
    if not _is_computer_pf_cpu_delay(entry.get("delay_samples")):
        errors.append(
            "delay_samples 必须是整数 0"
            f"（实际为 {entry.get('delay_samples')!r}）"
        )
    return errors


def _localvqe_capability_identity_errors(entry: dict) -> list[str]:
    errors: list[str] = []
    if entry.get("architecture") != _LOCALVQE_ARCHITECTURE:
        errors.append(
            "architecture 必须是 localvqe_daf_neural_mask"
            f"（实际为 {entry.get('architecture')!r}）"
        )
    if entry.get("execution_provider") != "cpu":
        errors.append(
            "execution_provider 必须是 cpu"
            f"（实际为 {entry.get('execution_provider')!r}）"
        )
    if entry.get("device_id") != "host_cpu":
        errors.append(
            "device_id 必须是 host_cpu"
            f"（实际为 {entry.get('device_id')!r}）"
        )
    if not _is_model_id(entry.get("model_id")):
        errors.append(
            "model_id 必须匹配 [A-Za-z0-9_-]{1,63}"
            f"（实际为 {entry.get('model_id')!r}）"
        )
    for field in ("model_sha256", "frontend_sha256", "provider_sha256"):
        if not _is_sha256(entry.get(field)):
            errors.append(f"{field} 必须是 64 位十六进制字符串")
    if not _is_localvqe_delay(entry.get("delay_samples")):
        errors.append(
            f"delay_samples 必须是整数 {_LOCALVQE_DELAY}"
            f"（实际为 {entry.get('delay_samples')!r}）"
        )
    for field, expected_delay in (
        ("model_codec_delay_samples_16k", 256),
        ("linear_frontend_delay_samples_16k", 0),
    ):
        if type(entry.get(field)) is not int or entry.get(field) != expected_delay:
            errors.append(
                f"{field} 必须是整数 {expected_delay}"
                f"（实际为 {entry.get(field)!r}）"
            )
    return errors


def _validate_complex_pf_input_envelope(
    backend: str,
    *,
    sample_rate: int | None,
    reference_channels: int | None,
    capture_channels: int | None,
    beamforming: bool | None,
) -> None:
    if sample_rate is not None:
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
            raise ValueError(f"{backend} sample_rate 必须是整数")
        if sample_rate not in ({16000, 24000, 48000} if backend == _LOCALVQE_ROUTE else {16000}):
            if backend == _LOCALVQE_ROUTE:
                raise ValueError(
                    f"{backend} 仅支持 sample_rate=16000/24000/48000"
                    f"（实际为 {sample_rate!r}）"
                )
            raise ValueError(
                f"{backend} v1 仅支持 sample_rate=16000"
                f"（实际为 {sample_rate!r}）"
            )
    if reference_channels is not None:
        if (
            isinstance(reference_channels, bool)
            or not isinstance(reference_channels, int)
        ):
            raise ValueError(f"{backend} reference_channels 必须是整数")
        if reference_channels != 1:
            raise ValueError(
                f"{backend} v1 仅支持 reference_channels=1"
                f"（实际为 {reference_channels!r}）"
            )
    if capture_channels is not None:
        if (
            isinstance(capture_channels, bool)
            or not isinstance(capture_channels, int)
            or not 1 <= capture_channels <= 8
        ):
            raise ValueError(
                f"{backend} capture_channels 必须是非布尔整数 1..8"
                f"（实际为 {capture_channels!r}）"
            )
    if beamforming is not None and not isinstance(beamforming, bool):
        raise ValueError(
            f"{backend} beamforming 必须是严格布尔值"
            f"（实际为 {beamforming!r}）"
        )
    if capture_channels is not None and capture_channels > 1 and beamforming is not True:
        raise ValueError(
            f"{backend} 多声道采集必须启用 beamforming=True"
            f"（capture_channels={capture_channels!r}, "
            f"beamforming={beamforming!r}）"
        )


def admit_aec_route(
    cli: Path,
    configuration: dict | None,
    *,
    sample_rate: int | None = None,
    reference_channels: int | None = None,
    capture_channels: int | None = None,
    beamforming: bool | None = None,
) -> dict | None:
    """Admit the selected route and return its frozen capability identity.

    Legacy CPU routes intentionally do not query ``--aec-capabilities`` so they
    remain usable with pre-capability CLIs. Neural routes have no silent CPU
    fallback and must prove their frozen execution identity before running.
    """
    backend = aec_backend_for(configuration)
    selection = configuration.get("selection", {}) if isinstance(configuration, dict) else {}
    if isinstance(selection, dict) and selection.get("ns") == "dpdfnet2_48k_hr":
        if backend != "webrtc":
            raise ValueError("DPDFNet2 48 kHz HR 质量路线必须使用 WebRTC AEC3")
        if sample_rate is not None and sample_rate != 48000:
            raise ValueError("DPDFNet2 48 kHz HR 质量路线只接受原生 48 kHz 输入")
    if backend not in _CAPABILITY_ROUTES:
        return None

    if (backend == _LOCALVQE_ROUTE and isinstance(selection, dict)
            and selection.get("ns") in {
                "cofilite", "dpdfnet2", "dpdfnet2_48k_hr", "hybrid_v2_pitch",
            }
            and sample_rate is not None and sample_rate != 16000):
        raise ValueError("LocalVQE 与所选外部 NS 的组合仅支持原生 16 kHz 输入，请在测试前显式重采样")

    _validate_complex_pf_input_envelope(
        backend,
        sample_rate=sample_rate,
        reference_channels=reference_channels,
        capture_channels=capture_channels,
        beamforming=beamforming,
    )

    capabilities = read_aec_capabilities(cli)
    entry = next(
        (
            item
            for item in capabilities["backends"]
            if item["id"] == backend
        ),
        None,
    )
    if entry is None:
        raise ValueError(
            f"{backend} 准入失败：AEC capabilities 缺少 "
            f"id='{backend}' 条目"
        )
    if not entry["available"]:
        reason = entry["reason"].strip() or "CLI 未提供原因"
        raise ValueError(f"{backend} 不可用：{reason}")

    if backend == _DSP_ROUTE:
        identity_errors = _dsp_capability_identity_errors(entry)
    elif backend == _COMPUTER_PF_CPU_ROUTE:
        identity_errors = _computer_pf_cpu_capability_identity_errors(entry)
    else:
        identity_errors = _localvqe_capability_identity_errors(entry)
    if identity_errors:
        raise ValueError(
            f"{backend} capability 身份不完整："
            + "；".join(identity_errors)
        )
    return deepcopy(entry)


# Native16 registration: additive dispatch; original source prefix remains frozen.

_native16_original_admit_aec_route = admit_aec_route
_native16_original_read_aec_capabilities = read_aec_capabilities
_native16_original_validate_aec_route_report = validate_aec_route_report
_native16_original_validate_admitted_aec_identity = validate_admitted_aec_identity

def _is_native16_configuration(configuration):
    return isinstance(configuration, dict) and isinstance(configuration.get("selection"), dict) and configuration["selection"].get("ns") == "cofilite_native16"

def _is_native16_cli(cli):
    return Path(cli).resolve() == (Path(__file__).resolve().parents[1] / "build/native16-20261117/host-v2/cpp/itu_audio_cli").resolve()

def admit_aec_route(cli, configuration, **kwargs):
    if _is_native16_configuration(configuration):
        from server.aec_routes_native16 import admit_aec_route as native
        return native(cli, configuration, **kwargs)
    if _is_native16_cli(cli):
        raise ValueError("原生16 ms CLI 只能用于明确选择的 native16 算法簇")
    return _native16_original_admit_aec_route(cli, configuration, **kwargs)

def read_aec_capabilities(cli):
    if _is_native16_cli(cli):
        from server.aec_routes_native16 import read_aec_capabilities as native
        return native(cli)
    return _native16_original_read_aec_capabilities(cli)

def validate_aec_route_report(report, configuration):
    if _is_native16_configuration(configuration):
        from server.aec_routes_native16 import validate_aec_route_report as native
        return native(report, configuration)
    return _native16_original_validate_aec_route_report(report, configuration)

def validate_admitted_aec_identity(report, capability):
    if isinstance(capability, dict) and capability.get("id") == "localvqe_aec200k" and capability.get("delay_samples") == 256:
        from server.aec_routes_native16 import validate_admitted_aec_identity as native
        return native(report, capability)
    return _native16_original_validate_admitted_aec_identity(report, capability)

# Geometry selection requires a matching native16 execution identity; a forged
# legacy capability with delay=256 must continue to fail the original contract.
def validate_admitted_aec_identity(report, capability):
    native_report = isinstance(report, dict) and report.get("frame_size") == 256 and report.get("ns_backend_requested") == "cofilite_native16"
    if native_report and isinstance(capability, dict) and capability.get("id") == "localvqe_aec200k" and capability.get("delay_samples") == 256:
        from server.aec_routes_native16 import validate_admitted_aec_identity as native
        return native(report, capability)
    return _native16_original_validate_admitted_aec_identity(report, capability)
