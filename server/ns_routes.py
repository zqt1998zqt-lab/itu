"""Shared external NS execution for the pinned cluster models.

The C++ engine owns BF/AEC/EQ/AGC/DRC/limiter and diagnostic taps. CoFi-Lite
and DPDFNet2 48 kHz HR run in persistent stateful workers called by the native
NS callback. DPDFNet2 16 kHz and hybrid V2 retain their explicit offline
capture/replay path. There is deliberately no model fallback path.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
MODEL_WORKER = ROOT / "scripts/run-ns-model.py"
RUNTIME_RECEIPTS = ROOT / "build/ns-clusters-20261115/runtime-model-receipts"
EXTERNAL_NS_IDS = frozenset({
    "cofilite",
    "dpdfnet2",
    "dpdfnet2_48k_hr",
    "hybrid_v2_pitch",
})
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAMPLE_RATE = 16_000
_FRAME_SAMPLES = 160
_REPLAY_DRAIN_SAMPLES = 16_000
_DPDFNET2_48K_MODEL_LATENCY_SAMPLES = 1_920
_DPDFNET2_48K_STARTUP_DELAY_SAMPLES = 480
_DPDFNET2_48K_ONLINE_DELAY_SAMPLES = (
    _DPDFNET2_48K_MODEL_LATENCY_SAMPLES + _DPDFNET2_48K_STARTUP_DELAY_SAMPLES
)


@dataclass(frozen=True)
class NSRouteSpec:
    backend: str
    model_id: str
    processing_mode: str
    model_path: Path
    model_sha256: str
    frontend_path: Path
    frontend_sha256: str
    delay_samples: int
    offline_whole_utterance: bool
    lookahead_samples: int
    sample_rate: int = 16_000
    frame_samples: int = 160
    window_samples: int = 160
    hop_samples: int = 160
    replay_drain_samples: int = 16_000
    extra_assets: tuple[tuple[Path, str], ...] = ()
    frame_executor: str | None = None
    frame_model_path: Path | None = None
    frame_model_sha256: str | None = None


NS_ROUTES: dict[str, NSRouteSpec] = {
    "cofilite": NSRouteSpec(
        backend="cofilite",
        model_id="cofilite",
        processing_mode="stateful_streaming",
        model_path=ROOT / "build/ns-cofilite-20261115/upstream/checkpoint/model_on_dns3.tar",
        model_sha256="40d1bb69dcd553ce93c745c8bd136e1593397f529aa53609827c69df95ce2a30",
        frontend_path=ROOT / "server/cofilite_onnx.py",
        frontend_sha256="21d3072fbb0879912911c391534ae0e6d5d2276c6e440601c936f23a9733e682",
        delay_samples=480,
        offline_whole_utterance=False,
        lookahead_samples=511,
        window_samples=512,
        hop_samples=256,
        extra_assets=((
            ROOT / "build/ns-cofilite-20261115/upstream/model/cofilite.py",
            "e77db8c40fbeef33b2496aca3f258d47205a1453c1bfb44c6b644f543c0afb9c",
        ), (
            ROOT / "build/ns-cofilite-20261115/cofi_infer.py",
            "a885236e79a2df861d396f8d3cf0dafbf2e5d448513c96c3db30a68e3e0c274e",
        ), (
            ROOT / "server/cofilite_stream.py",
            "3b7e08a21460d1f9195db180c2868e075045d642564806f1af3fee1fe39d4ae1",
        ), (
            ROOT / "server/cofilite_frame.py",
            "aad70af318a73d1010b0e8f92708b5137e167645a6e0e731fdda0b3f2b236e5a",
        ), (
            ROOT / "build/cofilite-ort-20261117/cofilite-frame-v1-opset18.identity.json",
            "cb2dce2870eaa6035c5409c3743baf4eef950e9fb2ccc251171b95578792c29a",
        ), (
            ROOT / "build/cofilite-ort-20261117/cofilite-frame-v1-opset18.checker.json",
            "3c6acb110e95a0bd78073dbdb1dc97033476aa804413562bbbe307cfd99ea66b",
        ),),
        frame_executor="onnxruntime_cpu",
        frame_model_path=ROOT / "build/cofilite-ort-20261117/cofilite-frame-v1-opset18.onnx",
        frame_model_sha256="f754898f6721c3180f53b602183db1275c09e08355b8ddff2b0b289181a7a2e0",
    ),
    "dpdfnet2": NSRouteSpec(
        backend="dpdfnet2",
        model_id="dpdfnet2",
        processing_mode="causal_streaming_precomputed",
        model_path=ROOT / "build/ns-next-20261115/dpdfnet/assets/dpdfnet2.onnx",
        model_sha256="4f0ee28935b4a32abecc717d745416976565834d839601acf43031094b4dc94c",
        frontend_path=ROOT / "build/ns-next-20261115/dpdfnet/infer.py",
        frontend_sha256="71d90cfb4c338808573d84e9b9b89d0641cbbf5f1a7b67c829f1543c66fa4a08",
        delay_samples=640,
        offline_whole_utterance=False,
        lookahead_samples=320,
        window_samples=320,
        hop_samples=160,
    ),
    "dpdfnet2_48k_hr": NSRouteSpec(
        backend="dpdfnet2_48k_hr",
        model_id="dpdfnet2_48k_hr",
        processing_mode="causal_streaming_precomputed",
        model_path=ROOT / "build/ns-dpdfnet2-48k-hr-20261118/assets/dpdfnet2_48khz_hr.onnx",
        model_sha256="7f0575a5cec0ba4ffd8f8bd657e06d007e4ccdd955d76faab922b9d3291dc14b",
        frontend_path=ROOT / "server/dpdfnet2_48k_hr.py",
        frontend_sha256="481ac722385edfd99a880200a9cf58cbe21a457d38218634a637841f6b2777be",
        delay_samples=1920,
        offline_whole_utterance=False,
        lookahead_samples=960,
        sample_rate=48_000,
        frame_samples=480,
        window_samples=960,
        hop_samples=480,
        replay_drain_samples=48_000,
        extra_assets=((
            ROOT / "build/ns-dpdfnet2-48k-hr-20261118/upstream/stream.py",
            "0926e33beda8a29a16c40476aead3b7879a0adc7e45b2c10fbdffc3698c23cd6",
        ), (
            ROOT / "build/ns-dpdfnet2-48k-hr-20261118/upstream/infer_dpdfnet_onnx.py",
            "df8d9beb2f937ad9810aa501f022eb3f1c6721da844d3fe741d623498312890a",
        ), (
            ROOT / "build/ns-dpdfnet2-48k-hr-20261118/upstream/dpdfnet_48khz_hr.py",
            "a13d107b7ebd1745c4fdd29849ec52c1eb2a8597627fe1382346bd42c589c991",
        )),
    ),
    "hybrid_v2_pitch": NSRouteSpec(
        backend="hybrid_v2_pitch",
        model_id="hybrid_v2_pitch",
        processing_mode="causal_streaming_precomputed",
        model_path=ROOT / "build/ns-hybrid-v2-20261115/train/newloss320/export/weights.bin",
        model_sha256="662377b8b2c011046226d2b457beb3af45486071fde6ae8273ee37be1fe0ac63",
        frontend_path=ROOT / "build/ns-hybrid-v2-20261115/native/hybrid_v2_probe-fixed",
        frontend_sha256="cca8c9ab5955b2bdde7912203c1c61a1ffdcbd45b124c26c78d952ab7540c152",
        delay_samples=160,
        offline_whole_utterance=False,
        lookahead_samples=0,
        window_samples=512,
        hop_samples=160,
    ),
}


def _selected_ns(configuration: dict | None) -> str | None:
    if configuration is None:
        return None
    if not isinstance(configuration, dict):
        raise ValueError("NS 配置必须是对象")
    selection = configuration.get("selection")
    if not isinstance(selection, dict):
        return None
    selected = selection.get("ns")
    if selected is None:
        return None
    if not isinstance(selected, str):
        raise ValueError("NS selection.ns 必须是字符串")
    return selected


def _offline_48k_replay_requested(configuration: dict | None) -> bool:
    """Return true only for the explicit historical replay diagnostic.

    Production chains do not carry this key and therefore use the native
    48 kHz callback worker.  The opt-out keeps the old capture/replay contract
    available for regression evidence and failure analysis.
    """
    return (
        isinstance(configuration, dict)
        and configuration.get("ns_execution_mode") == "offline_replay"
    )


def ns_backend_for(configuration: dict | None) -> str | None:
    """Return one of the three external NS ids, otherwise ``None``.

    Unknown non-external catalog tokens retain the native C++ path.  A model
    path or command can never be supplied through configuration.
    """
    selected = _selected_ns(configuration)
    return selected if selected in EXTERNAL_NS_IDS else None


def resolve_ns_cli(default_cli: Path | None, configuration: dict | None) -> Path | None:
    """Select the executable owning the requested model and hop contract."""
    backend = ns_backend_for(configuration)
    selection = configuration.get("selection", {}) if isinstance(configuration, dict) else {}
    localvqe = selection.get("aec") == "localvqe_aec200k" if isinstance(selection, dict) else False
    if backend is None and not localvqe:
        return default_cli
    if backend == "dpdfnet2_48k_hr":
        candidate = ROOT / "build/itu-48k-online-20261119-v1/cpp/itu_audio_cli"
    else:
        candidate = ROOT / (
            "build/cofilite-opt-20261116/host-v3/cpp/itu_audio_cli"
            if backend == "cofilite" or localvqe
            else "build/ns-clusters-20261115/host-root/cpp/itu_audio_cli"
        )
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise ValueError("外部 NS 引擎不存在或不可执行，请构建所选模型的固定版本目标")
    return candidate


def ns_route_identity(configuration: dict | None) -> dict[str, Any] | None:
    """Return frozen metadata only for one of the three external NS routes.

    Legacy/native routes return ``None`` so historical batch snapshots do not
    gain a previously absent identity field.
    """
    selected = _selected_ns(configuration)
    backend = selected if selected in EXTERNAL_NS_IDS else None
    if backend is None:
        return None
    spec = NS_ROUTES[backend]
    online_48k = (
        backend == "dpdfnet2_48k_hr"
        and not _offline_48k_replay_requested(configuration)
    )
    processing_mode = "stateful_streaming" if online_48k else spec.processing_mode
    identity = {
        "ns_backend_requested": backend,
        "ns_backend_effective": backend,
        "ns_processing_mode": processing_mode,
        "ns_model_id": spec.model_id,
        "ns_model_sha256": spec.model_sha256,
        "ns_frontend_sha256": spec.frontend_sha256,
        "ns_delay_samples": (
            _DPDFNET2_48K_ONLINE_DELAY_SAMPLES if online_48k else spec.delay_samples
        ),
        "ns_offline_whole_utterance": spec.offline_whole_utterance,
        "ns_lookahead_samples": spec.lookahead_samples,
        "ns_sample_rate": spec.sample_rate,
        "ns_frame_samples": spec.frame_samples,
        "ns_window_samples": spec.window_samples,
        "ns_hop_samples": spec.hop_samples,
        "ns_pcm16_quantization": "once_effective",
    }
    if online_48k:
        identity.update({
            "ns_model_latency_samples": _DPDFNET2_48K_MODEL_LATENCY_SAMPLES,
            "ns_stream_startup_delay_samples": _DPDFNET2_48K_STARTUP_DELAY_SAMPLES,
        })
    if spec.frame_executor is not None:
        identity.update({
            "ns_frame_executor": spec.frame_executor,
            "ns_frame_model_sha256": spec.frame_model_sha256,
            "ns_frame_execution_provider": "CPUExecutionProvider",
        })
    return identity


def _positive_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _zero_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value == 0


def validate_ns_route_report(report: dict, configuration: dict | None) -> list[str]:
    """Validate model identity and the C++ external-PCM execution proof."""
    backend = ns_backend_for(configuration)
    if backend is None:
        return []
    if not isinstance(report, dict):
        return ["NS 报告必须是 JSON 对象"]
    spec = NS_ROUTES[backend]
    errors: list[str] = []
    expected = {
        "ns_backend_requested": backend,
        "ns_backend_effective": backend,
        "ns_model_id": spec.model_id,
        "ns_model_sha256": spec.model_sha256,
        "ns_frontend_sha256": spec.frontend_sha256,
        "ns_delay_samples": spec.delay_samples,
        "ns_sample_rate": spec.sample_rate,
        "ns_frame_samples": spec.frame_samples,
        "ns_window_samples": spec.window_samples,
        "ns_hop_samples": spec.hop_samples,
    }
    online_48k = (
        backend == "dpdfnet2_48k_hr"
        and report.get("ns_external_stream_enabled") is True
        and report.get("ns_external_pcm_enabled") is False
    )
    if not online_48k:
        expected["ns_processing_mode"] = spec.processing_mode
    else:
        expected.update({
            "ns_delay_samples": _DPDFNET2_48K_ONLINE_DELAY_SAMPLES,
            "ns_model_latency_samples": _DPDFNET2_48K_MODEL_LATENCY_SAMPLES,
            "ns_stream_startup_delay_samples": _DPDFNET2_48K_STARTUP_DELAY_SAMPLES,
        })
    for field, value in expected.items():
        if report.get(field) != value:
            errors.append(f"{field} 必须是 {value!r}（实际为 {report.get(field)!r}）")
    if spec.frame_executor is not None:
        for field, value in {
            "ns_frame_executor": spec.frame_executor,
            "ns_frame_model_sha256": spec.frame_model_sha256,
            "ns_frame_execution_provider": "CPUExecutionProvider",
        }.items():
            if report.get(field) != value:
                errors.append(f"{field} 必须是 {value!r}（实际为 {report.get(field)!r}）")
    actual_backend = report.get("ns_backend")
    allowed_backends = {
        backend,
        "external_stream" if backend == "cofilite" or online_48k else "external_pcm",
    }
    if actual_backend not in allowed_backends:
        errors.append(f"ns_backend 必须属于 {sorted(allowed_backends)!r}（实际为 {actual_backend!r}）")
    for field in ("ns_boundary_input_sha256", "ns_output_sha256"):
        value = report.get(field)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            errors.append(f"{field} 必须是 64 位小写十六进制 SHA-256")
    if not _positive_int(report.get("ns_inference_calls")):
        errors.append("ns_inference_calls 必须是正整数")
    if backend == "cofilite" or online_48k:
        required_stream = {
            "ns_external_stream_enabled": True, "ns_external_pcm_enabled": False,
            "ns_offline_whole_utterance": False, "ns_lookahead_samples": spec.lookahead_samples,
            "ns_precomputed_replay": False, "ns_stateful": True,
            "ns_stream_chunk_samples": spec.frame_samples,
            "ns_execution_mode": "online_callback", "ns_worker_transport": "persistent_f32le_v1",
            "ns_state_resets": 1, "ns_realtime_verified": False,
        }
        if online_48k:
            required_stream.update({
                "ns_processing_mode": "stateful_streaming",
                "ns_stream_frame_samples": spec.frame_samples,
                "ns_stream_model_latency_samples": _DPDFNET2_48K_MODEL_LATENCY_SAMPLES,
                "ns_stream_startup_delay_samples": _DPDFNET2_48K_STARTUP_DELAY_SAMPLES,
            })
        for key, expected_value in required_stream.items():
            actual_value = report.get(key)
            if type(actual_value) is not type(expected_value) or actual_value != expected_value:
                errors.append(f"{key} 必须是 {expected_value!r}")
        if report.get("ns_stream_callback_frames") != report.get("ns_external_applied_frames"):
            errors.append("ns_stream_callback_frames 与实际 native callback 数量不一致")
    elif report.get("ns_external_pcm_enabled") is not True:
        errors.append("ns_external_pcm_enabled 必须是 true")
    if not _positive_int(report.get("ns_external_applied_frames")):
        errors.append("ns_external_applied_frames 必须是正整数")
    if not _zero_int(report.get("ns_external_fallback_frames")):
        errors.append("ns_external_fallback_frames 必须是整数 0")
    expected_external_delay = (
        _DPDFNET2_48K_ONLINE_DELAY_SAMPLES if online_48k else spec.delay_samples
    )
    if report.get("ns_external_delay_samples") != expected_external_delay:
        errors.append(
            f"ns_external_delay_samples 必须是 {expected_external_delay}"
            f"（实际为 {report.get('ns_external_delay_samples')!r}）"
        )
    return errors


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_route_assets(backend: str) -> None:
    spec = NS_ROUTES[backend]
    checks = [(spec.model_path, spec.model_sha256), (spec.frontend_path, spec.frontend_sha256)]
    checks.extend(spec.extra_assets)
    if spec.frame_executor is not None:
        if (spec.frame_executor != "onnxruntime_cpu" or spec.frame_model_path is None
                or not isinstance(spec.frame_model_sha256, str)
                or _SHA256.fullmatch(spec.frame_model_sha256) is None):
            raise ValueError("NS 帧模型执行身份缺失或非法")
        checks.append((spec.frame_model_path, spec.frame_model_sha256))
    for path, expected in checks:
        if not path.is_file():
            raise ValueError(f"NS 模型文件不存在：{path}")
        actual = _sha256_file(path)
        if actual != expected:
            raise ValueError(f"NS 模型 SHA-256 不匹配：{path}（{actual}）")


def _command_values(command: Sequence[str], flag: str) -> list[str]:
    values: list[str] = []
    prefix = flag + "="
    for index, value in enumerate(command):
        if value == flag:
            if index + 1 >= len(command):
                raise ValueError(f"{flag} 缺少参数")
            values.append(str(command[index + 1]))
        elif value.startswith(prefix):
            values.append(value[len(prefix):])
    return values


def _has_flag(command: Sequence[str], flag: str) -> bool:
    return any(value == flag or value.startswith(flag + "=") for value in command)


def _external_command(command: Sequence[str]) -> tuple[Path, list[str]] | None:
    items = list(command)
    if not _command_values(items, "--mic"):
        return None
    if len(_command_values(items, "--out")) != 1:
        raise ValueError("外部 NS 命令必须且只能提供一次 --out prefix")
    if _has_flag(items, "--post-ns-pcm") or _has_flag(items, "--post-ns-delay-samples"):
        raise ValueError("外部 NS 命令不能预置 post-NS replay 参数")
    if any(_has_flag(items, flag) for flag in (
        "--ns-stream-input-fd", "--ns-stream-output-fd", "--ns-stream-delay-samples",
    )):
        raise ValueError("外部 NS 命令不能预置 streaming file descriptors")
    profiles = _command_values(items, "--profile")
    if len(profiles) != 1 or profiles[0] not in {"vcp", "vcp_adaptive", "vcp_mild"}:
        raise ValueError("外部 NS 只允许原 VCP profile，不能切换到 off-profile 覆盖 AEC")
    ns_backends = _command_values(items, "--ns-backend")
    if ns_backends != ["webrtc"]:
        raise ValueError("外部 NS boundary 必须保留唯一的 --ns-backend webrtc")
    prefix = Path(_command_values(items, "--out")[0])
    if not prefix.is_absolute():
        raise ValueError("外部 NS --out prefix 必须是绝对路径")
    return prefix, items


def _validate_mic_header(command: Sequence[str], sample_rate: int = _SAMPLE_RATE) -> None:
    mic_values = _command_values(command, "--mic")
    if len(mic_values) != 1:
        raise ValueError("外部 NS TX 命令必须且只能提供一个 --mic WAV")
    mic_path = Path(mic_values[0])
    if not mic_path.is_file():
        raise ValueError(f"外部 NS mic WAV 不存在：{mic_path}")
    info = sf.info(mic_path)
    if info.samplerate != sample_rate:
        raise ValueError(
            f"外部 NS mic WAV 必须是原生 {sample_rate} Hz（实际 {info.samplerate} Hz）"
        )
    if info.channels not in {1, 2, 4}:
        raise ValueError(f"外部 NS mic WAV 声道数不支持：{info.channels}")


def _artifact(prefix: Path, suffix: str) -> Path:
    return Path(str(prefix) + suffix)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _read_stage(path: Path, sample_rate: int = _SAMPLE_RATE) -> tuple[dict[str, Any], Path]:
    if not path.is_file():
        raise ValueError(f"NS diagnostic stages 不存在：{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    stages = payload.get("pipeline_stages")
    if not isinstance(stages, list):
        raise ValueError("NS diagnostic stages 缺少 pipeline_stages")
    matches = [item for item in stages if isinstance(item, dict) and item.get("id") == "tx_ns_input"]
    if len(matches) != 1:
        raise ValueError("诊断阶段必须恰好包含一个 tx_ns_input")
    stage = matches[0]
    if (stage.get("direction") != "tx"
            or stage.get("sample_rate") != sample_rate
            or stage.get("channels") != 1):
        raise ValueError(f"tx_ns_input 必须是 {sample_rate} Hz mono TX FLOAT tap")
    file_value = stage.get("file")
    if not isinstance(file_value, str) or not file_value:
        raise ValueError("tx_ns_input 缺少 file")
    return stage, Path(file_value)


def _read_float_mono(path: Path, sample_rate: int = _SAMPLE_RATE) -> np.ndarray:
    if not path.is_file():
        raise ValueError(f"NS FLOAT tap 不存在：{path}")
    info = sf.info(path)
    if info.samplerate != sample_rate or info.channels != 1 or info.subtype != "FLOAT":
        raise ValueError(
            f"NS tap 必须是 {sample_rate} Hz mono IEEE FLOAT32 WAV：{path}"
        )
    values, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != sample_rate or values.shape[1] != 1 or not len(values):
        raise ValueError(f"NS tap 音频几何无效：{path}")
    samples = np.ascontiguousarray(values[:, 0], dtype=np.float32)
    if not np.isfinite(samples).all():
        raise ValueError(f"NS tap 含 NaN/Inf：{path}")
    return samples


def _read_float_mono_16k(path: Path) -> np.ndarray:
    return _read_float_mono(path, _SAMPLE_RATE)


def _write_float_mono(path: Path, samples: np.ndarray,
                      sample_rate: int = _SAMPLE_RATE) -> None:
    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("拒绝写入空或非有限的 NS replay")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, values, sample_rate, subtype="FLOAT", format="WAV")


def _write_float_mono_16k(path: Path, samples: np.ndarray) -> None:
    _write_float_mono(path, samples, _SAMPLE_RATE)


def _quantize_pcm16_lround(samples: np.ndarray) -> tuple[np.ndarray, bytes]:
    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    scaled = np.clip(values.astype(np.float64) * 32768.0, -32768.0, 32767.0)
    integers = np.copysign(np.floor(np.abs(scaled) + 0.5), scaled).astype(np.int16)
    quantized = (integers.astype(np.float32) / np.float32(32768.0)).astype(np.float32)
    return np.ascontiguousarray(quantized), integers.astype("<i2", copy=False).tobytes()


def _float_payload_sha256(samples: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(samples, dtype="<f4").tobytes()).hexdigest()


def _remaining(deadline: float, command: Sequence[str], timeout: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise subprocess.TimeoutExpired(list(command), timeout, output="NS route timeout budget exhausted")
    return remaining


def _run_timed(command: Sequence[str], deadline: float, timeout: float, kwargs: dict[str, Any]):
    return subprocess.run(command, timeout=_remaining(deadline, command, timeout), **kwargs)


def _process_error(label: str, completed: subprocess.CompletedProcess) -> ValueError:
    detail = completed.stderr or completed.stdout or ""
    if isinstance(detail, bytes):
        detail = detail.decode(errors="replace")
    lines = str(detail).strip().splitlines()
    suffix = f"：{lines[-1]}" if lines else ""
    return ValueError(f"{label}失败（退出码 {completed.returncode}）{suffix}")


def _model_python() -> str:
    override = os.environ.get("ITU_NS_PYTHON")
    if override:
        return override
    default = Path("/home/zhangqingtian/.venv/bin/python")
    return str(default if default.is_file() else Path(sys.executable))


def _run_model_worker(backend: str, input_path: Path, output_path: Path,
                      receipt_path: Path, deadline: float, timeout: float) -> dict[str, Any]:
    command = [
        _model_python(), str(MODEL_WORKER), "--backend", backend,
        "--input", str(input_path), "--output", str(output_path),
        "--receipt", str(receipt_path),
    ]
    env = dict(os.environ)
    env.update({
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "TORCH_NUM_THREADS": "1", "ORT_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
    })
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=_remaining(deadline, command, timeout),
        env=env, cwd=ROOT, check=False,
    )
    if completed.returncode != 0:
        raise _process_error("NS 模型进程", completed)
    if not receipt_path.is_file():
        raise ValueError("NS 模型进程未生成 receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or receipt.get("backend") != backend:
        raise ValueError("NS 模型 receipt backend 不一致")
    spec = NS_ROUTES[backend]
    expected = {
        "requested": backend,
        "effective": backend,
        "model_id": spec.model_id,
        "model_sha256": spec.model_sha256,
        "frontend_sha256": spec.frontend_sha256,
        "processing_mode": spec.processing_mode,
        "sample_rate": spec.sample_rate,
        "frame_samples": spec.frame_samples,
        "window_samples": spec.window_samples,
        "hop_samples": spec.hop_samples,
        "delay_samples": spec.delay_samples,
        "offline_whole_utterance": spec.offline_whole_utterance,
        "lookahead_samples": spec.lookahead_samples,
    }
    for field, expected_value in expected.items():
        if receipt.get(field) != expected_value:
            raise ValueError(
                f"NS 模型 receipt {field} 与路由不一致："
                f"{receipt.get(field)!r} != {expected_value!r}"
            )
    input_info = receipt.get("input")
    aligned_info = receipt.get("aligned_output")
    if not isinstance(input_info, dict) or not isinstance(aligned_info, dict):
        raise ValueError("NS 模型 receipt 缺少 input/aligned_output")
    if input_info.get("sample_rate") != spec.sample_rate:
        raise ValueError("NS 模型 receipt input sample_rate 与路由不一致")
    if aligned_info.get("sample_rate") != spec.sample_rate:
        raise ValueError("NS 模型 receipt aligned sample_rate 与路由不一致")
    if aligned_info.get("samples") != input_info.get("samples"):
        raise ValueError("NS 模型 receipt aligned/input 样本数不一致")
    if aligned_info.get("finite") is not True:
        raise ValueError("NS 模型 receipt aligned output 未证明 finite")
    if backend == "dpdfnet2_48k_hr":
        details = receipt.get("details")
        if not isinstance(details, dict):
            raise ValueError("DPDFNet2 48 kHz HR receipt 缺少 details")
        if details.get("algorithm_latency_samples") != spec.delay_samples:
            raise ValueError("DPDFNet2 48 kHz HR receipt 算法延迟不一致")
        if int(details.get("frames_processed") or 0) <= 0:
            raise ValueError("DPDFNet2 48 kHz HR receipt 未证明 ONNX 推理")
    return receipt


def _same_float_payload(left: np.ndarray, right: np.ndarray) -> bool:
    return left.shape == right.shape and left.astype("<f4", copy=False).tobytes() == right.astype("<f4", copy=False).tobytes()


def _validate_boundary_identity(boundary: dict, final: dict) -> None:
    # Delay-bearing external replay legitimately extends the AEC drain, so
    # inference/processed counters are not stable identities.  Bind the route
    # identity, model/device assets, input geometry, and processing mode.
    keys = {
        "sample_rate", "microphones", "reference_channels", "input_samples_per_channel",
        "aec_backend", "aec_backend_requested", "aec_backend_effective",
        "aec_execution_provider", "aec_device_id", "aec_model_id",
        "aec_model_sha256", "aec_frontend_sha256", "aec_provider_sha256",
        "aec_neural_delay_samples", "aec_model_codec_delay_samples_16k",
        "aec_linear_frontend_delay_samples_16k", "aec_processing_mode",
        "aec_hop_adapter_delay_samples_16k", "aec_algorithm_delay_samples_16k",
    }
    changed = sorted(key for key in keys if boundary.get(key) != final.get(key))
    if changed:
        raise ValueError("boundary/final AEC 或输入身份不一致：" + ", ".join(changed))


def _validate_external_cpp(report: dict, spec: NSRouteSpec) -> None:
    if report.get("ns_backend") != "external_pcm":
        raise ValueError("final C++ report 未证明 ns_backend=external_pcm")
    if report.get("ns_external_pcm_enabled") is not True:
        raise ValueError("final C++ report 未启用 external PCM")
    if not _positive_int(report.get("ns_external_applied_frames")):
        raise ValueError("final C++ report 未证明 actual external applied frames > 0")
    if not _zero_int(report.get("ns_external_fallback_frames")):
        raise ValueError("final C++ report 出现 external NS fallback frames")
    if report.get("ns_external_delay_samples") != spec.delay_samples:
        raise ValueError("final C++ external NS delay 与模型固定延迟不一致")
    if report.get("sample_rate") != spec.sample_rate:
        raise ValueError("final C++ sample rate 与外部 NS 路由不一致")
    if report.get("frame_size") != spec.frame_samples:
        raise ValueError("final C++ frame size 与外部 NS 路由不一致")
    if (spec.backend == "dpdfnet2_48k_hr"
            and report.get("aec_backend_effective") != "webrtc_aec3"):
        raise ValueError("DPDFNet2 48 kHz HR 必须由 AEC3 上游执行")


def _append_stdout(completed: subprocess.CompletedProcess, marker: str) -> subprocess.CompletedProcess:
    stdout = completed.stdout
    if isinstance(stdout, bytes):
        stdout = stdout + (b"\n" if stdout and not stdout.endswith(b"\n") else b"") + marker.encode() + b"\n"
    elif isinstance(stdout, str):
        stdout = stdout + ("\n" if stdout and not stdout.endswith("\n") else "") + marker + "\n"
    return subprocess.CompletedProcess(completed.args, completed.returncode, stdout, completed.stderr)


def _publish_runtime_receipt(prefix: Path, backend: str, payload: dict[str, Any]) -> None:
    try:
        resolved = prefix.resolve()
        if not resolved.is_relative_to((ROOT / "build").resolve()):
            return
    except OSError:
        return
    name = f"{backend}-{hashlib.sha256(str(prefix).encode()).hexdigest()[:16]}.json"
    _write_json(RUNTIME_RECEIPTS / name, payload)


def execute_ns_command(
    command: Sequence[str], *, configuration: dict | None, timeout: float = 300,
    **subprocess_kwargs: Any,
) -> subprocess.CompletedProcess:
    """Execute the original CLI, inserting an allowlisted external NS when selected.

    Native and RX-only commands are passed to ``subprocess.run`` unchanged.
    """
    backend = ns_backend_for(configuration)
    if backend is None:
        return subprocess.run(command, timeout=timeout, **subprocess_kwargs)
    external = _external_command(command)
    if external is None:
        return subprocess.run(command, timeout=timeout, **subprocess_kwargs)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("NS 路由 timeout 必须是正的有限秒数")

    if backend == "cofilite":
        from server.ns_streaming import execute_stream_command
        return execute_stream_command(command, configuration=configuration, timeout=timeout, **subprocess_kwargs)
    if backend == "dpdfnet2_48k_hr" and not _offline_48k_replay_requested(configuration):
        from server.ns_streaming_48k_hr import execute_stream_command
        return execute_stream_command(
            command,
            configuration=configuration,
            timeout=timeout,
            **subprocess_kwargs,
        )

    prefix, original = external
    prefix.parent.mkdir(parents=True, exist_ok=True)
    model_dir = _artifact(prefix, ".ns-model")
    model_dir.mkdir(parents=True, exist_ok=True)
    spec = NS_ROUTES[backend]
    deadline = time.monotonic() + float(timeout)
    started = time.monotonic()
    try:
        if backend == "dpdfnet2_48k_hr":
            aec_backends = _command_values(original, "--aec-backend")
            if _has_flag(original, "--no-aec") or aec_backends != ["webrtc"]:
                raise ValueError(
                    "DPDFNet2 48 kHz HR 质量路线必须显式使用 --aec-backend webrtc"
                )
        _validate_mic_header(original, spec.sample_rate)
        cli_path = Path(original[0]).resolve()
        cli_sha256 = _sha256_file(cli_path)
        _verify_route_assets(backend)
        boundary_command = list(original)
        if not _has_flag(boundary_command, "--diagnostic-stages"):
            boundary_command.append("--diagnostic-stages")
        boundary = _run_timed(boundary_command, deadline, float(timeout), subprocess_kwargs)
        if boundary.returncode != 0:
            raise _process_error("NS boundary capture", boundary)

        boundary_stages_path = _artifact(prefix, ".stages.json")
        boundary_report_path = _artifact(prefix, ".json")
        boundary_stage, boundary_tap_path = _read_stage(
            boundary_stages_path, spec.sample_rate
        )
        boundary_samples = _read_float_mono(boundary_tap_path, spec.sample_rate)
        expected_stage_samples = boundary_stage.get("samples_per_channel")
        if expected_stage_samples != int(boundary_samples.size):
            raise ValueError("tx_ns_input metadata samples 与实际 WAV 不一致")

        boundary_copy = model_dir / "boundary.tx_ns_input.float.wav"
        shutil.copy2(boundary_tap_path, boundary_copy)
        shutil.copy2(boundary_stages_path, model_dir / "boundary.stages.json")
        if boundary_report_path.is_file():
            shutil.copy2(boundary_report_path, model_dir / "boundary.json")

        quantized, quantized_pcm16 = _quantize_pcm16_lround(boundary_samples)
        model_input = model_dir / "input.q.float.wav"
        _write_float_mono(model_input, quantized, spec.sample_rate)
        aligned_path = model_dir / "aligned.float.wav"
        receipt_path = model_dir / "receipt.json"
        receipt = _run_model_worker(backend, model_input, aligned_path, receipt_path, deadline, float(timeout))
        aligned = _read_float_mono(aligned_path, spec.sample_rate)
        if aligned.size != boundary_samples.size:
            raise ValueError(f"NS aligned output 长度错误：{aligned.size} != {boundary_samples.size}")
        if receipt.get("aligned_output", {}).get("payload_sha256") != _float_payload_sha256(aligned):
            raise ValueError("NS aligned payload SHA 与 receipt 不一致")

        delay = spec.delay_samples
        replay_core = np.concatenate((np.zeros(delay, dtype=np.float32), aligned)).astype(np.float32)
        padded = int(
            math.ceil(replay_core.size / spec.frame_samples) * spec.frame_samples
        )
        replay = np.concatenate((
            replay_core,
            np.zeros(
                padded - replay_core.size + spec.replay_drain_samples,
                dtype=np.float32,
            ),
        ))
        replay_path = model_dir / "replay.float.wav"
        _write_float_mono(replay_path, replay, spec.sample_rate)
        replay_core = replay[:boundary_samples.size + delay]
        output_sha = _float_payload_sha256(replay_core)
        boundary_file_sha = _sha256_file(boundary_copy)

        final_command = list(original)
        if not _has_flag(final_command, "--diagnostic-stages"):
            final_command.append("--diagnostic-stages")
        final_command.extend(["--post-ns-pcm", str(replay_path), "--post-ns-delay-samples", str(delay)])
        final = _run_timed(final_command, deadline, float(timeout), subprocess_kwargs)
        if final.returncode != 0:
            raise _process_error("NS final replay", final)

        final_stages_path = _artifact(prefix, ".stages.json")
        final_report_path = _artifact(prefix, ".json")
        final_input_stage, final_input_path = _read_stage(
            final_stages_path, spec.sample_rate
        )
        final_input = _read_float_mono(final_input_path, spec.sample_rate)
        if not _same_float_payload(boundary_samples, final_input):
            raise ValueError("boundary/final tx_ns_input FLOAT payload 不精确一致")

        final_payload = json.loads(final_stages_path.read_text(encoding="utf-8"))
        tx_ns_stage = next((item for item in final_payload.get("pipeline_stages", [])
                            if isinstance(item, dict) and item.get("id") == "tx_ns"), None)
        if not tx_ns_stage or not isinstance(tx_ns_stage.get("file"), str):
            raise ValueError("final diagnostic stages 缺少 tx_ns tap")
        downstream = _read_float_mono(Path(tx_ns_stage["file"]), spec.sample_rate)
        if downstream.size != boundary_samples.size + delay:
            raise ValueError("final tx_ns 长度不是 boundary N + NS delay")
        if not _same_float_payload(downstream, replay_core[:downstream.size]):
            raise ValueError("下游 tx_ns PCM 不是 physical model replay 的精确前缀")

        if not final_report_path.is_file():
            raise ValueError("final C++ report 不存在")
        final_report = json.loads(final_report_path.read_text(encoding="utf-8"))
        boundary_report = {}
        preserved_boundary_report = model_dir / "boundary.json"
        if preserved_boundary_report.is_file():
            boundary_report = json.loads(preserved_boundary_report.read_text(encoding="utf-8"))
        if boundary_report:
            _validate_boundary_identity(boundary_report, final_report)
        _validate_external_cpp(final_report, spec)
        _verify_route_assets(backend)
        if _sha256_file(cli_path) != cli_sha256:
            raise ValueError("NS capture/replay 期间 CLI 文件身份发生变化")

        identity = ns_route_identity(configuration)
        if identity is None:
            raise ValueError("外部 NS route identity 意外为空")
        execution = {
            **identity,
            "ns_cli_sha256": cli_sha256,
            "ns_external_backend": "external_pcm",
            "ns_boundary_input_samples": int(boundary_samples.size),
            "ns_boundary_fixed_delay_samples": int(final_input_stage.get("fixed_delay_samples") or 0),
            "ns_replay_core_samples": int(replay_core.size),
            "ns_replay_file_samples": int(replay.size),
            "ns_replay_padding_samples": int(padded - replay_core.size),
            "ns_replay_drain_samples": spec.replay_drain_samples,
            "ns_aligned_output_samples": int(aligned.size),
            "ns_aligned_output_payload_sha256": _float_payload_sha256(aligned),
            "ns_boundary_input_sha256": boundary_file_sha,
            "ns_output_sha256": output_sha,
            "ns_input_payload_sha256": hashlib.sha256(quantized_pcm16).hexdigest(),
            "ns_model_worker_invocations": 1,
            "ns_model_onnx_inferences": int(
                receipt.get("inference", {}).get("onnx_inferences") or 1
            ),
            "ns_model_inference_calls": 1,
            "ns_model_inference_s": receipt.get("inference", {}).get("elapsed_sec"),
            "ns_pcm16_quantization": "once_effective",
        }
        final_report.update(execution)
        final_report.update({
            "ns_backend": backend,
            "ns_backend_requested": backend,
            "ns_backend_effective": backend,
            "ns_model_id": spec.model_id,
            "ns_model_sha256": spec.model_sha256,
            "ns_frontend_sha256": spec.frontend_sha256,
            "ns_processing_mode": spec.processing_mode,
            "ns_delay_samples": delay,
            "ns_boundary_input_sha256": boundary_file_sha,
            "ns_output_sha256": output_sha,
            "ns_inference_calls": 1,
            "ns_model_inference_s": receipt.get("inference", {}).get("elapsed_sec"),
            "ns_external_backend": "external_pcm",
            "ns_execution": execution,
        })
        errors = validate_ns_route_report(final_report, configuration)
        if errors:
            raise ValueError("NS route report 校验失败：" + "；".join(errors))
        _write_json(final_report_path, final_report)

        final_payload["ns_model"] = {key: execution[key] for key in (
            "ns_backend_effective", "ns_model_id", "ns_model_sha256", "ns_frontend_sha256",
            "ns_processing_mode", "ns_delay_samples", "ns_boundary_input_sha256", "ns_output_sha256",
        )}
        tx_ns_stage["channel_policy"] = "mono external model NS output"
        tx_ns_stage["measurement_domain"] = "selected model output; before EQ, AGC, DRC and limiter"
        _write_json(final_stages_path, final_payload)

        summary = {
            "schema_version": 1,
            "backend": backend,
            "identity": identity,
            "execution": execution,
            "worker_receipt": receipt,
            "artifacts": {
                "boundary_stages": str(model_dir / "boundary.stages.json"),
                "boundary_input": str(boundary_copy),
                "model_input": str(model_input),
                "aligned_output": str(aligned_path),
                "physical_replay": str(replay_path),
                "worker_receipt": str(receipt_path),
            },
            "elapsed_sec": time.monotonic() - started,
        }
        _write_json(_artifact(prefix, ".ns.json"), summary)
        _write_json(model_dir / "execution.json", summary)
        _publish_runtime_receipt(prefix, backend, summary)
        marker = (
            f"NS model: backend={backend} model_sha256={spec.model_sha256} "
            f"delay={delay} inference_calls=1 fallback=0"
        )
        return _append_stdout(final, marker)
    except Exception as error:
        try:
            _write_json(model_dir / "failure.json", {
                "schema_version": 1, "backend": backend, "error": {
                    "type": type(error).__name__, "message": str(error),
                },
            })
        except Exception:
            pass
        raise


# Native16 registration: additive dispatch; original source prefix remains frozen.

_native16_original_ns_backend_for = ns_backend_for
_native16_original_resolve_ns_cli = resolve_ns_cli
_native16_original_ns_route_identity = ns_route_identity
_native16_original_validate_ns_route_report = validate_ns_route_report
_native16_original_execute_ns_command = execute_ns_command

def _is_native16_configuration(configuration):
    return isinstance(configuration, dict) and isinstance(configuration.get("selection"), dict) and configuration["selection"].get("ns") == "cofilite_native16"

def ns_backend_for(configuration):
    if _is_native16_configuration(configuration):
        from server.ns_routes_native16 import ns_backend_for as native
        return native(configuration)
    return _native16_original_ns_backend_for(configuration)

def resolve_ns_cli(default_cli, configuration):
    if _is_native16_configuration(configuration):
        from server.ns_routes_native16 import resolve_ns_cli as native
        return native(default_cli, configuration)
    return _native16_original_resolve_ns_cli(default_cli, configuration)

def ns_route_identity(configuration):
    if _is_native16_configuration(configuration):
        from server.ns_routes_native16 import ns_route_identity as native
        return native(configuration)
    return _native16_original_ns_route_identity(configuration)

def validate_ns_route_report(report, configuration):
    if _is_native16_configuration(configuration):
        from server.ns_routes_native16 import validate_ns_route_report as native
        return native(report, configuration)
    return _native16_original_validate_ns_route_report(report, configuration)

def execute_ns_command(command, *, configuration, timeout=300, **kwargs):
    if _is_native16_configuration(configuration):
        from server.ns_routes_native16 import execute_ns_command as native
        return native(command, configuration=configuration, timeout=timeout, **kwargs)
    return _native16_original_execute_ns_command(command, configuration=configuration, timeout=timeout, **kwargs)
