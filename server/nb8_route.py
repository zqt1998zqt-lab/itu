"""Native 8 kHz VCP route: Speex MDF followed by DPDFNet-2 8 kHz.

This route intentionally has no resampling and no algorithm fallback.  The
Speex stage owns the far-end-aware linear echo cancellation; DPDFNet owns the
post-AEC noise suppression.  The output remains on the original 8 kHz PCM
timeline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any

import numpy as np
import soundfile as sf

from server import dpdfnet8


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEEX_CLI = ROOT / "build/host/cpp/itu_nb8_speex"
SAMPLE_RATE = 8_000
FRAME_SAMPLES = 80
CLUSTER_ID = "speex_dpdfnet2_8khz"
SCHEMA = "itu.nb8-vcp.report.v1"
RX_SCHEMA = "itu.nb8-rx.report.v1"


class NB8RouteError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _resolve_speex_cli(path: str | Path | None) -> Path:
    override = path or os.environ.get("ITU_NB8_SPEEX_CLI")
    candidate = Path(override).expanduser().resolve() if override else DEFAULT_SPEEX_CLI
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise NB8RouteError(f"8 kHz Speex MDF CLI 不存在或不可执行：{candidate}")
    return candidate


def _validate_input(path: Path, *, role: str) -> sf._SoundFileInfo:
    if not path.is_file():
        raise NB8RouteError(f"{role} WAV 不存在：{path}")
    info = sf.info(path)
    if info.format != "WAV":
        raise NB8RouteError(f"{role} 必须是 WAV：{path}")
    if info.samplerate != SAMPLE_RATE:
        raise NB8RouteError(
            f"{role} 必须是原生 8 kHz，实际为 {info.samplerate} Hz：{path}"
        )
    if info.channels != 1:
        raise NB8RouteError(f"{role} 必须是 mono，实际为 {info.channels} 声道：{path}")
    if info.frames <= 0:
        raise NB8RouteError(f"{role} WAV 不能为空：{path}")
    return info


def _run_speex(
    cli: Path,
    microphone: Path,
    reference: Path,
    prefix: Path,
    *,
    tail_ms: int,
    timeout: float,
) -> tuple[Path, dict[str, Any], list[str]]:
    command = [
        str(cli),
        "--mic",
        str(microphone),
        "--echo",
        str(reference),
        "--out",
        str(prefix),
        "--tail-ms",
        str(tail_ms),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        cwd=ROOT,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()
        raise NB8RouteError(
            "8 kHz Speex MDF 失败："
            + (detail[-1] if detail else f"退出码 {completed.returncode}")
        )
    output = Path(str(prefix) + ".wav")
    report_path = Path(str(prefix) + ".json")
    if not output.is_file() or not report_path.is_file():
        raise NB8RouteError("8 kHz Speex MDF 未生成完整输出或报告")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("schema") != "itu.nb8_speex.report.v1"
        or report.get("aec_backend") != "speex_mdf_nb8"
        or report.get("sample_rate") != SAMPLE_RATE
        or report.get("channels") != 1
        or report.get("frame_samples") != FRAME_SAMPLES
    ):
        raise NB8RouteError("8 kHz Speex MDF 报告的采样率或帧长不一致")
    if report.get("no_resample") is not True or report.get("fallback_frames") != 0:
        raise NB8RouteError("8 kHz Speex MDF 未证明无重采样、无 fallback")
    microphone_info = sf.info(microphone)
    reference_info = sf.info(reference)
    output_info = sf.info(output)
    expected_samples = int(microphone_info.frames)
    if not (
        int(reference_info.frames) == expected_samples
        and report.get("input_samples") == expected_samples
        and report.get("mic_input_samples") == expected_samples
        and report.get("echo_input_samples") == expected_samples
        and report.get("output_samples") == expected_samples
        and output_info.frames == expected_samples
        and output_info.samplerate == SAMPLE_RATE
        and output_info.channels == 1
    ):
        raise NB8RouteError("8 kHz Speex MDF 输入/输出样本数或格式不一致")
    if (
        report.get("input_sha256", {}).get("mic") != _sha256(microphone)
        or report.get("input_sha256", {}).get("echo") != _sha256(reference)
        or report.get("output_sha256") != _sha256(output)
    ):
        raise NB8RouteError("8 kHz Speex MDF 报告哈希与输入/输出文件不一致")
    samples, sample_rate = sf.read(output, dtype="float64", always_2d=True)
    if (
        sample_rate != SAMPLE_RATE
        or samples.shape != (expected_samples, 1)
        or not np.isfinite(samples).all()
    ):
        raise NB8RouteError("8 kHz Speex MDF 输出不是有限的 8 kHz mono 同长音频")
    return output, report, command


def _refuse_existing(paths: tuple[Path, ...]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise NB8RouteError(
            "拒绝覆盖既有 8 kHz 证据文件：" + ", ".join(existing)
        )


def run_receive(
    far_end_path: str | Path,
    out_prefix: str | Path,
    *,
    model_path: str | Path = dpdfnet8.DEFAULT_MODEL_PATH,
    runtime: dpdfnet8.RuntimeModel | None = None,
) -> dict[str, Any]:
    """Run the native 8 kHz receive branch.

    Receive-only processing has no microphone and no echo reference, so AEC is
    not applicable.  The receive path therefore owns DPDFNet-2 noise
    suppression while preserving the original 8 kHz mono timeline.
    """
    far_end = Path(far_end_path).expanduser().resolve()
    prefix = Path(out_prefix).expanduser().resolve()
    far_end_info = _validate_input(far_end, role="远端播放信号 / 回声参考")
    output = Path(str(prefix) + ".rx-spk.wav")
    report_path = Path(str(prefix) + ".json")
    _refuse_existing((output, report_path))
    prefix.parent.mkdir(parents=True, exist_ok=True)

    selected_runtime = (
        dpdfnet8.load_runtime(model_path) if runtime is None else runtime
    )
    if not isinstance(selected_runtime, dpdfnet8.RuntimeModel):
        raise TypeError("runtime must be a dpdfnet8.RuntimeModel or None")
    started = time.perf_counter()
    ns_info = dpdfnet8.enhance_file(
        far_end,
        output,
        runtime=selected_runtime,
        chunk_size=FRAME_SAMPLES,
    )
    elapsed = time.perf_counter() - started
    output_info = sf.info(output)
    if (
        output_info.format != "WAV"
        or output_info.samplerate != SAMPLE_RATE
        or output_info.channels != 1
        or output_info.frames != far_end_info.frames
    ):
        raise NB8RouteError("8 kHz RX 输出未保持原生 8 kHz mono 同长契约")

    report = {
        "schema": RX_SCHEMA,
        "schema_version": 1,
        "cluster_id": CLUSTER_ID,
        "direction": "receive",
        "sample_rate": SAMPLE_RATE,
        "frame_samples": FRAME_SAMPLES,
        "input_samples": int(far_end_info.frames),
        "output_samples": int(output_info.frames),
        "no_resample": True,
        "resampling": "none",
        "fallback_frames": 0,
        "aec_applicability": "not_applicable_receive_only",
        "algorithm_delay_samples": int(dpdfnet8.ALGORITHM_LATENCY_SAMPLES),
        "elapsed_sec": round(elapsed, 6),
        "far_end": {
            "path": str(far_end),
            "sha256": _sha256(far_end),
            "samples": int(far_end_info.frames),
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
        },
        "stages": {
            "dpdfnet2_8khz": {
                "backend": "dpdfnet2_8khz",
                "sample_rate": SAMPLE_RATE,
                "model_path": str(selected_runtime.model_path),
                "model_sha256": selected_runtime.model_sha256,
                "model_size_bytes": selected_runtime.model_size_bytes,
                "metadata": selected_runtime.metadata,
                "info": ns_info,
            }
        },
        "operations": [
            {
                "kind": "library",
                "backend": "dpdfnet2_8khz",
                "sample_rate": SAMPLE_RATE,
                "model_sha256": selected_runtime.model_sha256,
            }
        ],
        "output": {
            "path": str(output),
            "sha256": _sha256(output),
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
            "samples": int(output_info.frames),
            "subtype": output_info.subtype,
        },
    }
    _write_json(report_path, report)
    return {
        **report,
        "report_path": str(report_path),
        "output_path": str(output),
    }


def run(
    microphone_path: str | Path,
    reference_path: str | Path,
    out_prefix: str | Path,
    *,
    speex_cli: str | Path | None = None,
    model_path: str | Path = dpdfnet8.DEFAULT_MODEL_PATH,
    runtime: dpdfnet8.RuntimeModel | None = None,
    tail_ms: int = 500,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Run the native 8 kHz Speex MDF + DPDFNet-2 route."""
    microphone = Path(microphone_path).expanduser().resolve()
    reference = Path(reference_path).expanduser().resolve()
    prefix = Path(out_prefix).expanduser().resolve()
    microphone_info = _validate_input(microphone, role="麦克风混合输入")
    reference_info = _validate_input(reference, role="远端播放信号 / 回声参考")
    if microphone_info.frames != reference_info.frames:
        raise NB8RouteError(
            "麦克风混合输入与远端播放信号 / 回声参考必须具有相同样本数"
        )
    if isinstance(tail_ms, bool) or not isinstance(tail_ms, int) or not 10 <= tail_ms <= 10_000:
        raise NB8RouteError("tail_ms 必须是 10..10000 的整数")
    if not isinstance(timeout, (int, float)) or not 0 < float(timeout) <= 3600:
        raise NB8RouteError("timeout 必须是 0..3600 秒")

    cli = _resolve_speex_cli(speex_cli)
    speex_output = Path(str(prefix) + ".speex.wav")
    speex_report_path = Path(str(prefix) + ".speex.json")
    after = Path(str(prefix) + ".after.wav")
    report_path = Path(str(prefix) + ".json")
    _refuse_existing(
        (speex_output, speex_report_path, after, report_path)
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    speex_prefix = prefix.with_name(prefix.name + ".speex")
    speex_output, speex_report, speex_command = _run_speex(
        cli,
        microphone,
        reference,
        speex_prefix,
        tail_ms=tail_ms,
        timeout=float(timeout),
    )

    selected_runtime = (
        dpdfnet8.load_runtime(model_path) if runtime is None else runtime
    )
    if not isinstance(selected_runtime, dpdfnet8.RuntimeModel):
        raise TypeError("runtime must be a dpdfnet8.RuntimeModel or None")
    ns_started = time.perf_counter()
    ns_info = dpdfnet8.enhance_file(
        speex_output,
        after,
        runtime=selected_runtime,
        chunk_size=FRAME_SAMPLES,
    )
    ns_elapsed = time.perf_counter() - ns_started
    elapsed = time.perf_counter() - started

    after_info = sf.info(after)
    if (
        after_info.format != "WAV"
        or after_info.samplerate != SAMPLE_RATE
        or after_info.channels != 1
        or after_info.frames != microphone_info.frames
    ):
        raise NB8RouteError("最终输出未保持原生 8 kHz mono 同长契约")

    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "cluster_id": CLUSTER_ID,
        "sample_rate": SAMPLE_RATE,
        "frame_samples": FRAME_SAMPLES,
        "input_samples": int(microphone_info.frames),
        "output_samples": int(after_info.frames),
        "no_resample": True,
        "resampling": "none",
        "fallback_frames": 0,
        "algorithm_delay_samples": int(dpdfnet8.ALGORITHM_LATENCY_SAMPLES),
        "dpdfnet_algorithm_latency_samples": int(
            dpdfnet8.ALGORITHM_LATENCY_SAMPLES
        ),
        "speex_explicit_delay_samples": int(
            speex_report.get("algorithm_delay_samples", 0)
        ),
        "elapsed_sec": round(elapsed, 6),
        "ns_elapsed_sec": round(ns_elapsed, 6),
        "speex_cli": str(cli),
        "speex_cli_sha256": _sha256(cli),
        "microphone": {
            "path": str(microphone),
            "sha256": _sha256(microphone),
            "samples": int(microphone_info.frames),
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
        },
        "reference": {
            "path": str(reference),
            "sha256": _sha256(reference),
            "samples": int(reference_info.frames),
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
        },
        "stages": {
            "speex_mdf": {
                "backend": "speex_mdf_nb8",
                "sample_rate": SAMPLE_RATE,
                "frame_samples": FRAME_SAMPLES,
                "report": speex_report,
            },
            "dpdfnet2_8khz": {
                "backend": "dpdfnet2_8khz",
                "sample_rate": SAMPLE_RATE,
                "model_path": str(selected_runtime.model_path),
                "model_sha256": selected_runtime.model_sha256,
                "model_size_bytes": selected_runtime.model_size_bytes,
                "metadata": selected_runtime.metadata,
                "info": ns_info,
            },
        },
        "commands": [speex_command],
        "operations": [
            {
                "kind": "subprocess",
                "backend": "speex_mdf_nb8",
                "command": speex_command,
            },
            {
                "kind": "library",
                "backend": "dpdfnet2_8khz",
                "sample_rate": SAMPLE_RATE,
                "model_sha256": selected_runtime.model_sha256,
            },
        ],
        "output": {
            "path": str(after),
            "sha256": _sha256(after),
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
            "samples": int(after_info.frames),
            "subtype": after_info.subtype,
        },
    }
    _write_json(report_path, report)
    return {**report, "report_path": str(report_path), "output_path": str(after)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run native 8 kHz Speex MDF + DPDFNet-2")
    parser.add_argument("--mic", type=Path, required=True)
    parser.add_argument("--echo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--speex-cli", type=Path)
    parser.add_argument("--model", type=Path, default=dpdfnet8.DEFAULT_MODEL_PATH)
    parser.add_argument("--tail-ms", type=int, default=500)
    args = parser.parse_args()
    try:
        result = run(
            args.mic,
            args.echo,
            args.out,
            speex_cli=args.speex_cli,
            model_path=args.model,
            tail_ms=args.tail_ms,
        )
    except (NB8RouteError, FileNotFoundError, RuntimeError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
