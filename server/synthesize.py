"""Run the pinned C++ engine on an uploaded microphone/reference pair.

`server.audio.analyze` only measures audio the user already has. This module is
the other half: it feeds a microphone recording and its playback reference
through the same `itu_audio_cli` binary the acceptance runs use, and returns the
processed output plus the engine's own report.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from server.audio import MAX_SECONDS, AudioError, db, decode, power

ROOT = Path(__file__).resolve().parents[1]
# The engine takes its rate from the microphone WAV and only implements 10 ms
# frames at these rates, so both inputs are converted to one of them.
SUPPORTED_RATES = (16000, 24000, 48000)
DEFAULT_RATE = 16000
# The stable path is a symlink refreshed after each cpp build; the pinned build
# directory is the fallback. ITU_CLI overrides both.
CANDIDATE_BINARIES = (
    "build/itu_audio_cli",
    "build/rtd-low-delay-20261014-top/cpp/itu_audio_cli",
)
ROLE_TITLE = {"microphone": "麦克音", "reference": "参考音"}


def cli_path() -> Path:
    override = os.environ.get("ITU_CLI")
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise AudioError(f"ITU_CLI 指向的引擎不存在：{override}")
        return path
    for relative in CANDIDATE_BINARIES:
        path = ROOT / relative
        if path.is_file() and os.access(path, os.X_OK):
            return path
    raise AudioError("找不到引擎 itu_audio_cli；请先构建 cpp 目标，或用 ITU_CLI 指定路径。")


def _resample(x: np.ndarray, source: int, target: int) -> np.ndarray:
    if source == target:
        return x
    common = math.gcd(source, target)
    return signal.resample_poly(x, target // common, source // common)


def _levels(x: np.ndarray) -> dict:
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    return {
        "rms_dbfs": db(power(x)),
        "peak_dbfs": db(peak**2),
        "clipping_percent": float(np.mean(np.abs(x) >= 0.999) * 100),
        "silent": not bool(np.any(x)),
    }


def synthesize(
    microphone_name: str,
    microphone_blob: bytes,
    reference_name: str,
    reference_blob: bytes,
    rate: int | None = None,
    profile: str | None = None,
) -> dict:
    """Decode, align and run the engine. Returns metadata plus the output WAV."""
    microphone = decode("before", microphone_name, microphone_blob)
    reference = decode("echo", reference_name, reference_blob)
    target = int(rate) if rate else (microphone.rate if microphone.rate in SUPPORTED_RATES else DEFAULT_RATE)
    if target not in SUPPORTED_RATES:
        raise AudioError("引擎只支持 16 / 24 / 48 kHz，请选择其中一个采样率")
    mic = _resample(microphone.data, microphone.rate, target)[: target * MAX_SECONDS]
    ref = _resample(reference.data, reference.rate, target)[: target * MAX_SECONDS]
    frames = min(len(mic), len(ref))
    if frames < target // 10:
        raise AudioError("麦克音与参考音都至少需要 0.1 秒")
    if power(mic) < 1e-12:
        raise AudioError("麦克音是静音，无法生成输出")
    trimmed = (len(mic) - frames) / target + (len(ref) - frames) / target
    mic, ref = mic[:frames], ref[:frames]

    binary = cli_path()
    engine_profile = profile or os.environ.get("ITU_SYNTH_PROFILE", "vehicle_stream")
    timeout = int(os.environ.get("ITU_SYNTH_TIMEOUT", "300"))
    with tempfile.TemporaryDirectory(prefix="itu-synthesize-") as directory:
        work = Path(directory)
        sf.write(work / "mic.wav", mic.astype(np.float32), target, subtype="PCM_16")
        sf.write(work / "ref.wav", ref.astype(np.float32), target, subtype="PCM_16")
        prefix = work / "out"
        command = [
            str(binary), "--profile", engine_profile,
            "--mic", str(work / "mic.wav"), "--echo", str(work / "ref.wav"),
            "--out", str(prefix), "--microphones", "1",
        ]
        try:
            run = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            raise AudioError(f"引擎超过 {timeout} 秒未返回，请缩短音频后重试") from error
        if run.returncode != 0:
            lines = (run.stderr or run.stdout or "").strip().splitlines()
            raise AudioError("引擎执行失败：" + (lines[-1] if lines else f"退出码 {run.returncode}"))
        after_path = prefix.with_suffix(".after.wav")
        if not after_path.is_file():
            raise AudioError("引擎没有生成输出音频")
        processed, output_rate = sf.read(after_path, dtype="float32", always_2d=True)
        engine_report: dict = {}
        report_path = prefix.with_suffix(".json")
        if report_path.is_file():
            engine = json.loads(report_path.read_text())
            engine_report = {
                key: engine.get(key)
                for key in (
                    "sample_rate", "dsp_delay_samples", "stream_resets", "process_p95_ms",
                    "process_max_ms", "process_rtf", "processing_profile", "microphones",
                    "output_channels", "wind_noise_control", "reference_unreliable_frames",
                )
            }
        buffer = BytesIO()
        sf.write(buffer, processed[:, 0], output_rate, format="WAV", subtype="PCM_16")

    output = processed[:, 0]
    return {
        "engine": "itu-engineering-dsp/2.0",
        "profile": engine_profile,
        "binary": str(binary),
        "sample_rate": target,
        "seconds": frames / target,
        "trimmed_seconds": round(trimmed, 6),
        "inputs": {
            "microphone": {
                "name": microphone.name, "sample_rate": microphone.rate,
                "channels": microphone.original_channels,
                "selected_channel": microphone.channel + 1, **_levels(mic),
            },
            "reference": {
                "name": reference.name, "sample_rate": reference.rate,
                "channels": reference.original_channels,
                "selected_channel": reference.channel + 1, **_levels(ref),
            },
        },
        "output": {
            "name": "processed.wav", "sample_rate": output_rate,
            "samples": int(len(output)), **_levels(output),
        },
        "engine_report": engine_report,
        "wav": buffer.getvalue(),
    }
