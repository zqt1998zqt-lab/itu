"""Run the pinned C++ engine on an uploaded microphone/reference pair.

`server.audio.analyze` only measures audio the user already has. This module is
the other half: it feeds a microphone recording and its playback reference
through the same `itu_audio_cli` binary the acceptance runs use, and returns the
processed output plus the engine's own report.
"""
from __future__ import annotations

import hashlib
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

from server.audio import MAX_SECONDS, Audio, AudioError, db, decode, describe, power

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


def _chart(role: str, name: str, data: np.ndarray, rate: int, channels: int,
           digest: str, byte_count: int) -> dict:
    """Reuse the offline description so the plots match the rest of the app.

    `describe` also returns the per-file levels, so the page can label the
    charts without a second round trip. Values are rounded because the payload
    is JSON: the waveform is 640 min/max pairs and the spectrum about 400 bins
    per signal.
    """
    entry = describe(Audio(role=role, name=name, data=np.asarray(data, dtype=np.float64),
                           rate=rate, original_channels=channels, channel=0,
                           format="WAV", sha256=digest, byte_count=byte_count))
    spectrum = entry["spectrum"]
    return {
        "role": entry["role"], "name": entry["name"],
        "sample_rate": entry["sample_rate"], "channels": entry["channels"],
        "selected_channel": entry["selected_channel"], "format": entry["format"],
        "duration": round(entry["duration"], 6),
        "rms_dbfs": round(entry["rms_dbfs"], 4),
        "peak_dbfs": round(entry["peak_dbfs"], 4),
        "dc_offset": round(entry["dc_offset"], 8),
        "clipping_percent": round(entry["clipping_percent"], 5),
        "silent": entry["silent"], "sha256": digest, "bytes": byte_count,
        "waveform": {
            "min": [round(v, 5) for v in entry["waveform"]["min"]],
            "max": [round(v, 5) for v in entry["waveform"]["max"]],
        },
        "spectrum": {
            "frequency": [round(v, 1) for v in spectrum["frequency"]],
            "dbfs_hz": [round(v, 2) for v in spectrum["dbfs_hz"]],
            "peak_hz": None if spectrum["peak_hz"] is None else round(spectrum["peak_hz"], 1),
        },
    }


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
        # Waveform and spectrum for the two inputs and the generated output, so
        # the page can plot them without decoding audio in the browser.
        "charts": [
            _chart("before", microphone.name, mic, target, microphone.original_channels,
                   microphone.sha256, microphone.byte_count),
            _chart("echo", reference.name, ref, target, reference.original_channels,
                   reference.sha256, reference.byte_count),
            _chart("after", "processed.wav", output, output_rate, 1,
                   hashlib.sha256(buffer.getvalue()).hexdigest(), len(buffer.getvalue())),
        ],
        "wav": buffer.getvalue(),
    }


# ------------------------------------------------------------------ demo pairs
# Deterministic synthetic fixtures. They are diagnostic material, never claimed
# to be real speech or real vehicle recordings.
DEMO_RATE = 16000
DEMO_SECONDS = 6.0
DEMOS = (
    {
        "id": "echo_and_noise",
        "label": "回声 + 车内噪声",
        "description": "远端单讲与近端语音交替，麦克风含声学回声与稳态噪声。",
    },
    {
        "id": "near_only",
        "label": "仅近端语音",
        "description": "参考音为数字静音，麦克风只有近端语音与车内噪声。",
    },
    {
        "id": "wrong_reference",
        "label": "参考不可靠",
        "description": "参考音与麦克风内容不相关，用于观察参考可靠性保护。",
    },
    {
        "id": "far_end_stops",
        "label": "远端中途停止",
        "description": "远端播放 2 秒后停止，用于观察回声尾迹与保护释放。",
    },
)
DEMO_IDS = tuple(item["id"] for item in DEMOS)


def _band(rng, length: int, low: float, high: float, gain: float) -> np.ndarray:
    from scipy import signal as _signal
    data = _signal.sosfilt(
        _signal.butter(4, [low, high], fs=DEMO_RATE, btype="bandpass", output="sos"),
        rng.normal(size=length),
    )
    return data / max(float(np.max(np.abs(data))), 1e-12) * gain


def _demo_signals(scenario: str) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(200 + DEMO_IDS.index(scenario))
    length = int(DEMO_SECONDS * DEMO_RATE)
    time_axis = np.arange(length) / DEMO_RATE

    def gate(signal, start, end):
        return signal * ((time_axis >= start) & (time_axis < end))

    near = gate(_band(rng, length, 200, 3400, 0.18), 0.4, 3.0)
    near_late = gate(_band(rng, length, 200, 3400, 0.18), 3.6, 5.4)
    far = gate(_band(rng, length, 150, 3000, 0.20), 0.6, 3.2)
    far_short = gate(_band(rng, length, 150, 3000, 0.20), 0.6, 2.0)
    unrelated = gate(_band(rng, length, 400, 2400, 0.20), 0.8, 3.4)
    noise = _band(rng, length, 40, 1200, 0.02)
    # A 40 ms acoustic path is long enough to require real delay handling.
    echo = np.r_[np.zeros(640), far[:-640]] * 0.65

    if scenario == "echo_and_noise":
        reference, microphone = far, echo + near + near_late + noise
    elif scenario == "near_only":
        reference, microphone = np.zeros(length), near + near_late + noise
    elif scenario == "wrong_reference":
        reference = unrelated
        microphone = echo + near + near_late + noise
    elif scenario == "far_end_stops":
        reference = far_short
        echo = np.r_[np.zeros(640), far_short[:-640]] * 0.65
        microphone = echo + near + near_late + noise
    else:
        raise AudioError(f"未知示例场景：{scenario}")
    return microphone.astype(np.float32), reference.astype(np.float32)


def demo_pair(scenario: str) -> tuple[bytes, bytes]:
    """Return (microphone WAV, reference WAV) for a built-in scenario."""
    if scenario not in DEMO_IDS:
        raise AudioError(f"未知示例场景：{scenario}")
    microphone, reference = _demo_signals(scenario)
    payload = []
    for channel in (microphone, reference):
        buffer = BytesIO()
        sf.write(buffer, channel, DEMO_RATE, format="WAV", subtype="PCM_16")
        payload.append(buffer.getvalue())
    return payload[0], payload[1]
