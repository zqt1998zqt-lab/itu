"""Audio engineering measurements. No laboratory MOS/SLR/TCL stand-ins."""
from __future__ import annotations

import hashlib
import io
import math
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
from pydantic import BaseModel, Field, model_validator
from scipy import signal

ROLES = {"reference": "干净参考", "before": "麦克风处理前", "after": "麦克风处理后", "echo": "扬声器回声参考"}
MAX_BYTES = 32 * 1024 * 1024
MAX_SECONDS = 120
EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".aif", ".aiff"}


class AudioError(ValueError):
    pass


class Window(BaseModel):
    start: float = Field(default=0, ge=0, le=MAX_SECONDS)
    end: float | None = Field(default=None, gt=0, le=MAX_SECONDS)

    @model_validator(mode="after")
    def ordered(self):
        if self.end is not None and self.end <= self.start:
            raise ValueError("片段结束时间必须大于开始时间")
        return self


class Options(BaseModel):
    sample_rate: Literal[8000, 16000, 24000, 32000, 48000] = 16000
    channels: dict[str, int] = Field(default_factory=dict)
    speech: Window = Field(default_factory=Window)
    noise: Window | None = None
    far_end: Window | None = None
    max_delay_ms: float = Field(default=1000, ge=10, le=2000)
    after_offset_ms: float = Field(default=0, ge=-2000, le=2000)
    common_clock: bool = False
    far_end_only: bool = False
    aec_only: bool = False
    stationary_noise: bool = False
    speech_confirmed: bool = False


@dataclass
class Audio:
    role: str
    name: str
    data: np.ndarray
    rate: int
    original_channels: int
    channel: int | None
    format: str
    sha256: str
    byte_count: int


def decode(role: str, name: str, blob: bytes, channel: int | None = 0) -> Audio:
    """Decode one measurement channel, or preserve every channel when None."""
    if Path(name).suffix.lower() not in EXTENSIONS:
        raise AudioError(f"{ROLES[role]}：请上传 WAV、FLAC、MP3、OGG、M4A、AAC 或 AIFF 音频")
    if not blob or len(blob) > MAX_BYTES:
        raise AudioError(f"{ROLES[role]}：文件为空或超过 32 MB")
    if channel is not None and (channel < 0 or channel > 7):
        raise AudioError("声道编号无效")
    try:
        with sf.SoundFile(io.BytesIO(blob)) as f:
            rate, channels, frames, fmt = f.samplerate, f.channels, len(f), f.format
            validate_dimensions(rate, channels, frames)
            data = f.read(dtype="float32", always_2d=True)
    except AudioError:
        raise
    except (RuntimeError, ValueError):
        # Compressed files that libsndfile cannot read are decoded without a shell.
        import json
        try:
            with tempfile.TemporaryDirectory(prefix="itu-audio-") as directory:
                path = Path(directory) / ("audio" + Path(name).suffix.lower())
                path.write_bytes(blob)
                probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate,channels,codec_name:format=duration", "-of", "json", str(path)], capture_output=True, timeout=15, check=True)
                info = json.loads(probe.stdout)
                stream = info["streams"][0]
                rate, channels = int(stream["sample_rate"]), int(stream["channels"])
                duration = float(info.get("format", {}).get("duration", MAX_SECONDS))
                validate_dimensions(rate, channels, math.ceil(duration * rate))
                decoded = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-t", str(MAX_SECONDS + 0.1), "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"], capture_output=True, timeout=30, check=True)
                data = np.frombuffer(decoded.stdout, dtype="<f4").reshape(-1, channels)
                validate_dimensions(rate, channels, len(data))
                fmt = stream["codec_name"]
        except AudioError:
            raise
        except (OSError, subprocess.SubprocessError, KeyError, IndexError, ValueError) as error:
            raise AudioError(f"{ROLES[role]}：音频无法解码，文件可能损坏或并非音频") from error
    if channel is not None and channel >= channels:
        raise AudioError(f"{ROLES[role]}：文件只有 {channels} 个声道，所选声道不存在")
    mono = (data if channel is None else data[:, channel]).astype(np.float64)
    if not np.all(np.isfinite(mono)):
        raise AudioError(f"{ROLES[role]}：音频含有非有限样本")
    return Audio(role, Path(name).name, mono, rate, channels, channel, fmt, hashlib.sha256(blob).hexdigest(), len(blob))


def validate_dimensions(rate: int, channels: int, frames: int):
    if rate < 4000 or rate > 192000 or channels < 1 or channels > 8:
        raise AudioError("支持 4–192 kHz、1–8 声道音频")
    if frames < rate * 0.1 or frames / rate > MAX_SECONDS or frames * channels > 20_000_000:
        raise AudioError("每个音频需为 0.1–120 秒，解码后最多 2000 万样本")


def power(x):
    return float(np.mean(np.square(x)))


def db(value, multiplier=10):
    return float(multiplier * np.log10(max(float(value), 1e-15)))


def resample(audio: Audio, rate: int):
    common = math.gcd(rate, audio.rate)
    return signal.resample_poly(audio.data, rate // common, audio.rate // common) if rate != audio.rate else audio.data


def segment(x, rate, window: Window, offset=0.0, min_seconds=0.1):
    start = window.start + offset
    end = (window.end if window.end is not None else len(x) / rate) + offset
    if start < 0 or end > len(x) / rate + 1 / rate or end - start < min_seconds:
        raise AudioError("片段超出音频时长或不足有效长度；请检查开始、结束及延迟补偿")
    result = x[round(start * rate):round(end * rate)]
    if len(result) < round(min_seconds * rate):
        raise AudioError("片段没有足够样本")
    return result


def describe(a: Audio):
    x = a.data
    count = min(640, len(x))
    blocks = np.array_split(x, count)
    nperseg = min(4096, len(x))
    frequency, psd = signal.welch(x, a.rate, nperseg=nperseg, detrend="constant")
    # Grouped PSD bins retain narrow peaks rather than averaging them away.
    groups = np.array_split(np.arange(len(frequency)), min(400, len(frequency)))
    indices = [int(g[np.argmax(psd[g])]) for g in groups]
    peak = int(np.argmax(psd[1:]) + 1) if len(psd) > 1 else 0
    return {
        "role": a.role, "name": a.name, "sha256": a.sha256, "bytes": a.byte_count,
        "sample_rate": a.rate, "channels": a.original_channels, "selected_channel": a.channel + 1,
        "format": a.format, "duration": len(x) / a.rate,
        "rms_dbfs": db(power(x)), "peak_dbfs": db(np.max(np.abs(x)), 20),
        "dc_offset": float(np.mean(x)), "clipping_percent": float(np.mean(np.abs(x) >= 0.999) * 100),
        "silent": not bool(np.any(x)), "low_energy": power(x) < 1e-12,
        "waveform": {"min": [float(b.min()) for b in blocks], "max": [float(b.max()) for b in blocks]},
        "spectrum": {"frequency": frequency[indices].tolist(), "dbfs_hz": [db(p) for p in psd[indices]],
                     "peak_hz": float(frequency[peak]) if power(x) >= 1e-12 else None},
    }


def metric(id, name, value=None, unit="", status="unavailable", note="", role=None, method=""):
    return {"id": id, "name": name, "value": value, "unit": unit, "status": status, "note": note, "role": role, "method": method}


def align(reference, target, rate, max_delay_ms):
    n = min(len(reference), len(target))
    x, y = reference[:n].copy(), target[:n].copy()
    x -= x.mean()
    y -= y.mean()
    if n < rate * 0.25 or min(power(x), power(y)) < 1e-12:
        raise AudioError("参考或待测片段静音、低能量或过短，无法可靠对齐")
    bound = min(round(rate * max_delay_ms / 1000), n // 3)
    lags = np.arange(-bound, bound + 1)
    correlation = signal.correlate(y, x, mode="full", method="fft")[n - 1 + lags]
    starts_x, starts_y = np.maximum(-lags, 0), np.maximum(lags, 0)
    ends_x, ends_y = n - np.maximum(lags, 0), n - np.maximum(-lags, 0)
    px, py = np.r_[0, np.cumsum(x * x)], np.r_[0, np.cumsum(y * y)]
    denominator = np.sqrt((px[ends_x] - px[starts_x]) * (py[ends_y] - py[starts_y]))
    score = np.abs(correlation) / np.maximum(denominator, 1e-15)
    best = int(np.argmax(score))
    lag = int(lags[best])
    competitors = score[np.abs(lags - lag) > max(1, rate * 0.008)]
    second = float(np.max(competitors)) if len(competitors) else 0
    confidence = float(np.clip(score[best], 0, 1))
    reason = ""
    if confidence < 0.3:
        reason = "跨相关不足 0.3，可能内容不一致或失真严重"
    elif abs(lag) == bound:
        reason = "相关峰位于搜索边界，请扩大搜索范围或检查音频"
    elif second >= confidence * 0.98:
        reason = "存在接近的多个相关峰，周期信号或重复内容使时延不唯一"
    sx, sy = max(-lag, 0), max(lag, 0)
    overlap = n - abs(lag)
    return x[sx:sx + overlap], y[sy:sy + overlap], {"delay_ms": lag * 1000 / rate, "correlation": confidence, "overlap_seconds": overlap / rate, "reliable": not reason, "note": reason, "search_ms": bound * 1000 / rate}


def compare(reference, target, rate, options: Options, role, bounds=None, fixed_alignment=None):
    x, y, alignment = align(reference, target, rate, options.max_delay_ms) if fixed_alignment is None else (None, None, dict(fixed_alignment))
    if bounds is not None and alignment["reliable"]:
        start, end = bounds
        lag = round(alignment["delay_ms"] * rate / 1000)
        x, y = reference[start:end].copy(), target[start + lag:end + lag].copy()
        x -= x.mean()
        y -= y.mean()
        alignment["overlap_seconds"] = len(x) / rate
        alignment["common_reference_start"] = options.speech.start + start / rate
        alignment["common_reference_end"] = options.speech.start + end / rate
    label = ROLES[role]
    if alignment["reliable"]:
        # Crop first, then remove DC on the actual shared support used by SI-SDR.
        # Whole-window means differ when leading/trailing samples are removed.
        x = x - x.mean()
        y = y - y.mean()
    metrics = [metric(f"delay_{role}", f"{label}相对时延", alignment["delay_ms"] if alignment["reliable"] else None, "ms", "ok" if alignment["reliable"] else "unavailable", alignment["note"] or "正值表示待测晚于干净参考；不是往返 RTD", role, "归一化跨相关 / FFT / 去直流 / 多峰与边界检查")]
    if not alignment["reliable"]:
        for id, name, unit in [("sisdr", "SI-SDR", "dB"), ("gain", "相对增益", "dB"), ("stoi", "STOI", "")]:
            metrics.append(metric(f"{id}_{role}", f"{label} {name}", unit=unit, note="可靠对齐后才计算。" + alignment["note"], role=role))
        return {"role": role, "alignment": alignment, "metrics": metrics, "response": None}
    projection = np.dot(y, x) / np.dot(x, x) * x
    residual = y - projection
    residual_power = power(residual)
    censored = residual_power <= power(projection) * 1e-12
    sisdr = min(120.0, db(power(projection) / max(residual_power, 1e-30)))
    metrics.extend([
        metric(f"sisdr_{role}", f"{label} SI-SDR", sisdr, "dB", "warning" if censored else "ok", "残差达到数值精度下界，结果显示为 ≥120 dB" if censored else "已对齐、去直流；对整体增益不敏感，不能替代 MOS", role, "10log10(||αs||² / ||y−αs||²), α=(yᵀs)/(sᵀs)"),
        metric(f"gain_{role}", f"{label}相对增益", db(power(y) / power(x)), "dB", "ok", "已对齐片段的 RMS 电平比；同时包含噪声与失真能量", role, "20log10(RMS(y)/RMS(s))"),
    ])
    stoi_metric = metric(f"stoi_{role}", f"{label} STOI", note="请确认参考片段是干净语音；STOI 不评价 MOS", role=role)
    if options.speech_confirmed:
        try:
            from pystoi import stoi
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                value = float(stoi(x, y, rate, extended=False))
            if not np.isfinite(value):
                raise ValueError("输出无效")
            stoi_metric = metric(f"stoi_{role}", f"{label} STOI", value, "", "ok", "可懂度工程估计；未套用原表 MOS 门限", role, "pystoi 0.4.1 / standard STOI / 内部10 kHz重采样与静音剔除")
        except (ImportError, Warning, ValueError) as error:
            stoi_metric["note"] = f"STOI 无有效结果：{error}"
    metrics.append(stoi_metric)
    size = min(1024, len(x))
    f, pxx = signal.welch(x, rate, nperseg=size)
    _, pyy = signal.welch(y, rate, nperseg=size)
    _, pxy = signal.csd(x, y, rate, nperseg=size)
    coherence = np.abs(pxy) ** 2 / np.maximum(pxx * pyy, 1e-30)
    valid = (pxx > np.max(pxx) * 1e-4) & (coherence >= 0.6) & (f > 0)
    response = {"frequency": f.tolist(), "gain_db": [db(abs(h), 20) if ok else None for h, ok in zip(pxy / np.maximum(pxx, 1e-30), valid)], "coherence": coherence.clip(0, 1).tolist()}
    return {"role": role, "alignment": alignment, "metrics": metrics, "response": response}


STANDARD_REQUIREMENTS = [
    ("RTD", "往返时延", "需同步发送/返回测试路径；录音相对时延不等于 RTD"),
    ("SLR", "发送响度", "需人工嘴、发送灵敏度、标准频带/响度算法与标定"),
    ("RLR", "接收响度", "需人工耳/声压标定、接收灵敏度与标准响度算法"),
    ("ΔRLR", "接收响度线性偏差", "需多个规定输入电平的校准 RLR 测试"),
    ("FR", "频响掩模余量", "需测试激励、校准传递函数及对应模式上下限掩模"),
    ("Noise", "空闲信道噪声", "需规定静音片段、测量点标定及标准频带/加权；dBFS 仅工程诊断"),
    ("Peak", "噪声谱峰余量", "需校准噪声频谱与标准上限掩模"),
    ("MOS", "TMOS / POLQA", "专用质量模型未接入；STOI、SI-SDR 不替代 MOS"),
    ("OOB", "带外信号", "需规定单频激励、带宽与仪器修正定义"),
    ("DT", "双讲类别占比", "需双讲测试序列、分类算法及有效片段分母"),
    ("TCL", "TCL / TCLw", "需标准测量点、声学标定和加权；ERLE 不替代 TCL"),
    ("SNR", "HATS 语音信噪比", "需 HATS 声学标定、规定场景和信号/噪声测量定义"),
    ("G-MOS", "背景噪声整体质量", "需实验室指定算法/主观协议及场景，不使用线性 SNR 映射"),
    ("N-MOS", "背景噪声噪声质量", "需实验室指定算法/主观协议及场景"),
    ("S-MOS", "背景噪声语音质量", "需实验室指定算法/主观协议及场景"),
    ("CNG ΔL", "舒适噪声电平误差", "需规定舒适噪声片段、背景基准与校准电平"),
    ("CNG mask", "舒适噪声谱余量", "需规定片段、背景频谱与标准掩模"),
]


def analyze(audios: dict[str, Audio], options: Options):
    if "after" not in audios:
        raise AudioError("请上传麦克风处理后音频")
    rate = options.sample_rate
    samples = {role: resample(audio, rate) for role, audio in audios.items()}
    descriptions = [describe(audio) for audio in audios.values()]
    notices = []
    if any(audio.rate != rate for audio in audios.values()):
        notices.append(f"成对计算统一重采样到 {rate} Hz；波形、PSD 和文件电平保留原始采样率。")
    notices.append("各文件只使用指定声道，不做多声道平均。波形与 PSD 显示完整文件；成对指标使用参考比较片段。")
    metrics, pairs = [], []
    for item in descriptions:
        role = item["role"]
        for key, title, unit in [("rms_dbfs", "RMS 电平", "dBFS"), ("peak_dbfs", "峰值电平", "dBFS"), ("clipping_percent", "满幅样本比例", "%"), ("dc_offset", "直流偏置", "FS")]:
            warning = item["low_energy"] or (key == "clipping_percent" and item[key] > 0)
            metrics.append(metric(f"{key}_{role}", f"{ROLES[role]} {title}", item[key], unit, "warning" if warning else "ok", "数字零信号：对数显示下界 −150 dBFS，不视为有效语音" if item["silent"] else "低于可靠对齐的有效电平，保留数字电平实测值" if item["low_energy"] else ("幅值 ≥0.999 FS 的样本比例；压缩音频只能诊断解码后满幅" if key == "clipping_percent" else "完整文件 / 所选声道 / 数字满量程参考，未经声学标定"), role))
    pair_inputs = {}
    if "reference" in samples:
        for role in ("before", "after"):
            if role not in samples:
                continue
            try:
                x = segment(samples["reference"], rate, options.speech, min_seconds=0.25)
                y = segment(samples[role], rate, options.speech, min_seconds=0.25)
                pair = compare(x, y, rate, options, role)
                pair_inputs[role] = (x, y)
                pairs.append(pair)
                metrics.extend(pair["metrics"])
            except AudioError as error:
                for id, name, unit in [("delay", "相对时延", "ms"), ("sisdr", "SI-SDR", "dB"), ("gain", "相对增益", "dB"), ("stoi", "STOI", "")]:
                    metrics.append(metric(f"{id}_{role}", f"{ROLES[role]} {name}", unit=unit, note=str(error), role=role))
    else:
        metrics.append(metric("reference_missing", "相对时延 / SI-SDR / STOI / 传递函数", note="缺少干净参考音频；可以查看单文件数字电平和频谱"))

    # Front/back quality changes must use the identical reference sample support.
    common_support = False
    if len(pairs) == 2 and all(p["alignment"]["reliable"] for p in pairs):
        lags = {p["role"]: round(p["alignment"]["delay_ms"] * rate / 1000) for p in pairs}
        start = max(max(-lag, 0) for lag in lags.values())
        end = min(min(len(x), len(y)) - max(lags[role], 0) for role, (x, y) in pair_inputs.items())
        if end - start >= rate * 0.25:
            common_support = True
            metrics = [m for m in metrics if m["id"] not in {m["id"] for p in pairs for m in p["metrics"]}]
            pairs = [compare(*pair_inputs[p["role"]], rate, options, p["role"], (start, end), p["alignment"]) for p in pairs]
            for pair in pairs:
                metrics.extend(pair["metrics"])
            notices.append(f"前后保真对比使用相同参考样本：{options.speech.start + start / rate:.4f}–{options.speech.start + end / rate:.4f} s。")
        else:
            notices.append("前后共有参考片段不足 0.25 秒，不计算质量改善量；请扩大参考比较片段。")

    noise_values = {}
    for role in ("before", "after"):
        if role not in samples:
            continue
        result = metric(f"noise_{role}", f"{ROLES[role]}噪声片段 RMS", unit="dBFS", note="请填写无语音、无回声的噪声片段")
        if options.noise is not None:
            try:
                offset = options.after_offset_ms / 1000 if role == "after" else 0
                noise = segment(samples[role], rate, options.noise, offset)
                noise_values[role] = power(noise)
                result = metric(f"noise_{role}", f"{ROLES[role]}噪声片段 RMS", db(power(noise)), "dBFS", "warning" if power(noise) < 1e-12 else "ok", "在重采样后所选噪声片段计算；需确认无语音、无回声", role)
            except AudioError as error:
                result["note"] = str(error)
        metrics.append(result)
    attenuation = metric("noise_attenuation", "噪声衰减", unit="dB", note="需处理前/后、噪声片段及共享时钟/电平确认")
    if {"before", "after"} <= noise_values.keys() and options.common_clock:
        if min(noise_values.values()) < 1e-12:
            attenuation["note"] = "噪声片段静音或低于有效电平，无法给出有效衰减量"
        else:
            attenuation = metric("noise_attenuation", "噪声衰减", db(noise_values["before"] / noise_values["after"]), "dB", "ok", "正值表示处理后更安静；整体增益降低也会提高此值，需联合相对增益/SI-SDR观察", method="10log10(Pnoise_before/Pnoise_after)")
    metrics.append(attenuation)
    for role, pn in noise_values.items():
        result = metric(f"snr_{role}", f"{ROLES[role]}片段估计 SNR", unit="dB", note="需确认噪声平稳且参考比较片段包含语音与同一噪声")
        if options.stationary_noise:
            try:
                mixture = segment(samples[role], rate, options.speech, options.after_offset_ms / 1000 if role == "after" else 0)
                ps = power(mixture) - pn
                if pn < 1e-12 or ps <= 0:
                    raise AudioError("噪声静音/低能量或混合功率不大于噪声功率；不能有效线性扣除")
                result = metric(f"snr_{role}", f"{ROLES[role]}片段估计 SNR", db(ps / pn), "dB", "ok", "基于平稳、不相关噪声假设；不是原表 HATS SNR，未套用其限值", role, "10log10((Pmixture−Pnoise)/Pnoise)")
            except AudioError as error:
                result["note"] = str(error)
        metrics.append(result)

    echo_attenuation = metric("echo_attenuation", "远端单讲回声片段衰减", unit="dB", note="需处理前/后、回声参考、远端单讲片段及共享时钟/电平确认")
    if options.far_end is not None and options.far_end_only and options.common_clock and {"before", "after", "echo"} <= samples.keys():
        try:
            pre = segment(samples["before"], rate, options.far_end, min_seconds=0.25)
            post = segment(samples["after"], rate, options.far_end, options.after_offset_ms / 1000, min_seconds=0.25)
            echo = segment(samples["echo"], rate, options.far_end, min_seconds=0.25)
            _, _, check = align(echo, pre, rate, options.max_delay_ms)
            if not check["reliable"]:
                raise AudioError("处理前与回声参考未通过相关性检查：" + check["note"])
            if min(power(pre), power(post), power(echo)) < 1e-12:
                raise AudioError("片段静音或低于有效电平，无法给出有效回声片段衰减")
            echo_attenuation = metric("echo_attenuation", "远端单讲回声片段衰减", db(power(pre) / power(post)), "dB", "ok", f"处理前/回声参考相关 {check['correlation']:.3f}；处理后补偿 {options.after_offset_ms:g} ms。功率比可能含后级抑制/降噪/增益影响，含底噪，不是标准 TCL 或滤波器 ERLE", method="10log10(Pmic_before/Pmic_after)，所选片段总功率衰减")
        except AudioError as error:
            echo_attenuation["note"] = str(error)
    metrics.append(echo_attenuation)
    erle = metric("erle", "AEC ERLE 工程估计", unit="dB", note="需有效远端单讲结果，并确认处理前/后为 AEC 滤波器测量点，排除 NLP、降噪和 AGC 附加衰减")
    if options.aec_only and echo_attenuation["status"] == "ok":
        erle = metric("erle", "AEC ERLE 工程估计", echo_attenuation["value"], "dB", "ok", "用户已确认 AEC 滤波器前/后测量点且排除后级处理。片段平均功率比含底噪；未执行 G.168 全套测试及 125 ms 时域测量", method="10log10(Pecho_before/Presidual)，排除后级 NLP；不是 TCL/TCLw")
    metrics.append(erle)
    for id, name, a, b, unit in [("sisdr_improvement", "SI-SDR 改善", "sisdr_before", "sisdr_after", "dB"), ("stoi_improvement", "STOI 变化", "stoi_before", "stoi_after", "")]:
        previous = next((m for m in metrics if m["id"] == a), None)
        current = next((m for m in metrics if m["id"] == b), None)
        comparable = previous and current and previous["status"] == current["status"] == "ok" and options.common_clock and common_support
        metrics.append(metric(id, name, current["value"] - previous["value"] if comparable else None, unit, "ok" if comparable else "unavailable", "处理后 − 处理前；同一参考、片段和电平条件" if comparable else "需前后有效结果、足够的共同参考样本范围与共享时钟/电平确认"))
    return {"schema_version": 2, "engine": "itu-engineering-dsp/2.0", "files": descriptions, "options": options.model_dump(), "metrics": metrics, "pairs": pairs, "notices": notices, "standards": [{"symbol": symbol, "name": name, "status": "unavailable", "requirement": requirement} for symbol, name, requirement in STANDARD_REQUIREMENTS]}


def demo_wav(role):
    """Deterministic diagnostic fixture, never claimed to be real speech."""
    rate = 16000
    rng = np.random.default_rng(200)
    t = np.arange(6 * rate) / rate
    clean = signal.sosfilt(signal.butter(4, [200, 3400], fs=rate, btype="bandpass", output="sos"), rng.normal(size=len(t))) * 0.18
    clean *= ((t >= 0.4) & (t < 3.0))
    far = signal.sosfilt(signal.butter(4, [150, 3000], fs=rate, btype="bandpass", output="sos"), rng.normal(size=len(t))) * 0.2
    far *= ((t >= 3.3) & (t < 4.8))
    noise = rng.normal(size=len(t)) * 0.008
    delayed = np.r_[np.zeros(1280), clean[:-1280]]
    acoustic_echo = np.r_[np.zeros(400), far[:-400]] * 0.65
    values = {"reference": clean, "echo": far, "before": delayed + acoustic_echo + noise, "after": delayed * 0.9 + acoustic_echo * 0.1 + noise * 0.25}
    buffer = io.BytesIO()
    sf.write(buffer, values[role], rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()
