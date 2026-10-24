"""Generate aligned audio comparison figures without external services."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import math
from typing import Literal

import matplotlib

matplotlib.use("Agg")

from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
import numpy as np
from pydantic import BaseModel, Field
from scipy import signal

from server.audio import Audio, AudioError, align, db, power, resample


class GraphOptions(BaseModel):
    sample_rate: Literal[8000, 16000, 24000, 32000, 48000] = 16000
    before_channel: int = Field(default=0, ge=0, le=7)
    after_channel: int = Field(default=0, ge=0, le=7)
    align: bool = True
    max_delay_ms: float = Field(default=1000, ge=10, le=2000)
    time_offset_seconds: float = Field(default=0, ge=0, le=120)
    before_label: str = Field(default="处理前", min_length=1, max_length=40)
    after_label: str = Field(default="处理后", min_length=1, max_length=40)


@dataclass(frozen=True)
class FigureSpec:
    id: str
    title: str
    category: str
    description: str
    detail: str


@dataclass
class Prepared:
    before: np.ndarray | None
    after: np.ndarray
    rate: int
    time_offset_seconds: float
    alignment: dict
    notices: list[str]


FIGURES = {
    "waveform-energy": FigureSpec(
        "waveform-energy",
        "波形与短时能量",
        "dynamics",
        "对齐后的输入/输出波形和 10 ms 帧能量，检查增益、瞬态、削波及静音。",
        "波形使用 min/max 包络；能量为真实 10 ms 矩形帧的样本平方均值（含 DC），末帧按实际样本数计算。",
    ),
    "log-stft-before": FigureSpec(
        "log-stft-before",
        "对数频率 STFT · 处理前",
        "time-frequency",
        "保留窄带峰、谐波和低频噪声结构。",
        "32 ms Hann 窗；10 ms hop；单边 FFT bin 均方功率（FS²/bin），频带显示 bin 功率均值；色标与处理后共享。",
    ),
    "log-stft-after": FigureSpec(
        "log-stft-after",
        "对数频率 STFT · 处理后",
        "time-frequency",
        "与处理前使用同一时间轴、频率轴和 dB 色标。",
        "颜色变化可直接对应实际输出，不使用逐图自动归一化。",
    ),
    "spectral-delta": FigureSpec(
        "spectral-delta",
        "前后频谱差分",
        "comparison",
        "红色表示能量增加，蓝色表示能量减少，适合定位降噪、回声抑制和 EQ 增益。",
        "差分按 10log10(P_after/P_before) 计算；低能量静音区被遮罩，刻度限制为 ±30 dB。",
    ),
    "ltas": FigureSpec(
        "ltas",
        "LTAS 长时平均频谱",
        "bands",
        "比较整体底噪、频响倾斜和长期能量平衡。",
        "Welch 单边功率谱密度，单位 dBFS/Hz；保留逐窗去均值，会平均掉短时瞬态。",
    ),
    "erb": FigureSpec(
        "erb",
        "ERB 听觉频带谱",
        "bands",
        "按近似听觉临界带压缩频率，观察感知频带中的能量变化。",
        "ERB-spaced 三角滤波器组，不是 Mel，也不是受试者主观听感。",
    ),
    "high-time": FigureSpec(
        "high-time",
        "高时间分辨率频谱",
        "transient",
        "短窗显示咔嗒、爆音、门限切换和快速包络变化。",
        "8 ms 窗、2 ms hop；时间分辨率提高，频率分辨率相应下降。",
    ),
    "triple": FigureSpec(
        "triple",
        "共享标尺原始/处理/差分三连图",
        "comparison",
        "在同一张图中按顺序比较原始频谱、处理频谱和 dB 差分。",
        "前两幅共享 dB 色标，第三幅固定 ±30 dB 发散色标。",
    ),
}


def _frame_size(rate: int, seconds: float, minimum: int = 128) -> int:
    return max(minimum, 2 ** round(math.log2(max(rate * seconds, minimum))))


def _stft_power(values: np.ndarray, rate: int, fft_size: int, hop: int):
    """One-sided mean-square power per FFT bin, in FS^2/bin.

    SciPy's complex STFT does not double one-sided interior bins. Its PSD
    scaling is already normalized by fs*sum(window^2); multiply by bin width
    and fold negative frequencies exactly once. Summing bins gives the
    Hann-squared-weighted time-domain mean square, including DC/Nyquist.
    """
    if values.size < fft_size:
        values = np.pad(values, (0, fft_size - values.size))
    frequency, time, spectrum = signal.stft(
        values,
        fs=rate,
        window="hann",
        nperseg=fft_size,
        noverlap=max(0, fft_size - hop),
        detrend=False,
        return_onesided=True,
        scaling="psd",
        boundary=None,
        padded=False,
    )
    bin_power = np.abs(spectrum) ** 2 * (rate / fft_size)
    if fft_size % 2 == 0:
        bin_power[1:-1] *= 2
    else:
        bin_power[1:] *= 2
    return frequency, time, bin_power


def _frame_energy(values: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Actual unwindowed 10 ms sample mean square; never pad the last frame."""
    hop = max(1, rate // 100)
    starts = np.arange(0, values.size, hop, dtype=np.int64)
    counts = np.minimum(hop, values.size - starts)
    square = np.asarray(values, dtype=np.float64) ** 2
    means = np.array([np.mean(square[start:start + count]) for start, count in zip(starts, counts)])
    times = (starts + (counts - 1) / 2.0) / rate
    return times, means


def _ltas_density(values: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray]:
    """One-sided Welch PSD in FS^2/Hz, retaining the original DC detrend."""
    size = min(4096, len(values))
    return signal.welch(values, rate, nperseg=size, noverlap=size // 2,
                        detrend="constant", return_onesided=True, scaling="density")


def _time_reduce(values: np.ndarray, maximum: int = 640) -> tuple[np.ndarray, np.ndarray]:
    if values.shape[1] <= maximum:
        return np.arange(values.shape[1], dtype=float), values
    edges = np.linspace(0, values.shape[1], maximum + 1, dtype=np.int64)
    reduced = np.column_stack(
        [
            np.mean(values[:, start:max(start + 1, end)], axis=1)
            for start, end in zip(edges[:-1], edges[1:])
        ]
    )
    return (edges[:-1] + edges[1:] - 1) / 2.0, reduced


def _log_bands(
    power_values: np.ndarray,
    frequency: np.ndarray,
    rate: int,
    bands: int = 160,
    minimum_hz: float = 20.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    upper = rate / 2.0
    lower = min(minimum_hz, upper / 4)
    edges = np.geomspace(lower, upper, bands + 1)
    result = np.empty((bands, power_values.shape[1]), dtype=float)
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (frequency >= left) & (frequency < right if index < bands - 1 else frequency <= right)
        if not np.any(selected):
            nearest = int(np.argmin(np.abs(frequency - (left + right) / 2)))
            result[index] = power_values[nearest]
        else:
            result[index] = np.mean(power_values[selected], axis=0)
    centers = np.sqrt(edges[:-1] * edges[1:])
    return centers, result, edges


def _db(values: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(values, 1e-20))


def _shared_range(*matrices: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([matrix[np.isfinite(matrix)].ravel() for matrix in matrices])
    if not values.size:
        return -120.0, -20.0
    upper = float(np.percentile(values, 99.7))
    return upper - 70.0, upper


def _new_figure(width: float = 12.0, height: float = 6.0) -> Figure:
    figure = Figure(figsize=(width, height), facecolor="#ffffff")
    return figure


def _plot_label(value: str | None, fallback: str) -> str | None:
    if value is None:
        return None
    return value if value.isascii() else fallback


def _save_png(figure: Figure) -> bytes:
    buffer = BytesIO()
    figure.savefig(
        buffer,
        format="png",
        dpi=120,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    return buffer.getvalue()


def _style_axis(axis, *, xlabel: str, ylabel: str, title: str):
    axis.set_facecolor("#fbfcfe")
    axis.set_title(title, loc="left", fontsize=12, color="#253850", fontweight="semibold")
    axis.set_xlabel(xlabel, color="#62758c")
    axis.set_ylabel(ylabel, color="#62758c")
    axis.tick_params(colors="#6d7e91", labelsize=8)
    axis.grid(color="#e6edf4", linewidth=0.6, alpha=0.8)
    for spine in axis.spines.values():
        spine.set_color("#dce5ee")


def _wave_envelope(values: np.ndarray, points: int = 1800) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(0, values.size, min(points, values.size) + 1, dtype=np.int64)
    low = np.empty(len(edges) - 1)
    high = np.empty(len(edges) - 1)
    centers = np.empty(len(edges) - 1)
    for index, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        chunk = values[start:max(start + 1, end)]
        low[index], high[index] = float(np.min(chunk)), float(np.max(chunk))
        centers[index] = (start + end - 1) / 2
    return centers, low, high


def _prepare(before: Audio | None, after: Audio, options: GraphOptions) -> Prepared:
    rate = options.sample_rate
    before_values = resample(before, rate) if before is not None else None
    after_values = resample(after, rate)
    notices = [
        "图形用于工程诊断，不替代 SI-SDR、STOI、MOS 或正式 ITU 验收。",
        "频谱颜色采用原始/处理共享标尺；差分图使用有符号 dB 变化。",
    ]
    alignment = {"requested": options.align and before_values is not None, "applied": False}
    if before_values is None:
        alignment.update(
            {
                "delay_ms": None,
                "correlation": None,
                "reliable": None,
                "note": "仅输入一路音频，不执行前后对齐。",
            }
        )
        return Prepared(None, after_values, rate, options.time_offset_seconds, alignment, notices)

    if options.align:
        try:
            aligned_before, aligned_after, result = align(
                before_values, after_values, rate, options.max_delay_ms
            )
            alignment.update(result)
            alignment["applied"] = bool(result["reliable"])
            if result["reliable"]:
                before_values, after_values = aligned_before, aligned_after
                notices.append(
                    f"已自动对齐，处理后相对处理前延迟 {result['delay_ms']:+.3f} ms。"
                )
            else:
                common = min(len(before_values), len(after_values))
                before_values, after_values = before_values[:common], after_values[:common]
                notices.append("自动对齐不可靠，比较图按共同起点截齐；请在网页复核原始文件。")
        except AudioError as error:
            common = min(len(before_values), len(after_values))
            before_values, after_values = before_values[:common], after_values[:common]
            alignment.update(
                {
                    "applied": False,
                    "delay_ms": None,
                    "correlation": None,
                    "reliable": False,
                    "note": str(error),
                }
            )
            notices.append(f"自动对齐未完成：{error}。比较图按共同起点截齐。")
    else:
        common = min(len(before_values), len(after_values))
        before_values, after_values = before_values[:common], after_values[:common]
        alignment.update(
            {
                "delay_ms": None,
                "correlation": None,
                "reliable": None,
                "note": "用户关闭自动对齐，按共同起点截齐。",
            }
        )
    if len(before_values) != len(after_values):
        notices.append("两路文件长度不同，分析图使用共同重叠长度。")
    return Prepared(before_values, after_values, rate, options.time_offset_seconds, alignment, notices)


def _erc_edges(rate: int, bands: int = 48, minimum_hz: float = 20.0) -> np.ndarray:
    upper = rate / 2.0
    lower = min(minimum_hz, upper / 4)

    def hz_to_erb(value):
        return 21.4 * np.log10(1.0 + 4.37 * np.asarray(value) / 1000.0)

    def erb_to_hz(value):
        return (10.0 ** (np.asarray(value) / 21.4) - 1.0) * 1000.0 / 4.37

    return erb_to_hz(np.linspace(hz_to_erb(lower), hz_to_erb(upper), bands + 1))


def _band_filter_power(power_values: np.ndarray, frequency: np.ndarray, edges: np.ndarray) -> np.ndarray:
    result = np.empty((len(edges) - 1, power_values.shape[1]), dtype=float)
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (frequency >= left) & (frequency < right if index < len(edges) - 2 else frequency <= right)
        if not np.any(selected):
            nearest = int(np.argmin(np.abs(frequency - (left + right) / 2)))
            result[index] = power_values[nearest]
        else:
            result[index] = np.mean(power_values[selected], axis=0)
    return result


def _plot_waveform_energy(
    prepared: Prepared,
    labels: tuple[str | None, str],
) -> bytes:
    pair = prepared.before is not None
    figure = _new_figure(12.5, 7.2 if pair else 5.5)
    axes = figure.subplots(3 if pair else 2, 1, sharex=True)
    axes = np.atleast_1d(axes)
    time_offset = prepared.time_offset_seconds
    duration = len(prepared.after) / prepared.rate
    rows = [(prepared.before, labels[0], "#c48738")] if pair else []
    rows.append((prepared.after, labels[1], "#177f96"))
    for axis, (values, label, color) in zip(axes[: len(rows)], rows):
        centers, low, high = _wave_envelope(values)
        time = time_offset + centers / prepared.rate
        axis.fill_between(time, low, high, color=color, alpha=0.72, linewidth=0)
        axis.set_ylim(-1.04, 1.04)
        _style_axis(axis, xlabel="", ylabel="Amplitude (FS)", title=label or "Input")
        axis.set_xlim(time_offset, time_offset + duration)
    energy_axis = axes[-1]
    if pair:
        before_time, before_energy = _frame_energy(prepared.before, prepared.rate)
        energy_axis.plot(time_offset + before_time, _db(before_energy), color="#c48738", linewidth=1.1, label=labels[0])
    after_time, after_energy = _frame_energy(prepared.after, prepared.rate)
    energy_axis.plot(time_offset + after_time, _db(after_energy), color="#177f96", linewidth=1.1, label=labels[1])
    _style_axis(energy_axis, xlabel="Time (s)", ylabel="Mean-square level (dBFS)", title="10 ms unwindowed frame energy")
    energy_axis.set_xlim(time_offset, time_offset + duration)
    energy_axis.legend(loc="upper right", frameon=False, fontsize=8)
    figure.tight_layout(h_pad=1.2)
    return _save_png(figure)


def _plot_log_spectrogram(
    time: np.ndarray,
    frequencies: np.ndarray,
    matrix: np.ndarray,
    low: float,
    high: float,
    title: str,
) -> bytes:
    figure = _new_figure(12.5, 5.8)
    axis = figure.subplots()
    mesh = axis.pcolormesh(
        time,
        frequencies,
        matrix,
        shading="nearest",
        cmap="magma",
        norm=Normalize(low, high),
        rasterized=True,
    )
    axis.set_yscale("log")
    axis.set_ylim(max(frequencies[0], 20), frequencies[-1])
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}k" if value >= 1000 else f"{value:g}"))
    _style_axis(axis, xlabel="Time (s)", ylabel="Frequency (Hz)", title=title)
    figure.tight_layout()
    figure.colorbar(mesh, ax=axis, pad=0.012, label="Mean bin power (dBFS)")
    return _save_png(figure)


def _plot_delta(time: np.ndarray, frequencies: np.ndarray, delta: np.ndarray) -> bytes:
    figure = _new_figure(12.5, 5.8)
    axis = figure.subplots()
    mesh = axis.pcolormesh(
        time,
        frequencies,
        delta,
        shading="nearest",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-30, vcenter=0, vmax=30),
        rasterized=True,
    )
    axis.set_yscale("log")
    axis.set_ylim(max(frequencies[0], 20), frequencies[-1])
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}k" if value >= 1000 else f"{value:g}"))
    _style_axis(axis, xlabel="Time (s)", ylabel="Frequency (Hz)", title="After - before spectral change")
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.colorbar(mesh, ax=axis, pad=0.012, label="Delta (dB)")
    figure.text(
        0.012,
        0.015,
        "Red: energy increased · Blue: energy decreased · silence masked",
        fontsize=8,
        color="#687b91",
    )
    return _save_png(figure)


def _plot_ltas(
    frequencies: np.ndarray,
    before_db: np.ndarray | None,
    after_db: np.ndarray,
    labels: tuple[str | None, str],
) -> bytes:
    figure = _new_figure(12.5, 5.3)
    axis = figure.subplots()
    if before_db is not None:
        axis.semilogx(frequencies, before_db, color="#c48738", linewidth=1.5, label=labels[0])
    axis.semilogx(frequencies, after_db, color="#177f96", linewidth=1.5, label=labels[1])
    axis.set_xlim(max(frequencies[0], 20), frequencies[-1])
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}k" if value >= 1000 else f"{value:g}"))
    _style_axis(axis, xlabel="Frequency (Hz)", ylabel="PSD (dBFS/Hz)", title="Long-term average spectrum")
    axis.legend(loc="best", frameon=False, fontsize=8)
    figure.tight_layout()
    return _save_png(figure)


def _plot_erb_pair(
    prepared: Prepared,
    frequency: np.ndarray,
    before_power: np.ndarray | None,
    after_power: np.ndarray,
    labels: tuple[str | None, str],
) -> bytes:
    edges = _erc_edges(prepared.rate)
    centers = np.sqrt(edges[:-1] * edges[1:])
    after_matrix = _db(_band_filter_power(after_power, frequency, edges))
    before_matrix = _db(_band_filter_power(before_power, frequency, edges)) if before_power is not None else None
    low, high = _shared_range(after_matrix, before_matrix) if before_matrix is not None else _shared_range(after_matrix)
    times = np.linspace(
        prepared.time_offset_seconds,
        prepared.time_offset_seconds + len(prepared.after) / prepared.rate,
        after_matrix.shape[1],
    )
    rows = []
    if before_matrix is not None:
        rows.append((before_matrix, labels[0]))
    rows.append((after_matrix, labels[1]))
    figure = _new_figure(12.5, 6.6 if len(rows) == 2 else 4.8)
    axes = np.atleast_1d(figure.subplots(len(rows), 1, sharex=True, sharey=True))
    for axis, (matrix, label) in zip(axes, rows):
        mesh = axis.pcolormesh(
            times,
            centers,
            matrix,
            shading="nearest",
            cmap="viridis",
            norm=Normalize(low, high),
            rasterized=True,
        )
        axis.set_yscale("log")
        axis.set_ylim(centers[0], centers[-1])
        _style_axis(axis, xlabel="", ylabel="ERB centre (Hz)", title=label or "Input")
    axes[-1].set_xlabel("Time (s)")
    figure.tight_layout()
    figure.colorbar(mesh, ax=axes, pad=0.012, label="Mean bin power (dBFS)")
    return _save_png(figure)


def _plot_high_time(
    prepared: Prepared,
    before_short: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    after_short: tuple[np.ndarray, np.ndarray, np.ndarray],
    labels: tuple[str | None, str],
) -> bytes:
    frequency, time, after_power = after_short
    centers, after_matrix, edges = _log_bands(after_power, frequency, prepared.rate, bands=96)
    before_matrix = None
    if before_short is not None:
        before_matrix = _log_bands(before_short[2], before_short[0], prepared.rate, bands=96)[1]
    low, high = _shared_range(_db(after_matrix), _db(before_matrix)) if before_matrix is not None else _shared_range(_db(after_matrix))
    times = np.linspace(
        prepared.time_offset_seconds,
        prepared.time_offset_seconds + len(prepared.after) / prepared.rate,
        after_matrix.shape[1],
    )
    rows = []
    if before_matrix is not None:
        rows.append((_db(before_matrix), labels[0]))
    rows.append((_db(after_matrix), labels[1]))
    figure = _new_figure(12.5, 6.8 if len(rows) == 2 else 4.8)
    axes = np.atleast_1d(figure.subplots(len(rows), 1, sharex=True, sharey=True))
    for axis, (matrix, label) in zip(axes, rows):
        mesh = axis.pcolormesh(
            times,
            centers,
            matrix,
            shading="nearest",
            cmap="magma",
            norm=Normalize(low, high),
            rasterized=True,
        )
        axis.set_yscale("log")
        axis.set_ylim(centers[0], centers[-1])
        _style_axis(axis, xlabel="", ylabel="Frequency (Hz)", title=label or "Input")
    axes[-1].set_xlabel("Time (s)")
    figure.tight_layout()
    figure.colorbar(mesh, ax=axes, pad=0.012, label="Mean bin power (dBFS)")
    return _save_png(figure)


def _plot_triple(
    prepared: Prepared,
    time: np.ndarray,
    frequencies: np.ndarray,
    before_matrix: np.ndarray,
    after_matrix: np.ndarray,
    delta_matrix: np.ndarray,
    labels: tuple[str, str],
) -> bytes:
    low, high = _shared_range(before_matrix, after_matrix)
    times = np.linspace(
        prepared.time_offset_seconds,
        prepared.time_offset_seconds + len(prepared.after) / prepared.rate,
        before_matrix.shape[1],
    )
    figure = _new_figure(12.5, 9.2)
    axes = figure.subplots(3, 1, sharex=True, sharey=True)
    for axis, matrix, title in zip(
        axes[:2],
        (before_matrix, after_matrix),
        (labels[0], labels[1]),
    ):
        mesh = axis.pcolormesh(
            times,
            frequencies,
            matrix,
            shading="nearest",
            cmap="magma",
            norm=Normalize(low, high),
            rasterized=True,
        )
        axis.set_yscale("log")
        axis.set_ylim(max(frequencies[0], 20), frequencies[-1])
        _style_axis(axis, xlabel="", ylabel="Frequency (Hz)", title=title)
    delta = axes[2].pcolormesh(
        times,
        frequencies,
        delta_matrix,
        shading="nearest",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-30, vcenter=0, vmax=30),
        rasterized=True,
    )
    axes[2].set_yscale("log")
    axes[2].set_ylim(max(frequencies[0], 20), frequencies[-1])
    _style_axis(axes[2], xlabel="Time (s)", ylabel="Frequency (Hz)", title="Delta")
    figure.tight_layout()
    figure.colorbar(mesh, ax=axes[:2], pad=0.012, label="Mean bin power (dBFS)")
    figure.colorbar(delta, ax=axes[2], pad=0.012, label="Delta (dB)")
    return _save_png(figure)


def _band_powers(values: np.ndarray, rate: int) -> dict[str, float]:
    size = min(4096, len(values))
    frequency, psd = signal.welch(values, rate, nperseg=size, noverlap=size // 2, detrend="constant")
    upper = rate / 2
    bands = {
        "low_20_300": (20.0, min(300.0, upper)),
        "mid_300_3000": (300.0, min(3000.0, upper)),
        "high_3000_nyquist": (3000.0, upper),
    }
    step = frequency[1] - frequency[0] if len(frequency) > 1 else rate
    return {
        name: float(np.sum(psd[(frequency >= left) & (frequency <= right)]) * step)
        for name, (left, right) in bands.items()
        if right > left
    }


def _metrics(prepared: Prepared, options: GraphOptions) -> tuple[list[dict], list[dict]]:
    labels = (options.before_label if prepared.before is not None else None, options.after_label)
    values = [("after", prepared.after, labels[1])]
    if prepared.before is not None:
        values.insert(0, ("before", prepared.before, labels[0]))
    rows = []
    for role, values_array, label in values:
        rms = math.sqrt(power(values_array))
        peak = float(np.max(np.abs(values_array)))
        rows.extend(
            [
                {
                    "id": f"rms_{role}",
                    "label": f"{label} RMS",
                    "value": db(rms**2),
                    "unit": "dBFS",
                    "status": "warning" if rms < 1e-6 else "ok",
                    "detail": "完整对齐后的分析信号。",
                },
                {
                    "id": f"peak_{role}",
                    "label": f"{label} 峰值",
                    "value": db(peak, 20),
                    "unit": "dBFS",
                    "status": "warning" if peak >= 0.999 else "ok",
                    "detail": "数字满量程为 0 dBFS。",
                },
                {
                    "id": f"crest_{role}",
                    "label": f"{label} 峰值因数",
                    "value": db(peak / max(rms, 1e-15), 20),
                    "unit": "dB",
                    "status": "ok",
                    "detail": "峰值 / RMS；用于观察瞬态和压缩。",
                },
                {
                    "id": f"clipping_{role}",
                    "label": f"{label} 满幅比例",
                    "value": float(np.mean(np.abs(values_array) >= 0.999) * 100),
                    "unit": "%",
                    "status": "warning" if np.any(np.abs(values_array) >= 0.999) else "ok",
                    "detail": "幅值 ≥ 0.999 FS 的样本比例。",
                },
            ]
        )
    if prepared.before is not None and prepared.after is not None:
        before_power = _band_powers(prepared.before, prepared.rate)
        after_power = _band_powers(prepared.after, prepared.rate)
        rows.extend(
            [
                {
                    "id": "rms_delta",
                    "label": "RMS 变化",
                    "value": db(power(prepared.after) / max(power(prepared.before), 1e-20)),
                    "unit": "dB",
                    "status": "ok",
                    "detail": "处理后 RMS 相对处理前；包含增益、噪声和失真变化。",
                },
                {
                    "id": "peak_delta",
                    "label": "峰值变化",
                    "value": db(
                        np.max(np.abs(prepared.after)) / max(np.max(np.abs(prepared.before)), 1e-20),
                        20,
                    ),
                    "unit": "dB",
                    "status": "ok",
                    "detail": "输出峰值相对输入峰值。",
                },
            ]
        )
        band_rows = []
        labels_map = {
            "low_20_300": "低频 20–300 Hz",
            "mid_300_3000": "中频 300–3000 Hz",
            "high_3000_nyquist": "高频 3 kHz–Nyquist",
        }
        for key in before_power:
            after_value, before_value = after_power.get(key), before_power[key]
            band_rows.append(
                {
                    "id": key,
                    "label": labels_map[key],
                    "before_db": db(before_value),
                    "after_db": db(after_value),
                    "delta_db": db(after_value / max(before_value, 1e-20)),
                }
            )
        return rows, band_rows
    return rows, []


def analyze_graphics(before: Audio | None, after: Audio, options: GraphOptions) -> tuple[dict, dict[str, bytes]]:
    prepared = _prepare(before, after, options)
    duration = len(prepared.after) / prepared.rate
    time_offset = prepared.time_offset_seconds
    labels = (
        options.before_label if prepared.before is not None else None,
        options.after_label,
    )
    plot_labels = (
        _plot_label(labels[0], "Input / before"),
        _plot_label(labels[1], "Output / after"),
    )
    fft_size = _frame_size(prepared.rate, 0.032)
    hop = max(1, prepared.rate // 100)
    frequency, short_time, after_power = _stft_power(prepared.after, prepared.rate, fft_size, hop)
    before_power = None
    if prepared.before is not None:
        before_frequency, before_time, before_power = _stft_power(
            prepared.before, prepared.rate, fft_size, hop
        )
        if not np.array_equal(before_frequency, frequency):
            raise ValueError("STFT frequency grids differ")
        if len(before_time) != len(short_time):
            common = min(len(before_time), len(short_time))
            before_power, after_power = before_power[:, :common], after_power[:, :common]
            short_time = short_time[:common]

    log_frequency, after_log, log_edges = _log_bands(after_power, frequency, prepared.rate)
    before_log = _log_bands(before_power, frequency, prepared.rate)[1] if before_power is not None else None
    after_db = _db(after_log)
    before_db = _db(before_log) if before_log is not None else None
    level_low, level_high = (
        _shared_range(before_db, after_db) if before_db is not None else _shared_range(after_db)
    )

    figures: dict[str, bytes] = {
        "waveform-energy": _plot_waveform_energy(
            prepared, plot_labels
        )
    }
    if before_db is not None:
        figures["log-stft-before"] = _plot_log_spectrogram(
            np.linspace(time_offset, time_offset + duration, before_db.shape[1]),
            log_frequency,
            before_db,
            level_low,
            level_high,
            f"{plot_labels[0]} · log-frequency STFT",
        )
    figures["log-stft-after"] = _plot_log_spectrogram(
        np.linspace(time_offset, time_offset + duration, after_db.shape[1]),
        log_frequency,
        after_db,
        level_low,
        level_high,
        f"{plot_labels[1]} · log-frequency STFT",
    )

    if before_power is not None and before_log is not None:
        valid = np.maximum(before_log, after_log) > max(
            float(np.max(before_log)), float(np.max(after_log))
        ) * 1e-7
        delta = np.where(
            valid,
            np.clip(
                10.0 * np.log10(np.maximum(after_log, 1e-20) / np.maximum(before_log, 1e-20)),
                -30,
                30,
            ),
            np.nan,
        )
        figures["spectral-delta"] = _plot_delta(
            np.linspace(time_offset, time_offset + duration, delta.shape[1]), log_frequency, delta
        )
        plot_stft_time = np.linspace(
            time_offset, time_offset + duration, before_db.shape[1]
        )
        figures["triple"] = _plot_triple(
            prepared,
            plot_stft_time,
            log_frequency,
            before_db,
            after_db,
            delta,
            (plot_labels[0], plot_labels[1]),
        )

    frequency_ltas, after_psd = _ltas_density(prepared.after, prepared.rate)
    before_psd = _ltas_density(prepared.before, prepared.rate)[1] if prepared.before is not None else None
    ltas_frequency = np.geomspace(20, prepared.rate / 2, 220)
    after_ltas = np.interp(ltas_frequency, frequency_ltas, _db(after_psd), left=-140, right=-140)
    before_ltas = (
        np.interp(ltas_frequency, frequency_ltas, _db(before_psd), left=-140, right=-140)
        if before_psd is not None
        else None
    )
    figures["ltas"] = _plot_ltas(ltas_frequency, before_ltas, after_ltas, plot_labels)
    figures["erb"] = _plot_erb_pair(
        prepared, frequency, before_power, after_power, plot_labels
    )
    short_fft = _frame_size(prepared.rate, 0.008)
    short_hop = max(1, round(prepared.rate * 0.002))
    short_frequency, short_time2, after_short_power = _stft_power(
        prepared.after, prepared.rate, short_fft, short_hop
    )
    before_short = (
        _stft_power(prepared.before, prepared.rate, short_fft, short_hop)
        if prepared.before is not None
        else None
    )
    figures["high-time"] = _plot_high_time(
        prepared,
        before_short,
        (short_frequency, short_time2, after_short_power),
        plot_labels,
    )

    metrics, bands = _metrics(prepared, options)
    if before_power is not None:
        valid_ratio = np.where(
            np.isfinite(delta),
            np.clip(
                10.0
                * np.log10(
                    np.maximum(after_log, 1e-20) / np.maximum(before_log, 1e-20)
                ),
                -60,
                60,
            ),
            np.nan,
        )
        finite = valid_ratio[np.isfinite(valid_ratio)]
        attenuation = float(np.mean(finite < -1.0) * 100) if finite.size else None
        boost = float(np.mean(finite > 1.0) * 100) if finite.size else None
        metrics.extend(
            [
                {
                    "id": "spectral_median",
                    "label": "频谱变化中位数",
                    "value": float(np.median(finite)) if finite.size else None,
                    "unit": "dB",
                    "status": "ok" if finite.size else "unavailable",
                    "detail": "有效时频单元内，处理后相对处理前的 dB 中位数。",
                },
                {
                    "id": "spectral_reduced",
                    "label": "衰减时频单元",
                    "value": attenuation,
                    "unit": "%",
                    "status": "ok" if attenuation is not None else "unavailable",
                    "detail": "有效时频单元中变化小于 −1 dB 的比例。",
                },
                {
                    "id": "spectral_boosted",
                    "label": "增强时频单元",
                    "value": boost,
                    "unit": "%",
                    "status": "ok" if boost is not None else "unavailable",
                    "detail": "有效时频单元中变化大于 +1 dB 的比例。",
                },
            ]
        )

    sources = [
        {
            "role": "before",
            "name": before.name,
            "sha256": before.sha256,
            "original_sample_rate": before.rate,
            "original_channels": before.original_channels,
            "selected_channel": options.before_channel + 1,
            "format": before.format,
        }
        for role in ("before",)
        if before is not None
    ]
    sources.append(
        {
            "role": "after",
            "name": after.name,
            "sha256": after.sha256,
            "original_sample_rate": after.rate,
            "original_channels": after.original_channels,
            "selected_channel": options.after_channel + 1,
            "format": after.format,
        }
    )
    report = {
        "schema_version": 1,
        "scope": "paired audio visualization and engineering diagnostics",
        "standard_compliance": False,
        "analysis_rate": prepared.rate,
        "duration_seconds": duration,
        "source": sources,
        "alignment": prepared.alignment,
        "metrics": metrics,
        "bands": bands,
        "notices": prepared.notices,
        "figure_scope": {
            "stft_fft_size": fft_size,
            "stft_hop": hop,
            "power_calibration": "one_sided_mean_square_v2",
            "stft_power_unit": "FS^2/bin",
            "stft_frequency_reduction": "mean FFT-bin power within displayed band",
            "stft_integrated_power": "Hann-squared-weighted time-domain mean square",
            "frame_energy_hop_samples": hop,
            "frame_energy_method": "unwindowed sample mean square; includes DC; last frame not zero padded",
            "ltas_unit": "dBFS/Hz",
            "ltas_detrend": "constant per Welch segment",
            "time_offset_seconds": time_offset,
            "log_frequency_edge_min_hz": float(log_edges[0]),
            "log_frequency_edge_max_hz": float(log_edges[-1]),
            "high_time_fft_size": short_fft,
            "high_time_hop": short_hop,
        },
        "figures": [],
    }
    return report, figures
