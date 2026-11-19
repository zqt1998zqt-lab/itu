"""Supplemental reference metrics for verified scene-condition TX artifacts.

These are derived views, never replacements for persisted contract/MOS records.
Speech retention measures are engineering proxies, not phoneme annotations.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.metadata
import math
from pathlib import Path
from threading import BoundedSemaphore
from typing import Any

import numpy as np
import soundfile as sf
from scipy.ndimage import maximum_filter1d

SCHEMA = "scene-condition-ns-quality-v1"
_COMPUTE_SLOTS = BoundedSemaphore(2)
_EPS = 1e-20


def _metric(value: float | None = None, *, status: str = "measured",
            reason: str = "", display: str | None = None) -> dict[str, Any]:
    if value is not None and not math.isfinite(float(value)):
        raise ValueError("NS quality metrics must contain finite JSON values")
    result: dict[str, Any] = {"value": value, "status": status, "reason": reason}
    if display is not None:
        result["display"] = display
    return result


def _missing(reason: str, status: str = "not_applicable") -> dict[str, Any]:
    return _metric(status=status, reason=reason)


def _db_ratio(numerator: float, denominator: float) -> dict[str, Any]:
    if numerator <= _EPS and denominator <= _EPS:
        return _missing("分子与分母均为数字静音，比例未定义", "undefined")
    if denominator <= _EPS:
        return _metric(display="+∞", reason="分母为零，理论值为正无穷")
    if numerator <= _EPS:
        return _metric(display="−∞", reason="分子为零，理论值为负无穷")
    return _metric(10.0 * math.log10(numerator / denominator))


def _rms_dbfs(samples: np.ndarray) -> dict[str, Any]:
    rms = float(np.sqrt(np.mean(np.square(samples))))
    result = (_metric(20.0 * math.log10(rms)) if rms > 0
              else _metric(display="−∞", reason="数字静音，线性 RMS 为零"))
    result["linear_rms"] = rms
    return result


def _scale_invariant(reference: np.ndarray, output: np.ndarray,
                     *, zero_mean: bool) -> dict[str, Any]:
    ref, out = reference.copy(), output.copy()
    if zero_mean:
        ref -= ref.mean()
        out -= out.mean()
    energy = float(np.dot(ref, ref))
    if energy <= _EPS:
        return _missing("近端参考无有效能量")
    if float(np.dot(out, out)) <= _EPS:
        return _missing("输出为数字静音，尺度不变指标未定义", "undefined")
    target = ref * (float(np.dot(ref, out)) / energy)
    return _db_ratio(float(np.dot(target, target)), float(np.dot(out - target, out - target)))


def _frames(samples: np.ndarray, window: int, hop: int) -> np.ndarray:
    return np.lib.stride_tricks.sliding_window_view(samples, window)[::hop]


def _activity(samples: np.ndarray, window: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    power = np.mean(np.square(_frames(samples, window, hop)), axis=1)
    p95 = float(np.percentile(power, 95))
    threshold = max(1e-8, p95 * 10.0 ** (-35.0 / 10.0))
    return power, power > threshold


def _loss_metrics(reference: np.ndarray, output: np.ndarray, active: np.ndarray,
                  ref_power: np.ndarray, window: int, hop: int) -> dict[str, dict[str, Any]]:
    keys = ("weak_speech_retention_pct", "onset_loss_db", "tail_loss_db", "speech_truncation_rate_pct")
    if not np.any(active):
        return {key: _missing("没有可用的近端语音活动帧") for key in keys}
    rf, yf = _frames(reference, window, hop), _frames(output, window, hop)
    rf = rf - rf.mean(axis=1, keepdims=True)
    yf = yf - yf.mean(axis=1, keepdims=True)
    ref_energy = np.sum(rf * rf, axis=1)
    gains = np.sum(rf * yf, axis=1) / np.maximum(ref_energy, _EPS)
    valid = active & (ref_energy > _EPS)
    if not np.any(valid):
        return {key: _missing("参考活动帧仅含直流或没有有效交流能量") for key in keys}
    overall = float(np.sum(gains[valid] * ref_energy[valid]) / np.sum(ref_energy[valid]))
    silent = float(np.mean(output * output)) <= _EPS
    if abs(overall) <= 1e-8 and not silent:
        return {key: _missing("输出与参考相关性过低，不能可靠估计语音保持", "undefined") for key in keys}
    relative = np.zeros_like(gains) if silent else np.abs(gains / overall)
    # Bridge <=100 ms internal gaps before choosing the first/last 80 ms.
    bridged = active.copy()
    positions = np.flatnonzero(active)
    for left, right in zip(positions[:-1], positions[1:]):
        if right - left <= max(1, round(0.1 * 16000 / hop)):
            bridged[left:right + 1] = True
    changes = np.diff(np.pad(bridged.astype(np.int8), (1, 1)))
    starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    onset, tail = np.zeros_like(active), np.zeros_like(active)
    edge_frames = max(1, round(0.08 * 16000 / hop))
    for start, end in zip(starts, ends):
        onset[start:min(end, start + edge_frames)] = True
        tail[max(start, end - edge_frames):end] = True
    weak = valid & (ref_power < float(np.percentile(ref_power, 95)) * 0.01)

    def retained(mask: np.ndarray) -> float | None:
        selected = mask & valid
        if not np.any(selected):
            return None
        return float(np.sum(relative[selected] ** 2 * ref_energy[selected]) / np.sum(ref_energy[selected]))

    weak_ratio = retained(weak)
    result = {"weak_speech_retention_pct": (
        _metric(100.0 * weak_ratio) if weak_ratio is not None
        else _missing("没有低于参考 P95 能量 20 dB 的有效弱语音帧")
    )}
    for key, mask in (("onset_loss_db", onset), ("tail_loss_db", tail)):
        ratio = retained(mask)
        result[key] = (_db_ratio(1.0, ratio) if ratio is not None
                       else _missing("没有可用的起音或尾音活动窗"))
    result["speech_truncation_rate_pct"] = _metric(100.0 * float(np.mean(relative[valid] < 0.1)))
    result["speech_truncation_rate_pct"]["eligible_frames"] = int(np.count_nonzero(valid))
    result["weak_speech_retention_pct"]["eligible_frames"] = int(np.count_nonzero(weak))
    return result


def calculate_metrics(reference: np.ndarray, microphone: np.ndarray,
                      transmitted: np.ndarray, far_end: np.ndarray, *,
                      sample_rate: int, analysis_start_s: float = 0.0,
                      residual_alignment: bool = True) -> dict[str, Any]:
    """Compare fixed-delay-aligned TX to the clean reference and mic channel 0."""
    arrays = [np.asarray(x, dtype=np.float64) for x in (reference, microphone, transmitted, far_end)]
    if any(x.ndim != 1 or not len(x) or not np.isfinite(x).all() for x in arrays):
        raise ValueError("指标输入必须为非空、有限的单通道音频")
    if sample_rate != 16000 or len({len(x) for x in arrays}) != 1:
        raise ValueError("补充指标当前要求四路音频等长，且均为 16 kHz")
    if not math.isfinite(analysis_start_s) or not 0 <= analysis_start_s < len(arrays[0]) / sample_rate:
        raise ValueError("分析起点不在音频范围内")
    first = round(analysis_start_s * sample_rate)
    reference, microphone, transmitted, far_end = [x[first:] for x in arrays]
    window, hop = 320, 160
    if len(reference) < window:
        raise ValueError("分析片段短于 20 ms")
    _, active = _activity(reference, window, hop)
    near_samples = int(np.count_nonzero(active) * hop)
    before: dict[str, dict[str, Any]] = {}
    after: dict[str, dict[str, Any]] = {}
    lag, correlation = 0, None
    if residual_alignment and near_samples >= 1600:
        from .conversation_diagnostics import _bounded_alignment
        aligned = _bounded_alignment(reference, transmitted)
        if aligned is not None:
            lag, correlation = int(aligned["lag_samples"]), float(aligned["correlation"])
    # A separate alignment is required for mic channel 0 (acoustic arrival time).
    mic_lag = 0
    if residual_alignment and near_samples >= 1600:
        from .conversation_diagnostics import _bounded_alignment
        mic_alignment = _bounded_alignment(reference, microphone)
        if mic_alignment is not None:
            mic_lag = int(mic_alignment["lag_samples"])
    # Both columns use exactly the same reference samples after bounded alignment.
    common_start = max(0, -lag, -mic_lag)
    common_end = min(len(reference), len(transmitted) - lag, len(microphone) - mic_lag)
    reference = reference[common_start:common_end]
    microphone = microphone[common_start + mic_lag:common_end + mic_lag]
    transmitted = transmitted[common_start + lag:common_end + lag]
    far_end = far_end[common_start:common_end]
    ref_power, active = _activity(reference, window, hop)
    _, far_active = _activity(far_end, window, hop)
    near_samples = int(np.count_nonzero(active) * hop)
    guarded = maximum_filter1d((active | far_active).astype(np.uint8), size=21, mode="constant")
    quiet = np.repeat(guarded == 0, hop)
    quiet = np.pad(quiet, (0, len(reference) - len(quiet)), constant_values=False)
    quiet_count = int(np.count_nonzero(quiet))

    for output, ref, destination in ((microphone, reference, before), (transmitted, reference, after)):
        if near_samples < 4000:
            for key in ("pesq", "stoi", "si_sdr_db", "si_snr_db"):
                destination[key] = _missing("近端有效语音不足 250 ms，不计算语音质量或重建指标")
        else:
            destination["si_sdr_db"] = _scale_invariant(ref, output, zero_mean=False)
            destination["si_snr_db"] = _scale_invariant(ref, output, zero_mean=True)
            try:
                from pesq import pesq
                destination["pesq"] = _metric(float(pesq(sample_rate, ref, output, "wb")))
            except ImportError:
                destination["pesq"] = _missing("未安装 PESQ 计算引擎", "unavailable")
            except Exception as error:
                destination["pesq"] = _missing(f"PESQ 无法计算：{str(error)[:160]}", "undefined")
            try:
                from pystoi import stoi
                destination["stoi"] = _metric(float(stoi(ref, output, sample_rate, extended=False)))
            except ImportError:
                destination["stoi"] = _missing("未安装 STOI 计算引擎", "unavailable")
            except Exception as error:
                destination["stoi"] = _missing(f"STOI 无法计算：{str(error)[:160]}", "undefined")
        destination["polqa"] = _missing("未配置 POLQA / P.863 引擎，PESQ 数值不作替代", "unavailable")
        pwr, act = _activity(ref, window, hop)
        destination.update(_loss_metrics(ref, output, act, pwr, window, hop))
        destination["clipping_rate_pct"] = _metric(100.0 * float(np.mean(np.abs(output) >= 1.0)))
    if quiet_count >= 1600:
        input_noise, output_noise = microphone[quiet], transmitted[quiet]
        in_power, out_power = float(np.mean(input_noise ** 2)), float(np.mean(output_noise ** 2))
        before["noise_attenuation_db"] = (_metric(0.0) if in_power > _EPS else _missing("输入噪声为数字静音，衰减量未定义", "undefined"))
        after["noise_attenuation_db"] = _db_ratio(in_power, out_power)
        before["residual_noise_rms_dbfs"], after["residual_noise_rms_dbfs"] = _rms_dbfs(input_noise), _rms_dbfs(output_noise)
    else:
        for key in ("noise_attenuation_db", "residual_noise_rms_dbfs"):
            before[key] = after[key] = _missing("没有至少 100 ms、避开近端和远端语音边缘的纯噪声观察窗")
    return {
        "before": before, "after": after,
        "analysis": {
            "start_s": (first + common_start) / sample_rate,
            "end_s": (first + common_end) / sample_rate,
            "sample_rate": sample_rate, "near_activity_s": near_samples / sample_rate,
            "noise_observation_s": quiet_count / sample_rate,
            "common_reference_samples": len(reference),
            "excluded_leading_samples": common_start,
            "excluded_trailing_samples": len(arrays[0]) - first - common_end,
            "residual_reference_lag_samples": lag, "mic_reference_lag_samples": mic_lag,
            "reference_tx_correlation": correlation,
            "alignment": "已扣除声明的 DSP 固定延迟；另作 ±100 ms 有界参考相关对齐（工程诊断）",
            "activity": "20 ms 帧 / 10 ms 帧移；参考 P95 能量 −35 dB，绝对下限 −80 dBFS；噪声窗避开两端语音各 100 ms",
            "retention": "20 ms 参考投影增益，相对整段语音增益归一；弱音低于参考 P95 能量 20 dB",
            "edges": "合并不超过 100 ms 的活动间隙，取每段首尾 80 ms 的相对投影能量损失",
            "truncation": "参考活动帧的相对投影幅度低于 0.1（−20 dB）的帧比例；是吞音代理，不是人工音素标注",
            "si_sdr": "不去均值，投影到参考的尺度不变信号失真比",
            "si_snr": "参考与输出分别去均值后计算尺度不变信噪比",
        },
    }


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_audio(path: Path, expected_sha: str) -> tuple[np.ndarray, int]:
    if path.is_symlink() or not path.is_file() or _sha(path) != expected_sha:
        raise ValueError("补充指标音频哈希或文件类型校验失败")
    audio, rate = sf.read(path, dtype="float64", always_2d=True)
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError("补充指标音频为空或含非有限值")
    return audio[:, 0], int(rate)


@lru_cache(maxsize=128)
def _calculate_cached(paths: tuple[str, ...], hashes: tuple[str, ...],
                      start: float) -> dict[str, Any]:
    with _COMPUTE_SLOTS:
        loaded = [_verified_audio(Path(path), digest) for path, digest in zip(paths, hashes, strict=True)]
        if len({rate for _, rate in loaded}) != 1:
            raise ValueError("补充指标音频采样率不一致")
        reference, mic, tx, far = [audio for audio, _ in loaded]
        return calculate_metrics(reference, mic, tx, far, sample_rate=loaded[0][1], analysis_start_s=start)


def case_ns_metrics(identifier: str, case_id: str) -> dict[str, Any]:
    from . import scene_condition_batch as batches
    from .acceptance_scene_condition_store import asset_path, load_scene_condition_dataset
    snapshot = batches.get_batch(identifier)
    if snapshot is None:
        raise FileNotFoundError("场景条件批次不存在")
    case = next((x for x in snapshot["cases"] if x["case_id"] == case_id), None)
    if case is None or case.get("status") != "completed":
        raise FileNotFoundError("该组 TX 尚未完成")
    dataset = load_scene_condition_dataset(snapshot["dataset_id"])
    if (dataset["dataset_manifest_sha256"] != snapshot["dataset_manifest_sha256"]
            or dataset["dataset_assets_sha256"] != snapshot["dataset_assets_sha256"]):
        raise ValueError("补充指标数据集身份与测试批次不一致")
    dataset_case = next(x for x in dataset["cases"] if x["case_id"] == case_id)
    tx, tx_sha = batches.output_path(identifier, case_id, "tx")
    roles = ("reference", "microphone", "far_end")
    sources = {role: asset_path(dataset["root"], dataset_case["signals"][role]) for role in roles}
    expected = {role: case["signals_sha256"][role] for role in roles}
    for role in roles:
        if _sha(sources[role]) != expected[role]:
            raise ValueError(f"补充指标 {role} 身份校验失败")
    dynamic = case.get("execution", {}).get("dynamic_microphone")
    if isinstance(dynamic, dict):
        effective = case.get("effective_input_sha256") or case["execution"].get("effective_input_sha256")
        if not isinstance(effective, dict):
            raise ValueError("动态麦克风缺少实际输入哈希")
        sources["microphone"], expected["microphone"] = batches.dynamic_microphone_path(identifier, case_id)
        for role in ("reference",):
            sources[role] = Path(dynamic[f"{role}_path"])
            expected[role] = effective[role]
            # Only accept a frozen runtime reference in the case or dataset tree.
            resolved = sources[role].resolve(strict=True)
            roots = (batches._case_directory(identifier, case_id).resolve(), Path(dataset["root"]).resolve())
            if not any(resolved.is_relative_to(root) for root in roots):
                raise ValueError("动态参考音频越出条件组或数据集目录")
        sources["far_end"], expected["far_end"] = batches.output_path(identifier, case_id, "rx")
    paths = (str(sources["reference"]), str(sources["microphone"]), str(tx), str(sources["far_end"]))
    hashes = (expected["reference"], expected["microphone"], tx_sha, expected["far_end"])
    # Recheck source identities even when returning a memoized calculation.
    for path, digest in zip(paths, hashes, strict=True):
        if _sha(Path(path)) != digest:
            raise ValueError("补充指标实际音频已改变")
    start = float(dataset_case.get("provenance", {}).get("analysis_start_s", 0.0))
    result = _calculate_cached(paths, hashes, start)
    versions = {}
    for name in ("pesq", "pystoi"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "schema_version": SCHEMA, "status": "complete", "batch_id": identifier,
        "case_id": case_id, "dataset_id": snapshot["dataset_id"],
        "dataset_manifest_sha256": snapshot["dataset_manifest_sha256"],
        "input_sha256": expected, "output_sha256": tx_sha,
        "scope": "整条 TX 链输出相对近端参考；包含 BF/AEC/NS/EQ/AGC/DRC 的影响，不代表 NS 单级",
        "microphone_channel": 0, "formal_compliance": False,
        "engine_versions": versions, **result,
    }
