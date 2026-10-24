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
import csv
import os
import secrets
import time
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
# Prefer the production host target containing all published chain options.
# ITU_CLI remains an explicit deployment override; legacy binaries are fallbacks.
CANDIDATE_BINARIES = (
    "build/host/cpp/itu_audio_cli",
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



SUPPORTED_PROFILES = {"general", "vcp", "vcp_joint", "vehicle_stream"}
CHAIN_CATALOG = ROOT / "src/features/enhancement-chain-configs.json"


def _chain_flags(profile: str, chain: dict | None) -> tuple[list[str], dict | None, str | None]:
    """Resolve trusted catalog choices, never client-supplied CLI arguments."""
    if profile not in SUPPORTED_PROFILES:
        raise AudioError("不支持的处理档位；请选择 vcp、vcp_joint、general 或 vehicle_stream")
    if chain is None:
        return [], None, None
    if not isinstance(chain, dict) or set(chain) - {"profile", "selection", "extra_flags"}:
        raise AudioError("链路配置只允许 profile、selection 和预设声明的 extra_flags")
    if chain.get("profile") != profile or profile not in {"general", "vcp", "vcp_joint"}:
        raise AudioError("链路档位与请求档位不一致或不支持")
    catalog_bytes = CHAIN_CATALOG.read_bytes()
    catalog = json.loads(catalog_bytes)
    stages = catalog["stages"]
    selection = chain.get("selection")
    if not isinstance(selection, dict) or set(selection) != {stage["id"] for stage in stages}:
        raise AudioError("链路必须明确选择 BF / AEC / NS / EQ / AGC / DRC / limiter 七级")
    flags = ["--no-rx-processing"]
    if profile == "vcp_joint":
        defaults = catalog.get("joint_stage_defaults", {}).get("drc")
        if not isinstance(defaults, list) or not defaults:
            raise AudioError("联合链路目录缺少 DRC 默认定义，请更新前后端配置")
        if selection.get("aec") != "speex" or selection.get("ns") != "super_gaussian":
            raise AudioError("vcp_joint 联合控制要求 Speex AEC 和 Super-Gaussian NS；切换其它后端请使用 vcp 档位")
        flags.extend(defaults)
    for stage in stages:
        selected = next((item for item in stage["options"] if item["id"] == selection[stage["id"]]), None)
        if selected is None:
            raise AudioError(f"不支持的 {stage['id']} 选项：{selection[stage['id']]}")
        flags.extend(selected["flags"])
    preset = next((p for p in catalog["presets"] if p["profile"] == profile and p["selection"] == selection), None)
    trusted_extra = preset.get("extra_flags", []) if preset else []
    if "extra_flags" in chain and chain["extra_flags"] != trusted_extra:
        raise AudioError("不接受自定义 CLI 参数；只允许链路目录中声明的预设参数")
    if profile == "general" and (preset is None or preset["id"] != "array_control"):
        raise AudioError("general 档位仅支持目录中的 DAS 对照预设")
    flags.extend(trusted_extra)
    canonical = {"profile": profile, "selection": dict(selection)}
    if trusted_extra:
        canonical["extra_flags"] = trusted_extra
    return flags, canonical, hashlib.sha256(catalog_bytes).hexdigest()


def _microphone_flags(channels: int, mode: str | None, delays, chain: dict | None, rate: int):
    if channels not in {1, 2, 4}:
        raise AudioError("引擎麦克风录音必须保留 1、2 或 4 个同步声道")
    if mode not in {None, "channel0", "beamform"}:
        raise AudioError("microphone_mode 只能是 channel0 或 beamform")
    if channels > 1 and mode is None:
        raise AudioError("多麦录音需明确选择第 1 麦克风，或提供逐通道标定延迟进行波束形成")
    mode = mode or "channel0"
    bf = chain["selection"]["bf"] if chain else None
    if mode == "channel0":
        if delays not in (None, []):
            raise AudioError("第 1 麦克风模式不能同时提供波束延迟")
        if chain and ((channels > 1 and bf != "off") or (channels == 1 and bf not in {"off", "das"})):
            raise AudioError("所选 BF 需要多麦标定：请选择 BF 旁路，或选择波束形成并提供标定延迟")
        return mode, ["--no-bf"]
    if channels == 1:
        raise AudioError("波束形成需要 2 或 4 个麦克风声道")
    if bf == "off":
        raise AudioError("波束形成模式不能与 BF 旁路组合")
    if not isinstance(delays, list) or len(delays) != channels:
        raise AudioError(f"请提供 {channels} 个逐通道标定延迟，单位为目标采样率下的样本")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 4096 for v in delays):
        raise AudioError("标定延迟必须为 0–4096 范围内的有限非负样本数")
    return mode, ["--beam-delays", ",".join(format(v, ".12g") for v in delays)]


def _joint_summary(path: Path, enabled: bool) -> dict:
    result = {"enabled": enabled, "reference_evidence_valid_frames": 0,
              "near_state_frames": {"unknown": 0, "echo_only": 0, "near_or_double_talk": 0},
              "multirender_frames": 0, "raw_mix_budget_max": 0.0,
              "raw_mix_applied_min": None, "raw_mix_applied_max": 0.0,
              "budget_violations": 0}
    if not path.is_file():
        return result
    with path.open() as handle:
        for row in csv.DictReader(handle):
            state = int(row.get("post_aec_near_state") or 0)
            result["near_state_frames"][{1: "echo_only", 2: "near_or_double_talk"}.get(state, "unknown")] += 1
            result["reference_evidence_valid_frames"] += int(float(row.get("post_aec_reference_evidence_valid") or 0) > .5)
            result["multirender_frames"] = max(result["multirender_frames"], int(row.get("post_aec_multirender_frames") or 0))
            budget = float(row.get("vehicle_ns_raw_mix_budget") or 0)
            lower, upper = (float(row.get(key) or 0) for key in ("vehicle_ns_raw_mix_min", "vehicle_ns_raw_mix_max"))
            result["raw_mix_budget_max"] = max(result["raw_mix_budget_max"], budget)
            result["raw_mix_applied_min"] = lower if result["raw_mix_applied_min"] is None else min(result["raw_mix_applied_min"], lower)
            result["raw_mix_applied_max"] = max(result["raw_mix_applied_max"], upper)
            result["budget_violations"] += int(enabled and (lower < -1e-6 or upper > budget + 1e-6))
    return result


def _resample(x: np.ndarray, source: int, target: int) -> np.ndarray:
    if source == target:
        return x
    common = math.gcd(source, target)
    return signal.resample_poly(x, target // common, source // common, axis=0)


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
        "mel": _mel_spectrogram(data, rate),
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


def _band_metrics(mic: np.ndarray, reference: np.ndarray, output: np.ndarray, rate: int) -> dict:
    """Engineering deltas that need no clean near-end reference.

    These are measured on the whole file, so a near-end segment is included in
    the ratio. They describe what the chain did to energy and spectrum; they are
    not ERLE, SI-SDR or any standardised score.
    """
    n = min(len(mic), len(output))
    if n < rate // 10 or power(mic[:n]) < 1e-12 or power(output[:n]) < 1e-12:
        return {"available": False, "reason": "有效长度或电平不足，无法比较处理前后能量"}
    level_change = float(10 * np.log10(max(power(output[:n]), 1e-20) / max(power(mic[:n]), 1e-20)))
    nperseg = int(min(2048, n))
    freq, p_mic = signal.welch(mic[:n], rate, nperseg=nperseg)
    _, p_out = signal.welch(output[:n], rate, nperseg=nperseg)
    edges = [(0, 200), (200, 400), (400, 800), (800, 1600), (1600, 3200), (3200, rate / 2)]
    bands = []
    for low, high in edges:
        if low >= rate / 2:
            continue
        mask = (freq >= low) & (freq < high)
        if not np.any(mask):
            continue
        before, after = float(np.sum(p_mic[mask])), float(np.sum(p_out[mask]))
        if before <= 1e-20:
            continue
        bands.append({
            "low_hz": low, "high_hz": min(high, rate / 2),
            "change_db": round(float(10 * np.log10(max(after, 1e-20) / before)), 2),
        })
    frame = max(1, rate // 100)
    frames = min(len(mic), len(reference)) // frame
    if frames:
        r = reference[:frames * frame].reshape(frames, frame)
        rms = np.sqrt(np.mean(r.astype(np.float64) ** 2, axis=1))
        active = float(np.mean(rms > 10 ** (-50 / 20))) * 100
    else:
        active = 0.0
    return {
        "available": True,
        "scope": "全程第1麦克风与固定时延补偿后的输出（含近端段），非 ERLE 或 SI-SDR",
        "level_change_db": round(level_change, 2),
        "band_change_db": bands,
        "reference_active_percent": round(active, 1),
    }



def _mel_spectrogram(x: np.ndarray, rate: int, bands: int = 40, max_frames: int = 100):
    """Compact mel spectrogram (40 bands, <=100 frames) for the demo heatmap."""
    n = len(x)
    if n < 1024:
        return None
    win = 512 if rate <= 16000 else 1024
    hop = max(1, n // max_frames)
    _, _, z = signal.stft(x, fs=rate, nperseg=win, noverlap=max(0, win - hop),
                          boundary=None, padded=False)
    power = np.abs(z) ** 2
    f = np.linspace(0, rate / 2, power.shape[0])

    def hz2mel(h):
        return 2595 * np.log10(1 + h / 700)

    def mel2hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    fmax = min(8000.0, rate / 2)
    points = mel2hz(np.linspace(hz2mel(20), hz2mel(fmax), bands + 2))
    bank = np.zeros((bands, len(f)))
    for b in range(bands):
        lo, ce, hi = points[b], points[b + 1], points[b + 2]
        bank[b] = np.clip(np.minimum((f - lo) / max(ce - lo, 1e-9),
                                     (hi - f) / max(hi - ce, 1e-9)), 0, None)
    mel = bank @ power
    db = np.clip(10 * np.log10(np.maximum(mel, 1e-12)), -80, 0)
    db = db[::-1]
    return {"bands": bands, "frames": int(db.shape[1]), "duration": round(n / rate, 3),
            "db": np.round(db, 1).tolist()}


def _quality_scores(clean: np.ndarray | None, output: np.ndarray, rate: int = 16000) -> dict:
    """SI-SDR and STOI against a clean near-end reference, when one is supplied."""
    if clean is None:
        return {"available": False, "reason": "未提供干净近端，无法计算 SI-SDR / STOI"}
    n = min(len(clean), len(output))
    c = clean[:n].astype(np.float64)
    o = output[:n].astype(np.float64)
    if n < rate // 10 or power(c) < 1e-12 or power(o) < 1e-12:
        return {"available": False, "reason": "干净近端或输出电平过低/长度不足"}
    scale = float(np.dot(o, c) / max(np.dot(c, c), 1e-20))
    projected = scale * c
    residual = o - projected
    si_sdr = float(10 * np.log10(max(np.sum(projected ** 2), 1e-20) / max(np.sum(residual ** 2), 1e-20)))
    stoi_value = None
    try:
        from pystoi import stoi as stoi_fn
        stoi_value = float(stoi_fn(c.astype(np.float32), o.astype(np.float32), rate, extended=False))
    except Exception:
        stoi_value = None
    if stoi_value is not None and not np.isfinite(stoi_value):
        stoi_value = None
    return {
        "available": True,
        "si_sdr_db": round(si_sdr, 2),
        "stoi": None if stoi_value is None else round(stoi_value, 4),
        "scope": "按引擎声明固定时延补偿后比较用户干净近端；不搜索最佳时延；STOI 使用实际采样率",
    }


def synthesize(
    microphone_name: str,
    microphone_blob: bytes,
    reference_name: str,
    reference_blob: bytes,
    rate: int | None = None,
    profile: str | None = None,
    clean_name: str | None = None,
    clean_blob: bytes | None = None,
    chain: dict | None = None,
    microphone_mode: str | None = None,
    beam_delays_samples: list[float] | None = None,
) -> dict:
    """Decode, align and run the engine. Returns metadata plus the output WAV."""
    engine_profile = profile or (chain.get("profile") if isinstance(chain, dict) else None) or os.environ.get("ITU_SYNTH_PROFILE", "vcp")
    chain_flags, canonical_chain, catalog_sha = _chain_flags(engine_profile, chain)
    microphone = decode("before", microphone_name, microphone_blob, channel=None)
    reference = decode("echo", reference_name, reference_blob, channel=None)
    target = int(rate) if rate else (microphone.rate if microphone.rate in SUPPORTED_RATES else DEFAULT_RATE)
    if target not in SUPPORTED_RATES:
        raise AudioError("引擎只支持 16 / 24 / 48 kHz，请选择其中一个采样率")
    mode, microphone_flags = _microphone_flags(microphone.original_channels, microphone_mode, beam_delays_samples, canonical_chain, target)
    mic = _resample(microphone.data, microphone.rate, target)[: target * MAX_SECONDS]
    ref = _resample(reference.data, reference.rate, target)[: target * MAX_SECONDS]
    frames = min(len(mic), len(ref))
    if frames < target // 10:
        raise AudioError("麦克音与参考音都至少需要 0.1 秒")
    if power(mic) < 1e-12:
        raise AudioError("麦克音是静音，无法生成输出")
    trimmed = (len(mic) - frames) / target + (len(ref) - frames) / target
    mic, ref = mic[:frames], ref[:frames]
    clean = None
    if clean_blob:
        decoded_clean = decode("after", clean_name or "clean.wav", clean_blob)
        clean = _resample(decoded_clean.data, decoded_clean.rate, target)[:frames]

    binary = cli_path()
    binary_sha256 = hashlib.sha256(binary.read_bytes()).hexdigest()
    timeout = int(os.environ.get("ITU_SYNTH_TIMEOUT", "300"))
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + secrets.token_hex(4)
    work = ROOT / "build" / "demo-runs" / run_id
    work.mkdir(parents=True, exist_ok=True)
    if True:
        sf.write(work / "mic.wav", mic.astype(np.float32), target, subtype="PCM_16")
        sf.write(work / "ref.wav", ref.astype(np.float32), target, subtype="PCM_16")
        prefix = work / "out"
        command = [
            str(binary), "--profile", engine_profile, *chain_flags,
            "--mic", str(work / "mic.wav"), "--echo", str(work / "ref.wav"),
            "--out", str(prefix), "--microphones", str(microphone.original_channels),
            *microphone_flags,
            "--diagnostic-stages",
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
                    "reference_channels", "beamforming_enabled", "beamforming_before_aec",
                    "speex_multirender_context", "vehicle_ns_unified_raw_budget",
                    "vehicle_ns_dry_max", "vehicle_ns_policy", "tx_drc_config", "tx_drc_enabled", "ns_backend",
                )
            }
        summary = {}
        frames_path = prefix.with_suffix(".frames.csv")
        if frames_path.is_file():
            echo, near = [], []
            with frames_path.open() as handle:
                for row in csv.DictReader(handle):
                    try:
                        aec = float(row.get("aec_rms") or 0)
                        out = float(row.get("out_rms") or 0)
                    except ValueError:
                        continue
                    if aec <= 1e-9 or out <= 1e-9:
                        continue
                    delta = 20 * np.log10(out / aec)
                    (echo if float(row.get("far_active") or 0) > 0.5 else near).append(delta)
            if echo:
                summary["echo_frames"] = len(echo)
                summary["echo_attenuation_db"] = round(float(np.mean(echo)), 2)
            if near:
                summary["near_frames"] = len(near)
                summary["near_change_db"] = round(float(np.mean(near)), 2)
            # Mixed averages hide the failure modes: echo onset, echo-only,
            # double talk and near-only need separate numbers.
            buckets = {"onset": [], "echo_only": [], "double_talk": [], "near_only": [], "noise_only": []}
            far_prev = 0.0
            for row in csv.DictReader(frames_path.open()):
                try:
                    aec = float(row.get("aec_rms") or 0)
                    out = float(row.get("out_rms") or 0)
                except ValueError:
                    continue
                if aec <= 1e-9 or out <= 1e-9:
                    continue
                delta = 20 * np.log10(out / aec)
                far_now = float(row.get("far_active") or 0)
                mix = float(row.get("nearend_mix") or 0)
                if far_now > 0.5:
                    if far_prev <= 0.5:
                        buckets["onset"].append(delta)
                    elif mix > 0.1:
                        buckets["double_talk"].append(delta)
                    else:
                        buckets["echo_only"].append(delta)
                elif mix > 0.5:
                    buckets["near_only"].append(delta)
                else:
                    buckets["noise_only"].append(delta)
                far_prev = far_now
            summary["segments"] = {
                name: {"frames": len(values), "change_db": round(float(np.mean(values)), 2)}
                for name, values in buckets.items() if values
            }
        joint = _joint_summary(frames_path, engine_profile == "vcp_joint")
        (work / "request.json").write_text(json.dumps({
            "profile": engine_profile, "sample_rate": target, "clean": bool(clean_blob),
            "microphone": microphone_name, "reference": reference_name,
            "seconds": frames / target, "chain": canonical_chain,
            "command": command, "binary_sha256": binary_sha256, "catalog_sha256": catalog_sha,
            "microphone_mode": mode, "beam_delays_samples": beam_delays_samples,
            "microphone_channels": microphone.original_channels, "reference_channels": reference.original_channels,
        }, ensure_ascii=False, indent=2))
        (work / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        buffer = BytesIO()
        sf.write(buffer, processed[:, 0], output_rate, format="WAV", subtype="PCM_16")

    output = processed[:, 0]
    declared_delay = int(engine_report.get("dsp_delay_samples") or 0)
    aligned_output = output[declared_delay:declared_delay + frames]
    return {
        "engine": "itu-engineering-dsp/2.0",
        "run_id": run_id,
        "diagnostics_dir": str(work),
        "frame_summary": summary,
        "profile": engine_profile,
        "binary": str(binary),
        "binary_sha256": binary_sha256,
        "chain": canonical_chain,
        "catalog_sha256": catalog_sha,
        "joint_control": joint,
        "sample_rate": target,
        "seconds": frames / target,
        "trimmed_seconds": round(trimmed, 6),
        "inputs": {
            "microphone": {
                "name": microphone.name, "sample_rate": microphone.rate,
                "channels": microphone.original_channels,
                "selected_channel": 1 if mode == "channel0" else None,
                "processed_channels": microphone.original_channels, "channel_mode": mode, **_levels(mic),
            },
            "reference": {
                "name": reference.name, "sample_rate": reference.rate,
                "channels": reference.original_channels,
                "selected_channel": None, "processed_channels": reference.original_channels,
                "channel_mode": "all_synchronous", **_levels(ref),
            },
        },
        "output": {
            "name": "processed.wav", "sample_rate": output_rate,
            "samples": int(len(output)), **_levels(output),
        },
        "engine_report": engine_report,
        "metrics": {**_band_metrics(mic[:, 0], np.sqrt(np.mean(ref ** 2, axis=1)), aligned_output, output_rate),
                    "before_scope": "microphone_channel_1",
                    "fixed_delay_samples": declared_delay,
                    "quality": _quality_scores(clean, aligned_output, output_rate)},
        # Waveform and spectrum for the two inputs and the generated output, so
        # the page can plot them without decoding audio in the browser.
        "charts": [
            _chart("before", microphone.name, mic[:, 0], target, microphone.original_channels,
                   microphone.sha256, microphone.byte_count),
            _chart("echo", reference.name, ref[:, 0], target, reference.original_channels,
                   reference.sha256, reference.byte_count),
            _chart("after", "processed.wav", output, output_rate, 1,
                   hashlib.sha256(buffer.getvalue()).hexdigest(), len(buffer.getvalue())),
        ],
        "graphics_context": {
            "work": str(work), "run_id": run_id, "binary_sha256": binary_sha256,
            "profile": engine_profile, "microphone_mode": mode,
            "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "mic_sha256": hashlib.sha256((work / "mic.wav").read_bytes()).hexdigest(),
            "after_sha256": hashlib.sha256(after_path.read_bytes()).hexdigest(),
        },
        "wav": buffer.getvalue(),
    }


def synthesis_graphics(context: dict):
    """Render this run's immutable PCM with its declared delay, without lag fitting."""
    from server.graph_engine.engine import GraphOptions, analyze_graphics
    work = Path(context["work"])
    mic_path, after_path = work / "mic.wav", work / "out.after.wav"
    for path, key in ((mic_path, "mic_sha256"), (after_path, "after_sha256"), (work / "out.json", "report_sha256")):
        if hashlib.sha256(path.read_bytes()).hexdigest() != context[key]:
            raise AudioError("本次回放音频已改变，请重新生成后再绘图")
    microphone, rate = sf.read(mic_path, dtype="float64", always_2d=True)
    after, output_rate = sf.read(after_path, dtype="float64", always_2d=True)
    engine = json.loads((work / "out.json").read_text())
    delay = engine.get("dsp_delay_samples")
    if rate != output_rate or isinstance(delay, bool) or not isinstance(delay, int) or delay < 0 or delay >= len(after):
        raise AudioError("引擎固定时延元数据无效，不能生成对齐图")
    common = min(len(microphone), len(after) - delay)
    before_audio = Audio("before", "mic.wav · channel 1", microphone[:common, 0], rate,
                         microphone.shape[1], 0, "WAV", context["mic_sha256"], mic_path.stat().st_size)
    after_audio = Audio("after", "processed.wav", after[delay:delay + common, 0], rate,
                        1, 0, "WAV", context["after_sha256"], after_path.stat().st_size)
    report, figures = analyze_graphics(before_audio, after_audio, GraphOptions(
        sample_rate=rate, align=False, before_label="第1麦克风输入", after_label="本次链路输出"))
    report["alignment"] = {"requested": False, "applied": True, "method": "declared_dsp_delay",
        "delay_ms": delay * 1000 / rate, "correlation": None, "reliable": None,
        "note": "按本次引擎报告固定时延补偿输出；未搜索相关峰，未调整电平。"}
    report["notices"] = [note for note in report["notices"] if "共同起点" not in note]
    report["notices"].extend([
        f"本次输出按引擎声明固定补偿 {delay} 样本（{delay * 1000 / rate:.3f} ms），无最佳时延搜索或电平归一化。",
        "处理前显示上传录音第1麦克风，保留原电平；多麦波束形成时它不等同于校准DAS目标，差分图只观察电平与频谱变化。",
        "播放参考的全部同步声道均用于执行；输入预览仅显示参考第1声道，不代表执行被截断。",
    ])
    report["synthesis"] = {key: value for key, value in context.items() if key != "work"}
    report["synthesis"].update({"fixed_delay_samples": delay, "before_scope": "microphone_channel_1",
                               "level_normalized": False, "best_lag_search": False})
    return report, figures


# ------------------------------------------------------------------ demo pairs
# Deterministic synthetic fixtures. They are diagnostic material, never claimed
# to be real speech or real vehicle recordings.
DEMO_RATE = 16000
DEMO_SECONDS = 8.0
DEMOS = (
    {
        "id": "echo_and_noise",
        "label": "回声 + 车噪风噪",
        "description": "近端单讲、远端单讲与双讲依次出现，麦克风含声学回声、车内路噪与车窗阵风噪声。",
    },
    {
        "id": "near_only",
        "label": "仅近端 + 车噪风噪",
        "description": "参考音为数字静音，麦克风含近端语音、车内路噪与车窗风噪。",
    },
    {
        "id": "wrong_reference",
        "label": "参考不可靠",
        "description": "参考音与麦克风内容不相关，用于观察车噪风噪与回声共存时的参考可靠性保护。",
    },
    {
        "id": "far_end_stops",
        "label": "远端中途停止",
        "description": "远端播放至 2.6 秒停止，用于观察车噪风噪底噪下的回声尾迹与保护释放。",
    },
)
DEMO_IDS = tuple(item["id"] for item in DEMOS)
_NOISE_DIR = ROOT / "datasets" / "itu-eval-current" / "sources" / "noise"
_NOISE_CACHE: dict[str, np.ndarray] = {}


def _rms_scale(x: np.ndarray, target_dbfs: float) -> np.ndarray:
    p = max(float(np.mean(x * x)), 1e-12)
    return x * math.sqrt((10.0 ** (target_dbfs / 10.0)) / p)


def _source_slice(filename: str, offset: int, length: int) -> np.ndarray | None:
    if filename not in _NOISE_CACHE:
        path = _NOISE_DIR / filename
        if not path.is_file():
            return None
        audio, rate = sf.read(path, dtype="float64", always_2d=True)
        mono = audio[:, 0]
        if rate != DEMO_RATE:
            mono = _resample(mono, rate, DEMO_RATE)
        _NOISE_CACHE[filename] = mono
    source = _NOISE_CACHE[filename]
    if len(source) < length:
        return None
    start = offset % max(len(source) - length, 1)
    return source[start : start + length].copy()


def _cabin_and_wind_noise(rng, index: int, length: int) -> np.ndarray:
    """Build audible cabin road rumble + HVAC airflow + modulated window wind gusts."""
    t = np.arange(length) / DEMO_RATE
    road_raw = _source_slice("TCAR-ch01.wav", 160000 + index * 48000, length)
    if road_raw is not None:
        road = signal.sosfilt(
            signal.butter(2, 45.0, fs=DEMO_RATE, btype="highpass", output="sos"),
            road_raw,
        )
    else:
        road = signal.sosfilt(
            signal.butter(2, [45.0, 320.0], fs=DEMO_RATE, btype="bandpass", output="sos"),
            rng.normal(size=length),
        )
    air_raw = _source_slice("MS-SNSD-AirConditioner_3.wav", 32000 + index * 24000, length)
    if air_raw is not None:
        air = signal.sosfilt(
            signal.butter(2, 70.0, fs=DEMO_RATE, btype="highpass", output="sos"),
            air_raw,
        )
    else:
        air = signal.sosfilt(
            signal.butter(2, [90.0, 1800.0], fs=DEMO_RATE, btype="bandpass", output="sos"),
            rng.normal(size=length),
        )
    gust_env = 0.55 + 0.45 * np.sin(2.0 * np.pi * 0.45 * t + 0.3 + 0.5 * index) ** 2
    wind_low = (
        signal.sosfilt(
            signal.butter(2, [45.0, 260.0], fs=DEMO_RATE, btype="bandpass", output="sos"),
            rng.normal(size=length),
        )
        * gust_env
    )
    wind_mid = (
        signal.sosfilt(
            signal.butter(2, [200.0, 1500.0], fs=DEMO_RATE, btype="bandpass", output="sos"),
            rng.normal(size=length),
        )
        * (0.60 + 0.40 * gust_env)
    )
    combined = (
        _rms_scale(road, -26.5)
        + _rms_scale(air, -28.0)
        + _rms_scale(wind_low, -26.5)
        + _rms_scale(wind_mid, -27.5)
    )
    return _rms_scale(combined, -24.5)


def _band(rng, length: int, low: float, high: float, gain: float) -> np.ndarray:
    """Band-limited noise. Used for the cabin noise floor, never for speech."""
    from scipy import signal as _signal
    data = _signal.sosfilt(
        _signal.butter(4, [low, high], fs=DEMO_RATE, btype="bandpass", output="sos"),
        rng.normal(size=length),
    )
    return data / max(float(np.max(np.abs(data))), 1e-12) * gain


# Formant triples (F1, F2, F3) in Hz for five cardinal vowels. The usual
# measured adult-male values; a diagnostic fixture does not need more than this.
_VOWELS = (
    ("a", 730.0, 1090.0, 2440.0),
    ("e", 530.0, 1840.0, 2480.0),
    ("i", 270.0, 2290.0, 3010.0),
    ("o", 570.0, 840.0, 2410.0),
    ("u", 300.0, 870.0, 2240.0),
)


def _resonator(freq: float, bandwidth: float = 90.0):
    """One two-pole resonator normalised to unity gain at DC."""
    r = math.exp(-math.pi * bandwidth / DEMO_RATE)
    theta = 2.0 * math.pi * freq / DEMO_RATE
    a1 = -2.0 * r * math.cos(theta)
    a2 = r * r
    return 1.0 + a1 + a2, (1.0, a1, a2)


def _syllable(rng, seconds: float, f0: float, gain: float) -> np.ndarray:
    """One voiced syllable: a glottal pulse train shaped by three formants."""
    from scipy import signal as _signal
    n = max(int(seconds * DEMO_RATE), 1)
    t = np.arange(n) / DEMO_RATE

    # Pitch falls slightly across the syllable and drifts, so the result is a
    # voice rather than a flat buzz.
    drift = np.convolve(rng.normal(size=n), np.ones(64) / 64.0, mode="same")
    contour = f0 * (1.0 - 0.10 * t / max(seconds, 1e-6)) * (1.0 + 0.03 * drift)
    phase = np.cumsum(2.0 * np.pi * contour / DEMO_RATE)
    pulses = np.zeros(n)
    pulses[np.flatnonzero(np.diff(np.floor(phase / (2.0 * np.pi))) >= 1) + 1] = 1.0

    vowel = _VOWELS[int(rng.integers(0, len(_VOWELS)))]
    voiced = pulses
    for freq in vowel[1:]:
        b, a = _resonator(freq)
        voiced = _signal.lfilter([b], a, voiced)
    voiced = voiced / max(float(np.max(np.abs(voiced))), 1e-12)

    # A decaying noise burst stands in for the consonant that opens a syllable.
    onset = _band(rng, n, 1800.0, 6500.0, 1.0) * np.exp(-t / 0.035) * 0.35

    envelope = np.minimum(1.0, t / 0.020) * np.minimum(1.0, (seconds - t) / 0.045)
    syllable = (voiced + onset) * np.clip(envelope, 0.0, 1.0)
    level = gain * float(rng.uniform(0.70, 1.0))
    return syllable / max(float(np.max(np.abs(syllable))), 1e-12) * level


def _voice(rng, length: int, start: float, end: float, f0: float, gain: float) -> np.ndarray:
    """Fill [start, end) with deterministic voiced syllables separated by gaps."""
    out = np.zeros(length)
    cursor = float(start)
    while True:
        seconds = float(rng.uniform(0.14, 0.26))
        if cursor + seconds > end - 0.02:
            seconds = end - 0.02 - cursor
        if seconds < 0.09:
            break
        piece = _syllable(rng, seconds, f0, gain)
        head = int(round(cursor * DEMO_RATE))
        stop = min(head + len(piece), length)
        if stop > head:
            out[head:stop] += piece[: stop - head]
        cursor += seconds + float(rng.uniform(0.02, 0.07))
    return out


def _demo_signals(scenario: str) -> tuple[np.ndarray, np.ndarray]:
    index = DEMO_IDS.index(scenario)
    rng = np.random.default_rng(200 + index)
    length = int(DEMO_SECONDS * DEMO_RATE)

    noise = _cabin_and_wind_noise(rng, index, length)
    # Two talkers at different pitch, so the echo is not just a delayed copy of
    # the near end and the canceller has something real to track.
    # Leading/mid/trailing noise-only windows (0-0.85s, 2.0-3.3s, 6.8-8.0s)
    # give the noise estimator and hangover clean windows so both cabin/wind
    # noise and its suppression are clearly visible and audible.
    near = (
        _voice(rng, length, 0.85, 2.00, 118.0, 0.38)
        + _voice(rng, length, 4.35, 5.30, 118.0, 0.38)
        + _voice(rng, length, 5.70, 6.80, 118.0, 0.38)
    )
    far = _voice(rng, length, 3.30, 5.30, 205.0, 0.34)
    far_short = _voice(rng, length, 0.85, 2.60, 205.0, 0.34)
    unrelated = _voice(rng, length, 1.20, 4.80, 168.0, 0.34)
    # A 40 ms acoustic path is long enough to require real delay handling.
    echo = np.r_[np.zeros(640), far[:-640]] * 0.65

    if scenario == "echo_and_noise":
        # near 0.85-2.0, far 3.3-5.3, double talk 4.35-5.3, near 5.7-6.8
        reference, microphone = far, echo + near + noise
    elif scenario == "near_only":
        reference, microphone = np.zeros(length), near + noise
    elif scenario == "wrong_reference":
        reference = unrelated
        microphone = echo + near + noise
    elif scenario == "far_end_stops":
        # The far end stops at 2.6 s and the near end only starts at 3.4 s, so
        # the echo tail and the protection release are easy to hear alone.
        near_after = _voice(rng, length, 3.40, 6.60, 118.0, 0.38)
        echo_short = np.r_[np.zeros(640), far_short[:-640]] * 0.65
        reference = far_short
        microphone = echo_short + near_after + noise
    else:
        raise AudioError(f"未知示例场景：{scenario}")
    return microphone.astype(np.float32), reference.astype(np.float32)


def _demo_clean(scenario: str) -> np.ndarray:
    """Deterministic near-end reference for a built-in scenario (no echo, no noise)."""
    index = DEMO_IDS.index(scenario)
    rng = np.random.default_rng(200 + index)
    length = int(DEMO_SECONDS * DEMO_RATE)
    _cabin_and_wind_noise(rng, index, length)
    return (
        _voice(rng, length, 0.85, 2.00, 118.0, 0.38)
        + _voice(rng, length, 4.35, 5.30, 118.0, 0.38)
        + _voice(rng, length, 5.70, 6.80, 118.0, 0.38)
    )


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
