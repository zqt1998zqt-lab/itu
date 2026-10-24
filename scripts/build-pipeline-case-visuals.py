#!/usr/bin/env python3
"""Render auditable input/output waveform and Mel SVG assets.

Each figure is built directly from a frozen WAV pair.  Output samples are
shifted by their declared causal delay, then both signals are cropped to the
same scene-time window.  No gain normalization is applied.
"""

from __future__ import annotations

import hashlib
import html
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "public/evaluation/pipeline-cases"
WAVE_WIDTH = 1200
WAVE_HEIGHT = 570
MEL_WIDTH = 1200
MEL_HEIGHT = 650
ENVELOPE_BINS = 900
WAVEFORM_Y_SCALE = 0.9
FFT_SIZE = 512
HOP_SIZE = 160
MEL_BANDS = 64
MEL_DB_RANGE = 80.0
MEL_TIME_COLUMNS = 240


@dataclass(frozen=True)
class CaseSpec:
    asset_id: str
    case_id: str
    detail: str
    before: Path
    after: Path
    sample_rate: int
    window_seconds: tuple[float, float]
    delay_samples: int
    before_label: str
    after_label: str
    before_channel: int = 0
    after_channel: int = 0
    before_crop_samples: int = 0
    analysis: str = ""


def repo_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


BF_ROOT = ROOT / "build/bf-algorithm-matrix-20261021-r2"
BF_MANIFEST = ROOT / "datasets/itu-vehicle-fields-float-20261010/inputs-manifest.json"
DRC_ROOT = ROOT / "build/drc-algorithm-matrix-20261022-r1"


def bf_case_spec(
    asset_id: str,
    variant: str,
    scene_id: str,
    detail: str,
    analysis: str,
    after_label: str,
) -> CaseSpec:
    manifest = json.loads(BF_MANIFEST.read_text(encoding="utf-8"))
    scene = next(item for item in manifest["scenes"] if item["id"] == scene_id)
    window_name = "near_late" if "near_late" in scene["windows"] else "near_only"
    window = scene["windows"][window_name]
    engine = json.loads(
        (
            BF_ROOT / variant / "cases" / scene_id / "runs/full/result.json"
        ).read_text(encoding="utf-8")
    )
    return CaseSpec(
        asset_id,
        scene_id,
        detail,
        BF_ROOT / "no_bf/cases" / scene_id / "runs/full/result.after.wav",
        BF_ROOT / variant / "cases" / scene_id / "runs/full/result.after.wav",
        int(scene["sample_rate"]),
        (float(window[0]), float(window[1])),
        int(engine["dsp_delay_samples"]),
        "配对控制 · BF 关闭 / 通道 0",
        after_label,
        analysis=analysis,
    )


def drc_case_spec(
    asset_id: str,
    original_run: str,
    scene_id: str,
    noise: str,
    variant: str,
    detail: str,
    analysis: str,
    after_label: str,
) -> CaseSpec:
    case_key = f"{original_run}__{scene_id}__{noise}"
    scene_path = (
        ROOT / "build" / original_run / "cases" / scene_id / "scene.json"
    )
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    engine = json.loads(
        (
            DRC_ROOT / variant / "cases" / case_key / "runs/full/result.json"
        ).read_text(encoding="utf-8")
    )
    return CaseSpec(
        asset_id,
        case_key,
        detail,
        DRC_ROOT / "bypass/cases" / case_key / "runs/full/result.after.wav",
        DRC_ROOT / variant / "cases" / case_key / "runs/full/result.after.wav",
        int(scene["sample_rate"]),
        tuple(float(value) for value in scene["windows"]["near_late"]),
        int(engine["dsp_delay_samples"]),
        "配对控制 · 无 TX DRC / 无 limiter",
        after_label,
        analysis=analysis,
    )


SPECS: tuple[CaseSpec, ...] = (
    CaseSpec(
        "ns-webrtc",
        "coverage-16000-validation-2303b3f5-snr+10",
        "CAVEMOVE · Smart；16 kHz；SNR +10 dB；后段语音窗",
        repo_path("build/controlled-cavemove-20261017/cases/coverage-16000-validation-2303b3f5-snr+10/before.wav"),
        repo_path("build/ns-all-family-comparison-20261019/runtime-serial-complete/webrtc_v5_bounded09-0.after.wav"),
        16000,
        (13.0, 21.0),
        416,
        "输入 · 带噪麦克风",
        "输出 · WebRTC NS",
        analysis="语音窗；用于观察谱增益与语音保真",
    ),
    CaseSpec(
        "ns-speex",
        "coverage-16000-validation-2303b3f5-snr+10",
        "CAVEMOVE · Smart；16 kHz；SNR +10 dB；后段语音窗",
        repo_path("build/controlled-cavemove-20261017/cases/coverage-16000-validation-2303b3f5-snr+10/before.wav"),
        repo_path("build/ns-all-family-comparison-20261019/runtime-serial-complete/speex_v5_steadydual-0.after.wav"),
        16000,
        (13.0, 21.0),
        480,
        "输入 · 带噪麦克风",
        "输出 · Speex steady-dual",
        analysis="同输入、同时间窗；扣除 480 样本固定延迟",
    ),
    CaseSpec(
        "ns-rnnoise",
        "vehicle-16000-development-m1-10",
        "CAVEMOVE · Alfa Romeo；16 kHz；SNR +20 dB；后段语音窗",
        repo_path("build/repair-sequential-20261017/ns-r10-rnnoise/controlled-cavemove-20261017/cases/vehicle-16000-development-m1-10/before.wav"),
        repo_path("build/repair-sequential-20261017/ns-r10-rnnoise/controlled-cavemove-20261017/cases/vehicle-16000-development-m1-10/runs/full/result.after.wav"),
        16000,
        (13.0, 21.0),
        480,
        "输入 · 带噪麦克风",
        "输出 · RNNoise",
        analysis="代表性回放；网络输出经同一后级保护",
    ),
    CaseSpec(
        "ns-ulunas",
        "coverage-16000-validation-2303b3f5-snr+10",
        "CAVEMOVE · Smart；16 kHz；SNR +10 dB；后段语音窗",
        repo_path("build/controlled-cavemove-20261017/cases/coverage-16000-validation-2303b3f5-snr+10/before.wav"),
        repo_path("build/ns-all-family-comparison-20261019/runtime-serial-complete/ulunas_direct-0.after.wav"),
        16000,
        (13.0, 21.0),
        480,
        "输入 · 带噪麦克风",
        "输出 · UL-UNAS",
        analysis="ONNX 四缓存流式输出；同 692 场景集",
    ),
    CaseSpec(
        "ns-omlsa",
        "coverage-16000-validation-2303b3f5-snr+10",
        "CAVEMOVE · Smart；16 kHz；SNR +10 dB；后段语音窗",
        repo_path("build/controlled-cavemove-20261017/cases/coverage-16000-validation-2303b3f5-snr+10/before.wav"),
        repo_path("build/ns-all-family-comparison-20261019/runtime-serial-complete/omlsa_imcra_v2_mild-0.after.wav"),
        16000,
        (13.0, 21.0),
        480,
        "输入 · 带噪麦克风",
        "输出 · OM-LSA + IMCRA",
        analysis="统计噪声估计与 OM-LSA 软增益",
    ),
    CaseSpec(
        "ns-super-gaussian",
        "coverage-16000-validation-2303b3f5-snr+10",
        "CAVEMOVE · Smart；16 kHz；SNR +10 dB；后段语音窗",
        repo_path("build/controlled-cavemove-20261017/cases/coverage-16000-validation-2303b3f5-snr+10/before.wav"),
        repo_path("build/ns-all-family-comparison-20261019/runtime-serial-complete/super_gaussian_balanced-0.after.wav"),
        16000,
        (13.0, 21.0),
        480,
        "输入 · 带噪麦克风",
        "输出 · Super-Gaussian",
        analysis="共享 IMCRA；仅替换 H1 条件增益",
    ),
    CaseSpec(
        "ns-hybrid-gru",
        "coverage-16000-validation-2303b3f5-snr+10",
        "CAVEMOVE · Smart；16 kHz；SNR +10 dB；后段语音窗",
        repo_path("build/controlled-cavemove-20261017/cases/coverage-16000-validation-2303b3f5-snr+10/before.wav"),
        repo_path("build/ns-all-family-comparison-20261019/runtime-serial-complete/conv_gain_gru32-0.after.wav"),
        16000,
        (13.0, 21.0),
        160,
        "输入 · 带噪麦克风",
        "输出 · Conv-Gain GRU32",
        analysis="OM-LSA 生成音频；GRU 只修正控制量",
    ),
    CaseSpec(
        "ns-residual",
        "coverage-16000-development-9c7b62f0-snr+30",
        "TCAR；16 kHz；SNR +30 dB；后段语音窗；门控残噪恢复案例",
        repo_path("build/controlled-native-full-20261017/cases/coverage-16000-development-9c7b62f0-snr+30/before.wav"),
        repo_path("build/ns-traditional-residual-20261019-v6/full-gate/omlsa/r3/421f46769689383c7718/result.after.wav"),
        16000,
        (13.0, 21.0),
        480,
        "输入 · 带噪麦克风",
        "输出 · OM-LSA + 残噪门控",
        analysis="通过恢复案例；residual gate SNR -12 dB",
    ),
    CaseSpec(
        "ns-gtcrn",
        "coverage-16000-validation-2303b3f5-snr+10",
        "CAVEMOVE · Smart；16 kHz；SNR +10 dB；后段语音窗",
        repo_path("build/controlled-cavemove-20261017/cases/coverage-16000-validation-2303b3f5-snr+10/before.wav"),
        repo_path("build/ns-all-family-comparison-20261019/runtime-serial-complete/gtcrn_dns3_direct-0.after.wav"),
        16000,
        (13.0, 21.0),
        480,
        "输入 · 带噪麦克风",
        "输出 · GTCRN DNS3",
        analysis="512/256 流式直接探针；30 ms 信号时延",
    ),
    CaseSpec(
        "ns-dpcrn",
        "coverage-16000-validation-2303b3f5-snr+10",
        "CAVEMOVE · Smart；16 kHz；SNR +10 dB；后段语音窗",
        repo_path("build/controlled-cavemove-20261017/cases/coverage-16000-validation-2303b3f5-snr+10/before.wav"),
        repo_path("build/ns-all-family-comparison-20261019/runtime-serial-complete/dpcrn_direct-0.after.wav"),
        16000,
        (13.0, 21.0),
        480,
        "输入 · 带噪麦克风",
        "输出 · DPCRN small",
        analysis="作者 stateful TFLite；512/256 流式探针",
    ),
    CaseSpec(
        "ns-webrtc-pass",
        "coverage-16000-validation-2303b3f5-snr+10",
        "CAVEMOVE · Smart；16 kHz；SNR +10 dB；后段语音窗；完整工程门限通过",
        repo_path("build/controlled-cavemove-20261017/cases/coverage-16000-validation-2303b3f5-snr+10/before.wav"),
        repo_path("build/ns-case-replay-20261020/webrtc_v5_bounded09/coverage-16000-validation-2303b3f5-snr+10/result.after.wav"),
        16000,
        (13.0, 21.0),
        416,
        "输入 · 带噪麦克风",
        "输出 · WebRTC NS",
        analysis="同输入哈希、同门限；完整工程门限通过",
    ),
    CaseSpec(
        "ns-webrtc-fail",
        "sweep-16000-validation-3b7f955f-cavemove_smart_forfour_s60_w0-snr-07",
        "CAVEMOVE · Smart；16 kHz；SNR +7 dB；后段语音窗；晚期 STOI 门失败",
        repo_path("build/controlled-cavemove-20261017/cases/sweep-16000-validation-3b7f955f-cavemove_smart_forfour_s60_w0-snr-07/before.wav"),
        repo_path("build/ns-case-replay-20261020/webrtc_v5_bounded09/sweep-16000-validation-3b7f955f-cavemove_smart_forfour_s60_w0-snr-07/result.after.wav"),
        16000,
        (13.0, 21.0),
        416,
        "输入 · 带噪麦克风",
        "输出 · WebRTC NS",
        analysis="同算法自己的失败输出；晚期 STOI 低于冻结门限",
    ),
    CaseSpec(
        "ns-speex-pass",
        "coverage-16000-validation-2303b3f5-snr+10",
        "CAVEMOVE · Smart；16 kHz；SNR +10 dB；后段语音窗；完整工程门限通过",
        repo_path("build/controlled-cavemove-20261017/cases/coverage-16000-validation-2303b3f5-snr+10/before.wav"),
        repo_path("build/ns-case-replay-20261020/speex_v5_steadydual/coverage-16000-validation-2303b3f5-snr+10/result.after.wav"),
        16000,
        (13.0, 21.0),
        480,
        "输入 · 带噪麦克风",
        "输出 · Speex steady-dual",
        analysis="同输入哈希、同门限；完整工程门限通过",
    ),
    CaseSpec(
        "ns-speex-fail",
        "noise-step-16000-validation-m1-snr+20",
        "CAVEMOVE；16 kHz；SNR +20 dB；噪声阶跃；完整工程门限失败",
        repo_path("build/controlled-cavemove-20261017/cases/noise-step-16000-validation-m1-snr+20/before.wav"),
        repo_path("build/ns-case-replay-20261020/speex_v5_steadydual/noise-step-16000-validation-m1-snr+20/result.after.wav"),
        16000,
        (8.0, 16.0),
        480,
        "输入 · 带噪麦克风",
        "输出 · Speex steady-dual",
        analysis="同算法自己的失败输出；噪声阶跃门未通过",
    ),
    CaseSpec(
        "ns-omlsa-pass",
        "coverage-16000-validation-2303b3f5-snr+10",
        "CAVEMOVE · Smart；16 kHz；SNR +10 dB；后段语音窗；完整工程门限通过",
        repo_path("build/controlled-cavemove-20261017/cases/coverage-16000-validation-2303b3f5-snr+10/before.wav"),
        repo_path("build/ns-case-replay-20261020/omlsa_imcra_v2_mild/coverage-16000-validation-2303b3f5-snr+10/result.after.wav"),
        16000,
        (13.0, 21.0),
        480,
        "输入 · 带噪麦克风",
        "输出 · OM-LSA + IMCRA",
        analysis="同输入哈希、同门限；完整工程门限通过",
    ),
    CaseSpec(
        "ns-omlsa-fail",
        "noise-step-16000-validation-m1-snr+20",
        "CAVEMOVE；16 kHz；SNR +20 dB；噪声阶跃；完整工程门限失败",
        repo_path("build/controlled-cavemove-20261017/cases/noise-step-16000-validation-m1-snr+20/before.wav"),
        repo_path("build/ns-case-replay-20261020/omlsa_imcra_v2_mild/noise-step-16000-validation-m1-snr+20/result.after.wav"),
        16000,
        (8.0, 16.0),
        480,
        "输入 · 带噪麦克风",
        "输出 · OM-LSA + IMCRA",
        analysis="同算法自己的失败输出；噪声阶跃门未通过",
    ),
    CaseSpec(
        "ns-super-gaussian-pass",
        "coverage-16000-validation-2303b3f5-snr+10",
        "CAVEMOVE · Smart；16 kHz；SNR +10 dB；后段语音窗；完整工程门限通过",
        repo_path("build/controlled-cavemove-20261017/cases/coverage-16000-validation-2303b3f5-snr+10/before.wav"),
        repo_path("build/ns-case-replay-20261020/super_gaussian_balanced/coverage-16000-validation-2303b3f5-snr+10/result.after.wav"),
        16000,
        (13.0, 21.0),
        480,
        "输入 · 带噪麦克风",
        "输出 · Super-Gaussian",
        analysis="同输入哈希、同门限；完整工程门限通过",
    ),
    CaseSpec(
        "ns-super-gaussian-fail",
        "noise-step-16000-validation-m1-snr+20",
        "CAVEMOVE；16 kHz；SNR +20 dB；噪声阶跃；完整工程门限失败",
        repo_path("build/controlled-cavemove-20261017/cases/noise-step-16000-validation-m1-snr+20/before.wav"),
        repo_path("build/ns-case-replay-20261020/super_gaussian_balanced/noise-step-16000-validation-m1-snr+20/result.after.wav"),
        16000,
        (8.0, 16.0),
        480,
        "输入 · 带噪麦克风",
        "输出 · Super-Gaussian",
        analysis="同算法自己的失败输出；噪声阶跃门未通过",
    ),
    CaseSpec(
        "ns-hybrid-gru-pass",
        "coverage-16000-validation-2303b3f5-snr+10",
        "CAVEMOVE · Smart；16 kHz；SNR +10 dB；后段语音窗；完整工程门限通过",
        repo_path("build/controlled-cavemove-20261017/cases/coverage-16000-validation-2303b3f5-snr+10/before.wav"),
        repo_path("build/ns-case-replay-20261020/conv_gain_gru32/coverage-16000-validation-2303b3f5-snr+10/result.after.wav"),
        16000,
        (13.0, 21.0),
        160,
        "输入 · 带噪麦克风",
        "输出 · Conv-Gain GRU32",
        analysis="同输入哈希、同门限；完整工程门限通过",
    ),
    CaseSpec(
        "ns-hybrid-gru-fail",
        "sweep-16000-validation-3b7f955f-cavemove_smart_forfour_s60_w0-snr-08",
        "CAVEMOVE · Smart；16 kHz；SNR +8 dB；后段语音窗；投影增益门失败",
        repo_path("build/controlled-cavemove-20261017/cases/sweep-16000-validation-3b7f955f-cavemove_smart_forfour_s60_w0-snr-08/before.wav"),
        repo_path("build/ns-case-replay-20261020/conv_gain_gru32/sweep-16000-validation-3b7f955f-cavemove_smart_forfour_s60_w0-snr-08/result.after.wav"),
        16000,
        (13.0, 21.0),
        160,
        "输入 · 带噪麦克风",
        "输出 · Conv-Gain GRU32",
        analysis="同算法自己的失败输出；语音投影门未通过",
    ),
    CaseSpec(
        "ns-ulunas-pass",
        "coverage-16000-validation-2303b3f5-snr+10",
        "CAVEMOVE · Smart；16 kHz；SNR +10 dB；后段语音窗；完整工程门限通过",
        repo_path("build/controlled-cavemove-20261017/cases/coverage-16000-validation-2303b3f5-snr+10/before.wav"),
        repo_path("build/ns-ulunas-dpcrn-20261019/runs/ulunas/full/ulunas_direct/91353b9fe0bd39c9732e/result.after.wav"),
        16000,
        (13.0, 21.0),
        480,
        "输入 · 带噪麦克风",
        "输出 · UL-UNAS",
        analysis="同输入哈希、同门限；完整工程门限通过",
    ),
    CaseSpec(
        "ns-ulunas-fail",
        "coverage-16000-validation-bae79cad-snr+30",
        "CAVEMOVE · Smart；16 kHz；SNR +30 dB；后段语音窗；噪声衰减门失败",
        repo_path("build/controlled-cavemove-20261017/cases/coverage-16000-validation-bae79cad-snr+30/before.wav"),
        repo_path("build/ns-ulunas-dpcrn-20261019/runs/ulunas/full/ulunas_direct/d274bb73b2d6392c08f3/result.after.wav"),
        16000,
        (13.0, 21.0),
        480,
        "输入 · 带噪麦克风",
        "输出 · UL-UNAS",
        analysis="同算法自己的失败输出；噪声衰减不足",
    ),
    CaseSpec(
        "ns-gtcrn-pass",
        "coverage-16000-validation-2303b3f5-snr+10",
        "CAVEMOVE · Smart；16 kHz；SNR +10 dB；后段语音窗；完整工程门限通过",
        repo_path("build/controlled-cavemove-20261017/cases/coverage-16000-validation-2303b3f5-snr+10/before.wav"),
        repo_path("build/ns-gtcrn-direct-20261018/full/gtcrn_dns3_direct/91353b9fe0bd39c9732e/result.after.wav"),
        16000,
        (13.0, 21.0),
        480,
        "输入 · 带噪麦克风",
        "输出 · GTCRN DNS3",
        analysis="同输入哈希、同门限；完整工程门限通过",
    ),
    CaseSpec(
        "ns-gtcrn-fail",
        "coverage-16000-validation-755dfafe-snr+30",
        "CAVEMOVE；16 kHz；SNR +30 dB；后段语音窗；早期与晚期 STOI 门失败",
        repo_path("build/controlled-cavemove-20261017/cases/coverage-16000-validation-755dfafe-snr+30/before.wav"),
        repo_path("build/ns-gtcrn-direct-20261018/full/gtcrn_dns3_direct/12a375d39d02b561b0a4/result.after.wav"),
        16000,
        (13.0, 21.0),
        480,
        "输入 · 带噪麦克风",
        "输出 · GTCRN DNS3",
        analysis="同算法自己的失败输出；高 SNR 语音可懂度退化",
    ),
    CaseSpec(
        "ns-dpcrn-pass",
        "coverage-16000-validation-2303b3f5-snr+10",
        "CAVEMOVE · Smart；16 kHz；SNR +10 dB；后段语音窗；完整工程门限通过",
        repo_path("build/controlled-cavemove-20261017/cases/coverage-16000-validation-2303b3f5-snr+10/before.wav"),
        repo_path("build/ns-ulunas-dpcrn-20261019/runs/dpcrn/full/dpcrn_direct/91353b9fe0bd39c9732e/result.after.wav"),
        16000,
        (13.0, 21.0),
        480,
        "输入 · 带噪麦克风",
        "输出 · DPCRN small",
        analysis="同输入哈希、同门限；完整工程门限通过",
    ),
    CaseSpec(
        "ns-dpcrn-fail",
        "coverage-16000-validation-755dfafe-snr+30",
        "CAVEMOVE；16 kHz；SNR +30 dB；后段语音窗；早期/晚期 STOI 与晚期 SI-SDR 失败",
        repo_path("build/controlled-cavemove-20261017/cases/coverage-16000-validation-755dfafe-snr+30/before.wav"),
        repo_path("build/ns-ulunas-dpcrn-20261019/runs/dpcrn/full/dpcrn_direct/12a375d39d02b561b0a4/result.after.wav"),
        16000,
        (13.0, 21.0),
        480,
        "输入 · 带噪麦克风",
        "输出 · DPCRN small",
        analysis="同算法自己的失败输出；高 SNR 语音质量与可懂度退化",
    ),
    CaseSpec(
        "ns-rnnoise-pass",
        "vehicle-16000-development-m1-10",
        "CAVEMOVE · Alfa Romeo；16 kHz；SNR +20 dB；后段语音窗；代表性通过",
        repo_path("build/repair-sequential-20261017/ns-r10-rnnoise/controlled-cavemove-20261017/cases/vehicle-16000-development-m1-10/before.wav"),
        repo_path("build/repair-sequential-20261017/ns-r10-rnnoise/controlled-cavemove-20261017/cases/vehicle-16000-development-m1-10/runs/full/result.after.wav"),
        16000,
        (13.0, 21.0),
        480,
        "输入 · 带噪麦克风",
        "输出 · RNNoise",
        analysis="代表性回放；完整工程门限通过",
    ),
    CaseSpec(
        "ns-rnnoise-fail",
        "vehicle-16000-holdout-m1-02",
        "native；16 kHz；holdout；RNNoise 近端 SI-SDR、双讲投影与远端线性 ERLE 失败",
        repo_path("build/controlled-native-full-20261017/cases/vehicle-16000-holdout-m1-02/before.wav"),
        repo_path("build/repair-sequential-20261017/ns-r10-rnnoise/controlled-native-full-20261017/cases/vehicle-16000-holdout-m1-02/runs/full/result.after.wav"),
        16000,
        (13.0, 21.0),
        480,
        "输入 · 带噪麦克风",
        "输出 · RNNoise",
        analysis="RNNoise 自己的代表性失败输出",
    ),
    CaseSpec(
        "aec-aec3-pass",
        "factorial-16000-rt0.15-delay020",
        "16 kHz；RT60 0.15 s；声学延迟 20 ms；SER −5 dB；双讲窗；完整工程门限通过",
        repo_path("build/aec3-state-fusion-20261020-r3/aec3_off/cases/factorial-16000-rt0.15-delay020/before.wav"),
        repo_path("build/aec3-state-fusion-20261020-r3/aec3_off/cases/factorial-16000-rt0.15-delay020/runs/aec_config_probe/result.after.wav"),
        16000,
        (17.0, 23.0),
        224,
        "输入 · 麦克风 / 双讲",
        "输出 · aec3_off 冻结 VCP",
        analysis="aec3_off 通过案例；双讲 ΔSI-SDR +9.762 dB，ΔSTOI +0.2015",
    ),
    CaseSpec(
        "aec-aec3-fail",
        "factorial-48000-rt0.45-delay080",
        "48 kHz；RT60 0.45 s；声学延迟 80 ms；SER −5 dB；远端单讲窗；远端衰减与线性 ERLE 失败",
        repo_path("build/aec3-state-fusion-20261020-r3/aec3_off/cases/factorial-48000-rt0.45-delay080/before.wav"),
        repo_path("build/aec3-state-fusion-20261020-r3/aec3_off/cases/factorial-48000-rt0.45-delay080/runs/aec_config_probe/result.after.wav"),
        48000,
        (6.0, 9.0),
        1248,
        "输入 · 麦克风 / 远端单讲",
        "输出 · aec3_off 冻结 VCP",
        analysis="aec3_off 失败案例；远端衰减 11.522 dB，线性 ERLE 3.365 dB；双讲门通过",
    ),
    CaseSpec(
        "aec-speex-pass",
        "factorial-16000-rt0.45-delay020",
        "16 kHz；RT60 0.45 s；声学延迟 20 ms；SER −5 dB；双讲窗；完整工程门限通过",
        repo_path("build/speex-res-fallback-20261020-r1/speex_res_fallback/cases/factorial-16000-rt0.45-delay020/before.wav"),
        repo_path("build/speex-res-fallback-20261020-r1/speex_res_fallback/cases/factorial-16000-rt0.45-delay020/runs/aec_config_probe/result.aec.wav"),
        16000,
        (17.0, 23.0),
        0,
        "输入 · 麦克风 / 双讲",
        "输出 · Speex MDF",
        analysis="RES fallback 通过；双讲 SI-SDR +4.809 dB，STOI +0.1907",
    ),
    CaseSpec(
        "aec-speex-fail",
        "factorial-16000-rt0.15-delay180",
        "16 kHz；RT60 0.15 s；声学延迟 180 ms；SER −5 dB；双讲窗；运行时门失败",
        repo_path("build/speex-res-fallback-20261020-r1/speex_res_fallback/cases/factorial-16000-rt0.15-delay180/before.wav"),
        repo_path("build/speex-res-fallback-20261020-r1/speex_res_fallback/cases/factorial-16000-rt0.15-delay180/runs/aec_config_probe/result.aec.wav"),
        16000,
        (17.0, 23.0),
        0,
        "输入 · 麦克风 / 双讲",
        "输出 · Speex MDF",
        analysis="失败案例；process max 19.06 ms，6 帧超过 10 ms；不是双讲门失败",
    ),
    CaseSpec(
        "aec-speex-coherence-pass",
        "factorial-16000-rt0.15-delay080",
        "16 kHz；RT60 0.15 s；声学延迟 80 ms；SER −5 dB；双讲窗；相干残差候选通过",
        repo_path("build/aec-coherence-residual-20261023-r2/speex_coherence_residual/cases/factorial-16000-rt0.15-delay080/before.wav"),
        repo_path("build/aec-coherence-residual-20261023-r2/speex_coherence_residual/cases/factorial-16000-rt0.15-delay080/runs/aec_config_probe/result.after.wav"),
        16000,
        (17.0, 23.0),
        96,
        "输入 · 麦克风 / 双讲",
        "输出 · Speex MDF + Coherence RES",
        analysis="候选通过案例；far-only 衰减 26.734 dB，双讲门通过；r2 冻结矩阵",
    ),
    CaseSpec(
        "aec-speex-coherence-fail",
        "factorial-48000-rt0.45-delay180",
        "48 kHz；RT60 0.45 s；声学延迟 180 ms；SER −5 dB；双讲窗；运行时门失败",
        repo_path("build/aec-coherence-residual-20261023-r2/speex_coherence_residual/cases/factorial-48000-rt0.45-delay180/before.wav"),
        repo_path("build/aec-coherence-residual-20261023-r2/speex_coherence_residual/cases/factorial-48000-rt0.45-delay180/runs/aec_config_probe/result.after.wav"),
        48000,
        (17.0, 23.0),
        336,
        "输入 · 麦克风 / 双讲",
        "输出 · Speex MDF + Coherence RES",
        analysis="候选失败案例；process max 11.196 ms，1 帧超过 10 ms；质量门通过",
    ),
    CaseSpec(
        "aec-aec3",
        "factorial-16000-rt0.15-delay020",
        "16 kHz；RT60 0.15 s；声学延迟 20 ms；SER -5 dB；双讲窗",
        repo_path("build/aec3-state-fusion-20261020-r3/aec3_off/cases/factorial-16000-rt0.15-delay020/before.wav"),
        repo_path("build/aec3-state-fusion-20261020-r3/aec3_off/cases/factorial-16000-rt0.15-delay020/runs/aec_config_probe/result.aec.wav"),
        16000,
        (17.0, 23.0),
        128,
        "输入 · 麦克风 / 双讲",
        "输出 · AEC3 tap",
        analysis="双讲窗；线性滤波后仍有内部非线性抑制",
    ),
    CaseSpec(
        "aec-speex",
        "factorial-16000-rt0.15-delay020",
        "16 kHz；RT60 0.15 s；声学延迟 20 ms；SER -5 dB；双讲窗",
        repo_path("build/speex-res-fallback-20261020-r1/speex_res_fallback/cases/factorial-16000-rt0.15-delay020/before.wav"),
        repo_path("build/speex-res-fallback-20261020-r1/speex_res_fallback/cases/factorial-16000-rt0.15-delay020/runs/aec_config_probe/result.aec.wav"),
        16000,
        (17.0, 23.0),
        0,
        "输入 · 麦克风 / 双讲",
        "输出 · Speex MDF",
        analysis="RES 关闭；原始 MDF 误差在 NS/AGC 前导出",
    ),
    CaseSpec(
        "aec-pre-beam",
        "16000-baseline / radio_and_road",
        "4 麦连续流；16 kHz；radio_and_road 段；场景 17–23 s 双讲窗",
        repo_path("build/vehicle-v2-continuous-20261010/16000-baseline/stream.bf.wav"),
        repo_path("build/vehicle-v2-continuous-20261010/16000-baseline/stream.aec.wav"),
        16000,
        (96.0, 102.0),
        384,
        "BF 后输入 · 双讲",
        "输出 · AEC3 tap",
        after_channel=0,
        analysis="连续流原时间轴 96–102 s；BF 在 AEC 之前",
    ),
    CaseSpec(
        "agc-guarded5",
        "vehicle-noise-24000-m1-sim_rain_impulse-snr+30",
        "24 kHz 原生场景；AGC2 内部 48 kHz；雨滴脉冲；SNR +30 dB；窗内可见 0↔8 dB 增益切换",
        repo_path(
            "build/failure-cause-analysis-20261017/ablations/cases/vehicle-noise-24000-development-m1-sim_rain_impulse-snr+30/runs/diagnostic_full/result.nr-protected-agc-before.float.wav"
        ),
        repo_path(
            "build/failure-cause-analysis-20261017/ablations/cases/vehicle-noise-24000-development-m1-sim_rain_impulse-snr+30/runs/diagnostic_full/result.agc-after.float.wav"
        ),
        48000,
        (7.8, 9.2),
        496,
        "AGC 前 · protected NR",
        "AGC 后 · core-rate tap",
        analysis="真实 AGC 前后阶段 tap；窗内显示快速增益切换",
    ),
    CaseSpec(
        "agc-utterance-v4",
        "vehicle-16000-development-m1-10",
        "CAVEMOVE；16 kHz；−45 dBFS 弱语音代表；Stage4 代表失败集中的 2 个修复案例之一",
        repo_path(
            "build/repair-sequential-20261017/final-r16-experimental/controlled-cavemove-20261017/cases/vehicle-16000-development-m1-10/runs/full/result.nr-protected-agc-before.float.wav"
        ),
        repo_path(
            "build/repair-sequential-20261017/final-r16-experimental/controlled-cavemove-20261017/cases/vehicle-16000-development-m1-10/runs/full/result.agc-after.float.wav"
        ),
        16000,
        (10.5, 16.5),
        96,
        "AGC 前 · protected NR",
        "AGC 后 · Stage4",
        analysis="同一冻结案例；Stage4 在代表集内由 fail 转为 pass",
    ),
    CaseSpec(
        "agc-stage4-release",
        "vehicle-16000-development-m1-10",
        "CAVEMOVE；16 kHz；−45 dBFS 弱语音；Stage4 噪声域释放代表修复",
        repo_path(
            "build/agc-stage4-release-20261020/controlled-native-full-20261017/cases/vehicle-16000-development-m1-10/runs/full/result.nr-protected-agc-before.float.wav"
        ),
        repo_path(
            "build/agc-stage4-release-20261020/controlled-native-full-20261017/cases/vehicle-16000-development-m1-10/runs/full/result.agc-after.float.wav"
        ),
        16000,
        (10.5, 16.5),
        96,
        "AGC 前 · protected NR",
        "AGC 后 · Stage4 + release",
        analysis="同一 AGC 前后 tap；持续噪声证据释放增益",
    ),
    CaseSpec(
        "agc-best-candidate",
        "vehicle-16000-development-m1-12",
        "native；16 kHz；−15 dBFS；单麦；关闭 TX DRC 后的代表性近端质量修复",
        repo_path(
            "build/agc-stage4-release-no-tx-drc-20261020/controlled-native-full-20261017/cases/vehicle-16000-development-m1-12/runs/full/result.nr-protected-agc-before.float.wav"
        ),
        repo_path(
            "build/agc-stage4-release-no-tx-drc-20261020/controlled-native-full-20261017/cases/vehicle-16000-development-m1-12/runs/full/result.agc-after.float.wav"
        ),
        16000,
        (10.5, 16.5),
        96,
        "AGC 前 · protected NR",
        "AGC 后 · 组合候选",
        analysis="修复近端 SI-SDR；不表示整个组合候选已准入",
    ),
    CaseSpec(
        "agc-best-regression",
        "vehicle-48000-development-m1-11",
        "native；48 kHz；−30 dBFS；tcar；组合候选近端 SI-SDR/STOI 回归",
        repo_path(
            "build/agc-stage4-release-no-tx-drc-20261020/controlled-native-full-20261017/cases/vehicle-48000-development-m1-11/runs/full/result.nr-protected-agc-before.float.wav"
        ),
        repo_path(
            "build/agc-stage4-release-no-tx-drc-20261020/controlled-native-full-20261017/cases/vehicle-48000-development-m1-11/runs/full/result.agc-after.float.wav"
        ),
        48000,
        (10.5, 16.5),
        96,
        "AGC 前 · protected NR",
        "AGC 后 · 回归案例",
        analysis="近端 SI-SDR 与晚期 STOI 退化；必须阻断准入",
    ),
    CaseSpec(
        "eq-three-pass",
        "controlled-native-full-20261017__vehicle-16000-holdout-m2-11__tbus",
        "native · tbus；16 kHz；近端 −45 dBFS；三段 EQ 在同案例配对中通过；ΔSI-SDR +0.3157 dB",
        repo_path(
            "build/eq-algorithm-matrix-linear-20261022-r1/bypass/cases/controlled-native-full-20261017__vehicle-16000-holdout-m2-11__tbus/runs/full/result.after.wav"
        ),
        repo_path(
            "build/eq-algorithm-matrix-linear-20261022-r1/three_band/cases/controlled-native-full-20261017__vehicle-16000-holdout-m2-11__tbus/runs/full/result.after.wav"
        ),
        16000,
        (13.0, 21.0),
        98,
        "输入 · VCP bypass",
        "输出 · 三段 EQ",
        analysis="通过：ΔSI-SDR +0.3157 dB；ΔSTOI −0.000197；同输入、同 VCP",
    ),
    CaseSpec(
        "eq-three-fail",
        "controlled-native-full-20261017__vehicle-48000-holdout-m4-12__tbus",
        "native · tbus；48 kHz；近端 −30 dBFS；三段 EQ 的严重 SI-SDR 回归代表；ΔSI-SDR −14.0221 dB",
        repo_path(
            "build/eq-algorithm-matrix-linear-20261022-r1/bypass/cases/controlled-native-full-20261017__vehicle-48000-holdout-m4-12__tbus/runs/full/result.after.wav"
        ),
        repo_path(
        "build/eq-algorithm-matrix-linear-20261022-r1/three_band/cases/controlled-native-full-20261017__vehicle-48000-holdout-m4-12__tbus/runs/full/result.after.wav"
        ),
        48000,
        (13.0, 21.0),
        349,
        "输入 · VCP bypass",
        "输出 · 三段 EQ",
        analysis="失败：ΔSI-SDR −14.0221 dB；输出 SHA 已改变且无削波",
    ),
    CaseSpec(
        "eq-low-shelf-fail",
        "controlled-native-full-20261017__vehicle-16000-holdout-m1-10__tbus",
        "native · tbus；16 kHz；近端 −45 dBFS；低架削减的损失最小代表，仍为 −0.5881 dB",
        repo_path(
            "build/eq-algorithm-matrix-linear-20261022-r1/bypass/cases/controlled-native-full-20261017__vehicle-16000-holdout-m1-10__tbus/runs/full/result.after.wav"
        ),
        repo_path(
            "build/eq-algorithm-matrix-linear-20261022-r1/low_shelf/cases/controlled-native-full-20261017__vehicle-16000-holdout-m1-10__tbus/runs/full/result.after.wav"
        ),
        16000,
        (13.0, 21.0),
        96,
        "输入 · VCP bypass",
        "输出 · 低架 160 Hz −3 dB",
        analysis="失败：ΔSI-SDR −0.5881 dB；本算法 108 个案例全部未达门限",
    ),
    CaseSpec(
        "eq-parametric-fail",
        "controlled-native-full-20261017__vehicle-48000-validation-m2-13__straffic",
        "native · straffic；48 kHz；近端 −15 dBFS；双峰 EQ 的损失最小代表，仍为 −0.9098 dB",
        repo_path(
            "build/eq-algorithm-matrix-linear-20261022-r1/bypass/cases/controlled-native-full-20261017__vehicle-48000-validation-m2-13__straffic/runs/full/result.after.wav"
        ),
        repo_path(
            "build/eq-algorithm-matrix-linear-20261022-r1/parametric_peak/cases/controlled-native-full-20261017__vehicle-48000-validation-m2-13__straffic/runs/full/result.after.wav"
        ),
        48000,
        (13.0, 21.0),
        341,
        "输入 · VCP bypass",
        "输出 · 双参数峰",
        analysis="失败：ΔSI-SDR −0.9098 dB；−3 dB preamp 防止削波但不保证质量",
    ),
    CaseSpec(
        "eq-calibrated-fail",
        "controlled-native-full-20261017__vehicle-48000-validation-m2-13__straffic",
        "校准反演；48 kHz；近端 −15 dBFS；合成共振拟合参数在语音语料中的最小损失代表，仍为 −3.7123 dB",
        repo_path(
            "build/eq-algorithm-matrix-linear-20261022-r1/bypass/cases/controlled-native-full-20261017__vehicle-48000-validation-m2-13__straffic/runs/full/result.after.wav"
        ),
        repo_path(
            "build/eq-algorithm-matrix-linear-20261022-r1/calibrated_inverse/cases/controlled-native-full-20261017__vehicle-48000-validation-m2-13__straffic/runs/full/result.after.wav"
        ),
        48000,
        (13.0, 21.0),
        341,
        "输入 · VCP bypass",
        "输出 · 校准反演",
        analysis="失败：ΔSI-SDR −3.7123 dB；校准响应通过不等于语音质量通过",
    ),
    CaseSpec(
        "aec-fusion-fail",
        "factorial-48000-rt0.45-delay080",
        "AEC3 历史状态融合候选；48 kHz；RT60 0.45 s；80 ms；双讲 SI-SDR 新增失败；历史 7/27",
        repo_path(
            "build/aec3-state-fusion-20261020-r3/aec3_fusion/cases/factorial-48000-rt0.45-delay080/before.wav"
        ),
        repo_path(
            "build/aec3-state-fusion-20261020-r3/aec3_fusion/cases/factorial-48000-rt0.45-delay080/runs/aec_config_probe/result.after.wav"
        ),
        48000,
        (17.0, 23.0),
        1248,
        "输入 · 双讲麦克风",
        "输出 · 历史 AEC3 状态融合",
        analysis="历史负对照：双讲 raw SI-SDR −3.9075 dB，低于相对门限 −3.8213 dB；当前源码没有该 flag",
    ),
    CaseSpec(
        "eq-linear-fail",
        "controlled-native-full-20261017__vehicle-48000-holdout-m4-12__tbus",
        "native · tbus；48 kHz；近端 −30 dBFS；线性相位三段 FIR；961 taps，固定 480 样本群延迟",
        repo_path(
            "build/eq-algorithm-matrix-linear-20261022-r1/bypass/cases/controlled-native-full-20261017__vehicle-48000-holdout-m4-12__tbus/runs/full/result.after.wav"
        ),
        repo_path(
            "build/eq-algorithm-matrix-linear-20261022-r1/linear_phase_three_band/cases/controlled-native-full-20261017__vehicle-48000-holdout-m4-12__tbus/runs/full/result.after.wav"
        ),
        48000,
        (13.0, 21.0),
        829,
        "输入 · VCP bypass",
        "输出 · 线性相位 FIR",
        analysis="失败：真实 FIR 输出改变且无削波；固定 480 样本 EQ 群延迟之外仍有语音质量回归",
    ),
    bf_case_spec(
        "bf-das-pass",
        "delay_sum",
        "field-p0-m4-wind-snr15",
        "Corcos 风场；4 麦；SNR +15 dB；DAS 通过增量门限",
        "通过：ΔSI-SDR +2.0139 dB；ΔSTOI +0.000179",
        "输出 · DAS 延迟求和",
    ),
    bf_case_spec(
        "bf-das-fail",
        "delay_sum",
        "field-p0-m4-multirender-snr0",
        "多声源；4 麦；SNR 0 dB；仅有 near_only 窗；DAS 回归代表",
        "失败：ΔSI-SDR −1.0096 dB；目标语义仍按同案例 BF 控制配对",
        "输出 · DAS 延迟求和",
    ),
    bf_case_spec(
        "bf-mvdr-observed-pass",
        "mvdr_observed",
        "field-p1-m4-wind-snr0",
        "Corcos 风场；4 麦；SNR 0 dB；观测协方差 MVDR 通过",
        "通过：ΔSI-SDR +7.2895 dB；ΔSTOI +0.002041",
        "输出 · observed-covariance MVDR",
    ),
    bf_case_spec(
        "bf-mvdr-observed-fail",
        "mvdr_observed",
        "field-p0-m4-multirender-snr30",
        "多声源；4 麦；SNR +30 dB；near_only 窗；观测协方差 MVDR 回归代表",
        "失败：ΔSI-SDR −10.7618 dB；近端高 SNR 时目标导向风险最大",
        "输出 · observed-covariance MVDR",
    ),
    bf_case_spec(
        "bf-mvdr-fixed-pass",
        "mvdr_fixed",
        "field-p1-m4-wind-snr0",
        "Corcos 风场；4 麦；SNR 0 dB；固定导向 MVDR 通过",
        "通过：ΔSI-SDR +7.2637 dB；ΔSTOI +0.001979",
        "输出 · fixed-steering MVDR",
    ),
    bf_case_spec(
        "bf-mvdr-fixed-fail",
        "mvdr_fixed",
        "field-p0-m2-multirender-snr30",
        "多声源；2 麦；SNR +30 dB；固定导向 MVDR 回归代表",
        "失败：ΔSI-SDR −19.5974 dB；导向误差和低阵列自由度暴露",
        "输出 · fixed-steering MVDR",
    ),
    bf_case_spec(
        "bf-mpdr-pass",
        "mpdr_continuous",
        "field-p1-m4-wind-snr15",
        "Corcos 风场；4 麦；SNR +15 dB；连续混合协方差 MPDR 通过",
        "通过：ΔSI-SDR +7.2573 dB；ΔSTOI +0.000092",
        "输出 · continuous-covariance MPDR",
    ),
    bf_case_spec(
        "bf-mpdr-fail",
        "mpdr_continuous",
        "field-p0-m2-multirender-snr30",
        "多声源；2 麦；SNR +30 dB；连续协方差 MPDR 回归代表",
        "失败：ΔSI-SDR −19.6168 dB；连续训练不能替代准确导向",
        "输出 · continuous-covariance MPDR",
    ),
    bf_case_spec(
        "bf-das-wind-pass",
        "das_wind",
        "field-p0-m4-wind-snr15",
        "Corcos 风场；4 麦；SNR +15 dB；DAS 风压权重通过",
        "通过：ΔSI-SDR +2.0251 dB；平均噪声增量略高于 DAS",
        "输出 · DAS + 风压权重",
    ),
    bf_case_spec(
        "bf-das-wind-fail",
        "das_wind",
        "field-p0-m4-multirender-snr0",
        "多声源；4 麦；SNR 0 dB；near_only 窗；风压权重未挽救回归",
        "失败：ΔSI-SDR −1.0097 dB；低频权重对多声源目标无帮助",
        "输出 · DAS + 风压权重",
    ),
    bf_case_spec(
        "bf-mvdr-wind-pass",
        "mvdr_wind",
        "field-p1-m4-wind-snr0",
        "Corcos 风场；4 麦；SNR 0 dB；MVDR 与风压联合路径通过",
        "通过：ΔSI-SDR +6.7845 dB；ΔSTOI +0.002103",
        "输出 · MVDR + 风压权重",
    ),
    bf_case_spec(
        "bf-mvdr-wind-fail",
        "mvdr_wind",
        "field-p0-m4-multirender-snr30",
        "多声源；4 麦；SNR +30 dB；MVDR + 风压联合路径回归代表",
        "失败：ΔSI-SDR −10.7382 dB；组合不能跨所有空间场泛化",
        "输出 · MVDR + 风压权重",
    ),
    drc_case_spec(
        "drc-limiter-representative",
        "controlled-cavemove-20261017",
        "vehicle-16000-holdout-m4-13",
        "cavemove_volkswagen_golf_s60_w0",
        "limiter_only",
        "CAVEMOVE；16 kHz；4 麦；终级 limiter 保留；本案例未触发削顶",
        "代表案例：输出 SHA 与控制相同；108 案例均未实际触发 limiter",
        "输出 · 仅终级 limiter",
    ),
    drc_case_spec(
        "drc-default-pass",
        "controlled-native-full-20261017",
        "vehicle-16000-holdout-m1-11",
        "tbus",
        "default",
        "native；16 kHz；−15 dBFS；tbus；默认 TX DRC 通过且真实改变输出",
        "通过：ΔSI-SDR −0.0063 dB；峰值下降 0.641 dB",
        "输出 · 默认 TX DRC",
    ),
    drc_case_spec(
        "drc-default-fail",
        "controlled-native-full-20261017",
        "vehicle-48000-holdout-m1-12",
        "tbus",
        "default",
        "native；48 kHz；−15 dBFS；tbus；默认 TX DRC 严重回归",
        "失败：ΔSI-SDR −10.0886 dB；ΔSTOI −0.004614",
        "输出 · 默认 TX DRC",
    ),
    drc_case_spec(
        "drc-gentle-pass",
        "controlled-native-full-20261017",
        "vehicle-16000-holdout-m4-12",
        "tbus",
        "gentle",
        "native；16 kHz；−15 dBFS；4 麦；gentle 2:1 压缩通过",
        "通过：ΔSI-SDR −0.0013 dB；峰值下降 0.433 dB",
        "输出 · gentle TX DRC",
    ),
    drc_case_spec(
        "drc-gentle-fail",
        "controlled-native-full-20261017",
        "vehicle-48000-holdout-m1-12",
        "tbus",
        "gentle",
        "native；48 kHz；−15 dBFS；tbus；gentle 仍越过语音门限",
        "失败：ΔSI-SDR −6.5662 dB；较默认路径更轻但仍回归",
        "输出 · gentle TX DRC",
    ),
    drc_case_spec(
        "drc-aggressive-pass",
        "controlled-native-full-20261017",
        "vehicle-24000-validation-m2-12",
        "pink",
        "aggressive",
        "native；24 kHz；−15 dBFS；2 麦；aggressive 在峰值下降时仍通过",
        "通过：ΔSI-SDR −0.0102 dB；峰值下降 1.509 dB",
        "输出 · aggressive TX DRC",
    ),
    drc_case_spec(
        "drc-aggressive-fail",
        "controlled-native-full-20261017",
        "vehicle-48000-holdout-m1-12",
        "tbus",
        "aggressive",
        "native；48 kHz；−15 dBFS；tbus；aggressive 同时失败 SI-SDR 与 STOI",
        "失败：ΔSI-SDR −18.3867 dB；ΔSTOI −0.054759；峰值下降 5.369 dB",
        "输出 · aggressive TX DRC",
    ),
    drc_case_spec(
        "drc-fast-pass",
        "controlled-native-full-20261017",
        "vehicle-16000-holdout-m1-11",
        "tbus",
        "fast",
        "native；16 kHz；−15 dBFS；tbus；fast attack/release 通过",
        "通过：ΔSI-SDR −0.0610 dB；峰值下降 1.317 dB",
        "输出 · fast TX DRC",
    ),
    drc_case_spec(
        "drc-fast-fail",
        "controlled-native-full-20261017",
        "vehicle-48000-holdout-m1-12",
        "tbus",
        "fast",
        "native；48 kHz；−15 dBFS；tbus；快速增益平滑导致严重瞬态回归",
        "失败：ΔSI-SDR −11.8920 dB；STOI 仍未触发 −0.02 门限",
        "输出 · fast TX DRC",
    ),
    drc_case_spec(
        "drc-lookahead-pass",
        "controlled-native-full-20261017",
        "vehicle-16000-development-m1-10",
        "tcar",
        "lookahead_2ms",
        "native；tcar；16 kHz；−45 dBFS；look-ahead 2 ms",
        "通过：延迟 2 ms，语音与峰值门通过",
        "输出 · Look-ahead limiter 2 ms",
    ),
    drc_case_spec(
        "drc-lookahead-fail",
        "controlled-native-full-20261017",
        "vehicle-24000-development-m1-10",
        "pink",
        "lookahead_2ms",
        "native；pink；24 kHz；−45 dBFS；look-ahead 2 ms",
        "失败：声明总延迟 8.667 ms，超过 4.1 ms 门限",
        "输出 · Look-ahead limiter 2 ms",
    ),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_channel(
    path: Path, expected_rate: int, channel: int, crop_samples: int = 0
) -> np.ndarray:
    data, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != expected_rate:
        raise ValueError(f"{path}: expected {expected_rate} Hz, got {rate}")
    if channel < 0 or channel >= data.shape[1]:
        raise ValueError(f"{path}: channel {channel} absent from {data.shape[1]} channels")
    if crop_samples < 0 or crop_samples > data.shape[0]:
        raise ValueError(f"{path}: invalid crop {crop_samples}")
    return np.asarray(data[crop_samples:, channel], dtype=np.float64)


def crop_window(
    signal: np.ndarray,
    sample_rate: int,
    window: tuple[float, float],
    delay_samples: int,
) -> np.ndarray:
    start = round(window[0] * sample_rate)
    end = round(window[1] * sample_rate)
    source_start = start + delay_samples
    source_end = end + delay_samples
    if source_start < 0 or source_end > signal.shape[0]:
        raise ValueError(
            f"window {window} + delay {delay_samples} exceeds {signal.shape[0]} samples"
        )
    return signal[source_start:source_end]


def finite_nonzero(label: str, values: np.ndarray) -> None:
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"{label}: empty or non-finite samples")
    if float(np.max(np.abs(values))) < 1e-7:
        raise ValueError(f"{label}: visually silent samples")


def envelope(values: np.ndarray, bins: int = ENVELOPE_BINS) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0, values.size, bins + 1, dtype=np.int64)
    low = np.empty(bins, dtype=np.float64)
    high = np.empty(bins, dtype=np.float64)
    for index, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        chunk = values[start:max(start + 1, end)]
        low[index] = float(np.min(chunk))
        high[index] = float(np.max(chunk))
    return low, high


def waveform_polygon(
    low: np.ndarray,
    high: np.ndarray,
    x0: float,
    x1: float,
    center: float,
    amplitude: float,
    y_scale: float,
) -> str:
    xs = np.linspace(x0, x1, low.size)
    top = np.column_stack((xs, center - high * amplitude * y_scale))
    bottom = np.column_stack((xs[::-1], center - low[::-1] * amplitude * y_scale))
    points = np.vstack((top, bottom))
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def mel_filterbank(sample_rate: int, fft_size: int, bands: int) -> np.ndarray:
    def hz_to_mel(freq: np.ndarray | float) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + np.asarray(freq) / 700.0)

    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    nyquist = sample_rate / 2.0
    mel_edges = np.linspace(hz_to_mel(0.0), hz_to_mel(nyquist), bands + 2)
    hz_edges = mel_to_hz(mel_edges)
    fft_freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
    filters = np.zeros((bands, fft_freqs.size), dtype=np.float64)
    for band in range(bands):
        left, center, right = hz_edges[band : band + 3]
        rising = (fft_freqs - left) / max(center - left, 1e-9)
        falling = (right - fft_freqs) / max(right - center, 1e-9)
        filters[band] = np.maximum(0.0, np.minimum(rising, falling))
    return filters


def mel_spectrogram(values: np.ndarray, sample_rate: int) -> np.ndarray:
    if values.size < FFT_SIZE:
        raise ValueError("window too short for STFT")
    count = 1 + (values.size - FFT_SIZE) // HOP_SIZE
    indices = np.arange(FFT_SIZE)[None, :] + HOP_SIZE * np.arange(count)[:, None]
    frames = values[indices] * np.hanning(FFT_SIZE)[None, :]
    power = np.abs(np.fft.rfft(frames, n=FFT_SIZE, axis=1)) ** 2
    mel = power @ mel_filterbank(sample_rate, FFT_SIZE, MEL_BANDS).T
    mel = mel.T
    if mel.shape[1] > MEL_TIME_COLUMNS:
        edges = np.linspace(0, mel.shape[1], MEL_TIME_COLUMNS + 1, dtype=np.int64)
        mel = np.column_stack(
            [
                np.mean(mel[:, start : max(start + 1, end)], axis=1)
                for start, end in zip(edges[:-1], edges[1:])
            ]
        )
    return mel


def percentile_range(mel: np.ndarray) -> tuple[float, float]:
    db = 10.0 * np.log10(np.maximum(mel, 1e-18))
    finite = db[np.isfinite(db)]
    if finite.size == 0:
        raise ValueError("empty Mel spectrum")
    upper = float(np.percentile(finite, 99.7))
    return upper - MEL_DB_RANGE, upper


def mel_color(value: float, low: float, high: float) -> str:
    ratio = float(np.clip((value - low) / max(high - low, 1e-12), 0.0, 1.0))
    stops = (
        (0.00, (10, 24, 52)),
        (0.22, (28, 67, 112)),
        (0.43, (42, 128, 147)),
        (0.63, (111, 171, 113)),
        (0.81, (235, 185, 70)),
        (1.00, (247, 113, 51)),
    )
    for (left, first), (right, second) in zip(stops, stops[1:]):
        if ratio <= right:
            blend = (ratio - left) / max(right - left, 1e-12)
            color = tuple(
                round(a + (b - a) * blend) for a, b in zip(first, second)
            )
            return f"rgb({color[0]},{color[1]},{color[2]})"
    return "rgb(247,113,51)"


def spectrum_cells(
    mel: np.ndarray,
    x: float,
    y: float,
    width: float,
    height: float,
    low: float,
    high: float,
) -> str:
    db = 10.0 * np.log10(np.maximum(mel, 1e-18))
    bands, frames = db.shape
    cell_width = width / frames
    cell_height = height / bands
    cells: list[str] = []
    for frame in range(frames):
        for band in range(bands):
            color = mel_color(float(db[bands - 1 - band, frame]), low, high)
            cells.append(
                f'<rect x="{x + frame * cell_width:.3f}" '
                f'y="{y + band * cell_height:.3f}" '
                f'width="{cell_width + 0.04:.3f}" '
                f'height="{cell_height + 0.04:.3f}" fill="{color}"/>'
            )
    return "".join(cells)


def svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 14,
    color: str = "#65758b",
    weight: int = 400,
    anchor: str = "start",
    mono: bool = False,
) -> str:
    family = (
        'ui-monospace, SFMono-Regular, Consolas, monospace'
        if mono
        else "'Noto Sans CJK SC','Noto Sans SC',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif"
    )
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
        f'fill="{color}" font-size="{size}" font-weight="{weight}" '
        f'font-family="{family}">{html.escape(text)}</text>'
    )


def waveform_svg(
    spec: CaseSpec,
    before: np.ndarray,
    after: np.ndarray,
    before_sha: str,
    after_sha: str,
) -> str:
    before_low, before_high = envelope(before)
    after_low, after_high = envelope(after)
    maximum = max(
        float(np.max(np.abs(before))), float(np.max(np.abs(after))), 1e-12
    )
    start, end = spec.window_seconds
    plot_x0, plot_x1 = 116.0, 1174.0
    top_center, bottom_center = 225.0, 445.0
    amplitude = 79.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WAVE_WIDTH} {WAVE_HEIGHT}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{html.escape(spec.case_id)} 输入与输出波形</title>',
        f'<desc id="desc">{html.escape(spec.detail)}。输出扣除 {spec.delay_samples} 样本声明延迟，同一 {start:.1f}–{end:.1f} 秒时间窗，原幅度共享纵轴。</desc>',
        '<rect width="1200" height="570" fill="#ffffff"/>',
        svg_text(48, 42, "TYPICAL CASE · WAVEFORM", size=12, weight=650),
        svg_text(48, 69, spec.case_id, size=19, color="#233248", weight=650, mono=True),
        svg_text(48, 92, spec.detail, size=13),
        f'<rect x="{plot_x0}" y="135" width="{plot_x1 - plot_x0}" height="310" rx="7" fill="#fbfcfe" stroke="#e1e7ef"/>',
        svg_text(48, 154, spec.before_label, size=14, color="#31516f", weight=650),
        svg_text(48, 174, f"峰值 {maximum:.4f} FS", size=11),
        svg_text(48, 374, spec.after_label, size=14, color="#31516f", weight=650),
        svg_text(48, 394, f"峰值 {maximum:.4f} FS", size=11),
        f'<line x1="{plot_x0}" y1="{top_center}" x2="{plot_x1}" y2="{top_center}" stroke="#dbe3ed"/>',
        f'<line x1="{plot_x0}" y1="{bottom_center}" x2="{plot_x1}" y2="{bottom_center}" stroke="#dbe3ed"/>',
        f'<polygon points="{waveform_polygon(before_low, before_high, plot_x0, plot_x1, top_center, amplitude, WAVEFORM_Y_SCALE)}" fill="#526e91" fill-opacity=".78"/>',
        f'<polygon points="{waveform_polygon(after_low, after_high, plot_x0, plot_x1, bottom_center, amplitude, WAVEFORM_Y_SCALE)}" fill="#25899a" fill-opacity=".82"/>',
        f'<line x1="{plot_x0}" y1="464" x2="{plot_x1}" y2="464" stroke="#dbe3ed"/>',
        svg_text(plot_x0, 487, f"{start:.1f} s", size=11, anchor="start"),
        svg_text((plot_x0 + plot_x1) / 2, 487, f"{((start + end) / 2):.1f} s", size=11, anchor="middle"),
        svg_text(plot_x1, 487, f"{end:.1f} s", size=11, anchor="end"),
        svg_text(plot_x0, 510, "−1", size=10, anchor="middle"),
        svg_text(plot_x0 + 28, 510, "+1 FS", size=10, anchor="start"),
        svg_text(48, 538, f"{spec.sample_rate / 1000:g} kHz · delay {spec.delay_samples} samples · Hann envelope · no gain normalization", size=11),
        f'<line x1="48" y1="552" x2="1152" y2="552" stroke="#edf1f5"/>',
        svg_text(48, 565, f"before sha256 {before_sha[:20]}…   after sha256 {after_sha[:20]}…", size=9, color="#8a98aa", mono=True),
        "</svg>",
    ]
    return "".join(parts)


def mel_svg(
    spec: CaseSpec,
    before: np.ndarray,
    after: np.ndarray,
    before_sha: str,
    after_sha: str,
) -> str:
    before_mel = mel_spectrogram(before, spec.sample_rate)
    after_mel = mel_spectrogram(after, spec.sample_rate)
    low, high = percentile_range(np.concatenate((before_mel, after_mel), axis=1))
    start, end = spec.window_seconds
    x, width = 116.0, 1058.0
    first_y, second_y, height = 139.0, 356.0, 178.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {MEL_WIDTH} {MEL_HEIGHT}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{html.escape(spec.case_id)} 输入与输出 Mel 频谱</title>',
        f'<desc id="desc">{html.escape(spec.detail)}。64-band Mel，Hann {FFT_SIZE} 点，hop {HOP_SIZE} 点，{high - MEL_DB_RANGE:.0f} 至 {high:.0f} dB。</desc>',
        '<rect width="1200" height="650" fill="#ffffff"/>',
        svg_text(48, 42, "TYPICAL CASE · MEL SPECTROGRAM", size=12, weight=650),
        svg_text(48, 69, spec.case_id, size=19, color="#233248", weight=650, mono=True),
        svg_text(48, 92, spec.detail, size=13),
        svg_text(48, 132, spec.before_label, size=14, color="#31516f", weight=650),
        svg_text(48, 349, spec.after_label, size=14, color="#31516f", weight=650),
        f'<rect x="{x}" y="{first_y}" width="{width}" height="{height}" fill="#0a1834"/>',
        f'<rect x="{x}" y="{second_y}" width="{width}" height="{height}" fill="#0a1834"/>',
        spectrum_cells(before_mel, x, first_y, width, height, low, high),
        spectrum_cells(after_mel, x, second_y, width, height, low, high),
        f'<rect x="{x}" y="{first_y}" width="{width}" height="{height}" fill="none" stroke="#d7e0eb"/>',
        f'<rect x="{x}" y="{second_y}" width="{width}" height="{height}" fill="none" stroke="#d7e0eb"/>',
        svg_text(106, first_y + 8, f"{spec.sample_rate / 2000:g} kHz", size=10, anchor="end"),
        svg_text(106, first_y + height / 2, "0 Hz", size=10, anchor="end"),
        svg_text(106, second_y + 8, f"{spec.sample_rate / 2000:g} kHz", size=10, anchor="end"),
        svg_text(106, second_y + height / 2, "0 Hz", size=10, anchor="end"),
        f'<line x1="{x}" y1="559" x2="{x + width}" y2="559" stroke="#dbe3ed"/>',
        svg_text(x, 580, f"{start:.1f} s", size=11),
        svg_text(x + width / 2, 580, f"{((start + end) / 2):.1f} s", size=11, anchor="middle"),
        svg_text(x + width, 580, f"{end:.1f} s", size=11, anchor="end"),
    ]
    bar_x = 904.0
    bar_y = 589.0
    bar_width = 250.0
    bar_height = 12.0
    for index in range(64):
        left = bar_x + bar_width * index / 64
        right = bar_x + bar_width * (index + 1) / 64
        parts.append(
            f'<rect x="{left:.2f}" y="{bar_y}" width="{right - left + 0.1:.2f}" '
            f'height="{bar_height}" fill="{mel_color(low + (high - low) * index / 63, low, high)}"/>'
        )
    parts.extend(
        (
            svg_text(bar_x, 585, f"{low:.0f} dB", size=9, anchor="start"),
            svg_text(bar_x + bar_width, 585, f"{high:.0f} dB", size=9, anchor="end"),
            svg_text(48, 613, f"Hann {FFT_SIZE} / hop {HOP_SIZE} / 64 Mel bands · shared {MEL_DB_RANGE:g} dB scale · no gain normalization", size=11),
            f'<line x1="48" y1="627" x2="1152" y2="627" stroke="#edf1f5"/>',
            svg_text(48, 640, f"before sha256 {before_sha[:20]}…   after sha256 {after_sha[:20]}…", size=9, color="#8a98aa", mono=True),
            "</svg>",
        )
    )
    return "".join(parts)


def enhancement_case_specs() -> tuple[CaseSpec, ...]:
    report_file = ROOT / "public/evaluation/enhancement-chain/report.json"
    if not report_file.exists():
        return ()
    report = json.loads(report_file.read_text(encoding="utf-8"))
    specs = []
    for variant in report["variants"]:
        for group in variant["caseGroups"]:
            case = group["caseAudio"]
            asset_id = Path(case["waveformSrc"]).name.removesuffix("-waveform.svg")
            specs.append(CaseSpec(
                asset_id, case["id"], case["detail"], repo_path(case["beforeFile"]),
                repo_path(case["afterFile"]), case["sampleRate"], tuple(case["windowSeconds"]),
                case["delaySamples"], case["beforeLabel"], case["afterLabel"],
                analysis=case["analysis"],
            ))
    return tuple(specs)


def agc_case_specs() -> tuple[list[CaseSpec], list[dict[str, object]]]:
    """Bind each displayed state to the configuration that supplied its score."""
    matrix = ROOT / "build/agc-tech-matrix-agc108-20261020"
    variants = [
        ("guarded5", "guarded5", "Guarded5", matrix / "guarded5"),
        ("adaptive-agc2", "agc2_adaptive", "AGC2 Adaptive", matrix / "agc2_adaptive"),
        *[("fixed-digital", f"fixed_{db}", f"Fixed {db} dB", matrix / f"fixed_{db}")
          for db in (0, 3, 6, 8)],
        ("tx-drc", "no_tx_drc", "No TX DRC", matrix / "no_tx_drc"),
        ("utterance-v4", "stage4_release_no_tx_drc", "Stage4 + release + No TX DRC",
         ROOT / "build/agc-stage4-release-no-tx-drc-20261020"),
    ]
    metric_names = {
        "raw_sisdr_db": "SI-SDR (dB)", "stoi_change": "ΔSTOI",
        "projection_gain_min_db": "投影增益下限 (dB)",
        "projection_gain_max_db": "投影增益上限 (dB)",
        "total_attenuation_db": "噪声衰减 (dB)",
    }
    specs: list[CaseSpec] = []
    groups: list[dict[str, object]] = []
    for algorithm, variant, label, root in variants:
        report_path = root / "report.json"
        results_path = root / "results.jsonl"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
        if len(rows) != report["cases"] or sum(row["acceptance"]["passed"] for row in rows) != report["counts"]["pass"]:
            raise ValueError(f"AGC report/results disagree: {root}")
        for passed in (True, False):
            candidates = [row for row in rows if row["acceptance"]["passed"] is passed]
            # Show a repaired/regressed case for an ablation when one exists.
            if algorithm in ("tx-drc", "utterance-v4"):
                transitions = [row for row in candidates if (row["original_status"] == "pass") != passed]
                candidates = transitions or candidates
            row = candidates[0]
            scene = row["scene"]
            failures = [gate for gate in row["acceptance"]["gates"] if gate["status"] == "fail"]
            window_name = "near_late"
            if failures:
                window_name = next(name for name in ("near_late", "near_only", "noise_only")
                                   if any(gate["segment"] == name for gate in failures))
            gates = [gate for gate in (failures or row["acceptance"]["gates"])
                     if gate["segment"] == window_name]
            before = root / row["original_run"] / "cases" / scene["id"] / "before.wav"
            result_dir = before.parent / "runs/full"
            after = result_dir / "result.after.wav"
            engine = json.loads((result_dir / "result.json").read_text(encoding="utf-8"))
            if sha256(before) != scene["files"]["before.wav"]["sha256"] or sha256(after) != row["metrics"]["output_sha256"]:
                raise ValueError(f"AGC WAV differs from frozen evidence: {variant}/{scene['id']}")
            state = "pass" if passed else "fail"
            status = "通过" if passed else "失败"
            asset_id = f"agc-{variant.replace('_', '-')}-{state}"
            gate_text = "；".join(
                f"{metric_names.get(gate['metric'], gate['metric'])} {gate['value']:.4f} "
                f"{gate['operator']} {gate['threshold']:.4f}（{status}）"
                for gate in gates[:3]
            )
            note = (f"{label} 当前冻结配置；{'完整工程门全部通过' if passed else f'完整工程门失败 {len(failures)} 项'}。"
                    f"图示 {window_name} 窗：{gate_text}。"
                    "输入麦克风与最终 VCP 输出配对，包含其他增强模块，不作为隔离 AGC 验收。")
            spec = CaseSpec(
                asset_id, scene["id"],
                f"{label}；{scene['sample_rate'] / 1000:g} kHz；{scene['near_dbfs']} dBFS；"
                f"{scene['noise']}；SNR {scene['snr_db']:+g} dB；{window_name}",
                before, after, int(scene["sample_rate"]), tuple(scene["windows"][window_name]),
                int(engine["dsp_delay_samples"]), "输入 · 冻结麦克风 / 通道 0",
                f"输出 · {label} 冻结 VCP", analysis=f"{status}案例；{gate_text}",
            )
            specs.append(spec)
            groups.append({
                "algorithm": algorithm, "variant": variant, "id": state, "status": status,
                "headline": f"{label} · {status}案例", "note": note,
                "source": str(report_path.relative_to(ROOT)),
                "evidence": {
                    "results": str(results_path.relative_to(ROOT)),
                    "report_sha256": sha256(report_path), "results_sha256": sha256(results_path),
                    "passed": passed, "window": window_name, "gates": gates,
                    "original_status": row["original_status"],
                },
                "caseAudio": {
                    "id": spec.case_id, "detail": spec.detail, "sampleRate": spec.sample_rate,
                    "windowSeconds": list(spec.window_seconds), "delaySamples": spec.delay_samples,
                    "beforeLabel": spec.before_label, "afterLabel": spec.after_label,
                    "beforeFile": str(before.relative_to(ROOT)), "afterFile": str(after.relative_to(ROOT)),
                    "waveformSrc": f"/evaluation/pipeline-cases/{asset_id}-waveform.svg",
                    "melSrc": f"/evaluation/pipeline-cases/{asset_id}-mel.svg", "analysis": spec.analysis,
                },
            })
    return specs, groups


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    agc_specs, agc_groups = agc_case_specs()
    specs = (*SPECS, *enhancement_case_specs(), *agc_specs)
    if len({spec.asset_id for spec in specs}) != len(specs):
        raise ValueError("Duplicate pipeline case asset id")
    manifest: list[dict[str, object]] = []
    for spec in specs:
        if not spec.before.exists():
            raise FileNotFoundError(spec.before)
        if not spec.after.exists():
            raise FileNotFoundError(spec.after)
        before_all = read_channel(
            spec.before, spec.sample_rate, spec.before_channel, spec.before_crop_samples
        )
        after_all = read_channel(
            spec.after, spec.sample_rate, spec.after_channel, spec.before_crop_samples
        )
        before = crop_window(
            before_all, spec.sample_rate, spec.window_seconds, 0
        )
        after = crop_window(
            after_all, spec.sample_rate, spec.window_seconds, spec.delay_samples
        )
        if before.size != after.size:
            raise ValueError(f"{spec.asset_id}: unequal crop lengths")
        finite_nonzero(f"{spec.asset_id} before", before)
        finite_nonzero(f"{spec.asset_id} after", after)
        before_sha = sha256(spec.before)
        after_sha = sha256(spec.after)
        before_mel = mel_spectrogram(before, spec.sample_rate)
        after_mel = mel_spectrogram(after, spec.sample_rate)
        low, high = percentile_range(
            np.concatenate((before_mel, after_mel), axis=1)
        )
        waveform = OUTPUT / f"{spec.asset_id}-waveform.svg"
        mel = OUTPUT / f"{spec.asset_id}-mel.svg"
        waveform.write_text(
            waveform_svg(spec, before, after, before_sha, after_sha), encoding="utf-8"
        )
        mel.write_text(
            mel_svg(spec, before, after, before_sha, after_sha), encoding="utf-8"
        )
        manifest.append(
            {
                "asset_id": spec.asset_id,
                "case_id": spec.case_id,
                "detail": spec.detail,
                "before": str(spec.before.relative_to(ROOT)),
                "after": str(spec.after.relative_to(ROOT)),
                "before_sha256": before_sha,
                "after_sha256": after_sha,
                "before_channel": spec.before_channel,
                "after_channel": spec.after_channel,
                "before_crop_samples": spec.before_crop_samples,
                "sample_rate": spec.sample_rate,
                "window_seconds": list(spec.window_seconds),
                "delay_samples": spec.delay_samples,
                "before_label": spec.before_label,
                "after_label": spec.after_label,
                "analysis": spec.analysis,
                "waveform": f"/evaluation/pipeline-cases/{waveform.name}",
                "mel": f"/evaluation/pipeline-cases/{mel.name}",
                "waveform_bytes": waveform.stat().st_size,
                "mel_bytes": mel.stat().st_size,
                "waveform_rms_dbfs": {
                    "before": float(20 * np.log10(np.sqrt(np.mean(before**2)))),
                    "after": float(20 * np.log10(np.sqrt(np.mean(after**2)))),
                },
                "mel_scope": {
                    "fft_size": FFT_SIZE,
                    "hop_size": HOP_SIZE,
                    "bands": MEL_BANDS,
                    "db_range": MEL_DB_RANGE,
                    "low_db": low,
                    "high_db": high,
                },
            }
        )
        print(
            f"{spec.asset_id}: {before.size} samples, "
            f"{waveform.stat().st_size} + {mel.stat().st_size} bytes"
        )
    (OUTPUT / "manifest.json").write_text(
        json.dumps(
            {
                "scope": "Frozen input/output pairs; declared causal delay removed; no gain normalization.",
                "assets": manifest,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    # The UI imports the same timing and file identities used to draw the SVGs.
    # Keep this compact index separate from plotting statistics and pixel data.
    metadata_keys = (
        "asset_id", "case_id", "before", "after", "before_channel",
        "after_channel", "before_crop_samples", "sample_rate", "window_seconds",
        "delay_samples", "waveform", "mel",
    )
    (ROOT / "src/data/pipeline-case-metadata.json").write_text(
        json.dumps([{key: asset[key] for key in metadata_keys} for asset in manifest],
                   indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (ROOT / "src/data/agc-case-evidence.json").write_text(
        json.dumps(agc_groups, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
