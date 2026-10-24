import type {
  PipelineCaseGroup,
  PipelineEvaluationData,
} from "./PipelineEvaluation";

const NS_CASE_ID = "coverage-16000-validation-2303b3f5-snr+10";
const NS_CASE_DETAIL =
  "CAVEMOVE · Smart；16 kHz；SNR +10 dB；后段语音窗 13–21 s";
const NS_BEFORE =
  "build/controlled-cavemove-20261017/cases/coverage-16000-validation-2303b3f5-snr+10/before.wav";

function nsCase(
  assetId: string,
  afterFile: string,
  afterLabel: string,
  delaySamples: number,
  analysis: string,
) {
  return {
    id: NS_CASE_ID,
    detail: NS_CASE_DETAIL,
    sampleRate: 16000,
    windowSeconds: [13, 21] as [number, number],
    delaySamples,
    beforeLabel: "输入 · 带噪麦克风",
    afterLabel,
    beforeFile: NS_BEFORE,
    afterFile,
    waveformSrc: `/evaluation/pipeline-cases/${assetId}-waveform.svg`,
    melSrc: `/evaluation/pipeline-cases/${assetId}-mel.svg`,
    analysis,
  };
}

function nsCaseGroups(
  assetId: string,
  pass: {
    id: string;
    detail: string;
    afterFile: string;
    delaySamples: number;
    note: string;
    analysis: string;
    windowSeconds?: [number, number];
  },
  fail: {
    id: string;
    detail: string;
    afterFile: string;
    delaySamples: number;
    note: string;
    analysis: string;
    windowSeconds?: [number, number];
  },
) {
  const group = (
    status: "pass" | "fail",
    item: typeof pass,
  ): PipelineCaseGroup => ({
    id: status,
    status: status === "pass" ? ("通过" as const) : ("失败" as const),
    headline: status === "pass" ? "完整工程门限通过" : "完整工程门限失败",
    note: item.note,
    caseAudio: {
      id: item.id,
      detail: item.detail,
      sampleRate: 16000,
      windowSeconds: item.windowSeconds ?? ([13, 21] as [number, number]),
      delaySamples: item.delaySamples,
      beforeLabel: "输入 · 带噪麦克风",
      afterLabel:
        status === "pass" ? "输出 · 通过案例" : "输出 · 失败案例",
      beforeFile: `build/controlled-cavemove-20261017/cases/${item.id}/before.wav`,
      afterFile: item.afterFile,
      waveformSrc: `/evaluation/pipeline-cases/${assetId}-${status}-waveform.svg`,
      melSrc: `/evaluation/pipeline-cases/${assetId}-${status}-mel.svg`,
      analysis: item.analysis,
    },
  });

  return [group("pass", pass), group("fail", fail)];
}

export const nsEvaluations: Record<string, PipelineEvaluationData> = {
  webrtc: {
    tone: "warning",
    status: "692 场景冻结矩阵 · 2026-10-19",
    headline: "635 / 692 场景通过",
    scope: "665 稳态 + 27 瞬时；16 kHz 车噪全集及 24/48 kHz 补充",
    source: "build/ns-all-family-comparison-20261019/summary.json",
    note:
      "clean 67/67、cold 45/67。平均噪声衰减 10.418 dB，平均 STOI 变化 -0.001502；不是 ITU-T 整机认证。",
    metrics: [
      { label: "场景全通过", value: "635 / 692", detail: "完整工程门限" },
      { label: "噪声门失败", value: "12", detail: "噪声衰减不足" },
      { label: "语音门失败", value: "48", detail: "STOI / 投影等" },
    ],
    caseAudio: nsCase(
      "ns-webrtc",
      "build/ns-all-family-comparison-20261019/runtime-serial-complete/webrtc_v5_bounded09-0.after.wav",
      "输出 · WebRTC NS",
      416,
      "语音窗；用于观察谱增益与语音保真",
    ),
    caseGroups: nsCaseGroups(
      "ns-webrtc",
      {
        id: "coverage-16000-validation-2303b3f5-snr+10",
        detail: "CAVEMOVE · Smart；16 kHz；SNR +10 dB；后段语音窗",
        afterFile:
          "build/ns-case-replay-20261020/webrtc_v5_bounded09/coverage-16000-validation-2303b3f5-snr+10/result.after.wav",
        delaySamples: 416,
        note: "同输入哈希、同冻结门限；所有工程门通过。",
        analysis: "通过案例；观察语音窗内的谱增益与保真",
      },
      {
        id: "sweep-16000-validation-3b7f955f-cavemove_smart_forfour_s60_w0-snr-07",
        detail: "CAVEMOVE · Smart；16 kHz；SNR +7 dB；后段语音窗",
        afterFile:
          "build/ns-case-replay-20261020/webrtc_v5_bounded09/sweep-16000-validation-3b7f955f-cavemove_smart_forfour_s60_w0-snr-07/result.after.wav",
        delaySamples: 416,
        note: "失败门为 near_only projection_gain；噪声衰减 11.027 dB，SI-SDR +5.669 dB，STOI −0.00075。",
        analysis: "失败案例；早期语音投影增益不足",
      },
    ),
  },
  speex: {
    tone: "warning",
    status: "692 场景冻结矩阵 · 2026-10-19",
    headline: "608 / 692 场景通过",
    scope: "Speex v5 steady-dual 冻结配置；同一 692 场景",
    source: "build/ns-all-family-comparison-20261019/summary.json",
    note:
      "clean 66/67、cold 50/67。平均噪声衰减 8.824 dB，平均 STOI 变化 +0.003402；请求 suppression 不等于实际衰减。",
    metrics: [
      { label: "场景全通过", value: "608 / 692", detail: "完整工程门限" },
      { label: "噪声门失败", value: "61", detail: "噪声衰减不足" },
      { label: "语音门失败", value: "24", detail: "STOI / 投影等" },
    ],
    caseAudio: nsCase(
      "ns-speex",
      "build/ns-all-family-comparison-20261019/runtime-serial-complete/speex_v5_steadydual-0.after.wav",
      "输出 · Speex steady-dual",
      480,
      "同输入、同时间窗；扣除 480 样本固定延迟",
    ),
    caseGroups: nsCaseGroups(
      "ns-speex",
      {
        id: "coverage-16000-validation-2303b3f5-snr+10",
        detail: "CAVEMOVE · Smart；16 kHz；SNR +10 dB；后段语音窗",
        afterFile:
          "build/ns-case-replay-20261020/speex_v5_steadydual/coverage-16000-validation-2303b3f5-snr+10/result.after.wav",
        delaySamples: 480,
        note: "同输入哈希、同冻结门限；所有工程门通过。",
        analysis: "通过案例；稳态双分支输出",
      },
      {
        id: "noise-step-16000-validation-m1-snr+20",
        detail: "CAVEMOVE；16 kHz；SNR +20 dB；噪声阶跃",
        afterFile:
          "build/ns-case-replay-20261020/speex_v5_steadydual/noise-step-16000-validation-m1-snr+20/result.after.wav",
        delaySamples: 480,
        note: "失败门为 noise_only_after_step attenuation；噪声衰减 9.226 dB，SI-SDR +2.631 dB，STOI −0.00113。",
        analysis: "失败案例；噪声阶跃后的重新收敛不足",
        windowSeconds: [8, 16],
      },
    ),
  },
  rnnoise: {
    tone: "warning",
    status: "代表性回放 · 2026-10-17",
    headline: "6 / 15 场景通过",
    scope: "15 个代表性 NS 场景，显式 --ns-backend rnnoise",
    source: "build/repair-sequential-20261017/ns-r10-rnnoise/report.json",
    note:
      "9/15 仍失败；该结果规模较小，paced=false。高 SNR/说话人语音退化是未替换默认后端的主要原因。",
    metrics: [
      { label: "场景全通过", value: "6 / 15", detail: "代表性样本" },
      { label: "失败场景", value: "9", detail: "未达完整工程门限" },
      { label: "输出变化", value: "15 / 15", detail: "均不同于原输出" },
    ],
    caseAudio: {
      id: "vehicle-16000-development-m1-10",
      detail:
        "CAVEMOVE · Alfa Romeo；16 kHz；SNR +20 dB；后段语音窗 13–21 s",
      sampleRate: 16000,
      windowSeconds: [13, 21],
      delaySamples: 480,
      beforeLabel: "输入 · 带噪麦克风",
      afterLabel: "输出 · RNNoise",
      beforeFile:
        "build/repair-sequential-20261017/ns-r10-rnnoise/controlled-cavemove-20261017/cases/vehicle-16000-development-m1-10/before.wav",
      afterFile:
        "build/repair-sequential-20261017/ns-r10-rnnoise/controlled-cavemove-20261017/cases/vehicle-16000-development-m1-10/runs/full/result.after.wav",
      waveformSrc:
        "/evaluation/pipeline-cases/ns-rnnoise-waveform.svg",
      melSrc: "/evaluation/pipeline-cases/ns-rnnoise-mel.svg",
      analysis: "代表性回放；网络输出经同一后级保护",
    },
    caseGroups: [
      {
        id: "pass",
        status: "通过",
        headline: "代表性通过案例",
        note: "RNNoise 独立 15 场景回放中的完整工程门通过案例；不借用 692 场景矩阵结果。",
        caseAudio: {
          id: "vehicle-16000-development-m1-10",
          detail: "CAVEMOVE · Alfa Romeo；16 kHz；SNR +20 dB；后段语音窗",
          sampleRate: 16000,
          windowSeconds: [13, 21],
          delaySamples: 480,
          beforeLabel: "输入 · 带噪麦克风",
          afterLabel: "输出 · RNNoise",
          beforeFile:
            "build/repair-sequential-20261017/ns-r10-rnnoise/controlled-cavemove-20261017/cases/vehicle-16000-development-m1-10/before.wav",
          afterFile:
            "build/repair-sequential-20261017/ns-r10-rnnoise/controlled-cavemove-20261017/cases/vehicle-16000-development-m1-10/runs/full/result.after.wav",
          waveformSrc:
            "/evaluation/pipeline-cases/ns-rnnoise-pass-waveform.svg",
          melSrc: "/evaluation/pipeline-cases/ns-rnnoise-pass-mel.svg",
          analysis: "通过案例；网络输出经同一后级保护",
        },
      },
      {
        id: "fail",
        status: "失败",
        headline: "代表性失败案例",
        note: "失败门包括 near_only raw_sisdr、double_talk projection_gain 和 far_only linear_erle；这是 15 场景代表集，不是 692 场景矩阵。",
        caseAudio: {
          id: "vehicle-16000-holdout-m1-02",
          detail: "native；16 kHz；holdout；后段语音窗",
          sampleRate: 16000,
          windowSeconds: [13, 21],
          delaySamples: 480,
          beforeLabel: "输入 · 带噪麦克风",
          afterLabel: "输出 · RNNoise",
          beforeFile:
            "build/controlled-native-full-20261017/cases/vehicle-16000-holdout-m1-02/before.wav",
          afterFile:
            "build/repair-sequential-20261017/ns-r10-rnnoise/controlled-native-full-20261017/cases/vehicle-16000-holdout-m1-02/runs/full/result.after.wav",
          waveformSrc:
            "/evaluation/pipeline-cases/ns-rnnoise-fail-waveform.svg",
          melSrc: "/evaluation/pipeline-cases/ns-rnnoise-fail-mel.svg",
          analysis: "失败案例；近端、双讲和远端保护同时暴露问题",
        },
      },
    ],
  },
  ulunas: {
    tone: "warning",
    status: "692 场景直接对照 · 2026-10-19",
    headline: "588 / 692 场景通过",
    scope: "原生 16 kHz UL-UNAS 流式 ONNX；与 GTCRN/DPCRN 同集",
    source: "build/ns-all-family-comparison-20261019/summary.json",
    note:
      "clean 53/67、cold 59/67。平均噪声衰减 27.382 dB，SI-SDR 改善 8.851 dB；语音失败仍多于传统后端。",
    metrics: [
      { label: "场景全通过", value: "588 / 692", detail: "完整工程门限" },
      { label: "噪声门失败", value: "39", detail: "噪声衰减不足" },
      { label: "语音门失败", value: "65", detail: "STOI / 投影等" },
    ],
    caseAudio: nsCase(
      "ns-ulunas",
      "build/ns-all-family-comparison-20261019/runtime-serial-complete/ulunas_direct-0.after.wav",
      "输出 · UL-UNAS",
      480,
      "ONNX 四缓存流式输出；同 692 场景集",
    ),
    caseGroups: nsCaseGroups(
      "ns-ulunas",
      {
        id: "coverage-16000-validation-2303b3f5-snr+10",
        detail: "CAVEMOVE · Smart；16 kHz；SNR +10 dB；后段语音窗",
        afterFile:
          "build/ns-case-replay-20261020/ulunas_direct/coverage-16000-validation-2303b3f5-snr+10/result.after.wav",
        delaySamples: 480,
        note: "同输入哈希、同冻结门限；所有工程门通过。",
        analysis: "通过案例；四缓存流式推理",
      },
      {
        id: "coverage-16000-validation-bae79cad-snr+30",
        detail: "CAVEMOVE · Smart；16 kHz；SNR +30 dB；后段语音窗",
        afterFile:
          "build/ns-case-replay-20261020/ulunas_direct/coverage-16000-validation-bae79cad-snr+30/result.after.wav",
        delaySamples: 480,
        note: "失败门为 near_late raw_sisdr；噪声衰减 8.658 dB，SI-SDR −9.165 dB，STOI −0.00995。",
        analysis: "失败案例；高 SNR 下晚期近端语音失真",
      },
    ),
  },
  omlsa: {
    tone: "warning",
    status: "692 场景冻结矩阵 · 2026-10-19",
    headline: "626 / 692 场景通过",
    scope: "OM-LSA/IMCRA v2 mild；同一 692 场景",
    source: "build/ns-all-family-comparison-20261019/summary.json",
    note:
      "clean 67/67、cold 51/67。平均噪声衰减 7.861 dB，平均 STOI 变化 +0.001806；统计主干仍有噪底突变缺口。",
    metrics: [
      { label: "场景全通过", value: "626 / 692", detail: "完整工程门限" },
      { label: "噪声门失败", value: "51", detail: "噪声衰减不足" },
      { label: "语音门失败", value: "15", detail: "STOI / 投影等" },
    ],
    caseAudio: nsCase(
      "ns-omlsa",
      "build/ns-all-family-comparison-20261019/runtime-serial-complete/omlsa_imcra_v2_mild-0.after.wav",
      "输出 · OM-LSA + IMCRA",
      480,
      "统计噪声估计与 OM-LSA 软增益",
    ),
    caseGroups: nsCaseGroups(
      "ns-omlsa",
      {
        id: "coverage-16000-validation-2303b3f5-snr+10",
        detail: "CAVEMOVE · Smart；16 kHz；SNR +10 dB；后段语音窗",
        afterFile:
          "build/ns-case-replay-20261020/omlsa_imcra_v2_mild/coverage-16000-validation-2303b3f5-snr+10/result.after.wav",
        delaySamples: 480,
        note: "同输入哈希、同冻结门限；所有工程门通过。",
        analysis: "通过案例；统计噪声估计与软增益",
      },
      {
        id: "noise-step-16000-validation-m1-snr+20",
        detail: "CAVEMOVE；16 kHz；SNR +20 dB；噪声阶跃",
        afterFile:
          "build/ns-case-replay-20261020/omlsa_imcra_v2_mild/noise-step-16000-validation-m1-snr+20/result.after.wav",
        delaySamples: 480,
        note: "失败门为 noise_only_after_step attenuation；噪声衰减 6.618 dB，SI-SDR +0.691 dB，STOI −0.00004。",
        analysis: "失败案例；阶跃噪声后的最小值统计追踪不足",
        windowSeconds: [8, 16],
      },
    ),
  },
  "super-gaussian": {
    tone: "warning",
    status: "692 场景冻结矩阵 · 2026-10-19",
    headline: "625 / 692 场景通过",
    scope: "Super-Gaussian balanced；共享 IMCRA 对照",
    source: "build/ns-all-family-comparison-20261019/summary.json",
    note:
      "clean 67/67、cold 50/67。平均噪声衰减 9.127 dB，平均 STOI 变化 +0.002419；当前仍是工程混合而非完整 Super-Gaussian 噪声模型。",
    metrics: [
      { label: "场景全通过", value: "625 / 692", detail: "完整工程门限" },
      { label: "噪声门失败", value: "50", detail: "噪声衰减不足" },
      { label: "语音门失败", value: "17", detail: "STOI / 投影等" },
    ],
    caseAudio: nsCase(
      "ns-super-gaussian",
      "build/ns-all-family-comparison-20261019/runtime-serial-complete/super_gaussian_balanced-0.after.wav",
      "输出 · Super-Gaussian",
      480,
      "共享 IMCRA；仅替换 H1 条件增益",
    ),
    caseGroups: nsCaseGroups(
      "ns-super-gaussian",
      {
        id: "coverage-16000-validation-2303b3f5-snr+10",
        detail: "CAVEMOVE · Smart；16 kHz；SNR +10 dB；后段语音窗",
        afterFile:
          "build/ns-case-replay-20261020/super_gaussian_balanced/coverage-16000-validation-2303b3f5-snr+10/result.after.wav",
        delaySamples: 480,
        note: "同输入哈希、同冻结门限；所有工程门通过。",
        analysis: "通过案例；共享 IMCRA 的条件增益",
      },
      {
        id: "noise-step-16000-validation-m1-snr+20",
        detail: "CAVEMOVE；16 kHz；SNR +20 dB；噪声阶跃",
        afterFile:
          "build/ns-case-replay-20261020/super_gaussian_balanced/noise-step-16000-validation-m1-snr+20/result.after.wav",
        delaySamples: 480,
        note: "失败门为 noise_only_after_step attenuation；噪声衰减 8.041 dB，SI-SDR +0.736 dB，STOI +0.00007。",
        analysis: "失败案例；阶跃噪声后的抑制不足",
        windowSeconds: [8, 16],
      },
    ),
  },
  "hybrid-gru": {
    tone: "warning",
    status: "模型变体对照 · 2026-10-19",
    headline: "639 / 692 最佳变体",
    scope: "Conv + GRU32 + gain loss；与原 GRU32 对照",
    source: "build/ns-all-family-comparison-20261019/summary.json",
    note:
      "原 GRU32 为 629/692，最佳 Conv-Gain 变体为 639/692；平均噪声衰减 15.209 dB。模型仍需显式加载并默认关闭。",
    metrics: [
      { label: "最佳变体通过", value: "639 / 692", detail: "Conv-Gain GRU32" },
      { label: "原 GRU32", value: "629 / 692", detail: "同集对照" },
      { label: "clean / cold", value: "66 / 67 · 51 / 67", detail: "最佳变体" },
    ],
    caseAudio: nsCase(
      "ns-hybrid-gru",
      "build/ns-all-family-comparison-20261019/runtime-serial-complete/conv_gain_gru32-0.after.wav",
      "输出 · Conv-Gain GRU32",
      160,
      "OM-LSA 生成音频；GRU 只修正控制量",
    ),
    caseGroups: nsCaseGroups(
      "ns-hybrid-gru",
      {
        id: "coverage-16000-validation-2303b3f5-snr+10",
        detail: "CAVEMOVE · Smart；16 kHz；SNR +10 dB；后段语音窗",
        afterFile:
          "build/ns-case-replay-20261020/conv_gain_gru32/coverage-16000-validation-2303b3f5-snr+10/result.after.wav",
        delaySamples: 160,
        note: "同输入哈希、同冻结门限；所有工程门通过。",
        analysis: "通过案例；GRU 控制量修正",
      },
      {
        id: "sweep-16000-validation-3b7f955f-cavemove_smart_forfour_s60_w0-snr-08",
        detail: "CAVEMOVE · Smart；16 kHz；SNR +8 dB；后段语音窗",
        afterFile:
          "build/ns-case-replay-20261020/conv_gain_gru32/sweep-16000-validation-3b7f955f-cavemove_smart_forfour_s60_w0-snr-08/result.after.wav",
        delaySamples: 160,
        note: "失败门为 near_only projection_gain；噪声衰减 16.599 dB，SI-SDR +10.157 dB，STOI +0.01241。",
        analysis: "失败案例；强抑制下语音投影仍不足",
      },
    ),
  },
  "webrtc-speex-dual": {
    tone: "pending",
    status: "当前完整矩阵待补",
    headline: "尚无显式双分支结果",
    scope: "WebRTC + Speex 两状态组合，当前默认关闭",
    source: "docs/NS_HYBRID_GTCRN_DESIGN_20261018.md",
    note:
      "现有小样本历史和单后端结果不能代替双分支完整矩阵；没有找到 identity_verified 的当前 692 场景双分支报告。",
    metrics: [
      { label: "当前 692 矩阵", value: "0", detail: "未运行" },
      { label: "默认状态", value: "关闭", detail: "显式开关路径" },
      { label: "净收益", value: "未验证", detail: "不引用单分支成绩" },
    ],
  },
  "min-stat": {
    tone: "pending",
    status: "当前端到端矩阵待补",
    headline: "尚无冻结结果",
    scope: "主后端之后的附加宽带最小值模块",
    source: "docs/NS_HYBRID_GTCRN_DESIGN_20261018.md",
    note:
      "组件存在且默认关闭，但没有在当前 692 场景中对四个后端逐项加入后的冻结通过数；不能把组件单测当成质量收益。",
    metrics: [
      { label: "端到端通过数", value: "未运行", detail: "当前 692 场景" },
      { label: "默认状态", value: "关闭", detail: "辅助模块" },
      { label: "后端独立收益", value: "未验证", detail: "不能复用主后端结果" },
    ],
  },
  residual: {
    tone: "warning",
    status: "四后端门控对照 · 2026-10-19",
    headline: "未准入 · 净 -2",
    scope: "692 场景 × 4 个后端的 SNR 门控残噪衰减",
    source: "build/ns-traditional-residual-20261019-v6/full-gate-summary.json",
    note:
      "四分支通过数为 WebRTC 635→634、Speex 608→607、OM-LSA 626→626、Super-Gaussian 625→625；逐例恢复 4、回退 6。默认 residual_max_db 仍为 0。",
    metrics: [
      { label: "逐例恢复", value: "4", detail: "四后端合计" },
      { label: "逐例回退", value: "6", detail: "四后端合计" },
      { label: "净通过变化", value: "-2", detail: "未准入" },
    ],
    caseAudio: {
      id: "coverage-16000-development-9c7b62f0-snr+30",
      detail:
        "TCAR；16 kHz；SNR +30 dB；后段语音窗 13–21 s；门控残噪恢复案例",
      sampleRate: 16000,
      windowSeconds: [13, 21],
      delaySamples: 480,
      beforeLabel: "输入 · 带噪麦克风",
      afterLabel: "输出 · OM-LSA + 残噪门控",
      beforeFile:
        "build/controlled-native-full-20261017/cases/coverage-16000-development-9c7b62f0-snr+30/before.wav",
      afterFile:
        "build/ns-traditional-residual-20261019-v6/full-gate/omlsa/r3/421f46769689383c7718/result.after.wav",
      waveformSrc:
        "/evaluation/pipeline-cases/ns-residual-waveform.svg",
      melSrc: "/evaluation/pipeline-cases/ns-residual-mel.svg",
      analysis: "通过恢复案例；residual gate SNR -12 dB",
    },
  },
  gtcrn: {
    tone: "warning",
    status: "直接流式对照 · 2026-10-19",
    headline: "496 / 692 场景通过",
    scope: "官方 DNS3 权重，16 kHz，512/256 流式探针",
    source: "build/ns-all-family-comparison-20261019/summary.json",
    note:
      "clean 60/67、cold 54/67。强抑制同时带来 108 个语音门失败；30 ms 信号时延和低频/起音问题仍未解决。",
    metrics: [
      { label: "场景全通过", value: "496 / 692", detail: "完整工程门限" },
      { label: "噪声门失败", value: "88", detail: "噪声衰减不足" },
      { label: "语音门失败", value: "108", detail: "STOI / 投影等" },
    ],
    caseAudio: nsCase(
      "ns-gtcrn",
      "build/ns-all-family-comparison-20261019/runtime-serial-complete/gtcrn_dns3_direct-0.after.wav",
      "输出 · GTCRN DNS3",
      480,
      "512/256 流式直接探针；30 ms 信号时延",
    ),
    caseGroups: nsCaseGroups(
      "ns-gtcrn",
      {
        id: "coverage-16000-validation-2303b3f5-snr+10",
        detail: "CAVEMOVE · Smart；16 kHz；SNR +10 dB；后段语音窗",
        afterFile:
          "build/ns-case-replay-20261020/gtcrn_dns3_direct/coverage-16000-validation-2303b3f5-snr+10/result.after.wav",
        delaySamples: 480,
        note: "同输入哈希、同冻结门限；所有工程门通过。",
        analysis: "通过案例；官方 DNS3 流式模型",
      },
      {
        id: "coverage-16000-validation-755dfafe-snr+30",
        detail: "CAVEMOVE；16 kHz；SNR +30 dB；后段语音窗",
        afterFile:
          "build/ns-case-replay-20261020/gtcrn_dns3_direct/coverage-16000-validation-755dfafe-snr+30/result.after.wav",
        delaySamples: 480,
        note: "失败门为 near_only STOI 与 near_late STOI；噪声衰减 9.302 dB，SI-SDR −8.292 dB，STOI −0.02055。",
        analysis: "失败案例；高 SNR 下语音可懂度退化",
      },
    ),
  },
  dpcrn: {
    tone: "warning",
    status: "直接流式对照 · 2026-10-19",
    headline: "594 / 692 场景通过",
    scope: "作者 smaller stateful TFLite，16 kHz",
    source: "build/ns-all-family-comparison-20261019/summary.json",
    note:
      "噪声门失败 0，但 clean 仅 40/67、cold 64/67，语音门失败 98；强抑制以明显语音保真代价换来。",
    metrics: [
      { label: "场景全通过", value: "594 / 692", detail: "完整工程门限" },
      { label: "噪声门失败", value: "0", detail: "衰减全部达标" },
      { label: "语音门失败", value: "98", detail: "STOI / 投影等" },
    ],
    caseAudio: nsCase(
      "ns-dpcrn",
      "build/ns-all-family-comparison-20261019/runtime-serial-complete/dpcrn_direct-0.after.wav",
      "输出 · DPCRN small",
      480,
      "作者 stateful TFLite；512/256 流式探针",
    ),
    caseGroups: nsCaseGroups(
      "ns-dpcrn",
      {
        id: "coverage-16000-validation-2303b3f5-snr+10",
        detail: "CAVEMOVE · Smart；16 kHz；SNR +10 dB；后段语音窗",
        afterFile:
          "build/ns-case-replay-20261020/dpcrn_direct/coverage-16000-validation-2303b3f5-snr+10/result.after.wav",
        delaySamples: 480,
        note: "同输入哈希、同冻结门限；所有工程门通过。",
        analysis: "通过案例；stateful TFLite 模型",
      },
      {
        id: "coverage-16000-validation-755dfafe-snr+30",
        detail: "CAVEMOVE；16 kHz；SNR +30 dB；后段语音窗",
        afterFile:
          "build/ns-case-replay-20261020/dpcrn_direct/coverage-16000-validation-755dfafe-snr+30/result.after.wav",
        delaySamples: 480,
        note: "失败门为 near_only STOI、near_late raw_sisdr 与 near_late STOI；噪声衰减 23.893 dB，SI-SDR −10.743 dB，STOI −0.03144。",
        analysis: "失败案例；强抑制导致语音质量与可懂度退化",
      },
    ),
  },
  fastenhancer: {
    tone: "pending",
    status: "无当前冻结门槛结果",
    headline: "仅有历史离线探针",
    scope: "48 kHz Tiny 神经网络，Python/ORT 离线质量路径",
    source: "build/technology-survey-20261010/streaming_enhancement-agent-report.md",
    note:
      "技术调研记录称历史探针未正式通过，但没有与当前 692 场景同口径的逐门结果；不能补造通过数或实时数据。",
    metrics: [
      { label: "692 场景矩阵", value: "未运行", detail: "无同口径报告" },
      { label: "在线 C++ 路径", value: "0", detail: "不在 --ns-backend" },
      { label: "正式准入门", value: "未验证", detail: "历史探针不足" },
    ],
  },
  dpdfnet: {
    tone: "pending",
    status: "研究候选 · 未接入",
    headline: "尚无项目评测",
    scope: "8 / 16 / 48 kHz 官方模型候选",
    source: "cpp/tools/estimate_hybrid_ns_budget.py",
    note:
      "当前只有预算/设计方案，没有 C++ 流式实现、模型许可复核或统一 692 场景结果；不能出现在运行时后端选择中。",
    metrics: [
      { label: "生产运行路径", value: "0", detail: "尚未接入" },
      { label: "质量矩阵", value: "未运行", detail: "无冻结模型" },
      { label: "实时矩阵", value: "未运行", detail: "无 C++ 路径" },
    ],
  },
  deepfilternet: {
    tone: "pending",
    status: "研究候选 · 未接入",
    headline: "尚无项目评测",
    scope: "DeepFilterNet 2/3，48 kHz 为主的 Rust 候选",
    source: "build/technology-survey-20261010/streaming_enhancement-agent-report.md",
    note:
      "官方算法的延迟和性能数字不属于本项目目标 CPU；当前没有 C++/Android 流式前端、首帧状态或统一质量门槛结果。",
    metrics: [
      { label: "C++ 流式路径", value: "0", detail: "尚未接入" },
      { label: "统一质量门槛", value: "未运行", detail: "无项目输出" },
      { label: "目标设备实时", value: "未验证", detail: "不引用论文 RTF" },
    ],
  },
};
