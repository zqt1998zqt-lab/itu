import ScopedAcousticTest from "./ScopedAcousticTest";
import type {
  ScopedTestDefinition,
  ScopedTestPageProps,
} from "./scoped-acoustic-model";

export const BACKGROUND_NOISE_HIGHWAY_DEFINITION: ScopedTestDefinition = {
  scope: "background_noise",
  route: "background-noise-highway",
  apiRoute: "background-noise/scenes/sce3",
  scene: "sce3",
  title: "背景噪声 · 120 km/h 高速",
  eyebrow: "发送方向 · Sce3 单场景",
  description:
    "本页只执行背景噪声 Sce3 场景的 5 个合同测项：3 项 DNSMOS P.835 OVRL/BAK/SIG 工程代理，以及 2 项 CNG 电平/频谱工程代理。输入取同一条件组的麦克风混合输入（microphone）、干净近端参考（reference）、远端播放信号（回声参考，far_end）、独立噪声输入（noise）与该条件组实际 VCP 发送处理输出（TX）；禁止复用其他场景或其他条件组的输入和输出。",
  boundary:
    "“120 km/h 高速”是 Sce3 的工程噪声配方标签，不证明使用了真实 120 km/h 实车录音。本页 5 项均为工程代理，standard_compliance 恒为 false；正式 ITU-T 合规尚未测试。DNSMOS P.835 OVRL/BAK/SIG 分别只是 G-MOS、N-MOS、S-MOS 的工程代理，不等同于 ETSI TS 103 281 的 G-MOS、N-MOS、S-MOS，也不等同于人工主观 MOS。CNG 电平是同一分析范围内独立噪声输入（noise）与 VCP 发送处理输出（TX）的数字 RMS 差，未识别 CNG 开关状态或校准为 dBm0(A)；CNG 频谱是归一化 PSD 的 Jensen-Shannon 形状距离，不使用总功率差，也不等于合同容差掩模最小距离。",
  inputTitle: "Sce3 同一条件组输入与真实 VCP 发送处理输出（TX）",
  inputs: [
    "Sce3 同一条件组提供麦克风混合输入（microphone）、干净近端参考（reference）、远端播放信号（回声参考，far_end）、独立噪声输入（noise）四路输入，并保留共同时间线与输入身份。",
    "页面选中的 VCP 预设对该同一条件组执行真实上行处理，结果取该条件组的实际 tx.after.wav；不得用其他场景或其他条件组的发送处理输出（TX）代替。",
    "DNSMOS P.835 OVRL、BAK、SIG 使用该条件组完整 VCP 发送处理输出（TX），不以 analysis_start_s/window_ms 截短或重复短片；有效片段至少 9.01 s。",
    "CNG 电平与频谱使用该条件组的独立噪声输入（noise）与该条件组 VCP 发送处理输出（TX）；频谱上界按 NB/WB/SWB 分别取 4/8/14 kHz，并重采样到共同采样率。",
    "该批次只覆盖 Sce3 的 5 个合同项，分别保留输入哈希、VCP 参数与 CLI 身份、输出哈希及 case/report 身份。",
  ],
  stages: [
    {
      label: "锁定 Sce3 单场景",
      detail:
        "路由、API 请求和合同目录固定为 Sce3；本页只执行该场景 5 个合同测项，不执行也不借用 Sce1、Sce2、Sce4 的测项或结果。",
    },
    {
      label: "建立同一条件组",
      detail:
        "读取麦克风混合输入（microphone）、干净近端参考（reference）、远端播放信号（回声参考，far_end）、独立噪声输入（noise）的共同时间线与输入身份；所有 5 项只使用该条件组，不复用其他场景或其他条件组。",
    },
    {
      label: "执行真实 VCP 发送处理（TX）",
      detail:
        "按页面选中的 VCP 预设对 Sce3 同一条件组执行实际上行处理，保留 profile、参数、CLI 与输入身份，并读取该条件组的 tx.after.wav。",
    },
    {
      label: "DNSMOS P.835 三项代理",
      detail:
        "只对 Sce3 同一条件组的完整 VCP 发送处理输出（TX）计算 OVRL、BAK、SIG，分别作为 G-MOS、N-MOS、S-MOS 工程代理；不声明与 TS 103 281 正式指标等同。",
    },
    {
      label: "CNG 电平与频谱代理",
      detail:
        "只由 Sce3 同一条件组的独立噪声输入（noise）与该条件组 VCP 发送处理输出（TX）计算数字 RMS 电平差和归一化 PSD 形状距离，不跨条件组补值。",
    },
  ],
  outputKind: "tx",
  methods: {
    g_mos: {
      label: "高速背景整体质量代理",
      detail:
        "Sce3 同一条件组完整 VCP 发送处理输出（TX）的 DNSMOS P.835 OVRL，作为 G-MOS 工程代理；不是 ETSI TS 103 281 G-MOS，也不是人工主观评分，不套用合同门限。",
    },
    n_mos: {
      label: "高速背景噪声质量代理",
      detail:
        "Sce3 同一条件组完整 VCP 发送处理输出（TX）的 DNSMOS P.835 BAK，作为 N-MOS 工程代理；不是 ETSI TS 103 281 N-MOS，也不是人工主观评分，不套用合同门限。",
    },
    s_mos: {
      label: "高速噪声中语音质量代理",
      detail:
        "Sce3 同一条件组完整 VCP 发送处理输出（TX）的 DNSMOS P.835 SIG，作为 S-MOS 工程代理；不是 ETSI TS 103 281 S-MOS，也不是人工主观评分，不套用合同门限。",
    },
    cng_level: {
      label: "舒适噪声电平代理",
      detail:
        "在 Sce3 同一条件组的共同分析范围内，计算 VCP 发送处理输出（TX）噪声相对独立噪声输入（noise）的数字 RMS 电平差；未识别 CNG 状态，也不等同 dBm0(A) 合同结果。",
    },
    cng_spectrum: {
      label: "舒适噪声频谱代理",
      detail:
        "比较 Sce3 同一条件组独立噪声输入（noise）与 VCP 发送处理输出（TX）噪声段的归一化 PSD Jensen-Shannon 形状距离；不使用总功率差，不等于合同容差掩模最小距离。",
    },
  },
};

export default function BackgroundNoiseHighwayTest(
  props: ScopedTestPageProps,
) {
  return (
    <ScopedAcousticTest
      {...props}
      definition={BACKGROUND_NOISE_HIGHWAY_DEFINITION}
    />
  );
}
