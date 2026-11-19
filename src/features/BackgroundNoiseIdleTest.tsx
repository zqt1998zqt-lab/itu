import SceneConditionTest from "./SceneConditionTest";
import type {
  ScopedTestDefinition,
  ScopedTestPageProps,
} from "./scoped-acoustic-model";

export const BACKGROUND_NOISE_IDLE_DEFINITION: ScopedTestDefinition = {
  scope: "background_noise",
  route: "background-noise-idle",
  apiRoute: "background-noise/scenes/sce1",
  previewMode: "scene-condition-20",
  scene: "sce1",
  title: "背景噪声 · 怠速 / 静止",
  eyebrow: "发送方向 · Sce1 单场景",
  description:
    "使用怠速 / 静止场景的 100 种工程组合：近端单讲、远端单讲、持续双讲、单讲/双讲切换、CNG 噪声观测各 20 组。逐组执行 VCP，保存四路输入、TX、适用的质量代理和分段工程诊断。源录音可以跨组合复用，100 组不等于 100 份独立录音。",
  boundary:
    "Sce1 是工程噪声配方标签，不代表真实座舱采集。100 组数据支持全部 8 种业务模式。DNSMOS OVRL/BAK/SIG 仅用于持续近端单讲及双讲的质量代理；远端单讲和切换不产生 G/N/S-MOS。CNG 数字电平/频谱及回声/双讲诊断均未经声学校准。合同项仍为 5 个；正式 ITU-T 合规未测。",
  inputTitle: "Sce1 各条件组输入与对应真实 VCP 发送处理输出（TX）",
  inputs: [
    "100 组样本各自包含麦克风混合输入（microphone）、干净近端参考（reference）、远端播放信号 / 回声参考（far_end）、独立噪声输入（noise）四路音频与共同时间线；按实际通话模式决定近端和远端是否活动。",
    "页面选中的 VCP 预设对每个条件组分别执行真实上行处理，结果只取该条件组自己的实际发送处理输出（TX）；不得用其他场景或其他条件组的发送处理输出（TX）代替。",
    "DNSMOS P.835 OVRL、BAK、SIG 使用所绑定条件组的完整 VCP 发送处理输出（TX），不以 analysis_start_s/window_ms 截短或重复短片；有效片段至少 9.01 s。",
    "CNG 电平与频谱使用 CNG 条件组自己的独立噪声输入（noise）与该条件组 VCP 发送处理输出（TX）；频谱上界按 NB/WB/SWB 分别取 4/8/14 kHz，并重采样到共同采样率。",
    "该批次只覆盖 Sce1 的 5 个合同项，逐项保留条件组、输入哈希、VCP 参数与 CLI 身份、输出哈希及 case/report 身份。",
  ],
  stages: [
    {
      label: "锁定 Sce1 单场景",
      detail:
        "路由、API 请求和合同目录固定为 Sce1；本页只执行该场景 5 个合同测项，不执行也不借用 Sce2–Sce4 的测项或结果。",
    },
    {
      label: "建立同一条件组",
      detail:
        "按合同行识别 speech 或 CNG 条件组，读取各自的麦克风混合输入（microphone）、干净近端参考（reference）、远端播放信号（回声参考，far_end）、独立噪声输入（noise）共同时间线与输入身份；禁止跨场景或跨条件组复用。",
    },
    {
      label: "执行真实 VCP 发送处理（TX）",
      detail:
        "按页面选中的 VCP 预设对 Sce1 每个条件组分别执行实际上行处理，保留 profile、参数、CLI 与输入身份，并读取各条件组自己的发送处理输出（TX）。",
    },
    {
      label: "DNSMOS P.835 三项代理",
      detail:
        "只对近端单讲或持续双讲条件组的完整 VCP 发送处理输出（TX）计算 OVRL、BAK、SIG，分别作为 G-MOS、N-MOS、S-MOS 工程代理；不声明与 TS 103 281 正式指标等同。",
    },
    {
      label: "CNG 电平与频谱代理",
      detail:
        "只由 CNG 条件组的独立噪声输入（noise）与该条件组 VCP 发送处理输出（TX）计算数字 RMS 电平差和归一化 PSD 形状距离，不跨条件组补值。",
    },
  ],
  outputKind: "tx",
  methods: {
    g_mos: {
      label: "怠速 / 静止整体质量代理",
      detail:
        "Sce1 近端单讲或持续双讲条件组完整 VCP 发送处理输出（TX）的 DNSMOS P.835 OVRL，作为 G-MOS 工程代理；不是 ETSI TS 103 281 G-MOS，也不是人工主观评分。页面另显示其与 NV200 客户表数值下限的工程对照，不作为正式合同判定。",
    },
    n_mos: {
      label: "怠速 / 静止背景噪声质量代理",
      detail:
        "Sce1 近端单讲或持续双讲条件组完整 VCP 发送处理输出（TX）的 DNSMOS P.835 BAK，作为 N-MOS 工程代理；不是 ETSI TS 103 281 N-MOS，也不是人工主观评分。页面另显示其与 NV200 客户表数值下限的工程对照，不作为正式合同判定。",
    },
    s_mos: {
      label: "怠速 / 静止噪声中语音质量代理",
      detail:
        "Sce1 近端单讲或持续双讲条件组完整 VCP 发送处理输出（TX）的 DNSMOS P.835 SIG，作为 S-MOS 工程代理；不是 ETSI TS 103 281 S-MOS，也不是人工主观评分。页面另显示其与 NV200 客户表数值下限的工程对照，不作为正式合同判定。",
    },
    cng_level: {
      label: "舒适噪声电平代理",
      detail:
        "在 Sce1 CNG 条件组的共同分析范围内，计算 VCP 发送处理输出（TX）噪声相对该组独立噪声输入（noise）的数字 RMS 电平差；未识别 CNG 状态，也不等同 dBm0(A) 合同结果。",
    },
    cng_spectrum: {
      label: "舒适噪声频谱代理",
      detail:
        "比较 Sce1 CNG 条件组独立噪声输入（noise）与该组 VCP 发送处理输出（TX）噪声段的归一化 PSD Jensen-Shannon 形状距离；不使用总功率差，不等于合同容差掩模最小距离。",
    },
  },
};

export default function BackgroundNoiseIdleTest(
  props: ScopedTestPageProps,
) {
  return (
    <SceneConditionTest
      {...props}
      definition={BACKGROUND_NOISE_IDLE_DEFINITION}
    />
  );
}
