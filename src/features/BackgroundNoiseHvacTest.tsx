import ScopedAcousticTest from "./ScopedAcousticTest";
import type {
  ScopedTestDefinition,
  ScopedTestPageProps,
} from "./scoped-acoustic-model";

export const BACKGROUND_NOISE_HVAC_DEFINITION: ScopedTestDefinition = {
  scope: "background_noise",
  route: "background-noise-hvac",
  apiRoute: "background-noise/scenes/sce4",
  scene: "sce4",
  title: "背景噪声 · 空调大风档",
  eyebrow: "发送方向 · 背景噪声 · Sce4",
  description:
    "单场景只执行 Sce4 的 5 个合同项：3 项 DNSMOS P.835 OVRL/BAK/SIG 工程代理，以及 2 项 CNG 电平/频谱工程代理。Sce4 的 speech 与 cng 绑定 case 分别使用其麦克风混合输入（microphone）、干净近端参考（reference）、远端播放信号（回声参考，far_end）、独立噪声输入（noise）和对应真实 VCP 发送处理输出（TX），不得复用其他场景、其他条件组的输入或输出。",
  boundary:
    "Sce4 / 空调大风档是工程噪声配方标签，不证明已采集真实座舱空调大风档录音，也不代表量产环境覆盖。全部 5 项仅为工程代理，standard_compliance 恒为 false；正式 ITU-T 合规未测。DNSMOS P.835 OVRL/BAK/SIG 不是 ETSI TS 103 281 的 G-MOS、N-MOS、S-MOS，也不等同于人工主观 MOS。CNG 电平是同一分析范围内独立噪声输入（noise）与 VCP 发送处理输出（TX）的数字 RMS 差，未识别 CNG 开关状态或校准为 dBm0(A)；CNG 频谱是归一化 PSD 的 Jensen-Shannon 形状距离，不使用总功率差，也不等于合同容差掩模最小距离。speech case 的完整发送处理输出（TX）用于 3 项 DNSMOS 代理，cng case 的独立噪声输入（noise）与其发送处理输出（TX）用于 2 项 CNG 测量，禁止跨 case、条件组或场景复用。",
  inputTitle: "Sce4 绑定条件组的输入与真实 VCP 发送处理输出（TX）",
  inputs: [
    "当前测试集绑定的 Sce4 合同条件及 speech/cng case；5 个合同项由单场景合同目录确定，页面不硬编码合同单元格。",
    "Sce4 绑定条件组提供麦克风混合输入（microphone）、干净近端参考（reference）、远端播放信号（回声参考，far_end）、独立噪声输入（noise）四路输入并建立共同时间线；speech case 的完整发送处理输出（TX）用于 3 项 DNSMOS，cng case 的独立噪声输入（noise）与发送处理输出（TX）用于 2 项 CNG 工程测量。",
    "所选 VCP 预设按每个绑定 case 的实际麦克风混合输入（microphone）与远端播放信号（回声参考，far_end）路径执行上行发送处理（TX）；结果取该 case 的真实 tx.after.wav，存在固定 DSP 延时时按报告声明裁剪。",
    "DNSMOS P.835 OVRL/BAK/SIG 使用该 Sce4 speech case 的完整 VCP 发送处理输出（TX，after.wav），不以 analysis_start_s/window_ms 截短或重复短片；DNSMOS 有效片段至少 9.01 s。",
    "CNG 电平与频谱使用同一 Sce4 条件组 cng case 的独立噪声输入（noise）与该 case 的 VCP 发送处理输出（TX）；频谱上界按 NB/WB/SWB 分别取 4/8/14 kHz，并按实现重采样到共同采样率。",
    "每个绑定 case 保留独立输入哈希、输出哈希、case/report 身份及 VCP profile、参数与 CLI 身份；禁止用其他场景的发送处理输出（TX）或同一条发送处理输出（TX）冒充本场景全部 5 项。",
  ],
  stages: [
    {
      label: "Sce4 单场景条件",
      detail:
        "只建立 Sce4 的 speech/cng 绑定 case 条件组与共同时间线，分别保留输入、输出和 case/report 身份，不加载或复用 Sce1–Sce3 的发送处理输出（TX）。",
    },
    {
      label: "真实 VCP 发送处理（TX）",
      detail:
        "按页面选中的 VCP 预设对 Sce4 的每个绑定 case 分别运行实际 CLI，保留 profile、参数、CLI 与输入身份。",
    },
    {
      label: "发送处理输出（TX）对齐",
      detail:
        "逐 case 读取实际 tx.after.wav；若报告声明固定 DSP 延时，仅裁剪该前置样本并记录原始/分析输出哈希。",
    },
    {
      label: "DNSMOS P.835 代理",
      detail:
        "对 Sce4 speech case 的完整发送处理输出（TX）计算 OVRL、BAK、SIG，分别作为 G-MOS、N-MOS、S-MOS 的工程代理。",
    },
    {
      label: "CNG 工程测量",
      detail:
        "由 Sce4 cng case 的独立噪声输入（noise）与该 case 的发送处理输出（TX）计算数字电平差，并比较归一化 PSD 形状距离。",
    },
  ],
  outputKind: "tx",
  methods: {
    g_mos: {
      label: "空调大风档整体质量代理",
      detail:
        "取 Sce4 speech case 完整 VCP 发送处理输出（TX）的 DNSMOS P.835 OVRL 作为 G-MOS 工程代理；不是 TS 103 281 G-MOS 或人工主观评分，不套用合同门限。",
    },
    n_mos: {
      label: "空调大风档背景质量代理",
      detail:
        "取 Sce4 speech case 完整 VCP 发送处理输出（TX）的 DNSMOS P.835 BAK 作为 N-MOS 工程代理；不是 TS 103 281 N-MOS 或人工主观评分，不套用合同门限。",
    },
    s_mos: {
      label: "空调大风档噪声中语音质量代理",
      detail:
        "取 Sce4 speech case 完整 VCP 发送处理输出（TX）的 DNSMOS P.835 SIG 作为 S-MOS 工程代理；不是 TS 103 281 S-MOS 或人工主观评分，不套用合同门限。",
    },
    cng_level: {
      label: "舒适噪声电平代理",
      detail:
        "在同一合同分析范围内计算 Sce4 cng case 的 VCP 发送处理输出（TX）噪声相对该 case 独立噪声输入（noise）的数字 RMS 电平差；工程代理，未识别 CNG 状态，也不等同 dBm0(A) 合同结果。",
    },
    cng_spectrum: {
      label: "舒适噪声频谱代理",
      detail:
        "比较 Sce4 cng case 的独立噪声输入（noise）与其 VCP 发送处理输出（TX）噪声段归一化 PSD 的 Jensen-Shannon 形状距离；不使用总功率差，不等于合同容差掩模最小距离，不套用 0～120 门限。",
    },
  },
};

export default function BackgroundNoiseHvacTest(
  props: ScopedTestPageProps,
) {
  return (
    <ScopedAcousticTest
      {...props}
      definition={BACKGROUND_NOISE_HVAC_DEFINITION}
    />
  );
}
