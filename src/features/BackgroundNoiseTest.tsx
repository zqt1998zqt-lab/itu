import ScopedAcousticTest from "./ScopedAcousticTest";
import type {
  ScopedTestDefinition,
  ScopedTestPageProps,
} from "./scoped-acoustic-model";

export const BACKGROUND_NOISE_TEST_DEFINITION: ScopedTestDefinition = {
  scope: "background_noise",
  route: "background-noise",
  title: "背景噪声测试",
  eyebrow: "发送方向 · 背景噪声与舒适噪声",
  description:
    "按当前业务模式分别执行 Sce1–Sce4 四个背景噪声工况的真实 VCP 上行处理；每个场景按绑定 case 分别运行，3 项 DNSMOS P.835 OVRL/BAK/SIG 使用 speech case 的完整发送处理输出（TX）代理 G-MOS、N-MOS、S-MOS，2 项 CNG 使用 cng case 的独立噪声输入（noise）与发送处理输出（TX）计算电平差及归一化频谱形状距离，共 4 场景 × 5 项 = 20 个合同测项。",
  boundary:
    "全部 20 项仅为工程代理，standard_compliance 恒为 false。DNSMOS P.835 OVRL/BAK/SIG 不是 ETSI TS 103 281 的 G-MOS、N-MOS、S-MOS，也不等同于人工主观 MOS；CNG 电平是同一分析范围内 cng case 的独立噪声输入（noise）与 VCP 发送处理输出（TX）的数字 RMS 差，未识别 CNG 开关状态或校准为 dBm0(A)；CNG 频谱是归一化 PSD 的 Jensen-Shannon 形状距离，不使用总功率差，也不等于合同容差掩模最小距离。Sce1–Sce4 的 speech/cng 绑定 case 必须分别使用独立输入、独立条件组与独立的真实发送处理输出（TX）；speech case 的完整发送处理输出（TX）用于 3 项 DNSMOS 质量代理，cng case 的独立噪声输入（noise）与发送处理输出（TX）用于 2 项 CNG 工程测量，禁止跨 case 或跨场景复用输入/输出，也不得用一条发送处理输出（TX）冒充全部 20 项。",
  inputTitle: "四场景按绑定 case 的输入与真实 VCP 发送处理输出（TX）",
  inputs: [
    "当前测试集绑定的业务模式与 Sce1–Sce4 合同条件；场景身份由合同测项决定，页面不硬编码合同单元格。",
    "每个绑定 case 提供麦克风混合输入（microphone）、远端播放信号（回声参考，far_end）、干净近端参考（reference）、VCP 处理后的近端输出（near_end）与独立噪声输入（noise）五路输入并建立共同时间线；speech case 提供完整发送处理输出（TX），cng case 的独立噪声输入（noise）是该 case CNG 电平与频谱计算的输入信号。",
    "所选 VCP 预设按每个绑定 case 的实际麦克风混合输入（microphone）与远端播放信号（回声参考，far_end）路径执行上行发送处理（TX）；结果取该 case 的真实 tx.after.wav，存在固定 DSP 延时时按报告声明裁剪。",
    "G-MOS、N-MOS、S-MOS 使用该场景 speech case 的完整 VCP 发送处理输出（TX，after.wav），不以 analysis_start_s/window_ms 截短或重复短片；DNSMOS 有效片段至少 9.01 s。",
    "CNG 电平与频谱使用同一场景 cng case 的独立噪声输入（noise）与该 case 的 VCP 发送处理输出（TX）；频谱上界按 NB/WB/SWB 分别取 4/8/14 kHz，并按实现重采样到共同采样率。",
    "测量型测试集在 Sce1–Sce4 下共 8 个条件组（每场景 speech + cng），各组保留独立输入哈希、输出哈希及 case/report 身份；自定义测试集可按测项拆分更多条件组，数量不固定，但必须为所选模式覆盖全部 20 个合同测项。",
  ],
  stages: [
    {
      label: "独立场景条件",
      detail: "Sce1–Sce4 各按绑定 case 建立独立条件组与共同时间线；speech 与 cng 的输入/输出身份分别保留，不复用其他 case 或场景的发送处理输出（TX）。",
    },
    {
      label: "真实 VCP 发送处理（TX）",
      detail: "按页面选中的 VCP 预设对每个绑定 case 分别运行实际 CLI，保留 profile、参数、CLI 与输入身份。",
    },
    {
      label: "发送处理输出（TX）对齐",
      detail: "逐 case 读取实际 tx.after.wav；若报告声明固定 DSP 延时，仅裁剪该前置样本并记录原始/分析输出哈希。",
    },
    {
      label: "DNSMOS P.835 代理",
      detail: "对每个场景 speech case 的完整发送处理输出（TX）计算 OVRL、BAK、SIG，分别映射到 G-MOS、N-MOS、S-MOS 工程代理。",
    },
    {
      label: "CNG 工程测量",
      detail: "由每个场景 cng case 的独立噪声输入（noise）与该 case 的发送处理输出（TX）计算数字电平差，并比较归一化 PSD 形状距离。",
    },
  ],
  outputKind: "tx",
  methods: {
    g_mos: {
      label: "背景噪声整体质量代理",
      detail:
        "取该场景 speech case 完整 VCP 发送处理输出（TX）的 DNSMOS P.835 OVRL 作为 G-MOS 工程代理；不是 TS 103 281 G-MOS 或人工主观评分，不套用合同门限。",
    },
    n_mos: {
      label: "背景噪声质量代理",
      detail:
        "取该场景 speech case 完整 VCP 发送处理输出（TX）的 DNSMOS P.835 BAK 作为 N-MOS 工程代理；不是 TS 103 281 N-MOS 或人工主观评分，不套用合同门限。",
    },
    s_mos: {
      label: "噪声中语音质量代理",
      detail:
        "取该场景 speech case 完整 VCP 发送处理输出（TX）的 DNSMOS P.835 SIG 作为 S-MOS 工程代理；不是 TS 103 281 S-MOS 或人工主观评分，不套用合同门限。",
    },
    cng_level: {
      label: "舒适噪声电平代理",
      detail:
        "在同一合同分析范围内计算该场景 cng case 的 VCP 发送处理输出（TX）噪声相对该 case 独立噪声输入（noise）的数字 RMS 电平差；工程代理，未识别 CNG 状态，也不等同 dBm0(A) 合同结果。",
    },
    cng_spectrum: {
      label: "舒适噪声频谱代理",
      detail:
        "比较该场景 cng case 的独立噪声输入（noise）与其 VCP 发送处理输出（TX）噪声段归一化 PSD 的 Jensen-Shannon 形状距离；不使用总功率差，不等于合同容差掩模最小距离，不套用 0～120 门限。",
    },
  },
};

export default function BackgroundNoiseTest(props: ScopedTestPageProps) {
  return (
    <ScopedAcousticTest
      {...props}
      definition={BACKGROUND_NOISE_TEST_DEFINITION}
    />
  );
}
