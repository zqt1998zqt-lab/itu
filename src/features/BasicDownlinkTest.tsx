import ScopedAcousticTest from "./ScopedAcousticTest";
import type {
  ScopedTestDefinition,
  ScopedTestPageProps,
} from "./scoped-acoustic-model";

export const BASIC_DOWNLINK_DEFINITION: ScopedTestDefinition = {
  scope: "basic_downlink",
  route: "basic-downlink",
  title: "基础下行测试",
  eyebrow: "接收方向 · 基础声学",
  description:
    "远端标准信号按独立接收处理（RX）配置经过真实 VCP 下行链，再以 rx-spk 数字输出驱动虚拟双耳声场，计算接收响度、线性度、频响、空闲噪声、语音质量与带外工程代理。",
  boundary:
    "全部结果均为真实 VCP 接收处理（RX）的 rx-spk 数字输出与软件虚拟双耳/模型代理；没有物理 HATS、人工耳或声学校准，不构成 DRP/ERP 实测或正式 ITU-T 合规。DNSMOS P.835 不是 TMOS/POLQA，数字 dBFS 噪声与 RLR 代理也不替代校准声压级。",
  inputTitle: "远端标准信号、独立接收处理（RX）配置与合同工况",
  inputs: [
    "当前测试集绑定的 NB/WB/SWB 远端标准 WAV；模式、场景、音量与时间窗由合同测项决定。",
    "接收处理（RX）独立配置以 CLI --rx-in 路径和实际接收处理（RX）profile/参数为准；所选发送处理（TX）预设中的 BF、AEC、AGC 与 EQ 不配置下行接收链。",
    "接收声学配置：未校准 head 双耳代理，或成对 left/right HRIR、BRIR、端到端 IR。",
    "RLR 线性度仅出现在 NB/WB 合同目录：按当前目录执行同一工况 nominal、-5 dB、+5 dB 三档独立接收处理（RX）运行；SWB 不含该项。",
    "空闲噪声使用合同时间窗：NB/WB 为 100 ms；SWB 同时包含 50 ms 与 100 ms。",
    "接收频响使用已知远端激励作为参考；没有有效参考时不把输出频谱冒充传递函数。",
  ],
  stages: [
    {
      label: "远端标准信号",
      detail: "按 NB/WB/SWB 合同模式、场景与电平提供真实远端输入。",
    },
    {
      label: "真实 VCP 接收处理（RX）",
      detail: "接收处理（RX）下行链为 NR → EQ 槽 → DRC 与内嵌软限幅 → rx-spk；EQ 槽当前默认平直旁路。该链读取独立接收处理配置，所选发送处理（TX）预设的 BF、AEC、AGC 与 EQ 不配置接收处理链，也不能据此推断接收处理参数已被改变。",
    },
    {
      label: "rx-spk 数字输出",
      detail: "读取实际 rx-spk 数字 WAV；与 回声参考（远端播放信号）使用同一后处理 tap。",
    },
    {
      label: "虚拟 DRP 双耳近似",
      detail: "Woodworth ITD + 头影 shelf，或成对 IR 卷积；仅软件代理，无物理 HATS。",
    },
    {
      label: "合同工程测量",
      detail: "双耳输出按 family 进入数字电平、H1、噪声窗、DNSMOS 与带外计算。",
    },
  ],
  outputKind: "rx",
  methods: {
    rlr: {
      label: "接收响度",
      detail: "真实 VCP 接收处理（RX）→ 虚拟双耳 → 双耳数字 RMS；未校准，不等同 P.79 RLR。",
    },
    rlr_linearity: {
      label: "RLR 电平线性度",
      detail: "仅 NB/WB 合同目录：nominal/-5/+5 dB 三档独立接收处理输出（RX），共窗比较三分之一倍频程电平残差；SWB 不含该项。",
    },
    frequency_response: {
      label: "接收频响",
      detail: "已知远端激励与双耳输出做 Welch/CSD H1，按相干门控汇总三分之一倍频程跨度。",
    },
    listening_quality: {
      label: "单向语音质量",
      detail: "双耳分别运行 DNSMOS P.835，取最差耳 OVRL 预测；不是 TMOS 或 POLQA。",
    },
    idle_noise_level: {
      label: "空闲噪声电平",
      detail: "合同 50/100 ms 窗内计算双耳未计权数字 RMS；dBFS 代理，不是 dBPa。",
    },
    idle_noise_peaks: {
      label: "空闲噪声峰值",
      detail: "同一合同时间窗计算时域峰值减 RMS 的突出度；不是合同噪声曲线。",
    },
    out_of_band: {
      label: "接收带外信号",
      detail: "双耳 Welch 频谱计算模式上限以上能量占比；工程诊断，不是 Corrected value。",
    },
  },
};

export default function BasicDownlinkTest(props: ScopedTestPageProps) {
  return (
    <ScopedAcousticTest
      {...props}
      definition={BASIC_DOWNLINK_DEFINITION}
    />
  );
}
