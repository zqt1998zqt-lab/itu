export const ACOUSTIC_SIGNAL_LABELS = {
  microphone: "麦克风混合输入",
  far_end: "远端播放信号 / 回声参考",
  reference: "干净近端参考",
  noise: "独立噪声输入",
  near_end: "VCP 处理后的近端输出",
  before: "麦克风混合输入",
  after: "处理后发送输出",
  echo: "远端播放信号 / 回声参考",
} as const;

export type AcousticSignalRole = keyof typeof ACOUSTIC_SIGNAL_LABELS;

export const ACOUSTIC_OUTPUT_LABELS = {
  tx: "发送处理输出（TX）",
  rx: "接收处理输出（RX）",
} as const;

export type AcousticOutputDirection = keyof typeof ACOUSTIC_OUTPUT_LABELS;

export const ACOUSTIC_SIGNAL_DESCRIPTIONS = {
  microphone: "麦克风混合输入，保留近端语音、回声与背景噪声。",
  far_end:
    "远端播放信号（far_end）/ 回声参考，是送往扬声器、用于回声处理的同步信号。",
  reference:
    "干净近端参考（reference），是近端输入的真值，用于时延、保真与可懂度对照。",
  noise: "独立噪声输入，用于噪声功率、谱变化和抑制分析。",
  near_end:
    "VCP 处理后的近端输出（near_end），是 VCP 发送链路完成处理后的结果。",
} as const;
