import details from "../data/metric-details.json" with { type: "json" };

export type MetricId = `M${string}`;
export type ImprovementDirection =
  | "higher"
  | "lower"
  | "target"
  | "unconfirmed";
export interface Metric {
  id: MetricId;
  name: string;
  group: string;
  symbol: string;
  formula: string;
  description: string;
  direction: ImprovementDirection;
  method: string;
  requirements: string;
}

const definitions = [
  [
    "M01",
    "往返时延",
    "时延",
    "RTD",
    "D = 1000 × τ / fₛ",
    "信号发出到返回所经历的时间。关注上限超出量，不能把往返时延直接等分成单向时延。",
    "lower",
  ],
  [
    "M02",
    "发送响度",
    "响度",
    "SLR",
    "9 ≤ SLR ≤ 17 dB",
    "发送端的标准响度评价。13 dB 可作为调试参考目标，正式目标需确认。",
    "target",
  ],
  [
    "M03",
    "接收响度",
    "响度",
    "RLR",
    "−2 ≤ RLRₙₒₘ ≤ 6 dB",
    "标称与最大音量分别评价。响度并非越大越好，最大音量的目标需单独确认。",
    "target",
  ],
  [
    "M04",
    "接收响度线性偏差",
    "响度",
    "ΔRLR",
    "|ΔRLR| ≤ 0.5 dB",
    "输入电平改变 ±5 dB 后，相对理论变化的偏差。越接近零，电平响应越稳定。",
    "target",
  ],
  [
    "M05",
    "频响合规余量",
    "频谱",
    "FR",
    "m = minᶠ min(R(f) − L(f), U(f) − R(f))",
    "逐频率与上下限掩模比较。完整曲线和仪器余量符号确认前，不自动推断曲线通过。",
    "unconfirmed",
  ],
  [
    "M06",
    "空闲信道噪声",
    "底噪与信噪",
    "Noise",
    "Lₙ = 20 log₁₀(qᵣₘₛ / qᵣₑ𝒻)",
    "静音时的校准噪声电平。50 ms、100 ms 及上/下行必须分别评价。",
    "lower",
  ],
  [
    "M07",
    "噪声谱峰余量",
    "底噪与信噪",
    "Peak",
    "m = minᶠ(Uₙ(f) − N(f))",
    "检测总噪声较低但存在尖锐谱峰的情况。保留最严重频率和原始量纲。",
    "unconfirmed",
  ],
  [
    "M08",
    "安静语音质量",
    "语音质量",
    "MOS",
    "MOS ≥ 3 · SWB POLQA ≥ 3.6",
    "上下行分别评价。TMOS 与 POLQA 的算法和版本需保留，不能直接混成一个平均数。",
    "higher",
  ],
  [
    "M09",
    "带外信号",
    "频谱",
    "OOB",
    "保留仪器 Corrected value 与原表限值",
    "区分绝对电平、衰减量和仪器修正值。它们的优劣方向可能不同。",
    "unconfirmed",
  ],
  [
    "M10",
    "双讲目标类别占比",
    "回声与双讲",
    "DT",
    "P = 100 × 有效目标计数 / 有效双讲计数",
    "NB 按 A1+A2，WB/SWB 按 A1；标称与最大音量分开，分母不能为零。",
    "higher",
  ],
  [
    "M11",
    "终端回声耦合损耗",
    "回声与双讲",
    "TCL",
    "TCL ≥ 50 dB",
    "关注标称与最大音量下的回声控制。TCL/TCLw 不能用 AEC 的 ERLE 直接代替。",
    "higher",
  ],
  [
    "M12",
    "信噪比",
    "底噪与信噪",
    "SNR",
    "SNR = Lₛ − Lₙ ≥ 6 dB",
    "使用相同标定、加权和规定窗口评价语音与噪声的电平差。",
    "higher",
  ],
  [
    "M13",
    "背景噪声下整体质量",
    "噪声下语音",
    "G-MOS",
    "分别记录 G-MOS(Sce1…Sce4)",
    "同时保留各场景值和最差场景，不能用 S-MOS 与 N-MOS 的平均数替代。",
    "higher",
  ],
  [
    "M14",
    "背景噪声质量",
    "噪声下语音",
    "N-MOS",
    "N-MOS ≥ 对应模式 / 场景下限",
    "评价背景噪声的感知质量。改善时也应检查语音是否被过度抑制。",
    "higher",
  ],
  [
    "M15",
    "背景噪声下语音质量",
    "噪声下语音",
    "S-MOS",
    "S-MOS ≥ 对应模式 / 场景下限",
    "评价噪声下的语音保真，与 N-MOS 并列展示，避免只追求降噪。",
    "higher",
  ],
  [
    "M16",
    "舒适噪声电平误差",
    "舒适噪声",
    "ΔCN",
    "ΔL = L꜀ₙ − Lᵣₑ𝒻 · −5 ≤ ΔL ≤ 2",
    "舒适噪声相对参考的电平调整。目标接近零，参考噪声与单位需确认。",
    "target",
  ],
  [
    "M17",
    "舒适噪声频谱余量",
    "舒适噪声",
    "CN-FR",
    "m = minᶠ min(ΔN(f) − L(f), U(f) − ΔN(f))",
    "用规定频谱掩模衡量舒适噪声谱形。原表最小距离不能被当作平均谱误差。",
    "unconfirmed",
  ],
] as const;

export const METRICS: Metric[] = definitions.map(
  ([id, name, group, symbol, formula, description, direction]) => {
    const detail = details.find((item) => item.id === id)!;
    return {
      id,
      name,
      group,
      symbol,
      formula,
      description,
      direction,
      method: detail.method,
      requirements: detail.requirements,
    };
  },
);
export const METRIC_MAP = Object.fromEntries(
  METRICS.map((item) => [item.id, item]),
) as Record<MetricId, Metric>;
export const GROUPS = [...new Set(METRICS.map((item) => item.group))];

export function classifyMetric(name: string): MetricId {
  const n = name.toLowerCase();
  if (n.includes("round trip delay")) return "M01";
  if (n.includes("variation of rlr")) return "M04";
  if (n.includes("loudness rating - sending")) return "M02";
  if (n.includes("loudness rating - receiving")) return "M03";
  if (n.includes("frequency response") || n.includes("freq. response"))
    return "M05";
  if (n.includes("idle channel noise"))
    return n.includes("peaks") ? "M07" : "M06";
  if (n.includes("out-of-band")) return "M09";
  if (n.includes("one way") || n.includes("listening speech qual"))
    return "M08";
  if (n.includes("attn. range")) return "M10";
  if (n.includes("terminal coupl") || n.includes("tcl, nom.")) return "M11";
  if (n.includes("s/n calculation")) return "M12";
  if (n.includes("gmos")) return "M13";
  if (n.includes("nmos")) return "M14";
  if (n.includes("smos")) return "M15";
  if (n.includes("comfort noise"))
    return n.includes("level adj.") ? "M16" : "M17";
  throw new Error(`无法匹配指标：${name}`);
}
