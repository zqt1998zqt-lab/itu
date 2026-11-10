import type { NV200SheetData, NV200TestItem } from "../domain/nv200-table.ts";
import { ACOUSTIC_SIGNAL_LABELS } from "../domain/acoustic-terminology.ts";

export const FIVE_SIGNAL_MODES = [
  "NB-wired",
  "NB-wireless",
  "NB-bt",
  "WB-wired",
  "WB-wireless",
  "WB-bt",
  "SWB-wired",
  "SWB-wireless",
] as const;

export type FiveSignalMode = (typeof FIVE_SIGNAL_MODES)[number];
export const FIVE_SIGNAL_CONTRACT_DIRECTION_LABELS = {
  uplink: "上行",
  downlink: "下行",
  overall: "总体",
} as const;
export type FiveSignalContractDirection =
  keyof typeof FIVE_SIGNAL_CONTRACT_DIRECTION_LABELS;

export type FiveSignalContractGroup = {
  direction: FiveSignalContractDirection;
  label: string;
  count: number;
  detail: string;
};

export const FIVE_SIGNAL_CONTRACT_GROUPS: readonly FiveSignalContractGroup[] = [
  {
    direction: "uplink",
    label: "上行基础质量",
    count: 7,
    detail: "SLR、SND 频响、SND TMOS、上行底噪、谱峰与带外信号",
  },
  {
    direction: "uplink",
    label: "双讲透传",
    count: 2,
    detail: "标称音量与最大音量双讲",
  },
  {
    direction: "uplink",
    label: "背景噪声与舒适噪声",
    count: 20,
    detail: "Sce1–Sce4，每个场景 G-MOS、N-MOS、S-MOS、舒适噪声电平与频谱",
  },
  {
    direction: "downlink",
    label: "下行接收质量",
    count: 9,
    detail: "RLR、RLR 线性度、RCV 频响、底噪、谱峰、TMOS 与带外信号",
  },
  {
    direction: "overall",
    label: "往返时延",
    count: 1,
    detail: "Round Trip Delay",
  },
  {
    direction: "overall",
    label: "回声耦合",
    count: 2,
    detail: "TCLw 标称音量与最大音量",
  },
  {
    direction: "overall",
    label: "声学信噪比",
    count: 1,
    detail: "HATS Speech Level / S/N",
  },
] as const;

export const FIVE_SIGNAL_CONTRACT_DIRECTION_TOTALS = {
  uplink: 29,
  downlink: 9,
  overall: 4,
} as const;

export type FiveSignalStatus =
  | "pass"
  | "fail"
  | "review_required"
  | "not_measured"
  | "invalid";

export type FiveSignalSummary = {
  total: number;
  measured: number;
  pass: number;
  fail: number;
  review_required: number;
  not_measured: number;
  invalid: number;
};

export type FiveSignalFlowModeState = {
  mode: FiveSignalMode;
  evaluation_id: string;
  summary: FiveSignalSummary | null;
  valid: boolean;
  problem: string | null;
};

export type FiveSignalFlowSnapshot = {
  latestByMode: Partial<Record<FiveSignalMode, FiveSignalFlowModeState>>;
  availableModes: number;
  generatedItems: number;
  measuredItems: number;
  invalidModeCount: number;
  ignoredModeCount: number;
  counts: Pick<
    FiveSignalSummary,
    "pass" | "fail" | "review_required" | "not_measured" | "invalid"
  >;
  state: "pending" | "partial" | "complete" | "invalid";
};

export const METRIC_SOURCE_IDS = [
  "alignment",
  "sisdr",
  "stoi",
  "level",
  "spectrum",
  "doubleTalk",
  "echo",
] as const;

export type MetricSourceId = (typeof METRIC_SOURCE_IDS)[number];

export const METRIC_SOURCE_LABELS: Record<MetricSourceId, string> = {
  alignment: "对齐 / RTD",
  sisdr: "SI-SDR",
  stoi: "STOI",
  level: "电平 / SNR",
  spectrum: "频响 / 频谱",
  doubleTalk: "双讲",
  echo: "回声",
};

export type MetricMappingRow = {
  id: string;
  row: number;
  item: NV200TestItem;
  source: MetricSourceId | null;
  status: "engineering_evidence" | "alignment_proxy" | "not_measured";
};

export type MetricMappingSnapshot = {
  mode: string;
  bandwidth: "NB" | "WB" | "SWB" | null;
  valid: boolean;
  problem: string | null;
  rows: MetricMappingRow[];
  sourceCounts: Record<MetricSourceId, number>;
  mappedCount: number;
  unmappedCount: number;
};

const EMPTY_SOURCE_COUNTS = (): Record<MetricSourceId, number> => ({
  alignment: 0,
  sisdr: 0,
  stoi: 0,
  level: 0,
  spectrum: 0,
  doubleTalk: 0,
  echo: 0,
});

const NB_WB_ROW_SOURCES: Readonly<Record<number, MetricSourceId>> = {
  2: "alignment",
  12: "level",
  13: "spectrum",
  14: "stoi",
  15: "level",
  16: "spectrum",
  17: "spectrum",
  18: "spectrum",
  19: "doubleTalk",
  20: "doubleTalk",
  21: "echo",
  22: "echo",
  23: "level",
  27: "level",
  28: "spectrum",
  32: "level",
  33: "spectrum",
  37: "level",
  38: "spectrum",
  42: "level",
  43: "spectrum",
};

const SWB_ROW_SOURCES: Readonly<Record<number, MetricSourceId>> = {
  2: "alignment",
  11: "level",
  12: "spectrum",
  13: "stoi",
  14: "level",
  15: "spectrum",
  16: "level",
  17: "spectrum",
  18: "doubleTalk",
  19: "doubleTalk",
  20: "echo",
  21: "echo",
  22: "echo",
  23: "level",
  27: "level",
  28: "spectrum",
  32: "level",
  33: "spectrum",
  37: "level",
  38: "spectrum",
  42: "level",
  43: "spectrum",
};

function emptyMetricMapping(
  mode: string,
  bandwidth: MetricMappingSnapshot["bandwidth"],
  problem: string,
): MetricMappingSnapshot {
  return {
    mode,
    bandwidth,
    valid: false,
    problem,
    rows: [],
    sourceCounts: EMPTY_SOURCE_COUNTS(),
    mappedCount: 0,
    unmappedCount: 0,
  };
}

export function buildMetricMapping(
  sheet: NV200SheetData,
): MetricMappingSnapshot {
  const bandwidth = sheet.sheetId.split("-")[0];
  if (bandwidth !== "NB" && bandwidth !== "WB" && bandwidth !== "SWB") {
    return emptyMetricMapping(
      sheet.sheetId,
      null,
      `无法识别业务模式 ${sheet.sheetId} 的带宽`,
    );
  }

  const expectedRows = Array.from({ length: 42 }, (_, index) => index + 2);
  const actualRows = sheet.items.map((item) => item.row);
  const rowSet = new Set(actualRows);
  if (
    sheet.items.length !== 42 ||
    rowSet.size !== 42 ||
    !expectedRows.every((row) => rowSet.has(row))
  ) {
    return emptyMetricMapping(
      sheet.sheetId,
      bandwidth,
      `单模式应为 B2–B43 共 42 项，实际 ${sheet.items.length} 项`,
    );
  }

  const rowSources = bandwidth === "SWB" ? SWB_ROW_SOURCES : NB_WB_ROW_SOURCES;
  const sourceCounts = EMPTY_SOURCE_COUNTS();
  const rows = [...sheet.items]
    .sort((left, right) => left.row - right.row)
    .map((item): MetricMappingRow => {
      const source = rowSources[item.row] ?? null;
      if (source) sourceCounts[source] += 1;
      return {
        id: `${sheet.sheetId}:B${item.row}`,
        row: item.row,
        item,
        source,
        status:
          source === "alignment"
            ? "alignment_proxy"
            : source
              ? "engineering_evidence"
              : "not_measured",
      };
    });

  const mappedCount = rows.filter((row) => row.source !== null).length;
  return {
    mode: sheet.sheetId,
    bandwidth,
    valid: true,
    problem: null,
    rows,
    sourceCounts,
    mappedCount,
    unmappedCount: rows.length - mappedCount,
  };
}

export const DNSMOS_OUTPUT_IDS = ["sig", "bak", "ovrl"] as const;
export type DnsmosOutputId = (typeof DNSMOS_OUTPUT_IDS)[number];

export const DNSMOS_OUTPUT_LABELS: Record<DnsmosOutputId, string> = {
  sig: "SIG",
  bak: "BAK",
  ovrl: "OVRL",
};

export type DnsmosMappingRow = {
  id: string;
  row: number;
  item: NV200TestItem;
  output: DnsmosOutputId | null;
};

export type DnsmosMappingSnapshot = {
  mode: string;
  valid: boolean;
  problem: string | null;
  rows: DnsmosMappingRow[];
  outputCounts: Record<DnsmosOutputId, number>;
  mappedCount: number;
  unmappedCount: number;
};

const DNSMOS_ROW_OUTPUTS: Readonly<Record<number, DnsmosOutputId>> = {
  24: "ovrl",
  25: "bak",
  26: "sig",
  29: "ovrl",
  30: "bak",
  31: "sig",
  34: "ovrl",
  35: "bak",
  36: "sig",
  39: "ovrl",
  40: "bak",
  41: "sig",
};

const EMPTY_DNSMOS_COUNTS = (): Record<DnsmosOutputId, number> => ({
  sig: 0,
  bak: 0,
  ovrl: 0,
});

export function buildDnsmosMapping(
  sheet: NV200SheetData,
): DnsmosMappingSnapshot {
  const base = buildMetricMapping(sheet);
  if (!base.valid) {
    return {
      mode: sheet.sheetId,
      valid: false,
      problem: base.problem,
      rows: [],
      outputCounts: EMPTY_DNSMOS_COUNTS(),
      mappedCount: 0,
      unmappedCount: 0,
    };
  }

  const outputCounts = EMPTY_DNSMOS_COUNTS();
  const rows = base.rows.map((row): DnsmosMappingRow => {
    const output = DNSMOS_ROW_OUTPUTS[row.row] ?? null;
    if (output) outputCounts[output] += 1;
    return {
      id: `${sheet.sheetId}:dnsmos:B${row.row}`,
      row: row.row,
      item: row.item,
      output,
    };
  });
  const mappedCount = rows.filter((row) => row.output !== null).length;
  return {
    mode: sheet.sheetId,
    valid: true,
    problem: null,
    rows,
    outputCounts,
    mappedCount,
    unmappedCount: rows.length - mappedCount,
  };
}

export const RECEIVE_PROXY_IDS = [
  "binauralLevel",
  "frequencyResponse",
  "idleNoise",
  "noisePeaks",
  "outOfBand",
  "rlrLinearity",
  "binauralP835",
] as const;

export type ReceiveProxyId = (typeof RECEIVE_PROXY_IDS)[number];

export const RECEIVE_PROXY_LABELS: Record<ReceiveProxyId, string> = {
  binauralLevel: "双耳电平",
  frequencyResponse: "H1 接收频响",
  idleNoise: "噪声段 RMS",
  noisePeaks: "峰值余量",
  outOfBand: "带外能量比",
  rlrLinearity: "±5 dB 线性残差",
  binauralP835: "双耳 P.835 代理",
};

export const RECEIVE_PROXY_EVIDENCE: Record<ReceiveProxyId, string> = {
  binauralLevel: "双耳 / 左右耳 RMS、ILD、ITD",
  frequencyResponse: "已知激励到虚拟双耳的 H1 增益、逐带相干性与有效性；频带电平另作诊断",
  idleNoise: "噪声窗数字 RMS（必须提供指定窗长）",
  noisePeaks: "时域峰值余量 + 1/3 倍频程频带电平",
  outOfBand: "带外能量比（必须提供 upper_hz）",
  rlrLinearity:
    "标称 / −5 dB / +5 dB 三次运行的双耳频带电平与相对线性残差",
  binauralP835:
    "虚拟左耳 / 右耳 DNSMOS SIG、BAK、OVRL；OVRL 最小值作为保守 TMOS 代理",
};

export type ReceiveProxyMappingRow = {
  id: string;
  row: number;
  item: NV200TestItem;
  proxy: ReceiveProxyId | null;
  status: "engineering_proxy" | "not_measured";
  requirement: string;
  evidence: string | null;
  gap: string;
};

export type ReceiveProxyMappingSnapshot = {
  mode: string;
  bandwidth: "NB" | "WB" | "SWB" | null;
  valid: boolean;
  problem: string | null;
  rows: ReceiveProxyMappingRow[];
  proxyCounts: Record<ReceiveProxyId, number>;
  proxyCount: number;
  unmappedCount: number;
};

const EMPTY_RECEIVE_PROXY_COUNTS = (): Record<ReceiveProxyId, number> => ({
  binauralLevel: 0,
  frequencyResponse: 0,
  idleNoise: 0,
  noisePeaks: 0,
  outOfBand: 0,
  rlrLinearity: 0,
  binauralP835: 0,
});

type ReceiveMappingRule = {
  proxy: ReceiveProxyId | null;
  requirement: string;
  gap: string;
};

function receiveMappingRule(item: NV200TestItem): ReceiveMappingRule {
  const name = item.name.toLowerCase();

  if (name.includes("variation of rlr")) {
    return {
      proxy: "rlrLinearity",
      requirement:
        "同一夹具、音量与频响权重下的标称 / −5 dB / +5 dB 三档相对接收电平线性残差",
      gap:
        "工程代理必须由三次受控运行成组计算；未校准 dBFS 不能直接称为正式 P.79 RLR，正式合规仍需校准 DRP。",
    };
  }
  if (name.includes("loudness rating - receiving")) {
    return {
      proxy: "binauralLevel",
      requirement: "校准 dB[Pa]、P.79 接收响度 RLR",
      gap: "当前只提供未校准数字域电平，不能把 RMS 当作 RLR。",
    };
  }
  if (
    (name.includes("frequency response") || name.includes("freq. response")) &&
    (name.includes("rcv") || name.includes("receiving"))
  ) {
    return {
      proxy: "frequencyResponse",
      requirement: "归一化双耳频响与合同容差掩模距离",
      gap: "批次使用输入/输出 H1 频响并校验激励与相干性；仍缺声学校准和合同容差掩模判定。",
    };
  }
  if (
    name.includes("idle channel noise") &&
    (name.includes("spec. peaks") || name.includes("peaks"))
  ) {
    return {
      proxy: "noisePeaks",
      requirement: "指定 50 / 100 ms 窗内的频域谱峰容差",
      gap: "当前峰值余量是时域峰值减 RMS，不是合同频域谱峰距离。",
    };
  }
  if (name.includes("idle channel noise")) {
    return {
      proxy: "idleNoise",
      requirement: "A 计权、校准 dB[Pa]、指定窗长的接收底噪",
      gap: "当前噪声 RMS 是未校准 dBFS，且不提供 A 计权或 dB[Pa]。",
    };
  }
  if (
    name.includes("one way") ||
    name.includes("listening speech") ||
    name.includes("polqa") ||
    name.includes("tmos")
  ) {
    return {
      proxy: "binauralP835",
      requirement:
        "TMOS / POLQA 或经批准的等效 MOS；当前以双耳 DNSMOS P.835 做保守模型代理",
      gap:
        "DNSMOS P.835 不是 TMOS/POLQA，不能套用合同 MOS 下限自动判通过；正式值仍需获批实现与测试信号包。",
    };
  }
  if (name.includes("out-of-band") && (name.includes("rcv") || name.includes("receiving"))) {
    return {
      proxy: "outOfBand",
      requirement: "校准 Corrected value [dB[Pa]] 与正式带外门限",
      gap: "当前只计算数字域带外 / 总能量比，不是合同 Corrected value。",
    };
  }
  if (name.includes("loudness rating - sending")) {
    return {
      proxy: null,
      requirement: "发送响度 SLR",
      gap: "该行是发送方向指标，接收端虚拟 DRP 仿真不适用。",
    };
  }
  return {
    proxy: null,
    requirement: "正式接收测量",
    gap: "当前虚拟 DRP 报告没有与该合同行等价的字段。",
  };
}

export function buildReceiveProxyMapping(
  sheet: NV200SheetData,
): ReceiveProxyMappingSnapshot {
  const base = buildMetricMapping(sheet);
  if (!base.valid) {
    return {
      mode: sheet.sheetId,
      bandwidth: base.bandwidth,
      valid: false,
      problem: base.problem,
      rows: [],
      proxyCounts: EMPTY_RECEIVE_PROXY_COUNTS(),
      proxyCount: 0,
      unmappedCount: 0,
    };
  }

  const receiveRows = base.rows.filter(
    (row) => row.row >= 3 && row.row <= 11,
  );
  if (
    receiveRows.length !== 9 ||
    receiveRows.some((row, index) => row.row !== index + 3)
  ) {
    return {
      mode: sheet.sheetId,
      bandwidth: base.bandwidth,
      valid: false,
      problem: `B3–B11 应为连续 9 行，实际 ${receiveRows.length} 行`,
      rows: [],
      proxyCounts: EMPTY_RECEIVE_PROXY_COUNTS(),
      proxyCount: 0,
      unmappedCount: 0,
    };
  }

  const proxyCounts = EMPTY_RECEIVE_PROXY_COUNTS();
  const rows = receiveRows.map((row): ReceiveProxyMappingRow => {
    const rule = receiveMappingRule(row.item);
    if (rule.proxy) proxyCounts[rule.proxy] += 1;
    return {
      id: `${sheet.sheetId}:receive:B${row.row}`,
      row: row.row,
      item: row.item,
      proxy: rule.proxy,
      status: rule.proxy ? "engineering_proxy" : "not_measured",
      requirement: rule.requirement,
      evidence: rule.proxy ? RECEIVE_PROXY_EVIDENCE[rule.proxy] : null,
      gap: rule.gap,
    };
  });
  const proxyCount = rows.filter((row) => row.proxy !== null).length;

  return {
    mode: sheet.sheetId,
    bandwidth: base.bandwidth,
    valid: true,
    problem: null,
    rows,
    proxyCounts,
    proxyCount,
    unmappedCount: rows.length - proxyCount,
  };
}

const SUMMARY_FIELDS = [
  "total",
  "measured",
  "pass",
  "fail",
  "review_required",
  "not_measured",
  "invalid",
] as const;

const isMode = (value: unknown): value is FiveSignalMode =>
  typeof value === "string" &&
  (FIVE_SIGNAL_MODES as readonly string[]).includes(value);

const isNonNegativeInteger = (value: unknown): value is number =>
  typeof value === "number" && Number.isInteger(value) && value >= 0;

function parseSummary(value: unknown): FiveSignalSummary | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const candidate = value as Record<string, unknown>;
  if (!SUMMARY_FIELDS.every(field => isNonNegativeInteger(candidate[field]))) {
    return null;
  }
  const summary = Object.fromEntries(
    SUMMARY_FIELDS.map(field => [field, candidate[field]]),
  ) as FiveSignalSummary;
  const statusTotal =
    summary.pass +
    summary.fail +
    summary.review_required +
    summary.not_measured +
    summary.invalid;
  return statusTotal === summary.total &&
    summary.measured === summary.pass + summary.fail + summary.review_required
    ? summary : null;
}

export function deriveFiveSignalFlowSnapshot(
  rows: readonly unknown[],
): FiveSignalFlowSnapshot {
  const latestByMode: Partial<Record<FiveSignalMode, FiveSignalFlowModeState>> = {};
  let ignoredModeCount = 0;

  for (const row of rows) {
    if (!row || typeof row !== "object" || Array.isArray(row)) continue;
    const candidate = row as Record<string, unknown>;
    if (!isMode(candidate.mode)) {
      ignoredModeCount += 1;
      continue;
    }
    if (latestByMode[candidate.mode]) continue;

    const summary = parseSummary(candidate.summary);
    let problem: string | null = null;
    if (typeof candidate.evaluation_id !== "string" || !candidate.evaluation_id.trim()) {
      problem = "缺少报告 ID，无法关联证据";
    } else if (!summary) {
      problem = "汇总字段缺失或自相矛盾";
    } else if (summary.total !== 42) {
      problem = `单模式应为 42 项，实际 ${summary.total} 项`;
    }

    latestByMode[candidate.mode] = {
      mode: candidate.mode,
      evaluation_id:
        typeof candidate.evaluation_id === "string"
          ? candidate.evaluation_id
          : "",
      summary,
      valid: problem === null,
      problem,
    };
  }

  const selected = Object.values(latestByMode);
  const valid = selected.filter(
    (item): item is FiveSignalFlowModeState & { summary: FiveSignalSummary } =>
      item.valid && item.summary !== null,
  );
  const counts = {
    pass: 0,
    fail: 0,
    review_required: 0,
    not_measured: 0,
    invalid: 0,
  };
  for (const item of valid) {
    counts.pass += item.summary.pass;
    counts.fail += item.summary.fail;
    counts.review_required += item.summary.review_required;
    counts.not_measured += item.summary.not_measured;
    counts.invalid += item.summary.invalid;
  }

  const availableModes = valid.length;
  const generatedItems = valid.reduce(
    (total, item) => total + item.summary.total,
    0,
  );
  const invalidModeCount =
    selected.filter(item => !item.valid).length + ignoredModeCount;
  const complete =
    availableModes === FIVE_SIGNAL_MODES.length &&
    generatedItems === 336 &&
    invalidModeCount === 0;

  return {
    latestByMode,
    availableModes,
    generatedItems,
    measuredItems: valid.reduce((total, item) => total + item.summary.measured, 0),
    invalidModeCount,
    ignoredModeCount,
    counts,
    state: complete
      ? "complete"
      : invalidModeCount > 0
        ? "invalid"
        : availableModes > 0
          ? "partial"
          : "pending",
  };
}

// These are display groups, not a replacement for the contract's direction field.
export function contractGroupsForMode(mode: FiveSignalMode): FiveSignalContractGroup[] {
  const swb = mode.startsWith("SWB-");
  return FIVE_SIGNAL_CONTRACT_GROUPS.map(group => ({
    ...group,
    count: group.label === "下行接收质量" ? (swb ? 8 : 9)
      : group.label === "回声耦合" ? (swb ? 3 : 2) : group.count,
    detail: group.label === "回声耦合" && swb
      ? "SWB 合同的三项 TCLw 条目；音量条件须独立核验" : group.detail,
  }));
}

export const FIVE_SIGNAL_ROLES = ["near_end", "noise", "microphone", "far_end", "reference"] as const;
export const FIVE_SIGNAL_ROLE_LABELS = {
  near_end: ACOUSTIC_SIGNAL_LABELS.near_end,
  noise: ACOUSTIC_SIGNAL_LABELS.noise,
  microphone: ACOUSTIC_SIGNAL_LABELS.microphone,
  far_end: ACOUSTIC_SIGNAL_LABELS.far_end,
  reference: ACOUSTIC_SIGNAL_LABELS.reference,
} as const;
export type FiveSignalInput = { name: string; sha256: string; sample_rate: number; duration_s: number };
type Alignment = { delay_ms: number | null; correlation: number | null; available: boolean; reliable: boolean };
export type FiveSignalDetail = {
  evaluation_id: string;
  mode: FiveSignalMode;
  created_at: string;
  execution?: {
    source: "alignment_cli";
    algorithm_cluster_id: string;
    algorithm_cluster_label: string;
    output_sha256: string;
    cli_sha256: string;
    catalog_sha256: string;
    dataset_id: string;
    configuration: { profile: string; selection: Record<string, string>; extra_flags?: string[] };
    rx_processing: boolean;
  };
  inputs: Record<typeof FIVE_SIGNAL_ROLES[number], FiveSignalInput>;
  analysis: {
    values: Record<string, number | null>;
    segments: Record<string, number>;
    alignments: Record<"microphone" | "near_end", Alignment>;
  };
};
const record = (value: unknown): Record<string, unknown> | null =>
  value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown> : null;
export const finiteValue = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);

export function parseFiveSignalDetail(value: unknown, expectedId: string, expectedMode: FiveSignalMode, expectedCluster?: string): FiveSignalDetail {
  const report = record(value);
  if (!report || report.evaluation_id !== expectedId || report.mode !== expectedMode) {
    throw Error("报告身份与所选模式不一致");
  }
  const summary = parseSummary(report.summary);
  if (!summary || summary.total !== 42) throw Error("报告详情的 42 项汇总无效");
  const inputs = record(report.inputs);
  for (const role of FIVE_SIGNAL_ROLES) {
    const input = record(inputs?.[role]);
    if (!input || typeof input.name !== "string" || !input.name ||
      typeof input.sha256 !== "string" || !/^[a-f0-9]{64}$/i.test(input.sha256) ||
      !finiteValue(input.sample_rate) || input.sample_rate <= 0 ||
      !finiteValue(input.duration_s) || input.duration_s <= 0) {
      throw Error(`输入元数据无效：${FIVE_SIGNAL_ROLE_LABELS[role]}`);
    }
  }
  if (expectedCluster !== undefined) {
    const execution = record(report.execution);
    const output = record(inputs?.near_end);
    if (!execution || execution.source !== "alignment_cli" ||
      execution.algorithm_cluster_id !== expectedCluster || execution.output_sha256 !== output?.sha256) {
      throw Error("报告算法簇或实际输出身份与当前选择不一致");
    }
  }
  const analysis = record(report.analysis);
  const values = record(analysis?.values);
  const segments = record(analysis?.segments);
  const alignments = record(analysis?.alignments);
  if (!values || !segments || !alignments ||
    Object.values(values).some(v => v !== null && !finiteValue(v))) {
    throw Error("指标详情缺失或包含非有限数");
  }
  for (const key of ["near_only_seconds", "far_only_seconds", "double_talk_seconds", "noise_only_seconds"]) {
    if (!finiteValue(segments[key]) || segments[key] < 0) throw Error("活动区段元数据无效");
  }
  for (const role of ["microphone", "near_end"]) {
    const alignment = record(alignments[role]);
    if (!alignment || typeof alignment.available !== "boolean" || typeof alignment.reliable !== "boolean" ||
      [alignment.delay_ms, alignment.correlation].some(v => v !== null && !finiteValue(v))) {
      throw Error("对齐元数据无效");
    }
  }
  return report as unknown as FiveSignalDetail;
}

export const METRIC_VALUE_KEYS = [
  ["microphone_sisdr_db", "output_sisdr_db", "sisdr_delta_db"],
  ["microphone_stoi", "output_stoi", "stoi_delta"],
  ["send_gain_db", "microphone_noise_floor_dbfs", "output_noise_floor_dbfs", "input_snr_db", "output_snr_db"],
  ["frequency_response_margin_db", "frequency_flatness_db", "output_noise_peak_prominence_db", "noise_spectrum_change_db"],
  ["echo_loss_db", "echo_reduction_db"],
  ["double_talk_transparent_percent"],
] as const;
export type AnalysisAvailability = { state: "pending" | "partial" | "complete" | "invalid"; label: string };
export function analysisAvailability(detail: FiveSignalDetail | null, anomalous = false): AnalysisAvailability {
  if (anomalous) return { state: "invalid", label: "数据异常" };
  if (!detail) return { state: "pending", label: "等待报告详情" };
  const keys = METRIC_VALUE_KEYS.flat();
  const count = keys.filter(key => finiteValue(detail.analysis.values[key])).length;
  if (!count) return { state: "pending", label: "无可用指标值" };
  const aligned = Object.values(detail.analysis.alignments).every(a => a.available && a.reliable && finiteValue(a.delay_ms) && finiteValue(a.correlation));
  if (!aligned) return { state: "partial", label: "对齐需复核" };
  if (count !== keys.length || ["near_only_seconds", "far_only_seconds", "double_talk_seconds", "noise_only_seconds"].some(key => detail.analysis.segments[key] <= 0)) {
    return { state: "partial", label: "指标部分可用" };
  }
  return { state: "complete", label: "指标可用（非通过）" };
}
