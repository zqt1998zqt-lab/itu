import { classifyMetric, METRIC_MAP } from "./metrics.ts";
import type { MetricId, ImprovementDirection } from "./metrics.ts";

export const MODES = [
  "NB-wired",
  "WB-wired",
  "SWB-wired",
  "NB-wireless",
  "WB-wireless",
  "SWB-wireless",
  "NB-bt",
  "WB-bt",
] as const;
export const BATCHES = ["G", "H", "I"] as const;
export type Batch = (typeof BATCHES)[number];
export type Mode = (typeof MODES)[number];
export type Status = "pass" | "fail" | "pending" | "missing" | "invalid";
export const STATUS_LABEL: Record<Status, string> = {
  pass: "通过",
  fail: "未通过",
  pending: "待确认",
  missing: "未测",
  invalid: "数据异常",
};
export const BATCH_LABEL: Record<Batch, string> = {
  G: "首次测量 · G 列",
  H: "复测记录 · H 列",
  I: "后续测量 · I 列",
};
export interface Measurement {
  value: number | null;
  date: string;
  invalid: boolean;
  raw?: string;
  edited?: boolean;
}
export interface TestItem {
  id: string;
  mode: Mode;
  sourceRow: number;
  channel: string;
  metricId: MetricId;
  name: string;
  sourceUnit: string;
  unit: string;
  lower: number | null;
  upper: number | null;
  rule: "range" | "curve" | "unconfirmed";
  limitText: string;
  flags: string[];
  context: string;
  direction: ImprovementDirection;
  target: number | null;
  measurements: Record<Batch, Measurement>;
  comparable: boolean;
}
export interface ProjectData {
  schema: 1;
  sourceName: string;
  items: TestItem[];
  savedAt?: string;
}
export interface SourceSheet {
  sheet: string;
  rows: { row: string; values: Record<string, unknown> }[];
}
export interface Result {
  status: Status;
  value: number | null;
  margin: number | null;
  excess: number | null;
}
export interface Stats {
  total: number;
  measured: number;
  pass: number;
  fail: number;
  pending: number;
  missing: number;
  invalid: number;
  coverage: number;
  passRate: number | null;
}

export function numeric(value: unknown): number | null {
  if (
    value === null ||
    value === undefined ||
    (typeof value === "string" && value.trim() === "")
  )
    return null;
  if (typeof value !== "number" && typeof value !== "string") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}
export function dateLabel(value: unknown): string {
  if (value instanceof Date) return value.toISOString().slice(0, 10);
  const serial = numeric(value);
  if (serial !== null && serial > 20000 && serial < 100000)
    return new Date(Date.UTC(1899, 11, 30) + serial * 86400000)
      .toISOString()
      .slice(0, 10);
  return typeof value === "string" ? value.slice(0, 30) : "";
}
function measurement(value: unknown, date: unknown): Measurement {
  const n = numeric(value);
  const blank =
    value === null ||
    value === undefined ||
    (typeof value === "string" && value.trim() === "");
  return {
    value: n,
    date: dateLabel(date),
    invalid: !blank && n === null,
    ...(!blank && n === null ? { raw: String(value).slice(0, 120) } : {}),
  };
}
function displayUnit(id: MetricId, source: string): string {
  if (id === "M01") return "ms";
  if (["M02", "M03", "M04", "M11", "M12", "M17"].includes(id)) return "dB";
  if (["M08", "M13", "M14", "M15"].includes(id)) return "MOS";
  if (id === "M10") return "%";
  if (source.includes("dBm0(A)")) return "dBm0(A)";
  if (source.includes("dBm0")) return "dBm0";
  if (source.includes("dB[Pa]")) return "dB[Pa]";
  if (source.includes("[Pa/V]")) return "Pa/V";
  if (source.includes("[dB]")) return "dB";
  if (source.includes("[Pa]")) return "Pa";
  if (source.includes("[V]")) return "V";
  return "仪器值";
}

export function fromSource(
  sheets: SourceSheet[],
  sourceName = "NV200 ITUT测试表.xlsx",
): ProjectData {
  const items: TestItem[] = [];
  for (const sheet of sheets) {
    if (!MODES.includes(sheet.sheet as Mode)) continue;
    const header = sheet.rows.find((r) => r.row === "1")?.values ?? {};
    let channel = "";
    for (const row of sheet.rows) {
      const r = Number(row.row),
        v = row.values;
      if (r < 2 || !v[`B${r}`]) continue;
      const name = String(v[`B${r}`]);
      const metricId = classifyMetric(name);
      if (v[`A${r}`])
        channel = String(v[`A${r}`]) === "1" ? "下行" : String(v[`A${r}`]);
      const lower = numeric(v[`E${r}`]),
        upper = numeric(v[`D${r}`]);
      const rule =
        lower !== null && upper !== null
          ? "range"
          : String(v[`D${r}`] ?? "").includes("符合")
            ? "curve"
            : "unconfirmed";
      const flags = ["Optional", "Preliminary", "Recommended"].filter((flag) =>
        name.toLowerCase().includes(flag.toLowerCase()),
      );
      if (name.startsWith("U -")) flags.push("U-定义待确认");
      const scenario = name.match(/Sce\d/)?.[0];
      const volume = /maximum|max(?:imum)?[ .]/i.test(name)
        ? "最大音量"
        : /nominal|nom\.|nom /i.test(name)
          ? "标称音量"
          : "";
      const window = name.match(/(?:50|100)\s?ms/i)?.[0];
      const target =
        metricId === "M02"
          ? 13
          : ["M04", "M16"].includes(metricId)
            ? 0
            : metricId === "M03" && /nominal/i.test(name)
              ? 2
              : null;
      const direction =
        metricId === "M03" && target === null
          ? "unconfirmed"
          : METRIC_MAP[metricId].direction;
      items.push({
        id: `${sheet.sheet}:${r}`,
        mode: sheet.sheet as Mode,
        sourceRow: r,
        channel: metricId === "M01" ? "时延" : channel,
        metricId,
        name,
        sourceUnit: String(v[`C${r}`] ?? ""),
        unit: displayUnit(metricId, String(v[`C${r}`] ?? "")),
        lower,
        upper,
        rule,
        limitText: String(v[`D${r}`] ?? "缺少限值"),
        flags,
        context: [scenario, volume, window].filter(Boolean).join(" · "),
        direction,
        target,
        measurements: Object.fromEntries(
          BATCHES.map((b) => [b, measurement(v[`${b}${r}`], header[`${b}1`])]),
        ) as Record<Batch, Measurement>,
        comparable: false,
      });
    }
  }
  if (!items.length)
    throw new Error(
      "没有找到 NV200 测试项目。请使用含 NB-wired、WB-wired 等工作表的原始测试表。",
    );
  return { schema: 1, sourceName, items };
}

export function resultOf(item: TestItem, batch: Batch): Result {
  const m = item.measurements[batch],
    x = m.value;
  const physicallyInvalid =
    x !== null &&
    ((["M08", "M13", "M14", "M15"].includes(item.metricId) &&
      (x < 1 || x > 5)) ||
      (item.metricId === "M10" && (x < 0 || x > 100)) ||
      (item.metricId === "M01" && x < 0));
  if (m.invalid || physicallyInvalid || (x !== null && !Number.isFinite(x)))
    return { status: "invalid", value: x, margin: null, excess: null };
  if (x === null)
    return { status: "missing", value: null, margin: null, excess: null };
  if (
    item.rule !== "range" ||
    item.lower === null ||
    item.upper === null ||
    item.lower > item.upper
  )
    return { status: "pending", value: x, margin: null, excess: null };
  const margin = Math.min(x - item.lower, item.upper - x);
  return {
    status: margin >= 0 ? "pass" : "fail",
    value: x,
    margin,
    excess: Math.max(-margin, 0),
  };
}
export function statsOf(items: TestItem[], batch: Batch): Stats {
  const counts = { pass: 0, fail: 0, pending: 0, missing: 0, invalid: 0 };
  for (const item of items) counts[resultOf(item, batch).status]++;
  const measured = counts.pass + counts.fail + counts.pending;
  return {
    total: items.length,
    measured,
    ...counts,
    coverage: items.length ? measured / items.length : 0,
    passRate:
      counts.pass + counts.fail
        ? counts.pass / (counts.pass + counts.fail)
        : null,
  };
}
export function recordDelta(item: TestItem): number | null {
  const a = resultOf(item, "G"),
    b = resultOf(item, "H");
  return a.value !== null &&
    b.value !== null &&
    a.status !== "invalid" &&
    b.status !== "invalid"
    ? b.value - a.value
    : null;
}
export function engineeringDelta(item: TestItem): number | null {
  const delta = recordDelta(item);
  if (!item.comparable || delta === null) return null;
  if (item.direction === "higher") return delta;
  if (item.direction === "lower") return -delta;
  if (item.direction === "target" && item.target !== null)
    return (
      Math.abs(item.measurements.G.value! - item.target) -
      Math.abs(item.measurements.H.value! - item.target)
    );
  return null;
}
export function formatNumber(value: number | null, digits = 2): string {
  return value === null
    ? "—"
    : new Intl.NumberFormat("zh-CN", { maximumFractionDigits: digits }).format(
        value,
      );
}
export function percent(value: number | null): string {
  return value === null ? "—" : `${formatNumber(value * 100, 1)}%`;
}
export function signed(value: number | null): string {
  return value === null ? "—" : `${value > 0 ? "+" : ""}${formatNumber(value)}`;
}
export function threshold(item: TestItem): string {
  return item.rule === "range" && item.lower !== null && item.upper !== null
    ? `${formatNumber(item.lower)} ～ ${formatNumber(item.upper)}`
    : item.rule === "curve"
      ? "按曲线掩模"
      : "限值待确认";
}
export function pairedItems(items: TestItem[]): TestItem[] {
  return items.filter((item) => recordDelta(item) !== null);
}
export function tradeoffs(
  items: TestItem[],
): { mode: Mode; scenario: string; noise: number; speech: number }[] {
  const pairs = pairedItems(items),
    rows: { mode: Mode; scenario: string; noise: number; speech: number }[] =
      [];
  for (const noise of pairs.filter(
    (item) => item.metricId === "M14" && recordDelta(item)! > 0,
  )) {
    const scenario = noise.name.match(/Sce\d/)?.[0];
    const speech = pairs.find(
      (item) =>
        item.mode === noise.mode &&
        item.metricId === "M15" &&
        item.name.includes(scenario ?? "no-scenario"),
    );
    if (speech && recordDelta(speech)! < 0)
      rows.push({
        mode: noise.mode,
        scenario: scenario!,
        noise: recordDelta(noise)!,
        speech: recordDelta(speech)!,
      });
  }
  return rows.sort((a, b) => a.speech - b.speech);
}

export function validateProject(value: unknown): ProjectData {
  const data = value as ProjectData;
  if (
    !data ||
    data.schema !== 1 ||
    typeof data.sourceName !== "string" ||
    !Array.isArray(data.items) ||
    !data.items.length ||
    data.items.length > 5000
  )
    throw new Error("备份格式不正确，请选择本网站导出的 JSON 备份。");
  const ids = new Set<string>();
  for (const item of data.items) {
    if (
      !item ||
      !MODES.includes(item.mode) ||
      typeof item.name !== "string" ||
      typeof item.unit !== "string" ||
      typeof item.sourceUnit !== "string" ||
      !METRIC_MAP[item.metricId] ||
      !Number.isInteger(item.sourceRow) ||
      item.sourceRow < 2 ||
      item.id !== `${item.mode}:${item.sourceRow}` ||
      ids.has(item.id) ||
      !["range", "curve", "unconfirmed"].includes(item.rule) ||
      !Array.isArray(item.flags) ||
      typeof item.context !== "string" ||
      typeof item.channel !== "string"
    )
      throw new Error("备份中包含无效或重复测试项目。");
    if (
      classifyMetric(item.name) !== item.metricId ||
      !["higher", "lower", "target", "unconfirmed"].includes(item.direction) ||
      typeof item.comparable !== "boolean"
    )
      throw new Error("备份中的指标定义不一致。");
    for (const n of [item.lower, item.upper, item.target])
      if (n !== null && (typeof n !== "number" || !Number.isFinite(n)))
        throw new Error("备份的限值或目标不是有效数值。");
    for (const b of BATCHES) {
      const m = item.measurements?.[b];
      if (
        !m ||
        typeof m.invalid !== "boolean" ||
        typeof m.date !== "string" ||
        (m.value !== null &&
          (typeof m.value !== "number" || !Number.isFinite(m.value)))
      )
        throw new Error("备份中包含无效的测量数据。");
    }
    ids.add(item.id);
  }
  return data;
}

export function resultsCsv(items: TestItem[], batch: Batch): string {
  const headers = [
    "模式",
    "原表行",
    "指标",
    "方向",
    "场景/条件",
    "批次",
    "日期",
    "测量值",
    "单位",
    "下限",
    "上限",
    "表限判定",
    "带符号余量",
    "超限量",
    "原表项目",
  ];
  const rows = items.map((item) => {
    const r = resultOf(item, batch);
    return [
      item.mode,
      item.sourceRow,
      METRIC_MAP[item.metricId].name,
      item.channel,
      item.context,
      batch,
      item.measurements[batch].date,
      r.value,
      item.unit,
      item.lower,
      item.upper,
      STATUS_LABEL[r.status],
      r.margin,
      r.excess,
      item.name,
    ];
  });
  const escape = (v: unknown) => {
    let s = v === null || v === undefined ? "" : String(v);
    if (typeof v === "string" && /^[=+\-@\t\r]/.test(s)) s = `'${s}`;
    return `"${s.replaceAll('"', '""')}"`;
  };
  return (
    "\ufeff" +
    [headers, ...rows].map((row) => row.map(escape).join(",")).join("\r\n")
  );
}
