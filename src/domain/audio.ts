export const AUDIO_ROLES = ["reference", "before", "after", "echo"] as const;
export type AudioRole = (typeof AUDIO_ROLES)[number];
export const ROLE_LABEL: Record<AudioRole, string> = {
  reference: "干净参考",
  before: "麦克风处理前",
  after: "麦克风处理后",
  echo: "扬声器回声参考",
};
export const ROLE_COLOR: Record<AudioRole, string> = {
  reference: "#64748b",
  before: "#c48c38",
  after: "#2563eb",
  echo: "#8b5cf6",
};
export type AudioFiles = Record<AudioRole, File | null>;
export type MetricStatus = "ok" | "warning" | "unavailable";
export interface AudioMetric {
  id: string;
  name: string;
  value: number | null;
  unit: string;
  status: MetricStatus;
  note: string;
  role: AudioRole | null;
  method: string;
}
export interface AudioDescription {
  role: AudioRole;
  name: string;
  sha256: string;
  bytes: number;
  sample_rate: number;
  channels: number;
  selected_channel: number;
  duration: number;
  format: string;
  rms_dbfs: number;
  peak_dbfs: number;
  dc_offset: number;
  clipping_percent: number;
  silent: boolean;
  waveform: { min: number[]; max: number[] };
  spectrum: { frequency: number[]; dbfs_hz: number[]; peak_hz: number | null };
}
export interface AudioOptions {
  sample_rate: number;
  channels: Record<AudioRole, number>;
  speech: AudioWindow;
  noise: AudioWindow | null;
  far_end: AudioWindow | null;
  max_delay_ms: number;
  after_offset_ms: number;
  common_clock: boolean;
  far_end_only: boolean;
  aec_only: boolean;
  stationary_noise: boolean;
  speech_confirmed: boolean;
}
export interface AudioWindow {
  start: number;
  end: number | null;
}
export interface AudioPair {
  role: "before" | "after";
  alignment: {
    delay_ms: number;
    correlation: number;
    overlap_seconds: number;
    reliable: boolean;
    note: string;
    search_ms: number;
  };
  metrics: AudioMetric[];
  response: {
    frequency: number[];
    gain_db: (number | null)[];
    coherence: number[];
  } | null;
}
export interface AudioReport {
  schema_version: 2;
  engine: string;
  created_at: string;
  files: AudioDescription[];
  options: AudioOptions;
  metrics: AudioMetric[];
  pairs: AudioPair[];
  notices: string[];
  standards: {
    symbol: string;
    name: string;
    status: string;
    requirement: string;
  }[];
}
export const EMPTY_FILES: AudioFiles = {
  reference: null,
  before: null,
  after: null,
  echo: null,
};
export const REPORT_KEY = "itu-audio-lab:audio-report:v2";
export function metricOf(report: AudioReport, id: string) {
  return report.metrics.find((m) => m.id === id);
}
export function displayValue(metric?: AudioMetric, digits = 2) {
  if (!metric || metric.value === null) return "—";
  return metric.value.toLocaleString("zh-CN", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}
export function audioCsv(report: AudioReport): string {
  const rows: unknown[][] = [
    [
      "指标",
      "计算值",
      "单位",
      "状态",
      "说明",
      "方法",
      "文件SHA256",
      "计算配置",
    ],
  ];
  const hashes = report.files.map((f) => `${f.role}:${f.sha256}`).join("; ");
  const options = JSON.stringify(report.options);
  for (const m of report.metrics)
    rows.push([
      m.name,
      m.value,
      m.unit,
      m.status,
      m.note,
      m.method,
      hashes,
      options,
    ]);
  return (
    "\uFEFF" +
    rows
      .map((row) =>
        row
          .map((v) => {
            let s = v === null || v === undefined ? "" : String(v);
            if (typeof v === "string" && /^[=+\-@\t\r\n]/.test(s)) s = "'" + s;
            return '"' + s.replaceAll('"', '""') + '"';
          })
          .join(","),
      )
      .join("\r\n")
  );
}
export function loadAudioReport(): AudioReport | null {
  try {
    const saved = JSON.parse(localStorage.getItem(REPORT_KEY) || "null");
    return saved?.schema_version === 2 &&
      Array.isArray(saved.files) &&
      Array.isArray(saved.metrics) &&
      Array.isArray(saved.pairs) &&
      saved.options
      ? saved
      : null;
  } catch {
    return null;
  }
}
