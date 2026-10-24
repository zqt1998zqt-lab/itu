export type GraphicMetricStatus = "ok" | "warning" | "unavailable";

export type GraphicMetric = {
  id: string;
  label: string;
  value: number | null;
  unit: string;
  status: GraphicMetricStatus;
  detail: string;
};

export type GraphicBand = {
  id: string;
  label: string;
  before_db: number;
  after_db: number;
  delta_db: number;
};

export type GraphicFigure = {
  id: string;
  title: string;
  category: "dynamics" | "time-frequency" | "comparison" | "bands" | "transient";
  description: string;
  detail: string;
  url: string;
  download_name: string;
};

export type GraphicSource = {
  role: "before" | "after";
  name: string;
  sha256: string;
  original_sample_rate: number;
  original_channels: number;
  selected_channel: number;
  format: string;
};

export type GraphicAlignment = {
  requested: boolean;
  applied: boolean;
  delay_ms: number | null;
  correlation: number | null;
  reliable: boolean | null;
  note: string;
  overlap_seconds?: number;
  search_ms?: number | null;
};

export type GraphicEvaluationCase = {
  asset_id: string;
  case_id: string;
  detail: string;
  window_seconds: [number, number];
  delay_samples: number;
  before_sha256: string;
  after_sha256: string;
  manifest: string;
};

export type AudioGraphicFocus = "all" | "ns" | "aec" | "bf" | "drc" | "eq" | "agc";

export const AUDIO_GRAPH_FOCUS_FIGURES: Record<AudioGraphicFocus, readonly string[]> = {
  all: [
    "waveform-energy",
    "log-stft-before",
    "log-stft-after",
    "spectral-delta",
    "triple",
    "ltas",
    "erb",
    "high-time",
  ],
  ns: ["waveform-energy", "log-stft-before", "log-stft-after", "spectral-delta"],
  aec: ["waveform-energy", "log-stft-after", "spectral-delta", "high-time"],
  bf: ["waveform-energy", "log-stft-before", "log-stft-after", "spectral-delta"],
  drc: ["waveform-energy", "high-time"],
  eq: ["log-stft-before", "log-stft-after", "spectral-delta", "ltas"],
  agc: ["waveform-energy", "spectral-delta", "ltas"],
};

export const AUDIO_GRAPH_FOCUS_LABELS: Record<AudioGraphicFocus, string> = {
  all: "全部图形",
  ns: "降噪关键图",
  aec: "回声消除关键图",
  bf: "波束形成关键图",
  drc: "动态控制关键图",
  eq: "频响均衡关键图",
  agc: "增益控制关键图",
};

export type AudioGraphicReport = {
  schema_version: 1;
  id: string;
  created_at: string;
  scope: string;
  standard_compliance: false;
  analysis_rate: number;
  duration_seconds: number;
  source: GraphicSource[];
  alignment: GraphicAlignment;
  metrics: GraphicMetric[];
  bands: GraphicBand[];
  notices: string[];
  evaluation_case?: GraphicEvaluationCase;
  figure_scope: {
    stft_fft_size: number;
    stft_hop: number;
    time_offset_seconds: number;
    log_frequency_edge_min_hz: number;
    log_frequency_edge_max_hz: number;
    high_time_fft_size: number;
    high_time_hop: number;
    fixed_delay_samples?: number;
  };
  figures: GraphicFigure[];
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);
const isNonEmptyString = (value: unknown): value is string =>
  typeof value === "string" && value.length > 0;
const isFiniteNumber = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value);
const isFiniteNumberOrNull = (value: unknown): value is number | null =>
  value === null || isFiniteNumber(value);
const isBooleanOrNull = (value: unknown): value is boolean | null =>
  value === null || typeof value === "boolean";

export function parseAudioGraphicReport(value: unknown): AudioGraphicReport {
  const validCategories = new Set(["dynamics", "time-frequency", "comparison", "bands", "transient"]);
  const validRates = new Set([8000, 16000, 24000, 32000, 48000]);
  const validStatuses = new Set(["ok", "warning", "unavailable"]);
  if (!isRecord(value)) {
    throw new Error("图形引擎返回的报告不完整或图形地址无效，请重新分析。");
  }
  const report = value;
  const source = Array.isArray(report.source) ? report.source : null;
  const metrics = Array.isArray(report.metrics) ? report.metrics : null;
  const bands = Array.isArray(report.bands) ? report.bands : null;
  const notices = Array.isArray(report.notices) ? report.notices : null;
  const figures = Array.isArray(report.figures) ? report.figures : null;
  const alignment = isRecord(report.alignment) ? report.alignment : null;
  const figureScope = isRecord(report.figure_scope) ? report.figure_scope : null;
  const evaluationCase = isRecord(report.evaluation_case) ? report.evaluation_case : null;
  const sourceRoles = source?.map((item) => isRecord(item) ? item.role : undefined) ?? [];
  const figureIds = figures?.map((item) => isRecord(item) ? item.id : undefined) ?? [];
  const metricIds = metrics?.map((item) => isRecord(item) ? item.id : undefined) ?? [];
  const bandIds = bands?.map((item) => isRecord(item) ? item.id : undefined) ?? [];
  const beforeCount = sourceRoles.filter((role) => role === "before").length;
  const afterCount = sourceRoles.filter((role) => role === "after").length;
  const validSource = source !== null && source.length >= 1 && source.length <= 2 &&
    afterCount === 1 && beforeCount + afterCount === source.length &&
    new Set(sourceRoles).size === sourceRoles.length &&
    source.every((item) => isRecord(item) &&
      ["before", "after"].includes(String(item.role)) &&
      isNonEmptyString(item.name) &&
      typeof item.sha256 === "string" && /^[a-f0-9]{64}$/.test(item.sha256) &&
      isFiniteNumber(item.original_sample_rate) && item.original_sample_rate >= 4000 && item.original_sample_rate <= 192000 &&
      Number.isInteger(item.original_channels) && Number(item.original_channels) >= 1 && Number(item.original_channels) <= 8 &&
      Number.isInteger(item.selected_channel) && Number(item.selected_channel) >= 1 && Number(item.selected_channel) <= Number(item.original_channels) &&
      isNonEmptyString(item.format));
  const validAlignment = alignment !== null &&
    typeof alignment.requested === "boolean" &&
    typeof alignment.applied === "boolean" &&
    isFiniteNumberOrNull(alignment.delay_ms) &&
    (alignment.correlation === null ||
      (isFiniteNumber(alignment.correlation) && alignment.correlation >= 0 && alignment.correlation <= 1)) &&
    isBooleanOrNull(alignment.reliable) &&
    typeof alignment.note === "string" &&
    (alignment.overlap_seconds === undefined || isFiniteNumber(alignment.overlap_seconds)) &&
    (alignment.search_ms === undefined || isFiniteNumberOrNull(alignment.search_ms));
  const validMetrics = metrics !== null && metrics.length > 0 &&
    new Set(metricIds).size === metricIds.length &&
    metrics.every((item) => isRecord(item) &&
      isNonEmptyString(item.id) && isNonEmptyString(item.label) &&
      isFiniteNumberOrNull(item.value) && typeof item.unit === "string" &&
      typeof item.status === "string" && validStatuses.has(item.status) &&
      typeof item.detail === "string");
  const validBands = bands !== null &&
    new Set(bandIds).size === bandIds.length &&
    bands.every((item) => isRecord(item) &&
      isNonEmptyString(item.id) && isNonEmptyString(item.label) &&
      isFiniteNumber(item.before_db) && isFiniteNumber(item.after_db) && isFiniteNumber(item.delta_db));
  const validFigures = figures !== null && figures.length >= 5 && figures.length <= 8 &&
    new Set(figureIds).size === figureIds.length &&
    figures.every((item) => isRecord(item) &&
      isNonEmptyString(item.id) && /^[a-z0-9-]+$/.test(item.id) &&
      isNonEmptyString(item.title) && typeof item.category === "string" && validCategories.has(item.category) &&
      isNonEmptyString(item.description) && isNonEmptyString(item.detail) &&
      typeof item.url === "string" &&
      item.url === `/api/audio-graphics/${report.id}/figures/${item.id}.png` &&
      isNonEmptyString(item.download_name) && item.download_name.endsWith(".png"));
  const validFigureScope = figureScope !== null &&
    Number.isInteger(figureScope.stft_fft_size) && Number(figureScope.stft_fft_size) > 0 &&
    Number.isInteger(figureScope.stft_hop) && Number(figureScope.stft_hop) > 0 &&
    isFiniteNumber(figureScope.time_offset_seconds) && figureScope.time_offset_seconds >= 0 &&
    isFiniteNumber(figureScope.log_frequency_edge_min_hz) && figureScope.log_frequency_edge_min_hz > 0 &&
    isFiniteNumber(figureScope.log_frequency_edge_max_hz) &&
    figureScope.log_frequency_edge_max_hz > figureScope.log_frequency_edge_min_hz &&
    Number.isInteger(figureScope.high_time_fft_size) && Number(figureScope.high_time_fft_size) > 0 &&
    Number.isInteger(figureScope.high_time_hop) && Number(figureScope.high_time_hop) > 0 &&
    (figureScope.fixed_delay_samples === undefined ||
      (Number.isInteger(figureScope.fixed_delay_samples) && Number(figureScope.fixed_delay_samples) >= 0));
  const validEvaluationCase = report.evaluation_case === undefined ||
    (evaluationCase !== null &&
      isNonEmptyString(evaluationCase.asset_id) &&
      isNonEmptyString(evaluationCase.case_id) &&
      isNonEmptyString(evaluationCase.detail) &&
      Array.isArray(evaluationCase.window_seconds) && evaluationCase.window_seconds.length === 2 &&
      isFiniteNumber(evaluationCase.window_seconds[0]) && evaluationCase.window_seconds[0] >= 0 &&
      isFiniteNumber(evaluationCase.window_seconds[1]) &&
      evaluationCase.window_seconds[1] > evaluationCase.window_seconds[0] &&
      Number.isInteger(evaluationCase.delay_samples) && Number(evaluationCase.delay_samples) >= 0 &&
      typeof evaluationCase.before_sha256 === "string" && /^[a-f0-9]{64}$/.test(evaluationCase.before_sha256) &&
      typeof evaluationCase.after_sha256 === "string" && /^[a-f0-9]{64}$/.test(evaluationCase.after_sha256) &&
      isNonEmptyString(evaluationCase.manifest));
  if (
    report.schema_version !== 1 ||
    !isNonEmptyString(report.id) ||
    !isNonEmptyString(report.created_at) || !Number.isFinite(Date.parse(report.created_at)) ||
    !isNonEmptyString(report.scope) ||
    report.standard_compliance !== false ||
    !isFiniteNumber(report.analysis_rate) || !validRates.has(report.analysis_rate) ||
    !isFiniteNumber(report.duration_seconds) || report.duration_seconds <= 0 ||
    !validSource || !validAlignment || !validMetrics || !validBands || !validFigures ||
    notices === null || !notices.every((notice) => typeof notice === "string") ||
    !validFigureScope || !validEvaluationCase
  ) {
    throw new Error("图形引擎返回的报告不完整或图形地址无效，请重新分析。");
  }
  return report as unknown as AudioGraphicReport;
}
