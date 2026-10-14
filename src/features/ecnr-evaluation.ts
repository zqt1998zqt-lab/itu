export type EcnrTailLevel = { available?: boolean; rms_dbfs?: number | null; quantization_dbfs?: number | null; near_measurement_floor?: boolean };
export type EcnrTailEvent = {
  sequence_id: string; side: "candidate" | "baseline";
  details: { stop_s: number; quiet_reference_duration_s: number;
    scans: { horizon_s: number; observed_duration_s: number; complete: boolean;
      truncation_reason: string | null; before?: EcnrTailLevel; after?: EcnrTailLevel;
      envelope?: { block_ms: number; time_origin_s: number; before_dbfs: (number | null)[]; after_dbfs: (number | null)[] } }[];
  };
};
export type EcnrStage = {
  name: string; file: string; sample_rate: number; channels: number;
  fixed_delay_samples: number; trimmed_leading_samples: number;
  measurement_domain: string; duration_s: number; timeline_offset_ms?: number;
  waveform?: { min: number[]; max: number[]; channel?: number };
};
export type EcnrReport = {
  schema_version: 1; state: "running" | "completed" | "blocked";
  complete: boolean; identity_verified: boolean;
  baseline: { id: string; quality_passed: number; total_phases: number };
  quality?: { baseline_pass: number; candidate_pass: number; total_phases: number;
    phase_wins: number; phase_regressions: number; repaired_gates: number; new_failed_gates: number };
  configurations?: { id: string; rate: number; microphones: number; variant: string; baseline_pass: number; candidate_pass: number; phases: number }[];
  failures?: { sequence_id: string; label: string; segment: string; metric: string; value: number | null; operator: string; threshold: number | null; baseline_value?: number | null }[];
  calibration?: { status: string; summary: string };
  perceptual?: { status: string; scope: string; scores?: { metric: "SIG" | "BAK" | "OVRL"; baseline_median: number; candidate_median: number; paired_delta_median: number; windows: number }[]; short_repeated_count?: number };
  runtime?: { trials: { id: string; rate: number; microphones: number; process_p95_ms: number; process_max_ms: number; process_rtf: number; completion_deadline_misses: number | null; release_lateness_max_ms: number | null; dsp_delay_ms: number }[] };
  audits?: { post_playback_tails?: EcnrTailEvent[]; nearzero_windows?: { sequence_id: string; label: string; segment: string; input_dbfs: number | null; output_dbfs: number | null; total_attenuation_db: number | null; reason: string | null; details?: { before?: EcnrTailLevel; after?: EcnrTailLevel } }[] };
  recommendation?: { eligible: boolean; reason: string };
  sequences?: { id: string; stages: EcnrStage[] }[];
};
const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);
export function parseEcnrReport(value: unknown): EcnrReport {
  if (!value || typeof value !== "object") throw Error("报告内容无法读取。");
  const r = value as EcnrReport;
  if (r.schema_version !== 1 || !["running", "completed", "blocked"].includes(r.state) ||
      typeof r.complete !== "boolean" || typeof r.identity_verified !== "boolean" ||
      !r.baseline || typeof r.baseline.id !== "string" || !finite(r.baseline.quality_passed) ||
      !finite(r.baseline.total_phases)) throw Error("报告格式尚不受支持，请更新或重新生成报告。");
  for (const key of ["configurations", "failures", "sequences"] as const) {
    if (r[key] !== undefined && !Array.isArray(r[key])) throw Error("报告列表格式不正确。");
  }
  if (r.quality && Object.values(r.quality).some(v => !finite(v))) throw Error("质量指标包含不可用数值。");
  if (r.audits?.post_playback_tails !== undefined && !Array.isArray(r.audits.post_playback_tails)) throw Error("尾迹事件格式不正确。");
  return r;
}
export function isRecommendable(r: EcnrReport): boolean {
  return r.state === "completed" && r.complete && r.identity_verified && r.recommendation?.eligible === true;
}
// File sample zero is mapped to the original input timeline by declared delay and trimming.
export function stageOffsetSeconds(stage: EcnrStage): number | null {
  if (!finite(stage.sample_rate) || stage.sample_rate <= 0 || !finite(stage.duration_s) ||
      stage.duration_s <= 0 || !finite(stage.fixed_delay_samples) || stage.fixed_delay_samples < 0 ||
      !finite(stage.trimmed_leading_samples) || stage.trimmed_leading_samples < 0) return null;
  const derived = (stage.trimmed_leading_samples - stage.fixed_delay_samples) / stage.sample_rate;
  if (stage.timeline_offset_ms === undefined) return derived;
  if (!finite(stage.timeline_offset_ms) || Math.abs(stage.timeline_offset_ms / 1000 - derived) > 1 / stage.sample_rate) return null;
  return stage.timeline_offset_ms / 1000;
}
export function stageTimeline(stages: EcnrStage[]): { start: number; end: number } | null {
  const intervals = stages.flatMap(stage => {
    const offset = stageOffsetSeconds(stage);
    return offset === null ? [] : [{ start: offset, end: offset + stage.duration_s }];
  });
  return intervals.length ? { start: Math.min(0, ...intervals.map(x => x.start)), end: Math.max(...intervals.map(x => x.end)) } : null;
}
export function waveformPolygon(stage: EcnrStage, timeline: { start: number; end: number }): string | null {
  const offset = stageOffsetSeconds(stage), waveform = stage.waveform;
  if (offset === null || !waveform || !Number.isInteger(stage.channels) || stage.channels < 1 ||
      (stage.channels > 1 && waveform.channel === undefined) ||
      (waveform.channel !== undefined && (!Number.isInteger(waveform.channel) || waveform.channel < 0 || waveform.channel >= stage.channels)) || waveform.min.length < 2 ||
      waveform.min.length !== waveform.max.length || timeline.end <= timeline.start ||
      [...waveform.min, ...waveform.max].some(v => !finite(v) || Math.abs(v) > 1)) return null;
  const x = (i: number) => 10 + (offset + i * stage.duration_s / (waveform.min.length - 1) - timeline.start) / (timeline.end - timeline.start) * 780;
  const y = (v: number) => 50 - v * 42;
  const top = waveform.max.map((v, i) => `${x(i).toFixed(2)},${y(v).toFixed(2)}`);
  const bottom = waveform.min.map((v, i) => `${x(i).toFixed(2)},${y(v).toFixed(2)}`).reverse();
  return [...top, ...bottom].join(" ");
}
export function fileTimeAtInput(stage: EcnrStage, inputSeconds: number): number | null {
  const offset = stageOffsetSeconds(stage);
  if (offset === null || !finite(inputSeconds)) return null;
  const t = inputSeconds - offset;
  return t >= 0 && t < stage.duration_s ? t : null;
}
export function localAudioUrl(file: string): string | null {
  if (!file.startsWith("/evaluation/") || file.startsWith("//") || file.includes("\\") || file.split("/").includes("..")) return null;
  return file;
}
export function tailRmsText(level: EcnrTailLevel | undefined): string {
  if (!level || level.available === false || level.rms_dbfs === undefined) return "不可测";
  if (level.rms_dbfs === null) return "数字静音";
  return Number.isFinite(level.rms_dbfs) ? level.rms_dbfs.toFixed(3) : "不可测";
}
