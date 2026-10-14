import type { EcnrReport } from "./ecnr-evaluation";

export type FinalGate = "passed" | "failed" | "pending";
type Quality = { baseline_pass: number; candidate_pass: number; phase_regressions: number; new_failed_gates: number };
type Perceptual = { status: string; paired_windows: number; expected_windows: number; paired_delta_median: { SIG: number; BAK: number; OVRL: number } };
export type FinalTrial = {
  id: string; side: "historical" | "baseline" | "candidate"; repetition: number | null;
  rate: number; microphones: number; frames: number; process_p95_ms: number; process_max_ms: number;
  process_rtf: number; process_frames_over_10ms: number; release_lateness_max_ms: number;
  completion_deadline_misses: number; already_late_before_processing: number; remaining_misses: number;
  dsp_delay_ms: number; processing_pass?: boolean; offline_audio_parity?: boolean;
};
export type FinalAcceptance = {
  schema_version: 1; status: "waiting_full" | "waiting_validation" | "waiting_paced" | "completed" | "quality_rejected" | "blocked";
  complete: boolean; identity_verified: boolean; observed_at: string; scope: string;
  binding: Record<string, string>;
  continuous?: { complete: boolean; identity_verified: boolean; quality: Quality; calibration_passed: boolean; perceptual: Perceptual } | null;
  independent?: { state: string; complete: boolean; total_scenes: number; processed_scenes: number; quality?: Quality;
    validation?: { status: string; integrity_verified: boolean; checked_scenes: number; errors: number; error_details?: string[] };
    perceptual?: Perceptual } | null;
  performance?: { complete: boolean; trials: FinalTrial[]; candidate_processing_pass?: boolean; candidate_deadline_pass?: boolean;
    audio_parity_pass?: boolean; historical_misses_preserved: number } | null;
  acceptance: { quality: FinalGate; perceptual: FinalGate; real_time: FinalGate; integrity: FinalGate };
  recommendation: { eligible: boolean; reason: string };
  artifacts: { label: string; file: string; sha256: string; bytes: number }[];
  errors?: string[];
};
const finite = (n: unknown): n is number => typeof n === "number" && Number.isFinite(n);
const sha = (s: unknown) => typeof s === "string" && /^[a-f0-9]{64}$/.test(s);
export function parseFinalAcceptance(value: unknown): FinalAcceptance {
  if (!value || typeof value !== "object") throw Error("最终验收报告无法读取。");
  const r = value as FinalAcceptance;
  if (r.schema_version !== 1 || !["waiting_full", "waiting_validation", "waiting_paced", "completed", "quality_rejected", "blocked"].includes(r.status) ||
      typeof r.complete !== "boolean" || typeof r.identity_verified !== "boolean" || !r.binding || typeof r.binding !== "object" ||
      !r.acceptance || Object.keys(r.acceptance).sort().join(",") !== "integrity,perceptual,quality,real_time" ||
      Object.values(r.acceptance).some(s => !["passed", "failed", "pending"].includes(s)) ||
      typeof r.recommendation?.eligible !== "boolean" || typeof r.recommendation.reason !== "string" || !Array.isArray(r.artifacts) ||
      (r.performance && !Array.isArray(r.performance.trials))) throw Error("最终验收报告格式不受支持。");
  const validPerception = (p: Perceptual) => p && typeof p.status === "string" && finite(p.paired_windows) && finite(p.expected_windows) &&
    p.paired_delta_median && ["SIG", "BAK", "OVRL"].every(k => finite(p.paired_delta_median[k as "SIG" | "BAK" | "OVRL"]));
  const validQuality = (q: Quality) => q && ["baseline_pass", "candidate_pass", "phase_regressions", "new_failed_gates"].every(k => finite(q[k as keyof Quality]));
  if (r.continuous && (!validQuality(r.continuous.quality) || !validPerception(r.continuous.perceptual)) ||
      r.independent && (!finite(r.independent.total_scenes) || !finite(r.independent.processed_scenes) ||
        r.independent.quality && !validQuality(r.independent.quality) || r.independent.perceptual && !validPerception(r.independent.perceptual)) ||
      r.performance?.trials.some(t => !t || typeof t.id !== "string" || !["historical", "baseline", "candidate"].includes(t.side) ||
        ["rate", "microphones", "frames", "process_p95_ms", "process_max_ms", "process_rtf", "process_frames_over_10ms",
          "release_lateness_max_ms", "completion_deadline_misses", "already_late_before_processing", "remaining_misses", "dsp_delay_ms"].some(k => !finite(t[k as keyof FinalTrial]))) ||
      r.artifacts.some(a => !a || typeof a.label !== "string" || typeof a.file !== "string" || !sha(a.sha256) || !finite(a.bytes))) throw Error("最终验收指标格式不正确。");
  return r;
}
export function finalMatchesExperiment(final: FinalAcceptance, report: EcnrReport): boolean {
  const actual = report as EcnrReport & { identity?: { baseline_cli_sha256?: string; cli_sha256?: string }; publication?: { matrix_report_sha256?: string } };
  const b = final.binding;
  return sha(b.continuous_report_sha256) && b.continuous_report_sha256 === actual.publication?.matrix_report_sha256 &&
    sha(b.baseline_cli_sha256) && b.baseline_cli_sha256 === actual.identity?.baseline_cli_sha256 &&
    sha(b.candidate_cli_sha256) && b.candidate_cli_sha256 === actual.identity?.cli_sha256;
}
export function finalIsRecommendable(r: FinalAcceptance, report: EcnrReport): boolean {
  if (!finalMatchesExperiment(r, report) || r.status !== "completed" || !r.complete || !r.identity_verified ||
      r.recommendation.eligible !== true || Object.values(r.acceptance).some(s => s !== "passed")) return false;
  for (const key of ["full_report_sha256", "full_identity_sha256", "full_index_sha256", "validation_report_sha256",
    "validation_binding_sha256", "verifier_sha256", "paced_report_sha256", "paced_identity_sha256", "runner_sha256", "rules_sha256", "manifest_sha256"]) {
    if (!sha(r.binding[key])) return false;
  }
  const c = r.continuous, f = r.independent, p = r.performance;
  if (!c?.complete || !c.identity_verified || !c.calibration_passed || c.quality.baseline_pass !== 184 ||
      c.quality.candidate_pass <= 184 || c.quality.phase_regressions !== 0 || c.quality.new_failed_gates !== 0 ||
      !f?.complete || f.state !== "completed" || f.total_scenes !== 20298 || f.processed_scenes !== 20298 ||
      f.quality?.phase_regressions !== 0 || f.quality.new_failed_gates !== 0 || !f.validation?.integrity_verified ||
      f.validation.status !== "passed_full_integrity_and_quality" || f.validation.checked_scenes !== 20298 || f.validation.errors !== 0) return false;
  for (const scores of [c.perceptual, f.perceptual]) {
    if (!scores || scores.status !== "completed" || !finite(scores.paired_windows) || scores.paired_windows <= 0 ||
        scores.paired_windows !== scores.expected_windows || !finite(scores.paired_delta_median?.SIG) ||
        scores.paired_delta_median.SIG < 0 || !finite(scores.paired_delta_median.OVRL) || scores.paired_delta_median.OVRL < 0) return false;
  }
  if (c.perceptual.paired_windows !== 510 || !p?.complete || p.candidate_processing_pass !== true ||
      p.candidate_deadline_pass !== true || p.audio_parity_pass !== true || p.historical_misses_preserved !== 37 || p.trials.length !== 5) return false;
  const ids = ["historical-v1-37-misses", "baseline-repeat-01", "baseline-repeat-02", "candidate-repeat-01", "candidate-repeat-02"];
  if (new Set(p.trials.map(t => t.id)).size !== 5 || p.trials.some(t => !ids.includes(t.id))) return false;
  return p.trials.every(t => {
    if (t.rate !== 48000 || t.microphones !== 4 || t.frames !== 23500 ||
        [t.process_p95_ms, t.process_max_ms, t.process_rtf, t.release_lateness_max_ms, t.dsp_delay_ms].some(v => !finite(v) || v < 0)) return false;
    if (t.id === "historical-v1-37-misses") return t.side === "historical" && t.completion_deadline_misses === 37;
    if (t.offline_audio_parity !== true || t.id !== `${t.side}-repeat-${String(t.repetition).padStart(2, "0")}`) return false;
    return t.side === "baseline" || (t.side === "candidate" && t.processing_pass === true && t.completion_deadline_misses === 0 &&
      t.process_p95_ms <= 1 && t.process_max_ms <= 10 && t.process_rtf <= .1 && t.process_frames_over_10ms === 0);
  });
}
export function finalArtifactUrl(file: string): string | null {
  if (!file.startsWith("/evaluation/ecnr-v2/final/") || file.includes("\\") || file.includes("%") || file.split("/").includes("..") ||
      file.startsWith("//") || /[?#]/.test(file)) return null;
  return file;
}
