import test from "node:test";
import assert from "node:assert/strict";
import { fileTimeAtInput, isRecommendable, localAudioUrl, parseEcnrReport, stageOffsetSeconds, stageTimeline, waveformPolygon, tailRmsText } from "../src/features/ecnr-evaluation.ts";
import type { EcnrReport, EcnrStage } from "../src/features/ecnr-evaluation.ts";
const report: EcnrReport = { schema_version: 1, state: "running", complete: false, identity_verified: false, baseline: { id: "ecnr-v1", quality_passed: 184, total_phases: 270 } };
const stage: EcnrStage = { name: "output", file: "/evaluation/ecnr-v2/output.wav", sample_rate: 16000, channels: 1, fixed_delay_samples: 160, trimmed_leading_samples: 0, measurement_domain: "fullband", duration_s: 2, waveform: { min: [-.1, -.2], max: [.1, .2] } };
test("unfinished, blocked, or unverified reports cannot promote a configuration", () => {
  for (const partial of [{}, { state: "completed", complete: true }, { state: "blocked", complete: true, identity_verified: true }, { state: "completed", complete: false, identity_verified: true }]) {
    assert.equal(isRecommendable({ ...report, ...partial, recommendation: { eligible: true, reason: "quality passed" } } as EcnrReport), false);
  }
  assert.equal(isRecommendable({ ...report, state: "completed", complete: true, identity_verified: true, recommendation: { eligible: true, reason: "all checks passed" } }), true);
});
test("minimal running reports remain readable; unsupported and malformed reports fail", () => {
  assert.deepEqual(parseEcnrReport(report), report);
  for (const invalid of [null, {}, { ...report, schema_version: 2 }, { ...report, failures: {} }, { ...report, baseline: { id: "b", quality_passed: Infinity, total_phases: 270 } }]) assert.throws(() => parseEcnrReport(invalid));
});
test("timeline mapping uses declared fixed sample delay and trimming", () => {
  assert.equal(stageOffsetSeconds(stage), -.01);
  assert.equal(stageOffsetSeconds({ ...stage, trimmed_leading_samples: 160 }), 0);
  assert.equal(stageOffsetSeconds({ ...stage, sample_rate: 48000, fixed_delay_samples: 480 }), -.01);
  assert.equal(stageOffsetSeconds({ ...stage, timeline_offset_ms: -10 }), -.01);
  assert.equal(stageOffsetSeconds({ ...stage, timeline_offset_ms: 30 }), null);
  assert.equal(stageOffsetSeconds({ ...stage, sample_rate: 0 }), null);
});
test("all waveform stages share a common input timeline and original amplitude", () => {
  const undelayed = { ...stage, name: "reference", fixed_delay_samples: 0 };
  const extent = stageTimeline([stage, undelayed]);
  assert.deepEqual(extent, { start: -.01, end: 2 });
  const points = waveformPolygon(stage, extent!);
  assert.ok(points?.includes(",45.80"));
  assert.ok(points?.includes(",41.60"));
  assert.notEqual(points, waveformPolygon(undelayed, extent!));
  assert.equal(waveformPolygon({ ...stage, channels: 2 }, extent!), null);
  assert.equal(waveformPolygon({ ...stage, waveform: { min: [-2, 0], max: [0, 0] } }, extent!), null);
});
test("shared seek maps input time to each file and rejects absent time ranges", () => {
  assert.equal(fileTimeAtInput(stage, 1), 1.01);
  assert.equal(fileTimeAtInput({ ...stage, fixed_delay_samples: 0 }, 1), 1);
  assert.equal(fileTimeAtInput(stage, 2), null);
});
test("audio players only load published local evaluation files", () => {
  assert.equal(localAudioUrl(stage.file), stage.file);
  for (const file of ["https://example.com/x.wav", "//example.com/x.wav", "/evaluation/../secret.wav", "/tmp/a.wav", "/evaluation/\\secret.wav"]) assert.equal(localAudioUrl(file), null);
});
test("digital silence and missing tail measurements never render as zero dBFS", () => {
  assert.equal(tailRmsText({ rms_dbfs: null, available: true }), "数字静音");
  assert.equal(tailRmsText({ rms_dbfs: -101.1, available: true }), "-101.100");
  assert.equal(tailRmsText({ rms_dbfs: 0, available: true }), "0.000");
  assert.equal(tailRmsText({ rms_dbfs: null, available: false }), "不可测");
  assert.equal(tailRmsText(undefined), "不可测");
});
test("tail evidence is optional and preserves legacy reports", () => {
  assert.deepEqual(parseEcnrReport(report), report);
  const withTails = { ...report, audits: { post_playback_tails: [] } };
  assert.deepEqual(parseEcnrReport(withTails), withTails);
  assert.throws(() => parseEcnrReport({ ...report, audits: { post_playback_tails: {} } }));
});

test("multi-channel waveforms require an explicitly declared valid channel", () => {
  const multi = { ...stage, channels: 4 };
  const timeline = stageTimeline([multi])!;
  assert.equal(waveformPolygon(multi, timeline), null);
  assert.ok(waveformPolygon({ ...multi, waveform: { ...stage.waveform!, channel: 0 } }, timeline));
  assert.equal(waveformPolygon({ ...multi, waveform: { ...stage.waveform!, channel: 4 } }, timeline), null);
  assert.equal(waveformPolygon({ ...multi, waveform: { ...stage.waveform!, channel: -1 } }, timeline), null);
});

import { finalArtifactUrl, finalIsRecommendable, finalMatchesExperiment, parseFinalAcceptance } from "../src/features/ecnr-final-acceptance.ts";
import type { FinalAcceptance, FinalTrial } from "../src/features/ecnr-final-acceptance.ts";
const digest = "a".repeat(64), binary = "b".repeat(64), candidate = "c".repeat(64);
const experiment = { ...report, publication: { matrix_report_sha256: digest }, identity: { baseline_cli_sha256: binary, cli_sha256: candidate } } as EcnrReport;
const perception = { status: "completed", expected_windows: 510, paired_windows: 510, paired_delta_median: { SIG: 0, BAK: 0, OVRL: 0 } };
const finalTrial = (side: FinalTrial["side"], repetition: number | null): FinalTrial => ({
  id: side === "historical" ? "historical-v1-37-misses" : `${side}-repeat-${String(repetition).padStart(2, "0")}`,
  side, repetition, rate: 48000, microphones: 4, frames: 23500, process_p95_ms: .3, process_max_ms: .5,
  process_rtf: .03, process_frames_over_10ms: 0, release_lateness_max_ms: side === "historical" ? 120 : .1,
  completion_deadline_misses: side === "historical" ? 37 : 0, already_late_before_processing: side === "historical" ? 37 : 0,
  remaining_misses: 0, dsp_delay_ms: 2027 / 48000 * 1000, processing_pass: true, offline_audio_parity: true,
});
const final: FinalAcceptance = {
  schema_version: 1, status: "completed", complete: true, identity_verified: true, observed_at: "2026-10-11T00:00:00Z", scope: "HSAE host",
  binding: Object.fromEntries(["continuous_report_sha256", "full_report_sha256", "full_identity_sha256", "full_index_sha256", "validation_report_sha256", "validation_binding_sha256", "verifier_sha256", "paced_report_sha256", "paced_identity_sha256", "runner_sha256", "rules_sha256", "manifest_sha256"].map(k => [k, digest])).valueOf() as Record<string, string>,
  continuous: { complete: true, identity_verified: true, calibration_passed: true, quality: { baseline_pass: 184, candidate_pass: 193, phase_regressions: 0, new_failed_gates: 0 }, perceptual: perception },
  independent: { state: "completed", complete: true, total_scenes: 20298, processed_scenes: 20298, quality: { baseline_pass: 12000, candidate_pass: 12000, phase_regressions: 0, new_failed_gates: 0 },
    validation: { status: "passed_full_integrity_and_quality", integrity_verified: true, checked_scenes: 20298, errors: 0 }, perceptual: perception },
  performance: { complete: true, candidate_processing_pass: true, candidate_deadline_pass: true, audio_parity_pass: true, historical_misses_preserved: 37,
    trials: [finalTrial("historical", null), finalTrial("baseline", 1), finalTrial("candidate", 1), finalTrial("baseline", 2), finalTrial("candidate", 2)] },
  acceptance: { quality: "passed", perceptual: "passed", real_time: "passed", integrity: "passed" }, recommendation: { eligible: true, reason: "all gates passed" }, artifacts: [],
};
final.binding.baseline_cli_sha256 = binary; final.binding.candidate_cli_sha256 = candidate;

test("optional final contract leaves old reports compatible and rejects missing identity", () => {
  assert.deepEqual(parseEcnrReport(report), report);
  assert.equal(finalMatchesExperiment(final, report), false);
  assert.equal(finalIsRecommendable(final, report), false);
  assert.equal(finalIsRecommendable(parseFinalAcceptance(final), experiment), true);
  assert.throws(() => parseFinalAcceptance({ ...final, complete: "true" }));
});
test("final recommendation requires complete bound full coverage and every gate", () => {
  const mutations = [
    (r: FinalAcceptance) => { r.status = "waiting_full"; r.complete = false; },
    (r: FinalAcceptance) => { r.binding.validation_binding_sha256 = ""; },
    (r: FinalAcceptance) => { r.binding.continuous_report_sha256 = "d".repeat(64); },
    (r: FinalAcceptance) => { r.independent!.processed_scenes = 20297; },
    (r: FinalAcceptance) => { r.independent!.quality!.new_failed_gates = 1; },
    (r: FinalAcceptance) => { r.continuous!.perceptual.paired_delta_median.SIG = -.01; },
    (r: FinalAcceptance) => { r.performance!.trials.pop(); },
    (r: FinalAcceptance) => { r.performance!.trials[2].completion_deadline_misses = 1; },
    (r: FinalAcceptance) => { r.performance!.trials[2].process_p95_ms = 1.1; },
    (r: FinalAcceptance) => { r.performance!.trials[1].offline_audio_parity = false; },
    (r: FinalAcceptance) => { r.performance!.historical_misses_preserved = 0; },
  ];
  for (const mutate of mutations) { const r = structuredClone(final); mutate(r); assert.equal(finalIsRecommendable(r, experiment), false); }
});
test("verified quality rejection and failed pacing remain readable but never recommend", () => {
  const rejected = { ...final, status: "quality_rejected", complete: false, performance: null, acceptance: { ...final.acceptance, quality: "failed", real_time: "pending" }, recommendation: { eligible: false, reason: "quality rejected, paced not run" } } as FinalAcceptance;
  assert.equal(parseFinalAcceptance(rejected).acceptance.integrity, "passed"); assert.equal(finalIsRecommendable(rejected, experiment), false);
  const failed = { ...final, acceptance: { ...final.acceptance, real_time: "failed" } } as FinalAcceptance;
  assert.equal(finalIsRecommendable(failed, experiment), false);
});
test("final artifact links stay in their published static namespace", () => {
  assert.equal(finalArtifactUrl("/evaluation/ecnr-v2/final/abc/full-report.json"), "/evaluation/ecnr-v2/final/abc/full-report.json");
  for (const file of ["/evaluation/ecnr-v2/report.json", "/evaluation/ecnr-v2/final/../report.json", "/evaluation/ecnr-v2/final/%2e%2e/secret", "https://example.com/report", "/evaluation/ecnr-v2/final/abc/x.json?q=1"]) assert.equal(finalArtifactUrl(file), null);
});
