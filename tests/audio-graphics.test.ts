import assert from "node:assert/strict";
import { test } from "node:test";
import {
  AUDIO_GRAPH_FOCUS_FIGURES,
  AUDIO_GRAPH_FOCUS_LABELS,
  parseAudioGraphicReport,
} from "../src/domain/audio-graphics.ts";
import type { AudioGraphicReport } from "../src/domain/audio-graphics.ts";

const report: AudioGraphicReport = {
  schema_version: 1,
  id: "result_123",
  created_at: "2026-10-23T00:00:00Z",
  scope: "paired audio visualization and engineering diagnostics",
  standard_compliance: false,
  analysis_rate: 16000,
  duration_seconds: 2,
  source: [
    { role: "before", name: "before.wav", sha256: "a".repeat(64), original_sample_rate: 16000, original_channels: 1, selected_channel: 1, format: "WAV" },
    { role: "after", name: "after.wav", sha256: "b".repeat(64), original_sample_rate: 16000, original_channels: 1, selected_channel: 1, format: "WAV" },
  ],
  alignment: { requested: true, applied: true, delay_ms: 0, correlation: 0.99, reliable: true, note: "" },
  metrics: [
    { id: "rms_before", label: "处理前 RMS", value: -20, unit: "dBFS", status: "ok", detail: "" },
    { id: "rms_after", label: "处理后 RMS", value: -22, unit: "dBFS", status: "ok", detail: "" },
  ],
  bands: [],
  notices: [],
  figure_scope: {
    stft_fft_size: 512,
    stft_hop: 160,
    time_offset_seconds: 0,
    log_frequency_edge_min_hz: 20,
    log_frequency_edge_max_hz: 8000,
    high_time_fft_size: 128,
    high_time_hop: 32,
  },
  figures: ["waveform-energy", "log-stft-before", "log-stft-after", "spectral-delta", "triple"].map((id) => ({
    id,
    title: id,
    category: id === "waveform-energy" ? "dynamics" : "time-frequency",
    description: `${id} description`,
    detail: `${id} detail`,
    url: `/api/audio-graphics/result_123/figures/${id}.png`,
    download_name: `${id}.png`,
  })),
};

test("audio graphics report requires complete, same-origin figure identities", () => {
  assert.equal(parseAudioGraphicReport(report).figures.length, 5);
  const missing = structuredClone(report);
  missing.figures.pop();
  assert.throws(() => parseAudioGraphicReport(missing), /不完整/);
  const injected = structuredClone(report);
  injected.figures[0].url = "https://example.com/figure.png";
  assert.throws(() => parseAudioGraphicReport(injected), /图形地址无效/);
});

test("audio graphics report rejects invalid metric values and duplicate figures", () => {
  const invalidMetric = structuredClone(report);
  invalidMetric.metrics[0].value = Number.POSITIVE_INFINITY;
  assert.throws(() => parseAudioGraphicReport(invalidMetric), /不完整/);
  const duplicate = structuredClone(report);
  duplicate.figures[1].id = duplicate.figures[0].id;
  assert.throws(() => parseAudioGraphicReport(duplicate), /不完整/);
});

test("audio graphics report rejects malformed identity, alignment and band evidence", () => {
  const badHash = structuredClone(report);
  badHash.source[0].sha256 = "not-a-sha256";
  assert.throws(() => parseAudioGraphicReport(badHash), /不完整/);

  const badAlignment = structuredClone(report);
  badAlignment.alignment.correlation = 1.5;
  assert.throws(() => parseAudioGraphicReport(badAlignment), /不完整/);

  const mismatchedUrl = structuredClone(report);
  mismatchedUrl.figures[0].url = `/api/audio-graphics/result_123/figures/${mismatchedUrl.figures[1].id}.png`;
  assert.throws(() => parseAudioGraphicReport(mismatchedUrl), /不完整/);

  const duplicateRole = structuredClone(report);
  duplicateRole.source[0].role = "after";
  assert.throws(() => parseAudioGraphicReport(duplicateRole), /不完整/);
});

test("audio graphics report accepts finite paired band evidence", () => {
  const paired = structuredClone(report);
  paired.bands = [
    { id: "low_20_300", label: "低频 20–300 Hz", before_db: -30, after_db: -34, delta_db: -4 },
  ];
  assert.equal(parseAudioGraphicReport(paired).bands[0].delta_db, -4);
  const invalidBand = structuredClone(paired);
  invalidBand.bands[0].after_db = Number.NaN;
  assert.throws(() => parseAudioGraphicReport(invalidBand), /不完整/);
});

test("audio graphics report validates frozen evaluation-case identity", () => {
  const frozen = structuredClone(report);
  frozen.figure_scope.time_offset_seconds = 13;
  frozen.figure_scope.fixed_delay_samples = 165;
  frozen.evaluation_case = {
    asset_id: "chain-super_gaussian_optimal-pass",
    case_id: "field-p0-m4-cylindrical-snr15",
    detail: "4 麦 · cylindrical · SNR 15 dB · near_late",
    window_seconds: [13, 21],
    delay_samples: 165,
    before_sha256: "c".repeat(64),
    after_sha256: "d".repeat(64),
    manifest: "public/evaluation/pipeline-cases/manifest.json",
  };
  assert.equal(parseAudioGraphicReport(frozen).evaluation_case?.delay_samples, 165);
  const invalid = structuredClone(frozen);
  invalid.evaluation_case!.delay_samples = -1;
  assert.throws(() => parseAudioGraphicReport(invalid), /不完整/);
});

test("graph focus shows all eight figures for the chain and only module-critical figures elsewhere", () => {
  const valid = new Set([
    "waveform-energy",
    "log-stft-before",
    "log-stft-after",
    "spectral-delta",
    "triple",
    "ltas",
    "erb",
    "high-time",
  ]);
  assert.equal(AUDIO_GRAPH_FOCUS_FIGURES.all.length, 8);
  assert.deepEqual(new Set(AUDIO_GRAPH_FOCUS_FIGURES.all), valid);
  assert.deepEqual(AUDIO_GRAPH_FOCUS_FIGURES.drc, ["waveform-energy", "high-time"]);
  assert.equal(AUDIO_GRAPH_FOCUS_FIGURES.agc.length, 3);
  for (const focus of ["ns", "aec", "bf", "eq"] as const) {
    assert.ok(AUDIO_GRAPH_FOCUS_FIGURES[focus].length >= 2);
    assert.ok(AUDIO_GRAPH_FOCUS_FIGURES[focus].length <= 4);
  }
  for (const [focus, ids] of Object.entries(AUDIO_GRAPH_FOCUS_FIGURES)) {
    assert.ok(AUDIO_GRAPH_FOCUS_LABELS[focus as keyof typeof AUDIO_GRAPH_FOCUS_LABELS]);
    assert.ok(ids.every((id) => valid.has(id)));
  }
});
