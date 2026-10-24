import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "node:test";
import metadata from "../src/data/pipeline-case-metadata.json" with { type: "json" };
import agcEvidence from "../src/data/agc-case-evidence.json" with { type: "json" };
import { aecAlgorithms } from "../src/features/aec-pipeline-data.ts";
import { bfAlgorithms } from "../src/features/bf-pipeline-data.ts";
import { eqAlgorithms } from "../src/features/eq-pipeline-data.ts";
import { drcAlgorithms } from "../src/features/drc-pipeline-data.ts";
import { agcAlgorithms } from "../src/features/agc-pipeline-data.ts";
import { resolvePipelineCase } from "../src/features/pipeline-case-metadata.ts";

const root = fileURLToPath(new URL("..", import.meta.url));
const read = (path: string) => JSON.parse(readFileSync(resolve(root, path), "utf8"));
const hash = (path: string) => createHash("sha256").update(readFileSync(resolve(root, path))).digest("hex");
const manifest = read("public/evaluation/pipeline-cases/manifest.json").assets as typeof metadata;
type ChainCaseAsset = {
  asset_id: string;
  case_id: string;
  detail: string;
  before: string;
  after: string;
  window_seconds: number[];
  delay_samples: number;
  sample_rate: number;
  before_label: string;
  after_label: string;
  waveform: string;
  mel: string;
  analysis: string;
};
const sourceManifest = read("public/evaluation/pipeline-cases/manifest.json").assets as ChainCaseAsset[];

test("every displayed case gets its identity and timing from the generated figure manifest", () => {
  assert.equal(metadata.length, manifest.length);
  assert.equal(new Set(metadata.map(asset => asset.asset_id)).size, metadata.length);
  for (const asset of metadata) {
    const plotted = manifest.find(item => item.asset_id === asset.asset_id)!;
    assert.ok(plotted, asset.asset_id);
    for (const key of Object.keys(asset) as (keyof typeof asset)[]) assert.deepEqual(asset[key], plotted[key], asset.asset_id + ": " + key);
  }
  const displayed = new Set<string>();
  for (const algorithm of [...aecAlgorithms, ...bfAlgorithms, ...eqAlgorithms, ...drcAlgorithms, ...agcAlgorithms]) {
    const cases = algorithm.evaluation.caseGroups?.map(group => group.caseAudio)
      ?? (algorithm.evaluation.caseAudio ? [algorithm.evaluation.caseAudio] : []);
    for (const value of cases) {
      const actual = resolvePipelineCase(value);
      const asset = metadata.find(item => item.waveform === value.waveformSrc)!;
      assert.deepEqual(actual.windowSeconds, asset.window_seconds);
      assert.equal(actual.delaySamples, asset.delay_samples);
      assert.equal(actual.afterFile, asset.after);
      assert.equal(actual.assetId, asset.asset_id);
      displayed.add(value.waveformSrc);
    }
  }
  assert.ok(displayed.size >= 51, "current AEC/BF/EQ/DRC/AGC cases must be exercised");
  assert.equal(displayed.size, new Set(
    [...aecAlgorithms, ...bfAlgorithms, ...eqAlgorithms, ...drcAlgorithms, ...agcAlgorithms]
      .flatMap(algorithm => {
        const groups = algorithm.evaluation.caseGroups?.map(group => group.caseAudio)
          ?? (algorithm.evaluation.caseAudio ? [algorithm.evaluation.caseAudio] : []);
        return groups.map(value => value.waveformSrc);
      }),
  ).size);
});

test("AEC3 state-fusion history is frozen evidence and not advertised as current runtime", () => {
  const algorithm = aecAlgorithms.find(item => item.id === "aec3-state-fusion-history")!;
  const report = read(algorithm.evaluation.source);
  assert.deepEqual(report.variants.aec3_off, {
    cases: 27, errors: 0, far_ge_15: 13, linear_erle_ge_6: 11,
    unavailable_linear: 0, new_near_dt_failures: 0, passed: 10,
  });
  assert.deepEqual(report.variants.aec3_fusion, {
    cases: 27, errors: 0, far_ge_15: 13, linear_erle_ge_6: 11,
    unavailable_linear: 0, new_near_dt_failures: 1, passed: 7,
  });
  const identity = read("build/aec3-state-fusion-20261020-r3/identity.json");
  assert.deepEqual(identity.variants.aec3_fusion, [
    "--aec-backend", "webrtc", "--aec3-monitor-cancellation-fusion",
  ]);
  assert.match(identity.dependencies[
    resolve(root, "build/aec3-state-fusion-20261020-r3/artifacts/itu_audio_cli")
  ], /^c33c65a4ea8d2ad75f9a524b79c85af40ead24fc9400c514de4a79addf9963be$/);
  assert.match(algorithm.boundary, /当前可选算法/);
  assert.match(algorithm.boundary, /不能继承/);
  assert.match(algorithm.cli, /当前源码拒绝/);
});

test("linear-phase EQ is bound to current executable identity, delay and paired quality", () => {
  const algorithm = eqAlgorithms.find(item => item.id === "linear-phase-fir")!;
  const report = read(algorithm.evaluation.source);
  const variant = report.variants.linear_phase_three_band;
  assert.equal(report.cli_sha256, "b3833771ae2ffd5a79e9a10bb8b08101fd2c2f507400663d0db18bce7af90544");
  assert.equal(report.results_sha256, "b4e5fa04a8fbd1f2e4961fd2e358b5c0103e693c8e2eca1e4a7b268c689210a6");
  assert.equal(variant.cases, 108);
  assert.equal(variant.engineering_pass_cases, 0);
  assert.equal(variant.outputs_changed_vs_bypass, 108);
  assert.equal(variant.clipped_cases, 0);
  assert.equal(variant.paired_cases.length, 108);
  for (const rate of [16000, 24000, 48000]) {
    const rows = variant.paired_cases.filter(
      (row: { sample_rate: number }) => row.sample_rate === rate,
    );
    assert.equal(rows.length, rate === 16000 ? 54 : 27);
    const evidence = rows.map((row: { original_run: string; scene_id: string; noise: string }) => {
      const path = `${algorithm.evaluation.source.replace(/report\.json$/, "")}linear_phase_three_band/cases/${row.original_run}__${row.scene_id}__${row.noise}/runs/full/result.json`;
      return read(path);
    });
    assert.ok(evidence.every(
      (engine: { output_eq_phase: string; output_eq_delay_samples: number }) =>
        engine.output_eq_phase === "linear" &&
        engine.output_eq_delay_samples === ({ 16000: 160, 24000: 240, 48000: 480 } as Record<number, number>)[rate],
    ));
    const delays = evidence.map((engine: { dsp_delay_samples: number }) => engine.dsp_delay_samples);
    assert.ok(Math.min(...delays) >= ({ 16000: 256, 24000: 448, 48000: 816 } as Record<number, number>)[rate]);
  }
});

test("stale editorial timing is corrected and unknown/mixed figure pairs fail closed", () => {
  const original = bfAlgorithms.find(algorithm => algorithm.evaluation.caseGroups?.length)!.evaluation.caseGroups![0].caseAudio;
  const expected = resolvePipelineCase(original);
  const stale = { ...original, id: "stale-id", delaySamples: -1, windowSeconds: [0, 1] as [number, number], afterFile: "wrong.wav" };
  assert.deepEqual(resolvePipelineCase(stale), { ...stale, id: expected.id, assetId: expected.assetId,
    sampleRate: expected.sampleRate, delaySamples: expected.delaySamples,
    windowSeconds: expected.windowSeconds, beforeFile: expected.beforeFile, afterFile: expected.afterFile,
    beforeChannel: expected.beforeChannel, afterChannel: expected.afterChannel });
  assert.throws(() => resolvePipelineCase({ ...original, waveformSrc: "/unknown.svg" }), /清单不匹配/);
  assert.throws(() => resolvePipelineCase({ ...original, melSrc: metadata.find(asset => asset.mel !== original.melSrc)!.mel }), /清单不匹配/);
});

test("enhancement-chain cases expose the exact frozen asset id used by the graph engine", () => {
  const chainCases = sourceManifest.filter(asset => asset.asset_id.startsWith("chain-"));
  assert.equal(chainCases.length, 20);
  for (const asset of chainCases) {
    const value = {
      id: asset.case_id,
      detail: asset.detail,
      sampleRate: asset.sample_rate,
      windowSeconds: [asset.window_seconds[0], asset.window_seconds[1]] as [number, number],
      delaySamples: asset.delay_samples,
      beforeLabel: asset.before_label,
      afterLabel: asset.after_label,
      beforeFile: asset.before,
      afterFile: asset.after,
      waveformSrc: asset.waveform,
      melSrc: asset.mel,
      analysis: asset.analysis,
    } as const;
    const resolved = resolvePipelineCase(value);
    assert.equal(resolved.assetId, asset.asset_id);
    assert.deepEqual(resolved.windowSeconds, asset.window_seconds);
    assert.equal(resolved.delaySamples, asset.delay_samples);
    assert.match(hash(resolved.beforeFile), /^[a-f0-9]{64}$/);
    assert.match(hash(resolved.afterFile), /^[a-f0-9]{64}$/);
  }
});

test("AEC3 score, verdicts, final WAVs and causal delays belong to aec3_off", () => {
  const evaluation = aecAlgorithms.find(algorithm => algorithm.id === "aec3")!.evaluation;
  const report = read(evaluation.source);
  assert.equal(evaluation.headline, `${report.variants.aec3_off.passed} / ${report.variants.aec3_off.cases} 场景通过`);
  assert.equal(report.variants.aec3_fusion.passed, 7, "separate candidate negative control");
  assert.deepEqual(evaluation.caseGroups!.map(group => group.id), ["pass", "fail"]);
  for (const group of evaluation.caseGroups!) {
    const actual = resolvePipelineCase(group.caseAudio);
    assert.match(actual.afterFile, /\/aec3_off\/.*\/result\.after\.wav$/);
    assert.equal(actual.afterFile, group.caseAudio.afterFile);
    const dir = dirname(resolve(root, actual.afterFile));
    const acceptance = read(resolve(dir, "acceptance.json"));
    assert.equal(acceptance.passed, group.id === "pass");
    assert.equal(actual.delaySamples, read(resolve(dir, "result.json")).dsp_delay_samples);
    if (group.id === "fail") {
      assert.deepEqual(actual.windowSeconds, [6, 9]);
      assert.ok(acceptance.gates.some((gate: { status: string; segment: string }) => gate.status === "fail" && gate.segment === "far_only"));
      assert.ok(!acceptance.gates.some((gate: { status: string; segment: string }) => gate.status === "fail" && gate.segment === "double_talk"));
    }
  }
});

test("all six evaluated BF failure plots show the actual 12–15 second scene window", () => {
  const failures = bfAlgorithms.flatMap(algorithm => algorithm.evaluation.caseGroups ?? []).filter(group => group.id === "fail");
  assert.equal(failures.length, 6);
  for (const group of failures) {
    assert.deepEqual(group.caseAudio.windowSeconds, [12, 15]);
    assert.deepEqual(resolvePipelineCase(group.caseAudio).windowSeconds, [12, 15]);
  }
});

test("look-ahead DRC uses the current source matrix and separates delay from quality", () => {
  const algorithm = drcAlgorithms.find(item => item.id === "lookahead-drc")!;
  const report = read(algorithm.evaluation.source);
  assert.equal(report.variants.lookahead_2ms.cases, 108);
  assert.equal(report.variants.lookahead_2ms.engineering_pass_cases, 81);
  assert.equal(report.variants.lookahead_2ms_off.outputs_changed_vs_bypass, 36);
  assert.equal(
    report.variants.lookahead_2ms_off.paired_cases.filter(
      (row: { output_changed: boolean; extra_delay_samples: number }) =>
        row.output_changed,
    ).every(
      (row: { output_changed: boolean; extra_delay_samples: number }) =>
        row.extra_delay_samples === 0,
    ),
    true,
  );
  assert.equal(report.variants.lookahead_2ms.outputs_changed_vs_bypass, 108);
  assert.equal(report.engineering_gates.max_extra_delay_ms, 3.0);
  const groups = algorithm.evaluation.caseGroups!;
  assert.deepEqual(groups.map(group => group.id), ["pass", "fail"]);
  assert.equal(groups[0].caseAudio.delaySamples, 32);
  assert.equal(groups[1].caseAudio.delaySamples, 96);
  for (const group of groups) {
    const actual = resolvePipelineCase(group.caseAudio);
    assert.ok(hash(actual.beforeFile).length === 64);
    assert.ok(actual.afterFile.includes("/lookahead_2ms/"));
    assert.ok(actual.waveformSrc.includes("drc-lookahead-"));
  }
});

test("all eight measured AGC configurations have their own passing and failing evidence", () => {
  assert.equal(agcEvidence.length, 16);
  const variants = [...new Set(agcEvidence.map(group => group.variant))];
  assert.equal(variants.length, 8);
  for (const variant of variants) {
    const groups = agcEvidence.filter(group => group.variant === variant);
    assert.deepEqual(groups.map(group => group.id), ["pass", "fail"]);
    for (const group of groups) {
      const report = read(group.source);
      const records = readFileSync(resolve(root, group.evidence.results), "utf8").trim().split("\n").map(line => JSON.parse(line));
      assert.equal(records.length, 108);
      assert.equal(records.filter(row => row.acceptance.passed).length, report.counts.pass);
      assert.equal(hash(group.source), group.evidence.report_sha256);
      assert.equal(hash(group.evidence.results), group.evidence.results_sha256);
      const row = records.find(row => row.scene.id === group.caseAudio.id)!;
      assert.ok(row, variant);
      assert.equal(row.acceptance.passed, group.id === "pass");
      assert.equal(group.evidence.passed, row.acceptance.passed);
      assert.deepEqual(group.caseAudio.windowSeconds, row.scene.windows[group.evidence.window]);
      assert.ok(group.evidence.gates.length > 0);
      assert.ok(group.evidence.gates.every(gate => gate.status === group.id && gate.segment === group.evidence.window));
      assert.ok(group.caseAudio.afterFile.startsWith(dirname(group.source) + "/"));
      const actual = resolvePipelineCase({ ...group.caseAudio, windowSeconds: [group.caseAudio.windowSeconds[0], group.caseAudio.windowSeconds[1]] });
      assert.equal(hash(actual.afterFile), row.metrics.output_sha256);
      assert.equal(hash(actual.beforeFile), row.scene.files["before.wav"].sha256);
    }
  }
  for (const [id, count] of [["guarded5", 2], ["adaptive-agc2", 2], ["fixed-digital", 8], ["tx-drc", 2], ["utterance-v4", 2]] as const) {
    const groups = agcAlgorithms.find(algorithm => algorithm.id === id)!.evaluation.caseGroups!;
    assert.equal(groups.length, count);
    assert.equal(new Set(groups.map(group => `${group.variant}-${group.id}`)).size, count);
  }
  assert.deepEqual(agcAlgorithms.find(algorithm => algorithm.id === "fixed-digital")!.evaluation.sources,
    [...new Set(agcEvidence.filter(group => group.algorithm === "fixed-digital").map(group => group.source))]);
});
