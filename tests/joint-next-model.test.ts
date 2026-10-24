import assert from "node:assert/strict";
import { test } from "node:test";
import { jointNextFigureTitle, jointNextStatusLabel, parseJointNextReport } from "../src/features/joint-next-model.ts";

const hash = "a".repeat(64);
function fixture() {
  return {
    schema_version: 1, case_count: 72, scope: "Known frozen scenes; original clean target and gates.",
    candidate_id: "next", candidate_cli_sha256: hash,
    admission: { status: "rejected", checks: [{ id: "speech", label: "All speech windows", pass: false, detail: "A measured speech regression remains." }] },
    variants: [
      { id: "prior", label: "Prior joint", kind: "baseline", original_pass: 64, all_speech_pass: 59, mean_sisdr_delta_db: 1, mean_stoi_delta: .01 },
      { id: "next", label: "Next candidate", kind: "candidate", original_pass: 65, all_speech_pass: 58, mean_sisdr_delta_db: 1.1, mean_stoi_delta: .01 },
    ],
    far_windows: [{ window: "startup", label: "Far startup", cases: 24, mean_change_db: 1, min_change_db: -.2 }],
    limitations: ["Not an independent holdout or device acceptance."],
    figures: [{ scene_id: "case", window: "near_only", label: "Known scene", items: [{ id: "waveform", url: "/evaluation/vcp-joint-next/case/waveform.png", sha256: hash }] }],
    artifacts: [{ label: "Results", url: "/evaluation/vcp-joint-next/results.jsonl", sha256: hash }],
  };
}
test("a rejected candidate remains distinct from frozen baselines and has no implied execution grant", () => {
  const value = parseJointNextReport(fixture());
  assert.equal(value.candidate_id, "next");
  assert.equal(value.variants.find((row) => row.id === value.candidate_id)?.all_speech_pass, 58);
  assert.equal(jointNextStatusLabel(value.admission.status), "未通过准入");
  assert.equal("executable" in value, false);
});
test("graph types use Chinese titles matching the actual graph-engine semantics", () => {
  assert.equal(jointNextFigureTitle("waveform-energy"), "波形与短时能量");
  assert.equal(jointNextFigureTitle("high-time"), "高时间分辨率频谱");
  for (const id of ["log-stft-before", "log-stft-after", "spectral-delta", "ltas", "erb", "triple"])
    assert.match(jointNextFigureTitle(id), /[\u4e00-\u9fff]/);
  assert.equal(jointNextFigureTitle("unknown"), "图形分析");
});
test("admission status must agree with the complete explicit check list", () => {
  const value = fixture(); value.admission.status = "admitted";
  assert.throws(() => parseJointNextReport(value), /准入状态矛盾/);
  value.admission.checks[0].pass = true;
  assert.equal(parseJointNextReport(value).admission.status, "admitted");
  value.admission.status = "rejected";
  assert.throws(() => parseJointNextReport(value), /准入状态矛盾/);
  value.admission.status = "pending";
  assert.equal(jointNextStatusLabel(parseJointNextReport(value).admission.status), "准入待定");
});
test("partial scenes, non-finite scores, missing identities and duplicate variants are rejected", () => {
  for (const mutate of [
    (v: ReturnType<typeof fixture>) => { v.case_count = 2; },
    (v: ReturnType<typeof fixture>) => { v.variants[1].mean_sisdr_delta_db = NaN; },
    (v: ReturnType<typeof fixture>) => { v.variants[1].all_speech_pass = 73; },
    (v: ReturnType<typeof fixture>) => { v.variants[1].all_speech_pass = 66; },
    (v: ReturnType<typeof fixture>) => { v.variants[1].id = "prior"; },
    (v: ReturnType<typeof fixture>) => { v.candidate_cli_sha256 = "old"; },
    (v: ReturnType<typeof fixture>) => { v.candidate_id = "missing"; },
    (v: ReturnType<typeof fixture>) => { v.admission.checks = []; },
  ]) { const value = fixture(); mutate(value); assert.throws(() => parseJointNextReport(value), /不完整/); }
});
test("figures and downloads require same published evidence directory and a content hash", () => {
  for (const url of ["https://external.invalid/a.png", "javascript:alert(1)", "/evaluation/old/a.png", "/evaluation/vcp-joint-next/../old/a.png", "/evaluation/vcp-joint-next/%2e%2e/a.png", "/evaluation/vcp-joint-next/\\evil.png"]) {
    const value = fixture(); value.figures[0].items[0].url = url;
    assert.throws(() => parseJointNextReport(value), /证据地址/);
  }
  const value = fixture(); value.artifacts[0].sha256 = "bad";
  assert.throws(() => parseJointNextReport(value), /不完整/);
});
