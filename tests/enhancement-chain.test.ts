import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import {
  chainFlags, chainPresets, chainEvidencePresets, chainStages, changeStage, matchingEvidence,
  parseChainReport, restoreConfiguration,
} from "../src/features/enhancement-chain-model.ts";

const report = () => parseChainReport(JSON.parse(readFileSync(new URL("../public/evaluation/enhancement-chain/report.json", import.meta.url), "utf8")));
const serial = chainPresets.find((item) => item.id === "all_serial")!;

test("every published route matches the exact selectable CLI flags and a complete paired matrix", () => {
  const actual = report();
  assert.equal(actual.case_count, 72);
  assert.equal(actual.cli_runs, 72 * chainEvidencePresets.length);
  for (const preset of chainEvidencePresets) {
    const evidence = matchingEvidence(preset, actual);
    assert.ok(evidence, preset.id);
    assert.deepEqual(evidence.flags, chainFlags(preset));
    assert.equal(evidence.caseGroups.length, 2);
    assert.deepEqual(new Set(evidence.caseGroups.map((group) => group.id)), new Set(["pass", "fail"]));
    for (const stage of chainStages) assert.ok(stage.options.some((option) => option.id === preset.selection[stage.id]));
  }
  const recommended = chainPresets.find((item) => item.recommended);
  assert.equal(recommended?.id, "super_gaussian_optimal");
  assert.equal(matchingEvidence(recommended!, actual)?.pass, 60);
});

test("changing a real algorithm invalidates the old scores and case figures", () => {
  const actual = report();
  assert.equal(matchingEvidence(serial, actual)?.id, "all_serial");
  const draft = changeStage(serial, "ns", "super_gaussian");
  assert.equal(matchingEvidence(draft, actual), undefined);
  assert.ok(chainFlags(draft).includes("super_gaussian"));
  assert.equal(matchingEvidence(changeStage(draft, "ns", "webrtc"), actual)?.id, "all_serial");
});

test("same route name with different executor flags cannot borrow frozen evidence", () => {
  const actual = report();
  actual.variants.find((item) => item.id === "all_serial")!.flags.push("--no-agc");
  assert.equal(matchingEvidence(serial, actual), undefined);
});

test("invalid browser drafts and injected cached flags never alter the executor template", () => {
  assert.equal(restoreConfiguration("{malformed").selection.eq, serial.selection.eq);
  assert.equal(restoreConfiguration(JSON.stringify({ profile: "vcp", selection: { ...serial.selection, ns: "missing-backend" } })).selection.ns, serial.selection.ns);
  const restored = restoreConfiguration(JSON.stringify({ ...serial, extra_flags: ["--no-agc"] }));
  assert.ok(!chainFlags(restored).includes("--no-agc"));
  assert.equal(matchingEvidence(restored, report())?.id, "all_serial");
});

test("editing the DAS control creates a VCP draft with no hidden control-only flags", () => {
  const draft = changeStage(chainPresets[0], "ns", "webrtc");
  assert.equal(draft.profile, "vcp");
  assert.ok(!chainFlags(draft).includes("--no-low-cut"));
  assert.equal(matchingEvidence(draft, report()), undefined);
});

test("partial pilot reports and missing negative controls are rejected", () => {
  const partial = report();
  partial.case_count = 2;
  assert.throws(() => parseChainReport(partial), /不完整/);
  const vacuous = report();
  vacuous.negative_control.rejected = 0;
  assert.throws(() => parseChainReport(vacuous), /负对照/);
});

test("joint route is executable without borrowing the older matrix", () => {
  const joint = chainPresets.find((item) => item.id === "vcp_joint")!;
  assert.equal(joint.profile, "vcp_joint");
  assert.equal(chainFlags(joint)[1], "vcp_joint");
  assert.equal(matchingEvidence(joint, report()), undefined);
  const restored = restoreConfiguration(JSON.stringify({ ...joint, extra_flags: ["--no-aec"] }));
  assert.equal(restored.profile, "vcp_joint");
  assert.deepEqual(restored.selection, joint.selection);
  assert.ok(!chainFlags(restored).includes("--no-aec"));
  assert.equal(changeStage(restored, "eq", "three_band").profile, "vcp_joint");
  assert.equal(chainFlags(restored).at(chainFlags(restored).lastIndexOf("--tx-drc-threshold") + 1), "-6");
});
