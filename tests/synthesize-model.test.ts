import assert from "node:assert/strict";
import { test } from "node:test";
import { chainPresets, changeStage } from "../src/features/enhancement-chain-model.ts";
import { buildSynthesisOptions, synthesisConfigurationKey } from "../src/features/synthesize-model.ts";

const joint = chainPresets.find((item) => item.id === "vcp_joint")!;
test("submitted joint request owns a full immutable route and calibration snapshot", () => {
  const editable = { ...joint, selection: { ...joint.selection }, extra_flags: ["--no-aec"] };
  const options = buildSynthesisOptions(editable, "vcp", "16000", "beamform", "1.4403, 0");
  assert.equal(options.profile, "vcp_joint");
  assert.deepEqual(options.beam_delays_samples, [1.4403, 0]);
  assert.deepEqual(options.chain, { profile: "vcp_joint", selection: joint.selection });
  editable.selection.eq = "three_band";
  assert.equal(options.chain?.selection.eq, "off");
  assert.ok(!JSON.stringify(options).includes("--no-aec"));
});
test("a later route selection has a different result identity", () => {
  const key = synthesisConfigurationKey(joint, "vcp");
  const restoredKey = synthesisConfigurationKey({ profile: joint.profile, selection: { ...joint.selection } }, "vcp");
  assert.equal(key, restoredKey);
  assert.notEqual(key, synthesisConfigurationKey(changeStage(joint, "eq", "three_band"), "vcp"));
  assert.notEqual(key, synthesisConfigurationKey(undefined, "vcp"));
});
test("unsupported joint combinations and missing calibration fail before sending audio", () => {
  assert.throws(() => buildSynthesisOptions(changeStage(joint, "ns", "webrtc"), "vcp", "", "channel0", ""), /Super-Gaussian/);
  assert.throws(() => buildSynthesisOptions(joint, "vcp", "", "beamform", ""), /真实标定/);
  assert.throws(() => buildSynthesisOptions(joint, "vcp", "", "beamform", "1,NaN"), /真实标定/);
  assert.throws(() => buildSynthesisOptions(joint, "vcp", "8000", "channel0", ""), /16 \/ 24/);
  assert.deepEqual(buildSynthesisOptions(undefined, "vcp", "", "channel0", ""), { profile: "vcp", microphone_mode: "channel0" });
});
