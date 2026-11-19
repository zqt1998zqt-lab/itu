import assert from "node:assert/strict";
import { test } from "node:test";
import { chainFlags, chainPresets, changeStage } from "../src/features/enhancement-chain-model.ts";
import { buildSynthesisOptions, synthesisConfigurationKey } from "../src/features/synthesize-model.ts";

const joint = chainPresets.find((item) => item.id === "vcp_joint")!;
const dsp = chainPresets.find((item) => item.id === "speex_fused_dsp")!;
const localvqe = chainPresets.find((item) => item.id === "localvqe_aec200k")!;
const serverNsPresets = [
  chainPresets.find((item) => item.id === "localvqe_cofilite")!,
  chainPresets.find((item) => item.id === "localvqe_dpdfnet2")!,
  chainPresets.find((item) => item.id === "localvqe_hybrid_v2_pitch")!,
  chainPresets.find((item) => item.id === "vcp_aec3_dpdfnet2_48k_hr")!,
];
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
test("server-executed NS routes have distinct keys while legacy flag keys stay unchanged", () => {
  const serverNsIds = new Set(serverNsPresets.map((preset) => preset.selection.ns));
  for (const preset of chainPresets.filter((item) => !serverNsIds.has(item.selection.ns))) {
    assert.equal(
      synthesisConfigurationKey(preset, "vcp"),
      JSON.stringify(chainFlags(preset)),
      preset.id,
    );
  }

  const keys = new Set([
    synthesisConfigurationKey(localvqe, "vcp"),
    ...serverNsPresets.map((preset) => synthesisConfigurationKey(preset, "vcp")),
  ]);
  assert.equal(keys.size, serverNsPresets.length + 1);
  for (const preset of serverNsPresets) {
    assert.equal(
      preset.selection.aec,
      preset.id === "vcp_aec3_dpdfnet2_48k_hr" ? "aec3" : "localvqe_aec200k",
    );
    assert.equal(
      synthesisConfigurationKey(preset, "vcp"),
      JSON.stringify({ flags: chainFlags(preset), ns: preset.selection.ns }),
    );
  }

  const cofilite = serverNsPresets[0];
  const cofiliteKey = synthesisConfigurationKey(cofilite, "vcp");
  assert.notEqual(cofiliteKey, synthesisConfigurationKey({
    ...cofilite,
    selection: { ...cofilite.selection, aec: "speex_fused_cpu" },
  }, "vcp"));
  assert.notEqual(cofiliteKey, synthesisConfigurationKey({
    ...cofilite,
    selection: { ...cofilite.selection, bf: "mvdr" },
  }, "vcp"));
});
test("unsupported joint combinations and missing calibration fail before sending audio", () => {
  assert.throws(() => buildSynthesisOptions(changeStage(joint, "ns", "webrtc"), "vcp", "", "channel0", ""), /Super-Gaussian/);
  const quality48 = chainPresets.find((item) => item.id === "vcp_aec3_dpdfnet2_48k_hr")!;
  assert.throws(
    () => buildSynthesisOptions(changeStage(quality48, "aec", "speex"), "vcp", "48000", "channel0", ""),
    /AEC3/,
  );
  assert.throws(
    () => buildSynthesisOptions(quality48, "vcp", "16000", "channel0", ""),
    /48 kHz/,
  );
  assert.throws(() => buildSynthesisOptions(joint, "vcp", "", "beamform", ""), /真实标定/);
  assert.throws(() => buildSynthesisOptions(joint, "vcp", "", "beamform", "1,NaN"), /真实标定/);
  assert.throws(() => buildSynthesisOptions(joint, "vcp", "8000", "channel0", ""), /16 \/ 24/);
  const cofilite = chainPresets.find((item) => item.id === "localvqe_cofilite")!;
  assert.throws(
    () => buildSynthesisOptions(cofilite, "vcp", "48000", "channel0", ""),
    /16000 Hz/,
  );
  assert.throws(
    () => buildSynthesisOptions(dsp, "vcp", "48000", "channel0", ""),
    /16000 Hz/,
  );
  assert.equal(
    buildSynthesisOptions(cofilite, "vcp", "16000", "channel0", "").sample_rate,
    16000,
  );
  assert.deepEqual(buildSynthesisOptions(undefined, "vcp", "", "channel0", ""), { profile: "vcp", microphone_mode: "channel0" });
});
test("DSP requests are submitted as catalog routes while invalid AEC backends are rejected", () => {
  const options = buildSynthesisOptions(dsp, "vcp", "", "channel0", "");
  assert.equal(options.profile, "vcp");
  assert.equal(options.chain?.selection.aec, "speex_fused_dsp");
  assert.throws(() => buildSynthesisOptions({
    profile: "vcp",
    selection: { ...dsp.selection, aec: "speex_fused_invalid" },
  }, "vcp", "", "channel0", ""), /不支持的选项/);
});
