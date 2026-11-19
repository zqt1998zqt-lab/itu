import assert from "node:assert/strict";
import test from "node:test";
import { chainPresets, chainStages, chainEvidencePresets, chainFlags, chainCommand } from "../src/features/enhancement-chain-model.ts";
import { validateNative16Geometry } from "../src/features/enhancement-chain-native16.ts";

test("native16 is executable with EQ and keeps frozen evidence separate", () => {
  const preset = chainPresets.find((row) => row.id === "localvqe_cofilite_native16")!;
  assert.ok(preset);
  assert.equal(preset.selection.ns, "cofilite_native16");
  assert.equal(preset.selection.eq, "three_band");
  assert.equal(chainFlags(preset).filter((flag) => flag === "--eq-band").length, 3);
  assert.match(chainCommand(preset), /256/);
  assert.ok(chainStages.find((stage) => stage.id === "ns")!.options.some((row) => row.id === "cofilite_native16"));
  assert.ok(!chainEvidencePresets.some((row) => row.id === preset.id));
});

test("native16 geometry rejects legacy frame and adapter values", () => {
  const report = { sample_rate: 16000, frame_size: 256, aec_model_codec_delay_samples_16k: 256,
    aec_hop_adapter_delay_samples_16k: 0, aec_algorithm_delay_samples_16k: 256,
    ns_delay_samples: 256, ns_stream_chunk_samples: 256,
    ns_external_fallback_frames: 0, aec_localvqe_fallback_frames: 0 };
  assert.deepEqual(validateNative16Geometry(report), []);
  for (const [field, value] of Object.entries({ frame_size: 160, ns_delay_samples: 480,
    ns_stream_chunk_samples: 160, aec_hop_adapter_delay_samples_16k: 224, ns_external_fallback_frames: 1 })) {
    assert.ok(validateNative16Geometry({ ...report, [field]: value }).length > 0);
  }
});
