import assert from "node:assert/strict";
import { test } from "node:test";
import {
  AEC_CAPABILITY_SCHEMA,
  COMPUTER_PF_CPU_ROUTE_ID,
  LOCALVQE_ALGORITHM_DELAY_SAMPLES_16K,
  LOCALVQE_AEC_ROUTE_ID,
  LOCALVQE_HOP_ADAPTER_DELAY_SAMPLES_16K,
  LOCALVQE_NEURAL_DELAY_SAMPLES_16K,
  isCapabilityAecRoute,
  isComputerPfCpuRoute,
  isDspAecRoute,
  isLocalVqeRoute,
  parseAecCapabilities,
  resolveComputerPfCpuCapability,
  resolveDspCapability,
  resolveDspCapabilityFailure,
} from "../src/features/aec-route-capability.ts";

function realCapabilities() {
  return {
    schema: AEC_CAPABILITY_SCHEMA,
    cli_sha256: "75c9c2053bfd8dd5551122dac0e5838a586680c51ab1c07f4a61ce30b8d9daf1",
    backends: [
      {
        id: "webrtc", available: true, architecture: "webrtc_aec3", reason: "",
        execution_provider: "cpu", device_id: "host_cpu", model_id: "",
        model_sha256: "", frontend_sha256: "", provider_sha256: "", delay_samples: 0,
      },
      {
        id: "speex", available: true, architecture: "speex_mdf", reason: "",
        execution_provider: "cpu", device_id: "host_cpu", model_id: "",
        model_sha256: "", frontend_sha256: "", provider_sha256: "", delay_samples: 0,
      },
      {
        id: "speex_fused_cpu", available: true, architecture: "speex_mdf_gcc_phat_nlms", reason: "",
        execution_provider: "cpu", device_id: "host_cpu", model_id: "",
        model_sha256: "", frontend_sha256: "", provider_sha256: "", delay_samples: 0,
      },
      {
        id: "speex_fused_dsp",
        available: false,
        architecture: "mdf_complex_pf",
        reason: "speex_fused_dsp unavailable: HiFi SDK/provider is not linked; register a trained Complex_PF model with the HiFi adapter",
        execution_provider: "",
        device_id: "",
        model_id: "",
        model_sha256: "",
        frontend_sha256: "",
        provider_sha256: "",
        delay_samples: 0,
      },
      {
        id: "computer_pf_cpu",
        available: false,
        architecture: "mdf_complex_pf_cpu",
        reason: "computer_pf_cpu unavailable: locally trained Complex_PF model is not registered",
        execution_provider: "",
        device_id: "",
        model_id: "",
        model_sha256: "",
        frontend_sha256: "",
        provider_sha256: "",
        delay_samples: 0,
      },
      {
        id: "localvqe_aec200k",
        available: true,
        architecture: "localvqe_daf_neural_mask",
        reason: "",
        execution_provider: "cpu",
        device_id: "host_cpu",
        model_id: "localvqe_aec200k",
        model_sha256: "a".repeat(64),
        frontend_sha256: "b".repeat(64),
        provider_sha256: "c".repeat(64),
        delay_samples: 480,
        model_codec_delay_samples_16k: 256,
        linear_frontend_delay_samples_16k: 0,
      },
    ],
  };
}

function availableCapabilities() {
  const capabilities = realCapabilities();
  Object.assign(capabilities.backends[3], {
    available: true,
    reason: "",
    execution_provider: "hifi_dsp",
    device_id: "hifi0",
    model_id: "complex_pf_stream_v1",
    model_sha256: "b".repeat(64),
    frontend_sha256: "c".repeat(64),
    provider_sha256: "d".repeat(64),
    delay_samples: 320,
  });
  return capabilities;
}

function availableComputerPfCapabilities() {
  const capabilities = realCapabilities();
  Object.assign(capabilities.backends[4], {
    available: true,
    reason: "",
    execution_provider: "cpu",
    device_id: "host_cpu",
    model_id: "complex_pf_cpu_local_v1",
    model_sha256: "1".repeat(64),
    frontend_sha256: "2".repeat(64),
    provider_sha256: "3".repeat(64),
    delay_samples: 0,
  });
  return capabilities;
}

test("AEC capability parser accepts all six routes and preserves real unavailable reasons", () => {
  const document = parseAecCapabilities(realCapabilities());
  const resolution = resolveDspCapability(document);
  assert.equal(resolution.status, "unavailable");
  assert.match(resolution.reason, /HiFi SDK/);
  assert.equal(document.backends[2].architecture, "speex_mdf_gcc_phat_nlms");
  const cpu = resolveComputerPfCpuCapability(document);
  assert.equal(cpu.status, "unavailable");
  assert.match(cpu.reason, /locally trained Complex_PF model/);
  assert.equal(document.backends[4].architecture, "mdf_complex_pf_cpu");
  assert.equal(document.backends[5].architecture, "localvqe_daf_neural_mask");
});

test("capability route helpers generalize CPU Complex_PF while preserving the DSP helper", () => {
  assert.equal(isCapabilityAecRoute("speex_fused_dsp"), true);
  assert.equal(isCapabilityAecRoute(COMPUTER_PF_CPU_ROUTE_ID), true);
  assert.equal(isCapabilityAecRoute("speex_fused_cpu"), false);
  assert.equal(isDspAecRoute("speex_fused_dsp"), true);
  assert.equal(isDspAecRoute(COMPUTER_PF_CPU_ROUTE_ID), false);
  assert.equal(isComputerPfCpuRoute(COMPUTER_PF_CPU_ROUTE_ID), true);
  assert.equal(isCapabilityAecRoute(LOCALVQE_AEC_ROUTE_ID), true);
  assert.equal(isLocalVqeRoute(LOCALVQE_AEC_ROUTE_ID), true);
  assert.equal(isLocalVqeRoute(COMPUTER_PF_CPU_ROUTE_ID), false);
});

test("available DSP requires a complete HiFi provider, model, device, hash, and delay identity", () => {
  const document = parseAecCapabilities(availableCapabilities());
  assert.equal(resolveDspCapability(document).status, "available");
  const invalidCases: Array<[string, unknown]> = [
    ["execution_provider", "cpu"],
    ["device_id", ""],
    ["model_id", "../complex_pf.onnx"],
    ["model_sha256", ""],
    ["frontend_sha256", "not-a-hash"],
    ["provider_sha256", "e".repeat(63)],
    ["delay_samples", 212],
  ];
  for (const [field, value] of invalidCases) {
    const forged = availableCapabilities() as unknown as { backends: Array<Record<string, unknown>> };
    forged.backends[3][field] = value;
    assert.throws(() => parseAecCapabilities(forged), /execution_provider|device_id|model_id|SHA-256|delay_samples/, field);
  }
});

test("available CPU Complex_PF requires cpu/host_cpu and a complete local model identity", () => {
  const document = parseAecCapabilities(availableComputerPfCapabilities());
  assert.equal(resolveComputerPfCpuCapability(document).status, "available");
  const invalidCases: Array<[string, unknown]> = [
    ["execution_provider", "hifi_dsp"],
    ["device_id", "hifi0"],
    ["model_id", ""],
    ["model_sha256", ""],
    ["frontend_sha256", "not-a-hash"],
    ["provider_sha256", "4".repeat(63)],
    ["delay_samples", -1],
    ["delay_samples", 0.5],
  ];
  for (const [field, value] of invalidCases) {
    const forged = availableComputerPfCapabilities() as unknown as { backends: Array<Record<string, unknown>> };
    forged.backends[4][field] = value;
    assert.throws(() => parseAecCapabilities(forged), /execution_provider|device_id|model_id|SHA-256|delay_samples/, field);
  }
});

test("available LocalVQE requires the pinned CPU identity and exact 480-sample algorithm delay", () => {
  assert.equal(LOCALVQE_NEURAL_DELAY_SAMPLES_16K, 256);
  assert.equal(LOCALVQE_HOP_ADAPTER_DELAY_SAMPLES_16K, 224);
  assert.equal(
    LOCALVQE_ALGORITHM_DELAY_SAMPLES_16K,
    LOCALVQE_NEURAL_DELAY_SAMPLES_16K + LOCALVQE_HOP_ADAPTER_DELAY_SAMPLES_16K,
  );
  assert.equal(LOCALVQE_ALGORITHM_DELAY_SAMPLES_16K, 480);
  const document = parseAecCapabilities(realCapabilities());
  assert.equal(isLocalVqeRoute(LOCALVQE_AEC_ROUTE_ID), true);
  assert.equal(document.backends[5].delay_samples, 480);
  const invalidCases: Array<[string, unknown]> = [
    ["execution_provider", "hifi_dsp"],
    ["device_id", "hifi0"],
    ["model_id", ""],
    ["model_sha256", ""],
    ["frontend_sha256", "not-a-hash"],
    ["provider_sha256", "d".repeat(63)],
    ["delay_samples", 512],
    ["delay_samples", 640],
    ["delay_samples", 736],
    ["delay_samples", 479.5],
    ["model_codec_delay_samples_16k", 0],
    ["linear_frontend_delay_samples_16k", 256],
  ];
  for (const [field, value] of invalidCases) {
    const forged = realCapabilities() as unknown as { backends: Array<Record<string, unknown>> };
    forged.backends[5][field] = value;
    assert.throws(
      () => parseAecCapabilities(forged),
      /execution_provider|device_id|model_id|SHA-256|delay_samples/,
      field,
    );
  }
});

test("AEC capability parser rejects wrong schemas, unknown backends, and missing availability", () => {
  assert.throws(() => parseAecCapabilities({ ...realCapabilities(), schema: "v2" }), /schema/);

  const unknown = realCapabilities() as unknown as { backends: Array<Record<string, unknown>> };
  unknown.backends.push({ id: "unknown_backend", available: true, architecture: "unknown", reason: "" });
  assert.throws(() => parseAecCapabilities(unknown), /未知/);

  const missingBoolean = realCapabilities() as unknown as { backends: Array<Record<string, unknown>> };
  delete missingBoolean.backends[3].available;
  assert.throws(() => parseAecCapabilities(missingBoolean), /available 必须是布尔值/);
});

test("a CPU architecture cannot masquerade as the DSP route", () => {
  const forged = realCapabilities();
  forged.backends[3].architecture = "speex_mdf_nlms_fused_cpu";
  assert.throws(() => parseAecCapabilities(forged), /mdf_complex_pf/);
});

test("the CPU report name cannot replace the real CPU capability architecture", () => {
  const forged = realCapabilities();
  forged.backends[2].architecture = "speex_mdf_nlms_fused_cpu";
  assert.throws(() => parseAecCapabilities(forged), /speex_mdf_gcc_phat_nlms/);
});

test("missing DSP capability and request failure are never reported as available", () => {
  const missing = realCapabilities();
  missing.backends = missing.backends.filter((entry) => entry.id !== "speex_fused_dsp");
  const missingResolution = resolveDspCapability(parseAecCapabilities(missing));
  assert.equal(missingResolution.status, "unavailable");
  assert.match(missingResolution.reason, /缺少 speex_fused_dsp/);

  const failed = resolveDspCapabilityFailure(new Error("network down"));
  assert.equal(failed.status, "error");
  assert.match(failed.reason, /network down/);
});
