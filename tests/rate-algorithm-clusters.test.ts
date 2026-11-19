import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import test from "node:test";
import {
  algorithmClusterById,
  algorithmClusterGroupsForMode,
  algorithmClusterRateSummary,
  algorithmClustersForMode,
  algorithmClustersForRate,
  defaultAlgorithmClusterForMode,
  initialAlgorithmClustersByRate,
  isAlgorithmClusterSupportedForMode,
  NB8_CLUSTER_ID,
  NATIVE16_CLUSTER_ID,
  QUALITY_48K_HR_CLUSTER_ID,
  resolveAlgorithmClusterForMode,
  sampleRateForBusinessMode,
  supportsMultiRateFullBatch,
} from "../src/features/rate-algorithm-clusters.ts";

test("NB 8 kHz modes expose only the native 8 kHz Speex+DPDFNet2 cluster", () => {
  for (const mode of ["NB-wired", "NB-wireless", "NB-bt"]) {
    assert.equal(sampleRateForBusinessMode(mode), 8_000);
    const clusters = algorithmClustersForMode(mode, { vcpOnly: true });
    assert.deepEqual(
      clusters.map((item) => item.id),
      [NB8_CLUSTER_ID],
    );
    assert.equal(defaultAlgorithmClusterForMode(mode), NB8_CLUSTER_ID);
    assert.equal(isAlgorithmClusterSupportedForMode(NB8_CLUSTER_ID, mode), true);
    assert.equal(isAlgorithmClusterSupportedForMode("vcp_unified", mode), false);
    assert.equal(isAlgorithmClusterSupportedForMode(NATIVE16_CLUSTER_ID, mode), false);
    assert.equal(isAlgorithmClusterSupportedForMode(QUALITY_48K_HR_CLUSTER_ID, mode), false);
    assert.equal(resolveAlgorithmClusterForMode("vcp_unified", mode), NB8_CLUSTER_ID);
  }
  const nb8 = algorithmClusterById(NB8_CLUSTER_ID)!;
  assert.ok(nb8);
  assert.equal(nb8.stages.length, 7);
  assert.equal(algorithmClusterRateSummary(NB8_CLUSTER_ID), "仅 8 kHz");
  assert.equal(supportsMultiRateFullBatch(NB8_CLUSTER_ID), false);
});

test("WB 16 kHz modes expose 16 kHz clusters and exclude 8 kHz and 48 kHz-only clusters", () => {
  for (const mode of ["WB-wired", "WB-wireless", "WB-bt"]) {
    assert.equal(sampleRateForBusinessMode(mode), 16_000);
    const ids = algorithmClustersForMode(mode, { vcpOnly: true }).map((item) => item.id);
    assert.ok(ids.includes(NATIVE16_CLUSTER_ID));
    assert.ok(ids.includes("localvqe_cofilite"));
    assert.ok(ids.includes("localvqe_dpdfnet2"));
    assert.ok(ids.includes("localvqe_hybrid_v2_pitch"));
    assert.ok(ids.includes("localvqe_aec200k"));
    assert.ok(ids.includes("speex_fused_dsp"));
    assert.ok(ids.includes("computer_pf_cpu"));
    assert.ok(ids.includes("vcp_unified"));
    assert.ok(ids.includes("aec3_serial"));
    assert.ok(!ids.includes(NB8_CLUSTER_ID));
    assert.ok(!ids.includes(QUALITY_48K_HR_CLUSTER_ID));
    assert.equal(defaultAlgorithmClusterForMode(mode), "vcp_unified");
    assert.equal(resolveAlgorithmClusterForMode(QUALITY_48K_HR_CLUSTER_ID, mode), "vcp_unified");
    assert.equal(resolveAlgorithmClusterForMode(NATIVE16_CLUSTER_ID, mode), NATIVE16_CLUSTER_ID);
  }
  const groups = algorithmClusterGroupsForMode("WB-wired", { vcpOnly: true });
  assert.equal(groups.length, 2);
  assert.equal(groups[0].clusters[0].id, NATIVE16_CLUSTER_ID);
});

test("SWB 48 kHz modes expose 48 kHz clusters and exclude 8 kHz and 16 kHz-only clusters", () => {
  for (const mode of ["SWB-wired", "SWB-wireless"]) {
    assert.equal(sampleRateForBusinessMode(mode), 48_000);
    const ids = algorithmClustersForMode(mode, { vcpOnly: true }).map((item) => item.id);
    assert.ok(ids.includes(QUALITY_48K_HR_CLUSTER_ID));
    assert.ok(ids.includes("aec3_serial"));
    assert.ok(ids.includes("vcp_unified"));
    assert.ok(ids.includes("vcp_default"));
    for (const forbidden of [
      NB8_CLUSTER_ID,
      NATIVE16_CLUSTER_ID,
      "localvqe_cofilite",
      "localvqe_dpdfnet2",
      "localvqe_hybrid_v2_pitch",
      "localvqe_aec200k",
      "speex_fused_dsp",
      "computer_pf_cpu",
    ]) {
      assert.ok(!ids.includes(forbidden), `${mode} must exclude ${forbidden}`);
    }
    assert.equal(defaultAlgorithmClusterForMode(mode), QUALITY_48K_HR_CLUSTER_ID);
    assert.equal(
      resolveAlgorithmClusterForMode(NATIVE16_CLUSTER_ID, mode),
      QUALITY_48K_HR_CLUSTER_ID,
    );
    assert.equal(resolveAlgorithmClusterForMode("aec3_serial", mode), "aec3_serial");
  }
  assert.equal(supportsMultiRateFullBatch(QUALITY_48K_HR_CLUSTER_ID), false);
  assert.equal(supportsMultiRateFullBatch(NATIVE16_CLUSTER_ID), false);
  assert.equal(supportsMultiRateFullBatch("vcp_unified"), true);
  assert.equal(supportsMultiRateFullBatch("aec3_serial"), true);
});

test("initial per-rate defaults are distinct and valid for their respective sample rates", () => {
  const defaults = initialAlgorithmClustersByRate();
  assert.deepEqual(defaults, {
    8000: NB8_CLUSTER_ID,
    16000: "vcp_unified",
    48000: QUALITY_48K_HR_CLUSTER_ID,
  });
  assert.equal(new Set(Object.values(defaults)).size, 3);
});

test("frontend selectable clusters stay in exact parity with server/rate_paths.py", () => {
  const raw = execFileSync(
    "/home/zhangqingtian/.venv/bin/python",
    [
      "-c",
      "import json; from server.rate_paths import selectable_clusters_by_rate; print(json.dumps(selectable_clusters_by_rate(include_legacy=True)))",
    ],
    { encoding: "utf-8" },
  );
  const backend = JSON.parse(raw) as Record<string, string[]>;
  for (const rate of [8_000, 16_000, 24_000, 48_000] as const) {
    const frontendIds = algorithmClustersForRate(rate, { vcpOnly: true }).map((item) => item.id);
    assert.deepEqual(
      frontendIds,
      backend[String(rate)],
      `frontend and backend cluster lists differ at ${rate} Hz`,
    );
  }
});
