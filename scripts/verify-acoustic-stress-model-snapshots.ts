import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";

import {
  FIVE_SIGNAL_MODES,
  type FiveSignalMode,
} from "../src/features/five-signal-flow-model.ts";
import {
  SCOPED_INPUT_SIGNAL_ROLES,
  type ScopedItem,
} from "../src/features/scoped-acoustic-model.ts";
import {
  STRESS_CATEGORY_LABELS,
  STRESS_SCOPES,
  readStressBatch,
  readStressPreview,
  stressSampleRate,
  type StressBatch,
  type StressDiagnostics,
  type StressScope,
} from "../src/features/acoustic-stress-model.ts";

type ContractRow = {
  id: string;
  sheet: FiveSignalMode;
  source_cell: string;
  test_name: string;
  family: string;
  direction: string;
  unit: string;
  lower_limit: number | string | null;
  upper_limit: number | string | null;
};

const projectRoot = new URL("../", import.meta.url);
const contract = JSON.parse(
  readFileSync(
    new URL("cpp/evaluation/nv200-test-contract.json", projectRoot),
    "utf8",
  ),
) as { items: ContractRow[] };
const waveBase64 = Buffer.alloc(720 * 2).toString("base64");
const melBase64 = Buffer.alloc(40 * 240 * 2).toString("base64");
const metricKeys = [
  "primary",
  "tx_residual_reduction_db",
  "convergence_db",
  "cleanref_projection_gain_db",
  "cleanref_correlation",
  "tx_level_dbfs",
  "mic_level_dbfs",
  "far_level_dbfs",
  "near_level_dbfs",
].sort();

function selectedContractRows(
  scope: StressScope,
  mode: FiveSignalMode,
): ContractRow[] {
  const family = scope === "echo" ? "tclw" : "double_talk_classification";
  return contract.items.filter(
    (row) => row.sheet === mode && row.family === family,
  );
}

function scopedRows(rows: ContractRow[]): ScopedItem[] {
  return rows.map((row) => ({
    id: row.id,
    source_cell: row.source_cell,
    test_name: row.test_name,
    family: row.family,
    direction: row.direction,
    unit: row.unit,
    lower_limit: row.lower_limit,
    upper_limit: row.upper_limit,
  }));
}

function visualTrack(
  sourceHash: string,
  mode: FiveSignalMode,
  role: (typeof SCOPED_INPUT_SIGNAL_ROLES)[number],
) {
  return {
    source_sha256: sourceHash,
    duration_s: 12,
    sample_rate: stressSampleRate(mode),
    channels: role === "microphone" ? 2 : 1,
    peak: 0.5,
    wave_scale: 1,
    wave_min_i16: waveBase64,
    wave_max_i16: waveBase64,
    mel_i16: melBase64,
    mel_ceiling_db: -20,
  };
}

function latestSnapshot(directory: URL): string {
  const latest = readdirSync(directory)
    .filter((name) => /^\d{8}\.json$/.test(name))
    .sort()
    .at(-1);
  assert.ok(latest, `no snapshot JSON in ${fileURLToPath(directory)}`);
  return latest;
}

function readBatchSnapshot(
  snapshot: URL,
  scope: StressScope,
  mode: FiveSignalMode,
  expectedDatasetId: string,
): StressBatch {
  const value = JSON.parse(readFileSync(snapshot, "utf8")) as Record<
    string,
    unknown
  >;
  return readStressBatch(value, {
    scope,
    mode,
    datasetId: expectedDatasetId,
    algorithmClusterId: String(value.algorithm_cluster_id),
    contracts: scopedRows(selectedContractRows(scope, mode)),
  });
}

function assertMetrics(diagnostics: StressDiagnostics): void {
  assert.ok(diagnostics.segments.length > 0);
  for (const segment of diagnostics.segments) {
    assert.deepEqual(
      Object.keys(segment.metrics).sort(),
      metricKeys,
      `${diagnostics.case_id}/${segment.label} metric contract`,
    );
  }
}

function verifyRealPreview(): void {
  const root = new URL(
    "build/acoustic-qualification/acoustic-stress-datasets/0563b621e16a1a7702dd7994683ddda2/",
    projectRoot,
  );
  const manifestBytes = readFileSync(new URL("manifest.json", root));
  const assetsBytes = readFileSync(new URL("assets.json", root));
  const manifest = JSON.parse(manifestBytes.toString("utf8")) as {
    dataset_id: string;
    dataset_name: string;
    test_scope: "echo";
    mode: "WB-wired";
    scene: "baseline";
    contract_item_count: number;
    preview_condition_group_count: 100;
    cases: Array<{
      case_id: string;
      conditions: Record<string, unknown>;
      contract_item_ids: [string];
      families: string[];
      signals: Record<string, string>;
      signals_sha256: Record<string, string>;
      provenance: Record<string, unknown>;
    }>;
  };
  const rows = scopedRows(selectedContractRows("echo", "WB-wired"));
  const cases = manifest.cases.map((entry) => ({
    case_id: entry.case_id,
    conditions: entry.conditions,
    contract_item_ids: entry.contract_item_ids,
    families: entry.families,
    signals: [...Object.keys(entry.signals)].sort(),
    signals_sha256: entry.signals_sha256,
    provenance: entry.provenance,
  }));
  const visualCases = Object.fromEntries(
    cases.map((entry) => [
      entry.case_id,
      Object.fromEntries(
        SCOPED_INPUT_SIGNAL_ROLES.map((role) => [
          role,
          visualTrack(entry.signals_sha256[role], "WB-wired", role),
        ]),
      ),
    ]),
  );
  const preview = readStressPreview(
    {
      dataset_id: manifest.dataset_id,
      dataset_name: manifest.dataset_name,
      dataset_manifest_sha256: createHash("sha256")
        .update(manifestBytes)
        .digest("hex"),
      dataset_assets_sha256: createHash("sha256")
        .update(assetsBytes)
        .digest("hex"),
      test_scope: manifest.test_scope,
      mode: manifest.mode,
      scene: manifest.scene,
      schema_version: "acoustic-stress-preview-v1",
      dataset_kind: "acoustic_stress_preview",
      contract_item_count: manifest.contract_item_count,
      preview_condition_group_count: manifest.preview_condition_group_count,
      observation_count: 100,
      category_counts: {
        steady_echo: 20,
        convergence: 20,
        delay_change: 20,
        path_change: 20,
        noisy_echo: 20,
      },
      category_labels: STRESS_CATEGORY_LABELS.echo,
      contracts: rows,
      cases,
      visuals: {
        schema: "scoped-acoustic-visual-v1",
        waveform_bins: 720,
        mel_bands: 40,
        mel_frames: 240,
        mel_fft_size: 512,
        mel_hop_ms: 10,
        mel_dynamic_range_db: 80,
        mel_db_offset_db: -200,
        mel_db_step_db: 0.1,
        cases: visualCases,
      },
      standard_compliance: false,
      formal_test_status: "not_measured",
    },
    {
      scope: "echo",
      mode: "WB-wired",
      datasetId: manifest.dataset_id,
      contracts: rows,
    },
  );
  assert.equal(preview.cases.length, 100);
  assert.equal(Object.keys(preview.visuals.cases).length, 100);
  assert.equal(
    preview.cases[0].case_id,
    "echo-WB-wired-steady_echo-01-stress-v1",
  );
  assert.equal(
    preview.cases[0].signals_sha256.microphone,
    "174cebb8239ba940c83943e66edd1bb1278535669760ab8740b6f44cbdc66837",
  );
  console.log(
    "PASS preview echo/WB-wired cases=100 visuals=100 first=echo-WB-wired-steady_echo-01-stress-v1",
  );
}

function verifyLegacyBatch(
  batchId: string,
  scope: StressScope,
  mode: FiveSignalMode,
  datasetId: string,
): void {
  const snapshotDirectory = new URL(
    `build/acoustic-qualification/scene-condition-batches/${batchId}/snapshots/`,
    projectRoot,
  );
  const latest = latestSnapshot(snapshotDirectory);
  assert.equal(latest, "00000203.json");
  const batch = readBatchSnapshot(
    new URL(latest, snapshotDirectory),
    scope,
    mode,
    datasetId,
  );
  assert.equal(batch.batch_id, batchId);
  assert.equal(batch.status, "completed");
  assert.equal(batch.cases.length, 100);
  assert.equal(batch.items.length, 100);
  assert.deepEqual(batch.summary, {
    total: 100,
    computed: 100,
    pass: 0,
    fail: 0,
    review_required: 100,
    not_measured: 0,
    invalid: 0,
  });
  for (const caseRecord of batch.cases) {
    assert.equal(caseRecord.status, "completed");
    const diagnostics = caseRecord.stress_diagnostics;
    assert.ok(diagnostics);
    assert.equal(
      diagnostics.schema_version,
      "acoustic-stress-diagnostics-v1",
    );
    assert.equal(Object.hasOwn(diagnostics, "activity_gate"), false);
    assertMetrics(diagnostics);
    for (const segment of diagnostics.segments) {
      assert.equal(Object.hasOwn(segment, "activity"), false);
    }
  }
  console.log(
    `PASS legacy-v1 ${batchId} ${scope}/${mode} cases=100 computed=100`,
  );
}

function verifyRepresentativeSnapshots(): void {
  const root = new URL(
    "build/stress-vcp-representatives-d7s26mao/scene-condition-batches/",
    projectRoot,
  );
  const batchIds = readdirSync(root, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .sort();
  const observed = new Set<string>();
  let completedCases = 0;
  for (const batchId of batchIds) {
    const snapshotDirectory = new URL(`${batchId}/snapshots/`, root);
    const latest = latestSnapshot(snapshotDirectory);
    const snapshot = new URL(latest, snapshotDirectory);
    const raw = JSON.parse(readFileSync(snapshot, "utf8")) as Record<
      string,
      unknown
    >;
    const scope = raw.test_scope as StressScope;
    const mode = raw.selected_mode as FiveSignalMode;
    assert.ok(STRESS_SCOPES.includes(scope));
    assert.ok(FIVE_SIGNAL_MODES.includes(mode));
    const batch = readBatchSnapshot(
      snapshot,
      scope,
      mode,
      String(raw.dataset_id),
    );
    assert.equal(batch.summary.computed, 5);
    assert.equal(batch.summary.review_required, 5);
    assert.equal(
      batch.cases.filter((entry) => entry.status === "completed").length,
      5,
    );
    for (const caseRecord of batch.cases) {
      if (caseRecord.status !== "completed") continue;
      completedCases += 1;
      const diagnostics = caseRecord.stress_diagnostics;
      assert.ok(diagnostics);
      assert.equal(
        diagnostics.schema_version,
        "acoustic-stress-diagnostics-v2",
      );
      if (diagnostics.schema_version !== "acoustic-stress-diagnostics-v2") {
        throw Error(`${batchId}/${caseRecord.case_id} is not v2`);
      }
      assert.deepEqual(diagnostics.activity_gate, {
        frame_ms: 20,
        hop_ms: 10,
        rms_threshold: 0.00001,
        min_active_s: 0.05,
        min_coverage: 0.05,
      });
      assertMetrics(diagnostics);
      for (const segment of diagnostics.segments) {
        assert.equal(typeof segment.activity.valid, "boolean");
        assert.ok(segment.activity.reason.length > 0);
      }
    }
    if (scope === "echo") {
      const convergence = batch.cases.find(
        (entry) =>
          entry.status === "completed"
          && entry.provenance.workload_category === "convergence",
      );
      assert.ok(convergence?.stress_diagnostics);
      const quiet = convergence.stress_diagnostics.segments.find(
        (segment) => segment.label === "quiet",
      );
      assert.ok(quiet);
      assert.equal(quiet.mode, "not_applicable");
      assert.equal(quiet.status, "not_applicable");
    }
    observed.add(`${scope}/${mode}`);
  }
  assert.equal(observed.size, 16);
  assert.equal(completedCases, 80);
  console.log(
    `PASS representative-v2 profiles=${observed.size} completed_cases=${completedCases}`,
  );
}

verifyRealPreview();
verifyLegacyBatch(
  "fa84fcfd56ca427083c474cc261f1d1a",
  "echo",
  "WB-wired",
  "0563b621e16a1a7702dd7994683ddda2",
);
verifyLegacyBatch(
  "d89f8b9447574afc89d619ab2057684e",
  "double_talk",
  "WB-wired",
  "01d5e9176ea5abbbead40185fc9b4546",
);
verifyRepresentativeSnapshots();
console.log("PASS acoustic-stress real snapshot verification");
