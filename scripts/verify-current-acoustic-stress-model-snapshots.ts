import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

import {
  FIVE_SIGNAL_MODES,
  type FiveSignalMode,
} from "../src/features/five-signal-flow-model.ts";
import {
  SCOPED_INPUT_SIGNAL_ROLES,
  type ScopedItem,
} from "../src/features/scoped-acoustic-model.ts";
import {
  STRESS_CATEGORIES,
  STRESS_CATEGORY_LABELS,
  readStressBatch,
  readStressPreview,
  stressSampleRate,
  type StressBatch,
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

type CurrentPreviewRecord = {
  dataset_id: string;
  mode: FiveSignalMode;
  manifest_path: string;
  manifest_sha256: string;
  assets_path: string;
  assets_sha256: string;
};

type HistoricalDatasetRecord = CurrentPreviewRecord & {
  expected_first_case_microphone_sha256: string | null;
};

type HistoricalSnapshotRecord = {
  batch_id: string;
  scope: StressScope;
  mode: FiveSignalMode;
  dataset_id: string;
  snapshot_path: string;
  snapshot_sha256: string;
  expected_cases: number;
  expected_computed: number;
  expected_summary: {
    total: 100;
    computed: 100;
    pass: 0;
    fail: 0;
    review_required: 100;
    not_measured: 0;
    invalid: 0;
  };
  semantics: string;
};

type HistoricalRepresentativeSnapshotRecord = {
  batch_id: string;
  scope: StressScope;
  mode: FiveSignalMode;
  dataset_id: string;
  snapshot_path: string;
  snapshot_sha256: string;
};

const FIXTURE_SHA256 =
  "d94a7b3721e1832fb61ab5bb91bbb16c2a42b9cb076e39a16727ccc2d212af61";
const projectRoot = new URL("../", import.meta.url);
let metadataReads = 0;

function readMetadataJson(source: URL, label: string): Buffer {
  assert.equal(
    source.pathname.endsWith(".json"),
    true,
    `${label} must be metadata JSON, not a WAV or other asset`,
  );
  metadataReads += 1;
  return readFileSync(source);
}

const fixtureBytes = readMetadataJson(
  new URL(
    "benchmarks/fixtures/acoustic-stress-model-schema-v1/metadata.json",
    projectRoot,
  ),
  "fixture metadata",
);
assert.equal(
  createHash("sha256").update(fixtureBytes).digest("hex"),
  FIXTURE_SHA256,
  "metadata fixture hash changed",
);
const fixture = JSON.parse(fixtureBytes.toString("utf8")) as {
  current_revision: string;
  metadata_only: boolean;
  wav_files_required: boolean;
  current_previews: Record<StressScope, CurrentPreviewRecord>;
  historical_datasets: Record<StressScope, HistoricalDatasetRecord>;
  historical_cached_snapshots: HistoricalSnapshotRecord[];
  historical_representative_snapshots: HistoricalRepresentativeSnapshotRecord[];
  evaluation_protocol: {
    version: string;
    historical_current_evidence_equivalent: boolean;
  };
};
assert.equal(fixture.metadata_only, true);
assert.equal(fixture.wav_files_required, false);
assert.equal(fixture.current_revision, "measurement-20261117-r2");
assert.equal(
  fixture.evaluation_protocol.historical_current_evidence_equivalent,
  false,
);

const contract = JSON.parse(
  readMetadataJson(
    new URL("cpp/evaluation/nv200-test-contract.json", projectRoot),
    "test contract",
  ).toString("utf8"),
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

function sha256(value: Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}

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

function assertMetrics(
  diagnostics: NonNullable<
    StressBatch["cases"][number]["stress_diagnostics"]
  >,
): void {
  assert.ok(diagnostics.segments.length > 0);
  for (const segment of diagnostics.segments) {
    assert.deepEqual(
      Object.keys(segment.metrics).sort(),
      metricKeys,
      `${diagnostics.case_id}/${segment.label} metric contract`,
    );
  }
}

function verifyPreview(
  scope: StressScope,
  record: CurrentPreviewRecord | HistoricalDatasetRecord,
  label: "current-r2" | "historical-fixture",
  expectedRevision: string | null,
): void {
  assert.ok(FIVE_SIGNAL_MODES.includes(record.mode));
  assert.equal(record.mode, "WB-wired");
  const manifestBytes = readMetadataJson(
    new URL(record.manifest_path, projectRoot),
    `${label} ${scope} manifest`,
  );
  const assetsBytes = readMetadataJson(
    new URL(record.assets_path, projectRoot),
    `${label} ${scope} assets`,
  );
  assert.equal(sha256(manifestBytes), record.manifest_sha256);
  assert.equal(sha256(assetsBytes), record.assets_sha256);
  const manifest = JSON.parse(manifestBytes.toString("utf8")) as {
    dataset_id: string;
    dataset_name: string;
    test_scope: StressScope;
    mode: FiveSignalMode;
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
  assert.equal(manifest.dataset_id, record.dataset_id);
  assert.equal(manifest.test_scope, scope);
  assert.equal(manifest.mode, record.mode);
  const rows = scopedRows(selectedContractRows(scope, record.mode));
  const cases = manifest.cases.map((entry) => ({
    case_id: entry.case_id,
    conditions: entry.conditions,
    contract_item_ids: entry.contract_item_ids,
    families: entry.families,
    signals: [...Object.keys(entry.signals)].sort(),
    signals_sha256: entry.signals_sha256,
    provenance: entry.provenance,
  }));
  const visuals = {
    schema: "scoped-acoustic-visual-v1",
    waveform_bins: 720,
    mel_bands: 40,
    mel_frames: 240,
    mel_fft_size: 512,
    mel_hop_ms: 10,
    mel_dynamic_range_db: 80,
    mel_db_offset_db: -200,
    mel_db_step_db: 0.1,
    cases: Object.fromEntries(
      cases.map((entry) => [
        entry.case_id,
        Object.fromEntries(
          SCOPED_INPUT_SIGNAL_ROLES.map((role) => [
            role,
            visualTrack(entry.signals_sha256[role], record.mode, role),
          ]),
        ),
      ]),
    ),
  };
  const preview = readStressPreview(
    {
      dataset_id: manifest.dataset_id,
      dataset_name: manifest.dataset_name,
      dataset_manifest_sha256: record.manifest_sha256,
      dataset_assets_sha256: record.assets_sha256,
      test_scope: manifest.test_scope,
      mode: manifest.mode,
      scene: manifest.scene,
      schema_version: "acoustic-stress-preview-v1",
      dataset_kind: "acoustic_stress_preview",
      contract_item_count: manifest.contract_item_count,
      preview_condition_group_count: manifest.preview_condition_group_count,
      observation_count: 100,
      category_counts: Object.fromEntries(
        STRESS_CATEGORIES[scope].map((category) => [category, 20]),
      ),
      category_labels: STRESS_CATEGORY_LABELS[scope],
      contracts: rows,
      cases,
      visuals,
      standard_compliance: false,
      formal_test_status: "not_measured",
    },
    {
      scope,
      mode: record.mode,
      datasetId: record.dataset_id,
      contracts: rows,
    },
  );
  assert.equal(preview.dataset_id, record.dataset_id);
  assert.equal(preview.cases.length, 100);
  assert.equal(Object.keys(preview.visuals.cases).length, 100);
  if (expectedRevision !== null) {
    for (const entry of preview.cases) {
      assert.equal(entry.provenance.dataset_revision, expectedRevision);
    }
  }
  if (
    "expected_first_case_microphone_sha256" in record
    && record.expected_first_case_microphone_sha256 !== null
  ) {
    assert.equal(
      preview.cases[0].signals_sha256.microphone,
      record.expected_first_case_microphone_sha256,
    );
  }
  console.log(
    `PASS ${label} ${scope}/${record.mode} dataset=${record.dataset_id} cases=100 visuals=100`,
  );
}

function verifyHistoricalSnapshot(
  record: HistoricalSnapshotRecord,
): StressBatch {
  assert.equal(
    record.semantics,
    "historical_v1_cached_snapshot_schema_only_not_current_r2_evidence",
  );
  const snapshotBytes = readMetadataJson(
    new URL(record.snapshot_path, projectRoot),
    `historical snapshot ${record.batch_id}`,
  );
  assert.equal(sha256(snapshotBytes), record.snapshot_sha256);
  const raw = JSON.parse(snapshotBytes.toString("utf8")) as Record<
    string,
    unknown
  >;
  assert.equal(raw.dataset_id, record.dataset_id);
  assert.notEqual(
    record.dataset_id,
    fixture.current_previews[record.scope].dataset_id,
  );
  const rows = scopedRows(selectedContractRows(record.scope, record.mode));
  const batch = readStressBatch(raw, {
    scope: record.scope,
    mode: record.mode,
    datasetId: record.dataset_id,
    algorithmClusterId: String(raw.algorithm_cluster_id),
    contracts: rows,
  });
  assert.equal(batch.batch_id, record.batch_id);
  assert.equal(batch.status, "completed");
  assert.equal(batch.cases.length, record.expected_cases);
  assert.deepEqual(batch.summary, record.expected_summary);
  assert.equal(batch.summary.computed, record.expected_computed);
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
    `PASS historical-schema ${record.scope}/${record.mode} batch=${record.batch_id} cases=${batch.cases.length}`,
  );
  return batch;
}

function verifyHistoricalRepresentativeSnapshots(): void {
  assert.equal(fixture.historical_representative_snapshots.length, 16);
  const observed = new Set<string>();
  let completedCases = 0;
  for (const record of fixture.historical_representative_snapshots) {
    assert.match(record.snapshot_path, /\/00000011\.json$/);
    const snapshotBytes = readMetadataJson(
      new URL(record.snapshot_path, projectRoot),
      `historical representative snapshot ${record.batch_id}`,
    );
    assert.equal(sha256(snapshotBytes), record.snapshot_sha256);
    const raw = JSON.parse(snapshotBytes.toString("utf8")) as Record<
      string,
      unknown
    >;
    const rows = scopedRows(
      selectedContractRows(record.scope, record.mode),
    );
    const batch = readStressBatch(raw, {
      scope: record.scope,
      mode: record.mode,
      datasetId: record.dataset_id,
      algorithmClusterId: String(raw.algorithm_cluster_id),
      contracts: rows,
    });
    assert.equal(batch.summary.computed, 5);
    assert.equal(batch.summary.review_required, 5);
    assert.equal(batch.cases.length, 100);
    assert.equal(batch.items.length, 100);
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
    if (record.scope === "echo") {
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
    observed.add(`${record.scope}/${record.mode}`);
  }
  assert.equal(observed.size, 16);
  assert.equal(completedCases, 80);
  console.log(
    `PASS historical-representatives profiles=${observed.size} completed_cases=${completedCases} snapshot=00000011`,
  );
}

verifyPreview(
  "echo",
  fixture.historical_datasets.echo,
  "historical-fixture",
  null,
);
verifyPreview(
  "double_talk",
  fixture.historical_datasets.double_talk,
  "historical-fixture",
  null,
);
verifyPreview(
  "echo",
  fixture.current_previews.echo,
  "current-r2",
  fixture.current_revision,
);
verifyPreview(
  "double_talk",
  fixture.current_previews.double_talk,
  "current-r2",
  fixture.current_revision,
);
for (const record of fixture.historical_cached_snapshots) {
  verifyHistoricalSnapshot(record);
}
verifyHistoricalRepresentativeSnapshots();
console.log(
  `PASS metadata-only metadata_files_read=${metadataReads} wav_files_read=0 protocol=${fixture.evaluation_protocol.version}`,
);
