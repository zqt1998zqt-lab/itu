import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import {
  FIVE_SIGNAL_MODES,
  type FiveSignalMode,
} from "../src/features/five-signal-flow-model.ts";
import {
  SCOPED_INPUT_SIGNAL_ROLES,
  type ScopedItem,
} from "../src/features/scoped-acoustic-model.ts";
import {
  STRESS_ACTIVITY_FRAME_MS,
  STRESS_ACTIVITY_HOP_MS,
  STRESS_ACTIVITY_RMS_THRESHOLD,
  STRESS_ECHO_DELAY_GUARD_MS,
  STRESS_MIN_ACTIVE_S,
  STRESS_MIN_ACTIVITY_COVERAGE,
  STRESS_CATEGORIES,
  STRESS_CATEGORY_LABELS,
  readStressBatch,
  readStressDatasets,
  readStressPreview,
  stressDynamicMicrophoneEvidence,
  stressRxEvidence,
  stressSampleRate,
  stressTxEvidence,
  type StressActivitySegment,
  type StressActivityGate,
  type StressBatch,
  type StressDiagnostics,
  type StressDiagnosticsV1,
  type StressDiagnosticsV2,
  type StressPreview,
  type StressPreviewCase,
  type StressScope,
  type StressSegmentActivity,
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

const contract = JSON.parse(
  readFileSync(
    new URL("../cpp/evaluation/nv200-test-contract.json", import.meta.url),
    "utf8",
  ),
) as { items: ContractRow[] };
const diagnosticGolden = JSON.parse(
  readFileSync(
    new URL("./fixtures/acoustic-stress-diagnostics.json", import.meta.url),
    "utf8",
  ),
) as {
  v1: StressDiagnosticsV1;
  v2: StressDiagnosticsV2;
};

const SCOPES: StressScope[] = ["echo", "double_talk"];
const BATCH_ID = "a".repeat(32);
const ALGORITHM_ID = "vcp_default";
const CREATED_AT = "2026-11-10T08:30:00.000Z";
const UPDATED_AT = "2026-11-10T08:31:00.000Z";
const WAVE_BASE64 = Buffer.alloc(720 * 2).toString("base64");
const MEL_BASE64 = Buffer.alloc(40 * 240 * 2).toString("base64");

function sha(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function datasetId(scope: StressScope, mode: FiveSignalMode): string {
  return sha(`acoustic-stress-v1:${scope}:${mode}`).slice(0, 32);
}

function datasetName(scope: StressScope, mode: FiveSignalMode): string {
  return `${scope}-${mode}-100-workloads-v1`;
}

function expectedCells(scope: StressScope, mode: FiveSignalMode): string[] {
  if (scope === "double_talk") {
    return mode.startsWith("SWB-") ? ["B18", "B19"] : ["B19", "B20"];
  }
  return mode.startsWith("SWB-")
    ? ["B20", "B21", "B22"]
    : ["B21", "B22"];
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

function conditionsFor(
  row: ContractRow,
  mode: FiveSignalMode,
): StressPreviewCase["conditions"] {
  const name = row.test_name.toLowerCase();
  const scene = name.match(/sce\s*([1-4])/);
  const window = name.match(/(50|100)\s*ms/);
  const compactName = name.replaceAll(" ", "");
  const volume = name.includes("recommended")
    ? "recommended"
    : /\b(maximum|max)\b/.test(name)
      ? "maximum"
      : "nominal";
  const inputLevel = compactName.includes("-5db")
    ? -5
    : compactName.includes("+5db")
      ? 5
      : 0;
  return {
    mode,
    direction: row.direction,
    scene: scene
      ? (`sce${scene[1]}` as StressPreviewCase["conditions"]["scene"])
      : "baseline",
    volume,
    window_ms: window ? Number(window[1]) : null,
    input_level_db: inputLevel,
  };
}

function activitySegments(
  scope: StressScope,
  category: (typeof STRESS_CATEGORIES)[StressScope][number],
): StressActivitySegment[] {
  if (scope === "echo") {
    return category === "convergence"
      ? [
          { start_s: 0, end_s: 0.5, label: "quiet" },
          { start_s: 0.5, end_s: 12, label: "far_only" },
        ]
      : [{ start_s: 0, end_s: 12, label: "far_only" }];
  }
  if (category === "transition") {
    return [
      { start_s: 0, end_s: 0.5, label: "quiet" },
      { start_s: 0.5, end_s: 2, label: "far_only" },
      { start_s: 2, end_s: 5, label: "near_only" },
      { start_s: 5, end_s: 8, label: "far_only" },
      { start_s: 8, end_s: 11, label: "double_talk" },
      { start_s: 11, end_s: 12, label: "noise_only" },
    ];
  }
  if (category === "short_overlap") {
    return [
      { start_s: 0, end_s: 0.5, label: "quiet" },
      { start_s: 0.5, end_s: 6.5, label: "near_only" },
      { start_s: 6.5, end_s: 7, label: "double_talk" },
      { start_s: 7, end_s: 12, label: "far_only" },
    ];
  }
  return [{ start_s: 0, end_s: 12, label: "double_talk" }];
}

function inputHashes(
  scope: StressScope,
  mode: FiveSignalMode,
  caseId: string,
): Record<(typeof SCOPED_INPUT_SIGNAL_ROLES)[number], string> {
  return Object.fromEntries(
    SCOPED_INPUT_SIGNAL_ROLES.map((role) => [
      role,
      sha(`${scope}:${mode}:${caseId}:${role}`),
    ]),
  ) as Record<(typeof SCOPED_INPUT_SIGNAL_ROLES)[number], string>;
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
    wave_min_i16: WAVE_BASE64,
    wave_max_i16: WAVE_BASE64,
    mel_i16: MEL_BASE64,
    mel_ceiling_db: -20,
  };
}

function makeDataset(scope: StressScope, mode: FiveSignalMode) {
  const rows = selectedContractRows(scope, mode);
  const id = datasetId(scope, mode);
  const name = datasetName(scope, mode);
  const categories = STRESS_CATEGORIES[scope];
  const categoryCounts = Object.fromEntries(
    categories.map((category) => [category, 20]),
  );
  const identity = {
    dataset_id: id,
    dataset_name: name,
    test_scope: scope,
    mode,
    scene: "baseline" as const,
    contract_item_count: rows.length,
    preview_condition_group_count: 100 as const,
    observation_count: 100 as const,
    category_counts: categoryCounts,
    category_labels: STRESS_CATEGORY_LABELS[scope],
    standard_compliance: false as const,
    formal_test_status: "not_measured" as const,
  };
  return {
    ...identity,
    contracts: scopedRows(rows),
    datasets: [{ ...identity }],
  };
}

function previewCases(
  scope: StressScope,
  mode: FiveSignalMode,
): StressPreviewCase[] {
  const rows = selectedContractRows(scope, mode);
  const categories = STRESS_CATEGORIES[scope];
  return categories.flatMap((category) =>
    Array.from({ length: 20 }, (_, offset) => {
      const categoryIndex = offset + 1;
      const row = rows[offset % rows.length];
      const caseId =
        `${scope}-${mode}-${category}-${String(categoryIndex).padStart(2, "0")}-stress-v1`;
      return {
        case_id: caseId,
        conditions: conditionsFor(row, mode),
        families: [row.family],
        contract_item_ids: [row.id] as [string],
        provenance: {
          scope,
          mode,
          scene: "baseline",
          workload_category: category,
          category_index: categoryIndex,
          activity_segments: activitySegments(scope, category),
          conversation_mode: scope === "echo"
            ? "far_single"
            : category === "transition"
              ? "transition"
              : "double_talk",
          analysis_start_s: [
            "convergence",
            "transition",
            "short_overlap",
          ].includes(category)
            ? 0.5
            : 0,
          engineering_preview: true,
          engineering_combination: true,
          independent_source_recordings: false,
          actual_vehicle_recording: false,
          within_case_source_looping: false,
        },
        signals: [...SCOPED_INPUT_SIGNAL_ROLES],
        signals_sha256: inputHashes(scope, mode, caseId),
      };
    })
  );
}

function makePreview(scope: StressScope, mode: FiveSignalMode) {
  const dataset = makeDataset(scope, mode);
  const cases = previewCases(scope, mode);
  const visualCases = Object.fromEntries(
    cases.map((entry) => [
      entry.case_id,
      Object.fromEntries(
        SCOPED_INPUT_SIGNAL_ROLES.map((role) => [
          role,
          visualTrack(entry.signals_sha256[role], mode, role),
        ]),
      ),
    ]),
  );
  return {
    ...dataset,
    schema_version: "acoustic-stress-preview-v1" as const,
    dataset_kind: "acoustic_stress_preview" as const,
    dataset_manifest_sha256: sha(`manifest:${scope}:${mode}`),
    dataset_assets_sha256: sha(`assets:${scope}:${mode}`),
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
  };
}

type DiagnosticsOptions = {
  txMuted?: boolean;
  txDurationS?: number;
};

function diagnosticMetrics(
  label: StressActivitySegment["label"],
  scope: StressScope,
  primaryValue: number | null,
  txInvalid: boolean,
) {
  const metrics = {
    primary: null as number | null,
    tx_residual_reduction_db: null as number | null,
    convergence_db: null as number | null,
    cleanref_projection_gain_db: null as number | null,
    cleanref_correlation: null as number | null,
    tx_level_dbfs: null as number | null,
    mic_level_dbfs: null as number | null,
    far_level_dbfs: null as number | null,
    near_level_dbfs: null as number | null,
  };
  if (label === "quiet" || label === "noise_only") return metrics;
  if (scope === "double_talk" && txInvalid) return metrics;
  metrics.mic_level_dbfs = -32;
  if (label === "far_only") {
    metrics.far_level_dbfs = -36;
    metrics.tx_level_dbfs = txInvalid ? null : -30;
    if (scope === "echo") {
      metrics.primary = primaryValue;
      metrics.tx_residual_reduction_db = txInvalid ? null : 4.25;
    }
  } else if (label === "near_only") {
    metrics.near_level_dbfs = -30;
    metrics.tx_level_dbfs = txInvalid ? null : -30;
    if (!txInvalid) {
      metrics.cleanref_projection_gain_db = 0.25;
      metrics.cleanref_correlation = 0.98;
    }
  } else if (label === "double_talk") {
    metrics.far_level_dbfs = -36;
    metrics.near_level_dbfs = -30;
    metrics.tx_level_dbfs = txInvalid ? null : -30;
    if (!txInvalid) {
      metrics.primary = primaryValue;
      metrics.tx_residual_reduction_db = 1.5;
    }
  }
  return metrics;
}

function diagnosticStatus(
  label: StressActivitySegment["label"],
  scope: StressScope,
  metrics: StressDiagnostics["segments"][number]["metrics"],
  txInvalid: boolean,
): StressDiagnostics["segments"][number]["status"] {
  if (label === "quiet" || label === "noise_only") return "not_applicable";
  if (scope === "double_talk" && txInvalid) return "not_measurable";
  if (scope === "double_talk" && label === "far_only") {
    return "not_measurable";
  }
  return Object.values(metrics).some(
    (metric) => typeof metric === "number" && Number.isFinite(metric),
  )
    ? "review_required"
    : "not_measurable";
}

function segmentActivity(
  segment: StressActivitySegment,
  scope: StressScope,
  mode: FiveSignalMode,
): StressSegmentActivity {
  const rate = stressSampleRate(mode);
  const duration = segment.end_s - segment.start_s;
  const activeSamples = Math.round(duration * rate);
  const near = ["near_only", "double_talk"].includes(segment.label)
    ? activeSamples
    : 0;
  const far = ["far_only", "double_talk"].includes(segment.label)
    ? activeSamples
    : 0;
  const joint = segment.label === "double_talk" ? activeSamples : 0;
  const result: StressSegmentActivity = {
    frame_ms: STRESS_ACTIVITY_FRAME_MS,
    hop_ms: STRESS_ACTIVITY_HOP_MS,
    rms_threshold: STRESS_ACTIVITY_RMS_THRESHOLD,
    min_active_s: STRESS_MIN_ACTIVE_S,
    min_coverage: STRESS_MIN_ACTIVITY_COVERAGE,
    near_active_samples: near,
    far_active_samples: far,
    joint_active_samples: joint,
    near_coverage: near > 0 ? 1 : 0,
    far_coverage: far > 0 ? 1 : 0,
    joint_coverage: joint > 0 ? 1 : 0,
    valid: true,
    reason: "controlled activity evidence",
  };
  if (scope === "echo" && segment.label === "far_only") {
    result.echo_delay_guard_ms = STRESS_ECHO_DELAY_GUARD_MS;
    result.echo_guarded_active_samples = far;
    result.echo_guarded_active_s = far / rate;
  }
  return result;
}

function diagnosticSegments(
  entry: StressPreviewCase,
  scope: StressScope,
  mode: FiveSignalMode,
  primaryValue: number | null,
  txInvalid: boolean,
): StressDiagnostics["segments"] {
  return entry.provenance.activity_segments.map((segment) => {
    const metrics = diagnosticMetrics(
      segment.label,
      scope,
      primaryValue,
      txInvalid,
    );
    const segmentMode: StressDiagnostics["segments"][number]["mode"] =
      segment.label === "quiet"
        ? "not_applicable"
        : segment.label === "far_only"
          ? "far"
          : segment.label === "near_only"
            ? "near"
            : segment.label === "double_talk"
              ? "double_talk"
              : "noise";
    return {
      ...segment,
      mode: segmentMode,
      status: diagnosticStatus(segment.label, scope, metrics, txInvalid),
      metrics,
      reason: "controlled engineering diagnostic",
    };
  });
}

function adaptGoldenDiagnostics(
  template: StressDiagnostics,
  entry: StressPreviewCase,
  scope: StressScope,
  mode: FiveSignalMode,
  txHash: string,
): StressDiagnostics {
  assert.equal(
    template.segments.length,
    entry.provenance.activity_segments.length,
  );
  const segments = template.segments.map((source, index) => {
    const activity = entry.provenance.activity_segments[index];
    assert.equal(source.label, activity.label);
    assert.equal(source.start_s, activity.start_s);
    assert.equal(source.end_s, activity.end_s);
    return { ...source };
  });
  return {
    ...template,
    case_id: entry.case_id,
    tx_sha256: txHash,
    scope,
    mode,
    category: entry.provenance.workload_category,
    segments,
  } as StressDiagnostics;
}

function diagnosticCommon(
  entry: StressPreviewCase,
  scope: StressScope,
  mode: FiveSignalMode,
  txHash: string,
  options: DiagnosticsOptions,
) {
  const txMuted = options.txMuted ?? false;
  const txDurationS = options.txDurationS ?? 12;
  const txInvalid = txMuted || txDurationS < 0.95 * 12;
  const primaryValue = scope === "echo"
    ? 12.5
    : txInvalid
      ? null
      : 88;
  const primaryName: StressDiagnostics["primary"]["name"] = scope === "echo"
    ? "echo_primary_proxy_db"
    : "double_talk_energy_ratio_proxy_percent";
  const primaryUnit: StressDiagnostics["primary"]["unit"] =
    scope === "echo" ? "dB" : "percent";
  return {
    txInvalid,
    common: {
      case_id: entry.case_id,
      tx_sha256: txHash,
      scope,
      mode,
      category: entry.provenance.workload_category,
      evidence_grade: "engineering_proxy" as const,
      formal_compliance: false as const,
      standard_compliance: false as const,
      formal_test_status: "not_measured" as const,
      status: primaryValue === null ? "not_measurable" as const : "review_required" as const,
      reason: "controlled engineering diagnostic",
      review_required: primaryValue !== null,
      sample_rate_hz: stressSampleRate(mode),
      source_duration_s: 12,
      tx_duration_s: txDurationS,
      tx_muted: txMuted,
      primary: {
        name: primaryName,
        value: primaryValue,
        unit: primaryUnit,
        status: primaryValue === null
          ? "not_measured" as const
          : "review_required" as const,
      },
      convergence_db: scope === "echo" && !txInvalid ? 1.25 : null,
      limitations: ["engineering proxy only"],
    },
  };
}

function makeDiagnostics(
  entry: StressPreviewCase,
  scope: StressScope,
  mode: FiveSignalMode,
  txHash: string,
  options: DiagnosticsOptions = {},
): StressDiagnosticsV1 {
  const { txInvalid, common } = diagnosticCommon(
    entry,
    scope,
    mode,
    txHash,
    options,
  );
  return {
    ...common,
    schema_version: "acoustic-stress-diagnostics-v1",
    segments: diagnosticSegments(
      entry,
      scope,
      mode,
      common.primary.value,
      txInvalid,
    ) as StressDiagnosticsV1["segments"],
  };
}

function makeDiagnosticsV2(
  entry: StressPreviewCase,
  scope: StressScope,
  mode: FiveSignalMode,
  txHash: string,
  options: DiagnosticsOptions = {},
): StressDiagnosticsV2 {
  const { txInvalid, common } = diagnosticCommon(
    entry,
    scope,
    mode,
    txHash,
    options,
  );
  const baseSegments = diagnosticSegments(
    entry,
    scope,
    mode,
    common.primary.value,
    txInvalid,
  );
  return {
    ...common,
    schema_version: "acoustic-stress-diagnostics-v2",
    activity_gate: {
      frame_ms: STRESS_ACTIVITY_FRAME_MS,
      hop_ms: STRESS_ACTIVITY_HOP_MS,
      rms_threshold: STRESS_ACTIVITY_RMS_THRESHOLD,
      min_active_s: STRESS_MIN_ACTIVE_S,
      min_coverage: STRESS_MIN_ACTIVITY_COVERAGE,
    } satisfies StressActivityGate,
    convergence_note: "supplemental only; not an adaptive-filter convergence result",
    segments: baseSegments.map((segment) => ({
      ...segment,
      activity: segmentActivity(segment, scope, mode),
    })) as StressDiagnosticsV2["segments"],
  };
}

type DiagnosticsFactory = (
  entry: StressPreviewCase,
  scope: StressScope,
  mode: FiveSignalMode,
  txHash: string,
) => StressDiagnostics;

function makeBatch(
  preview: StressPreview,
  diagnosticsFactory: DiagnosticsFactory = makeDiagnostics,
): StressBatch {
  const scope = preview.test_scope;
  const mode = preview.mode;
  const datasetManifest = preview.dataset_manifest_sha256;
  const configuration = { profile: ALGORITHM_ID };
  const cases = preview.cases.map((entry, index) => {
    const outputHash = sha(`tx:${entry.case_id}`);
    const reportId = sha(`report:${entry.case_id}`).slice(0, 32);
    return {
      case_id: entry.case_id,
      status: "completed" as const,
      report_id: reportId,
      conditions: entry.conditions,
      contract_item_ids: [entry.contract_item_ids[0]] as [string],
      provenance: entry.provenance,
      signals: Object.fromEntries(
        SCOPED_INPUT_SIGNAL_ROLES.map((role) => [
          role,
          `assets/${entry.case_id}/${role}.wav`,
        ]),
      ) as Record<(typeof SCOPED_INPUT_SIGNAL_ROLES)[number], string>,
      signals_sha256: entry.signals_sha256,
      input_sha256: entry.signals_sha256,
      output_sha256: { tx: outputHash },
      tx_visual: visualTrack(outputHash, mode, "noise"),
      stress_diagnostics: diagnosticsFactory(entry, scope, mode, outputHash),
      assessment_sha256: sha(`assessment:${entry.case_id}:${index}`),
      execution: {
        output_sha256: { tx: outputHash },
        configuration,
        algorithm_cluster_id: ALGORITHM_ID,
        cli_sha256: sha("cli"),
      },
    };
  });
  const items = preview.cases.map((entry) => {
    const caseRecord = cases.find(
      (candidate) => candidate.case_id === entry.case_id,
    )!;
    const row = preview.contracts.find(
      (candidate) => candidate.id === entry.contract_item_ids[0],
    )!;
    const diagnostics = caseRecord.stress_diagnostics;
    const status = diagnostics.primary.value === null
      ? "not_measured" as const
      : "review_required" as const;
    return {
      ...row,
      contract_id: row.id,
      item_id: row.id,
      observation_id: `${entry.case_id}:${row.id}`,
      case_id: entry.case_id,
      mode,
      status,
      value: diagnostics.primary.value,
      metric: diagnostics.primary.name,
      reason: diagnostics.reason,
      evidence_binding: {
        ...entry.conditions,
        case_id: entry.case_id,
        report_id: caseRecord.report_id,
        batch_id: BATCH_ID,
        dataset_id: preview.dataset_id,
        input_sha256: entry.signals_sha256,
        output_sha256: { tx: caseRecord.output_sha256.tx },
      },
    };
  });
  const reviewRequired = items.filter(
    (item) => item.status === "review_required",
  ).length;
  const notMeasured = items.filter(
    (item) => item.status === "not_measured",
  ).length;
  return {
    schema_version: "scene-condition-batch-v1",
    batch_id: BATCH_ID,
    dataset_id: preview.dataset_id,
    dataset_name: preview.dataset_name,
    dataset_manifest_sha256: datasetManifest,
    dataset_assets_sha256: preview.dataset_assets_sha256,
    algorithm_cluster_id: ALGORITHM_ID,
    configuration,
    catalog_sha256: sha("catalog"),
    cli_sha256: sha("cli"),
    status: "completed",
    created_at: CREATED_AT,
    updated_at: UPDATED_AT,
    test_scope: scope,
    stress_scope: scope,
    selected_mode: mode,
    selected_scene: "baseline",
    standard_compliance: false,
    formal_test_status: "not_measured",
    contract_item_count: preview.contract_item_count,
    case_count: 100,
    preview_condition_group_count: 100,
    observation_count: 100,
    category_counts: preview.category_counts,
    progress: {
      completed_cases: 100,
      total_cases: 100,
      completed_items: 100,
      total_items: 100,
    },
    summary: {
      total: 100,
      computed: reviewRequired,
      pass: 0,
      fail: 0,
      review_required: reviewRequired,
      not_measured: notMeasured,
      invalid: 0,
    },
    cases,
    items,
  };
}

function makeSingleBatch(
  preview: StressPreview,
  caseId: string,
): StressBatch {
  const full = makeBatch(preview);
  const selectedCase = full.cases.find((entry) => entry.case_id === caseId);
  assert.ok(selectedCase);
  const items = structuredClone(
    full.items.filter((entry) => entry.case_id === caseId),
  );
  const category = selectedCase.provenance.workload_category;
  const categoryCounts = Object.fromEntries(
    STRESS_CATEGORIES[preview.test_scope].map((candidate) => [
      candidate,
      candidate === category ? 1 : 0,
    ]),
  );
  const reviewRequired = items.filter(
    (entry) => entry.status === "review_required",
  ).length;
  const notMeasured = items.filter(
    (entry) => entry.status === "not_measured",
  ).length;
  return {
    ...full,
    cases: [selectedCase],
    items,
    case_count: 1,
    observation_count: items.length,
    preview_condition_group_count: 100,
    execution_scope: "single_case",
    selected_case_id: caseId,
    dataset_case_count: 100,
    dataset_observation_count: 100,
    category_counts: categoryCounts,
    progress: {
      completed_cases: 1,
      total_cases: 1,
      completed_items: items.length,
      total_items: items.length,
    },
    summary: {
      total: items.length,
      computed: reviewRequired,
      pass: 0,
      fail: 0,
      review_required: reviewRequired,
      not_measured: notMeasured,
      invalid: 0,
    },
  };
}

function makeRxBatch(preview: StressPreview): StressBatch {
  const value = structuredClone(makeBatch(preview));
  const target = value.cases[0];
  const rxSha256 = sha("rx:processed");
  target.output_sha256 = {
    tx: target.output_sha256!.tx,
    rx: rxSha256,
  };
  target.rx_visual = visualTrack(rxSha256, preview.mode, "noise");
  target.execution = {
    ...target.execution,
    echo_reference: {
      source: "vcp_rx_processed",
      processed_output: "rx",
      processed_sha256: rxSha256,
      tx_echo_argument_sha256: rxSha256,
    },
  };
  for (const item of value.items) {
    if (item.case_id === target.case_id) {
      item.evidence_binding.output_sha256 = {
        tx: target.output_sha256.tx,
        rx: rxSha256,
      };
    }
  }
  return value;
}

function expectPreviewFailure(
  value: unknown,
  expected: {
    scope: StressScope;
    mode: FiveSignalMode;
    datasetId: string;
    contracts: readonly ScopedItem[];
  },
): void {
  assert.throws(() => readStressPreview(value, expected));
}

function expectBatchFailure(
  value: unknown,
  expected: {
    scope: StressScope;
    mode: FiveSignalMode;
    datasetId: string;
    algorithmClusterId: string;
    contracts: readonly ScopedItem[];
  },
): void {
  assert.throws(() => readStressBatch(value, expected));
}

test("all 16 stress scope/mode combinations bind real contract rows and 100 workloads", () => {
  const observed = new Set<string>();
  for (const scope of SCOPES) {
    for (const mode of FIVE_SIGNAL_MODES) {
      const rows = selectedContractRows(scope, mode);
      assert.deepEqual(
        rows.map((row) => row.source_cell),
        expectedCells(scope, mode),
        `${scope}/${mode} contract cells`,
      );
      const datasetValue = makeDataset(scope, mode);
      const dataset = readStressDatasets(datasetValue, { scope, mode });
      assert.equal(dataset.dataset_id, datasetId(scope, mode));
      assert.equal(dataset.dataset_name, datasetName(scope, mode));
      assert.equal(dataset.contracts.length, expectedCells(scope, mode).length);
      assert.deepEqual(
        dataset.category_labels,
        STRESS_CATEGORY_LABELS[scope],
      );

      const preview = readStressPreview(makePreview(scope, mode), {
        scope,
        mode,
        datasetId: dataset.dataset_id,
        contracts: dataset.contracts,
      });
      assert.equal(preview.cases.length, 100);
      assert.equal(Object.keys(preview.visuals.cases).length, 100);
      assert.deepEqual(
        Object.fromEntries(
          STRESS_CATEGORIES[scope].map((category) => [
            category,
            preview.cases.filter(
              (entry) =>
                entry.provenance.workload_category === category,
            ).length,
          ]),
        ),
        Object.fromEntries(
          STRESS_CATEGORIES[scope].map((category) => [category, 20]),
        ),
      );
      for (const entry of preview.cases) {
        assert.deepEqual(
          [...entry.signals].sort(),
          [...SCOPED_INPUT_SIGNAL_ROLES].sort(),
        );
        assert.equal(
          preview.visuals.cases[entry.case_id].microphone.sample_rate,
          stressSampleRate(mode),
        );
        assert.equal(
          preview.visuals.cases[entry.case_id].microphone.channels,
          2,
        );
        assert.equal(
          preview.visuals.cases[entry.case_id].reference.channels,
          1,
        );
      }
      observed.add(`${scope}/${mode}`);
    }
  }
  assert.equal(observed.size, 16);
});

test("compact real diagnostics golden covers v1, v2, and nine metrics", () => {
  const preview = readStressPreview(makePreview("echo", "WB-wired"), {
    scope: "echo",
    mode: "WB-wired",
    datasetId: datasetId("echo", "WB-wired"),
    contracts: scopedRows(
      selectedContractRows("echo", "WB-wired"),
    ),
  });
  const expected = {
    scope: "echo" as const,
    mode: "WB-wired" as const,
    datasetId: preview.dataset_id,
    algorithmClusterId: ALGORITHM_ID,
    contracts: preview.contracts,
  };
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

  const v1Batch = readStressBatch(
    makeBatch(
      preview,
      (entry, scope, mode, txHash) =>
        entry.provenance.workload_category === "steady_echo"
          && entry.provenance.category_index === 1
          ? adaptGoldenDiagnostics(
              diagnosticGolden.v1,
              entry,
              scope,
              mode,
              txHash,
            )
          : makeDiagnostics(entry, scope, mode, txHash),
    ),
    expected,
  );
  const v1Case = v1Batch.cases.find(
    (entry) =>
      entry.provenance.workload_category === "steady_echo"
      && entry.provenance.category_index === 1,
  );
  assert.ok(v1Case?.stress_diagnostics);
  assert.equal(
    v1Case.stress_diagnostics.schema_version,
    "acoustic-stress-diagnostics-v1",
  );
  assert.deepEqual(
    Object.keys(v1Case.stress_diagnostics.segments[0].metrics).sort(),
    metricKeys,
  );
  assert.equal(
    Object.hasOwn(v1Case.stress_diagnostics.segments[0], "activity"),
    false,
  );

  const v2Batch = readStressBatch(
    makeBatch(
      preview,
      (entry, scope, mode, txHash) =>
        entry.provenance.workload_category === "convergence"
          && entry.provenance.category_index === 2
          ? adaptGoldenDiagnostics(
              diagnosticGolden.v2,
              entry,
              scope,
              mode,
              txHash,
            )
          : makeDiagnosticsV2(entry, scope, mode, txHash),
    ),
    expected,
  );
  const v2Case = v2Batch.cases.find(
    (entry) =>
      entry.provenance.workload_category === "convergence"
      && entry.provenance.category_index === 2,
  );
  assert.ok(v2Case?.stress_diagnostics);
  const v2Diagnostics = v2Case.stress_diagnostics;
  assert.equal(
    v2Diagnostics.schema_version,
    "acoustic-stress-diagnostics-v2",
  );
  if (v2Diagnostics.schema_version !== "acoustic-stress-diagnostics-v2") {
    throw Error("expected v2 golden diagnostics");
  }
  assert.deepEqual(v2Diagnostics.activity_gate, {
    frame_ms: 20,
    hop_ms: 10,
    rms_threshold: 0.00001,
    min_active_s: 0.05,
    min_coverage: 0.05,
  });
  assert.deepEqual(
    Object.keys(v2Diagnostics.segments[1].metrics).sort(),
    metricKeys,
  );
  assert.equal(v2Diagnostics.segments[0].label, "quiet");
  assert.equal(v2Diagnostics.segments[0].status, "not_applicable");
});

test("v2 diagnostics require activity evidence and preserve echo input coupling on mute", () => {
  const preview = readStressPreview(makePreview("echo", "WB-wired"), {
    scope: "echo",
    mode: "WB-wired",
    datasetId: datasetId("echo", "WB-wired"),
    contracts: scopedRows(
      selectedContractRows("echo", "WB-wired"),
    ),
  });
  const value = makeBatch(preview, makeDiagnosticsV2);
  const batch = readStressBatch(value, {
    scope: "echo",
    mode: "WB-wired",
    datasetId: preview.dataset_id,
    algorithmClusterId: ALGORITHM_ID,
    contracts: preview.contracts,
  });
  const diagnostics = batch.cases[0].stress_diagnostics!;
  assert.equal(
    diagnostics.schema_version,
    "acoustic-stress-diagnostics-v2",
  );
  if (diagnostics.schema_version !== "acoustic-stress-diagnostics-v2") {
    throw Error("expected v2 diagnostics");
  }
  assert.deepEqual(diagnostics.activity_gate, {
    frame_ms: 20,
    hop_ms: 10,
    rms_threshold: 0.00001,
    min_active_s: 0.05,
    min_coverage: 0.05,
  });
  assert.equal(
    diagnostics.segments.every((segment) => segment.activity.valid),
    true,
  );

  const muted = makeBatch(
    preview,
    (entry, scope, mode, txHash) =>
      makeDiagnosticsV2(entry, scope, mode, txHash, { txMuted: true }),
  );
  const mutedBatch = readStressBatch(muted, {
    scope: "echo",
    mode: "WB-wired",
    datasetId: preview.dataset_id,
    algorithmClusterId: ALGORITHM_ID,
    contracts: preview.contracts,
  });
  const mutedDiagnostics = mutedBatch.cases[0].stress_diagnostics!;
  const far = mutedDiagnostics.segments.find(
    (segment) => segment.label === "far_only",
  )!;
  assert.equal(mutedDiagnostics.tx_muted, true);
  assert.equal(mutedDiagnostics.status, "review_required");
  assert.equal(typeof mutedDiagnostics.primary.value, "number");
  assert.equal(mutedDiagnostics.convergence_db, null);
  assert.equal(far.status, "review_required");
  assert.equal(typeof far.metrics.primary, "number");
  assert.equal(far.metrics.tx_residual_reduction_db, null);
  assert.equal(far.metrics.tx_level_dbfs, null);
});

test("v2 double-talk mute clears TX-dependent speech metrics", () => {
  const preview = readStressPreview(
    makePreview("double_talk", "WB-wired"),
    {
      scope: "double_talk",
      mode: "WB-wired",
      datasetId: datasetId("double_talk", "WB-wired"),
      contracts: scopedRows(
        selectedContractRows("double_talk", "WB-wired"),
      ),
    },
  );
  const value = makeBatch(
    preview,
    (entry, scope, mode, txHash) =>
      makeDiagnosticsV2(entry, scope, mode, txHash, { txMuted: true }),
  );
  const batch = readStressBatch(value, {
    scope: "double_talk",
    mode: "WB-wired",
    datasetId: preview.dataset_id,
    algorithmClusterId: ALGORITHM_ID,
    contracts: preview.contracts,
  });
  const diagnostics = batch.cases[0].stress_diagnostics!;
  assert.equal(diagnostics.tx_muted, true);
  assert.equal(diagnostics.status, "not_measurable");
  assert.equal(diagnostics.primary.value, null);
  assert.equal(diagnostics.convergence_db, null);
  const speech = diagnostics.segments.find(
    (segment) => segment.label === "double_talk",
  )!;
  assert.equal(speech.status, "not_measurable");
  assert.equal(
    Object.values(speech.metrics).every((metric) => metric === null),
    true,
  );
});

test("v2 rejects missing activity, forged counts, and illegal mute promotion", () => {
  const preview = readStressPreview(makePreview("echo", "WB-wired"), {
    scope: "echo",
    mode: "WB-wired",
    datasetId: datasetId("echo", "WB-wired"),
    contracts: scopedRows(
      selectedContractRows("echo", "WB-wired"),
    ),
  });
  const value = makeBatch(preview, makeDiagnosticsV2);
  const expected = {
    scope: "echo" as const,
    mode: "WB-wired" as const,
    datasetId: preview.dataset_id,
    algorithmClusterId: ALGORITHM_ID,
    contracts: preview.contracts,
  };
  const mutate = (callback: (copy: typeof value) => void) => {
    const copy = structuredClone(value);
    callback(copy);
    expectBatchFailure(copy, expected);
  };

  mutate((copy) => {
    delete (copy.cases[0].stress_diagnostics as unknown as {
      activity_gate?: unknown;
    }).activity_gate;
  });
  mutate((copy) => {
    (copy.cases[0].stress_diagnostics as unknown as {
      activity_gate: Record<string, unknown>;
    }).activity_gate.frame_ms = 10;
  });
  mutate((copy) => {
    delete (copy.cases[0].stress_diagnostics!.segments[0] as unknown as {
      activity?: unknown;
    }).activity;
  });
  mutate((copy) => {
    const activity = (
      copy.cases[0].stress_diagnostics!.segments[0] as unknown as {
        activity: Record<string, unknown>;
      }
    ).activity;
    activity.near_coverage = 1.1;
  });
  mutate((copy) => {
    const activity = (
      copy.cases[0].stress_diagnostics!.segments[0] as unknown as {
        activity: Record<string, unknown>;
      }
    ).activity;
    activity.near_active_samples = 0;
    activity.far_active_samples = 0;
    activity.joint_active_samples = 0;
    activity.near_coverage = 0;
    activity.far_coverage = 0;
    activity.joint_coverage = 0;
    activity.echo_guarded_active_samples = 0;
    activity.echo_guarded_active_s = 0;
    activity.valid = true;
  });
  mutate((copy) => {
    const activity = (
      copy.cases[0].stress_diagnostics!.segments[0] as unknown as {
        activity: Record<string, unknown>;
      }
    ).activity;
    const segmentSamples = 12 * stressSampleRate("WB-wired");
    activity.far_active_samples = segmentSamples + 1;
    activity.far_coverage = (segmentSamples + 1) / segmentSamples;
  });
  mutate((copy) => {
    const activity = (
      copy.cases[0].stress_diagnostics!.segments[0] as unknown as {
        activity: Record<string, unknown>;
      }
    ).activity;
    activity.far_coverage = Number(activity.far_coverage) - 0.01;
  });
  mutate((copy) => {
    const activity = (
      copy.cases[0].stress_diagnostics!.segments[0] as unknown as {
        activity: Record<string, unknown>;
      }
    ).activity;
    activity.joint_active_samples = 1;
    activity.joint_coverage =
      1 / (12 * stressSampleRate("WB-wired"));
  });
  mutate((copy) => {
    const activity = (
      copy.cases[0].stress_diagnostics!.segments[0] as unknown as {
        activity: Record<string, unknown>;
      }
    ).activity;
    activity.valid = false;
  });
  mutate((copy) => {
    const activity = (
      copy.cases[0].stress_diagnostics!.segments[0] as unknown as {
        activity: Record<string, unknown>;
      }
    ).activity;
    activity.near_active_samples = 1.5;
  });
  mutate((copy) => {
    const activity = (
      copy.cases[0].stress_diagnostics!.segments[0] as unknown as {
        activity: Record<string, unknown>;
      }
    ).activity;
    activity.bogus_activity = true;
  });
  mutate((copy) => {
    const activity = (
      copy.cases[0].stress_diagnostics!.segments[0] as unknown as {
        activity: Record<string, unknown>;
      }
    ).activity;
    delete activity.echo_delay_guard_ms;
  });
  mutate((copy) => {
    const activity = (
      copy.cases[0].stress_diagnostics!.segments[0] as unknown as {
        activity: Record<string, unknown>;
      }
    ).activity;
    activity.echo_guarded_active_samples =
      Number(activity.far_active_samples) + 1;
  });
  mutate((copy) => {
    (copy.cases[0].stress_diagnostics as unknown as {
      convergence_note: unknown;
    }).convergence_note = 42;
  });

  const muted = makeBatch(
    preview,
    (entry, scope, mode, txHash) =>
      makeDiagnosticsV2(entry, scope, mode, txHash, { txMuted: true }),
  );
  const mutedExpected = { ...expected };
  const mutedFar = structuredClone(muted);
  mutedFar.cases[0].stress_diagnostics!.segments.find(
    (segment) => segment.label === "far_only",
  )!.metrics.tx_residual_reduction_db = 3;
  expectBatchFailure(mutedFar, mutedExpected);

  const doublePreview = readStressPreview(
    makePreview("double_talk", "WB-wired"),
    {
      scope: "double_talk",
      mode: "WB-wired",
      datasetId: datasetId("double_talk", "WB-wired"),
      contracts: scopedRows(
        selectedContractRows("double_talk", "WB-wired"),
      ),
    },
  );
  const mutedDouble = makeBatch(
    doublePreview,
    (entry, scope, mode, txHash) =>
      makeDiagnosticsV2(entry, scope, mode, txHash, { txMuted: true }),
  );
  mutedDouble.cases[0].stress_diagnostics!.primary.value = 88;
  mutedDouble.cases[0].stress_diagnostics!.primary.status =
    "review_required";
  mutedDouble.cases[0].stress_diagnostics!.status = "review_required";
  mutedDouble.cases[0].stress_diagnostics!.review_required = true;
  expectBatchFailure(mutedDouble, {
    scope: "double_talk",
    mode: "WB-wired",
    datasetId: doublePreview.dataset_id,
    algorithmClusterId: ALGORITHM_ID,
    contracts: doublePreview.contracts,
  });

  const legacy = makeBatch(preview, makeDiagnostics);
  (legacy.cases[0].stress_diagnostics as unknown as {
    activity_gate: StressActivityGate;
  }).activity_gate = {
    frame_ms: 20,
    hop_ms: 10,
    rms_threshold: 0.00001,
    min_active_s: 0.05,
    min_coverage: 0.05,
  };
  expectBatchFailure(legacy, expected);
});

test("category timelines match the frozen generator templates", () => {
  for (const scope of SCOPES) {
    const value = makePreview(scope, "WB-wired");
    const expected = {
      scope,
      mode: "WB-wired" as const,
      datasetId: value.dataset_id,
      contracts: value.contracts,
    };
    for (const category of STRESS_CATEGORIES[scope]) {
      const categoryCase = value.cases.find(
        (entry) => entry.provenance.workload_category === category,
      );
      assert.ok(categoryCase);
      assert.deepEqual(
        categoryCase.provenance.activity_segments,
        activitySegments(scope, category),
      );
      const index = value.cases.findIndex(
        (entry) => entry.case_id === categoryCase.case_id,
      );
      const forged = {
        ...value,
        cases: value.cases.map((entry, caseIndex) =>
          caseIndex === index
            ? {
                ...entry,
                provenance: {
                  ...entry.provenance,
                  activity_segments: entry.provenance.activity_segments.map(
                    (segment, segmentIndex) =>
                      segmentIndex === 0
                        ? { ...segment, end_s: segment.end_s - 0.1 }
                        : segment,
                  ),
                },
              }
            : entry
        ),
      };
      expectPreviewFailure(forged, expected);
    }
    const wrongMode = {
      ...value,
      cases: value.cases.map((entry, index) =>
        index === 0
          ? {
              ...entry,
              provenance: { ...entry.provenance, mode: "NB-wired" },
            }
          : entry
      ),
    };
    expectPreviewFailure(wrongMode, expected);
  }
});

test("preview parser rejects cross-scope, count, field, hash, and mode forgery", () => {
  const value = makePreview("echo", "WB-wired");
  const expected = {
    scope: "echo" as const,
    mode: "WB-wired" as const,
    datasetId: value.dataset_id,
    contracts: value.contracts,
  };
  const firstCase = value.cases[0];

  expectPreviewFailure({ ...value, mode: "NB-wired" }, expected);
  expectPreviewFailure({ ...value, test_scope: "double_talk" }, expected);
  expectPreviewFailure(
    {
      ...value,
      category_labels: { ...value.category_labels, steady_echo: "伪造" },
    },
    expected,
  );
  expectPreviewFailure(
    {
      ...value,
      category_counts: { ...value.category_counts, steady_echo: 19 },
    },
    expected,
  );
  expectPreviewFailure({ ...value, cases: value.cases.slice(0, 99) }, expected);
  expectPreviewFailure(
    {
      ...value,
      cases: value.cases.map((entry, index) =>
        index === 0
          ? {
              ...entry,
              conditions: { ...entry.conditions, volume: "maximum" },
            }
          : entry
      ),
    },
    expected,
  );
  expectPreviewFailure(
    {
      ...value,
      cases: value.cases.map((entry, index) =>
        index === 0
          ? { ...entry, families: ["double_talk_classification"] }
          : entry
      ),
    },
    expected,
  );
  expectPreviewFailure(
    {
      ...value,
      cases: value.cases.map((entry, index) =>
        index === 0
          ? { ...entry, contract_item_ids: ["WB-wired:B19"] }
          : entry
      ),
    },
    expected,
  );
  expectPreviewFailure(
    {
      ...value,
      cases: value.cases.map((entry, index) =>
        index === 0
          ? {
              ...entry,
              signals_sha256: {
                ...entry.signals_sha256,
                microphone: "0".repeat(64),
              },
            }
          : entry
      ),
    },
    expected,
  );
  expectPreviewFailure(
    {
      ...value,
      cases: value.cases.map((entry, index) =>
        index === 1
          ? {
              ...entry,
              provenance: {
                ...entry.provenance,
                category_index: firstCase.provenance.category_index,
              },
            }
          : entry
      ),
    },
    expected,
  );
  expectPreviewFailure(
    {
      ...value,
      visuals: {
        ...value.visuals,
        cases: {
          ...value.visuals.cases,
          [firstCase.case_id]: {
            ...value.visuals.cases[firstCase.case_id],
            microphone: {
              ...value.visuals.cases[firstCase.case_id].microphone,
              source_sha256: "0".repeat(64),
            },
          },
        },
      },
    },
    expected,
  );
  assert.throws(() =>
    readStressDatasets(
      {
        ...makeDataset("echo", "WB-wired"),
        datasets: [
          {
            ...makeDataset("echo", "WB-wired").datasets[0],
            dataset_id: "0".repeat(32),
          },
        ],
      },
      { scope: "echo", mode: "WB-wired" },
    )
  );
});

test("batch diagnostics use the nine final metrics and preserve TX evidence binding", () => {
  const preview = readStressPreview(makePreview("echo", "WB-wired"), {
    scope: "echo",
    mode: "WB-wired",
    datasetId: datasetId("echo", "WB-wired"),
    contracts: scopedRows(
      selectedContractRows("echo", "WB-wired"),
    ),
  });
  const batchValue = makeBatch(preview);
  const batch = readStressBatch(batchValue, {
    scope: "echo",
    mode: "WB-wired",
    datasetId: preview.dataset_id,
    algorithmClusterId: ALGORITHM_ID,
    contracts: preview.contracts,
  });
  const diagnostics = batch.cases[0].stress_diagnostics!;
  assert.equal(batch.summary.review_required, 100);
  assert.equal(batch.summary.pass, 0);
  assert.equal(batch.summary.fail, 0);
  assert.deepEqual(
    Object.keys(diagnostics.segments[0].metrics).sort(),
    [
      "primary",
      "tx_residual_reduction_db",
      "convergence_db",
      "cleanref_projection_gain_db",
      "cleanref_correlation",
      "tx_level_dbfs",
      "mic_level_dbfs",
      "far_level_dbfs",
      "near_level_dbfs",
    ].sort(),
  );
  assert.equal(diagnostics.segments[0].mode, "far");
  assert.equal(diagnostics.segments[0].status, "review_required");
  const evidence = stressTxEvidence(
    preview.cases[0],
    batch,
    preview.dataset_manifest_sha256,
  );
  assert.equal(evidence.ready, true);
  assert.match(evidence.url, /outputs\/tx\?v=[a-f0-9]{64}$/);
  assert.equal(
    evidence.visual?.source_sha256,
    batch.cases[0].output_sha256?.tx,
  );
  assert.equal(
    stressTxEvidence(
      preview.cases[0],
      batch,
      "0".repeat(64),
    ).ready,
    false,
  );

  const doubleTalk = readStressPreview(
    makePreview("double_talk", "WB-wired"),
    {
      scope: "double_talk",
      mode: "WB-wired",
      datasetId: datasetId("double_talk", "WB-wired"),
      contracts: scopedRows(
        selectedContractRows("double_talk", "WB-wired"),
      ),
    },
  );
  const doubleBatch = readStressBatch(makeBatch(doubleTalk), {
    scope: "double_talk",
    mode: "WB-wired",
    datasetId: doubleTalk.dataset_id,
    algorithmClusterId: ALGORITHM_ID,
    contracts: doubleTalk.contracts,
  });
  assert.equal(
    doubleBatch.cases[0].stress_diagnostics?.primary.name,
    "double_talk_energy_ratio_proxy_percent",
  );
  assert.equal(
    doubleBatch.cases[0].stress_diagnostics?.segments.some(
      (segment) => segment.mode === "double_talk",
    ),
    true,
  );
});

test("RX-after evidence is bound to the runtime TX reference", () => {
  const preview = readStressPreview(makePreview("echo", "WB-wired"), {
    scope: "echo",
    mode: "WB-wired",
    datasetId: datasetId("echo", "WB-wired"),
    contracts: scopedRows(
      selectedContractRows("echo", "WB-wired"),
    ),
  });
  const expected = {
    scope: "echo" as const,
    mode: "WB-wired" as const,
    datasetId: preview.dataset_id,
    algorithmClusterId: ALGORITHM_ID,
    contracts: preview.contracts,
  };
  const value = makeRxBatch(preview);
  const batch = readStressBatch(value, expected);
  const rx = stressRxEvidence(
    preview.cases[0],
    batch,
    preview.dataset_manifest_sha256,
  );
  assert.equal(rx.ready, true);
  assert.match(rx.url, /outputs\/rx\?v=[a-f0-9]{64}$/);
  assert.equal(
    rx.visual?.source_sha256,
    batch.cases[0].output_sha256?.rx,
  );

  const forged = structuredClone(value);
  const forgedEcho = forged.cases[0].execution!.echo_reference as Record<
    string,
    unknown
  >;
  forgedEcho.processed_sha256 = "0".repeat(64);
  assert.throws(() => readStressBatch(forged, expected));

  const missingVisual = structuredClone(value);
  delete missingVisual.cases[0].rx_visual;
  assert.throws(() => readStressBatch(missingVisual, expected));
});

test("batch parser rejects pass/fail promotion, stale metrics, and forged evidence", () => {
  const preview = readStressPreview(makePreview("echo", "WB-wired"), {
    scope: "echo",
    mode: "WB-wired",
    datasetId: datasetId("echo", "WB-wired"),
    contracts: scopedRows(
      selectedContractRows("echo", "WB-wired"),
    ),
  });
  const value = makeBatch(preview);
  const expected = {
    scope: "echo" as const,
    mode: "WB-wired" as const,
    datasetId: preview.dataset_id,
    algorithmClusterId: ALGORITHM_ID,
    contracts: preview.contracts,
  };
  const mutate = (callback: (copy: typeof value) => void) => {
    const copy = structuredClone(value);
    callback(copy);
    expectBatchFailure(copy, expected);
  };

  mutate((copy) => {
    copy.items[0].status = "pass";
  });
  mutate((copy) => {
    copy.items[0].status = "fail";
  });
  mutate((copy) => {
    copy.summary.pass = 1;
  });
  mutate((copy) => {
    copy.items[0].evidence_binding.batch_id = "b".repeat(32);
  });
  mutate((copy) => {
    const output = copy.items[0].evidence_binding.output_sha256 as
      Record<string, unknown>;
    output.tx = "0".repeat(64);
  });
  mutate((copy) => {
    copy.cases[0].output_sha256!.tx = "0".repeat(64);
  });
  mutate((copy) => {
    const metrics =
      copy.cases[0].stress_diagnostics!.segments[0].metrics as
        Record<string, unknown>;
    delete metrics.primary;
    metrics.bogus_metric = null;
  });
  mutate((copy) => {
    const metrics =
      copy.cases[0].stress_diagnostics!.segments[0].metrics as
        Record<string, unknown>;
    metrics.bogus_extra = null;
  });
  mutate((copy) => {
    (copy.cases[0].stress_diagnostics!.segments[0] as unknown as {
      mode: string;
    }).mode = "quiet";
  });
  mutate((copy) => {
    copy.cases[0].stress_diagnostics!.primary.status = "not_measured";
  });
  mutate((copy) => {
    (copy.cases[0].stress_diagnostics as unknown as {
      status: string;
    }).status = "not_applicable";
    copy.cases[0].stress_diagnostics!.review_required = false;
  });
  mutate((copy) => {
    copy.cases[0].stress_diagnostics!.tx_duration_s = 1;
  });
  mutate((copy) => {
    copy.cases[0].stress_diagnostics!.tx_muted = true;
  });
  mutate((copy) => {
    copy.stress_scope = "double_talk";
  });
  mutate((copy) => {
    copy.cases[0].provenance.workload_category = "balanced";
  });
  mutate((copy) => {
    copy.dataset_id = "0".repeat(32);
  });
});

test("single-case stress batches keep source totals and one selected category", () => {
  const preview = readStressPreview(makePreview("echo", "WB-wired"), {
    scope: "echo",
    mode: "WB-wired",
    datasetId: datasetId("echo", "WB-wired"),
    contracts: scopedRows(
      selectedContractRows("echo", "WB-wired"),
    ),
  });
  const selectedCase = preview.cases[0];
  const single = makeSingleBatch(preview, selectedCase.case_id);
  const expected = {
    scope: "echo" as const,
    mode: "WB-wired" as const,
    datasetId: preview.dataset_id,
    algorithmClusterId: ALGORITHM_ID,
    contracts: preview.contracts,
    caseId: selectedCase.case_id,
  };
  const parsed = readStressBatch(single, expected);
  assert.equal(parsed.execution_scope, "single_case");
  assert.equal(parsed.selected_case_id, selectedCase.case_id);
  assert.equal(parsed.case_count, 1);
  assert.equal(parsed.observation_count, 1);
  assert.equal(parsed.preview_condition_group_count, 100);
  assert.equal(parsed.dataset_case_count, 100);
  assert.equal(parsed.dataset_observation_count, 100);
  assert.equal(parsed.contract_item_count, 2);
  assert.deepEqual(
    parsed.progress,
    {
      completed_cases: 1,
      total_cases: 1,
      completed_items: 1,
      total_items: 1,
    },
  );
  assert.deepEqual(parsed.cases.map((entry) => entry.case_id), [
    selectedCase.case_id,
  ]);
  assert.deepEqual(parsed.items.map((entry) => entry.case_id), [
    selectedCase.case_id,
  ]);
  assert.deepEqual(
    parsed.category_counts,
    Object.fromEntries(
      STRESS_CATEGORIES.echo.map((category) => [
        category,
        category === selectedCase.provenance.workload_category ? 1 : 0,
      ]),
    ),
  );
  assert.equal(
    stressTxEvidence(selectedCase, parsed, preview.dataset_manifest_sha256).ready,
    true,
  );
  const otherCase = preview.cases.find(
    (entry) => entry.case_id !== selectedCase.case_id,
  )!;
  assert.equal(
    stressTxEvidence(otherCase, parsed, preview.dataset_manifest_sha256).ready,
    false,
  );

  const historyLike = readStressBatch(single, {
    scope: "echo",
    mode: "WB-wired",
    datasetId: preview.dataset_id,
    algorithmClusterId: ALGORITHM_ID,
    contracts: preview.contracts,
  });
  assert.equal(historyLike.execution_scope, "single_case");
});

test("single-case stress selection is all-or-none and rejects forged totals", () => {
  const preview = readStressPreview(makePreview("double_talk", "WB-wired"), {
    scope: "double_talk",
    mode: "WB-wired",
    datasetId: datasetId("double_talk", "WB-wired"),
    contracts: scopedRows(
      selectedContractRows("double_talk", "WB-wired"),
    ),
  });
  const selectedCase = preview.cases[0];
  const single = makeSingleBatch(preview, selectedCase.case_id);
  const expected = {
    scope: "double_talk" as const,
    mode: "WB-wired" as const,
    datasetId: preview.dataset_id,
    algorithmClusterId: ALGORITHM_ID,
    contracts: preview.contracts,
  };
  const full = makeBatch(preview);
  assert.throws(
    () => readStressBatch(full, { ...expected, caseId: selectedCase.case_id }),
    /缺少单案选择字段/,
  );
  assert.throws(
    () => readStressBatch(single, {
      ...expected,
      caseId: "different-case",
    }),
    /请求 case_id 不一致/,
  );
  for (const field of [
    "execution_scope",
    "selected_case_id",
    "dataset_case_count",
    "dataset_observation_count",
  ] as const) {
    const partial = structuredClone(single);
    delete partial[field];
    assert.throws(
      () => readStressBatch(partial, expected),
      /四项同时存在/,
    );
  }

  const mutations: Array<(value: StressBatch) => void> = [
    (value) => {
      (value as unknown as { execution_scope: string }).execution_scope =
        "full";
    },
    (value) => { value.selected_case_id = "different-case"; },
    (value) => { value.dataset_case_count = 99; },
    (value) => { value.dataset_observation_count = 99; },
    (value) => { value.case_count = 100; },
    (value) => { value.observation_count = 100; },
    (value) => { value.preview_condition_group_count = 1; },
    (value) => { value.progress.total_cases = 100; },
    (value) => { value.progress.total_items = 100; },
    (value) => { value.summary.total = 100; },
    (value) => { value.cases.push(structuredClone(full.cases[1])); },
    (value) => { value.items.push(structuredClone(full.items[1])); },
    (value) => { value.items[0].case_id = "different-case"; },
    (value) => {
      const category = value.cases[0].provenance.workload_category;
      value.category_counts[category] = 0;
    },
    (value) => {
      delete value.category_counts[
        STRESS_CATEGORIES.double_talk[1]
      ];
    },
    (value) => { value.cases[0].status = "pending"; },
    (value) => { value.items[0].status = "pending"; value.items[0].value = null; },
  ];
  for (const mutate of mutations) {
    const value = structuredClone(single);
    mutate(value);
    assert.throws(() => readStressBatch(value, expected));
  }

  const forgedFull = structuredClone(full);
  Object.assign(forgedFull, {
    execution_scope: "single_case",
    selected_case_id: selectedCase.case_id,
    dataset_case_count: 100,
    dataset_observation_count: 100,
  });
  assert.throws(() => readStressBatch(forgedFull, expected));

  const deletedRows = structuredClone(full);
  deletedRows.cases = [deletedRows.cases[0]];
  deletedRows.items = [deletedRows.items[0]];
  assert.throws(() => readStressBatch(deletedRows, expected));
});


test("stressDynamicMicrophoneEvidence validates runtime post-RX microphone and NB8 beamforming", () => {
  const preview = readStressPreview(makePreview("echo", "WB-wired"), {
    scope: "echo",
    mode: "WB-wired",
    datasetId: datasetId("echo", "WB-wired"),
    contracts: scopedRows(selectedContractRows("echo", "WB-wired")),
  });
  const selectedCase = preview.cases[0];
  const expected = {
    scope: "echo" as const,
    mode: "WB-wired" as const,
    datasetId: preview.dataset_id,
    algorithmClusterId: ALGORITHM_ID,
    contracts: preview.contracts,
    caseId: selectedCase.case_id,
  };

  // Historical TX-only batch stays readable and dynamic mic evidence is not ready
  const legacySingle = makeSingleBatch(preview, selectedCase.case_id);
  const parsedLegacy = readStressBatch(legacySingle, expected);
  const legacyEvidence = stressDynamicMicrophoneEvidence(
    selectedCase,
    parsedLegacy,
    preview.dataset_manifest_sha256,
  );
  assert.equal(legacyEvidence.ready, false);

  // Batch with RX but without dynamic_microphone stays readable and reports missing dynamic mic
  const withRxOnly = structuredClone(legacySingle) as Record<string, any>;
  const rxSha = sha("rx-after-wb-echo-01");
  withRxOnly.cases[0].output_sha256 = {
    tx: withRxOnly.cases[0].output_sha256.tx,
    rx: rxSha,
  };
  withRxOnly.cases[0].rx_visual = {
    ...withRxOnly.cases[0].tx_visual,
    source_sha256: rxSha,
  };
  withRxOnly.cases[0].execution = {
    ...(withRxOnly.cases[0].execution ?? {}),
    echo_reference: {
      source: "vcp_rx_processed",
      processed_output: "rx",
      processed_sha256: rxSha,
      tx_echo_argument_sha256: rxSha,
    },
  };
  withRxOnly.items[0].evidence_binding.output_sha256 = {
    tx: withRxOnly.cases[0].output_sha256.tx,
    rx: rxSha,
  };
  const parsedRxOnly = readStressBatch(withRxOnly, expected);
  const rxOnlyEvidence = stressDynamicMicrophoneEvidence(
    selectedCase,
    parsedRxOnly,
    preview.dataset_manifest_sha256,
  );
  assert.equal(rxOnlyEvidence.ready, false);
  assert.match(rxOnlyEvidence.reason, /未记录运行时合成麦克风/);

  // Batch with dynamic_microphone + visual + effective_input_sha256 passes and resolves URL
  const withDynamic = structuredClone(withRxOnly) as Record<string, any>;
  const dynSha = sha("dynamic-mic-wb-echo-01");
  withDynamic.cases[0].dynamic_microphone = {
    source: "post_rx_runtime_mix",
    path: "/tmp/tx.dynamic-microphone.wav",
    sha256: dynSha,
    sample_rate: 16_000,
    channels: 2,
    frames: 192_000,
    capture_gain: 0.94,
    playback_sha256: rxSha,
  };
  withDynamic.cases[0].dynamic_microphone_visual = {
    source_sha256: dynSha,
    duration_s: 12,
    sample_rate: 16_000,
    channels: 2,
    peak: 0.42,
    wave_scale: 1,
    wave_min_i16: WAVE_BASE64,
    wave_max_i16: WAVE_BASE64,
    mel_i16: MEL_BASE64,
    mel_ceiling_db: -18,
  };
  withDynamic.cases[0].effective_input_sha256 = {
    microphone: dynSha,
    reference: withDynamic.cases[0].signals_sha256.reference,
    noise: withDynamic.cases[0].signals_sha256.noise,
    far_end: rxSha,
  };
  const parsedDynamic = readStressBatch(withDynamic, expected);
  const dynamicEvidence = stressDynamicMicrophoneEvidence(
    selectedCase,
    parsedDynamic,
    preview.dataset_manifest_sha256,
  );
  assert.equal(dynamicEvidence.ready, true);
  assert.equal(dynamicEvidence.sha256, dynSha);
  assert.equal(dynamicEvidence.visual?.source_sha256, dynSha);
  assert.ok(
    dynamicEvidence.url.includes(
      `/cases/${encodeURIComponent(selectedCase.case_id)}/inputs/dynamic-microphone?v=${dynSha}`,
    ),
  );

  // NB8 native path: echo_reference.source=native_nb8_rx_processed and beamformed effective mic hash
  const withNb8Beam = structuredClone(withDynamic) as Record<string, any>;
  const bfSha = sha("nb8-beamformed-mic");
  withNb8Beam.cases[0].execution.echo_reference.source = "native_nb8_rx_processed";
  withNb8Beam.cases[0].execution.native_preprocessing = {
    beamforming: {
      source_sha256: dynSha,
      output_sha256: bfSha,
    },
  };
  withNb8Beam.cases[0].effective_input_sha256.microphone = bfSha;
  const parsedNb8 = readStressBatch(withNb8Beam, expected);
  assert.equal(
    stressDynamicMicrophoneEvidence(
      selectedCase,
      parsedNb8,
      preview.dataset_manifest_sha256,
    ).ready,
    true,
  );

  // Tampered visual hash fails closed
  const badVisual = structuredClone(withDynamic) as Record<string, any>;
  badVisual.cases[0].dynamic_microphone_visual.source_sha256 = sha("tampered");
  assert.throws(() => readStressBatch(badVisual, expected));

  // Orphan visual without dynamic_microphone fails closed
  const orphanVisual = structuredClone(withDynamic) as Record<string, any>;
  delete orphanVisual.cases[0].dynamic_microphone;
  assert.throws(() => readStressBatch(orphanVisual, expected));
});
