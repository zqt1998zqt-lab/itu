import {
  FIVE_SIGNAL_MODES,
  type FiveSignalMode,
} from "./five-signal-flow-model.ts";
import {
  SCOPED_INPUT_SIGNAL_ROLES,
  SCOPED_STATUS_LABELS,
  readScopedCatalog,
  readSingleCaseSelection,
  type CompleteScopedPreviewVisuals,
  type ScopedInputSignalRole,
  type ScopedItem,
  type ScopedPreviewVisualTrack,
  type ScopedProgress,
  type ScopedSummary,
} from "./scoped-acoustic-model.ts";

export const STRESS_SCOPES = ["echo", "double_talk"] as const;
export type StressScope = (typeof STRESS_SCOPES)[number];

export const STRESS_CATEGORIES = {
  echo: [
    "steady_echo",
    "convergence",
    "delay_change",
    "path_change",
    "noisy_echo",
  ],
  double_talk: [
    "balanced",
    "near_dominant",
    "far_dominant",
    "transition",
    "short_overlap",
  ],
} as const;

export const STRESS_CATEGORY_LABELS = {
  echo: {
    steady_echo: "稳定远端单讲",
    convergence: "启动与收敛",
    delay_change: "回声时延变化",
    path_change: "回声路径变化",
    noisy_echo: "噪声中的回声",
  },
  double_talk: {
    balanced: "均衡持续双讲",
    near_dominant: "近端较强双讲",
    far_dominant: "远端较强双讲",
    transition: "单讲 / 双讲切换",
    short_overlap: "短时重叠双讲",
  },
} as const;

export type StressCategory =
  (typeof STRESS_CATEGORIES)[StressScope][number];
export type StressActivityLabel =
  | "quiet"
  | "far_only"
  | "near_only"
  | "double_talk"
  | "noise_only";
export type StressDiagnosticStatus =
  | "review_required"
  | "not_measurable"
  | "not_applicable";
export type StressDiagnosticAggregateStatus =
  | "review_required"
  | "not_measurable"
  | "not_applicable";
export type StressActivitySegment = {
  start_s: number;
  end_s: number;
  label: StressActivityLabel;
};

export type StressProvenance = {
  scope: StressScope;
  mode: FiveSignalMode;
  scene: "baseline";
  workload_category: StressCategory;
  category_index: number;
  activity_segments: StressActivitySegment[];
  conversation_mode: "far_single" | "transition" | "double_talk";
  analysis_start_s: number;
  engineering_preview: true;
  engineering_combination: true;
  independent_source_recordings: false;
  actual_vehicle_recording: false;
  within_case_source_looping: false;
  [key: string]: unknown;
};

export type StressPreviewCase = {
  case_id: string;
  conditions: {
    mode: FiveSignalMode;
    direction: string;
    scene: "baseline";
    volume: string | number | null;
    window_ms: number | null;
    input_level_db: number | null;
  };
  families: string[];
  contract_item_ids: [string];
  provenance: StressProvenance;
  signals: ScopedInputSignalRole[];
  signals_sha256: Record<ScopedInputSignalRole, string>;
};

export type StressDataset = {
  dataset_id: string;
  dataset_name: string;
  test_scope: StressScope;
  mode: FiveSignalMode;
  scene: "baseline";
  contract_item_count: number;
  preview_condition_group_count: 100;
  observation_count: 100;
  category_counts: Record<string, number>;
  category_labels: Record<StressCategory, string>;
  standard_compliance: false;
  formal_test_status: "not_measured";
};

export type StressDatasetView = {
  dataset_id: string;
  dataset_name: string;
  test_scope: StressScope;
  mode: FiveSignalMode;
  scene: "baseline";
  contract_item_count: number;
  preview_condition_group_count: 100;
  observation_count: 100;
  category_counts: Record<string, number>;
  category_labels: Record<StressCategory, string>;
  contracts: ScopedItem[];
  datasets: StressDataset[];
  standard_compliance: false;
  formal_test_status: "not_measured";
};

export type StressPreview = {
  dataset_id: string;
  dataset_name: string;
  dataset_manifest_sha256: string;
  dataset_assets_sha256: string;
  test_scope: StressScope;
  mode: FiveSignalMode;
  scene: "baseline";
  schema_version: "acoustic-stress-preview-v1";
  dataset_kind: "acoustic_stress_preview";
  contract_item_count: number;
  preview_condition_group_count: 100;
  observation_count: 100;
  category_counts: Record<string, number>;
  category_labels: Record<StressCategory, string>;
  contracts: ScopedItem[];
  cases: StressPreviewCase[];
  visuals: CompleteScopedPreviewVisuals;
  standard_compliance: false;
  formal_test_status: "not_measured";
};

export type StressDiagnosticMetrics = {
  primary: number | null;
  tx_residual_reduction_db: number | null;
  convergence_db: number | null;
  cleanref_projection_gain_db: number | null;
  cleanref_correlation: number | null;
  tx_level_dbfs: number | null;
  mic_level_dbfs: number | null;
  far_level_dbfs: number | null;
  near_level_dbfs: number | null;
};

export type StressActivityGate = {
  frame_ms: 20;
  hop_ms: 10;
  rms_threshold: 0.00001;
  min_active_s: 0.05;
  min_coverage: 0.05;
};

export type StressSegmentActivity = {
  frame_ms: 20;
  hop_ms: 10;
  rms_threshold: 0.00001;
  min_active_s: 0.05;
  min_coverage: 0.05;
  near_active_samples: number;
  far_active_samples: number;
  joint_active_samples: number;
  near_coverage: number;
  far_coverage: number;
  joint_coverage: number;
  valid: boolean;
  reason: string;
  echo_delay_guard_ms?: 100;
  echo_guarded_active_samples?: number;
  echo_guarded_active_s?: number;
};

export type StressDiagnosticSegmentBase = {
  label: StressActivityLabel;
  start_s: number;
  end_s: number;
  mode: "not_applicable" | "far" | "near" | "double_talk" | "noise";
  status: StressDiagnosticStatus;
  metrics: StressDiagnosticMetrics;
  reason: string;
};

export type StressDiagnosticSegmentV1 = StressDiagnosticSegmentBase;
export type StressDiagnosticSegmentV2 = StressDiagnosticSegmentBase & {
  activity: StressSegmentActivity;
};
export type StressDiagnosticSegment =
  | StressDiagnosticSegmentV1
  | StressDiagnosticSegmentV2;

type StressDiagnosticsCommon = {
  case_id: string;
  tx_sha256: string;
  scope: StressScope;
  mode: FiveSignalMode;
  category: StressCategory;
  evidence_grade: "engineering_proxy";
  formal_compliance: false;
  standard_compliance: false;
  formal_test_status: "not_measured";
  status: StressDiagnosticAggregateStatus;
  reason: string;
  review_required: boolean;
  sample_rate_hz: number;
  source_duration_s: number;
  tx_duration_s: number;
  tx_muted: boolean;
  primary: {
    name:
      | "echo_primary_proxy_db"
      | "double_talk_energy_ratio_proxy_percent";
    value: number | null;
    unit: "dB" | "percent";
    status: "review_required" | "not_measured";
  };
  convergence_db: number | null;
  limitations: string[];
};

export type StressDiagnosticsV1 = StressDiagnosticsCommon & {
  schema_version: "acoustic-stress-diagnostics-v1";
  segments: StressDiagnosticSegmentV1[];
  activity_gate?: never;
  convergence_note?: never;
};

export type StressDiagnosticsV2 = StressDiagnosticsCommon & {
  schema_version: "acoustic-stress-diagnostics-v2";
  activity_gate: StressActivityGate;
  convergence_note?: string;
  segments: StressDiagnosticSegmentV2[];
};

export type StressDiagnostics = StressDiagnosticsV1 | StressDiagnosticsV2;

export type StressBatchCase = {
  case_id: string;
  status: "pending" | "running" | "completed" | "failed";
  report_id: string | null;
  conditions: StressPreviewCase["conditions"];
  contract_item_ids: [string];
  provenance: StressProvenance;
  signals?: Record<ScopedInputSignalRole, string>;
  signals_sha256: Record<ScopedInputSignalRole, string>;
  input_sha256: Record<ScopedInputSignalRole, string>;
  output_sha256?: { tx: string; rx?: string };
  tx_visual?: ScopedPreviewVisualTrack;
  rx_visual?: ScopedPreviewVisualTrack;
  stress_diagnostics?: StressDiagnostics;
  assessment_sha256?: string;
  execution?: Record<string, unknown>;
  error?: string;
};

export type StressObservation = ScopedItem & {
  contract_id: string;
  item_id: string;
  observation_id: string;
  case_id: string;
  mode: FiveSignalMode;
  status: keyof typeof SCOPED_STATUS_LABELS;
  value: number | null;
  metric: string | null;
  reason: string;
  evidence_binding: Record<string, unknown>;
};

export type StressBatch = {
  schema_version: "scene-condition-batch-v1";
  batch_id: string;
  dataset_id: string;
  dataset_name: string;
  dataset_manifest_sha256: string;
  dataset_assets_sha256: string;
  algorithm_cluster_id: string;
  configuration: Record<string, unknown>;
  catalog_sha256: string;
  cli_sha256: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  created_at: string;
  updated_at: string;
  test_scope: StressScope;
  stress_scope?: StressScope;
  selected_mode: FiveSignalMode;
  selected_scene: "baseline";
  standard_compliance: false;
  formal_test_status: "not_measured";
  contract_item_count: number;
  case_count: number;
  preview_condition_group_count: number;
  observation_count: number;
  execution_scope?: "single_case";
  selected_case_id?: string;
  dataset_case_count?: number;
  dataset_observation_count?: number;
  category_counts: Record<string, number>;
  progress: ScopedProgress;
  summary: ScopedSummary;
  cases: StressBatchCase[];
  items: StressObservation[];
  error?: string | null;
};

export type StressTxEvidence = {
  ready: boolean;
  url: string;
  visual: ScopedPreviewVisualTrack | undefined;
  reason: string;
};

export type StressRxEvidence = StressTxEvidence;

export const STRESS_DATASETS_URL =
  "/api/alignment/acceptance/stress-datasets";
export const STRESS_PREVIEW_URL =
  "/api/alignment/acceptance/stress-datasets";
export const STRESS_BATCHES_URL =
  "/api/alignment/acceptance/scene-condition-batches";
export const STRESS_SCHEMA_VERSION = "acoustic-stress-preview-v1" as const;
export const STRESS_DATASET_KIND = "acoustic_stress_preview" as const;
export const STRESS_DIAGNOSTICS_SCHEMA_V1 =
  "acoustic-stress-diagnostics-v1" as const;
export const STRESS_DIAGNOSTICS_SCHEMA_V2 =
  "acoustic-stress-diagnostics-v2" as const;
export const STRESS_DIAGNOSTICS_SCHEMA = STRESS_DIAGNOSTICS_SCHEMA_V2;
export const STRESS_CASE_COUNT = 100 as const;
export const STRESS_OBSERVATION_COUNT = 100 as const;
export const STRESS_SOURCE_DURATION_S = 12 as const;
export const STRESS_DATASET_NAME_SUFFIX = "100-workloads-v1" as const;
export const STRESS_ACTIVITY_FRAME_MS = 20 as const;
export const STRESS_ACTIVITY_HOP_MS = 10 as const;
export const STRESS_ACTIVITY_RMS_THRESHOLD = 0.00001 as const;
export const STRESS_MIN_ACTIVE_S = 0.05 as const;
export const STRESS_MIN_ACTIVITY_COVERAGE = 0.05 as const;
export const STRESS_ECHO_DELAY_GUARD_MS = 100 as const;

const ID = /^[a-f0-9]{32}$/;
const SHA256 = /^[a-f0-9]{64}$/;
const BASE64 =
  /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/;
const STRESS_VISUAL_SCHEMA = "scoped-acoustic-visual-v1";
const WAVEFORM_BINS = 720;
const MEL_BANDS = 40;
const MEL_FRAMES = 240;
const MEL_FFT_SIZE = 512;
const MEL_HOP_MS = 10;
const MEL_DYNAMIC_RANGE_DB = 80;
const MEL_DB_OFFSET_DB = -200;
const MEL_DB_STEP_DB = 0.1;
const DIAGNOSTIC_STATUSES = new Set<string>([
  "review_required",
  "not_measurable",
  "not_applicable",
]);
const ITEM_STATUSES = new Set<string>([
  "pending",
  "running",
  "review_required",
  "not_measured",
  "invalid",
]);
const BATCH_STATUSES = new Set<string>([
  "queued",
  "running",
  "completed",
  "failed",
  "cancelled",
]);
const METRIC_KEYS = [
  "primary",
  "tx_residual_reduction_db",
  "convergence_db",
  "cleanref_projection_gain_db",
  "cleanref_correlation",
  "tx_level_dbfs",
  "mic_level_dbfs",
  "far_level_dbfs",
  "near_level_dbfs",
] as const;
const SEGMENT_MODE = {
  quiet: "not_applicable",
  far_only: "far",
  near_only: "near",
  double_talk: "double_talk",
  noise_only: "noise",
} as const;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function assert(
  condition: unknown,
  message: string,
): asserts condition {
  if (!condition) throw new Error(message);
}

function isCount(value: unknown): value is number {
  return (
    typeof value === "number"
    && Number.isInteger(value)
    && value >= 0
  );
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isFiniteJson(value: unknown, depth = 0): boolean {
  if (depth > 32) return false;
  if (
    value === null
    || typeof value === "string"
    || typeof value === "boolean"
  ) {
    return true;
  }
  if (typeof value === "number") return Number.isFinite(value);
  if (Array.isArray(value)) {
    return value.every((entry) => isFiniteJson(entry, depth + 1));
  }
  if (isRecord(value)) {
    return Object.values(value).every((entry) =>
      isFiniteJson(entry, depth + 1)
    );
  }
  return false;
}

function readId(value: unknown, message: string): string {
  assert(typeof value === "string" && ID.test(value), message);
  return value;
}

function readSha(value: unknown, message: string): string {
  assert(typeof value === "string" && SHA256.test(value), message);
  return value;
}

function readTimestamp(value: unknown, message: string): string {
  assert(
    typeof value === "string" && Number.isFinite(Date.parse(value)),
    message,
  );
  return value;
}

function sameStrings(
  left: readonly unknown[],
  right: readonly string[],
): boolean {
  return (
    left.length === right.length
    && left.every((entry, index) => entry === right[index])
  );
}

function sameInputHashes(
  left: Record<ScopedInputSignalRole, string>,
  right: Record<ScopedInputSignalRole, string>,
): boolean {
  return SCOPED_INPUT_SIGNAL_ROLES.every(
    (role) => left[role] === right[role],
  );
}

function sameConditions(
  left: StressPreviewCase["conditions"],
  right: StressPreviewCase["conditions"],
): boolean {
  return (
    left.mode === right.mode
    && left.direction === right.direction
    && left.scene === right.scene
    && left.volume === right.volume
    && left.window_ms === right.window_ms
    && left.input_level_db === right.input_level_db
  );
}

function stressCategories(scope: StressScope): readonly StressCategory[] {
  return STRESS_CATEGORIES[scope];
}

export function stressSampleRate(mode: FiveSignalMode): number {
  if (mode.startsWith("SWB-")) return 48_000;
  if (mode.startsWith("NB-")) return 8_000;
  if (mode.startsWith("WB-")) return 16_000;
  throw Error("不支持的 stress 业务模式");
}

function stressDatasetName(
  scope: StressScope,
  mode: FiveSignalMode,
): string {
  return `${scope}-${mode}-${STRESS_DATASET_NAME_SUFFIX}`;
}

function expectedContractCells(
  scope: StressScope,
  mode: FiveSignalMode,
): readonly string[] {
  if (scope === "double_talk") {
    return mode.startsWith("SWB-")
      ? ["B18", "B19"]
      : ["B19", "B20"];
  }
  return mode.startsWith("SWB-")
    ? ["B20", "B21", "B22"]
    : ["B21", "B22"];
}

function expectedContractFamily(scope: StressScope): string {
  return scope === "echo"
    ? "tclw"
    : "double_talk_classification";
}

function readContracts(
  value: unknown,
  scope: StressScope,
  mode: FiveSignalMode,
): ScopedItem[] {
  assert(Array.isArray(value), "stress contracts 必须为列表");
  const cells = expectedContractCells(scope, mode);
  assert(
    value.length === cells.length,
    "stress contracts 数量与真实合同不一致",
  );
  const catalog = readScopedCatalog({
    test_scope: scope,
    mode,
    item_count: value.length,
    items: value,
  }, scope, mode);
  const family = expectedContractFamily(scope);
  const expectedIds = new Set(
    cells.map((cell) => `${mode}:${cell}`),
  );
  const byId = new Map(catalog.items.map((row) => [row.id, row]));
  assert(
    byId.size === cells.length
      && [...expectedIds].every((id) => byId.has(id)),
    "stress contracts 未绑定真实 B18–B22 合同行",
  );
  for (const [index, cell] of cells.entries()) {
    const row = byId.get(`${mode}:${cell}`);
    assert(
      row !== undefined
        && row.source_cell === cell
        && row.family === family
        && row.direction === "send",
      `stress contracts ${cells[index]} 不是真实 scope 行`,
    );
  }
  return cells.map((cell) => byId.get(`${mode}:${cell}`)!);
}

function requiredConditions(
  contract: ScopedItem,
  mode: FiveSignalMode,
): StressPreviewCase["conditions"] {
  const name = contract.test_name.toLowerCase();
  const scene = name.match(/sce\s*([1-4])/);
  const window = name.match(/(50|100)\s*ms/);
  const volume = name.includes("recommended")
    ? "recommended"
    : /\b(maximum|max)\b/.test(name)
      ? "maximum"
      : "nominal";
  const compactName = name.replaceAll(" ", "");
  const inputLevel = compactName.includes("-5db")
    ? -5
    : compactName.includes("+5db")
      ? 5
      : 0;
  return {
    mode,
    direction: contract.direction,
    scene: scene ? (`sce${scene[1]}` as StressPreviewCase["conditions"]["scene"]) : "baseline",
    volume,
    window_ms: window ? Number(window[1]) : null,
    input_level_db: inputLevel,
  };
}

function readConditions(
  value: unknown,
  contract: ScopedItem,
  mode: FiveSignalMode,
): StressPreviewCase["conditions"] {
  assert(isRecord(value), "stress conditions 必须为对象");
  const keys = Object.keys(value).sort();
  assert(
    keys.length === 6
      && keys[0] === "direction"
      && keys[1] === "input_level_db"
      && keys[2] === "mode"
      && keys[3] === "scene"
      && keys[4] === "volume"
      && keys[5] === "window_ms",
    "stress conditions 字段不完整",
  );
  const expected = requiredConditions(contract, mode);
  assert(
    value.mode === expected.mode
      && value.direction === expected.direction
      && value.scene === expected.scene
      && value.volume === expected.volume
      && value.window_ms === expected.window_ms
      && value.input_level_db === expected.input_level_db,
    "stress conditions 与真实合同 required_conditions 不一致",
  );
  return {
    mode: expected.mode,
    direction: expected.direction,
    scene: "baseline",
    volume: expected.volume,
    window_ms: expected.window_ms,
    input_level_db: expected.input_level_db,
  };
}

function readCategoryCounts(
  value: unknown,
  scope: StressScope,
  selectedCategory?: StressCategory,
): Record<string, number> {
  assert(isRecord(value), "stress category_counts 必须为对象");
  const categories = stressCategories(scope);
  const keys = Object.keys(value).sort();
  const expectedKeys = [...categories].sort();
  assert(
    keys.length === expectedKeys.length
      && keys.every((key, index) => key === expectedKeys[index])
      && (selectedCategory === undefined
        ? categories.every((category) => value[category] === 20)
        : categories.includes(selectedCategory)
          && categories.every(
            (category) =>
              value[category] === (category === selectedCategory ? 1 : 0),
          )),
    selectedCategory === undefined
      ? "stress category_counts 必须每类恰好 20"
      : "single_case stress category_counts 必须仅所选类别为 1",
  );
  return Object.fromEntries(
    categories.map((category) => [
      category,
      selectedCategory === undefined
        ? 20
        : Number(category === selectedCategory),
    ]),
  );
}

function readCategoryLabels(
  value: unknown,
  scope: StressScope,
): Record<StressCategory, string> {
  assert(isRecord(value), "stress category_labels 必须为对象");
  const categories = stressCategories(scope);
  const expected = STRESS_CATEGORY_LABELS[scope] as Record<
    StressCategory,
    string
  >;
  const keys = Object.keys(value).sort();
  const expectedKeys = [...categories].sort();
  assert(
    keys.length === expectedKeys.length
      && keys.every((key, index) => key === expectedKeys[index])
      && categories.every(
        (category) => value[category] === expected[category],
      ),
    "stress category_labels 与 scope 不一致",
  );
  return Object.fromEntries(
    categories.map((category) => [category, expected[category]]),
  ) as Record<StressCategory, string>;
}

function expectedActivitySegments(
  scope: StressScope,
  category: StressCategory,
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

function readActivitySegments(
  value: unknown,
  scope: StressScope,
  category: StressCategory,
): StressActivitySegment[] {
  const expected = expectedActivitySegments(scope, category);
  assert(
    Array.isArray(value) && value.length === expected.length,
    "stress activity_segments 与 category 时间线不一致",
  );
  for (const [index, raw] of value.entries()) {
    const segment = expected[index];
    assert(
      isRecord(raw)
        && Object.keys(raw).length === 3
        && raw.label === segment.label
        && isFiniteNumber(raw.start_s)
        && isFiniteNumber(raw.end_s)
        && Math.abs(raw.start_s - segment.start_s) <= 1e-6
        && Math.abs(raw.end_s - segment.end_s) <= 1e-6,
      `stress activity_segments[${index}] 与 category 模板不一致`,
    );
  }
  return expected.map((segment) => ({ ...segment }));
}

function readStressProvenance(
  value: unknown,
  scope: StressScope,
  mode: FiveSignalMode,
): StressProvenance {
  assert(isRecord(value), "stress provenance 必须为对象");
  stressSampleRate(mode);
  const category = value.workload_category;
  assert(
    typeof category === "string"
      && stressCategories(scope).includes(category as StressCategory),
    "stress workload_category 与 scope 不一致",
  );
  assert(
    isCount(value.category_index)
      && value.category_index >= 1
      && value.category_index <= 20,
    "stress category_index 必须为 1..20",
  );
  const expectedConversationMode = scope === "echo"
    ? "far_single"
    : category === "transition"
      ? "transition"
      : "double_talk";
  const expectedAnalysisStart = [
    "convergence",
    "transition",
    "short_overlap",
  ].includes(category)
    ? 0.5
    : 0;
  assert(
    value.scope === scope
      && value.mode === mode
      && value.scene === "baseline"
      && value.conversation_mode === expectedConversationMode
      && isFiniteNumber(value.analysis_start_s)
      && Math.abs(value.analysis_start_s - expectedAnalysisStart) <= 1e-6
      && value.engineering_preview === true
      && value.engineering_combination === true
      && value.independent_source_recordings === false
      && value.actual_vehicle_recording === false
      && value.within_case_source_looping === false,
    "stress provenance scope/mode/category 或工程边界无效",
  );
  return {
    ...value,
    scope,
    mode,
    scene: "baseline",
    workload_category: category as StressCategory,
    category_index: value.category_index,
    activity_segments: readActivitySegments(
      value.activity_segments,
      scope,
      category as StressCategory,
    ),
    conversation_mode: expectedConversationMode,
    analysis_start_s: expectedAnalysisStart,
    engineering_preview: true,
    engineering_combination: true,
    independent_source_recordings: false,
    actual_vehicle_recording: false,
    within_case_source_looping: false,
  };
}

function readInputHashes(
  value: unknown,
  message: string,
): Record<ScopedInputSignalRole, string> {
  assert(isRecord(value), message);
  assert(
    Object.keys(value).length === SCOPED_INPUT_SIGNAL_ROLES.length,
    message,
  );
  const hashes = {} as Record<ScopedInputSignalRole, string>;
  for (const role of SCOPED_INPUT_SIGNAL_ROLES) {
    hashes[role] = readSha(value[role], message);
  }
  return hashes;
}

function readVisualTrack(
  value: unknown,
  expectedSha256: string,
  mode: FiveSignalMode,
  role: ScopedInputSignalRole,
): ScopedPreviewVisualTrack {
  assert(isRecord(value), "stress TX 可视化格式无效");
  assert(
    value.source_sha256 === expectedSha256
      && value.duration_s === STRESS_SOURCE_DURATION_S
      && value.sample_rate === stressSampleRate(mode)
      && value.channels === (role === "microphone" ? 2 : 1)
      && isFiniteNumber(value.peak)
      && value.peak >= 0
      && isFiniteNumber(value.wave_scale)
      && value.wave_scale >= 1
      && isFiniteNumber(value.mel_ceiling_db)
      && typeof value.wave_min_i16 === "string"
      && typeof value.wave_max_i16 === "string"
      && typeof value.mel_i16 === "string"
      && BASE64.test(value.wave_min_i16)
      && BASE64.test(value.wave_max_i16)
      && BASE64.test(value.mel_i16)
      && value.wave_min_i16.length === Math.ceil((WAVEFORM_BINS * 2) / 3) * 4
      && value.wave_max_i16.length === Math.ceil((WAVEFORM_BINS * 2) / 3) * 4
      && value.mel_i16.length === Math.ceil((MEL_BANDS * MEL_FRAMES * 2) / 3) * 4,
    "stress 可视化波形、采样率、通道数或数组无效",
  );
  return {
    source_sha256: expectedSha256,
    duration_s: value.duration_s,
    sample_rate: value.sample_rate,
    channels: value.channels as 1 | 2,
    peak: value.peak,
    wave_scale: value.wave_scale,
    wave_min_i16: value.wave_min_i16,
    wave_max_i16: value.wave_max_i16,
    mel_i16: value.mel_i16,
    mel_ceiling_db: value.mel_ceiling_db,
  };
}

function readVisuals(
  value: unknown,
  cases: StressPreviewCase[],
  mode: FiveSignalMode,
): CompleteScopedPreviewVisuals {
  assert(
    isRecord(value)
      && value.schema === STRESS_VISUAL_SCHEMA
      && value.waveform_bins === WAVEFORM_BINS
      && value.mel_bands === MEL_BANDS
      && value.mel_frames === MEL_FRAMES
      && value.mel_fft_size === MEL_FFT_SIZE
      && value.mel_hop_ms === MEL_HOP_MS
      && value.mel_dynamic_range_db === MEL_DYNAMIC_RANGE_DB
      && value.mel_db_offset_db === MEL_DB_OFFSET_DB
      && value.mel_db_step_db === MEL_DB_STEP_DB
      && isRecord(value.cases),
    "stress visuals 参数无效",
  );
  const caseMap = new Map(cases.map((entry) => [entry.case_id, entry]));
  assert(
    Object.keys(value.cases).length === caseMap.size
      && Object.keys(value.cases).every((caseId) => caseMap.has(caseId)),
    "stress visuals case 集合不一致",
  );
  const visualCases = {} as CompleteScopedPreviewVisuals["cases"];
  for (const [caseId, previewCase] of caseMap) {
    const tracks = value.cases[caseId];
    assert(
      isRecord(tracks)
        && Object.keys(tracks).length === SCOPED_INPUT_SIGNAL_ROLES.length,
      "stress visuals 输入角色不完整",
    );
    const visualTracks = {} as Record<
      ScopedInputSignalRole,
      ScopedPreviewVisualTrack
    >;
    for (const role of SCOPED_INPUT_SIGNAL_ROLES) {
      visualTracks[role] = readVisualTrack(
        tracks[role],
        previewCase.signals_sha256[role],
        mode,
        role,
      );
    }
    visualCases[caseId] = visualTracks;
  }
  return {
    schema: STRESS_VISUAL_SCHEMA,
    waveform_bins: WAVEFORM_BINS,
    mel_bands: MEL_BANDS,
    mel_frames: MEL_FRAMES,
    mel_fft_size: MEL_FFT_SIZE,
    mel_hop_ms: MEL_HOP_MS,
    mel_dynamic_range_db: MEL_DYNAMIC_RANGE_DB,
    mel_db_offset_db: MEL_DB_OFFSET_DB,
    mel_db_step_db: MEL_DB_STEP_DB,
    cases: visualCases,
  };
}

export function readStressDatasets(
  value: unknown,
  expected: { scope: StressScope; mode: FiveSignalMode },
): StressDatasetView {
  assert(isRecord(value), "stress datasets 响应必须为对象");
  const contracts = readContracts(value.contracts, expected.scope, expected.mode);
  const categoryCounts = readCategoryCounts(
    value.category_counts,
    expected.scope,
  );
  const categoryLabels = readCategoryLabels(
    value.category_labels,
    expected.scope,
  );
  assert(
    typeof value.dataset_id === "string"
      && ID.test(value.dataset_id)
      && value.dataset_name === stressDatasetName(expected.scope, expected.mode)
      && value.test_scope === expected.scope
      && value.mode === expected.mode
      && value.scene === "baseline"
      && value.contract_item_count === contracts.length
      && value.preview_condition_group_count === STRESS_CASE_COUNT
      && value.observation_count === STRESS_OBSERVATION_COUNT
      && value.standard_compliance === false
      && value.formal_test_status === "not_measured"
      && Array.isArray(value.datasets)
      && value.datasets.length === 1,
    "stress dataset 视图身份或数量无效",
  );
  const dataset = value.datasets[0];
  assert(
    isRecord(dataset)
      && typeof dataset.dataset_id === "string"
      && ID.test(dataset.dataset_id)
      && dataset.dataset_id === value.dataset_id
      && dataset.dataset_name === stressDatasetName(expected.scope, expected.mode)
      && dataset.dataset_name === value.dataset_name
      && dataset.test_scope === expected.scope
      && dataset.mode === expected.mode
      && dataset.scene === "baseline"
      && dataset.contract_item_count === contracts.length
      && dataset.preview_condition_group_count === STRESS_CASE_COUNT
      && dataset.observation_count === STRESS_OBSERVATION_COUNT
      && dataset.standard_compliance === false
      && dataset.formal_test_status === "not_measured",
    "stress dataset 条目身份无效",
  );
  assert(
    isRecord(dataset.category_counts)
      && sameCategoryCounts(dataset.category_counts, categoryCounts),
    "stress dataset 条目 category_counts 不一致",
  );
  const datasetLabels = readCategoryLabels(
    dataset.category_labels,
    expected.scope,
  );
  return {
    ...value,
    dataset_id: value.dataset_id as string,
    dataset_name: value.dataset_name as string,
    test_scope: expected.scope,
    mode: expected.mode,
    scene: "baseline",
    contract_item_count: contracts.length,
    preview_condition_group_count: STRESS_CASE_COUNT,
    observation_count: STRESS_OBSERVATION_COUNT,
    category_counts: categoryCounts,
    category_labels: categoryLabels,
    contracts,
    datasets: [{
      ...dataset,
      dataset_id: dataset.dataset_id,
      dataset_name: dataset.dataset_name,
      test_scope: expected.scope,
      mode: expected.mode,
      scene: "baseline",
      contract_item_count: contracts.length,
      preview_condition_group_count: STRESS_CASE_COUNT,
      observation_count: STRESS_OBSERVATION_COUNT,
      category_counts: categoryCounts,
      category_labels: datasetLabels,
      standard_compliance: false,
      formal_test_status: "not_measured",
    }],
    standard_compliance: false,
    formal_test_status: "not_measured",
  } as StressDatasetView;
}

function sameCategoryCounts(
  left: Record<string, unknown>,
  right: Record<string, number>,
): boolean {
  const keys = Object.keys(right).sort();
  return (
    Object.keys(left).length === keys.length
    && keys.every((key) => left[key] === right[key])
  );
}

export function readStressPreview(
  value: unknown,
  expected: {
    scope: StressScope;
    mode: FiveSignalMode;
    datasetId: string;
    contracts: readonly ScopedItem[];
  },
): StressPreview {
  assert(isRecord(value), "stress preview 响应必须为对象");
  const contracts = readContracts(expected.contracts, expected.scope, expected.mode);
  const categoryCounts = readCategoryCounts(
    value.category_counts,
    expected.scope,
  );
  const categoryLabels = readCategoryLabels(
    value.category_labels,
    expected.scope,
  );
  assert(
    value.dataset_id === expected.datasetId
      && value.dataset_name === stressDatasetName(expected.scope, expected.mode)
      && value.test_scope === expected.scope
      && value.mode === expected.mode
      && value.scene === "baseline"
      && value.schema_version === STRESS_SCHEMA_VERSION
      && value.dataset_kind === STRESS_DATASET_KIND
      && value.contract_item_count === contracts.length
      && value.preview_condition_group_count === STRESS_CASE_COUNT
      && value.observation_count === STRESS_OBSERVATION_COUNT
      && value.standard_compliance === false
      && value.formal_test_status === "not_measured"
      && typeof value.dataset_manifest_sha256 === "string"
      && SHA256.test(value.dataset_manifest_sha256)
      && typeof value.dataset_assets_sha256 === "string"
      && SHA256.test(value.dataset_assets_sha256)
      && Array.isArray(value.cases)
      && value.cases.length === STRESS_CASE_COUNT,
    "stress preview 身份、schema 或数量无效",
  );
  if (value.contracts !== undefined) {
    const responseContracts = readContracts(
      value.contracts,
      expected.scope,
      expected.mode,
    );
    assert(
      responseContracts.length === contracts.length
        && responseContracts.every(
          (row, index) => row.id === contracts[index].id,
        ),
      "stress preview contracts 与请求合同不一致",
    );
  }
  const contractById = new Map(contracts.map((row) => [row.id, row]));
  const cases: StressPreviewCase[] = [];
  const seen = new Set<string>();
  const categoryIndices = new Map<StressCategory, Set<number>>();
  const usedContracts = new Set<string>();
  for (const raw of value.cases) {
    assert(isRecord(raw), "stress preview case 必须为对象");
    assert(
      typeof raw.case_id === "string"
        && raw.case_id.length > 0
        && !seen.has(raw.case_id),
      "stress preview case_id 为空或重复",
    );
    const contractId = Array.isArray(raw.contract_item_ids)
      && raw.contract_item_ids.length === 1
      ? raw.contract_item_ids[0]
      : null;
    assert(
      typeof contractId === "string"
        && contractById.has(contractId),
      "stress preview case 必须绑定一个真实合同行",
    );
    const contract = contractById.get(contractId)!;
    assert(
      Array.isArray(raw.families)
        && raw.families.length === 1
        && raw.families[0] === contract.family,
      "stress preview families 与合同 family 不一致",
    );
    const conditions = readConditions(raw.conditions, contract, expected.mode);
    const provenance = readStressProvenance(
      raw.provenance,
      expected.scope,
      expected.mode,
    );
    const signals = raw.signals;
    assert(
      Array.isArray(signals)
        && signals.length === SCOPED_INPUT_SIGNAL_ROLES.length
        && new Set(signals).size === signals.length
        && signals.every(
          (role) => typeof role === "string"
            && (SCOPED_INPUT_SIGNAL_ROLES as readonly string[]).includes(role),
        ),
      "stress preview signals 必须为四路合法输入",
    );
    const hashes = readInputHashes(
      raw.signals_sha256,
      "stress preview signals_sha256 无效",
    );
    const categoryIndicesForScope =
      categoryIndices.get(provenance.workload_category)
      ?? new Set<number>();
    assert(
      !categoryIndicesForScope.has(provenance.category_index),
      "stress preview category_index 重复",
    );
    categoryIndicesForScope.add(provenance.category_index);
    categoryIndices.set(provenance.workload_category, categoryIndicesForScope);
    usedContracts.add(contractId);
    seen.add(raw.case_id);
    cases.push({
      case_id: raw.case_id,
      conditions,
      families: [contract.family],
      contract_item_ids: [contractId],
      provenance,
      signals: signals as ScopedInputSignalRole[],
      signals_sha256: hashes,
    });
  }
  for (const category of stressCategories(expected.scope)) {
    assert(
      categoryIndices.get(category)?.size === 20,
      `stress preview ${category} 必须覆盖 20 个 category_index`,
    );
  }
  assert(
    [...contractById.keys()].every((id) => usedContracts.has(id)),
    "stress preview 未覆盖全部 scope 合同行",
  );
  const visuals = readVisuals(value.visuals, cases, expected.mode);
  return {
    ...value,
    dataset_id: expected.datasetId,
    dataset_name: stressDatasetName(expected.scope, expected.mode),
    dataset_manifest_sha256: value.dataset_manifest_sha256,
    dataset_assets_sha256: value.dataset_assets_sha256,
    test_scope: expected.scope,
    mode: expected.mode,
    scene: "baseline",
    schema_version: STRESS_SCHEMA_VERSION,
    dataset_kind: STRESS_DATASET_KIND,
    contract_item_count: contracts.length,
    preview_condition_group_count: STRESS_CASE_COUNT,
    observation_count: STRESS_OBSERVATION_COUNT,
    category_counts: categoryCounts,
    category_labels: categoryLabels,
    contracts,
    cases,
    visuals,
    standard_compliance: false,
    formal_test_status: "not_measured",
  };
}

function readDiagnosticMetrics(value: unknown): StressDiagnosticMetrics {
  assert(isRecord(value), "stress diagnostics metrics 必须为对象");
  const keys = Object.keys(value).sort();
  const expected = [...METRIC_KEYS].sort();
  assert(
    keys.length === expected.length
      && keys.every((key, index) => key === expected[index]),
    "stress diagnostics metrics 字段不完整",
  );
  for (const metric of Object.values(value)) {
    assert(
      metric === null || isFiniteNumber(metric),
      "stress diagnostics metric 必须为有限数或 null",
    );
  }
  return {
    primary: value.primary as number | null,
    tx_residual_reduction_db: value.tx_residual_reduction_db as number | null,
    convergence_db: value.convergence_db as number | null,
    cleanref_projection_gain_db: value.cleanref_projection_gain_db as number | null,
    cleanref_correlation: value.cleanref_correlation as number | null,
    tx_level_dbfs: value.tx_level_dbfs as number | null,
    mic_level_dbfs: value.mic_level_dbfs as number | null,
    far_level_dbfs: value.far_level_dbfs as number | null,
    near_level_dbfs: value.near_level_dbfs as number | null,
  };
}

function metricsAllNull(metrics: StressDiagnosticMetrics): boolean {
  return Object.values(metrics).every((metric) => metric === null);
}

function readActivityGate(value: unknown): StressActivityGate {
  assert(isRecord(value), "stress_diagnostics.activity_gate 必须为对象");
  assert(
    Object.keys(value).sort().join(",")
      === [
        "frame_ms",
        "hop_ms",
        "rms_threshold",
        "min_active_s",
        "min_coverage",
      ].sort().join(",")
      && value.frame_ms === STRESS_ACTIVITY_FRAME_MS
      && value.hop_ms === STRESS_ACTIVITY_HOP_MS
      && value.rms_threshold === STRESS_ACTIVITY_RMS_THRESHOLD
      && value.min_active_s === STRESS_MIN_ACTIVE_S
      && value.min_coverage === STRESS_MIN_ACTIVITY_COVERAGE,
    "stress_diagnostics.activity_gate 参数无效",
  );
  return {
    frame_ms: STRESS_ACTIVITY_FRAME_MS,
    hop_ms: STRESS_ACTIVITY_HOP_MS,
    rms_threshold: STRESS_ACTIVITY_RMS_THRESHOLD,
    min_active_s: STRESS_MIN_ACTIVE_S,
    min_coverage: STRESS_MIN_ACTIVITY_COVERAGE,
  };
}

function readSegmentActivity(
  value: unknown,
  segment: StressActivitySegment,
  scope: StressScope,
  rate: number,
): StressSegmentActivity {
  assert(isRecord(value), "stress segment activity 必须为对象");
  const label = segment.label;
  const baseFields = [
    "frame_ms",
    "hop_ms",
    "rms_threshold",
    "min_active_s",
    "min_coverage",
    "near_active_samples",
    "far_active_samples",
    "joint_active_samples",
    "near_coverage",
    "far_coverage",
    "joint_coverage",
    "valid",
    "reason",
  ];
  const echoFields = [
    "echo_delay_guard_ms",
    "echo_guarded_active_samples",
    "echo_guarded_active_s",
  ];
  const expectedFields = scope === "echo" && label === "far_only"
    ? [...baseFields, ...echoFields]
    : baseFields;
  assert(
    Object.keys(value).sort().join(",") === expectedFields.sort().join(",")
      && value.frame_ms === STRESS_ACTIVITY_FRAME_MS
      && value.hop_ms === STRESS_ACTIVITY_HOP_MS
      && value.rms_threshold === STRESS_ACTIVITY_RMS_THRESHOLD
      && value.min_active_s === STRESS_MIN_ACTIVE_S
      && value.min_coverage === STRESS_MIN_ACTIVITY_COVERAGE
      && isCount(value.near_active_samples)
      && isCount(value.far_active_samples)
      && isCount(value.joint_active_samples)
      && isFiniteNumber(value.near_coverage)
      && value.near_coverage >= 0
      && value.near_coverage <= 1
      && isFiniteNumber(value.far_coverage)
      && value.far_coverage >= 0
      && value.far_coverage <= 1
      && isFiniteNumber(value.joint_coverage)
      && value.joint_coverage >= 0
      && value.joint_coverage <= 1
      && typeof value.valid === "boolean"
      && typeof value.reason === "string"
      && value.reason.trim().length > 0,
    "stress segment activity 字段或数值无效",
  );
  const segmentSamples = Math.round(segment.end_s * rate)
    - Math.round(segment.start_s * rate);
  const nearSamples = value.near_active_samples as number;
  const farSamples = value.far_active_samples as number;
  const jointSamples = value.joint_active_samples as number;
  const nearCoverage = value.near_coverage as number;
  const farCoverage = value.far_coverage as number;
  const jointCoverage = value.joint_coverage as number;
  const coverageTolerance = 1e-9;
  assert(
    segmentSamples > 0
      && nearSamples <= segmentSamples
      && farSamples <= segmentSamples
      && jointSamples <= segmentSamples
      && jointSamples <= Math.min(nearSamples, farSamples)
      && Math.abs(nearCoverage - nearSamples / segmentSamples)
        <= coverageTolerance
      && Math.abs(farCoverage - farSamples / segmentSamples)
        <= coverageTolerance
      && Math.abs(jointCoverage - jointSamples / segmentSamples)
        <= coverageTolerance,
    "stress segment activity count/coverage 不自洽",
  );
  if (value.valid) {
    const minSamples = Math.round(STRESS_MIN_ACTIVE_S * rate);
    const nearValid = nearSamples >= minSamples
      && nearCoverage >= STRESS_MIN_ACTIVITY_COVERAGE;
    const farValid = farSamples >= minSamples
      && farCoverage >= STRESS_MIN_ACTIVITY_COVERAGE;
    if (label === "far_only") {
      assert(
        farValid && nearSamples === 0 && jointSamples === 0,
        "far_only valid activity 不满足远端单讲规则",
      );
    } else if (label === "near_only") {
      assert(
        nearValid && farSamples === 0 && jointSamples === 0,
        "near_only valid activity 不满足近端单讲规则",
      );
    } else if (label === "double_talk") {
      assert(
        nearValid
          && farValid
          && jointSamples >= minSamples
          && jointCoverage >= STRESS_MIN_ACTIVITY_COVERAGE,
        "double_talk valid activity 不满足双端同帧重叠规则",
      );
    } else {
      assert(
        nearSamples === 0
          && farSamples === 0
          && jointSamples === 0,
        "quiet/noise valid activity 必须无近远端活动",
      );
    }
  }
  const result: StressSegmentActivity = {
    frame_ms: STRESS_ACTIVITY_FRAME_MS,
    hop_ms: STRESS_ACTIVITY_HOP_MS,
    rms_threshold: STRESS_ACTIVITY_RMS_THRESHOLD,
    min_active_s: STRESS_MIN_ACTIVE_S,
    min_coverage: STRESS_MIN_ACTIVITY_COVERAGE,
    near_active_samples: nearSamples,
    far_active_samples: farSamples,
    joint_active_samples: jointSamples,
    near_coverage: nearCoverage,
    far_coverage: farCoverage,
    joint_coverage: jointCoverage,
    valid: value.valid,
    reason: value.reason,
  };
  if (scope === "echo" && label === "far_only") {
    assert(
      value.echo_delay_guard_ms === STRESS_ECHO_DELAY_GUARD_MS
        && isCount(value.echo_guarded_active_samples)
        && value.echo_guarded_active_samples <= value.far_active_samples
        && isFiniteNumber(value.echo_guarded_active_s)
        && Math.abs(
          value.echo_guarded_active_s
            - value.echo_guarded_active_samples / rate,
        ) <= 1e-9,
      "stress echo delay-guard activity 无效",
    );
    result.echo_delay_guard_ms = STRESS_ECHO_DELAY_GUARD_MS;
    result.echo_guarded_active_samples =
      value.echo_guarded_active_samples;
    result.echo_guarded_active_s = value.echo_guarded_active_s;
  }
  return result;
}

function readStressDiagnostics(
  value: unknown,
  expected: {
    caseId: string;
    txSha256: string;
    scope: StressScope;
    mode: FiveSignalMode;
    category: StressCategory;
    segments: StressActivitySegment[];
  },
): StressDiagnostics {
  assert(isRecord(value), "stress_diagnostics 必须为对象");
  const schemaVersion = value.schema_version;
  assert(
    schemaVersion === STRESS_DIAGNOSTICS_SCHEMA_V1
      || schemaVersion === STRESS_DIAGNOSTICS_SCHEMA_V2,
    "stress_diagnostics schema_version 不受支持",
  );
  assert(
    value.case_id === expected.caseId
      && value.tx_sha256 === expected.txSha256
      && value.scope === expected.scope
      && value.mode === expected.mode
      && value.category === expected.category
      && value.evidence_grade === "engineering_proxy"
      && value.formal_compliance === false
      && value.standard_compliance === false
      && value.formal_test_status === "not_measured",
    "stress_diagnostics 身份或正式边界无效",
  );
  assert(
    DIAGNOSTIC_STATUSES.has(String(value.status))
      && typeof value.reason === "string"
      && value.reason.trim().length > 0
      && typeof value.review_required === "boolean"
      && value.review_required === (value.status === "review_required")
      && typeof value.tx_muted === "boolean"
      && value.sample_rate_hz === stressSampleRate(expected.mode)
      && isFiniteNumber(value.source_duration_s)
      && Math.abs(value.source_duration_s - STRESS_SOURCE_DURATION_S) <= 1e-6
      && isFiniteNumber(value.tx_duration_s)
      && value.tx_duration_s >= 0
      && Array.isArray(value.limitations)
      && value.limitations.length > 0
      && value.limitations.every(
        (item) => typeof item === "string" && item.length > 0,
      ),
    "stress_diagnostics envelope 无效",
  );
  assert(isRecord(value.primary), "stress_diagnostics primary 无效");
  const expectedPrimaryName = expected.scope === "echo"
    ? "echo_primary_proxy_db"
    : "double_talk_energy_ratio_proxy_percent";
  const expectedPrimaryUnit = expected.scope === "echo" ? "dB" : "percent";
  assert(
    Object.keys(value.primary).sort().join(",")
      === ["name", "status", "unit", "value"].sort().join(",")
      && value.primary.name === expectedPrimaryName
      && value.primary.unit === expectedPrimaryUnit
      && (value.primary.value === null || isFiniteNumber(value.primary.value))
      && (value.primary.value === null
        ? value.primary.status === "not_measured"
        : value.primary.status === "review_required")
      && (expected.scope !== "double_talk"
        || value.primary.value === null
        || value.primary.value >= 0 && value.primary.value <= 150),
    "stress_diagnostics primary 与 scope/value 不一致",
  );
  assert(
    value.convergence_db === null || isFiniteNumber(value.convergence_db),
    "stress_diagnostics convergence_db 必须有限或 null",
  );
  if (expected.scope === "double_talk") {
    assert(
      value.convergence_db === null,
      "double_talk stress_diagnostics 不应携带 echo convergence",
    );
  }
  if (
    (value.primary.value !== null)
      !== (value.status === "review_required")
  ) {
    throw Error("stress_diagnostics primary 与顶层状态不一致");
  }
  const txInvalid =
    value.tx_duration_s < 0.95 * STRESS_SOURCE_DURATION_S
    || value.tx_muted;
  if (txInvalid && expected.scope === "echo") {
    assert(
      value.convergence_db === null,
      "echo TX mute/short 不能携带 convergence",
    );
  }
  if (txInvalid && expected.scope === "double_talk") {
    assert(
      value.primary.value === null && value.convergence_db === null,
      "double-talk TX mute/short 不能携带 primary/convergence",
    );
  }

  assert(
    Array.isArray(value.segments)
      && value.segments.length === expected.segments.length,
    "stress_diagnostics segments 数量不一致",
  );
  let activityGate: StressActivityGate | undefined;
  if (schemaVersion === STRESS_DIAGNOSTICS_SCHEMA_V2) {
    activityGate = readActivityGate(value.activity_gate);
    assert(
      value.convergence_note === undefined
        || typeof value.convergence_note === "string"
          && value.convergence_note.trim().length > 0,
      "stress_diagnostics.convergence_note 无效",
    );
  } else {
    assert(
      !Object.hasOwn(value, "activity_gate")
        && !Object.hasOwn(value, "convergence_note"),
      "v1 stress_diagnostics 不得声称 activity gate",
    );
  }

  const rate = stressSampleRate(expected.mode);
  const normalizedSegments: StressDiagnosticSegment[] = [];
  for (const [index, raw] of value.segments.entries()) {
    const activity = expected.segments[index];
    assert(
      isRecord(raw)
        && raw.label === activity.label
        && raw.start_s === activity.start_s
        && raw.end_s === activity.end_s
        && raw.mode === SEGMENT_MODE[activity.label]
        && DIAGNOSTIC_STATUSES.has(String(raw.status))
        && typeof raw.reason === "string"
        && raw.reason.trim().length > 0,
      "stress_diagnostics segment 身份、模式或时间不一致",
    );
    const segmentFields = [
      "label",
      "start_s",
      "end_s",
      "mode",
      "status",
      "metrics",
      "reason",
    ];
    assert(
      Object.keys(raw).sort().join(",")
        === (
          schemaVersion === STRESS_DIAGNOSTICS_SCHEMA_V2
            ? [...segmentFields, "activity"]
            : segmentFields
        ).sort().join(","),
      "stress_diagnostics segment 字段不完整",
    );
    const metrics = readDiagnosticMetrics(raw.metrics);
    if (schemaVersion === STRESS_DIAGNOSTICS_SCHEMA_V1) {
      assert(
        !Object.hasOwn(raw, "activity"),
        "v1 stress segment 不得声称 frame activity",
      );
      if (
        txInvalid
        && expected.scope === "double_talk"
        && ["near_only", "far_only", "double_talk"].includes(activity.label)
      ) {
        assert(
          raw.status === "not_measurable" && metricsAllNull(metrics),
          "double-talk mute/short 必须清空 speech segment metrics",
        );
      }
      normalizedSegments.push({
        label: activity.label,
        start_s: activity.start_s,
        end_s: activity.end_s,
        mode: SEGMENT_MODE[activity.label],
        status: raw.status as StressDiagnosticStatus,
        metrics,
        reason: raw.reason,
      });
      continue;
    }

    assert(
      Object.hasOwn(raw, "activity"),
      "v2 stress segment 缺少 activity",
    );
    const segmentActivity = readSegmentActivity(
      raw.activity,
      activity,
      expected.scope,
      rate,
    );
    if (!segmentActivity.valid) {
      assert(
        raw.status === "not_measurable" && metricsAllNull(metrics),
        "invalid activity 必须为 not_measurable 且 metrics 全空",
      );
    } else if (["quiet", "noise_only"].includes(activity.label)) {
      assert(
        raw.status === "not_applicable" && metricsAllNull(metrics),
        "quiet/noise activity 必须 not_applicable 且 metrics 全空",
      );
    } else {
      assert(
        raw.status !== "not_applicable",
        "speech activity 不得标为 not_applicable",
      );
    }
    if (raw.status === "review_required") {
      assert(
        Object.values(metrics).some((metric) => isFiniteNumber(metric)),
        "review_required segment 必须有有限指标",
      );
    }
    if (
      expected.scope === "echo"
      && activity.label === "far_only"
      && segmentActivity.echo_guarded_active_samples !== undefined
      && segmentActivity.echo_guarded_active_samples
        < Math.max(1, Math.round(STRESS_MIN_ACTIVE_S * rate)
          - Math.floor(STRESS_ACTIVITY_HOP_MS * rate / 1000 / 2))
    ) {
      assert(
        metrics.primary === null,
        "echo delay guard 不足时不得提升 primary",
      );
    }
    if (
      txInvalid
      && expected.scope === "double_talk"
      && ["near_only", "far_only", "double_talk"].includes(activity.label)
    ) {
      assert(
        raw.status === "not_measurable" && metricsAllNull(metrics),
        "double-talk mute/short 必须清空 speech segment metrics",
      );
    }
    if (
      txInvalid
      && expected.scope === "echo"
      && activity.label === "far_only"
    ) {
      assert(
        metrics.tx_residual_reduction_db === null
          && metrics.tx_level_dbfs === null,
        "echo mute/short 必须清空 TX residual/level",
      );
    }
    normalizedSegments.push({
      label: activity.label,
      start_s: activity.start_s,
      end_s: activity.end_s,
      mode: SEGMENT_MODE[activity.label],
      status: raw.status as StressDiagnosticStatus,
      metrics,
      reason: raw.reason,
      activity: segmentActivity,
    });
  }

  if (value.primary.value !== null) {
    const reviewLabel = expected.scope === "echo"
      ? "far_only"
      : "double_talk";
    assert(
      normalizedSegments.some(
        (segment) =>
          segment.label === reviewLabel
          && segment.status === "review_required"
          && isFiniteNumber(segment.metrics.primary),
      ),
      "stress primary 缺少对应 reviewable segment",
    );
  }

  const common = {
    case_id: expected.caseId,
    tx_sha256: expected.txSha256,
    scope: expected.scope,
    mode: expected.mode,
    category: expected.category,
    evidence_grade: "engineering_proxy" as const,
    formal_compliance: false as const,
    standard_compliance: false as const,
    formal_test_status: "not_measured" as const,
    status: value.status as StressDiagnosticAggregateStatus,
    reason: value.reason,
    review_required: value.review_required,
    sample_rate_hz: value.sample_rate_hz,
    source_duration_s: value.source_duration_s,
    tx_duration_s: value.tx_duration_s,
    tx_muted: value.tx_muted,
    primary: {
      name: value.primary.name as StressDiagnostics["primary"]["name"],
      value: value.primary.value as number | null,
      unit: value.primary.unit as StressDiagnostics["primary"]["unit"],
      status: value.primary.status as StressDiagnostics["primary"]["status"],
    },
    convergence_db: value.convergence_db as number | null,
    limitations: value.limitations as string[],
  };
  if (schemaVersion === STRESS_DIAGNOSTICS_SCHEMA_V1) {
    return {
      ...common,
      schema_version: STRESS_DIAGNOSTICS_SCHEMA_V1,
      segments: normalizedSegments as StressDiagnosticSegmentV1[],
    };
  }
  return {
    ...common,
    schema_version: STRESS_DIAGNOSTICS_SCHEMA_V2,
    activity_gate: activityGate!,
    ...(typeof value.convergence_note === "string"
      ? { convergence_note: value.convergence_note }
      : {}),
    segments: normalizedSegments as StressDiagnosticSegmentV2[],
  };
}

function readStressBatchCase(
  value: unknown,
  expected: {
    scope: StressScope;
    mode: FiveSignalMode;
    contracts: Map<string, ScopedItem>;
  },
): StressBatchCase {
  assert(isRecord(value), "stress batch case 必须为对象");
  assert(
    typeof value.case_id === "string"
      && value.case_id.length > 0
      && ["pending", "running", "completed", "failed"].includes(String(value.status)),
    "stress batch case ID 或状态无效",
  );
  const contractId = Array.isArray(value.contract_item_ids)
    && value.contract_item_ids.length === 1
    ? value.contract_item_ids[0]
    : null;
  assert(
    typeof contractId === "string" && expected.contracts.has(contractId),
    "stress batch case 合同行无效",
  );
  const contract = expected.contracts.get(contractId)!;
  const conditions = readConditions(value.conditions, contract, expected.mode);
  const provenance = readStressProvenance(
    value.provenance,
    expected.scope,
    expected.mode,
  );
  const signalsSha = readInputHashes(
    value.signals_sha256,
    "stress batch signals_sha256 无效",
  );
  const inputSha = readInputHashes(
    value.input_sha256,
    "stress batch input_sha256 无效",
  );
  assert(sameInputHashes(signalsSha, inputSha), "stress batch 输入哈希不一致");
  const reportId = value.report_id === null
    ? null
    : readId(value.report_id, "stress batch report_id 无效");
  const outputSha = value.output_sha256 === undefined
    ? undefined
    : (() => {
        const keys = isRecord(value.output_sha256)
          ? Object.keys(value.output_sha256).sort()
          : [];
        assert(
          isRecord(value.output_sha256)
            && (keys.length === 1 || keys.length === 2)
            && Object.hasOwn(value.output_sha256, "tx"),
          "stress batch output_sha256 无效",
        );
        if (keys.length === 2) {
          assert(
            keys[0] === "rx" && keys[1] === "tx",
            "stress batch output_sha256 无效",
          );
        }
        return {
          tx: readSha(value.output_sha256.tx, "stress batch TX hash 无效"),
          ...(Object.hasOwn(value.output_sha256, "rx")
            ? {
                rx: readSha(
                  value.output_sha256.rx,
                  "stress batch RX hash 无效",
                ),
              }
            : {}),
        };
      })();
  const txVisual = value.tx_visual === undefined || value.tx_visual === null
    ? undefined
    : readVisualTrack(
        value.tx_visual,
        outputSha?.tx ?? "",
        expected.mode,
        "noise",
      );
  const rxVisual = value.rx_visual === undefined || value.rx_visual === null
    ? undefined
    : readVisualTrack(
        value.rx_visual,
        outputSha?.rx ?? "",
        expected.mode,
        "noise",
      );
  if (value.status === "completed") {
    assert(
      reportId !== null && outputSha !== undefined && txVisual !== undefined,
      "completed stress case 缺少报告、TX 或可视化",
    );
    if (outputSha.rx !== undefined) {
      assert(
        rxVisual !== undefined,
        "completed stress case 缺少 RX 处理后可视化",
      );
    }
  }
  let diagnostics: StressDiagnostics | undefined;
  if (value.stress_diagnostics !== undefined) {
    assert(
      outputSha !== undefined,
      "stress_diagnostics 缺少 TX 输出",
    );
    diagnostics = readStressDiagnostics(value.stress_diagnostics, {
      caseId: value.case_id,
      txSha256: outputSha.tx,
      scope: expected.scope,
      mode: expected.mode,
      category: provenance.workload_category,
      segments: provenance.activity_segments,
    });
  }
  assert(
    value.status !== "completed" || diagnostics !== undefined,
    "completed stress case 缺少 stress_diagnostics",
  );
  const assessmentSha = value.assessment_sha256 === undefined
    ? undefined
    : readSha(value.assessment_sha256, "stress assessment_sha256 无效");
  assert(
    value.execution === undefined
      || isRecord(value.execution) && isFiniteJson(value.execution),
    "stress case execution 无效",
  );
  if (outputSha?.rx !== undefined) {
    const echoReference = isRecord(value.execution)
      ? value.execution.echo_reference
      : undefined;
    assert(
      isRecord(echoReference)
        && echoReference.source === "vcp_rx_processed"
        && echoReference.processed_output === "rx"
        && echoReference.processed_sha256 === outputSha.rx
        && echoReference.tx_echo_argument_sha256 === outputSha.rx,
      "stress TX 未绑定 RX 处理后回声参考",
    );
  }
  return {
    case_id: value.case_id,
    status: value.status as StressBatchCase["status"],
    report_id: reportId,
    conditions,
    contract_item_ids: [contractId],
    provenance,
    ...(isRecord(value.signals)
      ? { signals: value.signals as Record<ScopedInputSignalRole, string> }
      : {}),
    signals_sha256: signalsSha,
    input_sha256: inputSha,
    ...(outputSha ? { output_sha256: outputSha } : {}),
    ...(txVisual ? { tx_visual: txVisual } : {}),
    ...(rxVisual ? { rx_visual: rxVisual } : {}),
    ...(diagnostics ? { stress_diagnostics: diagnostics } : {}),
    ...(assessmentSha ? { assessment_sha256: assessmentSha } : {}),
    ...(isRecord(value.execution)
      ? { execution: value.execution }
      : {}),
    ...(typeof value.error === "string" ? { error: value.error } : {}),
  };
}

function readEvidenceHashes(
  value: unknown,
  role: "input_sha256" | "output_sha256",
  message: string,
): Record<ScopedInputSignalRole, string>
  | { tx: string; rx?: string }
  | Record<string, never> {
  assert(isRecord(value), message);
  if (role === "input_sha256") {
    return readInputHashes(value, message);
  }
  if (Object.keys(value).length === 0) return {};
  const keys = Object.keys(value).sort();
  assert(
    (keys.length === 1 || keys.length === 2)
      && Object.hasOwn(value, "tx"),
    message,
  );
  if (keys.length === 2) {
    assert(keys[0] === "rx" && keys[1] === "tx", message);
  }
  return {
    tx: readSha(value.tx, message),
    ...(Object.hasOwn(value, "rx")
      ? { rx: readSha(value.rx, message) }
      : {}),
  };
}

function readStressObservation(
  value: unknown,
  expected: {
    scope: StressScope;
    mode: FiveSignalMode;
    contracts: Map<string, ScopedItem>;
    cases: Map<string, StressBatchCase>;
    batchId: string;
    datasetId: string;
  },
): StressObservation {
  assert(isRecord(value), "stress observation 必须为对象");
  assert(
    typeof value.case_id === "string"
      && expected.cases.has(value.case_id),
    "stress observation 引用未知 case",
  );
  const caseRecord = expected.cases.get(value.case_id)!;
  const contractId = typeof value.contract_id === "string"
    ? value.contract_id
    : typeof value.item_id === "string"
      ? value.item_id
      : "";
  assert(
    expected.contracts.has(contractId)
      && value.item_id === contractId
      && value.contract_id === contractId,
    "stress observation 合同身份无效",
  );
  const contract = expected.contracts.get(contractId)!;
  for (const field of [
    "id",
    "source_cell",
    "test_name",
    "family",
    "direction",
    "unit",
    "lower_limit",
    "upper_limit",
  ] as const) {
    assert(value[field] === contract[field], `stress observation ${field} 不一致`);
  }
  assert(
    value.observation_id === `${value.case_id}:${contractId}`
      && value.mode === expected.mode
      && ITEM_STATUSES.has(String(value.status)),
    "stress observation 模式、ID 或状态无效",
  );
  const status = value.status as StressObservation["status"];
  const numericValue = value.value === null
    ? null
    : isFiniteNumber(value.value)
      ? value.value
      : (() => {
          throw Error("stress observation value 必须有限或 null");
        })();
  if (numericValue !== null) {
    assert(status === "review_required", "有限 stress item 只能是 review_required");
  } else {
    assert(
      status === "pending"
        || status === "running"
        || status === "not_measured"
        || status === "invalid",
      "无值 stress item 只能是 pending/running/not_measured/invalid",
    );
  }
  assert(
    value.metric === null || typeof value.metric === "string",
    "stress observation metric 无效",
  );
  assert(typeof value.reason === "string", "stress observation reason 无效");
  assert(isRecord(value.evidence_binding), "stress observation evidence_binding 无效");
  const binding = value.evidence_binding;
  assert(
    binding.case_id === caseRecord.case_id
      && binding.batch_id === expected.batchId
      && binding.dataset_id === expected.datasetId
      && binding.report_id === caseRecord.report_id
      && sameConditions({
        mode: binding.mode as FiveSignalMode,
        direction: String(binding.direction),
        scene: binding.scene as "baseline",
        volume: binding.volume as string | number | null,
        window_ms: binding.window_ms as number | null,
        input_level_db: binding.input_level_db as number | null,
      }, caseRecord.conditions)
      && sameInputHashes(
        readInputHashes(binding.input_sha256, "stress binding input_sha256 无效"),
        caseRecord.signals_sha256,
      ),
    "stress observation evidence_binding 身份无效",
  );
  const outputHashes = readEvidenceHashes(
    binding.output_sha256,
    "output_sha256",
    "stress binding output_sha256 无效",
  ) as Record<string, unknown>;
  if (caseRecord.output_sha256) {
    assert(
      isRecord(outputHashes)
        && outputHashes.tx === caseRecord.output_sha256.tx
        && outputHashes.rx === caseRecord.output_sha256.rx,
      "stress binding RX/TX hash 与 case 不一致",
    );
  } else {
    assert(
      isRecord(outputHashes) && Object.keys(outputHashes).length === 0,
      "pending stress binding 不应携带 TX hash",
    );
  }
  return {
    ...contract,
    contract_id: contractId,
    item_id: contractId,
    observation_id: value.observation_id,
    case_id: value.case_id,
    mode: expected.mode,
    status,
    value: numericValue,
    metric: typeof value.metric === "string" ? value.metric : null,
    reason: value.reason,
    evidence_binding: binding,
  };
}

export function readStressBatch(
  value: unknown,
  expected: {
    scope: StressScope;
    mode: FiveSignalMode;
    datasetId: string;
    algorithmClusterId: string;
    contracts: readonly ScopedItem[];
    caseId?: string;
  },
): StressBatch {
  assert(isRecord(value), "stress batch 必须为对象");
  const contracts = readContracts(expected.contracts, expected.scope, expected.mode);
  const contractById = new Map(contracts.map((row) => [row.id, row]));
  const selection = readSingleCaseSelection(value, {
    caseId: expected.caseId,
    label: "stress batch",
  });
  const expectedCaseCount = selection === null
    ? STRESS_CASE_COUNT
    : 1;
  assert(
    value.schema_version === "scene-condition-batch-v1"
      && value.batch_id !== undefined
      && ID.test(String(value.batch_id))
      && value.dataset_id === expected.datasetId
      && value.dataset_name === stressDatasetName(expected.scope, expected.mode)
      && value.selected_mode === expected.mode
      && value.selected_scene === "baseline"
      && value.test_scope === expected.scope
      && (value.stress_scope === undefined || value.stress_scope === expected.scope)
      && value.algorithm_cluster_id === expected.algorithmClusterId
      && value.standard_compliance === false
      && value.formal_test_status === "not_measured"
      && value.contract_item_count === contracts.length
      && value.case_count === expectedCaseCount
      && value.preview_condition_group_count === STRESS_CASE_COUNT
      && isCount(value.observation_count)
      && Number(value.observation_count) > 0
      && (selection === null
        ? value.observation_count === STRESS_OBSERVATION_COUNT
        : selection.dataset_case_count === STRESS_CASE_COUNT
          && selection.dataset_observation_count
            === STRESS_OBSERVATION_COUNT)
      && Array.isArray(value.cases)
      && value.cases.length === expectedCaseCount
      && Array.isArray(value.items)
      && value.items.length === value.observation_count,
    "stress batch 身份、schema、profile 或数量无效",
  );
  const createdAt = readTimestamp(value.created_at, "stress batch created_at 无效");
  const updatedAt = readTimestamp(value.updated_at, "stress batch updated_at 无效");
  assert(Date.parse(updatedAt) >= Date.parse(createdAt), "stress batch 更新时间无效");
  assert(
    typeof value.dataset_manifest_sha256 === "string"
      && SHA256.test(value.dataset_manifest_sha256)
      && typeof value.dataset_assets_sha256 === "string"
      && SHA256.test(value.dataset_assets_sha256)
      && typeof value.catalog_sha256 === "string"
      && SHA256.test(value.catalog_sha256)
      && typeof value.cli_sha256 === "string"
      && SHA256.test(value.cli_sha256)
      && isRecord(value.configuration)
      && isFiniteJson(value.configuration),
    "stress batch hash 或 configuration 无效",
  );
  const cases = value.cases.map((entry) =>
    readStressBatchCase(entry, {
      scope: expected.scope,
      mode: expected.mode,
      contracts: contractById,
    })
  );
  const caseById = new Map<string, StressBatchCase>();
  for (const caseRecord of cases) {
    assert(!caseById.has(caseRecord.case_id), "stress batch case_id 重复");
    caseById.set(caseRecord.case_id, caseRecord);
  }
  if (selection !== null) {
    assert(
      cases.length === 1
        && cases[0].case_id === selection.selected_case_id,
      "single_case stress batch case 与 selected_case_id 不一致",
    );
  }
  const categoryIndices = new Map<StressCategory, Set<number>>();
  for (const caseRecord of cases) {
    const category = caseRecord.provenance.workload_category;
    const indices = categoryIndices.get(category) ?? new Set<number>();
    assert(!indices.has(caseRecord.provenance.category_index), "stress batch category_index 重复");
    indices.add(caseRecord.provenance.category_index);
    categoryIndices.set(category, indices);
  }
  const selectedCategory = selection === null
    ? undefined
    : cases[0].provenance.workload_category;
  if (selection === null) {
    for (const category of stressCategories(expected.scope)) {
      assert(
        categoryIndices.get(category)?.size === 20,
        `stress batch ${category} category_index 未覆盖 20`,
      );
    }
  } else {
    assert(
      selectedCategory !== undefined
        && categoryIndices.size === 1
        && categoryIndices.get(selectedCategory)?.size === 1,
      "single_case stress batch 类别身份无效",
    );
  }
  const categoryCounts = readCategoryCounts(
    value.category_counts,
    expected.scope,
    selectedCategory,
  );
  const observations = value.items.map((entry) =>
    readStressObservation(entry, {
      scope: expected.scope,
      mode: expected.mode,
      contracts: contractById,
      cases: caseById,
      batchId: value.batch_id as string,
      datasetId: expected.datasetId,
    })
  );
  assert(
    new Set(observations.map((item) => item.observation_id)).size
      === value.observation_count,
    "stress batch observation_id 重复",
  );
  const statusCounts = {
    pass: 0,
    fail: 0,
    review_required: 0,
    not_measured: 0,
    invalid: 0,
  };
  let computed = 0;
  for (const observation of observations) {
    if (observation.status in statusCounts) {
      statusCounts[observation.status as keyof typeof statusCounts] += 1;
    }
    if (
      observation.value !== null
      && observation.status === "review_required"
    ) {
      computed += 1;
    }
  }
  assert(
    isRecord(value.summary)
      && Object.keys(value.summary).sort().join(",")
        === ["pass", "fail", "review_required", "not_measured", "invalid", "total", "computed"].sort().join(",")
      && value.summary.total === value.observation_count
      && value.summary.pass === 0
      && value.summary.fail === 0
      && value.summary.review_required === statusCounts.review_required
      && value.summary.not_measured === statusCounts.not_measured
      && value.summary.invalid === statusCounts.invalid
      && value.summary.computed === computed,
    "stress batch summary 与 items 不一致",
  );
  assert(
    isRecord(value.progress)
      && Object.keys(value.progress).sort().join(",")
        === ["completed_cases", "total_cases", "completed_items", "total_items"].sort().join(",")
      && value.progress.total_cases === expectedCaseCount
      && value.progress.total_items === value.observation_count
      && value.progress.completed_cases
        === cases.filter((entry) => entry.status === "completed" || entry.status === "failed").length
      && value.progress.completed_items
        === observations.filter((entry) => entry.status !== "pending" && entry.status !== "running").length,
    "stress batch progress 与 cases/items 不一致",
  );
  const status = String(value.status);
  assert(BATCH_STATUSES.has(status), "stress batch 状态无效");
  if (status === "completed") {
    assert(
      cases.every((entry) => entry.status === "completed")
        && value.progress.completed_cases === expectedCaseCount
        && value.progress.completed_items === value.observation_count
        && observations.every((entry) => entry.status !== "pending" && entry.status !== "running"),
      "completed stress batch 仍有未完成 case",
    );
  }
  return {
    ...value,
    schema_version: "scene-condition-batch-v1",
    batch_id: value.batch_id as string,
    dataset_id: expected.datasetId,
    dataset_name: stressDatasetName(expected.scope, expected.mode),
    dataset_manifest_sha256: value.dataset_manifest_sha256 as string,
    dataset_assets_sha256: value.dataset_assets_sha256 as string,
    algorithm_cluster_id: expected.algorithmClusterId,
    configuration: value.configuration as Record<string, unknown>,
    catalog_sha256: value.catalog_sha256 as string,
    cli_sha256: value.cli_sha256 as string,
    status: status as StressBatch["status"],
    created_at: createdAt,
    updated_at: updatedAt,
    test_scope: expected.scope,
    ...(value.stress_scope === expected.scope
      ? { stress_scope: expected.scope }
      : {}),
    selected_mode: expected.mode,
    selected_scene: "baseline",
    standard_compliance: false,
    formal_test_status: "not_measured",
    contract_item_count: contracts.length,
    case_count: expectedCaseCount,
    preview_condition_group_count: STRESS_CASE_COUNT,
    observation_count: observations.length,
    category_counts: categoryCounts,
    progress: {
      completed_cases: value.progress.completed_cases,
      total_cases: expectedCaseCount,
      completed_items: value.progress.completed_items,
      total_items: observations.length,
    },
    summary: {
      total: observations.length,
      computed,
      pass: 0,
      fail: 0,
      review_required: statusCounts.review_required,
      not_measured: statusCounts.not_measured,
      invalid: statusCounts.invalid,
    },
    cases,
    items: observations,
  };
}

function txUnavailable(reason: string): StressTxEvidence {
  return { ready: false, url: "", visual: undefined, reason };
}

export function stressTxEvidence(
  caseRecord: StressPreviewCase,
  batch: StressBatch | null,
  manifestSha256: string,
): StressTxEvidence {
  if (!batch) return txUnavailable("尚未执行 stress workload");
  if (batch.dataset_manifest_sha256 !== manifestSha256) {
    return txUnavailable("批次 manifest 与当前 stress 数据集不一致");
  }
  if (
    batch.test_scope !== (
      stressCategories("echo").includes(caseRecord.provenance.workload_category)
        ? "echo"
        : "double_talk"
    )
      || batch.selected_scene !== caseRecord.conditions.scene
  ) {
    return txUnavailable("stress case 与批次 scope 或 scene 不一致");
  }
  const candidate = batch.cases.find((entry) => entry.case_id === caseRecord.case_id);
  if (!candidate) return txUnavailable("批次缺少该 stress case");
  if (candidate.status !== "completed") {
    return txUnavailable("stress case 尚未完成，不能下载 TX");
  }
  if (
    candidate.conditions.mode !== caseRecord.conditions.mode
      || candidate.conditions.scene !== caseRecord.conditions.scene
      || candidate.contract_item_ids[0] !== caseRecord.contract_item_ids[0]
      || !sameConditions(candidate.conditions, caseRecord.conditions)
      || !sameInputHashes(candidate.signals_sha256, caseRecord.signals_sha256)
      || candidate.report_id === null
      || candidate.output_sha256 === undefined
      || candidate.tx_visual === undefined
  ) {
    return txUnavailable("stress case 条件、合同、输入或 TX 证据不一致");
  }
  const contract = batch.items.find(
    (entry) => entry.case_id === candidate.case_id,
  );
  if (!contract || contract.item_id !== caseRecord.contract_item_ids[0]) {
    return txUnavailable("stress case 缺少绑定合同观测");
  }
  if (
    contract.evidence_binding.case_id !== candidate.case_id
      || contract.evidence_binding.report_id !== candidate.report_id
      || contract.evidence_binding.batch_id !== batch.batch_id
      || contract.evidence_binding.dataset_id !== batch.dataset_id
      || !sameOutputHashRecord(
        contract.evidence_binding.output_sha256,
        candidate.output_sha256,
      )
  ) {
    return txUnavailable("stress case 观测证据绑定不一致");
  }
  try {
    readStressDiagnostics(candidate.stress_diagnostics, {
      caseId: candidate.case_id,
      txSha256: candidate.output_sha256.tx,
      scope: batch.test_scope,
      mode: batch.selected_mode,
      category: candidate.provenance.workload_category,
      segments: candidate.provenance.activity_segments,
    });
  } catch {
    return txUnavailable("stress diagnostics 证据无效");
  }
  if (candidate.tx_visual.source_sha256 !== candidate.output_sha256.tx) {
    return txUnavailable("stress TX 可视化哈希不一致");
  }
  return {
    ready: true,
    url: `${STRESS_BATCHES_URL}/${encodeURIComponent(batch.batch_id)}/cases/${encodeURIComponent(candidate.case_id)}/outputs/tx?v=${encodeURIComponent(candidate.output_sha256.tx)}`,
    visual: candidate.tx_visual,
    reason: "",
  };
}

export function stressRxEvidence(
  caseRecord: StressPreviewCase,
  batch: StressBatch | null,
  manifestSha256: string,
): StressRxEvidence {
  const txEvidence = stressTxEvidence(
    caseRecord,
    batch,
    manifestSha256,
  );
  if (!txEvidence.ready || !batch) {
    return {
      ready: false,
      url: "",
      visual: undefined,
      reason: txEvidence.reason,
    };
  }
  const candidate = batch.cases.find(
    (entry) => entry.case_id === caseRecord.case_id,
  );
  const rxSha256 = candidate?.output_sha256?.rx;
  if (!rxSha256) {
    return {
      ready: false,
      url: "",
      visual: undefined,
      reason: "该历史 stress 批次未记录 VCP RX 处理后回声参考",
    };
  }
  if (
    !candidate?.rx_visual
    || candidate.rx_visual.source_sha256 !== rxSha256
  ) {
    return {
      ready: false,
      url: "",
      visual: undefined,
      reason: "stress RX 处理后可视化哈希不一致",
    };
  }
  return {
    ready: true,
    url: `${STRESS_BATCHES_URL}/${encodeURIComponent(
      batch.batch_id,
    )}/cases/${encodeURIComponent(
      candidate.case_id,
    )}/outputs/rx?v=${encodeURIComponent(rxSha256)}`,
    visual: candidate.rx_visual,
    reason: "",
  };
}

function sameOutputHashRecord(
  left: unknown,
  right: { tx: string; rx?: string },
): boolean {
  return isRecord(left)
    && left.tx === right.tx
    && left.rx === right.rx;
}
