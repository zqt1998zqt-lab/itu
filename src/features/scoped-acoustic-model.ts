import { FIVE_SIGNAL_MODES, type FiveSignalMode } from "./five-signal-flow-model.ts";

export type AcousticTestScope = "basic_downlink" | "echo" | "double_talk" | "background_noise";
export const BACKGROUND_NOISE_SCENES = ["sce1", "sce2", "sce3", "sce4"] as const;
export type BackgroundNoiseScene = (typeof BACKGROUND_NOISE_SCENES)[number];
export const SCOPED_INPUT_SIGNAL_ROLES = [
  "microphone",
  "far_end",
  "reference",
  "noise",
] as const;
export const SCOPED_PREVIEW_VISUAL_SCHEMA =
  "scoped-acoustic-visual-v1" as const;
export const SCENE_CONDITION_PREVIEW_COUNT = 20 as const;
export const SCENE_CONDITION_SCHEMA_VERSION =
  "scene-condition-preview-v1" as const;
export const SCENE_CONDITION_DATASET_KIND =
  "background_noise_idle_preview" as const;
export const SCENE_CONDITION_CONTRACT_ITEM_IDS = [
  "WB-wired:B24",
  "WB-wired:B25",
  "WB-wired:B26",
  "WB-wired:B27",
  "WB-wired:B28",
] as const;
export type ScopedInputSignalRole =
  (typeof SCOPED_INPUT_SIGNAL_ROLES)[number];
export const BACKGROUND_NOISE_SCENE_LABELS: Record<BackgroundNoiseScene, string> = {
  sce1: "怠速 / 静止",
  sce2: "60 km/h 市区",
  sce3: "120 km/h 高速",
  sce4: "空调大风档",
};
export const SCENE_SUBPAGE_IDS = [
  "execute",
  "results",
  "waveforms",
  "description",
] as const;
export type SceneSubpageId = (typeof SCENE_SUBPAGE_IDS)[number];
export const SCENE_SUBPAGES: readonly {
  id: SceneSubpageId;
  label: string;
  hint: string;
}[] = [
  { id: "execute", label: "执行", hint: "测试集、VCP 与批次" },
  { id: "results", label: "结果", hint: "历史、状态与逐项证据" },
  { id: "waveforms", label: "波形", hint: "四路输入、TX 波形与梅尔频谱" },
  { id: "description", label: "说明", hint: "流程树、边界与合同目录" },
];
export const SCENE_SUBPAGE_LABELS: Record<SceneSubpageId, string> = {
  execute: "执行",
  results: "结果",
  waveforms: "波形",
  description: "说明",
};

export function sceneSubpageRoute(hash: string): string {
  const raw = hash.startsWith("#") ? hash.slice(1) : hash;
  const queryIndex = raw.indexOf("?");
  return queryIndex === -1 ? raw : raw.slice(0, queryIndex);
}

export function sceneSubpageFromHash(
  hash: string,
  route: string,
): SceneSubpageId {
  if (sceneSubpageRoute(hash) !== route) return "execute";
  const raw = hash.startsWith("#") ? hash.slice(1) : hash;
  const queryIndex = raw.indexOf("?");
  if (queryIndex === -1) return "execute";
  const values = new URLSearchParams(raw.slice(queryIndex + 1)).getAll("tab");
  if (values.length !== 1) return "execute";
  return SCENE_SUBPAGE_IDS.includes(values[0] as SceneSubpageId)
    ? (values[0] as SceneSubpageId)
    : "execute";
}

export function sceneSubpageHash(
  route: string,
  subpage: SceneSubpageId,
): string {
  return subpage === "execute" ? `#${route}` : `#${route}?tab=${subpage}`;
}

export const BACKGROUND_NOISE_SCENE_CELLS: Record<BackgroundNoiseScene, ReadonlySet<string>> = {
  sce1: new Set(["B24", "B25", "B26", "B27", "B28"]),
  sce2: new Set(["B29", "B30", "B31", "B32", "B33"]),
  sce3: new Set(["B34", "B35", "B36", "B37", "B38"]),
  sce4: new Set(["B39", "B40", "B41", "B42", "B43"]),
};
export type ScopedTestPageProps = {
  algorithmClusterId: string;
  mode: FiveSignalMode;
  onModeChange: (mode: FiveSignalMode) => void;
  onAlgorithmClusterChange: (id: string) => void;
  externalRunning?: boolean;
  onBusyChange?: (busy: boolean) => void;
  pageVisible?: boolean;
};
export type ScopedTestDefinition = {
  scope: AcousticTestScope;
  route:
    | "basic-downlink"
    | "echo"
    | "double-talk"
    | "background-noise"
    | "background-noise-idle"
    | "background-noise-urban"
    | "background-noise-highway"
    | "background-noise-hvac";
  apiRoute?: string;
  previewMode?: "idle-20-condition";
  scene?: BackgroundNoiseScene;
  title: string;
  eyebrow: string;
  description: string;
  boundary: string;
  inputTitle: string;
  inputs: string[];
  stages: { label: string; detail: string }[];
  outputKind: "tx" | "rx";
  methods: Record<string, { label: string; detail: string }>;
};

export const SCOPED_STATUS_LABELS = {
  pending: "等待计算", running: "计算中", pass: "工程通过", fail: "工程失败",
  review_required: "需复核", not_measured: "不可测", invalid: "无效",
};
export const SCOPED_BATCH_LABELS = {
  queued: "排队中", running: "执行中", completed: "已完成", failed: "执行失败", cancelled: "已取消",
};
export type ScopedItem = {
  id: string;
  source_cell: string;
  test_name: string;
  family: string;
  direction: string;
  unit: string;
  lower_limit: number | string | null;
  upper_limit: number | string | null;
};
export type ScopedResultItem = Partial<ScopedItem> & {
  item_id: string;
  mode: FiveSignalMode;
  status: keyof typeof SCOPED_STATUS_LABELS;
  value: number | null;
  metric?: string | null;
  reason?: string;
  evidence_binding?: {
    mode: string; direction: string; scene: string;
    volume: string | number | null; window_ms: number | null;
    input_level_db: number | null; case_id: string; report_id: string | null;
    batch_id: string; dataset_id: string;
    input_sha256?: Record<string, string>;
    output_sha256?: Record<string, string>;
  };
};
export type ScopedCatalog = {
  test_scope: AcousticTestScope;
  mode: FiveSignalMode;
  scene?: BackgroundNoiseScene | null;
  item_count: number;
  items: ScopedItem[];
};
export type ScopedDataset = {
  dataset_id: string;
  name: string;
  item_count: number;
  case_count: number;
  created_at: string;
  manifest_sha256?: string;
  assets_sha256?: string;
};
export type ScopedDatasetView = {
  datasets: ScopedDataset[];
  scene_case_count?: number;
  scene_item_count?: number;
  total_dataset_count?: number;
};
export type ScopedProgress = { completed_cases: number; total_cases: number; completed_items: number; total_items: number };
export type ScopedSummary = { total: number; computed: number; pass: number; fail: number; review_required: number; not_measured: number; invalid: number };
export type ScopedBatchSummary = {
  batch_id: string; dataset_id: string; dataset_name: string; algorithm_cluster_id: string;
  dataset_manifest_sha256?: string;
  test_scope: AcousticTestScope; selected_mode: FiveSignalMode;
  selected_scene?: BackgroundNoiseScene | null; created_at: string;
  status: keyof typeof SCOPED_BATCH_LABELS; progress: ScopedProgress; summary: ScopedSummary;
};
export type ScopedSnapshot = ScopedBatchSummary & {
  standard_compliance: false; items: ScopedResultItem[];
  cases: { case_id: string; report_id: string; status: string; error?: string }[];
  error?: string | null;
};
export type ScopedInputPreviewCase = {
  case_id: string;
  conditions: {
    mode: FiveSignalMode;
    direction: string;
    scene: string;
    volume: string | number | null;
    window_ms: number | null;
    input_level_db: number | null;
  };
  families: string[];
  signals: string[];
  signals_sha256: Record<string, string>;
};
export type ScopedPreviewVisualTrack = {
  source_sha256: string;
  duration_s: number;
  sample_rate: number;
  channels: number;
  peak: number;
  wave_scale: number;
  wave_min_i16: string;
  wave_max_i16: string;
  mel_i16: string;
  mel_ceiling_db: number;
};
export type ScopedPreviewVisuals = {
  schema: typeof SCOPED_PREVIEW_VISUAL_SCHEMA;
  waveform_bins: 720;
  mel_bands: 40;
  mel_frames: 240;
  mel_fft_size: 512;
  mel_hop_ms: 10;
  mel_dynamic_range_db: 80;
  mel_db_offset_db: -200;
  mel_db_step_db: 0.1;
  cases: Record<
    string,
    Partial<Record<ScopedInputSignalRole, ScopedPreviewVisualTrack>>
  >;
};
export type CompleteScopedPreviewVisuals =
  Omit<ScopedPreviewVisuals, "cases"> & {
    cases: Record<
      string,
      Record<ScopedInputSignalRole, ScopedPreviewVisualTrack>
    >;
  };
export const SCENE_CONDITION_DURATION_S = 12 as const;
export const SCENE_CONDITION_SAMPLE_RATE = 16_000 as const;
export const SCENE_CONDITION_CHANNELS: Record<
  ScopedInputSignalRole,
  1 | 2
> = {
  microphone: 2,
  reference: 1,
  far_end: 1,
  noise: 1,
};
export type ScopedBackgroundNoisePreview = {
  dataset_id: string;
  dataset_name: string;
  dataset_manifest_sha256: string;
  dataset_assets_sha256: string;
  test_scope: "background_noise";
  mode: FiveSignalMode;
  scene: BackgroundNoiseScene;
  cases: ScopedInputPreviewCase[];
  visuals?: ScopedPreviewVisuals;
};
export type ScopedSceneConditionDatasetView = {
  test_scope: "background_noise";
  mode: FiveSignalMode;
  scene: BackgroundNoiseScene;
  datasets: {
    dataset_id: string;
    dataset_name: string;
    test_scope: "background_noise";
    mode: FiveSignalMode;
    scene: BackgroundNoiseScene;
    contract_item_count: 5;
    preview_condition_group_count: typeof SCENE_CONDITION_PREVIEW_COUNT;
    standard_compliance: false;
    formal_test_status: "not_measured";
  }[];
  preview_condition_group_count: typeof SCENE_CONDITION_PREVIEW_COUNT;
  contract_item_count: 5;
  standard_compliance: false;
  formal_test_status: "not_measured";
};
export type ScopedSceneConditionPreview = Omit<
  ScopedBackgroundNoisePreview,
  "cases" | "visuals"
> & {
  cases: (ScopedInputPreviewCase & {
    contract_item_ids: string[];
    provenance: Record<string, unknown>;
  })[];
  visuals: CompleteScopedPreviewVisuals;
  preview_condition_group_count: typeof SCENE_CONDITION_PREVIEW_COUNT;
  contract_item_count: 5;
  standard_compliance: false;
  formal_test_status: "not_measured";
  schema_version: typeof SCENE_CONDITION_SCHEMA_VERSION;
  dataset_kind: typeof SCENE_CONDITION_DATASET_KIND;
};

const MODES = new Set<string>(FIVE_SIGNAL_MODES);
const SCOPES = new Set<string>(["basic_downlink", "echo", "double_talk", "background_noise"]);
const INPUT_SIGNAL_ROLES = new Set<string>(SCOPED_INPUT_SIGNAL_ROLES);
const ID = /^[a-f0-9]{32}$/;
const SHA256 = /^[a-f0-9]{64}$/;
const BASE64 = /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/;
const PREVIEW_VISUAL_SCHEMA = SCOPED_PREVIEW_VISUAL_SCHEMA;
const PREVIEW_WAVEFORM_BINS = 720;
const PREVIEW_MEL_BANDS = 40;
const PREVIEW_MEL_FRAMES = 240;
const PREVIEW_MEL_FFT_SIZE = 512;
const PREVIEW_MEL_HOP_MS = 10;
const PREVIEW_MEL_DYNAMIC_RANGE_DB = 80;
const PREVIEW_MEL_DB_OFFSET_DB = -200;
const PREVIEW_MEL_DB_STEP_DB = 0.1;
const record = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null && !Array.isArray(value);
const count = (value: unknown): value is number => typeof value === "number" && Number.isInteger(value) && value >= 0;
const limit = (value: unknown) => value === null || typeof value === "string" || typeof value === "number" && Number.isFinite(value);

function base64Length(byteLength: number): number {
  return Math.ceil(byteLength / 3) * 4;
}

function readVisualTrack(
  value: unknown,
  expectedSha256: string,
): ScopedPreviewVisualTrack {
  if (
    !record(value)
    || value.source_sha256 !== expectedSha256
    || typeof value.duration_s !== "number"
    || !Number.isFinite(value.duration_s)
    || value.duration_s <= 0
    || !count(value.sample_rate)
    || value.sample_rate < 8_000
    || value.sample_rate > 384_000
    || !count(value.channels)
    || value.channels === 0
    || typeof value.peak !== "number"
    || !Number.isFinite(value.peak)
    || value.peak < 0
    || typeof value.wave_scale !== "number"
    || !Number.isFinite(value.wave_scale)
    || value.wave_scale < 1
    || typeof value.mel_ceiling_db !== "number"
    || !Number.isFinite(value.mel_ceiling_db)
    || typeof value.wave_min_i16 !== "string"
    || typeof value.wave_max_i16 !== "string"
    || typeof value.mel_i16 !== "string"
    || !BASE64.test(value.wave_min_i16)
    || !BASE64.test(value.wave_max_i16)
    || !BASE64.test(value.mel_i16)
    || value.wave_min_i16.length !== base64Length(PREVIEW_WAVEFORM_BINS * 2)
    || value.wave_max_i16.length !== base64Length(PREVIEW_WAVEFORM_BINS * 2)
    || value.mel_i16.length !== base64Length(
      PREVIEW_MEL_BANDS * PREVIEW_MEL_FRAMES * 2,
    )
  ) {
    throw Error("背景噪声预计算波形或梅尔频谱格式无效");
  }
  return value as ScopedPreviewVisualTrack;
}

function readPreviewVisuals(
  value: unknown,
  cases: ScopedInputPreviewCase[],
): ScopedPreviewVisuals {
  if (
    !record(value)
    || value.schema !== PREVIEW_VISUAL_SCHEMA
    || value.waveform_bins !== PREVIEW_WAVEFORM_BINS
    || value.mel_bands !== PREVIEW_MEL_BANDS
    || value.mel_frames !== PREVIEW_MEL_FRAMES
    || value.mel_fft_size !== PREVIEW_MEL_FFT_SIZE
    || value.mel_hop_ms !== PREVIEW_MEL_HOP_MS
    || value.mel_dynamic_range_db !== PREVIEW_MEL_DYNAMIC_RANGE_DB
    || value.mel_db_offset_db !== PREVIEW_MEL_DB_OFFSET_DB
    || value.mel_db_step_db !== PREVIEW_MEL_DB_STEP_DB
    || !record(value.cases)
  ) {
    throw Error("背景噪声预计算可视化参数无效");
  }
  const expectedCases = new Map(cases.map(item => [item.case_id, item]));
  if (
    Object.keys(value.cases).length !== expectedCases.size
    || Object.keys(value.cases).some(caseId => !expectedCases.has(caseId))
  ) {
    throw Error("背景噪声预计算可视化条件组集合不一致");
  }
  for (const [caseId, previewCase] of expectedCases) {
    const tracks = value.cases[caseId];
    if (!record(tracks)) {
      throw Error("背景噪声预计算可视化条件组格式无效");
    }
    const expectedRoles = new Set(previewCase.signals);
    if (
      Object.keys(tracks).length !== expectedRoles.size
      || Object.keys(tracks).some(role => !expectedRoles.has(role))
    ) {
      throw Error("背景噪声预计算可视化输入角色不完整");
    }
    for (const role of previewCase.signals as ScopedInputSignalRole[]) {
      readVisualTrack(tracks[role], previewCase.signals_sha256[role]);
    }
  }
  return value as ScopedPreviewVisuals;
}

export function scopedItemCount(
  scope: AcousticTestScope,
  mode: FiveSignalMode,
  scene?: BackgroundNoiseScene,
): number {
  if (!SCOPES.has(scope) || !MODES.has(mode)) throw Error("测试范围或业务模式无效");
  if (scene !== undefined) {
    if (scope !== "background_noise" || !BACKGROUND_NOISE_SCENES.includes(scene)) {
      throw Error("只有背景噪声测试可以指定有效场景");
    }
  }
  if (scope === "basic_downlink") return mode.startsWith("SWB-") ? 8 : 9;
  if (scope === "echo") return mode.startsWith("SWB-") ? 3 : 2;
  if (scope === "double_talk") return 2;
  return scene === undefined ? 20 : 5;
}

export function scopedFamilyMatches(scope: AcousticTestScope, item: Pick<ScopedItem, "family" | "direction">): boolean {
  if (!SCOPES.has(scope)) return false;
  if (scope === "basic_downlink") return item.direction === "receive"
    && ["rlr", "rlr_linearity", "frequency_response", "listening_quality", "idle_noise_level", "idle_noise_peaks", "out_of_band"].includes(item.family);
  if (item.direction !== "send") return false;
  if (scope === "echo") return item.family === "tclw";
  if (scope === "double_talk") return item.family === "double_talk_classification";
  return ["g_mos", "n_mos", "s_mos", "cng_level", "cng_spectrum"].includes(item.family);
}

export function readScopedCatalog(
  value: unknown,
  scope: AcousticTestScope,
  mode: FiveSignalMode,
  scene?: BackgroundNoiseScene,
): ScopedCatalog {
  const expected = scopedItemCount(scope, mode, scene);
  if (!record(value) || value.test_scope !== scope || value.mode !== mode || value.item_count !== expected
    || scene !== undefined && value.scene !== scene
    || !Array.isArray(value.items) || value.items.length !== expected) throw Error("合同目录的范围、模式、场景或数量不匹配");
  const sceneCells = scene === undefined ? null : BACKGROUND_NOISE_SCENE_CELLS[scene];
  const seen = new Set<string>();
  const cells = new Set<string>();
  for (const item of value.items) {
    if (!record(item) || !["id", "source_cell", "test_name", "family", "direction", "unit"].every(key => typeof item[key] === "string")
      || !/^B(?:[2-9]|[1-3][0-9]|4[0-3])$/.test(String(item.source_cell))
      || item.id !== `${mode}:${item.source_cell}` || cells.has(String(item.source_cell))
      || !limit(item.lower_limit) || !limit(item.upper_limit) || seen.has(String(item.id))
      || sceneCells !== null && !sceneCells.has(String(item.source_cell))
      || !scopedFamilyMatches(scope, item as ScopedItem)) throw Error("合同目录包含重复或范围外测项");
    seen.add(String(item.id));
    cells.add(String(item.source_cell));
  }
  return value as ScopedCatalog;
}

function readSummary(
  value: unknown,
  scope: AcousticTestScope,
  scene?: BackgroundNoiseScene,
): ScopedBatchSummary {
  if (!record(value) || value.test_scope !== scope || !MODES.has(String(value.selected_mode))
    || !ID.test(String(value.batch_id)) || !ID.test(String(value.dataset_id))
    || typeof value.dataset_name !== "string" || typeof value.algorithm_cluster_id !== "string"
    || !value.algorithm_cluster_id || typeof value.created_at !== "string" || !Number.isFinite(Date.parse(value.created_at))
    || value.dataset_manifest_sha256 !== undefined && !SHA256.test(String(value.dataset_manifest_sha256))
    || scene !== undefined && value.selected_scene !== scene
    || !Object.hasOwn(SCOPED_BATCH_LABELS, String(value.status)) || !record(value.progress) || !record(value.summary)) {
    throw Error("批次身份或历史记录格式无效");
  }
  const expected = scopedItemCount(
    scope,
    value.selected_mode as FiveSignalMode,
    scene,
  );
  const progress = value.progress;
  const summary = value.summary;
  if (!["completed_cases", "total_cases", "completed_items", "total_items"].every(key => count(progress[key]))
    || !["total", "computed", "pass", "fail", "review_required", "not_measured", "invalid"].every(key => count(summary[key]))
    || progress.total_items !== expected || summary.total !== expected
    || Number(progress.completed_items) > expected || Number(progress.completed_cases) > Number(progress.total_cases)
    || Number(summary.computed) > expected
    || ["pass", "fail", "review_required", "not_measured", "invalid"].some(key => Number(summary[key]) > expected)
    || ["pass", "fail", "review_required", "not_measured", "invalid"].reduce((sum, key) => sum + Number(summary[key]), 0) > expected) {
    throw Error("批次进度或统计与所选范围不一致");
  }
  return value as ScopedBatchSummary;
}

export function readScopedHistory(
  value: unknown,
  scope: AcousticTestScope,
  scene?: BackgroundNoiseScene,
): ScopedBatchSummary[] {
  if (!Array.isArray(value)) throw Error("历史记录必须为列表");
  return value.map(item => readSummary(item, scope, scene));
}

export function readScopedSnapshot(value: unknown, scope: AcousticTestScope, expected: {
  batchId?: string; datasetId?: string; algorithmClusterId?: string;
  mode?: FiveSignalMode; scene?: BackgroundNoiseScene;
} = {}): ScopedSnapshot {
  const summary = readSummary(value, scope, expected.scene);
  const row = value as Record<string, unknown>;
  if (row.standard_compliance !== false || !Array.isArray(row.items) || row.items.length !== summary.summary.total
    || !Array.isArray(row.cases) || expected.batchId && summary.batch_id !== expected.batchId
    || expected.datasetId && summary.dataset_id !== expected.datasetId
    || expected.algorithmClusterId && summary.algorithm_cluster_id !== expected.algorithmClusterId
    || expected.mode && summary.selected_mode !== expected.mode
    || expected.scene && summary.selected_scene !== expected.scene) throw Error("快照与请求身份或合规边界不一致");
  const seen = new Set<string>();
  const actual = { total: row.items.length, computed: 0, pass: 0, fail: 0, review_required: 0, not_measured: 0, invalid: 0 };
  for (const item of row.items) {
    if (!record(item) || typeof item.item_id !== "string" || seen.has(item.item_id)
      || typeof item.source_cell !== "string" || !/^B(?:[2-9]|[1-3][0-9]|4[0-3])$/.test(item.source_cell)
      || item.item_id !== `${summary.selected_mode}:${item.source_cell}`
      || item.mode !== summary.selected_mode || !Object.hasOwn(SCOPED_STATUS_LABELS, String(item.status))
      || !(item.value === null || typeof item.value === "number" && Number.isFinite(item.value))) throw Error("逐项结果的身份、状态或数值无效");
    seen.add(item.item_id);
    const binding = item.evidence_binding;
    if (!record(binding) || binding.mode !== summary.selected_mode || binding.batch_id !== summary.batch_id
      || binding.dataset_id !== summary.dataset_id || typeof binding.case_id !== "string"
      || typeof binding.scene !== "string" || !["baseline", "sce1", "sce2", "sce3", "sce4"].includes(binding.scene)
      || expected.scene && binding.scene !== expected.scene
      || binding.direction !== item.direction || !["nominal", "recommended", "maximum"].includes(String(binding.volume))
      || !(binding.window_ms === null || binding.window_ms === 50 || binding.window_ms === 100)
      || typeof binding.input_level_db !== "number" || !Number.isFinite(binding.input_level_db)
      || !(binding.report_id === null || typeof binding.report_id === "string")) throw Error("逐项证据绑定无效");
    for (const key of ["input_sha256", "output_sha256"] as const) {
      const hashes = binding[key];
      if (hashes !== undefined && (!record(hashes)
        || Object.values(hashes).some(hash => typeof hash !== "string" || !SHA256.test(hash)))) {
        throw Error("逐项证据哈希绑定无效");
      }
    }
    if (typeof item.family !== "string" || typeof item.direction !== "string"
      || !scopedFamilyMatches(scope, item as ScopedItem)) throw Error("报告包含测试范围外的测项");
    if ((item.status === "not_measured" || item.status === "invalid") && item.value !== null) throw Error("不可测或无效项不能计为有效数值");
    if (item.status !== "pending" && item.status !== "running") {
      actual[item.status as keyof Omit<ScopedSummary, "total" | "computed">] += 1;
      if (typeof item.value === "number" && item.status !== "invalid" && item.status !== "not_measured") actual.computed += 1;
    } else if (item.value !== null) throw Error("待执行项不能携带已计算数值");
  }
  if (Object.entries(actual).some(([key, value]) => summary.summary[key as keyof ScopedSummary] !== value)) throw Error("汇总统计与逐项结果不一致");
  if (summary.status === "completed" && (summary.progress.completed_items !== summary.summary.total
    || summary.progress.completed_cases !== summary.progress.total_cases
    || actual.pass + actual.fail + actual.review_required + actual.not_measured + actual.invalid !== actual.total)) throw Error("未完成全部条件组，不能标记计算完成");
  if (row.cases.some(item => !record(item) || typeof item.case_id !== "string" || typeof item.report_id !== "string"
    || typeof item.status !== "string")) throw Error("条件组记录无效");
  if (new Set(row.cases.map(item => (item as Record<string, unknown>).case_id)).size !== row.cases.length
    || row.cases.length !== summary.progress.completed_cases) throw Error("条件组数量或身份不一致");
  return value as ScopedSnapshot;
}

export function readScopedDatasets(value: unknown): ScopedDataset[] {
  if (!Array.isArray(value) || value.some(item => !record(item) || !ID.test(String(item.dataset_id))
    || typeof item.name !== "string" || !count(item.item_count) || !count(item.case_count)
    || typeof item.created_at !== "string"
    || item.manifest_sha256 !== undefined && !SHA256.test(String(item.manifest_sha256))
    || item.assets_sha256 !== undefined && !SHA256.test(String(item.assets_sha256)))) {
    throw Error("测试集列表格式无效");
  }
  return value as ScopedDataset[];
}

export function readScopedDatasetView(
  value: unknown,
  expected: { mode: FiveSignalMode; scene: BackgroundNoiseScene },
): ScopedDatasetView {
  if (
    !record(value)
    || value.test_scope !== "background_noise"
    || value.mode !== expected.mode
    || value.scene !== expected.scene
    || !count(value.scene_item_count)
    || value.scene_item_count !== 5
    || !count(value.scene_case_count)
    || value.scene_case_count <= 0
    || !count(value.total_dataset_count)
    || !Array.isArray(value.datasets)
    || value.datasets.length !== 1
  ) {
    throw Error("场景测试集视图格式无效");
  }
  const datasets = readScopedDatasets(value.datasets);
  return {
    datasets,
    scene_item_count: value.scene_item_count as number,
    scene_case_count: value.scene_case_count as number,
    total_dataset_count: value.total_dataset_count as number,
  };
}

export function readSceneConditionDatasetView(
  value: unknown,
  expected: { mode: FiveSignalMode; scene: BackgroundNoiseScene },
): ScopedSceneConditionDatasetView {
  if (
    !record(value)
    || value.test_scope !== "background_noise"
    || value.mode !== expected.mode
    || value.scene !== expected.scene
    || !count(value.preview_condition_group_count)
    || value.preview_condition_group_count !== SCENE_CONDITION_PREVIEW_COUNT
    || value.contract_item_count !== 5
    || value.standard_compliance !== false
    || value.formal_test_status !== "not_measured"
    || !Array.isArray(value.datasets)
    || value.datasets.length !== 1
  ) {
    throw Error("20 组工程预览数据集视图格式无效");
  }
  const dataset = value.datasets[0];
  if (
    !record(dataset)
    || typeof dataset.dataset_id !== "string"
    || !ID.test(dataset.dataset_id)
    || typeof dataset.dataset_name !== "string"
    || dataset.dataset_name.trim().length === 0
    || dataset.test_scope !== "background_noise"
    || dataset.mode !== expected.mode
    || dataset.scene !== expected.scene
    || dataset.contract_item_count !== 5
    || dataset.preview_condition_group_count !== SCENE_CONDITION_PREVIEW_COUNT
    || dataset.standard_compliance !== false
    || dataset.formal_test_status !== "not_measured"
  ) {
    throw Error("20 组工程预览数据集身份无效");
  }
  return {
    test_scope: "background_noise",
    mode: expected.mode,
    scene: expected.scene,
    datasets: [{
      dataset_id: dataset.dataset_id,
      dataset_name: dataset.dataset_name,
      test_scope: "background_noise",
      mode: expected.mode,
      scene: expected.scene,
      contract_item_count: 5,
      preview_condition_group_count: SCENE_CONDITION_PREVIEW_COUNT,
      standard_compliance: false,
      formal_test_status: "not_measured",
    }],
    preview_condition_group_count: SCENE_CONDITION_PREVIEW_COUNT,
    contract_item_count: 5,
    standard_compliance: false,
    formal_test_status: "not_measured",
  };
}

export function readScopedBackgroundNoisePreview(
  value: unknown,
  datasetId: string,
  mode: FiveSignalMode,
  scene: BackgroundNoiseScene,
): ScopedBackgroundNoisePreview {
  if (
    !record(value)
    || value.dataset_id !== datasetId
    || value.test_scope !== "background_noise"
    || value.mode !== mode
    || value.scene !== scene
    || typeof value.dataset_name !== "string"
    || !SHA256.test(String(value.dataset_manifest_sha256))
    || !SHA256.test(String(value.dataset_assets_sha256))
    || !Array.isArray(value.cases)
    || value.cases.length === 0
  ) {
    throw Error("背景噪声预览身份或格式无效");
  }
  const seen = new Set<string>();
  for (const item of value.cases) {
    if (
      !record(item)
      || typeof item.case_id !== "string"
      || seen.has(item.case_id)
      || !record(item.conditions)
      || item.conditions.mode !== mode
      || item.conditions.scene !== scene
      || !Array.isArray(item.families)
      || item.families.some((family) => typeof family !== "string")
      || !Array.isArray(item.signals)
      || item.signals.some((signal) => typeof signal !== "string")
      || item.signals.length !== SCOPED_INPUT_SIGNAL_ROLES.length
      || new Set(item.signals).size !== SCOPED_INPUT_SIGNAL_ROLES.length
      || item.signals.some((signal) => !INPUT_SIGNAL_ROLES.has(signal))
      || !record(item.signals_sha256)
      || Object.keys(item.signals_sha256).length !== item.signals.length
    ) {
      throw Error("背景噪声预览条件组身份无效");
    }
    const signalHashes = item.signals_sha256;
    if (
      item.signals.some(
        (signal) => !SHA256.test(String(signalHashes[signal])),
      )
    ) {
      throw Error("背景噪声预览输入哈希无效");
    }
    seen.add(item.case_id);
  }
  if (value.visuals !== undefined) {
    readPreviewVisuals(
      value.visuals,
      value.cases as ScopedInputPreviewCase[],
    );
  }
  return value as ScopedBackgroundNoisePreview;
}

export function readSceneConditionPreview(
  value: unknown,
  datasetId: string,
  mode: FiveSignalMode,
  scene: BackgroundNoiseScene,
): ScopedSceneConditionPreview {
  const preview = readScopedBackgroundNoisePreview(
    value,
    datasetId,
    mode,
    scene,
  );
  const row = value as Record<string, unknown>;
  if (
    !count(row.preview_condition_group_count)
    || row.preview_condition_group_count !== SCENE_CONDITION_PREVIEW_COUNT
    || preview.cases.length !== SCENE_CONDITION_PREVIEW_COUNT
    || row.contract_item_count !== 5
    || row.standard_compliance !== false
    || row.formal_test_status !== "not_measured"
    || row.schema_version !== SCENE_CONDITION_SCHEMA_VERSION
    || row.dataset_kind !== SCENE_CONDITION_DATASET_KIND
    || !record(row.visuals)
  ) {
    throw Error("20 组工程预览合同边界或格式无效");
  }
  const visuals = readPreviewVisuals(
    row.visuals,
    preview.cases as ScopedInputPreviewCase[],
  ) as CompleteScopedPreviewVisuals;
  for (const previewCase of preview.cases) {
    const tracks = visuals.cases[previewCase.case_id];
    for (const role of previewCase.signals as ScopedInputSignalRole[]) {
      const track = tracks[role];
      if (
        track.duration_s !== SCENE_CONDITION_DURATION_S
        || track.sample_rate !== SCENE_CONDITION_SAMPLE_RATE
        || track.channels !== SCENE_CONDITION_CHANNELS[role]
      ) {
        throw Error("20 组工程预览波形时长、采样率或通道数无效");
      }
    }
  }
  const coveredItems = new Set<string>();
  for (const item of preview.cases as Array<ScopedInputPreviewCase & {
    contract_item_ids?: unknown;
    provenance?: unknown;
  }>) {
    if (
      !item.case_id
      || !Array.isArray(item.contract_item_ids)
      || item.contract_item_ids.length === 0
      || item.contract_item_ids.some(
        (entry) => typeof entry !== "string"
          || !SCENE_CONDITION_CONTRACT_ITEM_IDS.includes(
            entry as (typeof SCENE_CONDITION_CONTRACT_ITEM_IDS)[number],
          ),
      )
      || !record(item.provenance)
      || item.provenance.engineering_preview !== true
      || item.provenance.actual_vehicle_recording !== false
      || !Array.isArray(item.families)
      || item.families.length === 0
      || item.families.some((family) => typeof family !== "string")
      || item.conditions.direction !== "send"
      || item.conditions.volume !== "nominal"
      || item.conditions.window_ms !== null
      || item.conditions.input_level_db !== 0
    ) {
      throw Error("20 组工程预览条件组合同或来源无效");
    }
    for (const itemId of item.contract_item_ids as string[]) {
      coveredItems.add(itemId);
    }
  }
  if (
    coveredItems.size !== SCENE_CONDITION_CONTRACT_ITEM_IDS.length
    || SCENE_CONDITION_CONTRACT_ITEM_IDS.some(
      (itemId) => !coveredItems.has(itemId),
    )
  ) {
    throw Error("20 组工程预览未覆盖 5 个合同测项");
  }
  return {
    ...(value as ScopedBackgroundNoisePreview),
    cases: preview.cases as ScopedSceneConditionPreview["cases"],
    visuals,
    preview_condition_group_count: SCENE_CONDITION_PREVIEW_COUNT,
    contract_item_count: 5,
    standard_compliance: false,
    formal_test_status: "not_measured",
    schema_version: SCENE_CONDITION_SCHEMA_VERSION,
    dataset_kind: SCENE_CONDITION_DATASET_KIND,
  };
}

export function isIdle20ConditionPreview(
  definition: Pick<ScopedTestDefinition, "previewMode" | "scene">,
  mode: FiveSignalMode,
): boolean {
  return definition.previewMode === "idle-20-condition"
    && definition.scene === "sce1"
    && mode === "WB-wired";
}

export function scopedCaseEvidenceIdentity(
  previewCase: ScopedInputPreviewCase,
  snapshot: ScopedSnapshot | null,
  datasetManifestSha256: string,
) {
  const caseRecord = snapshot?.cases.find(
    (entry) => entry.case_id === previewCase.case_id,
  );
  const caseItems =
    snapshot?.items.filter(
      (item) =>
        item.evidence_binding?.case_id === previewCase.case_id,
    ) ?? [];
  const reportMatches = Boolean(
    caseRecord &&
      caseItems.length > 0 &&
      caseItems.every(
        (item) =>
          item.evidence_binding?.report_id === caseRecord.report_id,
      ),
  );
  const completeInputHashes =
    caseItems.length > 0 &&
    caseItems.every((item) => {
      const hashes = item.evidence_binding?.input_sha256;
      return (
        hashes !== undefined &&
        SCOPED_INPUT_SIGNAL_ROLES.every(
          (role) =>
            typeof hashes[role] === "string" &&
            SHA256.test(hashes[role]),
        )
      );
    });
  const inputHashesMatch =
    completeInputHashes &&
    caseItems.every((item) => {
      const hashes = item.evidence_binding?.input_sha256;
      return SCOPED_INPUT_SIGNAL_ROLES.every(
        (role) =>
          hashes?.[role] === previewCase.signals_sha256[role],
      );
    });
  const manifestMatches =
    snapshot?.dataset_manifest_sha256 === datasetManifestSha256;
  return {
    caseRecord,
    caseItems,
    manifestMatches,
    reportMatches,
    completeInputHashes,
    inputHashesMatch,
    txReady:
      manifestMatches &&
      reportMatches &&
      inputHashesMatch &&
      caseRecord?.status === "completed",
  };
}

export function matchingScopedSnapshot(
  snapshot: ScopedSnapshot | null,
  mode: FiveSignalMode,
  algorithm: string,
  dataset: string,
  scene?: BackgroundNoiseScene,
): ScopedSnapshot | null {
  return snapshot?.selected_mode === mode
    && snapshot.algorithm_cluster_id === algorithm
    && snapshot.dataset_id === dataset
    && (scene === undefined || snapshot.selected_scene === scene)
    ? snapshot
    : null;
}

export function scopedValue(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toLocaleString("zh-CN", { maximumFractionDigits: 3 }) : "—";
}

export function scopedMetricUnit(metric: string | null | undefined): string {
  if (!metric) return "";
  if (metric.startsWith("dnsmos_") || metric.includes("mos_proxy")) return "预测分";
  if (metric.includes("dbfs")) return "dBFS";
  if (metric.includes("_db")) return "dB";
  if (metric.includes("_ms")) return "ms";
  if (metric.includes("percent")) return "%";
  if (metric.includes("ratio")) return "比例";
  return "";
}
