import type { FiveSignalMode } from "./five-signal-flow-model.ts";
import {
  SCOPED_INPUT_SIGNAL_ROLES,
  SCOPED_STATUS_LABELS,
  type ScopedInputPreviewCase,
  type ScopedPreviewVisualTrack,
  type ScopedSnapshot,
} from "./scoped-acoustic-model.ts";

export const SCENE_CONDITION_BATCHES_URL =
  "/api/alignment/acceptance/scene-condition-batches";
export const SCENE_CONDITION_DATASETS_URL =
  "/api/alignment/acceptance/scene-condition-datasets";
export const SCENE_CONDITION_SCHEMA_VERSION =
  "scene-condition-batch-v1" as const;
export const SCENE_CONDITION_BATCH_COUNT = 20 as const;
export const SCENE_CONDITION_OBSERVATION_COUNT = 50 as const;
export const SCENE_CONDITION_CONTRACT_ITEM_COUNT = 5 as const;
export const SCENE_CONDITION_ITEM_IDS = [
  "WB-wired:B24",
  "WB-wired:B25",
  "WB-wired:B26",
  "WB-wired:B27",
  "WB-wired:B28",
] as const;

export type SceneConditionItemId =
  (typeof SCENE_CONDITION_ITEM_IDS)[number];
export type SceneConditionBatchStatus =
  | "queued"
  | "running"
  | "completed"
  | "failed";
export type SceneConditionCaseStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed";
export type SceneConditionConditions =
  ScopedInputPreviewCase["conditions"];
export type SceneConditionInputHashes = Record<
  (typeof SCOPED_INPUT_SIGNAL_ROLES)[number],
  string
>;
export type SceneConditionOutputHashes = {
  tx: string;
};

export type SceneConditionEvidenceBinding = {
  conditions: SceneConditionConditions;
  case_id: string;
  report_id: string | null;
  batch_id: string;
  dataset_id: string;
  input_sha256: SceneConditionInputHashes;
  output_sha256?: SceneConditionOutputHashes;
};

export type SceneConditionObservation = {
  observation_id: string;
  item_id: SceneConditionItemId;
  case_id: string;
  mode: FiveSignalMode;
  status: keyof typeof SCOPED_STATUS_LABELS;
  value: number | null;
  metric: string | null;
  reason: string;
  family: string;
  test_name: string;
  source_cell: string;
  evidence_binding: SceneConditionEvidenceBinding;
};

export type SceneConditionCase = {
  case_id: string;
  status: SceneConditionCaseStatus;
  report_id: string | null;
  conditions: SceneConditionConditions;
  signals_sha256: SceneConditionInputHashes;
  input_sha256: SceneConditionInputHashes;
  output_sha256?: SceneConditionOutputHashes;
  tx_visual?: ScopedPreviewVisualTrack;
  error?: string;
};

export type SceneConditionBatch = Omit<
  ScopedSnapshot,
  | "cases"
  | "items"
  | "status"
  | "selected_mode"
  | "selected_scene"
  | "dataset_manifest_sha256"
> & {
  schema_version: typeof SCENE_CONDITION_SCHEMA_VERSION;
  dataset_manifest_sha256: string;
  dataset_assets_sha256: string;
  configuration: Record<string, unknown>;
  catalog_sha256: string;
  cli_sha256: string;
  updated_at: string;
  status: SceneConditionBatchStatus;
  selected_mode: "WB-wired";
  selected_scene: "sce1";
  formal_test_status: "not_measured";
  contract_item_count: typeof SCENE_CONDITION_CONTRACT_ITEM_COUNT;
  observation_count: typeof SCENE_CONDITION_OBSERVATION_COUNT;
  items: SceneConditionObservation[];
  cases: SceneConditionCase[];
};

export type SceneConditionCaseTxEvidence = {
  ready: boolean;
  url: string;
  visual: ScopedPreviewVisualTrack | undefined;
  reason: string;
};

const ID = /^[a-f0-9]{32}$/;
const SHA256 = /^[a-f0-9]{64}$/;
const BASE64 =
  /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/;
const CASE_STATUSES = new Set<SceneConditionCaseStatus>([
  "pending",
  "running",
  "completed",
  "failed",
]);
const BATCH_STATUSES = new Set<SceneConditionBatchStatus>([
  "queued",
  "running",
  "completed",
  "failed",
]);
const TERMINAL_ITEM_STATUSES = new Set<string>([
  "pass",
  "fail",
  "review_required",
  "not_measured",
  "invalid",
]);
const ITEM_IDS = new Set<string>(SCENE_CONDITION_ITEM_IDS);
const SPEECH_ITEM_IDS = new Set<string>([
  "WB-wired:B24",
  "WB-wired:B25",
  "WB-wired:B26",
]);
const CNG_ITEM_IDS = new Set<string>([
  "WB-wired:B27",
  "WB-wired:B28",
]);
const FAMILY_BY_ITEM: Record<SceneConditionItemId, string> = {
  "WB-wired:B24": "g_mos",
  "WB-wired:B25": "n_mos",
  "WB-wired:B26": "s_mos",
  "WB-wired:B27": "cng_level",
  "WB-wired:B28": "cng_spectrum",
};
const WAVEFORM_BINS = 720;
const MEL_BANDS = 40;
const MEL_FRAMES = 240;

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

function readConditions(value: unknown): SceneConditionConditions {
  assert(isRecord(value), "20 组批次条件组格式无效");
  const keys = Object.keys(value).sort();
  assert(
    keys.length === 6
      && keys[0] === "direction"
      && keys[1] === "input_level_db"
      && keys[2] === "mode"
      && keys[3] === "scene"
      && keys[4] === "volume"
      && keys[5] === "window_ms",
    "20 组批次条件组字段不完整",
  );
  assert(
    value.mode === "WB-wired"
      && value.direction === "send"
      && value.scene === "sce1"
      && value.volume === "nominal"
      && value.window_ms === null
      && value.input_level_db === 0,
    "20 组批次条件组超出 WB-wired / sce1 合同",
  );
  return {
    mode: "WB-wired",
    direction: "send",
    scene: "sce1",
    volume: "nominal",
    window_ms: null,
    input_level_db: 0,
  };
}

function readInputHashes(
  value: unknown,
  message: string,
): SceneConditionInputHashes {
  assert(isRecord(value), message);
  assert(
    Object.keys(value).length === SCOPED_INPUT_SIGNAL_ROLES.length,
    message,
  );
  const hashes = {} as SceneConditionInputHashes;
  for (const role of SCOPED_INPUT_SIGNAL_ROLES) {
    hashes[role] = readSha(value[role], message);
  }
  return hashes;
}

function readOutputHashes(
  value: unknown,
  message: string,
): SceneConditionOutputHashes {
  assert(
    isRecord(value)
      && Object.keys(value).length === 1
      && Object.hasOwn(value, "tx"),
    message,
  );
  return { tx: readSha(value.tx, message) };
}

function readOptionalOutputHashes(
  value: unknown,
  message: string,
): SceneConditionOutputHashes | undefined {
  if (value === undefined || value === null) return undefined;
  if (isRecord(value) && Object.keys(value).length === 0) {
    return undefined;
  }
  return readOutputHashes(value, message);
}

function readVisualTrack(
  value: unknown,
): ScopedPreviewVisualTrack {
  assert(isRecord(value), "20 组 TX 可视化格式无效");
  const sourceSha256 = readSha(
    value.source_sha256,
    "20 组 TX 可视化来源哈希无效",
  );
  assert(
    typeof value.duration_s === "number"
      && Number.isFinite(value.duration_s)
      && value.duration_s > 0,
    "20 组 TX 可视化时长无效",
  );
  assert(
    isCount(value.sample_rate)
      && value.sample_rate >= 8_000
      && value.sample_rate <= 384_000,
    "20 组 TX 可视化采样率无效",
  );
  assert(
    isCount(value.channels)
      && value.channels >= 1
      && value.channels <= 8,
    "20 组 TX 可视化通道数无效",
  );
  assert(
    typeof value.peak === "number"
      && Number.isFinite(value.peak)
      && value.peak >= 0,
    "20 组 TX 可视化峰值无效",
  );
  assert(
    typeof value.wave_scale === "number"
      && Number.isFinite(value.wave_scale)
      && value.wave_scale >= 1,
    "20 组 TX 可视化波形尺度无效",
  );
  assert(
    typeof value.mel_ceiling_db === "number"
      && Number.isFinite(value.mel_ceiling_db),
    "20 组 TX 可视化梅尔色标无效",
  );
  assert(
    typeof value.wave_min_i16 === "string"
      && typeof value.wave_max_i16 === "string"
      && typeof value.mel_i16 === "string"
      && BASE64.test(value.wave_min_i16)
      && BASE64.test(value.wave_max_i16)
      && BASE64.test(value.mel_i16)
      && value.wave_min_i16.length
        === Math.ceil((WAVEFORM_BINS * 2) / 3) * 4
      && value.wave_max_i16.length
        === Math.ceil((WAVEFORM_BINS * 2) / 3) * 4
      && value.mel_i16.length
        === Math.ceil((MEL_BANDS * MEL_FRAMES * 2) / 3) * 4,
    "20 组 TX 可视化预计算数组无效",
  );
  return {
    source_sha256: sourceSha256,
    duration_s: value.duration_s,
    sample_rate: value.sample_rate,
    channels: value.channels,
    peak: value.peak,
    wave_scale: value.wave_scale,
    wave_min_i16: value.wave_min_i16,
    wave_max_i16: value.wave_max_i16,
    mel_i16: value.mel_i16,
    mel_ceiling_db: value.mel_ceiling_db,
  };
}

function sameConditions(
  left: SceneConditionConditions,
  right: SceneConditionConditions,
): boolean {
  return left.mode === right.mode
    && left.direction === right.direction
    && left.scene === right.scene
    && left.volume === right.volume
    && left.window_ms === right.window_ms
    && left.input_level_db === right.input_level_db;
}

function sameInputHashes(
  left: SceneConditionInputHashes | undefined,
  right: SceneConditionInputHashes | undefined,
): boolean {
  return Boolean(left && right) && SCOPED_INPUT_SIGNAL_ROLES.every(
    (role) => left?.[role] === right?.[role],
  );
}

function sameOutputHashes(
  left: SceneConditionOutputHashes | undefined,
  right: SceneConditionOutputHashes | undefined,
): boolean {
  return left?.tx === right?.tx;
}

function readCase(value: unknown): SceneConditionCase {
  assert(isRecord(value), "20 组条件组记录格式无效");
  assert(
    typeof value.case_id === "string"
      && value.case_id.trim().length > 0,
    "20 组条件组 ID 无效",
  );
  assert(
    CASE_STATUSES.has(value.status as SceneConditionCaseStatus),
    "20 组条件组状态无效",
  );
  const status = value.status as SceneConditionCaseStatus;
  const reportId =
    value.report_id === null
      ? null
      : readId(value.report_id, "20 组条件组报告 ID 无效");
  const conditions = readConditions(value.conditions);
  const inputSha256 = readInputHashes(
    value.input_sha256,
    "20 组条件组四路输入哈希无效",
  );
  const signalsSha256 = readInputHashes(
    value.signals_sha256,
    "20 组条件组 manifest 四路信号哈希无效",
  );
  assert(
    sameInputHashes(inputSha256, signalsSha256),
    "20 组条件组 input_sha256 必须完全等于 signals_sha256",
  );
  const outputSha256 = readOptionalOutputHashes(
    value.output_sha256,
    "20 组条件组 TX 输出哈希无效",
  );
  const txVisual =
    value.tx_visual === undefined || value.tx_visual === null
      ? undefined
      : readVisualTrack(value.tx_visual);
  assert(
    value.error === undefined || typeof value.error === "string",
    "20 组条件组错误字段无效",
  );
  if (status === "completed") {
    assert(
      reportId !== null,
      "已完成条件组缺少报告 ID",
    );
    assert(
      outputSha256 !== undefined,
      "已完成条件组缺少 TX 输出哈希",
    );
    assert(
      txVisual !== undefined,
      "已完成条件组缺少 TX 波形与梅尔频谱",
    );
  }
  if (txVisual !== undefined && outputSha256 !== undefined) {
    assert(
      txVisual.source_sha256 === outputSha256.tx,
      "20 组 TX 可视化来源哈希与输出哈希不一致",
    );
  }
  return {
    case_id: value.case_id,
    status,
    report_id: reportId,
    conditions,
    signals_sha256: signalsSha256,
    input_sha256: inputSha256,
    ...(outputSha256 ? { output_sha256: outputSha256 } : {}),
    ...(txVisual ? { tx_visual: txVisual } : {}),
    ...(typeof value.error === "string" ? { error: value.error } : {}),
  };
}

function readObservation(
  value: unknown,
  cases: Map<string, SceneConditionCase>,
  batchId: string,
  datasetId: string,
): SceneConditionObservation {
  assert(isRecord(value), "20 组逐观测记录格式无效");
  assert(
    typeof value.case_id === "string"
      && value.case_id.trim().length > 0,
    "20 组逐观测条件组 ID 无效",
  );
  const caseId = value.case_id;
  const caseRecord = cases.get(caseId);
  assert(caseRecord !== undefined, "20 组逐观测绑定了未知条件组");
  assert(
    typeof value.item_id === "string"
      && ITEM_IDS.has(value.item_id),
    "20 组逐观测合同测项超出 B24–B28",
  );
  const itemId = value.item_id as SceneConditionItemId;
  const sourceCell = itemId.slice("WB-wired:".length);
  assert(
    typeof value.observation_id === "string"
      && value.observation_id === `${caseId}:${itemId}`,
    "20 组观测 ID 与条件组/合同测项不一致",
  );
  assert(
    value.mode === "WB-wired",
    "20 组逐观测模式不是 WB-wired",
  );
  assert(
    Object.hasOwn(SCOPED_STATUS_LABELS, String(value.status)),
    "20 组逐观测状态无效",
  );
  const status = value.status as keyof typeof SCOPED_STATUS_LABELS;
  assert(
    value.value === null
      || typeof value.value === "number" && Number.isFinite(value.value),
    "20 组逐观测数值不是有限数字或 null",
  );
  const numericValue =
    typeof value.value === "number" ? value.value : null;
  if (status === "pending" || status === "running") {
    assert(numericValue === null, "待执行观测不能携带数值");
  }
  if (status === "not_measured" || status === "invalid") {
    assert(numericValue === null, "不可测或无效观测不能携带数值");
  }
  if (status === "pass") {
    assert(numericValue !== null, "工程通过观测必须有有限数值");
  }
  if (caseRecord.status === "completed") {
    assert(
      TERMINAL_ITEM_STATUSES.has(status),
      "已完成条件组仍含未终态观测",
    );
  }
  assert(
    value.metric === null || typeof value.metric === "string",
    "20 组逐观测指标字段无效",
  );
  assert(
    typeof value.reason === "string"
      && typeof value.test_name === "string"
      && value.test_name.trim().length > 0
      && typeof value.family === "string",
    "20 组逐观测说明字段无效",
  );
  assert(
    value.family === FAMILY_BY_ITEM[itemId]
      && typeof value.source_cell === "string"
      && value.source_cell === sourceCell,
    "20 组逐观测合同族或源行不一致",
  );
  assert(isRecord(value.evidence_binding), "20 组逐观测证据绑定无效");
  const binding = value.evidence_binding;
  const conditions = readConditions({
    mode: binding.mode,
    direction: binding.direction,
    scene: binding.scene,
    volume: binding.volume,
    window_ms: binding.window_ms,
    input_level_db: binding.input_level_db,
  });
  assert(
    sameConditions(conditions, caseRecord.conditions),
    "20 组逐观测条件与条件组不一致",
  );
  assert(
    binding.case_id === caseId
      && binding.batch_id === batchId
      && binding.dataset_id === datasetId,
    "20 组逐观测批次、数据集或条件组身份不一致",
  );
  const bindingReportId =
    binding.report_id === null
      ? null
      : readId(
          binding.report_id,
          "20 组逐观测报告 ID 无效",
        );
  assert(
    bindingReportId === caseRecord.report_id,
    "20 组逐观测报告 ID 与条件组不一致",
  );
  const inputSha256 = readInputHashes(
    binding.input_sha256,
    "20 组逐观测四路输入哈希无效",
  );
  assert(
    sameInputHashes(inputSha256, caseRecord.input_sha256),
    "20 组逐观测输入哈希与条件组不一致",
  );
  const outputSha256 = readOptionalOutputHashes(
    binding.output_sha256,
    "20 组逐观测 TX 输出哈希无效",
  );
  assert(
    sameOutputHashes(outputSha256, caseRecord.output_sha256),
    "20 组逐观测 TX 输出哈希与条件组不一致",
  );
  if (caseRecord.status === "completed") {
    assert(
      outputSha256 !== undefined,
      "已完成条件组的逐观测缺少 TX 输出哈希",
    );
  }
  return {
    observation_id: value.observation_id,
    item_id: itemId,
    case_id: caseId,
    mode: "WB-wired",
    status,
    value: numericValue,
    metric: typeof value.metric === "string" ? value.metric : null,
    reason: value.reason,
    family: value.family,
    test_name: value.test_name,
    source_cell: sourceCell,
    evidence_binding: {
      conditions,
      case_id: caseId,
      report_id: bindingReportId,
      batch_id: batchId,
      dataset_id: datasetId,
      input_sha256: inputSha256,
      ...(outputSha256 ? { output_sha256: outputSha256 } : {}),
    },
  };
}

export function readSceneConditionBatch(
  value: unknown,
  expected: {
    datasetId: string;
    algorithmClusterId: string;
  },
): SceneConditionBatch {
  assert(isRecord(value), "20 组批次快照必须为对象");
  assert(
    value.schema_version === SCENE_CONDITION_SCHEMA_VERSION,
    "20 组快照 schema_version 必须为 scene-condition-batch-v1，拒绝旧批次",
  );
  const batchId = readId(value.batch_id, "20 组批次 ID 无效");
  const datasetId = readId(value.dataset_id, "20 组数据集 ID 无效");
  const algorithmClusterId =
    typeof value.algorithm_cluster_id === "string"
      ? value.algorithm_cluster_id
      : "";
  assert(
    datasetId === expected.datasetId,
    "20 组快照数据集与当前测试集不一致",
  );
  assert(
    algorithmClusterId === expected.algorithmClusterId,
    "20 组快照算法簇与当前选择不一致",
  );
  const datasetName = value.dataset_name;
  assert(
    typeof datasetName === "string"
      && datasetName.trim().length > 0,
    "20 组数据集名称为空",
  );
  const datasetManifestSha256 = readSha(
    value.dataset_manifest_sha256,
    "20 组数据集 manifest 哈希无效",
  );
  const datasetAssetsSha256 = readSha(
    value.dataset_assets_sha256,
    "20 组数据集 assets 哈希无效",
  );
  const catalogSha256 = readSha(
    value.catalog_sha256,
    "20 组合同目录哈希无效",
  );
  const cliSha256 = readSha(
    value.cli_sha256,
    "20 组 CLI 哈希无效",
  );
  assert(
    isRecord(value.configuration)
      && isFiniteJson(value.configuration),
    "20 组算法配置不是有限 JSON 对象",
  );
  assert(
    BATCH_STATUSES.has(value.status as SceneConditionBatchStatus),
    "20 组批次状态无效",
  );
  const status = value.status as SceneConditionBatchStatus;
  const createdAt = readTimestamp(
    value.created_at,
    "20 组批次创建时间无效",
  );
  const updatedAt = readTimestamp(
    value.updated_at,
    "20 组批次更新时间无效",
  );
  assert(
    Date.parse(updatedAt) >= Date.parse(createdAt),
    "20 组批次更新时间早于创建时间",
  );
  assert(
    value.test_scope === "background_noise"
      && value.selected_mode === "WB-wired"
      && value.selected_scene === "sce1"
      && value.standard_compliance === false
      && value.formal_test_status === "not_measured",
    "20 组批次范围、模式、场景或合规边界无效",
  );
  assert(
    value.contract_item_count === SCENE_CONDITION_CONTRACT_ITEM_COUNT
      && value.observation_count === SCENE_CONDITION_OBSERVATION_COUNT,
    "20 组批次合同项或观测数无效",
  );
  assert(isRecord(value.progress), "20 组批次进度格式无效");
  assert(
    isCount(value.progress.completed_cases)
      && value.progress.total_cases === SCENE_CONDITION_BATCH_COUNT
      && isCount(value.progress.completed_items)
      && value.progress.total_items === SCENE_CONDITION_OBSERVATION_COUNT,
    "20 组批次进度合同无效",
  );
  assert(isRecord(value.summary), "20 组批次汇总格式无效");
  const summary = value.summary;
  const summaryKeys = [
    "total",
    "computed",
    "pass",
    "fail",
    "review_required",
    "not_measured",
    "invalid",
  ] as const;
  assert(
    summaryKeys.every((key) => isCount(summary[key]))
      && summary.total === SCENE_CONDITION_OBSERVATION_COUNT,
    "20 组批次汇总字段无效",
  );
  assert(
    Array.isArray(value.cases)
      && value.cases.length === SCENE_CONDITION_BATCH_COUNT,
    "20 组批次必须包含 20 个预建条件组",
  );
  assert(
    Array.isArray(value.items)
      && value.items.length === SCENE_CONDITION_OBSERVATION_COUNT,
    "20 组批次必须包含 50 条观测",
  );

  const cases = value.cases.map(readCase);
  const caseMap = new Map<string, SceneConditionCase>();
  for (const caseRecord of cases) {
    assert(
      !caseMap.has(caseRecord.case_id),
      "20 组批次包含重复条件组 ID",
    );
    caseMap.set(caseRecord.case_id, caseRecord);
  }
  const observations = value.items.map((item) =>
    readObservation(item, caseMap, batchId, datasetId)
  );
  const observationIds = new Set<string>();
  const itemCounts = new Map<SceneConditionItemId, number>();
  const observationsByCase = new Map<
    string,
    SceneConditionObservation[]
  >();
  const statusCounts = {
    pass: 0,
    fail: 0,
    review_required: 0,
    not_measured: 0,
    invalid: 0,
  };
  let computed = 0;
  for (const observation of observations) {
    assert(
      !observationIds.has(observation.observation_id),
      "20 组批次包含重复观测 ID",
    );
    observationIds.add(observation.observation_id);
    itemCounts.set(
      observation.item_id,
      (itemCounts.get(observation.item_id) ?? 0) + 1,
    );
    const caseObservations =
      observationsByCase.get(observation.case_id) ?? [];
    caseObservations.push(observation);
    observationsByCase.set(observation.case_id, caseObservations);
    if (
      observation.status === "pass"
      || observation.status === "fail"
      || observation.status === "review_required"
    ) {
      statusCounts[observation.status] += 1;
    } else if (
      observation.status === "not_measured"
      || observation.status === "invalid"
    ) {
      statusCounts[observation.status] += 1;
    }
    if (
      observation.value !== null
      && observation.status !== "not_measured"
      && observation.status !== "invalid"
      && observation.status !== "pending"
      && observation.status !== "running"
    ) {
      computed += 1;
    }
  }
  for (const itemId of SCENE_CONDITION_ITEM_IDS) {
    assert(
      itemCounts.get(itemId) === 10,
      `${itemId} 必须恰好有 10 条观测`,
    );
  }
  assert(
    observationsByCase.size === SCENE_CONDITION_BATCH_COUNT,
    "20 组预建条件组必须全部绑定观测",
  );
  let speechCaseCount = 0;
  let cngCaseCount = 0;
  for (const [caseId, caseObservations] of observationsByCase) {
    const itemIds = new Set<string>(
      caseObservations.map((observation) => observation.item_id),
    );
    const isSpeech =
      itemIds.size === SPEECH_ITEM_IDS.size
      && [...SPEECH_ITEM_IDS].every((itemId) => itemIds.has(itemId));
    const isCng =
      itemIds.size === CNG_ITEM_IDS.size
      && [...CNG_ITEM_IDS].every((itemId) => itemIds.has(itemId));
    assert(
      isSpeech || isCng,
      `20 组条件组 ${caseId} 的合同测项分组无效`,
    );
    if (isSpeech) speechCaseCount += 1;
    else cngCaseCount += 1;
  }
  assert(
    speechCaseCount === 10 && cngCaseCount === 10,
    "20 组必须由 10 个语音条件组与 10 个 CNG 条件组组成",
  );
  const completedCases =
    cases.filter(
      (caseRecord) =>
        caseRecord.status === "completed"
        || caseRecord.status === "failed",
    ).length;
  const completedItems = observations.filter(
    (observation) =>
      observation.status !== "pending"
      && observation.status !== "running",
  ).length;
  assert(
    value.progress.completed_cases === completedCases,
    "20 组批次完成条件组数与预建条件组状态不一致",
  );
  assert(
    value.progress.completed_items === completedItems,
    "20 组批次完成观测数与已完成条件组不一致",
  );
  assert(
    value.summary.computed === computed,
    "20 组批次计算数不等于实际有限数值观测数",
  );
  for (const key of Object.keys(statusCounts) as Array<
    keyof typeof statusCounts
  >) {
    assert(
      value.summary[key] === statusCounts[key],
      `20 组批次 ${key} 汇总与逐观测状态不一致`,
    );
  }
  if (status === "completed") {
    assert(
      cases.every((caseRecord) => caseRecord.status === "completed")
        && value.progress.completed_cases
          === SCENE_CONDITION_BATCH_COUNT
        && value.progress.completed_items
          === SCENE_CONDITION_OBSERVATION_COUNT,
      "20 组批次未完成全部条件组，不能标记 completed",
    );
  }

  return {
    schema_version: SCENE_CONDITION_SCHEMA_VERSION,
    batch_id: batchId,
    dataset_id: datasetId,
    dataset_name: datasetName,
    dataset_manifest_sha256: datasetManifestSha256,
    dataset_assets_sha256: datasetAssetsSha256,
    algorithm_cluster_id: algorithmClusterId,
    configuration: value.configuration,
    catalog_sha256: catalogSha256,
    cli_sha256: cliSha256,
    status,
    created_at: createdAt,
    updated_at: updatedAt,
    test_scope: "background_noise",
    selected_mode: "WB-wired",
    selected_scene: "sce1",
    standard_compliance: false,
    formal_test_status: "not_measured",
    contract_item_count: SCENE_CONDITION_CONTRACT_ITEM_COUNT,
    observation_count: SCENE_CONDITION_OBSERVATION_COUNT,
    progress: {
      completed_cases: value.progress.completed_cases,
      total_cases: SCENE_CONDITION_BATCH_COUNT,
      completed_items: value.progress.completed_items,
      total_items: SCENE_CONDITION_OBSERVATION_COUNT,
    },
    summary: {
      total: SCENE_CONDITION_OBSERVATION_COUNT,
      computed,
      pass: statusCounts.pass,
      fail: statusCounts.fail,
      review_required: statusCounts.review_required,
      not_measured: statusCounts.not_measured,
      invalid: statusCounts.invalid,
    },
    items: observations,
    cases,
    ...(value.error === undefined || value.error === null
      ? {}
      : (() => {
          assert(
            typeof value.error === "string",
            "20 组批次错误字段无效",
          );
          return { error: value.error };
        })()),
  };
}

export function readSceneConditionBatchHistory(
  value: unknown,
  expected: {
    datasetId: string;
    algorithmClusterId: string;
  },
): SceneConditionBatch[] {
  assert(
    Array.isArray(value) && value.length <= 20,
    "20 组批次历史必须是至多 20 条的数组",
  );
  const batches = value.map((entry) =>
    readSceneConditionBatch(entry, expected)
  );
  const ids = new Set<string>();
  for (const batch of batches) {
    assert(!ids.has(batch.batch_id), "20 组批次历史包含重复批次 ID");
    ids.add(batch.batch_id);
  }
  return batches;
}

function txUnavailable(reason: string): SceneConditionCaseTxEvidence {
  return {
    ready: false,
    url: "",
    visual: undefined,
    reason,
  };
}

export function sceneConditionCaseTxEvidence(
  previewCase: ScopedInputPreviewCase,
  batch: SceneConditionBatch | null,
  manifestSha256: string,
): SceneConditionCaseTxEvidence {
  if (!batch) return txUnavailable("尚未执行 20 组测试");
  if (batch.dataset_manifest_sha256 !== manifestSha256) {
    return txUnavailable("批次数据集 manifest 与当前测试集不一致");
  }
  if (
    batch.selected_mode !== previewCase.conditions.mode
    || batch.selected_scene !== previewCase.conditions.scene
  ) {
    return txUnavailable("批次模式或场景与当前条件组不一致");
  }
  const caseRecord = batch.cases.find(
    (candidate) => candidate.case_id === previewCase.case_id,
  );
  if (!caseRecord) return txUnavailable("批次缺少该条件组");
  if (caseRecord.status !== "completed") {
    return txUnavailable(
      caseRecord.status === "failed"
        ? "条件组执行失败，没有 TX"
        : `条件组状态为 ${caseRecord.status}，TX 尚未完成`,
    );
  }
  if (
    caseRecord.report_id === null
    || !ID.test(caseRecord.report_id)
  ) {
    return txUnavailable("已完成条件组缺少有效报告 ID");
  }
  if (!sameConditions(caseRecord.conditions, previewCase.conditions)) {
    return txUnavailable("批次条件组条件与当前待测条件不一致");
  }
  if (
    !sameInputHashes(
      caseRecord.signals_sha256,
      previewCase.signals_sha256 as SceneConditionInputHashes,
    )
  ) {
    return txUnavailable("批次 manifest 四路信号哈希与当前待测输入不一致");
  }
  if (
    !sameInputHashes(
      caseRecord.input_sha256,
      caseRecord.signals_sha256,
    )
  ) {
    return txUnavailable("条件组执行输入哈希与 manifest 信号哈希不一致");
  }
  const outputSha256 = caseRecord.output_sha256;
  if (!outputSha256) return txUnavailable("已完成条件组缺少 TX 输出哈希");
  const visual = caseRecord.tx_visual;
  if (!visual) return txUnavailable("已完成条件组缺少 TX 波形与梅尔频谱");
  if (visual.source_sha256 !== outputSha256.tx) {
    return txUnavailable("TX 可视化来源哈希与输出哈希不一致");
  }
  const caseObservations = batch.items.filter(
    (item) => item.case_id === previewCase.case_id,
  );
  const observedItemIds = new Set<string>(
    caseObservations.map((item) => item.item_id),
  );
  const expectedItemIds =
    [...observedItemIds].some((itemId) => CNG_ITEM_IDS.has(itemId))
      ? CNG_ITEM_IDS
      : SPEECH_ITEM_IDS;
  if (
    caseObservations.length !== expectedItemIds.size
    || [...expectedItemIds].some((itemId) => !observedItemIds.has(itemId))
  ) {
    return txUnavailable("批次缺少该条件组完整的逐观测证据");
  }
  for (const item of caseObservations) {
    const binding = item.evidence_binding;
    if (
      binding.case_id !== caseRecord.case_id
      || binding.batch_id !== batch.batch_id
      || binding.dataset_id !== batch.dataset_id
      || binding.report_id !== caseRecord.report_id
      || !sameInputHashes(
        binding.input_sha256,
        caseRecord.signals_sha256,
      )
      || !sameOutputHashes(binding.output_sha256, outputSha256)
    ) {
      return txUnavailable("逐观测证据未绑定同一条件组、报告与 TX 输出");
    }
  }
  return {
    ready: true,
    url: `${SCENE_CONDITION_BATCHES_URL}/${encodeURIComponent(
      batch.batch_id,
    )}/cases/${encodeURIComponent(
      previewCase.case_id,
    )}/outputs/tx?v=${encodeURIComponent(outputSha256.tx)}`,
    visual,
    reason: "",
  };
}
