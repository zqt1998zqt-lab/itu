import type { FiveSignalMode } from "./five-signal-flow-model.ts";
import {
  SCOPED_INPUT_SIGNAL_ROLES,
  SCOPED_STATUS_LABELS,
  type BackgroundNoiseScene,
  type ScopedInputPreviewCase,
  type ScopedPreviewVisualTrack,
  type ScopedSnapshot,
  readSingleCaseSelection,
} from "./scoped-acoustic-model.ts";
import {
  CONVERSATION_MODES,
  conversationModeOf,
  sceneConditionCounts,
  sceneConditionProfileForDatasetId,
  type ConversationMode,
  type SceneConditionItemId,
  type SceneConditionProfile,
} from "./scene-condition-profiles.ts";

export type { SceneConditionItemId } from "./scene-condition-profiles.ts";

export const SCENE_CONDITION_BATCHES_URL =
  "/api/alignment/acceptance/scene-condition-batches";
export const SCENE_CONDITION_DATASETS_URL =
  "/api/alignment/acceptance/scene-condition-datasets";
export const SCENE_CONDITION_SCHEMA_VERSION =
  "scene-condition-batch-v1" as const;
export const SCENE_CONDITION_BATCH_COUNT = 20 as const;
export const SCENE_CONDITION_OBSERVATION_COUNT = 50 as const;
export const SCENE_CONDITION_CONTRACT_ITEM_COUNT = 5 as const;
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
  rx?: string;
};

export type SceneConditionDynamicMicrophone = {
  source: "post_rx_runtime_mix";
  path: string;
  sha256: string;
  sample_rate: number;
  channels: number;
  frames: number;
  capture_gain: number;
  playback_sha256: string;
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
  provenance?: Record<string, unknown>;
  conversation_diagnostics?: Record<string, unknown>;
  signals_sha256: SceneConditionInputHashes;
  input_sha256: SceneConditionInputHashes;
  output_sha256?: SceneConditionOutputHashes;
  dynamic_microphone?: SceneConditionDynamicMicrophone;
  dynamic_microphone_visual?: ScopedPreviewVisualTrack;
  effective_input_sha256?: SceneConditionInputHashes;
  execution?: Record<string, unknown>;
  tx_visual?: ScopedPreviewVisualTrack;
  rx_visual?: ScopedPreviewVisualTrack;
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
  selected_mode: FiveSignalMode;
  selected_scene: BackgroundNoiseScene;
  formal_test_status: "not_measured";
  contract_item_count: typeof SCENE_CONDITION_CONTRACT_ITEM_COUNT;
  case_count: number;
  preview_condition_group_count: number;
  observation_count: number;
  speech_case_count: number;
  cng_case_count: number;
  conversation_counts: Record<string, number>;
  items: SceneConditionObservation[];
  cases: SceneConditionCase[];
};

export type SceneConditionCaseTxEvidence = {
  ready: boolean;
  url: string;
  visual: ScopedPreviewVisualTrack | undefined;
  reason: string;
};

export type SceneConditionCaseRxEvidence = SceneConditionCaseTxEvidence;

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
const FAMILY_BY_POSITION = [
  "g_mos",
  "n_mos",
  "s_mos",
  "cng_level",
  "cng_spectrum",
] as const;
const SPEECH_FAMILIES = new Set<string>([
  "g_mos",
  "n_mos",
  "s_mos",
]);
const CNG_FAMILIES = new Set<string>([
  "cng_level",
  "cng_spectrum",
]);
const CONVERSATION_DIAGNOSTICS_SCHEMA =
  "scene-condition-conversation-diagnostics-v1";
const CONVERSATION_DIAGNOSTIC_STATUSES = new Set<string>([
  "review_required",
  "not_measurable",
  "not_applicable",
]);
const CONVERSATION_SEGMENT_MODES = new Set<string>([
  "near",
  "far",
  "double_talk",
  "noise",
  "not_applicable",
]);
const CONVERSATION_METRIC_KEYS = [
  "near_gain_db",
  "near_correlation",
  "echo_reduction_db",
  "tx_level_dbfs",
] as const;
const CONVERSATION_SEGMENT_MODE_BY_LABEL: Record<string, string> = {
  speech: "near",
  far_speech: "far",
  double_talk: "double_talk",
  quiet: "not_applicable",
  far_only: "far",
  near_only: "near",
  noise_only: "noise",
  cng_active: "not_applicable",
  cng_settling: "not_applicable",
  cng_noise_observation: "noise",
};
const CONVERSATION_SEGMENT_LABELS_BY_MODE: Record<
  ConversationMode,
  readonly string[]
> = {
  near_single: ["speech"],
  far_single: ["far_speech"],
  double_talk: ["double_talk"],
  transition: [
    "quiet",
    "far_only",
    "near_only",
    "far_only",
    "double_talk",
    "noise_only",
  ],
  noise_only: [
    "cng_active",
    "cng_settling",
    "cng_noise_observation",
  ],
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

function readConversationDiagnostics(
  value: unknown,
  expected: {
    caseId: string;
    txSha256?: string;
    provenance: Record<string, unknown>;
    require: boolean;
    sampleRateHz?: number;
  },
): Record<string, unknown> | undefined {
  if (value === undefined || value === null) {
    assert(
      !expected.require,
      "v2 完成 case 缺少 conversation_diagnostics",
    );
    return undefined;
  }
  assert(isRecord(value), "conversation_diagnostics 必须为对象");
  const mode = conversationModeOf({ provenance: expected.provenance });
  assert(mode !== null, "conversation_diagnostics 缺少有效 provenance 模式");
  const renderMetadata = expected.provenance.render_metadata;
  assert(
    isRecord(renderMetadata)
      && typeof renderMetadata.duration_s === "number"
      && Number.isFinite(renderMetadata.duration_s)
      && Math.abs(renderMetadata.duration_s - 12) <= 1e-6
      && renderMetadata.sample_rate_hz === (expected.sampleRateHz ?? 16_000)
      && Array.isArray(renderMetadata.segments),
    "conversation_diagnostics 缺少有效 provenance 活动段",
  );
  const expectedLabels = CONVERSATION_SEGMENT_LABELS_BY_MODE[mode];
  const renderSegments: Array<{
    label: string;
    start_s: number;
    end_s: number;
  }> = [];
  let cursor = 0;
  assert(
    renderMetadata.segments.length === expectedLabels.length,
    "conversation_diagnostics provenance 活动段数量无效",
  );
  for (const [index, segment] of renderMetadata.segments.entries()) {
    assert(
      isRecord(segment)
        && segment.label === expectedLabels[index]
        && typeof segment.start_s === "number"
        && Number.isFinite(segment.start_s)
        && typeof segment.end_s === "number"
        && Number.isFinite(segment.end_s)
        && Math.abs(segment.start_s - cursor) <= 1e-9
        && segment.end_s > segment.start_s
        && segment.end_s <= 12 + 1e-9,
      "conversation_diagnostics provenance 活动段标签或时间无效",
    );
    renderSegments.push({
      label: segment.label as string,
      start_s: segment.start_s,
      end_s: segment.end_s,
    });
    cursor = segment.end_s;
  }
  assert(
    Math.abs(cursor - 12) <= 1e-9,
    "conversation_diagnostics provenance 活动段未覆盖 12 秒",
  );
  assert(
    value.schema_version === CONVERSATION_DIAGNOSTICS_SCHEMA,
    "conversation_diagnostics schema_version 无效",
  );
  assert(
    value.case_id === expected.caseId,
    "conversation_diagnostics case_id 与 case 不一致",
  );
  assert(
    value.conversation_mode === mode,
    "conversation_diagnostics conversation_mode 与 provenance 不一致",
  );
  assert(
    value.evidence_grade === "engineering_proxy"
      && value.formal_compliance === false
      && value.standard_compliance === false
      && value.formal_test_status === "not_measured",
    "conversation_diagnostics 工程代理或正式边界无效",
  );
  assert(
    CONVERSATION_DIAGNOSTIC_STATUSES.has(String(value.status)),
    "conversation_diagnostics 顶层状态无效",
  );
  assert(
    typeof value.reason === "string" && value.reason.trim().length > 0,
    "conversation_diagnostics reason 无效",
  );
  assert(
    typeof value.review_required === "boolean"
      && value.review_required === (value.status === "review_required"),
    "conversation_diagnostics review_required 与状态不一致",
  );
  assert(
    typeof value.tx_sha256 === "string"
      && SHA256.test(value.tx_sha256)
      && (expected.txSha256 === undefined
        || value.tx_sha256 === expected.txSha256),
    "conversation_diagnostics tx_sha256 与输出不一致",
  );
  assert(
    typeof value.sample_rate_hz === "number"
      && Number.isInteger(value.sample_rate_hz)
      && value.sample_rate_hz === (expected.sampleRateHz ?? 16_000)
      && typeof value.source_duration_s === "number"
      && Number.isFinite(value.source_duration_s)
      && Math.abs(value.source_duration_s - 12) <= 1e-6
      && typeof value.tx_duration_s === "number"
      && Number.isFinite(value.tx_duration_s)
      && value.tx_duration_s >= 0,
    "conversation_diagnostics 时长或采样率无效",
  );
  assert(
    Array.isArray(value.segments)
      && value.segments.length === renderSegments.length,
    "conversation_diagnostics segments 数量与 provenance 不一致",
  );
  const segmentStatuses: string[] = [];
  for (const [index, segment] of value.segments.entries()) {
    assert(
      isRecord(segment)
        && segment.label === renderSegments[index].label
        && segment.start_s === renderSegments[index].start_s
        && segment.end_s === renderSegments[index].end_s
        && typeof segment.mode === "string"
        && CONVERSATION_SEGMENT_MODES.has(segment.mode)
        && segment.mode === CONVERSATION_SEGMENT_MODE_BY_LABEL[String(segment.label)]
        && CONVERSATION_DIAGNOSTIC_STATUSES.has(String(segment.status))
        && typeof segment.reason === "string"
        && segment.reason.trim().length > 0
        && isRecord(segment.metrics),
      "conversation_diagnostics 活动段字段、模式或时间无效",
    );
    const metricKeys = Object.keys(segment.metrics).sort();
    const expectedMetricKeys = [...CONVERSATION_METRIC_KEYS].sort();
    assert(
      metricKeys.length === expectedMetricKeys.length
        && metricKeys.every(
          (key, keyIndex) => key === expectedMetricKeys[keyIndex],
        )
        && Object.values(segment.metrics).every(
          (metric) =>
            metric === null
            || typeof metric === "number" && Number.isFinite(metric),
        ),
      "conversation_diagnostics metrics 不是有限数或 null",
    );
    segmentStatuses.push(String(segment.status));
  }
  const aggregateStatus = segmentStatuses.includes("review_required")
    ? "review_required"
    : segmentStatuses.includes("not_measurable")
      ? "not_measurable"
      : "not_applicable";
  assert(
    value.status === aggregateStatus,
    "conversation_diagnostics 顶层状态与活动段不一致",
  );
  return value;
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

function readConditions(
  value: unknown,
  scene: BackgroundNoiseScene,
  mode: FiveSignalMode = "WB-wired",
): SceneConditionConditions {
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
    value.mode === mode
      && value.direction === "send"
      && value.scene === scene
      && value.volume === "nominal"
      && value.window_ms === null
      && value.input_level_db === 0,
    `20 组批次条件组超出 ${mode} / ${scene} 合同`,
  );
  return {
    mode,
    direction: "send",
    scene,
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
  const keys = isRecord(value) ? Object.keys(value).sort() : [];
  assert(
    isRecord(value)
      && (keys.length === 1 || keys.length === 2)
      && Object.hasOwn(value, "tx"),
    message,
  );
  if (keys.length === 2) {
    assert(
      keys[0] === "rx" && keys[1] === "tx",
      message,
    );
  }
  return {
    tx: readSha(value.tx, message),
    ...(Object.hasOwn(value, "rx")
      ? { rx: readSha(value.rx, message) }
      : {}),
  };
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

function readDynamicMicrophone(
  value: unknown,
): SceneConditionDynamicMicrophone {
  assert(isRecord(value), "20 组动态麦克风格式无效");
  assert(
    value.source === "post_rx_runtime_mix"
      && typeof value.path === "string"
      && value.path.length > 0
      && isCount(value.sample_rate)
      && isCount(value.channels)
      && value.channels > 0
      && isCount(value.frames)
      && value.frames > 0
      && typeof value.capture_gain === "number"
      && Number.isFinite(value.capture_gain)
      && value.capture_gain > 0
      && value.capture_gain <= 1,
    "20 组动态麦克风来源或参数无效",
  );
  return {
    source: "post_rx_runtime_mix",
    path: value.path,
    sha256: readSha(value.sha256, "20 组动态麦克风哈希无效"),
    sample_rate: value.sample_rate,
    channels: value.channels,
    frames: value.frames,
    capture_gain: value.capture_gain,
    playback_sha256: readSha(
      value.playback_sha256,
      "20 组动态麦克风 post-RX playback 哈希无效",
    ),
  };
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
  return left?.tx === right?.tx && left?.rx === right?.rx;
}

function readCase(
  value: unknown,
  profile: SceneConditionProfile,
): SceneConditionCase {
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
  const conditions = readConditions(value.conditions, profile.scene, profile.mode ?? "WB-wired");
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
  const rxVisual =
    value.rx_visual === undefined || value.rx_visual === null
      ? undefined
      : readVisualTrack(value.rx_visual);
  const dynamicMicrophone =
    value.dynamic_microphone === undefined
      || value.dynamic_microphone === null
      ? undefined
      : readDynamicMicrophone(value.dynamic_microphone);
  const dynamicMicrophoneVisual =
    value.dynamic_microphone_visual === undefined
      || value.dynamic_microphone_visual === null
      ? undefined
      : readVisualTrack(value.dynamic_microphone_visual);
  const effectiveInputSha256 =
    value.effective_input_sha256 === undefined
      || value.effective_input_sha256 === null
      ? undefined
      : readInputHashes(
          value.effective_input_sha256,
          "20 组运行时四路输入哈希无效",
        );
  const provenance =
    value.provenance === undefined
      ? undefined
      : (() => {
          assert(
            isRecord(value.provenance)
              && isFiniteJson(value.provenance),
            "场景条件组 provenance 无效",
          );
          return value.provenance;
        })();
  if (profile.conversationCounts !== undefined) {
    assert(
      provenance !== undefined
        && conversationModeOf({ provenance }) !== null
        && provenance.engineering_preview === true
        && provenance.actual_vehicle_recording === false,
      "100 组条件组 provenance.conversation_mode 无效",
    );
  } else if (
    provenance?.conversation_mode !== undefined
    && conversationModeOf({ provenance }) === null
  ) {
    throw Error("20 组条件组 provenance.conversation_mode 无效");
  }
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
    if (outputSha256.rx !== undefined) {
      assert(
        rxVisual !== undefined,
        "已完成条件组缺少 RX 处理后波形与梅尔频谱",
      );
    }
  }
  if (txVisual !== undefined && outputSha256 !== undefined) {
    assert(
      txVisual.source_sha256 === outputSha256.tx,
      "20 组 TX 可视化来源哈希与输出哈希不一致",
    );
  }
  if (rxVisual !== undefined && outputSha256 !== undefined) {
    assert(
      outputSha256.rx !== undefined
        && rxVisual.source_sha256 === outputSha256.rx,
      "20 组 RX 处理后可视化来源哈希与输出哈希不一致",
    );
  }
  if (outputSha256?.rx !== undefined) {
    const execution = value.execution;
    const echoReference = isRecord(execution)
      ? execution.echo_reference
      : undefined;
    assert(
      isRecord(echoReference)
        && echoReference.source === "vcp_rx_processed"
        && echoReference.processed_output === "rx"
        && echoReference.processed_sha256 === outputSha256.rx
        && echoReference.tx_echo_argument_sha256 === outputSha256.rx,
      "20 组 TX 未绑定 RX 处理后回声参考",
    );
  }
  if (dynamicMicrophone !== undefined) {
    assert(
      dynamicMicrophoneVisual !== undefined
        && dynamicMicrophoneVisual.source_sha256
          === dynamicMicrophone.sha256,
      "20 组动态麦克风缺少绑定的波形与梅尔频谱",
    );
    assert(
      effectiveInputSha256 !== undefined
        && effectiveInputSha256.microphone
          === dynamicMicrophone.sha256
        && outputSha256?.rx !== undefined
        && effectiveInputSha256.far_end === outputSha256.rx,
      "20 组动态麦克风未绑定 post-RX 输入身份",
    );
  } else {
    assert(
      dynamicMicrophoneVisual === undefined
        && effectiveInputSha256 === undefined,
      "20 组动态麦克风证据缺少主字段",
    );
  }
  let conversationDiagnostics: Record<string, unknown> | undefined;
  if (profile.conversationCounts !== undefined) {
    assert(provenance !== undefined, "100 组条件组缺少 provenance");
    conversationDiagnostics = readConversationDiagnostics(
      value.conversation_diagnostics,
      {
        caseId: value.case_id,
        txSha256: outputSha256?.tx,
        provenance,
        require: status === "completed",
        sampleRateHz: profile.sampleRateHz ?? 16_000,
      },
    );
  } else if (value.conversation_diagnostics !== undefined) {
    assert(
      isRecord(value.conversation_diagnostics)
        && isFiniteJson(value.conversation_diagnostics),
      "场景条件组通话诊断字段无效",
    );
    conversationDiagnostics = value.conversation_diagnostics;
  }
  return {
    case_id: value.case_id,
    status,
    report_id: reportId,
    conditions,
    ...(provenance ? { provenance } : {}),
    ...(conversationDiagnostics
      ? { conversation_diagnostics: conversationDiagnostics }
      : {}),
    signals_sha256: signalsSha256,
    input_sha256: inputSha256,
    ...(outputSha256 ? { output_sha256: outputSha256 } : {}),
    ...(dynamicMicrophone ? { dynamic_microphone: dynamicMicrophone } : {}),
    ...(dynamicMicrophoneVisual
      ? { dynamic_microphone_visual: dynamicMicrophoneVisual }
      : {}),
    ...(effectiveInputSha256
      ? { effective_input_sha256: effectiveInputSha256 }
      : {}),
    ...(isRecord(value.execution)
      ? { execution: value.execution }
      : {}),
    ...(txVisual ? { tx_visual: txVisual } : {}),
    ...(rxVisual ? { rx_visual: rxVisual } : {}),
    ...(typeof value.error === "string" ? { error: value.error } : {}),
  };
}

function readObservation(
  value: unknown,
  cases: Map<string, SceneConditionCase>,
  batchId: string,
  datasetId: string,
  profile: SceneConditionProfile,
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
      && profile.contractItemIds.includes(
        value.item_id as SceneConditionItemId,
      ),
    `20 组逐观测合同测项超出 ${profile.sceneLabel ?? profile.scene} 合同`,
  );
  const expectedMode = profile.mode ?? "WB-wired";
  const itemId = value.item_id as SceneConditionItemId;
  const sourceCell = itemId.slice(`${expectedMode}:`.length);
  assert(
    typeof value.observation_id === "string"
      && value.observation_id === `${caseId}:${itemId}`,
    "20 组观测 ID 与条件组/合同测项不一致",
  );
  assert(
    value.mode === expectedMode,
    `20 组逐观测模式不是 ${expectedMode}`,
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
  const itemPosition = profile.contractItemIds.indexOf(itemId);
  assert(
    FAMILY_BY_POSITION[itemPosition] === value.family
      && typeof value.source_cell === "string"
      && value.source_cell === sourceCell,
    "20 组逐观测合同族或源行不一致",
  );
  const conversationMode = conversationModeOf(caseRecord);
  if (
    profile.conversationCounts !== undefined
    && itemPosition < 3
    && (
      conversationMode === "far_single"
      || conversationMode === "transition"
    )
  ) {
    assert(
      status === "not_measured"
        || status === "pending"
        || status === "running"
        || status === "invalid",
      `${conversationMode} 的 GNS 合同行不能工程通过`,
    );
  }
  assert(isRecord(value.evidence_binding), "20 组逐观测证据绑定无效");
  const binding = value.evidence_binding;
  const conditions = readConditions({
    mode: binding.mode,
    direction: binding.direction,
    scene: binding.scene,
    volume: binding.volume,
    window_ms: binding.window_ms,
    input_level_db: binding.input_level_db,
  }, profile.scene, expectedMode);
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
    mode: expectedMode,
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
    caseId?: string;
  },
): SceneConditionBatch {
  assert(isRecord(value), "20 组批次快照必须为对象");
  assert(
    value.schema_version === SCENE_CONDITION_SCHEMA_VERSION,
    "20 组快照 schema_version 必须为 scene-condition-batch-v1，拒绝旧批次",
  );
  const batchId = readId(value.batch_id, "20 组批次 ID 无效");
  const datasetId = readId(value.dataset_id, "20 组数据集 ID 无效");
  const profile = sceneConditionProfileForDatasetId(datasetId);
  assert(profile !== null, "20 组快照绑定了未注册的场景条件数据集");
  const expectedCounts = sceneConditionCounts(profile);
  const selection = readSingleCaseSelection(value, {
    caseId: expected.caseId,
    label: "场景条件批次",
  });
  const expectedCaseCount = selection === null
    ? expectedCounts.cases
    : 1;
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
      && datasetName === profile.datasetName,
    "20 组数据集名称与画像不一致",
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
  const expectedBatchMode = profile.mode ?? "WB-wired";
  assert(
    value.test_scope === "background_noise"
      && value.selected_mode === expectedBatchMode
      && value.selected_scene === profile.scene
      && value.standard_compliance === false
      && value.formal_test_status === "not_measured",
    "20 组批次范围、模式、场景或合规边界无效",
  );
  assert(
    value.contract_item_count === SCENE_CONDITION_CONTRACT_ITEM_COUNT
      && isCount(value.observation_count)
      && Number(value.observation_count) > 0
      && value.case_count === expectedCaseCount
      && value.preview_condition_group_count === expectedCounts.cases
      && (selection === null
        ? value.observation_count === expectedCounts.observations
        : selection.dataset_case_count === expectedCounts.cases
          && selection.dataset_observation_count
            === expectedCounts.observations),
    "场景条件批次合同项或观测数无效",
  );
  assert(isRecord(value.progress), "20 组批次进度格式无效");
  assert(
    isCount(value.progress.completed_cases)
      && value.progress.total_cases === expectedCaseCount
      && isCount(value.progress.completed_items)
      && value.progress.total_items === value.observation_count,
    "场景条件批次进度合同无效",
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
      && summary.total === value.observation_count,
    "场景条件批次汇总字段无效",
  );
  assert(
    Array.isArray(value.cases)
      && value.cases.length === expectedCaseCount,
    "场景条件批次条件组数量与画像不一致",
  );
  assert(
    Array.isArray(value.items)
      && value.items.length === value.observation_count,
    "场景条件批次观测数量与画像不一致",
  );
  assert(
    isCount(value.speech_case_count)
      && isCount(value.cng_case_count)
      && isRecord(value.conversation_counts),
    "场景条件批次语音、CNG 或通话模式计数无效",
  );
  const conversationCounts = value.conversation_counts;

  const cases = value.cases.map((entry) => readCase(entry, profile));
  const caseMap = new Map<string, SceneConditionCase>();
  for (const caseRecord of cases) {
    assert(
      !caseMap.has(caseRecord.case_id),
      "20 组批次包含重复条件组 ID",
    );
    caseMap.set(caseRecord.case_id, caseRecord);
  }
  if (selection !== null) {
    assert(
      cases.length === 1
        && cases[0].case_id === selection.selected_case_id,
      "单案场景批次 case 与 selected_case_id 不一致",
    );
  }
  const observations = value.items.map((item) =>
    readObservation(item, caseMap, batchId, datasetId, profile)
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
  assert(
    observationsByCase.size === expectedCaseCount,
    "场景条件批次预建条件组必须全部绑定观测",
  );
  let speechCaseCount = 0;
  let cngCaseCount = 0;
  const conversationModeCounts = new Map<ConversationMode, number>();
  const speechItemIds = new Set<string>(
    profile.contractItemIds.slice(0, 3),
  );
  const cngItemIds = new Set<string>(
    profile.contractItemIds.slice(3),
  );
  for (const [caseId, caseObservations] of observationsByCase) {
    const itemIds = new Set<string>(
      caseObservations.map((observation) => observation.item_id),
    );
    const caseRecord = caseMap.get(caseId);
    assert(caseRecord !== undefined, `场景条件组 ${caseId} 不存在`);
    const conversationMode = conversationModeOf(caseRecord);
    if (profile.conversationCounts !== undefined) {
      assert(
        conversationMode !== null,
        `100 组条件组 ${caseId} 缺少通话模式`,
      );
      conversationModeCounts.set(
        conversationMode,
        (conversationModeCounts.get(conversationMode) ?? 0) + 1,
      );
      const expectedItemIds = conversationMode === "noise_only"
        ? cngItemIds
        : speechItemIds;
      const expectedFamilies = conversationMode === "noise_only"
        ? CNG_FAMILIES
        : SPEECH_FAMILIES;
      assert(
        itemIds.size === expectedItemIds.size
          && [...expectedItemIds].every((itemId) => itemIds.has(itemId))
          && caseObservations.every(
            (observation) => expectedFamilies.has(observation.family),
          ),
        `100 组条件组 ${caseId} 的通话模式与合同族分组无效`,
      );
      if (conversationMode === "noise_only") cngCaseCount += 1;
      else speechCaseCount += 1;
    } else {
      const isSpeech =
        itemIds.size === speechItemIds.size
        && [...speechItemIds].every((itemId) => itemIds.has(itemId));
      const isCng =
        itemIds.size === cngItemIds.size
        && [...cngItemIds].every((itemId) => itemIds.has(itemId));
      assert(
        isSpeech || isCng,
        `20 组条件组 ${caseId} 的合同测项分组无效`,
      );
      if (isSpeech) speechCaseCount += 1;
      else cngCaseCount += 1;
    }
  }
  assert(
    speechCaseCount + cngCaseCount === expectedCaseCount,
    "场景条件批次语音/CNG 条件组数量与批次范围不一致",
  );
  const selectedKind = selection === null
    ? null
    : speechCaseCount === 1
      ? "speech"
      : "cng";
  for (const [index, itemId] of profile.contractItemIds.entries()) {
    const expectedPerItem = selection === null
      ? index < 3
        ? expectedCounts.speech
        : expectedCounts.cng
      : selectedKind === "speech" && index < 3
        || selectedKind === "cng" && index >= 3
        ? 1
        : 0;
    assert(
      (itemCounts.get(itemId) ?? 0) === expectedPerItem,
      `${itemId} 必须恰好有 ${expectedPerItem} 条观测`,
    );
  }
  assert(
    value.speech_case_count === speechCaseCount
      && value.cng_case_count === cngCaseCount,
    "场景条件批次语音/CNG 数量与逐 case 结果不一致",
  );
  if (profile.conversationCounts !== undefined) {
    for (const conversationMode of CONVERSATION_MODES) {
      const expectedModeCount = selection === null
        ? profile.conversationCounts[conversationMode]
        : conversationModeCounts.get(conversationMode) ?? 0;
      assert(
        (conversationModeCounts.get(conversationMode) ?? 0)
          === expectedModeCount,
        `100 组 ${conversationMode} 条件组数量与画像不一致`,
      );
    }
    if (selection !== null) {
      const selectedCase = caseMap.get(selection.selected_case_id);
      assert(selectedCase !== undefined, "单案场景批次缺少所选 case");
      const selectedConversationMode = conversationModeOf(selectedCase);
      assert(
        selectedConversationMode !== null,
        "单案场景批次所选 case 缺少通话模式",
      );
      const keys = Object.keys(conversationCounts).sort();
      const expectedKeys = [...CONVERSATION_MODES].sort();
      assert(
        keys.length === expectedKeys.length
          && keys.every((key, index) => key === expectedKeys[index])
          && CONVERSATION_MODES.every(
            (conversationMode) =>
              conversationCounts[conversationMode]
                === (conversationMode === selectedConversationMode ? 1 : 0),
          ),
        "单案场景批次 conversation_counts 无效",
      );
    } else {
      assert(
        CONVERSATION_MODES.every(
          (conversationMode) =>
            conversationCounts[conversationMode]
              === profile.conversationCounts![conversationMode],
        )
          && Object.keys(conversationCounts).length
            === CONVERSATION_MODES.length,
        "场景条件批次 conversation_counts 与画像不一致",
      );
    }
  } else {
    assert(
      Object.keys(conversationCounts).length === 0,
      "v1 场景条件批次 conversation_counts 必须为空对象",
    );
  }
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
          === expectedCaseCount
        && value.progress.completed_items
          === value.observation_count
        && observations.every(
          (observation) =>
            observation.status !== "pending"
            && observation.status !== "running",
        ),
      "场景条件批次未完成全部条件组，不能标记 completed",
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
    selected_mode: expectedBatchMode,
    selected_scene: profile.scene,
    standard_compliance: false,
    formal_test_status: "not_measured",
    contract_item_count: SCENE_CONDITION_CONTRACT_ITEM_COUNT,
    ...(selection ?? {}),
    case_count: expectedCaseCount,
    preview_condition_group_count: expectedCounts.cases,
    observation_count: observations.length,
    speech_case_count: speechCaseCount,
    cng_case_count: cngCaseCount,
    conversation_counts: profile.conversationCounts === undefined
      ? {}
      : Object.fromEntries(
          CONVERSATION_MODES.map((conversationMode) => [
            conversationMode,
            conversationModeCounts.get(conversationMode) ?? 0,
          ]),
        ),
    progress: {
      completed_cases: value.progress.completed_cases,
      total_cases: expectedCaseCount,
      completed_items: value.progress.completed_items,
      total_items: observations.length,
    },
    summary: {
      total: observations.length,
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
    latestPerCase?: boolean;
  },
): SceneConditionBatch[] {
  const latestPerCase = expected.latestPerCase ?? false;
  assert(
    typeof latestPerCase === "boolean",
    "latestPerCase 必须为布尔值",
  );
  let maxBatches = 20;
  if (latestPerCase) {
    const profile = sceneConditionProfileForDatasetId(expected.datasetId);
    assert(
      profile !== null,
      "latestPerCase 历史绑定了未注册的场景条件数据集",
    );
    maxBatches = sceneConditionCounts(profile).cases;
  }
  assert(
    Array.isArray(value) && value.length <= maxBatches,
    latestPerCase
      ? `latestPerCase 历史必须至多 ${maxBatches} 条`
      : "20 组批次历史必须是至多 20 条的数组",
  );
  const batches = value.map((entry) =>
    readSceneConditionBatch(entry, expected)
  );
  const ids = new Set<string>();
  for (const batch of batches) {
    assert(!ids.has(batch.batch_id), "20 组批次历史包含重复批次 ID");
    ids.add(batch.batch_id);
  }
  if (latestPerCase) {
    const covered = new Set<string>();
    for (const batch of batches) {
      const newlyCovered = batch.cases.filter(
        (caseRecord) => !covered.has(caseRecord.case_id),
      );
      assert(
        newlyCovered.length > 0,
        "latestPerCase 历史必须按服务端顺序逐行覆盖新的条件组",
      );
      for (const caseRecord of newlyCovered) {
        covered.add(caseRecord.case_id);
      }
      assert(
        covered.size <= maxBatches,
        "latestPerCase 历史覆盖的条件组数超过数据集画像",
      );
    }
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
  const batchProfile = sceneConditionProfileForDatasetId(batch.dataset_id);
  if (!batchProfile || batchProfile.scene !== batch.selected_scene) {
    return txUnavailable("批次数据集与场景合同身份不一致");
  }
  if (batchProfile.conversationCounts !== undefined) {
    if (caseRecord.provenance === undefined) {
      return txUnavailable("条件组缺少 provenance，拒绝展示诊断");
    }
    try {
      readConversationDiagnostics(
        caseRecord.conversation_diagnostics,
        {
          caseId: caseRecord.case_id,
          txSha256: outputSha256.tx,
          provenance: caseRecord.provenance,
          require: true,
          sampleRateHz: batchProfile.sampleRateHz ?? 16_000,
        },
      );
    } catch {
      return txUnavailable("条件组诊断证据身份或边界无效");
    }
  }
  const observedItemIds = new Set<string>(
    caseObservations.map((item) => item.item_id),
  );
  const cngItemIds = new Set<string>(
    batchProfile.contractItemIds.slice(3),
  );
  const speechItemIds = new Set<string>(
    batchProfile.contractItemIds.slice(0, 3),
  );
  const caseConversationMode = conversationModeOf(caseRecord);
  const previewConversationMode = conversationModeOf(previewCase);
  if (
    batchProfile.conversationCounts !== undefined
    && (
      caseConversationMode === null
      || caseConversationMode !== previewConversationMode
    )
  ) {
    return txUnavailable("条件组通话模式与当前待测样本不一致");
  }
  const expectedItemIds = caseConversationMode === "noise_only"
    ? cngItemIds
    : caseConversationMode === null
      && [...observedItemIds].some((itemId) => cngItemIds.has(itemId))
      ? cngItemIds
      : speechItemIds;
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

export function sceneConditionCaseRxEvidence(
  previewCase: ScopedInputPreviewCase,
  batch: SceneConditionBatch | null,
  manifestSha256: string,
): SceneConditionCaseRxEvidence {
  const txEvidence = sceneConditionCaseTxEvidence(
    previewCase,
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
  const caseRecord = batch.cases.find(
    (candidate) => candidate.case_id === previewCase.case_id,
  );
  const rxSha256 = caseRecord?.output_sha256?.rx;
  if (!rxSha256) {
    return {
      ready: false,
      url: "",
      visual: undefined,
      reason: "该历史批次未记录 VCP RX 处理后回声参考",
    };
  }
  const visual = caseRecord?.rx_visual;
  if (!visual || visual.source_sha256 !== rxSha256) {
    return {
      ready: false,
      url: "",
      visual: undefined,
      reason: "RX 处理后波形与输出哈希不一致",
    };
  }
  return {
    ready: true,
    url: `${SCENE_CONDITION_BATCHES_URL}/${encodeURIComponent(
      batch.batch_id,
    )}/cases/${encodeURIComponent(
      previewCase.case_id,
    )}/outputs/rx?v=${encodeURIComponent(rxSha256)}`,
    visual,
    reason: "",
  };
}
