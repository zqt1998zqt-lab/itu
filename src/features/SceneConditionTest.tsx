import {
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import {
  AlertCircle,
  CheckCircle2,
  Clock3,
  Download,
  History,
  ListChecks,
  MinusCircle,
  Play,
  RefreshCw,
  ShieldAlert,
  XCircle,
} from "lucide-react";
import OutputAudioPreview from "./OutputAudioPreview.tsx";
import NsQualityMetrics from "./NsQualityMetrics.tsx";
import {
  algorithmClusterById,
  algorithmClusterGroupsForMode,
  algorithmClusterLabel,
  algorithmClustersForMode,
  isAlgorithmClusterSupportedForMode,
  sampleRateForBusinessMode,
} from "./rate-algorithm-clusters.ts";
import AlgorithmClusterSummary from "./AlgorithmClusterSummary";
import {
  FIVE_SIGNAL_MODES,
  type FiveSignalMode,
} from "./five-signal-flow-model.ts";
import ScopedAcousticWaveforms from "./ScopedAcousticWaveforms";
import ScopedDatasetVisualization, {
  type ScopedCaseRenderContext,
} from "./ScopedDatasetVisualization";
import {
  SCENE_SUBPAGES,
  SCENE_SUBPAGE_LABELS,
  SCOPED_BATCH_LABELS,
  SCOPED_INPUT_SIGNAL_ROLES,
  SCOPED_STATUS_LABELS,
  readSceneConditionDatasetView,
  sceneSubpageFromHash,
  sceneSubpageHash,
  sceneSubpageRoute,
  scopedMetricUnit,
  scopedValue,
  type SceneSubpageId,
  type ScopedDataset,
  type ScopedInputPreviewCase,
  type ScopedPreviewWaveformTrack,
  type ScopedTestDefinition,
  type ScopedTestPageProps,
} from "./scoped-acoustic-model.ts";
import {
  SCENE_CONDITION_BATCHES_URL,
  SCENE_CONDITION_CONTRACT_ITEM_COUNT,
  SCENE_CONDITION_DATASETS_URL,
  readSceneConditionBatch,
  readSceneConditionBatchHistory,
  sceneConditionCaseRxEvidence,
  sceneConditionCaseTxEvidence,
  type SceneConditionBatch,
  type SceneConditionCase,
  type SceneConditionObservation,
} from "./scene-condition-test-model.ts";
import {
  sceneConditionProfile,
  sceneConditionCounts,
  conversationModeOf,
  CONVERSATION_MODE_LABELS,
  type SceneConditionItemId,
  type SceneConditionProfile,
} from "./scene-condition-profiles.ts";
import {
  evaluateSceneConditionTableLowerLimit,
  sceneConditionLimitUnresolvedCount,
  sceneConditionTableLowerLimit,
  summarizeSceneConditionTableLimits,
  type SceneConditionLimitResult,
  type SceneConditionLimitSummary,
} from "./scene-condition-limits.ts";
import { businessModeLabel as modeLabel } from "../domain/business-modes.ts";
import "./basic-uplink-test.css";
import "./scoped-acoustic-test.css";
import "./scene-condition-test.css";

const POLL_INTERVAL_MS = 2_000;

const CONTRACT_FAMILIES = [
  {
    family: "g_mos",
    label: "整体质量",
    detail: "DNSMOS P.835 OVRL 工程代理",
  },
  {
    family: "n_mos",
    label: "背景噪声质量",
    detail: "DNSMOS P.835 BAK 工程代理",
  },
  {
    family: "s_mos",
    label: "噪声中语音质量",
    detail: "DNSMOS P.835 SIG 工程代理",
  },
  {
    family: "cng_level",
    label: "舒适噪声电平",
    detail: "noise 与 TX 的数字 RMS 电平差",
  },
  {
    family: "cng_spectrum",
    label: "舒适噪声频谱",
    detail: "归一化 PSD 的 Jensen-Shannon 距离",
  },
] as const;

function contractGroups(profile: SceneConditionProfile) {
  return profile.contractItemIds.map((itemId, index) => ({
    itemId,
    ...CONTRACT_FAMILIES[index],
  }));
}

type ContractGroup = ReturnType<typeof contractGroups>[number];

function batchScope(
  batch: SceneConditionBatch,
  counts: ReturnType<typeof sceneConditionCounts>,
): BatchScope {
  const fields = batch;
  const single = fields.execution_scope === "single_case";
  return {
    single,
    selectedCaseId: single ? fields.selected_case_id ?? "" : "",
    caseCount: single ? fields.case_count ?? 1 : counts.cases,
    observationCount: batch.observation_count,
    datasetCaseCount: single
      ? fields.dataset_case_count ?? counts.cases
      : counts.cases,
    datasetObservationCount: single
      ? fields.dataset_observation_count ?? counts.observations
      : counts.observations,
  };
}

function batchCaseScopeMatches(batch: SceneConditionBatch): boolean {
  const fields = batch;
  if (fields.execution_scope !== "single_case") return true;
  const caseId = fields.selected_case_id ?? "";
  return Boolean(
    caseId
      && batch.cases.length === 1
      && batch.cases[0]?.case_id === caseId
      && batch.items.every((item) => item.case_id === caseId)
      && batch.progress.total_cases === 1
      && batch.progress.completed_cases <= 1
      && batch.progress.total_items === batch.observation_count,
  );
}

function mergeBatchRows(
  ...groups: readonly SceneConditionBatch[][]
): SceneConditionBatch[] {
  const rows = new Map<string, SceneConditionBatch>();
  for (const group of groups) {
    for (const row of group) {
      const current = rows.get(row.batch_id);
      if (
        !current
        || Date.parse(row.updated_at) > Date.parse(current.updated_at)
        || (
          Date.parse(row.updated_at) === Date.parse(current.updated_at)
          && row.created_at > current.created_at
        )
      ) {
        rows.set(row.batch_id, row);
      }
    }
  }
  return [...rows.values()].sort(
    (left, right) =>
      Date.parse(right.created_at) - Date.parse(left.created_at)
      || right.batch_id.localeCompare(left.batch_id),
  );
}

function latestCaseResultIsNewer(
  candidate: LatestCaseResult,
  current: LatestCaseResult,
): boolean {
  const createdDelta =
    Date.parse(candidate.batch.created_at)
    - Date.parse(current.batch.created_at);
  if (createdDelta !== 0) return createdDelta > 0;
  const updatedDelta =
    Date.parse(candidate.batch.updated_at)
    - Date.parse(current.batch.updated_at);
  if (updatedDelta !== 0) return updatedDelta > 0;
  return candidate.batch.batch_id > current.batch.batch_id;
}

function caseIdentityMatches(
  batch: SceneConditionBatch,
  caseRecord: SceneConditionCase,
  previewCase: ScopedInputPreviewCase,
  context: ScopedCaseRenderContext,
  algorithmClusterId: string,
): boolean {
  return Boolean(
    batch.dataset_id === context.datasetId
      && batch.selected_mode === context.mode
      && batch.selected_scene === context.scene
      && batch.algorithm_cluster_id === algorithmClusterId
      && caseRecord.case_id === previewCase.case_id
      && caseRecord.conditions.mode === previewCase.conditions.mode
      && caseRecord.conditions.direction === previewCase.conditions.direction
      && caseRecord.conditions.scene === previewCase.conditions.scene
      && caseRecord.conditions.volume === previewCase.conditions.volume
      && caseRecord.conditions.window_ms === previewCase.conditions.window_ms
      && caseRecord.conditions.input_level_db
        === previewCase.conditions.input_level_db
      && SCOPED_INPUT_SIGNAL_ROLES.every(
        (role) =>
          caseRecord.signals_sha256[role]
            === previewCase.signals_sha256[role],
      ),
  );
}

function decodeWaveformInt16(encoded: string): Int16Array {
  const binary = atob(encoded);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  if (bytes.byteLength % 2 !== 0) {
    throw new Error("TX 预计算波形字节长度无效");
  }
  const view = new DataView(
    bytes.buffer,
    bytes.byteOffset,
    bytes.byteLength,
  );
  const values = new Int16Array(bytes.byteLength / 2);
  for (let index = 0; index < values.length; index += 1) {
    values[index] = view.getInt16(index * 2, true);
  }
  return values;
}

function txWaveformPath(
  visual: ScopedPreviewWaveformTrack,
): string {
  const minimum = decodeWaveformInt16(visual.wave_min_i16);
  const maximum = decodeWaveformInt16(visual.wave_max_i16);
  if (
    minimum.length !== TX_WAVEFORM_WIDTH
    || maximum.length !== TX_WAVEFORM_WIDTH
  ) {
    throw new Error("TX 预计算波形数组长度无效");
  }
  const scale = visual.wave_scale / 32767;
  const denominator = visual.peak > 0 ? visual.peak : 1;
  const points: string[] = [];
  for (let index = 0; index < TX_WAVEFORM_WIDTH; index += 1) {
    const normalized = Math.max(
      -1,
      Math.min(1, (maximum[index] * scale) / denominator),
    );
    points.push(
      `${index},${
        (TX_WAVEFORM_CENTER - normalized * TX_WAVEFORM_AMPLITUDE).toFixed(2)
      }`,
    );
  }
  for (let index = TX_WAVEFORM_WIDTH - 1; index >= 0; index -= 1) {
    const normalized = Math.max(
      -1,
      Math.min(1, (minimum[index] * scale) / denominator),
    );
    points.push(
      `${index},${
        (TX_WAVEFORM_CENTER - normalized * TX_WAVEFORM_AMPLITUDE).toFixed(2)
      }`,
    );
  }
  return `M${points.join(" L")} Z`;
}
const DATE_FORMAT = new Intl.DateTimeFormat("zh-CN", {
  timeZone: "Asia/Shanghai",
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
});
const INTEGER_FORMAT = new Intl.NumberFormat("zh-CN");

type Props = ScopedTestPageProps & {
  definition: ScopedTestDefinition;
};

type DatasetState = {
  dataset: ScopedDataset | null;
  datasetId: string;
};

type ContractStats = {
  count: number;
  mean: number | null;
  minimum: number | null;
  maximum: number | null;
  lowerLimit: number | null;
  tableLimit: SceneConditionLimitSummary;
  unit: string;
  pass: number;
  fail: number;
  reviewRequired: number;
  notMeasured: number;
  invalid: number;
};

type BatchScope = {
  single: boolean;
  selectedCaseId: string;
  caseCount: number;
  observationCount: number;
  datasetCaseCount: number;
  datasetObservationCount: number;
};

type LatestCaseResult = {
  batch: SceneConditionBatch;
  caseRecord: SceneConditionCase;
  items: SceneConditionObservation[];
};

const TX_WAVEFORM_WIDTH = 720;
const TX_WAVEFORM_HEIGHT = 48;
const TX_WAVEFORM_CENTER = TX_WAVEFORM_HEIGHT / 2;
const TX_WAVEFORM_AMPLITUDE = TX_WAVEFORM_CENTER - 1;
const TX_SIGNAL_COLOR = "#4f7fb8";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function apiErrorDetail(value: unknown): string {
  if (!isRecord(value)) return "";
  if (typeof value.detail === "string") return value.detail;
  if (!Array.isArray(value.detail)) return "";
  return value.detail
    .map((entry) =>
      isRecord(entry) && typeof entry.msg === "string" ? entry.msg : ""
    )
    .filter(Boolean)
    .join("；");
}

async function requestJson(
  url: string,
  init: RequestInit,
  operation: string,
): Promise<unknown> {
  const response = await fetch(url, {
    cache: "no-store",
    ...init,
  });
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    throw new Error(
      response.ok
        ? `${operation}返回了无法解析的 JSON`
        : `${operation}失败（HTTP ${response.status}）`,
    );
  }
  if (!response.ok) {
    throw new Error(
      apiErrorDetail(body)
        || `${operation}失败（HTTP ${response.status}）`,
    );
  }
  return body;
}

function errorMessage(reason: unknown, fallback: string): string {
  return reason instanceof Error && reason.message
    ? reason.message
    : fallback;
}

function formatDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value || "—" : DATE_FORMAT.format(date);
}

function algorithmLabel(id: string): string {
  return id ? algorithmClusterLabel(id) : "未选择";
}

function latestStorageKey(
  profile: SceneConditionProfile,
  algorithmClusterId: string,
  mode: FiveSignalMode = profile.mode ?? "WB-wired",
): string {
  return (
    `scene-condition:${profile.scene}:${profile.datasetId}:`
    + `${mode}:${algorithmClusterId}:latest`
  );
}

function readStoredBatchId(key: string): {
  batchId: string;
  invalid: boolean;
} {
  if (!key || typeof window === "undefined") {
    return { batchId: "", invalid: false };
  }
  try {
    const value = window.localStorage.getItem(key)?.trim() ?? "";
    if (!value) return { batchId: "", invalid: false };
    return /^[a-f0-9]{32}$/.test(value)
      ? { batchId: value, invalid: false }
      : { batchId: "", invalid: true };
  } catch {
    return { batchId: "", invalid: false };
  }
}

function storeBatchId(key: string, batchId: string) {
  if (!key || typeof window === "undefined") return;
  try {
    window.localStorage.setItem(key, batchId);
  } catch {
    // Storage can be disabled without invalidating the live server snapshot.
  }
}

function batchMatches(
  batch: SceneConditionBatch | null,
  datasetId: string,
  algorithmClusterId: string,
  mode: FiveSignalMode,
  scene: SceneConditionProfile["scene"],
): batch is SceneConditionBatch {
  return Boolean(
    batch
      && batch.dataset_id === datasetId
      && batch.algorithm_cluster_id === algorithmClusterId
      && batch.selected_mode === mode
      && batch.selected_scene === scene
      && batchCaseScopeMatches(batch),
  );
}

function toScopedDataset(
  value: ReturnType<typeof readSceneConditionDatasetView>,
): DatasetState {
  const dataset = value.datasets[0];
  return {
    datasetId: dataset?.dataset_id ?? "",
    dataset: dataset
      ? {
          dataset_id: dataset.dataset_id,
          name: dataset.dataset_name,
          item_count: SCENE_CONDITION_CONTRACT_ITEM_COUNT,
          case_count: value.preview_condition_group_count,
          created_at: "",
        }
      : null,
  };
}

function previewCaseFromBatch(
  batch: SceneConditionBatch,
  caseRecord: SceneConditionCase,
): ScopedInputPreviewCase {
  const items = batch.items.filter(
    (item) => item.case_id === caseRecord.case_id,
  );
  return {
    case_id: caseRecord.case_id,
    conditions: caseRecord.conditions,
    families: [...new Set(items.map((item) => item.family))],
    signals: [...SCOPED_INPUT_SIGNAL_ROLES],
    signals_sha256: { ...caseRecord.signals_sha256 },
    provenance: caseRecord.provenance,
  };
}

function contractStats(
  items: SceneConditionObservation[],
  itemId: SceneConditionItemId,
): ContractStats {
  const rows = items.filter((item) => item.item_id === itemId);
  const values = rows
    .map((item) => item.value)
    .filter(
      (value): value is number =>
        typeof value === "number" && Number.isFinite(value),
    );
  return {
    count: values.length,
    mean:
      values.length > 0
        ? values.reduce((sum, value) => sum + value, 0) / values.length
        : null,
    minimum: values.length > 0 ? Math.min(...values) : null,
    maximum: values.length > 0 ? Math.max(...values) : null,
    lowerLimit: sceneConditionTableLowerLimit(itemId),
    tableLimit: summarizeSceneConditionTableLimits(rows),
    unit: scopedMetricUnit(rows.find((item) => item.metric)?.metric),
    pass: rows.filter((item) => item.status === "pass").length,
    fail: rows.filter((item) => item.status === "fail").length,
    reviewRequired: rows.filter(
      (item) => item.status === "review_required",
    ).length,
    notMeasured: rows.filter(
      (item) => item.status === "not_measured",
    ).length,
    invalid: rows.filter((item) => item.status === "invalid").length,
  };
}

function ItemStatusBadge({
  status,
}: {
  status: SceneConditionObservation["status"];
}) {
  const Icon =
    status === "pass"
      ? CheckCircle2
      : status === "fail"
        ? XCircle
        : status === "review_required"
          ? AlertCircle
          : status === "not_measured" || status === "invalid"
            ? MinusCircle
            : Clock3;
  return (
    <span className={`basic-uplink-item-status is-${status}`}>
      <Icon size={13} aria-hidden="true" />
      {SCOPED_STATUS_LABELS[status]}
    </span>
  );
}

function TableLimitBadge({
  result,
}: {
  result: SceneConditionLimitResult;
}) {
  const weakLimit = result.state === "meets" && result.lowerLimit === 1;
  const Icon =
    result.state === "meets"
      ? CheckCircle2
      : result.state === "below"
        ? XCircle
        : result.state === "pending"
          || result.state === "unavailable"
          ? AlertCircle
          : MinusCircle;
  const labels = {
    meets: "不低于表内下限",
    below: "低于表内下限",
    pending: "表内不可对照 · 待执行",
    not_measured: "表内不可对照 · 不可测",
    invalid: "表内不可对照 · 无效",
    unavailable: "表内不可对照 · 缺值",
    not_applicable: "表内下限不适用",
  } as const;
  const label = weakLimit
    ? "不低于表内弱下限"
    : labels[result.state];
  const detail =
    result.lowerLimit === null
      ? "当前工程代理没有可比较的 NV200 数值下限"
      : `${result.sourceCell} 下限 ${scopedValue(result.lowerLimit)}；`
        + "仅作 DNSMOS 工程代理与客户表数值的对照，不代表正式合同或 ITU-T 合规通过"
        + (weakLimit ? "；G-MOS 表内下限 1 的区分能力很弱" : "");
  return (
    <span
      className={`sct-table-limit-status is-${result.state}${
        weakLimit ? " is-weak" : ""
      }`}
      title={detail}
      aria-label={`${label}。${detail}`}
    >
      <Icon size={13} aria-hidden="true" />
      {label}
    </span>
  );
}

function CaseTableLimitSummary({
  summary,
  mode,
}: {
  summary: SceneConditionLimitSummary;
  mode: FiveSignalMode;
}) {
  const unresolved = sceneConditionLimitUnresolvedCount(summary);
  const state =
    summary.below > 0
      ? "below"
      : summary.comparable > 0 && unresolved === 0
        ? "meets"
        : summary.comparable > 0
          ? "partial"
          : summary.notMeasured > 0
            ? "not-measured"
            : summary.invalid > 0
              ? "invalid"
              : summary.pending > 0
                ? "pending"
                : summary.unavailable > 0
            ? "unavailable"
            : "not-applicable";
  let label = "表内下限不适用";
  if (state === "below") {
    label = `${summary.below} 项低于表内下限`;
  } else if (state === "meets") {
    label = `${summary.meets}/${summary.comparable} 项不低于表内下限`;
  } else if (state === "partial") {
    label = `${summary.meets}/${summary.comparable} 项达到 · ${
      unresolved
    } 项表内不可对照`;
  } else if (state === "not-measured") {
    label = `${summary.notMeasured} 项表内不可对照 · 不可测`;
  } else if (state === "invalid") {
    label = `${summary.invalid} 项表内不可对照 · 无效`;
  } else if (state === "pending") {
    label = `${summary.pending} 项表内不可对照 · 待执行`;
  } else if (state === "unavailable") {
    label = `${summary.unavailable} 项表内不可对照 · 缺值`;
  }
  return (
    <span
      className={`sct-case-limit-summary is-${state}`}
      title={`按 NV200 ${mode} 客户表 E 列数值对照；不会改变服务端合同状态`}
    >
      {label}
    </span>
  );
}

function CompactOutputWaveform({
  visual,
  caseId,
  label,
}: {
  visual: ScopedPreviewWaveformTrack;
  caseId: string;
  label: string;
}) {
  let path = "";
  let pathError = "";
  try {
    path = txWaveformPath(visual);
  } catch (reason) {
    pathError = errorMessage(reason, "TX 波形无法绘制");
  }
  return (
    <div className="sct-inline-tx-waveform">
      {pathError ? (
        <span>{pathError}</span>
      ) : (
        <svg
          viewBox={`0 0 ${TX_WAVEFORM_WIDTH} ${TX_WAVEFORM_HEIGHT}`}
          preserveAspectRatio="none"
          role="img"
          aria-label={`条件组 ${caseId} 的 ${label}波形`}
        >
          {visual.peak === 0 ? (
            <line
              x1="0"
              x2={TX_WAVEFORM_WIDTH}
              y1={TX_WAVEFORM_CENTER}
              y2={TX_WAVEFORM_CENTER}
              stroke={TX_SIGNAL_COLOR}
              strokeWidth="1"
              opacity={0.76}
            />
          ) : (
            <path
              d={path}
              fill={TX_SIGNAL_COLOR}
              opacity={0.76}
            />
          )}
        </svg>
      )}
      <span>
        {visual.duration_s.toFixed(2)} s ·{" "}
        {(visual.sample_rate / 1000).toLocaleString("zh-CN", {
          maximumFractionDigits: 2,
        })}{" "}
        kHz · {visual.channels} 通道
      </span>
    </div>
  );
}

function InlineCaseResult({
  previewCase,
  result,
  groups,
  launching,
  error,
  datasetCaseCount,
  farEndVisual,
  onOpenWaveforms,
  onToggleOutputPlayback,
  outputPlaybackKey,
  mode,
}: {
  previewCase: ScopedInputPreviewCase;
  result: LatestCaseResult | null;
  groups: ContractGroup[];
  launching: boolean;
  error: string;
  datasetCaseCount: number;
  farEndVisual: ScopedPreviewWaveformTrack | null;
  onOpenWaveforms: () => void;
  onToggleOutputPlayback: (key: string) => void;
  outputPlaybackKey: string;
  mode: FiveSignalMode;
}) {
  if (!result && !launching && !error) return null;
  if (!result) {
    return (
      <div
        className={`sct-inline-case-result ${
          error ? "is-error" : "is-pending"
        }`}
        role={error ? "alert" : "status"}
        aria-live="polite"
      >
        <strong>
          {error ? "单组测试创建失败" : "正在创建单组测试"}
        </strong>
        <p>
          {error
            || `条件组 ${previewCase.case_id} 正在提交；不会跳转页面。`}
        </p>
      </div>
    );
  }

  const tx = sceneConditionCaseTxEvidence(
    previewCase,
    result.batch,
    result.batch.dataset_manifest_sha256,
  );
  const rx = sceneConditionCaseRxEvidence(
    previewCase,
    result.batch,
    result.batch.dataset_manifest_sha256,
  );
  const items = [...result.items].sort(
    (left, right) => left.item_id.localeCompare(right.item_id),
  );
  const tableLimitSummary = summarizeSceneConditionTableLimits(items);
  const scope = result.batch;
  const selectedCaseId = scope.selected_case_id ?? previewCase.case_id;
  const single = scope.execution_scope === "single_case";
  const playbackBase = `${result.batch.batch_id}:${previewCase.case_id}`;
  const rxBeforeUrl = `${SCENE_CONDITION_DATASETS_URL}/${encodeURIComponent(
    result.batch.dataset_id,
  )}/cases/${encodeURIComponent(
    previewCase.case_id,
  )}/inputs/far_end?v=${encodeURIComponent(
    previewCase.signals_sha256.far_end,
  )}`;

  return (
    <section
      className={`sct-inline-case-result is-${result.caseRecord.status}`}
      aria-label={`条件组 ${previewCase.case_id} 的${
        single ? "单组" : ""
      }测试结果`}
      aria-live="polite"
    >
      <header>
        <div>
          <strong>{single ? "单组测试结果" : "该组测试结果"}</strong>
          <span>
            {single
              ? `单组 ${result.batch.progress.completed_cases}/1 · `
              : ""}
            数据集 {datasetCaseCount} 组
          </span>
        </div>
        <div className="sct-inline-case-badges">
          <CaseTableLimitSummary summary={tableLimitSummary} mode={mode} />
          <span className={`sct-inline-case-status is-${result.batch.status}`}>
            {SCOPED_BATCH_LABELS[result.batch.status]}
            {" · "}
            {result.caseRecord.status}
          </span>
        </div>
      </header>
      <p className="sct-inline-case-identity">
        <code>{selectedCaseId}</code>
        <span>
          {result.batch.progress.completed_items}/
          {result.batch.observation_count} 条该组合同记录
        </span>
      </p>
      <p className="sct-inline-limit-boundary">
        “不低于表内下限”仅比较 DNSMOS P.835 原始分数与 NV200
        {mode} 客户表 E 列数值；不会改变服务端合同状态，
        也不代表正式 ITU-T 合规通过。
      </p>
      {error ? (
        <p className="sct-inline-case-error" role="alert">
          {error}
        </p>
      ) : null}
      {items.length > 0 ? (
        <dl className="sct-inline-contract-values">
          {items.map((item) => {
            const group = groups.find(
              (candidate) => candidate.itemId === item.item_id,
            );
            const tableLimit =
              evaluateSceneConditionTableLowerLimit(
                item.item_id,
                item.value,
                item.status,
              );
            return (
              <div key={item.observation_id}>
                <dt>
                  <code>{item.item_id}</code>
                  <span>{group?.label ?? item.test_name}</span>
                </dt>
                <dd>
                  <strong>{scopedValue(item.value)}</strong>
                  <span>
                    {scopedMetricUnit(item.metric) || "无量纲/未声明"}
                  </span>
                  <TableLimitBadge result={tableLimit} />
                  <ItemStatusBadge status={item.status} />
                  <small>
                    {tableLimit.lowerLimit === null
                      ? "NV200 表内无适用数值下限。"
                      : `NV200 ${tableLimit.sourceCell} 下限 ${scopedValue(
                          tableLimit.lowerLimit,
                        )}。`}
                    {" "}
                    {item.reason || "未返回原因"}
                  </small>
                </dd>
              </div>
            );
          })}
        </dl>
      ) : (
        <p className="sct-inline-case-waiting">
          正在等待该组的合同观测。
        </p>
      )}
      <NsQualityMetrics
        batchId={result.batch.batch_id}
        caseId={previewCase.case_id}
        outputSha={result.caseRecord.output_sha256?.tx}
        completed={result.caseRecord.status === "completed"}
      />
      <div className="sct-inline-echo">
        <div className="sct-inline-tx-heading">
          <div>
            <strong>回声参考 · VCP RX 前 / 后</strong>
            <span>
              RX 前来自数据集 far_end；RX 后在本次测试中现场计算，
              并作为 TX 的实际回声参考。
            </span>
          </div>
          <button
            type="button"
            aria-label={`查看条件组 ${previewCase.case_id} 完整波形`}
            onClick={onOpenWaveforms}
          >
            查看波形
          </button>
        </div>
        <div className="sct-inline-echo-grid">
          <article>
            <header>
              <strong>RX 前</strong>
              <span>far_end 原始播放 PCM</span>
            </header>
            {farEndVisual ? (
              <CompactOutputWaveform
                visual={farEndVisual}
                caseId={previewCase.case_id}
                label="RX 前回声参考"
              />
            ) : (
              <p className="sct-inline-case-unavailable">
                RX 前波形暂不可用
              </p>
            )}
            <OutputAudioPreview
              caseId={previewCase.case_id}
              label="RX 前回声参考音频"
              url={rxBeforeUrl}
              playing={outputPlaybackKey === `${playbackBase}:rx-before`}
              onToggle={() =>
                onToggleOutputPlayback(`${playbackBase}:rx-before`)}
            />
          </article>
          <article>
            <header>
              <strong>RX 后</strong>
              <span>本次 VCP RX 输出 · TX 实际使用</span>
            </header>
            {rx.ready && rx.visual ? (
              <>
                <CompactOutputWaveform
                  visual={rx.visual}
                  caseId={previewCase.case_id}
                  label="RX 后回声参考"
                />
                <div className="sct-inline-tx-actions">
                  <a
                    href={rx.url}
                    download
                    aria-label={`下载条件组 ${previewCase.case_id} RX 后回声参考`}
                  >
                    <Download size={13} aria-hidden="true" />
                    下载 RX 后
                  </a>
                </div>
                <OutputAudioPreview
                  caseId={previewCase.case_id}
                  label="RX 后回声参考音频"
                  url={rx.url}
                  playing={
                    outputPlaybackKey === `${playbackBase}:rx-after`
                  }
                  onToggle={() =>
                    onToggleOutputPlayback(`${playbackBase}:rx-after`)}
                />
              </>
            ) : (
              <p className="sct-inline-case-unavailable">
                RX 后：{rx.reason}
              </p>
            )}
          </article>
        </div>
      </div>
      {tx.ready && tx.visual ? (
        <div className="sct-inline-tx">
          <div className="sct-inline-tx-heading">
            <div>
              <strong>TX 输出波形</strong>
              <span>来源哈希已与该组结果绑定</span>
            </div>
            <div className="sct-inline-tx-actions">
              <a
                href={tx.url}
                download
                aria-label={`下载条件组 ${previewCase.case_id} TX`}
              >
                <Download size={13} aria-hidden="true" />
                下载 TX
              </a>
              <button
                type="button"
                aria-label={`查看条件组 ${previewCase.case_id} 完整波形`}
                onClick={onOpenWaveforms}
              >
                查看波形
              </button>
            </div>
          </div>
          <CompactOutputWaveform
            visual={tx.visual}
            caseId={previewCase.case_id}
            label="TX 输出"
          />
          <OutputAudioPreview
            caseId={previewCase.case_id}
            label="TX 输出音频"
            url={tx.url}
            playing={outputPlaybackKey === `${playbackBase}:tx`}
            onToggle={() =>
              onToggleOutputPlayback(`${playbackBase}:tx`)}
          />
        </div>
      ) : (
        <p className="sct-inline-case-unavailable">
          TX：{tx.reason}
        </p>
      )}
      {result.caseRecord.error ? (
        <p className="sct-inline-case-error" role="alert">
          {result.caseRecord.error}
        </p>
      ) : null}
    </section>
  );
}

function ActiveSubpage({
  active,
  label,
  children,
}: {
  active: boolean;
  label: string;
  children: ReactNode;
}) {
  if (!active) return null;
  return (
    <div
      className="scoped-acoustic-subpage-block"
      role="group"
      aria-label={label}
    >
      {children}
    </div>
  );
}

function ConversationDiagnostics({ value }: { value: unknown }) {
  if (!isRecord(value) || !Array.isArray(value.segments)) return null;
  const segments = value.segments.filter(isRecord);
  const number = (candidate: unknown, suffix = "") =>
    typeof candidate === "number" && Number.isFinite(candidate)
      ? `${candidate.toFixed(2)}${suffix}` : "—";
  return <details className="sct-conversation-diagnostics">
    <summary>单讲 / 双讲诊断 · 工程代理</summary>
    <p>按实际活动段检查近端保留和远端抑制；静音、短片段和不适用的指标保留为空。</p>
    {typeof value.reason === "string" ? <p>{value.reason}</p> : null}
    <div className="table-scroll">
      <table className="audio-table">
        <caption className="sr-only">通话区段工程诊断，正式 ITU-T 合规未测</caption>
        <thead><tr><th>区段</th><th>时间</th><th>近端增益</th><th>近端相关度</th><th>远端段降低</th></tr></thead>
        <tbody>{segments.map((segment, index) => {
          const metrics = isRecord(segment.metrics) ? segment.metrics : {};
          const label = segment.mode === "near" ? "近端单讲"
            : segment.mode === "far" ? "远端单讲"
            : segment.mode === "noise" ? "噪声观测"
            : typeof segment.mode === "string" && Object.hasOwn(CONVERSATION_MODE_LABELS, segment.mode)
            ? CONVERSATION_MODE_LABELS[segment.mode as keyof typeof CONVERSATION_MODE_LABELS]
            : String(segment.label ?? "通话区段");
          return <tr key={index}>
            <td>{label}{["not_measured", "not_measurable"].includes(String(segment.status)) ? " · 不可测" : ""}</td>
            <td>{number(segment.start_s)}–{number(segment.end_s)} s</td>
            <td>{number(metrics.near_gain_db, " dB")}</td>
            <td>{number(metrics.near_correlation)}</td>
            <td>{number(metrics.echo_reduction_db, " dB")}</td>
          </tr>;
        })}</tbody>
      </table>
    </div>
    <p>远端段降低包含背景噪声变化，不等同于纯回声 ERLE；近端增益与相关度不等同于正式双讲分类。</p>
  </details>;
}

export default function SceneConditionTest({
  definition,
  algorithmClusterId,
  mode,
  onModeChange,
  onAlgorithmClusterChange,
  externalRunning = false,
  onBusyChange,
  pageVisible = true,
}: Props) {
  if (!definition.scene) {
    throw new Error("SceneConditionTest 必须显式声明背景噪声场景");
  }
  const scene = definition.scene;
  const profile = sceneConditionProfile(scene, mode);
  const counts = sceneConditionCounts(profile);
  const SCENE_CONDITION_BATCH_COUNT = counts.cases;
  const SCENE_CONDITION_OBSERVATION_COUNT = counts.observations;
  const groups = contractGroups(profile);
  const uid = useId().replace(/[^A-Za-z0-9_-]/g, "");
  const titleId = `scene-condition-${uid}-title`;
  const modeSelectId = `scene-condition-${uid}-mode`;
  const clusterSelectId = `scene-condition-${uid}-cluster`;
  const historySelectId = `scene-condition-${uid}-history`;
  const modeSupported = FIVE_SIGNAL_MODES.includes(mode);
  const modeRateHz = sampleRateForBusinessMode(mode);
  const availableAlgorithmGroups = algorithmClusterGroupsForMode(mode, {
    vcpOnly: true,
  });
  const availableAlgorithmCount = algorithmClustersForMode(mode, {
    vcpOnly: true,
  }).length;
  const algorithmIsSupported = isAlgorithmClusterSupportedForMode(
    algorithmClusterId,
    mode,
    { vcpOnly: true },
  );
  const selectedAlgorithmCluster = algorithmClusterById(algorithmClusterId);
  const storageKey =
    modeSupported && algorithmIsSupported
      ? latestStorageKey(profile, algorithmClusterId, mode)
      : "";
  const [sceneSubpage, setSceneSubpage] = useState<SceneSubpageId>(() =>
    sceneSubpageFromHash(window.location.hash, definition.route),
  );

  const [dataset, setDataset] = useState<ScopedDataset | null>(null);
  const [datasetId, setDatasetId] = useState("");
  const [datasetLoading, setDatasetLoading] = useState(false);
  const [datasetError, setDatasetError] = useState("");
  const [datasetRefresh, setDatasetRefresh] = useState(0);

  const [historyRows, setHistoryRows] = useState<SceneConditionBatch[]>([]);
  const [historySelection, setHistorySelection] = useState("");
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState("");
  const [restoreError, setRestoreError] = useState("");
  const [restoring, setRestoring] = useState(false);
  const [restoreNonce, setRestoreNonce] = useState(0);

  const [snapshot, setSnapshot] = useState<SceneConditionBatch | null>(null);
  const [launching, setLaunching] = useState(false);
  const [launchingCaseId, setLaunchingCaseId] = useState("");
  const [caseRunErrors, setCaseRunErrors] = useState<Record<string, string>>(
    {},
  );
  const [runError, setRunError] = useState("");
  const [pollError, setPollError] = useState("");
  const [pollRetry, setPollRetry] = useState(0);
  const [outputPlaybackKey, setOutputPlaybackKey] = useState("");
  const toggleOutputPlayback = (key: string) =>
    setOutputPlaybackKey((current) => (current === key ? "" : key));

  const launchAbortRef = useRef<AbortController | null>(null);
  const restoreAbortRef = useRef<AbortController | null>(null);
  const restoreTargetRef = useRef("");

  const waveformDefinition = useMemo<ScopedTestDefinition>(
    () => ({
      ...definition,
      scope: "background_noise",
      scene,
      previewMode: "scene-condition-20",
    }),
    [definition, scene],
  );

  useEffect(() => {
    const syncSubpage = () => {
      const hash = window.location.hash;
      if (sceneSubpageRoute(hash) !== definition.route) return;
      const next = sceneSubpageFromHash(hash, definition.route);
      setSceneSubpage(next);
      const canonical = sceneSubpageHash(definition.route, next);
      if (hash !== canonical) {
        window.history.replaceState(null, "", canonical);
      }
    };
    syncSubpage();
    window.addEventListener("hashchange", syncSubpage);
    window.addEventListener("popstate", syncSubpage);
    return () => {
      window.removeEventListener("hashchange", syncSubpage);
      window.removeEventListener("popstate", syncSubpage);
    };
  }, [definition.route]);

  useEffect(() => {
    setSnapshot(null);
    setHistoryRows([]);
    setHistorySelection("");
    setLaunchingCaseId("");
    setCaseRunErrors({});
    setOutputPlaybackKey("");
    setRunError("");
    setPollError("");
    setHistoryError("");
    setRestoreError("");
    restoreTargetRef.current = "";
  }, [algorithmClusterId, mode]);

  useEffect(() => {
    if (!modeSupported) {
      setDataset(null);
      setDatasetId("");
      setDatasetLoading(false);
      setDatasetError("");
      return;
    }
    const controller = new AbortController();
    setDataset(null);
    setDatasetId("");
    setDatasetLoading(true);
    setDatasetError("");
    const params = new URLSearchParams({
      scope: "background_noise",
      mode,
      scene,
    });
    requestJson(
      `${SCENE_CONDITION_DATASETS_URL}?${params.toString()}`,
      { signal: controller.signal },
      `读取场景${profile.sceneLabel}测试集`,
    )
      .then((body) => {
        const view = readSceneConditionDatasetView(body, {
          mode,
          scene,
        });
        const next = toScopedDataset(view);
        if (controller.signal.aborted) return;
        setDataset(next.dataset);
        setDatasetId(next.datasetId);
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) {
          setDataset(null);
          setDatasetId("");
          setDatasetError(
            errorMessage(
              reason,
              `场景${profile.sceneLabel}测试集读取失败`,
            ),
          );
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setDatasetLoading(false);
      });
    return () => controller.abort();
  }, [datasetRefresh, mode, modeSupported, profile.sceneLabel, scene]);

  useEffect(() => {
    if (
      !modeSupported
      || !algorithmIsSupported
      || !datasetId
      || !storageKey
    ) {
      setHistoryRows([]);
      setHistorySelection("");
      setHistoryLoading(false);
      setRestoring(false);
      return;
    }
    const controller = new AbortController();
    restoreAbortRef.current = controller;
    setHistoryLoading(true);
    setRestoring(true);
    setHistoryError("");
    setRestoreError("");

    const restore = async () => {
      const params = new URLSearchParams({
        limit: "20",
        dataset_id: datasetId,
        algorithm_cluster_id: algorithmClusterId,
        latest_per_case: "true",
      });
      let body: unknown;
      try {
        body = await requestJson(
          `${SCENE_CONDITION_BATCHES_URL}?${params.toString()}`,
          { signal: controller.signal },
          "读取场景测试历史",
        );
      } catch (reason) {
        if (!controller.signal.aborted) {
          setHistoryError(
            errorMessage(reason, "场景测试历史读取失败"),
          );
        }
        return;
      }
      let history: SceneConditionBatch[];
      try {
        history = readSceneConditionBatchHistory(body, {
          datasetId,
          algorithmClusterId,
          latestPerCase: true,
        });
      } catch (reason) {
        if (!controller.signal.aborted) {
          setHistoryRows([]);
          setSnapshot(null);
          setHistorySelection("");
          setHistoryError(
            `${errorMessage(
              reason,
              "场景测试历史格式无效",
            )}；旧报告不会进入本页。`,
          );
        }
        return;
      }
      if (controller.signal.aborted) return;
      setHistoryRows((current) => mergeBatchRows(current, history));
      const manualTarget = restoreTargetRef.current;
      const stored = readStoredBatchId(storageKey);
      const storedExists =
        Boolean(stored.batchId)
        && history.some((row) => row.batch_id === stored.batchId);
      const target =
        manualTarget
          ? history.some((row) => row.batch_id === manualTarget)
            ? manualTarget
            : ""
          : storedExists
            ? stored.batchId
            : history[0]?.batch_id ?? "";
      if (!target) {
        setHistorySelection("");
        setSnapshot(null);
        if (manualTarget) {
          setRestoreError("所选场景批次已不在新历史中。");
          restoreTargetRef.current = "";
        } else if (stored.invalid) {
          setRestoreError(
            "本地保存的场景批次编号无效；新的服务端历史为空。",
          );
        } else if (stored.batchId) {
          setRestoreError(
            "本地保存的场景批次不在新历史中；新的服务端历史为空。",
          );
        }
        return;
      }
      setHistorySelection(target);
      try {
        // History already contains freshly fetched, fully validated snapshots.
        // Reuse the selected one instead of downloading its visuals a second time.
        const next = history.find((row) => row.batch_id === target);
        if (!next) throw new Error("所选场景批次不在已校验的历史中");
        if (controller.signal.aborted) return;
        if (
          !batchMatches(
            next,
            datasetId,
            algorithmClusterId,
            mode,
            scene,
          )
        ) {
          throw new Error("恢复的快照与当前模式、场景、数据集或算法不一致");
        }
        setSnapshot(next);
        setHistorySelection(next.batch_id);
        setRunError("");
        setPollError("");
        storeBatchId(storageKey, next.batch_id);
        restoreTargetRef.current = "";
        if (manualTarget) {
          // A manual history selection takes precedence over the stored latest.
        } else if (stored.invalid) {
          setRestoreError(
            "本地保存的场景批次编号无效，已恢复新历史第一条。",
          );
        } else if (stored.batchId && !storedExists) {
          setRestoreError(
            "本地保存的场景批次不在新历史中，已恢复新历史第一条。",
          );
        }
      } catch (reason) {
        if (controller.signal.aborted) return;
        setSnapshot(null);
        restoreTargetRef.current = "";
        setRestoreError(
          errorMessage(reason, "恢复场景测试结果失败"),
        );
      }
    };

    void restore().finally(() => {
      if (!controller.signal.aborted) {
        setHistoryLoading(false);
        setRestoring(false);
        if (restoreAbortRef.current === controller) {
          restoreAbortRef.current = null;
        }
      }
    });
    return () => controller.abort();
  }, [
    algorithmClusterId,
    algorithmIsSupported,
    datasetId,
    modeSupported,
    mode,
    restoreNonce,
    scene,
    storageKey,
  ]);

  const isBatchActive =
    snapshot?.status === "queued" || snapshot?.status === "running";
  const snapshotCaseId =
    snapshot?.execution_scope === "single_case"
      ? snapshot.selected_case_id ?? ""
      : "";

  useEffect(() => {
    if (!snapshot || !isBatchActive) return;
    const controller = new AbortController();
    const batchId = snapshot.batch_id;
    const expectedDatasetId = snapshot.dataset_id;
    const expectedAlgorithmId = snapshot.algorithm_cluster_id;
    let disposed = false;
    let timer: number | undefined;

    const poll = async () => {
      let nextStatus: SceneConditionBatch["status"] =
        snapshot.status;
      try {
        const body = await requestJson(
          `${SCENE_CONDITION_BATCHES_URL}/${encodeURIComponent(batchId)}`,
          { signal: controller.signal },
          "查询场景测试进度",
        );
        const next = readSceneConditionBatch(body, {
          datasetId: expectedDatasetId,
          algorithmClusterId: expectedAlgorithmId,
          ...(snapshotCaseId ? { caseId: snapshotCaseId } : {}),
        });
        if (disposed || controller.signal.aborted) return;
        nextStatus = next.status;
        setSnapshot(next);
        setHistoryRows((current) => mergeBatchRows(current, [next]));
        setPollError("");
      } catch (reason) {
        if (!disposed && !controller.signal.aborted) {
          setPollError(
            `${errorMessage(
              reason,
              "场景测试进度查询失败",
            )}；继续保留最近成功快照，每 2 秒重试。`,
          );
        }
      }
      if (
        !disposed
        && !controller.signal.aborted
        && (nextStatus === "queued" || nextStatus === "running")
      ) {
        timer = window.setTimeout(poll, POLL_INTERVAL_MS);
      }
    };

    timer = window.setTimeout(poll, POLL_INTERVAL_MS);
    return () => {
      disposed = true;
      if (timer !== undefined) window.clearTimeout(timer);
      controller.abort();
    };
  }, [
    isBatchActive,
    pollRetry,
    snapshot?.batch_id,
    snapshot?.status,
    snapshotCaseId,
  ]);

  useEffect(() => {
    onBusyChange?.(
      launching || restoring || Boolean(isBatchActive),
    );
  }, [
    isBatchActive,
    launching,
    onBusyChange,
    restoring,
  ]);

  useEffect(() => {
    return () => {
      launchAbortRef.current?.abort();
      restoreAbortRef.current?.abort();
    };
  }, []);

  const currentSnapshot = batchMatches(
    snapshot,
    datasetId,
    algorithmClusterId,
    mode,
    scene,
  )
    ? snapshot
    : null;
  const controlsLocked =
    externalRunning
    || launching
    || restoring
    || Boolean(isBatchActive);
  const canStart =
    modeSupported
    && algorithmIsSupported
    && Boolean(dataset)
    && Boolean(datasetId)
    && !datasetLoading
    && !datasetError
    && !controlsLocked;

  const historyChoices = useMemo(() => {
    return mergeBatchRows(
      historyRows,
      currentSnapshot ? [currentSnapshot] : [],
    );
  }, [currentSnapshot, historyRows]);

  const latestCaseResults = useMemo(() => {
    const latest = new Map<string, LatestCaseResult>();
    const rows = mergeBatchRows(
      historyRows,
      snapshot ? [snapshot] : [],
    );
    for (const batch of rows) {
      if (
        !batchMatches(
          batch,
          datasetId,
          algorithmClusterId,
          mode,
          scene,
        )
      ) {
        continue;
      }
      const itemsByCase = new Map<string, SceneConditionObservation[]>();
      for (const item of batch.items) {
        const items = itemsByCase.get(item.case_id) ?? [];
        items.push(item);
        itemsByCase.set(item.case_id, items);
      }
      for (const caseRecord of batch.cases) {
        const candidate: LatestCaseResult = {
          batch,
          caseRecord,
          items: itemsByCase.get(caseRecord.case_id) ?? [],
        };
        const current = latest.get(caseRecord.case_id);
        if (!current || latestCaseResultIsNewer(candidate, current)) {
          latest.set(caseRecord.case_id, candidate);
        }
      }
    }
    return latest;
  }, [
    algorithmClusterId,
    datasetId,
    historyRows,
    mode,
    scene,
    snapshot,
  ]);

  const observationsByCase = useMemo(() => {
    const grouped = new Map<string, SceneConditionObservation[]>();
    for (const result of latestCaseResults.values()) {
      grouped.set(
        result.caseRecord.case_id,
        [...result.items].sort(
          (left, right) => left.item_id.localeCompare(right.item_id),
        ),
      );
    }
    return grouped;
  }, [latestCaseResults]);

  const latestItems = useMemo(
    () =>
      [...latestCaseResults.values()].flatMap(
        (result) => result.items,
      ),
    [latestCaseResults],
  );

  function showSceneSubpage(next: SceneSubpageId) {
    setSceneSubpage(next);
    const hash = sceneSubpageHash(definition.route, next);
    if (window.location.hash !== hash) {
      window.history.pushState(null, "", hash);
    }
  }

  function clearResultForSelectionChange() {
    setSnapshot(null);
    setHistoryRows([]);
    setHistorySelection("");
    setLaunchingCaseId("");
    setCaseRunErrors({});
    setRunError("");
    setPollError("");
    setHistoryError("");
    setRestoreError("");
  }

  async function handleStart(caseId?: string) {
    if (
      !canStart
      || !datasetId
      || launchAbortRef.current
    ) {
      return;
    }
    const selectedCaseId = caseId?.trim() ?? "";
    const controller = new AbortController();
    launchAbortRef.current = controller;
    setLaunching(true);
    setLaunchingCaseId(selectedCaseId);
    setRunError("");
    setPollError("");
    setRestoreError("");
    if (selectedCaseId) {
      setCaseRunErrors((current) => {
        if (!(selectedCaseId in current)) return current;
        const next = { ...current };
        delete next[selectedCaseId];
        return next;
      });
    }
    try {
      const body = await requestJson(
        SCENE_CONDITION_BATCHES_URL,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            dataset_id: datasetId,
            algorithm_cluster_id: algorithmClusterId,
            mode,
            scene,
            ...(selectedCaseId ? { case_id: selectedCaseId } : {}),
          }),
          signal: controller.signal,
        },
        selectedCaseId
          ? `运行条件组 ${selectedCaseId}`
          : `创建场景${profile.sceneLabel}测试批次`,
      );
      const next = readSceneConditionBatch(body, {
        datasetId,
        algorithmClusterId,
        ...(selectedCaseId ? { caseId: selectedCaseId } : {}),
      });
      if (controller.signal.aborted) return;
      if (
        !batchMatches(
          next,
          datasetId,
          algorithmClusterId,
          mode,
          scene,
        )
      ) {
        throw new Error("新批次身份与当前页面选择不一致");
      }
      const nextScope = batchScope(next, counts);
      if (
        selectedCaseId
          ? (
              !nextScope.single
              || nextScope.selectedCaseId !== selectedCaseId
              || next.cases.length !== 1
              || next.cases[0]?.case_id !== selectedCaseId
            )
          : nextScope.single
      ) {
        throw new Error("新批次的执行范围与请求不一致");
      }
      setSnapshot(next);
      setHistorySelection(next.batch_id);
      setHistoryRows((current) => mergeBatchRows(current, [next]));
      storeBatchId(storageKey, next.batch_id);
      if (selectedCaseId) {
        setCaseRunErrors((current) => {
          if (!(selectedCaseId in current)) return current;
          const cleaned = { ...current };
          delete cleaned[selectedCaseId];
          return cleaned;
        });
      }
    } catch (reason) {
      if (!controller.signal.aborted) {
        const message = errorMessage(
          reason,
          selectedCaseId
            ? `条件组 ${selectedCaseId} 执行请求失败`
            : `场景${profile.sceneLabel}测试批次创建失败`,
        );
        if (selectedCaseId) {
          setCaseRunErrors((current) => ({
            ...current,
            [selectedCaseId]: message,
          }));
        }
        setRunError(message);
      }
    } finally {
      if (launchAbortRef.current === controller) {
        launchAbortRef.current = null;
        if (!controller.signal.aborted) {
          setLaunching(false);
          setLaunchingCaseId("");
        }
      }
    }
  }

  const currentScope = currentSnapshot
    ? batchScope(currentSnapshot, counts)
    : null;
  const progressText =
    currentSnapshot && currentScope
      ? currentScope.single
        ? `单组 ${currentSnapshot.progress.completed_cases}/${currentScope.caseCount} · ${currentSnapshot.progress.completed_items}/${currentScope.observationCount} 条 · 数据集 ${currentScope.datasetCaseCount} 组/${currentScope.datasetObservationCount} 条`
        : `${currentSnapshot.progress.completed_cases}/${currentScope.caseCount} 组 · ${currentSnapshot.progress.completed_items}/${currentScope.observationCount} 次测量`
      : "尚无批次";
  const currentTableLimitSummary = summarizeSceneConditionTableLimits(
    currentSnapshot?.items ?? [],
  );
  const currentTableLimitUnresolved =
    sceneConditionLimitUnresolvedCount(currentTableLimitSummary);
  const latestSettledCases = [...latestCaseResults.values()].filter(
    (result) =>
      result.caseRecord.status === "completed"
      || result.caseRecord.status === "failed",
  ).length;
  const latestSettledItems = latestItems.filter(
    (item) => item.status !== "pending" && item.status !== "running",
  ).length;
  const latestCoverageText =
    `最新逐组结果 ${latestSettledCases}/${SCENE_CONDITION_BATCH_COUNT} 组 · `
    + `${latestSettledItems}/${SCENE_CONDITION_OBSERVATION_COUNT} 条`;
  const liveIssue =
    runError || pollError || restoreError || historyError;
  const activeExecute =
    pageVisible && sceneSubpage === "execute";
  const activeResults =
    pageVisible && sceneSubpage === "results";
  const activeWaveforms =
    pageVisible && sceneSubpage === "waveforms";
  const activeDescription =
    pageVisible && sceneSubpage === "description";
  const progressPercent = currentSnapshot && currentScope
    ? Math.min(
        100,
        (currentSnapshot.progress.completed_items
          / currentScope.observationCount) * 100,
      )
    : 0;
  const activeSingleCaseId =
    currentSnapshot && isBatchActive && snapshotCaseId
      ? snapshotCaseId
      : "";

  function renderCaseActions(
    previewCase: ScopedInputPreviewCase,
    context: ScopedCaseRenderContext,
  ) {
    const caseId = previewCase.case_id;
    const launchingThisCase = launchingCaseId === caseId;
    const runningThisCase = activeSingleCaseId === caseId;
    const contextMatches =
      context.datasetId === datasetId
      && context.mode === mode
      && context.scene === scene;
    const disabled = !canStart || controlsLocked || !contextMatches;
    let disabledReason = "";
    if (!contextMatches) {
      disabledReason = "条件组不属于当前测试集绑定";
    } else if (externalRunning) {
      disabledReason = "其他测试正在执行";
    } else if (launching || isBatchActive) {
      disabledReason = "已有测试正在执行";
    } else if (restoring) {
      disabledReason = "正在恢复测试结果";
    } else if (!dataset) {
      disabledReason = "测试集尚未就绪";
    }
    return (
      <button
        className="button secondary sdv-run-case"
        type="button"
        data-testid="run-single-case"
        data-case-id={caseId}
        data-dataset-id={context.datasetId}
        aria-label={`运行本组 ${caseId}`}
        aria-busy={launchingThisCase || runningThisCase}
        disabled={disabled}
        title={disabledReason || `只运行条件组 ${caseId}`}
        onClick={() => void handleStart(caseId)}
      >
        {launchingThisCase || runningThisCase ? (
          <Clock3 size={12} aria-hidden="true" />
        ) : (
          <Play size={12} fill="currentColor" aria-hidden="true" />
        )}
        {launchingThisCase
          ? "提交中…"
          : runningThisCase
            ? "运行中…"
            : "运行本组"}
      </button>
    );
  }

  function renderCaseResult(
    previewCase: ScopedInputPreviewCase,
    context: ScopedCaseRenderContext,
  ) {
    const caseId = previewCase.case_id;
    const candidate = latestCaseResults.get(caseId) ?? null;
    const result =
      candidate
      && caseIdentityMatches(
        candidate.batch,
        candidate.caseRecord,
        previewCase,
        context,
        algorithmClusterId,
      )
        ? candidate
        : null;
    const error =
      caseRunErrors[caseId]
      || (activeSingleCaseId === caseId ? pollError : "");
    return (
      <InlineCaseResult
        previewCase={previewCase}
        result={result}
        groups={groups}
        launching={launchingCaseId === caseId}
        error={error}
        datasetCaseCount={currentScope?.datasetCaseCount ?? counts.cases}
        farEndVisual={context.farEndVisual ?? null}
        onOpenWaveforms={() => showSceneSubpage("waveforms")}
        onToggleOutputPlayback={toggleOutputPlayback}
        outputPlaybackKey={outputPlaybackKey}
        mode={mode}
      />
    );
  }

  return (
    <section
      className="basic-uplink-test scoped-acoustic-test scene-condition-test"
      aria-labelledby={titleId}
    >
      <header className="basic-uplink-header scoped-acoustic-header">
        <div className="basic-uplink-heading">
          <span className="basic-uplink-eyebrow">
            {counts.cases}-CONDITION {profile.scene.toUpperCase()} · {mode.toUpperCase()}
          </span>
          <h2 id={titleId}>{definition.title} · {counts.cases} 组条件测试</h2>
          <p>
            近端单讲、远端单讲、持续双讲、单讲 / 双讲切换、CNG 噪声观测各 20 组。
            每组独立执行 VCP；按通话状态选择可用指标和补充诊断。
          </p>
        </div>
        <div
          className="basic-uplink-badges scoped-acoustic-badges"
          aria-label="场景测试口径"
        >
          <span>{SCENE_CONDITION_BATCH_COUNT} 组</span>
          <span>{SCENE_CONDITION_OBSERVATION_COUNT} 次测量</span>
          <span>{mode}</span>
          <span>{SCENE_CONDITION_CONTRACT_ITEM_COUNT} 个合同测项</span>
          <span>工程代理</span>
        </div>
      </header>

      <nav
        className="scoped-acoustic-subpage-nav"
        aria-label={`${definition.title}场景测试子页面`}
      >
        <h3 className="sr-only">{definition.title}子页面导航</h3>
        {SCENE_SUBPAGES.map((subpage) => (
          <button
            key={subpage.id}
            className="scoped-acoustic-subpage-tab"
            type="button"
            aria-current={
              sceneSubpage === subpage.id ? "page" : undefined
            }
            title={subpage.hint}
            onClick={() => showSceneSubpage(subpage.id)}
          >
            {subpage.label}
          </button>
        ))}
      </nav>

      <div className="scoped-acoustic-subpage-status" role="status">
        <span className="scoped-acoustic-subpage-status-meta">
          <strong>当前子页面</strong>
          {SCENE_SUBPAGE_LABELS[sceneSubpage]}
        </span>
        <span className="scoped-acoustic-subpage-status-meta">
          <strong>进度</strong>
          {progressText}
        </span>
        <span className="scoped-acoustic-subpage-status-meta">
          <strong>最新覆盖</strong>
          {latestCoverageText}
        </span>
        <span className="scoped-acoustic-subpage-status-note">
          {modeSupported
            ? "批次在后台轮询，切换子页面不会中断。"
            : "当前模式没有场景执行入口。"}
        </span>
        {liveIssue ? (
          <a
            className="scoped-acoustic-subpage-status-issue"
            href={sceneSubpageHash(definition.route, "results")}
            onClick={(event) => {
              event.preventDefault();
              showSceneSubpage("results");
            }}
          >
            查看状态与错误
          </a>
        ) : null}
      </div>

      <ActiveSubpage
        active={activeExecute}
        label={`执行场景${profile.sceneLabel}测试`}
      >
        <div className="basic-uplink-controls scoped-acoustic-controls">
          <label htmlFor={modeSelectId}>
            <span>业务模式</span>
            <select
              id={modeSelectId}
              value={mode}
              disabled={controlsLocked}
              onChange={(event) => {
                clearResultForSelectionChange();
                onModeChange(event.target.value as FiveSignalMode);
              }}
            >
              {FIVE_SIGNAL_MODES.map((candidate) => (
                <option key={candidate} value={candidate}>
                  {modeLabel(candidate)}
                </option>
              ))}
            </select>
            <small>选项为业务参考率；当前模式测试输入/输出为 {(profile.sampleRateHz ?? 16_000) / 1000} kHz。</small>
          </label>

          <label htmlFor={clusterSelectId}>
            <span>VCP 算法簇（{modeRateHz / 1000} kHz 可选 {availableAlgorithmCount} 项）</span>
            <select
              id={clusterSelectId}
              value={algorithmClusterId}
              disabled={controlsLocked}
              onChange={(event) => {
                clearResultForSelectionChange();
                onAlgorithmClusterChange(event.target.value);
              }}
            >
              {!algorithmIsSupported ? (
                <option value={algorithmClusterId}>
                  {algorithmClusterId
                    ? `${algorithmClusterId}（当前 ${modeRateHz / 1000} kHz 不可执行）`
                    : "未选择算法簇"}
                </option>
              ) : null}
              {availableAlgorithmGroups.map((group) => (
                <optgroup key={group.id} label={group.label}>
                  {group.clusters.map((cluster) => (
                    <option key={cluster.id} value={cluster.id}>
                      {cluster.label} · {cluster.id}
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
            <small>
              {algorithmIsSupported
                ? `${modeRateHz / 1000} kHz 可选 ${availableAlgorithmCount} 项 · ${algorithmLabel(algorithmClusterId)}`
                : `当前算法簇不支持 ${modeRateHz / 1000} kHz（${mode}），请在上方选择对应采样率的 VCP 算法簇。`}
            </small>
          </label>

          <AlgorithmClusterSummary cluster={selectedAlgorithmCluster} />
        </div>

        {!modeSupported ? (
          <div className="sct-mode-support" role="status">
            <AlertCircle size={18} aria-hidden="true" />
            <div>
              <strong>当前业务模式不可执行</strong>
              <p>
                请选择已支持的 NB、WB 或 SWB 业务模式。
              </p>
            </div>
            <button
              className="button secondary"
              type="button"
              disabled={controlsLocked}
              onClick={() => {
                clearResultForSelectionChange();
                onModeChange("WB-wired");
              }}
            >
              切回默认模式
            </button>
          </div>
        ) : null}

        {modeSupported && datasetError ? (
          <p className="basic-uplink-inline-error" role="alert">
            {datasetError}；测试集元数据无效时禁止执行。
          </p>
        ) : null}
        {modeSupported && !algorithmIsSupported ? (
          <p className="basic-uplink-inline-error" role="alert">
            当前算法簇不是 VCP 预设，禁止执行场景测试。
          </p>
        ) : null}

        {modeSupported && activeExecute ? (
          <ScopedDatasetVisualization
            source="scene-condition"
            dataset={dataset}
            mode={mode}
            scene={scene}
            expectedItemCount={SCENE_CONDITION_CONTRACT_ITEM_COUNT}
            active={pageVisible}
            loading={datasetLoading}
            refreshToken={datasetRefresh}
            refreshDisabled={controlsLocked || datasetLoading}
            onRefresh={() =>
              setDatasetRefresh((current) => current + 1)
            }
            onOpenWaveforms={() => showSceneSubpage("waveforms")}
            renderCaseActions={renderCaseActions}
            renderCaseResult={renderCaseResult}
          />
        ) : null}

        <div className="basic-uplink-run">
          <div>
            <strong>
              {modeLabel(mode)} · {algorithmLabel(algorithmClusterId)}
            </strong>
            <p>
              逐组执行 VCP → 分工况计算 → 保存 {counts.cases} 组输出、
              {counts.observations} 条合同关联记录与单讲 / 双讲诊断。
            </p>
          </div>
          <button
            className="button primary"
            type="button"
            disabled={!canStart}
            onClick={() => void handleStart()}
          >
            <Play size={15} fill="currentColor" aria-hidden="true" />
            执行{counts.cases}组测试
          </button>
        </div>

        {currentSnapshot ? (
          <div className="basic-uplink-progress sct-progress">
            <div>
              <strong>{progressText}</strong>
              <span>
                {SCOPED_BATCH_LABELS[currentSnapshot.status]}
                {" · "}
                {INTEGER_FORMAT.format(currentSnapshot.summary.computed)}
                /{currentScope?.observationCount ?? 0} 个有效结果
              </span>
            </div>
            <progress
              max={currentScope?.observationCount ?? 1}
              value={currentSnapshot.progress.completed_items}
              aria-label="场景测试完成进度"
            >
              {scopedValue(progressPercent)}%
            </progress>
          </div>
        ) : null}

        {runError ? (
          <div className="basic-uplink-alert" role="alert">
            <AlertCircle size={18} aria-hidden="true" />
            <div>
              <strong>场景测试创建失败</strong>
              <p>{runError}</p>
              <p>失败请求不会改写最近成功快照，也不会生成通过值。</p>
            </div>
          </div>
        ) : null}
      </ActiveSubpage>

      <ActiveSubpage active={activeResults} label="场景测试结果">
        <div className="basic-uplink-history">
          <label htmlFor={historySelectId}>
            <History size={15} aria-hidden="true" />
           场景测试历史
          </label>
          <select
            id={historySelectId}
            value={historySelection || currentSnapshot?.batch_id || ""}
            disabled={
              controlsLocked
              || historyLoading
              || historyChoices.length === 0
            }
            onChange={(event) => {
              restoreTargetRef.current = event.target.value;
              setHistorySelection(event.target.value);
              setRestoreNonce((current) => current + 1);
            }}
          >
            {historyChoices.length === 0 ? (
              <option value="">
                {historyLoading ? "正在读取新历史…" : "暂无场景历史"}
              </option>
            ) : null}
            {historyChoices.map((row) => {
              const scope = batchScope(row, counts);
              return (
                <option key={row.batch_id} value={row.batch_id}>
                  {formatDate(row.created_at)} · {row.algorithm_cluster_id} ·{" "}
                  {scope.single
                    ? `单组 ${scope.selectedCaseId.slice(0, 12)}`
                    : `整批 ${scope.caseCount} 组`}{" "}
                  · {SCOPED_BATCH_LABELS[row.status]} ·{" "}
                  {row.batch_id.slice(0, 8)}
                </option>
              );
            })}
          </select>
          <button
            className="button secondary"
            type="button"
            disabled={controlsLocked || historyLoading}
            onClick={() => setRestoreNonce((current) => current + 1)}
          >
            <RefreshCw size={15} aria-hidden="true" />
            刷新测试历史
          </button>
          {historyLoading || restoring ? (
            <p className="basic-uplink-history-note" role="status">
              正在读取测试历史并核对结果来源。
            </p>
          ) : null}
          {historyError ? (
            <p className="basic-uplink-inline-error" role="alert">
              {historyError}
            </p>
          ) : null}
          {restoreError ? (
            <p className="basic-uplink-inline-warning" role="alert">
              {restoreError}
            </p>
          ) : null}
        </div>

        {pollError ? (
          <div className="basic-uplink-alert" role="alert">
            <AlertCircle size={18} aria-hidden="true" />
            <div>
              <strong>后台轮询暂时失败</strong>
              <p>{pollError}</p>
              <button
                className="button secondary"
                type="button"
                onClick={() => setPollRetry((current) => current + 1)}
              >
                <RefreshCw size={15} aria-hidden="true" />
                立即重试
              </button>
            </div>
          </div>
        ) : null}
        {runError ? (
          <div className="basic-uplink-alert" role="alert">
            <AlertCircle size={18} aria-hidden="true" />
            <div>
              <strong>最近一次创建批次失败</strong>
              <p>{runError}</p>
              <p>当前仍保留上次成功读取的测试结果。</p>
            </div>
          </div>
        ) : null}

        {!currentSnapshot ? (
          <div className="basic-uplink-empty">
            <ListChecks size={24} aria-hidden="true" />
            <div>
              <h3>尚无当前配置的场景测试结果</h3>
              <p>
                执行场景测试后，可在这里查看每组测量值和处理后音频。
              </p>
            </div>
          </div>
        ) : (
          <>
            <section
              className="basic-uplink-result"
              aria-label="场景批次状态"
            >
              <header className="basic-uplink-result-header">
                <div>
                  <span className="basic-uplink-eyebrow">
                    {currentScope?.single ? "单组测试结果" : "场景测试结果"}
                  </span>
                  <h3>{SCOPED_BATCH_LABELS[currentSnapshot.status]}</h3>
                  <p>
                    {currentSnapshot.dataset_name}
                    {" · "}
                    {algorithmLabel(currentSnapshot.algorithm_cluster_id)}
                    {" · "}
                    {formatDate(currentSnapshot.updated_at)}
                    {currentScope?.single
                      ? ` · 仅运行 ${currentScope.selectedCaseId}`
                      : ""}
                  </p>
                </div>
                <span
                  className={`basic-uplink-run-status is-${currentSnapshot.status}`}
                >
                  {currentSnapshot.status === "queued"
                    || currentSnapshot.status === "running" ? (
                      <Clock3 size={14} aria-hidden="true" />
                    ) : null}
                  {progressText}
                </span>
              </header>

              <div className="sct-latest-coverage" role="status">
                <strong>
                  {currentScope?.single
                    ? `本次单组 ${currentSnapshot.progress.completed_cases}/${currentScope.caseCount}，全集 ${currentScope.datasetCaseCount} 组`
                    : `本次整批 ${currentSnapshot.progress.completed_cases}/${currentScope?.caseCount ?? counts.cases} 组`}
                </strong>
                <span>{latestCoverageText}</span>
              </div>

              <div className="basic-uplink-progress">
                <div>
                  <strong>{progressText}</strong>
                  <span>
                    {INTEGER_FORMAT.format(currentSnapshot.summary.computed)}
                    /{currentScope?.observationCount ?? 0} 个有效结果
                  </span>
                </div>
                <progress
                  max={currentScope?.observationCount ?? 1}
                  value={currentSnapshot.progress.completed_items}
                  aria-label="场景测试结果进度"
                >
                  {scopedValue(progressPercent)}%
                </progress>
              </div>

              <div className="basic-uplink-result-boundary">
                <ShieldAlert size={18} aria-hidden="true" />
                <div>
                  <strong>工程代理结果 · 正式 ITU-T 合规未测</strong>
                  <p>
                    5 个合同测项按适用工况计算。远端单讲和切换段不提供 G/N/S-MOS，
                    对应记录标为不可测；回声和双讲补充诊断单独展示。
                  </p>
                </div>
              </div>

              <dl className="basic-uplink-summary">
                {(
                  [
                    ["总观测", currentSnapshot.summary.total],
                    ["有限数值", currentSnapshot.summary.computed],
                    ["工程通过", currentSnapshot.summary.pass],
                    ["工程失败", currentSnapshot.summary.fail],
                    ["需复核", currentSnapshot.summary.review_required],
                    ["不可测", currentSnapshot.summary.not_measured],
                    ["无效", currentSnapshot.summary.invalid],
                    ["表内达到", currentTableLimitSummary.meets],
                    ["表内低于", currentTableLimitSummary.below],
                    ["表内不可对照", currentTableLimitUnresolved],
                  ] as const
                ).map(([label, value], index) => (
                  <div
                    key={label}
                    className={index === 1 ? "is-computed" : ""}
                  >
                    <dt>{label}</dt>
                    <dd>{INTEGER_FORMAT.format(value)}</dd>
                  </div>
                ))}
              </dl>
              <p className="sct-table-limit-note">
                表内达到/低于只比较 DNSMOS P.835 原始分数与 NV200
                {mode} 客户表 E 列数值；不改变任何服务端合同状态，
                也不代表正式 ITU-T 合规通过。
              </p>

              <div className="basic-uplink-identity">
                <span>
                  批次 <code>{currentSnapshot.batch_id}</code>
                </span>
                <span>
                  manifest{" "}
                  <code>{currentSnapshot.dataset_manifest_sha256}</code>
                </span>
                <span>
                  catalog <code>{currentSnapshot.catalog_sha256}</code>
                </span>
                <span>
                  CLI <code>{currentSnapshot.cli_sha256}</code>
                </span>
              </div>
            </section>

            <section className="sct-contract-summary">
              <header>
                <span className="basic-uplink-eyebrow">
                  CONTRACT DIAGNOSTICS
                </span>
                <h3>最新逐组合同汇总</h3>
                <p>
                  {latestCoverageText}。每类按 created_at 取同一数据集内
                  各条件组的最新同身份结果；单组运行不会抹去其他组已有结果。
                </p>
              </header>
              <div className="sct-contract-grid">
                {groups.map((group) => {
                  const stats = contractStats(
                    latestItems,
                    group.itemId,
                  );
                  return (
                    <article key={group.itemId}>
                      <header>
                        <div>
                          <code>{group.itemId}</code>
                          <strong>{group.label}</strong>
                          <span>{group.detail}</span>
                        </div>
                        <span className="sct-finite-count">
                          {stats.count}/{group.family.startsWith("cng_") ? counts.cng : counts.speech} 有限值
                        </span>
                      </header>
                      <dl>
                        <div>
                          <dt>均值</dt>
                          <dd>{scopedValue(stats.mean)}</dd>
                        </div>
                        <div>
                          <dt>最小</dt>
                          <dd>{scopedValue(stats.minimum)}</dd>
                        </div>
                        <div>
                          <dt>最大</dt>
                          <dd>{scopedValue(stats.maximum)}</dd>
                        </div>
                      </dl>
                      <p>
                        单位 {stats.unit || "未声明"} ·{" "}
                        {stats.lowerLimit === null
                          ? "当前工程代理无适用 NV200 数值下限"
                          : `NV200 ${group.itemId.slice(-3)} 下限 ${
                              scopedValue(stats.lowerLimit)
                            }，仅作工程数值对照`}
                      </p>
                      <div className="sct-status-counts">
                        <span>通过 {stats.pass}</span>
                        <span>失败 {stats.fail}</span>
                        <span>需复核 {stats.reviewRequired}</span>
                        <span>不可测 {stats.notMeasured}</span>
                        <span>无效 {stats.invalid}</span>
                      </div>
                      <div className="sct-status-counts sct-table-limit-counts">
                        {stats.lowerLimit === null ? (
                          <span>表内下限不适用 {stats.tableLimit.notApplicable}</span>
                        ) : (
                          <>
                            <span>表内达到 {stats.tableLimit.meets}</span>
                            <span>表内低于 {stats.tableLimit.below}</span>
                            <span>
                              表内不可对照{" "}
                              {sceneConditionLimitUnresolvedCount(
                                stats.tableLimit,
                              )}
                            </span>
                          </>
                        )}
                      </div>
                    </article>
                  );
                })}
              </div>
            </section>

            <section className="sct-case-results">
              <header>
                <span className="basic-uplink-eyebrow">
                  CASE PROCESSING
                </span>
                <h3>最新逐组处理结果</h3>
                <p>
                  {latestCoverageText}。{" "}
                  TX 仅在条件组完成、报告与四路输入哈希、逐观测证据和可视化来源全部一致时开放。
                </p>
              </header>
              <div className="sct-case-grid">
                {[...latestCaseResults.values()]
                  .sort((left, right) =>
                    left.caseRecord.case_id.localeCompare(
                      right.caseRecord.case_id,
                    )
                  )
                  .map(({ batch, caseRecord, items: caseItems }) => {
                    const previewCase = previewCaseFromBatch(
                      batch,
                      caseRecord,
                    );
                    const tx = sceneConditionCaseTxEvidence(
                      previewCase,
                      batch,
                      batch.dataset_manifest_sha256,
                    );
                    return (
                      <article
                        key={caseRecord.case_id}
                        className={`sct-case-card is-${caseRecord.status}`}
                      >
                        <header>
                          <code>{caseRecord.case_id}</code>
                          <span>{caseRecord.status}</span>
                        </header>
                        <p className="sct-case-source">
                          最新批次 <code>{batch.batch_id.slice(0, 8)}</code>
                          {" · "}
                          {formatDate(batch.created_at)}
                        </p>
                        <p>{conversationModeOf(caseRecord)
                          ? CONVERSATION_MODE_LABELS[conversationModeOf(caseRecord)!]
                          : "历史条件组"}</p>
                        <p>
                          {caseItems.length} 次测量 ·{" "}
                          {new Set(caseItems.map((item) => item.family))
                            .size}{" "}
                          个指标族
                        </p>
                        <dl>
                          <div>
                            <dt>报告</dt>
                            <dd>{caseRecord.report_id ?? "待生成"}</dd>
                          </div>
                          <div>
                            <dt>TX 哈希</dt>
                            <dd>
                              {caseRecord.output_sha256?.tx ?? "待生成"}
                            </dd>
                          </div>
                        </dl>
                        <NsQualityMetrics
                          batchId={batch.batch_id}
                          caseId={caseRecord.case_id}
                          outputSha={caseRecord.output_sha256?.tx}
                          completed={caseRecord.status === "completed"}
                        />
                        <ConversationDiagnostics value={caseRecord.conversation_diagnostics} />
                        {tx.ready ? (
                          <div className="sct-case-output-actions">
                            <a
                              href={tx.url}
                              download
                              aria-label={`下载条件组 ${caseRecord.case_id} TX`}
                            >
                              <Download size={14} aria-hidden="true" />
                              下载 TX
                            </a>
                            <OutputAudioPreview
                              caseId={caseRecord.case_id}
                              label="TX 输出音频"
                              url={tx.url}
                              playing={outputPlaybackKey === `${batch.batch_id}:${caseRecord.case_id}:tx`}
                              onToggle={() => toggleOutputPlayback(`${batch.batch_id}:${caseRecord.case_id}:tx`)}
                            />
                          </div>
                        ) : (
                          <p className="sct-case-unavailable">
                            {tx.reason}
                          </p>
                        )}
                        {caseRecord.error ? (
                          <p className="sct-case-error" role="alert">
                            {caseRecord.error}
                          </p>
                        ) : null}
                      </article>
                    );
                  })}
              </div>
            </section>

            <details className="sct-observation-details">
              <summary>
                <span>{latestItems.length} 条最新合同关联明细</span>
                <small>
                  按条件组与合同测项分组；默认折叠
                </small>
              </summary>
              <div className="table-scroll">
                <table className="audio-table basic-uplink-items-table sct-observation-table">
                  <caption className="sr-only">
                    {latestSettledCases} 个最新已收口条件组的{" "}
                    {latestItems.length} 条合同关联结果
                  </caption>
                  <thead>
                    <tr>
                      <th scope="col">条件组</th>
                      <th scope="col">观测 / 测项</th>
                      <th scope="col">状态</th>
                      <th scope="col">工程值</th>
                      <th scope="col">指标与原因</th>
                      <th scope="col">证据</th>
                    </tr>
                  </thead>
                  {[...observationsByCase.entries()].map(
                    ([caseId, rows]) => (
                      <tbody key={caseId}>
                        {rows.map((item, index) => (
                          <tr key={item.observation_id}>
                            {index === 0 ? (
                              <th
                                scope="rowgroup"
                                rowSpan={rows.length}
                                className="sct-case-cell"
                              >
                                <code>{caseId}</code>
                                <span>
                                  {latestCaseResults.get(caseId)
                                    ?.caseRecord.status ?? "unknown"}
                                </span>
                              </th>
                            ) : null}
                            <td>
                              <div className="basic-uplink-item-name">
                                <code>{item.observation_id}</code>
                                <strong>{item.test_name}</strong>
                                <span>
                                  {item.item_id} · {item.family}
                                </span>
                              </div>
                            </td>
                            <td>
                              <div className="sct-observation-statuses">
                                <TableLimitBadge
                                  result={evaluateSceneConditionTableLowerLimit(
                                    item.item_id,
                                    item.value,
                                    item.status,
                                  )}
                                />
                                <ItemStatusBadge status={item.status} />
                              </div>
                            </td>
                            <td>
                              <div className="scoped-acoustic-engineering-value">
                                <strong>{scopedValue(item.value)}</strong>
                                <span>
                                  {scopedMetricUnit(item.metric)
                                    || "无量纲/未声明"}
                                </span>
                              </div>
                            </td>
                            <td>
                              <div className="sct-observation-copy">
                                <code>{item.metric ?? "计算指标未返回"}</code>
                                <span>{item.reason || "未返回原因"}</span>
                              </div>
                            </td>
                            <td>
                              <div className="basic-uplink-evidence">
                                <code>
                                  report{" "}
                                  {item.evidence_binding.report_id
                                    ?? "待生成"}
                                </code>
                                <span>
                                  TX{" "}
                                  {item.evidence_binding.output_sha256?.tx
                                    ?? "待生成"}
                                </span>
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    ),
                  )}
                </table>
              </div>
            </details>
          </>
        )}
      </ActiveSubpage>

      <ActiveSubpage active={activeWaveforms} label="场景波形与梅尔频谱">
        {modeSupported && dataset && datasetId ? (
          <ScopedAcousticWaveforms
            active={activeWaveforms}
            sceneConditionBatch={currentSnapshot}
            definition={waveformDefinition}
            datasetId={datasetId}
            mode={mode}
            algorithmClusterId={algorithmClusterId}
            snapshot={null}
          />
        ) : (
          <div className="basic-uplink-empty">
            <AlertCircle size={24} aria-hidden="true" />
            <div>
              <h3>波形页暂不可用</h3>
              <p>
                {modeSupported
                  ? datasetError || "正在等待场景测试集元数据。"
                  : "当前业务模式没有场景波形入口。"}
              </p>
            </div>
          </div>
        )}
      </ActiveSubpage>

      <ActiveSubpage active={activeDescription} label="场景测试说明">
        <div className="sct-description">
          <section>
            <span className="basic-uplink-eyebrow">SCOPE</span>
            <h3>观测数量</h3>
            <p>
              4 种含语音工况各 20 组，关联 3 个质量代理；CNG 20 组关联 2 个噪声代理，
              合计 {counts.observations} 条记录。不适用的质量代理保留为空，
              远端单讲、双讲和切换同时生成专用工程诊断。
            </p>
          </section>
          <section>
            <span className="basic-uplink-eyebrow">CONTRACT</span>
            <h3>合同口径</h3>
            <p>
              合同测项是 {profile.contractItemIds[0]?.split(":")[1]}–
              {profile.contractItemIds[4]?.split(":")[1]} 共 5 个；
              {counts.observations} 是条件组与指标组合后的记录数，
              合同项仍为 5 个；增加组合不增加正式认证项目。
            </p>
          </section>
          <section>
            <span className="basic-uplink-eyebrow">EVIDENCE</span>
            <h3>结果范围</h3>
            <p>
              每组样本分别保存四路输入、VCP 处理后音频和测量值，
              使用输入与输出哈希核对来源。工程测量不代表正式合规认证。
            </p>
          </section>
        </div>
      </ActiveSubpage>
    </section>
  );
}
