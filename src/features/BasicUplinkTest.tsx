import {
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type FormEvent,
} from "react";
import {
  AlertCircle,
  CheckCircle2,
  Clock3,
  Download,
  FileArchive,
  History,
  ListChecks,
  MinusCircle,
  Play,
  RefreshCw,
  ShieldAlert,
  Upload,
  XCircle,
} from "lucide-react";
import { chainPresets } from "./enhancement-chain-model";
import {
  FIVE_SIGNAL_MODES,
  type FiveSignalMode,
} from "./five-signal-flow-model";
import "./basic-uplink-test.css";
import BasicUplinkTree from "./BasicUplinkTree";

const DATASETS_URL = "/api/alignment/acceptance/datasets";
const CATALOG_URL = "/api/alignment/acceptance/basic-uplink/catalog";
const BASIC_BATCHES_URL =
  "/api/alignment/acceptance/basic-uplink/batches";
const BATCHES_URL = "/api/alignment/acceptance/batches";
const RUN_STORAGE_KEY =
  "itu.alignment.basic-uplink.latest-batch-id.v1";
const POLL_INTERVAL_MS = 2000;
const BASIC_ITEM_COUNT = 8;
const MAX_ARCHIVE_BYTES = 256 * 1024 * 1024;

const MODE_LABELS: Record<FiveSignalMode, string> = {
  "NB-wired": "窄带 · 有线",
  "NB-wireless": "窄带 · 无线",
  "NB-bt": "窄带 · 蓝牙",
  "WB-wired": "宽带 · 有线",
  "WB-wireless": "宽带 · 无线",
  "WB-bt": "宽带 · 蓝牙",
  "SWB-wired": "超宽带 · 有线",
  "SWB-wireless": "超宽带 · 无线",
};

const BATCH_STATUSES = [
  "queued",
  "running",
  "completed",
  "failed",
  "cancelled",
] as const;
const ITEM_STATUSES = [
  "pending",
  "running",
  "pass",
  "fail",
  "review_required",
  "not_measured",
  "invalid",
] as const;

type BatchStatus = (typeof BATCH_STATUSES)[number];
type ItemStatus = (typeof ITEM_STATUSES)[number];
type Limit = string | number | null;

export type DatasetSummary = {
  dataset_id: string;
  name: string;
  case_count: number;
  item_count: number;
  created_at: string;
};

type CatalogItem = {
  id: string;
  source_cell: string;
  test_name: string;
  family: string;
  unit: string;
  lower_limit: Limit;
  upper_limit: Limit;
};

export type CatalogResponse = {
  test_scope: "basic_uplink";
  mode: FiveSignalMode;
  item_count: 8;
  items: CatalogItem[];
};

type EvidenceBinding = {
  mode: string;
  direction: string;
  scene: string;
  volume: string | number | null;
  window_ms: number | null;
  input_level_db: number | null;
  case_id: string;
  report_id: string | null;
  batch_id: string;
  dataset_id: string;
};

type BasicBatchItem = {
  item_id: string;
  id?: string;
  mode: string;
  source_cell?: string;
  test_name?: string;
  family?: string;
  unit?: string;
  lower_limit?: Limit;
  upper_limit?: Limit;
  status: ItemStatus;
  value: number | null;
  metric?: string | null;
  reason?: string;
  evidence_binding?: EvidenceBinding;
};

type BasicBatchCase = {
  case_id: string;
  report_id: string;
  status: string;
  conditions?: Record<string, unknown>;
  error?: string;
  assessment_sha256?: string;
};

export type BasicBatchSnapshot = {
  batch_id: string;
  dataset_id: string;
  dataset_name: string;
  algorithm_cluster_id: string;
  status: BatchStatus;
  created_at: string;
  test_scope: "basic_uplink";
  selected_mode: FiveSignalMode;
  standard_compliance: false;
  progress: {
    completed_cases: number;
    total_cases: number;
    completed_items: number;
    total_items: number;
  };
  summary: {
    total: number;
    computed: number;
    pass: number;
    fail: number;
    review_required: number;
    not_measured: number;
    invalid: number;
  };
  items: BasicBatchItem[];
  cases?: BasicBatchCase[];
  error?: string | null;
};

type BasicBatchSummary = {
  batch_id: string;
  dataset_id: string;
  dataset_name: string;
  algorithm_cluster_id: string;
  status: BatchStatus;
  created_at: string;
  test_scope: "basic_uplink";
  selected_mode: FiveSignalMode;
  progress: BasicBatchSnapshot["progress"];
  summary: BasicBatchSnapshot["summary"];
};

type BasicBatchSnapshotShape = Omit<
  BasicBatchSnapshot,
  "test_scope" | "selected_mode" | "standard_compliance"
> & {
  test_scope: string;
  selected_mode: string;
  standard_compliance: boolean;
};

type ExpectedBatchIdentity = {
  batchId?: string;
  datasetId?: string;
  algorithmClusterId?: string;
  mode?: FiveSignalMode;
};

type Props = {
  algorithmClusterId: string;
  mode: FiveSignalMode;
  onModeChange: (mode: FiveSignalMode) => void;
  onAlgorithmClusterChange: (id: string) => void;
  externalRunning?: boolean;
  onBusyChange?: (busy: boolean) => void;
};

const BATCH_STATUS_LABEL: Record<BatchStatus, string> = {
  queued: "批次排队中",
  running: "批次执行中",
  completed: "批次已完成",
  failed: "批次执行失败",
  cancelled: "批次已取消",
};

const ITEM_STATUS_LABEL: Record<ItemStatus, string> = {
  pending: "等待计算",
  running: "计算中",
  pass: "工程通过",
  fail: "工程失败",
  review_required: "需复核",
  not_measured: "不可测",
  invalid: "无效",
};

const VCP_PRESETS = chainPresets.filter((preset) =>
  preset.profile.startsWith("vcp"),
);
const MODE_SET = new Set<string>(FIVE_SIGNAL_MODES);
const BATCH_STATUS_SET = new Set<string>(BATCH_STATUSES);
const ITEM_STATUS_SET = new Set<string>(ITEM_STATUSES);
const INTEGER_FORMAT = new Intl.NumberFormat("zh-CN");
const VALUE_FORMAT = new Intl.NumberFormat("zh-CN", {
  maximumFractionDigits: 3,
});
const DATE_FORMAT = new Intl.DateTimeFormat("zh-CN", {
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
});

class ApiResponseError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiResponseError";
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isString(value: unknown): value is string {
  return typeof value === "string";
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isCount(value: unknown): value is number {
  return (
    isFiniteNumber(value) &&
    Number.isInteger(value) &&
    value >= 0
  );
}

function isNullableNumber(value: unknown): value is number | null {
  return value === null || isFiniteNumber(value);
}

function isNullableStringOrNumber(
  value: unknown,
): value is string | number | null {
  return value === null || isString(value) || isFiniteNumber(value);
}

function isMode(value: unknown): value is FiveSignalMode {
  return isString(value) && MODE_SET.has(value);
}

function isLimit(value: unknown): value is Limit {
  return (
    value === null ||
    isString(value) ||
    isFiniteNumber(value)
  );
}

function isDatasetSummary(value: unknown): value is DatasetSummary {
  return (
    isRecord(value) &&
    isString(value.dataset_id) &&
    isString(value.name) &&
    isCount(value.case_count) &&
    isCount(value.item_count) &&
    isString(value.created_at)
  );
}

function isDatasetList(value: unknown): value is DatasetSummary[] {
  return Array.isArray(value) && value.every(isDatasetSummary);
}

function isCatalogItem(value: unknown): value is CatalogItem {
  return (
    isRecord(value) &&
    isString(value.id) &&
    isString(value.source_cell) &&
    isString(value.test_name) &&
    isString(value.family) &&
    isString(value.unit) &&
    isLimit(value.lower_limit) &&
    isLimit(value.upper_limit)
  );
}

function isCatalogResponse(value: unknown): value is CatalogResponse {
  return (
    isRecord(value) &&
    value.test_scope === "basic_uplink" &&
    isMode(value.mode) &&
    value.item_count === BASIC_ITEM_COUNT &&
    Array.isArray(value.items) &&
    value.items.length === BASIC_ITEM_COUNT &&
    value.items.every(isCatalogItem) &&
    new Set(value.items.map((item) => item.id)).size ===
      BASIC_ITEM_COUNT
  );
}

function isEvidenceBinding(value: unknown): value is EvidenceBinding {
  return (
    isRecord(value) &&
    isString(value.mode) &&
    isString(value.direction) &&
    isString(value.scene) &&
    isNullableStringOrNumber(value.volume) &&
    isNullableNumber(value.window_ms) &&
    isNullableNumber(value.input_level_db) &&
    isString(value.case_id) &&
    (value.report_id === null || isString(value.report_id)) &&
    isString(value.batch_id) &&
    isString(value.dataset_id)
  );
}

function isBasicBatchItem(value: unknown): value is BasicBatchItem {
  if (
    !isRecord(value) ||
    !isString(value.item_id) ||
    !isString(value.mode) ||
    !ITEM_STATUS_SET.has(String(value.status)) ||
    !isNullableNumber(value.value)
  ) {
    return false;
  }
  if (value.id !== undefined && !isString(value.id)) return false;
  if (
    value.source_cell !== undefined &&
    !isString(value.source_cell)
  ) {
    return false;
  }
  if (
    value.test_name !== undefined &&
    !isString(value.test_name)
  ) {
    return false;
  }
  if (value.family !== undefined && !isString(value.family)) {
    return false;
  }
  if (value.unit !== undefined && !isString(value.unit)) {
    return false;
  }
  if (value.lower_limit !== undefined && !isLimit(value.lower_limit)) {
    return false;
  }
  if (value.upper_limit !== undefined && !isLimit(value.upper_limit)) {
    return false;
  }
  if (
    value.metric !== undefined &&
    value.metric !== null &&
    !isString(value.metric)
  ) {
    return false;
  }
  if (value.reason !== undefined && !isString(value.reason)) {
    return false;
  }
  return (
    value.evidence_binding === undefined ||
    isEvidenceBinding(value.evidence_binding)
  );
}

function isBasicBatchCase(value: unknown): value is BasicBatchCase {
  return (
    isRecord(value) &&
    isString(value.case_id) &&
    isString(value.report_id) &&
    isString(value.status) &&
    (value.conditions === undefined || isRecord(value.conditions)) &&
    (value.error === undefined || isString(value.error)) &&
    (value.assessment_sha256 === undefined ||
      isString(value.assessment_sha256))
  );
}

function isProgress(value: unknown): value is BasicBatchSnapshot["progress"] {
  return (
    isRecord(value) &&
    isCount(value.completed_cases) &&
    isCount(value.total_cases) &&
    isCount(value.completed_items) &&
    isCount(value.total_items)
  );
}

function isSummary(value: unknown): value is BasicBatchSnapshot["summary"] {
  return (
    isRecord(value) &&
    isCount(value.total) &&
    isCount(value.computed) &&
    isCount(value.pass) &&
    isCount(value.fail) &&
    isCount(value.review_required) &&
    isCount(value.not_measured) &&
    isCount(value.invalid)
  );
}

function isBasicBatchSnapshotShape(
  value: unknown,
): value is BasicBatchSnapshotShape {
  return (
    isRecord(value) &&
    isString(value.batch_id) &&
    isString(value.dataset_id) &&
    isString(value.dataset_name) &&
    isString(value.algorithm_cluster_id) &&
    isString(value.status) &&
    BATCH_STATUS_SET.has(value.status) &&
    isString(value.created_at) &&
    isString(value.test_scope) &&
    isString(value.selected_mode) &&
    typeof value.standard_compliance === "boolean" &&
    isProgress(value.progress) &&
    isSummary(value.summary) &&
    Array.isArray(value.items) &&
    value.items.every(isBasicBatchItem) &&
    (value.cases === undefined ||
      (Array.isArray(value.cases) &&
        value.cases.every(isBasicBatchCase))) &&
    (value.error === undefined ||
      value.error === null ||
      isString(value.error))
  );
}

function isBasicBatchSummary(
  value: unknown,
): value is BasicBatchSummary {
  return (
    isRecord(value) &&
    isString(value.batch_id) &&
    isString(value.dataset_id) &&
    isString(value.dataset_name) &&
    isString(value.algorithm_cluster_id) &&
    isString(value.status) &&
    BATCH_STATUS_SET.has(value.status) &&
    isString(value.created_at) &&
    value.test_scope === "basic_uplink" &&
    isMode(value.selected_mode) &&
    isProgress(value.progress) &&
    isSummary(value.summary) &&
    value.progress.total_items === BASIC_ITEM_COUNT &&
    value.summary.total === BASIC_ITEM_COUNT
  );
}

function isBasicBatchHistory(
  value: unknown,
): value is BasicBatchSummary[] {
  return Array.isArray(value) && value.every(isBasicBatchSummary);
}

function assertBasicCatalog(
  value: unknown,
  expectedMode: FiveSignalMode,
): CatalogResponse {
  if (!isCatalogResponse(value)) {
    throw new Error("基础上行目录返回结构不符合 8 项合同");
  }
  if (value.mode !== expectedMode) {
    throw new Error(
      `基础上行目录模式不一致：请求 ${expectedMode}，返回 ${value.mode}`,
    );
  }
  return value;
}

function assertBasicBatchSnapshot(
  value: unknown,
  operation: string,
  expected: ExpectedBatchIdentity = {},
): BasicBatchSnapshot {
  if (!isBasicBatchSnapshotShape(value)) {
    throw new Error(`${operation}返回结构不符合批次快照契约`);
  }
  if (value.test_scope !== "basic_uplink") {
    throw new Error(
      `${operation}拒绝：返回结果不属于基础上行测试范围`,
    );
  }
  if (!isMode(value.selected_mode)) {
    throw new Error(`${operation}拒绝：报告业务模式无效`);
  }
  if (value.standard_compliance !== false) {
    throw new Error(
      `${operation}拒绝：正式合规状态必须明确为未测`,
    );
  }
  if (expected.batchId && value.batch_id !== expected.batchId) {
    throw new Error(`${operation}返回的批次 ID 与请求不一致`);
  }
  if (expected.datasetId && value.dataset_id !== expected.datasetId) {
    throw new Error(`${operation}返回的数据集与请求不一致`);
  }
  if (
    expected.algorithmClusterId &&
    value.algorithm_cluster_id !== expected.algorithmClusterId
  ) {
    throw new Error(`${operation}返回的算法簇与请求不一致`);
  }
  if (expected.mode && value.selected_mode !== expected.mode) {
    throw new Error(`${operation}返回的模式与请求不一致`);
  }
  if (
    value.items.length !== BASIC_ITEM_COUNT ||
    value.progress.total_items !== BASIC_ITEM_COUNT ||
    value.summary.total !== BASIC_ITEM_COUNT
  ) {
    throw new Error(
      `${operation}拒绝：基础上行必须恰好包含 8 个逐项结果`,
    );
  }
  if (
    value.items.some((item) => item.mode !== value.selected_mode)
  ) {
    throw new Error(
      `${operation}拒绝：逐项业务模式与报告模式不一致`,
    );
  }
  if (
    new Set(value.items.map((item) => item.item_id)).size !==
    BASIC_ITEM_COUNT
  ) {
    throw new Error(`${operation}拒绝：8 个基础上行 item_id 必须唯一`);
  }
  return value as BasicBatchSnapshot;
}

function apiErrorDetail(body: unknown): string {
  if (!isRecord(body)) return "";
  const detail = body.detail;
  if (isString(detail)) return detail;
  if (!Array.isArray(detail)) return "";
  return detail
    .map((item) => {
      if (!isRecord(item) || !isString(item.msg)) return "";
      return item.msg;
    })
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
    const message = response.ok
      ? `${operation}返回了无法解析的 JSON`
      : `${operation}失败（HTTP ${response.status}）`;
    throw response.ok
      ? new Error(message)
      : new ApiResponseError(message, response.status);
  }
  if (!response.ok) {
    throw new ApiResponseError(
      apiErrorDetail(body) ||
        `${operation}失败（HTTP ${response.status}）`,
      response.status,
    );
  }
  return body;
}

async function fetchBasicBatch(
  batchId: string,
  signal: AbortSignal,
  operation: string,
  expected: ExpectedBatchIdentity = {},
): Promise<BasicBatchSnapshot> {
  const body = await requestJson(
    `${BATCHES_URL}/${encodeURIComponent(batchId)}`,
    { signal },
    operation,
  );
  return assertBasicBatchSnapshot(body, operation, {
    batchId,
    ...expected,
  });
}

function errorMessage(reason: unknown, fallback: string): string {
  return reason instanceof Error && reason.message
    ? reason.message
    : fallback;
}

function readStoredBatchId(): string {
  if (typeof window === "undefined") return "";
  try {
    const value =
      window.localStorage.getItem(RUN_STORAGE_KEY)?.trim() ?? "";
    return /^[a-f0-9]{32}$/.test(value) ? value : "";
  } catch {
    return "";
  }
}

function storeBatchId(batchId: string) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(RUN_STORAGE_KEY, batchId);
  } catch {
    // Storage can be unavailable in privacy-restricted contexts.
  }
}

function formatDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : DATE_FORMAT.format(date);
}

function formatValue(value: number): string {
  return VALUE_FORMAT.format(value);
}

function formatLimit(value: Limit | undefined): string {
  if (value === undefined || value === null || value === "") {
    return "未定义";
  }
  return isFiniteNumber(value) ? formatValue(value) : String(value);
}

function formatRange(
  lower: Limit | undefined,
  upper: Limit | undefined,
): string {
  if (
    (lower === undefined || lower === null) &&
    (upper === undefined || upper === null)
  ) {
    return "未定义";
  }
  return `${formatLimit(lower)} … ${formatLimit(upper)}`;
}

function formatBytes(value: number): string {
  if (value < 1024) return `${INTEGER_FORMAT.format(value)} B`;
  if (value < 1024 * 1024) {
    return `${VALUE_FORMAT.format(value / 1024)} KiB`;
  }
  return `${VALUE_FORMAT.format(value / (1024 * 1024))} MiB`;
}

function metricUnit(metric: string | null | undefined): string | null {
  if (!metric) return null;
  const normalized = metric.toLocaleLowerCase();
  if (/(?:^|_)dbfs(?:_|$)/.test(normalized)) return "dBFS";
  if (/(?:^|_)db(?:_|$)/.test(normalized)) return "dB";
  if (/(?:^|_)mos(?:_|$)/.test(normalized)) return "MOS";
  if (/(?:^|_)(?:percent|pct)(?:_|$)/.test(normalized)) {
    return "%";
  }
  if (/(?:^|_)ms(?:_|$)/.test(normalized)) return "ms";
  if (/(?:^|_)hz(?:_|$)/.test(normalized)) return "Hz";
  return null;
}

function isZipArchive(file: File): boolean {
  return (
    file.name.toLocaleLowerCase().endsWith(".zip") ||
    file.type === "application/zip" ||
    file.type === "application/x-zip-compressed"
  );
}

function modeLabel(mode: FiveSignalMode): string {
  return `${MODE_LABELS[mode]} · ${mode}`;
}

function algorithmLabel(id: string): string {
  const preset = chainPresets.find((item) => item.id === id);
  return preset ? `${preset.label} · ${preset.id}` : id || "未选择";
}

function conditionLabels(
  binding: EvidenceBinding | undefined,
): string[] {
  if (!binding) return [];
  const rows = [
    `方向 ${binding.direction}`,
    `场景 ${binding.scene}`,
  ];
  if (binding.volume !== null) {
    rows.push(`音量 ${binding.volume}`);
  }
  if (binding.window_ms !== null) {
    rows.push(`窗口 ${formatValue(binding.window_ms)} ms`);
  }
  if (binding.input_level_db !== null) {
    rows.push(`输入 ${formatValue(binding.input_level_db)} dB`);
  }
  return rows;
}

function batchCaseUrl(
  batchId: string,
  caseId: string,
  leaf: "assessment" | "outputs/tx",
): string {
  return `${BATCHES_URL}/${encodeURIComponent(
    batchId,
  )}/cases/${encodeURIComponent(caseId)}/${leaf}`;
}

function asBatchSummary(
  snapshot: BasicBatchSnapshot,
): BasicBatchSummary {
  return {
    batch_id: snapshot.batch_id,
    dataset_id: snapshot.dataset_id,
    dataset_name: snapshot.dataset_name,
    algorithm_cluster_id: snapshot.algorithm_cluster_id,
    status: snapshot.status,
    created_at: snapshot.created_at,
    test_scope: snapshot.test_scope,
    selected_mode: snapshot.selected_mode,
    progress: snapshot.progress,
    summary: snapshot.summary,
  };
}

function ItemStatusBadge({ status }: { status: ItemStatus }) {
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
      {ITEM_STATUS_LABEL[status]}
    </span>
  );
}

export default function BasicUplinkTest({
  algorithmClusterId,
  mode,
  onModeChange,
  onAlgorithmClusterChange,
  externalRunning = false,
  onBusyChange,
}: Props) {
  const modeSelectId = useId();
  const clusterSelectId = useId();
  const datasetSelectId = useId();
  const historySelectId = useId();
  const archiveInputId = useId();

  const [catalog, setCatalog] = useState<CatalogResponse | null>(
    null,
  );
  const [catalogLoading, setCatalogLoading] = useState(true);
  const [catalogError, setCatalogError] = useState("");
  const [catalogRefresh, setCatalogRefresh] = useState(0);
  const [catalogOpen, setCatalogOpen] = useState(true);

  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [datasetsLoading, setDatasetsLoading] = useState(true);
  const [datasetError, setDatasetError] = useState("");
  const [datasetRefresh, setDatasetRefresh] = useState(0);
  const [selectedDatasetId, setSelectedDatasetId] = useState("");
  const [archive, setArchive] = useState<File | null>(null);
  const [importing, setImporting] = useState(false);
  const [importError, setImportError] = useState("");
  const [importSuccess, setImportSuccess] = useState("");

  const [historyRows, setHistoryRows] = useState<
    BasicBatchSummary[]
  >([]);
  const [historyError, setHistoryError] = useState("");
  const [historySelection, setHistorySelection] = useState("");
  const [restoreNonce, setRestoreNonce] = useState(0);
  const [restoring, setRestoring] = useState(true);
  const [restoreError, setRestoreError] = useState("");

  const [snapshot, setSnapshot] =
    useState<BasicBatchSnapshot | null>(null);
  const [launching, setLaunching] = useState(false);
  const [runError, setRunError] = useState("");
  const [pollError, setPollError] = useState("");
  const [pollRetry, setPollRetry] = useState(0);

  const archiveInputRef = useRef<HTMLInputElement>(null);
  const importAbortRef = useRef<AbortController | null>(null);
  const launchAbortRef = useRef<AbortController | null>(null);
  const restoreAbortRef = useRef<AbortController | null>(null);
  const restoreTargetRef = useRef("");
  const snapshotRef = useRef<BasicBatchSnapshot | null>(null);

  snapshotRef.current = snapshot;

  const selectedDataset =
    datasets.find(
      (dataset) => dataset.dataset_id === selectedDatasetId,
    ) ?? null;
  const selectedDatasetMissing =
    Boolean(selectedDatasetId) && selectedDataset === null;
  const selectedDatasetIsComplete =
    selectedDataset?.item_count === 336;
  const algorithmIsSupported = VCP_PRESETS.some(
    (preset) => preset.id === algorithmClusterId,
  );
  const isBatchActive =
    snapshot?.status === "queued" ||
    snapshot?.status === "running";
  const controlsLocked =
    externalRunning ||
    launching ||
    importing ||
    restoring ||
    Boolean(isBatchActive);
  const catalogReady =
    catalog?.mode === mode &&
    catalog.items.length === BASIC_ITEM_COUNT;
  const canStart =
    !controlsLocked &&
    !datasetsLoading &&
    !catalogLoading &&
    Boolean(catalogReady) &&
    Boolean(selectedDataset) &&
    selectedDatasetIsComplete &&
    Boolean(algorithmClusterId) &&
    algorithmIsSupported;

  const reportMode = snapshot?.selected_mode ?? null;
  const reportModeMismatch =
    Boolean(snapshot) && snapshot?.selected_mode !== mode;
  const reportClusterMismatch =
    Boolean(snapshot) &&
    snapshot?.algorithm_cluster_id !== algorithmClusterId;
  const reportDatasetMismatch =
    Boolean(snapshot) &&
    Boolean(selectedDatasetId) &&
    snapshot?.dataset_id !== selectedDatasetId;
  const hasConfigurationMismatch =
    reportModeMismatch ||
    reportClusterMismatch ||
    reportDatasetMismatch;
  const showingPreviousSnapshot =
    Boolean(snapshot) &&
    (launching ||
      restoring ||
      Boolean(runError) ||
      Boolean(pollError));

  const historyChoices = useMemo(() => {
    const rows = new Map<string, BasicBatchSummary>();
    for (const row of historyRows) rows.set(row.batch_id, row);
    if (snapshot) rows.set(snapshot.batch_id, asBatchSummary(snapshot));
    return [...rows.values()].sort(
      (left, right) =>
        Date.parse(right.created_at) - Date.parse(left.created_at) ||
        right.batch_id.localeCompare(left.batch_id),
    );
  }, [historyRows, snapshot]);

  const caseStatusById = useMemo(
    () =>
      new Map(
        (snapshot?.cases ?? []).map((entry) => [
          entry.case_id,
          entry.status,
        ]),
      ),
    [snapshot?.cases],
  );

  const reportCatalogById = useMemo(() => {
    if (!catalog || catalog.mode !== reportMode) {
      return new Map<string, CatalogItem>();
    }
    return new Map(catalog.items.map((item) => [item.id, item]));
  }, [catalog, reportMode]);

  const progressTotal = snapshot?.progress.total_items ?? 0;
  const progressDone = snapshot?.progress.completed_items ?? 0;
  const progressPercent =
    progressTotal > 0
      ? Math.min(100, Math.max(0, (progressDone / progressTotal) * 100))
      : 0;
  const failedCases =
    snapshot?.cases?.filter((entry) => entry.status === "failed") ?? [];
  const canDownloadReport =
    snapshot?.status === "completed" ||
    snapshot?.status === "failed" ||
    snapshot?.status === "cancelled";

  useEffect(() => {
    const controller = new AbortController();
    setCatalog(null);
    setCatalogLoading(true);
    setCatalogError("");
    requestJson(
      `${CATALOG_URL}?mode=${encodeURIComponent(mode)}`,
      { signal: controller.signal },
      "读取基础上行目录",
    )
      .then((body) => assertBasicCatalog(body, mode))
      .then((next) => {
        if (!controller.signal.aborted) setCatalog(next);
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) {
          setCatalogError(
            errorMessage(reason, "基础上行目录读取失败"),
          );
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setCatalogLoading(false);
      });
    return () => controller.abort();
  }, [catalogRefresh, mode]);

  useEffect(() => {
    const controller = new AbortController();
    setDatasetsLoading(true);
    setDatasetError("");
    requestJson(
      DATASETS_URL,
      { signal: controller.signal },
      "读取测试集",
    )
      .then((body) => {
        if (!isDatasetList(body)) {
          throw new Error("测试集列表返回结构不符合接口契约");
        }
        return body;
      })
      .then((rows) => {
        if (controller.signal.aborted) return;
        setDatasets(rows);
        setSelectedDatasetId((current) =>
          rows.some((row) => row.dataset_id === current)
            ? current
            : (rows[0]?.dataset_id ?? ""),
        );
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) {
          setDatasetError(
            errorMessage(reason, "读取测试集列表失败"),
          );
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setDatasetsLoading(false);
      });
    return () => controller.abort();
  }, [datasetRefresh]);

  useEffect(() => {
    const controller = new AbortController();
    restoreAbortRef.current = controller;
    setRestoring(true);
    setRestoreError("");
    setHistoryError("");

    const restore = async () => {
      let history: BasicBatchSummary[] | null = null;
      try {
        const body = await requestJson(
          `${BASIC_BATCHES_URL}?limit=20`,
          { signal: controller.signal },
          "读取基础上行历史",
        );
        if (!isBasicBatchHistory(body)) {
          throw new Error(
            "基础上行历史包含非基础范围或非 8 项记录",
          );
        }
        history = body;
        if (!controller.signal.aborted) setHistoryRows(body);
      } catch (reason) {
        if (!controller.signal.aborted) {
          setHistoryError(
            errorMessage(reason, "读取基础上行历史失败"),
          );
        }
      }

      if (controller.signal.aborted) return;

      const manualTarget = restoreTargetRef.current;
      const storedTarget = manualTarget ? "" : readStoredBatchId();
      const target =
        manualTarget ||
        storedTarget ||
        history?.[0]?.batch_id ||
        "";
      const currentSnapshotId =
        snapshotRef.current?.batch_id ?? "";

      if (!target) {
        setHistorySelection(currentSnapshotId);
        if (history === null) {
          setRestoreError(
            "基础上行历史暂不可用，且本地没有可恢复的报告编号。",
          );
        }
        return;
      }

      setHistorySelection(target);
      try {
        const next = await fetchBasicBatch(
          target,
          controller.signal,
          "恢复基础上行结果",
        );
        if (controller.signal.aborted) return;
        setSnapshot(next);
        setSelectedDatasetId(next.dataset_id);
        setHistorySelection(next.batch_id);
        setRunError("");
        setPollError("");
        storeBatchId(next.batch_id);
        restoreTargetRef.current = "";
      } catch (reason) {
        if (controller.signal.aborted) return;

        const fallbackId =
          !manualTarget &&
          history?.[0]?.batch_id &&
          history[0].batch_id !== target
            ? history[0].batch_id
            : "";
        if (fallbackId) {
          try {
            const fallback = await fetchBasicBatch(
              fallbackId,
              controller.signal,
              "恢复最新基础上行结果",
            );
            if (controller.signal.aborted) return;
            setSnapshot(fallback);
            setSelectedDatasetId(fallback.dataset_id);
            setHistorySelection(fallback.batch_id);
            setRunError("");
            setPollError("");
            storeBatchId(fallback.batch_id);
            restoreTargetRef.current = "";
            setRestoreError(
              `${errorMessage(
                reason,
                "本地保存的报告编号不可用",
              )}；已拒绝该记录并恢复最新基础上行批次 ${fallback.batch_id.slice(
                0,
                8,
              )}。`,
            );
            return;
          } catch (fallbackReason) {
            if (controller.signal.aborted) return;
            setRestoreError(
              `${errorMessage(
                reason,
                "恢复本地记录失败",
              )}；最新基础上行批次也恢复失败：${errorMessage(
                fallbackReason,
                "未知错误",
              )}`,
            );
            setHistorySelection(currentSnapshotId);
            return;
          }
        }

        setRestoreError(
          errorMessage(reason, "恢复基础上行结果失败"),
        );
        setHistorySelection(currentSnapshotId);
      }
    };

    void restore().finally(() => {
      if (!controller.signal.aborted) {
        setRestoring(false);
        if (restoreAbortRef.current === controller) {
          restoreAbortRef.current = null;
        }
      }
    });

    return () => controller.abort();
  }, [restoreNonce]);

  useEffect(() => {
    const controller = new AbortController();
    const current = snapshot;
    if (!current || !isBatchActive) return;

    const batchId = current.batch_id;
    const batchMode = current.selected_mode;
    const batchCluster = current.algorithm_cluster_id;
    let timer: number | undefined;
    let disposed = false;
    let inFlight = false;

    const poll = async () => {
      if (disposed || inFlight) return;
      inFlight = true;
      try {
        const next = await fetchBasicBatch(
          batchId,
          controller.signal,
          "查询基础上行进度",
          {
            datasetId: current.dataset_id,
            algorithmClusterId: batchCluster,
            mode: batchMode,
          },
        );
        if (disposed) return;
        setSnapshot(next);
        setPollError("");
        if (next.status === "queued" || next.status === "running") {
          timer = window.setTimeout(poll, POLL_INTERVAL_MS);
        }
      } catch (reason) {
        if (!disposed && !controller.signal.aborted) {
          setPollError(
            errorMessage(
              reason,
              "基础上行进度查询失败，已停止自动轮询",
            ),
          );
        }
      } finally {
        inFlight = false;
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
  ]);

  useEffect(() => {
    onBusyChange?.(
      launching || importing || restoring || Boolean(isBatchActive),
    );
  }, [
    importing,
    isBatchActive,
    launching,
    onBusyChange,
    restoring,
  ]);

  useEffect(() => {
    if (snapshot?.batch_id) setCatalogOpen(false);
  }, [snapshot?.batch_id]);

  useEffect(() => {
    return () => {
      importAbortRef.current?.abort();
      launchAbortRef.current?.abort();
      restoreAbortRef.current?.abort();
    };
  }, []);

  function handleArchiveChange(file: File | null) {
    setArchive(file);
    setImportError("");
    setImportSuccess("");
    if (!file) return;
    if (!isZipArchive(file)) {
      setArchive(null);
      setImportError("仅支持 .zip 测试集归档。");
      return;
    }
    if (file.size > MAX_ARCHIVE_BYTES) {
      setArchive(null);
      setImportError("ZIP 超过 256 MiB，服务端不会接收。");
    }
  }

  async function handleImport(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (
      !archive ||
      controlsLocked ||
      importAbortRef.current
    ) {
      return;
    }
    const controller = new AbortController();
    importAbortRef.current = controller;
    setImporting(true);
    setImportError("");
    setImportSuccess("");
    try {
      const form = new FormData();
      form.set("archive", archive);
      const body = await requestJson(
        DATASETS_URL,
        {
          method: "POST",
          body: form,
          signal: controller.signal,
        },
        "导入测试集",
      );
      if (!isDatasetSummary(body)) {
        throw new Error("测试集导入结果不符合接口契约");
      }
      if (controller.signal.aborted) return;
      setDatasets((current) => [
        body,
        ...current.filter(
          (row) => row.dataset_id !== body.dataset_id,
        ),
      ]);
      setSelectedDatasetId(body.dataset_id);
      setImportSuccess(
        `已导入 ${body.name}：${INTEGER_FORMAT.format(
          body.item_count,
        )} 项 / ${INTEGER_FORMAT.format(
          body.case_count,
        )} 个条件组。`,
      );
      setArchive(null);
      if (archiveInputRef.current) {
        archiveInputRef.current.value = "";
      }
    } catch (reason) {
      if (!controller.signal.aborted) {
        setImportError(
          errorMessage(reason, "测试集 ZIP 导入失败"),
        );
      }
    } finally {
      if (importAbortRef.current === controller) {
        importAbortRef.current = null;
        if (!controller.signal.aborted) setImporting(false);
      }
    }
  }

  async function handleStart() {
    if (
      !canStart ||
      !selectedDataset ||
      !catalog ||
      launchAbortRef.current
    ) {
      return;
    }
    const requestedDatasetId = selectedDataset.dataset_id;
    const requestedClusterId = algorithmClusterId;
    const requestedMode = mode;
    const controller = new AbortController();
    launchAbortRef.current = controller;
    setLaunching(true);
    setRunError("");
    setPollError("");
    setRestoreError("");
    restoreTargetRef.current = "";
    try {
      const body = await requestJson(
        BASIC_BATCHES_URL,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            dataset_id: requestedDatasetId,
            algorithm_cluster_id: requestedClusterId,
            mode: requestedMode,
          }),
          signal: controller.signal,
        },
        "创建基础上行批次",
      );
      const next = assertBasicBatchSnapshot(
        body,
        "创建基础上行批次",
        {
          datasetId: requestedDatasetId,
          algorithmClusterId: requestedClusterId,
          mode: requestedMode,
        },
      );
      if (controller.signal.aborted) return;
      setSnapshot(next);
      setHistorySelection(next.batch_id);
      setHistoryRows((current) => [
        asBatchSummary(next),
        ...current.filter((row) => row.batch_id !== next.batch_id),
      ]);
      storeBatchId(next.batch_id);
    } catch (reason) {
      if (!controller.signal.aborted) {
        setRunError(
          errorMessage(reason, "创建基础上行批次失败"),
        );
      }
    } finally {
      if (launchAbortRef.current === controller) {
        launchAbortRef.current = null;
        if (!controller.signal.aborted) setLaunching(false);
      }
    }
  }

  function retryPolling() {
    if (!isBatchActive || !pollError) return;
    setPollError("");
    setPollRetry((current) => current + 1);
  }

  const buttonLabel = externalRunning
    ? "其他测试正在执行"
    : restoring
      ? "正在恢复基础结果…"
      : launching
        ? "正在创建批次…"
        : isBatchActive
          ? "基础上行执行中…"
          : runError
            ? "重试创建 8 项批次"
            : "执行所选模式的 8 项";

  return (
    <section
      className="basic-uplink-test"
      aria-labelledby="basic-uplink-title"
    >
      <header className="basic-uplink-header">
        <div className="basic-uplink-heading">
          <span className="basic-uplink-eyebrow">
            BASIC UPLINK ACCEPTANCE
          </span>
          <h2 id="basic-uplink-title">基础上行测试</h2>
          <p>
            复用已导入测试集，只执行所选模式的 8
            项基础上行；结果为工程代理评估，正式 ITU-T 合规未测。
          </p>
        </div>
        <div className="basic-uplink-badges" aria-label="测试口径">
          <span>单模式</span>
          <span>8 项基础上行</span>
          <span>工程代理</span>
          <span>正式合规未测</span>
        </div>
      </header>

      <BasicUplinkTree
        mode={mode}
        algorithmClusterId={algorithmClusterId}
        datasetId={selectedDatasetId}
        dataset={selectedDataset}
        catalog={catalog}
        catalogLoading={catalogLoading}
        catalogError={catalogError}
        snapshot={snapshot}
        syncFailed={Boolean(pollError || restoreError)}
        isPreviousSnapshot={showingPreviousSnapshot}
      />

      <div className="basic-uplink-history">
        <label htmlFor={historySelectId}>
          <History size={15} aria-hidden="true" />
          已保存的基础上行结果
        </label>
        <select
          id={historySelectId}
          value={historySelection || snapshot?.batch_id || ""}
          disabled={
            controlsLocked || historyChoices.length === 0
          }
          onChange={(event) => {
            restoreTargetRef.current = event.target.value;
            setHistorySelection(event.target.value);
            setRestoreNonce((current) => current + 1);
          }}
        >
          {historyChoices.length === 0 ? (
            <option value="">
              {restoring ? "正在恢复…" : "暂无基础上行历史"}
            </option>
          ) : null}
          {historyChoices.map((row) => (
            <option key={row.batch_id} value={row.batch_id}>
              {formatDate(row.created_at)} ·{" "}
              {MODE_LABELS[row.selected_mode]} ·{" "}
              {row.algorithm_cluster_id} ·{" "}
              {BATCH_STATUS_LABEL[row.status]} ·{" "}
              {row.batch_id.slice(0, 8)}
            </option>
          ))}
        </select>
        <button
          className="button secondary"
          type="button"
          disabled={controlsLocked}
          onClick={() =>
            setRestoreNonce((current) => current + 1)
          }
        >
          <RefreshCw size={15} aria-hidden="true" />
          刷新历史与恢复
        </button>
        {restoring ? (
          <p className="basic-uplink-history-note" role="status">
            正在恢复最近一次基础上行结果，并校验业务模式与 8 项范围。
          </p>
        ) : null}
        {historyError ? (
          <p
            className="basic-uplink-inline-error"
            role="alert"
          >
            {historyError}
            {snapshot ? "；当前仍保留最近一次成功快照。" : ""}
          </p>
        ) : null}
        {restoreError ? (
          <p
            className="basic-uplink-inline-warning"
            role="alert"
          >
            {restoreError}
          </p>
        ) : null}
      </div>

      <div className="basic-uplink-controls">
        <label htmlFor={modeSelectId}>
          <span>业务模式</span>
          <select
            id={modeSelectId}
            aria-label="业务模式"
            value={mode}
            disabled={controlsLocked}
            onChange={(event) =>
              onModeChange(event.target.value as FiveSignalMode)
            }
          >
            {FIVE_SIGNAL_MODES.map((value) => (
              <option key={value} value={value}>
                {modeLabel(value)}
              </option>
            ))}
          </select>
          <small>目录与批次都按此模式严格校验。</small>
        </label>

        <label htmlFor={clusterSelectId}>
          <span>VCP 算法簇</span>
          <select
            id={clusterSelectId}
            aria-label="VCP 算法簇"
            value={algorithmClusterId}
            disabled={controlsLocked}
            onChange={(event) =>
              onAlgorithmClusterChange(event.target.value)
            }
          >
            {!algorithmIsSupported ? (
              <option value={algorithmClusterId}>
                {algorithmClusterId
                  ? `${algorithmClusterId}（非 VCP，不可执行）`
                  : "未选择算法簇"}
              </option>
            ) : null}
            {VCP_PRESETS.map((preset) => (
              <option key={preset.id} value={preset.id}>
                {preset.label} · {preset.id}
              </option>
            ))}
          </select>
          <small>
            {algorithmIsSupported
              ? algorithmLabel(algorithmClusterId)
              : "基础上行只接受当前目录中的 VCP 算法簇。"}
          </small>
        </label>

        <label htmlFor={datasetSelectId}>
          <span>已导入测试集</span>
          <select
            id={datasetSelectId}
            aria-label="已导入测试集"
            value={selectedDatasetId}
            disabled={
              controlsLocked ||
              datasetsLoading ||
              (datasets.length === 0 &&
                !selectedDatasetMissing)
            }
            onChange={(event) =>
              setSelectedDatasetId(event.target.value)
            }
          >
            {datasets.length === 0 ? (
              <option value="">
                {datasetsLoading
                  ? "正在读取测试集…"
                  : "暂无已导入测试集"}
              </option>
            ) : null}
            {selectedDatasetMissing ? (
              <option value={selectedDatasetId}>
                {snapshot?.dataset_name ?? selectedDatasetId}
                （当前列表未返回）
              </option>
            ) : null}
            {datasets.map((dataset) => (
              <option
                key={dataset.dataset_id}
                value={dataset.dataset_id}
              >
                {dataset.name} · {formatDate(dataset.created_at)}
                {" · "}
                {dataset.dataset_id.slice(0, 8)} ·{" "}
                {dataset.item_count} 项 / {dataset.case_count} 条件组
              </option>
            ))}
          </select>
          <small>
            {selectedDataset
              ? `${formatDate(selectedDataset.created_at)} 导入；${
                  selectedDatasetIsComplete
                    ? "覆盖 336 项目录"
                    : `仅 ${selectedDataset.item_count} 项，不能执行`
                }`
              : "只列出服务端已校验的测试集。"}
          </small>
        </label>

        <div className="basic-uplink-refresh-cell">
          <span>测试集列表</span>
          <button
            className="button secondary"
            type="button"
            disabled={controlsLocked || datasetsLoading}
            onClick={() =>
              setDatasetRefresh((current) => current + 1)
            }
          >
            <RefreshCw size={15} aria-hidden="true" />
            {datasetsLoading ? "正在刷新…" : "刷新数据集"}
          </button>
          <small>
            切换测试集不会改写已保留的报告。
          </small>
        </div>
      </div>

      {datasetError ? (
        <p className="basic-uplink-inline-error" role="alert">
          {datasetError}
        </p>
      ) : null}
      {selectedDataset && !selectedDatasetIsComplete ? (
        <p className="basic-uplink-inline-error" role="alert">
          当前测试集未覆盖 336 项完整合同，不能创建基础上行批次。
        </p>
      ) : null}

      <form
        className="basic-uplink-upload"
        onSubmit={handleImport}
      >
        <div>
          <label htmlFor={archiveInputId}>
            <Upload size={15} aria-hidden="true" />
            上传现有 336 项 ZIP
          </label>
          <input
            ref={archiveInputRef}
            id={archiveInputId}
            type="file"
            accept=".zip,application/zip,application/x-zip-compressed"
            disabled={controlsLocked || importing}
            onChange={(event) => {
              const file = event.target.files?.[0] ?? null;
              handleArchiveChange(file);
              if (
                file &&
                (!isZipArchive(file) ||
                  file.size > MAX_ARCHIVE_BYTES)
              ) {
                event.currentTarget.value = "";
              }
            }}
          />
        </div>
        <div className="basic-uplink-file-meta">
          <FileArchive size={15} aria-hidden="true" />
          <span>
            {archive
              ? `${archive.name} · ${formatBytes(archive.size)}`
              : "manifest.json + audio/"}
          </span>
        </div>
        <button
          className="button secondary"
          type="submit"
          disabled={
            !archive ||
            importing ||
            controlsLocked ||
            datasetsLoading
          }
        >
          <Upload size={15} aria-hidden="true" />
          {importing ? "正在导入并校验…" : "导入 ZIP"}
        </button>
      </form>
      {importError ? (
        <p className="basic-uplink-inline-error" role="alert">
          {importError}
        </p>
      ) : null}
      {importSuccess ? (
        <p className="basic-uplink-inline-success" role="status">
          <CheckCircle2 size={14} aria-hidden="true" />
          {importSuccess}
        </p>
      ) : null}

      <div className="basic-uplink-run">
        <div>
          <strong>
            {modeLabel(mode)} · {algorithmLabel(algorithmClusterId)}
          </strong>
          <p>
            执行路径：按当前模式提取条件组 → VCP 上行处理 → 8 项计算
            → 保存报告。
          </p>
        </div>
        <button
          className="button primary"
          type="button"
          disabled={!canStart}
          onClick={() => void handleStart()}
        >
          <Play size={15} fill="currentColor" aria-hidden="true" />
          {buttonLabel}
        </button>
      </div>

      {runError ? (
        <div className="basic-uplink-alert" role="alert">
          <AlertCircle size={18} aria-hidden="true" />
          <div>
            <strong>本次批次创建失败</strong>
            <p>{runError}</p>
            <p>
              下方仍只显示最后一次成功快照；失败请求没有被记作通过或零值。
            </p>
          </div>
        </div>
      ) : null}

      <details
        className="basic-uplink-catalog"
        aria-labelledby="basic-uplink-catalog-title"
        open={catalogOpen}
        onToggle={(event) =>
          setCatalogOpen(event.currentTarget.open)
        }
      >
        <summary className="basic-uplink-catalog-summary">
          <span className="basic-uplink-catalog-summary-copy">
            <span className="basic-uplink-eyebrow">
              CONTRACT CATALOG
            </span>
            <span
              className="basic-uplink-catalog-title"
              id="basic-uplink-catalog-title"
            >
              8 项基础上行合同目录
            </span>
            <span className="basic-uplink-catalog-description">
              {modeLabel(mode)}
              ；展开查看合同行、单位与限值。
            </span>
          </span>
          <span className="basic-uplink-catalog-summary-meta">
            <span className="basic-uplink-catalog-count">
              {catalog
                ? `${catalog.item_count} / 8`
                : catalogError
                  ? "读取失败"
                  : "8 行空状态"}
            </span>
            <span className="basic-uplink-catalog-toggle">
              {catalogOpen ? "收起目录" : "展开目录"}
            </span>
          </span>
        </summary>
        <div className="basic-uplink-catalog-body">
          {catalogError ? (
            <div className="basic-uplink-catalog-error">
              <p className="basic-uplink-inline-error" role="alert">
                {catalogError}
              </p>
              <button
                className="button secondary"
                type="button"
                disabled={catalogLoading}
                onClick={() =>
                  setCatalogRefresh((current) => current + 1)
                }
              >
                <RefreshCw size={15} aria-hidden="true" />
                重试读取目录
              </button>
            </div>
          ) : null}
          <div className="table-scroll">
            <table className="audio-table basic-uplink-catalog-table">
              <caption className="sr-only">
                当前模式的 8 项基础上行合同目录
              </caption>
              <thead>
                <tr>
                  <th scope="col">合同行</th>
                  <th scope="col">测项</th>
                  <th scope="col">分类</th>
                  <th scope="col">合同单位</th>
                  <th scope="col">合同限值</th>
                </tr>
              </thead>
              <tbody>
                {Array.from(
                  { length: BASIC_ITEM_COUNT },
                  (_, index) => {
                    const item = catalog?.items[index];
                    return (
                      <tr
                        key={
                          item?.id ?? `catalog-empty-${index}`
                        }
                      >
                        <td>
                          <div className="basic-uplink-source-row">
                            <code>{item?.id ?? "—"}</code>
                            <span>
                              {item?.source_cell ??
                                (catalogLoading
                                  ? "目录读取中"
                                  : "目录不可用")}
                            </span>
                          </div>
                        </td>
                        <td>
                          <strong>
                            {item?.test_name ?? "—"}
                          </strong>
                        </td>
                        <td>
                          <code>{item?.family ?? "—"}</code>
                        </td>
                        <td>{item?.unit ?? "—"}</td>
                        <td>
                          {item
                            ? formatRange(
                                item.lower_limit,
                                item.upper_limit,
                              )
                            : "—"}
                        </td>
                      </tr>
                    );
                  },
                )}
              </tbody>
            </table>
          </div>
        </div>
      </details>

      {!snapshot ? (
        <div className="basic-uplink-empty">
          <ListChecks size={24} aria-hidden="true" />
          <div>
            <h3>尚无基础上行批次结果</h3>
            <p>
              选择测试集与 VCP 算法簇后执行。上方仍固定展示当前模式的
              8 行合同目录，不会预填通过、数值或状态。
            </p>
          </div>
        </div>
      ) : (
        <>
          <section
            className="basic-uplink-result"
            aria-labelledby="basic-uplink-result-title"
          >
            <header className="basic-uplink-result-header">
              <div>
                <span className="basic-uplink-eyebrow">
                  SERVER SNAPSHOT
                </span>
                <h3 id="basic-uplink-result-title">
                  {showingPreviousSnapshot
                    ? "上次成功快照 · "
                    : ""}
                  {BATCH_STATUS_LABEL[snapshot.status]}
                </h3>
                <p>
                  报告实际配置：{modeLabel(snapshot.selected_mode)}
                  {" · "}
                  {algorithmLabel(snapshot.algorithm_cluster_id)}
                  {" · "}
                  {snapshot.dataset_name}
                </p>
              </div>
              <div className="basic-uplink-result-actions">
                <span
                  className={`basic-uplink-run-status is-${snapshot.status}`}
                  role="status"
                  aria-live="polite"
                >
                  {isBatchActive ? (
                    <Clock3 size={14} aria-hidden="true" />
                  ) : null}
                  {isBatchActive
                    ? pollError
                      ? "轮询已停止"
                      : `每 2 秒读取 · ${formatValue(
                          progressPercent,
                        )}%`
                    : BATCH_STATUS_LABEL[snapshot.status]}
                </span>
                {canDownloadReport ? (
                  <a
                    className="button secondary"
                    href={`${BATCHES_URL}/${encodeURIComponent(
                      snapshot.batch_id,
                    )}/report`}
                    download={`basic-uplink-${snapshot.batch_id}.json`}
                  >
                    <Download size={15} aria-hidden="true" />
                    下载报告 JSON
                  </a>
                ) : (
                  <span className="basic-uplink-report-pending">
                    批次结束后可下载 JSON
                  </span>
                )}
              </div>
            </header>

            <div className="basic-uplink-progress">
              <div>
                <strong>
                  条件组 {snapshot.progress.completed_cases} /{" "}
                  {snapshot.progress.total_cases}
                </strong>
                <span>
                  逐项 {snapshot.progress.completed_items} /{" "}
                  {snapshot.progress.total_items}
                </span>
              </div>
              <progress
                max={Math.max(1, progressTotal)}
                value={Math.min(
                  progressDone,
                  Math.max(1, progressTotal),
                )}
                aria-label="基础上行逐项完成进度"
              >
                {formatValue(progressPercent)}%
              </progress>
            </div>

            <div className="basic-uplink-result-boundary">
              <ShieldAlert size={18} aria-hidden="true" />
              <div>
                <strong>
                  工程代理结果 · 正式合规未测
                </strong>
                <p>
                  “工程通过”只表示服务端计算满足当前代理判据；
                  需复核、不可测与无效项保持原状态，不能据此宣称正式
                  ITU-T 合规。
                </p>
              </div>
            </div>

            {hasConfigurationMismatch ? (
              <div
                className="basic-uplink-mismatch"
                role="status"
              >
                <AlertCircle size={18} aria-hidden="true" />
                <div>
                  <strong>当前选择与报告实际配置不同</strong>
                  <p>
                    报告实际执行：
                    {modeLabel(snapshot.selected_mode)}
                    {" · "}
                    {algorithmLabel(snapshot.algorithm_cluster_id)}
                    {" · "}
                    {snapshot.dataset_name}。当前选择：
                    {modeLabel(mode)}
                    {" · "}
                    {algorithmLabel(
                      algorithmClusterId || "未选择",
                    )}
                    {" · "}
                    {selectedDataset?.name ?? "未选择测试集"}。旧结果继续供查看，
                    但不会被伪装成当前配置。
                  </p>
                </div>
              </div>
            ) : null}

            {showingPreviousSnapshot ? (
              <p className="basic-uplink-stale" role="status">
                当前保留最后一次成功快照；恢复、创建或轮询失败不会改写逐项值与状态。
              </p>
            ) : null}

            <dl className="basic-uplink-summary">
              {(
                [
                  ["总项", snapshot.summary.total],
                  ["已计算", snapshot.summary.computed],
                  ["工程通过", snapshot.summary.pass],
                  ["工程失败", snapshot.summary.fail],
                  ["需复核", snapshot.summary.review_required],
                  ["不可测", snapshot.summary.not_measured],
                  ["无效", snapshot.summary.invalid],
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

            <div className="basic-uplink-identity">
              <span>
                报告编号 <code>{snapshot.batch_id}</code>
              </span>
              <span>
                测试范围{" "}
                <code>基础上行 · 8 项</code>
              </span>
              <span>
                业务模式{" "}
                <code>{modeLabel(snapshot.selected_mode)}</code>
              </span>
              <span>
                算法{" "}
                <code>
                  {algorithmLabel(snapshot.algorithm_cluster_id)}
                </code>
              </span>
              <span>
                报告时间 <code>{formatDate(snapshot.created_at)}</code>
              </span>
            </div>

            {pollError ? (
              <div
                className="basic-uplink-alert"
                role="alert"
              >
                <AlertCircle size={18} aria-hidden="true" />
                <div>
                  <strong>自动轮询已停止</strong>
                  <p>{pollError}</p>
                  <button
                    className="button secondary"
                    type="button"
                    onClick={retryPolling}
                  >
                    <RefreshCw size={15} aria-hidden="true" />
                    重新查询批次
                  </button>
                </div>
              </div>
            ) : null}

            {snapshot.error ? (
              <div
                className="basic-uplink-alert"
                role="alert"
              >
                <AlertCircle size={18} aria-hidden="true" />
                <div>
                  <strong>服务端报告批次错误</strong>
                  <p>{snapshot.error}</p>
                </div>
              </div>
            ) : null}

            {failedCases.length > 0 ? (
              <div
                className="basic-uplink-alert"
                role="alert"
              >
                <AlertCircle size={18} aria-hidden="true" />
                <div>
                  <strong>
                    {failedCases.length} 个条件组执行失败
                  </strong>
                  {failedCases.slice(0, 3).map((entry) => (
                    <p key={entry.case_id}>
                      <code>{entry.case_id}</code>
                      {entry.error
                        ? `：${entry.error}`
                        : "：条件组执行失败"}
                    </p>
                  ))}
                </div>
              </div>
            ) : null}
          </section>

          <section
            className="basic-uplink-items"
            aria-labelledby="basic-uplink-items-title"
          >
            <header>
              <div>
                <span className="basic-uplink-eyebrow">
                  ITEM EVIDENCE
                </span>
                <h3 id="basic-uplink-items-title">
                  8 项逐项结果
                </h3>
                <p>
                  展示值是服务端返回的工程代理计算值，不是正式合同测项值；
                  合同单位不自动套用于计算指标。条件组完成后才显示报告和
                  发送处理输出（TX）音频链接。
                </p>
              </div>
              <span>{snapshot.items.length} / 8</span>
            </header>

            <div className="table-scroll">
              <table className="audio-table basic-uplink-items-table">
                <caption className="sr-only">
                  基础上行 8 项逐项工程结果
                </caption>
                <thead>
                  <tr>
                    <th scope="col">合同行 / 测项</th>
                    <th scope="col">条件组</th>
                    <th scope="col">工程实测值 / 计算指标</th>
                    <th scope="col">合同单位</th>
                    <th scope="col">状态</th>
                    <th scope="col">原因</th>
                    <th scope="col">证据与发送处理输出（TX）</th>
                  </tr>
                </thead>
                <tbody>
                  {snapshot.items.map((item) => {
                    const catalogItem =
                      reportCatalogById.get(item.item_id) ??
                      (item.id
                        ? reportCatalogById.get(item.id)
                        : undefined);
                    const sourceId =
                      item.id ??
                      catalogItem?.id ??
                      item.item_id;
                    const sourceCell =
                      item.source_cell ??
                      catalogItem?.source_cell ??
                      "未返回源行";
                    const testName =
                      item.test_name ??
                      catalogItem?.test_name ??
                      "未命名测项";
                    const family =
                      item.family ??
                      catalogItem?.family ??
                      "未分类";
                    const unit =
                      item.unit ??
                      catalogItem?.unit ??
                      "";
                    const lower =
                      item.lower_limit !== undefined
                        ? item.lower_limit
                        : catalogItem?.lower_limit;
                    const upper =
                      item.upper_limit !== undefined
                        ? item.upper_limit
                        : catalogItem?.upper_limit;
                    const binding = item.evidence_binding;
                    const conditions = conditionLabels(binding);
                    const caseId = binding?.case_id;
                    const caseCompleted =
                      Boolean(caseId) &&
                      caseStatusById.get(caseId ?? "") ===
                        "completed";
                    const computedUnit = metricUnit(item.metric);
                    return (
                      <tr key={item.item_id}>
                        <td>
                          <div className="basic-uplink-item-name">
                            <code>{sourceId}</code>
                            <strong>{testName}</strong>
                            <span>
                              {sourceCell} · {family}
                            </span>
                            <span>
                              合同限值{" "}
                              {formatRange(lower, upper)}
                            </span>
                          </div>
                        </td>
                        <td>
                          {conditions.length > 0 ? (
                            <div className="basic-uplink-condition">
                              {conditions.map((condition) => (
                                <span key={condition}>
                                  {condition}
                                </span>
                              ))}
                            </div>
                          ) : (
                            <span className="basic-uplink-muted">
                              未绑定条件组
                            </span>
                          )}
                          {caseId ? (
                            <code className="basic-uplink-case-id">
                              case {caseId}
                            </code>
                          ) : null}
                        </td>
                        <td>
                          <div className="basic-uplink-value">
                            {item.value !== null ? (
                              <strong>
                                {formatValue(item.value)}
                                {computedUnit
                                  ? ` ${computedUnit}`
                                  : ""}
                              </strong>
                            ) : (
                              <span>无数值</span>
                            )}
                            {item.metric ? (
                              <code>{item.metric}</code>
                            ) : (
                              <span>计算指标未返回</span>
                            )}
                          </div>
                        </td>
                        <td>
                          <div className="basic-uplink-contract-unit">
                            <span>{unit || "—"}</span>
                            <small>合同字段</small>
                          </div>
                        </td>
                        <td>
                          <ItemStatusBadge status={item.status} />
                        </td>
                        <td>
                          <div className="basic-uplink-reason">
                            {item.reason ?? "服务端未返回原因"}
                          </div>
                        </td>
                        <td>
                          {binding ? (
                            <div className="basic-uplink-evidence">
                              <code>
                                {binding.report_id ??
                                  "report 待生成"}
                              </code>
                              {caseCompleted && caseId ? (
                                <div>
                                  <a
                                    href={batchCaseUrl(
                                      snapshot.batch_id,
                                      caseId,
                                      "assessment",
                                    )}
                                    download
                                  >
                                    条件组报告
                                  </a>
                                  <a
                                    href={batchCaseUrl(
                                      snapshot.batch_id,
                                      caseId,
                                      "outputs/tx",
                                    )}
                                    download
                                  >
                                    发送处理输出（TX）音频
                                  </a>
                                </div>
                              ) : (
                                <span>
                                  条件组完成后可下载报告与发送处理输出（TX）
                                </span>
                              )}
                            </div>
                          ) : (
                            <span className="basic-uplink-muted">
                              未绑定证据
                            </span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}
    </section>
  );
}
