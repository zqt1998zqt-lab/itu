import {
  useEffect,
  useId,
  useLayoutEffect,
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
  ExternalLink,
  FileArchive,
  FileJson,
  MinusCircle,
  Play,
  RefreshCw,
  Search,
  ShieldAlert,
  Upload,
  XCircle,
} from "lucide-react";
import {
  FIVE_SIGNAL_MODES,
  type FiveSignalMode,
} from "./five-signal-flow-model";
import "./alignment-acceptance-batch.css";

const DATASETS_URL = "/api/alignment/acceptance/datasets";
const TEMPLATE_URL = "/api/alignment/acceptance/template";
const BATCHES_URL = "/api/alignment/acceptance/batches";
const PAGE_SIZE = 42;
const MAX_ARCHIVE_BYTES = 256 * 1024 * 1024;
const LATEST_BATCH_STORAGE_KEY =
  "itu.alignment.acceptance.latest-batch-id.v1";

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
type OutputRole = "tx" | "rx";
const OUTPUT_ROLE_LABELS: Record<OutputRole, string> = {
  tx: "发送处理输出（TX）",
  rx: "接收处理输出（RX）",
};
type ModeFilter = "all" | FiveSignalMode;
type StatusFilter = "all" | ItemStatus;

type DatasetSummary = {
  dataset_id: string;
  name: string;
  case_count: number;
  item_count: number;
  manifest_sha256: string;
  created_at: string;
};

type EvidenceBinding = {
  mode: string;
  direction?: string;
  scene: string;
  volume: string | number | null;
  window_ms: number | null;
  input_level_db: number | null;
  case_id: string;
  report_id: string | null;
  batch_id: string;
  dataset_id: string;
  input_sha256?: Record<string, string>;
  output_sha256?: Record<string, string>;
};

type AcceptanceBatchItem = {
  item_id: string;
  mode?: string;
  family?: string;
  test_name?: string;
  status: ItemStatus;
  value: number | null;
  metric?: string | null;
  reason?: string;
  evidence_binding?: EvidenceBinding;
};

type AcceptanceBatchModeReport = {
  mode: string;
  evaluation_id: string;
  summary: {
    total: number;
    measured: number;
    pass: number;
    fail: number;
    review_required: number;
    not_measured: number;
    invalid: number;
  };
};

type AcceptanceBatchCase = {
  case_id: string;
  report_id: string;
  status: string;
  conditions?: Record<string, unknown>;
  error?: string;
  assessment_sha256?: string;
  [key: string]: unknown;
};

export type AcceptanceBatchSnapshot = {
  batch_id: string;
  dataset_id: string;
  dataset_name: string;
  algorithm_cluster_id: string;
  status: BatchStatus;
  progress: {
    completed_cases: number;
    total_cases: number;
    completed_items: number;
    total_items: number;
  };
  summary: {
    total: number;
    computed: number;
    review_required: number;
    not_measured: number;
    invalid: number;
    pass: number;
    fail: number;
  };
  items: AcceptanceBatchItem[];
  mode_reports?: AcceptanceBatchModeReport[];
  cases?: AcceptanceBatchCase[];
  error?: string | null;
  created_at: string;
};

type AcceptanceBatchSummary = Pick<
  AcceptanceBatchSnapshot,
  | "batch_id"
  | "dataset_id"
  | "dataset_name"
  | "algorithm_cluster_id"
  | "status"
  | "progress"
  | "summary"
  | "created_at"
>;

type Props = {
  algorithmClusterId: string;
  externalRunning?: boolean;
  onBusyChange?: (busy: boolean) => void;
  onRestoringChange?: (restoring: boolean) => void;
  onBatchChange?: (snapshot: AcceptanceBatchSnapshot | null) => void;
  onSelectMode?: (mode: FiveSignalMode) => void;
};

const BATCH_STATUS_LABEL: Record<BatchStatus, string> = {
  queued: "批次排队中",
  running: "批次执行中",
  completed: "批次已完成",
  failed: "批次失败",
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

const MODE_SET = new Set<string>(FIVE_SIGNAL_MODES);
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

function isNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isNullableNumber(value: unknown): value is number | null {
  return value === null || isNumber(value);
}

function isNullableStringOrNumber(
  value: unknown,
): value is string | number | null {
  return value === null || isString(value) || isNumber(value);
}

function isOptionalStringRecord(
  value: unknown,
): value is Record<string, string> | undefined {
  return (
    value === undefined ||
    (isRecord(value) && Object.values(value).every(isString))
  );
}

function isDatasetSummary(value: unknown): value is DatasetSummary {
  return (
    isRecord(value) &&
    isString(value.dataset_id) &&
    isString(value.name) &&
    isNumber(value.case_count) &&
    isNumber(value.item_count) &&
    isString(value.manifest_sha256) &&
    isString(value.created_at)
  );
}

function isDatasetList(value: unknown): value is DatasetSummary[] {
  return Array.isArray(value) && value.every(isDatasetSummary);
}

function isEvidenceBinding(value: unknown): value is EvidenceBinding {
  return (
    isRecord(value) &&
    isString(value.mode) &&
    (value.direction === undefined || isString(value.direction)) &&
    isString(value.scene) &&
    isNullableStringOrNumber(value.volume) &&
    isNullableNumber(value.window_ms) &&
    isNullableNumber(value.input_level_db) &&
    isString(value.case_id) &&
    (value.report_id === null || isString(value.report_id)) &&
    isString(value.batch_id) &&
    isString(value.dataset_id) &&
    isOptionalStringRecord(value.input_sha256) &&
    isOptionalStringRecord(value.output_sha256)
  );
}

function isAcceptanceBatchItem(value: unknown): value is AcceptanceBatchItem {
  if (!isRecord(value) || !isString(value.item_id)) return false;
  if (!(ITEM_STATUSES as readonly string[]).includes(String(value.status))) {
    return false;
  }
  if (!isNullableNumber(value.value)) return false;
  if (value.mode !== undefined && !isString(value.mode)) return false;
  if (value.family !== undefined && !isString(value.family)) return false;
  if (value.test_name !== undefined && !isString(value.test_name)) return false;
  if (value.metric !== undefined && value.metric !== null && !isString(value.metric)) return false;
  if (value.reason !== undefined && !isString(value.reason)) return false;
  return (
    value.evidence_binding === undefined ||
    isEvidenceBinding(value.evidence_binding)
  );
}

function isAcceptanceBatchModeReport(
  value: unknown,
): value is AcceptanceBatchModeReport {
  if (
    !isRecord(value) ||
    !isString(value.mode) ||
    !isString(value.evaluation_id) ||
    !isRecord(value.summary)
  ) {
    return false;
  }
  const summary = value.summary;
  return (
    isNumber(summary.total) &&
    isNumber(summary.measured) &&
    isNumber(summary.pass) &&
    isNumber(summary.fail) &&
    isNumber(summary.review_required) &&
    isNumber(summary.not_measured) &&
    isNumber(summary.invalid)
  );
}

function isAcceptanceBatchCase(value: unknown): value is AcceptanceBatchCase {
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

function isBatchSnapshot(value: unknown): value is AcceptanceBatchSnapshot {
  if (!isRecord(value)) return false;
  if (
    !isString(value.batch_id) ||
    !isString(value.dataset_id) ||
    !isString(value.dataset_name) ||
    !isString(value.algorithm_cluster_id) ||
    !isString(value.created_at) ||
    !(BATCH_STATUSES as readonly string[]).includes(String(value.status)) ||
    !Array.isArray(value.items) ||
    !value.items.every(isAcceptanceBatchItem) ||
    (value.mode_reports !== undefined &&
      (!Array.isArray(value.mode_reports) ||
        !value.mode_reports.every(isAcceptanceBatchModeReport))) ||
    (value.cases !== undefined &&
      (!Array.isArray(value.cases) ||
        !value.cases.every(isAcceptanceBatchCase)))
  ) {
    return false;
  }
  if (
    value.error !== undefined &&
    value.error !== null &&
    !isString(value.error)
  ) {
    return false;
  }
  const progress = value.progress;
  const summary = value.summary;
  return (
    isRecord(progress) &&
    isNumber(progress.completed_cases) &&
    isNumber(progress.total_cases) &&
    isNumber(progress.completed_items) &&
    isNumber(progress.total_items) &&
    isRecord(summary) &&
    isNumber(summary.total) &&
    isNumber(summary.computed) &&
    isNumber(summary.review_required) &&
    isNumber(summary.not_measured) &&
    isNumber(summary.invalid) &&
    isNumber(summary.pass) &&
    isNumber(summary.fail)
  );
}

function isBatchHistory(value: unknown): value is AcceptanceBatchSummary[] {
  return (
    Array.isArray(value) &&
    value.every(
      (row) =>
        isRecord(row) &&
        isString(row.batch_id) &&
        /^[a-f0-9]{32}$/.test(row.batch_id) &&
        isBatchSnapshot({ ...row, items: [] }),
    )
  );
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

async function readApiJson<T>(
  response: Response,
  validator: (value: unknown) => value is T,
  operation: string,
): Promise<T> {
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
      apiErrorDetail(body) || `${operation}失败（HTTP ${response.status}）`,
      response.status,
    );
  }
  if (!validator(body)) {
    throw new Error(`${operation}返回结构不符合接口契约`);
  }
  return body;
}

function errorMessage(reason: unknown, fallback: string): string {
  return reason instanceof Error && reason.message ? reason.message : fallback;
}

function formatDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : DATE_FORMAT.format(date);
}

function formatValue(value: number): string {
  return VALUE_FORMAT.format(value);
}

function formatBytes(value: number): string {
  if (value < 1024) return `${INTEGER_FORMAT.format(value)} B`;
  if (value < 1024 * 1024) {
    return `${VALUE_FORMAT.format(value / 1024)} KiB`;
  }
  return `${VALUE_FORMAT.format(value / (1024 * 1024))} MiB`;
}

function shortHash(value: string): string {
  return value.length > 16 ? `${value.slice(0, 16)}…` : value;
}

function isZipArchive(file: File): boolean {
  return (
    file.name.toLocaleLowerCase().endsWith(".zip") ||
    file.type === "application/zip" ||
    file.type === "application/x-zip-compressed"
  );
}

function outputRolesForDirection(direction: unknown): OutputRole[] {
  if (direction === "receive") return ["rx"];
  if (direction === "round_trip") return ["tx", "rx"];
  if (direction === "send") return ["tx"];
  return [];
}

function batchCaseEvidenceUrl(
  batchId: string,
  caseId: string,
  leaf: "assessment" | `outputs/${OutputRole}`,
): string {
  return `${BATCHES_URL}/${encodeURIComponent(batchId)}/cases/${encodeURIComponent(caseId)}/${leaf}`;
}

function readLatestBatchId(): string {
  if (typeof window === "undefined") return "";
  try {
    return window.localStorage.getItem(LATEST_BATCH_STORAGE_KEY)?.trim() ?? "";
  } catch {
    return "";
  }
}

function storeLatestBatchId(batchId: string) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(LATEST_BATCH_STORAGE_KEY, batchId);
  } catch {
    // Local storage may be unavailable in privacy-restricted contexts.
  }
}

function clearLatestBatchId() {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.removeItem(LATEST_BATCH_STORAGE_KEY);
  } catch {
    // Local storage may be unavailable in privacy-restricted contexts.
  }
}

function itemMode(item: AcceptanceBatchItem): string {
  if (item.mode) return item.mode;
  if (item.evidence_binding?.mode) return item.evidence_binding.mode;
  const prefix = item.item_id.split(":", 1)[0];
  return MODE_SET.has(prefix) ? prefix : "";
}

function itemCondition(item: AcceptanceBatchItem): string[] {
  const binding = item.evidence_binding;
  if (!binding) return [];
  const rows: string[] = [];
  if (binding.direction) rows.push(`方向 ${binding.direction}`);
  if (binding.scene) rows.push(`场景 ${binding.scene}`);
  if (binding.volume !== null && binding.volume !== "") {
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

function itemSearchText(item: AcceptanceBatchItem): string {
  const binding = item.evidence_binding;
  return [
    ...itemCondition(item),
    item.item_id,
    itemMode(item),
    item.family,
    item.test_name,
    item.metric,
    item.reason,
    binding?.scene,
    binding?.volume,
    binding?.case_id,
    binding?.report_id,
  ]
    .filter((value) => value !== null && value !== undefined)
    .join(" ")
    .toLocaleLowerCase();
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
    <span className={`acceptance-batch-item-status is-${status}`}>
      <Icon size={13} aria-hidden="true" />
      {ITEM_STATUS_LABEL[status]}
    </span>
  );
}

export default function AlignmentAcceptanceBatch({
  algorithmClusterId,
  externalRunning = false,
  onBusyChange,
  onRestoringChange,
  onBatchChange,
  onSelectMode,
}: Props) {
  const fileInputId = useId();
  const datasetSelectId = useId();
  const historySelectId = useId();
  const searchId = useId();
  const modeFilterId = useId();
  const statusFilterId = useId();

  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [datasetsLoading, setDatasetsLoading] = useState(true);
  const [datasetError, setDatasetError] = useState("");
  const [datasetRefresh, setDatasetRefresh] = useState(0);
  const [selectedDatasetId, setSelectedDatasetId] = useState("");
  const [archive, setArchive] = useState<File | null>(null);
  const [importing, setImporting] = useState(false);
  const [importError, setImportError] = useState("");
  const [importSuccess, setImportSuccess] = useState("");
  const [snapshot, setSnapshot] = useState<AcceptanceBatchSnapshot | null>(null);
  const [restoringBatch, setRestoringBatch] = useState(true);
  const [restoreUnresolved, setRestoreUnresolved] = useState(false);
  const [restoreError, setRestoreError] = useState("");
  const [restoreRefresh, setRestoreRefresh] = useState(0);
  const [recentBatches, setRecentBatches] = useState<AcceptanceBatchSummary[]>([]);
  const [historyError, setHistoryError] = useState("");
  const [launching, setLaunching] = useState(false);
  const [runError, setRunError] = useState("");
  const [pollError, setPollError] = useState("");
  const [pollRetry, setPollRetry] = useState(0);
  const [modeFilter, setModeFilter] = useState<ModeFilter>("all");
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  const [query, setQuery] = useState("");
  const [page, setPage] = useState(0);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const importAbortRef = useRef<AbortController | null>(null);
  const launchAbortRef = useRef<AbortController | null>(null);
  const restoreTargetRef = useRef("");
  const revealBatchRef = useRef("");
  const resultRef = useRef<HTMLElement>(null);
  const itemsRef = useRef<HTMLElement>(null);
  const lastNotifiedRef = useRef<AcceptanceBatchSnapshot | null | undefined>(
    undefined,
  );
  const callbackRef = useRef(onBatchChange);

  const currentSnapshot = snapshot;
  const snapshotMatchesCurrentCluster =
    !currentSnapshot ||
    currentSnapshot.algorithm_cluster_id === algorithmClusterId;
  const callbackSnapshot = snapshotMatchesCurrentCluster && !restoringBatch && !restoreUnresolved
    ? currentSnapshot
    : null;
  const selectedDataset =
    datasets.find((dataset) => dataset.dataset_id === selectedDatasetId) ?? null;
  const isBatchActive =
    currentSnapshot?.status === "queued" ||
    currentSnapshot?.status === "running";
  useLayoutEffect(() => {
    onRestoringChange?.(restoringBatch || restoreUnresolved);
  }, [onRestoringChange, restoringBatch, restoreUnresolved]);
  useEffect(() => {
    onBusyChange?.(launching || restoringBatch || restoreUnresolved || isBatchActive);
  }, [launching, restoringBatch, restoreUnresolved, isBatchActive, onBusyChange]);
  const pollingStopped = Boolean(pollError) && isBatchActive;
  const snapshotMatchesExpectedDataset =
    !currentSnapshot ||
    !selectedDataset ||
    currentSnapshot.dataset_id === selectedDataset.dataset_id;
  const failedCases =
    currentSnapshot?.cases?.filter((entry) => entry.status === "failed") ?? [];
  const displayedBatchStatus =
    currentSnapshot?.status === "completed" && failedCases.length > 0
      ? "failed"
      : currentSnapshot?.status;
  const modeReportCount = currentSnapshot?.mode_reports?.length ?? 0;
  const caseStatusById = useMemo(
    () =>
      new Map(
        (currentSnapshot?.cases ?? []).map((entry) => [
          entry.case_id,
          entry.status,
        ]),
      ),
    [currentSnapshot?.cases],
  );

  useEffect(() => {
    const controller = new AbortController();
    setDatasetsLoading(true);
    setDatasetError("");
    fetch(DATASETS_URL, {
      signal: controller.signal,
      cache: "no-store",
    })
      .then((response) =>
        readApiJson(response, isDatasetList, "读取数据集"),
      )
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
          setDatasetError(errorMessage(reason, "读取数据集失败"));
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setDatasetsLoading(false);
      });
    return () => controller.abort();
  }, [datasetRefresh]);

  useEffect(() => {
    return () => {
      importAbortRef.current?.abort();
      launchAbortRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    const preferredBatchId = restoreTargetRef.current || readLatestBatchId();
    const controller = new AbortController();
    setRestoreError("");
    setHistoryError("");
    setRestoreUnresolved(false);
    setRestoringBatch(true);

    const restore = async () => {
      try {
        let history: AcceptanceBatchSummary[] | null = null;
        try {
          const response = await fetch(`${BATCHES_URL}?limit=20`, {
            signal: controller.signal,
            cache: "no-store",
          });
          history = await readApiJson(response, isBatchHistory, "读取历史批次");
          if (controller.signal.aborted) return;
          setRecentBatches(history);
        } catch (reason) {
          if (controller.signal.aborted) return;
          setHistoryError(errorMessage(reason, "读取历史批次失败"));
        }

        let batchId = /^[a-f0-9]{32}$/.test(preferredBatchId)
          ? preferredBatchId
          : (history?.[0]?.batch_id ?? "");
        if (!batchId) {
          if (history === null) throw new Error("历史批次暂时不可用，请刷新批次列表重试。");
          return;
        }

        const fetchBatch = async (id: string) => {
          const response = await fetch(`${BATCHES_URL}/${encodeURIComponent(id)}`, {
            signal: controller.signal,
            cache: "no-store",
          });
          const next = await readApiJson(response, isBatchSnapshot, "恢复批次结果");
          if (next.batch_id !== id) throw new Error("恢复的批次身份与请求不一致");
          return next;
        };
        let next: AcceptanceBatchSnapshot;
        try {
          next = await fetchBatch(batchId);
        } catch (reason) {
          if (!(reason instanceof ApiResponseError) || reason.status !== 404) throw reason;
          const fallback = history?.find((row) => row.batch_id !== batchId);
          if (!fallback) {
            clearLatestBatchId();
            setRestoreError("保存的批次不存在，请从历史批次中重新选择。");
            return;
          }
          batchId = fallback.batch_id;
          next = await fetchBatch(batchId);
        }
        if (controller.signal.aborted) return;
        setSnapshot(next);
        setSelectedDatasetId(next.dataset_id);
        storeLatestBatchId(next.batch_id);
        setRunError("");
        setPollError("");
        restoreTargetRef.current = "";
        setPage(0);
      } catch (reason) {
        if (controller.signal.aborted) return;
        setRestoreUnresolved(true);
        setRestoreError(errorMessage(reason, "恢复最近批次失败"));
      } finally {
        if (!controller.signal.aborted) setRestoringBatch(false);
      }
    };
    void restore();
    return () => controller.abort();
  }, [restoreRefresh]);

  useEffect(() => {
    if (!currentSnapshot || !isBatchActive) return;
    const controller = new AbortController();
    const batchId = currentSnapshot.batch_id;
    const clusterId = currentSnapshot.algorithm_cluster_id;
    let timer: number | undefined;
    let disposed = false;

    const poll = async () => {
      try {
        const response = await fetch(
          `${BATCHES_URL}/${encodeURIComponent(batchId)}`,
          {
            signal: controller.signal,
            cache: "no-store",
          },
        );
        const next = await readApiJson(
          response,
          isBatchSnapshot,
          "查询批次进度",
        );
        if (disposed) return;
        if (
          next.batch_id !== batchId ||
          next.algorithm_cluster_id !== clusterId
        ) {
          throw new Error("批次轮询返回的批次或算法簇身份不一致");
        }
        setPollError("");
        setSnapshot(next);
        if (next.status === "queued" || next.status === "running") {
          timer = window.setTimeout(poll, 1000);
        }
      } catch (reason) {
        if (!disposed && !controller.signal.aborted) {
          setPollError(
            errorMessage(reason, "批次进度查询失败，已停止自动轮询"),
          );
        }
      }
    };

    timer = window.setTimeout(poll, 1000);
    return () => {
      disposed = true;
      if (timer !== undefined) window.clearTimeout(timer);
      controller.abort();
    };
  }, [
    currentSnapshot?.batch_id,
    currentSnapshot?.status,
    isBatchActive,
    pollRetry,
  ]);

  useEffect(() => {
    const callbackAppeared =
      callbackRef.current == null && onBatchChange != null;
    callbackRef.current = onBatchChange;
    if (!onBatchChange) return;
    if (lastNotifiedRef.current === callbackSnapshot && !callbackAppeared) {
      return;
    }
    lastNotifiedRef.current = callbackSnapshot;
    onBatchChange(callbackSnapshot);
  }, [callbackSnapshot, onBatchChange]);

  useEffect(() => {
    resetFilters();
    if (currentSnapshot && revealBatchRef.current === currentSnapshot.batch_id) {
      revealBatchRef.current = "";
      resultRef.current?.scrollIntoView({ block: "start" });
      resultRef.current?.focus({ preventScroll: true });
    }
  }, [currentSnapshot?.batch_id]);

  const historyChoices = useMemo(() => {
    const rows = new Map(recentBatches.map((row) => [row.batch_id, row]));
    if (currentSnapshot) rows.set(currentSnapshot.batch_id, currentSnapshot);
    return [...rows.values()].sort((a, b) =>
      Date.parse(b.created_at) - Date.parse(a.created_at) || b.batch_id.localeCompare(a.batch_id),
    );
  }, [recentBatches, currentSnapshot]);

  const filteredItems = useMemo(() => {
    const normalizedQuery = query.trim().toLocaleLowerCase();
    return (currentSnapshot?.items ?? []).filter((item) => {
      if (modeFilter !== "all" && itemMode(item) !== modeFilter) return false;
      if (statusFilter !== "all" && item.status !== statusFilter) return false;
      return !normalizedQuery || itemSearchText(item).includes(normalizedQuery);
    });
  }, [currentSnapshot?.items, modeFilter, query, statusFilter]);

  const pageCount = Math.max(1, Math.ceil(filteredItems.length / PAGE_SIZE));
  const pageIndex = Math.min(page, pageCount - 1);
  const visibleItems = filteredItems.slice(
    pageIndex * PAGE_SIZE,
    (pageIndex + 1) * PAGE_SIZE,
  );
  const progressTotal = currentSnapshot?.progress.total_items ?? 0;
  const progressDone = currentSnapshot?.progress.completed_items ?? 0;
  const progressPercent =
    progressTotal > 0
      ? Math.min(100, Math.max(0, (progressDone / progressTotal) * 100))
      : 0;
  const selectedDatasetIsComplete = selectedDataset?.item_count === 336;
  const canLaunch =
    !externalRunning &&
    Boolean(selectedDataset) &&
    Boolean(algorithmClusterId) &&
    selectedDatasetIsComplete &&
    !launching &&
    !isBatchActive &&
    !restoringBatch &&
    !restoreUnresolved;
  const canShowReport =
    currentSnapshot?.status === "completed" ||
    currentSnapshot?.status === "failed" ||
    currentSnapshot?.status === "cancelled";
  const showingPreviousSnapshot =
    Boolean(currentSnapshot) &&
    (launching || restoringBatch || Boolean(runError) || Boolean(pollError) || Boolean(restoreError));

  async function handleImport(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!archive || datasetsLoading || importAbortRef.current) return;
    const controller = new AbortController();
    importAbortRef.current = controller;
    setImporting(true);
    setImportError("");
    setImportSuccess("");
    try {
      const form = new FormData();
      form.set("archive", archive);
      const response = await fetch(DATASETS_URL, {
        method: "POST",
        body: form,
        signal: controller.signal,
      });
      const summary = await readApiJson(
        response,
        isDatasetSummary,
        "导入数据集",
      );
      if (controller.signal.aborted) return;
      setDatasets((current) => [
        summary,
        ...current.filter((row) => row.dataset_id !== summary.dataset_id),
      ]);
      setSelectedDatasetId(summary.dataset_id);
      setImportSuccess(
        `ZIP 校验通过：${INTEGER_FORMAT.format(summary.item_count)} 项、${INTEGER_FORMAT.format(summary.case_count)} 个条件组；manifest ${shortHash(summary.manifest_sha256)}`,
      );
      setArchive(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch (reason) {
      if (!controller.signal.aborted) {
        setImportError(errorMessage(reason, "ZIP 导入失败"));
      }
    } finally {
      if (importAbortRef.current === controller) {
        importAbortRef.current = null;
        if (!controller.signal.aborted) setImporting(false);
      }
    }
  }

  function handleArchiveChange(file: File | null) {
    setArchive(file);
    setImportError("");
    setImportSuccess("");
    if (!file) return;
    if (!isZipArchive(file)) {
      setArchive(null);
      setImportError("仅支持 .zip 数据集归档。");
      return;
    }
    if (file.size > MAX_ARCHIVE_BYTES) {
      setArchive(null);
      setImportError("ZIP 超过 256 MiB，服务端不会接收。");
    }
  }

  async function handleLaunch() {
    if (!selectedDataset || !algorithmClusterId || !canLaunch) return;
    if (launchAbortRef.current) return;
    const requestedDatasetId = selectedDataset.dataset_id;
    const requestedClusterId = algorithmClusterId;
    const controller = new AbortController();
    launchAbortRef.current = controller;
    setLaunching(true);
    setRunError("");
    setPollError("");
    setRestoreError("");
    try {
      const response = await fetch(BATCHES_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_id: requestedDatasetId,
          algorithm_cluster_id: requestedClusterId,
        }),
        signal: controller.signal,
      });
      const next = await readApiJson(
        response,
        isBatchSnapshot,
        "创建验收批次",
      );
      if (
        next.dataset_id !== requestedDatasetId ||
        next.algorithm_cluster_id !== requestedClusterId
      ) {
        throw new Error("创建批次返回的数据集或算法簇与请求不一致");
      }
      if (controller.signal.aborted) return;
      revealBatchRef.current = next.batch_id;
      setSnapshot(next);
      storeLatestBatchId(next.batch_id);
      setRecentBatches((rows) => [next, ...rows.filter((row) => row.batch_id !== next.batch_id)]);
      setPage(0);
    } catch (reason) {
      if (!controller.signal.aborted) {
        setRunError(errorMessage(reason, "创建验收批次失败"));
      }
    } finally {
      if (launchAbortRef.current === controller) {
        launchAbortRef.current = null;
        if (!controller.signal.aborted) setLaunching(false);
      }
    }
  }

  function resetFilters() {
    setModeFilter("all");
    setStatusFilter("all");
    setQuery("");
    setPage(0);
  }

  function retryPolling() {
    if (!isBatchActive || pollError === "") return;
    setPollError("");
    setPollRetry((current) => current + 1);
  }

  return (
    <section
      className="acceptance-batch"
      aria-labelledby="acceptance-batch-title"
    >
      <header className="acceptance-batch-header">
        <div className="acceptance-batch-heading">
          <span className="acceptance-batch-eyebrow">
            REAL ACCEPTANCE BATCH
          </span>
          <h2 id="acceptance-batch-title">336 项真实批次验收</h2>
          <p>
            导入受校验的数据集，以当前算法簇创建 8 模式 × 42 项 = 336 项批次；
            模板提供 106 个基准条件组，同一条件可按测项拆成更多 case，校验以 336 项覆盖为准。
            页面只展示服务端返回的逐项工程证据，
            正式 ITU-T 合规仍为未测。
          </p>
        </div>
        <div className="acceptance-batch-badges" aria-label="验收口径">
          <span>工程代理结果</span>
          <span>正式合规未测</span>
          <span>{algorithmClusterId || "未选择算法簇"}</span>
        </div>
      </header>

      <div className="acceptance-batch-history">
        <label htmlFor={historySelectId}>已保存的批次报告</label>
        <select
          id={historySelectId}
          value={currentSnapshot?.batch_id ?? ""}
          disabled={restoringBatch || launching || isBatchActive || historyChoices.length === 0}
          onChange={(event) => {
            restoreTargetRef.current = event.target.value;
            setRestoreRefresh((current) => current + 1);
          }}
        >
          {!currentSnapshot ? (
            <option value="">{restoringBatch ? "正在查找已保存的结果…" : "暂无已保存批次"}</option>
          ) : null}
          {historyChoices.map((row) => (
            <option key={row.batch_id} value={row.batch_id}>
              {formatDate(row.created_at)} · {row.algorithm_cluster_id} · {BATCH_STATUS_LABEL[row.status]} · {row.batch_id.slice(0, 8)}
            </option>
          ))}
        </select>
        <button
          className="button secondary"
          type="button"
          disabled={restoringBatch || launching || isBatchActive}
          onClick={() => setRestoreRefresh((current) => current + 1)}
        >
          <RefreshCw size={15} aria-hidden="true" />
          刷新批次列表
        </button>
        {historyError ? <p role="alert">{historyError}；已显示的结果会继续保留。</p> : null}
      </div>

      {currentSnapshot ? (
        <aside className="acceptance-batch-result-preview" aria-label="当前批次结果摘要">
          <div className="acceptance-batch-preview-heading">
            <div aria-live="polite">
              <strong>
                {showingPreviousSnapshot ? "保留的上次结果 · " : ""}
                {BATCH_STATUS_LABEL[displayedBatchStatus ?? currentSnapshot.status]}
              </strong>
              <p>
                条件组 {currentSnapshot.progress.completed_cases}/{currentSnapshot.progress.total_cases}
                {" · "}已执行 {progressDone}/{progressTotal} 项
                {" · "}{currentSnapshot.dataset_name}
              </p>
            </div>
            <div className="acceptance-batch-preview-actions">
              <button
                className="button primary"
                type="button"
                onClick={() => {
                  itemsRef.current?.scrollIntoView({ block: "start" });
                  itemsRef.current?.focus({ preventScroll: true });
                }}
              >
                <ExternalLink size={15} aria-hidden="true" />
                查看逐项结果
              </button>
              {canShowReport ? (
                <a
                  className="button secondary"
                  href={`${BATCHES_URL}/${encodeURIComponent(currentSnapshot.batch_id)}/report`}
                  download={`acceptance-${currentSnapshot.batch_id}.json`}
                >
                  <Download size={15} aria-hidden="true" />
                  下载报告
                </a>
              ) : null}
            </div>
          </div>
          <dl className="acceptance-batch-preview-counts">
            {([
              ["已计算", currentSnapshot.summary.computed],
              ["工程通过", currentSnapshot.summary.pass],
              ["工程失败", currentSnapshot.summary.fail],
              ["需复核", currentSnapshot.summary.review_required],
              ["不可测", currentSnapshot.summary.not_measured],
              ["无效", currentSnapshot.summary.invalid],
            ] as const).map(([label, value]) => (
              <div key={label}><dt>{label}</dt><dd>{INTEGER_FORMAT.format(value)}</dd></div>
            ))}
          </dl>
          <p className="acceptance-batch-preview-note">
            {pollError ? "同步已停止，当前保留上次成功快照。" : ""}
            {restoreError ? "结果恢复失败，当前保留上次成功快照。" : ""}
            {isBatchActive ? "执行中，数值随服务端进度更新。" : "执行结束表示计算流程结束；需复核和不可测不代表认证通过。"}
            {!snapshotMatchesCurrentCluster ? ` 该报告算法簇为 ${currentSnapshot.algorithm_cluster_id}，与当前选择不同。` : ""}
          </p>
        </aside>
      ) : null}

      <ol className="acceptance-batch-steps" aria-label="批次执行步骤">
        <li className="acceptance-batch-step">
          <div className="acceptance-batch-step-head">
            <span className="acceptance-batch-step-index" aria-hidden="true">
              1
            </span>
            <div>
              <h3>导入 ZIP</h3>
              <p>ZIP 包含清单及各条件的麦克风混合输入、干净近端参考、远端播放信号 / 回声参考、独立噪声输入；VCP 处理后的近端输出由 VCP 生成。</p>
            </div>
          </div>
          <form className="acceptance-batch-form" onSubmit={handleImport}>
            <label htmlFor={fileInputId}>数据集归档</label>
            <input
              ref={fileInputRef}
              id={fileInputId}
              type="file"
              accept=".zip,application/zip,application/x-zip-compressed"
              disabled={importing}
              onChange={(event) => {
                const file = event.target.files?.[0] ?? null;
                handleArchiveChange(file);
                if (file && (!isZipArchive(file) || file.size > MAX_ARCHIVE_BYTES)) {
                  event.currentTarget.value = "";
                }
              }}
            />
            <div className="acceptance-batch-file-meta">
              <FileArchive size={15} aria-hidden="true" />
              <span>
                {archive
                  ? `${archive.name} · ${formatBytes(archive.size)}`
                  : "选择包含 manifest.json 与 audio/ 的 ZIP"}
              </span>
            </div>
            <button
              className="button secondary"
              type="submit"
              disabled={!archive || importing || datasetsLoading}
            >
              <Upload size={15} aria-hidden="true" />
              {importing ? "正在导入并校验…" : "导入并校验 ZIP"}
            </button>
          </form>
          <p className="acceptance-batch-muted">MOS 片段至少 9.01 秒；带外项需高于被测频带两倍的采样率。多麦输入须在清单提供逐通道校准时延。</p>
          <p className="acceptance-batch-muted">analysis_start_s 仅定位 50/100 ms 测项和三档 RLR 线性分析的活动窗；MOS 与完整五路诊断始终使用完整片段。</p>
          {importError ? (
            <p className="acceptance-batch-inline-error" role="alert">
              {importError}
            </p>
          ) : null}
          {importSuccess ? (
            <p className="acceptance-batch-inline-success" role="status">
              <CheckCircle2 size={14} aria-hidden="true" />
              {importSuccess}
            </p>
          ) : null}
        </li>

        <li className="acceptance-batch-step">
          <div className="acceptance-batch-step-head">
            <span className="acceptance-batch-step-index" aria-hidden="true">
              2
            </span>
            <div>
              <h3>选择数据集</h3>
              <p>只列出通过完整性检查的已导入批次。</p>
            </div>
          </div>
          <div
            className="acceptance-batch-form"
            aria-busy={datasetsLoading}
          >
            <label htmlFor={datasetSelectId}>已导入数据集</label>
            <select
              id={datasetSelectId}
              value={selectedDatasetId}
              disabled={
                datasetsLoading ||
                datasets.length === 0 ||
                launching ||
                restoringBatch
              }
              onChange={(event) => {
                setSelectedDatasetId(event.target.value);
                setPage(0);
              }}
            >
              {datasets.length === 0 ? (
                <option value="">
                  {datasetsLoading ? "正在读取…" : "暂无完整数据集"}
                </option>
              ) : null}
              {datasets.map((dataset) => (
                <option key={dataset.dataset_id} value={dataset.dataset_id}>
                  {dataset.name} · {dataset.item_count} 项 /{" "}
                  {dataset.case_count} 条件组
                </option>
              ))}
            </select>
            {selectedDataset ? (
              <dl className="acceptance-batch-dataset-meta">
                <div>
                  <dt>测项覆盖</dt>
                  <dd>{selectedDataset.item_count} / 336</dd>
                </div>
                <div>
                  <dt>条件组</dt>
                  <dd>{INTEGER_FORMAT.format(selectedDataset.case_count)}</dd>
                </div>
                <div>
                  <dt>导入时间</dt>
                  <dd>{formatDate(selectedDataset.created_at)}</dd>
                </div>
                <div>
                  <dt>Manifest SHA-256</dt>
                  <dd title={selectedDataset.manifest_sha256}>
                    <code>{shortHash(selectedDataset.manifest_sha256)}</code>
                  </dd>
                </div>
              </dl>
            ) : (
              <p className="acceptance-batch-help">
                先导入 ZIP，或从服务端已有完整数据集中选择。
              </p>
            )}
            {selectedDataset && !selectedDatasetIsComplete ? (
              <p className="acceptance-batch-inline-error" role="alert">
                该数据集未覆盖 336 项，不能创建正式批次。
              </p>
            ) : null}
            <button
              className="button secondary"
              type="button"
              disabled={datasetsLoading || importing}
              onClick={() => setDatasetRefresh((current) => current + 1)}
            >
              <RefreshCw size={15} aria-hidden="true" />
              {datasetsLoading ? "正在刷新…" : "刷新数据集列表"}
            </button>
          </div>
          {datasetError ? (
            <p className="acceptance-batch-inline-error" role="alert">
              {datasetError}
            </p>
          ) : null}
        </li>

        <li className="acceptance-batch-step">
          <div className="acceptance-batch-step-head">
            <span className="acceptance-batch-step-index" aria-hidden="true">
              3
            </span>
            <div>
              <h3>准备清单</h3>
              <p>下载含 106 个基准条件组的规范模板，再补齐真实音频。</p>
            </div>
          </div>
          <div className="acceptance-batch-form">
            <a
              className="button secondary"
              href={TEMPLATE_URL}
              download="manifest.json"
            >
              <Download size={15} aria-hidden="true" />
              下载 manifest.json
            </a>
            <p className="acceptance-batch-help">
              模板的 106 个基准条件组可按测项拆成更多 case；数据集 case 数不固定，但必须覆盖全部 336 项，同一条件音频不得复制到多个业务模式。
            </p>
            <dl className="acceptance-batch-format-meta">
              <div>
                <dt>归档入口</dt>
                <dd>
                  <code>manifest.json</code>
                </dd>
              </div>
              <div>
                <dt>音频目录</dt>
                <dd>
                  <code>audio/&lt;case&gt;/</code>
                </dd>
              </div>
            </dl>
          </div>
        </li>

        <li className="acceptance-batch-step">
          <div className="acceptance-batch-step-head">
            <span className="acceptance-batch-step-index" aria-hidden="true">
              4
            </span>
            <div>
              <h3>执行当前算法簇</h3>
              <p>批次创建后每秒读取一次真实进度，不推测完成。</p>
            </div>
          </div>
          <div className="acceptance-batch-form">
            <span className="acceptance-batch-field-label">当前算法簇</span>
            <code className="acceptance-batch-cluster">
              {algorithmClusterId || "未选择算法簇"}
            </code>
            <button
              className="button primary"
              type="button"
              disabled={!canLaunch}
              onClick={() => void handleLaunch()}
            >
              <Play size={15} fill="currentColor" aria-hidden="true" />
              {restoringBatch
                ? "正在恢复最近批次…"
                : restoreUnresolved
                  ? "最近批次状态待确认…"
                : launching
                ? "正在创建批次…"
                : isBatchActive
                  ? "批次执行中…"
                  : "一键执行 336 项"}
            </button>
            <p className="acceptance-batch-help">
              按钮会锁定到当前批次结束，避免重复创建。
            </p>
          </div>
        </li>
      </ol>

      {restoringBatch ? (
        <div className="acceptance-batch-restore" role="status">
          <RefreshCw size={17} aria-hidden="true" />
          <div>
            <strong>正在恢复最近批次</strong>
            <p>已读取本地 batch_id，正在通过 GET 接口恢复真实快照。</p>
          </div>
        </div>
      ) : null}

      {restoreError ? (
        <div className="acceptance-batch-alert" role="alert">
          <AlertCircle size={18} aria-hidden="true" />
          <div>
            <strong>最近批次恢复失败</strong>
            <p>{restoreError}</p>
            <button
              className="button secondary"
              type="button"
              onClick={() => setRestoreRefresh((current) => current + 1)}
            >
              <RefreshCw size={15} aria-hidden="true" />
              重新检查恢复记录
            </button>
          </div>
        </div>
      ) : null}

      {runError ? (
        <div className="acceptance-batch-alert" role="alert">
          <AlertCircle size={18} aria-hidden="true" />
          <div>
            <strong>本次批次请求失败</strong>
            <p>{runError}</p>
            {currentSnapshot ? (
              <p>
                下方仍显示最后一次成功获取的批次快照，不代表本次请求完成或通过。
              </p>
            ) : null}
          </div>
        </div>
      ) : null}

      {!currentSnapshot ? (
        restoringBatch ? (
          <div className="acceptance-batch-empty" role="status">
            <RefreshCw size={24} aria-hidden="true" />
            <div>
              <h3>正在恢复最近批次</h3>
              <p>恢复完成前不会创建新批次，也不会向父流程图回传旧统计。</p>
            </div>
          </div>
        ) : (
          <div className="acceptance-batch-empty">
            <FileJson size={24} aria-hidden="true" />
            <div>
              <h3>尚未创建真实批次</h3>
              <p>
                选择完整数据集并执行后，这里会显示服务端返回的进度、统计和逐项证据。
                工程代理批次不会自动生成正式合规结论。
              </p>
            </div>
          </div>
        )
      ) : (
        <>
          <section
            className="acceptance-batch-result"
            aria-labelledby="acceptance-batch-result-title"
            ref={resultRef}
            tabIndex={-1}
          >
            <header className="acceptance-batch-result-header">
              <div>
                <span className="acceptance-batch-eyebrow">
                  SERVER SNAPSHOT
                </span>
                <h3 id="acceptance-batch-result-title">
                  {showingPreviousSnapshot ? "上次快照 · " : ""}
                  {BATCH_STATUS_LABEL[
                    displayedBatchStatus ?? currentSnapshot.status
                  ]}
                </h3>
                <p>
                  {currentSnapshot.dataset_name} ·{" "}
                  {currentSnapshot.algorithm_cluster_id} ·{" "}
                  {formatDate(currentSnapshot.created_at)}
                </p>
              </div>
              <div className="acceptance-batch-result-actions">
                <span
                  className={`acceptance-batch-run-status is-${
                    displayedBatchStatus ?? currentSnapshot.status
                  }${
                    pollingStopped ? " is-poll-stopped" : ""
                  }`}
                  role="status"
                  aria-live="polite"
                >
                  {isBatchActive ? <Clock3 size={14} aria-hidden="true" /> : null}
                  {isBatchActive
                    ? pollingStopped
                      ? "轮询已停止 · 可重新查询"
                      : `自动轮询 · 每 1 秒 · ${formatValue(progressPercent)}%`
                    : BATCH_STATUS_LABEL[
                        displayedBatchStatus ?? currentSnapshot.status
                      ]}
                </span>
                {canShowReport ? (
                  <a
                    className="button secondary"
                    href={`${BATCHES_URL}/${encodeURIComponent(
                      currentSnapshot.batch_id,
                    )}/report`}
                    download={`acceptance-${currentSnapshot.batch_id}.json`}
                  >
                    <Download size={15} aria-hidden="true" />
                    下载批次报告
                  </a>
                ) : (
                  <span className="acceptance-batch-report-pending">
                    报告在批次结束后可下载
                  </span>
                )}
              </div>
            </header>

            <div className="acceptance-batch-progress">
              <div className="acceptance-batch-progress-copy">
                <strong>
                  条件组 {currentSnapshot.progress.completed_cases} /{" "}
                  {currentSnapshot.progress.total_cases}
                </strong>
                <span>
                  逐项 {currentSnapshot.progress.completed_items} /{" "}
                  {currentSnapshot.progress.total_items}
                </span>
              </div>
              <progress
                max={Math.max(1, progressTotal)}
                value={Math.min(progressDone, Math.max(1, progressTotal))}
                aria-label="逐项批次完成进度"
              >
                {formatValue(progressPercent)}%
              </progress>
            </div>

            <div className="acceptance-batch-scope" role="note">
              <ShieldAlert size={19} aria-hidden="true" />
              <div>
                <strong>工程代理批次 · 正式合规未测</strong>
                <p>
                  “工程通过”只表示当前批次中的工程计算满足代理判据；
                  需复核、不可测与无效项仍必须保留，不能作为 ITU-T 正式合规、
                  声学台架认证或客户验收通过证明。
                </p>
              </div>
            </div>

            <dl className="acceptance-batch-summary">
              {(
                [
                  ["总项", currentSnapshot.summary.total],
                  ["已计算（非空值）", currentSnapshot.summary.computed],
                  ["工程通过", currentSnapshot.summary.pass],
                  ["工程失败", currentSnapshot.summary.fail],
                  ["需复核", currentSnapshot.summary.review_required],
                  ["不可测", currentSnapshot.summary.not_measured],
                  ["无效", currentSnapshot.summary.invalid],
                ] as const
              ).map(([label, value], index) => (
                <div
                  className={index === 1 ? "is-computed" : ""}
                  key={label}
                >
                  <dt>{label}</dt>
                  <dd>{INTEGER_FORMAT.format(value)}</dd>
                </div>
              ))}
            </dl>

            <div className="acceptance-batch-identity">
              <span>
                批次 <code>{currentSnapshot.batch_id}</code>
              </span>
              <span>
                数据集 <code>{currentSnapshot.dataset_id}</code>
              </span>
              <span>
                模式报告{" "}
                <code>
                  {modeReportCount} / {FIVE_SIGNAL_MODES.length}
                </code>
              </span>
              {failedCases.length > 0 ? (
                <span>
                  失败条件组 <code>{failedCases.length}</code>
                </span>
              ) : null}
            </div>

            {!snapshotMatchesCurrentCluster ? (
              <div className="acceptance-batch-mismatch" role="status">
                <AlertCircle size={18} aria-hidden="true" />
                <div>
                  <strong>报告算法簇与当前算法簇不同</strong>
                  <p>
                    当前流程图使用 <code>{algorithmClusterId || "未选择"}</code>；
                    报告来自{" "}
                    <code>{currentSnapshot.algorithm_cluster_id}</code>。
                    旧报告继续保留供查看，但不会自动覆盖父流程图统计；切回对应算法簇后才会重新关联。
                  </p>
                </div>
              </div>
            ) : null}

            {showingPreviousSnapshot ? (
              <p className="acceptance-batch-stale" role="status">
                {launching
                  ? "正在创建新批次；当前仍为上一次成功快照。"
                  : "当前保留最后一次成功快照；本次请求或轮询失败，不能视为完成或通过。"}
              </p>
            ) : null}

            {pollError ? (
              <div className="acceptance-batch-alert" role="alert">
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

            {currentSnapshot.error ? (
              <div className="acceptance-batch-alert" role="alert">
                <AlertCircle size={18} aria-hidden="true" />
                <div>
                  <strong>服务端报告批次错误</strong>
                  <p>{currentSnapshot.error}</p>
                </div>
              </div>
            ) : null}

            {failedCases.length > 0 ? (
              <div className="acceptance-batch-alert" role="alert">
                <AlertCircle size={18} aria-hidden="true" />
                <div>
                  <strong>
                    存在 {failedCases.length} 个失败条件组，批次不能按完成通过处理
                  </strong>
                  <p>
                    已完成条件组的结果和逐项原因继续保留；失败条件组不会被改写为 completed。
                  </p>
                  {failedCases.slice(0, 3).map((entry) => (
                    <p key={entry.case_id}>
                      <code>{entry.case_id}</code>
                      {entry.error ? `：${entry.error}` : "：条件组执行失败"}
                    </p>
                  ))}
                </div>
              </div>
            ) : null}

            {!snapshotMatchesExpectedDataset ? (
              <p className="acceptance-batch-stale" role="status">
                当前快照来自另一数据集；切换数据集不会改写既有批次证据。
              </p>
            ) : null}
          </section>

          <section
            className="acceptance-batch-items"
            aria-labelledby="acceptance-batch-items-title"
            ref={itemsRef}
            tabIndex={-1}
          >
            <header className="acceptance-batch-items-header">
              <div>
                <span className="acceptance-batch-eyebrow">
                  ITEM EVIDENCE
                </span>
                <h3 id="acceptance-batch-items-title">逐项工程结果</h3>
                <p>
                  按 8 个业务模式、状态或关键字筛选；每页固定 {PAGE_SIZE} 行。
                </p>
              </div>
              <span className="acceptance-batch-item-count">
                {INTEGER_FORMAT.format(filteredItems.length)} /{" "}
                {INTEGER_FORMAT.format(currentSnapshot.items.length)} 项
              </span>
            </header>

            <div className="acceptance-batch-filters">
              <label htmlFor={modeFilterId}>
                业务模式
                <select
                  id={modeFilterId}
                  value={modeFilter}
                  onChange={(event) => {
                    const mode = event.target.value as ModeFilter;
                    setModeFilter(mode);
                    if (mode !== "all") onSelectMode?.(mode);
                    setPage(0);
                  }}
                >
                  <option value="all">全部 8 模式</option>
                  {FIVE_SIGNAL_MODES.map((mode) => (
                    <option key={mode} value={mode}>
                      {mode}
                    </option>
                  ))}
                </select>
              </label>
              <label htmlFor={statusFilterId}>
                状态
                <select
                  id={statusFilterId}
                  value={statusFilter}
                  onChange={(event) => {
                    setStatusFilter(event.target.value as StatusFilter);
                    setPage(0);
                  }}
                >
                  <option value="all">全部状态</option>
                  {ITEM_STATUSES.map((status) => (
                    <option key={status} value={status}>
                      {ITEM_STATUS_LABEL[status]}
                    </option>
                  ))}
                </select>
              </label>
              <label className="acceptance-batch-search" htmlFor={searchId}>
                搜索
                <span>
                  <Search size={15} aria-hidden="true" />
                  <input
                    id={searchId}
                    type="search"
                    value={query}
                    placeholder="测项、条件、原因或 report ID"
                    onChange={(event) => {
                      setQuery(event.target.value);
                      setPage(0);
                    }}
                  />
                </span>
              </label>
              <button
                className="button secondary"
                type="button"
                onClick={resetFilters}
                disabled={
                  modeFilter === "all" &&
                  statusFilter === "all" &&
                  query.trim() === ""
                }
              >
                清除筛选
              </button>
            </div>

            {visibleItems.length > 0 ? (
              <div className="table-scroll">
                <table className="audio-table acceptance-batch-table">
                  <caption className="sr-only">
                    336 项批次逐项工程结果，第 {pageIndex + 1} 页
                  </caption>
                  <thead>
                    <tr>
                      <th scope="col">测项</th>
                      <th scope="col">条件</th>
                      <th scope="col">模型 / 模式</th>
                      <th scope="col">状态</th>
                      <th scope="col">数值 / 原因</th>
                      <th scope="col">证据 report ID</th>
                    </tr>
                  </thead>
                  <tbody>
                    {visibleItems.map((item) => {
                      const conditions = itemCondition(item);
                      const binding = item.evidence_binding;
                      return (
                        <tr key={item.item_id}>
                          <td>
                            <div className="acceptance-batch-item-name">
                              <code>{item.item_id}</code>
                              <strong>{item.test_name ?? "未命名测项"}</strong>
                              <span>{item.family ?? "未分类"}</span>
                            </div>
                          </td>
                          <td>
                            {conditions.length > 0 ? (
                              <div className="acceptance-batch-condition">
                                {conditions.map((condition) => (
                                  <span key={condition}>{condition}</span>
                                ))}
                              </div>
                            ) : (
                              <span className="acceptance-batch-muted">
                                未绑定条件
                              </span>
                            )}
                          </td>
                          <td>
                            <code className="acceptance-batch-mode">
                              {itemMode(item) || "未标注"}
                            </code>
                          </td>
                          <td>
                            <ItemStatusBadge status={item.status} />
                          </td>
                          <td>
                            <div className="acceptance-batch-value">
                              {item.value !== null ? (
                                <strong>
                                  {item.metric ? `${item.metric} ` : ""}
                                  {formatValue(item.value)}
                                </strong>
                              ) : null}
                              {item.reason ? <span>{item.reason}</span> : null}
                              {item.value === null && !item.reason ? (
                                <span>未返回数值或原因</span>
                              ) : null}
                            </div>
                          </td>
                          <td>
                            {binding ? (
                              <div className="acceptance-batch-evidence">
                                <code title={binding.report_id ?? undefined}>
                                  {binding.report_id ?? "report 待生成"}
                                </code>
                                <span title={binding.case_id}>
                                  case {binding.case_id}
                                </span>
                                {caseStatusById.get(binding.case_id) === "completed" ? <div className="acceptance-batch-artifact-links">
                                  <a href={batchCaseEvidenceUrl(currentSnapshot.batch_id, binding.case_id, "assessment")} download>{item.family === "rlr_linearity" ? "单档计算证据" : "计算报告"}</a>
                                  {item.family === "rlr_linearity" && item.value !== null ? <a href={`${BATCHES_URL}/${encodeURIComponent(currentSnapshot.batch_id)}/report`} download>三档汇总结果</a> : null}
                                  {outputRolesForDirection(binding.direction).map(role => <a
                                    key={role}
                                    href={batchCaseEvidenceUrl(currentSnapshot.batch_id, binding.case_id, `outputs/${role}`)}
                                    download
                                  >{OUTPUT_ROLE_LABELS[role]} WAV</a>)}
                                </div> : null}
                              </div>
                            ) : (
                              <span className="acceptance-batch-muted">
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
            ) : (
              <div className="acceptance-batch-items-empty" role="status">
                <FileJson size={20} aria-hidden="true" />
                <div>
                  <strong>
                    {currentSnapshot.items.length === 0
                      ? isBatchActive
                        ? "批次尚未返回逐项结果"
                        : "批次没有逐项结果"
                      : "没有匹配的逐项结果"}
                  </strong>
                  <p>
                    {currentSnapshot.items.length === 0
                      ? isBatchActive
                        ? "页面会每秒读取一次真实批次快照。"
                        : "请检查服务端批次状态与报告错误。"
                      : "调整业务模式、状态或搜索条件后重试。"}
                  </p>
                </div>
                {currentSnapshot.items.length > 0 ? (
                  <button
                    className="button secondary"
                    type="button"
                    onClick={resetFilters}
                  >
                    清除筛选
                  </button>
                ) : null}
              </div>
            )}

            <footer className="acceptance-batch-pagination">
              <p role="status">
                {filteredItems.length > 0
                  ? `第 ${pageIndex * PAGE_SIZE + 1}–${Math.min(
                      (pageIndex + 1) * PAGE_SIZE,
                      filteredItems.length,
                    )} 项 · 第 ${pageIndex + 1} / ${pageCount} 页`
                  : "当前筛选无结果"}
              </p>
              <div>
                <button
                  className="button secondary"
                  type="button"
                  disabled={pageIndex === 0}
                  onClick={() => setPage(Math.max(0, pageIndex - 1))}
                >
                  上一页
                </button>
                <button
                  className="button secondary"
                  type="button"
                  disabled={pageIndex >= pageCount - 1}
                  onClick={() =>
                    setPage(Math.min(pageCount - 1, pageIndex + 1))
                  }
                >
                  下一页
                </button>
              </div>
            </footer>
          </section>
        </>
      )}
    </section>
  );
}
