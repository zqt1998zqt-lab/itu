import {
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type ReactNode,
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
import ScopedAcousticTree from "./ScopedAcousticTree";
import ScopedAcousticWaveforms from "./ScopedAcousticWaveforms";
import ScopedDatasetVisualization from "./ScopedDatasetVisualization";
import {
  SCENE_SUBPAGES,
  SCENE_SUBPAGE_LABELS,
  SCOPED_BATCH_LABELS,
  SCOPED_STATUS_LABELS,
  isIdle20ConditionPreview,
  readScopedDatasetView,
  sceneSubpageFromHash,
  sceneSubpageHash,
  sceneSubpageRoute,
  type SceneSubpageId,
  type BackgroundNoiseScene,
  readScopedCatalog,
  readScopedDatasets,
  readScopedHistory,
  readScopedSnapshot,
  scopedItemCount,
  scopedMetricUnit,
  scopedValue,
  type ScopedBatchSummary,
  type ScopedCatalog,
  type ScopedDataset,
  type ScopedDatasetView,
  type ScopedItem,
  type ScopedResultItem,
  type ScopedSnapshot,
  type ScopedTestDefinition,
  type ScopedTestPageProps,
} from "./scoped-acoustic-model";
import { ACOUSTIC_OUTPUT_LABELS } from "../domain/acoustic-terminology.ts";
import "./basic-uplink-test.css";
import "./scoped-acoustic-test.css";

const DATASETS_URL = "/api/alignment/acceptance/datasets";
const BATCHES_URL = "/api/alignment/acceptance/batches";
const POLL_INTERVAL_MS = 2000;
const FULL_DATASET_ITEM_COUNT = 336;
const MAX_ARCHIVE_BYTES = 256 * 1024 * 1024;

function SceneSubpagePanel({
  enabled,
  active,
  label,
  children,
}: {
  enabled: boolean;
  active: boolean;
  label: string;
  children: ReactNode;
}) {
  if (!enabled) return <>{children}</>;
  return (
    <div
      className="scoped-acoustic-subpage-block"
      hidden={!active}
      role="group"
      aria-label={label}
    >
      {children}
    </div>
  );
}

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

const VCP_PRESETS = chainPresets.filter((preset) =>
  preset.profile.startsWith("vcp"),
);
const INTEGER_FORMAT = new Intl.NumberFormat("zh-CN");
const DATE_FORMAT = new Intl.DateTimeFormat("zh-CN", {
  timeZone: "Asia/Shanghai",
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
});

type Props = ScopedTestPageProps & {
  definition: ScopedTestDefinition;
};

type BatchStatus = keyof typeof SCOPED_BATCH_LABELS;
type ItemStatus = keyof typeof SCOPED_STATUS_LABELS;
type ContractLimit = number | string | null;
type OutputLeaf = "outputs/tx" | "outputs/rx";

type ExpectedSnapshotIdentity = {
  batchId?: string;
  datasetId?: string;
  algorithmClusterId?: string;
  mode?: FiveSignalMode;
  scene?: BackgroundNoiseScene;
};

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

function errorMessage(reason: unknown, fallback: string): string {
  return reason instanceof Error && reason.message
    ? reason.message
    : fallback;
}

function formatDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : DATE_FORMAT.format(date);
}

function formatLimit(value: ContractLimit | undefined): string {
  if (value === undefined || value === null || value === "") {
    return "未定义";
  }
  return typeof value === "number" ? scopedValue(value) : value;
}

function formatRange(
  lower: ContractLimit | undefined,
  upper: ContractLimit | undefined,
): string {
  if (
    (lower === undefined || lower === null || lower === "") &&
    (upper === undefined || upper === null || upper === "")
  ) {
    return "未定义";
  }
  return `${formatLimit(lower)} … ${formatLimit(upper)}`;
}

function formatBytes(value: number): string {
  if (value < 1024) return `${INTEGER_FORMAT.format(value)} B`;
  if (value < 1024 * 1024) {
    return `${scopedValue(value / 1024)} KiB`;
  }
  return `${scopedValue(value / (1024 * 1024))} MiB`;
}

function formatNullable(value: string | number | null | undefined): string {
  if (value === null || value === undefined || value === "") return "—";
  return typeof value === "number" ? scopedValue(value) : String(value);
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

function storageKey(definition: ScopedTestDefinition): string {
  return `itu.alignment.scoped-acoustic.${definition.route}.latest-batch-id.v1`;
}

function definitionApiRoute(definition: ScopedTestDefinition): string {
  return definition.apiRoute ?? definition.route;
}

function readStoredBatchId(key: string): string {
  if (typeof window === "undefined") return "";
  try {
    const value = window.localStorage.getItem(key)?.trim() ?? "";
    return /^[a-f0-9]{32}$/.test(value) ? value : "";
  } catch {
    return "";
  }
}

function storeBatchId(key: string, batchId: string) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(key, batchId);
  } catch {
    // Storage can be unavailable in privacy-restricted contexts.
  }
}

function batchCaseUrl(
  batchId: string,
  caseId: string,
  leaf: "assessment" | OutputLeaf,
): string {
  return `${BATCHES_URL}/${encodeURIComponent(
    batchId,
  )}/cases/${encodeURIComponent(caseId)}/${leaf}`;
}

function asBatchSummary(snapshot: ScopedSnapshot): ScopedBatchSummary {
  return {
    batch_id: snapshot.batch_id,
    dataset_id: snapshot.dataset_id,
    dataset_name: snapshot.dataset_name,
    algorithm_cluster_id: snapshot.algorithm_cluster_id,
    test_scope: snapshot.test_scope,
    selected_mode: snapshot.selected_mode,
    created_at: snapshot.created_at,
    status: snapshot.status,
    progress: snapshot.progress,
    summary: snapshot.summary,
  };
}

function sameItemIdentity(
  snapshot: ScopedSnapshot,
  catalog: ScopedCatalog,
): boolean {
  if (
    snapshot.selected_mode !== catalog.mode ||
    snapshot.test_scope !== catalog.test_scope
  ) {
    return true;
  }
  const reportIds = snapshot.items.map((item) => item.item_id).sort();
  const catalogIds = catalog.items.map((item) => item.id).sort();
  return (
    reportIds.length === catalogIds.length &&
    reportIds.every((id, index) => id === catalogIds[index])
  );
}

function assertSnapshotCatalogIdentity(
  snapshot: ScopedSnapshot,
  catalog: ScopedCatalog | null,
): ScopedSnapshot {
  if (catalog && !sameItemIdentity(snapshot, catalog)) {
    throw new Error(
      "报告逐项身份与当前合同目录不一致；拒绝把不同合同行显示为本次结果",
    );
  }
  return snapshot;
}

async function fetchScopedSnapshot(
  batchId: string,
  definition: ScopedTestDefinition,
  signal: AbortSignal,
  operation: string,
  expected: ExpectedSnapshotIdentity = {},
): Promise<ScopedSnapshot> {
  const body = await requestJson(
    `${BATCHES_URL}/${encodeURIComponent(batchId)}`,
    { signal },
    operation,
  );
  return readScopedSnapshot(body, definition.scope, {
    batchId,
    scene: definition.scene,
    ...expected,
  });
}

function familyGroup(
  definition: ScopedTestDefinition,
  family: string,
): string {
  if (definition.scope !== "background_noise") return "工程测项";
  if (family === "cng_level" || family === "cng_spectrum") return "CNG";
  if (family === "g_mos" || family === "n_mos" || family === "s_mos") {
    return "GNS";
  }
  return "范围外";
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
      {SCOPED_STATUS_LABELS[status]}
    </span>
  );
}

function ResultItem({
  item,
  catalogItem,
  definition,
  snapshot,
  caseStatus,
  caseError,
}: {
  item: ScopedResultItem;
  catalogItem: ScopedItem | undefined;
  definition: ScopedTestDefinition;
  snapshot: ScopedSnapshot;
  caseStatus: string | undefined;
  caseError: string | undefined;
}) {
  const sourceId = item.id ?? catalogItem?.id ?? item.item_id;
  const sourceCell =
    item.source_cell ?? catalogItem?.source_cell ?? "未返回源行";
  const testName =
    item.test_name ?? catalogItem?.test_name ?? "未命名测项";
  const family = item.family ?? catalogItem?.family ?? "未分类";
  const contractUnit = item.unit ?? catalogItem?.unit ?? "";
  const lower =
    item.lower_limit !== undefined
      ? item.lower_limit
      : catalogItem?.lower_limit;
  const upper =
    item.upper_limit !== undefined
      ? item.upper_limit
      : catalogItem?.upper_limit;
  const binding = item.evidence_binding;
  const group = familyGroup(definition, family);
  const metricUnit = scopedMetricUnit(item.metric);
  const caseCompleted = caseStatus === "completed";
  const caseFailed = caseStatus === "failed";
  const outputLeaf: OutputLeaf =
    definition.outputKind === "rx" ? "outputs/rx" : "outputs/tx";
  const outputLabel = ACOUSTIC_OUTPUT_LABELS[definition.outputKind];
  const conditions = binding
    ? [
        ["模式", binding.mode],
        ["场景", binding.scene],
        ["音量", formatNullable(binding.volume)],
        [
          "窗口",
          binding.window_ms === null
            ? "—"
            : `${scopedValue(binding.window_ms)} ms`,
        ],
        ["方向", binding.direction],
        [
          "输入",
          binding.input_level_db === null
            ? "—"
            : `${scopedValue(binding.input_level_db)} dB`,
        ],
      ]
    : [];

  return (
    <tr>
      <td>
        <div className="basic-uplink-item-name">
          <code>{sourceId}</code>
          <strong>{testName}</strong>
          <span>
            {sourceCell} · {family}
          </span>
          <span>合同限值 {formatRange(lower, upper)}</span>
        </div>
      </td>
      <td>
        <div className="scoped-acoustic-family">
          <span
            className={`scoped-acoustic-family-chip is-${group.toLocaleLowerCase()}`}
          >
            {group}
          </span>
          <code>{family}</code>
        </div>
      </td>
      <td>
        {binding ? (
          <div className="scoped-acoustic-condition">
            {conditions.map(([label, value]) => (
              <span key={label}>
                <b>{label}</b>
                {value}
              </span>
            ))}
            <code>{binding.case_id}</code>
            <code>
              report {binding.report_id ?? "待生成"}
            </code>
          </div>
        ) : (
          <span className="basic-uplink-muted">未绑定条件组</span>
        )}
      </td>
      <td>
        <div className="basic-uplink-contract-unit">
          <span>{contractUnit || "—"}</span>
          <small>合同字段</small>
        </div>
      </td>
      <td>
        <div className="scoped-acoustic-limit">
          {formatRange(lower, upper)}
        </div>
      </td>
      <td>
        <div className="basic-uplink-value">
          <code>{item.metric ?? "计算指标未返回"}</code>
          <span>工程指标</span>
        </div>
      </td>
      <td>
        <div
          className={`scoped-acoustic-engineering-value${
            item.value === null ? " is-empty" : ""
          }`}
        >
          <strong>
            {item.value === null ? "无数值" : scopedValue(item.value)}
          </strong>
          <span>{metricUnit || "无量纲/未声明"}</span>
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
          <div className="basic-uplink-evidence scoped-acoustic-evidence">
            <code>{binding.report_id ?? "report 待生成"}</code>
            {caseCompleted ? (
              <div>
                <a
                  href={batchCaseUrl(
                    snapshot.batch_id,
                    binding.case_id,
                    "assessment",
                  )}
                  download
                  aria-label={`下载条件组 ${binding.case_id} 报告`}
                >
                  条件组报告
                </a>
                <a
                  href={batchCaseUrl(
                    snapshot.batch_id,
                    binding.case_id,
                    outputLeaf,
                  )}
                  download
                  aria-label={`下载条件组 ${binding.case_id} ${outputLabel}`}
                >
                  {outputLabel}
                </a>
              </div>
            ) : caseFailed ? (
              <div className="scoped-acoustic-case-failure" role="status">
                <strong>条件组失败</strong>
                <span>{caseError || "服务端未返回失败原因"}</span>
              </div>
            ) : (
              <span>
                {caseStatus
                  ? `条件组 ${caseStatus}，暂无产物`
                  : "条件组完成后可下载报告与输出"}
              </span>
            )}
          </div>
        ) : (
          <span className="basic-uplink-muted">未绑定证据</span>
        )}
      </td>
    </tr>
  );
}

export default function ScopedAcousticTest({
  definition,
  algorithmClusterId,
  mode,
  onModeChange,
  onAlgorithmClusterChange,
  externalRunning = false,
  onBusyChange,
  pageVisible = true,
}: Props) {
  const uid = useId().replace(/[^A-Za-z0-9_-]/g, "");
  const idFor = (name: string) =>
    `scoped-acoustic-${definition.scope}-${uid}-${name}`;
  const titleId = idFor("title");
  const resultTitleId = idFor("result-title");
  const itemsTitleId = idFor("items-title");
  const catalogTitleId = idFor("catalog-title");
  const modeSelectId = idFor("mode");
  const clusterSelectId = idFor("cluster");
  const datasetSelectId = idFor("dataset");
  const historySelectId = idFor("history");
  const archiveInputId = idFor("archive");
  const isScenePage = Boolean(definition.scene);
  const datasetsUrl = useMemo(() => {
    const params = new URLSearchParams();
    if (definition.scene) {
      params.set("scope", "background_noise");
      params.set("mode", mode);
      params.set("scene", definition.scene);
    }
    const query = params.toString();
    return query ? `${DATASETS_URL}?${query}` : DATASETS_URL;
  }, [definition.scene, mode]);

  const expectedItemCount = scopedItemCount(
    definition.scope,
    mode,
    definition.scene,
  );
  const idle20Preview = isIdle20ConditionPreview(definition, mode);
  const latestStorageKey = storageKey(definition);

  const [catalog, setCatalog] = useState<ScopedCatalog | null>(null);
  const [catalogLoading, setCatalogLoading] = useState(true);
  const [catalogError, setCatalogError] = useState("");
  const [catalogRefresh, setCatalogRefresh] = useState(0);
  const [catalogOpen, setCatalogOpen] = useState(true);

  const [datasets, setDatasets] = useState<ScopedDataset[]>([]);
  const [datasetView, setDatasetView] = useState<ScopedDatasetView | null>(null);
  const [datasetsLoading, setDatasetsLoading] = useState(true);
  const [datasetError, setDatasetError] = useState("");
  const [datasetRefresh, setDatasetRefresh] = useState(0);
  const [selectedDatasetId, setSelectedDatasetId] = useState("");
  const [archive, setArchive] = useState<File | null>(null);
  const [importing, setImporting] = useState(false);
  const [importError, setImportError] = useState("");
  const [importSuccess, setImportSuccess] = useState("");

  const [historyRows, setHistoryRows] = useState<ScopedBatchSummary[]>([]);
  const [historyError, setHistoryError] = useState("");
  const [historySelection, setHistorySelection] = useState("");
  const [restoreNonce, setRestoreNonce] = useState(0);
  const [restoring, setRestoring] = useState(true);
  const [restoreError, setRestoreError] = useState("");

  const [snapshot, setSnapshot] = useState<ScopedSnapshot | null>(null);
  const [launching, setLaunching] = useState(false);
  const [runError, setRunError] = useState("");
  const [pollError, setPollError] = useState("");
  const [pollRetry, setPollRetry] = useState(0);
  const [sceneSubpage, setSceneSubpage] = useState<SceneSubpageId>(() =>
    sceneSubpageFromHash(window.location.hash, definition.route),
  );
  const [waveformSource, setWaveformSource] = useState<"execution" | "extended">("extended");

  useEffect(() => {
    if (!isScenePage) return;
    const syncSubpage = () => {
      const hash = window.location.hash;
      if (sceneSubpageRoute(hash) !== definition.route) return;
      const next = sceneSubpageFromHash(hash, definition.route);
      setSceneSubpage(next);
      const canonicalHash = sceneSubpageHash(definition.route, next);
      if (hash !== canonicalHash) {
        window.history.replaceState(null, "", canonicalHash);
      }
    };
    syncSubpage();
    window.addEventListener("hashchange", syncSubpage);
    window.addEventListener("popstate", syncSubpage);
    return () => {
      window.removeEventListener("hashchange", syncSubpage);
      window.removeEventListener("popstate", syncSubpage);
    };
  }, [definition.route, isScenePage]);

  function showSceneSubpage(next: SceneSubpageId) {
    setSceneSubpage(next);
    const nextHash = sceneSubpageHash(definition.route, next);
    if (window.location.hash === nextHash) return;
    window.history.pushState(null, "", nextHash);
  }

  const archiveInputRef = useRef<HTMLInputElement>(null);
  const importAbortRef = useRef<AbortController | null>(null);
  const launchAbortRef = useRef<AbortController | null>(null);
  const restoreAbortRef = useRef<AbortController | null>(null);
  const restoreTargetRef = useRef("");
  const snapshotRef = useRef<ScopedSnapshot | null>(null);
  const catalogRef = useRef<ScopedCatalog | null>(null);

  snapshotRef.current = snapshot;
  catalogRef.current = catalog;

  const selectedDataset =
    datasets.find(
      (dataset) => dataset.dataset_id === (datasetView?.datasets[0]?.dataset_id ?? selectedDatasetId),
    ) ?? null;
  const currentDatasetId = isScenePage
    ? datasetView?.datasets[0]?.dataset_id ?? ""
    : selectedDatasetId;
  const selectedDatasetMissing =
    Boolean(currentDatasetId) && selectedDataset === null;
  const selectedDatasetIsComplete =
    selectedDataset?.item_count === FULL_DATASET_ITEM_COUNT;
  const selectedDatasetHasData =
    Boolean(selectedDataset) && (selectedDataset?.case_count ?? 0) > 0;
  const selectedDatasetReady =
    selectedDatasetIsComplete && selectedDatasetHasData;
  const algorithmIsSupported = VCP_PRESETS.some(
    (preset) => preset.id === algorithmClusterId,
  );
  const isBatchActive =
    snapshot?.status === "queued" || snapshot?.status === "running";
  const controlsLocked =
    externalRunning ||
    launching ||
    importing ||
    restoring ||
    Boolean(isBatchActive);
  const catalogReady =
    catalog?.test_scope === definition.scope &&
    catalog.mode === mode &&
    catalog.item_count === expectedItemCount &&
    catalog.items.length === expectedItemCount;
  const canStart =
    !controlsLocked &&
    !datasetError &&
    !datasetsLoading &&
    !catalogLoading &&
    Boolean(catalogReady) &&
    Boolean(selectedDatasetReady) &&
    Boolean(algorithmClusterId) &&
    algorithmIsSupported;

  const catalogIdentityError = useMemo(() => {
    if (!snapshot || !catalog) return "";
    return sameItemIdentity(snapshot, catalog)
      ? ""
      : "报告逐项身份与当前合同目录不一致；结果已隔离，请刷新目录后恢复历史。";
  }, [catalog, snapshot]);

  const reportModeMismatch =
    Boolean(snapshot) && snapshot?.selected_mode !== mode;
  const reportClusterMismatch =
    Boolean(snapshot) &&
    snapshot?.algorithm_cluster_id !== algorithmClusterId;
  const reportDatasetMismatch =
    !isScenePage &&
    Boolean(snapshot) &&
    Boolean(currentDatasetId) &&
    snapshot?.dataset_id !== currentDatasetId;
  const hasConfigurationMismatch =
    reportModeMismatch ||
    reportClusterMismatch ||
    reportDatasetMismatch;
  const showingPreviousSnapshot =
    Boolean(snapshot) &&
    (launching ||
      restoring ||
      Boolean(runError) ||
      Boolean(pollError) ||
      Boolean(restoreError) ||
      Boolean(historyError));

  const treeSnapshot = catalogIdentityError ? null : snapshot;

  const historyChoices = useMemo(() => {
    const rows = new Map<string, ScopedBatchSummary>();
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
  const caseErrorById = useMemo(
    () =>
      new Map(
        (snapshot?.cases ?? []).map((entry) => [
          entry.case_id,
          entry.error ?? "",
        ]),
      ),
    [snapshot?.cases],
  );
  const reportCatalogById = useMemo(() => {
    if (!catalog || catalog.mode !== snapshot?.selected_mode) {
      return new Map<string, ScopedItem>();
    }
    return new Map(catalog.items.map((item) => [item.id, item]));
  }, [catalog, snapshot?.selected_mode]);

  const progressTotal = snapshot?.progress.total_items ?? 0;
  const progressDone = snapshot?.progress.completed_items ?? 0;
  const progressPercent =
    progressTotal > 0
      ? Math.min(100, Math.max(0, (progressDone / progressTotal) * 100))
      : 0;
  const failedCases =
    snapshot?.cases.filter((entry) => entry.status === "failed") ?? [];
  const canDownloadReport =
    snapshot?.status === "completed" ||
    snapshot?.status === "failed" ||
    snapshot?.status === "cancelled";
  const outputLeaf: OutputLeaf =
    definition.outputKind === "rx" ? "outputs/rx" : "outputs/tx";
  const outputLabel = ACOUSTIC_OUTPUT_LABELS[definition.outputKind];

  useEffect(() => {
    const controller = new AbortController();
    setCatalog(null);
    setCatalogLoading(true);
    setCatalogError("");
    requestJson(
      `/api/alignment/acceptance/${definitionApiRoute(definition)}/catalog?mode=${encodeURIComponent(
        mode,
      )}`,
      { signal: controller.signal },
      `读取${definition.title}目录`,
    )
      .then((body) =>
        readScopedCatalog(
          body,
          definition.scope,
          mode,
          definition.scene,
        ),
      )
      .then((next) => {
        if (!controller.signal.aborted) setCatalog(next);
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) {
          setCatalogError(
            errorMessage(reason, `${definition.title}目录读取失败`),
          );
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setCatalogLoading(false);
      });
    return () => controller.abort();
  }, [
    catalogRefresh,
    definition.apiRoute,
    definition.route,
    definition.scope,
    definition.scene,
    definition.title,
    mode,
  ]);

  useEffect(() => {
    const controller = new AbortController();
    setDatasetsLoading(true);
    setDatasetError("");
    if (definition.scene) {
      setDatasetView(null);
      setDatasets([]);
    }
    requestJson(
      datasetsUrl,
      { signal: controller.signal },
      "读取测试集",
    )
      .then((body) =>
        definition.scene
          ? readScopedDatasetView(body, { mode, scene: definition.scene })
          : { datasets: readScopedDatasets(body) },
      )
      .then((view) => {
        if (controller.signal.aborted) return;
        setDatasetView(definition.scene ? view : null);
        setDatasets(view.datasets);
        setSelectedDatasetId((current) =>
          view.datasets.some((row) => row.dataset_id === current)
            ? current
            : view.datasets[0]?.dataset_id ?? "",
        );
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) {
          setDatasetError(errorMessage(reason, "读取测试集列表失败"));
          setDatasetView(null);
          setDatasets([]);
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setDatasetsLoading(false);
      });
    return () => controller.abort();
  }, [datasetRefresh, datasetsUrl, definition.scene, mode]);

  useEffect(() => {
    const controller = new AbortController();
    restoreAbortRef.current = controller;
    setRestoring(true);
    setRestoreError("");
    setHistoryError("");

    const restore = async () => {
      let history: ScopedBatchSummary[] | null = null;
      try {
        const body = await requestJson(
          `/api/alignment/acceptance/${definitionApiRoute(definition)}/batches?limit=20`,
          { signal: controller.signal },
          `读取${definition.title}历史`,
        );
        history = readScopedHistory(
          body,
          definition.scope,
          definition.scene,
        );
        if (!controller.signal.aborted) setHistoryRows(history);
      } catch (reason) {
        if (!controller.signal.aborted) {
          setHistoryError(
            errorMessage(reason, `${definition.title}历史读取失败`),
          );
        }
      }

      if (controller.signal.aborted) return;

      const manualTarget = restoreTargetRef.current;
      const storedTarget = manualTarget
        ? ""
        : readStoredBatchId(latestStorageKey);
      const target =
        manualTarget ||
        storedTarget ||
        history?.[0]?.batch_id ||
        "";
      const currentSnapshotId = snapshotRef.current?.batch_id ?? "";

      if (!target) {
        setHistorySelection(currentSnapshotId);
        if (history === null) {
          setRestoreError(
            `${definition.title}历史暂不可用，且本地没有可恢复的报告编号。`,
          );
        }
        return;
      }

      setHistorySelection(target);
      try {
        const next = await fetchScopedSnapshot(
          target,
          definition,
          controller.signal,
          `恢复${definition.title}结果`,
        );
        assertSnapshotCatalogIdentity(next, catalogRef.current);
        if (controller.signal.aborted) return;
        setSnapshot(next);
        if (!isScenePage) setSelectedDatasetId(next.dataset_id);
        setHistorySelection(next.batch_id);
        setRunError("");
        setPollError("");
        storeBatchId(latestStorageKey, next.batch_id);
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
            const fallback = await fetchScopedSnapshot(
              fallbackId,
              definition,
              controller.signal,
              `恢复最新${definition.title}结果`,
            );
            assertSnapshotCatalogIdentity(fallback, catalogRef.current);
            if (controller.signal.aborted) return;
            setSnapshot(fallback);
            if (!isScenePage) setSelectedDatasetId(fallback.dataset_id);
            setHistorySelection(fallback.batch_id);
            setRunError("");
            setPollError("");
            storeBatchId(latestStorageKey, fallback.batch_id);
            restoreTargetRef.current = "";
            setRestoreError(
              `${errorMessage(
                reason,
                "本地保存的报告编号不可用",
              )}；已拒绝该记录并恢复最新${definition.title}批次 ${fallback.batch_id.slice(
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
              )}；最新${definition.title}批次也恢复失败：${errorMessage(
                fallbackReason,
                "未知错误",
              )}`,
            );
            setHistorySelection(currentSnapshotId);
            return;
          }
        }

        setRestoreError(
          errorMessage(reason, `恢复${definition.title}结果失败`),
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
  }, [
    definition.apiRoute,
    definition.route,
    definition.scope,
    definition.scene,
    definition.title,
    latestStorageKey,
    restoreNonce,
  ]);

  useEffect(() => {
    const current = snapshot;
    if (!current || !isBatchActive) return;

    const controller = new AbortController();
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
        const next = await fetchScopedSnapshot(
          batchId,
          definition,
          controller.signal,
          `查询${definition.title}进度`,
          {
            datasetId: current.dataset_id,
            algorithmClusterId: batchCluster,
            mode: batchMode,
          },
        );
        assertSnapshotCatalogIdentity(next, catalogRef.current);
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
              `${definition.title}进度查询失败，已停止自动轮询`,
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
    definition.apiRoute,
    definition.route,
    definition.scope,
    definition.scene,
    definition.title,
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
    if (!archive || controlsLocked || importAbortRef.current) return;
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
      const rows = readScopedDatasets([body]);
      const imported = rows[0];
      if (!imported) throw new Error("测试集导入结果为空");
      if (controller.signal.aborted) return;
      if (isScenePage) {
        setDatasetRefresh((current) => current + 1);
      } else {
        setDatasets((current) => [
          imported,
          ...current.filter(
            (row) => row.dataset_id !== imported.dataset_id,
          ),
        ]);
        setSelectedDatasetId(imported.dataset_id);
      }
      setImportSuccess(
        `已导入 ${imported.name}：${INTEGER_FORMAT.format(
          imported.item_count,
        )} 项 / ${INTEGER_FORMAT.format(imported.case_count)} 个条件组。`,
      );
      setArchive(null);
      if (archiveInputRef.current) archiveInputRef.current.value = "";
    } catch (reason) {
      if (!controller.signal.aborted) {
        setImportError(errorMessage(reason, "测试集 ZIP 导入失败"));
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
    const requestedDatasetId = currentDatasetId || selectedDataset.dataset_id;
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
        `/api/alignment/acceptance/${definitionApiRoute(definition)}/batches`,
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
        `创建${definition.title}批次`,
      );
      const next = readScopedSnapshot(body, definition.scope, {
        datasetId: requestedDatasetId,
        algorithmClusterId: requestedClusterId,
        mode: requestedMode,
        scene: definition.scene,
      });
      assertSnapshotCatalogIdentity(next, catalog);
      if (controller.signal.aborted) return;
      setSnapshot(next);
      setHistorySelection(next.batch_id);
      setHistoryRows((current) => [
        asBatchSummary(next),
        ...current.filter((row) => row.batch_id !== next.batch_id),
      ]);
      storeBatchId(latestStorageKey, next.batch_id);
    } catch (reason) {
      if (!controller.signal.aborted) {
        setRunError(
          errorMessage(reason, `创建${definition.title}批次失败`),
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
      ? "正在恢复历史结果…"
      : launching
        ? "正在创建批次…"
        : isBatchActive
          ? `${definition.title}执行中…`
          : runError
            ? "重试创建批次"
            : `执行 ${expectedItemCount} 项${definition.title}`;
  const liveIssue = runError || pollError || restoreError || historyError;
  const liveStatusText = restoring
    ? "正在恢复最近结果"
    : launching
      ? "正在创建批次"
      : isBatchActive
        ? `${SCOPED_BATCH_LABELS[snapshot.status]} · ${progressDone}/${progressTotal} 项`
        : snapshot
          ? `${SCOPED_BATCH_LABELS[snapshot.status]} · ${snapshot.summary.computed}/${snapshot.summary.total} 项已计算`
          : "尚无批次结果";
  const panel = (value: SceneSubpageId) =>
    !isScenePage || sceneSubpage === value;

  return (
    <section
      className="basic-uplink-test scoped-acoustic-test"
      aria-labelledby={titleId}
    >
      <header className="basic-uplink-header scoped-acoustic-header">
        <div className="basic-uplink-heading">
          <span className="basic-uplink-eyebrow">
            {definition.eyebrow} · CONDITION-BOUND
          </span>
          <h2 id={titleId}>{definition.title}</h2>
          <p>{definition.description}</p>
        </div>
        <div
          className="basic-uplink-badges scoped-acoustic-badges"
          aria-label="测试口径"
        >
          <span>单模式</span>
          <span>
            {expectedItemCount} 项 · {mode}
          </span>
          <span>{outputLabel}证据</span>
          <span>工程代理</span>
          <span>正式 ITU-T 未测</span>
          {definition.scope === "background_noise" && !definition.scene ? (
            <span className="scoped-acoustic-badge-split">
              12 项 GNS + 8 项 CNG
            </span>
          ) : null}
        </div>
      </header>

      {idle20Preview ? (
        <div className="scoped-acoustic-preview-status" role="status">
          <span>工程预览口径</span>
          <strong>
            5 个合同测项 · 20 个工程条件组预览 · 2 个可执行旧批次条件组
          </strong>
          <small>
            20 组不等于 20 个合同测项；standard_compliance=false，正式 ITU-T
            合规未测，旧批次执行语义不变。
          </small>
        </div>
      ) : null}

      {isScenePage ? (
        <>
          <nav
            className="scoped-acoustic-subpage-nav"
            aria-label={`${definition.title}子页面`}
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
              <strong>批次</strong>
              {liveStatusText}
            </span>
            <span className="scoped-acoustic-subpage-status-note">
              工程代理 · 正式 ITU-T 合规未测：结果只代表当前场景的持久化工程计算。
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
            {snapshot && sceneSubpage !== "results" ? (
              <a
                href={sceneSubpageHash(definition.route, "results")}
                onClick={(event) => {
                  event.preventDefault();
                  showSceneSubpage("results");
                }}
              >
                前往结果
              </a>
            ) : null}
          </div>
        </>
      ) : null}

      <SceneSubpagePanel
        enabled={isScenePage}
        active={panel(isScenePage ? "description" : "execute")}
        label="流程树与测试说明"
      >
        <ScopedAcousticTree
          definition={definition}
          mode={mode}
          algorithmClusterId={algorithmClusterId}
          datasetId={currentDatasetId}
          dataset={selectedDataset}
          catalog={catalog}
          catalogLoading={catalogLoading}
          catalogError={catalogError}
          snapshot={treeSnapshot}
          syncFailed={Boolean(pollError || restoreError)}
          isPreviousSnapshot={
            showingPreviousSnapshot && Boolean(treeSnapshot)
          }
        />
        <aside className="scoped-acoustic-boundary" role="note">
          <ShieldAlert size={19} aria-hidden="true" />
          <div>
            <strong>工程代理边界</strong>
            <p>{definition.boundary}</p>
          </div>
        </aside>
        <section
          className="scoped-acoustic-context"
          aria-label={`${definition.title}输入、执行阶段与计算方法`}
        >
        <details>
          <summary>
            <span>测试输入与数据边界</span>
            <small>{definition.inputTitle}</small>
          </summary>
          <ul>
            {definition.inputs.map((input) => (
              <li key={input}>{input}</li>
            ))}
          </ul>
        </details>
        <details>
          <summary>
            <span>执行阶段</span>
            <small>{definition.stages.length} 个阶段</small>
          </summary>
          <ol>
            {definition.stages.map((stage) => (
              <li key={stage.label}>
                <strong>{stage.label}</strong>
                <span>{stage.detail}</span>
              </li>
            ))}
          </ol>
        </details>
        <details>
          <summary>
            <span>工程计算方法</span>
            <small>{Object.keys(definition.methods).length} 类方法</small>
          </summary>
          <dl>
            {Object.entries(definition.methods).map(([family, method]) => (
              <div key={family}>
                <dt>
                  <code>{family}</code>
                  <strong>{method.label}</strong>
                </dt>
                <dd>{method.detail}</dd>
              </div>
            ))}
          </dl>
        </details>
        </section>
      </SceneSubpagePanel>

      <SceneSubpagePanel
        enabled={isScenePage}
        active={panel("results")}
        label="历史恢复"
      >
      <div className="basic-uplink-history">
        <label htmlFor={historySelectId}>
          <History size={15} aria-hidden="true" />
          已保存的{definition.title}结果
        </label>
        <select
          id={historySelectId}
          aria-label={`${definition.title}历史结果`}
          value={historySelection || snapshot?.batch_id || ""}
          disabled={controlsLocked || historyChoices.length === 0}
          onChange={(event) => {
            restoreTargetRef.current = event.target.value;
            setHistorySelection(event.target.value);
            setRestoreNonce((current) => current + 1);
          }}
        >
          {historyChoices.length === 0 ? (
            <option value="">
              {restoring ? "正在恢复…" : "暂无历史"}
            </option>
          ) : null}
          {historyChoices.map((row) => (
            <option key={row.batch_id} value={row.batch_id}>
              {formatDate(row.created_at)} · {modeLabel(row.selected_mode)} ·{" "}
              {row.algorithm_cluster_id} ·{" "}
              {SCOPED_BATCH_LABELS[row.status]} · {row.batch_id.slice(0, 8)}
            </option>
          ))}
        </select>
        <button
          className="button secondary"
          type="button"
          aria-label={`刷新${definition.title}历史并恢复报告`}
          disabled={controlsLocked}
          onClick={() => setRestoreNonce((current) => current + 1)}
        >
          <RefreshCw size={15} aria-hidden="true" />
          刷新历史与恢复
        </button>
        {restoring ? (
          <p className="basic-uplink-history-note" role="status">
            正在恢复最近一次结果，并校验范围、模式、数量与逐项身份。
          </p>
        ) : null}
        {historyError ? (
          <p className="basic-uplink-inline-error" role="alert">
            {historyError}
            {snapshot ? "；当前仍保留上次成功快照。" : ""}
          </p>
        ) : null}
        {restoreError ? (
          <p className="basic-uplink-inline-warning" role="alert">
            {restoreError}
            {snapshot ? "；当前仍保留上次成功快照。" : ""}
          </p>
        ) : null}
      </div>
      </SceneSubpagePanel>

      <SceneSubpagePanel
        enabled={isScenePage}
        active={panel("execute")}
        label="执行控制"
      >
      <div className="basic-uplink-controls scoped-acoustic-controls">
        <label htmlFor={modeSelectId}>
          <span>业务模式</span>
          <select
            id={modeSelectId}
            aria-label={`${definition.title}业务模式`}
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
          <small>
            当前模式要求 {expectedItemCount} 个唯一合同测项；目录、批次和报告
            都按此模式严格校验。
          </small>
        </label>

        <label htmlFor={clusterSelectId}>
          <span>VCP 算法簇</span>
          <select
            id={clusterSelectId}
            aria-label={`${definition.title}VCP 算法簇`}
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
              : "本页只允许执行 VCP 预设；非 VCP 算法簇保持不可执行。"}
          </small>
        </label>

        {!isScenePage ? (
        <label htmlFor={datasetSelectId}>
          <span>完整 336 项测试集</span>
          <select
            id={datasetSelectId}
            aria-label={`${definition.title}测试集`}
            value={selectedDatasetId}
            disabled={
              controlsLocked ||
              datasetsLoading ||
              (datasets.length === 0 && !selectedDatasetMissing)
            }
            onChange={(event) =>
              setSelectedDatasetId(event.target.value)
            }
          >
            {datasets.length === 0 ? (
              <option value="">
                {datasetsLoading ? "正在读取测试集…" : "暂无已导入测试集"}
              </option>
            ) : null}
            {selectedDatasetMissing ? (
              <option value={selectedDatasetId}>
                {snapshot?.dataset_name ?? selectedDatasetId}
                （当前列表未返回）
              </option>
            ) : null}
            {datasets.map((dataset) => (
              <option key={dataset.dataset_id} value={dataset.dataset_id}>
                {dataset.name} · {formatDate(dataset.created_at)} ·{" "}
                {dataset.dataset_id.slice(0, 8)} · {dataset.item_count} 项 /{" "}
                {dataset.case_count} 条件组
              </option>
            ))}
          </select>
          <small>
            {selectedDataset
              ? `${formatDate(selectedDataset.created_at)} 导入；${
                  selectedDatasetIsComplete
                    ? `覆盖 ${FULL_DATASET_ITEM_COUNT} 项`
                    : `仅 ${selectedDataset.item_count} 项，不能执行`
                }；${selectedDataset.case_count} 个条件组。`
              : "只列出服务端已校验的测试集。"}
          </small>
        </label>
        ) : null}

        {!isScenePage ? (
        <div className="basic-uplink-refresh-cell">
          <span>测试集列表</span>
          <button
            className="button secondary"
            type="button"
            aria-label={`刷新${definition.title}可用测试集`}
            disabled={controlsLocked || datasetsLoading}
            onClick={() => setDatasetRefresh((current) => current + 1)}
          >
            <RefreshCw size={15} aria-hidden="true" />
            {datasetsLoading ? "正在刷新…" : "刷新数据集"}
          </button>
          <small>
            刷新不会替换已选报告；切换数据集也不会改写历史。
          </small>
        </div>
        ) : null}
      </div>

      {datasetError ? (
        <p className="basic-uplink-inline-error" role="alert">
          {datasetError}
          {datasets.length > 0 ? "；当前仍保留上次读取的测试集列表。" : ""}
        </p>
      ) : null}
      {selectedDatasetMissing ? (
        <p className="basic-uplink-inline-error" role="alert">
          当前选择的测试集不在服务端列表中，不能创建新批次。
        </p>
      ) : null}
      {selectedDataset && !selectedDatasetIsComplete ? (
        <p className="basic-uplink-inline-error" role="alert">
          当前测试集未覆盖 336 项完整合同，不能执行本页批次。
        </p>
      ) : null}
      {selectedDataset && !selectedDatasetHasData ? (
        <p className="basic-uplink-inline-error" role="alert">
          当前测试集没有任何条件组，不能运行缺数据批次。
        </p>
      ) : null}

      {definition.scene ? (
        <ScopedDatasetVisualization
          dataset={selectedDataset}
          mode={mode}
          scene={definition.scene}
          expectedItemCount={expectedItemCount}
          active={pageVisible && panel("execute")}
          loading={datasetsLoading}
          refreshToken={datasetRefresh}
          refreshDisabled={controlsLocked || datasetsLoading}
          onRefresh={() => setDatasetRefresh((current) => current + 1)}
          onOpenWaveforms={() => {
            setWaveformSource("execution");
            showSceneSubpage("waveforms");
          }}
          onOpenExtendedWaveforms={() => {
            setWaveformSource("extended");
            showSceneSubpage("waveforms");
          }}
          extendedPreviewCount={idle20Preview ? 20 : undefined}
        />
      ) : null}

      {!isScenePage ? (
        <div className="scoped-acoustic-import-block">
        <form className="basic-uplink-upload" onSubmit={handleImport}>
        <div>
          <label htmlFor={archiveInputId}>
            <Upload size={15} aria-hidden="true" />
            上传完整 336 项 ZIP
          </label>
          <input
            ref={archiveInputRef}
            id={archiveInputId}
            aria-label={`上传${definition.title}完整 336 项 ZIP`}
            type="file"
            accept=".zip,application/zip,application/x-zip-compressed"
            disabled={controlsLocked || importing}
            onChange={(event) => {
              const file = event.target.files?.[0] ?? null;
              handleArchiveChange(file);
              if (
                file &&
                (!isZipArchive(file) || file.size > MAX_ARCHIVE_BYTES)
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
              : "manifest.json + audio/ · 最大 256 MiB"}
          </span>
        </div>
        <button
          className="button secondary"
          type="submit"
          aria-label={`导入${definition.title}测试集 ZIP`}
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
        </div>
      ) : null}

      <div className="basic-uplink-run">
        <div>
          <strong>
            {modeLabel(mode)} · {algorithmLabel(algorithmClusterId)}
          </strong>
          <p>
            执行路径：按当前模式提取唯一合同条件组 → 所选 VCP 处理 →{" "}
            {expectedItemCount} 项工程计算 → 保存服务端报告。
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
              {isScenePage ? "结果子页面" : "下方"}仍只显示最后一次成功快照；失败请求没有被记作通过或零值。
            </p>
          </div>
        </div>
      ) : null}
      </SceneSubpagePanel>

      <SceneSubpagePanel
        enabled={isScenePage}
        active={panel("waveforms")}
        label="波形与梅尔频谱"
      >
        <ScopedAcousticWaveforms
          active={pageVisible && panel("waveforms")}
          previewSource={waveformSource}
          definition={definition}
          datasetId={currentDatasetId}
          mode={mode}
          algorithmClusterId={algorithmClusterId}
          snapshot={snapshot}
        />
      </SceneSubpagePanel>

      <SceneSubpagePanel
        enabled={isScenePage}
        active={panel("description")}
        label="合同目录"
      >
      <details
        className="basic-uplink-catalog"
        aria-labelledby={catalogTitleId}
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
              id={catalogTitleId}
            >
              {expectedItemCount} 项{definition.title}合同目录
            </span>
            <span className="basic-uplink-catalog-description">
              {modeLabel(mode)}
              {definition.scope === "background_noise"
                ? "；GNS 12 项、CNG 8 项分别归类。"
                : "；合同单位与限值独立于工程计算指标。"}
            </span>
          </span>
          <span className="basic-uplink-catalog-summary-meta">
            <span className="basic-uplink-catalog-count">
              {catalog
                ? `${catalog.item_count} / ${expectedItemCount}`
                : catalogError
                  ? "读取失败"
                  : `${expectedItemCount} 行空状态`}
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
                aria-label={`重试读取${definition.title}合同目录`}
                disabled={catalogLoading}
                onClick={() => setCatalogRefresh((current) => current + 1)}
              >
                <RefreshCw size={15} aria-hidden="true" />
                重试读取目录
              </button>
            </div>
          ) : null}
          <div className="table-scroll">
            <table className="audio-table basic-uplink-catalog-table scoped-acoustic-catalog-table">
              <caption className="sr-only">
                当前模式的 {expectedItemCount} 项{definition.title}合同目录
              </caption>
              <thead>
                <tr>
                  <th scope="col">合同行 / 测项</th>
                  <th scope="col">范围分类</th>
                  <th scope="col">合同单位</th>
                  <th scope="col">合同限值</th>
                </tr>
              </thead>
              <tbody>
                {Array.from({ length: expectedItemCount }, (_, index) => {
                  const item = catalog?.items[index];
                  const family = item?.family ?? "";
                  const group = familyGroup(definition, family);
                  return (
                    <tr key={item?.id ?? `catalog-empty-${index}`}>
                      <td>
                        <div className="basic-uplink-source-row">
                          <code>{item?.id ?? "—"}</code>
                          <strong>
                            {item?.test_name ??
                              (catalogLoading ? "目录读取中" : "目录不可用")}
                          </strong>
                          <span>{item?.source_cell ?? "—"}</span>
                        </div>
                      </td>
                      <td>
                        <div className="scoped-acoustic-family">
                          <span
                            className={`scoped-acoustic-family-chip is-${group.toLocaleLowerCase()}`}
                          >
                            {group}
                          </span>
                          <code>{item?.family ?? "—"}</code>
                        </div>
                      </td>
                      <td>
                        <div className="basic-uplink-contract-unit">
                          <span>{item?.unit || "—"}</span>
                          <small>合同字段</small>
                        </div>
                      </td>
                      <td>
                        {item
                          ? formatRange(item.lower_limit, item.upper_limit)
                          : "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </details>
      </SceneSubpagePanel>

      <SceneSubpagePanel
        enabled={isScenePage}
        active={panel("results")}
        label="批次结果"
      >
      {!snapshot ? (
        <div className="basic-uplink-empty">
          <ListChecks size={24} aria-hidden="true" />
          <div>
            <h3>尚无{definition.title}批次结果</h3>
            <p>
              {isScenePage
                ? `使用本页自动匹配的场景测试集与 VCP 预设后执行。说明子页面固定展示当前模式的 ${expectedItemCount} 行合同目录，不预填通过、数值或状态。`
                : `选择完整 336 项测试集与 VCP 预设后执行。上方固定展示当前模式的 ${expectedItemCount} 行合同目录，不预填通过、数值或状态。`}
            </p>
          </div>
        </div>
      ) : catalogIdentityError ? (
        <div className="basic-uplink-alert scoped-acoustic-identity-alert" role="alert">
          <AlertCircle size={18} aria-hidden="true" />
          <div>
            <strong>报告身份隔离</strong>
            <p>{catalogIdentityError}</p>
          </div>
        </div>
      ) : (
        <>
          <section
            className="basic-uplink-result"
            aria-labelledby={resultTitleId}
          >
            <header className="basic-uplink-result-header">
              <div>
                <span className="basic-uplink-eyebrow">
                  SERVER SNAPSHOT
                </span>
                <h3 id={resultTitleId}>
                  {showingPreviousSnapshot ? "上次成功快照 · " : ""}
                  {SCOPED_BATCH_LABELS[snapshot.status]}
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
                      : `每 2 秒读取 · ${scopedValue(progressPercent)}%`
                    : SCOPED_BATCH_LABELS[snapshot.status]}
                </span>
                {canDownloadReport ? (
                  <a
                    className="button secondary"
                    href={`${BATCHES_URL}/${encodeURIComponent(
                      snapshot.batch_id,
                    )}/report`}
                    download={`scoped-${definition.scope}-${snapshot.batch_id}.json`}
                    aria-label={`下载${definition.title}完整报告 JSON`}
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
                aria-label={`${definition.title}逐项完成进度`}
              >
                {scopedValue(progressPercent)}%
              </progress>
            </div>

            <div className="basic-uplink-result-boundary">
              <ShieldAlert size={18} aria-hidden="true" />
              <div>
                <strong>工程代理结果 · 正式 ITU-T 合规未测</strong>
                <p>
                  “工程通过”只表示服务端计算满足当前代理判据；需复核、
                  不可测与无效项保持原状态，不能据此宣称正式合规。
                </p>
              </div>
            </div>

            {hasConfigurationMismatch ? (
              <div className="basic-uplink-mismatch" role="status">
                <AlertCircle size={18} aria-hidden="true" />
                <div>
                  <strong>当前选择与报告实际配置不同</strong>
                  <p>
                    报告实际执行：{modeLabel(snapshot.selected_mode)}
                    {" · "}
                    {algorithmLabel(snapshot.algorithm_cluster_id)}
                    {" · "}
                    {snapshot.dataset_name}。当前选择：{modeLabel(mode)}
                    {" · "}
                    {algorithmLabel(algorithmClusterId || "未选择")}
                    {" · "}
                    {selectedDataset?.name ?? "未选择测试集"}。历史结果继续供查看，
                    但不会送入当前配置的树状图。
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
                <code>
                  {definition.title} ·{" "}
                  {scopedItemCount(
                    definition.scope,
                    snapshot.selected_mode,
                    definition.scene,
                  )}{" "}
                  项
                </code>
              </span>
              <span>
                业务模式 <code>{modeLabel(snapshot.selected_mode)}</code>
              </span>
              <span>
                算法 <code>{algorithmLabel(snapshot.algorithm_cluster_id)}</code>
              </span>
              <span>
                报告时间 <code>{formatDate(snapshot.created_at)}</code>
              </span>
            </div>

            {pollError ? (
              <div className="basic-uplink-alert" role="alert">
                <AlertCircle size={18} aria-hidden="true" />
                <div>
                  <strong>自动轮询已停止</strong>
                  <p>{pollError}</p>
                  <button
                    className="button secondary"
                    type="button"
                    aria-label={`重新查询${definition.title}批次`}
                    onClick={retryPolling}
                  >
                    <RefreshCw size={15} aria-hidden="true" />
                    重新查询批次
                  </button>
                </div>
              </div>
            ) : null}

            {snapshot.error ? (
              <div className="basic-uplink-alert" role="alert">
                <AlertCircle size={18} aria-hidden="true" />
                <div>
                  <strong>服务端报告批次错误</strong>
                  <p>{snapshot.error}</p>
                </div>
              </div>
            ) : null}

            {failedCases.length > 0 ? (
              <div className="basic-uplink-alert" role="alert">
                <AlertCircle size={18} aria-hidden="true" />
                <div className="scoped-acoustic-failed-cases">
                  <strong>{failedCases.length} 个条件组执行失败</strong>
                  <div>
                    {failedCases.map((entry) => (
                      <p key={entry.case_id}>
                        <code>{entry.case_id}</code>
                        {entry.error
                          ? `：${entry.error}`
                          : "：条件组执行失败"}
                      </p>
                    ))}
                  </div>
                </div>
              </div>
            ) : null}
          </section>

          <section
            className="basic-uplink-items"
            aria-labelledby={itemsTitleId}
          >
            <header>
              <div>
                <span className="basic-uplink-eyebrow">
                  ITEM EVIDENCE
                </span>
                <h3 id={itemsTitleId}>
                  {snapshot.summary.total} 项逐项结果
                </h3>
                <p>
                  合同单位、合同限值、工程指标、工程值和工程单位分列展示；
                  条件组完成后才显示报告与 {outputLabel}链接。
                </p>
              </div>
              <span>
                {snapshot.items.length} / {snapshot.summary.total}
              </span>
            </header>

            <div className="table-scroll">
              <table className="audio-table basic-uplink-items-table scoped-acoustic-items-table">
                <caption className="sr-only">
                  {definition.title}逐项工程结果与合同证据
                </caption>
                <thead>
                  <tr>
                    <th scope="col">合同行 / 测项</th>
                    <th scope="col">范围分类</th>
                    <th scope="col">条件绑定</th>
                    <th scope="col">合同单位</th>
                    <th scope="col">合同限值</th>
                    <th scope="col">工程指标</th>
                    <th scope="col">工程值 / 工程单位</th>
                    <th scope="col">状态</th>
                    <th scope="col">原因</th>
                    <th scope="col">证据与 {outputLabel}</th>
                  </tr>
                </thead>
                <tbody>
                  {snapshot.items.map((item) => {
                    const catalogItem =
                      reportCatalogById.get(item.item_id) ??
                      (item.id
                        ? reportCatalogById.get(item.id)
                        : undefined);
                    const caseId = item.evidence_binding?.case_id;
                    return (
                      <ResultItem
                        key={item.item_id}
                        item={item}
                        catalogItem={catalogItem}
                        definition={definition}
                        snapshot={snapshot}
                        caseStatus={
                          caseId ? caseStatusById.get(caseId) : undefined
                        }
                        caseError={
                          caseId ? caseErrorById.get(caseId) : undefined
                        }
                      />
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}
      </SceneSubpagePanel>
    </section>
  );
}
