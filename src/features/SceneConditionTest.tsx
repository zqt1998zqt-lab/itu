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
import { chainPresets } from "./enhancement-chain-model.ts";
import {
  FIVE_SIGNAL_MODES,
  type FiveSignalMode,
} from "./five-signal-flow-model.ts";
import ScopedAcousticWaveforms from "./ScopedAcousticWaveforms";
import ScopedDatasetVisualization from "./ScopedDatasetVisualization";
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
  type ScopedTestDefinition,
  type ScopedTestPageProps,
} from "./scoped-acoustic-model.ts";
import {
  SCENE_CONDITION_BATCHES_URL,
  SCENE_CONDITION_BATCH_COUNT,
  SCENE_CONDITION_CONTRACT_ITEM_COUNT,
  SCENE_CONDITION_DATASETS_URL,
  SCENE_CONDITION_OBSERVATION_COUNT,
  readSceneConditionBatch,
  readSceneConditionBatchHistory,
  sceneConditionCaseTxEvidence,
  type SceneConditionBatch,
  type SceneConditionCase,
  type SceneConditionItemId,
  type SceneConditionObservation,
} from "./scene-condition-test-model.ts";
import "./basic-uplink-test.css";
import "./scoped-acoustic-test.css";
import "./scene-condition-test.css";

const POLL_INTERVAL_MS = 2_000;
const VCP_PRESETS = chainPresets.filter((preset) =>
  preset.profile.startsWith("vcp"),
);
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
const CONTRACT_GROUPS: readonly {
  itemId: SceneConditionItemId;
  family: string;
  label: string;
  detail: string;
}[] = [
  {
    itemId: "WB-wired:B24",
    family: "g_mos",
    label: "整体质量",
    detail: "DNSMOS P.835 OVRL 工程代理",
  },
  {
    itemId: "WB-wired:B25",
    family: "n_mos",
    label: "背景噪声质量",
    detail: "DNSMOS P.835 BAK 工程代理",
  },
  {
    itemId: "WB-wired:B26",
    family: "s_mos",
    label: "噪声中语音质量",
    detail: "DNSMOS P.835 SIG 工程代理",
  },
  {
    itemId: "WB-wired:B27",
    family: "cng_level",
    label: "舒适噪声电平",
    detail: "noise 与 TX 的数字 RMS 电平差",
  },
  {
    itemId: "WB-wired:B28",
    family: "cng_spectrum",
    label: "舒适噪声频谱",
    detail: "归一化 PSD 的 Jensen-Shannon 距离",
  },
];
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
  unit: string;
  pass: number;
  fail: number;
  reviewRequired: number;
  notMeasured: number;
  invalid: number;
};

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

function modeLabel(mode: FiveSignalMode): string {
  return `${MODE_LABELS[mode]} · ${mode}`;
}

function algorithmLabel(id: string): string {
  const preset = chainPresets.find((candidate) => candidate.id === id);
  return preset ? `${preset.label} · ${preset.id}` : id || "未选择";
}

function latestStorageKey(algorithmClusterId: string): string {
  return `scene-condition:sce1:WB-wired:${algorithmClusterId}:latest`;
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
): batch is SceneConditionBatch {
  return Boolean(
    batch
      && batch.dataset_id === datasetId
      && batch.algorithm_cluster_id === algorithmClusterId
      && batch.selected_mode === mode
      && batch.selected_scene === "sce1",
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
          case_count: SCENE_CONDITION_BATCH_COUNT,
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
  const uid = useId().replace(/[^A-Za-z0-9_-]/g, "");
  const titleId = `scene-condition-${uid}-title`;
  const modeSelectId = `scene-condition-${uid}-mode`;
  const clusterSelectId = `scene-condition-${uid}-cluster`;
  const historySelectId = `scene-condition-${uid}-history`;
  const isWbWired = mode === "WB-wired";
  const algorithmIsSupported = VCP_PRESETS.some(
    (preset) => preset.id === algorithmClusterId,
  );
  const storageKey =
    isWbWired && algorithmIsSupported
      ? latestStorageKey(algorithmClusterId)
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
  const [runError, setRunError] = useState("");
  const [pollError, setPollError] = useState("");
  const [pollRetry, setPollRetry] = useState(0);

  const launchAbortRef = useRef<AbortController | null>(null);
  const restoreAbortRef = useRef<AbortController | null>(null);
  const restoreTargetRef = useRef("");

  const waveformDefinition = useMemo<ScopedTestDefinition>(
    () => ({
      ...definition,
      scope: "background_noise",
      scene: "sce1",
      previewMode: "idle-20-condition",
    }),
    [definition],
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
    setRunError("");
    setPollError("");
    setHistoryError("");
    setRestoreError("");
    restoreTargetRef.current = "";
  }, [algorithmClusterId, mode]);

  useEffect(() => {
    if (!isWbWired) {
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
      mode: "WB-wired",
      scene: "sce1",
    });
    requestJson(
      `${SCENE_CONDITION_DATASETS_URL}?${params.toString()}`,
      { signal: controller.signal },
      "读取 20 组怠速测试集",
    )
      .then((body) => {
        const view = readSceneConditionDatasetView(body, {
          mode: "WB-wired",
          scene: "sce1",
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
            errorMessage(reason, "20 组怠速测试集读取失败"),
          );
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setDatasetLoading(false);
      });
    return () => controller.abort();
  }, [datasetRefresh, isWbWired]);

  useEffect(() => {
    if (
      !isWbWired
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
      });
      let body: unknown;
      try {
        body = await requestJson(
          `${SCENE_CONDITION_BATCHES_URL}?${params.toString()}`,
          { signal: controller.signal },
          "读取 20 组测试历史",
        );
      } catch (reason) {
        if (!controller.signal.aborted) {
          setHistoryError(
            errorMessage(reason, "20 组测试历史读取失败"),
          );
        }
        return;
      }
      let history: SceneConditionBatch[];
      try {
        history = readSceneConditionBatchHistory(body, {
          datasetId,
          algorithmClusterId,
        });
      } catch (reason) {
        if (!controller.signal.aborted) {
          setHistoryRows([]);
          setSnapshot(null);
          setHistorySelection("");
          setHistoryError(
            `${errorMessage(
              reason,
              "20 组测试历史格式无效",
            )}；旧报告不会进入本页。`,
          );
        }
        return;
      }
      if (controller.signal.aborted) return;
      setHistoryRows(history);
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
          setRestoreError("所选 20 组批次已不在新历史中。");
          restoreTargetRef.current = "";
        } else if (stored.invalid) {
          setRestoreError(
            "本地保存的 20 组批次编号无效；新的服务端历史为空。",
          );
        } else if (stored.batchId) {
          setRestoreError(
            "本地保存的 20 组批次不在新历史中；新的服务端历史为空。",
          );
        }
        return;
      }
      setHistorySelection(target);
      try {
        const nextBody = await requestJson(
          `${SCENE_CONDITION_BATCHES_URL}/${encodeURIComponent(target)}`,
          { signal: controller.signal },
          "恢复 20 组测试结果",
        );
        const next = readSceneConditionBatch(nextBody, {
          datasetId,
          algorithmClusterId,
        });
        if (controller.signal.aborted) return;
        if (!batchMatches(next, datasetId, algorithmClusterId, mode)) {
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
            "本地保存的 20 组批次编号无效，已恢复新历史第一条。",
          );
        } else if (stored.batchId && !storedExists) {
          setRestoreError(
            "本地保存的 20 组批次不在新历史中，已恢复新历史第一条。",
          );
        }
      } catch (reason) {
        if (controller.signal.aborted) return;
        setSnapshot(null);
        restoreTargetRef.current = "";
        setRestoreError(
          errorMessage(reason, "恢复 20 组测试结果失败"),
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
    isWbWired,
    mode,
    restoreNonce,
    storageKey,
  ]);

  const isBatchActive =
    snapshot?.status === "queued" || snapshot?.status === "running";

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
          "查询 20 组测试进度",
        );
        const next = readSceneConditionBatch(body, {
          datasetId: expectedDatasetId,
          algorithmClusterId: expectedAlgorithmId,
        });
        if (disposed || controller.signal.aborted) return;
        nextStatus = next.status;
        setSnapshot(next);
        setPollError("");
      } catch (reason) {
        if (!disposed && !controller.signal.aborted) {
          setPollError(
            `${errorMessage(
              reason,
              "20 组测试进度查询失败",
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
  )
    ? snapshot
    : null;
  const controlsLocked =
    externalRunning
    || launching
    || restoring
    || Boolean(isBatchActive);
  const canStart =
    isWbWired
    && algorithmIsSupported
    && Boolean(dataset)
    && Boolean(datasetId)
    && !datasetLoading
    && !datasetError
    && !controlsLocked;

  const historyChoices = useMemo(() => {
    const rows = new Map<string, SceneConditionBatch>();
    for (const row of historyRows) rows.set(row.batch_id, row);
    if (currentSnapshot) rows.set(currentSnapshot.batch_id, currentSnapshot);
    return [...rows.values()].sort(
      (left, right) =>
        Date.parse(right.created_at) - Date.parse(left.created_at)
        || right.batch_id.localeCompare(left.batch_id),
    );
  }, [currentSnapshot, historyRows]);

  const observationsByCase = useMemo(() => {
    const grouped = new Map<string, SceneConditionObservation[]>();
    for (const item of currentSnapshot?.items ?? []) {
      const rows = grouped.get(item.case_id) ?? [];
      rows.push(item);
      grouped.set(item.case_id, rows);
    }
    for (const rows of grouped.values()) {
      rows.sort((left, right) => left.item_id.localeCompare(right.item_id));
    }
    return grouped;
  }, [currentSnapshot]);

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
    setRunError("");
    setPollError("");
    setHistoryError("");
    setRestoreError("");
  }

  async function handleStart() {
    if (
      !canStart
      || !datasetId
      || launchAbortRef.current
    ) {
      return;
    }
    const controller = new AbortController();
    launchAbortRef.current = controller;
    setLaunching(true);
    setRunError("");
    setPollError("");
    setRestoreError("");
    try {
      const body = await requestJson(
        SCENE_CONDITION_BATCHES_URL,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            dataset_id: datasetId,
            algorithm_cluster_id: algorithmClusterId,
            mode: "WB-wired",
            scene: "sce1",
          }),
          signal: controller.signal,
        },
        "创建 20 组怠速测试批次",
      );
      const next = readSceneConditionBatch(body, {
        datasetId,
        algorithmClusterId,
      });
      if (controller.signal.aborted) return;
      if (!batchMatches(next, datasetId, algorithmClusterId, mode)) {
        throw new Error("新批次身份与当前页面选择不一致");
      }
      setSnapshot(next);
      setHistorySelection(next.batch_id);
      setHistoryRows((current) => [
        next,
        ...current.filter((row) => row.batch_id !== next.batch_id),
      ]);
      storeBatchId(storageKey, next.batch_id);
    } catch (reason) {
      if (!controller.signal.aborted) {
        setRunError(
          errorMessage(reason, "20 组怠速测试批次创建失败"),
        );
      }
    } finally {
      if (launchAbortRef.current === controller) {
        launchAbortRef.current = null;
        if (!controller.signal.aborted) setLaunching(false);
      }
    }
  }

  const progressText = currentSnapshot
    ? `${currentSnapshot.progress.completed_cases}/${SCENE_CONDITION_BATCH_COUNT}组 · ${currentSnapshot.progress.completed_items}/${SCENE_CONDITION_OBSERVATION_COUNT}次测量`
    : "尚无批次";
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
  const progressPercent = currentSnapshot
    ? Math.min(
        100,
        (currentSnapshot.progress.completed_items
          / SCENE_CONDITION_OBSERVATION_COUNT) * 100,
      )
    : 0;

  return (
    <section
      className="basic-uplink-test scoped-acoustic-test scene-condition-test"
      aria-labelledby={titleId}
    >
      <header className="basic-uplink-header scoped-acoustic-header">
        <div className="basic-uplink-heading">
          <span className="basic-uplink-eyebrow">
            20-CONDITION IDLE · WB-WIRED
          </span>
          <h2 id={titleId}>{definition.title} · 20 组条件测试</h2>
          <p>
            10 个语音条件组各产生 3 个工程指标，10 个 CNG
            条件组各产生 2 个工程指标，共 50 次测量。测试前不自动执行。
          </p>
        </div>
        <div
          className="basic-uplink-badges scoped-acoustic-badges"
          aria-label="20 组测试口径"
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
        aria-label={`${definition.title}20 组测试子页面`}
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
        <span className="scoped-acoustic-subpage-status-note">
          {isWbWired
            ? "批次在后台轮询，切换子页面不会中断。"
            : "当前模式没有 20 组执行入口。"}
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

      <ActiveSubpage active={activeExecute} label="执行 20 组怠速测试">
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
            <small>保留全部 8 个业务模式；当前只有 WB-wired 支持 20 组测试集。</small>
          </label>

          <label htmlFor={clusterSelectId}>
            <span>VCP 算法簇</span>
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
                : "只接受现有 VCP 预设；非 VCP 算法簇不可执行。"}
            </small>
          </label>
        </div>

        {!isWbWired ? (
          <div className="sct-mode-support" role="status">
            <AlertCircle size={18} aria-hidden="true" />
            <div>
              <strong>20 组测试集目前支持 WB-wired</strong>
              <p>
                当前测试集仅提供 WB-wired 样本，请切换业务模式后执行。
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
              切回 WB-wired
            </button>
          </div>
        ) : null}

        {isWbWired && datasetError ? (
          <p className="basic-uplink-inline-error" role="alert">
            {datasetError}；测试集元数据无效时禁止执行。
          </p>
        ) : null}
        {isWbWired && !algorithmIsSupported ? (
          <p className="basic-uplink-inline-error" role="alert">
            当前算法簇不是 VCP 预设，禁止执行 20 组测试。
          </p>
        ) : null}

        {isWbWired && activeExecute ? (
          <ScopedDatasetVisualization
            source="scene-condition"
            dataset={dataset}
            mode={mode}
            scene="sce1"
            expectedItemCount={SCENE_CONDITION_CONTRACT_ITEM_COUNT}
            active={pageVisible}
            loading={datasetLoading}
            refreshToken={datasetRefresh}
            refreshDisabled={controlsLocked || datasetLoading}
            onRefresh={() =>
              setDatasetRefresh((current) => current + 1)
            }
            onOpenWaveforms={() => showSceneSubpage("waveforms")}
          />
        ) : null}

        <div className="basic-uplink-run">
          <div>
            <strong>
              {modeLabel(mode)} · {algorithmLabel(algorithmClusterId)}
            </strong>
            <p>
              逐组执行 VCP → 计算 5 类指标 → 保存 20 组输出与 50 次测量结果。
            </p>
          </div>
          <button
            className="button primary"
            type="button"
            disabled={!canStart}
            onClick={() => void handleStart()}
          >
            <Play size={15} fill="currentColor" aria-hidden="true" />
            执行20组测试
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
                /{SCENE_CONDITION_OBSERVATION_COUNT} 个有效结果
              </span>
            </div>
            <progress
              max={SCENE_CONDITION_OBSERVATION_COUNT}
              value={currentSnapshot.progress.completed_items}
              aria-label="20 组测试完成进度"
            >
              {scopedValue(progressPercent)}%
            </progress>
          </div>
        ) : null}

        {runError ? (
          <div className="basic-uplink-alert" role="alert">
            <AlertCircle size={18} aria-hidden="true" />
            <div>
              <strong>20 组测试创建失败</strong>
              <p>{runError}</p>
              <p>失败请求不会改写最近成功快照，也不会生成通过值。</p>
            </div>
          </div>
        ) : null}
      </ActiveSubpage>

      <ActiveSubpage active={activeResults} label="20 组测试结果">
        <div className="basic-uplink-history">
          <label htmlFor={historySelectId}>
            <History size={15} aria-hidden="true" />
            20 组测试历史
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
                {historyLoading ? "正在读取新历史…" : "暂无 20 组历史"}
              </option>
            ) : null}
            {historyChoices.map((row) => (
              <option key={row.batch_id} value={row.batch_id}>
                {formatDate(row.created_at)} · {row.algorithm_cluster_id} ·{" "}
                {SCOPED_BATCH_LABELS[row.status]} ·{" "}
                {row.batch_id.slice(0, 8)}
              </option>
            ))}
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
              <h3>尚无当前配置的 20 组测试结果</h3>
              <p>
                执行 20 组测试后，可在这里查看每组测量值和处理后音频。
              </p>
            </div>
          </div>
        ) : (
          <>
            <section
              className="basic-uplink-result"
              aria-label="20 组批次状态"
            >
              <header className="basic-uplink-result-header">
                <div>
                  <span className="basic-uplink-eyebrow">
                    20 组测试结果
                  </span>
                  <h3>{SCOPED_BATCH_LABELS[currentSnapshot.status]}</h3>
                  <p>
                    {currentSnapshot.dataset_name}
                    {" · "}
                    {algorithmLabel(currentSnapshot.algorithm_cluster_id)}
                    {" · "}
                    {formatDate(currentSnapshot.updated_at)}
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

              <div className="basic-uplink-progress">
                <div>
                  <strong>{progressText}</strong>
                  <span>
                    {INTEGER_FORMAT.format(currentSnapshot.summary.computed)}
                    /{SCENE_CONDITION_OBSERVATION_COUNT} 个有效结果
                  </span>
                </div>
                <progress
                  max={SCENE_CONDITION_OBSERVATION_COUNT}
                  value={currentSnapshot.progress.completed_items}
                  aria-label="20 组测试结果进度"
                >
                  {scopedValue(progressPercent)}%
                </progress>
              </div>

              <div className="basic-uplink-result-boundary">
                <ShieldAlert size={18} aria-hidden="true" />
                <div>
                  <strong>工程代理结果 · 正式 ITU-T 合规未测</strong>
                  <p>
                    5 个合同测项各包含 10 次条件组观测；“工程通过”只按逐观测状态计数，
                    不从均值、缺值或旧报告推断。
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
                <h3>5 类合同汇总</h3>
                <p>
                  每类显示真实有限观测数及诊断均值、最小值、最大值；不足
                  10 次时仍保持诊断口径，不标记通过。
                </p>
              </header>
              <div className="sct-contract-grid">
                {CONTRACT_GROUPS.map((group) => {
                  const stats = contractStats(
                    currentSnapshot.items,
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
                          {stats.count}/10 有限值
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
                        单位 {stats.unit || "未声明"} · 诊断汇总，不推断通过
                      </p>
                      <div className="sct-status-counts">
                        <span>通过 {stats.pass}</span>
                        <span>失败 {stats.fail}</span>
                        <span>需复核 {stats.reviewRequired}</span>
                        <span>不可测 {stats.notMeasured}</span>
                        <span>无效 {stats.invalid}</span>
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
                <h3>20 组处理结果</h3>
                <p>
                  TX 仅在条件组完成、报告与四路输入哈希、逐观测证据和可视化来源全部一致时开放。
                </p>
              </header>
              <div className="sct-case-grid">
                {[...currentSnapshot.cases]
                  .sort((left, right) =>
                    left.case_id.localeCompare(right.case_id)
                  )
                  .map((caseRecord) => {
                    const previewCase = previewCaseFromBatch(
                      currentSnapshot,
                      caseRecord,
                    );
                    const tx = sceneConditionCaseTxEvidence(
                      previewCase,
                      currentSnapshot,
                      currentSnapshot.dataset_manifest_sha256,
                    );
                    const caseItems =
                      observationsByCase.get(caseRecord.case_id) ?? [];
                    return (
                      <article
                        key={caseRecord.case_id}
                        className={`sct-case-card is-${caseRecord.status}`}
                      >
                        <header>
                          <code>{caseRecord.case_id}</code>
                          <span>{caseRecord.status}</span>
                        </header>
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
                        {tx.ready ? (
                          <a
                            href={tx.url}
                            download
                            aria-label={`下载条件组 ${caseRecord.case_id} TX`}
                          >
                            <Download size={14} aria-hidden="true" />
                            下载 TX
                          </a>
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
                <span>50 条逐观测明细</span>
                <small>
                  按条件组与合同测项分组；默认折叠
                </small>
              </summary>
              <div className="table-scroll">
                <table className="audio-table basic-uplink-items-table sct-observation-table">
                  <caption className="sr-only">
                    20 组测试的 50 条逐观测结果
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
                                  {currentSnapshot.cases.find(
                                    (caseRecord) =>
                                      caseRecord.case_id === caseId,
                                  )?.status ?? "unknown"}
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
                              <ItemStatusBadge status={item.status} />
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

      <ActiveSubpage active={activeWaveforms} label="20 组波形与梅尔频谱">
        {isWbWired && dataset && datasetId ? (
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
                {isWbWired
                  ? datasetError || "正在等待 20 组测试集元数据。"
                  : "20 组波形目前只支持 WB-wired。"}
              </p>
            </div>
          </div>
        )}
      </ActiveSubpage>

      <ActiveSubpage active={activeDescription} label="20 组测试说明">
        <div className="sct-description">
          <section>
            <span className="basic-uplink-eyebrow">SCOPE</span>
            <h3>观测数量</h3>
            <p>
              10 个语音条件组 × 3 个指标 = 30 次测量；10 个 CNG
              条件组 × 2 个指标 = 20 次测量；合计 50 次测量。
            </p>
          </section>
          <section>
            <span className="basic-uplink-eyebrow">CONTRACT</span>
            <h3>合同口径</h3>
            <p>
              合同测项是 B24–B28 共 5 个；50 是条件组与指标组合后的观测数，
              不是 50 个正式合同测项。
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
