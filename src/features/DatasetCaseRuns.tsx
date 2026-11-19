import {
  useEffect,
  useId,
  useMemo,
  useState,
} from "react";
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Download,
  ListChecks,
  Play,
  RefreshCw,
  XCircle,
} from "lucide-react";
import OutputAudioPreview from "./OutputAudioPreview.tsx";
import { PrecomputedAudioFigures } from "./ScopedAcousticWaveforms";
import {
  SCOPED_INPUT_SIGNAL_ROLES,
  type ScopedInputSignalRole,
  type ScopedPreviewVisualTrack,
} from "./scoped-acoustic-model";
import type { FiveSignalMode } from "./five-signal-flow-model.ts";
import {
  caseRunIdentityKey,
  mergeCaseRunHistory,
  readAcceptanceCaseCatalog,
  readBasicContractCatalog,
  readSingleCaseSelection,
  sameCaseRunIdentity,
  type AcceptanceCaseRecord,
  type BasicContractCatalog,
  type CaseRunHistory,
  type CaseRunIdentity,
  type CaseRunResult,
  type CaseRunScope,
  type CaseRunSnapshot,
} from "./dataset-case-runs-model.ts";
import "./dataset-case-runs.css";

const ACCEPTANCE_URL = "/api/alignment/acceptance";
const BATCHES_URL = `${ACCEPTANCE_URL}/batches`;
const INTEGER_FORMAT = new Intl.NumberFormat("zh-CN");
const VALUE_FORMAT = new Intl.NumberFormat("zh-CN", {
  maximumFractionDigits: 3,
});

const CONDITION_LABELS: Record<string, string> = {
  mode: "模式",
  direction: "方向",
  scene: "场景",
  volume: "音量",
  window_ms: "窗口",
  input_level_db: "输入",
};

const INPUT_ROLE_LABELS: Record<ScopedInputSignalRole, string> = {
  microphone: "麦克风混合输入 · microphone",
  far_end: "远端播放信号 / 回声参考 · far_end",
  reference: "干净近端参考 · reference",
  noise: "独立噪声输入 · noise",
};

function formatSampleRate(sampleRate: number): string {
  return `${(sampleRate / 1000).toLocaleString("zh-CN", {
    maximumFractionDigits: 2,
  })} kHz`;
}

function inputAudioUrl(
  datasetId: string,
  caseId: string,
  role: ScopedInputSignalRole,
  sha256: string | undefined,
): string {
  const base = `${ACCEPTANCE_URL}/datasets/${encodeURIComponent(datasetId)}/cases/${encodeURIComponent(caseId)}/inputs/${encodeURIComponent(role)}`;
  return sha256 ? `${base}?v=${encodeURIComponent(sha256)}` : base;
}

function primaryTrackVisual(
  caseVisuals: Record<string, ScopedPreviewVisualTrack> | undefined,
): ScopedPreviewVisualTrack | null {
  if (!caseVisuals) return null;
  for (const role of SCOPED_INPUT_SIGNAL_ROLES) {
    if (caseVisuals[role]) return caseVisuals[role];
  }
  return Object.values(caseVisuals)[0] ?? null;
}

const STATUS_LABELS: Record<string, string> = {
  queued: "排队中",
  running: "执行中",
  completed: "已完成",
  failed: "执行失败",
  cancelled: "已取消",
  pending: "等待计算",
  pass: "工程通过",
  fail: "工程失败",
  review_required: "需复核",
  not_measured: "不可测",
  invalid: "无效",
};

type DatasetIdentity = {
  dataset_id: string;
  name: string;
  manifest_sha256?: string;
};

type Props = {
  dataset: DatasetIdentity | null;
  mode: FiveSignalMode;
  algorithm: string;
  scope: CaseRunScope;
  currentSnapshot: CaseRunSnapshot | null;
  canRun: boolean;
  onRunCase: (caseId: string) => void | Promise<void>;
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

function errorMessage(reason: unknown, fallback: string): string {
  return reason instanceof Error && reason.message
    ? reason.message
    : fallback;
}

function apiErrorDetail(body: unknown): string {
  if (!isRecord(body)) return "";
  const detail = body.detail;
  if (typeof detail === "string") return detail;
  if (!Array.isArray(detail)) return "";
  return detail
    .map((entry) =>
      isRecord(entry) && typeof entry.msg === "string" ? entry.msg : "",
    )
    .filter(Boolean)
    .join("；");
}

async function requestJson(
  url: string,
  signal: AbortSignal,
  operation: string,
): Promise<unknown> {
  const response = await fetch(url, { cache: "no-store", signal });
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
  return body;
}

function contractApiRoute(scope: CaseRunScope): string {
  return scope === "basic_uplink" ? "basic-uplink" : "basic-downlink";
}

function formatConditionValue(
  key: string,
  value: string | number | null,
): string {
  if (value === null || value === "") return "—";
  const formatted =
    typeof value === "number" ? VALUE_FORMAT.format(value) : value;
  if (key === "window_ms") return `${formatted} ms`;
  if (key === "input_level_db") {
    return `${formatted} dB`;
  }
  return formatted;
}

function statusLabel(status: string): string {
  return STATUS_LABELS[status] ?? (status || "状态未返回");
}

function statusClass(status: string): string {
  return /^[a-z_]+$/.test(status) ? status : "unknown";
}

function CaseStatusIcon({ status }: { status: string }) {
  if (status === "completed" || status === "pass") {
    return <CheckCircle2 size={14} aria-hidden="true" />;
  }
  if (status === "failed" || status === "fail" || status === "invalid") {
    return <XCircle size={14} aria-hidden="true" />;
  }
  return <AlertCircle size={14} aria-hidden="true" />;
}

function caseDownloads(
  batchId: string,
  caseId: string,
  outputLabel: "TX" | "RX",
): { report: string; output: string } {
  const root = `${BATCHES_URL}/${encodeURIComponent(
    batchId,
  )}/cases/${encodeURIComponent(caseId)}`;
  return {
    report: `${root}/assessment`,
    output: `${root}/outputs/${outputLabel.toLowerCase()}`,
  };
}

function failureText(result: CaseRunResult): string {
  if (result.error) return result.error;
  const reasons = result.items
    .filter((item) =>
      ["fail", "invalid", "not_measured"].includes(item.status),
    )
    .map((item) => item.reason)
    .filter((reason): reason is string => Boolean(reason));
  return [...new Set(reasons)].join("；");
}

function CaseResult({
  record,
  contract,
  result,
  outputLabel,
  outputPlaying,
  onToggleOutputPlayback,
}: {
  record: AcceptanceCaseRecord;
  contract: BasicContractCatalog;
  result: CaseRunResult | undefined;
  outputLabel: "TX" | "RX";
  outputPlaying: boolean;
  onToggleOutputPlayback: () => void;
}) {
  const contractById = new Map(
    contract.items.map((item) => [item.id, item]),
  );
  const resultById = new Map(
    (result?.items ?? []).map((item) => [item.item_id, item]),
  );
  const failure = result ? failureText(result) : "";
  const canDownload =
    result?.status === "completed" && Boolean(result.report_id);
  const links =
    canDownload && result
      ? caseDownloads(result.batch_id, result.case_id, outputLabel)
      : null;

  return (
    <>
      <ul
        className="dataset-case-runs-contracts"
        aria-label={`${record.case_id} 合同测项`}
      >
        {record.item_ids.map((itemId) => {
          const contractItem = contractById.get(itemId);
          const itemResult = resultById.get(itemId);
          return (
            <li key={itemId}>
              <div>
                <strong>{contractItem?.test_name ?? itemId}</strong>
                <code>{itemId}</code>
              </div>
              {itemResult ? (
                <span
                  className={`dataset-case-runs-item-result is-${statusClass(
                    itemResult.status,
                  )}`}
                >
                  <b>
                    {itemResult.value === null
                      ? "无数值"
                      : VALUE_FORMAT.format(itemResult.value)}
                  </b>
                  <small>{statusLabel(itemResult.status)}</small>
                </span>
              ) : (
                <span className="dataset-case-runs-item-result is-pending">
                  <b>无数值</b>
                  <small>尚未执行本组</small>
                </span>
              )}
            </li>
          );
        })}
      </ul>

      {failure ? (
        <p className="dataset-case-runs-failure" role="status">
          <AlertCircle size={13} aria-hidden="true" />
          {failure}
        </p>
      ) : null}

      {links ? (
        <div className="dataset-case-runs-downloads">
          <a
            href={links.report}
            download
            aria-label={`下载单组 ${record.case_id} 条件组报告`}
          >
            <Download size={13} aria-hidden="true" />
            条件组报告
          </a>
          <a
            href={links.output}
            download
            aria-label={`下载单组 ${record.case_id} ${outputLabel}`}
          >
            <Download size={13} aria-hidden="true" />
            下载 {outputLabel}
          </a>
          <OutputAudioPreview
            caseId={record.case_id}
            label={`${outputLabel} 输出音频`}
            url={links.output}
            playing={outputPlaying}
            onToggle={onToggleOutputPlayback}
          />
        </div>
      ) : null}
    </>
  );
}

export default function DatasetCaseRuns({
  dataset,
  mode,
  algorithm,
  scope,
  currentSnapshot,
  canRun,
  onRunCase,
}: Props) {
  const headingId = useId();
  const outputLabel = scope === "basic_uplink" ? "TX" : "RX";
  const [contract, setContract] = useState<BasicContractCatalog | null>(null);
  const [cases, setCases] = useState<
    ReturnType<typeof readAcceptanceCaseCatalog> | null
  >(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [refresh, setRefresh] = useState(0);
  const [runningCaseId, setRunningCaseId] = useState("");
  const [runError, setRunError] = useState("");
  const [outputPlaybackKey, setOutputPlaybackKey] = useState("");
  const [inputPlaybackKey, setInputPlaybackKey] = useState("");
  const [expandedWaveforms, setExpandedWaveforms] = useState<
    Record<string, boolean>
  >({});
  const [showMelByCase, setShowMelByCase] = useState<Record<string, boolean>>(
    {},
  );
  const toggleOutputPlayback = (key: string) => {
    setInputPlaybackKey("");
    setOutputPlaybackKey((current) => (current === key ? "" : key));
  };
  const toggleInputPlayback = (key: string) => {
    setOutputPlaybackKey("");
    setInputPlaybackKey((current) => (current === key ? "" : key));
  };
  const [historyState, setHistoryState] = useState<{
    identityKey: string;
    results: CaseRunHistory;
  }>({ identityKey: "", results: {} });

  useEffect(() => {
    setExpandedWaveforms({});
    setShowMelByCase({});
    setInputPlaybackKey("");
    setOutputPlaybackKey("");
  }, [dataset?.dataset_id, mode, scope]);

  useEffect(() => {
    if (!dataset) {
      setContract(null);
      setCases(null);
      setLoading(false);
      setError("");
      return;
    }
    const controller = new AbortController();
    setLoading(true);
    setError("");
    const contractParams = new URLSearchParams({ mode });
    const caseParams = new URLSearchParams({
      test_scope: scope,
      mode,
    });
    Promise.all([
      requestJson(
        `${ACCEPTANCE_URL}/${contractApiRoute(scope)}/catalog?${contractParams}`,
        controller.signal,
        "读取基础范围合同目录",
      ),
      requestJson(
        `${ACCEPTANCE_URL}/datasets/${encodeURIComponent(
          dataset.dataset_id,
        )}/cases?${caseParams}`,
        controller.signal,
        "读取数据集单组目录",
      ),
    ])
      .then(([contractBody, caseBody]) => {
        const nextContract = readBasicContractCatalog(
          contractBody,
          scope,
          mode,
        );
        const nextCases = readAcceptanceCaseCatalog(caseBody, {
          datasetId: dataset.dataset_id,
          datasetManifestSha256: dataset.manifest_sha256,
          scope,
          mode,
          contractItemIds: nextContract.items.map((item) => item.id),
        });
        if (controller.signal.aborted) return;
        setContract(nextContract);
        setCases(nextCases);
      })
      .catch((reason: unknown) => {
        if (controller.signal.aborted) return;
        setContract(null);
        setCases(null);
        setError(errorMessage(reason, "读取数据集单组目录失败"));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [
    dataset?.dataset_id,
    dataset?.manifest_sha256,
    mode,
    refresh,
    scope,
  ]);

  const identity = useMemo<CaseRunIdentity | null>(() => {
    if (!dataset || !cases) return null;
    return {
      dataset_id: dataset.dataset_id,
      dataset_manifest_sha256: cases.dataset_manifest_sha256,
      algorithm_cluster_id: algorithm,
      test_scope: scope,
      selected_mode: mode,
    };
  }, [algorithm, cases, dataset, mode, scope]);
  const identityKey = identity ? caseRunIdentityKey(identity) : "";

  useEffect(() => {
    if (!identity) {
      setHistoryState((current) =>
        current.identityKey
          ? { identityKey: "", results: {} }
          : current,
      );
      return;
    }
    setHistoryState((current) => {
      const results =
        current.identityKey === identityKey ? current.results : {};
      return {
        identityKey,
        results: currentSnapshot
          ? mergeCaseRunHistory(results, currentSnapshot, identity)
          : results,
      };
    });
  }, [currentSnapshot, identity, identityKey]);

  const snapshotIdentityError = useMemo(() => {
    if (!identity || !currentSnapshot) return "";
    if (sameCaseRunIdentity(currentSnapshot, identity)) return "";
    return "当前快照与所选数据集、模式或算法不同；不会把旧结果串入本组目录。";
  }, [currentSnapshot, identity]);

  async function runCase(caseId: string) {
    if (
      !canRun ||
      loading ||
      Boolean(runningCaseId) ||
      !contract ||
      !cases
    ) {
      return;
    }
    setRunningCaseId(caseId);
    setRunError("");
    try {
      await onRunCase(caseId);
    } catch (reason) {
      setRunError(errorMessage(reason, "创建单组批次失败"));
    } finally {
      setRunningCaseId("");
    }
  }

  const controlsDisabled =
    !canRun || loading || Boolean(runningCaseId) || !contract || !cases;

  return (
    <section
      className="dataset-case-runs"
      aria-labelledby={headingId}
    >
      <header className="dataset-case-runs-header">
        <div>
          <span className="basic-uplink-eyebrow">
            CONDITION GROUPS
          </span>
          <h3 id={headingId}>数据集单组执行</h3>
          <p>
            按当前 {mode} 的真实条件组逐组执行；每组只计算本组
            item_ids，正式 ITU-T 合规仍未测。
          </p>
        </div>
        <div className="dataset-case-runs-header-meta">
          <span>
            {cases
              ? `${INTEGER_FORMAT.format(
                  cases.case_count,
                )} 组 · ${INTEGER_FORMAT.format(
                  cases.observation_count,
                )} 个观测`
              : loading
                ? "正在读取单组目录"
                : "单组目录不可用"}
          </span>
          <button
            className="button secondary"
            type="button"
            disabled={loading || Boolean(runningCaseId)}
            onClick={() => setRefresh((current) => current + 1)}
          >
            <RefreshCw size={14} aria-hidden="true" />
            刷新目录
          </button>
        </div>
      </header>

      {error ? (
        <div className="dataset-case-runs-error" role="alert">
          <AlertCircle size={17} aria-hidden="true" />
          <div>
            <strong>单组目录读取失败</strong>
            <p>{error}</p>
          </div>
        </div>
      ) : null}
      {runError ? (
        <div className="dataset-case-runs-error" role="alert">
          <AlertCircle size={17} aria-hidden="true" />
          <div>
            <strong>单组批次创建失败</strong>
            <p>{runError}</p>
          </div>
        </div>
      ) : null}
      {snapshotIdentityError ? (
        <p className="dataset-case-runs-identity" role="status">
          {snapshotIdentityError}
        </p>
      ) : null}

      {!dataset ? (
        <div className="dataset-case-runs-empty">
          <ListChecks size={20} aria-hidden="true" />
          <p>先选择已导入测试集，再显示可单独执行的条件组。</p>
        </div>
      ) : loading && !cases ? (
        <div className="dataset-case-runs-empty" role="status">
          <RefreshCw size={20} aria-hidden="true" />
          <p>正在校验合同目录、数据集身份与条件组。</p>
        </div>
      ) : cases && contract ? (
        <div
          className="dataset-case-runs-list"
          role="region"
          aria-label={`${dataset.name} 单组执行列表，可局部滚动`}
          tabIndex={0}
        >
          {cases.cases.map((record, index) => {
            const result = historyState.results[record.case_id];
            const selection = currentSnapshot
              ? readSingleCaseSelection(currentSnapshot)
              : null;
            const active =
              selection?.selected_case_id === record.case_id &&
              sameCaseRunIdentity(currentSnapshot!, identity!);
            const status = result?.status ?? "not_started";
            const conditions = Object.entries(record.conditions);
            const caseVisuals = cases.visuals?.cases[record.case_id];
            const primaryVisual = primaryTrackVisual(caseVisuals);
            const isWaveExpanded = Boolean(expandedWaveforms[record.case_id]);
            const isMelShown = Boolean(showMelByCase[record.case_id]);
            const visualTracks = caseVisuals
              ? SCOPED_INPUT_SIGNAL_ROLES.map((role) => caseVisuals[role]).filter(
                  (track): track is ScopedPreviewVisualTrack => Boolean(track),
                )
              : [];
            const peakScale = Math.max(
              0,
              ...visualTracks.map((track) => track.peak),
            );
            const melCeilingDb = Math.max(
              -120,
              ...visualTracks.map((track) => track.mel_ceiling_db),
            );
            return (
              <article
                key={record.case_id}
                className={`dataset-case-runs-card${
                  active ? " is-active" : ""
                }`}
              >
                <header>
                  <div className="dataset-case-runs-card-title">
                    <span>{String(index + 1).padStart(2, "0")}</span>
                    <div>
                      <code>{record.case_id}</code>
                      <strong>
                        {record.item_ids.length} 个合同观测
                        {primaryVisual
                          ? ` · ${primaryVisual.duration_s.toFixed(2)} s · ${formatSampleRate(primaryVisual.sample_rate)}`
                          : ""}
                      </strong>
                    </div>
                  </div>
                  <span
                    className={`dataset-case-runs-status is-${statusClass(
                      status,
                    )}`}
                  >
                    <CaseStatusIcon status={status} />
                    {statusLabel(status)}
                  </span>
                </header>

                <dl className="dataset-case-runs-conditions">
                  {conditions.map(([key, value]) => (
                    <div key={key}>
                      <dt>{CONDITION_LABELS[key] ?? key}</dt>
                      <dd>{formatConditionValue(key, value)}</dd>
                    </div>
                  ))}
                  {primaryVisual ? (
                    <div>
                      <dt>输入采样率</dt>
                      <dd>{formatSampleRate(primaryVisual.sample_rate)}</dd>
                    </div>
                  ) : null}
                </dl>

                {caseVisuals && primaryVisual ? (
                  <div className="dataset-case-runs-wave-section">
                    <div className="dataset-case-runs-wave-bar">
                      <button
                        type="button"
                        className="dataset-case-runs-wave-toggle"
                        aria-expanded={isWaveExpanded}
                        onClick={() =>
                          setExpandedWaveforms((current) => ({
                            ...current,
                            [record.case_id]: !current[record.case_id],
                          }))
                        }
                      >
                        <Activity size={13} aria-hidden="true" />
                        <span>
                          {isWaveExpanded
                            ? "收起四路输入波形与试听"
                            : `查看四路输入波形与试听（${formatSampleRate(primaryVisual.sample_rate)}）`}
                        </span>
                        {isWaveExpanded ? (
                          <ChevronUp size={13} aria-hidden="true" />
                        ) : (
                          <ChevronDown size={13} aria-hidden="true" />
                        )}
                      </button>
                      {isWaveExpanded ? (
                        <button
                          type="button"
                          className="dataset-case-runs-mel-toggle"
                          aria-pressed={isMelShown}
                          onClick={() =>
                            setShowMelByCase((current) => ({
                              ...current,
                              [record.case_id]: !current[record.case_id],
                            }))
                          }
                        >
                          {isMelShown ? "隐藏 Mel 频谱" : "显示 Mel 频谱"}
                        </button>
                      ) : null}
                    </div>
                    {isWaveExpanded ? (
                      <div className="dataset-case-runs-wave-grid">
                        {SCOPED_INPUT_SIGNAL_ROLES.map((role) => {
                          const visual = caseVisuals[role];
                          if (!visual) return null;
                          const playKey = `${record.case_id}:${role}`;
                          const playing = inputPlaybackKey === playKey;
                          const url = inputAudioUrl(
                            dataset.dataset_id,
                            record.case_id,
                            role,
                            record.signals_sha256[role],
                          );
                          return (
                            <div
                              key={role}
                              className={`dataset-case-runs-wave-track${
                                visual.peak === 0 ? " is-silent" : ""
                              }`}
                              data-signal-role={role}
                            >
                              <div className="dataset-case-runs-wave-meta">
                                <strong>{INPUT_ROLE_LABELS[role]}</strong>
                                <span>
                                  {visual.peak === 0 ? "数字静音 · " : ""}
                                  {visual.duration_s.toFixed(2)} s ·{" "}
                                  {formatSampleRate(visual.sample_rate)} ·{" "}
                                  {visual.channels} 通道
                                </span>
                              </div>
                              <PrecomputedAudioFigures
                                visual={visual}
                                label={INPUT_ROLE_LABELS[role]}
                                caseId={record.case_id}
                                peakScale={peakScale}
                                melCeilingDb={melCeilingDb}
                                showMel={isMelShown}
                              />
                              <div className="dataset-case-runs-wave-audio">
                                {playing ? (
                                  <audio
                                    controls
                                    autoPlay
                                    preload="none"
                                    src={url}
                                    aria-label={`试听 ${record.case_id} ${INPUT_ROLE_LABELS[role]}`}
                                  >
                                    当前浏览器不支持音频播放。
                                  </audio>
                                ) : (
                                  <button
                                    type="button"
                                    onClick={() => toggleInputPlayback(playKey)}
                                    aria-label={`试听 ${record.case_id} ${INPUT_ROLE_LABELS[role]}`}
                                  >
                                    <Play
                                      size={11}
                                      fill="currentColor"
                                      aria-hidden="true"
                                    />
                                    按需试听
                                  </button>
                                )}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    ) : null}
                  </div>
                ) : null}

                <CaseResult
                  record={record}
                  contract={contract}
                  result={result}
                  outputLabel={outputLabel}
                  outputPlaying={
                    result
                      ? outputPlaybackKey === `${result.batch_id}:${record.case_id}:${outputLabel.toLowerCase()}`
                      : false
                  }
                  onToggleOutputPlayback={() =>
                    toggleOutputPlayback(
                      `${result?.batch_id ?? ""}:${record.case_id}:${outputLabel.toLowerCase()}`,
                    )
                  }
                />

                <footer>
                  <span>
                    {result
                      ? `批次 ${result.batch_id.slice(0, 8)}`
                      : "尚未执行本组"}
                  </span>
                  <button
                    className="button secondary"
                    type="button"
                    data-testid="run-single-case"
                    data-case-id={record.case_id}
                    aria-label={`运行本组 ${record.case_id}`}
                    disabled={
                      controlsDisabled || runningCaseId === record.case_id
                    }
                    onClick={() => void runCase(record.case_id)}
                  >
                    <Play size={14} fill="currentColor" aria-hidden="true" />
                    {runningCaseId === record.case_id
                      ? "正在创建本组批次…"
                      : "运行本组"}
                  </button>
                </footer>
              </article>
            );
          })}
        </div>
      ) : !loading && !error ? (
        <div className="dataset-case-runs-empty">
          <ListChecks size={20} aria-hidden="true" />
          <p>当前测试集没有可执行的条件组。</p>
        </div>
      ) : null}
    </section>
  );
}
