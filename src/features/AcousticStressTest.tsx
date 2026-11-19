import { memo, useCallback, useEffect, useId, useMemo, useRef, useState, type ReactNode } from "react";
import { Activity, AlertCircle, Clock3, Download, History, ListChecks, Play, RefreshCw, ShieldAlert } from "lucide-react";
import { ACOUSTIC_SIGNAL_LABELS } from "../domain/acoustic-terminology.ts";
import {
  algorithmClusterById,
  algorithmClusterGroupsForMode,
  algorithmClusterLabel,
  algorithmClustersForMode,
  isAlgorithmClusterSupportedForMode,
  sampleRateForBusinessMode,
} from "./rate-algorithm-clusters";
import { FIVE_SIGNAL_MODES, type FiveSignalMode } from "./five-signal-flow-model";
import AlgorithmClusterSummary from "./AlgorithmClusterSummary";
import { PrecomputedAudioFigures } from "./ScopedAcousticWaveforms";
import OutputAudioPreview from "./OutputAudioPreview.tsx";
import {
  SCENE_SUBPAGES,
  SCENE_SUBPAGE_LABELS,
  SCOPED_BATCH_LABELS,
  SCOPED_INPUT_SIGNAL_ROLES,
  SCOPED_STATUS_LABELS,
  sceneSubpageFromHash,
  sceneSubpageHash,
  sceneSubpageRoute,
  scopedMetricUnit,
  scopedValue,
  type SceneSubpageId,
  type ScopedItem,
  type ScopedPreviewVisualTrack,
  type ScopedTestDefinition,
  type ScopedTestPageProps,
} from "./scoped-acoustic-model";
import {
  STRESS_BATCHES_URL,
  STRESS_CATEGORIES,
  STRESS_CATEGORY_LABELS,
  STRESS_DATASETS_URL,
  STRESS_PREVIEW_URL,
  readStressBatch,
  readStressDatasets,
  readStressPreview,
  stressDynamicMicrophoneEvidence,
  stressSampleRate,
  stressRxEvidence,
  stressTxEvidence,
  type StressBatch,
  type StressBatchCase,
  type StressCategory,
  type StressDiagnosticSegment,
  type StressDiagnostics,
  type StressDatasetView,
  type StressPreview,
  type StressPreviewCase,
  type StressScope,
  type StressSegmentActivity,
} from "./acoustic-stress-model";
import { businessModeLabel as modeLabel } from "../domain/business-modes.ts";
import "./basic-uplink-test.css";
import "./scoped-acoustic-test.css";
import "./acoustic-stress-test.css";

const POLL_INTERVAL_MS = 2000;
const DATE_FORMAT = new Intl.DateTimeFormat("zh-CN", {
  timeZone: "Asia/Shanghai",
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
});
const ACTIVITY_LABELS = {
  quiet: "静音",
  far_only: "远端单讲",
  near_only: "近端单讲",
  double_talk: "双讲",
  noise_only: "噪声尾段",
} as const;
const DIAGNOSTIC_STATUS_LABELS = {
  review_required: "需复核",
  not_measurable: "不可测",
  not_applicable: "不适用",
} as const;
const DIAGNOSTIC_MODE_LABELS = {
  not_applicable: "不适用",
  far: "远端",
  near: "近端",
  double_talk: "双讲",
  noise: "噪声",
} as const;
const CASE_STATUS_LABELS = {
  pending: "等待",
  running: "执行中",
  completed: "已完成",
  failed: "失败",
} as const;
const TRACK_DESCRIPTIONS = {
  microphone: "执行时先经 VCP RX 生成回声参考，再由干净近端参考 + RX 后分段卷积回声 + 独立噪声实时合成。",
  far_end: "远端播放信号 / 回声参考，是送往扬声器的同步参考。",
  reference: "干净近端参考，是近端输入的真值。",
  noise: "独立噪声输入，用于背景噪声和噪声中的回声。",
} as const;

type Props = ScopedTestPageProps & {
  scope: StressScope;
  definition: ScopedTestDefinition;
};

type CategoryFilter = "all" | StressCategory;
type PlayingTrack = {
  caseId: string;
  role: (typeof SCOPED_INPUT_SIGNAL_ROLES)[number] | "rx" | "tx";
} | null;

type LatestCaseResult = {
  batch: StressBatch | null;
  caseRecord: StressBatchCase | undefined;
  item: StressBatch["items"][number] | undefined;
};

const EMPTY_LATEST_CASE_RESULT: LatestCaseResult = {
  batch: null,
  caseRecord: undefined,
  item: undefined,
};

type SingleCaseError = {
  caseId: string;
  message: string;
} | null;

type AsyncState<T> = {
  value: T;
  loading: boolean;
  error: string;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

async function requestJson(url: string, init: RequestInit, operation: string): Promise<unknown> {
  const response = await fetch(url, init);
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    throw new Error(response.ok ? `${operation}返回了无法解析的 JSON` : `${operation}失败（HTTP ${response.status}）`);
  }
  if (!response.ok) {
    const detail = isRecord(body) && typeof body.detail === "string" ? body.detail : "";
    throw new Error(detail || `${operation}失败（HTTP ${response.status}）`);
  }
  return body;
}

function errorMessage(reason: unknown, fallback: string): string {
  return reason instanceof Error && reason.message ? reason.message : fallback;
}

function categoryLabel(scope: StressScope, category: string): string {
  return (STRESS_CATEGORY_LABELS[scope] as Record<string, string>)[category] ?? category;
}

function categoryCounts(cases: StressPreviewCase[], scope: StressScope): Record<string, number> {
  const counts = Object.fromEntries(STRESS_CATEGORIES[scope].map((category) => [category, 0])) as Record<string, number>;
  for (const previewCase of cases) {
    const category = previewCase.provenance.workload_category;
    counts[category] = (counts[category] ?? 0) + 1;
  }
  return counts;
}

function activitySummary(previewCase: StressPreviewCase): string {
  return previewCase.provenance.activity_segments
    .map((segment) => `${segment.start_s.toFixed(Number.isInteger(segment.start_s) ? 0 : 2)}–${segment.end_s.toFixed(Number.isInteger(segment.end_s) ? 0 : 2)}s ${ACTIVITY_LABELS[segment.label]}`)
    .join(" · ");
}

function hasSegmentActivity(segment: StressDiagnosticSegment): segment is StressDiagnosticSegment & { activity: StressSegmentActivity } {
  return "activity" in segment;
}

function diagnosticValue(value: number | null, digits = 2): string {
  return value === null ? "—" : value.toLocaleString("zh-CN", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function activeSeconds(samples: number, sampleRate: number): string {
  return (samples / sampleRate).toLocaleString("zh-CN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function mainProxyLabel(scope: StressScope): string {
  return scope === "echo" ? "麦克风输入回声耦合代理（dB）" : "双讲功率区间裁剪代理（%）";
}

function activityCoverage(value: number): string {
  return `${diagnosticValue(value * 100, 1)}%`;
}

function ActivityEvidence({ activity, sampleRate }: { activity: StressSegmentActivity; sampleRate: number }) {
  const evidence = [
    ["近端", activity.near_active_samples, activity.near_coverage],
    ["远端", activity.far_active_samples, activity.far_coverage],
    ["重叠", activity.joint_active_samples, activity.joint_coverage],
  ] as const;
  return (
    <div className="ast-activity-evidence">
      {evidence.map(([label, samples, coverage]) => (
        <span key={label}>
          <b>{label}</b>
          {activeSeconds(samples, sampleRate)} s · {activityCoverage(coverage)}
        </span>
      ))}
      <span className={`ast-activity-validity ${activity.valid ? "is-valid" : "is-invalid"}`}>{activity.valid ? "活动门限通过" : "活动门限未通过"}</span>
      {activity.echo_guarded_active_samples !== undefined ? (
        <span className="ast-activity-guard">
          {activity.echo_delay_guard_ms ?? "—"} ms 回声保护后 {activeSeconds(activity.echo_guarded_active_samples, sampleRate)} s
        </span>
      ) : null}
    </div>
  );
}

function SegmentDiagnostics({ scope, diagnostics }: { scope: StressScope; diagnostics: StressDiagnostics }) {
  const gate = diagnostics.schema_version === "acoustic-stress-diagnostics-v2" ? diagnostics.activity_gate : null;
  return (
    <details className="ast-diagnostics">
      <summary>
        <span>逐段诊断</span>
        <span>
          {diagnostics.segments.length} 段 · {DIAGNOSTIC_STATUS_LABELS[diagnostics.status]}
        </span>
      </summary>
      <p className="ast-diagnostics-note">
        主代理、TX 与参考指标均为工程代理，不代表正式 TCLw/ERLE 或 A1/A2 分类。“—” 表示后端未提供、不可测或不适用，不等同于 0；噪声、静音和活动门限未通过等原因见“分段原因”列。
        {gate
          ? ` 活动门限：${gate.frame_ms} ms 帧 / ${gate.hop_ms} ms 步进，最短 ${diagnosticValue(gate.min_active_s * 1000, 0)} ms，覆盖率 ≥ ${activityCoverage(gate.min_coverage)}。`
          : " 历史工程口径，未提供逐帧活动证据。"}
      </p>
      <div className="ast-diagnostics-scroll">
        <table>
          <caption className="sr-only">逐时间段工程诊断指标与活动证据</caption>
          <thead>
            <tr>
              <th scope="col">时间段</th>
              <th scope="col">分段 / 模式 / 状态</th>
              <th scope="col">{mainProxyLabel(scope)}</th>
              <th scope="col">TX 残余降低（含噪声明）/ dB</th>
              <th scope="col">干净近端参考投影增益 / dB</th>
              <th scope="col">干净近端参考相关系数</th>
              <th scope="col">逐帧活动证据</th>
              <th scope="col">分段原因</th>
            </tr>
          </thead>
          <tbody>
            {diagnostics.segments.map((segment, index) => (
              <tr key={`${segment.label}-${segment.start_s}-${segment.end_s}-${index}`}>
                <td className="ast-diagnostic-time">
                  {segment.start_s.toFixed(2)}–{segment.end_s.toFixed(2)} s
                </td>
                <td>
                  <div className="ast-segment-identity">
                    <strong>{ACTIVITY_LABELS[segment.label]}</strong>
                    <span>{DIAGNOSTIC_MODE_LABELS[segment.mode]}</span>
                    <span className={`ast-segment-status is-${segment.status}`}>{DIAGNOSTIC_STATUS_LABELS[segment.status]}</span>
                  </div>
                </td>
                <td className="ast-diagnostic-value">{diagnosticValue(segment.metrics.primary)}</td>
                <td className="ast-diagnostic-value">{diagnosticValue(segment.metrics.tx_residual_reduction_db)}</td>
                <td className="ast-diagnostic-value">{diagnosticValue(segment.metrics.cleanref_projection_gain_db)}</td>
                <td className="ast-diagnostic-value">{diagnosticValue(segment.metrics.cleanref_correlation, 3)}</td>
                <td>
                  {hasSegmentActivity(segment) ? (
                    <ActivityEvidence activity={segment.activity} sampleRate={diagnostics.sample_rate_hz} />
                  ) : (
                    <span className="ast-diagnostics-legacy">历史工程口径，未提供逐帧活动证据</span>
                  )}
                </td>
                <td>
                  <div className="ast-diagnostic-reasons">
                    <span>诊断：{segment.reason || "后端未提供分段原因"}</span>
                    {hasSegmentActivity(segment) ? <small>活动：{segment.activity.reason || "后端未提供活动原因"}</small> : null}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}

function designedOverlapSeconds(previewCase: StressPreviewCase): number {
  return previewCase.provenance.activity_segments.filter((segment) => segment.label === "double_talk").reduce((total, segment) => total + segment.end_s - segment.start_s, 0);
}

function swbResamplingNotice(previewCase: StressPreviewCase, mode: FiveSignalMode): string {
  const sourceInfo = [previewCase.provenance.near_speech, previewCase.provenance.far_input];
  if (mode.startsWith("NB-")) {
    const has16To8 = sourceInfo.some((value) => {
      if (!isRecord(value) || !isRecord(value.resampling)) return false;
      return value.resampling.source_rate_hz === 16_000 && value.resampling.target_rate_hz === 8_000;
    });
    return has16To8 ? "8 kHz 窄带处理适配；语音源由 16 kHz 经 3.4 kHz 窄带低通滤波与多相降采样至 8 kHz。" : "";
  }
  if (!mode.startsWith("SWB-")) return "";
  const has16To48 = sourceInfo.some((value) => {
    if (!isRecord(value) || !isRecord(value.resampling)) return false;
    return value.resampling.source_rate_hz === 16_000 && value.resampling.target_rate_hz === 48_000;
  });
  return has16To48 ? "48 kHz 处理适配；语音源 16 kHz，不覆盖 8 kHz 以上语音带宽；噪声高频不等于语音证据。" : "";
}

function caseTitle(scope: StressScope, previewCase: StressPreviewCase): string {
  return `${categoryLabel(scope, previewCase.provenance.workload_category)} ${String(previewCase.provenance.category_index).padStart(2, "0")}`;
}

function volumeLabel(value: string | number | null): string {
  if (typeof value === "number") return `${value} dB`;
  if (value === "nominal") return "标称音量";
  if (value === "maximum") return "最大音量";
  if (value === "recommended") return "推荐音量";
  return value ?? "—";
}

function formatRate(sampleRate: number): string {
  return `${(sampleRate / 1000).toLocaleString("zh-CN", {
    maximumFractionDigits: 2,
  })} kHz`;
}

function formatDuration(seconds: number): string {
  return `${seconds.toLocaleString("zh-CN", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })} s`;
}

function storageKey(scope: StressScope, mode: FiveSignalMode, datasetId: string, algorithmClusterId: string): string {
  return ["acoustic-stress", scope, mode, datasetId, algorithmClusterId, "latest"].join(":");
}

function readStoredBatchId(key: string): string {
  if (!key) return "";
  try {
    return window.localStorage.getItem(key) ?? "";
  } catch {
    return "";
  }
}

function writeStoredBatchId(key: string, batchId: string): void {
  if (!key || !batchId) return;
  try {
    window.localStorage.setItem(key, batchId);
  } catch {
    // Storage denial must not hide a successfully created server batch.
  }
}

type BatchIdentity = {
  scope: StressScope;
  mode: FiveSignalMode;
  datasetId: string;
  algorithmClusterId: string;
  caseId?: string;
};

function batchMatches(
  batch: StressBatch | null,
  expected: BatchIdentity,
): batch is StressBatch {
  if (!batch || batch.test_scope !== expected.scope || batch.selected_mode !== expected.mode || batch.dataset_id !== expected.datasetId || batch.algorithm_cluster_id !== expected.algorithmClusterId) {
    return false;
  }
  if (!expected.caseId) {
    return true;
  }
  return batch.execution_scope === "single_case" && batch.selected_case_id === expected.caseId;
}

function batchIsSingle(batch: StressBatch | null): batch is StressBatch & { execution_scope: "single_case"; selected_case_id: string } {
  return batch?.execution_scope === "single_case" && typeof batch.selected_case_id === "string";
}

function batchScopeProgress(batch: StressBatch): string {
  if (!batchIsSingle(batch)) {
    return `${batch.progress.completed_cases}/${batch.progress.total_cases}`;
  }
  return `单组测试 ${batch.progress.completed_cases}/${batch.progress.total_cases} · 全集${batch.dataset_case_count ?? 100}组`;
}

function newerBatch(left: StressBatch, right: StressBatch): boolean {
  const leftTime = Date.parse(left.created_at);
  const rightTime = Date.parse(right.created_at);
  return leftTime > rightTime || (leftTime === rightTime && left.batch_id > right.batch_id);
}

function batchHistoryLabel(batch: StressBatch, scope: StressScope): string {
  const timestamp = DATE_FORMAT.format(new Date(batch.created_at));
  if (batchIsSingle(batch)) {
    const caseRecord = batch.cases[0];
    const caseLabel = caseRecord
      ? caseTitle(scope, previewCaseFromBatchCase(caseRecord))
      : batch.selected_case_id;
    return `${timestamp} · 单组 · ${caseLabel} · ${batch.selected_case_id} · ${SCOPED_BATCH_LABELS[batch.status]} · ${batch.batch_id.slice(0, 8)}`;
  }
  return `${timestamp} · 整批 100组 · ${SCOPED_BATCH_LABELS[batch.status]} · ${batch.batch_id.slice(0, 8)}`;
}

function inputAudioUrl(datasetId: string, previewCase: StressPreviewCase, role: (typeof SCOPED_INPUT_SIGNAL_ROLES)[number]): string {
  return `${STRESS_PREVIEW_URL}/${encodeURIComponent(datasetId)}/cases/${encodeURIComponent(previewCase.case_id)}/inputs/${encodeURIComponent(role)}?v=${encodeURIComponent(previewCase.signals_sha256[role])}`;
}

function previewCaseFromBatchCase(caseRecord: StressBatchCase): StressPreviewCase {
  return {
    case_id: caseRecord.case_id,
    conditions: caseRecord.conditions,
    families: [],
    contract_item_ids: caseRecord.contract_item_ids,
    provenance: caseRecord.provenance,
    signals: [...SCOPED_INPUT_SIGNAL_ROLES],
    signals_sha256: caseRecord.signals_sha256,
  };
}

function CategoryFilterBar({ scope, counts, value, total, onChange, testId }: { scope: StressScope; counts: Record<string, number>; value: CategoryFilter; total: number; onChange: (value: CategoryFilter) => void; testId: string }) {
  return (
    <div className="ast-filter" role="group" aria-label="按工作负载类别筛选" data-testid={testId}>
      <button type="button" className={value === "all" ? "is-active" : ""} aria-pressed={value === "all"} data-workload-category="all" onClick={() => onChange("all")}>
        <span>全部</span>
        <strong>{total}</strong>
      </button>
      {STRESS_CATEGORIES[scope].map((category) => (
        <button key={category} type="button" className={value === category ? "is-active" : ""} aria-pressed={value === category} data-workload-category={category} onClick={() => onChange(category)}>
          <span>{categoryLabel(scope, category)}</span>
          <strong>{counts[category] ?? 0}</strong>
        </button>
      ))}
    </div>
  );
}

function StressTrack({
  role,
  caseId,
  visual,
  peakScale,
  melCeilingDb,
  showMel,
  audioUrl,
  playing,
  onPlay,
  outputPlaying,
  onToggleOutputPlayback,
  runtimeDynamic = false,
}: {
  role: (typeof SCOPED_INPUT_SIGNAL_ROLES)[number] | "rx" | "tx";
  caseId: string;
  visual: ScopedPreviewVisualTrack;
  peakScale: number;
  melCeilingDb: number;
  showMel: boolean;
  audioUrl?: string;
  playing?: boolean;
  onPlay?: () => void;
  outputPlaying?: boolean;
  onToggleOutputPlayback?: () => void;
  runtimeDynamic?: boolean;
}) {
  const isOutput = role === "rx" || role === "tx" || runtimeDynamic;
  const label = role === "tx"
    ? "测试后音频 · TX"
    : role === "rx"
      ? "回声参考 · VCP RX 后"
      : role === "microphone" && runtimeDynamic
        ? "麦克风混合输入 · 运行时合成（RX 后）"
      : role === "far_end"
        ? "回声参考 · VCP RX 前 · far_end"
      : ACOUSTIC_SIGNAL_LABELS[role];
  return (
    <article
      className={`ast-track is-${role}${visual.peak === 0 ? " is-silent" : ""}${runtimeDynamic ? " is-runtime-ready" : ""}`}
      data-signal-role={role}
      data-runtime-microphone={runtimeDynamic ? "ready" : undefined}
    >
      <header>
        <div>
          <span>{runtimeDynamic ? "运行时" : isOutput ? "测试后" : "测试前"}</span>
          <strong>{label}</strong>
        </div>
        <div className="ast-track-meta">
          {runtimeDynamic ? <span className="ast-runtime-badge is-ready">运行时已合成</span> : null}
          {visual.peak === 0 ? <span className="ast-silence">数字静音</span> : null}
          <span>{formatDuration(visual.duration_s)}</span>
          <span>{formatRate(visual.sample_rate)}</span>
          <span>{visual.channels} 通道</span>
        </div>
      </header>
      <p>
        {role === "tx"
          ? "VCP 发送处理输出。"
          : role === "rx"
            ? "本次测试现场生成的 VCP RX 处理后音频，作为 TX 的 --echo 输入。"
            : TRACK_DESCRIPTIONS[role]}
      </p>
      <PrecomputedAudioFigures visual={visual} label={label} caseId={caseId} peakScale={peakScale} melCeilingDb={melCeilingDb} showMel={showMel} />
      {isOutput && audioUrl && onToggleOutputPlayback ? (
        <footer>
          <OutputAudioPreview
            caseId={caseId}
            label={
              runtimeDynamic
                ? "运行时合成麦克风音频"
                : role === "rx"
                  ? "RX 后回声参考音频"
                  : "TX 输出音频"
            }
            url={audioUrl}
            playing={Boolean(outputPlaying)}
            onToggle={onToggleOutputPlayback}
          />
        </footer>
      ) : audioUrl ? (
        <footer>
          {playing ? (
            <audio controls autoPlay preload="none" src={audioUrl} aria-label={`试听 ${caseId} ${label}`}>
              当前浏览器不支持音频播放。
            </audio>
          ) : (
            <button type="button" onClick={onPlay} aria-label={`试听 ${caseId} ${label}`}>
              <Play size={12} fill="currentColor" aria-hidden="true" />
              按需试听
            </button>
          )}
        </footer>
      ) : null}
    </article>
  );
}

function StressCaseFigures({
  scope,
  previewCase,
  visuals,
  datasetId,
  dynamicMicVisual,
  dynamicMicAudioUrl,
  dynamicMicReason,
  dynamicMicPlaying,
  onToggleDynamicMicPlayback,
  rxVisual,
  rxAudioUrl,
  txVisual,
  txAudioUrl,
  showMel,
  playingTrack,
  onPlay,
  rxPlaying,
  onToggleRxPlayback,
  outputPlaying,
  onToggleOutputPlayback,
}: {
  scope: StressScope;
  previewCase: StressPreviewCase;
  visuals: Record<(typeof SCOPED_INPUT_SIGNAL_ROLES)[number], ScopedPreviewVisualTrack>;
  datasetId: string;
  dynamicMicVisual?: ScopedPreviewVisualTrack;
  dynamicMicAudioUrl?: string;
  dynamicMicReason?: string;
  dynamicMicPlaying?: boolean;
  onToggleDynamicMicPlayback?: () => void;
  rxVisual?: ScopedPreviewVisualTrack;
  rxAudioUrl?: string;
  txVisual?: ScopedPreviewVisualTrack;
  txAudioUrl?: string;
  showMel: boolean;
  playingTrack?: PlayingTrack;
  onPlay?: (track: PlayingTrack) => void;
  rxPlaying?: boolean;
  onToggleRxPlayback?: () => void;
  outputPlaying?: boolean;
  onToggleOutputPlayback?: () => void;
}) {
  const staticInputVisuals = SCOPED_INPUT_SIGNAL_ROLES
    .filter((role) => role !== "microphone")
    .map((role) => visuals[role]);
  const allVisuals = [
    ...staticInputVisuals,
    ...(dynamicMicVisual ? [dynamicMicVisual] : []),
    ...(rxVisual ? [rxVisual] : []),
    ...(txVisual ? [txVisual] : []),
  ];
  const peakScale = Math.max(0, ...allVisuals.map((visual) => visual.peak));
  const melCeilingDb = Math.max(-120, ...allVisuals.map((visual) => visual.mel_ceiling_db));
  return (
    <div className={`ast-track-grid${showMel ? " has-mel" : ""}`}>
      {SCOPED_INPUT_SIGNAL_ROLES.map((role) => {
        if (role === "microphone") {
          if (dynamicMicVisual) {
            return (
              <StressTrack
                key={role}
                role="microphone"
                runtimeDynamic
                caseId={previewCase.case_id}
                visual={dynamicMicVisual}
                peakScale={peakScale}
                melCeilingDb={melCeilingDb}
                showMel={showMel}
                audioUrl={dynamicMicAudioUrl}
                outputPlaying={dynamicMicPlaying}
                onToggleOutputPlayback={onToggleDynamicMicPlayback}
              />
            );
          }
          return (
            <article
              key={role}
              className="ast-track is-microphone is-unavailable"
              data-signal-role="microphone"
              data-runtime-microphone="pending"
            >
              <header>
                <div>
                  <span>运行时合成</span>
                  <strong>麦克风混合输入 · 运行时合成（RX 后）</strong>
                </div>
                <div className="ast-track-meta">
                  <span className="ast-runtime-badge is-pending">运行时合成 · 待执行</span>
                </div>
              </header>
              <p>
                {dynamicMicReason
                  || "尚未执行；点击「运行本组」或「运行 100 组」后，先经 VCP RX 生成回声参考，再实时合成麦克风波形与音频。"}
              </p>
              <div className="ast-runtime-placeholder" role="status">
                <span>执行后实时生成麦克风波形{showMel ? "与梅尔频谱" : ""}（不展示数据集静态预录麦克风）</span>
              </div>
            </article>
          );
        }
        return (
          <StressTrack
            key={role}
            role={role}
            caseId={previewCase.case_id}
            visual={visuals[role]}
            peakScale={peakScale}
            melCeilingDb={melCeilingDb}
            showMel={showMel}
            audioUrl={onPlay ? inputAudioUrl(datasetId, previewCase, role) : undefined}
            playing={playingTrack?.caseId === previewCase.case_id && playingTrack.role === role}
            onPlay={
              onPlay
                ? () =>
                    onPlay({
                      caseId: previewCase.case_id,
                      role,
                    })
                : undefined
            }
          />
        );
      })}
      {rxVisual ? (
        <StressTrack
          role="rx"
          caseId={previewCase.case_id}
          visual={rxVisual}
          peakScale={peakScale}
          melCeilingDb={melCeilingDb}
          showMel={showMel}
          audioUrl={onPlay ? rxAudioUrl : undefined}
          playing={
            playingTrack?.caseId === previewCase.case_id
            && playingTrack.role === "rx"
          }
          onPlay={
            onPlay
              ? () =>
                  onPlay({
                    caseId: previewCase.case_id,
                    role: "rx",
                  })
              : undefined
          }
          outputPlaying={rxPlaying}
          onToggleOutputPlayback={onToggleRxPlayback}
        />
      ) : null}
      {txVisual ? (
        <StressTrack
          role="tx"
          caseId={previewCase.case_id}
          visual={txVisual}
          peakScale={peakScale}
          melCeilingDb={melCeilingDb}
          showMel={showMel}
          audioUrl={onPlay ? txAudioUrl : undefined}
          playing={playingTrack?.caseId === previewCase.case_id && playingTrack.role === "tx"}
          onPlay={onPlay ? () => onPlay({ caseId: previewCase.case_id, role: "tx" }) : undefined}
          outputPlaying={outputPlaying}
          onToggleOutputPlayback={onToggleOutputPlayback}
        />
      ) : showMel ? (
        <article className="ast-track is-tx is-unavailable">
          <header>
            <div>
              <span>测试后</span>
              <strong>测试后音频 · TX</strong>
            </div>
          </header>
          <p>尚未执行或当前批次尚未完成该工程组合。</p>
        </article>
      ) : null}
      <span className="sr-only">{scope === "echo" ? "回声工作负载四路输入" : "双讲工作负载四路输入"}</span>
    </div>
  );
}

function SingleCaseRunButton({
  caseId,
  disabled,
  disabledReason,
  submitting,
  onRun,
}: {
  caseId: string;
  disabled: boolean;
  disabledReason: string;
  submitting: boolean;
  onRun: (caseId: string) => void;
}) {
  return (
    <button
      className="ast-run-single"
      type="button"
      data-testid="run-single-case"
      data-case-id={caseId}
      aria-label={disabled ? `运行本组 ${caseId}，不可用：${disabledReason || "当前不可运行"}` : `运行本组 ${caseId}`}
      disabled={disabled}
      title={disabled ? disabledReason : `运行本组 ${caseId}`}
      onClick={() => onRun(caseId)}
    >
      <Play size={12} fill="currentColor" aria-hidden="true" />
      {submitting ? "正在提交…" : "运行本组"}
    </button>
  );
}

function SingleCaseStatus({
  latest,
  submitting,
  error,
  onViewResult,
  onViewWaveform,
  outputPlaying,
  onToggleOutputPlayback,
}: {
  latest: LatestCaseResult;
  submitting: boolean;
  error: string;
  onViewResult: (caseId: string) => void;
  onViewWaveform: (caseId: string) => void;
  outputPlaying?: boolean;
  onToggleOutputPlayback?: () => void;
}) {
  const { batch, caseRecord, item } = latest;
  if (!batch && !submitting && !error) return null;
  const tx = caseRecord && batch
    ? stressTxEvidence(
        previewCaseFromBatchCase(caseRecord),
        batch,
        batch.dataset_manifest_sha256,
      )
    : null;

  const caseStatus = submitting ? "running" : caseRecord?.status;
  const statusLabel = submitting ? "正在提交" : caseStatus ? CASE_STATUS_LABELS[caseStatus] : "尚未执行";
  const active = submitting || caseStatus === "pending" || caseStatus === "running";
  const contractValue = item?.value != null
    ? `${scopedValue(item.value)} ${scopedMetricUnit(item.metric)}`
    : item
      ? "无数值"
      : "—";
  const contractStatus = item ? SCOPED_STATUS_LABELS[item.status] : "尚无合同观测";
  const visibleError = error || caseRecord?.error || "";
  const canViewResult = Boolean(caseRecord && (caseRecord.status === "completed" || caseRecord.status === "failed"));
  const canViewWaveform = caseRecord?.status === "completed";

  return (
    <section className={`ast-single-status is-${caseStatus ?? "idle"}`} aria-busy={active} aria-live={active ? "polite" : undefined}>
      <div className="ast-single-status-heading">
        <span>单组执行</span>
        <strong>{statusLabel}</strong>
        {batch ? <small>{batchIsSingle(batch) ? "单组批次" : "整批批次"} · {batch.batch_id.slice(0, 8)}</small> : null}
      </div>
      <dl>
        <div>
          <dt>合同观测</dt>
          <dd>{item?.item_id ?? caseRecord?.contract_item_ids[0] ?? "—"}</dd>
        </div>
        <div>
          <dt>合同状态</dt>
          <dd>{contractStatus}</dd>
        </div>
        <div>
          <dt>合同值</dt>
          <dd>{contractValue}</dd>
        </div>
      </dl>
      {visibleError ? (
        <p className="ast-single-status-error" role="alert">
          <AlertCircle size={13} aria-hidden="true" />
          {visibleError}
        </p>
      ) : null}
      {tx?.ready && onToggleOutputPlayback ? (
        <div className="ast-single-output-preview">
          <OutputAudioPreview
            caseId={caseRecord!.case_id}
            label="TX 输出音频"
            url={tx.url}
            playing={Boolean(outputPlaying)}
            onToggle={onToggleOutputPlayback}
          />
        </div>
      ) : null}
      {canViewResult || canViewWaveform ? (
        <div className="ast-single-status-actions">
          {canViewResult ? (
            <button type="button" onClick={() => onViewResult(caseRecord!.case_id)}>
              查看本组结果
            </button>
          ) : null}
          {canViewWaveform ? (
            <button type="button" onClick={() => onViewWaveform(caseRecord!.case_id)}>
              查看本组波形 / Mel
            </button>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}

function CaseHeader({ scope, previewCase, action, ordinal }: { scope: StressScope; previewCase: StressPreviewCase; action?: ReactNode; ordinal?: number }) {
  const resamplingNotice = swbResamplingNotice(previewCase, previewCase.conditions.mode);
  return (
    <header className="ast-case-header">
      <div>
        <div className="ast-case-eyebrow">
          {ordinal !== undefined ? (
            <span className="ast-case-ordinal" aria-label={`第 ${String(ordinal).padStart(2, "0")} 组`}>
              <span aria-hidden="true">{String(ordinal).padStart(2, "0")}</span>
            </span>
          ) : null}
          <span>{categoryLabel(scope, previewCase.provenance.workload_category)}</span>
        </div>
        <h4>{caseTitle(scope, previewCase)}</h4>
        <code>{previewCase.case_id}</code>
      </div>
      <div className="ast-case-header-tools">
        <dl>
          <div>
            <dt>音量</dt>
            <dd>{volumeLabel(previewCase.conditions.volume)}</dd>
          </div>
          <div>
            <dt>合同</dt>
            <dd>{previewCase.contract_item_ids[0]}</dd>
          </div>
          {scope === "double_talk" ? (
            <div>
              <dt>设计重叠窗口</dt>
              <dd>{formatDuration(designedOverlapSeconds(previewCase))}</dd>
            </div>
          ) : (
            <div>
              <dt>参考边界</dt>
              <dd>干净近端参考数字静音</dd>
            </div>
          )}
        </dl>
        {action ? <div className="ast-case-run-control">{action}</div> : null}
      </div>
      <p>{activitySummary(previewCase)}</p>
      {resamplingNotice ? <p className="ast-resampling-note is-case">{resamplingNotice}</p> : null}
    </header>
  );
}

const ExecuteCase = memo(function ExecuteCase({
  scope,
  previewCase,
  ordinal,
  visuals,
  datasetId,
  playingTrack,
  onPlay,
  latest,
  singleDisabled,
  singleDisabledReason,
  singleSubmitting,
  singleError,
  onRunSingle,
  onViewResult,
  onViewWaveform,
  outputPlaybackKey,
  onToggleOutputPlayback,
}: {
  scope: StressScope;
  previewCase: StressPreviewCase;
  ordinal?: number;
  visuals: Record<(typeof SCOPED_INPUT_SIGNAL_ROLES)[number], ScopedPreviewVisualTrack>;
  datasetId: string;
  playingTrack: PlayingTrack;
  onPlay: (track: PlayingTrack) => void;
  latest: LatestCaseResult;
  singleDisabled: boolean;
  singleDisabledReason: string;
  singleSubmitting: boolean;
  singleError: string;
  onRunSingle: (caseId: string) => void;
  onViewResult: (caseId: string) => void;
  onViewWaveform: (caseId: string) => void;
  outputPlaybackKey: string;
  onToggleOutputPlayback: (key: string) => void;
}) {
  const rx = latest.batch && latest.caseRecord
    ? stressRxEvidence(
        previewCaseFromBatchCase(latest.caseRecord),
        latest.batch,
        latest.batch.dataset_manifest_sha256,
      )
    : null;
  const dynamicMic = latest.batch && latest.caseRecord
    ? stressDynamicMicrophoneEvidence(
        previewCaseFromBatchCase(latest.caseRecord),
        latest.batch,
        latest.batch.dataset_manifest_sha256,
      )
    : null;
  const playbackBase = latest.batch && latest.caseRecord
    ? `${latest.batch.batch_id}:${latest.caseRecord.case_id}`
    : "";
  return (
    <article className="ast-case" data-case-id={previewCase.case_id} data-workload-category={previewCase.provenance.workload_category}>
      <CaseHeader
        scope={scope}
        previewCase={previewCase}
        ordinal={ordinal}
        action={
          <SingleCaseRunButton
            caseId={previewCase.case_id}
            disabled={singleDisabled}
            disabledReason={singleDisabledReason}
            submitting={singleSubmitting}
            onRun={onRunSingle}
          />
        }
      />
      <SingleCaseStatus latest={latest} submitting={singleSubmitting} error={singleError} onViewResult={onViewResult} onViewWaveform={onViewWaveform} outputPlaying={Boolean(playbackBase) && outputPlaybackKey === `${playbackBase}:tx`} onToggleOutputPlayback={() => onToggleOutputPlayback(`${playbackBase}:tx`)} />
      <StressCaseFigures
        scope={scope}
        previewCase={previewCase}
        visuals={visuals}
        datasetId={datasetId}
        dynamicMicVisual={dynamicMic?.ready ? dynamicMic.visual : undefined}
        dynamicMicAudioUrl={dynamicMic?.ready ? dynamicMic.url : undefined}
        dynamicMicReason={dynamicMic && !dynamicMic.ready ? dynamicMic.reason : undefined}
        dynamicMicPlaying={Boolean(playbackBase) && outputPlaybackKey === `${playbackBase}:dynamic-mic`}
        onToggleDynamicMicPlayback={() => onToggleOutputPlayback(`${playbackBase}:dynamic-mic`)}
        rxVisual={rx?.ready ? rx.visual : undefined}
        rxAudioUrl={rx?.ready ? rx.url : undefined}
        rxPlaying={Boolean(playbackBase) && outputPlaybackKey === `${playbackBase}:rx`}
        onToggleRxPlayback={() => onToggleOutputPlayback(`${playbackBase}:rx`)}
        showMel={false}
        playingTrack={playingTrack}
        onPlay={onPlay}
        outputPlaying={Boolean(playbackBase) && outputPlaybackKey === `${playbackBase}:tx`}
        onToggleOutputPlayback={() => onToggleOutputPlayback(`${playbackBase}:tx`)}
      />
    </article>
  );
});

function useNearViewport(lazy: boolean) {
  const ref = useRef<HTMLDivElement>(null);
  const [nearViewport, setNearViewport] = useState(!lazy);
  useEffect(() => {
    if (!lazy || nearViewport) {
      setNearViewport(true);
      return;
    }
    const node = ref.current;
    if (!node || typeof IntersectionObserver === "undefined") {
      setNearViewport(true);
      return;
    }
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          setNearViewport(true);
          observer.disconnect();
        }
      },
      { rootMargin: "1000px 0px" },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [lazy, nearViewport]);
  return { ref, nearViewport };
}

const WaveformCase = memo(function WaveformCase({
  scope,
  previewCase,
  ordinal,
  preview,
  batch,
  playingTrack,
  onPlay,
  latest,
  singleDisabled,
  singleDisabledReason,
  singleSubmitting,
  singleError,
  onRunSingle,
  onViewResult,
  onViewWaveform,
  outputPlaybackKey,
  onToggleOutputPlayback,
}: {
  scope: StressScope;
  previewCase: StressPreviewCase;
  ordinal?: number;
  preview: StressPreview;
  batch: StressBatch | null;
  playingTrack: PlayingTrack;
  onPlay: (track: PlayingTrack) => void;
  latest: LatestCaseResult;
  singleDisabled: boolean;
  singleDisabledReason: string;
  singleSubmitting: boolean;
  singleError: string;
  onRunSingle: (caseId: string) => void;
  onViewResult: (caseId: string) => void;
  onViewWaveform: (caseId: string) => void;
  outputPlaybackKey: string;
  onToggleOutputPlayback: (key: string) => void;
}) {
  const { ref, nearViewport } = useNearViewport(true);
  const tx = stressTxEvidence(previewCase, batch, preview.dataset_manifest_sha256);
  const rx = stressRxEvidence(
    previewCase,
    batch,
    preview.dataset_manifest_sha256,
  );
  const dynamicMic = stressDynamicMicrophoneEvidence(
    previewCase,
    batch,
    preview.dataset_manifest_sha256,
  );
  const playbackBase = batch ? `${batch.batch_id}:${previewCase.case_id}` : "";
  return (
    <div ref={ref} className="ast-waveform-case-slot" data-case-id={previewCase.case_id} data-workload-category={previewCase.provenance.workload_category} data-lazy-state={nearViewport ? "mounted" : "deferred"}>
      {nearViewport ? (
        <article className="ast-case is-waveform">
          <CaseHeader
            scope={scope}
            previewCase={previewCase}
            ordinal={ordinal}
            action={
              <SingleCaseRunButton
                caseId={previewCase.case_id}
                disabled={singleDisabled}
                disabledReason={singleDisabledReason}
                submitting={singleSubmitting}
                onRun={onRunSingle}
              />
            }
          />
          <SingleCaseStatus latest={latest} submitting={singleSubmitting} error={singleError} onViewResult={onViewResult} onViewWaveform={onViewWaveform} outputPlaying={Boolean(playbackBase) && outputPlaybackKey === `${playbackBase}:tx`} onToggleOutputPlayback={() => onToggleOutputPlayback(`${playbackBase}:tx`)} />
          <StressCaseFigures
            scope={scope}
            previewCase={previewCase}
            visuals={preview.visuals.cases[previewCase.case_id]}
            datasetId={preview.dataset_id}
            dynamicMicVisual={dynamicMic.ready ? dynamicMic.visual : undefined}
            dynamicMicAudioUrl={dynamicMic.ready ? dynamicMic.url : undefined}
            dynamicMicReason={!dynamicMic.ready ? dynamicMic.reason : undefined}
            dynamicMicPlaying={Boolean(playbackBase) && outputPlaybackKey === `${playbackBase}:dynamic-mic`}
            onToggleDynamicMicPlayback={() => onToggleOutputPlayback(`${playbackBase}:dynamic-mic`)}
            rxVisual={rx.ready ? rx.visual : undefined}
            rxAudioUrl={rx.ready ? rx.url : undefined}
            rxPlaying={Boolean(playbackBase) && outputPlaybackKey === `${playbackBase}:rx`}
            onToggleRxPlayback={() => onToggleOutputPlayback(`${playbackBase}:rx`)}
            txVisual={tx.ready ? tx.visual : undefined}
            txAudioUrl={tx.ready ? tx.url : undefined}
            showMel
            playingTrack={playingTrack}
            onPlay={onPlay}
            outputPlaying={Boolean(playbackBase) && outputPlaybackKey === `${playbackBase}:tx`}
            onToggleOutputPlayback={() => onToggleOutputPlayback(`${playbackBase}:tx`)}
          />
          {!tx.ready ? <p className="ast-inline-note">{tx.reason}</p> : null}
        </article>
      ) : (
        <article className="ast-case ast-case-deferred">
          <CaseHeader
            scope={scope}
            previewCase={previewCase}
            ordinal={ordinal}
            action={
              <SingleCaseRunButton
                caseId={previewCase.case_id}
                disabled={singleDisabled}
                disabledReason={singleDisabledReason}
                submitting={singleSubmitting}
                onRun={onRunSingle}
              />
            }
          />
          <p>滚动到此处后计算四路波形与梅尔频谱。</p>
        </article>
      )}
    </div>
  );
});

function StatCard({ label, value, detail }: { label: string; value: ReactNode; detail?: string }) {
  return (
    <div className="ast-stat">
      <dt>{label}</dt>
      <dd>{value}</dd>
      {detail ? <small>{detail}</small> : null}
    </div>
  );
}

export default function AcousticStressTest({ scope, definition, algorithmClusterId, mode, onModeChange, onAlgorithmClusterChange, externalRunning = false, onBusyChange, pageVisible = true }: Props) {
  const titleId = useId().replace(/[^A-Za-z0-9_-]/g, "");
  const [activeTab, setActiveTab] = useState<SceneSubpageId>(() => sceneSubpageFromHash(window.location.hash, definition.route));
  const [datasetState, setDatasetState] = useState<AsyncState<StressDatasetView | null>>({ value: null, loading: true, error: "" });
  const [previewState, setPreviewState] = useState<AsyncState<StressPreview | null>>({ value: null, loading: false, error: "" });
  const [historyState, setHistoryState] = useState<AsyncState<StressBatch[]>>({ value: [], loading: false, error: "" });
  const [snapshot, setSnapshot] = useState<StressBatch | null>(null);
  const [launching, setLaunching] = useState(false);
  const [singleLaunchingCaseId, setSingleLaunchingCaseId] = useState<string | null>(null);
  const [singleCaseError, setSingleCaseError] = useState<SingleCaseError>(null);
  const [outputPlaybackKey, setOutputPlaybackKey] = useState("");
  const toggleOutputPlayback = useCallback((key: string) => {
    setPlayingTrack(null);
    setOutputPlaybackKey((current) => (current === key ? "" : key));
  }, []);
  const [runError, setRunError] = useState("");
  const [pollError, setPollError] = useState("");
  const [datasetRefresh, setDatasetRefresh] = useState(0);
  const [previewRefresh, setPreviewRefresh] = useState(0);
  const [filter, setFilter] = useState<CategoryFilter>("all");
  const [page, setPage] = useState(1);
  const [playingTrack, setPlayingTrack] = useState<PlayingTrack>(null);
  const [focusedResultCaseId, setFocusedResultCaseId] = useState("");
  const [focusedWaveformCaseId, setFocusedWaveformCaseId] = useState("");
  const launchAbortRef = useRef<AbortController | null>(null);
  const singleLaunchingRef = useRef(false);
  const historyAbortRef = useRef<AbortController | null>(null);

  const validatedDatasetView = datasetState.value?.test_scope === scope && datasetState.value.mode === mode ? datasetState.value : null;
  const datasetCandidate = validatedDatasetView?.datasets[0] ?? null;
  const dataset = datasetCandidate?.test_scope === scope && datasetCandidate.mode === mode && datasetCandidate.scene === "baseline" ? datasetCandidate : null;
  const previewCandidate = previewState.value;
  const preview = dataset && previewCandidate?.dataset_id === dataset.dataset_id && previewCandidate.test_scope === scope && previewCandidate.mode === mode && previewCandidate.scene === "baseline" ? previewCandidate : null;
  const contracts = preview?.contracts ?? validatedDatasetView?.contracts ?? [];
  const datasetId = dataset?.dataset_id ?? "";
  const currentBatch = batchMatches(snapshot, {
    scope,
    mode,
    datasetId,
    algorithmClusterId,
  })
    ? snapshot
    : null;
  const identity = useMemo(
    () => ({
      scope,
      mode,
      datasetId,
      algorithmClusterId,
    }),
    [algorithmClusterId, datasetId, mode, scope],
  );
  const historyWithSnapshot = useMemo(() => {
    const byId = new Map<string, StressBatch>();
    for (const batch of [...historyState.value, ...(snapshot ? [snapshot] : [])]) {
      if (batchMatches(batch, identity)) {
        byId.set(batch.batch_id, batch);
      }
    }
    return [...byId.values()].sort((left, right) => {
      if (newerBatch(left, right)) return -1;
      if (newerBatch(right, left)) return 1;
      return left.batch_id.localeCompare(right.batch_id);
    });
  }, [historyState.value, identity, snapshot]);
  const latestCaseResultByCase = useMemo(() => {
    const latestByCase = new Map<string, LatestCaseResult>();
    for (const batch of historyWithSnapshot) {
      for (const caseRecord of batch.cases) {
        const current = latestByCase.get(caseRecord.case_id);
        if (!current || newerBatch(batch, current.batch!)) {
          latestByCase.set(caseRecord.case_id, {
            batch,
            caseRecord,
            item: batch.items.find((entry) => entry.case_id === caseRecord.case_id),
          });
        }
      }
    }
    return latestByCase;
  }, [historyWithSnapshot]);
  const activeBatches = historyWithSnapshot.filter((batch) => batch.status === "queued" || batch.status === "running");
  const pollingBatch = activeBatches[0] ?? null;
  const isBatchActive = activeBatches.length > 0;
  const busy = launching || singleLaunchingCaseId !== null || isBatchActive;
  const controlsLocked = externalRunning || busy;
  const modeRateHz = sampleRateForBusinessMode(mode);
  const availableAlgorithmGroups = algorithmClusterGroupsForMode(mode, { vcpOnly: true });
  const availableAlgorithmCount = algorithmClustersForMode(mode, { vcpOnly: true }).length;
  const algorithmSupported = isAlgorithmClusterSupportedForMode(algorithmClusterId, mode, { vcpOnly: true });
  const selectedAlgorithmCluster = algorithmClusterById(algorithmClusterId);
  const singleDisabled = controlsLocked || !algorithmSupported || datasetState.loading || previewState.loading;
  const singleDisabledReason = externalRunning
    ? "其他测试正在运行，当前选择已锁定"
    : !algorithmSupported
      ? `当前算法簇不支持 ${modeRateHz / 1000} kHz（${mode}），不能启动单组测试`
      : datasetState.loading
        ? "正在刷新回声 / 双讲测试集"
        : previewState.loading
          ? "正在读取 100 组工程输入"
          : launching
            ? "整批测试正在提交"
            : singleLaunchingCaseId
              ? "已有单组测试正在提交"
              : isBatchActive
                ? "已有批次执行中，不能重复提交"
                : "";
  const storage = storageKey(scope, mode, datasetId, algorithmClusterId);

  useEffect(() => {
    const syncTab = () => {
      if (sceneSubpageRoute(window.location.hash) !== definition.route) {
        return;
      }
      setActiveTab(sceneSubpageFromHash(window.location.hash, definition.route));
    };
    syncTab();
    window.addEventListener("hashchange", syncTab);
    window.addEventListener("popstate", syncTab);
    return () => {
      window.removeEventListener("hashchange", syncTab);
      window.removeEventListener("popstate", syncTab);
    };
  }, [definition.route]);

  useEffect(() => {
    launchAbortRef.current?.abort();
    singleLaunchingRef.current = false;
    setSingleLaunchingCaseId(null);
    setSingleCaseError(null);
    setFilter("all");
    setPage(1);
    setPlayingTrack(null);
    setFocusedResultCaseId("");
    setFocusedWaveformCaseId("");
  }, [algorithmClusterId, datasetId, mode, scope]);

  useEffect(() => {
    if (!pageVisible) return;
    const controller = new AbortController();
    let disposed = false;
    setDatasetState({ value: null, loading: true, error: "" });
    const params = new URLSearchParams({ scope, mode });
    void requestJson(`${STRESS_DATASETS_URL}?${params.toString()}`, { signal: controller.signal, cache: "no-store" }, "读取回声/双讲测试集")
      .then((body) => readStressDatasets(body, { scope, mode }))
      .then((value) => {
        if (disposed || controller.signal.aborted) return;
        setDatasetState({ value, loading: false, error: "" });
      })
      .catch((reason: unknown) => {
        if (disposed || controller.signal.aborted) return;
        setDatasetState({
          value: null,
          loading: false,
          error: errorMessage(reason, "回声/双讲测试集读取失败"),
        });
      });
    return () => {
      disposed = true;
      controller.abort();
    };
  }, [datasetRefresh, mode, pageVisible, scope]);

  useEffect(() => {
    if (!pageVisible || !dataset || !validatedDatasetView) return;
    const controller = new AbortController();
    let disposed = false;
    setPreviewState({ value: null, loading: true, error: "" });
    void requestJson(`${STRESS_PREVIEW_URL}/${encodeURIComponent(dataset.dataset_id)}/preview`, { signal: controller.signal, cache: "no-cache" }, "读取回声/双讲预览")
      .then((body) =>
        readStressPreview(body, {
          scope,
          mode,
          datasetId: dataset.dataset_id,
          contracts: validatedDatasetView.contracts,
        }),
      )
      .then((value) => {
        if (disposed || controller.signal.aborted) return;
        setPreviewState({ value, loading: false, error: "" });
      })
      .catch((reason: unknown) => {
        if (disposed || controller.signal.aborted) return;
        setPreviewState({
          value: null,
          loading: false,
          error: errorMessage(reason, "回声/双讲预览读取失败"),
        });
      });
    return () => {
      disposed = true;
      controller.abort();
    };
  }, [dataset, mode, pageVisible, previewRefresh, scope, validatedDatasetView]);

  useEffect(() => {
    if (!pageVisible || !dataset || !algorithmClusterId || !validatedDatasetView) {
      return;
    }
    const controller = new AbortController();
    let disposed = false;
    setSnapshot(null);
    setHistoryState({ value: [], loading: true, error: "" });
    const expected = {
      scope,
      mode,
      datasetId: dataset.dataset_id,
      algorithmClusterId,
    };
    const params = new URLSearchParams({
      limit: "20",
      dataset_id: dataset.dataset_id,
      algorithm_cluster_id: algorithmClusterId,
      latest_per_case: "true",
    });
    void requestJson(`${STRESS_BATCHES_URL}?${params.toString()}`, { signal: controller.signal, cache: "no-store" }, "读取回声/双讲批次历史")
      .then((body) => {
        if (!Array.isArray(body)) throw new Error("批次历史格式无效");
        return body.map((entry) =>
          readStressBatch(entry, {
            ...expected,
            contracts: validatedDatasetView.contracts,
          }),
        );
      })
      .then(async (history) => {
        if (disposed || controller.signal.aborted) return;
        setHistoryState({ value: history, loading: false, error: "" });
        const storedId = readStoredBatchId(storage);
        let selected = history.find((batch) => batch.batch_id === storedId) ?? history[0] ?? null;
        if (!selected && storedId) {
          const body = await requestJson(`${STRESS_BATCHES_URL}/${encodeURIComponent(storedId)}`, { signal: controller.signal, cache: "no-store" }, "恢复回声/双讲批次");
          selected = readStressBatch(body, {
            ...expected,
            contracts: validatedDatasetView.contracts,
          });
        }
        if (disposed || controller.signal.aborted) return;
        setSnapshot(selected);
        if (selected) {
          writeStoredBatchId(storage, selected.batch_id);
        }
      })
      .catch((reason: unknown) => {
        if (disposed || controller.signal.aborted) return;
        setHistoryState({
          value: [],
          loading: false,
          error: errorMessage(reason, "批次历史读取失败"),
        });
      });
    return () => {
      disposed = true;
      controller.abort();
    };
  }, [algorithmClusterId, dataset, mode, pageVisible, scope, storage, validatedDatasetView]);

  useEffect(() => {
    if (!pollingBatch) return;
    let disposed = false;
    let timer: number | undefined;
    const controller = new AbortController();
    const poll = async () => {
      try {
        const body = await requestJson(`${STRESS_BATCHES_URL}/${encodeURIComponent(pollingBatch.batch_id)}`, { signal: controller.signal, cache: "no-store" }, "同步回声/双讲批次");
        const next = readStressBatch(body, {
          scope,
          mode,
          datasetId,
          algorithmClusterId,
          contracts,
          ...(pollingBatch.execution_scope === "single_case" && pollingBatch.selected_case_id
            ? { caseId: pollingBatch.selected_case_id }
            : {}),
        });
        if (disposed || controller.signal.aborted) return;
        if ((pollingBatch.execution_scope === "single_case") !== (next.execution_scope === "single_case")) {
          throw new Error("批次执行范围在轮询中发生变化");
        }
        setSnapshot((current) => !current || current.batch_id === next.batch_id ? next : current);
        setHistoryState((current) => ({
          ...current,
          value: [next, ...current.value.filter((batch) => batch.batch_id !== next.batch_id)],
        }));
        writeStoredBatchId(storage, next.batch_id);
        setPollError("");
      } catch (reason) {
        if (disposed || controller.signal.aborted) return;
        setPollError(errorMessage(reason, "批次状态同步失败"));
      } finally {
        if (!disposed && !controller.signal.aborted) {
          timer = window.setTimeout(poll, POLL_INTERVAL_MS);
        }
      }
    };
    timer = window.setTimeout(poll, POLL_INTERVAL_MS);
    return () => {
      disposed = true;
      controller.abort();
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [algorithmClusterId, contracts, datasetId, mode, pollingBatch, scope, storage]);

  useEffect(() => {
    onBusyChange?.(busy);
  }, [busy, onBusyChange]);

  useEffect(() => {
    if (!pageVisible || activeTab !== "execute") {
      setPlayingTrack(null);
    }
  }, [activeTab, pageVisible]);

  useEffect(() => {
    return () => {
      launchAbortRef.current?.abort();
      historyAbortRef.current?.abort();
    };
  }, []);

  const filteredCases = useMemo(() => {
    const cases = preview?.cases ?? [];
    return filter === "all" ? cases : cases.filter((previewCase) => previewCase.provenance.workload_category === filter);
  }, [filter, preview?.cases]);
  const pageSize = 10;
  const totalPages = Math.max(1, Math.ceil(filteredCases.length / pageSize));
  const safePage = Math.min(Math.max(1, page), totalPages);
  const pageStartIndex = (safePage - 1) * pageSize;
  const pagedExecuteCases = useMemo(
    () => filteredCases.slice(pageStartIndex, pageStartIndex + pageSize),
    [filteredCases, pageStartIndex],
  );
  const focusedWaveformCase = useMemo(
    () => (focusedWaveformCaseId ? preview?.cases.find((previewCase) => previewCase.case_id === focusedWaveformCaseId) ?? null : null),
    [focusedWaveformCaseId, preview?.cases],
  );
  const waveformCases = focusedWaveformCase ? [focusedWaveformCase] : pagedExecuteCases;
  const counts = useMemo(() => categoryCounts(preview?.cases ?? [], scope), [preview?.cases, scope]);
  const swbNotice = preview?.cases[0] ? swbResamplingNotice(preview.cases[0], mode) : "";

  const navigateTab = useCallback((tab: SceneSubpageId) => {
    setActiveTab(tab);
    setPlayingTrack(null);
    const hash = sceneSubpageHash(definition.route, tab);
    if (window.location.hash !== hash) {
      window.history.pushState(null, "", hash);
    }
  }, [definition.route]);

  function changeFilter(next: CategoryFilter) {
    setPlayingTrack(null);
    setOutputPlaybackKey("");
    setFilter(next);
    setPage(1);
    setFocusedResultCaseId("");
    setFocusedWaveformCaseId("");
  }

  const togglePlayback = useCallback((track: PlayingTrack) => {
    setOutputPlaybackKey("");
    setPlayingTrack((current) => (current?.caseId === track?.caseId && current?.role === track?.role ? null : track));
  }, []);

  async function launchBatch() {
    if (!dataset || !preview || controlsLocked || !algorithmClusterId || !algorithmSupported) {
      return;
    }
    launchAbortRef.current?.abort();
    const controller = new AbortController();
    launchAbortRef.current = controller;
    setLaunching(true);
    setRunError("");
    setPollError("");
    try {
      const body = await requestJson(
        STRESS_BATCHES_URL,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: controller.signal,
          body: JSON.stringify({
            dataset_id: dataset.dataset_id,
            algorithm_cluster_id: algorithmClusterId,
            mode,
            scene: "baseline",
            test_scope: scope,
          }),
        },
        "创建回声/双讲测试批次",
      );
      const next = readStressBatch(body, {
        scope,
        mode,
        datasetId: dataset.dataset_id,
        algorithmClusterId,
        contracts: preview.contracts,
      });
      if (batchIsSingle(next) || !batchMatches(next, {
          scope,
          mode,
          datasetId: dataset.dataset_id,
          algorithmClusterId,
        })) {
        throw new Error("新批次身份与当前页面选择不一致");
      }
      setSnapshot(next);
      setHistoryState((current) => ({
        value: [next, ...current.value.filter((batch) => batch.batch_id !== next.batch_id)],
        loading: false,
        error: "",
      }));
      writeStoredBatchId(storage, next.batch_id);
      navigateTab("results");
    } catch (reason) {
      if (!controller.signal.aborted) {
        setRunError(errorMessage(reason, "回声/双讲测试批次创建失败"));
      }
    } finally {
      if (!controller.signal.aborted) setLaunching(false);
    }
  }

  const launchSingleCase = useCallback(async (caseId: string) => {
    const previewCase = preview?.cases.find((candidate) => candidate.case_id === caseId);
    if (!previewCase || !dataset || !preview || singleDisabled || !algorithmClusterId || !algorithmSupported || singleLaunchingRef.current) {
      return;
    }
    launchAbortRef.current?.abort();
    const controller = new AbortController();
    launchAbortRef.current = controller;
    singleLaunchingRef.current = true;
    setSingleLaunchingCaseId(caseId);
    setSingleCaseError(null);
    setPlayingTrack(null);
    setRunError("");
    setPollError("");
    try {
      const body = await requestJson(
        STRESS_BATCHES_URL,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: controller.signal,
          body: JSON.stringify({
            dataset_id: dataset.dataset_id,
            algorithm_cluster_id: algorithmClusterId,
            mode,
            scene: "baseline",
            test_scope: scope,
            case_id: caseId,
          }),
        },
        "创建回声/双讲单组测试批次",
      );
      const next = readStressBatch(body, {
        scope,
        mode,
        datasetId: dataset.dataset_id,
        algorithmClusterId,
        contracts: preview.contracts,
        caseId,
      });
      if (
        !batchMatches(next, {
          scope,
          mode,
          datasetId: dataset.dataset_id,
          algorithmClusterId,
          caseId,
        })
      ) {
        throw new Error("单组批次身份或 case_id 与当前工程组合不一致");
      }
      setSnapshot(next);
      setHistoryState((current) => ({
        value: [next, ...current.value.filter((batch) => batch.batch_id !== next.batch_id)],
        loading: false,
        error: "",
      }));
      writeStoredBatchId(storage, next.batch_id);
    } catch (reason) {
      if (!controller.signal.aborted) {
        setSingleCaseError({
          caseId,
          message: errorMessage(reason, "回声/双讲单组测试批次创建失败"),
        });
      }
    } finally {
      if (launchAbortRef.current === controller) {
        singleLaunchingRef.current = false;
        if (!controller.signal.aborted) setSingleLaunchingCaseId(null);
      }
    }
  }, [
    algorithmClusterId,
    algorithmSupported,
    dataset,
    mode,
    preview,
    scope,
    singleDisabled,
    storage,
  ]);

  const viewCaseResult = useCallback((caseId: string) => {
    const latest = latestCaseResultByCase.get(caseId);
    if (!latest?.batch) return;
    setSnapshot(latest.batch);
    setFocusedResultCaseId(caseId);
    setPlayingTrack(null);
    writeStoredBatchId(storage, latest.batch.batch_id);
    navigateTab("results");
  }, [latestCaseResultByCase, navigateTab, storage]);

  const viewCaseWaveform = useCallback((caseId: string) => {
    if (!preview?.cases.some((previewCase) => previewCase.case_id === caseId)) return;
    setFocusedWaveformCaseId(caseId);
    navigateTab("waveforms");
  }, [navigateTab, preview?.cases]);

  async function selectHistory(batchId: string) {
    if (!batchId || !dataset || controlsLocked) return;
    historyAbortRef.current?.abort();
    const controller = new AbortController();
    historyAbortRef.current = controller;
    try {
      const body = await requestJson(`${STRESS_BATCHES_URL}/${encodeURIComponent(batchId)}`, { signal: controller.signal, cache: "no-store" }, "读取历史批次");
      const next = readStressBatch(body, {
        scope,
        mode,
        datasetId: dataset.dataset_id,
        algorithmClusterId,
        contracts,
      });
      setSnapshot(next);
      setFocusedResultCaseId("");
      setFocusedWaveformCaseId("");
      setSingleCaseError(null);
      setHistoryState((current) => ({
        ...current,
        value: [next, ...current.value.filter((batch) => batch.batch_id !== next.batch_id)],
        error: "",
      }));
      writeStoredBatchId(storage, next.batch_id);
      setPollError("");
    } catch (reason) {
      if (!controller.signal.aborted) {
        setHistoryState((current) => ({
          ...current,
          error: errorMessage(reason, "历史批次读取失败"),
        }));
      }
    }
  }

  const items = currentBatch?.items ?? [];
  const finiteItems = items.filter((item) => item.value !== null && Number.isFinite(item.value));
  const reviewItems = items.filter((item) => item.status === "review_required");
  const notMeasuredItems = items.filter((item) => item.status === "not_measured");
  const invalidItems = items.filter((item) => item.status === "invalid");
  const focusedResultCase = currentBatch?.cases.find((caseRecord) => caseRecord.case_id === focusedResultCaseId) ?? null;
  const resultCases = focusedResultCase ? [focusedResultCase] : currentBatch?.cases ?? [];
  const itemByCase = new Map(items.map((item) => [item.case_id, item]));

  return (
    <section className="acoustic-stress-test scoped-acoustic-test" aria-labelledby={`stress-title-${titleId}`} aria-busy={datasetState.loading || previewState.loading || busy}>
      <header className="basic-uplink-header scoped-acoustic-header">
        <div className="basic-uplink-heading">
          <span className="basic-uplink-eyebrow">{scope === "echo" ? "回声压力测试 · 100" : "双讲压力测试 · 100"}</span>
          <h2 id={`stress-title-${titleId}`}>
            {definition.title} · 100 工程组合{mode.startsWith("SWB-") ? " · 48 kHz 适配" : mode.startsWith("NB-") ? " · 8 kHz 窄带" : " · 16 kHz 宽带"}
          </h2>
          <p>{definition.description}</p>
          {swbNotice ? <p className="ast-resampling-note">{swbNotice}</p> : null}
        </div>
      </header>

      <nav className="scoped-acoustic-subpage-nav" aria-label={`${definition.title}子页面`}>
        {SCENE_SUBPAGES.map((tab) => (
          <button key={tab.id} type="button" className="scoped-acoustic-subpage-tab" aria-current={activeTab === tab.id ? "page" : undefined} onClick={() => navigateTab(tab.id)}>
            {SCENE_SUBPAGE_LABELS[tab.id]}
          </button>
        ))}
      </nav>

      <div className="scoped-acoustic-subpage-status">
        <span className="scoped-acoustic-subpage-status-meta">
          <strong>{modeLabel(mode)}</strong>
        </span>
        <span className="scoped-acoustic-subpage-status-meta">五类各 20 · 全部 100</span>
        <span className="scoped-acoustic-subpage-status-meta">{currentBatch ? `${SCOPED_BATCH_LABELS[currentBatch.status]} · ${batchScopeProgress(currentBatch)}` : "尚无批次"}</span>
        <span className="scoped-acoustic-subpage-status-note">工程组合允许跨组合复用源录音，不是 100 段独立录音；正式 ITU-T 合规未测。</span>
        {externalRunning ? (
          <span className="scoped-acoustic-subpage-status-issue">
            <ShieldAlert size={13} aria-hidden="true" />
            其他测试正在运行，当前选择已锁定
          </span>
        ) : null}
      </div>

      {activeTab === "execute" ? (
        <div>
          <section className="ast-panel" aria-label={`${definition.title}执行`}>
            <div className="ast-controls">
              <label>
                <span>业务模式</span>
                <select value={mode} disabled={controlsLocked} onChange={(event) => onModeChange(event.target.value as FiveSignalMode)}>
                  {FIVE_SIGNAL_MODES.map((candidate) => (
                    <option key={candidate} value={candidate}>
                      {modeLabel(candidate)}
                    </option>
                  ))}
                </select>
                <small>选项为业务参考率；当前测试输入/输出：{stressSampleRate(mode) / 1000} kHz。</small>
              </label>
              <label>
                <span>VCP 算法簇（{modeRateHz / 1000} kHz 可选 {availableAlgorithmCount} 项）</span>
                <select value={algorithmClusterId} disabled={controlsLocked} onChange={(event) => onAlgorithmClusterChange(event.target.value)}>
                  {!algorithmSupported ? (
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
                  {algorithmSupported
                    ? `${modeRateHz / 1000} kHz 可选 ${availableAlgorithmCount} 项 · ${algorithmClusterLabel(algorithmClusterId)}`
                    : `当前算法簇不支持 ${modeRateHz / 1000} kHz（${mode}），请切换为该采样率可选算法簇。`}
                </small>
              </label>
              <button className="button secondary" type="button" disabled={controlsLocked || datasetState.loading} onClick={() => setDatasetRefresh((current) => current + 1)}>
                <RefreshCw size={14} aria-hidden="true" />
                刷新数据集
              </button>
              <button className="button primary" type="button" disabled={controlsLocked || datasetState.loading || previewState.loading || !preview || !algorithmSupported} onClick={() => void launchBatch()}>
                <Play size={14} fill="currentColor" aria-hidden="true" />
                {launching ? "正在创建批次…" : isBatchActive ? "批次执行中" : "运行 100 组"}
              </button>
              <AlgorithmClusterSummary cluster={selectedAlgorithmCluster} />
            </div>
            {!algorithmSupported ? (
              <p className="ast-warning" role="status">
                <ShieldAlert size={14} aria-hidden="true" />
                当前算法簇不支持 {modeRateHz / 1000} kHz（{mode}），不能启动回声 / 双讲 100 组批次。请在上方选择该采样率对应的 VCP 算法簇。
              </p>
            ) : null}
            {datasetState.error ? (
              <p className="ast-error" role="alert">
                <AlertCircle size={14} aria-hidden="true" />
                {datasetState.error}
              </p>
            ) : null}
            {previewState.error ? (
              <p className="ast-error" role="alert">
                <AlertCircle size={14} aria-hidden="true" />
                {previewState.error}
                <button type="button" onClick={() => setPreviewRefresh((current) => current + 1)}>
                  重试
                </button>
              </p>
            ) : null}
            {runError ? (
              <p className="ast-error" role="alert">
                <AlertCircle size={14} aria-hidden="true" />
                {runError}
              </p>
            ) : null}
            {pollError ? (
              <p className="ast-warning" role="status">
                <Clock3 size={14} aria-hidden="true" />
                {pollError}；服务器批次继续执行，恢复后自动刷新。
              </p>
            ) : null}
            {preview ? (
              <>
                <div className="ast-list-heading">
                  <div>
                    <span>工程工作负载</span>
                    <h3>{definition.title} 100 工程组合</h3>
                  </div>
                  <strong>
                    显示 {filteredCases.length} / 全部 {preview.cases.length}
                  </strong>
                </div>
                <CategoryFilterBar scope={scope} counts={counts} value={filter} total={preview.cases.length} onChange={changeFilter} testId="stress-category-filter" />
                {totalPages > 1 ? (
                  <div className="sdv-pagination" role="navigation" aria-label="回声与双讲列表分页">
                    <button type="button" className="button secondary" disabled={safePage <= 1} onClick={() => setPage((p) => Math.max(1, p - 1))}>
                      上一页
                    </button>
                    <span>
                      第 {safePage}/{totalPages} 页 · 第 {pageStartIndex + 1}–{Math.min(pageStartIndex + pagedExecuteCases.length, filteredCases.length)} 组（共 {filteredCases.length} 组）
                    </span>
                    <button type="button" className="button secondary" disabled={safePage >= totalPages} onClick={() => setPage((p) => Math.min(totalPages, p + 1))}>
                      下一页
                    </button>
                  </div>
                ) : null}
                <div className="ast-case-list">
                  {pagedExecuteCases.map((previewCase, caseIndex) => (
                    (() => {
                      const latest = latestCaseResultByCase.get(previewCase.case_id);
                      return (
                    <ExecuteCase
                      key={previewCase.case_id}
                      scope={scope}
                      previewCase={previewCase}
                      ordinal={pageStartIndex + caseIndex + 1}
                      visuals={preview.visuals.cases[previewCase.case_id]}
                      datasetId={preview.dataset_id}
                      playingTrack={playingTrack?.caseId === previewCase.case_id ? playingTrack : null}
                      onPlay={togglePlayback}
                      latest={latestCaseResultByCase.get(previewCase.case_id) ?? EMPTY_LATEST_CASE_RESULT}
                      singleDisabled={singleDisabled}
                      singleDisabledReason={singleDisabledReason}
                      singleSubmitting={singleLaunchingCaseId === previewCase.case_id}
                      singleError={singleCaseError?.caseId === previewCase.case_id ? singleCaseError.message : ""}
                      onRunSingle={launchSingleCase}
                      onViewResult={viewCaseResult}
                      onViewWaveform={viewCaseWaveform}
                      outputPlaybackKey={outputPlaybackKey}
                      onToggleOutputPlayback={toggleOutputPlayback}
                    />
                      );
                    })()
                  ))}
                </div>
              </>
            ) : previewState.loading ? (
              <div className="ast-empty" role="status">
                <Activity size={20} aria-hidden="true" />
                正在读取 100 组工程输入…
              </div>
            ) : null}
          </section>
        </div>
      ) : null}

      {activeTab === "waveforms" ? (
        <div>
          <section className="ast-panel" aria-label={`${definition.title}波形`}>
            {preview ? (
              <>
                <div className="ast-list-heading">
                  <div>
                    <span>波形与梅尔频谱</span>
                    <h3>四路输入、RX 前后参考、TX 与梅尔频谱</h3>
                  </div>
                  <strong>
                    显示 {waveformCases.length} / 全部 {preview.cases.length}
                  </strong>
                </div>
                <CategoryFilterBar scope={scope} counts={counts} value={filter} total={preview.cases.length} onChange={changeFilter} testId="stress-waveform-category-filter" />
                {focusedWaveformCase ? (
                  <div className="ast-focus-banner" role="status">
                    <span>正在查看本组波形 / Mel · {focusedWaveformCase.case_id}</span>
                    <button type="button" onClick={() => setFocusedWaveformCaseId("")}>
                      查看全部当前筛选
                    </button>
                  </div>
                ) : totalPages > 1 ? (
                  <div className="sdv-pagination" role="navigation" aria-label="波形列表分页">
                    <button type="button" className="button secondary" disabled={safePage <= 1} onClick={() => setPage((p) => Math.max(1, p - 1))}>
                      上一页
                    </button>
                    <span>
                      第 {safePage}/{totalPages} 页 · 第 {pageStartIndex + 1}–{Math.min(pageStartIndex + pagedExecuteCases.length, filteredCases.length)} 组（共 {filteredCases.length} 组）
                    </span>
                    <button type="button" className="button secondary" disabled={safePage >= totalPages} onClick={() => setPage((p) => Math.min(totalPages, p + 1))}>
                      下一页
                    </button>
                  </div>
                ) : null}
                <div className="ast-waveform-list">
                  {waveformCases.map((previewCase, caseIndex) => (
                    (() => {
                      const latest = latestCaseResultByCase.get(previewCase.case_id);
                      return (
                    <WaveformCase
                      key={`${filter}:${previewCase.case_id}`}
                      scope={scope}
                      previewCase={previewCase}
                      ordinal={(focusedWaveformCase ? 0 : pageStartIndex) + caseIndex + 1}
                      preview={preview}
                      batch={latestCaseResultByCase.get(previewCase.case_id)?.batch ?? null}
                      playingTrack={playingTrack?.caseId === previewCase.case_id ? playingTrack : null}
                      onPlay={togglePlayback}
                      latest={latestCaseResultByCase.get(previewCase.case_id) ?? EMPTY_LATEST_CASE_RESULT}
                      singleDisabled={singleDisabled}
                      singleDisabledReason={singleDisabledReason}
                      singleSubmitting={singleLaunchingCaseId === previewCase.case_id}
                      singleError={singleCaseError?.caseId === previewCase.case_id ? singleCaseError.message : ""}
                      onRunSingle={launchSingleCase}
                      onViewResult={viewCaseResult}
                      onViewWaveform={viewCaseWaveform}
                      outputPlaybackKey={outputPlaybackKey}
                      onToggleOutputPlayback={toggleOutputPlayback}
                    />
                      );
                    })()
                  ))}
                </div>
              </>
            ) : previewState.loading ? (
              <div className="ast-empty" role="status">
                <Activity size={20} aria-hidden="true" />
                正在读取预计算波形…
              </div>
            ) : (
              <div className="ast-empty" role="status">
                当前没有可显示的波形。
              </div>
            )}
          </section>
        </div>
      ) : null}

      {activeTab === "results" ? (
        <div>
          <section className="ast-panel" aria-label={`${definition.title}结果`}>
            <div className="ast-results-toolbar">
              <div>
                <span>批次历史</span>
                <h3>批次进度与历史</h3>
              </div>
              <label>
                <span>本地绑定历史</span>
                <select value={currentBatch?.batch_id ?? ""} disabled={historyState.loading || controlsLocked} onChange={(event) => void selectHistory(event.target.value)}>
                  <option value="">选择历史批次</option>
                  {historyState.value.map((batch) => (
                    <option key={batch.batch_id} value={batch.batch_id}>
                      {batchHistoryLabel(batch, scope)}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            {historyState.error ? (
              <p className="ast-error" role="alert">
                <AlertCircle size={14} aria-hidden="true" />
                {historyState.error}
              </p>
            ) : null}
            {pollError ? (
              <p className="ast-warning" role="status">
                <Clock3 size={14} aria-hidden="true" />
                {pollError}；服务器批次继续执行，恢复后自动刷新。
              </p>
            ) : null}
            {currentBatch ? (
              <>
                {batchIsSingle(currentBatch) ? (
                  <div className="ast-single-summary" role="status">
                    <strong>{batchScopeProgress(currentBatch)}</strong>
                    <span>本快照只执行 {currentBatch.selected_case_id}；其余未执行组仍保留在执行页和波形页，不伪造为完成。</span>
                  </div>
                ) : null}
                {focusedResultCase ? (
                  <div className="ast-focus-banner" role="status">
                    <span>正在查看本组结果 · {focusedResultCase.case_id}</span>
                    <button type="button" onClick={() => setFocusedResultCaseId("")}>
                      查看当前批次全部结果
                    </button>
                  </div>
                ) : null}
                <dl className="ast-stats">
                  <StatCard label="批次进度" value={`${currentBatch.progress.completed_cases}/${currentBatch.progress.total_cases}`} detail={batchIsSingle(currentBatch) ? `单组测试 · 全集${currentBatch.dataset_case_count ?? 100}组` : SCOPED_BATCH_LABELS[currentBatch.status]} />
                  <StatCard label="有限值合同观测" value={finiteItems.length} detail={`${items.length} 个当前快照观测`} />
                  <StatCard label="需复核" value={reviewItems.length} detail="等待工程复核" />
                  <StatCard label="不可测" value={notMeasuredItems.length} detail="不以 0 补造" />
                  <StatCard label="无效" value={invalidItems.length} detail="正式合规未测" />
                </dl>
                <div className="ast-progress" aria-hidden="true">
                  <span
                    style={{
                      width: `${currentBatch.progress.total_cases > 0 ? (currentBatch.progress.completed_cases / currentBatch.progress.total_cases) * 100 : 0}%`,
                    }}
                  />
                </div>
                <div className="ast-result-list">
                  {resultCases.map((caseRecord) => {
                    const previewIdentity = previewCaseFromBatchCase(caseRecord);
                    const item = itemByCase.get(caseRecord.case_id);
                    const tx = stressTxEvidence(previewIdentity, currentBatch, currentBatch.dataset_manifest_sha256);
                    const rx = stressRxEvidence(
                      previewIdentity,
                      currentBatch,
                      currentBatch.dataset_manifest_sha256,
                    );
                    const dynamicMic = stressDynamicMicrophoneEvidence(
                      previewIdentity,
                      currentBatch,
                      currentBatch.dataset_manifest_sha256,
                    );
                    const diagnostics = caseRecord.stress_diagnostics;
                    return (
                      <article key={caseRecord.case_id} className="ast-result-row" data-case-id={caseRecord.case_id} data-workload-category={caseRecord.provenance.workload_category}>
                        <header>
                          <div>
                            <span>{categoryLabel(scope, caseRecord.provenance.workload_category)}</span>
                            <h4>{caseTitle(scope, previewIdentity)}</h4>
                            <code>{caseRecord.case_id}</code>
                          </div>
                          <span className={`ast-status is-${caseRecord.status}`}>{CASE_STATUS_LABELS[caseRecord.status]}</span>
                        </header>
                        <dl>
                          <div>
                            <dt>合同观测</dt>
                            <dd>{item?.item_id ?? caseRecord.contract_item_ids[0]}</dd>
                          </div>
                          <div>
                            <dt>状态</dt>
                            <dd>{item ? SCOPED_STATUS_LABELS[item.status] : "尚无合同观测"}</dd>
                          </div>
                          <div>
                            <dt>合同观测值</dt>
                            <dd>{item?.value != null ? `${scopedValue(item.value)} ${scopedMetricUnit(item.metric)}` : item ? "无数值" : "—"}</dd>
                          </div>
                        </dl>
                        {dynamicMic.ready && dynamicMic.visual ? (
                          <div className="ast-echo-output">
                            <PrecomputedAudioFigures
                              visual={dynamicMic.visual}
                              label="麦克风混合输入 · 运行时合成（RX 后）"
                              caseId={caseRecord.case_id}
                              peakScale={dynamicMic.visual.peak}
                              melCeilingDb={dynamicMic.visual.mel_ceiling_db}
                              showMel={false}
                            />
                            <div className="ast-tx-actions">
                              <a
                                className="ast-tx-download"
                                href={dynamicMic.url}
                                download={`${caseRecord.case_id}-dynamic-microphone.wav`}
                              >
                                <Download size={12} aria-hidden="true" />
                                下载运行时麦克风
                              </a>
                              <OutputAudioPreview
                                caseId={caseRecord.case_id}
                                label="运行时合成麦克风音频"
                                url={dynamicMic.url}
                                playing={
                                  outputPlaybackKey
                                  === `${currentBatch.batch_id}:${caseRecord.case_id}:dynamic-mic`
                                }
                                onToggle={() =>
                                  toggleOutputPlayback(
                                    `${currentBatch.batch_id}:${caseRecord.case_id}:dynamic-mic`,
                                  )}
                              />
                            </div>
                          </div>
                        ) : (
                          <p className="ast-inline-note">
                            运行时麦克风：{dynamicMic.reason}
                          </p>
                        )}
                        {rx.ready && rx.visual ? (
                          <div className="ast-echo-output">
                            <PrecomputedAudioFigures
                              visual={rx.visual}
                              label="回声参考 · VCP RX 后"
                              caseId={caseRecord.case_id}
                              peakScale={rx.visual.peak}
                              melCeilingDb={rx.visual.mel_ceiling_db}
                              showMel={false}
                            />
                            <div className="ast-tx-actions">
                              <a
                                className="ast-tx-download"
                                href={rx.url}
                                download={`${caseRecord.case_id}-rx-after.wav`}
                              >
                                <Download size={12} aria-hidden="true" />
                                下载 RX 后
                              </a>
                              <OutputAudioPreview
                                caseId={caseRecord.case_id}
                                label="RX 后回声参考音频"
                                url={rx.url}
                                playing={
                                  outputPlaybackKey
                                  === `${currentBatch.batch_id}:${caseRecord.case_id}:rx`
                                }
                                onToggle={() =>
                                  toggleOutputPlayback(
                                    `${currentBatch.batch_id}:${caseRecord.case_id}:rx`,
                                  )}
                              />
                            </div>
                          </div>
                        ) : (
                          <p className="ast-inline-note">
                            RX 后：{rx.reason}
                          </p>
                        )}
                        {tx.ready && tx.visual ? (
                          <PrecomputedAudioFigures visual={tx.visual} label="测试后音频 · TX" caseId={caseRecord.case_id} peakScale={tx.visual.peak} melCeilingDb={tx.visual.mel_ceiling_db} showMel={false} />
                        ) : (
                          <p className="ast-inline-note">{tx.reason}</p>
                        )}
                        {tx.ready ? (
                          <div className="ast-tx-actions">
                            <a className="ast-tx-download" href={tx.url} download={`${caseRecord.case_id}-tx.wav`}>
                              <Download size={12} aria-hidden="true" />
                              下载 TX 音频
                            </a>
                            <OutputAudioPreview
                              caseId={caseRecord.case_id}
                              label="TX 输出音频"
                              url={tx.url}
                              playing={outputPlaybackKey === `${currentBatch.batch_id}:${caseRecord.case_id}:tx`}
                              onToggle={() => toggleOutputPlayback(`${currentBatch.batch_id}:${caseRecord.case_id}:tx`)}
                            />
                            {caseRecord.output_sha256?.tx ? <code className="ast-tx-hash">SHA-256 {caseRecord.output_sha256.tx}</code> : null}
                          </div>
                        ) : null}
                        {diagnostics ? <SegmentDiagnostics scope={scope} diagnostics={diagnostics} /> : null}
                        {item?.reason || diagnostics?.reason ? <p className="ast-reason">{item?.reason || diagnostics?.reason}</p> : null}
                      </article>
                    );
                  })}
                </div>
                <footer className="ast-boundary">
                  <ShieldAlert size={14} aria-hidden="true" />
                  汇总来自真实合同观测；需复核、不可测和无效状态均保留。{batchIsSingle(currentBatch) ? " 当前为单组快照，其余工程组合未在本批次执行。" : ""} 工程代理结果，正式 ITU-T 合规未测。
                </footer>
              </>
            ) : historyState.loading ? (
              <div className="ast-empty" role="status">
                <Activity size={20} aria-hidden="true" />
                正在恢复批次历史…
              </div>
            ) : (
              <div className="ast-empty" role="status">
                <History size={20} aria-hidden="true" />
                当前模式 / 数据集 / 算法尚无批次。
              </div>
            )}
          </section>
        </div>
      ) : null}

      {activeTab === "description" ? (
        <div>
          <section className="ast-panel" aria-label={`${definition.title}说明`}>
            <div className="ast-description-heading">
              <div>
                <span>真实测试树</span>
                <h3>五阶段工程测试树</h3>
              </div>
              <strong>{scope === "echo" ? "干净近端参考静音，不依赖近端活动检测" : "双讲活动采用设计重叠窗口"}</strong>
            </div>
            {swbNotice ? <p className="ast-resampling-note is-section">{swbNotice}</p> : null}
            <ol className="ast-tree">
              {definition.stages.map((stage, index) => (
                <li key={stage.label}>
                  <span>{String(index + 1).padStart(2, "0")}</span>
                  <div>
                    <strong>{stage.label}</strong>
                    <p>{stage.detail}</p>
                  </div>
                </li>
              ))}
            </ol>
            <section className="ast-boundary-card">
              <h4>专属测试范围边界</h4>
              <p>{definition.boundary}</p>
            </section>
            <div className="ast-contract-grid">
              {contracts.map((contract: ScopedItem) => (
                <article key={contract.id}>
                  <span>{contract.source_cell}</span>
                  <h4>{contract.test_name}</h4>
                  <p>
                    {contract.family} · {contract.unit} · {contract.lower_limit ?? "—"} / {contract.upper_limit ?? "—"}
                  </p>
                </article>
              ))}
            </div>
            <div className="ast-boundary">
              <ListChecks size={15} aria-hidden="true" />
              目录合同行由 API 返回；界面不把 100 个工程观测宣称为正式 TCLw、A1/A2 或标准合规。
            </div>
          </section>
        </div>
      ) : null}
    </section>
  );
}
