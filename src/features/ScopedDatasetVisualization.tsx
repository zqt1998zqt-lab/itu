import {
  useEffect,
  useId,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import {
  AlertCircle,
  ExternalLink,
  Play,
  RefreshCw,
} from "lucide-react";
import {
  ACOUSTIC_SIGNAL_DESCRIPTIONS,
  ACOUSTIC_SIGNAL_LABELS,
} from "../domain/acoustic-terminology.ts";
import type { FiveSignalMode } from "./five-signal-flow-model";
import {
  BACKGROUND_NOISE_SCENE_LABELS,
  readScopedBackgroundNoisePreview,
  readSceneConditionWaveformPreview,
  type BackgroundNoiseScene,
  type ScopedBackgroundNoisePreview,
  type ScopedDataset,
  type ScopedInputPreviewCase,
  type ScopedInputSignalRole,
  type ScopedPreviewWaveformTrack,
  type ScopedSceneConditionWaveformPreview,
} from "./scoped-acoustic-model";
import {
  paginateScopedCases,
  type ScopedCasePageCursor,
} from "./scoped-case-pagination.ts";
import {
  CONVERSATION_MODE_LABELS,
  CONVERSATION_MODES,
  conversationModeOf,
  sceneConditionCounts,
  sceneConditionProfileForDatasetId,
  type ConversationMode,
} from "./scene-condition-profiles";
import "./scoped-dataset-visualization.css";

const PREVIEW_URL =
  "/api/alignment/acceptance/background-noise/scenes";
const DATASET_URL = "/api/alignment/acceptance/datasets";
const SCENE_DATASET_URL = "/api/alignment/acceptance/scene-condition-datasets";
const SCENE_PREVIEW_URL = "/api/alignment/acceptance/scene-condition-preview";
const SIGNAL_COLOR = "#4f7fb8";
const WAVEFORM_WIDTH = 720;
const WAVEFORM_HEIGHT = 48;
const WAVEFORM_CENTER = WAVEFORM_HEIGHT / 2;
const WAVEFORM_AMPLITUDE = WAVEFORM_CENTER - 1;

const TRACK_ROLES = [
  "microphone",
  "reference",
  "far_end",
  "noise",
] as const satisfies readonly ScopedInputSignalRole[];

const FAMILY_LABELS: Record<string, string> = {
  g_mos: "整体质量",
  n_mos: "背景噪声质量",
  s_mos: "语音质量",
  cng_level: "舒适噪声电平",
  cng_spectrum: "舒适噪声频谱",
};

const SPEECH_FAMILIES = new Set(["g_mos", "n_mos", "s_mos"]);
const CNG_FAMILIES = new Set(["cng_level", "cng_spectrum"]);

export type ScopedCaseRenderContext = {
  source: "execution" | "scene-condition";
  datasetId: string;
  mode: FiveSignalMode;
  scene: BackgroundNoiseScene;
  requestKey: string;
  farEndVisual?: ScopedPreviewWaveformTrack | null;
};

type ScopedPreview =
  | ScopedBackgroundNoisePreview
  | ScopedSceneConditionWaveformPreview;

type Props = {
  source?: "execution" | "scene-condition";
  dataset: ScopedDataset | null;
  mode: FiveSignalMode;
  scene: BackgroundNoiseScene;
  expectedItemCount: number;
  active: boolean;
  loading: boolean;
  refreshToken: number;
  refreshDisabled: boolean;
  onRefresh: () => void;
  onOpenWaveforms: () => void;
  onOpenExtendedWaveforms?: () => void;
  extendedPreviewCount?: number;
  renderCaseActions?: (
    previewCase: ScopedInputPreviewCase,
    context: ScopedCaseRenderContext,
  ) => ReactNode;
  renderCaseResult?: (
    previewCase: ScopedInputPreviewCase,
    context: ScopedCaseRenderContext,
  ) => ReactNode;
};

type CaseCategory = "speech" | "cng" | "mixed" | "other";
type ConversationFilter = "all" | ConversationMode;

type PreviewSegment = {
  start_s: number;
  end_s: number;
  label: string;
};

type PreviewState = {
  key: string;
  preview: ScopedPreview | null;
  loading: boolean;
  error: string;
};

type PlayingTrack = {
  requestKey: string;
  caseId: string;
  role: (typeof TRACK_ROLES)[number];
};

type RenderedTrack = {
  role: (typeof TRACK_ROLES)[number];
  visual: ScopedPreviewWaveformTrack | null;
  path: string;
  renderable: boolean;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function apiErrorDetail(value: unknown): string {
  if (!isRecord(value)) return "";
  if (typeof value.detail === "string") return value.detail;
  if (!Array.isArray(value.detail)) return "";
  return value.detail
    .map((item) =>
      isRecord(item) && typeof item.msg === "string" ? item.msg : "",
    )
    .filter(Boolean)
    .join("；");
}

async function requestPreview(
  url: string,
  signal: AbortSignal,
): Promise<unknown> {
  const response = await fetch(url, {
    cache: "no-cache",
    signal,
  });
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    throw new Error(
      response.ok
        ? "测试集预览返回了无法解析的 JSON"
        : `测试集预览读取失败（HTTP ${response.status}）`,
    );
  }
  if (!response.ok) {
    throw new Error(
      apiErrorDetail(body) ||
        `测试集预览读取失败（HTTP ${response.status}）`,
    );
  }
  return body;
}

function errorMessage(reason: unknown, fallback: string): string {
  return reason instanceof Error && reason.message
    ? reason.message
    : fallback;
}

function previewMatches(
  preview: ScopedPreview | null,
  datasetId: string,
  mode: FiveSignalMode,
  scene: BackgroundNoiseScene,
): preview is ScopedPreview {
  return (
    preview?.dataset_id === datasetId &&
    preview.mode === mode &&
    preview.scene === scene
  );
}

function caseCategory(previewCase: ScopedInputPreviewCase): CaseCategory {
  const hasSpeech = previewCase.families.some((family) =>
    SPEECH_FAMILIES.has(family),
  );
  const hasCng = previewCase.families.some((family) =>
    CNG_FAMILIES.has(family),
  );
  if (hasSpeech && hasCng) return "mixed";
  if (hasSpeech) return "speech";
  if (hasCng) return "cng";
  return "other";
}

function caseCategoryLabel(category: CaseCategory): string {
  if (category === "speech") return "语音质量样本";
  if (category === "cng") return "舒适噪声样本";
  if (category === "mixed") return "语音与 CNG 样本";
  return "测试样本";
}

function caseCategoryShortLabel(category: CaseCategory): string {
  if (category === "speech") return "语音";
  if (category === "cng") return "CNG";
  if (category === "mixed") return "混合";
  return "其他";
}

const SEGMENT_LABELS: Record<string, string> = {
  speech: "近端语音",
  near_only: "近端单讲",
  far_speech: "远端语音",
  far_only: "远端单讲",
  double_talk: "双讲",
  transition: "单讲 / 双讲切换",
  quiet: "静音",
  noise_only: "噪声尾段",
  cng_active: "CNG 活动",
  cng_settling: "CNG 收敛",
  cng_noise_observation: "噪声观测",
  idle: "静音",
};

function formatSegmentTime(seconds: number): string {
  return Number.isInteger(seconds)
    ? String(seconds)
    : seconds.toLocaleString("zh-CN", {
        maximumFractionDigits: 2,
      });
}

function renderMetadataOf(
  previewCase: ScopedInputPreviewCase,
): Record<string, unknown> | null {
  const metadata = previewCase.provenance?.render_metadata;
  return isRecord(metadata) ? metadata : null;
}

function metadataWindow(
  value: unknown,
): [number, number] | null {
  if (
    !Array.isArray(value) ||
    value.length !== 2 ||
    typeof value[0] !== "number" ||
    !Number.isFinite(value[0]) ||
    typeof value[1] !== "number" ||
    !Number.isFinite(value[1]) ||
    value[0] < 0 ||
    value[1] <= value[0]
  ) {
    return null;
  }
  return [value[0], value[1]];
}

function windowSummary(window: [number, number] | null): string {
  return window
    ? `${formatSegmentTime(window[0])}–${formatSegmentTime(window[1])}s`
    : "provenance 声明的时段";
}

function previewSegments(
  previewCase: ScopedInputPreviewCase,
): PreviewSegment[] {
  const renderMetadata = renderMetadataOf(previewCase);
  if (!renderMetadata || !Array.isArray(renderMetadata.segments)) {
    return [];
  }
  return renderMetadata.segments.flatMap((segment) => {
    if (
      !isRecord(segment) ||
      typeof segment.start_s !== "number" ||
      !Number.isFinite(segment.start_s) ||
      typeof segment.end_s !== "number" ||
      !Number.isFinite(segment.end_s) ||
      typeof segment.label !== "string"
    ) {
      return [];
    }
    return [{
      start_s: segment.start_s,
      end_s: segment.end_s,
      label: segment.label,
    }];
  });
}

function segmentSummary(previewCase: ScopedInputPreviewCase): string {
  return previewSegments(previewCase)
    .map(
      (segment) =>
        `${formatSegmentTime(segment.start_s)}–${formatSegmentTime(
          segment.end_s,
        )}s ${
          SEGMENT_LABELS[segment.label] ?? segment.label
        }`,
    )
    .join(" · ");
}

function conversationComposition(
  mode: ConversationMode,
  previewCase?: ScopedInputPreviewCase,
): string {
  const metadata = previewCase ? renderMetadataOf(previewCase) : null;
  const echoPathCount =
    metadata && Array.isArray(metadata.echo_paths)
      ? metadata.echo_paths.length
      : null;
  const echoPhrase =
    echoPathCount !== null ? `经 ${echoPathCount} 条回声路径` : "经回声路径";
  const farEndDelivery = previewCase?.provenance?.far_end_delivery;
  const farEndSilent =
    isRecord(farEndDelivery) && farEndDelivery.digital_silence === true;
  const nearSpeech = previewCase?.provenance?.near_speech;
  const nearSilent =
    isRecord(nearSpeech) && nearSpeech.source_kind === "digital_silence";

  if (mode === "near_single") {
    return `reference 承载干净近端参考；${
      farEndSilent
        ? "far_end 为数字静音"
        : "far_end 关系以 provenance 为准"
    }；microphone 含近端语音与背景噪声。`;
  }
  if (mode === "far_single") {
    return `far_end 承载 VCP RX 前回声参考，${echoPhrase}进入 microphone；点击测试后生成 RX 后参考并用于 TX；${
      nearSilent
        ? "reference 为数字静音"
        : "reference 关系以 provenance 为准"
    }；背景噪声叠加在麦克风输入中。`;
  }
  if (mode === "double_talk") {
    return `reference 与 far_end 同时活动，${echoPhrase}的远端成分和背景噪声共同叠加在 microphone；点击测试后生成 RX 后参考并用于 TX。`;
  }
  if (mode === "transition") {
    return `reference 与 far_end 按 provenance 分段切换远端单讲、近端单讲、双讲和噪声尾段；${echoPhrase}只在远端活动区段进入 microphone；点击测试后生成 RX 后参考并用于 TX。`;
  }
  const preheat = metadataWindow(metadata?.cng_preheat_window_s);
  const observation = metadataWindow(metadata?.noise_observation_window_s);
  const preheatText = preheat
    ? `（${windowSummary(preheat)}）`
    : "";
  const observationText = observation
    ? `${windowSummary(observation)} 才进入噪声观测窗`
    : "随后进入 provenance 声明的噪声观测窗";
  return `CNG 渲染先进入预热窗${preheatText}，随后收敛，${observationText}；整段不按纯噪声解释。`;
}

function engineeringCombinationSummary(
  totalCases: number,
  previewCase?: ScopedInputPreviewCase,
): string {
  const policy = previewCase?.provenance?.source_reuse_policy;
  const sourceReuseDeclared =
    isRecord(policy) &&
    policy.source_audio_may_be_reused_across_cases === true &&
    previewCase?.provenance?.independent_source_recordings === false;
  return sourceReuseDeclared
    ? `${totalCases}种工程组合，跨组合允许复用源录音；并非${totalCases}段独立录音。`
    : `${totalCases}种工程组合；来源复用关系以 provenance 为准。`;
}

function conversationModeCounts(
  cases: ScopedInputPreviewCase[],
): Record<ConversationMode, number> {
  const counts = Object.fromEntries(
    CONVERSATION_MODES.map((mode) => [mode, 0]),
  ) as Record<ConversationMode, number>;
  for (const previewCase of cases) {
    const mode = conversationModeOf(previewCase);
    if (mode) counts[mode] += 1;
  }
  return counts;
}

function buildCaseLabels(
  cases: ScopedInputPreviewCase[],
): Map<string, string> {
  const counters: Record<CaseCategory, number> = {
    speech: 0,
    cng: 0,
    mixed: 0,
    other: 0,
  };
  const conversationCounters = Object.fromEntries(
    CONVERSATION_MODES.map((mode) => [mode, 0]),
  ) as Record<ConversationMode, number>;
  const labels = new Map<string, string>();
  for (const previewCase of cases) {
    const conversationMode = conversationModeOf(previewCase);
    if (conversationMode) {
      conversationCounters[conversationMode] += 1;
      labels.set(
        previewCase.case_id,
        `${CONVERSATION_MODE_LABELS[conversationMode]} ${String(
          conversationCounters[conversationMode],
        ).padStart(2, "0")}`,
      );
      continue;
    }
    const category = caseCategory(previewCase);
    counters[category] += 1;
    labels.set(
      previewCase.case_id,
      `${caseCategoryLabel(category)} ${counters[category]}`,
    );
  }
  return labels;
}

function humanFamilyLabels(
  previewCase: ScopedInputPreviewCase,
): string {
  const labels = [...new Set(previewCase.families)]
    .map((family) => FAMILY_LABELS[family])
    .filter(Boolean);
  return labels.length > 0 ? labels.join(" · ") : "未声明合同族";
}

function formatDuration(seconds: number): string {
  return `${seconds.toLocaleString("zh-CN", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })} s`;
}

function formatSampleRate(sampleRate: number): string {
  return `${(sampleRate / 1000).toLocaleString("zh-CN", {
    maximumFractionDigits: 2,
  })} kHz`;
}

function decodeInt16LittleEndian(encoded: string): Int16Array {
  const binary = atob(encoded);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  if (bytes.byteLength % 2 !== 0) {
    throw new Error("预计算波形数组字节长度无效");
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

function envelopePath(
  visual: ScopedPreviewWaveformTrack,
  groupPeak: number,
): string {
  const minimum = decodeInt16LittleEndian(visual.wave_min_i16);
  const maximum = decodeInt16LittleEndian(visual.wave_max_i16);
  if (minimum.length !== WAVEFORM_WIDTH || maximum.length !== WAVEFORM_WIDTH) {
    throw new Error("预计算波形数组长度无效");
  }
  const scale = visual.wave_scale / 32767;
  const denominator = groupPeak > 0 ? groupPeak : 1;
  const points: string[] = [];
  for (let index = 0; index < WAVEFORM_WIDTH; index += 1) {
    const normalized = Math.max(
      -1,
      Math.min(1, (maximum[index] * scale) / denominator),
    );
    points.push(
      `${index},${(WAVEFORM_CENTER - normalized * WAVEFORM_AMPLITUDE).toFixed(2)}`,
    );
  }
  for (let index = WAVEFORM_WIDTH - 1; index >= 0; index -= 1) {
    const normalized = Math.max(
      -1,
      Math.min(1, (minimum[index] * scale) / denominator),
    );
    points.push(
      `${index},${(WAVEFORM_CENTER - normalized * WAVEFORM_AMPLITUDE).toFixed(2)}`,
    );
  }
  return `M${points.join(" L")} Z`;
}

function inputAssetUrl(
  datasetId: string,
  caseId: string,
  role: ScopedInputSignalRole,
  version: string,
  source: "execution" | "scene-condition",
): string {
  const base = source === "scene-condition" ? SCENE_DATASET_URL : DATASET_URL;
  return `${base}/${encodeURIComponent(
    datasetId,
  )}/cases/${encodeURIComponent(caseId)}/inputs/${encodeURIComponent(
    role,
  )}?v=${encodeURIComponent(version)}`;
}

function PreviewSkeleton() {
  return (
    <div className="sdv-loading" role="status" aria-live="polite">
      <span className="sr-only">正在读取场景测试集…</span>
      <div className="sdv-stats sdv-stats-skeleton" aria-hidden="true">
        {Array.from({ length: 5 }, (_, index) => (
          <span className="sdv-skeleton-line" key={index} />
        ))}
      </div>
      <div className="sdv-distribution-skeleton" aria-hidden="true">
        <span className="sdv-skeleton-bar" />
        <span className="sdv-skeleton-line is-short" />
      </div>
      <div className="sdv-case-skeleton" aria-hidden="true">
        <span className="sdv-skeleton-line is-select" />
        <span className="sdv-skeleton-line is-short" />
      </div>
      <div className="sdv-card-grid" aria-hidden="true">
        {TRACK_ROLES.map((role) => (
          <div className="sdv-card sdv-card-skeleton" key={role}>
            <span className="sdv-skeleton-line is-title" />
            <span className="sdv-skeleton-line is-meta" />
            <span className="sdv-skeleton-wave" />
            <span className="sdv-skeleton-button" />
          </div>
        ))}
      </div>
    </div>
  );
}

type WaveformUnitProps = {
  previewCase: ScopedInputPreviewCase;
  caseLabel: string;
  index?: number;
  datasetId: string;
  visuals:
    | ScopedBackgroundNoisePreview["visuals"]
    | ScopedSceneConditionWaveformPreview["visuals"]
    | undefined;
  source: "execution" | "scene-condition";
  requestKey: string;
  playingTrack: PlayingTrack | null;
  waveformA11yId: string;
  variant: "list" | "detail";
  renderCaseActions?: Props["renderCaseActions"];
  renderCaseResult?: Props["renderCaseResult"];
  renderContext: ScopedCaseRenderContext;
  onPlay: (track: PlayingTrack) => void;
  onOpenWaveforms: () => void;
};

function WaveformUnit({
  previewCase,
  caseLabel,
  index,
  datasetId,
  visuals,
  source,
  requestKey,
  playingTrack,
  waveformA11yId,
  variant,
  renderCaseActions,
  renderCaseResult,
  renderContext,
  onPlay,
  onOpenWaveforms,
}: WaveformUnitProps) {
  const caseVisuals = visuals?.cases[previewCase.case_id];
  const waveformTracks = useMemo<RenderedTrack[]>(() => {
    const groupPeak = Math.max(
      0,
      ...TRACK_ROLES.map((role) => caseVisuals?.[role]?.peak ?? 0),
    );
    return TRACK_ROLES.map((role) => {
      const visual = caseVisuals?.[role] ?? null;
      if (!visual) {
        return {
          role,
          visual,
          path: "",
          renderable: false,
        };
      }
      try {
        return {
          role,
          visual,
          path: envelopePath(visual, groupPeak),
          renderable: true,
        };
      } catch {
        return {
          role,
          visual,
          path: "",
          renderable: false,
        };
      }
    });
  }, [caseVisuals]);
  const allWaveformsAvailable =
    waveformTracks.length === TRACK_ROLES.length &&
    waveformTracks.every((track) => track.renderable);
  const farEndVisual =
    waveformTracks.find((track) => track.role === "far_end")?.visual ?? null;
  const caseRenderContext: ScopedCaseRenderContext = {
    ...renderContext,
    farEndVisual,
  };
  const category = caseCategory(previewCase);
  const conversationMode = conversationModeOf(previewCase);
  const segments = conversationMode
    ? segmentSummary(previewCase)
    : "";
  const caseA11yId = `${waveformA11yId}-${previewCase.case_id}`;

  return (
    <article
      className={`sdv-waveform-unit is-${variant}`}
      data-case-id={previewCase.case_id}
      data-conversation-mode={conversationMode ?? "legacy"}
    >
      {variant === "list" ? (
        <header className="sdv-waveform-unit-header">
          <div className="sdv-waveform-unit-identity">
            {index !== undefined ? (
              <span className="sdv-waveform-unit-index" aria-label={`第 ${String(index).padStart(2, "0")} 组`}>
                <span aria-hidden="true">{String(index).padStart(2, "0")}</span>
              </span>
            ) : null}
            <div className="sdv-waveform-unit-copy">
              <div className="sdv-waveform-unit-title">
                <strong>{caseLabel}</strong>
                <span className={`sdv-case-list-kind is-${category}`}>
                  {caseCategoryShortLabel(category)}
                </span>
                {conversationMode ? (
                  <span className="sdv-waveform-unit-mode">
                    {CONVERSATION_MODE_LABELS[conversationMode]}
                  </span>
                ) : null}
              </div>
              <small title={previewCase.case_id}>
                {previewCase.case_id}
              </small>
            </div>
          </div>
          <div className="sdv-waveform-unit-controls">
            <span className="sdv-waveform-unit-contract">
              适用合同：{humanFamilyLabels(previewCase)}
            </span>
            {renderCaseActions?.(previewCase, caseRenderContext)}
          </div>
        </header>
      ) : null}

      {variant === "list"
        ? renderCaseResult?.(previewCase, caseRenderContext)
        : null}

      {conversationMode ? (
        <p className="sdv-composition is-conversation">
          <strong>{CONVERSATION_MODE_LABELS[conversationMode]}</strong>
          <span>
            {conversationComposition(conversationMode, previewCase)}
          </span>
          {segments ? (
            <span className="sdv-composition-segments">
              分段：{segments}
            </span>
          ) : null}
        </p>
      ) : null}

      <div className="sdv-card-grid">
        {waveformTracks.map(({ role, visual, path, renderable }) => {
          const label = ACOUSTIC_SIGNAL_LABELS[role];
          const titleElementId = `${caseA11yId}-${role}-title`;
          const descriptionElementId = `${caseA11yId}-${role}-description`;
          const isPlaying =
            playingTrack?.requestKey === requestKey &&
            playingTrack?.caseId === previewCase.case_id &&
            playingTrack.role === role;
          const assetUrl = inputAssetUrl(
            datasetId,
            previewCase.case_id,
            role,
            previewCase.signals_sha256[role],
            source,
          );
          return (
            <article className="sdv-card" key={role}>
              <div className="sdv-card-heading">
                <h3>{label}</h3>
                {visual?.peak === 0 &&
                (role === "far_end" || role === "reference") ? (
                  <span className="sdv-silence-badge">数字静音</span>
                ) : null}
              </div>
              <p className="sdv-card-meta">
                {visual
                  ? `${formatDuration(visual.duration_s)} · ${formatSampleRate(
                      visual.sample_rate,
                    )} · ${visual.channels} 通道`
                  : "时长 / 采样率 / 通道数未提供"}
              </p>
              <div className="sdv-waveform">
                {renderable && visual ? (
                  <svg
                    viewBox={`0 0 ${WAVEFORM_WIDTH} ${WAVEFORM_HEIGHT}`}
                    preserveAspectRatio="none"
                    role="img"
                    aria-labelledby={`${titleElementId} ${descriptionElementId}`}
                  >
                    <title id={titleElementId}>
                      {label}预计算波形包络
                    </title>
                    <desc id={descriptionElementId}>
                      {ACOUSTIC_SIGNAL_DESCRIPTIONS[role]}
                      纵轴按当前样本四路共同峰值归一化，数据来自预计算
                      wave_min_i16 与 wave_max_i16。
                    </desc>
                    {visual.peak === 0 ? (
                      <line
                        x1="0"
                        x2={WAVEFORM_WIDTH}
                        y1={WAVEFORM_CENTER}
                        y2={WAVEFORM_CENTER}
                        stroke={SIGNAL_COLOR}
                        strokeWidth="1"
                        opacity={0.76}
                      />
                    ) : (
                      <path
                        d={path}
                        fill={SIGNAL_COLOR}
                        opacity={0.76}
                      />
                    )}
                  </svg>
                ) : (
                  <span>暂无预计算波形</span>
                )}
              </div>
              {isPlaying ? (
                <audio
                  key={`${previewCase.case_id}:${role}`}
                  controls
                  autoPlay
                  preload="none"
                  src={assetUrl}
                  aria-label={`试听 ${label}`}
                >
                  当前浏览器不支持音频播放。
                </audio>
              ) : (
                <button
                  className="sdv-listen"
                  type="button"
                  aria-label={`试听 ${caseLabel} ${label}`}
                  onClick={() =>
                    onPlay({
                      requestKey,
                      caseId: previewCase.case_id,
                      role,
                    })
                  }
                >
                  <Play size={11} fill="currentColor" aria-hidden="true" />
                  试听
                </button>
              )}
            </article>
          );
        })}
      </div>

      {!allWaveformsAvailable ? (
        <div className="sdv-missing-visuals">
          <span>暂无预计算波形</span>
          <button type="button" onClick={onOpenWaveforms}>
            查看完整波形
          </button>
        </div>
      ) : null}

      {!conversationMode && farEndVisual ? (
        <p className="sdv-composition">
          {farEndVisual.peak === 0
            ? "当前样本：干净近端参考 + 噪声 → 麦克风混合输入（未注入远端回声）"
            : "远端播放信号经回声路径参与麦克风混合输入，具体组成以样本定义为准"}
        </p>
      ) : null}
    </article>
  );
}

export default function ScopedDatasetVisualization({
  source = "execution",
  dataset,
  mode,
  scene,
  expectedItemCount,
  active,
  loading,
  refreshToken,
  refreshDisabled,
  onRefresh,
  onOpenWaveforms,
  onOpenExtendedWaveforms,
  extendedPreviewCount,
  renderCaseActions,
  renderCaseResult,
}: Props) {
  const titleId = useId().replace(/[^A-Za-z0-9_-]/g, "");
  const caseSelectId = useId().replace(/[^A-Za-z0-9_-]/g, "");
  const waveformA11yId = useId().replace(/[^A-Za-z0-9_-]/g, "");
  const datasetId = dataset?.dataset_id ?? "";
  const requestKey =
    active && datasetId
      ? [
          source,
          datasetId,
          mode,
          scene,
          String(refreshToken),
        ].join(":")
      : "";
  const [previewState, setPreviewState] = useState<PreviewState>({
    key: "",
    preview: null,
    loading: false,
    error: "",
  });
  const [selectedCaseId, setSelectedCaseId] = useState("");
  const [playingTrack, setPlayingTrack] = useState<PlayingTrack | null>(null);
  const [conversationFilter, setConversationFilter] =
    useState<ConversationFilter>("all");
  const [casePageCursor, setCasePageCursor] =
    useState<ScopedCasePageCursor>({
      scopeKey: "",
      pageIndex: 0,
    });
  const casePageScopeKey = [
    active ? "active" : "inactive",
    requestKey,
    conversationFilter,
  ].join(":");

  useEffect(() => {
    setPlayingTrack(null);
    setConversationFilter("all");
    if (!active || !datasetId || !requestKey) {
      setPreviewState({
        key: requestKey,
        preview: null,
        loading: false,
        error: "",
      });
      return;
    }

    const controller = new AbortController();
    let disposed = false;
    setPreviewState({
      key: requestKey,
      preview: null,
      loading: true,
      error: "",
    });

    const params = new URLSearchParams({
      dataset_id: datasetId,
      mode,
    });
    if (source === "scene-condition") {
      params.set("scene", scene);
      params.set("visual_detail", "waveform");
    }
    const url = source === "scene-condition"
      ? `${SCENE_PREVIEW_URL}?${params.toString()}`
      : `${PREVIEW_URL}/${encodeURIComponent(scene)}/preview?${params.toString()}`;
    void requestPreview(
      url,
      controller.signal,
    )
      .then((body) =>
        (source === "scene-condition" ? readSceneConditionWaveformPreview : readScopedBackgroundNoisePreview)(
          body,
          datasetId,
          mode,
          scene,
        ),
      )
      .then((next) => {
        if (disposed || controller.signal.aborted) return;
        setPreviewState({
          key: requestKey,
          preview: next,
          loading: false,
          error: "",
        });
        setSelectedCaseId((current) =>
          next.cases.some((item) => item.case_id === current)
            ? current
            : next.cases[0]?.case_id ?? "",
        );
      })
      .catch((reason: unknown) => {
        if (disposed || controller.signal.aborted) return;
        setPreviewState({
          key: requestKey,
          preview: null,
          loading: false,
          error: errorMessage(reason, "场景测试集预览读取失败"),
        });
      });

    return () => {
      disposed = true;
      controller.abort();
    };
  }, [active, datasetId, mode, refreshToken, requestKey, scene, source]);

  const currentState = previewState.key === requestKey ? previewState : null;
  const previewLoading =
    Boolean(active && dataset) &&
    (!currentState || currentState.loading);
  const previewError = currentState?.error ?? "";
  const visiblePreview =
    dataset &&
    currentState?.preview &&
    previewMatches(
      currentState.preview,
      dataset.dataset_id,
      mode,
      scene,
    )
      ? currentState.preview
      : null;
  const caseLabels = useMemo(
    () => buildCaseLabels(visiblePreview?.cases ?? []),
    [visiblePreview?.cases],
  );
  const activeProfile = useMemo(
    () =>
      datasetId
        ? sceneConditionProfileForDatasetId(datasetId)
        : null,
    [datasetId],
  );
  const expectedCounts = activeProfile
    ? sceneConditionCounts(activeProfile)
    : null;
  const conversationCounts = useMemo(
    () => conversationModeCounts(visiblePreview?.cases ?? []),
    [visiblePreview?.cases],
  );
  const hasConversationModes = CONVERSATION_MODES.some(
    (conversationMode) => conversationCounts[conversationMode] > 0,
  );
  const showConversationFilter =
    source === "scene-condition" && hasConversationModes;
  const filteredCases = useMemo(() => {
    const cases = visiblePreview?.cases ?? [];
    return conversationFilter === "all"
      ? cases
      : cases.filter(
          (previewCase) =>
            conversationModeOf(previewCase) === conversationFilter,
        );
  }, [conversationFilter, visiblePreview?.cases]);
  const casePage = useMemo(
    () =>
      paginateScopedCases(
        filteredCases,
        casePageScopeKey,
        casePageCursor,
      ),
    [casePageCursor, casePageScopeKey, filteredCases],
  );
  const caseOrdinals = useMemo(
    () =>
      new Map(
        (visiblePreview?.cases ?? []).map((previewCase, index) => [
          previewCase.case_id,
          index + 1,
        ]),
      ),
    [visiblePreview?.cases],
  );
  const activeConversationLabel =
    conversationFilter === "all"
      ? "全部"
      : CONVERSATION_MODE_LABELS[conversationFilter];
  const casePageAriaLabel = [
    BACKGROUND_NOISE_SCENE_LABELS[scene],
    dataset?.name,
    `${activeConversationLabel}工程组合`,
  ]
    .filter(Boolean)
    .join(" · ");
  const conversationPreview =
    activeProfile?.schemaVersion === "scene-condition-preview-v2";
  const engineeringSummary = engineeringCombinationSummary(
    visiblePreview?.cases.length ?? expectedCounts?.cases ?? 0,
    visiblePreview?.cases[0],
  );
  const filteredEngineeringSummary = engineeringCombinationSummary(
    filteredCases.length,
    filteredCases[0],
  );
  const selectedCase =
    visiblePreview?.cases.find(
      (previewCase) => previewCase.case_id === selectedCaseId,
    ) ??
    visiblePreview?.cases[0] ??
    null;
  const renderContext = useMemo<ScopedCaseRenderContext>(
    () => ({
      source,
      datasetId,
      mode,
      scene,
      requestKey,
    }),
    [datasetId, mode, requestKey, scene, source],
  );
  const summary = useMemo(() => {
    const cases = visiblePreview?.cases ?? [];
    const hashes = new Set<string>();
    let speechGroups = 0;
    let cngGroups = 0;
    for (const previewCase of cases) {
      Object.values(previewCase.signals_sha256).forEach((hash) =>
        hashes.add(hash),
      );
      if (
        previewCase.families.some((family) => SPEECH_FAMILIES.has(family))
      ) {
        speechGroups += 1;
      }
      if (
        previewCase.families.some((family) => CNG_FAMILIES.has(family))
      ) {
        cngGroups += 1;
      }
    }
    return {
      groups: cases.length,
      uniqueInputs: hashes.size,
      speechGroups,
      cngGroups,
    };
  }, [visiblePreview?.cases]);
  const speechPercent =
    summary.groups > 0
      ? Math.round((summary.speechGroups / summary.groups) * 100)
      : 0;
  const cngPercent =
    summary.groups > 0
      ? Math.round((summary.cngGroups / summary.groups) * 100)
      : 0;

  function moveCasePage(offset: number) {
    const lastPageIndex = Math.max(0, casePage.pageCount - 1);
    const pageIndex = Math.min(
      lastPageIndex,
      Math.max(0, casePage.pageIndex + offset),
    );
    setPlayingTrack(null);
    setCasePageCursor({
      scopeKey: casePageScopeKey,
      pageIndex,
    });
  }

  if (!active) return null;

  return (
    <section
      className="sdv"
      aria-labelledby={titleId}
      aria-busy={previewLoading || loading}
    >
      <header className="sdv-header">
        <div className="sdv-heading">
          <span className="sdv-eyebrow">
            {conversationPreview
              ? `${expectedCounts?.cases ?? summary.groups} 种工程组合`
              : source === "scene-condition"
                ? "20 组测试数据"
                : "执行数据"}
          </span>
          <h2 id={titleId}>测试数据集</h2>
          <p className="sdv-context">
            <span>{BACKGROUND_NOISE_SCENE_LABELS[scene]}</span>
            {dataset ? (
              <>
                <span aria-hidden="true">·</span>
                <span>{dataset.name}</span>
              </>
            ) : null}
          </p>
        </div>
        <div className="sdv-header-actions">
          <button
            className="button secondary sdv-action"
            type="button"
            disabled={refreshDisabled}
            onClick={onRefresh}
          >
            <RefreshCw size={13} aria-hidden="true" />
            重新选择
          </button>
          <button
            className="button secondary sdv-action"
            type="button"
            onClick={onOpenWaveforms}
          >
            <ExternalLink size={13} aria-hidden="true" />
            查看完整波形
          </button>
        </div>
      </header>

      {!dataset ? (
        <div className="sdv-empty" role="status">
          <span>
            {loading ? "正在读取场景测试集…" : "暂无可用测试集"}
          </span>
        </div>
      ) : previewLoading ? (
        <PreviewSkeleton />
      ) : previewError ? (
        <p className="sdv-error" role="alert">
          <AlertCircle size={14} aria-hidden="true" />
          {previewError}
        </p>
      ) : visiblePreview && selectedCase ? (
        <div className="sdv-body">
          <dl className="sdv-stats" aria-label="当前场景测试集统计">
            <div>
              <dt>
                {conversationPreview
                  ? "工程组合"
                  : source === "scene-condition"
                    ? "样本组数"
                    : "实际条件组"}
              </dt>
              <dd>{summary.groups} 组</dd>
            </div>
            <div>
              <dt>合同测项</dt>
              <dd>{expectedItemCount} 项</dd>
            </div>
            <div>
              <dt>输入文件</dt>
              <dd>{summary.uniqueInputs} 个</dd>
            </div>
            <div>
              <dt>语音组</dt>
              <dd>{summary.speechGroups} 组</dd>
            </div>
            <div>
              <dt>CNG 组</dt>
              <dd>{summary.cngGroups} 组</dd>
            </div>
          </dl>

          <div
            className="sdv-distribution"
            role="group"
            aria-label={`语音组 ${summary.speechGroups} 组，占 ${speechPercent}%；CNG 组 ${summary.cngGroups} 组，占 ${cngPercent}%`}
          >
            <div className="sdv-distribution-bar" aria-hidden="true">
              <span
                className="is-speech"
                style={{ flexGrow: summary.speechGroups }}
              />
              <span
                className="is-cng"
                style={{ flexGrow: summary.cngGroups }}
              />
            </div>
            <div className="sdv-distribution-labels">
              <span>
                <i className="is-speech" aria-hidden="true" />
                语音组 {summary.speechGroups} · {speechPercent}%
              </span>
              <span>
                <i className="is-cng" aria-hidden="true" />
                CNG 组 {summary.cngGroups} · {cngPercent}%
              </span>
            </div>
          </div>

          {showConversationFilter ? (
            <div
              className="sdv-conversation-filter"
              role="group"
              aria-label="按通话状态筛选工程组合"
              data-testid="conversation-mode-filter"
            >
              <button
                type="button"
                className={conversationFilter === "all" ? "is-active" : ""}
                aria-pressed={conversationFilter === "all"}
                data-filter-mode="all"
                onClick={() => {
                  setPlayingTrack(null);
                  setConversationFilter("all");
                }}
              >
                <span>全部</span>
                <strong>{visiblePreview.cases.length}</strong>
              </button>
              {CONVERSATION_MODES.map((conversationMode) => (
                <button
                  key={conversationMode}
                  type="button"
                  className={
                    conversationFilter === conversationMode
                      ? "is-active"
                      : ""
                  }
                  aria-pressed={conversationFilter === conversationMode}
                  data-filter-mode={conversationMode}
                  title={conversationComposition(conversationMode)}
                  onClick={() => {
                    setPlayingTrack(null);
                    setConversationFilter(conversationMode);
                  }}
                >
                  <span>{CONVERSATION_MODE_LABELS[conversationMode]}</span>
                  <strong>{conversationCounts[conversationMode]}</strong>
                </button>
              ))}
            </div>
          ) : null}

          {source === "scene-condition" ? (
            <section
              className="sdv-case-list sdv-case-unit-shell"
              aria-labelledby={`${caseSelectId}-list-title`}
            >
              <header className="sdv-case-list-header">
                <div>
                  <span>TEST CASES</span>
                  <h3 id={`${caseSelectId}-list-title`}>
                    {conversationPreview
                      ? conversationFilter === "all"
                        ? `${visiblePreview.cases.length} 种工程组合`
                        : `${activeConversationLabel} · ${filteredCases.length} 种工程组合`
                      : "20 组测试样本"}
                  </h3>
                </div>
                <strong>
                  {conversationPreview
                    ? `共 ${filteredCases.length} 种`
                    : `全部 ${visiblePreview.cases.length} 组`}
                </strong>
              </header>
              {filteredCases.length > 0 ? (
                <>
                  <nav
                    className="sdv-case-pagination"
                    aria-label={`${casePageAriaLabel}分页`}
                  >
                    <button
                      type="button"
                      aria-controls={`${caseSelectId}-list-items`}
                      aria-label={`${casePageAriaLabel}上一页`}
                      disabled={casePage.pageIndex <= 0}
                      onClick={() => moveCasePage(-1)}
                    >
                      上一页
                    </button>
                    <span aria-live="polite">
                      {`第${casePage.pageNumber}/${casePage.pageCount}页 · 当前${casePage.start}–${casePage.end} / 共${casePage.total}种`}
                    </span>
                    <button
                      type="button"
                      aria-controls={`${caseSelectId}-list-items`}
                      aria-label={`${casePageAriaLabel}下一页`}
                      disabled={
                        casePage.pageIndex >= casePage.pageCount - 1
                      }
                      onClick={() => moveCasePage(1)}
                    >
                      下一页
                    </button>
                  </nav>
                  <ol
                    id={`${caseSelectId}-list-items`}
                    key={`${casePageScopeKey}:${casePage.pageIndex}`}
                    className="sdv-case-list-items sdv-case-unit-list"
                    role="list"
                  >
                    {casePage.items.map((previewCase) => (
                      <li key={previewCase.case_id}>
                        <WaveformUnit
                          previewCase={previewCase}
                          caseLabel={
                            caseLabels.get(previewCase.case_id) ??
                            "测试样本"
                          }
                          index={caseOrdinals.get(previewCase.case_id)}
                          datasetId={dataset.dataset_id}
                          visuals={visiblePreview.visuals}
                          source={source}
                          requestKey={requestKey}
                          playingTrack={playingTrack}
                          waveformA11yId={waveformA11yId}
                          variant="list"
                          renderCaseActions={renderCaseActions}
                          renderCaseResult={renderCaseResult}
                          renderContext={renderContext}
                          onPlay={setPlayingTrack}
                          onOpenWaveforms={onOpenWaveforms}
                        />
                      </li>
                    ))}
                  </ol>
                </>
              ) : (
                <div className="sdv-filter-empty" role="status">
                  <span>当前筛选没有工程组合</span>
                  <button
                    type="button"
                    onClick={() => {
                      setPlayingTrack(null);
                      setConversationFilter("all");
                    }}
                  >
                    查看全部
                  </button>
                </div>
              )}
              <p className="sdv-case-list-hint">
                {conversationPreview
                  ? `${filteredEngineeringSummary}分页查看。`
                  : `每组是一个独立单元，分页查看全部 ${visiblePreview.cases.length} 组四路输入。`}
              </p>
            </section>
          ) : (
            <div className="sdv-case-browser">
              <label className="sdv-case-select" htmlFor={caseSelectId}>
                <span>具体样本</span>
                <select
                  id={caseSelectId}
                  value={selectedCase.case_id}
                  onChange={(event) => {
                    setPlayingTrack(null);
                    setSelectedCaseId(event.target.value);
                  }}
                >
                  {visiblePreview.cases.map((previewCase) => (
                    <option
                      key={previewCase.case_id}
                      value={previewCase.case_id}
                    >
                      {caseLabels.get(previewCase.case_id) ??
                        "测试样本"}
                    </option>
                  ))}
                </select>
              </label>

              <div className="sdv-case-detail">
                <div className="sdv-case-context">
                  <span className="sdv-case-context-title">
                    {caseLabels.get(selectedCase.case_id) ?? "测试样本"}
                  </span>
                  <span>适用合同：{humanFamilyLabels(selectedCase)}</span>
                  <small title={selectedCase.case_id}>
                    {selectedCase.case_id}
                  </small>
                </div>
                <WaveformUnit
                  previewCase={selectedCase}
                  caseLabel={
                    caseLabels.get(selectedCase.case_id) ?? "测试样本"
                  }
                  datasetId={dataset.dataset_id}
                  visuals={visiblePreview.visuals}
                  source={source}
                  requestKey={requestKey}
                  playingTrack={playingTrack}
                  waveformA11yId={waveformA11yId}
                  variant="detail"
                  renderContext={renderContext}
                  onPlay={setPlayingTrack}
                  onOpenWaveforms={onOpenWaveforms}
                />
              </div>
            </div>
          )}

          <footer className="sdv-footer">
            <span>工程测试数据 · 正式 ITU-T 合规未测</span>
            {source === "scene-condition" ? (
              <span>
                {conversationPreview && expectedCounts
                  ? `${engineeringSummary}共 ${expectedCounts.observations} 次指标测量。`
                  : "20 组样本全部参与测试 · 共 50 次指标测量"}
              </span>
            ) : null}
            {extendedPreviewCount !== undefined ? (
              <span className="sdv-extended">
                <button
                  type="button"
                  onClick={onOpenExtendedWaveforms ?? onOpenWaveforms}
                >
                  另有 {extendedPreviewCount} 组扩展样本预览
                </button>
                <span>不参与本次批次</span>
              </span>
            ) : null}
          </footer>
        </div>
      ) : (
        <div className="sdv-empty" role="status">
          <span>暂无可用测试集</span>
        </div>
      )}
    </section>
  );
}
