import {
  useEffect,
  useId,
  useMemo,
  useState,
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
  readSceneConditionPreview,
  type BackgroundNoiseScene,
  type ScopedBackgroundNoisePreview,
  type ScopedDataset,
  type ScopedInputPreviewCase,
  type ScopedInputSignalRole,
  type ScopedPreviewVisualTrack,
} from "./scoped-acoustic-model";
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
};

type CaseCategory = "speech" | "cng" | "mixed" | "other";

type PreviewState = {
  key: string;
  preview: ScopedBackgroundNoisePreview | null;
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
  visual: ScopedPreviewVisualTrack | null;
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
  preview: ScopedBackgroundNoisePreview | null,
  datasetId: string,
  mode: FiveSignalMode,
  scene: BackgroundNoiseScene,
): preview is ScopedBackgroundNoisePreview {
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

function buildCaseLabels(
  cases: ScopedInputPreviewCase[],
): Map<string, string> {
  const counters: Record<CaseCategory, number> = {
    speech: 0,
    cng: 0,
    mixed: 0,
    other: 0,
  };
  const labels = new Map<string, string>();
  for (const previewCase of cases) {
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
  visual: ScopedPreviewVisualTrack,
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
  visuals: ScopedBackgroundNoisePreview["visuals"] | undefined;
  source: "execution" | "scene-condition";
  requestKey: string;
  playingTrack: PlayingTrack | null;
  waveformA11yId: string;
  variant: "list" | "detail";
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
  const category = caseCategory(previewCase);
  const caseA11yId = `${waveformA11yId}-${previewCase.case_id}`;

  return (
    <article className={`sdv-waveform-unit is-${variant}`}>
      {variant === "list" ? (
        <header className="sdv-waveform-unit-header">
          <div className="sdv-waveform-unit-identity">
            {index !== undefined ? (
              <span className="sdv-waveform-unit-index">
                {String(index).padStart(2, "0")}
              </span>
            ) : null}
            <div className="sdv-waveform-unit-copy">
              <div className="sdv-waveform-unit-title">
                <strong>{caseLabel}</strong>
                <span className={`sdv-case-list-kind is-${category}`}>
                  {caseCategoryShortLabel(category)}
                </span>
              </div>
              <small title={previewCase.case_id}>
                {previewCase.case_id}
              </small>
            </div>
          </div>
          <span className="sdv-waveform-unit-contract">
            适用合同：{humanFamilyLabels(previewCase)}
          </span>
        </header>
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
                {role === "far_end" && visual?.peak === 0 ? (
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

      {farEndVisual ? (
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

  useEffect(() => {
    setPlayingTrack(null);
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
    if (source === "scene-condition") params.set("scene", scene);
    const url = source === "scene-condition"
      ? `${SCENE_PREVIEW_URL}?${params.toString()}`
      : `${PREVIEW_URL}/${encodeURIComponent(scene)}/preview?${params.toString()}`;
    void requestPreview(
      url,
      controller.signal,
    )
      .then((body) =>
        (source === "scene-condition" ? readSceneConditionPreview : readScopedBackgroundNoisePreview)(
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
  const selectedCase =
    visiblePreview?.cases.find(
      (previewCase) => previewCase.case_id === selectedCaseId,
    ) ??
    visiblePreview?.cases[0] ??
    null;
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

  if (!active) return null;

  return (
    <section
      className="sdv"
      aria-labelledby={titleId}
      aria-busy={previewLoading || loading}
    >
      <header className="sdv-header">
        <div className="sdv-heading">
          <span className="sdv-eyebrow">{source === "scene-condition" ? "20 组测试数据" : "执行数据"}</span>
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
              <dt>{source === "scene-condition" ? "样本组数" : "实际条件组"}</dt>
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

          {source === "scene-condition" ? (
            <section
              className="sdv-case-list sdv-case-unit-shell"
              aria-labelledby={`${caseSelectId}-list-title`}
            >
              <header className="sdv-case-list-header">
                <div>
                  <span>TEST CASES</span>
                  <h3 id={`${caseSelectId}-list-title`}>20 组测试样本</h3>
                </div>
                <strong>全部 {visiblePreview.cases.length} 组</strong>
              </header>
              <ol
                className="sdv-case-list-items sdv-case-unit-list"
                role="list"
              >
                {visiblePreview.cases.map((previewCase, index) => (
                  <li key={previewCase.case_id}>
                    <WaveformUnit
                      previewCase={previewCase}
                      caseLabel={
                        caseLabels.get(previewCase.case_id) ?? "测试样本"
                      }
                      index={index + 1}
                      datasetId={dataset.dataset_id}
                      visuals={visiblePreview.visuals}
                      source={source}
                      requestKey={requestKey}
                      playingTrack={playingTrack}
                      waveformA11yId={waveformA11yId}
                      variant="list"
                      onPlay={setPlayingTrack}
                      onOpenWaveforms={onOpenWaveforms}
                    />
                  </li>
                ))}
              </ol>
              <p className="sdv-case-list-hint">
                每组是一个独立单元，滚动查看全部{" "}
                {visiblePreview.cases.length} 组四路输入
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
                  onPlay={setPlayingTrack}
                  onOpenWaveforms={onOpenWaveforms}
                />
              </div>
            </div>
          )}

          <footer className="sdv-footer">
            <span>工程测试数据 · 正式 ITU-T 合规未测</span>
            {source === "scene-condition" ? (
              <span>20 组样本全部参与测试 · 共 50 次指标测量</span>
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
