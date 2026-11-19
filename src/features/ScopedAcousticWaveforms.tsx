import {
  memo,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  Activity,
  AlertCircle,
  Download,
  Play,
  RefreshCw,
} from "lucide-react";
import {
  FIVE_SIGNAL_MODES,
  type FiveSignalMode,
} from "./five-signal-flow-model";
import {
  effectiveMicrophoneMatchesDynamic,
  sceneConditionCaseDynamicMicrophoneEvidence,
  sceneConditionCaseRxEvidence,
  sceneConditionCaseTxEvidence,
  SCENE_CONDITION_BATCHES_URL,
  type SceneConditionBatch,
  type SceneConditionCase,
} from "./scene-condition-test-model";
import {
  isSceneCondition20Preview,
  matchingScopedSnapshot,
  readScopedBackgroundNoisePreview,
  readSceneConditionDatasetView,
  readSceneConditionPreview,
  scopedCaseEvidenceIdentity,
  SCENE_CONDITION_SCHEMA_VERSION,
  SCOPED_INPUT_SIGNAL_ROLES,
  SCOPED_PREVIEW_VISUAL_SCHEMA,
  type BackgroundNoiseScene,
  type ScopedBackgroundNoisePreview,
  type ScopedInputPreviewCase,
  type ScopedInputSignalRole,
  type ScopedPreviewVisualTrack,
  type ScopedSceneConditionPreview,
  type ScopedSnapshot,
  type ScopedTestDefinition,
} from "./scoped-acoustic-model";
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
const SCENE_CONDITION_DATASETS_URL =
  "/api/alignment/acceptance/scene-condition-datasets";
const SCENE_CONDITION_PREVIEW_URL =
  "/api/alignment/acceptance/scene-condition-preview";
const DATASET_URL = "/api/alignment/acceptance/datasets";
const BATCH_URL = "/api/alignment/acceptance/batches";
const WAVEFORM_BINS = 720;
const WAVEFORM_WIDTH = 720;
const WAVEFORM_HEIGHT = 150;
const MEL_BANDS = 40;
const MEL_FFT_SIZE = 512;
const MEL_MAX_FRAMES = 240;
const MEL_WIDTH = 720;
const MEL_HEIGHT = 96;
const MEL_DYNAMIC_RANGE_DB = 80;
const MEL_DB_OFFSET_DB = -200;
const MEL_DB_STEP_DB = 0.1;
const INPUT_SIGNAL_LABELS: Record<ScopedInputSignalRole, string> = {
  microphone: "麦克风混合输入 · microphone",
  far_end: "远端播放信号 / 回声参考 · far_end",
  reference: "干净近端参考 · reference",
  noise: "独立噪声输入 · noise",
};
const DYNAMIC_MICROPHONE_LABEL =
  "麦克风混合输入 · 运行时 post-RX 合成";

const SIGNAL_COLOR = "#4f7fb8";

type Props = {
  definition: ScopedTestDefinition;
  datasetId: string;
  mode: FiveSignalMode;
  algorithmClusterId: string;
  snapshot: ScopedSnapshot | null;
  active: boolean;
  previewSource?: "execution" | "extended";
  sceneConditionBatch?: SceneConditionBatch | null;
  sceneConditionCasesById?: ReadonlyMap<
    string,
    { batch: SceneConditionBatch; caseRecord: SceneConditionCase }
  >;
};

type ConversationFilter = "all" | ConversationMode;

type PreviewSegment = {
  start_s: number;
  end_s: number;
  label: string;
};

type WaveformEnvelope = {
  url: string;
  min: number[];
  max: number[];
  peak: number;
  durationSeconds: number;
  sampleRate: number;
  channels: number;
  mel: MelSpectrogram | null;
};

type MelSpectrogram = {
  bands: number;
  frames: number;
  values: Float32Array;
  maxDb: number;
};

type WavFormat = {
  sampleRate: number;
  channels: number;
};

function hexToRgb(hex: string): [number, number, number] {
  const value = Number.parseInt(hex.slice(1), 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

const MEL_ACCENT = hexToRgb(SIGNAL_COLOR);

function errorMessage(reason: unknown, fallback: string): string {
  return reason instanceof Error && reason.message
    ? reason.message
    : fallback;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
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
    }；执行时由 RX 后远端、近端语音与背景噪声实时合成 microphone 及波形。`;
  }
  if (mode === "far_single") {
    return `far_end 承载 VCP RX 前回声参考；执行时先生成 RX 后参考，再${echoPhrase}与背景噪声实时合成 microphone 波形并送入 TX；${
      nearSilent
        ? "reference 为数字静音"
        : "reference 关系以 provenance 为准"
    }。`;
  }
  if (mode === "double_talk") {
    return `reference 与 far_end 同时活动；执行时先生成 RX 后参考，再把${echoPhrase}的远端成分、近端语音与背景噪声实时合成 microphone 波形并送入 TX。`;
  }
  if (mode === "transition") {
    return `reference 与 far_end 按 provenance 分段切换远端单讲、近端单讲、双讲和噪声尾段；执行时先生成 RX 后参考，再按远端活动区段${echoPhrase}实时合成 microphone 波形并送入 TX。`;
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

function decodeBase64Bytes(encoded: string): Uint8Array {
  const binary = window.atob(encoded);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

function decodeInt16LittleEndian(bytes: Uint8Array): Int16Array {
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

function visualizationToWaveform(
  url: string,
  visual: ScopedPreviewVisualTrack,
): WaveformEnvelope {
  const minimum = decodeInt16LittleEndian(
    decodeBase64Bytes(visual.wave_min_i16),
  );
  const maximum = decodeInt16LittleEndian(
    decodeBase64Bytes(visual.wave_max_i16),
  );
  const melValues = decodeInt16LittleEndian(
    decodeBase64Bytes(visual.mel_i16),
  );
  if (
    minimum.length !== WAVEFORM_BINS
    || maximum.length !== WAVEFORM_BINS
    || melValues.length !== MEL_BANDS * MEL_MAX_FRAMES
  ) {
    throw new Error("预计算波形或梅尔频谱数组长度无效");
  }
  const scale = visual.wave_scale / 32767;
  const mel = new Float32Array(melValues.length);
  for (let index = 0; index < melValues.length; index += 1) {
    mel[index] = melValues[index] * MEL_DB_STEP_DB + MEL_DB_OFFSET_DB;
  }
  return {
    url,
    min: Array.from(minimum, value => value * scale),
    max: Array.from(maximum, value => value * scale),
    peak: visual.peak,
    durationSeconds: visual.duration_s,
    sampleRate: visual.sample_rate,
    channels: visual.channels,
    mel: {
      bands: MEL_BANDS,
      frames: MEL_MAX_FRAMES,
      values: mel,
      maxDb: visual.mel_ceiling_db,
    },
  };
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

function inputUrl(
  datasetId: string,
  caseId: string,
  role: ScopedInputSignalRole,
  version?: string,
): string {
  const suffix = version ? `?v=${encodeURIComponent(version)}` : "";
  return `${DATASET_URL}/${encodeURIComponent(
    datasetId,
  )}/cases/${encodeURIComponent(caseId)}/inputs/${encodeURIComponent(role)}${suffix}`;
}

function acceptanceDynamicMicrophoneUrl(
  batchId: string,
  caseId: string,
  version: string,
): string {
  return `${BATCH_URL}/${encodeURIComponent(
    batchId,
  )}/cases/${encodeURIComponent(caseId)}/inputs/dynamic-microphone?v=${encodeURIComponent(
    version,
  )}`;
}

function sceneConditionInputUrl(
  datasetId: string,
  caseId: string,
  role: ScopedInputSignalRole,
  version?: string,
): string {
  const suffix = version ? `?v=${encodeURIComponent(version)}` : "";
  return `${SCENE_CONDITION_DATASETS_URL}/${encodeURIComponent(
    datasetId,
  )}/cases/${encodeURIComponent(caseId)}/inputs/${encodeURIComponent(role)}${suffix}`;
}

function outputUrl(
  batchId: string,
  caseId: string,
  version?: string,
): string {
  const suffix = version ? `?v=${encodeURIComponent(version)}` : "";
  return `${BATCH_URL}/${encodeURIComponent(
    batchId,
  )}/cases/${encodeURIComponent(caseId)}/outputs/tx${suffix}`;
}

async function requestSceneConditionJson(
  url: string,
  signal: AbortSignal,
  operation: string,
  cache: RequestCache,
): Promise<unknown> {
  const response = await fetch(url, { cache, signal });
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
    const detail =
      typeof body === "object"
      && body !== null
      && !Array.isArray(body)
      && typeof (body as { detail?: unknown }).detail === "string"
        ? (body as { detail: string }).detail
        : "";
    throw new Error(
      detail || `${operation}失败（HTTP ${response.status}）`,
    );
  }
  return body;
}

function isSceneConditionPreview(
  preview: ScopedBackgroundNoisePreview | null,
  scene: BackgroundNoiseScene,
): preview is ScopedSceneConditionPreview {
  if (!preview) return false;
  const profile = sceneConditionProfileForDatasetId(preview.dataset_id);
  if (!profile || profile.scene !== scene) return false;
  const expectedCounts = sceneConditionCounts(profile);
  const expectedSchema =
    profile.schemaVersion ?? SCENE_CONDITION_SCHEMA_VERSION;
  const candidate = preview as Partial<ScopedSceneConditionPreview>;
  const visualCases = candidate.visuals?.cases;
  if (
    candidate.preview_condition_group_count
      !== expectedCounts.cases
    || candidate.contract_item_count !== 5
    || candidate.standard_compliance !== false
    || candidate.formal_test_status !== "not_measured"
    || candidate.schema_version !== expectedSchema
    || candidate.dataset_kind !== profile.datasetKind
    || candidate.cases?.length !== expectedCounts.cases
    || !visualCases
    || Object.keys(visualCases).length
      !== expectedCounts.cases
    || Object.values(visualCases).some(
      (tracks) =>
        !tracks
        || Object.keys(tracks).length !== SCOPED_INPUT_SIGNAL_ROLES.length
        || SCOPED_INPUT_SIGNAL_ROLES.some((role) => !(role in tracks)),
    )
  ) {
    return false;
  }
  return true;
}

function readWavFormat(bytes: ArrayBuffer): WavFormat {
  if (bytes.byteLength < 44) {
    throw new Error("WAV 文件头不完整");
  }
  const view = new DataView(bytes);
  const fourCc = (offset: number) =>
    String.fromCharCode(
      view.getUint8(offset),
      view.getUint8(offset + 1),
      view.getUint8(offset + 2),
      view.getUint8(offset + 3),
    );
  if (fourCc(0) !== "RIFF" || fourCc(8) !== "WAVE") {
    throw new Error("波形接口没有返回 WAV 音频");
  }

  let offset = 12;
  while (offset + 8 <= bytes.byteLength) {
    const chunkId = fourCc(offset);
    const chunkSize = view.getUint32(offset + 4, true);
    const dataOffset = offset + 8;
    if (dataOffset + chunkSize > bytes.byteLength) {
      throw new Error("WAV 数据块越界");
    }
    if (chunkId === "fmt ") {
      if (chunkSize < 16) {
        throw new Error("WAV fmt 数据块不完整");
      }
      const channels = view.getUint16(dataOffset + 2, true);
      const sampleRate = view.getUint32(dataOffset + 4, true);
      if (channels < 1 || sampleRate < 8_000 || sampleRate > 384_000) {
        throw new Error("WAV 声道或采样率无效");
      }
      return { sampleRate, channels };
    }
    offset = dataOffset + chunkSize + (chunkSize % 2);
  }
  throw new Error("WAV 缺少 fmt 数据块");
}

function hzToMel(hz: number): number {
  return 2595 * Math.log10(1 + hz / 700);
}

function melToHz(mel: number): number {
  return 700 * (10 ** (mel / 2595) - 1);
}

function fft(real: Float32Array, imaginary: Float32Array): void {
  const size = real.length;
  for (let index = 1, reversed = 0; index < size; index += 1) {
    let bit = size >> 1;
    for (; reversed & bit; bit >>= 1) reversed ^= bit;
    reversed ^= bit;
    if (index < reversed) {
      [real[index], real[reversed]] = [real[reversed], real[index]];
      [imaginary[index], imaginary[reversed]] = [
        imaginary[reversed],
        imaginary[index],
      ];
    }
  }

  for (let length = 2; length <= size; length <<= 1) {
    const angle = (-2 * Math.PI) / length;
    const stepReal = Math.cos(angle);
    const stepImaginary = Math.sin(angle);
    for (let start = 0; start < size; start += length) {
      let twiddleReal = 1;
      let twiddleImaginary = 0;
      for (let offset = 0; offset < length / 2; offset += 1) {
        const even = start + offset;
        const odd = even + length / 2;
        const oddReal =
          real[odd] * twiddleReal - imaginary[odd] * twiddleImaginary;
        const oddImaginary =
          real[odd] * twiddleImaginary + imaginary[odd] * twiddleReal;
        real[odd] = real[even] - oddReal;
        imaginary[odd] = imaginary[even] - oddImaginary;
        real[even] += oddReal;
        imaginary[even] += oddImaginary;
        const nextTwiddleReal =
          twiddleReal * stepReal - twiddleImaginary * stepImaginary;
        twiddleImaginary =
          twiddleReal * stepImaginary + twiddleImaginary * stepReal;
        twiddleReal = nextTwiddleReal;
      }
    }
  }
}

function buildMelBank(sampleRate: number): Array<Array<{ bin: number; weight: number }>> {
  const maximumHz = Math.min(8_000, sampleRate / 2);
  const melStart = hzToMel(20);
  const melEnd = hzToMel(maximumHz);
  const edges = Array.from({ length: MEL_BANDS + 2 }, (_, index) =>
    melToHz(melStart + ((melEnd - melStart) * index) / (MEL_BANDS + 1)),
  );
  const binHz = sampleRate / MEL_FFT_SIZE;
  return Array.from({ length: MEL_BANDS }, (_, band) => {
    const left = edges[band];
    const center = edges[band + 1];
    const right = edges[band + 2];
    const weights: Array<{ bin: number; weight: number }> = [];
    for (let bin = 1; bin <= MEL_FFT_SIZE / 2; bin += 1) {
      const hz = bin * binHz;
      if (hz < left || hz > right) continue;
      const weight =
        hz <= center
          ? (hz - left) / Math.max(center - left, Number.EPSILON)
          : (right - hz) / Math.max(right - center, Number.EPSILON);
      if (weight > 0) weights.push({ bin, weight });
    }
    return weights;
  });
}

function buildMelSpectrogram(
  audio: AudioBuffer,
  sampleRate: number,
): MelSpectrogram {
  const hop = Math.max(1, Math.round(sampleRate * 0.01));
  const frameCount = Math.max(1, Math.ceil(audio.length / hop));
  const melBank = buildMelBank(sampleRate);
  const window = new Float32Array(MEL_FFT_SIZE);
  const real = new Float32Array(MEL_FFT_SIZE);
  const imaginary = new Float32Array(MEL_FFT_SIZE);
  let windowEnergy = 0;
  for (let index = 0; index < MEL_FFT_SIZE; index += 1) {
    window[index] =
      0.5 - 0.5 * Math.cos((2 * Math.PI * index) / (MEL_FFT_SIZE - 1));
    windowEnergy += window[index] * window[index];
  }
  const channelCount = Math.max(1, audio.numberOfChannels);
  const channels = Array.from({ length: channelCount }, (_, channel) =>
    audio.getChannelData(channel),
  );
  const values = new Float32Array(MEL_BANDS * frameCount);
  let maximumDb = -Infinity;

  for (let frame = 0; frame < frameCount; frame += 1) {
    const start = frame * hop;
    for (let index = 0; index < MEL_FFT_SIZE; index += 1) {
      const sampleIndex = start + index;
      let mixed = 0;
      for (const channel of channels) {
        if (sampleIndex < channel.length) mixed += channel[sampleIndex];
      }
      real[index] = (mixed / channelCount) * window[index];
      imaginary[index] = 0;
    }
    fft(real, imaginary);
    for (let band = 0; band < MEL_BANDS; band += 1) {
      let power = 0;
      for (const { bin, weight } of melBank[band]) {
        power += (real[bin] ** 2 + imaginary[bin] ** 2) * weight;
      }
      const db = 10 * Math.log10(Math.max(power / windowEnergy, 1e-20));
      values[band * frameCount + frame] = db;
      maximumDb = Math.max(maximumDb, db);
    }
  }

  return {
    bands: MEL_BANDS,
    frames: frameCount,
    values,
    maxDb: Number.isFinite(maximumDb) ? maximumDb : -120,
  };
}

function buildEnvelope(
  url: string,
  audio: AudioBuffer,
  format: WavFormat,
): WaveformEnvelope {
  const binCount = Math.min(WAVEFORM_BINS, audio.length);
  const minimum = new Array<number>(binCount).fill(0);
  const maximum = new Array<number>(binCount).fill(0);
  const filled = new Uint8Array(binCount);

  for (let channelIndex = 0; channelIndex < audio.numberOfChannels; channelIndex += 1) {
    const samples = audio.getChannelData(channelIndex);
    for (let frame = 0; frame < samples.length; frame += 1) {
      const bin = Math.min(
        binCount - 1,
        Math.floor((frame * binCount) / samples.length),
      );
      const sample = samples[frame];
      if (!filled[bin]) {
        minimum[bin] = sample;
        maximum[bin] = sample;
        filled[bin] = 1;
      } else {
        minimum[bin] = Math.min(minimum[bin], sample);
        maximum[bin] = Math.max(maximum[bin], sample);
      }
    }
  }

  let peak = 0;
  for (let index = 0; index < binCount; index += 1) {
    peak = Math.max(
      peak,
      Math.abs(minimum[index]),
      Math.abs(maximum[index]),
    );
  }
  return {
    url,
    min: minimum,
    max: maximum,
    peak,
    durationSeconds: audio.duration,
    sampleRate: format.sampleRate,
    channels: format.channels,
    mel: null,
  };
}

async function decodeWaveform(
  url: string,
  signal: AbortSignal,
): Promise<{ envelope: WaveformEnvelope; audio: AudioBuffer }> {
  // Input hashes and batch IDs are part of the URL, so audio is immutable per version.
  const response = await fetch(url, { cache: "force-cache", signal });
  if (!response.ok) {
    throw new Error(`波形音频读取失败（HTTP ${response.status}）`);
  }
  const bytes = await response.arrayBuffer();
  if (bytes.byteLength < 12) {
    throw new Error("波形音频为空或格式无效");
  }
  const format = readWavFormat(bytes);
  const decodeRate =
    format.sampleRate >= 8_000 && format.sampleRate <= 96_000
      ? format.sampleRate
      : 48_000;
  const context = new OfflineAudioContext(1, 1, decodeRate);
  const audio = await context.decodeAudioData(bytes.slice(0));
  return { envelope: buildEnvelope(url, audio, format), audio };
}

function useWaveform(
  url: string,
  refreshToken: number,
  visual?: ScopedPreviewVisualTrack,
) {
  const precomputed = useMemo(
    () => visual ? visualizationToWaveform(url, visual) : null,
    [url, visual],
  );
  const [waveform, setWaveform] = useState<WaveformEnvelope | null>(
    precomputed,
  );
  const [loading, setLoading] = useState(Boolean(url) && !precomputed);
  const [melLoading, setMelLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!url) {
      setLoading(false);
      setMelLoading(false);
      setError("");
      return;
    }
    if (precomputed) {
      setWaveform(precomputed);
      setLoading(false);
      setMelLoading(false);
      setError("");
      return;
    }
    const controller = new AbortController();
    let disposed = false;
    setLoading(true);
    setMelLoading(false);
    setError("");

    decodeWaveform(url, controller.signal)
      .then(({ envelope, audio }) => {
        if (!disposed && !controller.signal.aborted) {
          setWaveform(envelope);
          setMelLoading(true);
          requestAnimationFrame(() => {
            const computeMel = () => {
              if (disposed || controller.signal.aborted) return;
              try {
                const mel = buildMelSpectrogram(audio, audio.sampleRate);
                setWaveform((current) =>
                  current?.url === url ? { ...current, mel } : current,
                );
              } catch {
                setError("梅尔频谱计算失败；波形仍可查看");
              } finally {
                if (!disposed && !controller.signal.aborted) {
                  setMelLoading(false);
                }
              }
            };
            if (typeof window.requestIdleCallback === "function") {
              window.requestIdleCallback(computeMel, { timeout: 500 });
            } else {
              window.setTimeout(computeMel, 0);
            }
          });
        }
      })
      .catch((reason: unknown) => {
        if (!disposed && !controller.signal.aborted) {
          setError(errorMessage(reason, "波形解码失败"));
        }
      })
      .finally(() => {
        if (!disposed && !controller.signal.aborted) {
          setLoading(false);
        }
      });

    return () => {
      disposed = true;
      controller.abort();
    };
  }, [precomputed, refreshToken, url]);

  return {
    waveform: precomputed ?? (waveform?.url === url ? waveform : null),
    loading,
    melLoading,
    error,
  };
}

function waveformPath(
  envelope: WaveformEnvelope,
  peakScale: number,
): string {
  if (envelope.max.length === 0) return "";
  const usableHeight = WAVEFORM_HEIGHT - 18;
  const center = WAVEFORM_HEIGHT / 2;
  const scale = peakScale > 0 ? 1 / peakScale : 1;
  const x = (index: number) =>
    (index / Math.max(1, envelope.max.length - 1)) * WAVEFORM_WIDTH;
  const top = envelope.max.map(
    (value, index) =>
      `${x(index).toFixed(2)} ${(
        center -
        value * scale * usableHeight * 0.5
      ).toFixed(2)}`,
  );
  const bottom = [...envelope.min].reverse().map((value, reverseIndex) => {
    const index = envelope.min.length - 1 - reverseIndex;
    return `${x(index).toFixed(2)} ${(
      center -
      value * scale * usableHeight * 0.5
    ).toFixed(2)}`;
  });
  return `M ${top.join(" L ")} L ${bottom.join(" L ")} Z`;
}

function WaveformSvg({
  label,
  caseId,
  envelope,
  peakScale,
  color,
}: {
  label: string;
  caseId: string;
  envelope: WaveformEnvelope;
  peakScale: number;
  color: string;
}) {
  const uid = useId().replace(/[^A-Za-z0-9_-]/g, "");
  const titleId = `waveform-${uid}-title`;
  const descriptionId = `waveform-${uid}-description`;
  const path = waveformPath(envelope, peakScale);
  return (
    <svg
      className="scoped-waveform-svg"
      viewBox={`0 0 ${WAVEFORM_WIDTH} ${WAVEFORM_HEIGHT}`}
      role="img"
      aria-labelledby={`${titleId} ${descriptionId}`}
      preserveAspectRatio="none"
    >
      <title id={titleId}>{caseId} · {label}</title>
      <desc id={descriptionId}>
        {`${label}，时长 ${formatDuration(
          envelope.durationSeconds,
        )}，采样率 ${formatSampleRate(
          envelope.sampleRate,
        )}，${envelope.channels} 声道。纵轴显示相对峰值包络。`}
      </desc>
      <line
        className="scoped-waveform-zero"
        x1="0"
        y1={WAVEFORM_HEIGHT / 2}
        x2={WAVEFORM_WIDTH}
        y2={WAVEFORM_HEIGHT / 2}
      />
      {path ? (
        <path
          className="scoped-waveform-envelope"
          d={path}
          style={{ fill: color }}
        />
      ) : null}
    </svg>
  );
}

function drawMelSpectrogram(
  canvas: HTMLCanvasElement,
  mel: MelSpectrogram,
  ceilingDb: number,
  accent: [number, number, number],
): void {
  const context = canvas.getContext("2d");
  if (!context) return;
  const image = context.createImageData(MEL_WIDTH, MEL_HEIGHT);
  const frameColumns = Math.min(MEL_MAX_FRAMES, mel.frames);
  const floorDb = ceilingDb - MEL_DYNAMIC_RANGE_DB;
  const columnValues = new Float32Array(MEL_BANDS * frameColumns);
  for (let column = 0; column < frameColumns; column += 1) {
    const start = Math.floor((column * mel.frames) / frameColumns);
    const end = Math.max(
      start + 1,
      Math.floor(((column + 1) * mel.frames) / frameColumns),
    );
    for (let band = 0; band < MEL_BANDS; band += 1) {
      let peak = -Infinity;
      for (let frame = start; frame < end; frame += 1) {
        peak = Math.max(peak, mel.values[band * mel.frames + frame]);
      }
      columnValues[band * frameColumns + column] = peak;
    }
  }

  for (let y = 0; y < MEL_HEIGHT; y += 1) {
    const band = Math.min(
      MEL_BANDS - 1,
      Math.floor((1 - (y + 0.5) / MEL_HEIGHT) * MEL_BANDS),
    );
    for (let x = 0; x < MEL_WIDTH; x += 1) {
      const column = Math.min(
        frameColumns - 1,
        Math.floor(((x + 0.5) / MEL_WIDTH) * frameColumns),
      );
      const value = columnValues[band * frameColumns + column];
      const normalized = Math.max(
        0,
        Math.min(1, (value - floorDb) / MEL_DYNAMIC_RANGE_DB),
      );
      const alpha = 0.04 + 0.96 * normalized ** 0.82;
      const pixel = (y * MEL_WIDTH + x) * 4;
      image.data[pixel] = Math.round(255 * (1 - alpha) + accent[0] * alpha);
      image.data[pixel + 1] = Math.round(255 * (1 - alpha) + accent[1] * alpha);
      image.data[pixel + 2] = Math.round(255 * (1 - alpha) + accent[2] * alpha);
      image.data[pixel + 3] = 255;
    }
  }
  context.putImageData(image, 0, 0);
}

function MelSpectrogramView({
  label,
  caseId,
  mel,
  ceilingDb,
  accent,
  maximumHz,
}: {
  label: string;
  caseId: string;
  mel: MelSpectrogram;
  ceilingDb: number;
  accent: [number, number, number];
  maximumHz: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const uid = useId().replace(/[^A-Za-z0-9_-]/g, "");
  const titleId = `mel-${uid}-title`;
  const descriptionId = `mel-${uid}-description`;
  const accentCss = `rgb(${accent.join(", ")})`;
  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas) drawMelSpectrogram(canvas, mel, ceilingDb, accent);
  }, [accent, ceilingDb, mel]);

  return (
    <div
      className="scoped-mel-spectrogram"
      role="img"
      aria-labelledby={`${titleId} ${descriptionId}`}
      style={{ borderTopColor: accentCss }}
    >
      <div className="scoped-mel-heading">
        <span style={{ color: accentCss }}>MEL SPECTROGRAM</span>
        <span>{MEL_BANDS} 频带 · {MEL_DYNAMIC_RANGE_DB} dB</span>
      </div>
      <canvas
        ref={canvasRef}
        className="scoped-mel-canvas"
        width={MEL_WIDTH}
        height={MEL_HEIGHT}
        aria-hidden="true"
      />
      <span id={titleId} className="sr-only">
        {caseId} · {label} 梅尔频谱图
      </span>
      <span id={descriptionId} className="sr-only">
        多声道按等权混合，显示 20 Hz 至 {formatSampleRate(maximumHz)}
        频段的 40 个梅尔频带；五条轨道使用同一 {MEL_DYNAMIC_RANGE_DB} dB 色标。
      </span>
    </div>
  );
}

export function PrecomputedAudioFigures({
  visual, label, caseId, peakScale, melCeilingDb, showMel = true,
}: {
  visual: ScopedPreviewVisualTrack;
  label: string;
  caseId: string;
  peakScale: number;
  melCeilingDb: number;
  showMel?: boolean;
}) {
  const waveform = useMemo(() => visualizationToWaveform("", visual), [visual]);
  return <>
    <WaveformSvg label={label} caseId={caseId} envelope={waveform}
      peakScale={peakScale} color={SIGNAL_COLOR} />
    {showMel && waveform.mel ? <MelSpectrogramView label={label} caseId={caseId}
      mel={waveform.mel} ceilingDb={melCeilingDb} accent={[79, 127, 184]}
      maximumHz={Math.min(20_000, visual.sample_rate / 2)} /> : null}
  </>;
}

function WaveformTrack({
  label,
  caseId,
  url,
  kind,
  signalRole,
  peakScale,
  unavailable,
  waveform,
  melCeilingDb,
  melLoading,
  loading,
  error,
  playbackEpoch,
}: {
  label: string;
  caseId: string;
  url: string;
  kind: "input" | "output";
  signalRole: ScopedInputSignalRole | "rx" | "tx";
  peakScale: number;
  unavailable?: string;
  waveform: WaveformEnvelope | null;
  melCeilingDb: number;
  melLoading: boolean;
  loading: boolean;
  error: string;
  playbackEpoch: number;
}) {
  const [playbackRequested, setPlaybackRequested] = useState(false);
  useEffect(() => setPlaybackRequested(false), [playbackEpoch, url]);

  return (
    <article
      className={`scoped-waveform-track is-${kind} is-role-${signalRole}`}
      data-signal-role={signalRole}
      data-runtime-microphone={
        signalRole === "microphone"
          ? waveform
            ? "ready"
            : "pending"
          : undefined
      }
      aria-label={`${caseId} ${label}`}
      style={{ borderTopColor: SIGNAL_COLOR }}
    >
      <header>
        <div>
          <span>{kind === "input" ? "PRE-TEST" : "POST-TEST"}</span>
          <strong>{label}</strong>
        </div>
        {waveform ? (
          <dl>
            <div>
              <dt>时长</dt>
              <dd>{formatDuration(waveform.durationSeconds)}</dd>
            </div>
            <div>
              <dt>采样率</dt>
              <dd>{formatSampleRate(waveform.sampleRate)}</dd>
            </div>
            <div>
              <dt>声道</dt>
              <dd>{waveform.channels}</dd>
            </div>
          </dl>
        ) : (
          <span className="scoped-waveform-state">
            {unavailable || (loading ? "正在解码波形…" : "等待波形")}
          </span>
        )}
      </header>

      {waveform ? (
        <>
          <WaveformSvg
            label={label}
            caseId={caseId}
            envelope={waveform}
            peakScale={peakScale}
            color={SIGNAL_COLOR}
          />
          {waveform.mel ? (
            <MelSpectrogramView
              label={label}
              caseId={caseId}
              mel={waveform.mel}
              ceilingDb={melCeilingDb}
              accent={MEL_ACCENT}
              maximumHz={Math.min(8_000, waveform.sampleRate / 2)}
            />
          ) : melLoading ? (
            <div className="scoped-mel-placeholder" role="status">
              <Activity size={16} aria-hidden="true" />
              <span>正在计算梅尔频谱…</span>
            </div>
          ) : null}
        </>
      ) : (
        <div className="scoped-waveform-placeholder" role="status">
          <Activity size={22} aria-hidden="true" />
          <span>{unavailable || (loading ? "正在读取音频…" : "暂无波形")}</span>
        </div>
      )}

      {error ? (
        <p className="scoped-waveform-error" role="alert">
          <AlertCircle size={14} aria-hidden="true" />
          {error}
          {waveform ? "；当前仍显示上次成功解码的波形。" : ""}
        </p>
      ) : null}

      {url && waveform ? (
        <footer>
          {playbackRequested ? (
            <audio
              controls
              autoPlay
              preload="metadata"
              src={url}
              aria-label={`播放 ${caseId} ${label}`}
            >
              当前浏览器不支持音频播放。
            </audio>
          ) : (
            <button
              className="scoped-waveform-play"
              type="button"
              aria-label={`播放 ${caseId} ${label}`}
              onClick={() => setPlaybackRequested(true)}
            >
              <Play size={14} fill="currentColor" aria-hidden="true" />
              播放 WAV
            </button>
          )}
          <a href={url} download aria-label={`下载 ${caseId} ${label}`}>
            <Download size={14} aria-hidden="true" />
            下载 WAV
          </a>
        </footer>
      ) : null}
    </article>
  );
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

function buildWaveformCaseLabels(
  cases: ScopedInputPreviewCase[],
): Map<string, string> {
  const counters = Object.fromEntries(
    CONVERSATION_MODES.map((mode) => [mode, 0]),
  ) as Record<ConversationMode, number>;
  const fallbackCounters = new Map<string, number>();
  const labels = new Map<string, string>();
  for (const previewCase of cases) {
    const mode = conversationModeOf(previewCase);
    if (mode) {
      counters[mode] += 1;
      labels.set(
        previewCase.case_id,
        `${CONVERSATION_MODE_LABELS[mode]} ${String(counters[mode]).padStart(
          2,
          "0",
        )}`,
      );
      continue;
    }
    const fallback = caseLabel(previewCase);
    const count = (fallbackCounters.get(fallback) ?? 0) + 1;
    fallbackCounters.set(fallback, count);
    labels.set(previewCase.case_id, `${fallback} ${count}`);
  }
  return labels;
}

function caseLabel(previewCase: ScopedInputPreviewCase): string {
  const families = new Set(previewCase.families);
  if (
    [...families].some((family) =>
      ["g_mos", "n_mos", "s_mos"].includes(family),
    ) &&
    [...families].some((family) =>
      ["cng_level", "cng_spectrum"].includes(family),
    )
  ) {
    return "语音 + CNG 条件组";
  }
  if (
    [...families].some((family) =>
      ["g_mos", "n_mos", "s_mos"].includes(family),
    )
  ) {
    return "语音条件组";
  }
  if (
    [...families].some((family) =>
      ["cng_level", "cng_spectrum"].includes(family),
    )
  ) {
    return "CNG 条件组";
  }
  return "背景噪声条件组";
}

function ConditionSummary({
  previewCase,
}: {
  previewCase: ScopedInputPreviewCase;
}) {
  const conditions = previewCase.conditions;
  return (
    <dl className="scoped-waveform-conditions">
      <div>
        <dt>场景</dt>
        <dd>{conditions.scene}</dd>
      </div>
      <div>
        <dt>音量</dt>
        <dd>{conditions.volume ?? "—"}</dd>
      </div>
      <div>
        <dt>窗口</dt>
        <dd>
          {conditions.window_ms === null
            ? "—"
            : `${conditions.window_ms} ms`}
        </dd>
      </div>
      <div>
        <dt>输入</dt>
        <dd>
          {conditions.input_level_db === null
            ? "—"
            : `${conditions.input_level_db} dB`}
        </dd>
      </div>
    </dl>
  );
}

function CaseWaveforms({
  previewCase,
  datasetId,
  snapshot,
  algorithmClusterId,
  mode,
  scene,
  datasetManifestSha256,
  refreshToken,
  visuals,
  sceneConditionPreview,
  sceneConditionBatch,
  sceneConditionCasesById,
  caseTitle,
  playbackEpoch,
}: {
  previewCase: ScopedInputPreviewCase;
  datasetId: string;
  snapshot: ScopedSnapshot | null;
  algorithmClusterId: string;
  mode: FiveSignalMode;
  scene: BackgroundNoiseScene;
  datasetManifestSha256: string;
  refreshToken: number;
  visuals?: Partial<
    Record<ScopedInputSignalRole, ScopedPreviewVisualTrack>
  >;
  sceneConditionPreview: boolean;
  sceneConditionBatch?: SceneConditionBatch | null;
  sceneConditionCasesById?: ReadonlyMap<
    string,
    { batch: SceneConditionBatch; caseRecord: SceneConditionCase }
  >;
  caseTitle: string;
  playbackEpoch: number;
}) {
  const conversationMode = conversationModeOf(previewCase);
  const segments = conversationMode
    ? segmentSummary(previewCase)
    : "";
  const effectiveSceneBatch = (() => {
    if (!sceneConditionPreview) return null;
    if (
      sceneConditionBatch
      && sceneConditionBatch.algorithm_cluster_id === algorithmClusterId
      && sceneConditionBatch.cases.some(
        (candidate) => candidate.case_id === previewCase.case_id,
      )
    ) {
      return sceneConditionBatch;
    }
    const perCase = sceneConditionCasesById?.get(previewCase.case_id);
    if (
      perCase
      && perCase.batch.algorithm_cluster_id === algorithmClusterId
    ) {
      return perCase.batch;
    }
    return sceneConditionBatch?.algorithm_cluster_id === algorithmClusterId
      ? sceneConditionBatch
      : null;
  })();
  const sceneEvidence = sceneConditionPreview
    ? sceneConditionCaseTxEvidence(
        previewCase,
        effectiveSceneBatch,
        datasetManifestSha256,
      )
    : null;
  const sceneRxEvidence = sceneConditionPreview
    ? sceneConditionCaseRxEvidence(
        previewCase,
        effectiveSceneBatch,
        datasetManifestSha256,
      )
    : null;
  const sceneDynamicEvidence = sceneConditionPreview
    ? sceneConditionCaseDynamicMicrophoneEvidence(
        previewCase,
        effectiveSceneBatch,
        datasetManifestSha256,
      )
    : null;
  const matchedSnapshot = sceneConditionPreview
    ? null
    : matchingScopedSnapshot(
        snapshot,
        mode,
        algorithmClusterId,
        datasetId,
        scene,
      );
  const evidence = scopedCaseEvidenceIdentity(
    previewCase,
    matchedSnapshot,
    datasetManifestSha256,
  );
  const {
    caseRecord,
    manifestMatches: manifestIdentityMatches,
    reportMatches,
    completeInputHashes,
    inputHashesMatch: inputIdentityMatches,
    txReady: outputIdentityMatches,
  } = evidence;
  const scopedDynamicMicrophoneEvidence = (() => {
    if (
      sceneConditionPreview
      || !matchedSnapshot
      || !outputIdentityMatches
      || !caseRecord
      || caseRecord.status !== "completed"
      || !caseRecord.dynamic_microphone
      || !caseRecord.dynamic_microphone_visual
      || caseRecord.dynamic_microphone_visual.source_sha256
        !== caseRecord.dynamic_microphone.sha256
      || (
        caseRecord.effective_input_sha256
        && !effectiveMicrophoneMatchesDynamic(
          caseRecord.effective_input_sha256.microphone,
          caseRecord.dynamic_microphone.sha256,
          caseRecord.execution,
        )
      )
    ) {
      return null;
    }
    return {
      url: acceptanceDynamicMicrophoneUrl(
        matchedSnapshot.batch_id,
        previewCase.case_id,
        caseRecord.dynamic_microphone.sha256,
      ),
      visual: caseRecord.dynamic_microphone_visual,
    };
  })();
  const dynamicMicrophoneEvidence = sceneConditionPreview
    ? sceneDynamicEvidence?.ready && sceneDynamicEvidence.visual
      ? {
          url: sceneDynamicEvidence.url,
          visual: sceneDynamicEvidence.visual,
        }
      : null
    : scopedDynamicMicrophoneEvidence;
  const runtimeMicrophonePendingReason = sceneConditionPreview
    ? (sceneDynamicEvidence?.reason && sceneDynamicEvidence.reason !== "批次缺少该条件组"
        ? sceneDynamicEvidence.reason
        : "运行时合成 · 待执行；点击测试后先经 VCP RX，再实时生成麦克风波形与梅尔频谱")
    : "运行时合成 · 待执行；点击测试后先经 VCP RX，再实时生成麦克风波形与梅尔频谱";
  const outputAudioUrl =
    sceneConditionPreview
      ? sceneEvidence?.ready ? sceneEvidence.url : ""
      :
    caseRecord?.status === "completed" &&
    matchedSnapshot &&
    outputIdentityMatches
      ? outputUrl(
          matchedSnapshot.batch_id,
          previewCase.case_id,
          matchedSnapshot.batch_id,
        )
      : "";
  const inputAssetUrl = (
    role: ScopedInputSignalRole,
    version: string,
  ) =>
    sceneConditionPreview
      ? sceneConditionInputUrl(
          datasetId,
          previewCase.case_id,
          role,
          version,
        )
      : inputUrl(datasetId, previewCase.case_id, role, version);
  const inputWaveformUrl = (
    role: ScopedInputSignalRole,
    version: string,
  ) =>
    role === "microphone"
      ? dynamicMicrophoneEvidence?.url ?? ""
      : inputAssetUrl(role, version);
  const microphone = useWaveform(
    dynamicMicrophoneEvidence ? dynamicMicrophoneEvidence.url : "",
    refreshToken,
    dynamicMicrophoneEvidence ? dynamicMicrophoneEvidence.visual : undefined,
  );
  const farEnd = useWaveform(
    inputAssetUrl("far_end", previewCase.signals_sha256.far_end),
    refreshToken,
    visuals?.far_end,
  );
  const reference = useWaveform(
    inputAssetUrl("reference", previewCase.signals_sha256.reference),
    refreshToken,
    visuals?.reference,
  );
  const noise = useWaveform(
    inputAssetUrl("noise", previewCase.signals_sha256.noise),
    refreshToken,
    visuals?.noise,
  );
  const rx = useWaveform(
    sceneRxEvidence?.ready ? sceneRxEvidence.url : "",
    refreshToken,
    sceneRxEvidence?.visual,
  );
  const output = useWaveform(outputAudioUrl, refreshToken, sceneEvidence?.visual);
  const inputTracks: Array<{
    role: ScopedInputSignalRole;
    state: ReturnType<typeof useWaveform>;
  }> = [
    { role: "microphone", state: microphone },
    { role: "far_end", state: farEnd },
    { role: "reference", state: reference },
    { role: "noise", state: noise },
  ];
  const peakScale = Math.max(
    ...inputTracks.map((track) => track.state.waveform?.peak ?? 0),
    rx.waveform?.peak ?? 0,
    output.waveform?.peak ?? 0,
  );
  const melCeilingDb = Math.max(
    ...inputTracks.map((track) => track.state.waveform?.mel?.maxDb ?? -120),
    rx.waveform?.mel?.maxDb ?? -120,
    output.waveform?.mel?.maxDb ?? -120,
  );
  const outputUnavailable = sceneConditionPreview
    ? sceneEvidence?.reason || "尚未执行该样本，完成后显示对应的 VCP 处理后输出"
    : matchedSnapshot
      ? !manifestIdentityMatches
        ? "历史批次的数据集清单身份与当前测试集不一致，不显示测试后波形"
        : !caseRecord
          ? "历史快照缺少该条件组的完成记录，不显示测试后波形"
          : !reportMatches
            ? "历史报告的条件组 report_id 与逐项证据不一致，不显示测试后波形"
            : !completeInputHashes
              ? "历史报告缺少完整四路输入哈希，无法绑定当前待测音频与 TX"
          : !inputIdentityMatches
            ? "当前四路输入已与历史批次不同，拒绝把新输入与旧 TX 并排显示"
            : caseRecord?.status === "completed"
              ? ""
              : caseRecord?.status === "failed"
                ? "条件组执行失败，没有测试后波形"
                : `条件组 ${caseRecord?.status ?? "尚未完成"}，测试后波形将在完成后出现`
      : snapshot
        ? "当前报告的数据集、模式或算法与页面选择不一致，不显示测试后波形"
        : "尚未执行；测试后波形将在条件组完成后出现";
  const rxUnavailable = sceneConditionPreview
    ? sceneRxEvidence?.reason
      || "本次测试尚未生成 VCP RX 处理后回声参考"
    : "该历史批次未记录 VCP RX 处理后回声参考";

  return (
    <article
      className={`scoped-waveform-case${
        conversationMode ? " is-conversation-preview" : ""
      }`}
      data-case-id={previewCase.case_id}
      data-conversation-mode={conversationMode ?? "legacy"}
    >
      <header>
        <div>
          <span>
            {conversationMode
              ? CONVERSATION_MODE_LABELS[conversationMode]
              : caseLabel(previewCase)}
          </span>
          <h4>
            <span>{caseTitle}</span>{" "}
            <code>{previewCase.case_id}</code>
          </h4>
          <p>
            {previewCase.families.join(" · ")}
          </p>
        </div>
        <ConditionSummary previewCase={previewCase} />
      </header>
      {conversationMode ? (
        <p className="scoped-waveform-composition">
          <strong>
            {conversationComposition(conversationMode, previewCase)}
          </strong>
          {segments ? <span>分段：{segments}</span> : null}
        </p>
      ) : null}
      <div className="scoped-waveform-tracks">
        {SCOPED_INPUT_SIGNAL_ROLES.map((role) => {
          const track = inputTracks.find(
            (candidate) => candidate.role === role,
          )?.state;
          if (!track) return null;
          return (
            <WaveformTrack
              key={role}
              label={
                role === "microphone"
                  ? DYNAMIC_MICROPHONE_LABEL
                  : role === "far_end" && sceneConditionPreview
                  ? "回声参考 · VCP RX 前 · far_end"
                  : INPUT_SIGNAL_LABELS[role]
              }
              caseId={previewCase.case_id}
              url={inputWaveformUrl(
                role,
                previewCase.signals_sha256[role],
              )}
              kind={role === "microphone" ? "output" : "input"}
              signalRole={role}
              peakScale={peakScale}
              unavailable={
                role === "microphone" && !dynamicMicrophoneEvidence
                  ? runtimeMicrophonePendingReason
                  : undefined
              }
              waveform={track.waveform}
              melCeilingDb={melCeilingDb}
              melLoading={track.melLoading}
              loading={track.loading}
              error={track.error}
              playbackEpoch={playbackEpoch}
            />
          );
        })}
        {sceneConditionPreview ? (
          <WaveformTrack
            label="回声参考 · VCP RX 后 · TX 实际使用"
            caseId={previewCase.case_id}
            url={sceneRxEvidence?.ready ? sceneRxEvidence.url : ""}
            kind="output"
            signalRole="rx"
            peakScale={peakScale}
            unavailable={rxUnavailable}
            waveform={rx.waveform}
            melCeilingDb={melCeilingDb}
            melLoading={rx.melLoading}
            loading={rx.loading}
            error={rx.error}
            playbackEpoch={playbackEpoch}
          />
        ) : null}
        <WaveformTrack
          label="测试后音频 · TX"
          caseId={previewCase.case_id}
          url={outputAudioUrl}
          kind="output"
          signalRole="tx"
          peakScale={peakScale}
          unavailable={outputUnavailable}
          waveform={output.waveform}
          melCeilingDb={melCeilingDb}
          melLoading={output.melLoading}
          loading={output.loading}
          error={output.error}
          playbackEpoch={playbackEpoch}
        />
      </div>
    </article>
  );
}

type CaseWaveformsProps = Parameters<typeof CaseWaveforms>[0];

const LazyCaseWaveforms = memo(function LazyCaseWaveforms({
  lazy,
  ...props
}: CaseWaveformsProps & { lazy: boolean }) {
  const slotRef = useRef<HTMLDivElement>(null);
  const [nearViewport, setNearViewport] = useState(!lazy);
  const conversationMode = conversationModeOf(props.previewCase);

  useEffect(() => {
    if (!lazy || nearViewport) {
      setNearViewport(true);
      return;
    }
    const node = slotRef.current;
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
      { rootMargin: "1200px 0px" },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [lazy, nearViewport]);

  return (
    <div
      ref={slotRef}
      className="scoped-waveform-case-slot"
      data-case-id={props.previewCase.case_id}
      data-lazy-state={nearViewport ? "mounted" : "deferred"}
    >
      {nearViewport ? (
        <CaseWaveforms {...props} />
      ) : (
        <article className="scoped-waveform-case scoped-waveform-case-lazy">
          <header>
            <div>
              <span>
                {conversationMode
                  ? CONVERSATION_MODE_LABELS[conversationMode]
                  : caseLabel(props.previewCase)}
              </span>
              <h4>{props.caseTitle}</h4>
              <p>滚动到此处后读取四路输入并计算波形与梅尔频谱。</p>
            </div>
          </header>
        </article>
      )}
    </div>
  );
});

export default function ScopedAcousticWaveforms({
  definition,
  datasetId,
  mode,
  algorithmClusterId,
  snapshot,
  active,
  previewSource = "extended",
  sceneConditionBatch = null,
  sceneConditionCasesById,
}: Props) {
  const titleId = useId().replace(/[^A-Za-z0-9_-]/g, "");
  const scene = definition.scene;
  const idle20Preview = previewSource !== "execution" && (isSceneCondition20Preview(definition, mode) || (definition.previewMode === "scene-condition-20" && Boolean(definition.scene) && FIVE_SIGNAL_MODES.includes(mode)));
  const idle20Definition = previewSource !== "execution" && definition.previewMode === "scene-condition-20";
  const [preview, setPreview] =
    useState<ScopedBackgroundNoisePreview | null>(null);
  const [expectedDatasetId, setExpectedDatasetId] = useState("");
  const [sceneConditionPreviewValidated, setSceneConditionPreviewValidated] =
    useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [refresh, setRefresh] = useState(0);
  const [conversationFilter, setConversationFilter] =
    useState<ConversationFilter>("all");
  const [playbackEpoch, setPlaybackEpoch] = useState(0);

  useEffect(() => {
    if (!active || !scene || (!idle20Preview && !datasetId)) {
      setPreview(null);
      setExpectedDatasetId("");
      setSceneConditionPreviewValidated(false);
      setLoading(false);
      setError("");
      return;
    }
    const controller = new AbortController();
    let disposed = false;
    setLoading(true);
    setError("");
    setSceneConditionPreviewValidated(false);

    const loadPreview = async () => {
      try {
        if (idle20Preview) {
          const datasetParams = new URLSearchParams({
            scope: "background_noise",
            mode,
            scene,
          });
          const datasetBody = await requestSceneConditionJson(
            `${SCENE_CONDITION_DATASETS_URL}?${datasetParams.toString()}`,
            controller.signal,
            "读取场景工程预览数据集",
            "no-store",
          );
          const datasetView = readSceneConditionDatasetView(datasetBody, {
            mode,
            scene,
          });
          const previewDatasetId =
            datasetView.datasets[0]?.dataset_id ?? "";
          if (!disposed && !controller.signal.aborted) {
            setExpectedDatasetId(previewDatasetId);
            setPreview(null);
            setSceneConditionPreviewValidated(false);
          }
          const previewParams = new URLSearchParams({
            dataset_id: previewDatasetId,
            mode,
            scene,
          });
          const previewBody = await requestSceneConditionJson(
            `${SCENE_CONDITION_PREVIEW_URL}?${previewParams.toString()}`,
            controller.signal,
            "读取场景工程预览",
            "no-cache",
          );
          const next = readSceneConditionPreview(
            previewBody,
            previewDatasetId,
            mode,
            scene,
          );
          if (!disposed && !controller.signal.aborted) {
            setPreview(next);
            setSceneConditionPreviewValidated(true);
          }
          return;
        }

        const params = new URLSearchParams({
          dataset_id: datasetId,
          mode,
          visual_schema: SCOPED_PREVIEW_VISUAL_SCHEMA,
        });
        const body = await requestSceneConditionJson(
          `${PREVIEW_URL}/${encodeURIComponent(scene)}/preview?${params.toString()}`,
          controller.signal,
          "条件组预览",
          "no-cache",
        );
        const next = readScopedBackgroundNoisePreview(
          body,
          datasetId,
          mode,
          scene,
        );
        if (!disposed && !controller.signal.aborted) {
          setExpectedDatasetId(datasetId);
          setPreview(next);
          setSceneConditionPreviewValidated(false);
        }
      } catch (reason) {
        if (!disposed && !controller.signal.aborted) {
          setError(errorMessage(reason, "条件组预览读取失败"));
          setSceneConditionPreviewValidated(false);
        }
      } finally {
        if (!disposed && !controller.signal.aborted) setLoading(false);
      }
    };

    void loadPreview();

    return () => {
      disposed = true;
      controller.abort();
    };
  }, [active, datasetId, idle20Preview, mode, refresh, scene]);

  const visiblePreview =
    scene && idle20Preview
      ? preview
        && sceneConditionPreviewValidated
        && isSceneConditionPreview(preview, scene)
        && preview.dataset_id === expectedDatasetId
        && preview.mode === mode
        && preview.scene === scene
        ? preview
        : null
      : scene
        && preview
        && !isSceneConditionPreview(preview, scene)
        && previewMatches(preview, datasetId, mode, scene)
        ? preview
        : null;
  const profileDatasetId =
    visiblePreview?.dataset_id ?? (expectedDatasetId || datasetId);
  const activeProfile = useMemo(
    () =>
      profileDatasetId
        ? sceneConditionProfileForDatasetId(profileDatasetId)
        : null,
    [profileDatasetId],
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
    idle20Preview && hasConversationModes;
  const filteredCases = useMemo(() => {
    const cases = visiblePreview?.cases ?? [];
    return conversationFilter === "all"
      ? cases
      : cases.filter(
          (previewCase) =>
            conversationModeOf(previewCase) === conversationFilter,
        );
  }, [conversationFilter, visiblePreview?.cases]);
  const caseLabels = useMemo(
    () => buildWaveformCaseLabels(visiblePreview?.cases ?? []),
    [visiblePreview?.cases],
  );
  const conversationPreview =
    activeProfile?.schemaVersion === "scene-condition-preview-v2";
  const engineeringSummary = engineeringCombinationSummary(
    visiblePreview?.cases.length ?? expectedCounts?.cases ?? 0,
    visiblePreview?.cases[0],
  );
  const lazyConversationPreview =
    conversationPreview && filteredCases.length > 0;
  const activeConversationLabel =
    conversationFilter === "all"
      ? "全部"
      : CONVERSATION_MODE_LABELS[conversationFilter];
  const activeConversationDescription =
    conversationFilter === "all"
      ? "五种状态是工程组合，跨组合允许复用源录音；每组显示四路输入和同一样本绑定的 TX。"
      : conversationComposition(conversationFilter);

  useEffect(() => {
    setConversationFilter("all");
    setPlaybackEpoch((current) => current + 1);
  }, [
    expectedDatasetId,
    scene,
    sceneConditionPreviewValidated,
  ]);

  const matchedSnapshot = useMemo(
    () =>
      scene
        ? matchingScopedSnapshot(
            snapshot,
            mode,
            algorithmClusterId,
            datasetId,
            scene,
          )
        : null,
    [algorithmClusterId, datasetId, mode, scene, snapshot],
  );
  const txReadyCaseIds = useMemo(() => {
    const ready = new Set<string>();
    if (!visiblePreview) return ready;
    for (const previewCase of visiblePreview.cases) {
      const perCaseBatch = (() => {
        if (
          sceneConditionBatch
          && sceneConditionBatch.algorithm_cluster_id === algorithmClusterId
          && sceneConditionBatch.cases.some(
            (candidate) => candidate.case_id === previewCase.case_id,
          )
        ) {
          return sceneConditionBatch;
        }
        const entry = sceneConditionCasesById?.get(previewCase.case_id);
        if (entry && entry.batch.algorithm_cluster_id === algorithmClusterId) {
          return entry.batch;
        }
        return sceneConditionBatch?.algorithm_cluster_id === algorithmClusterId
          ? sceneConditionBatch
          : null;
      })();
      const isReady = idle20Preview
        ? sceneConditionCaseTxEvidence(
            previewCase,
            perCaseBatch,
            visiblePreview.dataset_manifest_sha256,
          ).ready
        : Boolean(
            matchedSnapshot &&
            scopedCaseEvidenceIdentity(
              previewCase,
              matchedSnapshot,
              visiblePreview.dataset_manifest_sha256,
            ).txReady,
          );
      if (isReady) ready.add(previewCase.case_id);
    }
    return ready;
  }, [
    algorithmClusterId,
    idle20Preview,
    matchedSnapshot,
    sceneConditionBatch,
    sceneConditionCasesById,
    visiblePreview,
  ]);
  const txReadyCases = txReadyCaseIds.size;
  const filteredTxReadyCases = filteredCases.filter((previewCase) =>
    txReadyCaseIds.has(previewCase.case_id),
  ).length;

  if (definition.scope !== "background_noise" || !scene) return null;

  if (!active) {
    return (
      <section
        className="scoped-waveforms"
        aria-labelledby={`scoped-waveforms-${titleId}`}
        aria-hidden="true"
      >
        <div className="scoped-waveforms-empty" role="status">
          <Activity size={22} aria-hidden="true" />
          <span>切换到“波形”子页面后加载四路输入、波形与梅尔频谱。</span>
        </div>
      </section>
    );
  }

  return (
    <section
      className="scoped-waveforms"
      aria-labelledby={`scoped-waveforms-${titleId}`}
    >
      <header className="scoped-waveforms-header">
        <div>
          <span className="basic-uplink-eyebrow">AUDIO WAVEFORM + MEL</span>
          <h3 id={`scoped-waveforms-${titleId}`}>
            {previewSource === "execution"
              ? "本次执行数据 · 波形与梅尔频谱"
              : conversationPreview
                ? "五类通话状态 · 波形与梅尔频谱"
                : "待测音频、波形与梅尔频谱"}
          </h3>
          <p>
            {conversationPreview
              ? `显示工程组合各自的四路输入、VCP RX 处理后回声参考和发送处理输出（TX）。${engineeringSummary}远端单讲包含卷积回声，双讲叠加两端与噪声；CNG 的预热、收敛和观测窗按 provenance 分段显示，整段不按纯噪声解释。RX 后音频在点击测试后现场计算，TX 使用同一 RX 输出作为 --echo。波形和梅尔频谱使用统一蓝色色标。`
              : idle20Preview
                ? "显示 20 组样本各自的四路输入、VCP RX 处理后回声参考和发送处理输出（TX）。干净近端参考用于质量对照；far_end 是 RX 前，RX 后音频在点击测试后生成并作为 TX 的实际回声参考。每条输出均绑定同一样本的四路输入哈希。"
              : "测试前显示该条件组已校验的四路输入。这里的 reference 是“干净近端参考”；far_end 是“VCP RX 前回声参考”。点击测试后现场计算 RX 后音频，TX 使用同一 RX 输出作为 --echo，并显示 RX 前、RX 后和 TX 三类波形。"}
          </p>
          {conversationPreview && expectedCounts ? (
            <p className="scoped-waveforms-preview-note">
              {engineeringSummary}5 类指标 · {expectedCounts.observations}
              次测量；工程代理，正式 ITU-T 合规未测。
            </p>
          ) : idle20Preview ? (
            <p className="scoped-waveforms-preview-note">
              20 组样本 · 5 类指标 · 50 次测量；工程代理，正式 ITU-T 合规未测。
            </p>
          ) : idle20Definition ? (
            <p className="scoped-waveforms-preview-note is-warning">
              20 组预览仅支持 WB-wired；当前模式显示旧 2 个可执行条件组
            </p>
          ) : null}
        </div>
        <div className="scoped-waveforms-actions">
          <span>
            {visiblePreview
              ? conversationPreview
                ? `显示 ${filteredCases.length} / 全部 ${visiblePreview.cases.length} 种 · 筛选内 ${filteredTxReadyCases} 个 TX 可用 · 全部 ${txReadyCases} 个`
                : idle20Preview
                  ? `${visiblePreview.cases.length} 组样本 · ${txReadyCases} 个 TX 可用`
                : `${visiblePreview.cases.length} 个条件组`
              : loading
                ? conversationPreview
                  ? "正在读取 100 种工程组合"
                  : idle20Preview
                    ? "正在读取工程预览"
                  : "正在读取条件组"
                : conversationPreview
                  ? "暂无工程组合预览"
                  : idle20Preview
                    ? "暂无工程预览"
                  : "暂无条件组"}
          </span>
          {!idle20Preview && matchedSnapshot ? (
            <span>
              {txReadyCases} / {matchedSnapshot.progress.total_cases} 个 TX 可用
            </span>
          ) : null}
          <button
            className="button secondary"
            type="button"
            disabled={(!idle20Preview && !datasetId) || loading}
            aria-label={
              conversationPreview && expectedCounts
                ? `刷新${definition.title}${expectedCounts.cases}种工程组合四路输入、波形与梅尔频谱`
                : idle20Preview
                  ? `刷新${definition.title}工程预览四路输入、波形与梅尔频谱`
                : `刷新${definition.title}四路输入、RX 前后参考、TX 波形与梅尔频谱`
            }
            onClick={() => setRefresh((current) => current + 1)}
          >
            <RefreshCw size={14} aria-hidden="true" />
            刷新波形
          </button>
        </div>
      </header>

      {showConversationFilter && visiblePreview ? (
        <div className="scoped-waveform-filter-shell">
          <div
            className="scoped-waveform-filter"
            role="group"
            aria-label="按通话状态筛选波形工程组合"
            data-testid="waveform-conversation-mode-filter"
          >
            <button
              type="button"
              className={conversationFilter === "all" ? "is-active" : ""}
              aria-pressed={conversationFilter === "all"}
              data-filter-mode="all"
              onClick={() => {
                setConversationFilter("all");
                setPlaybackEpoch((current) => current + 1);
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
                  setConversationFilter(conversationMode);
                  setPlaybackEpoch((current) => current + 1);
                }}
              >
                <span>{CONVERSATION_MODE_LABELS[conversationMode]}</span>
                <strong>{conversationCounts[conversationMode]}</strong>
              </button>
            ))}
          </div>
          <p className="scoped-waveform-filter-note">
            <strong>{activeConversationLabel}</strong>
            <span>{activeConversationDescription}</span>
            <span>
              显示 {filteredCases.length} / 全部 {visiblePreview.cases.length}
            </span>
          </p>
        </div>
      ) : null}

      {!idle20Preview && !datasetId ? (
        <div className="scoped-waveforms-empty" role="status">
          <Activity size={22} aria-hidden="true" />
          <span>选择完整测试集后显示待测音频波形。</span>
        </div>
      ) : null}

      {error ? (
        <p className="scoped-waveforms-warning" role="alert">
          <AlertCircle size={15} aria-hidden="true" />
          {error}
          {visiblePreview ? "；当前仍显示上次成功读取的条件组。" : ""}
        </p>
      ) : null}

      {!visiblePreview && (idle20Preview || datasetId) && loading ? (
        <div className="scoped-waveforms-empty" role="status">
          <Activity size={22} aria-hidden="true" />
          <span>正在读取待测音频与条件组身份…</span>
        </div>
      ) : null}

      {visiblePreview ? (
        <div className="scoped-waveform-cases">
          {filteredCases.length > 0 ? (
            filteredCases.map((previewCase) => (
              <LazyCaseWaveforms
                key={[
                  visiblePreview.dataset_id,
                  visiblePreview.mode,
                  conversationFilter,
                  previewCase.case_id,
                ].join(":")}
                previewCase={previewCase}
                datasetId={visiblePreview.dataset_id}
                snapshot={snapshot}
                algorithmClusterId={algorithmClusterId}
                mode={visiblePreview.mode}
                scene={visiblePreview.scene}
                datasetManifestSha256={
                  visiblePreview.dataset_manifest_sha256
                }
                refreshToken={refresh}
                visuals={visiblePreview.visuals?.cases[previewCase.case_id]}
                sceneConditionPreview={idle20Preview}
                sceneConditionBatch={sceneConditionBatch}
                sceneConditionCasesById={sceneConditionCasesById}
                caseTitle={
                  caseLabels.get(previewCase.case_id) ?? "测试样本"
                }
                playbackEpoch={playbackEpoch}
                lazy={lazyConversationPreview}
              />
            ))
          ) : (
            <div className="scoped-waveform-filter-empty" role="status">
              <span>当前筛选没有工程组合</span>
              <button
                type="button"
                onClick={() => {
                  setConversationFilter("all");
                  setPlaybackEpoch((current) => current + 1);
                }}
              >
                查看全部
              </button>
            </div>
          )}
        </div>
      ) : null}
    </section>
  );
}
