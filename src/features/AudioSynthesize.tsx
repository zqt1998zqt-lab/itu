import { useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  Check,
  Cpu,
  Download,
  FileAudio,
  LoaderCircle,
  Sparkles,
  Upload,
  X,
} from "lucide-react";
import { SpectrumPlot, Waveform } from "../components/AudioCharts";
import { ROLE_COLOR } from "../domain/audio";
import type { AudioDescription } from "../domain/audio";
import { downloadFile } from "../lib/files";
import { matchingPreset, chainExecutionIssue } from "./enhancement-chain-model";
import type { ChainConfiguration } from "./enhancement-chain-model";
import { buildSynthesisOptions, synthesisConfigurationKey } from "./synthesize-model";
import type { MicrophoneMode, SynthesisOptions, SynthesisProfile } from "./synthesize-model";
import { parseAudioGraphicReport } from "../domain/audio-graphics";
import type { AudioGraphicReport } from "../domain/audio-graphics";
import "./audio-graph-engine.css";
import "./synthesize-controls.css";
import NsFailureAudio from "./NsFailureAudio";
import "./synthesize.css";

type Slot = "microphone" | "reference" | "clean";

type Levels = {
  rms_dbfs: number;
  peak_dbfs: number;
  clipping_percent: number;
  silent: boolean;
};

type Side = Levels & {
  name: string;
  sample_rate: number;
  channels: number;
  selected_channel: number | null;
  processed_channels?: number;
  channel_mode?: string;
};

type Result = {
  engine: string;
  profile: string;
  binary: string;
  sample_rate: number;
  seconds: number;
  trimmed_seconds: number;
  inputs: { microphone: Side; reference: Side };
  output: Side & { samples: number };
  engine_report: Record<string, number | string | boolean | null>;
  charts: AudioDescription[];
  metrics?: {
    available: boolean;
    reason?: string;
    scope?: string;
    level_change_db?: number;
    reference_active_percent?: number;
    band_change_db?: { low_hz: number; high_hz: number; change_db: number }[];
    quality?: { available: boolean; reason?: string; si_sdr_db?: number; stoi?: number | null; scope?: string };
  };
  download_url: string;
  graphics_url?: string;
  created_at: string;
};

type Demo = { id: string; label: string; description: string };
type Operation = { id: number; kind: "demo" | "generate"; demoId?: string; controller: AbortController };
type RequestIdentity = {
  id: string;
  inputs: Record<Slot, { name: string; bytes: number; last_modified: number } | null>;
  options: SynthesisOptions;
  configuration_key: string;
};
type CompletedResult = Result & { request: RequestIdentity };

// The engine's role vocabulary, relabelled for this page.
const CHART_LABEL: Record<string, string> = {
  before: "麦克音",
  echo: "参考音",
  after: "输出",
};

const SLOTS: { id: Slot; label: string; hint: string }[] = [
  {
    id: "microphone",
    label: "麦克音",
    hint: "算法处理前的麦克风录音，保留噪声与回声。",
  },
  {
    id: "reference",
    label: "参考音",
    hint: "与麦克音同步的实际播放信号；保留全部 1–8 路声道，每路对应一条播放参考。",
  },
  {
    id: "clean",
    label: "干净近端",
    hint: "可选的近端纯净参考，用于计算 SI-SDR / STOI；不提供则跳过得分。",
  },
];

const RATES = [
  { value: "", label: "自动（跟随麦克音）" },
  { value: "16000", label: "16000 Hz" },
  { value: "24000", label: "24000 Hz" },
  { value: "48000", label: "48000 Hz" },
];

const REPORT_LABEL: Record<string, string> = {
  sample_rate: "采样率",
  dsp_delay_samples: "算法固定时延",
  stream_resets: "引擎重置次数",
  process_p95_ms: "处理 p95",
  process_max_ms: "处理最大",
  process_rtf: "实时率 RTF",
  processing_profile: "处理档位",
  microphones: "麦克风数",
  output_channels: "输出声道",
  wind_noise_control: "空间风噪控制",
  reference_unreliable_frames: "参考不可靠帧数",
};

function formatValue(key: string, value: number | string | boolean | null) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "boolean") return value ? "是" : "否";
  if (typeof value === "number") {
    if (key === "process_rtf") return value.toFixed(4);
    if (key.endsWith("_ms")) return `${value.toFixed(3)} ms`;
    return Number.isInteger(value) ? String(value) : value.toFixed(4);
  }
  return String(value);
}

function UploadCard({
  slot,
  file,
  disabled,
  onFile,
}: {
  slot: (typeof SLOTS)[number];
  file: File | null;
  disabled: boolean;
  onFile: (file: File | null) => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [url, setUrl] = useState("");
  const [error, setError] = useState("");
  const [drag, setDrag] = useState(false);
  useEffect(() => {
    setError("");
    if (!file) {
      setUrl("");
      return;
    }
    const next = URL.createObjectURL(file);
    setUrl(next);
    return () => URL.revokeObjectURL(next);
  }, [file]);
  function accept(next?: File) {
    if (!next || disabled) return;
    if (!/\.(wav|flac|mp3|ogg|m4a|aac|aif|aiff)$/i.test(next.name)) {
      setError("请选择音频文件。支持 WAV / FLAC / MP3 / OGG / M4A / AAC / AIFF。");
      return;
    }
    if (!next.size || next.size > 32 * 1024 * 1024) {
      setError("请选择非空、32 MB 以内的音频。");
      return;
    }
    setError("");
    onFile(next);
  }
  return (
    <section
      className={`audio-upload-card ${file ? "has-file" : ""} ${drag ? "dragging" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setDrag(true);
      }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDrag(false);
        accept(e.dataTransfer.files[0]);
      }}
    >
      <div className="audio-role-heading">
        <span className="role-icon">
          <FileAudio size={19} />
        </span>
        <div>
          <h3>{slot.label}</h3>
          <span>{slot.id === "clean" ? "可选" : "必需"}</span>
        </div>
        {file ? <Check size={17} className="file-check" /> : null}
      </div>
      <p className="audio-role-hint">{slot.hint}</p>
      <input
        ref={input}
        type="file"
        className="sr-only"
        aria-label={`上传${slot.label}`}
        accept=".wav,.flac,.mp3,.ogg,.m4a,.aac,.aif,.aiff"
        disabled={disabled}
        onChange={(e) => {
          accept(e.target.files?.[0]);
          e.target.value = "";
        }}
      />
      {file ? (
        <>
          <div className="selected-audio">
            <FileAudio size={16} />
            <div>
              <strong title={file.name}>{file.name}</strong>
              <small>{(file.size / 1024 / 1024).toFixed(2)} MB · 生成时统一到目标采样率</small>
            </div>
            <button
              disabled={disabled}
              className="icon-button"
              aria-label={`移除${slot.label}`}
              onClick={() => onFile(null)}
            >
              <X size={16} />
            </button>
          </div>
          <audio controls preload="metadata" src={url || undefined} aria-label={`${slot.label}试听`} />
          <div className="audio-card-bottom">
            <small className="preview-note">{slot.id === "reference" ? "试听原文件；生成保留全部参考声道。" : slot.id === "microphone" ? "试听原文件；处理声道由下方麦克风方式决定。" : "试听原文件；评分使用第 1 声道。"}</small>
            <button className="text-link" disabled={disabled} onClick={() => input.current?.click()}>
              重新选择
            </button>
          </div>
        </>
      ) : (
        <button className="audio-dropzone" disabled={disabled} onClick={() => input.current?.click()}>
          <Upload size={22} />
          <strong>选择音频或拖放到这里</strong>
          <span>每文件 ≤32 MB · 0.1–120 秒</span>
        </button>
      )}
      {error ? (
        <p className="field-error" role="alert">
          {error}
        </p>
      ) : null}
    </section>
  );
}


function drawMel(canvas: HTMLCanvasElement | null, mel?: { db: number[][] }) {
  if (!canvas || !mel?.db?.length) return;
  const rows = mel.db.length;
  const cols = mel.db[0].length;
  const width = 720;
  const height = 180;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const cw = width / cols;
  const ch = height / rows;
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const v = Math.max(0, Math.min(1, (mel.db[r][c] + 80) / 80));
      ctx.fillStyle = `hsl(${Math.round(235 - 235 * v)} 85% ${Math.round(12 + 52 * v)}%)`;
      ctx.fillRect(c * cw, r * ch, Math.ceil(cw), Math.ceil(ch));
    }
  }
}

export default function AudioSynthesize({ configuration, embedded = false }: {
  configuration?: ChainConfiguration; embedded?: boolean;
} = {}) {
  const [mode, setMode] = useState<"failures" | "generate">(embedded ? "generate" : "failures");
  const [files, setFiles] = useState<Record<Slot, File | null>>({ microphone: null, reference: null, clean: null });
  const [rate, setRate] = useState("");
  const [profile, setProfile] = useState<SynthesisProfile>("vcp");
  const [microphoneMode, setMicrophoneMode] = useState<MicrophoneMode>("channel0");
  const [delayText, setDelayText] = useState("");
  const [graphics, setGraphics] = useState<AudioGraphicReport | null>(null);
  const [graphicsError, setGraphicsError] = useState("");
  const [graphicsBusy, setGraphicsBusy] = useState(false);
  const graphicsPending = useRef<AbortController | null>(null);
  const configurationKey = synthesisConfigurationKey(configuration, profile);
  const currentConfigurationKey = useRef(configurationKey);
  currentConfigurationKey.current = configurationKey;
  const executionIssue = configuration ? chainExecutionIssue(configuration) : null;
  const profileLabel = configuration ? matchingPreset(configuration)?.label ?? "自定义组合" : profile === "vcp_joint" ? "VCP 联合优化" : "VCP 默认";
  const [operation, setOperation] = useState<Operation | null>(null);
  const pending = useRef<Operation | null>(null);
  const sequence = useRef(0);
  const [error, setError] = useState("");
  const [result, setResult] = useState<CompletedResult | null>(null);
  const [demos, setDemos] = useState<Demo[]>([]);
  const [activeDemo, setActiveDemo] = useState("");
  const busy = operation?.kind === "generate";
  const locked = operation !== null || graphicsBusy;
  const ready = Boolean(files.microphone && files.reference);

  useEffect(() => () => {
    graphicsPending.current?.abort();
    pending.current?.controller.abort();
    pending.current = null;
  }, []);

  useEffect(() => {
    // A result belongs to its submitted route, never to a later preset or stage selection.
    const task = pending.current;
    if (task) { task.controller.abort(); pending.current = null; setOperation(null); }
    graphicsPending.current?.abort();
    graphicsPending.current = null;
    setGraphicsBusy(false);
    setGraphics(null);
    setGraphicsError("");
    setResult(null);
    setError("");
  }, [configurationKey]);

  function begin(kind: Operation["kind"], demoId?: string) {
    // The ref closes the gap before React renders the disabled controls.
    if (pending.current) return null;
    const next: Operation = { id: ++sequence.current, kind, demoId, controller: new AbortController() };
    pending.current = next;
    setOperation(next);
    setError("");
    setResult(null);
    setGraphics(null);
    setGraphicsError("");
    return next;
  }

  function finish(task: Operation) {
    if (pending.current !== task) return;
    pending.current = null;
    setOperation(null);
  }

  function cancel() {
    graphicsPending.current?.abort();
    graphicsPending.current = null;
    setGraphicsBusy(false);
    const task = pending.current;
    if (task) {
      task.controller.abort();
      finish(task);
    }
  }

  function changeMode(next: typeof mode) {
    if (mode !== next) cancel();
    setMode(next);
  }

  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/synthesize/demos", { signal: controller.signal })
      .then((response) => (response.ok ? response.json() : null))
      .then((data) => {
        if (data && Array.isArray(data.scenarios)) setDemos(data.scenarios as Demo[]);
      })
      .catch(() => undefined);
    return () => controller.abort();
  }, []);

  async function loadDemo(demo: Demo) {
    const task = begin("demo", demo.id);
    if (!task) return;
    try {
      const fetched = await Promise.all(
        (["microphone", "reference", "clean"] as Slot[]).map(async (slot) => {
          const response = await fetch(`/api/synthesize/demo/${demo.id}/${slot}`, { signal: task.controller.signal });
          if (!response.ok) throw Error("示例音频暂不可用，请检查计算服务。");
          const label = SLOTS.find((item) => item.id === slot)?.label ?? slot;
          const blob = await response.blob();
          return [slot, new File([blob], `示例_${demo.label}_${label}.wav`, { type: "audio/wav" })] as const;
        }),
      );
      if (pending.current !== task || task.controller.signal.aborted) return;
      setFiles(Object.fromEntries(fetched) as Record<Slot, File | null>);
      setActiveDemo(demo.id);
      setResult(null);
    } catch (e) {
      if (pending.current === task && !task.controller.signal.aborted) {
        setError(e instanceof Error ? e.message : "示例加载失败");
      }
    } finally {
      task.controller.abort();
      finish(task);
    }
  }

  async function generate() {
    if (!files.microphone || !files.reference) {
      setError("请先选择麦克音与参考音。");
      return;
    }
    let options: SynthesisOptions;
    try { options = buildSynthesisOptions(configuration, profile, rate, microphoneMode, delayText); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "请检查处理条件。"); return; }
    const task = begin("generate");
    if (!task) return;
    const request: RequestIdentity = {
      id: `${Date.now()}-${task.id}`,
      inputs: Object.fromEntries(SLOTS.map(({ id }) => {
        const file = files[id];
        return [id, file ? { name: file.name, bytes: file.size, last_modified: file.lastModified } : null];
      })) as RequestIdentity["inputs"],
      options,
      configuration_key: configurationKey,
    };
    const timer = setTimeout(() => task.controller.abort(), 300000);
    try {
      const body = new FormData();
      body.append("microphone", files.microphone);
      body.append("reference", files.reference);
      if (files.clean) body.append("clean", files.clean);
      body.append("options", JSON.stringify(request.options));
      const response = await fetch("/api/synthesize", { method: "POST", body, signal: task.controller.signal });
      const payload = await response.json().catch(() => null);
      if (!response.ok) {
        throw Error((payload && (payload.detail as string)) || `生成失败（HTTP ${response.status}）`);
      }
      if (pending.current === task && !task.controller.signal.aborted && currentConfigurationKey.current === request.configuration_key) {
        setResult({ ...(payload as Result), request });
      }
    } catch (e) {
      if (pending.current === task) setError(
        e instanceof DOMException && e.name === "AbortError"
          ? "生成超时，请缩短音频后重试。"
          : e instanceof Error
            ? e.message
            : "生成失败",
      );
    } finally {
      clearTimeout(timer);
      finish(task);
    }
  }

  async function generateGraphics() {
    if (!result?.graphics_url || graphicsPending.current) return;
    const submittedResult = result;
    const controller = new AbortController();
    graphicsPending.current = controller;
    setGraphicsBusy(true);
    setGraphicsError("");
    try {
      const response = await fetch(submittedResult.graphics_url!, { method: "POST", signal: controller.signal });
      const payload = await response.json().catch(() => null);
      if (!response.ok) throw new Error(typeof payload?.detail === "string" ? payload.detail : "对比图生成失败，请重试。");
      if (!controller.signal.aborted && graphicsPending.current === controller &&
        currentConfigurationKey.current === submittedResult.request.configuration_key) {
        setGraphics(parseAudioGraphicReport(payload));
      }
    } catch (reason) {
      if (!controller.signal.aborted) setGraphicsError(reason instanceof Error ? reason.message : "对比图生成失败。");
    } finally {
      if (graphicsPending.current === controller) { graphicsPending.current = null; setGraphicsBusy(false); }
    }
  }

  return (
    <div className="synthesize">
      {!embedded ? <div className="synthesize-modes" role="group" aria-label="音频展示方式">
        <button type="button" aria-pressed={mode === "failures"} onClick={() => changeMode("failures")}>失败样本试听</button>
        <button type="button" aria-pressed={mode === "generate"} onClick={() => changeMode("generate")}>上传并生成输出</button>
      </div> : null}
      {mode === "failures" ? <NsFailureAudio /> : <>
      <div className="synthesize-route" aria-live="polite">
        {embedded ? <p><strong>当前处理方案：{profileLabel}</strong> · {configuration?.profile}</p> :
          <label><span>处理方案</span><select aria-label="选择处理方案" value={profile} disabled={locked} onChange={(event) => setProfile(event.target.value as SynthesisProfile)}>
            <option value="vcp">VCP 默认</option><option value="vcp_joint">VCP 联合优化 · 工程可选</option>
          </select></label>}
        {(configuration?.profile ?? profile) === "vcp_joint" ? <p className="config-helper">联合方案优先保护双讲；启动期回声抑制和高信噪比场景仍需用实际录音确认。</p> : null}
        {executionIssue ? <p className="field-error" role="alert">{executionIssue}</p> : null}
      </div>
      {demos.length ? (
        <div className="synthesize-demos">
          <span className="synthesize-demos-label">默认示例</span>
          {demos.map((demo) => (
            <button
              key={demo.id}
              type="button"
              className={`synthesize-demo ${activeDemo === demo.id ? "active" : ""}`}
              disabled={locked}
              title={demo.description}
              onClick={() => loadDemo(demo)}
            >
              {operation?.demoId === demo.id ? <LoaderCircle size={13} className="spin" /> : null}
              {demo.label}
            </button>
          ))}
          <small className="config-helper">
            确定性合成诊断信号，用来验证链路是否通；不是真实语音或实车录音。
          </small>
        </div>
      ) : null}

      <div className="audio-upload-grid">
        {SLOTS.map((slot) => (
          <UploadCard
            key={slot.id}
            slot={slot}
            file={files[slot.id]}
            disabled={locked}
            onFile={(file) => {
              if (pending.current) return;
              setFiles((old) => ({ ...old, [slot.id]: file }));
              setActiveDemo("");
              setResult(null);
              setError("");
            }}
          />
        ))}
      </div>

      <div className="synthesize-input-controls">
        <label><span>麦克风处理方式</span><select aria-label="麦克风处理方式" value={microphoneMode} disabled={locked} onChange={(event) => {
          setMicrophoneMode(event.target.value as MicrophoneMode); setResult(null); setError("");
        }}><option value="channel0">使用第 1 麦（单麦 / 阵列旁路）</option><option value="beamform">使用标定阵列</option></select></label>
        {microphoneMode === "beamform" ? <label><span>逐通道标定延迟 / 目标采样率的样本数</span><input
          aria-label="逐通道标定延迟" type="text" value={delayText} disabled={locked} placeholder="例如 1.44, 0"
          onChange={(event) => { setDelayText(event.target.value); setResult(null); setError(""); }} /></label> : null}
      </div>
      <p className="config-helper">麦克风支持 1 / 2 / 4 路同步声道。多麦启用 BF 时请选择“使用标定阵列”并填写每路真实延迟；{configuration ? "只使用第 1 麦时需在上方旁路 BF。" : "单麦可直接使用第 1 麦。"}播放参考保留全部声道，不做平均。</p>
      <div className="synthesize-bar">
        <label>
          目标采样率
          <select aria-label="目标采样率" value={rate} disabled={locked} onChange={(e) => {
            if (pending.current) return;
            setRate(e.target.value);
            setResult(null);
            setError("");
          }}>
            {RATES.map((item) => (
              <option key={item.value} value={item.value}>
                {item.label}
              </option>
            ))}
          </select>
        </label>
        <small className="config-helper">
          引擎只实现 16 / 24 / 48 kHz 的 10 ms 帧；两路音频都会转换到该采样率，并按较短长度对齐。
        </small>
        <button className="button primary" disabled={locked || !ready || Boolean(executionIssue)} onClick={generate}>
          {busy ? <LoaderCircle size={16} className="spin" /> : <Sparkles size={16} />}
          {busy ? "生成中…" : "生成输出音频"}
        </button>
        {locked ? <button className="button secondary" onClick={cancel}>取消任务</button> : null}
      </div>
      {locked ? <p role="status" className="config-helper">{busy ? "正在处理本次输入…" : "正在加载示例音频…"}</p> : null}

      {error ? (
        <p className="audio-error" role="alert">
          <AlertTriangle size={16} />
          {error}
        </p>
      ) : null}

      {result && result.request.configuration_key === configurationKey ? (
        <div className="synthesize-result">
          <div className="synthesize-result-head">
            <div>
              <h3>处理后音频</h3>
              <span>
                {result.seconds.toFixed(2)} s · {result.sample_rate / 1000} kHz · 档位 {result.profile}
                {result.trimmed_seconds > 0.0005
                  ? ` · 已按较短输入对齐，截去 ${result.trimmed_seconds.toFixed(3)} s`
                  : ""}
              </span>
            </div>
            <button className="button secondary" onClick={() => downloadFile(
              JSON.stringify({ schema_version: 1, report_type: "synthesis", ...result }, null, 2),
              "NV200_音频生成报告.json", "application/json",
            )}><Download size={16} />生成报告 JSON</button>
            <a className="button secondary" href={result.download_url} download="itu-processed.wav">
              <Download size={16} />
              下载 WAV
            </a>
          </div>
          <p className="config-helper">本次输入：{result.request.inputs.microphone?.name} + {result.request.inputs.reference?.name}；
            目标采样率：{result.request.options.sample_rate ? `${result.request.options.sample_rate} Hz` : "自动"}；
            任务 {result.request.id}。</p>
          <p className="config-helper" data-testid="synthesis-input-channels">本次实际处理：档位 {result.profile}；
            麦克风 {result.inputs.microphone.processed_channels ?? 1} 路；播放参考 {result.inputs.reference.processed_channels ?? 1} 路（全部保留）。
            {result.request.options.microphone_mode === "beamform" ? "使用标定阵列。" : "使用第 1 麦。"}
          </p>
          <audio controls preload="metadata" src={result.download_url} aria-label="处理后音频试听" />

          <div className="synthesize-levels">
            <table className="audio-table">
              <thead>
                <tr>
                  <th>信号</th>
                  <th>RMS</th>
                  <th>峰值</th>
                  <th>满幅样本</th>
                </tr>
              </thead>
              <tbody>
                {(
                  [
                    ["麦克音", result.inputs.microphone],
                    ["参考音", result.inputs.reference],
                    ["输出", result.output],
                  ] as const
                ).map(([label, side]) => (
                  <tr key={label}>
                    <td>{label}</td>
                    <td>{side.rms_dbfs.toFixed(2)} dBFS</td>
                    <td>{side.peak_dbfs.toFixed(2)} dBFS</td>
                    <td>{side.clipping_percent.toFixed(3)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {result.charts?.length ? (
            <>
              {result.metrics ? (
                <section className="synthesize-chart">
                  <h4>指标</h4>
                  <span className="synthesize-chart-note">
                    {result.metrics.available ? result.metrics.scope : result.metrics.reason}
                  </span>
                  {result.metrics.available ? (
                    <div className="synthesize-levels">
                      <table className="audio-table">
                        <thead>
                          <tr><th>指标</th><th>数值</th><th>口径</th></tr>
                        </thead>
                        <tbody>
                          <tr>
                            <td>总电平变化</td>
                            <td>{(result.metrics.level_change_db ?? 0).toFixed(2)} dB</td>
                            <td>输出相对麦克音 RMS，全程</td>
                          </tr>
                          <tr>
                            <td>参考活动占比</td>
                            <td>{(result.metrics.reference_active_percent ?? 0).toFixed(1)}%</td>
                            <td>参考音高于 −50 dBFS 的 10 ms 帧</td>
                          </tr>
                          {result.metrics.quality?.available ? (
                            <>
                              <tr>
                                <td>SI-SDR</td>
                                <td>{(result.metrics.quality.si_sdr_db ?? 0).toFixed(2)} dB</td>
                                <td>干净近端最小二乘投影</td>
                              </tr>
                              <tr>
                                <td>STOI</td>
                                <td>{result.metrics.quality.stoi === null || result.metrics.quality.stoi === undefined ? "—" : result.metrics.quality.stoi.toFixed(4)}</td>
                                <td>pystoi · 16 kHz 内部分析</td>
                              </tr>
                            </>
                          ) : null}
                          {(result.metrics.band_change_db ?? []).map((band) => (
                            <tr key={band.low_hz}>
                              <td>{band.low_hz}–{Math.round(band.high_hz)} Hz</td>
                              <td>{band.change_db.toFixed(2)} dB</td>
                              <td>该频带输出相对输入</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : null}
                </section>
              ) : null}
              <section className="synthesize-chart">
                <h4>梅尔频谱图</h4>
                <span className="synthesize-chart-note">40 带 · 时间 × 梅尔频率 · 色标 −80…0 dB</span>
                <div className="waveform-list">
                  {result.charts.map((chart) => {
                    const mel = (chart as unknown as { mel?: { db: number[][]; frames: number } }).mel;
                    return (
                      <figure key={chart.role} style={{ margin: 0 }}>
                        <figcaption style={{ fontSize: "0.72rem", color: "var(--muted)" }}>
                          {CHART_LABEL[chart.role] ?? chart.role} · {mel ? `${mel.frames} 帧` : "无数据"}
                        </figcaption>
                        <canvas
                          ref={(el) => drawMel(el, mel)}
                          style={{ width: "100%", height: 180, imageRendering: "pixelated", borderRadius: 6 }}
                        />
                      </figure>
                    );
                  })}
                </div>
              </section>
              <section className="synthesize-chart">
                <h4>功率谱密度</h4>
                <span className="synthesize-chart-note">Welch PSD · dBFS/Hz · 按频段保留峰值用于绘图</span>
                <SpectrumPlot
                  unit="dBFS/Hz"
                  label="麦克音、参考音与输出的功率谱密度对比"
                  series={result.charts.map((chart) => ({
                    label: CHART_LABEL[chart.role] ?? chart.role,
                    color: ROLE_COLOR[chart.role],
                    x: chart.spectrum.frequency,
                    y: chart.spectrum.dbfs_hz,
                  }))}
                />
              </section>
              <section className="synthesize-chart">
                <h4>波形</h4>
                <span className="synthesize-chart-note">
                  完整输入 / 输出 · 输入图显示第 1 声道用于观察，参考处理保留全部声道 · 纵轴 ±1 FS
                </span>
                <div className="waveform-list">
                  {result.charts.map((chart) => (
                    <Waveform key={chart.role} file={chart} label={CHART_LABEL[chart.role] ?? chart.role} />
                  ))}
                </div>
              </section>
            </>
          ) : null}

          <section className="synthesize-chart synthesize-graphics" aria-label="本次音频的图片引擎分析">
            <div className="synthesize-result-head"><div><h3>本次输出对比图</h3><p className="config-helper">使用这次实际处理的输入和输出，按声明的算法延迟对齐，不归一化响度。多麦输入图观察第 1 麦，与阵列输出的差异包含波束形成影响。</p></div>
              {result.graphics_url ? <button className="button secondary" type="button" disabled={locked} onClick={generateGraphics}>
                {graphicsBusy ? <LoaderCircle size={15} className="spin" /> : <Sparkles size={15} />}
                {graphicsBusy ? "正在生成对比图…" : "生成本次对比图"}
              </button> : <p className="config-helper">本次结果未提供对比图入口。</p>}
            </div>
            {graphicsError ? <p className="audio-error" role="alert">{graphicsError}</p> : null}
            {graphics ? <>
              <p className="config-helper">{graphics.alignment.note}</p>
              {graphics.notices.map((notice) => <p key={notice} className="config-helper">{notice}</p>)}
              <div className="synthesize-graphics-grid">{graphics.figures.map((figure) => <figure key={figure.id} className="graph-figure">
                <div className="graph-figure-head"><h4>{figure.title}</h4><a className="text-link" href={figure.url} download={figure.download_name}>下载 PNG</a></div>
                <a href={figure.url} target="_blank" rel="noreferrer" aria-label={`打开本次${figure.title}原图`}><img src={figure.url} alt={figure.title} loading="lazy" /></a>
                <figcaption><p>{figure.description}</p></figcaption>
              </figure>)}</div>
              <button type="button" className="button secondary" onClick={() => downloadFile(JSON.stringify({ ...graphics, synthesis_request: result.request }, null, 2), "本次音频图形分析.json", "application/json")}>下载图形分析记录</button>
            </> : null}
          </section>

          <div className="audio-provenance">
            <Cpu size={15} />
            <span>
              引擎 {result.engine} · {result.binary}
            </span>
          </div>

          <details className="synthesize-report">
            <summary>引擎运行报告</summary>
            <dl>
              {Object.entries(result.engine_report).map(([key, value]) => (
                <div key={key}>
                  <dt>{REPORT_LABEL[key] ?? key}</dt>
                  <dd>{formatValue(key, value)}</dd>
                </div>
              ))}
            </dl>
          </details>
        </div>
      ) : null}
      </>}
    </div>
  );
}
