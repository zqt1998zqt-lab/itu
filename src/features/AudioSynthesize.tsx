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
import "./synthesize.css";

type Slot = "microphone" | "reference";

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
  selected_channel: number;
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
  download_url: string;
  created_at: string;
};

type Demo = { id: string; label: string; description: string };

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
    hint: "与麦克音同步的远端播放信号，作为回声参考。",
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
          <span>必需</span>
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
            <small className="preview-note">试听原文件；生成只使用第 1 声道。</small>
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

export default function AudioSynthesize() {
  const [files, setFiles] = useState<Record<Slot, File | null>>({ microphone: null, reference: null });
  const [rate, setRate] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<Result | null>(null);
  const [demos, setDemos] = useState<Demo[]>([]);
  const [demoBusy, setDemoBusy] = useState("");
  const [activeDemo, setActiveDemo] = useState("");
  const ready = Boolean(files.microphone && files.reference);

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
    setDemoBusy(demo.id);
    setError("");
    try {
      const fetched = await Promise.all(
        (["microphone", "reference"] as Slot[]).map(async (slot) => {
          const response = await fetch(`/api/synthesize/demo/${demo.id}/${slot}`);
          if (!response.ok) throw Error("示例音频暂不可用，请检查计算服务。");
          const label = SLOTS.find((item) => item.id === slot)?.label ?? slot;
          const blob = await response.blob();
          return [slot, new File([blob], `示例_${demo.label}_${label}.wav`, { type: "audio/wav" })] as const;
        }),
      );
      setFiles(Object.fromEntries(fetched) as Record<Slot, File | null>);
      setActiveDemo(demo.id);
      setResult(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "示例加载失败");
    } finally {
      setDemoBusy("");
    }
  }

  async function generate() {
    if (!files.microphone || !files.reference) {
      setError("请先选择麦克音与参考音。");
      return;
    }
    setBusy(true);
    setError("");
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 300000);
    try {
      const body = new FormData();
      body.append("microphone", files.microphone);
      body.append("reference", files.reference);
      body.append("options", JSON.stringify(rate ? { sample_rate: Number(rate) } : {}));
      const response = await fetch("/api/synthesize", { method: "POST", body, signal: controller.signal });
      const payload = await response.json().catch(() => null);
      if (!response.ok) {
        throw Error((payload && (payload.detail as string)) || `生成失败（HTTP ${response.status}）`);
      }
      setResult(payload as Result);
    } catch (e) {
      setError(
        e instanceof DOMException && e.name === "AbortError"
          ? "生成超时，请缩短音频后重试。"
          : e instanceof Error
            ? e.message
            : "生成失败",
      );
    } finally {
      clearTimeout(timer);
      setBusy(false);
    }
  }

  return (
    <div className="synthesize">
      {demos.length ? (
        <div className="synthesize-demos">
          <span className="synthesize-demos-label">默认示例</span>
          {demos.map((demo) => (
            <button
              key={demo.id}
              type="button"
              className={`synthesize-demo ${activeDemo === demo.id ? "active" : ""}`}
              disabled={busy || Boolean(demoBusy)}
              title={demo.description}
              onClick={() => loadDemo(demo)}
            >
              {demoBusy === demo.id ? <LoaderCircle size={13} className="spin" /> : null}
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
            disabled={busy}
            onFile={(file) => {
              setFiles((old) => ({ ...old, [slot.id]: file }));
              setResult(null);
              setError("");
            }}
          />
        ))}
      </div>

      <div className="synthesize-bar">
        <label>
          目标采样率
          <select value={rate} disabled={busy} onChange={(e) => setRate(e.target.value)}>
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
        <button className="button primary" disabled={busy || !ready} onClick={generate}>
          {busy ? <LoaderCircle size={16} className="spin" /> : <Sparkles size={16} />}
          {busy ? "生成中…" : "生成输出音频"}
        </button>
      </div>

      {error ? (
        <p className="audio-error" role="alert">
          <AlertTriangle size={16} />
          {error}
        </p>
      ) : null}

      {result ? (
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
            <a className="button secondary" href={result.download_url} download="itu-processed.wav">
              <Download size={16} />
              下载 WAV
            </a>
          </div>
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
              <section className="synthesize-chart">
                <h4>波形</h4>
                <span className="synthesize-chart-note">
                  完整文件 · 引擎实际处理的采样率与声道 · 纵轴 ±1 FS
                </span>
                <div className="waveform-list">
                  {result.charts.map((chart) => (
                    <Waveform key={chart.role} file={chart} label={CHART_LABEL[chart.role] ?? chart.role} />
                  ))}
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
            </>
          ) : null}

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
    </div>
  );
}
