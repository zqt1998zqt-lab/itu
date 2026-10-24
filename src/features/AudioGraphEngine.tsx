import { useMemo, useState } from "react";
import {
  AudioLines,
  Download,
  FileAudio,
  FlaskConical,
  Layers3,
  LoaderCircle,
  ScanSearch,
} from "lucide-react";
import {
  parseAudioGraphicReport,
} from "../domain/audio-graphics";
import type {
  AudioGraphicReport,
  GraphicFigure,
  GraphicMetric,
} from "../domain/audio-graphics";
import { downloadFile } from "../lib/files";
import "./audio-graph-engine.css";

const CATEGORY_LABELS: Record<string, string> = {
  all: "全部图形",
  dynamics: "动态与波形",
  "time-frequency": "时频结构",
  comparison: "前后对比",
  bands: "频带与感知",
  transient: "瞬态细节",
};

const WIDE_FIGURES = new Set(["triple"]);

function formatMetric(metric: GraphicMetric | undefined, digits = 2): string {
  if (!metric || metric.value === null) return "—";
  const signed = ["rms_delta", "peak_delta", "spectral_median"].includes(metric.id) && metric.value > 0;
  return `${signed ? "+" : ""}${metric.value.toFixed(digits)}`;
}

function FileSlot({
  id,
  label,
  hint,
  file,
  required,
  onFile,
}: {
  id: string;
  label: string;
  hint: string;
  file: File | null;
  required?: boolean;
  onFile: (file: File | null) => void;
}) {
  return (
    <label className={`graph-file${file ? " has-file" : ""}`} htmlFor={id}>
      <span className="graph-file-icon"><FileAudio size={20} /></span>
      <span className="graph-file-copy">
        <small>{required ? "REQUIRED AUDIO" : "OPTIONAL COMPARISON AUDIO"}</small>
        <strong>{label}</strong>
        <span>{file ? file.name : hint}</span>
      </span>
      <span className="graph-file-action">{file ? "更换" : "选择"}</span>
      <input
        id={id}
        type="file"
        accept=".wav,.flac,.mp3,.ogg,.m4a,.aac,.aif,.aiff,audio/*"
        onChange={(event) => onFile(event.target.files?.[0] ?? null)}
      />
    </label>
  );
}

function FigureCard({ figure }: { figure: GraphicFigure }) {
  return (
    <figure className={`graph-figure${WIDE_FIGURES.has(figure.id) ? " is-wide" : ""}`}>
      <div className="graph-figure-head">
        <div><span>{CATEGORY_LABELS[figure.category]}</span><h3>{figure.title}</h3></div>
        <a className="button secondary" href={figure.url} download={figure.download_name}>
          <Download size={14} />PNG
        </a>
      </div>
      <a className="graph-figure-image" href={figure.url} target="_blank" rel="noreferrer" aria-label={`打开${figure.title}原图`}>
        <img src={figure.url} alt={figure.title} loading="lazy" />
      </a>
      <figcaption><p>{figure.description}</p><small>{figure.detail}</small></figcaption>
    </figure>
  );
}

export default function AudioGraphEngine() {
  const [beforeFile, setBeforeFile] = useState<File | null>(null);
  const [afterFile, setAfterFile] = useState<File | null>(null);
  const [rate, setRate] = useState("16000");
  const [beforeChannel, setBeforeChannel] = useState("1");
  const [afterChannel, setAfterChannel] = useState("1");
  const [autoAlign, setAutoAlign] = useState(true);
  const [report, setReport] = useState<AudioGraphicReport | null>(null);
  const [category, setCategory] = useState("all");
  const [busy, setBusy] = useState(false);
  const [loadingDemo, setLoadingDemo] = useState(false);
  const [error, setError] = useState("");

  const categories = useMemo(() => {
    if (!report) return ["all"];
    return ["all", ...new Set(report.figures.map((figure) => figure.category))];
  }, [report]);
  const figures = report?.figures.filter((figure) => category === "all" || figure.category === category) ?? [];
  const metric = (id: string) => report?.metrics.find((item) => item.id === id);
  const summaryMetrics = report
    ? report.source.some((source) => source.role === "before")
      ? [
          ["rms_delta", "RMS 变化", "dB"],
          ["peak_delta", "峰值变化", "dB"],
          ["spectral_median", "谱变化中位数", "dB"],
          ["spectral_reduced", "衰减时频单元", "%"],
        ]
      : [
          ["rms_after", "RMS", "dBFS"],
          ["peak_after", "峰值", "dBFS"],
          ["crest_after", "峰值因数", "dB"],
          ["clipping_after", "满幅比例", "%"],
        ]
    : [];

  const reset = () => {
    setReport(null);
    setCategory("all");
    setError("");
  };

  const loadDemo = async () => {
    setLoadingDemo(true);
    setError("");
    try {
      const [beforeResponse, afterResponse] = await Promise.all([
        fetch("/api/demo/before"),
        fetch("/api/demo/after"),
      ]);
      if (!beforeResponse.ok || !afterResponse.ok) throw new Error("示例音频暂不可用，请检查计算服务。");
      setBeforeFile(new File([await beforeResponse.blob()], "示例_处理前.wav", { type: "audio/wav" }));
      setAfterFile(new File([await afterResponse.blob()], "示例_处理后.wav", { type: "audio/wav" }));
      reset();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "示例音频载入失败");
    } finally {
      setLoadingDemo(false);
    }
  };

  const analyze = async () => {
    if (!afterFile) {
      setError("请先选择处理后音频；处理前音频可用于前后差分。");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const form = new FormData();
      if (beforeFile) form.append("before", beforeFile);
      form.append("after", afterFile);
      form.append("options", JSON.stringify({
        sample_rate: Number(rate),
        before_channel: Number(beforeChannel) - 1,
        after_channel: Number(afterChannel) - 1,
        align: autoAlign,
        max_delay_ms: 1000,
      }));
      const response = await fetch("/api/audio-graphics", { method: "POST", body: form });
      const payload = await response.json().catch(() => null);
      if (!response.ok) {
        throw new Error(typeof payload?.detail === "string" ? payload.detail : "图形生成失败，请检查音频和参数。");
      }
      setReport(parseAudioGraphicReport(payload));
      setCategory("all");
    } catch (reason) {
      setReport(null);
      setError(reason instanceof Error ? reason.message : "图形生成失败");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="nsp-page graph-engine-page">
      <section className="graph-intro">
        <div>
          <span>AUDIO GRAPHICS ENGINE · PAIR OR SINGLE FILE</span>
          <h2>把波形、频谱和增强变化拆成可审计的图</h2>
          <p>上传处理后音频，可选上传处理前音频。引擎自动对齐、统一色标并生成对数 STFT、差分频谱、LTAS、ERB、短时能量和瞬态视图。</p>
        </div>
        <div className="graph-summary">
          <div><strong>{report?.source.length ?? "—"}</strong><span>分析音频</span></div>
          <div><strong>{report?.figures.length ?? "—"}</strong><span>生成图形</span></div>
          <div><strong>{report ? report.duration_seconds.toFixed(2) : "—"}</strong><span>有效时长 / s</span></div>
          <div><strong>{report ? report.analysis_rate / 1000 : "—"}</strong><span>分析采样率 / kHz</span></div>
        </div>
      </section>

      <section className="graph-config">
        <div className="graph-section-head">
          <div><span>INPUT</span><h2>音频与计算条件</h2><p>单文件可直接生成频谱；处理前/处理后配对时会生成差分和三联图。文件最大 32 MB、时长 0.1–120 秒。</p></div>
          <button type="button" className="button secondary" onClick={loadDemo} disabled={busy || loadingDemo}>
            {loadingDemo ? <LoaderCircle className="spin" size={15} /> : <FlaskConical size={15} />}
            载入配对示例
          </button>
        </div>
        <div className="graph-file-grid">
          <FileSlot id="graph-before-file" label="处理前音频" hint="可选；用于差分、LTAS 和前后共享标尺" file={beforeFile}
            onFile={(file) => { setBeforeFile(file); reset(); }} />
          <FileSlot id="graph-after-file" label="处理后音频" hint="必需；单文件也可生成频谱" file={afterFile} required
            onFile={(file) => { setAfterFile(file); reset(); }} />
        </div>
        <div className="graph-controls">
          <label><span>分析采样率</span><select value={rate} onChange={(event) => setRate(event.target.value)}>
            <option value="8000">8 kHz</option><option value="16000">16 kHz</option><option value="24000">24 kHz</option>
            <option value="32000">32 kHz</option><option value="48000">48 kHz</option>
          </select></label>
          <label><span>处理前声道</span><input type="number" min="1" max="8" value={beforeChannel} onChange={(event) => setBeforeChannel(event.target.value)} /></label>
          <label><span>处理后声道</span><input type="number" min="1" max="8" value={afterChannel} onChange={(event) => setAfterChannel(event.target.value)} /></label>
          <label className="graph-switch"><input type="checkbox" checked={autoAlign} onChange={(event) => setAutoAlign(event.target.checked)} /><span><strong>自动对齐</strong><small>归一化跨相关，最多 1000 ms；不可靠时不伪造对齐。</small></span></label>
        </div>
        <div className="graph-actions">
          <div>{afterFile ? <><AudioLines size={16} /><span>已选择 {beforeFile ? "配对音频" : "单文件"}</span></> : <><Layers3 size={16} /><span>等待处理后音频</span></>}</div>
          <button type="button" className="button primary" disabled={busy || !afterFile} onClick={analyze}>
            {busy ? <LoaderCircle className="spin" size={17} /> : <ScanSearch size={17} />}
            {busy ? "正在生成图形…" : "生成图形与分析"}
          </button>
        </div>
        {error ? <div className="graph-error" role="alert">{error}</div> : null}
      </section>

      {report ? <>
        <section className="graph-analysis">
          <div className="graph-section-head">
            <div><span>ANALYSIS SUMMARY</span><h2>自动分析结果</h2><p>对齐、电平、削波、频谱变化和频带能量均来自本次上传文件；不是通用质量评分。</p></div>
            <button type="button" className="button secondary" onClick={() => downloadFile(JSON.stringify(report, null, 2), "audio-graphics-analysis.json", "application/json")}><Download size={14} />分析 JSON</button>
          </div>
          <div className="graph-metric-grid">{summaryMetrics.map(([id, label, unit]) => {
            const value = metric(id);
            return <article key={id} className={`graph-metric status-${value?.status ?? "unavailable"}`}><span>{label}</span><strong>{formatMetric(value, id === "spectral_reduced" ? 1 : 2)}<small>{unit}</small></strong><p>{value?.detail ?? "当前输入没有可用值。"}</p></article>;
          })}</div>
          <div className="graph-alignment">
            <div><span>ALIGNMENT</span><strong>{report.alignment.reliable === null ? "未请求" : report.alignment.applied ? "已对齐" : "未应用"}</strong></div>
            <dl>
              <div><dt>相对延迟</dt><dd>{report.alignment.delay_ms === null ? "—" : `${report.alignment.delay_ms.toFixed(3)} ms`}</dd></div>
              <div><dt>相关系数</dt><dd>{report.alignment.correlation === null ? "—" : report.alignment.correlation.toFixed(4)}</dd></div>
              <div><dt>STFT</dt><dd>{report.figure_scope.stft_fft_size} FFT / {report.figure_scope.stft_hop} hop</dd></div>
              <div><dt>瞬态视图</dt><dd>{report.figure_scope.high_time_fft_size} FFT / {report.figure_scope.high_time_hop} hop</dd></div>
            </dl>
          </div>
          {report.bands.length ? <div className="graph-band-wrap"><table className="graph-band-table"><thead><tr><th>频带</th><th>处理前</th><th>处理后</th><th>变化</th></tr></thead><tbody>
            {report.bands.map((band) => <tr key={band.id}><th>{band.label}</th><td>{band.before_db.toFixed(2)} dB</td><td>{band.after_db.toFixed(2)} dB</td><td className={band.delta_db < 0 ? "is-reduced" : "is-boosted"}>{band.delta_db > 0 ? "+" : ""}{band.delta_db.toFixed(2)} dB</td></tr>)}
          </tbody></table></div> : null}
        </section>

        <section className="graph-figures-section">
          <div className="graph-section-head">
            <div><span>GENERATED FIGURES</span><h2>图形分析</h2><p>{report.figures.length} 张图共享本次文件的身份与时间条件；点击图像可查看原图。</p></div>
            <div className="graph-filters" role="group" aria-label="图形分类">
              {categories.map((item) => <button type="button" key={item} className={category === item ? "is-active" : ""} onClick={() => setCategory(item)}>{CATEGORY_LABELS[item]}</button>)}
            </div>
          </div>
          <div className="graph-figure-grid">{figures.map((figure) => <FigureCard key={figure.id} figure={figure} />)}</div>
        </section>

        <details className="graph-source">
          <summary>文件身份、图形参数与分析边界</summary>
          {report.source.map((source) => <div key={source.role}><strong>{source.name}</strong><code>{source.format} · {source.original_sample_rate / 1000} kHz · {source.original_channels} ch · 使用通道 {source.selected_channel} · SHA256 {source.sha256}</code></div>)}
          {report.notices.map((notice) => <p key={notice}>{notice}</p>)}
        </details>
      </> : null}
    </div>
  );
}
