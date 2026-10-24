import { useEffect, useRef, useState } from "react";
import { AlertTriangle, CheckCircle2, ChevronLeft, ChevronRight, Download, Headphones, Search } from "lucide-react";
import "./ns-failure-audio.css";

type Failure = { label: string; window: string; metric: string; value: number | null; input_value: number | null; unit: string; requirement: string };
type Backend = { variant: string; delay_samples: number; output_sha256: string; failures: Failure[] };
type Sample = {
  id: string; scope: string; original_run: string; scene_id: string; noise: string;
  sample_rate: number; snr_db: number; split: string; speaker: string; duration_s: number;
  speech_failure: boolean; input_crop_samples: number;
  windows: { id: string; start_s: number; end_s: number }[];
  backends: Record<"omlsa" | "super_gaussian", Backend>;
  audio: Record<"input" | "omlsa" | "super_gaussian" | "clean", string>;
};
type Catalog = { total: number; scope_counts: Record<string, number>; items: Sample[]; comparison: string; audio_policy: string };
const PAGE_SIZE = 6;
const SCOPES: Record<string, string> = { steady: "稳态噪声", transient: "瞬时噪声", cold: "冷启动", clean: "纯净语音" };
const WINDOWS: Record<string, string> = {
  near_only: "前段语音", near_late: "后段语音", near_startup: "启动语音",
  noise_only: "纯噪声", noise_only_before_step: "噪声阶跃前", noise_only_after_step: "噪声阶跃后",
};
const METRICS: Record<string, string> = { attenuation: "噪声衰减", projection_gain: "语音投影增益", stoi: "STOI 变化", sisdr: "SI-SDR", raw_sisdr: "SI-SDR", clipping: "削波" };
const NOISES: Record<string, string> = {
  tcar: "TCAR 车内低频噪声", sim_engine_ramp: "发动机渐变噪声", sim_indicator: "转向灯",
  sim_wind_gust: "阵风", sim_rain_impulse: "雨滴脉冲",
  cavemove_alfaromeo_146_s60_w0: "CAVEMOVE · Alfa Romeo", cavemove_smart_forfour_s60_w0: "CAVEMOVE · Smart",
  cavemove_volkswagen_golf_s60_w0: "CAVEMOVE · Golf",
};
function valueText(value: number | null, unit = "") {
  return value === null || !Number.isFinite(value) ? "未提供" : `${value.toFixed(unit ? 2 : 4)}${unit ? ` ${unit}` : ""}`;
}
function windowText(id: string) { return WINDOWS[id] ?? id; }

function AudioPlayer({ src, title, sampleId, onPlay }: { src: string; title: string; sampleId: string; onPlay: (player: HTMLAudioElement) => void }) {
  const [error, setError] = useState(false);
  const audio = useRef<HTMLAudioElement>(null);
  useEffect(() => {
    const player = audio.current;
    return () => { player?.pause(); };
  }, []);
  return <>
    <audio ref={audio} controls preload="none" src={src} aria-label={`${sampleId} ${title}`}
      onPlay={(event) => onPlay(event.currentTarget)} onError={() => setError(true)}>
      您的浏览器不支持音频播放，请下载 WAV。
    </audio>
    <div className="nsf-player-footer">
      <span>原幅度 · WAV</span>
      <a href={`${src}${src.includes("?") ? "&" : "?"}download=true`} download aria-label={`下载 ${sampleId} ${title}`}>
        <Download size={13} aria-hidden="true" /> 下载
      </a>
    </div>
    {error ? <p className="nsf-player-error" role="alert">音频加载失败，请重试或下载文件。</p> : null}
  </>;
}

function FailureCard({ sample, number, onPlay }: { sample: Sample; number: number; onPlay: (player: HTMLAudioElement) => void }) {
  const [windowId, setWindowId] = useState("full");
  const selected = sample.windows.find((window) => window.id === windowId);
  const url = (role: keyof Sample["audio"]) => `${sample.audio[role]}?window=${encodeURIComponent(windowId)}`;
  return <article className="nsf-case" aria-labelledby={`case-${sample.id}`}>
    <header className="nsf-case-header">
      <div className="nsf-case-title">
        <span className="nsf-case-number">{String(number).padStart(3, "0")}</span>
        <h3 id={`case-${sample.id}`}>{NOISES[sample.noise] ?? sample.noise}</h3>
      </div>
      <span className={`nsf-failure-kind ${sample.speech_failure ? "speech" : "noise"}`}>
        {sample.speech_failure ? "含语音指标失败" : "仅噪声衰减不足"}
      </span>
    </header>
    <div className="nsf-case-tags">
      <span>{SCOPES[sample.scope]}</span><span>{sample.sample_rate / 1000} kHz</span>
      <span>SNR {sample.snr_db > 0 ? "+" : ""}{sample.snr_db} dB</span>
      <span>{sample.duration_s.toFixed(2)} s</span><span>{sample.split}</span>
      {sample.speaker ? <span>说话人 {sample.speaker}</span> : null}
    </div>
    <p className="nsf-scene-id">{sample.original_run} / {sample.scene_id}</p>
    <div className="nsf-window-bar">
      <span>试听区间</span>
      <div className="nsf-window-buttons" role="group" aria-label={`${sample.scene_id} 试听区间`}>
        <button type="button" aria-pressed={windowId === "full"} onClick={() => setWindowId("full")}>整段</button>
        {sample.windows.map((window) => <button type="button" key={window.id} aria-pressed={windowId === window.id} onClick={() => setWindowId(window.id)}>
          {windowText(window.id)} · {window.start_s}–{window.end_s} s
        </button>)}
      </div>
      <small>{selected ? `播放原时间轴 ${selected.start_s}–${selected.end_s} s` : "播放完整测试音频"}</small>
    </div>
    <div className="nsf-audio-grid">
      <section className="nsf-player input">
        <h4>输入 · 带噪音频</h4>
        <p>两个算法使用同一份输入</p>
        <AudioPlayer key={url("input")} src={url("input")} title="带噪输入" sampleId={sample.scene_id} onPlay={onPlay} />
        <p className="nsf-alignment">{sample.input_crop_samples ? `冷启动已裁去原音频前 ${(sample.input_crop_samples / sample.sample_rate * 1000).toFixed(0)} ms` : "与评测输入一致的 PCM16"}</p>
      </section>
      {(["omlsa", "super_gaussian"] as const).map((role) => {
        const backend = sample.backends[role];
        return <section key={role} className={`nsf-player ${role}`}>
          <h4>{role === "omlsa" ? "OM-LSA + IMCRA" : "Super-Gaussian"} <span>平衡档</span></h4>
          <p>已扣除 {(backend.delay_samples / sample.sample_rate * 1000).toFixed(0)} ms 固定算法延时</p>
          <AudioPlayer key={url(role)} src={url(role)} title={role === "omlsa" ? "OM-LSA 降噪后" : "Super-Gaussian 降噪后"} sampleId={sample.scene_id} onPlay={onPlay} />
          <ul className="nsf-metrics">
            {backend.failures.map((failure) => <li key={failure.label} className={windowId !== "full" && failure.window !== windowId ? "other-window" : ""}>
              <div><span>{windowText(failure.window)} · {METRICS[failure.metric] ?? failure.metric}</span><strong>{valueText(failure.value, failure.unit)}</strong></div>
              <small>要求 {failure.requirement}{failure.input_value !== null ? ` · 输入 ${valueText(failure.input_value, failure.unit)}` : ""}</small>
            </li>)}
          </ul>
        </section>;
      })}
    </div>
    <details className="nsf-evidence">
      <summary>干净语音参考与结果来源</summary>
      <div className="nsf-reference">
        <div><h4>干净近端参考</h4><p>用于听辨语音损伤；纯噪声窗口内可能静音。</p></div>
        <AudioPlayer key={url("clean")} src={url("clean")} title="干净语音参考" sampleId={sample.scene_id} onPlay={onPlay} />
      </div>
      <p>冻结结果重放已通过逐文件 SHA-256 校验。试听仅裁剪固定延时和所选窗口，不调节增益、不重采样。</p>
      {(["omlsa", "super_gaussian"] as const).map((role) => <p key={role} className="nsf-hash">{sample.backends[role].variant} · SHA-256 {sample.backends[role].output_sha256}</p>)}
    </details>
  </article>;
}

export default function NsFailureAudio() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  const [scope, setScope] = useState("");
  const [query, setQuery] = useState("");
  const [rate, setRate] = useState("");
  const [noise, setNoise] = useState("");
  const [kind, setKind] = useState("");
  const [page, setPage] = useState(1);
  const container = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const controller = new AbortController();
    setError("");
    fetch("/api/ns-failures", { signal: controller.signal }).then(async (response) => {
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "失败样本加载失败");
      setCatalog(data as Catalog);
    }).catch((reason) => {
      if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : "失败样本加载失败");
    });
    return () => controller.abort();
  }, [retry]);
  function pauseOthers(player: HTMLAudioElement) {
    container.current?.querySelectorAll("audio").forEach((other) => { if (other !== player) other.pause(); });
  }
  function changeFilter(setter: (value: string) => void, value: string) { setter(value); setPage(1); }
  function reset() { setScope(""); setQuery(""); setRate(""); setNoise(""); setKind(""); setPage(1); }
  if (error) return <div className="nsf-state" role="alert"><AlertTriangle size={22} /><p>{error}</p><button className="button" onClick={() => setRetry(retry + 1)}>重新加载</button></div>;
  if (!catalog) return <div className="nsf-state" role="status">正在加载共同失败样本与音频索引…</div>;
  const needle = query.trim().toLowerCase();
  const filtered = catalog.items.filter((item) => (!scope || item.scope === scope)
    && (!rate || item.sample_rate === Number(rate)) && (!noise || item.noise === noise)
    && (!kind || (kind === "speech" ? item.speech_failure : !item.speech_failure))
    && (!needle || [item.scene_id, item.noise, NOISES[item.noise] ?? "", item.speaker, item.original_run].join(" ").toLowerCase().includes(needle)));
  const pages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const shownPage = Math.min(page, pages);
  const start = (shownPage - 1) * PAGE_SIZE;
  return <div className="nsf" ref={container}>
    <section className="nsf-overview">
      <div className="nsf-overview-title"><Headphones size={22} aria-hidden="true" /><h2>共同失败样本试听</h2><span><CheckCircle2 size={14} aria-hidden="true" /> 已验证的真实算法输出</span></div>
      <p>逐项比较带噪输入与两个算法的降噪结果。可切换到失败窗口，结合原始指标判断噪声残留和语音损伤。</p>
      <div className="nsf-scope-filters" role="group" aria-label="测试类型筛选">
        <button type="button" aria-pressed={!scope} onClick={() => changeFilter(setScope, "")}>全部共同失败 <strong>{catalog.total}</strong></button>
        {["steady", "transient", "cold"].map((key) => <button type="button" key={key} aria-pressed={scope === key} onClick={() => changeFilter(setScope, key)}>{SCOPES[key]} <strong>{catalog.scope_counts[key] ?? 0}</strong></button>)}
      </div>
      <small>冷启动为同一原始场景的派生测试。瞬时噪声残留与语音受损分别标示，原失败门槛保持不变。</small>
    </section>
    <section className="nsf-filters" aria-label="失败样本筛选">
      <label className="nsf-search"><span>搜索样本</span><div><Search size={16} aria-hidden="true" /><input type="search" value={query} placeholder="噪声、说话人或场景 ID" onChange={(event) => changeFilter(setQuery, event.target.value)} /></div></label>
      <label><span>采样率</span><select value={rate} onChange={(event) => changeFilter(setRate, event.target.value)}><option value="">全部采样率</option>{[16000, 24000, 48000].map((r) => <option key={r} value={r}>{r / 1000} kHz</option>)}</select></label>
      <label><span>噪声来源</span><select value={noise} onChange={(event) => changeFilter(setNoise, event.target.value)}><option value="">全部噪声</option>{[...new Set(catalog.items.map((item) => item.noise))].sort().map((n) => <option key={n} value={n}>{NOISES[n] ?? n}</option>)}</select></label>
      <label><span>失败类型</span><select value={kind} onChange={(event) => changeFilter(setKind, event.target.value)}><option value="">全部失败</option><option value="speech">含语音指标失败</option><option value="noise">仅噪声衰减不足</option></select></label>
    </section>
    <div className="nsf-list-heading"><p role="status">找到 <strong>{filtered.length}</strong> 项{filtered.length ? ` · 当前 ${start + 1}–${Math.min(start + PAGE_SIZE, filtered.length)} 项` : ""}</p><button type="button" className="text-link" onClick={reset}>重置筛选</button></div>
    {filtered.length ? <div className="nsf-cases">{filtered.slice(start, start + PAGE_SIZE).map((sample, index) => <FailureCard key={sample.id} sample={sample} number={start + index + 1} onPlay={pauseOthers} />)}</div>
      : <div className="nsf-state"><Search size={24} aria-hidden="true" /><p>没有匹配的失败样本，请调整筛选条件。</p><button className="button" onClick={reset}>显示全部样本</button></div>}
    <nav className="nsf-pagination" aria-label="失败样本分页">
      <button type="button" disabled={shownPage <= 1} onClick={() => setPage(shownPage - 1)}><ChevronLeft size={16} aria-hidden="true" />上一页</button>
      <span>第 {shownPage} / {pages} 页</span>
      <button type="button" disabled={shownPage >= pages} onClick={() => setPage(shownPage + 1)}>下一页<ChevronRight size={16} aria-hidden="true" /></button>
    </nav>
  </div>;
}
