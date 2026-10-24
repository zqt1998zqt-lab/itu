import { useEffect, useMemo, useState } from "react";
import {
  AudioLines,
  Download,
  LoaderCircle,
  RefreshCw,
} from "lucide-react";
import {
  AUDIO_GRAPH_FOCUS_FIGURES,
  AUDIO_GRAPH_FOCUS_LABELS,
  parseAudioGraphicReport,
} from "../domain/audio-graphics";
import type {
  AudioGraphicFocus,
  AudioGraphicReport,
  GraphicFigure,
  GraphicMetric,
} from "../domain/audio-graphics";
import { downloadFile } from "../lib/files";
import "./case-graph-engine.css";

const CATEGORY_LABELS: Record<GraphicFigure["category"], string> = {
  dynamics: "动态与波形",
  "time-frequency": "时频结构",
  comparison: "前后对比",
  bands: "频带与感知",
  transient: "瞬态细节",
};
const CACHE_MS = 20 * 60 * 1000;
const completed = new Map<string, { storedAt: number; report: AudioGraphicReport }>();
const pending = new Map<string, Promise<AudioGraphicReport>>();

function formatMetric(metric: GraphicMetric | undefined, digits = 2): string {
  if (!metric || metric.value === null) return "—";
  const signed = metric.value > 0 && ["rms_delta", "peak_delta", "spectral_median"].includes(metric.id);
  return `${signed ? "+" : ""}${metric.value.toFixed(digits)}`;
}

async function requestCaseReport(assetId: string): Promise<AudioGraphicReport> {
  const cached = completed.get(assetId);
  if (cached && Date.now() - cached.storedAt < CACHE_MS) return cached.report;
  const running = pending.get(assetId);
  if (running) return running;

  const task = (async () => {
    const response = await fetch(
      `/api/audio-graphics/evaluation-cases/${encodeURIComponent(assetId)}`,
      { method: "POST", cache: "no-store" },
    );
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      throw new Error(
        typeof payload?.detail === "string"
          ? payload.detail
          : "冻结案例图形生成失败，请检查计算服务。",
      );
    }
    const report = parseAudioGraphicReport(payload);
    if (report.evaluation_case?.asset_id !== assetId) {
      throw new Error("图形引擎返回的冻结案例身份不一致，已拒绝展示。");
    }
    completed.set(assetId, { storedAt: Date.now(), report });
    return report;
  })();
  pending.set(assetId, task);
  try {
    return await task;
  } finally {
    if (pending.get(assetId) === task) pending.delete(assetId);
  }
}

function EngineFigure({ figure, wide }: { figure: GraphicFigure; wide: boolean }) {
  return (
    <figure className={`case-engine-figure${wide ? " is-wide" : ""}`}>
      <div className="case-engine-figure-head">
        <div><span>{CATEGORY_LABELS[figure.category]}</span><strong>{figure.title}</strong></div>
        <a href={figure.url} download={figure.download_name} aria-label={`下载${figure.title} PNG`}>
          <Download size={13} />PNG
        </a>
      </div>
      <a href={figure.url} target="_blank" rel="noreferrer" aria-label={`打开${figure.title}原图`}>
        <img src={figure.url} alt={`${figure.title} · 冻结案例图形引擎结果`} loading="lazy" decoding="async" />
      </a>
      <figcaption><strong>{figure.description}</strong><small>{figure.detail}</small></figcaption>
    </figure>
  );
}

export default function CaseGraphEngine({
  assetId,
  beforeLabel,
  afterLabel,
  focus,
}: {
  assetId: string;
  beforeLabel: string;
  afterLabel: string;
  focus: AudioGraphicFocus;
}) {
  const [report, setReport] = useState<AudioGraphicReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);

  useEffect(() => {
    let active = true;
    setReport(null);
    setLoading(true);
    setError("");
    requestCaseReport(assetId)
      .then((next) => {
        if (!active) return;
        setReport(next);
      })
      .catch((reason: unknown) => {
        if (!active) return;
        setReport(null);
        setError(reason instanceof Error ? reason.message : "冻结案例图形生成失败");
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => { active = false; };
  }, [assetId, retry]);

  const visibleFigures = useMemo(
    () => {
      if (!report) return [];
      const byId = new Map(report.figures.map((figure) => [figure.id, figure]));
      return AUDIO_GRAPH_FOCUS_FIGURES[focus]
        .map((id) => byId.get(id))
        .filter((figure): figure is GraphicFigure => Boolean(figure));
    },
    [report, focus],
  );
  const metric = (id: string) => report?.metrics.find((item) => item.id === id);
  const evaluationCase = report?.evaluation_case;

  return (
    <section className="case-engine" aria-busy={loading}>
      <div className="case-engine-head">
        <div>
          <span>AUDIO GRAPHICS ENGINE · FROZEN PAIR</span>
          <h4>典型案例图形引擎</h4>
          <p>直接读取本案例冻结的输入/输出 WAV，校验 SHA-256 后按清单窗口和固定延迟生成图形。</p>
        </div>
        {report ? (
          <div className="case-engine-count">
            <AudioLines size={15} />
            <strong>{visibleFigures.length}</strong>
            <span>{focus === "all" ? "张全部图" : "张关键图"}</span>
          </div>
        ) : null}
      </div>

      {loading ? (
        <div className="case-engine-state" role="status" aria-live="polite">
          <LoaderCircle className="spin" size={18} />
          <div><strong>正在调用图形引擎</strong><span>校验案例文件、裁剪固定延迟并生成 PNG…</span></div>
        </div>
      ) : error ? (
        <div className="case-engine-state is-error" role="alert">
          <RefreshCw size={18} />
          <div><strong>图形引擎结果暂不可用</strong><span>{error}</span></div>
          <button type="button" className="button secondary" onClick={() => setRetry((value) => value + 1)}>重试</button>
        </div>
      ) : report ? (
        <>
          <div className="case-engine-identity">
            <div><span>案例身份</span><code>{evaluationCase?.case_id ?? "—"}</code></div>
            <div><span>固定窗口</span><strong>{evaluationCase ? `${evaluationCase.window_seconds[0].toFixed(1)}–${evaluationCase.window_seconds[1].toFixed(1)} s` : "—"}</strong></div>
            <div><span>固定延迟</span><strong>{report.alignment.delay_ms === null ? "—" : `${report.alignment.delay_ms.toFixed(3)} ms · ${evaluationCase?.delay_samples ?? "—"} samples`}</strong></div>
            <div><span>分析率</span><strong>{report.analysis_rate / 1000} kHz</strong></div>
          </div>
          <div className="case-engine-metrics">
            <div><span>RMS 变化</span><strong>{formatMetric(metric("rms_delta"))}<small>dB</small></strong></div>
            <div><span>峰值变化</span><strong>{formatMetric(metric("peak_delta"))}<small>dB</small></strong></div>
            <div><span>谱变化中位数</span><strong>{formatMetric(metric("spectral_median"))}<small>dB</small></strong></div>
            <div><span>衰减时频单元</span><strong>{formatMetric(metric("spectral_reduced"), 1)}<small>%</small></strong></div>
          </div>
          <div className="case-engine-source-pair">
            <div><span>INPUT</span><strong>{beforeLabel}</strong><code>{report.source.find((item) => item.role === "before")?.name}</code></div>
            <div><span>OUTPUT</span><strong>{afterLabel}</strong><code>{report.source.find((item) => item.role === "after")?.name}</code></div>
          </div>
          <div className="case-engine-gallery-head">
            <div>
              <strong>{AUDIO_GRAPH_FOCUS_LABELS[focus]}</strong>
              <span>
                {visibleFigures.length === report.figures.length
                  ? `全部 ${visibleFigures.length} 张图`
                  : `从 ${report.figures.length} 张引擎图中选取 ${visibleFigures.length} 张`}
                {" "}· 图像由本次冻结音频即时生成
              </span>
            </div>
          </div>
          <div className="case-engine-gallery">
            {visibleFigures.map((figure) => (
              <EngineFigure
                key={figure.id}
                figure={figure}
                wide={figure.id === "triple" || (focus === "all" && figure.id === "high-time")}
              />
            ))}
          </div>
          <div className="case-engine-footer">
            <p>图形用于定位波形、频带和时频变化；PASS / FAIL 仍来自该组合自己的整链工程门限。图内 dB 变化不替代 ΔSI-SDR、ΔSTOI 或工程判定。</p>
            <div>
              <button type="button" className="button secondary" onClick={() => downloadFile(JSON.stringify(report, null, 2), `${assetId}-audio-graphics.json`, "application/json")}><Download size={14} />分析 JSON</button>
              <a className="text-link" href="#graphs">打开完整图形引擎</a>
            </div>
          </div>
          <details className="case-engine-provenance">
            <summary>查看文件哈希、窗口与冻结清单</summary>
            {report.source.map((source) => <div key={source.role}><strong>{source.role}</strong><code>{source.sha256}</code></div>)}
            <p>{evaluationCase?.manifest}</p>
            {report.notices.map((notice) => <p key={notice}>{notice}</p>)}
          </details>
        </>
      ) : null}
    </section>
  );
}
