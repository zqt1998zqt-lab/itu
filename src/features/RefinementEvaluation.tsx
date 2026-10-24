import { useEffect, useState } from "react";
import { ArrowRight, LoaderCircle, Microscope } from "lucide-react";
import { jointNextFigureTitle } from "./joint-next-model";
import "./joint-next-evaluation.css";

const PREFIX = "/evaluation/vcp-joint-refine/";
const REPORT = `${PREFIX}report.json`;
type Evidence = { label: string; url: string; sha256: string };
type RefinementReport = {
  schema_version: 1; title: string; summary: string; scope: string;
  status: "diagnostic" | "rejected" | "admitted";
  final_admission?: { status: "admitted"; checks: Record<string, boolean> };
  candidates: { id: string; label: string; stage: string; cases: number; verdict: string;
    metrics: { label: string; value: string }[] }[];
  findings: { title: string; detail: string }[];
  limitations: string[];
  figures: { scene_id: string; window: string; label: string;
    items: { id: string; url: string; sha256: string }[] }[];
  artifacts: Evidence[];
};

function parse(value: unknown): RefinementReport {
  const obj = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object" && !Array.isArray(v);
  const str = (v: unknown): v is string => typeof v === "string" && v.trim().length > 0;
  const hash = (v: unknown) => typeof v === "string" && /^[a-f\d]{64}$/i.test(v);
  const url = (v: unknown) => {
    if (typeof v !== "string" || !v.startsWith(PREFIX) || /\\|%2e|%2f|%5c/i.test(v)) return false;
    const parsed = new URL(v, "https://evidence.invalid");
    return parsed.origin === "https://evidence.invalid" && parsed.pathname.startsWith(PREFIX);
  };
  if (!obj(value) || value.schema_version !== 1 || !str(value.title) || !str(value.summary) || !str(value.scope) ||
      !["diagnostic", "rejected", "admitted"].includes(String(value.status)) ||
      (value.status === "admitted" && (!obj(value.final_admission) || value.final_admission.status !== "admitted" ||
        !obj(value.final_admission.checks) || !Object.keys(value.final_admission.checks).length ||
        !Object.values(value.final_admission.checks).every((check) => check === true))) ||
      !Array.isArray(value.candidates) || !value.candidates.length || !value.candidates.every((c: unknown) =>
        obj(c) && str(c.id) && str(c.label) && str(c.stage) && Number.isInteger(c.cases) && Number(c.cases) >= 0 &&
        str(c.verdict) && Array.isArray(c.metrics) && c.metrics.every((m: unknown) => obj(m) && str(m.label) && str(m.value))) ||
      !Array.isArray(value.findings) || !value.findings.length || !value.findings.every((f: unknown) => obj(f) && str(f.title) && str(f.detail)) ||
      !Array.isArray(value.limitations) || !value.limitations.length || !value.limitations.every(str) ||
      !Array.isArray(value.figures) || !value.figures.length || !value.figures.every((f: unknown) =>
        obj(f) && str(f.scene_id) && str(f.window) && str(f.label) && Array.isArray(f.items) && f.items.length === 8 &&
        new Set(f.items.map((i) => i.id)).size === 8 && f.items.every((i: unknown) => obj(i) && str(i.id) && url(i.url) && hash(i.sha256))) ||
      !Array.isArray(value.artifacts) || !value.artifacts.length || !value.artifacts.every((a: unknown) =>
        obj(a) && str(a.label) && url(a.url) && hash(a.sha256))) {
    throw new Error("本轮证据不完整，无法展示。请检查报告与图片清单。");
  }
  return value as RefinementReport;
}

export default function RefinementEvaluation() {
  const [report, setReport] = useState<RefinementReport | null>(null);
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  const [caseIndex, setCaseIndex] = useState(0);
  const [figureIndex, setFigureIndex] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setReport(null); setError(""); setCaseIndex(0); setFigureIndex(0);
    fetch(REPORT, { signal: controller.signal, cache: "no-store" })
      .then((response) => {
        if (!response.ok || !response.headers.get("content-type")?.includes("application/json")) throw new Error(`本轮证据读取失败（HTTP ${response.status}）`);
        return response.json();
      })
      .then((data: unknown) => setReport(parse(data)))
      .catch((reason: unknown) => { if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : "本轮证据读取失败"); });
    return () => controller.abort();
  }, [retry]);
  const group = report?.figures[caseIndex];
  const figure = group?.items[figureIndex];
  return <section className="joint-refine-card chain-builder" style={{ margin: "28px 0" }} aria-labelledby="refinement-title">
    <div className="chain-section-head"><div><span>VCP REFINEMENT · CAUSAL DIAGNOSTICS</span><h2 id="refinement-title">{report?.title ?? "继续优化：延迟证据与频谱增益"}</h2>
      <p>{report?.summary ?? "正在读取本轮独立实验与实际输出图片。"}</p></div><a className="text-link" href={REPORT} download>下载本轮报告 <ArrowRight size={14} /></a></div>
    {error ? <div role="alert" className="joint-next-load"><p>{error}</p><button type="button" className="button secondary" onClick={() => setRetry((v) => v + 1)}>重新读取</button></div> : !report ?
      <p role="status" className="joint-next-load"><LoaderCircle size={18} />正在读取本轮证据…</p> : <>
      <div className={`joint-next-admission is-${report.status}`} role="status"><Microscope size={22} /><div><strong>{report.status === "admitted" ? "工程准入通过" : report.status === "rejected" ? "本轮未通过推广准入" : "因果诊断与候选筛选"}</strong><p>{report.scope}</p></div></div>
      <ul className="joint-next-checks">{report.candidates.map((c) => <li key={c.id}><strong>{c.label} · {c.stage}</strong><p>{c.verdict}</p><small>{c.cases} 个完整场景回放</small>{c.metrics.map((m) => <p key={m.label}>{m.label}：{m.value}</p>)}</li>)}</ul>
      <div className="joint-next-limits"><h3>本轮定位与实测</h3><ul>{report.findings.map((f) => <li key={f.title}><strong>{f.title}</strong>：{f.detail}</li>)}</ul></div>
      <div className="joint-next-figures"><h3>8 类图：冻结 state v1 与本轮实际候选输出</h3><p>每路按自身声明时延对齐，横轴保留原录音时间；语音分数另与同一 clean 目标计算。</p>
        <label>本轮分析案例<select aria-label="选择本轮优化分析案例" value={caseIndex} onChange={(e) => { setCaseIndex(Number(e.target.value)); setFigureIndex(0); }}>{report.figures.map((f, i) => <option value={i} key={`${f.scene_id}-${f.window}`}>{f.label} · {f.window}</option>)}</select></label>
        <div className="joint-next-figure-tabs" role="group" aria-label="选择本轮图形分析类型">{group?.items.map((f, i) => <button type="button" key={f.id} aria-pressed={i === figureIndex} onClick={() => setFigureIndex(i)}>{jointNextFigureTitle(f.id)}</button>)}</div>
        {group && figure ? <figure><a href={figure.url} target="_blank" rel="noreferrer"><img src={figure.url} alt={`${group.label}，${group.window}，${jointNextFigureTitle(figure.id)}，冻结 state v1 与本轮候选输出`} loading="lazy" /></a><figcaption>{group.scene_id} · {group.window} · SHA256 <code>{figure.sha256}</code></figcaption></figure> : null}
      </div>
      <div className="joint-next-limits"><h3>适用范围与未解决问题</h3><ul>{report.limitations.map((l) => <li key={l}>{l}</li>)}</ul></div>
      <div className="joint-next-artifacts">{report.artifacts.map((a) => <a key={a.url} href={a.url} target="_blank" rel="noreferrer">{a.label}<ArrowRight size={13} /><small>SHA256 {a.sha256}</small></a>)}</div>
    </>}
  </section>;
}
