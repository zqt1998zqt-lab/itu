import { useEffect, useState } from "react";
import { ArrowRight, CheckCircle2, LoaderCircle, ShieldAlert } from "lucide-react";
import { JOINT_NEXT_REPORT_URL, jointNextFigureTitle, jointNextStatusLabel, parseJointNextReport } from "./joint-next-model";
import type { JointNextReport } from "./joint-next-model";
import "./joint-next-evaluation.css";

const signed = (value: number, digits = 2) => `${value > 0 ? "+" : ""}${value.toFixed(digits)}`;

export default function JointNextEvaluation() {
  const [report, setReport] = useState<JointNextReport | null>(null);
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  const [caseIndex, setCaseIndex] = useState(0);
  const [figureIndex, setFigureIndex] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setReport(null); setError(""); setCaseIndex(0); setFigureIndex(0);
    fetch(JOINT_NEXT_REPORT_URL, { signal: controller.signal, cache: "no-store" })
      .then((response) => {
        if (!response.ok) throw new Error(`新候选报告读取失败（HTTP ${response.status}）`);
        if (!response.headers.get("content-type")?.includes("application/json")) throw new Error("新候选入口没有返回 JSON。");
        return response.json();
      })
      .then((value: unknown) => setReport(parseJointNextReport(value)))
      .catch((reason: unknown) => { if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : "新候选报告读取失败"); });
    return () => controller.abort();
  }, [retry]);
  const selectedCase = report?.figures[caseIndex];
  const selectedFigure = selectedCase?.items[figureIndex];
  return <section className="joint-next-card chain-builder" aria-labelledby="joint-next-title">
    <div className="chain-section-head"><div><span>JOINT V2 · INDEPENDENT ADMISSION</span><h2 id="joint-next-title">联合链路下一轮：质量、回声与预算一起验收</h2>
      <p>这是新候选自己的冻结回放与图片分析。原有十条路线的分数保持独立；本卡只展示证据，不提供新预设执行。</p></div>
      <a className="text-link" href={JOINT_NEXT_REPORT_URL} download>下载新候选报告 <ArrowRight size={14} /></a></div>
    {error ? <div role="alert" className="joint-next-load"><strong>新候选证据暂不可用</strong><p>{error}</p><button type="button" className="button secondary" onClick={() => setRetry((value) => value + 1)}>重新读取</button></div> : !report ?
      <p role="status" className="joint-next-load"><LoaderCircle size={18} />正在读取独立准入结论…</p> : <>
        <div className={`joint-next-admission is-${report.admission.status}`} role="status">
          {report.admission.status === "admitted" ? <CheckCircle2 size={22} /> : <ShieldAlert size={22} />}
          <div><strong>{jointNextStatusLabel(report.admission.status)}</strong><p>{report.scope}</p><small>候选：{report.candidate_id} · {report.case_count} 个已使用冻结场景；不代表独立实录或整车验收。</small></div>
        </div>
        <ul className="joint-next-checks">{report.admission.checks.map((check) => <li key={check.id} className={check.pass ? "is-pass" : "is-fail"}><strong>{check.pass ? "通过" : "未通过"} · {check.label}</strong><p>{check.detail}</p></li>)}</ul>
        <div className="chain-matrix-scroll" tabIndex={0} role="region" aria-label="联合候选独立评估对比表，可横向滚动"><table className="joint-next-table"><thead><tr><th>配置</th><th>原门限通过</th><th>全部语音窗通过</th><th>主语音 ΔSI-SDR</th><th>ΔSTOI</th></tr></thead><tbody>
          {report.variants.map((variant) => <tr key={variant.id} className={variant.id === report.candidate_id ? "is-candidate" : ""}><th scope="row">{variant.label}<small>{variant.kind === "candidate" ? "本轮候选 · 独立计分" : "冻结对照"}</small></th><td>{variant.original_pass}/{report.case_count}</td><td>{variant.all_speech_pass}/{report.case_count}</td><td>{signed(variant.mean_sisdr_delta_db)} dB</td><td>{signed(variant.mean_stoi_delta, 4)}</td></tr>)}
        </tbody></table></div>
        <div className="joint-next-far">{report.far_windows.map((window) => <article key={window.window}><strong>{window.label}</strong><span>均值 {signed(window.mean_change_db)} dB</span><small>最差 {signed(window.min_change_db)} dB · {window.cases} 个适用窗 · 以报告声明的前版联合链路为对照</small></article>)}</div>
        <div className="joint-next-figures"><h3>图形引擎：相同输入与固定时延的输出对比</h3><p>图片展示前版与本轮候选的输出差异；语音质量单独对照同一 clean 目标，不搜索最佳时延、不做电平归一化。</p>
          <label>分析案例<select aria-label="选择联合候选分析案例" value={caseIndex} onChange={(event) => { setCaseIndex(Number(event.target.value)); setFigureIndex(0); }}>{report.figures.map((group, index) => <option value={index} key={`${group.scene_id}-${group.window}`}>{group.label} · {group.window}</option>)}</select></label>
          <div className="joint-next-figure-tabs" role="group" aria-label="选择图形分析类型">{selectedCase?.items.map((figure, index) => <button type="button" key={figure.id} aria-pressed={index === figureIndex} onClick={() => setFigureIndex(index)}>{jointNextFigureTitle(figure.id)}</button>)}</div>
          {selectedCase && selectedFigure ? <figure><a href={selectedFigure.url} target="_blank" rel="noreferrer"><img src={selectedFigure.url} alt={`${selectedCase.label}，${selectedCase.window}，${jointNextFigureTitle(selectedFigure.id)}，前版联合与本轮候选输出对比`} loading="lazy" /></a><figcaption>{selectedCase.scene_id} · {selectedCase.window} · SHA256 <code>{selectedFigure.sha256}</code></figcaption></figure> : null}
        </div>
        <div className="joint-next-limits"><h3>保留的限制与失败</h3><ul>{report.limitations.map((limit) => <li key={limit}>{limit}</li>)}</ul></div>
        <div className="joint-next-artifacts">{report.artifacts.map((artifact) => <a key={artifact.url} href={artifact.url} target="_blank" rel="noreferrer">{artifact.label}<ArrowRight size={13} /><small>SHA256 {artifact.sha256}</small></a>)}</div>
        <p className="joint-next-identity">本轮候选 CLI SHA256 <code>{report.candidate_cli_sha256}</code>。本卡不会将准入状态转换为预设执行权限。</p>
      </>}
  </section>;
}
