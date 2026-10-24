import { lazy, Suspense, useEffect, useState } from "react";
import { Download, RotateCcw } from "lucide-react";
const VehicleFrontend = lazy(() => import("./VehicleFrontend"));
const ActiveDataset = lazy(() => import("./ActiveDataset"));
import "../evaluation.css";
import "./itut-alignment.css";

type Failure = { pair?: number; label: string; segment: string; metric: string; value: number | null; operator: string; threshold: number | null };
type Job = { name: string; status: string; acceptance_passed: boolean; passed_phases?: number; quality_passed_phases?: number; phases?: number; failures?: Failure[] };
type Item = { id: string; sheet: string; source_cell: string; test_name: string; family: string; bandwidth: string; direction: string; unit: string; lower_limit: unknown; upper_limit: unknown };
type Ledger = { standard_compliance: false; formal_test_status: "not_measured"; items: Item[];
  paced_diagnostic?: { sample_rate: number; microphones: number; seconds: number; process_p95_ms: number; process_max_ms: number; completion_deadline_misses: number; passed_phases: number; phases: number };
  family_counts: Record<string, number>; simulation_identity: { cli_sha256: string; rules_sha256: string; paced: boolean; seconds_per_sequence?: number };
  standards: Record<string, { recommendation: string; edition?: string; url: string }>;
  engineering: { complete: boolean; passed_jobs: number; jobs: number; results: Job[] } };

const BASE = "/evaluation/itut-current";
const PAGE_SIZE = 25;
const SECTIONS = [
  { id: "formal", label: "正式测项" },
  { id: "engineering", label: "工程矩阵" },
  { id: "dataset", label: "当前数据集" },
  { id: "frontend", label: "车载前端" },
  { id: "standards", label: "标准与边界" },
] as const;
const FAMILY: Record<string, [string, string]> = {
  round_trip_delay: ["往返时延", "物理回路、编解码与播放/采集时间戳"],
  slr: ["发送响度 SLR", "人工嘴 / HATS 标定及标准响度算法"],
  rlr: ["接收响度 RLR", "人工耳 / HATS、音量设置及标准响度算法"],
  rlr_linearity: ["RLR 线性度", "规定输入电平下的标定响度测量"],
  frequency_response: ["频率响应", "声学传递响应、业务带宽对应掩模及归一化条件"],
  idle_noise_level: ["空闲噪声电平", "声学电平、规定计权与 50/100 ms 窗口"],
  idle_noise_peaks: ["空闲噪声谱峰", "标定功率谱及正式频谱掩模"],
  listening_quality: ["单向语音质量", "POLQA / TMOS 算法、规定信号及收发测点"],
  out_of_band: ["带外响应", "规定激励、标定测点和仪器修正"],
  double_talk_classification: ["双讲分类", "规定句序、电平轨迹与 A1/A2 判据"],
  tclw: ["声学耦合 TCLw", "标定终端声学路径、计权及音量"],
  hats_snr: ["HATS 信噪比", "HATS 语音/噪声电平、计权及窗口"],
  g_mos: ["整体质量 GMOS", "规定噪声场景及独立质量评估方法"],
  n_mos: ["噪声质量 NMOS", "规定噪声场景及独立噪声质量评分"],
  s_mos: ["语音质量 SMOS", "规定噪声场景及独立语音质量评分"],
  cng_level: ["舒适噪声电平", "CNG 实现、标定参考噪声及计权"],
  cng_spectrum: ["舒适噪声频谱", "CNG 实现、标定参考谱及容差掩模"],
};
const value = (v: unknown) => v === null || v === undefined ? "未定义" : String(v);
const number = (v: number | null) => v === null ? "未测" : v.toFixed(3);

function JobFailures({ job }: { job: Job }) {
  const [open, setOpen] = useState(false);
  return <details className="alignment-failures" onToggle={event => setOpen(event.currentTarget.open)}>
    <summary>{job.name} · {job.failures?.length} 个失败指标</summary>
    {open ? <div className="table-scroll"><table className="audio-table">
      <thead><tr><th>说话人组</th><th>阶段 / 窗口</th><th>指标</th><th>实测</th><th>工程要求</th></tr></thead>
      <tbody>{job.failures?.map((gate, index) => <tr key={index}><td>{gate.pair ?? "未定义"}</td><td>{gate.label} / {gate.segment}</td><td>{gate.metric}</td><td>{number(gate.value)}</td><td>{gate.operator} {number(gate.threshold)}</td></tr>)}</tbody>
    </table></div> : null}
  </details>;
}

export default function ItuAlignment() {
  const [ledger, setLedger] = useState<Ledger | null>(null);
  const [error, setError] = useState("");
  const [refresh, setRefresh] = useState(0);
  const [family, setFamily] = useState("all");
  const [bandwidth, setBandwidth] = useState("all");
  const [query, setQuery] = useState("");
  const [section, setSection] = useState<(typeof SECTIONS)[number]["id"]>("formal");
  const [page, setPage] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setLedger(null); setError("");
    fetch(`${BASE}/itut-ledger.json`, { signal: controller.signal, cache: "no-store" })
      .then(async response => {
        if (!response.ok) throw Error("验收报告暂不可用");
        const data = await response.json() as Ledger;
        if (data.standard_compliance !== false || data.formal_test_status !== "not_measured" || !Array.isArray(data.items))
          throw Error("验收报告格式与仿真口径不匹配");
        if (!controller.signal.aborted) setLedger(data);
      }).catch(reason => { if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : "报告读取失败"); });
    return () => controller.abort();
  }, [refresh]);
  if (!ledger) return <div className="alignment-page" role={error ? "alert" : "status"}>
    <p>{error || "正在读取验收报告…"}</p>
    {error ? <button className="button secondary" onClick={() => setRefresh(n => n + 1)}><RotateCcw size={16} />重新加载</button> : null}
  </div>;
  const items = ledger.items.filter(item => (family === "all" || item.family === family) &&
    (bandwidth === "all" || item.bandwidth === bandwidth) && `${item.id} ${item.test_name}`.toLowerCase().includes(query.toLowerCase()));
  const pageIndex = Math.min(page, Math.max(0, Math.ceil(items.length / PAGE_SIZE) - 1));
  const visibleItems = items.slice(pageIndex * PAGE_SIZE, (pageIndex + 1) * PAGE_SIZE);
  return <div className="evaluation-page alignment-page">
    <div className="alignment-status" role="status"><strong>正式声学测项：{ledger.items.length} 项未测</strong><span>工程任务：{ledger.engineering.passed_jobs} / {ledger.engineering.jobs} 通过{ledger.engineering.complete ? "" : " · 尚未运行完整"}</span></div>
    <p className="alignment-note">SI-SDR、STOI、ERLE 与 dBFS 保留为工程指标。正式测项需要对应的声学标定、测试方法与规定信号。</p>
    <div className="alignment-links"><a href={`${BASE}/itut-ledger.json`} download><Download size={15} />完整验收 JSON</a><a href={`${BASE}/itut-ledger.md`} download><Download size={15} />证据清单</a><a href={`${BASE}/summary.json`} download><Download size={15} />矩阵报告</a></div>
    <div className="alignment-sections" role="group" aria-label="验收分区">
      {SECTIONS.map(item => <button className="button secondary" key={item.id} type="button"
        aria-pressed={section === item.id} aria-controls="alignment-panel" onClick={() => setSection(item.id)}>{item.label}</button>)}
    </div>
    <div id="alignment-panel" role="region" aria-label={SECTIONS.find(item => item.id === section)?.label}>
    <Suspense fallback={<p role="status">正在读取分区…</p>}>
    {section === "engineering" ? <section className="evaluation-section" aria-label="车载流式矩阵">
      <h2>车载流式矩阵</h2>
      <p className="alignment-note">16 / 24 / 48 kHz，1 / 2 / 4 路麦克风；每条连续 {ledger.simulation_identity.seconds_per_sequence ?? 259} 秒，阶段之间不重置。{ledger.simulation_identity.paced ? "本批按 10 ms 节拍运行。" : "本批验证音质，实时期限另测。"} 高采样率输入由 16 kHz 素材转换，尚未验证原生高频带。</p>
      <div className="table-scroll"><table className="audio-table"><thead><tr><th>任务</th><th>音质阶段通过</th><th>包含耗时阶段通过</th><th>工程判定</th><th>未通过项</th></tr></thead><tbody>{ledger.engineering.results.map(job => <tr key={job.name}><td>{job.name}</td><td>{job.phases ? `${job.quality_passed_phases ?? "未测"} / ${job.phases}` : "成对时钟对照"}</td><td>{job.phases ? `${job.passed_phases} / ${job.phases}` : "—"}</td><td className={job.acceptance_passed ? "evaluation-status-pass" : "evaluation-status-fail"}>{job.acceptance_passed ? "通过" : "未通过"}</td><td>{job.failures?.length ?? "报告缺失"}</td></tr>)}</tbody></table></div>
      {ledger.engineering.results.filter(job => job.failures?.length).map(job => <JobFailures key={job.name} job={job} />)}
      {ledger.paced_diagnostic ? <p className="alignment-note">单独 10 ms 节拍复测：{ledger.paced_diagnostic.sample_rate / 1000} kHz / {ledger.paced_diagnostic.microphones} 麦 / {ledger.paced_diagnostic.seconds} s；P95 {ledger.paced_diagnostic.process_p95_ms.toFixed(3)} ms，最大 {ledger.paced_diagnostic.process_max_ms.toFixed(3)} ms，完成期限错过 {ledger.paced_diagnostic.completion_deadline_misses} 帧；包含全部工程门限 {ledger.paced_diagnostic.passed_phases} / {ledger.paced_diagnostic.phases} 阶段通过。</p> : null}
    </section> : null}
    {section === "dataset" ? <ActiveDataset /> : null}
    {section === "frontend" ? <VehicleFrontend /> : null}
    {section === "standards" ? <section className="evaluation-section" aria-label="标准与自适应边界">
      <h2>标准与自适应边界</h2>
      <div className="alignment-links">{Object.entries(ledger.standards).map(([key, standard]) => <a key={key} href={standard.url} target="_blank" rel="noreferrer">{standard.recommendation} {standard.edition}</a>)}</div>
      <p className="alignment-note">NS、AEC、AGC 与模拟时钟校正使用实际 PCM 和状态自适应。BF 使用固定标定几何；EQ 根据实测频响离线拟合，默认关闭。业务 NB / WB / SWB 与 PCM 采样率分别定义。24 kHz 无法覆盖 P.1120 的 14 kHz SWB 测量带宽；48 kHz 也仍需原生高频素材。</p>
      <p className="alignment-note">P.1120 已登记正式发送/接收频响掩模，以及目标车辆怠速、城市 60 km/h、高速 120 km/h 与 HVAC 条件。现有 TCAR 合成场景尚无这些实录条件证据；标准要求的预适应测量与启动压力测试分别保留。</p>
      <details className="alignment-failures"><summary>当前二进制与工程规则</summary><dl className="evaluation-identities"><dt>CLI SHA256</dt><dd>{ledger.simulation_identity.cli_sha256}</dd><dt>工程阈值 SHA256</dt><dd>{ledger.simulation_identity.rules_sha256}</dd></dl></details>
    </section> : null}
    {section === "formal" ? <section className="evaluation-section" aria-label="正式测试证据清单">
      <h2>正式测试证据清单</h2>
      <p className="alignment-note">原表条件与限值保留，尚需逐条核对现行标准。历史设备结果不作为当前引擎实测值。</p>
      <div className="alignment-filters"><label>测项类别<select value={family} onChange={event => { setFamily(event.target.value); setPage(0); }}><option value="all">全部 {Object.keys(ledger.family_counts).length} 类</option>{Object.keys(ledger.family_counts).map(key => <option value={key} key={key}>{FAMILY[key]?.[0] ?? key}</option>)}</select></label><label>业务带宽<select value={bandwidth} onChange={event => { setBandwidth(event.target.value); setPage(0); }}><option value="all">全部带宽</option>{[...new Set(ledger.items.map(item => item.bandwidth))].map(key => <option key={key}>{key}</option>)}</select></label><label>测项搜索<input type="search" value={query} onChange={event => { setQuery(event.target.value); setPage(0); }} /></label></div>
      <div className="alignment-pagination" role="group" aria-label="正式测项分页">
        <p role="status" className="alignment-note">{items.length} 个测项{items.length ? ` · 第 ${pageIndex * PAGE_SIZE + 1}–${Math.min((pageIndex + 1) * PAGE_SIZE, items.length)} 项 · ${pageIndex + 1} / ${Math.ceil(items.length / PAGE_SIZE)} 页` : ""}</p>
        <button className="button secondary" disabled={pageIndex === 0} onClick={() => setPage(pageIndex - 1)}>上一页</button>
        <button className="button secondary" disabled={(pageIndex + 1) * PAGE_SIZE >= items.length} onClick={() => setPage(pageIndex + 1)}>下一页</button>
      </div>
      {items.length ? <div className="table-scroll"><table className="audio-table"><thead><tr><th>原表位置 / 测项</th><th>客户表限值</th><th>当前状态</th><th>所缺证据</th></tr></thead><tbody>{visibleItems.map(item => <tr key={item.id}><td><small>{item.sheet} · {item.source_cell}</small><br />{item.test_name}</td><td>{value(item.lower_limit)} 至 {value(item.upper_limit)}<br />{item.unit}</td><td>未测</td><td>{FAMILY[item.family]?.[1] ?? item.family}</td></tr>)}</tbody></table></div> : <div className="evaluation-empty"><p>没有匹配测项。</p><button className="button secondary" onClick={() => { setFamily("all"); setBandwidth("all"); setQuery(""); setPage(0); }}>清除筛选</button></div>}
    </section> : null}
    </Suspense>
    </div>
  </div>;
}
