import { useEffect, useRef, useState } from "react";
import { ArrowLeft, CheckCircle2, CircleAlert, Download, RefreshCw } from "lucide-react";
import { fileTimeAtInput, isRecommendable, localAudioUrl, parseEcnrReport, stageOffsetSeconds, stageTimeline, waveformPolygon, tailRmsText } from "./ecnr-evaluation";
import type { EcnrReport, EcnrStage, EcnrTailEvent } from "./ecnr-evaluation";
import EcnrFinalAcceptance from "./EcnrFinalAcceptance";
import { parseFinalAcceptance } from "./ecnr-final-acceptance";
import type { FinalAcceptance } from "./ecnr-final-acceptance";
import "../evaluation.css";
import "./ecnr-evaluation.css";

const REPORT_URL = "/evaluation/ecnr-v2/report.json";
const FINAL_URL = "/evaluation/ecnr-v2/final-acceptance.json";
const PHASE: Record<string, string> = { cold_start_speech: "启动即说话", engine_change: "发动机变化", window_wind: "开窗风噪", radio_and_road: "广播＋路噪", long_radio_path: "长回声路径", reference_clock_drift: "参考时钟漂移", reference_mismatch: "播放参考错配", path_change: "回声路径变化", radio_restart: "广播重启" };
const SEGMENT: Record<string, string> = { near_only: "近端语音", near_late: "后段近端", double_talk: "双讲", far_only: "远端单讲", far_recovery: "回声恢复", noise_only: "纯噪声", runtime: "逐帧耗时", whole_output: "完整输出", reference: "播放参考" };
const METRIC: Record<string, string> = { raw_sisdr_db: "原始 SI-SDR / dB", stoi_change: "STOI 变化", projection_gain_min_db: "语音增益下限 / dB", projection_gain_max_db: "语音增益上限 / dB", rms_min_dbfs: "RMS 下限 / dBFS", rms_max_dbfs: "RMS 上限 / dBFS", total_attenuation_db: "总衰减 / dB", attenuation_lower_bound_db: "PCM16 衰减下界 / dB", linear_erle_engineering_db: "线性 ERLE / dB", clipping_percent: "削波 / %", peak_dbfs: "峰值 / dBFS" };
const STAGE: Record<string, string> = { "microphones": "麦克风", "aec-input": "回声消除输入", "linear-aec": "线性回声消除后", "protected-aec": "回声与语音保护后", "final-output": "最终输出", "nr-selected": "降噪选择后", "nr-protected-agc-before": "降噪与语音保护后（增益前）", "agc-after": "语音增益后", "eq-before": "频响补偿前", "eq-after": "频响补偿后", reference: "播放参考", microphone: "麦克风", mic: "麦克风", linear_aec: "线性回声消除后", protected_aec: "语音保护后", nr: "降噪后", eq: "频响补偿后", output: "最终输出" };
const fmt = (v: number | null | undefined, n = 2) => v == null || !Number.isFinite(v) ? "—" : v.toFixed(n);
const signed = (v: number | null | undefined, n = 3) => v == null || !Number.isFinite(v) ? "—" : `${v > 0 ? "+" : ""}${v.toFixed(n)}`;
const label = (value: string, dictionary: Record<string, string>) => dictionary[value] ?? value;
function stageLabel(name: string): string {
  const [side, ...parts] = name.split("/");
  if ((side === "baseline" || side === "candidate") && parts.length) return (side === "baseline" ? "基线 · " : "候选 · ") + (STAGE[parts.join("/")] ?? parts.join("/"));
  return STAGE[name] ?? name;
}
function Pending({ children = "尚未完成这项测量。" }: { children?: React.ReactNode }) {
  return <p className="ecnr-pending">{children}</p>;
}
function StageEvidence({ sequences }: { sequences: NonNullable<EcnrReport["sequences"]> }) {
  const [selected, setSelected] = useState(""), [inputTime, setInputTime] = useState(0);
  const audioRefs = useRef(new Map<string, HTMLAudioElement>());
  const sequence = sequences.find(s => s.id === selected) ?? sequences[0];
  const timeline = sequence ? stageTimeline(sequence.stages) : null;
  function seek(time: number) {
    setInputTime(time);
    sequence?.stages.forEach((stage, index) => {
      const node = audioRefs.current.get(`${sequence.id}-${index}`);
      const fileTime = fileTimeAtInput(stage, time);
      if (!node) return;
      node.pause();
      if (fileTime !== null) node.currentTime = fileTime;
    });
  }
  if (!sequence || !sequence.stages.length) return <Pending>阶段音频尚未导出。</Pending>;
  return <>
    <div className="ecnr-stage-controls">
      <label>连续流<select aria-label="阶段证据连续流" value={sequence.id} onChange={e => { setSelected(e.target.value); setInputTime(0); audioRefs.current.forEach(a => a.pause()); }}>{sequences.map(s => <option key={s.id} value={s.id}>{s.id}</option>)}</select></label>
      {timeline ? <label>共同输入时间 · {fmt(inputTime, 2)} s<input aria-label="共同输入时间" type="range" min={Math.max(0, timeline.start)} max={timeline.end} step="0.01" value={inputTime} onChange={e => seek(Number(e.target.value))} /></label> : null}
    </div>
    <p className="ecnr-note">波形共用输入时间轴和 ±1 FS 幅度。移动时间滑块将各路定位到相同输入时刻，试听保留文件原电平。多声道波形按声明的单一路显示；试听保留完整多声道原文件。</p>
    <div className="ecnr-stage-list">{sequence.stages.map((stage: EcnrStage, index) => {
      const offset = stageOffsetSeconds(stage), points = timeline ? waveformPolygon(stage, timeline) : null, url = localAudioUrl(stage.file);
      return <article className="ecnr-stage" key={`${sequence.id}-${index}`}>
        <div className="ecnr-stage-title"><h3>{stageLabel(stage.name)}</h3><span>{fmt(stage.sample_rate / 1000, 0)} kHz · {stage.channels} 声道{stage.channels > 1 && stage.waveform?.channel !== undefined ? ` · 波形第${stage.waveform.channel + 1}声道（通道 ${stage.waveform.channel}）` : ""}</span></div>
        {points && timeline ? <><svg viewBox="0 0 800 100" role="img" aria-label={`${stageLabel(stage.name)}波形，共用输入时间轴，幅度正负1 FS`}><line x1="10" x2="790" y1="50" y2="50" stroke="#dfe5ee" /><polygon points={points} fill="#527dac" opacity=".82" /></svg><div className="ecnr-wave-axis"><span>{fmt(timeline.start)} s</span><span>±1 FS</span><span>{fmt(timeline.end)} s</span></div></> : <Pending>{offset === null ? "时间元数据缺失或不一致，不能与其他阶段同轴比较。" : stage.channels !== 1 ? "多声道波形尚未声明有效声道，暂不能同轴比较。" : "尚未提供原幅波形包络。"}</Pending>}
        {url ? <audio ref={node => { if (node) audioRefs.current.set(`${sequence.id}-${index}`, node); else audioRefs.current.delete(`${sequence.id}-${index}`); }} aria-label={`${stageLabel(stage.name)}原电平试听`} controls preload="none" src={url} /> : <Pending>阶段音频尚未发布。</Pending>}
        <details><summary>测量位置与固定时间映射</summary><p>{stage.measurement_domain} · 固定延迟 {stage.fixed_delay_samples} 样本 · 已裁切 {stage.trimmed_leading_samples} 样本 · 文件起点对应输入 {fmt(offset, 6)} s</p></details>
      </article>;
    })}</div>
  </>;
}
function PostPlaybackTails({ events }: { events: EcnrTailEvent[] }) {
  const [selectedSequence, setSelectedSequence] = useState(""), [selectedEvent, setSelectedEvent] = useState(0);
  const ids = [...new Set(events.map(event => event.sequence_id))];
  const sequence = ids.includes(selectedSequence) ? selectedSequence : ids[0];
  const inSequence = events.filter(event => event.sequence_id === sequence);
  const index = Math.min(selectedEvent, Math.max(0, inSequence.length - 1));
  const event = inSequence[index];
  if (!event) return null;
  const reasons: Record<string, string> = { reference_restarted: "数字参考恢复，观察被截断", file_end: "文件结束，观察被截断", end_of_file: "文件结束，观察被截断", stream_ended: "连续流结束，观察被截断" };
  return <div className="ecnr-tail-evidence" aria-label="数字参考暂停后的混合尾迹">
    <div className="ecnr-tail-heading"><h3>数字参考暂停后的混合尾迹</h3><p>数字参考间隙不代表物理关闭播放；输入和输出尾迹仍可能含近端语音、背景噪声及回声。这里没有评估参考可靠性，也不把混合电平作为纯回声衰减。</p></div>
    <div className="ecnr-stage-controls">
      <label>连续流<select aria-label="尾迹连续流" value={sequence} onChange={e => { setSelectedSequence(e.target.value); setSelectedEvent(0); }}>{ids.map(id => <option key={id} value={id}>{id}</option>)}</select></label>
      <label>数字参考停止事件<select aria-label="数字参考停止事件" value={index} onChange={e => setSelectedEvent(Number(e.target.value))}>{inSequence.map((item, i) => <option key={i} value={i}>{fmt(item.details.stop_s, 2)} s · 静音间隙 {fmt(item.details.quiet_reference_duration_s, 2)} s{item.side === "baseline" ? " · 基线" : " · 候选"}</option>)}</select></label>
    </div>
    <p className="ecnr-note">事件时刻 {fmt(event.details.stop_s, 2)} s 使用共同输入时间；下表时长从该时刻开始。只显示所选事件，完整证据保留在实验报告。</p>
    <p className="ecnr-tail-scroll-hint">横向滑动表格，查看输入、输出电平及观察状态。</p>
    <div className="table-scroll"><table className="audio-table"><caption>所选参考停止事件的实际观察电平</caption><thead><tr><th>目标观察时长</th><th>实际观察时长</th><th>输入 RMS / dBFS</th><th>输出 RMS / dBFS</th><th>观察状态</th></tr></thead><tbody>{[...event.details.scans].sort((a, b) => a.horizon_s - b.horizon_s).map(scan => <tr key={scan.horizon_s}><td>{fmt(scan.horizon_s, 1)} s</td><td>{fmt(scan.observed_duration_s, 3)} s</td><td>{tailRmsText(scan.before)}</td><td>{tailRmsText(scan.after)}</td><td>{scan.complete ? "完整观察" : reasons[scan.truncation_reason ?? ""] ?? (scan.truncation_reason ? `截断：${scan.truncation_reason}` : "观察不完整")}</td></tr>)}</tbody></table></div>
    <p className="ecnr-note">数字静音没有有限的 RMS dBFS 值，单独标记；不可测值不会填成 0。截断窗口只报告实际观察长度。</p>
  </div>;
}
export function EcnrReportView({ report, finalAcceptance }: { report: EcnrReport; finalAcceptance?: FinalAcceptance | null }) {
  const [failureFilter, setFailureFilter] = useState("all");
  const q = report.quality, perceptual = report.perceptual, trials = report.runtime?.trials ?? [], recommended = isRecommendable(report);
  const failures = report.failures ?? [], shownFailures = failures.filter(f => failureFilter === "all" || f.label === failureFilter);
  const complete = report.state === "completed" && report.complete;
  const quality = q ? `${q.candidate_pass} / ${q.total_phases}` : "待完成";
  return <div className="ecnr-report">
    {finalAcceptance ? <EcnrFinalAcceptance final={finalAcceptance} report={report} /> : null}
    <section className="panel ecnr-status-panel" aria-label="ECNR 实验状态">
      <div className="ecnr-verdict" data-status={!finalAcceptance && recommended ? "pass" : report.state === "blocked" ? "fail" : "pending"}>
        {recommended ? <CheckCircle2 size={22} /> : <CircleAlert size={22} />}
        <div><h2>{finalAcceptance ? "连续流实验（冻结证据）" : recommended ? "已验证 · 可推荐配置" : report.state === "blocked" ? "验证受阻 · 保留当前配置" : complete ? "实验已完成 · 尚未推荐" : "实验进行中 · 保留当前配置"}</h2><p>{finalAcceptance ? "此处保留连续流实验冻结时的结果；最终验收结论见上方独立证据。" : report.recommendation?.reason ?? "完成质量、预测感知评分、身份核验与实时检查后再决定是否推荐。"}</p></div>
      </div>
      <p className="ecnr-note">比较基线：{report.baseline.quality_passed} / {report.baseline.total_phases} 个连续流阶段通过。{report.identity_verified ? "批次身份已核验。" : "批次身份核验尚未完成。"} 历史结果继续保留。</p>
    </section>
    <div className="ecnr-summary-grid">
      <section className="panel ecnr-summary"><h3>质量门限</h3><strong>{quality}</strong><p>{q ? `基线 ${q.baseline_pass} · 改善 ${q.phase_wins} · 回退 ${q.phase_regressions}` : "等待连续流配对回归"}</p></section>
      <section className="panel ecnr-summary"><h3>输入校准</h3><strong>{report.calibration?.status === "passed" ? "通过" : report.calibration?.status === "failed" ? "未通过" : "待完成"}</strong><p>{report.calibration?.summary?.replace(/^(\d+)\/(\d+) independent normal\/negative expectations$/, "$1/$2 项独立校准与负控符合预期") ?? "独立校准信号与负控"}</p></section>
      <section className="panel ecnr-summary"><h3>预测感知评分</h3><strong>{perceptual?.status === "completed" || perceptual?.status === "passed" ? "已测量" : "待完成"}</strong><p>DNSMOS · 语音、背景、整体</p></section>
      <section className="panel ecnr-summary"><h3>实际节拍</h3><strong>{trials.length ? `${trials.length} 次回放` : "待完成"}</strong><p>算法耗时、进入迟到、完成期限分别检查</p></section>
    </div>
    <section className="panel ecnr-section" aria-label="版本质量比较">
      <div className="panel-heading"><div><h2>版本质量比较</h2><p>保持原有门限；改善与新增失败同时展示。</p></div></div>
      {q ? <div className="ecnr-change-stats"><div><span>修复的失败项</span><strong>{q.repaired_gates}</strong></div><div data-fail={q.new_failed_gates > 0}><span>新增失败项</span><strong>{q.new_failed_gates}</strong></div><div><span>改善阶段</span><strong>{q.phase_wins}</strong></div><div data-fail={q.phase_regressions > 0}><span>回退阶段</span><strong>{q.phase_regressions}</strong></div></div> : <Pending />}
      {report.configurations?.length ? <div className="table-scroll"><table className="audio-table"><caption>连续流配置配对结果</caption><thead><tr><th>采样率 / 麦克风</th><th>配置</th><th>基线通过</th><th>候选通过</th><th>阶段数</th></tr></thead><tbody>{report.configurations.map(c => <tr key={c.id}><td>{c.rate / 1000} kHz / {c.microphones} 路</td><td>{c.variant === "spatial" ? "空间阵列处理" : c.variant === "control" ? "阵列对照" : c.variant}</td><td>{c.baseline_pass}</td><td>{c.candidate_pass}</td><td>{c.phases}</td></tr>)}</tbody></table></div> : null}
    </section>
    <section className="panel ecnr-section" aria-label="未通过窗口">
      <div className="panel-heading"><div><h2>未通过窗口</h2><p>极低电平诊断不会改变原失败统计。</p></div>{failures.length ? <label className="ecnr-filter">场景<select aria-label="失败场景筛选" value={failureFilter} onChange={e => setFailureFilter(e.target.value)}><option value="all">全部场景</option>{[...new Set(failures.map(f => f.label))].map(p => <option key={p} value={p}>{label(p, PHASE)}</option>)}</select></label> : null}</div>
      {report.failures ? failures.length ? <div className="table-scroll"><table className="audio-table"><caption>{shownFailures.length} 个未通过门限</caption><thead><tr><th>连续流 / 场景</th><th>窗口</th><th>指标</th><th>基线</th><th>候选</th><th>要求</th></tr></thead><tbody>{shownFailures.map((f, i) => <tr key={`${f.sequence_id}-${f.metric}-${i}`}><td><span className="ecnr-id">{f.sequence_id}</span>{label(f.label, PHASE)}</td><td>{label(f.segment, SEGMENT)}</td><td>{label(f.metric, METRIC)}</td><td>{fmt(f.baseline_value, f.metric.includes("stoi") ? 4 : 3)}</td><td className="evaluation-status-fail">{fmt(f.value, f.metric.includes("stoi") ? 4 : 3)}</td><td>{f.operator} {fmt(f.threshold, f.metric.includes("stoi") ? 4 : 3)}</td></tr>)}</tbody></table></div> : <Pending>{complete ? "本批次没有失败项。" : "当前已完成部分没有失败项，完整回归仍在进行。"}</Pending> : <Pending>失败窗口尚未汇总。</Pending>}
    </section>
    <section className="panel ecnr-section" aria-label="预测感知评分">
      <div className="panel-heading"><div><h2>预测感知评分</h2><p>DNSMOS P.835 · 模型预测，不是人工 MOS</p></div></div>
      {perceptual?.scores?.length ? <div className="table-scroll"><table className="audio-table"><caption>配对语音窗口比较</caption><thead><tr><th>指标</th><th>基线中位数</th><th>候选中位数</th><th>配对变化中位数</th><th>窗口数</th></tr></thead><tbody>{perceptual.scores.map(s => <tr key={s.metric}><td>{s.metric === "SIG" ? "语音质量 SIG" : s.metric === "BAK" ? "背景质量 BAK" : "整体质量 OVRL"}</td><td>{fmt(s.baseline_median, 3)}</td><td>{fmt(s.candidate_median, 3)}</td><td>{signed(s.paired_delta_median)}</td><td>{s.windows}</td></tr>)}</tbody></table></div> : <Pending>预测评分尚未完成或模型不可用。</Pending>}
      <p className="ecnr-note">{perceptual?.scope ?? "仅评价含近端语音的窗口；纯静音和纯远端不用于语音质量评分。"}{perceptual?.short_repeated_count !== undefined ? ` 短片段按官方方法重复补长：${perceptual.short_repeated_count} 个。` : ""} 背景分提升不能抵消语音损伤。</p>
    </section>
    <section className="panel ecnr-section" aria-label="实际节拍运行">
      <div className="panel-heading"><div><h2>实际节拍运行</h2><p>处理耗时、进入处理的迟到、完成期限分别报告。</p></div></div>
      {trials.length ? <div className="table-scroll"><table className="audio-table"><caption>每次完整回放的结果</caption><thead><tr><th>回放 / 配置</th><th>处理 P95 / ms</th><th>处理最大 / ms</th><th>处理 RTF</th><th>进入最大迟到 / ms</th><th>完成期限错过</th><th>信号延迟 / ms</th></tr></thead><tbody>{trials.map(t => <tr key={t.id}><td><span className="ecnr-id">{t.id}</span>{t.rate / 1000} kHz / {t.microphones} 路</td><td>{fmt(t.process_p95_ms, 3)}</td><td>{fmt(t.process_max_ms, 3)}</td><td>{fmt(t.process_rtf, 4)}</td><td>{fmt(t.release_lateness_max_ms, 3)}</td><td className={t.completion_deadline_misses != null && t.completion_deadline_misses > 0 ? "evaluation-status-fail" : ""}>{t.completion_deadline_misses ?? "—"}</td><td>{fmt(t.dsp_delay_ms, 3)}</td></tr>)}</tbody></table></div> : <Pending>完整实际节拍回放尚未完成。</Pending>}
      <p className="ecnr-note">离线处理 RTF 不代表实际节拍达标；“—”表示该项尚未测得。信号延迟与 CPU 耗时不同。</p>
    </section>
    <section className="panel ecnr-section" aria-label="极低电平窗口诊断">
      <div className="panel-heading"><div><h2>极低电平窗口诊断</h2><p>同时查看绝对电平和原始电平比；接近量化边界时，比值仅供诊断，原失败仍计入统计。</p></div></div>
      {report.audits?.nearzero_windows?.length ? <div className="table-scroll"><table className="audio-table"><thead><tr><th>连续流 / 场景 / 窗口</th><th>输入 / dBFS</th><th>输出 / dBFS</th><th>电平比 / dB</th><th>诊断依据</th></tr></thead><tbody>{report.audits.nearzero_windows.map((a, i) => <tr key={`${a.sequence_id}-${i}`}><td><span className="ecnr-id">{a.sequence_id}</span>{label(a.label, PHASE)} · {label(a.segment, SEGMENT)}</td><td>{tailRmsText(a.details?.before ?? { rms_dbfs: a.input_dbfs })}</td><td>{tailRmsText(a.details?.after ?? { rms_dbfs: a.output_dbfs })}</td><td>{fmt(a.total_attenuation_db, 3)}</td><td>{a.reason === "near_zero_denominator" ? "输入接近量化边界或为数字静音，电平比仅供诊断。" : a.reason ?? "绝对电平诊断"}{a.details?.before?.quantization_dbfs != null ? <small className="ecnr-id">量化参考 {fmt(a.details.before.quantization_dbfs, 3)} dBFS</small> : null}</td></tr>)}</tbody></table></div> : <Pending>极低电平窗口尚未审计。</Pending>}
      {report.audits?.post_playback_tails?.length ? <PostPlaybackTails events={report.audits.post_playback_tails} /> : null}
    </section>
    <section className="panel ecnr-section" aria-label="阶段音频证据">
      <div className="panel-heading"><div><h2>阶段音频证据</h2><p>参考、麦克风与各处理阶段使用声明的固定时间映射。阶段试听为已导出的证据子集，覆盖范围以所列连续流为准。</p></div></div>
      {report.sequences?.length ? <StageEvidence sequences={report.sequences} /> : <Pending>阶段音频按需导出后显示。</Pending>}
    </section>
  </div>;
}
export default function EcnrEvaluation() {
  const [report, setReport] = useState<EcnrReport | null>(null), [state, setState] = useState<"loading" | "ready" | "missing" | "error">("loading"), [error, setError] = useState(""), [revision, setRevision] = useState(0);
  const [finalAcceptance, setFinalAcceptance] = useState<FinalAcceptance | null>(null), [finalError, setFinalError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    setFinalAcceptance(null); setFinalError("");
    fetch(FINAL_URL, { signal: controller.signal, cache: "no-store" }).then(async response => {
      if (response.status === 404 || (response.ok && (response.headers.get("content-type") ?? "").includes("text/html"))) return;
      if (!response.ok) throw Error("最终验收状态暂时无法读取。");
      const data = parseFinalAcceptance(await response.json());
      if (!controller.signal.aborted) setFinalAcceptance(data);
    }).catch(e => { if (!controller.signal.aborted) setFinalError(e instanceof Error ? e.message : "最终验收状态无法读取。"); });
    setState("loading"); setError("");
    fetch(REPORT_URL, { signal: controller.signal, cache: "no-store" }).then(async r => {
      if (r.status === 404 || (r.ok && (r.headers.get("content-type") ?? "").includes("text/html"))) { setState("missing"); return; }
      if (!r.ok) throw Error("ECNR 实验报告加载失败。");
      const data = parseEcnrReport(await r.json());
      if (!controller.signal.aborted) { setReport(data); setState("ready"); }
    }).catch(e => { if (!controller.signal.aborted) { setError(e instanceof Error ? e.message : "报告加载失败。"); setState("error"); } });
    return () => controller.abort();
  }, [revision]);
  return <div className="evaluation-page ecnr-page">
    <div className="ecnr-toolbar"><a className="button secondary" href="#evaluation"><ArrowLeft size={16} />历史评测批次</a><div>{state === "ready" ? <a className="button secondary" href={REPORT_URL} download><Download size={16} />下载实验报告</a> : null}<button className="button secondary" onClick={() => setRevision(r => r + 1)} disabled={state === "loading"}><RefreshCw size={16} />刷新结果</button></div></div>
    {finalError ? <p className="ecnr-final-error" role="alert">{finalError} 连续流实验仍可查看；尚不能确认最终推荐。</p> : null}
    {state === "ready" && report ? <EcnrReportView report={report} finalAcceptance={finalAcceptance} /> : <section className="panel evaluation-empty" role={state === "error" ? "alert" : "status"}><h2>{state === "loading" ? "正在读取 ECNR 实验结果…" : state === "missing" ? "ECNR 实验报告尚未发布" : "无法读取实验报告"}</h2><p>{state === "error" ? error : state === "missing" ? "实验完成后将显示校准、质量、预测感知评分和真实节拍结果。当前配置与历史批次继续保留。" : "报告来自实际仿真批次。"}</p></section>}
  </div>;
}
