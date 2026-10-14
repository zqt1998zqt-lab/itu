import { CheckCircle2, CircleAlert, Download } from "lucide-react";
import { finalArtifactUrl, finalIsRecommendable, finalMatchesExperiment } from "./ecnr-final-acceptance";
import type { FinalAcceptance, FinalGate } from "./ecnr-final-acceptance";
import type { EcnrReport } from "./ecnr-evaluation";

const gateText: Record<FinalGate, string> = { passed: "通过", failed: "未通过", pending: "待完成" };
const num = (v: number | undefined | null, digits = 3) => v == null || !Number.isFinite(v) ? "—" : v.toFixed(digits);
const delta = (v: number | undefined) => v == null || !Number.isFinite(v) ? "—" : `${v > 0 ? "+" : ""}${v.toFixed(4)}`;
export default function EcnrFinalAcceptance({ final, report }: { final: FinalAcceptance; report: EcnrReport }) {
  const matches = finalMatchesExperiment(final, report), recommended = finalIsRecommendable(final, report);
  const full = final.independent, perf = final.performance;
  const trials = perf?.trials.filter(t => t.side !== "historical") ?? [];
  const historical = perf?.trials.find(t => t.side === "historical");
  const titles: Record<FinalAcceptance["status"], string> = {
    waiting_full: "最终验收 · 全量回归进行中", waiting_validation: "最终验收 · 等待独立核验",
    waiting_paced: "最终验收 · 等待真实节拍", blocked: "最终验收 · 验证受阻", quality_rejected: "最终验收 · 质量拒绝，节拍未运行", completed: "最终验收 · 已完成，保留实验结果",
  };
  const status = recommended ? "pass" : final.status === "blocked" || Object.values(final.acceptance).includes("failed") ? "fail" : "pending";
  return <section className="panel ecnr-section ecnr-final" aria-label="独立最终验收">
    <div className="ecnr-verdict" data-status={matches ? status : "fail"}>
      {recommended ? <CheckCircle2 size={24} /> : <CircleAlert size={24} />}
      <div><h2>{!matches ? "最终验收 · 证据身份不匹配" : recommended ? "最终验收 · 已验证，可推荐冻结候选" : titles[final.status]}</h2>
        <p>{matches ? final.recommendation.reason : "此验收结果没有绑定当前实验批次，不能据此推荐配置。"}</p></div>
    </div>
    <p className="ecnr-note">本页将冻结的连续流实验与独立全量、最终节拍证据关联。推荐标记仅表示本轮 HSAE 主机仿真全部验收通过；页面不更改配置。</p>
    <div className="ecnr-summary-grid">
      {([["quality", "质量验收"], ["perceptual", "预测感知验收"], ["real_time", "实时验收"], ["integrity", "完整身份核验"]] as const).map(([key, title]) =>
        <div className="ecnr-summary" key={key}><h3>{title}</h3><strong className={matches && final.acceptance[key] === "passed" ? "evaluation-status-pass" : final.acceptance[key] === "failed" ? "evaluation-status-fail" : ""}>
          {matches ? gateText[final.acceptance[key]] : "不可确认"}</strong></div>)}
    </div>
    <div className="ecnr-final-coverage">
      <h3>独立场景回归</h3><p><strong>{full?.processed_scenes?.toLocaleString() ?? "—"} / {full?.total_scenes?.toLocaleString() ?? "20,298"}</strong> 个场景已记录。{full?.complete ? "全量批次已结束。" : "当前为过程快照，最终结论需完整批次。"} </p>
      {full?.quality ? <p>基线通过 {full.quality.baseline_pass.toLocaleString()} · 候选通过 {full.quality.candidate_pass.toLocaleString()} · 新增门限失败 <strong>{full.quality.new_failed_gates}</strong> · 回退场景 <strong>{full.quality.phase_regressions}</strong></p> : null}
      <p>独立核验：{full?.validation ? full.validation.integrity_verified ? `已核验 ${full.validation.checked_scenes.toLocaleString()} 个场景，错误 ${full.validation.errors} 项。` : `未通过或未确认，已检查 ${full.validation.checked_scenes ?? "—"} 个，错误 ${full.validation.errors ?? "—"} 项。` : "待完成。"} </p>
      {full?.validation?.error_details?.length ? <details><summary>独立核验失败记录</summary><ul>{full.validation.error_details.map((e, i) => <li key={i}>{e}</li>)}</ul></details> : null}
    </div>
    <div className="ecnr-final-perceptual">
      <h3>预测感知配对变化</h3>
      <p>DNSMOS 是模型预测，未设置绝对 MOS 合格线。语音 SIG、整体 OVRL 的配对变化中位数需不下降；背景分不能抵消语音损伤。</p>
      <div className="table-scroll"><table className="audio-table"><caption>完整冻结批次的配对变化中位数</caption><thead><tr><th>证据范围</th><th>有效 / 预期窗口</th><th>SIG 语音</th><th>BAK 背景</th><th>OVRL 整体</th></tr></thead><tbody>
        {([["连续流", final.continuous?.perceptual], ["独立全量", full?.perceptual]] as const).map(([name, p]) => <tr key={name}><td>{name}</td><td>{p ? `${p.paired_windows} / ${p.expected_windows}` : "待完成"}</td><td>{delta(p?.paired_delta_median.SIG)}</td><td>{delta(p?.paired_delta_median.BAK)}</td><td>{delta(p?.paired_delta_median.OVRL)}</td></tr>)}
      </tbody></table></div>
    </div>
    <div className="ecnr-final-runtime">
      <h3>四次串行真实节拍</h3>
      <p>处理 P95 ≤ 1 ms、最大 ≤ 10 ms、处理 RTF ≤ 0.1，处理超 10 ms 帧数为 0。候选两次均须满足处理门限与完成期限，四次音频须与对应冻结离线输出一致。</p>
      {trials.length ? <div className="table-scroll"><table className="audio-table"><caption>新增重复全部结果；每次为完整 235 秒、48 kHz、四麦连续流</caption><thead><tr><th>回放</th><th>处理 P95 / ms</th><th>最大 / ms</th><th>RTF</th><th>处理超 10 ms</th><th>进入最大迟到 / ms</th><th>完成错过</th><th>其中进入已迟到</th><th>音频一致</th><th>固定信号延迟 / ms</th></tr></thead><tbody>{trials.map(t => <tr key={t.id}>
        <td>{t.side === "candidate" ? "候选" : "基线"} · 第 {t.repetition} 次</td><td>{num(t.process_p95_ms)}</td><td>{num(t.process_max_ms)}</td><td>{num(t.process_rtf, 4)}</td><td>{t.process_frames_over_10ms}</td><td>{num(t.release_lateness_max_ms)}</td>
        <td className={t.completion_deadline_misses > 0 ? "evaluation-status-fail" : ""}>{t.completion_deadline_misses}</td><td>{t.already_late_before_processing}</td><td>{t.offline_audio_parity === true ? "一致" : t.offline_audio_parity === false ? "不一致" : "待核验"}</td><td>{num(t.dsp_delay_ms)}</td>
      </tr>)}</tbody></table></div> : <p className="ecnr-pending">四次完整真实节拍尚未结束或未通过身份核验。</p>}
      <p className="ecnr-note">历史 37 次完成期限错过继续保留{historical ? `：处理 P95 ${num(historical.process_p95_ms)} ms，进入最大迟到 ${num(historical.release_lateness_max_ms)} ms，完成错过 ${historical.completion_deadline_misses} 次。` : "，与本轮新增重复分别统计。"} 进入迟到、CPU 处理耗时和固定信号延迟分别测量。</p>
    </div>
    <p className="ecnr-note">{final.scope} 记录时间：{final.observed_at ? new Date(final.observed_at).toLocaleString("zh-CN") : "—"}；过程快照不会自动更新分母。</p>
    {final.artifacts.length ? <details><summary>完整验收证据与下载</summary><div className="ecnr-final-artifacts">{final.artifacts.map(a => {
      const url = finalArtifactUrl(a.file);
      return url ? <a key={a.file} href={url} download><Download size={14} />{a.label}</a> : null;
    })}</div></details> : null}
  </section>;
}
