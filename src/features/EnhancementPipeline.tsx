import { useEffect, useId, useMemo, useState } from "react";
import { ArrowRight, Check, Copy, Download, GitBranch, Layers3, LoaderCircle, RotateCcw, Trophy, Workflow } from "lucide-react";
import { downloadFile } from "../lib/files";
import AudioSynthesize from "./AudioSynthesize";
import JointNextEvaluation from "./JointNextEvaluation";
import RefinementEvaluation from "./RefinementEvaluation";
import StructureEvaluation from "./StructureEvaluation";
import PipelineEvaluation from "./PipelineEvaluation";
import type { PipelineEvaluationData } from "./PipelineEvaluation";
import {
  CHAIN_REPORT_URL, CHAIN_STORAGE_KEY, chainCommand, chainFlags, chainPresets, chainStages,
  changeStage, matchingEvidence, matchingPreset, optionFor, parseChainReport, restoreConfiguration,
} from "./enhancement-chain-model";
import type { ChainConfiguration, ChainReport, ChainVariant, StageId } from "./enhancement-chain-model";
import "./ns-pipeline.css";
import "./enhancement-pipeline.css";

const fmt = (value: number | null, digits = 2) => value === null ? "—" : `${value > 0 ? "+" : ""}${value.toFixed(digits)}`;
const TX_ORDER: StageId[] = ["limiter", "drc", "agc", "eq", "ns", "aec", "bf"];
const nodeStyle = (x: number, y: number) => ({ left: `${x / 1390 * 100}%`, top: `${y / 438 * 100}%` });
const rankValue = (value: number | null) => value === null ? Number.NEGATIVE_INFINITY : value;

function compareVariants(left: ChainVariant, right: ChainVariant): number {
  return right.pass - left.pass ||
    right.mean_sisdr_delta_db - left.mean_sisdr_delta_db ||
    right.mean_stoi_delta - left.mean_stoi_delta ||
    right.mean_noise_delta_db - left.mean_noise_delta_db ||
    rankValue(right.mean_far_delta_db) - rankValue(left.mean_far_delta_db) ||
    left.clipped_cases - right.clipped_cases ||
    left.limited_samples - right.limited_samples ||
    left.mean_rtf - right.mean_rtf;
}

function ChainFlow({ configuration, selected, onSelect }: {
  configuration: ChainConfiguration; selected: StageId; onSelect: (stage: StageId) => void;
}) {
  const uid = useId().replace(/:/g, "");
  return (
    <figure className="chain-flow">
      <div className="chain-section-head">
        <div><span>SIGNAL TOPOLOGY</span><h2>沿流程图串联增强技术</h2><p>上行为从右向左的实际处理顺序；点击节点查看本级配置。小屏可在画布内横向滑动。</p></div>
        <a className="text-link" href="#workflow">原始流程图 <ArrowRight size={14} /></a>
      </div>
      <div className="chain-flow-scroll" tabIndex={0} role="region" aria-label="综合增强 TX / RX 流程图，可横向滚动">
        <div className="chain-canvas">
          <svg width="1390" height="438" viewBox="0 0 1390 438" aria-hidden="true">
            <defs>
              <marker id={`${uid}-audio`} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7" fill="#70829a" /></marker>
              <marker id={`${uid}-ref`} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7" fill="#b58a2b" /></marker>
              <marker id={`${uid}-echo`} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7" fill="#9a78b4" /></marker>
            </defs>
            <rect x="14" y="65" width="1330" height="145" rx="10" fill="#f5f8fd" />
            <rect x="14" y="284" width="1330" height="128" rx="10" fill="#fbfaf6" />
            <text x="32" y="41" className="chain-svg-lane">TX UPLINK · 单路增强输出</text>
            <text x="1195" y="41" className="chain-svg-note">近端语音 + 噪声 + 回声</text>
            <text x="32" y="263" className="chain-svg-lane">RX DOWNLINK · 现有下行实现</text>
            {Array.from({ length: 8 }, (_, index) => <path key={index} d={`M${180 + index * 145},134 H${index === 0 ? 164 : 167 + index * 145}`} stroke="#70829a" fill="none" markerEnd={`url(#${uid}-audio)`} />)}
            {["M164 358 H255", "M387 358 H470", "M602 358 H685", "M817 358 H1195"].map((path) => <path key={path} d={path} stroke="#70829a" fill="none" markerEnd={`url(#${uid}-audio)`} />)}
            <path d="M817 358 H971 V178" stroke="#b58a2b" strokeWidth="1.5" fill="none" markerEnd={`url(#${uid}-ref)`} />
            <path d="M1327 358 H1360 V134 H1327" stroke="#9a78b4" strokeDasharray="5 4" fill="none" markerEnd={`url(#${uid}-echo)`} />
            <text x="840" y="243" className="chain-svg-reference">实际播放 PCM → AEC 参考</text>
            <text x="1146" y="243" className="chain-svg-note">声学耦合回到麦克风</text>
          </svg>
          <div className="chain-node chain-node--endpoint" style={nodeStyle(32, 90)}><span>TX OUT</span><strong>上行输出</strong><small>送网络 / Codec</small></div>
          {TX_ORDER.map((id, index) => {
            const stage = chainStages.find((item) => item.id === id)!;
            const option = optionFor(stage, configuration);
            const off = option.id === "off";
            return <button key={id} type="button" style={nodeStyle(180 + index * 145, 90)} title={option.label}
              className={`chain-node${selected === id ? " is-selected" : ""}${off ? " is-bypassed" : ""}`}
              aria-label={`查看${stage.label}节点：${option.label}`} aria-pressed={selected === id} aria-controls="chain-node-detail"
              onClick={() => onSelect(id)}><span>{id.toUpperCase()} · {off ? "旁路" : "启用"}</span><strong>{stage.label.split(" ").slice(1).join(" ") || stage.label}</strong><small>{option.label.split(" · ")[0]}</small></button>;
          })}
          <div className="chain-node chain-node--endpoint" style={nodeStyle(1195, 90)}><span>TX IN</span><strong>同步麦克风阵列</strong><small>实际标定 · 逐通道延迟</small></div>
          {[
            [32, "RX IN", "下行输入", "远端网络音频"],
            [255, "RX NR", "轻度降噪", "下行独立状态"],
            [470, "RX EQ", "播放均衡", "独立 EQ 配置"],
            [685, "RX DRC", "动态范围控制", "软压缩与饱和"],
            [1195, "SPEAKER", "扬声器播放", "与 AEC 参考同源"],
          ].map(([left, tag, title, sub]) => <div key={String(tag)} className="chain-node chain-node--rx" style={nodeStyle(Number(left), 314)}><span>{tag}</span><strong>{title}</strong><small>{sub}</small></div>)}
        </div>
      </div>
      <figcaption><GitBranch size={15} /><span>本轮整链评测使用冻结的实际播放参考（<code>--echo</code>），下行未重复处理。RX NR → EQ → DRC 是现有实现拓扑；其声学效果不计入本轮 TX 结论。</span></figcaption>
    </figure>
  );
}

function evaluationFor(variant: ChainVariant, report: ChainReport): PipelineEvaluationData {
  return {
    tone: variant.pass < variant.cases ? "warning" : "pending",
    status: variant.id === "array_control" ? "配对控制，不参与方案排名" : "工程探索矩阵 · 非整车准入",
    headline: `${variant.pass}/${variant.cases} 通过`,
    scope: `${variant.label}：同一 72 个冻结阵列场景的完整 CLI 回放。主语音、双讲、噪声、回声与电平门限共同判定。`,
    source: report.source,
    note: "所有路线用同一个标定 clean delay-sum 目标；只扣声明的 DSP 时延，不搜索最佳 lag。通过数属于整条组合，不能相加各模块通过数。典型案例波形与频谱由图形引擎从该案例冻结的输入/输出重新生成，分数仍以 DAS 配对控制为基线。",
    metrics: [
      { label: "主语音 ΔSI-SDR / ΔSTOI", value: `${fmt(variant.mean_sisdr_delta_db)} dB`, detail: `ΔSTOI ${fmt(variant.mean_stoi_delta, 4)} · 相对 DAS 控制的均值` },
      { label: "双讲 ΔSI-SDR / ΔSTOI", value: `${fmt(variant.mean_double_sisdr_delta_db)} dB`, detail: `${variant.double_cases} 个适用案例 · ΔSTOI ${fmt(variant.mean_double_stoi_delta, 4)}` },
      { label: "纯噪声 / 远端单讲衰减增量", value: `${variant.noise_censored_cases ? "≥ " : ""}${fmt(variant.mean_noise_delta_db)} dB`, detail: `远端单讲 ${variant.far_censored_cases ? "≥ " : ""}${fmt(variant.mean_far_delta_db)} dB · ${variant.far_cases} 案例 · 此项不是 ERLE` },
      { label: "输出真实变化", value: `${variant.changed}/${variant.cases}`, detail: "完整输出文件 SHA256 与 DAS 控制逐一配对" },
      { label: "削波 / 终级限幅触发", value: `${variant.clipped_cases} / ${variant.limited_samples}`, detail: "削波案例数 / 输入时段限幅样本数；0 次不证明机制已被覆盖" },
      { label: "均值处理 RTF", value: variant.mean_rtf.toFixed(4), detail: "aarch64 主机 · 4 worker 离线回放；不是设备 RTD 或回调准入" },
    ],
    caseGroups: variant.caseGroups,
  };
}

export default function EnhancementPipeline() {
  const [configuration, setConfiguration] = useState<ChainConfiguration>(() => {
    try { return restoreConfiguration(localStorage.getItem(CHAIN_STORAGE_KEY)); } catch { return restoreConfiguration(null); }
  });
  const [selectedStage, setSelectedStage] = useState<StageId>("aec");
  const [report, setReport] = useState<ChainReport | null>(null);
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  const [feedback, setFeedback] = useState("");
  const preset = matchingPreset(configuration);
  const variant = matchingEvidence(configuration, report);
  const rankedVariants = useMemo(() => {
    if (!report) return [];
    const control = report.variants.filter((row) => row.id === "array_control");
    return [...control, ...report.variants.filter((row) => row.id !== "array_control").sort(compareVariants)];
  }, [report]);
  const bestVariant = rankedVariants.find((row) => row.id !== "array_control");
  const bestPreset = chainPresets.find((item) => item.id === bestVariant?.id);
  const selected = chainStages.find((stage) => stage.id === selectedStage)!;
  const option = optionFor(selected, configuration);
  const command = useMemo(() => chainCommand(configuration), [configuration]);

  useEffect(() => {
    try { localStorage.setItem(CHAIN_STORAGE_KEY, JSON.stringify(configuration)); } catch { /* The draft still works without browser persistence. */ }
    setFeedback("");
  }, [configuration]);

  useEffect(() => {
    const controller = new AbortController();
    setError("");
    setReport(null);
    fetch(CHAIN_REPORT_URL, { signal: controller.signal, cache: "no-store" })
      .then((response) => {
        if (!response.ok) throw new Error(`评测文件读取失败（HTTP ${response.status}）`);
        if (!response.headers.get("content-type")?.includes("application/json")) throw new Error("评测入口未返回 JSON，请检查评测文件是否已发布。");
        return response.json();
      })
      .then((data: unknown) => setReport(parseChainReport(data)))
      .catch((reason: unknown) => { if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : "评测读取失败"); });
    return () => controller.abort();
  }, [retry]);

  const usePreset = (id: string, reveal = false) => {
    const next = chainPresets.find((item) => item.id === id);
    if (next) {
      setConfiguration(next);
      if (reveal) requestAnimationFrame(() => {
        document.getElementById("chain-selected-algorithm")?.scrollIntoView({ behavior: "smooth", block: "center" });
      });
    }
  };
  const exportConfig = () => {
    downloadFile(JSON.stringify({
      schema_version: 1, ...configuration, preset_id: preset?.id ?? null,
      evaluation_status: variant ? "measured_configuration" : "unmeasured_configuration",
      frozen_evaluation_source: variant ? report?.source : null,
      cli_flags: chainFlags(configuration), beam_delays_samples: null,
      requires_calibrated_beam_delays: configuration.selection.bf !== "off",
      playback_reference: "actual-playback.wav", rx_reprocessing: false, standard_compliance: false,
    }, null, 2) + "\n", "enhancement-chain.json", "application/json");
    setFeedback("链路配置已导出；这是回放配置，未在设备上应用。");
  };
  const copyCommand = async () => {
    try { await navigator.clipboard.writeText(command); setFeedback("CLI 模板已复制；执行前填写真实文件与逐通道标定延迟。"); }
    catch { setFeedback("当前浏览器无法复制，请展开下方 CLI 模板并手动选择文本。"); }
  };

  return (
    <div className="nsp-page chain-page">
      <section className="nsp-intro">
        <div><span>INTEGRATED VOICE ENHANCEMENT · TX / RX</span><h2>把空间、回声、频谱与动态控制串成一条链路</h2><p>在同一条 VCP 音频路径中组合 BF、AEC、NS、EQ、AGC 与 DRC。更换任何一级都会形成新配置；整链结果以该组合自己的冻结回放为准。</p></div>
        <div className="nsp-summary" aria-label="综合链路汇总"><div><strong>6</strong><span>增强模块</span></div><div><strong>{chainPresets.length}</strong><span>可选组合</span></div><div><strong>{report?.case_count ?? "—"}</strong><span>冻结场景</span></div><div><strong>{report?.cli_runs ?? "—"}</strong><span>CLI 回放</span></div></div>
      </section>

      <StructureEvaluation />
      <RefinementEvaluation />
      <JointNextEvaluation />

      <section className="chain-builder" aria-labelledby="chain-config-title">
        <div className="chain-toolbar">
          <div><h2 id="chain-config-title">组合方案</h2><p>草案保存在本浏览器；选择预设或逐级替换技术。</p></div>
          <label className="chain-preset-label"><span>选择组合预设</span><select aria-label="选择综合链路预设" value={preset?.id ?? "custom"} onChange={(event) => usePreset(event.target.value)}><option value="custom" disabled>自定义组合 · 待评测</option>{chainPresets.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>
          <div className="chain-actions"><button type="button" className="button secondary" onClick={() => usePreset("all_serial")}><RotateCcw size={14} />恢复六级串联</button><button type="button" className="button secondary" onClick={exportConfig}><Download size={14} />导出配置</button></div>
        </div>
        <div className={`chain-config-state${variant ? " is-measured" : ""}`} role="status">
          {variant ? <Check size={16} /> : <Layers3 size={16} />}<div><strong>{!report ? error ? "整链评测读取失败" : "正在读取整链评测" : variant ? `${variant.label} · 已有整链评测` : "当前组合 · 待评测"}</strong><span>{preset?.description ?? "已更换技术组合。导出配置后需重新回放，才能得到该组合自己的分数与案例图。"}</span></div><span className="chain-state-tag">{!report ? error ? "结果不可用" : "读取中" : variant ? `${variant.pass}/${variant.cases} 通过` : "无匹配结果"}</span>
        </div>
      </section>

      {preset?.id === "vcp_joint" ? <section className="chain-config-state" aria-label="联合方案的独立回放证据">
        <Check size={16} /><div><strong>联合方案 · 独立 72 例回放</strong><span>原门限 64/72 通过，全语音窗 59/72。启动期远端抑制仍有代价；这些是已使用场景的工程回放，不代表新录音或整车准入。</span></div>
        <a className="text-link" href="/evaluation/vcp-joint-20261024/index.html" target="_blank" rel="noreferrer">查看联合分析与图片 <ArrowRight size={14} /></a>
      </section> : null}

      <section id="chain-run" className="chain-builder chain-run" aria-labelledby="chain-run-title">
        <div className="chain-section-head"><div><span>REPLAY YOUR AUDIO</span><h2 id="chain-run-title">用当前方案处理音频</h2><p>上传同步麦克风与实际播放参考，生成可以试听、下载和查看对比图的真实输出。下方冻结评测是独立证据，不会作为本次录音的得分。</p></div></div>
        <AudioSynthesize configuration={configuration} embedded />
      </section>

      <ChainFlow configuration={configuration} selected={selectedStage} onSelect={setSelectedStage} />

      <section id="chain-node-detail" className="chain-node-detail" aria-label="所选综合链路节点说明" aria-live="polite">
        <div><span>当前节点 · {selected.id.toUpperCase()}</span><h3>{selected.label} <small>{option.id === "off" ? "旁路" : option.candidate ? "候选技术" : "已接入执行器"}</small></h3><p>{selected.description}</p><strong>{option.label}</strong><p>{option.note}</p></div>
        <div className="chain-detail-evidence"><span>实际实现</span><code>{selected.source}</code><a href={`#${selected.tab}`} className="text-link">查看本模块的独立评测、波形与 Mel 图 <ArrowRight size={14} /></a></div>
      </section>

      <section className="chain-builder" aria-labelledby="chain-stages-title">
        <div className="chain-section-head"><div><span>SERIAL MODULE SELECTION</span><h2 id="chain-stages-title">逐级选择增强技术</h2><p>每级选择一条音频执行路径，控制面保留在所属模块内。ULUNAS 需另配模型；GTCRN、DPCRN 离线对照与研究技术可在各自模块页查看。</p></div></div>
        <div className="chain-stage-grid">{chainStages.map((stage, index) => {
          const choice = optionFor(stage, configuration);
          return <article key={stage.id} className={`chain-stage${choice.id === "off" ? " is-bypassed" : ""}${selectedStage === stage.id ? " is-selected" : ""}`}>
            <label htmlFor={`chain-stage-${stage.id}`}><span>{String(index + 1).padStart(2, "0")}</span><strong>{stage.label}</strong></label>
            <select id={`chain-stage-${stage.id}`} value={choice.id} onFocus={() => setSelectedStage(stage.id)} onChange={(event) => { setSelectedStage(stage.id); setConfiguration((current) => changeStage(current, stage.id, event.target.value)); }}>
              {stage.options.map((item) => <option key={item.id} value={item.id}>{item.label}{item.candidate ? "（候选）" : ""}</option>)}
            </select><p>{choice.note}</p><a href={`#${stage.tab}`}>本级独立评测 <ArrowRight size={12} /></a>
          </article>;
        })}</div>
        <p className="chain-builder-note">TX 顺序固定为 BF → AEC → NS → EQ → AGC → DRC → limiter。修改 DAS 对照会回到 VCP 草案；单麦输入请旁路 BF，多麦启用 BF 时必须填写真实标定延迟。</p>
      </section>

      {error ? <section className="chain-load-state" role="alert"><strong>整链评测暂不可用</strong><p>{error} 配置仍可编辑与导出。</p><button className="button secondary" onClick={() => setRetry((value) => value + 1)}>重新读取评测</button></section> : !report ? <div className="chain-load-state" role="status"><LoaderCircle size={18} />正在读取独立的整链评测结果…</div> : <>
        <section className="chain-matrix" aria-labelledby="chain-matrix-title">
          <div className="chain-section-head"><div><span>PAIRED END-TO-END MATRIX</span><h2 id="chain-matrix-title">整链组合对比</h2><p>16 kHz · 2/4 麦 · 2 组近远端说话人 · SNR 0/15/30 dB · 6 类模拟空间场。全部路线配对同一 DAS 控制与同一 clean 波束目标；本表按工程通过、语音质量、衰减与开销综合排序。</p></div><a className="text-link" href={CHAIN_REPORT_URL} download="enhancement-chain-report.json"><Download size={14} />下载完整汇总</a></div>
          {bestVariant ? <div className="chain-best-summary" aria-label="当前单级筛选最优候选">
            <Trophy size={20} />
            <div><span>CURRENT SINGLE-STAGE SCREENING BEST</span><strong>{bestVariant.label}</strong><p>{bestVariant.pass}/{bestVariant.cases} 通过 · ΔSI-SDR {fmt(bestVariant.mean_sisdr_delta_db)} dB · ΔSTOI {fmt(bestVariant.mean_stoi_delta, 4)} · 噪声/远端衰减余量低于 WebRTC NS。</p></div>
            {bestPreset ? <button type="button" className="button secondary" aria-controls="chain-selected-algorithm" onClick={() => usePreset(bestPreset.id, true)}>载入最优候选</button> : null}
          </div> : null}
          <div className="chain-matrix-scroll" tabIndex={0} role="region" aria-label="综合增强组合评测表，可横向滚动"><table>
            <thead><tr><th scope="col">组合 · 排序后点击载入</th><th scope="col">工程通过</th><th scope="col">主语音 ΔSI-SDR</th><th scope="col">ΔSTOI</th><th scope="col">噪声衰减增量</th><th scope="col">双讲 ΔSI-SDR</th><th scope="col">远端单讲衰减增量</th><th scope="col">输出变化 / 削波案例</th></tr></thead>
            <tbody>{rankedVariants.map((row) => <tr key={row.id} className={`${variant?.id === row.id ? "is-current" : ""} ${row.id === "array_control" ? "is-control" : ""} ${row.id === bestVariant?.id ? "is-best" : ""}`}>
              <th scope="row"><button type="button" onClick={() => usePreset(row.id, true)} aria-controls="chain-selected-algorithm" aria-pressed={variant?.id === row.id}>{row.label}{row.id === bestVariant?.id ? <span className="chain-best-pill">单级最优</span> : null}<ArrowRight size={12} /></button><small>{row.id === "array_control" ? "配对控制 · 不参与排名" : row.description}</small></th>
              <td><strong>{row.pass}/{row.cases}</strong></td><td>{fmt(row.mean_sisdr_delta_db)} dB</td><td>{fmt(row.mean_stoi_delta, 4)}</td>
              <td>{row.noise_censored_cases ? "≥ " : ""}{fmt(row.mean_noise_delta_db)} dB{row.noise_censored_cases ? <small>{row.noise_censored_cases} 案例使用量化下界</small> : null}</td>
              <td>{fmt(row.mean_double_sisdr_delta_db)} dB<small>{row.double_cases} 个适用案例</small></td><td>{row.far_censored_cases ? "≥ " : ""}{fmt(row.mean_far_delta_db)} dB<small>{row.far_cases} 个适用案例</small></td><td>{row.changed} / {row.clipped_cases}</td>
            </tr>)}</tbody>
          </table></div>
          <div className="chain-gates"><p><strong>联合门限：</strong>主语音与适用双讲窗 ΔSI-SDR ≥ −0.1 dB、ΔSTOI ≥ −0.02，投影增益 −6～18 dB；噪声衰减增量 ≥ 0 dB；有远端单讲时其衰减增量 ≥ 6 dB；整段峰值 ≤ −0.1 dBFS，削波为 0%。</p><p><strong>可失败性：</strong>同门限拒绝了静音负对照的 {report.negative_control.rejected}/{report.negative_control.cases} 个案例。PCM16 低于量化分辨率的衰减按半 LSB 保守下界汇总，并标记“≥”；本矩阵 standard_compliance=false，RX 与整车声学验收需独立测量。</p></div>
        </section>
        <section id="chain-selected-algorithm" className="chain-selected-algorithm" aria-labelledby="chain-selected-algorithm-title" aria-live="polite">
          <div className="chain-selected-head">
            <div><span>SELECTED ALGORITHM</span><h2 id="chain-selected-algorithm-title">选中算法组合</h2><strong>{variant?.label ?? preset?.label ?? "自定义组合"}</strong></div>
            <span className={`chain-selected-status${variant ? " is-measured" : ""}`}>{variant ? `${variant.pass}/${variant.cases} 通过` : "待评测"}</span>
          </div>
          <p>{preset?.description ?? "当前自定义组合尚未完成整链评测；以下为该配置的实际执行顺序。"}</p>
          <ol className="chain-selected-path" aria-label="当前选中算法的执行顺序">
            {chainStages.map((stage) => {
              const selection = optionFor(stage, configuration);
              return <li key={stage.id} className={selection.id === "off" ? "is-bypassed" : ""}><span>{stage.id.toUpperCase()}</span><strong>{selection.label}</strong><small>{selection.id === "off" ? "旁路" : selection.candidate ? "候选技术" : "当前执行"}</small></li>;
            })}
          </ol>
        </section>
        {variant ? <PipelineEvaluation title={variant.label} evaluation={evaluationFor(variant, report)} graphFocus="all" /> : <section className="chain-pending" aria-label="自定义组合评测状态"><Workflow size={23} /><div><h2>这个组合尚未完成整链评测</h2><p>当前配置没有完全匹配的冻结回放。上表展示已评测组合；选择一行可载入对应技术与案例图。自定义组合不会沿用旧分数。</p></div></section>}
      </>}

      <section className="chain-builder chain-export" aria-labelledby="chain-export-title">
        <div className="chain-section-head"><div><span>REPRODUCIBLE CONFIGURATION</span><h2 id="chain-export-title">配置与执行证据</h2><p>导出记录当前完整选择；CLI 模板使用已处理的实际播放参考。执行前替换输入文件、标定延迟和输出目录。</p></div><button className="button secondary" type="button" onClick={copyCommand}><Copy size={14} />复制 CLI 模板</button></div>
        <details><summary>查看当前配置的 CLI 模板</summary><pre><code>{command}</code></pre></details>
        <p className="chain-feedback" role="status">{feedback || "配置与 CLI 导出不会在后台启动处理或应用到设备。"}</p>
        {report ? <dl className="chain-proof"><div><dt>真实回放来源</dt><dd>{report.source}</dd></div><div><dt>CLI SHA256</dt><dd>{report.cli_sha256}</dd></div><div><dt>配对结果 SHA256</dt><dd>{report.results_sha256}</dd></div><div><dt>冻结输入清单 SHA256</dt><dd>{report.manifest_sha256}</dd></div></dl> : null}
      </section>
    </div>
  );
}
