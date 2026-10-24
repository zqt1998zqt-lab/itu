import { Fragment, useId, useState } from "react";
import { ArrowDown, ArrowRight, Check, FileAudio, Info } from "lucide-react";
import { AUDIO_ROLES } from "../domain/audio";
import type { AudioFiles } from "../domain/audio";
import { DIAGRAMS, renderDiagram } from "../domain/flow-diagram";
import type { DiagramMode } from "../domain/flow-diagram";
import { FLOW_NODE_DETAILS } from "./audio-flow-details";
import "./audio-flow.css";

const NEEDS = [
  { title: "基础电平与频谱", inputs: "处理后音频即可", condition: "选择实际文件与分析声道", note: "无需先运行降噪或 AEC" },
  { title: "时延与语音保真", inputs: "干净参考 + 待测录音", condition: "同一语音、有效比较段；STOI 需真实语音确认", note: "前后改善量另需处理前录音、共有参考段和时钟 / 电平确认" },
  { title: "降噪效果", inputs: "处理前 + 处理后", condition: "纯噪声段 + 时钟 / 电平确认", note: "SNR 另需含语音的比较段及平稳噪声假设；联合保真与增益判断" },
  { title: "回声片段衰减", inputs: "处理前 + 处理后 + 播放参考", condition: "远端单讲段 + 条件确认 + 相关性检查", note: "得到整条链路的总功率衰减" },
  { title: "AEC ERLE 工程估计", inputs: "AEC 输入 / 输出 + 播放参考", condition: "满足回声片段条件，并确认 AEC 测量点", note: "排除后级处理；未执行完整标准测试" },
];

export default function AudioFlow({ files }: { files: AudioFiles }) {
  const [mode, setMode] = useState<DiagramMode>("device");
  const [selected, setSelected] = useState("before");
  const markerId = useId().replace(/:/g, "");
  const diagrams = mode === "device" ? [DIAGRAMS.device, DIAGRAMS.website] : [DIAGRAMS.website];
  const nodes = diagrams.flatMap((diagram) => diagram.nodes);
  const zones = diagrams.flatMap((diagram) => diagram.zones.map((zone, index) => ({ ...zone, key: `${diagram.mode}-${index}` })));
  const title = mode === "device" ? "完整信号链：从声源到指标报告" : DIAGRAMS.website.title;
  const node = { ...nodes.find((item) => item.id === selected)!, ...FLOW_NODE_DETAILS[selected] };
  const Icon = node.icon;
  const count = AUDIO_ROLES.filter((role) => files[role]).length;
  function changeMode(next: DiagramMode) {
    setMode(next);
    setSelected(next === "device" ? "before" : "upload");
  }

  return (
    <div className="audio-flow-page">
      <div className="audio-flow-context">
        <Info size={18} aria-hidden="true" />
        <p><strong>设备负责处理，网站负责测量。</strong> AEC、BF、DTD 及后级模块是设备链路示意，启用与顺序以实际实现为准；网站不执行这些处理算法。各指标独立检查条件，单文件电平与 PSD 不要求参考音频或对齐。</p>
      </div>
      <div className="audio-flow-toolbar">
        <div className="audio-flow-modes" role="group" aria-label="流程图视角">
          <button type="button" aria-pressed={mode === "device"} onClick={() => changeMode("device")}><span>01</span>完整信号链</button>
          <button type="button" aria-pressed={mode === "website"} onClick={() => changeMode("website")}><span>02</span>指标计算细节</button>
        </div>
        <a href="/diagrams/nv200-audio-flow.html" target="_blank" rel="noreferrer" className="text-link">查看独立页面 <ArrowRight size={14} aria-hidden="true" /></a>
      </div>
      <nav className="audio-flow-stage-nav" aria-label="流程阶段导航">
        <span>跳到阶段</span>{zones.map((zone, index) => <button type="button" key={zone.key} onClick={() => document.querySelectorAll(".audio-flow-board .df-zone")[index]?.scrollIntoView({ block: "start", inline: "nearest" })}>{zone.title.split(" · ")[0].replace(/^\d+\s+/, "")} <ArrowDown size={12} aria-hidden="true" /></button>)}
      </nav>
      <figure className="audio-flow-figure" aria-labelledby="audio-flow-title" aria-describedby="audio-flow-caption">
        <div className="audio-flow-heading">
          <div><h2 id="audio-flow-title">{title}</h2><p>{nodes.length} 个关键节点 · {mode === "device" ? "上图展开 AEC、BF、DTD 与后级处理，下图保留全部指标计算与最终报告。" : "解码、条件确认、对齐与有效片段，再分支计算五类指标并汇总报告。"}</p></div>
          <span className="audio-flow-count">已选择录音 <strong>{count} / 4</strong></span>
        </div>
        <div className="audio-flow-recordings" aria-label="四路录音选择状态">
          {AUDIO_ROLES.map((role) => <span key={role} className={files[role] ? "has-file" : ""}><span aria-hidden="true">{files[role] ? <Check size={13} /> : <FileAudio size={13} />}</span>{({ reference: "干净参考", before: "处理前", after: "处理后", echo: "播放参考" })[role]}<small>{files[role] ? "已选择" : "待选择"}</small></span>)}
        </div>
        <p className="audio-flow-scroll-hint">点击节点查看说明；小屏可在画布内左右滑动，连线与字体保持完整。</p>
        {diagrams.map((diagram, index) => <Fragment key={diagram.mode}>
          {index > 0 ? <div className="audio-flow-handoff"><ArrowDown size={20} aria-hidden="true" /><p><strong>四路实际录音进入下方指标计算</strong><span>干净参考、处理前、处理后、播放参考；单独测 ERLE 时，滤波器输入 / 输出替换前后测点。</span></p></div> : null}
          <div className="audio-flow-scroll" role="region" aria-label={`${diagram.title}，可横向滚动`} tabIndex={0}>
            <div className="audio-flow-board" style={{ width: diagram.width, height: diagram.height }}>
              <div dangerouslySetInnerHTML={{ __html: renderDiagram(diagram, { prefix: `nv200-${diagram.mode}-${markerId}`, selected, uploaded: AUDIO_ROLES.filter((role) => files[role]) }) }} />
              {diagram.nodes.map((item) => <button key={item.id} type="button" className="audio-flow-hit"
                style={{ left: item.x, top: item.y, width: item.width, height: item.height }}
                aria-label={`查看${item.title}节点说明`} aria-pressed={selected === item.id} aria-controls="audio-flow-node-detail"
                onClick={() => setSelected(item.id)}><span className="sr-only">{item.title}</span></button>)}
            </div>
          </div>
        </Fragment>)}
        <figcaption id="audio-flow-caption">
          <label>当前节点<select aria-label="选择流程图节点" value={selected} onChange={(event) => setSelected(event.target.value)}>{nodes.map((item) => <option key={item.id} value={item.id}>{item.title}</option>)}</select></label>
          <button type="button" className="text-link" onClick={() => document.getElementById("audio-flow-node-detail")?.scrollIntoView({ block: "start" })}>查看节点说明 <ArrowDown size={14} aria-hidden="true" /></button>
        </figcaption>
      </figure>

      <section id="audio-flow-node-detail" className="audio-flow-detail" aria-live="polite" aria-atomic="true" aria-label="所选节点说明">
        <div className="audio-flow-detail-title"><span><Icon size={21} aria-hidden="true" /></span><div><small>{node.kind}</small><h2>{node.title}</h2></div>
          {node.role ? <a href="#workbench" className="button secondary">选择录音 <ArrowRight size={14} aria-hidden="true" /></a> : null}
        </div>
        <div className="audio-flow-detail-copy"><div><h3>这个节点做什么</h3><p>{node.description}</p></div><div><h3>需要什么条件</h3><p>{node.requirements}</p></div></div>
        <div className="audio-flow-detail-metrics"><span>关联指标</span>{node.metrics.map((metric) => <span key={metric}>{metric}</span>)}</div>
        {node.role && files[node.role] ? <p className="audio-flow-selected-file">当前选择：{files[node.role]!.name}（仅表示文件已选择，计算有效性需另行检查）</p> : null}
      </section>

      <section className="audio-flow-needs" aria-labelledby="audio-flow-needs-title">
        <div className="audio-flow-heading"><div><h2 id="audio-flow-needs-title">每类指标需要什么</h2><p>文件齐全只是第一步，片段和测量条件也必须满足。</p></div></div>
        <dl className="audio-flow-needs-list">{NEEDS.map((need) => <div key={need.title}><dt>{need.title}</dt><dd><strong>录音输入</strong><p>{need.inputs}</p></dd><dd><strong>有效条件</strong><p>{need.condition}</p><small>{need.note}</small></dd></div>)}</dl>
        <p className="audio-flow-standard-note">原表的 MOS、SLR / RLR、TCL 与正式合规曲线另需专用算法、测试配置或声学标定；当前工程计算不等于完整 ITU 认证。<a href="#metrics">查看指标与方法 <ArrowRight size={13} aria-hidden="true" /></a></p>
      </section>
    </div>
  );
}
