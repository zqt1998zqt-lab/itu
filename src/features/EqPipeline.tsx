import { useMemo, useState } from "react";
import {
  Beaker,
  ChevronRight,
  CircleDot,
  GitBranch,
  Layers3,
  ScanSearch,
  SlidersHorizontal,
  TableProperties,
} from "lucide-react";
import {
  eqAlgorithms,
  eqMatrix,
} from "./eq-pipeline-data";
import type { EqAlgorithm } from "./eq-pipeline-data";
import type { AlgorithmCategory } from "./ns-pipeline-data";
import { FlowDiagram } from "./NsPipeline";
import PipelineEvaluation from "./PipelineEvaluation";
import "./ns-pipeline.css";
import "./eq-pipeline.css";

type Filter = "all" | AlgorithmCategory;

const CATEGORY_META: Record<
  AlgorithmCategory,
  { label: string; note: string }
> = {
  runtime: {
    label: "运行时形状",
    note: "经 ParametricEq 与真实 VCP 执行的形状",
  },
  composite: {
    label: "组合与控制面",
    note: "校准、位置和信号拓扑的组合边界",
  },
  experiment: {
    label: "实验候选",
    note: "已有离线路径但尚未进入默认链路",
  },
  research: {
    label: "仅调研",
    note: "本轮没有实现、CLI 或冻结质量矩阵",
  },
};

const FILTERS: Array<{ id: Filter; label: string }> = [
  { id: "all", label: "全部" },
  { id: "runtime", label: "运行时形状" },
  { id: "composite", label: "组合与控制面" },
  { id: "experiment", label: "实验候选" },
  { id: "research", label: "仅调研" },
];

function categoryIcon(category: AlgorithmCategory) {
  if (category === "runtime") return SlidersHorizontal;
  if (category === "composite") return Layers3;
  if (category === "experiment") return Beaker;
  return ScanSearch;
}

function AlgorithmCard({
  algorithm,
  active,
  onSelect,
}: {
  algorithm: EqAlgorithm;
  active: boolean;
  onSelect: () => void;
}) {
  const Icon = categoryIcon(algorithm.category);
  return (
    <button
      type="button"
      className={`nsp-algo-card${active ? " is-active" : ""}`}
      aria-pressed={active}
      onClick={onSelect}
    >
      <span className={`nsp-algo-icon nsp-algo-icon--${algorithm.category}`}>
        <Icon size={17} strokeWidth={1.7} />
      </span>
      <span className="nsp-algo-copy">
        <span className="nsp-algo-kicker">
          {CATEGORY_META[algorithm.category].label}
        </span>
        <strong>{algorithm.title}</strong>
        <small>{algorithm.short}</small>
        <span
          className={`aec-card-result aec-card-result--${algorithm.evaluation.tone}`}
        >
          评测：{algorithm.evaluation.headline}
        </span>
      </span>
      <span className="nsp-algo-state">{algorithm.status}</span>
      <ChevronRight size={16} aria-hidden="true" />
    </button>
  );
}

export default function EqPipeline() {
  const [filter, setFilter] = useState<Filter>("all");
  const [selectedAlgorithmId, setSelectedAlgorithmId] = useState("vcp-cascade");
  const [selectedNodeId, setSelectedNodeId] = useState("bands");
  const selectedAlgorithm =
    eqAlgorithms.find((item) => item.id === selectedAlgorithmId) ??
    eqAlgorithms[0];
  const filtered = useMemo(
    () =>
      filter === "all"
        ? eqAlgorithms
        : eqAlgorithms.filter((item) => item.category === filter),
    [filter],
  );
  const selectedNode =
    selectedAlgorithm.flow.nodes.find((item) => item.id === selectedNodeId) ??
    selectedAlgorithm.flow.nodes[0];

  return (
    <div className="nsp-page eq-page">
      <section className="nsp-intro">
        <div>
          <span>VCP · UPLINK EQUALIZATION</span>
          <h2>EQ 链路的多种技术路径</h2>
          <p>
            同一 108 个冻结语音案例上，五种真实 EQ 路径都改变了输出，
            但只有最小相位三段形状中的 9 个案例通过了 EQ
            专属配对门限；新增线性相位 FIR 为 0/108。
            频响校准、语音质量和标准符合性是三个不同的问题。
          </p>
        </div>
        <div className="nsp-summary" aria-label="EQ 汇总">
          <div><strong>{eqAlgorithms.length}</strong><span>技术路径</span></div>
          <div><strong>5</strong><span>EQ 形状</span></div>
          <div><strong>{eqMatrix.caseCount}</strong><span>冻结案例</span></div>
          <div><strong>9</strong><span>最高通过数</span></div>
        </div>
      </section>

      <section className="nsp-boundary-row">
        <div className="nsp-bounds">
          <GitBranch size={17} />
          <p>
            <strong>边界：</strong>
            EQ 执行器确实工作：五种形状在 108/108 个案例都改变了输出，削波为
            0。这个事实不代表固定频响适合所有语音；当前门限把同一案例的 EQ
            输出与 bypass 配对，全部结果 standard_compliance=false。
          </p>
        </div>
        <div className="nsp-legend-inline" aria-label="连线图例">
          <span><i className="line audio" />音频</span>
          <span><i className="line control" />参数 / 证据</span>
          <span><i className="line optional" />候选</span>
        </div>
      </section>

      <section className="nsp-catalog" aria-labelledby="eq-catalog-title">
        <div className="nsp-section-head">
          <div>
            <span>EQUALIZATION PATH INDEX</span>
            <h2 id="eq-catalog-title">项目内 EQ 形状、控制面与研究方向</h2>
            <p>
              运行时形状使用 biquad 级联；线性相位 FIR 使用独立对称卷积执行器；
              校准反演是离线控制面。动态、自适应与神经路线仍没有可归入本矩阵的真实实现。
            </p>
          </div>
          <div className="nsp-filters" role="group" aria-label="筛选 EQ 类别">
            {FILTERS.map((item) => (
              <button
                key={item.id}
                type="button"
                aria-pressed={filter === item.id}
                onClick={() => setFilter(item.id)}
              >
                {item.label}
                <span>
                  {item.id === "all"
                    ? eqAlgorithms.length
                    : eqAlgorithms.filter((algorithm) => algorithm.category === item.id)
                        .length}
                </span>
              </button>
            ))}
          </div>
        </div>
        <div className="nsp-catalog-grid">
          {filtered.map((algorithm) => (
            <AlgorithmCard
              key={algorithm.id}
              algorithm={algorithm}
              active={algorithm.id === selectedAlgorithm.id}
              onSelect={() => {
                setSelectedAlgorithmId(algorithm.id);
                setSelectedNodeId(algorithm.flow.nodes[0].id);
              }}
            />
          ))}
        </div>
      </section>

      <section className="eq-matrix" aria-labelledby="eq-matrix-title">
        <div className="eq-matrix-head">
          <div>
            <span>CURRENT SOURCE MATRIX</span>
            <h2 id="eq-matrix-title">五种真实 EQ 形状 · 加 bypass 配对</h2>
            <p>
              评测输出并非跨案例均值：每条 EQ 路径都与相同 original_run +
              scene_id + noise 的 bypass 配对。门限只衡量 EQ 带来的增量贡献。
            </p>
          </div>
          <div className="eq-matrix-stamp">
            <span>当前源码</span>
            <code>{eqMatrix.cliSha256.slice(0, 12)}…</code>
            <small>{eqMatrix.caseCount} cases · 648 runs</small>
          </div>
        </div>
        <div
          className="eq-matrix-scroll"
          role="region"
          aria-label="EQ 配对矩阵，可横向滚动"
          tabIndex={0}
        >
          <table className="eq-matrix-table">
            <thead>
              <tr>
                <th>路径</th>
                <th>配置</th>
                <th>通过</th>
                <th>平均 Δ SI-SDR</th>
                <th>中位 Δ SI-SDR</th>
                <th>平均 Δ STOI</th>
                <th>输出改变</th>
                <th>削波</th>
              </tr>
            </thead>
            <tbody>
              {eqMatrix.variants.map((variant) => (
                <tr
                  key={variant.id}
                  className={variant.pass > 0 ? "is-best" : ""}
                >
                  <th scope="row"><strong>{variant.label}</strong><small>{variant.id}</small></th>
                  <td>{variant.configuration}</td>
                  <td className={variant.pass === 0 ? "eq-negative" : "eq-positive"}>
                    <strong>{variant.pass} / {eqMatrix.caseCount}</strong>
                    <small>{variant.id === "bypass" ? "配对控制" : "工程门限"}</small>
                  </td>
                  <td>{variant.meanSisdr.toFixed(3)} dB</td>
                  <td>{variant.medianSisdr.toFixed(3)} dB</td>
                  <td>{variant.meanStoi.toFixed(6)}</td>
                  <td>{variant.changed} / 108</td>
                  <td>{variant.clipped} / 108</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="eq-matrix-notes">
          <p>
            <strong>门限：</strong>{eqMatrix.gate}。这是 EQ
            专属增量诊断，不是完整 ITU 验收。三段形状的 9 个通过案例全部在 16
            kHz，且只覆盖 6 个场景；48 kHz 最坏案例 ΔSI-SDR 为 −14.022 dB。
          </p>
          <p>
            <strong>不要误读：</strong>STOI 几乎不变不等于 EQ
            无影响；byte hash 已证明输出全部不同。校准反演的
            response-mask 通过也不表示自然语音质量通过。
          </p>
        </div>
        <div className="eq-build-proof">
          <div><span>有效结果</span><code>{eqMatrix.source}</code></div>
          <div><span>结果 SHA256</span><code>{eqMatrix.resultsSha256.slice(0, 20)}…</code></div>
          <div><span>运行覆盖</span><strong>16 / 24 / 48 kHz</strong></div>
        </div>
      </section>

      <section className="eq-layer-card" aria-labelledby="eq-layer-title">
        <div className="eq-layer-head">
          <div>
            <span>EVIDENCE LAYERS</span>
            <h2 id="eq-layer-title">频响、语音、标准不能混成一条结论</h2>
            <p>
              运行时 biquad、一次性校准闭环、冻结语音矩阵与正式认证分别对应不同的证据。
            </p>
          </div>
          <TableProperties size={22} />
        </div>
        <div className="eq-layer-grid">
          <article>
            <span>01 · RESPONSE</span>
            <h3>校准反演</h3>
            <strong>
              {eqMatrix.calibration.beforeMargin.toFixed(4)} →{" "}
              {eqMatrix.calibration.remeasuredMargin.toFixed(4)} dB
            </strong>
            <p>
              拟合与重测通过 ±0.25 dB 的合成数字诊断掩膜；主反演段为
              peak,250,−5.985,1。输入是合成共振，不是车舱 RIR。
            </p>
          </article>
          <article>
            <span>02 · SPEECH</span>
            <h3>同一反演参数的泛化</h3>
            <strong>0 / 108</strong>
            <p>
              相同参数在自然语音矩阵没有一对案例通过配对门限。calibration
              与语音评测是两个不同证据域。
            </p>
          </article>
          <article>
            <span>03 · ENGINEERING</span>
            <h3>固定目标形状</h3>
            <strong>0–9 / 108</strong>
            <p>
              各形状全部改变输出且 0 削波；三段形状最高 9/108。固定 EQ
              需要场景或车型特定目标才有机会通过。
            </p>
          </article>
          <article>
            <span>04 · COMPLIANCE</span>
            <h3>正式标准</h3>
            <strong>未覆盖</strong>
            <p>
              没有车舱声学标定、设备链路、受控播放、POLQA 或正式 ITU-T
              验收；standard_compliance=false。
            </p>
          </article>
        </div>
      </section>

      <section className="eq-target-card">
        <div>
          <span>TARGET RESPONSE DIAGNOSTIC</span>
          <h2>目标曲线 RMS 误差只用于响应诊断</h2>
          <p>
            目标曲线是人工指定的“低中高相对增益”；没有资料证明它是本语料的最优目标。因此不按 RMS
            最小挑赢家，更不把 bypass 的 1.509 dB 解读为“语音质量最好”。
          </p>
        </div>
        <div
          className="eq-target-scroll"
          role="region"
          aria-label="目标响应误差表，可横向滚动"
          tabIndex={0}
        >
          <table>
            <thead><tr><th>形状</th><th>RMS error</th><th>Max error</th><th>Max boost</th><th>Max cut</th></tr></thead>
            <tbody>
              {eqMatrix.targetResponse.map((item) => (
                <tr key={item.label}>
                  <th scope="row">{item.label}</th>
                  <td>{item.rms.toFixed(3)} dB</td>
                  <td>{item.max.toFixed(3)} dB</td>
                  <td>{item.boost.toFixed(3)} dB</td>
                  <td>{item.cut.toFixed(3)} dB</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="nsp-selected" aria-live="polite">
        <div className="nsp-selected-main">
          <div className="nsp-selected-kicker">
            <CircleDot size={13} />
            {CATEGORY_META[selectedAlgorithm.category].label}
            <span>{selectedAlgorithm.status}</span>
          </div>
          <h2>{selectedAlgorithm.title}</h2>
          <p>{selectedAlgorithm.summary}</p>
        </div>
        <dl className="nsp-selected-meta">
          <div><dt>采样率</dt><dd>{selectedAlgorithm.rate}</dd></div>
          <div><dt>入口</dt><dd><code>{selectedAlgorithm.entry}</code></dd></div>
          <div><dt>配置</dt><dd><code>{selectedAlgorithm.cli}</code></dd></div>
        </dl>
        <div className="nsp-selected-boundary">
          <strong>工程边界</strong>
          <span>{selectedAlgorithm.boundary}</span>
        </div>
      </section>

      <PipelineEvaluation
        title={selectedAlgorithm.title}
        evaluation={selectedAlgorithm.evaluation}
        graphFocus="eq"
      />

      <figure className="nsp-figure">
        <div className="nsp-head">
          <div>
            <span>{selectedAlgorithm.flow.eyebrow}</span>
            <h2>{selectedAlgorithm.flow.title}</h2>
            <p>{selectedAlgorithm.flow.description}</p>
          </div>
          <label className="nsp-select">
            <span>查看节点</span>
            <select
              value={selectedNode.id}
              onChange={(event) => setSelectedNodeId(event.target.value)}
            >
              {selectedAlgorithm.flow.nodes.map((node) => (
                <option key={node.id} value={node.id}>{node.title}</option>
              ))}
            </select>
          </label>
        </div>
        <div
          className="nsp-scroll"
          role="region"
          aria-label={`${selectedAlgorithm.title} EQ 工作流，可横向滚动`}
          tabIndex={0}
        >
          <FlowDiagram
            flow={selectedAlgorithm.flow}
            selectedId={selectedNode.id}
            onSelect={setSelectedNodeId}
          />
        </div>
        <div className="nsp-legend">
          <span><i className="line audio" />音频</span>
          <span><i className="line control" />控制 / 证据</span>
          <span><i className="line optional" />实验 / 候选</span>
          <span><i className="box" />当前节点</span>
        </div>
      </figure>

      <section className="nsp-detail" aria-live="polite" aria-atomic="true">
        <div className="nsp-detail-copy">
          <span>{selectedNode.tag}</span>
          <h2>{selectedNode.title}</h2>
          <p>{selectedNode.body}</p>
          <code>{selectedNode.formula}</code>
        </div>
        <dl>
          <div><dt>实现</dt><dd>{selectedNode.where}</dd></div>
          <div><dt>条件</dt><dd>{selectedNode.condition}</dd></div>
        </dl>
        <div className="nsp-fields">
          <span>实时字段</span>
          {selectedNode.fields.map((field) => <code key={field}>{field}</code>)}
        </div>
      </section>
    </div>
  );
}
