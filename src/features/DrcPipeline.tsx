import { useMemo, useState } from "react";
import {
  Beaker,
  ChevronRight,
  CircleDot,
  Gauge,
  GitBranch,
  Layers3,
  ScanSearch,
  ShieldAlert,
  TableProperties,
} from "lucide-react";
import { drcAlgorithms, drcMatrix } from "./drc-pipeline-data";
import type { DrcAlgorithm } from "./drc-pipeline-data";
import type { AlgorithmCategory } from "./ns-pipeline-data";
import { FlowDiagram } from "./NsPipeline";
import PipelineEvaluation from "./PipelineEvaluation";
import "./ns-pipeline.css";
import "./drc-pipeline.css";

type Filter = "all" | AlgorithmCategory;

const CATEGORY_META: Record<
  AlgorithmCategory,
  { label: string; note: string }
> = {
  runtime: {
    label: "运行时后级",
    note: "TX DRC 与独立终级 limiter",
  },
  composite: {
    label: "组合与下行",
    note: "TX/RX 状态分离和系统保护边界",
  },
  experiment: {
    label: "实验配置",
    note: "已有真实 CLI 路径但未准入",
  },
  research: {
    label: "仅调研",
    note: "多段、前瞻、true-peak 与响度路线",
  },
};

const FILTERS: Array<{ id: Filter; label: string }> = [
  { id: "all", label: "全部" },
  { id: "runtime", label: "运行时" },
  { id: "composite", label: "组合 / RX" },
  { id: "experiment", label: "实验" },
  { id: "research", label: "仅调研" },
];

function categoryIcon(category: AlgorithmCategory) {
  if (category === "runtime") return Gauge;
  if (category === "composite") return Layers3;
  if (category === "experiment") return Beaker;
  return ScanSearch;
}

function AlgorithmCard({
  algorithm,
  active,
  onSelect,
}: {
  algorithm: DrcAlgorithm;
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

export default function DrcPipeline() {
  const [filter, setFilter] = useState<Filter>("all");
  const [selectedAlgorithmId, setSelectedAlgorithmId] = useState("default");
  const [selectedNodeId, setSelectedNodeId] = useState("detector");
  const selectedAlgorithm =
    drcAlgorithms.find((item) => item.id === selectedAlgorithmId) ??
    drcAlgorithms[0];
  const filtered = useMemo(
    () =>
      filter === "all"
        ? drcAlgorithms
        : drcAlgorithms.filter((item) => item.category === filter),
    [filter],
  );
  const selectedNode =
    selectedAlgorithm.flow.nodes.find((node) => node.id === selectedNodeId) ??
    selectedAlgorithm.flow.nodes[0];
  const integrated = drcAlgorithms.filter(
    (item) => item.category === "runtime" || item.category === "composite",
  ).length;
  const experiments = drcAlgorithms.filter(
    (item) => item.category === "experiment",
  ).length;
  const research = drcAlgorithms.filter(
    (item) => item.category === "research",
  ).length;

  const selectAlgorithm = (algorithm: DrcAlgorithm) => {
    setSelectedAlgorithmId(algorithm.id);
    const preferred =
      algorithm.flow.nodes.find((node) =>
        ["input", "detector", "knee", "analysis"].includes(node.id),
      ) ?? algorithm.flow.nodes[0];
    setSelectedNodeId(preferred.id);
  };

  return (
    <div className="nsp-page drc-page">
      <section className="nsp-intro">
        <div>
          <span>UPLINK / DOWNLINK · DYNAMIC RANGE CONTROL</span>
          <h2>从软拐点压缩到峰值保护的多种 DRC 路径</h2>
          <p>
            同一 108 个冻结语音案例上，默认、gentle、aggressive、fast
            与新增 look-ahead 路径都真实改变 PCM。look-ahead 在 16/48 kHz
            通过语音、峰值与 3 ms 新增延迟门限，但 24 kHz 新增延迟为
            4–4.27 ms，因此该采样率下 27 个案例全部失败。
          </p>
        </div>
        <div className="nsp-summary" aria-label="DRC 汇总">
          <div><strong>{drcAlgorithms.length}</strong><span>技术路径</span></div>
          <div><strong>{integrated}</strong><span>运行时 / 组合</span></div>
          <div><strong>{experiments}</strong><span>实验配置</span></div>
          <div><strong>{research}</strong><span>仅调研</span></div>
        </div>
      </section>

      <section className="nsp-boundary-row">
        <div className="nsp-bounds">
          <GitBranch size={17} />
          <p>
            <strong>边界：</strong>
            TX DRC、DRC 内部 tanh 软饱和、AudioEngine 独立硬 ceiling 和 RX
            扬声器保护是四个不同层次。本矩阵只隔离 TX/post-stage PCM；
            limiter_only 从未实际触发，不能把 108/108 写成保护机制已验证。
          </p>
        </div>
        <div className="nsp-legend-inline" aria-label="连线图例">
          <span><i className="line audio" />TX PCM</span>
          <span><i className="line control" />包络 / 增益</span>
          <span><i className="line optional" />未覆盖 / 候选</span>
        </div>
      </section>

      <section className="nsp-catalog" aria-labelledby="drc-catalog-title">
        <div className="nsp-section-head">
          <div>
            <span>DYNAMIC RANGE INDEX</span>
            <h2 id="drc-catalog-title">TX / RX DRC、终级保护与研究方向</h2>
            <p>
              DynamicRangeCompressor 使用因果样本包络、soft-knee、固定 5%
              gain slew 和 85% ceiling 处的 tanh 软饱和。终级硬 limiter
              在 DRC 外单独执行；RX 复用压缩器类但拥有独立状态，本轮未做声学回放。
            </p>
          </div>
          <div className="nsp-filters" role="group" aria-label="筛选 DRC 类别">
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
                    ? drcAlgorithms.length
                    : drcAlgorithms.filter(
                        (algorithm) => algorithm.category === item.id,
                      ).length}
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
              onSelect={() => selectAlgorithm(algorithm)}
            />
          ))}
        </div>
      </section>

      <section className="drc-matrix" aria-labelledby="drc-matrix-title">
        <div className="drc-matrix-head">
          <div>
            <span>CURRENT SOURCE DRC MATRIX</span>
            <h2 id="drc-matrix-title">
              108 案例 · bypass 与七条后级路径
            </h2>
            <p>
              16/24/48 kHz 各 36 个场景；−45/−30/−15 dBFS 各 36 个。
              每条路径与相同 original_run + scene_id + noise 的 bypass
              配对，语音贡献与完整输出峰值分别测量。
            </p>
          </div>
          <div className="drc-matrix-stamp">
            <span>当前源码</span>
            <code>{drcMatrix.cliSha256.slice(0, 12)}…</code>
            <small>108 cases · 864 runs</small>
          </div>
        </div>
        <div
          className="drc-matrix-scroll"
          role="region"
          aria-label="DRC 配对矩阵，可横向滚动"
          tabIndex={0}
        >
          <table className="drc-matrix-table">
            <thead>
              <tr>
                <th>路径</th>
                <th>配置</th>
                <th>工程通过</th>
                <th>平均 Δ SI-SDR</th>
                <th>中位 Δ SI-SDR</th>
                <th>平均 Δ STOI</th>
                <th>平均峰值下降</th>
                <th>输出改变</th>
                <th>削波</th>
              </tr>
            </thead>
            <tbody>
              {drcMatrix.variants.map((variant) => (
                <tr
                  key={variant.id}
                  className={
                    variant.id === "limiter_only"
                      ? "is-uncovered"
                      : variant.control
                        ? "is-control"
                        : variant.pass >= 76
                          ? "is-best"
                          : ""
                  }
                >
                  <th scope="row">
                    <strong>{variant.label}</strong>
                    <small>{variant.id}</small>
                  </th>
                  <td>{variant.configuration}</td>
                  <td
                    className={
                      variant.id === "limiter_only"
                        ? "drc-uncovered"
                        : variant.pass <= 49
                          ? "drc-negative"
                          : "drc-positive"
                    }
                  >
                    <strong>{variant.pass} / 108</strong>
                    <small>
                      {variant.id === "limiter_only"
                        ? "未触发，不等同验证"
                        : variant.control
                          ? "自配对控制"
                          : "语音 / 峰值 / 时延门限"}
                    </small>
                  </td>
                  <td>{variant.meanSisdr.toFixed(3)} dB</td>
                  <td>{variant.medianSisdr.toFixed(3)} dB</td>
                  <td>{variant.meanStoi.toFixed(6)}</td>
                  <td>{variant.meanPeakReduction.toFixed(3)} dB</td>
                  <td>{variant.changed} / 108</td>
                  <td>{variant.clipped} / 108</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="drc-matrix-notes">
          <p>
            <strong>门限：</strong>{drcMatrix.gate}。所有路径 0 削波；
            默认、gentle、aggressive、fast 与 look-ahead 分别在
            53、57、72、60、108 个案例真实改变输出。
          </p>
          <p>
            <strong>覆盖警告：</strong>limiter_only 的 108/108 来自 0 个触发、
            0 dB 峰值下降和输出 SHA 全相同。终级 limiter 属于
            “实现存在、证据未覆盖”，不是当前源码的成功通过。
          </p>
        </div>
        <div className="drc-build-proof">
          <div><span>有效结果</span><code>{drcMatrix.source}</code></div>
          <div><span>结果 SHA256</span><code>{drcMatrix.resultsSha256.slice(0, 20)}…</code></div>
          <div><span>采样率</span><strong>16 / 24 / 48 kHz</strong></div>
        </div>
      </section>

      <section className="drc-layer-card" aria-labelledby="drc-layer-title">
        <div className="drc-layer-head">
          <div>
            <span>EVIDENCE LAYERS</span>
            <h2 id="drc-layer-title">压缩曲线、增益瞬态、峰值与声学保护要分开</h2>
            <p>
              aggressive 把平均峰值下降提高到 2.133 dB，也把平均 SI-SDR
              推到 −3.294 dB；fast 提高峰值控制，同时比默认更伤语音。
            </p>
          </div>
          <ShieldAlert size={22} />
        </div>
        <div className="drc-layer-grid">
          <article>
            <span>01 · CURVE</span>
            <h3>Soft knee</h3>
            <strong>4 组真实配置</strong>
            <p>
              阈值、比例、拐点与 attack/release 均由 CLI
              实际传入并回显；参数范围在 AudioEngine 校验。
            </p>
          </article>
          <article>
            <span>02 · SLEW</span>
            <h3>固定 5% 平滑</h3>
            <strong>未可配置</strong>
            <p>
              包络之后还有独立线性 gain slew。Look-ahead 只延迟音频，
              不会自动绕过这层增益平滑。
            </p>
          </article>
          <article>
            <span>03 · HIGH LEVEL</span>
            <h3>−15 dBFS</h3>
            <strong>0–4 / 36</strong>
            <p>
              默认、aggressive、fast 为 0/36，gentle 仅 4/36；
              低电平通过多数来自压缩未触发。
            </p>
          </article>
          <article>
            <span>04 · PEAK</span>
            <h3>Sample peak</h3>
            <strong>非 true peak</strong>
            <p>
              没有过采样、编码器 round-trip 或扬声器回放保护；
              不能宣称 true-peak 与 RX 声学达标。
            </p>
          </article>
        </div>
      </section>

      <section className="drc-level-card">
        <div>
          <span>LEVEL / PROFILE READOUT</span>
          <h2>低电平“通过”不能替代高电平语音证据</h2>
          <p>
            工程门限分三层记录。limiter_only 在所有电平都未触发；实体 DRC
            的通过数随信号电平迅速变化。这个表用于解释总通过数，而不是掩盖弱项。
          </p>
        </div>
        <div
          className="drc-level-scroll"
          role="region"
          aria-label="DRC 近端电平覆盖表，可横向滚动"
          tabIndex={0}
        >
          <table>
            <thead>
              <tr><th>路径</th><th>−45 dBFS</th><th>−30 dBFS</th><th>−15 dBFS</th><th>真实改变</th></tr>
            </thead>
            <tbody>
              <tr><th scope="row">默认</th><td>36/36</td><td>36/36</td><td className="drc-negative">0/36</td><td>53/108</td></tr>
              <tr><th scope="row">Gentle</th><td>36/36</td><td>36/36</td><td className="drc-negative">4/36</td><td>57/108</td></tr>
              <tr><th scope="row">Aggressive</th><td>36/36</td><td>13/36</td><td className="drc-negative">0/36</td><td>72/108</td></tr>
              <tr><th scope="row">Fast</th><td>36/36</td><td>36/36</td><td className="drc-negative">0/36</td><td>60/108</td></tr>
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
        graphFocus="drc"
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
          aria-label={`${selectedAlgorithm.title} DRC 工作流，可横向滚动`}
          tabIndex={0}
        >
          <FlowDiagram
            flow={selectedAlgorithm.flow}
            selectedId={selectedNode.id}
            onSelect={setSelectedNodeId}
          />
        </div>
        <div className="nsp-legend">
          <span><i className="line audio" />TX 音频</span>
          <span><i className="line control" />包络 / 增益</span>
          <span><i className="line optional" />终级保护 / 候选</span>
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
