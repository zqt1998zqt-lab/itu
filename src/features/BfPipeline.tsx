import { useMemo, useState } from "react";
import {
  Beaker,
  ChevronRight,
  CircleDot,
  GitBranch,
  Layers3,
  Radar,
  ScanSearch,
  TableProperties,
} from "lucide-react";
import { bfAlgorithms, bfMatrix } from "./bf-pipeline-data";
import type { BfAlgorithm } from "./bf-pipeline-data";
import type { AlgorithmCategory } from "./ns-pipeline-data";
import { FlowDiagram } from "./NsPipeline";
import PipelineEvaluation from "./PipelineEvaluation";
import "./ns-pipeline.css";
import "./bf-pipeline.css";

type Filter = "all" | AlgorithmCategory;

const CATEGORY_META: Record<
  AlgorithmCategory,
  { label: string; note: string }
> = {
  runtime: {
    label: "运行时空间路径",
    note: "真实 SpatialFrontend 与 CLI 可执行",
  },
  composite: {
    label: "组合与控制面",
    note: "低频风压与安全回退组合",
  },
  experiment: {
    label: "实验候选",
    note: "项目内有真实 CLI 路径但未准入",
  },
  research: {
    label: "仅调研",
    note: "没有模型、实现或冻结矩阵",
  },
};

const FILTERS: Array<{ id: Filter; label: string }> = [
  { id: "all", label: "全部" },
  { id: "runtime", label: "运行时" },
  { id: "composite", label: "组合" },
  { id: "experiment", label: "实验" },
  { id: "research", label: "仅调研" },
];

function categoryIcon(category: AlgorithmCategory) {
  if (category === "runtime") return Radar;
  if (category === "composite") return Layers3;
  if (category === "experiment") return Beaker;
  return ScanSearch;
}

function AlgorithmCard({
  algorithm,
  active,
  onSelect,
}: {
  algorithm: BfAlgorithm;
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

export default function BfPipeline() {
  const [filter, setFilter] = useState<Filter>("all");
  const [selectedAlgorithmId, setSelectedAlgorithmId] = useState("das");
  const [selectedNodeId, setSelectedNodeId] = useState("capture");
  const selectedAlgorithm =
    bfAlgorithms.find((item) => item.id === selectedAlgorithmId) ??
    bfAlgorithms[0];
  const filtered = useMemo(
    () =>
      filter === "all"
        ? bfAlgorithms
        : bfAlgorithms.filter((item) => item.category === filter),
    [filter],
  );
  const selectedNode =
    selectedAlgorithm.flow.nodes.find((node) => node.id === selectedNodeId) ??
    selectedAlgorithm.flow.nodes[0];
  const integrated = bfAlgorithms.filter(
    (item) => item.category === "runtime" || item.category === "composite",
  ).length;
  const experiments = bfAlgorithms.filter(
    (item) => item.category === "experiment",
  ).length;
  const research = bfAlgorithms.filter(
    (item) => item.category === "research",
  ).length;

  const selectAlgorithm = (algorithm: BfAlgorithm) => {
    setSelectedAlgorithmId(algorithm.id);
    const preferred =
      algorithm.flow.nodes.find((node) =>
        ["capture", "wola", "weights", "model"].includes(node.id),
      ) ?? algorithm.flow.nodes[0];
    setSelectedNodeId(preferred.id);
  };

  return (
    <div className="nsp-page bf-page">
      <section className="nsp-intro">
        <div>
          <span>UPLINK · MULTI-MICROPHONE BEAMFORMING</span>
          <h2>从延迟求和到数据驱动空间零点</h2>
          <p>
            同一 72 个冻结多麦空间场案例上，DAS、观测/固定导向 MVDR、
            连续 MPDR 和两条风压组合路径都真实改变了 PCM。
            最高工程通过数为 36/72，且通过集合集中在 Corcos 风场；当前结论不能外推为车舱实测阵列准入。
          </p>
        </div>
        <div className="nsp-summary" aria-label="BF 汇总">
          <div><strong>{bfAlgorithms.length}</strong><span>技术路径</span></div>
          <div><strong>{integrated + experiments}</strong><span>真实 BF 路径</span></div>
          <div><strong>{bfMatrix.caseCount}</strong><span>冻结场景</span></div>
          <div><strong>36</strong><span>最高通过数</span></div>
        </div>
      </section>

      <section className="nsp-boundary-row">
        <div className="nsp-bounds">
          <GitBranch size={17} />
          <p>
            <strong>目标语义：</strong>
            no-BF 控制用 clean channel 0；BF 路由用声明因果延迟下的 calibrated
            delay-sum clean beam。两条路径的“语音目标”故意不同，因此这里评的是 BF
            对各自目标的贡献，不能把 36/72 直接解释为任意车内目标都已通过。
          </p>
        </div>
        <div className="nsp-legend-inline" aria-label="连线图例">
          <span><i className="line audio" />多麦音频</span>
          <span><i className="line control" />协方差 / 门限</span>
          <span><i className="line optional" />候选 / 风压</span>
        </div>
      </section>

      <section className="nsp-catalog" aria-labelledby="bf-catalog-title">
        <div className="nsp-section-head">
          <div>
            <span>SPATIAL PATH INDEX</span>
            <h2 id="bf-catalog-title">项目内 BF 路由、组合控制与研究方向</h2>
            <p>
              SpatialFrontend 已实现 WOLA、噪声训练协方差、distortionless
              MVDR、连续混合协方差、对角加载、权范数上限、DAS 回退与低频风压权。
              神经波束、波束域后置与多麦神经后置没有当前实现。
            </p>
          </div>
          <div className="nsp-filters" role="group" aria-label="筛选 BF 类别">
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
                    ? bfAlgorithms.length
                    : bfAlgorithms.filter(
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

      <section className="bf-matrix" aria-labelledby="bf-matrix-title">
        <div className="bf-matrix-head">
          <div>
            <span>CURRENT SOURCE BF MATRIX</span>
            <h2 id="bf-matrix-title">
              72 个空间场 · 7 条控制/波束路径
            </h2>
            <p>
              36 个 2 麦与 36 个 4 麦场景，每个 SNR
              0/15/30 dB 各 24 个；球面、柱面、Corcos 与规则场共存。
              每条路由都从同一个显式 no-BF 控制配对。
            </p>
          </div>
          <div className="bf-matrix-stamp">
            <span>当前源码</span>
            <code>{bfMatrix.cliSha256.slice(0, 12)}…</code>
            <small>72 cases · 504 runs</small>
          </div>
        </div>
        <div
          className="bf-matrix-scroll"
          role="region"
          aria-label="BF 配对矩阵，可横向滚动"
          tabIndex={0}
        >
          <table className="bf-matrix-table">
            <thead>
              <tr>
                <th>路径</th>
                <th>权重 / 控制</th>
                <th>工程通过</th>
                <th>平均 Δ SI-SDR</th>
                <th>中位 Δ SI-SDR</th>
                <th>平均 Δ STOI</th>
                <th>噪声增量</th>
                <th>输出改变</th>
                <th>削波</th>
              </tr>
            </thead>
            <tbody>
              {bfMatrix.variants.map((variant) => (
                <tr
                  key={variant.id}
                  className={
                    variant.control
                      ? "is-control"
                      : variant.pass >= 36
                        ? "is-best"
                        : ""
                  }
                >
                  <th scope="row">
                    <strong>{variant.label}</strong>
                    <small>{variant.id}</small>
                  </th>
                  <td>{variant.configuration}</td>
                  <td className={variant.pass <= 26 ? "bf-negative" : "bf-positive"}>
                    <strong>{variant.pass} / 72</strong>
                    <small>{variant.control ? "自配对控制" : "三项门限"}</small>
                  </td>
                  <td>{variant.meanSisdr.toFixed(3)} dB</td>
                  <td>{variant.medianSisdr.toFixed(3)} dB</td>
                  <td>{variant.meanStoi.toFixed(6)}</td>
                  <td>{variant.meanNoise.toFixed(3)} dB</td>
                  <td>{variant.changed} / 72</td>
                  <td>{variant.clipped} / 72</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="bf-matrix-notes">
          <p>
            <strong>门限：</strong>{bfMatrix.gate}。缺失指标不通过；多声源场景没有
            near_late，脚本显式使用 near_only 并记录 speech_window，不按跨案例均值替代。
          </p>
          <p>
            <strong>覆盖事实：</strong>六条真实 BF 路由的 432/432
            个输出全部改变且 0 削波。观测 MVDR 与 MVDR+风压同为 36/72；
            固定导向为 34，连续 MPDR 为 33，DAS 与 DAS+风压为 26。
          </p>
        </div>
        <div className="bf-build-proof">
          <div><span>有效结果</span><code>{bfMatrix.source}</code></div>
          <div><span>结果 SHA256</span><code>{bfMatrix.resultsSha256.slice(0, 20)}…</code></div>
          <div><span>空间场</span><strong>2/4 麦 · 0/15/30 dB</strong></div>
        </div>
      </section>

      <section className="bf-layer-card" aria-labelledby="bf-layer-title">
        <div className="bf-layer-head">
          <div>
            <span>EVIDENCE LAYERS</span>
            <h2 id="bf-layer-title">空间抑制、语音目标和设备几何必须分层</h2>
            <p>
              合成空间场通过不等于实测车舱阵列通过；同一条 MVDR
              在风场可提升 7 dB，也可能在两麦多声源案例损失近 20 dB。
            </p>
          </div>
          <TableProperties size={22} />
        </div>
        <div className="bf-layer-grid">
          <article>
            <span>01 · IMPLEMENTATION</span>
            <h3>SpatialFrontend</h3>
            <strong>真实执行</strong>
            <p>
              WOLA、协方差、MVDR/MPDR、对角加载、WNG、DAS 回退与风压权重均在当前源码中。
            </p>
          </article>
          <article>
            <span>02 · SIMULATED FIELD</span>
            <h3>72 个空间场</h3>
            <strong>36 / 72 最高</strong>
            <p>
              2/4 麦，三种 SNR，四种空间统计；几何与噪声场是确定性模拟，不是量车录音。
            </p>
          </article>
          <article>
            <span>03 · TARGET</span>
            <h3>双目标语义</h3>
            <strong>显式披露</strong>
            <p>
              BF 用 calibrated clean beam；no-BF 用通道 0。通过数衡量 BF
              对声明目标，不等于原始近端通道最优。
            </p>
          </article>
          <article>
            <span>04 · COMPLIANCE</span>
            <h3>车舱 / ITU</h3>
            <strong>未覆盖</strong>
            <p>
              没有真实阵列标定、车内传播、设备时钟、双讲或正式 ITU
              BF 验收；standard_compliance=false。
            </p>
          </article>
        </div>
      </section>

      <section className="bf-target-card">
        <div>
          <span>TARGET BOUNDARY</span>
          <h2>评测分数不应被包装成同一目标上的排名</h2>
          <p>
            no-BF 的 72/72 是它与自己的控制配对；不是“无 BF 获得 100%
            质量”。BF 的通过集合使用校准干净波束目标。两种分数并列，是为了证明
            每路实现真实改变，并防止跨目标误读。
          </p>
        </div>
        <div
          className="bf-target-scroll"
          role="region"
          aria-label="BF 场景覆盖表，可横向滚动"
          tabIndex={0}
        >
          <table>
            <thead>
              <tr><th>覆盖</th><th>2 麦</th><th>4 麦</th><th>0 dB</th><th>15 dB</th><th>30 dB</th></tr>
            </thead>
            <tbody>
              <tr>
                <th scope="row">场景数</th>
                <td>{bfMatrix.byMics["2"]}</td>
                <td>{bfMatrix.byMics["4"]}</td>
                <td>{bfMatrix.bySnr["0"]}</td>
                <td>{bfMatrix.bySnr["15"]}</td>
                <td>{bfMatrix.bySnr["30"]}</td>
              </tr>
              <tr>
                <th scope="row">空间统计</th>
                <td colSpan={5}>
                  spherical {bfMatrix.fields.spherical} · cylindrical{" "}
                  {bfMatrix.fields.cylindrical} · Corcos{" "}
                  {bfMatrix.fields.corcos} · regular {bfMatrix.fields.regular}
                </td>
              </tr>
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
        graphFocus="bf"
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
          aria-label={`${selectedAlgorithm.title} BF 工作流，可横向滚动`}
          tabIndex={0}
        >
          <FlowDiagram
            flow={selectedAlgorithm.flow}
            selectedId={selectedNode.id}
            onSelect={setSelectedNodeId}
          />
        </div>
        <div className="nsp-legend">
          <span><i className="line audio" />多麦音频</span>
          <span><i className="line control" />空间状态 / 门限</span>
          <span><i className="line optional" />风压 / 候选</span>
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
