import { useMemo, useState } from "react";
import {
  Beaker,
  ChevronRight,
  CircleDot,
  GitBranch,
  Layers3,
  RadioTower,
  ScanSearch,
} from "lucide-react";
import { aecAlgorithms } from "./aec-pipeline-data";
import type { AecAlgorithm } from "./aec-pipeline-data";
import type {
  AlgorithmCategory,
} from "./ns-pipeline-data";
import { FlowDiagram } from "./NsPipeline";
import PipelineEvaluation from "./PipelineEvaluation";
import "./ns-pipeline.css";

type Filter = "all" | AlgorithmCategory;

const CATEGORY_META: Record<
  AlgorithmCategory,
  { label: string; note: string }
> = {
  runtime: {
    label: "运行时后端",
    note: "由 --aec-backend 或当前处理 profile 直接选择",
  },
  composite: {
    label: "组合链路",
    note: "AEC 与前置波束、后级共享状态一起工作",
  },
  experiment: {
    label: "诊断增强",
    note: "项目内已有实现，但默认关闭并要求显式开关",
  },
  research: {
    label: "仅调研",
    note: "当前没有生产运行路径",
  },
};

const FILTERS: Array<{ id: Filter; label: string }> = [
  { id: "all", label: "全部" },
  { id: "runtime", label: "运行时后端" },
  { id: "composite", label: "组合链路" },
  { id: "experiment", label: "诊断增强" },
];

function categoryIcon(category: AlgorithmCategory) {
  if (category === "runtime") return RadioTower;
  if (category === "composite") return Layers3;
  if (category === "experiment") return Beaker;
  return ScanSearch;
}

function AlgorithmCard({
  algorithm,
  active,
  onSelect,
}: {
  algorithm: AecAlgorithm;
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

export default function AecPipeline() {
  const [filter, setFilter] = useState<Filter>("all");
  const [selectedAlgorithmId, setSelectedAlgorithmId] = useState("speex-coherence-residual");
  const [selectedNodeId, setSelectedNodeId] = useState("features");
  const selectedAlgorithm =
    aecAlgorithms.find((item) => item.id === selectedAlgorithmId) ??
    aecAlgorithms[0];
  const filtered = useMemo(
    () =>
      filter === "all"
        ? aecAlgorithms
        : aecAlgorithms.filter((item) => item.category === filter),
    [filter],
  );
  const selectedNode =
    selectedAlgorithm.flow.nodes.find((node) => node.id === selectedNodeId) ??
    selectedAlgorithm.flow.nodes[0];

  const selectAlgorithm = (algorithm: AecAlgorithm) => {
    setSelectedAlgorithmId(algorithm.id);
    const preferred =
      algorithm.flow.nodes.find((node) =>
        ["linear", "mdf", "beam", "subspace"].includes(node.id),
      ) ?? algorithm.flow.nodes[0];
    setSelectedNodeId(preferred.id);
  };

  return (
    <div className="nsp-page">
      <section className="nsp-intro">
        <div>
          <span>ACOUSTIC ECHO CANCELLATION</span>
          <h2>从播放参考出发，看回声怎样被估计、消除和保护</h2>
          <p>
            本页只覆盖 AEC。网页生成的 VCP 档位默认使用 Speex MDF；底层 Config 默认使用 AEC3。两种后端、前置波束和多参考保护分别展开，保留延迟估计、线性滤波、非线性残差与近端保护的测量边界。
          </p>
        </div>
        <div className="nsp-summary" aria-label="AEC 链路数量">
          <div>
            <strong>{aecAlgorithms.length}</strong>
            <span>链路总数</span>
          </div>
          <div>
            <strong>
              {
                aecAlgorithms.filter((item) => item.category === "runtime")
                  .length
              }
            </strong>
            <span>运行时后端</span>
          </div>
          <div>
            <strong>
              {
                aecAlgorithms.filter((item) => item.category === "composite")
                  .length
              }
            </strong>
            <span>组合链路</span>
          </div>
          <div>
            <strong>
              {
                aecAlgorithms.filter((item) => item.category === "experiment")
                  .length
              }
            </strong>
            <span>诊断增强</span>
          </div>
        </div>
      </section>

      <section className="nsp-boundary-row">
        <div className="nsp-bounds">
          <GitBranch size={17} />
          <p>
            <strong>边界：</strong>
            AEC 接收麦克风与同步播放参考，输出消除后的 uplink PCM；NR、EQ、AGC 和 ITU-T 指标计算在本页只作为下游出现。灰线为音频，蓝线为延迟或近端判断，金色虚线为显式开关的增强路径。
          </p>
        </div>
        <div className="nsp-legend-inline" aria-label="连线图例">
          <span>
            <i className="line audio" />
            音频
          </span>
          <span>
            <i className="line control" />
            控制 / 估计
          </span>
          <span>
            <i className="line optional" />
            可选
          </span>
        </div>
      </section>

      <section className="nsp-catalog" aria-labelledby="aec-catalog-title">
        <div className="nsp-section-head">
          <div>
            <span>ECHO PATH INDEX</span>
            <h2 id="aec-catalog-title">项目内 AEC 后端与信号路径</h2>
            <p>
              网页默认 VCP 路径为 Speex MDF + Coherence RES：500 ms 尾长、原生输入调理、单参考相干残差门控与共享后级上下文。底层 Config 保留 WebRTC AEC3 默认值；多参考输入自动回退为未门控 MDF。冻结对照结果不代表当前 VCP 整链已通过验收。
            </p>
          </div>
          <div className="nsp-filters" role="group" aria-label="筛选 AEC 类别">
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
                    ? aecAlgorithms.length
                    : aecAlgorithms.filter(
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
          <div>
            <dt>采样率</dt>
            <dd>{selectedAlgorithm.rate}</dd>
          </div>
          <div>
            <dt>入口</dt>
            <dd>
              <code>{selectedAlgorithm.entry}</code>
            </dd>
          </div>
          <div>
            <dt>配置</dt>
            <dd>
              <code>{selectedAlgorithm.cli}</code>
            </dd>
          </div>
        </dl>
        <div className="nsp-selected-boundary">
          <strong>工程边界</strong>
          <span>{selectedAlgorithm.boundary}</span>
        </div>
      </section>

      <PipelineEvaluation
        title={selectedAlgorithm.title}
        evaluation={selectedAlgorithm.evaluation}
        graphFocus="aec"
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
                <option key={node.id} value={node.id}>
                  {node.title}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div
          className="nsp-scroll"
          role="region"
          aria-label={`${selectedAlgorithm.title} AEC 工作流，可横向滚动`}
          tabIndex={0}
        >
          <FlowDiagram
            flow={selectedAlgorithm.flow}
            selectedId={selectedNode.id}
            onSelect={setSelectedNodeId}
          />
        </div>
        <div className="nsp-legend">
          <span>
            <i className="line audio" />
            音频
          </span>
          <span>
            <i className="line control" />
            控制 / 估计
          </span>
          <span>
            <i className="line optional" />
            可选 / 候选
          </span>
          <span>
            <i className="box" />
            当前节点
          </span>
        </div>
      </figure>

      <section
        className="nsp-detail"
        aria-live="polite"
        aria-atomic="true"
      >
        <div className="nsp-detail-copy">
          <span>{selectedNode.tag}</span>
          <h2>{selectedNode.title}</h2>
          <p>{selectedNode.body}</p>
          <code>{selectedNode.formula}</code>
        </div>
        <dl>
          <div>
            <dt>实现</dt>
            <dd>{selectedNode.where}</dd>
          </div>
          <div>
            <dt>条件</dt>
            <dd>{selectedNode.condition}</dd>
          </div>
        </dl>
        <div className="nsp-fields">
          <span>实时字段</span>
          {selectedNode.fields.map((field) => (
            <code key={field}>{field}</code>
          ))}
        </div>
      </section>
    </div>
  );
}
