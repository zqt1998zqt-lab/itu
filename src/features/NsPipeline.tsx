import { useId, useMemo, useState } from "react";
import {
  Beaker,
  Boxes,
  ChevronRight,
  CircleDot,
  GitBranch,
  Layers3,
  SearchCode,
} from "lucide-react";
import { algorithms, categoryMeta } from "./ns-pipeline-data";
import type {
  Algorithm,
  AlgorithmCategory,
  AlgorithmFlow,
  FlowNode,
} from "./ns-pipeline-data";
import { extraAlgorithms } from "./ns-pipeline-extra-data";
import { nsEvaluations } from "./ns-evaluation-data";
import PipelineEvaluation from "./PipelineEvaluation";
import type { PipelineEvaluationData } from "./PipelineEvaluation";
import "./ns-pipeline.css";

type Filter = "all" | AlgorithmCategory;
type NoiseAlgorithm = Algorithm & { evaluation: PipelineEvaluationData };

function withEvaluation(algorithm: Algorithm): NoiseAlgorithm {
  const evaluation = nsEvaluations[algorithm.id];
  if (!evaluation) {
    throw new Error(`Missing NS evaluation for ${algorithm.id}`);
  }
  return { ...algorithm, evaluation };
}

const ALL_ALGORITHMS = [...algorithms, ...extraAlgorithms].map(withEvaluation);
const FILTERS: Array<{ id: Filter; label: string }> = [
  { id: "all", label: "全部" },
  { id: "runtime", label: "运行时后端" },
  { id: "composite", label: "组合 / 辅助" },
  { id: "experiment", label: "离线实验" },
  { id: "research", label: "仅调研" },
];
const NODE_W = 172;
const NODE_H = 92;
const COL_W = 210;
const ROW_GAP = 184;
const LANE_H = 144;
const HEADER_H = 58;
const PAD_X = 28;

const categoryIcon = (category: AlgorithmCategory) => {
  if (category === "runtime") return Boxes;
  if (category === "composite") return Layers3;
  if (category === "experiment") return Beaker;
  return SearchCode;
};

function nodePosition(node: FlowNode) {
  return {
    x: PAD_X + node.col * COL_W,
    y: HEADER_H + 24 + node.row * ROW_GAP,
    w: NODE_W,
    h: NODE_H,
  };
}

function connectorPath(from: FlowNode, to: FlowNode) {
  const a = nodePosition(from);
  const b = nodePosition(to);
  const x1 = a.x + a.w;
  const y1 = a.y + a.h / 2;
  const y2 = b.y + b.h / 2;

  if (from.row === to.row) {
    if (b.x >= a.x) return `M${x1} ${y1}H${b.x}`;
    const tx = b.x + b.w;
    const laneY = a.y + a.h + 22;
    return `M${x1} ${y1}H${x1 + 12}Q${x1 + 18} ${y1} ${x1 + 18} ${y1 + 6}V${laneY - 6}Q${x1 + 18} ${laneY} ${x1 + 12} ${laneY}H${tx + 18}Q${tx + 12} ${laneY} ${tx + 12} ${laneY - 6}V${y2 + 6}Q${tx + 12} ${y2} ${tx + 6} ${y2}H${tx}`;
  }

  const down = from.row < to.row;
  const sourceX = x1;
  const targetX = down ? b.x + b.w / 2 : b.x + b.w;
  const midY = down ? a.y + a.h + 30 : b.y + b.h + 30;
  const verticalX = Math.max(a.x + a.w + 20, Math.min(b.x - 20, x1 + 34));
  return down
    ? `M${sourceX} ${y1}H${verticalX - 8}Q${verticalX} ${y1} ${verticalX} ${y1 + 8}V${midY - 8}Q${verticalX} ${midY} ${verticalX + 8} ${midY}H${targetX - 8}Q${targetX} ${midY} ${targetX} ${midY - 8}V${y2 - 8}Q${targetX} ${y2} ${targetX + 8} ${y2}H${b.x}`
    : `M${x1} ${y1}H${verticalX + 8}Q${verticalX} ${y1} ${verticalX} ${y1 + 8}V${midY - 8}Q${verticalX} ${midY} ${verticalX - 8} ${midY}H${targetX + 8}Q${targetX} ${midY} ${targetX} ${midY + 8}V${y2 - 8}Q${targetX} ${y2} ${targetX - 8} ${y2}H${b.x + b.w}`;
}

export function FlowDiagram({
  flow,
  selectedId,
  onSelect,
}: {
  flow: AlgorithmFlow;
  selectedId: string;
  onSelect: (id: string) => void;
}) {
  const uid = useId().replace(/:/g, "");
  const cols = Math.max(...flow.nodes.map((node) => node.col)) + 1;
  const rows = Math.max(...flow.nodes.map((node) => node.row)) + 1;
  const width = PAD_X * 2 + Math.max(3, cols) * COL_W - (COL_W - NODE_W);
  const height = HEADER_H + rows * ROW_GAP + 24;
  const byId = new Map(flow.nodes.map((node) => [node.id, node]));
  const audio = `nsp-${uid}-audio`;
  const control = `nsp-${uid}-control`;
  const optional = `nsp-${uid}-optional`;
  const titleId = `nsp-${uid}-title`;
  const descId = `nsp-${uid}-desc`;

  return (
    <div className="nsp-board" style={{ width }}>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-labelledby={`${titleId} ${descId}`}
      >
        <title id={titleId}>{flow.title} 工作流</title>
        <desc id={descId}>{flow.description}</desc>
        <defs>
          <marker
            id={audio}
            markerWidth="8"
            markerHeight="6"
            refX="7"
            refY="3"
            orient="auto"
          >
            <polygon points="0 0,8 3,0 6" fill="#65758b" />
          </marker>
          <marker
            id={control}
            markerWidth="8"
            markerHeight="6"
            refX="7"
            refY="3"
            orient="auto"
          >
            <polygon points="0 0,8 3,0 6" fill="#2563eb" />
          </marker>
          <marker
            id={optional}
            markerWidth="8"
            markerHeight="6"
            refX="7"
            refY="3"
            orient="auto"
          >
            <polygon points="0 0,8 3,0 6" fill="#9b741f" />
          </marker>
        </defs>
        <rect className="nsp-canvas" width={width} height={height} />
        {[0, 1].map((row) => (
          <rect
            key={row}
            className="nsp-lane"
            x="16"
            y={HEADER_H + row * ROW_GAP - 16}
            width={width - 32}
            height={LANE_H}
            rx="8"
          />
        ))}
        <text className="nsp-lane-title" x="32" y={HEADER_H + 8}>
          {flow.lanes[0]}
        </text>
        <text
          className="nsp-lane-title"
          x="32"
          y={HEADER_H + ROW_GAP + 8}
        >
          {flow.lanes[1]}
        </text>
        {flow.edges.map(([fromId, toId, kind]) => {
          const from = byId.get(fromId);
          const to = byId.get(toId);
          if (!from || !to) return null;
          return (
            <path
              key={`${fromId}-${toId}-${kind}`}
              className={`nsp-arrow nsp-arrow--${kind}`}
              d={connectorPath(from, to)}
              markerEnd={`url(#${
                kind === "audio"
                  ? audio
                  : kind === "control"
                    ? control
                    : optional
              })`}
            />
          );
        })}
        {flow.nodes.map((node) => {
          const position = nodePosition(node);
          const active = node.id === selectedId;
          return (
            <g
              key={node.id}
              className={`nsp-node nsp-node--${node.kind}${
                active ? " is-active" : ""
              }`}
              role="button"
              tabIndex={0}
              aria-label={`查看 ${node.title} 的实现详情`}
              aria-pressed={active}
              onClick={() => onSelect(node.id)}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  onSelect(node.id);
                }
              }}
            >
              <rect
                x={position.x}
                y={position.y}
                width={position.w}
                height={position.h}
                rx="7"
              />
              <rect
                className="nsp-tag-bg"
                x={position.x + 12}
                y={position.y + 12}
                width={Math.min(
                  position.w - 24,
                  node.tag.length * 6 + 20,
                )}
                height="18"
                rx="2"
              />
              <text
                className="nsp-tag"
                x={position.x + 20}
                y={position.y + 25}
              >
                {node.tag}
              </text>
              <text
                className="nsp-title"
                x={position.x + 12}
                y={position.y + 55}
              >
                {node.title}
              </text>
              <text
                className="nsp-sub"
                x={position.x + 12}
                y={position.y + 75}
              >
                {node.sub}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function AlgorithmCard({
  algorithm,
  active,
  onSelect,
}: {
  algorithm: NoiseAlgorithm;
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
          {categoryMeta[algorithm.category].label}
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

export default function NsPipeline() {
  const [filter, setFilter] = useState<Filter>("all");
  const [selectedAlgorithmId, setSelectedAlgorithmId] = useState("webrtc");
  const [selectedNodeId, setSelectedNodeId] = useState("analyze");
  const selectedAlgorithm =
    ALL_ALGORITHMS.find((item) => item.id === selectedAlgorithmId) ??
    ALL_ALGORITHMS[0];
  const filtered = useMemo(
    () =>
      filter === "all"
        ? ALL_ALGORITHMS
        : ALL_ALGORITHMS.filter((item) => item.category === filter),
    [filter],
  );
  const selectedNode =
    selectedAlgorithm.flow.nodes.find((node) => node.id === selectedNodeId) ??
    selectedAlgorithm.flow.nodes[0];

  const selectAlgorithm = (algorithm: NoiseAlgorithm) => {
    setSelectedAlgorithmId(algorithm.id);
    const preferred =
      algorithm.flow.nodes.find((node) =>
        [
          "gain",
          "gru",
          "network",
          "df",
          "tiny",
          "h1",
          "fuse",
          "suppress",
          "minimum",
        ].includes(node.id),
      ) ?? algorithm.flow.nodes[0];
    setSelectedNodeId(preferred.id);
  };

  return (
    <div className="nsp-page">
      <section className="nsp-intro">
        <div>
          <span>NOISE-SUPPRESSION CATALOG</span>
          <h2>先看有哪些算法，再拆一帧怎样流动</h2>
          <p>
            本页只覆盖降噪。运行时后端、组合路径、离线实验和调研候选分别标注；每个算法都有独立的处理工作流，不再用同一个“降噪核心”概括。
          </p>
        </div>
        <div className="nsp-summary" aria-label="算法数量">
          <div>
            <strong>{algorithms.length}</strong>
            <span>运行后端</span>
          </div>
          <div>
            <strong>
              {
                extraAlgorithms.filter((item) => item.category === "composite")
                  .length
              }
            </strong>
            <span>组合 / 辅助</span>
          </div>
          <div>
            <strong>
              {
                extraAlgorithms.filter((item) => item.category === "experiment")
                  .length
              }
            </strong>
            <span>离线实验</span>
          </div>
          <div>
            <strong>
              {
                extraAlgorithms.filter((item) => item.category === "research")
                  .length
              }
            </strong>
            <span>仅调研</span>
          </div>
        </div>
      </section>

      <section className="nsp-boundary-row">
        <div className="nsp-bounds">
          <GitBranch size={17} />
          <p>
            <strong>边界：</strong>
            输入输出均为 PCM；AEC、BF、EQ、AGC 和指标计算不在本页。蓝线为控制或估计，灰线为音频，金色虚线为可选或候选路径。
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

      <section className="nsp-catalog" aria-labelledby="nsp-catalog-title">
        <div className="nsp-section-head">
          <div>
            <span>ALGORITHM INDEX</span>
            <h2 id="nsp-catalog-title">项目内降噪算法与模块</h2>
            <p>
              “运行时后端”可以在统一入口选择；“组合与辅助”需要额外配置；“离线实验”和“仅调研”不会伪装成生产后端。每项同时标明冻结评测结果或待测状态。
            </p>
          </div>
          <div className="nsp-filters" role="group" aria-label="筛选算法类别">
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
                    ? ALL_ALGORITHMS.length
                    : ALL_ALGORITHMS.filter(
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
            {categoryMeta[selectedAlgorithm.category].label}
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
        graphFocus="ns"
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
          aria-label={`${selectedAlgorithm.title} 降噪工作流，可横向滚动`}
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
