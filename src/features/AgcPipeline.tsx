import { useMemo, useState } from "react";
import {
  Beaker,
  ChevronRight,
  CircleDot,
  GitBranch,
  Gauge,
  Layers3,
  ScanSearch,
  TableProperties,
} from "lucide-react";
import { agcAlgorithms } from "./agc-pipeline-data";
import type { AgcAlgorithm } from "./agc-pipeline-data";
import {
  AGC_METRIC_LABELS,
  agcAblationBuilds,
  agcBestCandidate,
  agcGainMethod,
  agcNegativeControls,
  agcFaultReplay,
  agcRegressionCases,
  agcReleaseCandidate,
  agcTechMatrix,
} from "./agc-tech-matrix-data";
import type { AlgorithmCategory } from "./ns-pipeline-data";
import { FlowDiagram } from "./NsPipeline";
import PipelineEvaluation from "./PipelineEvaluation";
import "./ns-pipeline.css";

type Filter = "all" | AlgorithmCategory;

const CATEGORY_META: Record<
  AlgorithmCategory,
  { label: string; note: string }
> = {
  runtime: {
    label: "运行时执行",
    note: "生产路径中的增益策略、执行器或固定校准",
  },
  composite: {
    label: "组合与后级",
    note: "与 AEC/NS 共享状态，或承担压缩、限幅和下行保护",
  },
  experiment: {
    label: "实验候选",
    note: "项目内已有实现和显式开关，但尚未准入",
  },
  research: {
    label: "仅调研",
    note: "当前没有生产运行路径或冻结评测",
  },
};

const FILTERS: Array<{ id: Filter; label: string }> = [
  { id: "all", label: "全部" },
  { id: "runtime", label: "运行时执行" },
  { id: "composite", label: "组合与后级" },
  { id: "experiment", label: "实验候选" },
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
  algorithm: AgcAlgorithm;
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

export default function AgcPipeline() {
  const [filter, setFilter] = useState<Filter>("all");
  const [selectedAlgorithmId, setSelectedAlgorithmId] = useState("guarded5");
  const [selectedNodeId, setSelectedNodeId] = useState("observe");
  const selectedAlgorithm =
    agcAlgorithms.find((item) => item.id === selectedAlgorithmId) ??
    agcAlgorithms[0];
  const filtered = useMemo(
    () =>
      filter === "all"
        ? agcAlgorithms
        : agcAlgorithms.filter((item) => item.category === filter),
    [filter],
  );
  const selectedNode =
    selectedAlgorithm.flow.nodes.find((node) => node.id === selectedNodeId) ??
    selectedAlgorithm.flow.nodes[0];

  const selectAlgorithm = (algorithm: AgcAlgorithm) => {
    setSelectedAlgorithmId(algorithm.id);
    const preferred =
      algorithm.flow.nodes.find((node) =>
        [
          "observe",
          "gain",
          "level",
          "cap",
          "onset",
          "knee",
          "drc",
          "model",
        ].includes(node.id),
      ) ?? algorithm.flow.nodes[0];
    setSelectedNodeId(preferred.id);
  };

  const integrated = agcAlgorithms.filter(
    (item) => item.category === "runtime" || item.category === "composite",
  ).length;
  const experiments = agcAlgorithms.filter(
    (item) => item.category === "experiment",
  ).length;
  const research = agcAlgorithms.filter(
    (item) => item.category === "research",
  ).length;

  return (
    <div className="nsp-page">
      <section className="nsp-intro">
        <div>
          <span>AUTOMATIC GAIN CONTROL</span>
          <h2>从语音证据到增益目标，再看后级怎样保护峰值</h2>
          <p>
            本页把 AGC 本体、AGC2 执行器、ECNR 协调控制、TX/RX
            动态范围保护和未接入的响度路线分开。每个技术只引用自己的冻结结果；代表失败集、完整矩阵和单元行为不会混用。
          </p>
        </div>
        <div className="nsp-summary" aria-label="AGC 技术数量">
          <div>
            <strong>{agcAlgorithms.length}</strong>
            <span>技术总数</span>
          </div>
          <div>
            <strong>{integrated}</strong>
            <span>已接入 / 组合</span>
          </div>
          <div>
            <strong>{experiments}</strong>
            <span>实验候选</span>
          </div>
          <div>
            <strong>{research}</strong>
            <span>仅调研</span>
          </div>
        </div>
      </section>

      <section className="nsp-boundary-row">
        <div className="nsp-bounds">
          <GitBranch size={17} />
          <p>
            <strong>边界：</strong>
            控制器输出的是 dB
            标量，AGC2 负责执行；TX DRC 与最终 limiter 是峰值保护，RX
            DRC 是扬声器播放保护。它们都不能被写成独立的“已通过 AGC”。
          </p>
        </div>
        <div className="nsp-legend-inline" aria-label="连线图例">
          <span>
            <i className="line audio" />
            音频
          </span>
          <span>
            <i className="line control" />
            控制 / 证据
          </span>
          <span>
            <i className="line optional" />
            实验 / 候选
          </span>
        </div>
      </section>

      <section className="nsp-catalog" aria-labelledby="agc-catalog-title">
        <div className="nsp-section-head">
          <div>
            <span>GAIN CONTROL INDEX</span>
            <h2 id="agc-catalog-title">项目内 AGC 策略、执行与保护技术</h2>
            <p>
              默认 VCP 使用 Guarded5
              生成语音增益目标，并由 AGC2 执行。Stage4 与 AGC2 Adaptive
              是实验；协调控制、TX/RX
              DRC 是组合后级；Speex AGC、AGC1、R128、look-ahead 和神经 AGC
              当前没有生产路径。
            </p>
          </div>
          <div className="nsp-filters" role="group" aria-label="筛选 AGC 类别">
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
                    ? agcAlgorithms.length
                    : agcAlgorithms.filter(
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

      <section
        className="agc-matrix"
        aria-labelledby="agc-matrix-title"
      >
        <div className="agc-matrix-head">
          <div>
            <span>CONTROLLED AGC BENCHMARK</span>
            <h2 id="agc-matrix-title">
              同一冻结集上的 {agcTechMatrix.caseCount} 场景技术对照
            </h2>
            <p>
              同一输入 PCM 与冻结门限，只改变 AGC
              配置；Stage4 组合由新增 release 开关后的第二个冻结构建执行。
              16 / 24 / 48 kHz 各 36 个 AGC
              场景，近端电平各 36 个 −45 / −30 / −15 dBFS。通过数为完整工程门限，
              “修复 / 回归”相对冻结原基线。
            </p>
          </div>
          <div className="agc-matrix-stamp">
            <TableProperties size={17} />
            <div>
              <span>二进制 SHA256</span>
              <code>{agcTechMatrix.cliSha256}</code>
            </div>
          </div>
        </div>
        <div
          className="agc-matrix-scroll"
          role="region"
          aria-label="AGC 技术统一评测表，可横向滚动"
          tabIndex={0}
        >
          <table className="agc-matrix-table">
            <thead>
              <tr>
                <th>技术 / 配置</th>
                <th>场景通过</th>
                <th>修复</th>
                <th>回归</th>
                <th>16k</th>
                <th>24k</th>
                <th>48k</th>
                <th>−45 dBFS</th>
                <th>−30 dBFS</th>
                <th>−15 dBFS</th>
                <th>实际 PCM 增益中位数</th>
                <th>主要失败门限</th>
              </tr>
            </thead>
            <tbody>
              {agcTechMatrix.variants.map((variant) => (
                <tr key={variant.id}>
                  <th scope="row">
                    <strong>{variant.label}</strong>
                    <code>{variant.configuration}</code>
                  </th>
                  <td className="agc-matrix-pass">
                    <strong>
                      {variant.pass} / {agcTechMatrix.caseCount}
                    </strong>
                    <span>{Math.round((variant.pass / agcTechMatrix.caseCount) * 100)}%</span>
                  </td>
                  <td className="agc-matrix-fix">+{variant.fixes}</td>
                  <td
                    className={
                      variant.regressions > 0
                        ? "agc-matrix-regression"
                        : "agc-matrix-zero"
                    }
                  >
                    −{variant.regressions}
                  </td>
                  {(["16000", "24000", "48000"] as const).map((rate) => (
                    <td key={rate}>
                      {variant.byRate[rate].pass}/
                      {variant.byRate[rate].total}
                    </td>
                  ))}
                  {(["-45", "-30", "-15"] as const).map((level) => (
                    <td key={level}>
                      {variant.byLevel[level].pass}/
                      {variant.byLevel[level].total}
                    </td>
                  ))}
                  <td>
                    {variant.pcmGainDb.median.toFixed(2)} dB
                    <small>
                      {variant.pcmGainDb.nonzero} / {agcTechMatrix.caseCount}
                      个案例有增益
                    </small>
                  </td>
                  <td>
                    <div className="agc-matrix-failures">
                      {variant.failureTop.slice(0, 3).map((failure) => (
                        <span key={failure.metric}>
                          {AGC_METRIC_LABELS[failure.metric] ??
                            failure.metric}
                          <b>{failure.count}</b>
                        </span>
                      ))}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="agc-matrix-notes">
          <p>
            新执行器故障代码在默认故障关闭时重跑了同一 108 场景：
            {agcFaultReplay.outputShaMatch} / {agcFaultReplay.caseCount}
            个输出 SHA 与候选逐例一致，通过仍为 {agcFaultReplay.pass}，
            回归仍为 {agcFaultReplay.regressions}。CLI 哈希{" "}
            <code>{agcFaultReplay.cliSha256.slice(0, 12)}…</code>，结果哈希{" "}
            <code>{agcFaultReplay.resultsSha256.slice(0, 12)}…</code>。
          </p>
          <p>
            <strong>结论边界：</strong>
            组合候选 Stage4 + noise release + No TX DRC 达到 71/108，
            相对原基线修复 26、回归 7；回归使准入状态保持
            <code>blocked_regressions</code>。它只比单开 release
            多 5 个通过，新增收益集中在 16 kHz、−15 dBFS、单麦。
          </p>
          <p>
            通过数为“全部工程 gate 通过”，不是单项投影增益。原基线的
            −45 dBFS 场景 0/36 通过；把某一档固定增益达到 6/8 dB
            只说明增益被执行，不能自动通过语音保真和噪声衰减门限。
          </p>
          <p>
            证据目录：<code>{agcTechMatrix.source}</code>
            。每个变体保留
            <code>selection.json</code>、<code>results.jsonl</code>、
            <code>report.json</code>、输入哈希、CLI 哈希和阶段 tap。
          </p>
          <p>
            PCM 增益使用独立脚本从
            <code>nr-protected-agc-before</code> 到 <code>agc-after</code>
            的整段 tap 复算；每案例一次能量比，绝对增益 &gt;{" "}
            {agcGainMethod.nonzeroDb} dB 才计为非零，不做最佳延迟拟合或响度归一化。
            脚本 SHA256：<code>{agcGainMethod.scriptSha256.slice(0, 12)}…</code>
          </p>
        </div>
        <section
          className="agc-negative-controls"
          aria-labelledby="agc-negative-controls-title"
        >
          <div className="agc-negative-controls-head">
            <span>NEGATIVE CONTROLS</span>
            <h3 id="agc-negative-controls-title">关闭与错误增益路径</h3>
            <p>
              这些结果用于检查门限能否拒绝错误 AGC。关闭 AGC 本身只覆盖零增益，
              不能替代错误时序、错误噪声提升和误触发测试。
            </p>
          </div>
          <div className="agc-negative-controls-grid">
            {agcNegativeControls.map((control) => (
              <article key={control.name}>
                <div>
                  <span
                    className={`agc-negative-control-status agc-negative-control-status--${control.status}`}
                  >
                    {control.status === "verified"
                      ? "已让错误实现失败"
                      : "观测对照"}
                  </span>
                  <h4>{control.name}</h4>
                  <code>{control.configuration}</code>
                </div>
                <dl>
                  <div>
                    <dt>预期失败</dt>
                    <dd>{control.expected}</dd>
                  </div>
                  <div>
                    <dt>实测</dt>
                    <dd>{control.evidence}</dd>
                  </div>
                  <div>
                    <dt>边界</dt>
                    <dd>{control.limitation}</dd>
                  </div>
                </dl>
              </article>
            ))}
          </div>
        </section>
        <div className="agc-matrix-followup">
          <div>
            <span>BEST EXPERIMENTAL COMBINATION · ADMISSION BLOCKED</span>
            <h3>Stage4 + 噪声域释放 + No TX DRC</h3>
            <p>
              在 <code>{agcReleaseCandidate.configuration}</code>
              上关闭上行 DRC，检验近端瞬态失真与噪声段释放能否同时改善。
              这不是默认配置。
            </p>
          </div>
          <dl>
            <div>
              <dt>场景通过</dt>
              <dd>
                {agcBestCandidate.pass} / {agcTechMatrix.caseCount}
              </dd>
            </div>
            <div>
              <dt>相对 Guarded5</dt>
              <dd>
                +{agcBestCandidate.fixes} / −
                {agcBestCandidate.regressions}
              </dd>
            </div>
            <div>
              <dt>相对 release</dt>
              <dd>+5 / −0</dd>
            </div>
            <div>
              <dt>−45 dBFS</dt>
              <dd>
                {agcBestCandidate.byLevel["-45"].pass} /
                {agcBestCandidate.byLevel["-45"].total}
              </dd>
            </div>
          </dl>
          <p>
            7 个回归里，5 个是 −45 dBFS 的噪声段衰减跌到
            4.38–5.87 dB，2 个是 48 kHz、−30 dBFS 的
            <code>tcar</code> 近端 SI-SDR/STOI 退化。No TX DRC 的
            4 个单项修复均为 16 kHz、−15 dBFS、单麦，因此不能把总收益外推到所有采样率。
          </p>
          <code>
            {agcBestCandidate.source} · CLI{" "}
            {agcBestCandidate.cliSha256.slice(0, 12)}…
          </code>
          <code>
            replay {agcBestCandidate.replaySha256.slice(0, 12)}… · gain report{" "}
            {agcBestCandidate.gainReportSha256.slice(0, 12)}… · results{" "}
            {agcBestCandidate.resultsSha256.slice(0, 12)}…
          </code>
        </div>
        <div
          className="agc-ablation"
          aria-labelledby="agc-ablation-title"
        >
          <div className="agc-ablation-head">
            <div>
              <span>PAIRED ABLATION · 2 × 2</span>
              <h3 id="agc-ablation-title">
                release 与 TX DRC 各自贡献多少
              </h3>
              <p>
                四格使用同一 108 场景、冻结输入和完整工程门限；Stage4
                基础格与 release 格分别来自两个已记录哈希的冻结构建，只切换
                <code>--stage4-noise-release</code> 与
                <code>--no-tx-drc</code>。四格都仍有回归，因此没有一格可准入。
              </p>
            </div>
            <code>
              build/agc-ablation-stage4-release-drc-20261020/summary.json
            </code>
          </div>
          <div
            className="agc-ablation-scroll"
            role="region"
            aria-label="AGC 2×2 配对消融，可横向滚动"
            tabIndex={0}
          >
            <table className="agc-ablation-table">
              <thead>
                <tr>
                  <th>Stage4 噪声域释放</th>
                  <th>TX DRC</th>
                  <th>完整场景通过</th>
                  <th>修复 / 回归</th>
                  <th>准入状态</th>
                  <th>解释</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>关闭</td>
                  <td>开启</td>
                  <td>
                    <strong>65 / 108</strong>
                  </td>
                  <td>+21 / −8</td>
                  <td>回归阻断</td>
                  <td>Stage4 基线</td>
                </tr>
                <tr>
                  <td>关闭</td>
                  <td>关闭</td>
                  <td>
                    <strong>70 / 108</strong>
                  </td>
                  <td>+26 / −8</td>
                  <td>回归阻断</td>
                  <td>No TX DRC 多修复 5 个</td>
                </tr>
                <tr>
                  <td>开启</td>
                  <td>开启</td>
                  <td>
                    <strong>66 / 108</strong>
                  </td>
                  <td>+21 / −7</td>
                  <td>回归阻断</td>
                  <td>噪声释放多通过 1 个、少 1 个回归</td>
                </tr>
                <tr className="is-best">
                  <td>开启</td>
                  <td>关闭</td>
                  <td>
                    <strong>71 / 108</strong>
                  </td>
                  <td>+26 / −7</td>
                  <td>回归阻断</td>
                  <td>当前最佳实验组合</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div className="agc-ablation-notes">
            <p>
              两个冻结构建哈希：
              {agcAblationBuilds.map((build) => (
                <span key={build.cliSha256}>
                  <code>{build.cliSha256.slice(0, 12)}…</code>
                  {build.branches}
                </span>
              ))}
              因此“只切换配置”适用于每一行内部的开关键，跨 release
              分支的变化还包含构建版本差异。
            </p>
            <p>
              噪声域释放改变 39/108 个输出 SHA；关闭 TX DRC 在两个 release
              分支都改变 52/108 个输出 SHA。这证明两个开关都实际执行，不能把变化归因给日志或配置未生效。
            </p>
            <p>
              改善 5 个不等于解决 5 个簇：release 主要修复噪声期增益，TX DRC
              主要影响高电平近端；剩余 7 个回归仍需按噪声衰减与 48 kHz STOI
              分开处理。
            </p>
          </div>
          <div
            className="agc-ablation-scroll"
            role="region"
            aria-label="AGC 剩余七个回归案例，可横向滚动"
            tabIndex={0}
          >
            <table className="agc-ablation-table">
              <caption>
                唯一键：原始运行 + 场景 ID + 噪声；不使用单独 scene ID
              </caption>
              <thead>
                <tr>
                  <th>原始运行 / 场景</th>
                  <th>噪声</th>
                  <th>采样率</th>
                  <th>近端电平</th>
                  <th>失败门限</th>
                </tr>
              </thead>
              <tbody>
                {agcRegressionCases.map((item) => (
                  <tr key={`${item.run}/${item.id}/${item.noise}`}>
                    <td>
                      <strong>{item.id}</strong>
                      <small>{item.run}</small>
                    </td>
                    <td>{item.noise}</td>
                    <td>{item.rate}</td>
                    <td>{item.near}</td>
                    <td>{item.failed}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
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
        graphFocus="agc"
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
          aria-label={`${selectedAlgorithm.title} AGC 工作流，可横向滚动`}
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
            控制 / 证据
          </span>
          <span>
            <i className="line optional" />
            实验 / 候选
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
