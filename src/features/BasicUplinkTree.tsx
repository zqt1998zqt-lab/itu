import { useId, useState } from "react";
import { ArrowRight, GitBranch, Layers3 } from "lucide-react";
import { chainPresets, chainStages } from "./enhancement-chain-model";
import type { FiveSignalMode } from "./five-signal-flow-model";
import type {
  BasicBatchSnapshot,
  CatalogResponse,
  DatasetSummary,
} from "./BasicUplinkTest";
import "./basic-uplink-tree.css";

type Props = {
  mode: FiveSignalMode;
  algorithmClusterId: string;
  datasetId: string;
  dataset: DatasetSummary | null;
  catalog: CatalogResponse | null;
  catalogLoading: boolean;
  catalogError: string;
  snapshot: BasicBatchSnapshot | null;
  syncFailed: boolean;
  isPreviousSnapshot: boolean;
};

const WIDTH = 1536;
const HEIGHT = 968;
const LEAF_X = 848;
const LEAF_WIDTH = 664;
const LEAF_HEIGHT = 96;
const LEAF_TOP = 88;
const LEAF_STEP = 108;
const CENTER_Y = 514;
const STATUS_LABELS = {
  pending: "等待计算",
  running: "计算中",
  pass: "工程通过",
  fail: "工程失败",
  review_required: "需复核",
  not_measured: "不可测",
  invalid: "无效",
};

const METHODS: Record<string, { title: string; detail: string; metric: string }> = {
  slr: {
    title: "发送响度",
    detail: "干净近端参考（reference）与发送处理输出（TX）对齐，计算语音段发送增益",
    metric: "发送增益代理 · dB",
  },
  frequency_response: {
    title: "发送频响",
    detail: "已知激励与发送处理输出（TX）的 H1 传函 → 三分之一倍频程跨度",
    metric: "H1 频响跨度 · dB",
  },
  listening_quality: {
    title: "单向语音质量",
    detail: "发送处理输出（TX）语音 → DNSMOS P.835 → OVRL 预测",
    metric: "DNSMOS OVRL 工程代理",
  },
  idle_noise_level: {
    title: "空闲噪声电平",
    detail: "选定发送处理输出（TX）时间窗 → 均方能量 → 数字 RMS 电平",
    metric: "RMS 电平 · dBFS",
  },
  idle_noise_peaks: {
    title: "空闲噪声峰值",
    detail: "选定发送处理输出（TX）时间窗 → 时域峰值与 RMS 之差",
    metric: "时域峰值突出度 · dB",
  },
  out_of_band: {
    title: "带外信号",
    detail: "选定发送处理输出（TX）片段 → Hann 窗 FFT → 带外 / 总能量",
    metric: "带外能量比 · dB",
  },
  hats_snr: {
    title: "HATS 信噪比",
    detail: "干净近端参考（reference）、独立噪声输入（noise）帧与发送处理输出（TX）→ 输出语音 / 噪声功率比",
    metric: "输出 SNR 代理 · dB",
  },
};

function metricUnit(metric: string | null | undefined): string {
  if (!metric) return "";
  if (/(?:^|_)dbfs(?:_|$)/.test(metric)) return "dBFS";
  if (/(?:^|_)db(?:_|$)/.test(metric)) return "dB";
  if (metric.startsWith("dnsmos_")) return "预测分";
  return "";
}

export default function BasicUplinkTree({
  mode, algorithmClusterId, datasetId, dataset, catalog, catalogLoading,
  catalogError, snapshot, syncFailed, isPreviousSnapshot,
}: Props) {
  const headingId = useId();
  const descriptionId = useId();
  const [hovered, setHovered] = useState<string | null>(null);
  const [locked, setLocked] = useState<string | null>(null);
  const items = catalog?.mode === mode ? catalog.items : [];
  const preset = chainPresets.find((item) => item.id === algorithmClusterId);
  const vcpSupported = Boolean(preset?.profile.startsWith("vcp"));
  const matchingReport = vcpSupported && snapshot?.selected_mode === mode
    && snapshot.algorithm_cluster_id === algorithmClusterId
    && snapshot.dataset_id === datasetId ? snapshot : null;
  const results = new Map(matchingReport?.items.map((item) => [item.item_id, item]));
  const activeId = items.some((item) => item.id === locked) ? locked
    : items.some((item) => item.id === hovered) ? hovered : null;
  const reportStatus = !vcpSupported ? "当前算法不可执行"
    : !matchingReport ? "当前配置待执行"
    : matchingReport.status === "completed" ? "计算完成"
      : matchingReport.status === "failed" ? "执行失败"
        : matchingReport.status === "cancelled" ? "已取消"
          : matchingReport.status === "queued" ? "等待执行" : "正在计算";

  return (
    <figure className="basic-uplink-tree" aria-labelledby={headingId} aria-describedby={descriptionId}>
      <figcaption className="basic-uplink-tree-header">
        <div>
          <h3 id={headingId}><GitBranch size={17} aria-hidden="true" />基础上行测试树</h3>
          <p id={descriptionId}>数据集进入 VCP 上行链路，发送处理输出（TX）与所需评估输入按工况送入 8 项计算模型。悬停查看分支，点击锁定或取消。</p>
        </div>
        <div className="basic-uplink-tree-meta">
          <code>{mode}</code>
          <span>{isPreviousSnapshot && matchingReport ? "上次快照 · " : ""}{reportStatus}{matchingReport ? ` · ${matchingReport.summary.computed}/8 项有数值` : ""}</span>
        </div>
      </figcaption>

      {!vcpSupported ? (
        <p className="basic-uplink-tree-notice" role="status">当前算法簇不支持基础上行测试，请在下方选择 VCP 算法簇。</p>
      ) : snapshot && !matchingReport ? (
        <p className="basic-uplink-tree-notice" role="status">当前模式、算法或数据集已变更；图中显示当前配置，历史报告仍在下方保留。</p>
      ) : syncFailed && matchingReport ? (
        <p className="basic-uplink-tree-notice" role="status">状态同步失败；图中保留上次成功快照，执行状态尚待重新确认。</p>
      ) : isPreviousSnapshot && matchingReport ? (
        <p className="basic-uplink-tree-notice" role="status">图中为上次成功快照；当前请求尚未取得新结果，请查看下方执行状态。</p>
      ) : null}

      <div className="basic-uplink-tree-scroll" role="region" aria-label="基础上行树状图，可横向滚动" tabIndex={0}>
        <div className="basic-uplink-tree-canvas" style={{ width: WIDTH, height: HEIGHT }}>
          <div className="basic-uplink-tree-column" style={{ left: 24 }}>01 <span>数据与工况</span></div>
          <div className="basic-uplink-tree-column" style={{ left: 360 }}>02 <span>VCP 上行链路</span></div>
          <div className="basic-uplink-tree-column" style={{ left: LEAF_X }}>03 <span>计算方法 → 8 项合同</span></div>

          <svg className="basic-uplink-tree-edges" width={WIDTH} height={HEIGHT} viewBox={`0 0 ${WIDTH} ${HEIGHT}`} aria-hidden="true" focusable="false">
            <path d={`M 288 ${CENTER_Y} H 360`} className="basic-uplink-tree-main-edge" />
            <path d={`M 724 ${CENTER_Y} H 784`} className="basic-uplink-tree-main-edge" />
            <path d={`M 784 ${LEAF_TOP + LEAF_HEIGHT / 2} V ${LEAF_TOP + LEAF_STEP * 7 + LEAF_HEIGHT / 2}`} className="basic-uplink-tree-spine" />
            {Array.from({ length: 8 }, (_, index) => {
              const y = LEAF_TOP + LEAF_STEP * index + LEAF_HEIGHT / 2;
              const item = items[index];
              const focused = Boolean(item && activeId === item.id);
              return <g key={index} className={activeId && !focused ? "is-muted" : ""}>
                <path d={`M 784 ${y} H ${LEAF_X}`} className={`basic-uplink-tree-branch${focused ? " is-focused" : ""}`} />
                <circle cx={784} cy={y} r={3} className={`basic-uplink-tree-junction${focused ? " is-focused" : ""}`} />
              </g>;
            })}
            {activeId ? (() => {
              const y = LEAF_TOP + LEAF_STEP * items.findIndex((item) => item.id === activeId) + LEAF_HEIGHT / 2;
              const direction = y < CENTER_Y ? -1 : 1;
              return <path className="basic-uplink-tree-focus-route" d={`M 724 ${CENTER_Y} H 776 Q 784 ${CENTER_Y} 784 ${CENTER_Y + direction * 8} V ${y - direction * 8} Q 784 ${y} 792 ${y} H ${LEAF_X}`} />;
            })() : null}
          </svg>

          <section className="basic-uplink-tree-node basic-uplink-tree-input" aria-label="测试数据集与输入">
            <span className="basic-uplink-tree-node-label">测试数据集</span>
            <strong className="basic-uplink-tree-dataset-name" title={dataset?.name}>{dataset?.name ?? "等待选择测试集"}</strong>
            <span className="basic-uplink-tree-secondary">{dataset ? `${dataset.case_count} 个可用条件组 · 本次提取当前模式` : "从下方已导入测试集中选择"}</span>
            <dl><div><dt>业务模式</dt><dd>{mode}</dd></div><div><dt>执行范围</dt><dd>仅 8 项基础上行</dd></div></dl>
            <p className="basic-uplink-tree-inputs">处理输入：麦克风混合输入 · 远端播放信号 / 回声参考<br />评估参考：干净近端参考 · 独立噪声输入</p>
            <span className="basic-uplink-tree-secondary">各测项使用自身绑定的场景与时间窗</span>
          </section>

          <section className="basic-uplink-tree-node basic-uplink-tree-vcp" aria-label={vcpSupported ? "VCP 配置的上行处理链" : "算法簇不支持基础上行测试"}>
            <span className="basic-uplink-tree-node-label"><Layers3 size={14} aria-hidden="true" />{vcpSupported ? "VCP 发送处理（TX）" : "待选择 VCP"}</span>
            <h4>{vcpSupported ? "上行语音处理" : "当前算法不可执行"}</h4>
            <p className="basic-uplink-tree-preset">{preset?.label ?? algorithmClusterId}</p>
            {vcpSupported ? <><ol className="basic-uplink-tree-stages" aria-label="所选算法簇的阶段配置">
              {chainStages.map((stage, index) => {
                const option = stage.options.find((entry) => entry.id === preset?.selection[stage.id]);
                return <li key={stage.id} className={option?.id === "off" ? "is-bypassed" : ""}>
                  <span className="basic-uplink-tree-stage-index">{String(index + 1).padStart(2, "0")}</span>
                  <div><strong>{stage.label}</strong><span>{option?.label ?? "配置不可用"}</span></div>
                </li>;
              })}
            </ol>
            <p className="basic-uplink-tree-chain-note">展示所选算法簇的配置；单麦 BF 自动旁路，实际执行以条件组报告为准。</p>
            <div className="basic-uplink-tree-output"><span>发送处理输出（TX）+ 干净近端参考 / 独立噪声输入</span><ArrowRight size={16} aria-hidden="true" /></div>
            </> : <p className="basic-uplink-tree-chain-note">此算法簇不属于 VCP。选择 VCP 算法簇后显示对应阶段配置并启用测试。</p>}
          </section>

          <ol className="basic-uplink-tree-leaves" aria-label="8 项基础上行计算模型与测项">
            {Array.from({ length: 8 }, (_, index) => {
              const item = items[index];
              const result = item ? results.get(item.id) : undefined;
              const method = item ? METHODS[item.family] : undefined;
              const window = item?.test_name.match(/(50|100)\s*ms/i)?.[1];
              const name = method ? `${method.title}${window ? ` · ${window} ms` : item?.family === "out_of_band" ? " · 全片段" : ""}` : "等待测项目录";
              const status = result ? STATUS_LABELS[result.status] : "尚未执行";
              const unit = metricUnit(result?.metric);
              const id = item?.id ?? `waiting-${index}`;
              return <li key={id} style={{ left: LEAF_X, top: LEAF_TOP + LEAF_STEP * index, width: LEAF_WIDTH, height: LEAF_HEIGHT }}>
                <button
                  type="button"
                  className={`basic-uplink-tree-leaf${activeId === id ? " is-focused" : activeId ? " is-muted" : ""}`}
                  aria-pressed={locked === id}
                  disabled={!item}
                  title={item ? `${item.id} · ${item.test_name}${result?.reason ? `\n${result.reason}` : ""}` : "等待服务端目录"}
                  onMouseEnter={() => setHovered(id)}
                  onMouseLeave={() => setHovered(null)}
                  onFocus={() => setHovered(id)}
                  onBlur={() => setHovered(null)}
                  onClick={() => setLocked(locked === id ? null : id)}
                >
                  <div className="basic-uplink-tree-row-id"><span>{String(index + 1).padStart(2, "0")}</span><code>{item?.source_cell ?? "—"}</code></div>
                  <div className="basic-uplink-tree-method"><strong>{name}</strong><span>{method?.detail ?? (catalogError ? "目录读取失败，请在下方重试" : catalogLoading ? "正在读取当前模式的 8 项定义" : "当前目录不可用")}</span><small>{method?.metric ?? "计算方法待确认"}</small></div>
                  <div className="basic-uplink-tree-leaf-result"><span className={`basic-uplink-tree-status is-${result?.status ?? "unrun"}`}>{item ? status : "等待目录"}</span><strong>{result?.value != null ? `${result.value.toLocaleString("zh-CN", { maximumFractionDigits: 3 })}${unit ? ` ${unit}` : ""}` : result ? "无数值" : "—"}</strong></div>
                </button>
              </li>;
            })}
          </ol>
        </div>
      </div>
      <div className="basic-uplink-tree-footer"><span>上行主干 → 按工况分支计算</span><span>工程代理结果 · 正式 ITU-T 合规未测</span></div>
    </figure>
  );
}
