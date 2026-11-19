import { useEffect, useId, useMemo, useState } from "react";
import {
  ArrowRight,
  Check,
  ChevronDown,
  ChevronUp,
  Copy,
  Download,
  GitBranch,
  Layers3,
  LoaderCircle,
  Microscope,
  Network,
  RotateCcw,
  SlidersHorizontal,
  Table,
  Terminal,
  Trophy,
  Workflow,
  Sparkles,
} from "lucide-react";
import { downloadFile } from "../lib/files";
import AudioSynthesize from "./AudioSynthesize";
import JointNextEvaluation from "./JointNextEvaluation";
import RefinementEvaluation from "./RefinementEvaluation";
import StructureEvaluation from "./StructureEvaluation";
import PipelineEvaluation from "./PipelineEvaluation";
import type { PipelineEvaluationData } from "./PipelineEvaluation";
import {
  CHAIN_REPORT_URL,
  CHAIN_STORAGE_KEY,
  chainCommand,
  chainFlags,
  chainPresets,
  chainStages,
  changeStage,
  matchingEvidence,
  matchingPreset,
  optionFor,
  parseChainReport,
  restoreConfiguration,
} from "./enhancement-chain-model";
import type { ChainConfiguration, ChainReport, ChainVariant, StageId } from "./enhancement-chain-model";
import { AecRouteAvailability } from "./AecRouteAvailability";
import "./ns-pipeline.css";
import "./enhancement-pipeline.css";

const fmt = (value: number | null, digits = 2) =>
  value === null ? "—" : `${value > 0 ? "+" : ""}${value.toFixed(digits)}`;
const TX_ORDER: StageId[] = ["limiter", "drc", "agc", "eq", "ns", "aec", "bf"];
const nodeStyle = (x: number, y: number) => ({
  left: `${(x / 1390) * 100}%`,
  top: `${(y / 438) * 100}%`,
});
const rankValue = (value: number | null) => (value === null ? Number.NEGATIVE_INFINITY : value);

function compareVariants(left: ChainVariant, right: ChainVariant): number {
  return (
    right.pass - left.pass ||
    right.mean_sisdr_delta_db - left.mean_sisdr_delta_db ||
    right.mean_stoi_delta - left.mean_stoi_delta ||
    right.mean_noise_delta_db - left.mean_noise_delta_db ||
    rankValue(right.mean_far_delta_db) - rankValue(left.mean_far_delta_db) ||
    left.clipped_cases - right.clipped_cases ||
    left.limited_samples - right.limited_samples ||
    left.mean_rtf - right.mean_rtf
  );
}

function ChainFlow({
  configuration,
  selected,
  onSelect,
}: {
  configuration: ChainConfiguration;
  selected: StageId;
  onSelect: (stage: StageId) => void;
}) {
  const uid = useId().replace(/:/g, "");
  return (
    <figure className="chain-flow" style={{ marginTop: 16 }}>
      <div className="chain-section-head">
        <div>
          <span>SIGNAL TOPOLOGY · TX & RX</span>
          <h2>全链路双向声学信号流图</h2>
          <p>
            发送处理（TX）为麦克风混合输入到网络发送的实际处理顺序（从右至左）；接收处理（RX）为远端播放信号 / 回声参考链路。可横向滑动。
          </p>
        </div>
        <a className="text-link" href="#workflow">
          原始流程图 <ArrowRight size={14} />
        </a>
      </div>
      <div
        className="chain-flow-scroll"
        tabIndex={0}
        role="region"
        aria-label="综合增强 TX / RX 流程图，可横向滚动"
      >
        <div className="chain-canvas">
          <svg width="1390" height="438" viewBox="0 0 1390 438" aria-hidden="true">
            <defs>
              <marker
                id={`${uid}-audio`}
                markerWidth="7"
                markerHeight="7"
                refX="6"
                refY="3.5"
                orient="auto"
              >
                <path d="M0,0 L7,3.5 L0,7" fill="#70829a" />
              </marker>
              <marker
                id={`${uid}-ref`}
                markerWidth="7"
                markerHeight="7"
                refX="6"
                refY="3.5"
                orient="auto"
              >
                <path d="M0,0 L7,3.5 L0,7" fill="#b58a2b" />
              </marker>
              <marker
                id={`${uid}-echo`}
                markerWidth="7"
                markerHeight="7"
                refX="6"
                refY="3.5"
                orient="auto"
              >
                <path d="M0,0 L7,3.5 L0,7" fill="#9a78b4" />
              </marker>
            </defs>
            <rect x="14" y="65" width="1330" height="145" rx="10" fill="#f5f8fd" />
            <rect x="14" y="284" width="1330" height="128" rx="10" fill="#fbfaf6" />
            <text x="32" y="41" className="chain-svg-lane">
              发送处理（TX） · 单路增强
            </text>
            <text x="1195" y="41" className="chain-svg-note">
              近端语音 + 噪声 + 回声
            </text>
            <text x="32" y="263" className="chain-svg-lane">
              接收处理（RX） · 现有下行实现
            </text>
            {Array.from({ length: 8 }, (_, index) => (
              <path
                key={index}
                d={`M${180 + index * 145},134 H${index === 0 ? 164 : 167 + index * 145}`}
                stroke="#70829a"
                fill="none"
                markerEnd={`url(#${uid}-audio)`}
              />
            ))}
            {["M164 358 H255", "M387 358 H470", "M602 358 H685", "M817 358 H1195"].map(
              (path) => (
                <path
                  key={path}
                  d={path}
                  stroke="#70829a"
                  fill="none"
                  markerEnd={`url(#${uid}-audio)`}
                />
              )
            )}
            <path
              d="M817 358 H971 V178"
              stroke="#b58a2b"
              strokeWidth="1.5"
              fill="none"
              markerEnd={`url(#${uid}-ref)`}
            />
            <path
              d="M1327 358 H1360 V134 H1327"
              stroke="#9a78b4"
              strokeDasharray="5 4"
              fill="none"
              markerEnd={`url(#${uid}-echo)`}
            />
            <text x="840" y="243" className="chain-svg-reference">
              远端播放信号 / 回声参考
            </text>
            <text x="1146" y="243" className="chain-svg-note">
              声学耦合回到麦克风
            </text>
          </svg>
          <div className="chain-node chain-node--endpoint" style={nodeStyle(32, 90)}>
            <span>TX OUT</span>
            <strong>发送处理输出（TX）</strong>
            <small>网络发送 / Codec</small>
          </div>
          {TX_ORDER.map((id, index) => {
            const stage = chainStages.find((item) => item.id === id)!;
            const option = optionFor(stage, configuration);
            const off = option.id === "off";
            return (
              <button
                key={id}
                type="button"
                style={nodeStyle(177 + index * 145, 90)}
                className={`chain-node${selected === id ? " is-selected" : ""}${
                  off ? " is-bypassed" : ""
                }`}
                aria-label={`查看${stage.label}节点：${option.label}`}
                aria-pressed={selected === id}
                onClick={() => onSelect(id)}
              >
                <span>
                  {stage.id.toUpperCase()} · STEP {7 - index}
                </span>
                <strong>{stage.label.split(" ")[0]}</strong>
                <small>{option.label.split(" · ")[0]}</small>
              </button>
            );
          })}
          <div className="chain-node chain-node--endpoint" style={nodeStyle(1195, 90)}>
            <span>TX IN</span>
            <strong>麦克风混合输入</strong>
            <small>同步阵列 · 实际标定</small>
          </div>
          {[
            [32, "RX IN", "远端播放信号 / 回声参考", "接收网络音频"],
            [255, "RX NR", "轻度降噪", "下行独立状态"],
            [470, "RX EQ", "播放均衡", "独立 EQ 配置"],
            [685, "RX DRC", "动态范围控制", "软压缩与饱和"],
            [1195, "SPEAKER", "接收处理输出（RX）", "扬声器播放 · 与远端播放信号 / 回声参考同源"],
          ].map(([left, tag, title, sub]) => (
            <div
              key={String(tag)}
              className="chain-node chain-node--rx"
              style={nodeStyle(Number(left), 314)}
            >
              <span>{tag}</span>
              <strong>{title}</strong>
              <small>{sub}</small>
            </div>
          ))}
        </div>
      </div>
      <figcaption>
        <GitBranch size={15} />
        <span>
          本轮整链评测使用冻结的远端播放信号 / 回声参考（<code>--echo</code>
          ），下行未重复处理。RX NR → EQ → DRC 是现有实现拓扑；其声学效果不计入本轮 TX 结论。
        </span>
      </figcaption>
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
      {
        label: "主语音 ΔSI-SDR / ΔSTOI",
        value: `${fmt(variant.mean_sisdr_delta_db)} dB`,
        detail: `ΔSTOI ${fmt(variant.mean_stoi_delta, 4)} · 相对 DAS 控制的均值`,
      },
      {
        label: "双讲 ΔSI-SDR / ΔSTOI",
        value: `${fmt(variant.mean_double_sisdr_delta_db)} dB`,
        detail: `${variant.double_cases} 个适用案例 · ΔSTOI ${fmt(variant.mean_double_stoi_delta, 4)}`,
      },
      {
        label: "纯噪声 / 远端单讲衰减增量",
        value: `${variant.noise_censored_cases ? "≥ " : ""}${fmt(variant.mean_noise_delta_db)} dB`,
        detail: `远端单讲 ${variant.far_censored_cases ? "≥ " : ""}${fmt(
          variant.mean_far_delta_db
        )} dB · ${variant.far_cases} 案例 · 此项不是 ERLE`,
      },
      {
        label: "输出真实变化",
        value: `${variant.changed}/${variant.cases}`,
        detail: "完整输出文件 SHA256 与 DAS 控制逐一配对",
      },
      {
        label: "削波 / 终级限幅触发",
        value: `${variant.clipped_cases} / ${variant.limited_samples}`,
        detail: "削波案例数 / 输入时段限幅样本数；0 次不证明机制已被覆盖",
      },
      {
        label: "均值处理 RTF",
        value: variant.mean_rtf.toFixed(4),
        detail: "aarch64 主机 · 4 worker 离线回放；不是设备 RTD 或回调准入",
      },
    ],
    caseGroups: variant.caseGroups,
  };
}

export default function EnhancementPipeline() {
  const [configuration, setConfiguration] = useState<ChainConfiguration>(() => {
    try {
      return restoreConfiguration(localStorage.getItem(CHAIN_STORAGE_KEY));
    } catch {
      return restoreConfiguration(null);
    }
  });

  // Main navigation tab
  const [activeTab, setActiveTab] = useState<"pipeline" | "matrix" | "research" | "export">("pipeline");
  const [researchTab, setResearchTab] = useState<"joint_next" | "structure" | "refinement">("joint_next");
  const [showSvgFlow, setShowSvgFlow] = useState(false);

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
    return [
      ...control,
      ...report.variants.filter((row) => row.id !== "array_control").sort(compareVariants),
    ];
  }, [report]);

  const bestVariant = rankedVariants.find((row) => row.id !== "array_control");
  const selected = chainStages.find((stage) => stage.id === selectedStage)!;
  const option = optionFor(selected, configuration);
  const command = useMemo(() => chainCommand(configuration), [configuration]);

  useEffect(() => {
    try {
      localStorage.setItem(CHAIN_STORAGE_KEY, JSON.stringify(configuration));
    } catch {
      /* The draft still works without browser persistence. */
    }
    setFeedback("");
  }, [configuration]);

  useEffect(() => {
    const controller = new AbortController();
    setError("");
    setReport(null);
    fetch(CHAIN_REPORT_URL, { signal: controller.signal, cache: "no-store" })
      .then((response) => {
        if (!response.ok) throw new Error(`评测文件读取失败（HTTP ${response.status}）`);
        if (!response.headers.get("content-type")?.includes("application/json"))
          throw new Error("评测入口未返回 JSON，请检查评测文件是否已发布。");
        return response.json();
      })
      .then((data: unknown) => setReport(parseChainReport(data)))
      .catch((reason: unknown) => {
        if (!controller.signal.aborted)
          setError(reason instanceof Error ? reason.message : "评测读取失败");
      });
    return () => controller.abort();
  }, [retry]);

  const usePreset = (id: string) => {
    const next = chainPresets.find((item) => item.id === id);
    if (next) {
      setConfiguration(next);
    }
  };

  const exportConfig = () => {
    downloadFile(
      JSON.stringify(
        {
          schema_version: 1,
          ...configuration,
          preset_id: preset?.id ?? null,
          evaluation_status: variant ? "measured_configuration" : "unmeasured_configuration",
          frozen_evaluation_source: variant ? report?.source : null,
          cli_flags: chainFlags(configuration),
          beam_delays_samples: null,
          requires_calibrated_beam_delays: configuration.selection.bf !== "off",
          playback_reference: "actual-playback.wav",
          rx_reprocessing: false,
          standard_compliance: false,
        },
        null,
        2
      ) + "\n",
      "enhancement-chain.json",
      "application/json"
    );
    setFeedback("链路配置已导出；这是回放配置，未在设备上应用。");
  };

  const copyCommand = async () => {
    try {
      await navigator.clipboard.writeText(command);
      setFeedback("CLI 模板已复制；执行前填写真实文件与逐通道标定延迟。");
    } catch {
      setFeedback("当前浏览器无法复制，请展开下方 CLI 模板并手动选择文本。");
    }
  };

  // 6 Forward stages in TX signal order
  const PIPELINE_STAGES: StageId[] = ["bf", "aec", "ns", "eq", "agc", "drc"];

  return (
    <div className="nsp-page chain-page">
      {/* Hero Introduction */}
      <section className="nsp-intro">
        <div>
          <span>INTEGRATED VOICE ENHANCEMENT · TX / RX</span>
          <h2>综合语音增强全链路 VCP 评测与编排平台</h2>
          <p>
            在同一车载音频流水线中串联波束形成 (BF)、回声消除 (AEC)、降噪 (NS)、均衡 (EQ)、增益控制 (AGC) 与动态范围限幅 (DRC)。
            支持整链冻结回放、多方案对比、多维独立准入与 CLI 自动化运行。
          </p>
        </div>
        <div className="nsp-summary" aria-label="综合链路汇总">
          <div>
            <strong>6</strong>
            <span>核心增强模块</span>
          </div>
          <div>
            <strong>{chainPresets.length}</strong>
            <span>可选方案组合</span>
          </div>
          <div>
            <strong>{report?.case_count ?? 72}</strong>
            <span>场景闭环测试</span>
          </div>
          <div>
            <strong>{report?.cli_runs ?? 10}</strong>
            <span>完整矩阵回放</span>
          </div>
        </div>
      </section>

      {/* Main Top Navigation Tabs */}
      <div className="chain-main-tabs">
        <button
          className={`chain-main-tab ${activeTab === "pipeline" ? "active" : ""}`}
          onClick={() => setActiveTab("pipeline")}
        >
          <Workflow size={16} />
          <span>链路拓扑与在线编排</span>
          <span className="tab-badge">核心</span>
        </button>
        <button
          className={`chain-main-tab ${activeTab === "matrix" ? "active" : ""}`}
          onClick={() => setActiveTab("matrix")}
        >
          <Table size={16} />
          <span>全链路方案对比矩阵</span>
          <span className="tab-badge">{rankedVariants.length} 方案</span>
        </button>
        <button
          className={`chain-main-tab ${activeTab === "research" ? "active" : ""}`}
          onClick={() => setActiveTab("research")}
        >
          <Microscope size={16} />
          <span>前沿演进与候选准入</span>
          <span className="tab-badge">Joint V2 / 结构 / 调优</span>
        </button>
        <button
          className={`chain-main-tab ${activeTab === "export" ? "active" : ""}`}
          onClick={() => setActiveTab("export")}
        >
          <Terminal size={16} />
          <span>配置导出与 CLI 证据</span>
        </button>
      </div>

      {/* TAB 1: 链路拓扑与在线编排 */}
      {activeTab === "pipeline" && (
        <>
          {/* Preset Selection & State */}
          <section className="chain-builder">
            <div className="chain-toolbar">
              <div>
                <h2>快速选择预设方案</h2>
                <p>点击下方推荐方案即可一键切换整链配置，或在下方逐级定制算法。</p>
              </div>
              <div className="chain-actions">
                <button
                  type="button"
                  className="button secondary"
                  onClick={() => usePreset("vcp_joint")}
                >
                  <RotateCcw size={14} />
                  恢复联合方案
                </button>
                <button type="button" className="button secondary" onClick={exportConfig}>
                  <Download size={14} />
                  导出 JSON
                </button>
              </div>
            </div>

            {/* Popular Preset Cards Grid */}
            <div style={{ padding: "0 22px 18px" }}>
              <div className="chain-preset-cards">
                {[
                  {
                    id: "vcp_joint",
                    title: "VCP 联合优化",
                    badge: "推荐 · 最优",
                    isRec: true,
                    desc: "多参考双讲保护、统一原声预算与轻度峰值控制，综合回放 64/72 通过。",
                  },
                  {
                    id: "vcp_unified",
                    title: "VCP Unified",
                    badge: "全链路融合",
                    isRec: true,
                    desc: "Speex MDF/GCC-PHAT/NLMS 融合 AEC + 四专家 NS + 融合 EQ/AGC。",
                  },
                  {
                    id: "vcp_default",
                    title: "VCP 默认",
                    badge: "基线标配",
                    isRec: false,
                    desc: "Speex MDF + Coherence RES + WebRTC NS + Guarded5 + 默认 DRC。",
                  },
                  {
                    id: "super_gaussian_optimal",
                    title: "Super-Gaussian 最优",
                    badge: "单级突破",
                    isRec: false,
                    desc: "Super-Gaussian H1 谱增益降噪，单级筛选 60/72 通过，保真度高。",
                  },
                  {
                    id: "all_serial",
                    title: "六级全串联",
                    badge: "全级开通",
                    isRec: false,
                    desc: "开通 BF、AEC、WebRTC NS、EQ、Guarded5 AGC 与 TX DRC 全部六级。",
                  },
                  {
                    id: "array_control",
                    title: "DAS 对照",
                    badge: "原始基准",
                    isRec: false,
                    desc: "仅阵列延迟求和，其余算法全部旁路，用于纯物理声学基准核验。",
                  },
                ].map((item) => {
                  const isAct = preset?.id === item.id;
                  return (
                    <div
                      key={item.id}
                      className={`chain-preset-card ${isAct ? "active" : ""}`}
                      onClick={() => usePreset(item.id)}
                    >
                      <div className="chain-preset-card-top">
                        <span className="chain-preset-card-title">{item.title}</span>
                        <span
                          className={`chain-preset-card-badge ${item.isRec ? "rec" : "norm"}`}
                        >
                          {item.badge}
                        </span>
                      </div>
                      <p className="chain-preset-card-desc">{item.desc}</p>
                    </div>
                  );
                })}
              </div>

              {/* Status bar */}
              <div className={`chain-config-state${variant ? " is-measured" : ""}`} style={{ borderRadius: 8 }}>
                {variant ? <Check size={16} /> : <Layers3 size={16} />}
                <div>
                  <strong>
                    {!report
                      ? error
                        ? "整链评测读取失败"
                        : "正在读取整链评测..."
                      : variant
                      ? `${variant.label} · 已有整链评测`
                      : "当前自定义组合 · 待评测"}
                  </strong>
                  <span>
                    {preset?.description ??
                      "已自定义更换技术组合。导出配置后重新回放，即可得到该组合自己的各项指标与案例图。"}
                  </span>
                </div>
                <span className="chain-state-tag">
                  {!report
                    ? error
                      ? "不可用"
                      : "读取中"
                    : variant
                    ? `${variant.pass}/${variant.cases} 通过`
                    : "自定义组合"}
                </span>
              </div>
            </div>
          </section>

          {/* Interactive Pipeline Stepper (Horizontal Flow in Signal Order) */}
          <section className="chain-builder" style={{ marginTop: 18 }}>
            <div className="chain-section-head">
              <div>
                <span>SIGNAL PIPELINE · TX PROCESSING STEPS</span>
                <h2>信号处理流水线步骤（沿信号流向从左到右）</h2>
                <p>点击任意处理节点，即可在下方查看该模块说明、算法实现源码与快速参数配置。</p>
              </div>
              <button
                type="button"
                className="button secondary"
                onClick={() => setShowSvgFlow(!showSvgFlow)}
                style={{ display: "inline-flex", alignItems: "center", gap: 6 }}
              >
                <GitBranch size={14} />
                <span>{showSvgFlow ? "收起双向信号拓扑图" : "展开双向信号拓扑图 (SVG)"}</span>
                {showSvgFlow ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
              </button>
            </div>

            <div style={{ padding: "0 22px 18px" }}>
              <div className="chain-pipeline-stepper">
                <div className="chain-step-node" style={{ background: "#edf3fa", borderColor: "#cbd5e1" }}>
                  <span className="chain-step-node-tag">INPUT</span>
                  <span className="chain-step-node-name">麦克风混合输入</span>
                  <span className="chain-step-node-sub">同步阵列采集</span>
                </div>
                <ArrowRight size={16} className="chain-step-arrow" />

                {PIPELINE_STAGES.map((id, idx) => {
                  const stage = chainStages.find((item) => item.id === id)!;
                  const currentOpt = optionFor(stage, configuration);
                  const isAct = selectedStage === id;
                  const isOff = currentOpt.id === "off";
                  return (
                    <div key={id} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <div
                        className={`chain-step-node ${isAct ? "active" : ""} ${isOff ? "is-bypassed" : ""}`}
                        onClick={() => setSelectedStage(id)}
                      >
                        <span className="chain-step-node-tag">
                          STEP {idx + 1} · {id.toUpperCase()}
                        </span>
                        <span className="chain-step-node-name">{stage.label.split(" ")[0]}</span>
                        <span className="chain-step-node-sub">
                          {isOff ? "旁路 (Bypass)" : currentOpt.label.split(" · ")[0]}
                        </span>
                      </div>
                      <ArrowRight size={16} className="chain-step-arrow" />
                    </div>
                  );
                })}

                <div className="chain-step-node" style={{ background: "#eff6ff", borderColor: "#93c5fd" }}>
                  <span className="chain-step-node-tag">OUTPUT</span>
                  <span className="chain-step-node-name" style={{ color: "#1d4ed8" }}>发送处理输出（TX）</span>
                  <span className="chain-step-node-sub">网络发送</span>
                </div>
              </div>

              {/* Collapsible SVG Flow if toggled on */}
              {showSvgFlow && (
                <ChainFlow
                  configuration={configuration}
                  selected={selectedStage}
                  onSelect={setSelectedStage}
                />
              )}

              {/* Selected Node Details Card */}
              <div className="chain-node-detail" style={{ borderRadius: 8, marginTop: 14 }}>
                <div>
                  <span>当前所选模块 · {selected.id.toUpperCase()}</span>
                  <h3>
                    {selected.label}{" "}
                    <small>
                      {option.id === "off"
                        ? "旁路 (Bypass)"
                        : option.candidate
                        ? "候选技术"
                        : "当前执行中"}
                    </small>
                  </h3>
                  <p>{selected.description}</p>
                  <strong style={{ color: "#2563eb" }}>当前算法：{option.label}</strong>
                  <p>{option.note}</p>
                  <AecRouteAvailability routeId={configuration.selection.aec} />
                </div>
                <div className="chain-detail-evidence">
                  <span>算法源码实现</span>
                  <code>{selected.source}</code>
                  <a href={`#${selected.tab}`} className="text-link">
                    查看 {selected.label.split(" ")[0]} 独立评测、波形与频谱图 <ArrowRight size={14} />
                  </a>
                </div>
              </div>

              {/* Stage Quick Customization Grid */}
              <div style={{ marginTop: 16 }}>
                <h4 style={{ margin: "0 0 10px 0", fontSize: "0.8125rem", color: "#475569" }}>
                  逐级微调配置（修改任意一项将自动生成自定义链路）：
                </h4>
                <div className="chain-stage-grid" style={{ borderRadius: 8, border: "1px solid #e2e8f0" }}>
                  {chainStages.map((stage) => {
                    const currentOpt = optionFor(stage, configuration);
                    const isBypassed = currentOpt.id === "off";
                    const isCur = stage.id === selectedStage;
                    return (
                      <div
                        key={stage.id}
                        className={`chain-stage ${isCur ? "is-selected" : ""} ${
                          isBypassed ? "is-bypassed" : ""
                        }`}
                        onClick={() => setSelectedStage(stage.id)}
                      >
                        <label>
                          <span>{stage.id.toUpperCase()}</span>
                          <strong>{stage.label}</strong>
                        </label>
                        <select
                          aria-label={`选择${stage.label}配置`}
                          value={currentOpt.id}
                          onChange={(e) => {
                            setConfiguration(changeStage(configuration, stage.id, e.target.value));
                            setSelectedStage(stage.id);
                          }}
                        >
                          {stage.options.map((opt) => (
                            <option key={opt.id} value={opt.id}>
                              {opt.label}
                            </option>
                          ))}
                        </select>
                        <p>{currentOpt.note}</p>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </section>

          {/* Pipeline Evaluation: Metrics & Cases */}
          {variant ? (
            <PipelineEvaluation
              title={`${variant.label} · 方案评测`}
              evaluation={evaluationFor(variant, report!)}
              graphFocus="all"
            />
          ) : (
            <section className="chain-pending" aria-label="自定义组合评测状态" style={{ marginTop: 18, borderRadius: 8 }}>
              <Workflow size={24} />
              <div>
                <h2>当前自定义组合尚未生成预存的 72 场景评测</h2>
                <p>
                  当前配置没有完全匹配的离线回放。上方的预设卡片展示了已完成评测的组合；点击任意预设即可直接载入对应的指标、波形与频谱。
                </p>
              </div>
            </section>
          )}

          {/* Live Audio Synthesize Area */}
          <section id="chain-run" className="chain-builder chain-run" style={{ marginTop: 18, borderRadius: 8 }}>
            <div className="chain-section-head">
              <div>
                <span>REPLAY YOUR AUDIO · INTERACTIVE AUDITION</span>
                <h2 id="chain-run-title">用当前链路处理并试听音频</h2>
                <p>上传麦克风混合输入与远端播放信号 / 回声参考，在线实时试听并对比处理后发送输出与输入的波形和频谱。</p>
              </div>
            </div>
            <AudioSynthesize configuration={configuration} embedded />
          </section>
        </>
      )}

      {/* TAB 2: 全链路方案对比矩阵 */}
      {activeTab === "matrix" && (
        <section id="chain-matrix" className="chain-builder chain-matrix" style={{ borderRadius: 8 }}>
          <div className="chain-section-head">
            <div>
              <span>BENCHMARK MATRIX · 10 VARIANTS</span>
              <h2>全链路方案多维指标对比矩阵</h2>
              <p>在同一 72 个冻结场景（安静、行车风噪、胎噪、空调大风、单双讲、远端回声）下的完整横向回归。</p>
            </div>
          </div>

          {bestVariant && (
            <div className="chain-best-summary">
              <Trophy size={20} />
              <div>
                <span>OPTIMAL CONFIGURATION</span>
                <strong>
                  推荐最优方案：{bestVariant.label}（{bestVariant.pass}/{bestVariant.cases} 场景通过）
                </strong>
                <p>{bestVariant.description}</p>
              </div>
              <button
                type="button"
                className="button primary"
                onClick={() => {
                  usePreset(bestVariant.id);
                  setActiveTab("pipeline");
                }}
              >
                套用此最优方案
              </button>
            </div>
          )}

          <div className="chain-matrix-scroll" tabIndex={0}>
            <table>
              <thead>
                <tr>
                  <th scope="col" style={{ width: 280 }}>方案名称与定位</th>
                  <th scope="col">综合通过数</th>
                  <th scope="col">主语音 ΔSI-SDR</th>
                  <th scope="col">主语音 ΔSTOI</th>
                  <th scope="col">噪声衰减增量</th>
                  <th scope="col">双讲 ΔSI-SDR</th>
                  <th scope="col">远端单讲衰减</th>
                  <th scope="col">真实变化 / 削波</th>
                </tr>
              </thead>
              <tbody>
                {rankedVariants.map((row) => {
                  const isCur = variant?.id === row.id;
                  const isBest = row.id === bestVariant?.id;
                  const isCtrl = row.id === "array_control";
                  return (
                    <tr
                      key={row.id}
                      className={`${isCur ? "is-current" : ""} ${isCtrl ? "is-control" : ""} ${
                        isBest ? "is-best" : ""
                      }`}
                    >
                      <th scope="row">
                        <button
                          type="button"
                          onClick={() => {
                            usePreset(row.id);
                            setActiveTab("pipeline");
                          }}
                          aria-pressed={isCur}
                        >
                          {row.label}
                          {isBest ? <span className="chain-best-pill">最优</span> : null}
                          <ArrowRight size={12} />
                        </button>
                        <small>{isCtrl ? "配对控制 · 不参与排名" : row.description}</small>
                      </th>
                      <td>
                        <strong style={{ color: row.pass >= 60 ? "#16a34a" : "#0f172a" }}>
                          {row.pass}/{row.cases}
                        </strong>
                      </td>
                      <td>{fmt(row.mean_sisdr_delta_db)} dB</td>
                      <td>{fmt(row.mean_stoi_delta, 4)}</td>
                      <td>
                        {row.noise_censored_cases ? "≥ " : ""}
                        {fmt(row.mean_noise_delta_db)} dB
                      </td>
                      <td>
                        {fmt(row.mean_double_sisdr_delta_db)} dB
                        <small>{row.double_cases} 个适用案例</small>
                      </td>
                      <td>
                        {row.far_censored_cases ? "≥ " : ""}
                        {fmt(row.mean_far_delta_db)} dB
                        <small>{row.far_cases} 个适用案例</small>
                      </td>
                      <td>
                        {row.changed} / {row.clipped_cases}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="chain-gates">
            <p>
              <strong>联合门限规范：</strong>主语音与适用双讲窗 ΔSI-SDR ≥ −0.1 dB、ΔSTOI ≥ −0.02，投影增益
              −6～18 dB；噪声衰减增量 ≥ 0 dB；有远端单讲时其衰减增量 ≥ 6 dB；整段峰值 ≤ −0.1 dBFS，削波为 0%。
            </p>
            <p>
              <strong>防退化机制：</strong>同门限严格拒绝了静音负对照的全部案例。低于量化分辨率的衰减按半
              LSB 保守下界汇总并标“≥”；本矩阵 standard_compliance=false，整车声学验收仍需独立测量。
            </p>
          </div>
        </section>
      )}

      {/* TAB 3: 前沿演进与准入证据 */}
      {activeTab === "research" && (
        <section className="chain-builder" style={{ borderRadius: 8 }}>
          <div className="chain-section-head">
            <div>
              <span>ADVANCED RESEARCH & ADMISSION EVIDENCE</span>
              <h2>前沿技术演进、结构解耦与独立准入证据</h2>
              <p>收纳 Joint V2 准入评估、预测保护与持续偏置统计、以及多维精细化调优实验。按需切换查看。</p>
            </div>
          </div>

          <div style={{ padding: "0 22px 22px" }}>
            {/* Subtabs */}
            <div className="chain-subtabs">
              <button
                className={`chain-subtab ${researchTab === "joint_next" ? "active" : ""}`}
                onClick={() => setResearchTab("joint_next")}
              >
                Joint V2 独立准入结论
              </button>
              <button
                className={`chain-subtab ${researchTab === "structure" ? "active" : ""}`}
                onClick={() => setResearchTab("structure")}
              >
                结构解耦与阶段验证 (Structure)
              </button>
              <button
                className={`chain-subtab ${researchTab === "refinement" ? "active" : ""}`}
                onClick={() => setResearchTab("refinement")}
              >
                多维精细化调优实验 (Refinement)
              </button>
            </div>

            {/* Subtab Contents */}
            {researchTab === "joint_next" && <JointNextEvaluation />}
            {researchTab === "structure" && <StructureEvaluation />}
            {researchTab === "refinement" && <RefinementEvaluation />}
          </div>
        </section>
      )}

      {/* TAB 4: 配置导出与 CLI 证据 */}
      {activeTab === "export" && (
        <section className="chain-builder chain-export" style={{ borderRadius: 8 }}>
          <div className="chain-section-head">
            <div>
              <span>REPRODUCIBLE CONFIGURATION · CLI PROOF</span>
              <h2>可复现配置与 CLI 运行指令</h2>
              <p>当前链路的完整命令行参数模板，可直接在主机环境中执行 72 场景回放验证。</p>
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <button type="button" className="button secondary" onClick={copyCommand}>
                <Copy size={14} />
                复制 CLI 指令
              </button>
              <button type="button" className="button secondary" onClick={exportConfig}>
                <Download size={14} />
                导出配置 JSON
              </button>
            </div>
          </div>

          <details open>
            <summary>查看当前配置对应的 CLI 运行脚本</summary>
            <pre>
              <code>{command}</code>
            </pre>
          </details>

          <p className="chain-feedback" role="status">
            {feedback || "配置与 CLI 导出仅用于本地环境自动化回放，不会直接修改当前硬件寄存器。"}
          </p>

          {report && (
            <dl className="chain-proof">
              <div>
                <dt>真实回放来源</dt>
                <dd>{report.source}</dd>
              </div>
              <div>
                <dt>CLI SHA256</dt>
                <dd>{report.cli_sha256}</dd>
              </div>
              <div>
                <dt>配对结果 SHA256</dt>
                <dd>{report.results_sha256}</dd>
              </div>
              <div>
                <dt>冻结输入清单 SHA256</dt>
                <dd>{report.manifest_sha256}</dd>
              </div>
            </dl>
          )}
        </section>
      )}
    </div>
  );
}
