import { lazy, Suspense, useEffect, useState } from "react";
import {
  Download,
  RotateCcw,
  Play,
  CheckCircle2,
  XCircle,
  AlertCircle,
  Terminal,
  ChevronDown,
  ChevronUp,
  Loader2,
  ShieldCheck,
  Activity,
  Check,
  MinusCircle,
} from "lucide-react";
const VehicleFrontend = lazy(() => import("./VehicleFrontend"));
const ActiveDataset = lazy(() => import("./ActiveDataset"));
import "../evaluation.css";
import "./itut-alignment.css";

type Failure = {
  pair?: number;
  label: string;
  segment: string;
  metric: string;
  value: number | null;
  operator: string;
  threshold: number | null;
};
type Job = {
  name: string;
  status: string;
  acceptance_passed: boolean;
  passed_phases?: number;
  quality_passed_phases?: number;
  phases?: number;
  failures?: Failure[];
};
type Item = {
  id: string;
  sheet: string;
  source_cell: string;
  test_name: string;
  family: string;
  bandwidth: string;
  direction: string;
  unit: string;
  lower_limit: unknown;
  upper_limit: unknown;
};
type Ledger = {
  standard_compliance: false;
  formal_test_status: "not_measured";
  items: Item[];
  paced_diagnostic?: {
    sample_rate: number;
    microphones: number;
    seconds: number;
    process_p95_ms: number;
    process_max_ms: number;
    completion_deadline_misses: number;
    passed_phases: number;
    phases: number;
  };
  family_counts: Record<string, number>;
  simulation_identity: {
    cli_sha256: string;
    rules_sha256: string;
    paced: boolean;
    seconds_per_sequence?: number;
  };
  standards: Record<string, { recommendation: string; edition?: string; url: string }>;
  engineering: { complete: boolean; passed_jobs: number; jobs: number; results: Job[] };
};

interface TestStage {
  id: number;
  name: string;
  detail: string;
  status: "pending" | "running" | "passed" | "failed" | "skipped";
  duration_ms: number;
}

interface TestRunResult {
  success: boolean;
  engineering_success: boolean;
  standard_compliance: false;
  formal_test_status: "not_measured";
  returncode: number;
  duration_s: number;
  total_count: number;
  passed_count: number;
  failed_count: number;
  error_count: number;
  skipped_count: number;
  pass_rate: number;
  failure_reasons: string[];
  stages: TestStage[];
  log: string;
  scope: string;
  strict: boolean;
  pipeline?: {
    id: string;
    cli_profile: string;
    name: string;
    topology: string;
    profile: string | null;
    dsp_delay_ms: number | null;
    rtf: number | null;
    aec_backend: string | null;
    ns_backend: string | null;
    weights: number[];
    tx_drc: boolean | null;
    tx_drc_config?: {
      threshold_dbfs?: number;
      ratio?: number;
      ceiling_dbfs?: number;
    } | null;
    standard_compliance: false;
    status: "passed" | "failed";
    returncode: number | null;
  };
  timestamp: string;
}

const BASE = "/evaluation/itut-current";
const PAGE_SIZE = 25;
const SECTIONS = [
  { id: "formal", label: "正式测项 (336项)" },
  { id: "engineering", label: "工程矩阵" },
  { id: "dataset", label: "当前数据集" },
  { id: "frontend", label: "车载前端" },
  { id: "standards", label: "标准与边界" },
] as const;

const FAMILY: Record<string, [string, string]> = {
  round_trip_delay: ["往返时延", "物理回路、编解码与播放/采集时间戳"],
  slr: ["发送响度 SLR", "人工嘴 / HATS 标定及标准响度算法"],
  rlr: ["接收响度 RLR", "人工耳 / HATS、音量设置及标准响度算法"],
  rlr_linearity: ["RLR 线性度", "规定输入电平下的标定响度测量"],
  frequency_response: ["频率响应", "声学传递响应、业务带宽对应掩模及归一化条件"],
  idle_noise_level: ["空闲噪声电平", "声学电平、规定计权与 50/100 ms 窗口"],
  idle_noise_peaks: ["空闲噪声谱峰", "标定功率谱及正式频谱掩模"],
  listening_quality: ["单向语音质量", "POLQA / TMOS 算法、规定信号及收发测点"],
  out_of_band: ["带外响应", "规定激励、标定测点和仪器修正"],
  double_talk_classification: ["双讲分类", "规定句序、电平轨迹与 A1/A2 判据"],
  tclw: ["声学耦合 TCLw", "标定终端声学路径、计权及音量"],
  hats_snr: ["HATS 信噪比", "HATS 语音/噪声电平、计权及窗口"],
  g_mos: ["整体质量 GMOS", "规定噪声场景及独立质量评估方法"],
  n_mos: ["噪声质量 NMOS", "规定噪声场景及独立噪声质量评分"],
  s_mos: ["语音质量 SMOS", "规定噪声场景及独立语音质量评分"],
  cng_level: ["舒适噪声电平", "CNG 实现、标定参考噪声及计权"],
  cng_spectrum: ["舒适噪声频谱", "CNG 实现、标定参考谱及容差掩模"],
};

const value = (v: unknown) => (v === null || v === undefined ? "未定义" : String(v));
const number = (v: number | null) => (v === null ? "未测" : v.toFixed(3));

function JobFailures({ job }: { job: Job }) {
  const [open, setOpen] = useState(false);
  return (
    <details className="alignment-failures" onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary>
        {job.name} · {job.failures?.length} 个失败指标
      </summary>
      {open ? (
        <div className="table-scroll">
          <table className="audio-table">
            <thead>
              <tr>
                <th>说话人组</th>
                <th>阶段 / 窗口</th>
                <th>指标</th>
                <th>实测</th>
                <th>工程要求</th>
              </tr>
            </thead>
            <tbody>
              {job.failures?.map((gate, index) => (
                <tr key={index}>
                  <td>{gate.pair ?? "未定义"}</td>
                  <td>
                    {gate.label} / {gate.segment}
                  </td>
                  <td>{gate.metric}</td>
                  <td>{number(gate.value)}</td>
                  <td>
                    {gate.operator} {number(gate.threshold)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </details>
  );
}

export default function ItuAlignment() {
  const [ledger, setLedger] = useState<Ledger | null>(null);
  const [error, setError] = useState("");
  const [refresh, setRefresh] = useState(0);
  const [family, setFamily] = useState("all");
  const [bandwidth, setBandwidth] = useState("all");
  const [query, setQuery] = useState("");
  const [section, setSection] = useState<(typeof SECTIONS)[number]["id"]>("formal");
  const [page, setPage] = useState(0);

  // One-click Test Runner States
  const [isRunning, setIsRunning] = useState(false);
  const [runProgress, setRunProgress] = useState(0);
  const [pipeline, setPipeline] = useState<"vcp_unified" | "vcp_joint" | "vcp_default">("vcp_unified");
  const [runScope, setRunScope] = useState<"full" | "methods" | "contract">("full");
  const [strictGate, setStrictGate] = useState(true);
  const [lastRunResult, setLastRunResult] = useState<TestRunResult | null>(null);
  const [showConsole, setShowConsole] = useState(false);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    setLedger(null);
    setError("");
    fetch(`${BASE}/itut-ledger.json`, { signal: controller.signal, cache: "no-store" })
      .then(async (response) => {
        if (!response.ok) throw Error("验收报告暂不可用");
        const data = (await response.json()) as Ledger;
        if (
          data.standard_compliance !== false ||
          data.formal_test_status !== "not_measured" ||
          !Array.isArray(data.items)
        )
          throw Error("验收报告格式与仿真口径不匹配");
        if (!controller.signal.aborted) setLedger(data);
      })
      .catch((reason) => {
        if (!controller.signal.aborted)
          setError(reason instanceof Error ? reason.message : "报告读取失败");
      });
    return () => controller.abort();
  }, [refresh]);

  const handleRunTest = async () => {
    setIsRunning(true);
    setRunProgress(15);
    setLastRunResult(null);
    setShowConsole(false);

    const ticker = setInterval(() => {
      setRunProgress((progress) => Math.min(85, progress + 3));
    }, 300);

    try {
      const resp = await fetch("/api/alignment/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pipeline, scope: runScope, strict: strictGate }),
      });

      clearInterval(ticker);
      setRunProgress(100);

      if (!resp.ok) {
        throw new Error(`测试执行服务响应错误 (HTTP ${resp.status})`);
      }

      const result = (await resp.json()) as TestRunResult;
      setLastRunResult(result);
      setRefresh((n) => n + 1);
    } catch (err) {
      clearInterval(ticker);
      setRunProgress(100);
      setLastRunResult({
        success: false,
        returncode: 1,
        duration_s: 0,
        total_count: 0,
        passed_count: 0,
        failed_count: 0,
        error_count: 1,
        skipped_count: 0,
        pass_rate: 0,
        engineering_success: false,
        standard_compliance: false,
        formal_test_status: "not_measured",
        failure_reasons: [err instanceof Error ? err.message : "连接失败"],
        stages: [
          {
            id: 1,
            name: "调用测试服务",
            detail: err instanceof Error ? err.message : "连接失败",
            status: "failed",
            duration_ms: 0,
          },
        ],
        log: err instanceof Error ? err.stack || err.message : "未知错误",
        scope: runScope,
        strict: strictGate,
        timestamp: new Date().toISOString(),
      });
    } finally {
      setIsRunning(false);
    }
  };

  const copyLog = () => {
    if (lastRunResult?.log) {
      navigator.clipboard.writeText(lastRunResult.log);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  if (!ledger) {
    return (
      <div className="alignment-page" role={error ? "alert" : "status"}>
        <p>{error || "正在读取验收报告…"}</p>
        {error ? (
          <button className="button secondary" onClick={() => setRefresh((n) => n + 1)}>
            <RotateCcw size={16} />
            重新加载
          </button>
        ) : null}
      </div>
    );
  }

  const items = ledger.items.filter(
    (item) =>
      (family === "all" || item.family === family) &&
      (bandwidth === "all" || item.bandwidth === bandwidth) &&
      `${item.id} ${item.test_name}`.toLowerCase().includes(query.toLowerCase())
  );
  const pageIndex = Math.min(page, Math.max(0, Math.ceil(items.length / PAGE_SIZE) - 1));
  const visibleItems = items.slice(pageIndex * PAGE_SIZE, (pageIndex + 1) * PAGE_SIZE);
  const runMetric = !lastRunResult
    ? "未运行"
    : lastRunResult.success
      ? `${lastRunResult.passed_count} / ${lastRunResult.total_count} PASS`
      : "未通过";
  const runMetricSub = !lastRunResult
    ? "点击执行后显示真实结果"
    : `失败 ${lastRunResult.failed_count} · 错误 ${lastRunResult.error_count} · 跳过 ${lastRunResult.skipped_count} · ${lastRunResult.duration_s}s`;
  const pipelineResult = lastRunResult?.pipeline;
  const hasUnifiedWeights =
    pipelineResult?.profile === "vcp_unified" &&
    pipelineResult.weights.length === 4 &&
    pipelineResult.weights.reduce((sum, weight) => sum + weight, 0) > 0;

  return (
    <div className="evaluation-page alignment-page">
      {/* Interactive Test Runner Console */}
      <section className="alignment-runner-card">
        <div className="alignment-runner-header">
          <div className="alignment-runner-title-group">
            <h2>
              <Activity size={20} color="#2563eb" />
              <span>ITU-T 声学工程自检控制台</span>
            </h2>
            <p>
              工程自检入口：真实 CLI 单场景处理、方法指标与契约门禁测试；不签发正式标准合规
            </p>
          </div>
          <div className="alignment-runner-badges">
            <span className="alignment-runner-badge" style={{ background: "#eff6ff", color: "#1d4ed8", borderColor: "#bfdbfe", fontWeight: 600 }}>
              工程自检 · 标准合规未测
            </span>
            <span className="alignment-runner-badge">ITU-T P.1100 / P.1110 / P.1120</span>
            <span className="alignment-runner-badge">NV200 336 项契约清单</span>
            <span className="alignment-runner-badge">
              {strictGate ? "严格模式：跳过项判失败" : "允许跳过项"}
            </span>
          </div>
        </div>

        {/* Live Metrics Grid */}
        <div className="alignment-metrics-grid">
          <div className="alignment-metric-box">
            <span className="metric-label">正式标准测项</span>
            <strong className="metric-value">{ledger.items.length} 项</strong>
            <span className="metric-sub">NV200 客户契约台账全覆盖</span>
          </div>
          <div className="alignment-metric-box pass">
            <span className="metric-label">工程流式任务</span>
            <strong className="metric-value">
              {ledger.engineering.passed_jobs} / {ledger.engineering.jobs} 通过
            </strong>
            <span className="metric-sub">
              {ledger.engineering.complete ? "全部已运行完成" : "流式场景已闭环"}
            </span>
          </div>
          <div className={`alignment-metric-box ${lastRunResult?.success ? "pass" : ""}`}>
            <span className="metric-label">工程自检状态</span>
            <strong className="metric-value">{runMetric}</strong>
            <span className="metric-sub">{runMetricSub}</span>
          </div>
          <div className="alignment-metric-box">
            <span className="metric-label">合规判定口径</span>
            <strong className="metric-value" style={{ fontSize: "1rem", color: "#2563eb" }}>
              真实证据链
            </strong>
            <span className="metric-sub">物理台架与算法仿真严格区分</span>
          </div>
        </div>

        {/* Controls & Execution Button */}
        <div className="alignment-runner-controls">
          <div className="alignment-scope-select">
            <label htmlFor="run-pipeline-select">车载语音增强流水线：</label>
            <select
              id="run-pipeline-select"
              className="alignment-select-input"
              value={pipeline}
              disabled={isRunning || runScope !== "full"}
              onChange={(e) => setPipeline(e.target.value as any)}
            >
              <option value="vcp_unified">⚡ VCP Unified · CLI profile: vcp_unified</option>
              <option value="vcp_joint">🔄 VCP Joint · CLI profile: vcp_joint</option>
              <option value="vcp_default">📦 VCP 传统标准基线 · CLI profile: vcp</option>
            </select>
          </div>

          <div className="alignment-scope-select">
            <label htmlFor="run-scope-select">测试执行范围：</label>
            <select
              id="run-scope-select"
              className="alignment-select-input"
              value={runScope}
              disabled={isRunning}
              onChange={(e) => setRunScope(e.target.value as any)}
            >
              <option value="full">⚡ 工程全检：真实 CLI 单场景 + 三组 pytest</option>
              <option value="methods">🔍 方法指标自检：仅 pytest，不运行音频 CLI</option>
              <option value="contract">📋 契约与防遮掩自检：仅 pytest，不运行音频 CLI</option>
            </select>
          </div>

          <div className="alignment-options-group">
            <label className="alignment-checkbox-label">
              <input
                type="checkbox"
                checked={strictGate}
                disabled={isRunning}
                onChange={(e) => setStrictGate(e.target.checked)}
              />
              <span>严格模式：所选测试出现跳过项即判失败</span>
            </label>

            {/* Main One-Click Execution Button */}
            <button
              className="alignment-run-btn"
              disabled={isRunning}
              onClick={handleRunTest}
              title="点击立即执行工程自检；结果不会升级为正式标准合规"
            >
              {isRunning ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  <span>正在执行测试...</span>
                </>
              ) : (
                <>
                  <Play size={16} fill="currentColor" />
                  <span>一键执行测试</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Execution Progress & Stepper HUD */}
        {(isRunning || lastRunResult) && (
          <div className="alignment-progress-hud">
            <div className="alignment-progress-bar-bg">
              <div
                className="alignment-progress-bar-fill"
                style={{ width: `${runProgress}%` }}
              />
            </div>

            {!lastRunResult ? (
              <p className="alignment-note">
                正在执行所选范围；完成后展示由 CLI、JUnit 与输入校验生成的真实阶段结果。
              </p>
            ) : (
              <div className="alignment-stages-grid">
              {lastRunResult.stages.map((stage) => {
                const isPassed = stage.status === "passed";
                const isRunningStage = stage.status === "running";
                const isFailed = stage.status === "failed";
                const isSkipped = stage.status === "skipped";
                return (
                  <div
                    key={stage.id}
                    className={`alignment-stage-card ${stage.status}`}
                  >
                    <div className="alignment-stage-icon">
                      {isPassed && <CheckCircle2 size={16} color="#16a34a" />}
                      {isRunningStage && <Loader2 size={16} color="#2563eb" className="animate-spin" />}
                      {isFailed && <XCircle size={16} color="#dc2626" />}
                      {isSkipped && <MinusCircle size={16} color="#64748b" />}
                      {!isPassed && !isRunningStage && !isFailed && !isSkipped && (
                        <div style={{ width: 16, height: 16, borderRadius: "50%", border: "2px solid #cbd5e1" }} />
                      )}
                    </div>
                    <div className="alignment-stage-text">
                      <strong>{stage.name}</strong>
                      <span>{stage.detail}</span>
                    </div>
                  </div>
                );
              })}
              </div>
            )}

            {/* Real CLI diagnostics are rendered only when the full scope ran the pipeline. */}
            {lastRunResult && pipelineResult && (
              <div className="vcp-diagnostics-panel">
                <div className="vcp-panel-header">
                  <div className="vcp-panel-title">
                    <Activity size={18} color="#2563eb" />
                    <span>{pipelineResult.name} 真实音频处理诊断</span>
                    <span className="vcp-topology-pill">
                      {pipelineResult.profile ?? "未生成 profile"} ·{" "}
                      {pipelineResult.status === "passed" ? "处理完成" : "处理失败"}
                    </span>
                  </div>
                  <span style={{ fontSize: "0.75rem", color: "#64748b" }}>
                    拓扑：{pipelineResult.topology}
                  </span>
                </div>

                <div className="vcp-metrics-grid">
                  <div className="vcp-metric-card">
                    <span className="vcp-metric-label">算法处理时延 (DSP Delay)</span>
                    <div className="vcp-metric-val">
                      {typeof pipelineResult.dsp_delay_ms === "number"
                        ? `${pipelineResult.dsp_delay_ms.toFixed(3)} ms`
                        : "不可用"}
                      <span className="vcp-metric-sub">
                        {typeof pipelineResult.dsp_delay_ms === "number"
                          ? pipelineResult.dsp_delay_ms <= 275
                            ? "当前工程门限 ≤275 ms"
                            : "超过当前工程门限"
                          : "CLI 未返回"}
                      </span>
                    </div>
                    <span style={{ fontSize: "0.6875rem", color: "#64748b" }}>
                      {typeof pipelineResult.dsp_delay_ms === "number"
                        ? `距当前工程门限 ${(275 - pipelineResult.dsp_delay_ms).toFixed(3)} ms`
                        : "不以缺省值补造结果"}
                    </span>
                  </div>

                  <div className="vcp-metric-card">
                    <span className="vcp-metric-label">实时处理倍率 (RTF)</span>
                    <div className="vcp-metric-val">
                      {typeof pipelineResult.rtf === "number"
                        ? `${pipelineResult.rtf.toFixed(4)}x`
                        : "不可用"}
                      <span className="vcp-metric-sub">
                        {typeof pipelineResult.rtf === "number"
                          ? `录音时长倍率 · 约 ${(pipelineResult.rtf * 100).toFixed(1)}%`
                          : "CLI 未返回"}
                      </span>
                    </div>
                    <span style={{ fontSize: "0.6875rem", color: "#64748b" }}>
                      固定输入上的工程诊断值，不是跨硬件性能承诺
                    </span>
                  </div>

                  <div className="vcp-metric-card">
                    <span className="vcp-metric-label">实际处理后端</span>
                    <div className="vcp-metric-val" style={{ fontSize: "0.9375rem" }}>
                      AEC: {pipelineResult.aec_backend ?? "不可用"}
                    </div>
                    <span style={{ fontSize: "0.6875rem", color: "#64748b" }}>
                      NS: {pipelineResult.ns_backend ?? "不可用"} · 不直接证明双讲标准等级
                    </span>
                  </div>

                  <div className="vcp-metric-card">
                    <span className="vcp-metric-label">动态范围控制 (DRC Limiter)</span>
                    <div className="vcp-metric-val" style={{ fontSize: "0.9375rem" }}>
                      {pipelineResult.tx_drc === true
                        ? `已激活${
                            typeof pipelineResult.tx_drc_config?.threshold_dbfs === "number"
                              ? ` (${pipelineResult.tx_drc_config.threshold_dbfs} dBFS / ${
                                  pipelineResult.tx_drc_config.ratio ?? "?"
                                }:1)`
                              : ""
                          }`
                        : pipelineResult.tx_drc === false
                          ? "未启用"
                          : "不可用"}
                    </div>
                    <span style={{ fontSize: "0.6875rem", color: "#64748b" }}>
                      DRC 配置来自 CLI 报告，不补造默认门限
                    </span>
                  </div>
                </div>

                {hasUnifiedWeights && (
                  <div className="vcp-expert-section">
                    <div className="vcp-expert-title">
                      <span>Unified NS 四专家动态融合权重分布 (Four-Expert Mixture)</span>
                      <span style={{ color: "#2563eb", fontWeight: 700 }}>
                        CLI 输出的本次固定样本权重
                      </span>
                    </div>
                    <div className="vcp-expert-bar-wrapper">
                      <div
                        className="vcp-expert-segment"
                        style={{
                          width: `${(pipelineResult.weights[0] * 100).toFixed(1)}%`,
                          background: "#3b82f6",
                        }}
                        title={`平稳噪声专家: ${(pipelineResult.weights[0] * 100).toFixed(1)}%`}
                      />
                      <div
                        className="vcp-expert-segment"
                        style={{
                          width: `${(pipelineResult.weights[1] * 100).toFixed(1)}%`,
                          background: "#10b981",
                        }}
                        title={`瞬态撞击专家: ${(pipelineResult.weights[1] * 100).toFixed(1)}%`}
                      />
                      <div
                        className="vcp-expert-segment"
                        style={{
                          width: `${(pipelineResult.weights[2] * 100).toFixed(1)}%`,
                          background: "#f59e0b",
                        }}
                        title={`单音共振专家: ${(pipelineResult.weights[2] * 100).toFixed(1)}%`}
                      />
                      <div
                        className="vcp-expert-segment"
                        style={{
                          width: `${(pipelineResult.weights[3] * 100).toFixed(1)}%`,
                          background: "#8b5cf6",
                        }}
                        title={`残余底噪专家: ${(pipelineResult.weights[3] * 100).toFixed(1)}%`}
                      />
                    </div>
                    <div className="vcp-expert-legend">
                      <div className="vcp-legend-item">
                        <span className="vcp-legend-dot" style={{ background: "#3b82f6" }} />
                        <span>平稳噪声专家 (Stationary): {(pipelineResult.weights[0] * 100).toFixed(1)}%</span>
                      </div>
                      <div className="vcp-legend-item">
                        <span className="vcp-legend-dot" style={{ background: "#10b981" }} />
                        <span>瞬态撞击专家 (Transient): {(pipelineResult.weights[1] * 100).toFixed(1)}%</span>
                      </div>
                      <div className="vcp-legend-item">
                        <span className="vcp-legend-dot" style={{ background: "#f59e0b" }} />
                        <span>单音共振专家 (Tonality): {(pipelineResult.weights[2] * 100).toFixed(1)}%</span>
                      </div>
                      <div className="vcp-legend-item">
                        <span className="vcp-legend-dot" style={{ background: "#8b5cf6" }} />
                        <span>残余底噪专家 (Residual): {(pipelineResult.weights[3] * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Test Conclusion & Terminal Toggle */}
            {lastRunResult && (
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  marginTop: "14px",
                  padding: "10px 14px",
                  borderRadius: "8px",
                  background: lastRunResult.success ? "#f0fdf4" : "#fef2f2",
                  border: `1px solid ${lastRunResult.success ? "#bbf7d0" : "#fecaca"}`,
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                  {lastRunResult.success ? (
                    <ShieldCheck size={20} color="#16a34a" />
                  ) : (
                    <AlertCircle size={20} color="#dc2626" />
                  )}
                  <div>
                    <strong style={{ color: lastRunResult.success ? "#166534" : "#991b1b", fontSize: "0.875rem" }}>
                      {lastRunResult.success
                        ? `工程自检通过：${lastRunResult.passed_count} / ${lastRunResult.total_count} 项 pytest 通过 (${lastRunResult.pass_rate}%)`
                        : `工程自检未通过：${
                            lastRunResult.failure_reasons[0] ??
                            `${lastRunResult.failed_count} 项失败，${lastRunResult.error_count} 项错误`
                          }`}
                    </strong>
                    <span style={{ display: "block", fontSize: "0.75rem", color: "#64748b" }}>
                      失败 {lastRunResult.failed_count} · 错误 {lastRunResult.error_count} · 跳过{" "}
                      {lastRunResult.skipped_count} · 耗时 {lastRunResult.duration_s}s · 范围{" "}
                      {lastRunResult.scope} · 标准合规：未测
                    </span>
                  </div>
                </div>

                <button
                  className="button secondary"
                  style={{ fontSize: "0.75rem", display: "inline-flex", alignItems: "center", gap: "6px" }}
                  onClick={() => setShowConsole(!showConsole)}
                >
                  <Terminal size={14} />
                  <span>{showConsole ? "收起执行日志" : "查看控制台日志"}</span>
                  {showConsole ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                </button>
              </div>
            )}

            {/* Collapsible Execution Console */}
            {showConsole && lastRunResult && (
              <div className="alignment-terminal-wrapper">
                <div className="alignment-terminal-header">
                  <div className="alignment-terminal-dots">
                    <span className="alignment-terminal-dot red" />
                    <span className="alignment-terminal-dot yellow" />
                    <span className="alignment-terminal-dot green" />
                  </div>
                  <span>engineering self-check output ({lastRunResult.duration_s}s)</span>
                  <button
                    onClick={copyLog}
                    style={{
                      background: "transparent",
                      border: 0,
                      color: "#94a3b8",
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      gap: "4px",
                      fontSize: "0.6875rem",
                    }}
                  >
                    {copied ? <Check size={12} color="#10b981" /> : null}
                    <span>{copied ? "已复制" : "复制日志"}</span>
                  </button>
                </div>
                <pre className="alignment-terminal-body">{lastRunResult.log}</pre>
              </div>
            )}
          </div>
        )}
      </section>

      {/* Notice & Download Links */}
      <p className="alignment-note">
        一键执行只做工程自检：全量范围运行一个固定车辆场景样本和测试套件，方法/契约范围只运行
        pytest。它不会执行 8 模式或 336 项正式声学实测；SI-SDR、STOI、ERLE 与 dBFS
        也不能替代正式测项所需的声学标定、测试方法与规定信号。
      </p>
      <div className="alignment-links">
        <a href={`${BASE}/itut-ledger.json`} download>
          <Download size={15} />
          完整验收 JSON
        </a>
        <a href={`${BASE}/itut-ledger.md`} download>
          <Download size={15} />
          证据清单
        </a>
        <a href={`${BASE}/summary.json`} download>
          <Download size={15} />
          矩阵报告
        </a>
      </div>

      {/* Section Tabs */}
      <div className="alignment-sections" role="group" aria-label="验收分区">
        {SECTIONS.map((item) => (
          <button
            className="button secondary"
            key={item.id}
            type="button"
            aria-pressed={section === item.id}
            aria-controls="alignment-panel"
            onClick={() => setSection(item.id)}
          >
            {item.label}
          </button>
        ))}
      </div>

      {/* Panels */}
      <div
        id="alignment-panel"
        role="region"
        aria-label={SECTIONS.find((item) => item.id === section)?.label}
      >
        <Suspense fallback={<p role="status">正在读取分区…</p>}>
          {section === "engineering" ? (
            <section className="evaluation-section" aria-label="车载流式矩阵">
              <h2>车载流式矩阵</h2>
              <p className="alignment-note">
                16 / 24 / 48 kHz，1 / 2 / 4 路麦克风；每条连续{" "}
                {ledger.simulation_identity.seconds_per_sequence ?? 259}{" "}
                秒，阶段之间不重置。
                {ledger.simulation_identity.paced
                  ? "本批按 10 ms 节拍运行。"
                  : "本批验证音质，实时期限另测。"}
                高采样率输入由 16 kHz 素材转换，尚未验证原生高频带。
              </p>
              <div className="table-scroll">
                <table className="audio-table">
                  <thead>
                    <tr>
                      <th>任务</th>
                      <th>音质阶段通过</th>
                      <th>包含耗时阶段通过</th>
                      <th>工程判定</th>
                      <th>未通过项</th>
                    </tr>
                  </thead>
                  <tbody>
                    {ledger.engineering.results.map((job) => (
                      <tr key={job.name}>
                        <td>{job.name}</td>
                        <td>
                          {job.phases
                            ? `${job.quality_passed_phases ?? "未测"} / ${job.phases}`
                            : "成对时钟对照"}
                        </td>
                        <td>{job.phases ? `${job.passed_phases} / ${job.phases}` : "—"}</td>
                        <td
                          className={
                            job.acceptance_passed
                              ? "evaluation-status-pass"
                              : "evaluation-status-fail"
                          }
                        >
                          {job.acceptance_passed ? "通过" : "未通过"}
                        </td>
                        <td>{job.failures?.length ?? "报告缺失"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {ledger.engineering.results
                .filter((job) => job.failures?.length)
                .map((job) => (
                  <JobFailures key={job.name} job={job} />
                ))}
              {ledger.paced_diagnostic ? (
                <p className="alignment-note">
                  单独 10 ms 节拍复测：{ledger.paced_diagnostic.sample_rate / 1000} kHz /{" "}
                  {ledger.paced_diagnostic.microphones} 麦 / {ledger.paced_diagnostic.seconds} s；P95{" "}
                  {ledger.paced_diagnostic.process_p95_ms.toFixed(3)} ms，最大{" "}
                  {ledger.paced_diagnostic.process_max_ms.toFixed(3)} ms，完成期限错过{" "}
                  {ledger.paced_diagnostic.completion_deadline_misses} 帧；包含全部工程门限{" "}
                  {ledger.paced_diagnostic.passed_phases} / {ledger.paced_diagnostic.phases} 阶段通过。
                </p>
              ) : null}
            </section>
          ) : null}

          {section === "dataset" ? <ActiveDataset /> : null}
          {section === "frontend" ? <VehicleFrontend /> : null}

          {section === "standards" ? (
            <section className="evaluation-section" aria-label="标准与自适应边界">
              <h2>标准与自适应边界</h2>
              <div className="alignment-links">
                {Object.entries(ledger.standards).map(([key, standard]) => (
                  <a key={key} href={standard.url} target="_blank" rel="noreferrer">
                    {standard.recommendation} {standard.edition}
                  </a>
                ))}
              </div>
              <p className="alignment-note">
                NS、AEC、AGC 与模拟时钟校正使用实际 PCM 和状态自适应。BF 使用固定标定几何；EQ 根据实测频响离线拟合，默认关闭。业务 NB / WB / SWB 与 PCM 采样率分别定义。24 kHz 无法覆盖 P.1120 的 14 kHz SWB 测量带宽；48 kHz 也仍需原生高频素材。
              </p>
              <p className="alignment-note">
                P.1120 已登记正式发送/接收频响掩模，以及目标车辆怠速、城市 60 km/h、高速 120 km/h 与 HVAC 条件。现有 TCAR 合成场景尚无这些实录条件证据；标准要求的预适应测量与启动压力测试分别保留。
              </p>
              <details className="alignment-failures">
                <summary>当前二进制与工程规则</summary>
                <dl className="evaluation-identities">
                  <dt>CLI SHA256</dt>
                  <dd>{ledger.simulation_identity.cli_sha256}</dd>
                  <dt>工程阈值 SHA256</dt>
                  <dd>{ledger.simulation_identity.rules_sha256}</dd>
                </dl>
              </details>
            </section>
          ) : null}

          {section === "formal" ? (
            <section className="evaluation-section" aria-label="正式测试证据清单">
              <h2>正式测试证据清单 (336项)</h2>
              <p className="alignment-note">
                原表条件与限值保留，尚需逐条核对现行标准。历史设备结果不作为当前引擎实测值。
              </p>
              <div className="alignment-filters">
                <label>
                  测项类别
                  <select
                    value={family}
                    onChange={(event) => {
                      setFamily(event.target.value);
                      setPage(0);
                    }}
                  >
                    <option value="all">
                      全部 {Object.keys(ledger.family_counts).length} 类
                    </option>
                    {Object.keys(ledger.family_counts).map((key) => (
                      <option value={key} key={key}>
                        {FAMILY[key]?.[0] ?? key}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  业务带宽
                  <select
                    value={bandwidth}
                    onChange={(event) => {
                      setBandwidth(event.target.value);
                      setPage(0);
                    }}
                  >
                    <option value="all">全部带宽</option>
                    {[...new Set(ledger.items.map((item) => item.bandwidth))].map((key) => (
                      <option key={key}>{key}</option>
                    ))}
                  </select>
                </label>
                <label>
                  测项搜索
                  <input
                    type="search"
                    placeholder="按用例ID、条款或关键字搜索..."
                    value={query}
                    onChange={(event) => {
                      setQuery(event.target.value);
                      setPage(0);
                    }}
                  />
                </label>
              </div>
              <div className="alignment-pagination" role="group" aria-label="正式测项分页">
                <p role="status" className="alignment-note">
                  {items.length} 个测项
                  {items.length
                    ? ` · 第 ${pageIndex * PAGE_SIZE + 1}–${Math.min(
                        (pageIndex + 1) * PAGE_SIZE,
                        items.length
                      )} 项 · ${pageIndex + 1} / ${Math.ceil(items.length / PAGE_SIZE)} 页`
                    : ""}
                </p>
                <button
                  className="button secondary"
                  disabled={pageIndex === 0}
                  onClick={() => setPage(pageIndex - 1)}
                >
                  上一页
                </button>
                <button
                  className="button secondary"
                  disabled={(pageIndex + 1) * PAGE_SIZE >= items.length}
                  onClick={() => setPage(pageIndex + 1)}
                >
                  下一页
                </button>
              </div>
              {items.length ? (
                <div className="table-scroll">
                  <table className="audio-table">
                    <thead>
                      <tr>
                        <th>原表位置 / 测项</th>
                        <th>客户表限值</th>
                        <th>当前状态</th>
                        <th>所缺证据</th>
                      </tr>
                    </thead>
                    <tbody>
                      {visibleItems.map((item) => (
                        <tr key={item.id}>
                          <td>
                            <small>
                              {item.sheet} · {item.source_cell}
                            </small>
                            <br />
                            <strong>{item.test_name}</strong>
                          </td>
                          <td>
                            <code>
                              {value(item.lower_limit)} 至 {value(item.upper_limit)}
                            </code>
                            <br />
                            <small style={{ color: "#64748b" }}>{item.unit}</small>
                          </td>
                          <td>
                            <span
                              style={{
                                display: "inline-block",
                                padding: "2px 6px",
                                borderRadius: "4px",
                                fontSize: "0.6875rem",
                                background: "#f1f5f9",
                                color: "#64748b",
                                fontWeight: 600,
                              }}
                            >
                              未测
                            </span>
                          </td>
                          <td style={{ fontSize: "0.8125rem", color: "#475569" }}>
                            {FAMILY[item.family]?.[1] ?? item.family}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="evaluation-empty">
                  <p>没有匹配测项。</p>
                  <button
                    className="button secondary"
                    onClick={() => {
                      setFamily("all");
                      setBandwidth("all");
                      setQuery("");
                      setPage(0);
                    }}
                  >
                    清除筛选
                  </button>
                </div>
              )}
            </section>
          ) : null}
        </Suspense>
      </div>
    </div>
  );
}
