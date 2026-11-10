import { useCallback, useEffect, useId, useRef, useState } from "react";
import {
  ArrowRight,
  Bot,
  CircleAlert,
  Loader2,
  RefreshCw,
  Sparkles,
  X,
} from "lucide-react";
import {
  FIVE_SIGNAL_CONTRACT_DIRECTION_LABELS,
  contractGroupsForMode,
  FIVE_SIGNAL_ROLES,
  FIVE_SIGNAL_ROLE_LABELS,
  parseFiveSignalDetail,
  analysisAvailability,
  finiteValue,
  METRIC_VALUE_KEYS,
  type FiveSignalDetail,
  FIVE_SIGNAL_MODES,
  deriveFiveSignalFlowSnapshot,
  type FiveSignalContractGroup,
  type FiveSignalMode,
  type FiveSignalFlowSnapshot,
} from "./five-signal-flow-model";
import { MODE_METAS } from "../domain/nv200-table";
import { chainPresets, chainStages, optionFor } from "./enhancement-chain-model";
import { FIVE_SIGNAL_FLOW_LAYOUT as FLOW } from "./five-signal-flow-layout";
import type { ReactNode } from "react";
import type { AcceptanceBatchSnapshot } from "./AlignmentAcceptanceBatch";

const EVALUATIONS_API = "/api/acoustic/five-signal/evaluations?limit=100";
const POLL_INTERVAL_MS = 5_000;

const EMPTY_SNAPSHOT = deriveFiveSignalFlowSnapshot([]);

type VcpStage = {
  readonly tag: string;
  readonly code: string;
  readonly title: string;
  readonly line1: string;
  readonly line2: string;
  readonly chip: string;
  readonly variant?: "acoustic";
};

const VCP_UPLINK_STAGES: readonly VcpStage[] = [
  {
    tag: "VCP UPLINK",
    code: "BF",
    title: "波束形成",
    line1: "多麦同步 · 空间滤波",
    line2: "定向拾音输出",
    chip: "阵列前端",
  },
  {
    tag: "VCP UPLINK",
    code: "AEC",
    title: "回声消除",
    line1: "远端播放信号自适应 · 双讲保护",
    line2: "消除扬声器回声",
    chip: "回声控制",
  },
  {
    tag: "VCP UPLINK",
    code: "NS",
    title: "语音降噪",
    line1: "稳态 / 瞬态噪声",
    line2: "保留语音细节",
    chip: "语音保真",
  },
  {
    tag: "VCP UPLINK",
    code: "EQ",
    title: "频响均衡",
    line1: "车舱频响补偿",
    line2: "清晰度校准",
    chip: "频响校准",
  },
  {
    tag: "VCP UPLINK",
    code: "AGC/DRC",
    title: "增益与动态",
    line1: "电平归一 · 满幅保护",
    line2: "动态范围控制",
    chip: "输出稳定",
  },
] as const;

const VCP_DOWNLINK_STAGES: readonly VcpStage[] = [
  {
    tag: "VCP DOWNLINK",
    code: "RX NR",
    title: "下行降噪",
    line1: "远端背景噪声净化",
    line2: "提升车内可懂度",
    chip: "远端清晰",
  },
  {
    tag: "VCP DOWNLINK",
    code: "RX EQ",
    title: "下行均衡",
    line1: "扬声器频响校准",
    line2: "车舱声学补偿",
    chip: "播放校准",
  },
  {
    tag: "VCP DOWNLINK",
    code: "RX DRC",
    title: "下行保护",
    line1: "削波失真防护",
    line2: "提供远端播放信号 / 回声参考",
    chip: "满幅保护",
  },
  {
    tag: "ACOUSTIC",
    code: "SPK",
    title: "扬声器 / 回声",
    line1: "功放播放 · 声学耦合",
    line2: "回声进入麦克风混合输入",
    chip: "物理回路",
    variant: "acoustic",
  },
];

const VCP_STAGE_WIDTH = FLOW.vcpCardWidth;
const VCP_STAGE_HEIGHT = FLOW.vcpCardHeight;
const VCP_STAGE_STEP = FLOW.vcpCardWidth + FLOW.vcpCardGap;
const CONTRACT_COLUMN_WIDTH = 200;
const CONTRACT_NODE_TOP = 504;
const CONTRACT_NODE_HEIGHT = 84;
const CONTRACT_NODE_STEP = 100;

function FlowGroup({
  x,
  y,
  width,
  height,
  tag,
  title,
  status,
  state,
  name,
  children,
}: {
  x: number;
  y: number;
  width: number;
  height: number;
  tag: string;
  title: string;
  status: string;
  state: string;
  name: string;
  children?: ReactNode;
}) {
  return (
    <g
      className={`five-signal-flow-analysis-group state-${state}`}
      transform={`translate(${x} ${y})`}
      data-flow-stage={name}
      role="group"
      aria-label={`${title}，${status}`}
    >
      <rect
        className="five-signal-flow-analysis-box"
        width={width}
        height={height}
        rx="14"
      />
      <text className="five-signal-flow-analysis-kicker" x="20" y="28">
        {tag}
      </text>
      <text className="five-signal-flow-analysis-title" x="20" y="58">
        {title}
      </text>
      <text
        className="five-signal-flow-analysis-status"
        x={width - 20}
        y="28"
        textAnchor="end"
      >
        {status}
      </text>
      <line
        className="five-signal-flow-analysis-rule"
        x1="20"
        y1="76"
        x2={width - 20}
        y2="76"
      />
      {children}
    </g>
  );
}

function FlowCard({
  x,
  y,
  width,
  height,
  tag,
  title,
  subtitle,
  state,
  name,
  children,
}: {
  x: number;
  y: number;
  width: number;
  height: number;
  tag: string;
  title: string;
  subtitle: string;
  state: string;
  name: string;
  children: ReactNode;
}) {
  return (
    <g
      className={`five-signal-flow-node state-${state}`}
      transform={`translate(${x} ${y})`}
      data-flow-stage={name}
      role="group"
      aria-label={title}
    >
      <rect
        className="five-signal-flow-node-box"
        width={width}
        height={height}
        rx="14"
      />
      <text className="five-signal-flow-node-tag" x="24" y="30">
        {tag}
      </text>
      <text className="five-signal-flow-node-title" x="24" y="68">
        {title}
      </text>
      <text className="five-signal-flow-node-copy" x="24" y="94">
        {subtitle}
      </text>
      {children}
    </g>
  );
}

function VcpStageCard({
  x,
  y,
  stage,
  state,
  label,
}: {
  x: number;
  y: number;
  stage: VcpStage;
  state: string;
  label: string;
}) {
  const contentX = x + 18;
  const contentEndX = x + VCP_STAGE_WIDTH - 18;
  return (
    <g
      className={`five-signal-flow-node five-signal-flow-vcp state-${state}${stage.variant === "acoustic" ? " is-acoustic" : ""}`}
      role="group"
      aria-label={`${label}：${stage.code} ${stage.title}，${stage.line1}，${stage.line2}`}
    >
      <rect
        className={`five-signal-flow-node-box is-vcp${stage.variant === "acoustic" ? " is-acoustic" : ""}`}
        x={x}
        y={y}
        width={VCP_STAGE_WIDTH}
        height={VCP_STAGE_HEIGHT}
        data-flow-node="vcp"
        rx="16"
      />
      <text className="five-signal-flow-node-tag" x={contentX} y={y + 26}>
        {stage.tag}
      </text>
      <text
        className="five-signal-flow-node-index"
        x={contentEndX}
        y={y + 26}
        textAnchor="end"
      >
        {stage.code}
      </text>
      <text
        className="five-signal-flow-vcp-stage-title"
        x={contentX}
        y={y + 60}
      >
        {stage.title}
      </text>
      <text className="five-signal-flow-vcp-stage-copy" x={contentX} y={y + 88}>
        {stage.line1}
      </text>
      <text
        className="five-signal-flow-vcp-stage-copy"
        x={contentX}
        y={y + 110}
      >
        {stage.line2}
      </text>
      <line
        className="five-signal-flow-node-rule"
        x1={contentX}
        y1={y + 134}
        x2={contentEndX}
        y2={y + 134}
      />
      <rect
        className="five-signal-flow-vcp-stage-chip"
        x={contentX}
        y={y + 152}
        width={VCP_STAGE_WIDTH - 36}
        height="28"
        rx="6"
      />
      <text
        className="five-signal-flow-vcp-stage-chip-label"
        x={x + VCP_STAGE_WIDTH / 2}
        y={y + 170}
        textAnchor="middle"
      >
        {stage.chip}
      </text>
    </g>
  );
}

function ContractDirectionColumn({
  x,
  label,
  total,
  groups,
}: {
  x: number;
  label: string;
  total: number;
  groups: readonly FiveSignalContractGroup[];
}) {
  const childX = x + 16;
  const childWidth = CONTRACT_COLUMN_WIDTH - 16;
  return (
    <g role="group" aria-label={`${label}方向，共 ${total} 项`}>
      <text
        className="five-signal-flow-mapping-direction-label"
        x={x + 14}
        y="473"
      >
        {label}
      </text>
      <text
        className="five-signal-flow-mapping-direction-total"
        x={x + CONTRACT_COLUMN_WIDTH - 14}
        y="473"
        textAnchor="end"
      >
        {total} 项
      </text>
      <path
        className="five-signal-flow-mapping-branch"
        d={`M ${x + 4} 482 V ${CONTRACT_NODE_TOP + (groups.length - 1) * CONTRACT_NODE_STEP + CONTRACT_NODE_HEIGHT / 2}`}
      />
      {groups.map((group, index) => {
        const y = CONTRACT_NODE_TOP + index * CONTRACT_NODE_STEP;
        return (
          <g
            role="group"
            aria-label={`${group.label}，${group.count} 项。${group.detail}`}
            key={group.label}
          >
            <path
              className="five-signal-flow-mapping-branch"
              d={`M ${x + 4} ${y + CONTRACT_NODE_HEIGHT / 2} H ${childX}`}
            />
            <rect
              className="five-signal-flow-mapping-group-node"
              x={childX}
              y={y}
              width={childWidth}
              height={CONTRACT_NODE_HEIGHT}
              rx="8"
              data-flow-node="contract"
            />
            <text
              className="five-signal-flow-mapping-group-label"
              x={childX + 14}
              y={y + 30}
            >
              {group.label}
            </text>
            <text
              className="five-signal-flow-mapping-group-count"
              x={childX + 14}
              y={y + 62}
            >
              {group.count} 项
            </text>
          </g>
        );
      })}
    </g>
  );
}

const ANALYSIS_METRICS = [
  {
    tag: "SDR",
    title: "SI-SDR",
    line1: "输入 / 输出",
    line2: "SI-SDR 变化量",
  },
  {
    tag: "STOI",
    title: "STOI",
    line1: "输入 / 输出",
    line2: "STOI 变化量",
  },
  {
    tag: "LEVEL",
    title: "电平与 SNR",
    line1: "发送增益 · 噪声电平",
    line2: "输入 / 输出 SNR",
  },
  {
    tag: "SPECTRUM",
    title: "频响与频谱",
    line1: "响应余量 · 频响平坦度",
    line2: "谱峰 · 噪声功率变化",
  },
  {
    tag: "ECHO",
    title: "回声损耗",
    line1: "工程 TCL 近似",
    line2: "回声降低 dB",
  },
  {
    tag: "DOUBLE TALK",
    title: "双讲功率比",
    line1: "双讲区段输出 / 输入",
    line2: "功率比 · 非逐帧透传率",
  },
] as const;

const METRIC_ROW_CENTERS = ANALYSIS_METRICS.map(
  (_, index) =>
    FLOW.metricTop + index * FLOW.metricStep + FLOW.metricHeight / 2,
);

const MODE_LABELS: Record<FiveSignalMode, string> = {
  "NB-wired": "NB 有线",
  "NB-wireless": "NB 无线",
  "NB-bt": "NB 蓝牙",
  "WB-wired": "WB 有线",
  "WB-wireless": "WB 无线",
  "WB-bt": "WB 蓝牙",
  "SWB-wired": "SWB 有线",
  "SWB-wireless": "SWB 无线",
};

const MODE_ROW_CENTERS = FIVE_SIGNAL_MODES.map(
  (_, index) => FLOW.modeTop + index * FLOW.modeStep + FLOW.modeHeight / 2,
);

type OutcomeKey = keyof FiveSignalFlowSnapshot["counts"];
type AgentOutcomeKey = Exclude<OutcomeKey, "pass">;

const OUTCOME_ROWS: ReadonlyArray<{
  key: AgentOutcomeKey;
  title: string;
  detail: string;
  dotClass: string;
  agentTitle: string;
  agentKicker: string;
}> = [
  {
    key: "fail",
    title: "失败",
    detail: "建议阻断，定位原因",
    dotClass: "is-fail",
    agentTitle: "失败根因 Agent",
    agentKicker: "ROOT CAUSE",
  },
  {
    key: "review_required",
    title: "需复核",
    detail: "人工确认证据",
    dotClass: "is-review",
    agentTitle: "复核证据 Agent",
    agentKicker: "EVIDENCE",
  },
  {
    key: "not_measured",
    title: "不可测",
    detail: "补齐测试条件",
    dotClass: "is-unmeasured",
    agentTitle: "测试条件 Agent",
    agentKicker: "READINESS",
  },
  {
    key: "invalid",
    title: "无效",
    detail: "修正采集或数据",
    dotClass: "is-invalid",
    agentTitle: "数据修复 Agent",
    agentKicker: "DATA INTEGRITY",
  },
] as const;

const OUTCOME_ROW_CENTERS = OUTCOME_ROWS.map(
  (_, index) =>
    FLOW.outcomeTop + index * FLOW.outcomeStep + FLOW.outcomeHeight / 2,
);

type AgentAnalysis = {
  status: "ready" | "empty";
  outcome: AgentOutcomeKey;
  agent: string;
  thread_id: string | null;
  thread_created: boolean;
  model: string | null;
  generated_at: string;
  source_evaluation_ids: string[];
  fingerprint: string | null;
  mode_count: number;
  item_count: number;
  cached: boolean;
  summary: string;
  root_causes: Array<{
    cause: string;
    evidence_refs: string[];
    confidence: "high" | "medium" | "low";
  }>;
  actions: Array<{
    priority: "P0" | "P1" | "P2";
    action: string;
    validation: string;
    evidence_refs: string[];
  }>;
  limitations: string[];
};

type AgentState = {
  status: "idle" | "loading" | "ready" | "empty" | "error";
  result: AgentAnalysis | null;
  error: string;
};

const EMPTY_AGENT_STATE: AgentState = {
  status: "idle",
  result: null,
  error: "",
};

const EMPTY_AGENT_STATES: Record<AgentOutcomeKey, AgentState> = {
  fail: EMPTY_AGENT_STATE,
  review_required: EMPTY_AGENT_STATE,
  not_measured: EMPTY_AGENT_STATE,
  invalid: EMPTY_AGENT_STATE,
};

const formatSyncTime = (date: Date | null) =>
  date?.toLocaleTimeString("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }) ?? "";

function statusCopy(snapshot: FiveSignalFlowSnapshot) {
  if (snapshot.state === "complete") return "336 项条目齐全";
  if (snapshot.state === "partial") return "部分条目已生成";
  if (snapshot.state === "invalid") return "最近结果数据异常";
  return "等待五路评估";
}

function nodeState(
  snapshot: FiveSignalFlowSnapshot,
  stage: "input" | "mapping" | "aggregate" | "classify",
) {
  if (stage === "input") {
    return snapshot.availableModes ? "complete" : "pending";
  }
  if (stage === "aggregate" || stage === "classify") return snapshot.state;
  if (snapshot.state === "complete") return "complete";
  if (snapshot.state === "invalid") return "invalid";
  return snapshot.availableModes ? "partial" : "pending";
}

function modeRowState(snapshot: FiveSignalFlowSnapshot, mode: FiveSignalMode) {
  const latest = snapshot.latestByMode[mode];
  if (!latest) return "pending";
  return latest.valid ? "complete" : "invalid";
}

function evaluationIds(snapshot: FiveSignalFlowSnapshot) {
  return FIVE_SIGNAL_MODES.flatMap((mode) => {
    const latest = snapshot.latestByMode[mode];
    return latest?.valid && latest.evaluation_id ? [latest.evaluation_id] : [];
  });
}

function sameIds(left: readonly string[], right: readonly string[]) {
  if (left.length !== right.length) return false;
  const sortedLeft = [...left].sort();
  const sortedRight = [...right].sort();
  return sortedLeft.every((value, index) => value === sortedRight[index]);
}

function agentIsStale(state: AgentState, currentIds: readonly string[]) {
  return Boolean(
    state.result && !sameIds(state.result.source_evaluation_ids, currentIds),
  );
}

function agentStatusCopy(state: AgentState, count: number, stale: boolean) {
  if (state.status === "loading") return "分析中";
  if (state.status === "error") return "分析失败";
  if (stale) return "快照已变化";
  if (state.status === "empty") return "无此类结果";
  if (state.status === "ready") return "分析完成";
  return count > 0 ? "待分析" : "无此类结果";
}

function agentActionCopy(state: AgentState, count: number, stale: boolean) {
  if (state.status === "loading") return "正在续接 Codex 线程…";
  if (state.status === "error") return "点击重试";
  if (stale) return "点击分析最新快照";
  if (state.status === "ready" || state.status === "empty") {
    return "查看原因与方案";
  }
  return count > 0 ? "点击启动或续接线程" : "查看空态说明";
}

type FiveSignalFlowProps = {
  algorithmClusterId: string;
  onAlgorithmClusterChange: (id: string) => void;
  selectedMode: FiveSignalMode;
  onModeChange: (mode: FiveSignalMode) => void;
  isRunning: boolean;
  refreshToken: number;
  acceptanceBatch?: AcceptanceBatchSnapshot | null;
  suppressLegacySync?: boolean;
};

export default function FiveSignalFlow({algorithmClusterId, onAlgorithmClusterChange, selectedMode, onModeChange, isRunning, refreshToken, acceptanceBatch, suppressLegacySync = false}: FiveSignalFlowProps) {
  const uid = useId().replace(/[^a-zA-Z0-9_-]/g, "");
  const titleId = `five-signal-flow-${uid}-title`;
  const descId = `five-signal-flow-${uid}-desc`;
  const arrowMutedId = `five-signal-flow-${uid}-arrow-muted`;
  const arrowActiveId = `five-signal-flow-${uid}-arrow-active`;
  const arrowWarningId = `five-signal-flow-${uid}-arrow-warning`;
  const arrowReferenceId = `five-signal-flow-${uid}-arrow-reference`;
  const arrowEchoId = `five-signal-flow-${uid}-arrow-echo`;
  const agentDialogTitleId = `five-signal-flow-${uid}-agent-dialog-title`;
  const agentDialogDescriptionId = `five-signal-flow-${uid}-agent-dialog-description`;
  const requestRef = useRef<AbortController | null>(null);
  const dialogRef = useRef<HTMLDialogElement | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const [latestSnapshot, setSnapshot] =
    useState<FiveSignalFlowSnapshot>(EMPTY_SNAPSHOT);
  const batch = acceptanceBatch?.algorithm_cluster_id === algorithmClusterId ? acceptanceBatch : null;
  const batchSnapshot = batch ? deriveFiveSignalFlowSnapshot(batch.mode_reports ?? []) : null;
  const snapshot: FiveSignalFlowSnapshot = batch && batchSnapshot ? {
    ...batchSnapshot,
    generatedItems: batch.progress.completed_items,
    measuredItems: batch.summary.computed,
    counts: { pass: batch.summary.pass, fail: batch.summary.fail, review_required: batch.summary.review_required,
      not_measured: batch.summary.not_measured, invalid: batch.summary.invalid },
  } : suppressLegacySync ? EMPTY_SNAPSHOT : latestSnapshot;
  const [lastSyncedAt, setLastSyncedAt] = useState<Date | null>(null);
  const [syncing, setSyncing] = useState(false);
  const [syncError, setSyncError] = useState("");
  const [agentStates, setAgentStates] =
    useState<Record<AgentOutcomeKey, AgentState>>(EMPTY_AGENT_STATES);
  const [detail, setDetail] = useState<FiveSignalDetail | null>(null);
  const detailCache = useRef(new Map<string, FiveSignalDetail>());
  const [detailError, setDetailError] = useState("");
  const [detailInvalid, setDetailInvalid] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [scrollPosition, setScrollPosition] = useState(0);
  const [maxScroll, setMaxScroll] = useState(0);
  const [openOutcome, setOpenOutcome] = useState<AgentOutcomeKey | null>(null);

  const syncNow = useCallback(async () => {
    if (batch || suppressLegacySync) return;
    if (requestRef.current) return;
    const controller = new AbortController();
    requestRef.current = controller;
    setSyncing(true);
    try {
      const response = await fetch(`${EVALUATIONS_API}&algorithm_cluster_id=${encodeURIComponent(algorithmClusterId)}`, {
        cache: "no-store",
        signal: controller.signal,
      });
      if (!response.ok) {
        throw Error(`评估结果服务返回 HTTP ${response.status}`);
      }
      const payload: unknown = await response.json();
      if (!Array.isArray(payload)) {
        throw Error("评估结果响应不是数组");
      }
      setSnapshot(deriveFiveSignalFlowSnapshot(payload));
      setLastSyncedAt(new Date());
      setSyncError("");
    } catch (reason) {
      if (!controller.signal.aborted) {
        setSyncError(
          reason instanceof Error ? reason.message : "评估结果同步失败",
        );
      }
    } finally {
      if (requestRef.current === controller) {
        requestRef.current = null;
        setSyncing(false);
      }
    }
  }, [algorithmClusterId, Boolean(batch), suppressLegacySync]);

  useEffect(() => {
    const refreshWhenVisible = () => {
      if (document.visibilityState !== "hidden") void syncNow();
    };
    refreshWhenVisible();
    const timer = window.setInterval(refreshWhenVisible, POLL_INTERVAL_MS);
    window.addEventListener("focus", refreshWhenVisible);
    document.addEventListener("visibilitychange", refreshWhenVisible);
    return () => {
      window.clearInterval(timer);
      window.removeEventListener("focus", refreshWhenVisible);
      document.removeEventListener("visibilitychange", refreshWhenVisible);
      requestRef.current?.abort();
      requestRef.current = null;
    };
  }, [syncNow, refreshToken]);

  const algorithmCluster = chainPresets.find(preset => preset.id === algorithmClusterId)
    ?? chainPresets.find(preset => preset.id === "vcp_default")!;
  const selectedReport = snapshot.latestByMode[selectedMode];
  // Batch IDs identify condition-bound assessments, not legacy five-signal reports.
  const selectedId = batch || suppressLegacySync ? "" : selectedReport?.evaluation_id ?? "";
  useEffect(() => {
    const controller = new AbortController();
    setDetailError("");
    setDetailInvalid(false);
    if (!selectedId || !selectedReport?.valid) {
      setDetail(null);
      setDetailLoading(false);
      return () => controller.abort();
    }
    const cached = detailCache.current.get(selectedId);
    if (cached) {
      setDetail(cached);
      setDetailLoading(false);
      return () => controller.abort();
    }
    setDetail(current => current?.mode === selectedMode ? current : null);
    let inFlight = false;
    const load = async () => {
      if (inFlight || document.visibilityState === "hidden" || detailCache.current.has(selectedId)) return;
      inFlight = true;
      setDetailLoading(true);
      try {
        const response = await fetch(`/api/acoustic/five-signal/evaluations/${encodeURIComponent(selectedId)}`, {
          cache: "no-store", signal: controller.signal,
        });
        if (!response.ok) throw Error(`报告详情返回 HTTP ${response.status}`);
        const payload: unknown = await response.json();
        let parsed: FiveSignalDetail;
        try { parsed = parseFiveSignalDetail(payload, selectedId, selectedMode, algorithmClusterId); }
        catch (error) {
          if (!controller.signal.aborted) setDetailInvalid(true);
          throw error;
        }
        if (controller.signal.aborted) return;
        detailCache.current.set(selectedId, parsed);
        // Keep a bounded cache; polling may run for hours.
        if (detailCache.current.size > 24) detailCache.current.delete(detailCache.current.keys().next().value!);
        setDetail(parsed);
        setDetailError("");
        setDetailInvalid(false);
      } catch (error) {
        if (!controller.signal.aborted) setDetailError(error instanceof Error ? error.message : "详情同步失败");
      } finally {
        inFlight = false;
        if (!controller.signal.aborted) setDetailLoading(false);
      }
    };
    void load();
    const timer = window.setInterval(() => void load(), POLL_INTERVAL_MS);
    const refresh = () => void load();
    window.addEventListener("focus", refresh);
    document.addEventListener("visibilitychange", refresh);
    return () => {
      controller.abort();
      window.clearInterval(timer);
      window.removeEventListener("focus", refresh);
      document.removeEventListener("visibilitychange", refresh);
    };
  }, [selectedId, selectedMode, selectedReport?.valid, algorithmClusterId]);

  useEffect(() => {
    const viewport = scrollRef.current;
    if (!viewport) return;
    const update = () => setMaxScroll(Math.max(0, viewport.scrollWidth - viewport.clientWidth));
    const observer = new ResizeObserver(update);
    observer.observe(viewport);
    update();
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    if (openOutcome && !dialog.open) dialog.showModal();
    if (!openOutcome && dialog.open) dialog.close();
  }, [openOutcome]);

  const hasData = snapshot.availableModes > 0;
  const currentEvaluationIds = batch ? [] : evaluationIds(snapshot);
  const analyzeAgent = useCallback(
    async (outcome: AgentOutcomeKey, refresh = false) => {
      if (batch) return;
      setAgentStates((current) => ({
        ...current,
        [outcome]: {
          ...current[outcome],
          status: "loading",
          error: "",
        },
      }));
      try {
        const response = await fetch(
          `/api/acoustic/five-signal/outcomes/${outcome}/analyze`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              evaluation_ids: evaluationIds(snapshot),
              refresh,
            }),
          },
        );
        const payload: unknown = await response.json().catch(() => null);
        if (!response.ok) {
          const detail =
            payload &&
            typeof payload === "object" &&
            "detail" in payload &&
            typeof payload.detail === "string"
              ? payload.detail
              : `Codex 线程服务返回 HTTP ${response.status}`;
          throw Error(detail);
        }
        if (!payload || typeof payload !== "object") {
          throw Error("Codex 线程返回格式无效");
        }
        const result = payload as AgentAnalysis;
        setAgentStates((current) => ({
          ...current,
          [outcome]: {
            status: result.status === "empty" ? "empty" : "ready",
            result,
            error: "",
          },
        }));
      } catch (reason) {
        setAgentStates((current) => ({
          ...current,
          [outcome]: {
            status: "error",
            result: current[outcome].result,
            error:
              reason instanceof Error ? reason.message : "Codex 线程分析失败",
          },
        }));
      }
    },
    [snapshot, batch],
  );

  const openAgent = useCallback(
    (outcome: AgentOutcomeKey) => {
      const state = agentStates[outcome];
      const stale = agentIsStale(state, currentEvaluationIds);
      setOpenOutcome(outcome);
      if (state.status === "idle" || state.status === "error" || stale) {
        void analyzeAgent(outcome, stale);
      }
    },
    [agentStates, analyzeAgent, currentEvaluationIds],
  );

  const selectedDetail = !batch && detail?.mode === selectedMode ? detail : null;
  const detailStale = Boolean(selectedDetail && selectedDetail.evaluation_id !== selectedId);
  const execution = selectedDetail?.execution;
  const executionMatches = Boolean(execution?.source === "alignment_cli" && execution.algorithm_cluster_id === algorithmClusterId && !detailStale);
  const vcpState = batch ? (batch.status === "failed" ? "invalid" : batch.status === "completed" ? "complete" : batch.progress.completed_cases > 0 || isRunning ? "partial" : "pending") : isRunning ? "partial" : executionMatches ? "complete" : "pending";
  const availability = analysisAvailability(selectedDetail, snapshot.invalidModeCount > 0 || detailInvalid);
  const analysisState = batch ? (batch.summary.invalid > 0 ? "invalid" : batch.summary.computed > 0 ? "complete" : "pending") : availability.state;
  const analysisStatus = batch ? (batch.summary.invalid > 0 ? "存在执行异常" : batch.summary.computed > 0 ? "批次指标已生成" : "等待场景计算") : detailStale && analysisState !== "invalid"
    ? "旧报告快照" : availability.label;
  const groups = contractGroupsForMode(selectedMode);
  const groupsByDirection = {
    uplink: groups.filter(group => group.direction === "uplink"),
    downlink: groups.filter(group => group.direction === "downlink"),
    overall: groups.filter(group => group.direction === "overall"),
  };
  const directionTotals = Object.fromEntries(Object.entries(groupsByDirection).map(([key, rows]) => [key, rows.reduce((n, row) => n + row.count, 0)]));
  const knownThreads = new Set(Object.values(agentStates).map(state => state.result?.thread_id).filter(Boolean)).size;
  const sections = [
    { label: "数据与输入", x: 0 }, { label: "VCP 架构", x: FLOW.vcp.x - 24 },
    { label: "对齐与指标", x: FLOW.alignment.x - 24 }, { label: "42 项映射", x: FLOW.mapping.x - 24 },
    { label: "八模式", x: FLOW.modes.x - 24 }, { label: "分级与诊断", x: FLOW.classify.x - 24 },
    { label: "Agent", x: FLOW.agents.x - 24 }, { label: "报告与边界", x: FLOW.report.x - 24 },
  ];
  const activeSection = [...sections].reverse().find(section => section.x <= scrollPosition + 40) ?? sections[0];
  const numeric = (key: string, unit = "") => {
    const value = selectedDetail?.analysis.values[key];
    return finiteValue(value) ? `${value.toFixed(2)}${unit}` : "—";
  };
  const metricValues = [
    `麦克风混合输入 ${numeric("microphone_sisdr_db")} / VCP 处理后的近端输出 ${numeric("output_sisdr_db")} / Δ ${numeric("sisdr_delta_db")} dB`,
    `麦克风混合输入 ${numeric("microphone_stoi")} / VCP 处理后的近端输出 ${numeric("output_stoi")} / Δ ${numeric("stoi_delta")}`,
    `SNR 麦克风混合输入 ${numeric("input_snr_db")} / VCP 处理后的近端输出 ${numeric("output_snr_db")} dB`,
    `频响平坦度 ${numeric("frequency_flatness_db", " dB")} · 功率变化 ${numeric("noise_spectrum_change_db", " dB")}`,
    `损耗 ${numeric("echo_loss_db")} / 降低 ${numeric("echo_reduction_db")} dB`,
    `双讲区段输出 / 输入 ${numeric("double_talk_transparent_percent", "%")}`,
  ];
  const resultState =
    snapshot.state === "complete"
      ? "条目齐全 · 非验收通过"
      : snapshot.state === "invalid"
        ? `数据异常 ${snapshot.invalidModeCount}`
        : snapshot.state === "partial"
          ? `部分结果 ${snapshot.availableModes}/8`
          : "等待评估";
  const syncMeta = suppressLegacySync ? "正在恢复最近批次，等待真实批次统计…" : batch ? `同一批次 ${batch.batch_id.slice(0, 12)} · ${batch.progress.completed_cases}/${batch.progress.total_cases} 条件组` : syncError
    ? `显示上次成功快照${lastSyncedAt ? ` · ${formatSyncTime(lastSyncedAt)}` : " · 尚无可用快照"}`
    : lastSyncedAt
      ? `按模式取最新 · ${formatSyncTime(lastSyncedAt)} 同步`
      : syncing
        ? "正在同步最近结果…"
        : "等待结果同步";
  const visualState = !batch && syncError && !lastSyncedAt ? "invalid" : snapshot.state;
  const exportReport = () => {
    if (batch) {
      window.location.assign(`/api/alignment/acceptance/batches/${encodeURIComponent(batch.batch_id)}/report`);
      return;
    }
    const report = {
      scope: "最近持久化工程结果快照，非同一批次验收",
      exported_at: new Date().toISOString(),
      last_synced_at: lastSyncedAt?.toISOString() ?? null,
      standard_compliance: false,
      formal_test_status: "not_measured",
      snapshot,
      selected_mode: selectedMode,
      architecture_algorithm_cluster: {
        id: algorithmCluster.id,
        label: algorithmCluster.label,
        selection: algorithmCluster.selection,
        scope: executionMatches ? "本报告由所选算法簇实际上行处理生成" : "所选算法簇尚无当前模式执行报告",
      },
      selected_detail: selectedDetail,
      detail_stale: detailStale,
      sync_error: syncError || null,
      detail_error: detailError || null,
      diagnostics: Object.fromEntries(Object.entries(agentStates).map(([key, state]) => [key, {
        ...state,
        stale: agentIsStale(state, currentEvaluationIds),
      }])),
    };
    const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `five-signal-engineering-${selectedMode}.json`;
    anchor.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  const openOutcomeRow =
    OUTCOME_ROWS.find((row) => row.key === openOutcome) ?? null;
  const openAgentState = openOutcome ? agentStates[openOutcome] : null;
  const openAgentStale = openAgentState
    ? agentIsStale(openAgentState, currentEvaluationIds)
    : false;
  const flowArrow = (d: string, state: string, key: string) => {
    const active = state === "complete";
    const warning = state === "invalid";
    const marker = active
      ? arrowActiveId
      : warning
        ? arrowWarningId
        : arrowMutedId;
    return (
      <path
        className={`five-signal-flow-arrow ${active ? "is-active" : warning ? "is-warning" : ""}`}
        d={d}
        markerEnd={`url(#${marker})`}
        data-flow-link={key}
        key={key}
      />
    );
  };

  return (
    <section
      className="five-signal-flow-card"
      aria-labelledby="five-signal-flow-heading"
    >
      <header className="five-signal-flow-header">
        <div className="five-signal-flow-heading">
          <span className="five-signal-flow-eyebrow">
            FIVE-SIGNAL ENGINEERING FLOW
          </span>
          <h2 id="five-signal-flow-heading">
            五路录音评估 <span aria-hidden="true">→</span> 336 项工程条目
          </h2>
          {batch ? <p>
            当前显示测试集“{batch.dataset_name}”的同一批次。每个独立条件组经过所选 VCP 必要上行 / 下行算法，实际输出交给对应计算模型。
            场景、音量、窗长与报告身份逐项绑定；数值代理需复核，正式 ITU-T 合规未测。
          </p> : <p>
            在下方“336 项场景批次验收”导入测试集并一键执行，可联动本图。旧单场景自检仅生成所选模式 42 项合同条目。
            当前使用固定近端 / 噪声场景，下行保持架构示意。按算法簇筛选、按模式取最新报告，
            不代表同一批次；工程判定计数包含代理与待复核项，不保证每项都有有效数值。正式 ITU-T 合规未测。
          </p>}
        </div>
        <div className="five-signal-flow-side">
          <span
            className={`five-signal-flow-status is-${!batch && syncError ? "invalid" : visualState}`}
            aria-live="polite"
          >
            <i aria-hidden="true" />
            {suppressLegacySync ? "正在恢复批次" : batch ? (batch.status === "failed" ? "批次执行异常" : batch.status === "completed" ? "336 项处理结束 · 查看逐项状态" : "批次正在执行") : syncError ? "状态同步失败" : statusCopy(snapshot)}
          </span>
          <span className="five-signal-flow-meta">{syncMeta}</span>
          {!batch && syncError ? (
            <button
              className="five-signal-flow-retry"
              type="button"
              onClick={() => void syncNow()}
              disabled={syncing}
            >
              <RefreshCw size={13} aria-hidden="true" />
              立即重试
            </button>
          ) : null}
          <a className="five-signal-flow-link" href="#acoustic">
            前往五路验收
            <ArrowRight size={14} aria-hidden="true" />
          </a>
        </div>
      </header>

      <div className="five-signal-flow-report-context">
        <label>执行 / 查看业务模式 <select value={selectedMode} disabled={isRunning} onChange={event => onModeChange(event.target.value as FiveSignalMode)}>
          {FIVE_SIGNAL_MODES.map(mode => <option key={mode} value={mode}>{MODE_LABELS[mode]} · {mode}{snapshot.latestByMode[mode] ? " · 有报告" : " · 暂无报告"}</option>)}
        </select></label>
        <span role="status">{suppressLegacySync ? "正在恢复最近批次，尚未启动新的 VCP 处理" : isRunning ? `正在执行 ${algorithmCluster.label} · ${selectedMode}…` : `选定算法簇：${algorithmCluster.label}`}</span>
        <span role="status">{batch ? `当前批次 ${batch.batch_id} · 逐项条件及产物见下方批次表` : detailLoading ? "正在获取报告详情…" : selectedDetail ? `报告 ${selectedDetail.evaluation_id} · ${selectedDetail.created_at}` : "暂无可用报告详情"}</span>
        {detailError && <p role="alert">详情同步失败：{detailError}。{selectedDetail ? `保留报告 ${selectedDetail.evaluation_id}；不代表最新报告。` : "指标保持待同步，不标记通过。"}</p>}
        {selectedDetail && <details><summary>查看五路文件身份、采样率、时长与指标数值</summary>
          <div className="five-signal-flow-metadata-table"><table><thead><tr><th>输入角色</th><th>文件</th><th>采样率</th><th>时长</th><th>SHA-256</th></tr></thead><tbody>
            {FIVE_SIGNAL_ROLES.map(role => { const input = selectedDetail.inputs[role]; return <tr key={role}><th>{FIVE_SIGNAL_ROLE_LABELS[role]}（{role}）</th><td>{input.name}</td><td>{input.sample_rate} Hz</td><td>{input.duration_s.toFixed(2)} s</td><td><code>{input.sha256}</code></td></tr>; })}
          </tbody></table></div>
          <p>VCP 处理后的近端输出（near_end）来自本次 CLI 处理；麦克风混合输入（microphone）基线取混合输入第 0 通道，干净近端参考（reference）取波束目标；实际引擎使用标定多麦输入。固定场景含近端、远端与双讲区段，但业务模式只选择合同条目，不模拟传输链路。</p>
          {execution && <p>执行身份：{execution.algorithm_cluster_label} · 数据集 {execution.dataset_id} · 输出 SHA-256 <code>{execution.output_sha256}</code></p>}
          <div className="five-signal-flow-metadata-table"><table><thead><tr><th>指标字段</th><th>值（报告原单位）</th></tr></thead><tbody>
            {METRIC_VALUE_KEYS.flat().map(key => <tr key={key}><th>{key}</th><td>{numeric(key)}</td></tr>)}
          </tbody></table></div>
        </details>}
        {snapshot.invalidModeCount > 0 && <div className="five-signal-flow-report-anomalies" role="alert">
          <strong>报告异常 {snapshot.invalidModeCount} 份，与无效测项分开统计</strong>
          <ul>{Object.values(snapshot.latestByMode).filter(row => !row.valid).map(row => <li key={row.mode}>{row.mode} · {row.evaluation_id || "缺少报告 ID"}：{row.problem}。重新生成该模式的完整 42 项报告后再诊断。</li>)}</ul>
          {snapshot.ignoredModeCount > 0 && <p>另有 {snapshot.ignoredModeCount} 份未知模式报告被排除。</p>}
          <p>四类 Agent 接收有效报告中的测项证据；异常报告须先修复，不能作为无效测项发送。</p>
        </div>}
      </div>

      <div className="five-signal-flow-toolbar">
        <nav aria-label="流程图区段" className="five-signal-flow-sections">
          {sections.map((section) => (
            <button
              type="button"
              key={section.label}
              aria-current={section.label === activeSection.label ? "location" : undefined}
              onClick={() => scrollRef.current?.scrollTo({ left: section.x })}
            >
              {section.label}
              <ArrowRight size={12} aria-hidden="true" />
            </button>
          ))}
        </nav>
        <span className="five-signal-flow-totals" aria-live="polite">
          {snapshot.availableModes}/8 模式 · {snapshot.generatedItems}/336 条目 · 有限数值 {snapshot.measuredItems}/336 项（代理需复核）
        </span>
        <span className="five-signal-flow-scroll-hint">
          拖动下方滑杆或滚动图内画布
        </span>
      </div>
      <div className="five-signal-flow-scroll-control">
        <label htmlFor={`${uid}-scroll`}>横向位置 · {activeSection.label}</label>
        <input id={`${uid}-scroll`} type="range" min="0" max={Math.max(1, maxScroll)} value={scrollPosition} onChange={event => scrollRef.current?.scrollTo({ left: Number(event.target.value) })} aria-label="流程图横向滚动位置" />
        <span className="five-signal-flow-legend">{batch ? "按条件执行必要上行 / 下行链路 · 虚线：声学仿真或路径依赖" : "实线：实际处理 / 评估依赖 · 虚线：下行架构示意 · 回声虚线：声学耦合"}</span>
      </div>
      <div
        className="five-signal-flow-scroll"
        ref={scrollRef}
        onScroll={event => setScrollPosition(event.currentTarget.scrollLeft)}
        tabIndex={0}
        role="region"
        aria-label="工程验收流程图，可横向滚动"
      >
        <svg
          className="five-signal-flow-svg"
          viewBox={`0 0 ${FLOW.width} ${FLOW.height}`}
          style={{ width: FLOW.width, minWidth: FLOW.width }}
          role="group"
          aria-labelledby={`${titleId} ${descId}`}
        >
          <title id={titleId}>
            测试数据集、五路音频、VCP 双轨核心到 336 项工程验收流程图
          </title>
          <desc id={descId}>
            {batch ? "当前数据来自同一 336 项测试集批次，各条件组经过实际 VCP 上下行处理，再使用对应模型逐项计算；无数值项不计已测。" : "选定算法簇先处理固定数据集的麦克风混合输入（microphone），实际结果作为 VCP 处理后的近端输出（near_end），其余为独立噪声输入（noise）、麦克风混合输入（microphone）、远端播放信号（回声参考，far_end）和干净近端参考（reference）。"}
            实线直接连接输入与评估：解码重采样、活动区段识别、相对时延与对齐，随后逻辑并列展示
            SI-SDR、STOI、电平与 SNR、频响与频谱、回声损耗、双讲区段功率比六组指标。
            {batch ? "VCP 上行和下行按测试条件调用，扬声器至双耳 DRP 为软件仿真代理，详细产物在批次中下载。" : "VCP 上行 BF、AEC、NS、EQ、AGC/DRC 按选定预设真实执行，下行 NR、EQ、DRC、扬声器保持架构示意。"}
            六组指标按所选模式映射 42 项合同条目，八个模式分别汇总最多 336 条；
            条目齐全不代表场景或音量条件覆盖，也不代表指标全部可用或正式通过。
            工程通过进入报告归档；失败、需复核、不可测、无效分别进入四类诊断入口，点击后才启动或续接 Codex 线程。
            诊断只读结构化证据，不读原始音频。所有结果均受正式合规边界约束，正式 ITU-T 合规未测。
            当前 {snapshot.availableModes}/8 模式，{snapshot.generatedItems}/336 条目，工程判定 {snapshot.measuredItems} 项。

          </desc>
          <defs>
            <marker
              id={arrowMutedId}
              markerWidth="10"
              markerHeight="10"
              refX="9"
              refY="5"
              orient="auto"
              markerUnits="userSpaceOnUse"
            >
              <path d="M0 0 L10 5 L0 10 Z" fill="#cbd5e1" />
            </marker>
            <marker
              id={arrowActiveId}
              markerWidth="10"
              markerHeight="10"
              refX="9"
              refY="5"
              orient="auto"
              markerUnits="userSpaceOnUse"
            >
              <path d="M0 0 L10 5 L0 10 Z" fill="#2563eb" />
            </marker>
            <marker
              id={arrowWarningId}
              markerWidth="10"
              markerHeight="10"
              refX="9"
              refY="5"
              orient="auto"
              markerUnits="userSpaceOnUse"
            >
              <path d="M0 0 L10 5 L0 10 Z" fill="#d59b28" />
            </marker>
            <marker
              id={arrowReferenceId}
              markerWidth="10"
              markerHeight="10"
              refX="9"
              refY="5"
              orient="auto"
              markerUnits="userSpaceOnUse"
            >
              <path d="M0 0 L10 5 L0 10 Z" fill="#d97706" />
            </marker>
            <marker
              id={arrowEchoId}
              markerWidth="10"
              markerHeight="10"
              refX="9"
              refY="5"
              orient="auto"
              markerUnits="userSpaceOnUse"
            >
              <path d="M0 0 L10 5 L0 10 Z" fill="#9333ea" />
            </marker>
          </defs>

          <FlowCard
            {...FLOW.dataset}
            y={FLOW.inputTop}
            height={FLOW.inputHeight}
            tag="00 · DATASET"
            title="测试数据集信息"
            subtitle={batch ? `${batch.progress.total_cases} 条件组 · 场景 / 音量 / 窗长` : "等待导入场景测试集，或查看单场景报告"}
            state={batch || selectedDetail && !detailStale ? "complete" : "pending"}
            name="dataset"
          >
            <line
              className="five-signal-flow-node-rule"
              x1="24"
              y1="116"
              x2="296"
              y2="116"
            />
            {[
              ["业务模式", selectedMode],
              ["报告 ID", batch ? `${batch.batch_id.slice(0, 12)}…` : selectedDetail ? `${selectedDetail.evaluation_id.slice(0, 12)}…` : "等待报告"],
              ["输入身份", batch ? `${batch.dataset_id.slice(0, 12)}…` : execution?.dataset_id ?? "未导入批次测试集"],
            ].map(([label, value], index) => (
              <g key={label}>
                <text
                  className="five-signal-flow-dataset-label"
                  x="24"
                  y={148 + index * 30}
                >
                  {label}
                </text>
                <text
                  className="five-signal-flow-dataset-value"
                  x="296"
                  y={148 + index * 30}
                  textAnchor="end"
                >
                  {value}
                </text>
              </g>
            ))}
            <text className="five-signal-flow-node-status" x="24" y="250">
              {batch ? "批次表可查条件、哈希及实际输出" : selectedDetail ? (detailStale ? "旧报告输入 · 同步未完成" : "上方可展开查看完整文件信息") : "等待所选模式的报告详情"}
            </text>
          </FlowCard>

          <FlowCard
            {...FLOW.input}
            y={FLOW.inputTop}
            height={380}
            tag="01 · INPUT"
            title="五路音频输入"
            subtitle="VCP 处理后的近端输出由所选算法簇实际生成"
            state={nodeState(snapshot, "input")}
            name="input"
          >
            {FIVE_SIGNAL_ROLES.map((role, index) => <g key={role}>
              <rect className="five-signal-flow-input-chip" x="24" y={120 + index * 40} width={FLOW.input.width - 48} height="32" rx="6" />
              <text className="five-signal-flow-input-chip-label" x="40" y={141 + index * 40}>{FIVE_SIGNAL_ROLE_LABELS[role]}</text>
            </g>)}
            <text className="five-signal-flow-node-status" x="24" y="350">独立噪声输入无需与语音等长</text>

          </FlowCard>

          <FlowGroup
            {...FLOW.vcp}
            y={FLOW.top}
            height={FLOW.vcpGroupHeight}
            tag="02 · VCP PROCESSING"
            title={batch ? "VCP 上行 / 下行条件执行" : "VCP 上行执行 / 下行架构"}
            status={batch ? `${batch.progress.completed_cases}/${batch.progress.total_cases} 条件组 · ${batch.status}` : isRunning ? "上行执行中" : executionMatches ? "上行已执行 · 下行未执行" : "上行待执行 · 下行示意"}
            state={vcpState}
            name="vcp"
          >
            <text
              className="five-signal-flow-vcp-group-title"
              x="24"
              y={FLOW.uplinkY - FLOW.top - 68}
            >
              VCP 上行核心
            </text>
            <text
              className="five-signal-flow-vcp-group-meta"
              x="24"
              y={FLOW.uplinkY - FLOW.top - 44}
            >
              BF → AEC → NS → EQ → AGC/DRC → VCP 处理后的近端输出
            </text>
            {VCP_UPLINK_STAGES.map((stage, index) => (
              <VcpStageCard
                x={24 + index * VCP_STAGE_STEP}
                y={FLOW.uplinkY - FLOW.top}
                stage={stage}
                state={vcpState}
                label={`VCP 上行 ${index + 1}/5`}
                key={stage.code}
              />
            ))}
            {VCP_UPLINK_STAGES.slice(0, -1).map((stage, index) =>
              flowArrow(
                `M ${24 + index * VCP_STAGE_STEP + VCP_STAGE_WIDTH} ${FLOW.mainY - FLOW.top} H ${24 + (index + 1) * VCP_STAGE_STEP}`,
                vcpState,
                `uplink-${stage.code}`,
              ),
            )}
            <text
              className="five-signal-flow-vcp-group-title"
              x="24"
              y={FLOW.downlinkY - FLOW.top - 58}
            >
              VCP 下行核心
            </text>
            <text
              className="five-signal-flow-vcp-group-meta"
              x="24"
              y={FLOW.downlinkY - FLOW.top - 34}
            >
              RX NR → RX EQ → RX DRC → 扬声器 / 回声
            </text>
            {VCP_DOWNLINK_STAGES.map((stage, index) => (
              <VcpStageCard
                x={24 + index * VCP_STAGE_STEP}
                y={FLOW.downlinkY - FLOW.top}
                stage={stage}
                state={batch && stage.variant !== "acoustic" ? vcpState : "pending"}
                label={`VCP 下行 ${index + 1}/4`}
                key={stage.code}
              />
            ))}
            {VCP_DOWNLINK_STAGES.slice(0, -1).map((stage, index) =>
              flowArrow(
                `M ${24 + index * VCP_STAGE_STEP + VCP_STAGE_WIDTH} ${FLOW.downlinkY + VCP_STAGE_HEIGHT / 2 - FLOW.top} H ${24 + (index + 1) * VCP_STAGE_STEP}`,
                batch ? vcpState : "pending",
                `downlink-${stage.code}`,
              ),
            )}
            <path
              className="five-signal-flow-arrow is-reference"
              data-flow-link="aec-reference"
              d={`M ${24 + 2 * VCP_STAGE_STEP + VCP_STAGE_WIDTH / 2} ${FLOW.downlinkY - FLOW.top} V 416 H ${24 + VCP_STAGE_STEP + VCP_STAGE_WIDTH / 2} V ${FLOW.uplinkY + VCP_STAGE_HEIGHT - FLOW.top}`}
              markerEnd={`url(#${arrowReferenceId})`}
            />
            <rect
              className="five-signal-flow-route-label-bg"
              x="330"
              y="392"
              width="176"
              height="24"
              rx="4"
            />
            <text
              className="five-signal-flow-reference-label"
              x="418"
              y="408"
              textAnchor="middle"
            >
              远端播放信号 / 回声参考
            </text>
            <foreignObject
              x="24"
              y={FLOW.algorithmClusterTop - FLOW.top}
              width={FLOW.vcp.width - 48}
              height={FLOW.algorithmClusterHeight}
              data-flow-node="algorithm-cluster"
            >
              <div className="five-signal-flow-algorithm-cluster">
                <div className="five-signal-flow-algorithm-cluster-head">
                  <label htmlFor={`${uid}-algorithm-cluster`}>选择算法簇
                    <select
                      id={`${uid}-algorithm-cluster`}
                      value={algorithmCluster.id}
                      disabled={isRunning}
                      onChange={event => onAlgorithmClusterChange(event.target.value)}
                    >
                      {chainPresets.map(preset => (
                        <option key={preset.id} value={preset.id}>{preset.label}</option>
                      ))}
                    </select>
                  </label>
                  <p>{algorithmCluster.description}</p>
                </div>
                <dl className="five-signal-flow-algorithm-cluster-stages" aria-live="polite">
                  {chainStages.map(stage => (
                    <div key={stage.id}>
                      <dt>{stage.label}</dt>
                      <dd>{optionFor(stage, algorithmCluster).label}</dd>
                    </div>
                  ))}
                </dl>
                <p className="five-signal-flow-algorithm-cluster-note">
                  与下方执行控制台共享选型；点击“一键执行测试”后运行该上行预设并归档实际输出。下行保持架构示意。
                </p>
              </div>
            </foreignObject>
          </FlowGroup>

          <path
            className="five-signal-flow-arrow is-echo"
            data-flow-link="acoustic-echo"
            d={`M ${FLOW.vcp.x + 24 + 3 * VCP_STAGE_STEP + VCP_STAGE_WIDTH} ${FLOW.downlinkY + VCP_STAGE_HEIGHT / 2} H ${FLOW.vcp.right - 24} V 852 H ${FLOW.vcp.x + 8} V ${FLOW.mainY} H ${FLOW.vcp.x + 24}`}
            markerEnd={`url(#${arrowEchoId})`}
          />
          <rect
            className="five-signal-flow-route-label-bg is-echo"
            x={FLOW.vcp.x + 400}
            y="840"
            width="176"
            height="24"
            rx="4"
          />
          <text
            className="five-signal-flow-echo-label"
            x={FLOW.vcp.x + 488}
            y="856"
            textAnchor="middle"
          >
            声学耦合 → 麦克风混合输入
          </text>

          <FlowCard
            {...FLOW.alignment}
            y={FLOW.top}
            height={FLOW.stageHeight}
            tag="03 · ALIGN"
            title="相对时延 / 对齐"
            subtitle="解码重采样 · 输入检查 · 区段识别"
            state={analysisState}
            name="alignment"
          >
            {["干净近端参考 → 麦克风混合输入", "干净近端参考 → VCP 处理后的近端输出"].map((label, index) => (
              <g key={label}>
                <rect
                  className="five-signal-flow-align-item"
                  x="24"
                  y={144 + index * 112}
                  width="256"
                  height="88"
                  rx="8"
                />
                <text
                  className="five-signal-flow-align-label"
                  x="40"
                  y={176 + index * 112}
                >
                  {label}
                </text>
                <text
                  className="five-signal-flow-align-value"
                  x="40"
                  y={208 + index * 112}
                >
                  {selectedDetail ? (() => {
                    const value = selectedDetail.analysis.alignments[index === 0 ? "microphone" : "near_end"];
                    return `${finiteValue(value.delay_ms) ? value.delay_ms.toFixed(1) : "—"} ms · 相关 ${finiteValue(value.correlation) ? value.correlation.toFixed(2) : "—"} · ${value.reliable ? "可靠" : "需复核"}`;
                  })() : "等待时延与相关度"}
                </text>
              </g>
            ))}
            <line
              className="five-signal-flow-node-rule"
              x1="24"
              y1="376"
              x2="280"
              y2="376"
            />
            <text
              className="five-signal-flow-align-section-label"
              x="24"
              y="412"
            >
              对齐输出
            </text>
            {["先识别近端单讲 / 远端单讲", "再识别双讲 / 噪声区段", "对齐麦克风混合输入与 VCP 处理后的近端输出"].map(
              (label, index) => (
                <text
                  className="five-signal-flow-align-output"
                  x="24"
                  y={448 + index * 32}
                  key={label}
                >
                  {label}
                </text>
              ),
            )}
            <rect
              className="five-signal-flow-align-baseline-box"
              x="24"
              y="564"
              width="256"
              height="196"
              rx="8"
            />
            <text
              className="five-signal-flow-align-baseline-title"
              x="40"
              y="600"
            >
              共享分析基线
            </text>
            {[
              "六项指标共享对齐基线。",
              "并行表示逻辑依赖，",
              "不表示六个执行任务。",
              "相对时延不是完整 RTD。",
            ].map((label, index) => (
              <text
                className="five-signal-flow-align-baseline-copy"
                x="40"
                y={636 + index * 24}
                key={label}
              >
                {label}
              </text>
            ))}
            {METRIC_ROW_CENTERS.map((center) => (
              <circle
                className="five-signal-flow-port"
                cx={FLOW.alignment.width}
                cy={center - FLOW.top}
                r="3"
                key={center}
              />
            ))}
            <text className="five-signal-flow-node-status" x="24" y="844">
              {analysisStatus}
            </text>
          </FlowCard>

          <FlowGroup
            {...FLOW.analysis}
            y={FLOW.top}
            height={FLOW.stageHeight}
            tag="04 · PARALLEL ANALYSIS"
            title="六项指标并行分析"
            status={analysisStatus}
            state={analysisState}
            name="analysis"
          >
            {ANALYSIS_METRICS.map((metric, index) => {
              const y = FLOW.metricTop + index * FLOW.metricStep - FLOW.top;
              return (
                <g
                  className={`five-signal-flow-node five-signal-flow-metric state-${analysisState}`}
                  role="group"
                  aria-label={`${metric.title}，${metric.line1}，${metric.line2}，${analysisStatus}`}
                  key={metric.tag}
                >
                  <rect
                    className="five-signal-flow-node-box"
                    x="16"
                    y={y}
                    width={FLOW.analysis.width - 32}
                    height={FLOW.metricHeight}
                    rx="8"
                    data-flow-node="metric"
                  />
                  <rect
                    className="five-signal-flow-metric-index-bg"
                    x="32"
                    y={y + 16}
                    width="36"
                    height="36"
                    rx="6"
                  />
                  <text
                    className="five-signal-flow-metric-index-text"
                    x="50"
                    y={y + 39}
                    textAnchor="middle"
                  >
                    {String(index + 1).padStart(2, "0")}
                  </text>
                  <text className="five-signal-flow-node-tag" x="84" y={y + 27}>
                    {metric.tag}
                  </text>
                  <text
                    className="five-signal-flow-node-title"
                    x="84"
                    y={y + 59}
                  >
                    {metric.title}
                  </text>
                  <text
                    className="five-signal-flow-node-copy"
                    x="320"
                    y={y + 38}
                  >
                    {metric.line1}
                  </text>
                  <text
                    className="five-signal-flow-node-copy"
                    x="320"
                    y={y + 62}
                  >
                    {metric.line2}
                  </text>
                  <line
                    className="five-signal-flow-node-rule"
                    x1="84"
                    y1={y + 74}
                    x2={FLOW.analysis.width - 32}
                    y2={y + 74}
                  />
                  <text
                    className="five-signal-flow-metric-state"
                    x={FLOW.analysis.width - 32}
                    y={y + 95}
                    textAnchor="end"
                  >
                    {metricValues[index]} · {analysisStatus}
                  </text>
                </g>
              );
            })}
          </FlowGroup>

          <FlowCard
            {...FLOW.mapping}
            y={FLOW.top}
            height={FLOW.stageHeight}
            tag="05 · MAP"
            title="单模式映射"
            subtitle="六项指标进入固定契约 · 按业务模式分别生成"
            state={nodeState(snapshot, "mapping")}
            name="mapping"
          >
            <rect
              className="five-signal-flow-align-baseline-box"
              x="24"
              y="136"
              width="632"
              height="112"
              rx="10"
            />
            <text className="five-signal-flow-stat-number" x="48" y="206">
              42
            </text>
            <text className="five-signal-flow-stat-unit" x="128" y="206">
              项 / 模式
            </text>
            <text
              className="five-signal-flow-node-status"
              x="632"
              y="184"
              textAnchor="end"
            >
              {snapshot.availableModes}/8 模式有条目
            </text>
            <text
              className="five-signal-flow-node-copy"
              x="632"
              y="214"
              textAnchor="end"
            >
              {selectedMode} · 7 类展示分组
            </text>
            <text
              className="five-signal-flow-mapping-rule-title"
              x="24"
              y="304"
            >
              按方向分层
            </text>
            <text className="five-signal-flow-node-copy" x="24" y="332">
              上行、下行、总体 → 七个二级节点 → 42 项
            </text>
            <path
              className="five-signal-flow-mapping-branch"
              d="M 340 354 V 374 M 124 414 V 374 H 556 V 414 M 340 374 V 414"
            />
            {(
              [
                ["uplink", 24],
                ["downlink", 240],
                ["overall", 456],
              ] as const
            ).map(([direction, x]) => (
              <g key={direction} transform="translate(0 -32)">
                <rect
                  className="five-signal-flow-mapping-direction-box"
                  x={x}
                  y="446"
                  width={CONTRACT_COLUMN_WIDTH}
                  height="44"
                  rx="8"
                />
                <ContractDirectionColumn
                  x={x}
                  label={FIVE_SIGNAL_CONTRACT_DIRECTION_LABELS[direction]}
                  total={directionTotals[direction]}
                  groups={groupsByDirection[direction]}
                />
              </g>
            ))}
            <text className="five-signal-flow-metric-foot" x="24" y="844">
              {directionTotals.uplink} 项上行 + {directionTotals.downlink} 项下行 + {directionTotals.overall} 项总体 = 42 项
            </text>
            <text className="five-signal-flow-metric-foot" x="24" y="802">生成条目 ≠ 42 次独立测量；场景与音量条件须另行核验</text>
            <text className="five-signal-flow-metric-foot" x="24" y="824">总体是图中展示分组；相对时延仅作 RTD 代理并需复核</text>
            {METRIC_ROW_CENTERS.map((center) => (
              <circle
                className="five-signal-flow-port"
                cx="0"
                cy={center - FLOW.top}
                r="3"
                key={center}
              />
            ))}
            {MODE_ROW_CENTERS.map((center) => (
              <circle
                className="five-signal-flow-port"
                cx={FLOW.mapping.width}
                cy={center - FLOW.top}
                r="3"
                key={center}
              />
            ))}
          </FlowCard>

          <FlowGroup
            {...FLOW.modes}
            y={FLOW.top}
            height={FLOW.stageHeight}
            tag="06 · MODE AGGREGATE"
            title="八模式汇总"
            status={`${snapshot.availableModes}/8 模式`}
            state={snapshot.state}
            name="modes"
          >
            {FIVE_SIGNAL_MODES.map((mode, index) => {
              const y = FLOW.modeTop + index * FLOW.modeStep - FLOW.top;
              const latest = snapshot.latestByMode[mode];
              const summary = latest?.summary;
              const state = modeRowState(snapshot, mode);
              const status = latest
                ? latest.valid
                  ? "已汇总"
                  : "数据异常"
                : "等待评估";
              return (
                <g
                  className={`five-signal-flow-node five-signal-flow-mode-row state-${state}`}
                  role="group"
                  aria-label={`${MODE_LABELS[mode]} ${mode}。目标简述：${MODE_METAS[mode].role}。测试重点：${MODE_METAS[mode].testingFocus}。${status}，工程判定 ${summary?.measured ?? 0}/42 项。`}
                  key={mode}
                >
                  <rect
                    className="five-signal-flow-node-box"
                    x="16"
                    y={y}
                    width={FLOW.modes.width - 32}
                    height={FLOW.modeHeight}
                    rx="8"
                    data-flow-node="mode"
                  />
                  <rect
                    className="five-signal-flow-mode-index-bg"
                    x="32"
                    y={y + 14}
                    width="32"
                    height="32"
                    rx="6"
                  />
                  <text
                    className="five-signal-flow-mode-index-text"
                    x="48"
                    y={y + 35}
                    textAnchor="middle"
                  >
                    {String(index + 1).padStart(2, "0")}
                  </text>
                  <text
                    className="five-signal-flow-mode-title"
                    x="76"
                    y={y + 24}
                  >
                    {MODE_LABELS[mode]}
                  </text>
                  <text
                    className="five-signal-flow-mode-code"
                    x="188"
                    y={y + 24}
                  >
                    {mode}
                  </text>
                  <text
                    className="five-signal-flow-mode-total"
                    x={FLOW.modes.width - 32}
                    y={y + 24}
                    textAnchor="end"
                  >
                    {summary ? `${summary.total}/42` : "—/42"}
                  </text>
                  <text
                    className="five-signal-flow-mode-target"
                    x="76"
                    y={y + 48}
                  >
                    目标 · {MODE_METAS[mode].role}
                  </text>
                  <line
                    className="five-signal-flow-node-rule"
                    x1="76"
                    y1={y + 60}
                    x2={FLOW.modes.width - 32}
                    y2={y + 60}
                  />
                  <text
                    className="five-signal-flow-mode-measured"
                    x="76"
                    y={y + 78}
                  >
                    工程判定 {summary?.measured ?? 0}/42 项
                  </text>
                  <text
                    className="five-signal-flow-mode-status"
                    x={FLOW.modes.width - 32}
                    y={y + 78}
                    textAnchor="end"
                  >
                    {status}
                  </text>
                </g>
              );
            })}
            <text className="five-signal-flow-metric-foot" x="20" y="872">
              {snapshot.generatedItems}/336 项 · {batch ? "同一场景批次" : "按模式取最新"}
            </text>
          </FlowGroup>

          <FlowCard
            {...FLOW.classify}
            y={FLOW.top}
            height={FLOW.stageHeight}
            tag="07 · CLASSIFY"
            title="结果分级"
            subtitle="对全部模式结果统一分级"
            state={nodeState(snapshot, "classify")}
            name="classify"
          >
            {(
              [
                { key: "pass", label: "工程通过", dot: "is-pass" },
                ...OUTCOME_ROWS.map((row) => ({
                  key: row.key,
                  label: row.title,
                  dot: row.dotClass,
                })),
              ] as const
            ).map((row, index) => (
              <g key={row.key}>
                <circle
                  className={`five-signal-flow-result-dot ${row.dot}`}
                  cx="32"
                  cy={164 + index * 56}
                  r="4"
                />
                <text
                  className="five-signal-flow-result-label"
                  x="48"
                  y={169 + index * 56}
                >
                  {row.label}
                </text>
                <text
                  className="five-signal-flow-result-value"
                  x="296"
                  y={169 + index * 56}
                  textAnchor="end"
                >
                  {snapshot.counts[row.key]}
                </text>
              </g>
            ))}
            <line
              className="five-signal-flow-node-rule"
              x1="24"
              y1="440"
              x2="296"
              y2="440"
            />
            <text className="five-signal-flow-node-status" x="24" y="480">
              {resultState}
            </text>
            <rect
              className="five-signal-flow-classify-note-box"
              x="24"
              y="528"
              width="272"
              height="232"
              rx="8"
            />
            <text
              className="five-signal-flow-classify-note-title"
              x="40"
              y="564"
            >
              分级说明
            </text>
            {[
              "工程通过不代表正式合规",
              "不可测项不按通过处理",
              "无效数据保持显式标记",
              "正式 ITU-T 合规另行验证",
            ].map((label, index) => (
              <text
                className="five-signal-flow-classify-note-copy"
                x="40"
                y={604 + index * 32}
                key={label}
              >
                {label}
              </text>
            ))}
            {MODE_ROW_CENTERS.map((center) => (
              <circle
                className="five-signal-flow-port"
                cx="0"
                cy={center - FLOW.top}
                r="3"
                key={center}
              />
            ))}
            {OUTCOME_ROW_CENTERS.map((center) => (
              <circle
                className="five-signal-flow-port"
                cx={FLOW.classify.width}
                cy={center - FLOW.top}
                r="3"
                key={center}
              />
            ))}
          </FlowCard>

          <FlowGroup
            {...FLOW.outcomes}
            y={FLOW.top}
            height={FLOW.stageHeight}
            tag="08 · RESULT OUTCOMES"
            title="四类结果分流"
            status="并行 4 行"
            state={snapshot.state}
            name="outcomes"
          >
            <text className="five-signal-flow-node-copy" x="20" y="112">
              按结果类型进入对应诊断线程
            </text>
            {OUTCOME_ROWS.map((outcome, index) => {
              const y = FLOW.outcomeTop + index * FLOW.outcomeStep - FLOW.top;
              return (
                <g
                  className="five-signal-flow-node five-signal-flow-outcome-row state-pending"
                  key={outcome.key}
                >
                  <rect
                    className="five-signal-flow-node-box"
                    x="16"
                    y={y}
                    width="268"
                    height={FLOW.outcomeHeight}
                    rx="8"
                    data-flow-node="outcome"
                  />
                  <circle
                    className={`five-signal-flow-result-dot ${outcome.dotClass}`}
                    cx="36"
                    cy={y + 32}
                    r="4"
                  />
                  <text
                    className="five-signal-flow-outcome-title"
                    x="52"
                    y={y + 36}
                  >
                    {outcome.title}
                  </text>
                  <text
                    className="five-signal-flow-outcome-count"
                    x="264"
                    y={y + 36}
                    textAnchor="end"
                  >
                    {snapshot.counts[outcome.key]}
                  </text>
                  <line
                    className="five-signal-flow-node-rule"
                    x1="36"
                    y1={y + 56}
                    x2="264"
                    y2={y + 56}
                  />
                  <text
                    className="five-signal-flow-outcome-detail"
                    x="36"
                    y={y + 84}
                  >
                    {outcome.detail}
                  </text>
                </g>
              );
            })}
          </FlowGroup>

          <FlowGroup
            {...FLOW.agents}
            y={FLOW.top}
            height={FLOW.stageHeight}
            tag="09 · CODEX THREADS"
            title="四类诊断 Agent"
            status={batch ? "批次证据见下方逐项报告" : `本页已确认 ${knownThreads} 条线程`}
            state={snapshot.state}
            name="agents"
          >
            <text className="five-signal-flow-node-copy" x="20" y="112">
              {batch ? "批次暂不接旧单场景 Agent · 可下载证据" : "4 类入口 · 点击后才启动 / 续接 · 仅结构化证据"}
            </text>
            {OUTCOME_ROWS.map((outcome, index) => {
              const y = FLOW.outcomeTop + index * FLOW.outcomeStep - FLOW.top;
              const state = agentStates[outcome.key];
              const stale = agentIsStale(state, currentEvaluationIds);
              const status = agentStatusCopy(
                state,
                snapshot.counts[outcome.key],
                stale,
              );
              const action = agentActionCopy(
                state,
                snapshot.counts[outcome.key],
                stale,
              );
              return (
                <foreignObject
                  x="16"
                  y={y}
                  width="448"
                  height={FLOW.outcomeHeight}
                  key={outcome.key}
                  data-flow-node="agent"
                >
                  <button
                    className={`five-signal-flow-agent-card is-${state.status}${stale ? " is-stale" : ""}`}
                    type="button"
                    onClick={() => openAgent(outcome.key)}
                    disabled={Boolean(batch) || state.status === "loading"}
                    aria-busy={state.status === "loading"}
                    aria-label={`${outcome.agentTitle}，${status}。${action}`}
                  >
                    <span className="five-signal-flow-agent-topline">
                      <span className="five-signal-flow-agent-kicker">
                        AGENT {String(index + 1).padStart(2, "0")} ·{" "}
                        {outcome.agentKicker}
                      </span>
                      <span className="five-signal-flow-agent-status">
                        {state.status === "loading" ? (
                          <Loader2 size={11} aria-hidden="true" />
                        ) : state.status === "error" ? (
                          <CircleAlert size={11} aria-hidden="true" />
                        ) : (
                          <Sparkles size={11} aria-hidden="true" />
                        )}
                        {status}
                      </span>
                    </span>
                    <span className="five-signal-flow-agent-title">
                      <Bot size={15} aria-hidden="true" />
                      {outcome.agentTitle}
                    </span>
                    <span className="five-signal-flow-agent-action">
                      {action}
                      <ArrowRight size={12} aria-hidden="true" />
                    </span>
                  </button>
                </foreignObject>
              );
            })}
          </FlowGroup>

          <g className="five-signal-flow-connectors" aria-hidden="true">
            {flowArrow(
              `M ${FLOW.dataset.right} ${FLOW.mainY} H ${FLOW.input.x}`,
              hasData ? "complete" : "pending",
              "dataset-input",
            )}
            {flowArrow(`M ${FLOW.input.right} ${FLOW.mainY} H ${FLOW.vcp.x + 24}`, vcpState, "input-vcp")}
            <path className="five-signal-flow-arrow is-architecture" data-flow-link="input-downlink-architecture" d={`M ${FLOW.input.right} ${FLOW.mainY + 40} H ${FLOW.input.right + 32} V ${FLOW.downlinkY + FLOW.vcpCardHeight / 2} H ${FLOW.vcp.x + 24}`} markerEnd={`url(#${arrowMutedId})`} />
            {flowArrow(
              `M ${FLOW.vcp.right - 24} ${FLOW.mainY} H ${FLOW.alignment.x}`,
              analysisState, "input-evaluation",
            )}
            <text className="five-signal-flow-evaluation-label" x={FLOW.vcp.x + 24} y="28">{batch ? "场景测试集 → 条件组 VCP 发送处理（TX）/ 接收处理（RX）→ 实际输出 → 对应计算模型 → 336 项证据归档" : "实际执行：所选上行算法簇 → CLI 输出 → 五路评估 → 单模式 42 项 → 归档；单场景不代表 336 项验收"}</text>
            {METRIC_ROW_CENTERS.map((center, index) => (
              <g key={`metric-connectors-${index}`}>
                {flowArrow(
                  `M ${FLOW.alignment.right} ${center} H ${FLOW.analysis.x + 16}`,
                  analysisState,
                  `align-metric-${index}`,
                )}
                {flowArrow(
                  `M ${FLOW.analysis.right - 16} ${center} H ${FLOW.mapping.x}`,
                  analysisState,
                  `metric-map-${index}`,
                )}
              </g>
            ))}
            {MODE_ROW_CENTERS.map((center, index) => (
              <g key={`mode-connectors-${index}`}>
                {flowArrow(
                  `M ${FLOW.mapping.right} ${center} H ${FLOW.modes.x + 16}`,
                  modeRowState(snapshot, FIVE_SIGNAL_MODES[index]),
                  `map-mode-${index}`,
                )}
                {flowArrow(
                  `M ${FLOW.modes.right - 16} ${center} H ${FLOW.classify.x}`,
                  modeRowState(snapshot, FIVE_SIGNAL_MODES[index]),
                  `mode-classify-${index}`,
                )}
              </g>
            ))}
            {OUTCOME_ROW_CENTERS.map((center, index) => (
              <g key={`outcome-connectors-${index}`}>
                {flowArrow(
                  `M ${FLOW.classify.right} ${center} H ${FLOW.outcomes.x + 16}`,
                  snapshot.counts[OUTCOME_ROWS[index].key]
                    ? "complete"
                    : "pending",
                  `classify-outcome-${index}`,
                )}
                {flowArrow(
                  `M ${FLOW.outcomes.right - 16} ${center} H ${FLOW.agents.x + 16}`,
                  snapshot.counts[OUTCOME_ROWS[index].key]
                    ? "complete"
                    : "pending",
                  `outcome-agent-${index}`,
                )}
              </g>
            ))}
          </g>

          <FlowCard {...FLOW.report} y={428} height={248} tag="10 · ENGINEERING REPORT" title="工程报告 / 归档" subtitle="导出结果快照与已有诊断建议" state="pending" name="report">
            <text className="five-signal-flow-node-copy" x="24" y="140">工程通过 {snapshot.counts.pass} 项</text>
            <text className="five-signal-flow-node-copy" x="24" y="172">异常项携带原因、证据与改进方案</text>
            <foreignObject x="24" y="190" width={FLOW.report.width - 48} height="40">
              <button className="five-signal-flow-report-export" type="button" onClick={exportReport}>导出工程快照 JSON</button>
            </foreignObject>
          </FlowCard>
          {flowArrow(`M ${FLOW.classify.right} 244 H ${FLOW.classify.right + 16} V 168 H ${FLOW.report.x - 32} V 464 H ${FLOW.report.x}`, snapshot.counts.pass ? "complete" : "pending", "pass-report")}
          <text className="five-signal-flow-evaluation-label" x={FLOW.outcomes.x + 24} y="156">工程通过 → 工程报告 / 归档</text>
          <g
            className="five-signal-flow-boundary"
            data-flow-stage="boundary"
            role="group"
            aria-label="正式合规边界，工程代理，正式 ITU-T 合规未测"
            transform={`translate(${FLOW.boundary.x} 428)`}
          >
            <rect width={FLOW.boundary.width} height="248" rx="12" />
            <text className="five-signal-flow-boundary-badge" x="24" y="32">
              11 · COMPLIANCE BOUNDARY
            </text>
            <text className="five-signal-flow-boundary-title" x="24" y="76">
              正式合规边界
            </text>
            <text className="five-signal-flow-boundary-copy" x="24" y="116">
              当前结果：工程代理
            </text>
            <text className="five-signal-flow-boundary-copy" x="24" y="148">
              正式 ITU-T 合规未测
            </text>
            <text className="five-signal-flow-boundary-copy" x="24" y="204">
              正式合规需另行验证
            </text>
          </g>
          <g aria-hidden="true">
            {OUTCOME_ROW_CENTERS.map((center, index) => (
              <path
                className="five-signal-flow-arrow is-warning"
                d={`M ${FLOW.agents.right - 16} ${center} H ${FLOW.agents.right + 32}`}
                key={`boundary-input-${index}`}
              />
            ))}
            <path
              className="five-signal-flow-arrow is-warning"
              d={`M ${FLOW.agents.right + 32} ${OUTCOME_ROW_CENTERS[0]} V ${OUTCOME_ROW_CENTERS.at(-1)} M ${FLOW.agents.right + 32} 552 H ${FLOW.report.x}`}
              markerEnd={`url(#${arrowWarningId})`}
            />
          </g>
          {flowArrow(`M ${FLOW.report.right} 552 H ${FLOW.boundary.x}`, "invalid", "report-boundary")}
        </svg>
      </div>

      <dialog
        ref={dialogRef}
        className="five-signal-agent-dialog"
        aria-labelledby={agentDialogTitleId}
        aria-describedby={agentDialogDescriptionId}
        aria-modal="true"
        onCancel={(event) => {
          event.preventDefault();
          setOpenOutcome(null);
        }}
        onClose={() => setOpenOutcome(null)}
        onClick={(event) => {
          if (event.target === event.currentTarget) setOpenOutcome(null);
        }}
      >
        <div className="five-signal-agent-dialog-head">
          <div>
            <span className="five-signal-agent-dialog-kicker">
              {openOutcomeRow?.agentKicker ?? "CODEX AGENT"}
            </span>
            <h2
              className="five-signal-agent-dialog-title"
              id={agentDialogTitleId}
            >
              {openOutcomeRow?.agentTitle ?? "Codex 诊断 Agent"}
            </h2>
            <p
              className="five-signal-agent-dialog-description"
              id={agentDialogDescriptionId}
            >
              点击后按结果类型启动或续接独立 Codex 线程；只分析持久化报告中的结构化证据，不读取原始音频，也不改变验收判定。
            </p>
          </div>
          <button
            className="five-signal-agent-dialog-close"
            type="button"
            aria-label="关闭 AI 分析"
            onClick={() => setOpenOutcome(null)}
          >
            <X size={17} aria-hidden="true" />
          </button>
        </div>

        {openOutcomeRow && openAgentState ? (
          <>
            <div className="five-signal-agent-dialog-meta">
              <span
                className={`five-signal-agent-dialog-status is-${openAgentState.status}${openAgentStale ? " is-stale" : ""}`}
              >
                {openAgentState.status === "loading"
                  ? "Codex 线程分析中"
                  : openAgentState.status === "error"
                    ? "分析失败"
                    : openAgentStale
                      ? "结果对应旧快照"
                      : openAgentState.status === "empty"
                        ? "当前无此类结果"
                        : openAgentState.status === "ready"
                          ? "分析完成"
                          : "等待分析"}
              </span>
              <span>
                当前分类数量{" "}
                <strong>{snapshot.counts[openOutcomeRow.key]}</strong>
              </span>
              <span>
                输入模式{" "}
                <strong>
                  {openAgentState.result?.mode_count ?? snapshot.availableModes}
                  /8
                </strong>
              </span>
            </div>

            {openAgentState.status === "loading" ? (
              <div className="five-signal-agent-loading" role="status">
                <Loader2 size={20} aria-hidden="true" />
                <div>
                  <strong>正在调用 Codex 线程</strong>
                  <p>
                    正在发送结构化评估证据。首次会创建该结果类型的持久线程，
                    后续分析会续接同一线程。
                  </p>
                </div>
              </div>
            ) : null}

            {openAgentState.status === "error" ? (
              <div className="five-signal-agent-error" role="alert">
                <CircleAlert size={20} aria-hidden="true" />
                <div>
                  <strong>Codex 分析未完成</strong>
                  <p>{openAgentState.error}</p>
                </div>
              </div>
            ) : null}

            {openAgentState.result ? (
              <div className="five-signal-agent-result">
                <section className="five-signal-agent-summary">
                  <div className="five-signal-agent-summary-head">
                    <span>
                      <Sparkles size={15} aria-hidden="true" />
                      分析结论
                    </span>
                    <span>
                      {openAgentState.result.cached ? "缓存结果" : "新结果"}
                      {" · "}
                      {openAgentState.result.item_count} 项证据
                    </span>
                  </div>
                  <p>{openAgentState.result.summary}</p>
                  <div className="five-signal-agent-thread">
                    <span>Codex 线程 · {openAgentState.result.generated_at}</span>
                    <code>
                      {openAgentState.result.thread_id ??
                        "当前空态尚未创建线程"}
                    </code>
                    {openAgentState.result.thread_created ? (
                      <em>本次新建</em>
                    ) : openAgentState.result.thread_id ? (
                      <em>持久复用</em>
                    ) : null}
                  </div>
                </section>

                <section className="five-signal-agent-section">
                  <div className="five-signal-agent-section-title">
                    <span>可能根因</span>
                    <strong>{openAgentState.result.root_causes.length}</strong>
                  </div>
                  {openAgentState.result.root_causes.length ? (
                    <ol className="five-signal-agent-causes">
                      {openAgentState.result.root_causes.map((cause, index) => (
                        <li key={`${cause.cause}-${index}`}>
                          <div>
                            <span
                              className={`five-signal-agent-confidence is-${cause.confidence}`}
                            >
                              {cause.confidence === "high"
                                ? "高置信"
                                : cause.confidence === "medium"
                                  ? "中置信"
                                  : "低置信"}
                            </span>
                            <p>{cause.cause}</p>
                          </div>
                          <div className="five-signal-agent-evidence">
                            {cause.evidence_refs.map((ref) => (
                              <code key={ref}>{ref}</code>
                            ))}
                          </div>
                        </li>
                      ))}
                    </ol>
                  ) : (
                    <p className="five-signal-agent-empty-copy">
                      当前证据不足或没有此类结果，Codex 没有生成具体根因。
                    </p>
                  )}
                </section>

                <section className="five-signal-agent-section">
                  <div className="five-signal-agent-section-title">
                    <span>应对方案</span>
                    <strong>{openAgentState.result.actions.length}</strong>
                  </div>
                  {openAgentState.result.actions.length ? (
                    <ol className="five-signal-agent-actions">
                      {openAgentState.result.actions.map((action, index) => (
                        <li key={`${action.action}-${index}`}>
                          <span
                            className={`five-signal-agent-priority is-${action.priority.toLowerCase()}`}
                          >
                            {action.priority}
                          </span>
                          <div>
                            <p>{action.action}</p>
                            <small>验证：{action.validation}</small>
                            <div className="five-signal-agent-evidence">
                              {action.evidence_refs.map((ref) => (
                                <code key={ref}>{ref}</code>
                              ))}
                            </div>
                          </div>
                        </li>
                      ))}
                    </ol>
                  ) : (
                    <p className="five-signal-agent-empty-copy">
                      当前没有可执行的处置动作。
                    </p>
                  )}
                </section>

                <section className="five-signal-agent-limitations">
                  <div className="five-signal-agent-section-title">
                    <span>边界与限制</span>
                  </div>
                  <ul>
                    {openAgentState.result.limitations.map(
                      (limitation, index) => (
                        <li key={`${limitation}-${index}`}>{limitation}</li>
                      ),
                    )}
                  </ul>
                </section>
              </div>
            ) : null}

            {openAgentStale ? (
              <p className="five-signal-agent-stale-note">
                页面轮询到了新的评估快照。当前结果保留用于对照，重新分析后会续接同一
                Codex 线程并更新结论。
              </p>
            ) : null}

            <div className="five-signal-agent-dialog-actions">
              <button
                className="five-signal-agent-refresh"
                type="button"
                onClick={() => void analyzeAgent(openOutcomeRow.key, true)}
                disabled={openAgentState.status === "loading"}
              >
                {openAgentState.status === "loading" ? (
                  <Loader2 size={14} aria-hidden="true" />
                ) : (
                  <RefreshCw size={14} aria-hidden="true" />
                )}
                重新分析
              </button>
              <button
                className="five-signal-agent-dialog-secondary"
                type="button"
                onClick={() => setOpenOutcome(null)}
              >
                关闭
              </button>
            </div>
          </>
        ) : null}
      </dialog>
    </section>
  );
}
