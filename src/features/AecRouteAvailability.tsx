import { useCallback, useEffect, useState } from "react";
import { AlertTriangle, CheckCircle2, LoaderCircle, RefreshCw } from "lucide-react";
import {
  COMPUTER_PF_CPU_ROUTE_ID,
  isCapabilityAecRoute,
  isComputerPfCpuRoute,
  isLocalVqeRoute,
  loadAecCapabilities,
  resolveAecRouteCapability,
  resolveAecRouteCapabilityFailure,
} from "./aec-route-capability";
import type { AecCapabilityEntry } from "./aec-route-capability";

export type AecRouteCapabilityStatus = "idle" | "loading" | "available" | "unavailable" | "error";

export type AecRouteCapabilityState = {
  routeId?: string;
  status: AecRouteCapabilityStatus;
  reason: string;
  blockReason: string | null;
  submitBlocked: boolean;
  retry: () => void;
  identity?: {
    modelId: string;
    modelSha256: string;
    deviceId: string;
  };
};

type QueryState = {
  routeId?: string;
  status: "loading" | "available" | "unavailable" | "error";
  reason: string;
  entry?: AecCapabilityEntry;
};

const noRetry = () => undefined;

function blockedReason(routeId: string | undefined, status: QueryState["status"], reason: string): string {
  if (isComputerPfCpuRoute(routeId)) {
    if (status === "loading") return "正在检查 CPU Complex_PF 离线实验路由与本地训练权重，暂不允许提交。";
    if (status === "unavailable") return `CPU Complex_PF 离线实验路由不可用：${reason}`;
    return `无法确认 CPU Complex_PF 离线实验路由可用：${reason} 当前阻止提交。`;
  }
  if (isLocalVqeRoute(routeId)) {
    if (status === "loading") return "正在检查 LocalVQE 203K CPU 模型与共享库身份，暂不允许提交。";
    if (status === "unavailable") return `LocalVQE CPU 路由不可用：${reason}`;
    return `无法确认 LocalVQE CPU 路由可用：${reason} 当前阻止提交，避免误回退到 Speex/VCP。`;
  }
  if (status === "loading") return "正在检查 DSP 路由能力，暂不允许提交。";
  if (status === "unavailable") return `DSP 路由不可用：${reason}`;
  return `无法确认 DSP 路由可用：${reason} 当前阻止提交，避免误回退到 CPU。`;
}

export function useAecRouteCapability(routeId: string | undefined): AecRouteCapabilityState {
  const [query, setQuery] = useState<QueryState>({ status: "loading", reason: "" });
  const [attempt, setAttempt] = useState(0);
  const capabilityRoute = isCapabilityAecRoute(routeId);

  useEffect(() => {
    if (!capabilityRoute) return;
    const controller = new AbortController();
    let active = true;
    setQuery({ routeId, status: "loading", reason: "" });
    loadAecCapabilities(controller.signal, attempt > 0, routeId)
      .then((document) => {
        if (!active || controller.signal.aborted) return;
        const resolution = resolveAecRouteCapability(document, routeId);
        setQuery({ routeId, status: resolution.status, reason: resolution.reason, entry: resolution.entry });
      })
      .catch((error: unknown) => {
        if (!active || controller.signal.aborted) return;
        const failure = resolveAecRouteCapabilityFailure(error, routeId);
        setQuery({ routeId, status: failure.status, reason: failure.reason });
      });
    return () => {
      active = false;
      controller.abort();
    };
  }, [attempt, capabilityRoute, routeId]);

  const retry = useCallback(() => setAttempt((value) => value + 1), []);
  if (!capabilityRoute) {
    return { routeId, status: "idle", reason: "", blockReason: null, submitBlocked: false, retry: noRetry };
  }

  const current = query.routeId === routeId ? query : { status: "loading" as const, reason: "" };
  const identity = (isComputerPfCpuRoute(routeId) || isLocalVqeRoute(routeId)) &&
    current.entry?.available ? {
    modelId: String(current.entry.model_id),
    modelSha256: String(current.entry.model_sha256),
    deviceId: String(current.entry.device_id),
  } : undefined;
  return {
    routeId,
    status: current.status,
    reason: current.reason,
    blockReason: current.status === "available" ? null : blockedReason(routeId, current.status, current.reason),
    submitBlocked: current.status !== "available",
    retry,
    identity,
  };
}

function statusText(state: AecRouteCapabilityState): string {
  if (state.routeId === COMPUTER_PF_CPU_ROUTE_ID) {
    if (state.status === "loading") return "正在检查 CPU Complex_PF 离线实验路由与本地训练权重 capability…";
    if (state.status === "available") {
      const identity = state.identity
        ? ` 模型 ${state.identity.modelId} · SHA ${state.identity.modelSha256.slice(0, 12)}… · 设备 ${state.identity.deviceId}。`
        : "";
      return `CPU Complex_PF 离线实验路由可用：后端已报告本地训练权重身份满足准入。${identity}`;
    }
    if (state.status === "unavailable") return `CPU Complex_PF 离线实验路由不可用：${state.reason}`;
    if (state.status === "error") return `CPU Complex_PF capability 检查失败：${state.reason}`;
  }
  if (state.routeId === "localvqe_aec200k") {
    if (state.status === "loading") return "正在检查 LocalVQE 203K CPU 模型与共享库身份…";
    if (state.status === "available") {
      const identity = state.identity
        ? ` 模型 ${state.identity.modelId} · SHA ${state.identity.modelSha256.slice(0, 12)}… · 设备 ${state.identity.deviceId}。`
        : "";
      return `LocalVQE CPU 路由可用：后端已报告固定模型、共享库和 16 kHz 流式契约满足准入。${identity}`;
    }
    if (state.status === "unavailable") return `LocalVQE CPU 路由不可用：${state.reason}`;
    if (state.status === "error") return `LocalVQE capability 检查失败：${state.reason}`;
  }
  if (state.status === "loading") return "正在检查 HiFi DSP provider 与 Complex_PF 模型 capability…";
  if (state.status === "available") return "DSP 路由可用：后端已报告 HiFi provider 与 Complex_PF 模型满足准入。";
  if (state.status === "unavailable") return `DSP 路由不可用：${state.reason}`;
  if (state.status === "error") return `DSP capability 检查失败：${state.reason}`;
  return "";
}

export function AecRouteAvailabilityView({
  state,
  compact = false,
}: {
  state: AecRouteCapabilityState;
  compact?: boolean;
}) {
  if (state.status === "idle") return null;
  const problem = state.status === "unavailable" || state.status === "error";
  const Icon = state.status === "loading"
    ? LoaderCircle
    : state.status === "available"
      ? CheckCircle2
      : AlertTriangle;
  return (
    <p
      className={problem ? "field-error" : "config-helper"}
      role={problem ? "alert" : "status"}
      aria-live={problem ? "assertive" : "polite"}
      style={{
        display: "flex",
        flexWrap: "wrap",
        alignItems: "center",
        gap: "6px 8px",
        margin: compact ? 0 : undefined,
      }}
    >
      <Icon size={14} className={state.status === "loading" ? "spin" : undefined} aria-hidden="true" />
      <span>{statusText(state)}</span>
      {state.status !== "loading" ? (
        <button type="button" className="text-link" onClick={state.retry}>
          <RefreshCw size={13} aria-hidden="true" />
          重新检查
        </button>
      ) : null}
    </p>
  );
}

export function AecRouteAvailability({
  routeId,
  compact = false,
}: {
  routeId?: string;
  compact?: boolean;
}) {
  return <AecRouteAvailabilityView state={useAecRouteCapability(routeId)} compact={compact} />;
}
