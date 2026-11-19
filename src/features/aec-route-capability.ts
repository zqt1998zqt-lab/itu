export const AEC_CAPABILITIES_URL = "/api/alignment/aec-backends";
export const AEC_CAPABILITY_SCHEMA = "itu-aec-capabilities-v1";
export const DSP_AEC_ROUTE_ID = "speex_fused_dsp";
export const COMPUTER_PF_CPU_ROUTE_ID = "computer_pf_cpu";
export const LOCALVQE_AEC_ROUTE_ID = "localvqe_aec200k";
export const LOCALVQE_NEURAL_DELAY_SAMPLES_16K = 256;
export const LOCALVQE_HOP_ADAPTER_DELAY_SAMPLES_16K = 224;
export const LOCALVQE_ALGORITHM_DELAY_SAMPLES_16K =
  LOCALVQE_NEURAL_DELAY_SAMPLES_16K + LOCALVQE_HOP_ADAPTER_DELAY_SAMPLES_16K;
export const CAPABILITY_AEC_ROUTE_IDS = [
  DSP_AEC_ROUTE_ID,
  COMPUTER_PF_CPU_ROUTE_ID,
  LOCALVQE_AEC_ROUTE_ID,
] as const;

const AEC_ARCHITECTURES = {
  webrtc: "webrtc_aec3",
  speex: "speex_mdf",
  speex_fused_cpu: "speex_mdf_gcc_phat_nlms",
  speex_fused_dsp: "mdf_complex_pf",
  computer_pf_cpu: "mdf_complex_pf_cpu",
  localvqe_aec200k: "localvqe_daf_neural_mask",
} as const;

const MODEL_ID_RE = /^[A-Za-z0-9_-]{1,63}$/;
const SHA256_RE = /^[0-9a-fA-F]{64}$/;

export type AecBackendId = keyof typeof AEC_ARCHITECTURES;
export type CapabilityAecRouteId = typeof CAPABILITY_AEC_ROUTE_IDS[number];

export type AecCapabilityEntry = {
  id: AecBackendId;
  available: boolean;
  architecture: string;
  reason: string;
  [key: string]: unknown;
};

export type AecCapabilitiesDocument = {
  schema: typeof AEC_CAPABILITY_SCHEMA;
  backends: AecCapabilityEntry[];
  [key: string]: unknown;
};

export type AecRouteCapabilityResolution =
  | { status: "available"; entry: AecCapabilityEntry; reason: "" }
  | { status: "unavailable"; entry?: AecCapabilityEntry; reason: string };
export type DspCapabilityResolution = AecRouteCapabilityResolution;

const CAPABILITY_CACHE_MS = 5_000;
let cachedCapabilities: { expiresAt: number; document: AecCapabilitiesDocument } | null = null;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function invalid(detail: string): never {
  throw new Error(`AEC capabilities 响应无效：${detail}`);
}

function validateModelIdentity(raw: Record<string, unknown>, prefix: string): void {
  if (typeof raw.device_id !== "string" || !raw.device_id.trim()) {
    invalid(`${prefix}.device_id 必须是非空字符串`);
  }
  if (typeof raw.model_id !== "string" || !MODEL_ID_RE.test(raw.model_id)) {
    invalid(`${prefix}.model_id 必须匹配 [A-Za-z0-9_-]{1,63}`);
  }
  for (const field of ["model_sha256", "frontend_sha256", "provider_sha256"] as const) {
    if (typeof raw[field] !== "string" || !SHA256_RE.test(raw[field])) {
      invalid(`${prefix}.${field} 必须是 64 位十六进制 SHA-256`);
    }
  }
}

function validateAvailableDsp(raw: Record<string, unknown>, prefix: string): void {
  if (raw.execution_provider !== "hifi_dsp") {
    invalid(`${prefix}.execution_provider 必须是 hifi_dsp`);
  }
  validateModelIdentity(raw, prefix);
  if (typeof raw.delay_samples !== "number" || !Number.isInteger(raw.delay_samples) ||
    raw.delay_samples < 212 || raw.delay_samples > 16_000 || raw.delay_samples % 160 !== 0) {
    invalid(`${prefix}.delay_samples 必须是 212 到 16000 之间且能被 160 整除的整数`);
  }
}

function validateAvailableComputerPfCpu(raw: Record<string, unknown>, prefix: string): void {
  if (raw.execution_provider !== "cpu") {
    invalid(`${prefix}.execution_provider 必须是 cpu`);
  }
  if (raw.device_id !== "host_cpu") {
    invalid(`${prefix}.device_id 必须是 host_cpu`);
  }
  validateModelIdentity(raw, prefix);
  if (typeof raw.delay_samples !== "number" || !Number.isSafeInteger(raw.delay_samples) || raw.delay_samples < 0) {
    invalid(`${prefix}.delay_samples 必须是非负整数`);
  }
}

function validateAvailableLocalVqe(raw: Record<string, unknown>, prefix: string): void {
  if (raw.execution_provider !== "cpu") {
    invalid(`${prefix}.execution_provider 必须是 cpu`);
  }
  if (raw.device_id !== "host_cpu") {
    invalid(`${prefix}.device_id 必须是 host_cpu`);
  }
  validateModelIdentity(raw, prefix);
  if (raw.delay_samples !== LOCALVQE_ALGORITHM_DELAY_SAMPLES_16K) {
    invalid(`${prefix}.delay_samples 必须是整数 ${LOCALVQE_ALGORITHM_DELAY_SAMPLES_16K}`);
  }
  if (raw.model_codec_delay_samples_16k !== 256) {
    invalid(`${prefix}.model_codec_delay_samples_16k 必须是整数 256`);
  }
  if (raw.linear_frontend_delay_samples_16k !== 0) {
    invalid(`${prefix}.linear_frontend_delay_samples_16k 必须是整数 0`);
  }
}

export function isDspAecRoute(routeId: string | null | undefined): routeId is typeof DSP_AEC_ROUTE_ID {
  return routeId === DSP_AEC_ROUTE_ID;
}

export function isComputerPfCpuRoute(routeId: string | null | undefined): routeId is typeof COMPUTER_PF_CPU_ROUTE_ID {
  return routeId === COMPUTER_PF_CPU_ROUTE_ID;
}

export function isLocalVqeRoute(routeId: string | null | undefined): routeId is typeof LOCALVQE_AEC_ROUTE_ID {
  return routeId === LOCALVQE_AEC_ROUTE_ID;
}

export function isCapabilityAecRoute(routeId: string | null | undefined): routeId is CapabilityAecRouteId {
  return routeId === DSP_AEC_ROUTE_ID ||
    routeId === COMPUTER_PF_CPU_ROUTE_ID ||
    routeId === LOCALVQE_AEC_ROUTE_ID;
}

export function parseAecCapabilities(value: unknown): AecCapabilitiesDocument {
  if (!isRecord(value)) invalid("顶层必须是对象");
  if (value.schema !== AEC_CAPABILITY_SCHEMA) {
    invalid(`schema 必须是 ${AEC_CAPABILITY_SCHEMA}`);
  }
  if (!Array.isArray(value.backends)) invalid("backends 必须是数组");

  const seen = new Set<string>();
  const backends = value.backends.map((raw, index): AecCapabilityEntry => {
    const prefix = `backends[${index}]`;
    if (!isRecord(raw)) invalid(`${prefix} 必须是对象`);
    if (typeof raw.id !== "string" || !raw.id.trim()) invalid(`${prefix}.id 必须是非空字符串`);
    if (!(raw.id in AEC_ARCHITECTURES)) invalid(`${prefix}.id 未知：${raw.id}`);
    if (seen.has(raw.id)) invalid(`backend id 重复：${raw.id}`);
    seen.add(raw.id);
    if (typeof raw.available !== "boolean") invalid(`${prefix}.available 必须是布尔值`);
    if (typeof raw.architecture !== "string" || !raw.architecture.trim()) {
      invalid(`${prefix}.architecture 必须是非空字符串`);
    }
    if (typeof raw.reason !== "string") invalid(`${prefix}.reason 必须是字符串`);
    const expectedArchitecture = AEC_ARCHITECTURES[raw.id as AecBackendId];
    if (raw.architecture !== expectedArchitecture) {
      invalid(`${prefix}.architecture 必须是 ${expectedArchitecture}，实际为 ${raw.architecture}`);
    }
    if (raw.id === DSP_AEC_ROUTE_ID && raw.available) validateAvailableDsp(raw, prefix);
    if (raw.id === COMPUTER_PF_CPU_ROUTE_ID && raw.available) validateAvailableComputerPfCpu(raw, prefix);
    if (raw.id === LOCALVQE_AEC_ROUTE_ID && raw.available) validateAvailableLocalVqe(raw, prefix);
    return {
      ...raw,
      id: raw.id as AecBackendId,
      available: raw.available,
      architecture: raw.architecture,
      reason: raw.reason,
    };
  });

  return { ...value, schema: AEC_CAPABILITY_SCHEMA, backends };
}

function capabilityLabel(routeId: CapabilityAecRouteId): string {
  if (routeId === DSP_AEC_ROUTE_ID) return "DSP";
  if (routeId === LOCALVQE_AEC_ROUTE_ID) return "LocalVQE CPU";
  return "CPU Complex_PF";
}

export function resolveAecRouteCapability(
  document: AecCapabilitiesDocument,
  routeId: CapabilityAecRouteId,
): AecRouteCapabilityResolution {
  const entry = document.backends.find((item) => item.id === routeId);
  if (!entry) {
    return {
      status: "unavailable",
      reason: `capability 响应缺少 ${routeId} 条目，无法确认 ${capabilityLabel(routeId)} 可用。`,
    };
  }
  if (!entry.available) {
    return {
      status: "unavailable",
      entry,
      reason: entry.reason.trim() || `CLI 未提供 ${capabilityLabel(routeId)} 不可用原因。`,
    };
  }
  return { status: "available", entry, reason: "" };
}

export function resolveDspCapability(document: AecCapabilitiesDocument): DspCapabilityResolution {
  return resolveAecRouteCapability(document, DSP_AEC_ROUTE_ID);
}

export function resolveComputerPfCpuCapability(document: AecCapabilitiesDocument): AecRouteCapabilityResolution {
  return resolveAecRouteCapability(document, COMPUTER_PF_CPU_ROUTE_ID);
}

export function resolveDspCapabilityFailure(error: unknown): { status: "error"; reason: string } {
  return {
    status: "error",
    reason: error instanceof Error && error.message ? error.message : "DSP capability 请求失败。",
  };
}

export function resolveAecRouteCapabilityFailure(
  error: unknown,
  routeId: CapabilityAecRouteId,
): { status: "error"; reason: string } {
  if (routeId === DSP_AEC_ROUTE_ID) return resolveDspCapabilityFailure(error);
  if (routeId === LOCALVQE_AEC_ROUTE_ID) {
    return {
      status: "error",
      reason: error instanceof Error && error.message ? error.message : "LocalVQE CPU capability 请求失败。",
    };
  }
  return {
    status: "error",
    reason: error instanceof Error && error.message ? error.message : "CPU Complex_PF capability 请求失败。",
  };
}

export async function loadAecCapabilities(
  signal?: AbortSignal,
  forceRefresh = false,
  routeId?: CapabilityAecRouteId,
): Promise<AecCapabilitiesDocument> {
  signal?.throwIfAborted();
  const now = Date.now();
  if (!forceRefresh && cachedCapabilities && cachedCapabilities.expiresAt > now) {
    return cachedCapabilities.document;
  }

  const response = await fetch(AEC_CAPABILITIES_URL, {
    method: "GET",
    cache: "no-store",
    signal,
  });
  if (!response.ok) {
    let detail = "";
    try {
      const payload: unknown = await response.json();
      if (isRecord(payload) && typeof payload.detail === "string") detail = payload.detail;
    } catch {
      /* The HTTP status remains the authoritative failure. */
    }
    const label = routeId === DSP_AEC_ROUTE_ID
      ? "DSP capability"
      : routeId === COMPUTER_PF_CPU_ROUTE_ID
        ? "CPU Complex_PF capability"
        : routeId === LOCALVQE_AEC_ROUTE_ID
          ? "LocalVQE CPU capability"
          : "AEC capability";
    throw new Error(`${label} 请求失败（HTTP ${response.status}）${detail ? `：${detail}` : ""}`);
  }
  const document = parseAecCapabilities(await response.json());
  cachedCapabilities = { document, expiresAt: Date.now() + CAPABILITY_CACHE_MS };
  return document;
}
