import catalog from "./enhancement-chain-configs.json" with { type: "json" };
import type { PipelineCaseGroup } from "./PipelineEvaluation";

export type StageId = "bf" | "aec" | "ns" | "eq" | "agc" | "drc" | "limiter";
export type ChainOption = { id: string; label: string; flags: string[]; note: string; requires?: string[]; candidate?: boolean };
export type ChainStage = { id: StageId; label: string; tab: string; description: string; source: string; options: ChainOption[] };
export type ChainConfiguration = {
  profile: "general" | "vcp" | "vcp_joint" | "vcp_unified";
  selection: Record<StageId, string>;
  extra_flags?: string[];
};
export type ChainPreset = ChainConfiguration & {
  id: string;
  label: string;
  description: string;
  candidate?: boolean;
  recommended?: boolean;
};
export const chainStages = catalog.stages as ChainStage[];
export const chainPresets = catalog.presets as ChainPreset[];
// Frozen results are a separate, closed set; adding an executable preset must not relabel old evidence.
export const chainEvidencePresets = chainPresets.filter((preset) => (catalog.evidence_preset_ids as string[]).includes(preset.id));
export const CHAIN_STORAGE_KEY = "itu-enhancement-chain-v1";
export const CHAIN_REPORT_URL = "/evaluation/enhancement-chain/report.json";
const LEGACY_CPU_AEC_ID = "speex_fused";
const CPU_AEC_ID = "speex_fused_cpu";
const SERVER_EXECUTED_NS_IDS = new Set([
  "cofilite",
  "dpdfnet2",
  "dpdfnet2_48k_hr",
  "hybrid_v2_pitch",
]);

export type ChainVariant = {
  id: string;
  label: string;
  description: string;
  profile: string;
  selection: Record<StageId, string>;
  flags: string[];
  cases: number;
  pass: number;
  changed: number;
  mean_sisdr_delta_db: number;
  mean_stoi_delta: number;
  mean_noise_delta_db: number;
  noise_censored_cases: number;
  mean_double_sisdr_delta_db: number | null;
  mean_double_stoi_delta: number | null;
  double_cases: number;
  far_cases: number;
  mean_far_delta_db: number | null;
  far_censored_cases: number;
  clipped_cases: number;
  limited_samples: number;
  mean_rtf: number;
  failure_counts: Record<string, number>;
  caseGroups: PipelineCaseGroup[];
};

export type ChainReport = {
  schema_version: number;
  source: string;
  created_at: string;
  case_count: number;
  cli_runs: number;
  standard_compliance: false;
  cli_sha256: string;
  config_sha256: string;
  results_sha256: string;
  manifest_sha256: string;
  negative_control: { cases: number; rejected: number };
  variants: ChainVariant[];
};

export function optionFor(stage: ChainStage, configuration: ChainConfiguration): ChainOption {
  return stage.options.find((option) => option.id === configuration.selection[stage.id]) ?? stage.options[0];
}

export type ChainSelection = {
  ordinal: number;
  stage: ChainStage;
  option: ChainOption | null;
};

export function chainSelections(configuration: ChainConfiguration): ChainSelection[] {
  return chainStages.map((stage, index) => ({
    ordinal: index + 1,
    stage,
    option: stage.options.find((option) => option.id === configuration.selection[stage.id]) ?? null,
  }));
}

export function chainFlags(configuration: ChainConfiguration): string[] {
  return ["--profile", configuration.profile, "--no-rx-processing",
    ...(configuration.profile === "vcp_joint" ? catalog.joint_stage_defaults.drc : []),
    ...chainStages.flatMap((stage) => optionFor(stage, configuration).flags),
    ...(configuration.extra_flags ?? []),
  ];
}

export function matchingPreset(configuration: ChainConfiguration): ChainPreset | undefined {
  return chainPresets.find((preset) => preset.profile === configuration.profile &&
    chainStages.every((stage) => preset.selection[stage.id] === configuration.selection[stage.id]) &&
    JSON.stringify(preset.extra_flags ?? []) === JSON.stringify(configuration.extra_flags ?? []));
}

export function matchingEvidence(configuration: ChainConfiguration, report: ChainReport | null): ChainVariant | undefined {
  const preset = matchingPreset(configuration);
  return report?.variants.find((row) => row.id === preset?.id &&
    JSON.stringify(row.flags) === JSON.stringify(chainFlags(configuration)) &&
    chainStages.every((stage) => row.selection[stage.id] === configuration.selection[stage.id]));
}

function canonicalizeFrozenAecIdentity(row: ChainVariant): ChainVariant {
  // This migration covers the rename-only CPU backend. Keep the alias local
  // to frozen evidence parsing; a future DSP backend must never inherit it.
  if (row.selection?.aec !== LEGACY_CPU_AEC_ID) return row;
  return {
    ...row,
    selection: { ...row.selection, aec: CPU_AEC_ID },
    flags: row.flags.map((flag, index, values) =>
      flag === LEGACY_CPU_AEC_ID
        && values[index - 1] === "--aec-backend"
        ? CPU_AEC_ID
        : flag),
  };
}

export function restoreConfiguration(value: string | null): ChainConfiguration {
  const fallback = chainPresets.find((preset) => preset.id === "all_serial")!;
  try {
    const parsed = JSON.parse(value ?? "null") as ChainConfiguration | null;
    if (!parsed || !["vcp", "vcp_joint", "general", "vcp_unified"].includes(parsed.profile) || !parsed.selection ||
      !chainStages.every((stage) => stage.options.some((option) => option.id === parsed.selection[stage.id]))) return fallback;
    // Persist only supported catalog choices. Never accept arbitrary cached CLI flags.
    if (parsed.profile === "general") return matchingPreset(parsed)?.id === "array_control" ? chainPresets[0] : fallback;
    const selection = Object.fromEntries(
      chainStages.map((stage) => [stage.id, parsed.selection[stage.id]]),
    ) as Record<StageId, string>;
    const trustedExtra = chainPresets.find(
      (preset) =>
        preset.profile === parsed.profile &&
        chainStages.every((stage) => preset.selection[stage.id] === selection[stage.id]),
    )?.extra_flags;
    return {
      profile: parsed.profile,
      selection,
      ...(trustedExtra ? { extra_flags: trustedExtra } : {}),
    };
  } catch {
    return fallback;
  }
}

export function changeStage(configuration: ChainConfiguration, stage: StageId, option: string): ChainConfiguration {
  if (!chainStages.find((item) => item.id === stage)?.options.some((item) => item.id === option)) return configuration;
  return { profile: configuration.profile === "general" ? "vcp" : configuration.profile, selection: { ...configuration.selection, [stage]: option } };
}

const quote = (value: string) => `'${value.replace(/'/g, "'\\''")}'`;

export function chainCommand(configuration: ChainConfiguration): string {
  if (configuration.selection.ns === "cofilite_native16") {
    return native16ChainCommand(configuration);
  }
  if (configuration.selection.ns === "cofilite") {
    return [
      "# CoFi-Lite 由服务器 wrapper 创建持久模型进程，在实际 AEC 后每 160 样本在线处理，卷积与 GRU 状态跨帧保留。",
      "# 固定 NS 缓冲 480 样本（30 ms）；请使用界面运行此算法簇，进程通信句柄由服务器管理。",
    ].join("\n") + "\n";
  }
  if (SERVER_EXECUTED_NS_IDS.has(configuration.selection.ns)) {
    return [
      "# NS 由服务器 wrapper 在真实 AEC 边界执行模型推理，并用 --post-ns-pcm 回放同一 AEC 尾部；模型身份以运行报告为准。",
      "# 请使用界面运行此算法簇；边界捕获参数不是完整算法簇复现命令。",
    ].join("\n") + "\n";
  }
  const flags = chainFlags(configuration);
  const inputs = ["--mic", "capture.wav"];
  if (configuration.selection.aec !== "off") inputs.push("--echo", "actual-playback.wav");
  if (configuration.selection.bf !== "off") inputs.push("--beam-delays", "<逐通道标定延迟，以逗号分隔>");
  return "build/host/cpp/itu_audio_cli \\\n  " + [...flags, ...inputs, "--out", "build/chain-replay/result"]
    .map((part) => part.startsWith("--") ? part : quote(part)).join(" ") + "\n";
}

export function parseChainReport(value: unknown): ChainReport {
  const report = value as ChainReport;
  if (report?.schema_version !== 1 || report.standard_compliance !== false ||
    report.case_count !== 72 || report.cli_runs !== 72 * chainEvidencePresets.length || !Array.isArray(report.variants) ||
    report.variants.length !== chainEvidencePresets.length || new Set(report.variants.map((item) => item.id)).size !== chainEvidencePresets.length ||
    !report.variants.every((row) => chainEvidencePresets.some((preset) => preset.id === row.id) &&
      row.cases === report.case_count && Number.isInteger(row.pass) && row.pass >= 0 && row.pass <= row.cases &&
      Array.isArray(row.flags) && Array.isArray(row.caseGroups)) ||
    report.negative_control?.cases !== report.case_count || report.negative_control.rejected !== report.case_count) {
    throw new Error("综合评测文件不完整或负对照校验失败，请重新发布完整评测。");
  }
  return {
    ...report,
    variants: report.variants.map(canonicalizeFrozenAecIdentity),
  };
}

export function chainExecutionIssue(configuration: ChainConfiguration): string | null {
  if (configuration.profile === "vcp_joint" &&
    (configuration.selection.aec !== "speex" || configuration.selection.ns !== "super_gaussian")) {
    return "联合方案需要 Speex 回声消除与 Super-Gaussian 降噪。请恢复这两级，或选择 VCP 默认后再组合。";
  }
  if (configuration.selection.ns === "dpdfnet2_48k_hr" &&
    configuration.selection.aec !== "aec3") {
    return "DPDFNet2 48 kHz HR 质量路线必须使用 WebRTC AEC3 上游。";
  }
  return null;
}


// Native16 registration: independent preset; frozen evidence remains unchanged.
import { native16Preset, native16NsOption, NATIVE16_NS_ID } from "./enhancement-chain-native16.ts";
chainStages.find((stage) => stage.id === "ns")!.options.push(native16NsOption as ChainOption);
chainPresets.push(native16Preset as ChainPreset);
SERVER_EXECUTED_NS_IDS.add(NATIVE16_NS_ID);
export const native16ChainCommand = (configuration: ChainConfiguration): string => configuration.selection.ns === NATIVE16_NS_ID
  ? "# 原生16 ms：服务器持久模型进程每次处理256个16kHz样本，保持卷积/GRU状态。\n# AEC/NS各256样本固定延迟；需使用实际原生16 ms执行器和回执，不能交给10ms CLI。\n"
  : chainCommand(configuration);
