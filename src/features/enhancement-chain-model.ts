import catalog from "./enhancement-chain-configs.json" with { type: "json" };
import type { PipelineCaseGroup } from "./PipelineEvaluation";

export type StageId = "bf" | "aec" | "ns" | "eq" | "agc" | "drc" | "limiter";
export type ChainOption = { id: string; label: string; flags: string[]; note: string; candidate?: boolean };
export type ChainStage = { id: StageId; label: string; tab: string; description: string; source: string; options: ChainOption[] };
export type ChainConfiguration = {
  profile: "general" | "vcp" | "vcp_joint";
  selection: Record<StageId, string>;
  extra_flags?: string[];
};
export type ChainPreset = ChainConfiguration & {
  id: string;
  label: string;
  description: string;
  recommended?: boolean;
};
export const chainStages = catalog.stages as ChainStage[];
export const chainPresets = catalog.presets as ChainPreset[];
// Frozen results are a separate, closed set; adding an executable preset must not relabel old evidence.
export const chainEvidencePresets = chainPresets.filter((preset) => (catalog.evidence_preset_ids as string[]).includes(preset.id));
export const CHAIN_STORAGE_KEY = "itu-enhancement-chain-v1";
export const CHAIN_REPORT_URL = "/evaluation/enhancement-chain/report.json";

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

export function restoreConfiguration(value: string | null): ChainConfiguration {
  const fallback = chainPresets.find((preset) => preset.id === "all_serial")!;
  try {
    const parsed = JSON.parse(value ?? "null") as ChainConfiguration | null;
    if (!parsed || !["vcp", "vcp_joint", "general"].includes(parsed.profile) || !parsed.selection ||
      !chainStages.every((stage) => stage.options.some((option) => option.id === parsed.selection[stage.id]))) return fallback;
    // Persist only supported catalog choices. Never accept arbitrary cached CLI flags.
    if (parsed.profile === "general") return matchingPreset(parsed)?.id === "array_control" ? chainPresets[0] : fallback;
    return { profile: parsed.profile, selection: Object.fromEntries(chainStages.map((stage) => [stage.id, parsed.selection[stage.id]])) as Record<StageId, string> };
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
  return report;
}

export function chainExecutionIssue(configuration: ChainConfiguration): string | null {
  if (configuration.profile === "vcp_joint" &&
    (configuration.selection.aec !== "speex" || configuration.selection.ns !== "super_gaussian")) {
    return "联合方案需要 Speex 回声消除与 Super-Gaussian 降噪。请恢复这两级，或选择 VCP 默认后再组合。";
  }
  return null;
}
