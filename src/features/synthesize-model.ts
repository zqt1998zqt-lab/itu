import { chainExecutionIssue, chainFlags, chainStages, restoreConfiguration } from "./enhancement-chain-model.ts";
import type { ChainConfiguration } from "./enhancement-chain-model";

export type SynthesisProfile = "vcp" | "vcp_joint";
export type MicrophoneMode = "channel0" | "beamform";
export type SynthesisOptions = {
  profile: ChainConfiguration["profile"];
  sample_rate?: number;
  microphone_mode: MicrophoneMode;
  beam_delays_samples?: number[];
  chain?: Pick<ChainConfiguration, "profile" | "selection">;
};

const SERVER_EXECUTED_NS_IDS = new Set([
  "cofilite",
  "dpdfnet2",
  "dpdfnet2_48k_hr",
  "hybrid_v2_pitch",
]);

export function synthesisConfigurationKey(configuration: ChainConfiguration | undefined, profile: SynthesisProfile): string {
  if (!configuration) return profile;
  const flags = chainFlags(configuration);
  return JSON.stringify(SERVER_EXECUTED_NS_IDS.has(configuration.selection.ns)
    ? { flags, ns: configuration.selection.ns }
    : flags);
}

export function buildSynthesisOptions(
  configuration: ChainConfiguration | undefined, profile: SynthesisProfile,
  rate: string, microphoneMode: MicrophoneMode, delayText: string,
): SynthesisOptions {
  if (rate && ![16000, 24000, 48000].includes(Number(rate))) throw new Error("请选择 16 / 24 / 48 kHz。");
  if (configuration) {
    // Reuse the catalog validator, dropping unsupported cached fields and arbitrary CLI flags.
    const checked = restoreConfiguration(JSON.stringify(configuration));
    if (checked.profile !== configuration.profile ||
      !chainStages.every((stage) => checked.selection[stage.id] === configuration.selection[stage.id])) {
      throw new Error("当前组合包含不支持的选项，请重新选择组合方案。");
    }
    const issue = chainExecutionIssue(checked);
    if (issue) throw new Error(issue);
    if (checked.selection.ns === "dpdfnet2_48k_hr" && rate !== "48000") {
      throw new Error("DPDFNet2 48 kHz HR 质量路线只接受原生 48 kHz 输入。");
    }
    const is16kOnly =
      ["cofilite", "cofilite_native16", "dpdfnet2", "hybrid_v2_pitch"].includes(checked.selection.ns)
      || ["speex_fused_dsp", "computer_pf_cpu"].includes(checked.selection.aec);
    if (is16kOnly && rate && rate !== "16000") {
      throw new Error("当前所选 16 kHz 专属算法簇只接受 16000 Hz 输入。");
    }
  }
  const options: SynthesisOptions = {
    profile: configuration?.profile ?? profile,
    ...(rate ? { sample_rate: Number(rate) } : {}),
    microphone_mode: microphoneMode,
    ...(configuration ? { chain: {
      profile: configuration.profile,
      selection: Object.fromEntries(chainStages.map((stage) => [stage.id, configuration.selection[stage.id]])) as ChainConfiguration["selection"],
    } } : {}),
  };
  if (microphoneMode === "beamform") {
    const parts = delayText.trim().split(/[,，\s]+/).filter(Boolean);
    const delays = parts.map(Number);
    if (!delays.length || !delays.every((value) => Number.isFinite(value))) {
      throw new Error("使用阵列时，请填写每路麦克风的真实标定延迟，以逗号分隔。");
    }
    if (configuration?.selection.bf === "off") throw new Error("当前组合已旁路 BF，请选择使用第 1 麦，或先开启波束形成。");
    options.beam_delays_samples = delays;
  }
  return options;
}
