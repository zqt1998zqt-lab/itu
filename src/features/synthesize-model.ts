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

export function synthesisConfigurationKey(configuration: ChainConfiguration | undefined, profile: SynthesisProfile): string {
  return configuration ? JSON.stringify(chainFlags(configuration)) : profile;
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
