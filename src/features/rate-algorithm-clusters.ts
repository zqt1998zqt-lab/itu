import {
  businessModeReferenceRateHz,
  type BusinessSampleRateHz,
} from "../domain/business-modes.ts";
import {
  chainPresets,
  chainSelections,
  type ChainPreset,
} from "./enhancement-chain-model.ts";

export type { BusinessSampleRateHz } from "../domain/business-modes.ts";
export type SupportedSampleRateHz = 8_000 | 16_000 | 24_000 | 48_000;

export const NB8_CLUSTER_ID = "speex_dpdfnet2_8khz";
export const NATIVE16_CLUSTER_ID = "localvqe_cofilite_native16";
export const QUALITY_48K_HR_CLUSTER_ID = "vcp_aec3_dpdfnet2_48k_hr";

export type AlgorithmClusterStageDetail = {
  ordinal: number;
  stageId: string;
  stageLabel: string;
  optionLabel: string;
  note: string;
  bypassed: boolean;
  missing: boolean;
  candidate: boolean;
};

export type RateAlgorithmCluster = {
  id: string;
  label: string;
  description: string;
  profile: string;
  vcp: boolean;
  supportedRates: readonly SupportedSampleRateHz[];
  primaryForRate?: SupportedSampleRateHz;
  rateSpecific: boolean;
  candidate?: boolean;
  recommended?: boolean;
  preset: ChainPreset | null;
  aecRouteId?: string;
  selection: Record<string, string>;
  stages: readonly AlgorithmClusterStageDetail[];
  executionNote: string;
};

export type RateAlgorithmClusterGroup = {
  id: string;
  label: string;
  clusters: readonly RateAlgorithmCluster[];
};

const RATE_16K_ONLY_IDS = new Set([
  NATIVE16_CLUSTER_ID,
  "localvqe_cofilite",
  "localvqe_dpdfnet2",
  "localvqe_hybrid_v2_pitch",
  "localvqe_aec200k",
  "speex_fused_dsp",
  "computer_pf_cpu",
]);

const RATE_48K_ONLY_IDS = new Set([QUALITY_48K_HR_CLUSTER_ID]);

const NB8_CLUSTER: RateAlgorithmCluster = {
  id: NB8_CLUSTER_ID,
  label: "VCP · SpeexDSP MDF 8 kHz + DPDFNet-2 8 kHz",
  description:
    "原生 8 kHz 维护路径：8 kHz SpeexDSP MDF 线性回声消除 + DPDFNet-2 8 kHz 神经降噪；全程保持 8 kHz PCM，无重采样与静默回退。",
  profile: "vcp",
  vcp: true,
  supportedRates: [8_000],
  primaryForRate: 8_000,
  rateSpecific: true,
  recommended: true,
  preset: null,
  selection: {
    bf: "das",
    aec: "speex_mdf_nb8",
    ns: "dpdfnet2_8khz",
    eq: "off",
    agc: "off",
    drc: "off",
    limiter: "off",
  },
  stages: [
    {
      ordinal: 1,
      stageId: "bf",
      stageLabel: "BF 波束形成",
      optionLabel: "单麦旁路 / 多麦 DAS 对齐（8 kHz）",
      note: "原生 8 kHz 输入保持单麦或按标定延迟对齐后进入 MDF，不做升采样。",
      bypassed: false,
      missing: false,
      candidate: false,
    },
    {
      ordinal: 2,
      stageId: "aec",
      stageLabel: "AEC 回声消除",
      optionLabel: "SpeexDSP MDF · 8 kHz 原生",
      note: "itu_nb8_speex 在 8 kHz / 80 样本帧上完成线性回声消除（speex_mdf_nb8）。",
      bypassed: false,
      missing: false,
      candidate: false,
    },
    {
      ordinal: 3,
      stageId: "ns",
      stageLabel: "NS 语音降噪",
      optionLabel: "DPDFNet-2 8 kHz · ONNX 流式",
      note: "dpdfnet2_8khz 原生 8 kHz 降噪，固定算法延迟 320 样本（40 ms），无回退帧。",
      bypassed: false,
      missing: false,
      candidate: false,
    },
    {
      ordinal: 4,
      stageId: "eq",
      stageLabel: "EQ 频响均衡",
      optionLabel: "原生 8 kHz 路径未启用",
      note: "原生 8 kHz 维护路径不经 16/48 kHz C++ EQ 桥接。",
      bypassed: true,
      missing: false,
      candidate: false,
    },
    {
      ordinal: 5,
      stageId: "agc",
      stageLabel: "AGC 增益控制",
      optionLabel: "原生 8 kHz 路径未启用",
      note: "原生 8 kHz 维护路径不经 16/48 kHz C++ AGC 桥接。",
      bypassed: true,
      missing: false,
      candidate: false,
    },
    {
      ordinal: 6,
      stageId: "drc",
      stageLabel: "DRC 动态范围",
      optionLabel: "原生 8 kHz 路径未启用",
      note: "原生 8 kHz 维护路径不经 16/48 kHz C++ DRC 桥接。",
      bypassed: true,
      missing: false,
      candidate: false,
    },
    {
      ordinal: 7,
      stageId: "limiter",
      stageLabel: "终级限幅",
      optionLabel: "原生 8 kHz 路径未启用",
      note: "输出保持原生 8 kHz DPDFNet-2 波形时间轴。",
      bypassed: true,
      missing: false,
      candidate: false,
    },
  ],
  executionNote:
    "原生 8 kHz 执行顺序固定为（可选多麦 DAS）→ SpeexDSP MDF 8 kHz → DPDFNet-2 8 kHz；全程 8 kHz 无重采样，实际执行身份以运行报告为准。",
};

function presetSupportedRates(preset: ChainPreset): readonly SupportedSampleRateHz[] {
  if (RATE_48K_ONLY_IDS.has(preset.id) || preset.selection.ns === "dpdfnet2_48k_hr") {
    return [48_000];
  }
  if (
    RATE_16K_ONLY_IDS.has(preset.id)
    || preset.selection.ns === "cofilite"
    || preset.selection.ns === "cofilite_native16"
    || preset.selection.ns === "dpdfnet2"
    || preset.selection.ns === "hybrid_v2_pitch"
    || preset.selection.aec === "speex_fused_dsp"
    || preset.selection.aec === "computer_pf_cpu"
    || preset.selection.aec === "localvqe_aec200k"
  ) {
    return [16_000];
  }
  return [16_000, 24_000, 48_000];
}

function clusterFromPreset(preset: ChainPreset): RateAlgorithmCluster {
  const supportedRates = presetSupportedRates(preset);
  const primaryForRate: SupportedSampleRateHz | undefined =
    preset.id === NATIVE16_CLUSTER_ID
      ? 16_000
      : preset.id === "aec3_serial"
        ? 48_000
        : undefined;
  return {
    id: preset.id,
    label: preset.label,
    description: preset.description,
    profile: preset.profile,
    vcp: preset.profile.startsWith("vcp"),
    supportedRates,
    primaryForRate,
    rateSpecific: supportedRates.length === 1,
    candidate: preset.candidate,
    recommended: preset.recommended,
    preset,
    aecRouteId: preset.selection.aec,
    selection: { ...preset.selection },
    stages: chainSelections(preset).map(({ ordinal, stage, option }) => ({
      ordinal,
      stageId: stage.id,
      stageLabel: stage.label,
      optionLabel: option?.label ?? "配置不可用",
      note: option?.note ?? "当前配置找不到该阶段的算法",
      bypassed: option?.id === "off",
      missing: option === null,
      candidate: Boolean(option?.candidate),
    })),
    executionNote:
      "执行顺序固定为 BF → AEC → NS → EQ → AGC → DRC → Limiter。这里展示当前算法簇请求的完整配置；DSP、Complex_PF 或 LocalVQE 路由仍须通过后端能力、模型与库身份准入，实际执行身份以运行报告为准。",
  };
}

const ALL_CLUSTERS: readonly RateAlgorithmCluster[] = [
  NB8_CLUSTER,
  ...chainPresets.map(clusterFromPreset),
];

const CLUSTER_BY_ID = new Map<string, RateAlgorithmCluster>(
  ALL_CLUSTERS.map((cluster) => [cluster.id, cluster]),
);

const RATE_GROUPS_SPEC: Record<
  SupportedSampleRateHz,
  readonly { id: string; label: string; clusterIds: readonly string[] }[]
> = {
  8_000: [
    {
      id: "nb8_native",
      label: "原生 8 kHz 路径",
      clusterIds: [NB8_CLUSTER_ID],
    },
  ],
  16_000: [
    {
      id: "wb16_dedicated",
      label: "16 kHz 专属 / 主维护路径",
      clusterIds: [
        NATIVE16_CLUSTER_ID,
        "localvqe_cofilite",
        "localvqe_dpdfnet2",
        "localvqe_hybrid_v2_pitch",
        "localvqe_aec200k",
        "speex_fused_dsp",
        "computer_pf_cpu",
      ],
    },
    {
      id: "wb16_multirate",
      label: "通用多采样率 VCP 路径（支持 16 / 48 kHz）",
      clusterIds: [
        "vcp_unified",
        "vcp_unified_eq_fused",
        "vcp_default",
        "speex_fused_cpu",
        "vcp_joint",
        "super_gaussian_optimal",
        "aec3_serial",
        "all_serial",
        "mvdr_serial",
        "speex_serial",
        "rnnoise_serial",
        "omlsa_serial",
        "stage4_serial",
      ],
    },
  ],
  48_000: [
    {
      id: "swb48_dedicated",
      label: "48 kHz 专属 / 主维护路径",
      clusterIds: [QUALITY_48K_HR_CLUSTER_ID, "aec3_serial"],
    },
    {
      id: "swb48_multirate",
      label: "通用原生 48 kHz VCP 路径（支持 16 / 48 kHz）",
      clusterIds: [
        "vcp_unified",
        "vcp_unified_eq_fused",
        "vcp_default",
        "speex_fused_cpu",
        "vcp_joint",
        "super_gaussian_optimal",
        "all_serial",
        "mvdr_serial",
        "speex_serial",
        "rnnoise_serial",
        "omlsa_serial",
        "stage4_serial",
      ],
    },
  ],
  24_000: [
    {
      id: "legacy24_multirate",
      label: "24 kHz 兼容 VCP 路径",
      clusterIds: [
        "vcp_default",
        "vcp_unified",
        "vcp_unified_eq_fused",
        "speex_fused_cpu",
        "vcp_joint",
        "super_gaussian_optimal",
        "aec3_serial",
        "all_serial",
        "mvdr_serial",
        "speex_serial",
        "rnnoise_serial",
        "omlsa_serial",
        "stage4_serial",
      ],
    },
  ],
};

const DEFAULT_CLUSTER_BY_RATE: Record<SupportedSampleRateHz, string> = {
  8_000: NB8_CLUSTER_ID,
  16_000: "vcp_unified",
  24_000: "vcp_default",
  48_000: QUALITY_48K_HR_CLUSTER_ID,
};

export function sampleRateForBusinessMode(mode: string): BusinessSampleRateHz {
  const resolved = businessModeReferenceRateHz(mode);
  if (resolved !== null) return resolved;
  if (mode.startsWith("NB-")) return 8_000;
  if (mode.startsWith("SWB-")) return 48_000;
  return 16_000;
}

export function algorithmClusterById(id: string): RateAlgorithmCluster | null {
  return CLUSTER_BY_ID.get(id) ?? null;
}

export function algorithmClusterGroupsForRate(
  rateHz: SupportedSampleRateHz,
  options?: { vcpOnly?: boolean },
): RateAlgorithmClusterGroup[] {
  const vcpOnly = options?.vcpOnly ?? true;
  const groups: RateAlgorithmClusterGroup[] = [];
  for (const spec of RATE_GROUPS_SPEC[rateHz]) {
    const clusters = spec.clusterIds
      .map((id) => CLUSTER_BY_ID.get(id))
      .filter(
        (cluster): cluster is RateAlgorithmCluster =>
          cluster !== undefined && (!vcpOnly || cluster.vcp),
      );
    if (clusters.length > 0) {
      groups.push({ id: spec.id, label: spec.label, clusters });
    }
  }
  if (!vcpOnly && rateHz !== 8_000) {
    const control = CLUSTER_BY_ID.get("array_control");
    if (control) {
      groups.push({
        id: "non_vcp_control",
        label: "非 VCP 对照基线",
        clusters: [control],
      });
    }
  }
  return groups;
}

export function algorithmClustersForRate(
  rateHz: SupportedSampleRateHz,
  options?: { vcpOnly?: boolean },
): RateAlgorithmCluster[] {
  return algorithmClusterGroupsForRate(rateHz, options).flatMap(
    (group) => group.clusters,
  );
}

export function algorithmClusterGroupsForMode(
  mode: string,
  options?: { vcpOnly?: boolean },
): RateAlgorithmClusterGroup[] {
  return algorithmClusterGroupsForRate(sampleRateForBusinessMode(mode), options);
}

export function algorithmClustersForMode(
  mode: string,
  options?: { vcpOnly?: boolean },
): RateAlgorithmCluster[] {
  return algorithmClustersForRate(sampleRateForBusinessMode(mode), options);
}

export function isAlgorithmClusterSupportedForRate(
  id: string,
  rateHz: SupportedSampleRateHz,
  options?: { vcpOnly?: boolean },
): boolean {
  return algorithmClustersForRate(rateHz, options).some(
    (cluster) => cluster.id === id,
  );
}

export function isAlgorithmClusterSupportedForMode(
  id: string,
  mode: string,
  options?: { vcpOnly?: boolean },
): boolean {
  return isAlgorithmClusterSupportedForRate(
    id,
    sampleRateForBusinessMode(mode),
    options,
  );
}

export function defaultAlgorithmClusterForRate(
  rateHz: SupportedSampleRateHz,
): string {
  return DEFAULT_CLUSTER_BY_RATE[rateHz];
}

export function defaultAlgorithmClusterForMode(mode: string): string {
  return defaultAlgorithmClusterForRate(sampleRateForBusinessMode(mode));
}

export function initialAlgorithmClustersByRate(): Record<BusinessSampleRateHz, string> {
  return {
    8_000: DEFAULT_CLUSTER_BY_RATE[8_000],
    16_000: DEFAULT_CLUSTER_BY_RATE[16_000],
    48_000: DEFAULT_CLUSTER_BY_RATE[48_000],
  };
}

export function resolveAlgorithmClusterForMode(
  currentId: string,
  mode: string,
  options?: { vcpOnly?: boolean },
): string {
  if (isAlgorithmClusterSupportedForMode(currentId, mode, options)) {
    return currentId;
  }
  return defaultAlgorithmClusterForMode(mode);
}

export function supportsMultiRateFullBatch(id: string): boolean {
  const cluster = CLUSTER_BY_ID.get(id);
  if (!cluster) return false;
  return (
    cluster.supportedRates.includes(16_000)
    && cluster.supportedRates.includes(48_000)
  );
}

export function algorithmClusterLabel(id: string): string {
  const cluster = CLUSTER_BY_ID.get(id);
  return cluster ? `${cluster.label} · ${cluster.id}` : id;
}

export function algorithmClusterRateSummary(id: string): string {
  const cluster = CLUSTER_BY_ID.get(id);
  if (!cluster) return "未定义采样率";
  if (cluster.supportedRates.length === 1) {
    return `仅 ${cluster.supportedRates[0] / 1000} kHz`;
  }
  return `${cluster.supportedRates.map((rate) => rate / 1000).join(" / ")} kHz`;
}
