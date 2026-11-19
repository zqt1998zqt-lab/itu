import catalog from "./enhancement-chain-configs-native16.json" with { type: "json" };

export const NATIVE16_PRESET_ID = "localvqe_cofilite_native16";
export const NATIVE16_NS_ID = "cofilite_native16";
export const native16Preset = catalog.presets.find((item) => item.id === NATIVE16_PRESET_ID)!;
export const native16NsOption = catalog.stages.find((item) => item.id === "ns")!.options.find((item) => item.id === NATIVE16_NS_ID)!;

export function validateNative16Geometry(report: Record<string, unknown>): string[] {
  const expected = {
    sample_rate: 16000,
    frame_size: 256,
    aec_model_codec_delay_samples_16k: 256,
    aec_hop_adapter_delay_samples_16k: 0,
    aec_algorithm_delay_samples_16k: 256,
    ns_delay_samples: 256,
    ns_stream_chunk_samples: 256,
    ns_external_fallback_frames: 0,
    aec_localvqe_fallback_frames: 0,
  };
  return Object.entries(expected).filter(([key, value]) => report[key] !== value).map(([key, value]) => `${key} 必须为 ${value}`);
}
