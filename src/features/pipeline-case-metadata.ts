import metadata from "../data/pipeline-case-metadata.json" with { type: "json" };
import type { PipelineCaseAudio } from "./PipelineEvaluation";

// Generated together with the SVGs from the actual frozen WAV pair. Editorial
// text stays with the algorithm; file identity and timing always follow the plot.
const cases = new Map(metadata.map((asset) => [asset.waveform, asset]));

export function resolvePipelineCase(value: PipelineCaseAudio): PipelineCaseAudio {
  const asset = cases.get(value.waveformSrc);
  if (!asset || asset.mel !== value.melSrc) {
    throw Error(`案例图与生成清单不匹配：${value.waveformSrc}`);
  }
  return {
    ...value,
    assetId: asset.asset_id,
    id: asset.case_id,
    sampleRate: asset.sample_rate,
    windowSeconds: [asset.window_seconds[0], asset.window_seconds[1]],
    delaySamples: asset.delay_samples,
    beforeFile: asset.before,
    afterFile: asset.after,
    beforeChannel: asset.before_channel,
    afterChannel: asset.after_channel,
  };
}
