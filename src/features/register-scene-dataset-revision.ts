import revision from "./scene-dataset-revision-ids.json" with { type: "json" };
import type { FiveSignalMode } from "./five-signal-flow-model.ts";
import type { BackgroundNoiseScene } from "./scoped-acoustic-model.ts";
import type { SceneConditionProfile } from "./scene-condition-profiles.ts";

/** Keep historical profiles readable while registering a complete new release. */
export function registerSceneDatasetRevision(
  byScene: Record<BackgroundNoiseScene, Record<FiveSignalMode, SceneConditionProfile>>,
  allProfiles: readonly SceneConditionProfile[],
  defaults: Record<BackgroundNoiseScene, SceneConditionProfile>,
  selectDefaults: boolean,
  identities: typeof revision = revision,
): void {
  const mutableProfiles = allProfiles as SceneConditionProfile[];
  for (const [scene, identifiers] of Object.entries(identities.scenes)) {
    const sceneKey = scene as BackgroundNoiseScene;
    for (const [mode, datasetId] of Object.entries(identifiers)) {
      const modeKey = mode as FiveSignalMode;
      const prior = byScene[sceneKey][modeKey];
      if (!prior || !/^[a-f0-9]{32}$/.test(datasetId)) {
        throw new Error("新版场景数据集身份表无效");
      }
      let profile = mutableProfiles.find((item) => item.datasetId === datasetId);
      if (!profile) {
        profile = { ...prior, datasetId };
        mutableProfiles.push(profile);
      }
      if (selectDefaults) byScene[sceneKey][modeKey] = profile;
    }
    if (selectDefaults) defaults[sceneKey] = byScene[sceneKey]["WB-wired"];
  }
}

export const SCENE_DATASET_REVISION = revision.revision;
