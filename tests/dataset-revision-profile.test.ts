import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import revision from "../src/features/scene-dataset-revision-ids.json" with { type: "json" };
import revisionR2 from "../src/features/scene-dataset-revision-r2-ids.json" with { type: "json" };
import {
  CONVERSATION_DATASET_IDS_BY_SCENE_AND_MODE,
  sceneConditionProfileForDatasetId,
  CONVERSATION_MODES,
} from "../src/features/scene-condition-profiles.ts";
import { readSceneConditionDatasetView } from "../src/features/scoped-acoustic-model.ts";
import type { FiveSignalMode } from "../src/features/five-signal-flow-model.ts";
import type { BackgroundNoiseScene } from "../src/features/scoped-acoustic-model.ts";

for (const release of [revision, revisionR2]) test(`${release.revision}: all 32 revised identities preserve old profiles`, () => {
  const identifiers = new Set<string>();
  for (const [scene, modes] of Object.entries(release.scenes)) {
    for (const [mode, identifier] of Object.entries(modes)) {
      const expected = createHash("sha256").update(`${release.revision}:scene:${scene}:${mode}`).digest("hex").slice(0, 32);
      assert.equal(identifier, expected);
      identifiers.add(identifier);
      const profile = sceneConditionProfileForDatasetId(identifier);
      assert(profile);
      assert.equal(profile.scene, scene);
      assert.equal(profile.mode, mode);
      assert.equal(profile.caseCount, 100);
      const priorId = CONVERSATION_DATASET_IDS_BY_SCENE_AND_MODE[scene as BackgroundNoiseScene][mode as FiveSignalMode];
      assert.notEqual(identifier, priorId);
      assert(sceneConditionProfileForDatasetId(priorId));
      const view = readSceneConditionDatasetView({
        test_scope: "background_noise", mode, scene,
        contract_item_count: 5, preview_condition_group_count: 100,
        observation_count: 280,
        conversation_counts: Object.fromEntries(CONVERSATION_MODES.map((kind) => [kind, 20])),
        standard_compliance: false, formal_test_status: "not_measured",
        datasets: [{
          dataset_id: identifier, dataset_name: profile.datasetName,
          test_scope: "background_noise", mode, scene,
          contract_item_count: 5, preview_condition_group_count: 100,
          observation_count: 280, standard_compliance: false,
          conversation_counts: Object.fromEntries(CONVERSATION_MODES.map((kind) => [kind, 20])),
          formal_test_status: "not_measured",
        }],
      }, { mode: mode as FiveSignalMode, scene: scene as BackgroundNoiseScene });
      assert.equal(view.datasets[0].dataset_id, identifier);
    }
  }
  assert.equal(identifiers.size, 32);
});
