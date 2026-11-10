import assert from "node:assert/strict";
import { test } from "node:test";

import {
  readSceneConditionBatch,
  sceneConditionCaseTxEvidence,
  SCENE_CONDITION_BATCHES_URL,
  SCENE_CONDITION_SCHEMA_VERSION,
  type SceneConditionBatch,
  type SceneConditionItemId,
} from "../src/features/scene-condition-test-model.ts";
import type { ScopedInputPreviewCase } from "../src/features/scoped-acoustic-model.ts";

const DATASET_ID = "b".repeat(32);
const BATCH_ID = "a".repeat(32);
const ALGORITHM_ID = "vcp_default";
const MANIFEST_SHA = "1".repeat(64);
const ASSETS_SHA = "2".repeat(64);
const CATALOG_SHA = "3".repeat(64);
const CLI_SHA = "4".repeat(64);
const INPUT_ROLES = ["microphone", "reference", "far_end", "noise"] as const;
const SPEECH_ITEM_IDS: SceneConditionItemId[] = [
  "WB-wired:B24",
  "WB-wired:B25",
  "WB-wired:B26",
];
const CNG_ITEM_IDS: SceneConditionItemId[] = [
  "WB-wired:B27",
  "WB-wired:B28",
];
const FAMILY_BY_ITEM: Record<SceneConditionItemId, string> = {
  "WB-wired:B24": "g_mos",
  "WB-wired:B25": "n_mos",
  "WB-wired:B26": "s_mos",
  "WB-wired:B27": "cng_level",
  "WB-wired:B28": "cng_spectrum",
};
const EXPECTED = {
  datasetId: DATASET_ID,
  algorithmClusterId: ALGORITHM_ID,
};

function hash(character: string): string {
  return character.repeat(64);
}

function id(character: string): string {
  return character.repeat(32);
}

function conditions() {
  return {
    mode: "WB-wired" as const,
    direction: "send",
    scene: "sce1",
    volume: "nominal",
    window_ms: null,
    input_level_db: 0,
  };
}

function inputHashes(seed: number) {
  return Object.fromEntries(
    INPUT_ROLES.map((role, index) => [
      role,
      hash(((seed + index) % 10).toString()),
    ]),
  ) as Record<(typeof INPUT_ROLES)[number], string>;
}

function visual(sourceSha256: string) {
  return {
    source_sha256: sourceSha256,
    duration_s: 0.25,
    sample_rate: 16_000,
    channels: 1,
    peak: 0,
    wave_scale: 1,
    wave_min_i16: Buffer.alloc(720 * 2).toString("base64"),
    wave_max_i16: Buffer.alloc(720 * 2).toString("base64"),
    mel_i16: Buffer.alloc(40 * 240 * 2).toString("base64"),
    mel_ceiling_db: -120,
  };
}

function fixture(): Record<string, any> {
  const cases: Record<string, any>[] = [];
  const items: Record<string, any>[] = [];
  for (let index = 0; index < 20; index += 1) {
    const kind = index < 10 ? "speech" : "cng";
    const caseId = `synthetic-${kind}-${String(index + 1).padStart(2, "0")}`;
    const itemIds = kind === "speech" ? SPEECH_ITEM_IDS : CNG_ITEM_IDS;
    const reportId = id(String(index % 10));
    const outputHash = hash(String((index + 2) % 10));
    const hashes = inputHashes(index);
    cases.push({
      case_id: caseId,
      status: "completed",
      report_id: reportId,
      conditions: conditions(),
      signals_sha256: { ...hashes },
      input_sha256: hashes,
      output_sha256: { tx: outputHash },
      tx_visual: visual(outputHash),
    });
    for (const itemId of itemIds) {
      items.push({
        observation_id: `${caseId}:${itemId}`,
        item_id: itemId,
        case_id: caseId,
        mode: "WB-wired",
        status: "review_required",
        value: 1.25,
        metric: "controlled_engineering_proxy",
        reason: "controlled fixture",
        family: FAMILY_BY_ITEM[itemId],
        test_name: itemId,
        source_cell: itemId.slice("WB-wired:".length),
        evidence_binding: {
          ...conditions(),
          case_id: caseId,
          report_id: reportId,
          batch_id: BATCH_ID,
          dataset_id: DATASET_ID,
          input_sha256: { ...hashes },
          output_sha256: { tx: outputHash },
        },
      });
    }
  }
  return {
    schema_version: SCENE_CONDITION_SCHEMA_VERSION,
    batch_id: BATCH_ID,
    dataset_id: DATASET_ID,
    dataset_name: "synthetic-scene-condition-batch-v1",
    dataset_manifest_sha256: MANIFEST_SHA,
    dataset_assets_sha256: ASSETS_SHA,
    algorithm_cluster_id: ALGORITHM_ID,
    configuration: { profile: "vcp_default" },
    catalog_sha256: CATALOG_SHA,
    cli_sha256: CLI_SHA,
    status: "completed",
    created_at: "2026-11-10T01:00:00.000Z",
    updated_at: "2026-11-10T01:01:00.000Z",
    test_scope: "background_noise",
    selected_mode: "WB-wired",
    selected_scene: "sce1",
    standard_compliance: false,
    formal_test_status: "not_measured",
    contract_item_count: 5,
    observation_count: 50,
    progress: {
      completed_cases: 20,
      total_cases: 20,
      completed_items: 50,
      total_items: 50,
    },
    summary: {
      total: 50,
      computed: 50,
      pass: 0,
      fail: 0,
      review_required: 50,
      not_measured: 0,
      invalid: 0,
    },
    cases,
    items,
  };
}

function previewCase(): ScopedInputPreviewCase {
  const batch = fixture();
  const source = batch.cases[0];
  return {
    case_id: source.case_id,
    conditions: source.conditions,
    families: ["g_mos", "n_mos", "s_mos"],
    signals: [...INPUT_ROLES],
    signals_sha256: source.input_sha256,
  };
}

test("a complete scene-condition batch has 20 cases, 50 observations, and five repeated items", () => {
  const parsed = readSceneConditionBatch(fixture(), EXPECTED);
  assert.equal(parsed.cases.length, 20);
  assert.equal(parsed.items.length, 50);
  assert.equal(parsed.status, "completed");
  assert.equal(new Set(parsed.items.map((item) => item.observation_id)).size, 50);
  assert.deepEqual(
    Object.fromEntries(
      ["WB-wired:B24", "WB-wired:B25", "WB-wired:B26", "WB-wired:B27", "WB-wired:B28"]
        .map((itemId) => [
          itemId,
          parsed.items.filter((item) => item.item_id === itemId).length,
        ]),
    ),
    {
      "WB-wired:B24": 10,
      "WB-wired:B25": 10,
      "WB-wired:B26": 10,
      "WB-wired:B27": 10,
      "WB-wired:B28": 10,
    },
  );
});

test("old schemas, truncated case or observation lists, and duplicate observation identities are rejected", () => {
  const mutations: Array<(value: Record<string, any>) => void> = [
    (value) => { value.schema_version = "scene-condition-batch-v0"; },
    (value) => { value.cases = value.cases.slice(0, 2); },
    (value) => { value.items = value.items.slice(0, 5); },
    (value) => { value.items.pop(); },
    (value) => { value.items[1].observation_id = value.items[0].observation_id; },
  ];
  for (const mutate of mutations) {
    const value = fixture();
    mutate(value);
    assert.throws(() => readSceneConditionBatch(value, EXPECTED));
  }
});

test("wrong dataset, algorithm, non-finite values, and forged completion never parse", () => {
  const mutations: Array<(value: Record<string, any>) => void> = [
    (value) => { value.dataset_id = hash("f"); },
    (value) => { value.algorithm_cluster_id = "agc_default"; },
    (value) => { value.items[0].value = Number.NaN; },
    (value) => { delete value.cases[0].output_sha256; },
    (value) => { delete value.cases[0].tx_visual; },
    (value) => { value.cases[0].status = "pending"; },
    (value) => { value.items[0].status = "invalid"; },
  ];
  for (const mutate of mutations) {
    const value = fixture();
    mutate(value);
    assert.throws(() => readSceneConditionBatch(value, EXPECTED));
  }
});

test("case input hashes cannot replace manifest signals even when items match", () => {
  const value = fixture();
  const caseId = value.cases[0].case_id;
  const forged = hash("f");
  value.cases[0].input_sha256.microphone = forged;
  for (const item of value.items) {
    if (item.case_id === caseId) {
      item.evidence_binding.input_sha256.microphone = forged;
    }
  }
  assert.throws(() => readSceneConditionBatch(value, EXPECTED));
});

test("TX evidence requires the same preview case, input hashes, report, and output", () => {
  const parsed = readSceneConditionBatch(fixture(), EXPECTED);
  const preview = previewCase();
  const ready = sceneConditionCaseTxEvidence(
    preview,
    parsed,
    MANIFEST_SHA,
  );
  assert.equal(ready.ready, true);
  assert.equal(
    ready.url,
    `${SCENE_CONDITION_BATCHES_URL}/${BATCH_ID}/cases/${preview.case_id}/outputs/tx?v=${parsed.cases[0].output_sha256?.tx}`,
  );
  assert.equal(ready.visual?.source_sha256, parsed.cases[0].output_sha256?.tx);
});

test("TX evidence rejects manifest, input, output, and report identity mismatches", () => {
  const parsed = readSceneConditionBatch(fixture(), EXPECTED);
  const preview = previewCase();

  assert.equal(
    sceneConditionCaseTxEvidence(preview, parsed, hash("f")).ready,
    false,
  );

  const wrongInput = structuredClone(parsed);
  wrongInput.cases[0].input_sha256.microphone = hash("f");
  assert.equal(
    sceneConditionCaseTxEvidence(preview, wrongInput, MANIFEST_SHA).ready,
    false,
  );

  const wrongOutput = structuredClone(parsed);
  wrongOutput.cases[0].output_sha256 = { tx: hash("f") };
  wrongOutput.cases[0].tx_visual = visual(hash("f"));
  assert.equal(
    sceneConditionCaseTxEvidence(preview, wrongOutput, MANIFEST_SHA).ready,
    false,
  );

  const wrongReport = structuredClone(parsed);
  wrongReport.items[0].evidence_binding.report_id = id("f");
  assert.equal(
    sceneConditionCaseTxEvidence(preview, wrongReport, MANIFEST_SHA).ready,
    false,
  );
});

test("a failed or missing preview case cannot borrow another case TX", () => {
  const parsed = readSceneConditionBatch(fixture(), EXPECTED);
  const preview = previewCase();

  const failed = structuredClone(parsed);
  failed.cases[0].status = "failed";
  const failedEvidence = sceneConditionCaseTxEvidence(
    preview,
    failed,
    MANIFEST_SHA,
  );
  assert.equal(failedEvidence.ready, false);
  assert.equal(failedEvidence.url, "");
  assert.equal(failedEvidence.visual, undefined);

  const missing = structuredClone(parsed);
  missing.cases = missing.cases.filter(
    (candidate) => candidate.case_id !== preview.case_id,
  );
  const missingEvidence = sceneConditionCaseTxEvidence(
    preview,
    missing,
    MANIFEST_SHA,
  );
  assert.equal(missingEvidence.ready, false);
  assert.match(missingEvidence.reason, /缺少该条件组/);
});

test("a running batch exposes TX for its completed bound case", () => {
  const parsed = readSceneConditionBatch(fixture(), EXPECTED);
  const preview = previewCase();
  const unfinished: SceneConditionBatch = {
    ...structuredClone(parsed),
    status: "running",
  };
  const evidence = sceneConditionCaseTxEvidence(
    preview,
    unfinished,
    MANIFEST_SHA,
  );
  assert.equal(evidence.ready, true);
  assert.equal(evidence.visual?.source_sha256, parsed.cases[0].output_sha256?.tx);
});
