import assert from "node:assert/strict";
import { test } from "node:test";

import {
  readSceneConditionBatch,
  readSceneConditionBatchHistory,
  sceneConditionCaseRxEvidence,
  sceneConditionCaseTxEvidence,
  SCENE_CONDITION_BATCHES_URL,
  SCENE_CONDITION_SCHEMA_VERSION,
  type SceneConditionBatch,
  type SceneConditionItemId,
} from "../src/features/scene-condition-test-model.ts";
import {
  CONVERSATION_MODES,
  LEGACY_SCENE_CONDITION_PROFILES,
  SCENE_CONDITION_PROFILES,
  type SceneConditionProfile,
} from "../src/features/scene-condition-profiles.ts";
import type {
  BackgroundNoiseScene,
  ScopedInputPreviewCase,
} from "../src/features/scoped-acoustic-model.ts";

const SCENES = ["sce1", "sce2", "sce3", "sce4"] as const;
const SCENE = "sce1" as const;
const PROFILE = LEGACY_SCENE_CONDITION_PROFILES[SCENE];
const ACTIVE_PROFILE = SCENE_CONDITION_PROFILES[SCENE];
const DATASET_ID = PROFILE.datasetId;
const BATCH_ID = "a".repeat(32);
const ALGORITHM_ID = "vcp_default";
const MANIFEST_SHA = "1".repeat(64);
const ASSETS_SHA = "2".repeat(64);
const CATALOG_SHA = "3".repeat(64);
const CLI_SHA = "4".repeat(64);
const INPUT_ROLES = ["microphone", "reference", "far_end", "noise"] as const;
const SPEECH_ITEM_IDS = PROFILE.contractItemIds.slice(0, 3);
const CNG_ITEM_IDS = PROFILE.contractItemIds.slice(3);
const EXPECTED = {
  datasetId: DATASET_ID,
  algorithmClusterId: ALGORITHM_ID,
};

function expectedFor(profile: SceneConditionProfile) {
  return {
    datasetId: profile.datasetId,
    algorithmClusterId: ALGORITHM_ID,
  };
}

function familyByItem(profile: SceneConditionProfile) {
  const families = [
    "g_mos",
    "n_mos",
    "s_mos",
    "cng_level",
    "cng_spectrum",
  ] as const;
  return Object.fromEntries(
    profile.contractItemIds.map((itemId, index) => [
      itemId,
      families[index],
    ]),
  ) as Record<SceneConditionItemId, string>;
}

function hash(character: string): string {
  return character.repeat(64);
}

function id(character: string): string {
  return character.repeat(32);
}

function conditions(scene: BackgroundNoiseScene = SCENE) {
  return {
    mode: "WB-wired" as const,
    direction: "send",
    scene,
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

function fixture(scene: BackgroundNoiseScene = SCENE): Record<string, any> {
  const profile = LEGACY_SCENE_CONDITION_PROFILES[scene];
  const speechItemIds = profile.contractItemIds.slice(0, 3);
  const cngItemIds = profile.contractItemIds.slice(3);
  const familyMap = familyByItem(profile);
  const cases: Record<string, any>[] = [];
  const items: Record<string, any>[] = [];
  for (let index = 0; index < 20; index += 1) {
    const kind = index < 10 ? "speech" : "cng";
    const caseId = `synthetic-${kind}-${String(index + 1).padStart(2, "0")}`;
    const itemIds = kind === "speech" ? speechItemIds : cngItemIds;
    const reportId = id(String(index % 10));
    const outputHash = hash(String((index + 2) % 10));
    const hashes = inputHashes(index);
    cases.push({
      case_id: caseId,
      status: "completed",
      report_id: reportId,
      conditions: conditions(scene),
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
        family: familyMap[itemId],
        test_name: itemId,
        source_cell: itemId.slice("WB-wired:".length),
        evidence_binding: {
          ...conditions(scene),
          case_id: caseId,
          report_id: reportId,
          batch_id: BATCH_ID,
          dataset_id: profile.datasetId,
          input_sha256: { ...hashes },
          output_sha256: { tx: outputHash },
        },
      });
    }
  }
  return {
    schema_version: SCENE_CONDITION_SCHEMA_VERSION,
    batch_id: BATCH_ID,
    dataset_id: profile.datasetId,
    dataset_name: profile.datasetName,
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
    selected_scene: scene,
    standard_compliance: false,
    formal_test_status: "not_measured",
    contract_item_count: 5,
    case_count: 20,
    preview_condition_group_count: 20,
    speech_case_count: 10,
    cng_case_count: 10,
    conversation_counts: {},
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

function fixtureWithRx(): Record<string, any> {
  const value = fixture();
  const target = value.cases[0];
  const rxSha256 = hash("9");
  target.output_sha256 = {
    tx: target.output_sha256.tx,
    rx: rxSha256,
  };
  target.rx_visual = visual(rxSha256);
  target.execution = {
    echo_reference: {
      source: "vcp_rx_processed",
      processed_output: "rx",
      processed_sha256: rxSha256,
      tx_echo_argument_sha256: rxSha256,
    },
  };
  for (const item of value.items) {
    if (item.case_id === target.case_id) {
      item.evidence_binding.output_sha256 = {
        tx: target.output_sha256.tx,
        rx: rxSha256,
      };
    }
  }
  return value;
}

type ConversationModeName = (typeof CONVERSATION_MODES)[number];

function activeSegments(mode: ConversationModeName) {
  if (mode === "near_single") {
    return [{ label: "speech", start_s: 0, end_s: 12 }];
  }
  if (mode === "far_single") {
    return [{ label: "far_speech", start_s: 0, end_s: 12 }];
  }
  if (mode === "double_talk") {
    return [{ label: "double_talk", start_s: 0, end_s: 12 }];
  }
  if (mode === "transition") {
    return [
      { label: "quiet", start_s: 0, end_s: 0.5 },
      { label: "far_only", start_s: 0.5, end_s: 2 },
      { label: "near_only", start_s: 2, end_s: 5 },
      { label: "far_only", start_s: 5, end_s: 8 },
      { label: "double_talk", start_s: 8, end_s: 11 },
      { label: "noise_only", start_s: 11, end_s: 12 },
    ];
  }
  return [
    { label: "cng_active", start_s: 0, end_s: 2 },
    { label: "cng_settling", start_s: 2, end_s: 4 },
    { label: "cng_noise_observation", start_s: 4, end_s: 12 },
  ];
}

function segmentMode(label: string) {
  if (label === "speech" || label === "near_only") return "near";
  if (label === "far_speech" || label === "far_only") return "far";
  if (label === "double_talk") return "double_talk";
  if (label === "noise_only" || label === "cng_noise_observation") {
    return "noise";
  }
  return "not_applicable";
}

function activeDiagnostics(
  caseId: string,
  mode: ConversationModeName,
  outputHash: string,
) {
  const segments = activeSegments(mode).map((segment) => {
    const modeName = segmentMode(segment.label);
    return {
      ...segment,
      mode: modeName,
      metrics: {
        near_gain_db: modeName === "near" ? 0.25 : null,
        near_correlation: modeName === "near" ? 0.95 : null,
        echo_reduction_db: modeName === "far" ? 3.5 : null,
        tx_level_dbfs: -30,
      },
      reason: "controlled engineering diagnostics",
      status: modeName === "not_applicable"
        ? "not_applicable"
        : "review_required",
    };
  });
  return {
    schema_version: "scene-condition-conversation-diagnostics-v1",
    case_id: caseId,
    conversation_mode: mode,
    evidence_grade: "engineering_proxy",
    formal_compliance: false,
    standard_compliance: false,
    formal_test_status: "not_measured",
    status: "review_required",
    reason: "controlled engineering diagnostics",
    review_required: true,
    sample_rate_hz: 16_000,
    source_duration_s: 12,
    tx_duration_s: 12,
    tx_sha256: outputHash,
    alignment_method: {
      name: "bounded_fft_cross_correlation",
      max_lag_ms: 100,
      fixed_index_assumption: false,
    },
    segments,
    limitations: [
      "engineering proxy only",
      "not formal ITU-T compliance evidence",
    ],
  };
}

function activeFixture(
  scene: BackgroundNoiseScene = SCENE,
): Record<string, any> {
  const profile = SCENE_CONDITION_PROFILES[scene];
  const speechItemIds = profile.contractItemIds.slice(0, 3);
  const cngItemIds = profile.contractItemIds.slice(3);
  const familyMap = familyByItem(profile);
  const cases: Record<string, any>[] = [];
  const items: Record<string, any>[] = [];
  for (const [modeIndex, conversationMode] of CONVERSATION_MODES.entries()) {
    const isNoise = conversationMode === "noise_only";
    const isUnmeasured = conversationMode === "far_single"
      || conversationMode === "transition";
    const itemIds = isNoise ? cngItemIds : speechItemIds;
    for (let offset = 0; offset < 20; offset += 1) {
      const index = modeIndex * 20 + offset;
      const caseId =
        `synthetic-${conversationMode}-${String(offset + 1).padStart(2, "0")}`;
      const reportId = id(String(index % 10));
      const outputHash = hash(String((index + 2) % 10));
      const hashes = inputHashes(index);
      const renderSegments = activeSegments(conversationMode);
      cases.push({
        case_id: caseId,
        status: "completed",
        report_id: reportId,
        conditions: conditions(scene),
        provenance: {
          engineering_preview: true,
          actual_vehicle_recording: false,
          conversation_mode: conversationMode,
          render_metadata: {
            kind: isNoise ? "cng" : "speech",
            duration_s: 12,
            sample_rate_hz: 16_000,
            segments: renderSegments,
          },
        },
        conversation_diagnostics: activeDiagnostics(
          caseId,
          conversationMode,
          outputHash,
        ),
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
          status: isUnmeasured ? "not_measured" : "review_required",
          value: isUnmeasured ? null : 1.25,
          metric: "controlled_engineering_proxy",
          reason: "controlled fixture",
          family: familyMap[itemId],
          test_name: itemId,
          source_cell: itemId.slice("WB-wired:".length),
          evidence_binding: {
            ...conditions(scene),
            case_id: caseId,
            report_id: reportId,
            batch_id: BATCH_ID,
            dataset_id: profile.datasetId,
            input_sha256: { ...hashes },
            output_sha256: { tx: outputHash },
          },
        });
      }
    }
  }
  return {
    schema_version: SCENE_CONDITION_SCHEMA_VERSION,
    batch_id: BATCH_ID,
    dataset_id: profile.datasetId,
    dataset_name: profile.datasetName,
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
    selected_scene: scene,
    standard_compliance: false,
    formal_test_status: "not_measured",
    contract_item_count: 5,
    case_count: 100,
    preview_condition_group_count: 100,
    speech_case_count: 80,
    cng_case_count: 20,
    conversation_counts: Object.fromEntries(
      CONVERSATION_MODES.map((conversationMode) => [conversationMode, 20]),
    ),
    observation_count: 280,
    progress: {
      completed_cases: 100,
      total_cases: 100,
      completed_items: 280,
      total_items: 280,
    },
    summary: {
      total: 280,
      computed: 160,
      pass: 0,
      fail: 0,
      review_required: 160,
      not_measured: 120,
      invalid: 0,
    },
    cases,
    items,
  };
}

function singleSceneBatch(
  source: Record<string, any>,
  caseId: string,
): Record<string, any> {
  const caseRecord = source.cases.find(
    (entry: Record<string, any>) => entry.case_id === caseId,
  );
  assert.ok(caseRecord);
  const items = source.items.filter(
    (entry: Record<string, any>) => entry.case_id === caseId,
  );
  const value: Record<string, any> = {
    ...source,
    cases: [structuredClone(caseRecord)],
    items: structuredClone(items),
  };
  const counts = {
    pass: 0,
    fail: 0,
    review_required: 0,
    not_measured: 0,
    invalid: 0,
  };
  let computed = 0;
  for (const item of items) {
    if (item.status in counts) {
      counts[item.status as keyof typeof counts] += 1;
    }
    if (
      typeof item.value === "number"
      && item.status !== "not_measured"
      && item.status !== "invalid"
    ) {
      computed += 1;
    }
  }
  const conversationMode = caseRecord.provenance?.conversation_mode;
  const isCng = conversationMode === "noise_only"
    || items.every((entry: Record<string, any>) =>
      ["cng_level", "cng_spectrum"].includes(entry.family)
    );
  value.case_count = 1;
  value.observation_count = items.length;
  value.preview_condition_group_count = source.case_count;
  value.execution_scope = "single_case";
  value.selected_case_id = caseId;
  value.dataset_case_count = source.case_count;
  value.dataset_observation_count = source.observation_count;
  value.speech_case_count = isCng ? 0 : 1;
  value.cng_case_count = isCng ? 1 : 0;
  value.conversation_counts = Object.keys(source.conversation_counts).length
    ? Object.fromEntries(
        CONVERSATION_MODES.map((mode) => [
          mode,
          mode === conversationMode ? 1 : 0,
        ]),
      )
    : {};
  value.progress = {
    completed_cases: 1,
    total_cases: 1,
    completed_items: items.length,
    total_items: items.length,
  };
  value.summary = {
    total: items.length,
    computed,
    ...counts,
  };
  return value;
}

function singleCaseHistory(
  source: Record<string, any>,
  caseIds: readonly string[],
): Record<string, any>[] {
  return caseIds.map((caseId, index) => {
    const value = singleSceneBatch(source, caseId);
    const batchId = index.toString(16).padStart(32, "0");
    value.batch_id = batchId;
    value.items.forEach((item: Record<string, any>) => {
      item.evidence_binding.batch_id = batchId;
    });
    return value;
  });
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

function activePreviewCase(batch: SceneConditionBatch): ScopedInputPreviewCase {
  const source = batch.cases[0];
  return {
    case_id: source.case_id,
    provenance: source.provenance,
    conditions: source.conditions,
    families: ["g_mos", "n_mos", "s_mos"],
    signals: [...INPUT_ROLES],
    signals_sha256: source.input_sha256,
  };
}

test("all four scene-condition batches have 20 cases, 50 observations, and five repeated rows", () => {
  for (const scene of SCENES) {
    const profile = LEGACY_SCENE_CONDITION_PROFILES[scene];
    const parsed = readSceneConditionBatch(
      fixture(scene),
      expectedFor(profile),
    );
    assert.equal(parsed.cases.length, 20);
    assert.equal(parsed.items.length, 50);
    assert.equal(parsed.selected_scene, scene);
    assert.equal(parsed.status, "completed");
    assert.equal(
      new Set(parsed.items.map((item) => item.observation_id)).size,
      50,
    );
    assert.deepEqual(
      Object.fromEntries(
        profile.contractItemIds.map((itemId) => [
          itemId,
          parsed.items.filter((item) => item.item_id === itemId).length,
        ]),
      ),
      Object.fromEntries(
        profile.contractItemIds.map((itemId) => [itemId, 10]),
      ),
    );
  }
});

test("all four active conversation batches derive 100 cases, 280 observations, and 80/20 row counts", () => {
  for (const scene of SCENES) {
    const profile = SCENE_CONDITION_PROFILES[scene];
    const parsed = readSceneConditionBatch(
      activeFixture(scene),
      expectedFor(profile),
    );
    assert.equal(parsed.cases.length, 100);
    assert.equal(parsed.items.length, 280);
    assert.equal(parsed.observation_count, 280);
    assert.equal(parsed.selected_scene, scene);
    assert.equal(parsed.cases[0].provenance?.conversation_mode, "near_single");
    assert.equal(
      parsed.cases[0].conversation_diagnostics?.schema_version,
      "scene-condition-conversation-diagnostics-v1",
    );
    assert.equal(
      parsed.cases[0].conversation_diagnostics?.case_id,
      parsed.cases[0].case_id,
    );
    assert.equal(
      parsed.cases[0].conversation_diagnostics?.tx_sha256,
      parsed.cases[0].output_sha256?.tx,
    );
    assert.deepEqual(
      parsed.cases[0].conversation_diagnostics?.segments,
      activeDiagnostics(
        parsed.cases[0].case_id,
        "near_single",
        parsed.cases[0].output_sha256!.tx,
      ).segments,
    );
    for (const mode of CONVERSATION_MODES) {
      assert.equal(
        parsed.cases.filter(
          (entry) => entry.provenance?.conversation_mode === mode,
        ).length,
        20,
        `${scene}/${mode}`,
      );
    }
    const itemCounts = Object.fromEntries(
      profile.contractItemIds.map((itemId) => [
        itemId,
        parsed.items.filter((item) => item.item_id === itemId).length,
      ]),
    );
    assert.deepEqual(
      itemCounts,
      Object.fromEntries(
        profile.contractItemIds.map((itemId, index) => [
          itemId,
          index < 3 ? 80 : 20,
        ]),
      ),
    );
    for (const mode of ["near_single", "double_talk"] as const) {
      const caseIds = new Set(
        parsed.cases
          .filter((entry) => entry.provenance?.conversation_mode === mode)
          .map((entry) => entry.case_id),
      );
      assert.ok(
        parsed.items
          .filter((entry) => caseIds.has(entry.case_id))
          .every((entry) => entry.status === "review_required"),
      );
    }
  }
});

test("active batch rejects missing modes, duplicated mode counts, and mismatched family groups", () => {
  const missingMode = activeFixture();
  delete missingMode.cases[0].provenance.conversation_mode;
  assert.throws(
    () => readSceneConditionBatch(
      missingMode,
      expectedFor(ACTIVE_PROFILE),
    ),
    /conversation_mode/,
  );

  const duplicatedMode = activeFixture();
  const farCase = duplicatedMode.cases.find(
    (entry: Record<string, any>) =>
      entry.provenance.conversation_mode === "far_single",
  );
  assert.ok(farCase);
  farCase.provenance.conversation_mode = "near_single";
  farCase.provenance.render_metadata = {
    kind: "speech",
    duration_s: 12,
    sample_rate_hz: 16_000,
    segments: activeSegments("near_single"),
  };
  farCase.conversation_diagnostics = activeDiagnostics(
    farCase.case_id,
    "near_single",
    farCase.output_sha256.tx,
  );
  assert.throws(
    () => readSceneConditionBatch(
      duplicatedMode,
      expectedFor(ACTIVE_PROFILE),
    ),
    /条件组数量/,
  );

  const wrongFamily = activeFixture();
  wrongFamily.cases[0].provenance.conversation_mode = "noise_only";
  wrongFamily.cases[0].provenance.render_metadata = {
    kind: "cng",
    duration_s: 12,
    sample_rate_hz: 16_000,
    segments: activeSegments("noise_only"),
  };
  wrongFamily.cases[0].conversation_diagnostics = activeDiagnostics(
    wrongFamily.cases[0].case_id,
    "noise_only",
    wrongFamily.cases[0].output_sha256.tx,
  );
  assert.throws(
    () => readSceneConditionBatch(
      wrongFamily,
      expectedFor(ACTIVE_PROFILE),
    ),
    /通话模式与合同族分组/,
  );
});

test("active batch rejects forged 50/280 counts and cross-scene identities", () => {
  const forgedCount = activeFixture();
  forgedCount.observation_count = 50;
  assert.throws(
    () => readSceneConditionBatch(
      forgedCount,
      expectedFor(ACTIVE_PROFILE),
    ),
    /观测数/,
  );

  const sce2 = SCENE_CONDITION_PROFILES.sce2;
  const crossScene = activeFixture("sce2");
  crossScene.selected_scene = "sce1";
  assert.throws(
    () => readSceneConditionBatch(crossScene, expectedFor(sce2)),
    /范围、模式、场景/,
  );

  const crossRows = activeFixture("sce2");
  const first = crossRows.items[0];
  first.item_id = "WB-wired:B24";
  first.observation_id = `${first.case_id}:WB-wired:B24`;
  first.source_cell = "B24";
  assert.throws(
    () => readSceneConditionBatch(crossRows, expectedFor(sce2)),
    /合同测项/,
  );
});

test("far and transition GNS rows cannot pass while near and double-talk stay reviewable", () => {
  const parsed = readSceneConditionBatch(
    activeFixture(),
    expectedFor(ACTIVE_PROFILE),
  );
  for (const mode of ["near_single", "double_talk"] as const) {
    const caseIds = new Set(
      parsed.cases
        .filter((entry) => entry.provenance?.conversation_mode === mode)
        .map((entry) => entry.case_id),
    );
    assert.ok(
      parsed.items
        .filter((entry) => caseIds.has(entry.case_id))
        .every((entry) => entry.status === "review_required"),
    );
  }

  for (const mode of ["far_single", "transition"] as const) {
    const value = activeFixture();
    const targetCase = value.cases.find(
      (entry: Record<string, any>) =>
        entry.provenance.conversation_mode === mode,
    );
    assert.ok(targetCase);
    const target = value.items.find(
      (entry: Record<string, any>) => entry.case_id === targetCase.case_id,
    );
    assert.ok(target);
    target.status = "pass";
    target.value = 1.25;
    assert.throws(
      () => readSceneConditionBatch(value, expectedFor(ACTIVE_PROFILE)),
      /不能工程通过/,
    );
  }
});

test("active TX evidence follows the case conversation mode", () => {
  const parsed = readSceneConditionBatch(
    activeFixture(),
    expectedFor(ACTIVE_PROFILE),
  );
  const preview = activePreviewCase(parsed);
  const ready = sceneConditionCaseTxEvidence(preview, parsed, MANIFEST_SHA);
  assert.equal(ready.ready, true);
  assert.equal(
    ready.url,
    `${SCENE_CONDITION_BATCHES_URL}/${BATCH_ID}/cases/${preview.case_id}/outputs/tx?v=${parsed.cases[0].output_sha256?.tx}`,
  );

  const wrongMode = structuredClone(preview);
  wrongMode.provenance = {
    ...wrongMode.provenance,
    conversation_mode: "far_single",
  };
  assert.equal(
    sceneConditionCaseTxEvidence(wrongMode, parsed, MANIFEST_SHA).ready,
    false,
  );
});

test("v2 diagnostics require complete identity, boundary, segment, and metric evidence", () => {
  const reject = (
    mutate: (value: Record<string, any>) => void,
  ) => {
    const value = activeFixture();
    mutate(value);
    assert.throws(
      () => readSceneConditionBatch(
        value,
        expectedFor(ACTIVE_PROFILE),
      ),
    );
  };

  reject((value) => {
    delete value.cases[0].conversation_diagnostics;
  });
  reject((value) => {
    value.cases[0].conversation_diagnostics.case_id = "forged-case";
  });
  reject((value) => {
    value.cases[0].conversation_diagnostics.tx_sha256 = hash("f");
  });
  reject((value) => {
    value.cases[0].conversation_diagnostics.conversation_mode = "far_single";
  });
  reject((value) => {
    value.cases[0].conversation_diagnostics.schema_version = "other-schema";
  });
  reject((value) => {
    value.cases[0].conversation_diagnostics.formal_compliance = true;
  });
  reject((value) => {
    value.cases[0].conversation_diagnostics.status = "pass";
  });
  reject((value) => {
    value.cases[0].conversation_diagnostics.segments[0].start_s = 0.1;
  });
  reject((value) => {
    value.cases[0].conversation_diagnostics.segments[0].mode = "bogus";
  });
  reject((value) => {
    value.cases[0].conversation_diagnostics.segments[0].status = "fail";
  });
  reject((value) => {
    value.cases[0].conversation_diagnostics.segments[0].metrics.near_gain_db =
      Number.NaN;
  });

  const allowed = activeFixture();
  const diagnostics = allowed.cases[0].conversation_diagnostics;
  diagnostics.status = "not_measurable";
  diagnostics.review_required = false;
  diagnostics.segments[0].status = "not_measurable";
  for (const key of Object.keys(diagnostics.segments[0].metrics)) {
    diagnostics.segments[0].metrics[key] = null;
  }
  assert.doesNotThrow(() =>
    readSceneConditionBatch(allowed, expectedFor(ACTIVE_PROFILE))
  );
});

test("TX evidence rejects a forged diagnostics record even when batch parsing is bypassed", () => {
  const parsed = readSceneConditionBatch(
    activeFixture(),
    expectedFor(ACTIVE_PROFILE),
  );
  parsed.cases[0].conversation_diagnostics = {
    ...parsed.cases[0].conversation_diagnostics,
    case_id: "forged-case",
  };
  const evidence = sceneConditionCaseTxEvidence(
    activePreviewCase(parsed),
    parsed,
    MANIFEST_SHA,
  );
  assert.equal(evidence.ready, false);
  assert.match(evidence.reason, /诊断/);
});

test("a batch cannot mix a dataset with another scene or another scene's contract rows", () => {
  const sce2 = LEGACY_SCENE_CONDITION_PROFILES.sce2;
  const wrongScene = fixture("sce2");
  wrongScene.selected_scene = "sce1";
  assert.throws(
    () => readSceneConditionBatch(wrongScene, expectedFor(sce2)),
    /范围、模式、场景/,
  );

  const wrongRows = fixture("sce2");
  const firstCase = wrongRows.cases[0];
  firstCase.contract_item_ids = [
    "WB-wired:B24",
    "WB-wired:B25",
    "WB-wired:B26",
  ];
  const firstItem = wrongRows.items.find(
    (item: Record<string, any>) => item.case_id === firstCase.case_id,
  );
  assert.ok(firstItem);
  firstItem.item_id = "WB-wired:B24";
  firstItem.observation_id = `${firstCase.case_id}:WB-wired:B24`;
  firstItem.family = "g_mos";
  firstItem.source_cell = "B24";
  assert.throws(
    () => readSceneConditionBatch(wrongRows, expectedFor(sce2)),
    /合同测项/,
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

test("RX-after evidence is optional for legacy batches and bound to TX for new batches", () => {
  const legacy = readSceneConditionBatch(fixture(), EXPECTED);
  const preview = previewCase();
  assert.equal(
    sceneConditionCaseRxEvidence(preview, legacy, MANIFEST_SHA).ready,
    false,
  );

  const parsed = readSceneConditionBatch(fixtureWithRx(), EXPECTED);
  const rx = sceneConditionCaseRxEvidence(preview, parsed, MANIFEST_SHA);
  assert.equal(rx.ready, true);
  assert.equal(
    rx.url,
    `${SCENE_CONDITION_BATCHES_URL}/${BATCH_ID}/cases/${preview.case_id}/outputs/rx?v=${parsed.cases[0].output_sha256?.rx}`,
  );
  assert.equal(
    rx.visual?.source_sha256,
    parsed.cases[0].output_sha256?.rx,
  );

  const wrongEcho = fixtureWithRx();
  wrongEcho.cases[0].execution.echo_reference.processed_sha256 = hash("f");
  assert.throws(
    () => readSceneConditionBatch(wrongEcho, EXPECTED),
    /TX 未绑定 RX 处理后回声参考/,
  );

  const missingVisual = fixtureWithRx();
  delete missingVisual.cases[0].rx_visual;
  assert.throws(
    () => readSceneConditionBatch(missingVisual, EXPECTED),
    /RX 处理后波形/,
  );
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

test("single-case scene snapshots keep source totals and bind the requested case", () => {
  const legacy = fixture();
  const legacyCaseId = legacy.cases[0].case_id;
  const legacySingle = singleSceneBatch(legacy, legacyCaseId);
  const parsedLegacy = readSceneConditionBatch(legacySingle, {
    ...EXPECTED,
    caseId: legacyCaseId,
  });
  assert.equal(parsedLegacy.execution_scope, "single_case");
  assert.equal(parsedLegacy.selected_case_id, legacyCaseId);
  assert.equal(parsedLegacy.case_count, 1);
  assert.equal(parsedLegacy.observation_count, 3);
  assert.equal(parsedLegacy.preview_condition_group_count, 20);
  assert.equal(parsedLegacy.dataset_case_count, 20);
  assert.equal(parsedLegacy.dataset_observation_count, 50);
  assert.equal(parsedLegacy.contract_item_count, 5);
  assert.equal(parsedLegacy.speech_case_count, 1);
  assert.equal(parsedLegacy.cng_case_count, 0);
  assert.deepEqual(parsedLegacy.conversation_counts, {});
  assert.deepEqual(parsedLegacy.progress, {
    completed_cases: 1,
    total_cases: 1,
    completed_items: 3,
    total_items: 3,
  });
  assert.deepEqual(
    parsedLegacy.cases.map((entry) => entry.case_id),
    [legacyCaseId],
  );
  assert.ok(
    parsedLegacy.items.every((entry) => entry.case_id === legacyCaseId),
  );

  const active = activeFixture();
  const noiseCase = active.cases.find(
    (entry: Record<string, any>) =>
      entry.provenance.conversation_mode === "noise_only",
  );
  assert.ok(noiseCase);
  const activeSingle = singleSceneBatch(active, noiseCase.case_id);
  const parsedActive = readSceneConditionBatch(activeSingle, {
    ...expectedFor(ACTIVE_PROFILE),
    caseId: noiseCase.case_id,
  });
  assert.equal(parsedActive.observation_count, 2);
  assert.equal(parsedActive.preview_condition_group_count, 100);
  assert.equal(parsedActive.dataset_case_count, 100);
  assert.equal(parsedActive.dataset_observation_count, 280);
  assert.equal(parsedActive.speech_case_count, 0);
  assert.equal(parsedActive.cng_case_count, 1);
  assert.deepEqual(
    parsedActive.conversation_counts,
    Object.fromEntries(
      CONVERSATION_MODES.map((mode) => [
        mode,
        mode === "noise_only" ? 1 : 0,
      ]),
    ),
  );

  const historicalLegacySingle = structuredClone(legacySingle);
  historicalLegacySingle.batch_id = "b".repeat(32);
  historicalLegacySingle.items.forEach((item: Record<string, any>) => {
    item.evidence_binding.batch_id = historicalLegacySingle.batch_id;
  });
  const mixedLegacy = readSceneConditionBatchHistory(
    [legacy, historicalLegacySingle],
    EXPECTED,
  );
  assert.equal(mixedLegacy.length, 2);
  assert.equal(mixedLegacy[0].execution_scope, undefined);
  assert.equal(mixedLegacy[1].execution_scope, "single_case");
  const historicalActiveSingle = structuredClone(activeSingle);
  historicalActiveSingle.batch_id = "c".repeat(32);
  historicalActiveSingle.items.forEach((item: Record<string, any>) => {
    item.evidence_binding.batch_id = historicalActiveSingle.batch_id;
  });
  const mixedActive = readSceneConditionBatchHistory(
    [active, historicalActiveSingle],
    expectedFor(ACTIVE_PROFILE),
  );
  assert.equal(mixedActive.length, 2);
  assert.equal(mixedActive[0].execution_scope, undefined);
  assert.equal(mixedActive[1].execution_scope, "single_case");
});

test("latest-per-case history enforces profile coverage beyond the normal limit", () => {
  const active = activeFixture();
  const activeCaseIds = active.cases
    .slice(0, 21)
    .map((entry: Record<string, any>) => entry.case_id);
  const activeRows = singleCaseHistory(active, activeCaseIds);
  const latestExpected = {
    ...expectedFor(ACTIVE_PROFILE),
    latestPerCase: true,
  };
  const parsed = readSceneConditionBatchHistory(activeRows, latestExpected);
  assert.equal(parsed.length, 21);
  assert.deepEqual(
    parsed.map((batch) => batch.cases.map((entry) => entry.case_id)),
    activeCaseIds.map((caseId: string) => [caseId]),
  );
  assert.throws(
    () => readSceneConditionBatchHistory(
      activeRows,
      expectedFor(ACTIVE_PROFILE),
    ),
    /至多 20 条/,
  );

  const allActiveRows = singleCaseHistory(
    active,
    active.cases.map((entry: Record<string, any>) => entry.case_id),
  );
  assert.throws(
    () => readSceneConditionBatchHistory(
      [...allActiveRows, activeRows[0]],
      latestExpected,
    ),
    /至多 100 条/,
  );
  assert.throws(
    () => readSceneConditionBatchHistory(
      [activeRows[0], activeRows[0]],
      latestExpected,
    ),
    /重复批次 ID/,
  );

  const wrongDataset = singleCaseHistory(active, [activeCaseIds[0]])[0];
  wrongDataset.dataset_id = PROFILE.datasetId;
  assert.throws(
    () => readSceneConditionBatchHistory(
      [wrongDataset],
      latestExpected,
    ),
    /数据集与当前测试集不一致/,
  );
  const wrongAlgorithm = singleCaseHistory(active, [activeCaseIds[0]])[0];
  wrongAlgorithm.algorithm_cluster_id = "other_algorithm";
  assert.throws(
    () => readSceneConditionBatchHistory(
      [wrongAlgorithm],
      latestExpected,
    ),
    /算法簇与当前选择不一致/,
  );
  assert.throws(
    () => readSceneConditionBatchHistory(
      singleCaseHistory(active, [activeCaseIds[0], activeCaseIds[0]]),
      latestExpected,
    ),
    /覆盖新的条件组/,
  );

  const legacy = fixture();
  const legacyRows = singleCaseHistory(
    legacy,
    legacy.cases.map((entry: Record<string, any>) => entry.case_id),
  );
  assert.equal(
    readSceneConditionBatchHistory(legacyRows, {
      ...EXPECTED,
      latestPerCase: true,
    }).length,
    20,
  );
  assert.throws(
    () => readSceneConditionBatchHistory([], {
      datasetId: "f".repeat(32),
      algorithmClusterId: ALGORITHM_ID,
      latestPerCase: true,
    }),
    /未注册的场景条件数据集/,
  );
});

test("single-case scene selection is all-or-none and request-bound", () => {
  const source = fixture();
  const caseId = source.cases[0].case_id;
  const single = singleSceneBatch(source, caseId);

  assert.throws(
    () => readSceneConditionBatch(source, { ...EXPECTED, caseId }),
    /缺少单案选择字段/,
  );
  assert.throws(
    () => readSceneConditionBatch(single, {
      ...EXPECTED,
      caseId: source.cases[1].case_id,
    }),
    /请求 case_id 不一致/,
  );
  for (const field of [
    "execution_scope",
    "selected_case_id",
    "dataset_case_count",
    "dataset_observation_count",
  ] as const) {
    const partial = structuredClone(single);
    delete partial[field];
    assert.throws(
      () => readSceneConditionBatch(partial, EXPECTED),
      /四项同时存在/,
    );
  }

  const mutations: Array<(value: Record<string, any>) => void> = [
    (value) => { value.execution_scope = "full"; },
    (value) => { value.selected_case_id = "different-case"; },
    (value) => { value.dataset_case_count = 19; },
    (value) => { value.dataset_observation_count = 49; },
    (value) => { value.case_count = 20; },
    (value) => { value.observation_count = 50; },
    (value) => { value.preview_condition_group_count = 1; },
    (value) => { value.speech_case_count = 0; },
    (value) => { value.cng_case_count = 1; },
    (value) => { value.progress.total_cases = 20; },
    (value) => { value.progress.total_items = 50; },
    (value) => { value.summary.total = 50; },
    (value) => { value.cases.push(structuredClone(source.cases[1])); },
    (value) => { value.items.push(structuredClone(source.items[3])); },
  ];
  for (const mutate of mutations) {
    const value = structuredClone(single);
    mutate(value);
    assert.throws(() => readSceneConditionBatch(value, EXPECTED));
  }

  const forgedFull = structuredClone(source);
  Object.assign(forgedFull, {
    execution_scope: "single_case",
    selected_case_id: caseId,
    dataset_case_count: 20,
    dataset_observation_count: 50,
  });
  assert.throws(() => readSceneConditionBatch(forgedFull, EXPECTED));

  const deletedRows = structuredClone(source);
  deletedRows.cases = [deletedRows.cases[0]];
  deletedRows.items = deletedRows.items.filter(
    (entry: Record<string, any>) => entry.case_id === caseId,
  );
  assert.throws(() => readSceneConditionBatch(deletedRows, EXPECTED));
});

test("single-case scene snapshots reject tampered bindings and non-terminal completion", () => {
  const source = fixture();
  const caseId = source.cases[0].case_id;
  const single = singleSceneBatch(source, caseId);
  const mutations: Array<(value: Record<string, any>) => void> = [
    (value) => { value.dataset_id = ACTIVE_PROFILE.datasetId; },
    (value) => { value.selected_mode = "NB-wired"; },
    (value) => { value.algorithm_cluster_id = "agc_default"; },
    (value) => { value.selected_scene = "sce2"; },
    (value) => { value.cases[0].case_id = "different-case"; },
    (value) => { value.cases[0].conditions.mode = "NB-wired"; },
    (value) => { value.cases[0].input_sha256.microphone = hash("f"); },
    (value) => { value.items[0].evidence_binding.case_id = "different-case"; },
    (value) => { value.items[0].evidence_binding.mode = "NB-wired"; },
    (value) => { value.items[0].evidence_binding.input_sha256.noise = hash("f"); },
    (value) => { value.items[0].status = "pending"; value.items[0].value = null; },
    (value) => { value.cases[0].status = "pending"; },
    (value) => { value.conversation_counts = { near_single: 1 }; },
  ];
  for (const mutate of mutations) {
    const value = structuredClone(single);
    mutate(value);
    assert.throws(() => readSceneConditionBatch(value, EXPECTED));
  }

  const pending = structuredClone(single);
  pending.status = "queued";
  pending.cases[0].status = "pending";
  pending.items.forEach((item: Record<string, any>) => {
    item.status = "pending";
    item.value = null;
  });
  pending.progress = {
    completed_cases: 0,
    total_cases: 1,
    completed_items: 0,
    total_items: pending.items.length,
  };
  pending.summary = {
    total: pending.items.length,
    computed: 0,
    pass: 0,
    fail: 0,
    review_required: 0,
    not_measured: 0,
    invalid: 0,
  };
  assert.doesNotThrow(() =>
    readSceneConditionBatch(pending, { ...EXPECTED, caseId })
  );
});
