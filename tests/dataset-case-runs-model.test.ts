import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import {
  ACCEPTANCE_CASE_CATALOG_SCHEMA,
  assertCaseExecutionSnapshot,
  caseRunIdentityKey,
  mergeCaseRunHistory,
  readAcceptanceCaseCatalog,
  readBasicContractCatalog,
  readSingleCaseSelection,
  type AcceptanceCaseCatalog,
  type BasicContractCatalog,
  type CaseRunIdentity,
  type CaseRunItem,
  type CaseRunSnapshot,
} from "../src/features/dataset-case-runs-model.ts";

type ContractRow = {
  id: string;
  sheet: string;
  source_cell: string;
  test_name: string;
  family: string;
  direction: string;
  unit: string;
  lower_limit: number | string | null;
  upper_limit: number | string | null;
};

const DATASET_ID = "b".repeat(32);
const BATCH_ID = "a".repeat(32);
const MANIFEST_SHA = "c".repeat(64);
const SIGNAL_SHA = "d".repeat(64);
const contract = JSON.parse(
  readFileSync(
    new URL(
      "../cpp/evaluation/nv200-test-contract.json",
      import.meta.url,
    ),
    "utf8",
  ),
) as { items: ContractRow[] };

function contractCatalog(
  scope: "basic_uplink" | "basic_downlink",
  mode: "WB-wired",
): BasicContractCatalog {
  const direction = scope === "basic_uplink" ? "send" : "receive";
  const rows = contract.items.filter(
    (row) => row.sheet === mode && row.direction === direction,
  );
  const selected = scope === "basic_uplink"
    ? rows.filter((row) =>
        [
          "slr",
          "frequency_response",
          "listening_quality",
          "idle_noise_level",
          "idle_noise_peaks",
          "out_of_band",
          "hats_snr",
        ].includes(row.family),
      )
    : rows;
  return {
    test_scope: scope,
    mode,
    scene: null,
    item_count: selected.length,
    items: selected.map((row) => ({
      id: row.id,
      source_cell: row.source_cell,
      test_name: row.test_name,
      family: row.family,
      direction: row.direction,
      unit: row.unit,
      lower_limit: row.lower_limit,
      upper_limit: row.upper_limit,
    })),
  };
}

function caseCatalog(
  scope: "basic_uplink" | "basic_downlink",
  split = 3,
): AcceptanceCaseCatalog {
  const contract = contractCatalog(scope, "WB-wired");
  const direction = scope === "basic_uplink" ? "send" : "receive";
  const cases = Array.from({ length: split }, (_, index) => {
    const itemIds = contract.items
      .filter((__, itemIndex) => itemIndex % split === index)
      .map((item) => item.id);
    return {
      case_id: `case-${scope}-${index + 1}`,
      conditions: {
        mode: "WB-wired" as const,
        direction,
        scene: `sce${(index % 4) + 1}`,
        volume: "nominal" as const,
        window_ms: index === 0 ? 50 as const : null,
        input_level_db: 0,
      },
      item_ids: itemIds,
      signals_sha256: {
        microphone: SIGNAL_SHA,
        reference: SIGNAL_SHA,
      },
    };
  });
  return {
    schema_version: ACCEPTANCE_CASE_CATALOG_SCHEMA,
    dataset_id: DATASET_ID,
    dataset_manifest_sha256: MANIFEST_SHA,
    test_scope: scope,
    mode: "WB-wired",
    scene: null,
    case_count: cases.length,
    observation_count: contract.item_count,
    standard_compliance: false,
    formal_test_status: "not_measured",
    cases,
  };
}

function item(
  itemId: string,
  caseId: string,
  status = "pending",
  value: number | null = null,
): CaseRunItem {
  return {
    item_id: itemId,
    status,
    value,
    metric: null,
    reason: status === "pending" ? "等待对应场景执行" : "",
    evidence_binding: {
      case_id: caseId,
      report_id: status === "pending" ? null : `report-${caseId}`,
    },
  };
}

function singleSnapshot(
  catalog: AcceptanceCaseCatalog,
  caseIndex = 0,
): CaseRunSnapshot {
  const record = catalog.cases[caseIndex];
  const items = record.item_ids.map((itemId) =>
    item(itemId, record.case_id),
  );
  return {
    batch_id: BATCH_ID,
    dataset_id: DATASET_ID,
    dataset_manifest_sha256: MANIFEST_SHA,
    algorithm_cluster_id: "vcp_default",
    status: "queued",
    test_scope: catalog.test_scope,
    selected_mode: catalog.mode,
    progress: {
      completed_cases: 0,
      total_cases: 1,
      completed_items: 0,
      total_items: items.length,
    },
    summary: {
      total: items.length,
      computed: 0,
      pass: 0,
      fail: 0,
      review_required: 0,
      not_measured: 0,
      invalid: 0,
    },
    items,
    cases: [],
    execution_scope: "single_case",
    selected_case_id: record.case_id,
    dataset_case_count: catalog.case_count,
    dataset_observation_count: catalog.observation_count,
    preview_condition_group_count: catalog.case_count,
    case_count: 1,
    observation_count: items.length,
  };
}

function fullSnapshot(
  catalog: AcceptanceCaseCatalog,
): CaseRunSnapshot {
  const items = catalog.cases.flatMap((record) =>
    record.item_ids.map((itemId) => item(itemId, record.case_id)),
  );
  return {
    batch_id: BATCH_ID,
    dataset_id: DATASET_ID,
    dataset_manifest_sha256: MANIFEST_SHA,
    algorithm_cluster_id: "vcp_default",
    status: "queued",
    test_scope: catalog.test_scope,
    selected_mode: catalog.mode,
    progress: {
      completed_cases: 0,
      total_cases: catalog.case_count,
      completed_items: 0,
      total_items: items.length,
    },
    summary: {
      total: items.length,
      computed: 0,
      pass: 0,
      fail: 0,
      review_required: 0,
      not_measured: 0,
      invalid: 0,
    },
    items,
    cases: [],
  };
}

function identity(
  catalog: AcceptanceCaseCatalog,
): CaseRunIdentity {
  return {
    dataset_id: catalog.dataset_id,
    dataset_manifest_sha256: catalog.dataset_manifest_sha256,
    algorithm_cluster_id: "vcp_default",
    test_scope: catalog.test_scope,
    selected_mode: catalog.mode,
  };
}

test("单组目录严格绑定当前范围、模式并唯一覆盖合同 item_ids", () => {
  const contract = contractCatalog("basic_uplink", "WB-wired");
  const raw = caseCatalog("basic_uplink");
  const parsed = readAcceptanceCaseCatalog(raw, {
    datasetId: DATASET_ID,
    datasetManifestSha256: MANIFEST_SHA,
    scope: "basic_uplink",
    mode: "WB-wired",
    contractItemIds: contract.items.map((item) => item.id),
  });

  assert.equal(parsed.case_count, 3);
  assert.equal(parsed.observation_count, 8);
  assert.equal(
    parsed.cases.flatMap((entry) => entry.item_ids).length,
    contract.item_count,
  );
  assert.equal(
    new Set(parsed.cases.flatMap((entry) => entry.item_ids)).size,
    contract.item_count,
  );
  assert.equal(parsed.standard_compliance, false);
  assert.equal(parsed.formal_test_status, "not_measured");
  assert.equal(parsed.scene, null);
});

test("基础下行单组目录覆盖 9 个接收合同观测", () => {
  const contract = contractCatalog("basic_downlink", "WB-wired");
  const parsed = readAcceptanceCaseCatalog(
    caseCatalog("basic_downlink"),
    {
      datasetId: DATASET_ID,
      datasetManifestSha256: MANIFEST_SHA,
      scope: "basic_downlink",
      mode: "WB-wired",
      contractItemIds: contract.items.map((item) => item.id),
    },
  );
  assert.equal(contract.item_count, 9);
  assert.equal(parsed.observation_count, 9);
  assert.equal(
    parsed.cases.every((entry) =>
      entry.conditions.direction === "receive",
    ),
    true,
  );
});

test("单组目录拒绝重复 item、覆盖缺口、错误场景与伪合同范围", () => {
  const contract = contractCatalog("basic_uplink", "WB-wired");
  const expected = {
    datasetId: DATASET_ID,
    datasetManifestSha256: MANIFEST_SHA,
    scope: "basic_uplink" as const,
    mode: "WB-wired" as const,
    contractItemIds: contract.items.map((item) => item.id),
  };

  const duplicate = structuredClone(caseCatalog("basic_uplink"));
  duplicate.cases[1].item_ids[0] =
    duplicate.cases[0].item_ids[0];
  assert.throws(() =>
    readAcceptanceCaseCatalog(duplicate, expected),
  );

  const missing = structuredClone(caseCatalog("basic_uplink"));
  missing.cases[0].item_ids.pop();
  missing.observation_count -= 1;
  assert.throws(() =>
    readAcceptanceCaseCatalog(missing, expected),
  );

  const wrongScene = structuredClone(caseCatalog("basic_uplink"));
  wrongScene.scene = "sce1" as never;
  assert.throws(() =>
    readAcceptanceCaseCatalog(wrongScene, expected),
  );

  const wrongManifest = structuredClone(caseCatalog("basic_uplink"));
  wrongManifest.dataset_manifest_sha256 = "e".repeat(64);
  assert.throws(() =>
    readAcceptanceCaseCatalog(wrongManifest, expected),
  );
});

test("单组快照要求完整元数据、实际 item 数和 case 绑定", () => {
  const catalog = caseCatalog("basic_uplink");
  const contract = contractCatalog("basic_uplink", "WB-wired");
  const snapshot = singleSnapshot(catalog);
  const selection = assertCaseExecutionSnapshot(snapshot, {
    scope: "basic_uplink",
    mode: "WB-wired",
    fullItemCount: contract.item_count,
    caseId: catalog.cases[0].case_id,
    datasetCaseCount: catalog.case_count,
    datasetObservationCount: catalog.observation_count,
    caseItemIds: catalog.cases[0].item_ids,
    contractItemIds: contract.items.map((item) => item.id),
  });
  assert.equal(selection?.selected_case_id, catalog.cases[0].case_id);

  const wrongCase = structuredClone(snapshot);
  wrongCase.items[0].evidence_binding!.case_id = "other-case";
  assert.throws(() =>
    assertCaseExecutionSnapshot(wrongCase, {
      scope: "basic_uplink",
      mode: "WB-wired",
      fullItemCount: contract.item_count,
      caseId: catalog.cases[0].case_id,
      caseItemIds: catalog.cases[0].item_ids,
    }),
  );

  const partial = structuredClone(snapshot) as Record<string, unknown>;
  delete partial.preview_condition_group_count;
  assert.throws(() => readSingleCaseSelection(partial));

  const fakeFull = {
    ...fullSnapshot(catalog),
    items: snapshot.items,
    progress: snapshot.progress,
    summary: snapshot.summary,
  };
  assert.throws(() =>
    assertCaseExecutionSnapshot(fakeFull, {
      scope: "basic_uplink",
      mode: "WB-wired",
      fullItemCount: contract.item_count,
    }),
  );
});

test("同一身份历史保留多组结果，跨 dataset/mode/algorithm 不合并", () => {
  const catalog = caseCatalog("basic_uplink");
  const expected = identity(catalog);
  let history = mergeCaseRunHistory(
    {},
    singleSnapshot(catalog, 0),
    expected,
  );
  history = mergeCaseRunHistory(
    history,
    singleSnapshot(catalog, 1),
    expected,
  );
  assert.deepEqual(
    Object.keys(history).sort(),
    catalog.cases.slice(0, 2).map((entry) => entry.case_id).sort(),
  );

  const crossMode = {
    ...singleSnapshot(catalog, 2),
    selected_mode: "NB-wired",
  };
  const before = structuredClone(history);
  const after = mergeCaseRunHistory(history, crossMode, expected);
  assert.deepEqual(after, before);

  const full = fullSnapshot(catalog);
  const fullHistory = mergeCaseRunHistory({}, full, expected);
  assert.equal(Object.keys(fullHistory).length, catalog.case_count);
  assert.notEqual(
    caseRunIdentityKey(expected),
    caseRunIdentityKey({
      ...expected,
      selected_mode: "NB-wired",
    }),
  );
});

test("单组目录支持可选 precomputed visuals 校验并兼容缺失", () => {
  const catalog = caseCatalog("basic_downlink");
  const withVisuals = {
    ...catalog,
    visuals: {
      cases: {
        [catalog.cases[0].case_id]: {
          far_end: {
            source_sha256: SIGNAL_SHA,
            duration_s: 12,
            sample_rate: 8000,
            channels: 1,
            peak: 0.25,
            wave_scale: 1,
            wave_min_i16: "AAAA",
            wave_max_i16: "AAAA",
            mel_i16: "AAAA",
            mel_ceiling_db: -10,
          },
        },
      },
    },
  };
  const contract = contractCatalog("basic_downlink", "WB-wired");
  const parsed = readAcceptanceCaseCatalog(withVisuals, {
    datasetId: DATASET_ID,
    datasetManifestSha256: MANIFEST_SHA,
    scope: "basic_downlink",
    mode: "WB-wired",
    contractItemIds: contract.items.map((item) => item.id),
  });
  assert.equal(
    parsed.visuals?.cases[catalog.cases[0].case_id]?.far_end?.sample_rate,
    8000,
  );
  const malformedVisuals = {
    ...catalog,
    visuals: { cases: "not-an-object" },
  };
  const fallback = readAcceptanceCaseCatalog(malformedVisuals, {
    datasetId: DATASET_ID,
    datasetManifestSha256: MANIFEST_SHA,
    scope: "basic_downlink",
    mode: "WB-wired",
    contractItemIds: contract.items.map((item) => item.id),
  });
  assert.equal(fallback.visuals, undefined);
});
