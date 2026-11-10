import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import {
  FIVE_SIGNAL_MODES,
  type FiveSignalMode,
} from "../src/features/five-signal-flow-model.ts";
import {
  BACKGROUND_NOISE_SCENES,
  BACKGROUND_NOISE_SCENE_CELLS,
  BACKGROUND_NOISE_SCENE_LABELS,
  SCENE_SUBPAGE_IDS,
  SCENE_SUBPAGE_LABELS,
  matchingScopedSnapshot,
  readScopedCatalog,
  readScopedBackgroundNoisePreview,
  readScopedDatasets,
  readScopedDatasetView,
  readScopedHistory,
  readScopedSnapshot,
  isIdle20ConditionPreview,
  readSceneConditionDatasetView,
  readSceneConditionPreview,
  SCENE_CONDITION_DATASET_KIND,
  SCENE_CONDITION_SCHEMA_VERSION,
  sceneSubpageFromHash,
  sceneSubpageHash,
  sceneSubpageRoute,
  scopedFamilyMatches,
  scopedCaseEvidenceIdentity,
  scopedItemCount,
  scopedMetricUnit,
  type AcousticTestScope,
  type BackgroundNoiseScene,
  type ScopedCatalog,
  type ScopedInputPreviewCase,
  type ScopedResultItem,
  type ScopedSnapshot,
} from "../src/features/scoped-acoustic-model.ts";

type ContractItem = {
  id: string;
  sheet: FiveSignalMode;
  source_cell: string;
  test_name: string;
  family: string;
  direction: string;
  unit: string;
  lower_limit: number | string | null;
  upper_limit: number | string | null;
};

type Conditions = {
  mode: FiveSignalMode;
  direction: string;
  scene: "baseline" | "sce1" | "sce2" | "sce3" | "sce4";
  volume: "nominal" | "recommended" | "maximum";
  window_ms: 50 | 100 | null;
  input_level_db: number;
};

type FixtureState = "pending" | "partial" | "completed";
type TerminalStatus = "pass" | "fail" | "review_required" | "not_measured" | "invalid";
type ItemStatus = ScopedResultItem["status"];

const contract = JSON.parse(
  readFileSync(
    new URL("../cpp/evaluation/nv200-test-contract.json", import.meta.url),
    "utf8",
  ),
) as { items: ContractItem[] };

const SCOPES: AcousticTestScope[] = [
  "basic_downlink",
  "echo",
  "double_talk",
  "background_noise",
];
const NOISE_FAMILIES = new Set([
  "g_mos",
  "n_mos",
  "s_mos",
  "cng_level",
  "cng_spectrum",
]);
const BATCH_ID = "a".repeat(32);
const DATASET_ID = "b".repeat(32);
const ALGORITHM_ID = "vcp-balanced";
const CREATED_AT = "2026-11-05T08:30:00.000Z";

function selectedContractRows(
  scope: AcousticTestScope,
  mode: FiveSignalMode,
  scene?: BackgroundNoiseScene,
): ContractItem[] {
  return contract.items.filter((row) => {
    if (row.sheet !== mode) return false;
    if (scope === "basic_downlink") return row.direction === "receive";
    if (scope === "echo") return row.family === "tclw";
    if (scope === "double_talk") {
      return row.family === "double_talk_classification";
    }
    return NOISE_FAMILIES.has(row.family)
      && (scene === undefined
        || BACKGROUND_NOISE_SCENE_CELLS[scene].has(row.source_cell));
  });
}

function catalogFor(
  scope: AcousticTestScope,
  mode: FiveSignalMode,
  scene?: BackgroundNoiseScene,
): ScopedCatalog {
  const items = selectedContractRows(scope, mode, scene).map((row) => ({
    id: `${row.sheet}:${row.source_cell}`,
    source_cell: row.source_cell,
    test_name: row.test_name,
    family: row.family,
    direction: row.direction,
    unit: row.unit,
    lower_limit: row.lower_limit,
    upper_limit: row.upper_limit,
  }));
  return {
    test_scope: scope,
    mode,
    scene,
    item_count: items.length,
    items,
  };
}

// This is the independent, contract-facing form of
// server/acoustic/acceptance_contract.py::required_conditions.
function requiredConditions(row: ContractItem): Conditions {
  const name = row.test_name.toLowerCase();
  const scene = name.match(/sce\s*([1-4])/);
  const window = name.match(/(50|100)\s*ms/);
  const compactName = name.replaceAll(" ", "");
  const volume = name.includes("recommended")
    ? "recommended"
    : /\b(maximum|max)\b/.test(name)
      ? "maximum"
      : "nominal";
  const inputLevel = compactName.includes("-5db")
    ? -5
    : compactName.includes("+5db")
      ? 5
      : 0;
  return {
    mode: row.sheet,
    direction: row.direction,
    scene: scene ? (`sce${scene[1]}` as Conditions["scene"]) : "baseline",
    volume,
    window_ms: window ? (Number(window[1]) as 50 | 100) : null,
    input_level_db: inputLevel,
  };
}

function caseGroups(rows: ContractItem[]) {
  const groups = new Map<
    string,
    { case_id: string; conditions: Conditions; rows: ContractItem[] }
  >();
  for (const row of rows) {
    const conditions = requiredConditions(row);
    const key = JSON.stringify(conditions);
    const group = groups.get(key);
    if (group) {
      group.rows.push(row);
    } else {
      groups.set(key, {
        case_id: `case-${String(groups.size + 1).padStart(2, "0")}`,
        conditions,
        rows: [row],
      });
    }
  }
  return [...groups.values()];
}

function valueFor(status: ItemStatus, index: number): number | null {
  if (status === "pass") return Number((1.25 + index / 10).toFixed(2));
  if (status === "fail") return -1 - index / 10;
  if (status === "review_required") return index % 2 === 0 ? null : 2.5;
  return null;
}

function summarize(items: ScopedResultItem[]) {
  const summary = {
    total: items.length,
    computed: 0,
    pass: 0,
    fail: 0,
    review_required: 0,
    not_measured: 0,
    invalid: 0,
  };
  for (const item of items) {
    if (item.status !== "pending" && item.status !== "running") {
      summary[item.status] += 1;
    }
    if (
      typeof item.value === "number" &&
      item.status !== "pending" &&
      item.status !== "running" &&
      item.status !== "not_measured" &&
      item.status !== "invalid"
    ) {
      summary.computed += 1;
    }
  }
  return summary;
}

function makeSnapshot(
  state: FixtureState,
  scope: AcousticTestScope = "basic_downlink",
  mode: FiveSignalMode = "NB-wired",
  scene?: BackgroundNoiseScene,
): ScopedSnapshot {
  const rows = selectedContractRows(scope, mode, scene);
  const groups = caseGroups(rows);
  const completedGroups =
    state === "pending"
      ? []
      : state === "partial"
        ? groups.slice(0, 1)
        : groups;
  const completedItemIds = new Set(
    completedGroups.flatMap((group) => group.rows.map((row) => row.id)),
  );
  const caseByItem = new Map(
    groups.flatMap((group) => group.rows.map((row) => [row.id, group.case_id])),
  );
  const completedCaseByItem = new Map(
    completedGroups.flatMap((group) =>
      group.rows.map((row) => [row.id, group.case_id]),
    ),
  );
  const terminalPattern: TerminalStatus[] = [
    "pass",
    "fail",
    "review_required",
    "not_measured",
    "invalid",
    "pass",
    "fail",
    "review_required",
  ];

  const items = rows.map((row, index) => {
    const status: ItemStatus =
      state === "pending"
        ? "pending"
        : completedItemIds.has(row.id)
          ? terminalPattern[index % terminalPattern.length]
          : "pending";
    const caseId = caseByItem.get(row.id)!;
    return {
      item_id: `${mode}:${row.source_cell}`,
      source_cell: row.source_cell,
      test_name: row.test_name,
      family: row.family,
      direction: row.direction,
      unit: row.unit,
      lower_limit: row.lower_limit,
      upper_limit: row.upper_limit,
      mode,
      status,
      value: valueFor(status, index),
      metric: row.unit,
      reason:
        status === "pending"
          ? "等待对应场景执行"
          : status === "not_measured"
            ? "合同仅要求曲线或未声明数值门限"
            : status === "invalid"
              ? "执行证据无效"
              : undefined,
      evidence_binding: {
        ...requiredConditions(row),
        case_id: caseId,
        report_id: completedCaseByItem.has(row.id)
          ? `report-${completedCaseByItem.get(row.id)}`
          : null,
        batch_id: BATCH_ID,
        dataset_id: DATASET_ID,
      },
    } satisfies ScopedResultItem;
  });
  const summary = summarize(items);
  return {
    batch_id: BATCH_ID,
    dataset_id: DATASET_ID,
    dataset_name: "NV200 2026-11-05",
    algorithm_cluster_id: ALGORITHM_ID,
    test_scope: scope,
    selected_mode: mode,
    selected_scene: scene ?? null,
    created_at: CREATED_AT,
    status:
      state === "pending"
        ? "queued"
        : state === "partial"
          ? "running"
          : "completed",
    progress: {
      completed_cases: completedGroups.length,
      total_cases: groups.length,
      completed_items: completedItemIds.size,
      total_items: rows.length,
    },
    summary,
    standard_compliance: false,
    items,
    cases: completedGroups.map((group) => ({
      case_id: group.case_id,
      report_id: `report-${group.case_id}`,
      status: "completed",
    })),
    error: null,
  };
}

function rejectsSnapshot(
  label: string,
  mutate: (snapshot: ScopedSnapshot) => void,
  scope: AcousticTestScope = "basic_downlink",
) {
  const snapshot = makeSnapshot("completed");
  mutate(snapshot);
  assert.throws(() => readScopedSnapshot(snapshot, scope), Error, label);
}

test("eight contract modes and all four scopes agree with real NV200 rows", () => {
  assert.equal(contract.items.length, 336);
  assert.equal(new Set(contract.items.map((row) => row.id)).size, 336);
  assert.deepEqual(
    [...new Set(contract.items.map((row) => row.sheet))].sort(),
    [...FIVE_SIGNAL_MODES].sort(),
  );

  for (const mode of FIVE_SIGNAL_MODES) {
    const selected = SCOPES.flatMap((scope) =>
      selectedContractRows(scope, mode),
    );
    assert.equal(
      new Set(selected.map((row) => row.id)).size,
      selected.length,
      `${mode} 的四范围合同项不应重叠`,
    );
    assert.equal(selected.length, 33, mode);

    for (const scope of SCOPES) {
      const rows = selectedContractRows(scope, mode);
      const expected = rows.length;
      assert.equal(scopedItemCount(scope, mode), expected, `${scope}/${mode}`);
      assert.equal(
        readScopedCatalog(catalogFor(scope, mode), scope, mode).items.length,
        expected,
        `${scope}/${mode}`,
      );
      assert.ok(
        rows.every((row) => scopedFamilyMatches(scope, row)),
        `${scope}/${mode} 的合同族与方向应全部匹配`,
      );
    }
  }

  const familyCounts = (scope: AcousticTestScope, mode: FiveSignalMode) =>
    selectedContractRows(scope, mode).reduce<Record<string, number>>(
      (counts, row) => {
        counts[row.family] = (counts[row.family] ?? 0) + 1;
        return counts;
      },
      {},
    );
  assert.deepEqual(familyCounts("basic_downlink", "NB-wired"), {
    rlr: 2,
    rlr_linearity: 2,
    frequency_response: 1,
    idle_noise_level: 1,
    idle_noise_peaks: 1,
    listening_quality: 1,
    out_of_band: 1,
  });
  assert.deepEqual(familyCounts("basic_downlink", "SWB-wired"), {
    rlr: 2,
    frequency_response: 1,
    listening_quality: 1,
    idle_noise_level: 2,
    idle_noise_peaks: 2,
  });
  assert.deepEqual(familyCounts("echo", "NB-wired"), { tclw: 2 });
  assert.deepEqual(familyCounts("echo", "SWB-wired"), { tclw: 3 });
  assert.deepEqual(familyCounts("double_talk", "WB-wireless"), {
    double_talk_classification: 2,
  });
  assert.deepEqual(familyCounts("background_noise", "SWB-wireless"), {
    g_mos: 4,
    n_mos: 4,
    s_mos: 4,
    cng_level: 4,
    cng_spectrum: 4,
  });
  assert.equal(
    Object.entries(familyCounts("background_noise", "NB-bt"))
      .filter(([family]) => family !== "cng_level" && family !== "cng_spectrum")
      .reduce((sum, [, count]) => sum + count, 0),
    12,
  );
  assert.equal(
    Object.entries(familyCounts("background_noise", "NB-bt"))
      .filter(([family]) => family === "cng_level" || family === "cng_spectrum")
      .reduce((sum, [, count]) => sum + count, 0),
    8,
  );
});

test("four background-noise scenes expose five isolated contract items each", () => {
  assert.deepEqual(BACKGROUND_NOISE_SCENES, [
    "sce1",
    "sce2",
    "sce3",
    "sce4",
  ]);
  assert.deepEqual(
    BACKGROUND_NOISE_SCENES.map((scene) => BACKGROUND_NOISE_SCENE_LABELS[scene]),
    ["怠速 / 静止", "60 km/h 市区", "120 km/h 高速", "空调大风档"],
  );

  for (const scene of BACKGROUND_NOISE_SCENES) {
    const catalog = readScopedCatalog(
      catalogFor("background_noise", "WB-wired", scene),
      "background_noise",
      "WB-wired",
      scene,
    );
    assert.equal(scopedItemCount("background_noise", "WB-wired", scene), 5);
    assert.equal(catalog.item_count, 5);
    assert.deepEqual(
      new Set(catalog.items.map((item) => item.source_cell)),
      new Set(BACKGROUND_NOISE_SCENE_CELLS[scene]),
    );

    const snapshot = readScopedSnapshot(
      makeSnapshot("completed", "background_noise", "WB-wired", scene),
      "background_noise",
      { mode: "WB-wired", scene },
    );
    assert.equal(snapshot.selected_scene, scene);
    assert.equal(snapshot.progress.total_items, 5);
    assert.ok(
      snapshot.items.every(
        (item) => item.evidence_binding?.scene === scene,
      ),
    );
    assert.equal(
      readScopedHistory(
        [snapshot],
        "background_noise",
        scene,
      ).length,
      1,
    );
  }

  const wrongScene = catalogFor("background_noise", "WB-wired", "sce1");
  assert.throws(() =>
    readScopedCatalog(
      wrongScene,
      "background_noise",
      "WB-wired",
      "sce2",
    ),
  );
  assert.throws(() =>
    scopedItemCount("echo", "WB-wired", "sce1"),
  );
});

test("catalog rejects wrong scope, mode, count, duplicate identity, B-cell, and family", () => {
  const catalog = catalogFor("basic_downlink", "NB-wired");
  assert.throws(() =>
    readScopedCatalog(catalog, "echo", "NB-wired"),
  );

  const wrongMode = structuredClone(catalog);
  wrongMode.mode = "WB-wired";
  assert.throws(() => readScopedCatalog(wrongMode, "basic_downlink", "NB-wired"));

  const wrongCount = structuredClone(catalog);
  wrongCount.item_count = 8;
  assert.throws(() =>
    readScopedCatalog(wrongCount, "basic_downlink", "NB-wired"),
  );

  const duplicate = structuredClone(catalog);
  duplicate.items[1].id = duplicate.items[0].id;
  duplicate.items[1].source_cell = duplicate.items[0].source_cell;
  assert.throws(() =>
    readScopedCatalog(duplicate, "basic_downlink", "NB-wired"),
  );

  const wrongCell = structuredClone(catalog);
  wrongCell.items[0].source_cell = "B44";
  wrongCell.items[0].id = "NB-wired:B44";
  assert.throws(() =>
    readScopedCatalog(wrongCell, "basic_downlink", "NB-wired"),
  );

  const wrongFamily = structuredClone(catalog);
  wrongFamily.items[0].family = "slr";
  assert.throws(() =>
    readScopedCatalog(wrongFamily, "basic_downlink", "NB-wired"),
  );

  assert.throws(() =>
    scopedItemCount("unknown" as AcousticTestScope, "NB-wired"),
  );
  assert.throws(() =>
    scopedItemCount("echo", "FB-wired" as FiveSignalMode),
  );
});

test("pending, partial, and completed snapshots preserve real condition bindings", () => {
  const pending = makeSnapshot("pending");
  const pendingRead = readScopedSnapshot(pending, "basic_downlink", {
    batchId: BATCH_ID,
    datasetId: DATASET_ID,
    algorithmClusterId: ALGORITHM_ID,
    mode: "NB-wired",
  });
  assert.equal(pendingRead.status, "queued");
  assert.equal(pendingRead.progress.completed_items, 0);
  assert.equal(pendingRead.cases.length, 0);
  assert.ok(pendingRead.items.every((item) => item.status === "pending"));
  assert.ok(pendingRead.items.every((item) => item.value === null));

  const partial = makeSnapshot("partial");
  const partialRead = readScopedSnapshot(partial, "basic_downlink");
  assert.equal(partialRead.status, "running");
  assert.equal(
    partialRead.cases.length,
    partialRead.progress.completed_cases,
  );
  assert.ok(
    partialRead.items.some((item) => item.status === "pending") &&
      partialRead.items.some((item) => item.status !== "pending"),
  );

  const completed = makeSnapshot("completed");
  const completedRead = readScopedSnapshot(completed, "basic_downlink");
  assert.equal(completedRead.status, "completed");
  assert.equal(
    completedRead.cases.length,
    completedRead.progress.completed_cases,
  );
  assert.equal(
    completedRead.progress.completed_items,
    completedRead.progress.total_items,
  );
  assert.ok(
    completedRead.items.every(
      (item) => item.status !== "pending" && item.status !== "running",
    ),
  );
  assert.equal(completedRead.summary.pass, 3);
  assert.equal(completedRead.summary.fail, 2);
  assert.equal(completedRead.summary.review_required, 2);
  assert.equal(completedRead.summary.not_measured, 1);
  assert.equal(completedRead.summary.invalid, 1);
  assert.equal(completedRead.summary.computed, 6);
  assert.equal(
    completedRead.summary.computed,
    completedRead.items.filter(
      (item) =>
        typeof item.value === "number" &&
        item.status !== "not_measured" &&
        item.status !== "invalid",
    ).length,
  );

  for (const snapshot of [pendingRead, partialRead, completedRead]) {
    for (const item of snapshot.items) {
      const binding = item.evidence_binding!;
      assert.equal(binding.mode, snapshot.selected_mode);
      assert.equal(binding.direction, item.direction);
      assert.ok(
        ["baseline", "sce1", "sce2", "sce3", "sce4"].includes(
          binding.scene,
        ),
      );
      assert.ok(
        ["nominal", "recommended", "maximum"].includes(String(binding.volume)),
      );
      assert.ok(
        binding.window_ms === null ||
          binding.window_ms === 50 ||
          binding.window_ms === 100,
      );
      assert.ok(Number.isFinite(binding.input_level_db));
      assert.equal(binding.batch_id, snapshot.batch_id);
      assert.equal(binding.dataset_id, snapshot.dataset_id);
      assert.ok(binding.case_id.length > 0);
      assert.ok(
        binding.report_id === null || binding.report_id.length > 0,
      );
    }
  }

  const nbConditions = completedRead.items.map((item) => item.evidence_binding!);
  assert.deepEqual(
    new Set(nbConditions.map((binding) => binding.input_level_db)),
    new Set([-5, 0, 5]),
  );
  assert.ok(
    nbConditions.some(
      (binding) => binding.volume === "maximum" && binding.window_ms === null,
    ),
  );
  assert.ok(
    nbConditions.some(
      (binding) => binding.volume === "nominal" && binding.window_ms === 100,
    ),
  );

  const swb = readScopedSnapshot(
    makeSnapshot("completed", "basic_downlink", "SWB-wired"),
    "basic_downlink",
  );
  assert.deepEqual(
    new Set(swb.items.map((item) => item.evidence_binding!.window_ms)),
    new Set([null, 50, 100]),
  );
  const swbEcho = readScopedSnapshot(
    makeSnapshot("completed", "echo", "SWB-wired"),
    "echo",
  );
  assert.deepEqual(
    new Set(swbEcho.items.map((item) => item.evidence_binding!.volume)),
    new Set(["nominal", "recommended", "maximum"]),
  );
});

test("snapshot rejects scope, mode, duplicate item, wrong B-cell, and non-finite values", () => {
  rejectsSnapshot("错误范围", (snapshot) => {
    snapshot.test_scope = "echo";
  });
  rejectsSnapshot("错误模式", (snapshot) => {
    snapshot.selected_mode = "WB-wired";
  });
  rejectsSnapshot("重复测项身份", (snapshot) => {
    snapshot.items[1].item_id = snapshot.items[0].item_id;
  });
  rejectsSnapshot("B 号越界", (snapshot) => {
    snapshot.items[0].source_cell = "B44";
    snapshot.items[0].item_id = "NB-wired:B44";
  });
  rejectsSnapshot("NaN 数值", (snapshot) => {
    snapshot.items[0].value = Number.NaN;
  });
  rejectsSnapshot("不可测项携带数值", (snapshot) => {
    const item = snapshot.items.find(
      (candidate) => candidate.status === "not_measured",
    )!;
    item.value = 1;
  });
  rejectsSnapshot("无效项携带数值", (snapshot) => {
    const item = snapshot.items.find(
      (candidate) => candidate.status === "invalid",
    )!;
    item.value = 1;
  });
});

test("snapshot rejects incomplete, wrong, or missing evidence bindings", () => {
  const bindingMutations: Array<
    [string, (binding: NonNullable<ScopedResultItem["evidence_binding"]>) => void]
  > = [
    ["错误绑定模式", (binding) => (binding.mode = "WB-wired")],
    ["错误绑定方向", (binding) => (binding.direction = "send")],
    ["错误场景", (binding) => (binding.scene = "sce9")],
    ["错误音量", (binding) => (binding.volume = "loud")],
    ["错误窗长", (binding) => (binding.window_ms = 25)],
    ["非有限输入电平", (binding) => (binding.input_level_db = Infinity)],
    ["跨批次绑定", (binding) => (binding.batch_id = "c".repeat(32))],
    ["跨数据集绑定", (binding) => (binding.dataset_id = "d".repeat(32))],
    [
      "缺少条件组身份",
      (binding) => {
        delete (binding as Partial<typeof binding>).case_id;
      },
    ],
    [
      "缺少报告身份",
      (binding) => {
        delete (binding as Partial<typeof binding>).report_id;
      },
    ],
  ];
  for (const [label, mutate] of bindingMutations) {
    rejectsSnapshot(label, (snapshot) => {
      mutate(snapshot.items[0].evidence_binding!);
    });
  }
});

test("summary and completed status cannot be forged independently of items and cases", () => {
  rejectsSnapshot("伪造 pass 汇总", (snapshot) => {
    snapshot.summary.pass += 1;
  });
  rejectsSnapshot("伪造 computed 汇总", (snapshot) => {
    snapshot.summary.computed += 1;
  });
  rejectsSnapshot("未完成却标记 completed", (snapshot) => {
    snapshot.progress.completed_items -= 1;
  });
  rejectsSnapshot("仍有待执行项却标记 completed", (snapshot) => {
    snapshot.items[0].status = "pending";
    snapshot.items[0].value = null;
  });
  rejectsSnapshot("条件组少于 completed_cases", (snapshot) => {
    snapshot.cases.pop();
  });
  rejectsSnapshot("重复条件组身份", (snapshot) => {
    snapshot.cases[1].case_id = snapshot.cases[0].case_id;
  });

  const pending = makeSnapshot("pending");
  pending.items[0].value = 1;
  assert.throws(() => readScopedSnapshot(pending, "basic_downlink"));
});

test("history and datasets validate the same scoped identity and non-negative counts", () => {
  const completed = makeSnapshot("completed");
  const history = readScopedHistory([completed], "basic_downlink");
  assert.equal(history.length, 1);
  assert.equal(history[0].batch_id, BATCH_ID);
  assert.throws(() =>
    readScopedHistory([completed], "background_noise"),
  );

  const wrongModeHistory = structuredClone(completed);
  wrongModeHistory.selected_mode = "FB-wired" as FiveSignalMode;
  assert.throws(() =>
    readScopedHistory([wrongModeHistory], "basic_downlink"),
  );

  const datasets = readScopedDatasets([
    {
      dataset_id: DATASET_ID,
      name: "NV200 2026-11-05",
      item_count: 336,
      case_count: 42,
      created_at: CREATED_AT,
    },
  ]);
  assert.equal(datasets[0].dataset_id, DATASET_ID);
  assert.throws(() =>
    readScopedDatasets([
      {
        dataset_id: DATASET_ID,
        name: "invalid",
        item_count: -1,
        case_count: 1,
        created_at: CREATED_AT,
      },
    ]),
  );
  assert.throws(() =>
    readScopedDatasets([
      {
        dataset_id: "not-a-hex-id",
        name: "invalid",
        item_count: 1,
        case_count: 1,
        created_at: CREATED_AT,
      },
    ]),
  );
});

test("a scene dataset view exposes exactly one validated scene dataset", () => {
  const dataset = {
    dataset_id: DATASET_ID,
    name: "Sce1 measurement-aware",
    item_count: 336,
    case_count: 180,
    created_at: CREATED_AT,
    manifest_sha256: "a".repeat(64),
    assets_sha256: "b".repeat(64),
  };
  const view = readScopedDatasetView(
    {
      test_scope: "background_noise",
      mode: "WB-wired",
      scene: "sce1",
      scene_item_count: 5,
      scene_case_count: 2,
      scene_signal_count: 7,
      total_dataset_count: 6,
      deduplicated_dataset_count: 5,
      datasets: [dataset],
    },
    { mode: "WB-wired", scene: "sce1" },
  );
  assert.equal(view.datasets.length, 1);
  assert.equal(view.scene_item_count, 5);
  assert.equal(view.scene_case_count, 2);
  assert.equal(view.total_dataset_count, 6);
  assert.throws(() =>
    readScopedDatasetView(
      {
        test_scope: "background_noise",
        mode: "WB-wired",
        scene: "sce1",
        scene_item_count: 5,
        scene_case_count: 2,
        total_dataset_count: 6,
        datasets: [dataset, { ...dataset, dataset_id: "b".repeat(32) }],
      },
      { mode: "WB-wired", scene: "sce1" },
    ),
  );
  assert.throws(() =>
    readScopedDatasetView(
      {
        test_scope: "background_noise",
        mode: "WB-wired",
        scene: "sce1",
        scene_item_count: 20,
        scene_case_count: 8,
        total_dataset_count: 6,
        datasets: [dataset],
      },
      { mode: "WB-wired", scene: "sce1" },
    ),
  );
});

test("a previous report is never borrowed after mode, algorithm, or dataset changes", () => {
  const completed = readScopedSnapshot(
    makeSnapshot("completed"),
    "basic_downlink",
  );
  assert.equal(
    matchingScopedSnapshot(completed, "NB-wired", ALGORITHM_ID, DATASET_ID),
    completed,
  );
  assert.equal(
    matchingScopedSnapshot(completed, "WB-wired", ALGORITHM_ID, DATASET_ID),
    null,
  );
  assert.equal(
    matchingScopedSnapshot(completed, "NB-wired", "vcp-fast", DATASET_ID),
    null,
  );
  assert.equal(
    matchingScopedSnapshot(completed, "NB-wired", ALGORITHM_ID, "c".repeat(32)),
    null,
  );

  assert.throws(() =>
    readScopedSnapshot(completed, "basic_downlink", {
      mode: "WB-wired",
    }),
  );
  assert.throws(() =>
    readScopedSnapshot(completed, "basic_downlink", {
      algorithmClusterId: "vcp-fast",
    }),
  );
  assert.throws(() =>
    readScopedSnapshot(completed, "basic_downlink", {
      datasetId: "c".repeat(32),
    }),
  );
  assert.throws(() =>
    readScopedSnapshot(completed, "basic_downlink", {
      batchId: "c".repeat(32),
    }),
  );
});

test("metric units distinguish predicted scores, dBFS, dB, percent, ratio, and time", () => {
  assert.equal(scopedMetricUnit("dnsmos_p835"), "预测分");
  assert.equal(scopedMetricUnit("vendor_mos_proxy"), "预测分");
  assert.equal(scopedMetricUnit("level_dbfs"), "dBFS");
  assert.equal(scopedMetricUnit("tclw_db"), "dB");
  assert.equal(scopedMetricUnit("latency_ms"), "ms");
  assert.equal(scopedMetricUnit("coverage_percent"), "%");
  assert.equal(scopedMetricUnit("improvement_ratio"), "比例");
  assert.equal(scopedMetricUnit("plain_mos"), "");
  assert.equal(scopedMetricUnit(null), "");
  assert.equal(scopedMetricUnit(undefined), "");
});

test("background-noise waveform preview validates identity, cases, and all four inputs", () => {
  const preview = {
    dataset_id: DATASET_ID,
    dataset_name: "measurement-aware-v2-not-certification",
    dataset_manifest_sha256: "d".repeat(64),
    dataset_assets_sha256: "e".repeat(64),
    test_scope: "background_noise",
    mode: "WB-wired",
    scene: "sce1",
    cases: [
      {
        case_id: "v2-0039-speech",
        conditions: {
          mode: "WB-wired",
          direction: "send",
          scene: "sce1",
          volume: "nominal",
          window_ms: null,
          input_level_db: 0,
        },
        families: ["g_mos", "n_mos", "s_mos"],
        signals: ["far_end", "microphone", "noise", "reference"],
        signals_sha256: {
          far_end: "1".repeat(64),
          microphone: "2".repeat(64),
          noise: "3".repeat(64),
          reference: "4".repeat(64),
        },
      },
      {
        case_id: "v2-0040-cng",
        conditions: {
          mode: "WB-wired",
          direction: "send",
          scene: "sce1",
          volume: "nominal",
          window_ms: null,
          input_level_db: 0,
        },
        families: ["cng_level", "cng_spectrum"],
        signals: ["far_end", "microphone", "noise", "reference"],
        signals_sha256: {
          far_end: "5".repeat(64),
          microphone: "6".repeat(64),
          noise: "7".repeat(64),
          reference: "8".repeat(64),
        },
      },
    ],
  };
  const parsed = readScopedBackgroundNoisePreview(
    preview,
    DATASET_ID,
    "WB-wired",
    "sce1",
  );
  assert.deepEqual(
    parsed.cases.map((item) => item.case_id),
    ["v2-0039-speech", "v2-0040-cng"],
  );

  assert.throws(() =>
    readScopedBackgroundNoisePreview(
      { ...preview, dataset_id: "c".repeat(32) },
      DATASET_ID,
      "WB-wired",
      "sce1",
    ),
  );
  assert.throws(() =>
    readScopedBackgroundNoisePreview(
      { ...preview, mode: "NB-wired" },
      DATASET_ID,
      "WB-wired",
      "sce1",
    ),
  );
  assert.throws(() =>
    readScopedBackgroundNoisePreview(
      {
        ...preview,
        cases: [
          {
            ...preview.cases[0],
            signals: ["far_end", "noise", "reference"],
          },
        ],
      },
      DATASET_ID,
      "WB-wired",
      "sce1",
    ),
  );
  assert.throws(() =>
    readScopedBackgroundNoisePreview(
      {
        ...preview,
        cases: [preview.cases[0], preview.cases[0]],
      },
      DATASET_ID,
      "WB-wired",
      "sce1",
    ),
  );
  assert.throws(() =>
    readScopedBackgroundNoisePreview(
      {
        ...preview,
        cases: [
          {
            ...preview.cases[0],
            signals_sha256: {
              ...preview.cases[0].signals_sha256,
              microphone: "not-a-sha256",
            },
          },
        ],
      },
      DATASET_ID,
      "WB-wired",
      "sce1",
    ),
  );
  assert.throws(() =>
    readScopedBackgroundNoisePreview(
      {
        ...preview,
        cases: [
          {
            ...preview.cases[0],
            signals: [
              "far_end",
              "microphone",
              "near_end",
              "reference",
            ],
            signals_sha256: {
              far_end: "1".repeat(64),
              microphone: "2".repeat(64),
              near_end: "3".repeat(64),
              reference: "4".repeat(64),
            },
          },
        ],
      },
      DATASET_ID,
      "WB-wired",
      "sce1",
    ),
  );
});

test("background-noise preview validates compact server-side visuals", () => {
  const wave = "A".repeat(1_920);
  const mel = "A".repeat(25_600);
  const cases = [
    {
      case_id: "v2-0039-speech",
      conditions: {
        mode: "WB-wired",
        direction: "send",
        scene: "sce1",
        volume: "nominal",
        window_ms: null,
        input_level_db: 0,
      },
      families: ["g_mos", "n_mos", "s_mos"],
      signals: ["far_end", "microphone", "noise", "reference"],
      signals_sha256: {
        far_end: "1".repeat(64),
        microphone: "2".repeat(64),
        noise: "3".repeat(64),
        reference: "4".repeat(64),
      },
    },
  ];
  const track = (sourceSha256: string) => ({
    source_sha256: sourceSha256,
    duration_s: 12,
    sample_rate: 16_000,
    channels: 1,
    peak: 0.25,
    wave_scale: 1,
    wave_min_i16: wave,
    wave_max_i16: wave,
    mel_i16: mel,
    mel_ceiling_db: -5.5,
  });
  const preview = {
    dataset_id: DATASET_ID,
    dataset_name: "measurement-aware-v2-not-certification",
    dataset_manifest_sha256: "d".repeat(64),
    dataset_assets_sha256: "e".repeat(64),
    test_scope: "background_noise",
    mode: "WB-wired",
    scene: "sce1",
    cases,
    visuals: {
      schema: "scoped-acoustic-visual-v1",
      waveform_bins: 720,
      mel_bands: 40,
      mel_frames: 240,
      mel_fft_size: 512,
      mel_hop_ms: 10,
      mel_dynamic_range_db: 80,
      mel_db_offset_db: -200,
      mel_db_step_db: 0.1,
      cases: {
        "v2-0039-speech": {
          microphone: track("2".repeat(64)),
          far_end: track("1".repeat(64)),
          reference: track("4".repeat(64)),
          noise: track("3".repeat(64)),
        },
      },
    },
  };

  const parsed = readScopedBackgroundNoisePreview(
    preview,
    DATASET_ID,
    "WB-wired",
    "sce1",
  );
  assert.equal(
    parsed.visuals?.cases["v2-0039-speech"].microphone?.source_sha256,
    "2".repeat(64),
  );

  assert.throws(() =>
    readScopedBackgroundNoisePreview(
      {
        ...preview,
        visuals: {
          ...preview.visuals,
          mel_frames: 100,
        },
      },
      DATASET_ID,
      "WB-wired",
      "sce1",
    ),
  );
  assert.throws(() =>
    readScopedBackgroundNoisePreview(
      {
        ...preview,
        visuals: {
          ...preview.visuals,
          cases: {
            "v2-0039-speech": {
              ...preview.visuals.cases["v2-0039-speech"],
              microphone: {
                ...preview.visuals.cases["v2-0039-speech"].microphone,
                wave_min_i16: "A",
              },
            },
          },
        },
      },
      DATASET_ID,
      "WB-wired",
      "sce1",
    ),
  );
  assert.throws(() =>
    readScopedBackgroundNoisePreview(
      {
        ...preview,
        visuals: {
          ...preview.visuals,
          cases: {
            "v2-0039-speech": {
              ...preview.visuals.cases["v2-0039-speech"],
              microphone: {
                ...preview.visuals.cases["v2-0039-speech"].microphone,
                source_sha256: "f".repeat(64),
              },
            },
          },
        },
      },
      DATASET_ID,
      "WB-wired",
      "sce1",
    ),
  );
});

test("TX waveform gate requires every case item to share report and input hashes", () => {
  const snapshot = readScopedSnapshot(
    makeSnapshot("completed", "background_noise", "WB-wired", "sce1"),
    "background_noise",
    { mode: "WB-wired", scene: "sce1" },
  );
  snapshot.dataset_manifest_sha256 = "d".repeat(64);
  const caseRecord = snapshot.cases[0];
  const caseItems = snapshot.items.filter(
    (item) => item.evidence_binding?.case_id === caseRecord.case_id,
  );
  const hashes = {
    microphone: "2".repeat(64),
    far_end: "1".repeat(64),
    reference: "4".repeat(64),
    noise: "3".repeat(64),
  };
  for (const item of caseItems) {
    item.evidence_binding!.report_id = caseRecord.report_id;
    item.evidence_binding!.input_sha256 = { ...hashes };
  }
  const previewCase: ScopedInputPreviewCase = {
    case_id: caseRecord.case_id,
    conditions: {
      mode: "WB-wired",
      direction: "send",
      scene: "sce1",
      volume: "nominal",
      window_ms: null,
      input_level_db: 0,
    },
    families: ["g_mos", "n_mos", "s_mos"],
    signals: ["far_end", "microphone", "noise", "reference"],
    signals_sha256: { ...hashes },
  };

  assert.equal(
    scopedCaseEvidenceIdentity(
      previewCase,
      snapshot,
      snapshot.dataset_manifest_sha256,
    ).txReady,
    true,
  );

  const wrongReport = structuredClone(snapshot);
  const wrongReportItem = wrongReport.items.find(
    (item) => item.evidence_binding?.case_id === caseRecord.case_id,
  )!;
  wrongReportItem.evidence_binding!.report_id = "different-report";
  assert.equal(
    scopedCaseEvidenceIdentity(
      previewCase,
      wrongReport,
      wrongReport.dataset_manifest_sha256!,
    ).txReady,
    false,
  );

  const wrongHash = structuredClone(snapshot);
  const wrongHashItem = wrongHash.items.find(
    (item) => item.evidence_binding?.case_id === caseRecord.case_id,
  )!;
  wrongHashItem.evidence_binding!.input_sha256 = {
    ...hashes,
    noise: "f".repeat(64),
  };
  assert.equal(
    scopedCaseEvidenceIdentity(
      previewCase,
      wrongHash,
      wrongHash.dataset_manifest_sha256!,
    ).txReady,
    false,
  );
});

test("scene subpage hash parser keeps the base route and validates tab", () => {
  const route = "background-noise-idle";
  assert.equal(sceneSubpageRoute("#background-noise-idle"), route);
  assert.equal(sceneSubpageRoute("#background-noise-idle?tab=results"), route);
  assert.equal(sceneSubpageRoute("background-noise-idle?tab=waveforms"), route);
  assert.equal(sceneSubpageFromHash(`#${route}`, route), "execute");
  for (const subpage of SCENE_SUBPAGE_IDS) {
    assert.equal(
      sceneSubpageFromHash(sceneSubpageHash(route, subpage), route),
      subpage,
    );
    assert.equal(
      SCENE_SUBPAGE_LABELS[subpage],
      ({ execute: "执行", results: "结果", waveforms: "波形", description: "说明" } as const)[subpage],
    );
  }
  assert.equal(sceneSubpageHash(route, "execute"), `#${route}`);
  assert.equal(
    sceneSubpageHash(route, "results"),
    `#${route}?tab=results`,
  );
  assert.equal(
    sceneSubpageFromHash(`#${route}?tab=results&foo=bar`, route),
    "results",
  );
  assert.equal(sceneSubpageFromHash(`#${route}?tab=`, route), "execute");
  assert.equal(sceneSubpageFromHash(`#${route}?tab=unknown`, route), "execute");
  assert.equal(sceneSubpageFromHash(`#${route}?tab=RESULTS`, route), "execute");
  assert.equal(
    sceneSubpageFromHash(`#${route}?tab=results&tab=waveforms`, route),
    "execute",
  );
  assert.equal(sceneSubpageFromHash(`#${route}/results`, route), "execute");
  assert.equal(
    sceneSubpageFromHash(`#${route}?tab=results`, "background-noise-urban"),
    "execute",
  );
});

function sceneConditionPreviewFixture(caseCount = 20) {
  const roles = ["microphone", "reference", "far_end", "noise"] as const;
  const wave = "A".repeat(1_920);
  const mel = "A".repeat(25_600);
  const cases = Array.from({ length: caseCount }, (_, caseIndex) => {
    const caseId = `sce1-preview-${String(caseIndex + 1).padStart(2, "0")}`;
    const hashes = Object.fromEntries(
      roles.map((role, roleIndex) => [
        role,
        ((caseIndex * roles.length + roleIndex + 1) % 256)
          .toString(16)
          .padStart(2, "0")
          .repeat(32),
      ]),
    );
    return {
      case_id: caseId,
      conditions: {
        mode: "WB-wired",
        direction: "send",
        scene: "sce1",
        volume: "nominal",
        window_ms: null,
        input_level_db: 0,
      },
      families: ["g_mos", "n_mos", "s_mos"],
      signals: [...roles],
      signals_sha256: hashes,
      contract_item_ids: caseIndex < 10
        ? ["WB-wired:B24", "WB-wired:B25", "WB-wired:B26"]
        : ["WB-wired:B27", "WB-wired:B28"],
      provenance: {
        engineering_preview: true,
        actual_vehicle_recording: false,
      },
    };
  });
  const visualCases = Object.fromEntries(
    cases.map((previewCase) => [
      previewCase.case_id,
      Object.fromEntries(
        roles.map((role) => [
          role,
          {
            source_sha256: previewCase.signals_sha256[role],
            duration_s: 12,
            sample_rate: 16_000,
            channels: role === "microphone" ? 2 : 1,
            peak: 0.25,
            wave_scale: 1,
            wave_min_i16: wave,
            wave_max_i16: wave,
            mel_i16: mel,
            mel_ceiling_db: -5.5,
          },
        ]),
      ),
    ]),
  );
  return {
    dataset_id: "edea6f9a51df87d0a4aefb2303dd8e1f",
    dataset_name: "background-noise-sce1-wb-wired-20-condition-preview-v1",
    dataset_manifest_sha256: "d".repeat(64),
    dataset_assets_sha256: "e".repeat(64),
    test_scope: "background_noise",
    mode: "WB-wired",
    scene: "sce1",
    cases,
    visuals: {
      schema: "scoped-acoustic-visual-v1",
      waveform_bins: 720,
      mel_bands: 40,
      mel_frames: 240,
      mel_fft_size: 512,
      mel_hop_ms: 10,
      mel_dynamic_range_db: 80,
      mel_db_offset_db: -200,
      mel_db_step_db: 0.1,
      cases: visualCases,
    },
    preview_condition_group_count: 20,
    contract_item_count: 5,
    standard_compliance: false,
    formal_test_status: "not_measured",
    dataset_kind: "background_noise_idle_preview",
    schema_version: "scene-condition-preview-v1",
  };
}

function sceneConditionDatasetViewFixture() {
  return {
    test_scope: "background_noise",
    mode: "WB-wired",
    scene: "sce1",
    preview_condition_group_count: 20,
    contract_item_count: 5,
    standard_compliance: false,
    formal_test_status: "not_measured",
    datasets: [
      {
        dataset_id: "edea6f9a51df87d0a4aefb2303dd8e1f",
        dataset_name: "background-noise-sce1-wb-wired-20-condition-preview-v1",
        test_scope: "background_noise",
        mode: "WB-wired",
        scene: "sce1",
        contract_item_count: 5,
        preview_condition_group_count: 20,
        standard_compliance: false,
        formal_test_status: "not_measured",
      },
    ],
  };
}

test("scene-condition dataset view accepts exactly one 20-condition preview", () => {
  const view = sceneConditionDatasetViewFixture();
  const parsed = readSceneConditionDatasetView(
    view,
    { mode: "WB-wired", scene: "sce1" },
  );
  assert.equal(parsed.datasets.length, 1);
  assert.equal(parsed.preview_condition_group_count, 20);
  assert.equal(parsed.contract_item_count, 5);
  assert.equal(parsed.standard_compliance, false);
  assert.equal(parsed.formal_test_status, "not_measured");

  assert.throws(() =>
    readSceneConditionDatasetView(
      { ...view, datasets: [] },
      { mode: "WB-wired", scene: "sce1" },
    ),
  );
  assert.throws(() =>
    readSceneConditionDatasetView(
      { ...view, preview_condition_group_count: 19 },
      { mode: "WB-wired", scene: "sce1" },
    ),
  );
  assert.throws(() =>
    readSceneConditionDatasetView(
      { ...view, standard_compliance: true },
      { mode: "WB-wired", scene: "sce1" },
    ),
  );
  assert.throws(() =>
    readSceneConditionDatasetView(
      view,
      { mode: "WB-wired", scene: "sce2" },
    ),
  );
  assert.throws(() =>
    readSceneConditionDatasetView(
      { ...view, test_scope: "echo" },
      { mode: "WB-wired", scene: "sce1" },
    ),
  );
  assert.throws(() =>
    readSceneConditionDatasetView(
      { ...view, preview_condition_group_count: 21 },
      { mode: "WB-wired", scene: "sce1" },
    ),
  );
  const wrongDataset = sceneConditionDatasetViewFixture();
  wrongDataset.datasets[0].mode = "NB-wired";
  assert.throws(() =>
    readSceneConditionDatasetView(
      wrongDataset,
      { mode: "WB-wired", scene: "sce1" },
    ),
  );
  const missingDatasetFields = sceneConditionDatasetViewFixture();
  delete (missingDatasetFields.datasets[0] as Record<string, unknown>)
    .standard_compliance;
  assert.throws(() =>
    readSceneConditionDatasetView(
      missingDatasetFields,
      { mode: "WB-wired", scene: "sce1" },
    ),
  );
});

test("scene-condition preview validates all 20 cases and four visual tracks", () => {
  const preview = sceneConditionPreviewFixture();
  const parsed = readSceneConditionPreview(
    preview,
    preview.dataset_id,
    "WB-wired",
    "sce1",
  );
  assert.equal(parsed.cases.length, 20);
  assert.equal(new Set(parsed.cases.map((item) => item.case_id)).size, 20);
  assert.equal(Object.keys(parsed.visuals!.cases).length, 20);
  for (const previewCase of parsed.cases) {
    assert.deepEqual(
      Object.keys(parsed.visuals!.cases[previewCase.case_id]).sort(),
      ["far_end", "microphone", "noise", "reference"],
    );
  }

  const nineteenCases = sceneConditionPreviewFixture(19);
  assert.throws(() =>
    readSceneConditionPreview(
      nineteenCases,
      nineteenCases.dataset_id,
      "WB-wired",
      "sce1",
    ),
  );

  const duplicateCase = sceneConditionPreviewFixture();
  duplicateCase.cases[1].case_id = duplicateCase.cases[0].case_id;
  assert.throws(() =>
    readSceneConditionPreview(
      duplicateCase,
      duplicateCase.dataset_id,
      "WB-wired",
      "sce1",
    ),
  );

  const invalidHash = sceneConditionPreviewFixture();
  invalidHash.cases[0].signals_sha256.microphone = "not-a-sha256";
  assert.throws(() =>
    readSceneConditionPreview(
      invalidHash,
      invalidHash.dataset_id,
      "WB-wired",
      "sce1",
    ),
  );

  const compliantClaim = sceneConditionPreviewFixture();
  compliantClaim.standard_compliance = true;
  assert.throws(() =>
    readSceneConditionPreview(
      compliantClaim,
      compliantClaim.dataset_id,
      "WB-wired",
      "sce1",
    ),
  );

  const withoutVisuals = sceneConditionPreviewFixture() as Record<
    string,
    unknown
  >;
  delete withoutVisuals.visuals;
  assert.throws(() =>
    readSceneConditionPreview(
      withoutVisuals,
      withoutVisuals.dataset_id as string,
      "WB-wired",
      "sce1",
    ),
  );

  const incompleteVisuals = sceneConditionPreviewFixture();
  delete incompleteVisuals.visuals.cases[
    incompleteVisuals.cases[0].case_id
  ].noise;
  assert.throws(() =>
    readSceneConditionPreview(
      incompleteVisuals,
      incompleteVisuals.dataset_id,
      "WB-wired",
      "sce1",
    ),
  );

  const wrongCount = sceneConditionPreviewFixture();
  wrongCount.preview_condition_group_count = 21;
  assert.throws(() =>
    readSceneConditionPreview(
      wrongCount,
      wrongCount.dataset_id,
      "WB-wired",
      "sce1",
    ),
  );

  const wrongSchema = sceneConditionPreviewFixture();
  wrongSchema.schema_version = "scene-condition-preview-v2";
  assert.throws(() =>
    readSceneConditionPreview(
      wrongSchema,
      wrongSchema.dataset_id,
      "WB-wired",
      "sce1",
    ),
  );

  const wrongDatasetKind = sceneConditionPreviewFixture();
  wrongDatasetKind.dataset_kind = "other-kind";
  assert.throws(() =>
    readSceneConditionPreview(
      wrongDatasetKind,
      wrongDatasetKind.dataset_id,
      "WB-wired",
      "sce1",
    ),
  );

  const missingContractCoverage = sceneConditionPreviewFixture();
  missingContractCoverage.cases.forEach((previewCase, index) => {
    if (index >= 10) {
      previewCase.contract_item_ids = [];
    }
  });
  assert.throws(() =>
    readSceneConditionPreview(
      missingContractCoverage,
      missingContractCoverage.dataset_id,
      "WB-wired",
      "sce1",
    ),
  );

  const badProvenance = sceneConditionPreviewFixture();
  badProvenance.cases[0].provenance.actual_vehicle_recording = true;
  assert.throws(() =>
    readSceneConditionPreview(
      badProvenance,
      badProvenance.dataset_id,
      "WB-wired",
      "sce1",
    ),
  );

  const badConditions = sceneConditionPreviewFixture();
  badConditions.cases[0].conditions.volume = "loud";
  assert.throws(() =>
    readSceneConditionPreview(
      badConditions,
      badConditions.dataset_id,
      "WB-wired",
      "sce1",
    ),
  );

  const badVisualSemantics = sceneConditionPreviewFixture();
  badVisualSemantics.visuals.cases[
    badVisualSemantics.cases[0].case_id
  ].microphone.duration_s = 0.001;
  assert.throws(() =>
    readSceneConditionPreview(
      badVisualSemantics,
      badVisualSemantics.dataset_id,
      "WB-wired",
      "sce1",
    ),
  );

  const badVisualChannels = sceneConditionPreviewFixture();
  badVisualChannels.visuals.cases[
    badVisualChannels.cases[0].case_id
  ].noise.channels = 2;
  assert.throws(() =>
    readSceneConditionPreview(
      badVisualChannels,
      badVisualChannels.dataset_id,
      "WB-wired",
      "sce1",
    ),
  );

  const missingVisualsLegacy = sceneConditionPreviewFixture() as Record<
    string,
    unknown
  >;
  delete missingVisualsLegacy.visuals;
  assert.doesNotThrow(() =>
    readScopedBackgroundNoisePreview(
      missingVisualsLegacy,
      missingVisualsLegacy.dataset_id as string,
      "WB-wired",
      "sce1",
    ),
  );

  assert.equal(
    SCENE_CONDITION_SCHEMA_VERSION,
    sceneConditionPreviewFixture().schema_version,
  );
  assert.equal(
    SCENE_CONDITION_DATASET_KIND,
    sceneConditionPreviewFixture().dataset_kind,
  );
});

test("idle 20-condition mode is limited to Sce1 WB-wired previews", () => {
  assert.equal(
    isIdle20ConditionPreview(
      { previewMode: "idle-20-condition", scene: "sce1" },
      "WB-wired",
    ),
    true,
  );
  assert.equal(
    isIdle20ConditionPreview(
      { previewMode: "idle-20-condition", scene: "sce2" },
      "WB-wired",
    ),
    false,
  );
  assert.equal(
    isIdle20ConditionPreview(
      { previewMode: "idle-20-condition", scene: "sce1" },
      "NB-wired",
    ),
    false,
  );
  assert.equal(
    isIdle20ConditionPreview({ scene: "sce1" }, "WB-wired"),
    false,
  );
});
