import type { FiveSignalMode } from "./five-signal-flow-model.ts";

export const ACCEPTANCE_CASE_CATALOG_SCHEMA =
  "acceptance-case-catalog-v1" as const;

export type CaseRunScope = "basic_uplink" | "basic_downlink";

export type CaseConditionValue = string | number | null;

export type AcceptanceCaseConditions = {
  mode: string;
  direction: string;
  scene: string;
  volume: string | number | null;
  window_ms: number | null;
  input_level_db: number | null;
  [key: string]: CaseConditionValue;
};

export type AcceptanceCaseRecord = {
  case_id: string;
  conditions: AcceptanceCaseConditions;
  item_ids: string[];
  signals_sha256: Record<string, string>;
};

import type { ScopedPreviewVisualTrack } from "./scoped-acoustic-model.ts";

export type AcceptanceCaseCatalog = {
  schema_version: typeof ACCEPTANCE_CASE_CATALOG_SCHEMA;
  dataset_id: string;
  dataset_manifest_sha256: string;
  test_scope: CaseRunScope;
  mode: FiveSignalMode;
  scene: null;
  case_count: number;
  observation_count: number;
  standard_compliance: false;
  formal_test_status: "not_measured";
  cases: AcceptanceCaseRecord[];
  visuals?: {
    cases: Record<string, Record<string, ScopedPreviewVisualTrack>>;
  };
};

export type ContractLimit = string | number | null;

export type CaseRunContractItem = {
  id: string;
  source_cell: string;
  test_name: string;
  family: string;
  direction: string;
  unit: string;
  lower_limit: ContractLimit;
  upper_limit: ContractLimit;
};

export type BasicContractCatalog = {
  test_scope: CaseRunScope;
  mode: FiveSignalMode;
  scene?: null;
  item_count: number;
  items: CaseRunContractItem[];
};

export type SingleCaseSelection = {
  execution_scope: "single_case";
  selected_case_id: string;
  dataset_case_count: number;
  dataset_observation_count: number;
  preview_condition_group_count: number;
};

export type CaseRunItem = {
  item_id: string;
  status: string;
  value: number | null;
  metric?: string | null;
  reason?: string;
  evidence_binding?: {
    case_id?: string;
    report_id?: string | null;
    [key: string]: unknown;
  };
};

export type CaseRunCase = {
  case_id: string;
  report_id: string;
  status: string;
  error?: string;
};

export type CaseRunProgress = {
  completed_cases: number;
  total_cases: number;
  completed_items: number;
  total_items: number;
};

export type CaseRunSummary = {
  total: number;
  computed: number;
  pass: number;
  fail: number;
  review_required: number;
  not_measured: number;
  invalid: number;
};

export type CaseRunSnapshot = {
  batch_id: string;
  dataset_id: string;
  dataset_manifest_sha256?: string;
  algorithm_cluster_id: string;
  status: string;
  test_scope: string;
  selected_mode: string;
  progress: CaseRunProgress;
  summary: CaseRunSummary;
  items: CaseRunItem[];
  cases?: CaseRunCase[];
  execution_scope?: string;
  selected_case_id?: string;
  dataset_case_count?: number;
  dataset_observation_count?: number;
  preview_condition_group_count?: number;
  case_count?: number;
  observation_count?: number;
};

export type CaseRunIdentity = {
  dataset_id: string;
  dataset_manifest_sha256?: string;
  algorithm_cluster_id: string;
  test_scope: CaseRunScope;
  selected_mode: FiveSignalMode;
};

export type CaseRunResult = {
  case_id: string;
  batch_id: string;
  status: string;
  report_id: string | null;
  error: string;
  items: CaseRunItem[];
};

export type CaseRunHistory = Record<string, CaseRunResult>;

const CASE_ID = /^[A-Za-z0-9_-]{1,128}$/;
const SHA256 = /^[a-f0-9]{64}$/;
const SINGLE_SELECTION_FIELDS = [
  "execution_scope",
  "selected_case_id",
  "dataset_case_count",
  "dataset_observation_count",
  "preview_condition_group_count",
] as const;
const TERMINAL_BATCH_STATUSES = new Set([
  "completed",
  "failed",
  "cancelled",
]);
const CASE_SCOPES = new Set<string>(["basic_uplink", "basic_downlink"]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isCount(value: unknown): value is number {
  return (
    typeof value === "number" &&
    Number.isInteger(value) &&
    value >= 0
  );
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isCaseId(value: unknown): value is string {
  return typeof value === "string" && CASE_ID.test(value);
}

function isCaseScope(value: unknown): value is CaseRunScope {
  return typeof value === "string" && CASE_SCOPES.has(value);
}

function isLimit(value: unknown): value is ContractLimit {
  return (
    value === null ||
    typeof value === "string" ||
    isFiniteNumber(value)
  );
}

function expectedItemCount(
  scope: CaseRunScope,
  mode: FiveSignalMode,
): number {
  if (scope === "basic_uplink") return 8;
  return mode.startsWith("SWB-") ? 8 : 9;
}

function expectedDirection(scope: CaseRunScope): "send" | "receive" {
  return scope === "basic_uplink" ? "send" : "receive";
}

function readConditions(
  value: unknown,
  scope: CaseRunScope,
  mode: FiveSignalMode,
): AcceptanceCaseConditions {
  if (!isRecord(value)) {
    throw new Error("单组目录 conditions 必须为对象");
  }
  if (
    value.mode !== mode ||
    value.direction !== expectedDirection(scope) ||
    typeof value.scene !== "string" ||
    !(
      value.volume === null ||
      typeof value.volume === "string" ||
      isFiniteNumber(value.volume)
    ) ||
    !(value.window_ms === null || isFiniteNumber(value.window_ms)) ||
    !isFiniteNumber(value.input_level_db)
  ) {
    throw new Error("单组目录 conditions 与当前范围或模式不一致");
  }
  for (const [key, entry] of Object.entries(value)) {
    if (
      entry !== null &&
      typeof entry !== "string" &&
      !isFiniteNumber(entry)
    ) {
      throw new Error(`单组目录 conditions.${key} 不是可显示的标量`);
    }
  }
  return value as AcceptanceCaseConditions;
}

function readSignals(value: unknown): Record<string, string> {
  if (!isRecord(value) || Object.keys(value).length === 0) {
    throw new Error("单组目录 signals_sha256 不能为空");
  }
  const result: Record<string, string> = {};
  for (const [role, digest] of Object.entries(value)) {
    if (
      role.length === 0 ||
      typeof digest !== "string" ||
      !SHA256.test(digest)
    ) {
      throw new Error("单组目录 signals_sha256 包含无效角色或摘要");
    }
    result[role] = digest;
  }
  return result;
}

export function readBasicContractCatalog(
  value: unknown,
  expectedScope: CaseRunScope,
  expectedMode: FiveSignalMode,
): BasicContractCatalog {
  const count = expectedItemCount(expectedScope, expectedMode);
  if (
    !isRecord(value) ||
    value.test_scope !== expectedScope ||
    value.mode !== expectedMode ||
    value.item_count !== count ||
    (value.scene !== undefined && value.scene !== null) ||
    !Array.isArray(value.items) ||
    value.items.length !== count
  ) {
    throw new Error("基础范围合同目录与当前范围或模式不一致");
  }
  const seen = new Set<string>();
  for (const item of value.items) {
    if (
      !isRecord(item) ||
      typeof item.id !== "string" ||
      item.id.length === 0 ||
      seen.has(item.id) ||
      typeof item.source_cell !== "string" ||
      typeof item.test_name !== "string" ||
      typeof item.family !== "string" ||
      item.direction !== expectedDirection(expectedScope) ||
      typeof item.unit !== "string" ||
      !isLimit(item.lower_limit) ||
      !isLimit(item.upper_limit)
    ) {
      throw new Error("基础范围合同目录包含无效或重复测项");
    }
    seen.add(item.id);
  }
  return value as BasicContractCatalog;
}

export function readAcceptanceCaseCatalog(
  value: unknown,
  expected: {
    datasetId: string;
    datasetManifestSha256?: string;
    scope: CaseRunScope;
    mode: FiveSignalMode;
    contractItemIds?: readonly string[];
  },
): AcceptanceCaseCatalog {
  if (
    !isRecord(value) ||
    value.schema_version !== ACCEPTANCE_CASE_CATALOG_SCHEMA ||
    value.dataset_id !== expected.datasetId ||
    typeof value.dataset_manifest_sha256 !== "string" ||
    !SHA256.test(value.dataset_manifest_sha256) ||
    (
      expected.datasetManifestSha256 !== undefined &&
      value.dataset_manifest_sha256 !== expected.datasetManifestSha256
    ) ||
    value.test_scope !== expected.scope ||
    value.mode !== expected.mode ||
    value.scene !== null ||
    !isCount(value.case_count) ||
    value.case_count === 0 ||
    !isCount(value.observation_count) ||
    value.observation_count === 0 ||
    value.standard_compliance !== false ||
    value.formal_test_status !== "not_measured" ||
    !Array.isArray(value.cases) ||
    value.cases.length !== value.case_count
  ) {
    throw new Error("单组目录身份、范围或正式合规边界不一致");
  }

  const caseIds = new Set<string>();
  const itemIds = new Set<string>();
  const contractIds = expected.contractItemIds
    ? new Set(expected.contractItemIds)
    : null;
  let observationCount = 0;
  const cases: AcceptanceCaseRecord[] = [];
  for (const entry of value.cases) {
    if (
      !isRecord(entry) ||
      !isCaseId(entry.case_id) ||
      caseIds.has(entry.case_id) ||
      !Array.isArray(entry.item_ids) ||
      entry.item_ids.length === 0
    ) {
      throw new Error("单组目录包含重复或无效 case_id/item_ids");
    }
    caseIds.add(entry.case_id);
    const conditions = readConditions(
      entry.conditions,
      expected.scope,
      expected.mode,
    );
    const signalsSha256 = readSignals(entry.signals_sha256);
    const caseItemIds: string[] = [];
    for (const itemId of entry.item_ids) {
      if (
        typeof itemId !== "string" ||
        itemId.length === 0 ||
        itemIds.has(itemId) ||
        (contractIds !== null && !contractIds.has(itemId))
      ) {
        throw new Error("单组目录 item_ids 重复或不属于当前合同范围");
      }
      itemIds.add(itemId);
      caseItemIds.push(itemId);
      observationCount += 1;
    }
    cases.push({
      case_id: entry.case_id,
      conditions,
      item_ids: caseItemIds,
      signals_sha256: signalsSha256,
    });
  }

  if (observationCount !== value.observation_count) {
    throw new Error("单组目录 observation_count 与实际 item_ids 不一致");
  }
  if (
    contractIds !== null &&
    (
      observationCount !== contractIds.size ||
      itemIds.size !== contractIds.size ||
      [...contractIds].some((itemId) => !itemIds.has(itemId))
    )
  ) {
    throw new Error("单组目录未唯一覆盖当前范围的全部合同测项");
  }

  const visuals =
    isRecord(value.visuals) && isRecord(value.visuals.cases)
      ? {
          cases: value.visuals.cases as Record<
            string,
            Record<string, ScopedPreviewVisualTrack>
          >,
        }
      : undefined;

  return {
    schema_version: ACCEPTANCE_CASE_CATALOG_SCHEMA,
    dataset_id: expected.datasetId,
    dataset_manifest_sha256: value.dataset_manifest_sha256,
    test_scope: expected.scope,
    mode: expected.mode,
    scene: null,
    case_count: value.case_count,
    observation_count: value.observation_count,
    standard_compliance: false,
    formal_test_status: "not_measured",
    cases,
    ...(visuals ? { visuals } : {}),
  };
}

export function readSingleCaseSelection(
  value: unknown,
): SingleCaseSelection | null {
  if (!isRecord(value)) {
    throw new Error("单组快照必须为对象");
  }
  const present = SINGLE_SELECTION_FIELDS.filter((field) =>
    Object.hasOwn(value, field),
  );
  if (present.length === 0) return null;
  if (present.length !== SINGLE_SELECTION_FIELDS.length) {
    throw new Error("单组快照必须同时包含全部选择元数据");
  }
  if (
    value.execution_scope !== "single_case" ||
    !isCaseId(value.selected_case_id) ||
    !isCount(value.dataset_case_count) ||
    value.dataset_case_count === 0 ||
    !isCount(value.dataset_observation_count) ||
    value.dataset_observation_count === 0 ||
    !isCount(value.preview_condition_group_count) ||
    value.preview_condition_group_count !== value.dataset_case_count
  ) {
    throw new Error("单组快照选择元数据无效");
  }
  return {
    execution_scope: "single_case",
    selected_case_id: value.selected_case_id,
    dataset_case_count: value.dataset_case_count,
    dataset_observation_count: value.dataset_observation_count,
    preview_condition_group_count: value.preview_condition_group_count,
  };
}

export function assertCaseExecutionSnapshot(
  value: CaseRunSnapshot,
  expected: {
    scope: CaseRunScope;
    mode: FiveSignalMode;
    fullItemCount: number;
    caseId?: string;
    datasetCaseCount?: number;
    datasetObservationCount?: number;
    caseItemIds?: readonly string[];
    contractItemIds?: readonly string[];
  },
): SingleCaseSelection | null {
  if (
    value.test_scope !== expected.scope ||
    value.selected_mode !== expected.mode
  ) {
    throw new Error("批次快照的测试范围或业务模式与请求不一致");
  }
  const selection = readSingleCaseSelection(value);
  if (expected.caseId === undefined) {
    if (selection !== null) {
      throw new Error("完整批次快照不能携带单组选择元数据");
    }
    if (
      value.items.length !== expected.fullItemCount ||
      value.progress.total_cases === 0 ||
      value.progress.total_items !== expected.fullItemCount ||
      value.summary.total !== expected.fullItemCount
    ) {
      throw new Error("完整批次必须恰好包含当前范围的全部合同测项");
    }
    return null;
  }

  if (
    selection === null ||
    selection.selected_case_id !== expected.caseId
  ) {
    throw new Error("单组批次返回的 selected_case_id 与请求不一致");
  }
  if (
    (
      expected.datasetCaseCount !== undefined &&
      selection.dataset_case_count !== expected.datasetCaseCount
    ) ||
    (
      expected.datasetObservationCount !== undefined &&
      selection.dataset_observation_count !==
        expected.datasetObservationCount
    )
  ) {
    throw new Error("单组批次返回的数据集规模与请求目录不一致");
  }
  if (
    value.items.length === 0 ||
    value.items.length > expected.fullItemCount ||
    value.progress.total_cases !== 1 ||
    value.progress.completed_cases > 1 ||
    value.progress.total_items !== value.items.length ||
    value.progress.completed_items > value.items.length ||
    value.summary.total !== value.items.length ||
    (value.case_count !== undefined && value.case_count !== 1) ||
    (
      value.observation_count !== undefined &&
      value.observation_count !== value.items.length
    )
  ) {
    throw new Error("单组批次进度、数量或观测身份不一致");
  }
  const contractIds = expected.contractItemIds
    ? new Set(expected.contractItemIds)
    : null;
  const itemIds = new Set<string>();
  for (const item of value.items) {
    if (
      itemIds.has(item.item_id) ||
      (contractIds !== null && !contractIds.has(item.item_id)) ||
      item.evidence_binding?.case_id !== selection.selected_case_id
    ) {
      throw new Error("单组批次逐项结果未完整绑定所选 case");
    }
    itemIds.add(item.item_id);
  }
  if (
    expected.caseItemIds !== undefined &&
    (
      expected.caseItemIds.length !== itemIds.size ||
      expected.caseItemIds.some((itemId) => !itemIds.has(itemId))
    )
  ) {
    throw new Error("单组批次逐项集合与目录 case.item_ids 不一致");
  }
  const cases = value.cases ?? [];
  if (
    cases.length > 1 ||
    cases.some((entry) => entry.case_id !== selection.selected_case_id) ||
    (
      TERMINAL_BATCH_STATUSES.has(value.status) &&
      cases.length !== 1
    )
  ) {
    throw new Error("单组批次的条件组记录与终态不一致");
  }
  return selection;
}

export function caseRunIdentityKey(identity: CaseRunIdentity): string {
  return JSON.stringify([
    identity.dataset_id,
    identity.dataset_manifest_sha256 ?? "",
    identity.test_scope,
    identity.selected_mode,
    identity.algorithm_cluster_id,
  ]);
}

export function sameCaseRunIdentity(
  snapshot: CaseRunSnapshot,
  identity: CaseRunIdentity,
): boolean {
  return (
    snapshot.dataset_id === identity.dataset_id &&
    snapshot.test_scope === identity.test_scope &&
    snapshot.selected_mode === identity.selected_mode &&
    snapshot.algorithm_cluster_id === identity.algorithm_cluster_id &&
    (
      identity.dataset_manifest_sha256 === undefined ||
      snapshot.dataset_manifest_sha256 ===
        identity.dataset_manifest_sha256
    )
  );
}

export function deriveCaseRunResults(
  snapshot: CaseRunSnapshot,
): CaseRunHistory {
  const selection = readSingleCaseSelection(snapshot);
  const records = new Map(
    (snapshot.cases ?? []).map((entry) => [entry.case_id, entry]),
  );
  const caseIds = new Set<string>();
  for (const item of snapshot.items) {
    const caseId = item.evidence_binding?.case_id;
    if (caseId) caseIds.add(caseId);
  }
  for (const caseId of records.keys()) caseIds.add(caseId);
  if (selection) caseIds.add(selection.selected_case_id);

  const result: CaseRunHistory = {};
  for (const caseId of caseIds) {
    const record = records.get(caseId);
    const items = snapshot.items.filter(
      (item) => item.evidence_binding?.case_id === caseId,
    );
    result[caseId] = {
      case_id: caseId,
      batch_id: snapshot.batch_id,
      status: record?.status ?? snapshot.status,
      report_id: record?.report_id ?? null,
      error: record?.error ?? "",
      items,
    };
  }
  return result;
}

export function mergeCaseRunHistory(
  history: CaseRunHistory,
  snapshot: CaseRunSnapshot,
  identity: CaseRunIdentity,
): CaseRunHistory {
  if (!sameCaseRunIdentity(snapshot, identity)) return history;
  const next = { ...history };
  for (const [caseId, entry] of Object.entries(
    deriveCaseRunResults(snapshot),
  )) {
    next[caseId] = entry;
  }
  return next;
}
