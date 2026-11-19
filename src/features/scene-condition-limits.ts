import { FIVE_SIGNAL_MODES } from "./five-signal-flow-model.ts";
import type { SceneConditionItemId } from "./scene-condition-profiles.ts";

export type SceneConditionLimitState =
  | "meets"
  | "below"
  | "pending"
  | "not_measured"
  | "invalid"
  | "unavailable"
  | "not_applicable";

export type SceneConditionLimitObservationStatus =
  | "pending"
  | "running"
  | "pass"
  | "fail"
  | "review_required"
  | "not_measured"
  | "invalid";

export type SceneConditionLimitResult = {
  state: SceneConditionLimitState;
  lowerLimit: number | null;
  sourceCell: string;
};

export type SceneConditionLimitSummary = {
  comparable: number;
  meets: number;
  below: number;
  pending: number;
  notMeasured: number;
  invalid: number;
  unavailable: number;
  notApplicable: number;
};

const CELL_LOWER_LIMITS: Record<string, number> = {
  B24: 1,
  B25: 3,
  B26: 3,
  B29: 1,
  B30: 3,
  B31: 3,
  B34: 1,
  B35: 2.5,
  B36: 2.5,
  B39: 1,
  B40: 2.5,
  B41: 2.5,
};

// NV200 Lower Limit (column E) across all 8 business modes. SWB sce1 N-MOS/S-MOS
// (B25/B26) have lower limit 4 in the customer workbook; all other modes use 3.
export const SCENE_CONDITION_TABLE_LOWER_LIMITS: Partial<
  Record<SceneConditionItemId, number>
> = Object.fromEntries(
  FIVE_SIGNAL_MODES.flatMap((mode) =>
    Object.entries(CELL_LOWER_LIMITS).map(([cell, baseLimit]) => {
      const limit =
        mode.startsWith("SWB-") && (cell === "B25" || cell === "B26")
          ? 4
          : baseLimit;
      return [`${mode}:${cell}`, limit];
    }),
  ),
) as Partial<Record<SceneConditionItemId, number>>;

// NV200 WB-wired Lower Limit (column E). Kept for backward compatibility.
export const WB_WIRED_TABLE_LOWER_LIMITS: Partial<
  Record<SceneConditionItemId, number>
> = {
  "WB-wired:B24": 1,
  "WB-wired:B25": 3,
  "WB-wired:B26": 3,
  "WB-wired:B29": 1,
  "WB-wired:B30": 3,
  "WB-wired:B31": 3,
  "WB-wired:B34": 1,
  "WB-wired:B35": 2.5,
  "WB-wired:B36": 2.5,
  "WB-wired:B39": 1,
  "WB-wired:B40": 2.5,
  "WB-wired:B41": 2.5,
};

export function sceneConditionTableLowerLimit(
  itemId: SceneConditionItemId,
): number | null {
  return (
    SCENE_CONDITION_TABLE_LOWER_LIMITS[itemId]
    ?? WB_WIRED_TABLE_LOWER_LIMITS[itemId]
    ?? null
  );
}

export function evaluateSceneConditionTableLowerLimit(
  itemId: SceneConditionItemId,
  value: number | null | undefined,
  status?: SceneConditionLimitObservationStatus,
): SceneConditionLimitResult {
  const lowerLimit = sceneConditionTableLowerLimit(itemId);
  const sourceCell = itemId.slice(itemId.indexOf(":") + 1);
  if (lowerLimit === null) {
    return {
      state: "not_applicable",
      lowerLimit: null,
      sourceCell,
    };
  }
  if (status === "not_measured") {
    return {
      state: "not_measured",
      lowerLimit,
      sourceCell,
    };
  }
  if (status === "invalid") {
    return {
      state: "invalid",
      lowerLimit,
      sourceCell,
    };
  }
  if (status === "pending" || status === "running") {
    return {
      state: "pending",
      lowerLimit,
      sourceCell,
    };
  }
  if (
    typeof value !== "number"
    || !Number.isFinite(value)
  ) {
    return {
      state: "unavailable",
      lowerLimit,
      sourceCell,
    };
  }
  return {
    state: value >= lowerLimit ? "meets" : "below",
    lowerLimit,
    sourceCell,
  };
}

export function summarizeSceneConditionTableLimits(
  items: readonly {
    item_id: SceneConditionItemId;
    value: number | null;
    status?: SceneConditionLimitObservationStatus;
  }[],
): SceneConditionLimitSummary {
  const summary: SceneConditionLimitSummary = {
    comparable: 0,
    meets: 0,
    below: 0,
    pending: 0,
    notMeasured: 0,
    invalid: 0,
    unavailable: 0,
    notApplicable: 0,
  };
  for (const item of items) {
    const result = evaluateSceneConditionTableLowerLimit(
      item.item_id,
      item.value,
      item.status,
    );
    if (result.state === "meets" || result.state === "below") {
      summary.comparable += 1;
    }
    if (result.state === "meets") summary.meets += 1;
    else if (result.state === "below") summary.below += 1;
    else if (result.state === "pending") summary.pending += 1;
    else if (result.state === "not_measured") summary.notMeasured += 1;
    else if (result.state === "invalid") summary.invalid += 1;
    else if (result.state === "unavailable") summary.unavailable += 1;
    else summary.notApplicable += 1;
  }
  return summary;
}

export function sceneConditionLimitUnresolvedCount(
  summary: SceneConditionLimitSummary,
): number {
  return (
    summary.pending
    + summary.notMeasured
    + summary.invalid
    + summary.unavailable
  );
}
