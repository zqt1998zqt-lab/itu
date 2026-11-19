import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { formatNV200Limit, nv200LimitRange, nv200LimitScale } from "../src/domain/nv200-limits.ts";
import {
  evaluateSceneConditionTableLowerLimit,
  sceneConditionLimitUnresolvedCount,
  sceneConditionTableLowerLimit,
  summarizeSceneConditionTableLimits,
} from "../src/features/scene-condition-limits.ts";
import type { SceneConditionItemId } from "../src/features/scene-condition-profiles.ts";

// Exercise the actual table export, including SWB's separate row layout.
const sheets = JSON.parse(
  readFileSync(new URL("../src/data/nv200-itut-structure.json", import.meta.url), "utf8"),
) as { sheet: string; rows: { row: string; values: Record<string, string> }[] }[];

function limits(sheet: string, row: number) {
  const source = sheets.find(value => value.sheet === sheet)!;
  const cells = source.rows.find(value => Number(value.row) === row)!.values;
  return { lowerLimit: cells[`E${row}`] ?? "", upperLimit: cells[`D${row}`] ?? "" };
}

test("limits retain numeric bounds from the workbook, including negatives and zero", () => {
  assert.equal(formatNV200Limit(limits("NB-wired", 2)), "[170 ~ 275]");
  assert.equal(formatNV200Limit(limits("NB-wired", 8)), "[-200 ~ -53]");
  assert.equal(formatNV200Limit(limits("WB-bt", 43)), "[0 ~ 120]");
  assert.equal(formatNV200Limit(limits("SWB-wired", 6)), "[3.6 ~ 5]");
});

test("merged curve requirements remain text and missing NB Bluetooth delay stays unspecified", () => {
  assert.equal(formatNV200Limit(limits("NB-wired", 7)), "符合频响曲线");
  assert.equal(formatNV200Limit(limits("NB-wired", 9)), "符合噪声曲线");
  assert.equal(formatNV200Limit(limits("NB-bt", 2)), "原表未填写");
});

test("one-sided bounds are explicit, and non-numeric requirements are preserved", () => {
  assert.equal(formatNV200Limit({ lowerLimit: "", upperLimit: "0" }), "≤ 0");
  assert.equal(formatNV200Limit({ lowerLimit: "-5", upperLimit: "" }), "≥ -5");
  assert.equal(formatNV200Limit({ lowerLimit: "按掩模", upperLimit: "按掩模" }), "按掩模");
  assert.equal(formatNV200Limit({ lowerLimit: "要求 A", upperLimit: "要求 B" }), "下限：要求 A；上限：要求 B");
});


test("MOS and percent intervals use meaningful common scales", () => {
  const mos = nv200LimitScale("MOS", [limits("NB-wired", 10)]);
  assert.deepEqual(mos, { min: 1, max: 5 });
  const range = nv200LimitRange(limits("NB-wired", 10), mos)!;
  assert.equal(range.startPercent, 50);
  assert.equal(range.endPercent, 100);
  const percent = nv200LimitScale("DT Class A1 [%]", [limits("WB-bt", 19)]);
  assert.deepEqual(percent, { min: 0, max: 100 });
  assert.equal(nv200LimitRange({ lowerLimit: "25", upperLimit: "100" }, percent)!.startPercent, 25);
});

test("negative intervals retain sign and share a unit scale", () => {
  const scale = nv200LimitScale("RLR [dB]", [
    { lowerLimit: "-2", upperLimit: "6" },
    { lowerLimit: "15", upperLimit: "45" },
  ])!;
  assert.deepEqual(scale, { min: -2, max: 45 });
  const range = nv200LimitRange({ lowerLimit: "-2", upperLimit: "6" }, scale)!;
  assert.equal(range.startPercent, 0);
  assert.ok(Math.abs(range.endPercent - 800 / 47) < 1e-10);
  assert.equal(range.lowerLabel, "-2");
});

test("text, missing and invalid bounds never create an invented interval", () => {
  const scale = { min: 1, max: 5 };
  assert.equal(nv200LimitRange(limits("NB-wired", 7), scale), null);
  assert.equal(nv200LimitRange(limits("NB-bt", 2), scale), null);
  assert.equal(nv200LimitRange({ lowerLimit: "5", upperLimit: "1" }, scale), null);
  assert.equal(nv200LimitRange({ lowerLimit: "1", upperLimit: "6" }, scale), null);
  assert.equal(nv200LimitRange({ lowerLimit: "1", upperLimit: "Infinity" }, scale), null);
  assert.equal(nv200LimitRange({ lowerLimit: "1", upperLimit: "5" }, { min: 1, max: 1 }), null);
  assert.deepEqual(nv200LimitScale("constant", [{ lowerLimit: "0", upperLimit: "0" }]), { min: -1, max: 1 });
});

test("scene-condition WB-wired G/N/S lower limits match the workbook", () => {
  const rows = [
    [24, 1], [25, 3], [26, 3],
    [29, 1], [30, 3], [31, 3],
    [34, 1], [35, 2.5], [36, 2.5],
    [39, 1], [40, 2.5], [41, 2.5],
  ] as const;
  for (const [row, expected] of rows) {
    const itemId = `WB-wired:B${row}` as SceneConditionItemId;
    assert.equal(sceneConditionTableLowerLimit(itemId), expected);
    assert.equal(
      Number(limits("WB-wired", row).lowerLimit),
      expected,
    );
  }
});

test("scene-condition lower limits match all eight workbook modes", () => {
  const modes = [
    "NB-wired",
    "NB-wireless",
    "NB-bt",
    "WB-wired",
    "WB-wireless",
    "WB-bt",
    "SWB-wired",
    "SWB-wireless",
  ] as const;
  const rows = [
    { row: 24, scene: 1, family: "g" },
    { row: 25, scene: 1, family: "n" },
    { row: 26, scene: 1, family: "s" },
    { row: 29, scene: 2, family: "g" },
    { row: 30, scene: 2, family: "n" },
    { row: 31, scene: 2, family: "s" },
    { row: 34, scene: 3, family: "g" },
    { row: 35, scene: 3, family: "n" },
    { row: 36, scene: 3, family: "s" },
    { row: 39, scene: 4, family: "g" },
    { row: 40, scene: 4, family: "n" },
    { row: 41, scene: 4, family: "s" },
  ] as const;
  for (const mode of modes) {
    for (const { row, scene, family } of rows) {
      const expected =
        family === "g"
          ? 1
          : scene >= 3
            ? 2.5
            : mode.startsWith("SWB-") && scene === 1
              ? 4
              : 3;
      const itemId = `${mode}:B${row}` as SceneConditionItemId;
      assert.equal(sceneConditionTableLowerLimit(itemId), expected);
      assert.equal(
        Number(limits(mode, row).lowerLimit),
        expected,
      );
    }
  }
});

test("scene-condition table-limit comparison is inclusive and scene-specific", () => {
  assert.equal(
    evaluateSceneConditionTableLowerLimit("WB-wired:B25", 3).state,
    "meets",
  );
  assert.equal(
    evaluateSceneConditionTableLowerLimit("WB-wired:B25", 2.9999).state,
    "below",
  );
  assert.equal(
    evaluateSceneConditionTableLowerLimit("WB-wired:B30", 2.8).state,
    "below",
  );
  assert.equal(
    evaluateSceneConditionTableLowerLimit("WB-wired:B35", 2.8).state,
    "meets",
  );
  assert.equal(
    evaluateSceneConditionTableLowerLimit("WB-wired:B35", 2.4999).state,
    "below",
  );
});

test("scene-condition table-limit comparison never turns missing data into pass", () => {
  for (const value of [null, undefined, Number.NaN, Infinity, -Infinity]) {
    const result = evaluateSceneConditionTableLowerLimit(
      "WB-wired:B25",
      value,
    );
    assert.equal(result.state, "unavailable");
    assert.equal(result.lowerLimit, 3);
  }
  assert.equal(
    evaluateSceneConditionTableLowerLimit("WB-wired:B27", 1).state,
    "not_applicable",
  );
  assert.equal(
    evaluateSceneConditionTableLowerLimit(
      "WB-wired:B25",
      null,
      "not_measured",
    ).state,
    "not_measured",
  );
  assert.equal(
    evaluateSceneConditionTableLowerLimit(
      "WB-wired:B25",
      null,
      "invalid",
    ).state,
    "invalid",
  );
  assert.equal(
    evaluateSceneConditionTableLowerLimit(
      "WB-wired:B25",
      null,
      "pending",
    ).state,
    "pending",
  );
});

test("scene-condition table-limit summary preserves below and unavailable counts", () => {
  assert.deepEqual(
    summarizeSceneConditionTableLimits([
      { item_id: "WB-wired:B24", value: 2.785522386622494 },
      { item_id: "WB-wired:B25", value: 3.17537717194136 },
      { item_id: "WB-wired:B26", value: 3.549548711093021 },
    ]),
    {
      comparable: 3,
      meets: 3,
      below: 0,
      pending: 0,
      notMeasured: 0,
      invalid: 0,
      unavailable: 0,
      notApplicable: 0,
    },
  );
  const mixed = summarizeSceneConditionTableLimits([
    {
      item_id: "WB-wired:B24",
      value: null,
      status: "pending",
    },
    {
      item_id: "WB-wired:B25",
      value: 2.9999,
      status: "review_required",
    },
    {
      item_id: "WB-wired:B26",
      value: null,
      status: "not_measured",
    },
    { item_id: "WB-wired:B27", value: 1, status: "invalid" },
  ]);
  assert.deepEqual(mixed, {
    comparable: 1,
    meets: 0,
    below: 1,
    pending: 1,
    notMeasured: 1,
    invalid: 0,
    unavailable: 0,
    notApplicable: 1,
  });
  assert.equal(sceneConditionLimitUnresolvedCount(mixed), 2);
});
