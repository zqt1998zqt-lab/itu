import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  engineeringDelta,
  fromSource,
  numeric,
  recordDelta,
  resultOf,
  resultsCsv,
  statsOf,
  tradeoffs,
  validateProject,
} from "../src/domain/model.ts";
import type { TestItem } from "../src/domain/model.ts";

const source = JSON.parse(
  readFileSync(new URL("../NV200_ITUT_原表结构.json", import.meta.url), "utf8"),
);
const project = fromSource(source);
const item = (id: string): TestItem =>
  structuredClone(project.items.find((row) => row.id === id)!);

test("真实原表完整映射为 8 个模式、336 项、17 类指标", () => {
  assert.equal(project.items.length, 336);
  assert.equal(new Set(project.items.map((row) => row.mode)).size, 8);
  assert.equal(new Set(project.items.map((row) => row.metricId)).size, 17);
  assert.deepEqual(statsOf(project.items, "G"), {
    total: 336,
    measured: 209,
    pass: 156,
    fail: 31,
    pending: 22,
    missing: 127,
    invalid: 0,
    coverage: 209 / 336,
    passRate: 156 / 187,
  });
});
test("复测独立统计，缺项不会从首次测量补齐", () => {
  const stats = statsOf(project.items, "H");
  assert.equal(stats.measured, 19);
  assert.equal(stats.missing, 317);
  assert.equal(resultOf(item("NB-wired:2"), "H").status, "missing");
  assert.equal(statsOf(project.items, "I").passRate, null);
});
test("人为超限负对照能够改变判定、余量、超限量", () => {
  const delay = item("NB-wired:2");
  assert.equal(resultOf(delay, "G").status, "fail");
  assert.ok(Math.abs(resultOf(delay, "G").excess! - 373.6) < 1e-8);
  delay.measurements.G.value = 200;
  assert.deepEqual(resultOf(delay, "G"), {
    status: "pass",
    value: 200,
    margin: 30,
    excess: 0,
  });
  const rlr = item("NB-wired:3");
  rlr.measurements.G.value = 99;
  assert.deepEqual(resultOf(rlr, "G"), {
    status: "fail",
    value: 99,
    margin: -93,
    excess: 93,
  });
});
test("空值不是零，真正的零参与判定，边界包含", () => {
  assert.equal(numeric(""), null);
  assert.equal(numeric("  "), null);
  assert.equal(numeric(0), 0);
  assert.equal(numeric(false), null);
  const delta = item("NB-wired:4");
  delta.measurements.G.value = 0;
  assert.equal(resultOf(delta, "G").status, "pass");
  delta.measurements.G.value = 0.5;
  assert.equal(resultOf(delta, "G").status, "pass");
  delta.measurements.G.value = null;
  assert.equal(resultOf(delta, "G").status, "missing");
});
test("曲线距离符号未确认，正负数都不自动宣告通过", () => {
  const curve = item("WB-wireless:7");
  assert.equal(resultOf(curve, "G").status, "pending");
  assert.equal(resultOf(curve, "H").status, "pending");
});
test("MOS与占比的物理有效性异常不能计入通过", () => {
  const mos = item("NB-wired:10");
  mos.measurements.G.value = 6;
  assert.equal(resultOf(mos, "G").status, "invalid");
  const dt = item("NB-wired:19");
  dt.measurements.G.value = -1;
  assert.equal(resultOf(dt, "G").status, "invalid");
  mos.measurements.G.value = Number.NaN;
  assert.equal(resultOf(mos, "G").status, "invalid");
});
test("识别 NMOS 改善与 SMOS 退化；条件未确认不输出工程改善量", () => {
  const rows = tradeoffs(project.items);
  assert.equal(rows.length, 3);
  assert.equal(rows[0].mode, "WB-wireless");
  assert.equal(rows[0].scenario, "Sce4");
  assert.ok(Math.abs(rows[0].noise - 0.5) < 1e-8);
  assert.ok(Math.abs(rows[0].speech + 1.7) < 1e-8);
  const speech = item("WB-wireless:41");
  assert.equal(engineeringDelta(speech), null);
  speech.comparable = true;
  assert.ok(Math.abs(engineeringDelta(speech)! + 1.7) < 1e-8);
});
test("接近上限5的MOS改善不因双侧表限余量变小被错误扣分", () => {
  const mos = item("NB-wired:10");
  mos.measurements.G.value = 4.3;
  mos.measurements.H.value = 4.8;
  mos.comparable = true;
  assert.ok(resultOf(mos, "H").margin! < resultOf(mos, "G").margin!);
  assert.ok(Math.abs(engineeringDelta(mos)! - 0.5) < 1e-8);
  assert.ok(Math.abs(recordDelta(mos)! - 0.5) < 1e-8);
});
test("备份往返保留所有数据，重复行或非法测量被拒绝", () => {
  assert.deepEqual(
    validateProject(JSON.parse(JSON.stringify(project))),
    project,
  );
  const duplicate = structuredClone(project);
  duplicate.items.push(duplicate.items[0]);
  assert.throws(() => validateProject(duplicate));
  const corrupt = structuredClone(project);
  corrupt.items[0].measurements.G.value = Infinity;
  assert.throws(() => validateProject(corrupt));
});
test("CSV保留零与负数，转义公式注入及引号", () => {
  const row = item("NB-wired:4");
  row.name = '=HYPERLINK("example")';
  row.measurements.G.value = 0;
  const csv = resultsCsv([row], "G");
  assert.ok(csv.startsWith("\ufeff"));
  assert.ok(csv.includes('"0"'));
  assert.ok(csv.includes("'=HYPERLINK"));
  assert.ok(csv.includes('"-0.5"'));
});
