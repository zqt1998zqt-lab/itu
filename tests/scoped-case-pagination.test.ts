import assert from "node:assert/strict";
import { test } from "node:test";

import {
  paginateScopedCases,
  SCOPED_CASE_PAGE_SIZE,
} from "../src/features/scoped-case-pagination.ts";

const SCOPE = "dataset:all";
const CASES = Array.from({ length: 100 }, (_, index) => ({
  caseId: `case-${index + 1}`,
}));

function caseIds(items: readonly { caseId: string }[]): string[] {
  return items.map((item) => item.caseId);
}

test("empty case lists expose no page and no range", () => {
  const page = paginateScopedCases([], SCOPE, null);

  assert.deepEqual(page.items, []);
  assert.equal(page.pageNumber, 0);
  assert.equal(page.pageCount, 0);
  assert.equal(page.start, 0);
  assert.equal(page.end, 0);
  assert.equal(page.total, 0);
});

test("the default view contains ten of one hundred cases", () => {
  const page = paginateScopedCases(CASES, SCOPE, null);

  assert.deepEqual(caseIds(page.items), [
    "case-1",
    "case-2",
    "case-3",
    "case-4",
    "case-5",
    "case-6",
    "case-7",
    "case-8",
    "case-9",
    "case-10",
  ]);
  assert.equal(page.pageSize, SCOPED_CASE_PAGE_SIZE);
  assert.equal(page.pageNumber, 1);
  assert.equal(page.pageCount, 10);
  assert.equal(page.start, 1);
  assert.equal(page.end, 10);
  assert.equal(page.total, 100);
});

test("the final page is clipped to the remaining cases", () => {
  const page = paginateScopedCases(CASES, SCOPE, {
    scopeKey: SCOPE,
    pageIndex: 9,
  });

  assert.deepEqual(caseIds(page.items), [
    "case-91",
    "case-92",
    "case-93",
    "case-94",
    "case-95",
    "case-96",
    "case-97",
    "case-98",
    "case-99",
    "case-100",
  ]);
  assert.equal(page.pageNumber, 10);
  assert.equal(page.start, 91);
  assert.equal(page.end, 100);
});

test("negative and oversized page indexes clamp to valid pages", () => {
  const first = paginateScopedCases(CASES, SCOPE, {
    scopeKey: SCOPE,
    pageIndex: -4,
  });
  const last = paginateScopedCases(CASES, SCOPE, {
    scopeKey: SCOPE,
    pageIndex: 99,
  });

  assert.equal(first.pageIndex, 0);
  assert.equal(first.start, 1);
  assert.equal(last.pageIndex, 9);
  assert.equal(last.start, 91);
  assert.equal(last.end, 100);
});

test("invalid page sizes fall back to the ten-case product default", () => {
  for (const pageSize of [0, -10, 1.5, Number.NaN]) {
    const page = paginateScopedCases(CASES, SCOPE, null, pageSize);

    assert.equal(page.pageSize, SCOPED_CASE_PAGE_SIZE);
    assert.equal(page.items.length, 10);
    assert.equal(page.pageCount, 10);
  }
});

test("a new filter scope starts on page one before an effect can run", () => {
  const filtered = CASES.slice(0, 27);
  const page = paginateScopedCases(filtered, "dataset:double_talk", {
    scopeKey: "dataset:all",
    pageIndex: 2,
  });

  assert.equal(page.pageIndex, 0);
  assert.equal(page.pageNumber, 1);
  assert.equal(page.start, 1);
  assert.equal(page.end, 10);
  assert.equal(page.total, 27);
  assert.deepEqual(caseIds(page.items), caseIds(filtered.slice(0, 10)));
});

test("matching scopes preserve a valid page without mutating the full list", () => {
  const filtered = CASES.slice(0, 27);
  const original = filtered.slice();
  const page = paginateScopedCases(filtered, "dataset:double_talk", {
    scopeKey: "dataset:double_talk",
    pageIndex: 2,
  });

  assert.equal(page.pageNumber, 3);
  assert.equal(page.pageCount, 3);
  assert.equal(page.start, 21);
  assert.equal(page.end, 27);
  assert.deepEqual(caseIds(page.items), [
    "case-21",
    "case-22",
    "case-23",
    "case-24",
    "case-25",
    "case-26",
    "case-27",
  ]);
  assert.deepEqual(filtered, original);
});
