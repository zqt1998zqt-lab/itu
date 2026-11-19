export const SCOPED_CASE_PAGE_SIZE = 10;

export type ScopedCasePageCursor = {
  scopeKey: string;
  pageIndex: number;
};

export type ScopedCasePage<T> = {
  items: readonly T[];
  scopeKey: string;
  pageIndex: number;
  pageNumber: number;
  pageCount: number;
  start: number;
  end: number;
  total: number;
  pageSize: number;
};

function normalizedPageSize(pageSize: number): number {
  return Number.isInteger(pageSize) && pageSize > 0
    ? pageSize
    : SCOPED_CASE_PAGE_SIZE;
}

function clampedPageIndex(pageIndex: number, pageCount: number): number {
  if (pageCount === 0 || !Number.isInteger(pageIndex)) return 0;
  return Math.min(Math.max(pageIndex, 0), pageCount - 1);
}

export function paginateScopedCases<T>(
  cases: readonly T[],
  scopeKey: string,
  cursor: ScopedCasePageCursor | null | undefined,
  pageSize = SCOPED_CASE_PAGE_SIZE,
): ScopedCasePage<T> {
  const size = normalizedPageSize(pageSize);
  const total = cases.length;
  const pageCount = total === 0 ? 0 : Math.ceil(total / size);
  const requestedPageIndex =
    cursor?.scopeKey === scopeKey ? cursor.pageIndex : 0;
  const pageIndex = clampedPageIndex(requestedPageIndex, pageCount);
  const startIndex = pageIndex * size;
  const endIndex = Math.min(total, startIndex + size);

  return {
    items: cases.slice(startIndex, endIndex),
    scopeKey,
    pageIndex,
    pageNumber: pageCount === 0 ? 0 : pageIndex + 1,
    pageCount,
    start: total === 0 ? 0 : startIndex + 1,
    end: endIndex,
    total,
    pageSize: size,
  };
}
