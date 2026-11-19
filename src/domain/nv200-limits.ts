type NV200Limit = {
  lowerLimit: string;
  upperLimit: string;
};

// Preserve workbook values. A merged D:E curve requirement is text, while
// missing bounds remain unspecified rather than becoming zero or infinity.
export function formatNV200Limit(item: NV200Limit): string {
  const lower = item.lowerLimit.trim();
  const upper = item.upperLimit.trim();
  if (!lower && !upper) return "原表未填写";
  if (!lower) return Number.isFinite(Number(upper)) ? `≤ ${upper}` : upper;
  if (!upper) return Number.isFinite(Number(lower)) ? `≥ ${lower}` : lower;
  if (Number.isFinite(Number(lower)) && Number.isFinite(Number(upper))) {
    return `[${lower} ~ ${upper}]`;
  }
  return lower === upper ? lower : `下限：${lower}；上限：${upper}`;
}

export type NV200LimitScale = {
  min: number;
  max: number;
};

export type NV200LimitRange = {
  lower: number;
  upper: number;
  lowerLabel: string;
  upperLabel: string;
  scale: NV200LimitScale;
  startPercent: number;
  endPercent: number;
};

// MOS and percent scores have defined display domains. Other metrics share the
// full workbook bounds for their exact unit, so unrelated units never compete.
export function nv200LimitScale(
  unit: string,
  bounds: readonly { lowerLimit: string; upperLimit: string }[],
): NV200LimitScale | null {
  const values = bounds.flatMap(item => [item.lowerLimit, item.upperLimit])
    .map(value => value.trim())
    .filter(value => value !== "")
    .map(Number)
    .filter(Number.isFinite);
  if (!values.length) return null;
  let min = Math.min(...values);
  let max = Math.max(...values);
  if (/mos/i.test(unit)) {
    min = Math.min(1, min);
    max = Math.max(5, max);
  } else if (unit.includes("%")) {
    min = Math.min(0, min);
    max = Math.max(100, max);
  }
  if (min === max) return { min: min - 1, max: max + 1 };
  return { min, max };
}

export function nv200LimitRange(
  item: NV200Limit,
  scale: NV200LimitScale | null,
): NV200LimitRange | null {
  const lowerLabel = item.lowerLimit.trim();
  const upperLabel = item.upperLimit.trim();
  if (!lowerLabel || !upperLabel || !scale) return null;
  const lower = Number(lowerLabel);
  const upper = Number(upperLabel);
  if (![lower, upper, scale.min, scale.max].every(Number.isFinite)
      || upper < lower || scale.max <= scale.min
      || lower < scale.min || upper > scale.max) return null;
  const span = scale.max - scale.min;
  return {
    lower,
    upper,
    lowerLabel,
    upperLabel,
    scale,
    startPercent: 100 * (lower - scale.min) / span,
    endPercent: 100 * (upper - scale.min) / span,
  };
}
