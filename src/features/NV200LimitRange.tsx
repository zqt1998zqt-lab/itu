import type { CSSProperties } from "react";
import rawSheets from "../data/nv200-itut-structure.json";
import {
  formatNV200Limit,
  nv200LimitRange,
  nv200LimitScale,
} from "../domain/nv200-limits.ts";
import "./nv200-limit-range.css";

type LimitItem = {
  lowerLimit: string;
  upperLimit: string;
  unit?: string;
};

const unitBounds = new Map<string, LimitItem[]>();
for (const sheet of rawSheets as { rows: { row: string; values: Partial<Record<string, string>> }[] }[]) {
  for (const row of sheet.rows) {
    if (Number(row.row) < 2) continue;
    const cells = row.values;
    const unit = (cells[`C${row.row}`] ?? "").replace(/\n/g, " ").trim();
    const bounds = unitBounds.get(unit) ?? [];
    bounds.push({
      lowerLimit: cells[`E${row.row}`] ?? "",
      upperLimit: cells[`D${row.row}`] ?? "",
    });
    unitBounds.set(unit, bounds);
  }
}
const unitScales = new Map(
  [...unitBounds].map(([unit, bounds]) => [unit, nv200LimitScale(unit, bounds)]),
);

export default function NV200LimitRange({ item }: { item: LimitItem }) {
  const label = formatNV200Limit(item);
  const unit = item.unit?.trim() ?? "";
  const scale = unitScales.get(unit) ?? nv200LimitScale(unit, [item]);
  const range = nv200LimitRange(item, scale);
  if (!range) {
    const isEmpty = !item.lowerLimit.trim() && !item.upperLimit.trim();
    return (
      <span className={`nv200-limit-requirement${isEmpty ? " is-unspecified" : ""}`}>
        <span className="nv200-limit-requirement-mark" aria-hidden="true" />
        {label}
      </span>
    );
  }
  const style = {
    "--limit-start": `${range.startPercent}%`,
    "--limit-width": `${range.endPercent - range.startPercent}%`,
  } as CSSProperties;
  return (
    <span
      className="nv200-limit-range"
      role="img"
      aria-label={`允许区间：下限 ${range.lowerLabel}，上限 ${range.upperLabel}。显示刻度 ${range.scale.min} 至 ${range.scale.max}；单位：${unit || "未指定"}。`}
      title={`允许区间：${label}；显示刻度：${range.scale.min} 至 ${range.scale.max}`}
      data-limit-lower={range.lowerLabel}
      data-limit-upper={range.upperLabel}
      data-scale-min={range.scale.min}
      data-scale-max={range.scale.max}
      style={style}
    >
      <span className="nv200-limit-endpoints" aria-hidden="true">
        <span><small>下限</small><strong>{range.lowerLabel}</strong></span>
        <span><small>上限</small><strong>{range.upperLabel}</strong></span>
      </span>
      <span className="nv200-limit-track" aria-hidden="true">
        <span className="nv200-limit-band" />
      </span>
      <span className="nv200-limit-scale" aria-hidden="true">
        <span>{range.scale.min}</span><span>刻度</span><span>{range.scale.max}</span>
      </span>
    </span>
  );
}
