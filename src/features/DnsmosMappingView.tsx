import { useEffect, useMemo, useState } from "react";
import type { CSSProperties, KeyboardEvent } from "react";
import { AlertCircle, Network, Search } from "lucide-react";
import type {
  DnsmosMappingRow,
  DnsmosMappingSnapshot,
  DnsmosOutputId,
} from "./five-signal-flow-model";

export type ModelTrackId = "dnsmos" | "ts103281";

type ModelOutputMeta = {
  output: DnsmosOutputId;
  title: string;
  copy: string;
  contract: string;
  top: number;
  busLane: number;
};

type ModelTrack = {
  id: ModelTrackId;
  eyebrow: string;
  title: string;
  detail: string;
  status: string;
  statusTone: "engineering" | "experimental";
  top: number;
  outputs: ModelOutputMeta[];
};

export const OUTPUT_COLORS: Record<ModelTrackId, Record<DnsmosOutputId, string>> = {
  dnsmos: {
    sig: "#2563eb",
    bak: "#0f766e",
    ovrl: "#b45309",
  },
  ts103281: {
    sig: "#7c3aed",
    bak: "#c2410c",
    ovrl: "#be123c",
  },
};

export const MODEL_TRACKS: ModelTrack[] = [
  {
    id: "dnsmos",
    eyebrow: "NO-REFERENCE NEURAL MODEL",
    title: "DNSMOS P.835",
    detail: "16 kHz · 9.01 s window · SIG / BAK / OVRL",
    status: "工程代理",
    statusTone: "engineering",
    top: 170,
    outputs: [
      { output: "sig", title: "SIG", copy: "语音质量", contract: "S-MOS", top: 108, busLane: 350 },
      { output: "bak", title: "BAK", copy: "背景质量", contract: "N-MOS", top: 196, busLane: 364 },
      { output: "ovrl", title: "OVRL", copy: "整体质量", contract: "G-MOS", top: 284, busLane: 378 },
    ],
  },
  {
    id: "ts103281",
    eyebrow: "REFERENCE-BASED ETSI MODEL",
    title: "TS 103 281 Model A",
    detail: "reference + degraded · S-MOS / N-MOS / G-MOS",
    status: "实验值 · Annex C 0/48 · 未合规",
    statusTone: "experimental",
    top: 590,
    outputs: [
      { output: "sig", title: "S-MOS", copy: "语音质量", contract: "S-MOS", top: 108, busLane: 398 },
      { output: "bak", title: "N-MOS", copy: "背景质量", contract: "N-MOS", top: 196, busLane: 412 },
      { output: "ovrl", title: "G-MOS", copy: "整体质量", contract: "G-MOS", top: 284, busLane: 426 },
    ],
  },
];

const CANVAS_WIDTH = 940;
const CANVAS_HEIGHT = 1470;
const MODEL_X = 20;
const MODEL_WIDTH = 300;
const MODEL_HEIGHT = 390;
const CONTRACT_X = 540;
const CONTRACT_EDGE_X = CONTRACT_X - 1;
const CONTRACT_Y = 52;
const CONTRACT_WIDTH = 376;
const CONTRACT_HEIGHT = 1316;
const ROW_TOP = CONTRACT_Y + 52;
const ROW_HEIGHT = 28;
const ROW_STEP = 30;
const ROW_WIDTH = CONTRACT_WIDTH - 24;
const SOURCE_X = MODEL_X + MODEL_WIDTH;

type DnsmosFilter = "all" | "dnsmos" | "ts103281" | "unmapped";
type DnsmosFocus =
  | {
      kind: "output";
      track: ModelTrackId;
      output: DnsmosOutputId;
    }
  | { kind: "contract"; row: number }
  | null;

type BusinessGroup =
  | "latency"
  | "downlink"
  | "uplink"
  | "idle"
  | "city"
  | "highway"
  | "climate";

export function roundedOrthogonalPath(
  points: Array<readonly [number, number]>,
  radius = 8,
) {
  const compact = points.filter(
    (point, index) =>
      index === 0 ||
      point[0] !== points[index - 1]?.[0] ||
      point[1] !== points[index - 1]?.[1],
  );
  if (compact.length < 2) return "";
  if (compact.length === 2) {
    return `M ${compact[0][0]} ${compact[0][1]} L ${compact[1][0]} ${compact[1][1]}`;
  }

  let path = `M ${compact[0][0]} ${compact[0][1]}`;
  for (let index = 1; index < compact.length - 1; index += 1) {
    const previous = compact[index - 1];
    const current = compact[index];
    const next = compact[index + 1];
    const previousLength = Math.hypot(
      current[0] - previous[0],
      current[1] - previous[1],
    );
    const nextLength = Math.hypot(
      next[0] - current[0],
      next[1] - current[1],
    );
    const cornerRadius = Math.min(radius, previousLength / 2, nextLength / 2);
    if (cornerRadius === 0) {
      path += ` L ${current[0]} ${current[1]}`;
      continue;
    }
    const enterX =
      current[0] + ((previous[0] - current[0]) / previousLength) * cornerRadius;
    const enterY =
      current[1] + ((previous[1] - current[1]) / previousLength) * cornerRadius;
    const exitX =
      current[0] + ((next[0] - current[0]) / nextLength) * cornerRadius;
    const exitY =
      current[1] + ((next[1] - current[1]) / nextLength) * cornerRadius;
    path += ` L ${enterX} ${enterY} Q ${current[0]} ${current[1]} ${exitX} ${exitY}`;
  }
  const last = compact[compact.length - 1];
  return `${path} L ${last[0]} ${last[1]}`;
}

export function sourceRailPaths(
  sourceX: number,
  sourceY: number,
  lane: number,
  minY: number,
  maxY: number,
) {
  const approachEnd = sourceY <= minY ? maxY : minY;
  const main = roundedOrthogonalPath([
    [sourceX, sourceY],
    [lane, sourceY],
    [lane, approachEnd],
  ]);
  if (sourceY > minY && sourceY < maxY) {
    return [
      main,
      roundedOrthogonalPath([
        [lane, approachEnd === minY ? maxY : minY],
        [lane, sourceY],
      ]),
    ];
  }
  return [main];
}

function businessGroup(row: number): BusinessGroup {
  if (row === 2) return "latency";
  if (row <= 11) return "downlink";
  if (row <= 23) return "uplink";
  if (row <= 28) return "idle";
  if (row <= 33) return "city";
  if (row <= 38) return "highway";
  return "climate";
}

function isBusinessGroupStart(row: number) {
  return [2, 3, 12, 24, 29, 34, 39].includes(row);
}

function sameFocus(left: DnsmosFocus, right: DnsmosFocus) {
  if (left === null || right === null) return left === right;
  if (left.kind !== right.kind) return false;
  if (left.kind === "output" && right.kind === "output") {
    return left.track === right.track && left.output === right.output;
  }
  if (left.kind === "contract" && right.kind === "contract") {
    return left.row === right.row;
  }
  return false;
}

function rowCenter(row: number) {
  return ROW_TOP + (row - 2) * ROW_STEP + ROW_HEIGHT / 2;
}

function rowMatches(
  row: DnsmosMappingRow,
  filter: DnsmosFilter,
  query: string,
) {
  if (filter === "unmapped" && row.output !== null) return false;
  if ((filter === "dnsmos" || filter === "ts103281") && row.output === null) {
    return false;
  }
  const normalized = query.trim().toLowerCase();
  if (!normalized) return true;
  const output = row.output
    ? MODEL_TRACKS.map((track) => {
        const meta = track.outputs.find((item) => item.output === row.output);
        return `${track.title} ${meta?.title ?? ""} ${meta?.contract ?? ""}`;
      }).join(" ")
    : "不直接映射";
  return [
    `b${row.row}`,
    row.item.category,
    row.item.name,
    row.item.summary,
    row.item.unit,
    output,
  ]
    .join(" ")
    .toLowerCase()
    .includes(normalized);
}

function outputSourceY(track: ModelTrack, output: ModelOutputMeta) {
  return track.top + output.top + 38;
}

function DnsmosMappingCanvas({
  mapping,
  activeRows,
  filter,
}: {
  mapping: DnsmosMappingSnapshot;
  activeRows: Set<string>;
  filter: DnsmosFilter;
}) {
  const [hoverFocus, setHoverFocus] = useState<DnsmosFocus>(null);
  const [lockedFocus, setLockedFocus] = useState<DnsmosFocus>(null);
  const focusTarget = lockedFocus ?? hoverFocus;

  useEffect(() => {
    const clearMappingFocus = (event: globalThis.KeyboardEvent) => {
      if (event.key !== "Escape") return;
      setHoverFocus(null);
      setLockedFocus(null);
    };
    window.addEventListener("keydown", clearMappingFocus);
    return () => window.removeEventListener("keydown", clearMappingFocus);
  }, []);

  useEffect(() => {
    setHoverFocus(null);
    setLockedFocus(null);
  }, [activeRows]);

  const visibleTracks = useMemo(
    () =>
      filter === "unmapped"
        ? []
        : MODEL_TRACKS.filter(
            (track) => filter === "all" || track.id === filter,
          ),
    [filter],
  );
  const focusedContractRow =
    focusTarget?.kind === "contract"
      ? mapping.rows.find((row) => row.row === focusTarget.row) ?? null
      : null;

  const branches = useMemo(
    () =>
      visibleTracks.flatMap((track) =>
        track.outputs.flatMap((output) =>
          mapping.rows
            .filter(
              (row) =>
                row.output === output.output && activeRows.has(row.id),
            )
            .map((row) => ({
              row,
              track,
              output,
              lane: output.busLane,
              targetY: rowCenter(row.row) + (track.id === "dnsmos" ? -3 : 3),
              targetX: CONTRACT_EDGE_X,
            })),
        ),
      ),
    [activeRows, mapping.rows, visibleTracks],
  );

  const buses = useMemo(
    () =>
      visibleTracks.flatMap((track) =>
        track.outputs.flatMap((output) => {
          const outputRows = mapping.rows.filter(
            (row) =>
              row.output === output.output && activeRows.has(row.id),
          );
          if (!outputRows.length) return [];
          const sourceY = outputSourceY(track, output);
          const targetYs = outputRows.map((row) => rowCenter(row.row));
          return [
            {
              track,
              output,
              lane: output.busLane,
              sourceY,
              minY: Math.min(sourceY, ...targetYs),
              maxY: Math.max(sourceY, ...targetYs),
            },
          ];
        }),
      ),
    [activeRows, mapping.rows, visibleTracks],
  );

  function relationClass(related: boolean) {
    if (!focusTarget) return "";
    return related ? " is-focus-related" : " is-focus-muted";
  }

  function outputRelated(track: ModelTrackId, output: DnsmosOutputId) {
    if (!focusTarget) return true;
    if (focusTarget.kind === "output") {
      return (
        focusTarget.track === track && focusTarget.output === output
      );
    }
    return (
      focusedContractRow?.output === output &&
      track === "dnsmos"
        ? true
        : focusedContractRow?.output === output && track === "ts103281"
    );
  }

  function contractRelated(row: number) {
    if (!focusTarget) return true;
    if (focusTarget.kind === "contract") {
      const target = mapping.rows.find(
        (candidate) => candidate.row === focusTarget.row,
      );
      if (target?.output) {
        return (
          mapping.rows.find((candidate) => candidate.row === row)?.output ===
          target.output
        );
      }
      return focusTarget.row === row;
    }
    return (
      mapping.rows.find((candidate) => candidate.row === row)?.output ===
      focusTarget.output
    );
  }

  function interactionProps(target: NonNullable<DnsmosFocus>) {
    const locked = sameFocus(lockedFocus, target);
    return {
      tabIndex: 0,
      "aria-current": locked ? ("true" as const) : undefined,
      title: locked
        ? "再次点击或按 Enter 取消聚焦"
        : "点击锁定关联链路，按 Enter 也可切换",
      onMouseEnter: () => setHoverFocus(target),
      onMouseLeave: () =>
        setHoverFocus((current) =>
          sameFocus(current, target) ? null : current,
        ),
      onFocus: () => setHoverFocus(target),
      onBlur: () =>
        setHoverFocus((current) =>
          sameFocus(current, target) ? null : current,
        ),
      onClick: () => {
        setHoverFocus(null);
        setLockedFocus((current) =>
          sameFocus(current, target) ? null : target,
        );
      },
      onKeyDown: (event: KeyboardEvent<HTMLElement>) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          setHoverFocus(null);
          setLockedFocus((current) =>
            sameFocus(current, target) ? null : target,
          );
        }
      },
    };
  }

  const orderedTrackIds: ModelTrackId[] = ["ts103281", "dnsmos"];

  return (
    <div
      className="metric-mapping-canvas metric-dnsmos-canvas"
      style={{ width: CANVAS_WIDTH, height: CANVAS_HEIGHT }}
    >
      <svg
        className="metric-mapping-connectors"
        viewBox={`0 0 ${CANVAS_WIDTH} ${CANVAS_HEIGHT}`}
        aria-hidden="true"
        focusable="false"
      >
        {orderedTrackIds.map((trackId) => {
          const trackBuses = buses.filter(
            (bus) => bus.track.id === trackId,
          );
          const trackBranches = branches.filter(
            (branch) => branch.track.id === trackId,
          );
          return (
            <g key={`dual-track-layer-${trackId}`}>
              {trackBuses.map((bus) => (
                <g
                  key={`dual-bus-${trackId}-${bus.output.output}`}
                  className={`metric-mapping-link-layer${relationClass(
                    outputRelated(trackId, bus.output.output),
                  )}`}
                >
                  {sourceRailPaths(
                    SOURCE_X,
                    bus.sourceY,
                    bus.lane,
                    bus.minY,
                    bus.maxY,
                  ).map((path, index) => (
                    <path
                      key={`dual-bus-halo-${trackId}-${bus.output.output}-${index}`}
                      className={`metric-mapping-edge-halo is-model-trunk ${
                        trackId === "ts103281" ? "is-experimental" : ""
                      }`}
                      d={path}
                    />
                  ))}
                  {sourceRailPaths(
                    SOURCE_X,
                    bus.sourceY,
                    bus.lane,
                    bus.minY,
                    bus.maxY,
                  ).map((path, index) => (
                    <path
                      key={`dual-bus-line-${trackId}-${bus.output.output}-${index}`}
                      className={`metric-mapping-route-spine is-model-rail ${
                        trackId === "ts103281" ? "is-experimental" : ""
                      }`}
                      d={path}
                      style={{
                        stroke:
                          OUTPUT_COLORS[trackId][bus.output.output],
                      }}
                    />
                  ))}
                  <circle
                    className="metric-mapping-route-junction"
                    cx={SOURCE_X}
                    cy={bus.sourceY}
                    r="3"
                  />
                  <circle
                    className="metric-mapping-route-junction-core"
                    cx={SOURCE_X}
                    cy={bus.sourceY}
                    r="1.5"
                    fill={OUTPUT_COLORS[trackId][bus.output.output]}
                  />
                  <circle
                    className="metric-mapping-route-junction"
                    cx={bus.lane}
                    cy={bus.sourceY}
                    r="3"
                  />
                  <circle
                    className="metric-mapping-route-junction-core"
                    cx={bus.lane}
                    cy={bus.sourceY}
                    r="1.5"
                    fill={OUTPUT_COLORS[trackId][bus.output.output]}
                  />
                </g>
              ))}

              {trackBranches.map((branch) => (
                <g
                  key={`dual-branch-${trackId}-${branch.row.id}`}
                  className={`metric-mapping-link-layer${relationClass(
                    outputRelated(trackId, branch.output.output),
                  )}`}
                >
                  <path
                    className={`metric-mapping-edge-halo is-branch ${
                      trackId === "ts103281" ? "is-experimental" : ""
                    }`}
                    d={roundedOrthogonalPath([
                      [branch.lane, branch.targetY],
                      [branch.targetX, branch.targetY],
                    ])}
                  />
                  <path
                    className={`metric-mapping-edge ${
                      trackId === "ts103281" ? "is-experimental" : ""
                    }`}
                    d={roundedOrthogonalPath([
                      [branch.lane, branch.targetY],
                      [branch.targetX, branch.targetY],
                    ])}
                    style={{
                      stroke:
                        OUTPUT_COLORS[trackId][branch.output.output],
                    }}
                  />
                  {trackId === "dnsmos" ? (
                    <circle
                      className="metric-mapping-edge-port is-model"
                      cx={branch.targetX}
                      cy={branch.targetY}
                      r="3.2"
                      fill={
                        OUTPUT_COLORS[trackId][branch.output.output]
                      }
                    />
                  ) : (
                    <rect
                      className="metric-mapping-edge-port is-experimental-port"
                      x={branch.targetX - 2.6}
                      y={branch.targetY - 2.6}
                      width="5.2"
                      height="5.2"
                      transform={`rotate(45 ${branch.targetX} ${branch.targetY})`}
                      fill={
                        OUTPUT_COLORS[trackId][branch.output.output]
                      }
                    />
                  )}
                </g>
              ))}
            </g>
          );
        })}
      </svg>

      <div
        className="metric-mapping-column-label"
        style={{ top: 16, left: MODEL_X }}
      >
        双轨模型预测 · 2 路
      </div>
      <div
        className="metric-mapping-column-label"
        style={{ top: 16, left: CONTRACT_X }}
      >
        认证合同 · {mapping.mode} · 42 项
      </div>

      <aside
        className="metric-mapping-guide-summary"
        style={{
          top: 72,
          left: MODEL_X,
          width: MODEL_WIDTH,
        }}
      >
        <header>
          <span>DUAL-TRACK SCOPE</span>
          <strong>2 条模型轨道 · 12 项合同行</strong>
        </header>
        <div>
          <span>
            <i className="is-dnsmos" />
            DNSMOS P.835
          </span>
          <span>
            <i className="is-model-a" />
            TS 103 281 Model A
          </span>
          <b>{mapping.unmappedCount} 项不直接映射</b>
        </div>
      </aside>

      {MODEL_TRACKS.map((track) => (
        <article
          key={track.id}
          className={`metric-dnsmos-model is-${track.id} ${
            filter !== "all" &&
            filter !== "unmapped" &&
            filter !== track.id
              ? "is-filtered-out"
              : ""
          }`}
          style={
            {
              top: track.top,
              left: MODEL_X,
              width: MODEL_WIDTH,
              height: MODEL_HEIGHT,
              "--track-color": OUTPUT_COLORS[track.id].sig,
            } as CSSProperties
          }
        >
          <div className="metric-dnsmos-model-head">
            <div className="metric-dnsmos-model-head-top">
              <span>{track.eyebrow}</span>
              <em
                className={
                  track.statusTone === "experimental"
                    ? "is-experimental"
                    : "is-engineering"
                }
              >
                {track.status}
              </em>
            </div>
            <strong>{track.title}</strong>
            <p>{track.detail}</p>
          </div>
          <div
            className="metric-dnsmos-output-list"
            role="list"
            aria-label={`${track.title} 输出`}
          >
            {track.outputs.map((output) => (
              <article
                key={output.output}
                role="listitem"
                className={`metric-mapping-interactive metric-dnsmos-output ${
                  track.id === "ts103281" ? "is-experimental-track" : ""
                }${relationClass(
                  outputRelated(track.id, output.output),
                )}${
                  lockedFocus?.kind === "output" &&
                  lockedFocus.track === track.id &&
                  lockedFocus.output === output.output
                    ? " is-focus-locked"
                    : ""
                }`}
                style={
                  {
                    top: output.top,
                    "--mapping-color":
                      OUTPUT_COLORS[track.id][output.output],
                  } as CSSProperties
                }
                {...interactionProps({
                  kind: "output",
                  track: track.id,
                  output: output.output,
                })}
              >
                <span>{output.title}</span>
                <strong>{output.copy}</strong>
                <small>→ {output.contract} × 4 场景</small>
              </article>
            ))}
          </div>
        </article>
      ))}

      <div
        className="metric-dnsmos-contract-shell"
        style={{
          top: CONTRACT_Y,
          left: CONTRACT_X,
          width: CONTRACT_WIDTH,
          height: CONTRACT_HEIGHT,
        }}
      >
        <div className="metric-dnsmos-contract-head">
          <span>
            <strong>42 项认证合同</strong>
            <small>{mapping.mode}</small>
          </span>
          <em>
            <b>12 项 × 2 轨道</b>
            <small>{mapping.unmappedCount} 项不直接映射</small>
          </em>
        </div>
        <div
          className="metric-dnsmos-row-list"
          role="list"
          aria-label={`${mapping.mode} DNSMOS 与 42 项认证指标映射`}
        >
          {mapping.rows.map((row) => {
            const active = activeRows.has(row.id);
            const output = row.output;
            const dnsmosOutput = output
              ? MODEL_TRACKS[0].outputs.find(
                  (item) => item.output === output,
                )
              : null;
            const modelAOutput = output
              ? MODEL_TRACKS[1].outputs.find(
                  (item) => item.output === output,
                )
              : null;
            return (
              <article
                key={row.id}
                role="listitem"
                className={`metric-dnsmos-row metric-mapping-interactive ${
                  output ? "is-mapped" : "is-unmapped"
                } ${active ? "is-active" : "is-muted"}${
                  isBusinessGroupStart(row.row) ? " is-group-start" : ""
                }${relationClass(contractRelated(row.row))}${
                  lockedFocus?.kind === "contract" &&
                  lockedFocus.row === row.row
                    ? " is-focus-locked"
                    : ""
                }`}
                style={
                  {
                    top: (row.row - 2) * ROW_STEP,
                    left: 12,
                    width: ROW_WIDTH,
                    height: ROW_HEIGHT,
                    "--mapping-color": output ? "#475569" : "#94a3b8",
                  } as CSSProperties
                }
                data-business-group={businessGroup(row.row)}
                aria-label={`B${row.row}，${row.item.summary}，${
                  dnsmosOutput && modelAOutput
                    ? `DNSMOS P.835 ${dnsmosOutput.title} 与 TS 103 281 Model A ${modelAOutput.title} 双轨映射`
                    : "不直接映射"
                }`}
                {...interactionProps({ kind: "contract", row: row.row })}
              >
                <code>B{row.row}</code>
                <strong title={row.item.name}>{row.item.summary}</strong>
                {dnsmosOutput && modelAOutput ? (
                  <div className="metric-dnsmos-row-evidence">
                    <span
                      className="is-dnsmos"
                      style={
                        {
                          "--evidence-color":
                            OUTPUT_COLORS.dnsmos[dnsmosOutput.output],
                        } as CSSProperties
                      }
                    >
                      P.835 {dnsmosOutput.title}
                    </span>
                    <span
                      className="is-model-a"
                      style={
                        {
                          "--evidence-color":
                            OUTPUT_COLORS.ts103281[modelAOutput.output],
                        } as CSSProperties
                      }
                    >
                      281 {modelAOutput.title}
                    </span>
                  </div>
                ) : (
                  <span className="metric-dnsmos-row-unmapped">
                    不直接映射
                  </span>
                )}
              </article>
            );
          })}
        </div>
      </div>

      <div
        className="metric-mapping-legend metric-dnsmos-legend"
        style={{ top: 1388, left: 20 }}
      >
        <span className="metric-mapping-legend-item">
          <i
            aria-hidden="true"
            style={{ backgroundColor: OUTPUT_COLORS.dnsmos.sig }}
          />
          DNSMOS P.835 → S/N/G-MOS
        </span>
        <span className="metric-mapping-legend-item">
          <i
            aria-hidden="true"
            style={{ backgroundColor: OUTPUT_COLORS.ts103281.sig }}
          />
          TS 103 281 Model A → S/N/G-MOS
        </span>
        <span className="metric-mapping-legend-item">
          <i className="is-unmapped" aria-hidden="true" />
          30 项不直接映射
        </span>
      </div>
      <p
        className="metric-mapping-note metric-dnsmos-track-note"
        style={{ top: 1420, left: 20, width: 900 }}
      >
        双轨均只提供工程证据：DNSMOS P.835 是无参考的感知模型代理；
        TS 103 281 Model A 是参考音频驱动的实验模型，当前 Annex C 为 0/48，
        formal_compliance_claim_allowed=false。两条轨道都不改变 42 项合同的正式状态。
      </p>
    </div>
  );
}

export default function DnsmosMappingView({
  mapping,
}: {
  mapping: DnsmosMappingSnapshot;
}) {
  const [filter, setFilter] = useState<DnsmosFilter>("all");
  const [query, setQuery] = useState("");
  const activeRows = useMemo(
    () =>
      new Set(
        mapping.rows
          .filter((row) => rowMatches(row, filter, query))
          .map((row) => row.id),
      ),
    [filter, mapping.rows, query],
  );

  if (!mapping.valid) {
    return (
      <section className="metric-dnsmos-detail">
        <div className="metric-mapping-error" role="alert">
          <AlertCircle size={18} />
          {mapping.problem ?? "DNSMOS 映射数据不可用。"}
        </div>
      </section>
    );
  }

  return (
    <section className="metric-dnsmos-detail" aria-labelledby="metric-dnsmos-heading">
      <div className="metric-dnsmos-heading">
        <div>
          <div className="metric-mapping-eyebrow">
            <Network size={15} aria-hidden="true" />
            双轨感知模型映射
          </div>
          <h3 id="metric-dnsmos-heading">
            双轨模型预测 → 42 项认证指标
          </h3>
          <p>
            DNSMOS P.835 与 ETSI TS 103 281 Model A 分别形成独立轨道，
            共同对照四类噪声场景中的 S-MOS、N-MOS、G-MOS。
            其余 30 项保留为合同行；两条轨道都不产生正式合规结论。
          </p>
        </div>
        <div
          className="metric-mapping-summary metric-dnsmos-summary"
          aria-live="polite"
        >
          <strong>{mapping.mappedCount}</strong>
          <span>DNSMOS</span>
          <i aria-hidden="true" />
          <strong>{mapping.mappedCount}</strong>
          <span>Model A</span>
          <i aria-hidden="true" />
          <strong>{mapping.unmappedCount}</strong>
          <span>不直接映射</span>
        </div>
      </div>

      <div className="metric-mapping-table-toolbar metric-dnsmos-toolbar">
        <div
          className="metric-mapping-filter-group"
          role="group"
          aria-label="筛选双轨模型映射状态"
        >
          {(
            [
              ["all", "全部", mapping.rows.length],
              ["dnsmos", "DNSMOS", mapping.mappedCount],
              ["ts103281", "Model A", mapping.mappedCount],
              ["unmapped", "不直接映射", mapping.unmappedCount],
            ] as const
          ).map(([value, label, count]) => (
            <button
              key={value}
              type="button"
              className={filter === value ? "active" : ""}
              aria-pressed={filter === value}
              onClick={() => setFilter(value)}
            >
              {label}
              <span>{count}</span>
            </button>
          ))}
        </div>
        <label className="metric-mapping-search">
          <Search size={14} aria-hidden="true" />
          <span className="sr-only">搜索双轨模型映射合同行</span>
          <input
            type="search"
            value={query}
            placeholder="搜索合同行、场景或映射指标…"
            onChange={(event) => setQuery(event.target.value)}
          />
        </label>
        <span className="metric-mapping-table-count" aria-live="polite">
          当前高亮 {activeRows.size} / {mapping.rows.length} 项
        </span>
      </div>

      <p className="sr-only">
        当前 {mapping.mode} 的双轨模型映射中，{mapping.mappedCount} 项由
        DNSMOS P.835 的 SIG、BAK、OVRL 提供工程代理，另有
        {mapping.mappedCount} 项由 TS 103 281 Model A 的 S-MOS、N-MOS、
        G-MOS 提供实验预测；{mapping.unmappedCount} 项不直接映射。
        Model A 当前未通过官方 Annex C，两条轨道都不改变
        standard_compliance=false 的正式边界。
      </p>
      <div
        className="metric-mapping-scroll metric-dnsmos-scroll"
        role="region"
        aria-label={`${mapping.mode} DNSMOS P.835 与 TS 103 281 Model A 双轨映射到 42 项认证指标关系图`}
        tabIndex={0}
      >
        <DnsmosMappingCanvas
          mapping={mapping}
          activeRows={activeRows}
          filter={filter}
        />
      </div>
    </section>
  );
}
