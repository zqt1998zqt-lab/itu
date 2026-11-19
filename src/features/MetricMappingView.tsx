import { useEffect, useMemo, useState } from "react";
import type { CSSProperties, KeyboardEvent } from "react";
import { AlertCircle, Headphones, Network, Search } from "lucide-react";
import type { NV200SheetData } from "../domain/nv200-table";
import NV200LimitRange from "./NV200LimitRange.tsx";
import ModeSheetBar from "./ModeSheetBar";
import DnsmosMappingView, {
  MODEL_TRACKS,
  OUTPUT_COLORS,
  roundedOrthogonalPath,
  sourceRailPaths,
} from "./DnsmosMappingView";
import {
  buildDnsmosMapping,
  buildMetricMapping,
  buildReceiveProxyMapping,
  METRIC_SOURCE_LABELS,
  METRIC_SOURCE_IDS,
  RECEIVE_PROXY_IDS,
  RECEIVE_PROXY_LABELS,
  type DnsmosMappingSnapshot,
  type DnsmosOutputId,
  type MetricMappingRow,
  type MetricSourceId,
  type ReceiveProxyId,
  type ReceiveProxyMappingSnapshot,
} from "./five-signal-flow-model";

const SOURCE_COLORS: Record<MetricSourceId, string> = {
  alignment: "#0284c7",
  sisdr: "#94a3b8",
  stoi: "#2563eb",
  level: "#0f766e",
  spectrum: "#7c3aed",
  doubleTalk: "#b45309",
  echo: "#be123c",
};
const RECEIVE_PROXY_COLORS: Record<ReceiveProxyId, string> = {
  binauralLevel: "#0f766e",
  frequencyResponse: "#7c3aed",
  idleNoise: "#0369a1",
  noisePeaks: "#b45309",
  outOfBand: "#be123c",
  rlrLinearity: "#0e7490",
  binauralP835: "#4f46e5",
};

const MODEL_COLUMN_X = 48;
const MODEL_COLUMN_WIDTH = 340;
const MODEL_CARD_TOP = 88;
const MODEL_CARD_HEIGHT = 402;
const DNSMOS_TRACK = MODEL_TRACKS.find((track) => track.id === "dnsmos")!;
const DNSMOS_LANES: Record<DnsmosOutputId, number> = {
  sig: 776,
  bak: 790,
  ovrl: 804,
};
const RECEIVE_LANES: Record<ReceiveProxyId, number> = {
  binauralLevel: 744,
  frequencyResponse: 750,
  idleNoise: 756,
  noisePeaks: 762,
  outOfBand: 768,
  rlrLinearity: 732,
  binauralP835: 738,
};
const ENGINEERING_LANE_XS = [842, 868, 894, 920, 946] as const;

const SOURCE_CARDS: Array<{
  id: Exclude<MetricSourceId, "alignment">;
  code: string;
  title: string;
  copy: string;
  top: number;
}> = [
  {
    id: "stoi",
    code: "INTELLIGIBILITY",
    title: "STOI",
    copy: "输入 / 输出 / 变化量",
    top: 184,
  },
  {
    id: "level",
    code: "LEVEL / SNR",
    title: "电平与 SNR",
    copy: "发送增益 · 底噪 · 输出 SNR",
    top: 444,
  },
  {
    id: "spectrum",
    code: "SPECTRUM",
    title: "频响与频谱",
    copy: "频响 · 谱峰 · 带外 · CNG 频谱",
    top: 704,
  },
  {
    id: "doubleTalk",
    code: "DOUBLE TALK",
    title: "双讲透传",
    copy: "双讲区段输出 / 输入",
    top: 964,
  },
  {
    id: "echo",
    code: "ECHO",
    title: "回声损耗",
    copy: "工程耦合损耗 · 回声降低",
    top: 1224,
  },
];

const LEFT_WIDTH = 320;
const LEFT_HEIGHT = 128;
const SOURCE_X = 418;
const SOURCE_EDGE_X = SOURCE_X + LEFT_WIDTH;
const RIGHT_X = 1084;
const RIGHT_WIDTH = 448;
const DNSMOS_TARGET_Y_OFFSET = -7;
const SIG_BUS_REROUTE_GAP = 14;
const SIG_BUS_REROUTE_Y = 166;
const ROW_TOP = 88;
const ROW_HEIGHT = 34;
const ROW_STEP = 40;
const CANVAS_WIDTH = 1580;
const CANVAS_HEIGHT = 1860;
const RECEIVE_CARD_TOP = 512;
const RECEIVE_CARD_HEIGHT = 404;
const RECEIVE_OUTPUT_TOP = 78;
const RECEIVE_OUTPUT_HEIGHT = 36;
const RECEIVE_OUTPUT_STEP = 40;
const GUIDE_CARD_TOP = 892;
const GUIDE_CARD_HEIGHT = 860;

type MappingFocus =
  | { kind: "source"; source: MetricSourceId }
  | { kind: "dnsmos"; output: DnsmosOutputId }
  | { kind: "receive"; proxy: ReceiveProxyId }
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

function dnsmosBusPaths(
  output: DnsmosOutputId,
  sourceX: number,
  sourceY: number,
  lane: number,
  minY: number,
  maxY: number,
) {
  if (output === "sig") {
    return [
      roundedOrthogonalPath([
        [sourceX, sourceY],
        [sourceX + SIG_BUS_REROUTE_GAP, sourceY],
        [sourceX + SIG_BUS_REROUTE_GAP, SIG_BUS_REROUTE_Y],
        [lane, SIG_BUS_REROUTE_Y],
        [lane, maxY],
      ]),
    ];
  }
  return sourceRailPaths(sourceX, sourceY, lane, minY, maxY);
}

function rowTop(index: number) {
  return ROW_TOP + index * ROW_STEP;
}

function rowCenter(index: number) {
  return rowTop(index) + ROW_HEIGHT / 2;
}

function sameFocus(left: MappingFocus, right: MappingFocus) {
  if (left === null || right === null) return left === right;
  if (left.kind !== right.kind) return false;
  if (left.kind === "source" && right.kind === "source") {
    return left.source === right.source;
  }
  if (left.kind === "dnsmos" && right.kind === "dnsmos") {
    return left.output === right.output;
  }
  if (left.kind === "receive" && right.kind === "receive") {
    return left.proxy === right.proxy;
  }
  if (left.kind === "contract" && right.kind === "contract") {
    return left.row === right.row;
  }
  return false;
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

function businessGroupLabel(row: number) {
  const group = businessGroup(row);
  if (group === "latency") return "时延基线";
  if (group === "downlink") return "下行接收";
  if (group === "uplink") return "上行发送";
  if (group === "idle") return "怠速静止";
  if (group === "city") return "60km/h 市区";
  if (group === "highway") return "120km/h 高速";
  return "空调大风档";
}

function isBusinessGroupStart(row: number) {
  return [2, 3, 12, 24, 29, 34, 39].includes(row);
}

function sourceColor(source: MetricSourceId | null) {
  return SOURCE_COLORS[source ?? "sisdr"];
}

function statusLabel(
  status: "engineering_evidence" | "alignment_proxy" | "not_measured",
) {
  if (status === "alignment_proxy") return "对齐代理";
  if (status === "engineering_evidence") return "工程证据";
  return "不可测";
}

function mappingNote(row: MetricMappingRow) {
  if (row.status === "alignment_proxy") {
    return "对齐时间基线代理，不等同正式端到端 RTD。";
  }
  if (row.status === "engineering_evidence") {
    return "来自当前五路 PCM 的工程代理，不代表合同通过。";
  }
  return "当前五路输入不足以直接计算，需要校准声学或正式测量方法。";
}

function MetricMappingCanvas({
  mapping,
  dnsmosMapping,
  receiveMapping,
}: {
  mapping: ReturnType<typeof buildMetricMapping>;
  dnsmosMapping: DnsmosMappingSnapshot | null;
  receiveMapping: ReceiveProxyMappingSnapshot | null;
}) {
  const [hoverFocus, setHoverFocus] = useState<MappingFocus>(null);
  const [lockedFocus, setLockedFocus] = useState<MappingFocus>(null);
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

  const targetRows = useMemo(
    () =>
      mapping.rows.map((row, index) => {
        const dnsmosRow = dnsmosMapping?.rows.find(
          (candidate) => candidate.row === row.row,
        );
        const dnsmosOutput = dnsmosRow?.output ?? null;
        const receiveProxy =
          receiveMapping?.rows.find((candidate) => candidate.row === row.row)
            ?.proxy ?? null;
        return {
          ...row,
          index,
          top: rowTop(index),
          center: rowCenter(index),
          dnsmosOutput,
          receiveProxy,
          dnsmosMeta: dnsmosOutput
            ? DNSMOS_TRACK.outputs.find(
                (output) => output.output === dnsmosOutput,
              ) ?? null
            : null,
        };
      }),
    [dnsmosMapping?.rows, mapping.rows, receiveMapping?.rows],
  );

  const engineeringLanes = useMemo(() => {
    const ranked = SOURCE_CARDS.filter(
      (card) => targetRows.some((row) => row.source === card.id),
    )
      .map((card) => {
        const centers = targetRows
          .filter((row) => row.source === card.id)
          .map((row) => row.center);
        return {
          source: card.id,
          centroid:
            centers.reduce((sum, center) => sum + center, 0) /
            centers.length,
        };
      })
      .sort((left, right) => left.centroid - right.centroid);

    return Object.fromEntries(
      ranked.map((entry, index) => [
        entry.source,
        ENGINEERING_LANE_XS[index],
      ]),
    ) as Partial<Record<MetricSourceId, number>>;
  }, [targetRows]);

  const dnsmosBranches = useMemo(
    () =>
      targetRows
        .filter((row) => row.dnsmosOutput !== null && row.dnsmosMeta)
        .map((row) => ({
          row,
          output: row.dnsmosOutput as DnsmosOutputId,
          targetY: row.center + DNSMOS_TARGET_Y_OFFSET,
        })),
    [targetRows],
  );

  const dnsmosBuses = useMemo(
    () =>
      DNSMOS_TRACK.outputs.flatMap((output) => {
        const rows = targetRows.filter(
          (row) => row.dnsmosOutput === output.output,
        );
        if (!rows.length) return [];
        const sourceY = MODEL_CARD_TOP + output.top + 38;
        const targetYs = rows.map(
          (row) => row.center + DNSMOS_TARGET_Y_OFFSET,
        );
        return [
          {
            output: output.output,
            lane: DNSMOS_LANES[output.output],
            sourceY,
            minY: Math.min(sourceY, ...targetYs),
            maxY: Math.max(sourceY, ...targetYs),
          },
        ];
      }),
    [targetRows],
  );

  const receiveBranches = useMemo(
    () =>
      RECEIVE_PROXY_IDS.flatMap((proxy) =>
        targetRows
          .filter((row) => row.receiveProxy === proxy)
          .map((row) => ({
            proxy,
            lane: RECEIVE_LANES[proxy],
            targetY: row.center,
            row,
          })),
      ),
    [targetRows],
  );

  const receiveBuses = useMemo(
    () =>
      RECEIVE_PROXY_IDS.flatMap((proxy, index) => {
        const rows = targetRows.filter((row) => row.receiveProxy === proxy);
        if (!rows.length) return [];
        const sourceY =
          RECEIVE_CARD_TOP +
          RECEIVE_OUTPUT_TOP +
          index * RECEIVE_OUTPUT_STEP +
          RECEIVE_OUTPUT_HEIGHT / 2;
        const targetYs = rows.map((row) => row.center);
        return [
          {
            proxy,
            lane: RECEIVE_LANES[proxy],
            sourceY,
            minY: Math.min(sourceY, ...targetYs),
            maxY: Math.max(sourceY, ...targetYs),
          },
        ];
      }),
    [targetRows],
  );

  const dnsmosMappedCount = targetRows.filter(
    (row) => row.dnsmosOutput !== null,
  ).length;
  const receiveMappedCount = targetRows.filter(
    (row) => row.receiveProxy !== null,
  ).length;
  const mergedCoveredCount = targetRows.filter(
    (row) =>
      row.source !== null ||
      row.dnsmosOutput !== null ||
      row.receiveProxy !== null,
  ).length;

  const engineeringBranches = useMemo(
    () =>
      SOURCE_CARDS.flatMap((card) => {
        const lane = engineeringLanes[card.id];
        if (lane === undefined) return [];
        return targetRows
          .filter((row) => row.source === card.id)
          .map((row) => ({
            source: card.id,
            lane,
            targetY: row.center,
          }));
      }),
    [engineeringLanes, targetRows],
  );

  const engineeringBuses = useMemo(
    () =>
      SOURCE_CARDS.flatMap((card) => {
        const lane = engineeringLanes[card.id];
        if (lane === undefined) return [];
        const sourceY = card.top + LEFT_HEIGHT / 2;
        const rows = targetRows.filter((row) => row.source === card.id);
        if (!rows.length) return [];
        const targetYs = rows.map((row) => row.center);
        return [
          {
            source: card.id,
            lane,
            sourceY,
            minY: Math.min(sourceY, ...targetYs),
            maxY: Math.max(sourceY, ...targetYs),
          },
        ];
      }),
    [engineeringLanes, targetRows],
  );

  const alignmentCenter = rowCenter(0);
  const focusedContractRow =
    focusTarget?.kind === "contract"
      ? targetRows.find((row) => row.row === focusTarget.row) ?? null
      : null;

  function relationClass(related: boolean) {
    if (!focusTarget) return "";
    return related ? " is-focus-related" : " is-focus-muted";
  }

  function sourceRelated(source: MetricSourceId) {
    if (!focusTarget) return true;
    if (focusTarget.kind === "source") {
      return focusTarget.source === source;
    }
    if (focusTarget.kind === "contract") {
      return focusedContractRow?.source === source;
    }
    return false;
  }

  function dnsmosRelated(output: DnsmosOutputId) {
    if (!focusTarget) return true;
    if (focusTarget.kind === "dnsmos") {
      return focusTarget.output === output;
    }
    if (focusTarget.kind === "contract") {
      return focusedContractRow?.dnsmosOutput === output;
    }
    return false;
  }

  function receiveRelated(proxy: ReceiveProxyId) {
    if (!focusTarget) return true;
    if (focusTarget.kind === "receive") {
      return focusTarget.proxy === proxy;
    }
    if (focusTarget.kind === "contract") {
      return focusedContractRow?.receiveProxy === proxy;
    }
    return false;
  }

  function contractRelated(row: number) {
    if (!focusTarget) return true;
    if (focusTarget.kind === "contract") {
      const target = targetRows.find(
        (candidate) => candidate.row === focusTarget.row,
      );
      if (target?.source) {
        return (
          targetRows.find((candidate) => candidate.row === row)?.source ===
          target.source
        );
      }
      if (target?.dnsmosOutput) {
        return (
          targetRows.find((candidate) => candidate.row === row)
            ?.dnsmosOutput === target.dnsmosOutput
        );
      }
      if (target?.receiveProxy) {
        return (
          targetRows.find((candidate) => candidate.row === row)
            ?.receiveProxy === target.receiveProxy
        );
      }
      return focusTarget.row === row;
    }
    const target = targetRows.find((candidate) => candidate.row === row);
    if (focusTarget.kind === "source") {
      return target?.source === focusTarget.source;
    }
    if (focusTarget.kind === "dnsmos") {
      return target?.dnsmosOutput === focusTarget.output;
    }
    return target?.receiveProxy === focusTarget.proxy;
  }

  function interactionProps(target: NonNullable<MappingFocus>) {
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

  return (
    <div
      className="metric-mapping-canvas metric-mapping-main-canvas"
      style={
        {
          width: CANVAS_WIDTH,
          height: CANVAS_HEIGHT,
          "--mapping-alignment": SOURCE_COLORS.alignment,
        } as CSSProperties
      }
    >
      <svg
        className="metric-mapping-connectors"
        viewBox={`0 0 ${CANVAS_WIDTH} ${CANVAS_HEIGHT}`}
        aria-hidden="true"
        focusable="false"
      >
        {engineeringBuses.map((bus) => (
          <g
            key={`engineering-bus-${bus.source}`}
            className={`metric-mapping-link-layer${relationClass(
              sourceRelated(bus.source),
            )}`}
          >
            {sourceRailPaths(
              SOURCE_EDGE_X,
              bus.sourceY,
              bus.lane,
              bus.minY,
              bus.maxY,
            ).map((path, index) => (
              <path
                key={`engineering-bus-halo-${bus.source}-${index}`}
                className="metric-mapping-edge-halo is-trunk"
                d={path}
              />
            ))}
            {sourceRailPaths(
              SOURCE_EDGE_X,
              bus.sourceY,
              bus.lane,
              bus.minY,
              bus.maxY,
            ).map((path, index) => (
              <path
                key={`engineering-bus-line-${bus.source}-${index}`}
                className="metric-mapping-route-spine is-engineering-rail"
                d={path}
                style={{ stroke: SOURCE_COLORS[bus.source] }}
              />
            ))}
            <circle
              className="metric-mapping-route-junction"
              cx={SOURCE_EDGE_X}
              cy={bus.sourceY}
              r="3"
            />
            <circle
              className="metric-mapping-route-junction-core"
              cx={SOURCE_EDGE_X}
              cy={bus.sourceY}
              r="1.5"
              fill={SOURCE_COLORS[bus.source]}
            />
          </g>
        ))}

        {engineeringBranches.map((branch) => (
          <g
            key={`engineering-branch-${branch.source}-${branch.targetY}`}
            className={`metric-mapping-link-layer${relationClass(
              sourceRelated(branch.source),
            )}`}
          >
            <path
              className="metric-mapping-edge-halo is-branch"
              d={roundedOrthogonalPath([
                [branch.lane, branch.targetY],
                [RIGHT_X, branch.targetY],
              ])}
            />
            <path
              className="metric-mapping-edge"
              d={roundedOrthogonalPath([
                [branch.lane, branch.targetY],
                [RIGHT_X, branch.targetY],
              ])}
              style={{ stroke: SOURCE_COLORS[branch.source] }}
            />
            <circle
              className="metric-mapping-edge-port is-engineering"
              cx={RIGHT_X}
              cy={branch.targetY}
              r="2.7"
              fill={SOURCE_COLORS[branch.source]}
            />
          </g>
        ))}

        {dnsmosBuses.map((bus) => {
          const paths = dnsmosBusPaths(
            bus.output,
            MODEL_COLUMN_X + MODEL_COLUMN_WIDTH,
            bus.sourceY,
            bus.lane,
            bus.minY,
            bus.maxY,
          );
          return (
            <g
              key={`dnsmos-bus-${bus.output}`}
              className={`metric-mapping-link-layer${relationClass(
                dnsmosRelated(bus.output),
              )}`}
            >
              {paths.map((path, index) => (
                <path
                  key={`dnsmos-bus-halo-${bus.output}-${index}`}
                  className="metric-mapping-edge-halo is-model-trunk"
                  d={path}
                />
              ))}
              {paths.map((path, index) => (
                <path
                  key={`dnsmos-bus-line-${bus.output}-${index}`}
                  className="metric-mapping-bus is-model-rail"
                  d={path}
                  style={{ stroke: OUTPUT_COLORS.dnsmos[bus.output] }}
                />
              ))}
              <circle
                className="metric-mapping-source-port is-model"
                cx={MODEL_COLUMN_X + MODEL_COLUMN_WIDTH}
                cy={bus.sourceY}
                r="3.4"
                fill={OUTPUT_COLORS.dnsmos[bus.output]}
              />
              <circle
                className="metric-mapping-route-junction"
                cx={bus.lane}
                cy={bus.output === "sig" ? SIG_BUS_REROUTE_Y : bus.sourceY}
                r="3.2"
              />
              <circle
                className="metric-mapping-route-junction-core"
                cx={bus.lane}
                cy={bus.output === "sig" ? SIG_BUS_REROUTE_Y : bus.sourceY}
                r="1.5"
                fill={OUTPUT_COLORS.dnsmos[bus.output]}
              />
            </g>
          );
        })}

        {dnsmosBranches.map((branch) => (
          <g
            key={`dnsmos-target-${branch.row.row}`}
            className={`metric-mapping-link-layer${relationClass(
              dnsmosRelated(branch.output),
            )}`}
          >
            <path
              className="metric-mapping-edge-halo is-model-branch"
              d={roundedOrthogonalPath([
                [DNSMOS_LANES[branch.output], branch.targetY],
                [RIGHT_X, branch.targetY],
              ])}
            />
            <path
              className="metric-mapping-edge metric-dnsmos-model-edge"
              d={roundedOrthogonalPath([
                [DNSMOS_LANES[branch.output], branch.targetY],
                [RIGHT_X, branch.targetY],
              ])}
              style={{ stroke: OUTPUT_COLORS.dnsmos[branch.output] }}
            />
            <circle
              className="metric-mapping-edge-port is-model"
              cx={RIGHT_X}
              cy={branch.targetY}
              r="3.2"
              fill={OUTPUT_COLORS.dnsmos[branch.output]}
            />
          </g>
        ))}

        {receiveBuses.map((bus) => {
          const paths = sourceRailPaths(
            MODEL_COLUMN_X + MODEL_COLUMN_WIDTH,
            bus.sourceY,
            bus.lane,
            bus.minY,
            bus.maxY,
          );
          return (
            <g
              key={`receive-bus-${bus.proxy}`}
              className={`metric-mapping-link-layer${relationClass(
                receiveRelated(bus.proxy),
              )}`}
            >
              {paths.map((path, index) => (
                <path
                  key={`receive-bus-halo-${bus.proxy}-${index}`}
                  className="metric-mapping-edge-halo is-model-trunk"
                  d={path}
                />
              ))}
              {paths.map((path, index) => (
                <path
                  key={`receive-bus-line-${bus.proxy}-${index}`}
                  className="metric-mapping-bus is-receive-model-rail"
                  d={path}
                  style={{ stroke: RECEIVE_PROXY_COLORS[bus.proxy] }}
                />
              ))}
              <circle
                className="metric-mapping-source-port is-model"
                cx={MODEL_COLUMN_X + MODEL_COLUMN_WIDTH}
                cy={bus.sourceY}
                r="3.2"
                fill={RECEIVE_PROXY_COLORS[bus.proxy]}
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
                fill={RECEIVE_PROXY_COLORS[bus.proxy]}
              />
            </g>
          );
        })}

        {receiveBranches.map((branch) => (
          <g
            key={`receive-target-${branch.proxy}-${branch.row.row}`}
            className={`metric-mapping-link-layer${relationClass(
              receiveRelated(branch.proxy),
            )}`}
          >
            <path
              className="metric-mapping-edge-halo is-model-branch"
              d={roundedOrthogonalPath([
                [branch.lane, branch.targetY],
                [RIGHT_X, branch.targetY],
              ])}
            />
            <path
              className="metric-mapping-edge is-receive-model-edge"
              d={roundedOrthogonalPath([
                [branch.lane, branch.targetY],
                [RIGHT_X, branch.targetY],
              ])}
              style={{ stroke: RECEIVE_PROXY_COLORS[branch.proxy] }}
            />
            <circle
              className="metric-mapping-edge-port is-model"
              cx={RIGHT_X}
              cy={branch.targetY}
              r="3"
              fill={RECEIVE_PROXY_COLORS[branch.proxy]}
            />
          </g>
        ))}

        <g
          className={`metric-mapping-link-layer${relationClass(
            sourceRelated("alignment"),
          )}`}
        >
          <path
            className="metric-mapping-edge-halo is-alignment"
            d={roundedOrthogonalPath([
              [SOURCE_EDGE_X, alignmentCenter],
              [RIGHT_X, alignmentCenter],
            ])}
          />
          <path
            className="metric-mapping-edge is-alignment"
            d={roundedOrthogonalPath([
              [SOURCE_EDGE_X, alignmentCenter],
              [RIGHT_X, alignmentCenter],
            ])}
            style={{ stroke: SOURCE_COLORS.alignment }}
          />
          <circle
            className="metric-mapping-edge-port is-alignment"
            cx={RIGHT_X}
            cy={alignmentCenter}
            r="3.6"
            fill={SOURCE_COLORS.alignment}
          />
        </g>

      </svg>

      <div
        className="metric-mapping-column-label is-model"
        style={{ top: 16, left: MODEL_COLUMN_X }}
      >
        感知模型 / 虚拟 DRP · 代理
      </div>
      <div
        className="metric-mapping-column-label is-source"
        style={{ top: 16, left: SOURCE_X }}
      >
        可映射指标 · 5 类
      </div>
      <div
        className="metric-mapping-channel-note"
        style={{ top: 46, left: SOURCE_X }}
      >
        模型三轨独立；工程轨按目标重心排序
      </div>
      <div
        className="metric-mapping-column-label is-contract"
        style={{ top: 16, left: RIGHT_X }}
      >
        认证指标 · {mapping.mode} · 42 项
      </div>
      <div
        className="metric-mapping-column-meta"
        style={{ top: 44, left: RIGHT_X + RIGHT_WIDTH }}
      >
        {mergedCoveredCount} 项有映射 · {mapping.mappedCount}+{dnsmosMappedCount}+
        {receiveMappedCount} ·{" "}
        {42 - mergedCoveredCount} 项未建模
      </div>

      <article
        className="metric-dnsmos-model metric-mapping-model-card is-dnsmos"
        style={
          {
            top: MODEL_CARD_TOP,
            left: MODEL_COLUMN_X,
            width: MODEL_COLUMN_WIDTH,
            height: MODEL_CARD_HEIGHT,
            "--track-color": OUTPUT_COLORS.dnsmos.sig,
          } as CSSProperties
        }
      >
        <div className="metric-dnsmos-model-head">
          <div className="metric-dnsmos-model-head-top">
            <span>{DNSMOS_TRACK.eyebrow}</span>
            <em
              className={
                dnsmosMapping?.valid
                  ? "is-engineering"
                  : "is-experimental"
              }
            >
              {dnsmosMapping?.valid ? DNSMOS_TRACK.status : "映射数据异常"}
            </em>
          </div>
          <strong>{DNSMOS_TRACK.title}</strong>
          <p>{DNSMOS_TRACK.detail}</p>
        </div>
        <div
          className="metric-dnsmos-output-list"
          role="list"
          aria-label="DNSMOS P.835 输出及认证映射"
        >
          {DNSMOS_TRACK.outputs.map((output) => (
            <article
              key={output.output}
              role="listitem"
              className={`metric-mapping-interactive metric-dnsmos-output${relationClass(
                dnsmosRelated(output.output),
              )}${
                lockedFocus?.kind === "dnsmos" &&
                lockedFocus.output === output.output
                  ? " is-focus-locked"
                  : ""
              }`}
              style={
                {
                  top: output.top,
                  "--mapping-color":
                    OUTPUT_COLORS.dnsmos[output.output],
                } as CSSProperties
              }
              {...interactionProps({
                kind: "dnsmos",
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

      <article
        className="metric-receive-main-card"
        style={{
          top: RECEIVE_CARD_TOP,
          left: MODEL_COLUMN_X,
          width: MODEL_COLUMN_WIDTH,
          height: RECEIVE_CARD_HEIGHT,
        }}
        aria-label="虚拟 DRP 双耳仿真输出及认证映射"
      >
        <header>
          <div>
            <span>SOFTWARE-ONLY RECEIVE PROXY</span>
            <em>工程代理 · 未校准</em>
          </div>
          <h3>
            <Headphones size={15} aria-hidden="true" />
            虚拟 DRP 双耳仿真
          </h3>
          <p>
            双耳电平、频响、噪声、带外、三档线性残差与 P.835
            TMOS 代理；不构成正式 HATS 证据。
          </p>
        </header>
        <div className="metric-receive-main-output-list" role="list">
          {RECEIVE_PROXY_IDS.map((proxy, index) => {
            const rows =
              receiveMapping?.rows.filter((row) => row.proxy === proxy) ?? [];
            const count = receiveMapping?.proxyCounts[proxy] ?? 0;
            return (
              <article
                key={proxy}
                role="listitem"
                className={`metric-mapping-interactive metric-receive-main-output${relationClass(
                  receiveRelated(proxy),
                )}${
                  lockedFocus?.kind === "receive" &&
                  lockedFocus.proxy === proxy
                    ? " is-focus-locked"
                    : ""
                }`}
                style={
                  {
                    top:
                      RECEIVE_OUTPUT_TOP +
                      index * RECEIVE_OUTPUT_STEP,
                    "--mapping-color": RECEIVE_PROXY_COLORS[proxy],
                  } as CSSProperties
                }
                {...interactionProps({ kind: "receive", proxy })}
              >
                <span>{RECEIVE_PROXY_LABELS[proxy]}</span>
                <strong>{count} 条</strong>
                <small>
                  {rows.length
                    ? rows.map((row) => `B${row.row}`).join(" · ")
                    : "当前模式无对应行"}
                </small>
              </article>
            );
          })}
        </div>
        <footer>
          B5/B6 使用三档线性残差模型；B10 使用双耳 P.835 保守代理。
          正式 RLR、TMOS/POLQA 与 dB[Pa] 校准仍需测量。
        </footer>
      </article>

      <div
        className={`metric-mapping-alignment metric-mapping-interactive${relationClass(
          sourceRelated("alignment"),
        )}${
          lockedFocus?.kind === "source" &&
          lockedFocus.source === "alignment"
            ? " is-focus-locked"
            : ""
        }`}
        style={
          {
            top: ROW_TOP,
            left: SOURCE_X,
            width: LEFT_WIDTH,
            height: ROW_HEIGHT,
            "--mapping-color": SOURCE_COLORS.alignment,
          } as CSSProperties
        }
        {...interactionProps({ kind: "source", source: "alignment" })}
      >
        <span className="metric-mapping-alignment-tag">共同前置</span>
        <strong>相对时延 / 对齐时间基线</strong>
        <span className="metric-mapping-source-count">
          {mapping.sourceCounts.alignment} 条
        </span>
      </div>

      <div
        className="metric-mapping-source-list"
        role="list"
        aria-label="五项可映射测试指标"
      >
        {SOURCE_CARDS.map((card) => (
          <article
            key={card.id}
            role="listitem"
            className={`metric-mapping-source-card metric-mapping-interactive${relationClass(sourceRelated(card.id))}${
              lockedFocus?.kind === "source" &&
              lockedFocus.source === card.id
                ? " is-focus-locked"
                : ""
            }`}
            data-source={card.id}
            style={
              {
                top: card.top,
                left: SOURCE_X,
                width: LEFT_WIDTH,
                height: LEFT_HEIGHT,
                "--mapping-color": SOURCE_COLORS[card.id],
              } as CSSProperties
            }
            {...interactionProps({ kind: "source", source: card.id })}
          >
            <span className="metric-mapping-source-code">{card.code}</span>
            <strong>{card.title}</strong>
            <p>{card.copy}</p>
            <div className="metric-mapping-source-footer">
              <span>{mapping.sourceCounts[card.id]} 条认证行</span>
              <i aria-hidden="true" />
            </div>
          </article>
        ))}
      </div>

      <div
        className="metric-mapping-contract-list"
        role="list"
        aria-label={`${mapping.mode} 的 42 项认证指标`}
      >
        {targetRows.map((row) => {
          const modelMeta = row.dnsmosMeta;
          const modelOutput = row.dnsmosOutput;
          const receiveProxy = row.receiveProxy;
          const covered =
            row.source !== null ||
            modelOutput !== null ||
            receiveProxy !== null;
          const labels = [
            row.source
              ? `工程来源 ${METRIC_SOURCE_LABELS[row.source]}`
              : null,
            modelMeta
              ? `DNSMOS P.835 ${modelMeta.title} 模型代理 ${modelMeta.contract}`
              : null,
            receiveProxy
              ? `虚拟 DRP ${RECEIVE_PROXY_LABELS[receiveProxy]} 工程代理`
              : null,
          ].filter(Boolean);
          const evidenceLabel =
            row.source && (modelMeta || receiveProxy)
              ? "多源证据"
              : modelMeta && receiveProxy
                ? "P.835 + DRP"
                : modelMeta
                  ? row.source
                    ? "工程 + P.835"
                    : `P.835 ${modelMeta.title}`
                  : receiveProxy
                    ? "DRP 代理"
                    : statusLabel(row.status);
          return (
            <article
              key={row.id}
              role="listitem"
              className={`metric-mapping-contract-row metric-mapping-interactive ${
                covered ? "is-mapped" : "is-unmapped"
              } ${modelMeta ? "is-model-proxy" : ""}${
                receiveProxy ? " is-receive-proxy" : ""
              }${
                isBusinessGroupStart(row.row) ? " is-group-start" : ""
              }${relationClass(contractRelated(row.row))}${
                lockedFocus?.kind === "contract" &&
                lockedFocus.row === row.row
                  ? " is-focus-locked"
                  : ""
              }`}
              data-source={
                row.source ??
                (modelMeta ? "dnsmos" : receiveProxy ? "receive" : "none")
              }
              data-business-group={businessGroup(row.row)}
              style={
                {
                  top: row.top,
                  left: RIGHT_X,
                  width: RIGHT_WIDTH,
                  height: ROW_HEIGHT,
                  "--mapping-color":
                    row.source !== null
                      ? sourceColor(row.source)
                      : modelOutput
                        ? OUTPUT_COLORS.dnsmos[modelOutput]
                        : receiveProxy
                          ? "#0f766e"
                        : "#94a3b8",
                } as CSSProperties
              }
              aria-label={`B${row.row}，${row.item.summary}，${
                labels.length ? labels.join("；") : "不可测"
              }，${businessGroupLabel(row.row)}`}
              {...interactionProps({ kind: "contract", row: row.row })}
            >
              <span className="metric-mapping-row-id">B{row.row}</span>
              <div className="metric-mapping-row-copy">
                <strong title={row.item.name}>{row.item.summary}</strong>
                <small>
                  {businessGroupLabel(row.row)} ·{" "}
                  {row.item.unit || row.item.category}
                </small>
              </div>
              <span className="metric-mapping-row-status">
                {evidenceLabel}
              </span>
            </article>
          );
        })}
      </div>

      <aside
        className="metric-mapping-guide-card"
        style={{
          top: GUIDE_CARD_TOP,
          left: MODEL_COLUMN_X,
          width: MODEL_COLUMN_WIDTH,
          height: GUIDE_CARD_HEIGHT,
        }}
        aria-label="证据层级与四场景映射导读"
      >
        <header>
          <span>MAP READING GUIDE</span>
          <strong>证据层级与四场景映射</strong>
          <p>点击任意来源或合同行，锁定其完整关联链路。</p>
        </header>

        <section>
          <h4>线宽与线型</h4>
          <div className="metric-mapping-guide-line">
            <i className="is-model" />
            <span>DNSMOS 三路模型代理</span>
            <b>2.4px</b>
          </div>
          <div className="metric-mapping-guide-line">
            <i className="is-engineering" />
            <span>五类工程指标</span>
            <b>1.75px</b>
          </div>
          <div className="metric-mapping-guide-line">
            <i className="is-alignment" />
            <span>对齐时间基线</span>
            <b>直达</b>
          </div>
          <div className="metric-mapping-guide-line">
            <i className="is-receive" />
            <span>虚拟 DRP 接收代理</span>
            <b>{receiveMappedCount} 行</b>
          </div>
          <div className="metric-mapping-guide-line">
            <i className="is-unmapped" />
            <span>无直接合同映射</span>
            <b>虚线</b>
          </div>
        </section>

        <section>
          <div className="metric-mapping-guide-section-head">
            <h4>四类场景</h4>
            <strong>4 × 3 = 12</strong>
          </div>
          {[
            ["01", "怠速静止", "B24–B28"],
            ["02", "60km/h 市区", "B29–B33"],
            ["03", "120km/h 高速", "B34–B38"],
            ["04", "空调大风档", "B39–B43"],
          ].map(([index, label, rows]) => (
            <div className="metric-mapping-guide-scenario" key={index}>
              <em>{index}</em>
              <span>
                <strong>{label}</strong>
                <small>{rows}</small>
              </span>
              <div aria-label={`${label}包含 OVRL、BAK、SIG`}>
                <i>OVRL</i>
                <i>BAK</i>
                <i>SIG</i>
              </div>
            </div>
          ))}
        </section>

        <footer>
          <span>聚焦提示</span>
          <p>
            Hover 预览关联链路；Click / Enter 锁定；再次点击或 Esc 取消。
            其余路径退隐至 14%。
          </p>
        </footer>
      </aside>

      <div className="metric-mapping-legend" style={{ top: 1808, left: 48 }}>
        {(
          [
            "alignment",
            "stoi",
            "level",
            "spectrum",
            "doubleTalk",
            "echo",
          ] as MetricSourceId[]
        ).map((source) => (
          <span key={source} className="metric-mapping-legend-item">
            <i
              aria-hidden="true"
              style={{ backgroundColor: SOURCE_COLORS[source] }}
            />
            {METRIC_SOURCE_LABELS[source]}
          </span>
        ))}
        <span className="metric-mapping-legend-item">
          <i
            className="is-dnsmos"
            aria-hidden="true"
            style={{ backgroundColor: OUTPUT_COLORS.dnsmos.sig }}
          />
          DNSMOS P.835 · 12 模型代理行
        </span>
        <span className="metric-mapping-legend-item">
          <i className="is-receive" aria-hidden="true" />
          虚拟 DRP 接收代理 · {receiveMappedCount} 行
        </span>
        <span className="metric-mapping-legend-item">
          <i className="is-unmapped" aria-hidden="true" />
          不可测
        </span>
      </div>
      <p className="metric-mapping-note" style={{ top: 1836, left: 48 }}>
        映射口径：当前 {mapping.mode} 工程测量、DNSMOS 模型代理与虚拟 DRP
        接收代理并列展示；B5/B6 是受控三档线性残差模型，B10 是双耳
        P.835 保守代理，三类映射都不等于正式合同通过。
      </p>
    </div>
  );
}

function MetricMappingTable({
  mapping,
  dnsmosMapping,
  receiveMapping,
}: {
  mapping: ReturnType<typeof buildMetricMapping>;
  dnsmosMapping: DnsmosMappingSnapshot | null;
  receiveMapping: ReceiveProxyMappingSnapshot | null;
}) {
  const [filter, setFilter] = useState<"all" | "mapped" | "unmapped">("all");
  const [query, setQuery] = useState("");
  const mergedRows = useMemo(() => {
    const dnsmosByNumber = new Map(
      dnsmosMapping?.rows.map((row) => [row.row, row]) ?? [],
    );
    const receiveByNumber = new Map(
      receiveMapping?.rows.map((row) => [row.row, row]) ?? [],
    );
    return mapping.rows.map((row) => {
      const dnsmosOutput = dnsmosByNumber.get(row.row)?.output ?? null;
      const receive = receiveByNumber.get(row.row) ?? null;
      return {
        ...row,
        dnsmosOutput,
        receiveProxy: receive?.proxy ?? null,
        receiveRow: receive,
        dnsmosMeta: dnsmosOutput
          ? DNSMOS_TRACK.outputs.find(
              (output) => output.output === dnsmosOutput,
            ) ?? null
          : null,
      };
    });
  }, [dnsmosMapping?.rows, mapping.rows, receiveMapping?.rows]);
  const mergedCoveredCount = mergedRows.filter(
    (row) =>
      row.source !== null ||
      row.dnsmosOutput !== null ||
      row.receiveProxy !== null,
  ).length;
  const mergedUnmappedCount = mapping.rows.length - mergedCoveredCount;

  const filteredRows = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    return mergedRows.filter((row) => {
      const covered =
        row.source !== null ||
        row.dnsmosOutput !== null ||
        row.receiveProxy !== null;
      if (filter === "mapped" && !covered) return false;
      if (filter === "unmapped" && covered) return false;
      if (!normalized) return true;
      return [
        `b${row.row}`,
        row.item.category,
        row.item.name,
        row.item.summary,
        row.item.unit,
        row.source ? METRIC_SOURCE_LABELS[row.source] : "不可测",
        row.dnsmosMeta
          ? `DNSMOS P.835 ${row.dnsmosMeta.title} ${row.dnsmosMeta.contract} 模型代理`
          : "",
        row.receiveProxy
          ? `虚拟 DRP ${RECEIVE_PROXY_LABELS[row.receiveProxy]} 接收工程代理`
          : row.receiveRow
            ? `虚拟 DRP 接收映射未闭环 ${row.receiveRow.gap}`
            : "",
      ]
        .join(" ")
        .toLowerCase()
        .includes(normalized);
    });
  }, [filter, mergedRows, query]);

  const headingId = `metric-mapping-detail-${mapping.mode}`;

  return (
    <section className="metric-mapping-detail" aria-labelledby={headingId}>
      <div className="metric-mapping-detail-head">
        <div>
          <span className="metric-mapping-detail-eyebrow">映射明细</span>
          <h3 id={headingId}>42 项认证指标映射表</h3>
          <p>
            逐行核对认证合同、工程测量、DNSMOS P.835 与虚拟 DRP 接收代理。
            表格只描述映射关系，不改变正式合规状态。
          </p>
        </div>
        <div className="metric-mapping-source-strip" aria-label="证据来源分布">
          {METRIC_SOURCE_IDS.filter((source) => source !== "sisdr").map((source) => (
            <span
              key={source}
              className={mapping.sourceCounts[source] ? "has-count" : ""}
              style={{ "--mapping-color": SOURCE_COLORS[source] } as CSSProperties}
            >
              <i aria-hidden="true" />
              {METRIC_SOURCE_LABELS[source]}
              <strong>{mapping.sourceCounts[source]}</strong>
            </span>
          ))}
          <span
            className={dnsmosMapping?.mappedCount ? "has-count" : ""}
            style={{ "--mapping-color": OUTPUT_COLORS.dnsmos.sig } as CSSProperties}
          >
            <i aria-hidden="true" />
            DNSMOS P.835 模型代理
            <strong>{dnsmosMapping?.mappedCount ?? 0}</strong>
          </span>
          <span
            className={receiveMapping?.proxyCount ? "has-count" : ""}
            style={{ "--mapping-color": "#0f766e" } as CSSProperties}
          >
            <i aria-hidden="true" />
            虚拟 DRP 接收代理
            <strong>{receiveMapping?.proxyCount ?? 0}</strong>
          </span>
          <span className="is-unmapped">
            <i aria-hidden="true" />
            未建模
            <strong>{mergedUnmappedCount}</strong>
          </span>
        </div>
      </div>

      <div className="metric-mapping-table-toolbar">
        <div className="metric-mapping-filter-group" role="group" aria-label="筛选映射状态">
          {(
            [
              ["all", "全部", mapping.rows.length],
              ["mapped", "有映射 / 代理", mergedCoveredCount],
              ["unmapped", "未建模", mergedUnmappedCount],
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
          <span className="sr-only">搜索认证指标、分类或证据来源</span>
          <input
            type="search"
            value={query}
            placeholder="搜索认证指标、分类或证据来源…"
            onChange={(event) => setQuery(event.target.value)}
          />
        </label>
        <span className="metric-mapping-table-count" aria-live="polite">
          {filteredRows.length} / {mapping.rows.length} 项
        </span>
      </div>

      {filteredRows.length ? (
        <div
          className="metric-mapping-table-scroll"
          role="region"
          aria-label={`${mapping.mode} 42 项认证指标映射明细表`}
          tabIndex={0}
        >
          <table className="metric-mapping-table">
            <caption className="sr-only">
              {mapping.mode} 的 42 项认证指标、工程测量、DNSMOS P.835
              模型代理与虚拟 DRP 接收代理来源和映射说明
            </caption>
            <thead>
              <tr>
                <th scope="col">合同行</th>
                <th scope="col">合同分类</th>
                <th scope="col">认证指标</th>
                <th scope="col">单位</th>
                <th scope="col">标准限值</th>
                <th scope="col">证据来源</th>
                <th scope="col">证据等级</th>
                <th scope="col">说明</th>
              </tr>
            </thead>
            <tbody>
              {filteredRows.map((row) => {
                const modelMeta = row.dnsmosMeta;
                const receiveRow = row.receiveRow;
                const receiveProxy = row.receiveProxy;
                const evidenceCount = [
                  row.source !== null,
                  modelMeta !== null,
                  receiveProxy !== null,
                ].filter(Boolean).length;
                const hasEvidence = evidenceCount > 0;
                const evidenceStatus =
                  evidenceCount > 1
                    ? "多源证据"
                    : modelMeta
                      ? "模型代理"
                      : receiveProxy
                        ? "接收工程代理"
                        : statusLabel(row.status);
                const explanation = [
                  row.source ? mappingNote(row) : null,
                  modelMeta
                    ? `DNSMOS P.835 ${modelMeta.title} 映射 ${modelMeta.contract}，仅作为感知模型代理。`
                    : null,
                  receiveProxy && receiveRow
                    ? `虚拟 DRP ${RECEIVE_PROXY_LABELS[receiveProxy]} 可提供工程代理；${receiveRow.gap}`
                    : receiveRow
                      ? `虚拟 DRP 接收映射未闭环：${receiveRow.gap}`
                      : null,
                  !hasEvidence ? mappingNote(row) : null,
                ]
                  .filter(Boolean)
                  .join(" ");
                return (
                  <tr key={row.id}>
                    <th scope="row">
                      <code>B{row.row}</code>
                    </th>
                    <td>
                      <span className="metric-mapping-category">
                        {row.item.category}
                      </span>
                    </td>
                    <td className="metric-mapping-target">
                      <strong>{row.item.summary}</strong>
                      <small>{row.item.name}</small>
                    </td>
                    <td>{row.item.unit || "—"}</td>
                    <td className="metric-mapping-limit">
                      <NV200LimitRange item={row.item} />
                    </td>
                    <td>
                      <div className="metric-mapping-source-badges">
                        {row.source ? (
                          <span
                            className="metric-mapping-source-badge"
                            style={
                              {
                                "--mapping-color": SOURCE_COLORS[row.source],
                              } as CSSProperties
                            }
                          >
                            <i aria-hidden="true" />
                            {METRIC_SOURCE_LABELS[row.source]}
                          </span>
                        ) : null}
                        {modelMeta ? (
                          <span
                            className="metric-mapping-source-badge is-dnsmos"
                            style={
                              {
                                "--mapping-color":
                                  OUTPUT_COLORS.dnsmos[
                                    modelMeta.output
                                  ],
                              } as CSSProperties
                            }
                          >
                            <i aria-hidden="true" />
                            P.835 {modelMeta.title}
                          </span>
                        ) : null}
                        {receiveProxy ? (
                          <span
                            className="metric-mapping-source-badge is-receive"
                            style={
                              {
                                "--mapping-color": "#0f766e",
                              } as CSSProperties
                            }
                          >
                            <i aria-hidden="true" />
                            DRP {RECEIVE_PROXY_LABELS[receiveProxy]}
                          </span>
                        ) : null}
                        {!hasEvidence ? (
                          <span className="metric-mapping-source-badge is-unmapped">
                            <i aria-hidden="true" />
                            不可测
                          </span>
                        ) : null}
                      </div>
                    </td>
                    <td>
                      <span
                        className={`metric-mapping-evidence-status ${
                          modelMeta || receiveProxy
                            ? "engineering_evidence"
                            : row.status
                        }`}
                      >
                        {evidenceStatus}
                      </span>
                    </td>
                    <td className="metric-mapping-explanation">
                      {explanation}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="metric-mapping-no-results" role="status">
          没有符合当前筛选条件的映射项。
        </p>
      )}
    </section>
  );
}

export default function MetricMappingView({
  allSheets,
  selectedSheetId,
  onSelectSheetId,
}: {
  allSheets: NV200SheetData[];
  selectedSheetId: string;
  onSelectSheetId: (sheetId: string) => void;
}) {
  const currentSheet = useMemo(
    () =>
      allSheets.find((sheet) => sheet.sheetId === selectedSheetId) ??
      allSheets[0],
    [allSheets, selectedSheetId],
  );
  const mapping = useMemo(
    () => (currentSheet ? buildMetricMapping(currentSheet) : null),
    [currentSheet],
  );
  const dnsmosMapping = useMemo(
    () => (currentSheet ? buildDnsmosMapping(currentSheet) : null),
    [currentSheet],
  );
  const receiveMapping = useMemo(
    () => (currentSheet ? buildReceiveProxyMapping(currentSheet) : null),
    [currentSheet],
  );
  const mergedCoveredCount = useMemo(
    () =>
      mapping?.rows.filter(
        (row, index) =>
          row.source !== null ||
          Boolean(dnsmosMapping?.rows[index]?.output) ||
          Boolean(receiveMapping?.rows.find((candidate) => candidate.row === row.row)?.proxy),
      ).length ?? 0,
    [dnsmosMapping?.rows, mapping?.rows, receiveMapping?.rows],
  );

  if (!currentSheet || !mapping) {
    return (
      <section className="panel metric-mapping-panel">
        <div className="metric-mapping-error" role="alert">
          <AlertCircle size={18} />
          当前没有可用的 NV200 模式数据。
        </div>
      </section>
    );
  }

  return (
    <section className="panel metric-mapping-panel">
      <div className="metric-mapping-heading">
        <div>
          <div className="metric-mapping-eyebrow">
            <Network size={15} aria-hidden="true" />
            结构映射
          </div>
          <h2>五项可映射指标 + 感知模型 + 虚拟 DRP → 42 项认证指标</h2>
          <p>
            五项可映射工程指标、DNSMOS P.835 与虚拟 DRP 接收代理并列支撑当前业务模式的认证合同行；
            连线只表示证据来源，不代表正式 ITU-T 合规通过。
          </p>
        </div>
        <div className="metric-mapping-summary" aria-live="polite">
          <strong>{mergedCoveredCount}</strong>
          <span>有映射 / 代理</span>
          <i aria-hidden="true" />
          <strong>{42 - mergedCoveredCount}</strong>
          <span>未建模</span>
        </div>
      </div>

      <ModeSheetBar
        allSheets={allSheets}
        selectedSheetId={currentSheet.sheetId}
        onSelectSheetId={onSelectSheetId}
      />

      {!mapping.valid ? (
        <div className="metric-mapping-error" role="alert">
          <AlertCircle size={18} />
          {mapping.problem}
        </div>
      ) : (
        <>
          <p className="sr-only">
            当前 {mapping.mode} 共 42 项认证指标，{mergedCoveredCount} 项至少有一类映射或模型代理，
            {42 - mergedCoveredCount} 项尚未建模。图中展示五项可映射工程指标：
            STOI、电平与 SNR、频响与频谱、双讲透传、回声损耗；相对时延与对齐作为共同前置。
            SI-SDR 仍保留为后端诊断指标，但没有直接认证行，因此不占用映射图节点。
            左侧 DNSMOS P.835 是独立的模型代理层，12 项 G/N/S-MOS 合同行同时显示
            P.835 输出文字标记；虚拟 DRP 接收代理补充 B3–B11 中可对应的
            电平、频响、底噪、峰值、带外、三档 RLR 线性残差和双耳 P.835
            TMOS 代理。所有代理层都不代表正式 ITU-T
            合规通过。
          </p>
          <div
            key={mapping.mode}
            className="metric-mapping-scroll"
            role="region"
            aria-label={`${mapping.mode} 五项可映射测试指标、DNSMOS P.835 与虚拟 DRP 到 42 项认证指标关系图`}
            tabIndex={0}
          >
            <MetricMappingCanvas
              mapping={mapping}
              dnsmosMapping={dnsmosMapping}
              receiveMapping={receiveMapping}
            />
          </div>
          <MetricMappingTable
            key={`mapping-detail-${mapping.mode}`}
            mapping={mapping}
            dnsmosMapping={dnsmosMapping}
            receiveMapping={receiveMapping}
          />
          {dnsmosMapping ? (
            <DnsmosMappingView
              key={`dnsmos-mapping-${currentSheet.sheetId}`}
              mapping={dnsmosMapping}
            />
          ) : null}
        </>
      )}
    </section>
  );
}
