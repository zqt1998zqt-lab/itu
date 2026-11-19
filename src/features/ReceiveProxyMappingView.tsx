import { useEffect, useMemo, useState } from "react";
import type { CSSProperties, KeyboardEvent } from "react";
import { AlertCircle, Headphones, Network } from "lucide-react";
import {
  RECEIVE_PROXY_IDS,
  RECEIVE_PROXY_LABELS,
  type ReceiveProxyId,
  type ReceiveProxyMappingRow,
  type ReceiveProxyMappingSnapshot,
} from "./five-signal-flow-model";
import { roundedOrthogonalPath } from "./DnsmosMappingView";

const PROXY_COLORS: Record<ReceiveProxyId, string> = {
  binauralLevel: "#0f766e",
  frequencyResponse: "#7c3aed",
  idleNoise: "#0369a1",
  noisePeaks: "#b45309",
  outOfBand: "#be123c",
  rlrLinearity: "#0e7490",
  binauralP835: "#4f46e5",
};

const CANVAS_WIDTH = 1160;
const CANVAS_HEIGHT = 710;
const MODEL_X = 24;
const MODEL_WIDTH = 316;
const MODEL_TOP = 54;
const MODEL_HEIGHT = 600;
const MODEL_EDGE_X = MODEL_X + MODEL_WIDTH;
const CONTRACT_X = 728;
const CONTRACT_WIDTH = 408;
const CONTRACT_TOP = 54;
const CONTRACT_HEIGHT = 610;
const ROW_TOP = 104;
const ROW_HEIGHT = 48;
const ROW_STEP = 54;
const OUTPUT_TOP = 124;
const OUTPUT_HEIGHT = 54;
const OUTPUT_STEP = 64;
const LANE_XS: Record<ReceiveProxyId, number> = {
  binauralLevel: 372,
  frequencyResponse: 392,
  idleNoise: 412,
  noisePeaks: 432,
  outOfBand: 452,
  rlrLinearity: 472,
  binauralP835: 492,
};

type ReceiveFocus =
  | { kind: "proxy"; proxy: ReceiveProxyId }
  | { kind: "row"; row: number }
  | null;

function rowCenter(row: number) {
  return ROW_TOP + (row - 3) * ROW_STEP + ROW_HEIGHT / 2;
}

function sameFocus(left: ReceiveFocus, right: ReceiveFocus) {
  if (left === null || right === null) return left === right;
  if (left.kind !== right.kind) return false;
  if (left.kind === "proxy" && right.kind === "proxy") {
    return left.proxy === right.proxy;
  }
  if (left.kind === "row" && right.kind === "row") {
    return left.row === right.row;
  }
  return false;
}

function ReceiveProxyCanvas({
  mapping,
}: {
  mapping: ReceiveProxyMappingSnapshot;
}) {
  const [hoverFocus, setHoverFocus] = useState<ReceiveFocus>(null);
  const [lockedFocus, setLockedFocus] = useState<ReceiveFocus>(null);
  const focusTarget = lockedFocus ?? hoverFocus;

  useEffect(() => {
    const clearFocus = (event: globalThis.KeyboardEvent) => {
      if (event.key !== "Escape") return;
      setHoverFocus(null);
      setLockedFocus(null);
    };
    window.addEventListener("keydown", clearFocus);
    return () => window.removeEventListener("keydown", clearFocus);
  }, []);

  const focusedRow =
    focusTarget?.kind === "row"
      ? mapping.rows.find((row) => row.row === focusTarget.row) ?? null
      : null;

  const branches = useMemo(
    () =>
      RECEIVE_PROXY_IDS.flatMap((proxy, index) =>
        mapping.rows
          .filter((row) => row.proxy === proxy)
          .map((row) => ({
            proxy,
            lane: LANE_XS[proxy],
            sourceY: OUTPUT_TOP + index * OUTPUT_STEP + OUTPUT_HEIGHT / 2,
            targetY: rowCenter(row.row),
            row,
          })),
      ),
    [mapping.rows],
  );

  function relationClass(related: boolean) {
    if (!focusTarget) return "";
    return related ? " is-focus-related" : " is-focus-muted";
  }

  function proxyRelated(proxy: ReceiveProxyId) {
    if (!focusTarget) return true;
    if (focusTarget.kind === "proxy") return focusTarget.proxy === proxy;
    return focusedRow?.proxy === proxy;
  }

  function rowRelated(row: ReceiveProxyMappingRow) {
    if (!focusTarget) return true;
    if (focusTarget.kind === "row") {
      if (row.row === focusTarget.row) return true;
      if (!focusedRow?.proxy || !row.proxy) return false;
      return row.proxy === focusedRow.proxy;
    }
    return row.proxy === focusTarget.proxy;
  }

  function interactionProps(target: NonNullable<ReceiveFocus>) {
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
        if (event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault();
        setHoverFocus(null);
        setLockedFocus((current) =>
          sameFocus(current, target) ? null : target,
        );
      },
    };
  }

  return (
    <div
      className="metric-mapping-canvas metric-receive-canvas"
      style={{ width: CANVAS_WIDTH, height: CANVAS_HEIGHT }}
    >
      <svg
        className="metric-mapping-connectors"
        viewBox={`0 0 ${CANVAS_WIDTH} ${CANVAS_HEIGHT}`}
        aria-hidden="true"
        focusable="false"
      >
        {RECEIVE_PROXY_IDS.map((proxy, index) => {
          const outputRows = branches.filter((branch) => branch.proxy === proxy);
          if (!outputRows.length) return null;
          const sourceY = OUTPUT_TOP + index * OUTPUT_STEP + OUTPUT_HEIGHT / 2;
          const targetYs = outputRows.map((branch) => branch.targetY);
          const minY = Math.min(sourceY, ...targetYs);
          const maxY = Math.max(sourceY, ...targetYs);
          const lane = LANE_XS[proxy];
          const trunk = roundedOrthogonalPath([
            [MODEL_EDGE_X, sourceY],
            [lane, sourceY],
            [lane, maxY],
          ]);
          return (
            <g
              key={`receive-trunk-${proxy}`}
              className={`metric-mapping-link-layer${relationClass(proxyRelated(proxy))}`}
            >
              <path
                className="metric-mapping-edge-halo is-trunk"
                d={trunk}
              />
              <path
                className="metric-mapping-route-spine is-receive-rail"
                d={trunk}
                style={{ stroke: PROXY_COLORS[proxy] }}
              />
              {sourceY > minY && sourceY < maxY ? (
                <path
                  className="metric-mapping-route-spine is-receive-rail"
                  d={roundedOrthogonalPath([
                    [lane, minY],
                    [lane, sourceY],
                  ])}
                  style={{ stroke: PROXY_COLORS[proxy] }}
                />
              ) : null}
              {outputRows.map((branch) => (
                <g key={`receive-branch-${proxy}-${branch.row.row}`}>
                  <path
                    className="metric-mapping-edge-halo is-branch"
                    d={roundedOrthogonalPath([
                      [lane, branch.targetY],
                      [CONTRACT_X, branch.targetY],
                    ])}
                  />
                  <path
                    className="metric-mapping-edge is-receive-edge"
                    d={roundedOrthogonalPath([
                      [lane, branch.targetY],
                      [CONTRACT_X, branch.targetY],
                    ])}
                    style={{ stroke: PROXY_COLORS[proxy] }}
                  />
                  <circle
                    className="metric-mapping-edge-port is-receive"
                    cx={CONTRACT_X}
                    cy={branch.targetY}
                    r="3"
                    fill={PROXY_COLORS[proxy]}
                  />
                </g>
              ))}
              <circle
                className="metric-mapping-route-junction"
                cx={lane}
                cy={sourceY}
                r="3.2"
              />
              <circle
                className="metric-mapping-route-junction-core"
                cx={lane}
                cy={sourceY}
                r="1.5"
                fill={PROXY_COLORS[proxy]}
              />
            </g>
          );
        })}
      </svg>

      <div className="metric-receive-column-label is-model">
        虚拟 DRP 输出 · 5 类
      </div>
      <div className="metric-receive-column-label is-contract">
        下行接收合同 · B3–B11 · 9 项
      </div>

      <article className="metric-receive-model-card">
        <header>
          <span>SOFTWARE-ONLY RECEIVE PROXY</span>
          <em>工程代理 · 未校准</em>
          <h4>
            <Headphones size={16} aria-hidden="true" />
            虚拟 DRP 双耳仿真
          </h4>
          <p>
            Woodworth 球头模型或自定义双耳 IR；输出未校准 dBFS，不构成
            HATS / P.57 / P.58 正式证据。
          </p>
        </header>

        <div className="metric-receive-output-list" role="list">
          {RECEIVE_PROXY_IDS.map((proxy, index) => {
            const count = mapping.proxyCounts[proxy];
            return (
              <article
                key={proxy}
                role="listitem"
                className={`metric-mapping-interactive metric-receive-output${relationClass(
                  proxyRelated(proxy),
                )}${
                  lockedFocus?.kind === "proxy" &&
                  lockedFocus.proxy === proxy
                    ? " is-focus-locked"
                    : ""
                }`}
                style={
                  {
                    top: OUTPUT_TOP + index * OUTPUT_STEP,
                    "--mapping-color": PROXY_COLORS[proxy],
                  } as CSSProperties
                }
                {...interactionProps({ kind: "proxy", proxy })}
              >
                <span>{RECEIVE_PROXY_LABELS[proxy]}</span>
                <strong>{count} 条合同行</strong>
                <small>
                  {count
                    ? mapping.rows
                        .filter((row) => row.proxy === proxy)
                        .map((row) => `B${row.row}`)
                        .join(" · ")
                    : "当前模式无对应行"}
                </small>
              </article>
            );
          })}
        </div>

        <footer>
          <span>未闭环输出</span>
          <p>
            RLR / P.79、TMOS / POLQA、A 计权 dB[Pa] 与合同容差掩模仍需正式测量。
          </p>
        </footer>
      </article>

      <div
        className="metric-receive-contract-shell"
        style={{
          top: CONTRACT_TOP,
          left: CONTRACT_X,
          width: CONTRACT_WIDTH,
          height: CONTRACT_HEIGHT,
        }}
      >
        <header>
          <strong>{mapping.mode} · 下行接收</strong>
          <span>
            {mapping.proxyCount} 工程代理 · {mapping.unmappedCount} 未闭环
          </span>
        </header>
        <div className="metric-receive-row-list" role="list">
          {mapping.rows.map((row, index) => {
            const proxy = row.proxy;
            return (
              <article
                key={row.id}
                role="listitem"
                className={`metric-mapping-interactive metric-receive-row ${
                  proxy ? "is-proxy" : "is-gap"
                }${relationClass(rowRelated(row))}${
                  lockedFocus?.kind === "row" &&
                  lockedFocus.row === row.row
                    ? " is-focus-locked"
                    : ""
                }`}
                data-business-group="downlink"
                style={
                  {
                    top: 104 + index * ROW_STEP,
                    left: 12,
                    width: CONTRACT_WIDTH - 24,
                    height: ROW_HEIGHT,
                    "--mapping-color": proxy
                      ? PROXY_COLORS[proxy]
                      : "#94a3b8",
                  } as CSSProperties
                }
                aria-label={`B${row.row}，${row.item.summary}；${
                  proxy
                    ? `工程代理 ${RECEIVE_PROXY_LABELS[proxy]}，${row.gap}`
                    : `未闭环，${row.gap}`
                }`}
                {...interactionProps({ kind: "row", row: row.row })}
              >
                <code>B{row.row}</code>
                <div>
                  <strong>{row.item.summary}</strong>
                  <small>
                    {proxy
                      ? RECEIVE_PROXY_LABELS[proxy]
                      : row.requirement}
                  </small>
                </div>
                <span>{proxy ? "工程代理" : "未闭环"}</span>
              </article>
            );
          })}
        </div>
      </div>

      <div className="metric-receive-legend">
        {RECEIVE_PROXY_IDS.map((proxy) => (
          <span key={proxy}>
            <i style={{ backgroundColor: PROXY_COLORS[proxy] }} />
            {RECEIVE_PROXY_LABELS[proxy]}
          </span>
        ))}
        <span>
          <i className="is-gap" />
          未闭环
        </span>
      </div>
      <p className="metric-receive-note">
        所有连线只表示“可生成相关诊断量”，不表示该合同行可计算或通过。
        点击输出或合同行可锁定关联链路，按 Esc 取消。
      </p>
    </div>
  );
}

export default function ReceiveProxyMappingView({
  mapping,
}: {
  mapping: ReceiveProxyMappingSnapshot;
}) {
  if (!mapping.valid) {
    return (
      <section className="metric-receive-detail">
        <div className="metric-mapping-error" role="alert">
          <AlertCircle size={18} />
          {mapping.problem ?? "虚拟 DRP 接收映射不可用。"}
        </div>
      </section>
    );
  }

  return (
    <section
      className="metric-receive-detail"
      aria-labelledby="metric-receive-heading"
    >
      <div className="metric-receive-heading">
        <div>
          <div className="metric-mapping-eyebrow">
            <Network size={15} aria-hidden="true" />
            接收端工程代理映射
          </div>
          <h3 id="metric-receive-heading">
            虚拟 DRP 双耳仿真 → B3–B11 下行接收指标
          </h3>
          <p>
            只把现有仿真字段映射到语义相近的合同行；RLR、TMOS / POLQA、
            dB[Pa] 底噪与频域容差判定仍明确标为未闭环。
          </p>
        </div>
        <div className="metric-receive-summary" aria-live="polite">
          <strong>{mapping.proxyCount}</strong>
          <span>工程代理</span>
          <i aria-hidden="true" />
          <strong>{mapping.unmappedCount}</strong>
          <span>未闭环</span>
        </div>
      </div>

      <p className="sr-only">
        当前 {mapping.mode} 的 B3–B11 共 9 项下行接收指标，
        {mapping.proxyCount} 项可关联虚拟 DRP 工程代理，
        {mapping.unmappedCount} 项因缺少正式算法、校准或配对采集而保持未闭环。
        该映射不产生正式合规结论。
      </p>

      <div
        className="metric-mapping-scroll metric-receive-scroll"
        role="region"
        aria-label={`${mapping.mode} 虚拟 DRP 输出到 B3–B11 下行接收指标关系图`}
        tabIndex={0}
      >
        <ReceiveProxyCanvas mapping={mapping} />
      </div>

      <div className="metric-receive-boundary" role="note">
        <strong>正式边界</strong>
        <span>
          synthetic_proxy=true · standard_compliance=false ·
          formal_hats_evidence=false
        </span>
      </div>
    </section>
  );
}
