import type { AudioRole } from "./audio.ts";

export type DiagramMode = "device" | "website";
type Point = readonly [number, number];
export type DiagramNode = {
  id: string; title: string; tag: string; lines: string[];
  x: number; y: number; width: number; height: number;
  role?: AudioRole; focal?: boolean; optional?: boolean;
};
type Label = { text: string; x: number; y: number; width: number; height: number };
export type DiagramEdge = {
  from: string; to: string; points: Point[];
  kind: "signal" | "reference" | "control" | "optional" | "focal";
  label?: Label;
};
export type FlowDiagram = {
  mode: DiagramMode; title: string; description: string;
  detail: "balanced" | "faithful";
  width: number; height: number; nodes: DiagramNode[]; edges: DiagramEdge[];
  zones: { title: string; x: number; y: number; width: number; height: number; labelX?: number; lines?: string[] }[];
  notes: string[];
};

// Device mode: Dual-rail HSAE VCP architecture (TX Uplink / RX Downlink) with spacious layout
const vcpDeviceNodes: DiagramNode[] = [
  // Left Endpoints (Network / Codec)
  { id: "reference", title: "干净参考语音", tag: "录音 01 · 纯净", lines: ["未混噪的标准语音源", "SI-SDR / STOI 真实参考"], role: "reference", x: 40, y: 40, width: 140, height: 80 },
  { id: "after", title: "上行输出 · TX Out", tag: "录音 03 · 必需", lines: ["经 HSAE VCP 净化输出", "送至车机蓝牙 / 蜂窝网络"], role: "after", x: 40, y: 150, width: 140, height: 92 },
  { id: "echo", title: "下行输入 · RX In", tag: "录音 04 · 参考", lines: ["远端网络通话音频", "下行算法处理与播放基准"], role: "echo", x: 40, y: 350, width: 140, height: 92 },

  // HSAE VCP Box - Top Track: TX / Uplink path (flowing right to left)
  { id: "gain", title: "上行增益 · AGC/DRC", tag: "AGC / DRC", lines: ["动态范围控制", "满幅保护与平稳"], optional: true, x: 230, y: 150, width: 125, height: 92 },
  { id: "eq", title: "上行均衡 · EQ", tag: "Equalizer", lines: ["车规频响平坦", "提升语音清晰度"], optional: true, x: 375, y: 150, width: 110, height: 92 },
  { id: "denoise", title: "上行降噪 · NR", tag: "Noise Reduction", lines: ["平稳与瞬态降噪", "保留语音谐波细节"], x: 505, y: 150, width: 125, height: 92 },
  { id: "aec", title: "回声消除 · AEC", tag: "Echo Canceller", lines: ["自适应消除扬声器回声", "双讲抑制与保护"], focal: true, x: 650, y: 150, width: 130, height: 92 },
  { id: "bf", title: "自适应麦克风阵列", tag: "Mic Array / BF", lines: ["同步多麦克风空间滤波", "定向主驾拾音输出"], x: 800, y: 150, width: 145, height: 92 },

  // HSAE VCP Box - Bottom Track: RX / Downlink path (flowing left to right)
  { id: "rx_nr", title: "下行降噪 · RX NR", tag: "Noise Reduction", lines: ["远端背景噪声净化", "提升车内可懂度"], x: 230, y: 350, width: 125, height: 92 },
  { id: "rx_eq", title: "下行均衡 · RX EQ", tag: "Equalizer", lines: ["扬声器频响校准", "车舱声学补偿"], optional: true, x: 375, y: 350, width: 110, height: 92 },
  { id: "rx_drc", title: "下行 DRC · 保护", tag: "AGC / DRC", lines: ["削波失真防护", "无削波纯净参考源"], focal: true, x: 505, y: 350, width: 125, height: 92 },

  // Right Acoustic Coupling & Cabin Environment
  { id: "voice", title: "近端语音声源", tag: "语音声源", lines: ["主驾/乘员真实语音", "目标方向拾音"], x: 1020, y: 40, width: 130, height: 80 },
  { id: "noise", title: "车舱背景噪声", tag: "环境噪声", lines: ["胎噪 / 风噪 / 空调", "弥散声场混入"], x: 1170, y: 40, width: 130, height: 80 },
  { id: "before", title: "麦克风采集 (TX In)", tag: "录音 02 · 原始", lines: ["麦克风阵列拾音", "语音 + 噪声 + 回声"], role: "before", x: 1020, y: 150, width: 140, height: 92 },
  { id: "speaker", title: "扬声器播放与回声", tag: "车舱扬声器", lines: ["功放电声还原播放", "空间声学耦合形成回声"], x: 1020, y: 350, width: 140, height: 92 },
];

const vcpDeviceEdges: DiagramEdge[] = [
  // TX path: before -> bf -> aec -> denoise -> eq -> gain -> after
  { from: "before", to: "bf", kind: "signal", points: [[1020, 196], [945, 196]] },
  { from: "bf", to: "aec", kind: "signal", points: [[800, 196], [780, 196]] },
  { from: "aec", to: "denoise", kind: "signal", points: [[650, 196], [630, 196]] },
  { from: "denoise", to: "eq", kind: "signal", points: [[505, 196], [485, 196]] },
  { from: "eq", to: "gain", kind: "signal", points: [[375, 196], [355, 196]] },
  { from: "gain", to: "after", kind: "signal", points: [[230, 196], [180, 196]] },

  // RX path: echo -> rx_nr -> rx_eq -> rx_drc -> speaker
  { from: "echo", to: "rx_nr", kind: "signal", points: [[180, 396], [230, 396]] },
  { from: "rx_nr", to: "rx_eq", kind: "signal", points: [[355, 396], [375, 396]] },
  { from: "rx_eq", to: "rx_drc", kind: "signal", points: [[485, 396], [505, 396]] },
  { from: "rx_drc", to: "speaker", kind: "signal", points: [[630, 396], [1020, 396]] },

  // Yellow Reference line: leaves rx_drc at x=630,y=396, rises into aec [715, 242]
  {
    from: "rx_drc", to: "aec", kind: "reference",
    points: [[630, 396], [715, 396], [715, 242]],
    label: { text: "AEC 纯净回声参考 (Ref)", x: 640, y: 310, width: 150, height: 24 }
  },

  // Acoustic coupling: speaker -> before (purple echo)
  {
    from: "speaker", to: "before", kind: "optional",
    points: [[1160, 396], [1200, 396], [1200, 196], [1160, 196]],
    label: { text: "声学回声 Echo", x: 1205, y: 285, width: 100, height: 24 }
  },

  // Acoustic sources: voice -> before, noise -> before, reference -> voice
  { from: "voice", to: "before", kind: "signal", points: [[1085, 120], [1085, 150]] },
  { from: "noise", to: "before", kind: "signal", points: [[1235, 120], [1235, 135], [1120, 135], [1120, 150]] },
  { from: "reference", to: "voice", kind: "reference", points: [[180, 80], [1020, 80]] },
];

const metricGroups = [
  { id: "levels", title: "电平与失真检查", lines: ["RMS / 峰值 / DC", "满幅比例与波形"] },
  { id: "fidelity", title: "SI-SDR 与 STOI", lines: ["语音保真与可懂度", "另报时延与相对增益"] },
  { id: "noise-metrics", title: "噪声与片段 SNR", lines: ["噪声功率与衰减", "平稳噪声条件下估计"] },
  { id: "echo-metrics", title: "回声衰减与 ERLE", lines: ["远端单讲功率比", "ERLE 单独确认测点"] },
  { id: "spectrum", title: "频谱与传递函数", lines: ["Welch PSD / H1", "相干度与有效频点"] },
];
const websiteNodes: DiagramNode[] = [
  { id: "upload", title: "上传录音与声道选择", tag: "网站 · 输入", lines: ["四路角色与声道选择", "ERLE 替换前后测点"], x: 56, y: 936, width: 168, height: 112 },
  { id: "decode", title: "解码与文件校验", tag: "网站 · 校验", lines: ["格式 / 声道 / 采样率", "保留文件原始信息"], x: 296, y: 936, width: 176, height: 112 },
  { id: "conditions", title: "片段与条件确认", tag: "网站 · 配置", lines: ["语音、时钟、电平确认", "ERLE 与噪声条件"], x: 536, y: 936, width: 176, height: 112 },
  { id: "align", title: "重采样与时间对齐", tag: "网站 · 按需对齐", lines: ["成对互相关与重采样", "单文件分析无需对齐"], x: 776, y: 936, width: 160, height: 112 },
  { id: "windows", title: "有效片段与共有范围", tag: "网站 · 有效片段", lines: ["按指标复核有效片段", "延迟补偿与共同参考"], x: 424, y: 1120, width: 152, height: 112 },
  ...metricGroups.map((group, index) => ({ ...group, tag: "分项指标计算", x: 56 + index * 184, y: 1352, width: 152, height: 112 })),
  { id: "compare", title: "处理前后对比", tag: "网站 · 对比", lines: ["同一参考范围与电平", "保真、降噪、回声改善"], x: 412, y: 1552, width: 176, height: 112 },
  { id: "report", title: "最终指标与报告", tag: "最终结果", lines: ["含有效性与不可计算原因", "CSV / JSON 导出"], focal: true, x: 776, y: 1552, width: 160, height: 112 },
];
const websiteEdges: DiagramEdge[] = [
  { from: "upload", to: "decode", kind: "signal", points: [[224, 992], [296, 992]] },
  { from: "decode", to: "conditions", kind: "signal", points: [[472, 992], [536, 992]] },
  { from: "conditions", to: "align", kind: "signal", points: [[712, 992], [776, 992]] },
  { from: "align", to: "windows", kind: "signal", points: [[856, 1048], [856, 1080], [500, 1080], [500, 1120]] },
  ...metricGroups.flatMap((group, index): DiagramEdge[] => {
    const center = 132 + index * 184;
    const sourcePort = 424 + 152 * (index + 1) / 6;
    const targetPort = 412 + 176 * (index + 1) / 6;
    const branchY = index === 0 || index === 4 ? 1320 : 1336;
    const joinY = index === 0 || index === 4 ? 1520 : 1488;
    return [
      { from: "windows", to: group.id, kind: "signal", points: index === 2 ? [[500, 1232], [500, 1352]] : [[sourcePort, 1232], [sourcePort, branchY], [center, branchY], [center, 1352]] },
      { from: group.id, to: "compare", kind: "signal", points: index === 2 ? [[500, 1464], [500, 1552]] : [[center, 1464], [center, joinY], [targetPort, joinY], [targetPort, 1552]] },
    ];
  }),
  { from: "compare", to: "report", kind: "focal", points: [[588, 1608], [776, 1608]] },
];

export const DIAGRAMS: Record<DiagramMode, FlowDiagram> = {
  device: {
    mode: "device", detail: "faithful", title: "Hsae VCP：完整语音增强处理链路",
    description: "HSAE VCP 车载语音增强处理链路：上行拾音与降噪回声消除、频响补偿与动态管理，下行远端降噪、均衡校准与功放 DRC 保护。",
    width: 1640, height: 620, nodes: vcpDeviceNodes,
    zones: [
      { title: "01  车舱物理声学环境与系统端点 · 7 节点", x: 20, y: 20, width: 1600, height: 580 },
      { title: "02  HSAE VCP 核心引擎 (TX / RX 双轨处理) · 8 节点", x: 210, y: 80, width: 770, height: 420 },
    ],
    edges: vcpDeviceEdges,
    notes: [],
  },
  website: {
    mode: "website", detail: "faithful", title: "指标计算细节：12 个关键节点",
    description: "实际录音完成解码、条件确认、对齐和有效片段检查后，分支计算电平、语音保真、噪声、回声、频谱，最后按有效条件比较并导出报告。",
    width: 1000, height: 920,
    zones: [
      { title: "01  车舱物理声学耦合环境 · 6 节点", x: 40, y: 56, width: 920, height: 376 },
      { title: "02  HSAE VCP 核心引擎 (TX / RX 双轨处理) · 8 节点", x: 40, y: 480, width: 920, height: 368 },
    ],
    nodes: websiteNodes.map((node) => ({ ...node, y: node.y - 840, focal: node.id === "report" || node.id === "fidelity" })),
    edges: websiteEdges.map((edge) => ({ ...edge, points: edge.points.map(([x, y]) => [x, y - 840] as const) })),
    notes: [],
  },
};

const XML_ENTITIES: Record<string, string> = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&apos;" };
function xml(value: unknown): string { return String(value).replace(/[&<>"']/g, (c) => XML_ENTITIES[c] ?? c); }
function number(value: number): string { return Number(value.toFixed(2)).toString(); }
function roundedPath(points: Point[]): string {
  if (points.length < 2) return "";
  let result = `M ${number(points[0][0])} ${number(points[0][1])}`;
  for (let i = 1; i < points.length - 1; i++) {
    const [ax, ay] = points[i - 1], [bx, by] = points[i], [cx, cy] = points[i + 1];
    const radius = Math.min(8, (Math.abs(bx - ax) + Math.abs(by - ay)) / 2, (Math.abs(cx - bx) + Math.abs(cy - by)) / 2);
    result += ` L ${number(bx - Math.sign(bx - ax) * radius)} ${number(by - Math.sign(by - ay) * radius)}`;
    result += ` Q ${number(bx)} ${number(by)} ${number(bx + Math.sign(cx - bx) * radius)} ${number(by + Math.sign(cy - by) * radius)}`;
  }
  const end = points.at(-1)!;
  return result + ` L ${number(end[0])} ${number(end[1])}`;
}

export const DIAGRAM_CSS = `
.df-svg{display:block;height:auto;font-family:Inter,"Noto Sans CJK SC","Noto Sans SC",-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif;color:var(--df-ink,#233248)}
.df-paper{fill:var(--df-paper,#f6f8fb)}.df-zone{fill:var(--df-surface,#fff);stroke:var(--df-rule,#dce3ec);stroke-width:1}
.df-vcp-zone{fill:#f1f5f9;stroke:#94a3b8;stroke-width:1.5}
.df-zone-label,.df-tag,.df-line-label,.df-legend,.df-note{font-size:12px;fill:var(--df-muted,#65758b)}
.df-title{font-size:13px;font-weight:600;fill:var(--df-ink,#233248)}.df-sub{font-size:11px;fill:var(--df-muted,#65758b)}
.df-node{fill:var(--df-surface,#fff);stroke:var(--df-rule-strong,#bdcada);stroke-width:1}.df-focal{fill:var(--df-tint,#edf3ff);stroke:var(--df-accent,#2563eb)}
.df-optional{stroke-dasharray:5 4}.df-selected{stroke:var(--df-ink,#233248);stroke-width:2}.df-tag-box{fill:var(--df-paper,#f6f8fb);stroke:var(--df-rule,#dce3ec);stroke-width:.7}
.df-edge{fill:none;stroke:var(--df-muted,#65758b);stroke-width:1.8}.df-edge-reference,.df-edge-focal,.df-edge-control{stroke:var(--df-accent,#2563eb)}.df-edge-optional{stroke-dasharray:5 4;stroke-width:1.4}.df-edge-control{stroke-dasharray:3 3;stroke-width:1.4}
.df-edge-ref-gold{stroke:#d97706!important;stroke-width:2.5!important}
.df-edge-echo-purple{stroke:#9333ea!important;stroke-width:2.5!important;stroke-dasharray:5 3!important}
.df-label-mask{fill:var(--df-paper,#f6f8fb)}.df-file{font-size:11px;fill:var(--df-accent,#2563eb);font-weight:600}.df-hairline{stroke:var(--df-rule,#dce3ec);stroke-width:1}
@media print{.df-svg{min-width:0;width:100%;height:auto}.df-scroll{overflow:visible!important}}
`;

export function renderDiagram(diagram: FlowDiagram, options: { prefix?: string; selected?: string; uploaded?: AudioRole[] } = {}) {
  const prefix = options.prefix ?? `nv200-${diagram.mode}`;
  const uploaded = new Set(options.uploaded ?? []);
  const markers = ["signal", "reference", "control", "optional", "focal"].map((kind) => {
    let fill = "var(--df-muted,#65758b)";
    if (kind === "reference") fill = "#d97706";
    else if (kind === "optional" && diagram.mode === "device") fill = "#9333ea";
    else if (["control", "focal"].includes(kind)) fill = "var(--df-accent,#2563eb)";
    return `<marker id="${xml(prefix)}-${kind}" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L8 3 L0 6Z" fill="${fill}"/></marker>`;
  }).join("");

  const zones = diagram.zones.map((zone, idx) => {
    const isVcpZone = diagram.mode === "device" && idx === 1;
    const zoneClass = isVcpZone ? "df-zone df-vcp-zone" : "df-zone";
    const titleText = isVcpZone ? "" : `<text x="${zone.labelX ?? zone.x + 16}" y="${zone.y + 20}" class="df-zone-label">${xml(zone.title)}</text>`;
    return `<rect x="${zone.x}" y="${zone.y}" width="${zone.width}" height="${zone.height}" rx="10" class="${zoneClass}"/>${titleText}${(zone.lines ?? []).map((line, index) => `<text x="${zone.x + 16}" y="${zone.y + 52 + index * 24}" class="df-note">${xml(line)}</text>`).join("")}`;
  }).join("");

  const vcpDecor = diagram.mode === "device" ? `
    <g class="df-vcp-decor">
      <!-- HSAE VCP Container Title -->
      <text x="595" y="116" font-size="18" font-weight="700" letter-spacing="3" fill="#1e293b" text-anchor="middle">HSAE VCP</text>
      <!-- Dual-Rail Track Subtitles -->
      <text x="230" y="138" font-size="12" font-weight="600" fill="#475569">TX / Uplink path</text>
      <text x="230" y="338" font-size="12" font-weight="600" fill="#475569">RX / Downlink path</text>
      
      <!-- Right Experience Panel (Completely separated, no line collisions) -->
      <g transform="translate(1330, 80)">
        <rect x="0" y="0" width="270" height="420" rx="10" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.2"/>
        <text x="18" y="32" font-size="15" font-weight="700" fill="#0f172a">用户可感知的体验结果</text>
        <text x="18" y="66" font-size="12" fill="#334155">• 更少回声、更低背景噪声</text>
        <text x="18" y="94" font-size="12" fill="#334155">• 双讲时更少吞字、截断或压制</text>
        <text x="18" y="122" font-size="12" fill="#334155">• 不同座位和说话距离下响度更稳定</text>
        <text x="18" y="150" font-size="12" fill="#334155">• 通过EQ补偿频响，提升语音清晰度</text>
        <text x="18" y="178" font-size="12" fill="#334155">• AEC、NR、EQ、AGC/DRC协同处理</text>
        <text x="18" y="206" font-size="12" fill="#334155">• 适配单麦和多麦配置</text>
        <text x="18" y="234" font-size="12" fill="#334155">• 支持离线仿真、参数迭代和Profile固化</text>
        
        <rect x="14" y="310" width="242" height="90" rx="8" fill="#ecfdf5" stroke="#a7f3d0" stroke-width="1.2"/>
        <text x="135" y="348" font-size="12" font-weight="600" fill="#065f46" text-anchor="middle">把算法能力转化为可调、可测、</text>
        <text x="135" y="372" font-size="12" font-weight="600" fill="#065f46" text-anchor="middle">可量产的车载通话体验</text>
      </g>

      <!-- Bottom Banner -->
      <g transform="translate(40, 530)">
        <rect x="0" y="0" width="1560" height="44" rx="8" fill="#fffbeb" stroke="#fde68a" stroke-width="1.2"/>
        <text x="780" y="27" font-size="13" font-weight="600" fill="#92400e" text-anchor="middle">支持单麦 / 多麦克风方案；可适配主流蓝牙音频 codec 与不同车载平台</text>
      </g>
    </g>
  ` : "";

  const edges = diagram.edges.map((edge) => {
    let edgeClass = `df-edge df-edge-${edge.kind}`;
    if (diagram.mode === "device" && edge.from === "rx_drc" && edge.to === "aec") {
      edgeClass += " df-edge-ref-gold";
    } else if (diagram.mode === "device" && edge.from === "speaker" && edge.to === "before") {
      edgeClass += " df-edge-echo-purple";
    }
    return `<path data-from="${xml(edge.from)}" data-to="${xml(edge.to)}" class="${edgeClass}" d="${roundedPath(edge.points)}" marker-end="url(#${xml(prefix)}-${edge.kind})"/>${edge.label ? `<rect x="${edge.label.x}" y="${edge.label.y}" width="${edge.label.width}" height="${edge.label.height}" rx="2" class="df-label-mask"/><text x="${edge.label.x + edge.label.width / 2}" y="${edge.label.y + 17}" class="df-line-label" text-anchor="middle">${xml(edge.label.text)}</text>` : ""}`;
  }).join("");

  const nodes = diagram.nodes.map((node) => {
    const classes = `df-node${node.focal ? " df-focal" : ""}${node.optional ? " df-optional" : ""}${options.selected === node.id ? " df-selected" : ""}`;
    const tagWidth = Math.min(node.width - 20, Math.ceil(node.tag.length * 11 / 4) * 4 + 10);
    const labelY = node.y + 40;
    return `<g data-node-id="${xml(node.id)}"><rect x="${node.x}" y="${node.y}" width="${node.width}" height="${node.height}" rx="8" class="${classes}"/><rect x="${node.x + 8}" y="${node.y + 6}" width="${tagWidth}" height="18" rx="3" class="df-tag-box"/><text x="${node.x + 12}" y="${node.y + 19}" class="df-tag">${xml(node.tag)}</text><text x="${node.x + 8}" y="${labelY}" class="df-title">${xml(node.title)}</text>${node.lines.map((line, index) => `<text x="${node.x + 8}" y="${labelY + 16 + index * 14}" class="df-sub">${xml(line)}</text>`).join("")}${node.role ? `<text x="${node.x + 8}" y="${node.y + node.height - 8}" class="df-file">${uploaded.has(node.role) ? "已选择录音" : `保存 ${xml(node.role)}`}</text>` : ""}</g>`;
  }).join("");

  const legendKinds = diagram.mode === "device"
    ? [["signal", "主信号流 (TX / RX)"], ["reference", "AEC 纯净播放参考"], ["optional", "声学回声物理耦合 (Echo)"]]
    : [["signal", "录音与计算数据"], ["focal", "指标报告输出"]];
  const legendGap = diagram.mode === "device" ? 280 : 264;
  const legend = legendKinds.map(([kind, title], index) => {
    let lineClass = `df-edge df-edge-${kind}`;
    if (diagram.mode === "device" && kind === "reference") lineClass += " df-edge-ref-gold";
    if (diagram.mode === "device" && kind === "optional") lineClass += " df-edge-echo-purple";
    return `<path d="M${56 + index * legendGap} ${diagram.height - 24} H${88 + index * legendGap}" class="${lineClass}"/><text x="${100 + index * legendGap}" y="${diagram.height - 20}" class="df-legend">${xml(title)}</text>`;
  }).join("");

  return `<svg xmlns="http://www.w3.org/2000/svg" class="df-svg" width="${diagram.width}" height="${diagram.height}" viewBox="0 0 ${diagram.width} ${diagram.height}" style="min-width:${diagram.width}px;" role="img" aria-labelledby="${xml(prefix)}-title ${xml(prefix)}-desc"><title id="${xml(prefix)}-title">${xml(diagram.title)}</title><desc id="${xml(prefix)}-desc">${xml(diagram.description)}</desc><style>${DIAGRAM_CSS}</style><defs>${markers}</defs><rect x="0" y="0" width="${diagram.width}" height="${diagram.height}" class="df-paper"/>${zones}${vcpDecor}${edges}${nodes}${diagram.notes.map((note, index) => `<text x="56" y="${428 + index * 24}" class="df-note">${xml(note)}</text>`).join("")}<path d="M40 ${diagram.height - 40} H${diagram.width - 40}" class="df-hairline"/>${legend}</svg>`;
}

export function validateDiagram(diagram: FlowDiagram): string[] {
  const issues: string[] = [];
  const faithful = diagram.detail === "faithful";
  if (diagram.nodes.length > (faithful ? 24 : 9) || diagram.edges.length > (faithful ? 32 : 12)) issues.push("diagram exceeds complexity budget");
  if (faithful && diagram.nodes.length > 9) {
    if (diagram.zones.length < 2 || diagram.zones.length > 4) issues.push("detailed diagram requires 2–4 zones");
    for (const node of diagram.nodes) {
      if (!diagram.zones.some((zone) => node.x >= zone.x && node.y >= zone.y && node.x + node.width <= zone.x + zone.width && node.y + node.height <= zone.y + zone.height)) issues.push(`${node.id}: detailed node outside zones`);
    }
  }
  for (const edge of diagram.edges) {
    if (!diagram.nodes.some((n) => n.id === edge.from) || !diagram.nodes.some((n) => n.id === edge.to)) issues.push("edge has missing endpoint");
    for (let i = 1; i < edge.points.length; i++) {
      const [a, b] = [edge.points[i - 1], edge.points[i]];
      if (a[0] !== b[0] && a[1] !== b[1]) issues.push(`${edge.from}→${edge.to}: diagonal segment`);
      for (const node of diagram.nodes.filter((n) => n.id !== edge.from && n.id !== edge.to)) {
        const crossing = a[0] === b[0]
          ? a[0] > node.x && a[0] < node.x + node.width && Math.min(a[1], b[1]) < node.y + node.height && Math.max(a[1], b[1]) > node.y
          : a[1] > node.y && a[1] < node.y + node.height && Math.min(a[0], b[0]) < node.x + node.width && Math.max(a[0], b[0]) > node.x;
        if (crossing) issues.push(`${edge.from}→${edge.to}: crosses ${node.id}`);
      }
    }
    if (edge.label) {
      const label = edge.label;
      for (const node of diagram.nodes) {
        if (Math.min(label.x + label.width, node.x + node.width) > Math.max(label.x, node.x) && Math.min(label.y + label.height, node.y + node.height) > Math.max(label.y, node.y)) issues.push(`${edge.from}→${edge.to}: label overlaps ${node.id}`);
      }
    }
  }
  return issues;
}
