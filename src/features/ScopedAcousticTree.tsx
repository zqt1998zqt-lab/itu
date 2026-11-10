import { useId, useState } from "react";
import { ArrowRight, GitBranch, Layers3 } from "lucide-react";
import { chainPresets, chainStages } from "./enhancement-chain-model";
import type { FiveSignalMode } from "./five-signal-flow-model";
import {
  BACKGROUND_NOISE_SCENES,
  BACKGROUND_NOISE_SCENE_LABELS,
  matchingScopedSnapshot, scopedItemCount, scopedMetricUnit, scopedValue,
  SCOPED_STATUS_LABELS, type ScopedCatalog, type ScopedDataset,
  type ScopedSnapshot, type ScopedTestDefinition,
} from "./scoped-acoustic-model";
import "./scoped-acoustic-tree.css";

type Props = {
  definition: ScopedTestDefinition;
  mode: FiveSignalMode;
  algorithmClusterId: string;
  datasetId: string;
  dataset: ScopedDataset | null;
  catalog: ScopedCatalog | null;
  catalogLoading: boolean;
  catalogError: string;
  snapshot: ScopedSnapshot | null;
  syncFailed: boolean;
  isPreviousSnapshot: boolean;
};

const WIDTH = 1976;
const ANALYSIS_X = 824;
const ANALYSIS_RIGHT = 1164;
const LEAF_X = 1288;
const LEAF_WIDTH = 664;
const LEAF_HEIGHT = 112;
const STEP = 124;
const BACKGROUND_CASE_LAYOUT = { speech: { top: 110, height: 130 }, cng: { top: 254, height: 130 } };
const VOLUME: Record<string, string> = { nominal: "标称音量", maximum: "最大音量", recommended: "推荐音量" };
const SCENES = ["怠速 / 静止", "60 km/h 市区", "120 km/h 高速", "空调大风档"];
const OUTPUT_KIND_LABELS = {
  tx: "发送处理输出（TX）",
  rx: "接收处理输出（RX）",
} as const;

const TREE_COPY = {
  basic_downlink: {
    path: "远端标准信号 → VCP 接收处理（RX）→ 虚拟双耳 → 接收测项",
    input: "处理输入：远端标准信号\n测量参考：远端激励、双耳配置与合同电平",
    analysis: "虚拟 DRP 双耳仿真",
    evidence: "接收处理输出（RX）+ 已知远端激励",
    guide: "所选 VCP profile 加载接收处理（RX）内部配置，发送处理（TX）分级选项不覆盖。RLR 线性度需 nominal / −5 / +5 dB 三档独立运行；左右耳均为软件代理。",
  },
  echo: {
    path: "麦克风混合输入中的回声 + 远端播放信号 / 回声参考 → 远端单讲区段 → 耦合损耗测项",
    input: "处理输入：麦克风混合输入中的回声、远端播放信号 / 回声参考\n测量参考：干净近端参考活动与远端激励",
    analysis: "原始回声输入测量",
    evidence: "远端播放信号（far_end）+ 麦克风混合输入（microphone）+ 干净近端参考（reference）",
    guide: "实线为 VCP 执行支路，虚线为原始输入测量支路。合同公式读取麦克风混合输入中的回声；当前执行器仍需生成发送处理输出（TX），供补充诊断。",
  },
  double_talk: {
    path: "近远端同时讲话 → VCP 发送处理（TX）→ 双讲区段识别 → 功率比测项",
    input: "处理输入：麦克风混合输入、远端播放信号 / 回声参考\n测量参考：干净近端参考与同步远端激励",
    analysis: "双讲区段与功率比",
    evidence: "发送处理输出（TX）+ 麦克风混合输入 + 干净近端参考 / 远端播放信号 / 回声参考",
    guide: "标称与最大音量分别使用对应条件组。仅在有效双讲区段计算输出 / 输入功率比，不以此冒充 A1/A2 分类。",
  },
  background_noise: {
    path: "四套场景输入 → 分条件组运行 VCP 发送处理（TX）→ 场景计算分支 → 20 项测项",
    input: "四场景分别提供：麦克风混合输入、远端播放信号 / 回声参考、干净近端参考与独立噪声输入\n各条件组独立运行发送处理（TX），保留各自输入输出身份",
    analysis: "四场景计算分支",
    evidence: "各测项绑定组的发送处理输出（TX）；CNG 另取同组独立噪声输入（noise）",
    guide: "每场景 3 项 DNSMOS 质量预测 + 2 项 CNG 诊断。测量型数据集将语音与 CNG 分开运行；自定义数据集按各测项绑定的条件组取证。",
  },
};

function branchPath(startX: number, centerY: number, laneX: number, endX: number, targetY: number) {
  if (Math.abs(centerY - targetY) < 16) return `M ${startX} ${centerY} H ${endX}`;
  const sign = targetY < centerY ? -1 : 1;
  return `M ${startX} ${centerY} H ${laneX - 8} Q ${laneX} ${centerY} ${laneX} ${centerY + 8 * sign} V ${targetY - 8 * sign} Q ${laneX} ${targetY} ${laneX + 8} ${targetY} H ${endX}`;
}

export default function ScopedAcousticTree({
  definition, mode, algorithmClusterId, datasetId, dataset, catalog, catalogLoading,
  catalogError, snapshot, syncFailed, isPreviousSnapshot,
}: Props) {
  const headingId = useId();
  const descriptionId = useId();
  const [hovered, setHovered] = useState<string | null>(null);
  const [locked, setLocked] = useState<string | null>(null);
  const count = scopedItemCount(definition.scope, mode, definition.scene);
  const background = definition.scope === "background_noise";
  const sceneIndex = definition.scene
    ? BACKGROUND_NOISE_SCENES.indexOf(definition.scene)
    : null;
  const sceneLabel = definition.scene
    ? BACKGROUND_NOISE_SCENE_LABELS[definition.scene]
    : null;
  const echo = definition.scope === "echo";
  const downlink = definition.scope === "basic_downlink";
  const copy = TREE_COPY[definition.scope];
  const items = catalog?.mode === mode && catalog.test_scope === definition.scope ? catalog.items : [];
  const hasLinearity = downlink && (items.length ? items.some(item => item.family === "rlr_linearity") : !mode.startsWith("SWB-"));
  const hasOutOfBand = downlink && (items.length ? items.some(item => item.family === "out_of_band") : !mode.startsWith("SWB-"));
  const analysisHeight = background || downlink ? 448 : 416;
  const contentHeight = STEP * (count - 1) + LEAF_HEIGHT;
  const height = background
    ? contentHeight + (sceneIndex === null ? 3 * 48 : 0) + 168
    : Math.max(896, contentHeight + 176);
  const originY = background ? 120 : (height - contentHeight) / 2;
  const top = (index: number) =>
    originY
    + STEP * index
    + (background && sceneIndex === null ? 48 * Math.floor(index / 5) : 0);
  const centerY = background ? 424 : height / 2;
  const groups = Array.from({ length: background && sceneIndex === null ? 4 : 1 }, (_, groupIndex) => {
    const resolvedScene = sceneIndex ?? groupIndex;
    const indices = Array.from(
      { length: background ? 5 : count },
      (_, i) => background && sceneIndex === null ? groupIndex * 5 + i : i,
    );
    const center = background ? (top(indices[0]) + top(indices.at(-1)!) + LEAF_HEIGHT) / 2 : centerY;
    const branches = background ? [
      { id: "speech", center: center - analysisHeight / 2 + BACKGROUND_CASE_LAYOUT.speech.top + BACKGROUND_CASE_LAYOUT.speech.height / 2, indices: indices.filter((index, localIndex) => items[index] ? ["g_mos", "n_mos", "s_mos"].includes(items[index].family) : localIndex < 3) },
      { id: "cng", center: center - analysisHeight / 2 + BACKGROUND_CASE_LAYOUT.cng.top + BACKGROUND_CASE_LAYOUT.cng.height / 2, indices: indices.filter((index, localIndex) => items[index] ? ["cng_level", "cng_spectrum"].includes(items[index].family) : localIndex >= 3) },
    ] : [{ id: "measurement", center, indices }];
    return { id: `analysis-${resolvedScene}`, indices, center, scene: resolvedScene, branches };
  });
  const preset = chainPresets.find(item => item.id === algorithmClusterId);
  const supported = Boolean(preset?.profile.startsWith("vcp"));
  const report = supported
    ? matchingScopedSnapshot(
        snapshot,
        mode,
        algorithmClusterId,
        datasetId,
        definition.scene,
      )
    : null;
  const results = new Map(report?.items.map(item => [item.item_id, item]));
  const active = items.some(item => item.id === locked) ? locked : items.some(item => item.id === hovered) ? hovered : null;
  const activeIndex = items.findIndex(item => item.id === active);
  const activeGroup = groups.find(group => group.indices.includes(activeIndex));
  const referencePath = `M 288 ${centerY} H 320 Q 328 ${centerY} 328 ${centerY - 8} V 92 Q 328 84 336 84 H 772 Q 780 84 780 92 V ${centerY - 8} Q 780 ${centerY} 788 ${centerY} H ${ANALYSIS_X}`;
  const status = !supported ? "待选择 VCP 算法簇" : !report ? "当前配置待执行"
    : report.status === "completed" ? "计算完成" : report.status === "failed" ? "执行失败"
      : report.status === "cancelled" ? "已取消" : report.status === "queued" ? "等待执行" : "正在计算";
  const notice = !supported ? "当前算法不属于 VCP，请在下方切换算法簇后执行。"
    : snapshot && !report ? "模式、算法或数据集已变更。图中仅显示当前配置，下方保留原报告。"
      : syncFailed ? "状态同步失败。图中保留上次成功快照，执行状态待重新确认。"
        : isPreviousSnapshot && report ? "当前请求尚未取得新结果，图中展示上次成功快照。" : "";

  return <figure className="scoped-tree" aria-labelledby={headingId} aria-describedby={descriptionId}>
    <figcaption className="scoped-tree-header">
      <div>
        <h3 id={headingId}><GitBranch size={17} aria-hidden="true" />{definition.title}树</h3>
        <p id={descriptionId}>{sceneLabel ? `${sceneLabel}单场景：` : ""}{copy.path}。悬停或键盘聚焦查看分支，点击锁定或取消。</p>
      </div>
      <div className="scoped-tree-meta"><code>{mode}</code><span>{isPreviousSnapshot && report ? "上次快照 · " : ""}{status}{report ? ` · ${report.summary.computed}/${count} 项有数值` : ""}</span></div>
    </figcaption>
    {notice ? <p className="scoped-tree-notice" role="status">{notice}</p> : null}
    <div className="scoped-tree-scroll" role="region" aria-label={`${definition.title}树状图，可横向滚动`} tabIndex={0}>
      <div className="scoped-tree-canvas" style={{ width: WIDTH, height }}>
        <div className="scoped-tree-column" style={{ left: 24 }}>01 <span>数据与工况</span></div>
        <div className="scoped-tree-column" style={{ left: 360 }}>02 <span>VCP {OUTPUT_KIND_LABELS[definition.outputKind]} {echo ? "执行支路" : "处理链"}</span></div>
        <div className="scoped-tree-column" style={{ left: ANALYSIS_X }}>03 <span>{copy.analysis}</span></div>
        <div className="scoped-tree-column" style={{ left: LEAF_X }}>04 <span>计算方法 → {count} 项合同</span></div>
        {echo ? <span className="scoped-tree-reference-label">原始信号直接测量 · 发送处理输出（TX）不进入此公式</span> : null}
        <svg className="scoped-tree-edges" width={WIDTH} height={height} viewBox={`0 0 ${WIDTH} ${height}`} aria-hidden="true" focusable="false">
          <path d={`M 288 ${centerY} H 360`} className={`scoped-tree-main-edge${echo && active ? " is-muted" : ""}`} />
          {echo ? <path d={referencePath} className="scoped-tree-reference-edge" /> : null}
          {groups.map(group => <g key={group.id} className={active && activeGroup !== group ? "is-muted" : ""}>
            {!echo ? <path d={branchPath(724, centerY, 780, ANALYSIS_X, group.center)} className="scoped-tree-main-edge" data-edge="vcp-analysis" /> : null}
            {group.branches.filter(branch => branch.indices.length).map(branch => <g key={branch.id} data-branch={branch.id} className={active && !branch.indices.includes(activeIndex) ? "is-muted" : ""}>
            <path d={`M ${ANALYSIS_RIGHT} ${branch.center} H 1224`} className="scoped-tree-main-edge" />
            <path d={`M 1224 ${Math.min(branch.center, top(branch.indices[0]) + LEAF_HEIGHT / 2)} V ${Math.max(branch.center, top(branch.indices.at(-1)!) + LEAF_HEIGHT / 2)}`} className="scoped-tree-spine" />
            {branch.indices.map(index => {
              const y = top(index) + LEAF_HEIGHT / 2;
              const id = items[index]?.id;
              return <g key={index} className={active && id !== active ? "is-muted" : ""}>
                <path d={`M 1224 ${y} H ${LEAF_X}`} className="scoped-tree-branch" />
                <circle cx={1224} cy={y} r={3} className="scoped-tree-junction" />
                {id === active ? <path d={branchPath(ANALYSIS_RIGHT, branch.center, 1224, LEAF_X, y)} className="scoped-tree-focus-route" /> : null}
              </g>;
            })}
            </g>)}
            {activeGroup === group ? <path d={echo ? referencePath : `M 288 ${centerY} H 360 ${branchPath(724, centerY, 780, ANALYSIS_X, group.center)}`} className={`scoped-tree-focus-route${echo ? " is-reference" : ""}`} /> : null}
          </g>)}
        </svg>
        <div className="scoped-tree-source" style={{ top: centerY - 144 }}>
        <section className="scoped-tree-node scoped-tree-input" aria-label="测试数据集与输入条件">
          <span className="scoped-tree-node-label">测试数据集</span>
          <strong className="scoped-tree-dataset-name" title={dataset?.name}>{dataset?.name ?? "等待选择测试集"}</strong>
          <span className="scoped-tree-secondary">{dataset ? `${dataset.case_count} 个可用条件组 · 提取当前子范围` : "在下方选择或导入完整测试集"}</span>
          <dl><div><dt>业务模式</dt><dd>{mode}</dd></div><div><dt>本次范围</dt><dd>{count} 个合同项</dd></div></dl>
          <p className="scoped-tree-inputs">{copy.input}</p>
          <span className="scoped-tree-secondary">模式、场景、音量、时间窗与报告 ID 逐项绑定</span>
        </section>
        <section className="scoped-tree-node scoped-tree-guide" aria-label="计算证据说明">
          <h4>证据与结果</h4>
          <p>{downlink && !hasLinearity ? "所选 VCP profile 加载接收处理（RX）内部配置，发送处理（TX）分级选项不覆盖。当前目录不含 RLR 线性度和带外测项；左右耳均为软件代理。" : copy.guide}</p>
          <p>无数值与未执行均显示文字，不以 0 分冒充测量结果。</p>
          <a href="#alignment">前往完整 336 项验收 <ArrowRight size={13} aria-hidden="true" /></a>
        </section>
        </div>
        <section className={`scoped-tree-node scoped-tree-processing${downlink ? " is-receive" : ""}${echo && active ? " is-muted" : ""}`} style={{ top: centerY - (downlink ? 230 : 284) }} aria-label={supported ? `VCP ${OUTPUT_KIND_LABELS[definition.outputKind]} 配置的处理链` : "算法簇不支持此测试"}>
          <span className="scoped-tree-node-label"><Layers3 size={14} aria-hidden="true" />{supported ? `VCP ${OUTPUT_KIND_LABELS[definition.outputKind]}` : "待选择 VCP"}</span>
          <h4>{supported ? downlink ? "下行接收处理" : "上行语音处理" : "当前算法不可执行"}</h4>
          <p className="scoped-tree-preset" title={preset?.label}>{preset?.label ?? algorithmClusterId}</p>
          {supported ? <>
            <ol className="scoped-tree-stages" aria-label={downlink ? "下行接收配置的阶段" : "所选算法簇的阶段配置"}>
              {downlink ? [
                { label: "NR 接收降噪", detail: "接收处理（RX）内部配置 · 抑制上限 9 dB" },
                { label: "EQ 配置槽 · 当前旁路", detail: "默认平直，不沿用上行 EQ 选项", bypassed: true },
                { label: "DRC · 内嵌软限幅", detail: "4:1 压缩 · −0.5 dBFS 输出约束" },
                { label: "扬声器数字输出", detail: "接收处理输出（RX）WAV → 后续虚拟双耳" },
              ].map((stage, index) => <li key={stage.label} className={stage.bypassed ? "is-bypassed" : ""}><span>{String(index + 1).padStart(2, "0")}</span><div><strong>{stage.label}</strong><p>{stage.detail}</p></div></li>) : chainStages.map((stage, index) => {
                const option = stage.options.find(entry => entry.id === preset?.selection[stage.id]);
                return <li key={stage.id} className={option?.id === "off" ? "is-bypassed" : ""}><span>{String(index + 1).padStart(2, "0")}</span><div><strong>{stage.label}</strong><p title={option?.note}>{option?.label ?? "配置不可用"}</p></div></li>;
              })}
            </ol>
            <p className="scoped-tree-chain-note">{downlink ? "所选 profile 加载接收处理（RX）内部配置；发送处理（TX）的 BF / AEC / AGC 选项不进入接收链，实际执行以报告为准。" : "展示所选算法簇的配置；单麦 BF 自动旁路，实际执行以条件组报告为准。"}{echo ? "当前执行器仍生成发送处理输出（TX）；右侧公式读取原始输入。" : background ? "每个绑定条件组各自运行一次，不跨场景共用输出。" : ""}</p>
            <div className="scoped-tree-output"><span>{echo ? "发送处理输出（TX）· 仅供补充诊断" : `${OUTPUT_KIND_LABELS[definition.outputKind]} + 所需参考`}</span>{!echo ? <ArrowRight size={16} aria-hidden="true" /> : null}</div>
          </> : <p className="scoped-tree-chain-note">请在下方选择 VCP 算法簇，以显示执行步骤并启用测试。</p>}
        </section>
        {groups.map(group => {
          const groupResults = group.indices.map(index => items[index] ? results.get(items[index].id) : undefined);
          const computed = groupResults.filter(result => result?.value != null).length;
          const boundCases = (branchId: string) => [...new Set(group.branches.find(branch => branch.id === branchId)?.indices.flatMap(index => {
            const caseId = items[index] ? results.get(items[index].id)?.evidence_binding?.case_id : null;
            return caseId ? [caseId] : [];
          }))];
          const speechCases = boundCases("speech");
          const cngCases = boundCases("cng");
          return <section key={group.id} className={`scoped-tree-node scoped-tree-analysis${background ? " is-background" : ""}${active && activeGroup !== group ? " is-muted" : activeGroup === group ? " is-focused" : ""}`} style={{ top: group.center - analysisHeight / 2, height: analysisHeight }} aria-label={background ? `Sce${group.scene + 1} ${SCENES[group.scene]}的计算分支` : copy.analysis}>
            <span className="scoped-tree-node-label">{background ? `Sce${group.scene + 1} · ${SCENES[group.scene]}` : "测量分支"}</span>
            <h4>{background ? "质量预测与舒适噪声" : copy.analysis}</h4>
            <p className="scoped-tree-analysis-input">{copy.evidence}</p>
            {downlink ? <>
              <div className="scoped-tree-operation"><strong>声学模型</strong><span>按测试集选用 head 近似或成对 IR 卷积</span></div>
              <div className="scoped-tree-ears"><div><strong>左耳 L</strong><span>独立耳信号</span></div><div><strong>右耳 R</strong><span>独立耳信号</span></div></div>
              <div className="scoped-tree-operation"><strong>分项测量</strong><span>双耳电平 · H1 频响 · 噪声窗{hasOutOfBand ? " · 带外能量" : ""}</span></div>
              <div className="scoped-tree-operation"><strong>{hasLinearity ? "质量 / 线性度" : "语音质量预测"}</strong><span>最差耳 OVRL{hasLinearity ? "；三档独立接收处理输出（RX）比较" : "；当前目录不含 RLR 线性度"}</span></div>
            </> : echo ? <>
              <div className="scoped-tree-operation"><strong>活动与对齐</strong><span>远端活跃、近端参考不活跃；麦克风对齐近端参考</span></div>
              <div className="scoped-tree-operation"><strong>输入功率提取</strong><span>P_far 与 P_microphone_echo</span></div>
              <div className="scoped-tree-formula">10 log₁₀(P_far / P_mic_echo)</div>
              <p className="scoped-tree-analysis-note">麦克风混合输入回声耦合损耗代理；未做 TCLw 计权，也不是 AEC 后残余或 ERLE。</p>
            </> : background ? <>
              <div className={`scoped-tree-case is-speech${activeGroup === group && items[activeIndex] && !["g_mos", "n_mos", "s_mos"].includes(items[activeIndex].family) ? " is-muted" : ""}`} style={BACKGROUND_CASE_LAYOUT.speech} title={speechCases.length ? `绑定条件组：${speechCases.join("、")}` : "各语音测项按绑定条件组取证"}>
                <strong>语音类条件组 → 完整发送处理输出（TX）≥ 9.01 s</strong><span>DNSMOS P.835 · 3 项 · {speechCases.length ? `${speechCases.length} 个绑定组` : "按测项绑定"}</span>
                <div className="scoped-tree-mos-outputs"><span>OVRL → G-MOS</span><span>BAK → N-MOS</span><span>SIG → S-MOS</span></div>
              </div>
              <div className={`scoped-tree-case is-cng${activeGroup === group && items[activeIndex] && !["cng_level", "cng_spectrum"].includes(items[activeIndex].family) ? " is-muted" : ""}`} style={BACKGROUND_CASE_LAYOUT.cng} title={cngCases.length ? `绑定条件组：${cngCases.join("、")}` : "各 CNG 测项按绑定条件组取证"}>
                <strong>CNG 类条件组 → 本组独立噪声输入（noise）+ 发送处理输出（TX）</strong><span>数字 RMS 电平差 · 1 项</span><span>归一化 PSD 的 JS 形状距离 · 1 项</span>
                <small>{cngCases.length ? `${cngCases.length} 个绑定组；各取本组证据。` : "按测项绑定取证，语音 / CNG 可分组运行。"}</small>
              </div>
            </> : <>
              <div className="scoped-tree-operation"><strong>近远端活动掩模</strong><span>干净近端参考 ∩ 远端播放信号 / 回声参考 → 有效双讲区段</span></div>
              <div className="scoped-tree-operation"><strong>对齐与有效覆盖</strong><span>麦克风混合输入与发送处理输出（TX）对齐参考；有效双讲不少于约 50 ms</span></div>
              <div className="scoped-tree-formula">100 × P_TX / P_microphone</div>
              <p className="scoped-tree-analysis-note">区段功率比裁剪到 0–150%；不执行 A1/A2 分类，也不是逐帧透传率。</p>
            </>}
            <div className="scoped-tree-analysis-footer"><span>工程代理 · 正式合规未测</span><strong>{report ? `${computed}/${group.indices.length} 项有值` : "尚未执行"}</strong></div>
          </section>;
        })}
        {background ? groups.map(group => <div className="scoped-tree-scene" key={group.id} style={{ left: LEAF_X, top: top(group.indices[0]) - 25 }}><span>Sce{group.scene + 1} · {SCENES[group.scene]}</span><small>3 项质量预测 + 2 项舒适噪声</small></div>) : null}
        <ol className="scoped-tree-leaves" aria-label={`${count} 项${definition.title}计算模型与合同测项`}>
          {Array.from({ length: count }, (_, index) => {
            const item = items[index];
            const result = item ? results.get(item.id) : undefined;
            const method = item ? definition.methods[item.family] : undefined;
            const id = item?.id ?? `pending-${index}`;
            const binding = result?.evidence_binding;
            const window = binding?.window_ms ?? Number(item?.test_name.match(/(50|100)\s*ms/i)?.[1] ?? 0);
            const volume = binding?.volume ? VOLUME[String(binding.volume)] : /max(?:imum)?/i.test(item?.test_name ?? "") ? "最大音量" : /recommended/i.test(item?.test_name ?? "") ? "推荐音量" : "标称音量";
            const value = result?.value != null ? `${scopedValue(result.value)} ${scopedMetricUnit(result.metric)}`.trim() : result ? "无数值" : "—";
            return <li key={id} style={{ left: LEAF_X, top: top(index), width: LEAF_WIDTH, height: LEAF_HEIGHT }}>
              <button type="button" className={`scoped-tree-leaf${active === id ? " is-focused" : active ? " is-muted" : ""}`}
                aria-pressed={locked === id} disabled={!item}
                title={item ? `${item.id} · ${item.test_name}\n${method?.detail ?? ""}${binding?.case_id ? `\n条件组：${binding.case_id}` : ""}${binding?.report_id ? `\n报告：${binding.report_id}` : ""}${result?.reason ? `\n${result.reason}` : ""}` : "等待目录"}
                onMouseEnter={() => setHovered(id)} onMouseLeave={() => setHovered(null)}
                onFocus={() => setHovered(id)} onBlur={() => setHovered(null)} onClick={() => setLocked(locked === id ? null : id)}>
                <div className="scoped-tree-row-id"><span>{String(index + 1).padStart(2, "0")}</span><code>{item?.source_cell ?? "—"}</code></div>
                <div className="scoped-tree-method"><strong>{method?.label ?? "等待测项目录"}{window ? ` · ${window} ms` : ""}</strong><span>{method?.detail ?? (catalogError ? "目录读取失败，请在下方重试" : catalogLoading ? `正在读取 ${count} 项定义` : "目录不可用")}</span><small>{item ? `${volume}${window ? ` · ${window} ms` : " · 合同分析范围"}` : "工况待确认"}{binding?.case_id ? ` · 组 ${binding.case_id.slice(0, 10)}…` : ""}{binding?.report_id ? ` · 报告 ${binding.report_id.slice(0, 12)}…` : ""}</small></div>
                <div className="scoped-tree-leaf-result"><span className={`scoped-tree-status is-${result?.status ?? "unrun"}`}>{result ? SCOPED_STATUS_LABELS[result.status] : item ? "尚未执行" : "等待目录"}</span><strong>{value}</strong></div>
              </button>
            </li>;
          })}
        </ol>
      </div>
    </div>
    <div className="scoped-tree-footer"><span>{background ? sceneLabel ? `${sceneLabel} · 3 项质量 + 2 项 CNG，逐项绑定条件组证据` : "四场景 ×（3 项质量 + 2 项 CNG），逐项绑定条件组证据" : "按合同工况分支计算，保留逐项证据"}</span><span>工程代理结果 · 正式 ITU-T 合规未测</span></div>
  </figure>;
}
