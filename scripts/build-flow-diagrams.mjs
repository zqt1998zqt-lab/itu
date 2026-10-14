import { mkdir, writeFile } from "node:fs/promises";
import { DIAGRAMS, renderDiagram, validateDiagram } from "../src/domain/flow-diagram.ts";

const output = new URL("../public/diagrams/", import.meta.url);
await mkdir(output, { recursive: true });
for (const diagram of Object.values(DIAGRAMS)) {
  const issues = validateDiagram(diagram);
  if (issues.length) throw new Error(issues.join("\n"));
}
const variants = [
  { suffix: "", name: "ITU Audio Lab · 音频流程图", skin: "" },
  { suffix: "-dark", name: "ITU Audio Lab · 音频流程图（深色）", skin: "--df-paper:#182233;--df-surface:#202e42;--df-ink:#e9eef7;--df-muted:#b3c0d4;--df-rule:#35465f;--df-rule-strong:#627894;--df-accent:#8bb4ff;--df-tint:#253d60;" },
  { suffix: "-full", name: "NV200 · 从实际录音到计算指标", skin: "", editorial: true },
];
for (const variant of variants) {
  const figures = Object.values(DIAGRAMS).map((diagram) => `${diagram.mode === "website" ? '<p class="df-handoff">↓ 四路实际录音进入指标计算：干净参考、处理前、处理后、播放参考。单独测 ERLE 时，滤波器输入 / 输出替换前后测点。</p>' : ''}<section><h2>${diagram.title}</h2><p>${diagram.description}</p><div class="df-scroll">${renderDiagram(diagram, { prefix: `nv200-${diagram.mode}${variant.suffix || "-light"}` })}</div></section>`).join("");
  const explanation = variant.editorial ? `<section><h2>录音如何支撑指标</h2><p>处理后音频可用于电平、波形和频谱分析。加入同一段干净语音参考后，才能比较时延与语音保真；可懂度计算还需真实语音及有效片段。</p><p>评估降噪效果需处理前后录音及纯噪声片段。评估回声需同步的播放参考、远端单讲片段和共同电平基准。最终输出的总衰减包含整条链路的影响；隔离 AEC 滤波器效果时，需改用滤波器输入与 BF 等后级处理前的输出留样。</p><p>BF 需要同步多麦克风与阵列几何；单声道输出不能直接证明指向性或阵列增益。DTD 通过控制支路保护 AEC 自适应更新，检测率与误判率另需逐帧状态和标注。网站不执行 BF 或 DTD，不自动计算这些专门指标。</p><p>示意图说明录音与测量方法，不能证明 NV200 的实际处理效果。设备结果需由实际录音与对应测试条件确认。</p></section>` : "";
  const html = `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${variant.name}</title><style>:root{${variant.skin}}*{box-sizing:border-box}body{margin:0;padding:40px 24px;background:var(--df-paper,#f6f8fb);color:var(--df-ink,#233248);font-family:Inter,"Noto Sans CJK SC","Noto Sans SC",-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif}main{max-width:1080px;margin:auto}h1{font-size:28px;font-weight:600;line-height:1.5;margin:0 0 12px}h2{font-size:16px;font-weight:600;margin:0 0 12px}p{font-size:12px;color:var(--df-muted,#65758b);line-height:1.9;margin:0 0 20px}section{margin-top:32px;padding:24px;border:1px solid var(--df-rule,#dce3ec);border-radius:8px;background:var(--df-surface,#fff)}.df-scroll{overflow-x:auto}.df-scroll svg{margin-inline:auto}footer{margin-top:24px;font-size:12px;color:var(--df-muted,#65758b);line-height:1.9}@media print{body{padding:0}section{break-inside:avoid}.df-scroll{overflow:visible}}</style></head><body><main><h1>${variant.name}</h1><p>设备或离线程序执行处理，网站分析真实录音。两张图分别说明信号路径与计算步骤；模块顺序以设备实现为准。</p>${figures}<footer>ERLE 需排除后级处理；正式 MOS、SLR / RLR、TCL 另需专用算法、测试配置或声学标定。字体沿用 ITU 项目的本机字体栈，离线显示随本机字体回退。</footer></main></body></html>`;
  await writeFile(new URL(`nv200-audio-flow${variant.suffix}.html`, output), html.replace("<footer>", `${explanation}<footer>`));
}
console.log(`Audio flow: 2 diagrams, ${DIAGRAMS.device.nodes.length} + ${DIAGRAMS.website.nodes.length} nodes, 3 standalone HTML variants generated.`);
