import { useEffect, useState } from "react";
import { Download, RotateCcw } from "lucide-react";

type Group = { rows: number; quality_pass: number; all_pass: number; failed_metrics: Record<string, number> };
type FrontendReport = {
  complete: true; standard_compliance: false; total_runs: number; scene_rates: number;
  identity: { cli_sha256: string; manifest_sha256: string; model_sha256: string; rules_sha256: string };
  groups: Record<string, Group>;
  coverage: { scenes: number; near_speakers: number; snrs_db: number[]; field_labels: string[] };
  input_quality: { passed: boolean };
  sequence?: {
    native: { runs: number; groups: Record<string, { runs: number; passed: number }> };
    method: Record<string, { passed: number; failed: number; unavailable: number }>;
    streams: { phases: number; quality_passed: number; seconds_per_sequence?: number; identity?: { cli_sha256: string; manifest_sha256: string }; paced: { process_p95_ms: number; completion_deadline_misses: number } };
  };
};
const BASE = "/evaluation/vehicle-frontend-current";
const VARIANTS: Record<string, string> = { baseline: "WebRTC 基线", control: "冷启动 / 突变控制", spatial: "控制 + MVDR / 风噪", rnnoise: "控制 + RNNoise", ulunas: "控制 + UL-UNAS", single_render_negative: "仅第一播放参考 · 负控" };
const METHODS: Record<string, string> = { near_noise: "近端背景连续性", far_noise: "远端背景连续性", double_talk: "双讲投影衰减", cng: "舒适噪声" };

export default function VehicleFrontend() {
  const [data, setData] = useState<FrontendReport | null>(null);
  const [error, setError] = useState("");
  const [refresh, setRefresh] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setError(""); setData(null);
    fetch(`${BASE}/summary.json`, { signal: controller.signal, cache: "no-store" })
      .then(async response => {
        if (!response.ok) throw Error("车载前端实验报告尚未发布");
        const report = await response.json() as FrontendReport;
        if (report.complete !== true || report.standard_compliance !== false || !report.groups || !report.coverage)
          throw Error("车载前端实验报告格式不匹配");
        if (!controller.signal.aborted) setData(report);
      }).catch(reason => { if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : "实验报告读取失败"); });
    return () => controller.abort();
  }, [refresh]);
  return <section className="evaluation-section" aria-label="车载前端配对实验">
    <h2>车载前端配对实验</h2>
    <p className="alignment-note">声场补齐 → 冷启动与噪声突变控制 → 风噪 / MVDR → 多参考 AEC → 轻量神经 NS 对照。使用同一批音频和冻结工程阈值；各候选保留独立结果。</p>
    {!data ? <p role={error ? "alert" : "status"}>{error || "正在读取配对实验…"}</p> : <>
      <p className="alignment-note">{data.coverage.scenes} 个补充场景、{data.coverage.near_speakers} 位近端说话人、{data.scene_rates} 个场景与采样率组合，共 {data.total_runs} 次实际 C++ 处理。SNR：{data.coverage.snrs_db.join(" / ")} dB。输入质量审计：{data.input_quality.passed ? "通过" : "未通过"}。</p>
      <p className="alignment-note">补充场景使用仿真车舱声场；24 / 48 kHz 由 16 kHz 素材转换。UL-UNAS 增强带宽为 8 kHz。音质与包含耗时的通过数分别列出；本批为本机串行诊断，实时期限另测，正式声学测项仍需独立验证。</p>
      <div className="alignment-links"><a href={`${BASE}/report.json`} download><Download size={15} />逐次结果 JSON</a><a href={`${BASE}/field-quality.json`} download><Download size={15} />输入质量审计</a><a href={`${BASE}/report.md`} download><Download size={15} />实现与复现说明</a></div>
      <div className="table-scroll"><table className="audio-table"><thead><tr><th>采样率</th><th>候选</th><th>音质通过</th><th>含耗时通过</th><th>失败指标 / 次数</th></tr></thead><tbody>{Object.entries(data.groups).map(([key, group]) => {
        const [rate, ...parts] = key.split("-"); const variant = parts.join("-");
        return <tr key={key}><td>{Number(rate) / 1000} kHz</td><td>{VARIANTS[variant] ?? variant}</td><td>{group.quality_pass} / {group.rows}</td><td>{group.all_pass} / {group.rows}</td><td>{Object.entries(group.failed_metrics).map(([metric, count]) => `${metric}: ${count}`).join("；") || "无"}</td></tr>;
      })}</tbody></table></div>
      {data.sequence && <>
        <h3>原生高带与连续流回归</h3>
        <p className="alignment-note">原生 48 kHz clean / noisy 配对录音，转换为三种采样率并控制 SNR；两位说话人、8 条语音，共 {data.sequence.native.runs} 次处理。以下列出包含耗时的全部工程门槛通过数。</p>
        <div className="table-scroll"><table className="audio-table"><thead><tr><th scope="col">采样率</th><th scope="col">基线</th><th scope="col">冷启动 / 突变控制</th></tr></thead><tbody>{[16000, 24000, 48000].map(rate => <tr key={rate}><th scope="row">{rate / 1000} kHz</th>{["baseline", "control"].map(variant => {
          const group = data.sequence!.native.groups[`${rate}-${variant}`];
          return <td key={variant}>{group.passed} / {group.runs}</td>;
        })}</tr>)}</tbody></table></div>
        <p className="alignment-note">连续流覆盖三种采样率与 1 / 2 / 4 麦，每条 {data.sequence.streams.seconds_per_sequence ?? 259} 秒且保持状态：音质阶段通过 {data.sequence.streams.quality_passed} / {data.sequence.streams.phases}。另完成三率时钟校正负控。48 kHz 四麦 10 ms 节拍回放的处理 P95：{data.sequence.streams.paced.process_p95_ms.toFixed(3)} ms，完成期限错过：{data.sequence.streams.paced.completion_deadline_misses} 帧。</p>
        <h3>ITU-T 方法诊断</h3>
        {data.sequence.streams.identity ? <p className="alignment-note">连续流已按当前测试范围独立重跑，使用独立版本：CLI {data.sequence.streams.identity.cli_sha256.slice(0, 12)}；输入清单 {data.sequence.streams.identity.manifest_sha256.slice(0, 12)}。上方声场和原生录音实验保持各自批次身份。</p> : null}
        <p className="alignment-note">每项 12 个条件，使用自然语音与数字插入，尚未完成 P.501 信号及声学校准。远端背景连续性包含残余回声影响；CNG 未启用时不适用。</p>
        <div className="table-scroll"><table className="audio-table"><thead><tr><th scope="col">项目</th><th scope="col">通过</th><th scope="col">失败</th><th scope="col">不适用 / 缺测</th></tr></thead><tbody>{Object.entries(data.sequence.method).map(([name, result]) => <tr key={name}><th scope="row">{METHODS[name] ?? name}</th><td>{result.passed}</td><td>{result.failed}</td><td>{result.unavailable}</td></tr>)}</tbody></table></div>
        <div className="alignment-links">{[["native", "原生录音结果"], ["streams", "连续流及负控"], ["method", "方法诊断结果"], ["fir", "FIR 性能配对"]].map(([name, label]) => <a key={name} href={`${BASE}/${name}-report.json`} download><Download size={15} aria-hidden="true" />{label}</a>)}</div>
      </>}
      <details className="alignment-failures"><summary>本批输入、引擎、模型与阈值身份</summary><dl className="evaluation-identities"><dt>CLI SHA256</dt><dd>{data.identity.cli_sha256}</dd><dt>Manifest SHA256</dt><dd>{data.identity.manifest_sha256}</dd><dt>UL-UNAS SHA256</dt><dd>{data.identity.model_sha256}</dd><dt>工程阈值 SHA256</dt><dd>{data.identity.rules_sha256}</dd></dl></details>
    </>}
    <button className="button secondary" onClick={() => setRefresh(n => n + 1)}><RotateCcw size={16} />刷新配对实验</button>
  </section>;
}
