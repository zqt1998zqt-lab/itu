import { useEffect, useState } from "react";

const BASE = "/evaluation/no-passenger-current";
type Scope = {
  scenes: number; removed_scenes: number; near_speakers: number;
  sweep_noise_types: number; sweep_snr_db: number[];
  manifest_sha256: string; phases_per_sequence: number; seconds_per_sequence: number;
};
type Matrix = {
  complete: boolean; quality_phases: number; quality_passed_phases: number;
  rows: { rate: number; microphones: number; variant: string; phases: number; quality_passed_phases: number }[];
  paced: { process_p95_ms: number; process_max_ms: number; completion_deadline_misses: number };
};

export default function ActiveDataset() {
  const [scope, setScope] = useState<Scope | null>(null);
  const [matrix, setMatrix] = useState<Matrix | null>(null);
  const [error, setError] = useState("");
  const [refresh, setRefresh] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setError("");
    Promise.all(["scope.json", "matrix-report.json"].map(async name => {
      const response = await fetch(`${BASE}/${name}`, { signal: controller.signal, cache: "no-store" });
      if (!response.ok) throw Error("当前测试集报告尚未发布");
      return response.json();
    })).then(([inventory, replay]) => {
      if (!controller.signal.aborted) { setScope(inventory); setMatrix(replay); }
    }).catch(reason => { if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : "报告读取失败"); });
    return () => controller.abort();
  }, [refresh]);
  return <section className="evaluation-section" aria-label="当前车载测试集">
    <h2>当前车载测试集</h2>
    {scope && matrix ? <>
      <p className="evaluation-note">当前 {scope.scenes.toLocaleString("zh-CN")} 个场景、{scope.near_speakers} 位近端说话人。已排除 {scope.removed_scenes.toLocaleString("zh-CN")} 个乘客讲话干扰场景；保留环境噪声及有播放参考的收音机 AEC。恒定目标扫描：{scope.sweep_noise_types} 类条件，SNR {scope.sweep_snr_db[0]} 至 {scope.sweep_snr_db.at(-1)} dB，步长 1 dB。</p>
      <p className="evaluation-note">连续车流重新运行：每条 {scope.phases_per_sequence} 阶段、{scope.seconds_per_sequence} 秒，覆盖 16 / 24 / 48 kHz 与 1 / 2 / 4 麦。音质通过 {matrix.quality_passed_phases} / {matrix.quality_phases} 阶段。此处为新连续流结果，主测试集的全部场景尚未按新版清单重新完整运行。</p>
      <div className="table-scroll"><table className="audio-table"><thead><tr><th>采样率</th><th>麦克风</th><th>处理配置</th><th>音质通过阶段</th></tr></thead><tbody>{matrix.rows.map(row => <tr key={`${row.rate}-${row.microphones}-${row.variant}`}><td>{row.rate / 1000} kHz</td><td>{row.microphones} 路</td><td>{row.variant === "spatial" ? "启动控制 + MVDR / 风噪" : "启动控制"}</td><td>{row.quality_passed_phases} / {row.phases}</td></tr>)}</tbody></table></div>
      <p className="evaluation-note">48 kHz 四麦单实例 10 ms 节拍：P95 {matrix.paced.process_p95_ms.toFixed(3)} ms、最大 {matrix.paced.process_max_ms.toFixed(3)} ms，完成期限错过 {matrix.paced.completion_deadline_misses} 帧。质量回放与节拍测量独立记录，冻结验收阈值保持不变。</p>
      <div className="evaluation-links"><a href={`${BASE}/scope.json`} download>当前测试范围</a><a href={`${BASE}/inputs-manifest.json`} download>当前输入清单</a><a href={`${BASE}/matrix-report.json`} download>新连续流结果</a><a href={`${BASE}/publication.json`} download>输入与结果 SHA256</a></div>
      <details><summary>当前输入清单 SHA256</summary><p className="numeric">{scope.manifest_sha256}</p></details>
    </> : <p role={error ? "alert" : "status"}>{error || "正在读取当前测试集…"}</p>}
    <button className="button secondary" onClick={() => setRefresh(n => n + 1)}>刷新当前测试集</button>
  </section>;
}
