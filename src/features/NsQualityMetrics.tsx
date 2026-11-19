import { useEffect, useRef, useState } from "react";
import "./ns-quality-metrics.css";

type Metric = {
  value: number | null;
  status: string;
  reason: string;
  display?: string;
  linear_rms?: number;
};
type MetricsResult = {
  schema_version: string;
  batch_id: string;
  case_id: string;
  output_sha256: string;
  scope: string;
  before: Record<string, Metric>;
  after: Record<string, Metric>;
  analysis: {
    start_s: number;
    end_s: number;
    near_activity_s: number;
    noise_observation_s: number;
    residual_reference_lag_samples: number;
    reference_tx_correlation: number | null;
    alignment: string;
    activity: string;
    retention: string;
    edges: string;
    truncation: string;
    si_sdr: string;
    si_snr: string;
  };
};

const METRICS = [
  { key: "pesq", name: "PESQ · WB", unit: "分", digits: 3, direction: "越高越好" },
  { key: "polqa", name: "POLQA · P.863", unit: "分", digits: 3, direction: "越高越好" },
  { key: "stoi", name: "STOI", unit: "", digits: 4, direction: "越高越好" },
  { key: "si_sdr_db", name: "SI-SDR", unit: "dB", digits: 2, direction: "越高越好" },
  { key: "si_snr_db", name: "SI-SNR · 去均值", unit: "dB", digits: 2, direction: "越高越好" },
  { key: "noise_attenuation_db", name: "噪声衰减量", unit: "dB", digits: 2, direction: "正值表示衰减" },
  { key: "residual_noise_rms_dbfs", name: "残留噪声 RMS", unit: "dBFS", digits: 2, direction: "仅无语音片段" },
  { key: "weak_speech_retention_pct", name: "弱音保持 · 投影能量", unit: "%", digits: 1, direction: "接近 100% 为透明" },
  { key: "onset_loss_db", name: "起音损失 · 首 80 ms", unit: "dB", digits: 2, direction: "0 为透明，正值为损失" },
  { key: "tail_loss_db", name: "尾音损失 · 末 80 ms", unit: "dB", digits: 2, direction: "0 为透明，正值为损失" },
  { key: "speech_truncation_rate_pct", name: "语音截断率 · 帧代理", unit: "%", digits: 1, direction: "越低越好" },
  { key: "clipping_rate_pct", name: "波形削波率", unit: "%", digits: 3, direction: "与吞音截断分别统计" },
] as const;

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseResult(value: unknown, batchId: string, caseId: string, outputSha: string): MetricsResult {
  if (!record(value) || value.schema_version !== "scene-condition-ns-quality-v1"
      || value.batch_id !== batchId || value.case_id !== caseId || value.output_sha256 !== outputSha
      || typeof value.scope !== "string"
      || !record(value.before) || !record(value.after) || !record(value.analysis)) {
    throw new Error("补充指标与当前测试结果身份不一致");
  }
  for (const field of [value.before, value.after]) {
    for (const { key } of METRICS) {
      const metric = field[key];
      if (!record(metric) || typeof metric.status !== "string" || typeof metric.reason !== "string"
          || (metric.value !== null && (typeof metric.value !== "number" || !Number.isFinite(metric.value)))) {
        throw new Error(`补充指标 ${key} 返回值无效`);
      }
    }
  }
  for (const key of ["start_s", "end_s", "near_activity_s", "noise_observation_s", "residual_reference_lag_samples"]) {
    if (typeof value.analysis[key] !== "number" || !Number.isFinite(value.analysis[key])) {
      throw new Error(`补充指标分析区间 ${key} 无效`);
    }
  }
  return value as unknown as MetricsResult;
}

function valueText(metric: Metric | undefined, digits: number, unit: string): string {
  if (!metric) return "—";
  if (typeof metric.value === "number") return `${metric.value.toFixed(digits)}${unit ? ` ${unit}` : ""}`;
  if (metric.display) return `${metric.display}${unit ? ` ${unit}` : ""}`;
  return metric.status === "unavailable" ? "未配置" : "不适用";
}

export default function NsQualityMetrics({ batchId, caseId, outputSha, completed }: {
  batchId: string;
  caseId: string;
  outputSha?: string;
  completed: boolean;
}) {
  const container = useRef<HTMLElement>(null);
  const [visible, setVisible] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const [result, setResult] = useState<MetricsResult | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!container.current) return;
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) { setVisible(true); observer.disconnect(); }
    }, { rootMargin: "160px" });
    observer.observe(container.current);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    setResult(null);
    setError("");
    if (!completed || !outputSha || !visible) return;
    const controller = new AbortController();
    setLoading(true);
    const url = `/api/alignment/acceptance/scene-condition-batches/${encodeURIComponent(batchId)}/cases/${encodeURIComponent(caseId)}/ns-quality-metrics`;
    void fetch(url, { signal: controller.signal, cache: "no-store" })
      .then(async response => {
        const body: unknown = await response.json();
        if (!response.ok) throw new Error(record(body) && typeof body.detail === "string" ? body.detail : `指标计算失败（HTTP ${response.status}）`);
        return parseResult(body, batchId, caseId, outputSha);
      })
      .then(setResult)
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : "补充指标计算失败");
      })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [batchId, caseId, outputSha, completed, visible, attempt]);

  return <section ref={container} className="ns-quality-metrics" aria-label={`条件组 ${caseId} 的 NS 补充指标`}>
    <header className="ns-quality-heading">
      <div><strong>NS 补充指标</strong><p>输入麦克风通道 0 → TX 整链输出 · 与原始近端参考比较</p></div>
      {completed && outputSha ? <button type="button" disabled={loading} onClick={() => { setVisible(true); setAttempt(value => value + 1); }}>
        {loading ? "计算中…" : result ? "刷新指标" : "计算补充指标"}
      </button> : null}
    </header>
    {!completed || !outputSha ? <p role="status">等待该组 TX 执行完成后计算。</p> : null}
    {loading ? <p role="status" aria-live="polite">正在校验音频并计算 PESQ、STOI 与保真／噪声指标…</p> : null}
    {error ? <p className="ns-quality-error" role="alert">{error}</p> : null}
    <div className="table-scroll" tabIndex={0} role="region" aria-label="NS 补充指标对比表">
      <table className="audio-table ns-quality-table" aria-busy={loading}>
        <caption className="sr-only">相同参考下的处理前后指标；空值显示不可用原因，不参与合同 MOS 判定</caption>
        <thead><tr><th scope="col">指标</th><th scope="col">输入</th><th scope="col">TX 输出</th><th scope="col">变化 Δ</th><th scope="col">说明／适用条件</th></tr></thead>
        <tbody>{METRICS.map(metric => {
          const before = result?.before[metric.key], after = result?.after[metric.key];
          const delta = typeof before?.value === "number" && typeof after?.value === "number" ? after.value - before.value : null;
          const deltaUnit = metric.unit === "%" ? "百分点" : metric.unit === "dBFS" ? "dB" : metric.unit;
          return <tr key={metric.key}>
            <th scope="row">{metric.name}</th>
            <td>{valueText(before, metric.digits, metric.unit)}</td>
            <td>{valueText(after, metric.digits, metric.unit)}</td>
            <td>{delta === null ? "—" : `${delta > 0 ? "+" : ""}${delta.toFixed(metric.digits)}${deltaUnit ? ` ${deltaUnit}` : ""}`}</td>
            <td>{after?.reason || before?.reason || metric.direction}</td>
          </tr>;
        })}</tbody>
      </table>
    </div>
    {result ? <>
      <p className="ns-quality-scope">{result.scope}</p>
      <p>分析 {result.analysis.start_s.toFixed(2)}–{result.analysis.end_s.toFixed(2)} s · 近端活动 {result.analysis.near_activity_s.toFixed(2)} s · 噪声观察窗 {result.analysis.noise_observation_s.toFixed(2)} s</p>
      <details><summary>计算口径与指标定义</summary>
        <p>{result.analysis.alignment}；残余对齐 {result.analysis.residual_reference_lag_samples} samples。比较原始响度，不按输出另作增益匹配。</p>
        <p>输入与输出使用相同的参考重叠窗；对齐排除的文件边缘不参与截断率，不能据此排除文件首尾吞音。</p>
        <p>{result.analysis.activity}</p><p>{result.analysis.retention}</p><p>{result.analysis.edges}</p><p>{result.analysis.truncation}</p>
        <p>弱音保持／边缘损失对整段增益作归一，因此不等同于原始响度保持；噪声污染也可能影响投影估计。</p>
        <p>{result.analysis.si_sdr}；{result.analysis.si_snr}。两者口径不同，零均值音频的分数可接近。</p>
        <p>POLQA 必须使用独立 P.863 引擎；PESQ 不替代 POLQA。所有结果为补充工程指标，不改变原有 MOS 合同判定。</p>
      </details>
    </> : null}
  </section>;
}
