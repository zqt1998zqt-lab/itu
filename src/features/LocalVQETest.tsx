import { useEffect, useState } from "react";
import "./localvqe-test.css";

type ModelEntry = {
  id: string;
  label: string;
  architecture: string;
  file: string;
  sha256: string;
  available: boolean;
};

type ModelDocument = {
  engine: string;
  library_available: boolean;
  sample_rate: number;
  scope: string;
  models: ModelEntry[];
};

type RunResult = {
  id: string;
  model: { id: string; label: string; sha256: string };
  metrics: Record<string, number | string>;
  output: { download_url: string };
  boundary: string;
};

type AbRoute = {
  route: string;
  label: string;
  output_rms: number | null;
  output_attenuation_db: number | null;
  near_total_error_rms: number | null;
  near_target_gain_db: number | null;
  near_si_sdr_db: number | null;
  process_ms: number | null;
  path_windows?: Record<string, number | null>;
};
type AbDocument = {
  categories: Record<string, {
    label: string; case_count: number; measured_case_count: number; routes: AbRoute[];
    invalid_cases: { case_id: string; reason: string }[];
    cases: { case_id: string; source_category: string; measurement_status: string; audio: Record<string, string> }[];
  }>;
  negative_controls_passed: boolean;
  case_count: number;
  duration_s: number;
  report_url: string;
  assessment: { status: string; label: string; near_target_gain_db: number | null };
  boundary: string;
  formal_compliance: boolean;
};

export default function LocalVQETest() {
  const [document, setDocument] = useState<ModelDocument | null>(null);
  const [model, setModel] = useState("aec200k");
  const [threads, setThreads] = useState(4);
  const [microphone, setMicrophone] = useState<File | null>(null);
  const [playback, setPlayback] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<RunResult | null>(null);
  const [ab, setAb] = useState<AbDocument | null>(null);
  const [abError, setAbError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/localvqe/models", { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json() as Promise<ModelDocument>;
      })
      .then(setDocument)
      .catch((cause: unknown) => {
        if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : "模型清单读取失败");
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/localvqe/mdf-mask-ab", { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json() as Promise<AbDocument>;
      })
      .then(setAb)
      .catch((cause: unknown) => {
        if (!controller.signal.aborted) setAbError(cause instanceof Error ? cause.message : "A/B 报告读取失败");
      });
    return () => controller.abort();
  }, []);

  async function execute() {
    if (!microphone || !playback) {
      setError("请选择两路 16 kHz 单声道 WAV：麦克风混合输入与远端播放信号 / 回声参考。");
      return;
    }
    setBusy(true);
    setError("");
    setResult(null);
    try {
      const body = new FormData();
      body.append("microphone", microphone);
      body.append("playback", playback);
      body.append("model_id", model);
      body.append("threads", String(threads));
      const response = await fetch("/api/localvqe/process", { method: "POST", body });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
      setResult(payload as RunResult);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "执行失败");
    } finally {
      setBusy(false);
    }
  }

  const selected = document?.models.find((entry) => entry.id === model);
  return (
    <section className="localvqe-test" aria-labelledby="localvqe-title">
      <div className="audio-section-title">
        <div>
          <h2 id="localvqe-title">LocalVQE CPU 独立测试</h2>
          <p>使用官方 Apache-2.0 权重，在本机 CPU 上执行 LocalVQE 自身的线性自适应滤波与神经残差处理。</p>
        </div>
        <span className="status-badge">工程实验</span>
      </div>
      <div className="localvqe-boundary">
        当前入口使用 LocalVQE 自己的 DAF 前端，接收麦克风混合输入与远端播放信号 / 回声参考；它不是现有 VCP MDF 残差的神经后级，也不代表 ITU-T 合规结果。
      </div>
      {error ? <p className="field-error" role="alert">{error}</p> : null}
      <div className="localvqe-grid">
        <div className="localvqe-card">
          <label>
            <span>算法权重</span>
            <select aria-label="LocalVQE 权重" value={model} onChange={(event) => setModel(event.target.value)}>
              {(document?.models ?? [{ id: "aec200k", label: "v1.4-AEC 203K", available: false }]).map((entry) => (
                <option key={entry.id} value={entry.id} disabled={"available" in entry && !entry.available}>
                  {entry.label}{"available" in entry && !entry.available ? "（权重不可用）" : ""}
                </option>
              ))}
            </select>
          </label>
          {selected ? (
            <dl className="localvqe-meta">
              <dt>结构</dt><dd>{selected.architecture}</dd>
              <dt>文件</dt><dd>{selected.file}</dd>
              <dt>SHA256</dt><dd title={selected.sha256}>{selected.sha256.slice(0, 16)}…</dd>
            </dl>
          ) : null}
          <label>
            <span>CPU 线程</span>
            <input aria-label="CPU 线程" type="number" min={1} max={8} value={threads}
              onChange={(event) => setThreads(Number(event.target.value) || 1)} />
          </label>
        </div>
        <div className="localvqe-card">
          <label>
            <span>麦克风混合输入（16 kHz / 单声道）</span>
            <input aria-label="麦克风混合输入" type="file" accept="audio/wav,.wav"
              onChange={(event) => setMicrophone(event.target.files?.[0] ?? null)} />
          </label>
          <label>
            <span>远端播放信号 / 回声参考（16 kHz / 单声道）</span>
            <input aria-label="远端播放信号" type="file" accept="audio/wav,.wav"
              onChange={(event) => setPlayback(event.target.files?.[0] ?? null)} />
          </label>
          <p className="config-helper">两路音频需同步同长；系统只取公共长度，不会生成或补齐回声参考。</p>
        </div>
      </div>
      <div className="synthesize-actions">
        <button type="button" className="primary-button" disabled={busy || !document?.library_available || selected?.available === false}
          onClick={execute}>{busy ? "正在执行 LocalVQE…" : "执行 LocalVQE CPU 测试"}</button>
      </div>
      {result ? (
        <div className="localvqe-result" aria-live="polite">
          <h3>实际执行结果</h3>
          <p>权重 {result.model.label} 已完成推理，运行 ID <code>{result.id}</code>。</p>
          <table className="audio-table">
            <tbody>
              {Object.entries(result.metrics).map(([key, value]) => (
                <tr key={key}><th>{key}</th><td>{String(value)}</td></tr>
              ))}
            </tbody>
          </table>
          <audio controls preload="none" src={result.output.download_url} aria-label="LocalVQE 处理后输出" />
          <a className="text-link" href={result.output.download_url} download>下载输出 WAV</a>
          <p className="config-helper">{result.boundary}</p>
        </div>
      ) : null}
      {abError ? <p className="field-error" role="status">A/B 结果暂不可用：{abError}</p> : null}
      {ab ? (
        <section className="localvqe-ab" aria-labelledby="localvqe-ab-title">
          <div className="audio-section-title">
            <div>
              <h2 id="localvqe-ab-title">现有 MDF 残差 → 203K mask A/B</h2>
              <p>{ab.case_count} 组 × {ab.duration_s} 秒；先处理 RX，再重建回声，比较同一麦克风与播放信号。已补偿固定 256 样本编码延迟。</p>
            </div>
            <span className="status-badge">{ab.negative_controls_passed ? "接线与负控验证通过" : "验证未完成"}</span>
          </div>
          <p className="localvqe-ab-assessment" role="status">{ab.assessment.label}。
            近端单讲目标增益：{ab.assessment.near_target_gain_db?.toFixed(2) ?? "不可测"} dB。
          </p>
          {Object.entries(ab.categories).map(([category, group]) => (
            <div className="localvqe-ab-group" key={category}>
              <h3>{group.label} · 有效 {group.measured_case_count}/{group.case_count} 组</h3>
              {group.invalid_cases.map((invalid) => <p className="config-helper" key={invalid.case_id}>
                {invalid.case_id}：{invalid.reason}
              </p>)}
              <table className="audio-table">
                <thead><tr><th>路线</th><th>输出 RMS</th><th>单讲总电平降低</th><th>近端 SI-SDR</th><th>近端目标增益</th><th>处理耗时</th></tr></thead>
                <tbody>
                  {group.routes.map((route) => (
                    <tr key={route.route}>
                      <th>{route.label}</th>
                      <td>{route.output_rms?.toFixed(5) ?? "—"}</td>
                      <td>{route.output_attenuation_db === null ? "不适用" : `${route.output_attenuation_db.toFixed(2)} dB`}</td>
                      <td>{route.near_si_sdr_db === null ? "不适用" : `${route.near_si_sdr_db.toFixed(2)} dB`}</td>
                      <td>{route.near_target_gain_db === null ? "不适用" : `${route.near_target_gain_db.toFixed(2)} dB`}</td>
                      <td>{route.process_ms === null ? "—" : `${route.process_ms.toFixed(1)} ms`}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {category === "path_change" ? <table className="audio-table" aria-label="第6秒路径突变前后比较">
                <thead><tr><th>路线</th><th>切换前 3–5.5 秒</th><th>切换后 6–7 秒</th><th>恢复段 8–12 秒</th></tr></thead>
                <tbody>{group.routes.map((route) => <tr key={route.route}>
                  <th>{route.label}</th>
                  {["before_switch", "after_switch_0_1s", "after_switch_2_6s"].map((window) => <td key={window}>
                    {route.path_windows?.[window] == null ? "—" : `${route.path_windows[window]?.toFixed(2)} dB`}
                  </td>)}
                </tr>)}</tbody>
              </table> : null}
              <details>
                <summary>按条件组试听同一时间轴的 A/B 输出</summary>
                {group.cases.map((item) => <div className="localvqe-ab-audio-case" key={item.case_id}>
                  <p>{item.case_id} · {item.source_category}{item.measurement_status === "invalid_stimulus" ? " · 输入活动不足" : ""}</p>
                  <div className="localvqe-ab-audio-grid">{group.routes.filter((route) => route.route !== "mdf_residual_wrong_ref").map((route) => <label key={route.route}>
                    <span>{route.label}</span>
                    <audio controls preload="none" src={item.audio[route.route]} aria-label={`${group.label} ${item.case_id} ${route.label}`} />
                  </label>)}</div>
                </div>)}
              </details>
            </div>
          ))}
          <p className="config-helper">负控通过表示接口和测量能识别错误输入。单讲电平降低包含回声与噪声；双讲同时查看 SI-SDR 和近端目标增益，不能把静音当作消除效果。{ab.boundary}</p>
          <a className="text-link" href={ab.report_url} download>下载完整 A/B 报告</a>
        </section>
      ) : null}
    </section>
  );
}
