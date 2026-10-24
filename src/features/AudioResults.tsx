import { useState } from "react";
import { ArrowRight, Search } from "lucide-react";
import AudioCharts from "../components/AudioCharts";
import { displayValue, metricOf } from "../domain/audio";
import type { AudioReport } from "../domain/audio";
import { AudioStatus, Highlights } from "../components/AudioHighlights";
export default function AudioResults({ report }: { report: AudioReport }) {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState("all");
  const rows = report.metrics.filter(
    (m) =>
      (filter === "all" ||
        (filter === "computed" ? m.value !== null : m.value === null)) &&
      `${m.name} ${m.note}`.toLowerCase().includes(query.toLowerCase()),
  );
  return (
    <>
      <Highlights report={report} />
      <div className="report-context">
        <span>
          {report.files.length} 个实际音频 · 成对计算{" "}
          {report.options.sample_rate / 1000} kHz
        </span>
        <span>
          参考比较片段：{report.options.speech.start} s →{" "}
          {report.options.speech.end ?? "文件末尾"} s
        </span>
        <span>
          计算时间：{new Date(report.created_at).toLocaleString("zh-CN")}
        </span>
      </div>
      <AudioCharts report={report} />
      <section className="panel audio-results-panel">
        <div className="panel-heading">
          <div>
            <h2>计算明细</h2>
            <p>数字电平与工程估计不自动套用实验室限值。</p>
          </div>
          <a className="text-link" href="#metrics">
            查看指标条件
            <ArrowRight size={14} />
          </a>
        </div>
        <div className="audio-result-filters">
          <label className="audio-search">
            <Search size={16} />
            <input
              aria-label="搜索计算指标"
              placeholder="搜索指标或条件…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </label>
          <select
            aria-label="计算状态筛选"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          >
            <option value="all">全部指标</option>
            <option value="computed">已有计算值</option>
            <option value="missing">待补条件</option>
          </select>
          <span>{rows.length} 项</span>
        </div>
        <div className="table-scroll">
          <table className="audio-table">
            <thead>
              <tr>
                <th>指标 / 方法</th>
                <th>计算结果</th>
                <th>状态</th>
                <th>条件与解释</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((m) => (
                <tr key={m.id}>
                  <td>
                    <strong>{m.name}</strong>
                    {m.method ? <small>{m.method}</small> : null}
                  </td>
                  <td className="numeric">
                    {displayValue(
                      m,
                      m.id.startsWith("stoi") || m.id.startsWith("dc_") ? 4 : 2,
                    )}{" "}
                    <small>{m.unit}</small>
                  </td>
                  <td>
                    <AudioStatus metric={m} />
                  </td>
                  <td>{m.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {!rows.length ? (
            <p className="chart-empty">没有匹配的指标。</p>
          ) : null}
        </div>
      </section>
      <details className="panel audio-provenance">
        <summary>音频身份与计算配置</summary>
        {report.files.map((f) => (
          <div key={f.role}>
            <strong>{f.name}</strong>
            <code>SHA256 {f.sha256}</code>
          </div>
        ))}
        <pre>{JSON.stringify(report.options, null, 2)}</pre>
      </details>
    </>
  );
}
