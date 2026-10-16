import { useState } from "react";
import { ROLE_COLOR, ROLE_LABEL } from "../domain/audio";
import type { AudioDescription, AudioReport } from "../domain/audio";
export function Waveform({
  file,
  label,
}: {
  file: AudioDescription;
  // The workbench uses the offline role vocabulary; other pages pass their own
  // wording while keeping the same role-based colour.
  label?: string;
}) {
  const title = label ?? ROLE_LABEL[file.role];
  const n = file.waveform.min.length;
  const top = file.waveform.max.map(
    (v, i) => `${(i * 800) / Math.max(1, n - 1)},${50 - v * 46}`,
  );
  const bottom = file.waveform.min
    .map((v, i) => `${(i * 800) / Math.max(1, n - 1)},${50 - v * 46}`)
    .reverse();
  return (
    <div className="waveform-row">
      <div className="waveform-label">
        <span style={{ background: ROLE_COLOR[file.role] }} />
        <strong>{title}</strong>
        <small>
          {file.duration.toFixed(2)} s / {file.sample_rate / 1000} kHz / Ch{" "}
          {file.selected_channel}
        </small>
      </div>
      <svg
        viewBox="0 0 800 100"
        role="img"
        aria-label={`${title}完整文件波形，纵轴幅度正负1 FS`}
      >
        <line x1="0" y1="50" x2="800" y2="50" stroke="#dfe5ee" />
        <polygon
          points={[...top, ...bottom].join(" ")}
          fill={ROLE_COLOR[file.role]}
          opacity="0.8"
        />
      </svg>
      <div className="waveform-axis">
        <span>0 s</span>
        <span>±1 FS</span>
        <span>{file.duration.toFixed(2)} s</span>
      </div>
    </div>
  );
}
type Series = {
  label: string;
  color: string;
  x: number[];
  y: (number | null)[];
};
export function SpectrumPlot({
  series,
  unit,
  label,
}: {
  series: Series[];
  unit: string;
  label: string;
}) {
  const values = series.flatMap((s) =>
    s.y.filter((v): v is number => v !== null && Number.isFinite(v)),
  );
  if (!values.length)
    return (
      <div className="chart-empty">没有满足激励强度与相干性条件的频点。</div>
    );
  const min = Math.floor(Math.min(...values) / 10) * 10;
  const max = Math.max(min + 20, Math.ceil(Math.max(...values) / 10) * 10);
  const xmax = Math.max(1, ...series.map((s) => Math.max(...s.x)));
  const X = (x: number) => 54 + (x / xmax) * 626;
  const Y = (y: number) => 18 + ((max - y) / (max - min)) * 170;
  function paths(s: Series) {
    const out: string[] = [];
    let current = "";
    s.y.forEach((y, i) => {
      if (y === null) {
        if (current) out.push(current);
        current = "";
      } else
        current += `${current ? "L" : "M"}${X(s.x[i]).toFixed(1)},${Y(y).toFixed(1)} `;
    });
    if (current) out.push(current);
    return out;
  }
  return (
    <div className="spectrum-plot">
      <svg viewBox="0 0 710 225" role="img" aria-label={label}>
        {[0, 1, 2, 3, 4].map((i) => {
          const y = min + ((max - min) * i) / 4;
          return (
            <g key={i}>
              <line x1="54" x2="680" y1={Y(y)} y2={Y(y)} stroke="#e8edf4" />
              <text x="45" y={Y(y) + 4} textAnchor="end">
                {y.toFixed(0)}
              </text>
              <text x={X((xmax * i) / 4)} y="210" textAnchor="middle">
                {((xmax * i) / 4000).toFixed(1)} kHz
              </text>
            </g>
          );
        })}
        {series.map((s) => (
          <g key={s.label}>
            {paths(s).map((d, i) => (
              <path
                key={i}
                d={d}
                fill="none"
                stroke={s.color}
                strokeWidth="1.7"
              >
                <title>{s.label}</title>
              </path>
            ))}
          </g>
        ))}
      </svg>
      <div className="chart-legend">
        <span>{unit}</span>
        {series.map((s) => (
          <span key={s.label}>
            <i style={{ background: s.color }} />
            {s.label}
          </span>
        ))}
      </div>
    </div>
  );
}
export default function AudioCharts({ report }: { report: AudioReport }) {
  const [tab, setTab] = useState<"psd" | "response">("psd");
  const series =
    tab === "psd"
      ? report.files.map((f) => ({
          label: ROLE_LABEL[f.role],
          color: ROLE_COLOR[f.role],
          x: f.spectrum.frequency,
          y: f.spectrum.dbfs_hz,
        }))
      : report.pairs
          .filter((p) => p.response)
          .map((p) => ({
            label: ROLE_LABEL[p.role],
            color: ROLE_COLOR[p.role],
            x: p.response!.frequency,
            y: p.response!.gain_db,
          }));
  return (
    <div className="audio-visuals">
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>音频波形</h2>
            <p>完整文件 · 原始采样率 · 所选声道</p>
          </div>
        </div>
        <div className="waveform-list">
          {report.files.map((f) => (
            <Waveform key={f.role} file={f} />
          ))}
        </div>
      </section>
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>{tab === "psd" ? "功率谱密度" : "参考到录音的传递函数"}</h2>
            <p>
              {tab === "psd"
                ? "Welch PSD · 原始采样率 · dBFS/Hz"
                : "H1 工程估计 · 仅显示相干性 ≥0.6 且充分激励的频点"}
            </p>
          </div>
        </div>
        <div className="chart-tabs">
          <button
            className={tab === "psd" ? "active" : ""}
            onClick={() => setTab("psd")}
          >
            频谱
          </button>
          <button
            className={tab === "response" ? "active" : ""}
            onClick={() => setTab("response")}
          >
            传递函数
          </button>
        </div>
        <SpectrumPlot
          series={series}
          unit={tab === "psd" ? "dBFS/Hz" : "dB"}
          label={
            tab === "psd" ? "各音频功率谱密度对比" : "处理前后工程传递函数对比"
          }
        />
        <p className="chart-note">
          {tab === "psd"
            ? "按频段保留峰值用于绘图；谱峰不等于标准掩模余量。"
            : "需要相同内容和可靠对齐。该曲线未经声学标定，未进行原表频响合规判定。"}
        </p>
      </section>
    </div>
  );
}
