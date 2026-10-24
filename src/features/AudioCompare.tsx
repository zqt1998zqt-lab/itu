import { displayValue, metricOf } from "../domain/audio";
import type { AudioReport } from "../domain/audio";
import { AudioStatus } from "../components/AudioHighlights";
import { SpectrumPlot } from "../components/AudioCharts";
export default function AudioCompare({ report }: { report: AudioReport }) {
  const noise = metricOf(report, "noise_attenuation"),
    speech = metricOf(report, "sisdr_improvement");
  const tradeoff =
    noise?.value !== null &&
    noise?.value !== undefined &&
    noise.value > 0 &&
    speech?.value !== null &&
    speech?.value !== undefined &&
    speech.value < 0;
  const series = report.pairs
    .filter((p) => p.response)
    .map((p) => ({
      label: p.role === "before" ? "处理前" : "处理后",
      color: p.role === "before" ? "#c48c38" : "#2563eb",
      x: p.response!.frequency,
      y: p.response!.gain_db,
    }));
  return (
    <>
      <div className={`audio-info ${tradeoff ? "amber" : ""}`}>
        {tradeoff
          ? "检测到权衡：噪声变小，但 SI-SDR 下降。请同时检查语音失真与整体增益。"
          : "先比较噪声与语音保真，再判断改善是否有代价。所有结果使用同一干净参考；缺少有效条件时不计算变化量。"}
      </div>
      <div className="comparison-cards">
        {[
          ["sisdr_improvement", "SI-SDR 改善", "dB"],
          ["stoi_improvement", "STOI 变化", ""],
          ["noise_attenuation", "噪声衰减", "dB"],
        ].map(([id, label, unit]) => {
          const m = metricOf(report, id);
          return (
            <section className="panel audio-highlight" key={id}>
              <div>
                <h3>{label}</h3>
                <AudioStatus metric={m} />
              </div>
              <strong>
                {m?.value !== null && m?.value !== undefined && m.value > 0
                  ? "+"
                  : ""}
                {displayValue(m, id.startsWith("stoi") ? 4 : 2)}{" "}
                <small>{unit}</small>
              </strong>
              <p>{m?.note}</p>
            </section>
          );
        })}
      </div>
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>处理前 / 处理后</h2>
            <p>时延正值表示晚于参考；SI-SDR 与 STOI 越高通常越好。</p>
          </div>
        </div>
        <div className="table-scroll">
          <table className="audio-table comparison-table">
            <thead>
              <tr>
                <th>指标</th>
                <th>处理前</th>
                <th>处理后</th>
                <th>解释</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["delay", "相对时延", "ms", "不是往返 RTD"],
                ["sisdr", "SI-SDR", "dB", "去除整体增益影响后比较残差"],
                ["gain", "相对增益", "dB", "已对齐 RMS 电平比，含噪声与失真"],
                ["stoi", "STOI", "", "可懂度估计，不等于 MOS"],
                ["noise", "噪声片段 RMS", "dBFS", "需无语音、无回声片段"],
                [
                  "snr",
                  "片段估计 SNR",
                  "dB",
                  "平稳噪声假设，在线性域扣除噪声功率",
                ],
              ].map(([id, label, unit, note]) => (
                <tr key={id}>
                  <td>
                    <strong>{label}</strong>
                  </td>
                  {["before", "after"].map((role) => (
                    <td key={role}>
                      {displayValue(
                        metricOf(report, `${id}_${role}`),
                        id === "stoi" ? 4 : 2,
                      )}{" "}
                      <small>{unit}</small>
                    </td>
                  ))}
                  <td>{note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      <section className="panel comparison-spectrum">
        <div className="panel-heading">
          <div>
            <h2>处理前后的传递函数</h2>
            <p>统一分析采样率 · 已对齐片段 · 未套用标准掩模</p>
          </div>
        </div>
        <SpectrumPlot series={series} unit="dB" label="处理前后传递函数对比" />
      </section>
    </>
  );
}
