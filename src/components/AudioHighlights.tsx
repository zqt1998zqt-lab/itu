import { displayValue, metricOf } from "../domain/audio";
import type { AudioMetric, AudioReport } from "../domain/audio";

export function AudioStatus({ metric }: { metric?: AudioMetric }) {
  const status = metric?.status ?? "unavailable";
  return (
    <span className={`audio-status audio-status-${status}`}>
      <i />
      {status === "ok"
        ? "已计算"
        : status === "warning"
          ? "条件提醒"
          : "待补条件"}
    </span>
  );
}
export function Highlights({ report }: { report: AudioReport }) {
  return (
    <div className="audio-highlights">
      {[
        ["delay_after", "相对时延", "ms"],
        ["sisdr_after", "语音保真 SI-SDR", "dB"],
        ["noise_attenuation", "噪声衰减", "dB"],
        ["echo_attenuation", "回声片段衰减", "dB"],
      ].map(([id, title, unit]) => {
        const m = metricOf(report, id);
        return (
          <section key={id} className="panel audio-highlight">
            <div>
              <h3>{title}</h3>
              <AudioStatus metric={m} />
            </div>
            <strong>
              {displayValue(m)} <small>{unit}</small>
            </strong>
            <p>{m?.note ?? "需要干净参考及可靠对齐的有效片段"}</p>
          </section>
        );
      })}
    </div>
  );
}
