import { useState } from "react";
import {
  ArrowRight,
  BookOpen,
  Calculator,
  CheckCircle2,
  FileInput,
  Info,
} from "lucide-react";
import { METRICS } from "../domain/metrics";
import { formatNumber } from "../domain/model";
import { StatusBadge } from "../components/Shared";

export default function MetricLibrary() {
  const [selected, setSelected] = useState("M01");
  const metric = METRICS.find((item) => item.id === selected)!;
  return (
    <>
      <section className="panel audio-methods">
        <h2>实际音频可计算的工程指标</h2>
        <p>
          四路音频分别承担参考、处理前、处理后与远端回声角色。所有指标保留音频
          SHA256、声道、采样率和片段配置；缺少条件时不给数值。
        </p>
        <div className="audio-method-grid">
          {[
            [
              "数字电平 / 满幅比例",
              "RMS = 10log10(mean(x²))",
              "完整文件所选声道。以数字满量程为参考；不是 dB SPL、dB[Pa] 或 dBm0。满幅阈值为 |x|≥0.999。",
            ],
            [
              "相对时延",
              "τ = argmax |normalized cross-correlation|",
              "干净参考与待测录音去直流后对齐。检查静音、多峰、低相关与搜索边界。正值表示待测晚于参考，不是 RTD。",
            ],
            [
              "语音保真 SI-SDR",
              "10log10(||αs||² / ||y−αs||²)",
              "需可靠对齐；α=(yᵀs)/(sᵀs)。对整体增益不敏感，需要同时检查相对增益。不能替代 MOS。",
            ],
            [
              "STOI 可懂度",
              "pystoi · standard STOI",
              "需确认比较片段是真实语音并具有足够有效帧。内部 10 kHz 重采样；无有效帧时不输出分数。未接入 POLQA/TMOS。",
            ],
            [
              "噪声与片段 SNR",
              "SNR = 10log10((Pmix−Pn)/Pn)",
              "噪声片段需无语音、无回声；估计 SNR 额外要求平稳、不相关噪声。混合功率小于噪声功率或数字静音时无结果。",
            ],
            [
              "回声片段衰减 / ERLE",
              "10log10(Pbefore/Pafter)",
              "需远端单讲、同步电平和可靠回声相关性。最终输出的总衰减可能含后级 NLP、降噪和增益影响；只有确认 AEC 滤波器测量点并排除这些处理后，才启用 ERLE 工程估计。不是 TCL/TCLw。",
            ],
            [
              "PSD / 谱峰",
              "Welch power spectral density",
              "保留原始采样率，单位 dBFS/Hz。绘图按频段保留峰值；没有校准掩模时不做谱峰合规判断。",
            ],
            [
              "工程传递函数",
              "H1(f) = Pxy(f) / Pxx(f)",
              "可靠对齐后仅显示输入功率高于峰值 −40 dB、相干性≥0.6 的频点。未经声学标定，未套用原表频响掩模。",
            ],
          ].map(([name, formula, note]) => (
            <div key={name}>
              <h3>{name}</h3>
              <code>{formula}</code>
              <p>{note}</p>
            </div>
          ))}
        </div>
      </section>
      <h2 className="audio-reference-heading">
        原表 17 类指标 · 方法与限值参考
      </h2>
      <div className="metric-library">
        <section className="panel metric-navigation">
          <div className="metric-nav-heading">
            <BookOpen size={18} />
            <h2>17 类计算指标</h2>
          </div>
          {METRICS.map((item) => (
            <button
              key={item.id}
              className={`metric-nav-item ${item.id === selected ? "selected" : ""}`}
              onClick={() => setSelected(item.id)}
            >
              <span>{item.id}</span>
              <div>
                <strong>{item.name}</strong>
                <small>
                  {item.symbol} · {item.group}
                </small>
              </div>
              {item.id === selected ? <ArrowRight size={15} /> : null}
            </button>
          ))}
        </section>
        <section className="panel metric-content">
          <div className="metric-title">
            <span className="metric-id">{metric.id}</span>
            <div>
              <h2>{metric.name}</h2>
              <p>
                {metric.group} / {metric.symbol}
              </p>
            </div>
          </div>
          <p className="metric-description">{metric.description}</p>
          <div className="formula-box">
            <span>计算定义</span>
            <code>{metric.formula}</code>
          </div>
          <div className="metric-section">
            <h3>
              <Calculator size={18} />
              计算方法
            </h3>
            <p>{metric.method}</p>
          </div>
          <div className="metric-section">
            <h3>
              <FileInput size={18} />
              输入要求与判定条件
            </h3>
            <p>{metric.requirements}</p>
          </div>
          <RangeCalculator key={metric.id} id={metric.id} />
          <div className="metric-scope-note">
            <Info size={16} />
            <p>
              上方工程指标来自实际音频。此处保留原表定义与表限演算；SLR/RLR、MOS、TCL
              及合规曲线仍需标准测试条件、标定或专用算法，不能以工程计算值替代。
            </p>
          </div>
        </section>
      </div>
    </>
  );
}

function RangeCalculator({ id }: { id: string }) {
  const defaults: Record<string, [string, string, string]> = {
    M01: ["648.6", "170", "275"],
    M02: ["16.72", "9", "17"],
    M03: ["3.36", "-2", "6"],
    M04: ["0.13", "-0.5", "0.5"],
    M06: ["-73.39", "-200", "-53"],
    M08: ["3.4", "3", "5"],
    M09: ["14.49", "35", "200"],
    M10: ["92", "50", "100"],
    M11: ["25.87", "50", "200"],
    M12: ["11.61", "6", "200"],
    M13: ["3.5", "1", "5"],
    M14: ["4.5", "3", "5"],
    M15: ["3.3", "3", "5"],
    M16: ["-0.33", "-5", "2"],
    M17: ["10.84", "0", "120"],
  };
  const [values, setValues] = useState<string[]>(defaults[id] ?? ["", "", ""]);
  const numbers = values.map((v) => (v.trim() === "" ? null : Number(v)));
  const valid =
    numbers.every((v) => v !== null && Number.isFinite(v)) &&
    numbers[1]! <= numbers[2]!;
  const margin = valid
    ? Math.min(numbers[0]! - numbers[1]!, numbers[2]! - numbers[0]!)
    : null;
  if (!defaults[id])
    return (
      <div className="calculator-box">
        <h3>
          <Calculator size={17} />
          曲线项的计算准备
        </h3>
        <p>
          完整频率曲线、上下限掩模和余量符号确认后，才能进行逐频率判定。单个距离值不足以重建曲线。
        </p>
      </div>
    );
  return (
    <div className="calculator-box">
      <h3>
        <Calculator size={17} />
        表限判定演算
      </h3>
      <p>调整数值，即时查看是否落在指定区间内。</p>
      <div className="calculator-inputs">
        {["测量值 x", "下限 L", "上限 U"].map((label, index) => (
          <label key={label}>
            <span>{label}</span>
            <input
              type="number"
              step="any"
              aria-label={`演算${label}`}
              value={values[index]}
              onChange={(event) =>
                setValues((old) =>
                  old.map((v, i) => (i === index ? event.target.value : v)),
                )
              }
            />
          </label>
        ))}
      </div>
      {valid ? (
        <div className="calculator-answer">
          <StatusBadge status={margin! >= 0 ? "pass" : "fail"} />
          <span>
            带符号余量 <strong>{formatNumber(margin)}</strong>
          </span>
          <span>
            超限量 <strong>{formatNumber(Math.max(-margin!, 0))}</strong>
          </span>
        </div>
      ) : (
        <p className="field-error">
          请输入完整、有效的数值，并确保下限不大于上限。
        </p>
      )}
      <small>
        <CheckCircle2 size={13} />
        边界包含在区间内；此演算用于原表数值限值，不替代指标有效性检查。
      </small>
    </div>
  );
}
