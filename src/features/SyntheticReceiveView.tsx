import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import {
  AlertTriangle,
  AudioLines,
  Download,
  FileAudio,
  Headphones,
  Info,
  LoaderCircle,
  Play,
  RefreshCw,
  ShieldAlert,
  Square,
  Upload,
} from "lucide-react";
import "./synthetic-receive.css";

type ScalarValue = string | number | boolean | null;
type MetricRow = Record<string, ScalarValue>;

type SyntheticReceiveArtifact = {
  url: string;
  sha256: string | null;
};

type SyntheticReceiveInput = {
  speaker_sha256: string | null;
  rate_hz: number | null;
  sample_count: number | null;
  azimuth_deg: number | null;
  head_radius_m: number | null;
  speed_of_sound_mps: number | null;
  upper_hz: number | null;
  noise_start_s: number | null;
  noise_end_s: number | null;
  left_ir_sha256: string | null;
  right_ir_sha256: string | null;
};

type SyntheticReceiveMetrics = {
  left_rms_dbfs: number | null;
  right_rms_dbfs: number | null;
  binaural_rms_dbfs: number | null;
  interaural_level_difference_db: number | null;
  interaural_delay_ms: number | null;
  band_levels: MetricRow[];
  out_of_band_ratio_db: number | null;
  noise: MetricRow | null;
};

type SyntheticReceiveReport = {
  evaluation_id: string;
  created_at: string;
  mode: string;
  evidence_kind: string;
  standard_compliance: false;
  formal_hats_evidence: false;
  input: SyntheticReceiveInput;
  metrics: SyntheticReceiveMetrics;
  artifacts: {
    left: SyntheticReceiveArtifact | null;
    right: SyntheticReceiveArtifact | null;
  };
  limitations: string[];
};

type FileFieldProps = {
  label: string;
  description: string;
  file: File | null;
  required?: boolean;
  disabled: boolean;
  onChange: (file: File | null) => void;
};

const METRIC_LABELS: Record<string, string> = {
  left_rms_dbfs: "左耳 RMS",
  right_rms_dbfs: "右耳 RMS",
  binaural_rms_dbfs: "双耳 RMS",
  interaural_level_difference_db: "耳间电平差 ILD",
  interaural_delay_ms: "耳间时延 ITD",
  out_of_band_ratio_db: "带外能量比",
  noise_rms_dbfs: "噪声 RMS",
  noise_floor_dbfs: "本底噪声",
  start_s: "开始时间",
  end_s: "结束时间",
  digital_rms_dbfs: "数字 RMS",
  left_digital_rms_dbfs: "左耳数字 RMS",
  right_digital_rms_dbfs: "右耳数字 RMS",
  peak_dbfs: "峰值",
  peak_prominence_db: "峰值余量",
  weighting: "计权",
  calibration: "校准",
  a_weighted: "A 计权",
  band: "频带",
  label: "名称",
  center_hz: "中心频率",
  center_frequency_hz: "中心频率",
  frequency_hz: "频率",
  low_hz: "下限",
  high_hz: "上限",
  left_dbfs: "左耳电平",
  right_dbfs: "右耳电平",
  binaural_dbfs: "双耳电平",
  ild_db: "ILD",
};

const BAND_COLUMN_ORDER = [
  "band",
  "label",
  "low_hz",
  "high_hz",
  "center_hz",
  "center_frequency_hz",
  "frequency_hz",
  "left_dbfs",
  "right_dbfs",
  "binaural_dbfs",
  "ild_db",
  "interaural_level_difference_db",
];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringValue(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value : null;
}

function numberValue(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function scalarValue(value: unknown): ScalarValue {
  if (value === null) return null;
  if (typeof value === "string" || typeof value === "boolean") return value;
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  if (Array.isArray(value)) {
    const values = value.map(scalarValue).filter((item) => item !== null);
    return values.length ? values.join(" / ") : null;
  }
  if (isRecord(value)) {
    try {
      return JSON.stringify(value);
    } catch {
      return null;
    }
  }
  return null;
}

function flattenMetricRow(
  value: Record<string, unknown>,
  prefix = "",
  depth = 0,
): MetricRow {
  const result: MetricRow = {};
  for (const [key, item] of Object.entries(value)) {
    const field = prefix ? `${prefix}.${key}` : key;
    if (isRecord(item) && depth < 2) {
      Object.assign(result, flattenMetricRow(item, field, depth + 1));
    } else {
      result[field] = scalarValue(item);
    }
  }
  return result;
}

function parseArtifact(value: unknown): SyntheticReceiveArtifact | null {
  if (!isRecord(value)) return null;
  const url = stringValue(value.url);
  if (!url) return null;
  return { url, sha256: stringValue(value.sha256) };
}

function parseBandLevels(value: unknown): MetricRow[] {
  if (Array.isArray(value)) {
    return value.flatMap((item, index) => {
      if (isRecord(item)) return [flattenMetricRow(item)];
      const scalar = scalarValue(item);
      return scalar === null ? [] : [{ value: scalar, index }];
    });
  }
  if (!isRecord(value)) return [];

  const channels = [
    ["left", "left_dbfs"],
    ["right", "right_dbfs"],
    ["binaural", "binaural_dbfs"],
  ] as const;
  if (channels.some(([source]) => isRecord(value[source]))) {
    const frequencies = new Set<string>();
    for (const [source] of channels) {
      const channel = value[source];
      if (!isRecord(channel)) continue;
      for (const [frequency, level] of Object.entries(channel)) {
        if (numberValue(frequency) !== null && numberValue(level) !== null) {
          frequencies.add(frequency);
        }
      }
    }
    return Array.from(frequencies)
      .sort((left, right) => Number(left) - Number(right))
      .map((frequency) => {
        const row: MetricRow = { center_hz: Number(frequency) };
        for (const [source, field] of channels) {
          const channel = value[source];
          row[field] = isRecord(channel)
            ? numberValue(channel[frequency])
            : null;
        }
        return row;
      });
  }

  const rows = Object.values(value)
    .filter(isRecord)
    .map((item) => flattenMetricRow(item));
  return rows.length ? rows : [flattenMetricRow(value)];
}

function parseNoise(value: unknown): MetricRow | null {
  if (isRecord(value)) return flattenMetricRow(value);
  const scalar = scalarValue(value);
  return scalar === null ? null : { noise: scalar };
}

function parseReport(value: unknown): SyntheticReceiveReport {
  if (!isRecord(value)) throw new Error("最新仿真报告不是 JSON 对象");

  const evaluationId = stringValue(value.evaluation_id);
  const createdAt = stringValue(value.created_at);
  const mode = stringValue(value.mode);
  const evidenceKind = stringValue(value.evidence_kind);
  if (!evaluationId || !createdAt || !mode || !evidenceKind) {
    throw new Error("仿真报告缺少 evaluation_id、created_at、mode 或 evidence_kind");
  }
  if (
    value.standard_compliance !== false ||
    value.formal_hats_evidence !== false
  ) {
    throw new Error(
      "仿真报告未明确声明 standard_compliance=false 与 formal_hats_evidence=false",
    );
  }

  const input = isRecord(value.input) ? value.input : {};
  const inputIdentity = isRecord(value.input_identity)
    ? value.input_identity
    : {};
  const speakerIdentity = isRecord(inputIdentity.speaker)
    ? inputIdentity.speaker
    : {};
  const leftIrIdentity = isRecord(inputIdentity.left_ir)
    ? inputIdentity.left_ir
    : {};
  const rightIrIdentity = isRecord(inputIdentity.right_ir)
    ? inputIdentity.right_ir
    : {};
  const noiseInterval = Array.isArray(inputIdentity.noise_interval_s)
    ? inputIdentity.noise_interval_s
    : [];
  const metrics = isRecord(value.metrics) ? value.metrics : {};
  const artifacts = isRecord(value.artifacts) ? value.artifacts : {};
  const bandLevels =
    metrics.band_levels ??
    metrics.third_octave_band_levels_db ??
    metrics.third_octave_band_levels_dbfs;
  const limitations = Array.isArray(value.limitations)
    ? value.limitations.filter(
        (item): item is string => typeof item === "string" && Boolean(item.trim()),
      )
    : [];

  return {
    evaluation_id: evaluationId,
    created_at: createdAt,
    mode,
    evidence_kind: evidenceKind,
    standard_compliance: false,
    formal_hats_evidence: false,
    input: {
      speaker_sha256:
        stringValue(input.speaker_sha256) ??
        stringValue(speakerIdentity.sha256),
      rate_hz:
        numberValue(input.rate_hz) ??
        numberValue(speakerIdentity.sample_rate_hz),
      sample_count:
        numberValue(input.sample_count) ?? numberValue(speakerIdentity.frames),
      azimuth_deg:
        numberValue(input.azimuth_deg) ??
        numberValue(inputIdentity.azimuth_deg),
      head_radius_m:
        numberValue(input.head_radius_m) ??
        numberValue(inputIdentity.head_radius_m),
      speed_of_sound_mps:
        numberValue(input.speed_of_sound_mps) ??
        numberValue(inputIdentity.speed_of_sound_mps),
      upper_hz:
        numberValue(input.upper_hz) ?? numberValue(inputIdentity.upper_hz),
      noise_start_s:
        numberValue(input.noise_start_s) ?? numberValue(noiseInterval[0]),
      noise_end_s:
        numberValue(input.noise_end_s) ?? numberValue(noiseInterval[1]),
      left_ir_sha256:
        stringValue(input.left_ir_sha256) ??
        stringValue(leftIrIdentity.sha256),
      right_ir_sha256:
        stringValue(input.right_ir_sha256) ??
        stringValue(rightIrIdentity.sha256),
    },
    metrics: {
      left_rms_dbfs: numberValue(metrics.left_rms_dbfs),
      right_rms_dbfs: numberValue(metrics.right_rms_dbfs),
      binaural_rms_dbfs: numberValue(metrics.binaural_rms_dbfs),
      interaural_level_difference_db: numberValue(
        metrics.interaural_level_difference_db,
      ),
      interaural_delay_ms: numberValue(metrics.interaural_delay_ms),
      band_levels: parseBandLevels(bandLevels),
      out_of_band_ratio_db: numberValue(metrics.out_of_band_ratio_db),
      noise: parseNoise(metrics.noise),
    },
    artifacts: {
      left: parseArtifact(artifacts.left),
      right: parseArtifact(artifacts.right),
    },
    limitations,
  };
}

async function responseBody(response: Response): Promise<unknown> {
  const text = await response.text();
  if (!text.trim()) return null;
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return text;
  }
}

function detailMessage(body: unknown, status: number): string {
  if (typeof body === "string" && body.trim()) return body;
  if (isRecord(body) && "detail" in body) {
    const detail = body.detail;
    if (typeof detail === "string" && detail.trim()) return detail;
    if (Array.isArray(detail)) {
      const messages = detail.flatMap((item) => {
        if (typeof item === "string" && item.trim()) return [item];
        if (!isRecord(item)) return [];
        const message =
          stringValue(item.msg) ??
          stringValue(item.message) ??
          stringValue(item.detail);
        return message ? [message] : [];
      });
      if (messages.length) return messages.join("；");
    }
  }
  return `请求失败（HTTP ${status}）`;
}

async function checkedResponse(response: Response): Promise<unknown> {
  const body = await responseBody(response);
  if (!response.ok) throw new Error(detailMessage(body, response.status));
  return body;
}

function optionalNumber(
  raw: string,
  label: string,
  options: { min?: number; exclusiveMin?: number } = {},
): number | null {
  if (!raw.trim()) return null;
  const value = Number(raw);
  if (!Number.isFinite(value)) throw new Error(`${label}必须是有效数字`);
  if (options.min !== undefined && value < options.min) {
    throw new Error(`${label}不能小于 ${options.min}`);
  }
  if (options.exclusiveMin !== undefined && value <= options.exclusiveMin) {
    throw new Error(`${label}必须大于 ${options.exclusiveMin}`);
  }
  return value;
}

function requiredNumber(
  raw: string,
  label: string,
  options: { min?: number; exclusiveMin?: number } = {},
): number {
  const value = optionalNumber(raw, label, options);
  if (value === null) throw new Error(`${label}不能为空`);
  return value;
}

function metricText(
  value: number | null,
  digits: number,
  unit: string,
): string {
  return value === null ? "not measured" : `${value.toFixed(digits)} ${unit}`;
}

function compactNumber(value: number): string {
  return new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 3 }).format(
    value,
  );
}

function frequencyText(value: number): string {
  if (Math.abs(value) >= 1000) {
    return `${compactNumber(value / 1000)} kHz`;
  }
  return `${compactNumber(value)} Hz`;
}

function metricLabel(key: string): string {
  return METRIC_LABELS[key] ?? key.replaceAll("_", " ");
}

function tableValue(value: ScalarValue, key: string): string {
  if (value === null) return "not measured";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value !== "number") return value;
  if (key.endsWith("_hz")) return frequencyText(value);
  if (key.endsWith("_dbfs") || key.endsWith("_db")) {
    return `${value.toFixed(2)} dB`;
  }
  if (key.endsWith("_ms")) return `${value.toFixed(3)} ms`;
  if (key.endsWith("_s")) return `${value.toFixed(3)} s`;
  return compactNumber(value);
}

function bandColumns(rows: MetricRow[]): string[] {
  const available = new Set(rows.flatMap((row) => Object.keys(row)));
  const ordered = BAND_COLUMN_ORDER.filter((key) => available.has(key));
  for (const key of Array.from(available)) {
    if (!ordered.includes(key)) ordered.push(key);
  }
  return ordered;
}

function formatDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("zh-CN", {
    dateStyle: "medium",
    timeStyle: "medium",
  }).format(date);
}

function shortHash(value: string | null): string {
  if (!value) return "not measured";
  return value.length > 20 ? `${value.slice(0, 20)}…` : value;
}

function FileField({
  label,
  description,
  file,
  required = false,
  disabled,
  onChange,
}: FileFieldProps) {
  return (
    <label className="synthetic-receive-file-field">
      <span className="synthetic-receive-file-title">
        <FileAudio size={15} aria-hidden="true" />
        {label}
        <small>{required ? "必选" : "可选"}</small>
      </span>
      <input
        type="file"
        accept="audio/*,.wav,.flac,.mp3,.m4a,.aac,.ogg,.aiff"
        required={required}
        disabled={disabled}
        onChange={(event) => onChange(event.target.files?.[0] ?? null)}
      />
      <span className="synthetic-receive-file-note">
        {file ? file.name : description}
      </span>
    </label>
  );
}

function NumberField({
  label,
  unit,
  value,
  min,
  step,
  placeholder,
  disabled,
  onChange,
}: {
  label: string;
  unit: string;
  value: string;
  min?: number;
  step: string;
  placeholder?: string;
  disabled: boolean;
  onChange: (value: string) => void;
}) {
  return (
    <label className="synthetic-receive-number-field">
      <span>
        {label}
        <small>{unit}</small>
      </span>
      <input
        type="number"
        inputMode="decimal"
        min={min}
        step={step}
        value={value}
        placeholder={placeholder}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
      />
    </label>
  );
}

function VirtualEar({
  side,
  artifact,
}: {
  side: "left" | "right";
  artifact: SyntheticReceiveArtifact | null;
}) {
  const label = side === "left" ? "左虚拟耳 L" : "右虚拟耳 R";
  return (
    <div className="synthetic-receive-ear">
      <div className="synthetic-receive-ear-heading">
        <Headphones size={17} aria-hidden="true" />
        <strong>{label}</strong>
        {artifact ? (
          <a
            href={artifact.url}
            download
            className="synthetic-receive-download"
            aria-label={`下载${label}音频`}
          >
            <Download size={14} aria-hidden="true" />
            下载
          </a>
        ) : null}
      </div>
      {artifact ? (
        <>
          <audio
            controls
            preload="none"
            src={artifact.url}
            aria-label={`${label}音频播放器`}
          />
          <code title={artifact.sha256 ?? undefined}>
            SHA256 {shortHash(artifact.sha256)}
          </code>
        </>
      ) : (
        <p className="synthetic-receive-not-measured">not measured</p>
      )}
    </div>
  );
}

function ReportView({ report }: { report: SyntheticReceiveReport }) {
  const cards = [
    {
      label: METRIC_LABELS.left_rms_dbfs,
      value: metricText(report.metrics.left_rms_dbfs, 2, "dBFS"),
    },
    {
      label: METRIC_LABELS.right_rms_dbfs,
      value: metricText(report.metrics.right_rms_dbfs, 2, "dBFS"),
    },
    {
      label: METRIC_LABELS.binaural_rms_dbfs,
      value: metricText(report.metrics.binaural_rms_dbfs, 2, "dBFS"),
    },
    {
      label: METRIC_LABELS.interaural_level_difference_db,
      value: metricText(
        report.metrics.interaural_level_difference_db,
        2,
        "dB",
      ),
    },
    {
      label: METRIC_LABELS.interaural_delay_ms,
      value: metricText(report.metrics.interaural_delay_ms, 3, "ms"),
    },
  ];
  const columns = bandColumns(report.metrics.band_levels);
  const noiseRows = report.metrics.noise
    ? Object.entries(report.metrics.noise)
    : [];
  const hasSupplementalMetric =
    report.metrics.out_of_band_ratio_db !== null || noiseRows.length > 0;

  return (
    <article
      className="synthetic-receive-report"
      aria-labelledby="synthetic-receive-report-title"
    >
      <header className="synthetic-receive-report-header">
        <div>
          <span className="synthetic-receive-eyebrow">ENGINEERING REPORT</span>
          <h3 id="synthetic-receive-report-title">双耳接收仿真结果</h3>
          <p>
            报告模式 <code>{report.mode}</code> · 证据类型{" "}
            <code>{report.evidence_kind}</code>
          </p>
        </div>
        <div className="synthetic-receive-report-meta">
          <time dateTime={report.created_at}>{formatDate(report.created_at)}</time>
          <code title={report.evaluation_id}>
            ID {shortHash(report.evaluation_id)}
          </code>
        </div>
      </header>

      <div className="synthetic-receive-evidence-flags">
        <span>工程模拟</span>
        <span>非正式 HATS 证据</span>
        <code>standard_compliance=false</code>
      </div>

      <div className="synthetic-receive-metric-grid">
        {cards.map((card) => (
          <div className="synthetic-receive-metric-card" key={card.label}>
            <span>{card.label}</span>
            <strong>{card.value}</strong>
          </div>
        ))}
      </div>

      <section
        className="synthetic-receive-section"
        aria-labelledby="synthetic-receive-audio-title"
      >
        <div className="synthetic-receive-section-heading">
          <div>
            <h4 id="synthetic-receive-audio-title">虚拟耳输出</h4>
            <p>左右声道由软件卷积生成，可直接试听并下载。</p>
          </div>
          <AudioLines size={20} aria-hidden="true" />
        </div>
        <div className="synthetic-receive-ear-grid">
          <VirtualEar side="left" artifact={report.artifacts.left} />
          <VirtualEar side="right" artifact={report.artifacts.right} />
        </div>
      </section>

      <section
        className="synthetic-receive-section"
        aria-labelledby="synthetic-receive-band-title"
      >
        <div className="synthetic-receive-section-heading">
          <div>
            <h4 id="synthetic-receive-band-title">频带电平</h4>
            <p>按后端返回的频带与电平字段原样呈现；空值保留为 not measured。</p>
          </div>
        </div>
        {columns.length ? (
          <div
            className="synthetic-receive-table-scroll"
            role="region"
            aria-label="双耳仿真频带电平表，可横向滚动"
            tabIndex={0}
          >
            <table className="synthetic-receive-table">
              <thead>
                <tr>
                  {columns.map((column) => (
                    <th scope="col" key={column}>
                      {metricLabel(column)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {report.metrics.band_levels.map((row, rowIndex) => (
                  <tr key={`band-${rowIndex}`}>
                    {columns.map((column) => (
                      <td key={column}>
                        {tableValue(
                          Object.prototype.hasOwnProperty.call(row, column)
                            ? row[column]
                            : null,
                          column,
                        )}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="synthetic-receive-not-measured">not measured</p>
        )}
      </section>

      <section
        className="synthetic-receive-section"
        aria-labelledby="synthetic-receive-noise-title"
      >
        <div className="synthetic-receive-section-heading">
          <div>
            <h4 id="synthetic-receive-noise-title">噪声 / 带外指标</h4>
            <p>这些值为本次软件模拟的工程观测，不构成正式 HATS 结论。</p>
          </div>
        </div>
        {hasSupplementalMetric ? (
          <dl className="synthetic-receive-supplemental">
            <div>
              <dt>{METRIC_LABELS.out_of_band_ratio_db}</dt>
              <dd>
                {metricText(report.metrics.out_of_band_ratio_db, 2, "dB")}
              </dd>
            </div>
            {noiseRows.map(([key, value]) => (
              <div key={key}>
                <dt>{metricLabel(key)}</dt>
                <dd>{tableValue(value, key)}</dd>
              </div>
            ))}
          </dl>
        ) : (
          <p className="synthetic-receive-not-measured">not measured</p>
        )}
      </section>

      <details className="synthetic-receive-audit">
        <summary>
          <Info size={15} aria-hidden="true" />
          输入证据、几何参数与限制
        </summary>
        <dl>
          <div>
            <dt>Speaker SHA256</dt>
            <dd title={report.input.speaker_sha256 ?? undefined}>
              {shortHash(report.input.speaker_sha256)}
            </dd>
          </div>
          <div>
            <dt>采样率</dt>
            <dd>
              {report.input.rate_hz === null
                ? "not measured"
                : frequencyText(report.input.rate_hz)}
            </dd>
          </div>
          <div>
            <dt>样本数</dt>
            <dd>
              {report.input.sample_count === null
                ? "not measured"
                : compactNumber(report.input.sample_count)}
            </dd>
          </div>
          <div>
            <dt>方位角</dt>
            <dd>
              {report.input.azimuth_deg === null
                ? "not measured"
                : `${compactNumber(report.input.azimuth_deg)}°`}
            </dd>
          </div>
          <div>
            <dt>头半径</dt>
            <dd>
              {report.input.head_radius_m === null
                ? "not measured"
                : `${compactNumber(report.input.head_radius_m)} m`}
            </dd>
          </div>
          <div>
            <dt>声速</dt>
            <dd>
              {report.input.speed_of_sound_mps === null
                ? "not measured"
                : `${compactNumber(report.input.speed_of_sound_mps)} m/s`}
            </dd>
          </div>
          <div>
            <dt>左 IR SHA256</dt>
            <dd title={report.input.left_ir_sha256 ?? undefined}>
              {shortHash(report.input.left_ir_sha256)}
            </dd>
          </div>
          <div>
            <dt>右 IR SHA256</dt>
            <dd title={report.input.right_ir_sha256 ?? undefined}>
              {shortHash(report.input.right_ir_sha256)}
            </dd>
          </div>
        </dl>
        {report.limitations.length ? (
          <ul>
            {report.limitations.map((limitation) => (
              <li key={limitation}>{limitation}</li>
            ))}
          </ul>
        ) : (
          <p className="synthetic-receive-not-measured">
            后端未提供额外限制说明。
          </p>
        )}
      </details>
    </article>
  );
}

export function SyntheticReceiveView({ mode }: { mode: string }) {
  const [speakerOutput, setSpeakerOutput] = useState<File | null>(null);
  const [leftIr, setLeftIr] = useState<File | null>(null);
  const [rightIr, setRightIr] = useState<File | null>(null);
  const [azimuthDeg, setAzimuthDeg] = useState("30");
  const [headRadiusM, setHeadRadiusM] = useState("0.0875");
  const [speedOfSoundMps, setSpeedOfSoundMps] = useState("343");
  const [upperHz, setUpperHz] = useState("");
  const [noiseStartS, setNoiseStartS] = useState("");
  const [noiseEndS, setNoiseEndS] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [report, setReport] = useState<SyntheticReceiveReport | null>(null);
  const [latestMessage, setLatestMessage] = useState("正在读取最新报告…");
  const [latestFailed, setLatestFailed] = useState(false);
  const [latestRetry, setLatestRetry] = useState(0);
  const uploadController = useRef<AbortController | null>(null);
  const latestController = useRef<AbortController | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    latestController.current = controller;
    setLatestFailed(false);
    setLatestMessage("正在读取最新报告…");

    fetch("/api/acoustic/receive-simulation/evaluations?limit=1", {
      cache: "no-store",
      signal: controller.signal,
    })
      .then(async (response) => {
        const body = await checkedResponse(response);
        if (!Array.isArray(body)) {
          throw new Error("最新报告接口未返回数组");
        }
        if (!body.length) {
          setLatestMessage("尚无历史仿真报告。");
          return;
        }
        const latest = parseReport(body[0]);
        setReport(latest);
        setLatestMessage(`已载入最新报告 · ${formatDate(latest.created_at)}`);
      })
      .catch((reason: unknown) => {
        if (controller.signal.aborted) return;
        setLatestFailed(true);
        setLatestMessage(
          `最新报告读取失败：${
            reason instanceof Error ? reason.message : "未知错误"
          }。上传功能仍可使用。`,
        );
      })
      .finally(() => {
        if (latestController.current === controller) {
          latestController.current = null;
        }
      });

    return () => controller.abort();
  }, [latestRetry]);

  useEffect(
    () => () => {
      latestController.current?.abort();
      uploadController.current?.abort();
    },
    [],
  );

  function cancelUpload() {
    const controller = uploadController.current;
    if (!controller) return;
    uploadController.current = null;
    controller.abort();
    setBusy(false);
    setError("");
    setNotice("已取消本次仿真请求。");
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setNotice("");

    if (!speakerOutput) {
      setError("请上传 speaker output 音频。");
      return;
    }
    if (Boolean(leftIr) !== Boolean(rightIr)) {
      setError("left_ir 与 right_ir 必须同时提供，或同时留空。");
      return;
    }

    let azimuth: number;
    let headRadius: number;
    let speedOfSound: number;
    let analysisUpperHz: number | null;
    let noiseStart: number | null;
    let noiseEnd: number | null;
    try {
      azimuth = requiredNumber(azimuthDeg, "方位角");
      headRadius = requiredNumber(headRadiusM, "头半径", { exclusiveMin: 0 });
      speedOfSound = requiredNumber(speedOfSoundMps, "声速", {
        exclusiveMin: 0,
      });
      analysisUpperHz = optionalNumber(upperHz, "分析上限频率", {
        exclusiveMin: 0,
      });
      noiseStart = optionalNumber(noiseStartS, "噪声开始秒", { min: 0 });
      noiseEnd = optionalNumber(noiseEndS, "噪声结束秒", { min: 0 });
      if ((noiseStart === null) !== (noiseEnd === null)) {
        throw new Error("噪声开始秒与噪声结束秒必须同时提供");
      }
      if (
        noiseStart !== null &&
        noiseEnd !== null &&
        noiseEnd <= noiseStart
      ) {
        throw new Error("噪声结束秒必须大于噪声开始秒");
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "参数格式无效");
      return;
    }

    const body = new FormData();
    body.set("mode", mode);
    body.set("azimuth_deg", String(azimuth));
    body.set("head_radius_m", String(headRadius));
    body.set("speed_of_sound_mps", String(speedOfSound));
    if (analysisUpperHz !== null) body.set("upper_hz", String(analysisUpperHz));
    if (noiseStart !== null) body.set("noise_start_s", String(noiseStart));
    if (noiseEnd !== null) body.set("noise_end_s", String(noiseEnd));
    body.set("speaker_output", speakerOutput);
    if (leftIr && rightIr) {
      body.set("left_ir", leftIr);
      body.set("right_ir", rightIr);
    }

    uploadController.current?.abort();
    const controller = new AbortController();
    uploadController.current = controller;
    setBusy(true);

    try {
      const response = await fetch(
        "/api/acoustic/receive-simulation/evaluate",
        {
          method: "POST",
          body,
          signal: controller.signal,
        },
      );
      const result = parseReport(await checkedResponse(response));
      if (controller.signal.aborted) return;
      setReport(result);
      setLatestFailed(false);
      setLatestMessage(`已生成新报告 · ${formatDate(result.created_at)}`);
      setNotice("仿真完成，报告、虚拟耳音频与频带指标已更新。");
    } catch (reason) {
      if (!controller.signal.aborted) {
        setError(reason instanceof Error ? reason.message : "仿真请求失败");
      }
    } finally {
      if (uploadController.current === controller) {
        uploadController.current = null;
        setBusy(false);
      }
    }
  }

  return (
    <section
      className="synthetic-receive-page"
      aria-labelledby="synthetic-receive-title"
    >
      <header className="synthetic-receive-header">
        <div className="synthetic-receive-header-copy">
          <span className="synthetic-receive-eyebrow">
            <AudioLines size={15} aria-hidden="true" />
            SOFTWARE RECEIVE SIMULATION
          </span>
          <h2 id="synthetic-receive-title">纯软件虚拟 DRP / 双耳仿真</h2>
          <p>
            将扬声器输出与可选左右 IR 在软件中组合为左右虚拟耳信号，用于工程诊断和算法比较。
          </p>
        </div>
        <div className="synthetic-receive-evidence-flags">
          <span>工程模拟</span>
          <span>非正式 HATS 证据</span>
          <code>standard_compliance=false</code>
        </div>
      </header>

      <div className="synthetic-receive-warning" role="note">
        <ShieldAlert size={20} aria-hidden="true" />
        <div>
          <strong>非正式 HATS 证据</strong>
          <p>
            这是纯软件接收端工程模拟，不包含人工头与人工耳校准、声压级标定或实验室不确定度。
            任何指标都不能据此声明正式标准符合性。
          </p>
        </div>
      </div>

      <div
        className={`synthetic-receive-latest ${
          latestFailed ? "synthetic-receive-latest-error" : ""
        }`}
        role={latestFailed ? "alert" : "status"}
      >
        <span>{latestMessage}</span>
        {latestFailed ? (
          <button
            type="button"
            className="synthetic-receive-text-button"
            onClick={() => setLatestRetry((value) => value + 1)}
          >
            <RefreshCw size={14} aria-hidden="true" />
            重试
          </button>
        ) : null}
      </div>

      <form className="synthetic-receive-form" onSubmit={submit}>
        <div className="synthetic-receive-form-header">
          <div>
            <span className="synthetic-receive-eyebrow">INPUT</span>
            <h3>上传与分析参数</h3>
          </div>
          <span className="synthetic-receive-mode">
            当前模式 <code>{mode}</code>
          </span>
        </div>

        <div className="synthetic-receive-file-grid">
          <FileField
            label="Speaker output"
            description="必选；容器、声道、采样率与削波由后端校验"
            file={speakerOutput}
            required
            disabled={busy}
            onChange={setSpeakerOutput}
          />
          <FileField
            label="Left IR"
            description="可选；须与右 IR 成对"
            file={leftIr}
            disabled={busy}
            onChange={setLeftIr}
          />
          <FileField
            label="Right IR"
            description="可选；须与左 IR 成对"
            file={rightIr}
            disabled={busy}
            onChange={setRightIr}
          />
        </div>

        <fieldset className="synthetic-receive-parameter-set">
          <legend>几何、带宽与噪声窗口</legend>
          <div className="synthetic-receive-parameter-grid">
            <NumberField
              label="方位角"
              unit="deg"
              value={azimuthDeg}
              step="1"
              disabled={busy}
              onChange={setAzimuthDeg}
            />
            <NumberField
              label="头半径"
              unit="m"
              value={headRadiusM}
              min={0.001}
              step="0.0001"
              disabled={busy}
              onChange={setHeadRadiusM}
            />
            <NumberField
              label="声速"
              unit="m/s"
              value={speedOfSoundMps}
              min={1}
              step="0.1"
              disabled={busy}
              onChange={setSpeedOfSoundMps}
            />
            <NumberField
              label="分析上限频率"
              unit="Hz"
              value={upperHz}
              min={1}
              step="1"
              placeholder="自动"
              disabled={busy}
              onChange={setUpperHz}
            />
            <NumberField
              label="噪声开始"
              unit="s"
              value={noiseStartS}
              min={0}
              step="0.01"
              placeholder="可选"
              disabled={busy}
              onChange={setNoiseStartS}
            />
            <NumberField
              label="噪声结束"
              unit="s"
              value={noiseEndS}
              min={0}
              step="0.01"
              placeholder="可选"
              disabled={busy}
              onChange={setNoiseEndS}
            />
          </div>
        </fieldset>

        <div className="synthetic-receive-submit-row">
          <button
            type="submit"
            className="synthetic-receive-button synthetic-receive-button-primary"
            disabled={busy}
          >
            {busy ? (
              <LoaderCircle
                className="synthetic-receive-spin"
                size={16}
                aria-hidden="true"
              />
            ) : (
              <Play size={16} aria-hidden="true" />
            )}
            {busy ? "正在计算…" : "运行双耳仿真"}
          </button>
          {busy ? (
            <button
              type="button"
              className="synthetic-receive-button synthetic-receive-button-cancel"
              onClick={cancelUpload}
            >
              <Square size={14} aria-hidden="true" />
              取消请求
            </button>
          ) : null}
          <span
            className="synthetic-receive-submit-status"
            role="status"
            aria-live="polite"
          >
            {notice}
          </span>
        </div>

        {error ? (
          <div className="synthetic-receive-error" role="alert">
            <AlertTriangle size={17} aria-hidden="true" />
            <span>{error}</span>
          </div>
        ) : null}
      </form>

      {report ? (
        <ReportView report={report} />
      ) : (
        <div className="synthetic-receive-empty">
          <Upload size={21} aria-hidden="true" />
          <div>
            <strong>尚无可用报告</strong>
            <p>上传必选 speaker output 后运行仿真；历史报告读取失败不会阻断新请求。</p>
          </div>
        </div>
      )}
    </section>
  );
}

export default SyntheticReceiveView;
