import { useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  Download,
  FileAudio,
  LoaderCircle,
  Play,
  RefreshCw,
  ShieldAlert,
  Square,
} from "lucide-react";

const MODES = [
  "NB-wired",
  "NB-wireless",
  "NB-bt",
  "WB-wired",
  "WB-wireless",
  "WB-bt",
  "SWB-wired",
  "SWB-wireless",
] as const;
type Mode = (typeof MODES)[number];

const SIGNALS = ["near_end", "noise", "microphone", "far_end", "reference"] as const;
type SignalRole = (typeof SIGNALS)[number];
type Status = "pass" | "fail" | "review_required" | "not_measured" | "invalid";

const SIGNAL_LABEL: Record<SignalRole, string> = {
  near_end: "近端音频",
  noise: "噪音",
  microphone: "麦克音",
  far_end: "远端音频",
  reference: "参考音",
};
const SIGNAL_HELP: Record<SignalRole, string> = {
  near_end: "待测的近端输出，与参考音包含同一近端语音",
  noise: "隔离噪声或对应场景的噪声录音",
  microphone: "处理前麦克风混合信号，含近端、回声与噪声",
  far_end: "远端语音源或扬声器播放源",
  reference: "与近端输出对应的干净近端参考；用于对齐和保真分析",
};

type ModelFileMap = Partial<Record<SignalRole, File | null>>;
type FilesByMode = Record<Mode, ModelFileMap>;

type FiveSignalItem = {
  item_id: string;
  source_cell: string;
  family: string;
  direction: string;
  test_name: string;
  unit: string;
  lower_limit: number | string | null;
  upper_limit: number | string | null;
  metric: string | null;
  value: number | null;
  status: Status;
  reason: string;
  evidence_grade: "engineering_proxy" | "not_measured";
  standard_compliance: false;
};
type Summary = Record<Status | "total" | "measured", number>;
type FiveSignalReport = {
  schema_version: 1;
  evaluation_id: string;
  created_at: string;
  mode: Mode;
  scope: string;
  standard_compliance: false;
  formal_test_status: "not_measured";
  input_rate_hz: number;
  summary: Summary;
  items: FiveSignalItem[];
  limitations: string[];
  analysis: {
    segments: Record<string, number>;
    alignments: Record<string, { delay_ms: number | null; correlation: number | null; available: boolean; note: string }>;
    model_a?: {
      compliance_status: string;
      score_use?: string;
      formal_compliance_claim_allowed?: boolean;
      scores?: Record<string, { value: number; tree_standard_deviation: number }>;
      verification?: {
        status: string;
        condition_count: number;
        passed_conditions: number;
        tolerance_mos: number;
        failed_reason?: string | null;
        summary?: Record<string, {
          mae: number;
          rmse: number;
          maximum_absolute_error: number;
        }>;
      };
      error?: string;
    };
    values: Record<string, number | null>;
  };
  evidence: Record<SignalRole, { artifact: { sha256: string; path: string } }>;
  result: { sha256: string; path: string };
};

function emptyFiles(): FilesByMode {
  return Object.fromEntries(
    MODES.map(mode => [
      mode,
      Object.fromEntries(SIGNALS.map(role => [role, null])) as ModelFileMap,
    ]),
  ) as FilesByMode;
}

function complete(files: ModelFileMap): boolean {
  return SIGNALS.every(role => Boolean(files[role]));
}

async function jsonResponse(response: Response): Promise<any> {
  const body = await response.json();
  if (!response.ok) {
    const detail = Array.isArray(body.detail)
      ? body.detail.map((item: { msg?: string }) => item.msg).filter(Boolean).join("；")
      : body.detail;
    throw Error(detail || `HTTP ${response.status}`);
  }
  return body;
}

function fmt(value: number | null, digits = 3): string {
  return value === null || !Number.isFinite(value) ? "—" : value.toFixed(digits);
}

const STATUS_LABEL: Record<Status, string> = {
  pass: "工程通过",
  fail: "工程失败",
  review_required: "需复核",
  not_measured: "不可测",
  invalid: "无效",
};

export default function FiveSignalQualification() {
  const [files, setFiles] = useState<FilesByMode>(emptyFiles);
  const [channel, setChannel] = useState(0);
  const [maxDelay, setMaxDelay] = useState(2000);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState("");
  const [error, setError] = useState("");
  const [reports, setReports] = useState<FiveSignalReport[]>([]);
  const [filterMode, setFilterMode] = useState<Mode | "all">("all");
  const [filterStatus, setFilterStatus] = useState<Status | "all">("all");
  const controller = useRef<AbortController | null>(null);

  const allItems = useMemo(
    () => reports.flatMap(report => report.items),
    [reports],
  );
  const completeModes = useMemo(
    () => MODES.filter(mode => complete(files[mode])),
    [files],
  );
  const aggregate = useMemo(() => {
    const counts: Record<Status, number> = {
      pass: 0, fail: 0, review_required: 0, not_measured: 0, invalid: 0,
    };
    for (const item of allItems) counts[item.status]++;
    return {
      total: allItems.length,
      measured:
        counts.pass + counts.fail + counts.review_required + counts.invalid,
      ...counts,
    };
  }, [allItems]);
  const filteredItems = allItems.filter(item =>
    (filterMode === "all" || item.item_id.startsWith(`${filterMode}:`)) &&
    (filterStatus === "all" || item.status === filterStatus)
  );

  function updateFile(mode: Mode, role: SignalRole, file: File | null) {
    setFiles(old => ({
      ...old,
      [mode]: { ...old[mode], [role]: file },
    }));
    setReports(old => old.filter(report => report.mode !== mode));
  }

  async function execute(selectedModes: Mode[]) {
    if (controller.current) return;
    const incomplete = selectedModes.filter(mode => !complete(files[mode]));
    if (incomplete.length) {
      setError(`以下模式尚未选择完整五路音频：${incomplete.join("、")}`);
      return;
    }
    if (selectedModes.length > 1) {
      const seen = new Map<string, Mode>();
      for (const mode of selectedModes) {
        const signature = SIGNALS.map(role => {
          const file = files[mode][role]!;
          return `${file.name}:${file.size}:${file.lastModified}`;
        }).join("|");
        const previous = seen.get(signature);
        if (previous) {
          setError(
            `${previous} 与 ${mode} 使用了同一组五路文件；每个业务模式必须提供自己的测量音频。`,
          );
          return;
        }
        seen.set(signature, mode);
      }
    }
    const task = new AbortController();
    controller.current = task;
    setBusy(true);
    setError("");
    const next: FiveSignalReport[] = [];
    try {
      for (let index = 0; index < selectedModes.length; index++) {
        const mode = selectedModes[index];
        setProgress(`正在分析 ${mode} · ${index + 1}/${selectedModes.length}`);
        const body = new FormData();
        body.set("mode", mode);
        body.set("channel", String(channel));
        body.set("max_delay_ms", String(maxDelay));
        for (const role of SIGNALS) {
          const file = files[mode][role];
          if (!file) throw Error(`${mode} 缺少 ${SIGNAL_LABEL[role]}`);
          body.set(role, file);
        }
        const report = await jsonResponse(await fetch(
          "/api/acoustic/five-signal/evaluate",
          { method: "POST", body, signal: task.signal },
        )) as FiveSignalReport;
        next.push(report);
      }
      setReports(old => {
        const replaced = new Set(next.map(report => report.mode));
        return [...old.filter(report => !replaced.has(report.mode)), ...next];
      });
      setProgress(`完成：${next.reduce((sum, row) => sum + row.summary.total, 0)} 项已生成逐项结果`);
    } catch (reason) {
      if (!task.signal.aborted) {
        setError(reason instanceof Error ? reason.message : "五路音频评估失败");
      }
    } finally {
      controller.current = null;
      setBusy(false);
    }
  }

  function cancel() {
    controller.current?.abort();
    controller.current = null;
    setBusy(false);
    setProgress("已取消");
  }

  function exportJSON() {
    const payload = {
      schema_version: 1,
      scope: "five-signal engineering assessment; not formal ITU-T compliance",
      standard_compliance: false,
      formal_test_status: "not_measured",
      created_at: new Date().toISOString(),
      reports,
      items: allItems,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json",
    });
    const anchor = document.createElement("a");
    anchor.href = URL.createObjectURL(blob);
    anchor.download = "NV200-五路音频-工程验收.json";
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(anchor.href), 1000);
  }

  return (
    <section className="acoustic-panel five-signal-panel">
      <div className="acoustic-section-head">
        <div>
          <span>五路 PCM 自动工程验收</span>
          <h2>8 模式 × 42 项 = 336 项</h2>
        </div>
        <FileAudio size={22} />
      </div>

      <div className="acoustic-warning">
        <ShieldAlert size={20} />
        <div>
          <strong>每个模式必须使用自己的一组五路音频</strong>
          <p>
            不得把同一组音频复制成 8 个业务模式。系统能测的项给出工程值；
            需要校准声学、POLQA/TMOS、G/N/S-MOS 或另一测量方向的项保持不可测。
            所有结果始终 <code>standard_compliance=false</code>。
          </p>
        </div>
      </div>

      <div className="five-signal-toolbar">
        <label>
          分析声道
          <select value={channel} onChange={event => setChannel(Number(event.target.value))}>
            {Array.from({ length: 8 }, (_, index) => (
              <option key={index} value={index}>声道 {index + 1}</option>
            ))}
          </select>
        </label>
        <label>
          最大对齐时延
          <input
            type="number"
            min="10"
            max="5000"
            value={maxDelay}
            onChange={event => setMaxDelay(Number(event.target.value))}
          />
        </label>
        <button
          className="button primary"
          disabled={busy || completeModes.length === 0}
          onClick={() => void execute(completeModes)}
        >
          {busy ? <LoaderCircle className="spin" size={16} /> : <Play size={16} />}
          运行已装填 {completeModes.length} 模式 · {completeModes.length * 42} 项
        </button>
        <button
          className="button secondary"
          disabled={busy || !MODES.every(mode => complete(files[mode]))}
          onClick={() => void execute([...MODES])}
        >
          <Play size={16} />运行全部 8 模式 · 336 项
        </button>
        {busy ? (
          <button className="button secondary" onClick={cancel}>
            <Square size={14} />取消
          </button>
        ) : null}
      </div>

      <p className="five-signal-progress" aria-live="polite">
        {progress || `已装填 ${completeModes.length}/8 个模式`}
      </p>
      {error ? <p className="acoustic-error" role="alert">{error}</p> : null}

      <div className="five-signal-mode-grid">
        {MODES.map(mode => (
          <article key={mode} className={complete(files[mode]) ? "complete" : ""}>
            <header>
              <div>
                <strong>{mode}</strong>
                <span>{complete(files[mode]) ? "五路齐全" : "等待五路音频"}</span>
              </div>
              <button
                className="text-link"
                disabled={busy || !complete(files[mode])}
                onClick={() => void execute([mode])}
              >
                运行 42 项
              </button>
            </header>
            <div className="five-signal-inputs">
              {SIGNALS.map(role => (
                <label key={role} title={SIGNAL_HELP[role]}>
                  <span>{SIGNAL_LABEL[role]}</span>
                  <input
                    type="file"
                    accept=".wav,.flac,.mp3,.ogg,.m4a,.aac,.aif,.aiff,audio/*"
                    disabled={busy}
                    onChange={event => {
                      updateFile(mode, role, event.target.files?.[0] ?? null);
                      event.target.value = "";
                    }}
                  />
                  <small>{files[mode][role]?.name ?? SIGNAL_HELP[role]}</small>
                </label>
              ))}
            </div>
          </article>
        ))}
      </div>

      {reports.length ? (
        <>
          <div className="five-signal-summary">
            {([
              ["已生成", aggregate.total],
              ["有工程值", aggregate.measured],
              ["工程通过", aggregate.pass],
              ["工程失败", aggregate.fail],
              ["需复核", aggregate.review_required],
              ["不可测", aggregate.not_measured],
              ["无效", aggregate.invalid],
              ["模式数", reports.length],
            ] as const).map(([label, value]) => (
              <div key={label}><span>{label}</span><strong>{value}</strong></div>
            ))}
          </div>

          <div className="five-signal-result-toolbar">
            <label>
              模式
              <select
                value={filterMode}
                onChange={event => setFilterMode(event.target.value as Mode | "all")}
              >
                <option value="all">全部模式</option>
                {reports.map(report => (
                  <option key={report.mode} value={report.mode}>{report.mode}</option>
                ))}
              </select>
            </label>
            <label>
              状态
              <select
                value={filterStatus}
                onChange={event => setFilterStatus(event.target.value as Status | "all")}
              >
                <option value="all">全部状态</option>
                {Object.entries(STATUS_LABEL).map(([status, label]) => (
                  <option key={status} value={status}>{label}</option>
                ))}
              </select>
            </label>
            <button className="button secondary" onClick={exportJSON}>
              <Download size={15} />导出全部 JSON
            </button>
            <button
              className="button secondary"
              onClick={() => {
                setReports([]);
                setFilterMode("all");
                setFilterStatus("all");
              }}
            >
              <RefreshCw size={15} />清除结果
            </button>
          </div>

          {reports.filter(report =>
            filterMode === "all" || report.mode === filterMode
          ).map(report => (
            <div className="five-signal-report" key={report.evaluation_id}>
              <header>
                <div>
                  <strong>{report.mode}</strong>
                  <span>{report.evaluation_id.slice(0, 12)} · {report.input_rate_hz} Hz</span>
                </div>
                <span className={`acoustic-status ${report.summary.fail ? "fail" : "review_required"}`}>
                  {report.summary.measured}/42 有工程值
                </span>
              </header>
              <dl>
                {[
                  ["参考→麦克风", report.analysis.values.reference_to_microphone_delay_ms, "ms"],
                  ["参考→近端", report.analysis.values.reference_to_output_delay_ms, "ms"],
                  ["SI-SDR 输入", report.analysis.values.microphone_sisdr_db, "dB"],
                  ["SI-SDR 输出", report.analysis.values.output_sisdr_db, "dB"],
                  ["STOI 输入", report.analysis.values.microphone_stoi, ""],
                  ["STOI 输出", report.analysis.values.output_stoi, ""],
                  ["噪声变化", report.analysis.values.noise_level_change_db, "dB"],
                  ["回声损耗", report.analysis.values.echo_loss_db, "dB"],
                  ["估计 SNR", report.analysis.values.output_snr_db, "dB"],
                  ["双讲透传", report.analysis.values.double_talk_transparent_percent, "%"],
                ].map(([label, value, unit]) => (
                  <div key={String(label)}>
                    <dt>{label}</dt>
                    <dd>{fmt(typeof value === "number" ? value : null, unit === "%" ? 1 : 2)} {unit}</dd>
                  </div>
                ))}
              </dl>
              {report.analysis.model_a?.scores && (
                <section className="five-signal-model-a">
                  <header>
                    <div>
                      <strong>ETSI TS 103 281 Model A · 实验工程值</strong>
                      <span>官方随机森林已执行，但当前实现未通过 Annex C ±0.1 MOS 门限</span>
                    </div>
                    <span className="acoustic-status fail">未合规 · 不可验收</span>
                  </header>
                  <div>
                    {["S-MOS", "N-MOS", "G-MOS"].map(metric => (
                      <article key={metric}>
                        <span>{metric}</span>
                        <strong>
                          {fmt(report.analysis.model_a?.scores?.[metric]?.value ?? null, 3)}
                        </strong>
                        <small>
                          MAE {fmt(
                            report.analysis.model_a?.verification?.summary?.[metric]?.mae ?? null,
                            3,
                          )}
                        </small>
                      </article>
                    ))}
                  </div>
                  <p>
                    官方 Annex C：
                    {report.analysis.model_a.verification?.passed_conditions ?? 0}/
                    {report.analysis.model_a.verification?.condition_count ?? 48}
                    组通过，最大允许误差 ±
                    {report.analysis.model_a.verification?.tolerance_mos ?? 0.1} MOS。
                    这些数值只用于工程诊断，不能作为正式合规或验收证据。
                  </p>
                  <details>
                    <summary>为什么不显示“通过”</summary>
                    <p>
                      {report.analysis.model_a.verification?.failed_reason ??
                        "尚未生成可追溯的官方向量验证证据。"}
                      公开 ZIP 只包含随机森林参数；完整 Model A 还依赖 Sottek
                      听觉模型、参考优化、迭代噪声反卷积及特征定义。
                    </p>
                  </details>
                </section>
              )}
            </div>
          ))}

          <div className="acoustic-item-table-wrap">
            <table className="acoustic-item-table five-signal-table">
              <thead>
                <tr>
                  <th>条目</th>
                  <th>测量项</th>
                  <th>状态</th>
                  <th>工程值</th>
                  <th>合同限值</th>
                  <th>源指标</th>
                  <th>说明</th>
                </tr>
              </thead>
              <tbody>
                {filteredItems.map(item => (
                  <tr key={item.item_id}>
                    <td><code>{item.item_id}</code></td>
                    <td>{item.test_name}</td>
                    <td><span className={`acoustic-status ${item.status}`}>{STATUS_LABEL[item.status]}</span></td>
                    <td>{fmt(item.value)} <small>{item.unit}</small></td>
                    <td>{item.lower_limit ?? "—"} … {item.upper_limit ?? "—"}</td>
                    <td><code>{item.metric ?? "—"}</code></td>
                    <td>{item.reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {filteredItems.length === 0 ? (
            <p>当前筛选没有结果。</p>
          ) : null}
        </>
      ) : (
        <div className="five-signal-empty">
          <AlertTriangle size={20} />
          <span>为至少一个模式选择完整五路音频后，可运行 42 项；八个模式齐全后一次生成 336 项。</span>
        </div>
      )}
    </section>
  );
}
