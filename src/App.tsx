import { useEffect, useState } from "react";
import {
  ArrowDownToLine,
  ArrowRight,
  CheckCircle2,
  ChevronRight,
  FlaskConical,
  LoaderCircle,
  Play,
  RotateCcw,
  X,
  Workflow,
  Car,
} from "lucide-react";
import Sidebar, { NAV } from "./components/Sidebar";
import type { View } from "./components/Sidebar";
import AudioUpload from "./components/AudioUpload";
import { EmptyState } from "./components/Shared";
import AudioResults, { Highlights } from "./features/AudioResults";
import AudioCompare from "./features/AudioCompare";
import MetricLibrary from "./features/MetricLibrary";
import AudioFlow from "./features/AudioFlow";
import AudioEvaluation from "./features/AudioEvaluation";
import EcnrEvaluation from "./features/EcnrEvaluation";
import ItuAlignment from "./features/ItuAlignment";
import {
  AUDIO_ROLES,
  EMPTY_FILES,
  ROLE_LABEL,
  REPORT_KEY,
  audioCsv,
  loadAudioReport,
} from "./domain/audio";
import type {
  AudioFiles,
  AudioOptions,
  AudioReport,
  AudioRole,
  AudioWindow,
} from "./domain/audio";
import { downloadFile } from "./lib/files";
import "./audio.css";

type AppView = View | "ecnr-v2";
function viewFromHash(): AppView {
  const value = window.location.hash.slice(1);
  if (value === "ecnr-v2") return "ecnr-v2";
  return NAV.some((item) => item.id === value) ? (value as View) : "workbench";
}
const COPY = {
  "ecnr-v2": ["ECNR 协同优化实验", "对照固定基线，查看语音保真、噪声抑制和连续运行的实际结果。"],
  workbench: [
    "从音频，找到改善的依据",
    "上传参考与录音，计算时延、语音保真、噪声和回声抑制。",
  ],
  results: [
    "音频计算结果",
    "查看实际计算值、完整波形与频谱，以及每项指标的有效条件。",
  ],
  workflow: ["音频从哪里来，指标如何得到", "沿着信号路径，查看录音位置、处理节点与计算条件。"],
  car3d: ["智能座舱与车载声学", "探索高保真座舱 3D 数字孪生、原厂车漆定制与主驾顶棚双麦拾音波束透视。"],
  compare: ["处理前后对比", "把噪声变化与语音保真放在一起，看清优化的代价。"],
  metrics: ["指标与计算方法", "音频工程计算与原表 17 类实验室指标的对应关系。"],
  evaluation: ["多维音频仿真评测", "浏览真实语音素材的配对数据集，比较不同场景与处理模式。"],
  alignment: ["ITU-T 对齐验收", "车载连续流、工程门限与正式声学测项。"],
};
type RangeState = { start: string; end: string };
function RangeInputs({
  title,
  value,
  onChange,
  disabled,
}: {
  title: string;
  value: RangeState;
  onChange: (value: RangeState) => void;
  disabled: boolean;
}) {
  return (
    <div className="range-inputs">
      <label>
        <span>开始 / s</span>
        <input
          type="number"
          min="0"
          step="0.01"
          aria-label={`${title}开始秒`}
          value={value.start}
          disabled={disabled}
          onChange={(e) => onChange({ ...value, start: e.target.value })}
        />
      </label>
      <span className="range-arrow">→</span>
      <label>
        <span>结束 / s</span>
        <input
          type="number"
          min="0"
          step="0.01"
          aria-label={`${title}结束秒`}
          value={value.end}
          placeholder="文件末尾"
          disabled={disabled}
          onChange={(e) => onChange({ ...value, end: e.target.value })}
        />
      </label>
    </div>
  );
}
function parseWindow(value: RangeState, title: string): AudioWindow {
  const start = value.start.trim() === "" ? NaN : Number(value.start);
  const end = value.end.trim() === "" ? null : Number(value.end);
  if (
    !Number.isFinite(start) ||
    start < 0 ||
    start > 120 ||
    (end !== null && (!Number.isFinite(end) || end <= start || end > 120))
  )
    throw Error(`${title}：请输入有效时间，结束需大于开始，范围为 0–120 秒。`);
  return { start, end };
}
export default function App() {
  const [view, setView] = useState<AppView>(viewFromHash);
  const [files, setFiles] = useState<AudioFiles>({ ...EMPTY_FILES });
  const [channels, setChannels] = useState<Record<AudioRole, number>>({
    reference: 0,
    before: 0,
    after: 0,
    echo: 0,
  });
  const [report, setReport] = useState<AudioReport | null>(loadAudioReport);
  const [rate, setRate] = useState("16000"),
    [maxDelay, setMaxDelay] = useState("1000"),
    [offset, setOffset] = useState("0");
  const [speech, setSpeech] = useState<RangeState>({ start: "0", end: "" });
  const [noise, setNoise] = useState<RangeState>({ start: "0", end: "" });
  const [far, setFar] = useState<RangeState>({ start: "0", end: "" });
  const [noiseEnabled, setNoiseEnabled] = useState(false),
    [farEnabled, setFarEnabled] = useState(false);
  const [commonClock, setCommonClock] = useState(false),
    [farOnly, setFarOnly] = useState(false);
  const [stationary, setStationary] = useState(false),
    [isSpeech, setIsSpeech] = useState(false);
  const [aecOnly, setAecOnly] = useState(false);
  const [busy, setBusy] = useState(false),
    [loadingDemo, setLoadingDemo] = useState(false);
  const [error, setError] = useState(""),
    [notice, setNotice] = useState("");
  const [service, setService] = useState<"checking" | "online" | "offline">(
    "checking",
  );
  useEffect(() => {
    const sync = () => {
      setView(viewFromHash());
      window.scrollTo({ top: 0 });
    };
    window.addEventListener("hashchange", sync);
    return () => window.removeEventListener("hashchange", sync);
  }, []);
  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/health", { signal: controller.signal })
      .then((r) => {
        setService(r.ok ? "online" : "offline");
      })
      .catch(() => {
        if (!controller.signal.aborted) setService("offline");
      });
    return () => controller.abort();
  }, []);
  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(""), 6500);
    return () => clearTimeout(timer);
  }, [notice]);
  function invalidate() {
    setReport(null);
    setError("");
    try {
      localStorage.removeItem(REPORT_KEY);
    } catch {
      /* Results can still be exported. */
    }
  }
  function clearConfirmations() {
    setCommonClock(false);
    setFarOnly(false);
    setAecOnly(false);
    setStationary(false);
    setIsSpeech(false);
  }
  function changeFile(role: AudioRole, file: File | null) {
    invalidate();
    clearConfirmations();
    setFiles((old) => ({ ...old, [role]: file }));
    setChannels((old) => ({ ...old, [role]: 0 }));
  }
  async function demo() {
    setLoadingDemo(true);
    setError("");
    try {
      const fetched = await Promise.all(
        AUDIO_ROLES.map(async (role) => {
          const response = await fetch(`/api/demo/${role}`);
          if (!response.ok) throw Error("示例音频暂不可用，请检查计算服务。");
          return [
            role,
            new File([await response.blob()], `示例_${ROLE_LABEL[role]}.wav`, {
              type: "audio/wav",
            }),
          ] as const;
        }),
      );
      invalidate();
      setFiles(Object.fromEntries(fetched) as AudioFiles);
      setChannels({ reference: 0, before: 0, after: 0, echo: 0 });
      setSpeech({ start: "0.4", end: "3.1" });
      setNoise({ start: "5", end: "6" });
      setFar({ start: "3.4", end: "4.7" });
      setRate("16000");
      setMaxDelay("1000");
      setOffset("0");
      setNoiseEnabled(true);
      setFarEnabled(true);
      setCommonClock(true);
      setFarOnly(true);
      setStationary(false);
      setIsSpeech(false);
      setAecOnly(false);
      setService("online");
      setNotice(
        "已载入四路合成测试音频。点击开始计算验证 80 ms 时延；此示例不是语音，未启用 STOI。",
      );
    } catch (e) {
      setService("offline");
      setError(e instanceof Error ? e.message : "示例加载失败");
    } finally {
      setLoadingDemo(false);
    }
  }
  async function calculate() {
    if (!files.after) {
      setError("请先上传麦克风处理后音频。");
      return;
    }
    setBusy(true);
    setError("");
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 90000);
    try {
      const delay = Number(maxDelay),
        compensation = Number(offset);
      if (
        !maxDelay.trim() ||
        !offset.trim() ||
        !Number.isFinite(delay) ||
        delay < 10 ||
        delay > 2000 ||
        !Number.isFinite(compensation) ||
        Math.abs(compensation) > 2000
      )
        throw Error("时延搜索需为 10–2000 ms，处理后补偿需为 −2000–2000 ms。");
      const options: AudioOptions = {
        sample_rate: Number(rate),
        channels,
        speech: parseWindow(speech, "参考比较片段"),
        noise: noiseEnabled ? parseWindow(noise, "噪声片段") : null,
        far_end: farEnabled ? parseWindow(far, "远端单讲片段") : null,
        max_delay_ms: delay,
        after_offset_ms: compensation,
        common_clock: commonClock,
        far_end_only: farOnly,
        aec_only: aecOnly,
        stationary_noise: stationary,
        speech_confirmed: isSpeech,
      };
      const body = new FormData();
      AUDIO_ROLES.forEach((role) => {
        if (files[role]) body.append(role, files[role]);
      });
      body.append("options", JSON.stringify(options));
      const response = await fetch("/api/analyze", {
        method: "POST",
        body,
        signal: controller.signal,
      });
      if (!response.ok) {
        const failure = await response.json().catch(() => null);
        throw Error(
          typeof failure?.detail === "string"
            ? failure.detail
            : "计算未完成，请检查音频文件与片段配置。",
        );
      }
      const result = (await response.json()) as AudioReport;
      setReport(result);
      setService("online");
      try {
        localStorage.setItem(REPORT_KEY, JSON.stringify(result));
      } catch {
        setNotice("计算完成；浏览器无法保存结果，请导出报告。");
      }
      window.location.hash = "results";
    } catch (e) {
      setError(
        e instanceof DOMException && e.name === "AbortError"
          ? "计算超时，请缩短音频后重试。"
          : e instanceof Error
            ? e.message
            : "计算服务连接失败，请检查服务是否启动。",
      );
    } finally {
      clearTimeout(timer);
      setBusy(false);
    }
  }
  const disabled = busy || loadingDemo;
  const count = AUDIO_ROLES.filter((role) => files[role]).length;
  return (
    <div className="app-shell">
      <Sidebar
        view={view === "ecnr-v2" ? "evaluation" : view}
        sourceName={
          count
            ? `${count} 路实际音频`
            : report
              ? "最近一次音频计算"
              : "等待上传实际音频"
        }
      />
      <main className="main">
        <header className="topbar">
          <div className="breadcrumb">
            <span>NV200</span>
            <ChevronRight size={13} />
            <span>{view === "ecnr-v2" ? "ECNR 协同优化" : NAV.find((n) => n.id === view)?.label}</span>
          </div>
          <div className={`audio-service service-${service}`}>
            <i />
            {service === "online"
              ? "计算服务已连接"
              : service === "offline"
                ? "计算服务未连接"
                : "连接计算服务…"}
          </div>
        </header>
        <div className="main-content">
          <div className="page-header">
            <div>
              <h1>{COPY[view][0]}</h1>
              <p>{COPY[view][1]}</p>
            </div>
            <div className="header-actions">
              {report ? (
                <>
                  <button
                    className="button secondary"
                    onClick={() =>
                      downloadFile(
                        audioCsv(report),
                        "NV200_音频计算报告.csv",
                        "text/csv;charset=utf-8",
                      )
                    }
                  >
                    <ArrowDownToLine size={15} />
                    导出 CSV
                  </button>
                  <button
                    className="button secondary"
                    onClick={() =>
                      downloadFile(
                        JSON.stringify(report, null, 2),
                        "NV200_音频计算报告.json",
                        "application/json",
                      )
                    }
                  >
                    <ArrowDownToLine size={15} />
                    完整报告
                  </button>
                </>
              ) : (
                <span className="audio-version">音频分析 / v2.0</span>
              )}
            </div>
          </div>
          {view === "workbench" ? (
            <>
              <a href="#car3d" className="audio-flow-entry" style={{ borderColor: "rgba(56, 189, 248, 0.4)", background: "rgba(14, 165, 233, 0.08)", marginBottom: 12 }}>
                <Car size={20} aria-hidden="true" style={{ color: "#38bdf8" }} />
                <div>
                  <strong>智能座舱与车载声学仿真</strong>
                  <span>体验高保真座舱 3D 数字孪生、原厂专属车漆与主驾顶棚双麦拾音波束。</span>
                </div>
                <ArrowRight size={17} aria-hidden="true" />
              </a>
              <a href="#workflow" className="audio-flow-entry">
                <Workflow size={20} aria-hidden="true" />
                <div>
                  <strong>录音与指标流程图</strong>
                  <span>了解四路音频怎么录，以及降噪、回声消除在哪里完成。</span>
                </div>
                <ArrowRight size={17} aria-hidden="true" />
              </a>
              <div className="audio-workflow">
                <div className="active">
                  <span>01</span>
                  <strong>选择实际音频</strong>
                </div>
                <ArrowRight size={15} />
                <div>
                  <span>02</span>
                  <strong>确认片段与条件</strong>
                </div>
                <ArrowRight size={15} />
                <div>
                  <span>03</span>
                  <strong>计算与比较</strong>
                </div>
              </div>
              <div className="audio-section-title">
                <div>
                  <h2>
                    录音与参考信号 <span>{count} / 4</span>
                  </h2>
                  <p>
                    支持 WAV、FLAC、MP3、OGG、M4A、AAC、AIFF。优先使用未压缩
                    WAV。
                  </p>
                </div>
                <button
                  className="button secondary"
                  disabled={disabled}
                  onClick={demo}
                >
                  {loadingDemo ? (
                    <LoaderCircle size={15} className="spin" />
                  ) : (
                    <FlaskConical size={15} />
                  )}
                  载入示例音频
                </button>
              </div>
              <div className="audio-upload-grid">
                {AUDIO_ROLES.map((role) => (
                  <AudioUpload
                    key={role}
                    role={role}
                    file={files[role]}
                    channel={channels[role]}
                    info={report?.files.find((f) => f.role === role)}
                    disabled={disabled}
                    onFile={(file) => changeFile(role, file)}
                    onChannel={(channel) => {
                      invalidate();
                      clearConfirmations();
                      setChannels((old) => ({ ...old, [role]: channel }));
                    }}
                  />
                ))}
              </div>
              <p className="audio-upload-note">
                音频会发送到当前网站的计算服务，用于本次分析；服务不保留原文件。浏览器仅保存最近一份计算报告，刷新后需重新选择音频。
              </p>
              <section className="panel audio-config">
                <div className="panel-heading">
                  <div>
                    <h2>片段与计算条件</h2>
                    <p>
                      时间单位为秒。噪声与回声使用麦克风处理前的时间轴；处理后可指定延迟补偿。
                    </p>
                  </div>
                </div>
                <fieldset disabled={disabled}>
                  <div className="audio-config-top">
                    <label>
                      <span>成对分析采样率</span>
                      <select
                        aria-label="成对分析采样率"
                        value={rate}
                        onChange={(e) => {
                          invalidate();
                          setRate(e.target.value);
                        }}
                      >
                        <option value="8000">8 kHz / NB 工程分析</option>
                        <option value="16000">16 kHz / WB 工程分析</option>
                        <option value="24000">24 kHz / 工程分析</option>
                        <option value="32000">32 kHz / SWB 工程分析</option>
                        <option value="48000">48 kHz / 全带宽工程分析</option>
                      </select>
                    </label>
                    <label>
                      <span>时延搜索 / ms</span>
                      <input
                        type="number"
                        aria-label="最大时延搜索毫秒"
                        min="10"
                        max="2000"
                        value={maxDelay}
                        onChange={(e) => {
                          invalidate();
                          setMaxDelay(e.target.value);
                        }}
                      />
                    </label>
                    <label>
                      <span>处理后补偿 / ms</span>
                      <input
                        type="number"
                        aria-label="处理后延迟补偿毫秒"
                        min="-2000"
                        max="2000"
                        value={offset}
                        onChange={(e) => {
                          invalidate();
                          setOffset(e.target.value);
                        }}
                      />
                    </label>
                  </div>
                  <p className="config-helper">
                    采样率统一用于成对计算，文件信息保持原始值。补偿为正时，噪声/回声片段在处理后文件中向后移；相对时延仍独立测量。
                  </p>
                  <div className="audio-windows">
                    <div className="window-card">
                      <h3>参考比较片段</h3>
                      <p>相对时延 / SI-SDR / STOI / 传递函数</p>
                      <RangeInputs
                        title="参考比较片段"
                        value={speech}
                        disabled={disabled}
                        onChange={(value) => {
                          invalidate();
                          setSpeech(value);
                        }}
                      />
                    </div>
                    <div className="window-card">
                      <label className="window-enable">
                        <input
                          type="checkbox"
                          checked={noiseEnabled}
                          onChange={(e) => {
                            invalidate();
                            setNoiseEnabled(e.target.checked);
                          }}
                        />
                        噪声片段
                      </label>
                      <p>无近端语音、无扬声器回声</p>
                      <RangeInputs
                        title="噪声片段"
                        value={noise}
                        disabled={disabled || !noiseEnabled}
                        onChange={(value) => {
                          invalidate();
                          setNoise(value);
                        }}
                      />
                    </div>
                    <div className="window-card">
                      <label className="window-enable">
                        <input
                          type="checkbox"
                          checked={farEnabled}
                          onChange={(e) => {
                            invalidate();
                            setFarEnabled(e.target.checked);
                          }}
                        />
                        远端单讲片段
                      </label>
                      <p>只有扬声器发声，用于回声片段衰减</p>
                      <RangeInputs
                        title="远端单讲片段"
                        value={far}
                        disabled={disabled || !farEnabled}
                        onChange={(value) => {
                          invalidate();
                          setFar(value);
                        }}
                      />
                    </div>
                  </div>
                  <div className="audio-confirmations">
                    <label>
                      <input
                        type="checkbox"
                        checked={commonClock}
                        onChange={(e) => {
                          invalidate();
                          setCommonClock(e.target.checked);
                        }}
                      />
                      <span>
                        处理前后来自同一信号，时钟、录音电平一致；片段已对应，或已填写处理后补偿。
                      </span>
                    </label>
                    <label>
                      <input
                        type="checkbox"
                        checked={farOnly}
                        disabled={!farEnabled}
                        onChange={(e) => {
                          invalidate();
                          setFarOnly(e.target.checked);
                        }}
                      />
                      <span>
                        远端单讲片段没有近端语音，且处理前包含扬声器回声。（启用回声片段总衰减）
                      </span>
                    </label>
                    <label>
                      <input
                        type="checkbox"
                        checked={aecOnly}
                        disabled={!farEnabled || !farOnly}
                        onChange={(e) => {
                          invalidate();
                          setAecOnly(e.target.checked);
                        }}
                      />
                      <span>
                        前后录音为 AEC
                        滤波器的输入/输出，未包含后级非线性抑制、降噪或自动增益。（启用
                        ERLE 工程估计）
                      </span>
                    </label>
                    <label>
                      <input
                        type="checkbox"
                        checked={isSpeech}
                        onChange={(e) => {
                          invalidate();
                          setIsSpeech(e.target.checked);
                        }}
                      />
                      <span>
                        干净参考的比较片段为真实语音。（启用
                        STOI；非语音测试信号请勿勾选）
                      </span>
                    </label>
                    <label>
                      <input
                        type="checkbox"
                        checked={stationary}
                        disabled={!noiseEnabled}
                        onChange={(e) => {
                          invalidate();
                          setStationary(e.target.checked);
                        }}
                      />
                      <span>
                        噪声平稳，比较片段包含语音与同一噪声，噪声与语音不相关。（启用片段估计
                        SNR）
                      </span>
                    </label>
                  </div>
                </fieldset>
                <div className="audio-compute-bar">
                  <div>
                    <strong>
                      {files.after
                        ? "已具备基本计算输入"
                        : "等待麦克风处理后音频"}
                    </strong>
                    <span>缺少参考仍可计算电平、满幅比例和频谱。</span>
                  </div>
                  <button
                    className="button primary compute-button"
                    disabled={disabled || !files.after}
                    onClick={calculate}
                  >
                    {busy ? (
                      <LoaderCircle className="spin" size={18} />
                    ) : (
                      <Play size={16} />
                    )}
                    {busy ? "正在读取与计算…" : "开始计算"}
                  </button>
                </div>
              </section>
              {error ? (
                <div className="audio-error" role="alert">
                  {error}
                </div>
              ) : null}
              {report ? (
                <>
                  <div className="audio-section-title">
                    <div>
                      <h2>最近一次计算</h2>
                      <p>
                        {count
                          ? "当前音频的实际计算结果"
                          : "已保存的报告；重新计算需再次选择原音频"}
                      </p>
                    </div>
                    <a href="#results" className="text-link">
                      查看完整结果
                      <ArrowRight size={14} />
                    </a>
                  </div>
                  <Highlights report={report} />
                </>
              ) : null}
              <div className="audio-info">
                相对时延、SI-SDR、STOI 与 ERLE 是音频工程指标。原表的
                MOS、SLR/RLR、TCL
                和合规曲线需要专用算法、测试配置或标定；在“指标与方法”中查看对应条件。
              </div>
            </>
          ) : view === "car3d" ? (
            <div style={{ width: "100%", height: "calc(100vh - 140px)", borderRadius: 16, overflow: "hidden", border: "1px solid rgba(255,255,255,0.12)", background: "#07090e" }}>
              <iframe src="/car-3d.html" style={{ width: "100%", height: "100%", border: "none" }} title="3D 智能座舱与车门交互" />
            </div>
          ) : view === "workflow" ? (
            <AudioFlow files={files} />
          ) : view === "ecnr-v2" ? (
            <EcnrEvaluation />
          ) : view === "evaluation" ? (
            <>
              <a className="button secondary ecnr-entry" href="#ecnr-v2">查看 ECNR 协同优化实验<ArrowRight size={15} /></a>
              <AudioEvaluation />
            </>
          ) : view === "alignment" ? (
            <ItuAlignment />
          ) : view === "metrics" ? (
            <MetricLibrary />
          ) : report ? (
            view === "results" ? (
              <AudioResults report={report} />
            ) : (
              <AudioCompare report={report} />
            )
          ) : (
            <section className="panel">
              <EmptyState
                title="还没有音频计算结果"
                description="先上传麦克风处理后音频；添加参考与处理前录音，可以进行成对比较。"
                action={
                  <a href="#workbench" className="button primary">
                    选择实际音频
                    <ArrowRight size={15} />
                  </a>
                }
              />
            </section>
          )}
          {report && (view === "results" || view === "compare") ? (
            <div className="audio-report-notices">
              {report.notices.map((note) => (
                <p key={note}>{note}</p>
              ))}
            </div>
          ) : null}
          <footer className="app-footer">
            <span>
              ITU Audio Lab <span>·</span> 从实际音频计算
            </span>
            {count || report ? (
              <button
                disabled={disabled}
                onClick={() => {
                  invalidate();
                  setFiles({ ...EMPTY_FILES });
                  setChannels({ reference: 0, before: 0, after: 0, echo: 0 });
                  setSpeech({ start: "0", end: "" });
                  setNoise({ start: "0", end: "" });
                  setFar({ start: "0", end: "" });
                  setRate("16000");
                  setMaxDelay("1000");
                  setOffset("0");
                  setNoiseEnabled(false);
                  setFarEnabled(false);
                  setCommonClock(false);
                  setFarOnly(false);
                  setAecOnly(false);
                  setStationary(false);
                  setIsSpeech(false);
                  window.location.hash = "workbench";
                  setNotice("本次音频与计算结果已清空。");
                }}
              >
                <RotateCcw size={13} />
                清空本次分析
              </button>
            ) : (
              <a href="/downloads/metric-design.md" download>
                原表指标设计
              </a>
            )}
          </footer>
        </div>
      </main>
      {notice ? (
        <div className="toast" role="status">
          <CheckCircle2 size={18} />
          <span>{notice}</span>
          <button
            className="icon-button"
            aria-label="关闭通知"
            onClick={() => setNotice("")}
          >
            <X size={14} />
          </button>
        </div>
      ) : null}
    </div>
  );
}
