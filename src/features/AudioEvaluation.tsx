import { useEffect, useMemo, useState } from "react";
import { Download, CheckCircle2, CircleAlert } from "lucide-react";
import "../evaluation.css";
import ActiveDataset from "./ActiveDataset";
type Quality = { sisdr_db: number | null; stoi: number | null; rms_dbfs: number; peak_dbfs: number; clipping_percent: number; projection_gain_db: number | null };
type Segment = { before?: Quality; after?: Quality; hpf_matched_before?: Quality; hpf_matched_after?: Quality;
  sisdr_improvement_db?: number | null; hpf_matched_sisdr_improvement_db?: number | null; stoi_improvement?: number | null;
  total_attenuation_db?: number | null; attenuation_lower_bound_db?: number | null;
  linear_erle_engineering_db?: number | null; before_dbfs?: number; after_dbfs?: number };
type Run = { mode: string; segments: Record<string, Segment>; hpf_stages: number;
  engine: { process_p95_ms: number; process_max_ms: number; dsp_delay_samples: number; process_rtf: number } };
type Scene = { id: string; family: string; near_speaker: string; far_speaker: string; noise: string; snr_db: number | null;
  ser_db: number; rt60_s: number; echo_delay_ms: number; mics: number; near_dbfs: number; stress: string;
  seconds: number; audio_available: boolean; runs: Run[]; windows: Record<string, number[]>; noise_origin?: string;
  playback_source?: string; playback_reference_available?: boolean; playback_origin?: string };
type Group = { family: string; mode: string; runs: number; metrics: Record<string, { median: number }> };
type Gate = { segment: string; metric: string; value: number | null; operator: string; threshold: number | null;
  status: "pass" | "fail" | "unavailable"; reason: string | null; optional: boolean };
type Acceptance = { scenes: number; passed_scenes: number; counts: Record<string, number>; all_available_passed: boolean;
  controls_verified: boolean; required_measurements_complete: boolean; dataset_cli_sha256: string; rules_sha256: string;
  rules: { quality: Record<string, number>; suppression: Record<string, number>; level: Record<string, number>; runtime: Record<string, number> };
  negative_controls?: Record<string, { counts: Record<string, number>; verified: boolean }>;
  unavailable_capabilities: Record<string, string>; details: { id: string; passed: boolean; gates: Gate[] }[] };
type Catalog = { scenes: Scene[]; summary: { scenes: number; runs: number; groups: Group[] };
  acceptance?: Acceptance; batch?: { name: string; playable_scenes: number; profile?: string; evaluation_scope?: string }; sharded?: boolean;
  stratified?: { vehicle: { scenes: number; passed_scenes: number; counts: Record<string, number> };
    by_noise: { value: string; scenes: number; passed_scenes: number; counts: Record<string, number> }[];
    by_playback?: { value: string; scenes: number; passed_scenes: number; counts: Record<string, number> }[] };
    coverage?: { speakers: number; near_speakers: number; sex_counts: Record<string, number>; noise_types: string[]; snr_db: number[]; playback_sources?: string[] } };
type StreamReport = {
  identity: { seconds: number; frames: number; processing_profile: string; cli_sha256: string };
  engine: { process_p95_ms: number; process_max_ms: number; completion_deadline_misses: number | null; stream_resets: number; dsp_delay_samples: number };
  all_available_passed: boolean; required_measurements_complete: boolean; final_acceptance_eligible: boolean;
  details: { label: string; passed: boolean }[];
  failures: (Gate & { label: string })[];
};
type VehicleRegression = {
  sequences: number; phases: number; passed_phases: number; failed_gates: number;
  by_label: Record<string, { sequences: number; passed: number; failed_gates: number; unavailable_gates: number }>;
};
type VehicleArray = {
  sequences: number; phases: number; passed_phases: number;
  rows: { microphones: number; pair: number; phases: number; passed_phases: number;
    cold: Record<string, Segment>; failures: Gate[] }[];
};
const STREAM_LABEL: Record<string, string> = { cold_start_speech: "启动即说话", engine_change: "发动机变化", window_wind: "开窗风噪", radio_and_road: "收音机与路噪", long_radio_path: "180 ms 回声", reference_clock_drift: "300 ppm 时钟漂移", reference_mismatch: "播放参考错配", path_change: "回声路径变化", radio_restart: "收音机重启" };
const FAMILY: Record<string, string> = { ns: "降噪", aec: "回声消除", combined: "噪声＋回声", stress: "异常与变化", agc: "输入电平", bf: "多麦克风" };
const MODE: Record<string, string> = { full: "完整处理", bypass: "全部关闭", no_agc: "关闭 AGC", no_ns: "关闭 NS", no_aec: "关闭 AEC", aec_only: "仅 AEC", agc_only: "仅 AGC", beam_only: "仅 BF", no_bf: "关闭 BF", correct_reference: "正确播放参考" };
const NOISE: Record<string, string> = { none: "无噪声", white: "白噪声", pink: "粉红噪声", brown: "低频噪声", hum: "工频嗡声", burst: "突发噪声", babble: "多人交谈",
  dkitchen: "厨房", dliving: "客厅", ooffice: "办公室", ohallway: "走廊", omeeting: "会议室", dwashing: "洗衣房", straffic: "道路交通", spsquare: "广场", nfield: "运动场", npark: "公园", nriver: "溪流", pcafeter: "食堂", presto: "餐厅", pstation: "车站", tbus: "公交车内", tcar: "轿车内实录", tmetro: "地铁车内",
  car_hvac: "车内＋空调", car_window_wind: "车内＋开窗风噪", car_rain: "车内＋雨声", car_horn: "车内＋鸣笛", car_indicator: "车内＋转向灯", car_radio: "旧场景：无参考收音机干扰", car_engine_ramp: "车内＋发动机变化" };
const PLAYBACK: Record<string, string> = { none: "无播放", far_speech: "远端通话", radio: "收音机播放（AEC）" };
const playbackOf = (scene: Scene) => scene.playback_source ?? (scene.family === "ns" || scene.family === "agc" ? "none" : "far_speech");
const isVehicle = (scene: Scene) => scene.noise === "tcar" || scene.noise.startsWith("car_") || playbackOf(scene) === "radio";
const STRESS: Record<string, string> = { none: "无", path_change: "回声路径突变", clock_300ppm: "时钟漂移 300 ppm", wrong_reference: "播放参考错配", nonlinear_speaker: "扬声器非线性" };
const SEGMENT: Record<string, string> = { near_only: "近端语音", near_late: "后段近端", double_talk: "双讲", far_only: "远端单讲", far_recovery: "路径突变恢复", noise_only: "纯噪声", runtime: "逐帧耗时", whole_output: "完整输出", reference: "播放参考" };
const METRIC: Record<string, string> = { raw_sisdr_db: "原始 SI-SDR / dB", stoi_change: "STOI 变化", projection_gain_min_db: "语音投影增益下限 / dB", projection_gain_max_db: "语音投影增益上限 / dB", rms_min_dbfs: "RMS 下限 / dBFS", rms_max_dbfs: "RMS 上限 / dBFS", total_attenuation_db: "总衰减 / dB", attenuation_lower_bound_db: "PCM16 保守衰减下界 / dB", linear_erle_engineering_db: "线性 ERLE / dB", process_p95_ms: "P95 处理时间 / ms", process_max_ms: "最大处理时间 / ms", process_rtf: "处理 RTF", frames_over_10ms: "超过 10 ms 的帧数", peak_dbfs: "输出峰值 / dBFS", clipping_percent: "削波 / %", unreliable_frames: "参考不可靠检测帧数" };
const RULE: Record<string, string> = { near_sisdr_max_loss_db: "近端 SI-SDR 最大损失 / dB", near_sisdr_clean_ceiling_db: "近端 SI-SDR 基线封顶 / dB", stoi_max_loss: "STOI 最大损失", double_talk_sisdr_min_improvement_db: "双讲 SI-SDR 最低提升 / dB", wrong_reference_sisdr_max_loss_db: "错参考 SI-SDR 最大损失 / dB", projection_gain_min_db: "语音投影增益下限 / dB", projection_gain_max_db: "语音投影增益上限 / dB", noise_only_min_attenuation_db: "纯噪声最低衰减 / dB", far_only_min_attenuation_db: "远端单讲最低衰减 / dB", linear_erle_min_db: "无噪声线性 ERLE 下限 / dB", path_change_recovery_min_attenuation_db: "路径突变恢复最低衰减 / dB", wrong_reference_min_attenuation_db: "错参考最低衰减 / dB", near_rms_min_dbfs: "语音 RMS 下限 / dBFS", near_rms_max_dbfs: "语音 RMS 上限 / dBFS", output_peak_max_dbfs: "输出峰值上限 / dBFS", clipping_max_percent: "削波比例上限 / %", low_input_agc_min_projection_gain_db: "低电平 AGC 最低增益 / dB", low_input_agc_threshold_dbfs: "低电平 AGC 输入判据 / dBFS", process_p95_max_ms: "P95 处理时间上限 / ms", process_max_ms: "最大处理时间上限 / ms", frames_over_10ms: "超过 10 ms 的帧数上限", process_rtf_max: "处理 RTF 上限", bypass_max_pcm16_lsb_error: "关闭模块后 PCM16 最大误差 / LSB" };
const STATUS = { pass: "通过", fail: "未通过", unavailable: "不可用" };
const CONTROL: Record<string, string> = { "NS disabled": "关闭 NS · 降噪场景", "AEC disabled": "关闭 AEC · 回声场景", bypass: "全部关闭 · 纯噪声窗口" };
const unavailableLabel = (gate: Gate) => gate.reason === "no measurable input excitation" ? "无输入激励" : gate.optional ? "数值边界" : "必要测量缺失";
const fmt = (value: number | null | undefined, digits = 2) => value == null ? "—" : value.toFixed(digits);

export default function AudioEvaluation() {
  const [data, setData] = useState<Catalog | null>(null), [error, setError] = useState("");
  const [family, setFamily] = useState("all"), [selected, setSelected] = useState("004-ns-p0"), [mode, setMode] = useState("full");
  const [status, setStatus] = useState("all");
  const [batch, setBatch] = useState("no-passenger-current"), [batches, setBatches] = useState<{ id: string; label: string; path: string }[]>([{ id: "no-passenger-current", label: "当前车载测试集 · 已移除乘客讲话", path: "/evaluation/no-passenger-current" }, { id: "v3-native", label: "历史完整矩阵 · myserver 原生", path: "/evaluation/v3-native" }, { id: "v3-radio-final", label: "收音机专项验收", path: "/evaluation/v3-radio-final" }, { id: "v3-radio-optimized", label: "收音机 AEC 优化候选", path: "/evaluation/v3-radio-optimized" }, { id: "v3-radio", label: "收音机 AEC 修正专项", path: "/evaluation/v3-radio" }, { id: "v2", label: "扩展车载评测集（旧定义）", path: "/evaluation/v2" }, { id: "v1", label: "基础验收集", path: "/evaluation" }]);
  const [stream, setStream] = useState<StreamReport | null>(null), [streamError, setStreamError] = useState("");
  const [regression, setRegression] = useState<VehicleRegression | null>(null), [regressionError, setRegressionError] = useState("");
  const [array, setArray] = useState<VehicleArray | null>(null), [arrayError, setArrayError] = useState("");
  const [arrayPaced, setArrayPaced] = useState<StreamReport | null>(null);
  const [noise, setNoise] = useState("all"), [snr, setSnr] = useState("all"), [speaker, setSpeaker] = useState("all");
  const [playback, setPlayback] = useState("all");
  const [microphones, setMicrophones] = useState("all");
  const [detail, setDetail] = useState<{ scene: Scene; acceptance: { id: string; passed: boolean; gates: Gate[] } } | null>(null);
  const [detailError, setDetailError] = useState("");
  const base = batches.find(item => item.id === batch)?.path ?? "/evaluation";
  useEffect(() => {
    const abort = new AbortController();
    fetch("/evaluation/batches.json", { signal: abort.signal }).then(response => response.ok ? response.json() : null).then(result => { if (result?.batches?.length) setBatches(result.batches); }).catch(() => {});
    return () => abort.abort();
  }, []);
  useEffect(() => {
    if (batch === "no-passenger-current") return;
    const abort = new AbortController();
    fetch("/evaluation/vehicle-stream-v2/stream-evaluation.json", { signal: abort.signal }).then(async response => {
      if (!response.ok) throw Error("车载连续回放报告尚未就绪");
      setStream(await response.json());
    }).catch(e => { if (!abort.signal.aborted) setStreamError(e instanceof Error ? e.message : "连续回放报告加载失败"); });
    return () => abort.abort();
  }, [batch]);
  useEffect(() => {
    if (batch === "no-passenger-current") return;
    const abort = new AbortController();
    fetch("/evaluation/vehicle-regression-v2/regression-summary.json", { signal: abort.signal }).then(async response => {
      if (!response.ok) throw Error("车载说话人回归报告尚未就绪");
      setRegression(await response.json());
    }).catch(e => { if (!abort.signal.aborted) setRegressionError(e instanceof Error ? e.message : "车载回归报告加载失败"); });
    return () => abort.abort();
  }, [batch]);
  useEffect(() => {
    if (batch === "no-passenger-current") return;
    const abort = new AbortController();
    fetch("/evaluation/vehicle-array-v2/array-summary.json", { signal: abort.signal }).then(async response => {
      if (!response.ok) throw Error("多麦车载对比报告尚未就绪");
      setArray(await response.json());
    }).catch(e => { if (!abort.signal.aborted) setArrayError(e instanceof Error ? e.message : "多麦对比报告加载失败"); });
    return () => abort.abort();
  }, [batch]);
  useEffect(() => {
    if (batch === "no-passenger-current") return;
    const abort = new AbortController();
    fetch("/evaluation/vehicle-array-v2/paced-4ch.json", { signal: abort.signal })
      .then(async response => { if (response.ok) setArrayPaced(await response.json()); })
      .catch(() => {});
    return () => abort.abort();
  }, [batch]);
  useEffect(() => {
    if (batch === "no-passenger-current") return;
    const abort = new AbortController();
    setData(null); setDetail(null); setError(""); setDetailError("");
    fetch(`${base}/catalog.json`, { signal: abort.signal }).then(async response => {
      if (!response.ok) throw Error("评测数据加载失败，请确认数据已发布。");
      setData(await response.json());
    }).catch(e => { if (!abort.signal.aborted) setError(e instanceof Error ? e.message : "评测数据加载失败"); });
    return () => abort.abort();
  }, [base]);
  const verdicts = useMemo(() => new Map(data?.acceptance?.details.map(item => [item.id, item]) ?? []), [data]);
  const scenes = useMemo(() => data?.scenes.filter(scene => (family === "all" || scene.family === family) &&
    (status === "all" || verdicts.get(scene.id)?.passed === (status === "pass")) &&
    (noise === "all" || (noise === "vehicle" ? isVehicle(scene) : scene.noise === noise)) &&
    (playback === "all" || playbackOf(scene) === playback) &&
    (microphones === "all" || scene.mics === Number(microphones)) &&
    (snr === "all" || String(scene.snr_db) === snr) && (speaker === "all" || scene.near_speaker === speaker)) ?? [], [data, family, status, verdicts, noise, snr, speaker, playback, microphones]);
  const indexed = scenes.find(scene => scene.id === selected) ?? scenes[0];
  useEffect(() => {
    setDetail(null); setDetailError("");
    if (!data?.sharded || !indexed) return;
    const abort = new AbortController();
    fetch(`${base}/scene-data/${indexed.id}.json`, { signal: abort.signal }).then(async response => {
      if (!response.ok) throw Error("场景指标加载失败");
      setDetail(await response.json());
    }).catch(e => { if (!abort.signal.aborted) setDetailError(e instanceof Error ? e.message : "场景指标加载失败"); });
    return () => abort.abort();
  }, [base, data?.sharded, indexed?.id]);
  const resetFilters = () => { setStatus("all"); setFamily("all"); setNoise("all"); setSnr("all"); setSpeaker("all"); setPlayback("all"); setMicrophones("all"); setSelected(""); setMode("full"); };
  const batchControl = <label className="evaluation-batch">评测批次<select aria-label="评测批次" value={batch} onChange={e => { setBatch(e.target.value); resetFilters(); }}>{batches.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>;
  if (batch === "no-passenger-current") return <div className="evaluation-page">{batchControl}<ActiveDataset /></div>;
  if (error) return <div className="evaluation-page">{batchControl}<section className="evaluation-empty" role="alert"><p>{error}</p><button className="button secondary" onClick={() => window.location.reload()}>重新加载</button></section></div>;
  if (!data) return <div className="evaluation-page">{batchControl}<section className="evaluation-empty" role="status">正在加载仿真评测结果…</section></div>;
  const acceptance = data.acceptance;
  const loaded = data.sharded && detail?.scene.id === indexed?.id ? detail : null;
  const scene = loaded?.scene ?? indexed;
  const checkedScene = data.sharded ? loaded?.acceptance : verdicts.get(scene?.id ?? "");
  const problems = acceptance?.details.flatMap(item => item.gates.filter(gate => gate.status === "fail" || (gate.status === "unavailable" && !gate.optional)).map(gate => ({ id: item.id, gate }))) ?? [];
  const run = scene?.runs.find(run => run.mode === mode) ?? scene?.runs[0];
  const filters = <div className="evaluation-filters audio-config">
    <label>麦克风路数<select aria-label="麦克风路数" value={microphones} onChange={e => { setMicrophones(e.target.value); setMode("full"); }}><option value="all">全部路数</option>{[1, 2, 4].map(count => <option key={count} value={count}>{count} 路麦克风</option>)}</select></label>
    <label>背景噪声<select aria-label="背景噪声" value={noise} onChange={e => { setNoise(e.target.value); setMode("full"); }}><option value="all">全部背景噪声</option><option value="vehicle">全部车载条件</option>{[...new Set(data.scenes.map(item => item.noise))].sort().map(id => <option key={id} value={id}>{NOISE[id] ?? id}</option>)}</select></label>
    <label>播放来源<select aria-label="播放来源" value={playback} onChange={e => { setPlayback(e.target.value); setMode("full"); }}><option value="all">全部播放来源</option>{[...new Set(data.scenes.map(playbackOf))].sort().map(id => <option key={id} value={id}>{PLAYBACK[id] ?? id}</option>)}</select></label>
    <label>SNR<select aria-label="SNR" value={snr} onChange={e => { setSnr(e.target.value); setMode("full"); }}><option value="all">全部档位</option>{[...new Set(data.scenes.map(item => item.snr_db))].sort((a,b) => (a ?? 1000)-(b ?? 1000)).map(value => <option key={String(value)} value={String(value)}>{value === null ? "不适用" : `${value} dB`}</option>)}</select></label>
    <label>近端说话人<select aria-label="近端说话人" value={speaker} onChange={e => { setSpeaker(e.target.value); setMode("full"); }}><option value="all">全部说话人</option>{[...new Set(data.scenes.map(item => item.near_speaker))].sort().map(id => <option key={id} value={id}>{id}</option>)}</select></label>
    <label>技术维度<select aria-label="技术维度" value={family} onChange={e => { setFamily(e.target.value); setMode("full"); }}><option value="all">全部维度</option>{Object.entries(FAMILY).map(([id, name]) => <option key={id} value={id}>{name}</option>)}</select></label>
    {acceptance ? <label>验收状态<select aria-label="验收状态" value={status} onChange={e => { setStatus(e.target.value); setMode("full"); }}><option value="all">全部状态</option><option value="pass">通过</option><option value="fail">未通过</option></select></label> : null}
    <label>场景 · {scenes.length} 个<select aria-label="场景" disabled={!scene} value={scene?.id ?? ""} onChange={e => { setSelected(e.target.value); setMode("full"); }}>{scenes.map(scene => <option key={scene.id} value={scene.id}>{scene.id} · {NOISE[scene.noise]}{playbackOf(scene) === "radio" ? " / 收音机 AEC" : ""}{scene.snr_db !== null ? ` / SNR ${scene.snr_db} dB` : ""}{scene.audio_available ? " · 可试听" : ""}</option>)}</select></label>
    <label>处理模式<select aria-label="处理模式" disabled={!run} value={run?.mode ?? ""} onChange={e => setMode(e.target.value)}>{scene?.runs.map(run => <option value={run.mode} key={run.mode}>{MODE[run.mode]}</option>)}</select></label>
  </div>;
  if (!scene || !run) return <div className="evaluation-page">{batchControl}{filters}<section className="evaluation-empty" role={detailError ? "alert" : "status"}><p>{!scene ? "该筛选条件下没有场景。" : detailError || "正在加载所选场景指标…"}</p>{!scene ? <button className="button secondary" onClick={resetFilters}>查看全部场景</button> : detailError ? <button className="button secondary" onClick={() => window.location.reload()}>重新加载</button> : null}</section></div>;
  const near = run.segments.near_only, dt = run.segments.double_talk;
  const url = `${base}/audio/${scene.id}`;
  const rows = [
    ["近端 SI-SDR · 原始参考 / dB", near.before?.sisdr_db, near.after?.sisdr_db, near.sisdr_improvement_db],
    ["近端 SI-SDR · 高通匹配 / dB", near.hpf_matched_before?.sisdr_db, near.hpf_matched_after?.sisdr_db, near.hpf_matched_sisdr_improvement_db],
    ["近端 STOI", near.before?.stoi, near.after?.stoi, near.stoi_improvement],
    ["近端 RMS 电平 / dBFS", near.before?.rms_dbfs, near.after?.rms_dbfs, near.before && near.after ? near.after.rms_dbfs - near.before.rms_dbfs : null],
    ["近端峰值 / dBFS", near.before?.peak_dbfs, near.after?.peak_dbfs, null],
    ["近端满幅样本 / %", near.before?.clipping_percent, near.after?.clipping_percent, null],
    ["双讲 SI-SDR · 原始参考 / dB", dt?.before?.sisdr_db, dt?.after?.sisdr_db, dt?.sisdr_improvement_db],
    ["双讲 SI-SDR · 高通匹配 / dB", dt?.hpf_matched_before?.sisdr_db, dt?.hpf_matched_after?.sisdr_db, dt?.hpf_matched_sisdr_improvement_db],
    ["双讲 STOI", dt?.before?.stoi, dt?.after?.stoi, dt?.stoi_improvement],
    ["纯噪声电平 dBFS / 总衰减 dB", run.segments.noise_only.before_dbfs, run.segments.noise_only.after_dbfs, run.segments.noise_only.total_attenuation_db],
    ["远端单讲电平 dBFS / 总衰减 dB", run.segments.far_only?.before_dbfs, run.segments.far_only?.after_dbfs, run.segments.far_only?.total_attenuation_db],
    ["远端单讲 PCM16 衰减下界 / dB", null, null, run.segments.far_only?.attenuation_lower_bound_db],
    ["纯噪声 PCM16 衰减下界 / dB", null, null, run.segments.noise_only.attenuation_lower_bound_db],
    ["线性 AEC ERLE 工程估计 / dB", null, null, run.segments.far_only?.linear_erle_engineering_db],
  ] as const;
  return <div className="evaluation-page">
    {batchControl}
    {batch === "v3-native" ? <section className="evaluation-acceptance" aria-label="车载连续流式结果">
      <div className="evaluation-verdict" data-status={stream?.all_available_passed && stream.required_measurements_complete ? "pass" : "fail"}><h2>车载连续流式回放</h2><span>{stream ? `${stream.details.filter(item => item.passed).length} / ${stream.details.length} 阶段通过` : streamError || "正在读取连续回放报告"}</span></div>
      {stream ? <><dl className="evaluation-acceptance-stats">{[["连续音频", `${stream.identity.seconds} s`], ["10 ms 帧", stream.identity.frames], ["处理实例重置", stream.engine.stream_resets], ["P95 / ms", fmt(stream.engine.process_p95_ms, 3)], ["最大 / ms", fmt(stream.engine.process_max_ms, 3)], ["完成期限错过", stream.engine.completion_deadline_misses ?? "未测"]].map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl><div className="table-scroll"><table className="audio-table"><caption>连续回放未通过项</caption><thead><tr><th>阶段</th><th>窗口</th><th>指标</th><th>实测</th><th>要求</th></tr></thead><tbody>{stream.failures.map(gate => <tr key={`${gate.label}-${gate.segment}-${gate.metric}`}><td>{STREAM_LABEL[gate.label] ?? gate.label}</td><td>{SEGMENT[gate.segment] ?? gate.segment}</td><td>{METRIC[gate.metric] ?? gate.metric}</td><td>{fmt(gate.value, 3)}</td><td>{gate.operator} {fmt(gate.threshold, 3)}</td></tr>)}</tbody></table></div><div className="evaluation-links"><a href="/evaluation/vehicle-stream-v2/stream-evaluation.json" download>连续评测 JSON</a><a href="/evaluation/vehicle-stream-v2/timeline.json" download>时间轴与输入标识</a><a href="/evaluation/vehicle-stream-v2/stream.frames.csv" download>逐帧数据 CSV</a><a href="/evaluation/vehicle-stream-v2/stream.after.wav" download>连续处理后音频</a></div><p className="evaluation-note">同一实例持续处理，启动无静音学习要求。{stream.final_acceptance_eligible ? "已按 10 ms 节拍验证。" : "仅音质诊断，尚未验证实时完成期限。"} 当前为主机仿真，Android 设备连续运行尚未验收。</p></> : null}
    </section> : null}
    {batch === "v3-native" ? <section className="evaluation-section" aria-label="车载说话人连续回归">
      <div className="panel-heading"><h2>车载说话人连续回归</h2><span>{regression ? `${regression.passed_phases} / ${regression.phases} 阶段通过` : regressionError || "正在读取说话人回归报告"}</span></div>
      {regression ? <><p className="evaluation-note">{regression.sequences} 位说话人，每人连续 259 秒；共 {regression.failed_gates} 个失败指标。本批为并行音质回归，未验证实时完成期限。</p><div className="table-scroll"><table className="audio-table"><thead><tr><th>阶段</th><th>通过 / 说话人</th><th>失败指标</th><th>不可用值</th></tr></thead><tbody>{Object.entries(regression.by_label).map(([label, row]) => <tr key={label}><td>{STREAM_LABEL[label] ?? label}</td><td>{row.passed} / {row.sequences}</td><td>{row.failed_gates}</td><td>{row.unavailable_gates}</td></tr>)}</tbody></table></div><div className="evaluation-links"><a href="/evaluation/vehicle-regression-v2/regression-summary.json" download>全部说话人回归 JSON</a><a href="/evaluation/vehicle-regression-v2/identity.json" download>回归输入与版本标识</a></div></> : null}
    </section> : null}
    {batch === "v3-native" ? <section className="evaluation-section" aria-label="多麦车载连续对比">
      <div className="panel-heading"><h2>1 / 2 / 4 路车载连续对比</h2><span>{array ? `${array.passed_phases} / ${array.phases} 阶段通过` : arrayError || "正在读取多麦对比报告"}</span></div>
      {array ? <><div className="table-scroll"><table className="audio-table"><thead><tr><th>麦克风</th><th>说话人组</th><th>通过阶段</th><th>启动 SI-SDR / dB</th></tr></thead><tbody>{array.rows.map(row => <tr key={`${row.microphones}-${row.pair}`}><td>{row.microphones} 路</td><td>{row.pair}</td><td>{row.passed_phases} / {row.phases}</td><td>{fmt(row.cold.near_only.after?.sisdr_db)}</td></tr>)}</tbody></table></div><p className="evaluation-note">两位说话人，六个连续 259 秒序列。2 / 4 路使用独立声学到达时间与传感器噪声；BF 使用固定几何补偿。处理前以第 0 路干净语音为基线，处理后使用阵列目标。此批为并行音质诊断，未完成全部说话人、实时期限或实机阵列验收。</p><div className="evaluation-links"><a href="/evaluation/vehicle-array-v2/array-summary.json" download>多麦对比 JSON</a><a href="/evaluation/vehicle-array-v2/artifacts.sha256" download>输入与处理版本标识</a></div></> : null}
      {arrayPaced ? <><p className="evaluation-note">四麦单实例节拍验证：{arrayPaced.details.filter(row => row.passed).length} / {arrayPaced.details.length} 阶段通过。P95 {fmt(arrayPaced.engine.process_p95_ms, 3)} ms（门槛 1 ms），最大 {fmt(arrayPaced.engine.process_max_ms, 3)} ms，完成期限错过 {arrayPaced.engine.completion_deadline_misses ?? "未测"}。尚未满足全部实时门槛。</p><a href="/evaluation/vehicle-array-v2/paced-4ch.json" download>四麦节拍评测 JSON</a></> : null}
    </section> : null}
    {batch === "v3-radio-final" ? <div className="audio-info">收音机专项验收：{acceptance?.passed_scenes ?? 0} / {acceptance?.scenes ?? data.summary.scenes} 场景通过。使用独立收音机处理预设；完整扩展矩阵与 Android 实机声学尚未验收。<a href={`${base}/RADIO_OPTIMIZATION_REPORT.md`} download>优化回归报告</a></div> : null}
    {batch === "v3-radio-optimized" ? <div className="audio-info">收音机专项候选：473 / 480 场景通过，仍有 7 个失败项。该配置在基础集存在退化，未启用为通用默认链。<a href={`${base}/RADIO_OPTIMIZATION_REPORT.md`} download>优化回归报告</a></div> : null}
    {data.batch?.profile === "playback-check-v3" ? <div className="audio-info">收音机 AEC 修正专项：440 个播放场景＋40 个车内降噪对照。完整 30 类背景噪声矩阵尚未重跑。</div> : batch === "v2" ? <div className="audio-info">历史批次：收音机曾作为无参考背景干扰。该部分结果不代表有正确播放参考的 AEC 性能。</div> : null}
    <div className="audio-info">真实语音与环境噪声经模拟回声和阵列混合后，由 C++ WebRTC 逐帧处理。声学路径和多麦克风信号为仿真，结果不等于 Android 实机声学验收。</div>
    {acceptance ? <section className="evaluation-acceptance" aria-label="工程验收结果">
      <div className="evaluation-verdict" data-status={acceptance.all_available_passed ? "pass" : "fail"}>
        {acceptance.all_available_passed ? <CheckCircle2 size={20} aria-hidden="true" /> : <CircleAlert size={20} aria-hidden="true" />}
        <h2>{acceptance.all_available_passed ? "全部可用指标达标" : "工程验收未全部达标"}</h2>
        <span>{acceptance.passed_scenes} / {acceptance.scenes} 场景通过</span>
      </div>
      <dl className="evaluation-acceptance-stats">{[["通过项", acceptance.counts.pass ?? 0], ["失败项", acceptance.counts.fail ?? 0], ["不可用值", acceptance.counts.unavailable ?? 0], ["对照验证", acceptance.controls_verified ? "已验证" : "未验证"], ["必要测量", acceptance.required_measurements_complete ? "齐全" : "缺失"], ["批次", data.batch?.name ?? "未标记"]].map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>
      <p className="evaluation-note">冻结工程规则 v1 · SI-SDR / STOI 使用原始参考验收。数字静音的精确衰减仍为不可用；PCM16 保守衰减下界独立检验同一门槛。DTD 分类准确率、MOS / POLQA 均不可用，未计为通过。阈值不代表 ITU 通用合格线。</p>
      {problems.length > 0 ? <div className="table-scroll"><table className="audio-table"><caption>未通过项</caption><thead><tr><th>场景</th><th>窗口</th><th>指标</th><th>实测</th><th>要求</th></tr></thead><tbody>{problems.map(({ id, gate }) => <tr key={`${id}-${gate.segment}-${gate.metric}`}><td><button className="evaluation-scene-link" onClick={() => { resetFilters(); setSelected(id); }}>{id}</button></td><td>{SEGMENT[gate.segment] ?? gate.segment}</td><td>{METRIC[gate.metric] ?? gate.metric}</td><td>{fmt(gate.value, gate.metric.includes("stoi") ? 4 : 3)}</td><td>{gate.operator} {fmt(gate.threshold, gate.metric.includes("stoi") ? 4 : 3)}</td></tr>)}</tbody></table></div> : data.sharded && acceptance.counts.fail > 0 ? <div className="evaluation-links"><button className="evaluation-scene-link" onClick={() => { resetFilters(); setStatus("fail"); }}>查看未通过场景</button><a href={`${base}/acceptance.csv`} download>下载全部未通过项及验收值</a></div> : null}
      {acceptance.negative_controls ? <details className="evaluation-rules"><summary>关闭模块的负对照验证</summary><div className="table-scroll"><table className="audio-table"><thead><tr><th>对照</th><th>检出抑制失败</th><th>仍满足抑制</th><th>不可用窗口</th><th>验证</th></tr></thead><tbody>{Object.entries(acceptance.negative_controls).map(([name, check]) => <tr key={name}><td>{CONTROL[name] ?? name}</td><td>{check.counts.expected_failure ?? 0}</td><td>{check.counts.unexpected_pass ?? 0}</td><td>{check.counts.unavailable ?? 0}</td><td>{check.verified ? "已检出" : "未检出"}</td></tr>)}</tbody></table></div></details> : null}
      <details className="evaluation-rules"><summary>全部公开阈值与逐场景验收</summary>
        <div className="table-scroll"><table className="audio-table"><caption className="sr-only">冻结工程验收阈值</caption><thead><tr><th>类别</th><th>规则</th><th>数值</th></tr></thead><tbody>{(["quality", "suppression", "level", "runtime"] as const).flatMap(group => Object.entries(acceptance.rules[group]).map(([key, value]) => <tr key={`${group}-${key}`}><td>{{ quality: "语音保真", suppression: "抑制", level: "电平", runtime: "实时性" }[group]}</td><td>{RULE[key] ?? key}</td><td>{value}</td></tr>))}</tbody></table></div>
        <p className="evaluation-note">近端要求：SI-SDR ≥ min(处理前 SI-SDR, {acceptance.rules.quality.near_sisdr_clean_ceiling_db}) − {acceptance.rules.quality.near_sisdr_max_loss_db} dB。错误参考验收以语音保护为目标，需检测参考不可靠；含噪声远端窗口测总衰减。详细适用条件和所有逐场景数值保留在下载文件中。</p>
        <div className="evaluation-links"><a href={`${base}/acceptance-v1.json`} download>冻结阈值 JSON</a><a href={`${base}/acceptance.csv`} download>逐场景验收 CSV</a><a href={`${base}/acceptance.json`} download>完整验收 JSON</a><a href={`${base}/MEASUREMENT_NOTES.md`} download>测量定义与不可用边界</a></div>
        <dl className="evaluation-identities"><dt>处理产物 SHA256</dt><dd>{acceptance.dataset_cli_sha256}</dd><dt>阈值 SHA256</dt><dd>{acceptance.rules_sha256}</dd></dl>
      </details>
    </section> : <p className="audio-info">本批次尚未发布逐场景工程验收报告。</p>}
    <div className="comparison-cards">{[["场景", data.summary.scenes, "噪声、回声、双讲、电平与阵列"], ["C++ 处理次数", data.summary.runs, "含关闭模块和错误参考对照"], ["真实语音说话人", data.coverage?.speakers ?? new Set(data.scenes.flatMap(item => [item.near_speaker, item.far_speaker])).size, "LibriSpeech · 16 kHz / 10 ms"]].map(([label, value, note]) => <section className="panel audio-highlight" key={label}><h3>{label}</h3><strong>{value}</strong><p>{note}</p></section>)}</div>
    {data.coverage ? <><dl className="evaluation-conditions"><div><dt>背景噪声条件</dt><dd>{data.coverage.noise_types.length} 类</dd></div><div><dt>SNR 点</dt><dd>{data.coverage.snr_db.join(" / ")} dB</dd></div><div><dt>AEC 播放来源</dt><dd>{data.coverage.playback_sources?.map(id => PLAYBACK[id] ?? id).join(" / ") ?? "远端通话（旧批次）"}</dd></div><div><dt>近端说话人</dt><dd>{data.coverage.near_speakers} 人 · 男 {data.coverage.sex_counts.M ?? 0} / 女 {data.coverage.sex_counts.F ?? 0}</dd></div></dl><p className="evaluation-note">整体峰值保护可能同时降低语音电平；背景噪声 SNR 和播放回声 SER 分别报告。</p></> : null}
    {data.stratified ? <section className="evaluation-section" aria-label="车载评测结果"><div className="panel-heading"><h2>车载条件评测</h2><span>{data.stratified.vehicle.passed_scenes} / {data.stratified.vehicle.scenes} 场景通过</span></div><div className="table-scroll"><table className="audio-table"><thead><tr><th>车载条件</th><th>通过 / 全部场景</th><th>失败项</th><th>不可用值</th></tr></thead><tbody>{data.stratified.by_noise.filter(row => row.value === "tcar" || row.value.startsWith("car_")).map(row => <tr key={row.value}><td>{NOISE[row.value]}</td><td>{row.passed_scenes} / {row.scenes}</td><td>{row.counts.fail ?? 0}</td><td>{row.counts.unavailable ?? 0}</td></tr>)}</tbody></table></div><div className="evaluation-links"><a href={`${base}/COVERAGE_REPORT.md`} download>噪声、SNR 与说话人分组报告</a><a href={`${base}/coverage-summary.json`} download>分组结果 JSON</a></div></section> : null}
    {data.stratified?.by_playback ? <section className="evaluation-section" aria-label="播放来源评测结果"><div className="panel-heading"><h2>AEC 播放来源评测</h2></div><div className="table-scroll"><table className="audio-table"><thead><tr><th>播放来源</th><th>通过 / 全部场景</th><th>失败项</th><th>不可用值</th></tr></thead><tbody>{data.stratified.by_playback.filter(row => row.value !== "none").map(row => <tr key={row.value}><td>{PLAYBACK[row.value] ?? row.value}</td><td>{row.passed_scenes} / {row.scenes}</td><td>{row.counts.fail ?? 0}</td><td>{row.counts.unavailable ?? 0}</td></tr>)}</tbody></table></div></section> : null}
    <section className="panel evaluation-section">
      <div className="panel-heading"><div><h2>选择评测场景</h2><p>全部场景结果 · {data.batch?.playable_scenes ?? 8} 组音频可试听和下载</p></div><a className="button secondary" href={`${base}/metrics.csv`} download><Download size={16} aria-hidden="true" />下载全部指标 CSV</a></div>
      {filters}
      <dl className="evaluation-conditions">{[["近端 / 远端说话人", `${scene.near_speaker} / ${scene.far_speaker}`], ["背景噪声 / SNR", `${NOISE[scene.noise]} / ${scene.snr_db === null ? "不适用" : `${scene.snr_db} dB`}`], ["播放来源 / 参考", `${PLAYBACK[playbackOf(scene)] ?? playbackOf(scene)} / ${scene.playback_reference_available === true ? "对应播放参考" : scene.playback_reference_available === false ? "无有效参考" : "旧批次记录"}`], ["回声参数", scene.family === "ns" || scene.family === "agc" ? "无回声" : `SER ${scene.ser_db} dB · 附加延时 ${scene.echo_delay_ms} ms · RT60 ${scene.rt60_s} s`], ["麦克风 / 标称电平", `${scene.mics} 路 / ${scene.near_dbfs} dBFS`], ["异常条件", STRESS[scene.stress]], ["近端片段 / 总长", `${scene.windows.near_only.join("–")} s / ${scene.seconds} s`]].map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>
      <div className="table-scroll"><table className="audio-table"><caption className="sr-only">所选场景处理前后的配对指标</caption><thead><tr><th>指标</th><th>处理前</th><th>处理后</th><th>变化 / 抑制</th></tr></thead><tbody>{rows.map(([label, ...values]) => <tr key={label}><td>{label}</td>{values.map((value, index) => <td key={index}>{index < 2 && label.includes("dBFS") && value != null && value <= -200 ? "≤ −200" : fmt(value, label.includes("STOI") ? 4 : 2)}</td>)}</tr>)}</tbody></table></div>
      {scene.noise_origin ? <p className="evaluation-note">噪声来源：{scene.noise === "none" ? "无噪声分量" : scene.noise.startsWith("car_") ? "DEMAND 车内实录＋可控干扰混合；无实录车速或空调档位标注。" : scene.noise_origin === "real recording" ? "真实环境录音" : scene.noise_origin === "four independent recorded speakers" ? "四位独立说话人的录音混合" : "合成信号"}</p> : null}
      {playbackOf(scene) === "radio" ? <p className="evaluation-note">收音机内容由真实语音带限生成，经模拟扬声器声学路径形成回声；参考为对应播放信号，背景噪声中不含该播放内容。</p> : null}
      <p className="evaluation-note">{run.hpf_stages === 0 ? "当前链路关闭 WebRTC IIR 高通；高通匹配阶段为 0，与原始指标相同，不构成独立诊断。固定线性相位低切延时已补偿。" : `原始 SI-SDR 包含高通相位影响；匹配诊断使用同一固定高通，共 ${run.hpf_stages} 次。`} 不搜索最佳时延。ERLE 仅显示无噪声远端单讲的线性留样。“—”表示条件缺失、数值边界或不适用；SI-SDR 的 120 表示计算上限；“≤ −200”表示数字静音或计算下界，不是设备校准电平。</p>
      <p className="evaluation-note numeric">C++ P95 {fmt(run.engine.process_p95_ms, 3)} ms · 最大 {fmt(run.engine.process_max_ms, 3)} ms · RTF {fmt(run.engine.process_rtf, 4)} · DSP 延时 {run.engine.dsp_delay_samples / 16} ms。耗时排除文件读写及采集。</p>
      {checkedScene ? <details className="evaluation-rules"><summary>本场景完整处理验收 · {checkedScene.passed ? "通过" : "未通过"}</summary><div className="table-scroll"><table className="audio-table"><thead><tr><th>窗口</th><th>指标</th><th>实测</th><th>要求</th><th>状态</th></tr></thead><tbody>{checkedScene.gates.map(gate => <tr key={`${gate.segment}-${gate.metric}`}><td>{SEGMENT[gate.segment] ?? gate.segment}</td><td>{METRIC[gate.metric] ?? gate.metric}</td><td>{fmt(gate.value, gate.metric.includes("stoi") ? 4 : 3)}</td><td>{gate.operator} {fmt(gate.threshold, gate.metric.includes("stoi") ? 4 : 3)}</td><td className={`evaluation-status-${gate.status}`}>{STATUS[gate.status]}{gate.status === "unavailable" ? ` · ${unavailableLabel(gate)}` : ""}</td></tr>)}</tbody></table></div></details> : null}
    </section>
    <section className="panel evaluation-section">
      <div className="panel-heading"><div><h2>四路音频试听</h2><p>处理后音频对应“完整处理”模式；使用原始电平，不自动归一化。</p></div>{scene.audio_available ? <a className="button secondary" download href={`${url}/paired-audio.zip`}>下载本组配对音频</a> : null}</div>
      {scene.audio_available ? <div className="evaluation-audio">{[["clean.wav", "Clean · 干净近端目标"], ["ref.wav", "Ref · AEC 播放参考"], ["before.wav", "Before · 麦克风混合输入"], ["denoise.wav", "Denoise · C++ 处理后"]].map(([file, label]) => <div key={`${scene.id}-${file}`}><h3>{label}</h3><audio controls preload="none" aria-label={label} src={`${url}/${file}`} /><a download href={`${url}/${file}`}>下载 WAV</a></div>)}</div> : <div className="evaluation-empty"><p>此场景保留全部指标；选择标有“可试听”的场景可下载四路配对音频。</p><button className="button secondary" onClick={() => { resetFilters(); setSelected(data.scenes.find(item => item.audio_available)?.id ?? ""); }}>查看代表音频</button></div>}
      {scene.audio_available ? <p className="evaluation-note">clean / before / denoise / ref 对应工作台的干净参考 / 处理前 / 处理后 / 回声参考。ZIP 包含测量片段配置、来源与许可证；参考错配场景故意使用错误 ref。</p> : null}
    </section>
    <section className="panel evaluation-section">
      <div className="panel-heading"><div><h2>各维度汇总 · 完整处理</h2><p>各场景中位数；原始与高通匹配 SI-SDR 同时保留</p></div></div>
      <div className="table-scroll"><table className="audio-table"><thead><tr><th>维度</th><th>场景数</th><th>原始 SI-SDR Δ / dB</th><th>匹配 SI-SDR Δ / dB</th><th>STOI Δ</th><th>远端总衰减 / dB</th></tr></thead><tbody>{data.summary.groups.filter(group => group.mode === "full").map(group => <tr key={group.family}><td>{FAMILY[group.family]}</td><td>{group.runs}</td>{["near_only.sisdr_improvement_db", "near_only.hpf_matched_sisdr_improvement_db", "near_only.stoi_improvement", "far_only.total_attenuation_db"].map(key => <td key={key}>{fmt(group.metrics[key]?.median, key.includes("stoi") ? 4 : 2)}</td>)}</tr>)}</tbody></table></div>
      <p className="evaluation-note">LibriSpeech 英语朗读，CC BY 4.0；DEMAND 环境录音，CC BY-SA 3.0；MS-SNSD 空调录音按上游 CC0。车载组合含可控合成干扰，并非带车速标注的车内实录。SI-SDR/STOI 不等于 MOS 或 ITU 合规评分。</p>
      <div className="evaluation-links"><a href="https://www.openslr.org/12">LibriSpeech 来源</a><a href="https://zenodo.org/records/1227121">DEMAND 来源</a><a href={`${base}/评测结论.md`} download>中文评测结论</a><a href={`${base}/QUALITY_REPAIR_REPORT.md`} download>根因与修复验收报告</a><a href={`${base}/REPORT.md`} download>完整仿真报告</a><a href={`${base}/summary.json`} download>汇总 JSON</a><a href={`${base}/ATTRIBUTION.md`} download>来源与许可</a></div>
    </section>
  </div>;
}
