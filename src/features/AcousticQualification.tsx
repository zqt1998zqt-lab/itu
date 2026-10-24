import { useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  ChevronRight,
  Download,
  FileCheck2,
  FileUp,
  FlaskConical,
  LoaderCircle,
  Play,
  RefreshCw,
  ShieldCheck,
  XCircle,
} from "lucide-react";
import "./acoustic-qualification.css";

type Mode = "simulation" | "precompliance" | "formal";
type Bandwidth = "NB" | "WB" | "SWB" | "FB";
type Transport = "wired" | "wireless" | "bt";
type ResultStatus =
  | "pass" | "fail" | "review_required" | "invalid" | "not_measured";

type Artifact = { path: string; sha256: string; kind: string };
type EvidenceRecord = {
  artifact: Artifact;
  original_name: string;
  size_bytes: number;
  content_type: string;
  uploaded_at: string;
};
type Gate = {
  gate: string;
  passed: boolean;
  detail: string;
  blocking_for: Mode[];
};
type Standard = {
  recommendation: string;
  edition: string;
  official_url: string;
  full_text_status: "verified_local" | "not_present_locally";
  formal_text_gate: string;
};
type RunRecord = {
  run_id: string;
  created_at: string;
  status: string;
  gate_status: "ready" | "blocked";
  requested_mode: Mode;
  effective_mode: Mode;
  standard: Standard;
  contract: { path: string; sha256: string; rows_for_selection: number };
  request_sha256: string;
  gates: Gate[];
  blockers: string[];
  measured_items: number;
  not_measured_items: number;
  standard_compliance: false;
  limitation: string;
};
type ItemResult = {
  item_id: string;
  family: string;
  test_name: string;
  unit: string;
  status: ResultStatus;
  value: number | null;
  lower_limit: number | string | null;
  upper_limit: number | string | null;
  margin: number | null;
  reason: string;
};
type ExecutionRecord = {
  execution_id: string;
  run_id: string;
  created_at: string;
  status: string;
  mode: Mode;
  backend: string;
  message: string;
  result: Artifact | null;
  summary: Record<ResultStatus, number>;
  items: ItemResult[];
  standard_compliance: false;
};
type EvidenceKind =
  | "standard_text" | "review" | "hardware_calibration"
  | "method_package" | "polqa_build" | "p501_signal_pack"
  | "limit_reconciliation"
  | "measurement_bundle" | "calibration" | "noise_scenario"
  | "unclassified";
type CompileKind = "loopback" | "transfer" | "noise";
type TransferMaskKey = "SWB_send" | "FB_send" | "SWB_receive" | "FB_receive";
type TransferMask = {
  frequency_hz: number[];
  lower_db: number[];
  upper_db: number[];
  normalized_at_hz: number;
  clause: string;
};

const P1120_MASKS = {
  SWB_send: {
    clause: "P.1120 §11.4.1 · SWB send",
    frequency_hz: [80, 125, 8000, 12500, 14000],
    upper_db: [4, 4, 4, 4, 4],
    lower_db: [-10, -4, -4, -7, -7],
    normalized_at_hz: 1000,
  },
  FB_send: {
    clause: "P.1120 §11.4.1 · FB send",
    frequency_hz: [50, 100, 12500, 16000],
    upper_db: [4, 4, 4, 4],
    lower_db: [-10, -4, -4, -7],
    normalized_at_hz: 1000,
  },
  SWB_receive: {
    clause: "P.1120 §11.4.2 · SWB receive",
    frequency_hz: [80, 125, 250, 400, 8000, 12500, 14000],
    upper_db: [8, 8, 8, 4, 4, 4, 4],
    lower_db: [-10, -4, -4, -4, -4, -7, -7],
    normalized_at_hz: 1000,
  },
  FB_receive: {
    clause: "P.1120 §11.4.2 · FB receive",
    frequency_hz: [50, 100, 12500, 16000],
    upper_db: [4, 4, 4, 4],
    lower_db: [-10, -4, -4, -7],
    normalized_at_hz: 1000,
  },
} satisfies Record<string, TransferMask>;

const STAGES = [
  ["01", "夹具与声场", "P.58 HATS、MRP/DRP/HFRP、车辆声学区、四扬声器加超低音"],
  ["02", "标定与噪声场", "校准点、电平、频响、RT60、NC20/NC30 和 C1–C5"],
  ["03", "DUT 与协议", "固件、音量、codec/网络、POI/SRW、物理时间戳回环"],
  ["04", "信号与方法", "P.501、P.863/POLQA、ETSI TS 103 281、方法版本与哈希"],
  ["05", "声学测量", "RTD、SLR/RLR、频响、TCL、双讲、空闲噪声、MOS、CNG"],
  ["06", "执行与签署", "逐项 trace、失败保留、独立复核、不可变运行快照"],
] as const;

const METRICS = [
  ["发送/接收响度", "SLR / RLR", "dB", "P.1120 §11.3 · 校准 MRP/POI/DRP"],
  ["时延", "Ts / Tr / RTD", "ms", "P.1120 §11.2 · 物理回环与时间戳"],
  ["频响", "Tx / Rx masks", "dB", "P.1120 §11.4 · 三分之一倍频程与掩模"],
  ["语音质量", "MOS-LQO", "MOS", "P.1120 §11.5/11.6 · P.863"],
  ["回声", "TCL / TCLw / ERL", "dB", "P.1120 §11.11 · 扬声器至麦克风路径"],
  ["双讲", "AH,S,dt / AH,R,dt", "dB", "P.1120 §11.12/11.13 · 规定序列"],
  ["背景噪声", "N/S/G-MOS", "MOS-LQO", "P.1120 §11.14.3 · ETSI TS 103 281"],
  ["舒适噪声", "CNG level/spectrum", "dB", "P.1120 §11.14.6 · 只在适用时"],
] as const;

const EVENTS: Array<[EvidenceKind, string]> = [
  ["standard_text", "标准正文 / 条款核对"],
  ["limit_reconciliation", "客户限值条款映射"],
  ["hardware_calibration", "硬件与声场标定"],
  ["calibration", "校准点记录"],
  ["noise_scenario", "C1–C5 噪声场"],
  ["method_package", "测量方法包"],
  ["polqa_build", "P.863/POLQA 身份"],
  ["p501_signal_pack", "P.501 信号包"],
  ["measurement_bundle", "测量结果包"],
  ["review", "独立复核"],
];

const CALIBRATION_POINTS = [
  "HATS_MRP", "HATS_DRP", "HFRP", "POI_SEND", "POI_RECEIVE",
] as const;

const NOISE_SCENARIOS = [
  {
    id: "C1", label: "静止怠速", speed: 0, fan: "lowest",
    airflow: "record actual setting", windows: "up",
    road: "not applicable", optional: false,
  },
  {
    id: "C2", label: "城市道路 60 km/h", speed: 60, fan: "medium",
    airflow: "toward windows", windows: "up",
    road: "dry rough road; no impulse bumps", optional: false,
  },
  {
    id: "C3", label: "高速 120 km/h · 低风量", speed: 120,
    fan: "lowest", airflow: "record actual setting", windows: "up",
    road: "dry rough road; no impulse bumps", optional: false,
  },
  {
    id: "C4", label: "高速 120 km/h · 中风量", speed: 120,
    fan: "medium", airflow: "toward windows", windows: "up",
    road: "dry rough road; no impulse bumps", optional: false,
  },
  {
    id: "C5", label: "高速 ≥160 km/h · 可选", speed: 160,
    fan: "lowest", airflow: "toward windows", windows: "up",
    road: "dry rough road; no impulse bumps", optional: true,
  },
] as const;

function defaultPayload(mode: Mode, bandwidth: Bandwidth, transport: Transport) {
  return {
    mode,
    confirm_formal: false,
    bandwidth,
    transport,
    fixture: {
      fixture_id: "hats-lab-a",
      fixture_type: "HATS",
      room: "simulated_vehicle",
      zone_id: "driver",
      hats_position: "manufacturer marks; driver zone",
      background_noise_dba: -70,
      reverberation_rt60_ms: 150,
      calibrated_at: new Date().toISOString(),
      calibration_points: [] as Array<Record<string, unknown>>,
      noise_scenarios: [] as Array<Record<string, unknown>>,
      evidence: [] as Artifact[],
    },
    dut: {
      name: "Xiaomi SU7 head unit",
      firmware: "candidate",
      interface: "acoustic",
      network_profiles: ["VoLTE / EVS / 48 kHz"],
      volume_setting: "nominal",
      audio_path: "HATS → DUT → network simulator → POI",
      evidence: [] as Artifact[],
    },
    hardware: {
      backend: "imported",
      capabilities: [
        "playback", "capture", "loopback", "timestamps",
        "network_impairment", "level_calibration", "hats_rotation",
        "background_noise_control", "four_loudspeaker_noise",
        "subwoofer", "measurement_import",
      ],
      calibration_id: "pending-calibration",
      execution_identity: "external-test-house",
    },
    method: {
      package_id: "pending-method-package",
      p835_version: "latest approved",
      signal_pack_id: "P501-current",
    },
  };
}

function selectedTransferMask(
  key: TransferMaskKey,
  normalizedAtHz: number,
): TransferMask | null {
  const mask = P1120_MASKS[key];
  if (
    !Number.isFinite(normalizedAtHz) ||
    normalizedAtHz < mask.frequency_hz[0] ||
    normalizedAtHz > mask.frequency_hz[mask.frequency_hz.length - 1]
  ) return null;
  return { ...mask, normalized_at_hz: normalizedAtHz };
}

async function responseJSON(response: Response): Promise<any> {
  const body = await response.json();
  if (!response.ok) {
    const detail = Array.isArray(body.detail)
      ? body.detail.map((item: { msg?: string }) => item.msg).filter(Boolean).join("；")
      : body.detail;
    throw Error(detail || `HTTP ${response.status}`);
  }
  return body;
}

async function withEvidenceKind(
  row: EvidenceRecord,
  kind: string,
): Promise<EvidenceRecord> {
  return responseJSON(await fetch("/api/acoustic/evidence/classify", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ artifact: row.artifact, kind }),
  }));
}

export function AcousticQualification() {
  const [mode, setMode] = useState<Mode>("simulation");
  const [bandwidth, setBandwidth] = useState<Bandwidth>("SWB");
  const [transport, setTransport] = useState<Transport>("wireless");
  const [confirmFormal, setConfirmFormal] = useState(false);
  const [result, setResult] = useState<RunRecord | null>(null);
  const [history, setHistory] = useState<RunRecord[]>([]);
  const [execution, setExecution] = useState<ExecutionRecord | null>(null);
  const [evidence, setEvidence] = useState<Record<string, EvidenceRecord>>({});
  const [evidenceKind, setEvidenceKind] = useState<EvidenceKind>("standard_text");
  const [evidenceNamespace, setEvidenceNamespace] = useState("");
  const [compileKind, setCompileKind] = useState<CompileKind>("loopback");
  const [compileReference, setCompileReference] = useState("");
  const [compileCapture, setCompileCapture] = useState("");
  const [compileItem, setCompileItem] = useState("SWB-wireless:B2");
  const [compileUnit, setCompileUnit] = useState("Calculated Value [ms]");
  const [compileStart, setCompileStart] = useState("0");
  const [compileEnd, setCompileEnd] = useState("1");
  const [compileMaskKey, setCompileMaskKey] =
    useState<TransferMaskKey>("SWB_send");
  const [compileNormalizedHz, setCompileNormalizedHz] = useState("1000");
  const [calibrationEvidence, setCalibrationEvidence] = useState<Record<string, EvidenceRecord>>({});
  const [noiseEvidence, setNoiseEvidence] = useState<Record<string, EvidenceRecord>>({});
  const [dutEvidence, setDutEvidence] = useState<EvidenceRecord | null>(null);
  const [busy, setBusy] = useState(false);
  const [executing, setExecuting] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const activeRunId = useRef<string | null>(null);
  const runRequest = useRef<{
    kind: "plan" | "execute";
    controller: AbortController;
    runId?: string;
  } | null>(null);

  function cancelRunRequest() {
    const task = runRequest.current;
    if (!task) return;
    runRequest.current = null;
    task.controller.abort();
    if (task.kind === "plan") setBusy(false);
    else setExecuting(false);
  }

  function selectRun(run: RunRecord) {
    cancelRunRequest();
    activeRunId.current = run.run_id;
    setResult(run);
    setExecution(null);
    setError("");
    setNotice("");
  }

  useEffect(() => () => {
    runRequest.current?.controller.abort();
    runRequest.current = null;
    activeRunId.current = null;
  }, []);

  const loadHistory = async () => {
    const body = await responseJSON(
      await fetch("/api/acoustic/runs", { cache: "no-store" }),
    );
    setHistory(body);
  };

  useEffect(() => {
    loadHistory().catch(reason =>
      setError(reason instanceof Error ? reason.message : "无法读取声学运行记录"),
    );
  }, []);

  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(""), 5000);
    return () => clearTimeout(timer);
  }, [notice]);

  const passCount = result?.gates.filter(gate => gate.passed).length ?? 0;
  const gateTotal = result?.gates.length ?? 0;
  const allChecksPass = result?.gates.every(gate => gate.passed) ?? false;
  const selectedStandard = useMemo(
    () => bandwidth === "NB" ? "P.1100" : bandwidth === "WB" ? "P.1110" : "P.1120",
    [bandwidth],
  );
  const localStandard = bandwidth === "SWB" || bandwidth === "FB";
  const selectedTransportSupport = (
    !(bandwidth === "SWB" && transport === "bt") && bandwidth !== "FB"
  ) ? "当前合同支持" : "当前合同无该组合";

  const upload = async (
    selected: File,
    kind: EvidenceKind,
    namespace = "",
  ): Promise<EvidenceRecord> => {
    const form = new FormData();
    form.set("file", selected);
    form.set("kind", kind);
    if (namespace) form.set("namespace", namespace);
    return responseJSON(await fetch("/api/acoustic/evidence", {
      method: "POST",
      body: form,
    }));
  };

  const uploadSingle = async () => {
    const input = document.getElementById("acoustic-evidence-file") as HTMLInputElement | null;
    const file = input?.files?.[0];
    if (!file) {
      setError("请选择一个证据文件。");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const row = await upload(file, evidenceKind, evidenceNamespace);
      setEvidence(old => ({ ...old, [evidenceKind]: row }));
      if (input) input.value = "";
      setNotice(`已入库：${row.original_name} · SHA-256 ${row.artifact.sha256.slice(0, 12)}…`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "证据上传失败");
    } finally {
      setBusy(false);
    }
  };

  const uploadCalibration = async (point: string, file: File) => {
    setBusy(true);
    setError("");
    try {
      const row = await upload(file, "calibration", `calibration-${point}`);
      setCalibrationEvidence(old => ({ ...old, [point]: row }));
      setNotice(`${point} 校准证据已校验入库。`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "校准证据上传失败");
    } finally {
      setBusy(false);
    }
  };

  const uploadScenario = async (id: string, file: File) => {
    setBusy(true);
    setError("");
    try {
      const row = await upload(file, "noise_scenario", `noise-${id}`);
      setNoiseEvidence(old => ({ ...old, [id]: row }));
      setNotice(`${id} 噪声场证据已校验入库。`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "噪声场证据上传失败");
    } finally {
      setBusy(false);
    }
  };

  const uploadDut = async (file: File) => {
    setBusy(true);
    setError("");
    try {
      const uploaded = await upload(file, "unclassified", "dut-profile");
      const row = await withEvidenceKind(uploaded, "dut_evidence");
      setDutEvidence(row);
      setNotice("DUT 路径与配置证据已校验入库。");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "DUT 证据上传失败");
    } finally {
      setBusy(false);
    }
  };

  const compileMeasurement = async () => {
    setBusy(true);
    setError("");
    try {
      const transferMask = compileKind === "transfer"
        ? selectedTransferMask(compileMaskKey, Number(compileNormalizedHz)) : null;
      if (compileKind === "transfer" && !transferMask) {
        throw Error("归一化频率必须位于所选官方掩码的频率范围内。");
      }
      const body = await responseJSON(await fetch(
        "/api/acoustic/measurements/compile",
        {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            item_id: compileItem,
            unit: compileUnit,
            kind: compileKind,
            reference_path: compileReference,
            capture_path: compileCapture,
            max_delay_ms: 1000,
            start_s: Number(compileStart),
            end_s: compileKind === "noise" ? Number(compileEnd) : null,
            mask: transferMask,
            delay_samples: 0,
          }),
        },
      ));
      setEvidence(old => ({ ...old, measurement_bundle: body.bundle }));
      setNotice(
        `证据编译器完成 ${body.measurement.item_id}：` +
        `${body.measurement.value.toFixed(3)} ${body.measurement.unit}`,
      );
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "测量证据编译失败");
    } finally {
      setBusy(false);
    }
  };

  const planRun = async () => {
    cancelRunRequest();
    const task = { kind: "plan" as const, controller: new AbortController() };
    runRequest.current = task;
    activeRunId.current = null;
    setResult(null);
    setBusy(true);
    setError("");
    setExecution(null);
    try {
      const payload = defaultPayload(mode, bandwidth, transport);
      payload.confirm_formal = confirmFormal;
      payload.fixture.calibration_points = CALIBRATION_POINTS.map(point => ({
        point,
        evidence: calibrationEvidence[point]?.artifact,
      }));
      const selectedScenarios = NOISE_SCENARIOS.filter(
        scenario => !scenario.optional || noiseEvidence[scenario.id],
      );
      payload.fixture.noise_scenarios = selectedScenarios.map(scenario => ({
        id: scenario.id,
        vehicle_speed_kmh: scenario.speed,
        fan_setting: scenario.fan,
        airflow: scenario.airflow,
        windows: scenario.windows,
        wipers: "off",
        turn_signal: "off",
        background_talkers: "none",
        road_surface: scenario.road,
        wind_speed_mps: 1,
        precipitation: "none",
        temperature_c: 20,
        evidence: noiseEvidence[scenario.id]?.artifact,
      }));
      if (dutEvidence) payload.dut.evidence = [dutEvidence.artifact];
      payload.fixture.evidence = evidence["hardware_calibration"]
        ? [evidence["hardware_calibration"].artifact] : [];
      Object.assign(payload, {
        standard_evidence: evidence["standard_text"]?.artifact,
        limit_reconciliation: evidence["limit_reconciliation"]?.artifact,
        review_evidence: evidence["review"]?.artifact,
        measurement_bundle: evidence["measurement_bundle"]?.artifact,
      });
      Object.assign(payload.hardware, {
        evidence: evidence["hardware_calibration"]?.artifact,
      });
      Object.assign(payload.method, {
        package_sha256: evidence["method_package"]?.artifact.sha256,
        package_evidence: evidence["method_package"]?.artifact,
        polqa_build_sha256: evidence["polqa_build"]?.artifact.sha256,
        polqa_evidence: evidence["polqa_build"]?.artifact,
        signal_pack_sha256: evidence["p501_signal_pack"]?.artifact.sha256,
        signal_pack_evidence: evidence["p501_signal_pack"]?.artifact,
      });
      const body = await responseJSON(await fetch("/api/acoustic/runs", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(payload),
        signal: task.controller.signal,
      }));
      if (runRequest.current !== task || task.controller.signal.aborted) return;
      activeRunId.current = body.run_id;
      setResult(body);
      await loadHistory();
      if (runRequest.current === task) setNotice("测试计划已写入不可变运行目录。");
    } catch (reason) {
      if (runRequest.current === task && !task.controller.signal.aborted) {
        setError(reason instanceof Error ? reason.message : "创建测试计划失败");
      }
    } finally {
      if (runRequest.current === task) {
        runRequest.current = null;
        setBusy(false);
      }
    }
  };

  const executeRun = async () => {
    if (!result || runRequest.current) return;
    const task = { kind: "execute" as const, runId: result.run_id, controller: new AbortController() };
    runRequest.current = task;
    setExecuting(true);
    setExecution(null);
    setError("");
    try {
      const body = await responseJSON(await fetch(
        `/api/acoustic/runs/${task.runId}/execute`,
        {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ confirm_execute: true }),
          signal: task.controller.signal,
        },
      ));
      if (runRequest.current !== task || task.controller.signal.aborted || activeRunId.current !== task.runId) return;
      if (body.run_id !== task.runId) throw Error("执行结果与当前测试计划不一致，请重新执行。");
      setExecution(body);
      setNotice("执行完成，结果已冻结并记录哈希。");
    } catch (reason) {
      if (runRequest.current === task && !task.controller.signal.aborted) {
        setError(reason instanceof Error ? reason.message : "执行失败");
      }
    } finally {
      if (runRequest.current === task) {
        runRequest.current = null;
        setExecuting(false);
      }
    }
  };

  return (
    <main className="acoustic-page">
      <header className="acoustic-hero">
        <div>
          <span className="acoustic-kicker">ITU-T ACOUSTIC QUALIFICATION</span>
          <h1>P.1100 / P.1110 / P.1120 声学测试系统</h1>
          <p>
            从物理夹具、车辆噪声场、DUT 网络路径到测量证据的可审计流程。仿真诊断和正式声学结论严格隔离。
          </p>
        </div>
        <div className={`acoustic-score ${allChecksPass ? "ready" : "blocked"}`}>
          {allChecksPass ? <ShieldCheck size={30} /> : <AlertTriangle size={30} />}
          <strong>{result ? `${passCount}/${gateTotal}` : "—"}</strong>
          <span>{result ? (allChecksPass ? "证据门禁就绪" : "仍有阻断证据") : "等待计划"}</span>
        </div>
      </header>

      <section className="acoustic-warning">
        <ShieldCheck size={20} />
        <div>
          <strong>正式结论必须经得起回放</strong>
          <p>
            P.1120 的物理链路与当前项目中的 STOI、SI-SDR、DNSMOS、工程 ERLE
            不是同一口径。系统只按证据包中的原始测量值判定；缺失值保持未测，仿真不会自动升格。
            {localStandard ? " 本地 P.1120 全文已按双子哈希锁定。" : ` ${selectedStandard} 全文尚未落地，正式门禁会保持阻断。`}
          </p>
        </div>
      </section>

      {notice ? <p className="acoustic-notice">{notice}</p> : null}
      {error ? <p className="acoustic-error" role="alert">{error}</p> : null}

      <section className="acoustic-panel">
        <div className="acoustic-section-head">
          <div>
            <span>执行链路</span>
            <h2>六个不可跳过的测量节点</h2>
          </div>
          <span className="acoustic-standard">{selectedStandard}</span>
        </div>
        <div className="acoustic-stages">
          {STAGES.map(([number, title, detail]) => (
            <article key={number}>
              <b>{number}</b>
              <h3>{title}</h3>
              <p>{detail}</p>
            </article>
          ))}
        </div>
      </section>

      <div className="acoustic-grid">
        <section className="acoustic-panel">
          <div className="acoustic-section-head">
            <div><span>测试计划</span><h2>运行配置</h2></div>
            <FlaskConical size={21} />
          </div>
          <div className="acoustic-form">
            <label>
              执行模式
              <select value={mode} onChange={event => setMode(event.target.value as Mode)}>
                <option value="simulation">仿真诊断 · 只生成工程诊断</option>
                <option value="precompliance">预合规 · 导入实验室测量</option>
                <option value="formal">正式合规 · 全证据 + 独立复核</option>
              </select>
            </label>
            <div className="acoustic-selects">
              <label>
                业务带宽
                <select value={bandwidth} onChange={event => setBandwidth(event.target.value as Bandwidth)}>
                  <option value="NB">NB · P.1100</option>
                  <option value="WB">WB · P.1110</option>
                  <option value="SWB">SWB · P.1120</option>
                  <option value="FB">FB · P.1120</option>
                </select>
              </label>
              <label>
                连接
                <select value={transport} onChange={event => setTransport(event.target.value as Transport)}>
                  <option value="wired">有线</option>
                  <option value="wireless">无线</option>
                  <option value="bt">蓝牙 / SRW</option>
                </select>
              </label>
            </div>
            <p className={
              !(bandwidth === "SWB" && transport === "bt") && bandwidth !== "FB"
              ? "acoustic-combination ok" : "acoustic-combination"}
            >
              {selectedTransportSupport}：{selectedStandard} · {bandwidth} · {transport}
              {bandwidth === "FB" ? "；FB 尚无 NV200 工作簿行，只走正式标准证据" : ""}
            </p>
            {mode === "formal" ? (
              <label className="acoustic-attestation">
                <input
                  type="checkbox"
                  checked={confirmFormal}
                  onChange={event => setConfirmFormal(event.target.checked)}
                />
                我确认本次使用受控夹具、获批方法包、授权测量后端和独立复核人。
              </label>
            ) : null}
            <button className="button primary" onClick={planRun} disabled={busy}>
              {busy ? <LoaderCircle className="spin" size={16} /> : <ChevronRight size={16} />}
              生成并核验测试计划
            </button>
          </div>
        </section>

        <section className="acoustic-panel">
          <div className="acoustic-section-head">
            <div><span>标准测项</span><h2>P.1120 测量族</h2></div>
          </div>
          <div className="acoustic-metrics">
            {METRICS.map(([label, metric, unit, requirement]) => (
              <article key={metric}>
                <div><strong>{label}</strong><span>{unit}</span></div>
                <code>{metric}</code>
                <p>{requirement}</p>
              </article>
            ))}
          </div>
        </section>
      </div>

      <section className="acoustic-panel">
        <div className="acoustic-section-head">
          <div>
            <span>不可变证据库</span>
            <h2>上传、哈希校验与取证</h2>
          </div>
          <FileCheck2 size={21} />
        </div>
        <div className="acoustic-evidence-layout">
          <div className="acoustic-upload-card">
            <label>
              证据类别
              <select
                value={evidenceKind}
                onChange={event => setEvidenceKind(event.target.value as EvidenceKind)}
              >
                {EVENTS.map(([id, label]) => <option value={id} key={id}>{label}</option>)}
              </select>
            </label>
            <label>
              命名空间
              <input
                value={evidenceNamespace}
                onChange={event => setEvidenceNamespace(event.target.value)}
                placeholder="例如 P.1100-standard / testhouse-run-01"
              />
            </label>
            <label>
              证据文件
              <input id="acoustic-evidence-file" type="file" />
            </label>
            <button className="button primary" onClick={uploadSingle} disabled={busy}>
              <FileUp size={15} /> 上传并计算 SHA-256
            </button>
          </div>
          <div className="acoustic-evidence-grid">
            {EVENTS.map(([kind, label]) => {
              const row = evidence[kind];
              return (
                <article key={kind} className={row ? "verified" : ""}>
                  {row ? <CheckCircle2 size={17} /> : <XCircle size={17} />}
                  <div>
                    <strong>{label}</strong>
                    <span>{row ? row.original_name : "尚未提供"}</span>
                    <code>{row ? `${row.artifact.sha256.slice(0, 20)}…` : kind}</code>
                  </div>
                </article>
              );
            })}
          </div>
        </div>
        <div className="acoustic-fixed-uploads">
          <div>
            <h3>五个校准点</h3>
            {CALIBRATION_POINTS.map(point => (
              <label key={point}>
                <span>{point}</span>
                <input
                  type="file"
                  onChange={event => {
                    const file = event.target.files?.[0];
                    if (file) void uploadCalibration(point, file);
                  }}
                />
                {calibrationEvidence[point] ? <em>已校验</em> : null}
              </label>
            ))}
          </div>
          <div>
            <h3>P.1120 C1–C5 噪声场景</h3>
            {NOISE_SCENARIOS.map(scenario => (
              <label key={scenario.id}>
                <span>{scenario.id} · {scenario.label}</span>
                <input
                  type="file"
                  accept="audio/*,.wav,.flac"
                  onChange={event => {
                    const file = event.target.files?.[0];
                    if (file) void uploadScenario(scenario.id, file);
                  }}
                />
                {noiseEvidence[scenario.id] ? <em>已校验</em> : null}
              </label>
            ))}
          </div>
          <div>
            <h3>DUT 配置证据</h3>
            <label>
              <span>固件 / 音量 / 音频路径</span>
              <input
                type="file"
                onChange={event => {
                  const file = event.target.files?.[0];
                  if (file) void uploadDut(file);
                }}
              />
              {dutEvidence ? <em>已校验</em> : null}
            </label>
          </div>
        </div>
        <div className="acoustic-compiler">
          <div>
            <span>已录制 PCM → 可审计测量包</span>
            <h3>证据编译器</h3>
          </div>
          <label>
            测量类型
            <select
              value={compileKind}
              onChange={event => {
                const kind = event.target.value as CompileKind;
                setCompileKind(kind);
                setCompileMaskKey(
                  bandwidth === "FB" ? "FB_send" : "SWB_send",
                );
                setCompileUnit(
                  kind === "loopback" ? "Calculated Value [ms]" :
                  kind === "noise" ? "Level [dBFS]" :
                  "Min. dist. to tolerance scheme [dB]",
                );
              }}
            >
              <option value="loopback">物理/电气回环 RTD</option>
              <option value="transfer">数字传递响应</option>
              <option value="noise">空闲噪声电平</option>
            </select>
          </label>
          <label>
            合同条目
            <input value={compileItem} onChange={event => setCompileItem(event.target.value)} />
          </label>
          {compileKind === "transfer" ? (
            <>
              <label>
                官方掩码
                <select
                  value={compileMaskKey}
                  onChange={event =>
                    setCompileMaskKey(event.target.value as TransferMaskKey)}
                >
                  <option value="SWB_send">P.1120 §11.4.1 · SWB send</option>
                  <option value="SWB_receive">P.1120 §11.4.2 · SWB receive</option>
                  <option value="FB_send">P.1120 §11.4.1 · FB send</option>
                  <option value="FB_receive">P.1120 §11.4.2 · FB receive</option>
                </select>
              </label>
              <label>
                归一化频率 / Hz
                <input
                  type="number"
                  min="1"
                  value={compileNormalizedHz}
                  onChange={event => setCompileNormalizedHz(event.target.value)}
                />
              </label>
            </>
          ) : null}
          <label>
            单位
            <input value={compileUnit} onChange={event => setCompileUnit(event.target.value)} />
          </label>
          <label>
            Reference 路径
            <input
              value={compileReference}
              onChange={event => setCompileReference(event.target.value)}
              placeholder="/home/.../build/.../reference.wav"
            />
          </label>
          <label>
            Capture 路径
            <input
              value={compileCapture}
              onChange={event => setCompileCapture(event.target.value)}
              placeholder="/home/.../build/.../capture.wav"
            />
          </label>
          {compileKind === "noise" ? (
            <div className="acoustic-compiler-time">
              <label>开始 / s<input type="number" min="0" value={compileStart} onChange={event => setCompileStart(event.target.value)} /></label>
              <label>结束 / s<input type="number" min="0" value={compileEnd} onChange={event => setCompileEnd(event.target.value)} /></label>
            </div>
          ) : null}
          <button className="button primary" onClick={compileMeasurement} disabled={busy}>
            <FlaskConical size={15} />生成测量包
          </button>
          <p>路径只允许位于项目 build 目录；原始 PCM 会复制进证据库并独立计算 SHA-256。</p>
        </div>
      </section>

      {result ? (
      <section className="acoustic-panel acoustic-result">
          <div className="acoustic-section-head">
            <div>
              <span>运行快照 · {result.run_id.slice(0, 12)}</span>
              <h2>{allChecksPass ? "门禁就绪" : "缺少或未核验正式测量证据"}</h2>
            </div>
            <div className="acoustic-actions">
              <a
                className="button secondary"
                href={`/api/acoustic/runs/${result.run_id}`}
                target="_blank"
                rel="noreferrer"
              >
                <Download size={15} />运行 JSON
              </a>
              <button className="button primary" onClick={executeRun} disabled={executing || busy}>
                {executing ? <LoaderCircle className="spin" size={15} /> : <Play size={15} />}
                执行此计划
              </button>
            </div>
          </div>
          <div className="acoustic-summary">
            <div><span>请求模式</span><strong>{result.requested_mode}</strong></div>
            <div><span>实际模式</span><strong>{result.effective_mode}</strong></div>
            <div><span>标准正文</span><strong>{result.standard.full_text_status === "verified_local" ? "已核验" : "待提供"}</strong></div>
            <div><span>计划测项</span><strong>{result.contract.rows_for_selection}</strong></div>
          </div>
          <div className="acoustic-trace-chain">
            <div>
              <span>计划请求</span>
              <code>{result.run_id}</code>
            </div>
            <ChevronRight size={16} />
            <div>
              <span>证据链</span>
              <code>{result.request_sha256.slice(0, 16)}…</code>
            </div>
            <ChevronRight size={16} />
            <div>
              <span>合同源</span>
              <code>{result.contract.sha256.slice(0, 16)}…</code>
            </div>
            <ChevronRight size={16} />
            <div>
              <span>执行状态</span>
              <code>{result.status} / {result.gate_status}</code>
            </div>
          </div>
          <div className="acoustic-gates">
            {result.gates.map(gate => (
              <article className={gate.passed ? "pass" : "fail"} key={gate.gate}>
                {gate.passed ? <CheckCircle2 size={18} /> : <XCircle size={18} />}
                <div><strong>{gate.gate}</strong><p>{gate.detail}</p></div>
              </article>
            ))}
          </div>
          <p className="acoustic-limitation">{result.limitation}</p>
        </section>
      ) : null}

      {execution && execution.run_id === result?.run_id ? (
        <section className="acoustic-panel acoustic-execution">
          <div className="acoustic-section-head">
            <div>
              <span>执行 {execution.execution_id.slice(0, 12)} · {execution.backend}</span>
              <h2>逐项判定</h2>
            </div>
            {execution.result ? (
              <a
                className="button secondary"
                href={`/api/acoustic/executions/${execution.execution_id}/download`}
              >
                <Download size={15} />结果 JSON
              </a>
            ) : null}
          </div>
          <div className="acoustic-result-counts">
            {(["pass", "fail", "review_required", "invalid", "not_measured"] as ResultStatus[])
              .map(status => (
                <div key={status}>
                  <span>{status}</span>
                  <strong>{execution.summary[status] ?? 0}</strong>
                </div>
              ))}
          </div>
          <p className="acoustic-limitation">{execution.message}</p>
          <div className="acoustic-result-file">
            <FileCheck2 size={16} />
            <span>结果冻结哈希</span>
            <code>{execution.result?.sha256 ?? result?.request_sha256 ?? "—"}</code>
          </div>
          <div className="acoustic-item-table-wrap">
            <table className="acoustic-item-table">
              <thead>
                <tr>
                  <th>条目</th><th>测项</th><th>状态</th><th>值</th>
                  <th>限值</th><th>余量</th><th>依据</th>
                </tr>
              </thead>
              <tbody>
                {execution.items.map(item => (
                  <tr key={item.item_id}>
                    <td><code>{item.item_id}</code></td>
                    <td>{item.test_name}</td>
                    <td><span className={`acoustic-status ${item.status}`}>{item.status}</span></td>
                    <td>{item.value === null ? "—" : item.value}</td>
                    <td>{item.lower_limit ?? "—"} … {item.upper_limit ?? "—"}</td>
                    <td>{item.margin === null ? "—" : item.margin.toFixed(3)}</td>
                    <td>{item.reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      <section className="acoustic-panel">
        <div className="acoustic-section-head">
          <div><span>现场操作清单</span><h2>正式测量前必须完成</h2></div>
        </div>
        <div className="acoustic-checklist">
          {[
            ["标准范围", "确认 P.1100/P.1110/P.1120 当前版本和适用带宽；P.1100/P.1110 正文必须可追溯。"],
            ["夹具与车辆", "完成 HATS、P.57 人工耳、HFRP、MRP/DRP、POI 和测量麦克风校准。"],
            ["车辆噪声", "在目标车辆中采集 C1–C4，声明 C5 时也提供 C5；记录 HVAC、车窗、路面、海拔和温度。"],
            ["下行与参考", "扬声器播放 PCM 仅处理一次，AEC reference 直接抽取；保留播放/采集时间戳。"],
            ["测量方法", "验证 P.501 信号、P.863/POLQA 构建、ETSI TS 103 281 版本及信号哈希。"],
            ["原始 trace", "保存每个测项的输入、输出、仪器设置、自动分析的固定延迟和独立复核记录。"],
          ].map(([title, detail], index) => (
            <article key={title}>
              <b>{String(index + 1).padStart(2, "0")}</b>
              <div><strong>{title}</strong><p>{detail}</p></div>
            </article>
          ))}
        </div>
      </section>

      <section className="acoustic-panel">
        <div className="acoustic-section-head">
          <div><span>审计索引</span><h2>最近运行</h2></div>
          <button className="button secondary" onClick={() => void loadHistory()}>
            <RefreshCw size={15} />刷新
          </button>
        </div>
        <div className="acoustic-history">
          {history.length === 0 ? <p>尚无当前版运行记录。</p> : history.map(run => (
            <button key={run.run_id} onClick={() => selectRun(run)}>
              <span>
                {run.standard.recommendation} · {run.requested_mode} · {run.status}
              </span>
              <strong>{run.gate_status === "blocked" ? `${run.blockers.length} 项门禁阻断` : "门禁就绪"}</strong>
              <time>{new Date(run.created_at).toLocaleString()}</time>
            </button>
          ))}
        </div>
      </section>
    </main>
  );
}
