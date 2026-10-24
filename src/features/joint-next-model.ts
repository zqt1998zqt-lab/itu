export const JOINT_NEXT_REPORT_URL = "/evaluation/vcp-joint-next/report.json";
const EVIDENCE_PREFIX = "/evaluation/vcp-joint-next/";

export type JointNextVariant = {
  id: string; label: string; kind: "baseline" | "candidate";
  original_pass: number; all_speech_pass: number;
  mean_sisdr_delta_db: number; mean_stoi_delta: number;
};
export type JointNextReport = {
  schema_version: 1; case_count: 72; scope: string; candidate_id: string;
  candidate_cli_sha256: string;
  admission: { status: "pending" | "admitted" | "rejected"; checks: { id: string; label: string; pass: boolean; detail: string }[] };
  variants: JointNextVariant[];
  far_windows: { window: string; label: string; cases: number; mean_change_db: number; min_change_db: number }[];
  limitations: string[];
  figures: { scene_id: string; window: string; label: string; items: { id: string; url: string; sha256: string }[] }[];
  artifacts: { label: string; url: string; sha256: string }[];
};

const record = (value: unknown): value is Record<string, unknown> => !!value && typeof value === "object" && !Array.isArray(value);
const text = (value: unknown): value is string => typeof value === "string" && value.trim().length > 0;
const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);
const hash = (value: unknown): value is string => typeof value === "string" && /^[a-f\d]{64}$/i.test(value);
const count = (value: unknown) => Number.isInteger(value) && Number(value) >= 0 && Number(value) <= 72;
const distinct = (items: { id: string }[]) => new Set(items.map((item) => item.id)).size === items.length;
const evidenceUrl = (value: unknown) => {
  if (typeof value !== "string" || !value.startsWith(EVIDENCE_PREFIX)) return false;
  try {
    const url = new URL(value, "https://evidence.invalid");
    return url.origin === "https://evidence.invalid" && url.pathname.startsWith(EVIDENCE_PREFIX) &&
      !/%2e|%2f|%5c/i.test(value) && !value.includes("\\");
  } catch { return false; }
};

export function parseJointNextReport(value: unknown): JointNextReport {
  const fail = () => { throw new Error("新联合候选报告不完整、证据地址无效或准入状态矛盾；不展示旧分数替代。请重新发布完整报告。"); };
  if (!record(value) || value.schema_version !== 1 || value.case_count !== 72 || !text(value.scope) ||
      !text(value.candidate_id) || !hash(value.candidate_cli_sha256) || !record(value.admission)) return fail();
  const { admission } = value;
  if (!["pending", "admitted", "rejected"].includes(String(admission.status)) ||
      !Array.isArray(admission.checks) || !admission.checks.length || !admission.checks.every((item: unknown) =>
        record(item) && text(item.id) && text(item.label) && typeof item.pass === "boolean" && text(item.detail)) ||
      !distinct(admission.checks) ||
      (admission.status === "admitted" && !admission.checks.every((item) => item.pass)) ||
      (admission.status === "rejected" && admission.checks.every((item) => item.pass))) return fail();
  if (!Array.isArray(value.variants) || !value.variants.length || !value.variants.every((item: unknown) =>
    record(item) && text(item.id) && text(item.label) && ["baseline", "candidate"].includes(String(item.kind)) &&
    count(item.original_pass) && count(item.all_speech_pass) && Number(item.all_speech_pass) <= Number(item.original_pass) &&
    finite(item.mean_sisdr_delta_db) && finite(item.mean_stoi_delta)) || !distinct(value.variants) ||
    !value.variants.some((item) => item.id === value.candidate_id && item.kind === "candidate") ||
    !value.variants.some((item) => item.kind === "baseline")) return fail();
  if (!Array.isArray(value.far_windows) || !value.far_windows.length || !value.far_windows.every((item: unknown) =>
    record(item) && text(item.window) && text(item.label) && count(item.cases) && Number(item.cases) > 0 &&
    finite(item.mean_change_db) && finite(item.min_change_db) && item.min_change_db <= item.mean_change_db + 1e-8)) return fail();
  if (!Array.isArray(value.limitations) || !value.limitations.length || !value.limitations.every(text) ||
    !Array.isArray(value.figures) || !value.figures.length || !value.figures.every((group: unknown) =>
      record(group) && text(group.scene_id) && text(group.window) && text(group.label) && Array.isArray(group.items) && group.items.length > 0 &&
      group.items.every((item: unknown) => record(item) && text(item.id) && evidenceUrl(item.url) && hash(item.sha256)) && distinct(group.items)) ||
    !Array.isArray(value.artifacts) || !value.artifacts.length || !value.artifacts.every((item: unknown) =>
      record(item) && text(item.label) && evidenceUrl(item.url) && hash(item.sha256))) return fail();
  return value as JointNextReport;
}

export function jointNextStatusLabel(status: JointNextReport["admission"]["status"]): string {
  return { pending: "准入待定", rejected: "未通过准入", admitted: "工程准入通过" }[status];
}

export function jointNextFigureTitle(id: string): string {
  const titles: Record<string, string> = {
    "waveform-energy": "波形与短时能量",
    "log-stft-before": "前版对数频率 STFT",
    "log-stft-after": "候选对数频率 STFT",
    "spectral-delta": "前后频谱差分",
    ltas: "长期平均频谱",
    erb: "ERB 听觉频带谱",
    triple: "共享标尺三联分析",
    "high-time": "高时间分辨率频谱",
  };
  return titles[id] ?? "图形分析";
}
