import { useEffect, useRef, useState } from "react";
import {
  Activity,
  AudioLines,
  Car,
  ArrowLeftRight,
  BookOpen,
  ClipboardCheck,
  FlaskConical,
  FileAudio,
  Gauge,
  GitBranch,
  ListChecks,
  ListFilter,
  Radar,
  RadioTower,
  SlidersHorizontal,
  Sparkles,
  Upload,
  Waves,
  Workflow,
  Menu,
  X,
} from "lucide-react";

export type View =
  | "workbench" | "synthesize" | "workflow" | "car3d" | "results"
  | "enhancement" | "denoise" | "aec" | "bf" | "drc" | "eq" | "agc" | "compare" | "metrics" | "evaluation" | "alignment" | "basic-uplink" | "basic-downlink" | "echo" | "double-talk" | "background-noise" | "acoustic" | "graphs"
  | "background-noise-idle" | "background-noise-urban" | "background-noise-highway" | "background-noise-hvac";

export const NAV = [
  { id: "workbench", label: "音频工作台", icon: Upload },
  { id: "synthesize", label: "发送处理输出（TX）", icon: Sparkles },
  { id: "graphs", label: "音频图形引擎", icon: AudioLines },
  { id: "workflow", label: "流程图", icon: Workflow },
  { id: "enhancement", label: "综合增强链路", icon: GitBranch },
  { id: "denoise", label: "降噪链路", icon: Waves },
  { id: "aec", label: "AEC 链路", icon: RadioTower },
  { id: "bf", label: "BF 波束链路", icon: Radar },
  { id: "drc", label: "DRC 动态范围", icon: Activity },
  { id: "eq", label: "EQ 链路", icon: SlidersHorizontal },
  { id: "agc", label: "AGC 链路", icon: Gauge },
  { id: "car3d", label: "3D 智能座舱", icon: Car },
  { id: "results", label: "计算结果", icon: ListFilter },
  { id: "compare", label: "麦克风混合输入 / 处理后发送输出", icon: ArrowLeftRight },
  { id: "evaluation", label: "仿真评测数据集", icon: FlaskConical },
  { id: "alignment", label: "ITU-T 对齐验收", icon: ClipboardCheck },
  { id: "basic-uplink", label: "基础上行测试", icon: ListChecks },
  { id: "basic-downlink", label: "基础下行测试", icon: ListChecks },
  { id: "echo", label: "回声测试", icon: RadioTower },
  { id: "double-talk", label: "双讲测试", icon: ArrowLeftRight },
  { id: "background-noise", label: "背景噪声测试", icon: Waves },
  { id: "background-noise-idle", label: "背景噪声 · 怠速 / 静止", icon: Waves },
  { id: "background-noise-urban", label: "背景噪声 · 60 km/h 市区", icon: Waves },
  { id: "background-noise-highway", label: "背景噪声 · 120 km/h 高速", icon: Waves },
  { id: "background-noise-hvac", label: "背景噪声 · 空调大风档", icon: Waves },
  { id: "acoustic", label: "声学测试系统", icon: AudioLines },
  { id: "metrics", label: "指标与方法", icon: BookOpen },
] as const;

const GROUPS: { label: string; ids: View[] }[] = [
  { label: "工作与结果", ids: ["workbench", "synthesize", "graphs", "results", "compare"] },
  { label: "增强技术", ids: ["enhancement", "denoise", "aec", "bf", "drc", "eq", "agc"] },
  { label: "评测与方法", ids: ["workflow", "car3d", "evaluation", "alignment", "basic-uplink", "basic-downlink", "echo", "double-talk", "background-noise", "background-noise-idle", "background-noise-urban", "background-noise-highway", "background-noise-hvac", "acoustic", "metrics"] },
];

export default function Sidebar({
  view,
  sourceName,
}: {
  view: View;
  sourceName: string;
}) {
  const [open, setOpen] = useState(false);
  const toggle = useRef<HTMLButtonElement>(null);
  const navigation = useRef<HTMLElement>(null);
  useEffect(() => setOpen(false), [view]);
  useEffect(() => {
    const nav = navigation.current;
    const active = nav?.querySelector<HTMLElement>('[aria-current="page"]');
    if (!nav || !active || nav.clientHeight === 0) return;
    const navBounds = nav.getBoundingClientRect();
    const activeBounds = active.getBoundingClientRect();
    if (activeBounds.bottom > navBounds.bottom) {
      nav.scrollTop += activeBounds.bottom - navBounds.bottom;
    } else if (activeBounds.top < navBounds.top) {
      nav.scrollTop += activeBounds.top - navBounds.top;
    }
  }, [view]);
  return (
    <aside className="sidebar" onKeyDown={event => {
      if (event.key === "Escape" && open) {
        setOpen(false);
        toggle.current?.focus();
      }
    }}>
      <div className="sidebar-heading">
      <a className="brand" href="#workbench" aria-label="ITU Audio Lab 首页">
        <span className="brand-icon">
          <Activity size={24} strokeWidth={1.8} />
        </span>
        <span className="brand-copy">
          <strong>
            ITU<span className="brand-divider">/</span>Audio Lab
          </strong>
          <small>音频分析与声学诊断</small>
        </span>
      </a>
      <button ref={toggle} type="button" className="nav-toggle" aria-expanded={open}
        aria-controls="primary-navigation" onClick={() => setOpen(value => !value)}>
        {open ? <X size={18} /> : <Menu size={18} />}
        <span>{open ? "收起导航" : "导航"}</span>
      </button>
      </div>
      <div className="nav-label">NV200 测试项目</div>
      <nav ref={navigation} id="primary-navigation" className={open ? "mobile-open" : ""} aria-label="主要导航">
        {GROUPS.map(group => <div className="nav-group" key={group.label}>
          <h2>{group.label}</h2>
          {group.ids.map(id => NAV.find(item => item.id === id)!).map(({ id, label, icon: Icon }) => (
          <a
            key={id}
            href={`#${id}`}
            className={`nav-item ${view === id ? "active" : ""}${id.startsWith("background-noise-") ? " nav-item-sub" : ""}`}
            aria-current={view === id ? "page" : undefined}
            onClick={() => {
              if (open) toggle.current?.focus();
              setOpen(false);
            }}
          >
            <Icon size={18} strokeWidth={1.75} />
            <span>{label}</span>
            {view === id ? <span className="nav-active-dot" /> : null}
          </a>
          ))}
        </div>)}
      </nav>
      <div className="sidebar-bottom">
        <div className="source-card">
          <FileAudio size={18} />
          <div>
            <small>输入来源</small>
            <span title={sourceName}>{sourceName}</span>
          </div>
        </div>
        <div className="local-note">
          <span />
          计算结果保留在本浏览器
        </div>
      </div>
    </aside>
  );
}
