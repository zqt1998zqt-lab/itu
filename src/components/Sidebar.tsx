import {
  Activity,
  Car,
  ArrowLeftRight,
  BookOpen,
  ClipboardCheck,
  FlaskConical,
  FileAudio,
  ListFilter,
  Sparkles,
  Upload,
  Workflow,
} from "lucide-react";

export type View = "workbench" | "synthesize" | "workflow" | "car3d" | "results" | "compare" | "metrics" | "evaluation" | "alignment";

export const NAV = [
  { id: "workbench", label: "音频工作台", icon: Upload },
  { id: "synthesize", label: "生成输出", icon: Sparkles },
  { id: "workflow", label: "流程图", icon: Workflow },
  { id: "car3d", label: "3D 智能座舱", icon: Car },
  { id: "results", label: "计算结果", icon: ListFilter },
  { id: "compare", label: "处理前后对比", icon: ArrowLeftRight },
  { id: "evaluation", label: "仿真评测数据集", icon: FlaskConical },
  { id: "alignment", label: "ITU-T 对齐验收", icon: ClipboardCheck },
  { id: "metrics", label: "指标与方法", icon: BookOpen },
] as const;

export default function Sidebar({
  view,
  sourceName,
}: {
  view: View;
  sourceName: string;
}) {
  return (
    <aside className="sidebar">
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
      <div className="nav-label">NV200 测试项目</div>
      <nav aria-label="主要导航">
        {NAV.map(({ id, label, icon: Icon }) => (
          <a
            key={id}
            href={`#${id}`}
            className={`nav-item ${view === id ? "active" : ""}`}
            aria-current={view === id ? "page" : undefined}
          >
            <Icon size={18} strokeWidth={1.75} />
            <span>{label}</span>
            {view === id ? <span className="nav-active-dot" /> : null}
          </a>
        ))}
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
