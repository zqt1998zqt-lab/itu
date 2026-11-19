import { useState, useMemo } from "react";
import {
  ArrowRight,
  BookOpen,
  Calculator,
  CheckCircle2,
  FileInput,
  Headphones,
  Info,
  Network,
  Search,
  SlidersHorizontal,
  Table,
  CheckSquare,
  Square,
} from "lucide-react";
import { METRICS } from "../domain/metrics";
import { formatNumber } from "../domain/model";
import { formatNV200Limit } from "../domain/nv200-limits.ts";
import NV200LimitRange from "./NV200LimitRange.tsx";
import { StatusBadge } from "../components/Shared";
import {
  loadNV200Sheets,
  NV200SheetData,
  NV200TestItem,
  MODE_METAS,
} from "../domain/nv200-table";
import ModeSheetBar from "./ModeSheetBar";
import MetricMappingView from "./MetricMappingView";
import SyntheticReceiveView from "./SyntheticReceiveView";
import "./metric-library.css";

export default function MetricLibrary() {
  const allSheets = useMemo(() => loadNV200Sheets(), []);
  const [selectedSheetId, setSelectedSheetId] = useState("WB-wireless");
  const [activeTab, setActiveTab] = useState<
    "nv200" | "classes" | "engineering" | "mapping" | "synthetic"
  >("nv200");

  return (
    <div className="metric-library-wrapper">
      {/* Top Main Tab Navigation */}
      <div className="metric-view-tabs">
        <button
          className={`metric-view-tab ${activeTab === "nv200" ? "active" : ""}`}
          onClick={() => setActiveTab("nv200")}
        >
          <Table size={16} />
          <span>NV200 测试用例清单 (42 项 / 8 大模式)</span>
          <span className="metric-tab-badge">标准规范</span>
        </button>
        <button
          className={`metric-view-tab ${activeTab === "classes" ? "active" : ""}`}
          onClick={() => setActiveTab("classes")}
        >
          <BookOpen size={16} />
          <span>17 类计算指标与方法 (M01 ~ M17)</span>
        </button>
        <button
          className={`metric-view-tab ${activeTab === "engineering" ? "active" : ""}`}
          onClick={() => setActiveTab("engineering")}
        >
          <SlidersHorizontal size={16} />
          <span>实际音频工程计算算法</span>
        </button>
        <button
          className={`metric-view-tab ${activeTab === "mapping" ? "active" : ""}`}
          onClick={() => setActiveTab("mapping")}
        >
          <Network size={16} />
          <span>测试指标 → 42 项认证</span>
          <span className="metric-tab-badge">映射</span>
        </button>
        <button
          className={`metric-view-tab ${activeTab === "synthetic" ? "active" : ""}`}
          onClick={() => setActiveTab("synthetic")}
        >
          <Headphones size={16} />
          <span>纯软件虚拟 DRP 双耳仿真</span>
          <span className="metric-tab-badge">模拟</span>
        </button>
      </div>

      {activeTab === "nv200" && (
        <NV200SuiteView
          allSheets={allSheets}
          selectedSheetId={selectedSheetId}
          onSelectSheetId={setSelectedSheetId}
        />
      )}
      {activeTab === "classes" && <ClassesLibraryView />}
      {activeTab === "engineering" && <EngineeringMethodsView />}
      {activeTab === "mapping" && (
        <MetricMappingView
          allSheets={allSheets}
          selectedSheetId={selectedSheetId}
          onSelectSheetId={setSelectedSheetId}
        />
      )}
      {activeTab === "synthetic" && (
        <>
          <ModeSheetBar
            allSheets={allSheets}
            selectedSheetId={selectedSheetId}
            onSelectSheetId={setSelectedSheetId}
          />
          <SyntheticReceiveView mode={selectedSheetId} />
        </>
      )}
    </div>
  );
}

function NV200SuiteView({
  allSheets,
  selectedSheetId,
  onSelectSheetId,
}: {
  allSheets: NV200SheetData[];
  selectedSheetId: string;
  onSelectSheetId: (sheetId: string) => void;
}) {
  const [selectedCategory, setSelectedCategory] = useState<string>("all");
  const [searchQuery, setSearchQuery] = useState<string>("all");
  const [activeItem, setActiveItem] = useState<NV200TestItem | null>(null);
  const [showMeasurements, setShowMeasurements] = useState<boolean>(false);

  const currentSheet = useMemo(() => {
    return allSheets.find((s) => s.sheetId === selectedSheetId) || allSheets[0];
  }, [allSheets, selectedSheetId]);

  // Aggregate category counts
  const categoryStats = useMemo(() => {
    const counts: Record<string, number> = {
      "延时": 0,
      "下行接收": 0,
      "上行发送": 0,
      "回声与双讲": 0,
      "背景噪声与舒适噪声": 0,
    };
    currentSheet.items.forEach((item) => {
      if (item.category === "延时") counts["延时"]++;
      else if (item.category === "下行接收") counts["下行接收"]++;
      else if (item.category === "上行发送") counts["上行发送"]++;
      else if (item.category === "回声与双讲") counts["回声与双讲"]++;
      else if (item.category.startsWith("背景噪声")) counts["背景噪声与舒适噪声"]++;
    });
    return counts;
  }, [currentSheet]);

  const filteredItems = useMemo(() => {
    return currentSheet.items.filter((item) => {
      if (selectedCategory !== "all") {
        if (selectedCategory === "背景噪声与舒适噪声") {
          if (!item.category.startsWith("背景噪声")) return false;
        } else if (item.category !== selectedCategory) {
          return false;
        }
      }
      if (searchQuery.trim() && searchQuery !== "all") {
        const q = searchQuery.toLowerCase();
        return (
          item.name.toLowerCase().includes(q) ||
          item.summary.toLowerCase().includes(q) ||
          item.unit.toLowerCase().includes(q) ||
          item.category.toLowerCase().includes(q) ||
          item.row.toString().includes(q)
        );
      }
      return true;
    });
  }, [currentSheet, selectedCategory, searchQuery]);

  return (
    <>
      {/* 8 Mode Switcher */}
      <ModeSheetBar
        allSheets={allSheets}
        selectedSheetId={selectedSheetId}
        onSelectSheetId={(sheetId) => {
          setActiveItem(null);
          onSelectSheetId(sheetId);
        }}
      />

      {/* Active Mode Description Card */}
      {(() => {
        const meta = MODE_METAS[selectedSheetId];
        if (!meta) return null;
        return (
          <div className="nv200-mode-card">
            <div className="nv200-mode-header">
              <div className="nv200-mode-title-row">
                <span className="nv200-mode-badge">{meta.standard}</span>
                <span className="nv200-mode-title">{meta.sheetLabel}</span>
                <span className="nv200-mode-subtag">{meta.role}</span>
              </div>
              <div className="nv200-mode-tags">
                <span className="nv200-tag-pill">业务参考率：<code>{meta.sampleRate}</code></span>
                <span className="nv200-tag-pill">频宽：<code>{meta.bandwidth}</code></span>
                <span className="nv200-tag-pill">通信链路：<code>{meta.transport}</code></span>
                <span className="nv200-tag-pill">时延限值：<code>{meta.delayLimit}</code></span>
              </div>
            </div>
            <p className="nv200-mode-desc">
              {meta.description}
            </p>
            <div className="nv200-mode-focus">
              <strong>声学台架测试核心重点：</strong>
              <span>{meta.testingFocus}</span>
            </div>
          </div>
        );
      })()}

      {/* Clean Category Stats Grid */}
      <div className="nv200-stats-grid">
        <div className="nv200-stat-card">
          <span>总测试项</span>
          <strong>{currentSheet.items.length} 项</strong>
        </div>
        <div className="nv200-stat-card">
          <span>延时类</span>
          <strong>{categoryStats["延时"]} 项</strong>
        </div>
        <div className="nv200-stat-card">
          <span>接收处理输出（RX）</span>
          <strong>{categoryStats["下行接收"]} 项</strong>
        </div>
        <div className="nv200-stat-card">
          <span>发送处理输出（TX）</span>
          <strong>{categoryStats["上行发送"]} 项</strong>
        </div>
        <div className="nv200-stat-card">
          <span>双讲与回声 (Echo)</span>
          <strong>{categoryStats["回声与双讲"]} 项</strong>
        </div>
        <div className="nv200-stat-card">
          <span>背景噪声与舒适噪声</span>
          <strong>{categoryStats["背景噪声与舒适噪声"]} 项</strong>
        </div>
      </div>

      {/* Toolbar: Category Filters, Search, and Optional Toggle */}
      <div className="nv200-toolbar">
        <div className="nv200-filter-groups">
          <button
            className={`nv200-filter-btn ${selectedCategory === "all" ? "active" : ""}`}
            onClick={() => setSelectedCategory("all")}
          >
            全部测试项 ({currentSheet.items.length})
          </button>
          <button
            className={`nv200-filter-btn ${selectedCategory === "延时" ? "active" : ""}`}
            onClick={() => setSelectedCategory("延时")}
          >
            延时 ({categoryStats["延时"]})
          </button>
          <button
            className={`nv200-filter-btn ${selectedCategory === "下行接收" ? "active" : ""}`}
            onClick={() => setSelectedCategory("下行接收")}
          >
            接收处理输出（RX） ({categoryStats["下行接收"]})
          </button>
          <button
            className={`nv200-filter-btn ${selectedCategory === "上行发送" ? "active" : ""}`}
            onClick={() => setSelectedCategory("上行发送")}
          >
            发送处理输出（TX） ({categoryStats["上行发送"]})
          </button>
          <button
            className={`nv200-filter-btn ${selectedCategory === "回声与双讲" ? "active" : ""}`}
            onClick={() => setSelectedCategory("回声与双讲")}
          >
            回声与双讲 ({categoryStats["回声与双讲"]})
          </button>
          <button
            className={`nv200-filter-btn ${selectedCategory === "背景噪声与舒适噪声" ? "active" : ""}`}
            onClick={() => setSelectedCategory("背景噪声与舒适噪声")}
          >
            背景噪声与舒适噪声 ({categoryStats["背景噪声与舒适噪声"]})
          </button>
        </div>

        <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
          {/* Optional Toggle for measurement columns */}
          <button
            className="nv200-filter-btn"
            style={{ display: "flex", alignItems: "center", gap: "5px" }}
            onClick={() => setShowMeasurements(!showMeasurements)}
            title="点击切换是否在表格右侧附加历史实测数值列"
          >
            {showMeasurements ? <CheckSquare size={14} color="#2563eb" /> : <Square size={14} color="#94a3b8" />}
            <span>附加实测数据</span>
          </button>

          <div className="nv200-search-box">
            <Search size={14} color="#64748b" />
            <input
              type="text"
              placeholder="搜索测试项、条款或单位..."
              value={searchQuery === "all" ? "" : searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
        </div>
      </div>

      {/* Clean Standard Table */}
      <div className="nv200-table-wrapper">
        <table className="nv200-table">
          <thead>
            <tr>
              <th style={{ width: "45px" }}>序号</th>
              <th style={{ width: "120px" }}>功能分类</th>
              <th style={{ minWidth: "260px" }}>测试项与标准条款</th>
              <th style={{ minWidth: "280px" }}>测试项概括</th>
              <th style={{ width: "200px" }}>测试单位</th>
              <th style={{ width: "200px" }} title="按NV200客户测试表原值展示；文字曲线要求不按数值区间解释。蓝色为允许区间，灰色为显示刻度；相同单位使用相同刻度。">标准限值区间</th>
              {/* Optional columns only when toggled on */}
              {showMeasurements && (
                <>
                  <th style={{ width: "90px" }}>实测值 G</th>
                  {currentSheet.dates.some((d) => d.key === "H") && (
                    <th style={{ width: "90px" }}>复测值 H</th>
                  )}
                  <th style={{ width: "80px" }}>表限判定</th>
                  <th style={{ width: "90px" }}>余量 / 超标</th>
                </>
              )}
            </tr>
          </thead>
          <tbody>
            {filteredItems.map((item, idx) => {
              const isSelected = activeItem?.row === item.row;
              return (
                <tr
                  key={item.row}
                  className={isSelected ? "selected" : ""}
                  onClick={() => setActiveItem(item)}
                  style={{ cursor: "pointer" }}
                >
                  <td style={{ color: "#94a3b8", fontWeight: 600 }}>{idx + 1}</td>
                  <td>
                    <span className="nv200-cat-tag">{item.category}</span>
                  </td>
                  <td>
                    <strong style={{ color: "#0f172a", fontSize: "0.8125rem" }}>{item.name}</strong>
                  </td>
                  <td>
                    <span style={{ color: "#334155", fontSize: "0.8125rem", lineHeight: 1.45 }}>
                      {item.summary}
                    </span>
                  </td>
                  <td style={{ color: "#64748b", fontFamily: "ui-monospace, monospace" }}>
                    {item.unit || "—"}
                  </td>
                  <td title={`Excel：E${item.row} 下限 / D${item.row} 上限`}>
                    <NV200LimitRange item={item} />
                  </td>

                  {/* Optional columns */}
                  {showMeasurements && (
                    <>
                      <td><strong>{item.gValue || "—"}</strong></td>
                      {currentSheet.dates.some((d) => d.key === "H") && (
                        <td>
                          {item.hValue ? (
                            <span>
                              <strong>{item.hValue}</strong>
                              {item.retestImprovement !== null && item.retestImprovement !== undefined && (
                                <span
                                  className="retest-tag"
                                  style={{
                                    color: item.retestImprovement > 0 ? "#16a34a" : "#dc2626",
                                    background: item.retestImprovement > 0 ? "#dcfce7" : "#fee2e2",
                                  }}
                                >
                                  {item.retestImprovement > 0 ? `+${item.retestImprovement}` : item.retestImprovement}
                                </span>
                              )}
                            </span>
                          ) : (
                            <span style={{ color: "#cbd5e1" }}>—</span>
                          )}
                        </td>
                      )}
                      <td>
                        <span className={`nv200-badge ${item.verdict}`}>
                          {item.verdict === "pass" && "PASS"}
                          {item.verdict === "fail" && "FAIL"}
                          {item.verdict === "pending" && "待确认"}
                          {item.verdict === "unmeasured" && "未测"}
                        </span>
                      </td>
                      <td>
                        {item.margin !== null ? (
                          <span className={`nv200-margin ${item.margin >= 0 ? "positive" : "negative"}`}>
                            {item.margin >= 0 ? `+${item.margin}` : `${item.margin}`}
                          </span>
                        ) : item.excess !== null && item.excess > 0 ? (
                          <span className="nv200-margin negative">超 {item.excess}</span>
                        ) : (
                          <span style={{ color: "#cbd5e1" }}>—</span>
                        )}
                      </td>
                    </>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Selected Item Detail Inspector */}
      {activeItem && (
        <div className="panel nv200-detail-pane">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "8px" }}>
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "6px" }}>
                <span className="nv200-cat-tag">{activeItem.category}</span>
                <span style={{ fontSize: "0.875rem", fontWeight: 600, color: "#1d4ed8" }}>
                  {activeItem.summary}
                </span>
              </div>
              <strong style={{ fontSize: "0.9375rem", color: "#0f172a" }}>
                {activeItem.name}
              </strong>
              <p style={{ margin: "6px 0 0", color: "#64748b", fontSize: "0.75rem" }}>
                测试单位：<code>{activeItem.unit || "无单位"}</code> · 标准限值区间：<code>{formatNV200Limit(activeItem)}</code>
              </p>
            </div>
            <button className="button small" onClick={() => setActiveItem(null)}>
              收起
            </button>
          </div>
        </div>
      )}

      {/* Notice Card */}
      <div className="nv200-notice-card">
        <Info size={18} color="#2563eb" style={{ flexShrink: 0, marginTop: "2px" }} />
        <div>
          <p>
            <strong>测试项与限值规范说明：</strong>
          </p>
          <p style={{ marginTop: "4px" }}>
            1. <strong>限值来源</strong>：按《NV200 ITUT测试表》八个模式的原值展示；E 列为下限，D 列为上限。客户表限值与 ITU-T 标准条款需分别核验。
            <br />
            2. <strong>原表保留</strong>：曲线要求按文字显示；空白显示“原表未填写”。<code>200</code>、<code>-200</code>、MOS 的 <code>[1 ~ 5]</code> 均保留原表数值，不额外解释为正式质量合格线。
          </p>
        </div>
      </div>
    </>
  );
}

function ClassesLibraryView() {
  const [selected, setSelected] = useState("M01");
  const metric = METRICS.find((item) => item.id === selected)!;

  return (
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
        <LegacyRangeCalculator key={metric.id} id={metric.id} />
        <div className="metric-scope-note">
          <Info size={16} />
          <p>
            上方工程指标来自实际音频。此处保留原表定义与表限演算；SLR/RLR、MOS、TCL
            及合规曲线仍需标准测试条件、标定或专用算法，不能以工程计算值替代。
          </p>
        </div>
      </section>
    </div>
  );
}

function LegacyRangeCalculator({ id }: { id: string }) {
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

function EngineeringMethodsView() {
  return (
    <section className="panel audio-methods">
      <h2>实际音频可计算的工程指标</h2>
      <p>
        四路音频分别承担干净近端参考、麦克风混合输入、处理后发送输出与远端播放信号 / 回声参考角色。所有指标保留音频
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
            "干净近端参考与麦克风混合输入去直流后对齐。检查静音、多峰、低相关与搜索边界。正值表示麦克风混合输入晚于干净近端参考，不是 RTD。",
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
            "需远端单讲、同步电平和可靠回声相关性。处理后发送输出的总衰减可能含后级 NLP、降噪和增益影响；只有确认 AEC 滤波器测量点并排除后级处理影响，才启用 ERLE 工程估计。不是 TCL/TCLw。",
          ],
          [
            "PSD / 谱峰",
            "Welch power spectral density",
            "保留原始采样率，单位 dBFS/Hz。绘图按频段保留峰值；没有校准掩模时不做谱峰合规判断。",
          ],
          [
            "工程传递函数",
            "H1(f) = Pxy(f) / Pxx(f)",
            "可靠对齐后仅显示麦克风混合输入功率高于峰值 −40 dB、相干性≥0.6 的频点。未经声学标定，未套用原表频响掩模。",
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
  );
}
