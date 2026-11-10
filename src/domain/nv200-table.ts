import rawData from "../data/nv200-itut-structure.json";

export interface NV200TestItem {
  row: number;
  category: string;
  name: string;
  summary: string;
  unit: string;
  upperLimit: string;
  lowerLimit: string;
  gValue: string;
  hValue?: string;
  iValue?: string;
  isNumeric: boolean;
  lowerNum: number | null;
  upperNum: number | null;
  gNum: number | null;
  hNum: number | null;
  verdict: "pass" | "fail" | "pending" | "unmeasured";
  margin: number | null;
  excess: number | null;
  retestImprovement?: number | null;
}

export interface NV200SheetData {
  sheetId: string;
  sheetLabel: string;
  dates: { key: string; label: string }[];
  items: NV200TestItem[];
  stats: {
    total: number;
    measured: number;
    pass: number;
    fail: number;
    pending: number;
    unmeasured: number;
    retested: number;
  };
}

export interface ModeMeta {
  sheetId: string;
  sheetLabel: string;
  standard: string;
  role: string;
  sampleRate: string;
  bandwidth: string;
  transport: string;
  delayLimit: string;
  description: string;
  testingFocus: string;
}

export const MODE_METAS: Record<string, ModeMeta> = {
  "NB-wired": {
    sheetId: "NB-wired",
    sheetLabel: "窄带 · 有线 (NB-wired)",
    standard: "ITU-T P.1100",
    role: "传统网络有线互联兜底模式",
    sampleRate: "8 kHz",
    bandwidth: "300 Hz ~ 3.4 kHz",
    transport: "有线 USB 直连",
    delayLimit: "170 ms ~ 275 ms",
    description: "模拟通过 USB 线连接手机、但在 2G/3G 或非 VoLTE 传统窄带蜂窝网络下的通话场景。语音经过带通滤波裁剪，呈现经典电话听筒听感。重点考核车机底层音频 HAL 在 8kHz 重采样下的线性度、基础回声消除（AEC）深度以及防混叠特性。",
    testingFocus: "8kHz 重采样质量、抗混叠滤波、AEC 回声抑制、低频截断下的清晰度",
  },
  "WB-wired": {
    sheetId: "WB-wired",
    sheetLabel: "宽带 · 有线 (WB-wired)",
    standard: "ITU-T P.1110",
    role: "CarPlay 有线认证基准工况",
    sampleRate: "16 kHz",
    bandwidth: "100 Hz ~ 7.0 kHz",
    transport: "有线 USB 直连",
    delayLimit: "170 ms ~ 275 ms",
    description: "Apple CarPlay 有线连接的声学认证基石。由于排除了无线空口射频丢包与 Jitter 抖动，是实验室标定车内硬件声学（麦克风阵列拾音、扬声器频响校准、功放底噪、车身物理腔体共振）的最干净基准。指标要求严格，要求往返时延严格受控于 275ms 内，双讲 Class A1 透传率 ≥ 95%，回声损耗 TCLw ≥ 50dB。",
    testingFocus: "座舱物理声学标定基准、时延严格控制（≤275ms）、双讲透传度（≥95%）、TCLw 回声消除",
  },
  "SWB-wired": {
    sheetId: "SWB-wired",
    sheetLabel: "超宽带 · 有线 (SWB-wired)",
    standard: "ITU-T P.1120",
    role: "高保真超宽带语音前瞻标准",
    sampleRate: "24 / 32 kHz",
    bandwidth: "50 Hz ~ 14.0 kHz",
    transport: "有线 USB 直连",
    delayLimit: "170 ms ~ 275 ms",
    description: "面向下一代 5G EVS-SWB 语音和高保真车内语音助手的前瞻模式。语音频宽向上拓展至 14 kHz，几乎保留完整的人声泛音与微弱空气感，呈现广播级听觉质感。测试台架重点考核高采样率下 DSP 的瞬态响应、带外信号泄漏（Out-of-Band）抑制与大算力开销下的实时处理性能。",
    testingFocus: "14kHz 全频带保真度、带外杂散抑制、高采样率 DSP 算法稳定性",
  },
  "NB-wireless": {
    sheetId: "NB-wireless",
    sheetLabel: "窄带 · 无线 (NB-wireless)",
    standard: "ITU-T P.1100",
    role: "无线互联窄带通话兜底模式",
    sampleRate: "8 kHz",
    bandwidth: "300 Hz ~ 3.4 kHz",
    transport: "5GHz Wi-Fi (P2P 无线互联)",
    delayLimit: "170 ms ~ 415 ms",
    description: "在无线 CarPlay 连接状态下接入传统窄带电话的工况。因为引入了 Wi-Fi 射频空口传输与系统 Jitter Buffer 缓冲，时延上限放宽至 415ms。主要考核车机在低采样率与无线网络抖动并存工况下的时钟漂移补偿、静音消除与丢包补偿（PLC）鲁棒性。",
    testingFocus: "Wi-Fi 空口抗抖动、低采样率下 Jitter Buffer 适配、丢包补偿",
  },
  "WB-wireless": {
    sheetId: "WB-wireless",
    sheetLabel: "宽带 · 无线 CarPlay (WB-wireless)",
    standard: "ITU-T P.1110",
    role: "Apple CarPlay 认证最核心主战场",
    sampleRate: "16 kHz",
    bandwidth: "100 Hz ~ 7.0 kHz",
    transport: "5GHz Wi-Fi (P2P 无线互联)",
    delayLimit: "170 ms ~ 415 ms",
    description: "涵盖 90% 以上车主日常用车体验的核心工况，也是苹果认证中最重度的一张测试表。音频流由 5GHz Wi-Fi 承载，全量覆盖怠速（Sce1）、60km/h 市区（Sce2）、120km/h 高速（Sce3）以及空调 4 档大风（Sce4）等复合噪声环境。考核重点是在强行车胎噪与无线抖动下，AEC 不漏回声且不吞字，N-MOS 噪声压制彻底，S-MOS 人声保真自然。",
    testingFocus: "四类行车噪声工况 3-MOS 表现、Wi-Fi 抖动下 AEC 双讲透传、舒适噪声生成平稳度",
  },
  "SWB-wireless": {
    sheetId: "SWB-wireless",
    sheetLabel: "超宽带 · 无线 (SWB-wireless)",
    standard: "ITU-T P.1120",
    role: "无线超宽带高保真互联模式",
    sampleRate: "24 / 32 kHz",
    bandwidth: "50 Hz ~ 14.0 kHz",
    transport: "5GHz Wi-Fi (P2P 无线互联)",
    delayLimit: "170 ms ~ 415 ms",
    description: "结合了 5GHz Wi-Fi 高吞吐能力与 ITU-T P.1120 超宽带声学指标，主要用于下一代座舱的高清语音交互。重点考核在无线空口传输损耗与大计算量条件下的算法算力分配、实时多通道处理及超宽带频响平坦度。",
    testingFocus: "超宽带无线传输吞吐、高采样率降噪算力平衡、全频域语音保真",
  },
  "NB-bt": {
    sheetId: "NB-bt",
    sheetLabel: "窄带 · 蓝牙免提 (NB-bt)",
    standard: "ITU-T P.1100",
    role: "车机原生经典蓝牙免提兼容",
    sampleRate: "8 kHz (CVSD)",
    bandwidth: "300 Hz ~ 3.4 kHz",
    transport: "经典蓝牙 HFP 1.5",
    delayLimit: "170 ms ~ 415 ms",
    description: "脱离手机互联协议（不启动 CarPlay），车机原生通过经典蓝牙 HFP 协议连接手机并采用 CVSD 编码的模式。针对老旧手机及未开通高清语音蜂窝网络的兼容通话，考核车机原生蓝牙射频基带在 8kHz 窄带下的解调质量、防破音和回声消除。",
    testingFocus: "CVSD 编码抗丢包、经典蓝牙射频稳定性、车机独立免提回声消除",
  },
  "WB-bt": {
    sheetId: "WB-bt",
    sheetLabel: "宽带 · 蓝牙免提 (WB-bt)",
    standard: "ITU-T P.1110",
    role: "车机独立原生蓝牙通话标配",
    sampleRate: "16 kHz (mSBC)",
    bandwidth: "50 Hz ~ 7.0 kHz",
    transport: "经典蓝牙 HFP 1.6+",
    delayLimit: "170 ms ~ 415 ms",
    description: "现代车机的原生独立蓝牙电话标配模式，采用 16 kHz mSBC 宽带编码。用户日常不使用 CarPlay 而直接用车机蓝牙打电话走的就是该模式。声学频响、失真度和 MOS 指标要求与 WB-wired 相当，同时重点考验车机独立蓝牙协议栈的时钟同步与 Jitter 抑制能力。",
    testingFocus: "mSBC 宽带编解码时钟同步、车机原生通话频响掩模、双讲全双工表现",
  },
};

const SHEET_LABELS: Record<string, string> = {
  "NB-wired": "窄带 · 有线 (NB-wired)",
  "WB-wired": "宽带 · 有线 (WB-wired)",
  "SWB-wired": "超宽带 · 有线 (SWB-wired)",
  "NB-wireless": "窄带 · 无线 (NB-wireless)",
  "WB-wireless": "宽带 · 无线 CarPlay (WB-wireless)",
  "SWB-wireless": "超宽带 · 无线 (SWB-wireless)",
  "NB-bt": "窄带 · 蓝牙免提 (NB-bt)",
  "WB-bt": "宽带 · 蓝牙免提 (WB-bt)",
};

export function getTestItemSummary(name: string): string {
  const n = name.toLowerCase();

  // 延时
  if (n.includes("round trip delay")) {
    return "往返端到端时延（从嘴发声到扬声器返回耳部全链路时间）";
  }

  // 谱峰检测
  if (n.includes("spec. peaks") || n.includes("peaks")) {
    if (n.includes("11.07.2") || n.includes("9.7.2") || n.includes("rcv") || n.includes("receiving")) {
      return "下行底噪谱峰检测（排查扬声器/功放单频啸叫与尖峰）";
    }
    return "上行底噪谱峰检测（排查麦克风拾音通道单频/电源干扰峰）";
  }

  // 下行接收响度
  if (n.includes("loudness rating - receiving") || n.includes("loudness rating - rcv")) {
    if (n.includes("nominal")) return "标称音量接收响度（正常通话外放音量感知响度）";
    if (n.includes("maximum")) return "最大音量接收响度（车载功放最大外放输出响度）";
    return "下行接收响度 RLR（车载扬声器外放响度）";
  }
  // 接收响度线性度
  if (n.includes("variation of rlr")) {
    if (n.includes("-5db")) return "接收响度电平变化量（输入 -5dB 时外放响应稳定性）";
    if (n.includes("+5db")) return "接收响度电平变化量（输入 +5dB 时外放响应稳定性）";
    return "接收响度随电平变化偏差（考核 AGC/DRC 线性度）";
  }
  // 下行频响
  if ((n.includes("frequency response") || n.includes("freq. response")) && (n.includes("rcv") || n.includes("receiving"))) {
    return "下行接收频响曲线（扬声器外放双耳频响掩模合规度）";
  }
  // 下行空闲底噪
  if ((n.includes("idle channel noise") || n.includes("idle channel")) && (n.includes("rcv") || n.includes("receiving"))) {
    return "下行空闲信道底噪（通话静音时扬声器外放背景底噪）";
  }
  // 下行单向语音质量
  if ((n.includes("one way") || n.includes("listening speech")) && (n.includes("rcv") || n.includes("receiving"))) {
    return "下行单向语音质量（安静环境下下行接收清晰度与保真度 MOS）";
  }
  // 下行带外
  if (n.includes("out-of-band") && (n.includes("rcv") || n.includes("receiving"))) {
    return "下行带外信号抑制（防扬声器高频混叠与谐波失真泄漏）";
  }

  // 上行发送响度
  if (n.includes("loudness rating - sending") || n.includes("loudness rating - snd")) {
    return "上行发送响度 SLR（麦克风近场拾音灵敏度与感知响度）";
  }
  // 上行频响
  if ((n.includes("frequency response") || n.includes("freq. response")) && (n.includes("snd") || n.includes("sending"))) {
    return "上行发送频响曲线（麦克风拾音频响平坦度与掩模合规度）";
  }
  // 上行单向语音质量
  if ((n.includes("one way") || n.includes("listening speech")) && (n.includes("snd") || n.includes("sending"))) {
    return "上行单向语音质量（安静环境下麦克风通话清晰度与保真度 MOS）";
  }
  // 上行空闲底噪
  if ((n.includes("idle channel noise") || n.includes("idle channel")) && (n.includes("snd") || n.includes("sending"))) {
    return "上行空闲信道底噪（静音环境下麦克风发送给远端的编码底噪）";
  }
  // 上行带外
  if (n.includes("out-of-band") && (n.includes("snd") || n.includes("sending"))) {
    return "上行发送带外信号抑制（防止麦克风通道高频混叠干扰）";
  }

  // 双讲发送衰减
  if (n.includes("attn. range")) {
    if (n.includes("nominal") || n.includes("nom")) return "标称音量双讲透传度（双方同时说话无衰减帧 Class A1 占比）";
    if (n.includes("maximum") || n.includes("max")) return "最大音量双讲透传度（大音量下双讲不吞字帧 Class A1 占比）";
    return "双讲期间发送衰减范围（双方同时说话全双工透传表现）";
  }

  // 回声耦合损耗
  if (n.includes("terminal coupling loss") || n.includes("terminal coupl") || n.includes("tcl")) {
    if (n.includes("maximum") || n.includes("max")) return "最大音量回声损耗 TCLw（大音量外放防回声与啸叫深度）";
    if (n.includes("recommended")) return "推荐回声损耗 TCLw（推荐音量档位回声消除深度）";
    return "标称音量回声损耗 TCLw（回声消除算法 AEC 抑制深度）";
  }

  // HATS 语音电平与信噪比
  if (n.includes("speech level at hats") || n.includes("s/n calculation")) {
    return "人工头拾音信噪比（人工头说话声与座舱底噪之差）";
  }

  // 背景噪声质量
  if (n.includes("gmos") || n.includes("g-mos")) {
    const sce = n.match(/sce(\d)/)?.[1] || "";
    const sceNames: Record<string, string> = { "1": "怠速静止", "2": "60km/h 市区", "3": "120km/h 高速", "4": "空调大风档" };
    const prefix = sceNames[sce] ? `【${sceNames[sce]}】` : "";
    return `${prefix}综合听音质量（G-MOS 整体人声与降噪体验分）`;
  }
  if (n.includes("nmos") || n.includes("n-mos")) {
    const sce = n.match(/sce(\d)/)?.[1] || "";
    const sceNames: Record<string, string> = { "1": "怠速静止", "2": "60km/h 市区", "3": "120km/h 高速", "4": "空调大风档" };
    const prefix = sceNames[sce] ? `【${sceNames[sce]}】` : "";
    return `${prefix}背景噪声抑制分（N-MOS 胎噪/风噪消除纯净度得分）`;
  }
  if (n.includes("smos") || n.includes("s-mos")) {
    const sce = n.match(/sce(\d)/)?.[1] || "";
    const sceNames: Record<string, string> = { "1": "怠速静止", "2": "60km/h 市区", "3": "120km/h 高速", "4": "空调大风档" };
    const prefix = sceNames[sce] ? `【${sceNames[sce]}】` : "";
    return `${prefix}噪声中人声保真度（S-MOS 避免降噪过度损伤弱人声）`;
  }

  // 舒适噪声
  if (n.includes("comfort noise")) {
    const sce = n.match(/sce(\d)/)?.[1] || "";
    const sceNames: Record<string, string> = { "1": "怠速静止", "2": "60km/h 市区", "3": "120km/h 高速", "4": "空调大风档" };
    const prefix = sceNames[sce] ? `【${sceNames[sce]}】` : "";
    if (n.includes("level adj") || n.includes("level")) {
      return `${prefix}舒适噪声电平匹配（静音补噪与真车噪电平差，防抽吸感）`;
    }
    if (n.includes("spectral") || n.includes("spectrum")) {
      return `${prefix}舒适噪声频谱拟合（合成底噪拟合车噪频谱，防刺耳白噪）`;
    }
    return `${prefix}舒适噪声生成评估（静音期间人工底噪平稳度）`;
  }

  return "ITU 标准免提声学性能指标";
}

function formatExcelDate(v: string | undefined): string {
  if (!v) return "";
  const num = Number(v.trim());
  if (!isNaN(num) && num > 40000 && num < 60000) {
    const date = new Date(Math.round((num - 25569) * 86400 * 1000));
    return date.toISOString().split("T")[0];
  }
  return v.trim();
}

function parseNumber(v: string | undefined): number | null {
  if (v === undefined || v === null) return null;
  const s = String(v).trim().replace(/,/g, "");
  if (!s || s === "-" || s.includes("符合")) return null;
  const n = Number(s);
  return isNaN(n) ? null : n;
}

function getCategory(rowNum: number): string {
  if (rowNum === 2) return "延时";
  if (rowNum >= 3 && rowNum <= 11) return "下行接收";
  if (rowNum >= 12 && rowNum <= 18) return "上行发送";
  if (rowNum >= 19 && rowNum <= 23) return "回声与双讲";
  if (rowNum >= 24 && rowNum <= 28) return "背景噪声 Sce1 (怠速)";
  if (rowNum >= 29 && rowNum <= 33) return "背景噪声 Sce2 (60km/h)";
  if (rowNum >= 34 && rowNum <= 38) return "背景噪声 Sce3 (120km/h)";
  if (rowNum >= 39 && rowNum <= 43) return "背景噪声 Sce4 (空调大风)";
  return "综合";
}

export function loadNV200Sheets(): NV200SheetData[] {
  return (rawData as any[]).map((sheetObj) => {
    const sheetId = sheetObj.sheet as string;
    const sheetLabel = SHEET_LABELS[sheetId] || sheetId;
    const rows = sheetObj.rows as { row: string; values: Record<string, string> }[];
    const header = rows[0]?.values || {};

    const dates: { key: string; label: string }[] = [];
    ["G", "H", "I"].forEach((col) => {
      const cellKey = `${col}1`;
      if (header[cellKey]) {
        dates.push({
          key: col,
          label: `${col}列 (${formatExcelDate(header[cellKey])})`,
        });
      }
    });

    let passCount = 0;
    let failCount = 0;
    let pendingCount = 0;
    let unmeasuredCount = 0;
    let retestedCount = 0;
    let measuredCount = 0;

    const items: NV200TestItem[] = [];

    for (let i = 1; i < rows.length; i++) {
      const r = rows[i];
      const rowNum = Number(r.row);
      const vals = r.values || {};

      const name = (vals[`B${rowNum}`] || "").replace(/\n/g, " ").trim();
      const summary = getTestItemSummary(name);
      const unit = (vals[`C${rowNum}`] || "").replace(/\n/g, " ").trim();
      const upperLimit = (vals[`D${rowNum}`] || "").replace(/\n/g, " ").trim();
      const lowerLimit = (vals[`E${rowNum}`] || "").replace(/\n/g, " ").trim();
      const gValue = (vals[`G${rowNum}`] || "").trim();
      const hValue = vals[`H${rowNum}`] ? vals[`H${rowNum}`].trim() : undefined;
      const iValue = vals[`I${rowNum}`] ? vals[`I${rowNum}`].trim() : undefined;

      if (hValue) retestedCount++;

      const gNum = parseNumber(gValue);
      const hNum = parseNumber(hValue);
      const upperNum = parseNumber(upperLimit);
      const lowerNum = parseNumber(lowerLimit);

      const category = getCategory(rowNum);

      let verdict: "pass" | "fail" | "pending" | "unmeasured" = "unmeasured";
      let margin: number | null = null;
      let excess: number | null = null;

      if (!gValue) {
        verdict = "unmeasured";
        unmeasuredCount++;
      } else {
        measuredCount++;
        if (upperLimit.includes("符合") || lowerLimit.includes("符合") || gNum === null) {
          verdict = "pending";
          pendingCount++;
        } else {
          // Numerical check
          let pass = true;
          if (lowerNum !== null && gNum < lowerNum) pass = false;
          if (upperNum !== null && gNum > upperNum) pass = false;

          let m = Infinity;
          if (lowerNum !== null) m = Math.min(m, gNum - lowerNum);
          if (upperNum !== null) m = Math.min(m, upperNum - gNum);
          margin = m === Infinity ? null : Number(m.toFixed(3));

          let e = 0;
          if (lowerNum !== null && gNum < lowerNum) e = Math.max(e, lowerNum - gNum);
          if (upperNum !== null && gNum > upperNum) e = Math.max(e, gNum - upperNum);
          excess = Number(e.toFixed(3));

          if (pass) {
            verdict = "pass";
            passCount++;
          } else {
            verdict = "fail";
            failCount++;
          }
        }
      }

      let retestImprovement: number | null = null;
      if (hNum !== null && gNum !== null) {
        retestImprovement = Number((hNum - gNum).toFixed(3));
      }

      items.push({
        row: rowNum,
        category,
        name,
        summary,
        unit,
        upperLimit,
        lowerLimit,
        gValue,
        hValue,
        iValue,
        isNumeric: gNum !== null,
        lowerNum,
        upperNum,
        gNum,
        hNum,
        verdict,
        margin,
        excess,
        retestImprovement,
      });
    }

    return {
      sheetId,
      sheetLabel,
      dates,
      items,
      stats: {
        total: items.length,
        measured: measuredCount,
        pass: passCount,
        fail: failCount,
        pending: pendingCount,
        unmeasured: unmeasuredCount,
        retested: retestedCount,
      },
    };
  });
}
