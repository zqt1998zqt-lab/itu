export type AgcMatrixVariant = {
  id: string;
  label: string;
  configuration: string;
  pass: number;
  fail: number;
  fixes: number;
  regressions: number;
  byRate: Record<"16000" | "24000" | "48000", { pass: number; total: number }>;
  byLevel: Record<"-45" | "-30" | "-15", { pass: number; total: number }>;
  pcmGainDb: { median: number; min: number; max: number; nonzero: number };
  failureTop: Array<{ metric: string; count: number }>;
};

export type AgcGainMethod = {
  scope: string;
  nonzeroDb: number;
  formula: string;
  source: string;
  scriptSha256: string;
};

export type AgcNegativeControl = {
  name: string;
  configuration: string;
  status: "observational" | "verified";
  expected: string;
  evidence: string;
  limitation: string;
};

export const agcGainMethod: AgcGainMethod = {
  scope: "整段 protected NR → AGC 后 tap；每个案例一次能量比",
  nonzeroDb: 0.01,
  formula: "10·log10(mean(after²) / mean(before²))",
  source:
    "build/agc-stage4-release-no-tx-drc-admission-20261020/gain-analysis.json",
  scriptSha256:
    "877981fadb8fe3d810aa03e748b3561929fb02cd941346c9ef54b18d9997f620",
};

export const agcTechMatrix = {
  scope: "AGC family frozen core",
  caseCount: 108,
  cliSha256:
    "e1a5dca83d224e0642e58d4ead2aeb7495a0c476abfe9dc63bd178a0a4d3afb8",
  source: "build/agc-tech-matrix-agc108-20261020",
  variants: [
    {
      id: "guarded5",
      label: "Guarded5",
      configuration: "默认",
      pass: 52,
      fail: 56,
      fixes: 0,
      regressions: 0,
      byRate: {
        "16000": { pass: 30, total: 54 },
        "24000": { pass: 14, total: 27 },
        "48000": { pass: 8, total: 27 },
      },
      byLevel: {
        "-45": { pass: 9, total: 36 },
        "-30": { pass: 28, total: 36 },
        "-15": { pass: 15, total: 36 },
      },
      pcmGainDb: { median: 0, min: -0.002, max: 7.971, nonzero: 33 },
      failureTop: [
        { metric: "raw_sisdr_db", count: 49 },
        { metric: "projection_gain_min_db", count: 41 },
        { metric: "total_attenuation_db", count: 13 },
        { metric: "stoi_change", count: 1 },
      ],
    },
    {
      id: "guarded5_low_level",
      label: "Guarded5 + low-level latch",
      configuration: "--low-level-projection-gain",
      pass: 46,
      fail: 62,
      fixes: 4,
      regressions: 10,
      byRate: {
        "16000": { pass: 27, total: 54 },
        "24000": { pass: 16, total: 27 },
        "48000": { pass: 3, total: 27 },
      },
      byLevel: {
        "-45": { pass: 4, total: 36 },
        "-30": { pass: 27, total: 36 },
        "-15": { pass: 15, total: 36 },
      },
      pcmGainDb: { median: 0, min: -0.002, max: 7.997, nonzero: 44 },
      failureTop: [
        { metric: "raw_sisdr_db", count: 43 },
        { metric: "total_attenuation_db", count: 36 },
        { metric: "projection_gain_min_db", count: 8 },
        { metric: "stoi_change", count: 3 },
      ],
    },
    {
      id: "stage4",
      label: "UtteranceGain v4 / Stage4",
      configuration: "--repair-stage 4",
      pass: 65,
      fail: 43,
      fixes: 21,
      regressions: 8,
      byRate: {
        "16000": { pass: 41, total: 54 },
        "24000": { pass: 17, total: 27 },
        "48000": { pass: 7, total: 27 },
      },
      byLevel: {
        "-45": { pass: 17, total: 36 },
        "-30": { pass: 32, total: 36 },
        "-15": { pass: 16, total: 36 },
      },
      pcmGainDb: { median: 0, min: -0.004, max: 7.985, nonzero: 39 },
      failureTop: [
        { metric: "raw_sisdr_db", count: 45 },
        { metric: "total_attenuation_db", count: 13 },
        { metric: "stoi_change", count: 9 },
      ],
    },
    {
      id: "agc2_adaptive",
      label: "AGC2 Adaptive Digital",
      configuration: "--webrtc-adaptive-gain",
      pass: 18,
      fail: 90,
      fixes: 0,
      regressions: 34,
      byRate: {
        "16000": { pass: 11, total: 54 },
        "24000": { pass: 6, total: 27 },
        "48000": { pass: 1, total: 27 },
      },
      byLevel: {
        "-45": { pass: 0, total: 36 },
        "-30": { pass: 3, total: 36 },
        "-15": { pass: 15, total: 36 },
      },
      pcmGainDb: { median: 5.825, min: 0, max: 16.984, nonzero: 85 },
      failureTop: [
        { metric: "raw_sisdr_db", count: 92 },
        { metric: "total_attenuation_db", count: 67 },
        { metric: "projection_gain_min_db", count: 6 },
        { metric: "projection_gain_max_db", count: 2 },
      ],
    },
    {
      id: "fixed_0",
      label: "Fixed 0 dB",
      configuration: "--agc-fixed-gain 0",
      pass: 44,
      fail: 64,
      fixes: 1,
      regressions: 9,
      byRate: {
        "16000": { pass: 26, total: 54 },
        "24000": { pass: 13, total: 27 },
        "48000": { pass: 5, total: 27 },
      },
      byLevel: {
        "-45": { pass: 0, total: 36 },
        "-30": { pass: 29, total: 36 },
        "-15": { pass: 15, total: 36 },
      },
      pcmGainDb: { median: 0, min: -0.002, max: 0, nonzero: 0 },
      failureTop: [
        { metric: "projection_gain_min_db", count: 72 },
        { metric: "raw_sisdr_db", count: 37 },
        { metric: "total_attenuation_db", count: 10 },
      ],
    },
    {
      id: "fixed_3",
      label: "Fixed 3 dB",
      configuration: "--agc-fixed-gain 3",
      pass: 20,
      fail: 88,
      fixes: 1,
      regressions: 33,
      byRate: {
        "16000": { pass: 9, total: 54 },
        "24000": { pass: 7, total: 27 },
        "48000": { pass: 4, total: 27 },
      },
      byLevel: {
        "-45": { pass: 0, total: 36 },
        "-30": { pass: 19, total: 36 },
        "-15": { pass: 1, total: 36 },
      },
      pcmGainDb: { median: 3, min: 2.515, max: 3, nonzero: 108 },
      failureTop: [
        { metric: "projection_gain_min_db", count: 72 },
        { metric: "raw_sisdr_db", count: 55 },
        { metric: "total_attenuation_db", count: 36 },
        { metric: "stoi_change", count: 9 },
      ],
    },
    {
      id: "fixed_6",
      label: "Fixed 6 dB",
      configuration: "--agc-fixed-gain 6",
      pass: 6,
      fail: 102,
      fixes: 0,
      regressions: 46,
      byRate: {
        "16000": { pass: 1, total: 54 },
        "24000": { pass: 3, total: 27 },
        "48000": { pass: 2, total: 27 },
      },
      byLevel: {
        "-45": { pass: 0, total: 36 },
        "-30": { pass: 6, total: 36 },
        "-15": { pass: 0, total: 36 },
      },
      pcmGainDb: { median: 6, min: 4.001, max: 6, nonzero: 108 },
      failureTop: [
        { metric: "raw_sisdr_db", count: 72 },
        { metric: "total_attenuation_db", count: 70 },
        { metric: "projection_gain_min_db", count: 63 },
        { metric: "stoi_change", count: 29 },
      ],
    },
    {
      id: "fixed_8",
      label: "Fixed 8 dB",
      configuration: "--agc-fixed-gain 8",
      pass: 5,
      fail: 103,
      fixes: 3,
      regressions: 50,
      byRate: {
        "16000": { pass: 1, total: 54 },
        "24000": { pass: 4, total: 27 },
        "48000": { pass: 0, total: 27 },
      },
      byLevel: {
        "-45": { pass: 3, total: 36 },
        "-30": { pass: 2, total: 36 },
        "-15": { pass: 0, total: 36 },
      },
      pcmGainDb: { median: 8, min: 4.603, max: 8, nonzero: 108 },
      failureTop: [
        { metric: "total_attenuation_db", count: 89 },
        { metric: "raw_sisdr_db", count: 86 },
        { metric: "stoi_change", count: 56 },
      ],
    },
    {
      id: "no_agc",
      label: "No AGC",
      configuration: "--no-agc",
      pass: 44,
      fail: 64,
      fixes: 1,
      regressions: 9,
      byRate: {
        "16000": { pass: 26, total: 54 },
        "24000": { pass: 13, total: 27 },
        "48000": { pass: 5, total: 27 },
      },
      byLevel: {
        "-45": { pass: 0, total: 36 },
        "-30": { pass: 29, total: 36 },
        "-15": { pass: 15, total: 36 },
      },
      pcmGainDb: { median: 0, min: 0, max: 0, nonzero: 0 },
      failureTop: [
        { metric: "projection_gain_min_db", count: 72 },
        { metric: "raw_sisdr_db", count: 37 },
        { metric: "total_attenuation_db", count: 10 },
      ],
    },
    {
      id: "no_tx_drc",
      label: "No TX DRC",
      configuration: "--no-tx-drc",
      pass: 56,
      fail: 52,
      fixes: 4,
      regressions: 0,
      byRate: {
        "16000": { pass: 34, total: 54 },
        "24000": { pass: 14, total: 27 },
        "48000": { pass: 8, total: 27 },
      },
      byLevel: {
        "-45": { pass: 9, total: 36 },
        "-30": { pass: 28, total: 36 },
        "-15": { pass: 19, total: 36 },
      },
      pcmGainDb: { median: 0, min: -0.002, max: 7.971, nonzero: 33 },
      failureTop: [
        { metric: "projection_gain_min_db", count: 41 },
        { metric: "raw_sisdr_db", count: 39 },
        { metric: "total_attenuation_db", count: 13 },
        { metric: "stoi_change", count: 1 },
      ],
    },
  ] satisfies AgcMatrixVariant[],
} as const;

export const agcReleaseCandidate = {
  id: "stage4_noise_release",
  label: "Stage4 + noise release",
  configuration: "--repair-stage 4 --stage4-noise-release",
  pass: 66,
  fail: 42,
  fixes: 21,
  regressions: 7,
  byRate: {
    "16000": { pass: 42, total: 54 },
    "24000": { pass: 17, total: 27 },
    "48000": { pass: 7, total: 27 },
  },
  byLevel: {
    "-45": { pass: 18, total: 36 },
    "-30": { pass: 32, total: 36 },
    "-15": { pass: 16, total: 36 },
  },
  pcmGainDb: { median: 0, min: -0.004, max: 7.988, nonzero: 39 },
  failureTop: [
    { metric: "raw_sisdr_db", count: 45 },
    { metric: "total_attenuation_db", count: 12 },
    { metric: "stoi_change", count: 9 },
  ],
  cliSha256:
    "d7a38bbde294a5e5da4b234c621a2225eb412447a481467853c1bbc489573d3f",
  source: "build/agc-stage4-release-20261020",
} as const;

export const agcBestCandidate = {
  id: "stage4_noise_release_no_tx_drc",
  label: "Stage4 + release + No TX DRC",
  configuration:
    "--repair-stage 4 --stage4-noise-release --no-tx-drc",
  pass: 71,
  fail: 37,
  fixes: 26,
  regressions: 7,
  admission: "blocked_regressions",
  byRate: {
    "16000": { pass: 46, total: 54 },
    "24000": { pass: 17, total: 27 },
    "48000": { pass: 8, total: 27 },
  },
  byLevel: {
    "-45": { pass: 18, total: 36 },
    "-30": { pass: 33, total: 36 },
    "-15": { pass: 20, total: 36 },
  },
  pcmGainDb: { median: 0, min: -0.004, max: 7.988, nonzero: 39 },
  failureTop: [
    { metric: "raw_sisdr_db", count: 18 },
    { metric: "total_attenuation_db", count: 12 },
    { metric: "stoi_change", count: 7 },
  ],
  cliSha256:
    "b12ede46844ae221a26c797057c80233e9dfe5fc91a94dcc6c90acf04494eb5b",
  replaySha256:
    "c5787123019c2c567d9938f2b957246752aff415404f6e130e367e48f234a68a",
  gainReportSha256:
    "3973210e49adee2d6f78af3e63f97b2eb9951b83aa37c99924d8b494e0d57aab",
  resultsSha256:
    "5102886208afc0a1ae9aa0d78dece0cd8e1b42f7e29a353e6659fc57dcbee6d5",
  reportJsonSha256:
    "af20f99ebbb4b29dca203540be06c5664c5ea42119de4bac1ce606822a6852a2",
  source: "build/agc-stage4-release-no-tx-drc-admission-20261020",
} as const;

export const agcNegativeControls: AgcNegativeControl[] = [
  {
    name: "Executor fault injection",
    configuration: "--agc-fault previous-frame | inverted | always-boost",
    status: "verified",
    expected:
      "同一条 Stage4 正向增益路径必须让三种错误命令在 applied_gain_db 和最终 PCM 中可见",
    evidence:
      "整链测试先断言正常路径进入正增益；previous-frame 改变 PCM，inverted 实际命令负增益，always-boost 的总能量由 1.16222 升至 68.1507",
    limitation:
      "这是执行器故障注入，用来证明测试能失败；不等同于真实硬件时序故障或标准认证。",
  },
  {
    name: "No AGC",
    configuration: "--no-agc",
    status: "observational",
    expected: "−45 dBFS 的 36 个案例全部不能通过低输入投影增益门",
    evidence: "44 / 108 整例通过；−45 dBFS 0 / 36；36 / 36 未达 +6 dB",
    limitation:
      "固定 0 dB 与它在 102 / 108 案例输出相同，不能单独证明错误 AGC 一定会被通用质量门抓住。",
  },
  {
    name: "Fixed 0 dB",
    configuration: "--agc-fixed-gain 0",
    status: "observational",
    expected: "只验证零增益路径，不作为任意错误执行器的负控制",
    evidence: "44 / 108；−45 dBFS 0 / 36；实际 PCM 增益 0 dB",
    limitation:
      "只证明“完全没有增益”会失败，不能证明错误时序、错增益或误触发会被检出。",
  },
  {
    name: "Fixed 3 / 6 / 8 dB",
    configuration: "--agc-fixed-gain 3 | 6 | 8",
    status: "verified",
    expected: "恒定增益破坏噪声、语音与低输入组合门限",
    evidence: "通过数分别为 20、6、5；回归 33、46、50；实际增益中位数 3、6、8 dB",
    limitation:
      "这些是错误策略的工程负控，不是标准测试向量，也不能代替真实故障注入。",
  },
  {
    name: "Stage4 stale-noise control",
    configuration: "弱语音锁存后输入持续静音 / 底噪",
    status: "verified",
    expected: "旧版增益策略保留 8 dB，新策略单调释放回 0 dB",
    evidence: "test_ecnr_control 与 test_utterance_gain 的真实 PCM 状态测试",
    limitation:
      "组件测试只覆盖控制状态；仍需全链 108 案例确认噪声段没有新的回归。",
  },
];

export const agcMatrixById = Object.fromEntries(
  agcTechMatrix.variants.map((variant) => [variant.id, variant]),
) as Record<string, AgcMatrixVariant>;

export const agcAblationBuilds = [
  {
    branches: "Stage4 / Stage4 + No TX DRC",
    cliSha256:
      "d7a38bbde294a5e5da4b234c621a2225eb412447a481467853c1bbc489573d3f",
  },
  {
    branches: "Stage4 + release / Stage4 + release + No TX DRC",
    cliSha256:
      "b12ede46844ae221a26c797057c80233e9dfe5fc91a94dcc6c90acf04494eb5b",
  },
] as const;

export const agcFaultReplay = {
  source: "build/agc-fault-code-replay-20261020",
  cliSha256:
    "e1558f1e98d0849fee29742bef7d29d90e342f53dd5264a8b846914b155b3fca",
  resultsSha256:
    "5d4188dda9d7da98d0dbe025828d9a5649759b5105138e1e142652907b27e58b",
  reportSha256:
    "816c20352a83c77fd31190ae1312d0d56c114072eab7189317c9b29fdc3cb07e",
  outputShaMatch: 108,
  caseCount: 108,
  pass: 71,
  regressions: 7,
} as const;

export const agcRegressionCases = [
  {
    run: "controlled-native-full-20261017",
    id: "vehicle-16000-validation-m1-10",
    noise: "straffic",
    rate: "16 kHz",
    near: "−45 dBFS",
    failed: "噪声段总衰减 5.630 < 6 dB",
  },
  {
    run: "controlled-native-full-20261017",
    id: "vehicle-48000-development-m1-11",
    noise: "tcar",
    rate: "48 kHz",
    near: "−30 dBFS",
    failed: "晚期 STOI 变化 −0.02524 < −0.02",
  },
  {
    run: "controlled-native-full-20261017",
    id: "vehicle-48000-development-m2-12",
    noise: "tcar",
    rate: "48 kHz",
    near: "−30 dBFS",
    failed: "晚期 STOI 变化 −0.03016 < −0.02",
  },
  {
    run: "controlled-native-full-20261017",
    id: "vehicle-48000-validation-m2-11",
    noise: "straffic",
    rate: "48 kHz",
    near: "−45 dBFS",
    failed: "噪声段总衰减 4.388 < 6 dB",
  },
  {
    run: "controlled-native-full-20261017",
    id: "vehicle-48000-holdout-m1-10",
    noise: "tbus",
    rate: "48 kHz",
    near: "−45 dBFS",
    failed: "噪声段总衰减 4.375 < 6 dB",
  },
  {
    run: "controlled-native-full-20261017",
    id: "vehicle-48000-holdout-m4-11",
    noise: "tbus",
    rate: "48 kHz",
    near: "−45 dBFS",
    failed: "噪声段总衰减 5.668 < 6 dB",
  },
  {
    run: "controlled-cavemove-20261017",
    id: "vehicle-16000-validation-m4-11",
    noise: "cavemove_smart_forfour_s60_w0",
    rate: "16 kHz",
    near: "−45 dBFS",
    failed: "噪声段总衰减 5.873 < 6 dB",
  },
] as const;

export const AGC_METRIC_LABELS: Record<string, string> = {
  raw_sisdr_db: "SI-SDR",
  projection_gain_min_db: "投影增益不足",
  projection_gain_max_db: "投影增益超限",
  total_attenuation_db: "噪声衰减不足",
  stoi_change: "STOI 下降",
};
