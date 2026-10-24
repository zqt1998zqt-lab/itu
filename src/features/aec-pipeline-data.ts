import type {
  Algorithm,
  AlgorithmFlow,
  EdgeKind,
  FlowNode,
  NodeKind,
} from "./ns-pipeline-data";
import type { PipelineEvaluationData } from "./PipelineEvaluation";

export type AecEvaluation = PipelineEvaluationData;
export type AecAlgorithm = Algorithm & { evaluation: AecEvaluation };

const node = (
  id: string,
  col: number,
  row: number,
  title: string,
  sub: string,
  kind: NodeKind,
  tag: string,
  body: string,
  where: string,
  condition: string,
  fields: string[],
  formula: string,
): FlowNode => ({
  id,
  col,
  row,
  title,
  sub,
  kind,
  tag,
  body,
  where,
  condition,
  fields,
  formula,
});

const flow = (
  eyebrow: string,
  title: string,
  description: string,
  lanes: [string, string],
  nodes: FlowNode[],
  edges: Array<[string, string, EdgeKind]>,
): AlgorithmFlow => ({ eyebrow, title, description, lanes, nodes, edges });

const aec3: AecAlgorithm = {
  id: "aec3",
  title: "WebRTC AEC3",
  short: "延迟估计 · 线性自适应 · 非线性残差抑制",
  category: "runtime",
  status: "底层 Config 默认",
  rate: "16 / 24 / 48 kHz",
  entry: "AudioEngine · AecBackend::webrtc",
  cli: "--aec-backend webrtc",
  summary:
    "AEC3 是底层 Config 的默认后端；网页生成使用的 VCP 档位默认选择 Speex MDF。AEC3 在 16 kHz 子带上估计延迟并持续更新 80 ms 线性尾长，随后执行内部非线性残差抑制；AEC3 tap、参考监测和近端保护分别发布。",
  boundary:
    "默认 20 blocks 对应 80 ms；24 kHz 先桥接到 48 kHz。尾长、BF、参考延迟和正式 ITU-T 合规不能混为一谈。",
  evaluation: {
    tone: "warning",
    status: "冻结工程对照 · 2026-10-20",
    headline: "10 / 27 场景通过",
    scope: "aec3_off 冻结配置 · 16 / 24 / 48 kHz × 3 个 RT60 × 3 个声学延迟",
    source: "build/aec3-state-fusion-20261020-r3/report.json",
    note:
      "总分、案例判定与图示输出均来自 aec3_off 的完整 VCP 回放，不能当作隔离 AEC 模块的验收。状态融合候选 aec3_fusion 仅通过 7/27，新增 1 个近端/双讲失败，未进入默认链路。全部结果 standard_compliance=false。",
    metrics: [
      { label: "场景全通过", value: "10 / 27", detail: "完整工程门限" },
      { label: "远端衰减 ≥ 15 dB", value: "13 / 27", detail: "far-only" },
      { label: "线性 ERLE ≥ 6 dB", value: "11 / 27", detail: "linear tap" },
    ],
    caseAudio: {
      id: "factorial-16000-rt0.15-delay020",
      detail:
        "16 kHz；RT60 0.15 s；声学延迟 20 ms；SER -5 dB；双讲窗 17–23 s",
      sampleRate: 16000,
      windowSeconds: [17, 23],
      delaySamples: 128,
      beforeLabel: "输入 · 麦克风 / 双讲",
      afterLabel: "输出 · AEC3 tap",
      beforeFile:
        "build/aec3-state-fusion-20261020-r3/aec3_off/cases/factorial-16000-rt0.15-delay020/before.wav",
      afterFile:
        "build/aec3-state-fusion-20261020-r3/aec3_off/cases/factorial-16000-rt0.15-delay020/runs/aec_config_probe/result.aec.wav",
      waveformSrc:
        "/evaluation/pipeline-cases/aec-aec3-waveform.svg",
      melSrc: "/evaluation/pipeline-cases/aec-aec3-mel.svg",
      analysis: "双讲窗；线性滤波后仍有内部非线性抑制",
    },
    caseGroups: [
      {
        id: "pass",
        status: "通过",
        headline: "16 kHz 低混响双讲通过",
        note: "aec3_off 冻结配置；所有工程门通过。双讲 ΔSI-SDR +9.762 dB，ΔSTOI +0.2015。",
        caseAudio: {
          id: "factorial-16000-rt0.15-delay020",
          detail:
            "16 kHz；RT60 0.15 s；声学延迟 20 ms；SER −5 dB；双讲窗 17–23 s",
          sampleRate: 16000,
          windowSeconds: [17, 23],
          delaySamples: 224,
          beforeLabel: "输入 · 麦克风 / 双讲",
          afterLabel: "输出 · aec3_off 冻结 VCP",
          beforeFile:
            "build/aec3-state-fusion-20261020-r3/aec3_off/cases/factorial-16000-rt0.15-delay020/before.wav",
          afterFile:
            "build/aec3-state-fusion-20261020-r3/aec3_off/cases/factorial-16000-rt0.15-delay020/runs/aec_config_probe/result.after.wav",
          waveformSrc:
            "/evaluation/pipeline-cases/aec-aec3-pass-waveform.svg",
          melSrc: "/evaluation/pipeline-cases/aec-aec3-pass-mel.svg",
          analysis: "通过案例；双讲窗内近端语音保持",
        },
      },
      {
        id: "fail",
        status: "失败",
        headline: "48 kHz 混响远端单讲失败",
        note: "aec3_off 的失败门为远端单讲衰减 11.522 dB 与线性 ERLE 3.365 dB。双讲 raw SI-SDR −3.815 dB，双讲门通过；不沿用状态融合候选的双讲失败结论。",
        caseAudio: {
          id: "factorial-48000-rt0.45-delay080",
          detail:
            "48 kHz；RT60 0.45 s；声学延迟 80 ms；SER −5 dB；远端单讲窗 6–9 s",
          sampleRate: 48000,
          windowSeconds: [6, 9],
          delaySamples: 1248,
          beforeLabel: "输入 · 麦克风 / 远端单讲",
          afterLabel: "输出 · aec3_off 冻结 VCP",
          beforeFile:
            "build/aec3-state-fusion-20261020-r3/aec3_off/cases/factorial-48000-rt0.45-delay080/before.wav",
          afterFile:
            "build/aec3-state-fusion-20261020-r3/aec3_off/cases/factorial-48000-rt0.45-delay080/runs/aec_config_probe/result.after.wav",
          waveformSrc:
            "/evaluation/pipeline-cases/aec-aec3-fail-waveform.svg",
          melSrc: "/evaluation/pipeline-cases/aec-aec3-fail-mel.svg",
          analysis: "aec3_off 失败案例；远端衰减 11.522 dB，线性 ERLE 3.365 dB；双讲门通过",
        },
      },
    ],
  },
  flow: flow(
  "WEBRTC AEC3 · BASE CONFIG",
  "AEC3：延迟估计、线性自适应与非线性残差抑制",
  "麦克风和播放参考先进入同一处理时钟；AEC3 在线估计声学延迟，持续更新线性滤波器，再用非线性抑制处理剩余回声。",
  ["麦克风 / AEC3 音频", "播放参考 / 延迟与证据"],
  [
    node("capture", 0, 0, "麦克风 PCM", "10 ms 回调帧", "input", "CAPTURE",
      "入口保留 16/24/48 kHz；多麦克风在 AEC3 前必须先形成单波束或显式选择通道 0。",
      "AudioEngine::process()", "每帧长度等于采样率 / 100。",
      ["sample_rate", "frame_size", "microphones"], "frame = sample_rate / 100"),
    node("render", 0, 1, "播放参考", "1–8 路同步 render", "input", "RENDER",
      "参考必须与实际扬声器播放同步；多路参考保留为多个同步通道，不先把它们混成任意一路。",
      "AudioEngine::process_rx()", "参考通道数固定为 1–8。",
      ["reference_channels", "rx_processing"], "ref = post_rx_processing(reference)"),
    node("condition", 1, 0, "共享条件化", "低切 · 时钟 · BF", "core", "INPUT",
      "捕获与参考使用同一时基；启用前置波束时只把补偿后的单路 beam 送入 AEC3。",
      "AudioEngine::Impl", "24 kHz AEC3 经 48 kHz 内部桥接。",
      ["aec_input_delay_samples", "beamforming_before_aec"], "x_aec = condition(beam)"),
    node("delay", 1, 1, "延迟估计", "AEC3 / GCC-PHAT", "control", "DELAY",
      "估计设备缓冲、播放时钟和声学传播形成的总延迟；外部延迟估计可替代内部搜索。",
      "EchoCanceller3 · ReferenceMonitor", "默认 correlation_delay=true。",
      ["estimated_delay_ms", "reference_confidence"], "d = argmax Γ(x, r)"),
    node("linear", 2, 0, "线性自适应滤波", "20 blocks · 80 ms", "core", "LINEAR",
      "AEC3 在 16 kHz 子带上更新线性回声模型。每个 block 为 64 点、4 ms，默认 20 blocks 在 16/24/48 kHz 下都表示 80 ms 尾长。",
      "EchoCanceller3Config.filter", "aec_filter_blocks 默认 20，可配 13–128。",
      ["aec_filter_blocks", "aec_tail_ms"], "e = x - h * r"),
    node("nonlinear", 3, 0, "非线性残差抑制", "AEC3 suppressor", "core", "SUPPRESS",
      "线性误差继续经过 AEC3 内部的残留回声抑制；该输出已经是非线性 AEC tap，不等同于纯线性 ERLE。",
      "AudioProcessing::ProcessStream()", "AEC tap 含内部 residual suppression。",
      ["aec_rms", "echo_return_loss_enhancement_db"], "aec = suppress(e)"),
    node("tap", 2, 1, "线性 AEC tap", "诊断与保护证据", "control", "TAP",
      "AEC3 同时导出线性输出。它按声明时延对齐，供参考监测、语音保护和回归指标使用。",
      "GetLinearAecOutput()", "linear tap 可能包含 refined/coarse 切换。",
      ["linear_aec_available", "linear_aec_delay_samples"], "linear = export_linear_aec()"),
    node("monitor", 3, 1, "参考与相干性", "GCC-PHAT · coherence", "control", "VERIFY",
      "检测参考是否真实、是否持续抵消，并区分纯回声、双讲和错误参考；多参考场景另有 EchoSubspace。",
      "ReferenceMonitor · EchoCoherence", "GCC 不把单路结果外推到多路混合。",
      ["reference_state", "reference_residual_ratio", "incoherent_ratio"], "state = classify(ref, linear, coherence)"),
    node("protect", 4, 0, "近端保护", "DTD · dry fallback", "optional", "GUARD",
      "确认双讲时恢复更完整的近端语音；参考不可信且没有实际抵消证据时退回因果对齐干声。",
      "PostAecContext · UtteranceGain", "只使用当前与历史 PCM，不让 dry 路径绕过 AEC 历史。",
      ["nearend_mix", "near_protection_hold_frames", "reference_state"], "y = mix(aec, aligned_dry, near_conf)"),
    node("control", 4, 1, "共享后级状态", "NS · EQ · AGC", "control", "CONTEXT",
      "AEC 状态和线性残差送入共享后级，避免把纯回声误当近端语音；具体降噪链路在另一个页面展开。",
      "PostAecContext", "后级配置可由 profile 决定。",
      ["post_aec_context_applied", "linear_speech_probability"], "context = (far, refState, nearEvidence)"),
    node("output", 5, 0, "AEC 输出", "进入 NR / EQ / AGC", "output", "OUTPUT",
      "输出保持声明时基；AEC 前后 RMS、ERL/ERLE、估计延迟和参考状态同时发布。",
      "AudioEngine::process()", "未做正式 ITU-T 认证。",
      ["delay_samples", "aec_output_delay_samples", "post_aec_context_applied"], "y_ac = protected_aec_tap"),
  ],
  [
    ["capture", "condition", "audio"], ["render", "condition", "audio"],
    ["condition", "linear", "audio"], ["condition", "delay", "control"],
    ["delay", "linear", "control"], ["linear", "nonlinear", "audio"],
    ["linear", "tap", "control"], ["condition", "monitor", "control"],
    ["tap", "monitor", "control"], ["nonlinear", "protect", "audio"],
    ["monitor", "protect", "control"], ["monitor", "control", "control"],
    ["protect", "output", "audio"], ["control", "protect", "control"],
  ],
  ),
};

const speex: AecAlgorithm = {
  id: "speex",
  title: "Speex MDF",
  short: "可配置尾长 · 原生 24 kHz · 多参考",
  category: "runtime",
  status: "显式关闭 Coherence RES",
  rate: "16 / 24 / 48 kHz",
  entry: "SpeexAecPipeline",
  cli: "--aec-backend speex --no-speex-coherence-residual --speex-echo-tail-ms 500 --speex-native-conditioning --speex-post-context",
  summary:
    "Speex MDF 是 VCP 的基础 AEC 后端；当前 VCP 默认在它之后启用 Coherence RES。该卡片表示显式关闭 Coherence RES 的回退路径，仍启用 500 ms 尾长、原生输入调理与共享后级上下文。",
  boundary:
    "仅作为显式回退；VCP profile 默认使用 Speex MDF + Coherence RES，底层 Config 仍默认 AEC3。24 kHz MDF 原生运行，但公共 16 kHz linear tap 和既有后级转换仍然存在；不能声称整条链路完全无 SRC。",
  evaluation: {
    tone: "warning",
    status: "冻结工程对照 · 2026-10-20",
    headline: "25 / 27 场景通过",
    scope: "同一 27 场景 AEC 因子矩阵，Speex RES 基线",
    source: "build/speex-res-fallback-20261020-r1/report.json",
    note:
      "首次双讲入口候选 18/27、RES 回切候选 23/27，均低于对应基线 23/27 与 25/27，已撤回。全部结果 standard_compliance=false。",
    metrics: [
      { label: "场景全通过", value: "25 / 27", detail: "完整工程门限" },
      { label: "远端衰减 ≥ 15 dB", value: "27 / 27", detail: "far-only" },
      { label: "线性 ERLE ≥ 6 dB", value: "27 / 27", detail: "raw MDF tap" },
    ],
    caseAudio: {
      id: "factorial-16000-rt0.15-delay020",
      detail:
        "16 kHz；RT60 0.15 s；声学延迟 20 ms；SER -5 dB；双讲窗 17–23 s",
      sampleRate: 16000,
      windowSeconds: [17, 23],
      delaySamples: 0,
      beforeLabel: "输入 · 麦克风 / 双讲",
      afterLabel: "输出 · Speex MDF",
      beforeFile:
        "build/speex-res-fallback-20261020-r1/speex_res_fallback/cases/factorial-16000-rt0.15-delay020/before.wav",
      afterFile:
        "build/speex-res-fallback-20261020-r1/speex_res_fallback/cases/factorial-16000-rt0.15-delay020/runs/aec_config_probe/result.aec.wav",
      waveformSrc:
        "/evaluation/pipeline-cases/aec-speex-waveform.svg",
      melSrc: "/evaluation/pipeline-cases/aec-speex-mel.svg",
      analysis: "RES 关闭；原始 MDF 误差在 NS/AGC 前导出",
    },
    caseGroups: [
      {
        id: "pass",
        status: "通过",
        headline: "16 kHz 混响双讲通过",
        note: "Speex RES fallback 的完整工程门通过；双讲 SI-SDR +4.809 dB，STOI +0.1907。",
        caseAudio: {
          id: "factorial-16000-rt0.45-delay020",
          detail:
            "16 kHz；RT60 0.45 s；声学延迟 20 ms；SER −5 dB；双讲窗 17–23 s",
          sampleRate: 16000,
          windowSeconds: [17, 23],
          delaySamples: 0,
          beforeLabel: "输入 · 麦克风 / 双讲",
          afterLabel: "输出 · Speex MDF",
          beforeFile:
            "build/speex-res-fallback-20261020-r1/speex_res_fallback/cases/factorial-16000-rt0.45-delay020/before.wav",
          afterFile:
            "build/speex-res-fallback-20261020-r1/speex_res_fallback/cases/factorial-16000-rt0.45-delay020/runs/aec_config_probe/result.aec.wav",
          waveformSrc:
            "/evaluation/pipeline-cases/aec-speex-pass-waveform.svg",
          melSrc: "/evaluation/pipeline-cases/aec-speex-pass-mel.svg",
          analysis: "通过案例；RES fallback 保留双讲近端",
        },
      },
      {
        id: "fail",
        status: "失败",
        headline: "长延迟场景运行时失败",
        note: "失败门是 process max 19.06 ms 与 6 帧超过 10 ms；双讲 SI-SDR/STOI 本身通过，不能误写成双讲质量失败。",
        caseAudio: {
          id: "factorial-16000-rt0.15-delay180",
          detail:
            "16 kHz；RT60 0.15 s；声学延迟 180 ms；SER −5 dB；双讲窗 17–23 s",
          sampleRate: 16000,
          windowSeconds: [17, 23],
          delaySamples: 0,
          beforeLabel: "输入 · 麦克风 / 双讲",
          afterLabel: "输出 · Speex MDF",
          beforeFile:
            "build/speex-res-fallback-20261020-r1/speex_res_fallback/cases/factorial-16000-rt0.15-delay180/before.wav",
          afterFile:
            "build/speex-res-fallback-20261020-r1/speex_res_fallback/cases/factorial-16000-rt0.15-delay180/runs/aec_config_probe/result.aec.wav",
          waveformSrc:
            "/evaluation/pipeline-cases/aec-speex-fail-waveform.svg",
          melSrc: "/evaluation/pipeline-cases/aec-speex-fail-mel.svg",
          analysis: "失败案例；长延迟下运行时预算超限",
        },
      },
    ],
  },
  flow: flow(
  "SPEEXDSP MDF · VCP DEFAULT",
  "Speex MDF：可配置尾长与原生 24 kHz",
  "Speex 路径先量化到 PCM16，由多通道 MDF 完成自适应回声消除；VCP 默认启用原生输入调理、Coherence RES 与共享后级上下文。显式回退路径可以关闭 Coherence RES。",
  ["PCM16 / MDF 音频", "参考监测 / 近端控制"],
  [
    node("capture", 0, 0, "麦克风 PCM", "float → PCM16", "input", "CAPTURE",
      "捕获与参考经过一致的输入条件化；Speex MDF 在 16/24/48 kHz 原生采样率初始化。",
      "SpeexAecPipeline", "量化只发生在 AEC 内部。",
      ["sample_rate", "microphones"], "pcm16 = quantize(conditioned_capture)"),
    node("render", 0, 1, "播放参考", "同步 render PCM16", "input", "RENDER",
      "参考同样量化；多路 render 保留为 MDF 的多个参考通道。",
      "SpeexAecPipeline", "PCM 与捕获等长、等采样率。",
      ["reference_channels", "speex_echo_tail_ms"], "pcm16_ref = quantize(conditioned_render)"),
    node("condition", 1, 0, "输入条件化", "DC notch / 低切", "core", "CONDITION",
      "可启用 Speex 原生条件化，避免 MDF 前后重复外接高通造成相位旋转与重复延迟。",
      "LinearPhaseLowCut · MDF", "VCP 默认开启；其他配置可用 --speex-native-conditioning 开启。",
      ["speex_native_conditioning", "low_cut_hz"], "x = condition(mic), r = condition(ref)"),
    node("history", 1, 1, "参考历史", "100–500 ms 尾长", "control", "HISTORY",
      "MDF 保存可配置长度的参考历史。底层 Speex 参数默认 200 ms；VCP 档位设为 500 ms。",
      "speex_echo_state_init_mc()", "尾长必须按真实时延和混响选择。",
      ["speex_echo_tail_ms", "stream_delay_ms"], "history = r[t-T…t]"),
    node("mdf", 2, 0, "MDF 自适应", "多麦 · 多参考", "core", "MDF",
      "分块频域自适应滤波器持续更新；输出是原始 MDF 线性误差。",
      "speex_echo_cancellation()", "MDF 状态跨帧保留。",
      ["adaptation_frames", "linear_aec_delay_samples"], "e = x - h * history(r)"),
    node("monitor", 2, 1, "参考监测", "GCC-PHAT hop 500 ms", "control", "VERIFY",
      "单参考时监测可信度与一致延迟；多参考不根据任意单路推断整个混合场。",
      "ReferenceMonitor", "speex_post_context 可保留静音后的参考判断。",
      ["reference_state", "reference_confidence"], "confidence = GCC_PHAT(x, r)"),
    node("tap", 3, 1, "原始线性误差", "RES / 干声混合前", "control", "LINEAR TAP",
      "在 RES、干声恢复、NS 和 AGC 之前导出实际 MDF 误差，避免用最终保护输出冒充线性残差。",
      "SpeexAecPipeline::process()", "24 kHz 公共 tap 仍转换为 16 kHz。",
      ["linear_aec_available", "linear_residual_ratio"], "linear = e_before_RES"),
    node("res", 3, 0, "Coherence RES", "10 ms 输出延迟", "core", "RES",
      "VCP 默认在 MDF 误差后运行相干残差门控；多参考输入按 profile 规则回退，显式 fallback 才关闭该阶段。",
      "SpeexAecPipeline · coherence residual", "单参考默认；--no-speex-coherence-residual 可关闭。",
      ["aec_output_delay_samples", "post_aec_cancellation_ratio"], "res = RES(e)"),
    node("near", 4, 1, "双讲与近端证据", "相干性 · innovation", "control", "DTD",
      "用线性误差、非线性能量、相干性和连续帧确认双讲；错误参考需要独立保持干声回退。",
      "EchoCoherence · NsEvidence", "启用后级上下文时参与通用保护；VCP 默认开启。",
      ["post_aec_near_observed", "near_protection_hold_frames"], "near = f(cancelRatio, coherence, innovation)"),
    node("dry", 4, 0, "干声恢复", "错误参考 / 静音后回退", "optional", "FALLBACK",
      "当参考被判定为错误且没有实测抵消时，按对齐历史向真实 capture 混合；正确参考不得误回退。",
      "dry_history_ · dry_mix_", "R31 修复了静音后丢失 unrelated 判定的问题。",
      ["nearend_mix", "reference_state"], "y = (1-m)res + m*aligned_capture"),
    node("post", 5, 1, "PostAECContext", "共享后级状态", "control", "CONTEXT",
      "把真实 far/reference、线性概率、非线性概率和双讲 hold 送入后级；后级自身不再创建第二个 AEC。",
      "post_.process_impl()", "VCP 默认开启；其他配置可用 --speex-post-context 显式开启。",
      ["post_aec_context_applied", "linear_speech_probability", "nonlinear_speech_probability"], "post.aec = false; context supplied"),
    node("output", 5, 0, "AEC 输出", "继续 NR / EQ / AGC", "output", "OUTPUT",
      "输出声明 AEC 与 RES 固定延迟，并保留原始线性 tap 用于审计。",
      "SpeexAecPipeline", "VCP 单参考默认选用；Speex fallback 可关闭 Coherence RES。",
      ["delay_samples", "aec_output_delay_samples", "aec_tail_ms"], "y_ac = post(dry_blend(res))"),
  ],
  [
    ["capture", "condition", "audio"], ["render", "history", "audio"],
    ["condition", "mdf", "audio"], ["history", "mdf", "control"],
    ["capture", "monitor", "control"], ["render", "monitor", "control"],
    ["mdf", "tap", "control"], ["mdf", "res", "audio"],
    ["tap", "near", "control"], ["monitor", "near", "control"],
    ["res", "dry", "audio"], ["near", "dry", "control"],
    ["dry", "post", "control"], ["dry", "output", "audio"],
    ["post", "output", "control"],
  ],
  ),
};

const speexCoherenceResidual: AecAlgorithm = {
  id: "speex-coherence-residual",
  title: "Speex MDF + Coherence RES",
  short: "线性/非线性相干残差门控 · 平滑抑制",
  category: "runtime",
  status: "VCP 默认 · 单参考",
  rate: "16 / 24 / 48 kHz",
  entry: "SpeexAecPipeline · coherence residual postfilter",
  cli: "--aec-backend speex --speex-coherence-residual",
  summary:
    "VCP 默认在 Speex MDF 原始误差之后加入帧级相干性门控：同时观察残差与 render、render·|render| 的相干关系，只在单路参考可信、远端活跃且没有双讲证据时平滑降低相干残差。该路径不是神经 RES，也不改变线性 MDF tap。",
  boundary:
    "VCP 的单参考默认路径；多参考播放输入自动回退为未门控 Speex MDF，因为当前相干门只建模单路参考。冻结矩阵显示质量门仍有跨复跑的运行时抖动，因此默认替换不等于稳定实时预算已验收。",
  evaluation: {
    tone: "warning",
    status: "两次冻结矩阵 · 结果未稳定",
    headline: "22 / 27（复跑 23 / 27）",
    scope: "同一 27 场景 AEC 因子矩阵；Speex MDF 基线与相干残差候选逐场景配对",
    source: "build/aec-coherence-residual-20261023-r2/report.json",
    sources: [
      "build/aec-coherence-residual-20261023-r2/report.json",
      "build/aec-coherence-residual-20261023-r3/report.json",
    ],
    note:
      "r2：22/27 通过，远端衰减 ≥15 dB 为 27/27、线性 ERLE ≥6 dB 为 27/27；r3 独立复跑为 23/27，但失败场景从 24/48 kHz 长延迟运行时门之间移动。质量门没有新增 near/double-talk 失败，当前不宣称净收益或稳定实时预算。",
    metrics: [
      { label: "r2 场景全通过", value: "22 / 27", detail: "完整工程门限" },
      { label: "r3 独立复跑", value: "23 / 27", detail: "计时失败场景变化" },
      { label: "远端 / 线性门", value: "27 / 27 · 27 / 27", detail: "衰减 ≥15 dB · ERLE ≥6 dB" },
    ],
    caseGroups: [
      {
        id: "pass",
        status: "通过",
        headline: "16 kHz / 80 ms 远端延迟案例通过",
        note:
          "r2 候选输出通过完整工程门；far-only 总衰减 26.734 dB，双讲质量门通过。该图示只代表此场景，不代表全矩阵稳定通过。",
        source: "build/aec-coherence-residual-20261023-r2/results.jsonl",
        caseAudio: {
          id: "factorial-16000-rt0.15-delay080",
          detail:
            "16 kHz；RT60 0.15 s；声学延迟 80 ms；SER −5 dB；远端 6–9 s、双讲 17–23 s",
          sampleRate: 16000,
          windowSeconds: [17, 23],
          delaySamples: 96,
          beforeLabel: "输入 · 麦克风 / 双讲",
          afterLabel: "输出 · VCP Coherence RES",
          beforeFile:
            "build/aec-coherence-residual-20261023-r2/speex_coherence_residual/cases/factorial-16000-rt0.15-delay080/before.wav",
          afterFile:
            "build/aec-coherence-residual-20261023-r2/speex_coherence_residual/cases/factorial-16000-rt0.15-delay080/runs/aec_config_probe/result.after.wav",
          waveformSrc:
            "/evaluation/pipeline-cases/aec-speex-coherence-pass-waveform.svg",
          melSrc:
            "/evaluation/pipeline-cases/aec-speex-coherence-pass-mel.svg",
          analysis: "默认通过案例；相干残差门控只作用于远端可信帧",
        },
      },
      {
        id: "fail",
        status: "失败",
        headline: "48 kHz / 180 ms 长延迟运行时失败",
        note:
          "r2 候选质量门通过，但 process max 11.196 ms、1 帧超过 10 ms，因此完整工程门失败；不是双讲质量失败。",
        source: "build/aec-coherence-residual-20261023-r2/results.jsonl",
        caseAudio: {
          id: "factorial-48000-rt0.45-delay180",
          detail:
            "48 kHz；RT60 0.45 s；声学延迟 180 ms；SER −5 dB；双讲窗 17–23 s",
          sampleRate: 48000,
          windowSeconds: [17, 23],
          delaySamples: 336,
          beforeLabel: "输入 · 麦克风 / 双讲",
          afterLabel: "输出 · VCP Coherence RES",
          beforeFile:
            "build/aec-coherence-residual-20261023-r2/speex_coherence_residual/cases/factorial-48000-rt0.45-delay180/before.wav",
          afterFile:
            "build/aec-coherence-residual-20261023-r2/speex_coherence_residual/cases/factorial-48000-rt0.45-delay180/runs/aec_config_probe/result.after.wav",
          waveformSrc:
            "/evaluation/pipeline-cases/aec-speex-coherence-fail-waveform.svg",
          melSrc:
            "/evaluation/pipeline-cases/aec-speex-coherence-fail-mel.svg",
          analysis: "默认失败案例；质量通过，运行时最大值越过 10 ms 门",
        },
      },
    ],
  },
  flow: flow(
    "SPEEX MDF · COHERENCE RESIDUAL · VCP DEFAULT",
    "相干残差门控：针对非线性回声保留近端保护",
      "MDF 输出原始误差后，默认路径同时计算线性与非线性 render 特征的相干程度；只有可信远端且未确认双讲时才降低残差增益，最后继续进入共享 PostAECContext。",
    ["MDF 误差 / render 特征", "参考状态 / 双讲保护"],
    [
      node("capture", 0, 0, "MDF 原始误差", "RES 前线性 tap", "input", "ERROR", "保持真实 MDF 误差，不读取最终 NS/AGC 输出。", "SpeexAecPipeline::process()", "linear_aec_available=true", ["linear_residual_ratio"], "e = MDF(capture, render)"),
      node("render", 0, 1, "同步 render", "单路参考", "input", "RENDER", "默认只接受单路参考，避免把多路混合误判成单路相干；多参考输入回退为未门控 MDF。", "SpeexAecPipeline", "reference_channels=1", ["reference_channels", "far_active"], "r = render[t]"),
      node("features", 1, 0, "相干特征", "r 与 r·|r|", "core", "FEATURE", "同时检查线性和非线性回声关系，覆盖扬声器/放大链的非线性残差。", "SpeexAecPipeline", "逐帧、无模型推理", ["coherence"], "c = max(corr(e,r), corr(e,r|r|))"),
      node("gate", 2, 0, "可信门控", "far + reference_state", "control", "GATE", "远端活跃、参考状态可靠且没有双讲证据时才允许进入抑制。", "PostAecContext", "reference_state=1 ∧ ¬near", ["reference_state", "post_aec_near_observed"], "trusted = far ∧ ref=1 ∧ ¬near"),
      node("smooth", 3, 0, "attack / release", "帧级增益平滑", "core", "SMOOTH", "抑制进入慢于释放，避免相干判断抖动造成语音包络跳变。", "residual_gain_", "VCP 单参考默认启用", ["residual_gain"], "g[t] = smooth(target_gain)"),
      node("protect", 3, 1, "近端保护", "双讲 / 非相干残差", "control", "PROTECT", "双讲、非相干残差和错误参考保持原始 MDF 误差，不强行压制近端。", "PostAecContext", "near hold 或 incoherent evidence", ["near_protection_hold_frames", "incoherent_ratio"], "y = e when protect"),
      node("output", 4, 0, "共享后级", "NS · EQ · AGC · DRC", "output", "OUTPUT", "默认路径不改变线性 tap 和公开时基，继续交给共享后级。", "AudioEngine::process()", "VCP mono-reference default", ["post_aec_context_applied", "aec_output_delay_samples"], "out = post(g·e)"),
    ],
    [
      ["capture", "features", "audio"], ["render", "features", "audio"],
      ["features", "gate", "control"], ["gate", "smooth", "control"],
      ["gate", "protect", "control"], ["smooth", "output", "audio"],
      ["protect", "output", "control"],
    ],
  ),
};

const preAec: AecAlgorithm = {
  id: "pre-aec-beam",
  title: "前置波束 → AEC",
  short: "阵元对齐 · MVDR / 风噪控制 · 单波束",
  category: "composite",
  status: "组合链路",
  rate: "16 / 24 / 48 kHz",
  entry: "PreAecPipeline · AudioEngine core",
  cli: "--beam-delays d0,d1,… [--spatial-mvdr] [--wind-control]",
  summary:
    "阵列几何对齐和可选空间降噪先形成单路 beam，再交给 AEC。AEC/NR/EQ/AGC 消费同一条物理捕获，诊断 tap 不会把同一残差误当成多个独立麦克风。",
  boundary:
    "需要真实阵元通道、校准延迟和同步参考。公共诊断数组即使扩展为多通道，也只是同一 post-BF AEC tap 的副本。",
  evaluation: {
    tone: "warning",
    status: "三率连续流旧冻结对照 · 2026-10-10",
    headline: "基线 0 / 3 · MVDR 0 / 3",
    scope: "16 / 24 / 48 kHz 各 1 条 259 s 连续流",
    source: "build/vehicle-v2-continuous-20261010/report.json",
    note:
      "空间 MVDR 没有让任何连续流全通过。48 kHz 失败门从 12 增至 23，固定 DSP 延迟从 42.2 ms 增至 53.6 ms，因此保留为可选路径。",
    metrics: [
      { label: "连续流全通过", value: "0 / 3", detail: "两种前端都未全通过" },
      { label: "48k 失败门", value: "12 → 23", detail: "delay-sum → MVDR" },
      { label: "48k p95", value: "1.014 → 1.183 ms", detail: "同场连续流" },
    ],
    caseAudio: {
      id: "16000-baseline / radio_and_road",
      detail:
        "4 麦连续流；16 kHz；radio_and_road 段；原时间轴 96–102 s 双讲窗",
      sampleRate: 16000,
      windowSeconds: [96, 102],
      delaySamples: 384,
      beforeLabel: "BF 后输入 · 双讲",
      afterLabel: "输出 · AEC3 tap",
      beforeFile:
        "build/vehicle-v2-continuous-20261010/16000-baseline/stream.bf.wav",
      afterFile:
        "build/vehicle-v2-continuous-20261010/16000-baseline/stream.aec.wav",
      beforeChannel: 0,
      afterChannel: 0,
      waveformSrc:
        "/evaluation/pipeline-cases/aec-pre-beam-waveform.svg",
      melSrc: "/evaluation/pipeline-cases/aec-pre-beam-mel.svg",
      analysis: "连续流原时间轴 96–102 s；BF 在 AEC 之前",
    },
  },
  flow: flow(
  "BEAMFORMING → AEC · ARRAY PATH",
  "前置波束：先把多麦声场形成单路 AEC 输入",
  "阵列的几何对齐、空间降噪和风噪控制发生在 AEC 之前。AEC 之后只接收一条物理 beam，后级计量不会把同一残差复制成多个独立麦克风。",
  ["阵列 / 单波束音频", "几何 / 空间控制"],
  [
    node("mics", 0, 0, "多麦 PCM", "校准的阵元输入", "input", "MIC ARRAY",
      "每个阵元有独立的因果延迟补偿；通道数和真实 WAV 通道必须一致。",
      "PreAecPipeline", "未提供波束延迟时必须显式选择通道 0。",
      ["microphones", "beam_delays_samples"], "mic = [x0(t-d0), …]"),
    node("align", 1, 0, "几何对齐", "分数延迟", "core", "ALIGN",
      "按校准延迟对每个阵元做整数与线性插值补偿，公共延迟计入 front delay。",
      "PreAecPipeline::process()", "延迟必须满足因果约束。",
      ["bf_delay_samples", "spatial_delay_samples"], "aligned[ch] = delay(mic[ch], dch)"),
    node("spatial", 1, 1, "空间前端", "MVDR / 风噪控制", "optional", "SPATIAL",
      "可选噪声协方差 MVDR、固定 steering 或低频空间权重；只在前级处理，不改变 AEC 后端。",
      "SpatialFrontend", "spatial_mvdr 与 wind_control 默认关闭。",
      ["spatial_covariance_updates", "spatial_fallback_bins", "wind_score"], "w = spatial_weights(covariance)"),
    node("beam", 2, 0, "单波束", "delay-sum / MPDR", "core", "BEAM",
      "空间前端输出一条物理 beam，后续 AEC/NR/EQ/AGC 都在这条时基上运行。",
      "PreAecPipeline", "RF、空间和风噪延迟累计到公开时延。",
      ["spatial_delay_samples", "wind_noise_control"], "beam = Σ w_ch · aligned_ch"),
    node("render", 2, 1, "同步播放参考", "不变的多路 render", "control", "REFERENCE",
      "播放参考不经过麦克风阵列，但必须与 capture 同步；AEC 仍可接收 1–8 路同步参考。",
      "AudioEngine core", "参考通道数直接传给 AEC 后端。",
      ["reference_channels", "far_active"], "ref unchanged"),
    node("aec", 3, 0, "单通道 AEC", "AEC3 / Speex MDF", "core", "AEC",
      "AEC 接收波束，不接收所有阵元的“复制残差”；因此诊断 tap 明确为同一 post-BF 通道的副本。",
      "AudioEngine core_", "后级后端仍由 aec_backend 选择。",
      ["aec_backend", "aec_input_delay_samples"], "aec = backend(beam, ref)"),
    node("taps", 3, 1, "公共诊断 tap", "AEC / BF / linear / input", "control", "TAPS",
      "诊断数组扩展成原始阵元数量，但逐通道内容来自同一个单波束 AEC 结果，不能当成独立空间测量。",
      "PreAecPipeline::process()", "buffer extent 与物理语义分开记录。",
      ["beamforming_before_aec", "aec_capture_channels"], "tap[ch] = replica(post_bf_tap)"),
    node("output", 4, 0, "波束后 AEC 输出", "进入共享信号链", "output", "OUTPUT",
      "输出音频和前置延迟进入 NR/EQ/AGC；空间 fallback 与风噪分数继续发布。",
      "PreAecPipeline → AudioEngine", "多麦路径需要实测几何和相位。",
      ["delay_samples", "spatial_fallback_bins"], "y = post_bf_aec(beam, ref)"),
  ],
  [
    ["mics", "align", "audio"], ["align", "spatial", "control"],
    ["align", "beam", "audio"], ["spatial", "beam", "control"],
    ["beam", "aec", "audio"], ["render", "aec", "audio"],
    ["aec", "taps", "control"], ["aec", "output", "audio"],
    ["taps", "output", "control"],
  ],
  ),
};

const multiRender: AecAlgorithm = {
  id: "multirender-guard",
  title: "多参考残差保护",
  short: "EchoSubspace · 双讲门控 · 线性残差恢复",
  category: "experiment",
  status: "显式开关",
  rate: "AEC3 支持采样率",
  entry: "EchoSubspace · AudioEngine",
  cli: "--multirender-speech-protection --multirender-innovation-threshold 0.20",
  summary:
    "对 1–8 路相关或独立 render 做联合子空间建模，以无法解释的麦克风能量辅助判断双讲，再与 AEC3 线性残差和非线性语音概率共同约束近端保护。",
  boundary:
    "默认关闭，属于显式诊断增强。实现存在不等于已在当前全量 AEC 矩阵中证明净收益，页面不把它描述成默认修复。",
  evaluation: {
    tone: "pending",
    status: "端到端冻结评测待补",
    headline: "尚无显式开关结果",
    scope: "现有报告未发现 multirender_speech_protection=true 的冻结矩阵",
    source: "build/vehicle-v2-fields-float-ab-20261010/report.json",
    note:
      "历史 30 条多参考场测试与单参考负控都没有质量全通过；该旧测试并未开启当前 EchoSubspace 保护，不能据此证明或否定本链路收益。",
    metrics: [
      { label: "历史多参考场景", value: "30", detail: "16 / 24 / 48 kHz" },
      { label: "历史质量全通过", value: "0 / 30", detail: "保护开关未开启" },
      { label: "当前保护矩阵", value: "未运行", detail: "无冻结端到端报告" },
    ],
  },
  flow: flow(
  "AEC3 MULTI-RENDER · OPT-IN",
  "多参考残差保护：判断未被 render 子空间解释的近端语音",
  "多路参考可能高度相关，不能逐路相加或用任意单路推断。EchoSubspace 在每个频点联合建模 1–8 路 render，再与 AEC3 线性残差和非线性语音证据共同保护近端。",
  ["多参考 / AEC3", "子空间 / 双讲保护"],
  [
    node("capture", 0, 0, "麦克风 PCM", "单波束或通道 0", "input", "CAPTURE",
      "多参考保护只处理已形成单路物理捕获的 AEC 输入。",
      "AudioEngine", "前置 BF 的延迟已计入 AEC 时基。",
      ["microphones", "beamforming_before_aec"], "capture = beam"),
    node("render", 0, 1, "同步多参考", "2–8 路 render", "input", "RENDERS",
      "每路播放参考保持独立，支持 48 kHz 路径逐路下采样到 16 kHz 监测。",
      "AudioEngine", "不把多路 render 先混成任意单路。",
      ["reference_channels", "reference_confidence"], "R = [r1, r2, …]"),
    node("aec", 1, 0, "AEC3 多参考", "全 render 参与线性残差", "core", "AEC3",
      "AEC3 的多参考线性模型使用所有 render，并导出线性和非线性 tap。",
      "EchoCanceller3", "多参考 monitoring 与保护是额外的 opt-in 层。",
      ["linear_aec_available", "multirender_nearend_policy"], "e = x - Σ h_i * r_i"),
    node("sync", 1, 1, "参考对齐", "可信延迟", "control", "ALIGN",
      "参考监测先建立一致延迟；未获得稳定 delay 时不建立子空间。",
      "ReferenceMonitor", "延迟变化会清空历史与协方差。",
      ["estimated_delay_ms", "reference_confidence"], "r_aligned = delay(R, d)"),
    node("subspace", 2, 1, "EchoSubspace", "每频点 1–8 路联合建模", "core", "SUBSPACE",
      "使用正则化 Cholesky 拟合多路 render 张成的联合子空间，并计算麦克风功率中无法被子空间解释的比例。",
      "EchoSubspace::push()", "支持相关/dependent stereo render。",
      ["incoherent_ratio", "coherence_learning_frames"], "η[b] = 1 - P_explained / P_mic"),
    node("innovation", 2, 0, "残差 innovation", "线性误差 · 非线性概率", "control", "EVIDENCE",
      "未被 render 解释的能量还要同时满足线性残差和非线性语音证据，单一线高残差不能直接触发保护。",
      "AudioEngine::Impl", "innovation 阈值可显式配置。",
      ["linear_residual_ratio", "nonlinear_speech_probability"], "innovation = η > τ ∧ residual evidence"),
    node("hold", 3, 1, "双讲 hold", "连续确认 · 连贯回声清除", "control", "DTD",
      "双讲需要连续确认；启动纯回声或持续连贯回声会清零 hold，避免把残留播音当近端。",
      "NearProtectionGap · hold counters", "多参考不能只依赖单路 GCC。",
      ["v2_near_confirmation", "near_protection_hold_frames"], "hold ← confirm(near) / clear(echo)"),
    node("protect", 3, 0, "残差保护", "线性恢复 / 快速释放", "optional", "PROTECT",
      "确认近端时恢复线性残差中的语音；not near 时释放快于进入，继续保持强回声抑制。",
      "multirender_speech_protection", "默认关闭，需要显式开关。",
      ["nearend_mix", "echo_subtraction_gain"], "y = mix(aecNonlinear, safeLinear, near)"),
    node("output", 4, 0, "受保护 AEC tap", "交给共享后级", "output", "OUTPUT",
      "输出仍是一条 AEC 时基上的音频；子空间比例、hold 和保护混合量进入帧日志。",
      "AudioEngine", "当前页面描述实现，不宣称已通过全量 AEC 验收。",
      ["incoherent_ratio", "near_protection_hold_frames", "delay_samples"], "output = guarded_residual"),
  ],
  [
    ["capture", "aec", "audio"], ["render", "aec", "audio"],
    ["render", "sync", "control"], ["sync", "subspace", "control"],
    ["subspace", "innovation", "control"], ["aec", "innovation", "control"],
    ["subspace", "hold", "control"], ["innovation", "hold", "control"],
    ["aec", "protect", "audio"], ["hold", "protect", "control"],
    ["protect", "output", "audio"], ["hold", "output", "control"],
  ],
  ),
};

const historicalFusion: AecAlgorithm = {
  id: "aec3-state-fusion-history",
  title: "AEC3 历史状态融合",
  short: "已撤回的候选 · 10/27 → 7/27 · 保留负对照",
  category: "experiment",
  status: "历史冻结 · 当前无 flag",
  rate: "16 / 24 / 48 kHz",
  entry: "历史 AEC3 monitor/cancellation fusion",
  cli: "历史 flag：--aec3-monitor-cancellation-fusion（当前源码拒绝）",
  summary:
    "历史分支曾把 AEC3 monitor 与 cancellation 状态融合，尝试改善线性 ERLE；本轮审计确认实现已从当前源码撤回，因此这里只保留冻结证据和失败案例。",
  boundary:
    "这不是当前可选算法。历史 CLI SHA c33c65a4… 与当前构建不同；10/27 与 7/27 只描述旧二进制，不能继承到现在的 AEC3 或 Speex 路径。",
  evaluation: {
    tone: "warning",
    status: "历史冻结候选 · 非当前源码",
    headline: "10/27 → 7/27",
    scope: "同一 27 个冻结 AEC 因子场景；历史 off/fusion 配对",
    source: "build/aec3-state-fusion-20261020-r3/report.json",
    note:
      "aec3_off 通过 10/27，fusion 通过 7/27；新增 1 个近端/双讲失败。48 kHz/RT60 0.45/80 ms 的双讲 raw SI-SDR 为 −3.90748 dB，越过同案例相对门限 −3.82129 dB。",
    metrics: [
      { label: "历史基线", value: "10 / 27", detail: "aec3_off" },
      { label: "历史候选", value: "7 / 27", detail: "aec3_fusion" },
      { label: "当前源码", value: "不可运行", detail: "flag 已撤回" },
    ],
    caseGroups: [
      {
        id: "fail",
        status: "失败",
        headline: "48 kHz 双讲新增回归",
        note:
          "历史候选的双讲 raw SI-SDR 从基线通过变为失败；这个案例用于证明融合策略并非净改进，而不是证明当前 AEC3 行为。",
        caseAudio: {
          id: "factorial-48000-rt0.45-delay080",
          detail:
            "48 kHz；RT60 0.45 s；声学延迟 80 ms；SER −5 dB；双讲窗 17–23 s；历史候选",
          sampleRate: 48000,
          windowSeconds: [17, 23],
          delaySamples: 1248,
          beforeLabel: "输入 · 双讲麦克风",
          afterLabel: "输出 · 历史 AEC3 状态融合",
          beforeFile:
            "build/aec3-state-fusion-20261020-r3/aec3_fusion/cases/factorial-48000-rt0.45-delay080/before.wav",
          afterFile:
            "build/aec3-state-fusion-20261020-r3/aec3_fusion/cases/factorial-48000-rt0.45-delay080/runs/aec_config_probe/result.after.wav",
          waveformSrc:
            "/evaluation/pipeline-cases/aec-fusion-fail-waveform.svg",
          melSrc: "/evaluation/pipeline-cases/aec-fusion-fail-mel.svg",
          analysis:
            "历史负对照：双讲 raw SI-SDR −3.9075 dB，低于相对门限 −3.8213 dB；当前源码无此 flag",
        },
      },
    ],
  },
  flow: flow(
    "AEC3 STATE FUSION · HISTORICAL FREEZE",
    "历史 monitor/cancellation 状态融合：保留证据，不进入当前运行时",
    "旧分支尝试把 monitor 与 cancellation 状态融合；历史矩阵显示通过数从 10/27 降到 7/27，并新增双讲失败。当前源码已撤回该开关。",
    ["AEC3 / 历史音频", "状态融合 / 负对照"],
    [
      node("capture", 0, 0, "历史麦克风 PCM", "冻结输入", "input", "CAPTURE",
        "输入来自旧 AEC 因子矩阵，文件身份与 SHA 保留。",
        "frozen results.jsonl", "当前源码不能重放此配置。",
        ["case_sha256", "sample_rate"], "x[n]"),
      node("aec", 1, 0, "AEC3 旧分支", "历史 CLI SHA", "core", "AEC3",
        "历史 CLI 保留 monitor 与 cancellation 的独立状态。",
        "artifacts/itu_audio_cli", "只适用于冻结二进制。",
        ["cli_sha256", "aec_backend"], "e_lin, e_nl"),
      node("fusion", 2, 0, "状态融合", "monitor / cancellation", "optional", "FUSION",
        "融合状态改变非线性抑制路径；该策略已在后续源码中撤回。",
        "--aec3-monitor-cancellation-fusion", "当前源码拒绝此 flag。",
        ["aec3_monitor_cancellation_fusion"], "y = fuse(e_lin, e_nl)"),
      node("gate", 3, 0, "工程门", "27 案例配对", "control", "GATE",
        "同一近端、双讲、远端单讲和时延门限用于两个历史变体。",
        "check_audio_acceptance.py", "历史 10/27 与 7/27。",
        ["passed", "new_near_dt_failures"], "accept(y)"),
      node("output", 4, 0, "历史负对照", "不可准入", "output", "REJECT",
        "结果保留在证据页；当前默认和可选 AEC 配置均不包含该融合状态。",
        "frozen report only", "历史候选不能冒充现网能力。",
        ["standard_compliance"], "reject"),
    ],
    [
      ["capture", "aec", "audio"], ["aec", "fusion", "audio"],
      ["fusion", "gate", "control"], ["gate", "output", "control"],
    ],
  ),
};

export const aecAlgorithms: AecAlgorithm[] = [
  aec3, speex, speexCoherenceResidual, preAec, multiRender, historicalFusion,
];
