export type AlgorithmCategory = "runtime" | "composite" | "experiment" | "research";
export type NodeKind = "input" | "control" | "core" | "optional" | "output";
export type EdgeKind = "audio" | "control" | "optional";

export type FlowNode = {
  id: string;
  title: string;
  sub: string;
  col: number;
  row: number;
  kind: NodeKind;
  tag: string;
  body: string;
  where: string;
  condition: string;
  fields: string[];
  formula: string;
};

export type AlgorithmFlow = {
  eyebrow: string;
  title: string;
  description: string;
  lanes: [string, string];
  nodes: FlowNode[];
  edges: Array<[string, string, EdgeKind]>;
};

export type Algorithm = {
  id: string;
  title: string;
  short: string;
  category: AlgorithmCategory;
  status: string;
  rate: string;
  entry: string;
  cli: string;
  summary: string;
  boundary: string;
  flow: AlgorithmFlow;
};

export const categoryMeta: Record<
  AlgorithmCategory,
  { label: string; note: string }
> = {
  runtime: {
    label: "运行时后端",
    note: "由 NoiseBackend / --ns-backend 直接选择",
  },
  composite: {
    label: "组合与辅助",
    note: "通过额外开关、第二状态或后级模块接入",
  },
  experiment: {
    label: "离线实验",
    note: "项目内有 C++ 探针或冻结对照，但未进入常规后端枚举",
  },
  research: {
    label: "仅调研",
    note: "已有技术路线记录，当前没有生产运行路径",
  },
};

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

const webrtc = flow(
  "WEBRTC NS · ONLINE",
  "WebRTC 频谱降噪",
  "WebRTC 在同一内部状态中完成噪声估计、语音概率与谱增益；项目外层只补充连续语音保护和干湿混合。",
  ["音频 / WebRTC 状态", "连续证据 / 保护"],
  [
    node("pcm", 0, 0, "PCM 10 ms", "16 / 24 / 48 kHz", "input", "FRAME",
      "固定 10 ms 回调帧；24 kHz 由 AudioEngine 显式桥接到 48 kHz 核心。",
      "AudioEngine::process()", "帧长等于采样率 / 100。",
      ["sample_rate", "frame_size"], "hop = sample_rate / 100"),
    node("buffer", 1, 0, "AudioBuffer", "内部时域缓冲", "core", "BUFFER",
      "PCM 写入 WebRTC AudioBuffer；48 kHz 路径执行分带和合带。",
      "webrtc::AudioBuffer", "每个连续流持有独立状态。",
      ["core_rate", "ns_delay_samples"], "CopyFrom → SplitIntoFrequencyBands"),
    node("analyze", 2, 0, "Analyze", "噪声谱 · 语音概率", "core", "ANALYZE",
      "持续更新内部噪声谱和语音概率，不读取外部 VAD、clean 或场景标签。",
      "NoiseSuppressor::Analyze()", "状态跨帧连续。",
      ["noise_spectrum", "speech_probability"], "state ← analyze(power_spectrum)"),
    node("gain", 3, 0, "Spectral gain", "频点抑制", "core", "GAIN",
      "按内部语音概率和噪声估计计算频点增益；level 0..3 是配置档，不是实测 dB 衰减。",
      "NoiseSuppressor::Process()", "语音保护由概率与后续控制器共同承担。",
      ["noise_level", "ns_residual_gain"], "wet[k] = G[k] · X[k]"),
    node("output", 4, 0, "Merge / output", "回到 I/O 采样率", "output", "OUTPUT",
      "合带、复制输出并发布 pre/post NS RMS 和声明时延。",
      "AudioBuffer::MergeFrequencyBands()", "评分只扣公开时延，不搜索 lag。",
      ["post_ns_rms", "delay_samples"], "y = merge(GX)"),
    node("evidence", 1, 1, "连续证据", "SNR · confidence · flatness", "control", "EVIDENCE",
      "控制器仅用当前和历史 PCM 计算帧级 SNR、语音置信、谱平坦度和低频占比。",
      "VehicleNsController::observe()", "无外部 VAD，无未来帧。",
      ["snr_db", "confidence", "noise_floor_db"], "SNR = 10 log10((P-N)/N)"),
    node("protect", 3, 1, "Dry protection", "起音 · 弱语音 · 混合", "optional", "GUARD",
      "启用车辆策略时，用延迟对齐干声保护高 SNR、起音和弱语音；控制快攻慢放。",
      "NsRepairAlignment · startup_control", "原始后端选择不会自动开启公共保护。",
      ["dry_mix_target", "gain_target"], "out = gain · ((1-m)·wet + m·dry)"),
  ],
  [
    ["pcm", "buffer", "audio"], ["buffer", "analyze", "audio"],
    ["analyze", "gain", "control"], ["gain", "output", "audio"],
    ["buffer", "evidence", "control"], ["evidence", "protect", "control"],
    ["output", "protect", "audio"],
  ],
);

const speex = flow(
  "SPEEXDSP · ONLINE",
  "Speex 在线噪声估计",
  "PCM 先量化到 16 bit，再由 SpeexDSP 维护噪声估计；项目可在固定档、动态档和独立强噪声状态之间选择。",
  ["Speex 主状态", "动态策略 / 第二状态"],
  [
    node("pcm", 0, 0, "PCM float", "量化为 PCM16", "input", "PCM16",
      "入口 float PCM 量化为 16 bit 后进入 Speex preprocess。",
      "SpeexNs::process()", "每 10 ms 运行一次。",
      ["input_rms_dbfs"], "pcm16 = lround(clamp(x·32768))"),
    node("pre", 1, 0, "Preprocess", "denoise on · AGC/VAD off", "core", "PREPROCESS",
      "仅启用降噪；Speex AGC 与 VAD 关闭，避免与项目控制重复。",
      "speex_preprocess_state_init()", "支持 16/24/48 kHz。",
      ["core_rate", "strength"], "speex_preprocess_run(state, pcm16)"),
    node("estimate", 2, 0, "Noise estimate", "在线频带状态", "core", "ESTIMATE",
      "SpeexDSP 持续更新频带噪声估计，不需要预录噪声阶段。",
      "SpeexPreprocessState", "连续流状态不会逐帧重置。",
      ["noise_estimate", "suppression_db"], "N[t] = update(N[t-1], P[t])"),
    node("suppress", 3, 0, "Suppression", "配置增益底", "core", "SUPPRESS",
      "固定档按请求抑制量衰减；实际输出衰减还取决于噪声和语音估计。",
      "SPEEX_PREPROCESS_SET_NOISE_SUPPRESS", "有效范围 −40..0 dB。",
      ["speex_suppression_db"], "G = gain(SNR, suppression)"),
    node("strong", 2, 1, "Strong state", "可选第二状态", "optional", "STATE 2",
      "steady_dynamic 保持独立强噪声状态，切换时不清空主状态学习。",
      "VehicleNsProcessor::Impl", "仅动态或稳态策略启用。",
      ["steady_suppression_db"], "wet_strong = Speex_strong(x)"),
    node("blend", 3, 1, "Continuous blend", "confidence · floor", "control", "BLEND",
      "依据语音置信、噪声底和音节间隔连续混合主/强状态。",
      "VehicleNsController · NsRepairAlignment", "无外部二值 VAD。",
      ["confidence", "dry_mix_target"], "m = f(confidence, background, hold)"),
    node("output", 4, 0, "PCM output", "10 ms 流式输出", "output", "OUTPUT",
      "输出保持入口采样率并发布固定时延。",
      "SpeexNs::process()", "状态跨帧保留。",
      ["post_ns_rms", "delay_samples"], "y = (1-m)·wet + m·wet_strong"),
  ],
  [
    ["pcm", "pre", "audio"], ["pre", "estimate", "control"],
    ["estimate", "suppress", "control"], ["suppress", "output", "audio"],
    ["pre", "strong", "audio"], ["strong", "blend", "audio"],
    ["suppress", "blend", "audio"], ["blend", "output", "control"],
  ],
);

const rnnoise = flow(
  "RNNOISE · RECURRENT MODEL",
  "RNNoise 48 kHz 循环增强",
  "RNNoise 以 480 点帧驱动小型循环网络；16/24 kHz 输入通过显式 48 kHz 核心适配。",
  ["采样率 / 网络", "后级保护"],
  [
    node("input", 0, 0, "I/O PCM", "16 / 24 / 48 kHz", "input", "INPUT",
      "保留入口采样率；非 48 kHz 输入进入 NativePipeline 桥接。",
      "NativePipeline", "16/24 kHz 不宣称原生模型率。",
      ["io_rate", "core_rate"], "core_rate = 48000"),
    node("adapter", 1, 0, "48 kHz adapter", "帧打包与重采样", "control", "ADAPTER",
      "形成固定 480 点网络帧，转换延迟计入公开时延。",
      "RateConverter", "核心状态连续。",
      ["src_delay_samples"], "frame480 = adapt(input)"),
    node("gru", 2, 0, "RNNoise GRU", "增强帧 + VAD", "core", "GRU",
      "循环网络输出增强 PCM 和语音活动概率。",
      "rnnoise_process_frame()", "模型随源码编译，不使用 ORT。",
      ["vad_probability", "recurrent_state"], "y, vad = GRU(x, state)"),
    node("enhance", 3, 0, "Enhanced frame", "网络时域输出", "core", "ENHANCE",
      "网络直接产生增强帧；后级公共控制可继续保护近端语音。",
      "rnnoise_process_frame()", "不额外运行谱后处理网络。",
      ["pre_ns_rms", "post_ns_rms"], "wet = model(x)"),
    node("protect", 3, 1, "Common protection", "可选干湿混合", "optional", "GUARD",
      "复用起音、弱语音和低频保护，不重置 RNNoise 状态。",
      "VehicleNsProcessor", "显式启用车辆策略时生效。",
      ["dry_mix_target"], "out = mix(wet, dry)"),
    node("return", 5, 0, "Return rate", "回到 I/O 采样率", "output", "OUTPUT",
      "核心输出回到入口采样率并扣除声明延迟。",
      "NativePipeline::process()", "网络和 SRC 延迟分别记录。",
      ["delay_samples"], "yio = resample(protected)"),
  ],
  [
    ["input", "adapter", "audio"], ["adapter", "gru", "audio"],
    ["gru", "enhance", "audio"], ["enhance", "protect", "optional"],
    ["protect", "return", "audio"],
  ],
);

const ulunas = flow(
  "UL-UNAS · STREAMING ONNX",
  "UL-UNAS 有状态谱增强",
  "C++ 适配器按 16 kHz、512 点窗、256 点 hop 运行四输入/四输出 ONNX；24/48 kHz 经 SpeexDSP SRC 进入 16 kHz 核心。",
  ["16 kHz 前端与缓存", "谱增强与重建"],
  [
    node("input", 0, 0, "PCM input", "16 / 24 / 48 kHz", "input", "INPUT",
      "每 10 ms 接收 PCM；模型核心固定 16 kHz。",
      "NeuralNs::process()", "必须提供有效模型路径。",
      ["sample_rate", "model_path"], "x10ms = callback()"),
    node("src", 1, 0, "16 kHz adapter", "24/48 → 16 kHz", "control", "SRC",
      "非 16 kHz 使用 SpeexDSP quality 8 转换；输出再回到入口率。",
      "RateConverter", "SRC 延迟单独发布。",
      ["src_delay_samples"], "x16 = resample(xio)"),
    node("frame", 2, 0, "512 FFT / 256 hop", "Hann 分析", "core", "ANALYSIS",
      "当前和历史 PCM 组成 512 点窗，得到 257 个复频谱 bin。",
      "NeuralNs::State::infer()", "模型契约固定。",
      ["fft_size", "hop"], "X = FFT(w · [past,current])"),
    node("cache", 3, 0, "Four caches", "conv · tfa · inter", "core", "CACHE",
      "mix、conv_cache、tfa_cache、inter_cache 四输入四输出持续传递。",
      "Ort::Session::Run()", "每流独立状态；cache reset 有数值负控。",
      ["neural_inferences"], "cache_out ← Run(cache_in)"),
    node("mask", 4, 0, "Enhanced spectrum", "复杂谱输出", "core", "MODEL",
      "模型输出增强复频谱，并检查形状、有限值和缓存。",
      "ulunas_stream_simple.onnx", "错误显式返回 backend_error。",
      ["enhanced_spectrum", "nonfinite_outputs"], "Ŝ = model(X, cache)"),
    node("wola", 5, 0, "iSTFT / WOLA", "16 kHz 增强 PCM", "output", "SYNTH",
      "逆 FFT、Hann 合成和重叠相加。",
      "NeuralNs::State::infer()", "核心信号延迟 512 样本。",
      ["core_delay_samples"], "y = overlap_add(IFFT(Ŝ))"),
    node("back", 6, 0, "Return rate", "回到 I/O 采样率", "output", "OUTPUT",
      "24/48 kHz 路径上采样；总延迟包含 SRC。",
      "NeuralNs::process()", "不静默回退 WebRTC。",
      ["delay_samples"], "yio = resample(y16)"),
  ],
  [
    ["input", "src", "audio"], ["src", "frame", "audio"],
    ["frame", "cache", "audio"], ["cache", "mask", "audio"],
    ["mask", "wola", "audio"], ["wola", "back", "audio"],
  ],
);

const omlsa = flow(
  "OM-LSA + IMCRA · STATISTICAL",
  "OM-LSA + IMCRA 统计降噪",
  "IMCRA 估计条件噪声谱，OM-LSA 用局部/全局先验、贝叶斯语音后验和 DD 先验 SNR 计算软谱增益。",
  ["噪声估计 / 语音概率", "谱增益 / 合成"],
  [
    node("pcm", 0, 0, "PCM 10 ms", "原生三采样率", "input", "INPUT",
      "当前与上一 hop 组成 20 ms 分析帧。",
      "OmlsaImcra::process()", "只使用当前和历史 PCM。",
      ["core_rate", "hop"], "hop = rate / 100"),
    node("fft", 1, 0, "STFT / power", "sine WOLA", "core", "STFT",
      "加 sine 窗、补零并计算频点功率。",
      "OmlsaImcra", "FFT 为 2×hop 的下一 2 次幂。",
      ["fft_size"], "P[k] = |FFT(w·x)|²"),
    node("imcra1", 2, 0, "IMCRA I", "平滑 · 最小值", "core", "IMCRA I",
      "时间/频率平滑并跟踪短期最小值，标记语音候选频点。",
      "ImcraNoiseEstimator::update()", "alpha_s = .9。",
      ["imcra_excluded_bins"], "speech = P < 4.6·Bmin·d"),
    node("imcra2", 3, 0, "IMCRA II", "排除语音频点", "core", "IMCRA II",
      "只对噪声候选重做频率平滑，并跟踪条件最小值。",
      "ImcraNoiseEstimator::update()", "无可靠邻点时保留上一条件谱。",
      ["imcra_noise_updates"], "conditional = smooth(P·indicator)"),
    node("psd", 4, 0, "PSD update", "8 × 12 最小值历史", "core", "PSD",
      "用语音后验控制噪声 PSD 递归；突变恢复只做有界更新。",
      "ImcraNoiseEstimator::noise()", "禁止一帧学习大幅瞬态。",
      ["omlsa_noise_power"], "a=.85+.15p; N=aN+(1-a)P"),
    node("posterior", 2, 1, "Speech posterior", "Bayes p", "control", "POSTERIOR",
      "由先验概率、先验 SNR 和后验 SNR 得到语音后验。",
      "omlsa_posterior()", "无外部 VAD。",
      ["omlsa_probability"], "p = 1 / (1 + odds)"),
    node("dd", 3, 1, "DD prior SNR", "causal smoothing", "control", "DD",
      "上一帧 H1 条件 SNR 与当前后验 SNR 形成因果先验。",
      "OmlsaImcra::process() · xi_", "不读取未来谱。",
      ["omlsa_gain"], "ξ=.92ξprev+.08 max(γ-1,0)"),
    node("gain", 4, 1, "OM-LSA gain", "soft gain + floor", "core", "GAIN",
      "H1 增益与语音后验合成软增益，并施加可配置底限。",
      "omlsa_h1_gain()", "底限 −9/−12/−18/−24 dB；禁止放大。",
      ["omlsa_gain"], "G = Gh1^p · Gfloor^(1-p)"),
    node("synthesis", 5, 0, "WOLA synthesis", "IFFT · overlap", "output", "SYNTH",
      "逐频点乘增益，IFFT 后执行 sine WOLA。",
      "OmlsaImcra::process()", "窗口和 hop 固定，声明一个 hop 延迟。",
      ["core_delay_samples", "delay_samples"], "y = overlap_add(IFFT(GX))"),
  ],
  [
    ["pcm", "fft", "audio"], ["fft", "imcra1", "control"],
    ["imcra1", "imcra2", "control"], ["imcra2", "psd", "control"],
    ["psd", "posterior", "control"], ["posterior", "dd", "control"],
    ["dd", "gain", "control"], ["psd", "gain", "control"],
    ["fft", "gain", "audio"], ["gain", "synthesis", "audio"],
  ],
);

const superGaussian = flow(
  "SUPER-GAUSSIAN · STATISTICAL",
  "Super-Gaussian 联合 MAP 谱增益",
  "IMCRA、Gaussian 软语音证据、底限和 WOLA 与 OM-LSA 共用；唯一替换是 H1 条件增益规则。",
  ["共享 IMCRA / 语音后验", "Super-Gaussian 增益"],
  [
    node("input", 0, 0, "PCM 10 ms", "原生三采样率", "input", "INPUT",
      "与 OM-LSA 共用入口、20 ms 分析帧和 WOLA 时基。",
      "OmlsaImcra::process()", "原生 16/24/48 kHz。",
      ["core_rate"], "hop = rate / 100"),
    node("fft", 1, 0, "STFT / power", "共享前端", "core", "STFT",
      "分析窗、FFT 网格和功率谱与 OM-LSA 完全相同。",
      "OmlsaImcra", "保持单变量对照。",
      ["fft_size"], "P[k] = |FFT(w·x)|²"),
    node("imcra", 2, 0, "IMCRA I + II", "条件噪声谱", "core", "IMCRA",
      "两阶段 IMCRA 和最小值历史保持不变。",
      "ImcraNoiseEstimator::update()", "当前帧先使用 t 时刻 PSD。",
      ["imcra_noise_updates", "omlsa_noise_power"], "N[t] = IMCRA(P[t])"),
    node("posterior", 3, 0, "Gaussian posterior", "软语音存在概率", "control", "POSTERIOR",
      "共享 Gaussian 先验与局部/全局语音后验。",
      "omlsa_posterior()", "软概率，不用外部 VAD。",
      ["omlsa_probability"], "p = omlsa_posterior(prior, ξ, γ)"),
    node("h1", 4, 0, "Super-Gaussian H1", "联合 MAP 幅相", "core", "MAP H1",
      "使用 Lotter/Vary 联合 MAP H1 增益，替换 OM-LSA 的 H1 条件增益。",
      "super_gaussian_h1_gain()", "mu=1.74, nu=.126；仍禁止放大。",
      ["omlsa_gain"], "Gmap = f(ξ, γ)"),
    node("floor", 5, 0, "Soft floor mix", "同后验 · 同底限", "core", "FLOOR",
      "用同一语音后验混合 H1 增益和底限；未替换 IMCRA 为真正 Super-Gaussian 噪声模型。",
      "OmlsaImcra::process()", "底限档与 OM-LSA 相同。",
      ["omlsa_gain"], "G = Gmap^p · Gfloor^(1-p)"),
    node("wola", 6, 0, "WOLA synthesis", "共享输出路径", "output", "SYNTH",
      "共享逆 FFT、重叠相加和延迟声明。",
      "OmlsaImcra::process()", "不能把单帧增益差异称为整个前端差异。",
      ["delay_samples"], "y = WOLA(GX)"),
  ],
  [
    ["input", "fft", "audio"], ["fft", "imcra", "control"],
    ["imcra", "posterior", "control"], ["posterior", "h1", "control"],
    ["imcra", "h1", "control"], ["h1", "floor", "control"],
    ["fft", "floor", "audio"], ["floor", "wola", "audio"],
  ],
);

export const algorithms: Algorithm[] = [
  {
    id: "webrtc",
    title: "WebRTC NS",
    short: "默认基线 · 内部噪声估计与谱增益",
    category: "runtime",
    status: "默认可选",
    rate: "16 / 24 / 48 kHz；24 kHz 桥接 48 kHz 核心",
    entry: "NoiseBackend::webrtc",
    cli: "--ns-backend webrtc",
    summary: "低算力、连续在线、无需预录噪声；起音和高 SNR 语音保护主要依赖项目外层控制。",
    boundary: "上游内部 NS 不提供外部 DTD 分类标签；公共证据控制不等于 WebRTC 自身 VAD。",
    flow: webrtc,
  },
  {
    id: "speex",
    title: "Speex NS",
    short: "PCM16 在线估计 · 可双状态动态混合",
    category: "runtime",
    status: "可选",
    rate: "16 / 24 / 48 kHz",
    entry: "NoiseBackend::speex",
    cli: "--ns-backend speex --speex-suppression-db -12",
    summary: "在线噪声估计成本低，支持固定抑制、动态抑制和持续学习的第二强噪声状态。",
    boundary: "固定 suppression dB 是请求值，不是输出语音/噪声的实际衰减。",
    flow: speex,
  },
  {
    id: "rnnoise",
    title: "RNNoise",
    short: "小型循环网络 · 48 kHz 核心",
    category: "runtime",
    status: "可选 · 非默认",
    rate: "16 / 24 / 48 kHz I/O；48 kHz 核心",
    entry: "NoiseBackend::rnnoise",
    cli: "--ns-backend rnnoise",
    summary: "循环网络直接输出增强帧；无需外部模型运行时，适合作为轻量神经基线。",
    boundary: "历史结果出现部分高 SNR/说话人 STOI 退化，因此未替换默认 WebRTC。",
    flow: rnnoise,
  },
  {
    id: "ulunas",
    title: "UL-UNAS",
    short: "有状态 ONNX · 四路缓存",
    category: "runtime",
    status: "可选 · 需 ORT + 模型",
    rate: "16 / 24 / 48 kHz I/O；16 kHz 核心",
    entry: "NoiseBackend::ulunas",
    cli: "--ns-backend ulunas --neural-model <onnx>",
    summary: "因果时频 attention 与显式 cache 的流式谱增强；48 kHz 输出带宽仍为 8 kHz。",
    boundary: "模型缺失或算子失败会明确报错；尚无目标 Android 硬件验收。",
    flow: ulunas,
  },
  {
    id: "omlsa",
    title: "OM-LSA + IMCRA",
    short: "IMCRA 条件噪声 · OM-LSA 软增益",
    category: "runtime",
    status: "可选",
    rate: "原生 16 / 24 / 48 kHz",
    entry: "NoiseBackend::omlsa_imcra",
    cli: "--ns-backend omlsa_imcra --ns-level 2",
    summary: "不依赖训练的统计后端，内部状态和概率可观测；适合作为神经/传统混合主体。",
    boundary: "20 ms sine WOLA / 10 ms hop 是工程运行点，不是论文示例窗长。",
    flow: omlsa,
  },
  {
    id: "super-gaussian",
    title: "Super-Gaussian",
    short: "共享 IMCRA · 联合 MAP H1",
    category: "runtime",
    status: "可选 · 实验档",
    rate: "原生 16 / 24 / 48 kHz",
    entry: "NoiseBackend::super_gaussian",
    cli: "--ns-backend super_gaussian --ns-level 2",
    summary: "将 OM-LSA 的 H1 规则替换为 Super-Gaussian 联合 MAP 增益，前端与底限保持一致。",
    boundary: "当前实现是共享 Gaussian 后验的工程混合，不是完整 Super-Gaussian 噪声模型。",
    flow: superGaussian,
  },
];
