import { AudioLines, FileAudio, GitBranch, Mic, ScanLine, ShieldCheck, SlidersHorizontal, Upload } from "lucide-react";
import type { LucideIcon } from "lucide-react";

type NodeDetails = {
  kind: string;
  icon: LucideIcon;
  description: string;
  requirements: string;
  metrics: string[];
};

// Titles and recording roles come from the shared diagram model.
export const FLOW_NODE_DETAILS: Record<string, NodeDetails> = {
  reference: {
    kind: "录音 01 · 干净近端参考", icon: FileAudio,
    description: "保存进入近端声源的干净语音，与麦克风中的同一段近端语音对应。它和扬声器播放的远端参考是两路不同信号。",
    requirements: "保留同一段语音源与时序关系。STOI 需要真实干净语音确认，合成非语音信号不能用于可懂度评分。",
    metrics: ["相对时延", "SI-SDR / 相对增益", "STOI", "H1 传递函数"],
  },
  voice: {
    kind: "采集场景 · 近端语音", icon: AudioLines,
    description: "近端声源把参考语音带入实际声学场景，麦克风会同时采到环境噪声及可能的扬声器回声。",
    requirements: "记录声源与设备距离、播放电平和场景。用于语音保真比较时，应能找到对应的干净语音；正式声学测试另需规定声源及标定。",
    metrics: ["语音保真", "语音可懂度", "参考片段"],
  },
  noise: {
    kind: "采集场景 · 噪声", icon: AudioLines,
    description: "环境或测试噪声进入麦克风，纯噪声片段用来比较处理前后的噪声功率。",
    requirements: "纯噪声段不得含近端语音或扬声器回声。片段 SNR 还需同一噪声平稳且与语音不相关，不能只凭前后功率下降判断语音质量改善。",
    metrics: ["噪声片段 RMS", "噪声衰减", "片段估计 SNR"],
  },
  echo: {
    kind: "录音 04 · 远端播放参考", icon: FileAudio,
    description: "在播放前保存送往扬声器的远端信号。它进入声学播放路径，也作为 AEC 的参考输入。",
    requirements: "保留实际播放时序，与麦克风录音同步。网站用相关性核对远端单讲段，不用它代替干净近端语音。",
    metrics: ["回声相关性", "远端单讲回声衰减", "AEC ERLE 工程估计"],
  },
  speaker: {
    kind: "采集场景 · 扬声器与声学路径", icon: AudioLines,
    description: "远端参考经扬声器播放，经过房间、设备和摆放位置构成的声学路径进入麦克风，形成回声。",
    requirements: "保留只有扬声器发声、没有近端语音的远端单讲片段，记录播放电平与设备位置。数字录音的功率衰减不直接等于声学 TCL。",
    metrics: ["远端单讲片段", "回声相关性", "回声片段衰减"],
  },
  before: {
    kind: "录音 02 · 原始麦克风", icon: Mic,
    description: "在处理前导出实际麦克风的语音、噪声和回声混合信号，作为设备处理的输入及前后比较的基准。",
    requirements: "与处理后录音来自同一次采集，保留时钟、电平与时间关系。BF 另需同步阵列通道及麦克风位置；网站仅分析所选声道，不自动运行 BF。纯噪声和远端单讲窗口使用处理前时间轴，处理后可填写延迟补偿。",
    metrics: ["处理前电平", "处理前 SI-SDR / STOI", "噪声与回声功率基准"],
  },
  aec: {
    kind: "设备或离线程序 · AEC", icon: ShieldCheck,
    description: "AEC 使用前置 BF 的单路波束与实际播放参考估计并消除声学回声。图示与当前 VCP 实现均先 BF、再 AEC；DTD 控制自适应更新。网站测量已经录下来的结果，综合增强链路页展示可回放配置。",
    requirements: "确认真实设备的处理模块、参考时序与测量点。单独评估滤波器时需要其经过 BF 的输入和残余回声抑制、降噪及 AGC 之前的输出，保持同一波束与电平基准。",
    metrics: ["AEC ERLE 工程估计", "残余回声", "DTD 自适应保护", "远端单讲条件"],
  },
  bf: {
    kind: "设备或离线程序 · 波束形成示意", icon: AudioLines,
    description: "BF 利用多个同步麦克风的空间信息进行滤波与合成，形成面向目标方向的拾音输出。图示与当前 VCP 实现均在 AEC 之前形成单路波束；综合增强链路可选择标定延迟求和、MVDR 与风压组合。",
    requirements: "需要至少两个同步麦克风通道、阵列几何及目标方向。效果比较应固定声源方向、距离和电平，并保留 BF 前后测点。网站只分析所选声道，不执行阵列合成；单声道录音不能直接计算指向性或阵列增益。",
    metrics: ["输出 SI-SDR / STOI", "噪声衰减与片段 SNR", "空间指标需阵列测试"],
  },
  dtd: {
    kind: "设备控制支路 · 双讲检测示意", icon: ShieldCheck,
    description: "DTD 观察麦克风与播放参考，判断近端和远端是否同时讲话。蓝色虚线把检测状态送给 AEC，用于冻结或减缓自适应更新，保护近端语音；它不是串行音频处理器。",
    requirements: "需要同步近端与远端语音、单讲和双讲测试段。检测率、误判率及检测延迟另需逐帧 DTD 状态、时间戳和人工标注；网站目前不执行双讲分类，也不自动计算这些专门指标。双讲语音保真与单讲 ERLE 应分开评估。",
    metrics: ["双讲语音保留", "AEC 自适应保护", "检测率 / 误判率需状态日志"],
  },
  tap: {
    kind: "可选留样 · AEC 滤波器输出", icon: GitBranch,
    description: "在前置 BF 之后，保留 AEC 滤波器的输入与线性残差输出；输出留样位于残余回声抑制、降噪和增益控制之前，用来隔离滤波器本身的消除效果。灰色虚线表示单独的 ERLE 测量用途。",
    requirements: "单独测 ERLE 时，用滤波器输入和输出替换处理前后两路，仍按四个上传角色操作。确认排除后级附加衰减；最终输出不能自动当作滤波器输出。",
    metrics: ["AEC ERLE 工程估计", "与整条链路总衰减区分"],
  },
  residual: {
    kind: "可选设备后级 · 示意", icon: ShieldCheck,
    description: "残余回声抑制继续降低滤波器留下的回声，可能贡献额外衰减。它与滤波器 AEC 是两个测量边界不同的环节。",
    requirements: "是否启用及模块顺序以设备实现为准。评估最终输出总效果可包含本模块；隔离滤波器 ERLE 时必须使用本模块之前的留样。",
    metrics: ["最终输出回声衰减", "语音保留", "ERLE 测量边界"],
  },
  denoise: {
    kind: "设备后级 · 降噪示意", icon: SlidersHorizontal,
    description: "降噪降低背景噪声，同时可能影响语音细节。这里展示常见处理环节，网站没有执行降噪算法。",
    requirements: "导出实际处理前后录音及纯噪声片段。结合语音保真、可懂度和相对增益观察效果，避免把整体降音量误认为降噪改善。",
    metrics: ["噪声衰减", "片段 SNR", "SI-SDR / STOI 变化"],
  },
  gain: {
    kind: "可选设备后级 · AGC 示意", icon: SlidersHorizontal,
    description: "自动或固定增益控制调节最终电平。增益变化会同时改变语音、底噪与回声功率，需与算法消除效果分开观察。",
    requirements: "是否启用及位置以设备实现为准。记录共同电平基准，检查峰值和满幅比例；SI-SDR 对整体增益不敏感，另报相对增益。",
    metrics: ["相对增益", "RMS / 峰值", "满幅比例", "功率比解释"],
  },
  after: {
    kind: "录音 03 · 最终输出必需", icon: FileAudio,
    description: "保存设备实际最终输出。只有这一路时也能查看电平、波形和频谱；结合其他角色及条件后才能计算成对指标。",
    requirements: "优先使用未压缩 WAV，保留与处理前相同的时钟和电平基准。最终输出可能含多个模块效果，不直接等同于 AEC 滤波器输出。",
    metrics: ["单文件电平", "波形与频谱", "成对比较", "整条链路回声衰减"],
  },
  eq: {
    kind: "设备处理 · 上行均衡", icon: SlidersHorizontal,
    description: "车规级上行参量均衡补偿，修正座舱结构与麦克风安装频响缺陷，增强语音关键频段清晰度与通透感。",
    requirements: "频响补偿滤波器应在 AEC 消除之后或协同配置，避免滤波引起额外相位畸变破坏 AEC 回声消除。",
    metrics: ["Welch PSD", "频谱平坦度", "共振峰保真度"],
  },
  rx_nr: {
    kind: "设备下行 · 远端降噪", icon: SlidersHorizontal,
    description: "下行噪声抑制算法，净化接收自蜂窝网络或蓝牙远端的背景噪声，提升车舱乘客对远端语音的听感与可懂度。",
    requirements: "在远端语音播放前对输入流进行降噪，保证送入功放前底噪受控；需防范下行降噪带来的语音失真。",
    metrics: ["下行噪声衰减", "下行 STOI", "远端语音听感"],
  },
  rx_eq: {
    kind: "设备下行 · 扬声器均衡", icon: SlidersHorizontal,
    description: "根据车机扬声器频响与车舱内部声学特性进行频域均衡校准，消除车门或顶棚共振，确保下行通话声音自然饱满。",
    requirements: "需配合车规声学标定参数（如 5 段或 8 段 PEQ）；均衡应在线性工作区内调整，防止大增益引起扬声器削波。",
    metrics: ["扬声器频响校准", "H1 传递函数", "失真抑制"],
  },
  rx_drc: {
    kind: "设备下行 · DRC 与功放保护", icon: ShieldCheck,
    description: "下行动态范围控制（DRC）与智能限幅。限制下行最大输出电平，防止车机功放与扬声器破音削波，同时为上行 AEC 导出无削波的稳定参考信号。",
    requirements: "必须先于物理扬声器功放执行，且送往 AEC 的回声参考信号必须从本模块之后（或与功放输入完全同步的测量点）引出。",
    metrics: ["最大峰值保护", "满幅比例", "AEC 参考相关度", "下行响度稳定度"],
  },
  upload: {
    kind: "网站 · 输入", icon: Upload,
    description: "选择干净参考、麦克风处理前、处理后和播放参考四个角色的实际音频，并选择每个文件的分析声道。",
    requirements: "处理后音频必需，其余角色按指标提供。ERLE 测量用滤波器输入和输出替换前后两路。文件已选择只表示输入到位，不表示计算条件已满足。",
    metrics: ["录音角色", "分析声道", "实际音频输入"],
  },
  decode: {
    kind: "网站 · 解码与有效性检查", icon: ScanLine,
    description: "读取实际文件内容，检查格式、时长、采样率和声道；主解码失败时使用支持的回退路径，保留文件身份与原始信息。",
    requirements: "选择存在的声道，不对多声道自动取平均。损坏文件和不支持的内容需先解决；文件电平与频谱使用实际原始采样率。",
    metrics: ["文件 SHA256", "原始采样率 / 时长", "所选声道", "解码错误原因"],
  },
  conditions: {
    kind: "网站 · 片段与测量条件", icon: SlidersHorizontal,
    description: "填写语音比较段、纯噪声段和远端单讲段，按实际录音确认共享时钟、电平、真实语音、平稳噪声和 AEC 测量点。",
    requirements: "各指标确认项有独立前提。STOI 需真实语音，SNR 需平稳噪声，回声需远端单讲，ERLE 需滤波器测点；不能为了得到分数而全部勾选。",
    metrics: ["语音条件", "纯噪声条件", "远端单讲条件", "ERLE 测点"],
  },
  align: {
    kind: "网站 · 成对重采样与对齐", icon: ScanLine,
    description: "成对分析统一重采样，去直流后通过互相关搜索相对时延并核对可信度。文件电平、波形和 PSD 仍保留原始采样率。",
    requirements: "需要相同内容、有效时长和足够相关性。低相关、搜索边界峰和相近多峰会拒绝对齐结果；相对时延不是正式往返时延 RTD。",
    metrics: ["相对时延", "相关性", "有效重合时长", "对齐失败原因"],
  },
  windows: {
    kind: "网站 · 片段复核与共有参考", icon: ScanLine,
    description: "复核对齐后的有效重合范围，并让处理前后保真比较使用同一组参考样本。噪声与单讲片段按处理前时间轴及处理后补偿取窗。",
    requirements: "窗口不能越界或过短。数字静音和低能量片段不产生虚假的比值；共有参考不足时不输出前后改善量。",
    metrics: ["共有参考范围", "有效语音片段", "噪声 / 单讲窗口", "延迟补偿"],
  },
  levels: {
    kind: "网站 · 单文件指标计算", icon: ScanLine,
    description: "使用实际所选声道计算 RMS、峰值、直流偏置和满幅样本比例，并展示波形，检查静音、低能量及接近满幅的情况。",
    requirements: "幅度 1 FS 为数字满幅基准；满幅比例阈值为绝对幅度至少 0.999。数字 dBFS 不等于声压 dB SPL 或 dBm0，波形满幅比例也不直接证明模拟失真。",
    metrics: ["RMS dBFS", "峰值 dBFS", "DC 偏置", "满幅样本比例", "波形"],
  },
  fidelity: {
    kind: "网站 · 语音保真与可懂度计算", icon: ScanLine,
    description: "使用可信对齐后的干净参考和待测音频计算 SI-SDR 与相对增益；确认真实语音且有效帧足够时计算 standard STOI。",
    requirements: "SI-SDR 对整体增益不敏感，需同时观察相对增益。STOI 需真实干净语音，有效帧不足时不输出占位分数；两者均不映射为 MOS。",
    metrics: ["SI-SDR", "STOI", "相对增益", "相对时延"],
  },
  "noise-metrics": {
    kind: "网站 · 噪声指标计算", icon: ScanLine,
    description: "在纯噪声段计算前后 RMS 和 10log10(P前/P后) 噪声衰减。在确认平稳且与语音不相关的噪声后，用混合功率减噪声功率估计片段 SNR。",
    requirements: "衰减比较需共同电平和时钟。混合功率不大于噪声时不输出 SNR；整体降增益也会降低噪声功率，需结合语音保真解释。",
    metrics: ["噪声片段 RMS", "噪声衰减", "片段估计 SNR"],
  },
  "echo-metrics": {
    kind: "网站 · 回声指标计算", icon: ScanLine,
    description: "用处理前、处理后和播放参考核对远端单讲片段，计算整条链路功率衰减。额外确认滤波器测点后，单独输出 AEC ERLE 工程估计。",
    requirements: "需要远端单讲、共同电平和时钟，并通过播放参考相关性检查。ERLE 必须排除后级抑制、降噪和 AGC；没有执行完整 G.168 测试，也不替代 TCL。",
    metrics: ["远端单讲回声衰减", "AEC ERLE 工程估计", "回声相关性"],
  },
  spectrum: {
    kind: "网站 · 频域分析", icon: ScanLine,
    description: "使用 Welch 计算文件 PSD 和谱峰，在可信对齐的成对片段上估计 H1 传递函数，并展示相干度与有效频点。",
    requirements: "PSD 单位为 dBFS/Hz。H1 仅显示参考相对功率大于 −40 dB 且相干度至少 0.6 的频点；未做声学标定，没有原表正式合规掩模。",
    metrics: ["Welch PSD", "谱峰", "H1 传递函数", "相干度"],
  },
  compare: {
    kind: "网站 · 前后效果比较", icon: ScanLine,
    description: "汇总各分支的有效结果，对照噪声、回声、语音保真和相对增益，判断处理链路的改善与退化。各计算分支独立，缺失条件不会伪造分数。",
    requirements: "SI-SDR 与 STOI 前后变化需要共有参考样本、有效状态及共同电平时钟。不能单凭噪声或回声功率下降评价总体质量。",
    metrics: ["SI-SDR 改善", "STOI 变化", "噪声 / 回声衰减", "相对增益"],
  },
  report: {
    kind: "网站 · 最终指标与报告", icon: FileAudio,
    description: "展示计算结果、前后比较、波形和频谱，并导出 CSV 或完整 JSON，保留文件身份、声道、采样率、片段、方法和不可计算原因。",
    requirements: "工程结果与条件必须一同保存。MOS、SLR / RLR、TCL 等正式指标仍需专用算法、测试配置或声学标定，不能把当前数值作为完整 ITU 认证。",
    metrics: ["最终指标", "有效性与不可计算原因", "波形 / 频谱", "CSV / JSON"],
  },
};
