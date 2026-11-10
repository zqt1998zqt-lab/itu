import ScopedAcousticTest from "./ScopedAcousticTest";
import type {
  ScopedTestDefinition,
  ScopedTestPageProps,
} from "./scoped-acoustic-model";

export const DOUBLE_TALK_TEST_DEFINITION: ScopedTestDefinition = {
  scope: "double_talk",
  route: "double-talk",
  title: "双讲测试",
  eyebrow: "上行 · 双讲工程评估",
  description:
    "当近端与远端同时讲话时，验证 VCP 上行链路的回声消除、降噪、发送增益与近端保护能否保留近端语音。评估使用同一时间轴的麦克风混合输入（microphone）、远端播放信号（回声参考，far_end）和干净近端参考（reference），定位双讲区段后计算 VCP 处理后的近端输出（near_end）与麦克风混合输入（microphone）的双讲功率比。",
  boundary:
    "当前功率比是双讲区段内 100 × VCP 处理后的近端输出（near_end）功率 / 麦克风混合输入（microphone）功率，并裁剪到 0–150%，只用于工程复核；它不是逐帧透传率，也不等价于 ITU-T P.1120 A1/A2 正式分类。只有干净近端参考（reference）与远端播放信号（回声参考，far_end）被识别为同时活跃，且有效双讲覆盖不少于约 50 ms 时才产生数值；目标无法分离、有效覆盖不足或输入无效时显式标记为不可测，不补造检测率、误判率或标准等级。",
  inputTitle: "五路同步录音与 VCP 上行输出",
  inputs: [
    "VCP 处理后的近端输出（near_end）：同一时间轴上的 VCP 上行输出，作为双讲功率比的分子来源",
    "麦克风混合输入（microphone）：含近端语音、远端回声与背景噪声的设备麦克风录音，作为功率比分母来源",
    "干净近端参考（reference）：用于识别近端语音活动、对齐麦克风混合输入与 VCP 处理后的近端输出，并约束双讲区段",
    "远端播放信号（回声参考，far_end）：送往扬声器的远端语音，与干净近端参考同时活跃时构成双讲条件",
    "独立噪声输入（noise）：用于背景与静音区段诊断；五路输入必须来自同一场景与时间轴",
  ],
  stages: [
    {
      label: "输入与同步",
      detail:
        "核对 VCP 处理后的近端输出（near_end）、远端播放信号（回声参考，far_end）、麦克风混合输入（microphone）、干净近端参考（reference）与独立噪声输入（noise）的文件身份和采样率，并按共同时间轴裁齐；双讲必须由干净近端参考与远端播放信号的真实重叠活动形成。",
    },
    {
      label: "VCP 上行处理",
      detail:
        "选定 VCP 算法簇处理麦克风混合输入（microphone）：AEC 抑制扬声器回声，NS 降低背景噪声，发送增益调整上行电平，近端与双讲保护限制过度压制；后续测量只使用实际 VCP 处理后的近端输出（near_end）。",
    },
    {
      label: "区段识别",
      detail:
        "从干净近端参考（reference）与远端播放信号（回声参考，far_end）生成活动掩模，区分近端单讲、远端单讲、双讲与静音区段；双讲掩模为两路同时活跃帧，有效覆盖不足约 50 ms 时不计算功率比。",
    },
    {
      label: "功率比代理",
      detail:
        "将麦克风混合输入（microphone）与 VCP 处理后的近端输出（near_end）分别对齐到干净近端参考（reference），只在有效双讲帧累计功率，计算 100 × 输出 / 输入并裁剪到 0–150%；结果保留逐项证据绑定，供人工复核。",
    },
    {
      label: "合同与边界",
      detail:
        "每个业务模式从目录 API 读取两条合同项，分别覆盖标称音量与最大音量条件；界面显示 API 返回的合同行身份，不自行硬编码行号。",
    },
  ],
  outputKind: "tx",
  methods: {
    double_talk_classification: {
      label: "双讲区段输出/输入功率比",
      detail:
        "按双讲活动掩模累计 VCP 处理后的近端输出（near_end）与麦克风混合输入（microphone）的功率，报告 0–150% 的区段级工程代理；标称与最大音量各一项，具体合同行由目录 API 返回。它不是逐帧透传率，也不执行 A1/A2 分类；目标无法分离、覆盖不足或输入无效时不生成数值，也不以零值代替。",
    },
  },
};

export default function DoubleTalkTest(props: ScopedTestPageProps) {
  return (
    <ScopedAcousticTest
      {...props}
      definition={DOUBLE_TALK_TEST_DEFINITION}
    />
  );
}
