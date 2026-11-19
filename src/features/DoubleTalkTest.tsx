import AcousticStressTest from "./AcousticStressTest";
import type { ScopedTestDefinition, ScopedTestPageProps } from "./scoped-acoustic-model";

export const DOUBLE_TALK_TEST_DEFINITION: ScopedTestDefinition = {
  scope: "double_talk",
  route: "double-talk",
  title: "双讲测试",
  eyebrow: "上行 · 双讲工程评估",
  description: "在 100 种双讲工程组合上验证 VCP 对近端语音、远端回声和背景噪声的联合处理。在设计双讲窗口内，逐帧核验近端与远端同时活动，仅在实际重叠帧上计算 TX 与麦克风功率比。",
  boundary: "结果是工程代理，不是逐帧透传率，也不等价于 ITU-T P.1120 A1/A2 正式分类。目标无法分离、有效覆盖不足或输入无效时显式标记为不可测，不补造检测率、误判率或标准等级。",
  inputTitle: "四路同步输入与 VCP 发送输出",
  inputs: [
    "麦克风混合输入：含近端语音、远端卷积回声和背景噪声。",
    "干净近端参考：承载近端语音真值并声明近端活动段。",
    "远端播放信号 / 回声参考：承载远端语音并驱动回声路径。",
    "独立噪声输入：与麦克风混合输入、干净近端参考、远端播放信号保持同一时间轴。",
    "VCP 发送处理输出（TX）：批次实际执行后用于双讲功率比与诊断。",
  ],
  stages: [
    {
      label: "模式与工作负载",
      detail: "按业务模式选择 100 种双讲工程组合，五类各 20：均衡、近端较强、远端较强、切换、短时重叠。",
    },
    {
      label: "四路输入与设计重叠窗口",
      detail: "校验四路输入哈希与设计活动段；以 20 ms 帧、10 ms 步进核验近端与远端同时活动，实际重叠至少 50 ms 且覆盖率至少 5%。",
    },
    {
      label: "VCP 上行处理",
      detail: "选定 VCP 算法簇处理麦克风混合输入，保存 TX 与逐项证据；所有测量绑定同一工程组合的四路输入哈希。",
    },
    {
      label: "双讲区段诊断",
      detail: "仅在实际联合活动帧上比较 TX 与麦克风功率，记录近端、远端、重叠活动时长和覆盖率；活动不足或 TX 静音时显式标记不可测。",
    },
    {
      label: "工程复核边界",
      detail: "结果只作工程复核，不输出 P.1120 A1/A2 等级、检测率或误判率；正式合规状态始终为未测。",
    },
  ],
  outputKind: "tx",
  methods: {
    double_talk_classification: {
      label: "双讲区段输出/输入功率比",
      detail: "在设计双讲窗口内核验实际同帧近端 / 远端活动，仅累计满足活动门限的重叠帧功率。报告区段级工程代理，不是逐帧透传率，也不执行 A1/A2 分类。",
    },
  },
};

export default function DoubleTalkTest(props: ScopedTestPageProps) {
  return <AcousticStressTest {...props} scope="double_talk" definition={DOUBLE_TALK_TEST_DEFINITION} />;
}
