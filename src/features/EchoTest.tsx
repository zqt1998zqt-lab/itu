import ScopedAcousticTest from "./ScopedAcousticTest";
import type {
  ScopedTestDefinition,
  ScopedTestPageProps,
} from "./scoped-acoustic-model";

export const ECHO_TEST_DEFINITION: ScopedTestDefinition = {
  scope: "echo",
  route: "echo",
  title: "回声测试",
  eyebrow: "上行回声 · TCLw 工程代理",
  description:
    "从测试集已有的麦克风混合输入（microphone）回声录音出发，在远端单讲区段比较远端激励功率与麦克风混合输入中的回声功率，得到 TCLw 合同行的“麦克风输入回声耦合损耗”工程代理；VCP 发送处理输出（TX）仅作补充诊断。",
  boundary:
    "结果仅是已录或合成麦克风混合输入回声的软件工程代理：远端单讲功率比没有经过 TCLw 计权、声压标定和标准测点，不替代人工嘴/近端静默条件下的物理车载声耦合测量，也不是正式 ITU-T 合规。本页不会为每个合同行重放接收处理输出（RX）声场；该合同代理不依赖 VCP 发送处理输出（TX），也不是 AEC ERLE。",
  inputTitle: "回声采集、远端播放信号 / 回声参考与合同工况",
  inputs: [
    "测试集条件组中的麦克风混合输入（microphone）WAV；可以是已录或合成麦克风回声，已含耦合关系的录音直接复用，本页不重新生成物理声场。",
    "与同一条件组绑定的远端播放信号（回声参考，far_end）激励 WAV；VCP 将其作为 --echo 远端播放信号，TCLw 代理也用它计算远端单讲功率。",
    "干净近端参考（reference）、VCP 处理后的近端输出（near_end）与独立噪声输入（noise）证据；其中干净近端参考活动与远端播放信号活动共同限定远端单讲掩码，VCP 处理后的近端输出与独立噪声输入保留用于报告审计。",
    "合同模式与音量档从 tclw 测项名称解析：NB/WB 每个业务模式 2 项，SWB 每个业务模式 3 项；nominal、recommended 或 maximum 以目录中的实际行和门限为准。",
    "正式测量还要求人工嘴/近端静默、校准声耦合路径和对应计权；这些物理条件不在本软件批次内自动补齐。",
  ],
  stages: [
    {
      label: "合同条件与回声证据",
      detail:
        "按所选业务模式从测试集取出麦克风混合输入（microphone）、远端播放信号（回声参考，far_end）及其合同音量/场景绑定；NB/WB 为 2 项，SWB 为 3 项。",
    },
    {
      label: "远端激励与耦合录音",
      detail:
        "远端播放信号（回声参考，far_end）作为远端激励；麦克风混合输入（microphone）信号来自数据集已录或合成回声，不逐行声明真实接收处理输出（RX）声场重放。",
    },
    {
      label: "VCP 发送处理（TX）/ AEC / RES",
      detail:
        "CLI 使用 --mic 与 --echo 执行所选 VCP 发送处理（TX）；AEC 后端、Coherence RES 与后续处理以实际 profile/flags 和报告为准。",
    },
    {
      label: "远端单讲区段",
      detail:
        "20 ms 帧、10 ms 跳帧建立活动掩码；远端单讲要求远端播放信号（回声参考，far_end）活跃且干净近端参考（reference）不活跃，用于选取麦克风混合输入（microphone）中的回声，而非 AEC 后残余。",
    },
    {
      label: "麦克风输入回声耦合损耗",
      detail:
        "麦克风混合输入（microphone）对齐干净近端参考（reference）后，在远端单讲掩码上计算 10log10(P_far_end / P_microphone_echo)；结果映射到合同 tclw 行并标记为需复核。",
    },
  ],
  outputKind: "tx",
  methods: {
    tclw: {
      label: "TCLw 麦克风输入回声耦合损耗代理",
      detail:
        "以远端播放信号（回声参考，far_end）活跃且干净近端参考（reference）不活跃的远端单讲区段保留参考条件，麦克风混合输入（microphone）对齐干净近端参考后计算 10log10(P_far_end / P_microphone_echo)，即远端播放信号功率与麦克风混合输入中的回声功率之比；该合同代理不依赖 VCP 发送处理输出（TX），也不是 AEC ERLE。未执行 TCLw 计权与正式声学标定。",
    },
  },
};

export default function EchoTest(props: ScopedTestPageProps) {
  return (
    <ScopedAcousticTest
      {...props}
      definition={ECHO_TEST_DEFINITION}
    />
  );
}
