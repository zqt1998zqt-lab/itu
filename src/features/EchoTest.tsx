import AcousticStressTest from "./AcousticStressTest";
import type { ScopedTestDefinition, ScopedTestPageProps } from "./scoped-acoustic-model";

export const ECHO_TEST_DEFINITION: ScopedTestDefinition = {
  scope: "echo",
  route: "echo",
  title: "回声测试",
  eyebrow: "上行回声 · TCLw 工程代理",
  description: "在 100 种回声工程组合上比较远端播放信号与麦克风混合输入，形成 TCLw 合同行的软件工程代理。每个组合的干净近端参考为数字静音；在设计远端单讲区段内逐帧核验远端活动与近端静默，再计算输入回声耦合。",
  boundary:
    "结果是合成工程组合上的软件代理：没有执行 TCLw 计权、声压标定和正式标准测点，不替代物理车载声耦合测量，也不是正式 ITU-T 合规。麦克风混合输入包含噪声，输入耦合代理不是纯回声损耗；VCP 发送处理输出（TX）另作补充诊断。",
  inputTitle: "四路输入、数字静音参考与 VCP 补充输出",
  inputs: [
    "麦克风混合输入：本工作区由远端卷积回声和背景噪声合成，不含近端语音。",
    "远端播放信号 / 回声参考：驱动远端单讲与双通道卷积回声路径。",
    "干净近端参考：在回声工作区中为数字静音，用于核验近端静默边界，不作为远端活动的替代证据。",
    "独立噪声输入：与麦克风混合输入保持同一时间轴，用于噪声中的回声组合。",
    "VCP 发送处理输出（TX）：批次执行后的补充诊断，不进入 TCLw 合同代理主公式。",
  ],
  stages: [
    {
      label: "模式与工作负载",
      detail: "按业务模式选择 100 种回声工程组合，五类各 20：稳定远端单讲、启动与收敛、时延变化、路径变化、噪声中的回声。",
    },
    {
      label: "四路输入与静音边界",
      detail: "校验四路输入哈希与活动段，并从实际音频核验干净近端参考的数字静音、远端活动与噪声边界。",
    },
    {
      label: "远端活动区段",
      detail: "在声明的远端单讲区段内，以 20 ms 帧、10 ms 步进核验远端活动与近端静默；加入 100 ms 回声延迟保护，有效活动至少 50 ms 且覆盖率至少 5%。",
    },
    {
      label: "VCP 发送处理",
      detail: "以同一麦克风与远端参考运行所选 VCP 算法簇，保存 TX、诊断和逐项证据；TX 仅作补充诊断。",
    },
    {
      label: "耦合损耗代理",
      detail: "仅在有效远端活动帧上比较远端与麦克风混合输入功率，结果标记为需复核工程代理，不作正式 TCLw 或 AEC ERLE。",
    },
  ],
  outputKind: "tx",
  methods: {
    tclw: {
      label: "TCLw 麦克风输入回声耦合损耗代理",
      detail: "在设计远端单讲区段内核验实际远端活动与近端静默，加入 100 ms 延迟保护后比较远端与麦克风混合输入功率。主代理不依赖 TX；未执行 TCLw 计权、纯回声分离与正式声学标定。",
    },
  },
};

export default function EchoTest(props: ScopedTestPageProps) {
  return <AcousticStressTest {...props} scope="echo" definition={ECHO_TEST_DEFINITION} />;
}
