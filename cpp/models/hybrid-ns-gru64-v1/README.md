# Hybrid NS GRU64 v1

实验模型，**默认关闭，本轮不晋级替代 GRU32**。原生 16/24/48 kHz 各一份，固定 10 ms hop、20 ms 窗、10 ms NS 信号时延，不重采样。

128 维输入 → 单层因果 GRU64 → 64 带概率 + 64 带有界 SNR 修正；共 45,568 参数，每个部署文件 182,312 bytes。沿用 IMCRA/OM-LSA 与控制器 v3，不运行完整 GTCRN。

预设主种子 20261018 的完整隔离 NS 比较：GRU32 **736/847**，GRU64 **717/847**；holdout 均为 **147/180**。三个种子的 validation 通过数差为 **−4、0、+2**，没有稳定优势。模型按训练时的验证 loss 选 checkpoint，没有根据 full/holdout 结果重新选择。

使用需要支持 hidden=64 元数据的新 C++ 运行时；旧的仅 GRU32 加载器会拒绝该模型。CLI 指定 `--ns-backend omlsa_imcra --ns-level 2 --hybrid-ns-model cpp/models/hybrid-ns-gru64-v1/student-16000.bin`，输入率必须匹配模型。24 kHz + AEC3 桥接组合明确拒绝，可使用原生 Speex AEC 或关闭 AEC。

[模型身份与契约](manifest.json) · [完整实验、命令与限制](../../../docs/NS_HYBRID_GRU64_EXPERIMENT_20261018.md)
