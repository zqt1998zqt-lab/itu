# Hybrid NS GRU32 v1

状态：**实验、默认关闭**。原生 16/24/48 kHz 分别加载对应模型。在线算法为 OM-LSA/IMCRA + 64 带微型 GRU；不运行完整 GTCRN，无 Torch/ORT 推理依赖。

每个模型 79,144 bytes / 19,776 参数，固定 10 ms hop，20 ms 分析窗，NS 信号时延 10 ms。128 维因果特征 → GRU32 → 64 带语音优势事件概率及 64 带有界先验 SNR 修正；模型只控制谱增益。

本次交付为 `supervised-v2 / event_v2` 真值监督权重，**没有使用教师损失**。GTCRN 离线蒸馏对照已运行，但未提高验证通过数；不将该结果称为已蒸馏出 GTCRN 同等能力。

| 范围 | 统计基线 | Hybrid |
|---|---:|---:|
| 隔离 NS 场景 | 570/847 | 736/847 |
| holdout | 148/180 | 147/180 |
| cold-start 派生场景 | 40/88 | 47/88 |

总分提升主要在开发集，holdout 未改善且起音衰减加重，尚不能作为默认替代。模型哈希、前端、特征、训练身份及限制见 [manifest.json](manifest.json)，完整证据见[实现报告](../../../docs/NS_HYBRID_STUDENT_IMPLEMENTATION_20261018.md)。

C++ 通过 `VehicleNsConfig.student_model_path` 显式启用，后端必须为 `omlsa_imcra`。CLI 在现有参数后追加：

```bash
--ns-backend omlsa_imcra --ns-level 2 \
--hybrid-ns-model cpp/models/hybrid-ns-gru32-v1/student-16000.bin \
--hybrid-ns-spp-mix 0.5 --hybrid-ns-snr-mix 0.5
```

输入率必须匹配模型。24 kHz + AEC3 路径被明确拒绝，可改用原生 Speex AEC 或关闭 AEC。每流独立状态；模型失败显式回退统计 NS。API/CLI 已接入，C ABI 模型加载与 Android 部署未实现。
