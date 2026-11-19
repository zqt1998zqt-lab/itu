# VCP C64 单头残差模型

`vcp_c64` 是显式可选的原生 16 kHz 单头残差路径，不是默认 profile，也未部署为服务默认。没有指定
`--residual-ns-model` 时，只运行原有 OM-LSA 前端；指定权重后才启用 C64 head。

## 快速使用

```bash
build/ns-distilled-c64-20261116/bin/itu_audio_cli \
  --profile vcp_c64 \
  --mic /absolute/16k-mono.wav \
  --residual-ns-model build/ns-distilled-c64-20261116/train/kd-then-clean-floor-9/export/weights.bin \
  --residual-ns-floor-db -9 \
  --residual-ns-mix 1 \
  --diagnostic-stages \
  --out /tmp/vcp-c64-kd9
```

`vcp_c64` 固定为 16 kHz mono、AEC off、strength 0、原生 OM-LSA、无 raw-mix；后级 EQ/AGC/DRC
沿用 VCP 拓扑。非 16 kHz、非 mono、AEC/BF 开启或修改该 profile 的 NS/raw-mix 不变量会直接报错。

参数：

- `--residual-ns-model PATH`：C64 权重。空/省略表示禁用模型；加载失败在构造期报错。
- `--residual-ns-mix FLOAT`：`[0,1]`，默认 `1`。`0` 仍执行推理，但不提交增益修改。
- `--residual-ns-floor-db FLOAT`：最终增益 clamp 的 floor，`[-40,0] dB`，默认 `-9`。
- 不要把它与旧 `residual_max_db` 或 probe 的 `--floor-db` 混用；probe 中 `--floor-db` 是 OM-LSA base floor，
  `--absolute-floor-db` 才是 controller final floor。

## 训练产物与参数

当前 trained KD9 / supervised9 权重均为 `trained=true`：

| 模型 | weights | SHA256 |
|---|---|---|
| KD9 | `build/ns-distilled-c64-20261116/train/kd-then-clean-floor-9/export/weights.bin` | `eb8d1e3b7a04229a0266968d96bb3a9bc25eda73d09321db76802bfbdb47083a` |
| SUP9 | `build/ns-distilled-c64-20261116/train/supervised-floor-9/export/weights.bin` | `c15f199f6899c90624e8ebbc6dd706958856fa772809505e669be337c478e25d` |

KD9 launch receipt 的实际训练参数为：`mode=kd_then_clean`，KD 8 epochs + clean 12 epochs，
共 20 epochs / 6400 steps（320 steps/epoch），`batch_size=8`、`chunk_frames=200`、
`learning_rate=0.001`、`seed=20261116`、`absolute_floor_db=-9`、CUDA。导出的 best checkpoint
元数据为 step 5120。SUP9 使用相同 seed/batch/chunk/LR/floor，`mode=supervised`；导出元数据为
step 3520。原始 launch receipt 位于 `build/ns-distilled-c64-20261116/logs/train-*-launch.json`。

数据与特征位置：

- `build/ns-distilled-c64-20261116/data/manifest.json` 与 `data/audio/<case>/{mix,clean,noise}.wav`
- `build/ns-distilled-c64-20261116/features/manifest.json`
- `features/native-inputs/*.mix-pcm16.wav`：与 C++ 一致的 PCM16 量化输入
- `features/npz/*.npz`：训练张量；`features/teacher/*.teacher.wav`：冻结 teacher 输出
- `features/native/*.raw.f32`：native probe 特征审计副本
- `train/kd-then-clean-floor-9/training-manifest.json`：逐 case 路径、SHA、teacher 对齐和 4000-frame 校验

特征契约是 `dpdf-frame-v1-db`：16 kHz、FFT 512、hop 160、window 320、256 输入维、64 bands、
GRU hidden 64、45568 参数。teacher 仅用于离线蒸馏/导出，运行期
`residual_ns_teacher_in_runtime=false`。

KD9/SUP9 始终绑定稳定特征 manifest `ae676c55783a561f409cabe2f686e0f954e4e6373f826529af3eac3beb84408c`。
当前 data manifest `a0593ea5e33fbdd85dfd20c1aae6bda5c600e73482aaaadb91cb6e07a22348e3`
只是 feature cache 之后的 metadata enrichment：按 `build/ns-distilled-c64-20261116/data-provenance.md`
删除七个逐行字段和一个顶层
`features` 字段后可精确重建 feature-cache 输入 `c3001b...`；没有重建 waveform/feature，也不需要重训。

## 处理边界与验收

- Head 输出为 `12*tanh(head_logit)` dB，范围 `[-12,+12]`；以 OM-LSA base floor `-9` 计，
  未加最终 clamp 时可表达到 `-21 dB`。实际下限仍由 `--residual-ns-floor-db` 再夹紧。
- 模型只改 synthesis `gain[]`，不改 OM-LSA 的 PSD、DD、概率或旧 student 统计。
- 常规 `vcp_c64` NS delay 为 160 samples。JSON 分开报告 `residual_ns_*` 与旧 `hybrid_ns_*`。
- `--diagnostic-stages` 导出 `tx_ns_input`，语义是进入 `VehicleNsProcessor` 前的实际 mono PCM，
  不是 post-control 的 `tx_ns_selected`。
- 3 个新增 dev 结构验证例（en mixed / zh clean / en noise_only）使用正式 CLI 和
  trained KD9：integrated `state=1`、`inferences/applied=4001`、`fallback=0`、
  `parameters=45568`、`raw_mix_max=0`、`delay=160`；CLI `tx_ns` 与 native probe 输出逐样本
  `max_abs=0.0`。证据见 `build/ns-distilled-c64-20261116/reports/integrated-equivalence/`。
  这些是 dev 结构检查，不是 frozen-v1 质量测试；真正的 frozen 单例证据是 root 的
  `reports/integrated-trained/zh01-5_2399-snr00`。

## 质量评分与限制

需要 raw / -30 LUFS 的逐阶段 G/S/N 时，必须用冻结 scorer 生成，不得手填或从不同协议拼接：

```bash
/home/zhangqingtian/.venv/bin/python cpp/tools/vcp_stage_mos.py \
  --cli build/ns-distilled-c64-20261116/bin/itu_audio_cli \
  --profile vcp_c64 \
  --mic /absolute/16k-mono.wav \
  --assets-dir build/ecnr-v2-20261011/perceptual-assets \
  --output-dir /tmp/vcp-c64-score \
  -- --residual-ns-model build/ns-distilled-c64-20261116/train/kd-then-clean-floor-9/export/weights.bin \
     --residual-ns-mix 1 --residual-ns-floor-db -9
```

输出目录必须为新目录；scorer 同时保留实际电平与 -30 LUFS 副本、阶段 PCM、窗口和 hash。

限制：本轮 zh 训练数据/质量覆盖仅为 1 个 train + 1 个 dev，不能据此宣称跨说话人、跨设备或
所有噪声类型有效；这不是普适 SOTA，也不替代盲听、连续自然对话和真实车舱验证。不要修改
服务默认或把 `vcp_c64` 设为默认 profile。
