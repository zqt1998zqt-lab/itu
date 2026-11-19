# LocalVQE Android FP32 optimization candidates

本工具在独立目录构建和运行候选，复用本工作区已冻结的 Android 工具链、GGML/LocalVQE 对象与模型。不修改产品默认路线，不覆盖旧库，不自动删除文件。需要当前工作区中的 `build/android-native16-20261117/root`、原始 LocalVQE/GGML 依赖和通过 ADB 连接的 `cb8145db`。

| 候选 | 卷积布局 | FP32 SGEMM | DAF 批量 GRU |
| --- | --- | --- | --- |
| A | 原版 | 原版 | 原版 |
| B | 减少符合条件的 CONT 重排 | 原版 | 原版 |
| C | 原版 | 现有 ARM NEON SGEMM | 原版 |
| D | 优化 | 优化 | 原版 |
| E | 原版 | 原版 | 保序 SIMD |
| F | 优化 | 优化 | 保序 SIMD |

本轮实测选择 **D**。E/F 保留作可复现候选：DAF 批量实现数值正确，当前设备上的计时收益未胜过 D。不能根据更多 SIMD 代码或构建选项推断加速。

## 验证与构建

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zhangqingtian/.venv/bin/python -B \
  /home/zhangqingtian/project/itu/scripts/localvqe_fp32_opt/verify.py

PYTHONDONTWRITEBYTECODE=1 /home/zhangqingtian/.venv/bin/python -B \
  /home/zhangqingtian/project/itu/scripts/localvqe_fp32_opt/build.py \
  --variants A B C D --without-probe
```

构建器输出新的 `build/localvqe-fp32-opt/build-时间/MANIFEST.json`，绑定每个 CLI、库、实际编译命令与 SHA。`--without-probe` 保留原版 API；省略它会额外构建状态探针，并对调试 VIEW 的 backing owner 保持存活以便正确读取。诊断构建不用于最终性能数字。

## 手机运行

将下面 `--manifest` 替换为新构建输出的真实路径。此命令运行完整 BF → LocalVQE → CoFi-Lite → EQ → Guarded5 → DRC → Limiter；`--ns-backend webrtc` 是旧枚举兼容参数，实际 NS 仍由 CoFi 回调执行。

```bash
PYTHONDONTWRITEBYTECODE=1 /home/zhangqingtian/.venv/bin/python -B \
  /home/zhangqingtian/project/itu/scripts/localvqe_fp32_opt/run_v2.py \
  --manifest /绝对路径/MANIFEST.json --variant D \
  --label fp32-opt-check --seconds 60
```

使用 `--seconds 600` 运行冻结的 10 分钟双讲输入。五个 12 秒条件可用 `--seconds 12 --scene near_single|far_single|double_talk|transition|noise_only`；不加 `--unpaced` 时按 16 ms 节拍回放。性能测试关闭 `--audit`，需要真实 SGEMM 命中证据时单独增加它。

运行器核对每个模型和输入的**文件路径＋SHA**、CPU7 亲和性及单任务占用。测试由手机端独立 supervisor 执行，保存开始/结束/退出码；重新下载完整结果后才标记进程完成。`process_complete` 只表示退出码 0，不能当作质量或硬实时通过。

## 数值与边界

- B/E：逐帧 PCM 和递归状态目标为 FP32 bit-exact。
- C/D/F：状态 `atol=1e-5, rtol=1e-4`；整数状态完全相同；浮点输出最大差不超过 `1e-5`；整链 PCM16 最大差不超过 1 LSB。
- 原生 16 kHz/256 点、正确 EOS、513 点固定 PCM 时间轴和零 fallback 保持不变。
- 既有 debug taps 在性能基线和候选中都保留；本轮未将“去掉原有调试输出”混入优化变量。
- `validation-root` 的输出探针覆盖持久状态及有效的 block 边界输出，不读取已被 allocator 复用的任意中间节点。
- 最终截止时间验收必须同时统计处理超过 16 ms 和完成截止失约。文件回放不等于 AudioRecord/AudioTrack 端到端，更不代表正式 ITU-T 合规。

实施方案：[完整计划](/home/zhangqingtian/project/itu/docs/localvqe-android-optimization-implementation-20261117.md)。实测报告：[结果与证据](/home/zhangqingtian/project/itu/docs/localvqe-android-optimization-results-20261117.md)。
