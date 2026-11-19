# C64 student 版本化训练源码

本目录保存 `ns-distilled-c64-20261116` 已冻结并完成验证的训练源码，避免训练实现只存在于 build 产物中。除本 README 与 `stage.py` 外，下列文件均从研究根逐字节复制；`student/` 仍保持子目录布局。

## 版本化来源

- 研究根：`/home/zhangqingtian/project/itu/build/ns-distilled-c64-20261116`
- 版本化目录：`/home/zhangqingtian/project/itu/cpp/tools/c64_student`
- 源码位于 Git 管理的 ITU 仓库中；本次收尾不执行提交，交付时应随对应 commit 一并纳入版本历史。
- 数据、训练权重、特征、checkpoint、交付包和验证大件仍是 artifact，不在本目录。

## 源文件 SHA-256

| 相对路径 | SHA-256 |
| --- | --- |
| `prepare_data.py` | `c2f22c0587f3d5b099d6761790aa07ba2c5885c3937406f2e69f475173e7bba2` |
| `prepare_features.py` | `a24736319516c299932e0d314b15d42caf581c4095350a877a26cfe8a2f3cfb6` |
| `fetch_training_noise.py` | `5102bae3f80a13c443e9e82e6ca5b1f9350868573cbdd4d8c3942693d1147391` |
| `train_student.py` | `68eb4595c760d70fefd50752499314fc177ea43fafbcf48bd38ef7e3fa59418f` |
| `training-contract.md` | `9df1e693e708d0d659c0e325806e10e1fabb902c674b8aa8fb30278d51fe9807` |
| `student/model.py` | `43774737c7d7d553ac0eaf320b5f771c47eff2bd6ea5d899e42875596bc9058c` |
| `student/export.py` | `af796ebc50d35159eeaf898bf3eba55d5c416233699e75d40a14ea679632698e` |
| `student/verify.py` | `36d01870b90df44ba90140f18612b5d4c7315b03e4c319d8302499ed4f7a7f10` |
| `student/__init__.py` | `0add710656341cc39d2f3734d6bd4abc4b54d3222cfcd39ad5e892ac71bfdec5` |
| `student/README.md` | `aaaf0aadb5f615bf67044382010a7ddf492b7df921c6eb99c5b3d3132e97aded` |

## 必须先 staging

这些脚本仍保留研究时的路径假设：`prepare_data.py`、`prepare_features.py` 和 `train_student.py` 用 `Path(__file__).resolve().parents[2]` 定位 ITU 项目根，日志路径也仍绑定原研究根。因此不能直接从 `cpp/tools/c64_student/` 运行；先把源码 staging 到 `project/build/<newrun>/`，让脚本层级仍为 `project/build/<newrun>/<script>`。

```bash
cd /home/zhangqingtian/project/itu
python3 cpp/tools/c64_student/stage.py \
  --output-dir build/ns-c64-<newrun>
```

`stage.py` 只复制源码并建立 `data/features/train/logs/native/reports/verification/delivery`，不会下载数据、准备特征或启动训练。它拒绝已存在的非空目标目录。

如需放入已构建的 native probe：

```bash
python3 cpp/tools/c64_student/stage.py \
  --output-dir build/ns-c64-<newrun> \
  --probe /path/to/residual_ns_probe
```

当前研究严格锁定 probe SHA-256：

```text
4d1b3f139214bfad11a6c80e635635026dd2b0c3ce98fc5c4992c8ea4ce5209a
```

哈希不一致时 `stage.py` 会明确报错且不创建 staging 内容；不要改源码中的硬编码锁。未传 `--probe` 时只写 `native/README.md`，说明训练前必须复制 delivery 中已验 binary，或在当前 ITU 机器上按原 `g++ -std=c++20 -O2` 配置重建并核对哈希。

## 机器绑定与复现边界

当前数据和参考路径绑定这台 ITU 机器，尤其包括 `benchmarks/ns/dpdfnet2-16k-v1/`、`build/ns-next-20261115/dpdfnet/assets/dpdfnet2.onnx`、`datasets/itu-eval-v4-curated/sources/noise`、`datasets/itu-eval-v4-no-passenger-20261011/sources/speech` 和 `build/ns-highway120-20261115/`；`fetch_training_noise.py` 也保留绝对路径。不要宣称这套 staging 是跨机器一键通用复现。

精确复现应使用交付包中与 `receipt` 对应的 exact probe、weight 和验证 receipts。版本化目录只固定训练代码与合同，不替代 artifact 包。

## 固定流程

1. 在 `project/build/` 下 staging 到全新目录。
2. 放入 delivery 中已验证且 SHA 匹配的 `residual_ns_probe`，或在本机按锁定配置重建。
3. 按 `training-contract.md` 放入对应数据/特征/manifest artifact。
4. 运行数据准备、特征准备与训练；不要从本版本化目录直接执行。
5. 保存 checkpoint、export、run receipt 与验证 receipts，按交付流程归档。
