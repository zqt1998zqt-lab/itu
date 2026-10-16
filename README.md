# ITU Audio Lab

车载语音增强（ECNR / VCP）的研究与可视化平台：**React + TypeScript 前端**、**C++ 音频处理引擎**与**离线仿真评测链路**。

处理链：多麦阵列 → 波束形成（BF）→ AEC / 双讲检测（DTD）→ 噪声抑制（NS）→ EQ → AGC；
支持多采样率、多路参考回声、模拟参考时钟与神经网络降噪（可选 ONNX Runtime）。

## 目录结构

| 路径 | 说明 |
|---|---|
| `src/` | React + TypeScript 前端：指标工作台、评测看板、3D 座舱页入口 |
| `server/` | 前端使用的轻量服务层 |
| `scripts/` | 开发启动、流程图生成、评测服务脚本 |
| `tests/` | 前端 / 模型层单元测试（`node:test`）|
| `cpp/include/`、`cpp/src/` | C++20 引擎核心（`itu_audio` 静态库 + `itu_audio_native` 共享库）|
| `cpp/tools/` | 离线工具链：数据处理、指标计算、批量跑测、报告发布 |
| `cpp/tests/` | CTest 用例 |
| `cpp/android/` | Android JNI 集成与示例 App |
| `cpp/cmake/` | 第三方依赖拉取（WebRTC / SpeexDSP / RNNoise / ONNX Runtime）|
| `cpp/licenses/` | 第三方许可与声明 |
| `public/` | 静态资源：`car-3d.html`（3D 智能座舱页）、字体与图标 |

## 快速开始

### 前端工作台

```bash
npm install
npm run dev        # vite --host 0.0.0.0 --port 5173
npm run build      # 生成流程图 + tsc --noEmit 类型检查 + vite 构建
npm run test       # 单元测试
npm run dev:all    # 前端 + 评测服务一起启动
```

### C++ 引擎

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

可选开关：

- `-DITU_BUILD_TOOLS=ON`（默认开启）构建离线工具
- `-DITU_ORT_ROOT=<SDK 路径>` 链接 ONNX Runtime（神经网络降噪）
- `-DITU_ENABLE_SANITIZERS=ON` 开启 ASan / UBSan（仅宿主机）

交叉编译到 Android arm64 时，通过 NDK toolchain 指定 `-DCMAKE_TOOLCHAIN_FILE`。

### 评测服务

```bash
python3 scripts/evaluation_server.py     # 默认监听 8001
```

## 3D 智能座舱页

`public/car-3d.html` 是独立可交互页面（Three.js，纯前端单文件）：

- 四门独立开合（一键展翼 / 展收），门体为贴合车身曲面的曲面门壳
- 外观 360° / 主驾座舱 / 顶部鸟瞰 / 前脸 / 轮毂等视角预设
- 车漆配色、X-Ray 透视、车灯与氛围光开关
- 车内麦克风阵列与扬声器布局、按于扬声器的声学波束可视化

车身按目标车型实车尺寸建模：车长 4997 / 车宽 1963 / 车高 1455 mm，轴距 3000 mm，
前/后轮距 1693 / 1699 mm，轮胎 245/45 R19（胎径 703 mm）。

## 开源范围

**本仓库包含**：前端与引擎源码、构建脚本、单元测试、第三方许可声明、可独立运行的 3D 座舱页。

**本仓库不包含**（需本地生成或另行提供）：

| 内容 | 位置 |
|---|---|
| 评测数据集与冻结输入 | `datasets/` |
| 评测产物与缓存 | `public/evaluation*/`、`build/`、`dist/` |
| 测试契约与指标资料 | `cpp/evaluation/`、`public/downloads/` |
| 内部过程记录、设计与修复日志 | `docs/`、`.wiki/`、`cpp/docs/` |

因此，在全新克隆的仓库里：流程图会在 `npm run build` 时重新生成；
而依赖测试契约的离线跑测工具需要先在本地提供相应数据文件。

## 标准与致谢

算法口径参考 ITU-T 建议书：**G.168**（ERLE 与后级 NLP 延迟衰减的区分）、
**P.835** / **P.863**（语音质量与听感评分口径）。本仓库未接入 P.863 / P.835 的官方算法实现。

第三方组件及其许可见 `cpp/licenses/`。
