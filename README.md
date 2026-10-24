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

生产构建的 `dist/` 只包含网站代码、3D 模型、文档和评测资源清单。
`public/evaluation/` 与 `public/evaluation-cache/` 单独保存，不再随每次前端发布复制；
`dist/evaluation-assets.json` 为每个文件登记 URL、字节数和 SHA-256，并提供整批 release_id。

本机查看生产版本（仍使用同源 `/api`、`/evaluation/...`）：

```bash
npm run build
npm run assets:verify
npm run serve -- --port 5174
```

独立发布资源时，先复制到一个新的版本目录并自动核对哈希，再启动网站：

```bash
npm run assets:publish -- --to /srv/itu-assets/release-20261021
npm run assets:verify -- --root /srv/itu-assets/release-20261021
npm run serve -- --web-root dist --assets-root /srv/itu-assets/release-20261021 --host 0.0.0.0 --port 5173
```

`--to` 不覆盖已有目录。显式指定的资源目录必须含有 `evaluation-assets.json`，
缺失或与网站的 release_id 不一致时拒绝启动；即使清单一致，也会逐文件核对字节数和 SHA-256。
默认使用本地 `public/` 时允许没有发布清单，但仍须通过网站清单的全量文件校验才能启动。
校验仅在启动时进行，资源变更后须重新构建并重启网站服务。
部署时应同时保留对应的 `dist/` 与资源版本目录。原始与冻结数据不被发布脚本修改。
`npm run preview -- --port 5174` 也支持分离的评测资源，可通过
`ITU_EVALUATION_ASSETS_ROOT` 指定上述资源目录。

生产服务支持音频 Range/HEAD、真实 MIME 和缺失资源 404。
API 默认转发到 `127.0.0.1:8001`，历史远端音频按原有规则转发到 `127.0.0.1:18879`，
缓存元数据优先。独立运行 `scripts/evaluation_server.py` 时必须提供 `--directory`，
它只服务历史静态文件；不替代计算 API。使用已有远端评测库时还需保留对应的 SSH 转发。
可通过 `ITU_API_ORIGIN` 和 `ITU_HISTORICAL_ORIGIN` 指定上游；修改环境变量后需重启该网站服务。

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
npm run api                            # 计算 API，默认监听 8001
python3 scripts/evaluation_server.py --directory /path/to/evaluation --port 18879
```

### 音频图形引擎

网站入口：`#graphs`。上传单文件或处理前后音频后，服务会在同一份文件身份和对齐条件下生成
波形与短时能量、对数 STFT、频谱差分、LTAS、ERB、高时间分辨率频谱及共享标尺三联图，
并返回可供网页展示的 PNG 与工程分析 JSON。

接口、计算参数、生成图形和分析边界见
[`docs/AUDIO_GRAPH_ENGINE_20261023.md`](docs/AUDIO_GRAPH_ENGINE_20261023.md)。

### P.1100 / P.1110 / P.1120 声学测试系统

网站入口：`#acoustic`。系统提供夹具/噪声场门禁、服务器 SHA-256 证据库、
PCM 测量包编译、仿真/预合规/正式三种执行模式及逐项结果。

```bash
npm run dev:all
# http://127.0.0.1:5173/#acoustic
```

仓库内只有 P.1120 (10/2025) 全文，P.1100/P.1110 正文尚未落入本地；
系统会把这两类正式运行保持阻断，并根据客户工作簿独立标记诊断结果。
完整拓扑、C1–C5、标准测项、证据格式和验收边界见
[`ITU_ACOUSTIC_QUALIFICATION_SYSTEM.md`](ITU_ACOUSTIC_QUALIFICATION_SYSTEM.md)。

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
