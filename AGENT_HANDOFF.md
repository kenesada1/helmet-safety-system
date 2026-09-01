# AGENT HANDOFF — 头盔安全检测系统交接文档

> **本文件是什么**：一份交给下一个 AI agent 的上下文移交文档。它浓缩了项目定位、成熟度结论、模型对比、实验纪律与已知缺口。
> **生成日期**：2026-09-01
> **给谁**：任何将要接手本项目（`d:\codes\helmet-safety-system`）的 agent
> **如何使用**：先通读本文件建立认知；动手前**务必再读** `任务清单.md`；需要细节时按文末「关键文件索引」查证。**本文件是结论移交，不是全部事实——涉及具体数字与代码逻辑时，以当前代码与 `artifacts/` 报告为准。**

---

## 1. 一句话定位

基于 **PyTorch / OpenCV / Ultralytics YOLO** 的头盔佩戴安全检测系统，覆盖 **数据治理 → 训练 → 诊断评估 → 实时推理 → 多目标追踪 → ONNX 部署** 全链路，以 SHWD 为数据源，通过十轮受控实验完成模型选型与结构优化。

**核心信条：任何结论都必须有数据支撑，任何实验都必须可复现、可归因、可证伪。**

## 2. 项目当前状态总览

| 维度 | 状态 |
|---|---|
| 数据 | SHWD，两类（`helmet`/`no_helmet`），5,457 训练 / 607 验证 / 1,517 测试 |
| 主模型（生产候选） | **E4**：YOLO11s，imgsz=960，75 轮，权重 `artifacts/training/m45_yolo11s_e75_960_001/weights/best.pt` |
| E4 验证指标 | mAP50=**0.9650**，mAP50-95=**0.6430**，P=0.9491，R=0.9281 |
| 实时追踪 | 验收达标：25.02 FPS / P95 35.76 ms / 丢帧 0%（E4, stride=2, FP16） |
| 部署 | ONNX（opset 17）；PT CUDA 37.04 FPS / ORT CUDA 24.30 FPS / ORT CPU 3.71 FPS |
| 测试 | 263 个自动化测试（30 个文件）+ GitHub Actions 一轮真实训练冒烟 |
| 服务化 | FastAPI + E4 ONNX；同端口浏览器摄像头监控页、健康检查、上传门禁、Prometheus/JSON 监控 |
| 模型治理 | 版本化 registry 将实验—权重—报告关联并启动校验 SHA256 |
| 完整实验 | 10 轮（M4/E1-E8），含 2 轮已否定的数据级优化 |

## 3. 核心结论（已由前序分析得出，直接沿用，无需重做）

### 3.1 成熟度结论

- **判定**：科研/实验型维度**已相当成熟完善**（实验方法论是亮点）；生产化维度**有明确缺口**。
- **评分**：实验完备度 **9/10**，工程完备度 **7/10**，生产就绪度 **5/10**。
- **一句话**：结论可信、过程可复现、数据可追溯的成熟实验项目，缺最后一步「持续交付、可监控、可服务化」。

### 3.2 模型实验结论（哪些路通、哪些路不通）

| 实验 | 手段 | 结论 | 采用？ |
|---|---|---|---|
| M4→E1 | 轮数 50→100 | +0.67 pp，50 轮后平台期 | ❌ 不值得 |
| M4→E2 | 分辨率 640→960 | **+2.88 pp**，微小目标 R 0.484→0.578 | ✅ 主要增益来源 |
| M4→E3 | 容量 n→s | +2.42 pp，但 640 下受限于分辨率 | ✅ 主要增益来源 |
| M4→E4 | 容量+分辨率叠加 | **+3.92 pp**，当前最高 | ✅ **生产候选** |
| E5a | 微小困难样本定向重采样 | 微小 R 升但总体退化，不达判据 | ❌ 无效（留档） |
| E5b-B1 | 微小目标上下文裁剪放大 | 有限正结果，整体不优于 E4 | ❌ 保留研究记录 |
| E6 | P2 高分辨率检测头 | 微小目标 R **+4.69 pp**，但 GFLOPs+58%、速度近腰斩、混淆上升 | ⚠️ 实时性代价过大 |
| **E7** | P2 + 轻量特征精炼（LFR） | **mAP50-95 最高 0.6479**、类别混淆最少（10）、参数仅 +0.19% | ✅ 最优结构解，**在途** |
| E8 | P2+LFR 去 ECA 消融 | 精度回落、混淆回升 | ❌ ECA 不可缺（证明 E7 收益来自模块整体） |

- **阈值结论**：conf 从 0.15 起误检开始明显增加（FP 1.68×）；NMS IoU=0.70 为平衡点（≥0.80 出现明显重复框）。
- **生产决策**：E4 仍为默认生产候选（实时性+精度综合最优、权重路径稳定）；E7 作为结构优化方向在途。

### 3.3 ⚠️ 评估口径差异（极易踩坑，务必牢记）

同一份 val 数据存在**两套 E4 锚点口径**，报告中数字不能直接混比：

| 系列 | E4 锚点配置 | 引用出处 |
|---|---|---|
| E5a / E5b | **候选 C：conf=0.20 / NMS IoU=0.50**（TP=9420/FN=505/FP=951） | `e5a_*`、`e5b_*` 报告 |
| E6 / E7 / E8 | **标准配置：conf=0.25 / NMS IoU=0.70**（TP=9363/FN=562/FP=953） | `e6_*`、`e7_*`、`e8_*` 报告 |

推理管线默认参数是 conf=0.25 / iou=0.70（M5/M6 一致）。

## 4. 工作守则（下一个 agent 必须遵守）

### 4.1 实验纪律（项目的灵魂）

1. **只改一个变量**：每轮实验只变化一个维度，其余参数全部冻结。
2. **预声明判据**：跑实验前先写清"什么算成功"，用证据而非感觉决策。
3. **test 永不参与**：test 集不得用于训练、选模、调参或阈值选择。
4. **负结果留档**：失败的实验和成功的一样有价值，报告必须保留。

### 4.2 数据纪律

- 原始 SHWD 数据**只读**，绝不修改原始 XML/图片/划分文件。
- 用 `--force` 显式覆盖产物；依赖 `.shwd-generated-files.json` manifest 精确清理，绝不误删用户文件。
- 转换契约：`hat→0`、`person→1`、`dog` 忽略；未知类别在写出前失败；坐标回算误差 ≤0.01px。

### 4.3 训练与复现纪律

- **锁定 `ultralytics==8.4.120`**；升级版本前必须先重跑全部测试 + 冒烟训练。
- 训练固定 `deterministic=True`、`seed=42`；E4→P2 状态迁移有张量级迁移报告，勿手工改权重。
- 运行名已存在时自动选 `_002/_003`，**不覆盖历史实验**；E4 路径保持固定。

### 4.4 代码与质量纪律

- **先读后写**：修改前理解整体数据流与文件间关联（CLAUDE.md 要求）。
- 改完**跑全部测试**：`.\.venv\Scripts\python.exe -m pytest`（247 个）。
- 每次解决任务后，将**问题描述 + 解决方案**写入 `任务清单.md`（已完成区标注日期与原因）。

## 5. 常用操作入口（脚本索引）

```powershell
# 全部测试
.\.venv\Scripts\python.exe -m pytest

# 数据转换与独立校验
.\.venv\Scripts\python.exe scripts\data\convert_shwd.py --raw D:\datasets\SHWD\VOC2028 --output D:\datasets\SHWD\processed
.\.venv\Scripts\python.exe scripts\data\validate_yolo_dataset.py --raw D:\datasets\SHWD\VOC2028 --processed D:\datasets\SHWD\processed --output D:\datasets\SHWD\audit --visualize

# 训练（scripts/train/）
#   train_baseline.py（M4/E1-E3 风格）、train_e6_p2.py、train_e7_lfr.py、train_e8_no_eca.py、resume_*.py、calibrate_e6_thresholds.py
.\.venv\Scripts\python.exe scripts\train\train_e7_lfr.py --data D:\datasets\SHWD\processed\dataset.yaml --output artifacts\e7\<新目录> --device 0 --batch 2

# 评估（scripts/evaluate/）
#   evaluate_e1.py ~ evaluate_e8_no_eca.py；统一口径：完整 val 607 张 / imgsz=960 / batch=2 / seed=42

# 推理（scripts/inference/）
.\.venv\Scripts\python.exe scripts\inference\infer_image.py --source D:\path\to\image --output artifacts\inference\images --weights artifacts\training\m45_yolo11s_e75_960_001\weights\best.pt --imgsz 960 --conf 0.25 --iou 0.70
.\.venv\Scripts\python.exe scripts\inference\track_video_realtime.py --source D:\path\to\video.mp4 --output artifacts\tracking\realtime.mp4 --weights artifacts\training\m45_yolo11s_e75_960_001\weights\best.pt --device 0 --imgsz 960 --conf 0.12 --fp16 --tracker bytetrack --frame-stride 2 --jsonl

# 部署（scripts/deploy/）
.\.venv\Scripts\python.exe scripts\deploy\export_onnx.py --weights artifacts\training\m45_yolo11s_e75_960_001\weights\best.pt --output artifacts\deployment\e4_yolo11s_960.onnx --imgsz 960 --opset 17
.\.venv\Scripts\python.exe scripts\deploy\benchmark_backends.py --pt-weights artifacts\training\m45_yolo11s_e75_960_001\weights\best.pt --onnx-weights artifacts\deployment\e4_yolo11s_960.onnx --source D:\datasets\SHWD\processed\images\val\000000.jpg
```

> 提示：`scripts/` 按功能分子目录（`data/train/evaluate/analyze/inference/deploy`）；新增实验脚本请放进对应子目录并遵循 `动作_实验_内容` 命名。

## 6. 生产化缺口完成状态（2026-09-01）

1. **CI/CD 门禁 — 已补齐**：`.github/workflows/ci.yml` 在 main push/PR 执行依赖策略、全量测试和无需私有数据/网络下载的一轮真实 YOLO CPU 训练，并上传证据。
2. **推理服务化 — 已补齐**：`helmet_safety.service.api` 用 FastAPI 封装复用的 `OpenCVDetector`，默认服务 E4 ONNX，提供 liveness/readiness、上传门禁及稳定响应模型；根路径内置浏览器摄像头监控页，通过 Canvas 展示实时模型框和运行指标。
3. **权重与产物版本管理 — 本地治理已补齐**：`configs/model_registry.json` 将实验、阶段、后端、权重、报告与 SHA256 组成可校验记录；对象存储远端仍需部署方提供凭证后配置。
4. **生产监控 — 已补齐**：Prometheus/JSON 暴露请求量、错误量、P50/P95 延迟、类别检测量、滚动置信度漂移与 GPU 显存。
5. **元数据修正 — 已补齐**：`pyproject.toml` 已更新系统定位，并新增 service/pipeline 可选依赖与 FastAPI 入口。
6. **实验编排跨机复现 — 已补齐**：`dvc.yaml` + `params.yaml` 声明数据审计、转换、校验和训练冒烟，机器路径改为参数，不再写死在编排脚本中。

仍需基础设施侧完成的工作：为 DVC/对象存储配置组织级远端与凭证；CI 当前只做验证和训练冒烟，不自动发布镜像或推送生产环境。

## 7. 关键文件索引

| 类型 | 路径 | 用途 |
|---|---|---|
| 任务清单 | `任务清单.md` | **每次动手前必读**；问题记录与解决方案 |
| 项目总览 | `README.md` | 算法思想 + 三阶段模型对比表 + 架构图 |
| 产物索引 | `artifacts/ARTIFACTS_INDEX.md` | 全部实验权重/报告/SHA256 索引 |
| 实验报告 | `artifacts/evaluation/`、`artifacts/e5a/`、`artifacts/e5b/`、`artifacts/e6/`、`artifacts/e7/`、`artifacts/e8/` | 各实验对比报告（含口径） |
| 核心库 | `src/helmet_safety/{data,training,inference,tracking}/` | 数据/训练/推理/追踪模块 |
| HTTP 服务 | `src/helmet_safety/service/`、`scripts/serve_api.py` | ONNX 服务、模型注册、监控 |
| 生产门禁 | `.github/workflows/ci.yml`、`scripts/ci/` | 全测、依赖门禁、一轮训练冒烟 |
| 声明式流水线 | `dvc.yaml`、`params.yaml` | 跨机数据治理与冒烟编排 |
| 结构配置 | `configs/yolo11s-p2*.yaml`、`configs/m6_*.yaml` | P2/LFR 结构定义、tracker 参数 |
| 领域笔记 | `_knowledge_base/` | 微小目标优化、生产召回研究笔记 |
| 测试 | `tests/`（26 文件） | 247 个自动化测试 |

---

*本文件由前序成熟度分析与 README 重写工作产出。若项目状态发生重大变化（新实验、新生产候选、补齐 CI 等），请同步更新本文件。*
