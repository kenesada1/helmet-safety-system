# Helmet Safety System — 头盔佩戴安全检测系统

基于 **PyTorch / OpenCV / Ultralytics YOLO** 的工业级头盔佩戴检测系统，覆盖 **数据治理 → 模型训练 → 诊断评估 → 实时推理 → 多目标追踪 → ONNX 部署** 的完整链路。以 SHWD（Safety Helmet Wearing Dataset）为数据源，通过**十轮受控实验**完成模型选型与结构优化，当前生产候选为 E4（YOLO11s，imgsz=960），并已实现满足实时性验收（≥25 FPS）的多目标追踪流水线。

> 核心设计信条：**任何结论都必须有数据支撑，任何实验都必须可复现、可归因、可证伪。** 这是一套"像做科研一样做工程"的检测系统。

---

## 一、核心成果速览

| 维度 | 结果 |
|---|---|
| 数据集 | SHWD，两类（`helmet` / `no_helmet`），5,457 训练 / 607 验证 / 1,517 测试 |
| 主模型 | E4：YOLO11s，imgsz=960，训练 75 轮 |
| 主模型验证指标 | **mAP50 = 0.9650，mAP50-95 = 0.6430**，Precision = 0.9491，Recall = 0.9281 |
| 实验完备性 | 10 轮受控实验（M4/E1-E8），统一评估口径，判据预先声明 |
| 实时追踪 | 25.02 FPS 吞吐、P95 完整帧 35.76 ms、丢帧率 0%（E4 权重，stride=2，FP16） |
| 部署 | ONNX 导出（opset 17），PyTorch CUDA 37.04 FPS / ORT CUDA 24.30 / ORT CPU 3.71 FPS |
| 服务化 | FastAPI + E4 ONNX，同端口浏览器摄像头监控页、健康检查、上传门禁、Prometheus/JSON 监控 |
| 模型治理 | 版本化 registry 关联「实验—权重—报告」，启动 fail-closed 校验 SHA256 |
| 生产门禁 | GitHub Actions（依赖策略 + 全量测试 + 一轮真实训练冒烟）；DVC 跨机声明式流水线 |
| 质量保障 | 263 个自动化测试（30 文件），覆盖数据、训练、评估、推理、追踪、部署与服务 |

---

## 二、算法思想（设计哲学）

这个项目的价值不在于"调出了一个高精度模型"，而在于**用一套可复现的实验方法论，一步步把精度与实时性同时推上去，并明确知道每一步的代价与收益**。核心思想有四条。

### 2.1 数据可信是地基：先管好数据，再谈模型

模型精度的上限由数据质量决定。因此在训练之前，项目先建立了一条**只读、可审计、可独立验证**的数据管线：

- **只读审计**：`audit.py` 以只读方式扫描原始 SHWD VOC 数据，绝不修改原始 XML / 图片 / 划分文件。
- **严格数据规则**：`hat → helmet`、`person → no_helmet`、`dog` 仅忽略；出现任何未知类别在写出数据前**直接失败**，宁可失败也不产出脏数据。
- **独立验证**：`validate.py` 不信任转换报告，而是**重新读取 XML 与 YOLO 标签逐框比较**，坐标回算误差必须 ≤ 0.01 px。
- **可无损修复**：JPEG 若在有效 EOI 后带有多余字节，只在确认解码像素完全一致后才裁剪尾部字节。
- **manifest 治理**：`.shwd-generated-files.json` 记录工具生成过的文件，配合 `--force` 精确清理，绝不误删用户文件。

> 思想：**"先在数据上建立确定性，模型才有资格谈概率。"**

### 2.2 诊断驱动的受控实验：一个变量，一个判据

模型优化不是"乱调参"，而是一系列**单变量控制实验**：

- **只改一个变量**：每轮实验只变化一个维度（轮数 / 输入分辨率 / 模型容量 / 数据分布 / 检测头结构），其余参数全部冻结。
- **预声明判据**：每轮实验在跑之前就写好"什么算成功"（例如"微小目标召回率提升且总体 F1 下降不超过 0.002"），**用证据而不是感觉做决策**。
- **统一评估口径**：所有实验用同一套完整 val（607 张 / 9,925 个 GT），`test` 永不参与训练、选模、调参或阈值选择。
- **负结果同样留档**：E5-A 重采样被判无效、E5-B1 裁剪被保留为研究记录——**失败的实验和成功的实验一样有价值**，它们证明"此路不通"并防止后人重复踩坑。

> 思想：**"没有判据的实验等于掷骰子；记录负结果才是对时间的尊重。"**

### 2.3 小目标攻坚：分辨率、容量、数据、结构四层杠杆

SHWD 的核心难点是大量 ≤20 像素的微小目标（尤其 `no_helmet` 在密集人群中）。项目从四个层面系统化攻坚：

| 层级 | 手段 | 对应实验 |
|---|---|---|
| 输入分辨率 | 640 → 960，放大微小目标的有效像素 | E2, E4 |
| 模型容量 | YOLO11n → YOLO11s，提高特征表达能力 | E3, E4 |
| 数据分布 | 定向重采样困难样本 / 上下文裁剪放大微小目标 | E5a, E5b |
| 检测头结构 | 增加 P2 高分辨率检测头 + 轻量特征精炼 | E6, E7, E8 |

每一层都有量化结论：E2 提升输入分辨率把微小目标召回率从 0.484 拉到 0.578；E4 容量+分辨率双升把整体 mAP50-95 推到 0.643；P2 检测头把微小目标召回率推到 0.648 但算力暴涨 58%；E7 的轻量精炼用仅 +0.19% 参数换回部分总体精度。

> 思想：**"把精度瓶颈拆成可归因的维度，逐个击破，而不是一把梭。"**

### 2.4 精度-算力-实时性的三角权衡

本项目不是只在 Offline 上比精度，而是有明确的实时性验收线（≥25 FPS）。因此每一步结构优化都要同时回答三个问题：**精度涨了吗？算力涨了多少？还能实时吗？**

- P2 检测头把 GFLOPs 从 48.5 拉到 76.7（+58%），推理 FPS 从 79.9 跌到 46.8——**精度收益小于实时性代价**，因此 E4 仍是生产候选。
- E7 的轻量特征精炼只增加 +0.19% 参数，却改善了整体检测并减半类别混淆，是"花小钱办大事"的结构解。

> 思想：**"精度不是唯一目标，要在约束下求最优。"**

---

## 三、模型优化：十轮受控实验全记录

### 3.1 阶段一：主干选型与控制实验（M4 → E4）

从 M4 基线（YOLO11n, 640, 50 轮）出发，逐变量验证"延长训练 / 提高分辨率 / 加大容量"的贡献。评估口径：**标准 Ultralytics val**（完整 607 张验证集）。

| 实验 | 架构 | 输入 | 轮数 | Precision | Recall | mAP50 | mAP50-95 | 相对基线 ΔmAP50-95 | 结论 |
|---|---|---|---|---|---|---|---|---|---|
| **M4** | YOLO11n | 640 | 50 | 0.9289 | 0.8894 | 0.9383 | 0.6038 | — | 基线 |
| **E1** | YOLO11n | 640 | 100 | 0.9336 | 0.8809 | 0.9422 | 0.6105 | +0.67 pp | 延长轮数仅小幅提升，50 轮后平台期 |
| **E2** | YOLO11n | 960 | 50 | 0.9445 | 0.9134 | 0.9590 | 0.6326 | +2.88 pp | 提高分辨率收益显著，微小目标 R 0.484→0.578 |
| **E3** | YOLO11s | 640 | 50 | 0.9450 | 0.8950 | 0.9560 | 0.6281 | +2.42 pp | 加大容量有效，但 640 下仍受限于分辨率 |
| **E4** | YOLO11s | 960 | 75 | 0.9491 | 0.9281 | 0.9650 | 0.6430 | +3.92 pp | ✅ **当前生产候选**，容量+分辨率双升 |

> **阶段一结论**：输入分辨率（E2）与模型容量（E3）是主要增益来源，二者叠加（E4）收益最高；延长训练轮数（E1）收益有限，不值得。参数量从 2.59M（n）增至 9.41M（s），验证了"微小目标需要更大有效像素与更强特征"。

### 3.2 阶段二：数据级优化（E5a 定向重采样 / E5b 上下文裁剪）

针对 E4 的微小目标漏检，从数据分布层面尝试两条路线。评估口径：**固定阈值计数**（class-aware 一对一匹配），E5 系列以 E4 候选 C（conf=0.20 / NMS IoU=0.50）为锚点。

| 实验 | 方案 | 微小目标 R | 微小目标 F1 | 总体 TP/FN/FP | 总体 F1 | 判决 |
|---|---|---|---|---|---|---|
| E4（锚点） | 候选 C 配置 | 0.6172 | 0.6077 | 9420 / 505 / 951 | 0.9283 | 参考基准 |
| 延长训练对照 | 同轮数公平对照 | 0.6484 | 0.6014 | 9446 / 479 / 1067 | 0.9244 | 公平对照组 |
| **E5a** | 微小困难样本定向重采样 | 0.6719 | 0.5931 | 9409 / 516 / 1088 | 0.9215 | ❌ **无效**：微小 R 升但总体退化，不达预声明判据 |
| **E5b-B1** | 微小目标上下文裁剪放大 | 0.6563 | 0.6109 | 9431 / 494 / 1058 | 0.9240 | ⚠️ 有限正结果，整体不优于 E4，保留研究记录 |

> **阶段二结论**：数据级手段对微小目标有局部提升，但都伴随总体误检或综合指标损失，未通过成功门槛。这反向验证了**问题不在数据分布，而在网络结构对微小目标的感知能力**，为阶段三指明方向。

### 3.3 阶段三：结构级优化（E6 P2 检测头 / E7 P2+LFR / E8 去 ECA 消融）

通过自定义 P2 高分辨率检测头与轻量特征精炼模块（Lite Feature Refinement, LFR），直接增强网络对微小目标的感知。评估口径：**标准 val + 固定阈值计数**（conf=0.25 / NMS IoU=0.70），E6-E8 系列以 E4 标准配置为锚点。

| 实验 | 结构 | 参数量 | GFLOPs@960 | Precision | Recall | mAP50 | mAP50-95 | 固定阈值 F1 | 类别混淆 | 推理 ms/图 | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **E4** | YOLO11s | 9.41M | 48.5 | 0.9491 | 0.9281 | 0.9650 | 0.6430 | 0.9252 | 14 | 12.5 | 79.9 |
| **E6** | + P2 检测头 | 9.66M | 76.7 | 0.9434 | 0.9236 | 0.9668 | 0.6445 | 0.9242 | 20 | 21.4 | 46.8 |
| **E7** | P2 + LFR | 9.68M | 78.8 | 0.9532 | 0.9231 | 0.9672 | **0.6479** | **0.9283** | **10** | 23.5 | 42.5 |
| **E8** | P2 + LFR（去 ECA） | 9.68M | 78.8 | 0.9544 | 0.9228 | 0.9683 | 0.6440 | 0.9257 | 15 | 22.9 | 43.7 |

**微小目标（≤10px）召回率分档对比：**

| 实验 | tiny 总体 R | helmet tiny R | no_helmet tiny R |
|---|---|---|---|
| E4 | 0.6016 | 0.2857 | 0.6404 |
| E6 | **0.6484** | 0.3571 | **0.6842** |
| E7 | 0.6172 | 0.2857 | 0.6579 |
| E8 | 0.6172 | **0.4286** | 0.6404 |

> **阶段三结论**：
> - **E6（P2 头）**：微小目标召回率提升最明显（+4.69 pp），但 GFLOPs +58%、推理速度近腰斩，且类别混淆反而上升，实时性代价过大。
> - **E7（P2+LFR）**：轻量特征精炼（深度可分离卷积 + ECA 通道注意力 + 残差）用 +0.19% 参数换回总体 mAP50-95 最高分（0.6479）与最少的类别混淆（10），是结构与精度的最优平衡。判定为 P2 精炼**有效**。
> - **E8（去 ECA 消融）**：去掉 ECA 后精度回落、混淆回升，证明 ECA 在 P2 浅层增强中**不可或缺**。同时证明 E7 的收益来自精炼模块整体而非偶然。
> - **生产决策**：E4 仍是默认生产候选（实时性与精度综合最优，固定权重路径稳定）；E7 作为结构优化方向在途。

### 3.4 阈值校准：conf 与 NMS IoU 单变量扫描

选好模型后，用冻结的微小目标 val 子集做后处理参数的单变量扫描，回答"降置信度能救回多少漏检、代价是多少误检"：

- **conf 扫描**：conf 0.25 → 0.15 时微小目标 TP 77→83（+6），FP 250→420（+1.68×）；**从 conf=0.15 起误检开始"明显增加"**，这是 M6 追踪使用分层置信度的依据。
- **NMS IoU 扫描**：IoU ≤ 0.70 时无重复框、FP 稳定；**≥ 0.80 后出现明显重复框对**（0.80 时 98 对，涉及 26 张图）。0.70 是平衡点。
- **P0 四配置验证**：conf × IoU 四组合中，固定阈值总体 F1 最高为 conf=0.25 / IoU=0.50（0.9313）；E5 系列采用 conf=0.20 / IoU=0.50 作为锚点以兼顾微小目标。

> **思想**：阈值不是拍脑袋定的，是扫出来的；降阈值救回的微小目标，必须以可量化的误检增量买单。

---

## 四、系统架构与数据流

```text
原始 SHWD（只读）
  │
  ├─ audit.py ──────────────► audit_report（数据体检）
  └─ convert.py（预检查）──► processed/（images + labels + dataset.yaml）
                                  │
                                  └─ validate.py ─► validation_report（独立反算验证）
                                          │
                                          ▼
  ┌─────────────────────────────── 训练与评估 ───────────────────────────────┐
  │  train_*.py（基线 / E1-E8 / 续训）      evaluate_*.py（标准 val + 固定阈值）│
  │  单变量控制实验 + 预声明判据             错误样本分析 + 尺寸/密集度分层       │
  └───────────────────────────────────────────────────────────────────────┘
                                          │  best.pt
                                          ▼
  ┌─────────────────────────────── 推理与追踪 ───────────────────────────────┐
  │  infer_image / infer_video ──► OpenCV 检测                              │
  │  track_video ──► ByteTrack / BoT-SORT 多目标追踪                         │
  │  track_video_realtime ──► 有界异步队列 + 隔帧检测 + 外推预测（≥25 FPS）   │
  └───────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
  ┌─────────────────────────────── 部署 ────────────────────────────────────┐
  │  export_onnx（opset 17）──────► .onnx ──► ORT / 性能基准                  │
  └───────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
  ┌──────────────────────── 服务与生产门禁 ────────────────────────────────┐
  │  FastAPI：/v1/detections、/health/*、/metrics、同端口浏览器监控页        │
  │  model_registry.json：fail-closed SHA256 验真 + 并发互斥 + 上传/像素门禁  │
  │  .github/workflows/ci.yml：依赖策略 + 全量测试 + 一轮真实 CPU 训练冒烟    │
  │  dvc.yaml + params.yaml：数据审计/转换/校验/冒烟的跨机声明式复现          │
  └───────────────────────────────────────────────────────────────────────┘
```

**源代码模块**（[src/helmet_safety/](src/helmet_safety/)）：

| 模块 | 职责 |
|---|---|
| [data/](src/helmet_safety/data/) | SHWD 审计、VOC→YOLO 转换、独立验证 |
| [training/](src/helmet_safety/training/) | 各实验训练、评估、阈值校准、错误分析、E4→P2 状态迁移 |
| [inference/](src/helmet_safety/inference/) | OpenCV 图片/视频检测、tiling、FP16 推理 |
| [tracking/](src/helmet_safety/tracking/) | ByteTrack/BoT-SORT 封装、绘制、**实时追踪流水线** |
| [service/](src/helmet_safety/service/) | FastAPI 推理服务、版本化模型注册、线程安全监控、浏览器摄像头监控页 |
| [quality/](src/helmet_safety/quality/) | 依赖策略门禁（`ultralytics==8.4.120` 精确锁定校验） |

---

## 五、里程碑能力速览（M3 - M6）

### M3：训练冒烟测试
确定性 smoke test，验证训练链路，固定 `ultralytics==8.4.120`（升级前必须重跑测试与冒烟训练）。

### M4：全量基线训练与独立测试
5,457 张训练 + 607 张 val 选模 + 1,517 张 test 独立评估。**test 不参与训练、选模、调参或阈值选择**，这是评估可信度的底线。评估产出混淆矩阵、P/R 曲线、100 张固定 seed 预测图与人工检查清单。

### M5：OpenCV 图片/视频推理
默认模型为 E4（`artifacts/training/m45_yolo11s_e75_960_001/weights/best.pt`）。图片与视频共用推理入口，视频保持原宽高/FPS/时长，报告记录吞吐与完整帧耗时。统计的是**检测框数量**，不是去重后的人员数——事件级语义留给追踪。

### M6：多目标追踪
在 M5 检测之上叠加轨迹状态，默认 ByteTrack、可选 BoT-SORT：
- **分层置信度**：`0.10–0.25` 低置信框只参与已有轨迹的第二阶段关联，`≥0.25` 才能创建新轨迹，从源头抑制短命轨迹。
- **自适应 track_buffer**：按 `lost_ttl × source_fps / frame_stride` 计算，不写死。
- **逐帧 JSONL**：记录是否执行检测、轨迹观测、预测框标记，全链路可审计。

### M6-Realtime：实时追踪（验收达标 ✅）

满足实时性验收的追踪流水线：**有界异步采集队列 + 隔帧检测（stride=2）+ 中间帧匀速外推预测 + confirmed track 门控（hits≥3）+ 类别时序投票（窗口5/多数3）+ FP16 推理**。

2026-09-01 本机实测（RTX 3060 Laptop，E4 best.pt，imgsz=960，stride=2）：

| 验收项 | 目标 | 实测 | 状态 |
|---|---|---|---|
| 完整链路平均吞吐 | ≥ 25 FPS | 25.02 FPS | ✅ |
| P95 完整帧耗时 | ≤ 40 ms | 35.76 ms | ✅ |
| 队列端到端延迟 P95 | ≤ 200 ms | 36.45 ms | ✅ |
| 实际丢帧率 | < 1% | 0.0% | ✅ |
| hits≤2 短轨迹比例 | 显著下降 | 67.7% → 13.6% | ✅ |

> 不节流全速处理能力约 51 FPS（876 帧真实视频）。已修复检测帧/中间帧绘制风格不一致导致的标签闪烁问题，228/228 帧轨迹集完全一致。

### M7：HTTP 推理服务、生产监控与门禁（生产化补齐 ✅）

在 CLI 之外补齐生产化闭环：

- **FastAPI 推理服务**：默认从 `configs/model_registry.json` 选择 E4 ONNX，模型单次加载 + 并发互斥，提供 `/v1/detections`、`/health/live|ready`、`/metrics`（Prometheus）与 `/v1/metrics`（JSON）；**同端口**提供浏览器摄像头监控页（`getUserMedia` + Canvas 实时绘制检测框与运行指标）。
- **线程安全监控**：请求成功/失败、P50/P95 延迟、分类检测量、滚动置信度漂移告警、GPU 显存。
- **版本化模型 registry**：把「实验—阶段—权重—报告—类别—输入尺寸—SHA256」组成不可变记录，服务启动与独立 CLI 均 fail-closed 验真。
- **CI / DVC 门禁**：GitHub Actions 跑依赖策略 + 全量测试 + 一轮真实 CPU 训练冒烟；`dvc.yaml` 声明数据治理与冒烟的跨机复现。

详细接口、环境变量与 Docker 用法见「六、部署与性能基准」的 6.1 / 6.2。

---

## 六、部署与性能基准

E4 `best.pt` 已导出为静态 shape、batch=1、FP32、opset 17 的 ONNX（`artifacts/deployment/e4_yolo11s_960.onnx`），导出后自动运行 `onnx.checker`、ORT 加载与真实图片 smoke inference。

2026-08-31 本机实测（batch=1, imgsz=960，计时含预处理+推理+NMS）：

| 后端 | 设备 | 平均延迟 | P50 | P95 | FPS |
|---|---|---|---|---|---|
| PyTorch CUDA | RTX 3060 Laptop GPU | 26.99 ms | 25.46 ms | 36.18 ms | 37.04 |
| ONNX Runtime CUDA | CUDAExecutionProvider | 41.15 ms | 41.83 ms | 62.55 ms | 24.30 |
| ONNX Runtime CPU | Intel Core i7-10870H | 269.69 ms | 258.97 ms | 343.18 ms | 3.71 |

> 测试环境：Windows 10 / i7-10870H / RTX 3060 Laptop / Python 3.14.3 / PyTorch 2.12.1+cu130 / Ultralytics 8.4.120 / ONNX 1.22.0 / ORT 1.29.0。这里的数字是**部署延迟**，不是精度指标。报告确认 ONNX CUDA 实际使用 CUDAExecutionProvider，无 CPU 冒充。

### 6.1 HTTP 推理服务与生产监控

FastAPI 服务默认从 `configs/model_registry.json` 选择 E4 ONNX 模型，启动时校验「实验—权重—报告」关联和 SHA256，校验失败时保持存活但 readiness 返回 503，不会用未知权重带病接流量。

```powershell
# 安装并启动（默认 http://127.0.0.1:8000）
.\.venv\Scripts\python.exe -m pip install -e ".[service]"
.\.venv\Scripts\python.exe scripts\serve_api.py --host 127.0.0.1 --port 8000

# 推理、健康检查和指标
curl.exe -F "image=@D:\path\to\image.jpg" http://127.0.0.1:8000/v1/detections
curl.exe http://127.0.0.1:8000/health/ready
curl.exe http://127.0.0.1:8000/metrics
```

启动成功后访问 `http://127.0.0.1:8000/` 即可打开内置监控页，无需额外前端端口或 Node.js 服务。页面调用浏览器 `getUserMedia` 选择本机摄像头，以可配置间隔截取 JPEG 帧发送到 `/v1/detections`，并在 Canvas 上实时叠加绿色 `helmet` 框、红色 `no_helmet` 框、置信度、延迟、FPS、当前计数和最近事件。首次点击“启动监控”需要允许浏览器摄像头权限；摄像头 API 仅在 `localhost` 或 HTTPS 安全上下文中可用。

> “显示阈值”只控制前端框的显示过滤，不修改后端模型的固定生产推理阈值；“推理间隔”采用非重叠请求，上一帧完成后才会调度下一帧，避免慢设备积压请求。

服务限制上传字节数与解码后像素数；模型只加载一次，并用互斥锁保护同一模型实例。`/metrics` 输出 Prometheus 文本，持续暴露请求成功/失败、P50/P95 推理延迟、分类检测量、滚动置信度漂移告警和 PyTorch GPU 显存；`/v1/metrics` 提供同一快照的 JSON 形式。可通过以下环境变量配置：

| 变量 | 默认值 | 用途 |
|---|---|---|
| `HELMET_MODEL_REGISTRY` | `configs/model_registry.json` | 模型注册清单 |
| `HELMET_MODEL_ID` | `e4-yolo11s-960-onnx` | 部署模型版本 |
| `HELMET_DEVICE` | `cpu` | 推理设备 |
| `HELMET_MAX_UPLOAD_BYTES` | `10485760` | 单次上传上限 |
| `HELMET_MAX_IMAGE_PIXELS` | `20000000` | 解码后像素上限 |
| `HELMET_CONFIDENCE_BASELINE` | `0.80` | 置信度漂移基线 |
| `HELMET_CONFIDENCE_DRIFT_THRESHOLD` | `0.15` | 漂移告警阈值 |

Docker 镜像不内置模型产物，部署时将版本化的 `artifacts/` 挂载到 `/app/artifacts`；这样镜像与模型版本可以独立发布。

```powershell
docker build -t helmet-safety-api .
docker run -d -p 8000:8000 `
  -v D:\datasets\helmet-safety\artifacts:/app/artifacts `
  helmet-safety-api
```

> 镜像以 `python:3.11-slim` 为基础，安装 `libgl1`/`libglib2.0-0`（OpenCV 依赖），内置健康检查（`/health/ready`）；`HELMET_DEVICE=cpu` 为默认，GPU 部署需换 `onnxruntime-gpu` 并调整 `--gpus`。

### 6.2 CI、模型版本与跨机编排

- `.github/workflows/ci.yml` 在 main 分支 push/PR 时执行依赖策略检查、全量测试和一轮真实 YOLO CPU 训练；训练使用代码生成的小型双类别数据集，不依赖私有 SHWD 路径或下载预训练权重。
- `requirements/ci.lock` 固定 CI 直接依赖，`scripts/ci/check_dependencies.py` 强制 `ultralytics==8.4.120` 在训练、推理和部署组保持精确锁定。
- `configs/model_registry.json` 记录模型 ID、实验、运行阶段、后端、权重、报告、类别、输入尺寸和 SHA256；运行 `scripts/deploy/verify_model_registry.py` 可独立验真。
- `dvc.yaml` + `params.yaml` 声明数据审计、转换、独立校验与 CI 冒烟流水线。跨机仅需调整 `params.yaml` 的只读原始数据路径，然后用 `dvc repro <stage>` 运行；产物不覆盖历史实验。

---

## 七、常用命令

```powershell
# 全部测试
.\.venv\Scripts\python.exe -m pytest

# 数据转换与校验
.\.venv\Scripts\python.exe scripts\data\convert_shwd.py --raw D:\datasets\SHWD\VOC2028 --output D:\datasets\SHWD\processed
.\.venv\Scripts\python.exe scripts\data\validate_yolo_dataset.py --raw D:\datasets\SHWD\VOC2028 --processed D:\datasets\SHWD\processed --output D:\datasets\SHWD\audit --visualize

# 训练（M4 基线示例）
.\.venv\Scripts\python.exe scripts\train\train_baseline.py --data D:\datasets\SHWD\processed\dataset.yaml --model artifacts\models\yolo11n.pt --epochs 50 --imgsz 640 --batch 8 --workers 0 --device 0

# 图片 / 视频推理（默认 E4 权重）
.\.venv\Scripts\python.exe scripts\inference\infer_image.py --source D:\path\to\image --output artifacts\inference\images --weights artifacts\training\m45_yolo11s_e75_960_001\weights\best.pt --imgsz 960 --conf 0.25 --iou 0.70
.\.venv\Scripts\python.exe scripts\inference\infer_video.py --source D:\path\to\input.mp4 --output artifacts\inference\videos\output.mp4 --weights artifacts\training\m45_yolo11s_e75_960_001\weights\best.pt --imgsz 960 --frame-stride 1

# 实时追踪（M6-Realtime）
.\.venv\Scripts\python.exe scripts\inference\track_video_realtime.py --source D:\path\to\video.mp4 --output artifacts\tracking\realtime.mp4 --weights artifacts\training\m45_yolo11s_e75_960_001\weights\best.pt --device 0 --imgsz 960 --conf 0.12 --fp16 --tracker bytetrack --frame-stride 2 --jsonl

# ONNX 导出与性能基准
.\.venv\Scripts\python.exe scripts\deploy\export_onnx.py --weights artifacts\training\m45_yolo11s_e75_960_001\weights\best.pt --output artifacts\deployment\e4_yolo11s_960.onnx --imgsz 960 --opset 17
.\.venv\Scripts\python.exe scripts\deploy\benchmark_backends.py --pt-weights artifacts\training\m45_yolo11s_e75_960_001\weights\best.pt --onnx-weights artifacts\deployment\e4_yolo11s_960.onnx --source D:\datasets\SHWD\processed\images\val\000000.jpg

# 生产门禁、模型验真与声明式流水线
.\.venv\Scripts\python.exe scripts\ci\check_dependencies.py
.\.venv\Scripts\python.exe scripts\ci\run_training_smoke.py --output artifacts\ci-smoke --force
.\.venv\Scripts\python.exe scripts\deploy\verify_model_registry.py
dvc repro ci_smoke
```

各脚本默认拒绝覆盖已有产物，需重跑时显式加 `--force`；训练运行名已存在时自动选择 `_002/_003`，不覆盖历史。

---

## 八、数据规则（转换契约）

- `hat → 0 helmet`，`person → 1 no_helmet`；`dog` 仅忽略并写入报告。
- 任何其他类别 → 写出数据前失败。
- 无有效对象的图片保留并生成空 `.txt` 标签。
- 标签到像素坐标回算误差 ≤ 0.01 px。
- JPEG 必须严格解码；仅允许无损裁剪 EOI 后的尾部字节，并记录 SHA-256 与删除字节数。
- 找不到有效 EOI、解码失败或像素不一致 → 写出前失败。

---

## 九、环境与依赖

- Python ≥ 3.10（本机实测 3.14.3）
- 训练/推理核心依赖：`ultralytics==8.4.120`、`opencv-python>=4.10`
- 部署可选：`onnx>=1.17`、`onnxruntime>=1.19`（CPU）或 `onnxruntime-gpu>=1.19`（CUDA，勿同时安装）
- 服务：`fastapi`、`uvicorn`、`python-multipart`、`onnxruntime`
- 测试：`pytest>=8`
- 声明式流水线（可选）：`dvc>=3.63,<4`

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[train,inference,test]"
# 部署（CPU）
.\.venv\Scripts\python.exe -m pip install -e ".[deployment,test]"
# HTTP 服务
.\.venv\Scripts\python.exe -m pip install -e ".[service]"
# DVC 流水线
.\.venv\Scripts\python.exe -m pip install -e ".[pipeline]"
```

> ⚠️ 升级 `ultralytics` 版本前，必须先重新运行全部测试与冒烟训练——项目对训练管线有精确的确定性控制（`deterministic=True`、固定 seed 与 batch），版本漂移可能破坏可复现性。

---

## 十、目录结构

```text
src/helmet_safety/        核心库（data / training / inference / tracking / service / quality）
scripts/                  命令行入口（data / train / evaluate / analyze / inference / deploy / ci）
configs/                  YOLO/tracker 配置与版本化模型注册清单
tests/                    263 个自动化测试
artifacts/                实验权重、评估报告、部署产物（ARTIFACTS_INDEX.md 索引）
_knowledge_base/          领域研究笔记
dvc.yaml / params.yaml    跨机声明式数据与验证流水线
```

**Artifacts 治理约定**：正式实验只长期保留 `best.pt`、参数、结果表与审计报告；`last.pt` 仅在仍可续训（含优化器状态）时保留；数据集硬链接、裁剪、批次图属临时产物，验收后清理；E4 路径固定，新实验使用新目录，不覆盖历史。
