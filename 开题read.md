# 基于深度学习的安全帽佩戴检测系统设计与实现

## ——本科毕业设计（论文）开题报告

| 项目 | 内容 |
|---|---|
| 学生姓名 | 蔡秉昊 |
| 学号 | （待填写） |
| 专业 / 班级 | （待填写） |
| 指导教师 | （待填写） |
| 开题日期 | 2026 年 9 月 |
| 代码仓库 | <https://github.com/kenesada1/helmet-safety-system> |
| 前期工作 | 已完成数据治理、十轮受控实验、实时追踪验收与工程化闭环 |

---

## 摘要

本课题面向建筑、电力、制造等高风险作业场景中安全帽佩戴的自动监管需求，研究并实现一套基于深度学习的头盔佩戴检测系统。课题以 SHWD（Safety Helmet Wearing Dataset）为数据基础，以 YOLO11 系列单阶段检测器为技术主线，围绕**微小目标漏检**这一核心难点，采用**单变量受控实验**的方法论，完成了主干选型、数据级优化、结构级优化三阶段共十轮对照实验，逐层量化了输入分辨率、模型容量、数据分布与检测头结构对检测精度的影响及各自代价。

**已完成的前期工作**：只读可审计的数据治理管线（含独立反算验证）；E4 主模型（YOLO11s / imgsz=960 / 75 轮，mAP50 = 0.9650、mAP50-95 = 0.6430）；十轮受控实验的完整结论与负结果记录；后处理阈值单变量校准；满足 ≥25 FPS 验收线的实时多目标追踪流水线；以及可复现的工程化闭环（HTTP 推理服务、模型版本治理、持续集成门禁）。

**阶段性结论**：在 SHWD 上，输入分辨率与模型容量是主要增益来源；数据级手段（定向重采样、上下文裁剪）对微小目标有局部提升但伴随总体退化，分别判定为**无效**与**有限正结果**；P2 高分辨率检测头显著提升微小目标召回（+4.69 pp），但 GFLOPs 增加 58%、推理速度接近腰斩，实时性代价过大；在其上叠加轻量特征精炼模块（LFR）后，以仅 +0.19% 的参数量换回全实验最高的 mAP50-95（0.6479）与最少的类别混淆。据此确定 **E4 为当前生产候选、E7 为结构优化方向**。

**下一步重点**：建立现场/视频级**事件级**验收基准，验证 E7 在实时链路中的可行性，并完成毕业论文撰写。

---

## 一、选题背景与研究意义

### 1.1 行业背景与安全需求

头部伤害是建筑、电力、制造等高风险行业最主要的伤亡类型之一，高处坠落时的物体打击与碰撞事故中，安全帽是成本最低、有效性最高的个体防护装备。美国职业安全与健康管理局（OSHA）在 1910.135 条款中明确要求雇主必须确保相关员工佩戴头部防护装备[^osha]。

然而，法规约束的是**雇主的管理责任**，而非技术方案的性能指标。OSHA 并未规定视觉人工智能系统的召回率阈值——这意味着安全帽检测的自动化**不能替代任何安全管理责任**，只能作为辅助手段嵌入既有安全体系[^osha][^nist]。

### 1.2 现有监管方式的局限

当前施工现场对安全帽佩戴的监管主要依赖人工巡检与班组长目视检查，存在三个结构性缺陷：

1. **覆盖不连续**——巡检是抽样行为，无法对全部作业面、全部时段实现持续覆盖；
2. **成本高且难追溯**——依赖人力投入，且缺少结构化记录，事后难以追责与统计；
3. **无法实时干预**——即使发现违规，也往往滞后于风险发生时刻。

因此，需要一套**自动、连续、可追溯**的视觉监测方案，把"事后追责"前移为"事中预警"。

### 1.3 本课题面临的技术挑战

安全帽检测并非通用目标检测的简单套用，本课题在前期工作中识别出以下五个具体难点：

| 挑战 | 具体表现 |
|---|---|
| **目标尺度极小** | SHWD 中存在大量 ≤20 像素的目标，密集人群中未佩戴安全帽的头部尤甚，下采样后特征易消失 |
| **类别外观高度相似** | `helmet` 与 `no_helmet` 在低分辨率下的差异仅体现在头顶区域少量像素上，易发生类别混淆 |
| **实时性硬约束** | 安全告警需在秒级内完成，系统须满足 ≥25 FPS 的完整链路吞吐 |
| **数据集与现场存在差距** | 已有研究明确指出 SHWD 缺少复杂环境小目标，未戴帽图像大量来自非施工场景，不能代表真实生产现场[^shwd-limits] |
| **验收标准缺失** | 不存在权威统一的召回率及格线；NIST AI RMF 与 ISO/IEC 23894 均强调风险容忍度由具体应用上下文决定[^nist][^iso]，视频分析应按**事件级**而非逐帧指标验收[^bosch] |

### 1.4 研究意义

**方法层面的意义。** 小目标检测的改进方法众多（数据增强、损失函数、标签分配、高分辨率检测头等），但在**算力受限且带实时性约束**的实际系统中，这些方法的边际收益与代价排序**缺乏公开的受控对照数据**。本课题以一套完整可复现的系统为载体，用单变量受控实验逐个量化各优化维度，并完整留档负结果，为同类工业视觉任务提供可直接复用的实验方法论与决策依据。

**应用层面的意义。** 课题最终交付的不是一个孤立的模型文件，而是一套**可实时运行、可复现、可审计、可追溯**的检测系统，包含数据治理、模型训练、性能评估、实时追踪与部署服务全链路，具备向实际施工现场迁移的基础条件。

---

## 二、国内外研究现状

### 2.1 目标检测技术演进

深度学习目标检测大致经历三条技术路线：以 Faster R-CNN 为代表的**两阶段检测器**精度较高但推理较慢；以 YOLO、SSD、RetinaNet 为代表的**单阶段检测器**直接在特征图上回归目标，在速度与精度之间取得更好平衡；以 DETR 系列为代表的**基于 Transformer 的检测器**引入端到端集合预测，简化了后处理但训练成本较高。

在工业视觉落地场景中，YOLO 系列因其速度—精度—部署生态的综合优势应用最为广泛。本课题基线选用 Ultralytics YOLO11，正是基于其充分的工程成熟度与可控的部署路径[^ultralytics]。

### 2.2 安全帽 / PPE 检测研究现状

**公开研究的精度水平。** 目前已有多项针对安全帽检测的改进工作。基于改进 YOLOv10 的小波动态增强方法 YOLOv10n-WDE 在 SHWD 上取得整体 Precision = 92.9%、Recall = 87.6%[^yolov10]；电力作业防护用品检测研究（MRC-DETR）报告 helmet 类别 Precision = 0.966、Recall = 0.948[^mrcdetr]。公开工程项目中，`yolo11-hard-hat-detection` 报告 P = 0.925、R = 0.897、mAP50 = 0.943，但项目自述其为"可复现的实验管线，而非生产安全监控系统"。

**数据集层面的局限。** 已有研究明确指出，SHWD 缺少复杂环境下的极小目标，且未戴帽图像大量取自非施工场景，因此**不属于标准生产现场数据集**，不能满足真实生产环境的验证要求[^shwd-limits]。这一结论直接适用于本课题：即使模型在 SHWD 上取得较高指标，仍需构建独立的现场验证集。

**生产验收层面的断层。** 公开研究普遍报告 mAP、Precision、Recall，但极少给出生产误报/小时、违规事件检出率、持续时间与人工复核闭环，因此**无法由论文指标直接推出生产准入线**[^nist][^bosch]。工程实践中，`worksite-safety-monitor` 将逐帧的未戴帽检测聚合成持续事件（1.5 秒宽限窗口、3 秒最短持续时间），说明生产系统的最终输出应当是**事件**而非孤立检测框。

### 2.3 小目标检测研究现状

针对小目标的改进方法可归纳为三个层级：

**（1）数据级方法。** Kisantal 等提出面向小目标的复制增强，将小目标实例复制粘贴到图像中以增加单图内的小目标数量，但论文明确指出这是**用大目标质量换取小目标质量的权衡**[^kisantal]。Scale Match 指出训练与推理之间的目标尺度分布失配会损害微小目标的表征能力，支持对裁剪后的放大尺度进行约束[^scale-match]。SNIP 与 SNIPER 提出尺度受控训练，仅让处于合适尺度区间的目标参与梯度回传，并引入背景芯片控制误检[^snip][^sniper]。上下文驱动的数据增强研究则强调目标级复制必须保留上下文，孤立粘贴会产生位置语义不合理与局部外观不一致[^context-aug][^traffic-context]。

**（2）度量与标签分配级方法。** 归一化高斯瓦瑟斯坦距离（NWD）指出交并比对微小框的少量像素偏移极为敏感，并提出更适合微小明目标的相似度度量[^nwd]。RFLA 通过高斯感受野距离与分层标签分配改善微小目标的正样本分配[^rfla]。在线困难样本挖掘（OHEM）与 Focal Loss 则分别从样本选择与损失加权角度应对困难样本与前景背景不平衡[^ohem][^focal]。但这些方法的官方实现多基于 MMDetection 等框架，迁移到 YOLO11 的任务对齐分配器需要重写训练损失与分配逻辑，改造成本较高。

**（3）结构级方法。** 增加 P2 高分辨率检测头是最直接的手段。QueryDet 进一步指出小目标困难源于下采样后特征消失、背景噪声污染与感受野不匹配，提出"低分辨率粗定位 + 仅在潜在区域稀疏计算高分辨率特征"的级联稀疏查询方案，在降低全图高分辨率计算量的同时保持小目标精度[^querydet]。

### 2.4 现状总结与本课题切入点

综合上述现状，可以提炼出三点判断：

1. **"高精度"报告已不稀缺，但小目标漏检是共性瓶颈。** 现有工作在数据集内的整体指标上已较为成熟，然而对"微小目标为什么漏检、哪一层改进最有效"缺少**受控归因**实验。
2. **小目标方法众多，但在实时约束下的收益/代价排序缺乏系统对照。** 多数研究单独验证某一方法的有效性，鲜有在同一基线、同一评估口径下横向比较数据级、结构级改进的实际代价。
3. **论文指标与生产事件级验收之间存在断层。** 逐帧 Recall 无法直接回答"系统每月会漏掉几起真实违规"。

本课题正是针对以上三点切入：以一套可复现的检测系统为载体，用**单变量受控实验**回答"在算力与实时性约束下，哪一层杠杆最值得投入"，并给出可执行的验收路径。

---

## 三、研究目标、内容与关键问题

### 3.1 研究目标

1. 构建一条**数据可信**的检测链路：从原始数据审计、格式转换到独立验证，保证训练数据的正确性与可追溯性。
2. 用**单变量受控实验**量化输入分辨率、模型容量、数据分布、检测头结构四个优化维度对微小目标检测的边际收益与代价，给出可解释的优化优先级。
3. 在 **≥25 FPS 实时性约束**下确定生产候选模型，并完成多目标追踪链路的实时化验证。
4. 将逐帧检测指标转化为可解释的**事件级验收门槛**，并明确系统在安全体系中的定位与边界。

### 3.2 主要研究内容

| 编号 | 研究内容 | 说明 |
|---|---|---|
| **内容一** | 数据治理与可信性验证 | 只读审计、VOC→YOLO 转换契约、独立反算验证、数据划分与泄漏检查 |
| **内容二** | 单变量受控的小目标优化实验 | 三阶段十轮实验：主干选型（M4→E4）、数据级优化（E5a/E5b）、结构级优化（E6/E7/E8） |
| **内容三** | 后处理校准与实时追踪 | 置信度/NMS IoU 单变量扫描；实时多目标追踪流水线设计与验收 |
| **内容四** | 工程化与可复现性保障 | 模型版本治理、HTTP 推理服务、持续集成门禁、跨机声明式流水线 |

### 3.3 拟解决的关键问题

**关键问题一：如何在保证总体指标不退化前提下提升微小目标召回？**
数据级手段（重采样、裁剪）虽能提升微小目标召回，却带来误检上升与总体指标退化。需要回答：瓶颈究竟在数据分布，还是在网络结构对微小目标的感知能力？

**关键问题二：结构级优化如何在精度与实时性之间取得平衡？**
P2 高分辨率检测头能显著提升微小目标召回，但算力代价巨大。需要设计轻量化的结构增强方案，在实时约束下逼近精度收益。

**关键问题三：如何将逐帧检测指标转换为可解释的生产验收门槛？**
安全关键任务的验收不能只看离线 Recall。需要建立"允许漏报预算反推召回门槛"的换算逻辑，并设计事件级评价指标。

### 3.4 本课题特色与创新点

> 说明：本课题的创新主要体现在**方法学与工程实践层面**，而非提出全新的网络结构或损失函数。

1. **可证伪的受控实验范式。** 每轮实验只变化一个变量，在实验开始前**预先声明成功判据**，且 `test` 集不参与训练、选模、调参与阈值选择。所有实验共用同一套完整验证集与统一评估口径，保证结论可比、可归因、可复现。
2. **负结果完整留档。** E5-A 重采样与 E5-B1 裁剪均未达成功门槛，被作为负结果保留而非丢弃——它们证明了"此路不通"，反向定位出真正的瓶颈在网络结构，为后续实验指明方向。
3. **"精度—算力—实时性"三元权衡视角。** 不以单一精度指标作为优化目标，而是同步报告参数量、GFLOPs、推理延迟与吞吐，把实时性作为与精度并列的一等约束。
4. **数据可信优先的工程约束。** 转换结果不接受自证，而是由独立脚本重新读取原始标注逐框反算比对，坐标误差须 ≤ 0.01 px；宁可失败也不产出脏数据。
5. **科研结论直接落地为可运行系统。** 实验结论不停留在报告里，而是以模型注册清单、服务接口与验收门槛的形式固化到系统中，形成"研究—验证—交付"闭环。

---

## 四、技术路线与总体方案

### 4.1 总体技术路线

```text
                    ┌─────────────────────────────────┐
                    │  阶段一：数据治理（已完成）        │
                    │  只读审计 → VOC→YOLO 转换        │
                    │  → 独立反算验证 → 划分与泄漏检查   │
                    └────────────────┬────────────────┘
                                     ▼
                    ┌─────────────────────────────────┐
                    │  阶段二：主干选型受控实验（已完成）  │
                    │  M4 基线 → E1 轮数 → E2 分辨率   │
                    │  → E3 容量 → E4 组合             │
                    └────────────────┬────────────────┘
                                     ▼
        ┌────────────────────────────┴────────────────────────────┐
        ▼                                                         ▼
┌───────────────────────┐                        ┌───────────────────────┐
│ 阶段三：数据级优化      │                        │ 阶段四：结构级优化      │
│ （已完成，判为无效）    │                        │ （已完成，判定有效）    │
│ E5a 定向重采样         │                        │ E6  P2 检测头          │
│ E5b 上下文裁剪         │                        │ E7  P2 + LFR           │
└───────────┬───────────┘                        │ E8  去 ECA 消融        │
            │                                    └───────────┬───────────┘
            └────────────────────┬───────────────────────────┘
                                 ▼
                    ┌─────────────────────────────────┐
                    │  阶段五：后处理校准与实时化（已完成）│
                    │  conf / NMS IoU 单变量扫描        │
                    │  实时追踪流水线（≥25 FPS 验收）    │
                    └────────────────┬────────────────┘
                                     ▼
                    ┌─────────────────────────────────┐
                    │  阶段六：验收与交付（进行中）      │
                    │  事件级基准 + 工程化服务 + 论文    │
                    └─────────────────────────────────┘
```

### 4.2 数据集

采用 SHWD（Safety Helmet Wearing Dataset），以 VOC 格式提供，包含两个类别：`hat`（佩戴安全帽）与 `person`（未佩戴安全帽）。

| 划分 | 图像数 | 用途约束 |
|---|---|---|
| 训练集 | 5,457 | 仅用于训练 |
| 验证集 | 607 | 用于选模、调参、阈值选择（全部实验共用的统一口径） |
| 测试集 | 1,517 | **仅用于最终独立评估**，不参与任何训练、选模、调参或阈值选择 |

**数据转换契约**（转换器强制执行的规则）：

- `hat → 类别 0 helmet`，`person → 类别 1 no_helmet`；`dog` 仅忽略并记入报告；
- 出现任何其他未预期类别 → **在写出数据之前直接失败**；
- 无有效对象的图像予以保留并生成空标签文件；
- 标签回算到像素坐标的误差必须 ≤ 0.01 px；
- 仅允许在确认解码像素完全一致后裁剪 JPEG 在 EOI 之后的冗余字节，并记录 SHA-256 与删除字节数。

### 4.3 评价指标

| 层次 | 指标 | 用途 |
|---|---|---|
| 标准评估 | mAP50、mAP50-95、Precision、Recall | 与公开研究横向对标 |
| 固定阈值计数 | TP / FN / FP / F1（class-aware 一对一匹配） | 微小目标诊断，避免 mAP 平滑掩盖漏检 |
| 尺寸分层 | tiny（≤10 px）与 small（≤20 px）分档召回率 | 定位瓶颈所在 |
| 效率指标 | 参数量、GFLOPs、单图延迟、FPS | 实时性约束下的代价核算 |
| 部署指标 | PyTorch / ORT 后端延迟与吞吐 | 部署路径可行性 |
| **（拟补充）事件级** | 违规事件 Recall、误报 /（摄像头·小时）、告警延迟 | 生产验收门槛 |

> **说明：** 项目不一致地使用 mAP 与固定阈值 F1 是刻意的——mAP 是对全体目标的全阈值积分，容易被大量易检目标掩盖微小目标的漏检；固定阈值计数则直接回答"在部署阈值下究竟漏了几个、误报几个"。

### 4.4 实验方法论（核心控制原则）

1. **单变量控制**——每轮实验只改变一个维度（轮数 / 输入分辨率 / 模型容量 / 数据分布 / 检测头结构），其余超参数全部冻结。
2. **预声明判据**——每轮实验在开跑之前先写明"什么算成功"（例如"微小目标召回率提升且总体 F1 下降不超过 0.002"），以证据而非主观感受做决策。
3. **统一评估口径**——所有实验使用同一套完整验证集（607 张 / 9,925 个 GT 框），并明确标注各系列实验的锚点配置差异（E5 系列以 E4 候选 C，即 conf=0.20 / IoU=0.50 为锚点；E6-E8 系列以标准配置 conf=0.25 / IoU=0.70 为锚点）。
4. **测试集隔离**——`test` 集永不参与训练、选模、调参或阈值选择，这是评估可信度的底线。
5. **负结果留档**——失败的实验与成功的实验同等重要，完整保留权重、配置与报告。

### 4.5 系统总体架构

```text
数据层    SHWD（只读）→ 审计 → 转换 → 独立验证 → 训练集/验证集/测试集
              │
模型层    训练（单变量控制实验）→ 评估（标准 val + 固定阈值计数 + 尺寸分层）
              │
推理层    图片检测 / 视频检测 → 多目标追踪 → 实时追踪流水线（≥25 FPS）
              │
部署层    ONNX 导出（opset 17）→ ORT 性能基准 → HTTP 推理服务 + 模型版本治理
              │
保障层    持续集成门禁 / 声明式流水线 / 全量自动化测试
```

**源代码模块**（[`src/helmet_safety/`](src/helmet_safety/)）：

| 模块 | 职责 |
|---|---|
| [`data/`](src/helmet_safety/data/) | SHWD 只读审计、VOC→YOLO 转换、独立反算验证、VOC 解析 |
| [`training/`](src/helmet_safety/training/) | 各实验的训练、评估、阈值校准、错误分析与结构迁移 |
| [`inference/`](src/helmet_safety/inference/) | OpenCV 图片/视频检测、切片推理、FP16 推理 |
| [`tracking/`](src/helmet_safety/tracking/) | ByteTrack/BoT-SORT 封装、绘制、实时追踪流水线 |
| [`service/`](src/helmet_safety/service/) | HTTP 推理服务、版本化模型注册、线程安全监控 |
| [`quality/`](src/helmet_safety/quality/) | 依赖策略门禁 |

---

## 五、已完成的前期工作与阶段性成果

> 本节所有指标均来自本机实测报告，原始产物与 SHA256 校验值记录于 [`artifacts/ARTIFACTS_INDEX.md`](artifacts/ARTIFACTS_INDEX.md)。

### 5.1 内容一：数据治理（✅ 已完成）

建立了一条**只读、可审计、可独立验证**的数据管线：

- **只读审计**：以只读方式扫描原始 SHWD VOC 数据，绝不修改原始 XML、图像与划分文件；
- **严格转换契约**：遇到未知类别在写出数据前直接失败（详见 4.2）；
- **独立验证**：验证脚本不信任转换报告，而是**重新读取 XML 与 YOLO 标签逐框比较**，坐标回算误差必须 ≤ 0.01 px；
- **可无损修复**：JPEG 若在有效 EOI 之后带有多余字节，仅在确认解码像素完全一致后才裁剪；
- **生成物治理**：以清单文件记录工具生成过的文件，配合显式 `--force` 精确清理，绝不误删用户文件。

> **阶段思想：** 先在数据上建立确定性，模型才有资格谈概率。

### 5.2 内容二（阶段一）：主干选型与控制实验（✅ 已完成）

从 M4 基线（YOLO11n / 640 / 50 轮）出发，逐变量验证"延长训练 / 提高分辨率 / 加大容量"的贡献。评估口径为标准 Ultralytics val（完整 607 张验证集）。

| 实验 | 架构 | 输入 | 轮数 | Precision | Recall | mAP50 | mAP50-95 | ΔmAP50-95 | 结论 |
|---|---|---|---|---|---|---|---|---|---|
| **M4** | YOLO11n | 640 | 50 | 0.9289 | 0.8894 | 0.9383 | 0.6038 | — | 基线 |
| **E1** | YOLO11n | 640 | 100 | 0.9336 | 0.8809 | 0.9422 | 0.6105 | +0.67 pp | 延长轮数收益有限，50 轮后进入平台期 |
| **E2** | YOLO11n | 960 | 50 | 0.9445 | 0.9134 | 0.9590 | 0.6326 | +2.88 pp | 提高分辨率收益显著，微小目标 Recall 0.484 → 0.578 |
| **E3** | YOLO11s | 640 | 50 | 0.9450 | 0.8950 | 0.9560 | 0.6281 | +2.42 pp | 加大容量有效，但 640 下仍受限于分辨率 |
| **E4** | YOLO11s | 960 | 75 | 0.9491 | 0.9281 | 0.9650 | 0.6430 | +3.92 pp | ✅ **当前生产候选**，容量 + 分辨率双升 |

**阶段一结论**：输入分辨率（E2）与模型容量（E3）是主要增益来源，二者叠加（E4）收益最高；单纯延长训练轮数（E1）收益有限，不值得追加算力。参数量从 2.59M（n）增至 9.41M（s），验证了"微小目标需要更大的有效像素与更强的特征表达能力"。

### 5.3 内容二（阶段二）：数据级优化（✅ 已完成，判定为负结果 / 有限正结果）

针对 E4 的微小目标漏检，从数据分布层面尝试两条路线。评估口径为**固定阈值计数**（class-aware 一对一匹配），以 E4 候选 C（conf=0.20 / NMS IoU=0.50）为锚点。

| 实验 | 方案 | 微小目标 R | 微小目标 F1 | 总体 TP/FN/FP | 总体 F1 | 判决 |
|---|---|---|---|---|---|---|
| E4（锚点） | 候选 C 配置 | 0.6172 | 0.6077 | 9420 / 505 / 951 | 0.9283 | 参考基准 |
| 延长训练对照 | 同轮数公平对照 | 0.6484 | 0.6014 | 9446 / 479 / 1067 | 0.9244 | 公平对照组 |
| **E5a** | 微小困难样本定向重采样 | 0.6719 | 0.5931 | 9409 / 516 / 1088 | 0.9215 | ❌ **无效**：微小 R 升但总体退化，未达预声明判据 |
| **E5b-B1** | 微小目标上下文裁剪放大 | 0.6563 | 0.6109 | 9431 / 494 / 1058 | 0.9240 | ⚠️ 有限正结果，整体不优于 E4，保留为研究记录 |

**负结果的归因分析（本课题的重要发现之一）。** E5a 相对延长训练对照仅多正确检出 3 个微小目标，却多出 11 个微小误检。进一步审计发现，失败的关键并非"重复"本身，而是**重复高度集中**：1,043 个额外训练位置只覆盖 151 张图，前 10 张图占据额外抽样的 31%，单张图最高额外出现 74 次。被高频重复的图像中还存在极密集标注（单图 152 个真实框）。这说明模型被少数固定背景与拍摄条件反复推动，其表现符合**场景记忆 / 分布偏移**，而非真正学到了微小目标的多样性。

**阶段二结论**：数据级手段对微小目标有局部提升，但都伴随总体误检上升或综合指标损失，未通过成功门槛。这一负结果反向验证了**瓶颈不在数据分布，而在网络结构对微小目标的感知能力**，为阶段三指明方向。

### 5.4 内容二（阶段三）：结构级优化（✅ 已完成，判定为有效）

通过自定义 P2 高分辨率检测头与轻量特征精炼模块（Lite Feature Refinement, LFR），直接增强网络对微小目标的感知能力。评估口径为标准 val + 固定阈值计数（conf=0.25 / NMS IoU=0.70），以 E4 标准配置为锚点。

| 实验 | 结构 | 参数量 | GFLOPs@960 | Precision | Recall | mAP50 | mAP50-95 | 固定阈值 F1 | 类别混淆 | 推理 ms/图 | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **E4** | YOLO11s | 9.41M | 48.5 | 0.9491 | 0.9281 | 0.9650 | 0.6430 | 0.9252 | 14 | 12.5 | 79.9 |
| **E6** | + P2 检测头 | 9.66M | 76.7 | 0.9434 | 0.9236 | 0.9668 | 0.6445 | 0.9242 | 20 | 21.4 | 46.8 |
| **E7** | P2 + LFR | 9.68M | 78.8 | 0.9532 | 0.9231 | 0.9672 | **0.6479** | **0.9283** | **10** | 23.5 | 42.5 |
| **E8** | P2 + LFR（去 ECA） | 9.68M | 78.8 | 0.9544 | 0.9228 | 0.9683 | 0.6440 | 0.9257 | 15 | 22.9 | 43.7 |

**微小目标（≤10 px）召回率分档对比：**

| 实验 | tiny 总体 R | helmet tiny R | no_helmet tiny R |
|---|---|---|---|
| E4 | 0.6016 | 0.2857 | 0.6404 |
| E6 | **0.6484** | 0.3571 | **0.6842** |
| E7 | 0.6172 | 0.2857 | 0.6579 |
| E8 | 0.6172 | **0.4286** | 0.6404 |

**阶段三结论**：

- **E6（P2 检测头）**：微小目标召回率提升最明显（+4.69 pp），但 GFLOPs 增加 58%、推理速度接近腰斩，且类别混淆反而上升（14 → 20），**实时性代价过大**。
- **E7（P2 + LFR）**：轻量特征精炼（深度可分离卷积 + ECA 通道注意力 + 残差）以仅 +0.19% 的参数量，换回全实验最高的 mAP50-95（0.6479）与最少的类别混淆（10），是结构与精度的最优平衡点。判定为 P2 精炼**有效**。
- **E8（去 ECA 消融）**：移除 ECA 通道注意力后精度回落、类别混淆回升，证明 ECA 在 P2 浅层特征增强中**不可或缺**，同时证明 E7 的收益来自精炼模块整体而非偶然波动。
- **生产决策**：E4 仍作为默认生产候选（实时性与精度综合最优、权重路径稳定）；**E7 作为结构优化方向在途**。

### 5.5 内容三（上半）：后处理阈值校准（✅ 已完成）

模型选定后，用冻结的微小目标验证子集做后处理参数的单变量扫描，回答"降低置信度能救回多少漏检、代价是多少误检"：

- **conf 扫描**：conf 由 0.25 降至 0.15 时，微小目标 TP 由 77 增至 83（+6），但 FP 由 250 增至 420（+1.68×）。**从 conf = 0.15 起误检开始明显增加**，这是实时追踪采用分层置信度策略的依据。
- **NMS IoU 扫描**：IoU ≤ 0.70 时无重复框、FP 稳定；**≥ 0.80 后出现明显重复框对**（0.80 时 98 对，涉及 26 张图）。0.70 为平衡点。
- **四配置验证**：conf × IoU 四组合中，固定阈值总体 F1 最高为 conf=0.25 / IoU=0.50（0.9313）；E5 系列采用 conf=0.20 / IoU=0.50 作为锚点以兼顾微小目标。
- **E6 分类别阈值筛选**：在 37 个可行工作点中，按"约束全部满足后最大化总体 F1"选出分类别阈值（`helmet` = 0.42、`no_helmet` = 0.25），该工作点总体 F1 = 0.9251、`no_helmet` 逐框 Recall = 0.9399——**这一结果直接构成了 6.1 中"验收门槛"讨论的输入**。

> **思想：** 阈值不是拍脑袋定的，是扫出来的；降阈值救回的微小目标，必须以可量化的误检增量买单。

### 5.6 内容三（下半）：实时多目标追踪（✅ 已完成，验收达标）

在检测之上叠加轨迹状态（默认 ByteTrack，可选 BoT-SORT），并针对实时性做专项优化：**有界异步采集队列 + 隔帧检测（stride=2）+ 中间帧匀速外推预测 + confirmed track 门控（hits ≥ 3）+ 类别时序投票（窗口 5 / 多数 3）+ FP16 推理**。

2026-09-01 本机实测（RTX 3060 Laptop，E4 best.pt，imgsz=960，stride=2）：

| 验收项 | 目标 | 实测 | 状态 |
|---|---|---|---|
| 完整链路平均吞吐 | ≥ 25 FPS | 25.02 FPS | ✅ |
| P95 完整帧耗时 | ≤ 40 ms | 35.76 ms | ✅ |
| 队列端到端延迟 P95 | ≤ 200 ms | 36.45 ms | ✅ |
| 实际丢帧率 | < 1% | 0.0% | ✅ |
| hits ≤ 2 短轨迹比例 | 显著下降 | 67.7% → 13.6% | ✅ |

> 不节流的全速处理能力约为 51 FPS（876 帧真实视频）。已修复检测帧与中间帧绘制风格不一致导致的标签闪烁问题，最终 228/228 帧轨迹集完全一致。

### 5.7 内容四：工程化与可复现性保障（✅ 已完成）

- **HTTP 推理服务**：基于 FastAPI 的 E4 ONNX 推理接口，模型单次加载 + 并发互斥，提供检测接口、健康检查、Prometheus 与 JSON 双格式监控指标，并同端口提供浏览器摄像头实时监控页；
- **模型版本治理**：版本化注册清单关联"实验—阶段—权重—报告—类别—输入尺寸—SHA256"，服务启动时 fail-closed 校验，杜绝未知权重带病接流量；
- **持续集成门禁**：依赖策略检查 + 全量测试 + 一轮真实 CPU 训练冒烟；
- **声明式流水线**：以 DVC 声明数据审计、转换、独立校验与冒烟训练，实现跨机复现；
- **质量保障**：**263 个自动化测试**（30 个测试文件，`pytest --collect-only` 实测确认），覆盖数据、训练、评估、推理、追踪、部署与服务全链路。

### 5.8 阶段性成果小结

| 维度 | 当前结果 |
|---|---|
| 数据集 | SHWD，两类，5,457 / 607 / 1,517 |
| 主模型 | E4：YOLO11s，imgsz=960，75 轮 |
| 主模型验证指标 | **mAP50 = 0.9650，mAP50-95 = 0.6430**，P = 0.9491，R = 0.9281 |
| 结构优化最优 | E7（P2 + LFR）：mAP50-95 = 0.6479，类别混淆最少（10） |
| 实验完备性 | 10 轮受控实验，统一评估口径，判据预先声明 |
| 实时追踪 | 25.02 FPS 吞吐 / P95 完整帧 35.76 ms / 丢帧率 0% |
| 部署 | ONNX（opset 17）；PyTorch CUDA 37.04 FPS，ORT CUDA 24.30 FPS，ORT CPU 3.71 FPS |
| 质量保障 | 263 个自动化测试 |

---

## 六、存在的问题与下一步工作计划

### 6.1 当前存在的问题与不足

1. **数据集与真实现场存在差距。** 已有研究表明 SHWD 缺少复杂环境小目标，未戴帽样本大量来自非施工场景[^shwd-limits]。当前所有指标均为数据集内指标，**尚不能代表实际施工现场性能**。
2. **微小目标召回率仍然偏低，且与"整体达标"形成鲜明反差。** 从 E6 的阈值校准结果可以清楚看到这一断层：选定的部署工作点整体 `no_helmet` 逐框 Recall 已达 **0.9399**，但同期微小目标（128 个 GT 框）Recall 仅 **0.6406**（82 个 TP / 46 个 FN）。换言之，模型在"整体指标好看"的同时，仍会漏掉约三分之一的极小目标——而安全管理最关心的恰恰是这些远距离、小尺度的违规目标。此外，E7 的 `helmet` 类微小目标召回率最低仅 0.2857，且该档位只有 14 个验证框，统计波动大，**不足以支撑强结论**（这也是前期调研反复强调"不能用 1～2 个框的变化下强判断"的原因）。
3. **E7 尚未在实时链路中验证。** E7 的 mAP50-95 最高，但 GFLOPs 增加 62%、单图推理延迟由 12.5 ms 升至 23.5 ms，目前**只在离线评估中验证**，尚未接入隔帧检测 + 外推的实时追踪流水线验证其可行性。
4. **缺乏事件级评价基准。** 当前评估全部为逐帧、逐框指标，尚未建立违规事件级 Recall 与误报/小时的统计，无法直接回答"系统每月会漏掉几起真实违规"。
5. **生产验收门槛尚未确定。** 权威框架（NIST AI RMF、ISO/IEC 23894）均不提供统一数值门槛[^nist][^iso]，需结合具体应用场景（辅助提醒 / 人工复核 / 自动门禁）反向推导。

### 6.2 下一步工作内容

| 序号 | 工作内容 | 预期产出 |
|---|---|---|
| **1** | **构建事件级验收基准**：采集/整理现场或公开视频，覆盖不同摄像头、班次、距离、遮挡、逆光与极小目标场景；将逐帧检测聚合为持续事件（宽限窗口 + 最短持续时间） | 事件级 Recall、误报/（摄像头·小时）、告警延迟报告 |
| **2** | **E7 实时化验证**：将 E7 权重接入实时追踪流水线（FP16 + 隔帧检测 + 中间帧外推），复测吞吐、P95 延迟与丢帧率 | E7 实时性验收报告，确定是否可替代 E4 |
| **3** | **验收门槛推导**：基于"允许漏报预算"反推最低召回要求（若有 N 起真实违规、最多允许漏 M 起，则最低事件 Recall = 1 − M/N） | 分级验收门槛建议表（影子运行 / 人工复核 / 自动门禁） |
| **4** | **（条件触发）数据级路线收尾**：若 E7 实时化后微小目标仍有明显漏检，再验证 E5-B2 困难背景裁剪与 E5-C 上下文目标级复制 | 决定是否继续投入数据级路线 |
| **5** | **（研究性探索）**：若剩余错误主要是几像素级定位偏差，评估 RFLA 或归一化高斯瓦瑟斯坦距离的迁移可行性；若部署仍需局部高分辨率，考虑 QueryDet 式稀疏查询 | 可行性分析报告（非必选） |
| **6** | **论文撰写与系统整理** | 毕业设计论文、系统使用说明、答辩材料 |

**关于数据级路线的既有伏笔。** 前期调研已为本课题预留了后续实验的清晰路径：若继续数据级路线，应把 1,043 个集中重复的整图位置替换为**有上限、尺度受控、保留上下文的微小目标中心裁剪**，正裁剪来源扩展到全部 278 张含微小目标的训练图（而非仅 161 张历史漏检图），单张原图最多生成 4 个裁剪、单目标最多作为中心 2 次。只有当裁剪路线出现正证据后，才进入目标级复制实验。

### 6.3 进度安排

> 下表为拟定计划，将根据实际进展与导师意见调整。

| 阶段 | 时间 | 主要任务 | 阶段成果 |
|---|---|---|---|
| 开题准备 | 2026.09 | 文献调研、前期工作整理、开题报告撰写 | 开题报告、文献综述 |
| 数据与基线 | 2026.10 | 数据治理管线复现、基线训练与独立测试复现 | 数据审计报告、基线指标 |
| 中期检查 | 2026.11 - 2026.12 | 小目标优化实验复现与补充实验；E7 实时化验证 | 实验对比报告、E7 实时性报告 |
| 验收基准 | 2027.01 - 2027.02 | 事件级基准构建、验收门槛推导 | 事件级评价报告、门槛建议表 |
| 系统集成 | 2027.03 | 系统联调、演示材料与用户说明整理 | 可演示系统、使用文档 |
| 论文撰写 | 2027.04 - 2027.05 | 论文初稿、修改、定稿 | 毕业设计论文 |
| 答辩 | 2027.05 - 2027.06 | 答辩准备与演示 | 答辩 PPT、最终交付物 |

### 6.4 预期成果

1. **毕业设计论文一份**，系统阐述数据治理方法、三阶段受控实验的设计与结论、实时化方案与验收路径。
2. **一套可运行的检测系统**，包含数据治理、模型训练、性能评估、实时追踪与推理服务全链路，具备工程可复现性（263 项自动化测试保障）。
3. **一组可复用的实验结论**，量化输入分辨率、模型容量、数据分布、检测头结构四个维度的边际收益与代价，并提供负结果记录。
4. **一份事件级验收建议**，给出分级准入门槛与应用边界说明。

---

## 七、参考文献

> 本课题的文献调研过程与完整来源 URL 记录于 [`_knowledge_base/`](_knowledge_base/) 下的两篇研究笔记：
> `research-helmet-tiny-object-optimization-20260821.md`（小目标优化方向）与
> `research-ppe-detection-production-recall-20260826.md`（生产上线召回率）。

**小目标检测方法**

[1] KISANTAL M, 等. Augmentation for small object detection[EB/OL]. arXiv:1902.07296, 2019. <https://arxiv.org/abs/1902.07296>

[2] YU X, 等. Scale match for tiny person detection[C]//Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). 2020. <https://openaccess.thecvf.com/content_WACV_2020/html/Yu_Scale_Match_for_Tiny_Person_Detection_WACV_2020_paper.html>

[3] WANG J, 等. A normalized Gaussian Wasserstein distance for tiny object detection[EB/OL]. arXiv:2110.13389, 2021. <https://arxiv.org/abs/2110.13389>

[4] SINGH B, DAVIS L S. An analysis of scale invariance in object detection — SNIP[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2018.

[5] SINGH B, NAJIBI M, DAVIS L S. SNIPER: Efficient multi-scale training[EB/OL]. arXiv:1805.09300, 2018. <https://arxiv.org/abs/1805.09300>

[6] YANG C, 等. QueryDet: Cascaded sparse query for accelerating high-resolution small object detection[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2022.

[7] XU C, 等. RFLA: Gaussian receptive field based label assignment for tiny object detection[C]//Proceedings of the European Conference on Computer Vision (ECCV). 2022. <https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690518.pdf>

[8] SHRIVASTAVA A, GUPTA A, GIRSHICK R. Training region-based object detectors with online hard example mining[EB/OL]. arXiv:1604.03540, 2016. <https://arxiv.org/abs/1604.03540>

[9] LIN T Y, 等. Focal loss for dense object detection[C]//Proceedings of the IEEE International Conference on Computer Vision (ICCV). 2017. <https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper>

[10] 上下文驱动的目标检测数据增强方法[EB/OL]. 2018. <https://github.com/dvornikita/context_aug>

[11] Traffic context aware data augmentation for object detection[EB/OL]. arXiv:2205.00376, 2022. <https://arxiv.org/abs/2205.00376>

**安全帽 / PPE 检测**

[12] DONG, WANG, MIAO（同济大学）, 等. Improved YOLOv10-based real-time helmet detection algorithm for complex scenarios[J]. Journal of Real-Time Image Processing, 2025, 22(6). DOI: 10.1007/s11554-025-01775-y.

[13] 改进 YOLOv5 安全帽检测研究（含 SHWD 数据集局限性论述）[J/OL]. 2024. <https://pmc.ncbi.nlm.nih.gov/articles/PMC11021566/>

[14] MRC-DETR 电力作业防护用品检测研究[J/OL]. 2025. <https://pmc.ncbi.nlm.nih.gov/articles/PMC12251792/>

[15] Jocher G, 等. Ultralytics YOLO11[EB/OL]. <https://github.com/ultralytics/ultralytics>

[16] Ultralytics 数据增强官方文档[EB/OL]. <https://github.com/ultralytics/ultralytics/blob/main/docs/en/guides/yolo-data-augmentation.md>

**生产验收与风险管理**

[17] NIST. AI Risk Management Framework (AI RMF): Core[EB/OL]. <https://airc.nist.gov/airmf-resources/airmf/5-sec-core/>

[18] NIST. AI RMF: Risk tolerance[EB/OL]. <https://airc.nist.gov/airmf-resources/airmf/1-sec-risk/>

[19] ISO/IEC 23894:2023, Information technology — Artificial intelligence — Guidance on risk management[S]. 2023. <https://www.iso.org/es/contents/data/standard/07/73/77304.html>

[20] Bosch Security Systems. Video content analysis: How to benchmark (White Paper)[R]. 2023. <https://cdn.commerce.boschsecurity.com/public/documents/TN_VCA_HowToBenchmar_WhitePaper_enUS_24087854475.pdf>

[21] OSHA. 1910.135 — Head protection[S]. <https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.135>

[22] Industrial safety violation detection[EB/OL]. arXiv:2412.05531, 2024. <https://arxiv.org/abs/2412.05531>

---

## 附录 A　工程实现与部署细节

> 本附录为工程交付说明，供复现与部署使用。完整的工程版说明文档见 [`README_ENGINEERING.md`](README_ENGINEERING.md)。

### A.1 部署性能基准

E4 `best.pt` 已导出为静态 shape、batch=1、FP32、opset 17 的 ONNX（`artifacts/deployment/e4_yolo11s_960.onnx`），导出后自动运行 `onnx.checker`、ORT 加载与真实图片冒烟推理。

2026-08-31 本机实测（batch=1，imgsz=960，计时含预处理 + 推理 + NMS）：

| 后端 | 设备 | 平均延迟 | P50 | P95 | FPS |
|---|---|---|---|---|---|
| PyTorch CUDA | RTX 3060 Laptop GPU | 26.99 ms | 25.46 ms | 36.18 ms | 37.04 |
| ONNX Runtime CUDA | CUDAExecutionProvider | 41.15 ms | 41.83 ms | 62.55 ms | 24.30 |
| ONNX Runtime CPU | Intel Core i7-10870H | 269.69 ms | 258.97 ms | 343.18 ms | 3.71 |

> 测试环境：Windows 10 / i7-10870H / RTX 3060 Laptop / Python 3.14.3 / PyTorch 2.12.1+cu130 / Ultralytics 8.4.120 / ONNX 1.22.0 / ORT 1.29.0。报告已确认 ONNX CUDA 实际使用 CUDAExecutionProvider，无 CPU 冒充。

### A.2 HTTP 推理服务与监控

FastAPI 服务默认从 `configs/model_registry.json` 选择 E4 ONNX 模型，启动时校验"实验—权重—报告"关联与 SHA256；校验失败时进程存活但 readiness 返回 503，不会用未知权重带病接流量。

```powershell
# 安装并启动（默认 http://127.0.0.1:8000）
.\.venv\Scripts\python.exe -m pip install -e ".[service]"
.\.venv\Scripts\python.exe scripts\serve_api.py --host 127.0.0.1 --port 8000

# 推理、健康检查与指标
curl.exe -F "image=@D:\path\to\image.jpg" http://127.0.0.1:8000/v1/detections
curl.exe http://127.0.0.1:8000/health/ready
curl.exe http://127.0.0.1:8000/metrics
```

启动成功后访问 `http://127.0.0.1:8000/` 打开内置监控页，无需额外前端端口或 Node.js 服务。页面通过浏览器 `getUserMedia` 选择本机摄像头，以可配置间隔截取 JPEG 帧发送至 `/v1/detections`，并在 Canvas 上实时叠加检测框与运行指标。

| 环境变量 | 默认值 | 用途 |
|---|---|---|
| `HELMET_MODEL_REGISTRY` | `configs/model_registry.json` | 模型注册清单 |
| `HELMET_MODEL_ID` | `e4-yolo11s-960-onnx` | 部署模型版本 |
| `HELMET_DEVICE` | `cpu` | 推理设备 |
| `HELMET_MAX_UPLOAD_BYTES` | `10485760` | 单次上传上限 |
| `HELMET_MAX_IMAGE_PIXELS` | `20000000` | 解码后像素上限 |
| `HELMET_CONFIDENCE_BASELINE` | `0.80` | 置信度漂移基线 |
| `HELMET_CONFIDENCE_DRIFT_THRESHOLD` | `0.15` | 漂移告警阈值 |

### A.3 持续集成与跨机编排

- `.github/workflows/ci.yml` 在 main 分支 push/PR 时执行依赖策略检查、全量测试与一轮真实 YOLO CPU 训练；训练使用代码生成的小型双类别数据集，不依赖私有 SHWD 路径或下载预训练权重。
- `requirements/ci.lock` 固定 CI 直接依赖，`scripts/ci/check_dependencies.py` 强制 `ultralytics==8.4.120` 在训练、推理与部署组保持精确锁定。
- `dvc.yaml` + `params.yaml` 声明数据审计、转换、独立校验与 CI 冒烟流水线。跨机仅需调整 `params.yaml` 中的只读原始数据路径，然后执行 `dvc repro <stage>`。

### A.4 常用命令

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

### A.5 Docker 部署

镜像不内置模型产物，部署时将版本化的 `artifacts/` 挂载到 `/app/artifacts`，使镜像与模型版本可独立发布。

```powershell
docker build -t helmet-safety-api .
docker run -d -p 8000:8000 `
  -v D:\datasets\helmet-safety\artifacts:/app/artifacts `
  helmet-safety-api
```

### A.6 环境与依赖

- Python ≥ 3.10（本机实测 3.14.3）
- 训练/推理核心依赖：`ultralytics==8.4.120`、`opencv-python>=4.10`
- 部署可选：`onnx>=1.17`、`onnxruntime>=1.19`（CPU）或 `onnxruntime-gpu>=1.19`（CUDA，勿同时安装）
- 服务：`fastapi`、`uvicorn`、`python-multipart`、`onnxruntime`
- 测试：`pytest>=8`
- 声明式流水线（可选）：`dvc>=3.63,<4`

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[train,inference,test]"   # 训练与推理
.\.venv\Scripts\python.exe -m pip install -e ".[deployment,test]"        # 部署（CPU）
.\.venv\Scripts\python.exe -m pip install -e ".[service]"                # HTTP 服务
.\.venv\Scripts\python.exe -m pip install -e ".[pipeline]"               # DVC 流水线
```

> ⚠️ 升级 `ultralytics` 版本前，必须先重新运行全部测试与冒烟训练——项目对训练管线有精确的确定性控制（`deterministic=True`、固定 seed 与 batch），版本漂移可能破坏可复现性。

---

## 附录 B　仓库目录与代码索引

```text
src/helmet_safety/        核心库（data / training / inference / tracking / service / quality）
scripts/                  命令行入口（data / train / evaluate / analyze / inference / deploy / ci）
configs/                  YOLO / tracker 配置与版本化模型注册清单
tests/                    263 个自动化测试
artifacts/                实验权重、评估报告、部署产物（见 ARTIFACTS_INDEX.md）
_knowledge_base/          领域研究笔记（文献调研来源记录）
dvc.yaml / params.yaml    跨机声明式数据与验证流水线
```

**关键产物索引：**

| 内容 | 路径 |
|---|---|
| 主模型权重（E4） | `artifacts/training/m45_yolo11s_e75_960_001/weights/best.pt` |
| 产物总索引与 SHA256 | `artifacts/ARTIFACTS_INDEX.md` |
| 十轮实验统一对比 | `artifacts/evaluation/m45_yolo11s_e75_960_val_001/e0_e4_unified_comparison.md` |
| 微小目标基准报告 | `artifacts/evaluation/m45_yolo11s_e75_960_tiny_val_001/tiny_val_benchmark_report.json` |
| E5a / E5b 对比报告 | `artifacts/e5a/.../e5a_full_validation_comparison.md`、`artifacts/e5b/.../e5b_comparison.md` |
| 部署基准报告 | `artifacts/deployment/benchmark_report.md` |
| 实时追踪验收报告 | `artifacts/tracking/realtime_acceptance_report.md` |
| 模型注册清单 | `configs/model_registry.json` |
| 结构实验配置 | `configs/yolo11s-p2.yaml`、`yolo11s-p2-lfr.yaml`、`yolo11s-p2-lfr-no-eca.yaml` |

**Artifacts 治理约定：** 正式实验只长期保留 `best.pt`、参数、结果表与审计报告；`last.pt` 仅在仍可续训（含优化器状态）时保留；数据集硬链接、裁剪、批次图属临时产物，验收后清理；E4 路径固定，新实验使用新目录，不覆盖历史记录。

---

## 附录 C　术语与缩写

| 缩写 | 全称 / 含义 |
|---|---|
| mAP50 | 交并比阈值为 0.50 时的平均精度均值 |
| mAP50-95 | 交并比阈值从 0.50 到 0.95（步长 0.05）的平均精度均值 |
| TP / FN / FP | 正确检出 / 漏检 / 误检 |
| tiny / small | 按目标像素尺寸分档：≤10 px / ≤20 px |
| P2 | 高分辨率检测头层级（下采样 4 倍的特征图） |
| LFR | Lite Feature Refinement，轻量特征精炼模块 |
| ECA | Efficient Channel Attention，高效通道注意力 |
| GFLOPs | 十亿次浮点运算量，衡量模型计算复杂度 |
| NMS | Non-Maximum Suppression，非极大值抑制 |
| IoU | Intersection over Union，交并比 |
| ONNX / ORT | 开放神经网络交换格式 / ONNX Runtime 推理引擎 |
| SHWD | Safety Helmet Wearing Dataset，安全帽佩戴数据集 |
| PPE | Personal Protective Equipment，个体防护装备 |

---

*本开题报告的实验数据、图表与结论均来自项目仓库中可复现的实验产物，原始报告与校验值见 [`artifacts/ARTIFACTS_INDEX.md`](artifacts/ARTIFACTS_INDEX.md)。*

[^kisantal]: 见参考文献 [1]
[^scale-match]: 见参考文献 [2]
[^nwd]: 见参考文献 [3]
[^snip]: 见参考文献 [4]
[^sniper]: 见参考文献 [5]
[^querydet]: 见参考文献 [6]
[^rfla]: 见参考文献 [7]
[^ohem]: 见参考文献 [8]
[^focal]: 见参考文献 [9]
[^context-aug]: 见参考文献 [10]
[^traffic-context]: 见参考文献 [11]
[^yolov10]: 见参考文献 [12]
[^shwd-limits]: 见参考文献 [13]
[^mrcdetr]: 见参考文献 [14]
[^ultralytics]: 见参考文献 [15]、[16]
[^nist]: 见参考文献 [17]、[18]
[^iso]: 见参考文献 [19]
[^bosch]: 见参考文献 [20]
[^osha]: 见参考文献 [21]
