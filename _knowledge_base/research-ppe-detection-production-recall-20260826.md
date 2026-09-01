# PPE/安全帽检测生产上线召回率调研

调研日期：2026-08-26
调研目标：确定类似安全帽/PPE视觉检测模型达到何种召回水平才适合生产上线，并为 E4/E6 制定可执行的验收门槛。

## 关键问题

1. 行业标准或权威指南是否规定统一的 Recall/Precision 上线阈值？
2. GitHub、论文或公开项目中的安全帽/PPE模型通常能达到什么指标？
3. 对安全关键视频告警，应该采用逐帧指标还是事件级指标？
4. E4/E6 当前指标距离生产验收还有多大差距？

## 发现

### 第一轮：是否存在统一生产阈值

- NIST AI RMF 明确不规定统一风险容忍度；风险容忍度取决于组织、行业、法规和具体使用场景。部署前需在与生产条件相似的环境中验证，并确认剩余风险不超过组织容忍度；上线后还要持续监控。
- 因此未找到“安全帽模型 Recall 达到某个统一百分比即可上线”的权威标准。单一离线 Recall 不能单独构成生产准入证明。
- AWS Rekognition Custom Labels 也不提供统一生产及格线，而是要求按具体业务在 Precision/Recall 之间选择阈值；官方说明降低置信度提高 Recall、提高置信度改善 Precision，并强调测试数据必须代表最终推理场景。

### 第二轮：公开安全帽/PPE项目水平

- GitHub 项目 `yolo11-hard-hat-detection` 报告 P=0.925、R=0.897、mAP50=0.943，并明确说明它是可复现实验管线，不是生产安全监控系统。
- GitHub 项目 `worksite-safety-monitor` 的验证总体 R=0.598，no-helmet 约有 17% 漏检；项目仍采用较高置信度以减少需要人工处理的告警，说明“可运行的系统”不等于“满足安全闭环的自动执法系统”。
- GitHub 项目 `ppe-helmet-detection` 报告总体 R=0.821、no_helmet R=0.730；另一个 YOLO11 PPE 项目报告总体 R=0.69、no-helmet R=0.68。这些是研究/演示基线，不能作为生产验收下限。
- 公开研究常报告 mAP、Precision、Recall，但很少给出生产误报/小时、违规事件检出率、持续时间和人工复核闭环，因此无法从论文指标直接推出生产准入线。
- 一项工业安全违规研究指出，已部署应用常因对不同任务一概而论而产生大量误报，支持按生产工序/区域建立业务上下文，而不是只提高模型分数。

### 阶段摘要（第2轮）

当前证据表明：公开模型 0.6–0.9 左右的 Recall 只能说明“研究模型能工作”，不能证明可承担安全责任；权威框架要求由实际风险和部署环境决定门槛。后续需要寻找事件级/告警级验收依据，并把逐帧 Recall 转换为生产可理解的漏报风险。

### 第三轮：公开研究的较高水平与数据集局限

- 2025年的电力作业防护用品检测研究报告 helmet Precision=0.966、Recall=0.948，说明约95% Recall 是公开研究中可以达到的水平，但仍是数据集测试结果。
- 改进YOLOv10安全帽论文全文表3显示，YOLOv10n-WDE在SHWD上的整体Precision=92.9%、Recall=87.6%；表4中的95.5%、96.6%是对比模型的类别AP50，不是Precision/Recall。论文摘要、表格与结论还存在若干数字不一致，因此不应将搜索摘要中的数字直接作为生产参照。
- 另一篇安全帽研究明确指出：SHWD缺少复杂环境小目标，未戴帽图像大量来自非施工场景，因此不是标准生产现场数据集，不能满足真实生产环境验证要求。此点直接适用于当前项目：即使E4/E6也使用SHWD，仍需独立现场验证集。

### 第四轮：视频分析应按告警/事件验收

- Bosch视频分析基准白皮书同时要求考察漏报和误报；把Sensitivity定义为TP/(TP+FN)，并建议用每小时/每天/月的误报数量衡量鲁棒性。它还指出误报过多会导致操作员关闭系统，而漏报意味着系统未完成任务。
- `worksite-safety-monitor` 将逐帧no-helmet检测聚合成持续事件，使用1.5秒宽限窗口和3秒最短持续时间；这说明生产系统的最终输出应是事件，而不是孤立检测框。
- OSHA规定雇主必须确保相关员工佩戴头部防护，但没有规定视觉AI的Recall阈值。这意味着AI不能凭一个模型分数自动替代雇主管理责任或其他安全控制。
- ISO/IEC 23894与NIST一致，要求风险管理按组织和应用上下文定制，未给出通用数值门槛。

### 阶段摘要（第4轮）

没有权威来源支持“逐帧Recall≥95%即可上线”这种简单规则。更合理的准入指标是：现场独立视频上的违规事件级Recall、关键切片Recall下界、误报/摄像头小时、端到端延迟、系统可用性，以及人工复核/门禁等兜底控制。

## 调研结论

### 关键事实

1. 没有安全帽AI统一法定Recall及格线；NIST明确风险容忍度由应用上下文决定。
2. 公开研究中可见约94.8%的helmet Recall，但论文指标不等于生产准入；本次核对的YOLOv10n-WDE论文在SHWD上的整体Recall实际为87.6%，尤其SHWD并不代表真实生产现场。
3. 视频生产系统应按完整违规事件统计Recall，并按摄像头小时统计误报；逐帧框Recall只能用于模型开发诊断。
4. 当前E6平衡点no_helmet逐框Recall为93.99%，95% Wilson区间约93.48%–94.45%；极小目标Recall为64.06%，95%区间约55.45%–71.85%，证据不足以支持无人值守安全控制。

### 建议的内部验收门槛（工程建议，不是法规）

| 使用方式 | 违规事件级Recall | 其他必要条件 |
|---|---:|---|
| 影子运行/数据收集 | ≥95%作为目标，不触发生产动作 | 全量留痕、人工标注、统计误报/小时 |
| 人工复核型辅助告警 | ≥98%，95%置信下界≥95% | 关键场景不低于95%；误报量不超过操作员处理能力 |
| 自动门禁/自动停线等高影响控制 | ≥99.5%，并要求冗余与失效安全 | 不能以单一视觉模型作为唯一安全控制；需现场安全评审 |

门槛应由允许漏报预算反推：若每月有N个真实违规事件，最多允许漏M个，则最低事件Recall=1-M/N。例如每月1000个违规最多允许漏1个，需要≥99.9%，不是95%。

### 对E6的结论

- E6可以进入影子运行和现场数据收集，但不应仅凭当前SHWD逐框Recall作为正式安全告警系统，更不能单独承担自动停线/门禁决策。
- 下一步不是继续在SHWD上追逐0.5个百分点，而是建立现场视频事件基准：至少覆盖不同摄像头、班次、距离、遮挡、逆光和极小目标；按违规事件而非帧计数。
- 若定位为有人复核的辅助告警，建议以事件级Recall≥98%、置信下界≥95%作为内部准入目标；若用于自动控制，建议提高到≥99.5%并增加独立传感器或人工兜底。

### 待确认问题

- 生产线每月真实no_helmet违规事件数量及可接受漏报数。
- 每名操作员每小时可处理的误报告警上限。
- 系统是辅助提醒、人工复核，还是会直接联动门禁/停线。
- 摄像头数量、帧率、典型目标像素尺寸及最差光照条件。

## 来源列表

| 来源 | URL | 日期 | 可信度 |
|---|---|---:|---|
| NIST AI RMF Core | https://airc.nist.gov/airmf-resources/airmf/5-sec-core/ | 检索于2026-08-26 | 高，一手官方框架 |
| NIST Risk Tolerance | https://airc.nist.gov/airmf-resources/airmf/1-sec-risk/ | 检索于2026-08-26 | 高，一手官方框架 |
| AWS Custom Labels Metrics | https://docs.aws.amazon.com/rekognition/latest/customlabels-dg/im-metrics-use.html | 检索于2026-08-26 | 高，一手官方文档 |
| AWS Improving a Model | https://docs.aws.amazon.com/rekognition/latest/customlabels-dg/tr-improve-model.html | 检索于2026-08-26 | 高，一手官方文档 |
| yolo11-hard-hat-detection | https://github.com/Grigoriy-V/yolo11-hard-hat-detection | 检索于2026-08-26 | 中，公开项目自报指标 |
| worksite-safety-monitor | https://github.com/worksite-safety/worksite-safety-monitor | 检索于2026-08-26 | 中，公开工程项目 |
| PPE Helmet Detection | https://github.com/BUAksakal/ppe-helmet-detection | 检索于2026-08-26 | 中，公开课程项目 |
| PPE Safety System | https://github.com/NabaTamir/PPE-safety-system | 检索于2026-08-26 | 中，公开项目自报指标 |
| Industrial Safety Violation Detection | https://arxiv.org/abs/2412.05531 | 2024 | 中，一手论文预印本 |
| Bosch视频分析基准白皮书 | https://cdn.commerce.boschsecurity.com/public/documents/TN_VCA_HowToBenchmar_WhitePaper_enUS_24087854475.pdf | 2023版，检索于2026-08-26 | 高，厂商一手技术白皮书 |
| MRC-DETR电力作业PPE研究 | https://pmc.ncbi.nlm.nih.gov/articles/PMC12251792/ | 2025 | 高，同行评议一手论文 |
| 改进YOLOv10安全帽论文 | https://iip.tongji.edu.cn/2025_Improved_YOLOv10based.pdf | 2025 | 高，一手论文 |
| 改进YOLOv5安全帽研究/SHWD局限 | https://pmc.ncbi.nlm.nih.gov/articles/PMC11021566/ | 2024 | 高，同行评议一手论文 |
| OSHA 1910.135头部防护 | https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.135 | 检索于2026-08-26 | 高，监管机构一手标准 |
| ISO/IEC 23894:2023 | https://www.iso.org/es/contents/data/standard/07/73/77304.html | 2023 | 高，国际标准官方摘要 |
