# SHWD 数据审计与 VOC→YOLO 转换工具

这套工具以只读方式检查原始 SHWD Pascal VOC 数据，继承官方 `train`、`val`、`test` 划分，安全复制图片并生成 Ultralytics YOLO 标签。它不会修改原始 XML、图片或划分文件；复制到 `processed` 的 JPEG 若只在有效 EOI 标记后带有多余字节，则会在确认解码像素完全一致后无损裁掉尾部字节。

## 新手先看：各文件做什么

| 文件 | 职责 | 可以把它理解成 |
|---|---|---|
| `data/voc.py` | 读取一个 XML、检查框、计算 YOLO 坐标 | 基础零件 |
| `data/audit.py` | 扫描整套原始数据并生成审计报告 | 数据体检医生 |
| `data/convert.py` | 按官方 split 复制图片并生成标签 | 数据加工厂 |
| `data/validate.py` | 独立复查输出、反算坐标、生成可视化 | 质量验收员 |
| `scripts/*.py` | 接收命令行参数并调用上述模块 | 命令行开关 |
| `tests/*.py` | 用很小的临时数据验证每项规则 | 防回归安全网 |

建议阅读顺序是 `voc.py → audit.py → convert.py → validate.py`。前三个核心概念是：

1. **图片 ID / stem**：`000377.jpg`、`000377.xml` 和划分文件里的 `000377` 通过不带扩展名的部分对应。
2. **预检查**：转换程序先读完并检查所有 XML，确认全部安全后才开始写 `processed`。
3. **独立验证**：验证程序不直接相信转换报告，而是重新读取 XML 和 YOLO 标签逐框比较。

整体数据流如下：

```text
原始 VOC（只读）
  ├─ audit.py ───────────────> audit_report.json / .md
  └─ convert.py（先预检查）──> processed/images + labels + dataset.yaml
                                      │
                                      └─ validate.py ─> validation_report + visualizations
```

## 目录与命令

```powershell
python -m pytest

python scripts/audit_shwd.py `
  --raw D:\datasets\SHWD\VOC2028 `
  --output D:\datasets\SHWD\audit

python scripts/convert_shwd.py `
  --raw D:\datasets\SHWD\VOC2028 `
  --output D:\datasets\SHWD\processed

python scripts/validate_yolo_dataset.py `
  --raw D:\datasets\SHWD\VOC2028 `
  --processed D:\datasets\SHWD\processed `
  --output D:\datasets\SHWD\audit `
  --visualize --samples-per-split 10 --seed 2028
```

若 `processed` 已存在且非空，转换默认退出。只有显式传入 `--force` 才会根据 `.shwd-generated-files.json` 清单删除并重建本工具生成的文件；未知文件不会删除。可视化重跑同样需要 `--force-visualizations` 和已有可视化清单。

## M3：YOLO 训练冒烟测试

先安装训练依赖，再从项目根目录运行确定性 smoke test：

```powershell
python -m pip install -e ".[train,test]"
python scripts/smoke_train.py `
  --processed D:\datasets\SHWD\processed `
  --source-data-yaml D:\datasets\SHWD\processed\dataset.yaml `
  --epochs 1 --imgsz 640 --batch 4 --workers 0 --seed 42 `
  --train-count 48 --val-count 24 --device auto
```

预训练模型、图片路径列表和训练产物写入 `artifacts/`；Ultralytics 还会在 `processed/labels` 中生成带版本字段的官方 `.cache` 索引文件，它们可以安全删除并会在下次运行时自动重建。运行名已存在时会自动选择安全的新名称。转换阶段已经把可无损修复的 JPEG 规范化，因此训练阶段不再排除这些图片，也不会触发 Ultralytics 自动改写它们。训练依赖固定为已经验证过的 `ultralytics==8.4.120`，升级版本前应重新运行测试和冒烟训练。

## M4：全量 YOLO11n 基线训练与独立测试

M3 冒烟测试只验证训练链路；M4 才使用完整的 5,457 张 train 图片训练，并用 607 张 val 图片选择 `best.pt`。确认 `best.pt` 后，评估入口会先复算逐类别 val 指标，再对 1,517 张 test 图片做一次独立最终评估。test 不参与训练、模型选择、超参数调整或阈值选择。

先运行全部自动化测试，再启动基线训练：

```powershell
.\.venv\Scripts\python.exe -m pytest

.\.venv\Scripts\python.exe scripts\train_baseline.py `
  --data D:\datasets\SHWD\processed\dataset.yaml `
  --validation-report D:\datasets\SHWD\audit\validation_report.json `
  --model .\artifacts\models\yolo11n.pt `
  --epochs 50 --patience 15 --imgsz 640 --batch 8 `
  --workers 0 --device 0 --seed 42 `
  --run-name baseline_yolo11n_001
```

训练入口固定使用 `cache=False`、`amp=True`、`plots=True`、`pretrained=True` 和 `deterministic=True`，其余学习率、优化器及增强参数保持 Ultralytics 8.4.120 默认值。如果 CUDA OOM，会在日志中明确记录后按 8→4→2 降低 batch，绝不降低 `imgsz`；每次失败尝试都会保留自己的日志和失败报告。运行名、日志名或输出目录已存在时会选择下一个 `_002`、`_003` 名称，不覆盖旧实验。

训练成功后，把实际生成的训练报告传给独立评估入口：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_baseline.py `
  --training-report .\artifacts\training\baseline_yolo11n_001\baseline_training_report.json `
  --split test --imgsz 640 --workers 0 --device 0 `
  --seed 42 --prediction-count 100 --conf 0.25
```

评估会生成：

- best.pt 的 val 与 test 总体/逐类别指标、混淆矩阵和 P/R/F1/PR 曲线；
- 固定 seed=42 的 100 张 test 图片清单；
- 文件名一一对应、可单独打开的 `predicted/*.jpg` 与 `ground_truth/*.jpg`；
- `error_analysis.json` 与 `human_review_checklist.csv`，分别归纳漏检、误检、类别混淆、框偏移、小目标、密集场景，以及低光照/低画质候选；
- 初学者可读的 `baseline_evaluation_report.md`。

人工对比时，在预测目录和真实框目录中打开同名 JPG。真实框图中绿色表示 `helmet`，红色表示 `no_helmet`；预测图会同时显示预测类别与置信度。`conf=0.25` 只用于这 100 张可视化和人工错误检查，不是从 test 上寻找的“最佳阈值”。

## 数据规则

- `hat → 0 helmet`
- `person → 1 no_helmet`
- `dog` 仅忽略对象并写入报告
- 任何其他类别都会令转换在写出数据前失败
- 无有效对象的图片保留并生成空 `.txt` 标签
- 标签到像素坐标的回算误差默认必须不超过 `0.01 px`
- JPEG 必须能够严格解码；若有效 EOI 后存在尾部字节，只修改 `processed` 副本，并在报告中记录原始/输出 SHA-256 与删除字节数
- 找不到有效 EOI、严格解码失败或规范化前后像素不一致时，转换会在写出数据前失败

XML 首先按原字节进行标准解析。只有解析失败时，恢复逻辑才会在内存中精确替换 `folder` / `path` 元素的字符内容并重试，不会改写原文件，也不会触碰 `object`、`size` 或 `bndbox`。

## 看懂一行 YOLO 标签

例如：

```text
0 0.300000 0.250000 0.400000 0.300000
```

五列依次是 `class_id x_center y_center width height`。这里 `0` 代表 helmet；后四个数都已经除以图片宽高，所以范围是 0～1。中心点为 `(0.3, 0.25)`，框宽为图片宽度的 40%，框高为图片高度的 30%。

## 为什么需要 manifest

`.shwd-generated-files.json` 记录转换工具自己创建过的文件。普通重跑遇到非空 `processed` 会直接停止；只有使用 `--force` 且存在这份清单时，程序才会删除清单中的文件。这样即使输出目录里混入了你的其他文件，也不会被递归删除。
