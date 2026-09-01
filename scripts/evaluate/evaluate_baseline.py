#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback

from PIL import Image, ImageFilter, ImageStat


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from helmet_safety.training.baseline import (  # noqa: E402
    allocate_run_name,
    analyze_image_detections,
    format_evaluation_metrics,
    load_ground_truth_boxes,
    save_ground_truth_visualization,
    select_prediction_images,
    write_json_report,
)


class TeeStream:
    def __init__(self, terminal: object, log_file: object) -> None:
        self.terminal = terminal
        self.log_file = log_file

    def write(self, text: str) -> int:
        self.terminal.write(text)
        self.log_file.write(text)
        return len(text)

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.terminal, "isatty", lambda: False)())

    @property
    def encoding(self) -> str:
        return getattr(self.terminal, "encoding", "utf-8") or "utf-8"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a completed M4 baseline on val and one independent test run")
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--split", choices=("test",), default="test", help="Final independent split; deliberately test-only")
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=None, help="Defaults to the actual training batch")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prediction-count", type=int, default=100)
    parser.add_argument("--conf", type=float, default=0.25, help="Visualization/error-review threshold only; never tuned on test")
    parser.add_argument("--val-run-name", default="baseline_yolo11n_val_001")
    parser.add_argument("--test-run-name", default="baseline_yolo11n_test_001")
    parser.add_argument("--prediction-run-name", default="baseline_yolo11n_predictions_001")
    return parser


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_plot_outputs(save_dir: Path) -> dict[str, str]:
    required = {
        "confusion_matrix": "confusion_matrix.png",
        "confusion_matrix_normalized": "confusion_matrix_normalized.png",
        "pr_curve": "BoxPR_curve.png",
        "f1_curve": "BoxF1_curve.png",
        "precision_curve": "BoxP_curve.png",
        "recall_curve": "BoxR_curve.png",
    }
    outputs = {name: (save_dir / filename).resolve() for name, filename in required.items()}
    missing = [path for path in outputs.values() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"evaluation plots are missing or empty: {missing}")
    return {name: str(path) for name, path in outputs.items()}


def metric_payload(metrics: object) -> dict[str, object]:
    return format_evaluation_metrics(
        metrics.results_dict,
        metrics.summary(normalize=True, decimals=10),
        metrics.speed,
    )


def image_quality(image_path: Path) -> dict[str, float | bool]:
    with Image.open(image_path) as image:
        gray = image.convert("L")
        brightness = float(ImageStat.Stat(gray).mean[0])
        edge_variance = float(ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES)).var[0])
    return {
        "mean_brightness": brightness,
        "edge_variance": edge_variance,
        "low_light_candidate": brightness < 60.0,
        "low_quality_candidate": edge_variance < 50.0,
    }


def markdown_report(report: dict[str, object]) -> str:
    training = report["training"]
    val = report["val_evaluation"]["metrics"]
    test = report["test_evaluation"]["metrics"]
    error = report["error_analysis"]
    overfit = training["training_analysis"]["overfitting"]

    def metric_table(metrics: dict[str, object]) -> str:
        rows = [
            f"| overall | {metrics['overall']['precision']:.6f} | {metrics['overall']['recall']:.6f} | {metrics['overall']['map50']:.6f} | {metrics['overall']['map50_95']:.6f} |"
        ]
        for name in ("helmet", "no_helmet"):
            values = metrics["per_class"][name]
            rows.append(
                f"| {name} | {values['precision']:.6f} | {values['recall']:.6f} | {values['map50']:.6f} | {values['map50_95']:.6f} |"
            )
        return "\n".join(rows)

    paths = report["paths"]
    lines = [
        "# M4 YOLO11n 基线训练与独立测试报告",
        "",
        "## 实验协议",
        "",
        f"- 模型：`{training['pretrained_model']}`（迁移学习）",
        f"- 数据：`{training['data']['dataset_yaml']}`",
        f"- 实际 batch：{training['actual_batch']}；imgsz：{training['training_parameters']['imgsz']}；seed：{training['training_parameters']['seed']}",
        f"- 完成 epoch：{training['training_analysis']['epochs_completed']}；最佳 epoch：{training['training_analysis']['best_epoch']}；最后 epoch：{training['training_analysis']['last_epoch']}",
        f"- Early stopping：{training['training_analysis']['early_stopping_triggered']}",
        "- test 只在 best.pt 确定后评估一次；没有用于选模型、调参或选择阈值。",
        "",
        "## 验证集（best.pt 独立复算）",
        "",
        "| 类别 | Precision | Recall | AP50 / mAP50 | AP50-95 / mAP50-95 |",
        "|---|---:|---:|---:|---:|",
        metric_table(val),
        "",
        "## 测试集（最终独立评估）",
        "",
        "| 类别 | Precision | Recall | AP50 / mAP50 | AP50-95 / mAP50-95 |",
        "|---|---:|---:|---:|---:|",
        metric_table(test),
        "",
        f"- 每张图 GPU 推理：{test['speed_ms_per_image']['inference']:.4f} ms",
        f"- GPU 推理吞吐：{test['gpu_inference_images_per_second']:.2f} images/s",
        f"- 含预处理/后处理的平均流水线耗时：{test['average_pipeline_ms_per_image']:.4f} ms/image",
        "",
        "## Loss 与过拟合判断",
        "",
        f"保守规则判定过拟合：**{overfit['detected']}**。best 到最后相隔 {overfit['best_to_last_epoch_gap']} epoch；"
        f"train loss 继续下降={overfit['train_loss_continued_down']}，val loss 恶化={overfit['val_loss_worsened']}，"
        f"mAP50-95 恶化={overfit['map50_95_worsened']}，Precision/Recall 分化={overfit['precision_recall_diverged']}。",
        "",
    ]
    for scope in ("train", "val"):
        losses = training["training_analysis"]["losses"][scope]
        lines.append(
            f"- {scope}: box {losses['box_loss']['first']:.5f}→{losses['box_loss']['last']:.5f}；"
            f"cls {losses['cls_loss']['first']:.5f}→{losses['cls_loss']['last']:.5f}；"
            f"dfl {losses['dfl_loss']['first']:.5f}→{losses['dfl_loss']['last']:.5f}。"
        )
    lines.extend(
        [
            "",
            "## 100 张测试图错误分析（conf=0.25，仅供可视化）",
            "",
            f"- 有错误的图片：{error['images_with_errors']} / {error['sample_count']}",
            f"- 漏检：{error['aggregate_error_counts']['missed_detection']}；其中 no_helmet 漏检：{error['missed_by_class']['no_helmet']}",
            f"- 误检：{error['aggregate_error_counts']['false_positive']}；类别混淆：{error['aggregate_error_counts']['class_confusion']}",
            f"- 框偏移/尺寸不佳候选：{error['aggregate_error_counts']['box_misalignment']}；小目标漏检：{error['aggregate_error_counts']['small_target_miss']}",
            f"- 密集场景错误候选：{error['aggregate_error_counts']['dense_scene_with_errors']}；低照度候选：{error['low_light_error_images']}；低画质候选：{error['low_quality_error_images']}",
            "- 遮挡无法只靠标注框可靠自动判断；已把密集场景与高漏检图片放入人工检查清单。",
            "",
            "安全系统应优先看 no_helmet Recall：它表示真实未戴安全帽人员有多少被发现。漏掉违规人员通常比多报一个候选框风险更高。",
            "",
            "## 怎样人工对比",
            "",
            f"1. 打开真实框目录：`{paths['ground_truth_jpg_dir']}`。",
            f"2. 打开预测框目录：`{paths['prediction_jpg_dir']}`。",
            "3. 两个目录使用相同文件名，左右并排查看；绿色真实框是 helmet，红色真实框是 no_helmet。",
            f"4. 按 `{paths['human_review_checklist']}` 从漏检、误检、混淆、框偏移、小目标、遮挡候选、密集人群、光照/画质逐项复核。",
            "",
            "## 关键路径",
            "",
            f"- best.pt：`{paths['best_pt']}`",
            f"- last.pt：`{paths['last_pt']}`",
            f"- 完整训练日志：`{paths['training_log']}`",
            f"- test 评估目录：`{paths['test_evaluation_dir']}`",
            f"- 预测样本清单：`{paths['prediction_manifest']}`",
            f"- 错误分析 JSON：`{paths['error_analysis_json']}`",
            "",
            "## M5 建议",
            "",
            f"M5 必须从本次基线的 `{paths['best_pt']}` 出发做对照。优先目标是提升 no_helmet Recall，并保持 test 继续只用于阶段末独立评估。",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, object]:
    import torch
    from ultralytics import YOLO

    started_at = utc_now()
    started_clock = time.perf_counter()
    training_report_path = args.training_report.resolve()
    training = json.loads(training_report_path.read_text(encoding="utf-8"))
    if training.get("status") != "passed":
        raise ValueError(f"training report is not passed: {training_report_path}")
    best_pt = Path(training["outputs"]["best_pt"]).resolve()
    last_pt = Path(training["outputs"]["last_pt"]).resolve()
    if not best_pt.is_file() or not last_pt.is_file():
        raise FileNotFoundError("best.pt or last.pt is missing")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {args.device!r} requested but CUDA is unavailable")
    if args.imgsz != 640:
        raise ValueError("M4 final evaluation must use imgsz=640")
    if args.conf != 0.25:
        raise ValueError("M4 prediction visualization must use conf=0.25; test may not tune this threshold")
    batch = int(args.batch or training["actual_batch"])
    processed_root = Path(training["data"]["processed_root"]).resolve()
    data_yaml = Path(training["data"]["dataset_yaml"]).resolve()
    evaluation_dir = args.artifacts_dir.resolve() / "evaluation"
    logs_dir = args.artifacts_dir.resolve() / "logs"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    val_name = allocate_run_name(evaluation_dir, args.val_run_name)
    test_name = allocate_run_name(evaluation_dir, args.test_run_name)
    prediction_name = allocate_run_name(evaluation_dir, args.prediction_run_name)
    log_path = logs_dir / f"{test_name}.log"
    if log_path.exists():
        raise FileExistsError(f"refusing to overwrite evaluation log: {log_path}")

    with log_path.open("x", encoding="utf-8", buffering=1) as log_handle:
        with redirect_stdout(TeeStream(sys.stdout, log_handle)), redirect_stderr(TeeStream(sys.stderr, log_handle)):
            print(json.dumps({"training_report": str(training_report_path), "best_pt": str(best_pt)}, indent=2))
            model = YOLO(str(best_pt), task="detect")
            common_val = {
                "data": str(data_yaml),
                "imgsz": args.imgsz,
                "batch": batch,
                "workers": args.workers,
                "device": args.device,
                "plots": True,
                "seed": args.seed,
                "deterministic": True,
                "project": str(evaluation_dir),
                "exist_ok": False,
            }
            val_metrics_obj = model.val(split="val", name=val_name, **common_val)
            val_save_dir = Path(val_metrics_obj.save_dir).resolve()
            val_payload = metric_payload(val_metrics_obj)
            val_plots = validate_plot_outputs(val_save_dir)

            test_metrics_obj = model.val(split=args.split, name=test_name, **common_val)
            test_save_dir = Path(test_metrics_obj.save_dir).resolve()
            test_payload = metric_payload(test_metrics_obj)
            test_plots = validate_plot_outputs(test_save_dir)

            selected = select_prediction_images(processed_root, count=args.prediction_count, seed=args.seed)
            prediction_root = evaluation_dir / prediction_name
            prediction_root.mkdir(parents=True, exist_ok=False)
            prediction_jpg_dir = prediction_root / "predicted"
            ground_truth_jpg_dir = prediction_root / "ground_truth"
            prediction_jpg_dir.mkdir()
            ground_truth_jpg_dir.mkdir()
            results = model.predict(
                source=[str(path) for path in selected],
                conf=args.conf,
                imgsz=args.imgsz,
                batch=batch,
                device=args.device,
                verbose=False,
                save=False,
                stream=False,
            )
            if len(results) != len(selected):
                raise RuntimeError(f"expected {len(selected)} prediction results, got {len(results)}")

            manifest: list[dict[str, object]] = []
            error_cases: list[dict[str, object]] = []
            aggregate = {
                "missed_detection": 0,
                "false_positive": 0,
                "class_confusion": 0,
                "box_misalignment": 0,
                "small_target_miss": 0,
                "dense_scene_with_errors": 0,
            }
            missed_by_class = {"helmet": 0, "no_helmet": 0}
            false_positive_by_class = {"helmet": 0, "no_helmet": 0}
            confusion_pairs: dict[str, int] = {}
            low_light_errors = 0
            low_quality_errors = 0
            for result in results:
                source_path = Path(result.path).resolve()
                relative = source_path.relative_to(processed_root)
                predicted_path = prediction_jpg_dir / f"{source_path.stem}.jpg"
                ground_truth_path = ground_truth_jpg_dir / f"{source_path.stem}.jpg"
                plotted_bgr = result.plot()
                Image.fromarray(plotted_bgr[:, :, ::-1]).save(predicted_path, format="JPEG", quality=94)
                with Image.open(source_path) as image:
                    image_size = image.size
                label_path = processed_root / "labels" / "test" / f"{source_path.stem}.txt"
                ground_truth = load_ground_truth_boxes(label_path, image_size=image_size)
                save_ground_truth_visualization(source_path, ground_truth_path, ground_truth)
                boxes = result.boxes
                predictions = [
                    {"class_id": int(class_id), "box": box, "confidence": float(confidence)}
                    for class_id, box, confidence in zip(
                        boxes.cls.detach().cpu().tolist(),
                        boxes.xyxy.detach().cpu().tolist(),
                        boxes.conf.detach().cpu().tolist(),
                    )
                ]
                analysis = analyze_image_detections(ground_truth, predictions, image_size=image_size)
                quality = image_quality(source_path)
                has_error = any(value > 0 for value in analysis["error_counts"].values())
                if has_error and quality["low_light_candidate"]:
                    low_light_errors += 1
                if has_error and quality["low_quality_candidate"]:
                    low_quality_errors += 1
                for key, value in analysis["error_counts"].items():
                    aggregate[key] += value
                for key, value in analysis["missed_by_class"].items():
                    missed_by_class[key] += value
                for key, value in analysis["false_positive_by_class"].items():
                    false_positive_by_class[key] += value
                for key, value in analysis["class_confusion_pairs"].items():
                    confusion_pairs[key] = confusion_pairs.get(key, 0) + value
                error_types = [key for key, value in analysis["error_counts"].items() if value > 0]
                if analysis["error_counts"]["dense_scene_with_errors"]:
                    error_types.append("possible_occlusion_requires_human_review")
                entry = {
                    "image_path": str(source_path),
                    "relative_path": relative.as_posix(),
                    "prediction_jpg": str(predicted_path.resolve()),
                    "ground_truth_jpg": str(ground_truth_path.resolve()),
                    "error_types": error_types,
                    **analysis,
                    "image_quality": quality,
                }
                manifest.append(
                    {
                        "relative_path": relative.as_posix(),
                        "source": str(source_path),
                        "prediction_jpg": str(predicted_path.resolve()),
                        "ground_truth_jpg": str(ground_truth_path.resolve()),
                        "ground_truth_count": len(ground_truth),
                        "prediction_count": len(predictions),
                    }
                )
                if has_error:
                    error_cases.append(entry)

            manifest_path = prediction_root / "sample_manifest.json"
            write_json_report(
                manifest_path,
                {"seed": args.seed, "count": len(manifest), "confidence_for_visualization": args.conf, "images": manifest},
            )
            relative_list_path = prediction_root / "sample_relative_paths.txt"
            relative_list_path.write_text("".join(f"{item['relative_path']}\n" for item in manifest), encoding="utf-8")
            error_cases.sort(
                key=lambda item: (
                    item["missed_by_class"]["no_helmet"],
                    item["error_counts"]["missed_detection"],
                    item["error_counts"]["class_confusion"],
                    item["error_counts"]["false_positive"],
                ),
                reverse=True,
            )
            error_payload = {
                "sample_count": len(manifest),
                "images_with_errors": len(error_cases),
                "aggregate_error_counts": aggregate,
                "missed_by_class": missed_by_class,
                "false_positive_by_class": false_positive_by_class,
                "class_confusion_pairs": confusion_pairs,
                "low_light_error_images": low_light_errors,
                "low_quality_error_images": low_quality_errors,
                "automatic_analysis_limitations": [
                    "Occlusion cannot be reliably inferred from YOLO boxes alone.",
                    "Low-light and low-quality flags are heuristic candidates for human review.",
                    "Counts use conf=0.25 for visualization review and are not official mAP statistics.",
                ],
                "cases": error_cases,
            }
            error_json_path = prediction_root / "error_analysis.json"
            write_json_report(error_json_path, error_payload)
            checklist_path = prediction_root / "human_review_checklist.csv"
            with checklist_path.open("x", encoding="utf-8-sig", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "relative_path",
                        "error_types",
                        "ground_truth_count",
                        "prediction_count",
                        "related_classes",
                        "max_iou",
                        "no_helmet_misses",
                        "prediction_jpg",
                        "ground_truth_jpg",
                        "human_notes",
                    ]
                )
                for case in error_cases:
                    writer.writerow(
                        [
                            case["relative_path"],
                            ";".join(case["error_types"]),
                            case["ground_truth_count"],
                            case["prediction_count"],
                            ";".join(case["related_classes"]),
                            case["max_iou"],
                            case["missed_by_class"]["no_helmet"],
                            case["prediction_jpg"],
                            case["ground_truth_jpg"],
                            "",
                        ]
                    )

    report: dict[str, object] = {
        "status": "passed",
        "milestone": "M4",
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "duration_seconds": time.perf_counter() - started_clock,
        "training_report": str(training_report_path),
        "protocol": {
            "validation_split": "val",
            "final_split": "test",
            "imgsz": args.imgsz,
            "batch": batch,
            "workers": args.workers,
            "device": args.device,
            "seed": args.seed,
            "plots": True,
            "test_used_for_training_model_selection_or_threshold_tuning": False,
            "visualization_confidence": args.conf,
        },
        "training": training,
        "val_evaluation": {"save_dir": str(val_save_dir), "metrics": val_payload, "plots": val_plots},
        "test_evaluation": {"save_dir": str(test_save_dir), "metrics": test_payload, "plots": test_plots},
        "predictions": {
            "save_dir": str(prediction_root.resolve()),
            "count": len(manifest),
            "manifest": str(manifest_path.resolve()),
            "relative_path_list": str(relative_list_path.resolve()),
        },
        "error_analysis": error_payload,
        "paths": {
            "best_pt": str(best_pt),
            "last_pt": str(last_pt),
            "training_log": training["console_log"],
            "val_evaluation_dir": str(val_save_dir),
            "test_evaluation_dir": str(test_save_dir),
            "prediction_jpg_dir": str(prediction_jpg_dir.resolve()),
            "ground_truth_jpg_dir": str(ground_truth_jpg_dir.resolve()),
            "prediction_manifest": str(manifest_path.resolve()),
            "error_analysis_json": str(error_json_path.resolve()),
            "human_review_checklist": str(checklist_path.resolve()),
            "evaluation_log": str(log_path.resolve()),
        },
    }
    evaluation_json_path = test_save_dir / "baseline_evaluation_report.json"
    evaluation_md_path = test_save_dir / "baseline_evaluation_report.md"
    report["paths"]["evaluation_json"] = str(evaluation_json_path.resolve())
    report["paths"]["evaluation_markdown"] = str(evaluation_md_path.resolve())
    write_json_report(evaluation_json_path, report)
    evaluation_md_path.write_text(markdown_report(report), encoding="utf-8")
    training["m4_status"] = "complete"
    training["m4_completed_at_utc"] = report["ended_at_utc"]
    training["best_weight_val_evaluation"] = report["val_evaluation"]
    training["independent_test_evaluation"] = report["test_evaluation"]
    training["prediction_visualizations"] = report["predictions"]
    training["error_analysis_summary"] = {key: value for key, value in error_payload.items() if key != "cases"}
    training["evaluation_report_markdown"] = str(evaluation_md_path.resolve())
    write_json_report(training_report_path, training, overwrite=True)
    print(json.dumps({"status": "passed", "report": str(evaluation_md_path), "test_dir": str(test_save_dir)}, indent=2))
    return report


def main() -> int:
    args = build_parser().parse_args()
    try:
        run(args)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
