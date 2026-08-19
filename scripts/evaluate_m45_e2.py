#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
import traceback
from typing import Mapping, Sequence

from PIL import Image
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from helmet_safety.training.baseline import (  # noqa: E402
    allocate_experiment_name,
    allocate_run_name,
    format_evaluation_metrics,
    load_ground_truth_boxes,
    scan_training_log,
    write_json_report,
)
from helmet_safety.training.m45 import build_val_kwargs, compare_candidate_with_m4_baseline  # noqa: E402
from helmet_safety.training.m45_e2 import (  # noqa: E402
    compare_slice_recalls,
    reported_gpu_memory,
    run_sequential_model_stages,
    streaming_image_source,
    summarize_detection_slices,
)

FIXED_CONFIDENCE = 0.25
FIXED_MATCHING_IOU = 0.5


class TeeStream:
    def __init__(self, terminal: object, log_file: object) -> None:
        self.terminal, self.log_file = terminal, log_file

    def write(self, value: str) -> int:
        self.terminal.write(value)
        self.log_file.write(value)
        return len(value)

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.terminal, "isatty", lambda: False)())

    @property
    def encoding(self) -> str:
        return getattr(self.terminal, "encoding", "utf-8") or "utf-8"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate M4.5 E2 on val only with fixed conf=0.25 and class-aware matching IoU=0.5"
    )
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--run-name", default="m45_yolo11n_e50_960_val_001")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=None, help="Defaults to the actual training batch")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def metadata_snapshot(root: Path) -> dict[str, object]:
    files = [path for path in root.rglob("*") if path.is_file()]
    return {
        "root": str(root.resolve()),
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "latest_mtime_ns": max((path.stat().st_mtime_ns for path in files), default=None),
    }


def validate_val_plots(save_dir: Path) -> dict[str, str]:
    names = {
        "confusion_matrix": "confusion_matrix.png",
        "confusion_matrix_normalized": "confusion_matrix_normalized.png",
        "pr_curve": "BoxPR_curve.png",
        "f1_curve": "BoxF1_curve.png",
        "precision_curve": "BoxP_curve.png",
        "recall_curve": "BoxR_curve.png",
    }
    outputs = {key: (save_dir / value).resolve() for key, value in names.items()}
    missing = [path for path in outputs.values() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"missing or empty val plots: {missing}")
    return {key: str(path) for key, path in outputs.items()}


def average_speeds(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    keys = sorted({key for row in rows for key in row})
    averages = {key: sum(float(row.get(key, 0.0)) for row in rows) / len(rows) for key in keys}
    inference = averages.get("inference", 0.0)
    return {
        "images": len(rows),
        "average_ms_per_image": averages,
        "average_pipeline_ms_per_image": sum(averages.values()),
        "gpu_inference_images_per_second": 1000.0 / inference if inference > 0 else None,
    }


def collect_detection_records(
    model: object, *, images_dir: Path, labels_dir: Path, imgsz: int, batch: int, device: str
) -> tuple[list[dict[str, object]], dict[str, object]]:
    images = sorted(
        path.resolve()
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    records: list[dict[str, object]] = []
    speeds: list[Mapping[str, object]] = []
    started = time.perf_counter()
    results = model.predict(
        source=streaming_image_source(images_dir),
        imgsz=imgsz,
        batch=batch,
        conf=FIXED_CONFIDENCE,
        device=device,
        save=False,
        stream=True,
        verbose=True,
    )
    for result in results:
        image_path = Path(result.path).resolve()
        with Image.open(image_path) as image:
            image_size = image.size
        ground_truth = load_ground_truth_boxes(labels_dir / f"{image_path.stem}.txt", image_size=image_size)
        xyxy = result.boxes.xyxy.detach().cpu().tolist()
        classes = result.boxes.cls.detach().cpu().tolist()
        predictions = [
            {"class_id": int(class_id), "box": [float(value) for value in box]}
            for class_id, box in zip(classes, xyxy, strict=True)
        ]
        records.append({"image_id": image_path.name, "ground_truth": ground_truth, "predictions": predictions})
        speeds.append(result.speed)
    if len(records) != len(images):
        raise RuntimeError(f"prediction result count mismatch: {len(records)} != {len(images)}")
    speed = average_speeds(speeds)
    speed["wall_seconds"] = time.perf_counter() - started
    return records, speed


def finite_metrics(metrics: Mapping[str, object]) -> bool:
    scopes = (metrics["overall"], *metrics["per_class"].values())  # type: ignore[union-attr]
    values = [float(value) for scope in scopes for key, value in scope.items() if key not in {"images", "instances"}]
    return all(math.isfinite(value) for value in values)


def change(baseline: float, candidate: float) -> dict[str, float]:
    absolute = candidate - baseline
    return {
        "m4_baseline": baseline,
        "m45_e2": candidate,
        "absolute_change": absolute,
        "relative_percent_change": absolute / baseline * 100.0 if baseline else math.nan,
    }


def metric_rows(metrics: Mapping[str, object]) -> list[str]:
    scopes = (
        ("overall", metrics["overall"]),
        ("helmet", metrics["per_class"]["helmet"]),  # type: ignore[index]
        ("no_helmet", metrics["per_class"]["no_helmet"]),  # type: ignore[index]
    )
    return [
        f"| {name} | {row['precision']:.6f} | {row['recall']:.6f} | {row['map50']:.6f} | {row['map50_95']:.6f} |"
        for name, row in scopes
    ]


def slice_rows(summary: Mapping[str, object], group: str) -> list[str]:
    rows = []
    for name, row in summary[group].items():  # type: ignore[union-attr]
        values = [row[key] if row[key] is not None else "N/A" for key in ("recall", "helmet_recall", "no_helmet_recall")]
        rows.append(
            f"| {name} | {row['images']} | {row['ground_truth_instances']} | {row['helmet_instances']} | "
            f"{row['no_helmet_instances']} | {row['tp']} | {row['fn']} | {values[0]} | {values[1]} | {values[2]} |"
        )
    return rows


def render_markdown(report: Mapping[str, object]) -> str:
    training = report["training"]
    analysis = training["training_analysis"]
    metrics = report["val_evaluation"]["metrics"]
    comparison = report["comparison"]["overall_and_class_metrics"]
    slices = report["small_target_analysis"]
    slice_comparison = report["comparison"]["slice_recalls"]
    loss, trailing = analysis["losses"], analysis["trailing_window"]
    lines = [
        "# M4.5 控制实验 E2：imgsz 960 vs M4 baseline 640",
        "",
        "## 实验协议",
        "",
        f"- milestone={report['milestone']}；experiment_id={report['experiment_id']}；baseline_run={report['baseline_run']}。",
        f"- 原始预训练权重：`{training['pretrained_model']}`；核心变量 imgsz=960；epochs=50；patience=15。",
        f"- requested batch={report['requested_batch']}；actual batch={report['actual_batch']}；完整 train 数据参与每个 epoch。",
        "- 物理 batch 仅可因 6GB 显存 OOM 降低；因此不声称与 baseline 所有参数完全相同。",
        "- 仅使用 val；test 未评估，也未用于训练、选模、调参或阈值选择。",
        "",
        "## E2 best.pt 的 val 指标",
        "",
        "| 范围 | Precision | Recall | AP50 / mAP50 | AP50-95 / mAP50-95 |",
        "|---|---:|---:|---:|---:|",
        *metric_rows(metrics),
        "",
        "## E0 与 E2 对比",
        "",
        "| 范围 | 指标 | Baseline 640 | E2 960 | 绝对变化 | 百分点变化 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for scope in ("overall", "helmet", "no_helmet"):
        for name in ("precision", "recall", "map50", "map50_95"):
            item = comparison[scope][name]
            lines.append(
                f"| {scope} | {name} | {item['m4_baseline']:.6f} | {item['m45_e2']:.6f} | "
                f"{item['absolute_change']:+.6f} | {item['percentage_point_change']:+.4f} pp |"
            )
    table_header = [
        "| 层级 | 图片 | GT | helmet GT | no_helmet GT | TP | FN | Recall | helmet R | no_helmet R |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(["", "## E2 小目标尺寸分层", "", *table_header, *slice_rows(slices["m45_e2_960"], "size_bins")])
    lines.extend(["", "## Baseline 小目标尺寸分层（相同 val 规则）", "", *table_header, *slice_rows(slices["m4_baseline_640"], "size_bins")])
    lines.extend(["", "## E2 密集场景", "", *table_header, *slice_rows(slices["m45_e2_960"], "dense_scenes"), "", "## 重点 no_helmet Recall 变化", ""])
    for group, name in (
        ("size_bins", "10_lt_equivalent_size_le_20"),
        ("size_bins", "20_lt_equivalent_size_le_30"),
        ("dense_scenes", "ground_truth_gte_10"),
        ("dense_scenes", "ground_truth_gte_20"),
    ):
        item = slice_comparison[group][name]["no_helmet_recall"]
        lines.append(f"- {name}: {item['m4_baseline_640']} → {item['m45_e2_960']}（{item['percentage_point_change']} pp）。")
    lines.extend(
        [
            "",
            "## Loss 与过拟合",
            "",
            f"- train box/cls/dfl first→best→last：{loss['train']['box_loss']}；{loss['train']['cls_loss']}；{loss['train']['dfl_loss']}。",
            f"- val box/cls/dfl first→best→last：{loss['val']['box_loss']}；{loss['val']['cls_loss']}；{loss['val']['dfl_loss']}。",
            f"- 最后 {trailing['epoch_count']} epochs：train 总 loss {trailing['train_total_loss']['change']:+.6f}；val 总 loss {trailing['val_total_loss']['change']:+.6f}；mAP50-95 {trailing['map50_95']['change']:+.6f}。",
            f"- 过拟合：{analysis['overfitting']['detected']}；best={analysis['best_epoch']}；last={analysis['last_epoch']}；early stopping={analysis['early_stopping_triggered']}。",
            "",
            "## 时间、显存与结论",
            "",
            f"- 训练时间：{report['comparison']['training_time']}。",
            f"- 推理速度：{report['comparison']['inference_speed']}。",
            f"- 日志最大显存报告值：{report['comparison']['gpu_memory']}。",
            f"- Precision/误报：{report['conclusions']['precision_false_positive_assessment']}。",
            f"- imgsz=960 值得保留：{report['conclusions']['imgsz_960_worthwhile']}；判断规则：{report['conclusions']['worthwhile_rule']}。",
            "- 遮挡仅可作为人工检查候选；本报告没有自动宣称真实遮挡。",
            "",
            "## 关键路径",
            "",
            f"- training report：`{report['paths']['training_report']}`",
            f"- best.pt：`{report['paths']['best_pt']}`",
            f"- last.pt：`{report['paths']['last_pt']}`",
            f"- training log：`{report['paths']['training_log']}`",
            f"- E2 val dir：`{report['val_evaluation']['save_dir']}`",
            f"- baseline val-only dir：`{report['baseline_val_recheck']['save_dir']}`",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, object]:
    import torch
    from ultralytics import YOLO

    started_at, started_clock = utc_now(), time.perf_counter()
    training_report_path = args.training_report.resolve()
    training = json.loads(training_report_path.read_text(encoding="utf-8"))
    expected = {"milestone": "M4.5", "experiment_id": "E2", "baseline_run": "baseline_yolo11n_001"}
    if training.get("status") != "passed" or any(training.get(key) != value for key, value in expected.items()):
        raise ValueError(f"not a passed M4.5 E2 report: {training_report_path}")
    parameters = training["training_parameters"]
    if int(parameters["epochs"]) != 50 or int(parameters["imgsz"]) != 960 or args.imgsz != 960:
        raise ValueError("E2 requires epochs=50 and imgsz=960")
    if int(training["requested_training_protocol"]["requested_batch"]) != 4:
        raise ValueError("E2 must request batch=4")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {args.device!r} unavailable")

    artifacts = args.artifacts_dir.resolve()
    best_pt, last_pt = (Path(training["outputs"][key]).resolve() for key in ("best_pt", "last_pt"))
    baseline_dir = artifacts / "training" / "baseline_yolo11n_001"
    baseline_best = baseline_dir / "weights" / "best.pt"
    baseline_report_path = baseline_dir / "baseline_training_report.json"
    baseline_training = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    if not all(path.is_file() for path in (best_pt, last_pt, baseline_best)) or baseline_training.get("status") != "passed":
        raise FileNotFoundError("E2 or baseline weights/report are incomplete")

    dataset_yaml = Path(training["data"]["dataset_yaml"]).resolve()
    config = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    processed = Path(config["path"]).resolve()
    images_dir, labels_dir = (processed / config["val"]).resolve(), (processed / "labels" / "val").resolve()
    if int(training["data"]["inventory"]["val"]["images"]) != 607:
        raise ValueError("E2 requires all 607 val images")

    raw_current = metadata_snapshot(Path(training["raw_dataset_snapshot_after"]["root"]))
    baseline_current = metadata_snapshot(baseline_dir)
    preserved_current = {name: metadata_snapshot(Path(value["root"])) for name, value in training["preserved_run_snapshots_after"].items()}
    if raw_current != training["raw_dataset_snapshot_after"] or baseline_current != training["baseline_snapshot_after"]:
        raise RuntimeError("raw VOC2028 or M4 baseline changed after training")
    if preserved_current != training["preserved_run_snapshots_after"]:
        raise RuntimeError("a preserved run changed after training")

    evaluation_dir, logs_dir = artifacts / "evaluation", artifacts / "logs"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_name = allocate_experiment_name(evaluation_dir, logs_dir, args.run_name)
    baseline_val_name = allocate_run_name(evaluation_dir, f"{run_name}_baseline_640")
    log_path = logs_dir / f"{run_name}.log"
    batch, baseline_batch = int(args.batch or training["actual_batch"]), int(baseline_training["actual_batch"])

    with log_path.open("x", encoding="utf-8", buffering=1) as log_handle:
        with redirect_stdout(TeeStream(sys.stdout, log_handle)), redirect_stderr(TeeStream(sys.stderr, log_handle)):
            print(json.dumps({"split": "val", "conf": FIXED_CONFIDENCE, "matching_iou": FIXED_MATCHING_IOU}, indent=2))
            names = {0: "helmet", 1: "no_helmet"}
            e2_val_kwargs = build_val_kwargs(data_yaml=dataset_yaml, project_dir=evaluation_dir, run_name=run_name, imgsz=960, batch=batch, workers=args.workers, device=args.device, seed=args.seed)
            baseline_val_kwargs = build_val_kwargs(data_yaml=dataset_yaml, project_dir=evaluation_dir, run_name=baseline_val_name, imgsz=640, batch=baseline_batch, workers=args.workers, device=args.device, seed=args.seed)
            baseline_val_kwargs["plots"] = False

            def load_model(path: Path) -> object:
                return YOLO(str(path), task="detect")

            def run_stage(stage_name: str, model: object) -> dict[str, object]:
                if dict(model.names) != names:
                    raise ValueError(f"{stage_name} class mapping is not 0=helmet, 1=no_helmet")
                if stage_name == "last":
                    return {"class_mapping": names}
                val_kwargs = e2_val_kwargs if stage_name == "e2" else baseline_val_kwargs
                metrics_object = model.val(**val_kwargs)
                save_dir = Path(metrics_object.save_dir).resolve()
                metrics = format_evaluation_metrics(
                    metrics_object.results_dict,
                    metrics_object.summary(normalize=True, decimals=10),
                    metrics_object.speed,
                )
                del metrics_object
                model.validator = None
                torch.cuda.empty_cache()
                stage_imgsz = 960 if stage_name == "e2" else 640
                stage_batch = batch if stage_name == "e2" else baseline_batch
                records, prediction_speed = collect_detection_records(
                    model,
                    images_dir=images_dir,
                    labels_dir=labels_dir,
                    imgsz=stage_imgsz,
                    batch=stage_batch,
                    device=args.device,
                )
                return {
                    "save_dir": save_dir,
                    "metrics": metrics,
                    "records": records,
                    "prediction_speed": prediction_speed,
                }

            stage_results = run_sequential_model_stages(
                [("last", last_pt), ("e2", best_pt), ("baseline", baseline_best)],
                loader=load_model,
                runner=run_stage,
                cleanup=torch.cuda.empty_cache,
            )

    e2_stage, baseline_stage = stage_results["e2"], stage_results["baseline"]
    e2_save_dir, baseline_save_dir = e2_stage["save_dir"], baseline_stage["save_dir"]
    e2_metrics, baseline_metrics = e2_stage["metrics"], baseline_stage["metrics"]
    e2_records, baseline_records = e2_stage["records"], baseline_stage["records"]
    e2_predict_speed, baseline_predict_speed = e2_stage["prediction_speed"], baseline_stage["prediction_speed"]

    if not finite_metrics(e2_metrics) or not finite_metrics(baseline_metrics):
        raise ValueError("val metrics must all be finite")
    plots, warning_audit = validate_val_plots(e2_save_dir), scan_training_log(log_path)
    if any(warning_audit[key] for key in ("jpeg_auto_repair_warning", "cache_version_warning", "corrupt_data_warning")):
        raise RuntimeError(f"invalidating data/cache warning found in {log_path}")

    e2_slices = summarize_detection_slices(e2_records, iou_threshold=FIXED_MATCHING_IOU)
    baseline_slices = summarize_detection_slices(baseline_records, iou_threshold=FIXED_MATCHING_IOU)
    slice_comparison = compare_slice_recalls(baseline_slices, e2_slices)
    metric_comparison = compare_candidate_with_m4_baseline(e2_metrics, candidate_key="m45_e2")
    inference_speed = {
        "inference_ms_per_image": change(float(baseline_metrics["speed_ms_per_image"]["inference"]), float(e2_metrics["speed_ms_per_image"]["inference"])),
        "gpu_images_per_second": change(float(baseline_metrics["gpu_inference_images_per_second"]), float(e2_metrics["gpu_inference_images_per_second"])),
        "baseline_batch": baseline_batch,
        "e2_batch": batch,
    }
    gpu_memory = {"baseline": reported_gpu_memory(Path(baseline_training["console_log"])), "m45_e2": reported_gpu_memory(Path(training["console_log"]))}
    key_gains = [
        slice_comparison["size_bins"]["10_lt_equivalent_size_le_20"]["no_helmet_recall"]["percentage_point_change"],
        slice_comparison["size_bins"]["20_lt_equivalent_size_le_30"]["no_helmet_recall"]["percentage_point_change"],
        slice_comparison["dense_scenes"]["ground_truth_gte_10"]["no_helmet_recall"]["percentage_point_change"],
    ]
    positive_gains = sum(value is not None and float(value) > 0 for value in key_gains)
    no_helmet_recall = metric_comparison["no_helmet"]["recall"]["percentage_point_change"]
    no_helmet_map = metric_comparison["no_helmet"]["map50_95"]["percentage_point_change"]
    precision = metric_comparison["overall"]["precision"]["percentage_point_change"]
    conclusions = {
        "imgsz_960_worthwhile": positive_gains >= 2 and no_helmet_recall > 0 and no_helmet_map > 0 and precision >= -1.0,
        "worthwhile_rule": "at least 2/3 key no_helmet slices improve, no_helmet Recall and AP50-95 improve, and overall Precision falls by less than 1 pp",
        "positive_key_slice_gains": positive_gains,
        "no_helmet_recall_improved": no_helmet_recall > 0,
        "no_helmet_map50_95_improved": no_helmet_map > 0,
        "precision_false_positive_assessment": "material false-positive risk indicated" if precision < -1.0 else "no material false-positive deterioration indicated by Precision",
        "overfitting_detected": bool(training["training_analysis"]["overfitting"]["detected"]),
        "occlusion_claimed": False,
    }
    comparison = {
        "overall_and_class_metrics": metric_comparison,
        "slice_recalls": slice_comparison,
        "training_time": change(float(baseline_training["duration_seconds"]), float(training["duration_seconds"])),
        "inference_speed": inference_speed,
        "gpu_memory": gpu_memory,
        "physical_batch": {"m4_baseline": baseline_batch, "m45_e2": batch},
    }

    report: dict[str, object] = {
        "status": "passed", **expected,
        "started_at_utc": started_at, "ended_at_utc": utc_now(), "duration_seconds": time.perf_counter() - started_clock,
        "pretrained_model": training["pretrained_model"], "imgsz": 960, "requested_batch": 4, "actual_batch": batch,
        "cuda_oom_occurred": bool(training["parameter_adjustments"]), "oom_and_batch_adjustments": training["parameter_adjustments"],
        "environment": training["environment"], "data": training["data"], "training_parameters": training["training_parameters"],
        "training_started_at_utc": training["started_at_utc"], "training_ended_at_utc": training["ended_at_utc"], "training_duration_seconds": training["duration_seconds"],
        "best_epoch": training["training_analysis"]["best_epoch"], "last_epoch": training["training_analysis"]["last_epoch"],
        "actual_epochs": training["training_analysis"]["epochs_completed"], "early_stopping": training["training_analysis"]["early_stopping_triggered"],
        "training": training,
        "val_evaluation": {"split": "val", "save_dir": str(e2_save_dir), "parameters": e2_val_kwargs, "metrics": e2_metrics, "plots": plots, "warning_audit": warning_audit},
        "baseline_val_recheck": {"split": "val", "save_dir": str(baseline_save_dir), "parameters": baseline_val_kwargs, "metrics_recomputed_for_speed_check": baseline_metrics, "primary_comparison_uses_user_supplied_m4_metrics": True},
        "small_target_analysis": {
            "split": "val", "confidence": FIXED_CONFIDENCE, "matching_iou": FIXED_MATCHING_IOU, "class_aware_matching": True,
            "nms_parameters_overridden": False, "size_basis": "original image pixels; equivalent_size=sqrt(box_width_px*box_height_px)",
            "m45_e2_960": {**e2_slices, "prediction_speed": e2_predict_speed},
            "m4_baseline_640": {**baseline_slices, "prediction_speed": baseline_predict_speed},
        },
        "comparison": comparison, "loss_analysis": training["training_analysis"], "overfitting_judgment": training["training_analysis"]["overfitting"],
        "conclusions": conclusions, "weights": {"best_bytes": best_pt.stat().st_size, "last_bytes": last_pt.stat().st_size},
        "data_integrity": {"raw_dataset_unchanged": True, "baseline_unchanged": True, "preserved_runs_unchanged": True, "current_raw_snapshot": raw_current, "current_baseline_snapshot": baseline_current, "current_preserved_run_snapshots": preserved_current},
        "test_used_for_training_or_selection": False, "test_evaluated": False,
        "paths": {"training_report": str(training_report_path), "best_pt": str(best_pt), "last_pt": str(last_pt), "training_log": training["console_log"], "evaluation_log": str(log_path.resolve())},
    }
    report_json, comparison_json, comparison_md = (e2_save_dir / name for name in ("m45_e2_val_report.json", "e2_vs_baseline.json", "e2_vs_baseline.md"))
    report["paths"].update({"evaluation_report": str(report_json.resolve()), "comparison_json": str(comparison_json.resolve()), "comparison_markdown": str(comparison_md.resolve())})  # type: ignore[union-attr]
    write_json_report(report_json, report)
    write_json_report(comparison_json, {"milestone": "M4.5", "experiment_id": "E2", "comparison": comparison, "conclusions": conclusions, "test_evaluated": False})
    comparison_md.write_text(render_markdown(report), encoding="utf-8")

    training.update({
        "requested_batch": 4, "best_epoch": report["best_epoch"], "last_epoch": report["last_epoch"], "actual_epochs": report["actual_epochs"], "early_stopping": report["early_stopping"],
        "gpu_memory": gpu_memory["m45_e2"], "val_evaluation": report["val_evaluation"], "val_overall_metrics": e2_metrics["overall"], "val_per_class_metrics": e2_metrics["per_class"],
        "small_target_analysis": report["small_target_analysis"], "baseline_comparison": comparison, "loss_analysis": report["loss_analysis"], "overfitting_judgment": report["overfitting_judgment"],
        "experiment_conclusions": conclusions, "val_evaluation_dir": str(e2_save_dir), "test_used_for_training_or_selection": False, "test_evaluated": False,
    })
    write_json_report(training_report_path, training, overwrite=True)
    print(json.dumps({"status": "passed", "val_dir": str(e2_save_dir), "report": str(report_json)}, indent=2))
    return report


def main() -> int:
    try:
        run(build_parser().parse_args())
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
