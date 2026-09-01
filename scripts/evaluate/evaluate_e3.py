#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
import math
from pathlib import Path
import sys
import time
import traceback
from typing import Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate.evaluate_e2 import (  # noqa: E402
    FIXED_CONFIDENCE,
    FIXED_MATCHING_IOU,
    TeeStream,
    average_speeds,
    collect_detection_records,
    finite_metrics,
    metadata_snapshot,
    utc_now,
    validate_val_plots,
)
from helmet_safety.training.baseline import (  # noqa: E402
    allocate_experiment_name,
    allocate_run_name,
    format_evaluation_metrics,
    scan_training_log,
    write_json_report,
)
from helmet_safety.training.eval_common import build_val_kwargs, compare_candidate_with_m4_baseline  # noqa: E402
from helmet_safety.training.analysis_core import (  # noqa: E402
    class_aware_matches,
    compare_slice_recalls,
    reported_gpu_memory,
    run_sequential_model_stages,
    streaming_image_source,
    summarize_detection_slices,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate M4.5 E3 on val only at imgsz=640 with fixed conf=0.25, "
            "class-aware matching IoU=0.5, and a fair batch=1 speed benchmark"
        )
    )
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--run-name", default="m45_yolo11s_e50_640_val_001")
    parser.add_argument("--batch", type=int, default=None, help="Defaults to the actual training batch")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def benchmark_speed(model: object, *, images_dir: Path, device: str) -> dict[str, object]:
    images = sorted(
        path.resolve()
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise ValueError(f"no benchmark images in {images_dir}")
    warmup_count = 3
    for _ in range(warmup_count):
        model.predict(source=str(images[0]), imgsz=640, batch=1, conf=FIXED_CONFIDENCE, device=device, save=False, verbose=False)
    rows: list[Mapping[str, object]] = []
    started = time.perf_counter()
    for result in model.predict(
        source=streaming_image_source(images_dir),
        imgsz=640,
        batch=1,
        conf=FIXED_CONFIDENCE,
        device=device,
        save=False,
        stream=True,
        verbose=False,
    ):
        rows.append(result.speed)
    wall_seconds = time.perf_counter() - started
    if len(rows) != len(images):
        raise RuntimeError(f"benchmark result count mismatch: {len(rows)} != {len(images)}")
    result = average_speeds(rows)
    result.update(
        {
            "imgsz": 640,
            "batch": 1,
            "confidence": FIXED_CONFIDENCE,
            "warmup_images": warmup_count,
            "measurement_repeats": 1,
            "model_load_time_excluded": True,
            "same_full_val_images": True,
            "wall_seconds": wall_seconds,
            "wall_ms_per_image": wall_seconds * 1000.0 / len(images),
            "wall_images_per_second": len(images) / wall_seconds,
        }
    )
    return result


def detection_counts(records: list[dict[str, object]]) -> dict[str, int]:
    tp = fn = fp = 0
    for record in records:
        ground_truth = record["ground_truth"]
        predictions = record["predictions"]
        matched = class_aware_matches(ground_truth, predictions, iou_threshold=FIXED_MATCHING_IOU)
        tp += len(matched)
        fn += len(ground_truth) - len(matched)
        fp += len(predictions) - len(matched)
    return {"tp": tp, "fn": fn, "fp": fp}


def numeric_change(baseline: float, candidate: float) -> dict[str, float]:
    delta = candidate - baseline
    return {
        "m4_baseline": baseline,
        "m45_e3": candidate,
        "absolute_change": delta,
        "relative_percent_change": delta / baseline * 100.0 if baseline else math.nan,
    }


def render_markdown(report: Mapping[str, object]) -> str:
    metrics = report["val_evaluation"]["metrics"]
    comparison = report["comparison"]["overall_and_class_metrics"]
    slices = report["small_target_analysis"]
    slice_comparison = report["comparison"]["slice_recalls"]
    analysis = report["loss_analysis"]
    lines = [
        "# M4.5 控制实验 E3：YOLO11s vs YOLO11n baseline",
        "",
        "## 实验协议",
        "",
        "- 唯一核心变量：YOLO11n → YOLO11s；imgsz=640、epochs=50，其余核心参数不变。",
        f"- 预训练权重：`{report['pretrained_model']}`；SHA256=`{report['pretrained_model_sha256']}`。",
        f"- 实际 batch={report['actual_batch']}；CUDA OOM={report['cuda_oom_occurred']}；test_evaluated=False。",
        "",
        "## E3 val 指标",
        "",
        "| 范围 | Precision | Recall | AP50 / mAP50 | AP50-95 / mAP50-95 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in (("overall", metrics["overall"]), ("helmet", metrics["per_class"]["helmet"]), ("no_helmet", metrics["per_class"]["no_helmet"])):
        lines.append(f"| {name} | {row['precision']:.6f} | {row['recall']:.6f} | {row['map50']:.6f} | {row['map50_95']:.6f} |")
    lines.extend(["", "## E0 与 E3 指标变化", "", "| 范围 | 指标 | E0 | E3 | 变化 | 百分点 |", "|---|---|---:|---:|---:|---:|"])
    for scope in ("overall", "helmet", "no_helmet"):
        for metric in ("precision", "recall", "map50", "map50_95"):
            item = comparison[scope][metric]
            lines.append(f"| {scope} | {metric} | {item['m4_baseline']:.6f} | {item['m45_e3']:.6f} | {item['absolute_change']:+.6f} | {item['percentage_point_change']:+.4f} |")
    lines.extend(["", "## 困难场景", ""])
    for group, label in (("size_bins", "尺寸分层"), ("dense_scenes", "密集场景")):
        lines.extend([f"### {label}", "", "| 层级 | 图片 | GT | helmet GT | no_helmet GT | TP | FP | FN | Recall | helmet R | no_helmet R |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for key, row in slices["m45_e3_640"][group].items():
            lines.append(
                f"| {key} | {row['images']} | {row['ground_truth_instances']} | {row['helmet_instances']} | {row['no_helmet_instances']} | "
                f"{row['tp']} | {row.get('fp', 'N/A')} | {row['fn']} | {row['recall']} | {row['helmet_recall']} | {row['no_helmet_recall']} |"
            )
    lines.extend(["", "## 重点 no_helmet Recall 变化", ""])
    for group, key in (("size_bins", "10_lt_equivalent_size_le_20"), ("size_bins", "20_lt_equivalent_size_le_30"), ("dense_scenes", "ground_truth_gte_10"), ("dense_scenes", "ground_truth_gte_20")):
        item = slice_comparison[group][key]["no_helmet_recall"]
        lines.append(f"- {key}: {item['m4_baseline_640']} → {item['m45_e3_640']}（{item['percentage_point_change']} pp）。")
    lines.extend(
        [
            "",
            "## Loss、成本与判断",
            "",
            f"- best epoch={report['best_epoch']}，last epoch={report['last_epoch']}，early stopping={report['early_stopping']}。",
            f"- 最后 {analysis['trailing_window']['epoch_count']} epochs：train total loss 变化 {analysis['trailing_window']['train_total_loss']['change']:+.6f}，val total loss 变化 {analysis['trailing_window']['val_total_loss']['change']:+.6f}，mAP50-95 变化 {analysis['trailing_window']['map50_95']['change']:+.6f}。",
            f"- 过拟合判断：{report['overfitting_judgment']}。",
            f"- 公平 batch=1 速度比较：{report['comparison']['fair_speed_batch1']}。",
            f"- 参数量/权重/训练时间/显存：{report['comparison']['capacity_and_cost']}。",
            f"- 最终结论：{report['conclusions']}。",
            "- 遮挡仅标记为人工检查候选，本实验不自动宣称遮挡检测效果。",
            "",
            "## 关键路径",
            "",
            f"- training report：`{report['paths']['training_report']}`",
            f"- best.pt：`{report['paths']['best_pt']}`",
            f"- last.pt：`{report['paths']['last_pt']}`",
            f"- E3 val：`{report['val_evaluation']['save_dir']}`",
            f"- baseline val-only：`{report['baseline_val_recheck']['save_dir']}`",
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
    expected = {"milestone": "M4.5", "experiment_id": "E3", "baseline_run": "baseline_yolo11n_001"}
    if training.get("status") != "passed" or any(training.get(key) != value for key, value in expected.items()):
        raise ValueError(f"not a passed M4.5 E3 report: {training_report_path}")
    parameters = training["training_parameters"]
    if int(parameters["epochs"]) != 50 or int(parameters["imgsz"]) != 640:
        raise ValueError("E3 requires epochs=50 and imgsz=640")
    if int(training["requested_training_protocol"]["requested_batch"]) != 8:
        raise ValueError("E3 must initially request batch=8")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {args.device!r} unavailable")

    artifacts = args.artifacts_dir.resolve()
    best_pt, last_pt = (Path(training["outputs"][key]).resolve() for key in ("best_pt", "last_pt"))
    baseline_dir = artifacts / "training" / "baseline_yolo11n_001"
    baseline_best = baseline_dir / "weights" / "best.pt"
    baseline_training = json.loads((baseline_dir / "baseline_training_report.json").read_text(encoding="utf-8"))
    if not all(path.is_file() for path in (best_pt, last_pt, baseline_best)) or baseline_training.get("status") != "passed":
        raise FileNotFoundError("E3 or baseline weights/report are incomplete")

    dataset_yaml = Path(training["data"]["dataset_yaml"]).resolve()
    config = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    processed = Path(config["path"]).resolve()
    images_dir, labels_dir = (processed / config["val"]).resolve(), (processed / "labels" / "val").resolve()
    if int(training["data"]["inventory"]["train"]["images"]) != 5457 or int(training["data"]["inventory"]["val"]["images"]) != 607:
        raise ValueError("E3 requires all 5,457 train and 607 val images")

    raw_current = metadata_snapshot(Path(training["raw_dataset_snapshot_after"]["root"]))
    baseline_current = metadata_snapshot(baseline_dir)
    preserved_current = {name: metadata_snapshot(Path(value["root"])) for name, value in training["preserved_run_snapshots_after"].items()}
    if raw_current != training["raw_dataset_snapshot_after"] or baseline_current != training["baseline_snapshot_after"]:
        raise RuntimeError("raw VOC2028 or M4 baseline changed after training")
    if preserved_current != training["preserved_run_snapshots_after"]:
        raise RuntimeError("an E1/E2 preserved run changed after training")

    evaluation_dir, logs_dir = artifacts / "evaluation", artifacts / "logs"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_name = allocate_experiment_name(evaluation_dir, logs_dir, args.run_name)
    baseline_val_name = allocate_run_name(evaluation_dir, f"{run_name}_baseline_640")
    log_path = logs_dir / f"{run_name}.log"
    batch = int(args.batch or training["actual_batch"])

    with log_path.open("x", encoding="utf-8", buffering=1) as log_handle:
        with redirect_stdout(TeeStream(sys.stdout, log_handle)), redirect_stderr(TeeStream(sys.stderr, log_handle)):
            print(json.dumps({"split": "val", "imgsz": 640, "conf": FIXED_CONFIDENCE, "matching_iou": FIXED_MATCHING_IOU, "fair_speed_batch": 1}, indent=2))
            names = {0: "helmet", 1: "no_helmet"}
            e3_val_kwargs = build_val_kwargs(data_yaml=dataset_yaml, project_dir=evaluation_dir, run_name=run_name, imgsz=640, batch=batch, workers=args.workers, device=args.device, seed=args.seed)
            baseline_val_kwargs = build_val_kwargs(data_yaml=dataset_yaml, project_dir=evaluation_dir, run_name=baseline_val_name, imgsz=640, batch=1, workers=args.workers, device=args.device, seed=args.seed)
            baseline_val_kwargs["plots"] = False

            def load_model(path: Path) -> object:
                return YOLO(str(path), task="detect")

            def stage(stage_name: str, model: object) -> dict[str, object]:
                if dict(model.names) != names:
                    raise ValueError(f"{stage_name} class mapping is not 0=helmet, 1=no_helmet")
                parameters_count = sum(parameter.numel() for parameter in model.model.parameters())
                if stage_name == "last":
                    return {"class_mapping": names, "parameters": parameters_count}
                val_kwargs = e3_val_kwargs if stage_name == "e3" else baseline_val_kwargs
                metrics_object = model.val(**val_kwargs)
                save_dir = Path(metrics_object.save_dir).resolve()
                metrics = format_evaluation_metrics(metrics_object.results_dict, metrics_object.summary(normalize=True, decimals=10), metrics_object.speed)
                del metrics_object
                model.validator = None
                torch.cuda.empty_cache()
                records, prediction_speed = collect_detection_records(model, images_dir=images_dir, labels_dir=labels_dir, imgsz=640, batch=batch if stage_name == "e3" else 1, device=args.device)
                fair_speed = benchmark_speed(model, images_dir=images_dir, device=args.device)
                return {"save_dir": save_dir, "metrics": metrics, "records": records, "prediction_speed": prediction_speed, "fair_speed": fair_speed, "parameters": parameters_count}

            stages = run_sequential_model_stages(
                [("last", last_pt), ("e3", best_pt), ("baseline", baseline_best)],
                loader=load_model,
                runner=stage,
                cleanup=torch.cuda.empty_cache,
            )

    e3, baseline = stages["e3"], stages["baseline"]
    if not finite_metrics(e3["metrics"]) or not finite_metrics(baseline["metrics"]):
        raise ValueError("val metrics must all be finite")
    plots, warning_audit = validate_val_plots(e3["save_dir"]), scan_training_log(log_path)
    if any(warning_audit[key] for key in ("jpeg_auto_repair_warning", "cache_version_warning", "corrupt_data_warning")):
        raise RuntimeError(f"invalidating data/cache warning found in {log_path}")

    e3_slices = summarize_detection_slices(e3["records"], iou_threshold=FIXED_MATCHING_IOU)
    baseline_slices = summarize_detection_slices(baseline["records"], iou_threshold=FIXED_MATCHING_IOU)
    slice_comparison = compare_slice_recalls(baseline_slices, e3_slices, candidate_key="m45_e3_640")
    metric_comparison = compare_candidate_with_m4_baseline(e3["metrics"], candidate_key="m45_e3")
    e3_counts, baseline_counts = detection_counts(e3["records"]), detection_counts(baseline["records"])
    e3_fair, baseline_fair = e3["fair_speed"], baseline["fair_speed"]
    speed_comparison = {
        name: numeric_change(float(baseline_fair["average_ms_per_image"][name]), float(e3_fair["average_ms_per_image"][name]))
        for name in ("preprocess", "inference", "postprocess")
    }
    speed_comparison.update(
        {
            "pipeline_ms_per_image": numeric_change(float(baseline_fair["average_pipeline_ms_per_image"]), float(e3_fair["average_pipeline_ms_per_image"])),
            "images_per_second": numeric_change(float(baseline_fair["gpu_inference_images_per_second"]), float(e3_fair["gpu_inference_images_per_second"])),
            "conditions": {"imgsz": 640, "batch": 1, "warmup_images": 3, "measurement_repeats": 1, "images": 607, "model_load_time_excluded": True},
            "m4_baseline": baseline_fair,
            "m45_e3": e3_fair,
        }
    )
    capacity_cost = {
        "parameters": numeric_change(float(baseline["parameters"]), float(e3["parameters"])),
        "best_weight_bytes": numeric_change(float(baseline_best.stat().st_size), float(best_pt.stat().st_size)),
        "training_seconds": numeric_change(float(baseline_training["duration_seconds"]), float(training["duration_seconds"])),
        "gpu_memory": {"m4_baseline": reported_gpu_memory(Path(baseline_training["console_log"])), "m45_e3": reported_gpu_memory(Path(training["console_log"]))},
        "physical_batch": {"m4_baseline": baseline_training["actual_batch"], "m45_e3": batch},
    }
    key_slice_changes = [
        slice_comparison["size_bins"][key]["no_helmet_recall"]["percentage_point_change"]
        for key in ("10_lt_equivalent_size_le_20", "20_lt_equivalent_size_le_30")
    ]
    dense_changes = [
        slice_comparison["dense_scenes"][key]["no_helmet_recall"]["percentage_point_change"]
        for key in ("ground_truth_gte_10", "ground_truth_gte_20")
    ]
    precision_change = metric_comparison["overall"]["precision"]["percentage_point_change"]
    conclusions = {
        "overall_recall_improved": metric_comparison["overall"]["recall"]["absolute_change"] > 0,
        "no_helmet_recall_improved": metric_comparison["no_helmet"]["recall"]["absolute_change"] > 0,
        "no_helmet_map50_95_improved": metric_comparison["no_helmet"]["map50_95"]["absolute_change"] > 0,
        "small_10_to_30_no_helmet_improved": all(value is not None and value > 0 for value in key_slice_changes),
        "dense_no_helmet_improved": all(value is not None and value > 0 for value in dense_changes),
        "precision_materially_worse": precision_change < -1.0,
        "false_positives_increased": e3_counts["fp"] > baseline_counts["fp"],
        "overfitting_detected": bool(training["training_analysis"]["overfitting"]["detected"]),
        "e3_clearly_better_rule": "overall/no_helmet Recall and no_helmet AP50-95 improve, at least 3/4 difficult slices improve, and Precision loss is under 1 pp",
    }
    difficult_improvements = sum(value is not None and value > 0 for value in [*key_slice_changes, *dense_changes])
    conclusions["e3_clearly_better_than_m4"] = (
        conclusions["overall_recall_improved"]
        and conclusions["no_helmet_recall_improved"]
        and conclusions["no_helmet_map50_95_improved"]
        and difficult_improvements >= 3
        and not conclusions["precision_materially_worse"]
    )

    report: dict[str, object] = {
        "status": "passed", **expected,
        "started_at_utc": started_at, "ended_at_utc": utc_now(), "duration_seconds": time.perf_counter() - started_clock,
        "pretrained_model": training["pretrained_model"],
        "pretrained_model_source": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s.pt",
        "pretrained_model_sha256": training["pretrained_model_metadata"]["sha256"],
        "pretrained_model_bytes": training["pretrained_model_metadata"]["bytes"],
        "model_parameters": int(e3["parameters"]), "requested_batch": 8, "actual_batch": batch,
        "cuda_oom_occurred": bool(training["parameter_adjustments"]), "oom_and_batch_adjustments": training["parameter_adjustments"],
        "environment": training["environment"], "data": training["data"], "training_parameters": training["training_parameters"],
        "training_started_at_utc": training["started_at_utc"], "training_ended_at_utc": training["ended_at_utc"], "training_duration_seconds": training["duration_seconds"],
        "best_epoch": training["training_analysis"]["best_epoch"], "last_epoch": training["training_analysis"]["last_epoch"],
        "actual_epochs": training["training_analysis"]["epochs_completed"], "early_stopping": training["training_analysis"]["early_stopping_triggered"],
        "training": training,
        "val_evaluation": {"split": "val", "save_dir": str(e3["save_dir"]), "parameters": e3_val_kwargs, "metrics": e3["metrics"], "plots": plots, "warning_audit": warning_audit},
        "baseline_val_recheck": {"split": "val", "save_dir": str(baseline["save_dir"]), "parameters": baseline_val_kwargs, "metrics_recomputed_for_speed_check": baseline["metrics"], "primary_comparison_uses_user_supplied_m4_metrics": True},
        "small_target_analysis": {"split": "val", "confidence": FIXED_CONFIDENCE, "matching_iou": FIXED_MATCHING_IOU, "class_aware_matching": True, "nms_parameters_overridden": False, "size_basis": "original image pixels; equivalent_size=sqrt(box_width_px*box_height_px)", "m45_e3_640": {**e3_slices, "prediction_speed": e3["prediction_speed"]}, "m4_baseline_640": {**baseline_slices, "prediction_speed": baseline["prediction_speed"]}},
        "global_detection_counts_conf025_iou05": {"m45_e3": e3_counts, "m4_baseline": baseline_counts},
        "comparison": {"overall_and_class_metrics": metric_comparison, "slice_recalls": slice_comparison, "fair_speed_batch1": speed_comparison, "capacity_and_cost": capacity_cost, "global_detection_counts": {"m4_baseline": baseline_counts, "m45_e3": e3_counts}},
        "loss_analysis": training["training_analysis"], "overfitting_judgment": training["training_analysis"]["overfitting"],
        "conclusions": conclusions,
        "weights": {"best_bytes": best_pt.stat().st_size, "last_bytes": last_pt.stat().st_size, "baseline_best_bytes": baseline_best.stat().st_size},
        "data_integrity": {"raw_dataset_unchanged": True, "baseline_unchanged": True, "preserved_runs_unchanged": True, "current_raw_snapshot": raw_current, "current_baseline_snapshot": baseline_current, "current_preserved_run_snapshots": preserved_current},
        "test_used_for_training_or_selection": False, "test_evaluated": False,
        "paths": {"training_report": str(training_report_path), "best_pt": str(best_pt), "last_pt": str(last_pt), "training_log": training["console_log"], "evaluation_log": str(log_path.resolve())},
    }
    report_json, comparison_json, comparison_md = (Path(e3["save_dir"]) / name for name in ("m45_e3_val_report.json", "e3_vs_baseline.json", "e3_vs_baseline.md"))
    report["paths"].update({"evaluation_report": str(report_json.resolve()), "comparison_json": str(comparison_json.resolve()), "comparison_markdown": str(comparison_md.resolve())})
    write_json_report(report_json, report)
    write_json_report(comparison_json, {"milestone": "M4.5", "experiment_id": "E3", "comparison": report["comparison"], "conclusions": conclusions, "test_evaluated": False})
    comparison_md.write_text(render_markdown(report), encoding="utf-8")
    training.update({"model_parameters": int(e3["parameters"]), "pretrained_model_source": report["pretrained_model_source"], "requested_batch": 8, "best_epoch": report["best_epoch"], "last_epoch": report["last_epoch"], "actual_epochs": report["actual_epochs"], "early_stopping": report["early_stopping"], "gpu_memory": capacity_cost["gpu_memory"]["m45_e3"], "val_evaluation": report["val_evaluation"], "val_overall_metrics": e3["metrics"]["overall"], "val_per_class_metrics": e3["metrics"]["per_class"], "small_target_analysis": report["small_target_analysis"], "baseline_comparison": report["comparison"], "loss_analysis": report["loss_analysis"], "overfitting_judgment": report["overfitting_judgment"], "experiment_conclusions": conclusions, "val_evaluation_dir": str(e3["save_dir"]), "test_used_for_training_or_selection": False, "test_evaluated": False})
    write_json_report(training_report_path, training, overwrite=True)
    print(json.dumps({"status": "passed", "val_dir": str(e3["save_dir"]), "report": str(report_json)}, indent=2))
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
