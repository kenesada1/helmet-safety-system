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
    collect_detection_records,
    finite_metrics,
    metadata_snapshot,
    utc_now,
    validate_val_plots,
)
from scripts.evaluate.evaluate_e3 import detection_counts  # noqa: E402
from helmet_safety.training.baseline import (  # noqa: E402
    allocate_experiment_name,
    format_evaluation_metrics,
    scan_training_log,
    write_json_report,
)
from helmet_safety.training.eval_common import build_val_kwargs  # noqa: E402
from helmet_safety.training.analysis_core import reported_gpu_memory, run_sequential_model_stages, summarize_detection_slices  # noqa: E402
from helmet_safety.training.e4_evaluation import (  # noqa: E402
    build_unified_row,
    combined_no_helmet_recall_10_30,
    e4_metric_deltas,
    validate_e4_training_contract,
)


OFFICIAL_YOLO11S_SOURCE = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate M4.5 E4 on val only (imgsz=960, conf=0.25, class-aware IoU=0.5) "
            "and generate the E0 through E4 unified comparison"
        )
    )
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--run-name", default="m45_yolo11s_e75_960_val_001")
    parser.add_argument("--batch", type=int, default=None, help="Defaults to the actual training batch")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_or_none(value: object) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite comparison value: {value}")
    return number


def _slice_fn(slices: Mapping[str, object]) -> int:
    size_bins = slices["size_bins"]
    return sum(int(row["fn"]) for row in size_bins.values())  # type: ignore[union-attr,index]


def _dense_recall(slices: Mapping[str, object], threshold: int = 10) -> float | None:
    return _finite_or_none(slices["dense_scenes"][f"ground_truth_gte_{threshold}"]["no_helmet_recall"])  # type: ignore[index]


def _training_values(training: Mapping[str, object]) -> tuple[int, bool, bool]:
    analysis = training["training_analysis"]
    return (
        int(analysis["best_epoch"]),  # type: ignore[index]
        bool(analysis["early_stopping_triggered"]),  # type: ignore[index]
        bool(analysis["overfitting"]["detected"]),  # type: ignore[index]
    )


def _gpu_memory_gib(training: Mapping[str, object]) -> float | None:
    value = reported_gpu_memory(Path(str(training["console_log"])))
    return float(value["max_reported_gib"]) if value else None


def _format_number(value: object, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def render_e4_markdown(report: Mapping[str, object]) -> str:
    metrics = report["val_evaluation"]["metrics"]  # type: ignore[index]
    slices = report["small_target_analysis"]["m45_e4_960"]  # type: ignore[index]
    conclusions = report["conclusions"]
    lines = [
        "# M4.5 组合实验 E4：YOLO11s + imgsz 960 + 75 epochs",
        "",
        "## 实验协议",
        "",
        "- 这是组合实验，模型容量、输入分辨率与训练周期同时变化，不能把结果归因于单一变量。",
        f"- 官方预训练权重：`{report['pretrained_model']}`；来源：{report['pretrained_model_source']}。",
        f"- 权重大小：{report['pretrained_model_bytes']} bytes；SHA256：`{report['pretrained_model_sha256']}`。",
        f"- requested batch={report['requested_batch']}；actual batch={report['actual_batch']}；CUDA OOM={report['cuda_oom_occurred']}。",
        "- 仅执行 val；未执行 test，未使用 test 选模、选阈值或调整参数。",
        "",
        "## E4 best.pt 的 val 指标",
        "",
        "| 范围 | Precision | Recall | AP50 / mAP50 | AP50-95 / mAP50-95 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, row in (("overall", metrics["overall"]), ("helmet", metrics["per_class"]["helmet"]), ("no_helmet", metrics["per_class"]["no_helmet"])):
        lines.append(f"| {label} | {row['precision']:.6f} | {row['recall']:.6f} | {row['map50']:.6f} | {row['map50_95']:.6f} |")
    lines.extend(["", "## 困难场景（conf=0.25，IoU=0.5，class-aware）", "", "| 场景 | 图片 | GT | TP | FP | FN | Recall | helmet Recall | no_helmet Recall |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for group in ("size_bins", "dense_scenes"):
        for name, row in slices[group].items():
            lines.append(
                f"| {name} | {row['images']} | {row['ground_truth_instances']} | {row['tp']} | {row.get('fp', 'N/A')} | {row['fn']} | "
                f"{_format_number(row['recall'])} | {_format_number(row['helmet_recall'])} | {_format_number(row['no_helmet_recall'])} |"
            )
    analysis = report["loss_analysis"]
    lines.extend([
        "", "## 训练与结论", "",
        f"- 完成 epochs={report['actual_epochs']}；best epoch={report['best_epoch']}；early stopping={report['early_stopping']}。",
        f"- 最后 {analysis['trailing_window']['epoch_count']} epochs 的 mAP50-95 变化：{analysis['trailing_window']['map50_95']['change']:+.6f}。",
        f"- 过拟合：{report['overfitting_judgment']['detected']}。",
        f"- E4 是否超过所有既有实验：{conclusions['e4_exceeds_all_quality_candidates']}。",
        f"- 高分辨率与容量收益是否表现为可叠加：{conclusions['resolution_capacity_gains_stack']}。",
        f"- 75 epochs 状态：{conclusions['epochs_75_state']}。",
        f"- Precision/FP：{conclusions['precision_and_fp_assessment']}。",
        f"- 当前最终候选：{conclusions['recommended_as_current_candidate']}。",
        f"- 切片推理建议：{conclusions['recommend_sliced_inference_test']}。",
        "", "## 关键路径", "",
    ])
    for key, path in report["paths"].items():
        lines.append(f"- {key}：`{path}`")
    lines.append("")
    return "\n".join(lines)


def render_unified_markdown(comparison: Mapping[str, object]) -> str:
    lines = [
        "# M4 / M4.5 E0～E4 统一比较",
        "",
        "> E4 同时改变模型容量、输入分辨率与训练周期。E1/E2/E3 分别提供周期、分辨率、容量的控制参照；E4 的净变化不能归因于单一变量。",
        "",
        "| 实验 | 模型 | imgsz | epochs | batch | best epoch | early stop | P | R | mAP50 | mAP50-95 | no_helmet R | no_helmet AP50-95 | 10～30 no_helmet R | dense≥10 no_helmet R | FP | FN | 参数量 | 权重 MB | 训练小时 | 推理 ms | GPU img/s | 显存 GiB | 过拟合 |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in comparison["rows"]:
        lines.append(
            f"| {row['experiment']} | {row['model']} | {row['imgsz']} | {row['epochs']} | {row['actual_batch']} | {row['best_epoch']} | {row['early_stopping']} | "
            f"{_format_number(row['overall']['precision'])} | {_format_number(row['overall']['recall'])} | {_format_number(row['overall']['map50'])} | {_format_number(row['overall']['map50_95'])} | "
            f"{_format_number(row['no_helmet']['recall'])} | {_format_number(row['no_helmet']['map50_95'])} | {_format_number(row['no_helmet_recall_10_30'])} | {_format_number(row['dense_no_helmet_recall'])} | "
            f"{_format_number(row['fp'])} | {_format_number(row['fn'])} | {row['parameters']} | {row['weight_bytes'] / 1_000_000:.3f} | {row['training_seconds'] / 3600:.3f} | "
            f"{_format_number(row['inference_ms_per_image'], 3)} | {_format_number(row['gpu_throughput_images_per_second'], 3)} | {_format_number(row['gpu_memory_gib'], 2)} | {row['overfitting']} |"
        )
    lines.extend(["", "## helmet 与 no_helmet 完整分类指标", "", "| 实验 | 类别 | Precision | Recall | AP50 | AP50-95 |", "|---|---|---:|---:|---:|---:|"])
    for row in comparison["rows"]:
        for scope in ("helmet", "no_helmet"):
            item = row[scope]
            lines.append(f"| {row['experiment']} | {scope} | {item['precision']:.6f} | {item['recall']:.6f} | {item['map50']:.6f} | {item['map50_95']:.6f} |")
    lines.extend(["", "## E4 相对 E0～E3 的变化（绝对值）", ""])
    for reference, scopes in comparison["e4_metric_deltas"].items():
        lines.extend([f"### {reference}", ""])
        for scope in ("overall", "helmet", "no_helmet"):
            values = scopes[scope]
            lines.append(f"- {scope}: P {values['precision']:+.6f}，R {values['recall']:+.6f}，AP50 {values['map50']:+.6f}，AP50-95 {values['map50_95']:+.6f}。")
        lines.append("")
    lines.extend(["## 结论", ""])
    for key, value in comparison["conclusions"].items():
        lines.append(f"- {key}: {value}")
    lines.extend([
        "", "## 数据可用性说明", "",
        "- E1 的既有 val-only 报告没有保存固定 conf=0.25 的逐图困难场景记录，因此未重跑 E1，相关切片与 FP/FN 保持 N/A。",
        "- E2 的既有报告保存了尺寸/密集切片和 FN，但未保存全局 FP；为遵守不重跑 E2，统一表中 E2 全局 FP 保持 N/A。",
        "- 速度来自各实验既有 val-only 报告，反映各自实际 imgsz/batch，适合成本对比，但不是统一 batch 的纯模型微基准。",
        "- 所有质量指标仅来自 val；test 未执行。", "",
    ])
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, object]:
    import torch
    import ultralytics
    from ultralytics import YOLO

    started_at, started_clock = utc_now(), time.perf_counter()
    training_report_path = args.training_report.resolve()
    training = _read_json(training_report_path)
    validate_e4_training_contract(training)
    if ultralytics.__version__ != "8.4.120":
        raise RuntimeError(f"Ultralytics version drift: {ultralytics.__version__}")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {args.device!r} unavailable")

    artifacts = args.artifacts_dir.resolve()
    best_pt, last_pt = (Path(training["outputs"][key]).resolve() for key in ("best_pt", "last_pt"))
    if not best_pt.is_file() or not last_pt.is_file():
        raise FileNotFoundError("E4 best.pt or last.pt is missing")
    dataset_yaml = Path(training["data"]["dataset_yaml"]).resolve()
    config = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    processed = Path(config["path"]).resolve()
    images_dir = (processed / config["val"]).resolve()
    labels_dir = (processed / "labels" / "val").resolve()
    if int(training["data"]["inventory"]["train"]["images"]) != 5457 or int(training["data"]["inventory"]["val"]["images"]) != 607:
        raise ValueError("E4 requires all 5,457 train and 607 val images")

    baseline_dir = artifacts / "training" / "baseline_yolo11n_001"
    raw_current = metadata_snapshot(Path(training["raw_dataset_snapshot_after"]["root"]))
    baseline_current = metadata_snapshot(baseline_dir)
    preserved_current = {name: metadata_snapshot(Path(value["root"])) for name, value in training["preserved_run_snapshots_after"].items()}
    if raw_current != training["raw_dataset_snapshot_after"] or baseline_current != training["baseline_snapshot_after"]:
        raise RuntimeError("raw VOC2028 or M4 baseline changed after E4 training")
    if preserved_current != training["preserved_run_snapshots_after"]:
        raise RuntimeError("an E1/E2/E3 preserved run changed after E4 training")

    evaluation_dir, logs_dir = artifacts / "evaluation", artifacts / "logs"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_name = allocate_experiment_name(evaluation_dir, logs_dir, args.run_name)
    log_path = logs_dir / f"{run_name}.log"
    batch = int(args.batch or training["actual_batch"])
    val_kwargs = build_val_kwargs(data_yaml=dataset_yaml, project_dir=evaluation_dir, run_name=run_name, imgsz=960, batch=batch, workers=args.workers, device=args.device, seed=args.seed)

    with log_path.open("x", encoding="utf-8", buffering=1) as log_handle:
        with redirect_stdout(TeeStream(sys.stdout, log_handle)), redirect_stderr(TeeStream(sys.stderr, log_handle)):
            print(json.dumps({"split": "val", "imgsz": 960, "conf": FIXED_CONFIDENCE, "matching_iou": FIXED_MATCHING_IOU, "test_evaluated": False}, indent=2))
            names = {0: "helmet", 1: "no_helmet"}

            def load_model(path: Path) -> object:
                return YOLO(str(path), task="detect")

            def stage(stage_name: str, model: object) -> dict[str, object]:
                if dict(model.names) != names:
                    raise ValueError(f"{stage_name} class mapping is not 0=helmet, 1=no_helmet")
                parameters = sum(parameter.numel() for parameter in model.model.parameters())
                if stage_name == "last":
                    return {"parameters": parameters, "class_mapping": names}
                metrics_object = model.val(**val_kwargs)
                save_dir = Path(metrics_object.save_dir).resolve()
                metrics = format_evaluation_metrics(metrics_object.results_dict, metrics_object.summary(normalize=True, decimals=10), metrics_object.speed)
                del metrics_object
                model.validator = None
                torch.cuda.empty_cache()
                records, prediction_speed = collect_detection_records(model, images_dir=images_dir, labels_dir=labels_dir, imgsz=960, batch=batch, device=args.device)
                return {"parameters": parameters, "save_dir": save_dir, "metrics": metrics, "records": records, "prediction_speed": prediction_speed}

            stages = run_sequential_model_stages([("last", last_pt), ("e4", best_pt)], loader=load_model, runner=stage, cleanup=torch.cuda.empty_cache)

    e4 = stages["e4"]
    if not finite_metrics(e4["metrics"]):
        raise ValueError("E4 val metrics must all be finite")
    plots = validate_val_plots(e4["save_dir"])
    warning_audit = scan_training_log(log_path)
    if any(warning_audit[key] for key in ("jpeg_auto_repair_warning", "cache_version_warning", "corrupt_data_warning")):
        raise RuntimeError(f"invalidating data/cache warning found in {log_path}")
    slices = summarize_detection_slices(e4["records"], iou_threshold=FIXED_MATCHING_IOU)
    counts = detection_counts(e4["records"])

    e0_training = _read_json(baseline_dir / "baseline_training_report.json")
    e1_training = _read_json(artifacts / "training" / "m45_yolo11n_e100_640_001" / "baseline_training_report.json")
    e2_training = _read_json(artifacts / "training" / "m45_yolo11n_e50_960_001" / "baseline_training_report.json")
    e3_training = _read_json(artifacts / "training" / "m45_yolo11s_e50_640_001" / "baseline_training_report.json")
    e2_report = _read_json(artifacts / "evaluation" / "m45_yolo11n_e50_960_val_003" / "m45_e2_val_report.json")
    e3_report = _read_json(artifacts / "evaluation" / "m45_yolo11s_e50_640_val_001" / "m45_e3_val_report.json")
    if any(item.get("test_evaluated") is True for item in (e1_training, e2_report, e3_report)):
        raise RuntimeError("an M4.5 source report indicates test evaluation")

    e0_metrics = e0_training["best_weight_val_evaluation"]["metrics"]
    e1_metrics = e1_training["val_evaluation"]["metrics"]
    e2_metrics = e2_report["val_evaluation"]["metrics"]
    e3_metrics = e3_report["val_evaluation"]["metrics"]
    e0_slices = e3_report["small_target_analysis"]["m4_baseline_640"]
    e2_slices = e2_report["small_target_analysis"]["m45_e2_960"]
    e3_slices = e3_report["small_target_analysis"]["m45_e3_640"]
    e0_counts = e3_report["global_detection_counts_conf025_iou05"]["m4_baseline"]
    e3_counts = e3_report["global_detection_counts_conf025_iou05"]["m45_e3"]

    def row(experiment: str, model_name: str, imgsz: int, epochs: int, train: Mapping[str, object], metrics: Mapping[str, object], difficulty: Mapping[str, object] | None, detection: Mapping[str, object] | None, parameters: int) -> dict[str, object]:
        best_epoch, early_stopping, overfitting = _training_values(train)
        return build_unified_row(
            experiment=experiment, model=model_name, imgsz=imgsz, epochs=epochs, actual_batch=int(train["actual_batch"]),
            best_epoch=best_epoch, early_stopping=early_stopping, metrics=metrics,
            no_helmet_recall_10_30=combined_no_helmet_recall_10_30(difficulty) if difficulty else None,
            dense_no_helmet_recall=_dense_recall(difficulty) if difficulty else None,
            fp=int(detection["fp"]) if detection and detection.get("fp") is not None else None,
            fn=int(detection["fn"]) if detection and detection.get("fn") is not None else (_slice_fn(difficulty) if difficulty else None),
            parameters=parameters, weight_bytes=Path(train["outputs"]["best_pt"]).stat().st_size,
            training_seconds=float(train["duration_seconds"]), inference_ms=float(metrics["speed_ms_per_image"]["inference"]),
            throughput=float(metrics["gpu_inference_images_per_second"]), gpu_memory_gib=_gpu_memory_gib(train), overfitting=overfitting,
        )

    rows = [
        row("E0", "YOLO11n", 640, 50, e0_training, e0_metrics, e0_slices, e0_counts, 2_590_230),
        row("E1", "YOLO11n", 640, 100, e1_training, e1_metrics, None, None, 2_590_230),
        row("E2", "YOLO11n", 960, 50, e2_training, e2_metrics, e2_slices, None, 2_590_230),
        row("E3", "YOLO11s", 640, 50, e3_training, e3_metrics, e3_slices, e3_counts, 9_428_566),
        row("E4", "YOLO11s", 960, 75, training, e4["metrics"], slices, counts, int(e4["parameters"])),
    ]
    deltas = e4_metric_deltas(rows)
    by_id = {item["experiment"]: item for item in rows}
    quality_keys = (("overall", "map50_95"), ("no_helmet", "recall"), ("no_helmet", "map50_95"))
    exceeds_all = all(float(by_id["E4"][scope][metric]) > max(float(by_id[eid][scope][metric]) for eid in ("E0", "E1", "E2", "E3")) for scope, metric in quality_keys)
    stacks = all(float(by_id["E4"][scope][metric]) > max(float(by_id[eid][scope][metric]) for eid in ("E2", "E3")) for scope, metric in quality_keys)
    training_analysis = training["training_analysis"]
    best_epoch = int(training_analysis["best_epoch"])
    trailing_map = float(training_analysis["trailing_window"]["map50_95"]["change"])
    epoch_state = "continuing_improvement" if best_epoch > 50 and trailing_map > 0.005 else "plateau_or_marginal_gain"
    precision_floor = min(float(deltas[f"vs_E{i}"]["overall"]["precision"]) for i in range(4))
    e4_fp = int(counts["fp"])
    comparable_fp = {"E0": int(e0_counts["fp"]), "E3": int(e3_counts["fp"])}
    precision_fp_assessment = "material deterioration" if precision_floor < -0.01 or all(e4_fp > value for value in comparable_fp.values()) else "no material Precision/FP deterioration in the directly comparable fixed-conf evidence"
    tiny_no_helmet = slices["size_bins"]["equivalent_size_le_10"]["no_helmet_recall"]
    dense20_no_helmet = slices["dense_scenes"]["ground_truth_gte_20"]["no_helmet_recall"]
    recommend_slicing = float(tiny_no_helmet or 0.0) < 0.8 or float(dense20_no_helmet or 0.0) < 0.95
    current_candidate = exceeds_all and precision_fp_assessment.startswith("no material") and not bool(training_analysis["overfitting"]["detected"])
    conclusions = {
        "e4_exceeds_all_quality_candidates": exceeds_all,
        "resolution_capacity_gains_stack": stacks,
        "epochs_75_state": epoch_state,
        "no_helmet_recall_improved_vs_all": all(float(deltas[f"vs_E{i}"]["no_helmet"]["recall"]) > 0 for i in range(4)),
        "no_helmet_map50_95_improved_vs_all": all(float(deltas[f"vs_E{i}"]["no_helmet"]["map50_95"]) > 0 for i in range(4)),
        "small_and_dense_assessment": {
            "e4_10_30_no_helmet_recall": by_id["E4"]["no_helmet_recall_10_30"],
            "best_previous_available_10_30": max(float(by_id[eid]["no_helmet_recall_10_30"]) for eid in ("E0", "E2", "E3")),
            "e4_dense_gte10_no_helmet_recall": by_id["E4"]["dense_no_helmet_recall"],
            "best_previous_available_dense_gte10": max(float(by_id[eid]["dense_no_helmet_recall"]) for eid in ("E0", "E2", "E3")),
        },
        "precision_and_fp_assessment": precision_fp_assessment,
        "speed_and_memory_cost": {
            "inference_ms_per_image": by_id["E4"]["inference_ms_per_image"],
            "gpu_throughput_images_per_second": by_id["E4"]["gpu_throughput_images_per_second"],
            "gpu_memory_gib": by_id["E4"]["gpu_memory_gib"],
            "fits_6gb_gpu": by_id["E4"]["gpu_memory_gib"] is not None and float(by_id["E4"]["gpu_memory_gib"]) < 6.0,
        },
        "recommended_as_current_candidate": current_candidate,
        "recommend_sliced_inference_test": recommend_slicing,
        "causal_caveat": "E4 is a combined experiment; E1/E2/E3 are required to discuss individual cycle/resolution/capacity contributions.",
    }
    unified = {
        "status": "passed", "scope": "val-only E0 through E4", "rows": rows, "e4_metric_deltas": deltas,
        "difficulty_details": {"E0": e0_slices, "E1": None, "E2": e2_slices, "E3": e3_slices, "E4": slices},
        "conclusions": conclusions, "test_evaluated": False,
    }

    report: dict[str, object] = {
        "status": "passed", "milestone": "M4.5", "experiment_id": "E4", "baseline_run": "baseline_yolo11n_001",
        "started_at_utc": started_at, "ended_at_utc": utc_now(), "duration_seconds": time.perf_counter() - started_clock,
        "pretrained_model": training["pretrained_model"], "pretrained_model_source": OFFICIAL_YOLO11S_SOURCE,
        "pretrained_model_sha256": training["pretrained_model_metadata"]["sha256"], "pretrained_model_bytes": training["pretrained_model_metadata"]["bytes"],
        "official_weight_provenance": {"source": OFFICIAL_YOLO11S_SOURCE, "matches_E3_official_pretrained_sha256": training["pretrained_model_metadata"]["sha256"] == e3_report["pretrained_model_sha256"], "loaded_before_training": True},
        "model_parameters": int(e4["parameters"]), "requested_batch": 2, "actual_batch": batch,
        "cuda_oom_occurred": bool(training["parameter_adjustments"]), "oom_and_batch_adjustments": training["parameter_adjustments"],
        "environment": training["environment"], "data": training["data"], "training_parameters": training["training_parameters"],
        "training_started_at_utc": training["started_at_utc"], "training_ended_at_utc": training["ended_at_utc"], "training_duration_seconds": training["duration_seconds"],
        "best_epoch": best_epoch, "last_epoch": training_analysis["last_epoch"], "actual_epochs": training_analysis["epochs_completed"], "early_stopping": training_analysis["early_stopping_triggered"],
        "val_evaluation": {"split": "val", "save_dir": str(e4["save_dir"]), "parameters": val_kwargs, "metrics": e4["metrics"], "plots": plots, "warning_audit": warning_audit},
        "small_target_analysis": {"split": "val", "confidence": FIXED_CONFIDENCE, "matching_iou": FIXED_MATCHING_IOU, "class_aware_matching": True, "nms_parameters_overridden": False, "size_basis": "original image pixels; equivalent_size=sqrt(box_width_px*box_height_px)", "m45_e4_960": {**slices, "prediction_speed": e4["prediction_speed"]}},
        "global_detection_counts_conf025_iou05": {"m45_e4": counts}, "loss_analysis": training_analysis,
        "overfitting_judgment": training_analysis["overfitting"], "conclusions": conclusions, "unified_comparison": unified,
        "weights": {"best_bytes": best_pt.stat().st_size, "last_bytes": last_pt.stat().st_size},
        "weight_load_validation": {"best_pt": True, "last_pt": True, "class_mapping": {0: "helmet", 1: "no_helmet"}},
        "data_integrity": {"raw_dataset_unchanged": True, "baseline_unchanged": True, "preserved_runs_unchanged": True, "current_raw_snapshot": raw_current, "current_baseline_snapshot": baseline_current, "current_preserved_run_snapshots": preserved_current},
        "test_used_for_training_or_selection": False, "test_evaluated": False,
        "paths": {"training_report": str(training_report_path), "best_pt": str(best_pt), "last_pt": str(last_pt), "results_csv": training["outputs"]["results_csv"], "training_curves": training["outputs"]["results_plot"], "training_confusion_matrix": training["outputs"]["confusion_matrix"], "training_log": training["console_log"], "evaluation_log": str(log_path.resolve())},
    }
    save_dir = Path(e4["save_dir"])
    report_json = save_dir / "m45_e4_val_report.json"
    e4_comparison_json = save_dir / "e4_combination_comparison.json"
    e4_comparison_md = save_dir / "e4_combination_comparison.md"
    unified_json = save_dir / "e0_e4_unified_comparison.json"
    unified_md = save_dir / "e0_e4_unified_comparison.md"
    report["paths"].update({"evaluation_report": str(report_json.resolve()), "e4_comparison_json": str(e4_comparison_json.resolve()), "e4_comparison_markdown": str(e4_comparison_md.resolve()), "unified_comparison_json": str(unified_json.resolve()), "unified_comparison_markdown": str(unified_md.resolve())})
    write_json_report(report_json, report)
    write_json_report(e4_comparison_json, {"experiment_id": "E4", "deltas": deltas, "conclusions": conclusions, "test_evaluated": False})
    write_json_report(unified_json, unified)
    e4_comparison_md.write_text(render_e4_markdown(report), encoding="utf-8")
    unified_md.write_text(render_unified_markdown(unified), encoding="utf-8")

    training.update({
        "requested_batch": 2, "best_epoch": best_epoch, "last_epoch": training_analysis["last_epoch"], "actual_epochs": training_analysis["epochs_completed"], "early_stopping": training_analysis["early_stopping_triggered"],
        "gpu_memory": reported_gpu_memory(Path(training["console_log"])), "val_evaluation": report["val_evaluation"], "val_overall_metrics": e4["metrics"]["overall"], "val_per_class_metrics": e4["metrics"]["per_class"],
        "small_target_analysis": report["small_target_analysis"], "unified_comparison": unified, "experiment_conclusions": conclusions, "val_evaluation_dir": str(save_dir),
        "test_used_for_training_or_selection": False, "test_evaluated": False,
    })
    write_json_report(training_report_path, training, overwrite=True)
    print(json.dumps({"status": "passed", "val_dir": str(save_dir), "report": str(report_json), "test_evaluated": False}, indent=2))
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
