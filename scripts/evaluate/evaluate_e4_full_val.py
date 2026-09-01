#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Mapping

from PIL import Image
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from helmet_safety.training.baseline import allocate_run_name, load_ground_truth_boxes, write_json_report  # noqa: E402
from helmet_safety.training.analysis_core import (  # noqa: E402
    EXPECTED_E4_TINY_SUMMARY,
    assert_expected_tiny_summary,
    summarize_p0_records,
    validated_streaming_image_source,
)


FIXED_WEIGHT = PROJECT_ROOT / "artifacts" / "training" / "m45_yolo11s_e75_960_001" / "weights" / "best.pt"
FIXED_EXISTING_E4_REPORT = PROJECT_ROOT / "artifacts" / "evaluation" / "m45_yolo11s_e75_960_val_001" / "m45_e4_val_report.json"
FIXED_IMGSZ = 960
FIXED_MATCHING_IOU = 0.5
FIXED_MAX_DET = 300
EXPECTED_VAL_IMAGES = 607
EXPECTED_VAL_GT = 9925
EXPECTED_BASELINE_COUNTS = {"tp": 9363, "fn": 562, "fp": 953}
CONFIGS = (
    {"config": "baseline", "confidence": 0.25, "nms_iou": 0.70},
    {"config": "A", "confidence": 0.20, "nms_iou": 0.70},
    {"config": "B", "confidence": 0.25, "nms_iou": 0.50},
    {"config": "C", "confidence": 0.20, "nms_iou": 0.50},
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "E4 full-val P0 matrix at imgsz=960 and matching IoU=0.5:\n"
            "baseline: conf=0.25/NMS=0.70\n"
            "A: conf=0.20/NMS=0.70\n"
            "B: conf=0.25/NMS=0.50\n"
            "C: conf=0.20/NMS=0.50\n"
            "Test is never used."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--output-name", default="m45_yolo11s_e75_960_full_val_p0_001")
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    import torch
    import ultralytics
    from ultralytics import YOLO

    if args.batch < 1:
        raise ValueError("batch must be at least 1")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {args.device!r} unavailable")
    for path in (FIXED_WEIGHT, FIXED_EXISTING_E4_REPORT):
        if not path.is_file():
            raise FileNotFoundError(f"fixed input is unavailable: {path.resolve()}")

    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    fixed_hashes_before = {
        "weight": sha256(FIXED_WEIGHT),
        "existing_e4_report": sha256(FIXED_EXISTING_E4_REPORT),
    }
    existing_report = json.loads(FIXED_EXISTING_E4_REPORT.read_text(encoding="utf-8"))
    existing_counts = existing_report["global_detection_counts_conf025_iou05"]["m45_e4"]
    if {key: int(existing_counts[key]) for key in EXPECTED_BASELINE_COUNTS} != EXPECTED_BASELINE_COUNTS:
        raise RuntimeError("existing E4 full-val fixed-threshold anchor no longer matches 9363 TP / 562 FN / 953 FP")

    training_args = yaml.safe_load((FIXED_WEIGHT.parents[2] / "args.yaml").read_text(encoding="utf-8"))
    dataset_yaml = Path(str(training_args["data"])).resolve()
    dataset = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    processed_root = Path(str(dataset["path"])).resolve()
    val_images_dir = (processed_root / str(dataset["val"])).resolve()
    val_labels_dir = (processed_root / "labels" / "val").resolve()
    image_paths = sorted(
        path.resolve()
        for path in val_images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(image_paths) != EXPECTED_VAL_IMAGES:
        raise RuntimeError(f"full val image count mismatch: {len(image_paths)} != {EXPECTED_VAL_IMAGES}")
    prediction_source = validated_streaming_image_source(
        val_images_dir, expected_images=EXPECTED_VAL_IMAGES
    )

    ground_truth_by_image: dict[str, list[dict[str, object]]] = {}
    for image_path in image_paths:
        with Image.open(image_path) as image:
            image_size = image.size
        ground_truth_by_image[image_path.name] = load_ground_truth_boxes(
            val_labels_dir / f"{image_path.stem}.txt", image_size=image_size
        )
    val_gt = sum(len(items) for items in ground_truth_by_image.values())
    if val_gt != EXPECTED_VAL_GT:
        raise RuntimeError(f"full val GT count mismatch: {val_gt} != {EXPECTED_VAL_GT}")

    model = YOLO(str(FIXED_WEIGHT.resolve()), task="detect")
    if dict(model.names) != {0: "helmet", 1: "no_helmet"}:
        raise ValueError(f"unexpected class mapping: {model.names}")
    model.predict(
        source=str(image_paths[0]),
        imgsz=FIXED_IMGSZ,
        conf=0.25,
        iou=0.70,
        agnostic_nms=False,
        max_det=FIXED_MAX_DET,
        batch=1,
        device=args.device,
        save=False,
        verbose=False,
    )

    evaluations: list[dict[str, object]] = []
    total_started = time.perf_counter()
    for config in CONFIGS:
        records: list[dict[str, object]] = []
        speed_rows: list[Mapping[str, object]] = []
        started = time.perf_counter()
        results = model.predict(
            source=prediction_source,
            imgsz=FIXED_IMGSZ,
            conf=float(config["confidence"]),
            iou=float(config["nms_iou"]),
            agnostic_nms=False,
            max_det=FIXED_MAX_DET,
            batch=args.batch,
            device=args.device,
            save=False,
            stream=True,
            verbose=False,
        )
        for result in results:
            image_id = Path(result.path).name
            predictions = [
                {
                    "class_id": int(class_id),
                    "box": [float(value) for value in box],
                }
                for class_id, box in zip(
                    result.boxes.cls.detach().cpu().tolist(),
                    result.boxes.xyxy.detach().cpu().tolist(),
                    strict=True,
                )
            ]
            records.append(
                {
                    "image_id": image_id,
                    "ground_truth": ground_truth_by_image[image_id],
                    "predictions": predictions,
                }
            )
            speed_rows.append(result.speed)
        wall_seconds = time.perf_counter() - started
        if len(records) != EXPECTED_VAL_IMAGES or {str(record["image_id"]) for record in records} != set(ground_truth_by_image):
            raise RuntimeError(f"config {config['config']} did not return exactly all 607 val images")

        summaries = summarize_p0_records(
            records, matching_iou=FIXED_MATCHING_IOU, duplicate_iou=0.70
        )
        fixed_metrics = summaries["fixed_threshold_metrics"]
        tiny = summaries["tiny_metrics"]
        duplicates = summaries["duplicate_audit"]
        speed_keys = sorted({key for row in speed_rows for key in row})
        average_speed = {
            key: sum(float(row.get(key, 0.0)) for row in speed_rows) / len(speed_rows)
            for key in speed_keys
        }
        evaluations.append(
            {
                **config,
                "fixed_threshold_metrics": fixed_metrics,
                "tiny_metrics": tiny,
                "duplicate_audit": {key: value for key, value in duplicates.items() if key != "details"},
                "timing": {
                    "wall_seconds": wall_seconds,
                    "wall_ms_per_image": wall_seconds * 1000.0 / EXPECTED_VAL_IMAGES,
                    "average_ms_per_image": average_speed,
                },
            }
        )
        overall = fixed_metrics["overall"]
        print(
            json.dumps(
                {
                    "config": config["config"],
                    "conf": config["confidence"],
                    "nms_iou": config["nms_iou"],
                    "tp": overall["tp"],
                    "fn": overall["fn"],
                    "fp": overall["fp"],
                    "tiny_tp": tiny["tp"],
                    "tiny_fn": tiny["fn"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    total_wall_seconds = time.perf_counter() - total_started

    baseline = evaluations[0]
    baseline_overall = baseline["fixed_threshold_metrics"]["overall"]
    actual_baseline_counts = {key: int(baseline_overall[key]) for key in EXPECTED_BASELINE_COUNTS}
    if actual_baseline_counts != EXPECTED_BASELINE_COUNTS:
        raise RuntimeError(f"full-val baseline count mismatch: {actual_baseline_counts} != {EXPECTED_BASELINE_COUNTS}")
    baseline_tiny = baseline["tiny_metrics"]
    assert_expected_tiny_summary(baseline_tiny)

    rows: list[dict[str, object]] = []
    for evaluation in evaluations:
        metrics = evaluation["fixed_threshold_metrics"]
        overall = metrics["overall"]
        helmet = metrics["per_class"]["helmet"]
        no_helmet = metrics["per_class"]["no_helmet"]
        tiny = evaluation["tiny_metrics"]
        duplicates = evaluation["duplicate_audit"]
        timing = evaluation["timing"]
        rows.append(
            {
                "config": evaluation["config"],
                "confidence": float(evaluation["confidence"]),
                "nms_iou": float(evaluation["nms_iou"]),
                "overall_tp": int(overall["tp"]),
                "overall_fn": int(overall["fn"]),
                "overall_fp": int(overall["fp"]),
                "overall_precision": float(overall["precision"]),
                "overall_recall": float(overall["recall"]),
                "overall_f1": float(overall["f1"]),
                "fp_per_image": float(overall["fp_per_image"]),
                "helmet_tp": int(helmet["tp"]),
                "helmet_fn": int(helmet["fn"]),
                "helmet_fp": int(helmet["fp"]),
                "helmet_precision": float(helmet["precision"]),
                "helmet_recall": float(helmet["recall"]),
                "helmet_f1": float(helmet["f1"]),
                "no_helmet_tp": int(no_helmet["tp"]),
                "no_helmet_fn": int(no_helmet["fn"]),
                "no_helmet_fp": int(no_helmet["fp"]),
                "no_helmet_precision": float(no_helmet["precision"]),
                "no_helmet_recall": float(no_helmet["recall"]),
                "no_helmet_f1": float(no_helmet["f1"]),
                "tiny_tp": int(tiny["tp"]),
                "tiny_fn": int(tiny["fn"]),
                "tiny_recall": float(tiny["recall"]),
                "tiny_helmet_recall": float(tiny["helmet_recall"]),
                "tiny_no_helmet_recall": float(tiny["no_helmet_recall"]),
                "obvious_duplicate_pairs": int(duplicates["duplicate_pairs"]),
                "images_with_obvious_duplicates": int(duplicates["images_with_duplicates"]),
                "inference_ms_per_image": float(timing["average_ms_per_image"].get("inference", 0.0)),
                "wall_ms_per_image": float(timing["wall_ms_per_image"]),
                "precision_delta_vs_baseline": float(overall["precision"]) - float(baseline_overall["precision"]),
                "recall_delta_vs_baseline": float(overall["recall"]) - float(baseline_overall["recall"]),
                "f1_delta_vs_baseline": float(overall["f1"]) - float(baseline_overall["f1"]),
                "fp_delta_vs_baseline": int(overall["fp"]) - int(baseline_overall["fp"]),
                "tiny_recall_delta_vs_baseline": float(tiny["recall"]) - float(baseline_tiny["recall"]),
            }
        )

    by_name = {str(row["config"]): row for row in rows}
    candidate_a, candidate_b, candidate_c = by_name["A"], by_name["B"], by_name["C"]
    analysis_lines = [
        (
            f"候选 A 相对基线：Recall {candidate_a['recall_delta_vs_baseline']:+.6f}，Precision "
            f"{candidate_a['precision_delta_vs_baseline']:+.6f}，FP {int(candidate_a['fp_delta_vs_baseline']):+d}，"
            f"tiny Recall {candidate_a['tiny_recall_delta_vs_baseline']:+.6f}。"
        ),
        (
            f"候选 B 相对基线：Recall {candidate_b['recall_delta_vs_baseline']:+.6f}，Precision "
            f"{candidate_b['precision_delta_vs_baseline']:+.6f}，FP {int(candidate_b['fp_delta_vs_baseline']):+d}，"
            f"tiny Recall {candidate_b['tiny_recall_delta_vs_baseline']:+.6f}。"
        ),
        (
            f"候选 C 相对基线：Recall {candidate_c['recall_delta_vs_baseline']:+.6f}，Precision "
            f"{candidate_c['precision_delta_vs_baseline']:+.6f}，FP {int(candidate_c['fp_delta_vs_baseline']):+d}，"
            f"tiny Recall {candidate_c['tiny_recall_delta_vs_baseline']:+.6f}。"
        ),
    ]
    if int(candidate_c["overall_tp"]) >= int(candidate_a["overall_tp"]) and int(candidate_c["overall_fp"]) <= int(candidate_a["overall_fp"]):
        analysis_lines.append("候选 C 在 TP 不低于 A 的同时 FP 不高于 A，因此在固定阈值检测计数上支配候选 A。")
    if int(candidate_b["overall_tp"]) >= int(rows[0]["overall_tp"]) and int(candidate_b["overall_fp"]) <= int(rows[0]["overall_fp"]):
        analysis_lines.append("候选 B 在 TP 不低于基线的同时 FP 不高于基线，因此在固定阈值检测计数上支配基线。")
    best_f1 = max(rows, key=lambda row: float(row["overall_f1"]))
    analysis_lines.append(f"四配置中固定阈值 overall F1 最高的是 {best_f1['config']}：{float(best_f1['overall_f1']):.6f}。")
    analysis = "\n".join(analysis_lines)

    fixed_hashes_after = {
        "weight": sha256(FIXED_WEIGHT),
        "existing_e4_report": sha256(FIXED_EXISTING_E4_REPORT),
    }
    if fixed_hashes_after != fixed_hashes_before:
        raise RuntimeError("a fixed E4 input changed during P0 evaluation")

    evaluation_root = PROJECT_ROOT / "artifacts" / "evaluation"
    run_name = allocate_run_name(evaluation_root, args.output_name)
    output_dir = evaluation_root / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = output_dir / "e4_full_val_p0_comparison.csv"
    markdown_path = output_dir / "e4_full_val_p0_comparison.md"
    details_path = output_dir / "e4_full_val_p0_details.json"
    report_path = output_dir / "e4_full_val_p0_report.json"
    with csv_path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    markdown_lines = [
        "# E4 完整 val P0 四配置验证",
        "",
        "- 数据：完整 607 张 val，9,925 个 GT；未使用 test。",
        "- 固定：E4 best.pt、imgsz=960、matching IoU=0.5、class-aware NMS、max_det=300。",
        "- 指标：固定 conf 下对全部 GT 做 class-aware 一对一匹配；不是跨阈值 AP。",
        "",
        "## Overall",
        "",
        "| 配置 | conf | NMS | TP | FN | FP | Precision | Recall | F1 | FP/图 | tiny Recall | 重复框对 | 推理 ms/图 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown_lines.append(
            f"| {row['config']} | {float(row['confidence']):.2f} | {float(row['nms_iou']):.2f} | "
            f"{row['overall_tp']} | {row['overall_fn']} | {row['overall_fp']} | "
            f"{float(row['overall_precision']):.6f} | {float(row['overall_recall']):.6f} | "
            f"{float(row['overall_f1']):.6f} | {float(row['fp_per_image']):.3f} | "
            f"{float(row['tiny_recall']):.6f} | {row['obvious_duplicate_pairs']} | "
            f"{float(row['inference_ms_per_image']):.3f} |"
        )
    markdown_lines.extend(
        [
            "",
            "## 分类别",
            "",
            "| 配置 | helmet P | helmet R | helmet F1 | no_helmet P | no_helmet R | no_helmet F1 | tiny helmet R | tiny no_helmet R |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        markdown_lines.append(
            f"| {row['config']} | {float(row['helmet_precision']):.6f} | {float(row['helmet_recall']):.6f} | "
            f"{float(row['helmet_f1']):.6f} | {float(row['no_helmet_precision']):.6f} | "
            f"{float(row['no_helmet_recall']):.6f} | {float(row['no_helmet_f1']):.6f} | "
            f"{float(row['tiny_helmet_recall']):.6f} | {float(row['tiny_no_helmet_recall']):.6f} |"
        )
    markdown_lines.extend(["", "## 判断", "", *[f"- {line}" for line in analysis_lines], ""])
    markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")
    write_json_report(details_path, {"rows": rows, "evaluations": evaluations})

    report = {
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "weight": str(FIXED_WEIGHT.resolve()),
            "weight_sha256": fixed_hashes_before["weight"],
            "dataset_yaml": str(dataset_yaml),
            "split": "val",
            "test_evaluated": False,
            "images": EXPECTED_VAL_IMAGES,
            "ground_truth": EXPECTED_VAL_GT,
            "imgsz": FIXED_IMGSZ,
            "matching_iou": FIXED_MATCHING_IOU,
            "class_aware_matching": True,
            "class_aware_nms": True,
            "agnostic_nms": False,
            "max_det": FIXED_MAX_DET,
            "configs": list(CONFIGS),
            "only_variables": ["confidence", "nms_iou"],
            "batch": args.batch,
            "device": args.device,
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "rows": rows,
        "analysis": analysis,
        "total_wall_seconds": total_wall_seconds,
        "artifacts": {
            "csv": str(csv_path.resolve()),
            "markdown": str(markdown_path.resolve()),
            "details": str(details_path.resolve()),
            "report": str(report_path.resolve()),
        },
    }
    write_json_report(report_path, report)
    return report


def main() -> int:
    try:
        report = run(build_parser().parse_args())
    except Exception as exc:
        print(json.dumps({"status": "failed", "reason": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": report["status"],
                "rows": report["rows"],
                "analysis": report["analysis"],
                "artifacts": report["artifacts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
