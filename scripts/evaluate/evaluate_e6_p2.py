#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from helmet_safety.training.baseline import (  # noqa: E402
    format_evaluation_metrics,
    load_ground_truth_boxes,
    write_json_report,
)
from helmet_safety.training.e6_p2 import build_e6_comparison  # noqa: E402
from helmet_safety.training.analysis_core import summarize_p0_records, validated_streaming_image_source  # noqa: E402


E4_WEIGHT = PROJECT_ROOT / "artifacts" / "training" / "m45_yolo11s_e75_960_001" / "weights" / "best.pt"
E6_DIR = PROJECT_ROOT / "artifacts" / "e6" / "e6_yolo11s_p2_001"
E6_WEIGHT = E6_DIR / "weights" / "best.pt"
DATASET_YAML = Path(r"D:\datasets\SHWD\processed\dataset.yaml")
EVALUATION_DIR = E6_DIR / "evaluation"
REPORT_JSON = E6_DIR / "e6_vs_e4_full_val_report.json"
REPORT_MD = E6_DIR / "e6_vs_e4_full_val_report.md"
EXPECTED_IMAGES = 607
EXPECTED_GT = 9925
FIXED_CONF = 0.25
NMS_IOU = 0.70
MATCHING_IOU = 0.50
MAX_DET = 300


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare E4 and E6 on the complete val split; test is forbidden.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def collect_records(
    model: object,
    *,
    source: str,
    ground_truth: Mapping[str, list[dict[str, object]]],
    device: str,
    batch: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    records: list[dict[str, object]] = []
    speed_rows: list[Mapping[str, object]] = []
    started = time.perf_counter()
    results = model.predict(
        source=source,
        imgsz=960,
        batch=batch,
        conf=FIXED_CONF,
        iou=NMS_IOU,
        agnostic_nms=False,
        max_det=MAX_DET,
        device=device,
        save=False,
        stream=True,
        verbose=False,
    )
    for result in results:
        image_id = Path(result.path).name
        predictions = [
            {"class_id": int(class_id), "box": [float(value) for value in box]}
            for class_id, box in zip(
                result.boxes.cls.detach().cpu().tolist(),
                result.boxes.xyxy.detach().cpu().tolist(),
                strict=True,
            )
        ]
        records.append(
            {"image_id": image_id, "ground_truth": ground_truth[image_id], "predictions": predictions}
        )
        speed_rows.append(result.speed)
    if len(records) != EXPECTED_IMAGES or {str(row["image_id"]) for row in records} != set(ground_truth):
        raise RuntimeError("prediction did not cover the complete 607-image val split")
    speed_keys = sorted({key for row in speed_rows for key in row})
    return records, {
        "wall_seconds": time.perf_counter() - started,
        "average_ms_per_image": {
            key: sum(float(row.get(key, 0.0)) for row in speed_rows) / len(speed_rows)
            for key in speed_keys
        },
    }


def tiny_scopes(summary: Mapping[str, object]) -> dict[str, dict[str, object]]:
    return {
        "overall": {
            "ground_truth": int(summary["ground_truth_instances"]),
            "tp": int(summary["tp"]),
            "fn": int(summary["fn"]),
            "recall": float(summary["recall"]),
        },
        "helmet": {
            "ground_truth": int(summary["helmet_instances"]),
            "tp": int(summary["helmet_tp"]),
            "fn": int(summary["helmet_fn"]),
            "recall": float(summary["helmet_recall"]),
        },
        "no_helmet": {
            "ground_truth": int(summary["no_helmet_instances"]),
            "tp": int(summary["no_helmet_tp"]),
            "fn": int(summary["no_helmet_fn"]),
            "recall": float(summary["no_helmet_recall"]),
        },
    }


def render_markdown(report: Mapping[str, object]) -> str:
    experiments = report["experiments"]  # type: ignore[assignment]
    comparison = report["comparison"]  # type: ignore[assignment]
    lines = [
        "# E6 YOLO11s-P2 与 E4 完整 val 对比",
        "",
        "- 数据：原始 train 训练、完整 val（607 张、9,925 个 GT）评估；未使用 test。",
        "- 标准指标：Ultralytics val，imgsz=960、batch=2、seed=42。",
        "- 检出计数：conf=0.25、NMS IoU=0.70、class-aware matching IoU=0.50、max_det=300。",
        "- 极小目标：原图像素等效边长 sqrt(w×h) ≤10，共 35 张、128 个 GT。",
        "",
        "## 标准完整 val 指标",
        "",
        "| 实验 | 范围 | Precision | Recall | mAP50 | mAP50-95 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for experiment in ("E4", "E6"):
        metrics = experiments[experiment]["standard_val"]
        for scope in ("overall", "helmet", "no_helmet"):
            row = metrics["overall"] if scope == "overall" else metrics["per_class"][scope]
            lines.append(
                f"| {experiment} | {scope} | {row['precision']:.6f} | {row['recall']:.6f} | "
                f"{row['map50']:.6f} | {row['map50_95']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## 固定阈值：正确检出、漏检与误检",
            "",
            "| 实验 | 范围 | GT | 正确检出 TP | 漏检 FN | 误检 FP | 查准率 | 召回率 | F1 综合指标 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for experiment in ("E4", "E6"):
        fixed = experiments[experiment]["fixed_threshold"]
        for scope in ("overall", "helmet", "no_helmet"):
            row = fixed["overall"] if scope == "overall" else fixed["per_class"][scope]
            lines.append(
                f"| {experiment} | {scope} | {row['ground_truth']} | {row['tp']} | {row['fn']} | {row['fp']} | "
                f"{row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## 极小目标",
            "",
            "| 实验 | 范围 | 极小 GT | 正确检出 TP | 漏检 FN | Recall |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for experiment in ("E4", "E6"):
        for scope in ("overall", "helmet", "no_helmet"):
            row = experiments[experiment]["tiny_by_scope"][scope]
            lines.append(
                f"| {experiment} | {scope} | {row['ground_truth']} | {row['tp']} | {row['fn']} | {row['recall']:.6f} |"
            )
    deltas = comparison["deltas"]
    tiny_text = "改善" if comparison["tiny_improved"] else "未改善"
    fp_text = "造成明显误检增加" if comparison["obvious_false_positive_increase"] else "未造成明显误检增加"
    overall_text = "造成明显总体性能下降" if comparison["obvious_overall_decline"] else "未造成明显总体性能下降"
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- 极小目标：E6 {tiny_text}；Recall 变化 {deltas['tiny_recall_pp']:+.3f} 个百分点，"
            f"正确检出变化 {deltas['tiny_tp']:+d}，漏检变化 {deltas['tiny_fn']:+d}。",
            f"- 误检：E6 {fp_text}；全 val FP 变化 {deltas['false_positives']:+d} "
            f"（{deltas['false_positives_relative_percent']:+.2f}%），Precision 变化 {deltas['precision_pp']:+.3f} 个百分点。",
            f"- 总体：E6 {overall_text}；mAP50-95 变化 {deltas['map50_95_pp']:+.3f} 个百分点，"
            f"固定阈值 F1 变化 {deltas['f1_pp']:+.3f} 个百分点。",
            "- 极小目标表只对极小 GT 统计正确检出与漏检；误检不能唯一归属于某个 GT 尺寸，因此误检判断采用完整 val 的 FP 与 Precision。",
            "",
            "判定规则：" + str(comparison["rules"]),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    import torch
    import ultralytics
    from ultralytics import YOLO

    args = parse_args()
    if args.batch != 2:
        raise ValueError("E6 comparison requires batch=2")
    for path in (E4_WEIGHT, E6_WEIGHT, DATASET_YAML):
        if not path.is_file():
            raise FileNotFoundError(path.resolve())
    for path in (REPORT_JSON, REPORT_MD, EVALUATION_DIR):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite E6 evaluation output: {path.resolve()}")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {args.device!r} unavailable")

    dataset = yaml.safe_load(DATASET_YAML.read_text(encoding="utf-8"))
    root = Path(str(dataset["path"])).resolve()
    images_dir = (root / str(dataset["val"])).resolve()
    labels_dir = root / "labels" / "val"
    image_paths = sorted(
        path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(image_paths) != EXPECTED_IMAGES:
        raise RuntimeError(f"val image count mismatch: {len(image_paths)}")
    source = validated_streaming_image_source(images_dir, expected_images=EXPECTED_IMAGES)
    ground_truth: dict[str, list[dict[str, object]]] = {}
    for image_path in image_paths:
        with Image.open(image_path) as image:
            ground_truth[image_path.name] = load_ground_truth_boxes(
                labels_dir / f"{image_path.stem}.txt", image_size=image.size
            )
    if sum(len(row) for row in ground_truth.values()) != EXPECTED_GT:
        raise RuntimeError("val GT count mismatch")

    EVALUATION_DIR.mkdir(parents=True)
    experiments: dict[str, object] = {}
    e4_hash_before = sha256(E4_WEIGHT)
    for experiment, weight, expected_strides in (
        ("E4", E4_WEIGHT, [8.0, 16.0, 32.0]),
        ("E6", E6_WEIGHT, [4.0, 8.0, 16.0, 32.0]),
    ):
        model = YOLO(str(weight.resolve()), task="detect")
        strides = [float(value) for value in model.model.stride.detach().cpu().tolist()]
        if strides != expected_strides or dict(model.names) != {0: "helmet", 1: "no_helmet"}:
            raise RuntimeError(f"{experiment} architecture or class contract failed")
        val_result = model.val(
            data=str(DATASET_YAML.resolve()),
            split="val",
            imgsz=960,
            batch=2,
            workers=args.workers,
            device=args.device,
            plots=True,
            seed=42,
            deterministic=True,
            project=str(EVALUATION_DIR.resolve()),
            name=f"{experiment.lower()}_standard_val",
            exist_ok=False,
        )
        standard = format_evaluation_metrics(
            val_result.results_dict,
            val_result.summary(normalize=True, decimals=10),
            val_result.speed,
        )
        model.validator = None
        del val_result
        torch.cuda.empty_cache()
        records, prediction_speed = collect_records(
            model, source=source, ground_truth=ground_truth, device=args.device, batch=args.batch
        )
        summaries = summarize_p0_records(records, matching_iou=MATCHING_IOU, duplicate_iou=0.70)
        experiments[experiment] = {
            "weight": str(weight.resolve()),
            "weight_sha256": sha256(weight),
            "detect_strides": strides,
            "parameters": sum(parameter.numel() for parameter in model.model.parameters()),
            "standard_val": standard,
            "fixed_threshold": summaries["fixed_threshold_metrics"],
            "tiny": summaries["tiny_metrics"],
            "tiny_by_scope": tiny_scopes(summaries["tiny_metrics"]),
            "duplicate_audit": {
                key: value for key, value in summaries["duplicate_audit"].items() if key != "details"
            },
            "fixed_threshold_prediction_speed": prediction_speed,
        }
        del records, model
        torch.cuda.empty_cache()

    if sha256(E4_WEIGHT) != e4_hash_before:
        raise RuntimeError("E4 weight changed during E6 evaluation")
    comparison = build_e6_comparison(experiments["E4"], experiments["E6"])
    report = {
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"ultralytics": ultralytics.__version__, "torch": torch.__version__},
        "protocol": {
            "split": "val",
            "images": EXPECTED_IMAGES,
            "ground_truth": EXPECTED_GT,
            "imgsz": 960,
            "batch": 2,
            "seed": 42,
            "fixed_confidence": FIXED_CONF,
            "nms_iou": NMS_IOU,
            "matching_iou": MATCHING_IOU,
            "class_aware_matching": True,
            "test_used": False,
        },
        "experiments": experiments,
        "comparison": comparison,
    }
    write_json_report(REPORT_JSON, report)
    REPORT_MD.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)
    print(REPORT_MD.read_text(encoding="utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
