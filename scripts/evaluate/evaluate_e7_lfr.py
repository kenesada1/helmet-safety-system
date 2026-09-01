#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

from PIL import Image
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from helmet_safety.training.baseline import (  # noqa: E402
    format_evaluation_metrics,
    load_ground_truth_boxes,
    write_json_report,
)
from helmet_safety.training.e7_lfr import build_e7_comparison, register_lfr_module  # noqa: E402
from helmet_safety.training.analysis_core import summarize_p0_records, validated_streaming_image_source  # noqa: E402


E4_WEIGHT = PROJECT_ROOT / "artifacts" / "training" / "m45_yolo11s_e75_960_001" / "weights" / "best.pt"
E6_WEIGHT = PROJECT_ROOT / "artifacts" / "e6" / "e6_yolo11s_p2_001" / "weights" / "best.pt"
E7_DIR = PROJECT_ROOT / "artifacts" / "e7" / "e7_yolo11s_p2_lfr_001"
E7_WEIGHT = E7_DIR / "weights" / "best.pt"
DATASET_YAML = Path(r"D:\datasets\SHWD\processed\dataset.yaml")
EVALUATION_DIR = E7_DIR / "evaluation"
REPORT_JSON = E7_DIR / "e4_e6_e7_full_val_comparison.json"
REPORT_MD = E7_DIR / "e4_e6_e7_full_val_comparison_zh.md"
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


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def greedy_pairs(
    ground_truth: Sequence[Mapping[str, object]],
    predictions: Sequence[Mapping[str, object]],
    *,
    same_class: bool,
    excluded_gt: set[int] | None = None,
    excluded_pred: set[int] | None = None,
) -> list[tuple[int, int]]:
    excluded_gt = excluded_gt or set()
    excluded_pred = excluded_pred or set()
    candidates: list[tuple[float, int, int]] = []
    for gt_index, gt in enumerate(ground_truth):
        if gt_index in excluded_gt:
            continue
        for pred_index, prediction in enumerate(predictions):
            if pred_index in excluded_pred:
                continue
            classes_equal = int(gt["class_id"]) == int(prediction["class_id"])
            if classes_equal != same_class:
                continue
            overlap = box_iou(gt["box"], prediction["box"])  # type: ignore[arg-type]
            if overlap >= MATCHING_IOU:
                candidates.append((overlap, gt_index, pred_index))
    matches: list[tuple[int, int]] = []
    used_gt, used_pred = set(excluded_gt), set(excluded_pred)
    for _overlap, gt_index, pred_index in sorted(candidates, reverse=True):
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        matches.append((gt_index, pred_index))
    return matches


def count_class_confusions(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    total = 0
    pairs = {"helmet_as_no_helmet": 0, "no_helmet_as_helmet": 0}
    for record in records:
        ground_truth = record["ground_truth"]  # type: ignore[assignment]
        predictions = record["predictions"]  # type: ignore[assignment]
        correct = greedy_pairs(ground_truth, predictions, same_class=True)
        used_gt = {item[0] for item in correct}
        used_pred = {item[1] for item in correct}
        confused = greedy_pairs(
            ground_truth,
            predictions,
            same_class=False,
            excluded_gt=used_gt,
            excluded_pred=used_pred,
        )
        total += len(confused)
        for gt_index, _pred_index in confused:
            key = "helmet_as_no_helmet" if int(ground_truth[gt_index]["class_id"]) == 0 else "no_helmet_as_helmet"
            pairs[key] += 1
    return {"count": total, "pairs": pairs}


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
        records.append({"image_id": image_id, "ground_truth": ground_truth[image_id], "predictions": predictions})
        speed_rows.append(result.speed)
    if len(records) != EXPECTED_IMAGES or {str(row["image_id"]) for row in records} != set(ground_truth):
        raise RuntimeError("prediction did not cover the complete 607-image val split")
    keys = sorted({key for row in speed_rows for key in row})
    averages = {key: sum(float(row.get(key, 0.0)) for row in speed_rows) / len(speed_rows) for key in keys}
    return records, {
        "wall_seconds": time.perf_counter() - started,
        "average_ms_per_image": averages,
        "inference_fps": 1000.0 / averages["inference"] if averages.get("inference", 0.0) > 0 else None,
    }


def add_standard_f1(metrics: dict[str, object]) -> None:
    for row in [metrics["overall"], *metrics["per_class"].values()]:  # type: ignore[union-attr]
        precision, recall = float(row["precision"]), float(row["recall"])
        row["f1"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def render_markdown(report: Mapping[str, object]) -> str:
    experiments = report["experiments"]  # type: ignore[assignment]
    comparison = report["e7_vs_e6"]  # type: ignore[assignment]
    lines = [
        "# E7 YOLO11s-P2 轻量特征精炼实验：完整 val 中文对比报告",
        "",
        "- 训练：仅原始 train 5,457 张；验证：完整 val 607 张、9,925 个 GT；test 未参与训练、调参或选模。",
        "- 统一评估：imgsz=960、batch=2、seed=42、deterministic=True。",
        "- TP/FN/FP：conf=0.25、NMS IoU=0.70、class-aware matching IoU=0.50、max_det=300。",
        "- 极小目标：原图等效边长 sqrt(w×h) ≤ 10 像素，共 128 个 GT。",
        "",
        "## 总体与分类标准指标",
        "",
        "| 实验 | 范围 | Precision | Recall | F1 | AP50 | AP50-95 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for experiment in ("E4", "E6", "E7"):
        standard = experiments[experiment]["standard_val"]
        for scope in ("overall", "helmet", "no_helmet"):
            row = standard["overall"] if scope == "overall" else standard["per_class"][scope]
            lines.append(
                f"| {experiment} | {scope} | {row['precision']:.6f} | {row['recall']:.6f} | "
                f"{row['f1']:.6f} | {row['map50']:.6f} | {row['map50_95']:.6f} |"
            )
    lines.extend(["", "## 固定阈值检出计数", "", "| 实验 | 范围 | GT | TP | FN | FP | Precision | Recall | F1 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for experiment in ("E4", "E6", "E7"):
        fixed = experiments[experiment]["fixed_threshold"]
        for scope in ("overall", "helmet", "no_helmet"):
            row = fixed["overall"] if scope == "overall" else fixed["per_class"][scope]
            lines.append(
                f"| {experiment} | {scope} | {row['ground_truth']} | {row['tp']} | {row['fn']} | {row['fp']} | "
                f"{row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} |"
            )
    lines.extend(["", "## 极小目标", "", "| 实验 | 范围 | 极小 GT | TP | FN | Recall |", "|---|---|---:|---:|---:|---:|"])
    for experiment in ("E4", "E6", "E7"):
        for scope in ("overall", "helmet", "no_helmet"):
            row = experiments[experiment]["tiny_by_scope"][scope]
            lines.append(f"| {experiment} | {scope} | {row['ground_truth']} | {row['tp']} | {row['fn']} | {row['recall']:.6f} |")
    lines.extend(["", "## 规模、计算量、速度与类别混淆", "", "| 实验 | 参数量 | GFLOPs@960 | 标准 val 推理 ms/图 | 固定阈值推理 ms/图 | 固定阈值 FPS | 类别混淆 |", "|---|---:|---:|---:|---:|---:|---:|"])
    for experiment in ("E4", "E6", "E7"):
        row = experiments[experiment]
        std_ms = row["standard_val"]["speed_ms_per_image"]["inference"]
        pred_ms = row["fixed_threshold_prediction_speed"]["average_ms_per_image"]["inference"]
        fps = row["fixed_threshold_prediction_speed"]["inference_fps"]
        lines.append(f"| {experiment} | {row['parameters']:,} | {row['gflops_at_960']:.3f} | {std_ms:.3f} | {pred_ms:.3f} | {fps:.2f} | {row['class_confusions']} |")
    delta = comparison["deltas"]
    yes_no = lambda value: "是" if value else "否"
    lines.extend(
        [
            "",
            "## 最终判断（E7 相对 E6）",
            "",
            f"1. **是否减少极小目标漏检：{yes_no(comparison['tiny_false_negatives_reduced'])}。** 极小 TP {delta['tiny_tp']:+d}、FN {delta['tiny_fn']:+d}、Recall {delta['tiny_recall_pp']:+.3f} 个百分点。",
            f"2. **是否改善总体检测能力：{yes_no(comparison['overall_detection_improved'])}。** mAP50-95 {delta['map50_95_pp']:+.3f} 个百分点，标准 F1 {delta['standard_f1_pp']:+.3f} 个百分点。",
            f"3. **是否增加误检或类别混淆：误检 {yes_no(comparison['false_positives_increased'])}，类别混淆 {yes_no(comparison['class_confusion_increased'])}。** FP {delta['false_positives']:+d}（{delta['false_positives_relative_percent']:+.2f}%），类别混淆 {delta['class_confusions']:+d}（{delta['class_confusions_relative_percent']:+.2f}%）。",
            f"4. **计算开销是否合理：{yes_no(comparison['compute_overhead_reasonable'])}。** 参数 {delta['parameters']:+,d}（{delta['parameters_relative_percent']:+.3f}%），GFLOPs {delta['gflops_at_960']:+.3f}（{delta['gflops_relative_percent']:+.3f}%）。",
            f"5. **P2 轻量特征精炼是否有效：{yes_no(comparison['p2_lfr_effective'])}。** 判定采用报告内固定规则，兼顾极小漏检、总体指标、错误增幅和计算开销。",
            "",
            f"判定规则：`{comparison['rules']}`",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate E4/E6/E7 on complete val only; test is forbidden.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    args = parse_args()
    if args.batch != 2:
        raise ValueError("unified E4/E6/E7 evaluation requires batch=2")
    for path in (E4_WEIGHT, E6_WEIGHT, E7_WEIGHT, DATASET_YAML):
        if not path.is_file():
            raise FileNotFoundError(path.resolve())
    for path in (REPORT_JSON, REPORT_MD, EVALUATION_DIR):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite E7 evaluation output: {path.resolve()}")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {args.device!r} unavailable")
    dataset = yaml.safe_load(DATASET_YAML.read_text(encoding="utf-8"))
    if dataset.get("train") != "images/train" or dataset.get("val") != "images/val":
        raise RuntimeError("dataset split contract changed")
    root = Path(str(dataset["path"])).resolve()
    images_dir, labels_dir = root / "images" / "val", root / "labels" / "val"
    image_paths = sorted(path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(image_paths) != EXPECTED_IMAGES:
        raise RuntimeError(f"val image count mismatch: {len(image_paths)}")
    source = validated_streaming_image_source(images_dir, expected_images=EXPECTED_IMAGES)
    ground_truth: dict[str, list[dict[str, object]]] = {}
    for image_path in image_paths:
        with Image.open(image_path) as image:
            ground_truth[image_path.name] = load_ground_truth_boxes(labels_dir / f"{image_path.stem}.txt", image_size=image.size)
    if sum(len(row) for row in ground_truth.values()) != EXPECTED_GT:
        raise RuntimeError("val GT count mismatch")

    EVALUATION_DIR.mkdir(parents=True)
    register_lfr_module()
    weights = {"E4": E4_WEIGHT, "E6": E6_WEIGHT, "E7": E7_WEIGHT}
    hashes_before = {name: sha256(path) for name, path in weights.items()}
    experiments: dict[str, object] = {}
    expected_strides = {"E4": [8.0, 16.0, 32.0], "E6": [4.0, 8.0, 16.0, 32.0], "E7": [4.0, 8.0, 16.0, 32.0]}
    for experiment in ("E4", "E6", "E7"):
        weight = weights[experiment]
        model = YOLO(str(weight.resolve()), task="detect")
        strides = [float(value) for value in model.model.stride.detach().cpu().tolist()]
        if strides != expected_strides[experiment] or dict(model.names) != {0: "helmet", 1: "no_helmet"}:
            raise RuntimeError(f"{experiment} architecture or class contract failed")
        val_result = model.val(
            data=str(DATASET_YAML.resolve()), split="val", imgsz=960, batch=2, workers=args.workers,
            device=args.device, plots=True, seed=42, deterministic=True,
            project=str(EVALUATION_DIR.resolve()), name=f"{experiment.lower()}_standard_val", exist_ok=False,
        )
        standard = format_evaluation_metrics(val_result.results_dict, val_result.summary(normalize=True, decimals=10), val_result.speed)
        add_standard_f1(standard)
        model.validator = None
        del val_result
        torch.cuda.empty_cache()
        records, prediction_speed = collect_records(model, source=source, ground_truth=ground_truth, device=args.device, batch=args.batch)
        summaries = summarize_p0_records(records, matching_iou=MATCHING_IOU, duplicate_iou=0.70)
        confusion = count_class_confusions(records)
        experiments[experiment] = {
            "weight": str(weight.resolve()), "weight_sha256": hashes_before[experiment], "detect_strides": strides,
            "parameters": sum(parameter.numel() for parameter in model.model.parameters()),
            "gflops_at_960": float(get_flops(model.model, imgsz=960)),
            "standard_val": standard, "fixed_threshold": summaries["fixed_threshold_metrics"],
            "tiny": summaries["tiny_metrics"], "tiny_by_scope": tiny_scopes(summaries["tiny_metrics"]),
            "class_confusions": confusion["count"], "class_confusion_pairs": confusion["pairs"],
            "duplicate_audit": {key: value for key, value in summaries["duplicate_audit"].items() if key != "details"},
            "fixed_threshold_prediction_speed": prediction_speed,
        }
        del records, model
        torch.cuda.empty_cache()
    hashes_after = {name: sha256(path) for name, path in weights.items()}
    if hashes_after != hashes_before:
        raise RuntimeError("an evaluated weight changed during evaluation")
    comparison = build_e7_comparison(experiments["E6"], experiments["E7"])
    report = {
        "status": "passed", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"ultralytics": ultralytics.__version__, "torch": torch.__version__},
        "protocol": {
            "split": "val", "images": EXPECTED_IMAGES, "ground_truth": EXPECTED_GT, "imgsz": 960,
            "batch": 2, "seed": 42, "deterministic": True, "fixed_confidence": FIXED_CONF,
            "nms_iou": NMS_IOU, "matching_iou": MATCHING_IOU, "class_aware_matching": True,
            "tiny_definition": "sqrt(original_box_width_px * original_box_height_px) <= 10", "test_used": False,
        },
        "experiments": experiments, "e7_vs_e6": comparison,
        "weight_integrity": {"before": hashes_before, "after": hashes_after, "unchanged": True},
    }
    write_json_report(REPORT_JSON, report)
    REPORT_MD.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)
    print(REPORT_MD.read_text(encoding="utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
