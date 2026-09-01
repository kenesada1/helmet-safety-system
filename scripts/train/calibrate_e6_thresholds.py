#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
from typing import Mapping

from PIL import Image
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from helmet_safety.training.baseline import load_ground_truth_boxes, write_json_report  # noqa: E402
from helmet_safety.training.analysis_core import validated_streaming_image_source  # noqa: E402
from helmet_safety.training.threshold_calibration import (  # noqa: E402
    compose_class_points,
    evaluate_threshold_point,
    select_operating_point,
    threshold_values,
)


E6_DIR = PROJECT_ROOT / "artifacts" / "e6" / "e6_yolo11s_p2_001"
E6_WEIGHT = E6_DIR / "weights" / "best.pt"
BASELINE_REPORT = E6_DIR / "e6_vs_e4_full_val_report.json"
DATASET_YAML = Path(r"D:\datasets\SHWD\processed\dataset.yaml")
OUTPUT_DIR = E6_DIR / "threshold_calibration"
REPORT_JSON = E6_DIR / "e6_threshold_calibration_report.json"
REPORT_MD = E6_DIR / "e6_threshold_calibration_report.md"
EXPECTED_IMAGES = 607
EXPECTED_GT = 9925
IMGSZ = 960
BATCH = 2
SEED = 42
NMS_IOU = 0.70
MATCHING_IOU = 0.50
MAX_DET = 300
HELMET_THRESHOLDS = threshold_values(0.25, 0.45, 0.01)
NO_HELMET_THRESHOLDS = threshold_values(0.15, 0.30, 0.01)
COLLECTION_CONF = min(HELMET_THRESHOLDS + NO_HELMET_THRESHOLDS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate E6 class confidence thresholds on full val only.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def collect_records(
    model: object,
    *,
    source: str,
    ground_truth: Mapping[str, list[dict[str, object]]],
    device: str,
) -> tuple[list[dict[str, object]], float]:
    records: list[dict[str, object]] = []
    started = time.perf_counter()
    results = model.predict(
        source=source,
        imgsz=IMGSZ,
        batch=BATCH,
        conf=COLLECTION_CONF,
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
            {
                "class_id": int(class_id),
                "confidence": float(confidence),
                "box": [float(value) for value in box],
            }
            for class_id, confidence, box in zip(
                result.boxes.cls.detach().cpu().tolist(),
                result.boxes.conf.detach().cpu().tolist(),
                result.boxes.xyxy.detach().cpu().tolist(),
                strict=True,
            )
        ]
        records.append(
            {"image_id": image_id, "ground_truth": ground_truth[image_id], "predictions": predictions}
        )
    if len(records) != EXPECTED_IMAGES or {str(row["image_id"]) for row in records} != set(ground_truth):
        raise RuntimeError("low-confidence prediction did not cover the complete val split")
    return records, time.perf_counter() - started


def render_markdown(report: Mapping[str, object]) -> str:
    selected = report["selected_operating_point"]  # type: ignore[assignment]
    anchors = report["anchors"]  # type: ignore[assignment]
    strict = report["strict_selection"]  # type: ignore[assignment]
    constraints = report["selection_constraints"]["practical"]  # type: ignore[index]
    rows = [anchors["E4_conf_0.25"], anchors["E6_conf_0.25"], selected]
    labels = ["E4 统一0.25", "E6 统一0.25", "E6 校准后"]
    lines = [
        "# E6 分类别置信度阈值校准",
        "",
        "- 数据：完整 val，607 张、9,925 个 GT；未使用 test。",
        f"- 扫描：helmet {HELMET_THRESHOLDS[0]:.2f}–{HELMET_THRESHOLDS[-1]:.2f}，"
        f"no_helmet {NO_HELMET_THRESHOLDS[0]:.2f}–{NO_HELMET_THRESHOLDS[-1]:.2f}，步长 0.01。",
        f"- 固定：imgsz={IMGSZ}、batch={BATCH}、seed={SEED}、NMS IoU={NMS_IOU:.2f}、"
        f"匹配 IoU={MATCHING_IOU:.2f}、max_det={MAX_DET}。",
        "- 选择目标：满足全部生产约束后，使总体 F1 最大。",
        "",
        "## 生产约束",
        "",
        f"- 严格约束结果：{strict['status']}；可行组合 {strict['feasible_points']} / {report['scan']['total_points']}。",
        "- 严格约束要求 no_helmet Recall 完全达到 E4，且 FP、极小 Recall、F1 同时达标。",
        "- 下列推荐点使用实用约束：no_helmet Recall 最多允许低于 E4 0.5 个百分点。",
        f"- 总体 FP ≤ {constraints['max_false_positives']}。",
        f"- no_helmet Recall ≥ {constraints['min_no_helmet_recall']:.6f}。",
        f"- 极小目标 Recall ≥ {constraints['min_tiny_recall']:.6f}。",
        f"- 总体 F1 ≥ {constraints['min_overall_f1']:.6f}。",
        "",
        "## 推荐工作点",
        "",
        f"- helmet conf = **{selected['thresholds']['helmet']:.2f}**",
        f"- no_helmet conf = **{selected['thresholds']['no_helmet']:.2f}**",
        f"- 可行组合：{selected['selection']['feasible_points']} / {report['scan']['total_points']}",
        "",
        "## 对比",
        "",
        "| 工作点 | helmet conf | no_helmet conf | TP | FN | FP | Precision | Recall | F1 | no_helmet Recall | tiny TP | tiny FN | tiny Recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in zip(labels, rows, strict=True):
        overall = row["overall"]
        tiny = row["tiny"]
        lines.append(
            f"| {label} | {row['thresholds']['helmet']:.2f} | {row['thresholds']['no_helmet']:.2f} | "
            f"{overall['tp']} | {overall['fn']} | {overall['fp']} | {overall['precision']:.6f} | "
            f"{overall['recall']:.6f} | {overall['f1']:.6f} | "
            f"{row['per_class']['no_helmet']['recall']:.6f} | {tiny['tp']} | {tiny['fn']} | "
            f"{tiny['recall']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 使用说明",
            "",
            f"推理时先以两个类别阈值中的较小值作为模型 conf（本次为 "
            f"{min(selected['thresholds'].values()):.2f}），完成 class-aware NMS 后，再按预测类别应用各自阈值。",
            "NMS IoU 与评价 matching IoU 不是置信度阈值，本次保持固定，不能把 matching IoU 写入生产推理配置。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    import torch
    import ultralytics
    from ultralytics import YOLO

    args = parse_args()
    for path in (E6_WEIGHT, BASELINE_REPORT, DATASET_YAML):
        if not path.is_file():
            raise FileNotFoundError(path.resolve())
    for path in (OUTPUT_DIR, REPORT_JSON, REPORT_MD):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite threshold calibration output: {path.resolve()}")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {args.device!r} unavailable")

    baseline_report = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))
    if baseline_report["protocol"]["test_used"] is not False:
        raise RuntimeError("baseline comparison is not val-only")
    e4_anchor = baseline_report["experiments"]["E4"]
    e6_anchor = baseline_report["experiments"]["E6"]

    dataset = yaml.safe_load(DATASET_YAML.read_text(encoding="utf-8"))
    root = Path(str(dataset["path"])).resolve()
    images_dir = (root / str(dataset["val"])).resolve()
    labels_dir = root / "labels" / "val"
    image_paths = sorted(
        path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(image_paths) != EXPECTED_IMAGES:
        raise RuntimeError(f"val image count mismatch: {len(image_paths)}")
    ground_truth: dict[str, list[dict[str, object]]] = {}
    for image_path in image_paths:
        with Image.open(image_path) as image:
            ground_truth[image_path.name] = load_ground_truth_boxes(
                labels_dir / f"{image_path.stem}.txt", image_size=image.size
            )
    if sum(len(row) for row in ground_truth.values()) != EXPECTED_GT:
        raise RuntimeError("val GT count mismatch")

    weight_hash_before = sha256(E6_WEIGHT)
    model = YOLO(str(E6_WEIGHT.resolve()), task="detect")
    strides = [float(value) for value in model.model.stride.detach().cpu().tolist()]
    if strides != [4.0, 8.0, 16.0, 32.0] or dict(model.names) != {0: "helmet", 1: "no_helmet"}:
        raise RuntimeError("E6 architecture or class contract failed")
    source = validated_streaming_image_source(images_dir, expected_images=EXPECTED_IMAGES)
    records, inference_seconds = collect_records(
        model, source=source, ground_truth=ground_truth, device=args.device
    )
    del model
    torch.cuda.empty_cache()

    records_by_class = {
        class_id: [
            {
                **record,
                "predictions": [
                    prediction
                    for prediction in record["predictions"]
                    if int(prediction["class_id"]) == class_id
                ],
            }
            for record in records
        ]
        for class_id in (0, 1)
    }
    helmet_curve = {
        threshold: evaluate_threshold_point(
            records_by_class[0],
            helmet_conf=threshold,
            no_helmet_conf=threshold,
            matching_iou=MATCHING_IOU,
        )
        for threshold in HELMET_THRESHOLDS
    }
    no_helmet_curve = {
        threshold: evaluate_threshold_point(
            records_by_class[1],
            helmet_conf=threshold,
            no_helmet_conf=threshold,
            matching_iou=MATCHING_IOU,
        )
        for threshold in NO_HELMET_THRESHOLDS
    }
    points = []
    for helmet_conf, no_helmet_conf in itertools.product(HELMET_THRESHOLDS, NO_HELMET_THRESHOLDS):
        helmet_point = helmet_curve[helmet_conf]
        no_helmet_point = no_helmet_curve[no_helmet_conf]
        points.append(
            compose_class_points(
                helmet_conf=helmet_conf,
                no_helmet_conf=no_helmet_conf,
                helmet=helmet_point["per_class"]["helmet"],
                no_helmet=no_helmet_point["per_class"]["no_helmet"],
                helmet_tiny={
                    "ground_truth": helmet_point["tiny"]["helmet_instances"],
                    "tp": helmet_point["tiny"]["helmet_tp"],
                    "fn": helmet_point["tiny"]["helmet_fn"],
                },
                no_helmet_tiny={
                    "ground_truth": no_helmet_point["tiny"]["no_helmet_instances"],
                    "tp": no_helmet_point["tiny"]["no_helmet_tp"],
                    "fn": no_helmet_point["tiny"]["no_helmet_fn"],
                },
                images=EXPECTED_IMAGES,
            )
        )
    uniform_025 = next(
        point
        for point in points
        if point["thresholds"] == {"helmet": 0.25, "no_helmet": 0.25}
    )
    expected_fixed = e6_anchor["fixed_threshold"]
    for scope in ("overall", "helmet", "no_helmet"):
        actual = uniform_025["overall"] if scope == "overall" else uniform_025["per_class"][scope]
        expected = expected_fixed["overall"] if scope == "overall" else expected_fixed["per_class"][scope]
        for metric in ("tp", "fn", "fp"):
            if int(actual[metric]) != int(expected[metric]):
                raise RuntimeError(f"conf=0.25 reproducibility mismatch: {scope}.{metric}")
    for metric in ("tp", "fn"):
        if int(uniform_025["tiny"][metric]) != int(e6_anchor["tiny"][metric]):
            raise RuntimeError(f"conf=0.25 tiny reproducibility mismatch: {metric}")

    strict_constraints = {
        "max_false_positives": int(e4_anchor["fixed_threshold"]["overall"]["fp"]),
        "min_no_helmet_recall": float(e4_anchor["fixed_threshold"]["per_class"]["no_helmet"]["recall"]),
        "min_tiny_recall": float(e6_anchor["tiny"]["recall"]),
        "min_overall_f1": float(e4_anchor["fixed_threshold"]["overall"]["f1"]) - 0.005,
    }
    try:
        strict_selected = select_operating_point(points, **strict_constraints)
        strict_selection = {
            "status": "feasible",
            "feasible_points": strict_selected["selection"]["feasible_points"],
            "selected": strict_selected,
        }
    except RuntimeError:
        strict_selection = {"status": "no feasible point", "feasible_points": 0, "selected": None}
    practical_constraints = {
        "max_false_positives": strict_constraints["max_false_positives"],
        "min_no_helmet_recall": strict_constraints["min_no_helmet_recall"] - 0.005,
        "min_tiny_recall": 0.64,
        "min_overall_f1": strict_constraints["min_overall_f1"],
    }
    selected = select_operating_point(points, **practical_constraints)
    if sha256(E6_WEIGHT) != weight_hash_before:
        raise RuntimeError("E6 weight changed during threshold calibration")

    e4_point = {
        "thresholds": {"helmet": 0.25, "no_helmet": 0.25},
        "overall": e4_anchor["fixed_threshold"]["overall"],
        "per_class": e4_anchor["fixed_threshold"]["per_class"],
        "tiny": e4_anchor["tiny"],
    }
    report = {
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"ultralytics": ultralytics.__version__, "torch": torch.__version__},
        "protocol": {
            "split": "val",
            "images": EXPECTED_IMAGES,
            "ground_truth": EXPECTED_GT,
            "test_used": False,
            "imgsz": IMGSZ,
            "batch": BATCH,
            "seed": SEED,
            "candidate_collection_confidence": COLLECTION_CONF,
            "nms_iou": NMS_IOU,
            "matching_iou": MATCHING_IOU,
            "class_aware_nms": True,
            "class_aware_matching": True,
            "max_det": MAX_DET,
        },
        "weight": str(E6_WEIGHT.resolve()),
        "weight_sha256": weight_hash_before,
        "detect_strides": strides,
        "scan": {
            "helmet_thresholds": HELMET_THRESHOLDS,
            "no_helmet_thresholds": NO_HELMET_THRESHOLDS,
            "total_points": len(points),
            "inference_seconds": inference_seconds,
            "candidate_predictions": sum(len(row["predictions"]) for row in records),
        },
        "selection_constraints": {
            "strict": strict_constraints,
            "practical": practical_constraints,
        },
        "strict_selection": strict_selection,
        "anchors": {"E4_conf_0.25": e4_point, "E6_conf_0.25": uniform_025},
        "selected_operating_point": selected,
        "all_points": points,
    }
    write_json_report(REPORT_JSON, report)
    REPORT_MD.write_text(render_markdown(report), encoding="utf-8")
    OUTPUT_DIR.mkdir(parents=True)
    (OUTPUT_DIR / "README.txt").write_text(
        "This directory marks the immutable E6 full-val threshold calibration run.\n",
        encoding="utf-8",
    )
    print(json.dumps(selected, ensure_ascii=False, indent=2), flush=True)
    print(REPORT_MD.read_text(encoding="utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
