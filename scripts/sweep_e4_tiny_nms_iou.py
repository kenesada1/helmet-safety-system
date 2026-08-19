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

from PIL import Image
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from helmet_safety.training.baseline import allocate_run_name, load_ground_truth_boxes, write_json_report  # noqa: E402
from helmet_safety.training.m45_e2 import (  # noqa: E402
    EXPECTED_E4_TINY_SUMMARY,
    assert_expected_tiny_summary,
    audit_obvious_duplicate_boxes,
    evaluate_tiny_conf_records,
    render_tiny_nms_iou_markdown,
)


FIXED_WEIGHT = PROJECT_ROOT / "artifacts" / "training" / "m45_yolo11s_e75_960_001" / "weights" / "best.pt"
FIXED_SOURCE_DIR = PROJECT_ROOT / "artifacts" / "evaluation" / "m45_yolo11s_e75_960_tiny_val_001"
FIXED_MANIFEST = FIXED_SOURCE_DIR / "tiny_val_images.json"
FIXED_FALSE_NEGATIVES = FIXED_SOURCE_DIR / "tiny_val_false_negatives.json"
FIXED_NMS_IOUS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
ANCHOR_NMS_IOU = 0.70
FIXED_IMGSZ = 960
FIXED_CONFIDENCE = 0.25
FIXED_MATCHING_IOU = 0.5
FIXED_TINY_MAX_SIZE = 10.0
DUPLICATE_IOU_THRESHOLD = 0.70
EXPECTED_IMAGES = 35
EXPECTED_ORIGINAL_FN = 51


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "E4 tiny val-only class-aware NMS IoU sweep with fixed imgsz=960, conf=0.25, matching "
            "IoU=0.5. The only variable is NMS IoU: 0.50, 0.60, 0.70, 0.80, 0.90, 0.95"
        )
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--output-name", default="m45_yolo11s_e75_960_tiny_nms_iou_sweep_001")
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.cfg import DEFAULT_CFG

    if args.batch < 1:
        raise ValueError("batch must be at least 1")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {args.device!r} unavailable")
    for path in (FIXED_WEIGHT, FIXED_MANIFEST, FIXED_FALSE_NEGATIVES):
        if not path.is_file():
            raise FileNotFoundError(f"fixed input is unavailable: {path.resolve()}")

    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    frozen_hashes_before = {
        "manifest": sha256(FIXED_MANIFEST),
        "false_negatives": sha256(FIXED_FALSE_NEGATIVES),
    }
    manifest_document = json.loads(FIXED_MANIFEST.read_text(encoding="utf-8"))
    fn_document = json.loads(FIXED_FALSE_NEGATIVES.read_text(encoding="utf-8"))
    manifest = manifest_document["images"]
    original_false_negatives = fn_document["false_negatives"]
    image_ids = [str(item["image_id"]) for item in manifest]
    original_fn_keys = {
        (str(item["image_id"]), int(item["gt_index"])) for item in original_false_negatives
    }
    if (
        len(manifest) != EXPECTED_IMAGES
        or len(set(image_ids)) != EXPECTED_IMAGES
        or int(manifest_document["tiny_gt_count"]) != int(EXPECTED_E4_TINY_SUMMARY["ground_truth_instances"])
    ):
        raise RuntimeError("the frozen first-experiment manifest is not 35 unique val images with 128 tiny GT")
    if len(original_false_negatives) != EXPECTED_ORIGINAL_FN or len(original_fn_keys) != EXPECTED_ORIGINAL_FN:
        raise RuntimeError("the frozen first-experiment FN set is not 51 unique image_id/gt_index keys")

    training_args = yaml.safe_load((FIXED_WEIGHT.parents[1] / "args.yaml").read_text(encoding="utf-8"))
    dataset_yaml = Path(str(training_args["data"])).resolve()
    dataset = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    processed_root = Path(str(dataset["path"])).resolve()
    val_images_dir = (processed_root / str(dataset["val"])).resolve()
    val_labels_dir = (processed_root / "labels" / "val").resolve()
    image_paths = [Path(str(item["image_path"])).resolve() for item in manifest]
    if any(path.parent != val_images_dir or not path.is_file() for path in image_paths):
        raise RuntimeError("a frozen manifest image is missing or is not from the configured val split")

    ground_truth_by_image: dict[str, list[dict[str, object]]] = {}
    for image_path in image_paths:
        with Image.open(image_path) as image:
            image_size = image.size
        ground_truth_by_image[image_path.name] = load_ground_truth_boxes(
            val_labels_dir / f"{image_path.stem}.txt", image_size=image_size
        )
    for item in original_false_negatives:
        image_id, gt_index = str(item["image_id"]), int(item["gt_index"])
        ground_truth = ground_truth_by_image.get(image_id)
        if ground_truth is None or not 0 <= gt_index < len(ground_truth):
            raise RuntimeError(f"frozen FN key no longer resolves in val labels: {image_id}:{gt_index}")
        gt = ground_truth[gt_index]
        if int(gt["class_id"]) != int(item["class_id"]) or [float(v) for v in gt["box"]] != [
            float(v) for v in item["gt_box_xyxy"]
        ]:
            raise RuntimeError(f"frozen FN content no longer matches val labels: {image_id}:{gt_index}")

    model = YOLO(str(FIXED_WEIGHT.resolve()), task="detect")
    if dict(model.names) != {0: "helmet", 1: "no_helmet"}:
        raise ValueError(f"unexpected class mapping: {model.names}")

    evaluations: list[dict[str, object]] = []
    total_started = time.perf_counter()
    fixed_max_det = int(DEFAULT_CFG.max_det)
    for nms_iou in FIXED_NMS_IOUS:
        records: list[dict[str, object]] = []
        started = time.perf_counter()
        results = model.predict(
            source=[str(path) for path in image_paths],
            imgsz=FIXED_IMGSZ,
            conf=FIXED_CONFIDENCE,
            iou=nms_iou,
            agnostic_nms=False,
            max_det=fixed_max_det,
            batch=args.batch,
            device=args.device,
            save=False,
            stream=True,
            verbose=False,
        )
        prediction_count = 0
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
            prediction_count += len(predictions)
            records.append(
                {
                    "image_id": image_id,
                    "ground_truth": ground_truth_by_image[image_id],
                    "predictions": predictions,
                }
            )
        if len(records) != EXPECTED_IMAGES or {str(row["image_id"]) for row in records} != set(image_ids):
            raise RuntimeError(f"NMS IoU={nms_iou:.2f} did not return exactly the frozen 35 images")
        evaluation = evaluate_tiny_conf_records(
            records,
            original_false_negatives,
            max_equivalent_size=FIXED_TINY_MAX_SIZE,
            iou_threshold=FIXED_MATCHING_IOU,
        )
        duplicates = audit_obvious_duplicate_boxes(
            records, duplicate_iou_threshold=DUPLICATE_IOU_THRESHOLD
        )
        evaluations.append(
            {
                "nms_iou": nms_iou,
                "wall_seconds": time.perf_counter() - started,
                "prediction_count": prediction_count,
                **evaluation,
                "duplicate_audit": duplicates,
            }
        )
        print(
            json.dumps(
                {
                    "nms_iou": nms_iou,
                    "tiny_tp": evaluation["summary"]["tp"],
                    "tiny_fn": evaluation["summary"]["fn"],
                    "fp": evaluation["false_positives"],
                    "recovered_original_fn": evaluation["recovered_original_fn_count"],
                    "duplicate_pairs": duplicates["duplicate_pairs"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    total_wall_seconds = time.perf_counter() - total_started

    anchor = next(item for item in evaluations if float(item["nms_iou"]) == ANCHOR_NMS_IOU)
    assert_expected_tiny_summary(anchor["summary"])
    anchor_current_fn_keys = {
        (str(item["image_id"]), int(item["gt_index"])) for item in anchor["false_negatives"]
    }
    if anchor_current_fn_keys != original_fn_keys:
        raise RuntimeError("NMS IoU=0.70 did not reproduce the exact frozen set of 51 original FN keys")

    rows: list[dict[str, object]] = []
    anchor_fp = int(anchor["false_positives"])
    anchor_tp = int(anchor["summary"]["tp"])
    for evaluation in evaluations:
        summary = evaluation["summary"]
        duplicates = evaluation["duplicate_audit"]
        recovered = evaluation["recovered_original_false_negatives"]
        fp = int(evaluation["false_positives"])
        recovered_keys = "; ".join(
            f"{item['image_id']}:{item['gt_index']}:{item['class_name']}" for item in recovered
        )
        rows.append(
            {
                "nms_iou": float(evaluation["nms_iou"]),
                "tiny_tp": int(summary["tp"]),
                "tiny_fn": int(summary["fn"]),
                "tiny_recall": float(summary["recall"]),
                "helmet_recall": float(summary["helmet_recall"]),
                "no_helmet_recall": float(summary["no_helmet_recall"]),
                "fp_35_images": fp,
                "fp_delta_vs_070": fp - anchor_fp,
                "tiny_tp_delta_vs_070": int(summary["tp"]) - anchor_tp,
                "recovered_original_fn_count": int(evaluation["recovered_original_fn_count"]),
                "recovered_helmet_count": int(evaluation["recovered_helmet_count"]),
                "recovered_no_helmet_count": int(evaluation["recovered_no_helmet_count"]),
                "obvious_duplicate_pairs": int(duplicates["duplicate_pairs"]),
                "images_with_obvious_duplicates": int(duplicates["images_with_duplicates"]),
                "predictions_in_duplicate_pairs": int(duplicates["predictions_in_duplicate_pairs"]),
                "has_obvious_duplicates": bool(duplicates["has_obvious_duplicates"]),
                "recovered_original_fn_keys": recovered_keys,
            }
        )

    higher_rows = [row for row in rows if float(row["nms_iou"]) > ANCHOR_NMS_IOU]
    best_higher = max(
        higher_rows,
        key=lambda row: (
            int(row["tiny_tp"]),
            -int(row["fp_35_images"]),
            -int(row["obvious_duplicate_pairs"]),
        ),
    )
    first_gain = next((row for row in higher_rows if int(row["tiny_tp_delta_vs_070"]) > 0), None)
    first_duplicate = next((row for row in higher_rows if bool(row["has_obvious_duplicates"])), None)
    analysis_parts = []
    if first_gain is None:
        analysis_parts.append("提高 NMS IoU 没有增加 tiny TP，也没有减少这批密集困难图上的 tiny 漏检。")
    else:
        analysis_parts.append(
            f"从 NMS IoU={float(first_gain['nms_iou']):.2f} 开始减少 tiny 漏检：相对 0.70，tiny TP "
            f"增加 {first_gain['tiny_tp_delta_vs_070']}，原 51 个 FN 救回 {first_gain['recovered_original_fn_count']} 个。"
        )
    analysis_parts.append(
        f"高于 0.70 的档位中，tiny TP 最高的是 NMS IoU={float(best_higher['nms_iou']):.2f}："
        f"相对锚点净增 {best_higher['tiny_tp_delta_vs_070']}，FP 变化 {int(best_higher['fp_delta_vs_070']):+d}，"
        f"明显重复框对={best_higher['obvious_duplicate_pairs']}。"
    )
    if first_duplicate is None:
        analysis_parts.append("所有高 NMS IoU 档位均未出现符合定义的明显重复框。")
    else:
        analysis_parts.append(
            f"从 NMS IoU={float(first_duplicate['nms_iou']):.2f} 开始出现明显重复框："
            f"{first_duplicate['obvious_duplicate_pairs']} 对，涉及 {first_duplicate['images_with_obvious_duplicates']} 张图。"
        )
    analysis = "\n\n".join(analysis_parts)

    frozen_hashes_after = {
        "manifest": sha256(FIXED_MANIFEST),
        "false_negatives": sha256(FIXED_FALSE_NEGATIVES),
    }
    if frozen_hashes_after != frozen_hashes_before:
        raise RuntimeError("a frozen first-experiment input changed during the NMS IoU sweep")

    evaluation_root = PROJECT_ROOT / "artifacts" / "evaluation"
    run_name = allocate_run_name(evaluation_root, args.output_name)
    output_dir = evaluation_root / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = output_dir / "e4_tiny_nms_iou_comparison.csv"
    markdown_path = output_dir / "e4_tiny_nms_iou_comparison.md"
    details_path = output_dir / "e4_tiny_nms_iou_details.json"
    report_path = output_dir / "e4_tiny_nms_iou_report.json"
    with csv_path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    markdown_path.write_text(
        render_tiny_nms_iou_markdown(rows, analysis=analysis), encoding="utf-8"
    )
    write_json_report(details_path, {"rows": rows, "evaluations": evaluations})

    report = {
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "weight": str(FIXED_WEIGHT.resolve()),
            "weight_sha256": sha256(FIXED_WEIGHT),
            "source_manifest": str(FIXED_MANIFEST.resolve()),
            "source_false_negatives": str(FIXED_FALSE_NEGATIVES.resolve()),
            "source_hashes": frozen_hashes_before,
            "images": EXPECTED_IMAGES,
            "original_false_negatives": EXPECTED_ORIGINAL_FN,
            "split": "val",
            "test_evaluated": False,
            "imgsz": FIXED_IMGSZ,
            "confidence": FIXED_CONFIDENCE,
            "matching_iou": FIXED_MATCHING_IOU,
            "nms_iou_values": list(FIXED_NMS_IOUS),
            "only_variable": "nms_iou",
            "class_aware_nms": True,
            "agnostic_nms": False,
            "max_det": fixed_max_det,
            "all_gt_loaded_before_matching": True,
            "only_tiny_gt_counted_for_recall": True,
            "duplicate_definition": "same-image same-class post-NMS prediction pair IoU >= 0.70",
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
