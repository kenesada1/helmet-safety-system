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

from helmet_safety.training.baseline import allocate_run_name, load_ground_truth_boxes, write_json_report  # noqa: E402
from helmet_safety.training.analysis_core import (  # noqa: E402
    EXPECTED_E4_TINY_SUMMARY,
    assert_expected_tiny_summary,
    select_tiny_val_images,
    summarize_tiny_ground_truth,
)


FIXED_WEIGHT = PROJECT_ROOT / "artifacts" / "training" / "m45_yolo11s_e75_960_001" / "weights" / "best.pt"
FIXED_SPLIT = "val"
FIXED_IMGSZ = 960
FIXED_CONFIDENCE = 0.25
FIXED_MATCHING_IOU = 0.5
FIXED_TINY_MAX_SIZE = 10.0
EXPECTED_IMAGE_COUNT = 35


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the fixed E4 tiny-object benchmark on val only: imgsz=960, conf=0.25, "
            "class-aware matching IoU=0.5, and default Ultralytics NMS"
        )
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--output-name", default="m45_yolo11s_e75_960_tiny_val_001")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _average_speeds(rows: list[Mapping[str, object]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {key: sum(float(row.get(key, 0.0)) for row in rows) / len(rows) for key in keys}


def run(args: argparse.Namespace) -> dict[str, object]:
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.cfg import DEFAULT_CFG

    if args.batch < 1:
        raise ValueError("batch must be at least 1")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {args.device!r} unavailable")

    weight = FIXED_WEIGHT.resolve()
    if not weight.is_file():
        raise FileNotFoundError(f"fixed E4 weight does not exist: {weight}")
    training_args_path = weight.parents[2] / "args.yaml"
    training_args = yaml.safe_load(training_args_path.read_text(encoding="utf-8"))
    dataset_yaml = Path(str(training_args["data"])).resolve()
    dataset = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    processed_root = Path(str(dataset["path"])).resolve()
    images_dir = (processed_root / str(dataset[FIXED_SPLIT])).resolve()
    labels_dir = (processed_root / "labels" / FIXED_SPLIT).resolve()
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"fixed val images/labels unavailable: {images_dir}, {labels_dir}")

    manifest = select_tiny_val_images(
        images_dir, labels_dir, max_equivalent_size=FIXED_TINY_MAX_SIZE
    )
    selected_gt = sum(int(item["tiny_gt_count"]) for item in manifest)
    if len(manifest) != EXPECTED_IMAGE_COUNT or selected_gt != int(EXPECTED_E4_TINY_SUMMARY["ground_truth_instances"]):
        raise RuntimeError(
            "tiny val selection mismatch: "
            f"images={len(manifest)} expected={EXPECTED_IMAGE_COUNT}, "
            f"tiny_gt={selected_gt} expected={EXPECTED_E4_TINY_SUMMARY['ground_truth_instances']}"
        )

    model = YOLO(str(weight), task="detect")
    if dict(model.names) != {0: "helmet", 1: "no_helmet"}:
        raise ValueError(f"unexpected class mapping: {model.names}")

    records: list[dict[str, object]] = []
    speed_rows: list[Mapping[str, object]] = []
    started = time.perf_counter()
    # Intentionally omit iou and every other NMS option: Ultralytics defaults remain in force.
    results = model.predict(
        source=[str(item["image_path"]) for item in manifest],
        imgsz=FIXED_IMGSZ,
        conf=FIXED_CONFIDENCE,
        batch=args.batch,
        device=args.device,
        save=False,
        stream=True,
        verbose=True,
    )
    for result in results:
        image_path = Path(result.path).resolve()
        with Image.open(image_path) as image:
            image_size = image.size
        ground_truth = load_ground_truth_boxes(
            labels_dir / f"{image_path.stem}.txt", image_size=image_size
        )
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
                "image_id": image_path.name,
                "ground_truth": ground_truth,
                "predictions": predictions,
            }
        )
        speed_rows.append(result.speed)
    wall_seconds = time.perf_counter() - started
    if len(records) != EXPECTED_IMAGE_COUNT:
        raise RuntimeError(f"prediction result count mismatch: {len(records)} != {EXPECTED_IMAGE_COUNT}")

    evaluation = summarize_tiny_ground_truth(
        records,
        max_equivalent_size=FIXED_TINY_MAX_SIZE,
        iou_threshold=FIXED_MATCHING_IOU,
    )
    summary = evaluation["summary"]
    assert_expected_tiny_summary(summary)  # No output directory or report is created before this gate passes.
    false_negatives = evaluation["false_negatives"]
    if len(false_negatives) != int(EXPECTED_E4_TINY_SUMMARY["fn"]):
        raise RuntimeError(
            f"false-negative detail count mismatch: {len(false_negatives)} != {EXPECTED_E4_TINY_SUMMARY['fn']}"
        )

    evaluation_root = PROJECT_ROOT / "artifacts" / "evaluation"
    run_name = allocate_run_name(evaluation_root, args.output_name)
    output_dir = evaluation_root / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = output_dir / "tiny_val_images.json"
    fn_path = output_dir / "tiny_val_false_negatives.json"
    report_path = output_dir / "tiny_val_benchmark_report.json"
    write_json_report(
        manifest_path,
        {
            "split": FIXED_SPLIT,
            "image_count": len(manifest),
            "tiny_gt_count": selected_gt,
            "images": manifest,
        },
    )
    write_json_report(
        fn_path,
        {
            "split": FIXED_SPLIT,
            "matching_iou": FIXED_MATCHING_IOU,
            "false_negative_count": len(false_negatives),
            "gt_box_format": "xyxy in original-image pixels",
            "false_negatives": false_negatives,
        },
    )
    report = {
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "weight": str(weight),
            "weight_sha256": _sha256(weight),
            "dataset_yaml": str(dataset_yaml),
            "split": FIXED_SPLIT,
            "test_evaluated": False,
            "imgsz": FIXED_IMGSZ,
            "confidence": FIXED_CONFIDENCE,
            "matching_iou": FIXED_MATCHING_IOU,
            "class_aware_matching": True,
            "tiny_definition": "sqrt(original_box_width_px * original_box_height_px) <= 10",
            "all_gt_loaded_before_matching": True,
            "only_tiny_gt_counted": True,
            "nms_parameters_overridden": False,
            "ultralytics_default_nms_iou": float(DEFAULT_CFG.iou),
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
        "result": summary,
        "timing": {
            "wall_seconds": wall_seconds,
            "average_ms_per_image": _average_speeds(speed_rows),
        },
        "artifacts": {
            "image_manifest": str(manifest_path.resolve()),
            "false_negatives": str(fn_path.resolve()),
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
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
