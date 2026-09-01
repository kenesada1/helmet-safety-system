#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

from PIL import Image, ImageDraw, ImageFont
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from helmet_safety.training.baseline import load_ground_truth_boxes  # noqa: E402
from helmet_safety.training.e6_error_analysis import analyze_model_errors, compare_gt_outcomes  # noqa: E402


DATASET_YAML = Path(r"D:\datasets\SHWD\processed\dataset.yaml")
E4_WEIGHT = PROJECT_ROOT / "artifacts" / "training" / "m45_yolo11s_e75_960_001" / "weights" / "best.pt"
E6_WEIGHT = PROJECT_ROOT / "artifacts" / "e6" / "e6_yolo11s_p2_001" / "weights" / "best.pt"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "e6" / "e6_yolo11s_p2_001" / "error_analysis"
CONFIDENCE_THRESHOLD = 0.25
DIAGNOSTIC_CONFIDENCE = 0.001
MATCHING_IOU = 0.5
NMS_IOU = 0.7
EXPECTED_IMAGES = 607
EXPECTED_GT = 9925
CLASS_NAMES = {0: "helmet", 1: "no_helmet"}


def size_bucket(size: float) -> str:
    if size <= 10:
        return "le_10"
    if size <= 20:
        return "10_20"
    if size <= 30:
        return "20_30"
    if size <= 50:
        return "30_50"
    return "gt_50"


def summarize_errors(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "total": len(rows),
        "by_reason": dict(Counter(str(row["reason"]) for row in rows)),
        "by_class": dict(Counter(CLASS_NAMES[int(row["class_id"])] for row in rows)),
        "by_size": dict(Counter(size_bucket(float(row["equivalent_size"])) for row in rows)),
        "edge": sum(bool(row["edge"]) for row in rows),
        "dense_scene_gt_gte_10": sum(int(row["scene_gt_count"]) >= 10 for row in rows),
        "dense_scene_gt_gte_20": sum(int(row["scene_gt_count"]) >= 20 for row in rows),
    }


def summarize_transitions(rows: list[dict[str, object]]) -> dict[str, object]:
    overall = Counter(str(row["transition"]) for row in rows)
    by_size: dict[str, Counter[str]] = defaultdict(Counter)
    by_class: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_size[size_bucket(float(row["equivalent_size"]))][str(row["transition"])] += 1
        by_class[CLASS_NAMES[int(row["class_id"])]] [str(row["transition"])] += 1
    return {
        "overall": dict(overall),
        "by_size": {name: dict(values) for name, values in by_size.items()},
        "by_class": {name: dict(values) for name, values in by_class.items()},
    }


def diagnostic_predictions(model: object, images_dir: Path) -> tuple[dict[str, list[dict[str, object]]], float]:
    started = time.perf_counter()
    predictions: dict[str, list[dict[str, object]]] = {}
    results = model.predict(
        source=str(images_dir),
        imgsz=960,
        batch=2,
        conf=DIAGNOSTIC_CONFIDENCE,
        iou=NMS_IOU,
        max_det=300,
        agnostic_nms=False,
        device="0",
        save=False,
        stream=True,
        verbose=False,
    )
    for result in results:
        predictions[Path(result.path).name] = [
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
    if len(predictions) != EXPECTED_IMAGES:
        raise RuntimeError(f"diagnostic inference covered {len(predictions)} images, expected {EXPECTED_IMAGES}")
    return predictions, time.perf_counter() - started


def crop_geometry(box: list[float], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = image_size
    box_width = max(1.0, box[2] - box[0])
    box_height = max(1.0, box[3] - box[1])
    side = min(max(max(box_width, box_height) * 8, 160), min(width, height))
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    left = max(0, min(int(center_x - side / 2), width - int(side)))
    top = max(0, min(int(center_y - side / 2), height - int(side)))
    return left, top, min(width, left + int(side)), min(height, top + int(side))


def render_contact_sheet(
    output_path: Path,
    rows: list[dict[str, object]],
    *,
    images_dir: Path,
    ground_truth_by_image: dict[str, list[dict[str, object]]],
    e4_predictions: dict[str, list[dict[str, object]]],
    e6_predictions: dict[str, list[dict[str, object]]],
    limit: int = 20,
) -> list[dict[str, object]]:
    selected = rows[:limit]
    tile_width, tile_height, columns = 320, 280, 4
    rows_count = max(1, (len(selected) + columns - 1) // columns)
    sheet = Image.new("RGB", (tile_width * columns, tile_height * rows_count), "white")
    font = ImageFont.load_default()
    for position, row in enumerate(selected):
        image_id = str(row["image_id"])
        with Image.open(images_dir / image_id) as source:
            source = source.convert("RGB")
            focus_box = [float(value) for value in row["box"]]  # type: ignore[union-attr]
            crop_box = crop_geometry(focus_box, source.size)
            crop = source.crop(crop_box)
        scale_x = tile_width / crop.width
        canvas_height = tile_height - 42
        scale_y = canvas_height / crop.height
        crop = crop.resize((tile_width, canvas_height))
        draw = ImageDraw.Draw(crop)

        def draw_boxes(items: list[dict[str, object]], color: str, width: int) -> None:
            for item in items:
                box = [float(value) for value in item["box"]]  # type: ignore[union-attr]
                if box[2] < crop_box[0] or box[0] > crop_box[2] or box[3] < crop_box[1] or box[1] > crop_box[3]:
                    continue
                transformed = [
                    (box[0] - crop_box[0]) * scale_x,
                    (box[1] - crop_box[1]) * scale_y,
                    (box[2] - crop_box[0]) * scale_x,
                    (box[3] - crop_box[1]) * scale_y,
                ]
                draw.rectangle(transformed, outline=color, width=width)

        draw_boxes(ground_truth_by_image[image_id], "#00ff00", 3)
        draw_boxes(
            [item for item in e4_predictions[image_id] if float(item["confidence"]) >= CONFIDENCE_THRESHOLD],
            "#ff3030",
            2,
        )
        draw_boxes(
            [item for item in e6_predictions[image_id] if float(item["confidence"]) >= CONFIDENCE_THRESHOLD],
            "#00bfff",
            2,
        )
        x = (position % columns) * tile_width
        y = (position // columns) * tile_height
        sheet.paste(crop, (x, y + 42))
        label = f"{image_id[:24]}  {row.get('reason', row.get('transition', ''))}"
        detail = f"cls={CLASS_NAMES[int(row['class_id'])]} size={float(row['equivalent_size']):.1f}px"
        ImageDraw.Draw(sheet).text((x + 4, y + 4), label, fill="black", font=font)
        ImageDraw.Draw(sheet).text((x + 4, y + 20), detail, fill="black", font=font)
    sheet.save(output_path, quality=92)
    return selected


def main() -> int:
    import torch
    from ultralytics import YOLO

    if OUTPUT_DIR.exists():
        raise FileExistsError(f"refusing to overwrite existing analysis: {OUTPUT_DIR}")
    for path in (DATASET_YAML, E4_WEIGHT, E6_WEIGHT):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this full-val analysis")

    dataset = yaml.safe_load(DATASET_YAML.read_text(encoding="utf-8"))
    root = Path(str(dataset["path"])).resolve()
    images_dir = (root / str(dataset["val"])).resolve()
    labels_dir = root / "labels" / "val"
    image_paths = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(image_paths) != EXPECTED_IMAGES:
        raise RuntimeError(f"val image count mismatch: {len(image_paths)}")

    ground_truth_by_image: dict[str, list[dict[str, object]]] = {}
    image_sizes: dict[str, tuple[int, int]] = {}
    for image_path in image_paths:
        with Image.open(image_path) as image:
            image_sizes[image_path.name] = image.size
        ground_truth_by_image[image_path.name] = load_ground_truth_boxes(
            labels_dir / f"{image_path.stem}.txt", image_size=image_sizes[image_path.name]
        )
    if sum(len(items) for items in ground_truth_by_image.values()) != EXPECTED_GT:
        raise RuntimeError("val GT count mismatch")

    predictions_by_model: dict[str, dict[str, list[dict[str, object]]]] = {}
    inference_seconds: dict[str, float] = {}
    for name, weight, strides in (
        ("E4", E4_WEIGHT, [8.0, 16.0, 32.0]),
        ("E6", E6_WEIGHT, [4.0, 8.0, 16.0, 32.0]),
    ):
        model = YOLO(str(weight), task="detect")
        actual_strides = [float(value) for value in model.model.stride.detach().cpu().tolist()]
        if actual_strides != strides:
            raise RuntimeError(f"{name} stride mismatch: {actual_strides}")
        predictions_by_model[name], inference_seconds[name] = diagnostic_predictions(model, images_dir)
        del model
        torch.cuda.empty_cache()

    errors: dict[str, dict[str, list[dict[str, object]]]] = {}
    transitions: list[dict[str, object]] = []
    for image_path in image_paths:
        image_id = image_path.name
        ground_truth = ground_truth_by_image[image_id]
        errors[image_id] = {}
        for name in ("E4", "E6"):
            result = analyze_model_errors(
                ground_truth,
                predictions_by_model[name][image_id],
                image_size=image_sizes[image_id],
                confidence_threshold=CONFIDENCE_THRESHOLD,
                matching_iou=MATCHING_IOU,
            )
            for error_type in ("false_negatives", "false_positives"):
                enriched = []
                for item in result[error_type]:  # type: ignore[index]
                    enriched.append({"image_id": image_id, "scene_gt_count": len(ground_truth), **item})
                errors[image_id][f"{name}_{error_type}"] = enriched
        for item in compare_gt_outcomes(
            ground_truth,
            predictions_by_model["E4"][image_id],
            predictions_by_model["E6"][image_id],
            confidence_threshold=CONFIDENCE_THRESHOLD,
            matching_iou=MATCHING_IOU,
        ):
            transitions.append({"image_id": image_id, "scene_gt_count": len(ground_truth), **item})

    flat_errors: dict[str, list[dict[str, object]]] = {}
    for name in ("E4", "E6"):
        for error_type in ("false_negatives", "false_positives"):
            key = f"{name}_{error_type}"
            flat_errors[key] = [item for image in errors.values() for item in image[key]]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    samples_dir = OUTPUT_DIR / "contact_sheets"
    samples_dir.mkdir()
    sample_groups = {
        "tiny_recovered": [row for row in transitions if row["transition"] == "recovered" and float(row["equivalent_size"]) <= 10],
        "tiny_regressed": [row for row in transitions if row["transition"] == "regressed" and float(row["equivalent_size"]) <= 10],
        "tiny_both_missed": [row for row in transitions if row["transition"] == "both_missed" and float(row["equivalent_size"]) <= 10],
    }
    for reason in ("low_confidence", "localization", "class_confusion", "no_response"):
        sample_groups[f"e6_fn_{reason}"] = [row for row in flat_errors["E6_false_negatives"] if row["reason"] == reason]
    for reason in ("class_confusion", "duplicate", "near_object", "background"):
        sample_groups[f"e6_fp_{reason}"] = [row for row in flat_errors["E6_false_positives"] if row["reason"] == reason]

    selected_samples: dict[str, list[dict[str, object]]] = {}
    for name, rows in sample_groups.items():
        rows.sort(key=lambda row: (float(row["equivalent_size"]), -int(row["scene_gt_count"])))
        selected_samples[name] = render_contact_sheet(
            samples_dir / f"{name}.jpg",
            rows,
            images_dir=images_dir,
            ground_truth_by_image=ground_truth_by_image,
            e4_predictions=predictions_by_model["E4"],
            e6_predictions=predictions_by_model["E6"],
        )

    report = {
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "split": "val",
            "test_used": False,
            "images": EXPECTED_IMAGES,
            "ground_truth": EXPECTED_GT,
            "imgsz": 960,
            "batch": 2,
            "diagnostic_confidence": DIAGNOSTIC_CONFIDENCE,
            "reported_confidence": CONFIDENCE_THRESHOLD,
            "nms_iou": NMS_IOU,
            "matching_iou": MATCHING_IOU,
        },
        "inference_seconds": inference_seconds,
        "summary": {
            key: summarize_errors(rows) for key, rows in flat_errors.items()
        },
        "transitions": summarize_transitions(transitions),
        "selected_samples": selected_samples,
        "artifacts": {"contact_sheets": str(samples_dir.resolve())},
    }
    (OUTPUT_DIR / "e6_error_analysis_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "e6_error_analysis_details.json").write_text(
        json.dumps({"errors": errors, "transitions": transitions}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(json.dumps(report["transitions"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
