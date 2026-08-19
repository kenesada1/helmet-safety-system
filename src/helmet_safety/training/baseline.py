from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import random
import re
from typing import Mapping, Sequence

from PIL import Image, ImageDraw


def training_batch_attempts(requested_batch: int, *, auto_oom_retry: bool) -> list[int]:
    """Return safe physical-batch attempts without ever changing image size."""

    if requested_batch < 1:
        raise ValueError("requested batch must be at least 1")
    if not auto_oom_retry:
        return [requested_batch]
    return [requested_batch, *[batch for batch in (4, 2, 1) if batch < requested_batch]]


def allocate_run_name(project_dir: Path, requested_name: str) -> str:
    """Return a non-existing run name while preserving a numeric suffix convention."""

    if not (project_dir / requested_name).exists():
        return requested_name
    match = re.fullmatch(r"(.+_)(\d+)", requested_name)
    if match:
        prefix, digits = match.groups()
        index = int(digits) + 1
        width = len(digits)
        while (project_dir / f"{prefix}{index:0{width}d}").exists():
            index += 1
        return f"{prefix}{index:0{width}d}"
    index = 2
    while (project_dir / f"{requested_name}_{index:03d}").exists():
        index += 1
    return f"{requested_name}_{index:03d}"


def allocate_experiment_name(training_dir: Path, logs_dir: Path, requested_name: str) -> str:
    """Protect both Ultralytics run directories and their paired console logs."""

    def occupied(name: str) -> bool:
        return (training_dir / name).exists() or (logs_dir / f"{name}.log").exists()

    if not occupied(requested_name):
        return requested_name
    match = re.fullmatch(r"(.+_)(\d+)", requested_name)
    if match:
        prefix, digits = match.groups()
        index = int(digits) + 1
        width = len(digits)
        while occupied(f"{prefix}{index:0{width}d}"):
            index += 1
        return f"{prefix}{index:0{width}d}"
    index = 2
    while occupied(f"{requested_name}_{index:03d}"):
        index += 1
    return f"{requested_name}_{index:03d}"


def build_training_kwargs(
    *,
    data_yaml: Path,
    project_dir: Path,
    run_name: str,
    epochs: int = 50,
    patience: int = 15,
    imgsz: int = 640,
    batch: int = 8,
    workers: int = 0,
    device: str = "0",
    seed: int = 42,
    deterministic: bool = True,
    cache: bool = False,
    amp: bool = True,
    plots: bool = True,
) -> dict[str, object]:
    """Build the deliberately conservative M4 baseline arguments."""

    return {
        "data": str(data_yaml),
        "epochs": epochs,
        "patience": patience,
        "imgsz": imgsz,
        "batch": batch,
        "workers": workers,
        "device": device,
        "seed": seed,
        "deterministic": deterministic,
        "cache": cache,
        "amp": amp,
        "plots": plots,
        "pretrained": True,
        "project": str(project_dir),
        "name": run_name,
        "exist_ok": False,
    }


def dataset_inventory(processed_root: Path) -> dict[str, dict[str, object]]:
    """Count durable dataset artifacts and class instances for every split."""

    inventory: dict[str, dict[str, object]] = {}
    for split in ("train", "val", "test"):
        images_dir = processed_root / "images" / split
        labels_dir = processed_root / "labels" / split
        image_paths = [
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        label_paths = [path for path in labels_dir.glob("*.txt") if path.is_file()]
        counts = [0, 0]
        for label_path in label_paths:
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                columns = line.split()
                if len(columns) != 5 or columns[0] not in {"0", "1"}:
                    raise ValueError(f"invalid YOLO label at {label_path}:{line_number}")
                counts[int(columns[0])] += 1
        inventory[split] = {
            "images": len(image_paths),
            "labels": len(label_paths),
            "boxes": sum(counts),
            "class_boxes": {"helmet": counts[0], "no_helmet": counts[1]},
        }
    return inventory


def analyze_training_results(results_csv: Path, *, requested_epochs: int) -> dict[str, object]:
    """Summarize epoch history and make a conservative overfitting assessment."""

    with results_csv.open(encoding="utf-8-sig", newline="") as handle:
        raw_rows = [{key.strip(): value.strip() for key, value in row.items()} for row in csv.DictReader(handle)]
    if not raw_rows:
        raise ValueError(f"training results contain no completed epochs: {results_csv}")
    columns = {
        "epoch": "epoch",
        "train_box": "train/box_loss",
        "train_cls": "train/cls_loss",
        "train_dfl": "train/dfl_loss",
        "precision": "metrics/precision(B)",
        "recall": "metrics/recall(B)",
        "map50": "metrics/mAP50(B)",
        "map50_95": "metrics/mAP50-95(B)",
        "val_box": "val/box_loss",
        "val_cls": "val/cls_loss",
        "val_dfl": "val/dfl_loss",
    }
    rows: list[dict[str, float]] = []
    try:
        for raw_row in raw_rows:
            row = {name: float(raw_row[column]) for name, column in columns.items()}
            if not all(math.isfinite(value) for value in row.values()):
                raise ValueError(f"training results must contain only finite values: {row}")
            rows.append(row)
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and "finite" in str(exc):
            raise
        raise ValueError(f"training results are missing required finite numeric columns: {results_csv}") from exc

    best_index = max(range(len(rows)), key=lambda index: 0.1 * rows[index]["map50"] + 0.9 * rows[index]["map50_95"])
    first, best, last = rows[0], rows[best_index], rows[-1]

    def loss_series(prefix: str) -> dict[str, dict[str, float]]:
        return {
            loss_name: {
                "first": first[f"{prefix}_{short}"],
                "best": best[f"{prefix}_{short}"],
                "last": last[f"{prefix}_{short}"],
            }
            for loss_name, short in (("box_loss", "box"), ("cls_loss", "cls"), ("dfl_loss", "dfl"))
        }

    epoch_gap = int(last["epoch"] - best["epoch"])
    best_train_total = sum(best[name] for name in ("train_box", "train_cls", "train_dfl"))
    last_train_total = sum(last[name] for name in ("train_box", "train_cls", "train_dfl"))
    best_val_total = sum(best[name] for name in ("val_box", "val_cls", "val_dfl"))
    last_val_total = sum(last[name] for name in ("val_box", "val_cls", "val_dfl"))
    train_continued_down = last_train_total < best_train_total * 0.98
    val_loss_worsened = last_val_total > best_val_total * 1.05
    map_worsened = last["map50_95"] < best["map50_95"] - 0.02
    precision_recall_diverged = (
        (last["precision"] - best["precision"]) * (last["recall"] - best["recall"]) < 0
        and abs((last["precision"] - best["precision"]) - (last["recall"] - best["recall"])) > 0.05
    )
    evidence_count = sum((train_continued_down and val_loss_worsened, map_worsened, precision_recall_diverged))
    overfitting_detected = epoch_gap >= 3 and evidence_count >= 2

    trailing_rows = rows[-20:]

    def trend(values: Sequence[float]) -> dict[str, float]:
        epoch_span = max(1.0, trailing_rows[-1]["epoch"] - trailing_rows[0]["epoch"])
        change = values[-1] - values[0]
        return {
            "first": round(values[0], 6),
            "last": round(values[-1], 6),
            "change": round(change, 6),
            "slope_per_epoch": round(change / epoch_span, 6),
        }

    trailing_window = {
        "epoch_count": len(trailing_rows),
        "first_epoch": int(trailing_rows[0]["epoch"]),
        "last_epoch": int(trailing_rows[-1]["epoch"]),
        "train_total_loss": trend(
            [sum(row[name] for name in ("train_box", "train_cls", "train_dfl")) for row in trailing_rows]
        ),
        "val_total_loss": trend(
            [sum(row[name] for name in ("val_box", "val_cls", "val_dfl")) for row in trailing_rows]
        ),
        "map50_95": trend([row["map50_95"] for row in trailing_rows]),
        "precision": trend([row["precision"] for row in trailing_rows]),
        "recall": trend([row["recall"] for row in trailing_rows]),
    }

    return {
        "epochs_completed": len(rows),
        "best_epoch": int(best["epoch"]),
        "last_epoch": int(last["epoch"]),
        "early_stopping_triggered": len(rows) < requested_epochs,
        "stopping_reason": (
            "early_stopping_before_requested_epochs" if len(rows) < requested_epochs else "requested_epochs_completed"
        ),
        "best_val_metrics": {
            "precision": best["precision"],
            "recall": best["recall"],
            "map50": best["map50"],
            "map50_95": best["map50_95"],
        },
        "last_val_metrics": {
            "precision": last["precision"],
            "recall": last["recall"],
            "map50": last["map50"],
            "map50_95": last["map50_95"],
        },
        "losses": {"train": loss_series("train"), "val": loss_series("val")},
        "trailing_window": trailing_window,
        "overfitting": {
            "detected": overfitting_detected,
            "best_to_last_epoch_gap": epoch_gap,
            "train_loss_continued_down": train_continued_down,
            "val_loss_worsened": val_loss_worsened,
            "map50_95_worsened": map_worsened,
            "precision_recall_diverged": precision_recall_diverged,
            "assessment_rule": "at least 3 epochs after best and at least 2 sustained warning signals",
        },
    }


def _class_ids(label_path: Path) -> list[int]:
    class_ids: list[int] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        columns = line.split()
        if len(columns) != 5 or columns[0] not in {"0", "1"}:
            raise ValueError(f"invalid YOLO label at {label_path}:{line_number}")
        class_ids.append(int(columns[0]))
    return class_ids


def select_prediction_images(processed_root: Path, *, count: int, seed: int) -> list[Path]:
    """Select a deterministic test sample while anchoring both classes when possible."""

    images_dir = processed_root / "images" / "test"
    labels_dir = processed_root / "labels" / "test"
    image_paths = sorted(
        (
            path.resolve()
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ),
        key=lambda path: path.name,
    )
    if count <= 0 or count > len(image_paths):
        raise ValueError(f"prediction sample count must be between 1 and {len(image_paths)}")
    records = [(path, set(_class_ids(labels_dir / f"{path.stem}.txt"))) for path in image_paths]
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    selected: list[tuple[Path, set[int]]] = []
    mixed = next((record for record in shuffled if record[1] == {0, 1}), None)
    if mixed is not None:
        selected.append(mixed)
    for class_id in (0, 1):
        if any(class_id in classes for _, classes in selected):
            continue
        anchor = next((record for record in shuffled if record not in selected and class_id in record[1]), None)
        if anchor is not None:
            selected.append(anchor)
    selected.extend(record for record in shuffled if record not in selected and len(selected) < count)
    return [path for path, _ in selected[:count]]


def _box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def analyze_image_detections(
    ground_truth: Sequence[Mapping[str, object]],
    predictions: Sequence[Mapping[str, object]],
    *,
    image_size: tuple[int, int],
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    """Greedily match one image and expose safety-relevant error categories."""

    pairs = sorted(
        (
            (_box_iou(gt["box"], pred["box"]), gt_index, pred_index)  # type: ignore[arg-type]
            for gt_index, gt in enumerate(ground_truth)
            for pred_index, pred in enumerate(predictions)
        ),
        reverse=True,
    )
    max_iou = pairs[0][0] if pairs else 0.0
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    class_confusions = 0
    class_names = {0: "helmet", 1: "no_helmet"}
    confusion_pairs: dict[str, int] = {}
    for require_same_class in (True, False):
        for iou, gt_index, pred_index in pairs:
            same_class = ground_truth[gt_index]["class_id"] == predictions[pred_index]["class_id"]
            if iou < iou_threshold or same_class != require_same_class:
                continue
            if gt_index in matched_gt or pred_index in matched_pred:
                continue
            matched_gt.add(gt_index)
            matched_pred.add(pred_index)
            if not same_class:
                class_confusions += 1
                gt_name = class_names.get(int(ground_truth[gt_index]["class_id"]), str(ground_truth[gt_index]["class_id"]))
                pred_name = class_names.get(int(predictions[pred_index]["class_id"]), str(predictions[pred_index]["class_id"]))
                pair_name = f"{gt_name}_as_{pred_name}"
                confusion_pairs[pair_name] = confusion_pairs.get(pair_name, 0) + 1

    unmatched_gt = [index for index in range(len(ground_truth)) if index not in matched_gt]
    unmatched_pred = [index for index in range(len(predictions)) if index not in matched_pred]
    localization_pairs = [
        (iou, gt_index, pred_index)
        for iou, gt_index, pred_index in pairs
        if 0.1 <= iou < iou_threshold
        and gt_index in unmatched_gt
        and pred_index in unmatched_pred
        and ground_truth[gt_index]["class_id"] == predictions[pred_index]["class_id"]
    ]
    image_area = max(1, image_size[0] * image_size[1])
    small_misses = 0
    for gt_index in unmatched_gt:
        box = ground_truth[gt_index]["box"]  # type: ignore[assignment]
        area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])  # type: ignore[index,operator]
        if area / image_area <= 0.01:
            small_misses += 1
    related_ids = sorted(
        {int(item["class_id"]) for item in ground_truth} | {int(item["class_id"]) for item in predictions}
    )
    return {
        "ground_truth_count": len(ground_truth),
        "prediction_count": len(predictions),
        "matched_correctly": len(matched_gt) - class_confusions,
        "max_iou": max_iou,
        "related_classes": [class_names.get(class_id, str(class_id)) for class_id in related_ids],
        "missed_by_class": {
            name: sum(int(ground_truth[index]["class_id"]) == class_id for index in unmatched_gt)
            for class_id, name in class_names.items()
        },
        "false_positive_by_class": {
            name: sum(int(predictions[index]["class_id"]) == class_id for index in unmatched_pred)
            for class_id, name in class_names.items()
        },
        "class_confusion_pairs": confusion_pairs,
        "error_counts": {
            "missed_detection": len(unmatched_gt),
            "false_positive": len(unmatched_pred),
            "class_confusion": class_confusions,
            "box_misalignment": len(localization_pairs),
            "small_target_miss": small_misses,
            "dense_scene_with_errors": int(len(ground_truth) >= 10 and bool(unmatched_gt or unmatched_pred or class_confusions)),
        },
    }


def format_evaluation_metrics(
    results_dict: Mapping[str, object],
    class_rows: Sequence[Mapping[str, object]],
    speed: Mapping[str, object],
) -> dict[str, object]:
    """Normalize Ultralytics validation metrics into a stable JSON schema."""

    keys = {
        "precision": "metrics/precision(B)",
        "recall": "metrics/recall(B)",
        "map50": "metrics/mAP50(B)",
        "map50_95": "metrics/mAP50-95(B)",
    }
    try:
        overall = {name: float(results_dict[key]) for name, key in keys.items()}
        speeds = {name: float(value) for name, value in speed.items()}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("evaluation metrics are missing required numeric values") from exc
    if not all(math.isfinite(value) for value in [*overall.values(), *speeds.values()]):
        raise ValueError("evaluation metrics and speed values must be finite")
    per_class: dict[str, dict[str, object]] = {}
    for row in class_rows:
        try:
            name = str(row["Class"])
            values = {
                "images": int(row["Images"]),
                "instances": int(row["Instances"]),
                "precision": float(row["Box-P"]),
                "recall": float(row["Box-R"]),
                "map50": float(row["mAP50"]),
                "map50_95": float(row["mAP50-95"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid per-class metric row: {row}") from exc
        if not all(math.isfinite(value) for value in values.values() if isinstance(value, float)):
            raise ValueError(f"per-class metrics must be finite: {row}")
        per_class[name] = values
    inference_ms = speeds.get("inference", 0.0)
    return {
        "overall": overall,
        "per_class": per_class,
        "speed_ms_per_image": speeds,
        "average_pipeline_ms_per_image": sum(speeds.values()),
        "gpu_inference_images_per_second": 1000.0 / inference_ms if inference_ms > 0 else None,
    }


def write_json_report(path: Path, report: Mapping[str, object], *, overwrite: bool = False) -> None:
    """Write UTF-8 JSON while protecting completed experiment evidence by default."""

    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=lambda value: str(value)) + "\n",
        encoding="utf-8",
    )


def validate_baseline_outputs(run_dir: Path) -> dict[str, str]:
    """Require the complete set of durable training evidence requested for M4."""

    relative_paths = {
        "best_pt": Path("weights/best.pt"),
        "last_pt": Path("weights/last.pt"),
        "results_csv": Path("results.csv"),
        "results_plot": Path("results.png"),
        "confusion_matrix": Path("confusion_matrix.png"),
        "confusion_matrix_normalized": Path("confusion_matrix_normalized.png"),
        "pr_curve": Path("BoxPR_curve.png"),
        "f1_curve": Path("BoxF1_curve.png"),
        "precision_curve": Path("BoxP_curve.png"),
        "recall_curve": Path("BoxR_curve.png"),
    }
    outputs = {name: (run_dir / relative).resolve() for name, relative in relative_paths.items()}
    missing = [path for path in outputs.values() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"missing or empty baseline outputs: {missing}")
    return {name: str(path) for name, path in outputs.items()}


def load_ground_truth_boxes(label_path: Path, *, image_size: tuple[int, int]) -> list[dict[str, object]]:
    """Load normalized YOLO labels as pixel xyxy boxes."""

    width, height = image_size
    boxes: list[dict[str, object]] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        columns = line.split()
        if len(columns) != 5 or columns[0] not in {"0", "1"}:
            raise ValueError(f"invalid YOLO label at {label_path}:{line_number}")
        try:
            class_id = int(columns[0])
            center_x, center_y, box_width, box_height = (float(value) for value in columns[1:])
        except ValueError as exc:
            raise ValueError(f"non-numeric YOLO label at {label_path}:{line_number}") from exc
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in (center_x, center_y, box_width, box_height)):
            raise ValueError(f"out-of-range YOLO label at {label_path}:{line_number}")
        boxes.append(
            {
                "class_id": class_id,
                "box": [
                    (center_x - box_width / 2) * width,
                    (center_y - box_height / 2) * height,
                    (center_x + box_width / 2) * width,
                    (center_y + box_height / 2) * height,
                ],
            }
        )
    return boxes


def save_ground_truth_visualization(
    image_path: Path, output_path: Path, ground_truth: Sequence[Mapping[str, object]]
) -> None:
    """Render one standalone ground-truth JPEG for side-by-side human review."""

    class_names = {0: "helmet", 1: "no_helmet"}
    colors = {0: (0, 180, 0), 1: (220, 40, 40)}
    with Image.open(image_path) as source:
        rendered = source.convert("RGB")
    draw = ImageDraw.Draw(rendered)
    line_width = max(2, round(min(rendered.size) / 240))
    for item in ground_truth:
        class_id = int(item["class_id"])
        box = [float(value) for value in item["box"]]  # type: ignore[union-attr]
        color = colors.get(class_id, (255, 180, 0))
        draw.rectangle(box, outline=color, width=line_width)
        label = class_names.get(class_id, str(class_id))
        text_box = draw.textbbox((box[0], box[1]), label)
        draw.rectangle(text_box, fill=color)
        draw.text((box[0], box[1]), label, fill="white")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(output_path, format="JPEG", quality=94)


def scan_training_log(log_path: Path) -> dict[str, object]:
    """Extract dataset-integrity warnings that would invalidate the M4 evidence."""

    text = log_path.read_text(encoding="utf-8", errors="replace")
    lower = text.lower()
    warning_lines = []
    for line in text.splitlines():
        line_lower = line.lower()
        mentions_warning = "warning" in line_lower
        mentions_nonzero_corruption = "corrupt" in line_lower and not re.search(
            r"\b0\s+(?:backgrounds?,\s*)?corrupt\b", line_lower
        )
        if mentions_warning or mentions_nonzero_corruption:
            warning_lines.append(line.strip())
    return {
        "jpeg_auto_repair_warning": "corrupt jpeg restored and saved" in lower
        or ("jpeg" in lower and "restored" in lower and "saved" in lower),
        "cache_version_warning": "cache version" in lower and any(
            term in lower for term in ("mismatch", "error", "incompatible", "invalid")
        ),
        "corrupt_data_warning": "corrupt image" in lower
        or "corrupt label" in lower
        or "corrupt jpeg" in lower
        or "corrupt images" in lower,
        "warning_lines": warning_lines,
    }
