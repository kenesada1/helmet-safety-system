from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Sequence

import yaml


@dataclass(frozen=True)
class ImageRecord:
    image_path: Path
    class_counts: tuple[int, int]

    @property
    def has_both_classes(self) -> bool:
        return all(count > 0 for count in self.class_counts)


def choose_subset(records: Sequence[ImageRecord], *, count: int, seed: int) -> list[ImageRecord]:
    if count <= 0:
        raise ValueError("subset count must be greater than zero")
    if count > len(records):
        raise ValueError(f"requested {count} images but only {len(records)} are available")
    if any(count < 0 for record in records for count in record.class_counts):
        raise ValueError("class counts cannot be negative")
    if any(not any(record.class_counts[class_id] > 0 for record in records) for class_id in (0, 1)):
        raise ValueError("candidate split must contain both class IDs 0 and 1")

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    selected: list[ImageRecord] = []

    mixed = next((record for record in shuffled if record.has_both_classes), None)
    if mixed is not None:
        selected.append(mixed)

    for class_id in (0, 1):
        if any(record.class_counts[class_id] > 0 for record in selected):
            continue
        anchor = next(
            (record for record in shuffled if record not in selected and record.class_counts[class_id] > 0),
            None,
        )
        if anchor is not None:
            selected.append(anchor)

    if len(selected) > count:
        raise ValueError(f"cannot cover both classes with a subset of {count} images")
    selected.extend(record for record in shuffled if record not in selected and len(selected) < count)
    if not all(any(record.class_counts[class_id] > 0 for record in selected) for class_id in (0, 1)):
        raise ValueError(f"cannot cover both classes with a subset of {count} images")
    return selected


def subset_box_counts(records: Sequence[ImageRecord]) -> dict[str, int]:
    return {
        "helmet": sum(record.class_counts[0] for record in records),
        "no_helmet": sum(record.class_counts[1] for record in records),
    }


def next_run_name(project_dir: Path, base_name: str) -> str:
    if not (project_dir / base_name).exists():
        return base_name
    index = 2
    while (project_dir / f"{base_name}_{index:03d}").exists():
        index += 1
    return f"{base_name}_{index:03d}"


def _label_counts(label_path: Path) -> tuple[int, int]:
    if not label_path.is_file():
        raise FileNotFoundError(f"missing YOLO label: {label_path}")
    counts = [0, 0]
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        columns = line.split()
        if len(columns) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected 5 YOLO columns")
        try:
            class_id = int(columns[0])
            coordinates = [float(value) for value in columns[1:]]
        except ValueError as exc:
            raise ValueError(f"{label_path}:{line_number}: non-numeric YOLO value") from exc
        if class_id not in (0, 1):
            raise ValueError(f"{label_path}:{line_number}: unexpected class ID {class_id}")
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError(f"{label_path}:{line_number}: non-finite YOLO coordinate")
        counts[class_id] += 1
    return counts[0], counts[1]


def _has_complete_jpeg_eoi(image_path: Path) -> bool:
    if image_path.suffix.lower() not in {".jpg", ".jpeg"}:
        return True
    try:
        with image_path.open("rb") as handle:
            handle.seek(-2, 2)
            return handle.read(2) == bytes((0xFF, 0xD9))
    except OSError:
        return False


def _collect_records(processed_root: Path, split: str) -> tuple[list[ImageRecord], list[Path]]:
    images_dir = processed_root / "images" / split
    labels_dir = processed_root / "labels" / split
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"missing processed images/labels directory for split {split!r}")
    image_paths = sorted(
        (path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}),
        key=lambda path: path.name,
    )
    if not image_paths:
        raise ValueError(f"processed split {split!r} contains no images")
    incomplete_jpegs = [path.resolve() for path in image_paths if not _has_complete_jpeg_eoi(path)]
    safe_image_paths = [path for path in image_paths if _has_complete_jpeg_eoi(path)]
    return (
        [ImageRecord(path.resolve(), _label_counts(labels_dir / f"{path.stem}.txt")) for path in safe_image_paths],
        incomplete_jpegs,
    )


def create_smoke_dataset(
    processed_root: Path,
    output_dir: Path,
    *,
    train_count: int,
    val_count: int,
    seed: int,
) -> dict[str, object]:
    """Build deterministic train/val path lists while leaving processed data untouched."""

    processed_root = processed_root.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty smoke dataset directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_records, train_incomplete = _collect_records(processed_root, "train")
    val_records, val_incomplete = _collect_records(processed_root, "val")
    selections = {
        "train": choose_subset(train_records, count=train_count, seed=seed),
        "val": choose_subset(val_records, count=val_count, seed=seed),
    }
    incomplete_by_split = {"train": train_incomplete, "val": val_incomplete}
    split_reports: dict[str, object] = {}
    for split, selected in selections.items():
        (output_dir / f"{split}.txt").write_text(
            "".join(f"{record.image_path.as_posix()}\n" for record in selected), encoding="utf-8"
        )
        split_reports[split] = {
            "images": len(selected),
            "boxes": subset_box_counts(selected),
            "mixed_class_images": sum(record.has_both_classes for record in selected),
            "excluded_incomplete_jpegs": len(incomplete_by_split[split]),
            "excluded_incomplete_jpeg_paths": [str(path) for path in incomplete_by_split[split]],
            "image_paths": [str(record.image_path) for record in selected],
        }

    dataset_yaml = {
        "path": processed_root.as_posix(),
        "train": (output_dir / "train.txt").resolve().as_posix(),
        "val": (output_dir / "val.txt").resolve().as_posix(),
        "names": {0: "helmet", 1: "no_helmet"},
    }
    (output_dir / "dataset.yaml").write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    report: dict[str, object] = {"seed": seed, "splits": split_reports, "dataset_yaml": str(output_dir / "dataset.yaml")}
    (output_dir / "subset_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def validate_training_outputs(run_dir: Path, *, expected_epochs: int) -> dict[str, object]:
    """Validate the durable evidence emitted by an Ultralytics training run."""

    run_dir = run_dir.resolve()
    results_csv = run_dir / "results.csv"
    if not results_csv.is_file():
        raise FileNotFoundError(f"missing training record: {results_csv}")
    with results_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = [{key.strip(): value.strip() for key, value in row.items()} for row in csv.DictReader(handle)]
    if len(rows) != expected_epochs:
        raise ValueError(f"expected {expected_epochs} completed epoch rows, found {len(rows)}")
    final_row = rows[-1]
    loss_columns = {
        "box_loss": "train/box_loss",
        "cls_loss": "train/cls_loss",
        "dfl_loss": "train/dfl_loss",
    }
    try:
        losses = {name: float(final_row[column]) for name, column in loss_columns.items()}
    except (KeyError, ValueError) as exc:
        raise ValueError("results.csv is missing numeric YOLO training loss columns") from exc
    if not all(math.isfinite(value) for value in losses.values()):
        raise ValueError(f"training losses must be finite: {losses}")
    try:
        validation_metric = float(final_row["metrics/mAP50(B)"])
    except (KeyError, ValueError) as exc:
        raise ValueError("results.csv is missing a numeric validation metric") from exc
    if not math.isfinite(validation_metric):
        raise ValueError(f"validation metric must be finite: {validation_metric}")

    required_files = [run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt"]
    missing = [path for path in required_files if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"missing or empty saved weights: {missing}")
    train_visualizations = sorted(run_dir.glob("train_batch*.jpg"))
    val_visualizations = sorted(run_dir.glob("val_batch*_pred.jpg"))
    if not train_visualizations or not val_visualizations:
        raise FileNotFoundError("missing training batch or validation prediction visualization")
    return {
        "epochs_completed": len(rows),
        "losses": losses,
        "validation_completed": True,
        "validation_metric_map50": validation_metric,
        "results_csv": str(results_csv),
        "best_pt": str(required_files[0]),
        "last_pt": str(required_files[1]),
        "train_visualizations": [str(path) for path in train_visualizations],
        "val_visualizations": [str(path) for path in val_visualizations],
    }
