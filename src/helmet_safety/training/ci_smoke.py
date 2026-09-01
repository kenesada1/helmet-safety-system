"""Portable, private-data-free one-epoch training smoke gate for CI."""

from __future__ import annotations

import json
from pathlib import Path
import random
import shutil
from typing import Any

import cv2
import numpy as np
import yaml


def reset_generated_output(output_dir: Path, *, artifacts_root: Path) -> None:
    """Remove only a marked CI output strictly below the intended artifacts root."""

    output_dir = output_dir.expanduser().resolve()
    artifacts_root = artifacts_root.expanduser().resolve()
    try:
        relative = output_dir.relative_to(artifacts_root)
    except ValueError as exc:
        raise ValueError("CI output must be below artifacts root") from exc
    if relative == Path("."):
        raise ValueError("CI output must be strictly below artifacts root")
    markers = (
        output_dir / "smoke_report.json",
        output_dir / "dataset" / "dataset_report.json",
    )
    if not any(marker.is_file() for marker in markers):
        raise ValueError(f"refusing cleanup without a generated marker: {output_dir}")
    shutil.rmtree(output_dir)


def create_synthetic_dataset(output_dir: Path, *, seed: int = 42) -> dict[str, Any]:
    """Create a tiny valid two-class YOLO dataset without network or SHWD access."""

    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite CI smoke dataset: {output_dir}")
    randomizer = random.Random(seed)
    split_sizes = {"train": 4, "val": 2}
    split_reports: dict[str, dict[str, object]] = {}
    for split, count in split_sizes.items():
        images_dir = output_dir / "images" / split
        labels_dir = output_dir / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=False)
        labels_dir.mkdir(parents=True, exist_ok=False)
        for index in range(count):
            image = np.full(
                (96, 96, 3),
                (30 + index * 15, 45 + index * 10, 60 + index * 5),
                dtype=np.uint8,
            )
            jitter = randomizer.randint(-2, 2)
            cv2.rectangle(image, (12 + jitter, 12), (38 + jitter, 40), (0, 200, 0), -1)
            cv2.rectangle(image, (55 - jitter, 50), (82 - jitter, 84), (0, 0, 220), -1)
            image_path = images_dir / f"{index:03d}.jpg"
            if not cv2.imwrite(str(image_path), image):
                raise RuntimeError(f"failed to write CI image: {image_path}")
            (labels_dir / f"{index:03d}.txt").write_text(
                "0 0.260417 0.270833 0.270833 0.291667\n"
                "1 0.713542 0.697917 0.281250 0.354167\n",
                encoding="utf-8",
            )
        split_reports[split] = {
            "images": count,
            "boxes": {"helmet": count, "no_helmet": count},
        }
    dataset = {
        "path": output_dir.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "helmet", 1: "no_helmet"},
    }
    dataset_yaml = output_dir / "dataset.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump(dataset, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    report: dict[str, Any] = {
        "seed": seed,
        "dataset_yaml": str(dataset_yaml),
        "splits": split_reports,
    }
    (output_dir / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def run_one_epoch_smoke(
    output_dir: Path,
    *,
    seed: int = 42,
    imgsz: int = 64,
    batch: int = 2,
) -> dict[str, str | int]:
    """Run one real YOLO epoch from architecture config and validate durable outputs."""

    if imgsz < 32 or imgsz % 32:
        raise ValueError("imgsz must be at least 32 and divisible by 32")
    if batch < 1:
        raise ValueError("batch must be positive")
    output_dir = output_dir.expanduser().resolve()
    dataset_report = create_synthetic_dataset(output_dir / "dataset", seed=seed)
    from ultralytics import YOLO

    model = YOLO("yolo11n.yaml", task="detect")
    model.train(
        data=dataset_report["dataset_yaml"],
        epochs=1,
        imgsz=imgsz,
        batch=batch,
        workers=0,
        cache=False,
        seed=seed,
        pretrained=False,
        plots=False,
        device="cpu",
        deterministic=True,
        project=str(output_dir / "runs"),
        name="one_epoch",
        exist_ok=False,
        verbose=False,
    )
    run_dir = Path(model.trainer.save_dir).resolve()
    required = [
        run_dir / "results.csv",
        run_dir / "weights" / "best.pt",
        run_dir / "weights" / "last.pt",
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"CI smoke training did not produce required evidence: {missing}")
    report: dict[str, str | int] = {
        "status": "passed",
        "epochs": 1,
        "run_dir": str(run_dir),
        "best_pt": str(required[1]),
    }
    (output_dir / "smoke_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report
