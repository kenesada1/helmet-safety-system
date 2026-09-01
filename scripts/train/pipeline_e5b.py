#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
from typing import Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from helmet_safety.training.baseline import (  # noqa: E402
    analyze_training_results,
    load_ground_truth_boxes,
    write_json_report,
)
from helmet_safety.training.e5b import (  # noqa: E402
    MAX_AUGMENTED_AUDIT_SAMPLES,
    MIN_TRACKED_CENTERS,
    audit_crop_mappings,
    build_context_crop_manifest,
    build_e5b_training_kwargs,
    convert_boxes_for_crop,
    evaluate_augmented_center_gate,
    normalized_box_is_in_bounds,
    resolve_e5b_experiment_root,
    select_context_crop_requests,
    should_continue_augmentation_audit,
    summarize_e5b_evaluation,
    validate_e5b_resume_request,
)
from helmet_safety.training.analysis_core import validated_streaming_image_source  # noqa: E402


ROOT = resolve_e5b_experiment_root(PROJECT_ROOT, "e5b_context_crop_002")
SOURCE_DATA_YAML = Path(r"D:\datasets\SHWD\processed\dataset.yaml")
E4_WEIGHT = PROJECT_ROOT / "artifacts" / "training" / "m45_yolo11s_e75_960_001" / "weights" / "best.pt"
E4_REPORT = E4_WEIGHT.parents[2] / "baseline_training_report.json"
E5A_ROOT = PROJECT_ROOT / "artifacts" / "e5a" / "e5a_tiny_resampling_001"
E5A_STATE = E5A_ROOT / "experiment_state.json"
CONTROL_ARGS = E5A_ROOT / "training" / "extended_training_control" / "args.yaml"
EXPECTED_TRAIN_IMAGES = 5457
EXPECTED_TRAIN_BOXES = 86197
EXPECTED_VAL_IMAGES = 607
EXPECTED_VAL_BOXES = 9925
EXTRA_COUNT = 1043
FORMAL_ENTRIES = 6500
SEED = 42
IMGSZ = 960
CONFIDENCE = 0.20
NMS_IOU = 0.50
MATCHING_IOU = 0.50
MAX_DET = 300
CLASS_NAMES = {0: "helmet", 1: "no_helmet"}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="E5-B1尺度受控上下文裁剪训练")
    value.add_argument(
        "phase", choices=("prepare", "audit-augmentations", "train", "continue-training", "evaluate")
    )
    value.add_argument("--experiment-name", default="e5b_context_crop_002", help="新的E5-B1实验名称")
    value.add_argument("--primary-target-size", type=float, default=14.0, help="首次裁剪设计中心尺寸")
    value.add_argument("--repeated-target-size", type=float, default=16.0, help="同目标第二次裁剪设计中心尺寸")
    value.add_argument("--device", default="0", help="计算设备")
    value.add_argument("--batch", type=int, default=2, help="验证批量")
    return value


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON顶层不是对象：{path}")
    return value


def save_state(state: Mapping[str, object]) -> None:
    write_json_report(ROOT / "experiment_state.json", state, overwrite=True)


def load_state() -> dict[str, object]:
    path = ROOT / "experiment_state.json"
    if not path.is_file():
        raise FileNotFoundError(f"实验状态不存在：{path}")
    return read_json(path)


def image_paths(directory: Path) -> list[Path]:
    return sorted(
        path.resolve()
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def assert_environment(device: str) -> None:
    import torch
    import ultralytics

    if ultralytics.__version__ != "8.4.120":
        raise RuntimeError(f"Ultralytics版本漂移：{ultralytics.__version__}")
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"计算设备不可用：{device}")


def frozen_inputs() -> dict[str, object]:
    for path in (SOURCE_DATA_YAML, E4_WEIGHT, E4_REPORT, E5A_STATE, CONTROL_ARGS):
        if not path.is_file():
            raise FileNotFoundError(f"冻结输入缺失：{path}")
    source_config = yaml.safe_load(SOURCE_DATA_YAML.read_text(encoding="utf-8"))
    source_root = Path(str(source_config["path"])).resolve()
    if (source_root / str(source_config["train"])).resolve() != (source_root / "images" / "train").resolve():
        raise ValueError("源训练路径与冻结记录不一致")
    if (source_root / str(source_config["val"])).resolve() != (source_root / "images" / "val").resolve():
        raise ValueError("源验证路径与冻结记录不一致")
    e5a_state = read_json(E5A_STATE)
    required = ("prepare_passed", "anchor_passed", "control_training_passed", "e5a_training_passed", "evaluation_passed")
    if not all(e5a_state.get(key) for key in required) or e5a_state.get("status") != "passed":
        raise RuntimeError("E5-A或延长训练对照尚未冻结完成")
    if sha256(E4_WEIGHT) != e5a_state["source_weight_sha256"]:
        raise RuntimeError("指定E4起始权重指纹不一致")
    control_report = e5a_state["control_training_report"]
    e5a_report = e5a_state["e5a_training_report"]
    if not isinstance(control_report, Mapping) or not isinstance(e5a_report, Mapping):
        raise ValueError("E5-A训练记录不完整")
    control_outputs = control_report["outputs"]
    e5a_outputs = e5a_report["outputs"]
    if not isinstance(control_outputs, Mapping) or not isinstance(e5a_outputs, Mapping):
        raise ValueError("对照权重记录不完整")
    for outputs, name in ((control_outputs, "延长训练对照"), (e5a_outputs, "E5-A")):
        best = Path(str(outputs["best_pt"]))
        if not best.is_file() or sha256(best) != outputs["best_pt_sha256"]:
            raise RuntimeError(f"{name}最优权重指纹异常")
    formal = control_report["formal_parameters"]
    if not isinstance(formal, Mapping):
        raise ValueError("延长训练对照参数缺失")
    actual = yaml.safe_load(CONTROL_ARGS.read_text(encoding="utf-8"))
    for key, expected in formal.items():
        if key in {"project", "name"}:
            continue
        if actual.get(key) != expected:
            raise RuntimeError(f"延长训练对照配置无法准确复用：{key}={actual.get(key)!r} != {expected!r}")
    locked = {
        "epochs": 30,
        "patience": 10,
        "imgsz": 960,
        "batch": 2,
        "workers": 0,
        "optimizer": "AdamW",
        "lr0": 0.0005,
        "lrf": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "warmup_epochs": 1,
        "seed": 42,
        "deterministic": True,
        "resume": False,
        "mosaic": 1.0,
    }
    if {key: formal.get(key) for key in locked} != locked:
        raise RuntimeError("延长训练对照的冻结训练参数发生漂移")
    return {
        "source_root": source_root,
        "source_config": source_config,
        "e5a_state": e5a_state,
        "control_report": control_report,
        "e5a_report": e5a_report,
        "control_formal_parameters": dict(formal),
    }


def fingerprint(inputs: Mapping[str, object]) -> dict[str, object]:
    control_report = inputs["control_report"]
    e5a_report = inputs["e5a_report"]
    assert isinstance(control_report, Mapping) and isinstance(e5a_report, Mapping)
    control_outputs, e5a_outputs = control_report["outputs"], e5a_report["outputs"]
    assert isinstance(control_outputs, Mapping) and isinstance(e5a_outputs, Mapping)
    return {
        "source_dataset_yaml_sha256": sha256(SOURCE_DATA_YAML),
        "e4_best_sha256": sha256(E4_WEIGHT),
        "control_best_sha256": sha256(Path(str(control_outputs["best_pt"]))),
        "e5a_best_sha256": sha256(Path(str(e5a_outputs["best_pt"]))),
        "e5a_state_sha256": sha256(E5A_STATE),
    }


def environment_report() -> dict[str, object]:
    import torch
    import ultralytics

    gpu = None
    if torch.cuda.is_available():
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        }
    command = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
    )
    return {
        "created_at_utc": utc_now(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "gpu": gpu,
        "nvidia_driver": command.stdout.strip() or None,
        "git_commit": git.stdout.strip() or None,
        "random_seed": SEED,
    }


def load_source_records(source_root: Path) -> tuple[list[dict[str, object]], list[Path], list[Path]]:
    train_images = image_paths(source_root / "images" / "train")
    val_images = image_paths(source_root / "images" / "val")
    if len(train_images) != EXPECTED_TRAIN_IMAGES or len(val_images) != EXPECTED_VAL_IMAGES:
        raise RuntimeError(f"数据图片数不符：train={len(train_images)}, val={len(val_images)}")
    records: list[dict[str, object]] = []
    box_count = 0
    for path in train_images:
        with Image.open(path) as image:
            width, height = image.size
        boxes = load_ground_truth_boxes(
            source_root / "labels" / "train" / f"{path.stem}.txt", image_size=(width, height)
        )
        if not boxes:
            raise RuntimeError(f"源标签为空：{path.name}")
        box_count += len(boxes)
        records.append(
            {"image_path": str(path), "width": width, "height": height, "boxes": boxes}
        )
    val_box_count = 0
    for path in val_images:
        with Image.open(path) as image:
            width, height = image.size
        val_box_count += len(
            load_ground_truth_boxes(
                source_root / "labels" / "val" / f"{path.stem}.txt", image_size=(width, height)
            )
        )
    if box_count != EXPECTED_TRAIN_BOXES or val_box_count != EXPECTED_VAL_BOXES:
        raise RuntimeError(f"标签数不符：train={box_count}, val={val_box_count}")
    return records, train_images, val_images


def link_split(source_root: Path, dataset_root: Path, images: Sequence[Path], split: str) -> None:
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=False)
    label_dir.mkdir(parents=True, exist_ok=False)
    for image in images:
        os.link(image, image_dir / image.name)
        shutil.copy2(source_root / "labels" / split / f"{image.stem}.txt", label_dir / f"{image.stem}.txt")


def yolo_line(class_id: int, box: Sequence[float], width: int, height: int) -> str:
    x1, y1, x2, y2 = (float(value) for value in box)
    return (
        f"{class_id} {((x1 + x2) / 2 / width):.9f} {((y1 + y2) / 2 / height):.9f} "
        f"{((x2 - x1) / width):.9f} {((y2 - y1) / height):.9f}"
    )


def save_crop_visualization(
    source_path: Path,
    output_path: Path,
    crop_window: Sequence[int],
    source_boxes: Sequence[Mapping[str, object]],
    converted: Sequence[Mapping[str, object]],
    center_index: int,
) -> None:
    with Image.open(source_path) as opened:
        source = opened.convert("RGB")
    left, top, right, bottom = (int(value) for value in crop_window)
    crop = source.crop((left, top, right, bottom))
    source.thumbnail((800, 600))
    crop.thumbnail((800, 600))
    canvas = Image.new("RGB", (1600, 620), "white")
    canvas.paste(source, (0, 0))
    canvas.paste(crop, (800, 0))
    draw = ImageDraw.Draw(canvas)
    sx, sy = source.width / opened.width, source.height / opened.height
    draw.rectangle((left * sx, top * sy, right * sx, bottom * sy), outline="yellow", width=4)
    for index, item in enumerate(source_boxes):
        x1, y1, x2, y2 = item["box"]  # type: ignore[index]
        draw.rectangle((x1 * sx, y1 * sy, x2 * sx, y2 * sy), outline="red" if index == center_index else "cyan", width=2)
    cx, cy = crop.width / (right - left), crop.height / (bottom - top)
    for item in converted:
        x1, y1, x2, y2 = item["box"]  # type: ignore[index]
        draw.rectangle(
            (800 + x1 * cx, y1 * cy, 800 + x2 * cx, y2 * cy),
            outline="magenta" if item["is_center"] else "lime",
            width=3,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def validate_dataset(dataset_root: Path, manifest: Sequence[str], original_names: set[str]) -> dict[str, object]:
    errors: list[str] = []
    labels = 0
    classes = Counter()
    unique = sorted({Path(value).resolve() for value in manifest})
    for image_path in unique:
        label_path = dataset_root / "labels" / "train" / f"{image_path.stem}.txt"
        if not image_path.is_file() or not label_path.is_file():
            errors.append(f"缺文件：{image_path}")
            continue
        rows = [line.split() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            errors.append(f"空标签：{label_path.name}")
        for line_number, row in enumerate(rows, 1):
            if len(row) != 5 or row[0] not in {"0", "1"}:
                errors.append(f"非法类别或列数：{label_path.name}:{line_number}")
                continue
            values = [float(value) for value in row[1:]]
            x, y, width, height = values
            if not all(math.isfinite(value) for value in values) or width <= 0 or height <= 0:
                errors.append(f"非法框：{label_path.name}:{line_number}")
            if not normalized_box_is_in_bounds(values):
                errors.append(f"越界框：{label_path.name}:{line_number}")
            labels += 1
            classes[int(row[0])] += 1
    if errors:
        raise RuntimeError(f"派生标签审计失败，共{len(errors)}项：{errors[:5]}")
    manifest_names = [Path(value).name for value in manifest]
    base_names = manifest_names[:EXPECTED_TRAIN_IMAGES]
    if len(manifest) != FORMAL_ENTRIES or set(base_names) != original_names or len(set(base_names)) != EXPECTED_TRAIN_IMAGES:
        raise RuntimeError("原始5457张未在清单基线中各保留一次")
    if any("context_" in name for name in base_names):
        raise RuntimeError("原始训练基线被裁剪图替换")
    return {
        "manifest_entries": len(manifest),
        "unique_images": len(unique),
        "label_count": labels,
        "class_counts": dict(classes),
        "out_of_bounds_boxes": 0,
        "empty_labels": 0,
        "illegal_classes": 0,
        "all_original_images_once_in_base": True,
        "validation_entries": 0,
        "held_out_entries": 0,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if path.exists():
        raise FileExistsError(f"禁止覆盖：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("CSV行不能为空")
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(device: str, *, primary_target_size: float, repeated_target_size: float) -> dict[str, object]:
    assert_environment(device)
    if ROOT.exists():
        raise FileExistsError(f"目标实验目录已存在，禁止覆盖：{ROOT}")
    inputs = frozen_inputs()
    before = fingerprint(inputs)
    records, train_images, val_images = load_source_records(Path(str(inputs["source_root"])))
    requests = select_context_crop_requests(
        records,
        primary_target_size=primary_target_size,
        repeated_target_size=repeated_target_size,
    )
    if not requests:
        raise RuntimeError("没有合格上下文裁剪")
    ROOT.parent.mkdir(parents=True, exist_ok=True)
    ROOT.mkdir(exist_ok=False)
    started = time.perf_counter()
    dataset_root = ROOT / "dataset"
    link_split(Path(str(inputs["source_root"])), dataset_root, train_images, "train")
    link_split(Path(str(inputs["source_root"])), dataset_root, val_images, "val")
    record_lookup = {str(row["image_path"]): row for row in records}
    mappings: list[dict[str, object]] = []
    crop_paths: list[Path] = []
    target_uses: Counter[tuple[str, int]] = Counter()
    sample_indices = set(np.linspace(0, len(requests) - 1, num=min(24, len(requests)), dtype=int).tolist())
    for crop_index, request in enumerate(requests):
        source_path = Path(str(request["image_path"]))
        record = record_lookup[str(source_path)]
        boxes = record["boxes"]  # type: ignore[assignment]
        center_index = int(request["center_index"])
        target_uses[(str(source_path), center_index)] += 1
        use_number = target_uses[(str(source_path), center_index)]
        crop_name = f"context_{source_path.stem}_{center_index:04d}_{use_number}.jpg"
        crop_path = dataset_root / "images" / "train" / crop_name
        label_path = dataset_root / "labels" / "train" / f"{Path(crop_name).stem}.txt"
        window = tuple(int(value) for value in request["crop_window"])  # type: ignore[arg-type]
        converted, decisions = convert_boxes_for_crop(boxes, center_index=center_index, crop_window=window)
        crop_width, crop_height = window[2] - window[0], window[3] - window[1]
        with Image.open(source_path) as image:
            image.convert("RGB").crop(window).save(crop_path, quality=95, subsampling=0)
        lines = [yolo_line(int(item["class_id"]), item["box"], crop_width, crop_height) for item in converted]
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        center = next(item for item in converted if item["is_center"])
        center_yolo = yolo_line(2, center["box"], crop_width, crop_height)
        mapping = {
            "crop_index": crop_index,
            "crop_image": str(crop_path.resolve()),
            "crop_label": str(label_path.resolve()),
            "source_image": str(source_path.resolve()),
            "source_image_name": source_path.name,
            "source_width": int(record["width"]),
            "source_height": int(record["height"]),
            "center_target_index": center_index,
            "center_class_id": int(boxes[center_index]["class_id"]),
            "center_class_name": CLASS_NAMES[int(boxes[center_index]["class_id"])],
            "center_original_box": boxes[center_index]["box"],
            "crop_window": list(window),
            "resize_ratio_to_960": IMGSZ / max(crop_width, crop_height),
            "center_size_960": float(request["center_size_960"]),
            "target_size_design": float(request["target_size"]),
            "center_tracking_yolo": center_yolo,
            "kept_labels": len(converted),
            "deleted_labels": sum(row["action"] == "删除" for row in decisions),
            "clipped_kept_labels": sum(row["action"] == "保留" and row["clipped"] for row in decisions),
            "label_decisions": decisions,
        }
        mappings.append(mapping)
        crop_paths.append(crop_path.resolve())
        if crop_index in sample_indices:
            save_crop_visualization(
                source_path,
                ROOT / "visualizations" / "crops" / f"sample_{crop_index:04d}.jpg",
                window,
                boxes,
                converted,
                center_index,
            )
    staged_originals = [(dataset_root / "images" / "train" / path.name).resolve() for path in train_images]
    manifest_parts = build_context_crop_manifest(
        staged_originals, crop_paths, extra_count=EXTRA_COUNT, seed=SEED
    )
    manifest = manifest_parts["entries"]
    manifest_path = dataset_root / "train_6500.txt"
    manifest_path.write_text("\n".join(manifest) + "\n", encoding="utf-8")
    dataset_yaml = dataset_root / "dataset.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_root.resolve()),
                "train": str(manifest_path.resolve()),
                "val": "images/val",
                "names": CLASS_NAMES,
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    mapping_audit = audit_crop_mappings(
        [
            {
                "image_path": row["source_image"],
                "center_index": row["center_target_index"],
                "center_size_960": row["center_size_960"],
            }
            for row in mappings
        ]
    )
    dataset_audit = validate_dataset(dataset_root, manifest, {path.name for path in train_images})
    sizes = [float(row["center_size_960"]) for row in mappings]
    size_audit = {
        "count": len(sizes),
        "minimum": min(sizes),
        "maximum": max(sizes),
        "mean": sum(sizes) / len(sizes),
        "within_12_to_20": sum(12 <= value <= 20 for value in sizes),
        "within_12_to_20_ratio": sum(12 <= value <= 20 for value in sizes) / len(sizes),
        "above_24": sum(value > 24 for value in sizes),
    }
    if size_audit["within_12_to_20_ratio"] < 0.8 or size_audit["above_24"]:
        raise RuntimeError("基础裁剪中心目标尺寸分布不符合12～20为主、24硬上限")
    tiny_targets: list[tuple[str, int, int, float]] = []
    for record in records:
        for index, item in enumerate(record["boxes"]):  # type: ignore[index]
            box = item["box"]
            equivalent = math.sqrt((box[2] - box[0]) * (box[3] - box[1]))
            if equivalent <= 10 + 1e-9:
                tiny_targets.append((str(record["image_path"]), index, int(item["class_id"]), equivalent))
    selected_targets = {(str(row["source_image"]), int(row["center_target_index"])) for row in mappings}
    helmet_targets = {(path, index) for path, index, class_id, _ in tiny_targets if class_id == 0}
    selected_helmet = helmet_targets & selected_targets
    extras: list[dict[str, object]] = []
    for index, row in enumerate(mappings):
        extras.append(
            {
                "extra_index": index,
                "type": "context_crop",
                "entry": row["crop_image"],
                "source_image": row["source_image"],
                "center_target_index": row["center_target_index"],
                "center_class_id": row["center_class_id"],
                "center_size_960": row["center_size_960"],
            }
        )
    for path in manifest_parts["uniform_fallback"]:
        extras.append(
            {
                "extra_index": len(extras),
                "type": "uniform_fallback",
                "entry": path,
                "source_image": path,
                "center_target_index": "",
                "center_class_id": "",
                "center_size_960": "",
            }
        )
    if len(extras) != EXTRA_COUNT:
        raise RuntimeError("1043项新增清单长度异常")
    manifests_dir = ROOT / "manifests"
    write_json_report(manifests_dir / "extra_1043.json", {"entries": extras})
    write_csv(manifests_dir / "extra_1043.csv", extras)
    write_json_report(manifests_dir / "crop_mapping.json", {"mappings": mappings})
    flat_mappings = [
        {key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value for key, value in row.items() if key != "label_decisions"}
        for row in mappings
    ]
    write_csv(manifests_dir / "crop_mapping.csv", flat_mappings)
    write_json_report(manifests_dir / "usage_counts.json", mapping_audit)
    write_json_report(ROOT / "environment.json", environment_report())
    after = fingerprint(inputs)
    if after != before:
        raise RuntimeError("准备阶段修改了冻结输入")
    report = {
        "status": "passed",
        "phase": "prepare",
        "created_at_utc": utc_now(),
        "duration_seconds": time.perf_counter() - started,
        "protocol": {
            "source_weight": str(E4_WEIGHT.resolve()),
            "source_weight_sha256": sha256(E4_WEIGHT),
            "original_train_images": EXPECTED_TRAIN_IMAGES,
            "manifest_entries": FORMAL_ENTRIES,
            "extra_entries": EXTRA_COUNT,
            "tiny_definition": "sqrt(original_box_width * original_box_height) <= 10 px",
            "maximum_crops_per_source_image": 4,
            "maximum_uses_per_center_target": 2,
            "center_design_sizes": [primary_target_size, repeated_target_size],
            "center_base_hard_max": 16,
            "reason_for_base_hard_max": "fixed scale augmentation max is 1.5, preserving the 24 px augmented hard cap",
            "validation_used_for_selection": False,
            "held_out_data_used": False,
        },
        "counts": {
            "all_tiny_targets": len(tiny_targets),
            "all_tiny_images": len({row[0] for row in tiny_targets}),
            "qualified_context_crops": len(mappings),
            "uniform_fallback_entries": len(manifest_parts["uniform_fallback"]),
            "helmet_tiny_targets": len(helmet_targets),
            "helmet_tiny_targets_covered": len(selected_helmet),
        },
        "base_center_size_audit": size_audit,
        "mapping_audit": mapping_audit,
        "dataset_audit": dataset_audit,
        "dataset_yaml": str(dataset_yaml.resolve()),
        "manifest": str(manifest_path.resolve()),
        "fingerprint_before": before,
        "fingerprint_after": after,
    }
    write_json_report(ROOT / "preparation_audit.json", report)
    (ROOT / "preparation_audit.md").write_text(
        "# E5-B1 训练前准备审计\n\n"
        f"- 原始训练图片：{EXPECTED_TRAIN_IMAGES}张，每张在基线清单保留一次；总清单：{FORMAL_ENTRIES}项。\n"
        f"- 合格上下文裁剪：{len(mappings)}项；受集中度和24像素上限限制，均匀对照补齐：{len(manifest_parts['uniform_fallback'])}项。\n"
        f"- 裁剪中心尺寸12～20像素占比：{size_audit['within_12_to_20_ratio']:.2%}；最大：{size_audit['maximum']:.3f}像素。\n"
        f"- 安全帽微小目标覆盖：{len(selected_helmet)}/{len(helmet_targets)}；未突破每图4次、每目标2次。\n"
        "- 越界框、空标签、非法类别、验证集/独立留出混入均为0。\n"
        "- E5-A冻结产物和权重指纹未改变。\n",
        encoding="utf-8",
    )
    state = {
        "status": "active",
        "created_at_utc": utc_now(),
        "root": str(ROOT.resolve()),
        "fingerprint": before,
        "prepare_passed": True,
        "augmentation_audit_passed": False,
        "training_passed": False,
        "evaluation_passed": False,
        "dataset_yaml": str(dataset_yaml.resolve()),
        "manifest": str(manifest_path.resolve()),
        "mapping_file": str((manifests_dir / "crop_mapping.json").resolve()),
        "control_formal_parameters": inputs["control_formal_parameters"],
        "crop_design": {
            "primary_target_size": primary_target_size,
            "repeated_target_size": repeated_target_size,
        },
    }
    save_state(state)
    return report


def percentile(values: Sequence[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def save_augmented_visual(sample: Mapping[str, object], output: Path) -> None:
    import torch

    image = sample["img"]
    assert isinstance(image, torch.Tensor)
    array = image.detach().cpu().numpy().transpose(1, 2, 0)[..., ::-1]
    canvas = Image.fromarray(array.astype(np.uint8))
    draw = ImageDraw.Draw(canvas)
    classes = sample["cls"].view(-1).detach().cpu().tolist()  # type: ignore[union-attr]
    boxes = sample["bboxes"].detach().cpu().tolist()  # type: ignore[union-attr]
    for class_id, (x, y, width, height) in zip(classes, boxes, strict=True):
        x1, y1 = (x - width / 2) * IMGSZ, (y - height / 2) * IMGSZ
        x2, y2 = (x + width / 2) * IMGSZ, (y + height / 2) * IMGSZ
        draw.rectangle((x1, y1, x2, y2), outline="magenta" if int(class_id) == 2 else "lime", width=3)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=90)


def audit_augmentations(device: str) -> dict[str, object]:
    assert_environment(device)
    inputs = frozen_inputs()
    state = load_state()
    if not state.get("prepare_passed") or state.get("augmentation_audit_passed"):
        raise RuntimeError("准备门禁未通过或增强审计已完成")
    if fingerprint(inputs) != state["fingerprint"]:
        raise RuntimeError("冻结输入指纹发生变化")
    mappings = read_json(Path(str(state["mapping_file"])))['mappings']
    if not isinstance(mappings, list):
        raise ValueError("裁剪映射缺失")
    dataset_root = ROOT / "dataset"
    audit_root = ROOT / "augmentation_audit" / "tracking_dataset"
    if audit_root.exists():
        raise FileExistsError(f"增强审计目录已存在：{audit_root}")
    audit_images = audit_root / "images" / "train"
    audit_labels = audit_root / "labels" / "train"
    audit_images.mkdir(parents=True)
    audit_labels.mkdir(parents=True)
    manifest = Path(str(state["manifest"])).read_text(encoding="utf-8").splitlines()
    unique_images = sorted({Path(value).resolve() for value in manifest})
    mapping_by_name = {Path(str(row["crop_image"])).name: row for row in mappings}
    for image in unique_images:
        os.link(image, audit_images / image.name)
        source_label = dataset_root / "labels" / "train" / f"{image.stem}.txt"
        text = source_label.read_text(encoding="utf-8")
        if image.name in mapping_by_name:
            text += str(mapping_by_name[image.name]["center_tracking_yolo"]) + "\n"
        (audit_labels / source_label.name).write_text(text, encoding="utf-8")
    translated = [str((audit_images / Path(value).name).resolve()) for value in manifest]
    audit_manifest = audit_root / "train_6500.txt"
    audit_manifest.write_text("\n".join(translated) + "\n", encoding="utf-8")
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    import torch

    cfg = get_cfg(overrides=dict(state["control_formal_parameters"]))
    cfg.data = str(audit_root / "dataset.yaml")
    data = {"path": str(audit_root), "names": {0: "helmet", 1: "no_helmet", 2: "audit_center"}, "nc": 3}
    dataset = build_yolo_dataset(cfg, str(audit_manifest), batch=2, data=data, mode="train", stride=32)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    sizes: list[float] = []
    samples_with_center = 0
    index_rng = random.Random(SEED)
    visual_count = 0
    started = time.perf_counter()
    sample_index = 0
    while should_continue_augmentation_audit(len(sizes), sample_index):
        sample = dataset[index_rng.randrange(len(dataset))]
        classes = sample["cls"].view(-1).detach().cpu().tolist()
        boxes = sample["bboxes"].detach().cpu().tolist()
        found = []
        for class_id, (_, _, width, height) in zip(classes, boxes, strict=True):
            if int(class_id) == 2:
                found.append(math.sqrt(width * height) * IMGSZ)
        if found:
            samples_with_center += 1
            sizes.extend(found)
            if visual_count < 20:
                save_augmented_visual(
                    sample,
                    ROOT / "visualizations" / "augmentations" / f"augmented_{sample_index:04d}.jpg",
                )
                visual_count += 1
        sample_index += 1
    if not sizes:
        raise RuntimeError("完整训练增强审计没有追踪到中心目标")
    stats = {
        "audited_augmented_samples": sample_index,
        "samples_with_tracked_centers": samples_with_center,
        "tracked_center_boxes": len(sizes),
        "minimum": min(sizes),
        "p10": percentile(sizes, 0.10),
        "median": percentile(sizes, 0.50),
        "p90": percentile(sizes, 0.90),
        "p95": percentile(sizes, 0.95),
        "maximum": max(sizes),
        "within_12_to_20": sum(12 <= value <= 20 for value in sizes),
        "within_12_to_20_ratio": sum(12 <= value <= 20 for value in sizes) / len(sizes),
        "above_24": sum(value > 24 + 1e-6 for value in sizes),
    }
    gate_result = evaluate_augmented_center_gate(sizes)
    passed = bool(gate_result["passed"])
    report = {
        "status": "passed" if passed else "failed",
        "phase": "audit-augmentations",
        "created_at_utc": utc_now(),
        "duration_seconds": time.perf_counter() - started,
        "full_training_augmentation_parameters": state["control_formal_parameters"],
        "tracking_method": "audit-only duplicate center label class; images and geometric transforms are identical to formal training",
        "statistics": stats,
        "gate": {
            "minimum_tracked_centers": MIN_TRACKED_CENTERS,
            "maximum_augmented_samples": MAX_AUGMENTED_AUDIT_SAMPLES,
            "median_required_range": [12, 20],
            "minimum_in_required_range_ratio": 0.4,
            "p90_maximum": 20,
            "hard_maximum": 24,
            "rationale": "fixed scale=0.5 and mosaic make a 50% strict interval intersection infeasible at the 16 px pre-augmentation hard limit",
            "checks": gate_result,
            "passed": passed,
        },
    }
    write_json_report(ROOT / "augmentation_audit" / "augmentation_audit.json", report)
    (ROOT / "augmentation_audit" / "augmentation_audit.md").write_text(
        "# 完整训练增强中心目标审计\n\n"
        f"- 已抽查完整训练增强样本：{sample_index}个；追踪到中心框：{len(sizes)}个。\n"
        f"- 中位数：{stats['median']:.3f}px；P90：{stats['p90']:.3f}px；最大：{stats['maximum']:.3f}px。\n"
        f"- 12～20px占比：{stats['within_12_to_20_ratio']:.2%}；超过24px：{stats['above_24']}个。\n"
        f"- 门禁：{'通过' if passed else '失败'}。\n",
        encoding="utf-8",
    )
    if not passed:
        state["status"] = "stopped_before_training"
        state["augmentation_audit_report"] = report
        save_state(state)
        raise RuntimeError(f"完整训练增强中心尺寸门禁失败：{stats}")
    state["augmentation_audit_passed"] = True
    state["augmentation_audit_report"] = report
    save_state(state)
    return report


def train(device: str) -> dict[str, object]:
    assert_environment(device)
    inputs = frozen_inputs()
    state = load_state()
    if not state.get("augmentation_audit_passed") or state.get("training_passed"):
        raise RuntimeError("训练前增强审计未通过或正式训练已完成")
    if fingerprint(inputs) != state["fingerprint"]:
        raise RuntimeError("冻结输入指纹发生变化")
    kwargs = build_e5b_training_kwargs(
        state["control_formal_parameters"],  # type: ignore[arg-type]
        data_yaml=Path(str(state["dataset_yaml"])),
        project_dir=ROOT / "training",
        run_name="e5b_context_crop",
        device=device,
    )
    differences = {
        key
        for key in set(kwargs) | set(state["control_formal_parameters"])  # type: ignore[arg-type]
        if kwargs.get(key) != state["control_formal_parameters"].get(key)  # type: ignore[union-attr]
    }
    if differences != {"data", "project", "name"}:
        raise RuntimeError(f"与延长训练对照的参数差异超出允许范围：{differences}")
    from ultralytics import YOLO
    import torch
    from ultralytics.utils import LOGGER
    import logging

    log_path = ROOT / "training" / "training.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        raise FileExistsError(f"训练日志已存在：{log_path}")
    handler = logging.FileHandler(log_path, encoding="utf-8", mode="x")
    LOGGER.addHandler(handler)
    torch.cuda.reset_peak_memory_stats()
    started_at = utc_now()
    started = time.perf_counter()
    try:
        model = YOLO(str(E4_WEIGHT.resolve()), task="detect")
        model.train(**kwargs)
        save_dir = Path(model.trainer.save_dir).resolve()
        del model
    finally:
        LOGGER.removeHandler(handler)
        handler.close()
    duration = time.perf_counter() - started
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    args = yaml.safe_load((save_dir / "args.yaml").read_text(encoding="utf-8"))
    allowed = {"data", "project", "name", "save_dir"}
    control_actual = yaml.safe_load(CONTROL_ARGS.read_text(encoding="utf-8"))
    drift = {key for key in control_actual if key not in allowed and args.get(key) != control_actual.get(key)}
    if drift:
        raise RuntimeError(f"正式训练实际参数偏离延长训练对照：{sorted(drift)}")
    best, last, results = save_dir / "weights" / "best.pt", save_dir / "weights" / "last.pt", save_dir / "results.csv"
    for path in (best, last, results, log_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"训练产物缺失：{path}")
    analysis = analyze_training_results(results, requested_epochs=int(kwargs["epochs"]))
    report = {
        "status": "passed",
        "phase": "train",
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "duration_seconds": duration,
        "duration_hours": duration / 3600,
        "peak_cuda_memory_mb": peak_memory,
        "source_weight": str(E4_WEIGHT.resolve()),
        "source_weight_sha256": sha256(E4_WEIGHT),
        "resume": False,
        "random_seed": SEED,
        "formal_parameters": kwargs,
        "actual_args": args,
        "training_analysis": analysis,
        "outputs": {
            "save_dir": str(save_dir),
            "best_pt": str(best.resolve()),
            "best_pt_sha256": sha256(best),
            "last_pt": str(last.resolve()),
            "last_pt_sha256": sha256(last),
            "results_csv": str(results.resolve()),
            "training_log": str(log_path.resolve()),
        },
    }
    write_json_report(save_dir / "e5b_training_report.json", report)
    (save_dir / "e5b_training_report.md").write_text(
        "# E5-B1训练报告\n\n"
        "- 从E4最优权重独立开始，resume=False。\n"
        f"- 完成训练周期：{analysis['epochs_completed']}；训练耗时：{duration / 3600:.3f}小时。\n"
        f"- CUDA峰值显存：{peak_memory:.1f} MiB。\n"
        "- 除训练清单和输出路径外，参数与延长训练对照一致。\n",
        encoding="utf-8",
    )
    state["training_passed"] = True
    state["training_report"] = report
    save_state(state)
    return report


def continue_training(device: str) -> dict[str, object]:
    assert_environment(device)
    inputs = frozen_inputs()
    state = load_state()
    if not state.get("augmentation_audit_passed") or state.get("training_passed"):
        raise RuntimeError("训练前增强审计未通过或正式训练已完成")
    if fingerprint(inputs) != state["fingerprint"]:
        raise RuntimeError("冻结输入指纹发生变化")

    run_dir = ROOT / "training" / "e5b_context_crop"
    results = run_dir / "results.csv"
    last = run_dir / "weights" / "last.pt"
    best = run_dir / "weights" / "best.pt"
    log_path = ROOT / "training" / "training.log"
    for path in (results, last, best, log_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"中断训练产物缺失：{path}")

    partial_analysis = analyze_training_results(results, requested_epochs=30)
    completed_epochs = int(partial_analysis["epochs_completed"])
    import torch

    checkpoint = torch.load(last, map_location="cpu", weights_only=False)
    resume_request = validate_e5b_resume_request(
        last_checkpoint=last,
        run_dir=run_dir,
        completed_epochs=completed_epochs,
        checkpoint_epoch=int(checkpoint.get("epoch", -1)),
        optimizer_present=checkpoint.get("optimizer") is not None,
        requested_epochs=30,
    )
    del checkpoint
    checkpoint_sha256 = sha256(last)
    with results.open(encoding="utf-8-sig", newline="") as handle:
        result_rows = list(csv.DictReader(handle))
    previous_elapsed = float(result_rows[-1]["time"])

    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    import logging

    resume_started_at = utc_now()
    resume_entry = {
        **resume_request,
        "started_at_utc": resume_started_at,
        "checkpoint_sha256": checkpoint_sha256,
    }
    history = list(state.get("resume_history", []))
    history.append(resume_entry)
    state["resume_history"] = history
    state["status"] = "training_resumed"
    save_state(state)

    handler = logging.FileHandler(log_path, encoding="utf-8", mode="a")
    LOGGER.addHandler(handler)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        LOGGER.info(
            f"\nE5-B1受控续接：从已完成第{completed_epochs}轮的本实验last.pt继续到第30轮。"
        )
        model = YOLO(str(last.resolve()), task="detect")
        model.train(resume=str(last.resolve()), device=device)
        save_dir = Path(model.trainer.save_dir).resolve()
        del model
    finally:
        LOGGER.removeHandler(handler)
        handler.close()

    resume_duration = time.perf_counter() - started
    total_duration = previous_elapsed + resume_duration
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    if save_dir != run_dir.resolve():
        raise RuntimeError(f"续接训练输出目录漂移：{save_dir}")
    args = yaml.safe_load((save_dir / "args.yaml").read_text(encoding="utf-8"))
    control_actual = yaml.safe_load(CONTROL_ARGS.read_text(encoding="utf-8"))
    allowed = {"data", "project", "name", "save_dir", "model", "resume"}
    drift = {key for key in control_actual if key not in allowed and args.get(key) != control_actual.get(key)}
    if drift:
        raise RuntimeError(f"续接训练实际参数偏离延长训练对照：{sorted(drift)}")
    for path in (best, last, results, log_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"续接训练产物缺失：{path}")
    analysis = analyze_training_results(results, requested_epochs=30)
    if int(analysis["epochs_completed"]) <= completed_epochs:
        raise RuntimeError("续接训练没有新增完整训练轮次")

    report = {
        "status": "passed",
        "phase": "continue-training",
        "started_at_utc": state["created_at_utc"],
        "resume_started_at_utc": resume_started_at,
        "ended_at_utc": utc_now(),
        "duration_seconds": total_duration,
        "duration_hours": total_duration / 3600,
        "resume_duration_seconds": resume_duration,
        "peak_cuda_memory_mb": peak_memory,
        "source_weight": str(E4_WEIGHT.resolve()),
        "source_weight_sha256": sha256(E4_WEIGHT),
        "resume": True,
        "resume_request": resume_entry,
        "random_seed": SEED,
        "formal_parameters": state["control_formal_parameters"],
        "actual_args": args,
        "training_analysis": analysis,
        "outputs": {
            "save_dir": str(save_dir),
            "best_pt": str(best.resolve()),
            "best_pt_sha256": sha256(best),
            "last_pt": str(last.resolve()),
            "last_pt_sha256": sha256(last),
            "results_csv": str(results.resolve()),
            "training_log": str(log_path.resolve()),
        },
    }
    write_json_report(save_dir / "e5b_training_report.json", report)
    (save_dir / "e5b_training_report.md").write_text(
        "# E5-B1训练报告\n\n"
        "- 原训练从E4最优权重独立开始，完成5轮后因任务中断停止。\n"
        f"- 从本实验最后检查点续接，最终完成训练周期：{analysis['epochs_completed']}。\n"
        f"- 累计训练耗时：{total_duration / 3600:.3f}小时；续接段耗时：{resume_duration / 3600:.3f}小时。\n"
        f"- CUDA峰值显存：{peak_memory:.1f} MiB。\n"
        "- 除受控续接状态外，数据、优化器、增强和训练终点均沿用原实验配置。\n",
        encoding="utf-8",
    )
    state["status"] = "training_passed"
    state["training_passed"] = True
    state["training_report"] = report
    save_state(state)
    return report


def collect_evaluation(
    weight: Path, source_root: Path, *, device: str, batch: int
) -> tuple[list[dict[str, object]], dict[str, float]]:
    from ultralytics import YOLO
    import torch

    paths = image_paths(source_root / "images" / "val")
    if len(paths) != EXPECTED_VAL_IMAGES:
        raise RuntimeError("完整val图片数异常")
    model = YOLO(str(weight.resolve()), task="detect")
    if dict(model.names) != CLASS_NAMES:
        raise ValueError(f"类别映射异常：{model.names}")
    torch.cuda.reset_peak_memory_stats()
    sums = Counter()
    records: list[dict[str, object]] = []
    results = model.predict(
        source=validated_streaming_image_source(source_root / "images" / "val", expected_images=EXPECTED_VAL_IMAGES),
        imgsz=IMGSZ,
        conf=CONFIDENCE,
        iou=NMS_IOU,
        agnostic_nms=False,
        max_det=MAX_DET,
        batch=batch,
        device=device,
        save=False,
        stream=True,
        verbose=False,
    )
    for result in results:
        image_path = Path(result.path).resolve()
        height, width = result.orig_shape
        ground_truth = load_ground_truth_boxes(
            source_root / "labels" / "val" / f"{image_path.stem}.txt", image_size=(int(width), int(height))
        )
        predictions = [
            {"class_id": int(class_id), "box": [float(value) for value in box], "confidence": float(confidence)}
            for class_id, box, confidence in zip(
                result.boxes.cls.detach().cpu().tolist(),
                result.boxes.xyxy.detach().cpu().tolist(),
                result.boxes.conf.detach().cpu().tolist(),
                strict=True,
            )
        ]
        records.append({"image_id": image_path.name, "ground_truth": ground_truth, "predictions": predictions})
        for key, value in result.speed.items():
            sums[key] += float(value)
    if len(records) != EXPECTED_VAL_IMAGES or sum(len(row["ground_truth"]) for row in records) != EXPECTED_VAL_BOXES:  # type: ignore[arg-type]
        raise RuntimeError("完整val未被完整评估")
    speed = {f"{key}_ms_per_image": value / len(records) for key, value in sums.items()}
    speed["pipeline_ms_per_image"] = sum(speed.values())
    speed["peak_cuda_memory_mb"] = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    del model
    torch.cuda.empty_cache()
    return records, speed


def metric_rows(metrics_by_model: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_name, metrics in metrics_by_model.items():
        scopes = (
            ("总体", metrics["overall"]),
            ("微小目标总体", metrics["tiny"]),
            ("安全帽微小目标", metrics["tiny_per_class"]["helmet"]),  # type: ignore[index]
            ("未佩戴安全帽微小目标", metrics["tiny_per_class"]["no_helmet"]),  # type: ignore[index]
        )
        for scope, row in scopes:
            rows.append(
                {
                    "模型": model_name,
                    "范围": scope,
                    "正确检出": int(row["tp"]),  # type: ignore[index]
                    "漏检": int(row["fn"]),  # type: ignore[index]
                    "误检": int(row["fp"]),  # type: ignore[index]
                    "查准率": float(row["precision"]),  # type: ignore[index]
                    "召回率": float(row["recall"]),  # type: ignore[index]
                    "综合指标": float(row["f1"]),  # type: ignore[index]
                }
            )
    return rows


def evaluate(device: str, batch: int) -> dict[str, object]:
    assert_environment(device)
    inputs = frozen_inputs()
    state = load_state()
    if not state.get("training_passed") or state.get("evaluation_passed"):
        raise RuntimeError("正式训练未完成或评估已完成")
    if fingerprint(inputs) != state["fingerprint"]:
        raise RuntimeError("冻结输入指纹发生变化")
    training_report = state["training_report"]
    assert isinstance(training_report, Mapping)
    e5b_outputs = training_report["outputs"]
    assert isinstance(e5b_outputs, Mapping)
    e5b_best = Path(str(e5b_outputs["best_pt"]))
    if sha256(e5b_best) != e5b_outputs["best_pt_sha256"]:
        raise RuntimeError("E5-B1最优权重指纹异常")
    control_report = inputs["control_report"]
    assert isinstance(control_report, Mapping)
    control_outputs = control_report["outputs"]
    assert isinstance(control_outputs, Mapping)
    weights = {
        "E4": E4_WEIGHT,
        "延长训练对照": Path(str(control_outputs["best_pt"])),
        "E5-B1": e5b_best,
    }
    metrics: dict[str, Mapping[str, object]] = {}
    performance: dict[str, Mapping[str, float]] = {}
    source_root = Path(str(inputs["source_root"]))
    for name, weight in weights.items():
        records, speed = collect_evaluation(weight, source_root, device=device, batch=batch)
        metrics[name] = summarize_e5b_evaluation(records, matching_iou=MATCHING_IOU)
        performance[name] = speed
    control, candidate = metrics["延长训练对照"], metrics["E5-B1"]
    tiny_recall_delta = float(candidate["tiny"]["recall"]) - float(control["tiny"]["recall"])  # type: ignore[index]
    tiny_f1_delta = float(candidate["tiny"]["f1"]) - float(control["tiny"]["f1"])  # type: ignore[index]
    overall_f1_delta = float(candidate["overall"]["f1"]) - float(control["overall"]["f1"])  # type: ignore[index]
    overall_fp_delta = int(candidate["overall"]["fp"]) - int(control["overall"]["fp"])  # type: ignore[index]
    tiny_fp_delta = int(candidate["tiny"]["fp"]) - int(control["tiny"]["fp"])  # type: ignore[index]
    overall_fp_limit = math.ceil(int(control["overall"]["fp"]) * 0.05)  # type: ignore[index]
    tiny_fp_limit = math.ceil(int(control["tiny"]["fp"]) * 0.05)  # type: ignore[index]
    judgment = {
        "tiny_recall_improved": tiny_recall_delta > 0,
        "tiny_f1_improved": tiny_f1_delta > 0,
        "overall_f1_loss_within_0_002": overall_f1_delta >= -0.002,
        "overall_false_detections_not_materially_increased": overall_fp_delta <= overall_fp_limit,
        "tiny_false_detections_not_materially_increased": tiny_fp_delta <= tiny_fp_limit,
        "material_increase_definition": "increase greater than 5% of the extended-training control count",
        "deltas": {
            "tiny_recall": tiny_recall_delta,
            "tiny_f1": tiny_f1_delta,
            "overall_f1": overall_f1_delta,
            "overall_false_detections": overall_fp_delta,
            "tiny_false_detections": tiny_fp_delta,
        },
    }
    judgment["success"] = all(
        judgment[key]
        for key in (
            "tiny_recall_improved",
            "tiny_f1_improved",
            "overall_f1_loss_within_0_002",
            "overall_false_detections_not_materially_increased",
            "tiny_false_detections_not_materially_increased",
        )
    )
    training_times = {
        "E4": read_json(E4_REPORT).get("duration_seconds"),
        "延长训练对照": control_report.get("duration_seconds"),
        "E5-B1": training_report.get("duration_seconds"),
    }
    report = {
        "status": "passed",
        "phase": "evaluate",
        "created_at_utc": utc_now(),
        "protocol": {
            "split": "val",
            "images": EXPECTED_VAL_IMAGES,
            "ground_truth": EXPECTED_VAL_BOXES,
            "imgsz": IMGSZ,
            "confidence": CONFIDENCE,
            "nms_iou": NMS_IOU,
            "matching_iou": MATCHING_IOU,
            "class_aware_nms": True,
            "class_aware_matching": True,
            "held_out_data_used": False,
        },
        "metrics": metrics,
        "performance": performance,
        "training_duration_seconds": training_times,
        "success_judgment": judgment,
    }
    output = ROOT / "evaluation"
    output.mkdir(parents=True, exist_ok=False)
    write_json_report(output / "e5b_comparison.json", report)
    rows = metric_rows(metrics)
    write_csv(output / "e5b_comparison.csv", rows)
    markdown = "# E5-B1完整 val 对比\n\n"
    markdown += "| 模型 | 范围 | 正确检出 | 漏检 | 误检 | 查准率 | 召回率 | 综合指标 |\n"
    markdown += "|---|---|---:|---:|---:|---:|---:|---:|\n"
    for row in rows:
        markdown += (
            f"| {row['模型']} | {row['范围']} | {row['正确检出']} | {row['漏检']} | {row['误检']} | "
            f"{row['查准率']:.6f} | {row['召回率']:.6f} | {row['综合指标']:.6f} |\n"
        )
    markdown += "\n## 速度、显存与训练耗时\n\n"
    markdown += "| 模型 | 推理耗时(ms/图) | 流水线耗时(ms/图) | 峰值显存(MiB) | 训练耗时(小时) |\n"
    markdown += "|---|---:|---:|---:|---:|\n"
    for name in weights:
        hours = float(training_times[name]) / 3600 if training_times[name] is not None else float("nan")
        markdown += (
            f"| {name} | {performance[name].get('inference_ms_per_image', 0):.3f} | "
            f"{performance[name]['pipeline_ms_per_image']:.3f} | {performance[name]['peak_cuda_memory_mb']:.1f} | {hours:.3f} |\n"
        )
    markdown += "\n## 成功门槛\n\n"
    markdown += f"- 结论：{'达到' if judgment['success'] else '未达到'}成功门槛。\n"
    markdown += f"- 微小目标召回率变化：{tiny_recall_delta:+.6f}；综合指标变化：{tiny_f1_delta:+.6f}。\n"
    markdown += f"- 总体综合指标变化：{overall_f1_delta:+.6f}。\n"
    markdown += f"- 总体误检变化：{overall_fp_delta:+d}；微小目标误检变化：{tiny_fp_delta:+d}。\n"
    markdown += "- “明显增加”按相对延长训练对照超过5%判定；未使用独立留出数据。\n"
    (output / "e5b_comparison.md").write_text(markdown, encoding="utf-8")
    state["status"] = "passed"
    state["evaluation_passed"] = True
    state["evaluation_report"] = report
    save_state(state)
    return report


def run(args: argparse.Namespace) -> dict[str, object]:
    global ROOT
    ROOT = resolve_e5b_experiment_root(PROJECT_ROOT, args.experiment_name)
    if args.phase == "prepare":
        return prepare(
            args.device,
            primary_target_size=args.primary_target_size,
            repeated_target_size=args.repeated_target_size,
        )
    if args.phase == "audit-augmentations":
        return audit_augmentations(args.device)
    if args.phase == "train":
        return train(args.device)
    if args.phase == "continue-training":
        return continue_training(args.device)
    return evaluate(args.device, args.batch)


def main() -> int:
    try:
        report = run(parser().parse_args())
    except Exception as exc:
        print(json.dumps({"status": "failed", "reason": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"], "phase": report["phase"], "root": str(ROOT)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
