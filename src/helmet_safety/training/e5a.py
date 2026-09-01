from __future__ import annotations

from collections import Counter
import math
import os
from pathlib import Path
import random
import shutil
from typing import Mapping, Sequence

import yaml

from helmet_safety.training.analysis_core import (
    _class_aware_match_pairs,
    class_aware_matches,
    summarize_fixed_threshold_detections,
)


E4_AUGMENTATION_KEYS = (
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "flipud",
    "fliplr",
    "bgr",
    "mosaic",
    "mixup",
    "cutmix",
    "copy_paste",
    "copy_paste_mode",
    "auto_augment",
    "erasing",
    "close_mosaic",
)


def _equivalent_size(box: Sequence[float]) -> float:
    width = max(0.0, float(box[2]) - float(box[0]))
    height = max(0.0, float(box[3]) - float(box[1]))
    return math.sqrt(width * height)


def identify_tiny_difficult_samples(
    records: Sequence[Mapping[str, object]],
    *,
    max_equivalent_size: float = 10.0,
    iou_threshold: float = 0.5,
) -> list[dict[str, object]]:
    """Return train images containing tiny ground-truth instances missed by the model."""

    if max_equivalent_size <= 0:
        raise ValueError("微小目标等效边长上限必须大于0")
    if not 0 < iou_threshold <= 1:
        raise ValueError("匹配交并比阈值必须位于(0, 1]")
    difficult: list[dict[str, object]] = []
    for record in records:
        ground_truth = record["ground_truth"]  # type: ignore[assignment]
        predictions = record["predictions"]  # type: ignore[assignment]
        matched = class_aware_matches(
            ground_truth, predictions, iou_threshold=iou_threshold  # type: ignore[arg-type]
        )
        tiny_indices = [
            index
            for index, item in enumerate(ground_truth)
            if _equivalent_size(item["box"]) <= max_equivalent_size
        ]
        missed_indices = [index for index in tiny_indices if index not in matched]
        if not missed_indices:
            continue
        helmet_ground_truth = sum(int(ground_truth[index]["class_id"]) == 0 for index in tiny_indices)
        no_helmet_ground_truth = sum(int(ground_truth[index]["class_id"]) == 1 for index in tiny_indices)
        helmet_missed = sum(int(ground_truth[index]["class_id"]) == 0 for index in missed_indices)
        no_helmet_missed = sum(int(ground_truth[index]["class_id"]) == 1 for index in missed_indices)
        difficult.append(
            {
                "image_id": str(record["image_id"]),
                "image_path": str(record["image_path"]),
                "tiny_ground_truth": len(tiny_indices),
                "tiny_helmet_ground_truth": helmet_ground_truth,
                "tiny_no_helmet_ground_truth": no_helmet_ground_truth,
                "tiny_missed": len(missed_indices),
                "tiny_helmet_missed": helmet_missed,
                "tiny_no_helmet_missed": no_helmet_missed,
                "sampling_weight": 2 * helmet_missed + no_helmet_missed,
            }
        )
    return sorted(difficult, key=lambda item: str(item["image_path"]))


def build_resampled_manifests(
    image_paths: Sequence[Path],
    difficult_samples: Sequence[Mapping[str, object]],
    *,
    extra_count: int = 1043,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Build equal-length uniform-control and tiny-difficulty manifests."""

    if extra_count < 0:
        raise ValueError("额外抽样条数不能为负数")
    all_images = sorted(str(Path(path)) for path in image_paths)
    if not all_images or len(set(all_images)) != len(all_images):
        raise ValueError("原始训练图片清单必须非空且不能重复")
    difficult_paths = [str(Path(item["image_path"])) for item in difficult_samples]
    weights = [int(item["sampling_weight"]) for item in difficult_samples]
    if not difficult_paths or any(path not in set(all_images) for path in difficult_paths):
        raise ValueError("困难样本必须是原始训练图片的非空子集")
    if len(set(difficult_paths)) != len(difficult_paths) or any(weight <= 0 for weight in weights):
        raise ValueError("困难样本路径不能重复且抽样权重必须大于0")
    control_extra = random.Random(seed).choices(all_images, k=extra_count)
    e5a_extra = random.Random(seed).choices(difficult_paths, weights=weights, k=extra_count)
    return {
        "control_entries": [*all_images, *control_extra],
        "e5a_entries": [*all_images, *e5a_extra],
        "control_extra": control_extra,
        "e5a_extra": e5a_extra,
    }


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def audit_training_manifest(
    entries: Sequence[str],
    *,
    original_train_images: Sequence[Path],
    processed_root: Path,
    expected_entries: int = 6500,
) -> dict[str, object]:
    """Validate manifest length, full train coverage, labels, and split isolation."""

    root = processed_root.resolve()
    train_dir = (root / "images" / "train").resolve()
    val_dir = (root / "images" / "val").resolve()
    test_dir = (root / "images" / "test").resolve()
    resolved_entries = [Path(value).resolve() for value in entries]
    original = {Path(value).resolve() for value in original_train_images}
    if len(resolved_entries) != expected_entries:
        raise ValueError(f"训练清单条数异常：{len(resolved_entries)} != {expected_entries}")
    validation_present = any(_is_relative_to(path, val_dir) for path in resolved_entries)
    test_present = any(_is_relative_to(path, test_dir) for path in resolved_entries)
    outside_train = [path for path in resolved_entries if not _is_relative_to(path, train_dir)]
    if validation_present or test_present or outside_train:
        raise ValueError("训练清单禁止包含验证集或测试集图片，且所有条目必须来自训练集")
    unique = set(resolved_entries)
    missing = sorted(str(path) for path in original - unique)
    unexpected = sorted(str(path) for path in unique - original)
    if missing or unexpected:
        raise ValueError(f"训练清单全量覆盖异常：缺少{len(missing)}张，意外出现{len(unexpected)}张")
    missing_files = [str(path) for path in unique if not path.is_file()]
    missing_labels = [
        str(root / "labels" / "train" / f"{path.stem}.txt")
        for path in unique
        if not (root / "labels" / "train" / f"{path.stem}.txt").is_file()
    ]
    if missing_files or missing_labels:
        raise ValueError(f"训练清单文件异常：缺少图片{len(missing_files)}张，缺少标签{len(missing_labels)}份")
    counts = Counter(resolved_entries)
    return {
        "entries": len(resolved_entries),
        "unique_images": len(unique),
        "all_original_training_images_present": not missing,
        "minimum_occurrences": min(counts.values()),
        "maximum_occurrences": max(counts.values()),
        "validation_images_present": validation_present,
        "test_images_present": test_present,
        "missing_images": missing_files,
        "missing_labels": missing_labels,
    }


def build_e5a_training_kwargs(
    *,
    e4_args: Mapping[str, object],
    data_yaml: Path,
    project_dir: Path,
    run_name: str,
    device: str,
) -> dict[str, object]:
    """Build the locked, independent 30-epoch continuation arguments."""

    missing = [key for key in E4_AUGMENTATION_KEYS if key not in e4_args]
    if missing:
        raise ValueError(f"E4增强参数缺失：{missing}")
    return {
        "data": str(data_yaml.resolve()),
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
        "amp": True,
        "resume": False,
        "cache": False,
        "pretrained": True,
        "plots": True,
        "device": device,
        "project": str(project_dir.resolve()),
        "name": run_name,
        "exist_ok": False,
        **{key: e4_args[key] for key in E4_AUGMENTATION_KEYS},
    }


def _metric_row(tp: int, fn: int, fp: int) -> dict[str, object]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"tp": tp, "fn": fn, "fp": fp, "precision": precision, "recall": recall, "f1": f1}


def summarize_e5a_evaluation(
    records: Sequence[Mapping[str, object]],
    *,
    matching_iou: float = 0.5,
    max_equivalent_size: float = 10.0,
) -> dict[str, object]:
    """Summarize overall, per-class, and tiny-box fixed-threshold metrics."""

    fixed = summarize_fixed_threshold_detections(records, iou_threshold=matching_iou)
    tiny_tp = tiny_fn = tiny_fp = 0
    for record in records:
        ground_truth = record["ground_truth"]  # type: ignore[assignment]
        predictions = record["predictions"]  # type: ignore[assignment]
        matches = _class_aware_match_pairs(
            ground_truth, predictions, iou_threshold=matching_iou  # type: ignore[arg-type]
        )
        matched_ground_truth = {gt_index for gt_index, _, _ in matches}
        matched_predictions = {prediction_index for _, prediction_index, _ in matches}
        for index, item in enumerate(ground_truth):
            if _equivalent_size(item["box"]) > max_equivalent_size:
                continue
            if index in matched_ground_truth:
                tiny_tp += 1
            else:
                tiny_fn += 1
        tiny_fp += sum(
            prediction_index not in matched_predictions
            and _equivalent_size(prediction["box"]) <= max_equivalent_size
            for prediction_index, prediction in enumerate(predictions)
        )
    return {
        "matching": fixed["matching"],
        "overall": fixed["overall"],
        "per_class": fixed["per_class"],
        "tiny": _metric_row(tiny_tp, tiny_fn, tiny_fp),
    }


def validate_e4_candidate_c_anchor(metrics: Mapping[str, object]) -> dict[str, int]:
    """Require the frozen full-validation Candidate-C counts before training."""

    overall = metrics["overall"]  # type: ignore[assignment]
    tiny = metrics["tiny"]  # type: ignore[assignment]
    actual = {
        "overall_correct_detections": int(overall["tp"]),
        "overall_misses": int(overall["fn"]),
        "overall_false_detections": int(overall["fp"]),
        "tiny_correct_detections": int(tiny["tp"]),
        "tiny_misses": int(tiny["fn"]),
    }
    expected = {
        "overall_correct_detections": 9420,
        "overall_misses": 505,
        "overall_false_detections": 951,
        "tiny_correct_detections": 79,
        "tiny_misses": 49,
    }
    if actual != expected:
        raise RuntimeError(f"E4候选C验证锚点无法复现：实际{actual}，预期{expected}")
    return actual


def judge_resampling_effectiveness(
    control: Mapping[str, object],
    e5a: Mapping[str, object],
    *,
    material_overall_f1_loss: float = 0.002,
) -> dict[str, object]:
    """Judge targeted resampling with a predeclared tiny-gain/overall-loss rule."""

    control_overall = control["overall"]  # type: ignore[assignment]
    e5a_overall = e5a["overall"]  # type: ignore[assignment]
    control_tiny = control["tiny"]  # type: ignore[assignment]
    e5a_tiny = e5a["tiny"]  # type: ignore[assignment]
    deltas = {
        "tiny_recall": float(e5a_tiny["recall"]) - float(control_tiny["recall"]),
        "tiny_f1": float(e5a_tiny["f1"]) - float(control_tiny["f1"]),
        "overall_f1": float(e5a_overall["f1"]) - float(control_overall["f1"]),
    }
    material_loss = deltas["overall_f1"] < -material_overall_f1_loss
    return {
        "effective": deltas["tiny_recall"] > 0 and deltas["tiny_f1"] > 0 and not material_loss,
        "tiny_recall_improved": deltas["tiny_recall"] > 0,
        "tiny_f1_improved": deltas["tiny_f1"] > 0,
        "overall_f1_material_loss": material_loss,
        "material_overall_f1_loss_threshold": material_overall_f1_loss,
        "deltas": deltas,
    }


def stage_dataset_variant(
    *,
    source_root: Path,
    stage_root: Path,
    manifest_entries: Sequence[str],
    expected_train_images: int = 5457,
    expected_val_images: int = 607,
    expected_manifest_entries: int = 6500,
) -> dict[str, object]:
    """Create an isolated hard-linked image/copy-label dataset without a test split."""

    source = source_root.resolve()
    stage = stage_root.resolve()
    if stage.exists():
        raise FileExistsError(f"派生数据目录已存在：{stage}")
    source_train = source / "images" / "train"
    source_val = source / "images" / "val"
    train_images = sorted(
        path.resolve()
        for path in source_train.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    val_images = sorted(
        path.resolve()
        for path in source_val.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(train_images) != expected_train_images or len(val_images) != expected_val_images:
        raise ValueError(
            f"原数据图片数异常：训练集{len(train_images)}，验证集{len(val_images)}"
        )
    stage_train_images = stage / "images" / "train"
    stage_val_images = stage / "images" / "val"
    stage_train_labels = stage / "labels" / "train"
    stage_val_labels = stage / "labels" / "val"
    for directory in (stage_train_images, stage_val_images, stage_train_labels, stage_val_labels):
        directory.mkdir(parents=True, exist_ok=False)

    def stage_split(images: Sequence[Path], split: str) -> None:
        destination_images = stage / "images" / split
        destination_labels = stage / "labels" / split
        for image in images:
            label = source / "labels" / split / f"{image.stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(f"标签不存在：{label}")
            os.link(image, destination_images / image.name)
            shutil.copy2(label, destination_labels / label.name)

    stage_split(train_images, "train")
    stage_split(val_images, "val")
    source_train_set = {str(path): path for path in train_images}
    translated: list[str] = []
    for entry in manifest_entries:
        resolved = str(Path(entry).resolve())
        if resolved not in source_train_set:
            raise ValueError(f"派生清单出现非训练集条目：{entry}")
        translated.append(str((stage_train_images / source_train_set[resolved].name).resolve()))
    train_manifest = stage / "train_6500.txt"
    train_manifest.write_text("\n".join(translated) + "\n", encoding="utf-8")
    dataset_yaml = stage / "dataset.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(stage),
                "train": str(train_manifest),
                "val": "images/val",
                "names": {0: "helmet", 1: "no_helmet"},
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    audit = audit_training_manifest(
        translated,
        original_train_images=[stage_train_images / path.name for path in train_images],
        processed_root=stage,
        expected_entries=expected_manifest_entries,
    )
    return {
        "stage_root": str(stage),
        "dataset_yaml": str(dataset_yaml.resolve()),
        "train_manifest": str(train_manifest.resolve()),
        "audit": audit,
        "test_split_included": False,
    }


def render_chinese_evaluation_markdown(metrics_by_experiment: Mapping[str, Mapping[str, object]]) -> str:
    """Render fixed-threshold evaluation metrics using Chinese user-facing terminology."""

    lines = [
        "# E5-A 微小困难样本定向重采样实验",
        "",
        "| 实验 | 范围 | 正确检出 | 漏检 | 误检 | 查准率 | 召回率 | 综合指标 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    scopes = (("overall", "总体"), ("helmet", "安全帽"), ("no_helmet", "未戴安全帽"), ("tiny", "微小目标"))
    for experiment, metrics in metrics_by_experiment.items():
        per_class = metrics["per_class"]  # type: ignore[assignment]
        for key, chinese_name in scopes:
            row = metrics[key] if key in ("overall", "tiny") else per_class[key]  # type: ignore[index]
            lines.append(
                f"| {experiment} | {chinese_name} | {int(row['tp'])} | {int(row['fn'])} | {int(row['fp'])} | "
                f"{float(row['precision']):.6f} | {float(row['recall']):.6f} | {float(row['f1']):.6f} |"
            )
    return "\n".join(lines) + "\n"


def chinese_detection_count_keys(value: object) -> object:
    """Translate detection-count abbreviations in a user-facing report payload."""

    if isinstance(value, Mapping):
        translated_keys = {"tp": "正确检出", "fn": "漏检", "fp": "误检"}
        return {
            translated_keys.get(str(key), str(key)): chinese_detection_count_keys(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [chinese_detection_count_keys(item) for item in value]
    if isinstance(value, tuple):
        return tuple(chinese_detection_count_keys(item) for item in value)
    return value
