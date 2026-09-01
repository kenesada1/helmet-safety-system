from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import random
import re
import statistics
from typing import Mapping, Sequence

from helmet_safety.training.e5a import _metric_row, _equivalent_size
from helmet_safety.training.analysis_core import (
    _class_aware_match_pairs,
    summarize_fixed_threshold_detections,
)


INPUT_SIZE = 960
TINY_MAX_SIZE = 10.0
CENTER_HARD_MAX = 24.0
MIN_CONTEXT_SIDE = 128
MAX_CROPS_PER_IMAGE = 4
MAX_USES_PER_TARGET = 2
MIN_TRACKED_CENTERS = 1000
MAX_AUGMENTED_AUDIT_SAMPLES = 12_000


def resolve_e5b_experiment_root(project_root: Path, experiment_name: str) -> Path:
    match = re.fullmatch(r"e5b_context_crop_(\d{3})", experiment_name)
    if match is None or int(match.group(1)) < 2:
        raise ValueError("实验名称必须是编号不小于002的e5b_context_crop_NNN")
    return project_root / "artifacts" / "e5b" / experiment_name


def normalized_box_is_in_bounds(values: Sequence[float], *, tolerance: float = 1e-6) -> bool:
    center_x, center_y, width, height = (float(value) for value in values)
    return (
        all(math.isfinite(value) for value in (center_x, center_y, width, height))
        and width > 0
        and height > 0
        and center_x - width / 2 >= -tolerance
        and center_y - height / 2 >= -tolerance
        and center_x + width / 2 <= 1 + tolerance
        and center_y + height / 2 <= 1 + tolerance
    )


def equivalent_size_after_resize(
    box: Sequence[float], crop_window: Sequence[int], *, input_size: int = INPUT_SIZE
) -> float:
    crop_width = int(crop_window[2]) - int(crop_window[0])
    crop_height = int(crop_window[3]) - int(crop_window[1])
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError("裁剪窗口宽高必须大于0")
    return _equivalent_size(box) * input_size / max(crop_width, crop_height)


def compute_context_window(
    *,
    image_size: tuple[int, int],
    center_box: Sequence[float],
    target_size: float,
    input_size: int = INPUT_SIZE,
    minimum_context_side: int = MIN_CONTEXT_SIDE,
) -> tuple[int, int, int, int]:
    width, height = image_size
    if width <= 0 or height <= 0 or target_size <= 0:
        raise ValueError("图片尺寸和目标尺寸必须大于0")
    x1, y1, x2, y2 = (float(value) for value in center_box)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("中心目标必须完整位于原图内")
    requested_side = math.ceil(input_size * _equivalent_size(center_box) / target_size)
    side = max(minimum_context_side, requested_side, math.ceil(x2 - x1), math.ceil(y2 - y1))
    crop_width = min(width, side)
    crop_height = min(height, side)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    left = min(max(round(center_x - crop_width / 2), 0), width - crop_width)
    top = min(max(round(center_y - crop_height / 2), 0), height - crop_height)
    right = left + crop_width
    bottom = top + crop_height
    if left > x1 or top > y1 or right < x2 or bottom < y2:
        raise ValueError("无法完整保留中心目标")
    return int(left), int(top), int(right), int(bottom)


def convert_boxes_for_crop(
    boxes: Sequence[Mapping[str, object]],
    *,
    center_index: int,
    crop_window: Sequence[int],
    minimum_visible_ratio: float = 0.5,
    minimum_side: float = 2.0,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    left, top, right, bottom = (float(value) for value in crop_window)
    if not 0 <= center_index < len(boxes):
        raise IndexError("中心目标编号越界")
    converted: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    for index, item in enumerate(boxes):
        x1, y1, x2, y2 = (float(value) for value in item["box"])  # type: ignore[index]
        original_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        clipped = [max(x1, left), max(y1, top), min(x2, right), min(y2, bottom)]
        clipped_width = max(0.0, clipped[2] - clipped[0])
        clipped_height = max(0.0, clipped[3] - clipped[1])
        visible_ratio = clipped_width * clipped_height / original_area if original_area else 0.0
        was_clipped = clipped != [x1, y1, x2, y2]
        keep = visible_ratio >= minimum_visible_ratio and clipped_width >= minimum_side and clipped_height >= minimum_side
        if index == center_index:
            if visible_ratio < 1.0 - 1e-9:
                raise ValueError("中心目标没有完整保留")
            keep = True
        decisions.append(
            {
                "source_index": index,
                "action": "保留" if keep else "删除",
                "visible_ratio": visible_ratio,
                "clipped": was_clipped,
            }
        )
        if keep:
            converted.append(
                {
                    "source_index": index,
                    "class_id": int(item["class_id"]),
                    "box": [clipped[0] - left, clipped[1] - top, clipped[2] - left, clipped[3] - top],
                    "is_center": index == center_index,
                }
            )
    return converted, decisions


def select_context_crop_requests(
    records: Sequence[Mapping[str, object]],
    *,
    tiny_max_size: float = TINY_MAX_SIZE,
    hard_max_size: float = CENTER_HARD_MAX,
    augmentation_max_scale: float = 1.5,
    primary_target_size: float = 14.0,
    repeated_target_size: float = 16.0,
) -> list[dict[str, object]]:
    requests: list[dict[str, object]] = []
    base_hard_max = hard_max_size / augmentation_max_scale
    if not 0 < primary_target_size <= base_hard_max or not 0 < repeated_target_size <= base_hard_max:
        raise ValueError("设计中心尺寸必须为正且不得超过增强前硬上限")
    for record in records:
        image_path = str(record["image_path"])
        width, height = int(record["width"]), int(record["height"])
        boxes = record["boxes"]  # type: ignore[assignment]
        candidates: list[int] = []
        for index, item in enumerate(boxes):
            box = item["box"]
            if _equivalent_size(box) > tiny_max_size:
                continue
            full_image_size = _equivalent_size(box) * INPUT_SIZE / max(width, height)
            if full_image_size <= base_hard_max + 1e-9:
                candidates.append(index)
        candidates.sort(key=lambda index: (int(boxes[index]["class_id"]) != 0, index))
        selected = candidates[:MAX_CROPS_PER_IMAGE]
        for index in selected:
            window = compute_context_window(
                image_size=(width, height), center_box=boxes[index]["box"], target_size=primary_target_size
            )
            size = equivalent_size_after_resize(boxes[index]["box"], window)
            if size <= base_hard_max + 1e-9:
                requests.append(
                    {
                        "image_path": image_path,
                        "center_index": index,
                        "target_size": primary_target_size,
                        "crop_window": window,
                        "center_size_960": size,
                    }
                )
        image_requests = [row for row in requests if row["image_path"] == image_path]
        for existing in list(image_requests):
            if len(image_requests) >= MAX_CROPS_PER_IMAGE:
                break
            index = int(existing["center_index"])
            window = compute_context_window(
                image_size=(width, height), center_box=boxes[index]["box"], target_size=repeated_target_size
            )
            size = equivalent_size_after_resize(boxes[index]["box"], window)
            if size <= base_hard_max + 1e-9:
                duplicate = {
                    "image_path": image_path,
                    "center_index": index,
                    "target_size": repeated_target_size,
                    "crop_window": window,
                    "center_size_960": size,
                }
                requests.append(duplicate)
                image_requests.append(duplicate)
    return requests


def build_context_crop_manifest(
    original_images: Sequence[Path],
    crop_images: Sequence[Path],
    *,
    extra_count: int = 1043,
    seed: int = 42,
) -> dict[str, list[str]]:
    originals = sorted(str(path) for path in original_images)
    crops = [str(path) for path in crop_images]
    if not originals or len(set(originals)) != len(originals):
        raise ValueError("原始训练图片必须非空且不重复")
    if len(crops) > extra_count or len(set(crops)) != len(crops):
        raise ValueError("裁剪图不能重复且数量不得超过额外项")
    fallback = random.Random(seed).choices(originals, k=extra_count - len(crops))
    return {
        "entries": [*originals, *crops, *fallback],
        "crop_entries": crops,
        "uniform_fallback": fallback,
    }


def audit_crop_mappings(mappings: Sequence[Mapping[str, object]]) -> dict[str, object]:
    image_counts = Counter(str(row["image_path"]) for row in mappings)
    target_counts = Counter((str(row["image_path"]), int(row["center_index"])) for row in mappings)
    maximum_image = max(image_counts.values(), default=0)
    maximum_target = max(target_counts.values(), default=0)
    maximum_size = max((float(row["center_size_960"]) for row in mappings), default=0.0)
    if maximum_image > MAX_CROPS_PER_IMAGE:
        raise ValueError(f"单张原图裁剪数超过{MAX_CROPS_PER_IMAGE}")
    if maximum_target > MAX_USES_PER_TARGET:
        raise ValueError(f"单个真实目标使用数超过{MAX_USES_PER_TARGET}")
    if maximum_size > CENTER_HARD_MAX + 1e-9:
        raise ValueError("中心目标在960输入下超过24像素硬上限")
    return {
        "crop_count": len(mappings),
        "source_image_usage": dict(sorted(image_counts.items())),
        "target_usage": {f"{path}#{index}": count for (path, index), count in sorted(target_counts.items())},
        "maximum_crops_per_image": maximum_image,
        "maximum_uses_per_target": maximum_target,
        "maximum_center_size_960": maximum_size,
    }


def evaluate_augmented_center_gate(
    sizes: Sequence[float],
    *,
    minimum_tracked_centers: int = MIN_TRACKED_CENTERS,
    required_minimum: float = 12.0,
    required_maximum: float = 20.0,
    minimum_in_range_ratio: float = 0.4,
    hard_maximum: float = CENTER_HARD_MAX,
) -> dict[str, object]:
    tracked = len(sizes)
    median_in_range = bool(sizes) and required_minimum <= statistics.median(sizes) <= required_maximum
    in_range_ratio = (
        sum(required_minimum <= value <= required_maximum for value in sizes) / tracked if sizes else 0.0
    )
    ordered = sorted(float(value) for value in sizes)
    if ordered:
        position = (len(ordered) - 1) * 0.9
        lower = math.floor(position)
        upper = math.ceil(position)
        p90 = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    else:
        p90 = math.inf
    p90_within_range = p90 <= required_maximum
    hard_maximum_respected = all(value <= hard_maximum + 1e-6 for value in sizes)
    return {
        "minimum_tracked_centers": minimum_tracked_centers,
        "tracked_center_boxes": tracked,
        "minimum_in_required_range_ratio": minimum_in_range_ratio,
        "in_required_range_ratio": in_range_ratio,
        "median_in_required_range": median_in_range,
        "p90_within_required_range": p90_within_range,
        "hard_maximum_respected": hard_maximum_respected,
        "passed": (
            tracked >= minimum_tracked_centers
            and in_range_ratio >= minimum_in_range_ratio
            and median_in_range
            and p90_within_range
            and hard_maximum_respected
        ),
    }


def should_continue_augmentation_audit(
    tracked_center_boxes: int,
    audited_augmented_samples: int,
    *,
    minimum_tracked_centers: int = MIN_TRACKED_CENTERS,
    maximum_augmented_samples: int = MAX_AUGMENTED_AUDIT_SAMPLES,
) -> bool:
    return (
        tracked_center_boxes < minimum_tracked_centers
        and audited_augmented_samples < maximum_augmented_samples
    )


def build_e5b_training_kwargs(
    control_kwargs: Mapping[str, object],
    *,
    data_yaml: Path,
    project_dir: Path,
    run_name: str,
    device: str,
) -> dict[str, object]:
    result = dict(control_kwargs)
    result.update(
        {
            "data": str(data_yaml.resolve()),
            "project": str(project_dir.resolve()),
            "name": run_name,
            "device": device,
        }
    )
    if result.get("resume") is not False or result.get("exist_ok") is not False:
        raise ValueError("延长训练对照必须是resume=False且exist_ok=False")
    return result


def validate_e5b_resume_request(
    *,
    last_checkpoint: Path,
    run_dir: Path,
    completed_epochs: int,
    checkpoint_epoch: int,
    optimizer_present: bool,
    requested_epochs: int,
) -> dict[str, object]:
    expected = (run_dir / "weights" / "last.pt").resolve()
    if last_checkpoint.resolve() != expected or not expected.is_file():
        raise ValueError("只能从本实验的最后检查点继续训练")
    if not optimizer_present:
        raise ValueError("最后检查点缺少优化器状态，不能无损续接")
    if checkpoint_epoch + 1 != completed_epochs or not 0 < completed_epochs < requested_epochs:
        raise ValueError("结果记录与检查点轮次不连续或训练轮次已完成")
    return {
        "last_checkpoint": str(expected),
        "completed_epochs": completed_epochs,
        "resume_from_epoch": completed_epochs + 1,
        "requested_epochs": requested_epochs,
    }


def summarize_e5b_evaluation(
    records: Sequence[Mapping[str, object]],
    *,
    matching_iou: float = 0.5,
    max_equivalent_size: float = TINY_MAX_SIZE,
) -> dict[str, object]:
    fixed = summarize_fixed_threshold_detections(records, iou_threshold=matching_iou)
    counts = {0: {"tp": 0, "fn": 0, "fp": 0}, 1: {"tp": 0, "fn": 0, "fp": 0}}
    for record in records:
        ground_truth = record["ground_truth"]  # type: ignore[assignment]
        predictions = record["predictions"]  # type: ignore[assignment]
        matches = _class_aware_match_pairs(ground_truth, predictions, iou_threshold=matching_iou)
        matched_ground_truth = {ground_truth_index for ground_truth_index, _, _ in matches}
        matched_predictions = {prediction_index for _, prediction_index, _ in matches}
        for index, item in enumerate(ground_truth):
            if _equivalent_size(item["box"]) <= max_equivalent_size:
                key = "tp" if index in matched_ground_truth else "fn"
                counts[int(item["class_id"])][key] += 1
        for index, item in enumerate(predictions):
            if index not in matched_predictions and _equivalent_size(item["box"]) <= max_equivalent_size:
                counts[int(item["class_id"])]["fp"] += 1
    per_class = {
        "helmet": _metric_row(**counts[0]),
        "no_helmet": _metric_row(**counts[1]),
    }
    total = {key: counts[0][key] + counts[1][key] for key in ("tp", "fn", "fp")}
    return {
        "matching": fixed["matching"],
        "overall": fixed["overall"],
        "per_class": fixed["per_class"],
        "tiny": _metric_row(**total),
        "tiny_per_class": per_class,
    }
