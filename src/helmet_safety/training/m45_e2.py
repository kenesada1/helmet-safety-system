from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Callable, Mapping, Sequence

from PIL import Image

from helmet_safety.training.baseline import load_ground_truth_boxes


SIZE_BUCKETS = (
    "equivalent_size_le_10",
    "10_lt_equivalent_size_le_20",
    "20_lt_equivalent_size_le_30",
    "30_lt_equivalent_size_le_50",
    "equivalent_size_gt_50",
)

EXPECTED_E4_TINY_SUMMARY = {
    "images": 35,
    "ground_truth_instances": 128,
    "tp": 77,
    "fn": 51,
    "helmet_instances": 14,
    "helmet_tp": 4,
    "helmet_fn": 10,
    "no_helmet_instances": 114,
    "no_helmet_tp": 73,
    "no_helmet_fn": 41,
}

LOCALIZATION_BUCKETS = (
    "iou_ge_0_5",
    "iou_0_4_to_0_5",
    "iou_0_2_to_0_4",
    "iou_0_1_to_0_2",
    "unresolved",
)


def streaming_image_source(images_dir: Path) -> str:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"image source directory does not exist: {images_dir}")
    return str(images_dir.resolve())


def run_sequential_model_stages(
    stages: Sequence[tuple[str, Path]],
    *,
    loader: Callable[[Path], object],
    runner: Callable[[str, object], object],
    cleanup: Callable[[], None],
) -> dict[str, object]:
    """Run GPU model stages one at a time and release each before the next load."""

    results: dict[str, object] = {}
    for name, path in stages:
        model = loader(path)
        results[name] = runner(name, model)
        del model
        cleanup()
    return results


def reported_gpu_memory(log_path: Path) -> dict[str, object] | None:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    samples = [float(match) for match in re.findall(r"(?<![A-Za-z0-9])([0-9]+(?:\.[0-9]+)?)G(?![A-Za-z])", text)]
    if not samples:
        return None
    maximum = max(samples)
    return {
        "source": "maximum GiB value reported in the Ultralytics training log",
        "max_reported_gib": maximum,
        "max_reported_bytes": int(maximum * 1024**3),
    }


def equivalent_size_for_box(box: Sequence[float]) -> float:
    width = max(0.0, float(box[2]) - float(box[0]))
    height = max(0.0, float(box[3]) - float(box[1]))
    return math.sqrt(width * height)


def _is_tiny_box(box: Sequence[float], max_equivalent_size: float) -> bool:
    size = equivalent_size_for_box(box)
    return size <= max_equivalent_size or math.isclose(size, max_equivalent_size, rel_tol=0.0, abs_tol=1e-9)


def size_bucket_for_box(box: Sequence[float]) -> str:
    equivalent_size = equivalent_size_for_box(box)
    if equivalent_size <= 10:
        return SIZE_BUCKETS[0]
    if equivalent_size <= 20:
        return SIZE_BUCKETS[1]
    if equivalent_size <= 30:
        return SIZE_BUCKETS[2]
    if equivalent_size <= 50:
        return SIZE_BUCKETS[3]
    return SIZE_BUCKETS[4]


def select_tiny_val_images(
    images_dir: Path, labels_dir: Path, *, max_equivalent_size: float = 10.0
) -> list[dict[str, object]]:
    """Select val images containing tiny GT, measured in original-image pixels."""

    selected: list[dict[str, object]] = []
    for image_path in sorted(
        path.resolve()
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ):
        with Image.open(image_path) as image:
            image_size = image.size
        ground_truth = load_ground_truth_boxes(labels_dir / f"{image_path.stem}.txt", image_size=image_size)
        tiny_indices = [
            index
            for index, gt in enumerate(ground_truth)
            if _is_tiny_box(gt["box"], max_equivalent_size)  # type: ignore[arg-type]
        ]
        if tiny_indices:
            selected.append(
                {
                    "image_id": image_path.name,
                    "image_path": str(image_path),
                    "image_width": image_size[0],
                    "image_height": image_size[1],
                    "tiny_gt_indices": tiny_indices,
                    "tiny_gt_count": len(tiny_indices),
                }
            )
    return selected


def _box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def class_aware_matches(
    ground_truth: Sequence[Mapping[str, object]],
    predictions: Sequence[Mapping[str, object]],
    *,
    iou_threshold: float,
) -> set[int]:
    pairs = sorted(
        (
            (_box_iou(gt["box"], pred["box"]), gt_index, pred_index)  # type: ignore[arg-type]
            for gt_index, gt in enumerate(ground_truth)
            for pred_index, pred in enumerate(predictions)
            if int(gt["class_id"]) == int(pred["class_id"])
        ),
        reverse=True,
    )
    matched_gt: set[int] = set()
    matched_predictions: set[int] = set()
    for iou, gt_index, prediction_index in pairs:
        if iou < iou_threshold:
            break
        if gt_index in matched_gt or prediction_index in matched_predictions:
            continue
        matched_gt.add(gt_index)
        matched_predictions.add(prediction_index)
    return matched_gt


def _empty_accumulator() -> dict[str, object]:
    return {
        "image_ids": set(),
        "ground_truth_instances": 0,
        "helmet_instances": 0,
        "no_helmet_instances": 0,
        "tp": 0,
        "fn": 0,
        "fp": 0,
        "helmet_tp": 0,
        "helmet_fn": 0,
        "no_helmet_tp": 0,
        "no_helmet_fn": 0,
    }


def _record_ground_truth(accumulator: dict[str, object], *, image_id: str, class_id: int, matched: bool) -> None:
    accumulator["image_ids"].add(image_id)  # type: ignore[union-attr]
    accumulator["ground_truth_instances"] = int(accumulator["ground_truth_instances"]) + 1
    class_name = "helmet" if class_id == 0 else "no_helmet"
    accumulator[f"{class_name}_instances"] = int(accumulator[f"{class_name}_instances"]) + 1
    outcome = "tp" if matched else "fn"
    accumulator[outcome] = int(accumulator[outcome]) + 1
    accumulator[f"{class_name}_{outcome}"] = int(accumulator[f"{class_name}_{outcome}"]) + 1


def _recall(tp: int, fn: int) -> float | None:
    denominator = tp + fn
    return tp / denominator if denominator else None


def _finalize(accumulator: Mapping[str, object]) -> dict[str, object]:
    tp = int(accumulator["tp"])
    fn = int(accumulator["fn"])
    helmet_tp = int(accumulator["helmet_tp"])
    helmet_fn = int(accumulator["helmet_fn"])
    no_helmet_tp = int(accumulator["no_helmet_tp"])
    no_helmet_fn = int(accumulator["no_helmet_fn"])
    return {
        "images": len(accumulator["image_ids"]),  # type: ignore[arg-type]
        "ground_truth_instances": int(accumulator["ground_truth_instances"]),
        "helmet_instances": int(accumulator["helmet_instances"]),
        "no_helmet_instances": int(accumulator["no_helmet_instances"]),
        "tp": tp,
        "fn": fn,
        "recall": _recall(tp, fn),
        "helmet_tp": helmet_tp,
        "helmet_fn": helmet_fn,
        "helmet_recall": _recall(helmet_tp, helmet_fn),
        "no_helmet_tp": no_helmet_tp,
        "no_helmet_fn": no_helmet_fn,
        "no_helmet_recall": _recall(no_helmet_tp, no_helmet_fn),
    }


def summarize_detection_slices(
    records: Sequence[Mapping[str, object]], *, iou_threshold: float = 0.5
) -> dict[str, object]:
    """Summarize fixed val-only size and dense-scene recall slices."""

    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be within (0, 1]")
    size_accumulators = {name: _empty_accumulator() for name in SIZE_BUCKETS}
    dense_accumulators = {threshold: _empty_accumulator() for threshold in (10, 20)}

    for record in records:
        image_id = str(record["image_id"])
        ground_truth = record["ground_truth"]  # type: ignore[assignment]
        predictions = record["predictions"]  # type: ignore[assignment]
        matched_indices = class_aware_matches(
            ground_truth, predictions, iou_threshold=iou_threshold  # type: ignore[arg-type]
        )
        dense_thresholds = [threshold for threshold in (10, 20) if len(ground_truth) >= threshold]
        for threshold in dense_thresholds:
            dense_accumulators[threshold]["fp"] = (
                int(dense_accumulators[threshold]["fp"]) + len(predictions) - len(matched_indices)
            )
        for gt_index, gt in enumerate(ground_truth):
            class_id = int(gt["class_id"])
            if class_id not in (0, 1):
                raise ValueError(f"unexpected class id: {class_id}")
            matched = gt_index in matched_indices
            bucket = size_bucket_for_box(gt["box"])
            _record_ground_truth(size_accumulators[bucket], image_id=image_id, class_id=class_id, matched=matched)
            for threshold in dense_thresholds:
                _record_ground_truth(
                    dense_accumulators[threshold], image_id=image_id, class_id=class_id, matched=matched
                )

    return {
        "matching": {"class_aware": True, "iou_threshold": iou_threshold},
        "size_bins": {name: _finalize(accumulator) for name, accumulator in size_accumulators.items()},
        "dense_scenes": {
            f"ground_truth_gte_{threshold}": {**_finalize(accumulator), "fp": int(accumulator["fp"])}
            for threshold, accumulator in dense_accumulators.items()
        },
    }


def summarize_tiny_ground_truth(
    records: Sequence[Mapping[str, object]], *, max_equivalent_size: float = 10.0, iou_threshold: float = 0.5
) -> dict[str, object]:
    """Match against every GT in each selected image, then report only tiny GT."""

    accumulator = _empty_accumulator()
    false_negatives: list[dict[str, object]] = []
    for record in records:
        image_id = str(record["image_id"])
        ground_truth = record["ground_truth"]  # type: ignore[assignment]
        predictions = record["predictions"]  # type: ignore[assignment]
        matched_indices = class_aware_matches(
            ground_truth, predictions, iou_threshold=iou_threshold  # type: ignore[arg-type]
        )
        for gt_index, gt in enumerate(ground_truth):
            if not _is_tiny_box(gt["box"], max_equivalent_size):
                continue
            class_id = int(gt["class_id"])
            matched = gt_index in matched_indices
            _record_ground_truth(accumulator, image_id=image_id, class_id=class_id, matched=matched)
            if not matched:
                false_negatives.append(
                    {
                        "image_id": image_id,
                        "gt_index": gt_index,
                        "class_id": class_id,
                        "class_name": "helmet" if class_id == 0 else "no_helmet",
                        "gt_box_xyxy": [float(value) for value in gt["box"]],
                    }
                )
    return {"summary": _finalize(accumulator), "false_negatives": false_negatives}


def assert_expected_tiny_summary(summary: Mapping[str, object]) -> None:
    mismatches = {
        key: {"expected": expected, "actual": summary.get(key)}
        for key, expected in EXPECTED_E4_TINY_SUMMARY.items()
        if summary.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"tiny benchmark mismatch: {mismatches}")


def evaluate_tiny_conf_records(
    records: Sequence[Mapping[str, object]],
    original_false_negatives: Sequence[Mapping[str, object]],
    *,
    max_equivalent_size: float = 10.0,
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    """Evaluate one confidence setting and track which baseline tiny FNs were recovered."""

    tiny = summarize_tiny_ground_truth(
        records,
        max_equivalent_size=max_equivalent_size,
        iou_threshold=iou_threshold,
    )
    matched_by_image: dict[str, set[int]] = {}
    false_positives = 0
    for record in records:
        image_id = str(record["image_id"])
        ground_truth = record["ground_truth"]  # type: ignore[assignment]
        predictions = record["predictions"]  # type: ignore[assignment]
        matched = class_aware_matches(
            ground_truth, predictions, iou_threshold=iou_threshold  # type: ignore[arg-type]
        )
        matched_by_image[image_id] = matched
        false_positives += len(predictions) - len(matched)

    recovered = [
        dict(item)
        for item in original_false_negatives
        if int(item["gt_index"]) in matched_by_image.get(str(item["image_id"]), set())
    ]
    return {
        **tiny,
        "false_positives": false_positives,
        "recovered_original_fn_count": len(recovered),
        "recovered_helmet_count": sum(str(item["class_name"]) == "helmet" for item in recovered),
        "recovered_no_helmet_count": sum(str(item["class_name"]) == "no_helmet" for item in recovered),
        "recovered_original_false_negatives": recovered,
    }


def render_tiny_conf_markdown(rows: Sequence[Mapping[str, object]], *, analysis: str) -> str:
    """Render the fixed E4 tiny confidence sweep as a comparison report."""

    lines = [
        "# E4 极小目标 conf 单变量实验",
        "",
        "- 样本：第一组冻结的 35 张 val 图片与 51 个原始 FN；未使用 test。",
        "- 固定：E4 best.pt、imgsz=960、matching IoU=0.5、默认 NMS；仅改变 conf。",
        "- FP：35 张图上的全部预测先与图片内全部 GT 做 class-aware 一对一匹配后，未匹配预测的总数。",
        "",
        "## 对比表",
        "",
        "| conf | tiny TP | tiny FN | tiny R | helmet TP | helmet FN | helmet R | no_helmet TP | no_helmet FN | no_helmet R | FP | FP Δ | FP 倍率 | 救回 FN | 救回 helmet | 救回 no_helmet |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['conf']:g} | {row['tiny_tp']} | {row['tiny_fn']} | {float(row['tiny_recall']):.6f} | "
            f"{row['helmet_tp']} | {row['helmet_fn']} | {float(row['helmet_recall']):.6f} | "
            f"{row['no_helmet_tp']} | {row['no_helmet_fn']} | {float(row['no_helmet_recall']):.6f} | "
            f"{row['fp_35_images']} | {int(row['fp_delta_vs_025']):+d} | {float(row['fp_ratio_vs_025']):.3f}× | "
            f"{row['recovered_original_fn_count']} | {row['recovered_helmet_count']} | {row['recovered_no_helmet_count']} |"
        )
    lines.extend(["", "## 原 51 个 FN 的救回明细", ""])
    for row in rows:
        keys = str(row["recovered_original_fn_keys"]) or "无"
        lines.append(f"- conf={row['conf']:g}（{row['recovered_original_fn_count']} 个）：{keys}")
    lines.extend(["", "## 判断", "", analysis, ""])
    return "\n".join(lines)


def audit_obvious_duplicate_boxes(
    records: Sequence[Mapping[str, object]], *, duplicate_iou_threshold: float = 0.7
) -> dict[str, object]:
    """Find highly overlapping same-class post-NMS prediction pairs."""

    if not 0 < duplicate_iou_threshold <= 1:
        raise ValueError("duplicate IoU threshold must be within (0, 1]")
    details: list[dict[str, object]] = []
    images: set[str] = set()
    prediction_keys: set[tuple[str, int]] = set()
    for record in records:
        image_id = str(record["image_id"])
        predictions = record["predictions"]  # type: ignore[assignment]
        for first_index, first in enumerate(predictions):
            for second_index in range(first_index + 1, len(predictions)):
                second = predictions[second_index]
                if int(first["class_id"]) != int(second["class_id"]):
                    continue
                pair_iou = _box_iou(first["box"], second["box"])
                if pair_iou < duplicate_iou_threshold:
                    continue
                class_id = int(first["class_id"])
                details.append(
                    {
                        "image_id": image_id,
                        "class_id": class_id,
                        "class_name": "helmet" if class_id == 0 else "no_helmet",
                        "prediction_index_a": first_index,
                        "prediction_index_b": second_index,
                        "pair_iou": pair_iou,
                    }
                )
                images.add(image_id)
                prediction_keys.update(((image_id, first_index), (image_id, second_index)))
    return {
        "duplicate_iou_threshold": duplicate_iou_threshold,
        "duplicate_pairs": len(details),
        "images_with_duplicates": len(images),
        "predictions_in_duplicate_pairs": len(prediction_keys),
        "has_obvious_duplicates": bool(details),
        "details": details,
    }


def render_tiny_nms_iou_markdown(rows: Sequence[Mapping[str, object]], *, analysis: str) -> str:
    """Render the fixed E4 tiny NMS-IoU sweep as a comparison report."""

    lines = [
        "# E4 极小目标 NMS IoU 单变量实验",
        "",
        "- 样本：第一组冻结的 35 张 val 图片与 51 个原始 FN；未使用 test。",
        "- 固定：E4 best.pt、imgsz=960、conf=0.25、matching IoU=0.5、class-aware NMS；仅改变 NMS IoU。",
        "- 明显重复框：同图、同类别的后 NMS 预测框对 IoU >= 0.70。",
        "- FP：35 张图上的全部预测先与图片内全部 GT 做 class-aware 一对一匹配后，未匹配预测的总数。",
        "",
        "## 对比表",
        "",
        "| NMS IoU | tiny TP | tiny FN | tiny R | helmet R | no_helmet R | FP | 救回 FN | 救回 helmet | 救回 no_helmet | 重复框对 | 涉及图片 | 明显重复 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        duplicate_label = "是" if bool(row["has_obvious_duplicates"]) else "否"
        lines.append(
            f"| {float(row['nms_iou']):.2f} | {row['tiny_tp']} | {row['tiny_fn']} | {float(row['tiny_recall']):.6f} | "
            f"{float(row['helmet_recall']):.6f} | {float(row['no_helmet_recall']):.6f} | "
            f"{row['fp_35_images']} | {row['recovered_original_fn_count']} | {row['recovered_helmet_count']} | "
            f"{row['recovered_no_helmet_count']} | {row['obvious_duplicate_pairs']} | "
            f"{row['images_with_obvious_duplicates']} | {duplicate_label} |"
        )
    lines.extend(["", "## 原 51 个 FN 的救回明细", ""])
    for row in rows:
        keys = str(row["recovered_original_fn_keys"]) or "无"
        lines.append(f"- NMS IoU={float(row['nms_iou']):.2f}（{row['recovered_original_fn_count']} 个）：{keys}")
    lines.extend(["", "## 判断", "", analysis, ""])
    return "\n".join(lines)


def localization_bucket_for_iou(max_iou: float | None) -> str:
    if max_iou is None or max_iou < 0.1:
        return "unresolved"
    if max_iou < 0.2:
        return "iou_0_1_to_0_2"
    if max_iou < 0.4:
        return "iou_0_2_to_0_4"
    if max_iou < 0.5:
        return "iou_0_4_to_0_5"
    return "iou_ge_0_5"


def analyze_fn_candidate_evidence(
    original_false_negatives: Sequence[Mapping[str, object]],
    predictions_by_image: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    """Extract post-NMS low-confidence candidate evidence for frozen baseline FNs."""

    details: list[dict[str, object]] = []
    for item in original_false_negatives:
        image_id = str(item["image_id"])
        class_id = int(item["class_id"])
        gt_box = item["gt_box_xyxy"]  # type: ignore[assignment]
        predictions = predictions_by_image.get(image_id, ())
        correct = [prediction for prediction in predictions if int(prediction["class_id"]) == class_id]
        wrong = [prediction for prediction in predictions if int(prediction["class_id"]) != class_id]
        correct_max_confidence_candidate = max(correct, key=lambda prediction: float(prediction["confidence"]), default=None)
        correct_max_iou_candidate = max(correct, key=lambda prediction: _box_iou(gt_box, prediction["box"]), default=None)
        wrong_max_confidence_candidate = max(wrong, key=lambda prediction: float(prediction["confidence"]), default=None)
        wrong_max_iou_candidate = max(wrong, key=lambda prediction: _box_iou(gt_box, prediction["box"]), default=None)
        correct_confidence = (
            float(correct_max_confidence_candidate["confidence"]) if correct_max_confidence_candidate else None
        )
        correct_iou = _box_iou(gt_box, correct_max_iou_candidate["box"]) if correct_max_iou_candidate else None
        wrong_confidence = float(wrong_max_confidence_candidate["confidence"]) if wrong_max_confidence_candidate else None
        wrong_iou = _box_iou(gt_box, wrong_max_iou_candidate["box"]) if wrong_max_iou_candidate else None
        details.append(
            {
                **dict(item),
                "correct_class_candidate_count": len(correct),
                "correct_class_max_confidence": correct_confidence,
                "correct_class_max_confidence_candidate_iou": (
                    _box_iou(gt_box, correct_max_confidence_candidate["box"])
                    if correct_max_confidence_candidate
                    else None
                ),
                "correct_class_max_iou": correct_iou,
                "correct_class_max_iou_candidate_confidence": (
                    float(correct_max_iou_candidate["confidence"]) if correct_max_iou_candidate else None
                ),
                "wrong_class_candidate_count": len(wrong),
                "wrong_class_max_confidence": wrong_confidence,
                "wrong_class_max_confidence_candidate_iou": (
                    _box_iou(gt_box, wrong_max_confidence_candidate["box"])
                    if wrong_max_confidence_candidate
                    else None
                ),
                "wrong_class_max_iou": wrong_iou,
                "wrong_class_max_iou_candidate_confidence": (
                    float(wrong_max_iou_candidate["confidence"]) if wrong_max_iou_candidate else None
                ),
                "localization_bucket": localization_bucket_for_iou(correct_iou),
            }
        )

    summary: dict[str, object] = {}
    for scope, class_name in (("overall", None), ("helmet", "helmet"), ("no_helmet", "no_helmet")):
        scoped = details if class_name is None else [item for item in details if item["class_name"] == class_name]
        total = len(scoped)
        summary[scope] = {
            "total": total,
            "buckets": {
                bucket: {
                    "count": sum(item["localization_bucket"] == bucket for item in scoped),
                    "share": (
                        sum(item["localization_bucket"] == bucket for item in scoped) / total if total else None
                    ),
                }
                for bucket in LOCALIZATION_BUCKETS
            },
        }
    return {
        "details": details,
        "summary": summary,
        "helmet_false_negatives": [item for item in details if item["class_name"] == "helmet"],
    }


def compare_slice_recalls(
    baseline: Mapping[str, object], candidate: Mapping[str, object], *, candidate_key: str = "m45_e2_960"
) -> dict[str, object]:
    """Compare recall values from identically defined baseline and E2 slices."""

    comparison: dict[str, object] = {}
    for group in ("size_bins", "dense_scenes"):
        baseline_group = baseline[group]  # type: ignore[assignment]
        candidate_group = candidate[group]  # type: ignore[assignment]
        group_comparison: dict[str, object] = {}
        if set(baseline_group) != set(candidate_group):
            raise ValueError(f"slice keys differ for {group}")
        for slice_name in baseline_group:
            baseline_slice = baseline_group[slice_name]
            candidate_slice = candidate_group[slice_name]
            slice_comparison: dict[str, object] = {}
            for metric in ("recall", "helmet_recall", "no_helmet_recall"):
                baseline_value = baseline_slice[metric]
                candidate_value = candidate_slice[metric]
                if baseline_value is None or candidate_value is None:
                    slice_comparison[metric] = {
                        "m4_baseline_640": baseline_value,
                        candidate_key: candidate_value,
                        "absolute_change": None,
                        "percentage_point_change": None,
                    }
                    continue
                baseline_number = float(baseline_value)
                candidate_number = float(candidate_value)
                if not math.isfinite(baseline_number) or not math.isfinite(candidate_number):
                    raise ValueError(f"non-finite slice recall: {group}.{slice_name}.{metric}")
                change = round(candidate_number - baseline_number, 6)
                slice_comparison[metric] = {
                    "m4_baseline_640": baseline_number,
                    candidate_key: candidate_number,
                    "absolute_change": change,
                    "percentage_point_change": round(change * 100, 4),
                }
            group_comparison[slice_name] = slice_comparison
        comparison[group] = group_comparison
    return comparison
