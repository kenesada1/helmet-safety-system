from __future__ import annotations

from typing import Callable, Mapping, Sequence

from PIL import Image


TileWindow = tuple[int, int, int, int]


def _axis_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    last_start = length - tile_size
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def tile_windows(
    *,
    image_size: tuple[int, int],
    tile_size: int,
    overlap_ratio: float,
) -> list[TileWindow]:
    """Return deterministic windows that cover an image and align to its edges."""

    width, height = image_size
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    if tile_size < 1:
        raise ValueError("tile size must be positive")
    if not 0.0 <= overlap_ratio < 1.0:
        raise ValueError("overlap ratio must be within [0, 1)")
    stride = round(tile_size * (1.0 - overlap_ratio))
    if stride < 1:
        raise ValueError("overlap ratio leaves no positive stride")
    x_starts = _axis_starts(width, tile_size, stride)
    y_starts = _axis_starts(height, tile_size, stride)
    return [
        (left, top, min(left + tile_size, width), min(top + tile_size, height))
        for top in y_starts
        for left in x_starts
    ]


def translate_predictions(
    predictions: Sequence[Mapping[str, object]],
    *,
    offset: tuple[int, int],
    image_size: tuple[int, int],
) -> list[dict[str, object]]:
    """Map tile-local boxes to clipped original-image coordinates."""

    offset_x, offset_y = offset
    width, height = image_size
    translated: list[dict[str, object]] = []
    for prediction in predictions:
        box = prediction["box"]  # type: ignore[assignment]
        mapped = [
            min(max(float(box[0]) + offset_x, 0.0), float(width)),
            min(max(float(box[1]) + offset_y, 0.0), float(height)),
            min(max(float(box[2]) + offset_x, 0.0), float(width)),
            min(max(float(box[3]) + offset_y, 0.0), float(height)),
        ]
        translated.append({**dict(prediction), "box": mapped})
    return translated


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


def class_aware_nms(
    predictions: Sequence[Mapping[str, object]],
    *,
    iou_threshold: float,
    max_detections: int,
) -> list[dict[str, object]]:
    """Merge detections with greedy confidence-ordered class-aware NMS."""

    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("IoU threshold must be within (0, 1]")
    if max_detections < 1:
        raise ValueError("max detections must be positive")
    ordered = sorted(
        enumerate(predictions),
        key=lambda item: (-float(item[1]["confidence"]), item[0]),
    )
    kept: list[dict[str, object]] = []
    for _, prediction in ordered:
        same_class_kept = (
            item for item in kept if int(item["class_id"]) == int(prediction["class_id"])
        )
        if any(
            _box_iou(prediction["box"], item["box"]) > iou_threshold  # type: ignore[arg-type]
            for item in same_class_kept
        ):
            continue
        kept.append(dict(prediction))
        if len(kept) == max_detections:
            break
    return kept


def merge_hybrid_predictions(
    full_predictions: Sequence[Mapping[str, object]],
    tiled_predictions: Sequence[Mapping[str, object]],
    *,
    nms_iou: float,
    max_detections: int,
) -> list[dict[str, object]]:
    """Fuse full-image and translated tile predictions in original coordinates."""

    return class_aware_nms(
        [*full_predictions, *tiled_predictions],
        iou_threshold=nms_iou,
        max_detections=max_detections,
    )


def predict_tiled_image(
    image: Image.Image,
    *,
    tile_size: int,
    overlap_ratio: float,
    batch_size: int,
    predict_batch: Callable[
        [list[Image.Image]], Sequence[Sequence[Mapping[str, object]]]
    ],
) -> dict[str, object]:
    """Crop one image in bounded batches and return original-coordinate detections."""

    if batch_size < 1:
        raise ValueError("batch size must be positive")
    windows = tile_windows(
        image_size=image.size,
        tile_size=tile_size,
        overlap_ratio=overlap_ratio,
    )
    predictions: list[dict[str, object]] = []
    for start in range(0, len(windows), batch_size):
        batch_windows = windows[start : start + batch_size]
        crops = [image.crop(window) for window in batch_windows]
        try:
            batch_predictions = list(predict_batch(crops))
        finally:
            for crop in crops:
                crop.close()
        if len(batch_predictions) != len(batch_windows):
            raise RuntimeError(
                "tile prediction count mismatch: "
                f"{len(batch_predictions)} != {len(batch_windows)}"
            )
        for window, local_predictions in zip(
            batch_windows, batch_predictions, strict=True
        ):
            predictions.extend(
                translate_predictions(
                    local_predictions,
                    offset=(window[0], window[1]),
                    image_size=image.size,
                )
            )
    return {"tile_count": len(windows), "predictions": predictions}
