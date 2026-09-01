"""OpenCV rendering for M6 track observations."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from helmet_safety.inference.opencv import CLASS_COLORS, _clipped_box

from .core import TrackObservation


def _dashed_rectangle(
    image: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    x1, y1 = start
    x2, y2 = end
    dash = max(4, thickness * 4)
    for x in range(x1, x2 + 1, dash * 2):
        cv2.line(image, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(image, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2 + 1, dash * 2):
        cv2.line(image, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(image, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def draw_tracks(
    image: np.ndarray,
    observations: Sequence[TrackObservation],
    *,
    active_tracks: int,
    lost_tracks: int,
    created_tracks: int,
    processing_fps: float,
    tracker_type: str,
    lite: bool = False,
) -> np.ndarray:
    """Annotate a frame with track boxes.

    ``lite=True`` is the real-time pipeline style: it draws the active
    (solid) and predicted (dashed) box rectangles with a short ``ID <n>``
    label plus a single status line, and skips lost boxes and the verbose
    panel. It is used on BOTH detection and in-between frames so the overlay
    style stays constant and does not flicker.
    """
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or image.shape[0] < 1
        or image.shape[1] < 1
    ):
        raise ValueError("drawing input must be a non-empty uint8 BGR image")
    output = image.copy()
    height, width = output.shape[:2]
    thickness = max(1, round(min(width, height) / 500))
    font_scale = max(0.35, min(width, height) / 700)
    occupied: list[tuple[int, int, int, int]] = []
    for observation in sorted(observations, key=lambda item: (item.xyxy[1], item.track_id)):
        if lite and observation.track_state == "lost":
            continue

        color = CLASS_COLORS.get(observation.class_name)
        if color is None:
            raise ValueError(f"unsupported tracking class: {observation.class_name}")
        x1, y1, x2, y2 = _clipped_box(observation.xyxy, width, height)
        if observation.is_prediction:
            _dashed_rectangle(output, (x1, y1), (x2, y2), color, thickness)
        else:
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness, cv2.LINE_8)
        suffix = " [PRED]" if observation.is_prediction else ""
        if lite:
            label = f"ID {observation.track_id}"
        else:
            label = f"ID {observation.track_id} | {observation.class_name} {observation.confidence:.2f}{suffix}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        label_width = min(width, text_width + 6)
        label_height = text_height + baseline + 4
        label_x = min(max(0, x1), max(0, width - label_width))
        label_y = max(0, y1 - label_height)
        candidate = (label_x, label_y, label_x + label_width - 1, label_y + label_height - 1)
        while any(_overlaps(candidate, prior) for prior in occupied) and candidate[3] + label_height < height:
            label_y += label_height
            candidate = (label_x, label_y, label_x + label_width - 1, label_y + label_height - 1)
        occupied.append(candidate)
        text_pos = (label_x + 3, min(height - 1, label_y + text_height + 2))
        if lite:
            # Outlined text (dark halo + light fill) instead of a filled label
            # rectangle: keeps the ID readable on any background while saving a
            # full-width fill for the frame budget on detection frames.
            cv2.putText(output, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
            cv2.putText(output, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        else:
            cv2.rectangle(output, candidate[:2], candidate[2:], color, -1)
            cv2.putText(output, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    if lite:
        # Detection frames: draw a single FPS line with no translucent overlay
        # so drawing stays ~1 ms (the full-frame budget is inference-dominated).
        line = f"{active_tracks} active | {max(0.0, processing_fps):.1f} FPS"
        cv2.putText(
            output,
            line,
            (8, max(thickness + 10, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.36, min(width, height) / 680),
            (245, 245, 245),
            thickness,
            cv2.LINE_AA,
        )
        return output
    lines = (
        f"active tracks: {active_tracks}",
        f"lost tracks: {lost_tracks}",
        f"created IDs: {created_tracks}",
        f"processing FPS: {max(0.0, processing_fps):.1f}",
        f"tracker: {tracker_type}",
    )
    panel_scale = max(0.36, min(width, height) / 680)
    sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, panel_scale, thickness)[0] for line in lines]
    line_height = max(size[1] for size in sizes) + 7
    panel_width = min(width, max(size[0] for size in sizes) + 16)
    panel_height = min(height, line_height * len(lines) + 8)
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (panel_width - 1, panel_height - 1), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, output, 0.28, 0.0, output)
    for index, line in enumerate(lines):
        cv2.putText(
            output,
            line,
            (8, min(height - 1, 7 + (index + 1) * line_height - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            panel_scale,
            (245, 245, 245),
            thickness,
            cv2.LINE_AA,
        )
    return output
