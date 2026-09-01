"""Image and directory inference flow for M5."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .opencv import OpenCVDetector, draw_detections


SUPPORTED_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp"})


def collect_image_paths(source: Path | str) -> list[Path]:
    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"image input does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError(
                f"unsupported image extension {path.suffix!r}; expected JPG/JPEG/PNG/BMP"
            )
        return [path]
    if not path.is_dir():
        raise ValueError(f"image input is neither a file nor a directory: {path}")
    images = sorted(
        (
            candidate.resolve()
            for candidate in path.iterdir()
            if candidate.is_file()
            and candidate.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ),
        key=lambda candidate: candidate.name.casefold(),
    )
    if not images:
        raise ValueError(f"image directory has no supported images: {path}")
    return images


def _report_path(source: Path, output_dir: Path) -> Path:
    return (
        output_dir / f"{source.stem}.json"
        if source.is_file()
        else output_dir / "report.json"
    )


def _assert_writable_targets(
    sources: Sequence[Path], output_dir: Path, report_path: Path, force: bool
) -> None:
    for source in sources:
        destination = (output_dir / source.name).resolve()
        if destination == source.resolve():
            raise ValueError(f"output would overwrite input image: {source}")
        if destination.exists() and not force:
            raise FileExistsError(
                f"output already exists: {destination}; pass --force to overwrite"
            )
    if report_path.exists() and not force:
        raise FileExistsError(
            f"report already exists: {report_path}; pass --force to overwrite"
        )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _read_bgr(path: Path) -> np.ndarray | None:
    try:
        encoded = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def _write_bgr(path: Path, image: np.ndarray) -> None:
    extension = path.suffix.lower()
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise RuntimeError(f"OpenCV failed to encode output image: {path}")
    try:
        encoded.tofile(str(path))
    except OSError as exc:
        raise RuntimeError(f"failed to write output image {path}: {exc}") from exc


def run_image_inference(
    detector: OpenCVDetector,
    source: Path | str,
    output_dir: Path | str,
    *,
    force: bool = False,
    show: bool = False,
) -> dict[str, object]:
    source_path = Path(source).expanduser().resolve()
    image_paths = collect_image_paths(source_path)
    destination_dir = Path(output_dir).expanduser().resolve()
    report_path = _report_path(source_path, destination_dir)
    _assert_writable_targets(image_paths, destination_dir, report_path, force)
    destination_dir.mkdir(parents=True, exist_ok=True)

    items: list[dict[str, object]] = []
    failed = 0
    total_detections = 0
    helmet_count = 0
    no_helmet_count = 0
    total_inference_seconds = 0.0
    try:
        for image_path in image_paths:
            image = _read_bgr(image_path)
            if image is None:
                message = f"unable to read input image with OpenCV: {image_path}"
                if source_path.is_file():
                    raise ValueError(message)
                failed += 1
                items.append({"input_path": str(image_path), "status": "failed", "error": message})
                continue
            result = detector.predict_bgr(image)
            fps = 1.0 / result.inference_seconds if result.inference_seconds > 0 else 0.0
            annotated, counts = draw_detections(
                image, result.detections, processing_fps=fps
            )
            output_path = destination_dir / image_path.name
            _write_bgr(output_path, annotated)
            total_inference_seconds += result.inference_seconds
            total_detections += len(result.detections)
            helmet_count += counts["helmet"]
            no_helmet_count += counts["no_helmet"]
            items.append(
                {
                    "input_path": str(image_path),
                    "output_path": str(output_path),
                    "status": "success",
                    "inference_seconds": result.inference_seconds,
                    "detections": [item.to_dict() for item in result.detections],
                    "counts": counts,
                }
            )
            if show:
                cv2.imshow("helmet-safety M5", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if show:
            cv2.destroyAllWindows()

    successful = sum(item["status"] == "success" for item in items)
    report: dict[str, object] = {
        "schema_version": "m5-image-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source_path),
        "output_directory": str(destination_dir),
        "summary": {
            "successful": successful,
            "failed": failed,
            "total_detections": total_detections,
            "helmet": helmet_count,
            "no_helmet": no_helmet_count,
            "average_inference_seconds": (
                total_inference_seconds / successful if successful else 0.0
            ),
        },
        "items": items,
        "note": "Counts are per-image detection boxes, not unique people or violation events.",
    }
    _write_json(report_path, report)
    return report
