from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
DEFAULT_WEIGHTS = ROOT / "artifacts/training/m45_yolo11s_e75_960_001/weights/best.pt"
DEFAULT_OUTPUT = ROOT / "artifacts/deployment/e4_yolo11s_960.onnx"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def validate_export_request(
    *, weights: Path, output: Path, imgsz: int, opset: int, force: bool
) -> tuple[Path, Path]:
    weights = weights.expanduser().resolve()
    output = output.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"weights do not exist: {weights}")
    if weights.suffix.lower() != ".pt":
        raise ValueError(f"weights must be a .pt file: {weights}")
    if output.suffix.lower() != ".onnx":
        raise ValueError(f"output must be a .onnx file: {output}")
    if imgsz < 1:
        raise ValueError("imgsz must be positive")
    if opset < 1:
        raise ValueError("opset must be positive")
    report_path = output.with_name(f"{output.stem}_export.json")
    smoke_paths = (
        output.with_name(f"{output.stem}_smoke_pt.jpg"),
        output.with_name(f"{output.stem}_smoke_onnx.jpg"),
    )
    existing = [path for path in (output, report_path, *smoke_paths) if path.exists()]
    if existing and not force:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing files without --force: {joined}")
    return weights, output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_exported_file(exported: Path, output: Path) -> None:
    """Atomically replace the requested output, including on Windows."""

    exported.replace(output)


def _shape(value: object) -> list[int | str | None]:
    return [item if isinstance(item, (int, str)) else None for item in list(value)]


def _find_smoke_image(weights: Path, requested: Path | None) -> Path:
    if requested is not None:
        image = requested.expanduser().resolve()
        if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
            raise FileNotFoundError(f"smoke image does not exist or is unsupported: {image}")
        return image

    args_path = weights.parent.parent / "args.yaml"
    if args_path.is_file():
        import yaml

        training_args = yaml.safe_load(args_path.read_text(encoding="utf-8")) or {}
        data_value = training_args.get("data")
        if data_value:
            dataset_yaml = Path(str(data_value)).expanduser()
            if not dataset_yaml.is_absolute():
                dataset_yaml = (args_path.parent / dataset_yaml).resolve()
            if dataset_yaml.is_file():
                dataset = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8")) or {}
                dataset_root = Path(str(dataset.get("path", dataset_yaml.parent))).expanduser()
                if not dataset_root.is_absolute():
                    dataset_root = (dataset_yaml.parent / dataset_root).resolve()
                val_value = dataset.get("val")
                if val_value:
                    val_path = Path(str(val_value)).expanduser()
                    if not val_path.is_absolute():
                        val_path = dataset_root / val_path
                    if val_path.is_dir():
                        candidates = sorted(
                            path for path in val_path.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
                        )
                        if candidates:
                            return candidates[0].resolve()

    fallback = ROOT / "artifacts/inference/m5_smoke/images/000002.jpg"
    if fallback.is_file():
        return fallback.resolve()
    raise FileNotFoundError("no real smoke image found; pass --image explicitly")


def _letterbox_tensor(image: np.ndarray, imgsz: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(imgsz / height, imgsz / width)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    left = (imgsz - resized_width) // 2
    top = (imgsz - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def _counts(detections: list[object]) -> dict[str, int]:
    result = {"helmet": 0, "no_helmet": 0}
    for detection in detections:
        result[str(getattr(detection, "class_name"))] += 1
    return result


def _confidence_comparison(pt_detections: list[object], onnx_detections: list[object]) -> dict[str, Any]:
    pt_conf = sorted((float(getattr(item, "confidence")) for item in pt_detections), reverse=True)
    onnx_conf = sorted((float(getattr(item, "confidence")) for item in onnx_detections), reverse=True)
    paired = min(len(pt_conf), len(onnx_conf))
    differences = [abs(pt_conf[index] - onnx_conf[index]) for index in range(paired)]
    return {
        "paired_by_descending_confidence": paired,
        "pt_top_confidences": [round(value, 6) for value in pt_conf[:10]],
        "onnx_top_confidences": [round(value, 6) for value in onnx_conf[:10]],
        "mean_absolute_difference": round(sum(differences) / paired, 8) if paired else None,
        "max_absolute_difference": round(max(differences), 8) if differences else None,
    }


def export_and_verify(
    *,
    weights: Path,
    output: Path,
    image_path: Path | None,
    imgsz: int,
    opset: int,
    force: bool,
) -> dict[str, Any]:
    weights, output = validate_export_request(
        weights=weights, output=output, imgsz=imgsz, opset=opset, force=force
    )
    smoke_image_path = _find_smoke_image(weights, image_path)
    image = cv2.imread(str(smoke_image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV could not read smoke image: {smoke_image_path}")

    output.parent.mkdir(parents=True, exist_ok=True)
    from onnx import TensorProto, checker, load
    import onnxruntime as ort
    from ultralytics import YOLO

    with tempfile.TemporaryDirectory(prefix="onnx-export-", dir=output.parent) as temp_name:
        temp_dir = Path(temp_name)
        temp_weights = temp_dir / "model.pt"
        shutil.copy2(weights, temp_weights)
        model = YOLO(str(temp_weights))
        exported = Path(
            model.export(
                format="onnx",
                imgsz=imgsz,
                batch=1,
                dynamic=False,
                opset=opset,
                half=False,
                simplify=False,
                nms=False,
                device="cpu",
            )
        ).resolve()
        if not exported.is_file():
            raise RuntimeError(f"Ultralytics did not create the expected ONNX file: {exported}")

        onnx_model = load(str(exported))
        checker.check_model(onnx_model)
        float16_initializers = [
            initializer.name
            for initializer in onnx_model.graph.initializer
            if initializer.data_type == TensorProto.FLOAT16
        ]
        if float16_initializers:
            raise RuntimeError("FP16 initializers found in requested FP32 export")

        session = ort.InferenceSession(str(exported), providers=["CPUExecutionProvider"])
        inputs = [
            {"name": item.name, "shape": _shape(item.shape), "type": item.type}
            for item in session.get_inputs()
        ]
        outputs = [
            {"name": item.name, "shape": _shape(item.shape), "type": item.type}
            for item in session.get_outputs()
        ]
        if len(inputs) != 1:
            raise RuntimeError(f"expected one ONNX input, found {len(inputs)}")
        expected_shape = [1, 3, imgsz, imgsz]
        if inputs[0]["shape"] != expected_shape:
            raise RuntimeError(
                f"expected static ONNX input shape {expected_shape}, found {inputs[0]['shape']}"
            )
        raw_outputs = session.run(None, {inputs[0]["name"]: _letterbox_tensor(image, imgsz)})
        if not raw_outputs or not all(np.isfinite(value).all() for value in raw_outputs):
            raise RuntimeError("ONNX Runtime smoke inference returned empty or non-finite output")

        from helmet_safety.inference.opencv import OpenCVDetector, draw_detections

        pt_detector = OpenCVDetector(
            weights=weights, device="cpu", imgsz=imgsz, conf=0.25, iou=0.70, max_det=300
        )
        onnx_detector = OpenCVDetector(
            weights=exported, device="cpu", imgsz=imgsz, conf=0.25, iou=0.70, max_det=300
        )
        pt_result = pt_detector.predict_bgr(image)
        onnx_result = onnx_detector.predict_bgr(image)
        if not pt_result.detections or not onnx_result.detections:
            raise RuntimeError(
                "PT/ONNX smoke comparison requires both backends to return detection boxes"
            )
        for result in (pt_result, onnx_result):
            if any(item.class_name not in {"helmet", "no_helmet"} for item in result.detections):
                raise RuntimeError("smoke inference returned an unexpected class name")

        pt_annotated, _ = draw_detections(image, pt_result.detections, processing_fps=0.0)
        onnx_annotated, _ = draw_detections(image, onnx_result.detections, processing_fps=0.0)
        pt_image_path = output.with_name(f"{output.stem}_smoke_pt.jpg")
        onnx_image_path = output.with_name(f"{output.stem}_smoke_onnx.jpg")
        if not cv2.imwrite(str(pt_image_path), pt_annotated):
            raise RuntimeError(f"failed to save PT smoke image: {pt_image_path}")
        if not cv2.imwrite(str(onnx_image_path), onnx_annotated):
            raise RuntimeError(f"failed to save ONNX smoke image: {onnx_image_path}")

        replace_exported_file(exported, output)

    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "success": True,
        "config": {
            "format": "onnx",
            "precision": "FP32",
            "batch": 1,
            "imgsz": imgsz,
            "dynamic": False,
            "opset": opset,
            "conf": 0.25,
            "iou": 0.70,
            "max_det": 300,
            "classes": {"0": "helmet", "1": "no_helmet"},
        },
        "files": {
            "pt": str(weights),
            "onnx": str(output),
            "pt_size_bytes": weights.stat().st_size,
            "onnx_size_bytes": output.stat().st_size,
            "pt_sha256": _sha256(weights),
            "onnx_sha256": _sha256(output),
        },
        "validation": {
            "onnx_checker": "passed",
            "fp16_initializers": 0,
            "onnxruntime_load": "passed",
            "onnxruntime_provider": session.get_providers(),
            "inputs": inputs,
            "outputs": outputs,
            "raw_smoke_output_shapes": [list(value.shape) for value in raw_outputs],
            "raw_smoke_outputs_finite": True,
        },
        "smoke_inference": {
            "image": str(smoke_image_path),
            "pt_device": "cpu",
            "onnx_device": "cpu",
            "pt_detection_count": len(pt_result.detections),
            "onnx_detection_count": len(onnx_result.detections),
            "pt_class_counts": _counts(pt_result.detections),
            "onnx_class_counts": _counts(onnx_result.detections),
            "confidence_comparison": _confidence_comparison(
                pt_result.detections, onnx_result.detections
            ),
            "pt_annotated_image": str(pt_image_path),
            "onnx_annotated_image": str(onnx_image_path),
            "onnx_image_saved": onnx_image_path.is_file(),
            "obvious_anomaly": False,
        },
    }
    report_path = output.with_name(f"{output.stem}_export.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"PT size: {report['files']['pt_size_bytes']} bytes")
    print(f"ONNX size: {report['files']['onnx_size_bytes']} bytes")
    print(f"PT SHA256: {report['files']['pt_sha256']}")
    print(f"ONNX SHA256: {report['files']['onnx_sha256']}")
    print(f"ONNX inputs: {inputs}")
    print(f"ONNX outputs: {outputs}")
    print("Export successful: true")
    print(f"Report: {report_path}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export E4 YOLO weights to static FP32 ONNX")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image", type=Path, help="real image for smoke inference")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    export_and_verify(
        weights=args.weights,
        output=args.output,
        image_path=args.image,
        imgsz=args.imgsz,
        opset=args.opset,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
