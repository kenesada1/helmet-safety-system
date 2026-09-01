from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_tiled_script() -> object:
    script_path = PROJECT_ROOT / "scripts" / "evaluate" / "evaluate_e4_tiled.py"
    spec = importlib.util.spec_from_file_location("evaluate_e4_tiled", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tile_windows_cover_image_and_anchor_last_window_to_edges() -> None:
    tiling = importlib.import_module("helmet_safety.inference.tiling")

    windows = tiling.tile_windows(
        image_size=(1300, 600),
        tile_size=640,
        overlap_ratio=0.20,
    )

    assert windows == [
        (0, 0, 640, 600),
        (512, 0, 1152, 600),
        (660, 0, 1300, 600),
    ]


@pytest.mark.parametrize(
    ("image_size", "tile_size", "overlap_ratio", "message"),
    [
        ((0, 600), 640, 0.20, "image dimensions"),
        ((1300, 600), 0, 0.20, "tile size"),
        ((1300, 600), 640, 1.0, "overlap ratio"),
    ],
)
def test_tile_windows_reject_invalid_geometry(
    image_size: tuple[int, int],
    tile_size: int,
    overlap_ratio: float,
    message: str,
) -> None:
    tiling = importlib.import_module("helmet_safety.inference.tiling")

    with pytest.raises(ValueError, match=message):
        tiling.tile_windows(
            image_size=image_size,
            tile_size=tile_size,
            overlap_ratio=overlap_ratio,
        )


def test_translate_predictions_maps_to_original_coordinates_and_clips_bounds() -> None:
    tiling = importlib.import_module("helmet_safety.inference.tiling")
    local_predictions = [
        {
            "class_id": 1,
            "confidence": 0.81,
            "box": [-5.0, 5.0, 100.0, 120.0],
        }
    ]

    translated = tiling.translate_predictions(
        local_predictions,
        offset=(600, 0),
        image_size=(1300, 100),
    )

    assert translated == [
        {
            "class_id": 1,
            "confidence": 0.81,
            "box": [595.0, 5.0, 700.0, 100.0],
        }
    ]
    assert local_predictions[0]["box"] == [-5.0, 5.0, 100.0, 120.0]


def test_class_aware_nms_suppresses_only_lower_confidence_same_class_boxes() -> None:
    tiling = importlib.import_module("helmet_safety.inference.tiling")
    predictions = [
        {"class_id": 1, "confidence": 0.90, "box": [0.0, 0.0, 10.0, 10.0]},
        {"class_id": 1, "confidence": 0.80, "box": [0.0, 0.0, 10.0, 10.0]},
        {"class_id": 0, "confidence": 0.70, "box": [0.0, 0.0, 10.0, 10.0]},
        {"class_id": 1, "confidence": 0.60, "box": [20.0, 20.0, 30.0, 30.0]},
    ]

    kept = tiling.class_aware_nms(
        predictions,
        iou_threshold=0.50,
        max_detections=300,
    )

    assert kept == [predictions[0], predictions[2], predictions[3]]


def test_class_aware_nms_rejects_invalid_limits() -> None:
    tiling = importlib.import_module("helmet_safety.inference.tiling")

    with pytest.raises(ValueError, match="IoU threshold"):
        tiling.class_aware_nms([], iou_threshold=0.0, max_detections=300)
    with pytest.raises(ValueError, match="max detections"):
        tiling.class_aware_nms([], iou_threshold=0.5, max_detections=0)


def test_merge_hybrid_predictions_prefers_stronger_tile_duplicate() -> None:
    tiling = importlib.import_module("helmet_safety.inference.tiling")
    full = [
        {"class_id": 1, "confidence": 0.60, "box": [0.0, 0.0, 10.0, 10.0]},
        {"class_id": 0, "confidence": 0.55, "box": [30.0, 30.0, 40.0, 40.0]},
    ]
    tiled = [
        {"class_id": 1, "confidence": 0.90, "box": [0.0, 0.0, 10.0, 10.0]},
    ]

    merged = tiling.merge_hybrid_predictions(
        full,
        tiled,
        nms_iou=0.50,
        max_detections=300,
    )

    assert merged == [tiled[0], full[1]]


def test_predict_tiled_image_batches_crops_and_maps_every_result() -> None:
    tiling = importlib.import_module("helmet_safety.inference.tiling")
    image = Image.new("RGB", (1300, 600), "white")

    def predict_batch(crops: list[Image.Image]) -> list[list[dict[str, object]]]:
        return [
            [
                {
                    "class_id": 1,
                    "confidence": 0.75,
                    "box": [0.0, 0.0, 10.0, 10.0],
                }
            ]
            for _ in crops
        ]

    result = tiling.predict_tiled_image(
        image,
        tile_size=640,
        overlap_ratio=0.20,
        batch_size=2,
        predict_batch=predict_batch,
    )

    assert result["tile_count"] == 3
    assert [item["box"] for item in result["predictions"]] == [
        [0.0, 0.0, 10.0, 10.0],
        [512.0, 0.0, 522.0, 10.0],
        [660.0, 0.0, 670.0, 10.0],
    ]


def test_e4_tiled_val_cli_locks_candidate_c_and_slice_protocol() -> None:
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "evaluate" / "evaluate_e4_tiled.py"), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    for fixed in (
        "E4 best.pt",
        "val only",
        "imgsz=960",
        "conf=0.20",
        "NMS IoU=0.50",
        "tiles=768/640",
        "overlap=0.20",
        "matching IoU=0.5",
    ):
        assert fixed in result.stdout
    for forbidden in (
        "--weights",
        "--split",
        "--imgsz",
        "--conf",
        "--nms-iou",
        "--tile-size",
        "--overlap",
    ):
        assert forbidden not in result.stdout


def test_candidate_c_tiny_anchor_reads_prefixed_report_fields() -> None:
    script = _load_tiled_script()
    row = {
        "tiny_tp": 79,
        "tiny_fn": 49,
        "tiny_helmet_tp": 5,
        "tiny_helmet_fn": 9,
        "tiny_no_helmet_tp": 74,
        "tiny_no_helmet_fn": 40,
    }

    assert script.candidate_c_tiny_anchor(row) == {
        "tp": 79,
        "fn": 49,
        "helmet_tp": 5,
        "helmet_fn": 9,
        "no_helmet_tp": 74,
        "no_helmet_fn": 40,
    }


def test_full_scope_promotes_only_the_better_640_tile_configuration() -> None:
    script = _load_tiled_script()

    assert script.tile_configs_for_scope("tiny") == (
        ("S1_hybrid_768", 768),
        ("S2_hybrid_640", 640),
    )
    assert script.tile_configs_for_scope("full") == (("S2_hybrid_640", 640),)
