from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_e6_model_has_p2_p3_p4_p5_detection_strides() -> None:
    from ultralytics import YOLO

    config = PROJECT_ROOT / "configs" / "yolo11s-p2.yaml"
    model = YOLO(str(config), task="detect")

    assert model.model.stride.detach().cpu().tolist() == [4.0, 8.0, 16.0, 32.0]
    assert model.model.model[-1].f == [25, 16, 19, 22]


def test_transfer_keeps_new_p2_tensors_random_and_maps_existing_detect_heads() -> None:
    e6 = importlib.import_module("helmet_safety.training.e6_p2")
    source = {
        "model.0.conv.weight": torch.full((2, 2), 1.0),
        "model.23.cv2.0.0.conv.weight": torch.full((2, 2), 2.0),
        "model.23.cv3.2.2.bias": torch.full((2,), 3.0),
        "model.23.dfl.conv.weight": torch.full((1, 2), 4.0),
    }
    target = {
        "model.0.conv.weight": torch.zeros((2, 2)),
        "model.23.conv.weight": torch.full((2, 2), 9.0),
        "model.26.cv2.0.0.conv.weight": torch.full((2, 2), 8.0),
        "model.26.cv2.1.0.conv.weight": torch.zeros((2, 2)),
        "model.26.cv3.3.2.bias": torch.zeros((2,)),
        "model.26.dfl.conv.weight": torch.zeros((1, 2)),
    }

    transferred, report = e6.transfer_e4_state(source, target)

    assert torch.equal(transferred["model.0.conv.weight"], source["model.0.conv.weight"])
    assert torch.equal(
        transferred["model.26.cv2.1.0.conv.weight"], source["model.23.cv2.0.0.conv.weight"]
    )
    assert torch.equal(transferred["model.26.cv3.3.2.bias"], source["model.23.cv3.2.2.bias"])
    assert torch.equal(transferred["model.26.dfl.conv.weight"], source["model.23.dfl.conv.weight"])
    assert torch.equal(transferred["model.23.conv.weight"], target["model.23.conv.weight"])
    assert torch.equal(
        transferred["model.26.cv2.0.0.conv.weight"], target["model.26.cv2.0.0.conv.weight"]
    )
    assert report["exact_tensor_count"] == 1
    assert report["remapped_tensor_count"] == 3
    assert set(report["random_initialized_tensors"]) == {
        "model.23.conv.weight",
        "model.26.cv2.0.0.conv.weight",
    }


def test_real_e4_transfer_randomizes_only_new_p2_fusion_and_detect_branch() -> None:
    from ultralytics import YOLO

    e6 = importlib.import_module("helmet_safety.training.e6_p2")
    source = YOLO(
        str(PROJECT_ROOT / "artifacts" / "training" / "m45_yolo11s_e75_960_001" / "weights" / "best.pt")
    )
    target = YOLO(str(PROJECT_ROOT / "configs" / "yolo11s-p2.yaml"), task="detect")

    _, report = e6.transfer_e4_state(source.model.state_dict(), target.model.state_dict())
    unexpected = [
        name
        for name in report["random_initialized_tensors"]
        if not name.startswith("model.25.")
        and not name.startswith("model.26.cv2.0.")
        and not name.startswith("model.26.cv3.0.")
    ]

    assert unexpected == []


def test_e6_training_contract_rejects_overwrite_or_nonfixed_protocol(tmp_path: Path) -> None:
    e6 = importlib.import_module("helmet_safety.training.e6_p2")
    output = tmp_path / "e6_yolo11s_p2_001"
    kwargs = e6.build_training_kwargs(
        data_yaml=Path(r"D:\datasets\SHWD\processed\dataset.yaml"),
        output_dir=output,
        device="0",
        workers=0,
    )

    assert kwargs["epochs"] == 50
    assert kwargs["patience"] == 15
    assert kwargs["imgsz"] == 960
    assert kwargs["batch"] == 2
    assert kwargs["seed"] == 42
    assert kwargs["split"] == "val"
    assert kwargs["fraction"] == 1.0
    assert kwargs["exist_ok"] is False
    # train() receives a temporary E6 initialization checkpoint containing the
    # explicit E4 transfer, so Ultralytics must retain those in-memory weights.
    assert kwargs["pretrained"] is True
    assert kwargs["split"] != "test"
    assert "test" not in str(kwargs["data"]).lower()

    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        e6.assert_output_available(output)


def test_e6_comparison_applies_explicit_tiny_fp_and_overall_rules() -> None:
    e6 = importlib.import_module("helmet_safety.training.e6_p2")
    e4 = {
        "standard_val": {"overall": {"map50_95": 0.64}},
        "fixed_threshold": {"overall": {"precision": 0.90, "f1": 0.91, "fp": 100}},
        "tiny": {"recall": 0.60, "tp": 60, "fn": 40},
    }
    candidate = {
        "standard_val": {"overall": {"map50_95": 0.635}},
        "fixed_threshold": {"overall": {"precision": 0.895, "f1": 0.905, "fp": 109}},
        "tiny": {"recall": 0.70, "tp": 70, "fn": 30},
    }

    comparison = e6.build_e6_comparison(e4, candidate)

    assert comparison["tiny_improved"] is True
    assert comparison["obvious_false_positive_increase"] is False
    assert comparison["obvious_overall_decline"] is False
    assert comparison["deltas"]["tiny_recall_pp"] == pytest.approx(10.0)
