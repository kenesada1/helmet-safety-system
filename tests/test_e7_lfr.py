from __future__ import annotations

from pathlib import Path

import pytest
import torch


def test_lfr_preserves_shape_and_residual_identity_when_refinement_is_zero() -> None:
    """Catches channel/spatial changes or a missing residual connection."""

    from helmet_safety.training.e7_lfr import LiteFeatureRefinement

    module = LiteFeatureRefinement(16).eval()
    with torch.no_grad():
        module.depthwise.weight.zero_()
        module.pointwise.weight.zero_()
        module.pointwise_bn.weight.fill_(1.0)
        module.pointwise_bn.bias.zero_()
        module.pointwise_bn.running_mean.zero_()
        module.pointwise_bn.running_var.fill_(1.0)
    sample = torch.randn(2, 16, 31, 47)

    output = module(sample)

    assert output.shape == sample.shape
    assert torch.equal(output, sample)


def test_lfr_uses_depthwise_and_pointwise_convolutions_with_attention() -> None:
    """Catches replacement of the required lightweight operators."""

    from helmet_safety.training.e7_lfr import LiteFeatureRefinement

    module = LiteFeatureRefinement(16)

    assert module.depthwise.kernel_size == (3, 3)
    assert module.depthwise.groups == 16
    assert module.pointwise.kernel_size == (1, 1)
    assert module.pointwise.in_channels == module.pointwise.out_channels == 16
    assert module.channel_attention.conv.kernel_size == (3,)


def test_lfr_backward_reaches_depthwise_pointwise_and_attention() -> None:
    """Catches a detached or bypassed refinement branch."""

    from helmet_safety.training.e7_lfr import LiteFeatureRefinement

    module = LiteFeatureRefinement(8).train()
    module(torch.randn(2, 8, 12, 12)).square().mean().backward()

    assert module.depthwise.weight.grad is not None
    assert module.pointwise.weight.grad is not None
    assert module.channel_attention.conv.weight.grad is not None


def test_transfer_keeps_lfr_and_p2_detect_random_but_maps_e4_p3_to_e7_p3() -> None:
    """Catches accidental E4 Detect loading into the new P2/LFR parameters."""

    from helmet_safety.training.e7_lfr import transfer_e4_state

    source = {
        "model.0.conv.weight": torch.full((2, 2), 1.0),
        "model.23.cv2.0.0.conv.weight": torch.full((2, 2), 2.0),
        "model.23.dfl.conv.weight": torch.full((1, 4), 3.0),
    }
    target = {
        "model.0.conv.weight": torch.zeros((2, 2)),
        "model.26.depthwise.weight": torch.full((2, 1, 3, 3), 9.0),
        "model.27.cv2.0.0.conv.weight": torch.full((2, 2), 8.0),
        "model.27.cv2.1.0.conv.weight": torch.full((2, 2), 7.0),
        "model.27.dfl.conv.weight": torch.full((1, 4), 6.0),
    }

    transferred, report = transfer_e4_state(source, target)

    assert torch.equal(transferred["model.0.conv.weight"], source["model.0.conv.weight"])
    assert torch.equal(
        transferred["model.27.cv2.1.0.conv.weight"], source["model.23.cv2.0.0.conv.weight"]
    )
    assert torch.equal(transferred["model.27.dfl.conv.weight"], source["model.23.dfl.conv.weight"])
    assert torch.equal(transferred["model.26.depthwise.weight"], target["model.26.depthwise.weight"])
    assert torch.equal(transferred["model.27.cv2.0.0.conv.weight"], target["model.27.cv2.0.0.conv.weight"])
    assert report["random_initialized_tensors"] == [
        "model.26.depthwise.weight",
        "model.27.cv2.0.0.conv.weight",
    ]


def test_e7_comparison_reports_tiny_overall_errors_and_compute_judgments() -> None:
    """Catches a conclusion that is disconnected from the measured E6/E7 deltas."""

    from helmet_safety.training.e7_lfr import build_e7_comparison

    e6 = {
        "standard_val": {"overall": {"map50_95": 0.640, "f1": 0.930}},
        "fixed_threshold": {"overall": {"fp": 100}},
        "tiny": {"tp": 78, "fn": 50, "recall": 78 / 128},
        "class_confusions": 20,
        "parameters": 10_000_000,
        "gflops_at_960": 80.0,
    }
    e7 = {
        "standard_val": {"overall": {"map50_95": 0.645, "f1": 0.932}},
        "fixed_threshold": {"overall": {"fp": 102}},
        "tiny": {"tp": 83, "fn": 45, "recall": 83 / 128},
        "class_confusions": 18,
        "parameters": 10_018_000,
        "gflops_at_960": 82.0,
    }

    comparison = build_e7_comparison(e6, e7)

    assert comparison["tiny_false_negatives_reduced"] is True
    assert comparison["overall_detection_improved"] is True
    assert comparison["false_positives_increased"] is True
    assert comparison["class_confusion_increased"] is False
    assert comparison["compute_overhead_reasonable"] is True
    assert comparison["p2_lfr_effective"] is True
    assert comparison["deltas"]["tiny_fn"] == -5


def test_training_kwargs_match_e6_and_keep_nbs_64(tmp_path: Path) -> None:
    """Catches training drift beyond the selected feasible batch."""

    from helmet_safety.training.e7_lfr import build_training_kwargs

    kwargs = build_training_kwargs(
        data_yaml=tmp_path / "dataset.yaml",
        output_dir=tmp_path / "e7",
        device="0",
        workers=0,
        batch=4,
    )

    assert kwargs["epochs"] == 50
    assert kwargs["patience"] == 15
    assert kwargs["imgsz"] == 960
    assert kwargs["batch"] == 4
    assert kwargs["nbs"] == 64
    assert kwargs["seed"] == 42
    assert kwargs["deterministic"] is True
    assert kwargs["split"] == "val"
    assert kwargs["fraction"] == 1.0


def test_probe_hyperparameters_support_attribute_access_required_by_yolo_loss() -> None:
    """Catches the dict-vs-namespace failure seen when initializing loss from YAML."""

    from helmet_safety.training.e7_lfr import configure_probe_hyperparameters

    network = torch.nn.Identity()
    network.args = {"box": 999.0}  # type: ignore[attr-defined]

    configure_probe_hyperparameters(network)

    assert network.args.box == 7.5  # type: ignore[attr-defined]
    assert network.args.cls == 0.5  # type: ignore[attr-defined]
    assert network.args.dfl == 1.5  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("peak", "total", "expected"),
    [(4_000, 10_000, True), (5_000, 10_000, True), (5_001, 10_000, False)],
)
def test_probe_requires_fifty_percent_physical_vram_headroom(
    peak: int, total: int, expected: bool
) -> None:
    """Catches synthetic-to-dense-batch growth spilling into shared memory."""

    from helmet_safety.training.e7_lfr import fits_physical_vram

    assert fits_physical_vram(peak_allocated_bytes=peak, total_bytes=total) is expected


def test_register_lfr_module_makes_project_class_available_to_parser() -> None:
    """Catches failure to expose the project module to Ultralytics YAML parsing."""

    import ultralytics.nn.tasks as tasks

    from helmet_safety.training.e7_lfr import LiteFeatureRefinement, register_lfr_module

    previous = tasks.__dict__.pop("LiteFeatureRefinement", None)
    try:
        register_lfr_module()
        assert tasks.__dict__["LiteFeatureRefinement"] is LiteFeatureRefinement
    finally:
        if previous is None:
            tasks.__dict__.pop("LiteFeatureRefinement", None)
        else:
            tasks.__dict__["LiteFeatureRefinement"] = previous


def test_resume_plan_continues_incomplete_checkpoint_to_original_epoch_limit(tmp_path: Path) -> None:
    """Catches restarting the schedule or treating 50 as 50 additional epochs."""

    from helmet_safety.training.e7_lfr import build_resume_plan

    checkpoint_path = tmp_path / "last.pt"
    checkpoint_path.touch()
    checkpoint = {
        "epoch": 30,
        "best_fitness": 0.63845,
        "optimizer": {"state": {}},
        "scaler": {"scale": torch.tensor(65536.0)},
        "train_args": {
            "epochs": 50,
            "patience": 15,
            "batch": 2,
            "imgsz": 960,
            "nbs": 64,
            "seed": 42,
            "deterministic": True,
            "data": r"D:\datasets\SHWD\processed\dataset.yaml",
        },
    }

    plan = build_resume_plan(
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        completed_epochs=31,
        best_epoch=30,
        device="0",
        workers=0,
    )

    assert plan["completed_epochs"] == 31
    assert plan["next_epoch"] == 32
    assert plan["total_epochs"] == 50
    assert plan["remaining_epochs"] == 19
    assert plan["best_epoch"] == 30
    assert plan["best_fitness"] == pytest.approx(0.63845)
    assert plan["train_kwargs"] == {"resume": True, "device": "0", "workers": 0}


def test_resume_plan_rejects_checkpoint_csv_epoch_mismatch(tmp_path: Path) -> None:
    """Catches resuming from a checkpoint older than the visible training history."""

    from helmet_safety.training.e7_lfr import build_resume_plan

    checkpoint_path = tmp_path / "last.pt"
    checkpoint_path.touch()
    checkpoint = {
        "epoch": 29,
        "best_fitness": 0.63,
        "optimizer": {"state": {}},
        "scaler": {},
        "train_args": {
            "epochs": 50,
            "patience": 15,
            "batch": 2,
            "imgsz": 960,
            "nbs": 64,
            "seed": 42,
            "deterministic": True,
            "data": r"D:\datasets\SHWD\processed\dataset.yaml",
        },
    }

    with pytest.raises(RuntimeError, match="checkpoint/CSV epoch mismatch"):
        build_resume_plan(
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            completed_epochs=31,
            best_epoch=30,
            device="0",
            workers=0,
        )


def test_resume_early_stopping_restores_original_best_epoch_and_fitness() -> None:
    """Catches resetting patience after a paused run."""

    from ultralytics.utils.torch_utils import EarlyStopping

    from helmet_safety.training.e7_lfr import restore_early_stopping

    trainer = type("Trainer", (), {"stopper": EarlyStopping(patience=15)})()

    restore_early_stopping(trainer, best_epoch=30, best_fitness=0.63845)

    assert trainer.stopper.best_epoch == 30
    assert trainer.stopper.best_fitness == pytest.approx(0.63845)
    assert trainer.stopper.patience == 15


def test_e7_yaml_adds_only_lfr_between_e6_p2_and_detect() -> None:
    """Catches stride drift or unintended changes to the E6 P3/P4/P5 graph."""

    from ultralytics import YOLO

    from helmet_safety.training.e7_lfr import LiteFeatureRefinement, register_lfr_module

    project_root = Path(__file__).resolve().parents[1]
    register_lfr_module()
    e6 = YOLO(str(project_root / "configs" / "yolo11s-p2.yaml"), task="detect")
    e7 = YOLO(str(project_root / "configs" / "yolo11s-p2-lfr.yaml"), task="detect")

    assert [float(value) for value in e7.model.stride.tolist()] == [4.0, 8.0, 16.0, 32.0]
    assert isinstance(e7.model.model[26], LiteFeatureRefinement)
    assert list(e7.model.model[27].f) == [26, 16, 19, 22]
    assert [type(layer) for layer in e7.model.model[:26]] == [
        type(layer) for layer in e6.model.model[:26]
    ]


@pytest.mark.parametrize("bad_channels", [0, -8])
def test_lfr_rejects_nonpositive_channels(bad_channels: int) -> None:
    """Catches invalid groups/channel construction with an unclear framework error."""

    from helmet_safety.training.e7_lfr import LiteFeatureRefinement

    with pytest.raises(ValueError, match="channels must be positive"):
        LiteFeatureRefinement(bad_channels)
