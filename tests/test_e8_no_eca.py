from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def test_no_eca_refinement_preserves_shape_and_residual_identity() -> None:
    """Catches a missing residual or any channel/spatial shape change."""

    from helmet_safety.training.e8_no_eca import NoECAFeatureRefinement

    module = NoECAFeatureRefinement(16).eval()
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


def test_e8_keeps_e7_convolutions_and_removes_only_eca() -> None:
    """Catches any E8 refinement change beyond deleting E7 channel attention."""

    from helmet_safety.training.e7_lfr import LiteFeatureRefinement
    from helmet_safety.training.e8_no_eca import NoECAFeatureRefinement

    e7 = LiteFeatureRefinement(16)
    e8 = NoECAFeatureRefinement(16)

    assert e8.depthwise.kernel_size == e7.depthwise.kernel_size == (3, 3)
    assert e8.depthwise.groups == e7.depthwise.groups == 16
    assert e8.pointwise.kernel_size == e7.pointwise.kernel_size == (1, 1)
    assert e8.pointwise.in_channels == e8.pointwise.out_channels == 16
    assert not hasattr(e8, "channel_attention")
    assert [name for name, _ in e8.named_children()] == [
        name for name, _ in e7.named_children() if name != "channel_attention"
    ]


def test_no_eca_refinement_backward_reaches_both_convolutions() -> None:
    """Catches a detached or bypassed DW/PW refinement branch."""

    from helmet_safety.training.e8_no_eca import NoECAFeatureRefinement

    module = NoECAFeatureRefinement(8).train()
    module(torch.randn(2, 8, 12, 12)).square().mean().backward()

    assert module.depthwise.weight.grad is not None
    assert module.pointwise.weight.grad is not None


def test_equivalent_size_bucket_boundaries_match_e8_protocol() -> None:
    """Catches gaps or overlaps in the required <=10, 10-20, 20-30, 30-50, >50 bins."""

    from helmet_safety.training.e8_no_eca import equivalent_size_bucket

    assert equivalent_size_bucket(0.0) == "<=10"
    assert equivalent_size_bucket(10.0) == "<=10"
    assert equivalent_size_bucket(10.0001) == "10-20"
    assert equivalent_size_bucket(20.0) == "10-20"
    assert equivalent_size_bucket(20.0001) == "20-30"
    assert equivalent_size_bucket(30.0) == "20-30"
    assert equivalent_size_bucket(30.0001) == "30-50"
    assert equivalent_size_bucket(50.0) == "30-50"
    assert equivalent_size_bucket(50.0001) == ">50"


def test_size_bucket_summary_counts_ground_truth_outcomes_and_prediction_fps() -> None:
    """Catches bucketed recall or false-positive counts assigned by the wrong box size."""

    from helmet_safety.training.e8_no_eca import summarize_size_buckets

    records = [
        {
            "image_id": "fixture.jpg",
            "ground_truth": [
                {"class_id": 0, "box": [0.0, 0.0, 10.0, 10.0]},
                {"class_id": 1, "box": [100.0, 0.0, 115.0, 15.0]},
                {"class_id": 0, "box": [200.0, 0.0, 225.0, 25.0]},
                {"class_id": 1, "box": [300.0, 0.0, 340.0, 40.0]},
                {"class_id": 0, "box": [400.0, 0.0, 460.0, 60.0]},
            ],
            "predictions": [
                {"class_id": 0, "box": [0.0, 0.0, 10.0, 10.0]},
                {"class_id": 0, "box": [100.0, 0.0, 115.0, 15.0]},
                {"class_id": 1, "box": [300.0, 0.0, 340.0, 40.0]},
                {"class_id": 0, "box": [400.0, 0.0, 460.0, 60.0]},
                {"class_id": 1, "box": [700.0, 0.0, 705.0, 5.0]},
            ],
        }
    ]

    summary = summarize_size_buckets(records, matching_iou=0.5)

    assert summary["overall"]["<=10"] == {"ground_truth": 1, "tp": 1, "fn": 0, "recall": 1.0}
    assert summary["overall"]["10-20"] == {"ground_truth": 1, "tp": 0, "fn": 1, "recall": 0.0}
    assert summary["overall"]["20-30"] == {"ground_truth": 1, "tp": 0, "fn": 1, "recall": 0.0}
    assert summary["overall"]["30-50"] == {"ground_truth": 1, "tp": 1, "fn": 0, "recall": 1.0}
    assert summary["overall"][">50"] == {"ground_truth": 1, "tp": 1, "fn": 0, "recall": 1.0}
    assert summary["helmet"]["20-30"]["fn"] == 1
    assert summary["no_helmet"]["10-20"]["fn"] == 1
    assert summary["false_positives_by_prediction_size"] == {
        "<=10": 1,
        "10-20": 1,
        "20-30": 0,
        "30-50": 0,
        ">50": 0,
    }


def test_e8_resume_history_uses_best_completed_epoch(tmp_path: Path) -> None:
    """Catches an E8 resume that restarts or restores patience from the wrong epoch."""

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "train" / "resume_e8_no_eca.py"
    spec = importlib.util.spec_from_file_location("resume_e8_no_eca", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    results_csv = tmp_path / "results.csv"
    results_csv.write_text(
        "epoch,metrics/mAP50-95(B)\n"
        "1,0.50\n"
        "2,0.49\n"
        "3,0.51\n",
        encoding="utf-8",
    )

    rows, best_epoch = module.read_history(results_csv)

    assert len(rows) == 3
    assert best_epoch == 3


def test_e8_must_keep_a_positive_10_30_gain_to_count_as_retained() -> None:
    """Catches calling baseline-equivalent E8 recall a retained E7 small-object benefit."""

    from scripts.evaluate.evaluate_e8_no_eca import build_ablation_judgments

    def experiment(recall_10_20: float, recall_20_30: float) -> dict[str, object]:
        buckets = {
            "<=10": {"ground_truth": 10, "tp": 5, "fn": 5, "recall": 0.5},
            "10-20": {"ground_truth": 10, "tp": round(10 * recall_10_20), "fn": 0, "recall": recall_10_20},
            "20-30": {"ground_truth": 10, "tp": round(10 * recall_20_30), "fn": 0, "recall": recall_20_30},
        }
        for name in ("10-20", "20-30"):
            buckets[name]["fn"] = buckets[name]["ground_truth"] - buckets[name]["tp"]
        return {
            "size_buckets": {"overall": buckets},
            "standard_val": {"overall": {"precision": 0.9, "f1": 0.9, "map50_95": 0.6}},
            "fixed_threshold": {"overall": {"fp": 100}},
            "class_confusions": 10,
        }

    judgments = build_ablation_judgments(
        {
            "E6": experiment(0.5, 0.5),
            "E7": experiment(0.6, 0.6),
            "E8": experiment(0.5, 0.5),
        }
    )

    assert judgments["e7_has_10_30_benefit"] is True
    assert judgments["e8_retains_e7_10_30_benefit"] is False
