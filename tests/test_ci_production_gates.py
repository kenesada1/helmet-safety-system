from __future__ import annotations

from pathlib import Path

import cv2
import pytest
import yaml


def _ci_smoke() -> object:
    try:
        from helmet_safety.training import ci_smoke as module
    except ImportError:
        pytest.fail("helmet_safety.training.ci_smoke must build a portable CI dataset")
    return module


def _dependency_policy() -> object:
    try:
        from helmet_safety.quality import dependencies as module
    except ImportError:
        pytest.fail("helmet_safety.quality.dependencies must enforce production dependency policy")
    return module


def test_ci_smoke_dataset_is_self_contained_and_covers_both_classes(
    tmp_path: Path,
) -> None:
    ci_smoke = _ci_smoke()

    report = ci_smoke.create_synthetic_dataset(tmp_path / "dataset", seed=42)

    dataset_yaml = Path(report["dataset_yaml"])
    config = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    assert config["names"] == {0: "helmet", 1: "no_helmet"}
    assert Path(config["path"]) == (tmp_path / "dataset").resolve()
    assert report["splits"] == {
        "train": {"images": 4, "boxes": {"helmet": 4, "no_helmet": 4}},
        "val": {"images": 2, "boxes": {"helmet": 2, "no_helmet": 2}},
    }
    for split, expected_images in (("train", 4), ("val", 2)):
        images = sorted((tmp_path / "dataset" / "images" / split).glob("*.jpg"))
        labels = sorted((tmp_path / "dataset" / "labels" / split).glob("*.txt"))
        assert len(images) == len(labels) == expected_images
        assert all(cv2.imread(str(path)) is not None for path in images)
        assert all(len(path.read_text(encoding="utf-8").splitlines()) == 2 for path in labels)


def test_ci_smoke_dataset_refuses_to_overwrite_prior_evidence(tmp_path: Path) -> None:
    ci_smoke = _ci_smoke()
    output = tmp_path / "dataset"
    ci_smoke.create_synthetic_dataset(output, seed=42)

    with pytest.raises(FileExistsError, match="overwrite"):
        ci_smoke.create_synthetic_dataset(output, seed=42)


def test_ci_force_cleanup_only_removes_marked_outputs_below_artifacts(
    tmp_path: Path,
) -> None:
    ci_smoke = _ci_smoke()
    artifacts_root = tmp_path / "artifacts"
    marked = artifacts_root / "prior-smoke"
    marked.mkdir(parents=True)
    (marked / "smoke_report.json").write_text("{}", encoding="utf-8")
    ci_smoke.reset_generated_output(marked, artifacts_root=artifacts_root)
    assert not marked.exists()

    unmarked = artifacts_root / "user-files"
    unmarked.mkdir()
    (unmarked / "important.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="generated marker"):
        ci_smoke.reset_generated_output(unmarked, artifacts_root=artifacts_root)
    assert (unmarked / "important.txt").is_file()

    with pytest.raises(ValueError, match="below artifacts"):
        ci_smoke.reset_generated_output(artifacts_root, artifacts_root=artifacts_root)


def test_dependency_policy_catches_unpinned_ultralytics_and_missing_service_extra(
    tmp_path: Path,
) -> None:
    policy = _dependency_policy()
    project = tmp_path / "pyproject.toml"
    project.write_text(
        """
[project]
description = "Helmet safety detection system"
dependencies = ["Pillow>=10"]

[project.optional-dependencies]
train = ["ultralytics>=8"]
inference = ["ultralytics==8.4.120", "opencv-python>=4.10"]
deployment = ["ultralytics==8.4.120", "onnxruntime>=1.19"]
""".strip(),
        encoding="utf-8",
    )

    violations = policy.validate_dependency_policy(project)

    assert "train must pin ultralytics==8.4.120" in violations
    assert "missing optional dependency group: service" in violations


def test_repository_dependency_policy_passes() -> None:
    policy = _dependency_policy()
    project = Path(__file__).resolve().parents[1] / "pyproject.toml"

    assert policy.validate_dependency_policy(project) == []
