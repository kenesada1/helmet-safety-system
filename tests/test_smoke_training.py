from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest
import yaml
from PIL import Image

from helmet_safety.training import smoke
from helmet_safety.training.smoke import ImageRecord, choose_subset, next_run_name, subset_box_counts


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _record(name: str, helmet: int, no_helmet: int) -> ImageRecord:
    return ImageRecord(Path(name), (helmet, no_helmet))


def test_choose_subset_is_seeded_and_includes_both_classes_and_a_mixed_image() -> None:
    """固定 seed 的子集必须可复现，并满足双类别与混合标注图片约束。"""

    records = [
        _record("a.jpg", 1, 0),
        _record("b.jpg", 0, 2),
        _record("c.jpg", 3, 4),
        _record("d.jpg", 2, 0),
        _record("e.jpg", 0, 1),
        _record("f.jpg", 1, 0),
    ]

    selected = choose_subset(records, count=4, seed=42)

    assert [record.image_path.name for record in selected] == ["c.jpg", "d.jpg", "b.jpg", "e.jpg"]
    assert subset_box_counts(selected) == {"helmet": 5, "no_helmet": 7}
    assert any(record.has_both_classes for record in selected)
    assert choose_subset(records, count=4, seed=42) == selected


def test_choose_subset_rejects_a_split_without_both_classes() -> None:
    """若候选 split 缺少任一类别，冒烟子集必须失败而不是假通过。"""

    records = [_record("a.jpg", 1, 0), _record("b.jpg", 2, 0)]

    with pytest.raises(ValueError, match="both class IDs 0 and 1"):
        choose_subset(records, count=2, seed=42)


def test_next_run_name_never_reuses_an_existing_training_directory(tmp_path: Path) -> None:
    """已有训练目录必须保留，新运行应选择下一个未占用名称。"""

    (tmp_path / "smoke_test").mkdir()
    (tmp_path / "smoke_test_002").mkdir()

    assert next_run_name(tmp_path, "smoke_test") == "smoke_test_003"


def test_create_smoke_dataset_writes_path_lists_without_copying_images(tmp_path: Path) -> None:
    """smoke dataset 只能写路径列表与配置，并准确统计选中图片的框数。"""

    assert hasattr(smoke, "create_smoke_dataset"), "create_smoke_dataset must be implemented"
    processed = tmp_path / "processed"
    definitions = {
        "train": [("a", "0\n"), ("b", "1\n"), ("c", "0\n1\n"), ("d", "0\n"), ("e", "1\n")],
        "val": [("v1", "0\n1\n"), ("v2", "0\n"), ("v3", "1\n")],
    }
    for split, items in definitions.items():
        (processed / "images" / split).mkdir(parents=True)
        (processed / "labels" / split).mkdir(parents=True)
        for stem, classes in items:
            Image.new("RGB", (8, 8), "white").save(processed / "images" / split / f"{stem}.jpg")
            label_lines = "".join(f"{class_id} 0.5 0.5 0.2 0.2\n" for class_id in classes.splitlines())
            (processed / "labels" / split / f"{stem}.txt").write_text(label_lines, encoding="utf-8")

    (processed / "images" / "train" / "truncated.jpg").write_bytes(b"\xff\xd8truncated")
    (processed / "labels" / "train" / "truncated.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )

    output = tmp_path / "smoke"
    report = smoke.create_smoke_dataset(processed, output, train_count=5, val_count=2, seed=42)

    assert report["splits"]["train"]["images"] == 5
    assert report["splits"]["val"]["images"] == 2
    assert report["splits"]["train"]["excluded_incomplete_jpegs"] == 1
    assert all(report["splits"][split]["boxes"][name] > 0 for split in ("train", "val") for name in ("helmet", "no_helmet"))
    train_paths = (output / "train.txt").read_text(encoding="utf-8").splitlines()
    assert all(Path(line).is_absolute() for line in train_paths)
    assert all("truncated.jpg" not in line for line in train_paths)
    assert not (output / "images").exists()
    dataset = yaml.safe_load((output / "dataset.yaml").read_text(encoding="utf-8"))
    assert dataset["names"] == {0: "helmet", 1: "no_helmet"}


def test_validate_training_outputs_accepts_complete_finite_epoch(tmp_path: Path) -> None:
    """完整 epoch 必须同时具备有限损失、验证指标、权重和两类可视化。"""

    assert hasattr(smoke, "validate_training_outputs"), "validate_training_outputs must be implemented"
    run_dir = tmp_path / "run"
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights" / "best.pt").write_bytes(b"best")
    (run_dir / "weights" / "last.pt").write_bytes(b"last")
    (run_dir / "train_batch0.jpg").write_bytes(b"train")
    (run_dir / "val_batch0_pred.jpg").write_bytes(b"val")
    (run_dir / "results.csv").write_text(
        "epoch,train/box_loss,train/cls_loss,train/dfl_loss,metrics/mAP50(B)\n"
        "1,1.25,0.75,0.5,0.1\n",
        encoding="utf-8",
    )

    result = smoke.validate_training_outputs(run_dir, expected_epochs=1)

    assert result["losses"] == {"box_loss": 1.25, "cls_loss": 0.75, "dfl_loss": 0.5}
    assert result["validation_completed"] is True


def test_validate_training_outputs_rejects_non_finite_loss(tmp_path: Path) -> None:
    """训练记录中出现 NaN 时必须失败，不得把冒烟测试标记为成功。"""

    assert hasattr(smoke, "validate_training_outputs"), "validate_training_outputs must be implemented"
    run_dir = tmp_path / "run"
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights" / "best.pt").write_bytes(b"best")
    (run_dir / "weights" / "last.pt").write_bytes(b"last")
    (run_dir / "train_batch0.jpg").write_bytes(b"train")
    (run_dir / "val_batch0_pred.jpg").write_bytes(b"val")
    (run_dir / "results.csv").write_text(
        "epoch,train/box_loss,train/cls_loss,train/dfl_loss,metrics/mAP50(B)\n"
        "1,nan,0.75,0.5,0.1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="finite"):
        smoke.validate_training_outputs(run_dir, expected_epochs=1)


def test_smoke_train_help_exposes_core_training_parameters() -> None:
    """训练入口应可直接运行，并公开数据、批量、epoch、设备和运行名参数。"""

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "train" / "train_smoke.py"), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    for option in ("--processed", "--epochs", "--batch", "--device", "--run-name"):
        assert option in result.stdout

