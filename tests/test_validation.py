from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest
from PIL import Image

from conftest import make_voc_xml
from helmet_safety.data.convert import convert_dataset
from helmet_safety.data.validate import (
    compare_annotation_counts,
    generate_visualizations,
    parse_yolo_line,
    pixel_roundtrip_error,
    validate_dataset,
)


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("0 0.5 0.5 0.2", "exactly 5 columns"),
        ("2 0.5 0.5 0.2 0.2", "class_id"),
        ("0 1.1 0.5 0.2 0.2", "between 0 and 1"),
        ("1 0.5 0.5 0 0.2", "width and height must be greater than 0"),
    ],
)
def test_invalid_yolo_lines_are_rejected(line: str, message: str) -> None:
    """列数、类别、坐标范围或框尺寸非法的 YOLO 行都必须被拒绝。"""

    with pytest.raises(ValueError, match=message):
        parse_yolo_line(line, label_path=Path("bad.txt"), line_number=3)


def test_valid_yolo_line_parses() -> None:
    """合法 YOLO 行应解析成一个类别整数和四个浮点坐标。"""

    assert parse_yolo_line("1 0.500000 0.250000 0.200000 0.100000") == (
        1,
        0.5,
        0.25,
        0.2,
        0.1,
    )


def test_roundtrip_from_six_decimal_yolo_coordinates_stays_within_pixel_tolerance() -> None:
    """保留六位小数后的 YOLO 框反算到像素时，误差应保持在合理阈值内。"""

    error = pixel_roundtrip_error(
        voc_box=(17, 23, 48, 91),
        yolo_box=(0.216667, 0.285000, 0.206667, 0.340000),
        image_width=150,
        image_height=200,
    )

    assert error <= 0.001


def test_before_after_counts_allow_only_recorded_dogs_to_disappear() -> None:
    """有效类别数量一致且 dog 跳过数有记录时，前后计数核对应通过。"""

    result = compare_annotation_counts(
        before={"hat": 7, "person": 5, "dog": 3},
        after={"helmet": 7, "no_helmet": 5},
        recorded_ignored_dogs=3,
    )

    assert result["valid_targets_match"] is True
    assert result["ignored_dogs_match"] is True


def test_before_after_count_loss_is_detected() -> None:
    """任一 hat/person 有效目标在转换后减少时，计数核对应失败。"""

    result = compare_annotation_counts(
        before={"hat": 7, "person": 5, "dog": 3},
        after={"helmet": 6, "no_helmet": 5},
        recorded_ignored_dogs=3,
    )

    assert result["valid_targets_match"] is False


def test_converted_fixture_validates_counts_and_pixel_roundtrip(tmp_path: Path) -> None:
    """小型完整数据集转换后应通过文件、数量和逐框像素回算验证。"""

    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    report_dir = tmp_path / "audit"
    (raw / "Annotations").mkdir(parents=True)
    (raw / "JPEGImages").mkdir()
    (raw / "ImageSets" / "Main").mkdir(parents=True)
    for image_id, objects in {
        "train_image": (("hat", 17, 23, 48, 91),),
        "val_image": (("person", 10, 20, 50, 80),),
        "test_image": (("dog", 1, 2, 10, 20),),
    }.items():
        Image.new("RGB", (150, 200), "white").save(raw / "JPEGImages" / f"{image_id}.jpg")
        (raw / "Annotations" / f"{image_id}.xml").write_text(
            make_voc_xml(filename=f"{image_id}.jpg", width=150, objects=objects), encoding="utf-8"
        )
    split_ids = {"train": ["train_image"], "val": ["val_image"], "test": ["test_image"]}
    for split, ids in {**split_ids, "trainval": ["train_image", "val_image"]}.items():
        (raw / "ImageSets" / "Main" / f"{split}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    convert_dataset(raw, processed)

    report = validate_dataset(raw, processed, report_dir)

    assert report["status"] == "passed"
    assert report["counts"]["valid_targets_match"] is True
    assert report["counts"]["ignored_dogs_match"] is True
    assert report["roundtrip"]["maximum_pixel_error"] <= 0.001
    assert (report_dir / "validation_report.json").is_file()
    assert (report_dir / "validation_report.md").is_file()


def test_visualization_generation_does_not_modify_processed_images(tmp_path: Path) -> None:
    """可视化应写入独立副本，并保证 processed 训练图片哈希不变。"""

    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    visualizations = tmp_path / "visualizations"
    (raw / "Annotations").mkdir(parents=True)
    (raw / "JPEGImages").mkdir()
    (raw / "ImageSets" / "Main").mkdir(parents=True)
    split_ids = {"train": ["a"], "val": ["b"], "test": ["c"]}
    for image_id, class_name in (("a", "hat"), ("b", "person"), ("c", "hat")):
        Image.new("RGB", (100, 200), "white").save(raw / "JPEGImages" / f"{image_id}.jpg")
        (raw / "Annotations" / f"{image_id}.xml").write_text(
            make_voc_xml(filename=f"{image_id}.jpg", objects=((class_name, 10, 20, 50, 80),)),
            encoding="utf-8",
        )
    for split, ids in {**split_ids, "trainval": ["a", "b"]}.items():
        (raw / "ImageSets" / "Main" / f"{split}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    convert_dataset(raw, processed)
    processed_image = processed / "images" / "train" / "a.jpg"
    original_hash = sha256(processed_image.read_bytes()).hexdigest()

    report = generate_visualizations(processed, visualizations, samples_per_split=1, seed=42)

    assert report["seed"] == 42
    assert [path.name for path in (visualizations / "train").glob("*.jpg")] == ["a.jpg"]
    assert [path.name for path in (visualizations / "val").glob("*.jpg")] == ["b.jpg"]
    assert [path.name for path in (visualizations / "test").glob("*.jpg")] == ["c.jpg"]
    assert sha256(processed_image.read_bytes()).hexdigest() == original_hash


def test_validation_rejects_processed_jpeg_with_bytes_after_eoi(tmp_path: Path) -> None:
    """processed JPEG 若在 EOI 后仍有字节，验证必须失败以阻止 Ultralytics 运行时改写。"""

    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    report_dir = tmp_path / "audit"
    (raw / "Annotations").mkdir(parents=True)
    (raw / "JPEGImages").mkdir()
    (raw / "ImageSets" / "Main").mkdir(parents=True)
    Image.new("RGB", (100, 200), "white").save(raw / "JPEGImages" / "a.jpg")
    (raw / "Annotations" / "a.xml").write_text(
        make_voc_xml(filename="a.jpg", objects=(("hat", 10, 20, 50, 80),)), encoding="utf-8"
    )
    for split, contents in {"train": "a\n", "val": "", "test": "", "trainval": "a\n"}.items():
        (raw / "ImageSets" / "Main" / f"{split}.txt").write_text(contents, encoding="utf-8")
    convert_dataset(raw, processed)
    processed_image = processed / "images" / "train" / "a.jpg"
    processed_image.write_bytes(processed_image.read_bytes() + b"\r\n")

    report = validate_dataset(raw, processed, report_dir)

    assert report["status"] == "failed"
    assert {issue["type"] for issue in report["issues"]} == {"noncanonical_jpeg"}
