from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest
from PIL import Image

from conftest import make_voc_xml
from helmet_safety.data.convert import ConversionError, UnknownClassError, convert_annotation, convert_dataset
from helmet_safety.data.voc import parse_voc_xml, voc_box_to_yolo


def test_hat_and_person_map_to_expected_yolo_classes(tmp_path: Path) -> None:
    """hat/person 应分别写成 YOLO 类别 0/1，并保留各自数量。"""

    xml_path = tmp_path / "classes.xml"
    xml_path.write_text(
        make_voc_xml(objects=(("hat", 10, 20, 50, 80), ("person", 20, 40, 60, 100))),
        encoding="utf-8",
    )

    result = convert_annotation(parse_voc_xml(xml_path), xml_path=xml_path)

    assert [line.split()[0] for line in result.lines] == ["0", "1"]
    assert result.class_counts == {"hat": 1, "person": 1}


def test_dog_is_ignored_without_dropping_other_objects(tmp_path: Path) -> None:
    """dog 应单独跳过并计数，同一 XML 中的 hat/person 仍应正常转换。"""

    xml_path = tmp_path / "dog.xml"
    xml_path.write_text(
        make_voc_xml(objects=(("hat", 10, 20, 50, 80), ("dog", 1, 2, 10, 20), ("person", 20, 40, 60, 100))),
        encoding="utf-8",
    )

    result = convert_annotation(parse_voc_xml(xml_path), xml_path=xml_path)

    assert len(result.lines) == 2
    assert result.ignored_dog_count == 1
    assert result.class_counts == {"hat": 1, "person": 1}


def test_unknown_class_fails_with_xml_path_and_class_name(tmp_path: Path) -> None:
    """未知类别必须抛出包含 XML 文件名和类别名的明确错误。"""

    xml_path = tmp_path / "unknown.xml"
    xml_path.write_text(make_voc_xml(objects=(("bicycle", 10, 20, 50, 80),)), encoding="utf-8")

    with pytest.raises(UnknownClassError, match=r"unknown\.xml.*bicycle"):
        convert_annotation(parse_voc_xml(xml_path), xml_path=xml_path)


def test_voc_box_to_yolo_math_is_hand_checked() -> None:
    """VOC 到 YOLO 的中心点和宽高公式应与人工计算值一致。"""

    assert voc_box_to_yolo(10, 20, 50, 80, image_width=100, image_height=200) == pytest.approx(
        (0.3, 0.25, 0.4, 0.3)
    )


def test_empty_annotation_creates_empty_label_file(tmp_path: Path) -> None:
    """没有有效对象的图片仍应保留，并生成可追溯的空标签文件。"""

    raw = tmp_path / "raw"
    (raw / "Annotations").mkdir(parents=True)
    (raw / "JPEGImages").mkdir()
    (raw / "ImageSets" / "Main").mkdir(parents=True)
    Image.new("RGB", (100, 200), "white").save(raw / "JPEGImages" / "empty.jpg")
    (raw / "Annotations" / "empty.xml").write_text(make_voc_xml(filename="empty.jpg"), encoding="utf-8")
    for split, contents in {"train": "empty\n", "val": "", "test": "", "trainval": "empty\n"}.items():
        (raw / "ImageSets" / "Main" / f"{split}.txt").write_text(contents, encoding="utf-8")
    output = tmp_path / "processed"

    report = convert_dataset(raw, output)

    assert (output / "labels" / "train" / "empty.txt").read_text(encoding="utf-8") == ""
    assert report["splits"]["train"]["empty_label_files"] == ["empty.txt"]


def test_conversion_refuses_nonempty_output_without_force(tmp_path: Path) -> None:
    """未传入 force 时，非空输出目录必须被拒绝且其中原文件保持不变。"""

    raw = tmp_path / "raw"
    output = tmp_path / "processed"
    output.mkdir()
    (output / "unrelated.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--force"):
        convert_dataset(raw, output)

    assert (output / "unrelated.txt").read_text(encoding="utf-8") == "keep"


def test_conversion_losslessly_removes_bytes_after_jpeg_eoi_without_modifying_raw(tmp_path: Path) -> None:
    """转换应只清理 processed 副本的 EOI 尾随字节，并保持 raw 与解码像素不变。"""

    raw = tmp_path / "raw"
    output = tmp_path / "processed"
    (raw / "Annotations").mkdir(parents=True)
    (raw / "JPEGImages").mkdir()
    (raw / "ImageSets" / "Main").mkdir(parents=True)
    image_path = raw / "JPEGImages" / "tailed.jpg"
    Image.new("RGB", (100, 200), "white").save(image_path)
    canonical_bytes = image_path.read_bytes()
    assert canonical_bytes.endswith(b"\xff\xd9")
    image_path.write_bytes(canonical_bytes + b"\r\n")
    raw_bytes = image_path.read_bytes()
    raw_hash = sha256(raw_bytes).hexdigest()
    (raw / "Annotations" / "tailed.xml").write_text(
        make_voc_xml(filename="tailed.jpg", objects=(("hat", 10, 20, 50, 80),)), encoding="utf-8"
    )
    for split, contents in {"train": "tailed\n", "val": "", "test": "", "trainval": "tailed\n"}.items():
        (raw / "ImageSets" / "Main" / f"{split}.txt").write_text(contents, encoding="utf-8")

    report = convert_dataset(raw, output)

    processed_image = output / "images" / "train" / "tailed.jpg"
    assert image_path.read_bytes() == raw_bytes
    assert sha256(image_path.read_bytes()).hexdigest() == raw_hash
    assert processed_image.read_bytes() == canonical_bytes
    assert report["jpeg_normalization"]["count"] == 1
    assert report["jpeg_normalization"]["items"] == [
        {
            "split": "train",
            "image": "tailed.jpg",
            "method": "trim_bytes_after_eoi",
            "bytes_removed": 2,
            "raw_sha256": raw_hash,
            "processed_sha256": sha256(canonical_bytes).hexdigest(),
        }
    ]


def test_conversion_rejects_undecodable_jpeg_before_creating_processed_output(tmp_path: Path) -> None:
    """无法严格解码且没有合法 EOI 的 JPEG 必须在写出 processed 前令转换失败。"""

    raw = tmp_path / "raw"
    output = tmp_path / "processed"
    (raw / "Annotations").mkdir(parents=True)
    (raw / "JPEGImages").mkdir()
    (raw / "ImageSets" / "Main").mkdir(parents=True)
    (raw / "JPEGImages" / "broken.jpg").write_bytes(b"\xff\xd8not-a-complete-jpeg")
    (raw / "Annotations" / "broken.xml").write_text(
        make_voc_xml(filename="broken.jpg", objects=(("hat", 10, 20, 50, 80),)), encoding="utf-8"
    )
    for split, contents in {"train": "broken\n", "val": "", "test": "", "trainval": "broken\n"}.items():
        (raw / "ImageSets" / "Main" / f"{split}.txt").write_text(contents, encoding="utf-8")

    with pytest.raises(ConversionError, match="JPEG"):
        convert_dataset(raw, output)

    assert not output.exists()
