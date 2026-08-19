from __future__ import annotations

from pathlib import Path

import pytest

from conftest import make_voc_xml
from helmet_safety.data.voc import VocParseError, parse_voc_xml, validate_voc_annotation


def test_normal_voc_xml_parses_all_detection_fields(tmp_path: Path) -> None:
    """正常 XML 应解析文件元数据、图片尺寸和全部目标框字段。"""

    xml_path = tmp_path / "sample.xml"
    xml_path.write_text(
        make_voc_xml(objects=(("hat", 10, 20, 50, 80), ("person", 1, 2, 99, 199))),
        encoding="utf-8",
    )

    annotation = parse_voc_xml(xml_path)

    assert annotation.xml_path == xml_path
    assert annotation.filename == "sample.jpg"
    assert (annotation.width, annotation.height, annotation.depth) == (100, 200, 3)
    assert [(obj.name, obj.xmin, obj.ymin, obj.xmax, obj.ymax) for obj in annotation.objects] == [
        ("hat", 10.0, 20.0, 50.0, 80.0),
        ("person", 1.0, 2.0, 99.0, 199.0),
    ]
    assert annotation.parse_mode == "standard"
    assert annotation.repair_reason is None
    assert annotation.folder == "JPEGImages"
    assert annotation.image_path_metadata == "C:/dataset/sample.jpg"


def test_only_broken_folder_and_path_metadata_are_repaired_in_memory(tmp_path: Path) -> None:
    """folder/path 损坏时只能在内存中修复，检测对象和原 XML 字节必须保持不变。"""

    xml_path = tmp_path / "broken_metadata.xml"
    xml_path.write_text(
        make_voc_xml(
            folder="bad & folder",
            path="C:/损坏 & path/sample.jpg",
            objects=(("hat", 10, 20, 50, 80),),
        ),
        encoding="utf-8",
    )
    original = xml_path.read_bytes()

    annotation = parse_voc_xml(xml_path)

    assert annotation.parse_mode == "repaired_metadata"
    assert "folder/path" in annotation.repair_reason
    assert [(obj.name, obj.xmin, obj.ymin, obj.xmax, obj.ymax) for obj in annotation.objects] == [
        ("hat", 10.0, 20.0, 50.0, 80.0)
    ]
    assert xml_path.read_bytes() == original


def test_corruption_outside_folder_and_path_is_not_globally_rewritten(tmp_path: Path) -> None:
    """检测对象区域损坏时必须失败，不能用扩大范围的修复掩盖问题。"""

    xml_path = tmp_path / "broken_object.xml"
    xml_path.write_text(
        make_voc_xml(objects=(("hat & person", 10, 20, 50, 80),)), encoding="utf-8"
    )

    with pytest.raises(VocParseError, match="after folder/path metadata repair"):
        parse_voc_xml(xml_path)


@pytest.mark.parametrize(
    ("box", "reason"),
    [
        ((50, 20, 10, 80), "xmin must be less than xmax"),
        ((10, 80, 50, 20), "ymin must be less than ymax"),
        ((-1, 20, 50, 80), "out of bounds"),
        ((10, 20, 101, 80), "out of bounds"),
    ],
)
def test_illegal_boxes_are_reported(tmp_path: Path, box: tuple[int, int, int, int], reason: str) -> None:
    """反向或越界的 VOC 框应返回对应的可读问题原因。"""

    xml_path = tmp_path / "bad_box.xml"
    xml_path.write_text(make_voc_xml(objects=(("hat", *box),)), encoding="utf-8")

    problems = validate_voc_annotation(parse_voc_xml(xml_path))

    assert len(problems) == 1
    assert reason in problems[0]
