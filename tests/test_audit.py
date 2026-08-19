from __future__ import annotations

from pathlib import Path

from PIL import Image

from conftest import make_voc_xml
from helmet_safety.data.audit import audit_dataset


def test_audit_reports_pairing_classes_splits_and_cross_split_duplicate_content(tmp_path: Path) -> None:
    """审计应同时报告文件配对、类别统计、划分统计和跨 split 内容重复。"""

    raw = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    (raw / "Annotations").mkdir(parents=True)
    (raw / "JPEGImages").mkdir()
    (raw / "ImageSets" / "Main").mkdir(parents=True)
    for image_id in ("a", "b", "c", "image_only"):
        Image.new("RGB", (100, 200), "white" if image_id in {"a", "c"} else "black").save(
            raw / "JPEGImages" / f"{image_id}.jpg"
        )
    xml_objects = {
        "a": (("hat", 10, 20, 50, 80), ("dog", 1, 2, 10, 20)),
        "b": (("person", 20, 40, 60, 100),),
        "c": (("hat", 10, 20, 50, 80),),
        "xml_only": (),
    }
    for image_id, objects in xml_objects.items():
        (raw / "Annotations" / f"{image_id}.xml").write_text(
            make_voc_xml(filename=f"{image_id}.jpg", objects=objects), encoding="utf-8"
        )
    for split, ids in {"train": ["a"], "val": ["b"], "test": ["c"], "trainval": ["a", "b"]}.items():
        (raw / "ImageSets" / "Main" / f"{split}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")

    report = audit_dataset(raw, audit_dir)

    assert report["files"]["jpg_count"] == 4
    assert report["files"]["xml_count"] == 4
    assert report["missing"]["xml_for_images"] == ["image_only"]
    assert report["missing"]["image_for_xml"] == ["xml_only"]
    assert report["classes"] == {"hat": 2, "person": 1, "dog": 1, "unknown": 0}
    assert report["splits"]["train"]["class_boxes"] == {"hat": 1, "person": 0, "dog": 1, "unknown": 0}
    assert any(item["ids_by_split"] == {"test": ["c"], "train": ["a"]} for item in report["cross_split_duplicate_images"])
    assert report["ignored_dog_objects"][0]["xml"] == "a.xml"
    assert (audit_dir / "audit_report.json").is_file()
    assert (audit_dir / "audit_report.md").is_file()
