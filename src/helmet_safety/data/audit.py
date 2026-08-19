"""对整套原始 SHWD/VOC 数据做只读审计，并生成 JSON、Markdown 报告。

本模块把 :mod:`voc` 的“单 XML 能力”组织成全数据集扫描：图片可读性、文件配对、
类别与框统计、官方划分关系，以及通过 SHA256 发现跨划分的相同图片。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable

from PIL import Image

from .voc import VocAnnotation, VocParseError, parse_voc_xml, validate_voc_annotation


# trainval 只是 train 和 val 的合集，不会被转换为单独的 YOLO 目录。
SPLIT_NAMES = ("train", "val", "test")
KNOWN_CLASSES = ("hat", "person", "dog")


def load_splits(raw_root: Path | str) -> dict[str, list[str]]:
    """读取官方划分文件，每一行是图片 ID（不带扩展名）。"""
    root = Path(raw_root)
    split_root = root / "ImageSets" / "Main"
    result: dict[str, list[str]] = {}
    for split in (*SPLIT_NAMES, "trainval"):
        path = split_root / f"{split}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"missing official split file: {path}")
        # utf-8-sig 同时兼容普通 UTF-8 和带 BOM 的文本；空行不算图片 ID。
        result[split] = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    return result


def check_split_integrity(splits: dict[str, Iterable[str]]) -> dict[str, object]:
    """检查三个 split 是否互斥，以及 trainval 是否恰好等于 train 与 val 的并集。"""
    # 转成 set 后，交集、并集和差集可以直接表达划分关系。
    # list 适合保留原文件顺序，set 更适合做交集/并集；这里不修改传入的 list。
    sets = {name: set(splits.get(name, ())) for name in (*SPLIT_NAMES, "trainval")}
    intersections = {
        # ``&`` 是集合交集：同时出现在两个 split 的 ID 会被列出来。
        "train_val": sorted(sets["train"] & sets["val"]),
        "train_test": sorted(sets["train"] & sets["test"]),
        "val_test": sorted(sets["val"] & sets["test"]),
    }
    # ``|`` 是集合并集，代表 trainval 理论上应该包含的全部 ID。
    expected_trainval = sets["train"] | sets["val"]
    return {
        "intersections": intersections,
        # 三个交集列表全为空时，any(...) 为 False，因此 mutually_exclusive 为 True。
        "mutually_exclusive": not any(intersections.values()),
        "trainval_matches_union": sets["trainval"] == expected_trainval,
        # 集合差集分别给出 trainval 少了什么、又多了什么。
        "trainval_missing": sorted(expected_trainval - sets["trainval"]),
        "trainval_extra": sorted(sets["trainval"] - expected_trainval),
        "duplicate_ids_within_files": {
            # Counter 统计同一 txt 内每个 ID 出现次数，次数 > 1 就是重复行。
            name: sorted(item for item, count in Counter(splits.get(name, ())).items() if count > 1)
            for name in (*SPLIT_NAMES, "trainval")
        },
    }


def _files_by_stem(directory: Path, suffixes: set[str]) -> tuple[dict[str, Path], list[str]]:
    """按不带扩展名的 stem 建索引，例如 000377.jpg -> 000377。"""
    found: dict[str, Path] = {}
    duplicate_stems: list[str] = []
    # suffix.lower() 让 .JPG、.jpg 等大小写写法都能识别。
    for path in sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in suffixes):
        if path.stem in found:
            # 例如同时出现 a.jpg 和 a.jpeg，它们都会映射到 stem=a，必须报告歧义。
            duplicate_stems.append(path.stem)
        else:
            found[path.stem] = path
    return found, sorted(set(duplicate_stems))


def _empty_class_counts() -> dict[str, int]:
    """创建一份互不共享的零值类别计数字典。"""

    return {"hat": 0, "person": 0, "dog": 0, "unknown": 0}


def _sha256(path: Path) -> str:
    """分块计算文件哈希，避免一次把大图片全部读进内存。"""
    digest = sha256()
    with path.open("rb") as stream:
        # 每次读取 1 MiB；iter 遇到 b""（文件结束）便停止。
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cross_split_duplicates(
    splits: dict[str, list[str]], images: dict[str, Path]
) -> list[dict[str, object]]:
    """用内容哈希查找文件名不同、但图片字节完全相同的跨 split 样本。"""
    # 结构形如：{哈希: {split: [图片ID, ...]}}。
    hashes: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for split in SPLIT_NAMES:
        for image_id in splits[split]:
            image_path = images.get(image_id)
            if image_path is not None:
                # 文件名可以不同；只要完整字节相同，SHA256 就相同。
                hashes[_sha256(image_path)][split].append(image_id)
    duplicates = []
    for digest, by_split in sorted(hashes.items()):
        # 同一哈希只出现在一个 split 不算“跨 split”重复。
        if len(by_split) > 1:
            duplicates.append(
                {
                    "sha256": digest,
                    "ids_by_split": {split: sorted(ids) for split, ids in sorted(by_split.items())},
                }
            )
    return duplicates


def _audit_markdown(report: dict[str, object]) -> str:
    """把机器易读的 report 字典整理成人更容易浏览的 Markdown。"""
    # 先取常用子字典，后面的 f-string 更短、更容易阅读。
    files = report["files"]
    parsing = report["xml_parsing"]
    integrity = report["split_integrity"]
    lines = [
        "# SHWD 原始数据审计报告",
        "",
        f"- 原始目录：`{report['raw_root']}`",
        f"- JPG：{files['jpg_count']}；XML：{files['xml_count']}",
        f"- XML 标准解析成功：{parsing['standard_success_count']}",
        f"- XML 内存修复后成功：{parsing['repaired_success_count']}",
        f"- XML 最终失败：{parsing['final_failure_count']}",
        f"- train/val/test 互斥：{integrity['mutually_exclusive']}",
        f"- trainval 等于 train∪val：{integrity['trainval_matches_union']}",
        "",
        "## 类别统计",
        "",
        "| 类别 | 框数 |",
        "|---|---:|",
    ]
    lines.extend(f"| {name} | {count} |" for name, count in report["classes"].items())
    lines.extend(["", "## 划分统计", "", "| split | 图片 | hat | person | dog | unknown |", "|---|---:|---:|---:|---:|---:|"])
    for split in SPLIT_NAMES:
        item = report["splits"][split]
        counts = item["class_boxes"]
        lines.append(
            f"| {split} | {item['image_count']} | {counts['hat']} | {counts['person']} | "
            f"{counts['dog']} | {counts['unknown']} |"
        )
    sections = [
        ("内存修复 XML", report["xml_parsing"]["repaired_files"]),
        ("最终无法解析 XML", report["xml_parsing"]["failed_files"]),
        ("非法尺寸/检测框", report["invalid_annotations"]),
        ("缺失文件", report["missing"]),
        ("未知类别", report["unknown_classes"]),
        ("跨划分重复图片", report["cross_split_duplicate_images"]),
        ("转换时应忽略的 dog 对象", report["ignored_dog_objects"]),
        ("folder/path 含替代字符的可疑元数据", report["metadata_with_replacement_characters"]),
    ]
    for title, value in sections:
        lines.extend(["", f"## {title}", "", "```json", json.dumps(value, ensure_ascii=False, indent=2), "```"])
    return "\n".join(lines) + "\n"


def audit_dataset(raw_root: Path | str, audit_dir: Path | str) -> dict[str, object]:
    """审计全部图片和 XML；只读取 raw_root，只向 audit_dir 写报告。"""
    # resolve 转成绝对路径，报告不依赖运行命令时的当前目录。
    root = Path(raw_root).resolve()
    destination = Path(audit_dir).resolve()
    annotation_dir = root / "Annotations"
    image_dir = root / "JPEGImages"
    if not annotation_dir.is_dir() or not image_dir.is_dir():
        raise FileNotFoundError(f"expected Annotations and JPEGImages under {root}")

    # 阶段 1：建立文件索引并检查官方划分文本本身。
    images, duplicate_image_stems = _files_by_stem(image_dir, {".jpg", ".jpeg"})
    xmls, duplicate_xml_stems = _files_by_stem(annotation_dir, {".xml"})
    splits = load_splits(root)
    split_integrity = check_split_integrity(splits)

    # 阶段 2：Pillow 真正解码每张图片，而不只是判断文件是否存在。
    unreadable_images: list[dict[str, str]] = []
    image_dimensions: dict[str, tuple[int, int]] = {}
    for image_id, path in images.items():
        try:
            with Image.open(path) as image:
                # Image.open 通常是惰性打开；load() 才会真正解码完整像素数据。
                image.load()
                image_dimensions[image_id] = image.size
        except Exception as exc:  # Pillow exposes several decoder-specific exception types.
            unreadable_images.append({"image": path.name, "reason": f"{type(exc).__name__}: {exc}"})

    # 阶段 3：解析全部 XML，并积累问题明细和全局类别数量。
    annotations: dict[str, VocAnnotation] = {}
    repaired_files: list[dict[str, str]] = []
    failed_files: list[dict[str, str]] = []
    invalid_annotations: list[dict[str, object]] = []
    unknown_classes: list[dict[str, object]] = []
    ignored_dogs: list[dict[str, object]] = []
    suspicious_metadata: list[dict[str, str]] = []
    class_counts = _empty_class_counts()

    for image_id, xml_path in xmls.items():
        try:
            annotation = parse_voc_xml(xml_path)
        except (VocParseError, OSError, UnicodeError) as exc:
            failed_files.append({"xml": xml_path.name, "reason": str(exc)})
            continue
        # annotations 用图片 ID 做键，稍后按 split ID 可 O(1) 找到对应标注。
        annotations[image_id] = annotation
        if annotation.parse_mode == "repaired_metadata":
            repaired_files.append({"xml": xml_path.name, "reason": annotation.repair_reason or ""})
        # U+FFFD 是 Unicode “替代字符”，通常说明文本曾经发生过编码损坏。
        if "\ufffd" in annotation.folder or "\ufffd" in annotation.image_path_metadata:
            suspicious_metadata.append(
                {"xml": xml_path.name, "folder": annotation.folder, "path": annotation.image_path_metadata}
            )
        # 先检查 XML 自身声明的尺寸和每个框的几何关系。
        problems = validate_voc_annotation(annotation)
        actual_size = image_dimensions.get(image_id)
        # 再把 XML 声明尺寸与真实解码图片尺寸比较。
        if actual_size and actual_size != (annotation.width, annotation.height):
            problems.append(
                f"XML size {annotation.width}x{annotation.height} does not match image {actual_size[0]}x{actual_size[1]}"
            )
        if problems:
            invalid_annotations.append({"xml": xml_path.name, "problems": problems})
        for object_index, obj in enumerate(annotation.objects, start=1):
            if obj.name in KNOWN_CLASSES:
                class_counts[obj.name] += 1
                if obj.name == "dog":
                    # dog 在审计中仍计数并定位；真正转换时才会跳过。
                    ignored_dogs.append({"xml": xml_path.name, "object_index": object_index, "class": "dog"})
            else:
                # 不把未知类别混入 hat/person/dog，单独累计并保存 XML 与对象序号。
                class_counts["unknown"] += 1
                unknown_classes.append(
                    {"xml": xml_path.name, "object_index": object_index, "class": obj.name}
                )

    # 阶段 4：按官方列表重新统计每个 split，而不是进行任何随机划分。
    split_reports: dict[str, object] = {}
    for split in SPLIT_NAMES:
        counts = _empty_class_counts()
        missing_image_ids: list[str] = []
        missing_annotation_ids: list[str] = []
        for image_id in splits[split]:
            if image_id not in images:
                missing_image_ids.append(image_id)
            annotation = annotations.get(image_id)
            if annotation is None:
                # None 既可能表示 XML 缺失，也可能表示 XML 存在但最终解析失败。
                missing_annotation_ids.append(image_id)
                continue
            for obj in annotation.objects:
                # 未知类统一落入 unknown，保证报告表格的列固定。
                counts[obj.name if obj.name in KNOWN_CLASSES else "unknown"] += 1
        split_reports[split] = {
            "image_count": len(splits[split]),
            "class_boxes": counts,
            "missing_image_ids": missing_image_ids,
            "missing_or_unparsed_annotation_ids": missing_annotation_ids,
        }

    # 阶段 5：汇总统一报告。JSON 供程序读取，Markdown 供人阅读。
    report: dict[str, object] = {
        "raw_root": root.as_posix(),
        "files": {
            "jpg_count": len(images),
            "xml_count": len(xmls),
            "split_file_counts": {name: len(ids) for name, ids in splits.items()},
            "duplicate_image_stems": duplicate_image_stems,
            "duplicate_xml_stems": duplicate_xml_stems,
        },
        "missing": {
            # 两个 set 差集分别检查图片无 XML、XML 无图片。
            "xml_for_images": sorted(set(images) - set(xmls)),
            "image_for_xml": sorted(set(xmls) - set(images)),
            "unreadable_images": unreadable_images,
        },
        "xml_parsing": {
            "standard_success_count": len(annotations) - len(repaired_files),
            "repaired_success_count": len(repaired_files),
            "final_failure_count": len(failed_files),
            "repaired_files": repaired_files,
            "failed_files": failed_files,
        },
        "classes": class_counts,
        "splits": split_reports,
        "split_integrity": split_integrity,
        "invalid_annotations": invalid_annotations,
        "unknown_classes": unknown_classes,
        # SHA256 是全量图片扫描中较耗时的一步，最后统一执行并写入明细。
        "cross_split_duplicate_images": _cross_split_duplicates(splits, images),
        "ignored_dog_objects": ignored_dogs,
        "metadata_with_replacement_characters": suspicious_metadata,
    }
    # 这里是整个审计函数唯一的写入区域，目标位于 audit_dir 而非 raw_root。
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (destination / "audit_report.md").write_text(_audit_markdown(report), encoding="utf-8")
    return report
