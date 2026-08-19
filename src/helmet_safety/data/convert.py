"""把官方 SHWD Pascal VOC 数据安全转换为 Ultralytics YOLO 目录格式。

转换分为两个阶段：先在内存中预检查全部样本，再集中写文件。这样遇到未知类别、
缺失文件或坏框时，会在复制开始前失败，尽量避免生成看似可用的半套数据。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import shutil

from PIL import Image

from .audit import SPLIT_NAMES, check_split_integrity, load_splits
from .voc import VocAnnotation, parse_voc_xml, validate_voc_annotation, voc_box_to_yolo


# 这里同时保留“原始类别”和“训练时显示名称”两层映射。
# YOLO 标签文件只存数字 0/1；dataset.yaml 再把数字解释为可读名称。
CLASS_MAPPING = {"hat": 0, "person": 1}
YOLO_NAMES = {0: "helmet", 1: "no_helmet"}
IGNORED_CLASS = "dog"
MANIFEST_NAME = ".shwd-generated-files.json"


class UnknownClassError(ValueError):
    """Raised rather than silently losing an unexpected class."""


class ConversionError(RuntimeError):
    """Raised when preflight prevents a safe complete conversion."""


@dataclass(frozen=True)
class AnnotationConversion:
    """单个 XML 转换后的内存结果，此时还没有写入 txt 文件。"""

    lines: tuple[str, ...]
    class_counts: dict[str, int]
    ignored_dog_count: int
    ignored_objects: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class _PreparedItem:
    """预检查通过、等待统一写入 processed 的一个样本。"""

    split: str
    image_id: str
    image_path: Path
    label_text: str
    box_count: int
    class_counts: dict[str, int]
    ignored_objects: tuple[dict[str, object], ...]
    jpeg_normalization: dict[str, object] | None


def _inspect_jpeg(image_path: Path) -> dict[str, object] | None:
    """严格解码 JPEG，并为 EOI 后存在尾随字节的文件生成无损规范化计划。"""

    if image_path.suffix.lower() not in {".jpg", ".jpeg"}:
        return None
    data = image_path.read_bytes()
    try:
        with Image.open(BytesIO(data)) as image:
            image.load()
            original_signature = (image.mode, image.size, sha256(image.tobytes()).hexdigest())
    except Exception as exc:
        raise ConversionError(f"JPEG cannot be decoded strictly: {image_path.name}: {exc}") from exc
    if data.endswith(b"\xff\xd9"):
        return None
    eoi_index = data.rfind(b"\xff\xd9")
    if eoi_index < 0:
        raise ConversionError(f"JPEG has no EOI marker: {image_path.name}")
    normalized = data[: eoi_index + 2]
    try:
        with Image.open(BytesIO(normalized)) as image:
            image.load()
            normalized_signature = (image.mode, image.size, sha256(image.tobytes()).hexdigest())
    except Exception as exc:
        raise ConversionError(f"JPEG normalization is not decodable: {image_path.name}: {exc}") from exc
    if normalized_signature != original_signature:
        raise ConversionError(f"JPEG normalization changes decoded pixels: {image_path.name}")
    return {
        "method": "trim_bytes_after_eoi",
        "bytes_removed": len(data) - len(normalized),
        "raw_sha256": sha256(data).hexdigest(),
        "processed_sha256": sha256(normalized).hexdigest(),
        "normalized_size": len(normalized),
    }


def convert_annotation(
    annotation: VocAnnotation,
    *,
    xml_path: Path | str,
    precision: int = 6,
) -> AnnotationConversion:
    """转换 hat/person，记录并跳过 dog，遇到其他类别立即报错。"""
    annotation_xml_path = Path(xml_path)
    # lines 保存最终 txt 的每一行；现在只在内存里构造，不会立即写文件。
    lines: list[str] = []
    class_counts = {"hat": 0, "person": 0}
    ignored: list[dict[str, object]] = []
    for object_index, obj in enumerate(annotation.objects, start=1):
        # continue 只跳过当前 dog，不会跳过同一 XML 后面的有效对象。
        if obj.name == IGNORED_CLASS:
            # 保存 XML 文件名和对象序号，之后可以准确追溯被跳过的是哪个框。
            ignored.append({"xml": annotation_xml_path.name, "object_index": object_index, "class": obj.name})
            continue
        if obj.name not in CLASS_MAPPING:
            # 不能把未知类别静默丢掉，否则转换后的框数量看似合理却已经缺数据。
            raise UnknownClassError(
                f"{annotation_xml_path.name}: unknown class {obj.name!r} at object {object_index}; conversion aborted"
            )
        # values 的顺序固定为 x_center, y_center, width, height。
        values = voc_box_to_yolo(
            obj.xmin,
            obj.ymin,
            obj.xmax,
            obj.ymax,
            image_width=annotation.width,
            image_height=annotation.height,
        )
        # 每行严格是 5 列；固定小数位让重复运行得到相同文本。
        line = " ".join([str(CLASS_MAPPING[obj.name]), *(f"{value:.{precision}f}" for value in values)])
        lines.append(line)
        # 此处只统计真正写入 YOLO 的 hat/person，dog 不进入有效框计数。
        class_counts[obj.name] += 1
    return AnnotationConversion(
        lines=tuple(lines),
        class_counts=class_counts,
        ignored_dog_count=len(ignored),
        ignored_objects=tuple(ignored),
    )


def _index_files(directory: Path, suffixes: set[str]) -> dict[str, Path]:
    """以图片 ID 建索引，并拒绝同 stem 的重复源文件。"""
    result: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in suffixes:
            if path.stem in result:
                raise ConversionError(f"duplicate source stem {path.stem!r} in {directory}")
            result[path.stem] = path
    return result


def _is_nonempty(path: Path) -> bool:
    """判断路径是否存在且至少包含一个直接子项。"""

    # 不存在或存在但没有任何子项，都视为可安全首次写入。
    return path.exists() and any(path.iterdir())


def _safe_generated_path(root: Path, relative: str) -> Path:
    """确认 manifest 记录的文件仍在 output 根目录内，阻止 ``../`` 越界。"""
    # resolve 会消解 ``..``。随后 relative_to 用于确认最终路径仍在 root 下。
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ConversionError(f"unsafe generated path in manifest: {relative!r}") from exc
    return candidate


def _prepare_output(output_root: Path, force: bool) -> None:
    """准备输出目录；force 也只能删除 manifest 证明由本工具生成的文件。"""
    if not _is_nonempty(output_root):
        # parents=True 会连同缺失的上级目录一起创建；exist_ok 允许空目录已存在。
        output_root.mkdir(parents=True, exist_ok=True)
        return
    if not force:
        # 默认宁可停止，也不猜测非空目录中的文件属于谁。
        raise FileExistsError(f"processed output is not empty: {output_root}; pass --force for a safe rerun")
    manifest_path = output_root / MANIFEST_NAME
    if not manifest_path.is_file():
        # 即使传入 --force，没有清单也无法证明哪些文件是本工具生成的，因此仍拒绝。
        raise FileExistsError(
            f"cannot safely use --force because {manifest_path} is missing; existing files were not proven tool-generated"
        )
    # manifest 是上次成功转换留下的“所有权清单”。不在清单里的用户文件不删除。
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative in manifest.get("generated_files", []):
        path = _safe_generated_path(output_root, relative)
        if path.is_file():
            # 只删除清单中的普通文件；不对 output_root 使用递归删除。
            path.unlink()
    # 清单自身也是工具文件，在清单内文件处理完后单独删除。
    manifest_path.unlink()
    directories = sorted(
        (path for path in output_root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        try:
            # rmdir 只能删除空目录；只要里面有未知文件就会失败并进入 except，被安全保留。
            directory.rmdir()
        except OSError:
            pass


def _dataset_yaml(output_root: Path) -> str:
    """生成 Ultralytics 读取数据集所需的固定 YAML 内容。"""
    return (
        f"path: {output_root.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\n"
        "names:\n"
        "  0: helmet\n"
        "  1: no_helmet\n"
    )


def convert_dataset(
    raw_root: Path | str,
    output_root: Path | str,
    *,
    force: bool = False,
    precision: int = 6,
) -> dict[str, object]:
    """按官方 split 完整预检查，然后复制图片、写标签和转换报告。"""
    root = Path(raw_root).resolve()
    output = Path(output_root).resolve()
    # 尽早检查输出，避免花时间解析 7581 个 XML 后才告诉用户目录非空。
    if _is_nonempty(output) and not force:
        raise FileExistsError(f"processed output is not empty: {output}; pass --force for a safe rerun")

    # ---------- 阶段 1：只读预检查，不向 processed 写任何内容 ----------
    splits = load_splits(root)
    integrity = check_split_integrity(splits)
    if not integrity["mutually_exclusive"] or not integrity["trainval_matches_union"]:
        # 官方划分有交集或 trainval 不一致时，继续转换会把问题固化到 processed。
        raise ConversionError(f"official splits failed integrity checks: {integrity}")
    images = _index_files(root / "JPEGImages", {".jpg", ".jpeg"})
    xmls = _index_files(root / "Annotations", {".xml"})

    # label 文本先保存在 prepared 中。只有 failures 为空，才会进入写入阶段。
    prepared: list[_PreparedItem] = []
    failures: list[dict[str, str]] = []
    for split in SPLIT_NAMES:
        for image_id in splits[split]:
            # 官方 txt 只给 ID，因此分别在图片/XML 索引中查询真实文件路径。
            image_path = images.get(image_id)
            xml_path = xmls.get(image_id)
            if image_path is None or xml_path is None:
                # 先收集所有失败项，而不是遇到第一个问题就退出，方便一次修完。
                failures.append(
                    {
                        "split": split,
                        "id": image_id,
                        "reason": "missing image" if image_path is None else "missing XML",
                    }
                )
                continue
            try:
                jpeg_normalization = _inspect_jpeg(image_path)
                annotation = parse_voc_xml(xml_path)
                problems = validate_voc_annotation(annotation)
                if problems:
                    # 多个坏框问题用分号合并，保留在同一个文件的失败原因中。
                    raise ConversionError("; ".join(problems))
                converted = convert_annotation(annotation, xml_path=xml_path, precision=precision)
            except Exception as exc:
                failures.append(
                    {"split": split, "id": image_id, "file": xml_path.name, "reason": f"{type(exc).__name__}: {exc}"}
                )
                continue
            # YOLO 文件要求一行一个框。空 objects 会得到空字符串和空标签文件。
            label_text = "\n".join(converted.lines)
            if label_text:
                # 非空文件末尾统一保留换行，便于命令行工具和版本控制阅读。
                label_text += "\n"
            prepared.append(
                _PreparedItem(
                    split=split,
                    image_id=image_id,
                    image_path=image_path,
                    label_text=label_text,
                    box_count=len(converted.lines),
                    class_counts=converted.class_counts,
                    ignored_objects=converted.ignored_objects,
                    jpeg_normalization=jpeg_normalization,
                )
            )
    if failures:
        # 到这里仍未调用 _prepare_output，因此失败不会创建 processed 半成品。
        unknown_failures = [item for item in failures if "UnknownClassError" in item["reason"]]
        if unknown_failures:
            raise UnknownClassError(json.dumps(unknown_failures, ensure_ascii=False))
        raise ConversionError(f"conversion preflight failed: {json.dumps(failures, ensure_ascii=False)}")

    # ---------- 阶段 2：预检查全部通过后，才开始创建输出 ----------
    _prepare_output(output, force)
    report: dict[str, object] = {
        "raw_root": root.as_posix(),
        "output_root": output.as_posix(),
        "class_mapping": {"hat": "0 helmet", "person": "1 no_helmet"},
        "precision": precision,
        "splits": {},
        "ignored_dog_objects": [],
        "jpeg_normalization": {"count": 0, "items": []},
        "failed_files": [],
    }
    generated: list[str] = []
    for split in SPLIT_NAMES:
        # 每个 split 都有一套独立 images/labels 子目录。
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
        # 从完整预检查结果中筛选当前 split 的样本，官方顺序仍由 prepared 保留。
        items = [item for item in prepared if item.split == split]
        empty_labels: list[str] = []
        split_classes: Counter[str] = Counter()
        split_ignored: list[dict[str, object]] = []
        for item in items:
            image_target = output / "images" / split / item.image_path.name
            label_target = output / "labels" / split / f"{item.image_id}.txt"
            if image_target.exists() or label_target.exists():
                # --force 清理后仍存在同名文件，意味着它不在旧 manifest 中，不能覆盖。
                raise FileExistsError(f"refusing to overwrite untracked output file: {image_target} or {label_target}")
            # copy2 是独立复制，不是硬链接；修改 processed 图片不会影响 raw 图片。
            shutil.copy2(item.image_path, image_target)
            if item.jpeg_normalization is not None:
                # 只截掉 JPEG 主图 EOI 后的无关尾随字节，不进行有损重编码。
                with image_target.open("r+b") as handle:
                    handle.truncate(int(item.jpeg_normalization["normalized_size"]))
                report["jpeg_normalization"]["items"].append(
                    {
                        "split": split,
                        "image": image_target.name,
                        "method": item.jpeg_normalization["method"],
                        "bytes_removed": item.jpeg_normalization["bytes_removed"],
                        "raw_sha256": item.jpeg_normalization["raw_sha256"],
                        "processed_sha256": item.jpeg_normalization["processed_sha256"],
                    }
                )
            # newline="\n" 保证 Windows 上也输出一致的 LF，便于跨平台复现。
            label_target.write_text(item.label_text, encoding="utf-8", newline="\n")
            # manifest 使用相对路径，使整个 processed 目录移动后清单仍然有效。
            generated.extend(
                [image_target.relative_to(output).as_posix(), label_target.relative_to(output).as_posix()]
            )
            split_classes.update(item.class_counts)
            split_ignored.extend(item.ignored_objects)
            if not item.label_text:
                # 空标签不是失败，但必须记录，用户能知道哪些图片没有有效目标。
                empty_labels.append(label_target.name)
        report["splits"][split] = {
            "image_count": len(items),
            "label_count": len(items),
            "box_count": sum(item.box_count for item in items),
            "class_counts": {"hat": split_classes["hat"], "person": split_classes["person"]},
            "skipped_object_count": len(split_ignored),
            "empty_label_files": empty_labels,
            "failed_files": [],
            "normalized_jpeg_count": sum(item.jpeg_normalization is not None for item in items),
        }
        report["ignored_dog_objects"].extend(split_ignored)

    report["jpeg_normalization"]["count"] = len(report["jpeg_normalization"]["items"])

    # 所有图片和标签写完后，再生成 Ultralytics 的入口配置和报告。
    yaml_path = output / "dataset.yaml"
    yaml_path.write_text(_dataset_yaml(output), encoding="utf-8", newline="\n")
    generated.append(yaml_path.relative_to(output).as_posix())
    conversion_report_path = output / "conversion_report.json"
    conversion_report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    generated.append(conversion_report_path.relative_to(output).as_posix())
    # manifest 最后写：只有完整转换走到这里，未来 --force 才获得安全清理依据。
    manifest = {
        "tool": "helmet_safety.data.convert",
        "generated_files": sorted(generated),
    }
    (output / MANIFEST_NAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
