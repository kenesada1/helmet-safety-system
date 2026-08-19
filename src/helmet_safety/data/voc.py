"""读取单个 Pascal VOC XML，并提供检测框校验、坐标转换等基础能力。

这个模块只处理“一张图片对应的一个 XML”，不关心 train/val/test，也不写输出数据。
如果 XML 因 ``folder`` 或 ``path`` 元数据损坏而无法解析，恢复逻辑只会在内存里
替换这两个节点的文字；真正影响训练的 ``object``、``size``、``bndbox`` 不会被修改。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET


class VocParseError(ValueError):
    """Raised when a VOC annotation cannot be parsed safely."""


@dataclass(frozen=True)
class VocObject:
    """一个 Pascal VOC ``<object>``，坐标仍是原始像素坐标。"""

    name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass(frozen=True)
class VocAnnotation:
    """一份 XML 解析后的结构化结果。

    ``frozen=True`` 表示对象创建后不能被意外改写，有助于保证审计和转换使用的是
    同一份原始解析结果。``parse_mode`` 用来区分标准解析和内存修复解析。
    """

    # 当前标注 XML 在文件系统中的真实位置，不对应 XML 的 <source> 节点。
    xml_path: Path
    filename: str
    width: int
    height: int
    depth: int
    objects: tuple[VocObject, ...]
    parse_mode: str = "standard"
    repair_reason: str | None = None
    folder: str = ""
    # XML <path> 中保存的历史图片路径，仅作元数据审计，不用于寻找当前图片。
    image_path_metadata: str = ""


def _replace_element_contents(data: bytes, tag: bytes) -> tuple[bytes, bool]:
    """只替换指定 XML 节点的内容，不使用可能误伤检测节点的全局正则。"""
    # result 一开始和原始文件内容相同。后续每次替换都会生成新的 bytes，
    # 不会通过 Path.write_* 写回硬盘。
    result = data
    changed = False  # 告诉调用者：是否真的找到了这个标签并替换过。
    search_from = 0  # 下一轮从哪里继续搜索，避免重复命中同一个标签。
    opening = b"<" + tag
    closing = b"</" + tag + b">"
    # 直接在 bytes 上查找标签边界，因此即使元数据含非法 UTF-8 字节也能处理。
    while True:
        # 例如 tag=b"path" 时，先寻找 <path 的起始位置。
        start = result.find(opening, search_from)
        if start < 0:
            break
        name_end = start + len(opening)
        # 防止把 <pathname> 误认为 <path>：标签名后面只能是 > 或空白。
        if name_end >= len(result) or result[name_end : name_end + 1] not in (b">", b" ", b"\t", b"\r", b"\n"):
            search_from = name_end
            continue
        # opening 可能含属性，所以再寻找真正结束开始标签的 >。
        content_start = result.find(b">", name_end)
        if content_start < 0:
            break
        content_start += 1
        # content_start 到 content_end 之间才是需要舍弃的元数据文本。
        content_end = result.find(closing, content_start)
        if content_end < 0:
            break
        result = result[:content_start] + b"REPAIRED_METADATA" + result[content_end:]
        changed = True
        # 跳过刚写入的占位符和结束标签，再寻找可能出现的下一个同名节点。
        search_from = content_start + len(b"REPAIRED_METADATA") + len(closing)
    return result, changed


def _parse_required_int(root: ET.Element, xpath: str, xml_path: Path) -> int:
    """读取必需的 XML 节点并转成整数，缺失或格式错误时给出文件级定位信息。"""

    # findtext 返回节点文字；找不到节点时返回 None。
    text = root.findtext(xpath)
    if text is None:
        raise VocParseError(f"{xml_path.name}: missing required node {xpath}")
    try:
        # strip 去掉 XML 排版产生的换行、空格，再转成整数。
        return int(text.strip())
    except ValueError as exc:
        raise VocParseError(f"{xml_path.name}: {xpath} is not an integer: {text!r}") from exc


def _parse_required_float(node: ET.Element, tag: str, xml_path: Path, object_index: int) -> float:
    """读取一个检测框坐标并转成浮点数，错误信息包含 XML 和对象序号。"""

    text = node.findtext(tag)
    if text is None:
        raise VocParseError(f"{xml_path.name}: object {object_index} missing bndbox/{tag}")
    try:
        # 框坐标通常是整数，但使用 float 也能兼容带小数的 VOC 标注工具。
        return float(text.strip())
    except ValueError as exc:
        raise VocParseError(f"{xml_path.name}: object {object_index} bndbox/{tag} is not numeric: {text!r}") from exc


def _annotation_from_root(root: ET.Element, xml_path: Path, parse_mode: str, repair_reason: str | None) -> VocAnnotation:
    """把 ElementTree 节点转换为本项目使用的不可变数据类。"""
    if root.tag != "annotation":
        raise VocParseError(f"{xml_path.name}: root element must be annotation, got {root.tag!r}")
    # size 是后续边界检查和归一化计算的分母，所以三个值都必须存在。
    width = _parse_required_int(root, "size/width", xml_path)
    height = _parse_required_int(root, "size/height", xml_path)
    depth = _parse_required_int(root, "size/depth", xml_path)
    objects: list[VocObject] = []
    # XML 中每个 <object> 都独立解析；start=1 让错误信息更符合人类计数习惯。
    for index, node in enumerate(root.findall("object"), start=1):
        # ``or ""`` 把缺失 name 的 None 变成空字符串，随后统一报清晰错误。
        name = (node.findtext("name") or "").strip()
        if not name:
            raise VocParseError(f"{xml_path.name}: object {index} has no class name")
        box = node.find("bndbox")
        if box is None:
            raise VocParseError(f"{xml_path.name}: object {index} missing bndbox")
        # 到这里说明 name 和 bndbox 都存在，将 XML 文本转成结构化对象。
        objects.append(
            VocObject(
                name=name,
                xmin=_parse_required_float(box, "xmin", xml_path, index),
                ymin=_parse_required_float(box, "ymin", xml_path, index),
                xmax=_parse_required_float(box, "xmax", xml_path, index),
                ymax=_parse_required_float(box, "ymax", xml_path, index),
            )
        )
    # tuple(objects) 让目标序列不可变，同时保留 XML 中原本的 object 顺序。
    return VocAnnotation(
        xml_path=xml_path,
        filename=(root.findtext("filename") or f"{xml_path.stem}.jpg").strip(),
        width=width,
        height=height,
        depth=depth,
        objects=tuple(objects),
        parse_mode=parse_mode,
        repair_reason=repair_reason,
        folder=root.findtext("folder") or "",
        image_path_metadata=root.findtext("path") or "",
    )


def parse_voc_xml(path: Path | str) -> VocAnnotation:
    """解析一个 VOC XML；标准解析失败时，仅在内存中修复无关元数据后重试。"""
    xml_path = Path(path)
    # 用 read_bytes 而不是先按 UTF-8 解码，XML 解析器可以自行处理声明的编码，
    # 元数据里即使存在非法字节，后面的定点修复也仍然有机会工作。
    data = xml_path.read_bytes()
    # 第一条路径：绝大多数文件直接按原始字节标准解析。
    try:
        # fromstring 只在内存中构建 XML 树，不会改写源文件。
        root = ET.fromstring(data)
        return _annotation_from_root(root, xml_path, "standard", None)
    except ET.ParseError as standard_error:
        # 第二条路径：只清空 folder/path 的文本，再尝试一次。
        # repaired 是新 bytes，xml_path 指向的原始 XML 从未被写入。
        repaired = data
        changed_tags: list[str] = []
        for tag in (b"folder", b"path"):
            repaired, changed = _replace_element_contents(repaired, tag)
            if changed:
                # 记录到底替换了哪些节点，最终写进审计报告的修复原因。
                changed_tags.append(tag.decode("ascii"))
        try:
            root = ET.fromstring(repaired)
        except ET.ParseError as repaired_error:
            raise VocParseError(
                f"{xml_path.name}: XML parse failed after folder/path metadata repair; "
                f"standard={standard_error}; repaired={repaired_error}"
            ) from repaired_error
        # 如果根本没有 folder/path 可替换，说明错误来自其他区域；禁止扩大修复范围。
        if not changed_tags:
            raise VocParseError(
                f"{xml_path.name}: XML parse failed and no folder/path metadata could be isolated: {standard_error}"
            ) from standard_error
        reason = (
            "standard XML parse failed; replaced only folder/path metadata in memory "
            f"({', '.join(changed_tags)}): {standard_error}"
        )
        return _annotation_from_root(root, xml_path, "repaired_metadata", reason)


def validate_voc_annotation(annotation: VocAnnotation) -> list[str]:
    """检查图片尺寸和检测框，返回所有问题；空列表表示通过。"""
    problems: list[str] = []
    # 宽高会作为归一化分母，必须大于 0；depth 也应是有效的通道数。
    if annotation.width <= 0 or annotation.height <= 0 or annotation.depth <= 0:
        problems.append(
            f"invalid image size width={annotation.width}, height={annotation.height}, depth={annotation.depth}"
        )
    for index, obj in enumerate(annotation.objects, start=1):
        prefix = f"object {index} ({obj.name})"
        # 使用 if/elif：一个框只记录最先发现的主要几何错误，报告更易读。
        if obj.xmin >= obj.xmax:
            problems.append(f"{prefix}: xmin must be less than xmax ({obj.xmin} >= {obj.xmax})")
        elif obj.ymin >= obj.ymax:
            problems.append(f"{prefix}: ymin must be less than ymax ({obj.ymin} >= {obj.ymax})")
        elif (
            # 左上角不能为负，右下角不能超过 XML 声明的图片宽高。
            obj.xmin < 0
            or obj.ymin < 0
            or obj.xmax > annotation.width
            or obj.ymax > annotation.height
        ):
            problems.append(
                f"{prefix}: box out of bounds for {annotation.width}x{annotation.height}: "
                f"({obj.xmin}, {obj.ymin}, {obj.xmax}, {obj.ymax})"
            )
    return problems


def voc_box_to_yolo(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """把 VOC 边界坐标转换为 YOLO 的归一化中心坐标。

    VOC 使用 ``xmin, ymin, xmax, ymax`` 像素边界；YOLO 使用
    ``x_center, y_center, width, height``，并除以图片宽高缩放到 0～1。
    """
    return (
        # 横向中心点：先求两条竖边的平均值，再除以整张图宽度。
        (xmin + xmax) / 2.0 / image_width,
        # 纵向中心点：先求两条横边的平均值，再除以整张图高度。
        (ymin + ymax) / 2.0 / image_height,
        # 框宽和框高同样除以图片尺寸，最终四个值都应落在 0～1。
        (xmax - xmin) / image_width,
        (ymax - ymin) / image_height,
    )
