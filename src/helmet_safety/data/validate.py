"""独立验证转换后的 YOLO 数据，并生成固定 seed 的标注可视化。

验证不会相信 conversion_report 中的统计数字，而是重新读取官方 XML 和实际 txt，
逐行、逐框比较。这样转换代码即使产生错误，也更有机会被独立检查发现。
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFont

from .audit import SPLIT_NAMES, check_split_integrity, load_splits
from .convert import CLASS_MAPPING, MANIFEST_NAME
from .voc import parse_voc_xml


def parse_yolo_line(
    line: str,
    *,
    label_path: Path | None = None,
    line_number: int | None = None,
) -> tuple[int, float, float, float, float]:
    """严格解析一行 YOLO 标签，成功时返回 ``类别 + 4 个坐标``。"""
    # 把文件名和行号拼进错误信息，真实数据出错时可直接定位。
    location = f"{label_path or '<label>'}:{line_number or '?'}"
    # 合法格式固定为：class_id x_center y_center width height。
    columns = line.split()
    if len(columns) != 5:
        raise ValueError(f"{location}: expected exactly 5 columns, got {len(columns)}")
    try:
        # 类别必须是整数；例如 0.5 不能被当作合法类别。
        class_id = int(columns[0])
    except ValueError as exc:
        raise ValueError(f"{location}: class_id must be an integer") from exc
    if class_id not in (0, 1):
        # 本数据集只定义 helmet=0、no_helmet=1。
        raise ValueError(f"{location}: class_id must be 0 or 1, got {class_id}")
    try:
        # 后四列转换成 float；tuple 保证返回值顺序固定且不会被修改。
        values = tuple(float(value) for value in columns[1:])
    except ValueError as exc:
        raise ValueError(f"{location}: coordinates must be floating-point numbers") from exc
    # isfinite 会额外排除 nan、inf；仅比较 0～1 时容易漏掉 nan。
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"{location}: all four coordinates must be finite and between 0 and 1")
    # values[2]/[3] 是宽高。中心点允许在边界 0 或 1，框宽高不允许为 0。
    if values[2] <= 0.0 or values[3] <= 0.0:
        raise ValueError(f"{location}: width and height must be greater than 0")
    return class_id, values[0], values[1], values[2], values[3]


def pixel_roundtrip_error(
    *,
    voc_box: tuple[float, float, float, float],
    yolo_box: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> float:
    """把 YOLO 框反算回像素边界，返回四条边中最大的绝对误差。"""
    x_center, y_center, width, height = yolo_box
    # 左边界 = 中心 - 一半宽度；右边界 = 中心 + 一半宽度，y 轴同理。
    reconstructed = (
        (x_center - width / 2.0) * image_width,
        (y_center - height / 2.0) * image_height,
        (x_center + width / 2.0) * image_width,
        (y_center + height / 2.0) * image_height,
    )
    # zip(..., strict=True) 保证两个框都有且只有四个值；取最大边误差作为该框误差。
    return max(abs(expected - actual) for expected, actual in zip(voc_box, reconstructed, strict=True))


def compare_annotation_counts(
    *,
    before: dict[str, int],
    after: dict[str, int],
    recorded_ignored_dogs: int,
) -> dict[str, object]:
    """核对有效类别一项未丢，并确认 dog 数量与跳过记录相同。"""
    return {
        "before": dict(before),
        "after": dict(after),
        "recorded_ignored_dogs": recorded_ignored_dogs,
        # 两个有效类别都相等，valid_targets_match 才为 True。
        "valid_targets_match": before.get("hat", 0) == after.get("helmet", 0)
        and before.get("person", 0) == after.get("no_helmet", 0),
        "ignored_dogs_match": before.get("dog", 0) == recorded_ignored_dogs,
    }


def _select_visualization_ids(
    candidates: list[dict[str, object]], count: int, rng: random.Random
) -> list[dict[str, object]]:
    """固定随机抽取双类别、密集目标、单类别三类代表性图片。"""
    # 如果某个 split 少于期望样本数，就最多选择实际拥有的数量。
    count = min(count, len(candidates))
    quota = max(1, count // 3)
    selected: list[dict[str, object]] = []
    selected_ids: set[str] = set()

    def add(items: list[dict[str, object]], limit: int) -> None:
        """从一组候选中最多加入 limit 个，并自动跳过之前已选中的 ID。"""
        for item in items:
            image_id = str(item["id"])
            if image_id not in selected_ids:
                selected.append(item)
                selected_ids.add(image_id)
                # 总数达到 count，或当前类别配额用完，就停止本轮添加。
                if len(selected) >= count or limit <= 1:
                    return
                limit -= 1

    # 先从三种“有代表性”的集合取样，不足的名额再从所有候选中补齐。
    both = [item for item in candidates if item["classes"] == {0, 1}]
    rng.shuffle(both)
    add(both, quota)
    # 负号让 box_count 从大到小排序；ID 用作数量相同时的稳定次级排序。
    dense = sorted(candidates, key=lambda item: (-int(item["box_count"]), str(item["id"])))
    add(dense, quota)
    single = [item for item in candidates if len(item["classes"]) == 1]
    rng.shuffle(single)
    add(single, quota)
    # 三类代表样本不足时，从全集随机补齐；selected_ids 会自动去重。
    remainder = list(candidates)
    rng.shuffle(remainder)
    add(remainder, count - len(selected))
    return selected[:count]


def generate_visualizations(
    processed_root: Path | str,
    visualization_root: Path | str,
    *,
    samples_per_split: int = 10,
    seed: int = 2028,
    force: bool = False,
) -> dict[str, object]:
    """在图片副本上画 YOLO 框；固定 seed 可保证重复抽到同一批图片。"""
    if samples_per_split < 1:
        raise ValueError("samples_per_split must be at least 1")
    processed = Path(processed_root).resolve()
    output = Path(visualization_root).resolve()
    manifest_path = output / ".visualization_manifest.json"
    # 可视化也采用 manifest，避免 force 时误删用户放入目录的其他文件。
    if output.exists() and any(output.iterdir()):
        if not force:
            raise FileExistsError(f"visualization output is not empty: {output}; pass --force for a safe rerun")
        if not manifest_path.is_file():
            raise FileExistsError(f"cannot safely overwrite visualizations without manifest: {manifest_path}")
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for relative in old_manifest.get("generated_files", []):
            target = (output / relative).resolve()
            try:
                # 与转换 manifest 一样，先阻止 ../ 跳出 visualizations 目录。
                target.relative_to(output)
            except ValueError as exc:
                raise ValueError(f"unsafe visualization manifest path: {relative}") from exc
            if target.is_file():
                target.unlink()
        manifest_path.unlink()
    output.mkdir(parents=True, exist_ok=True)

    colors = {0: (0, 210, 70), 1: (255, 70, 70)}
    names = {0: "helmet", 1: "no_helmet"}
    generated: list[str] = []
    report: dict[str, object] = {"seed": seed, "samples_per_split": samples_per_split, "splits": {}}
    for split_index, split in enumerate(SPLIT_NAMES):
        image_paths = _index_stems(processed / "images" / split, {".jpg", ".jpeg"})
        label_paths = _index_stems(processed / "labels" / split, {".txt"})
        candidates: list[dict[str, object]] = []
        parsed_by_id: dict[str, list[tuple[int, float, float, float, float]]] = {}
        # 只把图片和标签同时存在的 ID 放入可视化候选集合。
        for image_id in sorted(set(image_paths) & set(label_paths)):
            parsed = [
                parse_yolo_line(line, label_path=label_paths[image_id], line_number=index)
                for index, line in enumerate(label_paths[image_id].read_text(encoding="utf-8").splitlines(), start=1)
                if line.strip()
            ]
            parsed_by_id[image_id] = parsed
            candidates.append(
                {
                    "id": image_id,
                    "box_count": len(parsed),
                    # set 自动去重，用 {0,1}、{0}、{1} 判断双类或单类图片。
                    "classes": {item[0] for item in parsed},
                }
            )
        # 为每个 split 派生不同但稳定的随机序列。
        rng = random.Random(seed + split_index * 1_000_003)
        selected = _select_visualization_ids(candidates, samples_per_split, rng)
        split_output = output / split
        split_output.mkdir(parents=True, exist_ok=True)
        split_items: list[dict[str, object]] = []
        for candidate in selected:
            image_id = str(candidate["id"])
            image_path = image_paths[image_id]
            with Image.open(image_path) as source_image:
                # convert 会创建独立的内存画布；后面的 draw 不会改 processed 原图。
                canvas = source_image.convert("RGB")
            draw = ImageDraw.Draw(canvas)
            font = ImageFont.load_default()
            # 绘图前把 0～1 的 YOLO 坐标乘回当前图片宽高。
            for class_id, x_center, y_center, width, height in parsed_by_id[image_id]:
                # 先从中心点减/加半宽高得到归一化边界，再乘图片像素尺寸。
                left = (x_center - width / 2.0) * canvas.width
                top = (y_center - height / 2.0) * canvas.height
                right = (x_center + width / 2.0) * canvas.width
                bottom = (y_center + height / 2.0) * canvas.height
                color = colors[class_id]
                # 大图使用更粗线条，小图至少 2 像素，保证框肉眼可见。
                line_width = max(2, round(min(canvas.size) / 300))
                draw.rectangle((left, top, right, bottom), outline=color, width=line_width)
                label = names[class_id]
                # 先测量文字尺寸，再画有颜色的背景块，避免类别名被复杂背景淹没。
                text_box = draw.textbbox((left, top), label, font=font)
                text_height = text_box[3] - text_box[1]
                text_width = text_box[2] - text_box[0]
                # 标签优先放在框上方；max(0, ...) 防止画出图片顶部。
                label_top = max(0, top - text_height - 4)
                draw.rectangle((left, label_top, left + text_width + 4, label_top + text_height + 4), fill=color)
                draw.text((left + 2, label_top + 2), label, fill=(0, 0, 0), font=font)
            target = split_output / f"{image_id}.jpg"
            # 保存到 audit/visualizations，而不是覆盖 processed/images 中的训练图片。
            canvas.save(target, format="JPEG", quality=92)
            relative = target.relative_to(output).as_posix()
            generated.append(relative)
            classes = sorted(candidate["classes"])
            split_items.append(
                {
                    "id": image_id,
                    "file": relative,
                    "box_count": candidate["box_count"],
                    "classes": [names[class_id] for class_id in classes],
                    # 记录来源图片哈希，之后可以确认抽样对应的是哪份原图内容。
                    "source_sha256": sha256(image_path.read_bytes()).hexdigest(),
                }
            )
        report["splits"][split] = {"count": len(split_items), "samples": split_items}
    # 所有可视化成功写完后才落 manifest，为下一次安全 --force 提供清单。
    manifest_path.write_text(
        json.dumps({"generated_files": generated}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _index_stems(directory: Path, suffixes: set[str]) -> dict[str, Path]:
    """按文件 stem 建索引；目录不存在时返回空字典，让验证报告缺失项。"""
    if not directory.is_dir():
        return {}
    return {
        path.stem: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in suffixes
    }


def _validation_markdown(report: dict[str, object]) -> str:
    """把 JSON 验证结果转换为适合人工阅读的 Markdown 摘要。"""
    lines = [
        "# SHWD YOLO 转换验证报告",
        "",
        f"- 状态：**{report['status']}**",
        f"- 原始目录：`{report['raw_root']}`",
        f"- 转换目录：`{report['processed_root']}`",
        f"- 有效目标数量一致：{report['counts']['valid_targets_match']}",
        f"- dog 忽略记录一致：{report['counts']['ignored_dogs_match']}",
        f"- 最大像素回算误差：{report['roundtrip']['maximum_pixel_error']:.6f}px",
        "",
        "## 划分统计",
        "",
        "| split | images | labels | boxes |",
        "|---|---:|---:|---:|",
    ]
    for split in SPLIT_NAMES:
        item = report["splits"][split]
        lines.append(f"| {split} | {item['image_count']} | {item['label_count']} | {item['box_count']} |")
    lines.extend(["", "## 问题", "", "```json", json.dumps(report["issues"], ensure_ascii=False, indent=2), "```"])
    return "\n".join(lines) + "\n"


def validate_dataset(
    raw_root: Path | str,
    processed_root: Path | str,
    report_dir: Path | str,
    *,
    pixel_tolerance: float = 0.01,
) -> dict[str, object]:
    """检查文件配对、标签语法、类别总数、split 和逐框像素回算。"""
    raw = Path(raw_root).resolve()
    processed = Path(processed_root).resolve()
    destination = Path(report_dir).resolve()
    splits = load_splits(raw)
    official_integrity = check_split_integrity(splits)
    # 所有问题都累积到 issues；最终 issues 为空才判定 passed。
    issues: list[dict[str, object]] = []
    # before 来自原 VOC，after 来自实际 YOLO txt，两者独立累计。
    before: Counter[str] = Counter()
    after: Counter[str] = Counter()
    maximum_error = 0.0
    checked_boxes = 0
    split_reports: dict[str, object] = {}
    output_ids: dict[str, list[str]] = {}

    # 转换报告只用来取得“明确记录的 dog 跳过数量”；其他数量会重新计算。
    conversion_report_path = processed / "conversion_report.json"
    # -1 是“报告缺失”的哨兵值，避免误把缺失报告当成 dog=0。
    recorded_ignored_dogs = -1
    if conversion_report_path.is_file():
        conversion_report = json.loads(conversion_report_path.read_text(encoding="utf-8"))
        recorded_ignored_dogs = len(conversion_report.get("ignored_dog_objects", []))
    else:
        issues.append({"type": "missing_conversion_report", "path": str(conversion_report_path)})

    # 逐个 split 检查：文件集合 -> 标签行 -> 类别/框数量 -> 像素回算。
    for split in SPLIT_NAMES:
        image_paths = _index_stems(processed / "images" / split, {".jpg", ".jpeg"})
        label_paths = _index_stems(processed / "labels" / split, {".txt"})
        # 三个 set 分别代表 processed 图片、processed 标签、官方应有 ID。
        image_ids = set(image_paths)
        label_ids = set(label_paths)
        official_ids = set(splits[split])
        output_ids[split] = sorted(image_ids)
        if image_ids != label_ids:
            # 两个方向分别报告：有图无标签，以及有标签无图。
            issues.append(
                {
                    "type": "image_label_mismatch",
                    "split": split,
                    "images_without_labels": sorted(image_ids - label_ids),
                    "labels_without_images": sorted(label_ids - image_ids),
                }
            )
        if image_ids != official_ids:
            # 即使图片和标签互相配对，也可能整体少于或多于官方 split。
            issues.append(
                {
                    "type": "official_split_mismatch",
                    "split": split,
                    "missing": sorted(official_ids - image_ids),
                    "extra": sorted(image_ids - official_ids),
                }
            )

        box_count = 0
        for image_id in sorted(official_ids):
            xml_path = raw / "Annotations" / f"{image_id}.xml"
            label_path = label_paths.get(image_id)
            image_path = image_paths.get(image_id)
            if not xml_path.is_file() or label_path is None or image_path is None:
                # 缺失问题已在集合检查中记录；这里跳过，避免二次读取不存在文件。
                continue
            try:
                with Image.open(image_path) as image:
                    image.load()
            except Exception as exc:
                issues.append(
                    {"type": "processed_image_decode_error", "split": split, "image": image_path.name, "reason": str(exc)}
                )
            if image_path.suffix.lower() in {".jpg", ".jpeg"}:
                with image_path.open("rb") as handle:
                    handle.seek(-2, 2)
                    if handle.read(2) != b"\xff\xd9":
                        issues.append({"type": "noncanonical_jpeg", "split": split, "image": image_path.name})
            try:
                annotation = parse_voc_xml(xml_path)
            except Exception as exc:
                issues.append({"type": "source_xml_parse_error", "xml": xml_path.name, "reason": str(exc)})
                continue
            # expected_objects 只保留应该进入 YOLO 的 hat/person，dog 不参与逐框配对。
            expected_objects = []
            for obj in annotation.objects:
                before[obj.name] += 1
                if obj.name in CLASS_MAPPING:
                    expected_objects.append(obj)
                elif obj.name != "dog":
                    issues.append({"type": "unknown_source_class", "xml": xml_path.name, "class": obj.name})

            parsed_lines: list[tuple[int, float, float, float, float]] = []
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    # 整个文件为空是合法空标签；文件中间出现空白行则可疑，单独报告。
                    issues.append({"type": "blank_label_line", "label": label_path.name, "line": line_number})
                    continue
                try:
                    parsed_lines.append(parse_yolo_line(line, label_path=label_path, line_number=line_number))
                except ValueError as exc:
                    issues.append({"type": "invalid_label", "label": label_path.name, "reason": str(exc)})
            box_count += len(parsed_lines)
            for class_id, *_ in parsed_lines:
                # 转成最终可读类别名，稍后与 hat/person 一一核对总数。
                after["helmet" if class_id == 0 else "no_helmet"] += 1
            if len(parsed_lines) != len(expected_objects):
                # 数量不同就无法安全 zip 配对，因此记录后直接处理下一张图片。
                issues.append(
                    {
                        "type": "box_count_mismatch",
                        "split": split,
                        "id": image_id,
                        "voc": len(expected_objects),
                        "yolo": len(parsed_lines),
                    }
                )
                continue
            # 转换保持 object 顺序，因此可以一一配对检查类别和四条边。
            for object_index, (obj, parsed) in enumerate(zip(expected_objects, parsed_lines, strict=True), start=1):
                expected_class = CLASS_MAPPING[obj.name]
                if parsed[0] != expected_class:
                    issues.append(
                        {
                            "type": "class_mismatch",
                            "split": split,
                            "id": image_id,
                            "object_index": object_index,
                            "expected": expected_class,
                            "actual": parsed[0],
                        }
                    )
                error = pixel_roundtrip_error(
                    voc_box=(obj.xmin, obj.ymin, obj.xmax, obj.ymax),
                    yolo_box=parsed[1:],
                    image_width=annotation.width,
                    image_height=annotation.height,
                )
                checked_boxes += 1
                # maximum_error 保存全数据集最坏情况，报告无需列出 12 万个正常框。
                maximum_error = max(maximum_error, error)
                if error > pixel_tolerance:
                    issues.append(
                        {
                            "type": "pixel_roundtrip_error",
                            "split": split,
                            "id": image_id,
                            "object_index": object_index,
                            "error": error,
                            "tolerance": pixel_tolerance,
                        }
                    )
        split_reports[split] = {
            "image_count": len(image_paths),
            "label_count": len(label_paths),
            "box_count": box_count,
            "images_without_labels": sorted(image_ids - label_ids),
            "labels_without_images": sorted(label_ids - image_ids),
        }

    # 文件名集合必须仍然互斥，不能在复制阶段把一张图放进两个 split。
    output_integrity = check_split_integrity(
        {**output_ids, "trainval": sorted(set(output_ids["train"]) | set(output_ids["val"]))}
    )
    if not official_integrity["mutually_exclusive"] or not output_integrity["mutually_exclusive"]:
        issues.append(
            {
                "type": "split_overlap",
                "official": official_integrity["intersections"],
                "processed": output_integrity["intersections"],
            }
        )
    # 逐框验证之外，再做一次全局总数兜底，防止局部逻辑遗漏。
    counts = compare_annotation_counts(
        before={"hat": before["hat"], "person": before["person"], "dog": before["dog"]},
        after={"helmet": after["helmet"], "no_helmet": after["no_helmet"]},
        recorded_ignored_dogs=recorded_ignored_dogs,
    )
    if not counts["valid_targets_match"]:
        issues.append({"type": "valid_target_count_mismatch", "details": counts})
    if not counts["ignored_dogs_match"]:
        issues.append({"type": "ignored_dog_count_mismatch", "details": counts})

    # 本工具自己生成固定 YAML，所以这里采用精确文本核对，能发现路径或类别被改动。
    required_yaml = (
        f"path: {processed.as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n\n"
        "names:\n  0: helmet\n  1: no_helmet\n"
    )
    yaml_path = processed / "dataset.yaml"
    if not yaml_path.is_file() or yaml_path.read_text(encoding="utf-8") != required_yaml:
        issues.append({"type": "dataset_yaml_mismatch", "path": str(yaml_path)})

    # status 只由 issues 决定；任何一个格式、数量、split 或误差问题都会 failed。
    report: dict[str, object] = {
        "status": "passed" if not issues else "failed",
        "raw_root": raw.as_posix(),
        "processed_root": processed.as_posix(),
        "splits": split_reports,
        "split_integrity": {"official": official_integrity, "processed": output_integrity},
        "counts": counts,
        "roundtrip": {
            "checked_box_count": checked_boxes,
            "pixel_tolerance": pixel_tolerance,
            "maximum_pixel_error": maximum_error,
        },
        "issues": issues,
        "manifest_present": (processed / MANIFEST_NAME).is_file(),
    }
    # 验证只向 report_dir 写报告，不改 raw，也不改 processed 标签或图片。
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (destination / "validation_report.md").write_text(_validation_markdown(report), encoding="utf-8")
    return report
