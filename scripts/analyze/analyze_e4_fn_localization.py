#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

from PIL import Image
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from helmet_safety.training.baseline import allocate_run_name, load_ground_truth_boxes, write_json_report  # noqa: E402
from helmet_safety.training.analysis_core import (  # noqa: E402
    LOCALIZATION_BUCKETS,
    analyze_fn_candidate_evidence,
)


FIXED_WEIGHT = PROJECT_ROOT / "artifacts" / "training" / "m45_yolo11s_e75_960_001" / "weights" / "best.pt"
FIXED_SOURCE_DIR = PROJECT_ROOT / "artifacts" / "evaluation" / "m45_yolo11s_e75_960_tiny_val_001"
FIXED_MANIFEST = FIXED_SOURCE_DIR / "tiny_val_images.json"
FIXED_FALSE_NEGATIVES = FIXED_SOURCE_DIR / "tiny_val_false_negatives.json"
FIXED_IMGSZ = 960
FIXED_CONFIDENCE = 0.001
FIXED_MATCHING_IOU = 0.5
EXPECTED_IMAGES = 35
EXPECTED_ORIGINAL_FN = 51

BUCKET_LABELS = {
    "iou_ge_0_5": "定位合格，主要可能是低置信度",
    "iou_0_4_to_0_5": "轻微定位偏差",
    "iou_0_2_to_0_4": "明显定位偏差",
    "iou_0_1_to_0_2": "弱位置响应",
    "unresolved": "暂标 unresolved",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the frozen 51 E4 tiny FNs on 35 val images using imgsz=960, conf=0.001, "
            "matching IoU=0.5, and default NMS; test is never used"
        )
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--output-name", default="m45_yolo11s_e75_960_fn_localization_001")
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.cfg import DEFAULT_CFG

    if args.batch < 1:
        raise ValueError("batch must be at least 1")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {args.device!r} unavailable")
    for path in (FIXED_WEIGHT, FIXED_MANIFEST, FIXED_FALSE_NEGATIVES):
        if not path.is_file():
            raise FileNotFoundError(f"fixed input is unavailable: {path.resolve()}")

    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    frozen_hashes_before = {
        "manifest": sha256(FIXED_MANIFEST),
        "false_negatives": sha256(FIXED_FALSE_NEGATIVES),
    }
    manifest_document = json.loads(FIXED_MANIFEST.read_text(encoding="utf-8"))
    fn_document = json.loads(FIXED_FALSE_NEGATIVES.read_text(encoding="utf-8"))
    manifest = manifest_document["images"]
    original_false_negatives = fn_document["false_negatives"]
    image_ids = [str(item["image_id"]) for item in manifest]
    original_fn_keys = {
        (str(item["image_id"]), int(item["gt_index"])) for item in original_false_negatives
    }
    helmet_fn_count = sum(str(item["class_name"]) == "helmet" for item in original_false_negatives)
    no_helmet_fn_count = sum(str(item["class_name"]) == "no_helmet" for item in original_false_negatives)
    if len(manifest) != EXPECTED_IMAGES or len(set(image_ids)) != EXPECTED_IMAGES:
        raise RuntimeError("the frozen first-experiment manifest is not 35 unique val images")
    if (
        len(original_false_negatives) != EXPECTED_ORIGINAL_FN
        or len(original_fn_keys) != EXPECTED_ORIGINAL_FN
        or helmet_fn_count != 10
        or no_helmet_fn_count != 41
    ):
        raise RuntimeError("the frozen first-experiment FN set is not 51 unique keys split as 10 helmet and 41 no_helmet")

    training_args = yaml.safe_load((FIXED_WEIGHT.parents[2] / "args.yaml").read_text(encoding="utf-8"))
    dataset_yaml = Path(str(training_args["data"])).resolve()
    dataset = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    processed_root = Path(str(dataset["path"])).resolve()
    val_images_dir = (processed_root / str(dataset["val"])).resolve()
    val_labels_dir = (processed_root / "labels" / "val").resolve()
    image_paths = [Path(str(item["image_path"])).resolve() for item in manifest]
    if any(path.parent != val_images_dir or not path.is_file() for path in image_paths):
        raise RuntimeError("a frozen manifest image is missing or is not from the configured val split")

    ground_truth_by_image: dict[str, list[dict[str, object]]] = {}
    for image_path in image_paths:
        with Image.open(image_path) as image:
            image_size = image.size
        ground_truth_by_image[image_path.name] = load_ground_truth_boxes(
            val_labels_dir / f"{image_path.stem}.txt", image_size=image_size
        )
    for item in original_false_negatives:
        image_id, gt_index = str(item["image_id"]), int(item["gt_index"])
        ground_truth = ground_truth_by_image.get(image_id)
        if ground_truth is None or not 0 <= gt_index < len(ground_truth):
            raise RuntimeError(f"frozen FN key no longer resolves in val labels: {image_id}:{gt_index}")
        gt = ground_truth[gt_index]
        if int(gt["class_id"]) != int(item["class_id"]) or [float(value) for value in gt["box"]] != [
            float(value) for value in item["gt_box_xyxy"]
        ]:
            raise RuntimeError(f"frozen FN content no longer matches val labels: {image_id}:{gt_index}")

    model = YOLO(str(FIXED_WEIGHT.resolve()), task="detect")
    if dict(model.names) != {0: "helmet", 1: "no_helmet"}:
        raise ValueError(f"unexpected class mapping: {model.names}")

    predictions_by_image: dict[str, list[dict[str, object]]] = {}
    started = time.perf_counter()
    # Intentionally omit every NMS argument. Evidence is post-default-NMS at conf >= 0.001.
    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=FIXED_IMGSZ,
        conf=FIXED_CONFIDENCE,
        batch=args.batch,
        device=args.device,
        save=False,
        stream=True,
        verbose=False,
    )
    for result in results:
        image_id = Path(result.path).name
        predictions_by_image[image_id] = [
            {
                "class_id": int(class_id),
                "confidence": float(confidence),
                "box": [float(value) for value in box],
            }
            for class_id, confidence, box in zip(
                result.boxes.cls.detach().cpu().tolist(),
                result.boxes.conf.detach().cpu().tolist(),
                result.boxes.xyxy.detach().cpu().tolist(),
                strict=True,
            )
        ]
    wall_seconds = time.perf_counter() - started
    if len(predictions_by_image) != EXPECTED_IMAGES or set(predictions_by_image) != set(image_ids):
        raise RuntimeError("low-confidence inference did not return exactly the frozen 35 images")

    analysis_result = analyze_fn_candidate_evidence(original_false_negatives, predictions_by_image)
    details = analysis_result["details"]
    summary = analysis_result["summary"]
    helmet_details = analysis_result["helmet_false_negatives"]
    if len(details) != EXPECTED_ORIGINAL_FN or len(helmet_details) != 10:
        raise RuntimeError("candidate analysis did not preserve all 51 FNs or the 10 helmet FNs")
    for item in details:
        item["localization_bucket_label"] = BUCKET_LABELS[str(item["localization_bucket"])]

    summary_rows: list[dict[str, object]] = []
    for scope in ("overall", "helmet", "no_helmet"):
        for bucket in LOCALIZATION_BUCKETS:
            bucket_result = summary[scope]["buckets"][bucket]
            summary_rows.append(
                {
                    "scope": scope,
                    "total": int(summary[scope]["total"]),
                    "bucket": bucket,
                    "bucket_label": BUCKET_LABELS[bucket],
                    "count": int(bucket_result["count"]),
                    "share": float(bucket_result["share"]),
                }
            )

    qualified = [item for item in details if item["localization_bucket"] == "iou_ge_0_5"]
    qualified_below_025 = [
        item
        for item in qualified
        if item["correct_class_max_iou_candidate_confidence"] is not None
        and float(item["correct_class_max_iou_candidate_confidence"]) < 0.25
    ]
    qualified_at_or_above_025 = [
        item
        for item in qualified
        if item["correct_class_max_iou_candidate_confidence"] is not None
        and float(item["correct_class_max_iou_candidate_confidence"]) >= 0.25
    ]
    strong_wrong_class_overlap = [
        item
        for item in details
        if item["wrong_class_max_iou"] is not None and float(item["wrong_class_max_iou"]) >= FIXED_MATCHING_IOU
    ]
    unresolved = [item for item in details if item["localization_bucket"] == "unresolved"]
    unresolved_without_correct_candidate = [item for item in unresolved if int(item["correct_class_candidate_count"]) == 0]
    conclusions = {
        "localization_qualified_count": len(qualified),
        "localization_qualified_with_best_overlap_conf_below_025": len(qualified_below_025),
        "localization_qualified_with_best_overlap_conf_at_or_above_025": len(qualified_at_or_above_025),
        "wrong_class_max_iou_ge_05_count": len(strong_wrong_class_overlap),
        "unresolved_count": len(unresolved),
        "unresolved_without_visible_correct_class_candidate": len(unresolved_without_correct_candidate),
        "interpretation_limit": (
            "All evidence is post-default-NMS and conf>=0.001. Absence of a visible candidate does not prove "
            "that no pre-NMS candidate existed, and these results do not attribute any FN to NMS."
        ),
    }

    def format_number(value: object, digits: int = 6) -> str:
        return "N/A" if value is None else f"{float(value):.{digits}f}"

    markdown_lines = [
        "# E4 原始 FN 定位误差分析",
        "",
        "- 输入：第一组冻结的 35 张 val 图片和 51 个原始 FN；未使用 test。",
        "- 固定：E4 best.pt、imgsz=960、conf=0.001、matching IoU=0.5、默认 NMS。",
        "- 分桶依据：每个 GT 与正确类别后 NMS 候选的最大 IoU。confidence 最大值与 IoU 最大值独立计算，可能来自不同候选框。",
        "- 证据边界：只观察到默认 NMS 后且 conf>=0.001 的候选；不能据此断言 NMS 导致 FN，也不能把无可见候选解释为完全无 NMS 前响应。",
        "",
        "## overall / helmet / no_helmet 汇总",
        "",
        "| 范围 | 分桶 | 数量 | 占比 |",
        "|---|---|---:|---:|",
    ]
    for row in summary_rows:
        markdown_lines.append(
            f"| {row['scope']} | {row['bucket_label']} | {row['count']} | {float(row['share']):.2%} |"
        )
    markdown_lines.extend(
        [
            "",
            "## 10 个 helmet FN",
            "",
            "| image_id | gt_index | GT xyxy | 正类最高 conf | 正类最大 IoU | 最大 IoU 候选 conf | 错类最高 conf | 错类最大 IoU | 分桶 |",
            "|---|---:|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for item in helmet_details:
        markdown_lines.append(
            f"| {item['image_id']} | {item['gt_index']} | {item['gt_box_xyxy']} | "
            f"{format_number(item['correct_class_max_confidence'])} | {format_number(item['correct_class_max_iou'])} | "
            f"{format_number(item['correct_class_max_iou_candidate_confidence'])} | "
            f"{format_number(item['wrong_class_max_confidence'])} | {format_number(item['wrong_class_max_iou'])} | "
            f"{item['localization_bucket_label']} |"
        )
    markdown_lines.extend(
        [
            "",
            "## 全部 51 个原始 FN",
            "",
            "| image_id | gt_index | 类别 | 正类最高 conf | 正类最大 IoU | 错类最高 conf | 错类最大 IoU | 分桶 |",
            "|---|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    for item in details:
        markdown_lines.append(
            f"| {item['image_id']} | {item['gt_index']} | {item['class_name']} | "
            f"{format_number(item['correct_class_max_confidence'])} | {format_number(item['correct_class_max_iou'])} | "
            f"{format_number(item['wrong_class_max_confidence'])} | {format_number(item['wrong_class_max_iou'])} | "
            f"{item['localization_bucket_label']} |"
        )
    markdown_lines.extend(
        [
            "",
            "## 证据摘要",
            "",
            f"- 正类最大 IoU>=0.5：{len(qualified)} 个；其中最大 IoU 候选 confidence<0.25：{len(qualified_below_025)} 个，>=0.25：{len(qualified_at_or_above_025)} 个。",
            f"- 错类最大 IoU>=0.5：{len(strong_wrong_class_overlap)} 个；这只表示存在重叠的错误类别后 NMS 候选，不单独证明分类错误。",
            f"- unresolved：{len(unresolved)} 个；其中 {len(unresolved_without_correct_candidate)} 个在 conf>=0.001 的后 NMS 输出中没有正确类别候选。",
            "- 没有 NMS 前候选证据，因此不做“NMS 导致”或“完全无候选”的断言。",
            "",
        ]
    )

    frozen_hashes_after = {
        "manifest": sha256(FIXED_MANIFEST),
        "false_negatives": sha256(FIXED_FALSE_NEGATIVES),
    }
    if frozen_hashes_after != frozen_hashes_before:
        raise RuntimeError("a frozen first-experiment input changed during localization analysis")

    evaluation_root = PROJECT_ROOT / "artifacts" / "evaluation"
    run_name = allocate_run_name(evaluation_root, args.output_name)
    output_dir = evaluation_root / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    detail_csv_path = output_dir / "e4_fn_localization_details.csv"
    summary_csv_path = output_dir / "e4_fn_localization_summary.csv"
    markdown_path = output_dir / "e4_fn_localization_report.md"
    details_json_path = output_dir / "e4_fn_localization_details.json"
    report_path = output_dir / "e4_fn_localization_report.json"

    detail_csv_rows = []
    for item in details:
        row = dict(item)
        row["gt_box_xyxy"] = json.dumps(row["gt_box_xyxy"], ensure_ascii=False)
        detail_csv_rows.append(row)
    with detail_csv_path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_csv_rows[0]))
        writer.writeheader()
        writer.writerows(detail_csv_rows)
    with summary_csv_path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")
    write_json_report(
        details_json_path,
        {
            "candidate_stage": "post-default-NMS",
            "minimum_confidence": FIXED_CONFIDENCE,
            "details": details,
            "helmet_false_negatives": helmet_details,
        },
    )

    report = {
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "weight": str(FIXED_WEIGHT.resolve()),
            "weight_sha256": sha256(FIXED_WEIGHT),
            "source_manifest": str(FIXED_MANIFEST.resolve()),
            "source_false_negatives": str(FIXED_FALSE_NEGATIVES.resolve()),
            "source_hashes": frozen_hashes_before,
            "images": EXPECTED_IMAGES,
            "original_false_negatives": EXPECTED_ORIGINAL_FN,
            "split": "val",
            "test_evaluated": False,
            "imgsz": FIXED_IMGSZ,
            "confidence": FIXED_CONFIDENCE,
            "matching_iou": FIXED_MATCHING_IOU,
            "nms_parameters_overridden": False,
            "default_nms": {
                "iou": float(DEFAULT_CFG.iou),
                "max_det": int(DEFAULT_CFG.max_det),
                "agnostic_nms": bool(DEFAULT_CFG.agnostic_nms),
            },
            "candidate_stage": "post-default-NMS",
            "pre_nms_candidates_available": False,
            "bucket_basis": "maximum IoU between each GT and same-class visible candidate",
            "batch": args.batch,
            "device": args.device,
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "summary": summary,
        "conclusions": conclusions,
        "wall_seconds": wall_seconds,
        "artifacts": {
            "detail_csv": str(detail_csv_path.resolve()),
            "summary_csv": str(summary_csv_path.resolve()),
            "markdown": str(markdown_path.resolve()),
            "details_json": str(details_json_path.resolve()),
            "report": str(report_path.resolve()),
        },
    }
    write_json_report(report_path, report)
    return report


def main() -> int:
    try:
        report = run(build_parser().parse_args())
    except Exception as exc:
        print(json.dumps({"status": "failed", "reason": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": report["status"],
                "summary": report["summary"],
                "conclusions": report["conclusions"],
                "artifacts": report["artifacts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
