#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Mapping

from PIL import Image
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.evaluate.evaluate_e7_lfr import (  # noqa: E402
    MATCHING_IOU,
    collect_records,
    count_class_confusions,
    sha256,
    tiny_scopes,
)
from helmet_safety.training.baseline import (  # noqa: E402
    format_evaluation_metrics,
    load_ground_truth_boxes,
    write_json_report,
)
from helmet_safety.training.e7_lfr import register_lfr_module  # noqa: E402
from helmet_safety.training.e8_no_eca import (  # noqa: E402
    SIZE_BUCKETS,
    register_no_eca_module,
    summarize_size_buckets,
)
from helmet_safety.training.analysis_core import summarize_p0_records, validated_streaming_image_source  # noqa: E402


E4_WEIGHT = PROJECT_ROOT / "artifacts" / "training" / "m45_yolo11s_e75_960_001" / "weights" / "best.pt"
E6_WEIGHT = PROJECT_ROOT / "artifacts" / "e6" / "e6_yolo11s_p2_001" / "weights" / "best.pt"
E7_WEIGHT = PROJECT_ROOT / "artifacts" / "e7" / "e7_yolo11s_p2_lfr_001" / "weights" / "best.pt"
E8_DIR = PROJECT_ROOT / "artifacts" / "e8" / "e8_yolo11s_p2_lfr_no_eca_001"
E8_WEIGHT = E8_DIR / "weights" / "best.pt"
DATASET_YAML = Path(r"D:\datasets\SHWD\processed\dataset.yaml")
EVALUATION_DIR = E8_DIR / "evaluation"
REPORT_JSON = E8_DIR / "e4_e6_e7_e8_full_val_comparison.json"
REPORT_MD = E8_DIR / "e4_e6_e7_e8_full_val_comparison_zh.md"
EXPECTED_IMAGES = 607
EXPECTED_GT = 9925
FIXED_CONF = 0.25
NMS_IOU = 0.70
MAX_DET = 300
EXPERIMENT_ORDER = ("E4", "E6", "E7", "E8")


def add_standard_f1(metrics: dict[str, object]) -> None:
    for row in [metrics["overall"], *metrics["per_class"].values()]:  # type: ignore[union-attr]
        precision, recall = float(row["precision"]), float(row["recall"])
        row["f1"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def combined_recall(size_metrics: Mapping[str, object], buckets: tuple[str, ...]) -> dict[str, object]:
    rows = [size_metrics["overall"][bucket] for bucket in buckets]  # type: ignore[index]
    ground_truth = sum(int(row["ground_truth"]) for row in rows)
    tp = sum(int(row["tp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    return {"ground_truth": ground_truth, "tp": tp, "fn": fn, "recall": tp / ground_truth if ground_truth else 0.0}


def build_ablation_judgments(experiments: Mapping[str, object]) -> dict[str, object]:
    e6, e7, e8 = experiments["E6"], experiments["E7"], experiments["E8"]  # type: ignore[assignment]
    medium = {name: combined_recall(experiments[name]["size_buckets"], ("10-20", "20-30")) for name in ("E6", "E7", "E8")}  # type: ignore[index]
    tiny = {name: experiments[name]["size_buckets"]["overall"]["<=10"] for name in ("E6", "E7", "E8")}  # type: ignore[index]
    e7_gain = float(medium["E7"]["recall"]) - float(medium["E6"]["recall"])
    e8_gain = float(medium["E8"]["recall"]) - float(medium["E6"]["recall"])
    e7_has_benefit = e7_gain > 0.0
    retained = e7_has_benefit and e8_gain > 0.0
    fully_retained = e7_has_benefit and e8_gain >= e7_gain
    retained_fraction = e8_gain / e7_gain if e7_has_benefit else None
    tiny_restored = float(tiny["E8"]["recall"]) >= float(tiny["E6"]["recall"])
    tiny_improved_vs_e7 = float(tiny["E8"]["recall"]) > float(tiny["E7"]["recall"])
    eca_effects = {
        "precision_delta_pp_e7_minus_e8": 100 * (float(e7["standard_val"]["overall"]["precision"]) - float(e8["standard_val"]["overall"]["precision"])),  # type: ignore[index]
        "map50_95_delta_pp_e7_minus_e8": 100 * (float(e7["standard_val"]["overall"]["map50_95"]) - float(e8["standard_val"]["overall"]["map50_95"])),  # type: ignore[index]
        "false_positive_delta_e7_minus_e8": int(e7["fixed_threshold"]["overall"]["fp"]) - int(e8["fixed_threshold"]["overall"]["fp"]),  # type: ignore[index]
        "class_confusion_delta_e7_minus_e8": int(e7["class_confusions"]) - int(e8["class_confusions"]),
    }
    eca_improves_all_three = (
        eca_effects["precision_delta_pp_e7_minus_e8"] > 0
        and eca_effects["false_positive_delta_e7_minus_e8"] < 0
        and eca_effects["class_confusion_delta_e7_minus_e8"] < 0
    )
    candidate_score = {
        name: (
            float(experiments[name]["standard_val"]["overall"]["map50_95"]),  # type: ignore[index]
            float(experiments[name]["standard_val"]["overall"]["f1"]),  # type: ignore[index]
            -int(experiments[name]["fixed_threshold"]["overall"]["fp"]),  # type: ignore[index]
        )
        for name in ("E6", "E7", "E8")
    }
    final_candidate = max(candidate_score, key=candidate_score.__getitem__)
    return {
        "medium_10_30": medium,
        "e7_gain_over_e6_recall_pp": 100 * e7_gain,
        "e8_gain_over_e6_recall_pp": 100 * e8_gain,
        "e7_has_10_30_benefit": e7_has_benefit,
        "e8_retains_e7_10_30_benefit": retained,
        "e8_fully_retains_e7_10_30_benefit": fully_retained,
        "e8_retained_fraction_of_e7_10_30_gain": retained_fraction,
        "tiny_le10": tiny,
        "e8_restores_tiny_to_e6_or_better": tiny_restored,
        "e8_improves_tiny_vs_e7": tiny_improved_vs_e7,
        "eca_effects_e7_minus_e8": eca_effects,
        "eca_improves_false_positives_class_separation_and_precision": eca_improves_all_three,
        "eca_suitable_for_current_p2_refinement": eca_improves_all_three and eca_effects["map50_95_delta_pp_e7_minus_e8"] >= 0,
        "final_candidate_rule": "highest mAP50-95, then F1, then fewer fixed-conf false positives",
        "final_candidate": final_candidate,
    }


def render_markdown(report: Mapping[str, object]) -> str:
    experiments = report["experiments"]  # type: ignore[assignment]
    judgments = report["judgments"]  # type: ignore[assignment]
    lines = [
        "# E8 去除 ECA 消融实验：E4/E6/E7/E8 完整 val 对比",
        "",
        "- 数据：仅原始 train 5,457 张用于训练；完整 val 607 张、9,925 个 GT 用于评估；test 未使用。",
        "- 统一设置：imgsz=960、batch=2、seed=42、deterministic=True。",
        "- 固定阈值计数：conf=0.25、NMS IoU=0.70、class-aware matching IoU=0.50、max_det=300。",
        "- 尺寸定义：原图框等效边长 sqrt(w×h)，边界为 ≤10、10–20、20–30、30–50、>50 像素。",
        "",
        "## 总体与分类标准指标",
        "",
        "| 实验 | 范围 | Precision | Recall | F1 | mAP50 | mAP50-95 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for experiment in EXPERIMENT_ORDER:
        standard = experiments[experiment]["standard_val"]
        for scope in ("overall", "helmet", "no_helmet"):
            row = standard["overall"] if scope == "overall" else standard["per_class"][scope]
            lines.append(f"| {experiment} | {scope} | {row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} | {row['map50']:.6f} | {row['map50_95']:.6f} |")
    lines.extend(["", "## conf=0.25 检出、漏检、误检与类别混淆", "", "| 实验 | 范围 | GT | 正确检出 TP | 漏检 FN | 误检 FP | 类别混淆 |", "|---|---|---:|---:|---:|---:|---:|"])
    for experiment in EXPERIMENT_ORDER:
        fixed = experiments[experiment]["fixed_threshold"]
        for scope in ("overall", "helmet", "no_helmet"):
            row = fixed["overall"] if scope == "overall" else fixed["per_class"][scope]
            confusion = experiments[experiment]["class_confusions"] if scope == "overall" else "—"
            lines.append(f"| {experiment} | {scope} | {row['ground_truth']} | {row['tp']} | {row['fn']} | {row['fp']} | {confusion} |")
    lines.extend(["", "## 原图等效边长尺寸分桶（class-aware TP/FN）", "", "| 实验 | 范围 | 桶 | GT | TP | FN | Recall |", "|---|---|---|---:|---:|---:|---:|"])
    for experiment in EXPERIMENT_ORDER:
        for scope in ("overall", "helmet", "no_helmet"):
            for bucket in SIZE_BUCKETS:
                row = experiments[experiment]["size_buckets"][scope][bucket]
                lines.append(f"| {experiment} | {scope} | {bucket} | {row['ground_truth']} | {row['tp']} | {row['fn']} | {row['recall']:.6f} |")
    lines.extend(["", "## 参数量、GFLOPs 与推理速度", "", "| 实验 | 参数量 | GFLOPs@960 | 标准 val 推理 ms/图 | 固定阈值推理 ms/图 | 固定阈值 FPS |", "|---|---:|---:|---:|---:|---:|"])
    for experiment in EXPERIMENT_ORDER:
        row = experiments[experiment]
        std_ms = row["standard_val"]["speed_ms_per_image"]["inference"]
        pred_ms = row["fixed_threshold_prediction_speed"]["average_ms_per_image"]["inference"]
        lines.append(f"| {experiment} | {row['parameters']:,} | {row['gflops_at_960']:.3f} | {std_ms:.3f} | {pred_ms:.3f} | {row['fixed_threshold_prediction_speed']['inference_fps']:.2f} |")
    eca = judgments["eca_effects_e7_minus_e8"]
    lines.extend([
        "", "## 结论", "",
        f"1. E8 是否保留 E7 的 10–30 像素收益：**{'是' if judgments['e8_retains_e7_10_30_benefit'] else '否'}**；是否完整保留：**{'是' if judgments['e8_fully_retains_e7_10_30_benefit'] else '否'}**。E7 相对 E6 为 {judgments['e7_gain_over_e6_recall_pp']:+.3f} pp，E8 相对 E6 为 {judgments['e8_gain_over_e6_recall_pp']:+.3f} pp。",
        f"2. E8 是否恢复或改善 ≤10 像素：恢复到 E6 或更好 **{'是' if judgments['e8_restores_tiny_to_e6_or_better'] else '否'}**；相对 E7 改善 **{'是' if judgments['e8_improves_tiny_vs_e7'] else '否'}**。",
        f"3. ECA 是否同时改善误检、类别区分和总体 Precision：**{'是' if judgments['eca_improves_false_positives_class_separation_and_precision'] else '否'}**。E7−E8：Precision {eca['precision_delta_pp_e7_minus_e8']:+.3f} pp、FP {eca['false_positive_delta_e7_minus_e8']:+d}、类别混淆 {eca['class_confusion_delta_e7_minus_e8']:+d}、mAP50-95 {eca['map50_95_delta_pp_e7_minus_e8']:+.3f} pp。",
        f"4. ECA 是否适合当前 P2 浅层增强：**{'是' if judgments['eca_suitable_for_current_p2_refinement'] else '否'}**。",
        f"5. 最终候选：**{judgments['final_candidate']}**（规则：{judgments['final_candidate_rule']}）。",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate only E4/E6/E7/E8 on complete val.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    args = parse_args()
    if args.batch != 2:
        raise ValueError("unified E4/E6/E7/E8 evaluation requires batch=2")
    weights = {"E4": E4_WEIGHT, "E6": E6_WEIGHT, "E7": E7_WEIGHT, "E8": E8_WEIGHT}
    for path in (*weights.values(), DATASET_YAML):
        if not path.is_file():
            raise FileNotFoundError(path.resolve())
    for path in (REPORT_JSON, REPORT_MD, EVALUATION_DIR):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite E8 evaluation output: {path.resolve()}")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {args.device!r} unavailable")
    dataset = yaml.safe_load(DATASET_YAML.read_text(encoding="utf-8"))
    if dataset.get("train") != "images/train" or dataset.get("val") != "images/val":
        raise RuntimeError("dataset split contract changed")
    root = Path(str(dataset["path"])).resolve()
    images_dir, labels_dir = root / "images" / "val", root / "labels" / "val"
    image_paths = sorted(path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(image_paths) != EXPECTED_IMAGES:
        raise RuntimeError(f"val image count mismatch: {len(image_paths)}")
    source = validated_streaming_image_source(images_dir, expected_images=EXPECTED_IMAGES)
    ground_truth: dict[str, list[dict[str, object]]] = {}
    for image_path in image_paths:
        with Image.open(image_path) as image:
            ground_truth[image_path.name] = load_ground_truth_boxes(labels_dir / f"{image_path.stem}.txt", image_size=image.size)
    if sum(len(row) for row in ground_truth.values()) != EXPECTED_GT:
        raise RuntimeError("val GT count mismatch")

    EVALUATION_DIR.mkdir(parents=True)
    register_lfr_module()
    register_no_eca_module()
    hashes_before = {name: sha256(path) for name, path in weights.items()}
    experiments: dict[str, object] = {}
    expected_strides = {"E4": [8.0, 16.0, 32.0], "E6": [4.0, 8.0, 16.0, 32.0], "E7": [4.0, 8.0, 16.0, 32.0], "E8": [4.0, 8.0, 16.0, 32.0]}
    for experiment in EXPERIMENT_ORDER:
        weight = weights[experiment]
        model = YOLO(str(weight.resolve()), task="detect")
        strides = [float(value) for value in model.model.stride.detach().cpu().tolist()]
        if strides != expected_strides[experiment] or dict(model.names) != {0: "helmet", 1: "no_helmet"}:
            raise RuntimeError(f"{experiment} architecture or class contract failed")
        val_result = model.val(
            data=str(DATASET_YAML.resolve()), split="val", imgsz=960, batch=2, workers=args.workers,
            device=args.device, plots=True, seed=42, deterministic=True,
            project=str(EVALUATION_DIR.resolve()), name=f"{experiment.lower()}_standard_val", exist_ok=False,
        )
        standard = format_evaluation_metrics(val_result.results_dict, val_result.summary(normalize=True, decimals=10), val_result.speed)
        add_standard_f1(standard)
        model.validator = None
        del val_result
        torch.cuda.empty_cache()
        records, prediction_speed = collect_records(model, source=source, ground_truth=ground_truth, device=args.device, batch=args.batch)
        summaries = summarize_p0_records(records, matching_iou=MATCHING_IOU, duplicate_iou=0.70)
        confusion = count_class_confusions(records)
        experiments[experiment] = {
            "weight": str(weight.resolve()), "weight_sha256": hashes_before[experiment], "detect_strides": strides,
            "parameters": sum(parameter.numel() for parameter in model.model.parameters()),
            "gflops_at_960": float(get_flops(model.model, imgsz=960)),
            "standard_val": standard, "fixed_threshold": summaries["fixed_threshold_metrics"],
            "tiny": summaries["tiny_metrics"], "tiny_by_scope": tiny_scopes(summaries["tiny_metrics"]),
            "size_buckets": summarize_size_buckets(records, matching_iou=MATCHING_IOU),
            "class_confusions": confusion["count"], "class_confusion_pairs": confusion["pairs"],
            "duplicate_audit": {key: value for key, value in summaries["duplicate_audit"].items() if key != "details"},
            "fixed_threshold_prediction_speed": prediction_speed,
        }
        del records, model
        torch.cuda.empty_cache()
    hashes_after = {name: sha256(path) for name, path in weights.items()}
    if hashes_after != hashes_before:
        raise RuntimeError("an evaluated weight changed during evaluation")
    judgments = build_ablation_judgments(experiments)
    report = {
        "status": "passed", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"ultralytics": ultralytics.__version__, "torch": torch.__version__},
        "protocol": {
            "split": "val", "images": EXPECTED_IMAGES, "ground_truth": EXPECTED_GT, "imgsz": 960,
            "batch": 2, "seed": 42, "deterministic": True, "fixed_confidence": FIXED_CONF,
            "nms_iou": NMS_IOU, "matching_iou": MATCHING_IOU, "class_aware_matching": True,
            "size_measure": "sqrt(original_box_width_px * original_box_height_px)", "size_buckets": list(SIZE_BUCKETS),
            "test_used": False,
        },
        "experiments": experiments, "judgments": judgments,
        "weight_integrity": {"before": hashes_before, "after": hashes_after, "unchanged": True},
    }
    write_json_report(REPORT_JSON, report)
    REPORT_MD.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(judgments, ensure_ascii=False, indent=2), flush=True)
    print(REPORT_MD.read_text(encoding="utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
