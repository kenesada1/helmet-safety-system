#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


# 支持从仓库根目录直接运行脚本，而不要求先执行 pip install。
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from helmet_safety.data.validate import generate_visualizations, validate_dataset  # noqa: E402


def main() -> int:
    """运行独立验证，并按需生成不影响训练图片的可视化副本。"""
    parser = argparse.ArgumentParser(description="Validate a converted SHWD YOLO dataset")
    # 验证需要同时知道原始 XML、转换结果，以及报告要写到哪里。
    parser.add_argument("--raw", type=Path, default=Path(r"D:\datasets\SHWD\VOC2028"))
    parser.add_argument("--processed", type=Path, default=Path(r"D:\datasets\SHWD\processed"))
    parser.add_argument("--output", type=Path, default=Path(r"D:\datasets\SHWD\audit"))
    # 6 位小数归一化会带来极小舍入误差，默认允许最多 0.01 像素。
    parser.add_argument("--pixel-tolerance", type=float, default=0.01)
    # 以下参数只控制可视化，不影响标签验证结果。
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--samples-per-split", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--force-visualizations", action="store_true")
    args = parser.parse_args()
    # 先做硬性验证；即使不传 --visualize，也会生成 validation_report。
    report = validate_dataset(
        args.raw,
        args.processed,
        args.output,
        pixel_tolerance=args.pixel_tolerance,
    )
    # --visualize 是可选步骤；验证本身不依赖可视化。
    if args.visualize:
        visualization_report = generate_visualizations(
            args.processed,
            args.output / "visualizations",
            samples_per_split=args.samples_per_split,
            seed=args.seed,
            force=args.force_visualizations,
        )
        # 抽样列表单独落盘，记录 seed、图片 ID、框数和原图 SHA256。
        (args.output / "visualization_report.json").write_text(
            json.dumps(visualization_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps({"status": report["status"], "issues": len(report["issues"]), "counts": report["counts"]}, ensure_ascii=False, indent=2))
    # 非零退出码可让自动化脚本明确知道数据验证失败。
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
