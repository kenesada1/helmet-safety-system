#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


# 把 src 临时加入模块搜索路径，使脚本可以直接从项目目录运行。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from helmet_safety.data.convert import convert_dataset  # noqa: E402


def main() -> int:
    """读取路径、安全覆盖开关和小数精度，然后启动完整转换。"""
    parser = argparse.ArgumentParser(description="Safely convert SHWD Pascal VOC annotations to YOLO")
    # --raw 默认指向只读原始数据；--output 默认指向独立 processed 目录。
    parser.add_argument("--raw", type=Path, default=Path(r"D:\datasets\SHWD\VOC2028"))
    parser.add_argument("--output", type=Path, default=Path(r"D:\datasets\SHWD\processed"))
    # store_true 表示用户不写 --force 时为 False，显式写出时才为 True。
    parser.add_argument("--force", action="store_true", help="overwrite only files tracked by this tool's manifest")
    # precision 控制 YOLO 四个浮点坐标保留几位小数，默认统一为 6 位。
    parser.add_argument("--precision", type=int, default=6)
    args = parser.parse_args()
    # 真正的业务逻辑全部在 data/convert.py；脚本只负责参数和终端输出。
    report = convert_dataset(args.raw, args.output, force=args.force, precision=args.precision)
    # ensure_ascii=False 让中文原样显示，indent=2 让 JSON 在终端中易读。
    print(json.dumps(report["splits"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
