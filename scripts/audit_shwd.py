#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


# 允许直接运行 ``python scripts/audit_shwd.py``，无需先把项目安装到 Python 环境。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from helmet_safety.data.audit import audit_dataset  # noqa: E402


def main() -> int:
    """解析命令行参数，调用核心审计函数，并打印一份简短摘要。"""
    parser = argparse.ArgumentParser(description="Read-only audit of a SHWD Pascal VOC dataset")
    # type=Path 会把命令行字符串自动转换成 pathlib.Path，后续无需手工拼路径。
    parser.add_argument("--raw", type=Path, default=Path(r"D:\datasets\SHWD\VOC2028"))
    parser.add_argument("--output", type=Path, default=Path(r"D:\datasets\SHWD\audit"))
    args = parser.parse_args()
    # audit_dataset 返回的字典与 audit_report.json 内容一致。
    report = audit_dataset(args.raw, args.output)
    # 终端只打印最关键数字；完整问题明细留在 JSON/Markdown 报告中。
    summary = {
        "jpg": report["files"]["jpg_count"],
        "xml": report["files"]["xml_count"],
        "standard_parsed": report["xml_parsing"]["standard_success_count"],
        "repaired": report["xml_parsing"]["repaired_success_count"],
        "failed": report["xml_parsing"]["final_failure_count"],
        "unknown": report["classes"]["unknown"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    # 退出码 0 表示成功；1 方便 CI/批处理发现无法解析 XML 或未知类别。
    return 0 if not report["xml_parsing"]["failed_files"] and not report["unknown_classes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
