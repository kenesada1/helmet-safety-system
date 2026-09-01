#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from helmet_safety.quality.dependencies import validate_dependency_policy  # noqa: E402


def main() -> int:
    violations = validate_dependency_policy(ROOT / "pyproject.toml")
    if violations:
        for violation in violations:
            print(f"dependency policy violation: {violation}", file=sys.stderr)
        return 1
    print("dependency policy: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

