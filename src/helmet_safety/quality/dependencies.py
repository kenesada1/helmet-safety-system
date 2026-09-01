"""Dependency policy consumed by both tests and CI."""

from __future__ import annotations

from pathlib import Path
import tomllib


ULTRALYTICS_PIN = "ultralytics==8.4.120"
REQUIRED_GROUPS = ("train", "inference", "deployment", "service")


def _package_name(requirement: str) -> str:
    return requirement.split("[", 1)[0].split("=", 1)[0].split(">", 1)[0].split("<", 1)[0].strip().lower()


def validate_dependency_policy(pyproject_path: Path | str) -> list[str]:
    path = Path(pyproject_path).expanduser().resolve()
    with path.open("rb") as stream:
        payload = tomllib.load(stream)
    project = payload.get("project", {})
    optional = project.get("optional-dependencies", {})
    violations: list[str] = []
    for group in REQUIRED_GROUPS:
        if group not in optional:
            violations.append(f"missing optional dependency group: {group}")
    for group in ("train", "inference", "deployment"):
        requirements = optional.get(group, [])
        if ULTRALYTICS_PIN not in requirements:
            violations.append(f"{group} must pin {ULTRALYTICS_PIN}")
    service_names = {_package_name(item) for item in optional.get("service", [])}
    for package in ("fastapi", "uvicorn", "python-multipart"):
        if package not in service_names:
            violations.append(f"service must include {package}")
    description = str(project.get("description", "")).lower()
    if "helmet" not in description or "detection" not in description:
        violations.append("project description must identify the helmet detection system")
    return violations

