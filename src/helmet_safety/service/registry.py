"""Fail-closed model registry for experiment, weight, and report provenance."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping


EXPECTED_CLASSES = {0: "helmet", 1: "no_helmet"}


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    model_id: str
    experiment: str
    stage: str
    backend: str
    artifact_path: Path
    report_path: Path
    sha256: str
    class_names: dict[int, str]
    imgsz: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside_root(root: Path, raw_path: object, *, field: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"model registry field {field!r} must be a non-empty path")
    candidate = (root / raw_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"model registry {field} must stay within repository root") from exc
    return candidate


def load_model_artifact(
    manifest_path: Path | str, model_id: str, *, verify: bool = True
) -> ModelArtifact:
    """Resolve one immutable model entry and optionally verify all durable evidence."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"model registry does not exist: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid model registry: {manifest_path}") from exc
    if payload.get("schema_version") != 1:
        raise ValueError("model registry schema_version must be 1")
    repository_value = payload.get("repository_root", ".")
    if not isinstance(repository_value, str):
        raise ValueError("model registry repository_root must be a path string")
    repository_root = (manifest_path.parent / repository_value).resolve()
    if not repository_root.is_dir():
        raise FileNotFoundError(f"model registry repository root does not exist: {repository_root}")
    models = payload.get("models")
    if not isinstance(models, Mapping) or model_id not in models:
        raise KeyError(f"unknown model {model_id!r} in {manifest_path}")
    entry = models[model_id]
    if not isinstance(entry, Mapping):
        raise ValueError(f"model registry entry {model_id!r} must be an object")

    artifact_path = _inside_root(repository_root, entry.get("artifact"), field="artifact")
    report_path = _inside_root(repository_root, entry.get("report"), field="report")
    digest = str(entry.get("sha256", "")).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"model registry entry {model_id!r} has an invalid SHA256")
    try:
        classes = {int(key): str(value) for key, value in dict(entry.get("classes", {})).items()}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"model registry entry {model_id!r} has invalid classes") from exc
    if classes != EXPECTED_CLASSES:
        raise ValueError(
            f"model registry classes must be {EXPECTED_CLASSES}, received {classes}"
        )
    experiment = str(entry.get("experiment", "")).strip()
    stage = str(entry.get("stage", "")).strip()
    backend = str(entry.get("backend", "")).strip().lower()
    imgsz = entry.get("imgsz")
    if not experiment or not stage or backend not in {"onnx", "pytorch"}:
        raise ValueError(f"model registry entry {model_id!r} has incomplete provenance")
    if not isinstance(imgsz, int) or imgsz < 1:
        raise ValueError(f"model registry entry {model_id!r} has invalid imgsz")
    if verify:
        if not artifact_path.is_file():
            raise FileNotFoundError(f"registered model artifact does not exist: {artifact_path}")
        if not report_path.is_file():
            raise FileNotFoundError(f"registered model report does not exist: {report_path}")
        actual_digest = _sha256(artifact_path)
        if actual_digest != digest:
            raise ValueError(
                f"model artifact SHA256 mismatch for {model_id!r}: "
                f"expected {digest}, received {actual_digest}"
            )
    return ModelArtifact(
        model_id=model_id,
        experiment=experiment,
        stage=stage,
        backend=backend,
        artifact_path=artifact_path,
        report_path=report_path,
        sha256=digest,
        class_names=classes,
        imgsz=imgsz,
    )

