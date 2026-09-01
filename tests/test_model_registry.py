from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


def _registry() -> object:
    try:
        from helmet_safety.service import registry as module
    except ModuleNotFoundError:
        pytest.fail("helmet_safety.service.registry must verify versioned model artifacts")
    return module


def _write_registry(path: Path, artifact: Path, sha256: str) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "repository_root": ".",
                "models": {
                    "e4-onnx": {
                        "experiment": "E4",
                        "stage": "production",
                        "backend": "onnx",
                        "artifact": artifact.name,
                        "sha256": sha256,
                        "report": "report.json",
                        "classes": {"0": "helmet", "1": "no_helmet"},
                        "imgsz": 960,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_registry_resolves_experiment_artifact_report_and_verifies_digest(
    tmp_path: Path,
) -> None:
    registry = _registry()
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"versioned-model")
    (tmp_path / "report.json").write_text("{}", encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = tmp_path / "registry.json"
    _write_registry(manifest, artifact, digest.upper())

    resolved = registry.load_model_artifact(manifest, "e4-onnx", verify=True)

    assert resolved.model_id == "e4-onnx"
    assert resolved.experiment == "E4"
    assert resolved.artifact_path == artifact.resolve()
    assert resolved.report_path == (tmp_path / "report.json").resolve()
    assert resolved.sha256 == digest
    assert resolved.class_names == {0: "helmet", 1: "no_helmet"}


def test_registry_fails_closed_for_unknown_model_or_checksum_mismatch(
    tmp_path: Path,
) -> None:
    registry = _registry()
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"versioned-model")
    (tmp_path / "report.json").write_text("{}", encoding="utf-8")
    manifest = tmp_path / "registry.json"
    _write_registry(manifest, artifact, "0" * 64)

    with pytest.raises(KeyError, match="unknown model"):
        registry.load_model_artifact(manifest, "missing", verify=True)
    with pytest.raises(ValueError, match="SHA256"):
        registry.load_model_artifact(manifest, "e4-onnx", verify=True)


def test_registry_rejects_incomplete_or_unsafe_manifest_entries(tmp_path: Path) -> None:
    registry = _registry()
    outside = tmp_path.parent / "outside.onnx"
    manifest = tmp_path / "registry.json"
    _write_registry(manifest, outside, "0" * 64)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["models"]["e4-onnx"]["artifact"] = "../outside.onnx"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="repository root"):
        registry.load_model_artifact(manifest, "e4-onnx", verify=False)

