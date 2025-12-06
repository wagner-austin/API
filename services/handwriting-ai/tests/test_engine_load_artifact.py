from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import torch
from platform_core.json_utils import dump_json_str

from handwriting_ai.config import (
    AppConfig,
    DigitsConfig,
    SecurityConfig,
    Settings,
)
from handwriting_ai.inference.engine import InferenceEngine, build_fresh_state_dict
from handwriting_ai.preprocess import preprocess_signature


def _settings(tmp: Path, model_id: str) -> Settings:
    app: AppConfig = {
        "data_root": Path("/tmp/data"),
        "artifacts_root": Path("/tmp/artifacts"),
        "logs_root": Path("/tmp/logs"),
        "threads": 0,
        "port": 8081,
    }
    digits: DigitsConfig = {
        "model_dir": tmp,
        "active_model": model_id,
        "tta": False,
        "uncertain_threshold": 0.70,
        "max_image_mb": 2,
        "max_image_side_px": 1024,
        "predict_timeout_seconds": 5,
        "visualize_max_kb": 16,
        "retention_keep_runs": 3,
    }
    sec: SecurityConfig = {"api_key": ""}
    return {"app": app, "digits": digits, "security": sec}


def _write_test_artifacts(
    out_dir: Path,
    model_id: str,
    state_dict: dict[str, torch.Tensor],
    val_acc: float = 0.0,
) -> Path:
    """Write test artifacts (model.pt and manifest.json) for engine loading tests.

    This is a test-only helper that creates v1.1 manifest format for testing.
    Production code uses v2 manifests via platform_ml.ArtifactStore.
    """
    model_dir = out_dir / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    # Write model state dict
    model_path = model_dir / "model.pt"
    torch.save(state_dict, model_path.as_posix())

    # Write v1.1 manifest (for testing without data-bank-api dependency)
    manifest = {
        "schema_version": "v1.1",
        "model_id": model_id,
        "arch": "resnet18",
        "n_classes": 10,
        "version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "preprocess_hash": preprocess_signature(),
        "val_acc": float(val_acc),
        "temperature": 1.0,
    }
    manifest_path = model_dir / "manifest.json"
    manifest_path.write_text(dump_json_str(manifest, compact=False), encoding="utf-8")

    return model_dir


def test_engine_try_load_active_success(tmp_path: Path) -> None:
    model_id = "mnist_resnet18_v1"
    # Build a fresh state dict compatible with our model arch
    sd = build_fresh_state_dict(arch="resnet18", n_classes=10)
    # Write test artifacts to tmp model dir
    out = _write_test_artifacts(
        out_dir=tmp_path,
        model_id=model_id,
        state_dict=sd,
        val_acc=0.0,
    )
    assert (out / "model.pt").exists() and (out / "manifest.json").exists()
    s = _settings(tmp_path, model_id)
    eng = InferenceEngine(s)
    # Initially not ready
    assert eng.ready is False
    eng.try_load_active()
    assert eng.ready is True


def test_engine_try_load_active_bad_model_file(tmp_path: Path) -> None:
    model_id = "mnist_resnet18_v1"
    dest = tmp_path / model_id
    dest.mkdir(parents=True, exist_ok=True)
    # Write an invalid model.pt and a minimal manifest with required fields
    (dest / "model.pt").write_bytes(b"not a torch file")
    from handwriting_ai.preprocess import preprocess_signature

    manifest = (
        "{"  # minimal valid manifest v1.1
        '"schema_version":"v1.1",'
        f'"model_id":"{model_id}",'
        '"arch":"resnet18",'
        '"n_classes":10,'
        '"version":"1.0.0",'
        '"created_at":"2024-01-01T00:00:00+00:00",'
        f'"preprocess_hash":"{preprocess_signature()}",'
        '"val_acc":0.0,'
        '"temperature":1.0'
        "}"
    )
    (dest / "manifest.json").write_text(manifest, encoding="utf-8")
    s = _settings(tmp_path, model_id)
    eng = InferenceEngine(s)
    # Should raise after logging due to bad model file
    import pickle

    import pytest

    with pytest.raises((OSError, ValueError, RuntimeError, pickle.UnpicklingError)):
        eng.try_load_active()
    # Should not be ready due to bad model file
    assert eng.ready is False
