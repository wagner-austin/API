from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.data_bank_protocol import FileUploadResponse
from platform_core.json_utils import JSONTypeError

from handwriting_ai import _test_hooks
from handwriting_ai._test_hooks import ArtifactStoreProtocol
from handwriting_ai.config import Settings
from handwriting_ai.inference.engine import InferenceEngine


def _settings(tmp: Path) -> Settings:
    return {
        "app": {
            "data_root": tmp,
            "artifacts_root": tmp,
            "logs_root": tmp,
            "threads": 0,
            "port": 8080,
            "data_bank_api_url": "http://test-db",
            "data_bank_api_key": "test-key",
        },
        "digits": {
            "model_dir": tmp / "models",
            "active_model": "m",
            "tta": False,
            "uncertain_threshold": 0.5,
            "max_image_mb": 1,
            "max_image_side_px": 64,
            "predict_timeout_seconds": 1,
            "visualize_max_kb": 16,
            "retention_keep_runs": 1,
        },
        "security": {"api_key": "", "api_key_enabled": False},
    }


def test_download_remote_manifest_read_error(tmp_path: Path) -> None:
    """Cover engine.py:171-173 - OSError when reading manifest."""
    s = _settings(tmp_path)
    model_dir = s["digits"]["model_dir"] / "m"
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = model_dir / "manifest.json"
    # Create manifest but make it unreadable by writing then removing read perms
    manifest_path.write_text('{"schema_version":"v2.0"}', encoding="utf-8")

    engine = InferenceEngine(s)
    # Simulate read error by removing the file right before download
    manifest_path.unlink()
    manifest_path.mkdir()  # Replace with directory to cause read error

    with pytest.raises(OSError):
        engine._download_remote_if_needed(model_dir, manifest_path)


def test_download_remote_manifest_not_object(tmp_path: Path) -> None:
    """Cover engine.py:176 - JSONTypeError when manifest is not a JSON object."""
    s = _settings(tmp_path)
    model_dir = s["digits"]["model_dir"] / "m"
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = model_dir / "manifest.json"
    manifest_path.write_text('"just a string"', encoding="utf-8")

    engine = InferenceEngine(s)
    with pytest.raises(JSONTypeError, match="manifest must be a JSON object"):
        engine._download_remote_if_needed(model_dir, manifest_path)


def test_download_remote_v2_missing_file_id(tmp_path: Path) -> None:
    """Cover engine.py:182 - v2 manifest missing file_id."""
    s = _settings(tmp_path)
    model_dir = s["digits"]["model_dir"] / "m"
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = model_dir / "manifest.json"
    manifest_path.write_text('{"schema_version":"v2.0"}', encoding="utf-8")

    engine = InferenceEngine(s)
    with pytest.raises(RuntimeError, match="v2 manifest missing file_id"):
        engine._download_remote_if_needed(model_dir, manifest_path)


def test_download_remote_missing_data_bank_config(tmp_path: Path) -> None:
    """Cover engine.py:188 - missing data-bank-api configuration."""
    s = _settings(tmp_path)
    # Clear the data bank config
    s["app"]["data_bank_api_url"] = ""
    s["app"]["data_bank_api_key"] = ""

    model_dir = s["digits"]["model_dir"] / "m"
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = model_dir / "manifest.json"
    manifest_path.write_text('{"schema_version":"v2.0","file_id":"abc123"}', encoding="utf-8")

    engine = InferenceEngine(s)
    with pytest.raises(RuntimeError, match="missing data-bank-api configuration"):
        engine._download_remote_if_needed(model_dir, manifest_path)


def test_download_remote_success(tmp_path: Path) -> None:
    """Cover engine.py:189-196 - successful remote download path."""
    s = _settings(tmp_path)
    model_dir = s["digits"]["model_dir"] / "m"
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = model_dir / "manifest.json"
    manifest_path.write_text('{"schema_version":"v2.0","file_id":"abc123"}', encoding="utf-8")

    class _FakeStore:
        def __init__(self) -> None:
            pass

        def upload_artifact(
            self,
            dir_path: Path,
            *,
            artifact_name: str,
            request_id: str,
        ) -> FileUploadResponse:
            return {
                "file_id": "fake",
                "size": 0,
                "sha256": "x",
                "content_type": "application/gzip",
                "created_at": None,
            }

        def download_artifact(
            self,
            file_id: str,
            *,
            dest_dir: Path,
            request_id: str,
            expected_root: str,
        ) -> Path:
            # Simulate extracting files
            target = dest_dir / expected_root
            target.mkdir(parents=True, exist_ok=True)
            (target / "model.pt").write_bytes(b"model")
            return target

    store: ArtifactStoreProtocol = _FakeStore()

    def _factory(api_url: str, api_key: str) -> ArtifactStoreProtocol:
        return store

    _test_hooks.artifact_store_factory = _factory

    engine = InferenceEngine(s)
    engine._download_remote_if_needed(model_dir, manifest_path)
    # Verify download was called (model.pt should now exist)
    assert (model_dir / "model.pt").exists()


def test_ensure_artifacts_with_manifest_only_triggers_download(tmp_path: Path) -> None:
    """Cover engine.py:99 - model_dir doesn't exist but manifest does triggers download."""
    s = _settings(tmp_path)
    model_dir = s["digits"]["model_dir"] / "m"

    # Create model_dir and manifest, but not model.pt
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = model_dir / "manifest.json"
    manifest_path.write_text('{"schema_version":"v1"}', encoding="utf-8")
    # model.pt doesn't exist

    download_called = {"count": 0}

    def _fake_download(model_dir: Path, manifest_path: Path) -> None:
        download_called["count"] += 1

    _test_hooks.download_remote_override = _fake_download

    engine = InferenceEngine(s)
    result = engine._ensure_artifacts_present(model_dir)
    assert result is None  # model.pt still doesn't exist
    assert download_called["count"] == 1


def test_ensure_artifacts_download_creates_model(tmp_path: Path) -> None:
    """Cover engine.py:105->108 - after download, files exist and return paths."""
    s = _settings(tmp_path)
    model_dir = s["digits"]["model_dir"] / "m"
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = model_dir / "manifest.json"
    manifest_path.write_text('{"schema_version":"v1"}', encoding="utf-8")
    # model.pt doesn't exist initially

    def _fake_download(model_dir: Path, manifest_path: Path) -> None:
        # Simulate download creating model.pt
        _ = manifest_path  # Unused
        (model_dir / "model.pt").write_bytes(b"model")

    _test_hooks.download_remote_override = _fake_download

    engine = InferenceEngine(s)
    result = engine._ensure_artifacts_present(model_dir)
    if result is None:
        raise AssertionError("expected artifacts")
    assert result[0] == manifest_path
    assert result[1] == model_dir / "model.pt"
