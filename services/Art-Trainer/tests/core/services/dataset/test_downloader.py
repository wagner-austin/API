"""Tests for dataset downloader service."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Literal

from art_trainer.core.config.settings import Settings
from art_trainer.core.services.dataset import _test_hooks
from art_trainer.core.services.dataset._test_hooks import UploadResult
from art_trainer.core.services.dataset.downloader import (
    dataset_exists,
    download_dataset,
)


def _make_test_settings(tmp_path: Path, api_key: str = "test-key") -> Settings:
    """Create test settings.

    Args:
        tmp_path: Temporary directory path.
        api_key: API key for data-bank (empty string for no auth).

    Returns:
        Test Settings.
    """
    kohya_path = tmp_path / "kohya_ss"
    kohya_path.mkdir(parents=True)
    (kohya_path / "train_network.py").touch()

    app_env: Literal["dev", "prod"] = "dev"

    return {
        "app_env": app_env,
        "logging": {"level": "INFO"},
        "redis": {"enabled": True, "url": "redis://localhost:6379/0"},
        "rq": {
            "queue_name": "art-trainer",
            "job_timeout_sec": 86400,
            "result_ttl_sec": 86400,
            "failure_ttl_sec": 604800,
            "retry_max": 1,
            "retry_intervals_sec": "300",
        },
        "app": {
            "data_root": str(tmp_path / "data"),
            "output_root": str(tmp_path / "output"),
            "logs_root": str(tmp_path / "logs"),
            "data_bank_api_url": "http://localhost:8000",
            "data_bank_api_key": api_key,
            "kohya_ss_path": str(kohya_path),
            "comfyui_lora_path": str(tmp_path / "comfyui" / "models" / "loras"),
            "blip_model_name": "Salesforce/blip-image-captioning-large",
            "caption_trigger_word": "sks person",
            "gemini_api_key": "",
            "openai_api_key": "",
        },
        "security": {"api_key": "test-api-key"},
    }


def _create_fake_dataset_zip() -> bytes:
    """Create a fake dataset ZIP with a dummy image and caption.

    Returns:
        ZIP file bytes.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add a dummy image (1x1 pixel PNG)
        png_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
            b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
            b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        zf.writestr("image001.png", png_bytes)
        zf.writestr("image001.txt", "test caption")
    return buffer.getvalue()


def test_download_dataset_with_api_key(tmp_path: Path) -> None:
    """Test download_dataset includes API key in headers."""
    settings = _make_test_settings(tmp_path, api_key="my-secret-key")
    zip_bytes = _create_fake_dataset_zip()

    captured_headers: list[dict[str, str]] = []

    def fake_http_get(url: str, headers: dict[str, str]) -> bytes:
        captured_headers.append(headers)
        return zip_bytes

    def fake_http_upload(
        url: str,
        headers: dict[str, str],
        filename: str,
        content: bytes,
    ) -> UploadResult:
        return {"file_id": "fake", "filename": filename}

    _test_hooks.http_get = fake_http_get
    _test_hooks.http_upload = fake_http_upload

    download_dataset(settings, "file-123", "dataset-456")

    assert len(captured_headers) == 1
    assert captured_headers[0]["X-API-Key"] == "my-secret-key"


def test_download_dataset_without_api_key(tmp_path: Path) -> None:
    """Test download_dataset works without API key."""
    settings = _make_test_settings(tmp_path, api_key="")
    zip_bytes = _create_fake_dataset_zip()

    captured_headers: list[dict[str, str]] = []

    def fake_http_get(url: str, headers: dict[str, str]) -> bytes:
        captured_headers.append(headers)
        return zip_bytes

    def fake_http_upload(
        url: str,
        headers: dict[str, str],
        filename: str,
        content: bytes,
    ) -> UploadResult:
        return {"file_id": "fake", "filename": filename}

    _test_hooks.http_get = fake_http_get
    _test_hooks.http_upload = fake_http_upload

    download_dataset(settings, "file-123", "dataset-456")

    assert len(captured_headers) == 1
    assert "X-API-Key" not in captured_headers[0]


def test_dataset_exists_returns_true_when_has_files(tmp_path: Path) -> None:
    """Test dataset_exists returns True when directory has files."""
    settings = _make_test_settings(tmp_path)

    # Create dataset directory with a file
    data_root = tmp_path / "data"
    dataset_path = data_root / "datasets" / "test-dataset"
    dataset_path.mkdir(parents=True)
    (dataset_path / "image.png").touch()

    result = dataset_exists(settings, "test-dataset")

    assert result is True


def test_dataset_exists_returns_false_when_not_exists(tmp_path: Path) -> None:
    """Test dataset_exists returns False when directory doesn't exist."""
    settings = _make_test_settings(tmp_path)

    result = dataset_exists(settings, "nonexistent-dataset")

    assert result is False


def test_dataset_exists_returns_false_when_empty(tmp_path: Path) -> None:
    """Test dataset_exists returns False when directory is empty."""
    settings = _make_test_settings(tmp_path)

    # Create empty dataset directory
    data_root = tmp_path / "data"
    dataset_path = data_root / "datasets" / "empty-dataset"
    dataset_path.mkdir(parents=True)

    result = dataset_exists(settings, "empty-dataset")

    assert result is False
