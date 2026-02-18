"""Tests for LoRA uploader service."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from art_trainer.core.config.settings import Settings
from art_trainer.core.services.dataset import _test_hooks
from art_trainer.core.services.dataset._test_hooks import UploadResult
from art_trainer.core.services.dataset.uploader import upload_lora


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


def test_upload_lora_with_api_key(tmp_path: Path) -> None:
    """Test upload_lora includes API key in headers."""
    settings = _make_test_settings(tmp_path, api_key="my-secret-key")

    # Create a fake LoRA file
    lora_path = tmp_path / "test.safetensors"
    lora_path.write_bytes(b"fake lora content")

    captured_headers: list[dict[str, str]] = []

    def fake_http_get(url: str, headers: dict[str, str]) -> bytes:
        return b"not used"

    def fake_http_upload(
        url: str,
        headers: dict[str, str],
        filename: str,
        content: bytes,
    ) -> UploadResult:
        captured_headers.append(headers)
        return {"file_id": "uploaded-123", "filename": filename}

    _test_hooks.http_get = fake_http_get
    _test_hooks.http_upload = fake_http_upload

    result = upload_lora(settings, lora_path)

    assert result["file_id"] == "uploaded-123"
    assert len(captured_headers) == 1
    assert captured_headers[0]["X-API-Key"] == "my-secret-key"


def test_upload_lora_without_api_key(tmp_path: Path) -> None:
    """Test upload_lora works without API key."""
    settings = _make_test_settings(tmp_path, api_key="")

    # Create a fake LoRA file
    lora_path = tmp_path / "test.safetensors"
    lora_path.write_bytes(b"fake lora content")

    captured_headers: list[dict[str, str]] = []

    def fake_http_get(url: str, headers: dict[str, str]) -> bytes:
        return b"not used"

    def fake_http_upload(
        url: str,
        headers: dict[str, str],
        filename: str,
        content: bytes,
    ) -> UploadResult:
        captured_headers.append(headers)
        return {"file_id": "uploaded-456", "filename": filename}

    _test_hooks.http_get = fake_http_get
    _test_hooks.http_upload = fake_http_upload

    result = upload_lora(settings, lora_path)

    assert result["file_id"] == "uploaded-456"
    assert len(captured_headers) == 1
    assert "X-API-Key" not in captured_headers[0]


def test_upload_lora_file_not_found(tmp_path: Path) -> None:
    """Test upload_lora raises FileNotFoundError for missing file."""
    settings = _make_test_settings(tmp_path)
    nonexistent_path = tmp_path / "nonexistent.safetensors"

    with pytest.raises(FileNotFoundError) as exc_info:
        upload_lora(settings, nonexistent_path)

    assert "nonexistent.safetensors" in str(exc_info.value)
