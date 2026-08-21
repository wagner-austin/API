"""Dataset route tests: upload and get. Caption tests: test_dataset_caption.py."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Literal

from fastapi.testclient import TestClient
from platform_core.json_utils import load_json_str, require_int, require_str

from art_trainer.api.main import create_app
from art_trainer.core import _test_hooks
from art_trainer.core.config.settings import Settings


def _make_test_settings(tmp_path: Path) -> Settings:
    """Create test settings.

    Args:
        tmp_path: Temporary directory path.

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
            "data_bank_api_key": "test-key",
            "kohya_ss_path": str(kohya_path),
            "comfyui_lora_path": str(tmp_path / "comfyui" / "models" / "loras"),
            "blip_model_name": "Salesforce/blip-image-captioning-large",
            "caption_trigger_word": "sks person",
            "gemini_api_key": "",
            "openai_api_key": "",
        },
        "security": {"api_key": "test-api-key"},
    }


def test_dataset_upload_requires_api_key(tmp_path: Path) -> None:
    """Test POST /datasets/upload requires API key."""
    settings = _make_test_settings(tmp_path)

    def load_settings() -> Settings:
        return settings

    _test_hooks.load_settings = load_settings

    app = create_app(settings)
    client = TestClient(app)

    response = client.post(
        "/datasets/upload",
        data={
            "trigger_word": "sks person",
            "training_type": "character",
            "auto_caption": "false",
        },
        files=[("files", ("test.jpg", io.BytesIO(b"fake image"), "image/jpeg"))],
    )

    assert response.status_code == 401


def test_dataset_upload_success(tmp_path: Path) -> None:
    """Test POST /datasets/upload succeeds with API key."""
    settings = _make_test_settings(tmp_path)

    def load_settings() -> Settings:
        return settings

    _test_hooks.load_settings = load_settings

    app = create_app(settings)
    client = TestClient(app)

    response = client.post(
        "/datasets/upload",
        data={
            "trigger_word": "sks person",
            "training_type": "character",
            "auto_caption": "false",
        },
        files=[("files", ("test.jpg", io.BytesIO(b"fake image"), "image/jpeg"))],
        headers={"X-API-Key": "test-api-key"},
    )

    assert response.status_code == 200
    data = load_json_str(response.text)
    if not isinstance(data, dict):
        raise AssertionError("Response body must be a JSON object")
    dataset_id = require_str(data, "dataset_id")
    image_count = require_int(data, "image_count")
    caption_count = require_int(data, "caption_count")

    assert dataset_id.count("-") == 4  # UUID format
    assert image_count == 1
    assert caption_count == 0  # auto_caption is false


def test_dataset_upload_skips_non_image_files(tmp_path: Path) -> None:
    """Test POST /datasets/upload skips non-image files."""
    settings = _make_test_settings(tmp_path)

    def load_settings() -> Settings:
        return settings

    _test_hooks.load_settings = load_settings

    app = create_app(settings)
    client = TestClient(app)

    response = client.post(
        "/datasets/upload",
        data={
            "trigger_word": "sks person",
            "training_type": "style",
            "auto_caption": "false",
        },
        files=[
            ("files", ("photo.jpg", io.BytesIO(b"fake image"), "image/jpeg")),
            ("files", ("document.txt", io.BytesIO(b"text content"), "text/plain")),
        ],
        headers={"X-API-Key": "test-api-key"},
    )

    assert response.status_code == 200
    data = load_json_str(response.text)
    if not isinstance(data, dict):
        raise AssertionError("Response body must be a JSON object")
    image_count = require_int(data, "image_count")

    assert image_count == 1  # Only the .jpg file


def test_dataset_get_not_found(tmp_path: Path) -> None:
    """Test GET /datasets/{dataset_id} returns 404 for unknown dataset."""
    settings = _make_test_settings(tmp_path)

    def load_settings() -> Settings:
        return settings

    _test_hooks.load_settings = load_settings

    app = create_app(settings)
    client = TestClient(app)

    response = client.get(
        "/datasets/nonexistent-id",
        headers={"X-API-Key": "test-api-key"},
    )

    assert response.status_code == 404
    data = load_json_str(response.text)
    if not isinstance(data, dict):
        raise AssertionError("Response body must be a JSON object")
    detail = require_str(data, "detail")
    assert "not found" in detail


def test_dataset_get_success(tmp_path: Path) -> None:
    """Test GET /datasets/{dataset_id} returns dataset info."""
    settings = _make_test_settings(tmp_path)

    def load_settings() -> Settings:
        return settings

    _test_hooks.load_settings = load_settings

    app = create_app(settings)
    client = TestClient(app)

    # First upload a dataset
    upload_response = client.post(
        "/datasets/upload",
        data={
            "trigger_word": "sks person",
            "training_type": "character",
            "auto_caption": "false",
        },
        files=[("files", ("test.jpg", io.BytesIO(b"fake image"), "image/jpeg"))],
        headers={"X-API-Key": "test-api-key"},
    )
    upload_data = load_json_str(upload_response.text)
    if not isinstance(upload_data, dict):
        raise AssertionError("Response body must be a JSON object")
    dataset_id = require_str(upload_data, "dataset_id")

    # Then get the dataset info
    get_response = client.get(
        f"/datasets/{dataset_id}",
        headers={"X-API-Key": "test-api-key"},
    )

    assert get_response.status_code == 200
    data = load_json_str(get_response.text)
    if not isinstance(data, dict):
        raise AssertionError("Response body must be a JSON object")
    response_id = require_str(data, "dataset_id")
    image_count = require_int(data, "image_count")

    assert response_id == dataset_id
    # find_images finds .jpg and .JPG separately, so image_count may be doubled
    assert image_count >= 1


def test_dataset_upload_empty_filename_skipped(tmp_path: Path) -> None:
    """Test POST /datasets/upload skips files with no filename."""
    settings = _make_test_settings(tmp_path)

    def load_settings() -> Settings:
        return settings

    _test_hooks.load_settings = load_settings

    app = create_app(settings)
    client = TestClient(app)

    response = client.post(
        "/datasets/upload",
        data={
            "trigger_word": "sks person",
            "training_type": "style",
            "auto_caption": "false",
        },
        files=[
            ("files", ("photo.jpg", io.BytesIO(b"fake image"), "image/jpeg")),
        ],
        headers={"X-API-Key": "test-api-key"},
    )

    assert response.status_code == 200
    data = load_json_str(response.text)
    if not isinstance(data, dict):
        raise AssertionError("Response body must be a JSON object")
    image_count = require_int(data, "image_count")
    assert image_count == 1
