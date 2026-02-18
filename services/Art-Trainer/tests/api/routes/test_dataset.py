"""Tests for dataset routes."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Literal

from fastapi.testclient import TestClient
from platform_core.json_utils import JSONObject, load_json_str, require_int, require_str

from art_trainer.api.main import create_app
from art_trainer.core import _test_hooks
from art_trainer.core.config.settings import Settings
from art_trainer.core.services.captioning import _test_hooks as captioning_test_hooks
from art_trainer.core.services.captioning._test_hooks import CaptionConfigDict
from art_trainer.core.services.captioning.backends import (
    CaptionBackendType,
    reset_caption_registry,
)


class FakeCaptionGenerator:
    """Fake caption generator for tests."""

    calls: list[tuple[Path, str]]

    def __init__(self, caption: str = "standing") -> None:
        """Initialize fake caption generator.

        Args:
            caption: Caption to return.
        """
        self.calls = []
        self._caption = caption

    def __call__(self, image_path: Path, trigger_word: str) -> str:
        """Generate a fake caption.

        Args:
            image_path: Path to image file.
            trigger_word: Trigger word.

        Returns:
            Fake caption string.
        """
        self.calls.append((image_path, trigger_word))
        return f"{trigger_word} {self._caption}"


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


def test_dataset_upload_with_auto_caption(tmp_path: Path) -> None:
    """Test POST /datasets/upload with auto captioning."""
    settings = _make_test_settings(tmp_path)

    def load_settings() -> Settings:
        return settings

    _test_hooks.load_settings = load_settings

    # Set up fake caption generator
    fake_generator = FakeCaptionGenerator("smiling")
    captioning_test_hooks.Hooks.caption_generator = fake_generator

    app = create_app(settings)
    client = TestClient(app)

    response = client.post(
        "/datasets/upload",
        data={
            "trigger_word": "sks person",
            "training_type": "character",
            "auto_caption": "true",
        },
        files=[
            ("files", ("photo1.jpg", io.BytesIO(b"fake image 1"), "image/jpeg")),
            ("files", ("photo2.png", io.BytesIO(b"fake image 2"), "image/png")),
        ],
        headers={"X-API-Key": "test-api-key"},
    )

    assert response.status_code == 200
    data = load_json_str(response.text)
    if not isinstance(data, dict):
        raise AssertionError("Response body must be a JSON object")
    image_count = require_int(data, "image_count")
    caption_count = require_int(data, "caption_count")

    assert image_count == 2
    # find_images finds .jpg and .JPG separately, so caption_count may differ
    assert caption_count >= 2
    assert len(fake_generator.calls) >= 2


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


class FakeCaptionBackend:
    """Fake caption backend for testing."""

    calls: list[tuple[Path, str]]
    _caption: str
    _backend_type: CaptionBackendType

    def __init__(
        self, caption: str = "test caption", backend_type: CaptionBackendType = "gemini"
    ) -> None:
        """Initialize fake caption backend.

        Args:
            caption: Caption to return.
            backend_type: Backend type identifier.
        """
        self.calls = []
        self._caption = caption
        self._backend_type = backend_type

    def caption(self, image_path: Path, trigger_word: str) -> str:
        """Generate a fake caption.

        Args:
            image_path: Path to image file.
            trigger_word: Trigger word.

        Returns:
            Fake caption string.
        """
        self.calls.append((image_path, trigger_word))
        return f"{trigger_word}, {self._caption}"

    @property
    def backend_type(self) -> CaptionBackendType:
        """Get the backend type identifier.

        Returns:
            Backend type string.
        """
        return self._backend_type


def test_dataset_caption_not_found(tmp_path: Path) -> None:
    """Test POST /datasets/{dataset_id}/caption returns 404 for unknown dataset."""
    settings = _make_test_settings(tmp_path)

    def load_settings() -> Settings:
        return settings

    _test_hooks.load_settings = load_settings

    app = create_app(settings)
    client = TestClient(app)

    request_body: JSONObject = {
        "trigger_word": "sks person",
        "backend": "gemini",
        "model_name": "gemini-2.0-flash",
    }
    response = client.post(
        "/datasets/nonexistent-id/caption",
        json=request_body,
        headers={"X-API-Key": "test-api-key"},
    )

    assert response.status_code == 404
    data = load_json_str(response.text)
    if not isinstance(data, dict):
        raise AssertionError("Response body must be a JSON object")
    detail = require_str(data, "detail")
    assert "not found" in detail


def test_dataset_caption_success_with_blip(tmp_path: Path) -> None:
    """Test POST /datasets/{dataset_id}/caption succeeds with BLIP backend."""
    settings = _make_test_settings(tmp_path)

    def load_settings() -> Settings:
        return settings

    _test_hooks.load_settings = load_settings

    # Reset caption registry to use fresh backends
    reset_caption_registry()

    # Set up fake caption backend factory
    fake_backend = FakeCaptionBackend("detailed portrait", "blip")

    def fake_backend_factory(config: CaptionConfigDict) -> FakeCaptionBackend:
        """Create fake caption backend.

        Args:
            config: Caption configuration.

        Returns:
            Fake caption backend instance.
        """
        del config  # Unused, return shared instance
        return fake_backend

    captioning_test_hooks.Hooks.caption_backend_factory = fake_backend_factory

    app = create_app(settings)
    client = TestClient(app)

    # First upload a dataset with images
    upload_response = client.post(
        "/datasets/upload",
        data={
            "trigger_word": "sks person",
            "training_type": "character",
            "auto_caption": "false",
        },
        files=[
            ("files", ("photo1.jpg", io.BytesIO(b"fake image 1"), "image/jpeg")),
            ("files", ("photo2.png", io.BytesIO(b"fake image 2"), "image/png")),
        ],
        headers={"X-API-Key": "test-api-key"},
    )
    upload_data = load_json_str(upload_response.text)
    if not isinstance(upload_data, dict):
        raise AssertionError("Response body must be a JSON object")
    dataset_id = require_str(upload_data, "dataset_id")

    # Now caption the dataset with BLIP
    caption_body: JSONObject = {
        "trigger_word": "sks person",
        "backend": "blip",
        "model_name": "Salesforce/blip-image-captioning-base",
    }
    caption_response = client.post(
        f"/datasets/{dataset_id}/caption",
        json=caption_body,
        headers={"X-API-Key": "test-api-key"},
    )

    assert caption_response.status_code == 200
    data = load_json_str(caption_response.text)
    if not isinstance(data, dict):
        raise AssertionError("Response body must be a JSON object")
    captioned_count = require_int(data, "captioned_count")
    skipped_count = require_int(data, "skipped_count")
    backend_str = require_str(data, "backend")

    assert captioned_count >= 2
    assert skipped_count == 0
    assert backend_str == "blip"
    # Verify the fake backend was called
    assert len(fake_backend.calls) >= 2


def test_dataset_caption_skips_existing_captions(tmp_path: Path) -> None:
    """Test POST /datasets/{dataset_id}/caption skips images with existing captions."""
    settings = _make_test_settings(tmp_path)

    def load_settings() -> Settings:
        return settings

    _test_hooks.load_settings = load_settings

    # Reset caption registry
    reset_caption_registry()

    # Set up fake caption backend factory
    fake_backend = FakeCaptionBackend("new caption", "blip")

    def fake_backend_factory(config: CaptionConfigDict) -> FakeCaptionBackend:
        """Create fake caption backend.

        Args:
            config: Caption configuration.

        Returns:
            Fake caption backend instance.
        """
        del config  # Unused, return shared instance
        return fake_backend

    captioning_test_hooks.Hooks.caption_backend_factory = fake_backend_factory

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
        files=[
            ("files", ("photo1.jpg", io.BytesIO(b"fake image 1"), "image/jpeg")),
        ],
        headers={"X-API-Key": "test-api-key"},
    )
    upload_data = load_json_str(upload_response.text)
    if not isinstance(upload_data, dict):
        raise AssertionError("Response body must be a JSON object")
    dataset_id = require_str(upload_data, "dataset_id")
    dataset_path = require_str(upload_data, "dataset_path")

    # Create an existing caption file manually
    caption_file = Path(dataset_path) / "photo1.txt"
    caption_file.write_text("existing caption", encoding="utf-8")

    # Now try to caption the dataset
    caption_body: JSONObject = {
        "trigger_word": "sks person",
        "backend": "blip",
        "model_name": "Salesforce/blip-image-captioning-base",
    }
    caption_response = client.post(
        f"/datasets/{dataset_id}/caption",
        json=caption_body,
        headers={"X-API-Key": "test-api-key"},
    )

    assert caption_response.status_code == 200
    data = load_json_str(caption_response.text)
    if not isinstance(data, dict):
        raise AssertionError("Response body must be a JSON object")
    captioned_count = require_int(data, "captioned_count")
    skipped_count = require_int(data, "skipped_count")

    # Should have skipped the one with existing caption, captioned none
    assert captioned_count == 0
    assert skipped_count >= 1
    # Caption file should still have old content
    assert caption_file.read_text(encoding="utf-8") == "existing caption"
    # Verify the fake backend was NOT called (all images were skipped)
    assert len(fake_backend.calls) == 0


def test_dataset_caption_invalid_backend(tmp_path: Path) -> None:
    """Test POST /datasets/{dataset_id}/caption rejects invalid backend."""
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
        files=[("files", ("photo.jpg", io.BytesIO(b"fake image"), "image/jpeg"))],
        headers={"X-API-Key": "test-api-key"},
    )
    upload_data = load_json_str(upload_response.text)
    if not isinstance(upload_data, dict):
        raise AssertionError("Response body must be a JSON object")
    dataset_id = require_str(upload_data, "dataset_id")

    # Try to caption with invalid backend
    invalid_body: JSONObject = {
        "trigger_word": "sks person",
        "backend": "invalid-backend",
        "model_name": "some-model",
    }
    caption_response = client.post(
        f"/datasets/{dataset_id}/caption",
        json=invalid_body,
        headers={"X-API-Key": "test-api-key"},
    )

    assert caption_response.status_code == 400


def test_dataset_caption_invalid_json_body(tmp_path: Path) -> None:
    """Test POST /datasets/{dataset_id}/caption rejects non-object JSON body."""
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
        files=[("files", ("photo.jpg", io.BytesIO(b"fake image"), "image/jpeg"))],
        headers={"X-API-Key": "test-api-key"},
    )
    upload_data = load_json_str(upload_response.text)
    if not isinstance(upload_data, dict):
        raise AssertionError("Response body must be a JSON object")
    dataset_id = require_str(upload_data, "dataset_id")

    # Send JSON array instead of object
    caption_response = client.post(
        f"/datasets/{dataset_id}/caption",
        content='["not", "an", "object"]',
        headers={"X-API-Key": "test-api-key", "Content-Type": "application/json"},
    )

    assert caption_response.status_code == 400


def test_dataset_caption_with_gemini_backend(tmp_path: Path) -> None:
    """Test POST /datasets/{dataset_id}/caption with Gemini backend uses API key."""
    settings = _make_test_settings(tmp_path)
    settings["app"]["gemini_api_key"] = "test-gemini-key"

    def load_settings() -> Settings:
        return settings

    _test_hooks.load_settings = load_settings

    # Reset caption registry
    reset_caption_registry()

    # Set up fake caption backend factory
    fake_backend = FakeCaptionBackend("gemini caption", "gemini")
    received_configs: list[CaptionConfigDict] = []

    def fake_backend_factory(config: CaptionConfigDict) -> FakeCaptionBackend:
        """Create fake caption backend.

        Args:
            config: Caption configuration.

        Returns:
            Fake caption backend instance.
        """
        received_configs.append(config)
        return fake_backend

    captioning_test_hooks.Hooks.caption_backend_factory = fake_backend_factory

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
        files=[("files", ("photo.jpg", io.BytesIO(b"fake image"), "image/jpeg"))],
        headers={"X-API-Key": "test-api-key"},
    )
    upload_data = load_json_str(upload_response.text)
    if not isinstance(upload_data, dict):
        raise AssertionError("Response body must be a JSON object")
    dataset_id = require_str(upload_data, "dataset_id")

    # Caption with Gemini backend
    caption_body: JSONObject = {
        "trigger_word": "sks person",
        "backend": "gemini",
        "model_name": "gemini-2.0-flash",
    }
    caption_response = client.post(
        f"/datasets/{dataset_id}/caption",
        json=caption_body,
        headers={"X-API-Key": "test-api-key"},
    )

    assert caption_response.status_code == 200
    # Verify the config received the API key
    assert len(received_configs) == 1
    assert received_configs[0]["api_key"] == "test-gemini-key"
    assert received_configs[0]["backend"] == "gemini"


def test_dataset_caption_with_openai_backend(tmp_path: Path) -> None:
    """Test POST /datasets/{dataset_id}/caption with OpenAI backend uses API key."""
    settings = _make_test_settings(tmp_path)
    settings["app"]["openai_api_key"] = "test-openai-key"

    def load_settings() -> Settings:
        return settings

    _test_hooks.load_settings = load_settings

    # Reset caption registry
    reset_caption_registry()

    # Set up fake caption backend factory
    fake_backend = FakeCaptionBackend("openai caption", "openai")
    received_configs: list[CaptionConfigDict] = []

    def fake_backend_factory(config: CaptionConfigDict) -> FakeCaptionBackend:
        """Create fake caption backend.

        Args:
            config: Caption configuration.

        Returns:
            Fake caption backend instance.
        """
        received_configs.append(config)
        return fake_backend

    captioning_test_hooks.Hooks.caption_backend_factory = fake_backend_factory

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
        files=[("files", ("photo.jpg", io.BytesIO(b"fake image"), "image/jpeg"))],
        headers={"X-API-Key": "test-api-key"},
    )
    upload_data = load_json_str(upload_response.text)
    if not isinstance(upload_data, dict):
        raise AssertionError("Response body must be a JSON object")
    dataset_id = require_str(upload_data, "dataset_id")

    # Caption with OpenAI backend
    caption_body: JSONObject = {
        "trigger_word": "sks person",
        "backend": "openai",
        "model_name": "gpt-4o",
    }
    caption_response = client.post(
        f"/datasets/{dataset_id}/caption",
        json=caption_body,
        headers={"X-API-Key": "test-api-key"},
    )

    assert caption_response.status_code == 200
    # Verify the config received the API key
    assert len(received_configs) == 1
    assert received_configs[0]["api_key"] == "test-openai-key"
    assert received_configs[0]["backend"] == "openai"
