"""Tests for caption backend registry."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image

from art_trainer.core.services.captioning.backends import (
    CaptionBackendRegistry,
    CaptionConfig,
    get_caption_registry,
    reset_caption_registry,
)


def test_registry_caches_backends() -> None:
    """Test that registry caches backend instances."""
    registry = CaptionBackendRegistry()

    config: CaptionConfig = {
        "backend": "blip",
        "model_name": "Salesforce/blip-image-captioning-base",
        "api_key": "",
    }

    backend1 = registry.get_backend(config)
    backend2 = registry.get_backend(config)

    # Same instance should be returned
    assert backend1 is backend2


def test_registry_creates_different_backends_for_different_configs() -> None:
    """Test that registry creates different backends for different configs."""
    registry = CaptionBackendRegistry()

    config1: CaptionConfig = {
        "backend": "blip",
        "model_name": "Salesforce/blip-image-captioning-base",
        "api_key": "",
    }

    config2: CaptionConfig = {
        "backend": "blip",
        "model_name": "Salesforce/blip-image-captioning-large",
        "api_key": "",
    }

    backend1 = registry.get_backend(config1)
    backend2 = registry.get_backend(config2)

    # Different instances for different model names
    assert backend1 is not backend2


def test_registry_creates_blip_backend() -> None:
    """Test that registry creates BLIP backend."""
    registry = CaptionBackendRegistry()

    config: CaptionConfig = {
        "backend": "blip",
        "model_name": "Salesforce/blip-image-captioning-base",
        "api_key": "",
    }

    backend = registry.get_backend(config)

    assert backend.backend_type == "blip"


def test_registry_creates_gemini_backend() -> None:
    """Test that registry creates Gemini backend."""
    registry = CaptionBackendRegistry()

    config: CaptionConfig = {
        "backend": "gemini",
        "model_name": "gemini-2.0-flash",
        "api_key": "test-api-key",
    }

    backend = registry.get_backend(config)

    assert backend.backend_type == "gemini"


def test_registry_creates_openai_backend() -> None:
    """Test that registry creates OpenAI backend."""
    registry = CaptionBackendRegistry()

    config: CaptionConfig = {
        "backend": "openai",
        "model_name": "gpt-4o",
        "api_key": "test-api-key",
    }

    backend = registry.get_backend(config)

    assert backend.backend_type == "openai"


def test_get_caption_registry_returns_singleton() -> None:
    """Test that get_caption_registry returns singleton."""
    reset_caption_registry()

    registry1 = get_caption_registry()
    registry2 = get_caption_registry()

    assert registry1 is registry2


def test_reset_caption_registry_clears_singleton() -> None:
    """Test that reset_caption_registry clears singleton."""
    registry1 = get_caption_registry()
    reset_caption_registry()
    registry2 = get_caption_registry()

    assert registry1 is not registry2


def test_blip_backend_adapter_caption(tmp_path: Path) -> None:
    """Test BLIP backend adapter caption method."""
    registry = CaptionBackendRegistry()

    config: CaptionConfig = {
        "backend": "blip",
        "model_name": "Salesforce/blip-image-captioning-base",
        "api_key": "",
    }

    backend = registry.get_backend(config)

    # Create a valid test image using PIL
    image_path = tmp_path / "test.png"
    img = Image.new("RGB", (32, 32), color=(128, 128, 128))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    image_path.write_bytes(buffer.getvalue())

    # Caption should work (lazy loads model)
    caption = backend.caption(image_path, "test_trigger")

    assert caption.startswith("test_trigger, ")
    assert len(caption) > len("test_trigger, ")
