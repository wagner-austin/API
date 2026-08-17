"""Integration tests for BLIP captioning model."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from art_trainer.core.services.captioning import _test_hooks
from art_trainer.core.services.captioning.blip_model import (
    BlipCaptioner,
)


@pytest.fixture(autouse=True)
def reset_blip_singleton() -> None:
    """Reset BlipCaptioner singleton before each test."""
    BlipCaptioner.reset_instance()
    _test_hooks.reset_hooks()


def _create_test_image(tmp_path: Path) -> Path:
    """Create a valid RGB PNG image for testing using PIL.

    Args:
        tmp_path: Temporary directory.

    Returns:
        Path to the test image.
    """
    image_path = tmp_path / "test_image.png"
    # Create a small 32x32 RGB image with PIL
    img = Image.new("RGB", (32, 32), color=(128, 128, 128))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    image_path.write_bytes(buffer.getvalue())
    return image_path


def test_blip_captioner_get_instance_returns_singleton() -> None:
    """Test get_instance returns the same instance."""
    model_name = "Salesforce/blip-image-captioning-base"

    instance1 = BlipCaptioner.get_instance(model_name)
    instance2 = BlipCaptioner.get_instance(model_name)

    assert instance1 is instance2


def test_blip_captioner_reset_instance_clears_singleton() -> None:
    """Test reset_instance clears the singleton."""
    model_name = "Salesforce/blip-image-captioning-base"

    instance1 = BlipCaptioner.get_instance(model_name)
    BlipCaptioner.reset_instance()
    instance2 = BlipCaptioner.get_instance(model_name)

    assert instance1 is not instance2


def test_blip_captioner_caption_file_not_found(tmp_path: Path) -> None:
    """Test caption raises FileNotFoundError for missing file."""
    captioner = BlipCaptioner("Salesforce/blip-image-captioning-base")
    nonexistent = tmp_path / "nonexistent.png"

    with pytest.raises(FileNotFoundError) as exc_info:
        captioner.caption(nonexistent, "trigger")

    assert "nonexistent.png" in str(exc_info.value)


def test_blip_captioner_caption_generates_text(tmp_path: Path) -> None:
    """Test caption generates text for a valid image."""
    captioner = BlipCaptioner("Salesforce/blip-image-captioning-base")
    image_path = _create_test_image(tmp_path)

    caption = captioner.caption(image_path, "sks")

    # Caption should start with trigger word
    assert caption.startswith("sks, ")
    # Caption should have more content after trigger
    assert len(caption) > len("sks, ")


def test_blip_captioner_lazy_loads_model(tmp_path: Path) -> None:
    """Test captioner lazy loads model on first caption call."""
    captioner = BlipCaptioner("Salesforce/blip-image-captioning-base")

    # Model should not be loaded yet
    assert captioner._model is None
    assert captioner._processor is None

    # Caption should trigger model loading
    image_path = _create_test_image(tmp_path)
    captioner.caption(image_path, "lazy")

    # Model should now be loaded - verify by checking they're callable/usable
    if captioner._model is None:
        raise AssertionError("Model should be loaded after caption call")
    if captioner._processor is None:
        raise AssertionError("Processor should be loaded after caption call")


def test_blip_captioner_reuses_loaded_model(tmp_path: Path) -> None:
    """Test captioner reuses already loaded model."""
    captioner = BlipCaptioner("Salesforce/blip-image-captioning-base")
    image_path = _create_test_image(tmp_path)

    # First caption loads model
    captioner.caption(image_path, "first")
    model1 = captioner._model
    processor1 = captioner._processor

    # Second caption reuses model
    captioner.caption(image_path, "second")
    model2 = captioner._model
    processor2 = captioner._processor

    assert model1 is model2
    assert processor1 is processor2
