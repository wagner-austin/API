"""Tests for BLIP captioning adapter."""

from __future__ import annotations

from pathlib import Path

from art_trainer.core.services.captioning.backends import CaptionBackendType
from art_trainer.core.services.captioning.blip_adapter import (
    IMAGE_EXTENSIONS,
    caption_image,
    caption_images,
    find_images,
)


class FakeCaptionBackend:
    """Fake caption backend for tests."""

    calls: list[tuple[Path, str]]

    def __init__(self, caption: str = "sks person standing") -> None:
        """Initialize fake caption generator.

        Args:
            caption: Caption to return.
        """
        self.calls = []
        self._caption = caption

    def caption(self, image_path: Path, trigger_word: str) -> str:
        """Generate a fake caption.

        Args:
            image_path: Path to image file.
            trigger_word: Trigger word.

        Returns:
            Fake caption string.
        """
        self.calls.append((image_path, trigger_word))
        return f"{trigger_word} {self._caption}"

    @property
    def backend_type(self) -> CaptionBackendType:
        """Identify which backend this stands in for.

        Returns:
            Always "blip", since this adapter's caller asks for BLIP.
        """
        return "blip"


def test_image_extensions() -> None:
    """Test IMAGE_EXTENSIONS contains expected formats."""
    assert ".jpg" in IMAGE_EXTENSIONS
    assert ".jpeg" in IMAGE_EXTENSIONS
    assert ".png" in IMAGE_EXTENSIONS
    assert ".webp" in IMAGE_EXTENSIONS
    assert ".bmp" in IMAGE_EXTENSIONS


def test_caption_image_success(tmp_path: Path) -> None:
    """Test caption_image generates caption and saves file."""
    fake_backend = FakeCaptionBackend("in a park")

    image_path = tmp_path / "photo.jpg"
    image_path.touch()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    result = caption_image(image_path, "sks person", output_dir, fake_backend)

    assert result["image_name"] == "photo.jpg"
    assert result["caption"] == "sks person in a park"
    assert result["caption_path"] == str(output_dir / "photo.txt")

    # Verify caption file was written
    caption_file = output_dir / "photo.txt"
    assert caption_file.exists()
    assert caption_file.read_text(encoding="utf-8") == "sks person in a park"

    # Verify generator was called correctly
    assert len(fake_backend.calls) == 1
    assert fake_backend.calls[0] == (image_path, "sks person")


def test_caption_images_multiple(tmp_path: Path) -> None:
    """Test caption_images processes multiple images."""
    fake_backend = FakeCaptionBackend("smiling")

    # Create test images
    image1 = tmp_path / "img1.jpg"
    image2 = tmp_path / "img2.png"
    image1.touch()
    image2.touch()

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    results = caption_images([image1, image2], "sks person", output_dir, fake_backend)

    assert len(results) == 2
    assert results[0]["image_name"] == "img1.jpg"
    assert results[1]["image_name"] == "img2.png"
    assert len(fake_backend.calls) == 2


def test_find_images_empty_directory(tmp_path: Path) -> None:
    """Test find_images returns empty list for empty directory."""
    result = find_images(tmp_path)
    assert result == []


def test_find_images_finds_all_formats(tmp_path: Path) -> None:
    """Test find_images finds all supported formats."""
    # Create test images
    (tmp_path / "photo.jpg").touch()
    (tmp_path / "image.jpeg").touch()
    (tmp_path / "screenshot.png").touch()
    (tmp_path / "web.webp").touch()
    (tmp_path / "bitmap.bmp").touch()
    (tmp_path / "upper.JPG").touch()

    # Create non-image file
    (tmp_path / "document.txt").touch()

    result = find_images(tmp_path)

    # Should find all image files (lowercase and uppercase)
    names = [p.name for p in result]
    assert "photo.jpg" in names
    assert "image.jpeg" in names
    assert "screenshot.png" in names
    assert "web.webp" in names
    assert "bitmap.bmp" in names
    assert "upper.JPG" in names
    assert "document.txt" not in names


def test_find_images_sorted(tmp_path: Path) -> None:
    """Test find_images returns sorted list."""
    (tmp_path / "c.jpg").touch()
    (tmp_path / "a.jpg").touch()
    (tmp_path / "b.jpg").touch()

    result = find_images(tmp_path)

    names = [p.name for p in result]
    assert names == sorted(names)
