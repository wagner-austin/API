"""Caption a directory of images with a caption backend.

The backend is passed in rather than reached through module state, so the
caller decides which one -- the upload route builds the BLIP one from
settings, and a test passes its own.
"""

from __future__ import annotations

from pathlib import Path

from art_trainer.core.contracts.dataset import CaptionResult

from .backends import CaptionBackend

# Supported image extensions
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


def caption_image(
    image_path: Path,
    trigger_word: str,
    output_dir: Path,
    backend: CaptionBackend,
) -> CaptionResult:
    """Generate a caption for a single image and save it.

    Args:
        image_path: Path to the image file.
        trigger_word: Trigger word to prepend to caption.
        output_dir: Directory to save the caption file.
        backend: Backend that produces the caption.

    Returns:
        CaptionResult with image name, caption, and caption file path.
    """
    caption = backend.caption(image_path, trigger_word)

    # Write caption to file
    caption_filename = image_path.stem + ".txt"
    caption_path = output_dir / caption_filename
    caption_path.write_text(caption, encoding="utf-8")

    return {
        "image_name": image_path.name,
        "caption": caption,
        "caption_path": str(caption_path),
    }


def caption_images(
    image_paths: list[Path],
    trigger_word: str,
    output_dir: Path,
    backend: CaptionBackend,
) -> list[CaptionResult]:
    """Generate captions for multiple images.

    Args:
        image_paths: List of paths to image files.
        trigger_word: Trigger word to prepend to captions.
        output_dir: Directory to save caption files.
        backend: Backend that produces the captions.

    Returns:
        List of CaptionResult for each image.
    """
    results: list[CaptionResult] = []
    for image_path in image_paths:
        result = caption_image(image_path, trigger_word, output_dir, backend)
        results.append(result)
    return results


def find_images(directory: Path) -> list[Path]:
    """Find all image files in a directory.

    Args:
        directory: Directory to search for images.

    Returns:
        List of paths to image files (deduplicated).
    """
    # Use set to deduplicate (Windows glob is case-insensitive)
    images: set[Path] = set()
    for ext in IMAGE_EXTENSIONS:
        images.update(directory.glob(f"*{ext}"))
        images.update(directory.glob(f"*{ext.upper()}"))
    return sorted(images)


__all__ = [
    "IMAGE_EXTENSIONS",
    "caption_image",
    "caption_images",
    "find_images",
]
