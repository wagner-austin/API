"""Typed Pillow image loading helpers.

This module isolates the dynamic Pillow import behind local protocols so the
rest of the codebase stays fully strict without relying on mypy import
exceptions.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol


class PillowImageProtocol(Protocol):
    """Minimal image protocol used by terrain and rendering tests."""

    @property
    def size(self) -> tuple[int, int]:
        """Return image dimensions."""
        ...

    def convert(self, mode: str) -> PillowImageProtocol:
        """Convert the image to a different mode."""
        ...

    def tobytes(self) -> bytes:
        """Return raw image bytes."""
        ...

    def putdata(self, data: Sequence[tuple[int, int, int]]) -> None:
        """Replace the image pixel data."""
        ...

    def save(self, fp: str | Path) -> None:
        """Save the image to disk."""
        ...


class PillowImageModuleProtocol(Protocol):
    """Minimal Pillow Image module protocol used by this project."""

    def open(self, fp: str | Path) -> PillowImageProtocol:
        """Open an image file."""
        ...

    def new(
        self,
        mode: str,
        size: tuple[int, int],
        color: tuple[int, int, int] | None = None,
    ) -> PillowImageProtocol:
        """Create a new image."""
        ...


def load_pillow_image_module() -> PillowImageModuleProtocol:
    """Load the Pillow Image module through a typed protocol boundary.

    Returns:
        Typed image module protocol.
    """
    image_module: PillowImageModuleProtocol = __import__("PIL.Image", fromlist=["open", "new"])
    return image_module


__all__ = [
    "PillowImageModuleProtocol",
    "PillowImageProtocol",
    "load_pillow_image_module",
]
