"""Test hooks for deployment service.

This module provides dependency injection hooks for the LoRA deployment service.
Production code sets the hooks to real implementations at startup.
Tests set them to fakes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class FileCopier(Protocol):
    """Protocol for copying files."""

    def __call__(self, src: Path, dst: Path) -> Path:
        """Copy a file from source to destination.

        Args:
            src: Source file path.
            dst: Destination file path.

        Returns:
            Path to the copied file.
        """
        ...


class Hooks:
    """Container for test hooks.

    Attributes:
        file_copier: Hook for file copying operations.
    """

    file_copier: FileCopier | None = None


def reset_hooks() -> None:
    """Reset all hooks to None for test isolation."""
    Hooks.file_copier = None


def _default_file_copier(src: Path, dst: Path) -> Path:
    """Default file copier using shutil.

    Args:
        src: Source file path.
        dst: Destination file path.

    Returns:
        Path to the copied file.
    """
    import shutil

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


__all__ = [
    "FileCopier",
    "Hooks",
    "reset_hooks",
]
