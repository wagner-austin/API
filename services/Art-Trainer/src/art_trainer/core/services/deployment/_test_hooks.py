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


class Hooks:
    """Container for deployment hooks.

    Attributes:
        file_copier: Hook for file copying operations, bound to shutil.copy2.
    """

    file_copier: FileCopier = _default_file_copier


def reset_hooks() -> None:
    """Restore every hook to the implementation the container binds."""
    Hooks.file_copier = _default_file_copier


__all__ = [
    "FileCopier",
    "Hooks",
    "reset_hooks",
]
