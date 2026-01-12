"""Test hooks for guard script.

Production code uses real implementations; tests override these module-level
symbols to inject fakes without conditionals in core logic.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class IsDirProtocol(Protocol):
    """Protocol for checking if a path is a directory."""

    def __call__(self, path: Path) -> bool:
        """Check if path is a directory.

        Args:
            path: Path to check.

        Returns:
            True if path is a directory, False otherwise.
        """
        ...


def _real_is_dir(path: Path) -> bool:
    """Real implementation using Path.is_dir().

    Args:
        path: Path to check.

    Returns:
        True if path is a directory, False otherwise.
    """
    return path.is_dir()


is_dir: IsDirProtocol = _real_is_dir


class GetScriptPathProtocol(Protocol):
    """Protocol for getting the script path."""

    def __call__(self) -> Path:
        """Get the script path.

        Returns:
            Path to the guard script.
        """
        ...


_SCRIPT_PATH: Path | None = None


def _real_get_script_path() -> Path:
    """Real implementation returning the actual guard.py path.

    Returns:
        Path to the guard script.

    Raises:
        RuntimeError: If script path not set.
    """
    if _SCRIPT_PATH is None:
        raise RuntimeError("Script path not set - call set_script_path first")
    return _SCRIPT_PATH


def set_script_path(path: Path) -> None:
    """Set the script path for production use.

    Args:
        path: Path to the guard script.
    """
    global _SCRIPT_PATH
    _SCRIPT_PATH = path


get_script_path: GetScriptPathProtocol = _real_get_script_path


__all__ = [
    "GetScriptPathProtocol",
    "IsDirProtocol",
    "get_script_path",
    "is_dir",
    "set_script_path",
]
