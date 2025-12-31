"""Test configuration and fixtures for TankpitBot tests."""

from __future__ import annotations

from collections.abc import Callable, Generator
from pathlib import Path

import pytest
from scripts import _test_hooks as scripts_test_hooks

from tankpit_bot import _test_hooks


@pytest.fixture(autouse=True)
def _restore_hooks() -> Generator[None, None, None]:
    """Restore all hooks after each test."""
    # Save original hooks
    original_get_env = _test_hooks.get_env
    original_write_text = _test_hooks.write_text
    original_read_text = _test_hooks.read_text
    original_path_exists = _test_hooks.path_exists
    original_sync_playwright = _test_hooks.sync_playwright
    original_get_sync_playwright = _test_hooks.get_sync_playwright

    yield

    # Restore hooks
    _test_hooks.get_env = original_get_env
    _test_hooks.write_text = original_write_text
    _test_hooks.read_text = original_read_text
    _test_hooks.path_exists = original_path_exists
    _test_hooks.sync_playwright = original_sync_playwright
    _test_hooks.get_sync_playwright = original_get_sync_playwright


@pytest.fixture(autouse=True)
def _restore_scripts_hooks() -> Generator[None, None, None]:
    """Restore scripts hooks after each test."""
    original_is_dir = scripts_test_hooks.is_dir
    yield
    scripts_test_hooks.is_dir = original_is_dir


class FakeEnv:
    """Fake environment for testing."""

    def __init__(self, env_vars: dict[str, str] | None = None) -> None:
        """Initialize with optional environment variables.

        Args:
            env_vars: Initial environment variables.
        """
        self._env: dict[str, str] = env_vars if env_vars is not None else {}

    def get(self, key: str) -> str | None:
        """Get environment variable.

        Args:
            key: Variable name.

        Returns:
            Variable value or None if not set.
        """
        return self._env.get(key)

    def set(self, key: str, value: str) -> None:
        """Set environment variable.

        Args:
            key: Variable name.
            value: Variable value.
        """
        self._env[key] = value

    def __call__(self, key: str) -> str | None:
        """Get environment variable (callable interface).

        Args:
            key: Variable name.

        Returns:
            Variable value or None if not set.
        """
        return self.get(key)


@pytest.fixture()
def fake_env() -> FakeEnv:
    """Create a FakeEnv and install it as the hook.

    Returns:
        FakeEnv instance.
    """
    env = FakeEnv()
    _test_hooks.get_env = env
    return env


class FakeFileSystem:
    """Fake file system for testing."""

    def __init__(self) -> None:
        """Initialize empty file system."""
        self._files: dict[str, str] = {}

    def write_text(self, path: Path, content: str) -> None:
        """Write text to fake file.

        Args:
            path: File path.
            content: File content.
        """
        self._files[str(path)] = content

    def read_text(self, path: Path) -> str:
        """Read text from fake file.

        Args:
            path: File path.

        Returns:
            File content.

        Raises:
            FileNotFoundError: If file does not exist.
        """
        key = str(path)
        if key not in self._files:
            raise FileNotFoundError(f"File not found: {path}")
        return self._files[key]

    def path_exists(self, path: Path) -> bool:
        """Check if path exists.

        Args:
            path: Path to check.

        Returns:
            True if path exists.
        """
        return str(path) in self._files

    def get_written_files(self) -> dict[str, str]:
        """Get all written files.

        Returns:
            Dict mapping path strings to contents.
        """
        return dict(self._files)


@pytest.fixture()
def fake_fs() -> FakeFileSystem:
    """Create a FakeFileSystem and install it as hooks.

    Returns:
        FakeFileSystem instance.
    """
    fs = FakeFileSystem()
    _test_hooks.write_text = fs.write_text
    _test_hooks.read_text = fs.read_text
    _test_hooks.path_exists = fs.path_exists
    return fs


def make_fake_get_env(env_vars: dict[str, str]) -> Callable[[str], str | None]:
    """Create a fake get_env function.

    Args:
        env_vars: Environment variables to return.

    Returns:
        Callable that looks up keys in env_vars.
    """

    def _get(key: str) -> str | None:
        return env_vars.get(key)

    return _get
