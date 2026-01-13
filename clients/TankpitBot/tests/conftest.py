"""Test configuration and fixtures for TankpitBot tests."""

from __future__ import annotations

from collections.abc import Callable, Generator
from pathlib import Path

import pytest
from platform_core.json_utils import JSONObject
from scripts import _test_hooks as scripts_test_hooks

from tankpit_bot import _test_hooks


class FakeCDPSessionSimple:
    """Simple fake CDP session for testing with configurable responses.

    Used for testing code that only needs send() without event handlers.
    """

    def __init__(self) -> None:
        """Initialize with empty response queue."""
        self._responses: list[JSONObject] = []
        self._response_index: int = 0
        self._calls: list[tuple[str, JSONObject | None]] = []

    def add_response(self, response: JSONObject) -> None:
        """Add a response to return on next send() call.

        Args:
            response: The response to return.
        """
        self._responses.append(response)

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Send a CDP command and return configured response.

        Args:
            method: CDP method name.
            params: Optional parameters.

        Returns:
            Next configured response, or empty dict if none configured.
        """
        self._calls.append((method, params))

        if self._response_index < len(self._responses):
            response = self._responses[self._response_index]
            self._response_index += 1
            return response
        return {}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        """Register event handler (no-op for testing).

        Args:
            event: Event name.
            handler: Event handler.
        """
        _ = (event, handler)

    def detach(self) -> None:
        """Detach session (no-op for testing)."""

    def get_calls(self) -> list[tuple[str, JSONObject | None]]:
        """Get all recorded send() calls.

        Returns:
            List of (method, params) tuples.
        """
        return list(self._calls)

    @property
    def call_count(self) -> int:
        """Get number of send() calls made.

        Returns:
            Number of calls.
        """
        return len(self._calls)


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
    original_get_argv = _test_hooks.get_argv

    yield

    # Restore hooks
    _test_hooks.get_env = original_get_env
    _test_hooks.write_text = original_write_text
    _test_hooks.read_text = original_read_text
    _test_hooks.path_exists = original_path_exists
    _test_hooks.sync_playwright = original_sync_playwright
    _test_hooks.get_sync_playwright = original_get_sync_playwright
    _test_hooks.get_argv = original_get_argv


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

    def remove(self, path: Path) -> None:
        """Remove a file from the fake file system.

        Args:
            path: Path to remove.
        """
        key = str(path)
        if key in self._files:
            del self._files[key]

    def get_written_files(self) -> dict[str, str]:
        """Get all written files.

        Returns:
            Dict mapping path strings to contents.
        """
        return dict(self._files)


@pytest.fixture()
def fake_fs() -> FakeFileSystem:
    """Create a FakeFileSystem and install it as hooks.

    Pre-populates the static XOR key file used by the codec and probe modules.

    Returns:
        FakeFileSystem instance.
    """
    from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

    # Create a 1000-character fake static key for testing
    # This matches the expected format of the real key
    fake_static_key = "Y" + "A" * 999

    fs = FakeFileSystem()
    _test_hooks.write_text = fs.write_text
    _test_hooks.read_text = fs.read_text
    _test_hooks.path_exists = fs.path_exists

    # Pre-populate the static key file
    fs.write_text(DEFAULT_STATIC_KEY_PATH, fake_static_key)

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


@pytest.fixture()
def fake_cdp() -> FakeCDPSessionSimple:
    """Create a FakeCDPSessionSimple for testing.

    Returns:
        FakeCDPSessionSimple instance.
    """
    return FakeCDPSessionSimple()
