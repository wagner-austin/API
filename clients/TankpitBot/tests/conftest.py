"""Test configuration and fixtures for TankpitBot tests."""

from __future__ import annotations

from collections.abc import Callable, Generator
from pathlib import Path

import pytest
from platform_core.json_utils import JSONObject

from tankpit_bot import _hooks_guard, _test_hooks
from tankpit_bot._test_hooks import CDPSessionProtocol


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


def _noop_start_watchdog(seconds: float, on_fire: Callable[[], None]) -> None:
    """Inert watchdog for tests: never arms a real timer.

    Args:
        seconds: Ignored delay.
        on_fire: Ignored callback.
    """
    del seconds, on_fire


def _unexpected_force_exit(exit_code: int) -> None:
    """Fail loudly if production code force-exits during a test.

    Args:
        exit_code: Exit code the production code requested.

    Raises:
        AssertionError: Always; a test that expects a forced exit must
            install its own recording fake.
    """
    raise AssertionError(f"force_exit({exit_code}) called without a test-installed fake")


def _noop_install_signal_handlers(on_interrupt: Callable[[], None]) -> None:
    """Inert signal-handler installer for tests.

    The real implementation binds SIGINT/SIGTERM via ``signal.signal``;
    tests must not mutate process-wide signal state. Tests that
    exercise the handler install their own recording fake.

    Args:
        on_interrupt: Ignored callback that production would register.
    """
    del on_interrupt


@pytest.fixture(autouse=True)
def _restore_hooks() -> Generator[None, None, None]:
    """Reset all shared test hooks to canonical defaults for each test."""
    _test_hooks.get_env = _test_hooks._default_get_env
    _test_hooks.write_text = _test_hooks._real_write_text
    _test_hooks.read_text = _test_hooks._real_read_text
    _test_hooks.append_text = _test_hooks._real_append_text
    _test_hooks.path_exists = _test_hooks._real_path_exists
    _test_hooks.find_best_static_byte = None
    _test_hooks.sync_playwright = None
    _test_hooks.get_sync_playwright = _test_hooks._real_get_sync_playwright
    _test_hooks.load_terrain_map = _test_hooks._real_load_terrain_map
    _test_hooks.get_argv = _test_hooks._real_get_argv
    _test_hooks.process_received_message_hook = _test_hooks._real_process_received_message
    # Watchdog hooks default to inert fakes in tests: a real daemon
    # timer armed by one test would os._exit the xdist worker seconds
    # later, killing unrelated tests. Tests that assert watchdog
    # behavior install recording fakes explicitly.
    _test_hooks.start_watchdog = _noop_start_watchdog
    _test_hooks.force_exit = _unexpected_force_exit
    _test_hooks.install_signal_handlers = _noop_install_signal_handlers

    yield

    _test_hooks.start_watchdog = _noop_start_watchdog
    _test_hooks.force_exit = _unexpected_force_exit
    _test_hooks.install_signal_handlers = _noop_install_signal_handlers
    _test_hooks.get_env = _test_hooks._default_get_env
    _test_hooks.write_text = _test_hooks._real_write_text
    _test_hooks.read_text = _test_hooks._real_read_text
    _test_hooks.append_text = _test_hooks._real_append_text
    _test_hooks.path_exists = _test_hooks._real_path_exists
    _test_hooks.find_best_static_byte = None
    _test_hooks.sync_playwright = None
    _test_hooks.get_sync_playwright = _test_hooks._real_get_sync_playwright
    _test_hooks.load_terrain_map = _test_hooks._real_load_terrain_map
    _test_hooks.get_argv = _test_hooks._real_get_argv
    _test_hooks.process_received_message_hook = _test_hooks._real_process_received_message


@pytest.fixture(autouse=True)
def _restore_guard_hooks() -> Generator[None, None, None]:
    """Restore guard hooks after each test."""
    yield
    _hooks_guard.guard_find_monorepo_root = None
    _hooks_guard.guard_load_orchestrator = None


@pytest.fixture(autouse=True)
def _restore_runtime_logging_state() -> Generator[None, None, None]:
    """Reset runtime logging globals and artifact handlers for each test."""
    from platform_core.logging import stdlib_logging

    from tankpit_bot import runtime_logging

    runtime_logging._BOT_ARTIFACTS = None
    runtime_logging._SNIFF_ARTIFACTS = None
    runtime_logging.clear_runtime_context()
    runtime_logging._remove_artifact_handlers(stdlib_logging.getLogger())

    yield

    runtime_logging._BOT_ARTIFACTS = None
    runtime_logging._SNIFF_ARTIFACTS = None
    runtime_logging.clear_runtime_context()
    runtime_logging._remove_artifact_handlers(stdlib_logging.getLogger())


@pytest.fixture(autouse=True)
def _isolate_protocol_singletons() -> Generator[None, None, None]:
    """Reset world-state and XOR singletons around every test.

    Several test paths (replay harnesses, sniffer/world_state tests,
    bot tick-loop tests) mutate module-level singletons -- the global
    world-state dict, the global XOR table built from the session
    magic key, etc. Without a consistent reset, tests that run later
    on the same xdist worker can decode bytes with a stale XOR key or
    read containers seeded by a prior run, producing failures that
    look like flakes.

    Centralising the reset here (top-level autouse) means every test
    inherits the same clean baseline regardless of which directory it
    lives in -- no duplicated per-file fixtures, no missed resets.
    """
    from tankpit_bot.bot.ai.recover_equipment_mode import reset_container_blacklist
    from tankpit_bot.diagnostics.teleport_attempts import reset_teleport_attempt_tracking
    from tankpit_bot.sniffer.world_state import reset_world_state
    from tankpit_bot.sniffer.xor import reset_xor_state

    reset_world_state()
    reset_xor_state()
    reset_teleport_attempt_tracking()
    reset_container_blacklist()
    yield
    reset_world_state()
    reset_xor_state()
    reset_teleport_attempt_tracking()
    reset_container_blacklist()


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

    def append_text(self, path: Path, content: str) -> None:
        """Append text to a fake file.

        Args:
            path: File path.
            content: Content to append.
        """
        key = str(path)
        existing = self._files.get(key, "")
        self._files[key] = existing + content

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

    def glob_paths(self, directory: Path, pattern: str) -> list[Path]:
        """List fake files in a directory matching a glob pattern.

        Args:
            directory: Directory to list.
            pattern: Glob pattern matched against file names.

        Returns:
            Matching paths in sorted order.
        """
        from fnmatch import fnmatch

        matches = [
            Path(key)
            for key in self._files
            if Path(key).parent == directory and fnmatch(Path(key).name, pattern)
        ]
        return sorted(matches)

    def get_written_files(self) -> dict[str, str]:
        """Get all written files.

        Returns:
            Dict mapping path strings to contents.
        """
        return dict(self._files)


@pytest.fixture()
def fake_fs() -> Generator[FakeFileSystem, None, None]:
    """Create a FakeFileSystem, install it as hooks, and restore on teardown.

    Pre-populates the static XOR key file used by the codec and probe
    modules. The original hooks are restored after the test: without
    teardown, every later test on the same xdist worker silently kept
    the fake file system, so any test reading a real fixture file (for
    example ``RecordedChromiumSession.from_capture_path``) passed or
    failed depending on scheduling order.

    Yields:
        FakeFileSystem instance.
    """
    from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

    original_write_text = _test_hooks.write_text
    original_read_text = _test_hooks.read_text
    original_append_text = _test_hooks.append_text
    original_path_exists = _test_hooks.path_exists
    original_glob_paths = _test_hooks.glob_paths

    # Create a 1000-character fake static key for testing
    # This matches the expected format of the real key
    fake_static_key = "Y" + "A" * 999

    fs = FakeFileSystem()
    _test_hooks.write_text = fs.write_text
    _test_hooks.read_text = fs.read_text
    _test_hooks.append_text = fs.append_text
    _test_hooks.path_exists = fs.path_exists
    _test_hooks.glob_paths = fs.glob_paths

    # Pre-populate the static key file
    fs.write_text(DEFAULT_STATIC_KEY_PATH, fake_static_key)

    yield fs

    _test_hooks.write_text = original_write_text
    _test_hooks.read_text = original_read_text
    _test_hooks.append_text = original_append_text
    _test_hooks.path_exists = original_path_exists
    _test_hooks.glob_paths = original_glob_paths


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


@pytest.fixture(scope="module")
def live_cdp() -> Generator[CDPSessionProtocol, None, None]:
    """Yield a real CDP session attached to a rendered headless page.

    Launches one genuine headless Chromium per test module (loadscope
    keeps a module's tests on one worker) and tears it down afterwards,
    so the launch cost is paid once. Used to exercise CDP screenshot
    capture against a real browser rather than a substitute.

    Yields:
        A live CDP session whose page has visible rendered content.
    """
    factory = _test_hooks.get_sync_playwright()
    with factory() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            context = browser.new_context()
            page = context.new_page()
            page.goto("data:text/html,<body style='margin:0;background:#33aa66'>tankpit</body>")
            yield context.new_cdp_session(page)
        finally:
            browser.close()
