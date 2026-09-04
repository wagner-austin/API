"""Test configuration and fixtures for TankpitBot tests."""

from __future__ import annotations

from collections.abc import Callable, Generator
from pathlib import Path

import pytest
from platform_core.json_utils import JSONObject

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import (
    AppendTextProtocol,
    CDPSessionProtocol,
    PathExistsProtocol,
    ReadTextProtocol,
)
from tankpit_bot.analysis import _test_hooks as analysis_test_hooks
from tankpit_bot.replay import _test_hooks as replay_test_hooks


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


def _unexpected_kill_browser_processes() -> list[int]:
    """Fail loudly if the teardown remedy fires during a test.

    The real implementation kills every browser-engine descendant of
    the test process — which would include a live module-scoped
    Playwright browser shared by sibling tests. A test that exercises
    the remedy rung must install its own recording fake.

    Raises:
        AssertionError: Always.
    """
    raise AssertionError("kill_browser_processes called without a test-installed fake")


def _fixed_resolve_build_ref() -> str:
    """Deterministic build ref for tests.

    The real implementation asks the environment and then git; a test
    that spawned a git subprocess per ``configure_bot_runtime_logging``
    call would pay ~30 ms hundreds of times per run for an answer no
    artifact assertion wants to depend on.
    """
    return "test-build-ref"


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
    """Reset all shared test hooks to canonical defaults for each test.

    ``get_current_time_ms`` is in the list because a leaked frozen
    clock is the nastiest cross-test poison: the scenarios harness
    installs a scenario clock at construction, and any path that skips
    ``close()`` used to freeze time for every later test on the same
    xdist worker. Frozen dispatch stamps + capture-epoch decide clocks
    made every replay enemy read as ``stale_map_data`` (the 4-test
    replay flake diagnosed 2026-07-03).
    """
    _test_hooks.get_env = _test_hooks._default_get_env
    _test_hooks.get_current_time_ms = _test_hooks._real_get_current_time_ms
    _test_hooks.write_text = _test_hooks._real_write_text
    _test_hooks.read_text = _test_hooks._real_read_text
    _test_hooks.append_text = _test_hooks._real_append_text
    _test_hooks.path_exists = _test_hooks._real_path_exists
    _test_hooks.remove_file = _test_hooks._real_remove_file
    _test_hooks.sync_playwright = None
    _test_hooks.get_sync_playwright = _test_hooks._real_get_sync_playwright
    _test_hooks.load_terrain_map = _test_hooks._real_load_terrain_map
    _test_hooks.get_argv = _test_hooks._real_get_argv
    # Watchdog hooks default to inert fakes in tests: a real daemon
    # timer armed by one test would os._exit the xdist worker seconds
    # later, killing unrelated tests. Tests that assert watchdog
    # behavior install recording fakes explicitly.
    _test_hooks.start_watchdog = _noop_start_watchdog
    _test_hooks.force_exit = _unexpected_force_exit
    _test_hooks.kill_browser_processes = _unexpected_kill_browser_processes
    _test_hooks.resolve_build_ref = _fixed_resolve_build_ref
    _test_hooks.install_signal_handlers = _noop_install_signal_handlers
    # The analysis layer owns its own seams (filesystem reads and
    # archive enumeration). Restoring them here rather than in a
    # per-file fixture keeps the single reset point this fixture's
    # docstring describes.
    analysis_test_hooks.reset_analysis_hooks()
    # The replay decode hook lives in the replay package, not the
    # process-wide one: it names WorldService, which _test_hooks cannot
    # ([[session-state-deglobalisation]] step 8).
    replay_test_hooks.process_received_message_hook = (
        replay_test_hooks._real_process_received_message
    )

    yield

    _test_hooks.start_watchdog = _noop_start_watchdog
    _test_hooks.force_exit = _unexpected_force_exit
    _test_hooks.kill_browser_processes = _unexpected_kill_browser_processes
    _test_hooks.resolve_build_ref = _fixed_resolve_build_ref
    _test_hooks.install_signal_handlers = _noop_install_signal_handlers
    _test_hooks.get_env = _test_hooks._default_get_env
    _test_hooks.get_current_time_ms = _test_hooks._real_get_current_time_ms
    _test_hooks.write_text = _test_hooks._real_write_text
    _test_hooks.read_text = _test_hooks._real_read_text
    _test_hooks.append_text = _test_hooks._real_append_text
    _test_hooks.path_exists = _test_hooks._real_path_exists
    _test_hooks.remove_file = _test_hooks._real_remove_file
    _test_hooks.sync_playwright = None
    _test_hooks.get_sync_playwright = _test_hooks._real_get_sync_playwright
    _test_hooks.load_terrain_map = _test_hooks._real_load_terrain_map
    _test_hooks.get_argv = _test_hooks._real_get_argv
    replay_test_hooks.process_received_message_hook = (
        replay_test_hooks._real_process_received_message
    )


@pytest.fixture(autouse=True)
def _restore_runtime_logging_state() -> Generator[None, None, None]:
    """Detach artifact handlers and clear the ambient run for each test.

    Both the run and the tick context are :class:`contextvars.ContextVar`
    slots, not module globals ([[session-state-deglobalisation]] step
    10). That buys isolation per thread and per async task, which is what
    lets two concurrent sessions keep their own artifacts — but pytest
    runs every test on one thread in one context, so the values persist
    between tests and this reset stays. The gain was never one fewer
    reset.

    Until step 10 this reset cleared the bot and sniff globals and
    silently skipped the probe one, so a probe test leaked its artifacts
    into every test that followed it on the same worker.
    :func:`clear_runtime_logging_state` clears all of them, which is why
    the reset now lives beside the state instead of reaching into it.
    """
    from tankpit_bot import runtime_context, runtime_logging

    runtime_logging.clear_runtime_logging_state()
    runtime_context.clear_runtime_context()

    yield

    runtime_logging.clear_runtime_logging_state()
    runtime_context.clear_runtime_context()


@pytest.fixture(autouse=True)
def _isolate_protocol_singletons() -> Generator[None, None, None]:
    """Reset the one remaining PROCESS-wide cache around every test.

    **This list is down to a single call, and that call is not session
    state.** It began as ten resets covering the world-state dict, the
    XOR table, the event counter, the outcome rings, the decision
    store, the mode-transition log, the outcome-pairing trackers, the
    teleport dispatch, the container blacklist, and the
    client-structure survey. Every one of those is now an instance
    attribute owned by the session that uses it, so constructing a
    session IS the reset ([[session-state-deglobalisation]]).

    What remains is ``reset_static_key_cache``: the XOR *static key* is
    a process-wide constant read from disk, not per-session state, and
    a test with a faked filesystem can poison it for later tests on the
    same xdist worker. It stays because it is genuinely process-scoped
    -- the exception that proves the rule rather than a leftover.
    """
    from tankpit_bot.capture.xor import reset_static_key_cache

    reset_static_key_cache()
    yield
    reset_static_key_cache()


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

    def replace_text(self, path: Path, content: str) -> None:
        """Atomically replace text in the fake file map.

        The in-memory dict assignment is already atomic from a
        reader's perspective, so the fake and the real hook share the
        same observable contract: whole old content or whole new
        content, never a torn write.

        Args:
            path: File path.
            content: File content.
        """
        self._files[str(path)] = content

    def create_text_exclusive(self, path: Path, content: str) -> bool:
        """Create a fake file only if no file is at the path.

        The in-memory check-and-insert is atomic within one process,
        matching the real hook's O_CREAT|O_EXCL contract: exactly one
        creator wins, and the loser's content never lands.

        Args:
            path: File path that must not already exist.
            content: File content.

        Returns:
            True when this call created the file; False when a file
            was already there.
        """
        key = str(path)
        if key in self._files:
            return False
        self._files[key] = content
        return True

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

        # Match the REAL hook's Path.glob contract exactly: the
        # pattern applies to the path relative to ``directory``,
        # segment by segment, and ``*`` never crosses a separator
        # (so "*/knowledge.json" finds instance files one level down
        # without a flat "*.json" ever matching nested paths).
        pattern_parts = pattern.split("/")
        matches: list[Path] = []
        for key in self._files:
            candidate = Path(key)
            if not candidate.is_relative_to(directory):
                continue
            parts = candidate.relative_to(directory).parts
            if len(parts) != len(pattern_parts):
                continue
            if all(fnmatch(part, glob) for part, glob in zip(parts, pattern_parts, strict=True)):
                matches.append(candidate)
        return sorted(matches)

    def get_written_files(self) -> dict[str, str]:
        """Get all written files.

        Returns:
            Dict mapping path strings to contents.
        """
        return dict(self._files)


def install_fake_filesystem() -> tuple[
    FakeFileSystem, PathExistsProtocol, ReadTextProtocol, AppendTextProtocol
]:
    """Install a fake filesystem on the read hooks; return originals to restore.

    The :func:`fake_fs` fixture is the preferred entry point, but a
    ``setup_method``/``teardown_method`` class cannot request a fixture,
    and four modules need exactly this swap from ``setup_method``. They
    each grew a private ``_FakeFileSystem`` plus installer to get it --
    five near-identical copies of a class this module already owned
    (consolidated 2026-08-08). This is that installer, once.

    Callers MUST restore the returned originals in teardown; the
    ``hook_restore`` guard rule fails the build if they do not.

    Returns:
        Tuple of ``(fake, original_path_exists, original_read_text,
        original_append_text)``.
    """
    fake = FakeFileSystem()
    original_path_exists: PathExistsProtocol = _test_hooks.path_exists
    original_read_text: ReadTextProtocol = _test_hooks.read_text
    original_append_text: AppendTextProtocol = _test_hooks.append_text
    _test_hooks.path_exists = fake.path_exists
    _test_hooks.read_text = fake.read_text
    _test_hooks.append_text = fake.append_text
    return (fake, original_path_exists, original_read_text, original_append_text)


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
    from tankpit_bot.resources import static_key_file_path

    original_write_text = _test_hooks.write_text
    original_read_text = _test_hooks.read_text
    original_append_text = _test_hooks.append_text
    original_path_exists = _test_hooks.path_exists
    original_glob_paths = _test_hooks.glob_paths
    original_replace_text = _test_hooks.replace_text
    original_remove_file = _test_hooks.remove_file
    original_create_text_exclusive = _test_hooks.create_text_exclusive

    # Create a 1000-character fake static key for testing
    # This matches the expected format of the real key
    fake_static_key = "Y" + "A" * 999

    fs = FakeFileSystem()
    _test_hooks.write_text = fs.write_text
    _test_hooks.read_text = fs.read_text
    _test_hooks.append_text = fs.append_text
    _test_hooks.path_exists = fs.path_exists
    _test_hooks.glob_paths = fs.glob_paths
    _test_hooks.replace_text = fs.replace_text
    _test_hooks.remove_file = fs.remove
    _test_hooks.create_text_exclusive = fs.create_text_exclusive

    # Pre-populate the static key file
    fs.write_text(static_key_file_path(), fake_static_key)

    yield fs

    _test_hooks.write_text = original_write_text
    _test_hooks.read_text = original_read_text
    _test_hooks.append_text = original_append_text
    _test_hooks.path_exists = original_path_exists
    _test_hooks.glob_paths = original_glob_paths
    _test_hooks.replace_text = original_replace_text
    _test_hooks.remove_file = original_remove_file
    _test_hooks.create_text_exclusive = original_create_text_exclusive


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
