"""Test hooks for tankpit_bot - allows injecting test dependencies.

Production code uses real implementations; tests override these module-level
symbols to inject fakes without conditionals in core logic.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

import types
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from platform_core.config import _optional_env_str
from platform_core.json_utils import JSONObject, JSONValue

# =============================================================================
# Environment Variable Hook
# =============================================================================


def _default_get_env(key: str) -> str | None:
    """Production implementation - reads via platform_core.

    Args:
        key: Environment variable name.

    Returns:
        Environment variable value or None if not set.
    """
    return _optional_env_str(key)


get_env: Callable[[str], str | None] = _default_get_env


# =============================================================================
# File System Hooks
# =============================================================================


class WriteTextProtocol(Protocol):
    """Protocol for writing text to a file."""

    def __call__(self, path: Path, content: str) -> None:
        """Write text content to a file.

        Args:
            path: File path to write to.
            content: Text content to write.
        """
        ...


class ReadTextProtocol(Protocol):
    """Protocol for reading text from a file."""

    def __call__(self, path: Path) -> str:
        """Read text content from a file.

        Args:
            path: File path to read from.

        Returns:
            Text content of the file.

        Raises:
            FileNotFoundError: If file does not exist.
        """
        ...


class PathExistsProtocol(Protocol):
    """Protocol for checking if a path exists."""

    def __call__(self, path: Path) -> bool:
        """Check if path exists.

        Args:
            path: Path to check.

        Returns:
            True if path exists, False otherwise.
        """
        ...


def _real_write_text(path: Path, content: str) -> None:
    """Real implementation using Path.write_text().

    Args:
        path: File path to write to.
        content: Text content to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _real_read_text(path: Path) -> str:
    """Real implementation using Path.read_text().

    Args:
        path: File path to read from.

    Returns:
        Text content of the file.

    Raises:
        FileNotFoundError: If file does not exist.
    """
    return path.read_text(encoding="utf-8")


def _real_path_exists(path: Path) -> bool:
    """Real implementation using Path.exists().

    Args:
        path: Path to check.

    Returns:
        True if path exists, False otherwise.
    """
    return path.exists()


write_text: WriteTextProtocol = _real_write_text
read_text: ReadTextProtocol = _real_read_text
path_exists: PathExistsProtocol = _real_path_exists


# =============================================================================
# Browser Static Key Hook
# =============================================================================


class FindBestStaticByteProtocol(Protocol):
    """Protocol for finding the best static key byte.

    Matches browser.find_best_static_byte signature.
    """

    def __call__(self, raw_first_bytes: list[int], magic_first_byte: int) -> tuple[int, int]:
        """Find the static key's first byte that maximizes known signature matches.

        Args:
            raw_first_bytes: First XOR-encoded bytes from binary messages.
            magic_first_byte: ASCII value of magic key's first character.

        Returns:
            Tuple of (best_static_byte, match_count).
        """
        ...


# Default is None - browser.py uses its own implementation when None
find_best_static_byte: FindBestStaticByteProtocol | None = None


# =============================================================================
# Playwright Protocols - matching real Playwright sync API signatures
# =============================================================================


class ResponseProtocol(Protocol):
    """Protocol for Playwright Response object."""

    @property
    def status(self) -> int:
        """HTTP status code.

        Returns:
            Status code (e.g., 200, 404).
        """
        ...

    @property
    def url(self) -> str:
        """Response URL.

        Returns:
            The URL of the response.
        """
        ...


class CDPSessionProtocol(Protocol):
    """Protocol for Playwright CDPSession.

    Matches playwright.sync_api.CDPSession interface.
    """

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Send a CDP command and return the result.

        Args:
            method: CDP method name (e.g., "Network.enable").
            params: Optional parameters for the method.

        Returns:
            Response from CDP as a JSON object.
        """
        ...

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        """Register an event handler for CDP events.

        Args:
            event: CDP event name (e.g., "Network.webSocketFrameReceived").
            handler: Callback that receives event params as JSONObject.
        """
        ...

    def detach(self) -> None:
        """Detach the CDP session from the target."""
        ...


class KeyboardProtocol(Protocol):
    """Protocol for Playwright Keyboard.

    Matches playwright.sync_api.Keyboard interface for methods we use.
    """

    def press(self, key: str, *, delay: float | None = None) -> None:
        """Press a keyboard key.

        Args:
            key: Key name (e.g., "f", "Enter", "ArrowUp").
            delay: Time to wait between keydown and keyup in milliseconds.
        """
        ...

    def type(self, text: str, *, delay: float | None = None) -> None:
        """Type text character by character.

        Args:
            text: Text to type.
            delay: Time to wait between key presses in milliseconds.
        """
        ...


class PageProtocol(Protocol):
    """Protocol for Playwright Page.

    Matches playwright.sync_api.Page interface for methods we use.
    """

    @property
    def url(self) -> str:
        """Get the current URL of the page."""
        ...

    @property
    def keyboard(self) -> KeyboardProtocol:
        """Get the keyboard interface for this page.

        Returns:
            Keyboard interface for sending key events.
        """
        ...

    def goto(
        self,
        url: str,
        *,
        referer: str | None = None,
        timeout: float | None = None,
        wait_until: str | None = None,
    ) -> ResponseProtocol | None:
        """Navigate to a URL.

        Args:
            url: URL to navigate to.
            referer: Referer header value.
            timeout: Maximum operation time in milliseconds.
            wait_until: When to consider operation succeeded ("load", "domcontentloaded",
                "networkidle", "commit").

        Returns:
            Response object or None if navigation failed.
        """
        ...

    def wait_for_timeout(self, timeout: float) -> None:
        """Wait for specified timeout in milliseconds.

        Args:
            timeout: Timeout in milliseconds.
        """
        ...

    def wait_for_event(self, event: str, *, timeout: float | None = None) -> None:
        """Wait for an event to be fired.

        Args:
            event: Event name to wait for (e.g., "close").
            timeout: Maximum wait time in milliseconds.
        """
        ...

    def wait_for_function(self, expression: str, *, timeout: float | None = None) -> None:
        """Wait for a JavaScript function to return truthy value.

        Args:
            expression: JavaScript expression to evaluate.
            timeout: Maximum wait time in milliseconds.
        """
        ...

    def close(self, *, reason: str | None = None, run_before_unload: bool | None = None) -> None:
        """Close the page.

        Args:
            reason: Reason to be reported to operations interrupted by page closure.
            run_before_unload: Whether to run the before unload page handlers.
        """
        ...

    def evaluate(self, expression: str) -> JSONValue:
        """Evaluate JavaScript expression in the page context.

        Args:
            expression: JavaScript expression to evaluate.

        Returns:
            Result of the expression evaluation.
        """
        ...


class BrowserContextProtocol(Protocol):
    """Protocol for Playwright BrowserContext.

    Matches playwright.sync_api.BrowserContext interface for methods we use.
    """

    def new_page(self) -> PageProtocol:
        """Create a new page in the browser context.

        Returns:
            New page instance.
        """
        ...

    def new_cdp_session(self, page: PageProtocol) -> CDPSessionProtocol:
        """Create a new CDP session attached to the page.

        Args:
            page: Page to attach CDP session to.

        Returns:
            CDP session instance.
        """
        ...

    def close(self, *, reason: str | None = None) -> None:
        """Close the browser context.

        Args:
            reason: Reason to be reported to operations interrupted by context closure.
        """
        ...


class BrowserProtocol(Protocol):
    """Protocol for Playwright Browser.

    Matches playwright.sync_api.Browser interface for methods we use.
    """

    def new_context(self) -> BrowserContextProtocol:
        """Create a new browser context.

        Returns:
            New browser context instance.
        """
        ...

    def close(self, *, reason: str | None = None) -> None:
        """Close the browser.

        Args:
            reason: Reason to be reported to operations interrupted by browser closure.
        """
        ...


class BrowserTypeLaunchProtocol(Protocol):
    """Protocol for BrowserType.launch method."""

    def __call__(
        self,
        *,
        headless: bool | None = None,
        slow_mo: float | None = None,
        timeout: float | None = None,
    ) -> BrowserProtocol:
        """Launch a browser instance.

        Args:
            headless: Whether to run browser in headless mode. Defaults to True.
            slow_mo: Slows down operations by the specified milliseconds.
            timeout: Maximum time to wait for browser to start in milliseconds.

        Returns:
            Browser instance.
        """
        ...


class BrowserTypeProtocol(Protocol):
    """Protocol for Playwright BrowserType (e.g., playwright.chromium)."""

    def launch(
        self,
        *,
        headless: bool | None = None,
        slow_mo: float | None = None,
        timeout: float | None = None,
    ) -> BrowserProtocol:
        """Launch a browser instance.

        Args:
            headless: Whether to run browser in headless mode. Defaults to True.
            slow_mo: Slows down operations by the specified milliseconds.
            timeout: Maximum time to wait for browser to start in milliseconds.

        Returns:
            Browser instance.
        """
        ...


class PlaywrightProtocol(Protocol):
    """Protocol for Playwright instance from sync_playwright().start()."""

    @property
    def chromium(self) -> BrowserTypeProtocol:
        """Chromium browser type.

        Returns:
            BrowserType for Chromium.
        """
        ...

    def stop(self) -> None:
        """Stop the Playwright instance."""
        ...


class SyncPlaywrightContextManagerProtocol(Protocol):
    """Protocol for sync_playwright() context manager."""

    def start(self) -> PlaywrightProtocol:
        """Start Playwright and return the instance.

        Returns:
            Playwright instance.
        """
        ...

    def __enter__(self) -> PlaywrightProtocol:
        """Enter context manager.

        Returns:
            Playwright instance.
        """
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Exit context manager.

        Args:
            exc_type: Exception type if an exception was raised.
            exc_val: Exception instance if an exception was raised.
            exc_tb: Traceback if an exception was raised.
        """
        ...


class SyncPlaywrightFactoryProtocol(Protocol):
    """Protocol for sync_playwright() function."""

    def __call__(self) -> SyncPlaywrightContextManagerProtocol:
        """Create a Playwright context manager.

        Returns:
            Context manager that yields Playwright instance.
        """
        ...


# Hook for sync_playwright - tests can override
sync_playwright: SyncPlaywrightFactoryProtocol | None = None


# =============================================================================
# Terrain Map Loading Hook
# =============================================================================


class TerrainMapProtocol(Protocol):
    """Protocol for TerrainMap interface."""

    ROCK: str
    GROUND: str
    WATER: str

    def get_terrain(self, x: int, y: int) -> str:
        """Get terrain type at game coordinates.

        Args:
            x: X coordinate (0-255).
            y: Y coordinate (0-255).

        Returns:
            Terrain character: '#' for rock, '.' for ground, 'W' for water.
        """
        ...

    def is_passable(self, x: int, y: int) -> bool:
        """Check if tile is passable (not rock or water).

        Args:
            x: X coordinate (0-255).
            y: Y coordinate (0-255).

        Returns:
            True if passable, False if rock or water.
        """
        ...

    def render_viewport(
        self,
        center_x: int,
        center_y: int,
        width: int = 16,
        height: int = 16,
    ) -> list[list[str]]:
        """Render a viewport grid centered on position.

        Args:
            center_x: Center X coordinate.
            center_y: Center Y coordinate.
            width: Viewport width.
            height: Viewport height.

        Returns:
            2D list of terrain characters.
        """
        ...


class LoadTerrainMapProtocol(Protocol):
    """Protocol for loading a terrain map."""

    def __call__(self, gif_path: Path) -> TerrainMapProtocol:
        """Load terrain map from GIF file.

        Args:
            gif_path: Path to field##_r.gif minimap file.

        Returns:
            TerrainMap instance.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If image is not 256x256.
        """
        ...


def _real_load_terrain_map(gif_path: Path) -> TerrainMapProtocol:
    """Real implementation - loads TerrainMap from GIF.

    Args:
        gif_path: Path to field##_r.gif minimap file.

    Returns:
        TerrainMap instance.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If image is not 256x256.
    """
    terrain_mod = __import__("tankpit_bot.terrain", fromlist=["TerrainMap"])
    terrain_map: TerrainMapProtocol = terrain_mod.TerrainMap(gif_path)
    return terrain_map


load_terrain_map: LoadTerrainMapProtocol = _real_load_terrain_map


# =============================================================================
# Tick Loop Bot Protocol
# =============================================================================


class BotProtocol(Protocol):
    """Interface for bot command dispatch used by executor and world_sync.

    Defines the minimal set of methods these consumers need from the Bot
    class.  tick_loop.py uses Bot directly for AI state access.  Tests
    inject a FakeBot satisfying this protocol instead of mocking.
    """

    @property
    def _cdp(self) -> CDPSessionProtocol | None:
        """CDP session for browser communication.

        Returns:
            CDP session or None if not connected.
        """
        ...

    _cdp_message_buffer: list[str]

    def move_to(self, x: int, y: int) -> bool:
        """Send move command.

        Args:
            x: Target X coordinate (0-255).
            y: Target Y coordinate (0-255).

        Returns:
            True if command was sent.
        """
        ...

    def pickup_move_to(self, x: int, y: int) -> bool:
        """Send pickup move command.

        Args:
            x: Target X coordinate (0-255).
            y: Target Y coordinate (0-255).

        Returns:
            True if command was sent.
        """
        ...

    def teleport_to(self, x: int, y: int) -> bool:
        """Send teleport command.

        Args:
            x: Target X coordinate (0-255).
            y: Target Y coordinate (0-255).

        Returns:
            True if command was sent.
        """
        ...

    def shoot_at(self, x: int, y: int, target_id: int) -> bool:
        """Send shoot command.

        Args:
            x: Target X coordinate.
            y: Target Y coordinate.
            target_id: Target entity ID.

        Returns:
            True if command was sent.
        """
        ...

    def use_radar(self) -> bool:
        """Send radar scan command.

        Returns:
            True if command was sent.
        """
        ...

    def open_map(self) -> bool:
        """Send map open command to reveal global enemy positions.

        Returns:
            True if command was sent.
        """
        ...

    def enable_equipment(self, slot: int) -> bool:
        """Enable equipment slot if not already enabled.

        Args:
            slot: Equipment slot (1-5).

        Returns:
            True if command was sent.
        """
        ...

    def disable_equipment(self, slot: int) -> bool:
        """Disable equipment slot if currently enabled.

        Args:
            slot: Equipment slot (1-5).

        Returns:
            True if command was sent.
        """
        ...

    def _has_equipment_stock(self, slot: int) -> bool:
        """Check if equipment slot has remaining stock.

        Args:
            slot: Equipment slot (1-5).

        Returns:
            True if equipment is available to use.
        """
        ...


# =============================================================================
# CLI Argument Hook
# =============================================================================


def _real_get_argv() -> list[str]:
    """Real implementation - returns sys.argv.

    Returns:
        The command line arguments.
    """
    import sys

    return sys.argv


get_argv: Callable[[], list[str]] = _real_get_argv


def _real_get_sync_playwright() -> SyncPlaywrightFactoryProtocol:
    """Real implementation - imports playwright.

    Returns:
        The sync_playwright factory from playwright.sync_api.
    """
    pw_module = __import__("playwright.sync_api", fromlist=["sync_playwright"])
    real_sync_playwright: SyncPlaywrightFactoryProtocol = pw_module.sync_playwright
    return real_sync_playwright


# Hook for getting playwright - tests can override
get_sync_playwright: Callable[[], SyncPlaywrightFactoryProtocol] = _real_get_sync_playwright


__all__ = [
    "BotProtocol",
    "BrowserContextProtocol",
    "BrowserProtocol",
    "BrowserTypeLaunchProtocol",
    "BrowserTypeProtocol",
    "CDPSessionProtocol",
    "FindBestStaticByteProtocol",
    "KeyboardProtocol",
    "LoadTerrainMapProtocol",
    "PageProtocol",
    "PathExistsProtocol",
    "PlaywrightProtocol",
    "ReadTextProtocol",
    "ResponseProtocol",
    "SyncPlaywrightContextManagerProtocol",
    "SyncPlaywrightFactoryProtocol",
    "TerrainMapProtocol",
    "WriteTextProtocol",
    "find_best_static_byte",
    "get_argv",
    "get_env",
    "get_sync_playwright",
    "load_terrain_map",
    "path_exists",
    "read_text",
    "sync_playwright",
    "write_text",
]
