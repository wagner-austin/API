"""Tests for the bot's opt-in shot-screenshot capture wiring.

The capture branch runs against a real headless-browser CDP session,
writing a real PNG to a temp directory; the opt-in directory is supplied
through the production ``get_env`` hook (the sanctioned dependency-
injection point, never a fake object). The no-op branches (variable
unset, or no CDP attached) need no browser. No fakes, mocks, or
monkeypatching.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from pathlib import Path

import pytest

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import CDPSessionProtocol
from tankpit_bot.bot import Bot

_SHOT_ENV = "TANKPIT_SHOT_SCREENSHOTS"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _env_pointing_to(directory: str | None) -> Callable[[str], str | None]:
    """Build a get_env implementation reporting the screenshot directory.

    Args:
        directory: Value to report for the screenshot env var, or ``None``
            to model the variable being unset.

    Returns:
        A get_env callable returning ``directory`` for the screenshot key
        and ``None`` for every other key.
    """

    def _get(key: str) -> str | None:
        return directory if key == _SHOT_ENV else None

    return _get


@pytest.fixture()
def shot_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Point the shot-screenshot env hook at a temp directory.

    Rebinds the production ``get_env`` hook to report ``tmp_path`` for the
    screenshot variable and restores the prior hook afterwards.

    Yields:
        The temp directory configured as the screenshot output dir.
    """
    previous = _test_hooks.get_env
    _test_hooks.get_env = _env_pointing_to(str(tmp_path))
    try:
        yield tmp_path
    finally:
        _test_hooks.get_env = previous


@pytest.fixture()
def no_shot_env() -> Generator[None, None, None]:
    """Bind the env hook so the shot-screenshot variable reads as unset.

    Yields:
        None; restores the prior hook afterwards.
    """
    previous = _test_hooks.get_env
    _test_hooks.get_env = _env_pointing_to(None)
    try:
        yield
    finally:
        _test_hooks.get_env = previous


def _bot() -> Bot:
    """Build a headless bot instance for screenshot wiring tests.

    Returns:
        A fresh ``Bot`` with no CDP session attached yet.
    """
    return Bot("https://test.tankpit.com/", headless=True)


def test_capture_writes_named_png_when_enabled(
    live_cdp: CDPSessionProtocol,
    shot_dir: Path,
) -> None:
    """With the env set and a CDP attached, one named PNG is written."""
    bot = _bot()
    bot._cdp = live_cdp

    bot._capture_shot_screenshot(34, 96, 517)

    written = shot_dir / "shot_0001_x34_y96_id517.png"
    assert written.read_bytes().startswith(_PNG_MAGIC)


def test_sequence_increments_per_shot(
    live_cdp: CDPSessionProtocol,
    shot_dir: Path,
) -> None:
    """The sequence counter advances so each shot gets a distinct file."""
    bot = _bot()
    bot._cdp = live_cdp

    bot._capture_shot_screenshot(1, 2, 3)
    bot._capture_shot_screenshot(4, 5, 6)

    assert (shot_dir / "shot_0001_x1_y2_id3.png").read_bytes().startswith(_PNG_MAGIC)
    assert (shot_dir / "shot_0002_x4_y5_id6.png").read_bytes().startswith(_PNG_MAGIC)


def test_capture_is_noop_when_env_unset(no_shot_env: None) -> None:
    """No screenshot is taken when the opt-in env var is absent."""
    bot = _bot()

    bot._capture_shot_screenshot(1, 2, 3)

    assert bot._shot_screenshot_seq == 0


def test_capture_is_noop_when_cdp_absent(shot_dir: Path) -> None:
    """No screenshot is taken when no CDP session is attached."""
    bot = _bot()
    bot._cdp = None

    bot._capture_shot_screenshot(1, 2, 3)

    assert bot._shot_screenshot_seq == 0
    assert list(shot_dir.glob("*.png")) == []
