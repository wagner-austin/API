"""Tests for combat screenshot capture against a real headless browser.

These exercise the real CDP ``Page.captureScreenshot`` path end to end:
a genuine headless Chromium renders a page, the production capture code
runs against the real CDP session, and the result is written through the
real binary filesystem hook to a temp directory. No fakes, mocks, or
recorded substitutes -- the only inputs are a real browser and real PNG
bytes.
"""

from __future__ import annotations

from pathlib import Path

from tankpit_bot._test_hooks import CDPSessionProtocol
from tankpit_bot.diagnostics.combat_screenshot import (
    capture_screenshot_png,
    save_screenshot,
)

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_capture_screenshot_png_returns_real_png_bytes(
    live_cdp: CDPSessionProtocol,
) -> None:
    """capture_screenshot_png returns decoded PNG bytes from a real canvas."""
    png = capture_screenshot_png(live_cdp)

    assert png.startswith(_PNG_MAGIC)
    assert len(png) > len(_PNG_MAGIC)


def test_save_screenshot_writes_real_png_file(
    live_cdp: CDPSessionProtocol,
    tmp_path: Path,
) -> None:
    """save_screenshot writes a real PNG to ``directory/label.png``."""
    result = save_screenshot(live_cdp, tmp_path, "shot_0001_x34_y96_id517")

    expected = tmp_path / "shot_0001_x34_y96_id517.png"
    assert result == expected
    assert expected.read_bytes().startswith(_PNG_MAGIC)
