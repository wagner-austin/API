"""Capture page screenshots via CDP for visual combat ground truth.

The decoded telemetry (registry locals, weapon bytes, damage tiers) is
noisy enough that a wrong conclusion about why a shot missed is easy to
reach. A PNG of the canvas at the instant the bot fires is the one
artifact that cannot be misread: it shows where the bot is, where the
enemy is, and what the screen looked like when the shot resolved.

Capture is a single CDP ``Page.captureScreenshot`` call returning a
base64 PNG, decoded to bytes and written through the binary filesystem
hook so the write target is injectable. No fallback: a malformed CDP
response raises immediately via ``require_str``.
"""

from __future__ import annotations

import base64
from pathlib import Path

from platform_core.json_utils import require_str

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import CDPSessionProtocol


def capture_screenshot_png(cdp: CDPSessionProtocol) -> bytes:
    """Capture the live page as PNG bytes via CDP.

    Args:
        cdp: Active CDP session attached to the live tankpit page.

    Returns:
        Raw PNG image bytes of the current canvas.

    Raises:
        JSONTypeError: If the CDP response omits the base64 ``data`` field.
    """
    result = cdp.send("Page.captureScreenshot", {"format": "png"})
    encoded = require_str(result, "data")
    return base64.b64decode(encoded)


def save_screenshot(cdp: CDPSessionProtocol, directory: Path, label: str) -> Path:
    """Capture and write a PNG screenshot under ``directory``.

    Args:
        cdp: Active CDP session attached to the live tankpit page.
        directory: Output directory; created if absent by the write hook.
        label: File stem (without extension) identifying the capture.

    Returns:
        Path of the written PNG file.

    Raises:
        JSONTypeError: If the CDP response omits the base64 ``data`` field.
    """
    png = capture_screenshot_png(cdp)
    path = directory / f"{label}.png"
    _test_hooks.write_bytes(path, png)
    return path


__all__ = [
    "capture_screenshot_png",
    "save_screenshot",
]
