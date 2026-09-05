"""Chrome launch flags and window sizing for a streaming session.

The display args a screencast session needs, the no-viewport variant,
and the CDP maximize call. Shared by the sniffer and the bot.
"""

from __future__ import annotations

from platform_core.json_utils import require_int
from platform_core.logging import get_logger

from tankpit_bot import _test_hooks

log = get_logger(__name__)


def _chrome_stream_display_args() -> list[str]:
    """Build Chromium args positioning the browser on the streamed virtual display.

    Reads SUNSHINE_STREAM_DISPLAY_X/Y/W/H env vars set by Vibeshine's
    ``run_command`` handler when a WebRTC session forks a launcher-button
    command (see ``vibeshine/src/webrtc_stream.cpp::launch_run_command_on_user_desktop``).
    When set, they carry the virtual monitor's position and size in the
    Windows virtual-desktop coordinate space; ``--window-position``
    pins the browser onto that display and ``--start-maximized`` fills
    the display's work area.

    Returns an empty list when any var is missing or unparseable — Chrome
    falls back to its last-used or default position.
    """
    keys = (
        "SUNSHINE_STREAM_DISPLAY_X",
        "SUNSHINE_STREAM_DISPLAY_Y",
        "SUNSHINE_STREAM_DISPLAY_W",
        "SUNSHINE_STREAM_DISPLAY_H",
    )
    values: list[int] = []
    for k in keys:
        raw = _test_hooks.get_env(k)
        if raw is None:
            return []
        try:
            values.append(int(raw))
        except ValueError:
            # A non-numeric override disables the stream geometry rather
            # than crashing the session; say so, or the operator's typo
            # looks like the flags were never wired.
            log.warning("stream display env %s is not an integer: %r", k, raw)
            return []
    x, y, w, h = values
    if w <= 0 or h <= 0:
        return []
    # KEY: use --window-size, NOT --start-maximized.
    #
    # Playwright issue microsoft/playwright#14314 (confirmed against
    # tankpit-sniff 2026-07-10): when both --window-position and
    # --start-maximized are passed, Chromium opens at the position
    # but skips the maximize entirely — the window comes up at the
    # default 800x600. Selenium doesn't have this bug; Playwright's
    # Chromium wrapping does. --window-size=W,H sized to the display
    # rect sidesteps the bug: the window is the same visual result
    # as maximized (fills the display) without hitting the
    # position+maximized interaction.
    #
    # ``no_viewport=True`` on ``new_context()`` (below) prevents
    # Playwright's default 1280x720 viewport from clamping the
    # content down inside the correctly-sized window.
    return [
        f"--window-position={x},{y}",
        f"--window-size={w},{h}",
    ]


def _maximize_via_cdp(cdp: _test_hooks.CDPSessionProtocol) -> None:
    """Flip the current window to the OS-level maximised state via CDP.

    ``--window-position=X,Y --window-size=W,H`` (see
    ``_chrome_stream_display_args``) puts Chromium on the streamed
    display at the correct dimensions, but the WINDOW STATE is still
    "normal" (a floating window that happens to be display-sized),
    not "maximized" (the OS-recognised maximised state). User report
    2026-07-10: "it was large didnt use the official maximize
    function".

    ``Browser.setWindowBounds`` with ``bounds.windowState =
    "maximized"`` is Chrome DevTools Protocol's supported way to
    trigger the actual maximise. ``--start-maximized`` would do the
    same but conflicts with ``--window-position`` (Playwright issue
    microsoft/playwright#14314). Post-launch CDP sidesteps that
    conflict.

    Args:
        cdp: The active CDP session attached to the target Chromium
            page.

    Raises:
        JSONTypeError: When the CDP surface returns a response missing
            or with the wrong type for ``windowId``. The failure
            propagates rather than being softened — a Playwright /
            Chromium version that renamed or reshaped this surface
            should surface as a loud sniff failure so we notice the
            drift, not silently degrade to a non-maximised window.
    """
    window = cdp.send("Browser.getWindowForTarget")
    window_id = require_int(window, "windowId")
    cdp.send(
        "Browser.setWindowBounds",
        {"windowId": window_id, "bounds": {"windowState": "maximized"}},
    )


def _chrome_stream_no_viewport() -> bool:
    """True when the launch is targeting the streamed virtual display.

    Playwright's ``browser.new_context()`` defaults to a fixed
    1280x720 viewport that OVERRIDES Chromium's ``--start-maximized``
    (2026-07-10 user report: browser opened at the right position
    but the content clamped to 1280x720). Passing
    ``no_viewport=True`` on the context disables that clamp and lets
    the natural (maximized) window size drive the viewport.

    Gated on the same SUNSHINE_STREAM_DISPLAY_* env vars as the
    Chromium args so a local ``make sniff`` (no env vars) keeps the
    stable 1280x720 test viewport it always had.
    """
    return bool(_chrome_stream_display_args())


__all__ = [
    "_chrome_stream_display_args",
    "_chrome_stream_no_viewport",
    "_maximize_via_cdp",
]
