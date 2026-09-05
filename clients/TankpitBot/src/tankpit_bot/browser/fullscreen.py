"""Entering the game client's OWN fullscreen, for the display capture.

The tankpit page centres a fixed-size game area inside the window, so
a captured frame carries dead margin around the picture. The client
ships the fix itself: a "Toggle Fullscreen" settings button whose
handler fullscreens the game element and re-lays the canvases to fill
it (``tpclient.js`` ``pc``/``Xa``). This module presses that button —
the client's own path, not a reimplementation of it.

Pressed via ``Input.dispatchMouseEvent`` rather than injected JS
because ``requestFullscreen`` demands a user gesture: a script call
from ``Runtime.evaluate`` has no transient activation and is refused,
while CDP input events are trusted and carry one. The coordinates come
from the button's own bounding rect, read immediately before the
click.

Nothing here affects control: the bot drives the tank over the
WebSocket wire, so the page's layout is only ever the picture.
"""

from __future__ import annotations

from platform_core.json_utils import narrow_json_to_dict, require_float
from platform_core.logging import get_logger

from tankpit_bot._test_hooks import CDPSessionProtocol
from tankpit_bot.browser.cdp_utils import _extract_runtime_value

log = get_logger(__name__)

FULLSCREEN_TOGGLE_SELECTOR = '#settings-bar span[data-tooltip="Toggle Fullscreen"]'
"""The client's own fullscreen button (``tpclient.js`` builds it with
exactly this tooltip). A selector, not a coordinate: the settings bar
moves with layout and the rect is read fresh at click time."""

_LOCATE_EXPRESSION = f"""
(() => {{
    const el = document.querySelector('{FULLSCREEN_TOGGLE_SELECTOR}');
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return {{ x: r.x + r.width / 2, y: r.y + r.height / 2 }};
}})()
"""


class FullscreenToggleMissingError(RuntimeError):
    """The client page has no fullscreen button where one is required.

    Loud on purpose: the selector encodes an assumption about
    ``tpclient.js``'s DOM, and a client update that moves the button
    should fail the streamed session's bootstrap visibly rather than
    quietly shipping a letterboxed public stream again.
    """


def enter_game_fullscreen(cdp: CDPSessionProtocol) -> None:
    """Click the client's fullscreen toggle with a trusted input event.

    Args:
        cdp: Active CDP session attached to the live tankpit page,
            with the game UI already built (post game-ready).

    Raises:
        FullscreenToggleMissingError: The button is not in the DOM —
            the client's structure drifted from the pinned selector.
        ValueError: The CDP evaluate response omits its value field.
        JSONTypeError: The located rect is not the expected shape.
    """
    result = cdp.send(
        "Runtime.evaluate",
        {"expression": _LOCATE_EXPRESSION, "returnByValue": True},
    )
    raw_value = _extract_runtime_value(result)
    if raw_value is None:
        raise FullscreenToggleMissingError(
            f"no element matches {FULLSCREEN_TOGGLE_SELECTOR!r}; the client's"
            " settings bar has drifted from the pinned structure"
        )
    point = narrow_json_to_dict(raw_value)
    x = require_float(point, "x")
    y = require_float(point, "y")
    for event_type in ("mousePressed", "mouseReleased"):
        cdp.send(
            "Input.dispatchMouseEvent",
            {
                "type": event_type,
                "x": x,
                "y": y,
                "button": "left",
                "clickCount": 1,
            },
        )
    log.info("Capture: clicked the client's fullscreen toggle at (%.0f, %.0f)", x, y)


__all__ = [
    "FULLSCREEN_TOGGLE_SELECTOR",
    "FullscreenToggleMissingError",
    "enter_game_fullscreen",
]
