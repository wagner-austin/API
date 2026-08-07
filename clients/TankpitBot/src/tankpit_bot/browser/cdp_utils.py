"""CDP session utility functions — standalone helpers for CDP interaction.

These functions operate on a CDPSessionProtocol and don't require a class.
Extracted from session.py to break circular imports and reduce file size.
"""

from __future__ import annotations

import base64
import re
from typing import TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    optional_str,
    require_dict,
    require_list,
    require_str,
)
from platform_core.logging import get_logger

from tankpit_bot._test_hooks import CDPSessionProtocol
from tankpit_bot.types.literals import SentFrameOrigin, require_sent_frame_origin

log = get_logger(__name__)

_BASE64_PATTERN = re.compile(r"^[A-Za-z0-9+/]*={0,2}$")


def _is_valid_base64(payload: str) -> bool:
    """Check if payload is valid base64.

    Args:
        payload: String to validate.

    Returns:
        True if valid base64, False otherwise.
    """
    if not payload:
        return False
    if not _BASE64_PATTERN.match(payload):
        return False
    return len(payload) % 4 == 0


def _extract_runtime_value(result: JSONObject) -> JSONValue:
    """Return the Runtime.evaluate value field.

    Args:
        result: Raw CDP result object.

    Returns:
        The evaluated JavaScript value.

    Raises:
        ValueError: If the CDP result is missing the value field.
    """
    result_obj = require_dict(result, "result")
    if "value" not in result_obj:
        raise ValueError(f"Runtime.evaluate result missing value: {result_obj}")
    return result_obj["value"]


class SentFrameMetadata(TypedDict):
    """Metadata captured at outbound WebSocket send time.

    Attributes:
        origin: Whether the send came from bot injection or the page client.
        label: Bot-side send label when known.
        stack: JavaScript stack recorded at send time.
    """

    origin: SentFrameOrigin
    label: str
    stack: str


def _pop_sent_frame_metadata(cdp: CDPSessionProtocol) -> SentFrameMetadata | None:
    """Pop the next outbound frame metadata record from the browser hook queue.

    Args:
        cdp: Active CDP session.

    Returns:
        The next queued outbound metadata record, or None when unavailable.

    Raises:
        ValueError: If the hook returned malformed metadata.
    """
    result = cdp.send(
        "Runtime.evaluate",
        {
            "expression": """
            (() => {
                if (!Array.isArray(window.__sentFrameMetaQueue)) {
                    return null;
                }
                if (window.__sentFrameMetaQueue.length === 0) {
                    return null;
                }
                return window.__sentFrameMetaQueue.shift();
            })()
            """,
            "returnByValue": True,
        },
    )
    raw_value = _extract_runtime_value(result)
    if raw_value is None or raw_value == "":
        return None
    metadata_obj = require_dict({"metadata": raw_value}, "metadata")
    return SentFrameMetadata(
        origin=require_sent_frame_origin(metadata_obj, "origin"),
        label=optional_str(metadata_obj, "label") or "",
        stack=optional_str(metadata_obj, "stack") or "",
    )


def get_captured_raw_messages(cdp: CDPSessionProtocol) -> list[str]:
    """Return the captured raw WebSocket message buffer from the page hook.

    Args:
        cdp: Active CDP session.

    Returns:
        Captured raw message payloads as base64 strings.

    Raises:
        ValueError: If the page hook returned malformed data.
    """
    result = cdp.send(
        "Runtime.evaluate",
        {
            "expression": """
            (() => Array.isArray(window.__rawMsgs) ? window.__rawMsgs.slice(-500) : [])()
            """,
            "returnByValue": True,
        },
    )
    raw_value = _extract_runtime_value(result)
    payloads_raw = require_list({"items": raw_value}, "items")
    payloads: list[str] = []
    for payload in payloads_raw:
        payloads.append(require_str({"payload": payload}, "payload"))
    return payloads


def send_websocket_bytes(
    cdp: CDPSessionProtocol,
    data: bytes,
    label: str = "direct_send",
) -> str:
    """Send raw bytes via the captured WebSocket.

    Args:
        cdp: Active CDP session.
        data: Raw framed bytes to send.
        label: Bot-side label for outbound provenance logging.

    Returns:
        Status string returned by the browser-side send helper.
    """
    b64 = base64.b64encode(data).decode()
    send_js = """
    (() => {
        let ws = window.__capturedWS;
        if (!ws && typeof tankpit !== 'undefined' && tankpit.ws) {
            ws = tankpit.ws;
        }
        if (!ws && typeof window.ws !== 'undefined') {
            ws = window.ws;
        }
        if (!ws) {
            const status = window.__capturedWS ? 'exists' : 'null';
            return 'NO_WEBSOCKET_FOUND (__capturedWS=' + status + ')';
        }
        if (ws.readyState !== 1) {
            return 'WEBSOCKET_NOT_OPEN: ' + ws.readyState;
        }
        const binary = atob('%B64%');
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {
            bytes[i] = binary.charCodeAt(i);
        }
        window.__codexCurrentSendLabel = %LABEL%;
        try {
            ws.send(bytes.buffer);
        } finally {
            window.__codexCurrentSendLabel = null;
        }
        return 'SENT_' + bytes.length + '_BYTES via ' + ws.url;
    })()
    """
    send_js = send_js.replace("%B64%", b64)
    send_js = send_js.replace("%LABEL%", repr(label))
    result = cdp.send("Runtime.evaluate", {"expression": send_js, "returnByValue": True})
    runtime_value = _extract_runtime_value(result)
    return require_str({"value": runtime_value}, "value")


def get_current_time_ms() -> int:
    """Get current time in milliseconds via the :mod:`_test_hooks` clock.

    Delegates to :data:`_test_hooks.get_current_time_ms` so scenario
    tests can replace the clock via save-and-restore on that
    attribute without monkey-patching :mod:`time` globally. Production
    use is the real wall-clock; tests get controlled time.

    Returns:
        Current Unix timestamp in milliseconds, or the
        scenario-controlled value when a test has installed a fake
        clock on the hook.
    """
    from tankpit_bot import _test_hooks

    return _test_hooks.get_current_time_ms()


class CDPClock:
    """Anchors one session's CDP monotonic clock to Unix milliseconds.

    CDP timestamps are monotonic seconds from an arbitrary origin, so
    the first frame of a session fixes the offset every later frame is
    read against. That anchor belongs to ONE browser session — a second
    session in the same process has a different origin and would read
    every timestamp through the first session's offset
    ([[session-state-deglobalisation]] step 4). A fresh instance is
    unanchored, which is why no reset function exists.
    """

    def __init__(self) -> None:
        self._offset_ms: int | None = None

    def to_unix_ms(self, timestamp: float) -> int:
        """Convert a CDP monotonic timestamp to Unix milliseconds.

        The first call anchors the offset from the wall clock; later
        calls reuse it, so frames stay ordered relative to each other.

        Args:
            timestamp: CDP monotonic timestamp in seconds.

        Returns:
            Unix timestamp in milliseconds.
        """
        cdp_ms = int(timestamp * 1000)
        if self._offset_ms is None:
            self._offset_ms = get_current_time_ms() - cdp_ms
        return cdp_ms + self._offset_ms


__all__ = [
    "CDPClock",
    "SentFrameMetadata",
    "_extract_runtime_value",
    "_is_valid_base64",
    "_pop_sent_frame_metadata",
    "get_captured_raw_messages",
    "get_current_time_ms",
    "send_websocket_bytes",
]
