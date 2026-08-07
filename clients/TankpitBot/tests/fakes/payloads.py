"""Wire payloads and page-runtime results the fakes replay.

The canned auth payload, the injected-WebSocket body decoder, and the
``Runtime.evaluate`` results a fake page returns. Shared by the page
and CDP fakes.
"""

from __future__ import annotations

import base64
import re

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
)

from tankpit_bot.protocol.framing import (
    decode_frame,
    encode_frame,
)

_FAKE_TPCLIENT_URL = "https://tankpit.com/game/tpclient-test.js"


_FAKE_STATIC_KEY = "A" * 1000


_FAKE_MAGIC = "test_magic_12345678"


def _make_auth_payload(magic: str) -> str:
    """Create a base64-encoded AUTH payload containing the magic key.

    The AUTH message format is:
    - 2-byte length prefix (little-endian)
    - Text body: %AUTH !be <session>|<hash>|<ts> <magic>

    Args:
        magic: The magic key to include in the AUTH payload.

    Returns:
        Base64-encoded AUTH payload string.
    """
    body = f"%AUTH !be test_session|test_hash|12345 {magic}"
    body_bytes = body.encode("utf-8")
    length_prefix = len(body_bytes).to_bytes(2, "little")
    return base64.b64encode(length_prefix + body_bytes).decode("ascii")


_WEBSOCKET_INJECTION_PATTERN = re.compile(r"atob\('([^']+)'\)")


def _build_captured_raw_messages(
    selected_room: str | None,
    entered_room: str | None = None,
) -> list[JSONValue]:
    """Build the synthetic raw-message buffer used by protocol join tests.

    Args:
        selected_room: Joined room ID, if any.
        entered_room: Entered room ID, if any.

    Returns:
        Base64-encoded framed room-discovery payloads.
    """
    payloads: list[JSONValue] = [
        base64.b64encode(
            encode_frame(b"+4|World (President Trump)|24|5,1,0,0,0,0,0|2|n|field24.gif|2026")
        ).decode("utf-8"),
        base64.b64encode(encode_frame(b"+1|Practice|1|0,0,0,0,0,0,0|2|p|field01.gif|2026")).decode(
            "utf-8"
        ),
    ]
    if selected_room is not None:
        confirm = f"={selected_room}|Sep. 25, 2012|Artax|4|9|9|9|9".encode()
        payloads.append(base64.b64encode(encode_frame(confirm)).decode("utf-8"))
    if entered_room is not None:
        response = f"${entered_room}|0".encode()
        payloads.append(base64.b64encode(encode_frame(response)).decode("utf-8"))
    return payloads


def _extract_injected_websocket_payload_data(expression: str) -> str | None:
    """Return the base64 websocket payload injected by the runtime helper.

    Args:
        expression: JavaScript passed to ``Runtime.evaluate``.

    Returns:
        Base64 payload string, or ``None`` when the helper is not used.
    """
    if "window.__capturedWS" not in expression or "atob('" not in expression:
        return None
    match = _WEBSOCKET_INJECTION_PATTERN.search(expression)
    if match is None:
        raise ValueError(f"missing websocket payload in expression: {expression}")
    return match.group(1)


def _extract_enter_room_id(body: bytes) -> str:
    """Return the room ID from a ``+room|...`` enter packet body.

    Args:
        body: Decoded websocket body beginning with ``+``.

    Returns:
        Room ID portion of the enter packet.
    """
    parts = body[1:].split(b"|", 4)
    if len(parts) != 5:
        raise ValueError(f"unexpected room enter payload: {body!r}")
    return parts[0].decode("utf-8")


def _runtime_raw_messages_result(
    expression: str,
    *,
    raw_messages_ready: bool,
    selected_room: str | None,
    entered_room: str | None,
) -> JSONObject | None:
    """Return the fake ``window.__rawMsgs`` snapshot response."""
    if "window.__rawMsgs" not in expression:
        return None
    if not raw_messages_ready:
        return {"result": {"value": []}}
    return {
        "result": {
            "value": _build_captured_raw_messages(
                selected_room,
                entered_room,
            )
        }
    }


_PAGE_CLIENT_SNAPSHOT_VALUE: JSONObject = {
    "timestamp_ms": 1000,
    "client_present": True,
    "map_visible": False,
    "client_state": 1,
    "client_busy": False,
    "pending_actions": 0,
    "heartbeat_age_ms": 50,
    "last_page_client_send_age_ms": 100,
    "last_bot_send_age_ms": 100,
    "ws_ready_state": 1,
    "current_send_label": None,
    "sent_frame_meta_queue_length": 0,
    "self_fields": {},
    "world_fields": {},
    "map_fields": {},
    "world_collections": {},
}


def _runtime_metadata_result(expression: str) -> JSONObject | None:
    """Return fake metadata/script lookup responses for login helpers."""
    if "MAX_DEPTH" in expression:
        # Structure survey walk: the real page returns null until the
        # game object exists; the fake models the pre-capture state.
        return {"result": {"value": None}}
    if "tankpit.magic" in expression:
        return {"result": {"value": _FAKE_MAGIC}}
    if "script[src]" in expression and "tpclient" in expression:
        return {"result": {"value": _FAKE_TPCLIENT_URL}}
    if "fetch(" in expression and "tpclient-test.js" in expression:
        return {"result": {"value": f'window.fakeTpclientKey="{_FAKE_STATIC_KEY}";'}}
    if "window.__tankpitActiveGame" in expression and "map_visible" in expression:
        return {"result": {"value": _PAGE_CLIENT_SNAPSHOT_VALUE}}
    return None


def _decode_injected_websocket_body(expression: str) -> bytes | None:
    """Decode the browser helper's injected websocket payload.

    Args:
        expression: JavaScript passed to ``Runtime.evaluate``.

    Returns:
        The decoded framed-message body for framed payloads, the raw byte
        payload for non-framed sends, or ``None`` when the expression does not
        invoke the websocket helper.
    """
    if "window.__capturedWS" not in expression or "atob('" not in expression:
        return None
    match = _WEBSOCKET_INJECTION_PATTERN.search(expression)
    if match is None:
        raise ValueError(f"missing websocket payload in expression: {expression}")
    framed = base64.b64decode(match.group(1))
    if len(framed) < 2:
        return framed
    expected_total = 2 + int.from_bytes(framed[:2], "little")
    if expected_total != len(framed):
        return framed
    body, remaining = decode_frame(framed)
    if remaining:
        raise ValueError(f"unexpected trailing framed data: {remaining.hex()}")
    return body
