"""Room discovery and join protocol for Tankpit lobby.

Handles the WebSocket-based room list → select → confirm → enter flow
after login completes. Split from ``login.py`` which keeps authentication.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import CDPSessionProtocol, RoomJoinPageProtocol
from tankpit_bot.browser.cdp_helpers import (
    decode_captured_body,
    get_magic_key,
    get_tpclient_url,
    load_tpclient_static_key,
)
from tankpit_bot.browser.cdp_utils import get_captured_raw_messages, send_websocket_bytes
from tankpit_bot.parser import RoomInfo, is_room_info_text
from tankpit_bot.parser_messages import parse_room_info
from tankpit_bot.protocol.codec import ProtocolCodec
from tankpit_bot.protocol.lobby import (
    ROOM_ENTRY_DEFAULT_X,
    ROOM_ENTRY_DEFAULT_Y,
    RoomEnterRequestDict,
    RoomSelectRequestDict,
    build_room_enter_metadata,
    serialize_room_enter_request,
    serialize_room_select_request,
)
from tankpit_bot.sniffer.world_service import WorldService

log = get_logger(__name__)

_ROOM_DISCOVERY_TIMEOUT_MS = 10000
_JOIN_CONFIRM_TIMEOUT_MS = 10000
_ROOM_ENTER_TIMEOUT_MS = 10000
_JOIN_POLL_INTERVAL_MS = 100.0


def _collect_room_entries(cdp: CDPSessionProtocol) -> list[RoomInfo]:
    """Return room metadata decoded from captured ROOM_LIST messages.

    Args:
        cdp: Active CDP session.

    Returns:
        Ordered room entries decoded from ROOM_LIST traffic.
    """
    entries: list[RoomInfo] = []
    for payload in get_captured_raw_messages(cdp):
        body = decode_captured_body(payload)
        if not body or body[0] != ord("+"):
            continue
        text = body.decode("utf-8")
        if not is_room_info_text(text[1:]):
            continue
        entries.append(parse_room_info(text[1:]))
    return entries


def _register_room_entries(ws: WorldService, entries: list[RoomInfo]) -> None:
    """Register discovered room images for later terrain-map loading.

    Args:
        entries: Room entries decoded from ROOM_LIST messages.
    """

    for entry in entries:
        ws.register_room_image(entry["room_id"], entry["image"])


def _resolve_room_entry(
    entries: list[RoomInfo],
    room_name: str,
) -> RoomInfo | None:
    """Resolve the desired room entry from decoded room metadata.

    Args:
        entries: Room entries decoded from ROOM_LIST messages.
        room_name: Desired room name from configuration.

    Returns:
        Matching room entry, or ``None`` if no room matches.
    """
    normalized_target = room_name.strip().lower()
    prefix_match: RoomInfo | None = None
    for entry in entries:
        normalized_candidate = entry["name"].strip().lower()
        if normalized_candidate == normalized_target:
            return entry
        if normalized_candidate.startswith(
            normalized_target + " "
        ) or normalized_candidate.startswith(normalized_target + "("):
            prefix_match = entry
    return prefix_match


def _resolve_room_id(
    entries: list[RoomInfo],
    room_name: str,
) -> str | None:
    """Resolve the desired room ID from decoded room metadata.

    Args:
        entries: Room entries decoded from ROOM_LIST messages.
        room_name: Desired room name from configuration.

    Returns:
        Matching room ID, or ``None`` if no room matches.
    """
    entry = _resolve_room_entry(entries, room_name)
    if entry is None:
        return None
    return entry["room_id"]


def _wait_for_room_entry(
    page: RoomJoinPageProtocol,
    cdp: CDPSessionProtocol,
    ws: WorldService,
    room_name: str,
) -> RoomInfo | None:
    """Wait for the desired room entry to appear in captured ROOM_LIST traffic.

    Args:
        page: Playwright page used for polling delays.
        cdp: Active CDP session.
        ws: The session's world service; room beliefs land here.
        room_name: Desired room name from configuration.

    Returns:
        Matching room entry, or ``None`` if the room never appears.
    """
    waited_ms = 0
    while waited_ms < _ROOM_DISCOVERY_TIMEOUT_MS:
        entries = _collect_room_entries(cdp)
        _register_room_entries(ws, entries)
        room_entry = _resolve_room_entry(entries, room_name)
        if room_entry is not None:
            return room_entry
        page.wait_for_timeout(_JOIN_POLL_INTERVAL_MS)
        waited_ms += int(_JOIN_POLL_INTERVAL_MS)
    return None


def _wait_for_room_id(
    page: RoomJoinPageProtocol,
    cdp: CDPSessionProtocol,
    ws: WorldService,
    room_name: str,
) -> str | None:
    """Wait for the desired room to appear in captured ROOM_LIST traffic.

    Args:
        page: Playwright page used for polling delays.
        cdp: Active CDP session.
        ws: The session's world service; room images land here.
        room_name: Desired room name from configuration.

    Returns:
        Matching room ID, or ``None`` if the room never appears.
    """
    room_entry = _wait_for_room_entry(page, cdp, ws, room_name)
    if room_entry is None:
        return None
    return room_entry["room_id"]


def _has_join_confirm(cdp: CDPSessionProtocol, room_id: str, *, start_index: int = 0) -> bool:
    """Return whether a JOIN_CONFIRM for the selected room was captured.

    Args:
        cdp: Active CDP session.
        room_id: Expected joined room ID.
        start_index: Raw-message index where matching should begin.

    Returns:
        True when a matching JOIN_CONFIRM has been captured.
    """
    expected_prefix = f"={room_id}|"
    payloads = get_captured_raw_messages(cdp)
    for payload in payloads[start_index:]:
        body = decode_captured_body(payload)
        if not body or body[0] != ord("="):
            continue
        if body.decode("utf-8").startswith(expected_prefix):
            return True
    return False


def _wait_for_join_confirm(
    page: RoomJoinPageProtocol,
    cdp: CDPSessionProtocol,
    room_id: str,
    *,
    start_index: int = 0,
) -> bool:
    """Wait for a JOIN_CONFIRM message for the selected room.

    Args:
        page: Playwright page used for polling delays.
        cdp: Active CDP session.
        room_id: Selected room ID.
        start_index: Raw-message index where matching should begin.

    Returns:
        True when the matching JOIN_CONFIRM arrives, False on timeout.
    """
    waited_ms = 0
    while waited_ms < _JOIN_CONFIRM_TIMEOUT_MS:
        if _has_join_confirm(cdp, room_id, start_index=start_index):
            return True
        page.wait_for_timeout(_JOIN_POLL_INTERVAL_MS)
        waited_ms += int(_JOIN_POLL_INTERVAL_MS)
    return False


def _has_enter_response(cdp: CDPSessionProtocol, room_id: str, *, start_index: int = 0) -> bool:
    """Return whether an enter response for the selected room was captured.

    Args:
        cdp: Active CDP session.
        room_id: Expected entered room ID.
        start_index: Raw-message index where matching should begin.

    Returns:
        True when a matching ``$room_id|...`` response has been captured.
    """
    expected_prefix = f"${room_id}|"
    payloads = get_captured_raw_messages(cdp)
    for payload in payloads[start_index:]:
        body = decode_captured_body(payload)
        if not body or body[0] != ord("$"):
            continue
        if body.decode("utf-8").startswith(expected_prefix):
            return True
    return False


def _wait_for_enter_response(
    page: RoomJoinPageProtocol,
    cdp: CDPSessionProtocol,
    room_id: str,
    *,
    start_index: int = 0,
) -> bool:
    """Wait for the room-enter response for the selected room.

    Args:
        page: Playwright page used for polling delays.
        cdp: Active CDP session.
        room_id: Selected room ID.
        start_index: Raw-message index where matching should begin.

    Returns:
        True when the matching enter response arrives, False on timeout.
    """
    waited_ms = 0
    while waited_ms < _ROOM_ENTER_TIMEOUT_MS:
        if _has_enter_response(cdp, room_id, start_index=start_index):
            return True
        page.wait_for_timeout(_JOIN_POLL_INTERVAL_MS)
        waited_ms += int(_JOIN_POLL_INTERVAL_MS)
    return False


def join_room(
    page: RoomJoinPageProtocol,
    cdp: CDPSessionProtocol,
    ws: WorldService,
) -> bool:
    """Join the configured room through the lobby websocket protocol.

    Args:
        page: Anything offering the two members the poll loop needs —
            the Playwright page in production, the simulator's link in
            a sim session.
        cdp: Active CDP session.

    Returns:
        True if the room was confirmed and the enter response arrived.
    """
    log.info("Joining game...")
    room_name = _test_hooks.get_env("TANKPIT_ROOM") or "Practice"
    room_entry = _wait_for_room_entry(page, cdp, ws, room_name)
    if room_entry is None:
        log.info("Room select failed: room list never exposed %s", room_name)
        return False
    room_id = room_entry["room_id"]
    join_confirm_start = len(get_captured_raw_messages(cdp))
    select_request: RoomSelectRequestDict = {"room_id": room_id}
    select_result = send_websocket_bytes(cdp, serialize_room_select_request(select_request))
    log.info("Room select: room=%s name=%s -> %s", room_id, room_name, select_result)
    if not select_result.startswith("SENT_"):
        return False

    if not _wait_for_join_confirm(page, cdp, room_id, start_index=join_confirm_start):
        log.info("Join confirm timeout: room=%s name=%s", room_id, room_name)
        return False
    ws.set_selected_room(room_id)
    magic = get_magic_key(cdp)
    if len(magic) == 0:
        log.info("Enter game failed: tankpit.magic was unavailable")
        return False
    tpclient_url = get_tpclient_url(cdp)
    if len(tpclient_url) == 0:
        log.info("Enter game failed: tpclient script URL was unavailable")
        return False
    static_key = load_tpclient_static_key(cdp, tpclient_url)
    metadata = build_room_enter_metadata(page.url, tpclient_url)
    codec = ProtocolCodec(static_key, magic)
    enter_request: RoomEnterRequestDict = {
        "room_id": room_id,
        "troop": room_entry["default_troop"],
        "preview_x": ROOM_ENTRY_DEFAULT_X,
        "preview_y": ROOM_ENTRY_DEFAULT_Y,
        "metadata": metadata,
    }
    enter_response_start = len(get_captured_raw_messages(cdp))
    enter_result = send_websocket_bytes(
        cdp,
        serialize_room_enter_request(enter_request, codec),
    )
    log.info(
        "Enter game: room=%s troop=%d -> %s",
        room_id,
        room_entry["default_troop"],
        enter_result,
    )
    if not enter_result.startswith("SENT_"):
        return False
    if not _wait_for_enter_response(page, cdp, room_id, start_index=enter_response_start):
        log.info("Enter response timeout: room=%s name=%s", room_id, room_name)
        return False
    return True


__all__ = [
    "join_room",
]
