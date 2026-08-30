"""Tests for lobby room discovery: what the ROOM_LIST frames decode to.

Split from ``test_join.py`` (2026-08-28, the room-dropdown lift) when
the combined module crossed the 600-line ceiling. This half covers
collecting and resolving room entries; the handshake that follows a
resolved room — select, confirm, enter — stays in ``test_join.py``.
"""

from __future__ import annotations

import base64

import pytest

from tankpit_bot.browser.cdp_helpers import decode_captured_body
from tankpit_bot.browser.room_join import (
    _collect_room_entries,
    _register_room_entries,
    _resolve_room_id,
    _wait_for_room_id,
)
from tankpit_bot.parser import RoomInfo
from tankpit_bot.protocol.framing import encode_frame
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.types.rooms import LOBBY_ROOMS
from tests.login.conftest import FakePageLogin, FakeRawMessageCDP, frame_payload


def test_decode_captured_body_rejects_trailing_bytes() -> None:
    """Captured body decode rejects trailing bytes after a framed payload."""
    payload = base64.b64encode(encode_frame(b"+1|Practice") + b"\x00").decode("utf-8")

    with pytest.raises(ValueError, match="unexpected trailing bytes"):
        decode_captured_body(payload)


def test_collect_room_entries_skips_non_room_messages_and_short_entries() -> None:
    """Room collection ignores non-ROOM_LIST payloads and malformed entries."""
    cdp = FakeRawMessageCDP(
        [
            frame_payload(b"=1|date|Artax|4"),
            frame_payload(b"+9"),
            frame_payload(b"+1|Practice|1"),
            frame_payload(b"+1|2|118|101|not-a-room"),
            frame_payload(b"+1|Practice|1|0,0,0,0,0,0,0|1|p|field01.gif|2026"),
        ]
    )

    entries = _collect_room_entries(cdp)

    assert len(entries) == 1
    assert entries[0]["room_id"] == "1"
    assert entries[0]["name"] == "Practice"
    assert entries[0]["image"] == "field01.gif"


def test_register_room_entries_skips_missing_images() -> None:
    """Room registration stores the field image from ROOM_LIST data."""

    ws = WorldService()
    _register_room_entries(
        ws,
        [
            {
                "room_id": "1",
                "name": "Practice",
                "field_id": 1,
                "game_modes": "0,0,0,0,0,0,0",
                "default_troop": 2,
                "mode_code": "p",
                "image": "field01.gif",
                "year": "2026",
            }
        ],
    )

    assert ws.room_images["1"] == "field01.gif"


def test_resolve_room_id_supports_prefix_match() -> None:
    """Room resolution accepts stable prefix matches for renamed world rooms."""
    room_id = _resolve_room_id(
        [
            {
                "room_id": "4",
                "name": "World (President Trump)",
                "field_id": 24,
                "game_modes": "5,1,0,0,0,0,0",
                "default_troop": 2,
                "mode_code": "n",
                "image": "field24.gif",
                "year": "2026",
            }
        ],
        "World",
    )

    assert room_id == "4"


def test_resolve_room_id_supports_exact_match() -> None:
    """Room resolution returns the exact room ID on exact name matches."""
    room_id = _resolve_room_id(
        [
            {
                "room_id": "1",
                "name": "Practice",
                "field_id": 1,
                "game_modes": "0,0,0,0,0,0,0",
                "default_troop": 2,
                "mode_code": "p",
                "image": "field01.gif",
                "year": "2026",
            }
        ],
        "Practice",
    )

    assert room_id == "1"


def test_every_offered_room_selector_resolves_against_a_live_lobby() -> None:
    """The fleet dropdown only offers selectors this resolver can match.

    Ground truth is the ROOM_LIST capture in
    ``runs/bot/arterial/bot-20260813-212329.log``: room 1 Practice and
    room 5 "World (Desert)". A selector that stopped resolving would
    strand the operator on a 10-second room-discovery timeout, so the
    offered vocabulary is pinned to the resolver here rather than
    trusted to stay in step.
    """
    lobby: list[RoomInfo] = [
        {
            "room_id": "1",
            "name": "Practice",
            "field_id": 1,
            "game_modes": "0,0,0,0,0,0,0",
            "default_troop": -1,
            "mode_code": "p",
            "image": "field01.gif",
            "year": "2026",
        },
        {
            "room_id": "5",
            "name": "World (Desert)",
            "field_id": 42,
            "game_modes": "5,1,0,0,0,0,0",
            "default_troop": 2,
            "mode_code": "n",
            "image": "field42.gif",
            "year": "2026",
        },
    ]

    resolved = {selector: _resolve_room_id(lobby, selector) for selector in LOBBY_ROOMS}

    assert resolved == {"World": "5", "Practice": "1"}


def test_resolve_room_id_returns_none_when_room_is_missing() -> None:
    """Room resolution returns None when no room matches the target name."""
    room_id = _resolve_room_id(
        [
            {
                "room_id": "4",
                "name": "World (President Trump)",
                "field_id": 24,
                "game_modes": "5,1,0,0,0,0,0",
                "default_troop": 2,
                "mode_code": "n",
                "image": "field24.gif",
                "year": "2026",
            }
        ],
        "Practice",
    )

    assert room_id is None


def test_room_discovery_timeout_diagnostic_tolerates_empty_frames() -> None:
    """The timeout dump skips undecodable frames and still reports.

    The 2026-08-13 arterial fresh-login lobby exposed one non-target
    room and the diagnostic dump is the discriminator between a parse
    rejection and a capture race — it must never crash on an empty
    captured frame while producing the report.
    """
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = FakeRawMessageCDP(
        [
            frame_payload(b""),
            frame_payload(b"+5|World (Desert)|2|d|2|n|desert.gif|2026"),
        ]
    )

    assert _wait_for_room_id(page, cdp, WorldService(), "Practice") is None
