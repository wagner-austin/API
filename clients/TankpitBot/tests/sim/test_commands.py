"""Typed client-command decoding for the sim server."""

from __future__ import annotations

import pytest

from tankpit_bot.protocol.commands import (
    CMD_KEEPALIVE,
    CMD_MAP_OPEN,
    CMD_MAP_TELEPORT,
    CMD_MINE,
    CMD_MOVE,
    CMD_PICKUP_EQUIPMENT,
    CMD_PICKUP_FUEL,
    CMD_RADAR,
    CMD_SHOOT,
    CMD_STATISTICS,
    CMD_TOGGLE_EQUIPMENT,
)
from tankpit_bot.sim.commands import decode_client_command
from tankpit_bot.wire.helpers import DecodeError


def test_coordinate_commands_decode_kind_and_coords() -> None:
    """Move / teleport / pickup commands carry their click tile."""
    expected = {
        CMD_MOVE: "move",
        CMD_MAP_TELEPORT: "teleport",
        CMD_PICKUP_FUEL: "pickup_fuel",
        CMD_PICKUP_EQUIPMENT: "pickup_equipment",
    }
    for command, kind in expected.items():
        decoded = decode_client_command(bytes([4, command, 42, 161]))
        assert decoded["kind"] == kind
        assert decoded["x"] == 42
        assert decoded["y"] == 161
        assert decoded["target_id"] == 0


def test_shoot_decodes_with_and_without_target_id() -> None:
    """The shoot command's trailing entity id is optional."""
    bare = decode_client_command(bytes([6, CMD_SHOOT, 55, 167]))
    assert bare["kind"] == "shoot"
    assert bare["target_id"] == 0
    targeted = decode_client_command(bytes([6, CMD_SHOOT, 55, 167, 0x10, 0x02]))
    assert targeted["target_id"] == 0x0210


def test_bare_commands_decode_without_coords() -> None:
    """Radar / mine / map-open / statistics carry no coordinates.

    The statistics key used to decode as ``other`` and fall through to
    the map-data arm, so the sim answered a stats request with a map
    ([[session-state-deglobalisation]]).
    """
    for command, kind in (
        (CMD_RADAR, "radar"),
        (CMD_MINE, "mine"),
        (CMD_MAP_OPEN, "map_open"),
        (CMD_STATISTICS, "statistics"),
    ):
        decoded = decode_client_command(bytes([2, command]))
        assert decoded["kind"] == kind
        assert decoded["x"] == 0


def test_toggle_equipment_decodes_the_slot_digit() -> None:
    """The 1-5 slot key char decodes to the numeric slot."""
    decoded = decode_client_command(bytes([0x23, CMD_TOGGLE_EQUIPMENT, ord("2")]))
    assert decoded["kind"] == "toggle_equipment"
    assert decoded["slot"] == 2
    with pytest.raises(DecodeError):
        decode_client_command(bytes([0x23, CMD_TOGGLE_EQUIPMENT]))


def test_chat_decodes_message_id_and_sender_tile() -> None:
    """The 6-byte Hb chat frame decodes to kind=chat with its preset id.

    Fixture mirrors the live HELLO send of sniff-20260729-214411:
    ``[6, 'm', 41, 141, 236, 0]``.
    """
    from tankpit_bot.protocol.chat import CMD_CHAT

    decoded = decode_client_command(bytes([6, CMD_CHAT, 41, 141, 236, 0]))
    assert decoded["kind"] == "chat"
    assert decoded["message_id"] == 41
    assert decoded["x"] == 141
    assert decoded["y"] == 236
    with pytest.raises(DecodeError):
        decode_client_command(bytes([6, CMD_CHAT, 41, 141]))


def test_unknown_command_preserves_raw_byte() -> None:
    """Unmapped commands decode as ``other`` with the byte kept.

    91 (``[``) is outside every mapped command byte — 90 (``Z``) was
    the old example until the scope-extend decode claimed it
    (2026-08-01, [[viewport-shift-protocol]]).
    """
    decoded = decode_client_command(bytes([2, 91]))
    assert decoded["kind"] == "other"
    assert decoded["command"] == 91


def test_short_payloads_raise() -> None:
    """Truncated payloads are a decode failure, never a guess."""
    with pytest.raises(DecodeError):
        decode_client_command(bytes([4]))
    with pytest.raises(DecodeError):
        decode_client_command(bytes([4, CMD_MOVE, 42]))
    with pytest.raises(DecodeError):
        decode_client_command(bytes([6, CMD_SHOOT, 55]))


def test_the_client_heartbeat_decodes_to_its_own_kind() -> None:
    """THE FRAME A REAL CLIENT SENDS EVERY TICK, AND OUR BOT NEVER DOES.

    Payload ``02 21`` — invariant across all 11,871 archived sends,
    two bytes, no arguments. [[client-commands]] has carried the row
    (JS class ``dc``) all along; the DECODER did not, so the frame
    resolved to ``other``, and ``other`` is the one kind the sim
    server refuses.
    """
    command = decode_client_command(bytes([0x02, CMD_KEEPALIVE]))

    assert command["kind"] == "keepalive"
    assert command["command"] == CMD_KEEPALIVE
    assert (command["x"], command["y"]) == (0, 0)


def test_an_unknown_command_byte_is_still_other() -> None:
    """Naming the heartbeat must not turn the decoder permissive.

    ``other`` still exists and still means "we have no law for this".
    A decoder that answered every byte with a plausible kind would
    hide the next unmodelled command exactly as this one was hidden.
    """
    assert decode_client_command(bytes([0x02, 0xFE]))["kind"] == "other"
