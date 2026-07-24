"""Tests for the wire-timeline extraction.

Frames are built with the production XOR table and decoded by the
production protocol decoders — the assertions check the exact typed
values that come out the other side.
"""

from __future__ import annotations

import base64

import pytest

from tankpit_bot.protocol.commands import CMD_MAP_TELEPORT
from tankpit_bot.types import CapturedMessage
from tankpit_bot.validate.wire_timeline import extract_wire_timeline
from tests.validate.builders import (
    ENEMY_ID,
    SELF_ID,
    frame_message,
    fuel_gain_message,
    identity_message,
    inventory_message,
    make_session,
    move_message,
    pickup_message,
    sent_command_message,
    shot_message,
    sync_message,
    xor_encode_body,
)


def test_full_session_extraction() -> None:
    """Every tracked message family lands in its timeline list with exact values."""
    session = make_session(
        [
            identity_message(1000, SELF_ID),
            sync_message(2000, SELF_ID, 1, 1000),
            shot_message(3000, SELF_ID, 1),
            shot_message(3500, ENEMY_ID, 0),
            move_message(4000, SELF_ID, "ee"),
            move_message(4200, ENEMY_ID, "nn"),
            inventory_message(5000, [3, 12, 5, 8, 20]),
            sent_command_message(6000, CMD_MAP_TELEPORT, 30, 40),
            fuel_gain_message(7000, 950),
            pickup_message(7500),
            sync_message(8000, SELF_ID, 1, 900),
        ]
    )
    timeline = extract_wire_timeline(session)
    assert timeline["self_id"] == SELF_ID
    assert timeline["rank"] == 1
    assert timeline["fuel_readings"] == [
        {"timestamp_ms": 2000, "fuel": 1000, "from_event": False},
        {"timestamp_ms": 7000, "fuel": 950, "from_event": True},
        {"timestamp_ms": 8000, "fuel": 900, "from_event": False},
    ]
    assert timeline["own_shots"] == [{"timestamp_ms": 3000, "weapon": 1}]
    assert timeline["enemy_shots"] == [{"timestamp_ms": 3500, "weapon": 0}]
    assert timeline["self_moves"] == [{"timestamp_ms": 4000, "tiles": 2}]
    assert timeline["sent_actions"] == [
        {"timestamp_ms": 6000, "command": CMD_MAP_TELEPORT, "x": 30, "y": 40}
    ]
    assert timeline["pickup_timestamps"] == [7500]
    assert timeline["detonation_timestamps"] == []
    assert timeline["inventory_snapshots"] == [{"timestamp_ms": 5000, "counts": [3, 12, 5, 8, 20]}]


def test_session_without_magic_raises() -> None:
    """A magic-less session cannot be XOR-decoded and fails loudly."""
    session = make_session([identity_message(1000, SELF_ID)], magic=None)
    with pytest.raises(ValueError, match="without magic key"):
        extract_wire_timeline(session)


def test_untracked_and_malformed_frames_are_skipped() -> None:
    """Garbage payloads, text frames, and untracked types leave no trace."""
    text_body = b"=room text"
    zero_len_frame = bytes([0, 0, 5])
    overflow_frame = bytes([200, 0]) + b"xy"
    session = make_session(
        [
            CapturedMessage(
                timestamp_ms=100,
                direction="received",
                payload="%%%not-base64%%%",
                ws_url="wss://tankpit.com/ws",
            ),
            frame_message(200, text_body, "received"),
            CapturedMessage(
                timestamp_ms=300,
                direction="received",
                payload=base64.b64encode(zero_len_frame).decode("ascii"),
                ws_url="wss://tankpit.com/ws",
            ),
            CapturedMessage(
                timestamp_ms=400,
                direction="received",
                payload=base64.b64encode(overflow_frame).decode("ascii"),
                ws_url="wss://tankpit.com/ws",
            ),
            identity_message(500, SELF_ID),
        ]
    )
    timeline = extract_wire_timeline(session)
    assert timeline["self_id"] == SELF_ID
    assert timeline["fuel_readings"] == []
    assert timeline["own_shots"] == []
    assert timeline["sent_actions"] == []


def test_below_min_length_message_is_skipped() -> None:
    """A tracked type shorter than its wire minimum is dropped, not decoded."""
    tiny_shot = frame_message(100, xor_encode_body(0x53, bytes(4)), "received")
    session = make_session([identity_message(50, SELF_ID), tiny_shot])
    timeline = extract_wire_timeline(session)
    assert timeline["own_shots"] == []
    assert timeline["enemy_shots"] == []


def test_first_identity_wins() -> None:
    """The first 0x21 names the self tank; later identities are other tanks."""
    session = make_session([identity_message(100, SELF_ID), identity_message(200, ENEMY_ID)])
    assert extract_wire_timeline(session)["self_id"] == SELF_ID


def test_enemy_sync_and_enemy_moves_are_ignored() -> None:
    """Fuel readings and moves belong to the self tank only."""
    session = make_session(
        [
            identity_message(100, SELF_ID),
            sync_message(200, ENEMY_ID, 5, 1400),
            move_message(300, ENEMY_ID, "ss"),
        ]
    )
    timeline = extract_wire_timeline(session)
    assert timeline["fuel_readings"] == []
    assert timeline["rank"] is None
    assert timeline["self_moves"] == []


def test_stationary_self_move_echo_is_ignored() -> None:
    """A 0x47 with no waypoint path walked zero tiles and is not recorded."""
    session = make_session([identity_message(100, SELF_ID), move_message(200, SELF_ID, "")])
    assert extract_wire_timeline(session)["self_moves"] == []


def test_short_sync_updates_rank_without_a_fuel_reading() -> None:
    """The 9-byte sync form carries rank but no absolute fuel."""
    from tests.validate.builders import short_sync_message

    session = make_session([identity_message(100, SELF_ID), short_sync_message(200, SELF_ID, 3)])
    timeline = extract_wire_timeline(session)
    assert timeline["rank"] == 3
    assert timeline["fuel_readings"] == []


def test_tracked_tunnel_without_timeline_slot_is_ignored() -> None:
    """A tunneled TankRemove decodes but has no timeline destination."""
    from tests.validate.builders import tank_remove_message

    session = make_session([identity_message(100, SELF_ID), tank_remove_message(200, ENEMY_ID)])
    timeline = extract_wire_timeline(session)
    assert timeline["fuel_readings"] == []
    assert timeline["own_shots"] == []
    assert timeline["enemy_shots"] == []


def test_deposit_and_detonation_are_tracked() -> None:
    """A 0x64 deposit is an absolute fuel reading; a 0x45 detonation a hazard."""
    from tests.validate.builders import deposit_message, detonation_message

    session = make_session(
        [
            identity_message(100, SELF_ID),
            deposit_message(200, 400),
            detonation_message(300),
        ]
    )
    timeline = extract_wire_timeline(session)
    assert timeline["fuel_readings"] == [{"timestamp_ms": 200, "fuel": 400, "from_event": True}]
    assert timeline["detonation_timestamps"] == [300]


def test_sent_frames_without_command_prefix_are_ignored() -> None:
    """A sent frame whose first byte is not '!' is not a client command."""
    not_command = frame_message(100, xor_encode_body(0x2E, bytes(12)), "sent")
    session = make_session([not_command])
    assert extract_wire_timeline(session)["sent_actions"] == []


def test_short_sent_command_is_ignored() -> None:
    """A sent command too short to carry a command byte is dropped."""
    stub = frame_message(100, xor_encode_body(0x21, bytes([4])), "sent")
    session = make_session([stub])
    assert extract_wire_timeline(session)["sent_actions"] == []


def test_sent_command_without_coords_defaults_to_zero() -> None:
    """A two-byte command decodes with (0, 0) target coords."""
    stub = frame_message(100, xor_encode_body(0x21, bytes([4, CMD_MAP_TELEPORT])), "sent")
    session = make_session([stub])
    assert extract_wire_timeline(session)["sent_actions"] == [
        {"timestamp_ms": 100, "command": CMD_MAP_TELEPORT, "x": 0, "y": 0}
    ]
