"""Shadow-timeline extraction over production-encoded frames."""

from __future__ import annotations

import base64

import pytest

from tankpit_bot.types import CapturedMessage
from tankpit_bot.validate.shadow_timeline import extract_shadow_timeline
from tests.validate.builders import (
    ENEMY_ID,
    SELF_ID,
    aimed_shot_message,
    deactivation_message,
    deposit_message,
    equipment_gain_message,
    frame_message,
    identity_message,
    inventory_message,
    make_session,
    move_message,
    movement_response_message,
    named_identity_message,
    sent_command_message,
    sync_message,
    tank_entry_message,
    tank_exit_message,
    tank_remove_message,
    xor_encode_body,
)


def test_extracts_every_event_family() -> None:
    session = make_session(
        [
            identity_message(0, SELF_ID),
            sync_message(1000, ENEMY_ID, 3, 400),
            deactivation_message(2000, ENEMY_ID, SELF_ID),
            equipment_gain_message(3000, [0, 7, 0, 0, 0], True),
            equipment_gain_message(3500, [0, 2, 0, 1, 1], False),
            tank_remove_message(4000, ENEMY_ID),
            tank_exit_message(5000, ENEMY_ID),
            inventory_message(6000, [1, 2, 3, 4, 5]),
        ]
    )
    timeline = extract_shadow_timeline(session)
    assert timeline["self_id"] == SELF_ID
    assert timeline["syncs"] == [
        {
            "timestamp_ms": 1000,
            "tank_id": ENEMY_ID,
            "damage_state": 0,
            "rank": 3,
            "fuel": 400,
        }
    ]
    assert timeline["kills"] == [
        {
            "timestamp_ms": 2000,
            "victim_id": ENEMY_ID,
            "killer_id": SELF_ID,
            "is_mine_kill": False,
        }
    ]
    assert timeline["gains"] == [
        {"timestamp_ms": 3000, "show_message": True, "gained": [0, 7, 0, 0, 0]},
        {"timestamp_ms": 3500, "show_message": False, "gained": [0, 2, 0, 1, 1]},
    ]
    assert timeline["removals"] == [{"timestamp_ms": 4000, "tank_id": ENEMY_ID}]
    assert timeline["exits"] == [{"timestamp_ms": 5000, "tank_id": ENEMY_ID}]
    assert timeline["inventories"] == [{"timestamp_ms": 6000, "counts": [1, 2, 3, 4, 5]}]


def test_first_identity_wins() -> None:
    session = make_session([identity_message(0, SELF_ID), identity_message(100, ENEMY_ID)])
    timeline = extract_shadow_timeline(session)
    assert timeline["self_id"] == SELF_ID


def test_requires_magic() -> None:
    session = make_session([identity_message(0, SELF_ID)], magic=None)
    with pytest.raises(ValueError, match="magic"):
        extract_shadow_timeline(session)


def test_ignores_sent_frames_and_untracked_types() -> None:
    untracked_radar_ack = frame_message(1000, xor_encode_body(0x46, bytes([1, 1])), "received")
    session = make_session(
        [
            sent_command_message(0, 0x74, 5, 5),
            untracked_radar_ack,
        ]
    )
    timeline = extract_shadow_timeline(session)
    assert timeline["self_id"] is None
    assert timeline["syncs"] == []
    assert timeline["kills"] == []
    assert timeline["shots"] == []


def test_extracts_names_shots_and_positions() -> None:
    session = make_session(
        [
            named_identity_message(0, SELF_ID, "Artax"),
            named_identity_message(100, ENEMY_ID, "orange-3"),
            movement_response_message(200, ENEMY_ID, 30, 31),
            tank_entry_message(300, SELF_ID, 10, 11),
            move_message(400, SELF_ID, "ee"),
            move_message(500, ENEMY_ID, ""),
            aimed_shot_message(600, ENEMY_ID, (30, 31), (10, 11), 0),
        ]
    )
    timeline = extract_shadow_timeline(session)
    assert timeline["names"] == {SELF_ID: "Artax", ENEMY_ID: "orange-3"}
    assert timeline["positions"] == [
        {"timestamp_ms": 200, "tank_id": ENEMY_ID, "x": 30, "y": 31},
        {"timestamp_ms": 300, "tank_id": SELF_ID, "x": 10, "y": 11},
        {"timestamp_ms": 400, "tank_id": SELF_ID, "x": 7, "y": 5},
        {"timestamp_ms": 500, "tank_id": ENEMY_ID, "x": 5, "y": 5},
    ]
    assert timeline["shots"] == [
        {
            "timestamp_ms": 600,
            "shooter_id": ENEMY_ID,
            "source_x": 30,
            "source_y": 31,
            "target_x": 10,
            "target_y": 11,
            "weapon": 0,
        }
    ]


def test_tunneled_inner_type_outside_families_is_skipped() -> None:
    session = make_session([deposit_message(0, 900)])
    timeline = extract_shadow_timeline(session)
    assert timeline["syncs"] == []
    assert timeline["inventories"] == []


def test_short_body_below_min_length_is_skipped() -> None:
    body = xor_encode_body(0x41, bytes([0, 9]))
    session = make_session([frame_message(0, body, "received")])
    timeline = extract_shadow_timeline(session)
    assert timeline["kills"] == []


def test_unparseable_payloads_yield_no_events() -> None:
    not_base64 = CapturedMessage(
        timestamp_ms=0,
        direction="received",
        payload="!!!not-base64!!!",
        ws_url="wss://tankpit.com/ws",
    )
    too_short = CapturedMessage(
        timestamp_ms=0,
        direction="received",
        payload=base64.b64encode(b"\x01\x00").decode("ascii"),
        ws_url="wss://tankpit.com/ws",
    )
    zero_length = CapturedMessage(
        timestamp_ms=0,
        direction="received",
        payload=base64.b64encode(b"\x00\x00\x21\x00").decode("ascii"),
        ws_url="wss://tankpit.com/ws",
    )
    torn_frame = CapturedMessage(
        timestamp_ms=0,
        direction="received",
        payload=base64.b64encode(b"\xff\x00\x21\x00").decode("ascii"),
        ws_url="wss://tankpit.com/ws",
    )
    session = make_session([not_base64, too_short, zero_length, torn_frame])
    timeline = extract_shadow_timeline(session)
    assert timeline["self_id"] is None
    assert timeline["syncs"] == []


def test_mine_kill_flag_survives_extraction() -> None:
    mine_killer = 65532
    session = make_session(
        [
            identity_message(0, SELF_ID),
            deactivation_message(1000, ENEMY_ID, mine_killer),
        ]
    )
    timeline = extract_shadow_timeline(session)
    assert timeline["kills"][0]["is_mine_kill"] is True
