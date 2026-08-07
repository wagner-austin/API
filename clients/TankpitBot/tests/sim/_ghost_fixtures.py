"""Shared capture fixtures and helpers for the ghost tests."""

from __future__ import annotations

from platform_core.json_utils import (
    JSONValue,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    narrow_json_to_list,
)

from tankpit_bot.capture.xor import (
    build_session_xor_table,
    xor_decode_body,
)
from tankpit_bot.protocol.types import (
    BinaryMessage,
    ChatMessageDict,
    FuelGainDict,
    InventoryDict,
    MapDataDict,
    ShootEventDict,
)
from tankpit_bot.sim.transport import encode_tick_payload
from tankpit_bot.sim.wire_statements import (
    identity_statement,
    position_statement,
)
from tankpit_bot.sim.world import (
    SimWorldDict,
    make_sim_tank,
    make_sim_world,
)

_MAGIC = "ghosttestmagic"


_T0 = 1_785_000_000_000


def _world_for_statements(ghost_name: str = "Yuppler") -> SimWorldDict:
    """A statement-builder world: recorded self 77 and ghost 500."""
    world = make_sim_world("field01_r.gif")
    world["tanks"][77] = make_sim_tank(77, 2, 3, 100, 100, 900)
    world["tanks"][500] = make_sim_tank(500, 1, 2, 108, 100, 600, name=ghost_name)
    return world


def _capture(timeline: list[tuple[int, list[BinaryMessage]]]) -> str:
    """Encode decoded message batches into a capture session text.

    Args:
        timeline: ``(timestamp_ms, batch)`` pairs, received direction.

    Returns:
        The ``capture_session.json`` text the compiler consumes.
    """
    table = build_session_xor_table(_MAGIC)
    messages = [
        {
            "timestamp_ms": t,
            "direction": "received",
            "payload": encode_tick_payload(batch, table),
        }
        for t, batch in timeline
    ]
    return dump_json_str(
        {
            "session_id": "ghost-test",
            "start_timestamp_ms": _T0,
            "end_timestamp_ms": _T0 + 60_000,
            "base_url": "wss://test/",
            "magic": _MAGIC,
            "messages": messages,
            "game_log": [],
            "tank_names": {},
        }
    )


def _fight_capture(ghost_name: str = "Yuppler") -> str:
    """A tiny recorded fight: join, ghost sighted, moves, shoots, chats."""
    world = _world_for_statements(ghost_name)
    join: list[BinaryMessage] = [
        identity_statement(world, 77),
        position_statement(world, 77),
        FuelGainDict(msg_type=0x44, fuel_total=900, is_free=False, flag=1),
        InventoryDict(
            msg_type=0x49,
            show=True,
            alternate=False,
            counts=[25, 20, 25, 19, 21],
            enabled=[True] * 5,
        ),
        MapDataDict(msg_type=0x4C, fuel_dots=[(140, 90), (150, 95)], tanks=[]),
        identity_statement(world, 500),
        position_statement(world, 500),
    ]
    world["tanks"][500]["x"] = 106
    ghost_moves: list[BinaryMessage] = [position_statement(world, 500)]
    ghost_shoots: list[BinaryMessage] = [
        ShootEventDict(
            msg_type=0x53,
            team=1,
            shooter_id=500,
            source_x=106,
            source_y=100,
            target_x=100,
            target_y=100,
            aim_x=100,
            aim_y=100,
            weapon=1,
        ),
        ChatMessageDict(msg_type=0x4D, sender_id=500, message_type=41, x=106, y=100),
    ]
    world["tanks"][77]["x"] = 102
    self_moves: list[BinaryMessage] = [position_statement(world, 77)]
    return _capture(
        [
            (_T0, join),
            (_T0 + 2000, ghost_moves),
            (_T0 + 4000, ghost_shoots),
            (_T0 + 6000, self_moves),
        ]
    )


def _rich_capture() -> str:
    """The fight capture plus every world-read and stray-frame flavor."""
    import base64

    from tankpit_bot.protocol.types import (
        MovementDict,
        RadarContainerDict,
        RadarScanResultDict,
        ViewportEntityDict,
        ViewportUpdateDict,
    )

    world = _world_for_statements()
    world["tanks"][501] = make_sim_tank(501, 3, 1, 50, 50, 500, name="stranger")
    join: list[BinaryMessage] = [
        identity_statement(world, 77),
        position_statement(world, 77),
        FuelGainDict(msg_type=0x44, fuel_total=900, is_free=False, flag=1),
        InventoryDict(
            msg_type=0x49,
            show=True,
            alternate=False,
            counts=[25, 20, 25, 19, 21],
            enabled=[True] * 5,
        ),
        MapDataDict(msg_type=0x4C, fuel_dots=[(140, 90)], tanks=[]),
        identity_statement(world, 500),
        position_statement(world, 500),
        identity_statement(world, 501),  # never sighted -> unplaced
    ]
    reads: list[BinaryMessage] = [
        RadarScanResultDict(
            msg_type=0x4F,
            containers=[
                RadarContainerDict(x=140, y=90, volume=520),
                RadarContainerDict(x=141, y=90, volume=-1),
                RadarContainerDict(x=142, y=90, volume=0),
            ],
            mines=[],
            mine_clears=[],
        ),
        ViewportUpdateDict(
            msg_type=0x5A,
            viewport_left=92,
            viewport_top=92,
            entities=[
                ViewportEntityDict(col=9, row=9, cache_value=77, overlay_value=8, terrain_type=0)
            ],
        ),
        MovementDict(
            msg_type=0x47,
            tank_id=500,
            start_x=108,
            start_y=100,
            direction=0,
            damage_state=0,
            lb_score=0,
            rank=2,
            flag=0,
            is_carrying=False,
            waypoints=[(106, 100)],
            path_tiles=2,
            path="ww",
        ),
        ShootEventDict(
            msg_type=0x53,
            team=2,
            shooter_id=77,  # SELF shot: never a ghost event
            source_x=100,
            source_y=100,
            target_x=106,
            target_y=100,
            aim_x=106,
            aim_y=100,
            weapon=1,
        ),
        ShootEventDict(
            msg_type=0x53,
            team=3,
            shooter_id=502,  # unknown shooter: filtered at assembly
            source_x=50,
            source_y=50,
            target_x=100,
            target_y=100,
            aim_x=100,
            aim_y=100,
            weapon=0,
        ),
        ChatMessageDict(msg_type=0x4D, sender_id=502, message_type=3, x=50, y=50),
        ChatMessageDict(msg_type=0x4D, sender_id=77, message_type=2, x=100, y=100),  # self chat
        identity_statement(world, 77),  # duplicate SELF identity: ignored
        MapDataDict(msg_type=0x4C, fuel_dots=[(1, 1)], tanks=[]),  # later atlas: first wins
        FuelGainDict(msg_type=0x44, fuel_total=950, is_free=False, flag=1),  # later fuel read
    ]
    text = _capture([(_T0, join), (_T0 + 2000, reads)])
    session = narrow_json_load(text)
    stray_messages = list(narrow_json_to_list(session["messages"]))
    # 0x43 CacheUpdate rides the wire as a TOP-LEVEL frame — inside a
    # 0x2E envelope the 0x43 subtype byte means container_pickup, so
    # the sim's envelope encoder cannot carry it. Craft the raw frame:
    # the type byte travels in the clear; ``xor_decode_body`` at
    # ``offset=1`` strips it and XORs the remainder, so encoding the
    # payload = decoding a body with any placeholder type byte in front.
    table = build_session_xor_table(_MAGIC)
    cache_payload = bytes([143, 90, 260 & 0xFF, 260 >> 8])
    cache_body = bytes([0x43]) + xor_decode_body(bytes(1) + cache_payload, table, offset=1)
    cache_frame = bytes([len(cache_body), 0]) + cache_body
    stray_messages.append(
        {
            "timestamp_ms": _T0 + 2100,
            "direction": "received",
            "payload": base64.b64encode(cache_frame).decode(),
        }
    )
    # stray-frame flavors the compiler must skip: a sent command, an
    # empty payload, a plaintext ack, a text row, garbage bytes, and
    # a torn frame.
    stray_messages.extend(
        [
            {"timestamp_ms": _T0 + 100, "direction": "sent", "payload": "AAA="},
            {"timestamp_ms": _T0 + 200, "direction": "received", "payload": ""},
            {
                "timestamp_ms": _T0 + 300,
                "direction": "received",
                "payload": base64.b64encode(bytes([2, 0]) + b"A1").decode(),
            },
            {
                "timestamp_ms": _T0 + 400,
                "direction": "received",
                "payload": base64.b64encode(bytes([3, 0]) + b"+r|").decode(),
            },
            {
                "timestamp_ms": _T0 + 500,
                "direction": "received",
                "payload": base64.b64encode(
                    bytes([2, 0, 0x99, 0x01]) + bytes([9, 0]) + b"xx"
                ).decode(),
            },
            {
                "timestamp_ms": _T0 + 600,
                "direction": "received",
                "payload": base64.b64encode(bytes([3, 0]) + b"$ok").decode(),
            },
        ]
    )
    session["messages"] = stray_messages
    return dump_json_str(session)


def narrow_json_load(text: str) -> dict[str, JSONValue]:
    """Load capture JSON back to a mutable dict for fixture edits."""
    return dict(narrow_json_to_dict(load_json_str(text)))
