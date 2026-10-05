"""Capture builders for the container-census tests.

Every binary message rides its 0x2E envelope exactly as the real server
sends it, built through the production encoder and the production cipher
(:mod:`tests.analysis._capture_fixtures`), so a change to either breaks
these captures rather than leaving them agreeing with nothing.
"""

from __future__ import annotations

from platform_core.json_utils import JSONObject, load_json_str, narrow_json_to_dict

from tankpit_bot.analysis.scan import decode_session_frames
from tankpit_bot.protocol.command_builders import build_query_command
from tankpit_bot.protocol.commands import CMD_MAP_OPEN, CMD_RADAR
from tankpit_bot.protocol.encoders import encode_envelope_body
from tankpit_bot.protocol.types import (
    BinaryMessage,
    InventoryDict,
    MovementResponseDict,
    RadarContainerDict,
    RadarScanResultDict,
    TankStatusSyncDict,
    ViewportUpdateDict,
)
from tankpit_bot.types import decode_capture_session
from tankpit_bot.validate.conformance_wire import ReplayCapture, read_replay
from tests.analysis._capture_fixtures import (
    OWN_TANK,
    _ciphered,
    _command,
    _payload,
    _received,
    _sent,
    _session_json,
    _tank_info,
)


def tunneled(message: BinaryMessage) -> bytes:
    """One binary message as a ciphered 0x2E frame body."""
    return _ciphered(bytes([0x2E]) + encode_envelope_body(message))


def placed(x: int, y: int) -> bytes:
    """The client placed at a tile."""
    return tunneled(
        MovementResponseDict(
            msg_type=0x3D,
            team=1,
            tank_id=OWN_TANK,
            x=x,
            y=y,
            direction=8,
            damage_state=3,
            rank=0,
            lb_score=0,
            carrying=0,
        )
    )


def ranked(rank: int) -> bytes:
    """The client's own status sync, stating its rank."""
    return tunneled(
        TankStatusSyncDict(
            msg_type=0x2E,
            subtype=1,
            tank_id=OWN_TANK,
            damage_state=3,
            rank=rank,
            lb_score=0,
            promo_state=0,
            promo_bar_lit=True,
            fuel=900,
        )
    )


def windowed(left: int, top: int) -> bytes:
    """The client's stored window."""
    return tunneled(
        ViewportUpdateDict(msg_type=0x5A, viewport_left=left, viewport_top=top, entities=[])
    )


def spent_radar() -> bytes:
    """The inventory snapshot an extra radar's consumption draws."""
    return tunneled(
        InventoryDict(
            msg_type=0x49,
            show=False,
            alternate=False,
            counts=[25, 25, 25, 25, 24],
            enabled=[False, True, False, True, True],
        )
    )


def scanned(*containers: tuple[int, int, int]) -> bytes:
    """A 0x4F listing ``(x, y, volume)`` containers (volume -1 is equipment)."""
    return tunneled(
        RadarScanResultDict(
            msg_type=0x4F,
            containers=[RadarContainerDict(x=x, y=y, volume=v) for x, y, v in containers],
            mines=[],
            mine_clears=[],
        )
    )


def radar(at: int) -> JSONObject:
    """The client sending a radar at ``at`` ms."""
    return _sent(_payload(_command(build_query_command(CMD_RADAR))), at)


def map_open(at: int) -> JSONObject:
    """The client opening the map at ``at`` ms."""
    return _sent(_payload(_command(build_query_command(CMD_MAP_OPEN))), at)


def received(at: int, *bodies: bytes) -> JSONObject:
    """One received message holding these frames."""
    return _received(_payload(*bodies), at)


def joined(room_image: str, at: int = 100) -> list[JSONObject]:
    """The lobby listing room 1 on a field, then the join confirming it."""
    listing = f"+1|Practice|1|0,0,0,0,0,0,0|-1|p|{room_image}|2026".encode()
    return [received(at, listing), received(at + 1, b"=1|2026|red-9|1|0|0|0|0")]


def introduced(at: int, x: int, y: int, rank: int, left: int, top: int) -> JSONObject:
    """The client introduced, placed, ranked and windowed in one batch."""
    return received(at, _tank_info(OWN_TANK), placed(x, y), ranked(rank), windowed(left, top))


def capture_text(messages: list[JSONObject]) -> str:
    """A capture file's text."""
    return _session_json(messages=messages)


def as_replay(messages: list[JSONObject]) -> ReplayCapture:
    """A capture read as ticks."""
    session = decode_capture_session(narrow_json_to_dict(load_json_str(capture_text(messages))))
    return read_replay(decode_session_frames(session))


__all__ = [
    "as_replay",
    "capture_text",
    "introduced",
    "joined",
    "map_open",
    "placed",
    "radar",
    "ranked",
    "received",
    "scanned",
    "spent_radar",
    "tunneled",
    "windowed",
]
