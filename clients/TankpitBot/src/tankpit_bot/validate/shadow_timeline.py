"""Extract the shadow-law event timeline from one capture session.

The shadow comparator (``make shadow``) replays the sim's laws over
the real archive: each law imports its predictor from the sim source
and prices the archive's events against it. This module extracts the
event families those laws consume, using the exact production decode
recipe (same as ``wire_timeline``): split each WebSocket frame on the
2-byte LE length prefix, XOR-decode, and hand known types to
``protocol.decode_message`` — which unwraps 0x2E-tunneled bodies to
their inner message, so kills, gains, and removals surface whether
they rode the envelope or arrived top-level.

Self-identification follows the sniffer convention: the first 0x21
TankIdentity received in a session names the player's own tank.
"""

from __future__ import annotations

from typing import TypedDict

from tankpit_bot.capture.xor import decode_base64_safe
from tankpit_bot.protocol import decode_message
from tankpit_bot.protocol.types import BinaryMessage
from tankpit_bot.sniffer.constants import MSG_MIN_LENGTHS
from tankpit_bot.sniffer.xor import build_global_xor_table, reset_xor_state, xor_decode
from tankpit_bot.types import CapturedMessage, CaptureSession

_TRACKED_TYPES = frozenset({0x21, 0x28, 0x29, 0x2E, 0x3D, 0x41, 0x47, 0x49, 0x53, 0x58, 0x67})
"""Top-level frame types the extraction decodes. The 0x2E envelope
route unwraps tunneled kills/gains/removals/syncs/shots/moves; the
rest cover the same families arriving top-level."""


class TankSyncEventDict(TypedDict):
    """One per-tank 0x2E TankStatusSync (any tank, any form).

    ``fuel`` is None for the short form; long-form syncs carry the
    absolute fuel alongside the damage tier — the supervised pairs
    the damage-tier shadow law judges.
    """

    timestamp_ms: int
    tank_id: int
    damage_state: int
    rank: int
    fuel: int | None


class KillEventDict(TypedDict):
    """One 0x41 Deactivation event."""

    timestamp_ms: int
    victim_id: int
    killer_id: int
    is_mine_kill: bool


class EquipmentGainEventDict(TypedDict):
    """One 0x67 EquipmentGain (loud pickup or silent bundle)."""

    timestamp_ms: int
    show_message: bool
    gained: list[int]


class TankRemoveEventDict(TypedDict):
    """One 0x58 TankRemove (corpse removal or viewport exit)."""

    timestamp_ms: int
    tank_id: int


class TankExitEventDict(TypedDict):
    """One 0x29 TankExit announcement (quit / elimination)."""

    timestamp_ms: int
    tank_id: int


class InventoryEventDict(TypedDict):
    """One 0x49 own-inventory snapshot (five slot counts)."""

    timestamp_ms: int
    counts: list[int]


class ShotEventDict(TypedDict):
    """One 0x53 ShootEvent (any shooter)."""

    timestamp_ms: int
    shooter_id: int
    source_x: int
    source_y: int
    target_x: int
    target_y: int
    weapon: int


class PositionEventDict(TypedDict):
    """One position statement for a tank (0x3D / 0x28 / 0x47 end)."""

    timestamp_ms: int
    tank_id: int
    x: int
    y: int


class ShadowTimelineDict(TypedDict):
    """Everything the shadow-law validators need from one session."""

    session_id: str
    self_id: int | None
    names: dict[int, str]
    syncs: list[TankSyncEventDict]
    kills: list[KillEventDict]
    gains: list[EquipmentGainEventDict]
    removals: list[TankRemoveEventDict]
    exits: list[TankExitEventDict]
    inventories: list[InventoryEventDict]
    shots: list[ShotEventDict]
    positions: list[PositionEventDict]


def _message_timestamp(msg: CapturedMessage) -> int:
    """Return the timestamp of one captured message (sort key).

    Args:
        msg: Captured message.

    Returns:
        The message's ``timestamp_ms``.
    """
    return msg["timestamp_ms"]


def _split_frame_bodies(payload: str) -> list[bytes]:
    """Split one base64 WebSocket payload into logical message bodies.

    Args:
        payload: Base64-encoded frame payload.

    Returns:
        Message bodies (without length prefixes); empty when the
        payload is unparseable or too short.
    """
    data = decode_base64_safe(payload)
    if data is None or len(data) < 3:
        return []
    bodies: list[bytes] = []
    offset = 0
    while offset + 2 < len(data):
        msg_len = data[offset] | (data[offset + 1] << 8)
        offset += 2
        if msg_len == 0 or offset + msg_len > len(data):
            break
        bodies.append(data[offset : offset + msg_len])
        offset += msg_len
    return bodies


def _ingest_tank_events(
    timeline: ShadowTimelineDict, timestamp_ms: int, message: BinaryMessage
) -> bool:
    """Record identity, sync, removal, and exit events.

    Args:
        timeline: Timeline being built.
        timestamp_ms: Frame timestamp.
        message: Decoded wire message.

    Returns:
        True when the message was consumed by this family.
    """
    if message["msg_type"] == 0x21:
        if timeline["self_id"] is None:
            timeline["self_id"] = message["tank_id"]
        timeline["names"][message["tank_id"]] = message["name"]
        return True
    if message["msg_type"] == 0x3D:
        timeline["positions"].append(
            PositionEventDict(
                timestamp_ms=timestamp_ms,
                tank_id=message["tank_id"],
                x=message["x"],
                y=message["y"],
            )
        )
        return True
    if message["msg_type"] == 0x28:
        timeline["positions"].append(
            PositionEventDict(
                timestamp_ms=timestamp_ms,
                tank_id=message["tank_id"],
                x=message["x"],
                y=message["y"],
            )
        )
        return True
    if message["msg_type"] == 0x47:
        end = message["waypoints"][-1] if message["waypoints"] else None
        end_x = end[0] if end is not None else message["start_x"]
        end_y = end[1] if end is not None else message["start_y"]
        timeline["positions"].append(
            PositionEventDict(
                timestamp_ms=timestamp_ms,
                tank_id=message["tank_id"],
                x=end_x,
                y=end_y,
            )
        )
        return True
    if message["msg_type"] == 0x2E:
        timeline["syncs"].append(
            TankSyncEventDict(
                timestamp_ms=timestamp_ms,
                tank_id=message["tank_id"],
                damage_state=message["damage_state"],
                rank=message["rank"],
                fuel=message["fuel"],
            )
        )
        return True
    if message["msg_type"] == 0x58:
        timeline["removals"].append(
            TankRemoveEventDict(timestamp_ms=timestamp_ms, tank_id=message["tank_id"])
        )
        return True
    if message["msg_type"] == 0x29:
        timeline["exits"].append(
            TankExitEventDict(timestamp_ms=timestamp_ms, tank_id=message["tank_id"])
        )
        return True
    return False


def _ingest_combat_events(
    timeline: ShadowTimelineDict, timestamp_ms: int, message: BinaryMessage
) -> None:
    """Record kill, equipment-gain, and inventory events.

    Args:
        timeline: Timeline being built.
        timestamp_ms: Frame timestamp.
        message: Decoded wire message.
    """
    if message["msg_type"] == 0x41:
        timeline["kills"].append(
            KillEventDict(
                timestamp_ms=timestamp_ms,
                victim_id=message["victim_id"],
                killer_id=message["killer_id"],
                is_mine_kill=message["is_mine_kill"],
            )
        )
        return
    if message["msg_type"] == 0x67:
        timeline["gains"].append(
            EquipmentGainEventDict(
                timestamp_ms=timestamp_ms,
                show_message=message["show_message"],
                gained=list(message["gained"]),
            )
        )
        return
    if message["msg_type"] == 0x49:
        timeline["inventories"].append(
            InventoryEventDict(timestamp_ms=timestamp_ms, counts=list(message["counts"]))
        )
        return
    if message["msg_type"] == 0x53:
        timeline["shots"].append(
            ShotEventDict(
                timestamp_ms=timestamp_ms,
                shooter_id=message["shooter_id"],
                source_x=message["source_x"],
                source_y=message["source_y"],
                target_x=message["target_x"],
                target_y=message["target_y"],
                weapon=message["weapon"],
            )
        )


def _ingest_received(timeline: ShadowTimelineDict, timestamp_ms: int, body: bytes) -> None:
    """Record one received wire message into the timeline.

    Args:
        timeline: Timeline being built.
        timestamp_ms: Frame timestamp.
        body: Raw message body (msg_type byte + XOR-encoded rest).
    """
    msg_type = body[0]
    if msg_type not in _TRACKED_TYPES:
        return
    decoded_data = xor_decode(body)
    if len(decoded_data) < MSG_MIN_LENGTHS[msg_type]:
        return
    message = decode_message(msg_type, decoded_data)
    if _ingest_tank_events(timeline, timestamp_ms, message):
        return
    _ingest_combat_events(timeline, timestamp_ms, message)


def extract_shadow_timeline(session: CaptureSession) -> ShadowTimelineDict:
    """Extract the shadow-law event timeline from one session.

    Resets and rebuilds the global XOR state for the session's magic
    key (same discipline as ``wire_timeline``), then walks every
    received frame in timestamp order.

    Args:
        session: Loaded and validated capture session.

    Returns:
        The typed event timeline.

    Raises:
        ValueError: If the session has no magic key (cannot XOR-decode).
    """
    magic = session["magic"]
    if magic is None:
        raise ValueError("Cannot extract shadow timeline without magic key")
    reset_xor_state()
    build_global_xor_table(magic)
    timeline = ShadowTimelineDict(
        session_id=session["session_id"],
        self_id=None,
        names={},
        syncs=[],
        kills=[],
        gains=[],
        removals=[],
        exits=[],
        inventories=[],
        shots=[],
        positions=[],
    )
    ordered = sorted(session["messages"], key=_message_timestamp)
    for msg in ordered:
        if msg["direction"] != "received":
            continue
        for body in _split_frame_bodies(msg["payload"]):
            _ingest_received(timeline, msg["timestamp_ms"], body)
    return timeline


__all__ = [
    "EquipmentGainEventDict",
    "InventoryEventDict",
    "KillEventDict",
    "PositionEventDict",
    "ShadowTimelineDict",
    "ShotEventDict",
    "TankExitEventDict",
    "TankRemoveEventDict",
    "TankSyncEventDict",
    "extract_shadow_timeline",
]
