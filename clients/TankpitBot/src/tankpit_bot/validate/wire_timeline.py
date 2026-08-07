"""Extract a typed wire-event timeline from one capture session.

Uses the exact production decode recipe (``sniffer.decoders``): split
each WebSocket frame on the 2-byte LE length prefix, XOR-decode the
body, and hand known message types to ``protocol.decode_message``.
Only the message families the validators consume are kept; everything
else is skipped by type before decoding.

Events carry their frame timestamp. The isolation windows close
INCLUSIVELY on the ending reading's timestamp: the wire delivers a
debit's cause (shot echo) and its effect (fuel sync) in back-to-back
frames within the same millisecond, and the 2026-07-20 sweep (738/738
lone-hit windows) validated exactly these semantics.

Self-identification follows the sniffer convention: the first 0x21
TankIdentity received in a session names the player's own tank.
"""

from __future__ import annotations

from typing import TypedDict

from platform_core.logging import get_logger

from tankpit_bot.capture.frames import split_payload_frames
from tankpit_bot.capture.xor import build_session_xor_table, xor_decode_body
from tankpit_bot.protocol import decode_message
from tankpit_bot.protocol.commands import COMMAND_PREFIX
from tankpit_bot.protocol.framing import FramingError
from tankpit_bot.protocol.types import BinaryMessage
from tankpit_bot.sniffer.constants import MSG_MIN_LENGTHS
from tankpit_bot.types import CapturedMessage, CaptureSession

log = get_logger(__name__)

_TRACKED_TYPES = frozenset({0x21, 0x2E, 0x47, 0x49, 0x53, 0x64})
"""Top-level message types the extraction decodes. Mine detonations
(0x45) and container pickups (0x43) have no top-level route — they
arrive only 0x2E-tunneled and surface through the 0x2E branch."""


class FuelReadingDict(TypedDict):
    """One absolute fuel reading for the player's own tank.

    ``from_event`` marks readings carried by explicit fuel-change
    events (0x44/0x64) rather than the periodic 0x2E sync — the
    radar-cost validator treats those as contamination, exactly as
    the 2026-07-24 mining sweep did.
    """

    timestamp_ms: int
    fuel: int
    from_event: bool


class ShotEchoDict(TypedDict):
    """One 0x53 shoot echo (own or enemy, split by the caller)."""

    timestamp_ms: int
    weapon: int


class SentActionDict(TypedDict):
    """One sent client command (command byte + target coords)."""

    timestamp_ms: int
    command: int
    x: int
    y: int


class SelfMoveDict(TypedDict):
    """One own 0x47 movement echo with its true wire step count.

    The step count is the echo's FULL commanded path; when the bot
    re-commands mid-walk, only part of it is actually stepped — the
    walk-episode validator therefore prices single-echo episodes only.
    """

    timestamp_ms: int
    tiles: int


class InventorySnapshotDict(TypedDict):
    """One 0x49 equipment-count snapshot ([armor, dual, missile, homing, radar])."""

    timestamp_ms: int
    counts: list[int]


class WireTimelineDict(TypedDict):
    """Everything the archive validators need from one session."""

    session_id: str
    self_id: int | None
    rank: int | None
    fuel_readings: list[FuelReadingDict]
    own_shots: list[ShotEchoDict]
    enemy_shots: list[ShotEchoDict]
    sent_actions: list[SentActionDict]
    self_moves: list[SelfMoveDict]
    pickup_timestamps: list[int]
    detonation_timestamps: list[int]
    inventory_snapshots: list[InventorySnapshotDict]


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

    A validator reads the archive to judge it, so a payload it cannot
    parse is reported and skipped rather than silently truncated —
    the hand-rolled walk this replaces dropped a torn tail without a
    word ([[session-state-deglobalisation]]).

    Args:
        payload: Base64-encoded frame payload.

    Returns:
        Message bodies (without length prefixes); empty when the
        payload cannot be parsed.
    """
    try:
        return split_payload_frames(payload)
    except FramingError as error:
        log.warning("wire timeline: skipping unparseable payload: %s", error)
        return []


def _ingest_sent(
    timeline: WireTimelineDict, timestamp_ms: int, body: bytes, xor_table: bytes
) -> None:
    """Record one sent client command into the timeline.

    Args:
        timeline: Timeline being built.
        timestamp_ms: Frame timestamp.
        body: Raw message body (XOR-encoded after the prefix byte).
        xor_table: The owning session's XOR table.
    """
    if body[0] != COMMAND_PREFIX:
        return
    decoded = xor_decode_body(body, xor_table, offset=1)
    if len(decoded) < 2:
        return
    x = decoded[2] if len(decoded) > 2 else 0
    y = decoded[3] if len(decoded) > 3 else 0
    timeline["sent_actions"].append(
        SentActionDict(timestamp_ms=timestamp_ms, command=decoded[1], x=x, y=y)
    )


def _ingest_fuel_and_hazards(
    timeline: WireTimelineDict, timestamp_ms: int, message: BinaryMessage
) -> bool:
    """Record fuel readings, pickups, and detonations from one message.

    Args:
        timeline: Timeline being built.
        timestamp_ms: Frame timestamp.
        message: Decoded wire message.

    Returns:
        True when the message was consumed by this family.
    """
    if message["msg_type"] == 0x2E:
        if message["tank_id"] == timeline["self_id"]:
            timeline["rank"] = message["rank"]
            fuel = message["fuel"]
            if fuel is not None:
                timeline["fuel_readings"].append(
                    FuelReadingDict(timestamp_ms=timestamp_ms, fuel=fuel, from_event=False)
                )
        return True
    if message["msg_type"] == 0x44:
        timeline["fuel_readings"].append(
            FuelReadingDict(timestamp_ms=timestamp_ms, fuel=message["fuel_total"], from_event=True)
        )
        return True
    if message["msg_type"] == 0x64:
        timeline["fuel_readings"].append(
            FuelReadingDict(timestamp_ms=timestamp_ms, fuel=message["fuel_total"], from_event=True)
        )
        return True
    if message["msg_type"] == "container_pickup":
        timeline["pickup_timestamps"].append(timestamp_ms)
        return True
    if message["msg_type"] == 0x45:
        timeline["detonation_timestamps"].append(timestamp_ms)
        return True
    return False


def _ingest_combat_and_identity(
    timeline: WireTimelineDict, timestamp_ms: int, message: BinaryMessage
) -> None:
    """Record identity, shot, movement, and inventory messages.

    Args:
        timeline: Timeline being built.
        timestamp_ms: Frame timestamp.
        message: Decoded wire message.
    """
    if message["msg_type"] == 0x21:
        if timeline["self_id"] is None:
            timeline["self_id"] = message["tank_id"]
        return
    if message["msg_type"] == 0x53:
        echo = ShotEchoDict(timestamp_ms=timestamp_ms, weapon=message["weapon"])
        if message["shooter_id"] == timeline["self_id"]:
            timeline["own_shots"].append(echo)
        else:
            timeline["enemy_shots"].append(echo)
        return
    if message["msg_type"] == 0x47:
        if message["tank_id"] == timeline["self_id"] and message["path_tiles"]:
            timeline["self_moves"].append(
                SelfMoveDict(timestamp_ms=timestamp_ms, tiles=message["path_tiles"])
            )
        return
    if message["msg_type"] == 0x49:
        timeline["inventory_snapshots"].append(
            InventorySnapshotDict(timestamp_ms=timestamp_ms, counts=list(message["counts"]))
        )


def _ingest_received(
    timeline: WireTimelineDict, timestamp_ms: int, body: bytes, xor_table: bytes
) -> None:
    """Record one received wire message into the timeline.

    Args:
        timeline: Timeline being built.
        timestamp_ms: Frame timestamp.
        body: Raw message body (msg_type byte + XOR-encoded rest).
        xor_table: The owning session's XOR table.
    """
    msg_type = body[0]
    if msg_type not in _TRACKED_TYPES:
        return
    decoded_data = xor_decode_body(body, xor_table, offset=1)
    if len(decoded_data) < MSG_MIN_LENGTHS[msg_type]:
        return
    message = decode_message(msg_type, decoded_data)
    if _ingest_fuel_and_hazards(timeline, timestamp_ms, message):
        return
    _ingest_combat_and_identity(timeline, timestamp_ms, message)


def extract_wire_timeline(session: CaptureSession) -> WireTimelineDict:
    """Extract the validator-relevant wire timeline from one session.

    Builds the XOR table from this session's own magic and holds it as
    a LOCAL, so two sessions can be extracted concurrently. It was a
    module global until 2026-08-06, which forced every archive walk to
    be sequential ([[session-state-deglobalisation]] step 1).

    Args:
        session: Loaded and validated capture session.

    Returns:
        The typed wire timeline.

    Raises:
        ValueError: If the session has no magic key (cannot XOR-decode).
        XorStaticKeyUnavailableError: If the static key cannot be read.
    """
    magic = session["magic"]
    if magic is None:
        raise ValueError("Cannot extract timeline without magic key")
    xor_table = build_session_xor_table(magic)
    timeline = WireTimelineDict(
        session_id=session["session_id"],
        self_id=None,
        rank=None,
        fuel_readings=[],
        own_shots=[],
        enemy_shots=[],
        sent_actions=[],
        self_moves=[],
        pickup_timestamps=[],
        detonation_timestamps=[],
        inventory_snapshots=[],
    )
    ordered = sorted(session["messages"], key=_message_timestamp)
    for msg in ordered:
        for body in _split_frame_bodies(msg["payload"]):
            if msg["direction"] == "sent":
                _ingest_sent(timeline, msg["timestamp_ms"], body, xor_table)
            else:
                _ingest_received(timeline, msg["timestamp_ms"], body, xor_table)
    return timeline


__all__ = [
    "FuelReadingDict",
    "InventorySnapshotDict",
    "SelfMoveDict",
    "SentActionDict",
    "ShotEchoDict",
    "WireTimelineDict",
    "extract_wire_timeline",
]
