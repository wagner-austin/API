"""Synthetic capture builders for the validator tests.

Frames are encoded with the REAL XOR table for the test magic (the
same table ``extract_wire_timeline`` rebuilds from the session's
magic), so the tests exercise the production decode path end to end —
no fakes in the decode chain.
"""

from __future__ import annotations

import base64

from tankpit_bot.capture.xor import build_session_xor_table
from tankpit_bot.types import CapturedMessage, CaptureSession

MAGIC = "auditmagic"

TEAM_SELF = 0
SELF_ID = 7
ENEMY_ID = 9


def xor_encode_body(msg_type: int, payload: bytes) -> bytes:
    """Encode a message body the way the wire carries it.

    Args:
        msg_type: Message type byte (kept in the clear).
        payload: Plaintext payload to XOR with the session table.

    Returns:
        ``[msg_type] + xor(payload)`` bytes.

    Raises:
        XorStaticKeyUnavailableError: If the repo's XOR static key is
            unavailable.
    """
    table = build_session_xor_table(MAGIC)
    encoded = bytearray(len(payload))
    for index in range(len(payload)):
        key = table[index] if index < len(table) else 0
        encoded[index] = payload[index] ^ key
    return bytes([msg_type]) + bytes(encoded)


def frame_message(timestamp_ms: int, body: bytes, direction: str) -> CapturedMessage:
    """Wrap one message body in a length-prefixed base64 frame.

    Args:
        timestamp_ms: Message timestamp.
        body: Full message body (type byte + encoded payload).
        direction: ``sent`` or ``received``.

    Returns:
        A captured message carrying the framed payload.
    """
    length = len(body)
    frame = bytes([length & 0xFF, (length >> 8) & 0xFF]) + body
    sent: bool = direction == "sent"
    return CapturedMessage(
        timestamp_ms=timestamp_ms,
        direction="sent" if sent else "received",
        payload=base64.b64encode(frame).decode("ascii"),
        ws_url="wss://tankpit.com/ws",
    )


def identity_message(timestamp_ms: int, tank_id: int) -> CapturedMessage:
    """Build a 0x21 TankIdentity for the given tank."""
    payload = bytes([TEAM_SELF, tank_id & 0xFF, tank_id >> 8]) + bytes(7)
    return frame_message(timestamp_ms, xor_encode_body(0x21, payload), "received")


def named_identity_message(timestamp_ms: int, tank_id: int, name: str) -> CapturedMessage:
    """Build a 0x21 TankIdentity carrying the tank's name."""
    payload = bytes([TEAM_SELF, tank_id & 0xFF, tank_id >> 8]) + bytes(7)
    payload += name.encode("ascii")
    return frame_message(timestamp_ms, xor_encode_body(0x21, payload), "received")


def movement_response_message(timestamp_ms: int, tank_id: int, x: int, y: int) -> CapturedMessage:
    """Build a 0x3D MovementResponse stating the tank's position."""
    payload = bytes([TEAM_SELF, tank_id & 0xFF, tank_id >> 8, x, y, 0, 0, 1, 0, 0, 0, 0])
    return frame_message(timestamp_ms, xor_encode_body(0x3D, payload), "received")


def tank_entry_message(timestamp_ms: int, tank_id: int, x: int, y: int) -> CapturedMessage:
    """Build a 0x28 TankEntry stating the tank's position."""
    payload = bytes([255, tank_id & 0xFF, tank_id >> 8, 16, 0, 0, 0, x, y, 0])
    return frame_message(timestamp_ms, xor_encode_body(0x28, payload), "received")


def aimed_shot_message(
    timestamp_ms: int,
    shooter_id: int,
    source: tuple[int, int],
    target: tuple[int, int],
    weapon: int,
) -> CapturedMessage:
    """Build a 0x53 ShootEvent with explicit source/target tiles."""
    payload = bytes(
        [
            TEAM_SELF,
            shooter_id & 0xFF,
            shooter_id >> 8,
            source[0],
            source[1],
            target[0],
            target[1],
            target[0],
            target[1],
            weapon,
            0,
            0,
        ]
    )
    return frame_message(timestamp_ms, xor_encode_body(0x53, payload), "received")


def sync_message(
    timestamp_ms: int, tank_id: int, rank: int, fuel: int, damage: int = 0
) -> CapturedMessage:
    """Build a long-form 0x2E TankStatusSync carrying an absolute fuel.

    Tunnel shape: the 0x2E container body repeats the 0x2E subtype
    byte, then carries the sync payload (team, id, damage, rank,
    lb_score, promo, fuel-bar flag, fuel).
    """
    payload = bytes(
        [
            0x2E,
            TEAM_SELF,
            tank_id & 0xFF,
            tank_id >> 8,
            damage,
            rank,
            0,
            0,
            0,
            0,
            1,
            fuel & 0xFF,
            fuel >> 8,
        ]
    )
    return frame_message(timestamp_ms, xor_encode_body(0x2E, payload), "received")


def short_sync_message(
    timestamp_ms: int, tank_id: int, rank: int, damage: int = 0
) -> CapturedMessage:
    """Build a short-form 0x2E TankStatusSync (rank/promo, no fuel)."""
    payload = bytes([0x2E, TEAM_SELF, tank_id & 0xFF, tank_id >> 8, damage, rank, 0, 0, 0, 2])
    return frame_message(timestamp_ms, xor_encode_body(0x2E, payload), "received")


def tank_remove_message(timestamp_ms: int, tank_id: int) -> CapturedMessage:
    """Build a 0x2E-tunneled 0x58 TankRemove (tracked type, no timeline slot)."""
    payload = bytes([0x58, tank_id & 0xFF, tank_id >> 8])
    return frame_message(timestamp_ms, xor_encode_body(0x2E, payload), "received")


def fuel_gain_message(timestamp_ms: int, fuel_total: int) -> CapturedMessage:
    """Build a 0x2E-tunneled 0x44 FuelGain with the new absolute fuel."""
    payload = bytes([0x44, fuel_total & 0xFF, fuel_total >> 8, 1])
    return frame_message(timestamp_ms, xor_encode_body(0x2E, payload), "received")


def deposit_message(timestamp_ms: int, fuel_total: int) -> CapturedMessage:
    """Build a 0x2E-tunneled 0x64 FuelDeposit with the new absolute fuel."""
    payload = bytes([0x64, fuel_total & 0xFF, fuel_total >> 8])
    return frame_message(timestamp_ms, xor_encode_body(0x2E, payload), "received")


def detonation_message(timestamp_ms: int) -> CapturedMessage:
    """Build a 0x2E-tunneled 0x45 MineDetonation at one position."""
    payload = bytes([0x45, 40, 41])
    return frame_message(timestamp_ms, xor_encode_body(0x2E, payload), "received")


def pickup_message(timestamp_ms: int) -> CapturedMessage:
    """Build a 0x2E-tunneled 0x43 single-record container pickup."""
    payload = bytes([0x43, 10, 20, 0, 0])
    return frame_message(timestamp_ms, xor_encode_body(0x2E, payload), "received")


def shot_message(timestamp_ms: int, shooter_id: int, weapon: int) -> CapturedMessage:
    """Build a 0x53 ShootEvent echo for the given shooter and weapon."""
    payload = bytes([TEAM_SELF, shooter_id & 0xFF, shooter_id >> 8, 1, 1, 2, 2, 2, 2, weapon, 0, 0])
    return frame_message(timestamp_ms, xor_encode_body(0x53, payload), "received")


def move_message(timestamp_ms: int, tank_id: int, path: str) -> CapturedMessage:
    """Build a 0x47 Movement echo walking ``path`` (nsew chars) from (5,5)."""
    payload = bytes([tank_id & 0xFF, tank_id >> 8, 5, 5, 0, 0, 0, 0, 0, 1, 0, 0])
    payload += path.encode("ascii")
    return frame_message(timestamp_ms, xor_encode_body(0x47, payload), "received")


def inventory_message(timestamp_ms: int, counts: list[int]) -> CapturedMessage:
    """Build a 0x49 Inventory snapshot with the five slot counts."""
    payload = bytes([1, *counts])
    return frame_message(timestamp_ms, xor_encode_body(0x49, payload), "received")


def deactivation_message(timestamp_ms: int, victim_id: int, killer_id: int) -> CapturedMessage:
    """Build a 0x41 Deactivation (kill) event."""
    payload = bytes([0, victim_id & 0xFF, victim_id >> 8, 0, killer_id & 0xFF, killer_id >> 8])
    return frame_message(timestamp_ms, xor_encode_body(0x41, payload), "received")


def equipment_gain_message(
    timestamp_ms: int, gained: list[int], show_message: bool
) -> CapturedMessage:
    """Build a 0x67 EquipmentGain (loud pickup or silent bundle)."""
    payload = bytes([1 if show_message else 0, *gained])
    return frame_message(timestamp_ms, xor_encode_body(0x67, payload), "received")


def tank_exit_message(timestamp_ms: int, tank_id: int) -> CapturedMessage:
    """Build a 0x29 TankExit announcement for the given tank."""
    payload = bytes([TEAM_SELF, tank_id & 0xFF, tank_id >> 8, 0, 0])
    return frame_message(timestamp_ms, xor_encode_body(0x29, payload), "received")


def sent_command_message(
    timestamp_ms: int, command: int, x: int = 0, y: int = 0
) -> CapturedMessage:
    """Build a sent client command frame (``!`` prefix + type + cmd + coords)."""
    payload = bytes([4, command, x, y])
    return frame_message(timestamp_ms, xor_encode_body(0x21, payload), "sent")


def make_session(
    messages: list[CapturedMessage],
    magic: str | None = MAGIC,
    start_timestamp_ms: int = 0,
) -> CaptureSession:
    """Assemble a capture session around the given messages.

    Args:
        messages: The session's captured messages.
        magic: XOR magic (None builds a magicless session).
        start_timestamp_ms: Session start anchor. Real captures share
            the calendar day with their messages — pass the fixture's
            first message timestamp when a consumer resolves
            wall-clock windows against the anchor
            (``diagnostics.fight_report.window_bounds_ms``).

    Returns:
        The assembled capture session.
    """
    return CaptureSession(
        session_id="validate-test",
        start_timestamp_ms=start_timestamp_ms,
        end_timestamp_ms=start_timestamp_ms + 100_000,
        base_url="https://tankpit.com/play",
        messages=messages,
        magic=magic,
        game_log=[],
        tank_names={},
    )


__all__ = [
    "ENEMY_ID",
    "MAGIC",
    "SELF_ID",
    "aimed_shot_message",
    "deactivation_message",
    "deposit_message",
    "detonation_message",
    "equipment_gain_message",
    "frame_message",
    "fuel_gain_message",
    "identity_message",
    "inventory_message",
    "make_session",
    "move_message",
    "movement_response_message",
    "named_identity_message",
    "pickup_message",
    "sent_command_message",
    "short_sync_message",
    "shot_message",
    "sync_message",
    "tank_entry_message",
    "tank_exit_message",
    "tank_remove_message",
    "xor_encode_body",
]
