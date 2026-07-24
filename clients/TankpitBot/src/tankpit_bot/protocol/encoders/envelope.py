"""Envelope encoding: any decoded BinaryMessage back to wire payload bytes.

Two public entry points, both exact inverses of the decode path:

- :func:`encode_message_payload` — the payload of a TOP-LEVEL frame
  (without the leading type byte), the inverse of what
  ``decode_message(msg_type, payload)`` consumes.
- :func:`encode_envelope_body` — the body of a 0x2E container frame
  (the inverse of ``decode_0x2e_message``): protocol messages get
  their msg_type prepended as the subtype byte; container-only
  messages already carry their lead byte.

Grouped helper dispatch mirrors ``decoders.routing`` to stay under the
complexity ceiling. Each group returns the ``(subtype, payload)`` pair
so the envelope can prepend the subtype without re-narrowing the
union.
"""

from __future__ import annotations

from tankpit_bot.container.encoders import (
    encode_container_pickup,
    encode_mine_detonation,
    encode_mine_placement,
    encode_teleport_landed,
    encode_unknown_container,
)
from tankpit_bot.protocol.encoders.combat import encode_deactivation, encode_shoot_event
from tankpit_bot.protocol.encoders.map_data import encode_map_data
from tankpit_bot.protocol.encoders.movement import (
    encode_movement,
    encode_movement_response,
)
from tankpit_bot.protocol.encoders.radar import (
    encode_enemy_detection,
    encode_radar_result,
    encode_radar_scan_result,
)
from tankpit_bot.protocol.encoders.resources import (
    encode_equipment_gain,
    encode_equipment_toggle,
    encode_fuel_deposit,
    encode_fuel_gain,
    encode_inventory,
)
from tankpit_bot.protocol.encoders.session_events import (
    encode_action_done,
    encode_active_forces,
    encode_active_players,
    encode_build_pickup,
    encode_chat_message,
    encode_connection_lost,
    encode_decoration,
    encode_ping_response,
    encode_promotion,
    encode_statistics,
    encode_top10,
)
from tankpit_bot.protocol.encoders.tank import (
    encode_tank_entry,
    encode_tank_exit,
    encode_tank_info,
    encode_tank_remove,
    encode_tank_status,
    encode_tank_status_sync,
)
from tankpit_bot.protocol.encoders.world import (
    encode_cache_update,
    encode_overlay_update,
    encode_supervisor,
    encode_supervisor_text,
    encode_sync,
    encode_terrain_update,
    encode_viewport_update,
)
from tankpit_bot.protocol.helpers import EncodeError
from tankpit_bot.protocol.types import BinaryMessage


def _pair_tank(message: BinaryMessage) -> tuple[int, bytes] | None:
    """Encode tank-family payloads, or None when the type is not ours."""
    if message["msg_type"] == 0x21:
        return 0x21, encode_tank_info(message)
    if message["msg_type"] == 0x28:
        return 0x28, encode_tank_entry(message)
    if message["msg_type"] == 0x29:
        return 0x29, encode_tank_exit(message)
    if message["msg_type"] == 0x2E:
        return 0x2E, encode_tank_status_sync(message)
    if message["msg_type"] == 0x3E:
        return 0x3E, encode_tank_status(message)
    if message["msg_type"] == 0x58:
        return 0x58, encode_tank_remove(message)
    return None


def _pair_movement_combat(message: BinaryMessage) -> tuple[int, bytes] | None:
    """Encode movement/combat payloads, or None when the type is not ours."""
    if message["msg_type"] == 0x3D:
        return 0x3D, encode_movement_response(message)
    if message["msg_type"] == 0x41:
        return 0x41, encode_deactivation(message)
    if message["msg_type"] == 0x47:
        return 0x47, encode_movement(message)
    if message["msg_type"] == 0x53:
        return 0x53, encode_shoot_event(message)
    return None


def _pair_resource(message: BinaryMessage) -> tuple[int, bytes] | None:
    """Encode resource payloads, or None when the type is not ours."""
    if message["msg_type"] == 0x44:
        return 0x44, encode_fuel_gain(message)
    if message["msg_type"] == 0x49:
        return 0x49, encode_inventory(message)
    if message["msg_type"] == 0x64:
        return 0x64, encode_fuel_deposit(message)
    if message["msg_type"] == 0x67:
        return 0x67, encode_equipment_gain(message)
    if message["msg_type"] == 0x74:
        return 0x74, encode_equipment_toggle(message)
    return None


def _pair_world(message: BinaryMessage) -> tuple[int, bytes] | None:
    """Encode world-sync/patch payloads, or None when the type is not ours."""
    if message["msg_type"] == 0x3C:
        return 0x3C, encode_supervisor_text(message)
    if message["msg_type"] == 0x3F:
        return 0x3F, encode_sync(message)
    if message["msg_type"] == 0x40:
        return 0x40, encode_overlay_update(message)
    if message["msg_type"] == 0x43:
        return 0x43, encode_cache_update(message)
    if message["msg_type"] == "chat_ack":
        return 0x43, bytes([1 if message["enabled"] else 0])
    if message["msg_type"] == "autoscroll_ack":
        return 0x41, bytes([1 if message["enabled"] else 0])
    if message["msg_type"] == 0x4A:
        return 0x4A, encode_terrain_update(message)
    if message["msg_type"] == 0x52:
        return 0x52, encode_supervisor(message)
    return None


def _pair_geometry(message: BinaryMessage) -> tuple[int, bytes] | None:
    """Encode map/radar/viewport payloads, or None when the type is not ours."""
    if message["msg_type"] == 0x46:
        return 0x46, encode_radar_result(message)
    if message["msg_type"] == 0x48:
        return 0x48, encode_enemy_detection(message)
    if message["msg_type"] == 0x4C:
        return 0x4C, encode_map_data(message)
    if message["msg_type"] == 0x4F:
        return 0x4F, encode_radar_scan_result(message)
    if message["msg_type"] == 0x5A:
        return 0x5A, encode_viewport_update(message)
    return None


def _pair_session(message: BinaryMessage) -> tuple[int, bytes] | None:
    """Encode session/roster payloads, or None when the type is not ours."""
    if message["msg_type"] == 0x2A:
        return 0x2A, encode_active_forces(message)
    if message["msg_type"] == 0x2B:
        return 0x2B, encode_promotion(message)
    if message["msg_type"] == 0x2F:
        return 0x2F, encode_active_players(message)
    if message["msg_type"] == 0x31:
        return 0x31, encode_top10(message)
    if message["msg_type"] == 0x42:
        return 0x42, encode_build_pickup(message)
    return None


def _pair_scoring(message: BinaryMessage) -> tuple[int, bytes] | None:
    """Encode scoring/heartbeat payloads, or None when the type is not ours."""
    if message["msg_type"] == 0x4D:
        return 0x4D, encode_chat_message(message)
    if message["msg_type"] == 0x4E:
        return 0x4E, encode_decoration(message)
    if message["msg_type"] == 0x54:
        return 0x54, encode_action_done(message)
    if message["msg_type"] == 0x56:
        return 0x56, encode_statistics(message)
    if message["msg_type"] == 0x60:
        return 0x60, encode_ping_response(message)
    if message["msg_type"] == 0x7E:
        return 0x7E, encode_connection_lost(message)
    return None


def _container_body(message: BinaryMessage) -> bytes | None:
    """Encode container-only bodies (lead byte included), or None."""
    if message["msg_type"] == "container_pickup":
        return encode_container_pickup(message)
    if message["msg_type"] == "teleport_landed":
        return encode_teleport_landed(message)
    if message["msg_type"] == "unknown_container":
        return encode_unknown_container(message)
    if message["msg_type"] == 0x45:
        return encode_mine_detonation(message)
    if message["msg_type"] == 0x4B:
        return encode_mine_placement(message)
    return None


def _require_protocol_pair(message: BinaryMessage) -> tuple[int, bytes]:
    """Resolve a protocol message to its (subtype, payload) pair.

    Args:
        message: Decoded binary message.

    Returns:
        The msg_type byte and the encoded payload.

    Raises:
        EncodeError: For container-only messages, which have no
            top-level protocol form.
    """
    for group in (
        _pair_tank,
        _pair_movement_combat,
        _pair_resource,
        _pair_world,
        _pair_geometry,
        _pair_session,
        _pair_scoring,
    ):
        pair = group(message)
        if pair is not None:
            return pair
    raise EncodeError(
        f"no protocol encoder for msg_type {message['msg_type']!r}; "
        "container-only messages must go through encode_envelope_body"
    )


def encode_message_payload(message: BinaryMessage) -> bytes:
    """Encode a protocol message's top-level frame payload.

    Args:
        message: Decoded binary message.

    Returns:
        Payload bytes WITHOUT the leading msg_type byte — the exact
        bytes ``decode_message(msg_type, payload)`` consumes.

    Raises:
        EncodeError: For container-only messages (they exist only
            inside a 0x2E envelope — use :func:`encode_envelope_body`).
    """
    return _require_protocol_pair(message)[1]


def encode_envelope_body(message: BinaryMessage) -> bytes:
    """Encode a message as a 0x2E container-frame body.

    Args:
        message: Decoded binary message.

    Returns:
        The full 0x2E body: for protocol messages the msg_type byte
        (the envelope subtype) followed by the payload; for
        container-only messages their body verbatim.
    """
    body = _container_body(message)
    if body is not None:
        return body
    subtype, payload = _require_protocol_pair(message)
    return bytes([subtype]) + payload


__all__ = [
    "encode_envelope_body",
    "encode_message_payload",
]
