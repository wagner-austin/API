"""Tank message decoders.

This module handles decoding of tank-related messages:
tank info, entry, exit, status, and status sync.
"""

from __future__ import annotations

from tankpit_bot.container.decoders import (
    decode_container_message,
    is_container_pickup_structure,
)
from tankpit_bot.container.decoders.events import decode_container_pickup
from tankpit_bot.protocol.types import (
    BinaryMessage,
    TankEntryDict,
    TankExitDict,
    TankInfoDict,
    TankRemoveDict,
    TankStatusDict,
    TankStatusSyncDict,
)
from tankpit_bot.wire.helpers import require_min_length, x16


def decode_tank_info(data: bytes) -> TankInfoDict:
    """Decode tank info from XOR-decoded data.

    Args:
        data: XOR-decoded message body (without 0x21 prefix).

    Returns:
        Decoded tank info.

    Raises:
        DecodeError: If decoding fails.
    """
    require_min_length(data, 10, "TankInfo")
    # Trace-verified: a[0]=team, X(a[1:3])=tank_id, a[3:7]=decoration,
    # 24bit-BE(a[7:10])=persistent_tank_id (a.aa), a[10:]=name
    team = data[0]
    tank_id = x16(data[1], data[2])
    decoration_state = bytes(data[3:7])
    persistent_tank_id = 256 * (256 * data[7] + data[8]) + data[9]
    name = data[10:].decode("utf-8", errors="replace") if len(data) > 10 else ""
    return TankInfoDict(
        msg_type=0x21,
        tank_id=tank_id,
        team=team,
        decoration_state=decoration_state,
        persistent_tank_id=persistent_tank_id,
        name=name,
    )


def decode_tank_entry(data: bytes) -> TankEntryDict:
    """Decode tank entry from XOR-decoded data.

    Layout from tpclient.js Uf.h (V["("]), verified 2026-06-19:
      a[0]   = flags (255=known)
      a[1:3] = tank_id (LE u16)
      a[3]   = packed: team(0-1), rank_category(2-3), rank(4-7)
      a[4:7] = score (24-bit BE)
      a[7]   = x
      a[8]   = y

    Args:
        data: XOR-decoded message body.

    Returns:
        Decoded tank entry.

    Raises:
        DecodeError: If decoding fails.
    """
    require_min_length(data, 9, "TankEntry")
    packed = data[3]
    return TankEntryDict(
        msg_type=0x28,
        team=packed & 3,
        tank_id=x16(data[1], data[2]),
        rank=(packed >> 4) & 15,
        damage_state=(packed >> 2) & 3,
        score=256 * (256 * data[4] + data[5]) + data[6],
        x=data[7],
        y=data[8],
    )


def decode_tank_exit(data: bytes) -> TankExitDict:
    """Decode tank exit/elimination announcement from XOR-decoded data.

    Trace-verified from tpclient.js Vf.h (V[")"]):
      a[0]   = team
      a[1:3] = tank_id (LE u16)
      a[3]   = was_silent (1 = no display text emitted)
      a[4]   = was_eliminated (1 = "eliminated from the game",
                               0 = "left the game")

    Args:
        data: XOR-decoded message body (without 0x29 prefix).

    Returns:
        Decoded tank exit.

    Raises:
        DecodeError: If decoding fails.
    """
    require_min_length(data, 5, "TankExit")
    return TankExitDict(
        msg_type=0x29,
        team=data[0],
        tank_id=x16(data[1], data[2]),
        was_silent=data[3] == 1,
        was_eliminated=data[4] == 1,
    )


def decode_tank_remove(data: bytes) -> TankRemoveDict:
    """Decode tank removal from world from XOR-decoded data.

    Trace-verified from tpclient.js Ug.h (V.X):
      a[0:2] = tank_id (LE u16)

    Args:
        data: XOR-decoded message body (without 0x58 prefix).

    Returns:
        Decoded tank remove.

    Raises:
        DecodeError: If decoding fails.
    """
    require_min_length(data, 2, "TankRemove")
    return TankRemoveDict(msg_type=0x58, tank_id=x16(data[0], data[1]))


def decode_tank_status_sync(data: bytes) -> TankStatusSyncDict:
    """Decode tank status sync from XOR-decoded data.

    Args:
        data: XOR-decoded message body (without 0x2E prefix).

    Returns:
        Decoded tank status sync.

    Raises:
        DecodeError: If decoding fails.
    """
    require_min_length(data, 8, "TankStatusSync")
    # Layout from tpclient.js Og.h (V["."]), verified 2026-06-19:
    #   a[0]    = subtype/team
    #   a[1:3]  = tank_id (LE u16)
    #   a[3]    = damage_state (b.u — dual-purpose: rank_category on init,
    #             overwritten with damage during gameplay)
    #   a[4]    = rank (b.l)
    #   a[5:8]  = lb_score (24-bit BE)
    #   If len > 8:
    #     a[8]  = promo_state
    #     a[9]  = has_fuel_bar
    #     a[10:12] = fuel (LE u16)
    # Caveat (2026-06-19): when this decoder is reached via the
    # subtype-first dispatcher for ``subtype == 0x2E`` (i.e. the
    # outer 0x2E body's first byte happened to ALSO be 0x2E), the
    # caller strips that byte and passes us ``inner`` = body[1:].
    # That throws away one byte of payload, so ``promo_state`` ends
    # up ``None`` for those samples even when the wire carried it.
    # The 9-byte length shortcut in ``decode_0x2e_message`` passes
    # the full body and recovers promo_state correctly. Crack
    # confirmed: 74/74 ex-TankStatusShort 9-byte bodies route via
    # the length shortcut with proper promo_state in [0, 5].
    subtype = data[0]
    tank_id = x16(data[1], data[2])
    damage_state = data[3]
    rank = data[4]
    lb_score = 256 * (256 * data[5] + data[6]) + data[7] if len(data) > 7 else 0

    promo_state: int | None = data[8] if len(data) >= 9 else None
    if len(data) >= 12:
        fuel: int | None = x16(data[10], data[11])
    else:
        fuel = None

    return TankStatusSyncDict(
        msg_type=0x2E,
        subtype=subtype,
        tank_id=tank_id,
        damage_state=damage_state,
        rank=rank,
        lb_score=lb_score,
        promo_state=promo_state,
        fuel=fuel,
    )


def decode_tank_status(data: bytes) -> TankStatusDict:
    """Decode full tank status from XOR-decoded data.

    Args:
        data: XOR-decoded message body (without 0x3E prefix).

    Returns:
        Decoded tank status.

    Raises:
        DecodeError: If decoding fails.
    """
    require_min_length(data, 13, "TankStatus")
    info_byte = data[0]
    team = info_byte & 0x03
    damage_state = (info_byte >> 2) & 0x03
    rank = (info_byte >> 4) & 0x0F
    tank_id = x16(data[1], data[2])
    decoration_state = bytes(data[3:7])
    lb_score = 256 * (256 * data[7] + data[8]) + data[9] if len(data) >= 10 else 0
    lb_pos = 256 * (256 * data[10] + data[11]) + data[12] if len(data) >= 13 else 0
    name = data[13:].decode("utf-8", errors="replace") if len(data) > 13 else ""
    return TankStatusDict(
        msg_type=0x3E,
        team=team,
        rank=rank,
        damage_state=damage_state,
        tank_id=tank_id,
        decoration_state=decoration_state,
        leaderboard_score=lb_score,
        leaderboard_position=lb_pos,
        name=name,
    )


def _is_radar_scan_structure(inner: bytes) -> bool:
    """Validate inner 0x4F payload length arithmetic.

    Mirrors JS ``ch.h``: a LE u16 cache-entry count, ``count`` 4-byte
    cache entries, then a 3-byte-aligned overlay tail. Bodies that fail
    the arithmetic fall through to the container path as
    ``unknown_container`` instead of raising mid-dispatch.
    """
    if len(inner) < 2:
        return False
    container_count = inner[0] | (inner[1] << 8)
    expected_container_bytes = container_count * 4
    if 2 + expected_container_bytes > len(inner):
        return False
    remaining = len(inner) - 2 - expected_container_bytes
    return remaining % 3 == 0


def _dispatch_protocol_tank(subtype: int, inner: bytes) -> BinaryMessage | None:
    """Subtypes that tunnel a protocol tank or movement message."""
    from tankpit_bot.protocol.decoders.movement import (
        decode_movement,
        decode_movement_response,
    )

    if subtype == 0x21 and len(inner) >= 10:
        return decode_tank_info(inner)
    if subtype == 0x28 and len(inner) >= 9:
        # JS Uf.h reads a[0..8] = 9 bytes. Lowering the threshold from
        # 10 to 9 matches the decoder's require_min_length(9) and the
        # JS source; without it the lone 10-byte 0x2E body with
        # subtype 0x28 in 150 production captures fell through to the
        # length-based "TankUpdateCompact" path. See
        # analysis_scripts/crack_tank_update.py.
        return decode_tank_entry(inner)
    if subtype == 0x2E and len(inner) >= 8:
        return decode_tank_status_sync(inner)
    if subtype == 0x3D and len(inner) >= 11:
        return decode_movement_response(inner)
    if subtype == 0x3E and len(inner) >= 13:
        return decode_tank_status(inner)
    if subtype == 0x47 and len(inner) >= 12:
        return decode_movement(inner)
    if subtype == 0x58 and len(inner) >= 2:
        # 231 corpus samples (analysis_scripts audit). Tunneled
        # TankRemove (Ug.h) inside 0x2E -- previously fell through to
        # UNKNOWN_CONTAINER.
        return decode_tank_remove(inner)
    return None


def _dispatch_protocol_resource(subtype: int, inner: bytes) -> BinaryMessage | None:
    """Subtypes that tunnel a protocol fuel/inventory/equipment message."""
    from tankpit_bot.protocol.decoders.resources import (
        decode_equipment_gain,
        decode_equipment_toggle,
        decode_fuel_deposit,
        decode_fuel_gain,
        decode_inventory,
    )

    if subtype == 0x44 and len(inner) >= 3:
        return decode_fuel_gain(inner)
    if subtype == 0x49 and len(inner) >= 6:
        return decode_inventory(inner)
    if subtype == 0x64 and len(inner) >= 2:
        return decode_fuel_deposit(inner)
    if subtype == 0x67 and len(inner) >= 6:
        return decode_equipment_gain(inner)
    if subtype == 0x74 and len(inner) >= 5:
        return decode_equipment_toggle(inner)
    return None


def _dispatch_protocol_world(subtype: int, inner: bytes) -> BinaryMessage | None:
    """Subtypes that tunnel a protocol world/combat/radar/misc message.

    Loose-body decoders (Sync, ActionDone, Viewport) are length-clamped
    so that container-only types of the same subtype byte but different
    body length (e.g. teleport_landed at 1 byte, tank_update_full at 15
    bytes with XOR-noise subtype 0x3F) fall through to length-based
    container identification.
    """
    from tankpit_bot.protocol.decoders.combat import (
        decode_deactivation,
        decode_shoot_event,
    )
    from tankpit_bot.protocol.decoders.map_data import decode_map_data
    from tankpit_bot.protocol.decoders.radar import (
        decode_radar_result,
        decode_radar_scan_result,
    )
    from tankpit_bot.protocol.decoders.world import (
        decode_supervisor,
        decode_sync,
        decode_terrain_update,
        decode_viewport_update,
    )

    if subtype == 0x3F and len(inner) == 1:
        return decode_sync(inner)
    if subtype == 0x41 and len(inner) >= 6:
        return decode_deactivation(inner)
    if subtype == 0x46 and len(inner) >= 2:
        return decode_radar_result(inner)
    if subtype == 0x4A:
        return decode_terrain_update(inner)
    if subtype == 0x4C and len(inner) >= 2:
        # 2941 corpus samples were being mis-identified as
        # length-based "WorldState" (500+ byte container blobs).
        # They are tunneled 0x4C MapData (Ig.h): u16 LE RLE byte
        # count + RLE fuel-dot section + 5-byte tank entries to
        # body end. See analysis_scripts/crack_tank_update.py for
        # the MapData layout proof.
        return decode_map_data(inner)
    if subtype == 0x4F and _is_radar_scan_structure(inner):
        return decode_radar_scan_result(inner)
    if subtype == 0x52 and len(inner) >= 3:
        # 705 corpus samples (analysis_scripts audit). Tunneled
        # CommandResult inside 0x2E -- previously fell through to
        # UNKNOWN_CONTAINER.
        return decode_supervisor(inner)
    if subtype == 0x53 and len(inner) >= 10:
        return decode_shoot_event(inner)
    if subtype == 0x5A and len(inner) >= 2:
        return decode_viewport_update(inner)
    return None


def _dispatch_protocol_misc(subtype: int, inner: bytes) -> BinaryMessage | None:
    """Subtypes that tunnel a protocol misc message.

    Covers ActionDone (0x54), BuildPickup (0x42), Statistics (0x56),
    and ChatMessage (0x4D). The first three were ground-truthed
    against production captures (analysis_scripts/crack_tank_update.py):
    0x56 fires on 239/239 samples in the corpus, 0x42 on 2/2 own-tank
    build/pickup events, and 0x54 is the ~1-byte ActionDone heartbeat.
    Chat is 0x2E-tunneled like the rest of the event stream: corpus
    sweep 2026-07-29 (320 sessions) found chat ONLY inside 0x2E,
    always exactly 5 inner bytes (``sender_id(2 LE) + msg_id + x +
    y``, sniff-20260729-214411 -- 8 echoed sends), never top-level.
    """
    from tankpit_bot.protocol.decoders.session_events import (
        decode_action_done,
        decode_build_pickup,
        decode_chat_message,
        decode_statistics,
    )

    if subtype == 0x42 and len(inner) >= 9:
        return decode_build_pickup(inner)
    if subtype == 0x4D and len(inner) >= 5:
        return decode_chat_message(inner)
    if subtype == 0x54 and len(inner) >= 1:
        return decode_action_done(inner)
    if subtype == 0x56 and len(inner) >= 12:
        return decode_statistics(inner)
    return None


def decode_0x2e_message(data: bytes) -> BinaryMessage:
    """Decode a 0x2E container body - single source of truth.

    Subtype-first dispatch covers every protocol-tunneled type (handed
    to the protocol decoder for that subtype). Container-only subtypes
    with length-based variants (PositionUpdate, ContainerPickup vs
    DeactivationDeath, PlayerListShort vs PlayerListExtended, mine
    placement / detonation) and container types without a unique
    subtype byte (TipNotification, TankUpdate*, TeleportLanded, etc.)
    are handled by `decode_container_message`, called as a fallback.

    Args:
        data: XOR-decoded body (without the outer 0x2E byte).

    Returns:
        Decoded `BinaryMessage`. `BinaryMessage` includes `ContainerMessage`
        so callers see one unified union.
    """
    if len(data) < 1:
        return decode_container_message(data)
    subtype = data[0]
    inner = data[1:]
    result = _dispatch_protocol_tank(subtype, inner)
    if result is not None:
        return result
    result = _dispatch_protocol_resource(subtype, inner)
    if result is not None:
        return result
    result = _dispatch_protocol_world(subtype, inner)
    if result is not None:
        return result
    result = _dispatch_protocol_misc(subtype, inner)
    if result is not None:
        return result
    # Multi-record ContainerPickup (subtype 0x43 + N*4 bytes of records,
    # N >= 1). The JS V.C = $g handler reads repeating 4-byte
    # ``[x, y, cache_lo, cache_hi]`` records; in the 0x2E envelope each
    # record is a container pickup notification. Corpus 2026-06-20:
    # 2653 single-record, 80 two-record, 2 three-record samples.
    # Replaces the previous ``if len(data) == 9: return
    # decode_tank_status_sync(data)`` shortcut, whose "74 sane samples"
    # were all 0x43-prefixed -- they were 2-record CacheUpdates being
    # misread as Og.h short-form bodies (where subtype=0x43 silently
    # became team=67, an invalid team byte).
    if subtype == 0x43 and is_container_pickup_structure(data):
        return decode_container_pickup(data)
    return decode_container_message(data)


__all__ = [
    "decode_0x2e_message",
    "decode_tank_entry",
    "decode_tank_exit",
    "decode_tank_info",
    "decode_tank_remove",
    "decode_tank_status",
    "decode_tank_status_sync",
]
