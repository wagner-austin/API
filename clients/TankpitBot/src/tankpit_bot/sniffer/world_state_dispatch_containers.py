"""Container and mine dispatch, including the pickup dedup window.

Placement, detonation, pickup application with its duplicate-broadcast
guard, and the teleport displacement that expires a stale ferry
belief.
"""

from __future__ import annotations

from tankpit_bot import browser, protocol
from tankpit_bot.container.types import ContainerPickupRecordDict
from tankpit_bot.ledger.fuel_book import record_fuel_entry
from tankpit_bot.ledger.outcome.teleport import pending_teleport_target
from tankpit_bot.physics.damage import MINE_DETONATION_COST
from tankpit_bot.runtime_logging import (
    emit_diagnostic,
    emit_world,
)
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.sniffer.world_state_combat import (
    mark_teleport_landed,
)
from tankpit_bot.sniffer.world_state_containers import (
    update_world_state_from_container_pickup,
)
from tankpit_bot.state import (
    add_mine,
    remove_mine,
)
from tankpit_bot.types.constants import TERRAIN_FERRY

#: Window during which a repeated ContainerPickup with identical pickup
#: signature is treated as the server's duplicate broadcast (one to the
#: picker, one to the world view). Empirically the two broadcasts arrive
#: within ~1-200 ms; 500 ms is the comfortable upper bound.
PICKUP_DEDUP_WINDOW_MS: int = 500


def _dispatch_mine_placement(
    ws: WorldService,
    mine_type: int,
    tank_id: int,
    positions: list[tuple[int, int]],
) -> bool:
    """Dispatch tunneled mine placement into world state.

    Args:
        ws: World service instance.
        mine_type: Mine type from protocol payload.
        tank_id: ID of the placing tank.
        positions: Absolute mine coordinates.

    Returns:
        True after attempting to apply the placement.
    """
    self_state = ws.world_state["self_state"]
    team: int | None = None
    if self_state is not None and self_state["tank_id"] == tank_id:
        team = self_state["team"]
    else:
        tank_state = ws.world_state["tanks"].get(str(tank_id))
        if tank_state is not None:
            team = tank_state["team"]
    if team is None:
        return True
    timestamp_ms = browser.get_current_time_ms()
    for x, y in positions:
        ws.world_state = add_mine(
            ws.world_state,
            x,
            y,
            mine_type,
            tank_id,
            team,
            timestamp_ms,
        )
    return True


def _dispatch_mine_detonation(
    ws: WorldService,
    positions: list[tuple[int, int]],
) -> bool:
    """Dispatch tunneled mine detonation into world state.

    Args:
        ws: World service instance.
        positions: Absolute mine coordinates removed by the detonation.

    Returns:
        True after applying the removals.
    """
    timestamp_ms = browser.get_current_time_ms()
    for x, y in positions:
        ws.world_state = remove_mine(ws.world_state, x, y, timestamp_ms)
    return True


def _is_duplicate_pickup_broadcast(
    ws: WorldService,
    pickups: tuple[ContainerPickupRecordDict, ...],
) -> bool:
    """Suppress the second copy of a server-broadcast ContainerPickup.

    Builds a signature from the pickup records (x, y, remaining_volume
    tuples) and checks the per-session recent-pickup ledger. If the same
    signature was already seen within :data:`PICKUP_DEDUP_WINDOW_MS`,
    this is the world-view broadcast that pairs with the picker
    broadcast; return True and the caller skips the world-state update.

    Args:
        ws: World service instance carrying the dedup ledger.
        pickups: Pickup records from the decoded message.

    Returns:
        True when this is a duplicate of a recent broadcast (caller
        should skip), False on the first sighting (caller should apply).
    """
    signature = tuple((record["x"], record["y"], record["remaining_volume"]) for record in pickups)
    now_ms = browser.get_current_time_ms()
    last_seen = ws.recent_pickup_signatures.get(signature)
    if last_seen is not None and now_ms - last_seen <= PICKUP_DEDUP_WINDOW_MS:
        ws.recent_pickup_signatures[signature] = now_ms
        return True
    ws.recent_pickup_signatures[signature] = now_ms
    # Bound the ledger so it doesn't grow without limit during long
    # sessions. Drop entries older than 2 windows.
    cutoff = now_ms - 2 * PICKUP_DEDUP_WINDOW_MS
    ws.recent_pickup_signatures = {
        sig: ts for sig, ts in ws.recent_pickup_signatures.items() if ts >= cutoff
    }
    return False


def _apply_container_pickups(
    ws: WorldService,
    pickups: tuple[ContainerPickupRecordDict, ...],
) -> None:
    """Apply one decoded ContainerPickup body (single- or multi-record).

    Drops duplicate server broadcasts via :func:`_is_duplicate_pickup_broadcast`
    and forwards each unique record to the world-state mutator.

    Args:
        ws: World service instance.
        pickups: Tuple of pickup records from one wire message.
    """
    if _is_duplicate_pickup_broadcast(ws, pickups):
        return
    for record in pickups:
        update_world_state_from_container_pickup(
            ws,
            record["x"],
            record["y"],
            record["remaining_volume"],
        )
        emit_diagnostic(
            diagnostic_kind="container_pickup_dispatched",
            x=record["x"],
            y=record["y"],
            remaining_volume=record["remaining_volume"],
            is_partial=record["remaining_volume"] > 0,
        )


def _note_own_mine_hit(ws: WorldService, positions: list[tuple[int, int]]) -> None:
    """Stamp the walk-over signature when a detonation hits our tile.

    Args:
        ws: World service instance.
        positions: Detonated mine tiles from the 0x45 payload.
    """
    self_state = ws.world_state["self_state"]
    if self_state is None:
        return
    if (self_state["x"], self_state["y"]) not in positions:
        return
    ws.last_own_mine_hit_ms = browser.get_current_time_ms()
    ws.mark_mine_reveal_pending(ws.last_own_mine_hit_ms)
    emit_world(
        "MINE_WALKOVER: detonation on own tile (%d,%d) - next approach teleports",
        self_state["x"],
        self_state["y"],
    )


def _expire_disproven_ferry_belief(ws: WorldService, requested_x: int, requested_y: int) -> None:
    """Delete a ferry belief the displaced landing just disproved.

    Flags s9-7/8 (2026-07-30, 17 extras burned): the equipment hop
    teleported to a 60-second-old ferry belief, the server displaced
    the landing -- its receipt that nothing boardable sits there --
    but the belief survived, so the identical boarding plan re-derived
    every lap. Ferries move; a displacement off a believed ferry tile
    is the wire's own proof the belief is stale, and consuming it here
    means the next derivation plans from truth (the same
    receipt-consumption discipline as code=4 and friendly-fire).

    Args:
        ws: World service instance.
        requested_x: The teleport's requested landing X.
        requested_y: The teleport's requested landing Y.
    """
    key = f"{requested_x},{requested_y}"
    tile = ws.world_state["terrain"].get(key)
    if tile is None or tile["terrain_type"] != TERRAIN_FERRY:
        return
    del ws.world_state["terrain"][key]
    emit_world(
        "FERRY_BELIEF_EXPIRED: displaced landing disproved ferry at (%d,%d)",
        requested_x,
        requested_y,
    )
    emit_diagnostic(
        diagnostic_kind="ferry_belief_expired",
        x=requested_x,
        y=requested_y,
    )


def _tank_stands_at(ws: WorldService, x: int, y: int) -> bool:
    """Return whether any registry tank occupies the tile.

    The displacement-tombstone exemption: a known tank body on the
    aimed tile fully explains a one-tile displacement (combat closes
    aim at the enemy's own tile and displace by one every time), so
    no mystery evidence is recorded for it.

    Args:
        ws: World service instance.
        x: Tile X.
        y: Tile Y.

    Returns:
        True when a tank in the registry stands on ``(x, y)``.
    """
    return any(tank["x"] == x and tank["y"] == y for tank in ws.world_state["tanks"].values())


def _emit_teleport_displacement(ws: WorldService) -> None:
    """Emit a receipt when the server landed the tank off the requested tile.

    Flag s2-7 (run bot-20260730-000030): beside the orange minefield
    the user watched teleports get "put back to the safe location"
    with nothing in the stream to prove it. The SelfMovement update
    precedes the landed confirm on the wire, so at confirm time the
    self position IS the landed tile; a mismatch against the
    executor's recorded dispatch target is a server displacement
    (mines on the landing, occupancy, refused ground). Exact landings
    stay silent -- the receipt exists to make bounce-backs visible in
    ``make analyze``, not to echo every teleport; combat closes that
    aim at the enemy's own tile displace by one routinely, and the
    ``displacement`` field lets the analyzer bucket those apart from
    minefield ejections.

    Args:
        ws: World service instance.
    """
    pending = pending_teleport_target(ws.ledger)
    self_state = ws.world_state["self_state"]
    if pending is None or self_state is None:
        return
    requested_x, requested_y = pending
    if (self_state["x"], self_state["y"]) == (requested_x, requested_y):
        return
    emit_world(
        "TELEPORT_DISPLACED: requested (%d,%d) landed (%d,%d)",
        requested_x,
        requested_y,
        self_state["x"],
        self_state["y"],
    )
    emit_diagnostic(
        diagnostic_kind="teleport_displacement",
        requested_x=requested_x,
        requested_y=requested_y,
        landed_x=self_state["x"],
        landed_y=self_state["y"],
        displacement=abs(self_state["x"] - requested_x) + abs(self_state["y"] - requested_y),
    )
    # The receipt is EVIDENCE, not just observability (2026-08-21 —
    # for four months it fed nothing, so the same hop could re-certify
    # against mine-blind beliefs forever: the 08-05 534-refusal
    # session and the marooning escape/harvest loops). A chebyshev
    # >= 2 "displacement" is the REFUSAL signature — the tank landed
    # back at its origin (137/137 archived receipts) — and the
    # landing selector consumes the ring-blocked verdict through the
    # composed decision terrain.
    chebyshev = max(abs(self_state["x"] - requested_x), abs(self_state["y"] - requested_y))
    ws.mark_landing_refused(
        requested_x,
        requested_y,
        chebyshev,
        browser.get_current_time_ms(),
    )
    if chebyshev == 1 and not _tank_stands_at(ws, requested_x, requested_y):
        # Routine one-tile displacement with NO known tank on the
        # aimed tile: an invisible occupant (hidden mine) displaced
        # us. Operator doctrine 2026-08-27: one displacement is
        # enough information — never re-aim the tile until the
        # evidence ages out or a radar reveal explains it. Aims at a
        # tank's own body stay exempt: those displace by one
        # legitimately on every combat close.
        ws.mark_displacement_tombstone(
            requested_x,
            requested_y,
            browser.get_current_time_ms(),
        )
    _expire_disproven_ferry_belief(ws, requested_x, requested_y)


def _dispatch_container_message(ws: WorldService, decoded: protocol.BinaryMessage) -> bool:
    """Dispatch container-level messages (mines, pickup, teleport landed).

    Args:
        ws: World service instance.
        decoded: Decoded binary protocol message.

    Returns:
        True if the message was handled, False otherwise.
    """
    match decoded:
        case {
            "msg_type": 0x4B,
            "mine_type": int(mine_type),
            "tank_id": int(tank_id),
            "positions": list(positions),
        }:
            return _dispatch_mine_placement(ws, mine_type, tank_id, positions)
        case {"msg_type": 0x45, "positions": list(positions)}:
            record_fuel_entry(book=ws.fuel_book, kind="detonation", lo=-MINE_DETONATION_COST, hi=0)
            _note_own_mine_hit(ws, positions)
            return _dispatch_mine_detonation(ws, positions)
        case {"msg_type": "container_pickup", "pickups": tuple(pickups)}:
            _apply_container_pickups(ws, pickups)
            return True
        case {"msg_type": "teleport_landed"}:
            emit_world("TELEPORT_LANDED: server confirmed teleport")
            _emit_teleport_displacement(ws)
            mark_teleport_landed(ws)
            return True
    return False


__all__ = [
    "PICKUP_DEDUP_WINDOW_MS",
]
