"""Combat dispatch: shot events, aim recording, and shot fuel entries.

The 0x41/0x4A family and the ledger entries a shot produces. Called by
the tank dispatcher in :mod:`tankpit_bot.sniffer.world_state_dispatch`.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot import browser
from tankpit_bot.ledger.ammo_book import record_ammo_enemy_shot, record_ammo_shot
from tankpit_bot.ledger.damage_book import record_incoming_shot, record_own_shot_echo
from tankpit_bot.ledger.fuel_book import FuelEntryKind, record_fuel_entry
from tankpit_bot.physics.costs import (
    DUAL_SHOT_COST,
    HOMING_SHOT_COST,
    MISSILE_SHOT_COST,
    SINGLE_SHOT_COST,
)
from tankpit_bot.physics.damage import DUAL_HIT_VICTIM_COST
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.sniffer.world_state_combat import (
    mark_combat_hit,
)
from tankpit_bot.sniffer.world_state_tanks import (
    _update_tank_position,
)

log = get_logger(__name__)

_SHOT_ENTRY_KINDS: dict[int, FuelEntryKind] = {
    0: "shot_single",
    1: "shot_dual",
    2: "shot_missile",
    3: "shot_homing",
}

_SHOT_ENTRY_COSTS: dict[int, int] = {
    0: SINGLE_SHOT_COST,
    1: DUAL_SHOT_COST,
    2: MISSILE_SHOT_COST,
    3: HOMING_SHOT_COST,
}


def _record_shot_fuel_entry(ws: WorldService, shooter_id: int, weapon: int) -> None:
    """Record a 0x53 echo's fuel effect into the live fuel book.

    Own shots debit their physics cost exactly (homing may split its
    debit across the sync boundary, so its ceiling is -5 and the book
    seeds a carry); enemy shots are optional debits bounded by the
    worst known victim cost — the shot may have targeted someone else
    — and count toward the ammo book's armor feasibility bound
    (shields may only fall for observed incoming fire; unwired until
    the 2026-07-22 fighting soak caught ``enemy_shots`` frozen at 0,
    which would have raised a FALSE ammo divergence on the first
    armor-absorbed hit).

    Args:
        ws: World service instance.
        shooter_id: Who fired the shot.
        weapon: Weapon byte (0=single, 1=dual, 2=missile, 3=homing).
    """
    if weapon not in _SHOT_ENTRY_KINDS:
        return
    self_state = ws.world_state["self_state"]
    if self_state is not None and shooter_id == self_state["tank_id"]:
        cost = _SHOT_ENTRY_COSTS[weapon]
        hi = -(cost // 2) if weapon == 3 else -cost
        record_fuel_entry(book=ws.fuel_book, kind=_SHOT_ENTRY_KINDS[weapon], lo=-cost, hi=hi)
        record_ammo_shot(book=ws.ammo_book, weapon=weapon)
        record_own_shot_echo(ws.damage_book, weapon)
    else:
        record_fuel_entry(book=ws.fuel_book, kind="enemy_hit", lo=-DUAL_HIT_VICTIM_COST, hi=0)
        record_ammo_enemy_shot(book=ws.ammo_book)
        shooter = ws.world_state["tanks"].get(str(shooter_id))
        record_incoming_shot(
            ws.damage_book,
            shooter_id,
            shooter["name"] if shooter is not None else f"tank-{shooter_id}",
            weapon,
            browser.get_current_time_ms(),
        )


def _find_tank_at_tile(ws: WorldService, x: int, y: int, exclude_id: int) -> int:
    """Return the tank id occupying (x, y), or -1 if the tile is empty.

    Used by the ShootEvent dispatch to determine whether our shot landed
    on a tank (hit) or empty terrain (miss). Tile occupancy is the
    authoritative wire-side hit signal per JS Gg.prototype.h.

    Args:
        ws: World service instance.
        x: Tile x coordinate from ShootEvent.target_x.
        y: Tile y coordinate from ShootEvent.target_y.
        exclude_id: Tank id to skip (typically our own tank, since
            the bot never shoots itself).

    Returns:
        Tank id at the tile, or -1 if no tracked tank occupies it.
    """
    for tid_str, tank in ws.world_state["tanks"].items():
        if tank["x"] == x and tank["y"] == y and int(tid_str) != exclude_id:
            return tank["tank_id"]
    return -1


def _record_enemy_aim(
    ws: WorldService,
    shooter_id: int,
    aim_x: int,
    aim_y: int,
    weapon: int,
) -> None:
    """Persist the enemy's last barrel-aim coordinates on the tank state.

    Threats consumers (combat AI, recover-fuel route planner) read
    ``last_aim_*`` on the tank state to reason about which tiles the
    enemy may fire on next tick. The fields are wire-fresh on every
    0x53 ShootEvent so they decay naturally with the tank's
    ``last_wire_seen_ms``.

    Args:
        ws: World service instance.
        shooter_id: Enemy tank id that fired.
        aim_x: Wire-reported barrel-aim X at the moment of fire.
        aim_y: Wire-reported barrel-aim Y at the moment of fire.
        weapon: Weapon byte (used downstream to discriminate which
            aim-target tile applies; logged here for traceability).
    """
    from tankpit_bot.state.tank_mutations import set_tank_last_aim

    ws.world_state = set_tank_last_aim(
        ws.world_state,
        shooter_id,
        aim_x,
        aim_y,
        weapon,
        browser.get_current_time_ms(),
    )


def _dispatch_shoot_event(
    ws: WorldService,
    shooter_id: int,
    sx: int,
    sy: int,
    tx: int,
    ty: int,
    aim_x: int,
    aim_y: int,
    weapon: int,
) -> None:
    """Apply a 0x53 ShootEvent to world state.

    The wire carries TWO target-ish coordinates: ``(tx, ty)`` is the
    tile the shot ultimately resolves against, while ``(aim_x, aim_y)``
    is the shooter's barrel aim at the moment of fire. For straight
    shots (weapon=0 single, weapon=1 dual) the two coincide; for
    homing / missile fire they can diverge as the projectile redirects
    mid-flight. The split is empirically promoted from ``unk1`` /
    ``unk2`` in task #73 against production captures.

    Effects on world state:

    * Own shot -> tile-occupancy hit detection: lookup tank at the
      target tile (``tx, ty``), record victim id. The aim coords are
      logged for observability so live runs surface barrel-vs-impact
      drift on homing fire.
    * Enemy shot -> their source tile (``sx, sy``) is a fresh
      wire-sourced position update for the shooter; ``(aim_x, aim_y)``
      is recorded on the enemy tank as ``last_aim_x`` /
      ``last_aim_y`` so the combat AI can reason about their barrel
      direction next tick.

    Args:
        ws: World service instance.
        shooter_id: Who fired the shot.
        sx: Shooter source tile X.
        sy: Shooter source tile Y.
        tx: Shot target tile X (resolved impact tile).
        ty: Shot target tile Y (resolved impact tile).
        aim_x: Shooter's barrel-aim X at the moment of fire.
        aim_y: Shooter's barrel-aim Y at the moment of fire.
        weapon: Weapon byte (0=single, 1=dual, 2=missile, 3=homing).
    """
    self_state = ws.world_state["self_state"]
    own_tank_id = self_state["tank_id"] if self_state is not None else -1
    aim_drift = (aim_x, aim_y) != (tx, ty)
    if shooter_id == own_tank_id:
        # OUR_SHOT's (tx, ty) is the server's homing-tracked landing
        # tile, which is often off-viewport once the target teleports
        # away. Overwriting the registry with it poisoned the
        # user-contract stay-put loop: the planner's next shoot would
        # dispatch at the off-viewport coord, and the server rejects
        # shoot commands targeted outside the 18x18 viewport (see
        # [[shot-range]]). Pre-098d3d7 the registry was not refreshed
        # from own shots; the bot kept aiming at the last on-viewport
        # tile and the server auto-tracked with homing on every shot
        # -- unlimited homings until the kill. Restored 2026-06-26.
        victim_id = _find_tank_at_tile(ws, tx, ty, exclude_id=own_tank_id)
        log.info(
            "OUR_SHOT: weapon=%d src=(%d,%d) tgt=(%d,%d) aim=(%d,%d)%s victim_id=%d",
            weapon,
            sx,
            sy,
            tx,
            ty,
            aim_x,
            aim_y,
            " [drift]" if aim_drift else "",
            victim_id,
        )
        mark_combat_hit(ws, weapon, victim_id)
    elif shooter_id > 0:
        _update_tank_position(ws, shooter_id, sx, sy, "wire_0x53_shoot_event")
        _record_enemy_aim(ws, shooter_id, aim_x, aim_y, weapon)
        if aim_drift:
            log.info(
                "ENEMY_SHOT: tid=%d weapon=%d src=(%d,%d) tgt=(%d,%d) aim=(%d,%d) [drift]",
                shooter_id,
                weapon,
                sx,
                sy,
                tx,
                ty,
                aim_x,
                aim_y,
            )


__all__ = [
    "log",
]
