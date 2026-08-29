"""Combat preparation: weapon check, refuel, and target refresh.

The stages that run BEFORE a shot is possible -- confirming a
damaging weapon is loaded, topping off fuel for the hunt, and
reopening the map to refresh a stale target. Reads
:mod:`tankpit_bot.bot.ai.combat_target` only.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.combat_target import (
    _set_combat_target,
    block_combat_target_and_replan,
)
from tankpit_bot.bot.ai.context import (
    DecideCtx,
    make_decision,
)
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.ai.world_types import EnemyThreatDict
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import (
    make_map_open_command,
)
from tankpit_bot.runtime_logging import emit_ai


def _has_damaging_weapon_available(ctx: DecideCtx) -> bool:
    """Return True when at least one damaging weapon slot can fire this tick.

    A "damaging" slot is a dual or homing that is both enabled and
    stocked. Radars, missiles, and shields don't damage tanks; single
    (weapon=0) is what the server picks as the fallback when neither
    dual nor homing is available. The predicate distinguishes the two
    causes of a ``weapon=0`` miss:

    * ``afterimage_confirmed`` -- dual OR homing was available and the
      server still picked single, so the target is not at the aim tile.
    * ``ammo_exhaustion_miss`` -- neither dual nor homing was
      available, so the server routed to single by default. The miss
      says nothing about the target's presence.

    Used by the stationary-miss classifier in :func:`engage_target`
    (Bug 0.6): only the ``afterimage_confirmed`` case warrants
    blacklisting the target.

    Args:
        ctx: Decision context.

    Returns:
        True when ``dual_shots`` or ``homing_shots`` has both
        ``enabled=True`` and ``count > 0``.
    """
    duals = ctx.inventory["dual_shots"]
    homings = ctx.inventory["homing_shots"]
    return (duals["enabled"] and duals["count"] > 0) or (
        homings["enabled"] and homings["count"] > 0
    )


def refuel_for_hunt(ctx: DecideCtx, target: EnemyThreatDict) -> TickDecisionDict:
    """Delegate the tick to the fuel planner when hunting is fuel-starved.

    Threats sort nearest-first and teleport cost is monotone in
    distance, so an unaffordable nearest target means EVERY target is
    unaffordable. Blocking and replanning instead cascaded through the
    whole roster and ended in a map-reopen spin: run 20260611-025636
    spawned at fuel 620 -- above the fuel-low entry rule (500) but
    below every engagement's cost-plus-reserve -- and spent its entire
    240s on 115 map reopens without a single shot. Collecting fuel is
    the only decision that changes the blocked condition.

    Refuel-then-RESUME (user ruling 2026-07-27, closing the last
    voluntary live-target drop): the lock is KEPT through the fuel
    detour, so the 2026-07-25 resume machinery returns to this exact
    target once the tank can fund the trip -- run 183703's red-1 was
    deferred at fuel 239 under the old clear-and-reacquire and never
    hunted again (the guest won the fresh distance race). The old
    anti-spin property survives because each deferred tick delegates
    to a real collect decision (fuel strictly grows), not a map
    reopen.

    When the collect cascade itself declines (fuel healthy but nothing
    collectible in reach), the fuel situation is not going to improve
    this tick, so the unaffordable target is blocked and replanned
    instead of exiting the session -- the terminator that also bounds
    the pathological corner where cost + reserve exceeds the tank's
    own fuel capacity.

    Args:
        ctx: Decision context.
        target: The unaffordable combat target being deferred.

    Returns:
        Fuel recovery decision with combat target cleared, or a
        blocked-target replanning decision when collection declines.
    """
    # Lazy import: collect_mode imports clear_combat_target from
    # this module at import time.
    from tankpit_bot.bot.ai.collect_mode import decide_collect_mode

    locked_ctx = ctx.derive(_set_combat_target(ctx.base, target))
    decision = decide_collect_mode(locked_ctx)
    if decision is None:
        emit_ai(
            "refuel-for-hunt found nothing collectible at fuel %d, blocking %s",
            ctx.fuel,
            target["name"],
        )
        return block_combat_target_and_replan(ctx, target)
    return decision


def open_map_for_target(ctx: DecideCtx, target: EnemyThreatDict) -> TickDecisionDict:
    """Phase 0: Open map to get fresh enemy positions.


    Args:
        ctx: Decision context.
        target: Combat target to refresh.

    Returns:
        Map-open decision that locks the target.
    """
    emit_ai("open map to find %s", target["name"])
    return make_decision(
        make_map_open_command(),
        "HUNT",
        800,
        0,
        0,
        "find_target",
        AIStateDict(
            **{
                **_set_combat_target(ctx.base, target),
                "last_map_open_ms": ctx.timestamp_ms,
            }
        ),
        ctx.equip,
        reason_context={"target_name": target["name"]},
    )


__all__ = [
    "open_map_for_target",
    "refuel_for_hunt",
]
