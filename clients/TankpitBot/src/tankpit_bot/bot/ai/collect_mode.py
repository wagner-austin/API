"""COLLECT mode: the per-tick decision for gathering resources.

Holds the arbiter and its sense/safety gates. The individual outcomes
it selects between are
:mod:`tankpit_bot.bot.ai.collect_mode_outcomes`.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.collect_common import COLLECT_SCORE
from tankpit_bot.bot.ai.collect_hops import (
    larder_harvest,
)
from tankpit_bot.bot.ai.collect_locks import continue_or_release_lock
from tankpit_bot.bot.ai.collect_mode_outcomes import (
    _desync_rescan_decision,
    _escape_under_fire_decision,
    _exhausted_collect_outcome,
    _scan_on_landing_decision,
)
from tankpit_bot.bot.ai.collect_pickups import (
    mine_clearance_decision,
    select_and_pickup_equipment,
    select_and_pickup_fuel,
)
from tankpit_bot.bot.ai.context import (
    DecideCtx,
    can_use_radar,
)
from tankpit_bot.bot.ai.equipment_search import describe_container_search
from tankpit_bot.bot.ai.forage import plan_forage_search
from tankpit_bot.bot.ai.resource_search import (
    make_resource_search_hop,
)
from tankpit_bot.bot.ai.scope_scout import scope_scout_for_ferry
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.runtime_logging import emit_ai


def decide_collect_mode(ctx: DecideCtx) -> TickDecisionDict | None:
    """Run the durable ``COLLECT`` owner for this tick.

    Cascade:

    0. Desync rescan: while a code=4 disproof of a remembered
       container awaits its radar resync, one radar outranks every
       pursuit branch (user ruling 2026-07-30: one stale item is
       worth a radar, never a second blind hop).
    1. Continue a held equipment or fuel lock from a previous tick.
    2. Scan-on-landing: fire one radar when the current viewport has
       zero scan coverage. Mirrors HUNT's scan_on_landing so the
       planner has a full picture (0x5A patch entries plus any tiles
       radar reveals) before committing to a pickup order. Without
       this gate, the cascade picks up whatever 0x5A enumerated first
       and only later discovers (via the forage step below) extra
       containers radar would have shown up front.
    3. Pick up the best equipment in the current viewport.
    4. Pick up the best fuel in the current viewport (skipped at cap).
    5. Larder ([[larder-plan]], 2026-07-27): harvest KNOWN stock
       before any discovery -- teleport to tracked equipment when
       below combat-ready, else to the best-scoring tracked fuel
       container (``min(volume, deficit) / cost``, profitable hops
       only). Larder hops hold a resource lock on the target and
       never spend the landing radar.
    6. Forage: radar when the viewport has unscanned tiles, or walk
       toward an unscanned tile so the next free radar covers it.
    7. Hop: teleport to the best-value fuel dot when nothing
       actionable remains here -- candidates are RANKED by
       ``dots_in_landing_viewport * walkable_fraction / cost``, hard
       gates are physics only (landing passable, affordable, not
       freshly scanned; the 2026-07-03 100%-walkable hard filter was
       replaced 2026-07-18 -- it starved the cascade). Landing
       auto-pickup makes the hop partially self-funding. With an
       empty dot atlas the hop opens the map first.

    Args:
        ctx: Decision context.

    Returns:
        Mode-owned collection decision, or ``None`` when every cascade
        branch declines but fuel is above ``fuel_low_threshold`` -- the
        tank is stocked, so collection is DONE and the caller should
        hand the tick to the hunt owner. Live run 2026-07-06 hit this
        state at fuel 1100 (every dot-hop landing filtered out) and
        wrongly exited ``out_of_fuel`` instead of going hunting.

    Raises:
        SessionExitError: When every cascade branch declines AND fuel is
            at or below ``fuel_low_threshold`` -- the bot is marooned
            and cannot produce a legal collection action, so the
            session ends with ``out_of_fuel`` (user contract
            2026-07-02).
    """
    gate_decision, base_state = _sense_and_safety_gates(ctx, ctx.base)
    if gate_decision is not None:
        return gate_decision

    locked_decision, base_state = continue_or_release_lock(ctx, base_state)
    if locked_decision is not None:
        return locked_decision

    equip_decision = select_and_pickup_equipment(ctx, base_state)
    if equip_decision is not None:
        return equip_decision

    fuel_decision = select_and_pickup_fuel(ctx, base_state)
    if fuel_decision is not None:
        return fuel_decision

    emit_ai(
        "no actionable collect target (equipment: %s; fuel: %s)",
        describe_container_search(
            ctx.filtered,
            ctx.self_state,
            ctx.terrain,
            want_fuel=False,
        ),
        describe_container_search(
            ctx.filtered,
            ctx.self_state,
            ctx.terrain,
            want_fuel=True,
            minimum_volume=1,
        ),
    )

    clearance_decision = mine_clearance_decision(ctx, base_state)
    if clearance_decision is not None:
        return clearance_decision

    larder_decision = larder_harvest(ctx, base_state)
    if larder_decision is not None:
        return larder_decision

    # The larder just declined — if what stopped it was a water-locked
    # container with no fresh ferry belief, one FREE viewport pan at
    # the water beats paying for discovery ([[viewport-shift-protocol]]
    # scope scout; F5 ferry doctrine). A ferry the pan reveals makes
    # the next tick's larder hop ``ferry_served``.
    scout_decision = scope_scout_for_ferry(ctx, base_state)
    if scout_decision is not None:
        return scout_decision

    forage_decision = plan_forage_search(
        ctx,
        base_state,
        score=COLLECT_SCORE,
        behavior_mode="COLLECT",
        radar_affordable=can_use_radar(ctx),
    )
    if forage_decision is not None:
        return forage_decision

    search = make_resource_search_hop(
        ctx,
        mode="COLLECT",
        score=COLLECT_SCORE,
        reason="search_collect_local",
        ai_state=base_state,
    )
    if search is not None:
        return search

    return _exhausted_collect_outcome(ctx, base_state)


def _sense_and_safety_gates(
    ctx: DecideCtx,
    base_state: AIStateDict,
) -> tuple[TickDecisionDict | None, AIStateDict]:
    """Run the pre-pursuit gates: landing scan, escape, desync rescan.

    Landing scan gates BEFORE lock continuation (reordered 2026-07-30,
    flag s4-3): the user policy is "always radar right on landing,
    before any pickup" (2026-07-03), and a DISPLACED harvest landing
    keeps its lock -- running the lock first walked blind into the
    unobserved minefield three ticks straight. Clean suppressed
    landings still latch silently and fall through to the lock. The
    under-fire escape and the desync rescan follow in that order:
    survival beats resync, resync beats pursuit.

    Args:
        ctx: Decision context.
        base_state: Base AI state threaded through the gates.

    Returns:
        ``(decision, base_state)`` -- the first gate's decision (or
        ``None`` when all gates pass) and the state the remaining
        cascade must thread.
    """
    landing_scan, base_state = _scan_on_landing_decision(ctx, base_state)
    if landing_scan is not None:
        return landing_scan, base_state

    under_fire = _escape_under_fire_decision(ctx, base_state)
    if under_fire is not None:
        return under_fire, base_state

    return _desync_rescan_decision(ctx, base_state), base_state


__all__ = [
    "decide_collect_mode",
]
