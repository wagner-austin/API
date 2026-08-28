"""COLLECT mode: the per-tick decision for gathering resources.

Holds the arbiter and its sense/safety gates. The individual outcomes
it selects between are
:mod:`tankpit_bot.bot.ai.collect_mode_outcomes`.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.block_harvest import plan_block_harvest_leg
from tankpit_bot.bot.ai.collect_common import COLLECT_SCORE
from tankpit_bot.bot.ai.collect_hops import (
    larder_harvest,
)
from tankpit_bot.bot.ai.collect_locks import continue_or_release_lock
from tankpit_bot.bot.ai.collect_mode_outcomes import (
    _desync_rescan_decision,
    _escape_under_fire_decision,
    _exhausted_collect_outcome,
    _mine_reveal_scan_decision,
    _scan_on_landing_decision,
)
from tankpit_bot.bot.ai.collect_pickups import (
    mine_clearance_decision,
    select_and_pickup_equipment,
    select_and_pickup_fuel,
)
from tankpit_bot.bot.ai.combat_opportunity import collect_return_fire
from tankpit_bot.bot.ai.context import (
    DecideCtx,
)
from tankpit_bot.bot.ai.equipment_atlas import plan_atlas_equipment_hop
from tankpit_bot.bot.ai.equipment_search import describe_container_search
from tankpit_bot.bot.ai.forage import plan_forage_search
from tankpit_bot.bot.ai.quad_sweep import plan_quad_sweep
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
    4b. Block harvest ([[quad-sweep-doctrine]]): frame a swept-block
       container with a free scope shift (or walk a leg toward a far
       one) so the next tick's pickup branches serve it -- strictly
       cheaper than any larder teleport while block stock remains.
    5. Larder ([[larder-plan]], 2026-07-27): harvest KNOWN stock
       before any discovery -- teleport to tracked equipment when
       below combat-ready, else to the best-scoring tracked fuel
       container (``min(volume, deficit) / cost``, profitable hops
       only). Larder hops hold a resource lock on the target and
       never spend the landing radar.
    5b. Quad sweep ([[quad-sweep-doctrine]], reordered 2026-08-13 —
       HUD flags 8/9/14, known stock preempts scanning): with extras
       stocked and the 31x31 block substantially unscanned, the
       opposite-corners recon fires once every collection branch
       above declined. A mid-sweep reveal is collected next tick and
       the movement aborts the remainder, so the sweep scans until
       found, not four windows by ritual.
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

    pursuit_decision = _known_stock_pursuit(ctx, base_state)
    if pursuit_decision is not None:
        return pursuit_decision

    # Equipment atlas hop ([[equipment-system]] hotspot law,
    # 2026-08-28): when nothing believed collectible remains in
    # reach, teleport to corpus-proven equipment ground instead of
    # buying blind reveals -- the landing's own viewport shows
    # whatever sits there.
    atlas_decision = plan_atlas_equipment_hop(ctx, base_state)
    if atlas_decision is not None:
        return atlas_decision

    # Quad sweep ([[quad-sweep-doctrine]], reordered 2026-08-13, HUD
    # flags 8/9/14): recon runs only when every collection branch
    # above declined -- known stock preempts scanning structurally.
    # A mid-sweep reveal is collected next tick; the movement aborts
    # the sweep's remainder via the anchor latch, making the sweep an
    # incremental scan-until-found. Since 2026-08-28 it is also
    # hoard-gated: below the radar hunt bar it declines outright and
    # the atlas hop above is the discovery strategy.
    sweep_decision = plan_quad_sweep(ctx, base_state)
    if sweep_decision is not None:
        return sweep_decision

    forage_decision = plan_forage_search(
        ctx,
        base_state,
        score=COLLECT_SCORE,
        behavior_mode="COLLECT",
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


def _known_stock_pursuit(
    ctx: DecideCtx,
    base_state: AIStateDict,
) -> TickDecisionDict | None:
    """Serve KNOWN stock: clearance, block harvest, larder, ferry scout.

    The middle of the cascade (steps between the in-window pickups and
    discovery), cheapest tool first:

    1. Mine clearance: a shot that exposes a covered container.
    2. Block harvest ([[quad-sweep-doctrine]]): a free scope shift
       that frames a swept-block container (or a walk leg toward a
       far one) -- strictly cheaper than any larder teleport, so it
       outranks the larder while the block still holds servable
       stock.
    3. Larder: teleport to tracked stock beyond the block.
    4. Ferry scope scout: when the larder's decline was a water-locked
       container with no fresh ferry belief, one FREE viewport pan at
       the water beats paying for discovery
       ([[viewport-shift-protocol]] scope scout; F5 ferry doctrine).
       A ferry the pan reveals makes the next tick's larder hop
       ``ferry_served``.

    Args:
        ctx: Decision context.
        base_state: Base AI state threaded from the gates.

    Returns:
        The first serving decision, or ``None`` when no known stock
        can be served (the cascade proceeds to discovery).
    """
    clearance_decision = mine_clearance_decision(ctx, base_state)
    if clearance_decision is not None:
        return clearance_decision

    harvest_decision = plan_block_harvest_leg(ctx, base_state)
    if harvest_decision is not None:
        return harvest_decision

    larder_decision = larder_harvest(ctx, base_state)
    if larder_decision is not None:
        return larder_decision

    return scope_scout_for_ferry(ctx, base_state)


def _sense_and_safety_gates(
    ctx: DecideCtx,
    base_state: AIStateDict,
) -> tuple[TickDecisionDict | None, AIStateDict]:
    """Run the pre-pursuit gates: landing scan, escape, mine reveal, desync.

    Landing scan gates BEFORE lock continuation (reordered 2026-07-30,
    flag s4-3): the user policy is "always radar right on landing,
    before any pickup" (2026-07-03), and a DISPLACED harvest landing
    keeps its lock -- running the lock first walked blind into the
    unobserved minefield three ticks straight. Clean suppressed
    landings still latch silently and fall through to the lock.
    Return fire follows (operator ruling 2026-08-26, Yuppler receipt:
    a stocked tank under fire spends the tick DAMAGING the attacker,
    with the refill riding as the shot's secondary — the survival
    bars inside the rung keep the escape doctrine senior when stock
    is broken). The under-fire escape, the own-mine-hit reveal scan
    (user ruling 2026-08-13, flag 2), and the desync rescan follow in
    that order: survival beats reveal, reveal beats resync, resync
    beats pursuit.

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

    return_fire = collect_return_fire(ctx, base_state)
    if return_fire is not None:
        return return_fire, base_state

    under_fire = _escape_under_fire_decision(ctx, base_state)
    if under_fire is not None:
        return under_fire, base_state

    mine_reveal = _mine_reveal_scan_decision(ctx, base_state)
    if mine_reveal is not None:
        return mine_reveal, base_state

    return _desync_rescan_decision(ctx, base_state), base_state


__all__ = [
    "decide_collect_mode",
]
