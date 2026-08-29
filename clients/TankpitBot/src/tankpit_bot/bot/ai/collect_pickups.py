"""In-viewport pickup steps of the COLLECT cascade.

Walk-reachable equipment and fuel pickups (cascade steps 3-4) plus the
mine-clearance shot that exposes covered containers. Hops and larder
harvest live in :mod:`collect_hops`; lock continuation in
:mod:`collect_locks`.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.collect_common import COLLECT_SCORE
from tankpit_bot.bot.ai.context import DecideCtx, make_decision
from tankpit_bot.bot.ai.equipment_search import (
    find_equipment_candidates,
    find_fuel_candidates,
)
from tankpit_bot.bot.ai.intent import set_resource_target
from tankpit_bot.bot.ai.mine_clearance import (
    find_mine_clearance_shot,
    find_walk_clearance_shot,
)
from tankpit_bot.bot.ai.mode_gates import hunt_entry_permitted
from tankpit_bot.bot.ai.movement import walk_or_teleport
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import BotCommand, make_shoot_command
from tankpit_bot.inventory import inventory_counts
from tankpit_bot.physics.capacity import fuel_capacity
from tankpit_bot.physics.supervisor import equipment_pickup_refusal
from tankpit_bot.runtime_logging import emit_ai
from tankpit_bot.state.types import ContainerStateDict


def select_equipment_target(
    ctx: DecideCtx,
) -> tuple[ContainerStateDict, BotCommand] | None:
    """Return the nearest executable walk-reachable equipment target.

    Targets with a live teleport-approach mark are skipped: a marked
    container already ate a teleport without becoming collectable.

    Args:
        ctx: Decision context.

    Returns:
        ``(container, command)`` for the nearest executable equipment target, or
        ``None`` when no visible equipment target can currently be executed.
    """
    candidates = find_equipment_candidates(
        ctx.filtered,
        ctx.self_state,
        ctx.terrain,
    )
    if not candidates:
        return None

    container = candidates[0]
    command = walk_or_teleport(ctx, container["x"], container["y"], pickup_kind="equipment")
    if command is None:
        return None
    return (container, command)


def select_and_pickup_equipment(
    ctx: DecideCtx,
    base_state: AIStateDict,
) -> TickDecisionDict | None:
    """Pick up the best viewport equipment, unless every slot is full.

    User mechanic (2026-07-18, verbatim): equipment containers "fill
    whatever is empty. you will only get a full inventory message if
    all your items are full" -- so at full inventory a pickup can gain
    nothing and would burn a tick on a guaranteed code-7 rejection
    (8 wasted ticks in the 2026-07-18 5-minute run before this gate).
    The refusal is the shared ``equipment_pickup_refusal`` law in
    ``physics/supervisor.py``.
    """
    refusal = equipment_pickup_refusal(inventory_counts(ctx.inventory), ctx.self_state["rank"])
    if refusal is not None:
        return None
    selection = select_equipment_target(ctx)
    if selection is None:
        return None
    container, command = selection
    target_x = container["x"]
    target_y = container["y"]
    emit_ai("collect equipment at (%d,%d)", target_x, target_y)
    return make_decision(
        command,
        "COLLECT",
        COLLECT_SCORE,
        target_x,
        target_y,
        "equipment_restock",
        set_resource_target(base_state, "equipment", target_x, target_y),
        ctx.equip,
    )


# How long a clearance aim stays suppressed after its shot: the 0x45
# detonation follows the shot echo within a wire tick, so two server
# windows comfortably covers apply-lag without stalling a genuine
# re-clear (a recruit's 1-mine blast leaving covered neighbors).
_MINE_CLEARANCE_EFFECT_MS = 5_000


def mine_clearance_decision(
    ctx: DecideCtx,
    base_state: AIStateDict,
) -> TickDecisionDict | None:
    """Shoot the best mine-covered container in view, if any.

    User doctrine ([[flag-triage-20260729]] F3: "there is equipment
    that we can see under the orange mines"): a container under an
    enemy mine needs no path clearing -- one clear-line shot at the
    container's tile detonates the covering mine plus the full
    adjacent 3x3 at private+, and the exposed containers become
    ordinary pickups on the very next tick (the 0x45 detonation
    removes them from the world's mine layer). Mine shots consume NO
    inventory (user law 2026-07-30: "shooting a mine doesnt cost any
    inventory. you click and it shoots a single shot") -- the server
    routes a free single -- so the only spend is the shot's tick. The
    aim comes from the pure planner in
    :mod:`tankpit_bot.bot.ai.mine_clearance`; mines never occlude the
    shot line, only rock and movable land blocks do.

    Args:
        ctx: Decision context.
        base_state: AI state for the produced command.

    Returns:
        Shoot decision at the covered container, at the mine denying
        its landing, or at the first corridor mine corking the walk
        to wanted stock (HUD flags 3/6, 2026-08-13) — ``None`` when
        no such shot exists.
    """
    fuel_deficit = fuel_capacity(ctx.self_state["rank"]) - ctx.fuel
    aim = find_mine_clearance_shot(
        ctx.filtered,
        ctx.self_state,
        ctx.terrain,
        fuel_deficit=fuel_deficit,
        fuel_gain_per_walk_tile=_FUEL_GAIN_PER_WALK_TILE,
    )
    if aim is None:
        aim = find_walk_clearance_shot(
            ctx.filtered,
            ctx.self_state,
            ctx.terrain,
            equipment_wanted=(
                equipment_pickup_refusal(inventory_counts(ctx.inventory), ctx.self_state["rank"])
                is None
            ),
            fuel_deficit=fuel_deficit,
            fuel_gain_per_walk_tile=_FUEL_GAIN_PER_WALK_TILE,
        )
    if aim is None:
        return None
    aim_x, aim_y = aim
    aim_key = f"{aim_x},{aim_y}"
    if (
        aim_key == base_state["mine_clearance_aim_key"]
        and ctx.timestamp_ms - base_state["mine_clearance_shot_ms"] < _MINE_CLEARANCE_EFFECT_MS
    ):
        # The previous clearance shot at this exact tile has not had
        # its detonation applied yet (the 0x45 follows the shot echo
        # by up to a wire tick); re-aiming inside the window is the
        # live double-shot at (162,94), 01:59:57/:59 -- one shot
        # wasted on mines that were already dead on the server.
        return None
    emit_ai(
        "mine clearance: shooting covered container at (%d,%d) to expose the pickups",
        aim_x,
        aim_y,
    )
    return make_decision(
        make_shoot_command(aim_x, aim_y),
        "COLLECT",
        COLLECT_SCORE,
        aim_x,
        aim_y,
        "mine_clearance_shot",
        AIStateDict(
            **{
                **base_state,
                "mine_clearance_aim_key": aim_key,
                "mine_clearance_shot_ms": ctx.timestamp_ms,
            }
        ),
        ctx.equip,
    )


_MIN_FUEL_SIP_GAIN = 25
"""Smallest clamped transfer a fuel pickup dispatch is worth.

Below this the pickup is choreography churn, not restocking: every
dispatch costs a ~2 s tick and the transfer choreography another,
so a sub-25 sip spends ~4 s moving less fuel than half a shot costs.
Live receipt (HUD flag 3, run bot-20260813-195231 19:58): each
6-fuel clearance-shot cost re-opened a 6-fuel deficit and the
adjacent container was re-sipped between every shot -- shoot -6,
drink +6, alternating for the whole clearance sequence. The floor is
waived when the sip completes hunt readiness (fuel is the LAST bar,
so topping to cap of any size stays legal — the hunt-only-when-full
contract requires it)."""

_FUEL_GAIN_PER_WALK_TILE = 3
"""Minimum effective fuel gained per tile of walking for a pickup to pay.

Priced from MEASURED walking speed, not the tick: the user timed
15 cardinal tiles in 3.30 s (2026-08-06) -- ~0.22 s per tile, with
diagonals costing two Manhattan steps. The previous constant (25)
justified itself with "each walk tile costs roughly one 2-second
tick", off by ~10x; at the artax flag scene it refused an 80-fuel
pickup four tiles (under one second) away. Keeping the same
opportunity value of time the old constant implied (~12.5 fuel/s),
a tile now prices at ~0.22 s x 12.5 = ~3 fuel. The rule still
refuses true dregs (a 5-fuel sliver ten tiles off) while taking any
meaningfully stocked container within honest walking range.
"""


def pickup_not_worth_walk(
    ctx: DecideCtx,
    container: ContainerStateDict,
) -> bool:
    """Return True when the pickup's real transfer is not worth the walk.

    The server clamps a fuel pickup to ``min(volume, headroom)`` and
    answers ``code=5`` when clamped; since 2026-07-06/-19 that code is
    handled cleanly (container kept, no blacklist, ledger resolved),
    so a clamped transfer costs nothing but the walk. The predicate
    therefore rates the ACTUAL transfer against the walk distance.

    This replaces the binary overfill gate, whose formula
    (``fuel + walk + min(volume, headroom) > cap``) refused ANY
    clamped pickup at walk >= 1 -- and refused earlier the bigger the
    container: at fuel 600 it walked past a 1000-volume container one
    tile away, forfeiting a 500-fuel transfer (falsified 2026-07-19;
    the 2026-06-23 minimum-volume lesson -- fuel is fuel -- applies at
    the cap end too).

    Args:
        ctx: Decision context.
        container: The candidate fuel container.

    Returns:
        True when ``min(volume, headroom)`` falls below the walk
        pricing or the minimum-sip floor, unless the sip completes
        hunt readiness.
    """
    headroom = fuel_capacity(ctx.self_state["rank"]) - ctx.fuel
    effective_gain = min(container["volume"], headroom)
    walk_tiles = abs(container["x"] - ctx.self_state["x"]) + abs(
        container["y"] - ctx.self_state["y"]
    )
    # The minimum-sip floor (HUD flag 3, 2026-08-13): near cap, every
    # 6-fuel shot cost re-opened a 6-fuel deficit and the adjacent
    # container got re-sipped between every clearance shot -- shoot
    # -6, drink +6, four seconds a round trip. A sip below the floor
    # is dispatch churn, not restocking -- UNLESS it is the last
    # requirement before hunting (the hunt-only-when-full contract
    # needs fuel exactly at cap, so a readiness-completing top-off of
    # any size stays legal, mirroring the larder's deficit-completing
    # waiver).
    completes_hunt_readiness = effective_gain == headroom and hunt_entry_permitted(ctx)
    if completes_hunt_readiness:
        return False
    floor = max(_MIN_FUEL_SIP_GAIN, _FUEL_GAIN_PER_WALK_TILE * walk_tiles)
    return effective_gain < floor


def _first_walkworthy_fuel(
    ctx: DecideCtx,
) -> tuple[ContainerStateDict, BotCommand] | None:
    """Return the best fuel candidate that is worth its walk.

    Iterates the ranked candidate list instead of vetoing only the
    single best -- flag s9-2/3 (2026-07-30): the best-scored container
    (volume 1183, 13 tiles) failed the worth-the-walk rate while a
    762-volume container sat 3 tiles away, and the single-candidate
    veto sent the cascade into an in-viewport larder teleport (map
    open + displaced landing + spent radar) for ground a 3-tile walk
    served. The worth-the-walk rate stays an efficiency rule for a
    healthy tank; at or below the fuel-low break any reachable fuel
    is taken (run bot-20260728-090813: exited out_of_fuel at 98 with
    a pickable 39-fuel container two tiles away).

    Args:
        ctx: Decision context.

    Returns:
        ``(container, command)`` for the best walk-worthy executable
        fuel target, or ``None`` when none qualifies.
    """
    fuel_critical = ctx.fuel <= ctx.config["fuel_low_threshold"]
    cap = fuel_capacity(ctx.self_state["rank"])
    for container in find_fuel_candidates(
        ctx.filtered,
        ctx.self_state,
        ctx.terrain,
        minimum_volume=1,
    ):
        if not fuel_critical and pickup_not_worth_walk(ctx, container):
            walk_tiles = abs(container["x"] - ctx.self_state["x"]) + abs(
                container["y"] - ctx.self_state["y"]
            )
            emit_ai(
                "skip fuel at (%d,%d) vol=%d: clamped gain %d not worth "
                "%d-tile walk (fuel=%d cap=%d)",
                container["x"],
                container["y"],
                container["volume"],
                min(container["volume"], cap - ctx.fuel),
                walk_tiles,
                ctx.fuel,
                cap,
            )
            continue
        command = walk_or_teleport(ctx, container["x"], container["y"], pickup_kind="fuel")
        if command is None:
            continue
        return (container, command)
    return None


def select_and_pickup_fuel(
    ctx: DecideCtx,
    base_state: AIStateDict,
) -> TickDecisionDict | None:
    """Pick up the best walk-worthy viewport fuel, skipped at capacity."""
    if ctx.fuel >= fuel_capacity(ctx.self_state["rank"]):
        return None
    selection = _first_walkworthy_fuel(ctx)
    if selection is None:
        return None
    container, command = selection
    target_x = container["x"]
    target_y = container["y"]
    emit_ai(
        "collect fuel at (%d,%d) vol=%d (fuel=%d)",
        target_x,
        target_y,
        container["volume"],
        ctx.fuel,
    )
    return make_decision(
        command,
        "COLLECT",
        COLLECT_SCORE,
        target_x,
        target_y,
        "fuel_collect",
        set_resource_target(base_state, "fuel", target_x, target_y),
        ctx.equip,
        reason_context={"volume": container["volume"]},
    )


__all__ = [
    "mine_clearance_decision",
    "pickup_not_worth_walk",
    "select_and_pickup_equipment",
    "select_and_pickup_fuel",
    "select_equipment_target",
]
