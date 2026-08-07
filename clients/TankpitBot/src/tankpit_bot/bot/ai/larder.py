"""Larder fuel selection: score remembered fuel stock for harvest hops.

Implements the [[larder-plan]] fuel scorer: per COLLECT tick with a
fuel deficit, every believed fuel container competes on
``min(volume, deficit) / teleport_cost`` and the argmax wins -- a 900
at 25 tiles beats a 300 at 10 when the tank is down 700, and loses
when it is down 200. The selection is re-run every tick so the plan
never goes stale (user ruling: no fixed-container errands).

Landing reuses ``find_attainable_landing_tile``: the container tile
itself when landing-legal and mine-free, else a cardinal shore
neighbor -- both are inside the server's fuel auto-pick reach (ON or
cardinally adjacent, [[fuel-system]]), so the hop needs no pickup
command at all. Attainability, not mere legality: a known mine on the
landing displaces the teleport outside auto-pick reach every time
([[mine-mechanics]]; session bot-20260805-173034).
"""

from __future__ import annotations

from collections.abc import Callable

from typing_extensions import TypedDict

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.ferry_landing import find_ferry_boarding_tile
from tankpit_bot.bot.ai.mode_gates import hunt_entry_permitted
from tankpit_bot.bot.ai.reachability import find_attainable_landing_tile
from tankpit_bot.physics.capacity import fuel_capacity
from tankpit_bot.physics.costs import teleport_cost
from tankpit_bot.state.types import ContainerStateDict
from tankpit_bot.state.viewport_geometry import viewport_visible_bounds

_WALK_DOMINANT_RANGE = 2
"""Manhattan distance at or below which walking beats a larder hop.

A 2-tile walk costs 2 ticks and zero fuel; the teleport costs the same
2 ticks (map open + jump) PLUS fuel and a map churn. Containers this
close belong to the walk-pickup step's economics — if the walk rule
refuses them, they are not worth ANY travel. Session-3 receipts
([[flag-triage-20260729]] F13, flags s3-1/3/4/9): repeated 2-tile
larder teleports the user read as chaos.
"""

_LARDER_MIN_GAIN = 100
"""Deliverable fuel below which a larder hop is a dreg, not a plan.

Flag s3-9's chain teleported to a 35-volume remnant (net ~23 after
cost) while a 355-volume container sat one real hop away —
``gain/cost`` structurally favors close dregs because the denominator
shrinks faster than the numerator. One exception: a gain that
COMPLETES the deficit while the inventory is already hunt-ready (the
F1 top-off microscope: refusing the last 17 points forces a wasteful
dot hop). The waiver is scoped to hunt-ready stock (flag s8-1): at
radars=0 it spent a map open + teleport on a 24-fuel top-off that
unlocked nothing.
No desperation exemption is needed HERE: since the F16 net-of-gain
reserve, a payable hop onto a RICH container is allowed at any fuel
(that refuel is exactly what survives lethal pressure), while a
desperate tank refusing a sub-floor dreg is served by the walk-pickup
economics and ``collect_mode._desperation_fuel_hop`` — the 2026-06-23
"fuel is fuel" lesson lives there.
"""


class FuelLarderSelectionDict(TypedDict):
    """Outcome of one fuel-larder scoring pass.

    Attributes:
        container: Winning container, or ``None`` when every candidate
            was filtered out.
        landing_x: Teleport landing X for the winner (0 when none).
        landing_y: Teleport landing Y for the winner (0 when none).
        cost: Teleport cost to the winning landing (0 when none).
        candidates: Believed fuel containers that entered scoring.
        too_close: Candidates inside the walk-dominant range
            (:data:`_WALK_DOMINANT_RANGE`) -- travel there is the walk
            step's business, never a teleport's.
        no_landing: Candidates with no legal landing tile.
        reserve_blocked: Candidates still below the fuel reserve
            AFTER their own pickup (net-of-gain gate, F16).
        unprofitable: Candidates whose clamped gain does not exceed
            the hop cost.
        dreg: Candidates profitable on ratio but below the
            :data:`_LARDER_MIN_GAIN` floor (and neither
            deficit-completing nor desperation-exempt).
        ferry_served: Candidates whose landing is a ferry boarding
            tile ([[flag-triage-20260729]] F5) rather than the
            container's own ground.
    """

    container: ContainerStateDict | None
    landing_x: int
    landing_y: int
    cost: int
    candidates: int
    too_close: int
    no_landing: int
    reserve_blocked: int
    unprofitable: int
    dreg: int
    ferry_served: int


def _live_fuel_beliefs(
    ctx: DecideCtx,
    is_blacklisted: Callable[[int, int], bool],
) -> list[ContainerStateDict]:
    """Return believed fuel containers eligible to enter larder scoring.

    Filters to fuel containers with positive volume, no failed
    pickups, and no session blacklist mark — the pre-candidate
    universe; every gate after this point is tallied per candidate.

    Args:
        ctx: Decision context.
        is_blacklisted: Session blacklist predicate for container tiles.

    Returns:
        Containers that count as larder candidates.
    """
    return [
        container
        for container in ctx.world["containers"].values()
        if container["is_fuel"]
        and container["volume"] > 0
        and container["failed_pickups"] == 0
        and not is_blacklisted(container["x"], container["y"])
    ]


def _is_walk_territory(
    ctx: DecideCtx,
    container: ContainerStateDict,
    sx: int,
    sy: int,
) -> bool:
    """Return True when the container belongs to the walk economics.

    Movement law (user 2026-07-30, flag s9-2/3): same-viewport
    destinations are WALKED -- an in-viewport larder teleport pays a
    map open + jump + displacement risk for ground a few walk ticks
    serve. Off-viewport but within :data:`_WALK_DOMINANT_RANGE` is
    equally walk territory (a tank at its viewport edge two tiles
    from off-frame stock). The larder is cross-viewport machinery.

    A container FLOATING ON WATER is never walk territory from land
    (F5 completion, 2026-08-01): the walk step's surface routing
    cannot reach it at any distance, so ceding it to the walk
    economics stranded in-viewport water fuel with nobody serving it
    -- the larder keeps it and its landing resolution falls through
    to the ferry boarding tile.

    Args:
        ctx: Decision context (viewport bounds).
        container: Candidate fuel container.
        sx: Self X.
        sy: Self Y.

    Returns:
        True when the walk step owns this container.
    """
    terrain = ctx.terrain
    if terrain is not None and not terrain.is_passable(container["x"], container["y"]):
        return False
    left, top, right, bottom = viewport_visible_bounds(ctx.world["viewport"])
    if left <= container["x"] <= right and top <= container["y"] <= bottom:
        return True
    return abs(container["x"] - sx) + abs(container["y"] - sy) <= _WALK_DOMINANT_RANGE


def select_fuel_larder_hop(
    ctx: DecideCtx,
    *,
    is_blacklisted: Callable[[int, int], bool],
) -> FuelLarderSelectionDict:
    """Score every believed fuel container and return the best harvest hop.

    Hard gates: a legal landing tile, the fuel reserve net of the
    landing pickup (F16: the transaction must clear the reserve, not
    the transit), net profitability (``min(volume, deficit)`` must exceed
    the hop cost), the walk-dominant range
    (:data:`_WALK_DOMINANT_RANGE` -- near containers are the walk
    step's business), and the dreg floor (:data:`_LARDER_MIN_GAIN`,
    waived for deficit-completing gains and at desperation fuel).
    Belief freshness is NOT gated: container belief expires only on
    hard evidence ([[larder-plan]] two-clocks ruling).

    Args:
        ctx: Decision context. ``ctx.terrain`` must not be ``None``.
        is_blacklisted: Session blacklist predicate for container tiles.

    Returns:
        Selection outcome with the winner (or ``None``) and the
        per-filter decline tallies for the hop-declined diagnostic.
    """
    terrain = ctx.terrain
    assert terrain is not None  # caller guarantees this
    deficit = fuel_capacity(ctx.self_state["rank"]) - ctx.fuel
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    reserve = ctx.config["fuel_low_threshold"]
    candidates = 0
    too_close = 0
    no_landing = 0
    reserve_blocked = 0
    unprofitable = 0
    dreg = 0
    ferry_served = 0
    best: ContainerStateDict | None = None
    best_landing_x = 0
    best_landing_y = 0
    best_cost = 0
    best_score = 0.0
    for container in _live_fuel_beliefs(ctx, is_blacklisted):
        candidates += 1
        if _is_walk_territory(ctx, container, sx, sy):
            too_close += 1
            continue
        landing = find_attainable_landing_tile(terrain, container["x"], container["y"])
        if landing is None:
            # Water-locked container: a believed ferry near it is the
            # boarding-tile landing — teleport to the ferry, then the
            # held lock rides to the pickup ([[flag-triage-20260729]]
            # F5; riding pickup law in [[ferry-mechanics]]).
            landing = find_ferry_boarding_tile(
                ctx.world,
                terrain,
                container["x"],
                container["y"],
            )
            if landing is not None:
                ferry_served += 1
        if landing is None:
            no_landing += 1
            continue
        landing_x, landing_y = landing
        cost = teleport_cost(sx, sy, landing_x, landing_y)
        gain = min(container["volume"], deficit)
        # Reserve gates the TRANSACTION, not the transit ([[flag-triage-20260729]]
        # F16, the Artax death): a harvest hop's purpose is landing on
        # fuel, so the fuel that must clear the reserve is
        # post-teleport-plus-pickup. The old post-teleport-only gate
        # created the 200-250 dead zone where a tank under fire could
        # not buy the refuel that would have saved it; a failed pickup
        # at the landing is the blacklist's business. The hard floor
        # stays absolute: fuel is health, so the teleport itself must
        # be payable with life left over (arriving at zero is
        # deactivation, and no landing pickup outruns being dead).
        if cost >= ctx.fuel or ctx.fuel - cost + gain < reserve:
            reserve_blocked += 1
            continue
        if gain <= cost:
            unprofitable += 1
            continue
        # The deficit-completing waiver is scoped to hunt-ready
        # inventory (flag s8-1, 2026-07-30): topping the last points
        # only matters when fuel is the FINAL hunt requirement. At
        # radars=0 the waiver spent a map open + teleport on a
        # 24-fuel top-off that unlocked nothing -- the walk step had
        # just refused the same area's fuel as "clamped gain 24 not
        # worth 5-tile walk".
        completes_hunt_readiness = gain == deficit and hunt_entry_permitted(ctx)
        if gain < _LARDER_MIN_GAIN and not completes_hunt_readiness:
            dreg += 1
            continue
        score = gain / max(cost, 1)
        if best is None or score > best_score:
            best = container
            best_landing_x = landing_x
            best_landing_y = landing_y
            best_cost = cost
            best_score = score
    return FuelLarderSelectionDict(
        container=best,
        landing_x=best_landing_x,
        landing_y=best_landing_y,
        cost=best_cost,
        candidates=candidates,
        too_close=too_close,
        no_landing=no_landing,
        reserve_blocked=reserve_blocked,
        unprofitable=unprofitable,
        dreg=dreg,
        ferry_served=ferry_served,
    )


__all__ = [
    "FuelLarderSelectionDict",
    "select_fuel_larder_hop",
]
