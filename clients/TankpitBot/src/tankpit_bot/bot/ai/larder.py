"""Larder fuel selection: score remembered fuel stock for harvest hops.

Implements the [[larder-plan]] fuel scorer: per COLLECT tick with a
fuel deficit, every believed fuel container competes on
``min(volume, deficit) / teleport_cost`` and the argmax wins -- a 900
at 25 tiles beats a 300 at 10 when the tank is down 700, and loses
when it is down 200. The selection is re-run every tick so the plan
never goes stale (user ruling: no fixed-container errands).

Landing reuses ``find_teleport_landing_tile``: the container tile
itself when passable, else a cardinal shore neighbor -- both are
inside the server's fuel auto-pick reach (ON or cardinally adjacent,
[[fuel-system]]), so the hop needs no pickup command at all.
"""

from __future__ import annotations

from collections.abc import Callable

from typing_extensions import TypedDict

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.equipment_search import find_teleport_landing_tile
from tankpit_bot.physics.capacity import fuel_capacity
from tankpit_bot.physics.costs import teleport_cost
from tankpit_bot.state.types import ContainerStateDict


class FuelLarderSelectionDict(TypedDict):
    """Outcome of one fuel-larder scoring pass.

    Attributes:
        container: Winning container, or ``None`` when every candidate
            was filtered out.
        landing_x: Teleport landing X for the winner (0 when none).
        landing_y: Teleport landing Y for the winner (0 when none).
        cost: Teleport cost to the winning landing (0 when none).
        candidates: Believed fuel containers that entered scoring.
        too_close: Candidates within auto-pick reach of the tank
            already -- a hop to them would be a zero-distance teleport.
        no_landing: Candidates with no legal landing tile.
        reserve_blocked: Candidates whose hop would break the fuel
            reserve.
        unprofitable: Candidates whose clamped gain does not exceed
            the hop cost.
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


def select_fuel_larder_hop(
    ctx: DecideCtx,
    *,
    is_blacklisted: Callable[[int, int], bool],
) -> FuelLarderSelectionDict:
    """Score every believed fuel container and return the best harvest hop.

    Hard gates are physics only: a legal landing tile, hop cost inside
    the fuel reserve, and net profitability (``min(volume, deficit)``
    must exceed the hop cost -- the plan's "empty or unprofitable"
    boundary that hands the tick to discovery). Belief freshness is
    NOT gated: container belief expires only on hard evidence
    ([[larder-plan]] two-clocks ruling).

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
    best: ContainerStateDict | None = None
    best_landing_x = 0
    best_landing_y = 0
    best_cost = 0
    best_score = 0.0
    for container in ctx.world["containers"].values():
        if not container["is_fuel"] or container["volume"] <= 0:
            continue
        if container["failed_pickups"] > 0:
            continue
        if is_blacklisted(container["x"], container["y"]):
            continue
        candidates += 1
        if max(abs(container["x"] - sx), abs(container["y"] - sy)) <= 1:
            too_close += 1
            continue
        landing = find_teleport_landing_tile(terrain, container["x"], container["y"])
        if landing is None:
            no_landing += 1
            continue
        landing_x, landing_y = landing
        cost = teleport_cost(sx, sy, landing_x, landing_y)
        if ctx.fuel - cost < reserve:
            reserve_blocked += 1
            continue
        gain = min(container["volume"], deficit)
        if gain <= cost:
            unprofitable += 1
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
    )


__all__ = [
    "FuelLarderSelectionDict",
    "select_fuel_larder_hop",
]
