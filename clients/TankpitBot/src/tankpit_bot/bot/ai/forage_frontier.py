"""Staleness-seeking forage: hop to the nearest unlooked-at block.

Replaces the corpus-mined equipment atlas (operator green light,
2026-08-28). The atlas's ground truth failed: across the last 50
archived runs 7,635 equipment pickups distributed FLAT over map x
(961/878/923/994/1017/992/955/915 per 32-wide band), so equipment
spawn is ~uniform and the atlas's "hotspots" were dwell bias -- tiles
many runs LOOKED AT (the spawn center, the stranded-session west
corner), self-reinforcing because the atlas then sent every session
back to stare at them. The live confirmation: seven west-edge atlas
hops yielded zero pickups while the first mid-map hop yielded three
in six seconds.

Under uniform spawn the optimal restock policy is staleness-seeking:
standing stock accumulates wherever nobody has harvested lately, so
the best ground is the NEAREST block with no live radar coverage.
The frontier hop picks it, travels by the walk-first rule (a leg in
window walks; beyond it teleports), and latches the chosen goal in
the AI state so the teleport's map-open beat cannot be re-planned
away (the atlas hop stamped its own target visited at PLAN time and
swapped targets across the open beat -- run bot-20260828-192801
19:31:00 opened for a cost-40 hop and threw a cost-227 one).
"""

from __future__ import annotations

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.collect_common import COLLECT_SCORE
from tankpit_bot.bot.ai.context import DecideCtx, make_decision
from tankpit_bot.bot.ai.movement import plan_viewport_walk
from tankpit_bot.bot.ai.tactics import combat_radar_min
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import make_teleport_command
from tankpit_bot.physics.capacity import inventory_capacity
from tankpit_bot.physics.costs import teleport_cost
from tankpit_bot.runtime_logging import emit_ai
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.scan_coverage import FORAGE_COVERAGE_TTL_MS
from tankpit_bot.state.viewport_geometry import viewport_visible_bounds

BLOCK_TILES = 16
"""Frontier granularity: one viewport (16x16 tiles) per block.

A landing reveals exactly one viewport, so coverage is earned and
spent in viewport-sized units; a finer grid would re-target ground
the last landing already showed.
"""

_BLOCK_GRID = 16
"""Blocks per map axis (256-tile maps, [[map-data-decode]])."""

FRONTIER_VISIT_TTL_MS = 180_000
"""How long an arrived-at (or seen-empty) block stays off the circuit.

Containers sit unclaimed until collected, so an empty block means
someone took its stock; the TTL lets the frontier come back after
the field has had time to repopulate. Same respawn rationale as
:data:`~tankpit_bot.state.scan_coverage.FORAGE_COVERAGE_TTL_MS`.
"""

_ARRIVE_TILES = 2
"""Standing within this of the goal counts as having looked at it."""


def _equipment_deficient(ctx: DecideCtx) -> bool:
    """Return True when the restock bars are not yet met.

    Mirrors the HUNT-entry readiness the restock exists to reach:
    duals and homings at the rank cap, extra radars at the hunt bar.

    Args:
        ctx: Decision context.

    Returns:
        True when any bar is unmet.
    """
    rank = ctx.self_state["rank"]
    cap = inventory_capacity(rank)
    return (
        ctx.inventory["dual_shots"]["count"] < cap
        or ctx.inventory["homing_shots"]["count"] < cap
        or ctx.inventory["extra_radars"]["count"] < combat_radar_min(rank)
    )


def _prune_visits(ws: WorldService, now_ms: int) -> None:
    """Drop expired visit tombstones.

    Args:
        ws: The session's world service.
        now_ms: Current timestamp.
    """
    ws.forage_visited = {
        key: ts for key, ts in ws.forage_visited.items() if now_ms - ts < FRONTIER_VISIT_TTL_MS
    }


def _covered_blocks(ctx: DecideCtx) -> set[tuple[int, int]]:
    """Blocks holding at least one live radar-coverage stamp.

    Args:
        ctx: Decision context.

    Returns:
        ``(bx, by)`` block indices with unexpired coverage.
    """
    covered: set[tuple[int, int]] = set()
    for key, ts in ctx.world["scanned_tiles"].items():
        if ctx.timestamp_ms - ts >= FORAGE_COVERAGE_TTL_MS:
            continue
        x_str, y_str = key.split(",")
        covered.add((int(x_str) // BLOCK_TILES, int(y_str) // BLOCK_TILES))
    return covered


def _stale_block_centers(ctx: DecideCtx, terrain: TerrainMapProtocol) -> list[tuple[int, int]]:
    """Rank reachable unlooked-at block centers nearest-first.

    A block qualifies when it holds no live coverage stamp, its
    center is passable, un-tombstoned, and not hostile ground, and
    its center is either inside the current window (walkable — the
    walk dispatcher only walks in-window legs) or teleport-affordable
    above the fuel floor.

    Args:
        ctx: Decision context.
        terrain: Loaded terrain map (the caller checked presence).

    Returns:
        Candidate ``(x, y)`` centers, nearest first.
    """
    covered = _covered_blocks(ctx)
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    budget = ctx.fuel - ctx.config["fuel_low_threshold"]
    hostile = ctx.ws.hostile_landing_keys(ctx.timestamp_ms)
    left, top, right, bottom = viewport_visible_bounds(ctx.world["viewport"])
    half = BLOCK_TILES // 2
    ranked: list[tuple[int, int, int]] = []
    for by in range(_BLOCK_GRID):
        for bx in range(_BLOCK_GRID):
            if (bx, by) in covered:
                continue
            x = bx * BLOCK_TILES + half
            y = by * BLOCK_TILES + half
            key = f"{x},{y}"
            if key in ctx.ws.forage_visited or key in hostile:
                continue
            if not terrain.is_passable(x, y):
                continue
            distance = max(abs(x - sx), abs(y - sy))
            if distance == 0:
                continue
            in_window = left <= x <= right and top <= y <= bottom
            if not in_window and teleport_cost(sx, sy, x, y) > budget:
                continue
            ranked.append((distance, x, y))
    ranked.sort()
    return [(x, y) for _, x, y in ranked]


def plan_forage_frontier_hop(ctx: DecideCtx, base_state: AIStateDict) -> TickDecisionDict | None:
    """Move toward the nearest unlooked-at block to restock equipment.

    Runs in the COLLECT cascade after every known-stock branch
    declined (nothing believed collectible in reach) and BEFORE the
    quad sweep: traveling to ground nobody has looked at lately
    beats buying blind reveals where we just harvested. The landing
    viewport shows whatever sits there (equipment containers are
    plain viewport entities, no radar needed).

    The chosen goal is LATCHED in the AI state and served first on
    following ticks until arrival, so the beat its teleport spends
    opening the map cannot be re-planned into a different hop.
    Arrival (within :data:`_ARRIVE_TILES`) tombstones the goal for
    :data:`FRONTIER_VISIT_TTL_MS` and the next call picks fresh.

    Args:
        ctx: Decision context.
        base_state: Base AI state to rewrite for the produced command.

    Returns:
        A walk or teleport decision toward the goal, or None when
        equipment is not needed, terrain is not yet loaded, or no
        reachable stale block remains.
    """
    if not _equipment_deficient(ctx):
        return None
    if ctx.fuel <= ctx.config["fuel_low_threshold"]:
        # Restocking is a healthy-fuel activity: critical-fuel ticks
        # belong to the survival ladder (desperation hop, walk for
        # fuel), which runs in the exhausted outcome BELOW this rung.
        # Without this gate the frontier walked a fuel-0 marooned
        # tank toward equipment blocks instead of fuel.
        return None
    terrain = ctx.terrain
    if terrain is None:
        return None
    _prune_visits(ctx.ws, ctx.timestamp_ms)
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    goal_x = base_state["forage_goal_x"]
    goal_y = base_state["forage_goal_y"]
    goal_live = goal_x >= 0 and goal_y >= 0
    if goal_live and max(abs(sx - goal_x), abs(sy - goal_y)) <= _ARRIVE_TILES:
        ctx.ws.forage_visited[f"{goal_x},{goal_y}"] = ctx.timestamp_ms
        goal_live = False
    candidates = _stale_block_centers(ctx, terrain)
    if goal_live:
        candidates = [(goal_x, goal_y)] + [c for c in candidates if c != (goal_x, goal_y)]
    left, top, right, bottom = viewport_visible_bounds(ctx.world["viewport"])
    for target_x, target_y in candidates:
        # In-window ground is walked for free ([[walk-mechanics]]);
        # the window itself never moves in COLLECT (autoscroll OFF),
        # so anything beyond it travels by teleport.
        if left <= target_x <= right and top <= target_y <= bottom:
            command = plan_viewport_walk(ctx, target_x, target_y)
        else:
            command = make_teleport_command(target_x, target_y)
        if command is None:
            continue
        emit_ai(
            "forage frontier: %s to unlooked block (%d,%d)",
            command["cmd_type"],
            target_x,
            target_y,
        )
        return make_decision(
            command,
            "COLLECT",
            COLLECT_SCORE,
            target_x,
            target_y,
            "forage_frontier_hop",
            # The held resource plan is NOT cleared: a plan whose
            # teleport is merely unaffordable this tick survives a
            # frontier leg (pinned by the locked-target cascade law);
            # the lock machinery re-validates it itself.
            AIStateDict(
                **{
                    **base_state,
                    "forage_goal_x": target_x,
                    "forage_goal_y": target_y,
                }
            ),
            ctx.equip,
            reason_context={"stale_blocks": len(candidates)},
        )
    return None


__all__ = [
    "BLOCK_TILES",
    "FRONTIER_VISIT_TTL_MS",
    "plan_forage_frontier_hop",
]
