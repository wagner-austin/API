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
The frontier hop picks it, teleports (in-window blocks are stamped
looked-at by sight -- 0x5A enumerates a window's containers
radar-free, so every goal is genuinely unseen ground), and latches
the chosen goal in the AI state so the teleport's map-open beat
cannot be re-planned away (the atlas hop stamped its own target visited at PLAN time and
swapped targets across the open beat -- run bot-20260828-192801
19:31:00 opened for a cost-40 hop and threw a cost-227 one).
"""

from __future__ import annotations

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.collect_common import COLLECT_SCORE
from tankpit_bot.bot.ai.context import DecideCtx, make_decision
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

_GOAL_ATTEMPT_CAP = 3
"""Travel dispatches a latched goal survives before it is tombstoned.

The displacement law can bounce every landing far from an unlandable
block center, so arrival never fires and the latch re-throws forever:
arterial (2026-08-28 20:52, run bot-20260828-205052) paid 14+
teleports at (120,104) with every landing 9-25 tiles out. Three
attempts prove the block unlandable; the tombstone sends the circuit
on and the TTL retries it later.
"""


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
    its teleport is affordable above the fuel floor. In-window
    centers never qualify: the caller stamps them looked-at by
    sight before ranking.

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
    claimed_blocks = {
        (gx // BLOCK_TILES, gy // BLOCK_TILES) for gx, gy in ctx.ws.fleet_forage_goals.values()
    }
    half = BLOCK_TILES // 2
    ranked: list[tuple[int, int, int]] = []
    for by in range(_BLOCK_GRID):
        for bx in range(_BLOCK_GRID):
            if (bx, by) in covered or (bx, by) in claimed_blocks:
                continue
            x = bx * BLOCK_TILES + half
            y = by * BLOCK_TILES + half
            key = f"{x},{y}"
            if key in ctx.ws.forage_visited or key in hostile:
                continue
            if not terrain.is_passable(x, y):
                continue
            # distance is never 0: the tank's own block center is
            # in-window and therefore sight-stamped before ranking.
            distance = max(abs(x - sx), abs(y - sy))
            if teleport_cost(sx, sy, x, y) > budget:
                continue
            ranked.append((distance, x, y))
    ranked.sort()
    return [(x, y) for _, x, y in ranked]


def _stamp_window_blocks_by_sight(ctx: DecideCtx) -> None:
    """Mark every block whose center the current window shows as looked-at.

    A window in view IS looked at: 0x5A enumerates its containers
    radar-free, so traveling to an in-window block center reveals
    nothing (operator flag, 2026-08-28 World watch: "walking to a
    spot, then walking back to another spot, not doing anything").
    Stamping by sight keeps every frontier goal genuinely unseen
    ground, and frontier travel is therefore teleport-only.

    Args:
        ctx: Decision context.
    """
    left, top, right, bottom = viewport_visible_bounds(ctx.world["viewport"])
    half = BLOCK_TILES // 2
    for by in range(top // BLOCK_TILES, bottom // BLOCK_TILES + 1):
        for bx in range(left // BLOCK_TILES, right // BLOCK_TILES + 1):
            center_x = bx * BLOCK_TILES + half
            center_y = by * BLOCK_TILES + half
            if left <= center_x <= right and top <= center_y <= bottom:
                ctx.ws.forage_visited[f"{center_x},{center_y}"] = ctx.timestamp_ms


def _resolve_standing_goal(ctx: DecideCtx, base_state: AIStateDict) -> tuple[int, int, bool]:
    """Release the latched goal on arrival or at the attempt cap.

    Args:
        ctx: Decision context.
        base_state: AI state carrying the goal latch.

    Returns:
        ``(goal_x, goal_y, live)`` -- the latched coordinates and
        whether the latch still holds after arrival and attempt-cap
        release (both releases tombstone the block).
    """
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    goal_x = base_state["forage_goal_x"]
    goal_y = base_state["forage_goal_y"]
    goal_live = goal_x >= 0 and goal_y >= 0
    if goal_live and max(abs(sx - goal_x), abs(sy - goal_y)) <= _ARRIVE_TILES:
        ctx.ws.forage_visited[f"{goal_x},{goal_y}"] = ctx.timestamp_ms
        goal_live = False
    if goal_live and base_state["forage_goal_attempts"] >= _GOAL_ATTEMPT_CAP:
        # Every landing bounced away from the center: the block is
        # unlandable, and re-throwing burns ~20 fuel per attempt.
        ctx.ws.forage_visited[f"{goal_x},{goal_y}"] = ctx.timestamp_ms
        goal_live = False
    return goal_x, goal_y, goal_live


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
    _stamp_window_blocks_by_sight(ctx)
    goal_x, goal_y, goal_live = _resolve_standing_goal(ctx, base_state)
    candidates = _stale_block_centers(ctx, terrain)
    if goal_live:
        candidates = [(goal_x, goal_y)] + [c for c in candidates if c != (goal_x, goal_y)]
    if not candidates:
        return None
    target_x, target_y = candidates[0]
    command = make_teleport_command(target_x, target_y)
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
                "forage_goal_attempts": (
                    base_state["forage_goal_attempts"] + 1
                    if goal_live and (target_x, target_y) == (goal_x, goal_y)
                    else 1
                ),
            }
        ),
        ctx.equip,
        reason_context={"stale_blocks": len(candidates)},
    )


__all__ = [
    "BLOCK_TILES",
    "FRONTIER_VISIT_TTL_MS",
    "plan_forage_frontier_hop",
]
