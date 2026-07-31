"""Resource-search hop: teleport to the nearest clean-viewport fuel dot.

Candidate destinations come from the 0x4C MapData fuel-dot atlas — the
map's yellow-pixel fuel positions (server-cached per session; ~40% of
dots still hold fuel when visited, and every wire-verified dot held
high-volume fuel). Hopping dot-to-dot replaces the old blind compass
hop: each landing is in fuel-rich ground and the landing auto-pickup
makes the hop partially self-funding (user contract 2026-07-03: "hop
to nearest yellow dot with a 100% clean viewport").

A candidate dot qualifies when its landing tile is passable, the
teleport is fuel-affordable, the landing viewport has no fresh scan
coverage, and the landing viewport is 100% walkable ground from the
static terrain map (the walk-only pickup contract makes rock/water
tiles uncollectable, so a dirty viewport wastes the hop). Qualifiers
are taken nearest-first. Without a terrain map the walkable check
degrades to 1.0 and selection is purely nearest-affordable-unscanned.

When the atlas is empty (no map open yet this session) the hop
dispatches ``map_open`` — the dots arrive with the 0x4C response —
guarded by ``map_open_cooldown_ms`` so a dotless map cannot loop.
When no dot qualifies, the function returns ``None`` and the caller
raises — the bot is genuinely stuck and no second hopping mechanism
papers over that.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.context import (
    DecideCtx,
    can_afford_teleport,
    clear_resource_target,
    make_decision,
    teleport_fuel_cost_to,
)
from tankpit_bot.bot.ai.mode_controller import hunt_entry_permitted
from tankpit_bot.bot.ai.types import AIStateDict, BehaviorMode, ReasonKind
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import make_map_open_command, make_teleport_command
from tankpit_bot.physics.capacity import fuel_capacity
from tankpit_bot.runtime_logging import emit_ai, emit_diagnostic
from tankpit_bot.state.scan_coverage import (
    HARVEST_MEMORY_TTL_MS,
    is_viewport_scanned_within,
    is_viewport_untouched,
)
from tankpit_bot.state.types import coord_key

_HUNT_BIAS_HALF_SCORE_TILES = 16
"""Distance at which the pre-hunt top-off bias halves a dot's score.

One viewport-width away from the intended prey costs a factor of two;
the flag-2 hop (26 tiles away from the target the very next tick's
acquisition teleported back to) would have been outscored ~2.6:1 by
an equal-value dot on the target's side.
"""


def _viewport_walkable_fraction(
    ctx: DecideCtx,
    left: int,
    top: int,
    width: int,
    height: int,
) -> float:
    """Return the fraction of viewport tiles that are walkable ground.

    Off-map tiles (viewport clipped at the field border) count as
    unwalkable -- the border is rock. Without a terrain map every
    candidate scores 1.0, so selection degrades to nearest-first.

    Args:
        ctx: Decision context.
        left: Viewport left X (inclusive).
        top: Viewport top Y (inclusive).
        width: Viewport width in tiles.
        height: Viewport height in tiles.

    Returns:
        Walkable tile count divided by the full viewport area.
    """
    terrain = ctx.terrain
    if terrain is None:
        return 1.0
    walkable = 0
    for y in range(max(0, top), min(255, top + height - 1) + 1):
        for x in range(max(0, left), min(255, left + width - 1) + 1):
            if terrain.is_passable(x, y):
                walkable += 1
    return walkable / (width * height)


def _landing_viewport_known_empty(
    ctx: DecideCtx,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> bool:
    """Return True when belief says this landing viewport is picked clean.

    True exactly when the bounds hold at least one believed container,
    every one of them is drained (volume <= 0), and the newest belief
    is younger than :data:`HARVEST_MEMORY_TTL_MS` — the harvest-memory
    veto of [[flag-triage-20260729]] F2. Ground with no beliefs at all
    is unknown, not empty; a single positive-volume belief means the
    hop has real value; beliefs older than the window may have
    respawned and the ground goes back to explorable.

    Args:
        ctx: Decision context.
        left: Landing viewport left X (inclusive).
        top: Landing viewport top Y (inclusive).
        right: Landing viewport right X (inclusive).
        bottom: Landing viewport bottom Y (inclusive).

    Returns:
        True when the viewport is a known-empty re-hop candidate.
    """
    newest_ms = -1
    seen = False
    for container in ctx.world["containers"].values():
        if not (left <= container["x"] <= right and top <= container["y"] <= bottom):
            continue
        if container["volume"] > 0:
            return False
        seen = True
        newest_ms = max(newest_ms, container["timestamp_ms"])
    return seen and ctx.timestamp_ms - newest_ms <= HARVEST_MEMORY_TTL_MS


def _landing_viewport_barren(
    ctx: DecideCtx,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> bool:
    """Return True when a recent full sweep found nothing worth landing on.

    The barren-memory veto — the other half of
    [[flag-triage-20260729]] F2. Ground the radar fully swept within
    :data:`HARVEST_MEMORY_TTL_MS` that revealed NO containers leaves
    no belief entries at all, so the known-empty veto cannot see it;
    once the 180 s forage coverage aged out it read fully clean and
    got re-hopped for a guaranteed zero-delta scan (the user's "zero
    deltas indicating they were scanned by us recently"). A single
    positive-volume belief inside the bounds means the ground has
    real value and the hop stands.

    Args:
        ctx: Decision context.
        left: Landing viewport left X (inclusive).
        top: Landing viewport top Y (inclusive).
        right: Landing viewport right X (inclusive).
        bottom: Landing viewport bottom Y (inclusive).

    Returns:
        True when the viewport is a known-barren re-hop candidate.
    """
    for container in ctx.world["containers"].values():
        if (
            left <= container["x"] <= right
            and top <= container["y"] <= bottom
            and container["volume"] > 0
        ):
            return False
    return is_viewport_scanned_within(
        ctx.world["scanned_tiles"],
        left,
        top,
        right,
        bottom,
        ctx.timestamp_ms,
        ttl_ms=HARVEST_MEMORY_TTL_MS,
    )


def _nearest_alive_enemy(ctx: DecideCtx) -> tuple[int, int] | None:
    """Return the nearest alive enemy's position, or None when none is known.

    Used by the pre-hunt top-off bias: when stocks are hunt-ready and
    only fuel is short, the final dot hop should top off TOWARD the
    prey instead of wherever dots are densest ([[flag-triage-20260729]]
    F1 — the flag-2 hop went 26 tiles NE and the very next acquisition
    teleported 30 tiles SW straight back).

    Args:
        ctx: Decision context.

    Returns:
        ``(x, y)`` of the closest alive, position-synced enemy tank.
    """
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    best: tuple[int, int] | None = None
    best_dist = 0
    for tank in ctx.world["tanks"].values():
        if tank["is_self"] or tank["team"] == ctx.self_state["team"]:
            continue
        if tank["liveness"] != "alive" or (tank["x"] == 0 and tank["y"] == 0):
            continue
        dist = abs(tank["x"] - sx) + abs(tank["y"] - sy)
        if best is None or dist < best_dist:
            best = (tank["x"], tank["y"])
            best_dist = dist
    return best


def _dot_hop_rejection(
    ctx: DecideCtx,
    target_x: int,
    target_y: int,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> str | None:
    """Classify why a dot candidate fails the hop gates.

    The gate order matches the historical loop: own tile, impassable
    landing, unaffordable teleport, live scan coverage, the
    known-empty harvest veto, and the barren-scan veto. The returned
    key doubles as the ``hop_declined`` tally field name so every
    decline states its arithmetic.

    Args:
        ctx: Decision context.
        target_x: Candidate dot X.
        target_y: Candidate dot Y.
        left: Landing viewport left X (inclusive).
        top: Landing viewport top Y (inclusive).
        right: Landing viewport right X (inclusive).
        bottom: Landing viewport bottom Y (inclusive).

    Returns:
        Tally key of the failed gate, or ``None`` when the dot
        qualifies for scoring.
    """
    if (target_x, target_y) == (ctx.self_state["x"], ctx.self_state["y"]):
        return "own_tile"
    if ctx.terrain is not None and not ctx.terrain.is_passable(target_x, target_y):
        return "impassable"
    if not can_afford_teleport(ctx, target_x, target_y):
        return "unaffordable"
    if not is_viewport_untouched(
        ctx.world["scanned_tiles"],
        left,
        top,
        right,
        bottom,
        ctx.timestamp_ms,
    ):
        return "already_scanned"
    if _landing_viewport_known_empty(ctx, left, top, right, bottom):
        return "known_empty"
    if _landing_viewport_barren(ctx, left, top, right, bottom):
        return "barren_scanned"
    return None


def _pick_fresh_dot_hop(ctx: DecideCtx) -> tuple[int, int] | None:
    """Return the best-value fuel dot to hop to, or None when none qualify.

    Hard gates (physics + memory): the dot is not the bot's own tile,
    its landing tile is passable, the teleport is fuel-affordable, the
    landing viewport is CLEAN — zero overlap with live scan coverage
    (user ruling, verbatim, 2026-07-26: "when i say it should collect
    on clean viewports, that means zero overlap"; the 2026-07-18
    implementation inverted this to "any unscanned tile counts as
    fresh" and run bot-20260725-235637 spent a third of every radar
    re-scanning old ground) — and the landing viewport is not KNOWN
    EMPTY from container beliefs (the harvest-memory veto, 2026-07-30:
    scan coverage ages out on the 180 s forage TTL so re-scans stay
    possible, but ground whose believed containers are all drained
    within :data:`HARVEST_MEMORY_TTL_MS` yields nothing and is
    skipped — [[flag-triage-20260729]] F2, 63% zero-yield hops).

    Pre-hunt top-off bias ([[flag-triage-20260729]] F1): when stocks
    are already hunt-ready (only fuel short), each dot's score is
    scaled by proximity to the nearest alive enemy so the final
    top-off lands on the prey's side of the map instead of wherever
    dots are densest.

    Loot-run bias (2026-07-30): the same enemy-proximity scaling also
    applies when the tank is equipment-hungry with fuel AT CAP --
    session 7 wandered barren fuel-dot viewports at fuel 1100 with
    zero radars, ranking hops by fuel it could not absorb. Equipment
    comes from kills, so a capped tank restocks fastest near the
    fights. Below cap the dots still fund the trip and the ranking
    stays unbiased.

    Qualifying dots are RANKED, not filtered, by hop value (user
    contract 2026-07-18: "the rule was to prioritize viewports with
    more dots, more walkable area. but not a 100% rule"):

        score = dots_in_landing_viewport * walkable_fraction / cost

    -- expected pickup value, scaled by how much of the landing
    viewport is actually reachable, per fuel spent. Closer dots win
    through the cost denominator. This replaces the 2026-07-03
    ``walkable_fraction == 1.0`` hard filter, which mis-read "100%
    clean viewport" as "zero terrain tiles" and rejected 428 of 622
    dots in the 2026-07-18 diagnostic run, starving the hop cascade
    into ``no_productive_collect`` exits.

    Args:
        ctx: Decision context.

    Returns:
        ``(target_x, target_y)`` of the highest-value qualifying dot,
        or ``None`` when none pass the hard gates.
    """
    viewport = ctx.world["viewport"]
    half_w = viewport["width"] // 2
    half_h = viewport["height"] // 2

    def _dots_in_viewport(left: int, top: int) -> int:
        right = left + viewport["width"] - 1
        bottom = top + viewport["height"] - 1
        return sum(
            1
            for dot_x, dot_y in ctx.map_fuel_dots
            if left <= dot_x <= right and top <= dot_y <= bottom
        )

    # Two bias regimes share the same mechanism and the same anchor
    # (the nearest alive enemy), for opposite deficits:
    # * hunt-ready, only fuel short -- the pre-hunt top-off bias
    #   ([[flag-triage-20260729]] F1): land the final fuel stop on the
    #   prey's side of the map.
    # * equipment-hungry with fuel AT CAP -- the loot-run bias
    #   (2026-07-30, session 7: radar-broke at fuel 1100, every hop
    #   ranked by fuel dots the tank could not even absorb, wandering
    #   barren ground while 8 tracked equipment drops sat water-locked).
    #   Fuel dots at cap carry zero pickup value; their only worth is
    #   WHERE they are, and equipment comes from kills -- so the search
    #   drifts toward the fights, which is also where the next hunt
    #   starts. Below cap the dots still refuel the restock trip, so
    #   the ranking stays unbiased.
    if hunt_entry_permitted(ctx) or ctx.fuel >= fuel_capacity(ctx.self_state["rank"]):
        hunt_bias = _nearest_alive_enemy(ctx)
    else:
        hunt_bias = None
    tallies = {
        "own_tile": 0,
        "impassable": 0,
        "unaffordable": 0,
        "already_scanned": 0,
        "known_empty": 0,
        "barren_scanned": 0,
    }
    best_score = -1.0
    best_cost = 0
    best_dot: tuple[int, int] | None = None
    for target_x, target_y in sorted(ctx.map_fuel_dots):
        landing_left = target_x - half_w
        landing_top = target_y - half_h
        rejection = _dot_hop_rejection(
            ctx,
            target_x,
            target_y,
            landing_left,
            landing_top,
            landing_left + viewport["width"] - 1,
            landing_top + viewport["height"] - 1,
        )
        if rejection is not None:
            tallies[rejection] += 1
            continue
        cost = teleport_fuel_cost_to(ctx, target_x, target_y)
        walkable = _viewport_walkable_fraction(
            ctx,
            landing_left,
            landing_top,
            viewport["width"],
            viewport["height"],
        )
        score = _dots_in_viewport(landing_left, landing_top) * walkable / max(cost, 1)
        if hunt_bias is not None:
            bias_dist = abs(target_x - hunt_bias[0]) + abs(target_y - hunt_bias[1])
            score *= _HUNT_BIAS_HALF_SCORE_TILES / (_HUNT_BIAS_HALF_SCORE_TILES + bias_dist)
        better_tie = score == best_score and best_dot is not None and cost < best_cost
        if score > best_score or better_tie:
            best_score = score
            best_cost = cost
            best_dot = (target_x, target_y)
    if best_dot is None:
        emit_diagnostic(
            diagnostic_kind="hop_declined",
            hop_kind="dot",
            dots_total=len(ctx.map_fuel_dots),
            fuel=ctx.fuel,
            **tallies,
        )
        return None
    emit_diagnostic(
        diagnostic_kind="hop_selected",
        hop_kind="dot",
        target_x=best_dot[0],
        target_y=best_dot[1],
        score=round(best_score, 4),
        cost=best_cost,
        hunt_biased=1 if hunt_bias is not None else 0,
    )
    return best_dot


def is_recently_attempted(
    attempted: dict[str, int],
    x: int,
    y: int,
    now_ms: int,
    *,
    ttl_ms: int,
) -> bool:
    """Return True when a coordinate carries a live attempt mark.

    Args:
        attempted: Attempt marks keyed by "x,y" with dispatch timestamps.
        x: Target X coordinate.
        y: Target Y coordinate.
        now_ms: Current timestamp for TTL evaluation.
        ttl_ms: Mark lifetime in milliseconds.

    Returns:
        True if the coordinate was attempted within the TTL.
    """
    attempted_ms = attempted.get(coord_key(x, y))
    return attempted_ms is not None and now_ms - attempted_ms <= ttl_ms


def record_attempt_mark(
    attempted: dict[str, int],
    x: int,
    y: int,
    now_ms: int,
    *,
    ttl_ms: int,
) -> dict[str, int]:
    """Return attempt marks with expired entries pruned and (x, y) recorded.

    Args:
        attempted: Attempt marks keyed by "x,y" with dispatch timestamps.
        x: Target X coordinate to record.
        y: Target Y coordinate to record.
        now_ms: Dispatch timestamp recorded for the new mark.
        ttl_ms: Mark lifetime in milliseconds used for pruning.

    Returns:
        New attempt-mark mapping.
    """
    pruned = {
        key: marked_ms for key, marked_ms in attempted.items() if now_ms - marked_ms <= ttl_ms
    }
    pruned[coord_key(x, y)] = now_ms
    return pruned


def _open_map_for_dots(
    ctx: DecideCtx,
    *,
    mode: BehaviorMode,
    score: int,
    base_state: AIStateDict,
) -> TickDecisionDict | None:
    """Dispatch ``map_open`` to populate the fuel-dot atlas.

    The atlas arrives with the 0x4C MapData response, so the first hop
    of a session may need one map open before any dot candidates
    exist. Guarded by ``map_open_cooldown_ms``: if a recent map open
    produced no dots there is nothing more to learn and the caller's
    exit path takes over.

    Args:
        ctx: Decision context.
        mode: Behavior mode label for the decision.
        score: Behavior score for the map open.
        base_state: Base AI state to rewrite for the produced command.

    Returns:
        Map-open decision, or ``None`` when a recent map open already
        failed to yield dots.
    """
    map_age_ms = ctx.timestamp_ms - ctx.ai_state["last_map_open_ms"]
    if ctx.ai_state["last_map_open_ms"] > 0 and map_age_ms <= ctx.config["map_open_cooldown_ms"]:
        return None
    emit_ai("opening map to load the fuel-dot atlas")
    return make_decision(
        make_map_open_command(),
        mode,
        score,
        0,
        0,
        "map_for_dots",
        AIStateDict(
            **{
                **base_state,
                "last_map_open_ms": ctx.timestamp_ms,
            }
        ),
        ctx.equip,
    )


def make_resource_search_hop(
    ctx: DecideCtx,
    *,
    mode: BehaviorMode,
    score: int,
    reason: ReasonKind,
    ai_state: AIStateDict | None = None,
) -> TickDecisionDict | None:
    """Create a teleport decision to the nearest clean-viewport fuel dot.

    Landing on a dot auto-picks any fuel there, so each restock hop is
    partially self-funding. With an empty atlas the decision is a
    ``map_open`` (dots arrive with the 0x4C response). Returns
    ``None`` when no dot qualifies -- the caller raises rather than
    fall back to a blind hop.

    Args:
        ctx: Decision context.
        mode: Behavior mode label for the decision.
        score: Behavior score for the hop.
        reason: Behavior reason label.
        ai_state: Optional AI state base to rewrite before returning.

    Returns:
        Teleport (or atlas-loading map-open) decision, or ``None``
        when no fresh-viewport dot hop is possible from here.
    """
    base_state = ctx.base if ai_state is None else ai_state
    if not ctx.map_fuel_dots:
        return _open_map_for_dots(ctx, mode=mode, score=score, base_state=base_state)
    target = _pick_fresh_dot_hop(ctx)
    if target is None:
        return None
    target_x, target_y = target
    emit_ai(
        "fuel-dot hop to (%d,%d) (dual=%d homing=%d radar=%d)",
        target_x,
        target_y,
        ctx.inventory["dual_shots"]["count"],
        ctx.inventory["homing_shots"]["count"],
        ctx.inventory["extra_radars"]["count"],
    )
    return make_decision(
        make_teleport_command(target_x, target_y),
        mode,
        score,
        target_x,
        target_y,
        reason,
        clear_resource_target(base_state),
        ctx.equip,
    )


__all__ = [
    "is_recently_attempted",
    "make_resource_search_hop",
    "record_attempt_mark",
]
