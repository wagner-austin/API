"""Tests for the staleness-seeking forage frontier."""

from __future__ import annotations

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.forage_frontier import (
    BLOCK_TILES,
    FRONTIER_VISIT_TTL_MS,
    plan_forage_frontier_hop,
)
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.inventory import InventoryState
from tankpit_bot.sniffer.world_service import WorldService
from tests.bot.ai._support import (
    make_inventory,
    make_scanned_ai_state,
    make_world,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap

_NOW = 100000


def _deficient_inventory() -> InventoryState:
    """Inventory below the dual cap so the restock gate is open."""
    inventory = make_inventory()
    inventory["dual_shots"]["count"] = 10
    return inventory


def _frontier_ctx(
    *,
    fuel: int = 800,
    inventory: InventoryState | None = None,
    terrain: InMemoryTerrainMap | None = None,
    ai_state: AIStateDict | None = None,
    stale_blocks: tuple[tuple[int, int], ...] = ((8, 6),),
    ws: WorldService | None = None,
) -> DecideCtx:
    """Build a ctx where EVERY block is covered except ``stale_blocks``.

    Self sits at (100,100) in block (6,6). Block (8,6)'s center is
    (136,104) — chebyshev 36, beyond any walking, a plain teleport
    goal. Block (6,5)'s center is (104,88) — chebyshev 12.
    """
    ws = ws if ws is not None else WorldService()
    # scanned=False: the frontier judges coverage itself, so the
    # helper stamps exactly one live tile per non-stale block.
    world, self_state = make_world(fuel=fuel, scanned=False)
    for bx in range(16):
        for by in range(16):
            if (bx, by) in stale_blocks:
                continue
            world["scanned_tiles"][f"{bx * BLOCK_TILES + 1},{by * BLOCK_TILES + 1}"] = _NOW
    return DecideCtx(
        world,
        self_state,
        ai_state if ai_state is not None else make_scanned_ai_state(),
        inventory if inventory is not None else _deficient_inventory(),
        _NOW,
        terrain if terrain is not None else InMemoryTerrainMap(),
        "",
        ws=ws,
    )


def _target(decision_x: int, decision_y: int) -> tuple[int, int]:
    """Readability helper for asserted goal coordinates."""
    return (decision_x, decision_y)


def test_stocked_inventory_declines() -> None:
    """Full restock bars: the frontier has no reason to move."""
    ctx = _frontier_ctx(inventory=make_inventory())

    assert plan_forage_frontier_hop(ctx, ctx.base) is None


def test_missing_terrain_declines() -> None:
    """No terrain map yet: no passability judgement, no hop."""
    world, self_state = make_world(fuel=800, scanned=False)
    ctx = DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        _deficient_inventory(),
        _NOW,
        None,
        "",
        ws=WorldService(),
    )

    assert plan_forage_frontier_hop(ctx, ctx.base) is None


def test_fully_covered_field_declines() -> None:
    """Nothing stale anywhere: the frontier yields to the sweep."""
    ctx = _frontier_ctx(stale_blocks=())

    assert plan_forage_frontier_hop(ctx, ctx.base) is None


def test_zero_extras_declines_so_the_walk_forager_can_run() -> None:
    """A hop is only worth its teleport if the landing can be revealed.

    With an extra, one ~10-fuel press reveals the whole 16x16 viewport.
    At zero extras the free radar reveals 25 tiles, so the hop has paid
    ~105 fuel for what walking five tiles and scanning buys for ~15
    ([[radar-mechanics]]). Measured on a fresh recruit 2026-09-01: 22
    hops, zero walk-forages, fuel oscillating 890-1000 against a 1000
    cap all session and banking nothing. Declining here is what lets
    the cascade reach ``plan_forage_search``.
    """
    inventory = _deficient_inventory()
    inventory["extra_radars"]["count"] = 0
    ctx = _frontier_ctx(inventory=inventory)

    assert plan_forage_frontier_hop(ctx, ctx.base) is None


def test_one_extra_is_enough_to_justify_the_hop() -> None:
    """The gate is on ABILITY to exploit the landing, not on a hoard.

    A single extra reveals the whole landing viewport, which is the
    entire premise of travelling to it — so the bar is one, not a
    reserve. Pinned separately so a future stock threshold cannot be
    slipped in here unnoticed.
    """
    inventory = _deficient_inventory()
    inventory["extra_radars"]["count"] = 1
    ctx = _frontier_ctx(inventory=inventory)

    decision = plan_forage_frontier_hop(ctx, ctx.base)

    if decision is None:
        raise AssertionError("one extra should justify the hop")
    assert decision["behavior"]["reason_kind"] == "forage_frontier_hop"


def test_targets_the_nearest_stale_block() -> None:
    """The nearest block with no live coverage wins."""
    ctx = _frontier_ctx(stale_blocks=((8, 6), (4, 6)))

    decision = plan_forage_frontier_hop(ctx, ctx.base)

    if decision is None:
        raise AssertionError("expected a frontier decision")
    assert decision["behavior"]["reason_kind"] == "forage_frontier_hop"
    # (72,104) at chebyshev 28 beats (136,104) at 36.
    assert _target(decision["behavior"]["target_x"], decision["behavior"]["target_y"]) == (72, 104)
    assert decision["updated_ai_state"]["forage_goal_x"] == 72
    assert decision["updated_ai_state"]["forage_goal_y"] == 104


def test_expired_coverage_reopens_a_block() -> None:
    """A stamp older than the forage TTL no longer counts as looked-at."""
    ctx = _frontier_ctx(stale_blocks=((8, 6), (6, 5)))
    # Block (6,5): expired stamp only -- it must re-qualify and win
    # over the farther (8,6).
    ctx.world["scanned_tiles"][f"{6 * BLOCK_TILES + 1},{5 * BLOCK_TILES + 1}"] = _NOW - 200_000

    decision = plan_forage_frontier_hop(ctx, ctx.base)

    if decision is None:
        raise AssertionError("expected a frontier decision")
    assert _target(decision["behavior"]["target_x"], decision["behavior"]["target_y"]) == (104, 88)


def test_visited_tombstone_is_skipped_until_ttl() -> None:
    """An arrived-at block stays off the circuit for the visit TTL."""
    ws = WorldService()
    ws.forage_visited["136,104"] = _NOW - 1000
    ctx = _frontier_ctx(ws=ws)

    assert plan_forage_frontier_hop(ctx, ctx.base) is None


def test_expired_tombstone_is_pruned() -> None:
    """A tombstone past the TTL is dropped and the block re-qualifies."""
    ws = WorldService()
    ws.forage_visited["136,104"] = _NOW - FRONTIER_VISIT_TTL_MS - 1
    ctx = _frontier_ctx(ws=ws)

    decision = plan_forage_frontier_hop(ctx, ctx.base)

    if decision is None:
        raise AssertionError("expected a frontier decision")
    assert _target(decision["behavior"]["target_x"], decision["behavior"]["target_y"]) == (136, 104)
    assert "136,104" not in ws.forage_visited


def test_impassable_center_is_skipped() -> None:
    """A water block center never becomes a goal."""
    terrain = InMemoryTerrainMap(terrain_data={(136, 104): "W"})
    ctx = _frontier_ctx(terrain=terrain)

    assert plan_forage_frontier_hop(ctx, ctx.base) is None


def test_unaffordable_beyond_window_is_skipped() -> None:
    """Out-of-window blocks must be teleport-affordable above the floor."""
    # Fuel 210 leaves a 10-fuel budget: the chebyshev-36 hop to
    # (136,104) costs far more, and the center is outside the window.
    ctx = _frontier_ctx(fuel=210)

    assert plan_forage_frontier_hop(ctx, ctx.base) is None


def test_in_window_blocks_are_looked_at_by_sight() -> None:
    """A window in view IS looked at: no walk to its own center.

    Operator flag (2026-08-28 World watch): "walking to a spot, then
    walking back to another spot, not doing anything" -- 0x5A
    enumerates a window's containers radar-free, so traveling to an
    in-window center reveals nothing. The center is stamped by sight
    and the frontier declines.
    """
    ws = WorldService()
    ctx = _frontier_ctx(fuel=210, stale_blocks=((6, 6),), ws=ws)

    decision = plan_forage_frontier_hop(ctx, ctx.base)

    assert decision is None
    assert "104,104" in ws.forage_visited


def test_sibling_goal_blocks_are_skipped() -> None:
    """A block a fleet sibling is already traveling to is not chosen.

    Operator observation (2026-08-28): "no awareness of who's
    collecting what" -- two bots latched the same stale block
    mid-flight; coverage sharing only helps after a scan lands.
    """
    ws = WorldService()
    ws.fleet_forage_goals = {"arterial": (136, 104)}
    ctx = _frontier_ctx(ws=ws, stale_blocks=((8, 6), (4, 6)))

    decision = plan_forage_frontier_hop(ctx, ctx.base)

    if decision is None:
        raise AssertionError("expected a frontier decision")
    # (136,104) is nearer... it belongs to arterial; (72,104) wins.
    assert _target(decision["behavior"]["target_x"], decision["behavior"]["target_y"]) == (72, 104)


def test_far_goal_travels_by_teleport() -> None:
    """A goal beyond the window dispatches a teleport."""
    ctx = _frontier_ctx()

    decision = plan_forage_frontier_hop(ctx, ctx.base)

    if decision is None:
        raise AssertionError("expected a frontier decision")
    assert decision["command"]["cmd_type"] == "teleport"
    assert _target(decision["behavior"]["target_x"], decision["behavior"]["target_y"]) == (136, 104)


def test_standing_goal_outranks_a_nearer_fresh_block() -> None:
    """The latched goal is served first: no target swap mid-travel."""
    ai_state = AIStateDict(
        **{**make_scanned_ai_state(), "forage_goal_x": 136, "forage_goal_y": 104}
    )
    ctx = _frontier_ctx(ai_state=ai_state, stale_blocks=((8, 6), (6, 5)))

    decision = plan_forage_frontier_hop(ctx, ctx.base)

    if decision is None:
        raise AssertionError("expected a frontier decision")
    # (104,88) is nearer and stale, but the standing goal wins.
    assert _target(decision["behavior"]["target_x"], decision["behavior"]["target_y"]) == (136, 104)


def test_own_tile_center_is_never_a_goal() -> None:
    """Standing exactly on the only stale center: nothing to travel to."""
    world, self_state = make_world(self_x=104, self_y=104, fuel=800, scanned=False)
    for bx in range(16):
        for by in range(16):
            if (bx, by) != (6, 6):
                world["scanned_tiles"][f"{bx * BLOCK_TILES + 1},{by * BLOCK_TILES + 1}"] = _NOW
    ctx = DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        _deficient_inventory(),
        _NOW,
        InMemoryTerrainMap(),
        "",
        ws=WorldService(),
    )

    assert plan_forage_frontier_hop(ctx, ctx.base) is None


def test_unwalkable_in_window_center_is_passed_over() -> None:
    """A failed-move in-window center falls through to no decision."""
    ws = WorldService()
    ws.failed_move_targets["104,104"] = _NOW
    ctx = _frontier_ctx(fuel=210, stale_blocks=((6, 6),), ws=ws)

    assert plan_forage_frontier_hop(ctx, ctx.base) is None


def test_goal_attempts_increment_and_reset() -> None:
    """Serving the standing goal counts up; a fresh goal starts at 1."""
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "forage_goal_x": 136,
            "forage_goal_y": 104,
            "forage_goal_attempts": 1,
        }
    )
    ctx = _frontier_ctx(ai_state=ai_state)

    decision = plan_forage_frontier_hop(ctx, ctx.base)

    if decision is None:
        raise AssertionError("expected a frontier decision")
    assert decision["updated_ai_state"]["forage_goal_attempts"] == 2


def test_unlandable_goal_is_tombstoned_at_the_attempt_cap() -> None:
    """Three bounced throws prove the block unlandable and move on.

    Arterial (2026-08-28 20:52) paid 14+ teleports at (120,104) with
    every landing displaced 9-25 tiles out: arrival never fired, the
    latch never released, and each re-throw burned ~20 fuel.
    """
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "forage_goal_x": 136,
            "forage_goal_y": 104,
            "forage_goal_attempts": 3,
        }
    )
    ws = WorldService()
    ctx = _frontier_ctx(ai_state=ai_state, ws=ws, stale_blocks=((8, 6), (6, 5)))

    decision = plan_forage_frontier_hop(ctx, ctx.base)

    assert ws.forage_visited["136,104"] == _NOW
    if decision is None:
        raise AssertionError("expected a fresh frontier decision")
    assert _target(decision["behavior"]["target_x"], decision["behavior"]["target_y"]) == (104, 88)
    assert decision["updated_ai_state"]["forage_goal_attempts"] == 1


def test_arrival_tombstones_the_goal_and_picks_fresh() -> None:
    """Standing within two tiles of the goal releases and tombstones it."""
    ai_state = AIStateDict(
        **{**make_scanned_ai_state(), "forage_goal_x": 101, "forage_goal_y": 101}
    )
    ws = WorldService()
    ctx = _frontier_ctx(ai_state=ai_state, ws=ws)

    decision = plan_forage_frontier_hop(ctx, ctx.base)

    assert ws.forage_visited["101,101"] == _NOW
    if decision is None:
        raise AssertionError("expected a fresh frontier decision")
    assert _target(decision["behavior"]["target_x"], decision["behavior"]["target_y"]) == (136, 104)


def test_claimed_container_filter_prunes_and_passes_through() -> None:
    """Sibling-claimed containers vanish from ctx.filtered; others stay."""
    from tankpit_bot.state.types import make_container_state

    ws = WorldService()
    ws.fleet_claimed_containers = {"120,100", "999,999"}
    world, self_state = make_world(fuel=800, scanned=False)
    world["containers"]["120,100"] = make_container_state(
        x=120, y=100, is_fuel=False, volume=0, timestamp_ms=_NOW
    )
    world["containers"]["121,101"] = make_container_state(
        x=121, y=101, is_fuel=False, volume=0, timestamp_ms=_NOW
    )
    ctx = DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        _deficient_inventory(),
        _NOW,
        InMemoryTerrainMap(),
        "",
        ws=ws,
    )

    assert "120,100" not in ctx.filtered["containers"]
    assert "121,101" in ctx.filtered["containers"]

    # A claim set that touches nothing believed returns the world as-is.
    ws_untouched = WorldService()
    ws_untouched.fleet_claimed_containers = {"999,999"}
    world2, self2 = make_world(fuel=800, scanned=False)
    world2["containers"]["121,101"] = make_container_state(
        x=121, y=101, is_fuel=False, volume=0, timestamp_ms=_NOW
    )
    ctx2 = DecideCtx(
        world2,
        self2,
        make_scanned_ai_state(),
        _deficient_inventory(),
        _NOW,
        InMemoryTerrainMap(),
        "",
        ws=ws_untouched,
    )
    assert "121,101" in ctx2.filtered["containers"]


def test_partially_affordable_frontier_picks_the_reachable_block() -> None:
    """An unaffordable far block is skipped for an affordable nearer one."""
    # Fuel 320 gives a 120 budget: (104,88) at chebyshev 12 is
    # affordable; (8,136) at chebyshev ~92 is not and must be skipped.
    ctx = _frontier_ctx(fuel=320, stale_blocks=((6, 5), (0, 8)))

    decision = plan_forage_frontier_hop(ctx, ctx.base)

    if decision is None:
        raise AssertionError("expected a frontier decision")
    assert _target(decision["behavior"]["target_x"], decision["behavior"]["target_y"]) == (104, 88)
