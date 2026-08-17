"""Tests for the tile-aware forage planner."""

from __future__ import annotations

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.forage import plan_forage_search, select_forage_target
from tankpit_bot.bot.ai.mode_gates import (
    should_enter_collect,
    should_exit_collect,
)
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.scan_coverage import tile_key
from tankpit_bot.state.types import TankStateDict, make_tank_state
from tests.bot.ai._support import make_inventory, make_scanned_ai_state, make_world
from tests.in_memory_terrain_map import InMemoryTerrainMap

_VIEWPORT_LEFT = 92
_VIEWPORT_TOP = 92
_VIEWPORT_RIGHT = 107
_VIEWPORT_BOTTOM = 107


def _full_viewport_coverage(now_ms: int) -> dict[str, int]:
    """Return a coverage map that marks every tile in the default viewport.

    The default test viewport is the 16x16 block (92..107) x (92..107)
    produced by ``make_world(self_x=100, self_y=100)``.

    Args:
        now_ms: Timestamp to stamp every tile with.

    Returns:
        Coverage map keyed by ``"x,y"`` covering all 256 viewport tiles.
    """
    return {
        tile_key(x, y): now_ms
        for y in range(_VIEWPORT_TOP, _VIEWPORT_BOTTOM + 1)
        for x in range(_VIEWPORT_LEFT, _VIEWPORT_RIGHT + 1)
    }


def _ctx(
    *,
    radars: int,
    dual: int = 25,
    homing: int = 25,
    fuel: int = 1200,
    self_x: int = 100,
    self_y: int = 100,
    scanned_tiles: dict[str, int] | None = None,
    tanks: dict[str, TankStateDict] | None = None,
    terrain: InMemoryTerrainMap | None = None,
) -> DecideCtx:
    """Build a decision context for forage tests.

    Args:
        radars: Extra-radar count.
        dual: Dual-shot count.
        homing: Homing-shot count.
        fuel: Current fuel.
        self_x: Tank X coordinate.
        self_y: Tank Y coordinate.
        scanned_tiles: Seed world tile-coverage map.
        tanks: Visible tanks (enemies) keyed by id.
        terrain: Optional terrain map; when ``None`` no terrain rules apply.

    Returns:
        Decision context at timestamp 100000.
    """
    ws = WorldService()
    world, self_state = make_world(self_x=self_x, self_y=self_y, fuel=fuel, scanned=False)
    if tanks is not None:
        world["tanks"].update(tanks)
    if scanned_tiles is not None:
        world["scanned_tiles"] = scanned_tiles
    inventory = make_inventory(default_count=30)
    inventory["dual_shots"]["count"] = dual
    inventory["homing_shots"]["count"] = homing
    inventory["extra_radars"]["count"] = radars
    return DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        inventory,
        100000,
        terrain,
        "",
        ws=ws,
    )


def _enemy(tank_id: int, x: int, y: int) -> dict[str, TankStateDict]:
    """Build a single visible enemy tank entry.

    Args:
        tank_id: Tank id.
        x: Enemy X coordinate.
        y: Enemy Y coordinate.

    Returns:
        ``{str(id): tank_state}`` fragment.
    """
    return {
        str(tank_id): make_tank_state(
            tank_id=tank_id,
            x=x,
            y=y,
            team=2,
            rank=1,
            name="red-1",
            is_self=False,
            is_bot=False,
            damage_state=0,
            timestamp_ms=100000,
        )
    }


class TestRadarRestockTrigger:
    """Tests for the radar-restock entry/exit that owns foraging."""

    def test_restock_enters_at_zero_extras(self) -> None:
        """Zero extras (below break) enters the restock mode."""
        assert should_enter_collect(_ctx(radars=0)) is True

    def test_restock_holds_below_the_resume_buffer(self) -> None:
        """The mode does not exit until radars reach the resume buffer."""
        assert should_exit_collect(_ctx(radars=0)) is False
        assert should_exit_collect(_ctx(radars=3)) is False

    def test_restock_releases_once_radars_reach_the_cap_floor(self) -> None:
        """Radars within 5 of cap release restock (contract 2026-07-25).

        At rank 2 the cap is 30, so the release floor is 25 -- the old
        resume buffer (20) no longer releases the mode.
        """
        assert should_exit_collect(_ctx(radars=20, dual=30, homing=30)) is False
        assert should_exit_collect(_ctx(radars=25, dual=30, homing=30)) is True

    def test_restock_ignores_visible_threats(self) -> None:
        """Rebuilding the kit outranks chasing a wanderer it cannot beat.

        Restock does NOT yield to a visible enemy below the buffer --
        "get the radars before fighting".
        """
        ctx = _ctx(radars=0, tanks=_enemy(7, 104, 100))

        assert should_enter_collect(ctx) is True
        assert should_exit_collect(ctx) is False


class TestSelectForageTarget:
    """Tests for ``select_forage_target``."""

    def test_picks_position_maximising_next_radar_coverage(self) -> None:
        """A destination is chosen so its 5x5 footprint covers many uncovered tiles.

        Coverage layout: every viewport tile scanned EXCEPT a 4x4
        unscanned cluster at (103..106) x (103..106). A free radar at
        (104, 104) has a 5x5 footprint of (102..106) x (102..106),
        which fully contains all 16 cluster tiles -- score 16, the
        maximum possible. Tank at (100, 100); distance 8 from (104,
        104). The nearest-unscanned picker would have walked to (103,
        103) (distance 6), missing 4 of the 16 cluster tiles on the
        next radar. The coverage-maximising picker walks 8 tiles to
        sweep the cluster in one shot.
        """
        coverage = _full_viewport_coverage(100000)
        for y in range(103, 107):
            for x in range(103, 107):
                del coverage[f"{x},{y}"]
        ctx = _ctx(radars=0, scanned_tiles=coverage)

        target = select_forage_target(ctx)

        if target is None:
            raise AssertionError("expected a destination covering the unscanned cluster")
        assert target == (104, 104)

    def test_returns_none_when_viewport_is_fully_covered(self) -> None:
        """Every viewport tile covered yields no walk target."""
        ctx = _ctx(radars=0, scanned_tiles=_full_viewport_coverage(100000))

        assert select_forage_target(ctx) is None


class TestForageSearch:
    """Tests for ``plan_forage_search`` (the parameterized forager)."""

    def test_dispatches_radar_when_viewport_has_unscanned_tiles(self) -> None:
        """An unscanned viewport with affordable radar triggers a scan."""
        ctx = _ctx(radars=0)

        decision = plan_forage_search(
            ctx,
            ctx.ai_state,
            score=925,
            behavior_mode="COLLECT",
        )

        if decision is None:
            raise AssertionError("expected a forage radar decision")
        assert decision["command"]["cmd_type"] == "radar"
        assert decision["behavior"]["reason_kind"] == "forage_radar"
        assert decision["behavior"]["mode"] == "COLLECT"

    def test_dispatches_radar_with_fuel_mode_tag(self) -> None:
        """Behavior-mode label is stamped from the caller-supplied value."""
        ctx = _ctx(radars=0)

        decision = plan_forage_search(
            ctx,
            ctx.ai_state,
            score=900,
            behavior_mode="COLLECT",
        )

        if decision is None:
            raise AssertionError("expected a forage radar decision")
        assert decision["behavior"]["mode"] == "COLLECT"
        assert decision["behavior"]["reason_kind"] == "forage_radar"
        assert decision["behavior"]["score"] == 900

    def test_stocked_extras_below_the_spend_floor_yield_nothing(self) -> None:
        """With extras stocked and a scan not worth its cost, foraging is over.

        An extra-radar scan reveals the WHOLE viewport from anywhere, so
        once the shared spend economics refuse the scan there is no walk
        that improves it and the free radar can never fire. The planner
        must yield ``None`` and let the caller teleport to fresh ground.

        Returning a decision anyway is the artax flags 1-3 regression
        (2026-08-06): the forager kept handing back one-tile walks,
        which starved the collect hop one rung below and produced a
        one-tile-per-tick edge crawl. Four uncovered tiles is under
        ``RADAR_SPEND_REVEAL_FLOOR_TILES`` (32) but above zero, so the
        viewport is NOT fully covered and a walk target genuinely
        exists -- which is exactly what the fall-through would return.
        """
        coverage = _full_viewport_coverage(100000)
        for uncovered in ("95,95", "95,96", "96,95", "96,96"):
            del coverage[uncovered]
        ctx = _ctx(radars=30, scanned_tiles=coverage, terrain=InMemoryTerrainMap())

        decision = plan_forage_search(
            ctx,
            ctx.ai_state,
            score=925,
            behavior_mode="COLLECT",
        )

        assert decision is None

    def test_control_stocked_extras_above_the_floor_still_scan(self) -> None:
        """Control for the test above: the same setup CAN produce a decision.

        Identical context except that enough of the viewport is
        uncovered to clear the spend floor. This must return a radar
        decision, so the ``None`` above is attributable to the spend
        gate rather than to a context that could never decide anything.
        """
        coverage = _full_viewport_coverage(100000)
        for row in range(92, 96):
            for col in range(92, 104):
                del coverage[f"{col},{row}"]
        ctx = _ctx(radars=30, scanned_tiles=coverage, terrain=InMemoryTerrainMap())

        decision = plan_forage_search(
            ctx,
            ctx.ai_state,
            score=925,
            behavior_mode="COLLECT",
        )

        if decision is None:
            raise AssertionError("48 uncovered tiles must clear the spend floor")
        assert decision["command"]["cmd_type"] == "radar"

    def test_walks_when_the_free_radar_would_reveal_nothing_new(self) -> None:
        """An already-covered radar footprint yields a walk, not a scan.

        This is the production-reachable version of the walk branch.
        It used to be forced with ``radar_affordable=False``, a gate no
        caller could ever set — radar is free at any fuel, so the only
        thing that makes the radar unproductive is the free 5x5 around
        the tank already being scanned while the wider viewport is not.
        """
        # The free radar covers (tank +/- 2); scan exactly that, so a
        # scan would reveal nothing, while the viewport stays uncovered.
        coverage = {f"{x},{y}": 100000 for x in range(98, 103) for y in range(98, 103)}
        ctx = _ctx(radars=0, scanned_tiles=coverage)

        decision = plan_forage_search(
            ctx,
            ctx.ai_state,
            score=925,
            behavior_mode="COLLECT",
        )

        if decision is None:
            raise AssertionError("expected a forage walk decision")
        assert decision["command"]["cmd_type"] == "move"
        assert decision["behavior"]["reason_kind"] == "forage_sweep"

    def test_covered_viewport_at_zero_extras_continues_the_lawnmower(self) -> None:
        """REVERSED 2026-08-14 (user free-radar doctrine): a covered
        viewport at zero extras used to yield None for a teleport-out;
        it now frontier-walks toward the adjacent unscanned band so
        the anchored window slides and the free-radar loop continues
        -- teleports never serve coverage."""
        ctx = _ctx(radars=0, scanned_tiles=_full_viewport_coverage(100000))

        decision = plan_forage_search(
            ctx,
            ctx.ai_state,
            score=925,
            behavior_mode="COLLECT",
        )

        if decision is None:
            raise AssertionError("expected a frontier walk")
        assert decision["behavior"]["reason_kind"] == "forage_frontier_walk"

    def test_returns_none_when_walk_falls_back_to_unaffordable_teleport(self) -> None:
        """A rock-walled tile reachable only by teleport that exceeds available fuel yields None.

        The ``hunt_min_fuel`` reserve was dropped 2026-06-24, so the
        unaffordable threshold is now the raw teleport cost, not
        cost + reserve. From (100,100) to (104,100) the teleport
        cost is 24 fuel; at fuel=10 even the short-hop fallback is
        unaffordable.
        """
        coverage = _full_viewport_coverage(100000)
        # Open one tile on the far side of a rock wall.
        del coverage["104,100"]
        rocks = {(102, y): InMemoryTerrainMap.ROCK for y in range(92, 108)}
        terrain = InMemoryTerrainMap(rocks)
        # Fuel below the raw teleport cost so no hop is affordable.
        ctx = _ctx(
            radars=0,
            fuel=10,
            scanned_tiles=coverage,
            terrain=terrain,
        )

        decision = plan_forage_search(
            ctx,
            ctx.ai_state,
            score=925,
            behavior_mode="COLLECT",
        )

        assert decision is None


class TestFrontierWalk:
    """The zero-extras lawnmower continues into the next viewport over."""

    def test_covered_viewport_walks_toward_the_unscanned_band(self) -> None:
        """User doctrine 2026-08-14: viewport done -> walk to the edge
        facing the most unscanned ground; the anchored window slides
        and the scan-walk-scan loop resumes -- no teleport, ever."""
        ctx = _ctx(radars=0, scanned_tiles=_full_viewport_coverage(100000))

        decision = plan_forage_search(
            ctx,
            ctx.ai_state,
            score=925,
            behavior_mode="COLLECT",
        )

        if decision is None:
            raise AssertionError("expected a frontier walk")
        assert decision["behavior"]["reason_kind"] == "forage_frontier_walk"
        assert decision["command"]["cmd_type"] == "move"

    def test_fully_scanned_surroundings_yield_to_the_search_hop(self) -> None:
        """Every adjacent band covered too: relocation is the hop's job."""
        coverage = _full_viewport_coverage(100000)
        from tankpit_bot.state.viewport_geometry import viewport_visible_bounds

        left, top, right, bottom = viewport_visible_bounds(_ctx(radars=0).world["viewport"])
        for x in range(max(left - 8, 0), min(right + 8, 255) + 1):
            for y in range(max(top - 8, 0), min(bottom + 8, 255) + 1):
                coverage[f"{x},{y}"] = 100000
        ctx = _ctx(radars=0, scanned_tiles=coverage)

        decision = plan_forage_search(
            ctx,
            ctx.ai_state,
            score=925,
            behavior_mode="COLLECT",
        )

        assert decision is None

    def test_stocked_covered_viewport_never_frontier_walks(self) -> None:
        """With extras the scan reaches the whole viewport from
        anywhere -- relocation stays the hop's job, not a walk's."""
        ctx = _ctx(radars=5, scanned_tiles=_full_viewport_coverage(100000))

        decision = plan_forage_search(
            ctx,
            ctx.ai_state,
            score=925,
            behavior_mode="COLLECT",
        )

        assert decision is None
