"""Tests for the equipment-foraging grid sweep and its mode trigger."""

from __future__ import annotations

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.forage import (
    plan_forage_search,
    select_forage_cell_target,
)
from tankpit_bot.bot.ai.mode_controller import (
    should_enter_recover_equipment,
    should_exit_recover_equipment,
)
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.state.types import TankStateDict, make_tank_state
from tests.bot.ai._support import make_inventory, make_scanned_ai_state, make_world
from tests.in_memory_terrain_map import InMemoryTerrainMap


def _ctx(
    *,
    radars: int,
    dual: int = 25,
    homing: int = 25,
    fuel: int = 800,
    self_x: int = 100,
    self_y: int = 100,
    local_scan_cells: dict[str, int] | None = None,
    tanks: dict[str, TankStateDict] | None = None,
) -> DecideCtx:
    """Build a decision context for forage tests.

    Args:
        radars: Extra-radar count.
        dual: Dual-shot count.
        homing: Homing-shot count.
        fuel: Current fuel.
        self_x: Tank X coordinate.
        self_y: Tank Y coordinate.
        local_scan_cells: Seed coverage grid.
        tanks: Visible tanks (enemies) keyed by id.

    Returns:
        Decision context at timestamp 100000.
    """
    world, self_state = make_world(self_x=self_x, self_y=self_y, fuel=fuel, scanned=False)
    if tanks is not None:
        world["tanks"].update(tanks)
    inventory = make_inventory(default_count=30)
    inventory["dual_shots"]["count"] = dual
    inventory["homing_shots"]["count"] = homing
    inventory["extra_radars"]["count"] = radars
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "local_scan_cells": local_scan_cells if local_scan_cells is not None else {},
        }
    )
    return DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")


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
        assert should_enter_recover_equipment(_ctx(radars=0)) is True

    def test_restock_holds_below_the_resume_buffer(self) -> None:
        """The mode does not exit until radars reach the resume buffer."""
        assert should_exit_recover_equipment(_ctx(radars=0)) is False
        assert should_exit_recover_equipment(_ctx(radars=3)) is False

    def test_restock_releases_once_the_buffer_is_full(self) -> None:
        """At the resume buffer the bot leaves restock and may hunt again."""
        assert should_exit_recover_equipment(_ctx(radars=20)) is True

    def test_restock_ignores_visible_threats(self) -> None:
        """Rebuilding the kit outranks chasing a wanderer it cannot beat.

        Unlike the first forager, restock does NOT yield to a visible
        enemy below the buffer -- "get the radars before fighting".
        """
        ctx = _ctx(radars=0, tanks=_enemy(7, 104, 100))

        assert should_enter_recover_equipment(ctx) is True
        assert should_exit_recover_equipment(ctx) is False


class TestCellSelection:
    """Tests for select_forage_cell_target."""

    def test_picks_nearest_uncovered_neighbor_center(self) -> None:
        """With the home cell covered, the nearest neighbor center wins.

        The tank cell at (100,100) is (20,20); its center is (102,102).
        Covering it forces selection to an adjacent ring-1 cell center.
        """
        ctx = _ctx(radars=0, local_scan_cells={"20,20": 100000})

        target = select_forage_cell_target(ctx)

        if target is None:
            raise AssertionError("expected a nearest uncovered cell center")
        cx, cy = target
        assert (cx // 5, cy // 5) != (20, 20)
        assert max(abs(cx // 5 - 20), abs(cy // 5 - 20)) == 1

    def test_returns_none_when_all_local_cells_covered(self) -> None:
        """Every cell within the search ring covered yields no target."""
        covered = {
            f"{cx},{cy}": 100000 for cx in range(20 - 12, 20 + 13) for cy in range(20 - 12, 20 + 13)
        }
        ctx = _ctx(radars=0, local_scan_cells=covered)

        assert select_forage_cell_target(ctx) is None


class TestForageSearch:
    """Tests for plan_forage_search."""

    def test_radars_the_uncovered_home_cell_first(self) -> None:
        """An uncovered home cell is scanned with the free built-in radar."""
        ctx = _ctx(radars=0)

        decision = plan_forage_search(ctx, ctx.ai_state, 925)

        if decision is None:
            raise AssertionError("expected a forage radar decision")
        assert decision["command"]["cmd_type"] == "radar"
        assert decision["behavior"]["reason"] == "forage_radar"
        assert decision["updated_ai_state"]["local_scan_cells"] == {"20,20": 100000}

    def test_moves_to_next_cell_once_home_is_covered(self) -> None:
        """A covered home cell makes the sweep walk to the next center."""
        ctx = _ctx(radars=0, local_scan_cells={"20,20": 100000})

        decision = plan_forage_search(ctx, ctx.ai_state, 925)

        if decision is None:
            raise AssertionError("expected a forage sweep move decision")
        assert decision["command"]["cmd_type"] == "move"
        assert decision["behavior"]["reason"] == "forage_sweep"

    def test_returns_none_when_swept_and_unscannable(self) -> None:
        """A covered home cell with no fuel for radar and all cells swept.

        At fuel 5 the radar is unaffordable and every nearby cell is
        covered, so the forager yields to the recovery fallback.
        """
        covered = {
            f"{cx},{cy}": 100000 for cx in range(20 - 12, 20 + 13) for cy in range(20 - 12, 20 + 13)
        }
        ctx = _ctx(radars=0, fuel=5, local_scan_cells=covered)

        assert plan_forage_search(ctx, ctx.ai_state, 925) is None


class TestForageEdgeCases:
    """Tests for edge-case paths in forage cell selection and planning."""

    def test_ring_cell_out_of_bounds_is_skipped(self) -> None:
        """Cells at negative indices near the map edge are silently skipped.

        The tank at position (2, 2) sits in cell (0, 0).  Covering that
        cell forces the sweep to ring-1, which includes cells at
        negative indices (e.g. (-1, -1)).  The bounds check must skip
        those and still return a valid in-bounds cell center.
        """
        ctx = _ctx(radars=0, self_x=2, self_y=2, local_scan_cells={"0,0": 100000})

        target = select_forage_cell_target(ctx)

        if target is None:
            raise AssertionError("expected an in-bounds ring-1 cell center")
        tx, ty = target
        cell_x, cell_y = tx // 5, ty // 5
        assert cell_x >= 0 and cell_y >= 0
        assert max(abs(cell_x - 0), abs(cell_y - 0)) == 1

    def test_plan_forage_returns_none_when_walk_or_teleport_returns_none(self) -> None:
        """An enemy occupying the only uncovered cell center blocks the move.

        Cell (21, 20) center is (107, 102).  An enemy parked on that
        tile makes ``walk_or_teleport`` return ``None`` because the
        no-terrain path rejects enemy-occupied move targets.
        """
        ring1_cells = [
            (cx, cy)
            for cx in range(19, 22)
            for cy in range(19, 22)
            if max(abs(cx - 20), abs(cy - 20)) == 1
        ]
        covered = {"20,20": 100000}
        for cx, cy in ring1_cells:
            if (cx, cy) != (21, 20):
                covered[f"{cx},{cy}"] = 100000
        ctx = _ctx(
            radars=0,
            local_scan_cells=covered,
            tanks=_enemy(7, 107, 102),
        )

        assert plan_forage_search(ctx, ctx.ai_state, 925) is None

    def test_plan_forage_returns_none_when_teleport_is_unaffordable(self) -> None:
        """A teleport fallback that exceeds the search reserve is rejected.

        Rocks wall off the walk path so ``walk_or_teleport`` falls back
        to teleport.  Fuel covers the raw teleport cost but not the
        search reserve (``hunt_min_fuel=100``), so
        ``can_afford_teleport_search`` returns False.
        """
        ring1_cells = [
            (cx, cy)
            for cx in range(19, 22)
            for cy in range(19, 22)
            if max(abs(cx - 20), abs(cy - 20)) == 1
        ]
        covered = {"20,20": 100000}
        for cx, cy in ring1_cells:
            if (cx, cy) != (21, 20):
                covered[f"{cx},{cy}"] = 100000

        rocks = {(103, y): InMemoryTerrainMap.ROCK for y in range(92, 108)}
        terrain = InMemoryTerrainMap(rocks)

        world, self_state = make_world(self_x=100, self_y=100, fuel=100, scanned=False)
        inventory = make_inventory(default_count=30)
        inventory["extra_radars"]["count"] = 0
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "local_scan_cells": covered,
            }
        )
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

        assert plan_forage_search(ctx, ctx.ai_state, 925) is None
