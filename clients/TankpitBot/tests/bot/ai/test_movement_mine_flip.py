"""Tests for the walk→teleport flip after a walk-over mine hit."""

from __future__ import annotations

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.movement import walk_or_teleport
from tankpit_bot.sniffer.world_state import get_world_service, reset_world_state
from tests.bot.ai._support import make_inventory, make_scanned_ai_state, make_world
from tests.in_memory_terrain_map import InMemoryTerrainMap


def _ctx(*, fuel: int = 800, terrain: InMemoryTerrainMap | None = None) -> DecideCtx:
    """Build a decision context at (100,100) with the given terrain.

    Args:
        fuel: Tank fuel.
        terrain: Static terrain map, or ``None`` for the wire-only view.

    Returns:
        Decision context.
    """
    world, self_state = make_world(fuel=fuel)
    return DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        make_inventory(),
        100000,
        terrain,
        "",
    )


def _stamp_mine_hit(at_ms: int) -> None:
    """Record a walk-over mine hit at the given time.

    Args:
        at_ms: Stamp for the flip window.
    """
    get_world_service().last_own_mine_hit_ms = at_ms


class TestMineFlip:
    """Tests for the reactive walk→teleport flip."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_recent_mine_hit_flips_the_approach_to_teleport(self) -> None:
        """After a walk-over the same destination is approached by air.

        User doctrine 2026-07-30 ("artax hit tons of yupplers mines"):
        walk in-viewport, but a mine hit means unrevealed mines sit on
        the walking route — the teleport landing is mine-immune by the
        displacement law, and walking resumes when the window lapses.
        """
        _stamp_mine_hit(99000)
        ctx = _ctx(terrain=InMemoryTerrainMap())

        command = walk_or_teleport(ctx, 105, 100, pickup_kind="fuel")

        if command is None:
            raise AssertionError("expected a command")
        assert command["cmd_type"] == "teleport"

    def test_expired_window_walks_again(self) -> None:
        """Outside the flip window the doctrine returns to walking."""
        _stamp_mine_hit(90000)
        ctx = _ctx(terrain=InMemoryTerrainMap())

        command = walk_or_teleport(ctx, 105, 100, pickup_kind="fuel")

        if command is None:
            raise AssertionError("expected a command")
        assert command["cmd_type"] != "teleport"

    def test_unaffordable_flip_falls_back_to_walking(self) -> None:
        """A broke tank keeps walking — one more 45 beats stranding."""
        _stamp_mine_hit(99000)
        ctx = _ctx(fuel=3, terrain=InMemoryTerrainMap())

        command = walk_or_teleport(ctx, 105, 100, pickup_kind="fuel")

        if command is None:
            raise AssertionError("expected a command")
        assert command["cmd_type"] != "teleport"

    def test_no_terrain_view_falls_back_to_walking(self) -> None:
        """Without the static map no landing can be planned — walk."""
        _stamp_mine_hit(99000)
        ctx = _ctx(terrain=None)

        command = walk_or_teleport(ctx, 105, 100, pickup_kind="fuel")

        if command is None:
            raise AssertionError("expected a command")
        assert command["cmd_type"] != "teleport"

    def test_no_landing_near_target_falls_back_to_walking(self) -> None:
        """A destination with no passable landing keeps the walk."""
        _stamp_mine_hit(99000)
        water = {(x, y): InMemoryTerrainMap.WATER for x in range(103, 108) for y in range(98, 103)}
        terrain = InMemoryTerrainMap(water)
        ctx = _ctx(terrain=terrain)

        command = walk_or_teleport(ctx, 105, 100, pickup_kind="fuel")

        assert command is None or command["cmd_type"] != "teleport"
