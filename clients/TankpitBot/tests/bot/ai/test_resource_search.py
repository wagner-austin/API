"""Tests for shared resource-search helpers."""

from __future__ import annotations

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.resource_search import select_fuel_dot_walk_targets
from tankpit_bot.bot.ai.types import AIStateDict
from tests.bot.ai._support import make_inventory, make_scanned_ai_state, make_world


def test_select_fuel_dot_walk_targets_skips_too_close_dots() -> None:
    """Dots within the minimum hop displacement are skipped."""
    world, self_state = make_world(self_x=100, self_y=100, fuel=300)
    # Dot at distance 3, below _MIN_HOP_DISPLACEMENT=4
    world["map_fuel_dots"] = {"103,100": 1}
    ai_state = make_scanned_ai_state()
    ctx = DecideCtx(world, self_state, ai_state, make_inventory(), 100000, None, "")

    result = select_fuel_dot_walk_targets(ctx)

    assert result == []


def test_select_fuel_dot_walk_targets_skips_recently_attempted_dots() -> None:
    """Dots attempted within the scan coverage TTL are skipped."""
    world, self_state = make_world(self_x=100, self_y=100, fuel=300)
    # Dot at distance 8 (passes displacement check)
    world["map_fuel_dots"] = {"108,100": 1}
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "attempted_fuel_dots": {"108,100": 90000},
        }
    )
    ctx = DecideCtx(world, self_state, ai_state, make_inventory(), 100000, None, "")

    result = select_fuel_dot_walk_targets(ctx)

    assert result == []


def test_select_fuel_dot_walk_targets_includes_eligible_dots() -> None:
    """Dots that pass displacement and attempt filters are included."""
    world, self_state = make_world(self_x=100, self_y=100, fuel=300)
    world["map_fuel_dots"] = {
        "102,100": 1,  # distance 2 < 4, too close
        "108,100": 1,  # distance 8 >= 4, eligible
        "104,100": 1,  # distance 4 >= 4, eligible
    }
    ai_state = make_scanned_ai_state()
    ctx = DecideCtx(world, self_state, ai_state, make_inventory(), 100000, None, "")

    result = select_fuel_dot_walk_targets(ctx)

    assert (104, 100) in result
    assert (108, 100) in result
    assert (102, 100) not in result
