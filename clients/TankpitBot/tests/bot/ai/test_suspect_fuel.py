"""Suspect fuel ranks below eligible alternatives without starving recovery."""

from __future__ import annotations

import pytest

from tankpit_bot.bot.ai.collect_hops import desperation_fuel_hop
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.equipment import is_fuel_lock_release_warranted
from tankpit_bot.bot.ai.equipment_search import find_best_fuel, find_fuel_candidates
from tankpit_bot.bot.ai.larder import select_fuel_larder_hop
from tankpit_bot.bot.combat_feedback import CombatFeedback
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import ContainerStateDict
from tests.bot.ai._support import (
    make_container,
    make_inventory,
    make_scanned_ai_state,
    make_world,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap


def test_walk_and_lock_prefer_confirmed_fuel() -> None:
    """A rich suspect tile cannot outscore or retain a lock over known fuel."""
    suspect = make_container(101, 100, 1000, is_fuel=True)
    suspect["confidence"] = 0.5
    confirmed = make_container(104, 100, 200, is_fuel=True)
    world, own = make_world(
        fuel=500,
        containers={"101,100": suspect, "104,100": confirmed},
    )
    assert find_best_fuel(world, own) == confirmed
    assert find_fuel_candidates(world, own) == [confirmed, suspect]
    assert is_fuel_lock_release_warranted(own, suspect, confirmed, 600) is True
    assert is_fuel_lock_release_warranted(own, confirmed, suspect, 600) is False
    del world["containers"]["104,100"]
    assert find_best_fuel(world, own) == suspect
    assert find_fuel_candidates(world, own) == [suspect]


def _context(containers: dict[str, ContainerStateDict]) -> DecideCtx:
    """Construct an ordinary recovery context with real decision services."""
    world, own = make_world(fuel=500, containers=containers)
    return DecideCtx(
        world,
        own,
        make_scanned_ai_state(),
        make_inventory(),
        100000,
        InMemoryTerrainMap(),
        CombatFeedback.NONE,
        ws=WorldService(),
    )


def test_larder_and_desperation_prefer_confirmed_then_allow_only_suspect() -> None:
    """Confidence precedes value/cost, with all affordability gates intact."""
    suspect = make_container(110, 100, 1000, is_fuel=True)
    suspect["confidence"] = 0.5
    confirmed = make_container(120, 100, 350, is_fuel=True)
    containers = {"110,100": suspect, "120,100": confirmed}
    ctx = _context(containers)
    assert select_fuel_larder_hop(ctx)["container"] == confirmed
    hop = desperation_fuel_hop(ctx, ctx.ai_state)
    if hop is None:
        pytest.fail("Confirmed affordable fuel must produce a recovery hop")
    assert hop["command"]["cmd_type"] == "teleport"
    assert (hop["command"]["target_x"], hop["command"]["target_y"]) == (120, 100)
    only_suspect = _context({"110,100": suspect})
    assert select_fuel_larder_hop(only_suspect)["container"] == suspect
    hop = desperation_fuel_hop(only_suspect, only_suspect.ai_state)
    if hop is None:
        pytest.fail("Sole suspect fuel must remain available for recovery")
    assert hop["command"]["cmd_type"] == "teleport"
    assert (hop["command"]["target_x"], hop["command"]["target_y"]) == (110, 100)
    unaffordable = make_container(250, 250, 500, is_fuel=True)
    ctx = _context({"110,100": suspect, "250,250": unaffordable})
    assert select_fuel_larder_hop(ctx)["container"] == suspect
